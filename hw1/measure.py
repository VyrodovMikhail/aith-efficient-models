"""Run section 4's grid on CUDA, recording successful forwards and OOMs."""

import argparse
from contextlib import ExitStack
import csv
import gc
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import torch
from torch import nn

from grid import Configuration, make_grid
from energy import (EnergyUnavailable, NvmlEnergyCounter, measure_energy,
                    validate_energy_seconds)
from models import build_model

FIELDS = ("S", "B", "status", "latency_seconds", "memory_bytes", "energy_joules",
          "energy_status", "energy_error", "energy_repeats", "energy_duration_seconds",
          "is_validation")


def configure_cuda() -> None:
    """Use the FP32 settings required by section 7 for the grid runner."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


@torch.inference_mode()
def measure_configuration(
    model: nn.Module, config: Configuration, *, warmup: int = 5, repeats: int = 20,
    energy_counter: NvmlEnergyCounter | None = None, energy_seconds: float = 5.0,
    energy_unavailable_reason: str = "",
) -> dict:
    """Measure one CUDA point; catch OOM during input creation or any forward.

    Peak allocated bytes include resident model and input. Timings exclude random
    input generation and include synchronization of each complete forward pass.
    """
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    validate_energy_seconds(energy_seconds)
    model.eval()
    row = dict.fromkeys(FIELDS, "")
    row.update(S=config.image_size, B=config.batch, status="OK",
               is_validation=config.is_validation,
               energy_status="UNAVAILABLE" if energy_unavailable_reason else "DISABLED",
               energy_error=energy_unavailable_reason)
    inputs = output = None
    try:
        inputs = torch.randn(config.batch, 3, config.image_size, config.image_size,
                             device="cuda", dtype=torch.float32)
        for _ in range(warmup):
            output = model(inputs)
            del output
        torch.cuda.synchronize()
        times, peaks = [], []
        for _ in range(repeats):
            torch.cuda.reset_peak_memory_stats()
            start = perf_counter()
            output = model(inputs)
            torch.cuda.synchronize()
            times.append(perf_counter() - start)
            peaks.append(torch.cuda.max_memory_allocated())
            del output
        row.update(latency_seconds=median(times), memory_bytes=max(peaks))
        if energy_counter is not None:
            try:
                energy = measure_energy(model, inputs, energy_counter,
                                        min_seconds=energy_seconds)
                row.update(energy_joules=energy.joules_per_forward, energy_status="OK",
                           energy_error="", energy_repeats=energy.repeats,
                           energy_duration_seconds=energy.duration_seconds)
            except EnergyUnavailable as exc:
                row.update(energy_status="ERROR", energy_error=str(exc))
    except torch.cuda.OutOfMemoryError:
        row.update(status="OOM", latency_seconds="", memory_bytes="", energy_joules="",
                   energy_status="OOM", energy_error="", energy_repeats="",
                   energy_duration_seconds="")
    finally:
        # Drop references before clearing cached blocks, including after an OOM.
        inputs = output = None
        gc.collect()
        torch.cuda.empty_cache()
    return row


def run_grid(model: nn.Module, grid: list[Configuration], destination: Path,
             *, warmup: int = 5, repeats: int = 20,
             energy_counter: NvmlEnergyCounter | None = None,
             energy_seconds: float = 5.0, energy_unavailable_reason: str = "") -> None:
    """Flush each result immediately so a long run preserves completed points."""
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    validate_energy_seconds(energy_seconds)
    model.eval()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for index, config in enumerate(grid, start=1):
            row = measure_configuration(
                model, config, warmup=warmup, repeats=repeats,
                energy_counter=energy_counter, energy_seconds=energy_seconds,
                energy_unavailable_reason=energy_unavailable_reason,
            )
            writer.writerow(row)
            stream.flush()
            print(f"[{index}/{len(grid)}] S={config.image_size} B={config.batch}: "
                  f"{row['status']}, energy={row['energy_status']}"
                  + (f" ({row['energy_joules']:.6g} J/forward)"
                     if row["energy_status"] == "OK" else ""), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/measurements.csv"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--energy-seconds", type=float, default=5.0,
                        help="Minimum duration of each energy series (default: 5 seconds)")
    parser.add_argument("--no-energy", action="store_true", help="Only measure latency and memory")
    parser.add_argument("--dry-run", action="store_true", help="Print grid without tensors or CUDA")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative and repeats must be positive")
    try:
        validate_energy_seconds(args.energy_seconds)
    except ValueError as exc:
        parser.error(str(exc))
    grid = make_grid(args.seed)
    if args.dry_run:
        print("S,B,is_validation")
        for config in grid:
            print(f"{config.image_size},{config.batch},{config.is_validation}")
        return
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required for measurements; use --dry-run to inspect the grid")
    configure_cuda()
    torch.manual_seed(args.seed)
    model = build_model().cuda().eval()
    with ExitStack() as stack:
        counter, reason = None, ""
        if not args.no_energy:
            try:
                counter = stack.enter_context(NvmlEnergyCounter(torch.cuda.current_device()))
            except EnergyUnavailable as exc:
                reason = str(exc)
                print(f"Warning: {reason}. Energy will be blank in the CSV.", file=sys.stderr)
        run_grid(model, grid, args.output, warmup=args.warmup, repeats=args.repeats,
                 energy_counter=counter, energy_seconds=args.energy_seconds,
                 energy_unavailable_reason=reason)


if __name__ == "__main__":
    main()
