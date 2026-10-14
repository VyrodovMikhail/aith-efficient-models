"""Whole-GPU energy per forward from NVIDIA's cumulative millijoule counter."""

from contextlib import suppress
from dataclasses import dataclass
import math
from time import perf_counter

import torch
from torch import nn


class EnergyUnavailable(RuntimeError):
    """NVML or its energy counter cannot provide a usable measurement."""


class NvmlEnergyCounter:
    """Own an NVML session matched to the CUDA device by UUID, not ordinal.

    CUDA_VISIBLE_DEVICES can reorder CUDA ordinals relative to NVML indices.
    Never silently substitute a different GPU when UUID lookup fails.
    """

    def __init__(self, device=None):
        self.device = device
        self._nvml = None
        self._handle = None
        self._initialized = False

    def __enter__(self):
        try:
            import pynvml
        except ImportError as exc:
            raise EnergyUnavailable("Install nvidia-ml-py to measure GPU energy") from exc
        self._nvml = pynvml
        try:
            pynvml.nvmlInit()
            self._initialized = True
            uuid = getattr(torch.cuda.get_device_properties(self.device), "uuid", None)
            if uuid is None:
                raise EnergyUnavailable("PyTorch did not expose a GPU UUID for NVML lookup")
            uuid = str(uuid)
            if not uuid.startswith(("GPU-", "MIG-")):
                uuid = "GPU-" + uuid
            self._handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            self.read_millijoules()  # Detect unsupported counters before the grid.
        except pynvml.NVMLError as exc:
            self.close()
            raise EnergyUnavailable(f"NVML energy unavailable: {exc}") from exc
        except BaseException:
            self.close()
            raise
        return self

    def read_millijoules(self) -> int:
        if not self._initialized or self._handle is None:
            raise EnergyUnavailable("NVML energy counter is not open")
        try:
            return self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        except self._nvml.NVMLError as exc:
            raise EnergyUnavailable(f"NVML energy unavailable: {exc}") from exc

    def close(self) -> None:
        if self._initialized:
            with suppress(self._nvml.NVMLError):
                self._nvml.nvmlShutdown()
            self._initialized = False
            self._handle = None

    def __exit__(self, exc_type, exc, traceback):
        self.close()


@dataclass(frozen=True)
class EnergyMeasurement:
    joules_per_forward: float
    repeats: int
    duration_seconds: float


def validate_energy_seconds(seconds: float) -> None:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("energy_seconds must be finite and positive")


@torch.inference_mode()
def measure_energy(
    model: nn.Module, inputs: torch.Tensor, counter: NvmlEnergyCounter,
    *, min_seconds: float = 5.0,
) -> EnergyMeasurement:
    """Average a series of warmed-up, synchronized forwards over a long window.

    Model and input must already be on the GPU selected by the counter. Energy
    includes the whole GPU's baseline consumption and host launch/sync gaps;
    input creation, warmup, and garbage collection are outside the interval.
    The caller warms up the model before invoking this function.
    """
    validate_energy_seconds(min_seconds)
    model.eval()
    torch.cuda.synchronize(inputs.device)
    before = counter.read_millijoules()
    start = perf_counter()
    elapsed = 0.0
    repeats = 0
    while elapsed < min_seconds:
        output = model(inputs)
        torch.cuda.synchronize(inputs.device)
        del output
        repeats += 1
        elapsed = perf_counter() - start
    after = counter.read_millijoules()
    delta = after - before
    if delta <= 0:
        raise EnergyUnavailable(
            "Energy counter did not increase or was reset; no valid energy result"
        )
    return EnergyMeasurement(delta / 1000.0 / repeats, repeats, elapsed)
