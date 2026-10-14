"""Fit latency/energy on training rows only; report held-out predictions.

Run: python calibrate.py
Legacy sources: --measurements measurements.csv --energy-log joules.txt
Use --energy-log - to read energy_joules directly from the measurement CSV.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import least_squares, nnls

from equations import bytes_moved, energy, flops, latency, layer_costs, memory, LAYER_NAMES

ENERGY_LINE = re.compile(
    r"\[\d+/\d+\]\s+S=(\d+)\s+B=(\d+):\s+(\w+),\s+energy=(\w+)"
    r"(?:\s+\(([-+\d.eE]+) J/forward\))?\s*"
)


def positive(value, label):
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive: {value!r}")
    return result


def configuration(s, b):
    s, b = int(s), int(b)
    if s < 16 or s % 16 or b < 1:
        raise ValueError(f"Invalid configuration S={s}, B={b}")
    return s, b


def read_energy_log(path):
    """Parse console output, retaining errors/OOM and rejecting duplicate pairs."""
    entries = {}
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        match = ENERGY_LINE.fullmatch(line.strip())
        if match is None:
            raise ValueError(f"{path}:{line_number}: invalid energy log line")
        s, b, status, energy_status, value = match.groups()
        key = configuration(s, b)
        if key in entries:
            raise ValueError(f"Duplicate energy configuration {key} at line {line_number}")
        if status == "OK" and energy_status == "OK":
            if value is None:
                raise ValueError(f"Missing energy for {key} at line {line_number}")
            value = positive(value, f"energy for {key}")
        else:
            if value is not None:
                raise ValueError(f"Failed energy row {key} unexpectedly contains a value")
            value = None
        entries[key] = {"energy_status": energy_status, "energy_joules": value}
    return entries


def load_measurements(path, energy_log=None):
    """Join by (S,B), preserving the CSV's explicit calibration/validation split."""
    rows, seen = [], set()
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"S", "B", "status", "latency_seconds", "is_validation"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"CSV must contain {sorted(required)}")
        for raw in reader:
            key = configuration(raw["S"], raw["B"])
            if key in seen:
                raise ValueError(f"Duplicate measurement configuration {key}")
            seen.add(key)
            flag = raw["is_validation"].strip().lower()
            if flag not in {"true", "false", "1", "0"}:
                raise ValueError(f"Invalid is_validation for {key}: {flag!r}")
            ok = raw["status"] == "OK"
            row = {"S": key[0], "B": key[1], "status": raw["status"],
                   "is_validation": flag in {"true", "1"},
                   "latency_seconds": positive(raw["latency_seconds"], "latency") if ok else None,
                   "memory_bytes": positive(raw["memory_bytes"], "memory")
                       if ok and raw.get("memory_bytes") else None,
                   "energy_joules": None, "energy_status": raw.get(
                       "energy_status", "OOM" if raw["status"] == "OOM" else "MISSING")}
            if energy_log is None and ok and raw.get("energy_joules"):
                if raw.get("energy_status", "OK") == "OK":
                    row["energy_joules"] = positive(raw["energy_joules"], "energy")
                    row["energy_status"] = "OK"
            rows.append(row)
    if energy_log is not None:
        entries = read_energy_log(energy_log)
        extra = set(entries) - seen
        if extra:
            raise ValueError(f"Energy log contains configurations absent from CSV: {sorted(extra)}")
        for row in rows:
            row.update(entries.get((row["S"], row["B"]),
                                   {"energy_status": "MISSING", "energy_joules": None}))
    return rows


def _fit_arrays(image_size, batch, observed):
    s, b, y = np.broadcast_arrays(np.asarray(image_size, dtype=float),
                                  np.asarray(batch, dtype=float),
                                  np.asarray(observed, dtype=float))
    if y.size < 4 or not np.all(np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("At least four finite, positive training observations are required")
    return s.ravel(), b.ravel(), y.ravel()


def matrix_diagnostics(matrix):
    norms = np.linalg.norm(matrix, axis=0)
    scaled = matrix / np.where(norms > 0, norms, 1)
    singular = np.linalg.svd(scaled, compute_uv=False)
    rank = int(np.linalg.matrix_rank(scaled))
    condition = float(singular[0] / singular[-1]) if rank == matrix.shape[1] else None
    return {"rank": rank, "columns": matrix.shape[1],
            "column_normalized_condition_number": condition}


def fit_latency(image_size, batch, observed):
    """Positive multi-start fit minimizing squared relative errors in seconds."""
    s, b, y = _fit_arrays(image_size, batch, observed)
    f, d = layer_costs(s, b)

    def residual(log_parameters):
        tau, compute, bandwidth = 10.0 ** log_parameters
        predicted = len(LAYER_NAMES) * tau + np.maximum(f / compute, d / bandwidth).sum(axis=-1)
        return (predicted - y) / y

    lower, upper = np.array([-12., 6., 5.]), np.array([0., 16., 15.])
    tau_start = np.clip(y.min() / (2 * len(LAYER_NAMES)), 1e-10, 0.1)
    candidates = []
    for compute in (1e11, 1e12, 1e13):
        for bandwidth in (1e9, 1e10, 1e11):
            fit = least_squares(residual, np.log10([tau_start, compute, bandwidth]),
                                bounds=(lower, upper), max_nfev=3000,
                                ftol=1e-11, xtol=1e-11, gtol=1e-11)
            if fit.success and np.all(np.isfinite(fit.x)):
                candidates.append(fit)
    if not candidates:
        raise RuntimeError("Latency optimizer did not converge from any initial point")
    best = min(candidates, key=lambda result: result.cost)
    names = ("launch_seconds", "compute_flops_per_second", "bandwidth_bytes_per_second")
    theta = dict(zip(names, map(float, 10.0 ** best.x)))
    diagnostics = matrix_diagnostics(best.jac)
    diagnostics.update(relative_rmse=float(np.sqrt(np.mean(best.fun**2))),
                       successful_starts=len(candidates), optimizer_message=best.message,
                       parameters_at_bounds=[names[i] for i in range(3)
                           if min(best.x[i] - lower[i], upper[i] - best.x[i]) < 1e-4])
    return theta, diagnostics


def fit_energy(image_size, batch, observed):
    """Nonnegative relative least squares; scale columns before solving NNLS."""
    s, b, y = _fit_arrays(image_size, batch, observed)
    design = np.column_stack([np.ones_like(y), flops(s, b) / 1e9, bytes_moved(s, b) / 1e6])
    weighted = design / y[:, None]
    norms = np.linalg.norm(weighted, axis=0)
    coefficients, _ = nnls(weighted / norms, np.ones_like(y), maxiter=1000)
    coefficients /= norms
    theta = {"fixed_joules": float(coefficients[0]),
             "joules_per_flop": float(coefficients[1] / 1e9),
             "joules_per_byte": float(coefficients[2] / 1e6)}
    diagnostics = matrix_diagnostics(weighted)
    diagnostics.update(relative_rmse=float(np.sqrt(np.mean(((design @ coefficients - y) / y)**2))),
                       flops_bytes_correlation=float(np.corrcoef(design[:, 1], design[:, 2])[0, 1]),
                       zero_parameters=[name for name, value in theta.items() if value == 0])
    return theta, diagnostics


def error_metrics(observed, predicted):
    y, prediction = np.asarray(observed), np.asarray(predicted)
    if y.size == 0:
        return {"count": 0}
    relative = np.abs(prediction - y) / y
    return {"count": int(y.size), "mae": float(np.mean(np.abs(prediction - y))),
            "rmse": float(np.sqrt(np.mean((prediction - y)**2))),
            "mape_percent": float(100 * np.mean(relative)),
            "median_ape_percent": float(100 * np.median(relative)),
            "max_ape_percent": float(100 * np.max(relative))}


def calibrate(rows):
    """Fit exclusively on !is_validation; holdout values only enter metrics."""
    result = {"latency": {}, "energy": {}, "metrics": {}, "diagnostics": {}, "warnings": []}
    for name, field, fitter, predictor in (
        ("latency", "latency_seconds", fit_latency, latency),
        ("energy", "energy_joules", fit_energy, energy),
    ):
        valid = [r for r in rows if r["status"] == "OK" and r[field] is not None]
        training = [r for r in valid if not r["is_validation"]]
        theta, diagnostics = fitter([r["S"] for r in training], [r["B"] for r in training],
                                    [r[field] for r in training])
        result[name], result["diagnostics"][name] = theta, diagnostics
        result["metrics"][name] = {}
        for split, flag in (("train", False), ("validation", True)):
            selected = [r for r in valid if r["is_validation"] == flag]
            predicted = predictor([r["S"] for r in selected], [r["B"] for r in selected], theta)
            result["metrics"][name][split] = error_metrics([r[field] for r in selected], predicted)
        condition = diagnostics["column_normalized_condition_number"]
        if diagnostics["rank"] < 3 or condition is None or condition > 1000:
            result["warnings"].append(f"{name}: coefficients are poorly identified; inspect diagnostics")
    if abs(result["diagnostics"]["energy"]["flops_bytes_correlation"]) > 0.99:
        result["warnings"].append(
            "Energy FLOPs and bytes are strongly correlated. Coefficients are effective fit "
            "parameters, not independently measured hardware energy costs."
        )
    for name in result["diagnostics"]["latency"]["parameters_at_bounds"]:
        result["warnings"].append(f"Latency parameter {name} reached its fitting bound")
    if result["diagnostics"]["energy"]["zero_parameters"]:
        result["warnings"].append("Energy NNLS placed some coefficients at zero; see diagnostics")
    return result


def write_predictions(rows, result, path):
    fields = ("S", "B", "status", "is_validation", "flops", "bytes_moved", "memory_predicted_bytes",
              "memory_measured_bytes", "latency_measured_seconds", "latency_predicted_seconds",
              "energy_status", "energy_measured_joules", "energy_predicted_joules")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            s, b = row["S"], row["B"]
            writer.writerow(dict(zip(fields, (
                s, b, row["status"], row["is_validation"], flops(s, b), bytes_moved(s, b), memory(s, b),
                row["memory_bytes"], row["latency_seconds"], latency(s, b, result["latency"]),
                row["energy_status"], row["energy_joules"], energy(s, b, result["energy"]),
            ))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, default=Path("results/measurements.csv"))
    parser.add_argument("--energy-log", default="-", help="Console log, or - to use CSV energy (default)")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    log = None if args.energy_log == "-" else Path(args.energy_log)
    try:
        rows = load_measurements(args.measurements, log)
        result = calibrate(rows)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    result.update(schema_version=1, objective="mean squared relative error on training rows only",
                  formulas={"latency": "17*tau + sum_i max(F_i/P, D_i/W)",
                            "energy": "E0 + eF*F + eD*D",
                            "flops": "B*(17714*S^2+313700)",
                            "bytes_moved": "4161296+B*(380*S^2+8592)",
                            "memory_baseline": "13729792+68*B*S^2"},
                  counts={"rows": len(rows), "oom": sum(r["status"] == "OOM" for r in rows),
                          "missing_energy_on_success": sum(r["status"] == "OK" and
                              r["energy_joules"] is None for r in rows)},
                  sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (args.measurements, log) if p is not None})
    if log is not None:
        result["warnings"].append(
            "Energy values come from console output rounded to 6 significant digits. "
            "Latency and energy may be separate runs; no E/T power estimate is used."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "theta.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_predictions(rows, result, args.output_dir / "predictions.csv")
    for name in ("latency", "energy"):
        print(f"{name} theta: {result[name]}")
        for split, metrics in result["metrics"][name].items():
            if metrics["count"]:
                print(f"  {split}: n={metrics['count']}, MAPE={metrics['mape_percent']:.2f}%")
    for warning in result["warnings"]:
        print(f"Warning: {warning}")
    print(f"Saved {args.output_dir / 'theta.json'} and {args.output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
