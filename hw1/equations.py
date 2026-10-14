"""Closed-form FP32 inference costs for the CNN in models.py.

    F = B * (17714 * S**2 + 313700)                       FLOPs
    D = 4161296 + B * (380 * S**2 + 8592)                logical bytes moved
    M = 13729792 + 68 * B * S**2                         baseline allocated bytes
    T = 17 * launch_seconds (theta_1) + sum_i max(F_i / theta_2, D_i / theta_3) seconds
    E = fixed_joules + joules_per_flop * F + joules_per_byte * D

S is a multiple of 16 for exact tensor shapes. Positive real arguments are
also accepted as a continuous extension for plotting. All functions broadcast.
D counts one input/weight read and one output write per operation, separate
in-place ReLU reads/writes, and int64 pool-index writes. It is not measured DRAM
traffic. M assumes T4/PyTorch 2.11 default cuBLAS workspaces after warmup and
omits additional convolution workspace. See FORMULAS.md for assumptions.
"""

import numpy as np

LAYER_NAMES = (
    "conv7", "relu1", "maxpool", "conv5", "relu2", "conv3_64_128", "relu3",
    "conv1_128_256", "relu4", "conv3_256_256", "relu5", "conv1_256_512",
    "relu6", "global_avg_pool", "linear512_256", "relu_head", "linear256_100",
)
# Per-stage coefficients: cost_i = q_i * (B*S*S) + b_i * B + constant_i.
F_Q = np.array([2352, 0, 0, 6400, 0, 2304, 0, 1024, 0, 4608, 0, 1024, 0, 2, 0, 0, 0])
F_B = np.array([0] * 14 + [262400, 0, 51300])
D_Q = np.array([44, 64, 56, 24, 32, 24, 16, 24, 32, 20, 8, 12, 16, 8, 0, 0, 0])
D_B = np.array([0] * 13 + [2048, 3072, 2048, 1424])
D_CONST = np.array([18816, 0, 0, 204800, 0, 294912, 0, 131072, 0,
                    2359296, 0, 524288, 0, 0, 525312, 0, 102800])


def _inputs(image_size, batch):
    s, b = np.broadcast_arrays(np.asarray(image_size, dtype=float),
                               np.asarray(batch, dtype=float))
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(b))
            and np.all(s > 0) and np.all(b > 0)):
        raise ValueError("image_size and batch must be finite and positive")
    return s, b


def _number_or_array(value):
    return float(value) if np.ndim(value) == 0 else value


def _parameter(theta, name, *, positive=False):
    value = float(theta[name])
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"Invalid parameter {name}: {value}")
    return value


def layer_costs(image_size, batch):
    """Return FLOPs and logical bytes with a trailing axis of 17 operations."""
    s, b = _inputs(image_size, batch)
    q = (b * s**2)[..., None]
    b = b[..., None]
    return q * F_Q + b * F_B, q * D_Q + b * D_B + D_CONST


def flops(image_size, batch):
    """Count MAC=2, linear bias additions, and global-average arithmetic.

    ReLU/max-pool comparisons are excluded; padded convolution positions use
    the full kernel cost. No backward pass or optimizer is counted.
    """
    s, b = _inputs(image_size, batch)
    return _number_or_array(b * (17714 * s**2 + 313700))


def bytes_moved(image_size, batch):
    """Logical operator traffic; no PCIe copies or cuDNN workspace traffic."""
    s, b = _inputs(image_size, batch)
    return _number_or_array(4161296 + b * (380 * s**2 + 8592))


def memory(image_size, batch):
    """Analytic baseline, not an exact peak or a universal OOM classifier."""
    s, b = _inputs(image_size, batch)
    return _number_or_array(13729792 + 68 * b * s**2)


def latency(image_size, batch, theta):
    """Layer-wise roofline with fitted effective compute/bandwidth rates."""
    tau = _parameter(theta, "launch_seconds")
    compute = _parameter(theta, "compute_flops_per_second", positive=True)
    bandwidth = _parameter(theta, "bandwidth_bytes_per_second", positive=True)
    f, d = layer_costs(image_size, batch)
    return _number_or_array(len(LAYER_NAMES) * tau + np.maximum(f / compute, d / bandwidth).sum(axis=-1))


def energy(image_size, batch, theta_energy):
    """Whole-GPU joules per batched forward; effective empirical coefficients."""
    fixed = _parameter(theta_energy, "fixed_joules")
    per_flop = _parameter(theta_energy, "joules_per_flop")
    per_byte = _parameter(theta_energy, "joules_per_byte")
    return _number_or_array(fixed + per_flop * flops(image_size, batch)
                            + per_byte * bytes_moved(image_size, batch))
