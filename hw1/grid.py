"""Reproducible Cartesian measurement grid from section 4."""

from dataclasses import dataclass
from itertools import product
import random

BASE_IMAGE_SIZES = (32, 64, 128, 224, 256, 384, 512, 1024)
BASE_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


@dataclass(frozen=True)
class Configuration:
    image_size: int
    batch: int
    is_validation: bool


def make_grid(seed: int = 42) -> list[Configuration]:
    """Sample distinct additional axis values, then return all 144 pairs.

    Pairs containing a sampled axis value are held out for later validation.
    Sampling uses a local RNG, independent of model/input random numbers.
    """
    rng = random.Random(seed)
    extra_sizes = rng.sample(
        [s for s in range(32, 513, 16) if s not in BASE_IMAGE_SIZES], 4
    )
    extra_batches = rng.sample(
        [b for b in range(1, 257) if b not in BASE_BATCH_SIZES], 3
    )
    return [
        Configuration(s, b, s in extra_sizes or b in extra_batches)
        for s, b in product(sorted((*BASE_IMAGE_SIZES, *extra_sizes)),
                            sorted((*BASE_BATCH_SIZES, *extra_batches)))
    ]
