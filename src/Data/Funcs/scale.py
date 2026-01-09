from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    Array = np.ndarray
else:
    Array = np.ndarray


def aug_scale(
    x: Array,
    scale_jitter: float,
    *,
    per_channel: bool = False,
    rng: np.random.Generator | None = None,
) -> Array:
    if scale_jitter <= 0:
        return x
    low, high = 1.0 - scale_jitter, 1.0 + scale_jitter
    batch, _, _, channels = x.shape
    if per_channel:
        shape = (batch, 1, 1, channels)
    else:
        shape = (batch, 1, 1, 1)
    rng = rng or np.random.default_rng()
    scale = rng.uniform(low=float(low), high=float(high), size=shape).astype(x.dtype, copy=False)
    return x * scale
