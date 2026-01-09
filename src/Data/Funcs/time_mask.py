from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    Array = np.ndarray
else:
    Array = np.ndarray


def aug_time_mask(
    x: Array, time_mask_ratio: float, *, rng: np.random.Generator | None = None
) -> Array:
    if time_mask_ratio <= 0 or x.shape[-2] <= 1:
        return x
    length = x.shape[-2]
    mask_width = max(1, int(length * time_mask_ratio))
    if mask_width >= length:
        return np.zeros_like(x)
    batch = x.shape[0]
    rng = rng or np.random.default_rng()
    starts = rng.integers(0, int(length - mask_width) + 1, size=int(batch))
    out = np.array(x, copy=True)
    for i, start in enumerate(starts.tolist()):
        out[i, :, int(start) : int(start) + int(mask_width), :] = 0.0
    return out
