from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    Array = np.ndarray
else:
    Array = np.ndarray


def aug_time_shift(x: Array, max_time_shift: int, *, rng: np.random.Generator | None = None) -> Array:
    # x: [B, 1, T, C]
    if max_time_shift <= 0 or x.shape[-2] <= 1:
        return x
    rng = rng or np.random.default_rng()
    batch = int(x.shape[0])
    shifts = rng.integers(-int(max_time_shift), int(max_time_shift) + 1, size=batch)
    if np.all(shifts == 0):
        return x

    out = np.array(x, copy=True)
    for i, s in enumerate(shifts.tolist()):
        if s != 0:
            # `out[i]` is [1, T, C], so shift along time axis (T).
            out[i] = np.roll(out[i], shift=int(s), axis=-2)
    return out
