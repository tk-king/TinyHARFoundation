from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    Array = np.ndarray
else:
    Array = np.ndarray


def aug_jitter(x: Array, jitter_std: float, *, rng: np.random.Generator | None = None) -> Array:
    if jitter_std <= 0:
        return x
    rng = rng or np.random.default_rng()
    noise = rng.normal(loc=0.0, scale=float(jitter_std), size=x.shape).astype(x.dtype, copy=False)
    return x + noise
