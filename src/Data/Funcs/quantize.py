from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    Array = np.ndarray
else:
    Array = np.ndarray


def aug_quantize(x: Array, step: float = 0.01) -> Array:
    if step <= 0:
        return x
    step = float(step)
    return (np.round(x / step) * step).astype(x.dtype, copy=False)
