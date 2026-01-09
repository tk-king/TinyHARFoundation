from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    Array = np.ndarray
else:
    Array = np.ndarray


def aug_channel_drop(
    x: Array, channel_drop_prob: float, *, rng: np.random.Generator | None = None
) -> Array:
    if channel_drop_prob <= 0:
        return x
    batch, _, _, channels = x.shape
    rng = rng or np.random.default_rng()
    drop_mask = (rng.random((batch, 1, 1, channels)) < float(channel_drop_prob)).astype(
        x.dtype, copy=False
    )
    return x * (1.0 - drop_mask)
