from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray


def _random_rotation_matrices(
    batch: int, *, max_deg: float, rng: np.random.Generator, dtype: np.dtype
) -> Array:
    axis = rng.normal(size=(batch, 3)).astype(np.float64, copy=False)
    axis_norm = np.linalg.norm(axis, axis=1, keepdims=True) + 1e-8
    axis = axis / axis_norm

    angles = (rng.random(batch) * 2.0 - 1.0) * (float(max_deg) * math.pi / 180.0)
    cos = np.cos(angles)
    sin = np.sin(angles)
    C = 1.0 - cos
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]

    R = np.zeros((batch, 3, 3), dtype=np.float64)
    R[:, 0, 0] = cos + x * x * C
    R[:, 0, 1] = x * y * C - z * sin
    R[:, 0, 2] = x * z * C + y * sin

    R[:, 1, 0] = y * x * C + z * sin
    R[:, 1, 1] = cos + y * y * C
    R[:, 1, 2] = y * z * C - x * sin

    R[:, 2, 0] = z * x * C - y * sin
    R[:, 2, 1] = z * y * C + x * sin
    R[:, 2, 2] = cos + z * z * C
    return R.astype(dtype, copy=False)


def aug_rotate_triads(
    x: Array,
    triads: Optional[Sequence[Tuple[int, int, int]]],
    max_rotate_deg: float,
    *,
    rng: np.random.Generator | None = None,
) -> Array:
    if not triads or max_rotate_deg <= 0:
        return x
    rng = rng or np.random.default_rng()
    batch = int(x.shape[0])
    rotations = _random_rotation_matrices(batch, max_deg=max_rotate_deg, rng=rng, dtype=x.dtype)
    Rt = np.swapaxes(rotations, 1, 2)
    out = np.array(x, copy=True)
    for (a, b, c) in triads:
        triad = out[:, :, :, [a, b, c]].reshape(batch, -1, 3)
        rotated = triad @ Rt
        out[:, :, :, [a, b, c]] = rotated.reshape(batch, 1, out.shape[2], 3)
    return out
