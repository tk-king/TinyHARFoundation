from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    Array = np.ndarray
else:
    Array = np.ndarray


def _resample_single(
    sample: Array, min_factor: float, max_factor: float, *, rng: np.random.Generator
) -> Array:
    timesteps = int(sample.shape[-2])
    if timesteps <= 2:
        return sample

    factor = float(rng.uniform(float(min_factor), float(max_factor)))
    new_timesteps = max(2, int(timesteps * factor))

    # sample: [1, 1, T, C]
    xi = sample[0, 0]  # [T, C]
    channels = int(xi.shape[-1])

    t_orig = np.arange(timesteps, dtype=np.float64)
    t_new = np.linspace(0.0, float(timesteps - 1), num=new_timesteps, dtype=np.float64)

    tmp = np.empty((new_timesteps, channels), dtype=np.float64)
    for c in range(channels):
        tmp[:, c] = np.interp(t_new, t_orig, xi[:, c].astype(np.float64, copy=False))

    yi = np.empty((timesteps, channels), dtype=np.float64)
    for c in range(channels):
        yi[:, c] = np.interp(t_orig, t_new, tmp[:, c])

    return yi.astype(sample.dtype, copy=False)[None, None, :, :]


def aug_resample(
    x: Array,
    min_factor: float = 0.7,
    max_factor: float = 1.3,
    *,
    rng: np.random.Generator | None = None,
) -> Array:
    if x.shape[-2] <= 2:
        return x
    rng = rng or np.random.default_rng()
    samples = []
    for i in range(x.shape[0]):
        samples.append(_resample_single(x[i : i + 1], min_factor, max_factor, rng=rng))
    return np.concatenate(samples, axis=0)
