from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .Funcs.channel_drop import aug_channel_drop
from .Funcs.jitter import aug_jitter
from .Funcs.quantize import aug_quantize
from .Funcs.resample import aug_resample
from .Funcs.rotate_triads import aug_rotate_triads
from .Funcs.scale import aug_scale
from .Funcs.time_mask import aug_time_mask
from .Funcs.time_shift import aug_time_shift

Array = np.ndarray


@dataclass
class AugConfig:
    # per-op probabilities (chance to be considered when sampling ops)
    probs: Dict[str, float]
    max_ops: int = 3
    min_ops: int = 1  # set to 0 if you want "no augmentation" possible

    # params
    jitter_std: float = 0.02
    scale_jitter: float = 0.2
    time_mask_ratio: float = 0.05
    channel_drop_prob: float = 0.05
    max_time_shift: int = 10

    # optional extras
    rotate_triads: Optional[List[Tuple[int, int, int]]] = None
    max_rotate_deg: float = 20.0
    resample_min_factor: float = 0.8
    resample_max_factor: float = 1.2
    quant_step: float = 0.01
    scale_per_channel: bool = False


class InertialAugmentRandomCombo:
    """
    Input: x [B, 1, T, C] (numpy array)
    For each sample i, randomly chooses k ops (k in [min_ops, max_ops]) without replacement,
    biased by probs, then applies them sequentially.
    """

    def __init__(self, cfg: AugConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()

        # registry: name -> callable(batch_subset -> batch_subset)
        self.ops: Dict[str, Callable[[Array], Array]] = {
            "time_shift": lambda s: aug_time_shift(s, cfg.max_time_shift, rng=self.rng),
            "scale": lambda s: aug_scale(s, cfg.scale_jitter, per_channel=cfg.scale_per_channel, rng=self.rng),
            "channel_drop": lambda s: aug_channel_drop(s, cfg.channel_drop_prob, rng=self.rng),
            "time_mask": lambda s: aug_time_mask(s, cfg.time_mask_ratio, rng=self.rng),
            "jitter": lambda s: aug_jitter(s, cfg.jitter_std, rng=self.rng),
            "rotate": lambda s: aug_rotate_triads(s, cfg.rotate_triads or [], cfg.max_rotate_deg, rng=self.rng),
            "resample": lambda s: aug_resample(s, cfg.resample_min_factor, cfg.resample_max_factor, rng=self.rng),
            "quantize": lambda s: aug_quantize(s, cfg.quant_step),
        }

        self.enabled = [k for k, p in cfg.probs.items() if p > 0 and k in self.ops]
        if not self.enabled:
            raise ValueError("No augmentations enabled. Set cfg.probs with >0 for at least one op.")

    def _sample_ops(self) -> List[str]:
        cfg = self.cfg
        kmax = max(0, int(cfg.max_ops))
        kmin = max(0, int(cfg.min_ops))
        kmax = min(kmax, len(self.enabled))
        kmin = min(kmin, kmax)
        if kmax == 0:
            return []

        k = int(self.rng.integers(kmin, kmax + 1))
        if k == 0:
            return []

        names = self.enabled
        weights = np.asarray([self.cfg.probs[n] for n in names], dtype=np.float64)
        weights = weights / (weights.sum() + 1e-12)

        chosen = self.rng.choice(names, size=k, replace=False, p=weights)
        return [str(x) for x in chosen.tolist()]

    def __call__(self, x: Array) -> Array:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [B,1,T,C], got {tuple(x.shape)}")
        B, one, _, _ = x.shape
        if one != 1:
            pass

        out = np.array(x, copy=True)
        plans = [self._sample_ops() for _ in range(B)]
        max_len = max((len(p) for p in plans), default=0)
        if max_len == 0:
            return out

        for step in range(max_len):
            buckets: Dict[str, List[int]] = {}
            for idx, ops in enumerate(plans):
                if step < len(ops):
                    buckets.setdefault(ops[step], []).append(idx)

            for name, indices in buckets.items():
                subset = out[indices]
                updated = self.ops[name](subset)
                out[indices] = updated

        return out


if __name__ == "__main__":
    cfg = AugConfig(
        probs={
            "time_shift": 1.0,
            "scale": 1.0,
            "channel_drop": 0.5,
            "time_mask": 0.5,
            "jitter": 1.0,
            "rotate": 0.7,
            "resample": 0.5,
            "quantize": 0.2,
        },
        max_ops=3,
        min_ops=1,
        rotate_triads=[(0, 1, 2), (3, 4, 5), (6, 7, 8)],
        max_rotate_deg=15.0,
        scale_per_channel=False,
    )

    aug = InertialAugmentRandomCombo(cfg)
    x = np.random.default_rng(0).standard_normal((32, 1, 128, 9), dtype=np.float32)
    y = aug(x)
    print(y.shape)
