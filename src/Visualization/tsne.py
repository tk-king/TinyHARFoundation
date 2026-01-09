from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TSNEResult:
    xy: np.ndarray  # (N, 2)


def tsne(
    embeddings: np.ndarray,
    *,
    perplexity: float = 30.0,
    n_iter: int = 1_000,
    learning_rate: str | float = "auto",
    init: str = "pca",
    random_state: int = 0,
    verbose: int = 1,
) -> TSNEResult:
    """
    Compute a 2D t-SNE projection for embedding vectors.

    Args:
        embeddings: (N, D) float array.
        perplexity: Typical values 5-50. Must be < N.
        n_iter: Optimization iterations.
        learning_rate: "auto" or float.
        init: "pca" or "random".
        random_state: Seed for deterministic output.
        verbose: scikit-learn verbosity (0/1/2).

    Returns:
        TSNEResult with `xy` shape (N, 2).
    """
    x = np.asarray(embeddings)
    if x.ndim != 2:
        raise ValueError(f"embeddings must have shape (N, D), got {x.shape}.")
    if x.shape[0] < 2:
        raise ValueError(f"Need at least 2 samples for t-SNE, got N={x.shape[0]}.")

    try:
        from sklearn.manifold import TSNE  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "t-SNE requires scikit-learn. Install it (e.g. `pip install scikit-learn`) "
            "and retry."
        ) from e

    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init=init,
        random_state=random_state,
        verbose=verbose,
    )
    xy = model.fit_transform(x).astype(np.float32, copy=False)
    return TSNEResult(xy=xy)


def plot_tsne(
    xy: np.ndarray,
    labels: Iterable[int] | None = None,
    *,
    title: str | None = None,
    label_names: dict[int, str] | None = None,
    s: float = 8.0,
    alpha: float = 0.7,
    savepath: str | None = None,
):
    """
    Scatter-plot a t-SNE projection.

    Args:
        xy: (N, 2) array from `tsne(...).xy`.
        labels: Optional class labels length N.
        title: Optional plot title.
        label_names: Optional mapping label_id -> readable name.
        s: Marker size.
        alpha: Marker alpha.
        savepath: If set, saves figure to this path.

    Returns:
        (fig, ax)
    """
    xy = np.asarray(xy)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must have shape (N, 2), got {xy.shape}.")

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "plot_tsne requires matplotlib. Install it (e.g. `pip install matplotlib`) and retry."
        ) from e

    fig, ax = plt.subplots(figsize=(7, 6))

    if labels is None:
        ax.scatter(xy[:, 0], xy[:, 1], s=s, alpha=alpha)
    else:
        y = np.asarray(list(labels))
        if y.shape[0] != xy.shape[0]:
            raise ValueError(f"labels length must match xy rows ({xy.shape[0]}), got {y.shape[0]}.")
        for lab in np.unique(y):
            idx = y == lab
            name = label_names.get(int(lab), str(int(lab))) if label_names else str(int(lab))
            ax.scatter(xy[idx, 0], xy[idx, 1], s=s, alpha=alpha, label=name)
        ax.legend(markerscale=2, fontsize=9, frameon=False, loc="best")

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    if title:
        ax.set_title(title)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)
    return fig, ax

