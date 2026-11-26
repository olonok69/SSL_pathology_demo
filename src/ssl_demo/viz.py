"""Plots for frozen features. UMAP is imported only when a projection is requested."""

from __future__ import annotations

import numpy as np

from ssl_demo.arrays import as_1d_labels
from ssl_demo.config import SEED


def plot_embedding(embedding, labels, class_names, ax=None, title: str | None = None):
    """Scatter a 2-D embedding coloured by tissue class. Returns the axes."""
    import matplotlib.pyplot as plt

    embedding = np.asarray(embedding)
    labels = as_1d_labels(labels)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    for class_id, name in enumerate(class_names):
        mask = labels == class_id
        if not np.any(mask):
            continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=8, alpha=0.75, label=name)
    ax.set_title(title or "")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=7, frameon=False)
    return ax


def umap_project(features, seed: int = SEED, n_neighbors: int = 15, min_dist: float = 0.1):
    """Cosine UMAP. Install ``umap-learn`` before calling this."""
    import umap

    features = np.asarray(features, dtype=np.float32)
    neighbours = min(int(n_neighbors), max(len(features) - 1, 2))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=neighbours,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(features)


def plot_umap_by_class(
    features,
    labels,
    class_names,
    seed: int = SEED,
    max_points: int = 2000,
    title: str | None = None,
    ax=None,
):
    """Project a stratified subset and plot it. Returns ``(figure, axes, embedding)``."""
    from ssl_demo.data import stratified_indices

    labels = as_1d_labels(labels)
    features = np.asarray(features)
    if len(features) > max_points:
        chosen = stratified_indices(labels, max_points, seed)
        features = features[chosen]
        labels = labels[chosen]
    embedding = umap_project(features, seed=seed)
    ax = plot_embedding(embedding, labels, class_names, ax=ax, title=title)
    return ax.figure, ax, embedding
