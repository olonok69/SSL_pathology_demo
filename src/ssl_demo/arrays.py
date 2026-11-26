"""Small array helpers with no torch dependency."""

from __future__ import annotations

import numpy as np


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation. Zero rows stay zero."""
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected shape (N, D), got {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, np.float32(eps))


def as_1d_labels(labels: np.ndarray) -> np.ndarray:
    return np.asarray(labels).reshape(-1).astype(np.int64, copy=False)
