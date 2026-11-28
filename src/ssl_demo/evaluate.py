"""kNN, linear probe, few-shot probes, and bootstrap intervals on frozen features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.neighbors import NearestNeighbors

from ssl_demo.arrays import as_1d_labels, l2_normalize
from ssl_demo.config import SEED
from ssl_demo.data import stratified_indices

PROBE_CS = (0.001, 0.01, 0.1, 1.0)

METRIC_FNS = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "macro_f1": lambda y_true, y_pred: f1_score(
        y_true, y_pred, average="macro", zero_division=0
    ),
}


def few_shot_indices(labels: np.ndarray, fraction: float, seed: int = SEED) -> np.ndarray:
    """Stratified subset of training rows. Fraction 1 keeps every row, in order."""
    labels = as_1d_labels(labels)
    if fraction >= 1:
        return np.arange(len(labels), dtype=np.int64)
    if fraction <= 0:
        raise ValueError("label fraction must be positive")
    n_take = max(len(np.unique(labels)), int(round(len(labels) * fraction)))
    return stratified_indices(labels, n_take, seed)


def knn_predict(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    k: int = 20,
    temperature: float = 0.07,
) -> np.ndarray:
    """Cosine kNN with temperature-weighted votes, as in the DINO evaluation."""
    train_x = l2_normalize(train_features)
    test_x = l2_normalize(test_features)
    train_y = as_1d_labels(train_labels)
    k = min(int(k), len(train_x))
    if k < 1:
        raise ValueError("kNN needs at least one training feature")
    neighbours = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    neighbours.fit(train_x)
    distances, indices = neighbours.kneighbors(test_x, return_distance=True)
    similarity = 1.0 - distances
    weights = np.exp(similarity / temperature)
    weights /= weights.sum(axis=1, keepdims=True)
    votes = np.zeros((len(test_x), int(train_y.max()) + 1), dtype=np.float64)
    neighbour_labels = train_y[indices]
    rows = np.arange(len(test_x))
    for column in range(k):
        votes[rows, neighbour_labels[:, column]] += weights[:, column]
    return votes.argmax(axis=1).astype(np.int64)


def fit_linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    seed: int = SEED,
) -> np.ndarray:
    """Standardised logistic regression on frozen features. The backbone is not updated.

    The scaler is fit on the training rows only, and C is chosen by stratified 3-fold
    cross-validation on those same rows (grid :data:`PROBE_CS`). Test tiles are never
    used to pick C. Without the scaler, an untrained backbone's nearly parallel features
    make the probe collapse towards a single class. When a class has fewer than three
    training rows, cross-validation is not possible and C falls back to 0.1.
    """
    labels = as_1d_labels(train_labels)
    _, counts = np.unique(labels, return_counts=True)
    if counts.min() >= 3:
        head = LogisticRegressionCV(
            Cs=list(PROBE_CS),
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
            max_iter=2000,
            random_state=seed,
        )
    else:
        head = LogisticRegression(C=0.1, max_iter=2000, random_state=seed)
    classifier = make_pipeline(StandardScaler(), head)
    classifier.fit(np.asarray(train_features, dtype=np.float64), labels)
    return classifier.predict(np.asarray(test_features, dtype=np.float64)).astype(np.int64)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    seed: int = SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval from resampling test rows. The point estimate is computed separately.

    Samples that make ``metric_fn`` undefined (a single class inside an AUROC
    resample, for example) are skipped.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot, dtype=np.float64)
    filled = 0
    attempts = 0
    while filled < n_boot and attempts < n_boot * 20:
        attempts += 1
        chosen = rng.integers(0, n, size=n)
        try:
            value = float(metric_fn(y_true[chosen], y_pred[chosen]))
        except ValueError:
            continue
        if not np.isfinite(value):
            continue
        stats[filled] = value
        filled += 1
    if filled == 0:
        point = float(metric_fn(y_true, y_pred))
        return point, point
    used = stats[:filled]
    low = float(np.percentile(used, 100 * alpha / 2))
    high = float(np.percentile(used, 100 * (1 - alpha / 2)))
    return low, high


def classification_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {name: float(fn(y_true, y_pred)) for name, fn in METRIC_FNS.items()}


def run_probe_suite(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    backbone: str,
    dataset: str,
    fractions: tuple[float, ...] = (0.01, 0.1, 1.0),
    seed: int = SEED,
    n_boot: int = 1000,
    k: int = 20,
    n_classes: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """kNN at full labels, linear probes at each label fraction, bootstrap CIs.

    The returned confusion matrix is the full-label linear probe.
    """
    train_y = as_1d_labels(train_labels)
    test_y = as_1d_labels(test_labels)
    if n_classes is None:
        n_classes = int(max(train_y.max(), test_y.max()) + 1)
    rows: list[dict] = []

    knn_pred = knn_predict(train_features, train_y, test_features, k=k)
    rows.extend(
        _metric_rows(
            y_true=test_y,
            y_pred=knn_pred,
            backbone=backbone,
            dataset=dataset,
            method="knn",
            fraction=1.0,
            n_boot=n_boot,
            seed=seed,
        )
    )

    full_probe = None
    for fraction in fractions:
        chosen = few_shot_indices(train_y, fraction, seed)
        probe_pred = fit_linear_probe(
            train_features[chosen], train_y[chosen], test_features, seed=seed
        )
        if fraction >= 1:
            full_probe = probe_pred
        rows.extend(
            _metric_rows(
                y_true=test_y,
                y_pred=probe_pred,
                backbone=backbone,
                dataset=dataset,
                method="linear_probe",
                fraction=float(fraction),
                n_boot=n_boot,
                seed=seed,
            )
        )
    if full_probe is None:
        full_probe = fit_linear_probe(train_features, train_y, test_features, seed=seed)
    matrix = confusion_matrix(test_y, full_probe, labels=list(range(n_classes)))
    frame = pd.DataFrame(rows)
    column_order = [
        "dataset",
        "backbone",
        "method",
        "label_fraction",
        "metric",
        "value",
        "ci_low",
        "ci_high",
    ]
    return frame[column_order], matrix


def random_floor_warning(frame: pd.DataFrame, margin: float = 0.05) -> str | None:
    """Flag the failure mode where an untrained backbone matches a trained one."""
    subset = frame[
        (frame["method"] == "knn")
        & (frame["metric"] == "accuracy")
        & (np.isclose(frame["label_fraction"].astype(float), 1.0))
    ]
    if subset.empty or "random_resnet50" not in set(subset["backbone"]):
        return None
    scores = {row.backbone: float(row.value) for row in subset.itertuples(index=False)}
    trained = {name: value for name, value in scores.items() if name != "random_resnet50"}
    if not trained:
        return None
    best_name = max(trained, key=trained.get)
    gap = trained[best_name] - scores["random_resnet50"]
    if gap < margin:
        return (
            f"Random ResNet-50 kNN accuracy is {scores['random_resnet50']:.3f}, "
            f"within {margin:.0%} of {best_name} ({trained[best_name]:.3f}). "
            "Check normalisation and that the cache is not being reused across different models."
        )
    return None


def _metric_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    backbone: str,
    dataset: str,
    method: str,
    fraction: float,
    n_boot: int,
    seed: int,
) -> list[dict]:
    rows = []
    for name, metric_fn in METRIC_FNS.items():
        value = float(metric_fn(y_true, y_pred))
        low, high = bootstrap_ci(y_true, y_pred, metric_fn, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "method": method,
                "label_fraction": fraction,
                "metric": name,
                "value": value,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return rows
