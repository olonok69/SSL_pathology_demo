"""Normal-only anomaly heads. The backbone stays frozen; only this head is fit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors

from ssl_demo.arrays import as_1d_labels, l2_normalize
from ssl_demo.config import SEED
from ssl_demo.evaluate import bootstrap_ci

DECISION_RULE = (
    "A backbone is clearly better if its AUROC 95% CI does not overlap the "
    "next-best backbone's CI on both Task A and Task B, for the kNN-distance "
    "head. Same tiles, same head, same seeds. The rule is fixed before looking "
    "at the numbers. If Task B features are not cached, the rule cannot declare "
    "a cross-task winner."
)

HEAD_NAMES = ("knn_distance", "mahalanobis", "isolation_forest")


def _as_features(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected features with shape (N, D), got {matrix.shape}")
    if len(matrix) == 0:
        raise ValueError("Need at least one feature vector")
    return matrix


class KNNDistanceHead:
    """Mean cosine distance to the k nearest normal features (PatchCore-style, image level).

    With ``center=True`` (the default) the mean of the normal fit features is subtracted
    before L2 normalisation, for the bank and for every query. Pretrained features share a
    large common direction (mean cosine to the normal centroid is about 0.82 for ImageNet
    ResNet-50 and DINOv2 on NCT-CRC tiles), so raw cosine distances are compressed and the
    ranking is driven by nuisance directions. An untrained ResNet-50 is the extreme case:
    every tile points almost exactly along the centroid (cosine 1.000), and its raw cosine
    distance behaves like a stain-intensity measure, not a tissue feature.
    """

    def __init__(
        self,
        k: int = 5,
        coreset_size: int | None = 2000,
        seed: int = SEED,
        center: bool = True,
    ):
        self.k = k
        self.coreset_size = coreset_size
        self.seed = seed
        self.center = center
        self.mean_: np.ndarray | None = None
        self._neighbours: NearestNeighbors | None = None

    def _prepare(self, features: np.ndarray) -> np.ndarray:
        matrix = _as_features(features)
        if self.mean_ is not None:
            matrix = matrix - self.mean_
        return l2_normalize(matrix).astype(np.float64)

    def fit(self, normal_features: np.ndarray) -> "KNNDistanceHead":
        matrix = _as_features(normal_features)
        self.mean_ = matrix.mean(axis=0) if self.center else None
        bank = self._prepare(matrix)
        if self.coreset_size is not None and len(bank) > self.coreset_size:
            rng = np.random.default_rng(self.seed)
            chosen = rng.choice(len(bank), size=self.coreset_size, replace=False)
            bank = bank[chosen]
        k = min(self.k, len(bank))
        self._neighbours = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        self._neighbours.fit(bank)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self._neighbours is None:
            raise RuntimeError("Call fit before score")
        distances, _ = self._neighbours.kneighbors(self._prepare(features), return_distance=True)
        return distances.mean(axis=1).astype(np.float64)


def centroid_cosine(features: np.ndarray) -> float:
    """Mean cosine between each feature vector and the mean vector.

    Close to 1 means every tile points the same way and raw cosine distances say little.
    """
    matrix = _as_features(features)
    unit = l2_normalize(matrix)
    mean = matrix.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm == 0:
        return 0.0
    return float(np.mean(unit @ (mean / norm)))


class MahalanobisHead:
    """Gaussian model of normal features with a shrinkage covariance (PaDiM-style)."""

    def __init__(self, shrinkage: float = 0.1, eps: float = 1e-6):
        if not 0 <= shrinkage <= 1:
            raise ValueError("shrinkage must be between 0 and 1")
        self.shrinkage = shrinkage
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None

    def fit(self, normal_features: np.ndarray) -> "MahalanobisHead":
        matrix = _as_features(normal_features)
        mean = matrix.mean(axis=0)
        centered = matrix - mean
        covariance = (centered.T @ centered) / max(len(matrix) - 1, 1)
        dimension = covariance.shape[0]
        target = (np.trace(covariance) / dimension) * np.eye(dimension)
        shrunk = (1.0 - self.shrinkage) * covariance + self.shrinkage * target
        shrunk = shrunk + self.eps * np.eye(dimension)
        self.mean_ = mean
        self.precision_ = np.linalg.pinv(shrunk, rcond=1e-6)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.precision_ is None:
            raise RuntimeError("Call fit before score")
        matrix = _as_features(features)
        delta = matrix - self.mean_
        left = delta @ self.precision_
        squared = np.sum(left * delta, axis=1)
        return np.sqrt(np.clip(squared, 0, None))


class IsolationForestHead:
    """Isolation Forest on normal features. Higher score means more anomalous.

    sklearn's ``decision_function`` is higher for inliers, so this head negates it.
    """

    def __init__(self, seed: int = SEED, n_estimators: int = 200):
        self.seed = seed
        self.n_estimators = n_estimators
        self._model: IsolationForest | None = None

    def fit(self, normal_features: np.ndarray) -> "IsolationForestHead":
        matrix = _as_features(normal_features)
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            n_jobs=1,
        )
        self._model.fit(matrix)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit before score")
        matrix = _as_features(features)
        return -self._model.decision_function(matrix)


def build_head(name: str, seed: int = SEED, **kwargs):
    if name == "knn_distance":
        return KNNDistanceHead(
            k=kwargs.get("k", 5),
            coreset_size=kwargs.get("coreset_size", 2000),
            seed=seed,
            center=kwargs.get("center", True),
        )
    if name == "mahalanobis":
        return MahalanobisHead(shrinkage=kwargs.get("shrinkage", 0.1))
    if name == "isolation_forest":
        return IsolationForestHead(
            seed=seed,
            n_estimators=kwargs.get("n_estimators", 200),
        )
    raise KeyError(f"Unknown head {name!r}. Known: {', '.join(HEAD_NAMES)}")


def split_fit_holdout(
    n: int,
    holdout_fraction: float = 0.2,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Split row positions into a fit set and a held-out normal set for the threshold."""
    if n < 2:
        raise ValueError("Need at least 2 normal samples so the threshold set is held out")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    n_hold = int(round(n * holdout_fraction))
    n_hold = min(max(n_hold, 1), n - 1)
    return permutation[n_hold:], permutation[:n_hold]


def threshold_at_percentile(normal_scores: np.ndarray, percentile: float = 99) -> float:
    return float(np.percentile(np.asarray(normal_scores, dtype=np.float64), percentile))


def rates_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    """Sensitivity and false-positive rate for scores at or above ``threshold``."""
    y_true = np.asarray(y_true).astype(bool)
    predicted = np.asarray(scores) >= threshold
    true_positive = int(np.sum(predicted & y_true))
    false_negative = int(np.sum(~predicted & y_true))
    false_positive = int(np.sum(predicted & ~y_true))
    true_negative = int(np.sum(~predicted & ~y_true))
    sensitivity = true_positive / max(true_positive + false_negative, 1)
    false_positive_rate = false_positive / max(false_positive + true_negative, 1)
    return float(sensitivity), float(false_positive_rate)


def fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, tpr_target: float = 0.95) -> float:
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
    hits = np.flatnonzero(true_positive_rate >= tpr_target)
    if len(hits) == 0:
        return 1.0
    return float(false_positive_rate[hits[0]])


def normal_pool(
    features: np.ndarray,
    labels: np.ndarray,
    normal_ids,
) -> tuple[np.ndarray, np.ndarray]:
    labels = as_1d_labels(labels)
    mask = np.isin(labels, list(normal_ids))
    return features[mask], np.flatnonzero(mask)


def anomaly_test_set(
    features: np.ndarray,
    labels: np.ndarray,
    normal_ids,
    anomaly_ids,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop classes that are neither normal nor anomaly (background, on Task A).

    Returns features, binary labels (1 = anomaly), and positions into the original split.
    """
    labels = as_1d_labels(labels)
    normal = np.isin(labels, list(normal_ids))
    anomaly = np.isin(labels, list(anomaly_ids))
    keep = normal | anomaly
    binary = anomaly[keep].astype(np.int64)
    return features[keep], binary, np.flatnonzero(keep)


def highest_scoring(positions: np.ndarray, scores: np.ndarray, k: int = 6):
    """Return the ``k`` positions with the highest anomaly scores, and those scores."""
    positions = np.asarray(positions)
    scores = np.asarray(scores, dtype=np.float64)
    k = min(int(k), len(scores))
    order = np.argsort(-scores)[:k]
    return positions[order], scores[order]


@dataclass
class AnomalyRun:
    table: pd.DataFrame
    scores: dict[str, np.ndarray]
    test_positions: np.ndarray
    test_y: np.ndarray
    thresholds: dict[str, float]


def run_anomaly_task(
    fit_pool: np.ndarray,
    test_features: np.ndarray,
    test_y: np.ndarray,
    backbone: str,
    task: str,
    dataset: str,
    seed: int = SEED,
    n_boot: int = 1000,
    holdout_fraction: float = 0.2,
    heads: tuple[str, ...] = HEAD_NAMES,
    background_features: np.ndarray | None = None,
    test_positions: np.ndarray | None = None,
    **head_kwargs,
) -> AnomalyRun:
    """Fit each head on held-in normals, threshold on held-out normals, score the test rows."""
    fit_positions, hold_positions = split_fit_holdout(len(fit_pool), holdout_fraction, seed)
    rows: list[dict] = []
    scores: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    for head_name in heads:
        head = build_head(head_name, seed=seed, **head_kwargs)
        head.fit(fit_pool[fit_positions])
        hold_scores = head.score(fit_pool[hold_positions])
        test_scores = head.score(test_features)
        threshold = threshold_at_percentile(hold_scores, 99)
        thresholds[head_name] = threshold
        scores[head_name] = test_scores
        rows.extend(
            _score_rows(
                y_true=test_y,
                scores=test_scores,
                threshold=threshold,
                backbone=backbone,
                task=task,
                dataset=dataset,
                head=head_name,
                n_boot=n_boot,
                seed=seed,
            )
        )
        if background_features is not None and len(background_features):
            background_scores = head.score(background_features)
            rows.append(
                _flag_rate_row(
                    background_scores,
                    threshold,
                    backbone=backbone,
                    task=f"{task}_background",
                    dataset=dataset,
                    head=head_name,
                    n_boot=n_boot,
                    seed=seed,
                )
            )
    frame = pd.DataFrame(rows)
    columns = [
        "dataset",
        "task",
        "backbone",
        "head",
        "metric",
        "value",
        "ci_low",
        "ci_high",
    ]
    if test_positions is None:
        test_positions = np.arange(len(test_y), dtype=np.int64)
    return AnomalyRun(
        table=frame[columns],
        scores=scores,
        test_positions=as_1d_labels(test_positions),
        test_y=np.asarray(test_y).astype(np.int64),
        thresholds=thresholds,
    )


def ci_overlap(low_a: float, high_a: float, low_b: float, high_b: float) -> bool:
    """True when the intervals touch or overlap. A gap is required to call a winner."""
    return not (high_a < low_b or high_b < low_a)


@dataclass
class Decision:
    winner: str | None
    reason: str
    rule: str = DECISION_RULE


def decide_backbone(
    frame: pd.DataFrame,
    head: str = "knn_distance",
    metric: str = "auroc",
) -> Decision:
    """Apply :data:`DECISION_RULE` to a results table. Does not look at other metrics."""
    subset = frame[(frame["head"] == head) & (frame["metric"] == metric)].copy()
    tasks = [task for task in sorted(subset["task"].unique()) if not str(task).endswith("_background")]
    if not tasks:
        return Decision(None, "No AUROC rows to apply the rule to.")
    leaders = []
    notes = []
    for task in tasks:
        block = subset[subset["task"] == task].sort_values("value", ascending=False)
        if len(block) < 2:
            notes.append(f"{task}: only one backbone, so there is no next-best CI to compare.")
            leaders.append(None)
            continue
        best = block.iloc[0]
        second = block.iloc[1]
        separated = not ci_overlap(best.ci_low, best.ci_high, second.ci_low, second.ci_high)
        if separated:
            notes.append(
                f"{task}: {best.backbone} AUROC {best.value:.3f} "
                f"[{best.ci_low:.3f}, {best.ci_high:.3f}] is clear of "
                f"{second.backbone} [{second.ci_low:.3f}, {second.ci_high:.3f}]."
            )
            leaders.append(best.backbone)
        else:
            notes.append(
                f"{task}: {best.backbone} leads on the point estimate "
                f"({best.value:.3f}) but its CI overlaps {second.backbone}."
            )
            leaders.append(None)
    required = {"task_a", "task_b"}
    have = set(tasks)
    if not required.issubset(have):
        missing = ", ".join(sorted(required - have))
        reason = (
            f"The two-task rule cannot declare a winner because {missing} is missing. "
            + " ".join(notes)
        )
        return Decision(None, reason)
    floor = _random_floor_note(subset, tasks)
    if floor:
        return Decision(None, floor + " " + " ".join(notes))
    if leaders[0] is not None and all(leader == leaders[0] for leader in leaders):
        reason = f"{leaders[0]} is clearly better on both tasks. " + " ".join(notes)
        return Decision(str(leaders[0]), reason)
    reason = "No backbone is clearly better on both tasks. " + " ".join(notes)
    return Decision(None, reason)


def _random_floor_note(subset: pd.DataFrame, tasks: list[str], margin: float = 0.02) -> str | None:
    """Refuse to name a winner when an untrained backbone leads, or sits within ``margin``."""
    for task in tasks:
        block = subset[subset["task"] == task]
        is_random = block["backbone"].astype(str).str.startswith("random")
        if not is_random.any() or is_random.all():
            continue
        random_value = float(block.loc[is_random, "value"].max())
        best_trained = float(block.loc[~is_random, "value"].max())
        if random_value >= best_trained - margin:
            return (
                f"Pipeline check: on {task} the untrained backbone scores {random_value:.3f} "
                f"against {best_trained:.3f} for the best trained one. A random network should "
                "be the floor, so no winner is declared until the head or the features are checked."
            )
    return None


def _auroc(y_true, scores) -> float:
    if len(np.unique(y_true)) < 2:
        raise ValueError("AUROC needs both classes")
    return float(roc_auc_score(y_true, scores))


def _auprc(y_true, scores) -> float:
    if len(np.unique(y_true)) < 2:
        raise ValueError("AUPRC needs both classes")
    return float(average_precision_score(y_true, scores))


def _score_rows(
    y_true,
    scores,
    threshold,
    backbone,
    task,
    dataset,
    head,
    n_boot,
    seed,
) -> list[dict]:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)

    def sensitivity(y, s, thr=threshold):
        return rates_at_threshold(y, s, thr)[0]

    def false_positive_rate(y, s, thr=threshold):
        return rates_at_threshold(y, s, thr)[1]

    def fpr95(y, s):
        return fpr_at_tpr(y, s, 0.95)

    named = {
        "auroc": _auroc,
        "auprc": _auprc,
        "fpr_at_95_tpr": fpr95,
        "sensitivity_at_99": sensitivity,
        "fpr_at_99": false_positive_rate,
    }
    rows = []
    for metric, fn in named.items():
        value = float(fn(y_true, scores))
        low, high = bootstrap_ci(y_true, scores, fn, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "backbone": backbone,
                "head": head,
                "metric": metric,
                "value": value,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return rows


def _flag_rate_row(scores, threshold, backbone, task, dataset, head, n_boot, seed) -> dict:
    scores = np.asarray(scores, dtype=np.float64)

    def flag_rate(y_ignored, s, thr=threshold):
        del y_ignored
        return float(np.mean(np.asarray(s) >= thr))

    dummy = np.zeros(len(scores), dtype=np.int64)
    value = flag_rate(dummy, scores)
    low, high = bootstrap_ci(dummy, scores, flag_rate, n_boot=n_boot, seed=seed)
    return {
        "dataset": dataset,
        "task": task,
        "backbone": backbone,
        "head": head,
        "metric": "flag_rate",
        "value": value,
        "ci_low": low,
        "ci_high": high,
    }
