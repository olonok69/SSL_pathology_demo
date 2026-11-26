import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from ssl_demo.anomaly import (
    DECISION_RULE,
    IsolationForestHead,
    KNNDistanceHead,
    MahalanobisHead,
    anomaly_test_set,
    ci_overlap,
    decide_backbone,
    fpr_at_tpr,
    highest_scoring,
    rates_at_threshold,
    run_anomaly_task,
    split_fit_holdout,
    threshold_at_percentile,
)


def test_knn_distance_orders_near_and_far():
    bank = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]])
    head = KNNDistanceHead(k=1, coreset_size=None, seed=0)
    head.fit(bank)
    scores = head.score(np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))
    assert scores[0] < 0.05
    assert scores[1] > 1.5


def test_mahalanobis_far_point_and_wide_features():
    rng = np.random.default_rng(0)
    normal = rng.normal(size=(80, 3))
    head = MahalanobisHead(shrinkage=0.1)
    head.fit(normal)
    scores = head.score(np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]]))
    assert scores[1] > scores[0]
    wide = MahalanobisHead(shrinkage=0.2)
    wide.fit(rng.normal(size=(6, 12)))
    assert wide.score(rng.normal(size=(3, 12))).shape == (3,)


def test_isolation_forest_matches_negated_decision_function():
    rng = np.random.default_rng(0)
    normal = rng.normal(size=(40, 4))
    query = np.vstack([rng.normal(size=(5, 4)), np.ones((5, 4)) * 6])
    head = IsolationForestHead(seed=0, n_estimators=30)
    head.fit(normal)
    scores = head.score(query)
    assert np.allclose(scores, -head._model.decision_function(query))
    assert scores[5:].mean() > scores[:5].mean()


def test_rates_threshold_and_fpr_match_sklearn():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    assert roc_auc_score(labels, scores) == 1.0
    assert average_precision_score(labels, scores) == 1.0
    assert fpr_at_tpr(labels, scores, 0.95) == 0.0
    curve_fpr, curve_tpr, _ = roc_curve(labels, scores)
    expected = float(curve_fpr[np.flatnonzero(curve_tpr >= 0.95)[0]])
    assert fpr_at_tpr(labels, scores) == expected
    sensitivity, false_positive_rate = rates_at_threshold(labels, scores, threshold=0.55)
    assert sensitivity == 1.0
    assert false_positive_rate == 0.0
    sensitivity, false_positive_rate = rates_at_threshold(
        np.array([0, 0, 1, 1]),
        np.array([0.0, 1.0, 2.0, 3.0]),
        threshold=1.0,
    )
    assert sensitivity == 1.0
    assert false_positive_rate == 0.5
    normals = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert threshold_at_percentile(normals, 99) == float(np.percentile(normals, 99))


def test_split_and_task_runner(tmp_path):
    del tmp_path
    rng = np.random.default_rng(0)
    fit_pool = rng.normal(size=(30, 4))
    test_normal = rng.normal(size=(20, 4))
    test_anomaly = rng.normal(loc=5.0, size=(20, 4))
    test = np.vstack([test_normal, test_anomaly])
    test_y = np.array([0] * 20 + [1] * 20)
    fit_idx, hold_idx = split_fit_holdout(len(fit_pool), 0.2, seed=0)
    assert len(np.intersect1d(fit_idx, hold_idx)) == 0
    assert len(fit_idx) + len(hold_idx) == len(fit_pool)
    run = run_anomaly_task(
        fit_pool,
        test,
        test_y,
        backbone="toy",
        task="task_a",
        dataset="pathmnist",
        n_boot=20,
        heads=("knn_distance", "mahalanobis"),
        coreset_size=None,
        k=3,
        n_estimators=10,
        background_features=rng.normal(loc=4.0, size=(6, 4)),
        test_positions=np.arange(40) + 100,
    )
    assert set(run.table["metric"]) >= {"auroc", "auprc", "fpr_at_95_tpr", "flag_rate"}
    auroc = run.table[(run.table["metric"] == "auroc") & (run.table["head"] == "knn_distance")]
    assert float(auroc["value"].iloc[0]) > 0.8
    assert np.array_equal(run.test_positions, np.arange(40) + 100)
    labels = np.array([0, 0, 8, 1, 8])
    features = np.zeros((5, 2))
    kept, binary, positions = anomaly_test_set(features, labels, normal_ids=(0, 2), anomaly_ids=(8,))
    assert binary.tolist() == [0, 0, 1, 1]
    assert positions.tolist() == [0, 1, 2, 4]
    assert len(kept) == 4
    top_positions, top_scores = highest_scoring(np.array([3, 1, 4]), np.array([0.2, 0.9, 0.4]), k=2)
    assert top_positions.tolist() == [1, 4]
    assert top_scores.tolist() == [0.9, 0.4]


def test_decision_rule():
    assert "both Task A and Task B" in DECISION_RULE
    assert ci_overlap(0.0, 1.0, 1.0, 2.0)
    assert not ci_overlap(0.0, 0.4, 0.5, 0.9)

    def row(backbone, task, value, low, high):
        return {
            "dataset": "pathmnist",
            "task": task,
            "backbone": backbone,
            "head": "knn_distance",
            "metric": "auroc",
            "value": value,
            "ci_low": low,
            "ci_high": high,
        }

    clear = pd.DataFrame(
        [
            row("dinov2_vits14", "task_a", 0.91, 0.88, 0.94),
            row("sup_resnet50_in1k", "task_a", 0.70, 0.66, 0.74),
            row("dinov2_vits14", "task_b", 0.90, 0.86, 0.93),
            row("sup_resnet50_in1k", "task_b", 0.62, 0.58, 0.66),
        ]
    )
    decision = decide_backbone(clear)
    assert decision.winner == "dinov2_vits14"

    overlap = pd.DataFrame(
        [
            row("dinov2_vits14", "task_a", 0.80, 0.70, 0.90),
            row("sup_resnet50_in1k", "task_a", 0.78, 0.69, 0.88),
            row("dinov2_vits14", "task_b", 0.90, 0.86, 0.93),
            row("sup_resnet50_in1k", "task_b", 0.62, 0.58, 0.66),
        ]
    )
    assert decide_backbone(overlap).winner is None

    one_task = clear[clear["task"] == "task_a"]
    only_a = decide_backbone(one_task)
    assert only_a.winner is None
    assert "task_b" in only_a.reason
