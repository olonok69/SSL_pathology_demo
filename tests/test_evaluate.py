import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from ssl_demo.evaluate import (
    bootstrap_ci,
    few_shot_indices,
    fit_linear_probe,
    knn_predict,
    random_floor_warning,
    run_probe_suite,
)
from ssl_demo.viz import plot_embedding


def test_knn_matches_sklearn_at_k1():
    rng = np.random.default_rng(0)
    centers = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    train = np.vstack([center + rng.normal(0, 0.01, size=(6, 3)) for center in centers])
    train_y = np.repeat(np.arange(3), 6)
    test = centers + rng.normal(0, 0.01, size=(3, 3))
    test_y = np.arange(3)
    ours = knn_predict(train, train_y, test, k=1)
    sklearn_knn = KNeighborsClassifier(n_neighbors=1, metric="cosine", algorithm="brute")
    sklearn_knn.fit(train, train_y)
    assert np.array_equal(ours, sklearn_knn.predict(test))
    assert np.array_equal(ours, test_y)


def test_linear_probe_matches_sklearn_metrics():
    rng = np.random.default_rng(1)
    train = np.vstack(
        [
            rng.normal(0, 0.1, size=(40, 4)),
            rng.normal(3, 0.1, size=(40, 4)),
        ]
    )
    train_y = np.array([0] * 40 + [1] * 40)
    test = np.vstack(
        [
            rng.normal(0, 0.1, size=(10, 4)),
            rng.normal(3, 0.1, size=(10, 4)),
        ]
    )
    test_y = np.array([0] * 10 + [1] * 10)
    predicted = fit_linear_probe(train, train_y, test, seed=0)
    assert accuracy_score(test_y, predicted) == 1.0
    assert f1_score(test_y, predicted, average="macro") == 1.0


def test_bootstrap_interval_on_perfect_predictions():
    labels = np.array([0, 1, 0, 1, 0, 1])
    low, high = bootstrap_ci(labels, labels, accuracy_score, n_boot=25, seed=0)
    assert low == 1.0 and high == 1.0
    mixed = np.array([0, 1, 1, 1, 0, 0])
    low, high = bootstrap_ci(labels, mixed, accuracy_score, n_boot=40, seed=2)
    assert 0.0 <= low <= high <= 1.0
    again = bootstrap_ci(labels, mixed, accuracy_score, n_boot=40, seed=2)
    assert again == (low, high)


def test_few_shot_and_probe_suite_columns():
    labels = np.array([0, 1, 2] * 10)
    assert np.array_equal(few_shot_indices(labels, 1.0, seed=0), np.arange(30))
    subset = few_shot_indices(labels, 0.5, seed=0)
    assert len(subset) == 15
    rng = np.random.default_rng(0)
    train = rng.normal(size=(30, 6))
    test = rng.normal(size=(12, 6))
    train[labels == 0, 0] += 4
    test_labels = np.array([0, 1, 2] * 4)
    test[test_labels == 0, 0] += 4
    frame, matrix = run_probe_suite(
        train,
        labels,
        test,
        test_labels,
        backbone="toy",
        dataset="pathmnist",
        fractions=(1.0,),
        n_boot=15,
        k=3,
        seed=0,
    )
    assert list(frame.columns) == [
        "dataset",
        "backbone",
        "method",
        "label_fraction",
        "metric",
        "value",
        "ci_low",
        "ci_high",
    ]
    assert set(frame["method"]) == {"knn", "linear_probe"}
    assert matrix.shape[0] == 3
    assert frame["ci_low"].le(frame["ci_high"]).all()


def test_random_floor_warning():
    frame = pd.DataFrame(
        [
            {"backbone": "random_resnet50", "method": "knn", "metric": "accuracy", "label_fraction": 1.0, "value": 0.40},
            {"backbone": "sup_resnet50_in1k", "method": "knn", "metric": "accuracy", "label_fraction": 1.0, "value": 0.80},
        ]
    )
    assert random_floor_warning(frame) is None
    frame.loc[frame["backbone"] == "random_resnet50", "value"] = 0.79
    assert random_floor_warning(frame) is not None


def test_plot_embedding_one_series_per_class():
    import matplotlib

    matplotlib.use("Agg", force=True)
    embedding = np.array([[0.0, 0.0], [1.0, 1.0], [0.2, 0.1], [1.2, 0.8]])
    labels = np.array([0, 1, 0, 1])
    ax = plot_embedding(embedding, labels, ["normal", "tumour"], title="toy")
    assert len(ax.collections) == 2
    import matplotlib.pyplot as plt

    plt.close(ax.figure)


def test_auroc_reference_is_importable():
    assert roc_auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
