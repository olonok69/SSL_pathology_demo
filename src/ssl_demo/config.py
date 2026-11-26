"""Paths, seeds, and the fast/full switch shared by every notebook."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SEED = 42

# Colab Drive cache. Tests patch these two constants.
DRIVE_ROOT = Path("/content/drive/MyDrive")
COLAB_FALLBACK = Path("/content/ssl-demo-cache")

PROBE_CSV = "probe_nct.csv"
ANOMALY_CSV = "anomaly.csv"
THROUGHPUT_CSV = "throughput.csv"

SUBSET_COUNTS = {
    "fast": {"train": 10_000, "test": 2_000},
    "full": {"train": 50_000, "test": 7_000},
}

# PCam is a second download. Fast mode leaves it off unless the index CSV is already there.
PCAM_COUNTS = {
    "fast": {"train": 5_000, "test": 2_000},
    "full": {"train": 20_000, "test": 5_000},
}


def in_colab() -> bool:
    return "google.colab" in sys.modules


def cache_root(override: str | Path | None = None) -> Path:
    """Directory that survives a Colab disconnect.

    On Colab this is ``/content/drive/MyDrive/ssl-demo`` once Drive is mounted.
    Locally it is ``./cache``. Pass ``override`` in tests and one-off runs.
    """
    if override is not None:
        path = Path(override)
    elif in_colab() and DRIVE_ROOT.exists():
        path = DRIVE_ROOT / "ssl-demo"
    elif in_colab():
        path = COLAB_FALLBACK
    else:
        path = Path("cache")
    path = path.expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def subset_count(mode: str, split: str, dataset: str = "pathmnist") -> int:
    mode = mode.lower()
    if mode not in ("fast", "full"):
        raise ValueError(f"MODE must be 'fast' or 'full', got {mode!r}")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    table = PCAM_COUNTS if dataset == "pcam" else SUBSET_COUNTS
    return table[mode][split]


def default_num_workers() -> int:
    """Spec: 2 workers on Colab. Windows local runs stay on the main process."""
    return 2 if in_colab() else 0


def save_results_csv(
    frame: pd.DataFrame,
    filename: str,
    cache: str | Path,
    mirror_repo: bool = True,
) -> list[Path]:
    """Write a results table to the Drive/cache folder and, optionally, ``./results``.

    The cache copy is the one that survives a Colab restart. The repo copy is
    what ``docs/guide/make_figures.py`` will read later.
    """
    cache_path = Path(cache) / "results" / filename
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    written = [cache_path]
    if mirror_repo:
        repo_path = Path("results") / filename
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(repo_path, index=False)
        written.append(repo_path)
    return written
