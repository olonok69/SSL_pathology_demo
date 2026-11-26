import pandas as pd

from ssl_demo.config import (
    COLAB_FALLBACK,
    DRIVE_ROOT,
    cache_root,
    save_results_csv,
    subset_count,
)


def test_subset_counts():
    assert subset_count("fast", "train") == 10_000
    assert subset_count("fast", "test") == 2_000
    assert subset_count("full", "train", dataset="pcam") == 20_000
    try:
        subset_count("slow", "train")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_cache_root_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ssl_demo.config.in_colab", lambda: False)
    path = cache_root()
    assert path == tmp_path / "cache"
    assert path.is_dir()


def test_cache_root_colab_drive(tmp_path, monkeypatch):
    drive = tmp_path / "MyDrive"
    drive.mkdir()
    monkeypatch.setattr("ssl_demo.config.in_colab", lambda: True)
    monkeypatch.setattr("ssl_demo.config.DRIVE_ROOT", drive)
    assert cache_root() == drive / "ssl-demo"


def test_cache_root_colab_without_drive(tmp_path, monkeypatch):
    monkeypatch.setattr("ssl_demo.config.in_colab", lambda: True)
    monkeypatch.setattr("ssl_demo.config.DRIVE_ROOT", tmp_path / "missing")
    fallback = tmp_path / "fallback"
    monkeypatch.setattr("ssl_demo.config.COLAB_FALLBACK", fallback)
    assert cache_root() == fallback
    assert DRIVE_ROOT.name == "MyDrive"
    assert COLAB_FALLBACK.name == "ssl-demo-cache"


def test_save_results_mirrors_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame = pd.DataFrame([{"backbone": "random_resnet50", "value": 0.5}])
    written = save_results_csv(frame, "probe_nct.csv", cache=tmp_path / "drive", mirror_repo=True)
    assert (tmp_path / "drive" / "results" / "probe_nct.csv").is_file()
    assert (tmp_path / "results" / "probe_nct.csv").is_file()
    assert len(written) == 2
