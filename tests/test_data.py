from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ssl_demo.data import (
    BACKGROUND_ID,
    CLASS_NAMES,
    TUMOUR_ID,
    StainJitter,
    augmentation_previews,
    build_or_load_index,
    class_balance,
    find_nct_directory,
    pcam_is_downloaded,
    per_class_positions,
    prepare_nct,
    scan_class_folders,
    stain_jitter_array,
    stratified_indices,
)


def test_stratified_indices_are_balanced_and_repeatable():
    labels = np.array([0] * 90 + [1] * 10)
    first = stratified_indices(labels, 50, seed=1)
    second = stratified_indices(labels, 50, seed=1)
    assert np.array_equal(first, second)
    assert len(first) == 50
    assert len(np.unique(first)) == 50
    counts = np.bincount(labels[first])
    assert counts[0] == 45
    assert counts[1] == 5


def test_index_csv_is_reused(tmp_path):
    labels = np.arange(20) % 4
    path = tmp_path / "train.csv"
    first = build_or_load_index(labels, 12, seed=0, csv_path=path)
    second = build_or_load_index(labels, 4, seed=1, csv_path=path)
    assert np.array_equal(first, second)
    saved = pd.read_csv(path)
    assert list(saved.columns) == ["index", "label"]
    assert len(saved) == 12


def test_stain_jitter_roundtrip_and_change():
    rgb = np.zeros((8, 8, 3), dtype=np.float64)
    rgb[..., 0] = 0.7
    rgb[..., 1] = 0.3
    rgb[..., 2] = 0.4
    identity = stain_jitter_array(rgb, sigma=0)
    assert identity.shape == rgb.shape
    assert np.max(np.abs(identity - rgb)) < 1e-5
    changed = stain_jitter_array(rgb, sigma=0.4, seed=2)
    assert changed.shape == rgb.shape
    assert np.all(changed >= 0) and np.all(changed <= 1)
    assert not np.allclose(changed, rgb)
    image = Image.fromarray(np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8))
    out = StainJitter(sigma=0)(image)
    assert isinstance(out, Image.Image)
    assert out.size == image.size


def test_augmentation_preview_keys():
    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, size=(32, 32, 3), dtype=np.uint8))
    previews = augmentation_previews(image, seed=0)
    assert set(previews) == {"random_crop", "colour_jitter", "blur", "solarize", "stain_jitter"}
    for preview in previews.values():
        assert isinstance(preview, Image.Image)


def test_class_balance_and_positions():
    labels = np.array([0, 0, 1, 2, 2, 2])
    frame = class_balance(labels, ["a", "b", "c"])
    assert frame["n"].tolist() == [2, 1, 3]
    positions = per_class_positions(labels, per_class=1, seed=0)
    assert set(positions) == {0, 1, 2}
    assert all(len(value) == 1 for value in positions.values())


def test_nct_folder_roundtrip(tmp_path):
    root = tmp_path / "NCT-CRC-HE-100K"
    val = tmp_path / "CRC-VAL-HE-7K"
    for folder, n_images in ((root, 2), (val, 1)):
        for name in ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]:
            class_dir = folder / name
            class_dir.mkdir(parents=True)
            for i in range(n_images):
                Image.new("RGB", (8, 8), color=(i * 20, 40, 80)).save(class_dir / f"{i}.png")
    assert find_nct_directory(tmp_path, "train") == root
    paths, labels = scan_class_folders(root, ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"])
    assert len(paths) == 18
    assert labels[-1] == 8
    sources = prepare_nct(tmp_path, mode="fast", cache=tmp_path / "cache", seed=0)
    train = sources["train"]
    assert train.dataset_name == "nct"
    assert train.tumour_id == TUMOUR_ID
    assert train.background_id == BACKGROUND_ID
    assert set(train.normal_ids()) == {0, 2, 3, 4, 5, 6, 7}
    assert len(train.subset_labels()) == 18
    image = train.image(int(train.indices[0]))
    assert image.size == (8, 8)
    assert len(CLASS_NAMES) == 9


def test_feature_dir_keeps_modes_apart(tmp_path):
    from ssl_demo.data import feature_dir

    fast = feature_dir(tmp_path, "pathmnist", "train", "fast")
    full = feature_dir(tmp_path, "pathmnist", "train", "full")
    assert fast != full
    assert fast.parts[-3:] == ("pathmnist", "fast", "train")
    try:
        feature_dir(tmp_path, "pathmnist", "train", "slow")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_pcam_missing(tmp_path):
    assert pcam_is_downloaded(tmp_path) is False
    assert find_nct_directory(tmp_path / "empty", "test") is None
    assert not Path(tmp_path / "nope").exists()
