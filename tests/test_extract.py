import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from ssl_demo.extract import extract_dataset, is_cached, load_cached, result_row


class ToyTiles(Dataset):
    def __init__(self, n=4):
        self.images = torch.rand(n, 3, 8, 8)
        self.labels = torch.arange(n) % 2
        self.indices = np.arange(n, dtype=np.int64) * 10

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], int(self.labels[i])


class CountingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.proj = nn.Linear(3, 5, bias=False)

    def forward(self, images):
        self.calls += 1
        return self.proj(images.mean(dim=(2, 3)))


class FakeDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward_features(self, images):
        self.calls += 1
        batch = images.shape[0]
        cls = torch.ones(batch, 4)
        patches = torch.ones(batch, 2, 4) * 2
        return {"x_norm_clstoken": cls, "x_norm_patchtokens": patches}

    def forward(self, images):
        return self.forward_features(images)["x_norm_clstoken"]


def test_cache_miss_then_hit(tmp_path):
    model = CountingEncoder()
    dataset = ToyTiles()
    cache = tmp_path / "features"
    first = extract_dataset(
        "toy",
        model,
        dataset,
        cache,
        kind="pooled",
        device="cpu",
        batch_size=2,
        num_workers=0,
    )
    assert first.cache_hit is False
    assert model.calls == 2
    assert first.features.shape == (4, 5)
    assert np.allclose(np.linalg.norm(first.features, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(first.labels, dataset.labels.numpy())
    assert np.array_equal(first.index, dataset.indices)
    assert is_cached(cache, "toy")

    calls = model.calls
    second = extract_dataset(
        "toy",
        model,
        dataset,
        cache,
        kind="pooled",
        device="cpu",
        batch_size=2,
        num_workers=0,
    )
    assert second.cache_hit is True
    assert model.calls == calls
    assert np.allclose(second.features, first.features)
    row = result_row(second, "toyset", "train")
    assert row["cache_hit"] is True
    assert row["n_tiles"] == 4


def test_cache_miss_when_the_tile_list_changes(tmp_path):
    model = CountingEncoder()
    dataset = ToyTiles()
    cache = tmp_path / "features"
    extract_dataset("toy", model, dataset, cache, device="cpu", batch_size=4, num_workers=0)
    dataset.indices = dataset.indices + 1
    model.calls = 0
    again = extract_dataset("toy", model, dataset, cache, device="cpu", batch_size=4, num_workers=0)
    assert again.cache_hit is False
    assert model.calls == 1
    assert np.array_equal(again.index, dataset.indices)


def test_load_cached_missing(tmp_path):
    try:
        load_cached(tmp_path, "missing")
    except FileNotFoundError as exc:
        assert "notebook 03" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_dinov2_style_patch_mean_is_cached(tmp_path):
    model = FakeDino()
    dataset = ToyTiles()
    cache = tmp_path / "dino"
    first = extract_dataset(
        "dinov2_vits14",
        model,
        dataset,
        cache,
        kind="cls+patchmean",
        device="cpu",
        batch_size=4,
        num_workers=0,
    )
    assert first.patchmean is not None
    assert first.patchmean.shape == (4, 4)
    assert np.allclose(first.features, first.patchmean)
    assert (cache / "dinov2_vits14__patchmean.npy").is_file()
    calls = model.calls
    second = extract_dataset(
        "dinov2_vits14",
        model,
        dataset,
        cache,
        kind="cls+patchmean",
        device="cpu",
        batch_size=4,
        num_workers=0,
    )
    assert second.cache_hit is True
    assert model.calls == calls
