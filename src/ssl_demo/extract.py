"""Extract frozen features and skip the work when the cache file is already there."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssl_demo.arrays import as_1d_labels, l2_normalize
from ssl_demo.config import default_num_workers


@dataclass
class ExtractResult:
    features: np.ndarray
    labels: np.ndarray
    index: np.ndarray
    patchmean: np.ndarray | None
    embed_dim: int
    n_params: int
    n_tiles: int
    seconds: float
    tiles_per_sec: float
    cache_hit: bool
    name: str


def feature_paths(cache_dir: str | Path, name: str) -> dict[str, Path]:
    cache_dir = Path(cache_dir)
    return {
        "features": cache_dir / f"{name}.npy",
        "patchmean": cache_dir / f"{name}__patchmean.npy",
        "labels": cache_dir / "labels.npy",
        "index": cache_dir / "index.npy",
    }


def is_cached(cache_dir: str | Path, name: str, kind: str = "pooled") -> bool:
    paths = feature_paths(cache_dir, name)
    if not paths["features"].is_file() or not paths["labels"].is_file():
        return False
    if kind == "cls+patchmean" and not paths["patchmean"].is_file():
        return False
    return True


def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array))
    temporary.replace(path)


def load_npy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def forward_embeddings(
    model: nn.Module,
    images: torch.Tensor,
    kind: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run one batch.

    ``kind='cls+patchmean'`` reads DINOv2 ``forward_features`` and returns the
    CLS token plus the mean of the patch tokens. Other kinds use ``forward``.
    """
    if kind == "cls+patchmean":
        features = model.forward_features(images)
        if not isinstance(features, dict):
            raise RuntimeError("Expected DINOv2 forward_features to return a dict.")
        cls = features["x_norm_clstoken"]
        patchmean = features["x_norm_patchtokens"].mean(dim=1)
        return cls, patchmean
    output = model(images)
    if isinstance(output, dict):
        output = output["x_norm_clstoken"]
    return output, None


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def extract_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str | None = None,
    kind: str = "pooled",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float]:
    """Run ``model`` under inference mode. Returns L2-normalised features, labels, patch means, seconds."""
    device = _resolve_device(device)
    model = model.to(device)
    model.eval()
    use_amp = device.type == "cuda"
    feature_chunks: list[np.ndarray] = []
    patch_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    start = time.perf_counter()
    for images, labels in tqdm(loader, desc="extract", leave=False):
        images = images.to(device, non_blocking=device.type == "cuda")
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                primary, extra = forward_embeddings(model, images, kind)
        feature_chunks.append(primary.float().cpu().numpy())
        label_chunks.append(as_1d_labels(labels.numpy()))
        if extra is not None:
            patch_chunks.append(extra.float().cpu().numpy())
    seconds = time.perf_counter() - start
    if not feature_chunks:
        raise RuntimeError("The loader was empty, so there are no features to write.")
    features = l2_normalize(np.concatenate(feature_chunks, axis=0))
    labels_out = np.concatenate(label_chunks, axis=0)
    patchmean = l2_normalize(np.concatenate(patch_chunks, axis=0)) if patch_chunks else None
    return features, labels_out, patchmean, seconds


def load_cached(cache_dir: str | Path, name: str) -> ExtractResult:
    """Load features written by :func:`extract_dataset`.

    Raises ``FileNotFoundError`` with a pointer to notebook 03 when the cache is cold.
    """
    paths = feature_paths(cache_dir, name)
    if not paths["features"].is_file() or not paths["labels"].is_file():
        raise FileNotFoundError(
            f"No cached features for {name!r} in {Path(cache_dir)}. "
            "Run notebook 03_backbone_zoo_feature_extraction first."
        )
    features = load_npy(paths["features"])
    labels = as_1d_labels(load_npy(paths["labels"]))
    if paths["index"].is_file():
        index = as_1d_labels(load_npy(paths["index"]))
    else:
        index = np.arange(len(labels), dtype=np.int64)
    patchmean = load_npy(paths["patchmean"]) if paths["patchmean"].is_file() else None
    if len(features) != len(labels):
        raise RuntimeError(
            f"Cache for {name} has {len(features)} features and {len(labels)} labels."
        )
    return ExtractResult(
        features=features,
        labels=labels,
        index=index,
        patchmean=patchmean,
        embed_dim=int(features.shape[1]),
        n_params=0,
        n_tiles=int(features.shape[0]),
        seconds=0.0,
        tiles_per_sec=float("nan"),
        cache_hit=True,
        name=name,
    )


def extract_dataset(
    name: str,
    model: nn.Module,
    dataset,
    cache_dir: str | Path,
    kind: str = "pooled",
    device: torch.device | str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> ExtractResult:
    """Extract ``dataset`` with ``model``, or return the cached arrays.

    Files land in ``cache_dir``:

    - ``{name}.npy`` primary embedding (pooled vector, or ViT CLS)
    - ``{name}__patchmean.npy`` mean of patch tokens when ``kind='cls+patchmean'``
    - ``labels.npy`` and ``index.npy`` aligned with those rows
    """
    cache_dir = Path(cache_dir)
    if is_cached(cache_dir, name, kind) and _cached_index_matches(cache_dir, name, dataset):
        loaded = load_cached(cache_dir, name)
        loaded.n_params = _safe_parameter_count(model)
        return loaded

    device = _resolve_device(device)
    if batch_size is None:
        batch_size = 256 if device.type == "cuda" else 32
    if num_workers is None:
        num_workers = default_num_workers()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features, labels, patchmean, seconds = extract_loader(model, loader, device=device, kind=kind)
    index = _dataset_index(dataset, len(labels))
    if len(index) != len(labels):
        raise RuntimeError(
            f"{name}: dataset index length {len(index)} does not match {len(labels)} labels."
        )
    paths = feature_paths(cache_dir, name)
    save_npy(paths["features"], features)
    save_npy(paths["labels"], labels)
    save_npy(paths["index"], index)
    if patchmean is not None:
        save_npy(paths["patchmean"], patchmean)
    n_tiles = int(features.shape[0])
    return ExtractResult(
        features=features,
        labels=labels,
        index=index,
        patchmean=patchmean,
        embed_dim=int(features.shape[1]),
        n_params=_safe_parameter_count(model),
        n_tiles=n_tiles,
        seconds=float(seconds),
        tiles_per_sec=float(n_tiles / seconds) if seconds > 0 else float("inf"),
        cache_hit=False,
        name=name,
    )


def result_row(result: ExtractResult, dataset: str, split: str) -> dict:
    """One throughput row for ``results/throughput.csv``."""
    return {
        "dataset": dataset,
        "split": split,
        "backbone": result.name,
        "embed_dim": result.embed_dim,
        "n_params": result.n_params,
        "n_tiles": result.n_tiles,
        "seconds": result.seconds,
        "tiles_per_sec": result.tiles_per_sec,
        "cache_hit": result.cache_hit,
    }


def _cached_index_matches(cache_dir: Path, name: str, dataset) -> bool:
    """A cache written for a different tile list (fast vs full, or a new seed) is a miss."""
    index_path = feature_paths(cache_dir, name)["index"]
    if not index_path.is_file():
        return False
    current = getattr(dataset, "indices", None)
    if current is None:
        return True
    saved = as_1d_labels(load_npy(index_path))
    return np.array_equal(saved, as_1d_labels(current))


def _dataset_index(dataset, n_rows: int) -> np.ndarray:
    indices = getattr(dataset, "indices", None)
    if indices is None:
        return np.arange(n_rows, dtype=np.int64)
    return as_1d_labels(indices)


def _safe_parameter_count(model: nn.Module) -> int:
    try:
        return sum(parameter.numel() for parameter in model.parameters())
    except Exception:
        return 0
