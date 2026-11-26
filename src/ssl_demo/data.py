"""Public histology tiles, fixed subsets, and the augmentations the later notebooks show.

Fast mode uses PathMNIST (CC BY 4.0), the MedMNIST release of Kather's NCT-CRC tiles.
The test split is CRC-VAL-HE-7K, which comes from different patients than the train set.
NCT-CRC-HE-100K itself is a full-mode folder layout (Zenodo record 1214456). PatchCamelyon
is the second anomaly task and is off until its index CSV is in the cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ssl_demo.arrays import as_1d_labels
from ssl_demo.config import SEED, cache_root, subset_count

CLASS_NAMES = [
    "adipose",
    "background",
    "debris",
    "lymphocytes",
    "mucus",
    "smooth muscle",
    "normal colon mucosa",
    "cancer-associated stroma",
    "colorectal adenocarcinoma epithelium",
]
TUMOUR_ID = 8
BACKGROUND_ID = 1
NORMAL_IDS = (0, 2, 3, 4, 5, 6, 7)

PCAM_CLASS_NAMES = ["normal lymph node", "metastasis"]
PCAM_NORMAL_ID = 0
PCAM_TUMOUR_ID = 1

NCT_CLASS_DIRS = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Ruifrok stain matrix, rows are H, E and DAB in RGB. Used only for the H&E jitter.
_RGB_FROM_HED = np.array(
    [
        [0.650, 0.704, 0.286],
        [0.072, 0.990, 0.105],
        [0.268, 0.570, 0.776],
    ],
    dtype=np.float64,
)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED)


def rgb_to_hed(rgb: np.ndarray) -> np.ndarray:
    optical_density = -np.log(np.clip(rgb, 1e-6, 1.0))
    return optical_density @ _HED_FROM_RGB


def hed_to_rgb(hed: np.ndarray) -> np.ndarray:
    optical_density = hed @ _RGB_FROM_HED
    return np.clip(np.exp(-optical_density), 0.0, 1.0)


def stain_jitter_array(
    rgb: np.ndarray,
    sigma: float = 0.15,
    seed: int | None = None,
) -> np.ndarray:
    """Scale H and E concentrations. ``sigma=0`` is a round-trip through HED space."""
    hed = rgb_to_hed(np.asarray(rgb, dtype=np.float64))
    if sigma == 0:
        scales = np.ones(3, dtype=np.float64)
    else:
        rng = np.random.default_rng(seed)
        scales = np.array(
            [
                1.0 + rng.uniform(-sigma, sigma),
                1.0 + rng.uniform(-sigma, sigma),
                1.0 + rng.uniform(-0.1 * sigma, 0.1 * sigma),
            ],
            dtype=np.float64,
        )
    return hed_to_rgb(hed * scales).astype(np.float32)


class StainJitter:
    """H&E stain jitter in HED space. ImageNet colour jitter is a different transform.

    Haematoxylin and eosin are dyes, not natural-image colours. Scaling them in
    HED space imitates scanner and lab variation. A strong RGB colour jitter can
    paint the tissue green.
    """

    def __init__(self, sigma: float = 0.15, seed: int | None = None):
        self.sigma = sigma
        self.seed = seed

    def __call__(self, image):
        array, kind = _to_float_hwc(image)
        jittered = stain_jitter_array(array, sigma=self.sigma, seed=self.seed)
        return _from_float_hwc(jittered, kind)


def augmentation_previews(image: Image.Image, seed: int = SEED) -> dict[str, Image.Image]:
    """One draw of the crops and colour transforms BYOL and DINO use, plus stain jitter."""
    from torchvision import transforms as T

    image = image.convert("RGB")
    torch.manual_seed(seed)
    side = min(image.size)
    crop = T.RandomResizedCrop(
        side,
        scale=(0.4, 1.0),
        interpolation=T.InterpolationMode.BICUBIC,
    )
    return {
        "random_crop": crop(image),
        "colour_jitter": T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)(image),
        "blur": T.GaussianBlur(kernel_size=23 if side >= 23 else 3, sigma=1.5)(image),
        "solarize": T.RandomSolarize(threshold=128, p=1.0)(image),
        "stain_jitter": StainJitter(sigma=0.2, seed=seed)(image),
    }


def stratified_indices(labels: np.ndarray, n: int, seed: int = SEED) -> np.ndarray:
    """Sample ``n`` indices, proportional to class size, without replacement."""
    labels = as_1d_labels(labels)
    if n < 0:
        raise ValueError("n must be non-negative")
    pools = {int(class_id): np.flatnonzero(labels == class_id) for class_id in np.unique(labels)}
    total = int(sum(len(pool) for pool in pools.values()))
    if total == 0 or n == 0:
        return np.zeros(0, dtype=np.int64)
    n = min(int(n), total)
    raw = {class_id: n * len(pool) / total for class_id, pool in pools.items()}
    quota = {class_id: int(np.floor(value)) for class_id, value in raw.items()}
    leftover = n - sum(quota.values())
    order = sorted(pools, key=lambda class_id: (raw[class_id] - quota[class_id], -class_id), reverse=True)
    for class_id in order:
        if leftover <= 0:
            break
        if quota[class_id] < len(pools[class_id]):
            quota[class_id] += 1
            leftover -= 1
    rng = np.random.default_rng(seed)
    chosen = []
    for class_id in sorted(pools):
        take = min(quota[class_id], len(pools[class_id]))
        if take <= 0:
            continue
        pick = rng.choice(pools[class_id], size=take, replace=False)
        chosen.append(np.asarray(pick, dtype=np.int64))
    indices = np.concatenate(chosen)
    rng.shuffle(indices)
    return indices.astype(np.int64, copy=False)


def per_class_positions(labels: np.ndarray, per_class: int, seed: int = SEED) -> dict[int, np.ndarray]:
    """Positions inside ``labels`` (already a subset), up to ``per_class`` per class."""
    labels = as_1d_labels(labels)
    rng = np.random.default_rng(seed)
    positions: dict[int, np.ndarray] = {}
    for class_id in np.unique(labels):
        pool = np.flatnonzero(labels == class_id)
        take = min(int(per_class), len(pool))
        positions[int(class_id)] = rng.choice(pool, size=take, replace=False)
    return positions


def class_balance(labels: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    labels = as_1d_labels(labels)
    rows = []
    for class_id, name in enumerate(class_names):
        rows.append({"id": class_id, "class": name, "n": int(np.sum(labels == class_id))})
    return pd.DataFrame(rows)


def index_csv_path(cache: str | Path, dataset: str, split: str, mode: str) -> Path:
    return Path(cache) / "indices" / f"{dataset}_{split}_{mode}.csv"


def feature_dir(cache: str | Path, dataset: str, split: str, mode: str) -> Path:
    """Feature cache for one subset. Fast and full never share a folder."""
    mode = mode.lower()
    if mode not in ("fast", "full"):
        raise ValueError(f"MODE must be 'fast' or 'full', got {mode!r}")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    return Path(cache) / "features" / dataset / mode / split


def build_or_load_index(
    labels: np.ndarray,
    n: int,
    seed: int,
    csv_path: str | Path,
) -> np.ndarray:
    """Return the saved subset if the CSV exists, otherwise sample and write it.

    Later notebooks must see the same tiles, so an existing file wins over ``n``.
    """
    labels = as_1d_labels(labels)
    csv_path = Path(csv_path)
    if csv_path.is_file():
        saved = pd.read_csv(csv_path)
        indices = saved["index"].to_numpy(dtype=np.int64)
        if len(indices) == 0 or indices.min() < 0 or indices.max() >= len(labels):
            raise ValueError(f"Index file {csv_path} does not match a label vector of length {len(labels)}")
        return indices
    indices = stratified_indices(labels, n, seed)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": indices, "label": labels[indices]}).to_csv(csv_path, index=False)
    return indices


class IndexedTiles(Dataset):
    """Subset of a tile collection, with that backbone's preprocess applied."""

    def __init__(self, get_image: Callable[[int], Image.Image], labels, indices, transform):
        self.get_image = get_image
        self.labels = as_1d_labels(labels)
        self.indices = as_1d_labels(indices)
        self.transform = transform

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, position: int):
        source = int(self.indices[position])
        image = self.get_image(source)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[source])


@dataclass
class TileSource:
    dataset_name: str
    split: str
    mode: str
    indices: np.ndarray
    labels: np.ndarray
    class_names: list[str]
    tumour_id: int
    background_id: int | None
    get_image: Callable[[int], Image.Image]
    index_csv: Path

    def subset_labels(self) -> np.ndarray:
        return as_1d_labels(self.labels)[as_1d_labels(self.indices)]

    def normal_ids(self) -> tuple[int, ...]:
        skip = {self.tumour_id}
        if self.background_id is not None:
            skip.add(self.background_id)
        return tuple(class_id for class_id in range(len(self.class_names)) if class_id not in skip)

    def dataset(self, transform) -> IndexedTiles:
        return IndexedTiles(self.get_image, self.labels, self.indices, transform)

    def image(self, source_index: int) -> Image.Image:
        return self.get_image(int(source_index))


def prepare_pathmnist(
    mode: str = "fast",
    cache: str | Path | None = None,
    seed: int = SEED,
    size: int = 224,
    download: bool = True,
) -> dict[str, TileSource]:
    """Download PathMNIST if needed and build the fixed train and test index CSVs."""
    cache = cache_root(cache)
    _check_pathmnist_label_order()
    root = cache / "data" / "medmnist"
    root.mkdir(parents=True, exist_ok=True)
    sources = {}
    for split in ("train", "test"):
        opened = _open_pathmnist(root, split, size, download=download)
        labels = as_1d_labels(opened.labels)
        n = subset_count(mode, split, "pathmnist")
        csv_path = index_csv_path(cache, "pathmnist", split, mode)
        indices = build_or_load_index(labels, n, seed, csv_path)
        sources[split] = TileSource(
            dataset_name="pathmnist",
            split=split,
            mode=mode,
            indices=indices,
            labels=labels,
            class_names=list(CLASS_NAMES),
            tumour_id=TUMOUR_ID,
            background_id=BACKGROUND_ID,
            get_image=MedMNISTSource(root, split, size),
            index_csv=csv_path,
        )
    return sources


class MedMNISTSource:
    """Re-open the memmap in each DataLoader worker instead of pickling it."""

    def __init__(self, root: str | Path, split: str, size: int):
        self.root = str(root)
        self.split = split
        self.size = int(size)
        self._dataset = None

    def __call__(self, index: int) -> Image.Image:
        image, _ = self._open()[int(index)]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        return image.convert("RGB")

    def __getstate__(self):
        return {"root": self.root, "split": self.split, "size": self.size}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._dataset = None

    def _open(self):
        if self._dataset is None:
            self._dataset = _open_pathmnist(self.root, self.split, self.size, download=False)
        return self._dataset


def prepare_pcam(
    mode: str = "full",
    cache: str | Path | None = None,
    seed: int = SEED,
    download: bool = False,
    root: str | Path | None = None,
) -> dict[str, TileSource]:
    """Subset of PatchCamelyon. ``download=True`` fetches the full h5 files (several GB)."""
    cache = cache_root(cache)
    data_root = Path(root) if root is not None else cache / "data"
    if download:
        from torchvision.datasets import PCAM

        for split in ("train", "test"):
            PCAM(root=str(data_root), split=split, download=True)
    if not pcam_is_downloaded(data_root):
        raise FileNotFoundError(
            "PatchCamelyon is not in the cache. It is about 7 GB. "
            "Call prepare_pcam(..., download=True) or set INCLUDE_PCAM in notebook 00. "
            "Task A on PathMNIST does not need it."
        )
    sources = {}
    for split in ("train", "test"):
        labels = pcam_labels(data_root, split)
        n = subset_count(mode, split, "pcam")
        csv_path = index_csv_path(cache, "pcam", split, mode)
        indices = build_or_load_index(labels, n, seed, csv_path)
        sources[split] = TileSource(
            dataset_name="pcam",
            split=split,
            mode=mode,
            indices=indices,
            labels=labels,
            class_names=list(PCAM_CLASS_NAMES),
            tumour_id=PCAM_TUMOUR_ID,
            background_id=None,
            get_image=PCamSource(data_root, split),
            index_csv=csv_path,
        )
    return sources


def pcam_is_downloaded(root: str | Path) -> bool:
    from torchvision.datasets import PCAM

    folder = Path(root) / "pcam"
    if not folder.is_dir():
        return False
    for split in ("train", "test"):
        images = PCAM._FILES[split]["images"][0]
        targets = PCAM._FILES[split]["targets"][0]
        if not (folder / images).is_file() or not (folder / targets).is_file():
            return False
    return True


def pcam_labels(root: str | Path, split: str) -> np.ndarray:
    import h5py
    from torchvision.datasets import PCAM

    filename = PCAM._FILES[split]["targets"][0]
    path = Path(root) / "pcam" / filename
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["y"][:, 0, 0, 0], dtype=np.int64)


class PCamSource:
    def __init__(self, root: str | Path, split: str):
        self.root = str(root)
        self.split = split
        self._file = None
        self._images = None

    def __call__(self, index: int) -> Image.Image:
        array = np.asarray(self._open()[int(index)])
        return Image.fromarray(array).convert("RGB")

    def __getstate__(self):
        return {"root": self.root, "split": self.split}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._file = None
        self._images = None

    def _open(self):
        if self._images is None:
            import h5py
            from torchvision.datasets import PCAM

            filename = PCAM._FILES[self.split]["images"][0]
            self._file = h5py.File(Path(self.root) / "pcam" / filename, "r")
            self._images = self._file["x"]
        return self._images


def find_nct_directory(root: str | Path, split: str) -> Path | None:
    """Locate an extracted Kather folder. This does not download the 11 GB archive."""
    root = Path(root)
    if split == "train":
        names = ["NCT-CRC-HE-100K", "NCT-CRC-HE-100K-NONORM"]
    else:
        names = ["CRC-VAL-HE-7K", "CRC-VAL-HE-7K-NONORM"]
    candidates = [root / name for name in names] + [root]
    for candidate in candidates:
        if (candidate / "TUM").is_dir() and (candidate / "NORM").is_dir():
            return candidate
    return None


def prepare_nct(
    root: str | Path,
    mode: str = "full",
    cache: str | Path | None = None,
    seed: int = SEED,
) -> dict[str, TileSource]:
    """Build index CSVs from an already extracted NCT-CRC / CRC-VAL folder.

    Zenodo: https://zenodo.org/records/1214456 (CC BY 4.0). The archive is about 11 GB,
    so this function never downloads it.
    """
    cache = cache_root(cache)
    sources = {}
    for split in ("train", "test"):
        folder = find_nct_directory(root, split)
        if folder is None:
            raise FileNotFoundError(
                f"No NCT-CRC {split} folder under {root}. "
                "Download Zenodo record 1214456 and extract NCT-CRC-HE-100K and CRC-VAL-HE-7K."
            )
        paths, labels = scan_class_folders(folder, NCT_CLASS_DIRS)
        n = subset_count(mode, split, "nct")
        csv_path = index_csv_path(cache, "nct", split, mode)
        indices = build_or_load_index(labels, n, seed, csv_path)
        sources[split] = TileSource(
            dataset_name="nct",
            split=split,
            mode=mode,
            indices=indices,
            labels=labels,
            class_names=list(CLASS_NAMES),
            tumour_id=TUMOUR_ID,
            background_id=BACKGROUND_ID,
            get_image=FolderSource(paths),
            index_csv=csv_path,
        )
    return sources


def scan_class_folders(folder: str | Path, class_dirs: list[str]) -> tuple[list[str], np.ndarray]:
    folder = Path(folder)
    paths: list[str] = []
    labels: list[int] = []
    for class_id, name in enumerate(class_dirs):
        class_dir = folder / name
        if not class_dir.is_dir():
            raise FileNotFoundError(class_dir)
        files = sorted(
            path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        paths.extend(str(path) for path in files)
        labels.extend([class_id] * len(files))
    if not paths:
        raise FileNotFoundError(f"No tile images under {folder}")
    return paths, np.asarray(labels, dtype=np.int64)


class FolderSource:
    def __init__(self, paths: list[str]):
        self.paths = list(paths)

    def __call__(self, index: int) -> Image.Image:
        return Image.open(self.paths[int(index)]).convert("RGB")


def prepared_datasets(cache: str | Path, mode: str) -> list[str]:
    """Datasets whose index CSV is already in the cache, in notebook order."""
    names = []
    for dataset in ("pathmnist", "nct", "pcam"):
        if index_csv_path(cache, dataset, "train", mode).is_file():
            names.append(dataset)
    return names


def _open_pathmnist(root, split: str, size: int, download: bool):
    from medmnist import PathMNIST

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return PathMNIST(
        split=split,
        root=str(root),
        size=size,
        download=download,
        mmap_mode="r",
    )


def _check_pathmnist_label_order() -> None:
    from medmnist import INFO

    official = INFO["pathmnist"]["label"]
    for class_id, name in enumerate(CLASS_NAMES):
        if official[str(class_id)] != name:
            raise RuntimeError(
                f"PathMNIST label {class_id} is {official[str(class_id)]!r}, "
                f"but this package expected {name!r}. Refusing to guess the tumour class."
            )


def _to_float_hwc(image) -> tuple[np.ndarray, str]:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        return array, "pil"
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)
        kind = "chw"
    else:
        kind = "hwc"
    array = array.astype(np.float64)
    if array.max() > 1.5:
        array = array / 255.0
    return array, kind


def _from_float_hwc(array: np.ndarray, kind: str):
    if kind == "pil":
        pixels = np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8)
        return Image.fromarray(pixels, mode="RGB")
    if kind == "chw":
        return np.moveaxis(array, -1, 0).astype(np.float32)
    return array.astype(np.float32)
