# Self-supervised features for pathology tiles

A small, public-data version of a loop used in preclinical toxicology histology: **tiles, frozen backbone, cached features, the same head, then compare**.

Labels on whole-slide images are slow. This demo asks whether a backbone can learn what tissue looks like without those labels, and how to tell whether a new backbone is actually better when the head stays fixed. It is a workshop demo, not a foundation-model training run and not a leaderboard entry.

This repository currently holds the minimum demo:

| Notebook | What it does | Fast-mode budget |
|---|---|---|
| [00_setup_and_data.ipynb](notebooks/00_setup_and_data.ipynb) | Drive cache, PathMNIST subset, class grid, augmentations | ~10 min |
| [03_backbone_zoo_feature_extraction.ipynb](notebooks/03_backbone_zoo_feature_extraction.ipynb) | Random ResNet-50, ImageNet ResNet-50, DINOv2 ViT-S/14 | ~20 min, then cached |
| [04_knn_linear_probe_fewshot.ipynb](notebooks/04_knn_linear_probe_fewshot.ipynb) | kNN, linear probe, 1/10/100% labels | ~10 min on cached features |
| [06_anomaly_detection_on_embeddings.ipynb](notebooks/06_anomaly_detection_on_embeddings.ipynb) | Normal-only heads, backbone swap, false-positive gallery | ~15 min on cached features |

[![00 setup](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/olonok69/SSL_pathology_demo/blob/main/notebooks/00_setup_and_data.ipynb)
[![03 features](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/olonok69/SSL_pathology_demo/blob/main/notebooks/03_backbone_zoo_feature_extraction.ipynb)
[![04 probes](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/olonok69/SSL_pathology_demo/blob/main/notebooks/04_knn_linear_probe_fewshot.ipynb)
[![06 anomaly](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/olonok69/SSL_pathology_demo/blob/main/notebooks/06_anomaly_detection_on_embeddings.ipynb)

Notebooks 01 (BYOL), 02 (mini DINO), 05 (attention vs Grad-CAM) and 07 (OpenSlide) are the later parts of the same spec. They are not in this slice.

## Run it

Open the notebooks from this repo on a Colab T4. Set `MODE = "fast"` in the first cell. Fast mode is a fixed PathMNIST subset (10k train / 2k test) and must finish in under 45 minutes. `MODE = "full"` raises that to 50k / 7k and is the place for NCT-CRC-HE-100K and a PatchCamelyon subset.

Datasets, checkpoints and `.npy` features go to `/content/drive/MyDrive/ssl-demo/` on Colab, or `./cache/` locally. Features are stored at `features/<dataset>/<mode>/<split>/<backbone>.npy`, so a fast run and a full run do not reuse each other's arrays. Every expensive step skips its work when that file is already there and the saved tile index still matches. Seeds are fixed (`SEED = 42` in `ssl_demo.config`). Tables are written to the cache `results/` folder and mirrored to [`results/`](results/).

```bash
uv sync
uv run pytest
```

`uv sync` creates `.venv`, installs this repo, and installs CPU PyTorch from the `cpu` dependency group in `pyproject.toml`. Point the notebook kernel at that `.venv`. `pytest` stays on CPU and uses toy tensors. It does not download PathMNIST, ImageNet weights, or DINOv2.

On Colab, install with `pip install -e .` from the setup cell. Torch is not a project dependency, so a preinstalled CUDA wheel stays in place. [`requirements.txt`](requirements.txt) is the pinned pip fallback for a local CPU stack. Do not pip-install that file, and do not `uv sync`, over a working Colab GPU runtime.

## Results

No measured table is checked in yet. After a real run, notebook 04 writes `results/probe_nct.csv` and notebook 06 writes `results/anomaly.csv`. The `dataset` column says whether a row came from PathMNIST or from the NCT-CRC folder. The file name `probe_nct.csv` is the one the later figure script expects. Empty cells below are intentional.

| Backbone | kNN accuracy | Linear probe accuracy | Task A AUROC (kNN distance) |
|---|---|---|---|
| `random_resnet50` | | | |
| `sup_resnet50_in1k` | | | |
| `dinov2_vits14` | | | |

Sanity check, not a target: ImageNet features should already be strong on these tiles, and a random network should sit clearly lower. If it does not, normalisation or the cache is wrong.

## Datasets and licences

| Data | Role here | Licence |
|---|---|---|
| [PathMNIST](https://medmnist.com/) (MedMNIST), from Kather NCT-CRC | Fast mode, and the default for notebooks 03, 04 and 06 | CC BY 4.0 |
| [NCT-CRC-HE-100K and CRC-VAL-HE-7K](https://zenodo.org/records/1214456) | Full mode, when the archive is already extracted. Train and val patients differ | CC BY 4.0 |
| [PatchCamelyon](https://github.com/basveeling/pcam) | Task B, only if `INCLUDE_PCAM` is set. Tumour patches are the anomaly | CC0 |
| OpenSlide `CMU-1-Small-Region.svs` | Not used in this slice (notebook 07) | Public test data |
| Phikon (`owkin/phikon`) | Not downloaded in this slice. Non-commercial licence: do not treat it as a free clinical model | Owkin non-commercial |
| UNI (`MahmoodLab/UNI`) | Not included. Gated weights, no token is stored here | Gated access |

Background tiles are kept in the classification probe and dropped from the anomaly fit. They are reported as a separate flag rate, because empty glass is easy to call "anomalous" and that is not a tumour.

Task A fits the head on non-tumour, non-background train tiles and scores colorectal adenocarcinoma epithelium as the anomaly. The test split is patient-disjoint from the train split in the Kather release. Task B, when the PCam cache exists, fits on normal lymph-node patches and scores metastasis.

## Models in this slice

| Name | Embedding | Licence |
|---|---|---|
| `random_resnet50` | 2048, global pool, no pretrained weights | torchvision (BSD-3-Clause). Random init is the floor |
| `sup_resnet50_in1k` | 2048, global pool | torchvision `ResNet50_Weights.IMAGENET1K_V1`. Checkpoint is downloaded, not vendored |
| `dinov2_vits14` | 384 CLS, plus a mean of patch tokens saved beside it | [DINOv2](https://github.com/facebookresearch/dinov2), Apache-2.0, via `torch.hub` |

Each model is normalised with its own mean and standard deviation (ImageNet stats for all three here; the supervised checkpoint's transform is what the code reads). ViT features used downstream are the CLS token. Features are L2-normalised before kNN. Extraction uses `torch.inference_mode` and CUDA autocast when a GPU is present.

DINOv2 was not trained on NCT-CRC. Still ask, before trusting a number, whether the evaluation tiles were in the pretraining set. That question is the same one to ask of any later pathology foundation model. Phikon, when it is added, was trained on TCGA with iBOT and is **non-commercial**.

## Anomaly heads

Same splits, same seeds, three heads, fit on normal tiles only:

- kNN cosine distance to a normal memory bank (image-level PatchCore pattern), with a seeded coreset cap
- Gaussian Mahalanobis distance with shrinkage covariance
- Isolation Forest

The threshold is the 99th percentile of scores on held-out normal tiles, chosen without looking at tumours. Metrics are AUROC, AUPRC, FPR at 95% TPR, and sensitivity / false-positive rate at that threshold, with a bootstrap interval over test tiles.

The decision rule is written down in `ssl_demo.anomaly.DECISION_RULE` and printed in notebook 06 before the fit: a backbone is clearly better only when its kNN-distance AUROC interval does not overlap the next best on **both** tasks.

## Credits

- Kather et al., NCT-CRC-HE-100K, and MedMNIST / PathMNIST
- Chen and He, BYOL; Caron et al., DINO; Oquab et al., DINOv2
- lightly, torchvision, timm, scikit-learn, umap-learn
- PatchCamelyon / CAMELYON16
- Juan's own teaching repos this workshop follows later: `olonok69/BYOL`, `olonok69/graimatter_ML`, `olonok69/Anomaly_detection`, `olonok69/pytorch-CNN`

Nothing in this repo is client data, a UK Biobank image, an EyePACS export, or an API key. Hugging Face tokens, if a later notebook needs one, stay in Colab secrets.
