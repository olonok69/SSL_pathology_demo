# The Demos, Explained: Notebooks, Live Cells and Fallbacks

**What each notebook does, what to run live, and what to do when something fails**

This guide follows the notebooks, not the slides. The [presenter guide](Presenter_Guide_EN.md) follows the deck and says where each notebook appears in the talk. The Spanish version is [Guia_Demos_ES.md](Guia_Demos_ES.md), with the same section numbers.

---

## 0. Before the session

### 0.1 Pre-flight checklist

* **Drive cache warm.** `MyDrive/ssl-demo/` holds `features/`, `indices/` and `results/` from a previous run. With the cache in place, notebooks 04 and 06 skip every forward pass.
* **GPU attached** in Colab (Runtime, Change runtime type, T4). Only notebooks 00 and 03 need it; 04 and 06 run on CPU from the cache.
* **Notebooks pre-run** with outputs saved, so every figure can be shown even if the network fails.
* **Browser zoom** at 125% or more for a projector; collapse the Colab file panel.
* **Open tabs**: notebook 04, notebook 06, the deck, `results/anomaly_centring.csv`.
* **No tokens needed** in v1. If a later notebook needs a Hugging Face token, it goes in Colab secrets (`google.colab.userdata`), never in a cell.

### 0.2 Run order and budget

| Notebook | What it does | First run (T4, fast mode) | With cache |
|---|---|---|---|
| `00_setup_and_data` | Mounts Drive, downloads PathMNIST, NCT-CRC and optionally PatchCamelyon, builds index CSVs, shows the class grid and augmentations | ~10 min, plus download time | ~1 min |
| `03_backbone_zoo_feature_extraction` | Extracts the three backbones for every prepared dataset; writes `throughput.csv` | ~20 min | seconds (all cache hits) |
| `04_knn_linear_probe_fewshot` | kNN, linear probe at 1/10/100% labels, confusion matrices, UMAP; writes `probe_nct.csv` | ~10 min | same; it only reads features |
| `06_anomaly_detection_on_embeddings` | Decision rule, three heads on Task A and Task B, raw against centred cosine, decision, gallery; writes `anomaly.csv` and `anomaly_centring.csv` | ~3 min | same |

Set `MODE = "fast"` in the first code cell of every notebook. Fast mode is the workshop setting: 10,000 training and 2,000 test tiles, seed 42.

### 0.3 Running 04 and 06 on a laptop

Notebooks 04 and 06 need only the cached features, not images or a GPU. Copy `MyDrive/ssl-demo/features` and `MyDrive/ssl-demo/indices` to `./cache/` in the repository, then:

```bash
uv sync
```

Open the notebooks in VS Code, or any Jupyter front end, with the `.venv` kernel. Skip the Colab clone cell at the top.

On a laptop CPU, notebook 06 takes about two minutes; notebook 04 about five, most of it the cross-validated probe at 100% labels. The false-positive gallery in 06 is skipped when the NCT-CRC images are not on disk; everything else runs.

### 0.4 After a code change

The committed notebook outputs are the record of the Colab runs. When code in `src/ssl_demo/` changes, re-run 04 and 06 on Colab and commit the notebooks with their outputs, together with the CSVs in `results/`. Then regenerate the figures and decks:

```bash
uv sync --group docs
uv run python docs/guide/make_figures.py
uv run python docs/make_deck.py
```

---

## 1. Notebook 00: setup and data

**Slide.** 5 (the class grid in the deck is drawn by `make_figures.py` from NCT-CRC tiles).

What it prints, in order:

1. The cache root (`/content/drive/MyDrive/ssl-demo` on Colab, `./cache` locally) and the mode.
2. The PathMNIST download and a check that its label order matches `medmnist.INFO`. The notebook raises an error rather than guess which class is tumour.
3. Class balance of the fixed subset, and a grid of tiles per class.
4. Augmentation previews: random resized crop, colour jitter, blur, solarisation, and the H&E stain jitter in `ssl_demo.data`. These are the views a self-supervised method would compare; use them on slide 8 or 9 if you want a real-data picture of "two views of the same image".
5. Optional NCT-CRC download to Drive and extraction to `/content/nct`, and optional PatchCamelyon (`INCLUDE_PCAM`, about 7 GB).

**Watch out.** `INCLUDE_PCAM` is set early in the notebook and again in the NCT cell. If PatchCamelyon is not downloaded, Task B is skipped in notebook 06 and the two-task rule cannot name a winner.

**Live?** No. Show the class grid and augmentation previews from saved outputs.

---

## 2. Notebook 03: frozen feature extraction

**Slide.** 3 (the loop) and 24 (engineering).

What it does:

1. Loads each backbone from `ssl_demo.backbones` with its own preprocessing: `random_resnet50` (no weights), `sup_resnet50_in1k` (torchvision ImageNet), `dinov2_vits14` (`torch.hub`, `trust_repo=True`).
2. For each dataset with an index (PathMNIST, NCT-CRC, PatchCamelyon), extracts features under `torch.inference_mode` with fp16 autocast on GPU, L2-normalises them, and saves `features/<dataset>/<mode>/<split>/<backbone>.npy`, plus `labels.npy` and `index.npy`. DINOv2 also saves the mean of its patch tokens as `dinov2_vits14__patchmean.npy`.
3. Skips any backbone whose file exists and whose saved `index.npy` matches the current tile list.
4. Writes `results/throughput.csv`.

**What to say about throughput.** It is not a speed ranking. The first backbone on each dataset paid for cold reads from Drive; the later ones read warm files. If you want a speed comparison, copy the data to local `/content` disk first and add a warm-up batch.

**Live?** No: 20 minutes. Show the output table and say the second run takes seconds.

---

## 3. Notebook 04: kNN, linear probe, label efficiency

**Slides.** 13 and 14.

What it prints:

1. Which dataset it probes: `nct` when NCT-CRC features are cached, `pathmnist` otherwise.
2. For each backbone, a kNN classifier (cosine, k = 20, temperature-weighted vote) and a linear probe at 1%, 10% and 100% of the training labels, each with accuracy, balanced accuracy and macro F1 and a 1,000-resample bootstrap interval.
3. The sanity line from `random_floor_warning`: it warns if the untrained backbone's kNN accuracy is within five points of a trained one.
4. Confusion matrices (all labels), the label-efficiency plot, and a UMAP of each backbone's test features.

**How the probe is fitted.** `fit_linear_probe` standardises the features (the scaler is fit on the training rows only) and chooses C from {0.001, 0.01, 0.1, 1} by stratified 3-fold cross-validation on the same training rows. Test tiles never influence C. Without the scaler, the untrained network's nearly parallel features made the probe collapse to one class at 10% of the labels.

**Numbers on screen** (`results/probe_nct.csv`, accuracy): kNN DINOv2 0.890, ImageNet 0.841, random 0.616. Probe at 1%: 0.837, 0.750, 0.532. Probe at 100%: 0.913, 0.903, 0.688.

**Talking points in the outputs.**

* Confusion matrices: for both pretrained backbones the largest confusion is between cancer-associated stroma and smooth muscle, in both directions. Both are spindle-cell tissue, and pathologists also find them close at this scale. Tumour epithelium recall with all labels is 0.86 for both.
* UMAP: a two-dimensional picture of the features, coloured by class. It is a picture, not a metric; say so, and read the numbers from the table.

**Live?** Show saved outputs. Re-running takes several minutes because of the cross-validated probe.

---

## 4. Notebook 06: anomaly detection and the decision

**Slides.** 16 to 23. This is the live demo.

### 4.1 The decision rule

The imports cell prints `DECISION_RULE` before anything is fitted, then the normal classes, the anomaly class and the class held out of the fit (background). Read the rule aloud. This is slide 17.

### 4.2 Task A

The next cell picks the dataset (NCT-CRC when cached), and for each backbone:

1. Takes the normal pool from the training split: the seven non-tumour, non-background classes (7,511 tiles).
2. Splits it 80/20 with seed 42: 6,009 tiles to fit the head, 1,502 held out to set the threshold.
3. Builds the test set: normal and tumour tiles from the test split (1,764 tiles, 344 tumour). Background is scored separately for its flag rate.
4. Runs `run_anomaly_task`, which fits each head on the 6,009, sets the threshold at the 99th percentile of the 1,502 held-out scores, and scores the test set.

It prints the pool and test sizes per backbone.

### 4.3 Raw against centred cosine (the live cell)

The cell under "Raw versus centred cosine" fits the kNN head twice on the same 6,009 tiles, once with `center=False` and once with `center=True`, and prints:

| backbone | centroid_cosine | auroc_raw_cosine | auroc_centred_cosine |
|---|---|---|---|
| random_resnet50 | 1.000 | 0.885 | 0.492 |
| sup_resnet50_in1k | 0.820 | 0.672 | 0.902 |
| dinov2_vits14 | 0.817 | 0.668 | 0.937 |

It also saves the table to `results/anomaly_centring.csv`. It takes seconds on cached features, so run it live during slides 18 to 20. What to point at:

* `centroid_cosine` of 1.000 for the random network: every vector points the same way.
* The two AUROC columns cross: the random network falls from first to chance, the pretrained backbones rise above 0.9.

### 4.4 Task B, the table and the decision

The next cell runs the same procedure on PatchCamelyon (2,000 normal patches to fit, 500 held out, 2,000 test) if its features are cached, concatenates all rows and writes `results/anomaly.csv`.

The cell after prints an AUROC pivot for every task and head, then the decision:

> No backbone is clearly better on both tasks. task_a: dinov2_vits14 AUROC 0.937 [0.925, 0.948] is clear of sup_resnet50_in1k [0.883, 0.919]. task_b: sup_resnet50_in1k leads on the point estimate (0.759) but its CI overlaps dinov2_vits14.

Show it on slide 22. If the untrained backbone ever leads a task, or sits within 0.02 of the best trained one, `decide_backbone` refuses and prints a "Pipeline check" message instead of a winner.

**The other heads.** The pivot also shows Mahalanobis and Isolation Forest. Both rank the untrained network first on both tasks, and Isolation Forest is below 0.5 for the pretrained backbones. They are not tuned in v1. If asked, say so, and use it as evidence that the head is part of the comparison.

### 4.5 Backbone-swap plot

A bar chart of kNN-distance AUROC per backbone and task with intervals: the notebook version of the deck's figure on slide 22.

### 4.6 False-positive gallery

Normal test tiles with the highest kNN-distance scores, six per backbone. Debris, mucus, folds and tile edges dominate. This is slide 23's "unusual, not malignant". It needs the tile images: on Colab, NCT-CRC extracted to `/content/nct` (or set `NCT_ROOT`); otherwise the cell prints that the gallery is skipped.

---

## 5. Fallbacks

| Problem | What to do |
|---|---|
| Colab is down or slow | Present from the saved notebook outputs and the deck; both tell the full story. Run the centring cell locally (section 0.3) if you have the cache on the laptop. |
| Drive will not mount | Use the local copy of the cache. If there is none, show saved outputs only. |
| DINOv2 download fails in 03 | Only matters on a cold cache. Skip that backbone row and say why; the cached features are what 04 and 06 use. |
| PatchCamelyon missing | Task B is skipped and the rule says it cannot declare a cross-task winner. Present Task A and explain the two-task rule with the saved Task B numbers from the deck. |
| Gallery skipped | Expected when the NCT-CRC images are not extracted in the session. Show the saved gallery output. |
| Numbers differ from the deck | Check `MODE`, the seed, and that `indices/*.csv` on Drive were not rebuilt with a different subset size. The cache key includes the tile list, so a changed subset re-extracts features. |

---

## 6. Reproducibility and known caveats

* Seeds are fixed (`SEED = 42` in `ssl_demo.config`). Bootstrap intervals use the same seed, so they are identical between runs on the same features.
* The fast-mode numbers in the deck come from features extracted on Colab GPUs (L4 for NCT-CRC and PatchCamelyon). Re-extracting on another GPU can move the third decimal because of fp16 autocast.
* The throughput file is I/O-bound and is not a model-speed comparison.
* The Mahalanobis and Isolation Forest heads are untuned.
* PatchCamelyon labels refer to the central 32 x 32 pixels of a 96 x 96 patch, upscaled to 224; a global embedding dilutes a small tumour focus. Centre-cropping before resizing is a v2 experiment.
* Nothing in the repository is clinical data. No result here is a clinical claim.
