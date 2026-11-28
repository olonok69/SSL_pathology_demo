# Handover: SSL pathology demo

Written 28 Nov 2026 for the next agent who picks this up. Owner: juan (GitHub `olonok69`).

- Local repo: `D:\repos3\BI_SSL_demo` (Windows)
- Remote: `https://github.com/olonok69/SSL_pathology_demo` (branch `main`, clean, up to date at commit `f7849a1`)
- Full spec: `ssl-pathology-demo-spec.md` in the repo root (same file as `/mnt/project-files/ssl-demo/ssl-pathology-demo-spec.md` in the project). Read it. This document only summarises it and says where things stand.

---

## 1. Goal in one paragraph

A small, public-data version of an FMAD-style loop for preclinical toxicology histology: **tiles → frozen backbone → cached features → the same head → compare backbones**. Two outputs: (1) Colab notebooks that compare self-supervised features (BYOL, DINO, DINOv2, a pathology FM such as Phikon) against supervised ImageNet features, with kNN / linear probe / few-shot evaluation and normal-only anomaly detection; (2) workshop and video material (presenter guide, demo guide, cheatsheet, figure script, generated deck) mirroring `olonok69/graimatter_ML/docs/`. Non-goals: SOTA, training a foundation model, multi-node anything.

Hard constraints from the spec: every notebook runs on a free Colab T4 in `MODE="fast"` in under 45 min; Google Drive (`/content/drive/MyDrive/ssl-demo/`) is the cache and every expensive step skips if cached; seeds fixed (`SEED=42`); every results table saved as CSV so figures come from files; public data only in anything shown or published.

## 2. Current state

Build order step 1 of the spec (the "minimum demo") is implemented and has been run once on Colab. Nothing from steps 2 to 5 exists yet.

| Spec item | Status |
|---|---|
| `src/ssl_demo/` `config, data, backbones, extract, evaluate, anomaly, viz, arrays` | Done |
| Tests (`tests/`, CPU, toy tensors) | Done. Last local run passed (`.pytest_cache/lastfailed` is empty, 26 Nov). Not re-run for this handover |
| 00 setup and data | Done, run on Colab |
| 03 backbone zoo (3 backbones: `random_resnet50`, `sup_resnet50_in1k`, `dinov2_vits14`) | Done, run on Colab (L4) on PathMNIST, PCam and NCT-CRC |
| 04 kNN / linear probe / few-shot | Done, run on NCT-CRC (L4) |
| 06 anomaly detection | Done, run on PathMNIST Task A + PCam Task B (T4). **Result is suspicious, see section 3** |
| 01 BYOL from scratch, 02 mini DINO, 05 attention vs Grad-CAM, 07 mini WSI | Not started |
| `byol.py`, `dino.py`, `wsi.py` | Not started |
| More backbones: `sup_vit_s16_in21k`, `dino_vits16`, `phikon`, (`uni`), `byol_resnet18_ours`, `dino_vit_tiny_ours` | Not started |
| `docs/` (guides, `make_figures.py`, `make_deck.py`) | Not started |
| README results table | Empty placeholder, still says "No measured table is checked in yet" |

## 3. Results so far

All in `results/` (also mirrored on Drive at `ssl-demo/results/`). Fast mode: 10k train / 2k test tiles (PCam: 5k / 2k), stratified, seed 42.

### 3.1 Probe on NCT-CRC (`results/probe_nct.csv`)

Train = 10k tiles from NCT-CRC-HE-100K, test = 2k from CRC-VAL-HE-7K (patient-disjoint). Accuracy, 95% bootstrap CI in brackets.

| Backbone | kNN (k=20) | Probe 1% | Probe 10% | Probe 100% |
|---|---|---|---|---|
| `random_resnet50` | 0.617 | 0.203 | 0.172 | 0.489 |
| `sup_resnet50_in1k` | 0.841 | 0.670 | 0.821 | 0.871 [0.856, 0.884] |
| `dinov2_vits14` | **0.890** | **0.724** | **0.884** | **0.916** [0.903, 0.927] |

This looks right: DINOv2 > ImageNet supervised > random, and the label-efficiency gap is biggest at 1%. One oddity: the random backbone's probe at 10% predicts a single class (balanced accuracy exactly 1/9). See issue 4.3.

### 3.2 Anomaly detection (`results/anomaly.csv`)

kNN-distance head AUROC, 95% CI:

| Backbone | Task A (PathMNIST, TUM vs non-tumour) | Task B (PCam, metastasis vs normal) | Background flag rate (Task A) | FPR at the 99th-pct threshold (Task A) |
|---|---|---|---|---|
| `random_resnet50` | **0.898** [0.881, 0.914] | **0.716** [0.694, 0.737] | 0.62 | 0.004 |
| `sup_resnet50_in1k` | 0.688 [0.660, 0.713] | 0.572 [0.548, 0.598] | 0.98 | 0.053 |
| `dinov2_vits14` | 0.675 [0.650, 0.702] | 0.512 [0.487, 0.538] | 0.91 | 0.031 |

Mahalanobis gives the same ordering. Isolation Forest is **below 0.5** for both pretrained backbones on both tasks (0.41 to 0.48).

The pre-registered decision rule in notebook 06 printed: *"random_resnet50 is clearly better on both tasks"*. The spec says a random backbone close to the others means something is broken; here it is far ahead. **This is the main open problem and the next agent should start here.** Don't put these numbers in the README or deck until it's understood.

### 3.3 Throughput (`results/throughput.csv`)

Recorded but **not usable as a model-speed comparison**. The first backbone on each dataset paid for cold Drive reads: NCT random ResNet-50 at 5.5 tiles/s and PCam at 16.8 tiles/s, then 400 to 500 tiles/s for the next backbones on the same files. All PathMNIST rows are ~85 tiles/s (memmap from Drive, I/O-bound). Re-measure with data copied to local `/content` disk and a warm-up batch.

## 4. Open issues, in priority order

### 4.1 Anomaly ranking is inverted (blocker for the story)

Hypotheses, with what the numbers already hint:

1. **Train→test domain shift hurts pretrained features more.** The threshold is the 99th percentile of held-out *train* normals, so without shift the test-normal FPR should be ~1%. It is 0.4% for random, 3.1% for DINOv2, 5.3% for ImageNet. Test normals (CRC-VAL-HE-7K) look "further away" than train normals to the pretrained models. Check score distributions of train-holdout normals vs test normals per backbone.
2. **Semantic features put tumour next to normal mucosa.** The normal pool is 7 very different classes. Pretrained features cluster by tissue semantics, and TUM (glandular epithelium) may sit right next to NORM, while a random conv net mostly measures colour and texture statistics where dense, purple TUM tiles stand out. Check: per-class score boxplots, AUROC for TUM vs each normal class separately, and the class of each TUM tile's nearest normal neighbour.
3. **Information present, head is the problem?** Fit a supervised logistic regression TUM vs normal on the same cached features. If DINOv2 gets ~0.99 AUROC supervised, the features are fine and the unsupervised head / normal-pool geometry is the issue.
4. **Head settings.** `k=5`, random coreset capped at 2,000 (from ~6,000 fit normals). Mahalanobis runs on L2-normalised features with shrinkage 0.1. Isolation Forest on 2,048-d raw features is known to be weak; below-0.5 AUROC suggests trying PCA (e.g. 64–128 dims) before IF and Mahalanobis. Try larger `k`, no coreset, and per-class (class-conditional) kNN distance: score = min over normal classes of distance to that class's bank.
5. **Use the same dataset as notebook 04.** Notebook 06 is hard-coded to `pathmnist` for Task A even though NCT-CRC features are cached. Make it pick `nct` when cached (same `DATASET` logic as notebook 04), so probe and anomaly numbers come from the same tiles.
6. **DINOv2 patch-mean features** are cached (`dinov2_vits14__patchmean.npy`) but not used anywhere. Spec asks to report both CLS and patch-mean. Patch-mean may be more texture-sensitive and do better on anomaly.
7. **PCam specifics.** 96 px tiles upscaled to 224 (bicubic). The PCam label refers only to the centre 32×32 region, so a global embedding of the whole patch can dilute a small tumour focus. DINOv2 at 0.51 is chance. Try centre-crop before resize, and native-resolution input for the ResNets.

Also add a guard to `decide_backbone` (like `random_floor_warning` in `evaluate.py`): if `random_resnet50` is the winner or within a margin, refuse to declare a winner and print a "pipeline check" warning instead.

Whatever the cause, it's a good workshop talking point once explained ("generic semantic features are not automatically good anomaly features, and domain shift shows up as false positives"). Spec section 10 point 5 makes a related argument.

### 4.2 README results table is empty

Fill it from the CSVs once 4.1 is resolved. Spec acceptance criteria also need the dataset licence table (present) and credits (present).

### 4.3 Linear probe on random features underfits

`fit_linear_probe` uses `LogisticRegression(C=1.0)` on L2-normalised features with no scaling. On random features this collapses to one class at 10% labels and gives 0.489 at 100%, below kNN (0.617). Add a `StandardScaler` in a pipeline (fit on the train subset only) and possibly a small C grid. Re-run 04 afterwards; trained backbones may move a little too.

### 4.4 Notebook hygiene

- **00**: cell 5 sets `INCLUDE_PCAM = True`, then cell 11 sets `NCT_ROOT = "/content/nct"` and `INCLUDE_PCAM = False` before cell 12 uses it. Run top to bottom and PCam is not downloaded. Cell 10 downloads the NCT zips to Drive (`ssl-demo/data/nct/*.zip`) and extracts to `/content/nct`, which is ephemeral.
- **03**: cell 3 unzips selected NCT-100K class folders into Drive (`/content/drive/MyDrive/ssl-demo/data/nct`); `NCT_ROOT` points at Drive. That works, but it's slow (Drive FUSE) and inconsistent with notebook 00. Pick one: extract to local `/content/nct` every session from the zip on Drive (fast) and point both notebooks there. Index CSVs depend only on sorted file order inside each class folder, so the location can change safely.
- **03**: typo `#NCLUDE_PCAM = True` (commented out, harmless). PCam extraction is triggered by the presence of `indices/pcam_train_fast.csv` on Drive.
- 03 has no Colab badge cell at the top (00, 04, 06 do).
- 04 has a stray `drive.mount` cell near the end (cell 9).
- Notebooks were run on mixed GPUs (03 and 04 on L4, 06 on T4). The T4 45-minute budget is not yet confirmed end to end on a fresh runtime.

### 4.5 Throughput re-measure

See 3.3.

## 5. Next steps (follow the spec's build order)

1. Resolve 4.1 (diagnostics notebook or extra cells in 06), fix 4.3, switch 06 to NCT, fill the README table.
2. Add `sup_vit_s16_in21k` (timm), `dino_vits16` (torch.hub `facebookresearch/dino:main`), `phikon` (HF `owkin/phikon`, non-commercial, flag it) to `backbones.py`. Each model with **its own normalisation and input size** (juan has been bitten by wrong ViT normalisation before). CLS token for ViTs. Then notebook 05 (DINO last-layer attention per head vs Grad-CAM on supervised ResNet-50; link to `olonok69/Anomaly_detection` CAM notebook).
3. Notebook 01: port BYOL from `olonok69/BYOL` into `src/ssl_demo/byol.py`, keep the repo's structure and kNN monitor, credit the repo; collapse ablation (no predictor, no EMA). Notebook 02: mini DINO, loss/centering/EMA from scratch in `dino.py`, collapse ablation (no centering vs no sharpening). Add both checkpoints to the zoo.
4. Notebook 07 + `wsi.py`: OpenSlide `CMU-1-Small-Region.svs`, Otsu tissue mask, tile by mpp (~0.5 µm/px), heatmap, top-k pooling, GeoJSON export for QuPath. Optional extra `wsi` in `pyproject.toml` already pins openslide.
5. `docs/`: `guide/make_figures.py` reading `results/*.csv`, `make_deck.py` (python-pptx), `Presenter_Guide_EN.md`, `Demo_Guide_EN.md`, `Concepts_Cheatsheet_EN.md`, mirroring `olonok69/graimatter_ML/docs/`. Note: `graimatter_ML` is not in the GitHub repos currently attached to the project; add it (or ask juan) before reading it.

## 6. How to run

### Colab (the real runs)

Open each notebook via its badge (they clone `olonok69/SSL_pathology_demo` to `/content/SSL_pathology_demo` and `pip install -e .`). Set `MODE = "fast"` in the first code cell. Order:

1. `00_setup_and_data` – mounts Drive, downloads PathMNIST-224, builds index CSVs; optional NCT zips (cell 10) and PCam (`INCLUDE_PCAM`, ~7 GB). Mind issue 4.4.
2. `03_backbone_zoo_feature_extraction` – extracts all 3 backbones for every prepared dataset (PathMNIST, PCam if its index exists, NCT if `NCT_ROOT` set). Writes `throughput.csv`.
3. `04_knn_linear_probe_fewshot` – uses NCT if cached, else PathMNIST. Writes `probe_nct.csv`.
4. `06_anomaly_detection_on_embeddings` – Task A on PathMNIST, Task B on PCam if cached. Writes `anomaly.csv`.

Second runs hit the cache and skip extraction. Results are written to Drive `ssl-demo/results/` and to `results/` in the Colab clone; juan then copies/commits the CSVs to the repo (that's what the "results ..." / "anomaly" commits are).

Do **not** `pip install -r requirements.txt` or `uv sync` on Colab: it would replace the CUDA torch wheel. Torch is intentionally not a project dependency.

### Local (Windows, CPU, tests only)

```bash
uv sync          # creates .venv, installs repo + CPU torch from the "cpu" dependency group
uv run pytest    # toy tensors, no downloads
```

`num_workers=0` locally (Windows), 2 on Colab (`config.default_num_workers`). Local cache is `./cache/` (gitignored; currently holds `data/medmnist/pathmnist_224.npz`, 1.8 GB).

## 7. Repo layout

```
README.md                      what it is, badges, (empty) results table, licences, credits
ssl-pathology-demo-spec.md     the full spec
pyproject.toml / uv.lock       uv project; torch only in the "cpu" dependency group
requirements.txt               pinned pip fallback for local CPU only
src/ssl_demo/
  config.py      SEED, MODE subset sizes (fast 10k/2k, full 50k/7k; PCam 5k/2k, 20k/5k),
                 cache_root() (Drive on Colab, ./cache locally), save_results_csv()
  data.py        PathMNIST / NCT folder / PCam sources, stratified fixed index CSVs,
                 HED stain jitter, augmentation previews. Class ids: TUM=8, BACK=1, normal=0,2-7
  backbones.py   registry -> (model.eval(), preprocess, embed_dim); PHASE1_BACKBONES
  extract.py     extract_dataset(): inference_mode + fp16 autocast, L2-normalised, cached .npy,
                 cache valid only if saved index.npy matches the current tile list
  evaluate.py    temperature-weighted cosine kNN, logistic probe, few-shot, bootstrap CI,
                 random_floor_warning()
  anomaly.py     KNNDistanceHead, MahalanobisHead, IsolationForestHead, 99th-pct threshold on
                 held-out normals, metrics, DECISION_RULE, decide_backbone()
  viz.py         UMAP helpers
notebooks/       00, 03, 04, 06 (thin, import from src)
results/         anomaly.csv, probe_nct.csv, throughput.csv
tests/           one test file per module
```

Drive cache layout (`/content/drive/MyDrive/ssl-demo/`): `data/` (medmnist, pcam, nct zips/folders), `indices/{dataset}_{split}_{mode}.csv`, `features/{dataset}/{mode}/{split}/{backbone}.npy` (+ `labels.npy`, `index.npy`, `__patchmean.npy` for DINOv2), `results/`.

CSV schemas:
- `probe_nct.csv`: dataset, backbone, method (knn | linear_probe), label_fraction, metric (accuracy | balanced_accuracy | macro_f1), value, ci_low, ci_high
- `anomaly.csv`: dataset, task (task_a | task_a_background | task_b), backbone, head, metric (auroc | auprc | fpr_at_95_tpr | sensitivity_at_99 | fpr_at_99 | flag_rate), value, ci_low, ci_high
- `throughput.csv`: dataset, split, backbone, embed_dim, n_params, n_tiles, seconds, tiles_per_sec, cache_hit

## 8. Datasets and constraints

| Data | Use | Licence |
|---|---|---|
| PathMNIST-224 (MedMNIST, from Kather NCT-CRC) | Fast mode, Task A today | CC BY 4.0 |
| NCT-CRC-HE-100K (train) + CRC-VAL-HE-7K (test), Zenodo 1214456 | Probe today; should become Task A too | CC BY 4.0 |
| PatchCamelyon | Task B | CC0 |
| OpenSlide `CMU-1-Small-Region.svs` | Notebook 07 (not started) | public test data |
| Phikon (`owkin/phikon`) | Future backbone | **Non-commercial**, flag in README |
| UNI (`MahmoodLab/UNI`) | Optional, gated | Needs juan's HF approval; token only via Colab secrets (`google.colab.userdata`), never committed |

**Off-limits on juan's machine (`F:\repos`):**
- `F:\repos\test_retina\demo_images\` – UK Biobank fundus images. **Never use**, in anything.
- `F:\repos\test_retina\eyePacs\` – EyePACS/EyeQ (Kaggle). Private learning only, never upload or show.
- `F:\repos\test_retina\` other files – has a `.env` and a notebook with a live Azure key. Copy nothing from there.
- `F:\repos\DeepDRiD\` – optional retina bridge only, after checking its LICENSE.
- There is no histology on F:. All pathology data is the public sets above.

No AZ code, data or numbers anywhere in the repo or the workshop material.

## 9. Gotchas

- **Wrong normalisation is the classic silent bug.** Each backbone carries its own mean/std/input size in `BACKBONES`; keep it that way for every new model.
- **Cache keys include the tile list.** Changing seed, mode or subset size invalidates features automatically (index mismatch → re-extract). Deleting or reordering files inside an NCT class folder changes the index too.
- **`build_or_load_index` lets an existing CSV win over `n`.** If you change subset sizes in `config.py`, delete the old `indices/*.csv` on Drive or nothing changes.
- **`probe_nct.csv` keeps that name even for PathMNIST runs**; the `dataset` column is the truth.
- **Background (class 1)** is in the probe, excluded from the anomaly fit and test, and reported as `task_a_background` flag rate.
- **First backbone pays the Drive I/O cost.** Don't read throughput from a cold run.
- **Torch on Colab**: `pip install -e .` only. Torch/torchvision pinned (`2.14.1` / `0.29.2`) for local CPU only.
- **DINOv2 via `torch.hub`** with `trust_repo=True` (so the cell doesn't hang on a prompt); xFormers warnings are harmless.
- **PathMNIST label order is checked** against `medmnist.INFO` at load; it raises rather than guess the tumour class.
- **Repo naming**: spec suggested `ssl-pathology-demo`; the actual repo is `olonok69/SSL_pathology_demo`, local folder `BI_SSL_demo`. Notebooks clone the GitHub one.
- **Git workflow in practice**: juan commits straight to `main`, often from Colab ("Created using Colab"). The spec said branch work, juan merges. Work on a branch and let juan merge unless juan says otherwise; pull before editing notebooks because Colab may have pushed newer versions.
- This repo is **not** in the GitHub repos currently attached to the Claude project; to use GitHub tools on it, add it first. Local folder access to `D:\repos3\BI_SSL_demo` works.
- Notebook outputs are committed (they're the record of the Colab runs). Keep outputs when editing, or re-run.

## 10. Useful links

- Spec: `ssl-pathology-demo-spec.md` (repo root) / `/mnt/project-files/ssl-demo/ssl-pathology-demo-spec.md`
- Role context and juan's past work: `/mnt/project-files/fmad/fmad-interview-prep.md`
- juan's repos to reuse: `olonok69/BYOL` (notebook 01), `olonok69/Anomaly_detection` (CAM link in 05), `olonok69/pytorch-CNN` (MRNet max-pool = MIL, notebook 07), `olonok69/graimatter_ML` (docs pattern)
