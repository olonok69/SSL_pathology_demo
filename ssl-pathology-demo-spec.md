# Spec: Self-supervised learning demo for pathology (BYOL, DINO, ViT)

Owner: juan (olonok69) · Written 26 Nov 2026 · Audience: the coding agent that builds the demo, then juan for the workshop.

## 0. What we are building and why

Two deliverables in one repo:

1. **A learning demo** (Colab notebooks) that shows, on real histology tiles, how self-supervised features (BYOL, DINO, DINOv2, a pathology foundation model) compare with supervised ImageNet features, and how a fixed anomaly-detection head behaves when only the backbone changes.
2. **Workshop / video material** built on top of the notebooks: presenter guide, demo script, slide deck, figures. Same pattern as juan's `olonok69/graimatter_ML` repo (`docs/guide/Presenter_Guide_EN.md`, `docs/guide/Demo_Guide_EN.md`, `docs/guide/make_figures.py`, `docs/make_deck.py`). Read that repo first and mirror its structure and tone.

The use case it mirrors (FMAD role): anomaly detection on whole-slide images of preclinical toxicology studies, where an internal histology foundation model (HFM) is plugged in as the backbone with the head unchanged, and benchmarked against DINO and external pathology FMs. Our demo is a small, public-data version of exactly that loop: **tiles → frozen backbone → cached features → same head → compare**.

Non-goals: beating any SOTA, training a foundation model, multi-node training.

## 1. Constraints

- **Runs on Colab.** Free T4 (16 GB) must be enough for every notebook in "fast mode". A100 (Colab Pro) is optional for "full mode". Each notebook must finish in under 45 minutes in fast mode.
- **Google Drive as the cache.** Datasets, checkpoints and feature arrays go to `/content/drive/MyDrive/ssl-demo/` so a Colab disconnect never means re-downloading or re-extracting. Every expensive step checks the cache first and skips if present.
- **One config flag for speed:** `MODE = "fast" | "full"` at the top of each notebook (fast = subset of data, fewer epochs, smaller backbones).
- **Seeds fixed**, and every results table saved as CSV to Drive (`results/`), so the deck figures are generated from files, not screenshots.
- **Public data only in anything that gets shown or published.** See section 3 for what not to use.

## 2. Repo layout

New repo, suggested name `olonok69/ssl-pathology-demo` (branch work only, juan merges).

```
ssl-pathology-demo/
  README.md                  # what it is, Colab badges per notebook, results table
  requirements.txt           # pinned: torch, torchvision, timm, lightly, scikit-learn, umap-learn,
                             #   medmnist, h5py, openslide-python, openslide-bin, matplotlib, pandas
  src/ssl_demo/
    data.py                  # dataset download + subset + transforms (incl. stain/colour aug)
    backbones.py             # registry: name -> (model, preprocess, embed_dim)
    extract.py               # extract_features(backbone, loader) -> np.ndarray, cached to Drive
    byol.py                  # BYOL modules, ported from olonok69/BYOL
    dino.py                  # mini DINO: multi-crop, DINO head, centering, sharpening, EMA
    evaluate.py              # kNN, linear probe, few-shot, bootstrap CI
    anomaly.py               # kNN-distance, Mahalanobis, IsolationForest, OCSVM heads
    viz.py                   # UMAP, attention maps, Grad-CAM, heatmap overlays
    wsi.py                   # (notebook 07) tissue mask, tiling by mpp, GeoJSON export
  notebooks/
    00_setup_and_data.ipynb
    01_byol_from_scratch.ipynb
    02_dino_from_scratch_mini.ipynb
    03_backbone_zoo_feature_extraction.ipynb
    04_knn_linear_probe_fewshot.ipynb
    05_attention_maps_vs_gradcam.ipynb
    06_anomaly_detection_on_embeddings.ipynb
    07_mini_wsi_pipeline.ipynb            # stretch
  docs/
    guide/Presenter_Guide_EN.md
    guide/Demo_Guide_EN.md
    guide/Concepts_Cheatsheet_EN.md
    guide/make_figures.py                 # reads results/*.csv -> docs/figures/*.png
    make_deck.py                          # python-pptx deck from figures + outline
  tests/                                  # small pytest: shapes, cache logic, metrics on toy data
```

Notebooks stay thin: they import from `src/ssl_demo` (pip install from the repo in the first cell) so the logic is testable and the notebooks read like a story.

## 3. Data

### What I found on juan's machine (F:\repos), and the verdict

| Location | What it is | Use it? |
|---|---|---|
| `F:\repos\DeepDRiD\` | DeepDRiD diabetic retinopathy dataset (regular fundus + ultra-widefield), with its LICENSE file | Optional, only for the retina "bridge" extra. Check the LICENSE before showing images in a video |
| `F:\repos\test_retina\images\` | 1,200 fundus JPGs (`1_l1.jpg` naming, DeepDRiD style), about 670 MB | Same as above |
| `F:\repos\test_retina\eyePacs\` | EyePACS / EyeQ (Kaggle) with quality labels, about 2 GB zipped plus 6 GB pre-processed | Private learning only. Kaggle terms don't allow redistribution, so don't upload it or show many images |
| `F:\repos\test_retina\demo_images\` | 875 PNGs named like UK Biobank fundus fields (`<id>_21015_0_0.png`) | **Do not use.** UK Biobank access terms don't allow this in a public demo or video |
| `F:\repos\test_retina\` (other files) | Contains a `.env` and a notebook with a live Azure endpoint key | Don't copy anything from this folder into the new repo |

No histology / WSI images on F:. So the core demo uses public pathology datasets.

### Datasets for the demo

| Dataset | What | Size | Licence | Role |
|---|---|---|---|---|
| **NCT-CRC-HE-100K** + **CRC-VAL-HE-7K** (Kather et al., Zenodo 1214456) | 224×224 H&E tiles at 0.5 µm/px, 9 tissue classes (adipose, background, debris, lymphocytes, mucus, smooth muscle, normal mucosa, stroma, tumour). Train and val sets come from **different patients** | 100K tiles ≈ 11 GB; 7K ≈ 0.8 GB | CC BY 4.0 | **Main dataset.** Train SSL + probes on a subset of 100K, test on 7K (patient-disjoint, which mirrors "never split one animal across train and test") |
| **PathMNIST** (MedMNIST v2, `pip install medmnist`) | Same source as NCT-CRC, resized to 28/64/128/224 | 224 version ≈ 1–2 GB | CC BY 4.0 | Fast mode and the from-scratch SSL notebooks (64 px) |
| **PatchCamelyon (PCam)** (`torchvision.datasets.PCAM`) | 96×96 lymph-node patches, metastasis vs normal, from CAMELYON16 | ≈ 7 GB (h5) | CC0 | Second anomaly-detection task: train the head on normal patches only, tumour = anomaly. Use a 20–50K subset |
| **OpenSlide test data** `CMU-1-Small-Region.svs` (openslide.cs.cmu.edu/download/openslide-testdata) | A tiny real .svs pyramid | ≈ 2 MB | public test data | Notebook 07: show OpenSlide reading, levels, mpp, tiling without big downloads |
| **CAMELYON16** one test slide (optional) | Real gigapixel WSI with tumour annotations | 1–3 GB per slide | CC0 | Notebook 07 full mode: real heatmap on a real slide |
| MVTec AD (optional) | Industrial anomaly benchmark | ≈ 5 GB | CC BY-NC-SA | Only if we want a non-medical sanity check. Skip by default |

### The anomaly framing (important, this is the FMAD link)

- **Task A (NCT-CRC):** "normal" = non-tumour tissue classes from the train set (fit the head only on these). "Abnormal" = TUM tiles. Evaluate on the 7K patient-disjoint set. Keep BACK (background) out of both, or report it separately, because it's trivially different (same as tissue masking in a real WSI pipeline).
- **Task B (PCam):** fit on normal lymph-node patches only, score tumour patches as anomalies.
- Same head, same splits, same seeds for every backbone. Only the backbone changes.

## 4. Backbones to compare

All behind one interface in `backbones.py`:

```python
get_backbone(name) -> (model.eval(), preprocess, embed_dim)
extract_features(name, dataset) -> np.ndarray  # cached at drive/ssl-demo/features/{dataset}/{name}.npy
```

| Name | Source | Why it's in the table |
|---|---|---|
| `random_resnet50` | torchvision, no weights | Floor. Shows the head alone is not doing the work |
| `sup_resnet50_in1k` | torchvision ImageNet weights | Classic supervised transfer (juan's Transfer-Learning repo) |
| `sup_vit_s16_in21k` | timm | Supervised ViT, to separate "ViT" from "self-supervised" |
| `byol_resnet18_ours` | trained in notebook 01 | Our own BYOL, small scale, in-domain |
| `dino_vit_tiny_ours` | trained in notebook 02 | Our own mini DINO, small scale, in-domain |
| `dino_vits16` | `torch.hub.load('facebookresearch/dino:main', 'dino_vits16')` | Official DINO v1 (the paper in the article) |
| `dinov2_vits14` | `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')` | DINOv2, the generic "foundation model" |
| `phikon` | HF `owkin/phikon` (ViT-B, iBOT on TCGA) | A pathology FM. Plays the role of "the HFM" in our story. Check the licence (non-commercial) and say so in the README |
| `uni` (optional) | HF `MahmoodLab/UNI` (gated, needs access approval) | Stronger pathology FM. Only if juan's HF access is approved. Don't commit any token |

Notes for the agent:
- Use each model's own normalisation and input size (DINOv2 uses patch 14, so 224 works; Phikon expects 224 with ImageNet stats). Wrong normalisation is the most common silent bug here, juan hit it before with a ViT image processor.
- Use the **CLS token** for ViTs (and also try mean of patch tokens for DINOv2, report both).
- Extract with `torch.inference_mode()`, AMP fp16, batch 256 on T4, `num_workers=2` (Colab), and L2-normalise features before kNN.
- Caveat to write in the notebook: some pathology FMs were trained on TCGA. NCT-CRC is not TCGA, but always ask "was the eval data in the pretraining set?" before trusting a number. Same question juan would ask about the internal HFM.

## 5. Notebook by notebook

Each notebook starts with a markdown cell: *goal, what you will see, runtime in fast/full mode, link to the concept slide*. And ends with *what we learned* in 3 bullets plus the CSV it wrote.

### 00 · Setup and data (≈10 min fast)
- Mount Drive, install the repo, download NCT-CRC (or PathMNIST in fast mode), PCam subset, OpenSlide test slide.
- Build fixed subsets and save the index lists (CSV) so every notebook uses the same tiles: fast = 10K train / 2K test; full = 50–100K / 7K.
- Show a tile grid per class, the class balance, and an example of the augmentations BYOL and DINO use (colour jitter, blur, solarise, crops) on a histology tile, plus an H&E-specific stain jitter (HED space). Short markdown on why ImageNet colour jitter is "wrong-ish" for H&E.

### 01 · BYOL from scratch (≈30–40 min fast)
- **Reuse juan's `olonok69/BYOL` repo**: online/target networks, projector + predictor MLPs, EMA target update, LARS optimiser (or AdamW in fast mode), and his **kNN monitor on frozen features every N epochs**. Port it into `src/ssl_demo/byol.py`, keep his structure and credit the repo in the notebook.
- ResNet18 on PathMNIST 64 px (fast) or NCT 224 subset (full), 20–50 epochs.
- Plots: loss, kNN accuracy over epochs, EMA τ schedule.
- **Collapse ablation (the teaching moment):** short runs with (a) no predictor, (b) no EMA (target = online copy). Plot the std of the normalised embeddings per dimension and kNN accuracy. Collapse shows as std → 0 and kNN → chance. Explain why the predictor + slow target avoid collapse with no negatives.
- Save checkpoint to Drive as `byol_resnet18_ours`.

### 02 · Mini DINO from scratch (≈30–40 min fast)
- Implement with `lightly` (`DINOProjectionHead`, `DINOLoss`, `DINOTransform`) or from scratch in `dino.py`, whichever is clearer. Prefer from scratch for the loss, centering and EMA, because that's the part we teach.
- ViT-Tiny (timm `vit_tiny_patch16_224` with small images, or patch 8 on 64 px), 2 global + 6 local crops in fast mode (paper: 2 × 224 + 10 × 96).
- Show: teacher temperature warm-up 0.04 → 0.07, centering EMA, teacher vs student kNN accuracy over training (the paper finds the teacher is better, the article explains it as Polyak-style averaging).
- **Collapse ablation:** no centering (one dimension dominates) vs no sharpening (uniform output). Plot output entropy and KL(teacher‖student), as in the paper's collapse figure.
- A small interactive cell: softmax with different temperatures on a toy vector, so sharpening is visible.
- Side-by-side table "BYOL vs DINO": predictor vs centering+sharpening, regression vs cross-entropy, multi-crop, typical backbone.

### 03 · Backbone zoo: feature extraction (≈20 min fast per dataset, cached)
- Run `extract_features` for every backbone in section 4 on NCT-CRC (and PCam for notebook 06).
- Log per backbone: embedding dim, params, throughput (tiles/s on the current GPU), extraction time. That throughput table is a real decision input (FMAD: thousands of tiles per slide).
- Everything cached, so notebooks 04–06 run in minutes.

### 04 · kNN, linear probe, few-shot (≈10 min)
- kNN (cosine, k = 20, temperature-weighted like DINO's eval) and logistic-regression linear probe on frozen features. Train on the 100K subset, test on 7K.
- **Few-shot / label efficiency:** linear probe with 1%, 10%, 100% of labels. This is where SSL usually shines and it's the "labels are expensive" argument from juan's AZ story.
- Metrics: accuracy, balanced accuracy, macro F1, per-class confusion matrix. 1,000× bootstrap CI over test tiles.
- UMAP of features coloured by tissue class, one panel per backbone (good slide).
- Output: `results/probe_nct.csv` (backbone, method, label_fraction, metric, ci_low, ci_high).
- Sanity check, not a target: ImageNet supervised features should already be strong on NCT-CRC, and pathology FMs should be at or near the top. If a random backbone is close to the others, something is broken.

### 05 · Attention maps vs Grad-CAM (≈10 min)
- DINO and DINOv2: last-layer CLS self-attention per head on 224 tiles (DINO repo's `get_last_selfattention`), thresholded to keep 60% of the mass, like the paper figure.
- Supervised ResNet50: Grad-CAM for the predicted class. **Link to juan's `olonok69/Anomaly_detection`** crack-detection CAM notebook in the narrative ("same idea I used on concrete cracks, now on tissue").
- Panel: same 6 tiles (tumour, lymphocytes, mucosa, stroma, debris, muscle) × (DINO head 1..6, DINOv2, Grad-CAM). Talking point: DINO heads attend to nuclei vs stroma without ever seeing a label.

### 06 · Anomaly detection on embeddings (≈15 min) · the core FMAD notebook
- Heads in `anomaly.py`, all fit on **normal-only** features:
  - kNN distance to a memory bank of normal features (PatchCore-style, image level), with optional coreset subsampling.
  - Gaussian + Mahalanobis distance (PaDiM-style, with shrinkage covariance).
  - Isolation Forest (link to the image-QC Isolation Forest in juan's AZ pipeline story).
  - One-class SVM (optional).
- Run Task A (NCT-CRC, TUM as anomaly) and Task B (PCam).
- Metrics: AUROC, AUPRC, FPR at 95% TPR, and **threshold from the 99th percentile of held-out normal scores**, then report sensitivity and false-positive rate at that threshold. Bootstrap CIs.
- **The backbone-swap table:** rows = backbones, columns = heads × tasks. Same head, same splits, only the backbone changes. Then a "decision" cell: pre-defined rule like "backbone X is clearly better if its AUROC CI doesn't overlap the next best on both tasks", written before looking at the numbers.
- A "false positive gallery": top-scored normal tiles per backbone (debris, folds, edges). This is the "artefacts and normal variation" talking point.
- Output: `results/anomaly.csv`.

### 07 · Mini WSI pipeline (stretch, ≈20 min fast)
- OpenSlide on `CMU-1-Small-Region.svs` (fast) or one CAMELYON16 slide (full): read levels, downsamples and `openslide.mpp-x`.
- Tissue mask: Otsu on saturation at a low level, remove small blobs.
- Tile tissue only at ≈0.5 µm/px (resample by mpp, not by level index), 224 px, keep coordinates.
- Features with the best backbone from 06, anomaly score with the fitted head (from NCT normals: a domain-shift caveat to say out loud).
- Heatmap overlay on the thumbnail, slide score = top-k mean of tile scores (MIL max/top-k pooling, **same idea as juan's `olonok69/pytorch-CNN` MRNet: max-pool over slices to one exam label**).
- Export high-score tiles as **GeoJSON** polygons that QuPath can open.
- CAMELYON16 full mode: compare heatmap with the tumour annotation.

### Optional extra · Retina bridge
Same notebook 03 + 06 pipeline on DeepDRiD fundus images from `F:\repos\DeepDRiD` (only after checking its licence), with "normal" = DR grade 0. Purpose: connect the demo to juan's AstraZeneca BYOL-on-fundus story. Never use the UK Biobank images.

## 6. Results the demo must produce (for the deck)

`docs/guide/make_figures.py` reads `results/*.csv` and writes:
1. BYOL and DINO training curves + collapse ablations (2 figures).
2. Linear probe / kNN bar chart per backbone with CIs.
3. Label-efficiency lines (1/10/100%).
4. UMAP grid.
5. Attention vs Grad-CAM panel.
6. Anomaly backbone-swap heatmap table + ROC curves for the top 3.
7. Throughput vs AUROC scatter (cost vs quality).
8. WSI heatmap overlay (if 07 is done).

## 7. Acceptance criteria for the coding agent

- Every notebook runs top to bottom on a fresh free Colab T4 in fast mode, under 45 min, and a second run uses the cache and is much faster.
- `pytest` passes locally on CPU with toy tensors (shapes, cache hit/miss, metrics against sklearn).
- No secrets, tokens, client data or UK Biobank / EyePACS images in the repo. HF token read from Colab secrets (`google.colab.userdata`) only.
- README has the final results table, the dataset licence table, and credits (DINO, DINOv2, BYOL paper, lightly, MedMNIST, Kather et al., juan's own repos it reuses).
- Licences of every model and dataset listed, with the non-commercial ones flagged.

## 8. Build order

1. 00 + `data.py` + `backbones.py` + `extract.py` with 3 backbones (random, supervised ResNet50, DINOv2). Get 04 and 06 working end to end on those. **This is the minimum demo.**
2. Add DINO, supervised ViT, Phikon to the zoo. Notebook 05.
3. 01 BYOL (port) and 02 mini DINO, add both to the zoo.
4. 07 mini WSI.
5. Figures, deck, guides (section 9).

---

## 9. Workshop / video material

Mirror `olonok69/graimatter_ML/docs/` (Presenter guide, Demo guide, `make_figures.py`, `make_deck.py`). Generated deck in `docs/` from the figures in section 6, so if numbers change, re-run and the slides update.

### Format options

- **Workshop, 90 min** (live, notebooks pre-run with cache on Drive).
- **YouTube video, 25–30 min** (main) plus **3 shorts of 5–8 min**: "BYOL without negatives", "DINO attention maps on tissue", "Swap the backbone, keep the head".

### Narrative (one story, not a list of methods)

> "Labels in pathology are slow and expensive. A pathologist can't annotate millions of tiles. So can a model learn what tissue looks like with no labels, and can we use that to flag what is *not* normal? And when a new, bigger foundation model arrives, how do we know if it's really better?"

Three acts: **learn without labels** (BYOL → DINO) → **measure the features** (kNN, probe, few-shot, attention) → **use them for anomaly detection and decide** (backbone swap, fair benchmark, WSI heatmap).

### Run of show (90 min workshop)

| Time | Block | Content | Notebook | Slides |
|---|---|---|---|---|
| 0–5 | Hook | A WSI heatmap with flagged regions. "No labels were used to train this backbone." | 07 output | 1–2 |
| 5–15 | Why SSL | Label cost; contrastive needs negatives (SimCLR/MoCo) vs BYOL/DINO don't; where foundation models come from | – | 3–6 |
| 15–30 | BYOL | Online/target, predictor, EMA. Live: loss and kNN curve; collapse ablation | 01 | 7–10 |
| 30–45 | DINO + ViT | Self-distillation, multi-crop, centering + sharpening, temperature demo; teacher beats student; DINOv2 in one slide (iBOT + curated data); MAE/BEiT/data2vec as the masking family (from Rohit Bandaru's post) | 02 | 11–16 |
| 45–50 | Break | | | |
| 50–62 | Measuring features | kNN, linear probe, few-shot curve, UMAPs; attention maps vs Grad-CAM | 04, 05 | 17–21 |
| 62–78 | Anomaly detection | Normal-only fitting, kNN/Mahalanobis/IForest; backbone-swap table; FP gallery; how to set a threshold; pre-registered decision rule | 06 | 22–27 |
| 78–85 | To the slide | OpenSlide, mpp, tissue mask, tiling, top-k pooling, GeoJSON to QuPath; feature caching and why I/O, not GPU, is the bottleneck | 07 | 28–31 |
| 85–90 | Wrap + Q&A | 5 takeaways; what I'd do next (DINOv2 fine-tune, ABMIL, stain normalisation, scanner shift) | – | 32–33 |

For the 25–30 min video: hook (1 min) → why SSL (3) → BYOL (5) → DINO (6) → features + attention (5) → anomaly + backbone swap (7) → WSI + wrap (3). Notebooks pre-run, show outputs, run only one or two cells live.

### Slide outline (≈33 slides)

1. Title · 2. Hook heatmap · 3. The label problem in pathology · 4. Supervised vs self-supervised vs foundation models · 5. Contrastive learning and its need for negatives · 6. The SSL family tree (contrastive / distillation / masking) · 7. BYOL architecture · 8. Why it doesn't collapse (predictor + EMA) · 9. BYOL training curves · 10. Collapse ablation · 11. ViT in one slide (patches, CLS token) · 12. DINO architecture · 13. Multi-crop · 14. Centering + sharpening (temperature demo) · 15. Teacher vs student · 16. DINOv2 / iBOT / MAE in one slide · 17. How we evaluate frozen features · 18. kNN + probe results · 19. Label efficiency · 20. UMAP grid · 21. Attention maps vs Grad-CAM · 22. Anomaly detection = learn normal · 23. The three heads · 24. Backbone-swap table · 25. ROC + threshold from normals · 26. False positive gallery · 27. How to decide fairly (same tiles, same head, CIs, rule first) · 28. WSI anatomy (pyramid, mpp) · 29. Tissue mask + tiling · 30. Heatmap + top-k pooling · 31. Engineering: cache features, I/O bottleneck, throughput vs quality · 32. Takeaways · 33. Resources + repo QR.

### Presenter guide content (`Presenter_Guide_EN.md`)

- Per slide: 2–4 speaking bullets, the one number to say, and the transition line.
- Where juan's own experience comes in (keep it honest): BYOL from scratch on CIFAR/STL (`olonok69/BYOL`) and BYOL at AstraZeneca on fundus images as "same family"; the CAM crack demo; MRNet max-pooling as MIL; the AZ image-QC Isolation Forest. Don't show any AZ code, data or numbers.
- Likely audience questions with short answers: "Why not just fine-tune?", "Is DINO better than BYOL?", "How do you pick the threshold?", "What about scanner/stain differences?", "Can I use UNI/Virchow commercially?" (licences).

### Demo guide content (`Demo_Guide_EN.md`)

- Pre-flight checklist: Drive cache warm, Colab GPU attached, HF token in Colab secrets, browser zoom, notebooks pre-run with outputs saved.
- Which cells run live (cheap ones: temperature demo, one attention panel, the anomaly head fit on cached features) and which only show saved outputs.
- Fallbacks: if Colab is down, the static outputs and the deck tell the full story; if a model download fails, skip that row.

### Cheatsheet (`Concepts_Cheatsheet_EN.md`)

One page: BYOL vs DINO vs DINOv2 vs MAE (teacher, target, loss, collapse trick, typical backbone), the evaluation protocols, the three AD heads, WSI vocabulary (pyramid, level, mpp, tissue mask, tile, MIL, GeoJSON). Doubles as juan's interview revision sheet.

## 10. Learning path after the demo (for juan)

1. Read the DINO and DINOv2 papers next to notebook 02; then iBOT (the masked-patch loss DINOv2 adds).
2. MAE and data2vec from the Rohit Bandaru post, as the masking family.
3. Attention-MIL (ABMIL, CLAM) for slide-level labels instead of top-k pooling.
4. Stain normalisation (Macenko) and per-scanner evaluation.
5. A short DINOv2 or Phikon fine-tune with LoRA on NCT normals, and check whether it helps or hurts anomaly detection (fine-tuning on normal-only data can make features worse for anomalies).

## Sources

- DINO article (Towards Data Science / Medium): https://towardsdatascience.com/dino-a-foundation-model-for-computer-vision-4cb08e821b18/
- BYOL article (Medium): https://medium.com/data-science/byol-the-alternative-to-contrastive-self-supervised-learning-5d0a26983d7c (couldn't be fetched automatically; spec relies on the BYOL paper, which juan has at `F:\repos\CV\Bootstrap Your Own Latent.pdf`, and his own BYOL repo)
- SSL with Vision Transformers (Rohit Bandaru): https://rohitbandaru.github.io/blog/SSL-with-Vision-Transformers/
- Role requirements and juan's past work: `/mnt/project-files/fmad/fmad-interview-prep.md`
