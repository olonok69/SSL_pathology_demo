# Concepts Cheatsheet

One page of definitions for the SSL pathology workshop. Spanish version: [Chuleta_Conceptos_ES.md](Chuleta_Conceptos_ES.md).

## Self-supervised methods

| Method | Two networks | Target | Loss | What prevents collapse | Typical backbone |
|---|---|---|---|---|---|
| SimCLR / MoCo (contrastive) | One encoder (MoCo adds a momentum encoder) | The other view of the same image, against many negatives | InfoNCE | Negatives push different images apart | ResNet-50 |
| BYOL | Online and target, target = EMA of online | Target projection of the other view | Mean squared error between normalised vectors | Predictor on the online side only, slow EMA target | ResNet-50 |
| DINO | Student and teacher, teacher = EMA of student | Teacher's probability distribution over global crops | Cross-entropy | Centring and sharpening of the teacher output | ViT-S/16, ViT-B/16 |
| DINOv2 | As DINO | As DINO, plus masked patch tokens (iBOT) | Cross-entropy, image and patch level | As DINO, plus regularisers | ViT-S/14 to ViT-g/14, 142M curated images |
| MAE / BEiT / data2vec (masking) | One encoder (and decoder) | Hidden patches: pixels, tokens or teacher features | Reconstruction or regression | The task itself; no trivial constant reconstructs pixels | ViT |

* **Collapse.** The network outputs the same vector for every image. It matches all views perfectly and has learned nothing.
* **EMA (exponential moving average).** Target weights = m × target + (1 − m) × online, with m close to 1. A slow, smoothed copy.
* **Multi-crop.** Large global crops and small local crops of one image; the student must match the teacher from a small part.
* **ViT and CLS token.** The image is cut into patches (14 × 14 px for ViT-S/14). A CLS token attends to all patches; its final state is the image embedding (384 numbers for ViT-S).
* **Foundation model.** A backbone pretrained at scale, often self-supervised, reused frozen for many tasks. Check what it was trained on and its licence.

## Evaluating frozen features

* **Frozen.** Backbone weights never change during evaluation; only a small head is fit.
* **kNN classifier.** Label a test tile by a vote of its k most similar training tiles (cosine similarity). Here k = 20, temperature-weighted.
* **Linear probe.** Logistic regression on the features. Here standardised, C by 3-fold cross-validation on training rows.
* **Label efficiency.** The probe trained on 1%, 10% and 100% of the labels.
* **Bootstrap interval.** Resample the test tiles with replacement 1,000 times; the 2.5th and 97.5th percentiles of the metric.
* **Random floor.** An untrained backbone with the same architecture. A trained backbone must beat it clearly.

## Anomaly detection on embeddings

* **Normal-only.** Fit the head on normal tissue only. Anomalies are seen only at test time.
* **kNN-distance head.** Mean cosine distance to the k nearest normal tiles (k = 5), after centring on the normal mean. Image-level version of the PatchCore memory bank.
* **Centring.** Subtract the mean of the normal fit features before the cosine. Removes the shared direction that otherwise dominates cosine distance.
* **Centroid cosine.** Mean cosine between each feature vector and the mean vector. Close to 1 means raw cosine distances say little.
* **Mahalanobis head.** Distance to a Gaussian fitted to normal features, with shrinkage covariance (PaDiM-style).
* **Isolation Forest.** Random trees that isolate unusual points quickly; short paths mean anomalous.
* **Threshold.** 99th percentile of scores on held-out normal tiles: about 1% of normal tissue flagged by construction.
* **AUROC.** Probability that a random anomaly scores higher than a random normal. 0.5 is chance.
* **AUPRC.** Area under precision against recall. Depends on how rare anomalies are.
* **FPR at 95% TPR.** Share of normals flagged when 95% of anomalies are caught.
* **Pre-registered decision rule.** Written before the numbers. A backbone wins only if its interval clears the next best on both tasks.

## Whole-slide vocabulary (for v2)

* **WSI.** Whole-slide image: a gigapixel scan of a glass slide.
* **Pyramid and level.** A WSI is stored at several resolutions; level 0 is full resolution.
* **mpp.** Microns per pixel. About 0.25 at 40x, 0.5 at 20x. Tile by mpp, not by level.
* **Tissue mask.** Separates tissue from glass, often by Otsu thresholding on a low-resolution level.
* **Tile.** A fixed-size patch (224 px here) cut from the tissue area.
* **MIL.** Multiple-instance learning: a slide label for a bag of tiles. Top-k pooling or attention (ABMIL, CLAM) aggregates tile scores.
* **GeoJSON.** Format for exporting regions or heatmaps to QuPath.

## The numbers to remember

| | Random ResNet-50 | ImageNet ResNet-50 | DINOv2 ViT-S/14 |
|---|---|---|---|
| kNN accuracy | 0.616 | 0.841 | 0.890 |
| Probe, 1% labels | 0.532 | 0.750 | 0.837 |
| Probe, 100% labels | 0.688 | 0.903 | 0.913 |
| Task A AUROC, raw cosine | 0.885 | 0.672 | 0.668 |
| Task A AUROC, centred | 0.492 | 0.902 | 0.937 |
| Task B AUROC, centred | 0.683 | 0.759 | 0.752 |
| Centroid cosine | 1.000 | 0.820 | 0.817 |
