# Brief: SSL pathology demo, workshop v1

Source document for the deck (`docs/make_deck.py`), the presenter guide, the demo guide and the cheatsheet, in English and Spanish. Every number on a slide or in a guide comes from this brief or from the CSV files in `results/`. If a number changes, change it here first.

Write for a mixed audience: data scientists, pathologists, engineers and managers. Explain the idea before the metric. One story, not a list of methods.

## Scope of v1

v1 covers what the repository runs today: notebooks 00, 03, 04 and 06, three frozen backbones, NCT-CRC and PatchCamelyon in fast mode. BYOL (01), mini DINO (02), attention against Grad-CAM (05) and the whole-slide notebook (07) are not built yet. In v1, BYOL and DINO are explained on slides only, and the deck says so. No slide shows a training curve, an attention map or a slide heatmap that the repository did not produce.

## The story

Labels in pathology are slow and expensive: a pathologist cannot annotate millions of tiles. Two questions follow.

1. Can a model learn what tissue looks like without labels, and can those features flag what is not normal?
2. When a new, bigger backbone appears, how do we know whether it is really better?

The loop that answers both: **tiles, frozen backbone, cached features, the same head, compare**. The backbone never changes during evaluation; only the head is fit. Changing the backbone and keeping everything else fixed is the whole comparison.

Three acts:

1. **Learn without labels.** Why self-supervised learning (SSL), the families (contrastive, self-distillation, masking), BYOL and DINO in concept, DINOv2 as the backbone we actually use.
2. **Measure the features.** kNN, linear probe, label efficiency.
3. **Detect anomalies and decide.** Normal-only heads, a pre-registered decision rule, and the bug that almost became the headline.

## Claims to keep accurate

- **Data is public.** NCT-CRC-HE-100K and CRC-VAL-HE-7K (Kather et al., Zenodo 1214456, CC BY 4.0), PathMNIST (MedMNIST, CC BY 4.0, derived from the same Kather data), PatchCamelyon (CC0). No client data, no UK Biobank, no EyePACS, nothing from a former employer.
- **Fast mode.** 10,000 training tiles from NCT-CRC-HE-100K and 2,000 test tiles from CRC-VAL-HE-7K, stratified, seed 42. The two sets come from different patients. PatchCamelyon: 5,000 / 2,000.
- **Backbones are frozen.** `random_resnet50` (no pretrained weights, the floor), `sup_resnet50_in1k` (torchvision ImageNet-1k supervised), `dinov2_vits14` (DINOv2 ViT-S/14, self-supervised on a curated natural-image set, Apache-2.0). None of them was trained on these tissue tiles. DINOv2 was not trained on NCT-CRC; still ask that question of any pathology foundation model.
- **We did not train BYOL or DINO in v1.** Say "in v2 we train them from scratch" if asked.
- **The random backbone is a floor, not a competitor.** If it ties or wins, the pipeline is wrong.
- **The decision rule was fixed before the numbers** and is printed in notebook 06 before any fit. It did not declare a winner. Do not present DINOv2 as "the winner"; present it as "clearly ahead on Task A, tied on Task B".
- **The threshold** is the 99th percentile of scores on held-out normal training tiles. Tumours are never used to set it.
- **Mahalanobis and Isolation Forest** are in `anomaly.csv` and rank the untrained network first. They were not tuned. Present them as "the head is part of the comparison", not as a backbone result.
- **Throughput** in `results/throughput.csv` is not a model-speed comparison (first backbone paid cold Drive reads). Do not put it on a slide as a speed ranking.
- **Licences.** Phikon is non-commercial; UNI is gated. Neither is used in v1.

## Numbers

### Feature quality (notebook 04, `results/probe_nct.csv`)

Accuracy on the 2,000 test tiles, nine classes. 95% bootstrap interval where shown.

| Backbone | kNN (k=20) | Probe 1% labels | Probe 10% | Probe 100% |
|---|---|---|---|---|
| `random_resnet50` | 0.616 [0.596, 0.635] | 0.532 | 0.547 | 0.688 [0.667, 0.708] |
| `sup_resnet50_in1k` | 0.841 [0.824, 0.857] | 0.750 [0.732, 0.769] | 0.870 | 0.903 [0.889, 0.916] |
| `dinov2_vits14` | **0.890** [0.876, 0.904] | **0.837** [0.821, 0.853] | **0.894** | **0.913** [0.901, 0.925] |

The line to say: with 1% of the labels (about 100 tiles), DINOv2 features reach 0.837 and ImageNet features 0.750. With all 10,000 labels the gap closes to one point and the intervals overlap. The value of better features is largest when labels are scarce.

Probe details: features standardised (scaler fit on the training rows), logistic regression with C chosen by 3-fold cross-validation from {0.001, 0.01, 0.1, 1} on the training rows only. kNN: cosine, k = 20, temperature-weighted, as in the DINO evaluation.

### Anomaly detection (notebook 06, `results/anomaly.csv`)

**Task A** (NCT-CRC): normal = adipose, debris, lymphocytes, mucus, smooth muscle, normal colon mucosa, cancer-associated stroma. Anomaly = colorectal adenocarcinoma epithelium (TUM). Background is excluded from the fit and the AUROC and reported as a flag rate. The head is fit on 6,009 normal training tiles; 1,502 more are held out to set the threshold. Test: 1,764 tiles, 344 of them tumour.

**Task B** (PatchCamelyon): normal lymph node against metastasis. Fit on 2,000 normal patches, 500 held out; test 2,000.

kNN-distance head (k = 5, coreset 2,000, features centred on the normal mean):

| Backbone | Task A AUROC | Task A AUPRC | Sensitivity at threshold | FPR at threshold | Background flagged | Task B AUROC |
|---|---|---|---|---|---|---|
| `random_resnet50` | 0.492 [0.463, 0.517] | 0.173 | 0.000 | 0.004 | 0.000 | 0.683 [0.658, 0.708] |
| `sup_resnet50_in1k` | 0.902 [0.883, 0.919] | 0.694 | 0.238 | 0.012 | 0.140 | **0.759** [0.739, 0.778] |
| `dinov2_vits14` | **0.937** [0.925, 0.948] | **0.777** | **0.285** | 0.006 | 0.102 | 0.752 [0.732, 0.772] |

Decision rule output: "No backbone is clearly better on both tasks. task_a: dinov2_vits14 AUROC 0.937 [0.925, 0.948] is clear of sup_resnet50_in1k [0.883, 0.919]. task_b: sup_resnet50_in1k leads on the point estimate (0.759) but its CI overlaps dinov2_vits14."

Sensitivity at the 99th-percentile threshold is low (about a quarter of tumour tiles) because the threshold allows about 1% false positives on normal tissue. That is a deliberate operating point, not a failure: a screening tool would pick a lower threshold and accept more false alarms.

Per-class view, DINOv2, Task A, AUROC of tumour against each normal class: adipose 1.00, lymphocytes 0.99, normal colon mucosa 0.93, stroma 0.92, smooth muscle 0.91, debris 0.88, mucus 0.87. The hard normals are debris and mucus, which also produce most false positives (debris 4.3% flagged). ImageNet ResNet-50: normal colon mucosa is its hardest class (0.80).

PatchCamelyon is harder for every backbone (about 0.75). The label refers only to the centre 32 x 32 pixels of a 96 x 96 patch, and the patch is upscaled to 224, so a small tumour focus is diluted in a global embedding.

### The bug that almost became the headline (`results/anomaly_centring.csv`)

The first run of notebook 06 scored raw cosine distance, and the untrained network won on both tasks, by a wide margin. Same tiles, same fit split:

| Backbone | Mean cosine to the normal centroid | Task A AUROC, raw cosine | Task A AUROC, centred cosine |
|---|---|---|---|
| `random_resnet50` | 1.000 | 0.885 | 0.492 |
| `sup_resnet50_in1k` | 0.820 | 0.672 | 0.902 |
| `dinov2_vits14` | 0.817 | 0.668 | 0.937 |

Explanation in plain words: cosine distance compares directions. Every untrained feature vector points almost exactly the same way, so the only thing left to measure is a tiny residue that tracks stain intensity, and dense purple tumour tiles happen to stand out on that. Pretrained features also share a large common direction (cosine about 0.82 to the centroid), which squeezes all distances into a narrow band where nuisance variation decides the ranking. Subtracting the mean of the normal tiles removes the shared direction; what remains is tissue structure. After centring, the untrained network falls to chance and the pretrained ones lead.

What else was checked before settling on that explanation:
- A supervised logistic regression (tumour against normal, on the same cached features) reaches 0.986 to 0.993 AUROC for both pretrained backbones and 0.87 for the random one. The information was in the features; the head was not reading it.
- Larger k, no coreset and class-conditional kNN changed little. Centring changed everything.
- Domain shift between training and test patients is real but secondary: on raw cosine, pretrained features flagged 3 to 8% of test normals at a threshold set for 1%; after centring that drops to 0.6 to 1.2%.

The lessons, which are the most useful minutes of the workshop:
1. A random network beating a trained one is a pipeline bug until proven otherwise. `decide_backbone` now refuses to name a winner in that case.
2. The head's geometry is part of the comparison. "Same head" must also mean "a head that is fit for these features".
3. A pre-registered rule protects you from your own excitement, in both directions.

## Concepts, in words a slide can carry

- **Self-supervised learning.** The model makes its own training signal from unlabelled data, for example by predicting one view of an image from another view of the same image.
- **Contrastive learning** (SimCLR, MoCo) pulls two views of the same image together and pushes other images away. It needs negatives, often many.
- **BYOL** (Grill et al., 2020) has no negatives. An online network predicts the output of a target network on another view; the target is a slow moving average (EMA) of the online network. The predictor and the EMA keep it from collapsing to a constant.
- **DINO** (Caron et al., 2021) is self-distillation: a student matches the probability distribution of a teacher over views at several scales (multi-crop). The teacher is an EMA of the student. Centring and sharpening the teacher's output prevent collapse.
- **DINOv2** (Oquab et al., 2023) scales DINO with a masked-patch objective (from iBOT) and a curated dataset of 142 million images. We use its small ViT, ViT-S/14, frozen.
- **Masking family** (MAE, BEiT, data2vec): hide patches, predict what was hidden.
- **ViT and the CLS token.** A Vision Transformer cuts the image into patches (14 x 14 pixels for ViT-S/14) and adds a CLS token whose final state summarises the image. We use the CLS token as the tile embedding.
- **Frozen evaluation.** kNN and linear probe score the features without changing the backbone. A better number means a better representation, not a better fine-tune.
- **Label efficiency.** The probe at 1%, 10% and 100% of the labels.
- **Normal-only anomaly detection.** Fit on normal tissue only. Score how far a new tile is from normal. No tumour is seen during fitting or thresholding.
- **kNN distance head.** Mean cosine distance to the k nearest normal tiles (the image-level version of PatchCore's memory bank).
- **AUROC.** Probability that a random tumour tile scores higher than a random normal tile. 0.5 is chance.
- **AUPRC.** Precision against recall; sensitive to how rare the anomaly is (here 344 of 1,764).
- **Threshold at the 99th percentile of held-out normals.** About 1% of normal tissue is flagged by construction.

## Suggested deck (about 26 slides)

Title each slide as a claim. Speaker notes carry the definitions.

1. Title.
2. The question: labels are slow; can a model learn tissue without them?
3. One loop: tiles, frozen backbone, cache, same head, compare.
4. Agenda: three acts.
5. The data: nine tissue classes, two patient groups, public licences (class grid figure).
6. Supervised, self-supervised, foundation model.
7. The SSL family tree: contrastive, self-distillation, masking.
8. BYOL in one picture: no negatives, predictor plus EMA.
9. DINO in one picture: student, teacher, multi-crop, centring and sharpening.
10. DINOv2 and the ViT CLS token: what we actually run, frozen.
11. Act 2 opener: how to judge a frozen backbone.
12. kNN and linear probe: DINOv2 0.890 / 0.913, ImageNet 0.841 / 0.903, random 0.616 / 0.688.
13. Label efficiency: with 1% of labels, 0.837 against 0.750.
14. Act 3 opener: learn normal, flag the rest.
15. Two tasks and three heads; the threshold comes from normals only.
16. The decision rule, written before the numbers.
17. First result: the untrained network wins. Something is wrong.
18. Why: every random vector points the same way (centroid cosine figure).
19. Centre the features: random falls to 0.49, DINOv2 rises to 0.937.
20. The information was there all along: supervised check 0.99.
21. Backbone swap, same head: Task A clear, Task B tied, no winner declared.
22. What the threshold means: 1% false alarms, a quarter of tumours caught, where the false positives come from.
23. Engineering: cache features, fixed seeds, results as CSV, figures from files.
24. Five takeaways.
25. What comes in v2: BYOL and DINO from scratch, attention maps, whole-slide heatmaps, Phikon.
26. Resources and repository.

## Suggested format

- **Workshop, 90 minutes**, notebooks pre-run with the cache on Drive. Live: notebook 06 on cached features (about 2 minutes on CPU), the centring cell, the decision rule.
- **Video, 25 to 30 minutes.** Act 1 (8), Act 2 (6), Act 3 (12), wrap (3).
- **Bilingual.** Every deck and guide exists in English and in Spanish (Spain). Same slide numbers in both.

## Repository map

| Path | Role |
|---|---|
| `notebooks/00_setup_and_data.ipynb` | Data, index CSVs, augmentation previews |
| `notebooks/03_backbone_zoo_feature_extraction.ipynb` | Frozen feature extraction and cache |
| `notebooks/04_knn_linear_probe_fewshot.ipynb` | kNN, probe, label efficiency |
| `notebooks/06_anomaly_detection_on_embeddings.ipynb` | Normal-only heads, centring check, decision rule |
| `src/ssl_demo/` | Library code the notebooks import |
| `results/*.csv` | Every number in the deck |
| `docs/guide/make_figures.py` | Figures from `results/`, English and Spanish |
| `docs/make_deck.py` | Decks in English and Spanish |
