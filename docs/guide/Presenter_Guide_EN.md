# Self-Supervised Features for Pathology: Presenter Guide

**A slide-by-slide companion to `docs/SSL_Pathology_EN.pptx` (27 slides)**

This guide is for whoever presents the deck, live as a workshop or recorded as a video. It follows the slides in order and gives, for each one, what is on screen, what to say, the numbers to have ready, the questions people tend to ask, and the line that leads into the next slide. The Spanish version is [Guia_Presentador_ES.md](Guia_Presentador_ES.md), with the same slide numbers. The [demo guide](Demo_Guide_EN.md) covers the notebooks: what to run live, what to show from saved outputs, and what to do when something fails. The [cheatsheet](Concepts_Cheatsheet_EN.md) is one page of definitions.

Every number in this guide comes from [`docs/brief.md`](../brief.md), which comes from the CSV files in `results/`. If a number changes, regenerate the figures and decks and update the brief; do not edit numbers here by hand.

---

## How to use this guide

Each slide has the same blocks:

* **On screen**: what the slide shows, so you can rehearse without the deck open.
* **Say**: a script in spoken style. Read it once or twice, then present in your own words. The deck's speaker notes carry a shorter version.
* **Key facts**: the numbers and sources you may be asked to repeat.
* **If asked**: short answers to questions that come up often.
* **Next**: one line to move to the following slide.

### Audience

The session is written for a mixed room: data scientists, pathologists, laboratory staff, engineers and managers. Explain the idea before the metric, and use the tissue images to anchor abstract points. When a technical term appears for the first time, the script defines it in one sentence.

### Run of show, 90-minute workshop

| Time | Block | Slides | Live |
|---|---|---|---|
| 0-10 | Opening: the problem, the loop, the data | 1-5 | – |
| 10-32 | Act 1 · Learn without labels | 6-11 | – |
| 32-45 | Act 2 · Measure the features | 12-14 | Notebook 04 saved outputs |
| 45-50 | Break | | |
| 50-80 | Act 3 · Detect and decide | 15-23 | Notebook 06: centring cell and decision rule |
| 80-90 | Engineering, takeaways, next steps, Q&A | 24-27 | – |

### Video cut, 25-30 minutes

Slides 1, 3, 5, 7, 8, 10, 11, 13, 14, 16, 17, 18, 19, 20, 22, 23, 25, 27. Show notebook outputs as screen recordings; run only the centring cell live. Suggested chapters: hook and the loop (3 min), Act 1 (8), Act 2 (5), Act 3 (11), wrap (3).

### Before the session

* Run the notebooks once on a fresh Colab runtime with the Drive cache in place (see the demo guide). The second run reads cached features and takes a few minutes.
* Keep open: the deck, notebook 06 in Colab, `results/anomaly.csv`, `results/anomaly_centring.csv` and `docs/brief.md`.
* Three messages to land, whatever you cut: (1) self-supervised features describe tissue well and pay off most when labels are scarce; (2) normal-only anomaly detection works, and the threshold is a choice the lab makes; (3) a random network that wins is a bug, and a rule written in advance is what lets you see it.

### What this version does not cover

BYOL and DINO are explained, not trained: those are notebooks 01 and 02 in v2. There are no attention maps (notebook 05) and no whole-slide heatmaps (notebook 07) yet. If someone asks to see them, say so plainly and point to the "What comes next" slide.

---

## Opening

### Slide 1 · Can a model learn what tissue looks like without labels?

*About 1 minute.*

**On screen.** Title, one-line summary, three tags (3 frozen backbones, 2 anomaly tasks, public data only), repository address.

**Say.** "Labelling tissue is slow and expensive. So the question for today is whether a model can learn what tissue looks like from images alone, with no labels, and whether we can use what it learned to flag tissue that is not normal. Along the way a second question turns out to matter just as much: when a new, bigger model appears, how do we know if it is really better? Everything you will see runs on public data and free Colab GPUs, and all of it is in the repository on the slide."

**Key facts.** Three backbones: an untrained ResNet-50, an ImageNet supervised ResNet-50, DINOv2 ViT-S/14. Two anomaly tasks: colorectal tumour (NCT-CRC) and lymph-node metastasis (PatchCamelyon).

**Next.** "Let us start with why labels are the bottleneck."

### Slide 2 · Labels are the bottleneck, not images

*About 2 minutes.*

**On screen.** Three cards: millions of tiles, few annotations, two questions. A band defining self-supervised learning.

**Say.** "A single whole-slide image, scanned at 20x, cuts into tens of thousands of small tiles. A preclinical study has hundreds of slides. No pathologist will ever label every tile. Self-supervised learning moves the effort: the model builds its own training signal from the images, for example by learning that two crops of the same picture should look alike to it. We then need labels only at the end, and only a few, to read what the model has learned for a specific task."

**If asked.** *Is this the same as unsupervised clustering?* Related, but no. Self-supervised learning trains a network with a pretext task built from the data; clustering groups data without training a representation. The features from self-supervised training can then be clustered, classified or used for anomaly detection.

**Next.** "Here is the loop we will use for every result in the session."

### Slide 3 · One loop: tiles, frozen backbone, cache, same head, compare

*About 2 minutes.*

**On screen.** Five boxes left to right; the backbone box is highlighted with "swap only this". Band: change the backbone, keep everything else fixed.

**Say.** "Every result today comes from the same loop. Tiles go into a backbone whose weights never change; we call it frozen. The output, a vector of a few hundred numbers per tile, is saved to disk once. On top we fit a small head: a nearest-neighbour vote, a logistic regression, or an anomaly score. To compare backbones we swap only the backbone and re-run the same head on the same tiles with the same seeds. If the number changes, the representation changed, because nothing else did."

**Key facts.** Features are cached as `.npy` files on Google Drive, keyed by the exact list of tiles. A second run skips the forward pass entirely.

**If asked.** *Why not fine-tune the backbone?* Fine-tuning mixes two questions: how good the representation is, and how well you trained. Frozen evaluation isolates the first. Fine-tuning is a later step, and on normal-only data it can make features worse for anomaly detection.

**Next.** "The session follows that loop in three acts."

### Slide 4 · Three acts, one story

*About 1 minute.*

**On screen.** Three cards: learn without labels, measure the features, detect and decide. Footer naming the notebooks in this version and those coming in v2.

**Say.** "First, the ideas: how a network can learn from images with no labels at all. Second, how to measure what it learned without retraining it. Third, how to use those features to find tissue that is not normal, and how to decide fairly between backbones. Act 3 contains what I think are the most useful ten minutes of the session: a result that looked spectacular and turned out to be a bug."

**Next.** "First, the data."

### Slide 5 · Nine tissue classes from colorectal cancer, two groups of patients

*About 2 minutes.*

**On screen.** Three tiles for each of the nine classes, tumour epithelium outlined. Band with the train, test and Task B sets.

**Say.** "This is the data: public colorectal cancer tiles released by Kather and colleagues. Nine classes, from fat and empty background to tumour epithelium, outlined on the right. Each tile is 224 pixels square at half a micron per pixel. We train on ten thousand tiles from one set of patients and test on two thousand from different patients, which is the honest way to evaluate. In the anomaly part, tumour epithelium plays 'not normal', and background is kept apart because empty glass is unusual but is not cancer. A second task uses PatchCamelyon: lymph-node patches with or without metastasis."

**Key facts.** NCT-CRC-HE-100K (train) and CRC-VAL-HE-7K (test), Zenodo 1214456, CC BY 4.0. PatchCamelyon, CC0. Fast mode: 10,000 / 2,000 tiles; PatchCamelyon 5,000 / 2,000. Seed 42.

**If asked.** *Is any of this clinical or company data?* No. Public datasets only, with their licences in the README. *Why do debris tiles look so different from each other?* Debris and necrosis are heterogeneous by nature, and stain varies between labs; that variety will come back when we look at false positives.

**Next.** "Act 1: where self-supervised backbones come from."

---

## Act 1 · Learn without labels

### Slide 6 · Act 1 opener

*Seconds.* "Where self-supervised backbones come from, and which one we actually run."

### Slide 7 · Supervised, self-supervised, foundation model

*About 3 minutes.*

**On screen.** Three cards. Band naming the three backbones in this workshop.

**Say.** "There are three ways a backbone can learn. Supervised: from human labels, like ImageNet's thousand object classes. Self-supervised: from the structure of the images themselves, with no labels. A foundation model is a self-supervised backbone trained on a very large dataset and reused, frozen, for many tasks. We compare three backbones. The untrained ResNet-50 is not a competitor: it is the floor, the score you get from the architecture alone. If it ever wins, something is wrong in the pipeline. Keep that sentence in mind; it comes back in Act 3."

**Key facts.** None of the three backbones was trained on these tissue tiles. DINOv2 was trained on natural images.

**If asked.** *Why include a random network at all?* Because it is the cheapest sanity check there is. A random convolutional network still captures colour and texture statistics, so it does better than chance on some tasks; a trained backbone should beat it clearly.

**Next.** "Self-supervised methods come in three families."

### Slide 8 · Three families: contrast, distil, mask

*About 3 minutes.*

**On screen.** Tree: contrastive (SimCLR, MoCo), self-distillation (BYOL, DINO, DINOv2), masking (MAE, BEiT, data2vec). Band on collapse.

**Say.** "Contrastive methods pull two views of the same image together and push other images apart; they need many negative examples. Self-distillation methods drop the negatives: one network learns to match another. Masking methods hide parts of the image and predict what was hidden, as language models do with words. All of them must avoid collapse: the trivial solution where the network outputs the same vector for every image. That vector matches every pair of views perfectly and has learned nothing. Each method's design is mostly a trick against collapse."

**If asked.** *Which family is best for pathology?* Current pathology foundation models mostly use self-distillation with masking, the DINOv2 and iBOT recipe. Contrastive methods work but need large batches for enough negatives.

**Next.** "BYOL, the first method that worked without negatives."

### Slide 9 · BYOL: no negatives, a predictor and a slow-moving target

*About 4 minutes.*

**On screen.** BYOL diagram: two views, online encoder, projector and predictor on top; target encoder and projector below; EMA arrow; stop-gradient into the comparison.

**Say.** "BYOL takes two augmented views of one image. The online network, at the top, encodes the first view, projects it and then predicts what the target network will output for the second view. The target network, at the bottom, is never trained by gradient. Its weights are an exponential moving average of the online weights: they follow slowly. Two asymmetries prevent collapse: the predictor exists only on the online side, and the target moves slowly. There are no negative examples at all, which surprised the field in 2020. I trained BYOL from scratch on small image datasets in a public repository; in v2 of this workshop that becomes notebook 01, on tissue tiles."

**Key facts.** Grill et al., 2020. Repository: `olonok69/BYOL`. v2 adds a collapse ablation: no predictor, no EMA.

**If asked.** *What is an exponential moving average here?* After each training step, target weights become 0.99 times their old value plus 0.01 times the online weights, or similar. The target is a smoothed history of the online network.

**Next.** "DINO keeps the idea and changes the target."

### Slide 10 · DINO: a student learns to agree with its own slow-moving teacher

*About 4 minutes.*

**On screen.** DINO diagram: global crops to the teacher, global and local crops to the student, centre and sharpen on the teacher, softmax on the student, cross-entropy, EMA arrow.

**Say.** "DINO calls the two networks student and teacher. Both are Vision Transformers. The teacher sees two large crops of the image; the student sees those plus several small crops. The student must produce the same probability distribution as the teacher, which forces it to infer the whole from a part. The teacher's weights, as in BYOL, are a moving average of the student's. Two operations on the teacher's output prevent collapse: centring, which stops one output dimension from dominating, and sharpening with a low temperature, which stops the output from becoming uniform. They pull in opposite directions, and the balance is what makes it train."

**Key facts.** Caron et al., 2021. Multi-crop: 2 global crops (224 px) and several local crops (96 px) in the original recipe.

**If asked.** *Why do people show DINO attention maps?* Because the teacher's attention heads, without any labels, often outline objects. In pathology they can highlight nuclei or glands. That is notebook 05 in v2.

**Next.** "The backbone we actually run is DINOv2."

### Slide 11 · DINOv2 ViT-S/14: the backbone we run, frozen

*About 3 minutes.*

**On screen.** Three cards: what DINOv2 adds, the ViT in one line, how we use it. Band: two questions to ask of any foundation model.

**Say.** "DINOv2 scales DINO in two ways: it adds a masked-patch objective borrowed from iBOT, and it trains on a curated set of 142 million natural images. No histology. We use the small Vision Transformer, ViT-S, which cuts each tile into 14-pixel patches. A special CLS token attends to all of them, and its final state, 384 numbers, is the embedding of the tile. Before trusting any number from a foundation model, ask two questions: was the evaluation data part of its training set, and what does its licence allow? Phikon, a pathology model, is non-commercial. UNI requires approved access."

**Key facts.** Oquab et al., 2023. Apache-2.0, loaded through `torch.hub`. ImageNet normalisation, 224 px input. Embedding: CLS token, 384 dimensions; the mean of patch tokens is also cached.

**If asked.** *Why not use a pathology foundation model?* Licences and access, mainly, in a public workshop. Phikon is planned for v2 with its non-commercial licence flagged. *Does the normalisation matter?* Very much. Each backbone needs its own mean, standard deviation and input size; getting that wrong is the classic silent bug.

**Next.** "Act 2: how to judge a frozen backbone."

---

## Act 2 · Measure the features

### Slide 12 · Act 2 opener

*Seconds.* "Judge a backbone without changing a single weight."

### Slide 13 · Nearest neighbours: DINOv2 0.890, ImageNet 0.841, random 0.616

*About 5 minutes. Show notebook 04's saved output table and confusion matrix if time allows.*

**On screen.** Two bar charts with 95% intervals: kNN accuracy, and linear-probe accuracy with all labels.

**Say.** "Two ways to read a frozen representation. The k-nearest-neighbour classifier labels a test tile by a vote of its 20 most similar training tiles: if similar tissue lands close together in feature space, it works. The linear probe fits a logistic regression on the features: if the nine classes can be separated by straight cuts, it works. Nothing in the backbone moves. DINOv2 leads on nearest neighbours, 0.890 against 0.841 for ImageNet. With all ten thousand labels, the probe nearly closes the gap: 0.913 against 0.903, and the intervals overlap. The untrained network sits well below, where a floor should be."

**Key facts.** `results/probe_nct.csv`. kNN: cosine similarity, k = 20, temperature-weighted vote. Probe: features standardised, logistic regression with C chosen by 3-fold cross-validation on the training rows only. 2,000 test tiles, nine classes, 1,000 bootstrap resamples.

**If asked.** *Why standardise the features?* Without it, the untrained network's features, which are nearly parallel, made the probe collapse to a single class at 10% of the labels. Standardising is fair to every backbone. *Are 0.913 and 0.903 different?* Not by this test: the intervals overlap.

**Next.** "The interesting part is what happens with fewer labels."

### Slide 14 · With 1% of the labels: 0.837 against 0.750

*About 4 minutes.*

**On screen.** Label-efficiency curves for the three backbones at 1%, 10% and 100% of the labels, with shaded intervals. A card on how to read it.

**Say.** "This is the curve that matters for a pathology lab. On the left, only one percent of the training labels: about a hundred tiles, eleven per class. DINOv2 features already reach 0.837 accuracy; ImageNet features 0.750. On the right, with all ten thousand labels, the two converge. The value of a better representation is largest exactly where labels are expensive, which in pathology is almost always."

**Key facts.** Probe accuracy at 1% / 10% / 100%: DINOv2 0.837 / 0.894 / 0.913; ImageNet 0.750 / 0.870 / 0.903; random 0.532 / 0.547 / 0.688.

**If asked.** *Why not report balanced accuracy?* It is in the CSV and tells the same story; classes are roughly balanced in the subset.

**Next.** "Act 3: can the same features flag tissue that is not normal?"

---

## Act 3 · Detect anomalies and decide

### Slide 15 · Act 3 opener

*Seconds.* "Learn what normal tissue looks like, flag the rest, and decide fairly between backbones."

### Slide 16 · Fit on normal tissue only; tumours appear only at test time

*About 4 minutes.*

**On screen.** Table of Task A and Task B: normal classes, anomaly, threshold, what is kept apart. Card with the three heads. Band: no tumour is seen during fitting or thresholding.

**Say.** "Anomaly detection turns the problem around: we only describe normal tissue. In Task A the head is fitted on seven normal classes, six thousand tiles. Tumour tiles appear only when we score the test set. The threshold is the score that 99 percent of a held-out set of normal tiles stays below, so about one percent of normal tissue is flagged by construction, and we never look at a tumour to choose it. Background is excluded and reported separately. There are three heads; the one named in our decision rule is the distance to the nearest normal tiles, which is the image-level version of the PatchCore memory bank."

**Key facts.** Task A: 6,009 normal tiles to fit, 1,502 held out for the threshold, 1,764 test tiles of which 344 are tumour. Task B: 2,000 normal patches to fit, 500 held out, 2,000 test. kNN head: k = 5, memory bank capped at 2,000 tiles.

**If asked.** *Why exclude background?* Empty glass is trivially far from tissue and would inflate the score while saying nothing about cancer. Its flag rate is reported so nothing is hidden.

**Next.** "Before computing any score, we wrote down how we would decide."

### Slide 17 · The rule is written down before any score is computed

*About 3 minutes.*

**On screen.** The decision rule in a dark box. Three cards: why two tasks, why intervals, why before.

**Say.** "The notebook prints this rule before it fits anything. A backbone is clearly better only if its confidence interval clears the next best on both tasks. Two tasks, because one dataset can flatter one backbone. Intervals, because a higher number inside an overlapping interval is not evidence. And before, so that the rule cannot bend towards the result we hoped for. A good rule must also be able to say 'no winner'."

**Key facts.** `ssl_demo.anomaly.DECISION_RULE`. AUROC with a 95% bootstrap interval over test tiles, kNN-distance head.

**Next.** "Here is what the first run said."

### Slide 18 · The untrained network wins. Something is wrong.

*About 3 minutes.*

**On screen.** Table of raw-cosine AUROC: random 0.885, ImageNet 0.672, DINOv2 0.668. A card with the tempting headline. Band with the rule of thumb.

**Say.** "This is what the first run of notebook 06 produced. The untrained network scored 0.885 on Task A. DINOv2, 0.668. The decision rule, applied mechanically, printed that the random network was clearly better on both tasks. It would have been easy to write a provocative post: 'pretrained features are worse than random for anomaly detection'. The right reaction is suspicion. A network that has learned nothing should be the floor. When the floor wins, assume a bug until you can prove otherwise."

**Key facts.** `results/anomaly_centring.csv`, raw-cosine column. Same tiles and the same fit split as the final result.

**If asked.** *Did you change the rule after seeing this?* No. The rule is unchanged. What changed is the head, after a diagnosis, and the code now also refuses to name a winner whenever the untrained backbone leads.

**Next.** "So what was going on?"

### Slide 19 · Every random vector points the same way

*About 4 minutes. This is the slide to slow down on.*

**On screen.** Top: a synthetic two-dimensional picture of raw and centred feature vectors. Bottom: mean cosine to the normal centroid per backbone (1.000, 0.820, 0.817). Card on what cosine distance sees.

**Say.** "Cosine distance compares directions, not lengths. Look at the left of the picture: when every vector points almost the same way, the angles between them are tiny, and whatever small residue is left decides the ranking. For the untrained network, the mean cosine between each tile and the average tile is 1.000: all vectors are essentially parallel. The residue that is left tracks how dark the stain is, and dense, purple tumour tiles happen to stand out on that. So the random network was not detecting tumours; it was measuring ink. The pretrained backbones have the same problem in a milder form: a cosine of about 0.82 to the centroid, which squeezes all their distances into a narrow band where irrelevant variation decides the ranking."

**Key facts.** Centroid cosine: random 1.000, ImageNet 0.820, DINOv2 0.817. The top picture is an illustration, not data.

**If asked.** *Why would an untrained network produce parallel vectors?* After many layers of random weights and ReLU activations, outputs are all positive and dominated by a common component. Batch-norm statistics from initialisation do not undo it.

**Next.** "The fix is one line."

### Slide 20 · Centre on the normal mean: random falls to 0.49, DINOv2 rises to 0.937

*About 3 minutes. Run the centring cell of notebook 06 live here (seconds on cached features).*

**On screen.** Slope chart of Task A AUROC from raw to centred cosine for each backbone. Card describing the change.

**Say.** "Subtract the average of the normal training features from every vector, bank and queries alike, and only then compute the cosine. That removes the shared direction and leaves the structure that distinguishes one tissue from another. The untrained network falls to 0.492, chance, exactly where a floor belongs. ImageNet rises to 0.902 and DINOv2 to 0.937. Same features, same tiles, same k. One line of code."

**Key facts.** `results/anomaly_centring.csv`. `KNNDistanceHead(center=True)` is now the default.

**If asked.** *Is centring a known technique?* Yes. Centring, or whitening, embeddings before cosine similarity is standard in retrieval and in anomaly detection on pretrained features; it is easy to forget when the code computes cosine for you.

**Next.** "How did we know this was the explanation and not just another tweak?"

### Slide 21 · The information was in the features all along

*About 3 minutes.*

**On screen.** Three cards: supervised ceiling, head settings, domain shift. Band with the debugging order.

**Say.** "Three checks. First: is the information in the features at all? A supervised classifier, tumour against normal, trained on the same cached features, reaches 0.99 AUROC for both pretrained backbones, and 0.87 for the random one. So yes. Second: is the head reading it? Changing k, removing the memory-bank cap, or scoring each normal class separately made little difference; centring made all of it. Third: is the test data shifted? A little. On raw cosine, three to eight percent of normal test tiles crossed a threshold set for one percent; after centring, 0.6 to 1.2 percent. That order of questions works in any embedding pipeline: is the signal there, is the head reading it, is the data moving?"

**Key facts.** Supervised check: 0.986 to 0.993 AUROC (pretrained), 0.87 (random). False-positive rate on test normals at the 99th-percentile threshold after centring: DINOv2 0.006, ImageNet 0.012.

**Next.** "Now the honest comparison."

### Slide 22 · Task A: DINOv2 clearly ahead. Task B: a tie. No winner declared.

*About 4 minutes. Show the decision-rule output in notebook 06.*

**On screen.** AUROC bars with intervals for both tasks. Band with the rule's output.

**Say.** "Same head, same tiles, only the backbone changes. On colorectal tissue DINOv2 is clearly ahead: 0.937, and its interval does not touch ImageNet's. On lymph-node metastasis both pretrained backbones sit around 0.75 and their intervals overlap. PatchCamelyon is harder for everyone, because its label refers only to the central 32 by 32 pixels of each patch, and an embedding of the whole patch dilutes a small tumour focus. So the rule says: no backbone is clearly better on both tasks. That is a useful answer. It tells us where DINOv2 helps and where we would need something else, such as a pathology-specific model or patch-level scoring."

**Key facts.** Task A AUROC: DINOv2 0.937 [0.925, 0.948], ImageNet 0.902 [0.883, 0.919], random 0.492. Task B: ImageNet 0.759 [0.739, 0.778], DINOv2 0.752 [0.732, 0.772], random 0.683.

**If asked.** *Why does the random network still score 0.68 on Task B?* Metastatic patches differ in colour and density from normal lymph node; random features still capture some of that. It is clearly below the pretrained backbones, which is what the floor should do. *What about Mahalanobis and Isolation Forest?* They are in `anomaly.csv`, untuned, and still rank the random network first. That is itself the lesson: the head is part of the comparison.

**Next.** "AUROC ranks; a lab needs a threshold."

### Slide 23 · A 1% false-alarm budget catches about a quarter of tumour tiles

*About 3 minutes.*

**On screen.** Table for DINOv2 on Task A: false alarms 0.006, tumour tiles flagged 0.285, background flagged 0.102, hardest normal classes. Card: not a failure, a choice.

**Say.** "AUROC tells you how well the scores put tumour above normal. A lab needs a decision: above this score, a pathologist looks. We set the threshold so that about one percent of normal tissue is flagged. At that point DINOv2 flags 28.5 percent of tumour tiles. That is a conservative operating point. A screening workflow would lower the threshold and accept more false alarms to miss fewer tumours; the curve lets you choose. And the false alarms that do appear cluster in debris and mucus: tissue that is unusual, not malignant. The gallery in notebook 06 shows them."

**Key facts.** `results/anomaly.csv`: `fpr_at_99`, `sensitivity_at_99`, `task_a_background` flag rate. Per-class AUROC for DINOv2: adipose 1.00, lymphocytes 0.99, normal mucosa 0.93, stroma 0.92, smooth muscle 0.91, debris 0.88, mucus 0.87.

**If asked.** *Is a quarter of tumour tiles useful?* At slide level, yes: a tumour region covers many tiles, so flagging a quarter of them still flags the region. Slide-level pooling is notebook 07 in v2.

**Next.** "A word on the engineering that made all this cheap."

---

## Wrap

### Slide 24 · Cheap to re-run, hard to fool yourself

*About 2 minutes.*

**On screen.** Four cards: cache features once, fix seeds and splits, numbers from files, floors and guards.

**Say.** "Four habits made the bug visible and the fix cheap. Features are cached, so re-running every head after the fix took two minutes on a laptop CPU. Seeds and splits are fixed, so numbers are reproducible and the cache invalidates itself when they change. Every number comes from a CSV, so when the fix changed the results, the figures and these slides were regenerated, not edited by hand. And there is always a random floor in the comparison, with code that refuses to declare a winner when the floor is ahead."

**Key facts.** Every notebook runs in fast mode on a free Colab T4 in under 45 minutes. The throughput file is not a speed ranking: the first backbone on each dataset paid for cold Drive reads.

**Next.** "Five things to take home."

### Slide 25 · Five things to take home

*About 2 minutes.*

**Say.** "One: self-supervised features already describe tissue; about a hundred labels reach 84 percent accuracy on nine classes. Two: their advantage is largest when labels are scarce. Three: you can build an anomaly detector from normal tissue alone, with a threshold chosen without seeing a tumour. Four, the one I most want you to remember: the head is part of the experiment, and a random network that wins is a bug until proven otherwise. Five: decide how you will decide before you look, and let the answer be 'no winner'."

**Next.** "What comes next."

### Slide 26 · From frozen backbones to training our own

*About 1 minute.*

**Say.** "This is version one. Version two trains our own backbones, BYOL and a small DINO from scratch, with ablations that show collapse happening. Then attention maps against Grad-CAM, and the step from tiles to whole slides, with a heatmap you can open in QuPath. And a pathology foundation model in the comparison, with its licence flagged."

### Slide 27 · Run it yourself

*Remaining time: questions.*

**Say.** "Everything is in the repository: notebooks with Colab badges, the results files, and these guides in English and Spanish. The data is public and the licences are in the README."

---

## Questions that come up

* **"Why not just fine-tune?"** Fine-tuning answers a different question and needs labels. Frozen evaluation tells you how good the representation is before you spend labelling effort. On normal-only data, fine-tuning can also make features worse for anomalies.
* **"Is DINO better than BYOL?"** On most benchmarks DINO-family models are stronger, and DINOv2 is the basis of most current pathology foundation models. v2 trains both on the same tiles so the comparison is ours.
* **"How do you pick the threshold?"** From normal tissue only, at a false-alarm rate the lab can afford to review. Here the 99th percentile of held-out normals, about 1% false alarms. Never tune it on tumours.
* **"What about scanner and stain differences?"** They are the main practical risk. The patient-disjoint test split shows a small shift already; stain normalisation and per-scanner evaluation are on the v2 list.
* **"Can I use UNI or Virchow commercially?"** Check each licence. Phikon is non-commercial; UNI is gated. DINOv2 is Apache-2.0.
* **"Was DINOv2 trained on these tiles?"** No; it was trained on natural images. Ask the same question of any pathology model before trusting its benchmark numbers.
