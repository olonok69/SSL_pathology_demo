"""Workshop figures, in English and Spanish, from the CSV files in results/.

Run from the repository root:

    uv sync --group docs
    uv run python docs/guide/make_figures.py

Writes SVG (for the guides) and PNG (for the decks) to docs/guide/img/en and docs/guide/img/es.
Every number drawn here is read from results/*.csv; nothing is typed in by hand. The
class grid (fig01) needs a few NCT-CRC tiles: set NCT_SAMPLES to a folder with one
subfolder per class (ADI, BACK, ...), or the figure is left as it is.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
IMG = Path(__file__).resolve().parent / "img"
NCT_SAMPLES = Path(os.environ.get("NCT_SAMPLES", ROOT / "cache" / "data" / "nct_samples"))

# Palette shared with docs/make_deck.py. Backbone colours follow the entity, never the rank.
NAVY, INK, MUTED, BODY = "#0F1B2D", "#1B2433", "#5F6979", "#3A4556"
CREAM, CARD, BORDER, GRID = "#F6F5F0", "#FDFCF9", "#E2DFD6", "#E8E5DC"
ORANGE, BLUE, GREEN, RED = "#B4521A", "#2F6DB5", "#2E7D4F", "#B23A3A"
PEACHBG, BLUEBG, GREENBG = "#FBE9DC", "#DCE7F5", "#DDEFE3"
FLOOR = "#7D838C"  # the untrained backbone: a reference floor, drawn neutral on purpose
COLOR = {"random_resnet50": FLOOR, "sup_resnet50_in1k": BLUE, "dinov2_vits14": ORANGE}
ORDER = ["random_resnet50", "sup_resnet50_in1k", "dinov2_vits14"]

plt.rcParams.update({
    "font.family": ["IBM Plex Sans", "Segoe UI", "DejaVu Sans"],
    "font.size": 12,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": BODY,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.facecolor": CARD,
    "axes.facecolor": CARD,
    "savefig.facecolor": CARD,
    "svg.fonttype": "none",
})

T = {
    "en": {
        "bb": {"random_resnet50": "Random ResNet-50", "sup_resnet50_in1k": "ImageNet ResNet-50",
               "dinov2_vits14": "DINOv2 ViT-S/14"},
        "classes": ["Adipose", "Background", "Debris", "Lymphocytes", "Mucus", "Smooth muscle",
                    "Normal mucosa", "Tumour stroma", "Tumour epithelium"],
        "accuracy": "Accuracy on 2,000 test tiles",
        "knn": "kNN (k = 20)", "probe": "Linear probe, all labels",
        "probe_title": "Frozen features, nine tissue classes",
        "labels_x": "Training labels used", "labels_title": "Label efficiency of the linear probe",
        "auroc": "AUROC (kNN-distance head)", "task_a": "Task A · NCT-CRC\ntumour vs normal tissue",
        "task_b": "Task B · PatchCamelyon\nmetastasis vs lymph node",
        "swap_title": "Same head, same tiles: only the backbone changes", "chance": "chance",
        "raw": "Raw cosine", "centred": "Centred cosine",
        "centring_title": "Task A AUROC before and after centring the features",
        "cos_title": "Mean cosine to the normal centroid", "cos_x": "1.0 = every tile points the same way",
        "geo_raw": "Raw features: all vectors share one direction",
        "geo_centred": "Centred: the shared direction is removed",
        "normal": "normal", "tumour": "tumour", "centroid": "centroid",
        "loop": ["Tiles", "Frozen\nbackbone", "Cached\nfeatures", "Same\nhead", "Compare"],
        "loop_sub": ["224 × 224 px\npublic data", "never updated", ".npy on Drive\nskip if cached",
                     "kNN · probe\nanomaly", "CIs + a rule\nfixed first"],
        "loop_swap": "swap only this",
        "fam": ["Self-supervised learning", "Contrastive", "Self-distillation", "Masking",
                "SimCLR · MoCo\nneeds negatives", "BYOL · DINO · DINOv2\nno negatives",
                "MAE · BEiT · data2vec\npredict hidden patches"],
        "byol": ["image", "view 1", "view 2", "online\nencoder", "projector", "predictor",
                 "target\nencoder", "projector", "match", "EMA of online weights", "stop-gradient"],
        "dino": ["image", "global crops", "global + local\ncrops", "teacher\nViT", "student\nViT",
                 "centre +\nsharpen", "softmax", "cross-entropy", "EMA of student weights",
                 "stop-gradient"],
    },
    "es": {
        "bb": {"random_resnet50": "ResNet-50 aleatoria", "sup_resnet50_in1k": "ResNet-50 ImageNet",
               "dinov2_vits14": "DINOv2 ViT-S/14"},
        "classes": ["Tejido adiposo", "Fondo", "Detritos", "Linfocitos", "Moco", "Músculo liso",
                    "Mucosa normal", "Estroma tumoral", "Epitelio tumoral"],
        "accuracy": "Exactitud en 2.000 teselas de test",
        "knn": "kNN (k = 20)", "probe": "Sonda lineal, todas las etiquetas",
        "probe_title": "Features congeladas, nueve clases de tejido",
        "labels_x": "Etiquetas de entrenamiento usadas", "labels_title": "Eficiencia de etiquetas de la sonda lineal",
        "auroc": "AUROC (cabeza de distancia kNN)", "task_a": "Tarea A · NCT-CRC\ntumor frente a tejido normal",
        "task_b": "Tarea B · PatchCamelyon\nmetástasis frente a ganglio",
        "swap_title": "Misma cabeza, mismas teselas: solo cambia el backbone", "chance": "azar",
        "raw": "Coseno sin centrar", "centred": "Coseno centrado",
        "centring_title": "AUROC de la tarea A antes y después de centrar las features",
        "cos_title": "Coseno medio con el centroide normal", "cos_x": "1,0 = todas las teselas apuntan igual",
        "geo_raw": "Sin centrar: todos los vectores comparten una dirección",
        "geo_centred": "Centradas: se elimina la dirección común",
        "normal": "normal", "tumour": "tumor", "centroid": "centroide",
        "loop": ["Teselas", "Backbone\ncongelado", "Features\nen caché", "Misma\ncabeza", "Comparar"],
        "loop_sub": ["224 × 224 px\ndatos públicos", "nunca se actualiza", ".npy en Drive\nse omite si existe",
                     "kNN · sonda\nanomalías", "IC + una regla\nfijada antes"],
        "loop_swap": "solo se cambia esto",
        "fam": ["Aprendizaje autosupervisado", "Contrastivo", "Autodestilación", "Enmascarado",
                "SimCLR · MoCo\nnecesita negativos", "BYOL · DINO · DINOv2\nsin negativos",
                "MAE · BEiT · data2vec\npredice parches ocultos"],
        "byol": ["imagen", "vista 1", "vista 2", "codificador\nonline", "proyector", "predictor",
                 "codificador\nobjetivo", "proyector", "comparar", "EMA de los pesos online", "sin gradiente"],
        "dino": ["imagen", "recortes globales", "recortes globales\ny locales", "profesor\nViT", "alumno\nViT",
                 "centrar +\nafilar", "softmax", "entropía cruzada", "EMA de los pesos del alumno",
                 "sin gradiente"],
    },
}


def fmt(value: float, lang: str, digits: int = 3) -> str:
    text = f"{value:.{digits}f}"
    return text.replace(".", ",") if lang == "es" else text


def spread(values, gap=0.035):
    """Label positions that keep at least ``gap`` between neighbours, in input order."""
    order = np.argsort(values)
    pos = np.array(values, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if pos[b] - pos[a] < gap:
            pos[b] = pos[a] + gap
    return pos


def save(fig, lang: str, name: str) -> None:
    if lang == "es":
        from matplotlib.ticker import FuncFormatter
        comma = FuncFormatter(lambda v, _p: f"{v:g}".replace(".", ","))
        for ax in fig.axes:
            if ax.axison and ax.get_yscale() == "linear" and len(ax.get_yticks()) and not getattr(ax, "_category_y", False):
                ax.yaxis.set_major_formatter(comma)
            if getattr(ax, "_numeric_x", False):
                ax.xaxis.set_major_formatter(comma)
    out = IMG / lang
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=120 if name.startswith("fig01") else 200, bbox_inches="tight")
    plt.close(fig)


def bar_with_ci(ax, x, value, low, high, color, label, lang):
    ax.bar(x, value, width=0.62, color=color, edgecolor=CARD, linewidth=2, zorder=2)
    ax.errorbar(x, value, yerr=[[value - low], [high - value]], fmt="none", ecolor=INK,
                elinewidth=1.2, capsize=4, zorder=3)
    ax.text(x, high + 0.02, fmt(value, lang), ha="center", va="bottom", fontsize=11, color=INK)


# ----------------------------------------------------------------------------- data figures
def fig_probe(lang, probe):
    t = T[lang]
    acc = probe[probe["metric"] == "accuracy"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    panels = [("knn", 1.0, t["knn"]), ("linear_probe", 1.0, t["probe"])]
    for ax, (method, frac, title) in zip(axes, panels):
        block = acc[(acc["method"] == method) & np.isclose(acc["label_fraction"], frac)].set_index("backbone")
        for i, name in enumerate(ORDER):
            r = block.loc[name]
            bar_with_ci(ax, i, r.value, r.ci_low, r.ci_high, COLOR[name], t["bb"][name], lang)
        ax.set_xticks(range(3), [t["bb"][n].replace(" ", chr(10), 1) for n in ORDER], fontsize=10.5)
        ax.set_title(title, fontsize=13, loc="left")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(t["accuracy"])
    save(fig, lang, "fig06_probe")


def fig_label_efficiency(lang, probe):
    t = T[lang]
    acc = probe[(probe["metric"] == "accuracy") & (probe["method"] == "linear_probe")]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ends = [acc[acc["backbone"] == n].sort_values("label_fraction")["value"].iloc[-1] for n in ORDER]
    end_pos = spread(ends, gap=0.03)
    for i, name in enumerate(ORDER):
        b = acc[acc["backbone"] == name].sort_values("label_fraction")
        x = b["label_fraction"].to_numpy()
        ax.fill_between(x, b["ci_low"], b["ci_high"], color=COLOR[name], alpha=0.14, linewidth=0)
        ax.plot(x, b["value"], color=COLOR[name], linewidth=2, marker="o", markersize=8,
                markeredgecolor=CARD, markeredgewidth=2)
        ax.text(x[-1] * 1.12, end_pos[i], t["bb"][name], va="center", fontsize=11, color=INK)
        ax.text(x[0], b["value"].iloc[0] + 0.03, fmt(b["value"].iloc[0], lang), ha="center", fontsize=10.5, color=INK)
    ax.set_xscale("log")
    ticks = ["1 %", "10 %", "100 %"] if lang == "es" else ["1%", "10%", "100%"]
    ax.set_xticks([0.01, 0.1, 1.0], ticks)
    ax.minorticks_off()
    ax.set_xlim(0.007, 4.5)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel(t["labels_x"])
    ax.set_ylabel(t["accuracy"])
    ax.set_title(t["labels_title"], fontsize=13, loc="left")
    save(fig, lang, "fig07_label_efficiency")


def fig_swap(lang, anomaly):
    t = T[lang]
    k = anomaly[(anomaly["head"] == "knn_distance") & (anomaly["metric"] == "auroc")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, task, title in zip(axes, ["task_a", "task_b"], [t["task_a"], t["task_b"]]):
        block = k[k["task"] == task].set_index("backbone")
        for i, name in enumerate(ORDER):
            r = block.loc[name]
            bar_with_ci(ax, i, r.value, r.ci_low, r.ci_high, COLOR[name], t["bb"][name], lang)
        ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.text(-0.62, 0.51, t["chance"], color=MUTED, fontsize=10, ha="left", va="bottom")
        ax.set_xlim(-0.65, 2.45)
        ax.set_xticks(range(3), [t["bb"][n].replace(" ", chr(10), 1) for n in ORDER], fontsize=10.5)
        ax.set_title(title, fontsize=12.5, loc="left")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(t["auroc"])
    save(fig, lang, "fig09_backbone_swap")


def fig_centring(lang, centring):
    t = T[lang]
    c = centring.set_index("backbone")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    raws = [c.loc[n, "auroc_raw_cosine"] for n in ORDER]
    cens = [c.loc[n, "auroc_centred_cosine"] for n in ORDER]
    raw_pos, cen_pos = spread(raws), spread(cens)
    for i, name in enumerate(ORDER):
        raw, cen = raws[i], cens[i]
        ax.plot([0, 1], [raw, cen], color=COLOR[name], linewidth=2.2, marker="o", markersize=9,
                markeredgecolor=CARD, markeredgewidth=2, zorder=3)
        ax.text(-0.05, raw_pos[i], fmt(raw, lang), ha="right", va="center", fontsize=11, color=INK)
        ax.text(1.05, cen_pos[i], f"{fmt(cen, lang)}  {t['bb'][name]}", ha="left", va="center", fontsize=11, color=INK)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.5, 0.505, t["chance"], color=MUTED, fontsize=10, ha="center", va="bottom")
    ax.set_xticks([0, 1], [t["raw"], t["centred"]], fontsize=12)
    ax.set_xlim(-0.35, 1.9)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC")
    ax.grid(axis="x", visible=False)
    ax.set_title(t["centring_title"], fontsize=13, loc="left")
    save(fig, lang, "fig08_centring")


def fig_centroid_cosine(lang, centring):
    t = T[lang]
    c = centring.set_index("backbone")
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    for i, name in enumerate(ORDER[::-1]):
        v = c.loc[name, "centroid_cosine"]
        ax.barh(i, v, height=0.56, color=COLOR[name], edgecolor=CARD, linewidth=2)
        ax.text(v + 0.01, i, fmt(v, lang), va="center", fontsize=11, color=INK)
    ax.set_yticks(range(3), [t["bb"][n] for n in ORDER[::-1]])
    ax._category_y = True
    ax.set_xlim(0, 1.1)
    ax._numeric_x = True
    ax.set_xlabel(t["cos_x"])
    ax.grid(axis="y", visible=False)
    ax.set_title(t["cos_title"], fontsize=13, loc="left")
    save(fig, lang, "fig08b_centroid_cosine")


# ----------------------------------------------------------------------------- concept figures
def fig_geometry(lang):
    """Synthetic 2-D picture of why centring matters. Illustrative, not data."""
    t = T[lang]
    rng = np.random.default_rng(3)
    shared = np.array([6.0, 0.6])
    normal = shared + rng.normal(0, [0.35, 0.35], size=(40, 2))
    tumour = shared + np.array([0.3, 0.9]) + rng.normal(0, 0.18, size=(10, 2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, centred in zip(axes, [False, True]):
        mu = normal.mean(0) if centred else np.zeros(2)
        for pts, col in [(normal, BLUE), (tumour, ORANGE)]:
            p = pts - mu
            for v in p:
                ax.annotate("", xy=v, xytext=(0, 0),
                            arrowprops=dict(arrowstyle="-", color=col, alpha=0.35, linewidth=1))
            ax.scatter(p[:, 0], p[:, 1], s=34, color=col, edgecolor=CARD, linewidth=1.2, zorder=3)
        ax.scatter([0], [0], s=40, color=INK, zorder=4)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(t["geo_centred"] if centred else t["geo_raw"], fontsize=12, loc="left")
        if centred:
            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(-1.4, 1.9)
        else:
            ax.set_xlim(-0.3, 7.6)
            ax.set_ylim(-0.6, 2.6)
    axes[0].scatter([], [], color=BLUE, label=t["normal"])
    axes[0].scatter([], [], color=ORANGE, label=t["tumour"])
    axes[0].legend(frameon=False, loc="upper left")
    save(fig, lang, "fig12_geometry")


def _box(ax, x, y, w, h, text, fc=CARD, ec=BORDER, color=INK, size=12, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, linewidth=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def _arrow(ax, a, b, color=MUTED, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=14, color=color,
                                 linewidth=1.5, linestyle=ls, shrinkA=0, shrinkB=0))


def _canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def fig_loop(lang):
    t = T[lang]
    fig, ax = _canvas(12, 3.4)
    xs = [1.2, 3.6, 6.0, 8.4, 10.8]
    for i, (x, title, sub) in enumerate(zip(xs, t["loop"], t["loop_sub"])):
        fc = PEACHBG if i == 1 else CARD
        ec = ORANGE if i == 1 else BORDER
        _box(ax, x, 2.15, 1.9, 1.15, title, fc=fc, ec=ec, bold=True, size=13)
        ax.text(x, 0.95, sub, ha="center", va="center", fontsize=10.5, color=MUTED, linespacing=1.3)
        if i < 4:
            _arrow(ax, (x + 1.0, 2.15), (xs[i + 1] - 1.0, 2.15))
    ax.text(xs[1], 3.1, t["loop_swap"], ha="center", fontsize=11, color=ORANGE)
    save(fig, lang, "fig02_loop")


def fig_family(lang):
    t = T[lang]
    fig, ax = _canvas(12, 4.2)
    _box(ax, 6, 3.5, 4.4, 0.8, t["fam"][0], fc=NAVY, ec=NAVY, color=CREAM, bold=True, size=13)
    for x, name, sub, fc, ec in [(2, t["fam"][1], t["fam"][4], CARD, BORDER),
                                 (6, t["fam"][2], t["fam"][5], PEACHBG, ORANGE),
                                 (10, t["fam"][3], t["fam"][6], CARD, BORDER)]:
        _arrow(ax, (6, 3.08), (x, 2.42))
        _box(ax, x, 2.0, 3.2, 0.8, name, fc=fc, ec=ec, bold=True, size=13)
        ax.text(x, 0.85, sub, ha="center", va="center", fontsize=11, color=BODY, linespacing=1.35)
    save(fig, lang, "fig03_ssl_family")


def fig_byol(lang):
    lab = T[lang]["byol"]
    fig, ax = _canvas(12, 4.6)
    _box(ax, 1.0, 2.3, 1.3, 0.8, lab[0], size=12)
    _box(ax, 3.0, 3.5, 1.4, 0.7, lab[1], size=11)
    _box(ax, 3.0, 1.1, 1.4, 0.7, lab[2], size=11)
    _arrow(ax, (1.65, 2.5), (2.3, 3.4))
    _arrow(ax, (1.65, 2.1), (2.3, 1.2))
    _box(ax, 5.2, 3.5, 1.7, 0.9, lab[3], fc=BLUEBG, ec=BLUE, size=11)
    _box(ax, 7.3, 3.5, 1.5, 0.7, lab[4], fc=BLUEBG, ec=BLUE, size=11)
    _box(ax, 9.3, 3.5, 1.5, 0.7, lab[5], fc=PEACHBG, ec=ORANGE, size=11, bold=True)
    _box(ax, 5.2, 1.1, 1.7, 0.9, lab[6], size=11)
    _box(ax, 7.3, 1.1, 1.5, 0.7, lab[7], size=11)
    _box(ax, 11.0, 2.3, 1.4, 0.8, lab[8], fc=NAVY, ec=NAVY, color=CREAM, size=12, bold=True)
    for a, b in [((3.7, 3.5), (4.35, 3.5)), ((6.05, 3.5), (6.55, 3.5)), ((8.05, 3.5), (8.55, 3.5)),
                 ((3.7, 1.1), (4.35, 1.1)), ((6.05, 1.1), (6.55, 1.1))]:
        _arrow(ax, a, b)
    _arrow(ax, (10.05, 3.3), (10.6, 2.65))
    _arrow(ax, (8.05, 1.1), (10.6, 1.95), color=MUTED, ls=(0, (4, 3)))
    ax.text(8.3, 0.72, lab[10], fontsize=10, color=MUTED, ha="left")
    _arrow(ax, (5.2, 3.0), (5.2, 1.6), color=BLUE, ls=(0, (2, 2)))
    ax.text(5.35, 2.3, lab[9], fontsize=10, color=BLUE, va="center")
    save(fig, lang, "fig04_byol")


def fig_dino(lang):
    lab = T[lang]["dino"]
    fig, ax = _canvas(12, 4.6)
    _box(ax, 1.0, 2.3, 1.3, 0.8, lab[0], size=12)
    _box(ax, 3.1, 3.5, 1.8, 0.7, lab[1], size=11)
    _box(ax, 3.1, 1.1, 1.8, 0.8, lab[2], size=11)
    _arrow(ax, (1.65, 2.5), (2.2, 3.4))
    _arrow(ax, (1.65, 2.1), (2.2, 1.2))
    _box(ax, 5.5, 3.5, 1.6, 0.9, lab[3], size=11)
    _box(ax, 5.5, 1.1, 1.6, 0.9, lab[4], fc=BLUEBG, ec=BLUE, size=11)
    _box(ax, 7.7, 3.5, 1.6, 0.9, lab[5], fc=PEACHBG, ec=ORANGE, size=11, bold=True)
    _box(ax, 7.7, 1.1, 1.6, 0.7, lab[6], fc=BLUEBG, ec=BLUE, size=11)
    _box(ax, 10.6, 2.3, 2.0, 0.8, lab[7], fc=NAVY, ec=NAVY, color=CREAM, size=12, bold=True)
    for a, b in [((4.0, 3.5), (4.7, 3.5)), ((6.3, 3.5), (6.9, 3.5)),
                 ((4.0, 1.1), (4.7, 1.1)), ((6.3, 1.1), (6.9, 1.1)), ((8.5, 1.2), (9.6, 2.0))]:
        _arrow(ax, a, b)
    _arrow(ax, (8.5, 3.4), (9.6, 2.6), ls=(0, (4, 3)))
    ax.text(9.5, 3.35, lab[9], fontsize=10, color=MUTED)
    _arrow(ax, (5.5, 1.55), (5.5, 3.05), color=BLUE, ls=(0, (2, 2)))
    ax.text(5.65, 2.3, lab[8], fontsize=10, color=BLUE, va="center")
    save(fig, lang, "fig05_dino")


def fig_class_grid(lang):
    folders = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
    if not all((NCT_SAMPLES / f).is_dir() for f in folders):
        print(f"fig01 skipped: no tiles under {NCT_SAMPLES}")
        return
    t = T[lang]
    fig, axes = plt.subplots(3, 9, figsize=(14, 5.2))
    for col, folder in enumerate(folders):
        files = sorted((NCT_SAMPLES / folder).iterdir())[:3]
        for row, path in enumerate(files):
            ax = axes[row, col]
            ax.imshow(Image.open(path).convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(col == 8)
                s.set_color(ORANGE)
                s.set_linewidth(2.5)
        axes[0, col].set_title(t["classes"][col], fontsize=10.5, color=ORANGE if col == 8 else INK)
    fig.subplots_adjust(wspace=0.06, hspace=0.06)
    save(fig, lang, "fig01_class_grid")


def main() -> None:
    probe = pd.read_csv(RESULTS / "probe_nct.csv")
    anomaly = pd.read_csv(RESULTS / "anomaly.csv")
    centring = pd.read_csv(RESULTS / "anomaly_centring.csv")
    for lang in ("en", "es"):
        fig_class_grid(lang)
        fig_loop(lang)
        fig_family(lang)
        fig_byol(lang)
        fig_dino(lang)
        fig_probe(lang, probe)
        fig_label_efficiency(lang, probe)
        fig_centring(lang, centring)
        fig_centroid_cosine(lang, centring)
        fig_swap(lang, anomaly)
        fig_geometry(lang)
        print("figures written:", IMG / lang)


if __name__ == "__main__":
    main()
