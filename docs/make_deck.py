"""Build the SSL pathology workshop decks, in English and Spanish.

Run from the repository root, after docs/guide/make_figures.py:

    uv sync --group docs
    uv run python docs/guide/make_figures.py
    uv run python docs/make_deck.py            # both languages
    uv run python docs/make_deck.py --lang es  # one language

Writes docs/SSL_Pathology_EN.pptx and docs/SSL_Pathology_ES.pptx. Figures come from
docs/guide/img/<lang>/*.png. Numbers on the slides come from docs/brief.md, which in turn
comes from results/*.csv. Speaker notes carry the definitions and the script.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
REPO_URL = "github.com/olonok69/SSL_pathology_demo"

NAVY, INK, MUTED, BODY = "#0F1B2D", "#1B2433", "#5F6979", "#3A4556"
ORANGE, PEACH, PEACHBG = "#B4521A", "#F2A35E", "#FBE9DC"
CREAM, CARD, BORDER = "#F6F5F0", "#FDFCF9", "#E2DFD6"
BLUE, BLUEBG = "#2F6DB5", "#DCE7F5"
GREEN, GREENBG, RED, REDBG = "#2E7D4F", "#DDEFE3", "#B23A3A", "#F6DEDE"
SANS, MONO = "IBM Plex Sans", "Consolas"
LEFT, WIDTH = 0.89, 11.55


def rgb(h):
    return RGBColor.from_string(h.lstrip("#"))


class Deck:
    def __init__(self, lang: str):
        self.lang = lang
        self.img = HERE / "guide" / "img" / lang
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(13.333), Inches(7.5)
        self.layout = self.prs.slide_layouts[5]
        self.prefix = ""
        self.count = 0

    def L(self, en, es):
        return es if self.lang == "es" else en

    def n(self, value, digits=3):
        text = f"{value:.{digits}f}"
        return text.replace(".", ",") if self.lang == "es" else text

    # -- primitives ------------------------------------------------------------------
    def slide(self, kicker, title, dark=False, title_size=None):
        if title_size is None:
            title_size = 30 if len(title) <= 46 else 24
        s = self.prs.slides.add_slide(self.layout)
        self.count += 1
        bg = s.background.fill
        bg.solid()
        bg.fore_color.rgb = rgb(NAVY if dark else CREAM)
        t = s.shapes.title
        t.left, t.top, t.width, t.height = Inches(LEFT), Inches(1.32), Inches(WIDTH), Inches(0.95)
        tf = t.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.TOP
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.text = title
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        for r in p.runs:
            r.font.name, r.font.size, r.font.bold = SANS, Pt(title_size), True
            r.font.color.rgb = rgb(CREAM if dark else NAVY)
        if kicker:
            self.text(s, LEFT, 0.89, WIDTH, 0.3, self.prefix + kicker, 12, PEACH if dark else ORANGE, font=MONO)
        if self.count > 1:
            self.text(s, 12.0, 6.82, 0.6, 0.3, str(self.count), 11, "#9AA7B8" if dark else MUTED, align=PP_ALIGN.RIGHT)
        return s

    def text(self, s, x, y, w, h, text, size=15, color=BODY, bold=False, font=SANS, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, italic=False):
        box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        paras = text if isinstance(text, list) else [text]
        for i, para in enumerate(paras):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            if i:
                p.space_before = Pt(6)
            runs = para if isinstance(para, list) else [(para, bold)]
            for chunk, b in runs:
                r = p.add_run()
                r.text = chunk
                r.font.name, r.font.size, r.font.bold, r.font.italic = font, Pt(size), b, italic
                r.font.color.rgb = rgb(color)
        return box

    def rect(self, s, x, y, w, h, fill, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
        r = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
        r.fill.solid()
        r.fill.fore_color.rgb = rgb(fill)
        if line:
            r.line.color.rgb = rgb(line)
            r.line.width = Pt(1)
        else:
            r.line.fill.background()
        r.shadow.inherit = False
        style = r._element.find("{http://schemas.openxmlformats.org/presentationml/2006/main}style")
        if style is not None:
            r._element.remove(style)
        if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
            r.adjustments[0] = min(0.08, 0.12 / min(w, h))
        return r

    def card(self, s, x, y, w, h, title, body, dark=False, fill=None, tcolor=None, tsize=16, bsize=13):
        self.rect(s, x, y, w, h, fill or (NAVY if dark else CARD), None if (dark or fill) else BORDER)
        self.text(s, x + 0.22, y + 0.2, w - 0.44, 0.4, title, tsize, tcolor or (PEACH if dark else NAVY), True)
        lines = 1 + (len(title) * tsize * 0.0064 > w - 0.44)
        self.text(s, x + 0.22, y + 0.25 + 0.3 * lines, w - 0.44, h - 0.6, body, bsize, "#D9DEE6" if dark else BODY)

    def band(self, s, y, h, runs, fill=PEACHBG, color=INK, size=14.5):
        self.rect(s, LEFT, y, WIDTH, h, fill)
        self.text(s, LEFT + 0.3, y, WIDTH - 0.6, h, [runs], size, color, anchor=MSO_ANCHOR.MIDDLE)

    def footer(self, s, text, dark=False):
        self.text(s, LEFT, 6.82, WIDTH - 0.8, 0.3, text, 11.5, "#9AA7B8" if dark else MUTED)

    def notes(self, s, en, es):
        s.notes_slide.notes_text_frame.text = self.L(en, es)

    def image(self, s, name, top=2.2, max_h=4.4, max_w=WIDTH, left=None):
        path = self.img / f"{name}.png"
        with Image.open(path) as im:
            ratio = im.width / im.height
        w = min(max_w, max_h * ratio)
        h = w / ratio
        x = left if left is not None else LEFT + (max_w - w) / 2
        s.shapes.add_picture(str(path), Inches(x), Inches(top), Inches(w), Inches(h))
        return h

    def table(self, s, x, y, w, rows, col_w, size=13, row_h=0.42, highlight=None):
        shape = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
        tbl = shape.table
        for j, cw in enumerate(col_w):
            tbl.columns[j].width = Inches(cw)
        for i, row in enumerate(rows):
            tbl.rows[i].height = Inches(row_h)
            for j, val in enumerate(row):
                c = tbl.cell(i, j)
                c.margin_left = c.margin_right = Inches(0.1)
                c.margin_top = c.margin_bottom = Inches(0.04)
                c.vertical_anchor = MSO_ANCHOR.MIDDLE
                c.fill.solid()
                if i == 0:
                    c.fill.fore_color.rgb = rgb(NAVY)
                elif highlight and (i, j) in highlight:
                    c.fill.fore_color.rgb = rgb(highlight[(i, j)])
                else:
                    c.fill.fore_color.rgb = rgb(CARD if i % 2 else "#F1EFE8")
                tf = c.text_frame
                tf.word_wrap = True
                tf.text = str(val)
                for p in tf.paragraphs:
                    for r in p.runs:
                        r.font.name, r.font.size = SANS, Pt(size)
                        r.font.bold = i == 0 or j == 0
                        r.font.color.rgb = rgb(CREAM if i == 0 else (NAVY if j == 0 else BODY))
        return tbl

    def chips(self, s, x, y, labels, dark=True):
        for lab in labels:
            w = 0.3 + len(lab) * 0.095
            self.rect(s, x, y, w, 0.44, "#1E2B40" if dark else PEACHBG)
            self.text(s, x, y, w, 0.44, lab, 14, CREAM if dark else NAVY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
            x += w + 0.16

    def opener(self, kicker, title, subtitle):
        s = self.slide(kicker, title, dark=True, title_size=36)
        s.shapes.title.top = Inches(2.2)
        s.shapes[1].top = Inches(1.7)
        self.rect(s, 0, 0, 0.17, 7.5, ORANGE, shape=MSO_SHAPE.RECTANGLE)
        self.text(s, LEFT, 3.6, 10.8, 1.4, subtitle, 20, "#D9DEE6")
        return s

    def save(self):
        props = self.prs.core_properties
        props.title = self.L("Self-supervised features for pathology", "Features autosupervisadas para patología")
        props.author, props.last_modified_by, props.comments = "", "", ""
        out = HERE / f"SSL_Pathology_{self.lang.upper()}.pptx"
        self.prs.save(out)
        return out


# ---------------------------------------------------------------------------------- numbers
def load_numbers():
    probe = pd.read_csv(RESULTS / "probe_nct.csv")
    anomaly = pd.read_csv(RESULTS / "anomaly.csv")
    centring = pd.read_csv(RESULTS / "anomaly_centring.csv").set_index("backbone")
    acc = probe[probe["metric"] == "accuracy"]

    def p(method, frac, bb):
        r = acc[(acc["method"] == method) & (acc["label_fraction"].round(3) == frac) & (acc["backbone"] == bb)]
        return float(r["value"].iloc[0])

    def a(task, metric, bb, head="knn_distance"):
        r = anomaly[(anomaly["task"] == task) & (anomaly["metric"] == metric) & (anomaly["backbone"] == bb)
                    & (anomaly["head"] == head)]
        return r.iloc[0]

    return p, a, centring


R, S, D = "random_resnet50", "sup_resnet50_in1k", "dinov2_vits14"


def build(lang: str):
    d = Deck(lang)
    L, n = d.L, d.n
    p, a, cen = load_numbers()

    # 1 · Title -------------------------------------------------------------------------
    s = d.slide(L("SELF-SUPERVISED LEARNING · DIGITAL PATHOLOGY · WORKSHOP", "APRENDIZAJE AUTOSUPERVISADO · PATOLOGÍA DIGITAL · TALLER"),
                L("Can a model learn what tissue looks like without labels?",
                  "¿Puede un modelo aprender cómo es un tejido sin etiquetas?"), dark=True, title_size=40)
    s.shapes.title.top, s.shapes.title.height = Inches(1.95), Inches(2.2)
    s.shapes[1].top = Inches(1.45)
    d.rect(s, 0, 0, 0.17, 7.5, ORANGE, shape=MSO_SHAPE.RECTANGLE)
    d.text(s, LEFT, 3.85, 10.8, 0.9, L(
        "Frozen self-supervised backbones on public histology tiles: how to measure them, how to use them to flag what is "
        "not normal, and how to decide fairly whether a new one is better.",
        "Backbones autosupervisados congelados sobre teselas de histología públicas: cómo medirlos, cómo usarlos para "
        "señalar lo que no es normal y cómo decidir con rigor si uno nuevo es mejor."), 20, "#D9DEE6")
    d.chips(s, LEFT, 5.15, [L("3 frozen backbones", "3 backbones congelados"), L("2 anomaly tasks", "2 tareas de anomalías"),
                            L("public data only", "solo datos públicos")])
    d.footer(s, L(f"Notebooks, guides (EN/ES) and figures: {REPO_URL}", f"Notebooks, guías (EN/ES) y figuras: {REPO_URL}"), dark=True)
    d.notes(s, "This workshop is about one practical question in digital pathology. Labelling tissue is slow and expensive, so can "
            "a model learn what tissue looks like from unlabelled images, and can we use what it learned to flag tissue that is "
            "not normal? Along the way we will meet a second question that matters just as much in practice: when a new, bigger "
            "model appears, how do we know whether it is really better? Everything runs on public data and free Colab GPUs.",
            "Este taller trata una pregunta práctica de la patología digital. Etiquetar tejido es lento y caro, así que: ¿puede "
            "un modelo aprender cómo es un tejido a partir de imágenes sin etiquetar, y podemos usar lo aprendido para señalar "
            "tejido que no es normal? Por el camino aparece una segunda pregunta igual de importante en la práctica: cuando sale "
            "un modelo nuevo y más grande, ¿cómo sabemos si de verdad es mejor? Todo funciona con datos públicos y GPU gratuitas de Colab.")

    # 2 · The question -------------------------------------------------------------------
    s = d.slide(L("THE PROBLEM", "EL PROBLEMA"), L("Labels are the bottleneck, not images", "El cuello de botella son las etiquetas, no las imágenes"))
    cards = [
        (L("Millions of tiles", "Millones de teselas"), L(
            "One whole-slide image at 20x cuts into tens of thousands of 224-pixel tiles. A study has hundreds of slides.",
            "Una sola preparación escaneada a 20x se corta en decenas de miles de teselas de 224 píxeles. Un estudio tiene cientos de preparaciones.")),
        (L("Few annotations", "Pocas anotaciones"), L(
            "A pathologist's time is the scarce resource. Tile-level labels for everything will never exist.",
            "El tiempo del patólogo es el recurso escaso. Nunca habrá etiquetas de cada tesela.")),
        (L("Two questions", "Dos preguntas"), L(
            "Can features learned without labels describe tissue? And when a new backbone appears, is it really better?",
            "¿Pueden unas features aprendidas sin etiquetas describir el tejido? Y cuando aparece un backbone nuevo, ¿es realmente mejor?")),
    ]
    for i, (t, b) in enumerate(cards):
        d.card(s, LEFT + i * 3.95, 2.25, 3.65, 2.55, t, b, dark=(i == 2), tsize=19, bsize=15)
    d.band(s, 5.15, 1.15, [(L("Self-supervised learning: ", "Aprendizaje autosupervisado: "), True), (L(
        "the model builds its own training signal from unlabelled images. Labels are needed only at the end, and only a few.",
        "el modelo construye su propia señal de entrenamiento a partir de imágenes sin etiquetar. Las etiquetas solo hacen falta al final, y pocas."), False)])
    d.notes(s, "Images are cheap; labels are not. A single slide gives tens of thousands of tiles, and no pathologist can label them "
            "all. Self-supervised learning changes where the effort goes: the model learns a representation from the images "
            "themselves, and we only need a small number of labels to read that representation for a specific task.",
            "Las imágenes son baratas; las etiquetas no. Una sola preparación da decenas de miles de teselas y ningún patólogo "
            "puede etiquetarlas todas. El aprendizaje autosupervisado cambia dónde se pone el esfuerzo: el modelo aprende una "
            "representación a partir de las propias imágenes, y solo necesitamos unas pocas etiquetas para leer esa "
            "representación en una tarea concreta.")

    # 3 · The loop ----------------------------------------------------------------------
    s = d.slide(L("THE METHOD", "EL MÉTODO"), L("One loop: tiles, frozen backbone, cache, same head, compare",
                                                  "Un bucle: teselas, backbone congelado, caché, misma cabeza, comparar"))
    d.image(s, "fig02_loop", top=2.35, max_h=3.1)
    d.band(s, 5.65, 0.9, [(L("The rule of the whole workshop: ", "La regla de todo el taller: "), True), (L(
        "change the backbone, keep everything else fixed. Any difference is then a difference in the representation.",
        "se cambia el backbone y todo lo demás queda fijo. Así, cualquier diferencia es una diferencia de representación."), False)])
    d.notes(s, "This is the loop we will use for every result. Tiles go through a backbone whose weights never change. The features "
            "are saved to disk once, so every later experiment reads them in seconds. A small head, a nearest-neighbour search, "
            "a logistic regression or an anomaly score, is fitted on top. To compare two backbones we swap only the backbone and "
            "re-run the same head on the same tiles with the same seeds.",
            "Este es el bucle que usaremos para todos los resultados. Las teselas pasan por un backbone cuyos pesos nunca cambian. "
            "Las features se guardan en disco una vez, así que cada experimento posterior las lee en segundos. Encima se ajusta "
            "una cabeza pequeña: una búsqueda de vecinos, una regresión logística o una puntuación de anomalía. Para comparar dos "
            "backbones cambiamos solo el backbone y repetimos la misma cabeza sobre las mismas teselas con las mismas semillas.")

    # 4 · Agenda ------------------------------------------------------------------------
    s = d.slide(L("AGENDA", "AGENDA"), L("Three acts, one story", "Tres actos, una historia"))
    acts = [
        (L("1 · Learn without labels", "1 · Aprender sin etiquetas"), L(
            "Why SSL. Contrastive, self-distillation, masking. BYOL and DINO in one picture each. DINOv2, the backbone we run.",
            "Por qué SSL. Contrastivo, autodestilación, enmascarado. BYOL y DINO en una imagen cada uno. DINOv2, el backbone que usamos.")),
        (L("2 · Measure the features", "2 · Medir las features"), L(
            "kNN and linear probe on frozen features. Label efficiency: 1%, 10% and 100% of the labels.",
            "kNN y sonda lineal sobre features congeladas. Eficiencia de etiquetas: 1 %, 10 % y 100 % de las etiquetas.")),
        (L("3 · Detect and decide", "3 · Detectar y decidir"), L(
            "Learn normal tissue, flag the rest. A rule written before the numbers, and the bug that almost became the headline.",
            "Aprender el tejido normal y señalar el resto. Una regla escrita antes de ver los números y el error que casi fue el titular.")),
    ]
    for i, (t, b) in enumerate(acts):
        d.card(s, LEFT + i * 3.95, 2.25, 3.65, 2.9, t, b, dark=(i == 2), tsize=18, bsize=14.5)
    d.footer(s, L("Notebooks 00, 03, 04 and 06. BYOL and DINO training (01, 02), attention maps (05) and whole slides (07) come in v2.",
                  "Notebooks 00, 03, 04 y 06. El entrenamiento de BYOL y DINO (01, 02), los mapas de atención (05) y las preparaciones completas (07) llegan en la v2."))
    d.notes(s, "Three acts. First the ideas: how a network can learn from images with no labels. Second, how to measure what it "
            "learned without fine-tuning it. Third, how to use those features to find tissue that is not normal, and how to "
            "decide fairly between backbones. The third act contains the most useful ten minutes of the session: a result that "
            "looked spectacular and was a bug.",
            "Tres actos. Primero las ideas: cómo puede aprender una red a partir de imágenes sin etiquetas. Segundo, cómo medir "
            "lo que ha aprendido sin reentrenarla. Tercero, cómo usar esas features para encontrar tejido que no es normal y cómo "
            "decidir con rigor entre backbones. El tercer acto contiene los diez minutos más útiles de la sesión: un resultado "
            "que parecía espectacular y era un error.")

    # 5 · Data --------------------------------------------------------------------------
    s = d.slide(L("THE DATA · NCT-CRC-HE · CC BY 4.0", "LOS DATOS · NCT-CRC-HE · CC BY 4.0"),
                L("Nine tissue classes from colorectal cancer, two groups of patients", "Nueve clases de tejido de cáncer colorrectal, dos grupos de pacientes"))
    d.image(s, "fig01_class_grid", top=2.2, max_h=3.6)
    d.band(s, 5.95, 0.7, [(L("Train: ", "Entrenamiento: "), True), (L("10,000 tiles from NCT-CRC-HE-100K.  ", "10.000 teselas de NCT-CRC-HE-100K.  "), False),
                          (L("Test: ", "Test: "), True), (L("2,000 from CRC-VAL-HE-7K, different patients.  ", "2.000 de CRC-VAL-HE-7K, otros pacientes.  "), False),
                          (L("Task B: ", "Tarea B: "), True), (L("PatchCamelyon, lymph-node metastasis (CC0).", "PatchCamelyon, metástasis en ganglio (CC0)."), False)], size=13.5)
    d.footer(s, L("Kather et al., Zenodo 1214456. Tiles are 224 x 224 px at 0.5 µm/px, H&E stained.",
                  "Kather et al., Zenodo 1214456. Teselas de 224 x 224 px a 0,5 µm/px, tinción H&E."))
    d.notes(s, "The data is public: Kather and colleagues' colorectal cancer tiles, nine classes, from fat and background to tumour "
            "epithelium. The test set comes from different patients than the training set, which is the honest way to "
            "evaluate. For the anomaly part, tumour epithelium plays the role of 'not normal', and background is kept apart "
            "because empty glass is trivially unusual and is not cancer. A second task uses PatchCamelyon, lymph-node patches "
            "with or without metastasis.",
            "Los datos son públicos: las teselas de cáncer colorrectal de Kather y colaboradores, nueve clases, desde grasa y "
            "fondo hasta epitelio tumoral. El conjunto de test procede de pacientes distintos a los de entrenamiento, que es la "
            "forma honesta de evaluar. En la parte de anomalías, el epitelio tumoral hace el papel de 'no normal', y el fondo se "
            "deja aparte porque el cristal vacío es trivialmente raro y no es cáncer. Una segunda tarea usa PatchCamelyon, "
            "parches de ganglio linfático con o sin metástasis.")

    # ACT 1 ------------------------------------------------------------------------------
    d.opener(L("ACT 1", "ACTO 1"), L("Learn without labels", "Aprender sin etiquetas"),
             L("Where self-supervised backbones come from, and which one we actually run.",
               "De dónde salen los backbones autosupervisados y cuál usamos realmente."))
    d.prefix = L("ACT 1 · ", "ACTO 1 · ")

    s = d.slide(L("THREE WAYS TO TRAIN", "TRES FORMAS DE ENTRENAR"), L("Supervised, self-supervised, foundation model",
                                                                         "Supervisado, autosupervisado, modelo fundacional"))
    cols = [
        (L("Supervised", "Supervisado"), L(
            "Learns from labels: 'this is a cat'. ImageNet ResNet-50 learned 1,000 object classes. Strong, but tied to what was labelled.",
            "Aprende de etiquetas: 'esto es un gato'. La ResNet-50 de ImageNet aprendió 1.000 clases de objetos. Potente, pero atada a lo etiquetado."), False),
        (L("Self-supervised", "Autosupervisado"), L(
            "Learns from the images themselves: two crops of the same image should look alike to the network. No labels.",
            "Aprende de las propias imágenes: dos recortes de la misma imagen deben parecerse para la red. Sin etiquetas."), True),
        (L("Foundation model", "Modelo fundacional"), L(
            "A self-supervised backbone trained on a very large dataset, reused frozen for many tasks. DINOv2, Phikon, UNI.",
            "Un backbone autosupervisado entrenado con un conjunto enorme y reutilizado congelado en muchas tareas. DINOv2, Phikon, UNI."), False),
    ]
    for i, (t, b, dark) in enumerate(cols):
        d.card(s, LEFT + i * 3.95, 2.25, 3.65, 2.7, t, b, dark=dark, tsize=19, bsize=15)
    d.band(s, 5.3, 0.95, [(L("In this workshop: ", "En este taller: "), True), (L(
        "a random ResNet-50 (the floor), an ImageNet supervised ResNet-50 and DINOv2 ViT-S/14. None saw these tissue tiles.",
        "una ResNet-50 aleatoria (el suelo), una ResNet-50 supervisada en ImageNet y DINOv2 ViT-S/14. Ninguno vio estas teselas."), False)])
    d.notes(s, "Three ways a backbone can learn. Supervised: from human labels, such as ImageNet's thousand classes. Self-supervised: "
            "from the structure of the images themselves, with no labels. A foundation model is a self-supervised backbone "
            "trained at scale and reused frozen. We compare three backbones. The random one is not a competitor: it is the floor, "
            "the score you get from architecture alone. If it ever wins, something is wrong in the pipeline.",
            "Tres formas de que un backbone aprenda. Supervisado: a partir de etiquetas humanas, como las mil clases de ImageNet. "
            "Autosupervisado: a partir de la estructura de las propias imágenes, sin etiquetas. Un modelo fundacional es un "
            "backbone autosupervisado entrenado a gran escala y reutilizado congelado. Comparamos tres backbones. El aleatorio "
            "no compite: es el suelo, la puntuación que da la arquitectura por sí sola. Si alguna vez gana, algo falla en el proceso.")

    s = d.slide(L("THE SSL FAMILY TREE", "EL ÁRBOL DE FAMILIA DEL SSL"), L("Three families: contrast, distil, mask",
                                                                            "Tres familias: contrastar, destilar, enmascarar"))
    d.image(s, "fig03_ssl_family", top=2.25, max_h=3.4)
    d.band(s, 5.85, 0.8, [(L("Collapse ", "El colapso "), True), (L(
        "is the common enemy: a network that outputs the same vector for every image matches its views perfectly and learns nothing.",
        "es el enemigo común: una red que devuelve el mismo vector para todas las imágenes iguala sus vistas a la perfección y no aprende nada."), False)])
    d.notes(s, "Most self-supervised methods fall into three families. Contrastive methods such as SimCLR and MoCo pull two views of "
            "the same image together and push other images away; they need many negatives. Self-distillation methods such as BYOL "
            "and DINO drop the negatives: one network learns to match another. Masking methods such as MAE hide parts of the image "
            "and predict them. All of them must avoid collapse, the trivial solution where every image maps to the same point.",
            "La mayoría de los métodos autosupervisados caen en tres familias. Los contrastivos, como SimCLR y MoCo, acercan dos "
            "vistas de la misma imagen y alejan las demás; necesitan muchos negativos. Los de autodestilación, como BYOL y DINO, "
            "prescinden de los negativos: una red aprende a imitar a otra. Los de enmascarado, como MAE, ocultan partes de la "
            "imagen y las predicen. Todos deben evitar el colapso, la solución trivial en la que todas las imágenes caen en el mismo punto.")

    s = d.slide("BYOL · GRILL ET AL., 2020", L("BYOL: no negatives, a predictor and a slow-moving target",
                                                  "BYOL: sin negativos, con un predictor y un objetivo que se mueve despacio"))
    d.image(s, "fig04_byol", top=2.25, max_h=3.5)
    d.band(s, 5.95, 0.75, [(L("Why it does not collapse: ", "Por qué no colapsa: "), True), (L(
        "the predictor is only on one side, and the target is an exponential moving average of the online network.",
        "el predictor solo está en un lado, y el objetivo es una media móvil exponencial de la red online."), False)], size=14)
    d.footer(s, L("Trained from scratch in notebook 01 (v2). Reference implementation: github.com/olonok69/BYOL.",
                  "Se entrenará desde cero en el notebook 01 (v2). Implementación de referencia: github.com/olonok69/BYOL."))
    d.notes(s, "BYOL takes two augmented views of the same image. The online network encodes one view, projects it and predicts the "
            "target network's projection of the other view. The target is not trained by gradient: its weights are a slow moving "
            "average of the online weights. That asymmetry, a predictor on one side and a slow target on the other, is what keeps "
            "it from collapsing, even though there are no negative examples at all. I trained BYOL from scratch on CIFAR and STL "
            "in a public repository; in v2 of this workshop it is notebook 01, on tissue tiles.",
            "BYOL toma dos vistas aumentadas de la misma imagen. La red online codifica una vista, la proyecta y predice la "
            "proyección que hace la red objetivo de la otra vista. El objetivo no se entrena por gradiente: sus pesos son una "
            "media móvil lenta de los pesos online. Esa asimetría, un predictor en un lado y un objetivo lento en el otro, es lo "
            "que evita el colapso aunque no haya ningún ejemplo negativo. Entrené BYOL desde cero con CIFAR y STL en un "
            "repositorio público; en la v2 del taller será el notebook 01, con teselas de tejido.")

    s = d.slide("DINO · CARON ET AL., 2021", L("DINO: a student learns to agree with its own slow-moving teacher",
                                                  "DINO: un alumno aprende a coincidir con su propio profesor lento"))
    d.image(s, "fig05_dino", top=2.25, max_h=3.5)
    d.band(s, 5.95, 0.75, [(L("Multi-crop: ", "Multirrecorte: "), True), (L(
        "the teacher sees only large crops, the student also small ones, so the student learns local-to-global correspondence.",
        "el profesor solo ve recortes grandes y el alumno también pequeños, así aprende la correspondencia de lo local a lo global."), False)], size=14)
    d.notes(s, "DINO is self-distillation with no labels. A student and a teacher network have the same architecture, a Vision "
            "Transformer. The teacher sees two large crops; the student sees those plus several small crops, and must produce the "
            "same probability distribution as the teacher. The teacher's weights are a moving average of the student's. Two "
            "tricks prevent collapse: centring the teacher's output so no single dimension dominates, and sharpening it with a "
            "low temperature so it does not become uniform.",
            "DINO es autodestilación sin etiquetas. Un alumno y un profesor tienen la misma arquitectura, un Vision Transformer. "
            "El profesor ve dos recortes grandes; el alumno ve esos y varios recortes pequeños, y debe producir la misma "
            "distribución de probabilidad que el profesor. Los pesos del profesor son una media móvil de los del alumno. Dos "
            "trucos evitan el colapso: centrar la salida del profesor para que ninguna dimensión domine y afilarla con una "
            "temperatura baja para que no se vuelva uniforme.")

    s = d.slide("DINOV2 · OQUAB ET AL., 2023", L("DINOv2 ViT-S/14: the backbone we run, frozen", "DINOv2 ViT-S/14: el backbone que usamos, congelado"))
    items = [
        (L("What it adds to DINO", "Qué añade a DINO"), L(
            "A masked-patch objective from iBOT, and a curated set of 142 million natural images. No histology.",
            "Un objetivo de parches enmascarados de iBOT y un conjunto curado de 142 millones de imágenes naturales. Nada de histología.")),
        (L("ViT in one line", "El ViT en una línea"), L(
            "The tile is cut into 14 x 14 px patches. A CLS token attends to all of them; its final state is the tile's embedding (384 numbers).",
            "La tesela se corta en parches de 14 x 14 px. Un token CLS atiende a todos; su estado final es el embedding de la tesela (384 números).")),
        (L("How we use it", "Cómo lo usamos"), L(
            "Frozen, through torch.hub, ImageNet normalisation, 224 px input. One forward pass per tile, saved to disk.",
            "Congelado, mediante torch.hub, normalización de ImageNet y entrada de 224 px. Una pasada por tesela, guardada en disco.")),
    ]
    for i, (t, b) in enumerate(items):
        d.card(s, LEFT + i * 3.95, 2.25, 3.65, 2.75, t, b, dark=(i == 2), tsize=18, bsize=14.5)
    d.band(s, 5.35, 0.95, [(L("Ask of any foundation model: ", "Pregunta a cualquier modelo fundacional: "), True), (L(
        "were the evaluation tiles in its training set? And can I use it commercially? (Phikon: non-commercial. UNI: gated.)",
        "¿estaban las teselas de evaluación en su entrenamiento? ¿Puedo usarlo comercialmente? (Phikon: no comercial. UNI: acceso restringido.)"), False)], size=14)
    d.notes(s, "DINOv2 scales DINO: it adds a masked-patch loss borrowed from iBOT and trains on a curated set of 142 million "
            "images. We use the small Vision Transformer, ViT-S with 14-pixel patches, frozen. Its CLS token, 384 numbers, is "
            "the embedding of each tile. Two questions to ask of any pathology foundation model before trusting a number: was "
            "the evaluation data in its pretraining set, and what does its licence allow? Phikon is non-commercial; UNI requires "
            "approved access.",
            "DINOv2 escala DINO: añade una pérdida de parches enmascarados tomada de iBOT y entrena con un conjunto curado de 142 "
            "millones de imágenes. Usamos el Vision Transformer pequeño, ViT-S con parches de 14 píxeles, congelado. Su token CLS, "
            "384 números, es el embedding de cada tesela. Dos preguntas que hay que hacer a cualquier modelo fundacional de "
            "patología antes de fiarse de un número: ¿estaban los datos de evaluación en su preentrenamiento? ¿Y qué permite su "
            "licencia? Phikon es no comercial; UNI exige acceso aprobado.")

    # ACT 2 ------------------------------------------------------------------------------
    d.prefix = ""
    d.opener(L("ACT 2", "ACTO 2"), L("Measure the features", "Medir las features"),
             L("Judge a backbone without changing a single weight.", "Juzgar un backbone sin tocar ni un solo peso."))
    d.prefix = L("ACT 2 · ", "ACTO 2 · ")

    s = d.slide(L("FROZEN EVALUATION · NOTEBOOK 04", "EVALUACIÓN CONGELADA · NOTEBOOK 04"),
                L(f"Nearest neighbours: DINOv2 {n(p('knn', 1.0, D))}, ImageNet {n(p('knn', 1.0, S))}, random {n(p('knn', 1.0, R))}",
                  f"Vecinos más cercanos: DINOv2 {n(p('knn', 1.0, D))}, ImageNet {n(p('knn', 1.0, S))}, aleatoria {n(p('knn', 1.0, R))}"), )
    d.image(s, "fig06_probe", top=2.2, max_h=3.85)
    d.footer(s, L("results/probe_nct.csv. kNN: cosine, k = 20. Probe: standardised logistic regression, C by 3-fold CV on training rows. Bars: 95% bootstrap CI.",
                  "results/probe_nct.csv. kNN: coseno, k = 20. Sonda: regresión logística estandarizada, C por validación cruzada de 3 particiones. Barras: IC 95 % bootstrap."))
    d.notes(s, "Two ways to read a frozen representation. The k-nearest-neighbour classifier labels a test tile by a vote of its 20 "
            "most similar training tiles: if similar tissue lands close together, it works. The linear probe fits a logistic "
            "regression on the features: if the classes can be separated by a straight cut, it works. Nothing in the backbone "
            f"moves. DINOv2 leads on kNN, {n(p('knn', 1.0, D))} against {n(p('knn', 1.0, S))}. With all the labels the probe "
            f"almost closes the gap: {n(p('linear_probe', 1.0, D))} against {n(p('linear_probe', 1.0, S))}, and the intervals overlap.",
            "Dos formas de leer una representación congelada. El clasificador de k vecinos etiqueta una tesela de test por "
            "votación de sus 20 teselas de entrenamiento más parecidas: si el tejido parecido queda cerca, funciona. La sonda "
            "lineal ajusta una regresión logística sobre las features: si las clases se separan con un corte recto, funciona. "
            f"Nada del backbone se mueve. DINOv2 gana en kNN, {n(p('knn', 1.0, D))} frente a {n(p('knn', 1.0, S))}. Con todas las "
            f"etiquetas la sonda casi cierra la distancia: {n(p('linear_probe', 1.0, D))} frente a {n(p('linear_probe', 1.0, S))}, "
            "y los intervalos se solapan.")

    s = d.slide(L("LABEL EFFICIENCY", "EFICIENCIA DE ETIQUETAS"),
                L(f"With 1% of the labels: {n(p('linear_probe', 0.01, D))} against {n(p('linear_probe', 0.01, S))}",
                  f"Con un 1 % de las etiquetas: {n(p('linear_probe', 0.01, D))} frente a {n(p('linear_probe', 0.01, S))}"))
    d.image(s, "fig07_label_efficiency", top=2.2, max_h=3.6, max_w=7.2, left=LEFT)
    d.card(s, 8.5, 2.3, 3.94, 3.4, L("Read it like this", "Cómo leerlo"), [
        L("1% is about 100 labelled tiles, eleven per class.", "Un 1 % son unas 100 teselas etiquetadas, once por clase."),
        L("The better representation matters most when labels are scarce.", "La mejor representación importa más cuando escasean las etiquetas."),
        L("With 10,000 labels, both pretrained backbones converge.", "Con 10.000 etiquetas, los dos backbones preentrenados convergen.")],
        bsize=14.5, tsize=18)
    d.footer(s, L("results/probe_nct.csv. Shaded bands: 95% bootstrap CI over test tiles.", "results/probe_nct.csv. Bandas: IC 95 % bootstrap sobre teselas de test."))
    d.notes(s, "This is the curve that matters for a pathology lab. On the left, only one percent of the training labels, about a "
            f"hundred tiles. DINOv2 features already reach {n(p('linear_probe', 0.01, D))}; ImageNet features "
            f"{n(p('linear_probe', 0.01, S))}. On the right, all ten thousand labels, and the two converge. The value of a better "
            "representation is largest exactly where labels are expensive.",
            "Esta es la curva que importa a un laboratorio de patología. A la izquierda, solo un uno por ciento de las etiquetas "
            f"de entrenamiento, unas cien teselas. Las features de DINOv2 ya llegan a {n(p('linear_probe', 0.01, D))}; las de "
            f"ImageNet, a {n(p('linear_probe', 0.01, S))}. A la derecha, las diez mil etiquetas, y los dos convergen. El valor de "
            "una representación mejor es máximo justo donde las etiquetas son caras.")

    # ACT 3 ------------------------------------------------------------------------------
    d.prefix = ""
    d.opener(L("ACT 3", "ACTO 3"), L("Detect anomalies and decide", "Detectar anomalías y decidir"),
             L("Learn what normal tissue looks like, flag the rest, and decide fairly between backbones.",
               "Aprender cómo es el tejido normal, señalar el resto y decidir con rigor entre backbones."))
    d.prefix = L("ACT 3 · ", "ACTO 3 · ")

    s = d.slide(L("NORMAL-ONLY ANOMALY DETECTION · NOTEBOOK 06", "DETECCIÓN DE ANOMALÍAS SOLO CON NORMALES · NOTEBOOK 06"),
                L("Fit on normal tissue only; tumours appear only at test time", "Se ajusta solo con tejido normal; los tumores aparecen solo en el test"))
    rows = [[L("", ""), L("Task A · NCT-CRC", "Tarea A · NCT-CRC"), L("Task B · PatchCamelyon", "Tarea B · PatchCamelyon")],
            [L("Normal", "Normal"), L("7 tissue classes, 6,009 fit tiles", "7 clases de tejido, 6.009 teselas de ajuste"),
             L("Lymph node, 2,000 fit patches", "Ganglio linfático, 2.000 parches de ajuste")],
            [L("Anomaly", "Anomalía"), L("Tumour epithelium (344 test tiles)", "Epitelio tumoral (344 teselas de test)"),
             L("Metastasis", "Metástasis")],
            [L("Threshold", "Umbral"), L("99th percentile of 1,502 held-out normals", "Percentil 99 de 1.502 normales reservadas"),
             L("99th percentile of 500 held-out normals", "Percentil 99 de 500 normales reservadas")],
            [L("Kept apart", "Aparte"), L("Background: reported as a flag rate", "Fondo: se informa como tasa de alertas"), "–"]]
    d.table(s, LEFT, 2.25, 7.3, rows, [1.5, 3.0, 2.8], size=12.5, row_h=0.6)
    d.card(s, 8.5, 2.25, 3.94, 3.0, L("Three heads", "Tres cabezas"), [
        L("kNN distance to normal tiles (PatchCore at image level). The head named in the rule.",
          "Distancia kNN a teselas normales (PatchCore a nivel de imagen). La cabeza que fija la regla."),
        L("Gaussian Mahalanobis distance.", "Distancia de Mahalanobis gaussiana."),
        L("Isolation Forest.", "Isolation Forest.")], bsize=13.5, tsize=18)
    d.band(s, 5.55, 0.85, [(L("No tumour is seen ", "Ningún tumor se ve "), True), (L(
        "to fit the head or to choose the threshold. That is what makes it anomaly detection, not classification.",
        "para ajustar la cabeza ni para elegir el umbral. Eso es lo que la convierte en detección de anomalías y no en clasificación."), False)], size=14)
    d.notes(s, "Anomaly detection turns the problem around: we only describe normal tissue. The head is fitted on seven normal "
            "classes; tumour tiles are used only to score the result. The threshold is the score that 99 percent of held-out "
            "normal tiles stay below, so about one percent of normal tissue is flagged by construction. Background is excluded, "
            "because empty glass is unusual but harmless. Three heads; the rule names the nearest-neighbour distance, which is "
            "the image-level version of PatchCore's memory bank.",
            "La detección de anomalías da la vuelta al problema: solo describimos el tejido normal. La cabeza se ajusta con siete "
            "clases normales; las teselas tumorales solo se usan para puntuar el resultado. El umbral es la puntuación por debajo "
            "de la cual queda el 99 por ciento de las teselas normales reservadas, así que, por construcción, se señala alrededor "
            "de un uno por ciento del tejido normal. El fondo se excluye, porque el cristal vacío es raro pero inofensivo. Tres "
            "cabezas; la regla fija la distancia a los vecinos, que es la versión a nivel de imagen del banco de memoria de PatchCore.")

    s = d.slide(L("PRE-REGISTERED DECISION RULE", "REGLA DE DECISIÓN PREREGISTRADA"),
                L("The rule is written down before any score is computed", "La regla se escribe antes de calcular ninguna puntuación"))
    d.rect(s, LEFT, 2.3, WIDTH, 1.6, NAVY)
    d.text(s, LEFT + 0.35, 2.3, WIDTH - 0.7, 1.6, L(
        "\"A backbone is clearly better if its AUROC 95% CI does not overlap the next-best backbone's CI on both Task A and "
        "Task B, for the kNN-distance head. Same tiles, same head, same seeds.\"",
        "\"Un backbone es claramente mejor si su IC 95 % de AUROC no se solapa con el del siguiente mejor en la tarea A y en la "
        "tarea B, con la cabeza de distancia kNN. Mismas teselas, misma cabeza, mismas semillas.\""), 19, CREAM, anchor=MSO_ANCHOR.MIDDLE, italic=True)
    for i, (t, b) in enumerate([
        (L("Why two tasks", "Por qué dos tareas"), L("One dataset can flatter one backbone. Agreement across two is harder to fake.",
                                                   "Un solo conjunto puede favorecer a un backbone. Coincidir en dos es más difícil de fingir.")),
        (L("Why intervals", "Por qué intervalos"), L("A higher point estimate inside an overlapping interval is not evidence.",
                                                   "Una estimación puntual más alta dentro de un intervalo solapado no es evidencia.")),
        (L("Why before", "Por qué antes"), L("So the rule cannot bend to the result we hoped for. It also has to be able to say 'no winner'.",
                                           "Para que la regla no se doble hacia el resultado que esperábamos. También tiene que poder decir 'no hay ganador'.")),
    ]):
        d.card(s, LEFT + i * 3.95, 4.2, 3.65, 2.1, t, b, tsize=17, bsize=14)
    d.footer(s, L("ssl_demo.anomaly.DECISION_RULE, printed in notebook 06 before the fit.", "ssl_demo.anomaly.DECISION_RULE, se imprime en el notebook 06 antes del ajuste."))
    d.notes(s, "Before computing a single anomaly score, the notebook prints this rule. A backbone wins only if its confidence "
            "interval clears the next best on both tasks. Writing it first protects us from the temptation to pick whichever "
            "metric makes our favourite look good, and it forces us to accept 'no clear winner' as a valid outcome.",
            "Antes de calcular una sola puntuación de anomalía, el notebook imprime esta regla. Un backbone gana solo si su "
            "intervalo de confianza supera al del siguiente mejor en las dos tareas. Escribirla antes nos protege de la tentación "
            "de elegir la métrica que favorece a nuestro preferido y nos obliga a aceptar 'no hay un ganador claro' como "
            "resultado válido.")

    s = d.slide(L("FIRST RUN", "PRIMERA EJECUCIÓN"), L("The untrained network wins. Something is wrong.", "Gana la red sin entrenar. Algo va mal."))
    rows = [[L("Backbone", "Backbone"), L("Task A AUROC, raw cosine", "AUROC tarea A, coseno sin centrar")],
            [L("Random ResNet-50", "ResNet-50 aleatoria"), n(cen.loc[R, "auroc_raw_cosine"])],
            [L("ImageNet ResNet-50", "ResNet-50 ImageNet"), n(cen.loc[S, "auroc_raw_cosine"])],
            ["DINOv2 ViT-S/14", n(cen.loc[D, "auroc_raw_cosine"])]]
    d.table(s, LEFT, 2.35, 6.0, rows, [3.0, 3.0], size=15, row_h=0.62, highlight={(1, 1): REDBG})
    d.card(s, 7.3, 2.35, 5.14, 2.5, L("The tempting headline", "El titular tentador"), L(
        "\"Pretrained features are worse than random for anomaly detection.\" The first version of the decision rule even "
        "printed 'random_resnet50 is clearly better on both tasks'.",
        "\"Las features preentrenadas son peores que las aleatorias para detectar anomalías.\" La primera versión de la regla "
        "llegó a imprimir 'random_resnet50 es claramente mejor en las dos tareas'."), fill=REDBG, tsize=18, bsize=14.5)
    d.band(s, 5.25, 1.0, [(L("The rule of thumb: ", "La regla práctica: "), True), (L(
        "a random network beating a trained one is a pipeline bug until proven otherwise.",
        "si una red aleatoria gana a una entrenada, es un error del proceso hasta que se demuestre lo contrario."), False)], fill=NAVY, color=CREAM, size=16)
    d.footer(s, L("results/anomaly_centring.csv, raw-cosine column. Same tiles and fit split as the final result.",
                  "results/anomaly_centring.csv, columna de coseno sin centrar. Mismas teselas y partición que el resultado final."))
    d.notes(s, f"This is what the first run of notebook 06 said. The untrained network scored {n(cen.loc[R, 'auroc_raw_cosine'])} on "
            f"Task A; DINOv2 {n(cen.loc[D, 'auroc_raw_cosine'])}. The decision rule, applied mechanically, declared the random "
            "network the winner. It would have been easy to write a provocative post about it. Instead, the right reaction is "
            "suspicion: a network that has learned nothing should be the floor.",
            "Esto es lo que dijo la primera ejecución del notebook 06. La red sin entrenar obtuvo "
            f"{n(cen.loc[R, 'auroc_raw_cosine'])} en la tarea A; DINOv2, {n(cen.loc[D, 'auroc_raw_cosine'])}. La regla de decisión, "
            "aplicada mecánicamente, declaró ganadora a la red aleatoria. Habría sido fácil escribir un artículo provocador. La "
            "reacción correcta es la sospecha: una red que no ha aprendido nada debería ser el suelo.")

    s = d.slide(L("DIAGNOSIS", "DIAGNÓSTICO"), L("Every random vector points the same way", "Todos los vectores aleatorios apuntan en la misma dirección"))
    d.image(s, "fig12_geometry", top=2.15, max_h=2.75)
    d.image(s, "fig08b_centroid_cosine", top=4.95, max_h=1.75, max_w=6.2, left=LEFT)
    d.card(s, 7.5, 5.0, 4.94, 1.65, L("What cosine distance sees", "Qué ve la distancia coseno"), L(
        "Only direction. If every tile shares one direction, what is left is a tiny residue: here, stain intensity.",
        "Solo la dirección. Si todas las teselas comparten una dirección, queda un residuo diminuto: aquí, la intensidad de tinción."), tsize=15, bsize=13)
    d.footer(s, L("Top: synthetic 2-D illustration. Bottom: results/anomaly_centring.csv.", "Arriba: ilustración sintética en 2D. Abajo: results/anomaly_centring.csv."))
    d.notes(s, "Cosine distance compares directions. For the random network, every tile's feature vector points almost exactly the "
            f"same way: the mean cosine to the centroid is {n(cen.loc[R, 'centroid_cosine'])}. What is left to compare is a "
            "tiny residue that tracks how dark the stain is, and dense purple tumour tiles stand out on that. Pretrained "
            f"features also share a large common direction, cosine about {n(cen.loc[D, 'centroid_cosine'], 2)}, which squeezes "
            "all their distances into a narrow band where nuisance variation decides the ranking.",
            "La distancia coseno compara direcciones. En la red aleatoria, el vector de features de cada tesela apunta casi "
            f"exactamente en la misma dirección: el coseno medio con el centroide es {n(cen.loc[R, 'centroid_cosine'])}. Lo que "
            "queda por comparar es un residuo diminuto que sigue lo oscura que es la tinción, y las teselas tumorales, densas y "
            "moradas, destacan en eso. Las features preentrenadas también comparten una dirección común grande, coseno de "
            f"alrededor de {n(cen.loc[D, 'centroid_cosine'], 2)}, que comprime todas sus distancias en una franja estrecha donde "
            "decide la variación irrelevante.")

    s = d.slide(L("THE FIX", "LA CORRECCIÓN"),
                L(f"Centre on the normal mean: random falls to {n(cen.loc[R, 'auroc_centred_cosine'], 2)}, DINOv2 rises to {n(cen.loc[D, 'auroc_centred_cosine'])}",
                  f"Centrar en la media normal: la aleatoria cae a {n(cen.loc[R, 'auroc_centred_cosine'], 2)} y DINOv2 sube a {n(cen.loc[D, 'auroc_centred_cosine'])}"), )
    d.image(s, "fig08_centring", top=2.2, max_h=3.75, max_w=7.2, left=LEFT)
    d.card(s, 8.5, 2.3, 3.94, 3.55, L("One line of code", "Una línea de código"), [
        L("Subtract the mean of the normal fit features, then compute the cosine. Bank and queries alike.",
          "Restar la media de las features normales de ajuste y después calcular el coseno. Igual en el banco y en las consultas."),
        L("Now the head reads tissue structure, not the shared direction.", "Ahora la cabeza lee estructura de tejido, no la dirección común."),
        L("Default in KNNDistanceHead(center=True).", "Por defecto en KNNDistanceHead(center=True).")], bsize=13.5, tsize=18)
    d.footer(s, L("results/anomaly_centring.csv. Notebook 06 shows both versions side by side.", "results/anomaly_centring.csv. El notebook 06 muestra las dos versiones lado a lado."))
    d.notes(s, "The fix is one line: subtract the mean of the normal training features before computing the cosine. That removes "
            "the shared direction and leaves the structure that distinguishes one tissue from another. The untrained network "
            f"falls to {n(cen.loc[R, 'auroc_centred_cosine'])}, which is chance, exactly where a floor should be. ImageNet rises "
            f"to {n(cen.loc[S, 'auroc_centred_cosine'])} and DINOv2 to {n(cen.loc[D, 'auroc_centred_cosine'])}.",
            "La corrección es una línea: restar la media de las features normales de entrenamiento antes de calcular el coseno. "
            "Eso elimina la dirección común y deja la estructura que distingue un tejido de otro. La red sin entrenar cae a "
            f"{n(cen.loc[R, 'auroc_centred_cosine'])}, es decir, al azar, justo donde debe estar un suelo. ImageNet sube a "
            f"{n(cen.loc[S, 'auroc_centred_cosine'])} y DINOv2 a {n(cen.loc[D, 'auroc_centred_cosine'])}.")

    s = d.slide(L("HOW WE KNEW", "CÓMO LO SUPIMOS"), L("The information was in the features all along", "La información estaba en las features desde el principio"))
    checks = [
        (L("Supervised ceiling", "Techo supervisado"), L(
            "A logistic regression, tumour against normal, on the same cached features: 0.986 to 0.993 AUROC for both pretrained backbones, 0.87 for random.",
            "Una regresión logística, tumor frente a normal, sobre las mismas features: AUROC de 0,986 a 0,993 en los dos preentrenados y 0,87 en el aleatorio."), GREENBG),
        (L("Head settings", "Ajustes de la cabeza"), L(
            "Larger k, no coreset, class-conditional kNN: small changes. Centring: the whole effect.",
            "k mayor, sin coreset, kNN condicionado por clase: cambios pequeños. Centrar: todo el efecto."), CARD),
        (L("Domain shift", "Cambio de dominio"), L(
            "Real but secondary: on raw cosine, 3 to 8% of test normals crossed a 1% threshold; after centring, 0.6 to 1.2%.",
            "Real pero secundario: con coseno sin centrar, del 3 al 8 % de los normales de test cruzaban un umbral del 1 %; tras centrar, del 0,6 al 1,2 %."), CARD),
    ]
    for i, (t, b, fill) in enumerate(checks):
        d.card(s, LEFT + i * 3.95, 2.25, 3.65, 2.85, t, b, fill=fill, tsize=18, bsize=14.5)
    d.band(s, 5.45, 0.9, [(L("Debugging order: ", "Orden de depuración: "), True), (L(
        "is the signal in the features? Then is the head reading it? Then is the data shifting?",
        "¿está la señal en las features? Después, ¿la lee la cabeza? Después, ¿se desplazan los datos?"), False)], size=15)
    d.footer(s, L("Diagnostics run on the cached fast-mode features; summarised in docs/brief.md.", "Diagnósticos sobre las features en caché del modo rápido; resumidos en docs/brief.md."))
    d.notes(s, "How did we find it? First question: is the information in the features at all? A supervised classifier on the same "
            "features separates tumour from normal at 0.99 AUROC, so yes. Second question: is the head reading it? Changing k "
            "or the memory bank made little difference; centring made all of it. Third: is the test data shifted? A little, "
            "and centring also removes most of that. This order of questions is reusable in any embedding pipeline.",
            "¿Cómo lo encontramos? Primera pregunta: ¿está la información en las features? Un clasificador supervisado sobre las "
            "mismas features separa tumor de normal con un AUROC de 0,99, así que sí. Segunda pregunta: ¿la lee la cabeza? Cambiar "
            "k o el banco de memoria apenas cambió nada; centrar lo cambió todo. Tercera: ¿están desplazados los datos de test? Un "
            "poco, y centrar también elimina casi todo eso. Este orden de preguntas sirve para cualquier proceso basado en embeddings.")

    a_d, a_s, b_d, b_s = a("task_a", "auroc", D), a("task_a", "auroc", S), a("task_b", "auroc", D), a("task_b", "auroc", S)
    s = d.slide(L("BACKBONE SWAP · THE DECISION", "CAMBIO DE BACKBONE · LA DECISIÓN"),
                L("Task A: DINOv2 clearly ahead. Task B: a tie. No winner declared.",
                  "Tarea A: DINOv2 claramente por delante. Tarea B: empate. No hay ganador."))
    d.image(s, "fig09_backbone_swap", top=2.2, max_h=3.55)
    d.band(s, 5.95, 0.75, [(L("Rule output: ", "Salida de la regla: "), True), (L(
        f"Task A, DINOv2 {n(a_d.value)} [{n(a_d.ci_low)}, {n(a_d.ci_high)}] clears ImageNet [{n(a_s.ci_low)}, {n(a_s.ci_high)}]. "
        f"Task B, ImageNet {n(b_s.value)} overlaps DINOv2 {n(b_d.value)}.",
        f"Tarea A, DINOv2 {n(a_d.value)} [{n(a_d.ci_low)}; {n(a_d.ci_high)}] supera a ImageNet [{n(a_s.ci_low)}; {n(a_s.ci_high)}]. "
        f"Tarea B, ImageNet {n(b_s.value)} se solapa con DINOv2 {n(b_d.value)}."), False)], size=13.5)
    d.footer(s, L("results/anomaly.csv, kNN-distance head. PCam labels cover only the central 32 x 32 px of each patch.",
                  "results/anomaly.csv, cabeza de distancia kNN. Las etiquetas de PCam solo cubren los 32 x 32 px centrales de cada parche."))
    d.notes(s, "Now the honest comparison. Same head, same tiles, only the backbone changes. On colorectal tissue DINOv2 is clearly "
            "ahead: its interval does not touch ImageNet's. On lymph-node metastasis both pretrained backbones sit around 0.75 "
            "and their intervals overlap. PatchCamelyon is harder because the label refers only to the centre of each patch, and "
            "a global embedding dilutes a small tumour focus. So the rule says: no backbone is clearly better on both tasks. That "
            "is a useful answer, not a disappointing one.",
            "Ahora, la comparación honesta. Misma cabeza, mismas teselas, solo cambia el backbone. En tejido colorrectal DINOv2 va "
            "claramente por delante: su intervalo no toca el de ImageNet. En metástasis de ganglio, los dos preentrenados están "
            "alrededor de 0,75 y sus intervalos se solapan. PatchCamelyon es más difícil porque la etiqueta se refiere solo al "
            "centro de cada parche, y un embedding global diluye un foco tumoral pequeño. Así que la regla dice: ningún backbone "
            "es claramente mejor en las dos tareas. Es una respuesta útil, no decepcionante.")

    sens, fpr, bg = a("task_a", "sensitivity_at_99", D), a("task_a", "fpr_at_99", D), a("task_a_background", "flag_rate", D)
    s = d.slide(L("OPERATING POINT", "PUNTO DE OPERACIÓN"),
                L("A 1% false-alarm budget catches about a quarter of tumour tiles", "Un presupuesto de falsas alarmas del 1 % detecta una cuarta parte de las teselas tumorales"), )
    rows = [[L("DINOv2, Task A", "DINOv2, tarea A"), L("Value", "Valor")],
            [L("False alarms on normal tissue", "Falsas alarmas en tejido normal"), n(fpr.value)],
            [L("Tumour tiles flagged", "Teselas tumorales señaladas"), n(sens.value)],
            [L("Background tiles flagged", "Teselas de fondo señaladas"), n(bg.value)],
            [L("Hardest normal classes", "Clases normales más difíciles"), L("mucus, debris", "moco, detritos")]]
    d.table(s, LEFT, 2.3, 6.2, rows, [3.9, 2.3], size=14, row_h=0.6)
    d.card(s, 7.5, 2.3, 4.94, 3.0, L("Not a failure, a choice", "No es un fallo, es una elección"), [
        L("The threshold was set for 1% false alarms. A screening tool would accept more alarms to miss fewer tumours.",
          "El umbral se fijó para un 1 % de falsas alarmas. Una herramienta de cribado aceptaría más alarmas para perder menos tumores."),
        L("AUROC says how well the scores rank. The threshold says what the lab can afford to review.",
          "El AUROC dice lo bien que ordenan las puntuaciones. El umbral dice lo que el laboratorio puede permitirse revisar.")], tsize=17, bsize=13.5)
    d.band(s, 5.65, 0.8, [(L("False positives ", "Los falsos positivos "), True), (L(
        "cluster in debris and mucus: unusual, not malignant. That is the gallery notebook 06 shows.",
        "se concentran en detritos y moco: raros, no malignos. Es la galería que muestra el notebook 06."), False)], size=14.5)
    d.footer(s, L("results/anomaly.csv: fpr_at_99, sensitivity_at_99, task_a_background flag_rate.", "results/anomaly.csv: fpr_at_99, sensitivity_at_99, flag_rate de task_a_background."))
    d.notes(s, "AUROC tells you how well the scores rank tumour above normal. A lab needs a threshold. We set it so that one percent "
            f"of normal tissue is flagged. At that point DINOv2 flags {n(sens.value)} of tumour tiles. That is a conservative "
            "operating point; a screening workflow would lower the threshold and accept more false alarms. The false alarms that "
            "do appear are mostly debris and mucus: tissue that is unusual but not cancer.",
            "El AUROC dice lo bien que las puntuaciones ponen el tumor por encima de lo normal. Un laboratorio necesita un umbral. "
            "Lo fijamos para que se señale un uno por ciento del tejido normal. En ese punto DINOv2 señala "
            f"{n(sens.value)} de las teselas tumorales. Es un punto de operación conservador; un flujo de cribado bajaría el umbral "
            "y aceptaría más falsas alarmas. Las falsas alarmas que aparecen son sobre todo detritos y moco: tejido raro, pero no cáncer.")

    # Wrap -------------------------------------------------------------------------------
    d.prefix = ""
    s = d.slide(L("ENGINEERING", "INGENIERÍA"), L("Cheap to re-run, hard to fool yourself", "Barato de repetir, difícil de engañarse"))
    eng = [
        (L("Cache features once", "Features en caché una vez"), L(
            "One forward pass per tile, saved as .npy and keyed by the tile list. Every head after that runs in seconds on a laptop.",
            "Una pasada por tesela, guardada en .npy y ligada a la lista de teselas. Cada cabeza posterior corre en segundos en un portátil.")),
        (L("Fix seeds and splits", "Semillas y particiones fijas"), L(
            "Seed 42, stratified subsets, patient-disjoint test set. Changing them invalidates the cache automatically.",
            "Semilla 42, subconjuntos estratificados, test con pacientes distintos. Cambiarlos invalida la caché automáticamente.")),
        (L("Numbers from files", "Números desde ficheros"), L(
            "Every table is a CSV; every figure and every number on these slides is generated from those CSVs.",
            "Cada tabla es un CSV; cada figura y cada número de estas diapositivas se genera a partir de esos CSV.")),
        (L("Floors and guards", "Suelos y salvaguardas"), L(
            "A random backbone in every comparison, and code that refuses to name a winner when the floor leads.",
            "Un backbone aleatorio en cada comparación y código que se niega a nombrar ganador cuando el suelo va delante.")),
    ]
    for i, (t, b) in enumerate(eng):
        d.card(s, LEFT + i * 2.95, 2.25, 2.75, 3.4, t, b, dark=(i == 3), tsize=16, bsize=13.5)
    d.footer(s, L("Free Colab T4, fast mode under 45 minutes per notebook. Throughput figures were I/O-bound and are not a speed ranking.",
                  "Colab T4 gratuito, modo rápido en menos de 45 minutos por notebook. Las cifras de rendimiento estaban limitadas por E/S y no son un ranking de velocidad."))
    d.notes(s, "Four habits that made the bug visible and the fix cheap. Features are cached, so re-running every head took two "
            "minutes on a laptop CPU. Seeds and splits are fixed, so numbers are reproducible. Every number comes from a CSV, so "
            "when the fix changed the results, the figures and these slides were regenerated, not edited by hand. And a random "
            "floor is always in the comparison, with code that refuses to declare a winner when the floor is ahead.",
            "Cuatro hábitos que hicieron visible el error y barata la corrección. Las features están en caché, así que repetir "
            "todas las cabezas llevó dos minutos en la CPU de un portátil. Semillas y particiones fijas, así que los números son "
            "reproducibles. Cada número sale de un CSV, así que cuando la corrección cambió los resultados, las figuras y estas "
            "diapositivas se regeneraron, no se editaron a mano. Y siempre hay un suelo aleatorio en la comparación, con código "
            "que se niega a declarar ganador cuando el suelo va delante.")

    s = d.slide(L("TAKEAWAYS", "CONCLUSIONES"), L("Five things to take home", "Cinco ideas para llevarse"))
    takeaways = [
        L("Self-supervised features describe tissue well enough that about 100 labels reach 84% accuracy on nine classes.",
          "Las features autosupervisadas describen el tejido lo bastante bien como para que unas 100 etiquetas den un 84 % de exactitud en nueve clases."),
        L("Their advantage is largest when labels are scarce, which is the normal situation in pathology.",
          "Su ventaja es máxima cuando escasean las etiquetas, que es la situación normal en patología."),
        L("Anomaly detection can be fitted on normal tissue only, with a threshold chosen without seeing a tumour.",
          "La detección de anomalías se puede ajustar solo con tejido normal y con un umbral elegido sin ver un tumor."),
        L("The head's geometry is part of the comparison. A random backbone that wins is a bug until proven otherwise.",
          "La geometría de la cabeza forma parte de la comparación. Un backbone aleatorio que gana es un error hasta que se demuestre lo contrario."),
        L("Write the decision rule first, and let it say 'no winner'.", "Escribe la regla de decisión antes y deja que pueda decir 'no hay ganador'."),
    ]
    for i, t in enumerate(takeaways):
        y = 2.2 + i * 0.86
        d.rect(s, LEFT, y, 0.62, 0.66, ORANGE if i == 3 else NAVY)
        d.text(s, LEFT, y, 0.62, 0.66, str(i + 1), 20, CREAM, True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        d.text(s, LEFT + 0.85, y, WIDTH - 0.85, 0.66, t, 16, INK, anchor=MSO_ANCHOR.MIDDLE)
    d.notes(s, "Five ideas. One: features learned without labels already describe tissue. Two: they pay off most when labels are "
            "scarce. Three: you can build an anomaly detector from normal tissue alone. Four, the one I most want you to remember: "
            "the head is part of the experiment, and a random network that wins is a bug. Five: decide how you will decide before "
            "you look.",
            "Cinco ideas. Una: las features aprendidas sin etiquetas ya describen el tejido. Dos: rinden más cuando escasean las "
            "etiquetas. Tres: se puede construir un detector de anomalías solo con tejido normal. Cuatro, la que más quiero que "
            "recordéis: la cabeza forma parte del experimento, y una red aleatoria que gana es un error. Cinco: decide cómo vas a "
            "decidir antes de mirar.")

    s = d.slide(L("WHAT COMES NEXT · V2", "LO QUE VIENE · V2"), L("From frozen backbones to training our own", "De backbones congelados a entrenar los nuestros"))
    nxt = [
        (L("BYOL from scratch", "BYOL desde cero"), L("Notebook 01, on tissue tiles, with the collapse ablation: no predictor, no EMA.",
                                                     "Notebook 01, con teselas de tejido y la ablación del colapso: sin predictor, sin EMA.")),
        (L("Mini DINO", "Mini DINO"), L("Notebook 02: loss, centring and EMA written from scratch; temperature demo.",
                                        "Notebook 02: pérdida, centrado y EMA escritos desde cero; demostración de temperatura.")),
        (L("Attention maps", "Mapas de atención"), L("Notebook 05: DINO attention heads against Grad-CAM on a supervised ResNet.",
                                                    "Notebook 05: cabezas de atención de DINO frente a Grad-CAM en una ResNet supervisada.")),
        (L("Whole slides", "Preparaciones completas"), L("Notebook 07: tissue mask, tiling by µm/px, heatmap, GeoJSON for QuPath.",
                                                         "Notebook 07: máscara de tejido, teselado por µm/px, mapa de calor, GeoJSON para QuPath.")),
    ]
    for i, (t, b) in enumerate(nxt):
        d.card(s, LEFT + i * 2.95, 2.25, 2.75, 2.9, t, b, tsize=16, bsize=13.5)
    d.band(s, 5.45, 0.85, [(L("Also: ", "Además: "), True), (L(
        "a pathology foundation model (Phikon, non-commercial) in the zoo, and tuned Mahalanobis and Isolation Forest heads.",
        "un modelo fundacional de patología (Phikon, no comercial) en el zoo y cabezas de Mahalanobis e Isolation Forest ajustadas."), False)], size=14.5)
    d.notes(s, "This is version one. Version two trains our own backbones: BYOL and a small DINO from scratch, with the ablations "
            "that show collapse. Then attention maps against Grad-CAM, and the step from tiles to whole slides with a heatmap you "
            "can open in QuPath. The Mahalanobis and Isolation Forest heads are in the results file but untuned; they still favour "
            "the random network, which is itself a lesson about heads.",
            "Esta es la versión uno. La versión dos entrena nuestros propios backbones: BYOL y un DINO pequeño desde cero, con las "
            "ablaciones que muestran el colapso. Después, mapas de atención frente a Grad-CAM y el paso de teselas a preparaciones "
            "completas con un mapa de calor que se puede abrir en QuPath. Las cabezas de Mahalanobis e Isolation Forest están en "
            "el fichero de resultados pero sin ajustar; todavía favorecen a la red aleatoria, lo que también es una lección sobre cabezas.")

    s = d.slide(L("RESOURCES", "RECURSOS"), L("Run it yourself", "Pruébalo tú mismo"), dark=True)
    d.text(s, LEFT, 2.3, 6.2, 3.8, [
        [(L("Repository  ", "Repositorio  "), True), (REPO_URL, False)],
        [(L("Notebooks  ", "Notebooks  "), True), ("00 · 03 · 04 · 06, " + L("Colab badges in the README", "insignias de Colab en el README"), False)],
        [(L("Results  ", "Resultados  "), True), ("results/*.csv", False)],
        [(L("Guides  ", "Guías  "), True), (L("docs/guide, English and Spanish", "docs/guide, en inglés y en español"), False)],
        [(L("Teaching repos  ", "Repos docentes  "), True), ("olonok69/BYOL · olonok69/Anomaly_detection", False)],
    ], 16, "#D9DEE6")
    d.text(s, 7.6, 2.3, 4.85, 3.8, [
        [(L("Data  ", "Datos  "), True), ("Kather et al. NCT-CRC-HE (CC BY 4.0) · PatchCamelyon (CC0)", False)],
        [(L("Papers  ", "Artículos  "), True), ("Grill et al. 2020 (BYOL) · Caron et al. 2021 (DINO) · Oquab et al. 2023 (DINOv2) · Roth et al. 2022 (PatchCore)", False)],
        [(L("Libraries  ", "Librerías  "), True), ("PyTorch · torchvision · timm · scikit-learn · umap-learn", False)],
    ], 14, "#D9DEE6")
    d.footer(s, L("Public data only. No clinical claim: this is a teaching demo.", "Solo datos públicos. Sin afirmaciones clínicas: es una demo docente."), dark=True)
    d.notes(s, "Everything is in the repository: the notebooks with Colab badges, the results files, and the guides in English and "
            "Spanish. The data is public and the licences are in the README.",
            "Todo está en el repositorio: los notebooks con insignias de Colab, los ficheros de resultados y las guías en inglés y "
            "en español. Los datos son públicos y las licencias están en el README.")

    return d.save(), d.count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "es"], nargs="*", default=["en", "es"])
    args = parser.parse_args()
    for lang in args.lang:
        out, count = build(lang)
        print(f"{out.name}: {count} slides")


if __name__ == "__main__":
    main()
