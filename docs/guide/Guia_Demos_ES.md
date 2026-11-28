# Las demos, explicadas: notebooks, celdas en directo y planes B

**Qué hace cada notebook, qué ejecutar en directo y qué hacer cuando algo falla**

Esta guía sigue los notebooks, no las diapositivas. La [guía del presentador](Guia_Presentador_ES.md) sigue el deck e indica en qué momento de la charla aparece cada notebook. La versión inglesa es [Demo_Guide_EN.md](Demo_Guide_EN.md), con la misma numeración de secciones.

---

## 0. Antes de la sesión

### 0.1 Lista de comprobación previa

* **Caché de Drive preparada.** `MyDrive/ssl-demo/` contiene `features/`, `indices/` y `results/` de una ejecución anterior. Con la caché en su sitio, los notebooks 04 y 06 se saltan todas las pasadas por la red.
* **GPU conectada** en Colab (Entorno de ejecución, Cambiar tipo, T4). Solo la necesitan los notebooks 00 y 03; el 04 y el 06 funcionan en CPU desde la caché.
* **Notebooks ejecutados de antemano** con las salidas guardadas, para poder enseñar todas las figuras aunque falle la red.
* **Zoom del navegador** al 125 % o más para un proyector; plegar el panel de ficheros de Colab.
* **Pestañas abiertas**: notebook 04, notebook 06, el deck y `results/anomaly_centring.csv`.
* **No hacen falta tokens** en la v1. Si un notebook posterior necesita un token de Hugging Face, va en los secretos de Colab (`google.colab.userdata`), nunca en una celda.

### 0.2 Orden de ejecución y tiempos

| Notebook | Qué hace | Primera ejecución (T4, modo rápido) | Con caché |
|---|---|---|---|
| `00_setup_and_data` | Monta Drive, descarga PathMNIST, NCT-CRC y, opcionalmente, PatchCamelyon, crea los CSV de índices, muestra la rejilla de clases y los aumentos | ~10 min, más la descarga | ~1 min |
| `03_backbone_zoo_feature_extraction` | Extrae los tres backbones para cada conjunto preparado; escribe `throughput.csv` | ~20 min | segundos (todo en caché) |
| `04_knn_linear_probe_fewshot` | kNN, sonda lineal al 1/10/100 % de etiquetas, matrices de confusión, UMAP; escribe `probe_nct.csv` | ~10 min | igual; solo lee features |
| `06_anomaly_detection_on_embeddings` | Regla de decisión, tres cabezas en las tareas A y B, coseno sin centrar frente a centrado, decisión, galería; escribe `anomaly.csv` y `anomaly_centring.csv` | ~3 min | igual |

Pon `MODE = "fast"` en la primera celda de código de cada notebook. El modo rápido es el del taller: 10.000 teselas de entrenamiento y 2.000 de test, semilla 42.

### 0.3 Ejecutar 04 y 06 en un portátil

Los notebooks 04 y 06 solo necesitan las features en caché, ni imágenes ni GPU. Copia `MyDrive/ssl-demo/features` y `MyDrive/ssl-demo/indices` a `./cache/` dentro del repositorio y después:

```bash
uv sync
```

Abre los notebooks en VS Code, o en cualquier entorno Jupyter, con el kernel de `.venv`. Sáltate la celda de clonado de Colab del principio.

En la CPU de un portátil, el notebook 06 tarda unos dos minutos y el 04 unos cinco, casi todo por la sonda con validación cruzada al 100 % de etiquetas. La galería de falsos positivos del 06 se omite cuando las imágenes de NCT-CRC no están en disco; todo lo demás funciona.

### 0.4 Después de un cambio de código

Las salidas guardadas en los notebooks son el registro de las ejecuciones en Colab. Cuando cambia el código de `src/ssl_demo/`, vuelve a ejecutar el 04 y el 06 en Colab y sube los notebooks con sus salidas, junto con los CSV de `results/`. Después regenera las figuras y los decks:

```bash
uv sync --group docs
uv run python docs/guide/make_figures.py
uv run python docs/make_deck.py
```

---

## 1. Notebook 00: configuración y datos

**Diapositiva.** 5 (la rejilla de clases del deck la dibuja `make_figures.py` a partir de teselas de NCT-CRC).

Lo que imprime, en orden:

1. La raíz de la caché (`/content/drive/MyDrive/ssl-demo` en Colab, `./cache` en local) y el modo.
2. La descarga de PathMNIST y la comprobación de que el orden de sus etiquetas coincide con `medmnist.INFO`. El notebook lanza un error antes que adivinar cuál es la clase tumoral.
3. El equilibrio de clases del subconjunto fijo y una rejilla de teselas por clase.
4. Vistas previas de los aumentos: recorte aleatorio redimensionado, variación de color, desenfoque, solarización y la variación de tinción H&E de `ssl_demo.data`. Son las vistas que compararía un método autosupervisado; úsalas en la diapositiva 8 o 9 si quieres una imagen real de "dos vistas de la misma imagen".
5. Descarga opcional de NCT-CRC a Drive y extracción a `/content/nct`, y PatchCamelyon opcional (`INCLUDE_PCAM`, unos 7 GB).

**Cuidado.** `INCLUDE_PCAM` se fija al principio del notebook y otra vez en la celda de NCT. Si PatchCamelyon no se descarga, la tarea B se omite en el notebook 06 y la regla de dos tareas no puede nombrar ganador.

**¿En directo?** No. Enseña la rejilla y los aumentos desde las salidas guardadas.

---

## 2. Notebook 03: extracción de features congeladas

**Diapositivas.** 3 (el bucle) y 24 (ingeniería).

Qué hace:

1. Carga cada backbone desde `ssl_demo.backbones` con su propio preprocesado: `random_resnet50` (sin pesos), `sup_resnet50_in1k` (ImageNet de torchvision), `dinov2_vits14` (`torch.hub`, `trust_repo=True`).
2. Para cada conjunto con índice (PathMNIST, NCT-CRC, PatchCamelyon), extrae las features con `torch.inference_mode` y autocast fp16 en GPU, las normaliza (L2) y guarda `features/<conjunto>/<modo>/<partición>/<backbone>.npy`, junto con `labels.npy` e `index.npy`. DINOv2 guarda además la media de sus tokens de parche como `dinov2_vits14__patchmean.npy`.
3. Se salta cualquier backbone cuyo fichero exista y cuyo `index.npy` coincida con la lista de teselas actual.
4. Escribe `results/throughput.csv`.

**Qué decir del rendimiento.** No es un ranking de velocidad. El primer backbone de cada conjunto pagó las lecturas en frío de Drive; los siguientes leyeron ficheros ya calientes. Para comparar velocidades hay que copiar antes los datos al disco local `/content` y añadir un lote de calentamiento.

**¿En directo?** No: 20 minutos. Enseña la tabla de salida y di que la segunda ejecución tarda segundos.

---

## 3. Notebook 04: kNN, sonda lineal y eficiencia de etiquetas

**Diapositivas.** 13 y 14.

Lo que imprime:

1. Qué conjunto evalúa: `nct` si están las features de NCT-CRC en caché y `pathmnist` si no.
2. Para cada backbone, un clasificador kNN (coseno, k = 20, voto ponderado por temperatura) y una sonda lineal al 1 %, 10 % y 100 % de las etiquetas de entrenamiento, cada uno con exactitud, exactitud equilibrada y F1 macro, y un intervalo bootstrap de 1.000 remuestreos.
3. La línea de control de `random_floor_warning`: avisa si la exactitud kNN del backbone sin entrenar queda a menos de cinco puntos de uno entrenado.
4. Matrices de confusión (todas las etiquetas), el gráfico de eficiencia de etiquetas y un UMAP de las features de test de cada backbone.

**Cómo se ajusta la sonda.** `fit_linear_probe` estandariza las features (el escalador se ajusta solo con las filas de entrenamiento) y elige C entre {0,001; 0,01; 0,1; 1} por validación cruzada estratificada de 3 particiones sobre esas mismas filas. Las teselas de test nunca influyen en C. Sin el escalador, las features casi paralelas de la red sin entrenar hacían que la sonda colapsara a una sola clase con un 10 % de las etiquetas.

**Cifras en pantalla** (`results/probe_nct.csv`, exactitud): kNN DINOv2 0,890, ImageNet 0,841, aleatoria 0,616. Sonda al 1 %: 0,837, 0,750, 0,532. Sonda al 100 %: 0,913, 0,903, 0,688.

**Puntos para comentar en las salidas.**

* Matrices de confusión: en los dos backbones preentrenados, la confusión más frecuente es entre estroma asociado al cáncer y músculo liso, en ambos sentidos. Los dos son tejido de células fusiformes, y a esta escala también a un patólogo le resultan cercanos. La sensibilidad para el epitelio tumoral con todas las etiquetas es de 0,86 en los dos.
* UMAP: una imagen en dos dimensiones de las features, coloreada por clase. Es una imagen, no una métrica; dilo y lee las cifras de la tabla.

**¿En directo?** Enseña las salidas guardadas. Volver a ejecutarlo lleva varios minutos por la sonda con validación cruzada.

---

## 4. Notebook 06: detección de anomalías y decisión

**Diapositivas.** De la 16 a la 23. Es la demo en directo.

### 4.1 La regla de decisión

La celda de importaciones imprime `DECISION_RULE` antes de ajustar nada y, a continuación, las clases normales, la clase anómala y la clase que queda fuera del ajuste (el fondo). Lee la regla en voz alta. Es la diapositiva 17.

### 4.2 Tarea A

La celda siguiente elige el conjunto (NCT-CRC si está en caché) y, para cada backbone:

1. Toma el conjunto normal de la partición de entrenamiento: las siete clases que no son tumor ni fondo (7.511 teselas).
2. Lo divide 80/20 con semilla 42: 6.009 teselas para ajustar la cabeza y 1.502 reservadas para fijar el umbral.
3. Construye el test: teselas normales y tumorales de la partición de test (1.764 teselas, 344 tumorales). El fondo se puntúa aparte para su tasa de alertas.
4. Llama a `run_anomaly_task`, que ajusta cada cabeza con las 6.009, fija el umbral en el percentil 99 de las puntuaciones de las 1.502 reservadas y puntúa el test.

Imprime los tamaños del conjunto normal y del test por backbone.

### 4.3 Coseno sin centrar frente a centrado (la celda en directo)

La celda bajo "Raw versus centred cosine" ajusta la cabeza kNN dos veces con las mismas 6.009 teselas, una con `center=False` y otra con `center=True`, e imprime:

| backbone | centroid_cosine | auroc_raw_cosine | auroc_centred_cosine |
|---|---|---|---|
| random_resnet50 | 1,000 | 0,885 | 0,492 |
| sup_resnet50_in1k | 0,820 | 0,672 | 0,902 |
| dinov2_vits14 | 0,817 | 0,668 | 0,937 |

También guarda la tabla en `results/anomaly_centring.csv`. Tarda segundos sobre las features en caché, así que ejecútala en directo durante las diapositivas 18 a 20. Qué señalar:

* El `centroid_cosine` de 1,000 de la red aleatoria: todos los vectores apuntan en la misma dirección.
* Las dos columnas de AUROC se cruzan: la red aleatoria pasa de primera al azar y los preentrenados suben por encima de 0,9.

### 4.4 Tarea B, la tabla y la decisión

La celda siguiente repite el procedimiento con PatchCamelyon (2.000 parches normales para ajustar, 500 reservados, 2.000 de test) si sus features están en caché, junta todas las filas y escribe `results/anomaly.csv`.

La siguiente imprime una tabla de AUROC por tarea y cabeza y, después, la decisión:

> No backbone is clearly better on both tasks. task_a: dinov2_vits14 AUROC 0.937 [0.925, 0.948] is clear of sup_resnet50_in1k [0.883, 0.919]. task_b: sup_resnet50_in1k leads on the point estimate (0.759) but its CI overlaps dinov2_vits14.

En español: ningún backbone es claramente mejor en las dos tareas; en la tarea A, DINOv2 supera con claridad a ImageNet; en la tarea B, ImageNet va por delante en la estimación puntual, pero su intervalo se solapa con el de DINOv2. Enséñala en la diapositiva 22. Si alguna vez el backbone sin entrenar va primero en una tarea, o queda a menos de 0,02 del mejor entrenado, `decide_backbone` se niega a nombrar ganador e imprime un mensaje "Pipeline check".

**Las otras cabezas.** La tabla también muestra Mahalanobis e Isolation Forest. Las dos ponen primera a la red sin entrenar en las dos tareas, e Isolation Forest queda por debajo de 0,5 con los backbones preentrenados. En la v1 no están ajustadas. Si preguntan, dilo, y úsalo como prueba de que la cabeza forma parte de la comparación.

### 4.5 Gráfico del cambio de backbone

Un gráfico de barras del AUROC de la distancia kNN por backbone y tarea, con intervalos: la versión del notebook de la figura de la diapositiva 22.

### 4.6 Galería de falsos positivos

Las teselas normales de test con mayor puntuación de distancia kNN, seis por backbone. Dominan los detritos, el moco, los pliegues y los bordes de tesela. Es el "raro, no maligno" de la diapositiva 23. Necesita las imágenes: en Colab, NCT-CRC extraído en `/content/nct` (o la variable `NCT_ROOT`); si no, la celda indica que la galería se omite.

---

## 5. Planes B

| Problema | Qué hacer |
|---|---|
| Colab no funciona o va lento | Presenta desde las salidas guardadas y el deck; los dos cuentan la historia completa. Ejecuta la celda de centrado en local (sección 0.3) si tienes la caché en el portátil. |
| Drive no se monta | Usa la copia local de la caché. Si no la hay, enseña solo salidas guardadas. |
| Falla la descarga de DINOv2 en el 03 | Solo importa con la caché vacía. Omite esa fila y explica por qué; el 04 y el 06 usan las features en caché. |
| Falta PatchCamelyon | La tarea B se omite y la regla indica que no puede declarar un ganador en las dos tareas. Presenta la tarea A y explica la regla con las cifras guardadas de la tarea B del deck. |
| Galería omitida | Es lo esperado cuando las imágenes de NCT-CRC no están extraídas en la sesión. Enseña la salida guardada. |
| Las cifras no coinciden con el deck | Revisa `MODE`, la semilla y que los `indices/*.csv` de Drive no se hayan regenerado con otro tamaño de subconjunto. La clave de la caché incluye la lista de teselas, así que un subconjunto distinto vuelve a extraer features. |

---

## 6. Reproducibilidad y limitaciones conocidas

* Las semillas son fijas (`SEED = 42` en `ssl_demo.config`). Los intervalos bootstrap usan la misma semilla, así que son idénticos entre ejecuciones sobre las mismas features.
* Las cifras del modo rápido del deck proceden de features extraídas en GPU de Colab (L4 para NCT-CRC y PatchCamelyon). Volver a extraerlas en otra GPU puede mover el tercer decimal por el autocast fp16.
* El fichero de rendimiento está limitado por E/S y no compara velocidades de modelos.
* Las cabezas de Mahalanobis e Isolation Forest no están ajustadas.
* Las etiquetas de PatchCamelyon se refieren a los 32 x 32 píxeles centrales de un parche de 96 x 96, reescalado a 224; un embedding global diluye un foco tumoral pequeño. Recortar el centro antes de reescalar es un experimento de la v2.
* Nada del repositorio son datos clínicos. Ningún resultado de aquí es una afirmación clínica.
