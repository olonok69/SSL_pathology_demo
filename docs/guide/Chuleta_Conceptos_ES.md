# Chuleta de conceptos

Las definiciones del taller de SSL en patología en una página. Versión inglesa: [Concepts_Cheatsheet_EN.md](Concepts_Cheatsheet_EN.md).

## Métodos autosupervisados

| Método | Dos redes | Objetivo | Pérdida | Qué evita el colapso | Backbone típico |
|---|---|---|---|---|---|
| SimCLR / MoCo (contrastivos) | Un codificador (MoCo añade un codificador con momento) | La otra vista de la misma imagen, frente a muchos negativos | InfoNCE | Los negativos separan imágenes distintas | ResNet-50 |
| BYOL | Online y objetivo; objetivo = EMA de la online | La proyección del objetivo de la otra vista | Error cuadrático medio entre vectores normalizados | Predictor solo en el lado online y objetivo EMA lento | ResNet-50 |
| DINO | Alumno y profesor; profesor = EMA del alumno | La distribución de probabilidad del profesor sobre los recortes globales | Entropía cruzada | Centrar y afilar la salida del profesor | ViT-S/16, ViT-B/16 |
| DINOv2 | Como DINO | Como DINO, más tokens de parche enmascarados (iBOT) | Entropía cruzada a nivel de imagen y de parche | Como DINO, más regularizadores | De ViT-S/14 a ViT-g/14, 142 M de imágenes curadas |
| MAE / BEiT / data2vec (enmascarado) | Un codificador (y un decodificador) | Parches ocultos: píxeles, tokens o features de un profesor | Reconstrucción o regresión | La propia tarea; ninguna constante reconstruye píxeles | ViT |

* **Colapso.** La red devuelve el mismo vector para todas las imágenes. Iguala todas las vistas a la perfección y no ha aprendido nada.
* **EMA (media móvil exponencial).** Pesos objetivo = m × objetivo + (1 − m) × online, con m cercano a 1. Una copia lenta y suavizada.
* **Multirrecorte.** Recortes globales grandes y locales pequeños de una imagen; el alumno debe coincidir con el profesor a partir de una parte pequeña.
* **ViT y token CLS.** La imagen se corta en parches (14 × 14 px en ViT-S/14). Un token CLS atiende a todos los parches; su estado final es el embedding de la imagen (384 números en ViT-S).
* **Modelo fundacional.** Un backbone preentrenado a gran escala, a menudo autosupervisado, reutilizado congelado en muchas tareas. Revisa con qué se entrenó y su licencia.

## Evaluar features congeladas

* **Congelado.** Los pesos del backbone no cambian durante la evaluación; solo se ajusta una cabeza pequeña.
* **Clasificador kNN.** Etiqueta una tesela de test por votación de sus k teselas de entrenamiento más parecidas (similitud coseno). Aquí k = 20, ponderado por temperatura.
* **Sonda lineal.** Regresión logística sobre las features. Aquí estandarizadas, con C por validación cruzada de 3 particiones sobre las filas de entrenamiento.
* **Eficiencia de etiquetas.** La sonda entrenada con el 1 %, el 10 % y el 100 % de las etiquetas.
* **Intervalo bootstrap.** Se remuestrean las teselas de test con reposición 1.000 veces; percentiles 2,5 y 97,5 de la métrica.
* **Suelo aleatorio.** Un backbone sin entrenar con la misma arquitectura. Uno entrenado debe superarlo con claridad.

## Detección de anomalías sobre embeddings

* **Solo normales.** La cabeza se ajusta solo con tejido normal. Las anomalías solo se ven en el test.
* **Cabeza de distancia kNN.** Distancia coseno media a las k teselas normales más cercanas (k = 5), tras centrar en la media normal. Versión a nivel de imagen del banco de memoria de PatchCore.
* **Centrar.** Restar la media de las features normales de ajuste antes del coseno. Elimina la dirección común que, si no, domina la distancia coseno.
* **Coseno con el centroide.** Coseno medio entre cada vector de features y el vector medio. Si se acerca a 1, las distancias coseno sin centrar dicen poco.
* **Cabeza de Mahalanobis.** Distancia a una gaussiana ajustada a las features normales, con covarianza con contracción (estilo PaDiM).
* **Isolation Forest.** Árboles aleatorios que aíslan rápido los puntos raros; caminos cortos significan anomalía.
* **Umbral.** Percentil 99 de las puntuaciones de teselas normales reservadas: por construcción, se señala alrededor de un 1 % del tejido normal.
* **AUROC.** Probabilidad de que una anomalía al azar puntúe más alto que una normal al azar. 0,5 es el azar.
* **AUPRC.** Área bajo la curva de precisión frente a sensibilidad. Depende de lo raras que sean las anomalías.
* **FPR al 95 % de TPR.** Proporción de normales señaladas cuando se detecta el 95 % de las anomalías.
* **Regla de decisión preregistrada.** Escrita antes de ver los números. Un backbone gana solo si su intervalo supera al del siguiente mejor en las dos tareas.

## Vocabulario de preparaciones completas (para la v2)

* **WSI.** Whole-slide image: un escaneado de gigapíxeles de una preparación.
* **Pirámide y nivel.** Una WSI se guarda a varias resoluciones; el nivel 0 es la resolución completa.
* **mpp.** Micras por píxel. Unas 0,25 a 40x y 0,5 a 20x. Se tesela por mpp, no por nivel.
* **Máscara de tejido.** Separa el tejido del cristal, a menudo con un umbral de Otsu sobre un nivel de baja resolución.
* **Tesela.** Un parche de tamaño fijo (aquí 224 px) cortado de la zona de tejido.
* **MIL.** Aprendizaje de instancias múltiples: una etiqueta de preparación para una bolsa de teselas. El top-k o la atención (ABMIL, CLAM) agregan las puntuaciones de las teselas.
* **GeoJSON.** Formato para exportar regiones o mapas de calor a QuPath.

## Las cifras que hay que recordar

| | ResNet-50 aleatoria | ResNet-50 ImageNet | DINOv2 ViT-S/14 |
|---|---|---|---|
| Exactitud kNN | 0,616 | 0,841 | 0,890 |
| Sonda, 1 % de etiquetas | 0,532 | 0,750 | 0,837 |
| Sonda, 100 % de etiquetas | 0,688 | 0,903 | 0,913 |
| AUROC tarea A, coseno sin centrar | 0,885 | 0,672 | 0,668 |
| AUROC tarea A, centrado | 0,492 | 0,902 | 0,937 |
| AUROC tarea B, centrado | 0,683 | 0,759 | 0,752 |
| Coseno con el centroide | 1,000 | 0,820 | 0,817 |
