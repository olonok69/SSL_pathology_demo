# Features autosupervisadas para patología: guía del presentador

**Acompañante diapositiva a diapositiva de `docs/SSL_Pathology_ES.pptx` (27 diapositivas)**

Esta guía es para quien presente el deck, en directo como taller o grabado como vídeo. Sigue las diapositivas en orden y da, para cada una, qué hay en pantalla, qué decir, las cifras que conviene tener a mano, las preguntas que suelen surgir y la frase que lleva a la siguiente diapositiva. La versión inglesa es [Presenter_Guide_EN.md](Presenter_Guide_EN.md), con la misma numeración. La [guía de demos](Guia_Demos_ES.md) cubre los notebooks: qué ejecutar en directo, qué enseñar desde salidas guardadas y qué hacer si algo falla. La [chuleta](Chuleta_Conceptos_ES.md) reúne las definiciones en una página.

Todas las cifras de esta guía proceden de [`docs/brief.md`](../brief.md), que a su vez procede de los ficheros CSV de `results/`. Si cambia una cifra, se regeneran las figuras y los decks y se actualiza el brief; aquí no se editan cifras a mano.

---

## Cómo usar esta guía

Cada diapositiva tiene los mismos bloques:

* **En pantalla**: lo que muestra la diapositiva, para ensayar sin el deck abierto.
* **Decir**: un guion en estilo hablado. Léelo una o dos veces y después preséntalo con tus palabras. Las notas del orador del deck llevan una versión más corta.
* **Datos clave**: las cifras y fuentes que pueden pedirte que repitas.
* **Si preguntan**: respuestas breves a preguntas frecuentes.
* **Siguiente**: una frase para pasar a la diapositiva siguiente.

### Público

La sesión está pensada para una sala mixta: científicos de datos, patólogos, personal de laboratorio, ingenieros y responsables. Explica la idea antes que la métrica y usa las imágenes de tejido para anclar los puntos abstractos. Cuando aparece un término técnico por primera vez, el guion lo define en una frase.

### Escaleta, taller de 90 minutos

| Minutos | Bloque | Diapositivas | En directo |
|---|---|---|---|
| 0-10 | Apertura: el problema, el bucle, los datos | 1-5 | – |
| 10-32 | Acto 1 · Aprender sin etiquetas | 6-11 | – |
| 32-45 | Acto 2 · Medir las features | 12-14 | Salidas guardadas del notebook 04 |
| 45-50 | Descanso | | |
| 50-80 | Acto 3 · Detectar y decidir | 15-23 | Notebook 06: celda de centrado y regla de decisión |
| 80-90 | Ingeniería, conclusiones, próximos pasos, preguntas | 24-27 | – |

### Montaje para vídeo, 25-30 minutos

Diapositivas 1, 3, 5, 7, 8, 10, 11, 13, 14, 16, 17, 18, 19, 20, 22, 23, 25 y 27. Las salidas de los notebooks se muestran como grabaciones de pantalla; solo se ejecuta en directo la celda de centrado. Capítulos sugeridos: gancho y bucle (3 min), acto 1 (8), acto 2 (5), acto 3 (11), cierre (3).

### Antes de la sesión

* Ejecuta los notebooks una vez en un entorno de Colab nuevo con la caché de Drive en su sitio (ver la guía de demos). La segunda ejecución lee las features en caché y tarda unos minutos.
* Ten abiertos: el deck, el notebook 06 en Colab, `results/anomaly.csv`, `results/anomaly_centring.csv` y `docs/brief.md`.
* Tres mensajes que deben quedar, recortes lo que recortes: (1) las features autosupervisadas describen bien el tejido y rinden más cuando escasean las etiquetas; (2) la detección de anomalías solo con tejido normal funciona, y el umbral es una decisión del laboratorio; (3) una red aleatoria que gana es un error, y una regla escrita de antemano es lo que permite verlo.

### Lo que esta versión no cubre

BYOL y DINO se explican, no se entrenan: eso son los notebooks 01 y 02 de la v2. Todavía no hay mapas de atención (notebook 05) ni mapas de calor de preparaciones completas (notebook 07). Si alguien pide verlos, dilo con claridad y remite a la diapositiva "Lo que viene".

---

## Apertura

### Diapositiva 1 · ¿Puede un modelo aprender cómo es un tejido sin etiquetas?

*Alrededor de 1 minuto.*

**En pantalla.** Título, resumen en una línea, tres etiquetas (3 backbones congelados, 2 tareas de anomalías, solo datos públicos) y la dirección del repositorio.

**Decir.** "Etiquetar tejido es lento y caro. Así que la pregunta de hoy es si un modelo puede aprender cómo es un tejido solo a partir de imágenes, sin etiquetas, y si podemos usar lo aprendido para señalar tejido que no es normal. Por el camino, una segunda pregunta resulta igual de importante: cuando aparece un modelo nuevo y más grande, ¿cómo sabemos si de verdad es mejor? Todo lo que vais a ver funciona con datos públicos y GPU gratuitas de Colab, y está en el repositorio de la diapositiva."

**Datos clave.** Tres backbones: una ResNet-50 sin entrenar, una ResNet-50 supervisada en ImageNet y DINOv2 ViT-S/14. Dos tareas de anomalías: tumor colorrectal (NCT-CRC) y metástasis en ganglio linfático (PatchCamelyon).

**Siguiente.** "Empecemos por qué las etiquetas son el cuello de botella."

### Diapositiva 2 · El cuello de botella son las etiquetas, no las imágenes

*Alrededor de 2 minutos.*

**En pantalla.** Tres tarjetas: millones de teselas, pocas anotaciones, dos preguntas. Una franja con la definición de aprendizaje autosupervisado.

**Decir.** "Una sola preparación escaneada a 20x se corta en decenas de miles de teselas pequeñas. Un estudio preclínico tiene cientos de preparaciones. Ningún patólogo va a etiquetar cada tesela. El aprendizaje autosupervisado traslada el esfuerzo: el modelo construye su propia señal de entrenamiento a partir de las imágenes, por ejemplo aprendiendo que dos recortes de la misma imagen deben parecerle iguales. Las etiquetas solo hacen falta al final, y pocas, para leer lo que el modelo ha aprendido en una tarea concreta."

**Si preguntan.** *¿Es lo mismo que el clustering no supervisado?* Está relacionado, pero no. El aprendizaje autosupervisado entrena una red con una tarea auxiliar construida a partir de los datos; el clustering agrupa datos sin entrenar una representación. Las features que salen del entrenamiento autosupervisado se pueden agrupar, clasificar o usar para detectar anomalías.

**Siguiente.** "Este es el bucle que usaremos en todos los resultados de la sesión."

### Diapositiva 3 · Un bucle: teselas, backbone congelado, caché, misma cabeza, comparar

*Alrededor de 2 minutos.*

**En pantalla.** Cinco cajas de izquierda a derecha; la del backbone resaltada con "solo se cambia esto". Franja: se cambia el backbone y todo lo demás queda fijo.

**Decir.** "Todos los resultados de hoy salen del mismo bucle. Las teselas entran en un backbone cuyos pesos nunca cambian; decimos que está congelado. La salida, un vector de unos cientos de números por tesela, se guarda en disco una vez. Encima se ajusta una cabeza pequeña: una votación de vecinos, una regresión logística o una puntuación de anomalía. Para comparar backbones cambiamos solo el backbone y repetimos la misma cabeza sobre las mismas teselas con las mismas semillas. Si cambia el número, cambió la representación, porque nada más cambió."

**Datos clave.** Las features se guardan como ficheros `.npy` en Google Drive, ligados a la lista exacta de teselas. Una segunda ejecución se salta la pasada por la red.

**Si preguntan.** *¿Por qué no reentrenar (fine-tune) el backbone?* El reentrenamiento mezcla dos preguntas: lo buena que es la representación y lo bien que has entrenado. La evaluación congelada aísla la primera. Reentrenar es un paso posterior y, con datos solo normales, puede empeorar las features para detectar anomalías.

**Siguiente.** "La sesión recorre ese bucle en tres actos."

### Diapositiva 4 · Tres actos, una historia

*Alrededor de 1 minuto.*

**En pantalla.** Tres tarjetas: aprender sin etiquetas, medir las features, detectar y decidir. Pie con los notebooks de esta versión y los que llegan en la v2.

**Decir.** "Primero, las ideas: cómo puede aprender una red a partir de imágenes sin ninguna etiqueta. Segundo, cómo medir lo que ha aprendido sin reentrenarla. Tercero, cómo usar esas features para encontrar tejido que no es normal y cómo decidir con rigor entre backbones. El acto 3 contiene los que para mí son los diez minutos más útiles de la sesión: un resultado que parecía espectacular y resultó ser un error."

**Siguiente.** "Primero, los datos."

### Diapositiva 5 · Nueve clases de tejido de cáncer colorrectal, dos grupos de pacientes

*Alrededor de 2 minutos.*

**En pantalla.** Tres teselas de cada una de las nueve clases, con el epitelio tumoral recuadrado. Franja con los conjuntos de entrenamiento, test y tarea B.

**Decir.** "Estos son los datos: teselas públicas de cáncer colorrectal publicadas por Kather y colaboradores. Nueve clases, desde grasa y fondo vacío hasta epitelio tumoral, recuadrado a la derecha. Cada tesela mide 224 píxeles de lado a media micra por píxel. Entrenamos con diez mil teselas de un grupo de pacientes y evaluamos con dos mil de pacientes distintos, que es la forma honesta de evaluar. En la parte de anomalías, el epitelio tumoral hace de 'no normal', y el fondo se deja aparte porque el cristal vacío es raro pero no es cáncer. Una segunda tarea usa PatchCamelyon: parches de ganglio linfático con o sin metástasis."

**Datos clave.** NCT-CRC-HE-100K (entrenamiento) y CRC-VAL-HE-7K (test), Zenodo 1214456, CC BY 4.0. PatchCamelyon, CC0. Modo rápido: 10.000 / 2.000 teselas; PatchCamelyon 5.000 / 2.000. Semilla 42.

**Si preguntan.** *¿Hay datos clínicos o de alguna empresa?* No. Solo conjuntos públicos, con sus licencias en el README. *¿Por qué las teselas de detritos son tan distintas entre sí?* Los detritos y la necrosis son heterogéneos por naturaleza, y la tinción varía entre laboratorios; esa variedad reaparece al ver los falsos positivos.

**Siguiente.** "Acto 1: de dónde salen los backbones autosupervisados."

---

## Acto 1 · Aprender sin etiquetas

### Diapositiva 6 · Apertura del acto 1

*Segundos.* "De dónde salen los backbones autosupervisados y cuál usamos realmente."

### Diapositiva 7 · Supervisado, autosupervisado, modelo fundacional

*Alrededor de 3 minutos.*

**En pantalla.** Tres tarjetas. Franja con los tres backbones del taller.

**Decir.** "Un backbone puede aprender de tres formas. Supervisado: a partir de etiquetas humanas, como las mil clases de objetos de ImageNet. Autosupervisado: a partir de la estructura de las propias imágenes, sin etiquetas. Un modelo fundacional es un backbone autosupervisado entrenado con un conjunto enorme y reutilizado, congelado, en muchas tareas. Comparamos tres backbones. La ResNet-50 sin entrenar no compite: es el suelo, la puntuación que da la arquitectura por sí sola. Si alguna vez gana, algo falla en el proceso. Quedaos con esta frase; vuelve en el acto 3."

**Datos clave.** Ninguno de los tres backbones se entrenó con estas teselas. DINOv2 se entrenó con imágenes naturales.

**Si preguntan.** *¿Para qué incluir una red aleatoria?* Porque es la comprobación más barata que existe. Una red convolucional aleatoria sigue captando estadísticas de color y textura, así que en algunas tareas supera al azar; un backbone entrenado debería ganarle con claridad.

**Siguiente.** "Los métodos autosupervisados se agrupan en tres familias."

### Diapositiva 8 · Tres familias: contrastar, destilar, enmascarar

*Alrededor de 3 minutos.*

**En pantalla.** Árbol: contrastivo (SimCLR, MoCo), autodestilación (BYOL, DINO, DINOv2), enmascarado (MAE, BEiT, data2vec). Franja sobre el colapso.

**Decir.** "Los métodos contrastivos acercan dos vistas de la misma imagen y alejan las demás imágenes; necesitan muchos ejemplos negativos. Los de autodestilación prescinden de los negativos: una red aprende a imitar a otra. Los de enmascarado ocultan partes de la imagen y predicen lo oculto, como hacen los modelos de lenguaje con las palabras. Todos deben evitar el colapso: la solución trivial en la que la red devuelve el mismo vector para todas las imágenes. Ese vector iguala a la perfección cualquier par de vistas y no ha aprendido nada. El diseño de cada método es, sobre todo, un truco contra el colapso."

**Si preguntan.** *¿Qué familia funciona mejor en patología?* Los modelos fundacionales de patología actuales usan sobre todo autodestilación con enmascarado, la receta de DINOv2 e iBOT. Los contrastivos funcionan, pero necesitan lotes grandes para tener negativos suficientes.

**Siguiente.** "BYOL, el primer método que funcionó sin negativos."

### Diapositiva 9 · BYOL: sin negativos, con un predictor y un objetivo que se mueve despacio

*Alrededor de 4 minutos.*

**En pantalla.** Diagrama de BYOL: dos vistas; arriba, codificador online, proyector y predictor; abajo, codificador objetivo y proyector; flecha de EMA; sin gradiente hacia la comparación.

**Decir.** "BYOL toma dos vistas aumentadas de una imagen. La red online, arriba, codifica la primera vista, la proyecta y predice lo que devolverá la red objetivo para la segunda vista. La red objetivo, abajo, nunca se entrena por gradiente. Sus pesos son una media móvil exponencial de los pesos online: los siguen despacio. Dos asimetrías evitan el colapso: el predictor solo existe en el lado online y el objetivo se mueve despacio. No hay ningún ejemplo negativo, y eso sorprendió en 2020. Entrené BYOL desde cero con conjuntos de imágenes pequeños en un repositorio público; en la v2 del taller será el notebook 01, con teselas de tejido."

**Datos clave.** Grill et al., 2020. Repositorio: `olonok69/BYOL`. La v2 añade una ablación del colapso: sin predictor, sin EMA.

**Si preguntan.** *¿Qué es aquí una media móvil exponencial?* Tras cada paso de entrenamiento, los pesos objetivo pasan a ser 0,99 veces su valor anterior más 0,01 veces los pesos online, o algo similar. El objetivo es una historia suavizada de la red online.

**Siguiente.** "DINO mantiene la idea y cambia el objetivo."

### Diapositiva 10 · DINO: un alumno aprende a coincidir con su propio profesor lento

*Alrededor de 4 minutos.*

**En pantalla.** Diagrama de DINO: recortes globales al profesor; recortes globales y locales al alumno; centrar y afilar en el profesor; softmax en el alumno; entropía cruzada; flecha de EMA.

**Decir.** "DINO llama a las dos redes alumno y profesor. Las dos son Vision Transformers. El profesor ve dos recortes grandes de la imagen; el alumno ve esos y varios recortes pequeños. El alumno debe producir la misma distribución de probabilidad que el profesor, lo que le obliga a deducir el todo a partir de una parte. Los pesos del profesor, como en BYOL, son una media móvil de los del alumno. Dos operaciones sobre la salida del profesor evitan el colapso: centrar, que impide que una dimensión domine, y afilar con una temperatura baja, que impide que la salida se vuelva uniforme. Tiran en sentidos opuestos, y el equilibrio es lo que permite entrenar."

**Datos clave.** Caron et al., 2021. Multirrecorte: 2 recortes globales (224 px) y varios locales (96 px) en la receta original.

**Si preguntan.** *¿Por qué se enseñan mapas de atención de DINO?* Porque las cabezas de atención del profesor, sin etiquetas, a menudo perfilan objetos. En patología pueden resaltar núcleos o glándulas. Es el notebook 05 de la v2.

**Siguiente.** "El backbone que usamos realmente es DINOv2."

### Diapositiva 11 · DINOv2 ViT-S/14: el backbone que usamos, congelado

*Alrededor de 3 minutos.*

**En pantalla.** Tres tarjetas: qué añade DINOv2, el ViT en una línea, cómo lo usamos. Franja: dos preguntas para cualquier modelo fundacional.

**Decir.** "DINOv2 escala DINO de dos formas: añade un objetivo de parches enmascarados tomado de iBOT y se entrena con un conjunto curado de 142 millones de imágenes naturales. Nada de histología. Usamos el Vision Transformer pequeño, ViT-S, que corta cada tesela en parches de 14 píxeles. Un token especial, el CLS, atiende a todos, y su estado final, 384 números, es el embedding de la tesela. Antes de fiarse de cualquier número de un modelo fundacional hay que hacer dos preguntas: ¿formaban parte los datos de evaluación de su entrenamiento? ¿Y qué permite su licencia? Phikon, un modelo de patología, es no comercial. UNI exige acceso aprobado."

**Datos clave.** Oquab et al., 2023. Apache-2.0, cargado mediante `torch.hub`. Normalización de ImageNet, entrada de 224 px. Embedding: token CLS, 384 dimensiones; la media de los tokens de parche también se guarda en caché.

**Si preguntan.** *¿Por qué no usar un modelo fundacional de patología?* Sobre todo por licencias y acceso, en un taller público. Phikon está previsto para la v2, señalando su licencia no comercial. *¿Importa la normalización?* Muchísimo. Cada backbone necesita su propia media, desviación típica y tamaño de entrada; equivocarse ahí es el error silencioso clásico.

**Siguiente.** "Acto 2: cómo juzgar un backbone congelado."

---

## Acto 2 · Medir las features

### Diapositiva 12 · Apertura del acto 2

*Segundos.* "Juzgar un backbone sin tocar ni un solo peso."

### Diapositiva 13 · Vecinos más cercanos: DINOv2 0,890, ImageNet 0,841, aleatoria 0,616

*Alrededor de 5 minutos. Si hay tiempo, enseña la tabla y la matriz de confusión guardadas del notebook 04.*

**En pantalla.** Dos gráficos de barras con intervalos del 95 %: exactitud de kNN y exactitud de la sonda lineal con todas las etiquetas.

**Decir.** "Hay dos formas de leer una representación congelada. El clasificador de k vecinos etiqueta una tesela de test por votación de sus 20 teselas de entrenamiento más parecidas: si el tejido parecido queda cerca en el espacio de features, funciona. La sonda lineal ajusta una regresión logística sobre las features: si las nueve clases se pueden separar con cortes rectos, funciona. Nada del backbone se mueve. DINOv2 gana en vecinos, 0,890 frente a 0,841 de ImageNet. Con las diez mil etiquetas, la sonda casi cierra la distancia: 0,913 frente a 0,903, y los intervalos se solapan. La red sin entrenar queda muy por debajo, donde debe estar un suelo."

**Datos clave.** `results/probe_nct.csv`. kNN: similitud coseno, k = 20, voto ponderado por temperatura. Sonda: features estandarizadas, regresión logística con C elegido por validación cruzada de 3 particiones solo sobre las filas de entrenamiento. 2.000 teselas de test, nueve clases, 1.000 remuestreos bootstrap.

**Si preguntan.** *¿Por qué estandarizar las features?* Sin ello, las features de la red sin entrenar, casi paralelas, hacían que la sonda colapsara a una sola clase con un 10 % de las etiquetas. Estandarizar es justo para todos los backbones. *¿Son distintos 0,913 y 0,903?* Con esta prueba, no: los intervalos se solapan.

**Siguiente.** "Lo interesante es qué pasa con menos etiquetas."

### Diapositiva 14 · Con un 1 % de las etiquetas: 0,837 frente a 0,750

*Alrededor de 4 minutos.*

**En pantalla.** Curvas de eficiencia de etiquetas de los tres backbones al 1 %, 10 % y 100 % de las etiquetas, con intervalos sombreados. Tarjeta sobre cómo leerlas.

**Decir.** "Esta es la curva que importa a un laboratorio de patología. A la izquierda, solo un uno por ciento de las etiquetas de entrenamiento: unas cien teselas, once por clase. Las features de DINOv2 ya llegan a 0,837 de exactitud; las de ImageNet, a 0,750. A la derecha, con las diez mil etiquetas, convergen. El valor de una representación mejor es máximo justo donde las etiquetas son caras, que en patología es casi siempre."

**Datos clave.** Exactitud de la sonda al 1 % / 10 % / 100 %: DINOv2 0,837 / 0,894 / 0,913; ImageNet 0,750 / 0,870 / 0,903; aleatoria 0,532 / 0,547 / 0,688.

**Si preguntan.** *¿Por qué no se da la exactitud equilibrada?* Está en el CSV y cuenta la misma historia; las clases están aproximadamente equilibradas en el subconjunto.

**Siguiente.** "Acto 3: ¿pueden las mismas features señalar tejido que no es normal?"

---

## Acto 3 · Detectar anomalías y decidir

### Diapositiva 15 · Apertura del acto 3

*Segundos.* "Aprender cómo es el tejido normal, señalar el resto y decidir con rigor entre backbones."

### Diapositiva 16 · Se ajusta solo con tejido normal; los tumores aparecen solo en el test

*Alrededor de 4 minutos.*

**En pantalla.** Tabla de las tareas A y B: clases normales, anomalía, umbral, lo que se deja aparte. Tarjeta con las tres cabezas. Franja: no se ve ningún tumor al ajustar ni al fijar el umbral.

**Decir.** "La detección de anomalías da la vuelta al problema: solo describimos el tejido normal. En la tarea A la cabeza se ajusta con siete clases normales, seis mil teselas. Las teselas tumorales solo aparecen al puntuar el test. El umbral es la puntuación por debajo de la cual queda el 99 por ciento de un conjunto reservado de teselas normales, así que, por construcción, se señala alrededor de un uno por ciento del tejido normal, y nunca miramos un tumor para elegirlo. El fondo se excluye y se informa aparte. Hay tres cabezas; la que fija nuestra regla de decisión es la distancia a las teselas normales más cercanas, que es la versión a nivel de imagen del banco de memoria de PatchCore."

**Datos clave.** Tarea A: 6.009 teselas normales para ajustar, 1.502 reservadas para el umbral, 1.764 teselas de test, de ellas 344 tumorales. Tarea B: 2.000 parches normales para ajustar, 500 reservados, 2.000 de test. Cabeza kNN: k = 5, banco de memoria limitado a 2.000 teselas.

**Si preguntan.** *¿Por qué excluir el fondo?* El cristal vacío está trivialmente lejos del tejido e inflaría la puntuación sin decir nada sobre el cáncer. Su tasa de alertas se informa para no ocultar nada.

**Siguiente.** "Antes de calcular ninguna puntuación, escribimos cómo íbamos a decidir."

### Diapositiva 17 · La regla se escribe antes de calcular ninguna puntuación

*Alrededor de 3 minutos.*

**En pantalla.** La regla de decisión en un recuadro oscuro. Tres tarjetas: por qué dos tareas, por qué intervalos, por qué antes.

**Decir.** "El notebook imprime esta regla antes de ajustar nada. Un backbone es claramente mejor solo si su intervalo de confianza supera al del siguiente mejor en las dos tareas. Dos tareas, porque un solo conjunto puede favorecer a un backbone. Intervalos, porque un número más alto dentro de un intervalo solapado no es evidencia. Y antes, para que la regla no se doble hacia el resultado que esperábamos. Una buena regla también tiene que poder decir 'no hay ganador'."

**Datos clave.** `ssl_demo.anomaly.DECISION_RULE`. AUROC con intervalo bootstrap del 95 % sobre teselas de test, cabeza de distancia kNN.

**Siguiente.** "Esto es lo que dijo la primera ejecución."

### Diapositiva 18 · Gana la red sin entrenar. Algo va mal.

*Alrededor de 3 minutos.*

**En pantalla.** Tabla de AUROC con coseno sin centrar: aleatoria 0,885, ImageNet 0,672, DINOv2 0,668. Tarjeta con el titular tentador. Franja con la regla práctica.

**Decir.** "Esto es lo que produjo la primera ejecución del notebook 06. La red sin entrenar obtuvo 0,885 en la tarea A. DINOv2, 0,668. La regla de decisión, aplicada mecánicamente, imprimió que la red aleatoria era claramente mejor en las dos tareas. Habría sido fácil escribir un artículo provocador: 'las features preentrenadas son peores que las aleatorias para detectar anomalías'. La reacción correcta es la sospecha. Una red que no ha aprendido nada debería ser el suelo. Cuando gana el suelo, supón un error hasta que puedas demostrar lo contrario."

**Datos clave.** `results/anomaly_centring.csv`, columna de coseno sin centrar. Mismas teselas y misma partición de ajuste que el resultado final.

**Si preguntan.** *¿Cambiasteis la regla al ver esto?* No. La regla no ha cambiado. Lo que cambió fue la cabeza, tras un diagnóstico, y el código ahora además se niega a nombrar ganador cuando el backbone sin entrenar va delante.

**Siguiente.** "¿Qué estaba pasando?"

### Diapositiva 19 · Todos los vectores aleatorios apuntan en la misma dirección

*Alrededor de 4 minutos. Es la diapositiva en la que conviene ir despacio.*

**En pantalla.** Arriba: una imagen sintética en dos dimensiones de vectores de features sin centrar y centrados. Abajo: coseno medio con el centroide normal por backbone (1,000, 0,820, 0,817). Tarjeta sobre lo que ve la distancia coseno.

**Decir.** "La distancia coseno compara direcciones, no longitudes. Mirad la parte izquierda de la imagen: cuando todos los vectores apuntan casi igual, los ángulos entre ellos son diminutos, y el pequeño residuo que queda decide el orden. En la red sin entrenar, el coseno medio entre cada tesela y la tesela media es 1,000: todos los vectores son prácticamente paralelos. El residuo que queda sigue lo oscura que es la tinción, y las teselas tumorales, densas y moradas, destacan justo en eso. Así que la red aleatoria no detectaba tumores: medía tinta. Los backbones preentrenados tienen el mismo problema en forma más suave: un coseno de alrededor de 0,82 con el centroide, que comprime todas sus distancias en una franja estrecha donde decide la variación irrelevante."

**Datos clave.** Coseno con el centroide: aleatoria 1,000, ImageNet 0,820, DINOv2 0,817. La imagen de arriba es una ilustración, no datos.

**Si preguntan.** *¿Por qué una red sin entrenar produce vectores paralelos?* Tras muchas capas de pesos aleatorios y activaciones ReLU, las salidas son todas positivas y las domina una componente común. Las estadísticas de batch-norm de la inicialización no lo corrigen.

**Siguiente.** "La corrección es una línea."

### Diapositiva 20 · Centrar en la media normal: la aleatoria cae a 0,49 y DINOv2 sube a 0,937

*Alrededor de 3 minutos. Ejecuta aquí en directo la celda de centrado del notebook 06 (segundos sobre las features en caché).*

**En pantalla.** Gráfico de pendiente del AUROC de la tarea A, de coseno sin centrar a centrado, para cada backbone. Tarjeta con el cambio.

**Decir.** "Se resta la media de las features normales de entrenamiento a cada vector, en el banco y en las consultas, y solo después se calcula el coseno. Eso elimina la dirección común y deja la estructura que distingue un tejido de otro. La red sin entrenar cae a 0,492, el azar, justo donde debe estar un suelo. ImageNet sube a 0,902 y DINOv2 a 0,937. Mismas features, mismas teselas, mismo k. Una línea de código."

**Datos clave.** `results/anomaly_centring.csv`. `KNNDistanceHead(center=True)` es ahora el valor por defecto.

**Si preguntan.** *¿Centrar es una técnica conocida?* Sí. Centrar, o blanquear, los embeddings antes de la similitud coseno es habitual en recuperación de información y en detección de anomalías sobre features preentrenadas; es fácil olvidarlo cuando el código calcula el coseno por ti.

**Siguiente.** "¿Cómo supimos que esta era la explicación y no otro ajuste más?"

### Diapositiva 21 · La información estaba en las features desde el principio

*Alrededor de 3 minutos.*

**En pantalla.** Tres tarjetas: techo supervisado, ajustes de la cabeza, cambio de dominio. Franja con el orden de depuración.

**Decir.** "Tres comprobaciones. Primera: ¿está la información en las features? Un clasificador supervisado, tumor frente a normal, entrenado sobre las mismas features en caché, llega a un AUROC de 0,99 en los dos backbones preentrenados y de 0,87 en el aleatorio. Así que sí. Segunda: ¿la lee la cabeza? Cambiar k, quitar el límite del banco de memoria o puntuar cada clase normal por separado apenas cambió nada; centrar lo cambió todo. Tercera: ¿están desplazados los datos de test? Un poco. Con coseno sin centrar, entre el tres y el ocho por ciento de las teselas normales de test cruzaban un umbral fijado para el uno por ciento; tras centrar, entre el 0,6 y el 1,2 por ciento. Ese orden de preguntas sirve para cualquier proceso con embeddings: ¿está la señal?, ¿la lee la cabeza?, ¿se mueven los datos?"

**Datos clave.** Comprobación supervisada: AUROC de 0,986 a 0,993 (preentrenados) y 0,87 (aleatoria). Tasa de falsos positivos en normales de test con el umbral del percentil 99 tras centrar: DINOv2 0,006, ImageNet 0,012.

**Siguiente.** "Ahora, la comparación honesta."

### Diapositiva 22 · Tarea A: DINOv2 claramente por delante. Tarea B: empate. No hay ganador.

*Alrededor de 4 minutos. Enseña la salida de la regla de decisión en el notebook 06.*

**En pantalla.** Barras de AUROC con intervalos para las dos tareas. Franja con la salida de la regla.

**Decir.** "Misma cabeza, mismas teselas, solo cambia el backbone. En tejido colorrectal DINOv2 va claramente por delante: 0,937, y su intervalo no toca el de ImageNet. En metástasis de ganglio, los dos preentrenados están alrededor de 0,75 y sus intervalos se solapan. PatchCamelyon es más difícil para todos, porque su etiqueta se refiere solo a los 32 por 32 píxeles centrales de cada parche, y un embedding de todo el parche diluye un foco tumoral pequeño. Así que la regla dice: ningún backbone es claramente mejor en las dos tareas. Es una respuesta útil. Nos dice dónde ayuda DINOv2 y dónde haría falta otra cosa, como un modelo específico de patología o puntuar a nivel de parche."

**Datos clave.** AUROC de la tarea A: DINOv2 0,937 [0,925; 0,948], ImageNet 0,902 [0,883; 0,919], aleatoria 0,492. Tarea B: ImageNet 0,759 [0,739; 0,778], DINOv2 0,752 [0,732; 0,772], aleatoria 0,683.

**Si preguntan.** *¿Por qué la red aleatoria sigue sacando 0,68 en la tarea B?* Los parches con metástasis difieren en color y densidad del ganglio normal; las features aleatorias captan algo de eso. Queda claramente por debajo de los preentrenados, que es lo que debe hacer un suelo. *¿Y Mahalanobis e Isolation Forest?* Están en `anomaly.csv`, sin ajustar, y siguen poniendo primera a la red aleatoria. Esa es precisamente la lección: la cabeza forma parte de la comparación.

**Siguiente.** "El AUROC ordena; un laboratorio necesita un umbral."

### Diapositiva 23 · Un presupuesto de falsas alarmas del 1 % detecta una cuarta parte de las teselas tumorales

*Alrededor de 3 minutos.*

**En pantalla.** Tabla de DINOv2 en la tarea A: falsas alarmas 0,006, teselas tumorales señaladas 0,285, fondo señalado 0,102, clases normales más difíciles. Tarjeta: no es un fallo, es una elección.

**Decir.** "El AUROC dice lo bien que las puntuaciones ponen el tumor por encima de lo normal. Un laboratorio necesita una decisión: por encima de esta puntuación, mira un patólogo. Fijamos el umbral para que se señale alrededor de un uno por ciento del tejido normal. En ese punto DINOv2 señala el 28,5 por ciento de las teselas tumorales. Es un punto de operación conservador. Un flujo de cribado bajaría el umbral y aceptaría más falsas alarmas para perder menos tumores; la curva permite elegir. Y las falsas alarmas que aparecen se concentran en detritos y moco: tejido raro, no maligno. La galería del notebook 06 las muestra."

**Datos clave.** `results/anomaly.csv`: `fpr_at_99`, `sensitivity_at_99` y la tasa de alertas de `task_a_background`. AUROC por clase de DINOv2: tejido adiposo 1,00, linfocitos 0,99, mucosa normal 0,93, estroma 0,92, músculo liso 0,91, detritos 0,88, moco 0,87.

**Si preguntan.** *¿Sirve detectar una cuarta parte de las teselas tumorales?* A nivel de preparación, sí: una región tumoral abarca muchas teselas, así que señalar una cuarta parte de ellas sigue señalando la región. La agregación a nivel de preparación es el notebook 07 de la v2.

**Siguiente.** "Unas palabras sobre la ingeniería que hizo todo esto barato."

---

## Cierre

### Diapositiva 24 · Barato de repetir, difícil de engañarse

*Alrededor de 2 minutos.*

**En pantalla.** Cuatro tarjetas: features en caché una vez, semillas y particiones fijas, números desde ficheros, suelos y salvaguardas.

**Decir.** "Cuatro hábitos hicieron visible el error y barata la corrección. Las features están en caché, así que repetir todas las cabezas tras la corrección llevó dos minutos en la CPU de un portátil. Semillas y particiones fijas, así que los números son reproducibles y la caché se invalida sola cuando cambian. Cada número sale de un CSV, así que cuando la corrección cambió los resultados, las figuras y estas diapositivas se regeneraron, no se editaron a mano. Y siempre hay un suelo aleatorio en la comparación, con código que se niega a declarar ganador cuando el suelo va delante."

**Datos clave.** Cada notebook funciona en modo rápido en un Colab T4 gratuito en menos de 45 minutos. El fichero de rendimiento no es un ranking de velocidad: el primer backbone de cada conjunto pagó las lecturas en frío de Drive.

**Siguiente.** "Cinco ideas para llevarse."

### Diapositiva 25 · Cinco ideas para llevarse

*Alrededor de 2 minutos.*

**Decir.** "Una: las features autosupervisadas ya describen el tejido; unas cien etiquetas dan un 84 por ciento de exactitud en nueve clases. Dos: su ventaja es máxima cuando escasean las etiquetas. Tres: se puede construir un detector de anomalías solo con tejido normal, con un umbral elegido sin ver un tumor. Cuatro, la que más quiero que recordéis: la cabeza forma parte del experimento, y una red aleatoria que gana es un error hasta que se demuestre lo contrario. Cinco: decide cómo vas a decidir antes de mirar, y deja que la respuesta pueda ser 'no hay ganador'."

**Siguiente.** "Lo que viene."

### Diapositiva 26 · De backbones congelados a entrenar los nuestros

*Alrededor de 1 minuto.*

**Decir.** "Esta es la versión uno. La versión dos entrena nuestros propios backbones, BYOL y un DINO pequeño desde cero, con ablaciones que muestran el colapso. Después, mapas de atención frente a Grad-CAM y el paso de teselas a preparaciones completas, con un mapa de calor que se puede abrir en QuPath. Y un modelo fundacional de patología en la comparación, con su licencia señalada."

### Diapositiva 27 · Pruébalo tú mismo

*Tiempo restante: preguntas.*

**Decir.** "Todo está en el repositorio: notebooks con insignias de Colab, los ficheros de resultados y estas guías en inglés y en español. Los datos son públicos y las licencias están en el README."

---

## Preguntas frecuentes

* **"¿Por qué no reentrenar sin más?"** Reentrenar responde a otra pregunta y necesita etiquetas. La evaluación congelada dice lo buena que es la representación antes de invertir esfuerzo en etiquetar. Con datos solo normales, además, el reentrenamiento puede empeorar las features para detectar anomalías.
* **"¿DINO es mejor que BYOL?"** En la mayoría de las pruebas, los modelos de la familia DINO son más fuertes, y DINOv2 es la base de la mayoría de los modelos fundacionales de patología actuales. La v2 entrena los dos con las mismas teselas para que la comparación sea nuestra.
* **"¿Cómo se elige el umbral?"** Solo con tejido normal, a una tasa de falsas alarmas que el laboratorio pueda revisar. Aquí, el percentil 99 de las normales reservadas, alrededor de un 1 % de falsas alarmas. Nunca se ajusta con tumores.
* **"¿Y las diferencias de escáner y de tinción?"** Son el principal riesgo práctico. La partición de test con pacientes distintos ya muestra un pequeño desplazamiento; la normalización de tinción y la evaluación por escáner están en la lista de la v2.
* **"¿Puedo usar UNI o Virchow comercialmente?"** Revisa cada licencia. Phikon es no comercial; UNI tiene acceso restringido. DINOv2 es Apache-2.0.
* **"¿Se entrenó DINOv2 con estas teselas?"** No; se entrenó con imágenes naturales. Haz la misma pregunta a cualquier modelo de patología antes de fiarte de sus cifras.
