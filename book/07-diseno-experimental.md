---
title: Diseño experimental
---

Este capítulo declara el diseño con el que se entrenan y evalúan los modelos base. Todas las decisiones se fijan antes de la ejecución definitiva, son comunes a los tres modelos y se derivan del análisis exploratorio y de la [declaración a priori](02-definicion-del-problema.md). Los parámetros se leen de `configs/base.yaml` y del archivo de configuración de cada modelo.

## Variables de entrada y objetivo

La entrada de cada modelo es la imagen dermatoscópica original de 600 × 450 píxeles, redimensionada a 224 × 224 × 3 por interpolación bilineal con antialiasing y sin conservar la relación de aspecto, con valores de intensidad en [0, 255]. El análisis exploratorio mostró que todas las imágenes comparten resolución, formato y profundidad de bits, por lo que el redimensionado es idéntico para todas las clases y no introduce una diferencia sistemática entre ellas. La variable objetivo es el diagnóstico, codificado como un entero entre 0 y 6. Al tratarse de un problema de clasificación, no hay horizonte de predicción.

## División del conjunto y esquema de validación

La partición es la definida en el análisis exploratorio: un conjunto de prueba de 1431 imágenes, congelado mediante su hash, y un conjunto de entrenamiento de 8584 imágenes dividido en cinco pliegues estratificados por clase y agrupados por lesión. La unidad experimental es la corrida, definida por un par formado por una semilla $s$ y un pliegue de validación $f$. En cada corrida, el modelo se entrena con los cuatro pliegues restantes, alrededor de 6870 imágenes, y el pliegue $f$, de unas 1717 imágenes, se usa para la parada temprana y la selección de la mejor época.

Los tres modelos se ejecutan sobre los mismos pares de semilla y pliegue, de modo que sus resultados forman observaciones pareadas, y cada par combina una semilla distinta con una partición de validación distinta, como recomienda el protocolo. El conjunto de prueba no interviene en el ajuste de hiperparámetros ni en el estudio piloto: se evalúa una sola vez al final de cada corrida definitiva, con los hiperparámetros y los pesos ya fijados.

Para la selección de hiperparámetros se usa una partición fija de validación, el pliegue 0, en lugar de una validación cruzada anidada. La validación cruzada anidada multiplicaría por cinco el costo del ajuste, que en VGG-16 resulta prohibitivo con el hardware disponible. La selección sobre un único pliegue introduce un sesgo optimista en el desempeño de validación de ese pliegue, pero el sesgo afecta por igual a los tres modelos y no al conjunto de prueba.

## Pipeline

### Preprocesamiento y aumento de datos

La normalización de intensidades se realiza dentro del modelo con la función `preprocess_input` asociada a cada arquitectura, que reproduce la normalización con la que se entrenaron sus pesos en ImageNet; en MobileNetV2, por ejemplo, lleva las intensidades al intervalo [−1, 1]. Ninguna transformación se ajusta con datos de validación o de prueba. El único estadístico calculado a partir de los datos son los pesos de la pérdida, que se obtienen del conjunto de entrenamiento de cada corrida.

Las imágenes redimensionadas se almacenan en memoria y, solo en entrenamiento, se les aplica un aumento de datos que preserva la etiqueta. Como las imágenes dermatoscópicas no tienen una orientación canónica, se aplican volteos horizontales y verticales y rotaciones en múltiplos de 90°. Se aplica además un acercamiento aleatorio de hasta el 10 %, mediante un recorte de lado $1/u$ con $u \sim U(1; 1{,}1)$ seguido de un nuevo redimensionado. El acercamiento se limita a ese valor porque el 15,9 % de las lesiones toca el borde de la imagen, y el 34,5 % en el caso de akiec. Por último, se varían el brillo, el contraste y la saturación con una amplitud de 0,10 y el tono con una de 0,02, para reducir la dependencia del modelo respecto del color, que el análisis exploratorio mostró fuertemente asociado a la fuente de adquisición. Todas las operaciones aleatorias se generan a partir de la semilla de la corrida.

### Arquitectura

Las arquitecturas se toman de `keras.applications`, con los pesos preentrenados en ImageNet, en lugar de implementarse capa por capa. Esta decisión garantiza que la arquitectura sea idéntica a la original y reproduce la estrategia de aprendizaje por transferencia que emplea la literatura revisada [@menegola2017; @brinker2018]. Entrenar desde una inicialización aleatoria, con 8584 imágenes de entrenamiento, produciría un baseline subestimado y no comparable con los estudios incluidos.

Cada arquitectura conserva su cabeza de clasificación original, con la capa de salida sustituida por una capa densa de siete neuronas con activación softmax, precedida por una capa de dropout cuya tasa es un hiperparámetro. En MobileNetV2, la cabeza consiste en un pooling global promedio, el dropout y la capa de salida. Las capas de normalización por lotes del backbone se mantienen en modo de inferencia durante todo el entrenamiento, de modo que sus estadísticas de ImageNet no se alteran con lotes pequeños.

### Entrenamiento

El entrenamiento se divide en dos fases. En la primera, de calentamiento, el backbone permanece congelado y solo se entrena la cabeza durante 5 épocas, con AdamW y una tasa de aprendizaje constante de $10^{-3}$, para que la capa de salida, inicializada al azar, no propague gradientes grandes hacia los pesos preentrenados. En la segunda, de ajuste fino, se descongela todo el modelo y se entrena con AdamW durante un máximo de 30 épocas. La tasa de aprendizaje, seleccionada en el ajuste de hiperparámetros, decae según una función coseno hasta 0 a lo largo de las 30 épocas. En ambas fases, el decaimiento de pesos no se aplica a los sesgos ni a los parámetros de las capas de normalización.

La pérdida es la entropía cruzada ponderada definida en la [formulación del problema](02-definicion-del-problema.md). El tamaño de lote es 32 y se usa precisión mixta (float16), con la capa de salida en float32 para preservar la estabilidad numérica de la softmax. Al final de cada época, el modelo predice una vez el pliegue de validación, y con esas predicciones se calculan la pérdida, la exactitud, la exactitud balanceada y el F1 macro de validación. La parada temprana actúa en la fase de ajuste fino: se detiene el entrenamiento si la exactitud balanceada de validación no mejora durante 5 épocas consecutivas y se restauran los pesos de la mejor época de esa fase.

### Evaluación

Con los pesos de la mejor época se calculan, sobre validación y, en las corridas definitivas, sobre prueba, la exactitud balanceada, la exactitud, el F1 macro y ponderado, la precisión, la sensibilidad y el F1 por clase, el AUC-ROC macro uno contra el resto, el AUC-PR macro, el coeficiente de correlación de Matthews, el error de calibración esperado con 15 intervalos y la matriz de confusión. Las probabilidades predichas se guardan por imagen, lo que permite aplicar después las pruebas de McNemar y DeLong y los remuestreos bootstrap sin volver a entrenar.

## Ajuste de hiperparámetros

Los tres modelos reciben el mismo esfuerzo de ajuste: el mismo método, el mismo presupuesto y el mismo espacio de búsqueda. El método es una búsqueda aleatoria de 12 configuraciones, generadas con una semilla fija (2026). Como la semilla es común, los tres modelos evalúan exactamente las mismas 12 configuraciones, lo que elimina la posibilidad de que un modelo resulte favorecido por una muestra de configuraciones más afortunada. El espacio de búsqueda es el siguiente.

| Hiperparámetro | Espacio |
|---|---|
| Tasa de aprendizaje del ajuste fino | Log-uniforme entre $10^{-5}$ y $10^{-3}$ |
| Decaimiento de pesos | Log-uniforme entre $10^{-6}$ y $10^{-3}$ |
| Tasa de dropout | {0,0; 0,2; 0,4} |

El resto de los hiperparámetros se mantiene fijo en los valores declarados en este capítulo: tamaño de lote, duración y tasa de la fase de calentamiento, paciencia de la parada temprana y aumento de datos. Cada configuración se evalúa en una corrida con semilla 0 y pliegue de validación 0, con un máximo de 15 épocas en la fase de ajuste fino para contener el costo, y sin evaluar nunca el conjunto de prueba. Se selecciona la configuración con mayor exactitud balanceada de validación; los empates se resuelven a favor de la configuración evaluada primero.

## Reproducibilidad

En cada corrida se fijan las semillas de Python, NumPy y TensorFlow, y las operaciones aleatorias del aumento de datos se generan con un generador sembrado con la semilla de la corrida. No se impone determinismo estricto, porque varias operaciones de cuDNN en GPU no son deterministas y forzarlo ralentizaría el entrenamiento; en consecuencia, repetir una corrida con la misma semilla puede producir diferencias pequeñas, que forman parte de la variabilidad entre ejecuciones que estima el estudio piloto.

Cada corrida guarda su configuración resuelta, el historial de entrenamiento por época de ambas fases, las métricas, las probabilidades predichas por imagen y un registro con los tiempos de entrenamiento por fase, el tiempo de inferencia por imagen (medido en lotes de 32 tras una predicción de calentamiento), el número de parámetros entrenables y no entrenables de cada fase, la GPU empleada y las versiones de Python, TensorFlow, CUDA y cuDNN. Las dependencias del proyecto se fijan en `uv.lock`. Los experimentos se ejecutan en una GPU NVIDIA GeForce RTX 3050 Laptop de 6 GB.

## Estudio piloto

Antes de la ejecución definitiva, cada modelo se ejecuta en cinco corridas independientes, con los pares de semilla y pliegue (0, 0), (1, 1), (2, 2), (3, 3) y (4, 4), evaluadas únicamente sobre validación. Para cada modelo se reportan la media, la desviación estándar y el coeficiente de variación de la exactitud balanceada.

Con los tres modelos, la desviación estándar de las diferencias pareadas entre modelos, $\hat{\sigma}_d$, permite expresar el margen de relevancia práctica en unidades de variabilidad, $\delta / \hat{\sigma}_d$, y calcular el número de corridas necesario para detectar $\delta = 0{,}02$ con potencia $1 - \beta = 0{,}80$ y $\alpha = 0{,}05$,

$$
R \approx \frac{\left(z_{1-\alpha/2} + z_{1-\beta}\right)^2 \hat{\sigma}_d^2}{\delta^2},
$$

así como la diferencia mínima detectable con las corridas realizadas, $\left(z_{1-\alpha/2} + z_{1-\beta}\right) \hat{\sigma}_d / \sqrt{R}$. Los resultados del piloto, sus supuestos y el número de corridas resultante se registran en `preregistro.md`, con fecha, antes de la ejecución definitiva.
