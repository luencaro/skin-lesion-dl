---
title: Diseño experimental
---

Este capítulo declara el diseño con el que se entrenan y evalúan los modelos base, junto con el plan de análisis estadístico. Todas las decisiones se fijan antes de la ejecución definitiva, son comunes a los tres modelos y se derivan del análisis exploratorio y de la [declaración a priori](02-definicion-del-problema.md). Los parámetros se leen de `configs/base.yaml` y del archivo de configuración de cada modelo.

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

## Plan de análisis estadístico

Este plan se declara antes de la ejecución definitiva, de modo que la métrica, las comparaciones y las pruebas no se eligen después de ver los resultados. El nivel de significancia es $\alpha = 0{,}05$ y se reportan siempre los p-valores exactos, ajustados y sin ajustar, junto con un intervalo de confianza del 95 % y un tamaño de efecto.

### Observaciones y variable de respuesta

La ejecución definitiva sigue una validación cruzada repetida de cinco semillas por cinco pliegues, de modo que cada modelo produce 25 corridas, número que se revisará con el resultado del piloto. Cada corrida $j$ aporta dos valores de la métrica primaria: la exactitud balanceada sobre su pliegue de validación y la exactitud balanceada sobre el conjunto de prueba. Como los tres modelos comparten los pares de semilla y pliegue, las observaciones se analizan de forma pareada, y la diferencia entre dos modelos $a$ y $b$ en la corrida $j$ es $d_j = m_{a,j} - m_{b,j}$.

La variable de respuesta del análisis principal es la exactitud balanceada sobre el conjunto de prueba. El desempeño de cada modelo se resume con la media, la desviación estándar y el intervalo de confianza del 95 % entre corridas, calculado por la distribución t de Student y por bootstrap. Ambos intervalos reflejan la variabilidad debida a la semilla y al pliegue, pero no la debida a la muestra de prueba, que es la misma en todas las corridas. Para incorporar esa fuente de variabilidad, se calcula además un intervalo bootstrap sobre el conjunto de prueba, remuestreando lesiones completas, de modo que las imágenes de una misma lesión entren o salgan juntas, y recalculando la exactitud balanceada media de las corridas en cada remuestreo.

### Comparación de los modelos base

Antes de cada comparación se verifica la normalidad de las diferencias pareadas con la prueba de Shapiro–Wilk y gráficos Q–Q. Si no se rechaza la normalidad, la comparación conjunta de los tres modelos se realiza con un ANOVA de medidas repetidas y las comparaciones por pares con la prueba t pareada; en caso contrario, con la prueba de Friedman y la prueba de rangos con signo de Wilcoxon, en su versión exacta. La prueba de Friedman se reporta en ambos casos, con el post-hoc de Nemenyi, por ser la recomendada para comparar clasificadores [@demsar2006]. Las tres comparaciones por pares se corrigen por el método de Holm.

Cada comparación por pares se acompaña de la diferencia media, su intervalo de confianza del 95 % por bootstrap, la d de Cohen para datos pareados, la delta de Cliff y la probabilidad de superioridad, es decir, la proporción de corridas en que un modelo supera al otro. Para cada prueba se reporta también la diferencia mínima detectable con el número de corridas realizado, con potencia de 0,80.

La significancia y la relevancia se evalúan por separado. Una diferencia se considera prácticamente relevante si su magnitud alcanza $\delta = 0{,}02$, y la ausencia de significancia no se interpreta como equivalencia: para afirmar que dos modelos son equivalentes se aplica la prueba de dos pruebas unilaterales (TOST) con margen $\delta$, como análisis separado.

### Análisis complementarios

Las corridas comparten el conjunto de prueba y sus conjuntos de entrenamiento se solapan, de modo que no son estrictamente independientes y las pruebas anteriores tienden a subestimar la varianza. Para acotar este problema, las diferencias de exactitud balanceada en validación se contrastan además con la prueba t corregida de @nadeau2003, que sustituye el factor $1/J$ de la varianza de la media por $1/J + n_{\text{val}}/n_{\text{train}}$, donde $J = 25$ es el número de corridas y $n_{\text{val}}/n_{\text{train}} \approx 1717/6867 \approx 0{,}25$ la razón entre los tamaños de validación y de entrenamiento.

Sobre las predicciones de prueba se aplican dos pruebas específicas de clasificación. La prueba de McNemar compara las tasas de error de cada par de modelos en cada corrida pareada, en su versión exacta cuando las discordancias son menos de 25 y con corrección de continuidad en caso contrario. La prueba de DeLong compara el AUC-ROC de cada par de modelos en cada clase, con un esquema uno contra el resto y corrección de Holm sobre las siete clases, y la diferencia de AUC macro se estima además por bootstrap estratificado. Por último, se descompone la varianza de la métrica primaria en la parte atribuible al modelo y la atribuible a la corrida, lo que indica si las diferencias entre modelos son grandes en relación con la variabilidad del entrenamiento.

### Definición del baseline

El baseline es el modelo con mayor exactitud balanceada media en el conjunto de prueba, reportada con su desviación estándar y su intervalo de confianza del 95 %. Las diferencias entre modelos se expresan en términos absolutos y relativos, respecto del mejor y del peor modelo, con su tamaño de efecto. A partir de la variabilidad observada se fija la mejora mínima detectable que una arquitectura propuesta debería lograr respecto del baseline para ser estadísticamente distinguible, y se compara con el margen de relevancia $\delta$.

El baseline se contrasta con dos cotas de referencia externas. La primera es el desempeño de los especialistas: en el estudio de lectores de @tschandl2019, el grupo con mejor desempeño, el de los lectores con más de diez años de experiencia, acertó en promedio 18,8 de 30 diagnósticos, frente a 25,4 de los tres mejores algoritmos del reto ISIC 2018. La segunda es el desempeño bajo un protocolo con partición agrupada por lesión, en torno a 0,70 de exactitud balanceada [@pinar2026]. Ninguna de las dos es directamente comparable con el baseline, porque se obtuvieron con otras muestras de prueba y, en el primer caso, con otra métrica, pero permiten situarlo respecto del desempeño humano y de la literatura con un protocolo semejante.

## Limitaciones inferenciales

Comparar solo tres modelos impone límites a la potencia de las pruebas que deben reconocerse antes de interpretarlas. Con cinco corridas por modelo, como en el estudio piloto, la prueba de Wilcoxon bilateral no puede alcanzar significancia: su p-valor exacto mínimo es $2/2^5 = 0{,}0625$, mayor que $\alpha$, incluso si un modelo supera al otro en todas las corridas. Por esa razón, el piloto se usa solo para estimar la variabilidad y no para contrastar hipótesis. Con las 25 corridas de la ejecución definitiva, el p-valor mínimo de la prueba de Wilcoxon es $2/2^{25} \approx 6 \times 10^{-8}$, y la prueba deja de estar limitada por el número de corridas.

La prueba de Friedman con tres modelos tiene poca potencia, y el post-hoc de Nemenyi es conservador, de modo que una ausencia de significancia no demuestra que los modelos sean equivalentes. La corrección de Holm con tres comparaciones exige, en el paso más estricto, un p-valor menor que $\alpha/3 \approx 0{,}0167$. Además, como se señaló, las corridas no son independientes, porque comparten el conjunto de prueba y se solapan en entrenamiento; las pruebas que las tratan como independientes son, por tanto, optimistas, lo que motiva la prueba corregida de Nadeau y Bengio.

Para compensar estas limitaciones, el análisis no se apoya solo en los p-valores. Se usa la validación cruzada repetida para disponer de más observaciones pareadas, se reporta la diferencia mínima detectable con el número de corridas disponible, se acompaña cada diferencia de su tamaño de efecto con intervalo de confianza y de la probabilidad de superioridad, y se recurre a la prueba TOST para cualquier afirmación de equivalencia. El objetivo es fijar el baseline y su variabilidad con honestidad; un resultado no concluyente, bien cuantificado, se reporta como tal.
