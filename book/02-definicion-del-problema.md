---
title: Definición del problema y del conjunto de datos
---

## Formulación del problema

El proyecto aborda un problema de clasificación de imágenes, el séptimo de los tipos de problema admitidos. Cada observación es una imagen dermatoscópica en color $x \in \mathcal{X} = \{0, \dots, 255\}^{H \times W \times 3}$, con $H = 450$ y $W = 600$ píxeles, y su etiqueta pertenece al conjunto de siete categorías diagnósticas

$$
\mathcal{Y} = \{\text{nv}, \text{mel}, \text{bkl}, \text{bcc}, \text{akiec}, \text{vasc}, \text{df}\}, \qquad K = |\mathcal{Y}| = 7,
$$

que corresponden a nevus melanocítico, melanoma, lesión queratósica benigna, carcinoma basocelular, queratosis actínica o carcinoma intraepitelial, lesión vascular y dermatofibroma. Se busca una función $f_\theta : \mathcal{X} \to \Delta^{K-1}$, parametrizada por una red neuronal con pesos $\theta$, que asigne a cada imagen un vector de probabilidades sobre el símplex $\Delta^{K-1}$, de modo que la clase predicha sea $\hat{y} = \arg\max_k f_\theta(x)_k$. Antes de entrar a la red, cada imagen se redimensiona a la resolución de entrada del modelo, que para los modelos base es de $224 \times 224 \times 3$.

Los parámetros se estiman minimizando la entropía cruzada ponderada sobre el conjunto de entrenamiento $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$,

$$
\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} w_{y_i} \log f_\theta(x_i)_{y_i}, \qquad w_k = \frac{N}{K \, N_k},
$$

donde $N_k$ es el número de imágenes de entrenamiento de la clase $k$. La ponderación compensa el desbalance documentado en el análisis exploratorio, en el que la clase mayoritaria tiene 58,30 veces más imágenes que la minoritaria, y hace que el riesgo empírico se alinee con la métrica primaria, que pondera por igual todas las clases.

## Pregunta de investigación e hipótesis

La pregunta que guía este entregable es si los modelos de referencia identificados en la revisión de la literatura, VGG-16, ResNet-50 y MobileNetV2, difieren en su exactitud balanceada sobre HAM10000 cuando se entrenan bajo un mismo pipeline, una misma partición agrupada por lesión y un mismo presupuesto de ajuste de hiperparámetros, y qué nivel de desempeño establece el mejor de ellos como umbral para una arquitectura propuesta.

La hipótesis operativa que contrasta el benchmark es la de igualdad de desempeño. Denotando por $\mu_m$ la exactitud balanceada esperada del modelo $m$ sobre la distribución de los datos, la hipótesis nula es $H_0 : \mu_{\text{VGG-16}} = \mu_{\text{ResNet-50}} = \mu_{\text{MobileNetV2}}$, frente a la alternativa de que al menos un modelo difiere de los demás. Una diferencia se considerará prácticamente relevante solo si su magnitud alcanza el margen $\delta$ declarado en la sección siguiente, de modo que la significancia estadística y la relevancia práctica se evalúan por separado. El mismo margen define la mejora mínima que una arquitectura propuesta deberá superar respecto del mejor modelo base.

## Declaración a priori

Las siguientes decisiones se fijan antes de entrenar cualquier modelo y se heredan en los entregables posteriores.

| Elemento | Declaración |
|---|---|
| Métrica primaria | Exactitud balanceada (promedio de la sensibilidad de las siete clases) |
| Métricas secundarias | F1 macro, exactitud, precisión, sensibilidad y F1 por clase, AUC-ROC macro uno contra el resto, AUC-PR macro, coeficiente de correlación de Matthews, error de calibración esperado y diagrama de fiabilidad |
| Margen de relevancia práctica | $\delta = 0{,}02$ en exactitud balanceada, en valor absoluto |
| Nivel de significancia | $\alpha = 0{,}05$, con corrección de Holm para las comparaciones pareadas |
| Comparaciones de interés | Las tres comparaciones pareadas entre los modelos base y la comparación conjunta de los tres |
| Unidad de independencia | La lesión (`lesion_id`) |
| Partición | Estratificada por clase y agrupada por lesión: prueba de 1431 imágenes (14,29 %) y entrenamiento de 8584 imágenes en cinco pliegues, congelada mediante su hash |

La exactitud balanceada se elige como métrica primaria porque el conjunto está fuertemente desbalanceado: un clasificador que asigne siempre la clase nv obtiene una exactitud de 0,670 y una exactitud balanceada de 0,143. Es además la métrica oficial de la tarea 3 del reto ISIC 2018 [@codella2019], lo que permite situar los resultados respecto de la literatura.

El margen $\delta = 0{,}02$ se justifica por la granularidad del conjunto de prueba. Como la exactitud balanceada promedia la sensibilidad de las siete clases, una sola imagen de df, de la que hay 17 en prueba, modifica la métrica en $1/(7 \cdot 17) \approx 0{,}0084$, y una de vasc, con 20 imágenes, en $1/(7 \cdot 20) \approx 0{,}0071$. Un margen de 0,02 equivale, por tanto, a clasificar correctamente entre dos y tres imágenes adicionales de las clases minoritarias; diferencias menores pueden deberse a unas pocas imágenes y no a una ventaja sistemática del modelo.

La unidad de independencia es la lesión y no la imagen, porque el 26,18 % de las lesiones tiene entre dos y seis imágenes, que comparten diagnóstico y apariencia. HAM10000 no incluye un identificador de paciente, de modo que lesiones distintas de un mismo paciente no pueden agruparse; esta dependencia residual se reconoce como una limitación.

## Justificación del conjunto de datos

El problema tiene relevancia clínica directa: la clasificación de las lesiones pigmentadas de la piel condiciona la decisión de extirpar o vigilar una lesión, y la concordancia entre expertos en la lectura dermatoscópica es apenas moderada [@carrera2016], como se desarrolla en la [descripción del proyecto](01-descripcion-del-proyecto.md).

HAM10000 es un conjunto público, distribuido en Harvard Dataverse [@tschandl2018data] bajo la licencia Creative Commons Atribución-NoComercial 4.0 (CC BY-NC 4.0), que permite su uso con fines académicos pero no su redistribución con fines comerciales. Fue construido a lo largo de veinte años a partir de dos centros, el Departamento de Dermatología de la Universidad Médica de Viena, en Austria, y la consulta de cáncer de piel de Cliff Rosendahl, en Queensland, Australia, con distintos equipos y protocolos de adquisición [@tschandl2018]. Esa procedencia se refleja en los metadatos como cuatro subconjuntos (rosendahl, vidir_modern, vidir_molemax y vienna_dias), cuya influencia sobre el color, el viñeteado y la composición por clase se documenta en el análisis exploratorio.

El conjunto es la referencia más utilizada en la literatura revisada para la clasificación dermatoscópica en siete clases y fue el conjunto de entrenamiento de la tarea 3 del reto ISIC 2018 [@codella2019], lo que garantiza que los resultados sean comparables con los estudios incluidos en el [estado del arte](03-estado-del-arte.md).

## Descripción técnica

El conjunto contiene 10015 imágenes dermatoscópicas de 7470 lesiones, todas en formato JPEG, RGB, con 8 bits por canal y una resolución de 600 × 450 píxeles, es decir, 810000 valores por imagen. Se acompaña de un archivo CSV de metadatos con ocho columnas, que registra para cada imagen su identificador, el de la lesión, el diagnóstico (`dx`), el método diagnóstico (`dx_type`), la edad, el sexo, la localización anatómica y la fuente, y de una máscara binaria de segmentación de la lesión por imagen. La unidad de observación es la imagen.

La variable objetivo es el diagnóstico, con siete categorías. Su etiquetado es manual y procede de cuatro métodos: la histopatología, que confirma el 53,32 % de las imágenes; el seguimiento clínico de nevus que no muestran cambios en controles sucesivos (36,98 %); el consenso de expertos (9,01 %) y la microscopía confocal (0,69 %) [@tschandl2018]. El ruido de etiqueta esperado es bajo en mel, bcc y akiec, cuyas etiquetas son histopatológicas en su totalidad, y mayor en vasc, df y bkl, donde el consenso de expertos sustenta entre el 24,0 % y el 52,8 % de las imágenes. No existe un segundo anotador que permita estimarlo directamente.

Las observaciones no son independientes. Las imágenes se agrupan por lesión, 1956 lesiones tienen más de una imagen y estas comparten el diagnóstico, por lo que cualquier partición que separe imágenes de una misma lesión introduce fuga de información. Las imágenes proceden además de cuatro dominios de adquisición, asociados tanto con la clase como con el color de las imágenes. No hay estructura temporal ni espacial. Esta dependencia determina la partición agrupada por lesión y obliga a interpretar con cautela las pruebas estadísticas que tratan las imágenes como independientes.

Se utiliza la versión 4 del conjunto en Harvard Dataverse (doi: 10.7910/DVN/DBW86T, UNF:6:KCZFcBLiFE5ObWcTc2ZBOA==), descargada el 12 de noviembre de 2026. Los hashes SHA-256 de los archivos descargados se registran en `data/hashes.json`, y los de la partición y de la lista de imágenes de prueba en `data/splits_hash.json`. El conjunto de prueba oficial del reto ISIC 2018, distribuido en la misma versión, no se utiliza.
