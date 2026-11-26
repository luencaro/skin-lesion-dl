---
title: Modelos del estado del arte
---

El benchmark del Entregable 2 incluye cinco modelos del estado del arte, además de los tres modelos base del Entregable 1, un baseline no profundo (regresión logística sobre embeddings congelados de MobileNetV2) y un baseline trivial (asignar siempre la clase nv). El protocolo pide diez modelos del estado del arte; por la limitación de cómputo, con una GPU de 6 GB y unos 35 minutos por corrida, se seleccionan cinco, y la desviación se declara en el preregistro. El décimo modelo del proyecto será la arquitectura propia del Entregable 3.

## Selección

Los candidatos provienen de la revisión ya realizada: la tabla de frecuencias de la fase sistemática y las fases narrativas del [estado del arte](03-estado-del-arte.md). Solo se usó Scopus, porque no se tuvo acceso a Web of Science ni a IEEE Xplore, lo que constituye una limitación de la búsqueda. Cada candidato recibe de 0 a 2 puntos en cinco criterios, con reglas explícitas:

- **Frecuencia:** número de estudios incluidos que usan la familia (0: uno o ninguno; 1: de dos a cuatro; 2: cinco o más).
- **Recencia:** año de publicación de la arquitectura (0: hasta 2015; 1: de 2016 a 2019; 2: desde 2020).
- **Desempeño comparable:** evidencia sobre HAM10000 (0: ninguna; 1: reportada con partición por imagen; 2: reportada con exactitud balanceada o partición agrupada por lesión).
- **Implementación:** disponibilidad en el entorno del proyecto (0: ninguna; 1: en `keras_hub` o código oficial en otro framework; 2: en `keras.applications` con pesos de ImageNet).
- **Representatividad:** familia arquitectónica (2: familia aún no representada; 1: familia ya representada por un candidato con mayor puntaje o por un modelo base).

Se eligen los cinco candidatos con mayor puntaje total, con un máximo de uno por familia; los empates se resuelven a favor de la familia aún no representada.

| Candidato | Familia | Frecuencia | Recencia | Desempeño | Implementación | Representatividad | Total | Seleccionado |
|---|---|---|---|---|---|---|---|---|
| DenseNet-201 | CNN densa | 2 | 1 | 2 [@nguyen2022] | 2 | 2 | 9 | Sí |
| Inception-V3 | CNN con módulos inception | 2 | 1 | 1 [@qian2022] | 2 | 2 | 8 | Sí |
| ConvNeXt-Tiny | CNN moderna | 0 | 2 | 2 [@pinar2026] | 2 | 2 | 8 | Sí |
| ViT-B/16 | Transformador | 0 | 2 | 2 [@pinar2026] | 1 | 2 | 7 | Sí |
| EfficientNet-B0 | CNN con escalado compuesto | 0 | 1 | 1 [@tahir2023] | 2 | 2 | 6 | Sí |
| Inception-ResNet-V2 | CNN con módulos inception | 1 | 1 | 1 [@nguyen2022] | 2 | 1 | 6 | No |
| Xception | CNN con módulos inception | 1 | 1 | 1 [@mehmood2023] | 2 | 1 | 6 | No |
| Swin-Tiny | Transformador | 0 | 2 | 2 [@pinar2026] | 1 | 1 | 6 | No |
| NASNet | CNN obtenida por búsqueda | 0 | 1 | 1 | 2 | 1 | 5 | No |
| AlexNet | CNN clásica | 1 | 0 | 1 [@alam2022] | 0 | 1 | 3 | No |

Los candidatos excluidos con más presencia en la literatura quedan fuera por redundancia o por costo. Inception-ResNet-V2 y Xception pertenecen a la misma familia que Inception-V3, que obtiene mayor puntaje. Swin-Tiny es el segundo transformador, y ViT-B/16 representa ya esa familia con una arquitectura más simple. NASNet comparte con EfficientNet el origen en la búsqueda automática de arquitecturas y es mucho más costoso en su versión grande. AlexNet es la arquitectura más antigua y su familia está representada por VGG-16.

Los cinco modelos seleccionados, junto con los tres modelos base, cubren cinco familias: CNN clásicas (VGG-16), residuales (ResNet-50), eficientes (MobileNetV2 y EfficientNet-B0), de conectividad densa o multirrama (DenseNet-201 e Inception-V3) y basadas en atención (ViT-B/16), con ConvNeXt-Tiny como CNN modernizada a partir del diseño de los transformadores. Todos reciben la misma resolución de entrada, 224 × 224, los mismos pesos de ImageNet como fuente de preentrenamiento y el mismo pipeline que los modelos base.

## Análisis por modelo

Todos los modelos tienen la arquitectura definida en su artículo original y en la implementación de Keras, por lo que la búsqueda de arquitectura en dos fases no se aplica a ninguno; cada uno conserva su cabeza de clasificación original con la capa de salida sustituida por siete clases. Las cifras de desempeño de los artículos originales se refieren a ImageNet y ninguno reporta varianza entre ejecuciones ni pruebas de significancia.

### Inception-V3

@szegedy2016 rediseñaron el módulo inception para reducir el costo sin perder capacidad. Cada módulo procesa la misma entrada con ramas paralelas de distinto campo receptivo y concatena sus salidas, $y = \operatorname{concat}\left(f_{1 \times 1}(x), f_{3 \times 3}(x), f_{5 \times 5}(x), f_{\text{pool}}(x)\right)$, y la innovación consiste en factorizar las convoluciones grandes en otras más pequeñas y asimétricas (una de 7 × 7 en una de 1 × 7 seguida de una de 7 × 1), además del suavizado de etiquetas. Tiene unos 23,8 millones de parámetros, un costo del orden de 5 GFLOPs a 299 × 299 y un error top-1 de 21,2 % en ImageNet. Su fortaleza es la extracción multiescala; su limitación, una arquitectura compleja y una resolución nativa de 299 × 299, que aquí se reduce a 224 × 224 por equidad con los demás modelos.

```{mermaid}
flowchart LR
  A[Imagen 224×224] --> B[Tallo convolucional]
  B --> C[Módulos inception con ramas paralelas]
  C --> D[Reducción de resolución]
  D --> E[Módulos con convoluciones factorizadas]
  E --> F[Pooling global]
  F --> G[Softmax de 7 clases]
```

### DenseNet-201

@huang2017 conectan cada capa con todas las anteriores dentro de un bloque denso: la capa $\ell$ recibe la concatenación de los mapas de todas las capas previas, $x_\ell = H_\ell\left([x_0, x_1, \dots, x_{\ell-1}]\right)$, donde $H_\ell$ es una secuencia de normalización por lotes, ReLU y convolución. Esa conectividad favorece la reutilización de características y el flujo del gradiente con menos parámetros que una red residual comparable. DenseNet-201 tiene unos 20 millones de parámetros, un costo de unos 4,3 GFLOPs a 224 × 224 y un error top-1 de 22,6 % en ImageNet. Su limitación es el consumo de memoria de las activaciones concatenadas durante el entrenamiento.

```{mermaid}
flowchart LR
  A[Imagen 224×224] --> B[Convolución inicial]
  B --> C[Bloque denso 1]
  C --> D[Transición]
  D --> E[Bloques densos 2 a 4 con transiciones]
  E --> F[Pooling global]
  F --> G[Softmax de 7 clases]
```

### EfficientNet-B0

@tan2019 proponen escalar de forma conjunta la profundidad, el ancho y la resolución con un coeficiente compuesto $\phi$, de modo que $d = \alpha^\phi$, $w = \beta^\phi$ y $r = \gamma^\phi$ con $\alpha \beta^2 \gamma^2 \approx 2$. La red base B0 se obtuvo por búsqueda de arquitectura y usa bloques MBConv, convoluciones invertidas con cuello de botella como las de MobileNetV2, con atención de canal squeeze-and-excitation. Tiene 5,3 millones de parámetros, 0,39 GFLOPs y un 77,1 % de exactitud top-1 en ImageNet. Su fortaleza es la eficiencia; su limitación, la sensibilidad del entrenamiento a la configuración de regularización.

```{mermaid}
flowchart LR
  A[Imagen 224×224] --> B[Convolución inicial]
  B --> C[Bloques MBConv con squeeze-and-excitation]
  C --> D[Convolución 1×1]
  D --> E[Pooling global]
  E --> F[Softmax de 7 clases]
```

### ConvNeXt-Tiny

@liu2022 modernizan una ResNet incorporando, paso a paso, decisiones de diseño de los transformadores de visión: un tallo de parches 4 × 4 sin solapamiento, convoluciones en profundidad de 7 × 7, bloques de cuello de botella invertido, normalización de capa y activación GELU. Cada bloque calcula $y = x + \operatorname{MLP}\left(\operatorname{LN}\left(\operatorname{DWConv}_{7 \times 7}(x)\right)\right)$. ConvNeXt-Tiny tiene unos 28 millones de parámetros, 4,5 GFLOPs y un 82,1 % de exactitud top-1 en ImageNet. Demuestra que una red puramente convolucional puede igualar a los transformadores; su limitación es un costo mayor que el de las CNN eficientes.

```{mermaid}
flowchart LR
  A[Imagen 224×224] --> B[Tallo de parches 4×4]
  B --> C[Etapas de bloques ConvNeXt]
  C --> D[Pooling global y normalización de capa]
  D --> E[Softmax de 7 clases]
```

### ViT-B/16

@dosovitskiy2021 aplican un transformador estándar a la imagen dividida en parches de 16 × 16, que se proyectan linealmente y se suman a un embedding de posición. Cada uno de los 12 bloques aplica autoatención multicabeza, $\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(QK^\top / \sqrt{d_k}\right) V$, seguida de un perceptrón multicapa, con conexiones residuales y normalización de capa, y la clasificación se hace sobre un token especial. Tiene unos 86 millones de parámetros y unos 17,6 GFLOPs a 224 × 224, y alcanza alrededor de 84 % de exactitud top-1 en ImageNet tras preentrenarse en ImageNet-21k. Su fortaleza es la capacidad de modelar relaciones globales; su limitación, la falta de sesgo inductivo local, que lo hace dependiente de grandes conjuntos de preentrenamiento, y un costo que crece con el cuadrado del número de parches.

```{mermaid}
flowchart LR
  A[Imagen 224×224] --> B[196 parches de 16×16]
  B --> C[Proyección lineal y embedding de posición]
  C --> D[12 bloques de autoatención y MLP]
  D --> E[Token de clasificación]
  E --> F[Softmax de 7 clases]
```

## Comparación crítica

| Modelo | Familia | Parámetros | GFLOPs (224 × 224) | Complejidad respecto a los píxeles | Datos de preentrenamiento | Fortaleza | Limitación |
|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | CNN con escalado compuesto | 5,3 M | 0,39 | Lineal | ImageNet-1k | Eficiencia | Sensibilidad a la regularización |
| DenseNet-201 | CNN densa | 20 M | 4,3 | Lineal | ImageNet-1k | Reutilización de características | Memoria de entrenamiento |
| Inception-V3 | CNN multirrama | 23,8 M | ≈ 5 (a 299 × 299) | Lineal | ImageNet-1k | Extracción multiescala | Resolución nativa distinta |
| ConvNeXt-Tiny | CNN moderna | 28 M | 4,5 | Lineal | ImageNet-1k | Desempeño de transformador con sesgo local | Costo intermedio |
| ViT-B/16 | Transformador | 86 M | 17,6 | Cuadrática en el número de parches | ImageNet-21k | Relaciones globales | Dependencia del preentrenamiento y mayor costo |

Los parámetros, FLOPs y resultados de ImageNet son los reportados por los autores; los parámetros exactos con la cabeza de siete clases, los FLOPs a 224 × 224 y la memoria pico se miden en la implementación y se reportan en el benchmark.

En cuanto a las tendencias, la literatura sobre HAM10000 pasa de las CNN clásicas a los transformadores y a los modelos fundacionales [@xin2022; @yan2025], y a la fusión con metadatos [@vachmanus2023], pero los estudios con protocolo estricto no encuentran diferencias significativas entre ResNet-50 y arquitecturas como ConvNeXt, Swin o ViT [@pinar2026]. La reproducibilidad de la literatura es baja: de los quince estudios de la fase sistemática, ninguno repite el entrenamiento con varias semillas de forma completa, solo uno aplica pruebas estadísticas de forma parcial y solo tres (20 %) declaran su código como público [@datta2021; @baig2023; @nguyen2022]; el resto se limita a señalar la disponibilidad del conjunto de datos o a ofrecer los datos bajo solicitud.

## Gap de investigación

Ningún estudio incluido compara redes convolucionales clásicas, eficientes, densas y modernas con un transformador de visión sobre HAM10000 con una partición agrupada por lesión, el mismo presupuesto de ajuste declarado para todos los modelos, validación cruzada repetida e inferencia con corrección por comparaciones múltiples, ni analiza si las diferencias entre ellas dependen de la fuente de adquisición de la imagen. Esta brecha motiva las preguntas RQ1 y RQ3 del [capítulo de preguntas de investigación](12-preguntas-y-datos.md) y puede refutarse con la hoja de extracción de la fase sistemática.
