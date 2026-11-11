---
title: Modelos base
---

Esta página analiza los tres modelos base identificados en el [estado del arte](estado-del-arte.md), los compara entre sí y concluye con la brecha que la literatura deja abierta.

## Elementos comunes

Los tres modelos son redes convolucionales aplicadas al mismo problema: la clasificación de imágenes dermatoscópicas en las siete categorías de HAM10000. Una capa convolucional con núcleo de $k \times k$, $C_{l-1}$ canales de entrada y $C_l$ de salida tiene $k^2 C_{l-1} C_l + C_l$ parámetros y un costo del orden de $\mathcal{O}(H_l W_l\, k^2 C_{l-1} C_l)$ multiplicaciones y acumulaciones, expresiones a partir de las cuales se obtiene la complejidad de cada arquitectura. Las tres terminan en una capa lineal con $K = 7$ salidas y softmax, y se entrenan minimizando la entropía cruzada, ponderada por clase cuando se quiere compensar el desbalance:

$$
\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} \omega_k \, y_{ik} \log \hat{p}_{ik}.
$$

En la literatura revisada se inicializan con pesos de ImageNet, se reemplaza la capa de 1.000 salidas por una de siete y se ajustan todos los parámetros con Adam o con descenso por gradiente estocástico con momento y parada temprana. Los conteos de parámetros que se presentan corresponden a las implementaciones de `keras.applications`, que son las que se emplearán en este proyecto, y se refieren a los parámetros entrenables; Keras contabiliza además como no entrenables las medias y varianzas móviles de la normalización por lotes. Si $F$ es el costo de una propagación hacia delante por imagen, un paso de entrenamiento con un lote de $B$ imágenes cuesta del orden de $3BF$ y la inferencia, $BF$. Los estudios que informan tiempos de entrenamiento [@ali2021; @baig2023] lo hacen en hardware heterogéneo y sin dispersión, por lo que sus cifras no son extrapolables.

## VGG-16

VGG-16 [@simonyan2015] (doi: 10.48550/arXiv.1409.1556) es una red convolucional plana: la información recorre una única ruta, sin ramificaciones ni atajos. Trece capas convolucionales de 3 × 3, agrupadas en cinco bloques con 64, 128, 256, 512 y 512 filtros y separadas por agrupaciones por máximo, reducen una imagen de 224 × 224 a un mapa de 512 × 7 × 7, al que siguen tres capas densas de 4.096, 4.096 y $K$ unidades. Usa activaciones ReLU, abandono de 0,5 en las capas densas y no incluye normalización por lotes. Se emplea como referencia en @ali2021, @tahir2023, @qian2022, @nguyen2022 y @datta2021, y en su versión de 19 capas en @srinivasu2021 y @shakya2025.

```{mermaid}
flowchart LR
    X["Imagen<br/>3×224×224"] --> B["5 bloques conv 3×3<br/>(2, 2, 3, 3, 3 capas)<br/>64 → 128 → 256 → 512 → 512<br/>+ max-pool por bloque"]
    B --> F["Aplanado<br/>25.088"]
    F --> D["Densa 4.096 → Densa 4.096<br/>ReLU + dropout"]
    D --> O["Densa K = 7<br/>softmax"]
```

Su sesgo inductivo es el convolucional en estado puro, es decir, patrones locales, pesos compartidos en todo el campo visual y composición jerárquica, junto con la hipótesis de que basta con apilar filtros pequeños. Una pila de $n$ capas de 3 × 3 tiene un campo receptivo de $(2n+1) \times (2n+1)$, de modo que dos capas equivalen en alcance a una de 5 × 5 con $18C^2$ parámetros en lugar de $25C^2$ y con una no linealidad adicional. Su fortaleza es la simplicidad: es fácil de implementar y analizar, transfiere bien y sirve como referencia sin mecanismos auxiliares. Sus limitaciones son el riesgo de sobreajuste, por la enorme capa densa frente a unas 7.000 imágenes de entrenamiento; la sensibilidad a la tasa de aprendizaje, por la ausencia de normalización; y la imposibilidad de crecer en profundidad más allá de unas veinte capas.

Su número de parámetros, en función de los canales $C_0 = 3, C_1, \dots, C_{13}$ y del número de clases, es

$$
P_{\text{VGG-16}} = \sum_{l=1}^{13} \left( 9\, C_{l-1} C_l + C_l \right) + (25088 \cdot 4096 + 4096) + (4096^2 + 4096) + (4096K + K),
$$

lo que da 14.714.688 parámetros convolucionales, 119.545.856 en las dos primeras capas densas y 134.289.223 en total para $K = 7$. Casi el 90 % de los parámetros está en las capas densas, pero casi todo el cómputo está en las convolucionales, del orden de $15{,}3 \times 10^{9}$ multiplicaciones y acumulaciones por imagen [@he2016].

## ResNet-50

ResNet-50 [@he2016] (doi: 10.1109/CVPR.2016.90) introduce el aprendizaje residual: cada bloque aprende una corrección $\mathcal{F}$ que se suma a su entrada mediante un atajo,

$$
\mathbf{y} = \sigma\left(\mathcal{F}(\mathbf{x}) + \mathbf{W}_s \mathbf{x}\right), \qquad \mathcal{F}(\mathbf{x}) = \mathrm{BN}\left(\mathbf{W}_3 * \sigma\left(\mathrm{BN}\left(\mathbf{W}_2 * \sigma\left(\mathrm{BN}\left(\mathbf{W}_1 * \mathbf{x}\right)\right)\right)\right)\right),
$$

donde $\mathbf{W}_s$ es la identidad o una proyección de 1 × 1, y $\mathbf{W}_1$, $\mathbf{W}_2$ y $\mathbf{W}_3$ forman un cuello de botella de 1 × 1, 3 × 3 y 1 × 1 que reduce los canales a $m$ y los restaura a $4m$. Cada convolución va seguida de normalización por lotes [@ioffe2015], $\mathrm{BN}(x) = \gamma (x - \mu_{\mathcal{B}}) / \sqrt{\sigma^2_{\mathcal{B}} + \epsilon} + \beta$. Tras una convolución inicial de 7 × 7, cuatro etapas de 3, 4, 6 y 3 bloques con anchos $m$ de 64 a 512 terminan en un promedio global de 2.048 componentes. Es la referencia en @mampitiya2023, @xin2022, @qian2022, @nguyen2022 y @datta2021, y otras variantes de la familia aparecen en cuatro estudios más.

```{mermaid}
flowchart LR
    X["Imagen<br/>3×224×224"] --> S["Conv 7×7, 64<br/>BN + ReLU + max-pool"]
    S --> E["4 etapas residuales<br/>3, 4, 6 y 3 bloques<br/>256 → 512 → 1.024 → 2.048"]
    E --> G["Promedio global<br/>2.048"]
    G --> O["Densa K = 7<br/>softmax"]
    subgraph Bloque["Bloque de cuello de botella"]
        direction LR
        bx["x"] --> c1["1×1, m"] --> c2["3×3, m"] --> c3["1×1, 4m"] --> sum(("+"))
        bx -- "atajo" --> sum
    end
```

ResNet-50 conserva el sesgo convolucional y añade el supuesto de que la transformación óptima de cada bloque está cerca de la identidad. Con atajos identidad, el gradiente que llega a la capa $l$ incluye un término que no se atenúa [@he2016identity],

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \left( 1 + \frac{\partial}{\partial \mathbf{x}_l} \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i) \right),
$$

lo que resuelve el problema de degradación de las redes planas profundas. Su fortaleza es el equilibrio entre capacidad, estabilidad de entrenamiento y costo, además de la abundancia de implementaciones y resultados de referencia. Sus limitaciones son un tamaño todavía grande en relación con el número de lesiones, la dependencia del tamaño del lote que introduce la normalización y la preferencia por la información local y la textura, que la hace vulnerable a atajos visuales.

En la implementación de Keras, cuyas convoluciones incluyen sesgo, un bloque con $C_{\text{in}}$ canales de entrada y ancho $m$ tiene $P_{\text{bloque}} = C_{\text{in}} m + 13m^2 + 18m$ parámetros entrenables, incluidos los parámetros afines de las normalizaciones, y el primero de cada etapa añade $4m\,C_{\text{in}} + 12m$ por la proyección. Sumando la convolución inicial (9.600), los dieciséis bloques y la clasificación ($2048K + K$), el extractor tiene 23.534.592 parámetros entrenables y el modelo de siete clases, 23.548.935, a los que se suman 53.120 estadísticos no entrenables de normalización. Su costo es del orden de $3{,}8 \times 10^{9}$ multiplicaciones y acumulaciones por imagen [@he2016], cuatro veces menos que VGG-16 con tres veces más capas.

## MobileNetV2

MobileNetV2 [@sandler2018] (doi: 10.1109/CVPR.2018.00474) se construye sobre la convolución separable en profundidad, que aplica un filtro espacial por canal y luego combina los canales con una convolución de 1 × 1. Frente a una convolución estándar, su costo relativo es

$$
\frac{H W C_{\text{in}} (k^2 + C_{\text{out}})}{H W k^2 C_{\text{in}} C_{\text{out}}} = \frac{1}{C_{\text{out}}} + \frac{1}{k^2},
$$

entre ocho y nueve veces menor con núcleos de 3 × 3. Estas operaciones se organizan en bloques residuales invertidos que expanden la entrada por un factor $t$ (normalmente 6), aplican la convolución en profundidad con activación $\mathrm{ReLU6}(z) = \min(\max(0, z), 6)$ y proyectan linealmente a la salida, con un atajo entre las representaciones estrechas cuando las dimensiones coinciden. Diecisiete bloques llevan los canales de 16 a 320, una convolución final los lleva a 1.280 y un promedio global con abandono de 0,2 precede a la clasificación. Se usa explícitamente como referencia en @srinivasu2021, @xin2022, @nguyen2022 y @shakya2025, y otras versiones de la familia aparecen en cuatro estudios más.

```{mermaid}
flowchart LR
    X["Imagen<br/>3×224×224"] --> S["Conv 3×3, 32"]
    S --> R["17 bloques residuales invertidos<br/>16 → 24 → 32 → 64 → 96 → 160 → 320"]
    R --> C["Conv 1×1, 1.280<br/>+ promedio global"]
    C --> O["Dropout 0,2<br/>Densa K = 7, softmax"]
    subgraph Bloque["Bloque residual invertido"]
        direction LR
        bx["x, C"] --> e["1×1, tC<br/>ReLU6"] --> d["3×3 en profundidad<br/>ReLU6"] --> p["1×1, C'<br/>lineal"] --> sum(("+"))
        bx -- "atajo" --> sum
    end
```

Además del sesgo convolucional, MobileNetV2 supone que la correlación espacial y la correlación entre canales pueden modelarse por separado, y que la información útil vive en un subespacio de baja dimensión que una no linealidad aplicada en el espacio estrecho destruiría; de ahí la proyección lineal. Su fortaleza es la eficiencia, que facilita el entrenamiento, reduce el riesgo de sobreajuste y la hace apta para el despliegue previsto en la última etapa del proyecto. Sus limitaciones son una menor capacidad de representación, un aprovechamiento de la GPU peor del que sugiere su recuento de operaciones y una mayor sensibilidad a la intensidad de la regularización.

Un bloque con factor $t > 1$ tiene $P_{\text{bloque}} = t\,C_{\text{in}}^2 + t\,C_{\text{in}}C_{\text{out}} + 13t\,C_{\text{in}} + 2C_{\text{out}}$ parámetros, incluidas las normalizaciones, frente a $9\,C_{\text{in}}C_{\text{out}}$ de una única convolución de 3 × 3. El extractor tiene 2.223.872 parámetros entrenables y el modelo de siete clases, 2.232.839, más 34.112 estadísticos no entrenables de normalización; un multiplicador de ancho $\alpha$ los reduce aproximadamente en $\alpha^2$. Su costo es del orden de $3 \times 10^{8}$ multiplicaciones y acumulaciones por imagen [@sandler2018], trece veces menos que ResNet-50 y cincuenta menos que VGG-16.

## Comparación crítica

| | VGG-16 | ResNet-50 | MobileNetV2 |
|---|---|---|---|
| Sesgo distintivo | Jerarquía plana de filtros 3 × 3 | Aprendizaje residual | Convolución separable y residuos invertidos |
| Capas con pesos | 16 | 50 | 53 |
| Parámetros entrenables (K = 7) | 134,3 M | 23,5 M | 2,2 M |
| Cómputo por imagen | ~15,3 × 10⁹ | ~3,8 × 10⁹ | ~0,3 × 10⁹ |
| Valores en la literatura | 0,689 de sensibilidad media [@qian2022] | 0,712 de sensibilidad media [@qian2022]; ≈ 0,92 de exactitud [@xin2022] | 0,83 [@nguyen2022] y ≈ 0,89 de exactitud [@xin2022] |
| Medidas de dispersión | No reportadas | No reportadas | No reportadas |

En desempeño, las cifras solo son comparables dentro de un mismo estudio, y aun así el panorama es inconsistente. ResNet-50 supera a VGG en 2,3 puntos en @qian2022 y a MobileNetV2 en unos tres en @xin2022, pero MobileNetV2 supera a ResNet-50 en siete puntos en @nguyen2022. El único patrón estable es que VGG tiende a quedar última. Sin dispersión, ninguna diferencia puede distinguirse de la variabilidad entre semillas; como referencia, seis arquitecturas evaluadas con partición agrupada por lesión abarcaron 3,6 puntos de exactitud balanceada sin diferencias significativas [@pinar2026].

En robustez, ningún estudio incluido evalúa a los modelos frente a perturbaciones o datos externos. La evidencia indirecta muestra que transformaciones menores alteran las predicciones de las redes en este dominio [@maron2021] y que el desempeño cae más de veinte puntos al pasar de HAM10000 a BCN20000 [@combalia2022; @pinar2026]. Cabe esperar que la normalización por lotes de ResNet-50 y MobileNetV2 estabilice su entrenamiento y que la gran capa densa de VGG-16 favorezca la memorización de artefactos, pero son hipótesis que el protocolo experimental deberá contrastar.

En escalabilidad, los tres modelos difieren en uno y dos órdenes de magnitud de parámetros y cómputo. Duplicar el lado de la imagen cuadruplica el costo de todos, pero VGG-16 es la más exigente en memoria y MobileNetV2 la única pensada para entornos con recursos limitados. ResNet escala en profundidad gracias a sus atajos, MobileNetV2 en ancho y resolución mediante multiplicadores, y VGG no escala más allá de 19 capas.

En cuanto a la validez de los estudios revisados, cinco problemas limitan su interpretación: ninguno agrupa por lesión, aunque las particiones por imagen inflan la exactitud entre uno y tres puntos [@pinar2026]; seis aplican aumento antes de partir y ninguno controla los duplicados señalados por @cassidy2022; ninguno repite el entrenamiento con varias semillas; ninguno aplica pruebas de hipótesis [@demsar2006]; y las tareas, subconjuntos, conjuntos de prueba y métricas difieren tanto que las cifras no son comparables. La métrica es un punto especialmente delicado, porque un clasificador que asignara siempre la clase mayoritaria obtendría un 67 % de exactitud en HAM10000 y solo 0,14 de exactitud balanceada, y la mayoría de los estudios reporta la exactitud global.

## Brecha entre modelos avanzados y modelos base

Cuando un estudio compara su propuesta con modelos base en condiciones homogéneas, la mejora suele ser modesta. En @qian2022 la propuesta alcanza 0,735 de sensibilidad media frente a 0,712 de ResNet-50 y 0,689 de VGG; en @xin2022 el transformador propuesto supera a ResNet-50 en unos dos puntos y a MobileNetV2 en unos cinco; en @datta2021 la atención mejora la precisión de los modelos base entre 0,5 y 3,2 puntos; y en @tahir2023 la propuesta supera a MobileNet en 1,7 puntos. Las brechas típicas están entre uno y cinco puntos, medidas sin dispersión, con particiones por imagen y con esfuerzos de ajuste posiblemente desiguales. Las brechas mayores, como el 0,885 de exactitud balanceada de los ganadores del reto ISIC 2018 [@codella2019], se deben a conjuntos de modelos, datos externos y metadatos, y no pueden atribuirse a la arquitectura; en sentido contrario, @pinar2026 no encontraron diferencias significativas entre ResNet-50 y arquitecturas mucho más recientes.

De ahí se desprenden dos expectativas para este proyecto. Con modelos preentrenados, partición agrupada por lesión y exactitud balanceada, el baseline debería situarse aproximadamente entre 0,66 y 0,75 [@qian2022; @nozdryn2018; @pinar2026], lejos del 90 % que predomina en la literatura. Y las mejoras atribuidas a los modelos avanzados son del mismo orden que la variabilidad entre semillas y que el sesgo de las particiones por imagen, por lo que un margen de relevancia de unos tres puntos parece razonable, pero su detectabilidad deberá comprobarse con el estudio piloto y el análisis de potencia, corrigiendo la varianza por la correlación entre pliegues [@nadeau2003].

## Brecha en la literatura

Ninguno de los quince estudios incluidos compara VGG-16, ResNet-50 y MobileNetV2, ni ningún par de ellas, sobre HAM10000 bajo una partición agrupada por lesión, con repeticiones independientes del entrenamiento y con pruebas estadísticas que permitan distinguir sus diferencias de la variabilidad experimental. @pinar2026 cumplen esas condiciones para ResNet-50 frente a arquitecturas modernas, pero no incluyen a VGG-16 ni a MobileNetV2. Además, ningún estudio declara haber asignado el mismo presupuesto de ajuste de hiperparámetros a los modelos comparados. Este proyecto aborda ambas carencias entrenando los tres modelos con el mismo pipeline, la misma partición, las mismas semillas y el mismo esfuerzo de ajuste, y comparándolos con pruebas de hipótesis, tamaños de efecto e intervalos de confianza.
