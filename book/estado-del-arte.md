---
title: Estado del arte
---

El propósito de la revisión de literatura es identificar los tres modelos de aprendizaje profundo más básicos que los estudios sobre clasificación de lesiones dermatoscópicas utilizan como referencia comparativa. Esta página documenta la estrategia de búsqueda, la evaluación de los estudios incluidos y el procedimiento que condujo a la selección de los modelos. El análisis detallado de cada modelo se presenta en la página [Modelos base](modelos-base.md).

## Estrategia de búsqueda

La búsqueda se organizó en dos niveles. Seis fases narrativas, sin restricción temporal, reconstruyeron la evolución del problema y sustentan los antecedentes del proyecto. Una séptima fase, de carácter sistemático, se centró en estudios de aprendizaje profundo sobre HAM10000 publicados entre 2018 y 2026; solo esta fase se sometió a criterios de inclusión, evaluación de calidad y diagrama PRISMA [@page2021]. El límite inferior coincide con la publicación del conjunto de datos [@tschandl2018].

Todas las cadenas se ejecutaron en Scopus el 6 de noviembre de 2026 sobre título, resumen y palabras clave. Scopus indexa las principales revistas del área recogidas en Web of Science y buena parte de las publicaciones de IEEE relevantes para el problema, como IEEE Transactions on Medical Imaging, IEEE Access, ISBI o los talleres de CVPR, pero Web of Science e IEEE Xplore no se consultaron de forma directa, lo que se reconoce como una limitación. arXiv y PubMed Central se usaron para recuperar documentos no indexados, como los informes de los retos ISIC, y la búsqueda en cadena a partir de los trabajos más citados permitió identificar estudios adicionales, entre ellos el de @pinar2026. La cadena de la fase sistemática fue la siguiente:

```text
TITLE-ABS-KEY ( ( "deep learning" OR "convolutional neural network*" OR CNN
  OR ResNet OR VGG OR DenseNet OR "transfer learning" )
AND ( HAM10000 OR "HAM 10000" OR "ISIC 2018" OR ISIC2018 )
AND ( classification OR diagnosis ) )
AND PUBYEAR > 2017 AND PUBYEAR < 2027
AND ( DOCTYPE ( ar ) OR DOCTYPE ( cp ) )
```

:::{dropdown} Cadenas de las fases narrativas
Fase 1, fundamentos clínicos (200 registros exportados):

```text
TITLE-ABS-KEY(dermoscop* OR dermatoscop* OR epiluminescence)
AND TITLE-ABS-KEY("ABCD rule" OR "seven-point checklist" OR "7-point checklist"
  OR "pattern analysis" OR menzies)
AND TITLE-ABS-KEY(melanoma)
```

Fase 2, diagnóstico asistido clásico, hasta 2015 (271 registros):

```text
TITLE-ABS-KEY ( ( dermoscop* OR "skin lesion*" OR melanoma )
AND ( "computer-aided diagnosis" OR "computer aided" OR "automated diagnosis" OR "image analysis" )
AND ( "feature extraction" OR handcrafted OR texture OR "color feature*"
  OR "support vector machine" OR SVM OR "k-nearest" ) )
AND PUBYEAR < 2016
```

Fase 3, llegada del aprendizaje profundo, 2015–2018 (166 registros):

```text
TITLE-ABS-KEY ( ( "deep learning" OR "convolutional neural network*" OR "transfer learning" )
AND ( dermoscop* OR "skin lesion*" OR melanoma OR "skin cancer" )
AND ( classification OR diagnosis ) )
AND PUBYEAR > 2014 AND PUBYEAR < 2019
```

Fase 4, humanos frente a algoritmos, desde 2018 (200 registros exportados):

```text
TITLE-ABS-KEY ( ( "deep learning" OR "artificial intelligence" OR "convolutional neural network*" )
AND ( dermatologist* OR "reader study" OR "human against machine" OR "human-computer collaboration" )
AND ( melanoma OR "skin cancer" OR dermoscop* ) )
AND PUBYEAR > 2017
```

Fase 5, validez y sesgos, desde 2019 (200 registros exportados):

```text
TITLE-ABS-KEY ( ( "deep learning" OR "convolutional neural network*" OR "artificial intelligence" )
AND ( dermoscop* OR "skin lesion*" OR melanoma )
AND ( bias OR shortcut OR "spurious correlation*" OR "data leakage" OR "external validation"
  OR generaliz* OR "domain shift" OR "skin tone" OR reproducib* ) )
AND PUBYEAR > 2018
```

Fase 6, tendencias recientes, desde 2021 (200 registros exportados):

```text
TITLE-ABS-KEY ( ( "vision transformer" OR ViT OR ConvNeXt OR "foundation model"
  OR "self-supervised" OR "multimodal" )
AND ( dermoscop* OR "skin lesion*" OR HAM10000 ) )
AND PUBYEAR > 2020
```
:::

Cuando una búsqueda devolvió más registros de los que Scopus permite exportar en una sola operación, se exportaron los 200 más citados. La cadena de la fase sistemática devolvió [COMPLETAR] registros; restringir el cribado a los 200 más citados introduce un posible sesgo hacia publicaciones de mayor visibilidad. Todas las exportaciones se conservan en `data/literatura/`.

## Criterios y proceso de selección

Se incluyeron los estudios que abordaban la clasificación de lesiones cutáneas en imágenes dermatoscópicas usando HAM10000 o el conjunto del reto ISIC 2018, total o parcialmente; que empleaban aprendizaje profundo; que reportaban métricas verificables; y que comparaban su propuesta con al menos un modelo de aprendizaje profundo de referencia descrito con suficiente detalle. Se excluyeron los trabajos de segmentación o de detección de atributos, los de otros dominios, las revisiones y comentarios, los métodos genéricos de imagen médica y los estudios sin descripción suficiente de la arquitectura de referencia o del protocolo. Dos estudios que abordaban variantes de la tarea, uno con cuatro clases [@tahir2023] y otro binario [@shakya2025], se incluyeron en el recuento de modelos, pero se señalan como no comparables en desempeño.

La selección fue realizada por un único revisor, el autor del proyecto. El cribado por título se apoyó en reglas de palabras clave implementadas con la asistencia de una herramienta de inteligencia artificial generativa, y todas las decisiones se revisaron manualmente. Al no haber un segundo revisor no hubo desacuerdos que resolver, pero tampoco fue posible estimar la concordancia, lo que constituye una fuente reconocida de sesgo. Entre los 146 registros elegibles por título se priorizaron para lectura completa los 24 que indicaban una comparación explícita entre varias arquitecturas estándar, por ser los más informativos para el recuento de modelos de referencia; los 122 restantes quedan documentados en `data/literatura/Fase7_cribado.csv`.

```{mermaid}
flowchart TD
    A["Registros identificados en Scopus<br/>n = [COMPLETAR]"] --> B["Registros exportados y cribados<br/>(200 más citados)<br/>n = 200"]
    B --> C["Excluidos por título: n = 54<br/>Segmentación: 37 · Imagen médica genérica: 6<br/>Otro dominio: 6 · Revisiones: 4 · Atributos: 1"]
    B --> D["Potencialmente elegibles<br/>n = 146"]
    D --> E["No priorizados<br/>n = 122"]
    D --> F["Textos completos buscados<br/>n = 24"]
    F --> G["No recuperados<br/>n = 9"]
    F --> H["Evaluados e incluidos<br/>n = 15"]
```

## Calidad de los estudios incluidos

Cada estudio se valoró en cuatro dimensiones: claridad del protocolo, ausencia aparente de fuga de datos, número de repeticiones y uso de pruebas estadísticas. En la dimensión de fuga se consideraron dos mecanismos. El más grave es aplicar aumento de datos o sobremuestreo antes de dividir los datos, de modo que versiones de una misma imagen terminan en entrenamiento y en prueba (calificación «No»). El segundo es la partición aleatoria por imagen, que ignora que las 10.015 imágenes de HAM10000 corresponden a 7.470 lesiones (calificación «Parcial»).

| Estudio | Protocolo | Ausencia de fuga | Repeticiones | Pruebas estadísticas |
|---|---|---|---|---|
| @srinivasu2021 | Parcial | No | No | No |
| @ali2021 | Parcial | Parcial | No | No |
| @shetty2022 | Parcial | No | Parcial | No |
| @tahir2023 | Parcial | No | No | No |
| @alam2022 | Parcial | No | No | No |
| @sun2021 | Sí | Sí | No | No |
| @mampitiya2023 | Parcial | Parcial | Parcial | No |
| @baig2023 | Sí | No | Parcial | No |
| @villapulgarin2022 | Sí | No | No | Parcial |
| @xin2022 | Parcial | Parcial | No | No |
| @qian2022 | Sí | Parcial | No | No |
| @mehmood2023 | Sí | Parcial | No | No |
| @nguyen2022 | Sí | Parcial | No | No |
| @datta2021 | Sí | Parcial | No | No |
| @shakya2025 | Parcial | Parcial | No | No |

Solo @sun2021 evita ambos mecanismos de fuga, al evaluar sobre el conjunto de prueba oficial de ISIC. Seis estudios aumentan o sobremuestrean antes de partir, y el efecto puede ser enorme: en @villapulgarin2022 las mismas arquitecturas pasan de 81 % de exactitud a valores entre 95 % y 98 % cuando el balanceo se hace antes de dividir. Ningún estudio agrupa por lesión, ninguno repite el entrenamiento con varias semillas y ninguno aplica pruebas de hipótesis para comparar modelos. Estas deficiencias no impiden usar los estudios para saber qué modelos se emplean como referencia, que es el objetivo de la revisión, pero sí limitan la interpretación de los valores que reportan.

## Identificación de los modelos base

### Extracción y frecuencias

En cada estudio se registraron los modelos usados como referencia, distinguiéndolos del propuesto, junto con su configuración. La práctica es muy homogénea: catorce de los quince estudios parten de pesos preentrenados en ImageNet [@deng2009], sustituyen la capa de clasificación, usan entradas de 224 × 224 píxeles y entrenan con Adam o descenso por gradiente estocástico, tasas de aprendizaje del orden de 10⁻³ y entropía cruzada, ponderada en algunos casos. La regularización se basa en aumento de datos y abandono (*dropout*) en la cabeza de clasificación. Ningún estudio declara haber dedicado el mismo esfuerzo de ajuste de hiperparámetros a todos los modelos comparados. @sun2021 se compara solo con los ganadores del reto ISIC y no aporta modelos al recuento, y @srinivasu2021 toma los valores de referencia de otros trabajos en lugar de reentrenarlos.

| Familia | Estudios | Variantes (número de estudios) | Parámetros de la variante más usada |
|---|---|---|---|
| ResNet | 9 | ResNet-50 (5), ResNet-18 (2), ResNet-152 (2), ResNet-34 (1), sin especificar (1) | 25,6 M |
| Inception | 9 | Inception-V3 (6), Inception-ResNet-V2 (3), Inception-V2 (1) | 23,9 M |
| MobileNet | 8 | MobileNetV2 (4), V1 o sin especificar (5), V3 (1) | 3,5 M |
| VGG | 7 | VGG-16 (4), VGG-19 (3), sin especificar (1) | 138,4 M |
| DenseNet | 5 | DenseNet-201 (3), sin especificar (2) | 20,0 M |
| AlexNet | 3 | AlexNet (3) | 61,1 M |
| Xception | 2 | Xception (2) | 22,9 M |
| EfficientNet | 1 | EfficientNet-B0 (1) | 5,3 M |
| NASNet | 1 | Large y Mobile (1) | 88,9 M y 5,3 M |

### Selección

La selección se decide entre las cuatro familias más frecuentes. ResNet encabeza el recuento y ResNet-50 es su variante más usada; aunque ResNet-18 es menos compleja, solo dos estudios la emplean, y ResNet-50 es además la referencia en los trabajos con partición agrupada por lesión [@nozdryn2018; @pinar2026]. MobileNet es la segunda familia por frecuencia y la de menor complejidad entre las principales, con 3,5 millones de parámetros en MobileNetV2, la versión identificada explícitamente con más frecuencia.

La tercera plaza se disputa entre Inception y VGG. Inception suma nueve estudios frente a siete, pero tres de ellos usan Inception-ResNet-V2, un híbrido con conexiones residuales que comparte el sesgo inductivo de ResNet; con las variantes puras, el recuento queda empatado en siete. El empate se resuelve por complejidad y no redundancia. VGG-16 es estructuralmente la más simple de las candidatas, una pila homogénea de convoluciones de 3 × 3 sin ramificaciones ni atajos, mientras que Inception-V3 organiza más de cuarenta capas en módulos multirrama [@szegedy2016]; el gran número de parámetros de VGG-16 se concentra casi en un 90 % en sus capas densas, por lo que su complejidad es de tamaño y no de diseño. Además, la factorización de las convoluciones que define a Inception es la misma idea que la convolución separable en profundidad lleva al extremo [@chollet2017], y esa convolución es la base de MobileNetV2, de modo que elegir Inception duplicaría un sesgo inductivo. Inception-V3 queda como candidata para los modelos avanzados de la siguiente etapa.

Los modelos base son, por tanto, VGG-16 [@simonyan2015], ResNet-50 [@he2016] y MobileNetV2 [@sandler2018]. Difieren en su sesgo inductivo: una jerarquía plana de filtros pequeños, el aprendizaje residual con normalización por lotes y la factorización de la convolución con cuellos de botella lineales. Los tres son simples, están ampliamente documentados, pueden entrenarse bajo el mismo pipeline y son de aprendizaje profundo. Como la literatura los usa casi siempre preentrenados, y el ajuste fino desde ImageNet ha mostrado ser la estrategia más eficaz en este dominio [@menegola2017; @brinker2018], se implementarán con esa inicialización.

### Tabla de síntesis

Los valores se transcriben tal como aparecen en cada artículo y no son comparables entre estudios.

| Referencia | Problema | Datos | Modelos básicos de referencia | Métricas | Valores de los modelos de referencia |
|---|---|---|---|---|---|
| @srinivasu2021 | 7 clases | HAM10000 | VGG-19, MobileNetV1, MobileNetV2, CNN (citados) | Exactitud, precisión, sensibilidad | VGG-19 82,5 %; MobileNetV2 86,4 % |
| @ali2021 | 7 clases | HAM10000 | AlexNet, ResNet, VGG-16, DenseNet, MobileNet | Precisión, sensibilidad, F1 | F1: VGG-16 91,6 %; ResNet 90,8 %; MobileNet 89,3 % |
| @shetty2022 | 7 clases | Subconjunto de HAM10000 | Inception-V3 | Exactitud, F1 | Inception-V3 95,1 % |
| @tahir2023 | 4 clases | ISIC 2020, Derm-IS, HAM10000 | VGG-16, VGG-19, MobileNet, ResNet-152, EfficientNet-B0, Inception-V3 | Exactitud, F1, AUC | MobileNet 92,5 %; VGG-16 91,1 %; ResNet-152 89,3 % |
| @alam2022 | 7 clases | HAM10000 | AlexNet, Inception-V3 | Exactitud, F1, AUC | Inception-V3 77 %; AlexNet 76 % |
| @sun2021 | 7 clases | ISIC 2018 oficial | Ninguno estándar | Exactitud balanceada | No aplica |
| @mampitiya2023 | 7 clases | HAM10000 | ResNet-50, MobileNet | Exactitud, F1 | ResNet-50 83 %; MobileNet 72 % |
| @baig2023 | 7 y 8 clases | HAM10000, ISIC 2019 | AlexNet, MobileNet, ResNet-18, Inception-V3, Xception | Exactitud, F1 | MobileNet 90,8 %; ResNet-18 89,3 % |
| @villapulgarin2022 | 7 y 8 clases | HAM10000, ISIC 2019 | DenseNet-201, Inception-ResNet-V2, Inception-V3 | Exactitud, F1 | 81 % sin balanceo; 95–98 % con balanceo previo |
| @xin2022 | 7 clases | HAM10000 y propio | MobileNetV2, ResNet-50, Inception-V2 | Precisión, exactitud | ResNet-50 ≈ 0,92; MobileNetV2 ≈ 0,89 |
| @qian2022 | 7 clases | HAM10000 | VGG, ResNet-50, DenseNet, Inception-V3 | Sensibilidad media, AUC | ResNet-50 0,712; Inception-V3 0,701; VGG 0,689 |
| @mehmood2023 | 7 clases | HAM10000 | Xception | Exactitud, F1, AUC | Xception 94,3 % |
| @nguyen2022 | 7 clases | HAM10000 | Inception-ResNet-V2, DenseNet-201, ResNet-50, MobileNetV2, VGG-16, otros | Exactitud, exactitud balanceada | DenseNet-201 0,84; MobileNetV2 0,83; ResNet-50 0,76 |
| @datta2021 | 7 clases | HAM10000, ISIC 2017 | VGG-16, ResNet-34, ResNet-50, DenseNet-201, Inception-ResNet-V2 | Precisión, AUC | Mejoras del modelo propuesto de 0,5 a 3,2 puntos |
| @shakya2025 | 2 clases | Subconjunto de ISIC 2018 | VGG-19, ResNet-18, MobileNetV2 | Exactitud, F1 | MobileNetV2 92,3 %; ResNet-18 90,3 %; VGG-19 84,6 % |

### Equivalencia con la regresión logística

Los tres modelos base cumplen en este problema el papel que la regresión logística cumple en clasificación clásica, y la equivalencia es también formal. Cada red puede escribirse como un extractor de características $\phi_{\theta}(\mathbf{x}) \in \mathbb{R}^{d}$ seguido de una capa lineal con softmax,

$$
p(y = k \mid \mathbf{x}) = \frac{\exp\left(\mathbf{w}_k^{\top} \phi_{\theta}(\mathbf{x}) + b_k\right)}{\sum_{j=1}^{K} \exp\left(\mathbf{w}_j^{\top} \phi_{\theta}(\mathbf{x}) + b_j\right)},
$$

que es una regresión logística multinomial sobre $\phi_{\theta}(\mathbf{x})$, con $d$ igual a 4.096, 2.048 y 1.280 en VGG-16, ResNet-50 y MobileNetV2, entrenada con la misma log-verosimilitud negativa. La diferencia está en que la regresión logística usa una representación fija, diseñada a mano, mientras que las redes la aprenden junto con el clasificador; si se congelara el extractor preentrenado, cada modelo sería literalmente una regresión logística sobre características de ImageNet. En capacidad, las redes son no lineales en el espacio de la imagen y lineales solo en el espacio aprendido. En supuestos, trasladan la linealidad del logaritmo de las razones de probabilidades a ese espacio y añaden la localidad, la equivariancia a traslaciones y la composición jerárquica propias de las convoluciones. En propósito, ambas son el modelo más sencillo que se aceptaría como competidor serio. La regresión logística y otros modelos clásicos se usan aquí solo como contexto: un clasificador de histogramas de color con XGBoost alcanzó 0,511 de exactitud balanceada en HAM10000 [@nozdryn2018], una cota que cualquier modelo base debería superar con holgura.
