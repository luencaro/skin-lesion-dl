---
title: Paper guía
---

## Selección

El paper guía del proyecto es @pinar2026, que propone un marco reproducible y explicable para la clasificación dermatoscópica en HAM10000. Se elige por tres razones. Es el único estudio de la revisión que agrupa la partición por lesión, el mismo control de fuga que usa este proyecto, y documenta que las particiones por imagen inflan la exactitud entre uno y tres puntos. Compara ResNet-50, uno de los modelos base, con ConvNeXt, Swin y ViT, dos de los cuales forman parte del benchmark, de modo que su resultado se sitúa dentro de este. Y evalúa la transferencia a un conjunto externo, BCN20000, donde la exactitud balanceada cae 25 puntos en promedio, lo que lo conecta con el riesgo de dependencia del dominio de adquisición que identificó el análisis exploratorio.

## Descripción

@pinar2026 comparan seis configuraciones preentrenadas en ImageNet-1k, ResNet-50, DenseNet-121, EfficientNet-B3, ConvNeXt-T, Swin-T y ViT-B/16, sobre HAM10000 con una validación cruzada externa de cinco pliegues agrupada por lesión (`StratifiedGroupKFold`) y una partición interna de validación del 10 % por lesión para la parada temprana y la calibración. Todas las configuraciones comparten la resolución de 224 × 224, el preentrenamiento, el optimizador AdamW y una pérdida focal balanceada por clase, y se entrenan con una semilla por pliegue. La exactitud balanceada es la métrica primaria y la unidad de análisis es el pliegue externo ($N = 5$).

El resultado central es que las seis configuraciones son estadísticamente indistinguibles: la prueba de Friedman da $\chi^2 = 7{,}057$ con $p = 0{,}216$ y la exactitud balanceada varía solo entre 0,664 (ViT-B/16) y 0,700 (ConvNeXt-T), frente a 0,143 del clasificador trivial. En la validación externa sobre BCN20000, la exactitud balanceada cae en promedio 25 puntos, a valores de 0,401 a 0,476, y el orden de los modelos se invierte. El supuesto implícito del diseño interno, que la evaluación dentro de HAM10000 con partición por lesión estima una propiedad estable de cada arquitectura, queda así cuestionado por los propios autores.

Dentro del benchmark, el paper guía ocupa la posición de referencia metodológica: ResNet-50 es uno de los modelos base y ConvNeXt-Tiny y ViT-B/16 forman parte de los cinco modelos del estado del arte, de modo que su comparación central se reproduce dentro del propio diseño.

## Replicación

Se replica el resultado de ResNet-50 con partición agrupada por lesión: una exactitud balanceada de 0,686, con un intervalo de confianza del 95 % de 0,653 a 0,721. La réplica usa el mismo conjunto, el mismo tipo de partición, la misma resolución, el mismo preentrenamiento y la misma métrica, pero difiere en tres aspectos que se declaran de antemano: la asignación concreta de lesiones a pliegues, la pérdida (entropía cruzada ponderada en lugar de pérdida focal) y el esquema de entrenamiento en dos fases con su propio presupuesto de ajuste.

El criterio de replicación exitosa se fija antes de ejecutar: la réplica se considera exitosa si el valor reportado por los autores, 0,686, cae dentro del intervalo de confianza del 95 % de la exactitud balanceada media de ResNet-50 en las corridas definitivas, o si la diferencia absoluta entre ambos es menor que $\delta = 0{,}02$. Si la réplica falla, la discrepancia se reporta con su intervalo y se analizan sus causas probables: la partición concreta, la pérdida, el preprocesamiento, el aumento de datos, el presupuesto de ajuste y las versiones de las librerías.

## Limitación identificada

La limitación que se estudia es la dependencia del desempeño respecto de la fuente de adquisición. @pinar2026 constatan la caída en un conjunto externo y la inversión del orden de los modelos, pero no analizan si, dentro de HAM10000, el desempeño ya difiere entre las cuatro fuentes, cuya influencia sobre el color, el viñeteado y la composición por clase documentó el análisis exploratorio. Si esa diferencia existe, la evaluación con partición agrupada por lesión sobreestima la generalización, porque train y prueba comparten las mismas fuentes.

La limitación se cuantifica como la diferencia de exactitud balanceada en prueba entre las imágenes de vidir_molemax y las del resto de fuentes, con su intervalo de confianza del 95 % por bootstrap sobre lesiones. Se verifica en al menos tres modelos, los tres modelos base del Entregable 1, para determinar si es específica de una arquitectura o compartida por la familia, y queda declarada en el preregistro antes de los experimentos definitivos. Es la candidata natural al hueco de investigación del modelo original del Entregable 3, en línea con los mecanismos de adaptación de dominio identificados en la búsqueda transversal [@tomar2021].
