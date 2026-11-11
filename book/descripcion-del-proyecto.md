---
title: Descripción del proyecto
---

## Planteamiento del problema

El cáncer de piel es una de las neoplasias más frecuentes en poblaciones de piel clara, y su pronóstico depende críticamente de un diagnóstico temprano. La dermatoscopia mejora la precisión diagnóstica frente a la inspección a simple vista [@dinnes2018], pero su rendimiento depende de la experiencia del observador, y la concordancia entre expertos en la lectura de los criterios dermatoscópicos es apenas moderada [@carrera2016]. Esta situación ha motivado el desarrollo de sistemas automáticos de clasificación de imágenes, que en la última década han pasado a basarse casi exclusivamente en redes neuronales profundas.

La publicación de HAM10000 [@tschandl2018] y su uso en el reto ISIC 2018 [@codella2019] proporcionaron a la comunidad un conjunto público de referencia con siete categorías diagnósticas. Desde entonces, cientos de trabajos reportan exactitudes superiores al 90 % sobre este conjunto. Esas cifras, sin embargo, contrastan con las obtenidas bajo protocolos más estrictos, que se sitúan alrededor de 0,70 de exactitud balanceada [@pinar2026], y con la caída de más de veinte puntos que se observa al evaluar en datos de otra institución [@combalia2022]. La diferencia se explica en buena parte por decisiones metodológicas: particiones que reparten imágenes de una misma lesión entre entrenamiento y prueba, aumento de datos aplicado antes de dividir los datos, métricas sensibles al desbalance de clases y comparaciones basadas en una sola ejecución.

En ese contexto, la pregunta de cuánto mejora un modelo avanzado respecto de uno sencillo no tiene hoy una respuesta fiable. El problema que aborda esta etapa del proyecto es, por tanto, previo al desarrollo de cualquier modelo nuevo: determinar con rigor el desempeño de los modelos base que la literatura emplea como referencia en la clasificación de lesiones dermatoscópicas.

## Antecedentes

La formalización del diagnóstico dermatoscópico comenzó con el análisis de patrones de @pehamberger1987 y continuó con algoritmos semicuantitativos más fáciles de aplicar, como la regla ABCD [@stolz1994; @nachbar1994] y la lista de siete puntos [@argenziano1998]. La regla ABCD es, en esencia, una suma ponderada de atributos visuales comparada con un umbral, es decir, un clasificador lineal sobre características definidas a mano. El consenso internacional de @argenziano2003 estandarizó la terminología, pero también dejó en evidencia la variabilidad entre observadores, una fuente de ruido que heredan las etiquetas de cualquier conjunto de datos construido a partir del juicio clínico.

Los primeros sistemas automáticos aparecieron en los años noventa con redes neuronales entrenadas sobre características extraídas de la imagen [@binder1998]. @dreiseitl2001 compararon varios clasificadores para lesiones pigmentadas y encontraron que la regresión logística rendía al nivel de las redes neuronales y de las máquinas de vectores de soporte, lo que la consolidó como referencia natural. Durante la década siguiente predominó la tubería de segmentación, extracción de características de color, forma y textura y clasificación [@celebi2007; @barata2014], hasta que @codella2015 mostraron la utilidad de las representaciones aprendidas por redes convolucionales preentrenadas.

El punto de inflexión fue el trabajo de @esteva2017, que alcanzó un desempeño comparable al de dermatólogos ajustando una red Inception preentrenada en ImageNet. En el dominio dermatoscópico, @yu2017 emplearon redes residuales profundas y @menegola2017 mostraron la ventaja del aprendizaje por transferencia desde ImageNet, la estrategia que @brinker2018 identificaron como la más común y eficaz. Los estudios de lectores confirmaron que las redes podían igualar o superar a los especialistas en condiciones controladas [@haenssle2018; @maron2019], incluso en la tarea de siete clases de HAM10000 [@tschandl2019].

A partir de 2019, la literatura empezó a cuestionar la validez de esos resultados. Las redes aprenden atajos como las marcas de tinta [@winkler2019] o los artefactos del fondo, hasta el punto de clasificar por encima del azar imágenes en las que la lesión ha sido ocultada [@bissoto2019]. Los conjuntos ISIC contienen duplicados que casi ningún estudio elimina [@cassidy2022], y el desempeño cae drásticamente fuera de la institución de origen [@combalia2022]. Con una partición agrupada por lesión, @pinar2026 no encontraron diferencias significativas entre ResNet-50 y arquitecturas recientes como ConvNeXt, Swin o ViT. Mientras tanto, la frontera técnica avanza hacia transformadores de visión [@xin2022], la fusión con metadatos clínicos [@vachmanus2023] y los modelos fundacionales [@yan2025], con un rigor comparativo que no ha crecido al mismo ritmo.

## Justificación

Cualquier afirmación sobre la superioridad de un modelo exige conocer antes, con su incertidumbre, el desempeño del punto de comparación. En aprendizaje automático clásico ese papel lo cumple la regresión logística; en aprendizaje profundo aplicado a dermatoscopia lo cumplen unas pocas arquitecturas estándar, pero sus valores de referencia se han medido casi siempre una sola vez, bajo particiones con fuga y con métricas poco apropiadas para el desbalance de clases. Establecer un baseline riguroso tiene, por tanto, un valor en sí mismo: permite saber qué magnitud de mejora es plausible, cuál es indistinguible de la variabilidad experimental y cuál constituye una contribución real. Además, como el conjunto de datos y el protocolo se mantendrán en todas las etapas del proyecto, la calidad de este punto de partida condiciona la validez de todo lo que se construya después.

## Objetivos

### Objetivo general

Establecer, con evidencia estadística, el desempeño de referencia de los modelos base de aprendizaje profundo empleados en la literatura para la clasificación de las siete categorías diagnósticas de HAM10000, bajo un protocolo que controle la dependencia entre imágenes de una misma lesión.

### Objetivos específicos

1. Identificar, mediante una revisión sistemática de la literatura, los tres modelos de aprendizaje profundo más básicos que los estudios sobre HAM10000 utilizan como referencia comparativa.
2. Caracterizar el conjunto de datos, su estructura de dependencia, su desbalance y sus posibles atajos visuales, y definir a partir de ello una partición estratificada y agrupada por lesión.
3. Implementar y entrenar los tres modelos base bajo el mismo pipeline, la misma partición y el mismo presupuesto de ajuste de hiperparámetros.
4. Comparar los modelos mediante pruebas estadísticas apropiadas y definir el baseline y el umbral mínimo de mejora que deberán superar los modelos avanzados.
