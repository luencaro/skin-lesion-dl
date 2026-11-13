---
title: Descripción del proyecto
---

## Planteamiento del problema

El cáncer de piel es una de las neoplasias más frecuentes en poblaciones de piel clara, y su pronóstico depende críticamente de un diagnóstico temprano. La dermatoscopia mejora la precisión diagnóstica frente a la inspección a simple vista [@dinnes2018], pero su rendimiento depende de la experiencia del observador, y la concordancia entre expertos en la lectura de los criterios dermatoscópicos es apenas moderada [@carrera2016]. Esta situación ha motivado el desarrollo de sistemas automáticos de clasificación de imágenes, que en la última década han pasado a basarse casi exclusivamente en redes neuronales profundas.

La publicación de HAM10000 [@tschandl2018] y su uso en el reto ISIC 2018 [@codella2019] proporcionaron a la comunidad un conjunto público de referencia con siete categorías diagnósticas. Desde entonces, cientos de trabajos reportan exactitudes superiores al 90 % sobre este conjunto. Esas cifras, sin embargo, contrastan con las obtenidas bajo protocolos más estrictos, que se sitúan alrededor de 0,70 de exactitud balanceada [@pinar2026], y con la caída de más de veinte puntos que se observa al evaluar en datos de otra institución [@combalia2022]. La diferencia se explica en buena parte por decisiones metodológicas: particiones que reparten imágenes de una misma lesión entre entrenamiento y prueba, aumento de datos aplicado antes de dividir los datos, métricas sensibles al desbalance de clases y comparaciones basadas en una sola ejecución.

En ese contexto, la pregunta de cuánto mejora una arquitectura nueva respecto de los modelos con los que se compara no tiene hoy una respuesta fiable. Proponer una arquitectura que supere a los modelos de referencia exige, en primer lugar, identificar cuáles son esos modelos y conocer su desempeño bajo un protocolo que controle la dependencia entre imágenes de una misma lesión.

## Antecedentes

La formalización del diagnóstico dermatoscópico comenzó con el análisis de patrones de @pehamberger1987 y continuó con algoritmos semicuantitativos más fáciles de aplicar, como la regla ABCD [@stolz1994; @nachbar1994] y la lista de siete puntos [@argenziano1998]. La regla ABCD es, en esencia, una suma ponderada de atributos visuales comparada con un umbral, es decir, un clasificador lineal sobre características definidas a mano. El consenso internacional de @argenziano2003 estandarizó la terminología, pero también dejó en evidencia la variabilidad entre observadores, una fuente de ruido que heredan las etiquetas de cualquier conjunto de datos construido a partir del juicio clínico.

Los primeros sistemas automáticos aparecieron en los años noventa con redes neuronales entrenadas sobre características extraídas de la imagen [@binder1998]. @dreiseitl2001 compararon varios clasificadores para lesiones pigmentadas y encontraron que la regresión logística rendía al nivel de las redes neuronales y de las máquinas de vectores de soporte, lo que la consolidó como referencia natural. Durante la década siguiente predominó la tubería de segmentación, extracción de características de color, forma y textura y clasificación [@celebi2007; @barata2014], hasta que @codella2015 mostraron la utilidad de las representaciones aprendidas por redes convolucionales preentrenadas.

El punto de inflexión fue el trabajo de @esteva2017, que alcanzó un desempeño comparable al de dermatólogos ajustando una red Inception preentrenada en ImageNet. En el dominio dermatoscópico, @yu2017 emplearon redes residuales profundas y @menegola2017 mostraron la ventaja del aprendizaje por transferencia desde ImageNet, la estrategia que @brinker2018 identificaron como la más común y eficaz. Los estudios de lectores confirmaron que las redes podían igualar o superar a los especialistas en condiciones controladas [@haenssle2018; @maron2019], incluso en la tarea de siete clases de HAM10000 [@tschandl2019].

A partir de 2019, la literatura empezó a cuestionar la validez de esos resultados. Las redes aprenden atajos como las marcas de tinta [@winkler2019] o los artefactos del fondo, hasta el punto de clasificar por encima del azar imágenes en las que la lesión ha sido ocultada [@bissoto2019]. Los conjuntos ISIC contienen duplicados que casi ningún estudio elimina [@cassidy2022], y el desempeño cae drásticamente fuera de la institución de origen [@combalia2022]. Con una partición agrupada por lesión, @pinar2026 no encontraron diferencias significativas entre ResNet-50 y arquitecturas recientes como ConvNeXt, Swin o ViT. Mientras tanto, la frontera técnica avanza hacia transformadores de visión [@xin2022], la fusión con metadatos clínicos [@vachmanus2023] y los modelos fundacionales [@yan2025], con un rigor comparativo que no ha crecido al mismo ritmo.

## Justificación

La literatura sobre HAM10000 propone continuamente arquitecturas nuevas, pero las mejoras que reporta se miden casi siempre contra modelos de referencia evaluados una sola vez, bajo particiones con fuga y con métricas poco apropiadas para el desbalance de clases. En esas condiciones, una mejora de pocos puntos puede deberse a la variabilidad del entrenamiento o a diferencias de protocolo y no a la arquitectura. Una propuesta que pretenda superar a los modelos de referencia necesita, por tanto, compararse con ellos bajo el mismo conjunto de datos, la misma partición y el mismo procedimiento de evaluación, y demostrar la mejora con evidencia estadística. Conocer con rigor el desempeño de esos modelos es la condición para juzgar cualquier propuesta posterior.

## Objetivos

### Objetivo general

Proponer una arquitectura de aprendizaje profundo para la clasificación de las siete categorías diagnósticas de HAM10000 que supere, con evidencia estadística, el desempeño de los modelos de referencia empleados en la literatura.

### Objetivos específicos

1. Identificar, mediante una revisión sistemática de la literatura, los modelos de aprendizaje profundo que los estudios sobre HAM10000 utilizan como referencia comparativa.
2. Caracterizar el conjunto de datos, su estructura de dependencia, su desbalance y sus posibles atajos visuales, y definir a partir de ello una partición estratificada y agrupada por lesión.
3. Implementar y evaluar los modelos de referencia bajo un mismo pipeline, una misma partición y un mismo presupuesto de ajuste de hiperparámetros, y compararlos mediante pruebas estadísticas.
