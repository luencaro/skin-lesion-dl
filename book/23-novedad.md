---
title: Revisión de literatura y novedad
---

El modelo original del proyecto es una capa de constancia de color aprendible e invariante a las ganancias de canal, situada antes del backbone y entrenada de extremo a extremo con la pérdida de clasificación. Este capítulo documenta la búsqueda que sustenta su novedad, la fuente del mecanismo transferido y su diferencia con el trabajo cercano.

## Búsqueda

La búsqueda se ejecutó el 26 de noviembre de 2026 en Scopus y en GitHub. Papers with Code, que el protocolo menciona como fuente de implementaciones, dejó de estar disponible en 2025. Por limitación de tiempo y de acceso, se usó una sola cadena en Scopus, que combina la familia del mecanismo, sus sinónimos en inglés y en español y el problema de aplicación; el protocolo pide al menos tres bases y cinco cadenas, y esta reducción se declara como limitación de la evidencia de novedad.

| Base | Cadena | Resultados | Revisados | Relevantes | Observación |
|---|---|---|---|---|---|
| Scopus | `TITLE-ABS-KEY ( ( "color constancy" OR "colour constancy" OR "constancia de color" OR "shades of gray" OR "shades of grey" OR "illuminant estimation" ) AND ( dermoscop* OR dermatoscop* OR "skin lesion*" OR "skin cancer" OR melanoma OR HAM10000 OR ISIC ) )` | 40 | 10 (los más citados) | 5 | Constancia de color fija como preprocesamiento o normalización aprendida por separado; ninguno entrena la constancia de extremo a extremo con el clasificador |
| GitHub | `"color constancy" skin lesion` (repositorios) | No registrado | Todos | 1 | Única implementación relacionada: la de @barata2015, con Shades of Gray fijo |

## Trabajo cercano

La diferencia con cada trabajo se clasifica en mecanismo, pérdida o entrenamiento, o solo aplicación; solo las dos primeras sustentan la novedad.

| Trabajo | Qué hace | Qué conserva la propuesta | Diferencia | Clase de diferencia |
|---|---|---|---|---|
| @finlayson2004 | Estimación del iluminante con la media de Minkowski de orden $p$ fijo (Shades of Gray) | La media de Minkowski como estimador del iluminante | $p$ aprendible por gradiente, pesos espaciales aprendidos e integración en la red | Mecanismo (fuente original) |
| @hu2017 | Red convolucional que estima el iluminante con pooling ponderado por confianza, supervisada con iluminantes reales (FC4) | La ponderación espacial aprendida de la estimación | Sin iluminantes etiquetados: se entrena solo con la pérdida de clasificación; pesos calculados sobre una representación invariante, lo que garantiza la invarianza exacta | Mecanismo y entrenamiento (fuente original) |
| @barata2015 y @barata2014icip | Shades of Gray con $p$ fijo como preprocesamiento antes de clasificar | El objetivo de normalizar el color en dermatoscopia | Capa entrenable de extremo a extremo con invarianza demostrada | Mecanismo |
| @salvi2022 | Red generativa adversaria entrenada por separado para estandarizar el color de imágenes dermatológicas | La normalización aprendida del color | Entrenamiento conjunto con el clasificador, sin red generativa y con garantía de invarianza | Mecanismo y entrenamiento |
| @ng2019 | Efecto de algoritmos de constancia fijos sobre la segmentación de lesiones | El uso de la constancia de color en dermatoscopia | Otra tarea; algoritmos fijos | Mecanismo |
| @tang2020 | Transformaciones de color fijas como fuente de diversidad de un ensamble | La transformación del color de entrada | Transformación aprendida dentro de un solo modelo | Mecanismo |
| @desale2024 | Transformador de visión con preprocesamiento de color fijo | El preprocesamiento del color | Capa aprendible e invariante | Mecanismo |
| @pinar2026 | Paper guía: compara seis arquitecturas con partición agrupada por lesión | El protocolo de evaluación | Ataca la dependencia de la fuente de adquisición que el paper guía no analiza | Mecanismo |

## Declaración de novedad

Al mejor de nuestro conocimiento y según la búsqueda descrita, no se encontraron estudios que apliquen a la clasificación de lesiones dermatoscópicas una capa de constancia de color aprendida de extremo a extremo con la pérdida de clasificación e invariante, por construcción, a las ganancias de canal. Los trabajos más cercanos aplican Shades of Gray con un orden fijo como preprocesamiento [@barata2015] o normalizan el color con una red entrenada por separado [@salvi2022]. El mecanismo se transfiere de la estimación del iluminante en fotografía computacional [@finlayson2004; @hu2017], y su adaptación consiste en sustituir la supervisión con iluminantes reales por la pérdida de clasificación y en calcular los pesos espaciales sobre una representación invariante a las ganancias, lo que convierte la invarianza en una propiedad demostrable de la arquitectura.

La búsqueda se limitó a una cadena en una base de datos y a GitHub, por lo que la declaración tiene un alcance menor que el exigido; la búsqueda se repetirá antes de la entrega, y la verificación por un tercero se anexa en [COMPLETAR: concepto del tercero].
