# Procedencia del mecanismo

**Modelo original:** capa de constancia de color aprendible e invariante a las ganancias de canal (CCI), antepuesta al backbone del benchmark.

**Tipo de mecanismo:** transferido de otra aplicación (estimación del iluminante en fotografía computacional) y adaptado.

## Fuentes originales

- Finlayson, G. D. y Trezzi, E. (2004). Shades of gray and colour constancy. *Color and Imaging Conference*, 37–41. Estimación del iluminante con la media de Minkowski de orden p fijo.
- Hu, Y., Wang, B. y Lin, S. (2017). FC4: Fully convolutional color constancy with confidence-weighted pooling. *CVPR*, 4085–4094. doi: 10.1109/CVPR.2017.43. Estimación del iluminante con pooling ponderado por confianza, supervisada con iluminantes medidos.

## Uso previo en el problema

Shades of Gray con p fijo se ha usado como preprocesamiento en la clasificación dermatoscópica (Barata et al., 2015, doi: 10.1109/JBHI.2014.2336473), y una red generativa entrenada por separado se ha usado para estandarizar el color (Salvi et al., 2022, doi: 10.1016/j.cmpb.2022.107040).

## Modificaciones algorítmicas introducidas

1. **Orden aprendible:** p = 1 + softplus(ϑ), aprendido por gradiente en lugar de fijo.
2. **Pesos espaciales sin supervisión de iluminante:** la ponderación por confianza de FC4 se conserva, pero se entrena solo con la pérdida de clasificación, sin iluminantes etiquetados.
3. **Pesos calculados sobre una representación invariante:** los pesos dependen del logaritmo de cada canal centrado en su media espacial, que no cambia con las ganancias de canal; con ello, la capa es exactamente invariante a ellas (teorema del capítulo de diseño). Ni Shades of Gray ponderado ni FC4 tienen esta garantía.
4. **Entrenamiento de extremo a extremo:** la capa se entrena conjuntamente con el backbone y la cabeza de clasificación, no como preprocesamiento.

## Implementación

El núcleo del mecanismo (ecuaciones de guía, pesos, iluminante y corrección) se programa desde su formulación matemática como una capa propia de Keras (`tf.keras.layers.Layer`) en `src/original_model.py`, usando solo operaciones primitivas del framework. No se usa ni se adapta código de terceros.
