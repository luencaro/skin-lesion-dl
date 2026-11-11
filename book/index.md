---
title: Clasificación de lesiones pigmentadas de la piel con aprendizaje profundo
subtitle: Un baseline riguroso de modelos base sobre HAM10000
---

## Resumen

El diagnóstico temprano del melanoma y de otras lesiones cutáneas depende en buena medida de la dermatoscopia, una técnica cuya precisión varía con la experiencia del observador. El aprendizaje profundo ha mostrado un desempeño comparable al de los dermatólogos en la clasificación de imágenes dermatoscópicas, y el conjunto de datos HAM10000 se ha convertido en la referencia más utilizada para esta tarea. Sin embargo, buena parte de los resultados publicados se obtiene con protocolos que permiten la fuga de información entre entrenamiento y prueba, sin repeticiones y sin inferencia estadística, de modo que es difícil saber cuánto mejora realmente un modelo avanzado respecto de los modelos sencillos con los que se compara.

Este proyecto aborda la clasificación de las siete categorías diagnósticas de HAM10000 con un objetivo inicial concreto: establecer, con evidencia estadística, el desempeño de los modelos base que la propia literatura utiliza como referencia. A partir de una revisión sistemática se identificaron tres de ellos, VGG-16, ResNet-50 y MobileNetV2, que se evaluarán bajo una partición agrupada por lesión, con repeticiones independientes y pruebas de hipótesis. El resultado servirá como vara de medida para los modelos avanzados que se desarrollarán en las etapas posteriores del proyecto.

## Contenido

- [Descripción del proyecto](book/descripcion-del-proyecto.md): planteamiento del problema, antecedentes, justificación y objetivos.
- [Estado del arte](book/estado-del-arte.md): estrategia de búsqueda, evaluación de los estudios e identificación de los modelos base.
- [Modelos base](book/modelos-base.md): análisis de VGG-16, ResNet-50 y MobileNetV2, comparación crítica y brecha en la literatura.
