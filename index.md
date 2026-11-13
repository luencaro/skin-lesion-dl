---
title: Clasificación de lesiones pigmentadas de la piel con aprendizaje profundo
---

## Resumen

El diagnóstico temprano del melanoma y de otras lesiones cutáneas depende en buena medida de la dermatoscopia, una técnica cuya precisión varía con la experiencia del observador. El aprendizaje profundo ha mostrado un desempeño comparable al de los dermatólogos en la clasificación de imágenes dermatoscópicas, y el conjunto de datos HAM10000 se ha convertido en la referencia más utilizada para esta tarea. Sin embargo, buena parte de los resultados publicados se obtiene con protocolos que permiten la fuga de información entre entrenamiento y prueba, sin repeticiones y sin inferencia estadística, de modo que es difícil saber cuánto mejora realmente un modelo nuevo respecto de los modelos con los que se compara.

Este proyecto tiene como propósito proponer una arquitectura de aprendizaje profundo para la clasificación de las siete categorías diagnósticas de HAM10000 que supere a los modelos de referencia empleados en la literatura. Como punto de partida, una revisión sistemática permitió identificar esos modelos de referencia, VGG-16, ResNet-50 y MobileNetV2, cuyo análisis se presenta en este documento.

## Contenido

- [Descripción del proyecto](book/01-descripcion-del-proyecto.md): planteamiento del problema, antecedentes, justificación y objetivos.
- [Estado del arte](book/02-estado-del-arte.md): estrategia de búsqueda, evaluación de los estudios e identificación de los modelos base.
- [Modelos base](book/03-modelos-base.md): análisis de VGG-16, ResNet-50 y MobileNetV2, comparación crítica y brecha en la literatura.
