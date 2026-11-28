---
title: Clasificación de lesiones pigmentadas de la piel con aprendizaje profundo
---

## Resumen

El diagnóstico temprano del melanoma y de otras lesiones cutáneas depende en buena medida de la dermatoscopia, una técnica cuya precisión varía con la experiencia del observador. El aprendizaje profundo ha mostrado un desempeño comparable al de los dermatólogos en la clasificación de imágenes dermatoscópicas, y el conjunto de datos HAM10000 se ha convertido en la referencia más utilizada para esta tarea. Sin embargo, buena parte de los resultados publicados se obtiene con protocolos que permiten la fuga de información entre entrenamiento y prueba, sin repeticiones y sin inferencia estadística, de modo que es difícil saber cuánto mejora realmente un modelo nuevo respecto de los modelos con los que se compara.

Este proyecto tiene como propósito proponer una arquitectura de aprendizaje profundo para la clasificación de las siete categorías diagnósticas de HAM10000 que supere a los modelos de referencia empleados en la literatura. Como punto de partida, una revisión sistemática permitió identificar esos modelos de referencia, VGG-16, ResNet-50 y MobileNetV2, cuyo análisis se presenta en este documento.

## Contenido

- [Descripción del proyecto](book/01-descripcion-del-proyecto.md): planteamiento del problema, antecedentes, justificación y objetivos.
- [Definición del problema y del conjunto de datos](book/02-definicion-del-problema.md): formulación del problema, hipótesis, declaración a priori y descripción de HAM10000.
- [Estado del arte](book/03-estado-del-arte.md): estrategia de búsqueda, evaluación de los estudios e identificación de los modelos base.
- [Modelos base](book/04-modelos-base.md): análisis de VGG-16, ResNet-50 y MobileNetV2, comparación crítica y brecha en la literatura.
- [EDA de metadatos y partición](book/05-eda_metadatos_particion.ipynb): integridad, duplicados, metadatos y partición agrupada por lesión.
- [EDA de imágenes](book/06-eda_imagenes.ipynb): artefactos, formato, color, lesión, atajos visuales y separabilidad.
- [Diseño experimental](book/07-diseno-experimental.md): esquema de validación, pipeline de entrenamiento, ajuste de hiperparámetros, reproducibilidad, estudio piloto, plan de análisis estadístico y limitaciones inferenciales.
- [MobileNetV2](book/08-mobilenetv2.ipynb): ajuste de hiperparámetros, estudio piloto, ejecución definitiva y evaluación.
- [ResNet-50](book/09-resnet50.ipynb): ajuste de hiperparámetros, estudio piloto, ejecución definitiva y evaluación.
- [VGG-16](book/10-vgg16.ipynb): ajuste de hiperparámetros, estudio piloto, ejecución definitiva y evaluación.
- [Benchmark](book/11-benchmark.ipynb): piloto conjunto, tabla comparativa, ranking, baseline y pruebas de hipótesis.
- [Preguntas de investigación y conjunto de datos](book/12-preguntas-y-datos.md): formulación, preguntas RQ1 a RQ3, justificación y ficha de HAM10000.
- [Modelos del estado del arte](book/13-modelos-estado-del-arte.md): selección, análisis por modelo, comparación crítica y gap de investigación.
- [Paper guía](book/14-paper-guia.md): selección, descripción, criterio de replicación y limitación identificada.
- [Revisión de literatura y novedad](book/23-novedad.md): búsqueda, trabajo cercano, fuente del mecanismo y declaración de novedad.
- [Modelo original: diseño](book/24-modelo-original-diseno.md): limitación abordada, hipótesis del mecanismo, formulación de la capa CCI, pseudocódigo, demostración de la invarianza, complejidad y ablación.
- [Modelo original: implementación](book/25-modelo-original-implementacion.ipynb): código de la capa, verificación numérica de la invarianza, comparación con el backbone, complejidad empírica y visualización de $p$, los pesos y el iluminante.
- [DenseNet-201](book/15-densenet201.ipynb): ajuste de hiperparámetros, estudio piloto, ejecución definitiva y evaluación.
- [Inception-V3](book/16-inceptionv3.ipynb): ajuste de hiperparámetros, estudio piloto, ejecución definitiva y evaluación.
- [ConvNeXt-Tiny](book/17-convnext_tiny.ipynb): ajuste de hiperparámetros, estudio piloto, ejecución definitiva y evaluación.
- [ViT-B/16](book/18-vit_b16.ipynb): ajuste de hiperparámetros, estudio piloto, ejecución definitiva y evaluación.
- [EfficientNet-B0](book/19-efficientnetb0.ipynb): ajuste de hiperparámetros, estudio piloto, ejecución definitiva y evaluación.
- [Benchmark del Entregable 2](book/20-benchmark-e2.ipynb): tabla principal, contrastes confirmatorios, comparación global, bondad de ajuste, sobreajuste, efectos mixtos, calibración, incertidumbre, robustez, interpretabilidad, errores, techo, costo, réplica y limitación del paper guía.
- [Auditoría de fuga y contrato de datos](book/21-auditoria-fuga.ipynb): intersección de lesiones, duplicados, vecinos cercanos, metadatos, validación adversarial, pruebas de sanidad y contrato de datos.
- [Reproducibilidad](book/22-reproducibilidad.md): entorno, comandos, seguimiento de experimentos, lista de verificación y desviaciones declaradas.
