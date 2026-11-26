# Preregistro del Entregable 2

Fecha: 2026-11-26
Versión: tag `e2-preregistro` del repositorio, creado antes del primer entrenamiento de los modelos del Entregable 2.

| Campo | Declaración |
|---|---|
| Tarea y unidad de independencia | Clasificación de imágenes en 7 clases (HAM10000); unidad de independencia: la lesión (`lesion_id`). Heredado del Entregable 1. |
| Métrica primaria | Exactitud balanceada en el conjunto de prueba. Heredada del Entregable 1. |
| Métricas secundarias | F1 macro y ponderado, exactitud, AUC-ROC macro, AUC-PR macro, MCC, log-loss, Brier, ECE (15 intervalos y adaptativo), métricas por clase. No sustentan conclusiones confirmatorias. |
| Margen de relevancia δ | 0,02 en exactitud balanceada (equivale a 2 o 3 imágenes de las clases minoritarias del conjunto de prueba). Heredado del Entregable 1. |
| Margen de sobreajuste δg | 0,15 en exactitud balanceada entre train sin aumento y validación, fijado con el ajuste del Entregable 1 en validación. |
| Modelos | Baseline trivial (siempre nv); regresión logística sobre embeddings congelados de MobileNetV2; VGG-16, ResNet-50 y MobileNetV2 (Entregable 1); DenseNet-201, Inception-V3, ConvNeXt-Tiny, ViT-B/16 y EfficientNet-B0. |
| Familias confirmatorias | Familia 1 (RQ1): los 5 modelos del estado del arte frente al mejor modelo base, con Holm. Familia 2 (RQ2): los 8 modelos profundos frente a la regresión logística, con Holm. |
| Pruebas por contraste | t pareado con corrección de Nadeau–Bengio; Wilcoxon de rangos con signo y bootstrap pareado por lesión (10000 remuestreos); Friedman con Iman–Davenport y comparaciones con Holm. McNemar y DeLong como complementarias. TOST con ±δ para afirmar equivalencia. |
| Supuestos | Shapiro–Wilk sobre las diferencias pareadas. Si se rechaza la normalidad, la decisión se basa en Wilcoxon y el bootstrap; la t se reporta como análisis de sensibilidad. |
| Número de corridas R | R = 5 por modelo (semilla 0 en los pliegues 0 a 4), fijado por cómputo antes de cualquier corrida definitiva y sin estudio piloto (desviaciones D2 y D3). Se reporta a posteriori la diferencia mínima detectable. Con 5 corridas, Wilcoxon no puede rechazar tras la corrección de Holm (capítulo 7, «Limitaciones inferenciales»). |
| Reglas de exclusión | Una corrida se considera divergente si su pérdida produce NaN o si su exactitud balanceada de validación en la mejor época es menor que 0,30; se repite con la siguiente semilla y se reporta. En el ajuste, las configuraciones divergentes cuentan como evaluadas y no se repiten. |
| Reglas de parada del ajuste | Búsqueda aleatoria de 6 configuraciones (las 6 primeras del muestreador, desviación D1; semilla 2026, idénticas para todos los modelos) sobre la tasa de aprendizaje, el decaimiento de pesos y el dropout, en el pliegue de validación 0 con un máximo de 15 épocas de ajuste fino; parada temprana con paciencia de 5 épocas sobre la exactitud balanceada de validación. |
| Búsqueda de arquitectura (sección 14) | No aplica: todos los modelos tienen arquitectura definida en su artículo y en su implementación de referencia, y conservan su cabeza de clasificación original. |
| Cómputo estimado | C_total ≈ Σ_m (6 · t_ajuste,m + 5 · t_final,m). Horas estimadas por modelo en `experiments/presupuesto_computo.md`. |
| Limitación del paper guía | Diferencia de exactitud balanceada en prueba entre vidir_molemax y el resto de fuentes, con IC 95 % por bootstrap sobre lesiones, verificada en los tres modelos base. |
| Análisis exploratorios | RQ3 (desempeño por fuente y por familia), desempeño por clase, sexo, edad y localización, con corrección de Benjamini–Hochberg; no se usan para elegir modelos. |
| Desviaciones declaradas | Búsqueda solo en Scopus, sin acceso a Web of Science ni a IEEE Xplore. Cinco modelos del estado del arte en lugar de diez, por límite de cómputo (GPU de 6 GB). El décimo modelo del proyecto es la arquitectura propia del Entregable 3. Ajuste con 6 configuraciones en lugar de 12 (D1), sin estudio piloto (D2) y con 5 corridas definitivas por modelo en lugar de al menos 6 (D3), por cómputo. |
| Congelamiento de datos | HAM10000 versión 4 (doi: 10.7910/DVN/DBW86T); hashes en `data/hashes.json` y de la partición y el conjunto de prueba en `data/splits_hash.json`. |
