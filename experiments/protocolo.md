# Protocolo experimental pre-registrado (Entregable 3)

Plantilla con los campos mínimos de la sección 8.1 del Entregable 3. Se congela antes de ejecutar los
experimentos definitivos del modelo original, con una etiqueta de Git y su fecha. Lo que ya está fijado
en el diseño (capítulos 23 y 24, `configs/`) se indica con su fuente; los marcadores [COMPLETAR] deben
llenarse antes de congelar. Cualquier desviación posterior se declara y justifica en el informe; los
análisis no previstos se reportan como exploratorios y no sustentan por sí solos la condición R3.

- **Fecha de congelamiento:** 2026-11-26
- **Tag de Git:** `e3-protocolo` · **Commit:** el que señala el tag
- **Responsable:** Luis Cabarcas

## 1. Estándar de profundidad

Se adopta el estándar único de la sección 2 y sus cinco requisitos, declarados en
`experiments/estandar.md` (tag `e3-estandar`): sustento teórico demostrado, evaluación amplia en tres
conjuntos adicionales, familia completa de contrastes, potencia ≥ 0,90 por contraste y al menos tres
pruebas de hipótesis por contraste.

- Compromiso confirmado sin cambios: Sí, con las desviaciones D1 a D5 de la sección 10, declaradas antes de congelar el protocolo. Afectan al requisito 2 (los conjuntos adicionales se evalúan sin reentrenar) y al requisito 4 (5 corridas con una semilla, por debajo de las 10 semillas y de la potencia de 0,90 exigidas).
- Si el cómputo no alcanza la potencia exigida, que es lo esperado con R = 5, se reporta la diferencia mínima detectable calculada a posteriori (capítulo 7, «Número de corridas») y el resultado se interpreta como no concluyente.

## 2. Hipótesis

### Confirmatorias

| Id | Hipótesis | H0 / H1 | Fuente |
|---|---|---|---|
| H1 (R3) | El modelo original supera a cada modelo del benchmark y a cada baseline fuerte en la métrica primaria en más de δ | H0: μ_CCI − μ_k ≤ 0; H1: μ_CCI − μ_k > 0, con relevancia si la diferencia ≥ δ | Entregable 3, sección 8.4 |
| H2 (mecanismo) | La capa CCI reduce en al menos δ la diferencia de exactitud balanceada en test entre vidir_molemax y el resto de fuentes respecto del mismo backbone sin la capa | IC 95 % por bootstrap de lesiones de la reducción con límite inferior > 0 y estimación ≥ δ | Capítulo 24 |
| H3 (no inferioridad) | La exactitud balanceada global del modelo original no es peor que la del backbone sin la capa | TOST con margen ±δ | Capítulo 24 |

### Predicciones secundarias del mecanismo (capítulo 24)

1. Bajo perturbaciones de ganancia por canal, la exactitud balanceada del modelo con la capa no cambia, mientras que la del backbone sin la capa cae con la severidad. La predicción se contrasta con ganancias por canal log-uniformes entre 0,5 y 1, que oscurecen la imagen sin saturar píxeles y permiten comprobar la invarianza exacta. Se reporta además, como análisis de sensibilidad, el rango entre 0,5 y 2, donde la saturación de alrededor del 20 % de los píxeles rompe parcialmente la invarianza.
2. La ventaja desaparece al retirar la invarianza (variante `cci_no_invariance`).
3. La capa no aporta ventaja frente a desenfoque, ruido gaussiano o compresión JPEG.
4. Un clasificador lineal predice la fuente con menor exactitud balanceada a partir de las
   representaciones del modelo con la capa.

- Clasificación de cada predicción (confirmatoria o exploratoria): H1, H2 y H3 son confirmatorias. La predicción 1 es confirmatoria, porque forma parte del criterio de éxito del capítulo 24, y se evalúa en el rango de 0,5 a 1. Las predicciones 2, 3 y 4, y el rango de 0,5 a 2 de la predicción 1, son exploratorios.

### Exploratorias

- Desempeño por fuente de adquisición (cada fuente frente al resto, no solo vidir_molemax), por clase, por sexo, por grupo de edad y por localización, con corrección de Benjamini–Hochberg. Interpretabilidad: mapas Grad-CAM, pesos espaciales $w$, orden $p$ aprendido e iluminante estimado $e_c$ por fuente. Desempeño por clase en los conjuntos adicionales.

## 3. Métricas y margen de relevancia

- **Métrica primaria:** exactitud balanceada en test (heredada del Entregable 1, `benchmark.primary_metric`).
- **Secundarias:** F1 macro, F1 ponderado, exactitud, AUC macro uno contra el resto, AUC-PR macro, MCC,
  ECE, ECE adaptativo, log-loss y Brier (`e2_stats.table_metrics`).
- **Margen de relevancia práctica:** δ = 0,02 en exactitud balanceada absoluta (`benchmark.delta`).
- Confirmación de que se fijaron antes de ver los resultados finales: Sí. La métrica primaria, las secundarias y δ se heredan del Entregable 1 (capítulo 7 y `preregistro.md`) y no se modificaron.

## 4. Fuente del mecanismo y justificación de la transferencia

- **Tipo:** transferido de la estimación del iluminante en fotografía computacional y adaptado
  (`experiments/procedencia.md`).
- **Fuentes:** Finlayson y Trezzi (2004), Shades of Gray; Hu, Wang y Lin (2017), FC4.
- **Justificación:** las imágenes de dermatoscopia de distintos equipos difieren en primer orden por
  una ganancia multiplicativa por canal (modelo diagonal de von Kries), y el EDA muestra que el color
  depende mucho más de la fuente (ε² = 0,669 en saturación) que de la clase (0,115).
- **Adaptaciones:** p aprendible, pesos espaciales sin iluminantes etiquetados, pesos sobre una
  representación invariante (invarianza exacta demostrada) y entrenamiento de extremo a extremo.

## 5. Conjuntos de datos adicionales

Elegidos sin conocer el desempeño del modelo en ellos.

| # | Conjunto | Fuente pública y licencia | Tamaño | Dominio | Tipo de dependencia o desplazamiento | Criterio de selección |
|---|---|---|---|---|---|---|
| 1 | PAD-UFES-20 (Pacheco et al., 2020) | Mendeley Data, doi: 10.17632/zr7vgbcyr2.1; CC BY 4.0 | 2298 imágenes de 1641 lesiones y 1373 pacientes; 2106 tras excluir SCC | Fotografía clínica con teléfono móvil, sin dermatoscopio (Brasil) | Varias imágenes por lesión (bootstrap por lesión); cambio de modalidad de imagen | Difiere en dominio (clínica frente a dermatoscopia) y en tamaño y desbalance (predominan bcc y akiec) |
| 2 | Derm7pt (Kawahara et al., 2019) | derm.cs.sfu.ca, con registro; uso para investigación (verificar la licencia al descargar) | 1011 casos; se usa solo la imagen dermatoscópica | Dermatoscopia digitalizada de un atlas, otro centro y otra época | Un caso por lesión; cambio de equipo y de proceso de captura | Difiere en tamaño (unas 10 veces menor) y en desbalance (25 % de melanomas) |
| 3 | BCN20000 (Hernández-Pérez et al., 2024) | Figshare, doi: 10.6084/m9.figshare.24140028; CC BY 4.0 | 12 413 imágenes etiquetadas de la partición de entrenamiento; se excluyen SCC | Dermatoscopia del Hospital Clínic de Barcelona, tres cámaras, 2010–2016, con imágenes difíciles sin recortar | Varias imágenes por lesión (bootstrap por lesión); cambio de centro y de equipo | Difiere en dominio (centro, equipo y población) y en desbalance; es el conjunto en el que el paper guía observa la caída de 25 puntos |

- Mapeo de clases a las 7 de HAM10000 y tratamiento de las clases no compartidas: PAD-UFES-20: ACK → akiec, BCC → bcc, MEL → mel, NEV → nv, SEK → bkl. Derm7pt: los subtipos de melanoma → mel, los subtipos de nevo → nv, basal cell carcinoma → bcc, seborrheic keratosis y lentigo → bkl, dermatofibroma → df, vascular lesion → vasc. BCN20000: NV → nv, MEL → mel, BCC → bcc, BKL → bkl, AK → akiec, DF → df, VASC → vasc. Se excluyen las imágenes de clases sin equivalente en HAM10000: SCC en PAD-UFES-20 y BCN20000 (el carcinoma escamoso invasivo no forma parte de akiec), y melanosis y miscellaneous en Derm7pt. El modelo predice siempre sobre las 7 clases. La exactitud balanceada de cada conjunto es la media de la sensibilidad sobre las clases presentes en él, de modo que predecir una clase ausente cuenta como error.
- Uso (solo evaluación externa, sin reentrenar): Solo evaluación externa, sin reentrenar ni ajustar (desviación D5). Cada una de las 5 corridas definitivas de cada modelo se evalúa en los tres conjuntos con el mismo preprocesamiento de HAM10000 (redimensionado a 224 × 224 sin conservar la relación de aspecto). Ninguna decisión de diseño se toma con ellos. Los resultados se reportan por conjunto, sin promediar entre conjuntos, y se consolidan con la prueba de Friedman con la corrección de Iman–Davenport sobre los cuatro conjuntos (HAM10000 y los tres adicionales), con comparaciones del modelo original frente a cada modelo con Holm. Las sumas SHA-256 de los archivos descargados se registran en `data/external_hashes.json`.

## 6. Particiones, ejecuciones, semillas y determinismo

- **Partición de HAM10000:** `data/splits.csv` (SHA-256 en `data/splits_hash.json`), agrupada por lesión:
  test de 1431 imágenes y train de 8584 en 5 pliegues.
- **Partición de desarrollo y de confirmación:** Desarrollo: el pliegue de validación 0, usado para el ajuste de hiperparámetros y para cualquier decisión de diseño. Confirmación: el conjunto de prueba, evaluado una sola vez al final de cada corrida definitiva, después de congelar el protocolo. Los pliegues de validación 1 a 4 solo se usan para la parada temprana de las corridas definitivas.
- **Ejecuciones:** 1 semilla × 5 pliegues = 5 corridas por modelo (desviación D3).
- **Semillas:** [0]. Por cómputo, por debajo de las 10 semillas exigidas (desviación D3).
- **Determinismo:** `train.deterministic` = false; precisión mixta `train.mixed_precision` = true, con la capa CCI en float32; versiones en `uv.lock` y `requirements-train.txt`; GPU NVIDIA GeForce RTX 3050 Laptop de 6 GB, con el controlador, CUDA y cuDNN registrados en cada corrida.

## 7. Familia de contrastes, pruebas y corrección

- **Familia confirmatoria (R3):** El modelo original frente a cada uno de los otros 9 modelos del benchmark (baseline trivial, MobileNetV2, ResNet-50, VGG-16, DenseNet-201, Inception-V3, ConvNeXt-Tiny, ViT-B/16 y EfficientNet-B0) y frente al baseline fuerte (regresión logística sobre embeddings congelados de MobileNetV2): 10 contrastes con Holm.
- **Pruebas por contraste (al menos tres):** t pareada unilateral con corrección de Nadeau y Bengio, Wilcoxon de rangos con signo unilateral y bootstrap pareado por lesión sobre el conjunto de prueba (10 000 remuestreos). McNemar y DeLong como complementarias. Con 5 corridas, el p-valor mínimo de Wilcoxon (0,031) no supera el umbral más estricto de Holm con 10 contrastes (0,005), de modo que la condición de tres pruebas significativas de la sección 8.4 no puede cumplirse; se declara antes de ver los resultados.
- **Corrección por comparaciones múltiples:** Holm sobre toda la familia (`benchmark.p_adjust`).
- **Nivel de significancia:** α = 0,05 (`benchmark.alpha`).
- **Regla de decisión:** La misma del Entregable 2. Superior y relevante: media favorable, contraste significativo tras Holm y diferencia estimada de al menos δ. Superior sin relevancia: significativo, con diferencia menor que δ. Equivalente: TOST con ±δ significativo. No concluyente: en cualquier otro caso.
- **Familia de la ablación (separada, Holm):** modelo completo frente a `cci_fixed_p`,
  `cci_uniform_weights`, `cci_no_invariance`, `cci_shades_of_gray` y el backbone sin la capa.

## 8. Presupuesto de ajuste y cómputo; análisis de potencia

- **Presupuesto de ajuste por modelo:** búsqueda aleatoria de `train.tuning.n_trials` = 6 configuraciones (desviación D1)
  con la misma semilla del muestreador, el mismo espacio relativo (tasa de aprendizaje, decaimiento de
  pesos y dropout), `train.tuning.max_epochs` = 15 y la misma parada temprana. Igual para el modelo
  original; las variantes de ablación usan los hiperparámetros del modelo completo (desviación D4).
- **Presupuesto de cómputo:** horas de GPU previstas por modelo en `experiments/presupuesto_computo.md` (GPU NVIDIA GeForce RTX 3050
  Laptop de 6 GB): en total, entre 28,0 h (caso típico, parada temprana hacia la época 15) y 36,3 h (peor caso) para los nueve modelos. Se reporta el consumido real (`src/cost.py`, `cost_from_runs`).
- **Análisis de potencia:** R = 5, fijado por cómputo sin piloto (desviación D3). σ̂_d, δ/σ̂_d, la potencia por contraste, la potencia conjunta y la diferencia mínima detectable se calculan a posteriori con las corridas definitivas (capítulo 7, «Número de corridas») y se reportan en el notebook del modelo original. Se espera que la potencia quede por debajo de 0,90.

## 9. Variantes de diseño y bitácora

- **Número de variantes de diseño evaluadas, Nd:** 1 (registro en `experiments/design_log.md`;
  1 a la fecha del diseño).
- Todas las variantes se evaluaron solo en la partición de desarrollo: Sí. La única variante de diseño es la del capítulo 24, definida sin evaluar ninguna corrida. Las variantes de ablación están predeclaradas y no son iteraciones de diseño.

## 10. Desviaciones

| Fecha | Desviación | Justificación | Efecto esperado |
|---|---|---|---|
| 2026-11-26 | D1: ajuste con 6 configuraciones en lugar de 12 | Cómputo | Hiperparámetros algo menos optimizados, por igual en todos los modelos |
| 2026-11-26 | D2: sin estudio piloto | Cómputo; el número de corridas se fija por cómputo, no por potencia | La potencia solo se conoce a posteriori |
| 2026-11-26 | D3: 5 corridas (semilla 0 × 5 pliegues) en lugar de al menos 10 semillas | Cómputo | Potencia baja; Wilcoxon no puede rechazar tras Holm; R3 no alcanzable |
| 2026-11-26 | D4: ablación sin ajuste propio | Cómputo | Ligera ventaja del modelo completo frente a sus variantes |
| 2026-11-26 | D5: conjuntos adicionales solo para evaluación externa | Cómputo; reentrenar los 11 modelos en cada conjunto multiplicaría el cómputo por cuatro | Mide la generalización fuera de distribución, no el desempeño entrenando en cada conjunto |
