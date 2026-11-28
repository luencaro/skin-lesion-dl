# Ficha del modelo: capa de constancia de color invariante (CCI)

Ficha del modelo original del proyecto, siguiendo el formato de *model cards* (Mitchell et al., 2019).
Las cifras de desempeño están pendientes de las corridas definitivas y se completan con
`python -m src.export_dashboard` y el informe del Entregable 3.

## Detalles del modelo

- **Tipo:** clasificador de imágenes dermatoscópicas en 7 clases de HAM10000 (akiec, bcc, bkl, df,
  mel, nv, vasc).
- **Arquitectura:** capa de constancia de color aprendible e invariante a las ganancias de canal
  (CCI) antepuesta a un backbone preentrenado en ImageNet: f = softmax ∘ b_ψ ∘ h. La capa estima el
  iluminante de cada canal con una media de Minkowski ponderada de orden p aprendible
  (Shades of Gray, Finlayson y Trezzi, 2004) y pesos espaciales aprendidos (como FC4, Hu et al.,
  2017), calculados sobre una representación que no cambia con las ganancias de canal, y divide
  cada canal por su iluminante. Añade 2786 parámetros al backbone.
- **Backbone:** [COMPLETAR: modelo del benchmark con mayor exactitud balanceada media en validación
  en el Entregable 2, regla del capítulo 24; `backbone` en `configs/cci.yaml`].
- **Propiedad garantizada:** h(Dx) = h(x) para toda ganancia diagonal D con d_c > 0 (demostración
  en el capítulo 24; verificación numérica en `tests/` y en el capítulo 25).
- **Implementación:** `src/original_model.py` (TensorFlow 2.20 / Keras 3); exportado a ONNX
  (`src/export.py`) para la inferencia en CPU.
- **Versión y fecha:** [COMPLETAR: tag de Git y fecha de la corrida exportada].
- **Autor:** Luis Cabarcas, Universidad del Norte. Licencia del código: MIT.

## Uso previsto

- Investigación académica sobre la dependencia del desempeño respecto de la fuente de adquisición
  en dermatoscopia, y docencia.
- Exploración interactiva en el dashboard de cómo el modelo predice, con su incertidumbre
  (probabilidades calibradas y conjunto de predicción conforme) y sus mapas de activación.

## Usos no recomendados

- **Diagnóstico, cribado o triaje clínico, o cualquier decisión sobre biopsia, tratamiento o
  vigilancia.** El modelo no está validado clínicamente ni aprobado por ninguna autoridad sanitaria.
- Imágenes que no sean dermatoscópicas (fotografías clínicas o de teléfono sin dermatoscopio),
  lesiones no pigmentadas o diagnósticos fuera de las 7 clases: el modelo siempre elige una de las
  7 clases, aunque la imagen no corresponda a ninguna.
- Poblaciones con fototipos oscuros u otros equipos de adquisición sin una validación externa
  específica.
- Uso comercial (la licencia de los datos de entrenamiento lo prohíbe).

## Datos

- **Entrenamiento y evaluación:** HAM10000 versión 4 (Tschandl, Rosendahl y Kittler, 2018), 10015
  imágenes de 7470 lesiones, partición agrupada por lesión: test de 1431 imágenes (14,3 %) y
  train de 8584 en 5 pliegues. Ver `DATA_STATEMENT.md`.
- **Preprocesamiento:** redimensionado a 224 × 224 (bilineal con antialias, sin conservar la
  relación de aspecto), RGB en [0, 255]; la normalización va dentro del modelo.
- **Conjuntos adicionales:** PAD-UFES-20, Derm7pt y BCN20000, solo para evaluación externa (ver `experiments/protocolo.md`, sección 5).

## Entrenamiento

Mismo pipeline que los modelos del benchmark: fase A con el backbone congelado (la capa y la cabeza
se entrenan), fase B con todo descongelado, AdamW con decaimiento coseno, entropía cruzada
ponderada por la frecuencia inversa de cada clase, el mismo aumento de datos, parada temprana sobre
la exactitud balanceada de validación y el mismo presupuesto de ajuste de hiperparámetros
(`configs/base.yaml`, sección `train`). ϑ (orden p) no lleva decaimiento de pesos.

## Métricas

Métrica primaria: exactitud balanceada en test (media ± desviación estándar entre corridas).
Secundarias: F1 macro, AUC macro, MCC, ECE, log-loss, Brier; diferencia de exactitud balanceada
entre vidir_molemax y el resto de fuentes; robustez a ganancias de canal, desenfoque, ruido y JPEG.

| Métrica | Modelo original | Backbone sin la capa |
|---|---|---|
| Exactitud balanceada (test) | [COMPLETAR] | [COMPLETAR] |
| F1 macro | [COMPLETAR] | [COMPLETAR] |
| ECE (antes / después de temperatura) | [COMPLETAR] | [COMPLETAR] |
| Diferencia vidir_molemax − resto (IC 95 %) | [COMPLETAR] | [COMPLETAR] |
| Cobertura conforme al 90 % | [COMPLETAR] | [COMPLETAR] |
| Latencia en CPU (mediana, ms) | [COMPLETAR] | [COMPLETAR] |

Veredicto del diseño (criterio del capítulo 24): [COMPLETAR].

## Incertidumbre y explicaciones

- Probabilidades calibradas con temperature scaling ajustado en validación.
- Conjunto de predicción conforme (LAC) con cobertura nominal del 90 % o el 95 %.
- Aviso de entrada fuera de dominio: distancia coseno media a los 5 vecinos más cercanos entre
  embeddings de train, con umbral en el cuantil 0,99 de validación.
- CAM sin gradientes en los backbones con pooling global, y visualización de p, de los pesos
  espaciales y del iluminante estimado de la capa.

## Limitaciones

- La invarianza es exacta frente a ganancias diagonales de canal, no frente al cambio de fuente
  real (saturación, gamma, viñeteado). Con píxeles saturados se rompe; el capítulo 25 muestra que,
  con ganancias de hasta 2 y recorte a 255, la capa solo reduce el cambio de las probabilidades.
- Un solo conjunto de entrenamiento de dos centros, con poblaciones mayoritariamente de piel clara y
  sin registro del fototipo.
- El 47 % de las etiquetas no tiene confirmación histopatológica.
- Clases minoritarias pequeñas (df: 115 imágenes; vasc: 142), con estimaciones de sensibilidad inestables.
- La búsqueda de novedad es limitada (una cadena en Scopus y GitHub; capítulo 23).

## Riesgos y consideraciones éticas

- Riesgo de uso indebido como herramienta diagnóstica: el dashboard lo advierte en cada predicción.
- Desempeño desigual entre fuentes y subgrupos: se reporta desagregado por fuente, sexo, edad y
  localización; un error sistemático en un grupo se trasladaría a decisiones clínicas.
- Falsos negativos de melanoma: un error de este tipo es mucho más costoso que uno benigno; la
  exactitud balanceada no captura esa asimetría.
- Privacidad: las imágenes cargadas en el dashboard no se guardan ni se registran.
