# Protocolo experimental pre-registrado (Entregable 3)

Plantilla con los campos mínimos de la sección 8.1 del Entregable 3. Se congela antes de ejecutar los
experimentos definitivos del modelo original, con una etiqueta de Git y su fecha. Lo que ya está fijado
en el diseño (capítulos 23 y 24, `configs/`) se indica con su fuente; los marcadores [COMPLETAR] deben
llenarse antes de congelar. Cualquier desviación posterior se declara y justifica en el informe; los
análisis no previstos se reportan como exploratorios y no sustentan por sí solos la condición R3.

- **Fecha de congelamiento:** [COMPLETAR: AAAA-MM-DD]
- **Tag de Git:** [COMPLETAR: p. ej. `e3-protocolo`] · **Commit:** [COMPLETAR]
- **Responsable:** [COMPLETAR]

## 1. Estándar de profundidad

Se adopta el estándar único de la sección 2 y sus cinco requisitos, declarados en
`experiments/estandar.md` (tag `e3-estandar`): sustento teórico demostrado, evaluación amplia en tres
conjuntos adicionales, familia completa de contrastes, potencia ≥ 0,90 por contraste y al menos tres
pruebas de hipótesis por contraste.

- Compromiso confirmado sin cambios: [COMPLETAR: sí / desviaciones declaradas]
- Si el cómputo no alcanza la potencia exigida: diferencia mínima detectable declarada [COMPLETAR] y
  resultado interpretado como no concluyente.

## 2. Hipótesis

### Confirmatorias

| Id | Hipótesis | H0 / H1 | Fuente |
|---|---|---|---|
| H1 (R3) | El modelo original supera a cada modelo del benchmark y a cada baseline fuerte en la métrica primaria en más de δ | H0: μ_CCI − μ_k ≤ 0; H1: μ_CCI − μ_k > 0, con relevancia si la diferencia ≥ δ | [COMPLETAR: confirmar redacción] |
| H2 (mecanismo) | La capa CCI reduce en al menos δ la diferencia de exactitud balanceada en test entre vidir_molemax y el resto de fuentes respecto del mismo backbone sin la capa | IC 95 % por bootstrap de lesiones de la reducción con límite inferior > 0 y estimación ≥ δ | Capítulo 24 |
| H3 (no inferioridad) | La exactitud balanceada global del modelo original no es peor que la del backbone sin la capa | TOST con margen ±δ | Capítulo 24 |

### Predicciones secundarias del mecanismo (capítulo 24)

1. Bajo ganancias por canal con factores entre 0,5 y 2, la exactitud balanceada del modelo con la capa
   no cambia (salvo por saturación); la del backbone sin la capa cae con la severidad.
2. La ventaja desaparece al retirar la invarianza (variante `cci_no_invariance`).
3. La capa no aporta ventaja frente a desenfoque, ruido gaussiano o compresión JPEG.
4. Un clasificador lineal predice la fuente con menor exactitud balanceada a partir de las
   representaciones del modelo con la capa.

- Clasificación de cada predicción (confirmatoria o exploratoria): [COMPLETAR]

### Exploratorias

- [COMPLETAR: p. ej. diferencias por subgrupo (sexo, edad, localización), comparación CNN frente a
  transformador, interpretabilidad de los pesos espaciales].

## 3. Métricas y margen de relevancia

- **Métrica primaria:** exactitud balanceada en test (heredada del Entregable 1, `benchmark.primary_metric`).
- **Secundarias:** F1 macro, F1 ponderado, exactitud, AUC macro uno contra el resto, AUC-PR macro, MCC,
  ECE, ECE adaptativo, log-loss y Brier (`e2_stats.table_metrics`).
- **Margen de relevancia práctica:** δ = 0,02 en exactitud balanceada absoluta (`benchmark.delta`).
- Confirmación de que se fijaron antes de ver los resultados finales: [COMPLETAR]

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
| 1 | [COMPLETAR: p. ej. BCN20000] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR: otro centro y equipo] | [COMPLETAR] |
| 2 | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |
| 3 | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |

- Mapeo de clases a las 7 de HAM10000 y tratamiento de las clases no compartidas: [COMPLETAR]
- Uso (solo evaluación externa, sin reentrenar): [COMPLETAR]

## 6. Particiones, ejecuciones, semillas y determinismo

- **Partición de HAM10000:** `data/splits.csv` (SHA-256 en `data/splits_hash.json`), agrupada por lesión:
  test de 1431 imágenes y train de 8584 en 5 pliegues.
- **Partición de desarrollo y de confirmación:** [COMPLETAR: p. ej. desarrollo = pliegue 0 de validación
  para el diseño y el ajuste; confirmación = test, usado una sola vez].
- **Ejecuciones:** semillas × pliegues de `final.seeds` × `final.folds` = [COMPLETAR: número] corridas por
  modelo.
- **Semillas:** [COMPLETAR: lista; al menos 10 o las del análisis de potencia].
- **Determinismo:** `train.deterministic` = [COMPLETAR]; precisión mixta `train.mixed_precision` = [COMPLETAR];
  versiones en `uv.lock` y `requirements-train.txt`; GPU [COMPLETAR].

## 7. Familia de contrastes, pruebas y corrección

- **Familia confirmatoria (R3):** el modelo original frente a cada uno de los modelos del Top 10
  [COMPLETAR: lista] y frente a cada baseline fuerte [COMPLETAR: lista].
- **Pruebas por contraste (al menos tres):** [COMPLETAR: p. ej. t pareada corregida de Nadeau y Bengio,
  Wilcoxon de rangos con signo y bootstrap por lesión; McNemar o DeLong como complemento].
- **Corrección por comparaciones múltiples:** Holm sobre toda la familia (`benchmark.p_adjust`).
- **Nivel de significancia:** α = 0,05 (`benchmark.alpha`).
- **Regla de decisión:** [COMPLETAR: superior y relevante / superior sin relevancia / equivalente / no
  concluyente, como en el Entregable 2].
- **Familia de la ablación (separada, Holm):** modelo completo frente a `cci_fixed_p`,
  `cci_uniform_weights`, `cci_no_invariance`, `cci_shades_of_gray` y el backbone sin la capa.

## 8. Presupuesto de ajuste y cómputo; análisis de potencia

- **Presupuesto de ajuste por modelo:** búsqueda aleatoria de `train.tuning.n_trials` = 12 configuraciones
  con la misma semilla del muestreador, el mismo espacio relativo (tasa de aprendizaje, decaimiento de
  pesos y dropout), `train.tuning.max_epochs` = 15 y la misma parada temprana. Igual para el modelo
  original y cada variante de ablación.
- **Presupuesto de cómputo:** [COMPLETAR: horas de GPU previstas por modelo; GPU NVIDIA GeForce RTX 3050
  Laptop de 6 GB]. Se reporta el consumido real (`src/cost.py`, `cost_from_runs`).
- **Análisis de potencia:** σ̂_d del piloto = [COMPLETAR]; δ / σ̂_d = [COMPLETAR]; potencia objetivo 0,90 por
  contraste; corridas necesarias R = [COMPLETAR]; potencia conjunta = [COMPLETAR]; diferencia mínima
  detectable con el cómputo disponible = [COMPLETAR].

## 9. Variantes de diseño y bitácora

- **Número de variantes de diseño evaluadas, Nd:** [COMPLETAR] (registro en `experiments/design_log.md`;
  1 a la fecha del diseño).
- Todas las variantes se evaluaron solo en la partición de desarrollo: [COMPLETAR: confirmar].

## 10. Desviaciones

| Fecha | Desviación | Justificación | Efecto esperado |
|---|---|---|---|
| [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |
