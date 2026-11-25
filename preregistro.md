# Preregistro

Plantilla creada el 2026-11-25. Se completa con los resultados del ajuste y del estudio piloto
y se fija, con fecha y commit, antes de ejecutar `python -m src.final`. Después de esa fecha no
se modifica; cualquier desviación se documenta en la sección final.

| Campo | Valor |
|---|---|
| Fecha de registro | [COMPLETAR: AAAA-MM-DD] |
| Commit del código | [COMPLETAR: `git rev-parse HEAD`] |
| Autor | [COMPLETAR] |

## 1. Declaración a priori

Copiada de la [declaración a priori](book/02-definicion-del-problema.md).

| Elemento | Declaración |
|---|---|
| Métrica primaria | Exactitud balanceada (promedio de la sensibilidad de las siete clases) |
| Margen de relevancia práctica | $\delta = 0{,}02$ en exactitud balanceada, en valor absoluto |
| Nivel de significancia | $\alpha = 0{,}05$, con corrección de Holm para las comparaciones pareadas |
| Unidad de independencia | La lesión (`lesion_id`) |

Los valores de $\delta$ y $\alpha$ que usa el código están en la sección `benchmark` de
`configs/base.yaml` (`delta`, `alpha`, `power`, `p_adjust`).

## 2. Partición y hash

Partición estratificada por clase y agrupada por lesión (`data/splits.csv`), registrada en
`data/splits_hash.json`.

| Elemento | Valor |
|---|---|
| Imágenes de entrenamiento (cinco pliegues) | 8584 |
| Imágenes de prueba | 1431 |
| SHA-256 de `data/splits.csv` | `982e907e9c32b02570f21a0fe404bffa7466f70af605aac3391822a715920ba5` |
| SHA-256 de la lista ordenada de `image_id` de prueba | `408c8566deeba2c2d45837108ee5ba38103d83903a6d43b9486270c5e39715c2` |
| Archivos de datos verificados con `python -m src.verify_data` | [COMPLETAR: sí / no] |

## 3. Hiperparámetros seleccionados por modelo

Mejor ensayo de `outputs/tuning/<modelo>/best.json` (búsqueda aleatoria de 12 configuraciones
con `sampler_seed` 2026, semilla 0 y pliegue de validación 0), copiado en la sección
`hyperparameters` de `configs/<modelo>.yaml`.

| Modelo | Ensayo | Tasa de aprendizaje (fase B) | Decaimiento de pesos | Dropout | Exactitud balanceada de validación | Época restaurada |
|---|---|---|---|---|---|---|
| MobileNetV2 | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |
| ResNet-50 | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |
| VGG-16 | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |

## 4. Resultados del piloto por modelo

Exactitud balanceada de validación de las cinco corridas de `train.pilot.pairs`, sin test
(`outputs/pilot/<modelo>/summary.csv`; sección 2 de `book/11-benchmark.ipynb`).

| Modelo | Corridas | Media | Desviación estándar | Coeficiente de variación |
|---|---|---|---|---|
| MobileNetV2 | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |
| ResNet-50 | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |
| VGG-16 | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |

## 5. Variabilidad de las diferencias pareadas y tamaño muestral

Diferencias pareadas (misma semilla y mismo pliegue) de la exactitud balanceada de validación del
piloto, con $\alpha = 0{,}05$, potencia $1 - \beta = 0{,}80$ y $\delta = 0{,}02$:
$R = (z_{1-\alpha/2} + z_{1-\beta})^2 \hat{\sigma}_d^2 / \delta^2$ y diferencia mínima detectable
$(z_{1-\alpha/2} + z_{1-\beta})\, \hat{\sigma}_d / \sqrt{n}$ con las $n$ corridas del piloto
(sección 2 de `book/11-benchmark.ipynb`).

| Par | $\hat{\sigma}_d$ | $\delta / \hat{\sigma}_d$ | $R$ | Diferencia mínima detectable con el piloto |
|---|---|---|---|---|
| MobileNetV2 - ResNet-50 | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |
| MobileNetV2 - VGG-16 | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |
| ResNet-50 - VGG-16 | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |

## 6. Supuestos y alternativa no paramétrica

Normalidad de las diferencias pareadas del piloto (Shapiro-Wilk):

| Par | W | p |
|---|---|---|
| MobileNetV2 - ResNet-50 | [COMPLETAR] | [COMPLETAR] |
| MobileNetV2 - VGG-16 | [COMPLETAR] | [COMPLETAR] |
| ResNet-50 - VGG-16 | [COMPLETAR] | [COMPLETAR] |

Criterio para la ejecución definitiva, aplicado de forma automática en la sección 6.1 de
`book/11-benchmark.ipynb` sobre las diferencias de la ejecución definitiva: si Shapiro-Wilk no
rechaza la normalidad en ningún par ($p \geq \alpha$), las pruebas principales son el ANOVA de
medidas repetidas y la t pareada; en otro caso, la alternativa no paramétrica, Friedman con
post-hoc de Nemenyi y Wilcoxon de rangos con signo exacto. Las dos familias se reportan siempre,
con corrección de Holm en las comparaciones pareadas.

Observaciones sobre los supuestos: [COMPLETAR]

## 7. Número de corridas definitivas

Las corridas definitivas son el producto semillas x pliegues de la sección `final` de
`configs/base.yaml` (validación cruzada repetida con los cinco pliegues de entrenamiento), de modo
que el número de corridas por modelo es $5 \times$ el número de semillas.

| Elemento | Valor |
|---|---|
| $R$ que se toma como referencia | [COMPLETAR] |
| Número de semillas decidido | [COMPLETAR] |
| Semillas (`final.seeds`) | [COMPLETAR] |
| Pliegues (`final.folds`) | 0, 1, 2, 3, 4 |
| Corridas definitivas por modelo | [COMPLETAR] |
| Diferencia mínima detectable prevista con esas corridas | [COMPLETAR] |
| Justificación (potencia y presupuesto de cómputo) | [COMPLETAR] |

## 8. Desviaciones respecto del preregistro

[COMPLETAR, si las hay, con fecha y motivo]
