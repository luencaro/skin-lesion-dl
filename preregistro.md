# Preregistro

Plantilla creada el 2026-11-25. Se completa con los resultados del ajuste
y se fija, con fecha y commit, antes de ejecutar `python -m src.final`. Después de esa fecha no
se modifica; cualquier desviación se documenta en la sección final.

| Campo | Valor |
|---|---|
| Fecha de registro | 2026-11-26 |
| Commit del código | el del tag `e2-preregistro` |
| Autor | Luis Cabarcas |

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
| Archivos de datos verificados con `python -m src.verify_data` | sí |

## 3. Hiperparámetros seleccionados por modelo

Mejor ensayo de `outputs/tuning/<modelo>/best.json` (búsqueda aleatoria de 6 configuraciones, desviación D1,
con `sampler_seed` 2026, semilla 0 y pliegue de validación 0), copiado en la sección
`hyperparameters` de `configs/<modelo>.yaml`.

| Modelo | Ensayo | Tasa de aprendizaje (fase B) | Decaimiento de pesos | Dropout | Exactitud balanceada de validación | Época restaurada |
|---|---|---|---|---|---|---|
| MobileNetV2 | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |
| ResNet-50 | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |
| VGG-16 | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |

## 4. Resultados del piloto por modelo

El diseño original preveía la exactitud balanceada de validación de las cinco corridas de
`train.pilot.pairs`, sin test.

No aplica: el estudio piloto se omitió (desviación D2, capítulo 7).

## 5. Variabilidad de las diferencias pareadas y tamaño muestral

El diseño original preveía calcular, con las diferencias pareadas (misma semilla y mismo pliegue) de
la exactitud balanceada de validación del piloto, $\alpha = 0{,}05$, potencia $1 - \beta = 0{,}80$ y
$\delta = 0{,}02$, el número de corridas $R = (z_{1-\alpha/2} + z_{1-\beta})^2 \hat{\sigma}_d^2 / \delta^2$.
Sin piloto, $R = 5$ se fija por cómputo (desviación D3) y la diferencia mínima detectable se calcula
a posteriori con las corridas definitivas (capítulo 7, «Número de corridas»; sección 2 de
`book/11-benchmark.ipynb`).

No aplica: el estudio piloto se omitió (desviación D2, capítulo 7).

## 6. Supuestos y alternativa no paramétrica

El diseño original preveía comprobar con el piloto la normalidad de las diferencias pareadas
(Shapiro-Wilk).

No aplica: el estudio piloto se omitió (desviación D2, capítulo 7).

Criterio para la ejecución definitiva, aplicado de forma automática en la sección 6.1 de
`book/11-benchmark.ipynb` sobre las diferencias de la ejecución definitiva: si Shapiro-Wilk no
rechaza la normalidad en ningún par ($p \geq \alpha$), las pruebas principales son el ANOVA de
medidas repetidas y la t pareada; en otro caso, la alternativa no paramétrica, Friedman con
post-hoc de Nemenyi y Wilcoxon de rangos con signo exacto. Las dos familias se reportan siempre,
con corrección de Holm en las comparaciones pareadas.

No aplica: el estudio piloto se omitió (desviación D2, capítulo 7).

## 7. Número de corridas definitivas

Las corridas definitivas son el producto semillas x pliegues de la sección `final` de
`configs/base.yaml` (validación cruzada repetida con los cinco pliegues de entrenamiento), de modo
que el número de corridas por modelo es $5 \times$ el número de semillas.

| Elemento | Valor |
|---|---|
| $R$ que se toma como referencia | 5, fijado por cómputo sin piloto (desviación D3) |
| Número de semillas decidido | 1 |
| Semillas (`final.seeds`) | [0] |
| Pliegues (`final.folds`) | 0, 1, 2, 3, 4 |
| Corridas definitivas por modelo | 5 (semilla 0 en los pliegues 0 a 4) |
| Diferencia mínima detectable prevista con esas corridas | Se calcula a posteriori con la desviación estándar de las diferencias pareadas de las corridas definitivas (capítulo 7, «Número de corridas»). |
| Justificación (potencia y presupuesto de cómputo) | En una GPU RTX 3050 Laptop de 6 GB, el protocolo reducido requiere entre 28 y 36 horas de GPU, y el original (12 configuraciones y 25 corridas por modelo) requeriría 95 horas: 25 corridas por modelo exceden el tiempo disponible. Se prioriza evaluar todos los modelos con el mismo protocolo reducido. |

## 8. Desviaciones respecto del preregistro

| Fecha | Desviación |
|---|---|
| 2026-11-26 | D1: Ajuste de hiperparámetros con 6 configuraciones en lugar de 12: las 6 primeras del muestreador con semilla 2026 (t00 a t05), iguales para todos los modelos |
| 2026-11-26 | D2: Se omite el estudio piloto |
| 2026-11-26 | D3: Ejecución definitiva de 5 corridas por modelo: semilla 0 en cada uno de los 5 pliegues de validación, en lugar de 5 × 5 = 25 (E1), al menos 6 (E2) y al menos 10 semillas (E3) |
