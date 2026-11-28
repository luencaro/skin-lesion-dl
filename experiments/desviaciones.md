# Desviaciones por cómputo (D1 a D5)

Decididas el 26 de noviembre de 2026, antes de cualquier corrida definitiva y sin haber evaluado el conjunto de prueba. Este archivo contiene los textos exactos que deben quedar en el repositorio. Cada bloque indica el archivo y el cambio. Los textos entre comillas angulares `«…»` se copian literalmente.

## Resumen de las desviaciones

| Id | Desviación | Afecta a |
|---|---|---|
| D1 | Ajuste de hiperparámetros con 6 configuraciones en lugar de 12: las 6 primeras del muestreador con semilla 2026 (t00 a t05), iguales para todos los modelos | E1, E2, E3 |
| D2 | Se omite el estudio piloto | E1, E2, E3 |
| D3 | Ejecución definitiva de 5 corridas por modelo: semilla 0 en cada uno de los 5 pliegues de validación, en lugar de 5 × 5 = 25 (E1), al menos 6 (E2) y al menos 10 semillas (E3) | E1, E2, E3 |
| D4 | Las variantes de ablación del modelo original no se ajustan por separado: usan los hiperparámetros del modelo completo | E3 |
| D5 | Los tres conjuntos adicionales se usan solo para evaluación externa de los modelos entrenados en HAM10000, sin reentrenar | E3 |

## 1. `configs/base.yaml`

- `train.tuning.n_trials: 6`, con el comentario `# D1: 6 configuraciones (antes 12); las 6 primeras del muestreador con sampler_seed 2026`.
- `final.seeds: [0]` y `final.folds: [0, 1, 2, 3, 4]`, con el comentario `# D3: 5 corridas por modelo (semilla 0 x 5 pliegues), fijadas por cómputo sin piloto`.
- Sobre `train.pilot`, el comentario `# D2: el estudio piloto se omite; la sección se conserva solo por compatibilidad del código`.
- Sustituir el comentario de `final` «El número de semillas se ajusta tras el piloto editando `seeds` (ver preregistro.md).» por «El número de corridas se fijó por cómputo (desviación D3, capítulo 7).».

## 2. `book/07-diseno-experimental.md`

### 2.1 Sección «División del conjunto y esquema de validación», párrafo que empieza «Los tres modelos se ejecutan…»

Sustituir «El conjunto de prueba no interviene en el ajuste de hiperparámetros ni en el estudio piloto:» por «El conjunto de prueba no interviene en el ajuste de hiperparámetros:».

### 2.2 Sección «Ajuste de hiperparámetros», primer párrafo

Sustituir las frases sobre las 12 configuraciones por:

«El método es una búsqueda aleatoria de 6 configuraciones, generadas con una semilla fija (2026). Como la semilla es común, todos los modelos evalúan exactamente las mismas 6 configuraciones, lo que elimina la posibilidad de que un modelo resulte favorecido por una muestra de configuraciones más afortunada. El diseño original preveía 12 configuraciones; se redujeron a las 6 primeras del mismo muestreador por el cómputo disponible, antes de ajustar cualquier modelo distinto de MobileNetV2, cuyas configuraciones séptima en adelante, ya evaluadas, se conservan en los registros pero se excluyen de la selección.»

### 2.3 Sección «Reproducibilidad»

Sustituir «que forman parte de la variabilidad entre ejecuciones que estima el estudio piloto.» por «que forman parte de la variabilidad entre corridas.».

### 2.4 Sustituir la sección completa «Estudio piloto» (título, ecuación y párrafos) por:

```markdown
## Número de corridas

El diseño original preveía un estudio piloto de cinco corridas por modelo para estimar la variabilidad entre corridas y, con ella, el número de corridas definitivas necesario para detectar $\delta = 0{,}02$ con potencia 0,80. Con el cómputo disponible, una GPU NVIDIA GeForce RTX 3050 Laptop de 6 GB en la que el protocolo reducido requiere entre 28 y 36 horas de GPU, y el original (12 configuraciones y 25 corridas por modelo) requeriría 95 horas, el piloto se omite y el número de corridas se fija de antemano en cinco por modelo: la semilla 0 en cada uno de los cinco pliegues de validación. La decisión se tomó el 26 de noviembre de 2026, antes de cualquier corrida definitiva y sin haber evaluado el conjunto de prueba, y se declara como desviación.

Como el número de corridas no se deriva de un análisis de potencia, la potencia se reporta a posteriori. Con la desviación estándar de las diferencias pareadas de la exactitud balanceada en prueba, $\hat{\sigma}_d$, se calcula la diferencia mínima detectable con $R = 5$ corridas,

$$
\mathrm{DMD} = \left(t_{1-\alpha/2,\,R-1} + t_{1-\beta,\,R-1}\right) \hat{\sigma}_d \sqrt{\frac{1}{R} + \frac{n_{\mathrm{val}}}{n_{\mathrm{train}}}},
$$

con $\alpha = 0{,}05$ y $1 - \beta = 0{,}80$, donde el término $n_{\mathrm{val}}/n_{\mathrm{train}} \approx 0{,}25$ es la corrección de Nadeau y Bengio por el solapamiento de los conjuntos de entrenamiento entre corridas. Si la diferencia mínima detectable supera $\delta$, la ausencia de significancia se interpreta como un resultado no concluyente y no como equivalencia.
```

### 2.5 Sección «Observaciones y variable de respuesta», primera frase

Sustituir «La ejecución definitiva sigue una validación cruzada repetida de cinco semillas por cinco pliegues, de modo que cada modelo produce 25 corridas, número que se revisará con el resultado del piloto.» por:

«La ejecución definitiva consta de cinco corridas por modelo, una por pliegue de validación con la semilla 0, de modo que cada pliegue actúa una vez como validación. La variabilidad entre corridas refleja, por tanto, la partición y el no determinismo de la GPU, y no se estima por separado el efecto de la semilla.»

### 2.6 Sección «Limitaciones inferenciales», sustituir el primer párrafo por:

«Comparar con cinco corridas por modelo impone límites a la potencia de las pruebas que deben reconocerse antes de interpretarlas. La prueba de Wilcoxon de rangos con signo no puede alcanzar significancia bilateral: su p-valor exacto mínimo es $2/2^5 = 0{,}0625$, mayor que $\alpha$, incluso si un modelo supera al otro en todas las corridas. En los contrastes unilaterales del Entregable 2 el mínimo es $1/2^5 \approx 0{,}031$, pero la corrección de Holm exige, en el paso más estricto, $p < 0{,}05/5 = 0{,}01$ en la familia de RQ1 y $p < 0{,}05/8 \approx 0{,}0063$ en la de RQ2, de modo que Wilcoxon tampoco puede rechazar allí. La prueba t corregida de Nadeau y Bengio, con cuatro grados de libertad, y el bootstrap por lesiones sobre el conjunto de prueba no tienen ese límite. En consecuencia, si Shapiro–Wilk rechaza la normalidad de las diferencias, la regla de decisión preregistrada, que exige que Wilcoxon y el bootstrap sean ambos significativos, no puede declarar superioridad, y el contraste se reporta como no concluyente; con cinco diferencias, además, Shapiro–Wilk tiene muy poca potencia para detectar la falta de normalidad.»

### 2.7 Resto del capítulo

Buscar «piloto», «25 corridas», «12 configuraciones» y «cinco semillas» en todo `book/*.md` y ajustar cada mención para que sea coherente con D1 a D3, sin cambiar nada más del texto.

## 3. `preregistro.md` (Entregable 1)

- **Fecha de registro:** la fecha en que se aplican estos cambios (AAAA-MM-DD).
- **Commit del código:** «el del tag `e2-preregistro`».
- **Autor:** «Luis Cabarcas».
- **Archivos de datos verificados con `python -m src.verify_data`:** ejecutar el comando y escribir «sí» o «no» según su salida.
- **Tablas del piloto (filas por modelo y por par de modelos, y «Observaciones sobre los supuestos»):** sustituir cada tabla por la frase «No aplica: el estudio piloto se omitió (desviación D2, capítulo 7).».
- **$R$ que se toma como referencia:** «5, fijado por cómputo sin piloto (desviación D3)».
- **Número de semillas decidido:** «1».
- **Semillas (`final.seeds`):** «[0]».
- **Corridas definitivas por modelo:** «5 (semilla 0 en los pliegues 0 a 4)».
- **Diferencia mínima detectable prevista con esas corridas:** «Se calcula a posteriori con la desviación estándar de las diferencias pareadas de las corridas definitivas (capítulo 7, «Número de corridas»).».
- **Justificación:** «En una GPU RTX 3050 Laptop de 6 GB, el protocolo reducido requiere entre 28 y 36 horas de GPU, y el original (12 configuraciones y 25 corridas por modelo) requeriría 95 horas: 25 corridas por modelo exceden el tiempo disponible. Se prioriza evaluar todos los modelos con el mismo protocolo reducido.».
- **Desviaciones:** una fila por cada una de D1, D2 y D3, con fecha 2026-11-26 y el texto de la tabla de resumen.

## 4. `preregistro_e2.md`

- **Fecha:** la misma que en `preregistro.md`.
- **Fila «Número de corridas R»:** «R = 5 por modelo (semilla 0 en los pliegues 0 a 4), fijado por cómputo antes de cualquier corrida definitiva y sin estudio piloto (desviaciones D2 y D3). Se reporta a posteriori la diferencia mínima detectable. Con 5 corridas, Wilcoxon no puede rechazar tras la corrección de Holm (capítulo 7, «Limitaciones inferenciales»).».
- **Fila «Reglas de parada del ajuste»:** cambiar «12 configuraciones» por «6 configuraciones (las 6 primeras del muestreador; desviación D1)».
- **Fila «Cómputo estimado»:** «C_total ≈ Σ_m (6 · t_ajuste,m + 5 · t_final,m). Horas estimadas por modelo en `experiments/presupuesto_computo.md`.».
- **Fila «Desviaciones declaradas»:** añadir al final «Ajuste con 6 configuraciones en lugar de 12 (D1), sin estudio piloto (D2) y con 5 corridas definitivas por modelo en lugar de al menos 6 (D3), por cómputo.».

## 5. Código y notebooks

- `src/tune.py`: la selección del mejor ajuste considera solo las configuraciones t00 a t(n_trials − 1). Las configuraciones con índice mayor que ya existan en `outputs/tuning/<modelo>/` (por ejemplo, t06 de MobileNetV2) se conservan, se marcan como excluidas en el resumen y no se borran.
- Comprobar que, con `n_trials: 6`, el muestreador genera exactamente las mismas configuraciones t00 a t05 que ya figuran en `outputs/tuning/mobilenetv2/`. Si no coinciden, detenerse e informar sin cambiar nada más.
- Notebooks de cada modelo (08, 09, 10, 15 a 19 y 25): en la sección del estudio piloto, sustituir el texto por «Estudio piloto omitido (desviación D2, capítulo 7).» y eliminar sus celdas de código.
- Notebooks 11 y 20: sustituir el piloto conjunto y la tabla de potencia por una tabla de diferencia mínima detectable a posteriori, calculada con las corridas definitivas mediante la fórmula del capítulo 7 (nueva función en `src/benchmark.py`, con test en `tests/`).
- No ejecutar ningún notebook.

## 6. Bitácora: memoria y velocidad del entrenamiento (2026-11-26)

Cambio de implementación, no del diseño experimental. Se hizo después de que MobileNetV2 completara 7 configuraciones del ajuste (t00 a t06, el 14 de noviembre de 2026) y antes de ajustar cualquier otro modelo y de cualquier corrida definitiva. No se ha evaluado el conjunto de prueba.

**Motivo.** El ajuste de MobileNetV2 murió dos veces por falta de RAM del sistema (OOM killer del kernel, 14-nov 12:06 y 13:41; 14,8 GB de RAM). Causas medidas en un mismo proceso con 3 configuraciones: la caché `tf.data` en float32 de train + val ocupaba 5,2 GB, y cada configuración dejaba ~1,1 GB más que `keras.backend.clear_session()` no libera, la memoria de las compilaciones de XLA (sin XLA el crecimiento era de 0,08 GB). La GPU no era el límite: el pico de MobileNetV2 era de 1,05 GB.

**Cambios.**

1. `train.cache: disk`: cada imagen redimensionada (bilineal con antialias a 224 × 224, la misma función dentro de `tf.data`) se calcula una vez y se guarda en float32 en `data/processed/resized_cache/`, con su SHA-256 en `manifest.json`, que se comprueba al cargar cada partición. Se barajan las rutas en lugar de las imágenes, con la misma semilla y el mismo tamaño de búfer, y la lectura y el aumento van en un solo `map`.
2. `train.isolate_runs: true`: cada corrida de `src.tune` y `src.final` se ejecuta en un proceso hijo, que devuelve al sistema la memoria de XLA al terminar.
3. `train.xla_autotune_cache: true`: el autotuning de XLA se guarda en `data/processed/xla_cache/` y lo reutilizan las corridas siguientes (compilación de la fase B de MobileNetV2: de 58 a 32 s).
4. `train.gpu_min_system_memory_mb: 512` y `train.gpu_allocator: cuda_malloc_async`: TensorFlow pasa de 4,2 a 5,2 GB de memoria de GPU, y el asignador de CUDA sustituye al BFC, que con VGG-16 y XLA abortaba de forma intermitente (`bfc_allocator.cc: Check failed`) al liberar memoria.

**Efecto en los valores numéricos.** Ninguno en las entradas ni en el orden: la caché en disco devuelve imágenes idénticas bit a bit al redimensionado anterior (300 imágenes reales con diferencia máxima 0,0; `tests/test_data_cache.py`), y la permutación del barajado es la misma (mismo test). Los cambios 2 a 4 no tocan ningún valor de la configuración experimental: semillas, hiperparámetros, épocas, lote efectivo, precisión mixta y XLA siguen iguales. El asignador y el autotuning solo eligen dónde se reserva la memoria y qué algoritmo equivalente de convolución se usa. Como antes, el entrenamiento en GPU no es determinista (capítulo 7, «Reproducibilidad»), así que repetir una corrida no da resultados idénticos bit a bit ni antes ni después del cambio. Las 7 configuraciones ya evaluadas de MobileNetV2 se conservan sin repetir.

**Verificación.** Tres configuraciones seguidas de MobileNetV2 en un mismo proceso padre, con la configuración final, las épocas del ajuste (5 de la fase A y 15 de la B) y salidas fuera de `outputs/`: pico de cada proceso hijo de 4,6, 3,9 y 4,0 GB, proceso padre en 0,67 GB constantes y al menos 4,3 GB de RAM disponibles en el sistema, frente a 9,4, 10,5 y 11,7 GB acumulados antes del cambio (con 2 épocas por fase). Cada configuración tardó entre 7,2 y 8,0 minutos, frente a unos 35 minutos el 14 de noviembre. Las mediciones de velocidad están en `outputs/speed/` (`python -m src.speed`) y el presupuesto de cómputo en `experiments/presupuesto_computo.md`.
