---
title: Reproducibilidad
---

Este capítulo resume cómo reproducir el Entregable 2 desde cero y qué elementos de la lista de
reproducibilidad del protocolo (Apéndice B) se cumplen. Todos los parámetros están en `configs/`, la
lógica en `src/` y los notebooks solo llaman funciones y muestran resultados.

## Entorno

| Elemento | Archivo o comando |
|---|---|
| Versiones exactas de las dependencias | `uv.lock` y `requirements.txt` (exportado con `uv export --format requirements-txt --frozen --no-hashes --no-emit-project --no-dev`) |
| Python | 3.11 (`.python-version`) |
| Contenedor | `Dockerfile` (Python 3.11, `uv sync --frozen`, TensorFlow 2.20 con CUDA desde pip); datos y resultados como volúmenes |
| Hardware de referencia | GPU NVIDIA GeForce RTX 3050 de 6 GB; versiones de CUDA, cuDNN y GPU registradas en `run_info.json` de cada corrida |
| Verificación de los datos | `python -m src.verify_data` (SHA-256 frente a `data/hashes.json`) |
| Verificación de los modelos | `python -m src.verify_models` (parámetros frente a `expected_params` y estructura del pipeline) |

## Comandos en orden

Cada paso reutiliza lo ya terminado, así que puede interrumpirse y reanudarse.

1. `uv sync` y `python -m src.verify_data`.
2. `python -m src.verify_models`.
3. Para cada modelo de `e2.sota_models` (y de `e2.base_models` si faltan sus corridas):
   `python -m src.tune --model <modelo>`, copiar `outputs/tuning/<modelo>/best.json` en
   `configs/<modelo>.yaml`, `python -m src.pilot --model <modelo>` y `python -m src.final --model <modelo>`.
4. `python -m src.baselines --compute-embeddings` (embeddings de test, ajuste de C y corridas de los
   dos baselines).
5. `python -m src.cost --all`, `python -m src.robustness --all` y `python -m src.interpret`.
6. `python -m src.tracking` para registrar en MLflow las corridas anteriores a su activación.
7. Notebooks `book/15` a `book/21`, y `python -m src.reproduce --glmm` para regenerar todas las
   tablas y figuras en `outputs/reproduce/` (con `indice.csv`).

## Seguimiento de experimentos

Con `mlflow.enabled: true`, `src.train` y `src.baselines` registran cada corrida al terminar en un
MLflow local (`outputs/mlflow/mlflow.db`, artefactos en `outputs/mlflow/artifacts`): parámetros
(modelo, semilla, pliegue, hiperparámetros y ajustes de entrenamiento), métricas por partición y por
clase, curva por época, tiempos y artefactos (`config.yaml`, `metrics.json`, `run_info.json`,
`history.csv` y probabilidades). Un fallo de MLflow solo emite un aviso. Interfaz:
`uv run mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlflow.db`.

## Lista de verificación (Apéndice B)

Estado: **cumple** si está implementado y queda registrado al ejecutar los comandos anteriores;
**parcial** si se cumple con una limitación declarada; **pendiente** si depende de un paso manual aún
no hecho; **desviación** si no se hace, con su motivo.

| Ítem | Estado | Evidencia |
|---|---|---|
| Código versionado y preregistro fechado antes de los entrenamientos | pendiente | repositorio Git y `preregistro_e2.md`; falta crear el tag `e2-preregistro` y completar sus campos antes del primer entrenamiento |
| Dependencias con versiones exactas | cumple | `uv.lock`, `requirements.txt` |
| Entorno reproducible en contenedor | cumple | `Dockerfile`, `.dockerignore` |
| Versión de los datos y hashes de archivos | cumple | `data/hashes.json`, `src.verify_data` |
| Partición congelada y agrupada por lesión, con hash del conjunto de prueba | cumple | `data/splits.csv`, `data/splits_hash.json` |
| Conjunto de prueba oficial de ISIC 2018 sellado | cumple | `data/raw/isic2018_test/` no se lee en ningún módulo |
| Auditoría de fuga con criterios de aprobación | cumple | `src/audit.py`, `book/21-auditoria-fuga.ipynb` |
| Contrato de datos, diccionario y conteo por población | cumple | `configs/data_contract.yaml`, `data/diccionario.csv`, `src.data.load_metadata_validated` |
| Semillas fijadas y registradas por corrida | cumple | `seed`, `train.tuning.sampler_seed`, `final.seeds`; `config.yaml` de cada corrida |
| Determinismo de las operaciones en GPU | parcial | `train.deterministic: false` por costo; la variabilidad entre semillas se mide y se reporta |
| Mismo pipeline, resolución, espacio de búsqueda y presupuesto para todos los modelos | cumple | `configs/base.yaml` (`train`), `src/tune.py` |
| Hiperparámetros seleccionados solo con validación | cumple | `outputs/tuning/<modelo>/best.json`; test solo en `src.final` |
| Número de corridas justificado con un piloto y la potencia | pendiente | `src.pilot` y `book/11-benchmark.ipynb` (sección 2); R se completa en `preregistro_e2.md` tras el piloto |
| Regla de exclusión de corridas divergentes | cumple | `e2.divergence`, `benchmark.divergent_runs` |
| Baselines trivial y no profundo con las mismas corridas | cumple | `src/baselines.py` |
| Métricas primaria y secundarias con variabilidad e intervalos | cumple | tabla principal de `book/20-benchmark-e2.ipynb` (t de Student y bootstrap por lesión) |
| Pruebas preregistradas con corrección por comparaciones múltiples | cumple | `benchmark.contrast_family` (Holm por familia), familias exploratorias con Benjamini-Hochberg |
| Equivalencia y tamaños de efecto | cumple | TOST, d de Cohen, delta de Cliff, A12 con IC y probabilidad de superioridad |
| Supuestos verificados | cumple | Shapiro-Wilk de las diferencias pareadas; regla de decisión según la normalidad |
| Calibración e incertidumbre | cumple | `src/calibration.py` (temperature scaling, ECE, Brier, deep ensemble, predicción conforme) |
| Robustez frente a perturbaciones | cumple | `src/robustness.py` |
| Interpretabilidad con pruebas de fidelidad y estabilidad | cumple | `src/interpret.py` (borrado e inserción, correlación entre semillas) |
| Análisis de errores, subgrupos y atajos | parcial | `src/errors.py`; ninguna imagen anotada de artefactos está en test, así que los atajos se miden en validación fuera de pliegue |
| Costo computacional (parámetros, FLOPs, memoria, latencia, tiempos) | cumple | `src/cost.py`, `cost.cost_from_runs` |
| Registro de configuración, métricas y artefactos de cada corrida | cumple | carpeta de cada corrida y MLflow local (`src/tracking.py`) |
| Pesos de los modelos guardados | parcial | `model.weights.h5` en cada corrida definitiva; fuera de Git por tamaño (`.gitignore`) |
| Regeneración de todas las tablas y figuras desde los resultados guardados | cumple | `python -m src.reproduce` |
| Réplica de un resultado publicado con criterio previo | cumple | `benchmark.replication_table`, [capítulo 14](14-paper-guia.md) |
| Importancia por permutación con reentrenamiento | desviación | no se hace por cómputo: cada reentrenamiento cuesta una corrida completa por variable |
| Curvas de aprendizaje con distintos tamaños de entrenamiento | desviación | no se hacen por cómputo; se reportan las curvas por época de cada corrida |
| Corrección del optimismo por bootstrap y validación anidada | desviación | no se hacen por cómputo; el ajuste usa un pliegue fijo y la estimación final, test sellado y validación cruzada repetida |

## Desviaciones declaradas

- **Sin importancia por permutación con reentrenamiento**, **sin curvas de aprendizaje** y **sin
  corrección del optimismo por bootstrap ni validación anidada**, por cómputo: con una GPU de 6 GB, cada
  corrida de los modelos más costosos ocupa horas, y estas técnicas multiplican el número de
  entrenamientos.
- Cinco modelos del estado del arte en lugar de diez y búsqueda bibliográfica solo en Scopus, como
  declara `preregistro_e2.md`.
- Determinismo de GPU desactivado (`train.deterministic: false`), heredado del Entregable 1.
- El diagnóstico de atajos por artefactos se calcula en validación fuera de pliegue porque las 280
  imágenes anotadas de `data/anotacion_artefactos.csv` pertenecen a train.
