# Cromia: dashboard del proyecto

Aplicación Dash + Plotly + una hoja CSS (`assets/cromia.css`), sin frameworks JS adicionales. Solo
lee `dashboard/data/`, que genera el pipeline; no importa TensorFlow ni `src/`.

> Herramienta académica de demostración. No es un dispositivo médico ni sustituye el diagnóstico
> de un dermatólogo.

## Instalación y arranque

```bash
pip install -r dashboard/requirements.txt          # o: uv sync --group dashboard
cd dashboard && python app.py                       # http://127.0.0.1:8050
cd dashboard && gunicorn app:server --workers 1 --threads 2 --timeout 120   # como en Render
```

## De dónde salen los datos

Todo lo que se ve sale de `python -m src.export_dashboard` (desde la raíz). No hay cifras escritas a
mano: si un archivo o una clave falta, la interfaz muestra un marcador `.ph` (borde discontinuo
naranja) y no falla.

```bash
python -m src.final --model <modelo>         # corridas definitivas
python -m src.export --all --fp16            # ONNX + metadata.json + ood_reference.npz (artifacts/)
python -m src.export_dashboard               # copia y resume todo en dashboard/data/
```

| Dato en pantalla | Archivo (`dashboard/data/`) |
|---|---|
| Conteos de Explora los datos (clase, fuente, edad, sexo, localización) | `eda/images.json` (metadatos de HAM10000, una fila por imagen) |
| Color medio (Lab) por fuente | `eda/images.json` (medias RGB por imagen de train, notebook 06) |
| Galería por clase | `eda/summary.json` → `gallery` y `eda/gallery/*.jpg` |
| Exactitud balanceada, F1 macro, ECE, parámetros, GFLOPs, latencia | `models/leaderboard.json` (tabla maestra y `outputs/cost`; F1 macro: `f1_media`) |
| Tamaño (MB) | `models/index.json` → `artifacts[].size_mb` (ONNX servido) |
| Matriz de confusión, diagrama de calibración | `models/<modelo>/diagnostics.json` |
| Nuestro modelo frente a su backbone | `models/original.json` (`backbone`) y las filas de ambos en la tabla |
| Variación entre fuentes | `models/original.json` → `source_spread` (aún no lo exporta el pipeline: marcador) |
| Demo: modelo, temperatura, conjunto conforme, umbral OOD | `models/<modelo>/metadata.json`, `model.onnx`, `ood_reference.npz` |
| Demo: modelos que superan el límite de tamaño | `models/<modelo>/precomputed.json` (solo los ejemplos) |
| Ejemplos de la demo | `examples/*.jpg` (imágenes de test) y `assets/samples/foto_no_dermatoscopica.jpg` |

La frontera de Pareto, la frase de la confusión más frecuente y los titulares de Explora los datos se
calculan en la app a partir de esos archivos.

## Inferencia (Demo en vivo)

- Imagen en memoria → RGB → 224 × 224 bilineal → float32 en [0, 255] → ONNX Runtime en CPU. La
  normalización de cada backbone y la capa CCI van dentro del grafo (`config.CCI_IN_GRAPH = True`).
- Probabilidades calibradas por temperatura; conjunto de predicción LAC con `1 − α = 90 %`.
- Mapa de atención: el ONNX de los modelos con cabeza GAP → [LayerNorm] → lineal expone
  `feature_map` y `head_weights` (`src/export.py`), y el CAM es `Σ_c A[..., c] W[c, k]` (equivale a
  Grad-CAM en esa arquitectura). ViT-B/16 no trae mapa: su attention rollout solo existe en el
  pipeline de TensorFlow (`src/interpret.py`). Paleta azul cielo → amarillo → naranja → bermellón.
- Fuera de distribución: distancia coseno media a los k vecinos de train en el espacio de
  embeddings frente al umbral fijado en validación. Sin `ood_reference.npz`, la detección queda
  desactivada y se avisa en el registro.
- La imagen cargada no se escribe en disco (lo comprueba `smoke_test.py`).

## Pruebas

```bash
cd dashboard && python smoke_test.py                         # en proceso
python dashboard/smoke_test.py --url https://<servicio>.onrender.com
pytest tests/test_dashboard.py                               # incluye modelos ONNX sintéticos (TensorFlow en CPU)
```

## Estructura

```
app.py              layout raíz, navbar, tema claro/oscuro (dcc.Store local) y rutas /media
core/config.py      rutas, límites, clases, colores Okabe-Ito, fuentes y modelos
core/theme.py       plantillas Plotly "cromia" y "cromia_dark"
core/components.py  aviso, marcadores, estados vacíos, tarjetas, formato es-CO
core/data.py        lectura de dashboard/data/ y cifras derivadas
core/inference.py   ONNX, CAM, conjunto conforme, OOD y mapa de calor
pages/              inicio, demo, modelos, datos (y 404)
assets/             cromia.css, clientside.js (callbacks de cliente) y samples/
```
