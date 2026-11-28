# skin-lesion-dl

Clasificación de imágenes dermatoscópicas del conjunto HAM10000 con TensorFlow/Keras.
El proyecto compara ocho arquitecturas preentrenadas y dos baselines con una partición agrupada
por lesión y pruebas estadísticas, y propone un modelo original: una **capa de constancia de color
invariante a las ganancias de canal (CCI)** que se antepone a cualquier backbone
(`src/original_model.py`, capítulos 23 a 25 del libro). El informe se publica como libro con
MyST, y los resultados se exploran en un dashboard de Dash (`dashboard/`).

> El dashboard y los modelos son una herramienta académica: **no son una herramienta
> diagnóstica** (ver `MODEL_CARD.md`).

## Requisitos

- [uv](https://docs.astral.sh/uv/). Si no tienes Python 3.11, uv lo descarga solo.
- Opcional: una GPU NVIDIA con su driver instalado, en Linux o WSL2. En Linux,
  `tensorflow[and-cuda]` instala las librerías de CUDA y cuDNN, así que no hace falta
  instalar el CUDA Toolkit aparte. En Windows nativo, TensorFlow no usa la GPU.
- Para construir el libro: Node.js y [MyST](https://mystmd.org) (`npm install -g mystmd`).

## Crear el entorno

```bash
uv sync
```

Este comando crea `.venv/` con las versiones exactas de `uv.lock`.

Para registrar el kernel de Jupyter (solo hace falta una vez por máquina):

```bash
uv run python -m ipykernel install --user --name ham10000 --display-name "ham10000"
```

## Seleccionar el kernel "ham10000"

- **JupyterLab:** ejecuta `uv run jupyter lab`, abre un notebook y elige
  *Kernel → Change Kernel… → ham10000*.
- **VS Code:** abre el notebook, haz clic en *Select Kernel* (arriba a la derecha) →
  *Jupyter Kernel…* → `ham10000`. Otra opción es elegir directamente el intérprete
  `.venv/bin/python`.

Para comprobar que TensorFlow ve la GPU:

```bash
uv run python -c "from src.utils import list_gpus; print(list_gpus())"
```

## Obtención de los datos

Se usa la versión 4 de HAM10000 en Harvard Dataverse
([doi:10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T)). La descarga es manual: abre el
enlace, comprueba que la versión seleccionada sea la 4 y descarga estos archivos en `data/raw/`:

| Archivo | Contenido |
|---|---|
| `HAM10000_images_part_1.zip` | 5000 imágenes JPEG |
| `HAM10000_images_part_2.zip` | 5015 imágenes JPEG |
| `HAM10000_metadata` | metadatos de las 10015 imágenes |
| `HAM10000_segmentations_lesion_tschandl.zip` | máscaras de segmentación de la lesión |

Dataverse muestra los metadatos como una tabla ingerida: descárgalos en su formato original
(CSV) y guárdalos como `data/raw/HAM10000_metadata.csv`. Otro formato, como `.tab`, tiene otro
SHA-256 y la verificación fallará. Los archivos del conjunto de prueba de ISIC 2018 que se
distribuyen en la misma versión no se usan.

Después, descomprime las imágenes y las máscaras donde las espera `src/data.py` (rutas
`images_dir` y `segmentations_dir` de `configs/base.yaml`), sin subcarpetas:

```bash
unzip -q data/raw/HAM10000_images_part_1.zip -d data/raw/images
unzip -q data/raw/HAM10000_images_part_2.zip -d data/raw/images
unzip -q -j data/raw/HAM10000_segmentations_lesion_tschandl.zip \
    "HAM10000_segmentations_lesion_tschandl/*" -d data/raw/segmentations
```

El resultado es:

```
data/raw/
  HAM10000_images_part_1.zip
  HAM10000_images_part_2.zip
  HAM10000_metadata.csv
  HAM10000_segmentations_lesion_tschandl.zip
  images/ISIC_0024306.jpg, ...                       (10015 archivos)
  segmentations/ISIC_0024306_segmentation.png, ...   (10015 archivos)
```

Los datos se distribuyen bajo la licencia Creative Commons Atribución-NoComercial 4.0
(CC BY-NC 4.0), que no permite su uso ni su redistribución con fines comerciales. Por eso este
repositorio no los incluye (`.gitignore` excluye todo `data/raw/` salvo `.gitkeep`) y cada
usuario debe descargarlos de la fuente original.

### Verificación

```bash
uv run python -m src.verify_data
```

Comprueba que existan los archivos registrados en `data/hashes.json` y que su SHA-256 coincida,
y que estén las 10015 imágenes y máscaras descomprimidas. Termina con código 1 si algo falla.

### Conjuntos externos (evaluación externa)

PAD-UFES-20, Derm7pt y BCN20000 se usan solo para evaluar las corridas definitivas, sin reentrenar
(desviación D5; `experiments/protocolo.md`, sección 5). La descarga es manual y cada conjunto va en
su carpeta de `data/external/`, que `.gitignore` excluye. Las rutas, los nombres de archivo y de
columna, el mapeo de clases y las clases excluidas están en `configs/external.yaml`.

| Conjunto | Fuente y licencia | Carpeta | Contenido esperado |
|---|---|---|---|
| PAD-UFES-20 | Mendeley Data, [doi:10.17632/zr7vgbcyr2.1](https://doi.org/10.17632/zr7vgbcyr2.1); CC BY 4.0 | `data/external/pad_ufes_20/` | `metadata.csv` (columnas `img_id`, `diagnostic` y `lesion_id`) y las imágenes PNG, descomprimidas en cualquier subcarpeta (p. ej. `imgs_part_1/` a `imgs_part_3/`) |
| Derm7pt | [derm.cs.sfu.ca](https://derm.cs.sfu.ca), con registro; uso para investigación (verifica la licencia al descargar) | `data/external/derm7pt/` | `meta/meta.csv` (columnas `case_num`, `diagnosis` y `derm`) y la carpeta `images/` con las rutas de la columna `derm` (imagen dermatoscópica) |
| BCN20000 | Figshare, [doi:10.6084/m9.figshare.24140028](https://doi.org/10.6084/m9.figshare.24140028); CC BY 4.0 | `data/external/bcn20000/` | El archivo de metadatos de la partición de entrenamiento (`bcn_20k_train.csv`, columnas `bcn_filename`, `diagnosis` y `lesion_id`) y sus imágenes, en cualquier subcarpeta |

Descomprime cada descarga de modo que el archivo de metadatos quede en la ruta indicada, relativa a
la carpeta del conjunto. Si el archivo o sus columnas tienen otro nombre, `src.external` termina con
un error que lista lo que encontró: corrige el nombre en `configs/external.yaml`, sin cambiar el
mapeo. Después de descargar:

```bash
uv run python -m src.external --hashes        # registra los SHA-256 en data/external_hashes.json y muestra la composición
uv run python -m src.external --all           # predice con todas las corridas definitivas terminadas
```

La primera ejecución registra los SHA-256 de todos los archivos de cada carpeta; las siguientes
los comprueban y terminan con un error si algún archivo cambió. Las predicciones por corrida quedan
en `outputs/external/<modelo>/<conjunto>/`.

## Ejecución de extremo a extremo

Los pasos van en este orden. Todos los comandos se ejecutan desde la raíz del proyecto. Los
entrenamientos son reanudables: si se interrumpen, al relanzar el mismo comando se reutilizan
las corridas terminadas. Cada uno registra su progreso y el tiempo restante estimado en
`outputs/logs/`.

1. **Entorno** (sección «Crear el entorno»):

   ```bash
   uv sync
   ```

2. **Datos**: descarga y descompresión (sección anterior) y verificación:

   ```bash
   uv run python -m src.verify_data
   ```

3. **Análisis exploratorio y partición**. El notebook 05 genera o valida `data/splits.csv`; el
   06 genera `data/atipicos_imagenes.csv` y `data/train_color_stats.json`:

   ```bash
   uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 book/05-eda_metadatos_particion.ipynb
   uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 book/06-eda_imagenes.ipynb
   ```

4. **Ajuste de hiperparámetros** (pliegue 0, sin test): 6 configuraciones por modelo (desviación
   D1), cada una en su propio proceso. Se reanuda solo: las configuraciones terminadas se
   reutilizan (MobileNetV2 ya tiene t00 a t05; su t06 se conserva pero se excluye de la selección).

   ```bash
   for m in mobilenetv2 resnet50 vgg16 densenet201 inceptionv3 convnext_tiny vit_b16 efficientnetb0; do
       uv run python -m src.tune --model "$m"
   done
   ```

5. **Hiperparámetros de la ejecución definitiva**: si la sección `hyperparameters` de
   `configs/<modelo>.yaml` tiene los tres valores (`learning_rate`, `weight_decay`, `dropout`),
   `src.final` usa esos. Si están todos en null, los lee de `outputs/tuning/<modelo>/best.json`,
   que solo se acepta si se seleccionó entre t00 y t05 y coincide con el muestreador; si no existe,
   `src.final` termina con un mensaje que indica ejecutar `src.tune`. El origen (YAML o best.json, con
   el ensayo) queda en `run.hyperparameters_source` del `config.yaml` de cada corrida. Unos valores
   en null y otros no es un error. Antes de la ejecución definitiva, completa `preregistro.md` y crea
   el tag `e2-preregistro`.

6. **Ejecución definitiva** (5 corridas por modelo: semilla 0 en los pliegues 0 a 4, desviación
   D3; con test, métricas de train sin aumento y pesos guardados), para cada modelo:

   ```bash
   for m in mobilenetv2 resnet50 vgg16 densenet201 inceptionv3 convnext_tiny vit_b16 efficientnetb0; do
       uv run python -m src.final --model "$m"
   done
   ```

   El modelo original sigue los mismos dos pasos (`src.tune` y `src.final` con `--model cci`) después
   de fijar su backbone (sección «Modelo original»).

7. **Notebooks de resultados** por modelo y benchmark:

   ```bash
   uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 book/08-mobilenetv2.ipynb
   uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 book/09-resnet50.ipynb
   uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 book/10-vgg16.ipynb
   uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 book/11-benchmark.ipynb
   ```

   Con los flags `RERUN_*` en `False` (valor por defecto), los notebooks no entrenan: solo cargan
   los resultados guardados y muestran un aviso en las secciones que no tienen resultados.

8. **Libro**: MyST no ejecuta los notebooks, usa las salidas guardadas en el paso anterior.
   Los capítulos que se publican son los de la tabla de contenidos de `myst.yml`.

   ```bash
   myst build --html     # genera _build/html
   myst start            # o sirve el libro en local
   ```

## Modelo original (capa CCI)

La capa implementa las ecuaciones `cci-guia`, `cci-pesos`, `cci-iluminante` y `cci-correccion` de
`book/24-modelo-original-diseno.md` como subclase de `keras.layers.Layer`, con el iluminante en
dominio logarítmico, `p = 1 + softplus(ϑ)` inicializado en 6 y la red de pesos inicializada en
pesos uniformes (empieza como Shades of Gray). Sus parámetros están en `configs/cci.yaml`.

```python
from src.original_model import build_cci_model, CCI_NAME
model = build_cci_model("convnext_tiny", "full")          # cualquier modelo del REGISTRY
info = model.get_layer(CCI_NAME).explain(images)          # p, pesos espaciales e iluminante
```

Se registra como `cci` en el REGISTRY y se entrena con el mismo pipeline que los demás modelos.
Antes hay que fijar `backbone` en `configs/cci.yaml` con la regla del capítulo 24 (el modelo
del benchmark con mayor exactitud balanceada media en validación en el Entregable 2):

```bash
uv run python -m src.tune  --model cci
uv run python -m src.final --model cci
```

Las variantes de ablación del capítulo 24 son `cci_fixed_p`, `cci_uniform_weights`,
`cci_no_invariance` y `cci_shades_of_gray` (`configs/cci_*.yaml`, que heredan de `cci.yaml`).

## Pruebas

```bash
uv sync --group dashboard            # pytest está en el grupo dev (por defecto); dash, en dashboard
uv run pytest                        # solo CPU, con un backbone diminuto (no descarga pesos)
uv run pytest --junitxml outputs/tests/junit.xml   # el dashboard muestra este resultado
```

Las pruebas cubren la capa CCI (sobreajuste de un lote sintético, reproducibilidad, formas y
errores de entrada, gradientes, invarianza a ganancias y su ausencia en la variante sin
invarianza, gradiente respecto de p frente a diferencias finitas, guardado, carga y ONNX), la
exportación (CAM, FP16 con TOST, latencia) y el dashboard (coincidencia con la tabla maestra y
prueba de humo). `.github/workflows/tests.yml` las ejecuta en cada push.

## Exportación a ONNX

```bash
uv run python -m src.export --model cci            # corrida export.run (s0_f0) de la etiqueta final
uv run python -m src.export --all --fp16           # todos los modelos con model.weights.h5
```

Cada modelo queda en `artifacts/<modelo>/`: `model.onnx` (salidas `logits`, `probabilities`,
`embedding` y, con pooling global, `feature_map` y `head_weights` para el CAM sin gradientes),
`model_fp16.onnx` opcional (se usa solo si el TOST frente a δ muestra equivalencia),
`ood_reference.npz` y `metadata.json` (versiones, semilla, hiperparámetros, preprocesamiento,
esquema de entrada, temperatura, cuantiles conformes, métricas, prueba de ida y vuelta,
latencia en CPU, memoria pico y tamaño). Los modelos que superan `export.max_size_mb`
(`configs/export.yaml`) se marcan `live: false` y el dashboard los muestra con resultados
precalculados.

## Dashboard

```bash
uv run python -m src.export_dashboard              # genera dashboard/data/ (EDA, tabla maestra, ONNX)
cd dashboard && uv run python app.py               # http://127.0.0.1:8050
cd dashboard && uv run python smoke_test.py        # prueba de humo: cada página y una predicción por modelo
```

O con un entorno solo de inferencia (el mismo de Render):

```bash
python -m venv .venv-dash && . .venv-dash/bin/activate
pip install -r dashboard/requirements.txt
cd dashboard && gunicorn app:server --workers 1 --threads 2 --timeout 120
```

Cromia: cuatro páginas (Inicio, Demo en vivo, Modelos y Explora los datos), con modo claro y oscuro.
Mientras no haya resultados de entrenamiento, Explora los datos funciona completo y las cifras de
rendimiento se ven como marcadores de dato pendiente. Las imágenes cargadas se procesan en memoria
y no se guardan ni se registran. Detalles en `dashboard/README.md`.

## Despliegue

- **Render**: `render.yaml` define el servicio web (`rootDir: dashboard`,
  `gunicorn app:server --workers 1 --threads 2 --timeout 120`, `PYTHON_VERSION` fija). Antes de
  desplegar, regenera y versiona `dashboard/data/` con `python -m src.export_dashboard`.
  Después: `python dashboard/smoke_test.py --url https://<servicio>.onrender.com`.
- **Entrenamiento reproducible**: `Dockerfile` (Python 3.11.18, TensorFlow 2.20 con CUDA desde
  pip, `uv.lock`); ver las instrucciones al inicio del archivo.
- **Dependencias**: `uv.lock` es la fuente de verdad; `requirements-train.txt` se exporta de él
  (`uv export --format requirements-txt --frozen --no-hashes --no-emit-project --no-dev -o requirements-train.txt`)
  y `requirements.txt` / `dashboard/requirements.txt` contienen solo las de inferencia.

## Estructura

```
configs/        configuración común (base.yaml), de cada modelo, del modelo original (cci*.yaml) y de exportación
src/            entrenamiento: datos, modelos, entrenamiento, evaluación, estadística, modelo original y exportación
dashboard/      aplicación Dash autocontenida (app.py, pages/, core/, assets/, data/, requirements.txt)
tests/          pruebas unitarias (pytest, CPU)
experiments/    estándar, protocolo, bitácora de diseño y procedencia del mecanismo
artifacts/      modelos exportados a ONNX y metadata.json (los .onnx no se versionan)
data/raw/       HAM10000 (no se versiona)
data/literatura/  exportaciones de Scopus y tablas de síntesis
outputs/        métricas, predicciones y pesos de cada corrida, ajuste y logs
book/           capítulos del informe (MyST)
MODEL_CARD.md, DATA_STATEMENT.md, CITATION.cff, LICENSE, render.yaml, Dockerfile
preregistro.md, preregistro_e2.md
```

**Desviación declarada respecto de la estructura sugerida en el Entregable 3.** El entrenamiento
vive en `src/` (no en `training/`) y el dashboard en `dashboard/` (no en la raíz con `pages/` y
`src/inference.py`). El código de entrenamiento ya existía en `src/` desde el Entregable 1 y lo usan
todos los notebooks del libro; moverlo rompería esas importaciones. La separación que pide el
enunciado se mantiene: el dashboard no importa TensorFlow ni `src/` (lo comprueba una prueba), solo
lee `dashboard/data/`, y su lógica está dividida en `dashboard/pages/` (interfaz),
`dashboard/core/data.py` (datos), `dashboard/core/inference.py` (ONNX) y `dashboard/core/theme.py`
(figuras). El `requirements.txt` de la raíz remite al de `dashboard/`.

## Licencia y cita

El código se distribuye bajo la licencia MIT (`LICENSE`). Los datos de HAM10000 tienen su propia
licencia, CC BY-NC 4.0, y no se incluyen en el repositorio (`DATA_STATEMENT.md`). Para citar el
repositorio, ver `CITATION.cff`.
