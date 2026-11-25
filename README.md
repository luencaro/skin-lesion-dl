# skin-lesion-dl

Clasificación de imágenes dermatoscópicas del conjunto HAM10000 con TensorFlow/Keras.
El proyecto compara arquitecturas preentrenadas (ResNet50, MobileNetV2 y VGG16) con una
partición fija y pruebas estadísticas. El informe se publica como libro con MyST.

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

4. **Ajuste de hiperparámetros** (pliegue 0, sin test), para cada modelo:

   ```bash
   uv run python -m src.tune --model mobilenetv2
   uv run python -m src.tune --model resnet50
   uv run python -m src.tune --model vgg16
   ```

5. **Copia de hiperparámetros**: copia `hyperparameters` de `outputs/tuning/<modelo>/best.json`
   en la sección `hyperparameters` de `configs/<modelo>.yaml` (`learning_rate`, `weight_decay`,
   `dropout`). Los pasos siguientes fallan con un mensaje claro si alguno sigue en null.

6. **Estudio piloto** (cinco corridas por modelo, sin test):

   ```bash
   uv run python -m src.pilot --model mobilenetv2
   uv run python -m src.pilot --model resnet50
   uv run python -m src.pilot --model vgg16
   ```

7. **Preregistro**: ejecuta el notebook del benchmark, cuya sección 2 calcula con el piloto
   $\hat{\sigma}_d$, $\delta / \hat{\sigma}_d$, el número de corridas $R$ y la diferencia mínima
   detectable (las secciones que necesitan la ejecución definitiva muestran un aviso):

   ```bash
   uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 book/11-benchmark.ipynb
   ```

   Completa `preregistro.md` (marcadores `[COMPLETAR]`), ajusta `final.seeds` en
   `configs/base.yaml` al número de semillas decidido y registra la fecha y el commit.

8. **Ejecución definitiva** (semillas x pliegues, con test, métricas de train sin aumento y
   pesos guardados), para cada modelo:

   ```bash
   uv run python -m src.final --model mobilenetv2
   uv run python -m src.final --model resnet50
   uv run python -m src.final --model vgg16
   ```

9. **Notebooks de resultados** por modelo y benchmark:

   ```bash
   uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 book/08-mobilenetv2.ipynb
   uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 book/09-resnet50.ipynb
   uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 book/10-vgg16.ipynb
   uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 book/11-benchmark.ipynb
   ```

   Con los flags `RERUN_*` en `False` (valor por defecto), los notebooks no entrenan: solo cargan
   los resultados guardados y muestran un aviso en las secciones que no tienen resultados.

10. **Libro**: MyST no ejecuta los notebooks, usa las salidas guardadas en el paso anterior.
    Los capítulos que se publican son los de la tabla de contenidos de `myst.yml`.

    ```bash
    myst build --html     # genera _build/html
    myst start            # o sirve el libro en local
    ```

## Estructura

```
configs/      configuración común (base.yaml) y de cada modelo
src/          código: datos, modelos, entrenamiento, evaluación, estadística y utilidades
data/raw/     HAM10000 (no se versiona)
data/literatura/  exportaciones de Scopus y tablas de síntesis
outputs/      métricas, predicciones y pesos de cada corrida, ajuste, piloto y logs
book/         capítulos del informe (MyST)
preregistro.md
```
