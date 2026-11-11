# skin-lesion-dl

Clasificación de imágenes dermatoscópicas del conjunto HAM10000 con TensorFlow/Keras.
El proyecto compara arquitecturas preentrenadas (ResNet50, MobileNetV2 y VGG16) con una
partición fija y pruebas estadísticas. El informe se publica como libro con MyST.

## Requisitos

- [uv](https://docs.astral.sh/uv/). Si no tienes Python 3.11, uv lo descarga solo.
- Opcional: una GPU NVIDIA con su driver instalado, en Linux o WSL2. En Linux,
  `tensorflow[and-cuda]` instala las librerías de CUDA y cuDNN, así que no hace falta
  instalar el CUDA Toolkit aparte. En Windows nativo, TensorFlow no usa la GPU.

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

## Datos

HAM10000 se descarga de Harvard Dataverse
([doi:10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T)) y se guarda en
`data/raw/`. Los datos no se versionan: `.gitignore` excluye todo el contenido de
`data/raw/` excepto `.gitkeep`.

## Estructura

```
configs/      configuración común (base.yaml) y de cada modelo
src/          código: datos, modelos, entrenamiento, estadística y utilidades
data/raw/     HAM10000 (no se versiona)
data/literatura/  exportaciones de Scopus y tablas de síntesis
outputs/      métricas y predicciones de cada corrida
book/         capítulos del informe (MyST)
preregistro.md
```
