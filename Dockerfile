# Entorno reproducible del proyecto (Python 3.11, TensorFlow 2.20 con CUDA desde pip, uv.lock).
#
#   docker build -t skin-lesion-dl .
#   docker run --gpus all --rm -it -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" \
#       -v "$HOME/.keras:/root/.keras" -v "$HOME/.cache/kagglehub:/root/.cache/kagglehub" \
#       -p 8888:8888 skin-lesion-dl
#
# Los datos (data/raw) y los resultados (outputs) se montan como volúmenes: no se copian a la
# imagen. La GPU requiere NVIDIA Container Toolkit; sin --gpus, TensorFlow usa la CPU. Los pesos
# de ImageNet (keras.applications y keras_hub) se descargan la primera vez en los volúmenes de cache.
FROM python:3.11.18-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.10 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias exactas de uv.lock (sin el grupo dev); capa separada para aprovechar la cache
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Código, configuración, libro y datos versionados (partición, hashes, anotaciones)
COPY src ./src
COPY configs ./configs
COPY book ./book
COPY data ./data
COPY index.md myst.yml references.bib preregistro.md preregistro_e2.md requirements.txt ./
RUN uv sync --frozen --no-dev

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.root_dir=/app"]
