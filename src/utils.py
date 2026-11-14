"""Utilidades generales del proyecto."""

import logging
import math
import platform
import time
from pathlib import Path

import tensorflow as tf

LOGGER_NAME = "src"
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fija las semillas de Python, NumPy y TensorFlow; si `deterministic`, activa el determinismo de operaciones."""
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


def list_gpus() -> list:
    """Devuelve las GPU físicas que TensorFlow detecta."""
    return tf.config.list_physical_devices("GPU")


def enable_memory_growth() -> None:
    """Activa memory growth en todas las GPU. No hace nada si la GPU ya se inicializó."""
    for gpu in list_gpus():
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def environment_info() -> dict:
    """GPU y versiones de Python, TensorFlow, Keras, CUDA y cuDNN (estas dos, las de compilación de TF)."""
    import keras

    build = tf.sysconfig.get_build_info()
    gpus = []
    for gpu in list_gpus():
        try:
            details = tf.config.experimental.get_device_details(gpu)
        except Exception:  # noqa: BLE001 - los detalles son opcionales
            details = {}
        cc = details.get("compute_capability")
        gpus.append(
            {
                "device": gpu.name,
                "name": details.get("device_name"),
                "compute_capability": ".".join(map(str, cc)) if cc else None,
            }
        )
    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "cuda": build.get("cuda_version"),
        "cudnn": build.get("cudnn_version"),
        "platform": platform.platform(),
        "gpus": gpus,
    }


def setup_logging(log_file: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Configura el logger "src" (y sus hijos src.train, src.tune, ...) para escribir con marca
    de hora en la terminal y en `log_file` (modo append, se conserva al reanudar)."""
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
    for h in (logging.StreamHandler(), logging.FileHandler(log_file, mode="a", encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def format_duration(seconds: float) -> str:
    """Duración legible, p. ej. 1h 02m 03s, 4m 05s o 6.2s."""
    if math.isnan(seconds):
        return "?"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(round(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


class RunProgress:
    """Progreso de una serie de corridas: "Corrida i de N", reutilizadas, tiempo transcurrido y
    tiempo restante estimado (media de las corridas ejecutadas en esta sesión x pendientes)."""

    def __init__(self, total: int, already_done, logger: logging.Logger):
        self.total = total
        self.pending = set(range(total)) - set(already_done)  # índices que hay que ejecutar
        self.logger = logger
        self.t0 = time.perf_counter()
        self.durations: list[float] = []
        self._start = 0.0
        n_reused = total - len(self.pending)
        logger.info(f"{total} corridas: {n_reused} ya terminadas (se reutilizan), {len(self.pending)} por ejecutar")

    def _eta(self) -> str:
        if not self.pending:
            return "0s"
        if not self.durations:
            return "desconocido"
        return format_duration(sum(self.durations) / len(self.durations) * len(self.pending))

    def _status(self) -> str:
        return (f"transcurrido {format_duration(time.perf_counter() - self.t0)}, "
                f"restante estimado {self._eta()}")

    def reused(self, i: int, desc: str) -> None:
        self.logger.info(f"Corrida {i + 1} de {self.total} ({desc}): reutilizada")

    def start(self, i: int, desc: str) -> None:
        self._start = time.perf_counter()
        self.logger.info(f"Corrida {i + 1} de {self.total} ({desc}): inicio; {self._status()}")

    def finish(self, i: int, desc: str, result: str = "") -> None:
        dt = time.perf_counter() - self._start
        self.durations.append(dt)
        self.pending.discard(i)
        extra = f"{result}; " if result else ""
        self.logger.info(
            f"Corrida {i + 1} de {self.total} ({desc}): terminada en {format_duration(dt)}; {extra}{self._status()}"
        )
