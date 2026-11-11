"""Utilidades generales del proyecto."""

import tensorflow as tf


def set_seed(seed: int) -> None:
    """Fija las semillas de Python, NumPy y TensorFlow y activa el determinismo de operaciones."""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def list_gpus() -> list:
    """Devuelve las GPU físicas que TensorFlow detecta."""
    return tf.config.list_physical_devices("GPU")
