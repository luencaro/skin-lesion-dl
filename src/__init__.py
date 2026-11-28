import os

# Oculta los mensajes informativos de TensorFlow (conserva warnings y errores). Debe fijarse
# antes de importar TensorFlow; todo módulo de src pasa por aquí primero.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
