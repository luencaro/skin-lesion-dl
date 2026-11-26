"""Núcleo del dashboard: configuración, tema, acceso a datos, inferencia ONNX y componentes.

Autocontenido: no importa TensorFlow ni el paquete src/ del entrenamiento. Solo lee dashboard/data/,
que genera `python -m src.export_dashboard` desde la raíz del repositorio.
"""
