"""Métricas de clasificación multiclase a partir de etiquetas enteras y probabilidades."""

from __future__ import annotations

import keras
import numpy as np
from keras import ops
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _normalize(proba) -> np.ndarray:
    p = np.asarray(proba, dtype=np.float64)
    return p / p.sum(axis=1, keepdims=True)


def cross_entropy(y_true, proba, eps: float = 1e-7) -> float:
    """Entropía cruzada media (sin pesos de clase)."""
    p = _normalize(proba)
    y = np.asarray(y_true)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], eps, 1.0))))


def expected_calibration_error(y_true, proba, n_bins: int) -> float:
    """ECE de la clase predicha: bins de igual ancho en la confianza, (lo, hi], el primero incluye 0."""
    p = _normalize(proba)
    y = np.asarray(y_true)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _safe(fn, *args, **kwargs) -> float:
    """Devuelve NaN si la métrica no está definida (p. ej. una clase ausente en y_true)."""
    try:
        return float(fn(*args, **kwargs))
    except ValueError:
        return float("nan")


def classification_metrics(y_true, proba, class_names, ece_bins: int) -> dict:
    """Exactitud balanceada, exactitud, F1 macro y ponderado, precisión/sensibilidad/F1 por clase,
    AUC-ROC macro uno contra el resto, AUC-PR macro, MCC, ECE y matriz de confusión."""
    y = np.asarray(y_true)
    p = _normalize(proba)
    k = len(class_names)
    labels = np.arange(k)
    y_pred = p.argmax(axis=1)
    onehot = np.eye(k)[y]
    prec, rec, f1, support = precision_recall_fscore_support(
        y, y_pred, labels=labels, zero_division=0
    )
    present = onehot.sum(axis=0) > 0  # AUC-PR solo definida para clases presentes
    return {
        "n": len(y),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, y_pred, labels=labels, average="weighted", zero_division=0)),
        "auc_roc_macro_ovr": _safe(
            roc_auc_score, y, p, multi_class="ovr", average="macro", labels=labels
        ),
        "auc_pr_macro": _safe(average_precision_score, onehot[:, present], p[:, present], average="macro"),
        "mcc": float(matthews_corrcoef(y, y_pred)),
        "ece": expected_calibration_error(y, p, ece_bins),
        "cross_entropy": cross_entropy(y, p),
        "per_class": {
            c: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, c in enumerate(class_names)
        },
        "confusion_matrix": {
            "labels": list(class_names),
            "matrix": confusion_matrix(y, y_pred, labels=labels).tolist(),  # filas = real, columnas = predicha
        },
    }


@keras.saving.register_keras_serializable(package="skin_lesion_dl")
class SparseBalancedAccuracy(keras.metrics.Metric):
    """Exactitud balanceada (media de la sensibilidad de las clases presentes) acumulada sobre
    la época, para etiquetas enteras y probabilidades. Ignora `sample_weight`: los pesos de clase
    son constantes dentro de cada clase y no cambian la sensibilidad por clase."""

    def __init__(self, num_classes: int, name: str = "balanced_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.total = self.add_variable(shape=(self.num_classes,), initializer="zeros", name="total")
        self.correct = self.add_variable(shape=(self.num_classes,), initializer="zeros", name="correct")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(ops.reshape(y_true, (-1,)), "int32")
        y_hat = ops.cast(ops.argmax(y_pred, axis=-1), "int32")
        onehot = ops.one_hot(y_true, self.num_classes, dtype=self.dtype)
        hit = ops.cast(ops.equal(y_true, y_hat), self.dtype)
        self.total.assign_add(ops.sum(onehot, axis=0))
        self.correct.assign_add(ops.sum(onehot * ops.expand_dims(hit, -1), axis=0))

    def result(self):
        present = ops.cast(ops.greater(self.total, 0), self.dtype)
        recall = ops.divide_no_nan(self.correct, self.total)
        return ops.sum(recall * present) / ops.maximum(ops.sum(present), 1.0)

    def reset_state(self):
        self.total.assign(ops.zeros_like(self.total))
        self.correct.assign(ops.zeros_like(self.correct))

    def get_config(self):
        return {**super().get_config(), "num_classes": self.num_classes}
