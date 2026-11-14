"""Lectura y gráficos de resultados de ajuste, piloto y corridas (genérico para cualquier modelo).

Una corrida es una carpeta outputs/runs/<modelo>/<etiqueta>/s<semilla>_f<pliegue>/ con
config.yaml, history.csv, metrics.json, run_info.json y probabilities_<split>.csv.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT, load_json

GLOBAL_METRICS = [
    "balanced_accuracy",
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "auc_roc_macro_ovr",
    "auc_pr_macro",
    "mcc",
    "ece",
]
STAT_ROWS = ["mean", "std"]


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------


def load_or_run(path: str | Path, run_fn: Callable[[], object], rerun: bool = False) -> tuple[pd.DataFrame, str]:
    """Lee el CSV `path` si existe y `rerun` es False; si no, llama a `run_fn()` (que debe
    escribirlo) y lo lee. Devuelve (tabla, "cargado" | "ejecutado")."""
    path = Path(path)
    if path.exists() and not rerun:
        return pd.read_csv(path), "cargado"
    run_fn()
    return pd.read_csv(path), "ejecutado"


def run_dirs_from_table(table: pd.DataFrame) -> list[Path]:
    """Carpetas de corrida de la columna `run_dir` (relativa a la raíz), sin filas de resumen."""
    if "row" in table.columns:
        table = table[table["row"] == "run"]
    return [PROJECT_ROOT / p for p in table["run_dir"]]


def run_label(info: dict) -> str:
    return f"s{info['seed']}_f{info['val_fold']}"


def load_run_info(run_dir: str | Path) -> dict:
    return load_json(Path(run_dir) / "run_info.json")


def load_run_metrics(run_dir: str | Path) -> dict:
    return load_json(Path(run_dir) / "metrics.json")


def environment_table(info: dict) -> pd.DataFrame:
    """Versiones y GPU de un dict de utils.environment_info() (o de run_info["environment"])."""
    gpus = info.get("gpus") or []
    out = {k: info.get(k) for k in ("python", "tensorflow", "keras", "cuda", "cudnn", "platform")}
    out["gpu"] = "; ".join(f"{g.get('name') or g.get('device')} (cc {g.get('compute_capability')})"
                           for g in gpus) or "ninguna"
    return pd.Series(out, name="valor").to_frame()


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------


def model_parameter_table(model) -> pd.DataFrame:
    """Parámetros entrenables y no entrenables con el backbone congelado y descongelado.
    Deja el backbone como estaba."""
    from src.models import count_params, get_backbone, set_backbone_trainable

    original = get_backbone(model).trainable
    rows = {}
    for name, trainable in (("backbone congelado", False), ("backbone descongelado", True)):
        set_backbone_trainable(model, trainable)
        rows[name] = count_params(model)
    set_backbone_trainable(model, original)
    return pd.DataFrame(rows).T[["trainable", "non_trainable", "total"]]


# ---------------------------------------------------------------------------
# Ajuste de hiperparámetros
# ---------------------------------------------------------------------------


def tuning_table(trials: pd.DataFrame, metric: str = "val_balanced_accuracy") -> pd.DataFrame:
    """Ensayos ordenados por `metric` (descendente; en empate, el de menor índice)."""
    return trials.sort_values([metric, "trial"], ascending=[False, True]).reset_index(drop=True)


def plot_metric_vs_hparams(trials: pd.DataFrame, space: dict, metric: str = "val_balanced_accuracy",
                           best_trial: int | None = None, cell_size: float = 4.0):
    """Un panel por hiperparámetro de `space`: `metric` frente a su valor (eje log si es
    loguniform). Resalta `best_trial` si se indica."""
    names = list(space)
    fig, axes = plt.subplots(1, len(names), figsize=(cell_size * len(names), cell_size * 0.8),
                             sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        ax.scatter(trials[name], trials[metric], color="tab:blue", alpha=0.8)
        for _, r in trials.iterrows():
            ax.annotate(str(int(r["trial"])), (r[name], r[metric]), textcoords="offset points",
                        xytext=(4, 3), fontsize=8)
        if best_trial is not None:
            b = trials[trials["trial"] == best_trial]
            ax.scatter(b[name], b[metric], s=120, facecolors="none", edgecolors="tab:red", linewidths=2,
                       label=f"mejor (ensayo {best_trial})")
            ax.legend(loc="lower right", fontsize=8)
        if space[name].get("type") == "loguniform":
            ax.set_xscale("log")
        if space[name].get("type") == "choice":
            ax.set_xticks(space[name]["values"])
        ax.set_xlabel(name)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(metric)
    fig.tight_layout()
    return fig


def compare_hparams(model_cfg: dict, best: dict | None, rtol: float = 1e-9) -> pd.DataFrame:
    """Hiperparámetros del YAML frente a best.json. Emite un warning si alguno es null o difiere."""
    yaml_hp = model_cfg.get("hyperparameters") or {}
    best_hp = (best or {}).get("hyperparameters") or {}
    rows = {}
    for k in dict.fromkeys([*yaml_hp, *best_hp]):
        y, b = yaml_hp.get(k), best_hp.get(k)
        if y is None:
            estado = "null en el YAML"
        elif b is None:
            estado = "sin best.json"
        elif not np.isclose(float(y), float(b), rtol=rtol, atol=0.0):
            estado = "difiere de best.json"
        else:
            estado = "coincide"
        rows[k] = {"yaml": y, "best.json": b, "estado": estado}
    table = pd.DataFrame(rows).T
    problemas = table[table["estado"] != "coincide"]
    if len(problemas):
        warnings.warn(
            f"configs/{model_cfg.get('name')}.yaml: "
            + ", ".join(f"{k} ({e})" for k, e in problemas["estado"].items()),
            stacklevel=2,
        )
    return table


# ---------------------------------------------------------------------------
# Corridas: tablas de variabilidad, métricas y costo
# ---------------------------------------------------------------------------


def add_mean_std(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Añade filas mean y std (ddof = 1) de las columnas numéricas `cols` (por defecto todas)."""
    cols = list(cols) if cols is not None else list(df.select_dtypes("number").columns)
    stats = pd.DataFrame([df[cols].mean(), df[cols].std(ddof=1)], index=STAT_ROWS)
    return pd.concat([df, stats])


def run_metrics_table(run_dirs, split: str = "val", metrics=GLOBAL_METRICS) -> pd.DataFrame:
    """Métricas globales de `split` por corrida (índice s<semilla>_f<pliegue>)."""
    rows = {}
    for d in run_dirs:
        m = load_run_metrics(d)[split]
        rows[run_label(load_run_info(d))] = {k: m.get(k) for k in metrics}
    return pd.DataFrame(rows).T.astype(float).rename_axis("run")


def per_class_table(run_dirs, split: str = "val") -> pd.DataFrame:
    """Precisión, sensibilidad, F1 y soporte por clase y corrida (índice run, class)."""
    rows = []
    for d in run_dirs:
        run = run_label(load_run_info(d))
        for c, v in load_run_metrics(d)[split]["per_class"].items():
            rows.append({"run": run, "class": c, **v})
    return pd.DataFrame(rows).set_index(["run", "class"])


def per_class_summary(per_class: pd.DataFrame, cols=("precision", "recall", "f1")) -> pd.DataFrame:
    """Media y desviación estándar (ddof = 1) entre corridas de cada métrica por clase."""
    order = list(dict.fromkeys(per_class.index.get_level_values("class")))
    g = per_class[list(cols)].groupby(level="class", sort=False)
    stats = {"mean": g.mean(), "std": g.std(ddof=1)}
    out = pd.DataFrame({(c, s): stats[s][c] for c in cols for s in STAT_ROWS})
    return out.loc[order]


def pooled_confusion(run_dirs, split: str = "val") -> tuple[np.ndarray, list[str]]:
    """Suma de las matrices de confusión de las corridas (filas = real, columnas = predicha)."""
    total, labels = None, None
    for d in run_dirs:
        cm = load_run_metrics(d)[split]["confusion_matrix"]
        m = np.asarray(cm["matrix"], dtype=np.int64)
        if labels is None:
            total, labels = m, cm["labels"]
        else:
            if cm["labels"] != labels:
                raise ValueError(f"{d}: etiquetas de la matriz distintas")
            total = total + m
    return total, labels


def plot_confusion_matrix(cm, labels, normalize: bool = True, title: str | None = None, ax=None):
    """Matriz de confusión; si `normalize`, cada fila se divide por su total (sensibilidad en la diagonal)."""
    cm = np.asarray(cm, dtype=float)
    values = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None) if normalize else cm
    if ax is None:
        _, ax = plt.subplots(figsize=(0.8 * len(labels) + 2, 0.8 * len(labels) + 1.5))
    im = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    for i in range(len(labels)):
        for j in range(len(labels)):
            txt = f"{values[i, j]:.2f}" if normalize else f"{int(values[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if values[i, j] > (0.5 if normalize else values.max() / 2) else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicha")
    ax.set_ylabel("Real")
    if title:
        ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.figure.tight_layout()
    return ax.figure


def cost_table(run_dirs) -> pd.DataFrame:
    """Tiempos de entrenamiento por fase, épocas, inferencia por imagen, parámetros y GPU por corrida."""
    rows = {}
    for d in run_dirs:
        info = load_run_info(d)
        gpus = info["environment"].get("gpus") or []
        row = {
            "warmup_s": info["train_time_s"]["warmup"],
            "finetune_s": info["train_time_s"]["finetune"],
            "total_s": info["train_time_s"]["total"],
            "epochs_warmup": info["epochs"]["warmup"],
            "epochs_finetune": info["epochs"]["finetune_run"],
            "finetune_s_per_epoch": info["train_time_s"]["finetune"] / max(info["epochs"]["finetune_run"], 1),
        }
        for split, sec in info["inference_s_per_image"].items():
            row[f"inference_ms_per_image_{split}"] = 1000 * sec
        row["trainable_params_finetune"] = info["params"]["finetune"]["trainable"]
        row["non_trainable_params_finetune"] = info["params"]["finetune"]["non_trainable"]
        row["gpu"] = "; ".join(g.get("name") or g.get("device") for g in gpus) or "ninguna"
        rows[run_label(info)] = row
    return pd.DataFrame(rows).T.rename_axis("run").infer_objects()


# ---------------------------------------------------------------------------
# Curvas de entrenamiento
# ---------------------------------------------------------------------------


def load_histories(run_dirs) -> pd.DataFrame:
    """history.csv de cada corrida con columnas run, warmup_epochs y restored_global_epoch
    (época global cuyos pesos se restauraron al final de la fase B)."""
    frames = []
    for d in run_dirs:
        info = load_run_info(d)
        h = pd.read_csv(Path(d) / "history.csv")
        h["run"] = run_label(info)
        h["warmup_epochs"] = info["epochs"]["warmup"]
        h["restored_global_epoch"] = info["epochs"]["warmup"] + info["epochs"]["finetune_best"]
        frames.append(h)
    return pd.concat(frames, ignore_index=True)


CURVES = (
    ("loss", "val_loss", "Pérdida"),
    ("balanced_accuracy", "val_balanced_accuracy", "Exactitud balanceada"),
)


def plot_training_curves(histories: pd.DataFrame, curves=CURVES, cell_size: float = 3.2):
    """Una columna por corrida y una fila por par (train, val) de `curves`, con global_epoch en
    el eje x, una línea discontinua en el cambio de fase A a B y una punteada en la época
    restaurada. Las columnas ausentes en history.csv se omiten."""
    runs = list(dict.fromkeys(histories["run"]))
    fig, axes = plt.subplots(len(curves), len(runs), figsize=(cell_size * len(runs), cell_size * len(curves)),
                             sharex=True, sharey="row", squeeze=False)
    for j, run in enumerate(runs):
        h = histories[histories["run"] == run].sort_values("global_epoch")
        switch = h["warmup_epochs"].iloc[0] + 0.5
        restored = h["restored_global_epoch"].iloc[0]
        for i, (tr, va, title) in enumerate(curves):
            ax = axes[i][j]
            for col, name, style in ((tr, "train", "--"), (va, "val", "-")):
                if col in h.columns:
                    ax.plot(h["global_epoch"], h[col], style, marker=".", label=name)
            if h["warmup_epochs"].iloc[0] > 0:
                ax.axvline(switch, color="gray", ls="--", lw=1, label="fase A → B")
            ax.axvline(restored, color="tab:red", ls=":", lw=1.5, label="época restaurada")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(run)
            if j == 0:
                ax.set_ylabel(title)
            if i == len(curves) - 1:
                ax.set_xlabel("global_epoch")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    return fig
