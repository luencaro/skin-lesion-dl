"""Resultados ausentes en los notebooks.

Cuando faltan corridas, las funciones de src/results.py, src/evaluation.py, src/gradcam.py y
src/benchmark.py devuelven un objeto `Missing` en lugar de lanzar un error. `Missing` se
muestra en Jupyter como un mensaje y se propaga: cualquier función decorada con `@tolerant`
que reciba un `Missing` como argumento lo devuelve sin ejecutarse. Así cada sección de un
notebook muestra un mensaje claro y la siguiente continúa, y el book compila sin resultados.
"""

from __future__ import annotations

import functools
import html
from collections.abc import Sized


class Missing:
    """Resultado ausente, con el motivo y cómo obtenerlo."""

    def __init__(self, message: str):
        self.message = message

    def __bool__(self) -> bool:
        return False

    def __len__(self) -> int:
        return 0

    def __iter__(self):
        return iter(())

    def __repr__(self) -> str:
        return f"Sin resultados: {self.message}"

    def _repr_html_(self) -> str:
        return (f'<div style="border-left:4px solid #d9822b;padding:6px 10px;background:#fdf6ec">'
                f"<b>Sin resultados:</b> {html.escape(self.message)}</div>")


def is_missing(obj) -> bool:
    return isinstance(obj, Missing)


def first_missing(*objs) -> Missing | None:
    """Primer `Missing` de `objs` (no entra en listas ni dicts)."""
    for o in objs:
        if isinstance(o, Missing):
            return o
    return None


def tolerant(empty_message: str | None = None, check: str | None = None):
    """Decorador: si algún argumento es `Missing`, devuelve ese `Missing` sin ejecutar la función.

    Si `check` es el nombre de un argumento (p. ej. "run_dirs") y llega vacío, devuelve
    Missing(empty_message).
    """

    def deco(fn):
        import inspect

        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            m = first_missing(*args, *kwargs.values())
            if m is not None:
                return m
            if check is not None:
                bound = sig.bind_partial(*args, **kwargs)
                value = bound.arguments.get(check)
                if value is not None and isinstance(value, Sized) and len(value) == 0:
                    return Missing(empty_message or f"{fn.__name__}: `{check}` está vacío.")
            return fn(*args, **kwargs)

        return wrapper

    return deco


def show(obj, close: bool = True):
    """Muestra `obj` en el notebook: figuras de matplotlib (y se cierran para no duplicarlas),
    `Missing` como mensaje, dicts como un título en negrita por clave seguido de su valor, y
    cualquier otro objeto con display()."""
    import matplotlib.pyplot as plt
    from IPython.display import Markdown, display

    if obj is None:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            display(Markdown(f"**{key}**"))
            show(value, close)
        return
    if isinstance(obj, Missing):
        display(obj)
    elif hasattr(obj, "savefig"):
        display(obj)
        if close:
            plt.close(obj)
    else:
        display(obj)


def configure_display(float_format: str = "{:.6g}") -> None:
    """Formato de los floats en las tablas de pandas: 6 cifras significativas, de modo que los
    p-valores pequeños se muestran en notación científica en lugar de redondearse a 0."""
    import pandas as pd

    pd.set_option("display.float_format", float_format.format)
    pd.set_option("display.max_columns", 60)
