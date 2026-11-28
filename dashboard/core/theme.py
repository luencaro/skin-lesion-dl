"""Plantillas de Plotly "cromia" y "cromia_dark" (design/Estilo.dc.html)."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from core.config import CLASS_COLORS, FONT

TEMPLATES = {"light": "cromia", "dark": "cromia_dark"}
GRAPH_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}

# Tokens de interfaz que las figuras necesitan como valores (Plotly no lee variables CSS)
TOKENS = {
    "light": {"ink": "#0E1E24", "ink2": "#3C5259", "ink3": "#5B6F76", "grid": "#E9EFF0", "line": "#DCE5E6",
              "surface": "#FFFFFF", "surface2": "#EEF3F3", "brand": "#0B5563", "accent": "#F08A4B"},
    "dark": {"ink": "#E7EFF1", "ink2": "#B3C4C8", "ink3": "#8DA1A7", "grid": "#1A2E34", "line": "#22373D",
             "surface": "#101E23", "surface2": "#162A30", "brand": "#4FB3C1", "accent": "#F59E63"},
}


def cromia_template(dark: bool = False) -> go.layout.Template:
    ink = "#E7EFF1" if dark else "#0E1E24"
    grid = "#1A2E34" if dark else "#E9EFF0"
    tip = "#E7EFF1" if dark else "#0E1E24"
    return go.layout.Template(
        data={"bar": [go.Bar(marker_line_width=0)]},
        layout={
            "font": {"family": FONT, "size": 14, "color": ink},
            "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
            "colorway": list(CLASS_COLORS.values()),
            "margin": {"l": 48, "r": 16, "t": 16, "b": 40},
            "xaxis": {"gridcolor": grid, "zeroline": False, "showline": False, "ticks": "", "automargin": True},
            "yaxis": {"gridcolor": grid, "zeroline": False, "showline": False, "ticks": "", "automargin": True},
            "hoverlabel": {"bgcolor": tip, "bordercolor": tip,
                            "font": {"family": "Inter", "size": 13, "color": "#0A1418" if dark else "#FFFFFF"}},
            "legend": {"orientation": "h", "y": -0.2, "bgcolor": "rgba(0,0,0,0)"},
            "bargap": 0.35,
            "dragmode": False,         # en móvil, arrastrar desplaza la página, no hace zoom
        },
    )


def register() -> None:
    pio.templates["cromia"] = cromia_template()
    pio.templates["cromia_dark"] = cromia_template(dark=True)
    pio.templates.default = "cromia"


def mode(theme: str | None) -> str:
    return "dark" if theme == "dark" else "light"


def template(theme: str | None) -> str:
    return TEMPLATES[mode(theme)]


def tokens(theme: str | None) -> dict:
    return TOKENS[mode(theme)]
