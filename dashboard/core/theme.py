"""Plantilla de Plotly con la tipografía y los colores del tema."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from core.config import CLASS_COLORS, FONT, OKABE_ITO, UI

TEMPLATE = "skin"
GRID = "rgba(27, 38, 49, 0.12)"   # UI["dark"] con transparencia


def register() -> None:
    colorway = [OKABE_ITO[k] for k in ("blue", "orange", "green", "vermillion", "purple", "sky", "yellow")]
    pio.templates[TEMPLATE] = go.layout.Template(
        layout=go.Layout(
            font={"family": FONT, "color": UI["dark"], "size": 13},
            title={"font": {"size": 15, "color": UI["dark"]}, "x": 0.0, "xanchor": "left"},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=colorway,
            margin={"l": 48, "r": 12, "t": 36, "b": 44},
            xaxis={"gridcolor": GRID, "zerolinecolor": GRID, "linecolor": GRID, "automargin": True,
                   "title": {"standoff": 8}},
            yaxis={"gridcolor": GRID, "zerolinecolor": GRID, "linecolor": GRID, "automargin": True,
                   "title": {"standoff": 8}},
            legend={"orientation": "h", "yanchor": "top", "y": -0.18, "x": 0, "font": {"size": 12}},
            hoverlabel={"font": {"family": FONT}, "bgcolor": UI["light"], "bordercolor": UI["dark"]},
            bargap=0.25,
            dragmode=False,                 # en móvil, arrastrar desplaza la página, no hace zoom
        ),
        data={"bar": [go.Bar(marker={"line": {"color": UI["dark"], "width": 0.8}})],
              "box": [go.Box(line={"width": 1.2}, marker={"line": {"color": UI["dark"], "width": 0.5}})],
              "scatter": [go.Scatter(marker={"line": {"color": UI["dark"], "width": 0.8}})]},
    )
    pio.templates.default = f"plotly_white+{TEMPLATE}"


GRAPH_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}


def class_color(c: str) -> str:
    return CLASS_COLORS.get(c, UI["dark"])
