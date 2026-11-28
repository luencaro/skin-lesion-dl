"""Página 404."""

import dash
from core.components import empty_state
from core.config import TITLE
from dash import dcc, html

dash.register_page(__name__, path="/404", name="No encontrada", title=f"Página no encontrada · {TITLE}")


def layout(**_):
    return html.Div([
        empty_state("Página no encontrada", "La dirección no existe o cambió de nombre.", ic="eye", heading="h1"),
        html.Div(dcc.Link("Volver al inicio", href="/", className="btn btn-primary"), style={"textAlign": "center"}),
    ], className="wrap page", style={"alignItems": "center"})
