"""Página 404."""

import dash
from dash import dcc, html

dash.register_page(__name__, path="/404", name="No encontrada")


def layout(**_):
    return html.Div([html.H1("Página no encontrada", className="page-title"),
                     dcc.Link("Volver al contexto", href="/")])
