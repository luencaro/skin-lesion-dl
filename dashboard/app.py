"""Dashboard del proyecto (Dash multipágina, móvil primero).

    cd dashboard && python app.py                       # desarrollo: http://127.0.0.1:8050
    cd dashboard && gunicorn app:server --workers 1 --threads 2 --timeout 120

Lee solo dashboard/data/ (python -m src.export_dashboard desde la raíz). No importa TensorFlow ni
src/. Las imágenes cargadas se procesan en memoria y no se guardan ni se registran.
"""

from __future__ import annotations

import logging
import os

import dash
import dash_bootstrap_components as dbc
from core import config, data, theme
from dash import (
    ClientsideFunction,
    Dash,
    Input,
    Output,
    State,
    clientside_callback,
    dcc,
    html,
)
from flask import abort, send_file

theme.register()
logging.getLogger("werkzeug").setLevel(logging.WARNING)   # sin registro de peticiones en desarrollo

app = Dash(
    __name__,
    use_pages=True,
    title=config.TITLE,
    update_title=None,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"},
               {"name": "description", "content": config.SUBTITLE},
               {"name": "theme-color", "content": config.UI["primary"]}],
)
server = app.server
server.config["MAX_CONTENT_LENGTH"] = 2 * config.MAX_UPLOAD_BYTES   # límite del servidor (base64 + JSON)


@server.route("/media/<kind>/<name>")
def media(kind: str, name: str):
    """Miniaturas del EDA y ejemplos de la galería (solo carpetas y extensión permitidas)."""
    path = data.media_path(kind, name)
    if path is None:
        abort(404)
    resp = send_file(path, mimetype="image/jpeg", max_age=86400)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


@server.after_request
def security_headers(resp):
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    if resp.mimetype == "application/json":          # respuestas de callbacks (predicciones): sin cache
        resp.headers["Cache-Control"] = "no-store"
    return resp


def navbar():
    links = [dbc.NavItem(dbc.NavLink(page["name"], href=page["relative_path"], active="exact", className="nav-touch"))
             for page in dash.page_registry.values() if page.get("order") is not None]
    return dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand([html.Span("●", className="brand-dot", **{"aria-hidden": "true"}), "Lesiones de la piel"],
                            href="/", className="brand"),
            dbc.NavbarToggler(id="nav-toggler", n_clicks=0),
            dbc.Collapse(dbc.Nav(links, className="ms-auto", navbar=True), id="nav-collapse", navbar=True, is_open=False),
        ], fluid=True),
        className="app-navbar", fixed="top", dark=True,
    )


PRIVACY = dcc.Markdown("""
**Política de privacidad.** Este sitio no usa cookies, no tiene cuentas de usuario ni analítica.
Las imágenes que cargues se procesan en la memoria del servidor solo para calcular la predicción y se
descartan al terminar: no se guardan en disco, no se registran y no se usan para entrenar. El
servidor no registra el contenido de las peticiones. La hoja de estilos de Bootstrap se descarga de
jsDelivr. Los datos de HAM10000 se muestran con fines académicos bajo su licencia CC BY-NC 4.0.
""", className="privacy-text")


def footer():
    return html.Footer(dbc.Container([
        html.P(["Herramienta académica: ", html.Strong("no es una herramienta diagnóstica"),
                " ni sustituye la valoración de un dermatólogo."], className="footer-warning"),
        PRIVACY,
        html.P(["Código bajo licencia MIT. Datos: HAM10000 (Tschandl, Rosendahl y Kittler, 2018), CC BY-NC 4.0. ",
                "Universidad del Norte, 2026."], className="footer-credits"),
    ], fluid=True), className="app-footer")


app.layout = html.Div([
    dcc.Location(id="url"),
    html.A("Saltar al contenido", href="#main", className="skip-link"),
    navbar(),
    html.Main(dbc.Container(dash.page_container, fluid=True, className="page"), id="main"),
    footer(),
])

# Menú colapsable en móvil (en el navegador) y cierre al navegar.
clientside_callback(
    ClientsideFunction(namespace="ui", function_name="toggleNav"),
    Output("nav-collapse", "is_open"),
    Input("nav-toggler", "n_clicks"),
    Input("url", "pathname"),
    State("nav-collapse", "is_open"),
    prevent_initial_call=True,
)


if __name__ == "__main__":
    app.run(debug=os.environ.get("DASH_DEBUG") == "1", host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "8050")))
