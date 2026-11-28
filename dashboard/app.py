"""Cromia: dashboard del proyecto (Dash multipágina, sin frameworks JS adicionales).

    cd dashboard && python app.py                       # desarrollo: http://127.0.0.1:8050
    cd dashboard && gunicorn app:server --workers 1 --threads 2 --timeout 120

Lee solo dashboard/data/ (python -m src.export_dashboard desde la raíz). No importa TensorFlow ni
src/. Las imágenes cargadas se procesan en memoria y no se guardan ni se registran.
"""

from __future__ import annotations

import logging
import os

import dash
from core import config, data, theme
from core.components import icon, logo
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
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("werkzeug").setLevel(logging.WARNING)   # sin registro de peticiones en desarrollo

app = Dash(
    __name__,
    use_pages=True,
    title=config.TITLE,
    update_title=None,
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"},
               {"name": "description", "content": config.SUBTITLE},
               {"name": "theme-color", "content": "#0B5563"}],
)
server = app.server
server.config["MAX_CONTENT_LENGTH"] = 2 * config.MAX_UPLOAD_BYTES   # límite del servidor (base64 + JSON)

app.index_string = app.index_string.replace("<html>", '<html lang="es">')


@server.route("/media/<kind>/<name>")
def media(kind: str, name: str):
    """Miniaturas de la galería y ejemplos de la demo (solo carpetas y extensión permitidas)."""
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


NAV = [("/", "Inicio", "nav-inicio"), ("/demo", "Demo en vivo", "nav-demo"), ("/modelos", "Modelos", "nav-modelos"),
       ("/datos", "Explora los datos", "nav-datos")]


def navbar():
    return html.Header(html.Div([
        dcc.Link([logo(32), html.Span(config.BRAND, className="disp brand-name")], href="/", className="brand-link",
                 title=f"{config.BRAND}: inicio"),
        html.Nav([dcc.Link(label, href=href, id=nid) for href, label, nid in NAV], className="nav",
                 id="site-nav", **{"aria-label": "Principal"}),
        html.Div([
            html.Button(icon("moon", 20), id="theme-toggle", className="icon-btn", n_clicks=0,
                        **{"aria-label": "Cambiar entre modo claro y oscuro"}),
            dcc.Link("Probar ahora", href="/demo", className="btn btn-primary btn-sm"),
            html.Button(icon("menu", 22), id="menu-toggle", className="icon-btn menu-btn", n_clicks=0,
                        **{"aria-label": "Abrir menú", "aria-controls": "site-nav", "aria-expanded": "false"}),
        ], className="header-actions"),
    ], className="wrap site-header-inner"), className="site-header", id="site-header")


def footer():
    return html.Footer(html.Div([
        html.Span(f"{config.BRAND} · Proyecto académico de clasificación de lesiones de piel · Universidad del Norte, 2026"),
        html.Span("Datos: HAM10000 (Tschandl, Rosendahl y Kittler, 2018), CC BY-NC 4.0 · Código: MIT"),
        html.Details([
            html.Summary("Privacidad"),
            html.P("Sin cookies, cuentas ni analítica. Las imágenes que cargues se procesan en la memoria del servidor solo "
                   "para calcular la predicción y se descartan al terminar: no se guardan en disco, no se registran y no se "
                   "usan para entrenar. El navegador guarda solo tu preferencia de tema claro u oscuro. Las tipografías "
                   "Manrope e Inter se descargan de Google Fonts."),
        ], className="privacy"),
    ], className="wrap site-footer-inner"), className="site-footer")


app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="theme", storage_type="local"),
    html.A("Saltar al contenido", href="#main", className="skip-link"),
    navbar(),
    html.Main(dash.page_container, id="main", tabIndex="-1"),
    footer(),
], id="root", className="cr light")

# Tema: el botón alterna el Store local; el Store fija la clase del div raíz (y las figuras lo leen).
clientside_callback(ClientsideFunction("ui", "toggleTheme"), Output("theme", "data"),
                    Input("theme-toggle", "n_clicks"), Input("url", "pathname"), State("theme", "data"))
clientside_callback(ClientsideFunction("ui", "applyTheme"), Output("root", "className"), Output("theme-toggle", "children"),
                    Input("theme", "data"))
# Navegación: enlace activo (.cur) y menú de móvil (se cierra al navegar).
clientside_callback(ClientsideFunction("ui", "navCurrent"), *[Output(nid, "className") for _, _, nid in NAV],
                    Input("url", "pathname"))
clientside_callback(ClientsideFunction("ui", "toggleMenu"), Output("site-header", "className"),
                    Output("menu-toggle", "aria-expanded"), Input("menu-toggle", "n_clicks"), Input("url", "pathname"),
                    State("site-header", "className"), prevent_initial_call=True)


if __name__ == "__main__":
    app.run(debug=os.environ.get("DASH_DEBUG") == "1", host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "8050")))
