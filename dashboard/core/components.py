"""Componentes reutilizables de la interfaz."""

from __future__ import annotations

import math

import dash_bootstrap_components as dbc
from dash import dcc, html

from core import data
from core.theme import GRAPH_CONFIG


def interpretation(key: str | None, fallback: str | None = None):
    """Frase de interpretación tomada de los notebooks 05 y 06 (con su procedencia)."""
    item = data.interpretation(key) if key else None
    if item is None and not fallback:
        return None
    text = item["text"] if item else fallback
    source = f" — {item['source']}" if item else ""
    return html.P([html.Span("Interpretación. ", className="interp-label"), text,
                   html.Span(source, className="interp-source")], className="interpretation")


def section(sid: str, title: str, body, lead: str | None = None, interp: str | None = None, controls=None):
    """Tarjeta de sección con título, controles opcionales, cuerpo e interpretación."""
    children = [html.H2(title, className="section-title", id=f"{sid}-title")]
    if lead:
        children.append(html.P(lead, className="section-lead"))
    if controls is not None:
        children.append(html.Div(controls, className="section-controls"))
    children.append(html.Div(body, id=f"{sid}-body"))
    note = interpretation(interp)
    if note is not None:
        children.append(note)
    return html.Section(children, className="card-section", id=sid, **{"aria-labelledby": f"{sid}-title"})


def graph(gid: str, height: int = 340, figure=None):
    return dcc.Loading(dcc.Graph(id=gid, config=GRAPH_CONFIG, figure=figure or {}, style={"height": f"{height}px"},
                                 className="graph"), type="dot", color="var(--color-primary)")


def static_graph(figure, height: int = 340):
    return dcc.Graph(figure=figure, config=GRAPH_CONFIG, style={"height": f"{height}px"}, className="graph")


def kpi(value: str, label: str, note: str | None = None, tone: str = "primary"):
    return html.Div([html.Div(value, className=f"kpi-value kpi-{tone}"), html.Div(label, className="kpi-label"),
                     html.Div(note, className="kpi-note") if note else None], className="kpi")


def pending(title: str, message, color: str = "warning"):
    return dbc.Alert([html.H3(title, className="alert-title"), html.Div(message)], color=color,
                     className="pending-alert")


def error_alert(message: str):
    return dbc.Alert(message, color="danger", className="error-alert", dismissable=True)


def swatch(label: str, color: str):
    return html.Span([html.Span(className="swatch", style={"backgroundColor": color}), label], className="swatch-label")


def fmt(x, digits: int = 3, pct: bool = False) -> str:
    """Número con coma decimal (es-CO); guion si falta."""
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v):
        return "—"
    s = f"{100 * v:.{digits}f} %" if pct else f"{v:,.{digits}f}"
    return s.replace(",", " ").replace(".", ",")
