"""Componentes compartidos de la interfaz (design/*.dc.html): iconos, logo, aviso, marcadores,
estados vacíos, tarjetas de cifra, encabezados de insight y formato numérico es-CO."""

from __future__ import annotations

import math

from dash import dcc, html

from core.config import CLASS_COLORS, CLASS_NAMES, DISCLAIMER
from core.theme import GRAPH_CONFIG

# ---------------------------------------------------------------------------
# Formato: miles con espacio y decimales con coma
# ---------------------------------------------------------------------------


def _finite(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def fmt_num(n, digits: int = 0) -> str:
    """10015 -> "10 015"; 1.25 (digits=2) -> "1,25"; None -> "—"."""
    v = _finite(n)
    if v is None:
        return "—"
    return f"{v:,.{digits}f}".replace(",", " ").replace(".", ",")


def fmt_pct(x, digits: int = 1, scale: bool = True) -> str:
    """Proporción 0,58 -> "58,0 %" (scale=False si ya está en %)."""
    v = _finite(x)
    if v is None:
        return "—"
    return f"{fmt_num(100 * v if scale else v, digits)} %"


def fmt_signed(x, digits: int = 1) -> str:
    v = _finite(x)
    if v is None:
        return "—"
    return ("+" if v > 0 else "−" if v < 0 else "±") + fmt_num(abs(v), digits)


def is_num(x) -> bool:
    return _finite(x) is not None


# ---------------------------------------------------------------------------
# Piezas básicas
# ---------------------------------------------------------------------------


def icon(name: str, size: int = 20, cls: str = "", style: dict | None = None):
    """Icono SVG monocromo (máscara CSS en assets/cromia.css): toma el color del texto."""
    return html.Span(className=f"ic ic-{name} {cls}".strip(), style={"width": size, "height": size, **(style or {})},
                     **{"aria-hidden": "true"})


def logo(size: int = 32):
    return html.Span(className="logo", style={"width": size, "height": size}, **{"aria-hidden": "true"})


def placeholder(text: str, cls: str = ""):
    """Marcador visible de dato pendiente (borde discontinuo naranja)."""
    return html.Span(text, className=f"ph {cls}".strip(), title="Dato pendiente: aún no hay resultados exportados")


def value_or_ph(value, render, ph: str):
    """`render(value)` si el valor es numérico; si no, el marcador `ph`."""
    return render(value) if is_num(value) else placeholder(ph)


def dot(color: str, size: int | None = None):
    style = {"background": color}
    if size:
        style |= {"width": size, "height": size}
    return html.Span(className="dot", style=style, **{"aria-hidden": "true"})


def class_dot(c: str, size: int | None = None):
    return dot(CLASS_COLORS.get(c, "var(--ink-3)"), size)


def class_label(c: str, cls: str = "class-label"):
    return html.Span([class_dot(c), CLASS_NAMES.get(c, c)], className=cls)


def eyebrow(text: str, accent: bool = False):
    return html.Span(text, className="eyebrow" + (" eyebrow-accent" if accent else ""))


def disclaimer(compact: bool = False):
    """Aviso legal (Inicio y Demo)."""
    return html.Div([
        icon("info", 22 if not compact else 20, "ic-accent"),
        html.P([html.Strong(DISCLAIMER[0]), " ", DISCLAIMER[1]]),
    ], className="disclaimer" + (" compact" if compact else ""), role="note")


def empty_state(title: str, body: str, size: int = 160, ic: str = "eye", heading: str = "h2", cls: str = ""):
    """Círculo brand-soft con icono, título en Manrope y una línea de texto."""
    tag = getattr(html, heading.upper() if heading.startswith("h") else "H2")
    return html.Div([
        html.Div([html.Div(className="empty-ring"), icon(ic, int(size * 0.35), "empty-ic")],
                 className="empty-art", style={"width": size, "height": size}),
        tag(title, className="disp empty-title"),
        html.P(body, className="empty-body"),
    ], className=f"empty {cls}".strip())


def stat_card(value, label: str, highlight: bool = False, extra=None):
    return html.Div([
        html.Span(value, className="disp num stat-value"),
        html.Span(label, className="stat-label"),
        extra,
    ], className="card stat" + (" stat-hl" if highlight else ""))


def insight_header(eyebrow_text: str | None, title, subtitle=None, size: int = 28, heading: str = "h2", hid: str | None = None):
    tag = html.H2 if heading == "h2" else html.H3
    kids = []
    if eyebrow_text:
        kids.append(eyebrow(eyebrow_text, accent=True))
    kids.append(tag(title, className="disp insight-title", style={"fontSize": size}, id=hid or None) if hid
                else tag(title, className="disp insight-title", style={"fontSize": size}))
    if subtitle is not None:
        kids.append(html.Span(subtitle, className="insight-sub"))
    return html.Div(kids, className="insight")


def graph(gid, height: int = 380, figure=None, label: str | None = None):
    props = {"aria-label": label} if label else {}
    return html.Div(dcc.Graph(id=gid, config=GRAPH_CONFIG, figure=figure or {}, style={"height": f"{height}px"}),
                    className="graph", role="img" if label else None, **props)


def prob_row(c: str, p: float, compact: bool = False):
    """Fila de probabilidad: punto + nombre, barra del color de la clase y porcentaje."""
    width = f"{max(100 * float(p), 1):.2f}%"
    bar = html.Div(html.Span(style={"width": width, "background": CLASS_COLORS[c]}), className="bar",
                   role="presentation")
    return html.Div([class_label(c, "prob-name"), bar, html.Span(fmt_pct(p), className="num prob-val")],
                    className="prob-row" + (" compact" if compact else ""))


def alert(message, title: str | None = None, kind: str = "warn"):
    return html.Div([icon("alert", 22), html.Div([html.Strong(title) if title else None, html.Span(message)],
                                                 className="alert-text")],
                    className=f"alert alert-{kind}", role="alert")
