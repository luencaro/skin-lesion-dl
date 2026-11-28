"""Modelos (design/Modelos.dc.html): podio, tabla de líderes, nuestro modelo frente a su backbone,
coste frente a desempeño con la frontera de Pareto, matriz de confusión y calibración.

Todas las cifras salen de dashboard/data/models/ (python -m src.export_dashboard). Donde falta un
dato se ve el marcador .ph; nada se escribe a mano.
"""

from __future__ import annotations

import dash
import numpy as np
import plotly.graph_objects as go
from core import data, theme
from core.components import (
    class_dot,
    eyebrow,
    fmt_num,
    fmt_pct,
    fmt_signed,
    graph,
    insight_header,
    is_num,
    placeholder,
)
from core.config import CLASS_NAMES, CLASS_ORDER, CLASS_SHORT, ORIGINAL_MODEL, TITLE
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update

dash.register_page(__name__, path="/modelos", name="Modelos", order=2, title=f"Modelos · {TITLE}")

SORTS = [("ba", "Exactitud bal."), ("params_m", "Parámetros"), ("gflops", "GFLOPs")]
X_AXES = [("gflops", "GFLOPs"), ("latency_ms", "Latencia")]


# ---------------------------------------------------------------------------
# Podio y tabla de líderes
# ---------------------------------------------------------------------------


def podium():
    top = data.ranked(data.leaderboard_rows())[:3]

    def card(rank: int):
        r = top[rank - 1] if len(top) >= rank else None
        name = html.Span(r["name"]) if r else placeholder("[ Modelo ]")
        if rank == 1:
            return html.Div([
                html.Div([html.Div("1", className="medal medal-1"), html.Span("Mejor modelo", className="kicker")],
                         className="podium-top"),
                html.Span(name, className="disp name"),
                html.Span(fmt_pct(r["ba"]) if r else placeholder("[ BA ] %"), className="disp num val"),
                html.Span("exactitud balanceada en test" + (f" · media de {r['runs']} corridas" if r and r.get("runs") else ""),
                          className="sub"),
            ], className="card podium-card first")
        return html.Div([
            html.Div(str(rank), className=f"medal medal-{rank}"),
            html.Span(name, className="disp name"),
            html.Span(["Exactitud balanceada ", html.Strong(fmt_pct(r["ba"]), className="num") if r else placeholder("[ — ] %")],
                      className="sub"),
        ], className="card podium-card")

    return html.Div([card(2), card(1), card(3)], className="podium", role="list", **{"aria-label": "Podio"})


def _cell(value, render, ph: str, cls: str = "num c2"):
    return html.Td(render(value) if is_num(value) else placeholder(ph), className=cls)


def table_rows(sort: str) -> list:
    rows = data.leaderboard_rows()
    medals = {r["id"]: i + 1 for i, r in enumerate(data.ranked(rows)[:3])}
    ranks = {r["id"]: i + 1 for i, r in enumerate(data.ranked(rows))}
    key = {"ba": lambda r: -(r["ba"] if r["ba"] is not None else -1), "params_m": lambda r: r["params_m"] or np.inf,
           "gflops": lambda r: r["gflops"] or np.inf}[sort]
    out = []
    for r in sorted(rows, key=lambda r: (r["is_baseline"] and sort != "ba", key(r))):
        rank = (html.Div(str(medals[r["id"]]), className=f"medal medal-sm medal-{medals[r['id']]}") if r["id"] in medals
                else str(ranks.get(r["id"], "—")))
        name = [r["name"]]
        if r["is_baseline"]:
            name.append(html.Span("baseline", className="base-tag"))
        if is_num(r["ba"]):
            ba = html.Div([html.Div(html.Div(html.Span(style={"width": f"{100 * r['ba']:.1f}%"}), className="ba-fill")),
                           html.Span(fmt_pct(r["ba"]), className="num",
                                     title=f"± {fmt_num(100 * r['ba_std'], 1)} puntos (DE entre corridas)" if is_num(r["ba_std"]) else None)],
                          className="ba-cell")
        else:
            ba = html.Div([html.Div(className="stripe"), placeholder("[ — ]")], className="ba-cell")
        out.append(html.Tr([
            html.Td(rank, className="num rank"),
            html.Td(html.Span(name, className="mname")),
            html.Td(ba),
            html.Td(html.Span(fmt_pct(r["f1"]), title=f"± {fmt_num(100 * r['f1_std'], 1)} puntos (DE entre corridas)"
                              if is_num(r["f1_std"]) else None) if is_num(r["f1"]) else placeholder("[ — ]"), className="num c2"),
            _cell(r["latency_ms"], lambda v: f"{fmt_num(v, 1)} ms", "[ — ] ms"),
            _cell(r["size_mb"], lambda v: f"{fmt_num(v, 1)} MB", "[ — ] MB"),
            _cell(r["params_m"], lambda v: f"{fmt_num(v, 1)} M", "[ — ] M"),
            _cell(r["gflops"], lambda v: fmt_num(v, 2), "[ — ]"),
        ]))
    return out


def leaderboard_card():
    rows = data.leaderboard_rows()
    has_ba = any(is_num(r["ba"]) for r in rows)
    current = "ba" if has_ba else "params_m"
    pills = [html.Button(label if (k != "ba" or has_ba) else "Exactitud bal. · pendiente", id={"type": "lb-sort", "id": k},
                         n_clicks=0, className="pill on" if k == current else "pill",
                         disabled=(k == "ba" and not has_ba), title=None if (k != "ba" or has_ba) else "Disponible cuando haya resultados")
             for k, label in SORTS]
    board = data.leaderboard() or {}
    return html.Div([
        html.Div([html.H2("Tabla de líderes", className="disp"),
                  html.Div([html.Span("Ordenar por"), *pills], className="lb-sort")], className="lb-head"),
        html.Div(html.Table([
            html.Thead(html.Tr([html.Th("#", scope="col"), html.Th("Modelo", scope="col"),
                                html.Th("Exactitud balanceada", scope="col", style={"minWidth": 220}),
                                html.Th("F1 macro", scope="col"), html.Th("Latencia", scope="col"), html.Th("Tamaño", scope="col"),
                                html.Th("Parámetros", scope="col"), html.Th("GFLOPs", scope="col")])),
            html.Tbody(table_rows(current), id="lb-body"),
        ], className="lb"), className="lb-wrap", tabIndex="0", role="region", **{"aria-label": "Tabla de líderes"}),
        html.P(["Exactitud balanceada: media en ", board.get("split", "test"), " de las corridas definitivas (tabla maestra",
                f", {board['source']}" if board.get("source") else "",
                ("). F1 macro: media en test de las mismas corridas. Latencia: mediana en CPU con ONNX Runtime. "
                 "Tamaño: archivo ONNX servido. Parámetros y GFLOPs: outputs/cost a 224 × 224. Los marcadores se llenan al "
                 "ejecutar python -m src.export_dashboard.")],
               className="lb-foot"),
    ], className="card lb-card")


@callback(Output("lb-body", "children"), Output({"type": "lb-sort", "id": ALL}, "className"),
          Input({"type": "lb-sort", "id": ALL}, "n_clicks"), State({"type": "lb-sort", "id": ALL}, "id"),
          prevent_initial_call=True)
def sort_table(_clicks, ids):
    trig = ctx.triggered_id
    if not isinstance(trig, dict):
        return no_update, no_update
    return table_rows(trig["id"]), ["pill on" if i["id"] == trig["id"] else "pill" for i in ids]


# ---------------------------------------------------------------------------
# Nuestro modelo frente al backbone
# ---------------------------------------------------------------------------


def _metric(label: str, ours, base, pct: bool = True, lower_is_better: bool = False):
    fmt = (lambda v: fmt_pct(v)) if pct else (lambda v: fmt_num(100 * v, 1))
    val = html.Span(fmt(ours), className="disp num val") if is_num(ours) else html.Span(placeholder("[ — ]"), className="val")
    base_v = html.Span(fmt(base), className="num") if is_num(base) else placeholder("[ — ]")
    if is_num(ours) and is_num(base):
        d = ours - base
        better = d < 0 if lower_is_better else d > 0
        delta = html.Span(f"Δ {fmt_signed(100 * d)} pts", className="delta" + ("" if better or d == 0 else " worse"))
    else:
        delta = html.Span("Δ [ " + ("−" if lower_is_better else "+") + "— ]", className="delta na")
    return html.Div([html.Span(label, className="lbl"), val, html.Span(["sin CCI · ", base_v], className="base"), delta],
                    className="ours-metric")


def ours_card():
    o = data.ours()
    cci, base = o["cci"], o["base"] or {}
    bb = html.Span(o["backbone_name"]) if o["backbone_name"] else placeholder("[ backbone ]")
    return html.Section([
        html.Div([
            html.Span("Nuestro modelo", className="ours-badge"),
            html.H2(["CCI + ", bb, " frente al mismo backbone sin la capa."], className="disp"),
            html.P("Misma arquitectura, mismo entrenamiento. La única diferencia es la capa de constancia de color." +
                   ("" if o["has_runs"] else " Sus corridas definitivas aún no tienen resultados.")),
        ], className="ours-copy"),
        html.Div([
            _metric("Exactitud balanceada", cci["ba"], base.get("ba")),
            _metric("F1 macro", cci["f1"], base.get("f1")),
            _metric("Variación entre fuentes", o["source_std"], o["backbone_source_std"], pct=False, lower_is_better=True),
        ], className="ours-metrics"),
    ], className="ours", **{"aria-label": "Nuestro modelo frente a su backbone"})


# ---------------------------------------------------------------------------
# Coste frente a desempeño
# ---------------------------------------------------------------------------


def pareto(points: list[dict], x: str) -> list[dict]:
    """No dominados: ningún otro punto tiene coste <= y exactitud >= (con una desigualdad estricta)."""
    front = [p for p in points if not any(q[x] <= p[x] and q["ba"] >= p["ba"] and (q[x] < p[x] or q["ba"] > p["ba"])
                                          for q in points if q is not p)]
    return sorted(front, key=lambda p: p[x])


def scatter_points(x: str) -> list[dict]:
    rows = [r for r in data.leaderboard_rows() if not r["is_baseline"]] + [data.model_row(ORIGINAL_MODEL)]
    return [r for r in rows if is_num(r.get(x)) and is_num(r["ba"]) and r[x] > 0]


def scatter_figure(x: str, theme_name: str | None):
    tk = theme.tokens(theme_name)
    pts = scatter_points(x)
    fig = go.Figure()
    others = [p for p in pts if p["id"] != ORIGINAL_MODEL]
    ours = [p for p in pts if p["id"] == ORIGINAL_MODEL]
    front = pareto(pts, x)
    if len(front) >= 2:
        fig.add_scatter(x=[p[x] for p in front], y=[100 * p["ba"] for p in front], mode="lines", name="Frontera de Pareto",
                        line={"color": tk["accent"], "width": 2.5, "dash": "dash"}, hoverinfo="skip")
    unit = " GFLOPs" if x == "gflops" else " ms"
    if others:
        fig.add_scatter(x=[p[x] for p in others], y=[100 * p["ba"] for p in others], mode="markers+text", name="Modelos",
                        text=[p["name"] for p in others], textposition="top center",
                        textfont={"color": tk["ink2"], "size": 12},
                        marker={"size": 14, "color": tk["brand"]},
                        hovertemplate="%{text}<br>%{x:.2f}" + unit + "<br>BA %{y:.1f} %<extra></extra>")
    if ours:
        fig.add_scatter(x=[p[x] for p in ours], y=[100 * p["ba"] for p in ours], mode="markers+text", name="Nuestro modelo",
                        text=["Nuestro modelo"], textposition="top center", textfont={"color": tk["ink"], "size": 13},
                        marker={"size": 20, "color": tk["accent"]},
                        hovertemplate="Nuestro modelo<br>%{x:.2f}" + unit + "<br>BA %{y:.1f} %<extra></extra>")
    fig.update_layout(template=theme.template(theme_name), showlegend=bool(pts), margin={"l": 60, "r": 20, "t": 20, "b": 50},
                      xaxis_title="GFLOPs por imagen (log)" if x == "gflops" else "Latencia en CPU (ms)",
                      yaxis_title="Exactitud balanceada (%) →", separators=", ")
    if x == "gflops":
        ticks = [0.1, 0.3, 1, 3, 10, 30, 100]
        fig.update_xaxes(type="log", tickvals=ticks, ticktext=[fmt_num(t, 1 if t < 1 else 0) for t in ticks])
    if not pts:
        fig.update_xaxes(range=[np.log10(0.3), np.log10(20)] if x == "gflops" else [0, 200], showgrid=True)
        fig.update_yaxes(range=[0, 100], showticklabels=False)
    return fig


def scatter_card():
    has = bool(scatter_points("gflops") or scatter_points("latency_ms"))
    overlay = None if has else html.Div([html.B("Resultados en camino"),
                                         html.Span("Cada punto aparecerá en su coste y su exactitud real al exportar los "
                                                   "resultados.")], className="overlay")
    return html.Div([
        html.Div([
            insight_header("Insight", "Más grande no siempre es mejor.",
                           "Coste frente a exactitud balanceada. La frontera de Pareto marca los modelos que no tienen rival "
                           "más barato y mejor."),
            html.Div([html.Button(label, id={"type": "sc-x", "id": k}, n_clicks=0, className="pill on" if k == "gflops" else "pill")
                      for k, label in X_AXES], className="toggle", role="group", **{"aria-label": "Eje de coste"}),
        ], className="viz-head"),
        html.Div([graph("scatter-graph", 400, label="Dispersión de coste frente a exactitud balanceada"), overlay],
                 className="viz-body"),
        dcc.Store(id="sc-x-store", data="gflops"),
    ], className="card viz-card")


@callback(Output("sc-x-store", "data"), Output({"type": "sc-x", "id": ALL}, "className"),
          Input({"type": "sc-x", "id": ALL}, "n_clicks"), State({"type": "sc-x", "id": ALL}, "id"), prevent_initial_call=True)
def pick_x(_clicks, ids):
    trig = ctx.triggered_id
    if not isinstance(trig, dict):
        return no_update, no_update
    return trig["id"], ["pill on" if i["id"] == trig["id"] else "pill" for i in ids]


@callback(Output("scatter-graph", "figure"), Input("sc-x-store", "data"), Input("theme", "data"))
def draw_scatter(x, theme_name):
    return scatter_figure(x or "gflops", theme_name)


# ---------------------------------------------------------------------------
# Matriz de confusión y calibración
# ---------------------------------------------------------------------------


def diag_model() -> str | None:
    """El mejor modelo con diagnósticos; si ninguno tiene exactitud, el primero con diagnósticos."""
    ranked = [r["id"] for r in data.ranked(data.leaderboard_rows() + [data.model_row(ORIGINAL_MODEL)])]
    with_diag = [m for m in data.models_index().get("models_with_runs", []) if data.diagnostics(m)]
    for m in ranked:
        if m in with_diag:
            return m
    return with_diag[0] if with_diag else None


def confusion(model: str | None):
    """(conteos, normalizada por fila) en el orden CLASS_ORDER, o (None, None)."""
    d = data.diagnostics(model) if model else None
    if not d or not d.get("confusion"):
        return None, None
    idx = [d["classes"].index(c) for c in CLASS_ORDER]
    cm = np.asarray(d["confusion"], dtype=float)[np.ix_(idx, idx)]
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    return cm, norm


def top_confusion(norm) -> tuple[int, int] | None:
    if norm is None:
        return None
    off = norm.copy()
    np.fill_diagonal(off, -1)
    i, j = np.unravel_index(int(np.argmax(off)), off.shape)
    return (int(i), int(j)) if off[i, j] > 0 else None


def cells(mode: str, sel: tuple[int, int] | None, cm, norm):
    classes, styles, texts, labels = [], [], [], []
    for i in range(7):
        for j in range(7):
            on = sel == (i, j)
            if norm is None:
                classes.append("cell" + (" on" if on else ""))
                styles.append({"background": "var(--brand-soft)" if i == j else "var(--surface-2)"})
                texts.append("?")
                labels.append(f"Real {CLASS_NAMES[CLASS_ORDER[i]]}, predicho {CLASS_NAMES[CLASS_ORDER[j]]}: pendiente")
                continue
            v = float(norm[i, j])
            pct = round(100 * v ** 0.6) if v > 0 else 0         # escala perceptual para que se vean las celdas pequeñas
            classes.append("cell" + (" on" if on else "") + (" hi" if pct > 55 else " lo" if pct > 0 else ""))
            styles.append({"background": f"color-mix(in srgb, var(--brand) {pct}%, var(--surface-2))"})
            texts.append(fmt_num(100 * v, 0) if mode == "norm" else fmt_num(cm[i, j]))
            labels.append(f"Real {CLASS_NAMES[CLASS_ORDER[i]]}, predicho {CLASS_NAMES[CLASS_ORDER[j]]}: "
                          f"{fmt_num(cm[i, j])} imágenes, {fmt_pct(v)} de los casos reales")
    return classes, styles, texts, labels


def detail(mode: str, sel, cm, norm):
    i, j = sel
    real, pred = CLASS_NAMES[CLASS_ORDER[i]], CLASS_NAMES[CLASS_ORDER[j]]
    if norm is None:
        val = placeholder("[ — ] % de los casos reales" if mode == "norm" else "[ — ] imágenes")
    else:
        n, p = cm[i, j], norm[i, j]
        val = html.Span(f"{fmt_num(n)} imágenes · {fmt_pct(p)} de los casos reales", className="num")
    return [html.Span(["Real ", html.Strong(real), " · Predicho ", html.Strong(pred)]), val]


def confusion_card():
    model = diag_model()
    cm, norm = confusion(model)
    top = top_confusion(norm)
    if top:
        title = (f"Donde más se equivoca: {CLASS_NAMES[CLASS_ORDER[top[0]]]} confundido con "
                 f"{CLASS_NAMES[CLASS_ORDER[top[1]]]} ({fmt_pct(norm[top])})")
    else:
        title = placeholder("[ Frase: la confusión más frecuente, p. ej. clase A ↔ clase B ]")
    sel = top or (0, 1)
    cls, sty, txt, lab = cells("norm", sel, cm, norm)
    d = data.diagnostics(model) if model else None
    sub = (f"Matriz de confusión de {data.model_row(model)['name']} en test ({fmt_num(d['n_predictions'])} predicciones de "
           f"{d['n_runs']} corridas). Toca una celda para ver el detalle." if d
           else "Matriz de confusión del mejor modelo. Toca una celda para ver el detalle.")
    grid = [html.Button(txt[k], id={"type": "cm-cell", "i": k // 7, "j": k % 7}, n_clicks=0, className=cls[k], style=sty[k],
                        disabled=norm is None, **{"aria-label": lab[k]}) for k in range(49)]
    return html.Div([
        insight_header("Insight", title, sub, size=24),
        html.Div([html.Button(label, id={"type": "cm-mode", "id": k}, n_clicks=0, className="pill on" if k == "norm" else "pill",
                              disabled=norm is None) for k, label in (("norm", "Normalizada"), ("count", "Conteos"))],
                 className="toggle", role="group", **{"aria-label": "Valores de la matriz"}),
        html.Div([
            html.Div(),
            html.Div([html.Span([class_dot(c), CLASS_SHORT[c]], title=CLASS_NAMES[c]) for c in CLASS_ORDER], className="cm-cols"),
            html.Div([html.Span([class_dot(c), CLASS_SHORT[c]], title=CLASS_NAMES[c]) for c in CLASS_ORDER], className="cm-rows"),
            html.Div(grid, className="cm-grid", role="group", **{"aria-label": "Celdas de la matriz de confusión"}),
        ], className="cm"),
        html.Div(detail("norm", sel, cm, norm), className="cm-detail", id="cm-detail", **{"aria-live": "polite"}),
        html.Span("Eje vertical: clase real · eje horizontal: clase predicha" +
                  (" · valores normalizados por fila en %" if norm is not None else ""), className="cm-axes"),
        dcc.Store(id="cm-state", data={"model": model, "mode": "norm", "sel": list(sel)}),
    ], className="card viz-card cm-card")


@callback(Output({"type": "cm-cell", "i": ALL, "j": ALL}, "className"), Output({"type": "cm-cell", "i": ALL, "j": ALL}, "style"),
          Output({"type": "cm-cell", "i": ALL, "j": ALL}, "children"), Output("cm-detail", "children"),
          Output({"type": "cm-mode", "id": ALL}, "className"), Output("cm-state", "data"),
          Input({"type": "cm-cell", "i": ALL, "j": ALL}, "n_clicks"), Input({"type": "cm-mode", "id": ALL}, "n_clicks"),
          State("cm-state", "data"), prevent_initial_call=True)
def update_confusion(_cells, _modes, state):
    trig = ctx.triggered_id
    if not isinstance(trig, dict) or not state:
        return (no_update,) * 6
    mode, sel = state["mode"], tuple(state["sel"])
    if trig["type"] == "cm-mode":
        mode = trig["id"]
    else:
        sel = (int(trig["i"]), int(trig["j"]))
    cm, norm = confusion(state.get("model"))
    cls, sty, txt, _ = cells(mode, sel, cm, norm)
    modes = ["pill on" if m == mode else "pill" for m in ("norm", "count")]
    return cls, sty, txt, detail(mode, sel, cm, norm), modes, {**state, "mode": mode, "sel": list(sel)}


def calibration_figure(model: str | None, theme_name: str | None):
    tk = theme.tokens(theme_name)
    d = data.diagnostics(model) if model else None
    rel = (d or {}).get("reliability") or []
    fig = go.Figure()
    if rel:
        n_bins = max(r["bin"] for r in rel) + 1
        fig.add_bar(x=[(r["bin"] + 0.5) / n_bins for r in rel], y=[r["exactitud"] for r in rel], width=0.8 / n_bins,
                    marker_color=tk["brand"], name="Exactitud observada", customdata=[[r["confianza"], r["n"]] for r in rel],
                    hovertemplate="Confianza media %{customdata[0]:.2f}<br>Exactitud %{y:.2f}<br>n = %{customdata[1]}<extra></extra>")
    else:
        fig.add_bar(x=[(b + 0.5) / 10 for b in range(10)], y=[(b + 1) / 10 * 0.93 for b in range(10)], width=0.06,
                    marker={"color": tk["surface2"], "line": {"color": tk["line"], "width": 1}}, hoverinfo="skip",
                    name="Pendiente")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibración perfecta",
                    line={"color": tk["ink3"], "width": 1.5, "dash": "dash"}, hoverinfo="skip")
    fig.add_annotation(x=0.02, y=0.98, xanchor="left", yanchor="top", text="- - -  calibración perfecta", showarrow=False,
                       font={"color": tk["ink2"], "size": 11})
    fig.update_layout(template=theme.template(theme_name), showlegend=False, bargap=0.2,
                      margin={"l": 44, "r": 12, "t": 12, "b": 36}, separators=", ")
    fig.update_xaxes(range=[0, 1], tickvals=[0, 0.5, 1], ticktext=["0", "0,5", "1"])
    fig.update_yaxes(range=[0, 1.02], tickvals=[0, 0.5, 1], ticktext=["0", "0,5", "1"])
    return fig


def calibration_card():
    model = diag_model()
    ece = data.model_row(model)["ece"] if model else None
    sub = ["Diagrama de calibración", f" de {data.model_row(model)['name']}" if model else "",
           ". Error de calibración esperado (ECE): ", html.Strong(fmt_num(ece, 3), className="num") if is_num(ece) else placeholder("[ — ]")]
    d = data.diagnostics(model) if model else None
    return html.Div([
        insight_header("Insight", "Cuando dice 80 %, ¿acierta el 80 %?", sub, size=24),
        graph("calibration-graph", 340, label="Diagrama de calibración: exactitud por tramo de confianza"),
        html.Div([html.Span("Confianza predicha →"), html.Span("↑ Exactitud observada")], className="cal-axes"),
        html.Div("Probabilidades sin calibrar por temperatura (las de la tabla maestra)." if d and d.get("reliability") else
                 "Las barras se llenarán con la exactitud real de cada tramo de confianza.", className="note"),
        dcc.Store(id="cal-model", data=model),
    ], className="card viz-card cal-card")


@callback(Output("calibration-graph", "figure"), Input("cal-model", "data"), Input("theme", "data"))
def draw_calibration(model, theme_name):
    return calibration_figure(model, theme_name)


# ---------------------------------------------------------------------------
# Ficha técnica del modelo original (procedencia, pseudocódigo, demostración, pruebas)
# ---------------------------------------------------------------------------

PROOF = r"""
**Teorema.** Para toda imagen $x$ con intensidades positivas y toda ganancia diagonal $D = \mathrm{diag}(d_1, d_2, d_3)$
con $d_c > 0$, la capa cumple $h(Dx) = h(x)$, y por tanto $f(Dx) = f(x)$ para cualquier backbone.

*Demostración resumida.* El promedio de áreas es lineal y por canal: $\overline{Dx} = D\bar{x}$. La guía
$z_{i,c} = \log \bar{x}_{i,c} - \tfrac{1}{\bar N}\sum_j \log \bar{x}_{j,c}$ no cambia, porque $\log d_c$ se cancela; luego los
pesos $w_i$ tampoco. La media de Minkowski es homogénea de grado uno: $e_c(D\bar{x}) = d_c\, e_c(\bar{x})$. Por último,
$h(Dx)_{i,c} = \kappa\, d_c x_{i,c} / (d_c\, e_c) = h(x)_{i,c}$. $\blacksquare$

*Supuestos.* Intensidades estrictamente positivas y sin saturación, y diferencia entre equipos aproximada por una ganancia
diagonal. La cota $\varepsilon$ (píxeles negros) y el recorte a 255 rompen la igualdad exacta en la práctica.
"""


def technical():
    o = data.original_section()
    prov = (o.get("provenance_md") or "").split("\n", 1)[-1]
    layer = o.get("layer") or {}
    tests = o.get("tests") or []
    test_table = html.Div(html.Table([
        html.Thead(html.Tr([html.Th("Prueba"), html.Th("Qué comprueba"), html.Th("Resultado")])),
        html.Tbody([html.Tr([html.Td(html.Code(t["prueba"])), html.Td(t.get("descripcion") or ""),
                             html.Td(t.get("resultado") or placeholder("sin registrar"))]) for t in tests]),
    ], className="tests"), style={"overflowX": "auto"}) if tests else html.P("Sin inventario de pruebas.")
    card = data.content("MODEL_CARD.md")
    return html.Details([
        html.Summary([html.Div([eyebrow("Ficha técnica"), html.H2("La capa CCI, por dentro", className="disp")],
                               className="insight")]),
        html.Div([
            html.Div([html.H3("Procedencia"), dcc.Markdown(prov or "Sin procedencia.", className="md")]),
            html.Div([html.H3("Pseudocódigo"), html.Pre(o.get("pseudocode") or "—"),
                      html.P(f"Parámetros: N̄ = {layer.get('reduced_size', '—')}², {layer.get('filters', '—')} filtros, "
                             f"ε = {layer.get('eps', '—')}, κ = {layer.get('kappa', '—')}, p inicial = {layer.get('p_init', '—')}.")]),
            html.Div([html.H3("Invarianza a las ganancias de canal"), dcc.Markdown(PROOF, mathjax=True, className="md")]),
            html.Div([html.H3("Pruebas unitarias"), test_table,
                      html.P("Resultado desde outputs/tests/junit.xml (pytest --junitxml).", className="small-note")]),
            html.Div([html.H3("Ficha del modelo"), dcc.Markdown(card, className="md") if card else html.P("Falta MODEL_CARD.md.")]),
        ], className="tech-body"),
    ], className="card tech")


def layout(**_):
    return html.Div([
        html.Div(html.Div([
            html.Span("Modelos", className="eyebrow"),
            html.H1(["8 arquitecturas, 2 baselines,", html.Br(), " una sola tabla de líderes."], className="disp page-title"),
            html.P("Mismos datos, misma partición por lesión, misma métrica: exactitud balanceada, para que el nevus no infle "
                   "los resultados.", className="page-lead"),
        ]), className="page-head"),
        podium(),
        leaderboard_card(),
        ours_card(),
        scatter_card(),
        html.Div([confusion_card(), calibration_card()], className="duo"),
        technical(),
    ], className="wrap page")

