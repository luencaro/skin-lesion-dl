"""Modelos: tabla de líderes, predicción en vivo, diagnósticos, comparación global, modelo original y ficha.

Mientras no haya resultados de entrenamiento (models/index.json con status "pendiente"), cada
sección que depende de ellos muestra un aviso de estado pendiente; el resto funciona.
"""

from __future__ import annotations

import base64
import io

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from core import data, inference
from core.components import error_alert, fmt, kpi, pending, section, static_graph
from core.config import (
    CLASS_COLORS,
    CLASS_NAMES,
    FAMILY_COLORS,
    FONT,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_MB,
    TITLE,
    UI,
)
from dash import (
    ALL,
    Input,
    Output,
    State,
    callback,
    ctx,
    dash_table,
    dcc,
    html,
    no_update,
)
from dash.dash_table.Format import Format, Scheme
from PIL import Image

dash.register_page(__name__, path="/modelos", name="Modelos", order=2, title=f"Modelos · {TITLE}")

PENDING_MSG = [
    html.P("Aún no hay corridas definitivas ni modelos exportados. Cuando existan, esta página se llena sola al ejecutar:"),
    html.Pre("python -m src.final --model <modelo>\npython -m src.export --all --fp16\npython -m src.export_dashboard",
             className="code-block"),
]
SHORT_PENDING = html.P("Se llena al ejecutar los comandos del aviso superior.", className="mb-0")
DISCLAIMER = dbc.Alert([html.Strong("No es una herramienta diagnóstica. "),
                        ("Resultado de un modelo de investigación entrenado en HAM10000; no sustituye la valoración de un "
                         "dermatólogo ni debe orientar decisiones clínicas.")], color="danger", className="disclaimer")


# ---------------------------------------------------------------------------
# Tabla de líderes
# ---------------------------------------------------------------------------


def leaderboard_section():
    board = data.leaderboard()
    if not board or not board.get("rows"):
        body = pending("Tabla de líderes pendiente", SHORT_PENDING)
    else:
        num = Format(precision=3, scheme=Scheme.fixed)
        cols = [{"name": "Modelo", "id": "modelo"}, {"name": "Familia", "id": "familia"},
                {"name": "Corridas", "id": "corridas", "type": "numeric"},
                {"name": "BA media", "id": "ba_media", "type": "numeric", "format": num},
                {"name": "± DE", "id": "ba_std", "type": "numeric", "format": num},
                {"name": "ECE", "id": "ece_media", "type": "numeric", "format": num},
                {"name": "Parámetros (M)", "id": "params_m", "type": "numeric", "format": Format(precision=1, scheme=Scheme.fixed)},
                {"name": "GFLOPs", "id": "gflops", "type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)},
                {"name": "Latencia CPU (ms)", "id": "latencia_ms", "type": "numeric", "format": Format(precision=1, scheme=Scheme.fixed)}]
        rows = [{**r, "params_m": r["params_total"] / 1e6 if r.get("params_total") else None} for r in board["rows"]]
        body = [dash_table.DataTable(
            id="leaderboard", columns=cols, data=rows, sort_action="native", filter_action="native",
            sort_by=[{"column_id": "ba_media", "direction": "desc"}], page_action="none", style_as_list_view=True,
            style_table={"overflowX": "auto", "minWidth": "100%"},
            style_cell={"fontFamily": FONT, "padding": "10px 8px", "minWidth": "84px", "backgroundColor": "transparent",
                        "color": UI["dark"], "textAlign": "right"},
            style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "left"} for c in ("modelo", "familia")],
            style_header={"fontWeight": 600, "backgroundColor": "rgba(15, 94, 110, 0.10)", "whiteSpace": "normal"},
            style_data_conditional=[{"if": {"filter_query": '{familia} = "Original (CCI)"'}, "fontWeight": 700}],
            filter_options={"case": "insensitive", "placeholder_text": "filtrar…"},
        ), html.P(f"Exactitud balanceada (BA) en {board.get('split', 'test')}: media ± desviación estándar entre corridas "
                  f"(tabla maestra: {board.get('source')}). Ordena tocando el encabezado; filtra en la fila superior "
                  "(p. ej. CNN en Familia o >0.7 en BA).", className="muted small")]
    return section("lideres", "Tabla de líderes", body)


# ---------------------------------------------------------------------------
# Predicción en vivo
# ---------------------------------------------------------------------------


def _model_options():
    opts = []
    for a in data.available_models():
        tag = "" if a["live"] else " (precalculado)"
        opts.append({"label": f"{a['model']} · {a['family']}{tag}", "value": a["model"]})
    return opts


def prediction_section():
    options = _model_options()
    if not options:
        body = [pending("Predicción en vivo pendiente", SHORT_PENDING), DISCLAIMER]
        return section("prediccion", "Predicción en vivo", body)
    gallery = html.Div([html.Button(html.Img(src=f"/media/examples/{ex['file']}", alt=f"Ejemplo de {ex['label']}",
                                             className="example-img"),
                                    id={"type": "example", "id": ex["image_id"]}, n_clicks=0, className="example-btn",
                                    title=f"{ex['label']} ({ex['image_id']})", **{"aria-label": f"Usar ejemplo de {ex['label']}"})
                        for ex in data.examples()], className="example-grid")
    controls = dbc.Row([
        dbc.Col([dbc.Label("Modelo", html_for="pred-model"),
                 dcc.Dropdown(id="pred-model", options=options, value=options[0]["value"], clearable=False,
                              searchable=False, className="touch-dropdown")], xs=12, md=5),
        dbc.Col([dbc.Label("Cobertura del conjunto conforme"),
                 dbc.RadioItems(id="pred-alpha", options=[{"label": "90 %", "value": "0.1"}, {"label": "95 %", "value": "0.05"}],
                                value="0.1", inline=True, inputClassName="btn-check",
                                labelClassName="btn btn-outline-primary btn-touch", labelCheckedClassName="active")], xs=7, md=4),
        dbc.Col([dbc.Label("Mapa CAM"), dbc.Switch(id="pred-cam", value=True, label="superponer", className="touch-switch")],
                xs=5, md=3),
    ], className="g-2")
    upload = dcc.Upload(id="pred-upload", accept="image/jpeg,image/png", max_size=MAX_UPLOAD_BYTES, multiple=False,
                        children=html.Div([html.Strong("Toca para elegir una imagen"), html.Span(f" JPEG o PNG, hasta {MAX_UPLOAD_MB} MB")]),
                        className="upload-box", className_active="upload-active", className_reject="upload-reject")
    body = [DISCLAIMER, controls, dbc.Row([
        dbc.Col([upload, html.Div("O usa un ejemplo de test de HAM10000:", className="muted small mt-2"), gallery], md=5),
        dbc.Col(dcc.Loading(html.Div(id="pred-output", className="pred-output",
                                     children=html.P("Elige una imagen o un ejemplo para ver la predicción.", className="muted")),
                            type="dot", color="var(--color-primary)"), md=7),
    ], className="g-3 mt-1"), dcc.Store(id="pred-source", storage_type="memory")]
    return section("prediccion", "Predicción en vivo", body,
                   lead="Probabilidades calibradas por temperatura, conjunto de predicción conforme, mapa CAM y aviso si la "
                        "imagen se aleja de las de entrenamiento. La imagen se procesa en memoria y no se guarda.")


@callback(Output("pred-source", "data"), Input("pred-upload", "contents"), Input({"type": "example", "id": ALL}, "n_clicks"),
          prevent_initial_call=True)
def choose_source(contents, _clicks):
    trig = ctx.triggered_id
    if isinstance(trig, dict) and trig.get("type") == "example":
        if not any(_clicks or []):
            return no_update
        return {"kind": "example", "id": trig["id"]}
    if trig == "pred-upload" and contents:
        return {"kind": "upload"}
    return no_update


@callback(Output("pred-output", "children"), Input("pred-source", "data"), Input("pred-model", "value"),
          Input("pred-alpha", "value"), Input("pred-cam", "value"), State("pred-upload", "contents"),
          prevent_initial_call=True)
def run_prediction(source, model, alpha, show_cam, contents):
    if not source or not model:
        return no_update
    try:
        meta = data.model_metadata(model) or {}
        live = bool(meta.get("live"))
        if source["kind"] == "example":
            if live:
                path = data.example_path(source["id"])
                if path is None:
                    raise inference.InvalidImage("El ejemplo no está disponible.")
                res = inference.predict(model, Image.open(path))
            else:
                res = inference.predict_precomputed(model, source["id"])
        else:
            if not live:
                raise inference.InvalidImage(f"{model} supera el límite de tamaño del despliegue y solo se muestra con los "
                                             "resultados precalculados de la galería de ejemplos.")
            res = inference.predict(model, inference.open_image(inference.decode_upload(contents)))
        return render_result(res, alpha, bool(show_cam), source)
    except inference.InvalidImage as e:
        return error_alert(str(e))
    except Exception as e:  # noqa: BLE001 - el usuario ve un aviso, no un error de servidor
        return error_alert(f"No se pudo calcular la predicción ({type(e).__name__}). Prueba con otra imagen o modelo.")


def _png_data_uri(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def cam_figure(image: np.ndarray, cam, show: bool):
    h, w = image.shape[:2]
    fig = go.Figure()
    fig.add_layout_image(source=_png_data_uri(image), x=0, y=0, sizex=w, sizey=h, xref="x", yref="y",
                         sizing="stretch", layer="below")
    if cam is not None and show:
        cam = np.asarray(cam)
        ch, cw = cam.shape
        fig.add_heatmap(z=cam, x=(np.arange(cw) + 0.5) * w / cw, y=(np.arange(ch) + 0.5) * h / ch, zsmooth="best",
                        colorscale="Cividis", opacity=0.45, showscale=False, zmin=0, zmax=1, hoverinfo="skip")
    fig.update_xaxes(visible=False, range=[0, w])
    fig.update_yaxes(visible=False, range=[h, 0], scaleanchor="x")
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    return fig


def render_result(res: dict, alpha: str, show_cam: bool, source: dict):
    classes, probs = res["classes"], np.asarray(res["probs"])
    in_set = set(res["sets"].get(alpha, res["sets"].get(str(float(alpha)), [])))
    order = np.argsort(probs)
    bars = go.Figure(go.Bar(
        y=[f"{'● ' if classes[i] in in_set else ''}{classes[i]}" for i in order], x=100 * probs[order], orientation="h",
        marker_color=[CLASS_COLORS[classes[i]] for i in order], text=[fmt(probs[i], 1, pct=True) for i in order],
        customdata=[CLASS_NAMES[classes[i]] for i in order], hovertemplate="%{customdata}: %{x:.1f} %<extra></extra>"))
    bars.update_layout(title="Probabilidad calibrada (%) · ● = en el conjunto conforme", xaxis={"range": [0, 112]},
                       showlegend=False, margin={"l": 70, "r": 10, "t": 40, "b": 30})
    bars.update_traces(textposition="outside", cliponaxis=False, textfont={"color": UI["dark"]})
    cov = 100 * (1 - float(alpha))
    chips = [dbc.Badge(c, className="set-chip", style={"backgroundColor": CLASS_COLORS[c],
                                                         "color": UI["dark"] if c in ("df", "akiec", "vasc") else UI["light"]})
             for c in classes if c in in_set]
    ood = res.get("ood")
    ood_note = None
    if ood and ood["out_of_domain"]:
        ood_note = dbc.Alert(["Imagen fuera de dominio: se parece poco a las de entrenamiento (distancia ",
                              fmt(ood["distance"]), " > umbral ", fmt(ood["threshold"]), "). La predicción no es fiable."],
                             color="warning", className="mt-2")
    elif ood:
        ood_note = html.P(f"Dentro del dominio de train (distancia {fmt(ood['distance'])} ≤ umbral {fmt(ood['threshold'])}).",
                          className="muted small")
    meta = [kpi(res["pred"], CLASS_NAMES.get(res["pred"], ""), "clase más probable"),
            kpi(f"{len(in_set)}", f"clases en el conjunto al {cov:.0f} %", "predicción conforme (LAC)", tone="accent"),
            kpi(fmt(res["latency_ms"], 0) + " ms" if res.get("latency_ms") is not None else "—",
                "latencia en CPU" if res["source"] == "en vivo" else "resultado precalculado", f"T = {fmt(res['temperature'], 2)}")]
    cam_part = (static_graph(cam_figure(res["image"], res.get("cam"), show_cam), 300) if res.get("image") is not None else None)
    cam_note = None if res.get("cam") is not None else html.P("Este modelo no tiene pooling global: sin CAM.", className="muted small")
    return html.Div([
        dbc.Row([dbc.Col(m, xs=4) for m in meta], className="g-2"),
        html.Div(chips, className="set-chips", **{"aria-label": "Conjunto de predicción"}),
        dbc.Row([dbc.Col([cam_part, cam_note], md=5), dbc.Col(static_graph(bars, 300), md=7)], className="g-2"),
        ood_note,
        html.P(f"Modelo {res['model']} · {res['source']}" + (f" · ejemplo {source.get('id')}" if source.get("kind") == "example" else ""),
               className="muted small"),
    ], className="fade-in")


# ---------------------------------------------------------------------------
# Diagnósticos del modelo elegido
# ---------------------------------------------------------------------------


def diagnostics_section():
    models = [m for m in data.models_index().get("models_with_runs", []) if data.diagnostics(m)]
    if not models:
        return section("diagnosticos", "Diagnósticos del modelo", pending("Diagnósticos pendientes", SHORT_PENDING))
    controls = dbc.Row([
        dbc.Col(dcc.Dropdown(id="diag-model", options=models, value=models[0], clearable=False, searchable=False,
                             className="touch-dropdown"), xs=12, md=5),
        dbc.Col(dbc.RadioItems(id="diag-kind", options=[{"label": x, "value": x} for x in ("Confusión", "ROC", "PR", "Fiabilidad")],
                               value="Confusión", inline=True, inputClassName="btn-check",
                               labelClassName="btn btn-outline-primary btn-touch", labelCheckedClassName="active"), xs=12, md=7),
    ], className="g-2")
    return section("diagnosticos", "Diagnósticos del modelo", dcc.Loading(html.Div(id="diag-output")), controls=controls,
                   lead="Predicciones de test de todas las corridas definitivas del modelo, agrupadas.")


@callback(Output("diag-output", "children"), Input("diag-model", "value"), Input("diag-kind", "value"))
def show_diagnostics(model, kind):
    d = data.diagnostics(model) if model else None
    if not d:
        return pending("Sin diagnósticos", "Este modelo no tiene predicciones de test.")
    classes = d["classes"]
    if kind == "Confusión":
        z = np.asarray(d["confusion_norm"])
        fig = go.Figure(go.Heatmap(z=z, x=classes, y=classes, colorscale="Cividis", zmin=0, zmax=1, text=np.round(z, 2),
                                   texttemplate="%{text}", colorbar={"title": "recall"}))
        fig.update_layout(title="Matriz de confusión normalizada por fila", xaxis_title="Predicha", yaxis_title="Real",
                          yaxis={"autorange": "reversed"})
    elif kind == "ROC":
        fig = go.Figure([go.Scatter(x=d["roc"][c]["fpr"], y=d["roc"][c]["tpr"], mode="lines", name=f"{c} ({fmt(d['roc'][c]['auc'])})",
                                    line={"color": CLASS_COLORS[c], "width": 2}) for c in classes])
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dot", "color": UI["dark"], "width": 1}, showlegend=False)
        fig.update_layout(title="ROC uno contra el resto (AUC)", xaxis_title="1 − especificidad", yaxis_title="Sensibilidad")
    elif kind == "PR":
        fig = go.Figure([go.Scatter(x=d["pr"][c]["recall"], y=d["pr"][c]["precision"], mode="lines",
                                    name=f"{c} ({fmt(d['pr'][c]['ap'])})", line={"color": CLASS_COLORS[c], "width": 2})
                         for c in classes])
        fig.update_layout(title="Precisión–exhaustividad (AP)", xaxis_title="Exhaustividad", yaxis_title="Precisión")
    else:
        rel = d["reliability"]
        fig = go.Figure([go.Bar(x=[r["confianza"] for r in rel], y=[r["exactitud"] for r in rel], width=1 / 15,
                                marker_color=UI["primary"], name="exactitud", text=[r["n"] for r in rel])])
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dot", "color": UI["dark"]}, name="calibración perfecta")
        fig.update_layout(title="Diagrama de fiabilidad (n por intervalo sobre la barra)", xaxis_title="Confianza",
                          yaxis_title="Exactitud", xaxis={"range": [0, 1]}, yaxis={"range": [0, 1.05]})
        fig.update_traces(textposition="outside", selector={"type": "bar"})
    return static_graph(fig, 420)


# ---------------------------------------------------------------------------
# Comparación global: diagrama de diferencia crítica y frontera de Pareto
# ---------------------------------------------------------------------------


def cd_figure(cd: dict):
    ranks = sorted(cd["ranks"].items(), key=lambda kv: kv[1])
    k, crit = cd["k"], cd["cd"]
    fig = go.Figure()
    for i, (m, r) in enumerate(ranks):
        y = -(i % 2) * 0.35 - 0.3
        fig.add_scatter(x=[r, r], y=[0, y], mode="lines", line={"color": UI["dark"], "width": 1}, showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=[r], y=[y], mode="markers+text", text=[f"{m} ({fmt(r, 2)})"], textposition="bottom center",
                        marker={"size": 9, "color": FAMILY_COLORS.get(_family(m), UI["primary"])}, showlegend=False)
    # grupos sin diferencia significativa (rangos a menos de la diferencia crítica)
    vals = [r for _, r in ranks]
    level, last_end = 0.18, -1
    for i in range(len(vals)):
        j = max(j for j in range(i, len(vals)) if vals[j] - vals[i] < crit)
        if j > i and j > last_end:
            fig.add_scatter(x=[vals[i], vals[j]], y=[level, level], mode="lines", line={"color": UI["accent"], "width": 5},
                            showlegend=False, hoverinfo="skip")
            level += 0.1
            last_end = j
    fig.add_scatter(x=[1, 1 + crit], y=[0.7, 0.7], mode="lines+text", text=["", f"CD = {fmt(crit, 2)}"],
                    textposition="middle right", line={"color": UI["alert"], "width": 3}, showlegend=False)
    fig.update_xaxes(range=[0.8, k + 0.2], title="Rango medio (mayor = mejor)", dtick=1)
    fig.update_yaxes(visible=False, range=[-1.4, 0.9])
    fig.update_layout(title=f"Diagrama de diferencia crítica (Nemenyi, α = {cd['alpha']}, {cd['n_runs']} corridas)")
    return fig


def _family(model: str) -> str:
    return next((r["familia"] for r in (data.leaderboard() or {}).get("rows", []) if r["modelo"] == model), "CNN")


def global_section():
    g = data.global_comparison()
    if not g:
        return section("global", "Diferencia crítica y frontera de Pareto", pending("Comparación global pendiente", SHORT_PENDING))
    parts = []
    if g.get("cd"):
        parts.append(static_graph(cd_figure(g["cd"]), 360))
        parts.append(html.P("Barras naranjas: modelos cuyos rangos medios no difieren más que la diferencia crítica.", className="muted small"))
    else:
        parts.append(pending("Diagrama CD pendiente", "Necesita al menos 3 modelos con corridas pareadas."))
    pareto = g.get("pareto") or {}
    if pareto:
        labels = {"gflops": "GFLOPs", "latencia_ms": "Latencia CPU (ms)", "params_total": "Parámetros"}
        parts.append(dbc.RadioItems(id="pareto-cost", options=[{"label": labels.get(c, c), "value": c} for c in pareto],
                                    value=next(iter(pareto)), inline=True, inputClassName="btn-check",
                                    labelClassName="btn btn-outline-primary btn-touch", labelCheckedClassName="active"))
        parts.append(html.Div(id="pareto-output"))
    return section("global", "Diferencia crítica y frontera de Pareto", parts)


@callback(Output("pareto-output", "children"), Input("pareto-cost", "value"))
def show_pareto(cost):
    rows = ((data.global_comparison() or {}).get("pareto") or {}).get(cost) or []
    if not rows:
        return None
    fig = go.Figure()
    for fam in dict.fromkeys(r["familia"] for r in rows):
        sub = [r for r in rows if r["familia"] == fam]
        fig.add_scatter(x=[r[cost] for r in sub], y=[r["ba_media"] for r in sub], mode="markers+text", name=fam,
                        text=[r["modelo"] for r in sub], textposition="top center",
                        marker={"size": 11, "color": FAMILY_COLORS.get(fam, UI["primary"])})
    front = sorted([r for r in rows if r["frontera"]], key=lambda r: r[cost])
    fig.add_scatter(x=[r[cost] for r in front], y=[r["ba_media"] for r in front], mode="lines", name="frontera",
                    line={"color": UI["alert"], "width": 2, "dash": "dot"})
    fig.update_layout(title="Frontera de Pareto: exactitud balanceada frente a costo", yaxis_title="BA media",
                      xaxis_title={"gflops": "GFLOPs (escala log)", "latencia_ms": "Latencia CPU (ms)",
                                   "params_total": "Parámetros (escala log)"}.get(cost, cost),
                      xaxis_type="log" if cost in ("gflops", "params_total") else "linear")
    return static_graph(fig, 380)


# ---------------------------------------------------------------------------
# Modelo original y ficha
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


def original_section():
    o = data.original_section()
    prov = (o.get("provenance_md") or "").split("\n", 1)[-1]
    tests = o.get("tests") or []
    test_table = dash_table.DataTable(
        columns=[{"name": "Prueba", "id": "prueba"}, {"name": "Qué comprueba", "id": "descripcion"},
                 {"name": "Resultado", "id": "resultado"}],
        data=[{**t, "resultado": t.get("resultado") or "sin registrar"} for t in tests], page_size=8,
        style_as_list_view=True, style_table={"overflowX": "auto"},
        style_cell={"fontFamily": FONT, "whiteSpace": "normal", "textAlign": "left", "padding": "8px",
                    "backgroundColor": "transparent", "color": UI["dark"], "maxWidth": "420px"},
        style_header={"fontWeight": 600, "backgroundColor": "rgba(15, 94, 110, 0.10)"},
        style_data_conditional=[{"if": {"filter_query": '{resultado} = "fallida"'}, "color": UI["alert"], "fontWeight": 700}],
    ) if tests else html.P("Sin inventario de pruebas.", className="muted")
    layer = o.get("layer") or {}
    backbone = o.get("backbone")
    verdict = (pending("Veredicto pendiente", [
        html.P("El diseño se considera exitoso si la diferencia de exactitud balanceada entre vidir_molemax y el resto se "
               "reduce en al menos δ = 0,02 (límite inferior del IC 95 % mayor que cero), si se cumple la predicción 1 y si "
               "la exactitud balanceada global no es peor que la del backbone (TOST con margen ±δ)."),
        html.P(f"Backbone: {backbone or 'pendiente (regla del capítulo 24: mejor modelo en validación del Entregable 2)'}."),
    ], color="info") if not o.get("verdict") else dbc.Alert(o["verdict"], color="success"))
    robust = (pending("Robustez pendiente", "Se calcula con python -m src.robustness --model cci tras el entrenamiento.", color="info")
              if not o.get("robustness") else dash_table.DataTable(
                  columns=[{"name": c, "id": c} for c in o["robustness"]["columns"]],
                  data=[dict(zip(o["robustness"]["columns"], r)) for r in o["robustness"]["data"]], page_size=10,
                  style_table={"overflowX": "auto"}))
    tabs = dbc.Tabs([
        dbc.Tab(dcc.Markdown(prov or "Sin procedencia.", className="md"), label="Procedencia", tab_id="t-prov"),
        dbc.Tab([html.Pre(o.get("pseudocode") or "—", className="code-block"),
                 html.P(f"Parámetros: N̄ = {layer.get('reduced_size', 56)}², {layer.get('filters', 16)} filtros, "
                        f"ε = {layer.get('eps')}, κ = {layer.get('kappa')}, p inicial = {layer.get('p_init')}.", className="muted small")],
                label="Pseudocódigo", tab_id="t-pseudo"),
        dbc.Tab(dcc.Markdown(PROOF, mathjax=True, className="md"), label="Demostración", tab_id="t-proof"),
        dbc.Tab([test_table, html.P("Resultado desde outputs/tests/junit.xml (pytest --junitxml).", className="muted small")],
                label="Pruebas", tab_id="t-tests"),
        dbc.Tab(robust, label="Robustez", tab_id="t-rob"),
        dbc.Tab(verdict, label="Veredicto", tab_id="t-verdict"),
    ], active_tab="t-prov", className="touch-tabs")
    return section("original", "Modelo original: capa CCI", tabs,
                   lead="Constancia de color aprendible e invariante a las ganancias de canal, antepuesta al backbone.")


def model_card_section():
    md = data.content("MODEL_CARD.md")
    body = dcc.Markdown(md, className="md") if md else pending("Ficha pendiente", "Falta MODEL_CARD.md en dashboard/data/content/.")
    return section("ficha-modelo", "Ficha del modelo", body)


def layout(**_):
    status = data.models_index().get("status", "pendiente")
    banner = (pending("Resultados de entrenamiento pendientes", PENDING_MSG) if status != "listo" else None)
    return html.Div([
        html.H1("Modelos", className="page-title"),
        html.P("Comparación de los modelos del benchmark y del modelo original, y predicción con los modelos exportados a ONNX.",
               className="page-lead"),
        banner,
        leaderboard_section(),
        prediction_section(),
        diagnostics_section(),
        global_section(),
        original_section(),
        model_card_section(),
    ])
