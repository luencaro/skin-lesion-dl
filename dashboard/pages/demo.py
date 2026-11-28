"""Demo en vivo (design/Demo.dc.html y MovilDemo.dc.html).

Estados del resultado: vacío -> analizando (running= del callback) -> resultado, o resultado con el
aviso fuera de distribución. La imagen cargada vive solo en el navegador (contents de dcc.Upload,
memoria de la pestaña) y en la memoria del proceso mientras dura la inferencia: no se escribe en
disco ni se registra, y se descarta al recargar.
"""

from __future__ import annotations

import logging

import dash
import numpy as np
from core import data, inference
from core.components import (
    alert,
    class_dot,
    disclaimer,
    empty_state,
    fmt_num,
    fmt_pct,
    icon,
    placeholder,
    prob_row,
)
from core.config import (
    ASSETS_DIR,
    CLASS_NAMES,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    DEFAULT_ALPHA,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_MB,
    ORIGINAL_MODEL,
    TITLE,
    model_name,
)
from dash import (
    ALL,
    ClientsideFunction,
    Input,
    Output,
    State,
    callback,
    clientside_callback,
    dcc,
    html,
    no_update,
)
from PIL import Image

dash.register_page(__name__, path="/demo", name="Demo en vivo", order=1, title=f"Demo en vivo · {TITLE}")
log = logging.getLogger(__name__)

DEMO_CLASSES = ["mel", "nv", "bcc", "vasc"]          # 4 ejemplos dermatoscópicos de test (uno por clase)
OOD_SAMPLE = {"key": "foto_normal", "file": "samples/foto_no_dermatoscopica.jpg", "label": "Foto normal"}


def samples() -> list[dict]:
    """5 ejemplos: 4 imágenes de test de HAM10000 y una foto no dermatoscópica (aviso OOD)."""
    out = []
    exs = data.examples()
    for c in DEMO_CLASSES:
        ex = next((e for e in exs if e["label"] == c), None)
        if ex:
            out.append({"key": ex["image_id"], "src": f"/media/examples/{ex['file']}", "label": f"Ejemplo {len(out) + 1}",
                        "alt": f"Ejemplo {len(out) + 1}: imagen de test de HAM10000 ({CLASS_NAMES[c]})", "kind": "example"})
    if (ASSETS_DIR / OOD_SAMPLE["file"]).exists():
        out.append({"key": OOD_SAMPLE["key"], "src": dash.get_asset_url(OOD_SAMPLE["file"]), "label": OOD_SAMPLE["label"],
                    "alt": "Foto normal, no dermatoscópica (sintética), para ver el aviso fuera de distribución",
                    "kind": "photo"})
    return out


def model_display(m: str) -> str:
    if m == ORIGINAL_MODEL:
        bb = data.ours().get("backbone_name")
        return f"Nuestro modelo (CCI + {bb})" if bb else "Nuestro modelo (CCI)"
    return model_name(m)


def default_model() -> str | None:
    avail = [m for m in data.demo_models() if m["available"]]
    return avail[0]["id"] if avail else None


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------


def input_column(models: list[dict], current: str | None):
    upload = dcc.Upload(
        id="demo-upload", accept="image/*", max_size=MAX_UPLOAD_BYTES, multiple=False,
        className="drop", className_active="drop drop-active", className_reject="drop drop-reject",
        children=[
            html.Span(icon("upload", 30), className="drop-ic"),
            html.Span([
                html.Span("Arrastra una imagen dermatoscópica", className="disp drop-title"),
                html.Span(["o ", html.U("búscala en tu equipo"), f" · JPG o PNG, hasta {MAX_UPLOAD_MB} MB"], className="drop-sub"),
                html.Strong("Toma o elige una foto", className="drop-title-m"),
                html.Span("Dermatoscópica · no se guarda", className="drop-sub-m"),
            ], className="drop-text"),
        ],
    )
    thumbs = [html.Button([html.Img(src=s["src"], alt=""), html.Span(s["label"])], id={"type": "sample", "id": s["key"]},
                          n_clicks=0, className="thumb", title=s["alt"], **{"aria-label": s["alt"]})
              for s in samples()]
    pills = []
    for m in models:
        label = ["Nuestro modelo", html.Span("CCI", className="tag")] if m["ours"] else [m["name"]]
        if m["available"] and not m["live"]:
            label.append(html.Span("ejemplos", className="sub"))
        pills.append(html.Button(label, id={"type": "model-pill", "id": m["id"]}, n_clicks=0,
                                 className="pill on" if m["id"] == current else "pill", disabled=not m["available"],
                                 title=None if m["available"] else "Aún no exportado a ONNX",
                                 **{"aria-pressed": "true" if m["id"] == current else "false"}))
    n_avail = sum(m["available"] for m in models)
    note = (html.Span("Ningún modelo está exportado todavía: las pastillas se activan al ejecutar python -m src.export "
                      "y python -m src.export_dashboard.", className="model-note") if not n_avail else
            html.Span("«ejemplos»: el modelo supera el límite de tamaño del despliegue y solo muestra resultados "
                      "precalculados de los ejemplos.", className="model-note")
            if any(m["available"] and not m["live"] for m in models) else None)
    return html.Div([
        upload,
        html.Div([html.Span("O prueba sin subir nada", className="samples-label"),
                  html.Div(thumbs, className="samples", role="group", **{"aria-label": "Imágenes de ejemplo"})],
                 className="stack", style={"gap": 14}),
        html.Div([
            html.Div([html.Span("Modelo"), dcc.Link("Comparar modelos →", href="/modelos")], className="model-card-head"),
            html.Div(pills, className="pills", role="group", **{"aria-label": "Modelo"}),
            note,
        ], className="card model-card"),
    ], className="demo-in")


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------


def empty_result():
    return html.Div(empty_state("Resultados en camino", "Sube una imagen o elige un ejemplo. Verás la clase más probable, "
                                "dónde miró el modelo y qué no puede descartar."), className="card result-card empty-card")


def loading_panel():
    return html.Div([
        html.Div([html.Img(id="demo-loading-img", alt="Imagen en análisis"), html.Div(className="scan fast")],
                 className="photo loading-img"),
        html.Div([
            html.Span("Analizando…", className="disp", style={"fontSize": 22, "fontWeight": 700}),
            html.Ol([html.Li([class_dot("df"), "Imagen recibida"]),
                     html.Li([html.Span(className="dot", style={"background": "var(--accent)"}),
                              html.Span("Corrigiendo color con CCI", id="demo-step2")], className="now"),
                     html.Li([html.Span(className="dot", style={"background": "var(--line)"}),
                              "Prediciendo y generando Grad-CAM"], className="next")]),
            html.Div(className="skel", style={"height": 52, "marginTop": 12}),
            html.Div(className="skel", style={"height": 14}),
            html.Div(className="skel", style={"height": 14, "width": "80%"}),
            html.Div(className="skel", style={"height": 14, "width": "60%"}),
        ], className="loading-steps"),
    ], className="card loading-card", id="demo-loading", style={"display": "none"}, role="status",
        **{"aria-live": "polite", "aria-label": "Analizando la imagen"})


def confidence_label(p: float, ood: bool) -> str:
    if ood:
        return "no fiable"
    return "alta" if p >= CONFIDENCE_HIGH else "media" if p >= CONFIDENCE_MEDIUM else "baja"


def render_result(res: dict, model: str, opacity: int):
    classes, probs = res["classes"], np.asarray(res["probs"], dtype=float)
    order = np.argsort(-probs)
    top = classes[int(order[0])]
    p_top = float(probs[order[0]])
    ood = res.get("ood") or None
    is_ood = bool(ood and ood.get("out_of_domain"))
    size = res["image"].shape[0] if res.get("image") is not None else 224
    heat = inference.heatmap_png(res.get("cam"), size)

    sets = res.get("sets") or {}
    in_set = sets.get(DEFAULT_ALPHA)
    if in_set is not None:
        in_set = sorted(in_set, key=lambda c: -probs[classes.index(c)])
        coverage = fmt_num(100 * (1 - float(DEFAULT_ALPHA)))
        pset = html.Div([
            html.Strong("Conjunto de predicción"),
            html.Span(f"Clases que el modelo no puede descartar (cobertura objetivo {coverage} %)"),
            html.Div([html.Span([class_dot(c), CLASS_NAMES[c]], className="set-chip") for c in in_set], className="set-chips")
            if in_set else html.Span("Conjunto vacío: ninguna clase alcanza el umbral de calibración, una señal más de que la "
                                     "imagen se aleja de lo que el modelo conoce.", className="model-note"),
        ], className="pset")
    else:
        pset = html.Div([html.Strong("Conjunto de predicción"),
                         html.Span(["Clases que el modelo no puede descartar: ", placeholder("[ calibración pendiente ]")])],
                        className="pset")

    img_layers = [html.Img(src=inference.png_data_uri(res["image"]), alt="Imagen analizada (224 × 224, como la ve el modelo)")]
    if heat:
        img_layers.append(html.Img(src=heat, alt="", id="demo-heat", style={"opacity": opacity / 100}))
        img_layers.append(html.Span("Grad-CAM", className="badge-dark"))
        controls = [
            html.Label([html.Span([html.Span("Opacidad del mapa de calor"),
                                   html.Span(f"{opacity} %", id="demo-opacity-val", className="num")]),
                        dcc.Input(id="demo-opacity", type="range", min=0, max=100, step=5, value=opacity,
                                  debounce=False)],
                       className="range-label"),
            html.Div([html.Span("Menos atención"), html.I(), html.Span("Más atención")], className="grad-legend",
                     **{"aria-hidden": "true"}),
        ]
    else:
        controls = [html.P("Este modelo no trae mapa de atención en su ONNX: el CAM sin gradientes necesita una cabeza "
                           "GAP → lineal, y el attention rollout de ViT-B/16 solo existe en el pipeline de entrenamiento.",
                           className="no-cam")]

    tagline = (f"En vivo · {fmt_num(res['latency_ms'])} ms en CPU" if res.get("latency_ms") is not None
               else "Resultado precalculado")
    main = html.Div([
        html.Div([html.Div(img_layers, className="photo cam-frame"), *controls], className="result-img-col"),
        html.Div([
            html.Div([html.Span("Clase predicha", className="kicker"), html.Span(tagline, className="tagline")],
                     className="pred-head"),
            html.Div([html.Span([class_dot(top, 16), html.Span(CLASS_NAMES[top], className="disp pred-name")]),
                      html.Span(f"con {model_display(model)}", className="pred-by")], className="pred-class"),
            html.Div([
                html.Div(html.Span(f"{fmt_num(100 * p_top)} %", className="num"), className="conf-ring",
                         style={"background": f"conic-gradient(var(--brand) {round(360 * p_top)}deg, var(--line) 0)"},
                         role="img", **{"aria-label": f"Confianza {fmt_pct(p_top)}"}),
                html.Div([html.Strong(f"Confianza {confidence_label(p_top, is_ood)}"),
                          html.Span("Probabilidad de la clase más alta" + (" (calibrada)" if res.get("temperature", 1) != 1 else ""))],
                         className="conf-text"),
            ], className="conf"),
            pset,
        ], className="result-info"),
    ], className="card result-main")
    rows = [prob_row(classes[int(i)], float(probs[int(i)])) for i in order]
    kids = []
    if is_ood:
        kids.append(alert("Está fuera de lo que el modelo conoce, así que el resultado no es fiable. Prueba con una imagen "
                          "tomada con dermatoscopio.", "Esta imagen no parece dermatoscópica."))
    kids += [main, html.Div([html.Strong("Probabilidad por clase"), *rows], className="card probs")]
    if ood is None:
        kids.append(html.P("La detección de imágenes fuera de distribución no está disponible para este modelo "
                           "(falta su referencia de embeddings).", className="result-meta"))
    return html.Div(kids, className="result pop")


def message_card(title: str, body: str, ic: str = "info"):
    return html.Div(empty_state(title, body, size=120, ic=ic), className="card result-card")


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------


def layout(**_):
    models = data.demo_models()
    current = default_model()
    if current is None:
        initial = message_card("Modelos en camino", "Aún no hay modelos exportados a ONNX. Cuando los haya, aquí verás la "
                               "clase más probable, dónde miró el modelo y qué no puede descartar.", "eye")
    else:
        initial = empty_result()
    return html.Div([
        html.Div([
            html.Div([html.Span("Demo en vivo", className="eyebrow"), html.H1("¿Qué ve el modelo?", className="disp page-title")]),
            html.Span([icon("lock", 16), "La imagen se analiza en memoria y no se guarda"], className="page-meta"),
        ], className="page-head"),
        html.Div([
            input_column(models, current),
            html.Div([loading_panel(), html.Div(initial, id="demo-output", **{"aria-live": "polite"})], className="demo-out"),
        ], className="demo-grid"),
        disclaimer(compact=True),
        dcc.Store(id="demo-source", storage_type="memory"),
        dcc.Store(id="demo-model", storage_type="memory", data=current),
        dcc.Store(id="demo-opacity-store", storage_type="memory", data=65),
        dcc.Store(id="demo-samples", data={s["key"]: {"src": s["src"]} for s in samples()}),
        dcc.Store(id="demo-model-names", data={m["id"]: model_display(m["id"]) for m in models}),
    ], className="wrap page demo")


clientside_callback(
    ClientsideFunction("demo", "pickSource"),
    Output("demo-source", "data"), Output({"type": "sample", "id": ALL}, "className"), Output("demo-loading-img", "src"),
    Input({"type": "sample", "id": ALL}, "n_clicks"), Input("demo-upload", "contents"),
    State("demo-samples", "data"), State({"type": "sample", "id": ALL}, "id"),
    prevent_initial_call=True,
)
clientside_callback(
    ClientsideFunction("demo", "pickModel"),
    Output("demo-model", "data"), Output({"type": "model-pill", "id": ALL}, "className"), Output("demo-step2", "children"),
    Output({"type": "model-pill", "id": ALL}, "aria-pressed"),
    Input({"type": "model-pill", "id": ALL}, "n_clicks"), State({"type": "model-pill", "id": ALL}, "id"),
    State("demo-model-names", "data"),
    prevent_initial_call=True,
)
clientside_callback(
    ClientsideFunction("demo", "opacity"),
    Output("demo-heat", "style"), Output("demo-opacity-val", "children"), Output("demo-opacity-store", "data"),
    Input("demo-opacity", "value"),
    prevent_initial_call=True,
)


def _predict(source: dict, model: str, contents: str | None) -> dict:
    meta = data.model_metadata(model) or {}
    live = bool(meta.get("live"))
    if source.get("kind") == "upload":
        if not live:
            raise inference.InvalidImage(f"{model_display(model)} supera el límite de tamaño del despliegue y solo se muestra "
                                         "con los resultados precalculados de los ejemplos. Elige otro modelo o un ejemplo.")
        return inference.predict(model, inference.open_image(inference.decode_upload(contents)))
    key = source.get("id")
    if key == OOD_SAMPLE["key"]:
        if not live:
            raise inference.InvalidImage(f"{model_display(model)} solo tiene resultados precalculados de las imágenes de "
                                         "HAM10000. Elige otro modelo para la foto normal.")
        with Image.open(ASSETS_DIR / OOD_SAMPLE["file"]) as im:
            return inference.predict(model, im.convert("RGB"))
    if live:
        path = data.example_path(key)
        if path is None:
            raise inference.InvalidImage("El ejemplo no está disponible.")
        with Image.open(path) as im:
            return inference.predict(model, im.convert("RGB"))
    return inference.predict_precomputed(model, key)


@callback(
    Output("demo-output", "children"),
    Input("demo-source", "data"), Input("demo-model", "data"),
    State("demo-upload", "contents"), State("demo-opacity-store", "data"),
    running=[(Output("demo-loading", "style"), {"display": "flex"}, {"display": "none"}),
             (Output("demo-output", "style"), {"display": "none"}, {"display": "block"})],
    prevent_initial_call=True,
)
def run_demo(source, model, contents, opacity):
    if not source:
        return no_update
    if not model:
        return message_card("Modelos en camino", "Aún no hay modelos exportados a ONNX, así que no se puede analizar la imagen.")
    try:
        res = _predict(source, model, contents)
        return render_result(res, model, int(opacity if opacity is not None else 65))
    except inference.InvalidImage as e:
        return html.Div(alert(str(e), "No se pudo analizar la imagen.", kind="error"), className="result pop")
    except Exception as e:  # noqa: BLE001 - el usuario ve un aviso, no un error de servidor
        log.warning("Fallo de inferencia con %s: %s", model, type(e).__name__)
        return html.Div(alert(f"No se pudo calcular la predicción ({type(e).__name__}). Prueba con otra imagen o modelo.",
                              "Algo falló.", kind="error"), className="result pop")
