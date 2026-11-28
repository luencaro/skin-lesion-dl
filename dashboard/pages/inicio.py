"""Inicio (design/Main.dc.html): hero, cifras, cómo funciona, capa CCI, CTA y aviso."""

from __future__ import annotations

import logging
from functools import lru_cache

import dash
from core import data, inference
from core.components import (
    class_dot,
    disclaimer,
    dot,
    fmt_num,
    fmt_pct,
    fmt_signed,
    icon,
    placeholder,
    stat_card,
)
from core.config import (
    BASELINES,
    BENCHMARK_MODELS,
    CLASS_COLORS,
    CLASS_NAMES,
    CLASS_ORDER,
    ORIGINAL_MODEL,
    TITLE,
    model_name,
)
from dash import dcc, html
from PIL import Image

dash.register_page(__name__, path="/", name="Inicio", order=0, title=TITLE,
                   redirect_from=["/contexto"])
log = logging.getLogger(__name__)

TINTS = [("#FFB25C", .55, "Cámara A · cálida", "A corregida"), ("#7FA8FF", .55, "Cámara B · fría", "B corregida"),
         ("#A6D98A", .5, "Cámara C · verdosa", "C corregida")]
ILLUSTRATIVE = [("mel", 64), ("nv", 24), ("bkl", 8)]     # anchos de la maqueta (sin cifras): solo si no hay modelo


def hero_example() -> dict | None:
    exs = data.examples()
    if not exs:
        return None
    return next((e for e in exs if e["label"] == "mel"), exs[0])


@lru_cache(maxsize=4)
def hero_prediction(image_id: str, model: str) -> dict | None:
    """Predicción real del ejemplo del hero (en vivo o precalculada), en caché."""
    try:
        meta = data.model_metadata(model) or {}
        if meta.get("live"):
            path = data.example_path(image_id)
            res = inference.predict(model, Image.open(path)) if path else None
        else:
            res = inference.predict_precomputed(model, image_id)
    except Exception as e:  # noqa: BLE001 - el hero cae a la ilustración
        log.warning("Predicción del hero no disponible con %s: %s", model, e)
        return None
    if res is None:
        return None
    probs = dict(zip(res["classes"], res["probs"]))
    size = res["image"].shape[0] if res.get("image") is not None else 224
    return {"model": model, "probs": probs, "heat": inference.heatmap_png(res.get("cam"), size)}


def hero_model() -> str | None:
    avail = {m["id"]: m for m in data.demo_models() if m["available"]}
    for m in [ORIGINAL_MODEL, *BENCHMARK_MODELS]:
        if m in avail:
            return m
    return None


def hero_visual():
    ex = hero_example()
    model = hero_model()
    pred = hero_prediction(ex["image_id"], model) if ex and model else None
    if ex:
        layers = [html.Img(src=f"/media/examples/{ex['file']}", alt=f"Imagen dermatoscópica de ejemplo de HAM10000 ({CLASS_NAMES[ex['label']]})")]
        layers.append(html.Img(src=pred["heat"], alt="", className="cam-layer") if pred and pred.get("heat")
                      else html.Div(className="heat"))
        frame = html.Div([*layers, html.Div(className="scan"),
                          html.Span([dot("#6FE3B5"), "CCI activa" if (pred and pred["model"] == ORIGINAL_MODEL) or not pred
                                     else model_name(pred["model"])], className="badge-dark l"),
                          html.Span("Grad-CAM", className="badge-dark r")], className="photo hero-img")
    else:
        frame = html.Div([html.Div(className="lesion"), html.Div(className="heat"), html.Div(className="scan"),
                          html.Span([dot("#6FE3B5"), "CCI activa"], className="badge-dark l"),
                          html.Span("Grad-CAM", className="badge-dark r")], className="derm hero-img")
    if pred:
        top = sorted(pred["probs"].items(), key=lambda kv: -kv[1])[:3]
        rows = [html.Div([html.Span([html.Span([class_dot(c), CLASS_NAMES[c]]), html.Span(fmt_pct(p), className="num")]),
                          html.Div(html.Span(style={"width": f"{max(100 * p, 1):.1f}%", "background": CLASS_COLORS[c],
                                                    "animationDelay": f"{.2 * (i + 1):.1f}s"}), className="bar thin")],
                         className="float-row" + (" top" if i == 0 else "")) for i, (c, p) in enumerate(top)]
        tag = f"Predicción real · {model_name(pred['model'])}"
    else:
        rows = [html.Div([html.Span(html.Span([class_dot(c), CLASS_NAMES[c]])),
                          html.Div(html.Span(style={"width": f"{w}%", "background": CLASS_COLORS[c],
                                                    "animationDelay": f"{.2 * (i + 1):.1f}s"}), className="bar thin")],
                         className="float-row" + (" top" if i == 0 else "")) for i, (c, w) in enumerate(ILLUSTRATIVE)]
        tag = "Ejemplo ilustrativo"
    card = html.Div([html.Div([html.Span("Predicción", className="kicker"), html.Span(tag, className="tagline")],
                              className="float-head"),
                     html.Div(rows, className="float-rows")], className="card float float-card")
    return html.Div([html.Div(frame, className="card hero-frame"), card], className="hero-visual")


def hero():
    return html.Section([
        html.Div([
            html.Span([html.B("NUEVO"), "Capa de constancia de color CCI"], className="hero-badge"),
            html.H1(["Misma lesión.", html.Br(), " Cualquier cámara.", html.Br(), " ",
                     html.Span("La misma respuesta.", className="hl")], className="disp hero-title"),
            html.P("Deep learning que clasifica 7 tipos de lesión cutánea y no se deja engañar por el color de cada equipo.",
                   className="hero-sub"),
            html.Div([dcc.Link([icon("upload", 20), "Pruébalo con una imagen"], href="/demo", className="btn btn-primary"),
                      dcc.Link(f"Ver los {len(BENCHMARK_MODELS)} modelos", href="/modelos", className="btn btn-ghost")],
                     className="ctas"),
            html.Div([html.Span([icon("lock", 16), "Tu imagen no se guarda"]),
                      html.Span([icon("bolt", 16), "Inferencia con ONNX"])], className="micro"),
        ], className="hero-copy"),
        hero_visual(),
    ], className="wrap wrap-1200 hero")


def stats():
    s = data.eda_summary()
    n_img = s.get("n_images")
    n_src = len(s.get("sources") or [])
    classes = s.get("classes") or CLASS_ORDER
    best = data.best_model()
    ba = (html.Span(fmt_pct(best["ba"]), title=f"{best['name']}") if best else placeholder("[ BA ]%"))
    ba_label = (f"exactitud balanceada del mejor modelo ({best['name']}, test)" if best
                else "exactitud balanceada del mejor modelo · pendiente")
    return html.Section([
        stat_card(fmt_num(n_img) if n_img else placeholder("[ N ]"),
                  f"imágenes dermatoscópicas de {n_src or 4} fuentes clínicas"),
        stat_card(str(len(classes)), "tipos de lesión, del nevus al melanoma",
                  extra=html.Div([class_dot(c) for c in CLASS_ORDER], className="stat-dots")),
        stat_card(str(len(BENCHMARK_MODELS)), f"arquitecturas comparadas, de MobileNet a ViT, más {len(BASELINES)} baselines"),
        stat_card(ba, ba_label, highlight=True),
    ], className="wrap wrap-1200 stats", **{"aria-label": "Cifras clave"})


STEPS = [
    ("upload", "", "Subes la imagen", "Arrastra una imagen dermatoscópica o elige una de ejemplo. Se procesa en memoria y no se guarda."),
    ("drop", "alt", "La capa CCI corrige el color",
     "Antes de que el modelo la vea, la imagen se lleva a un color canónico, sin importar la cámara que la tomó."),
    ("eye", "", "Predice y te muestra dónde miró",
     "Probabilidades para las 7 clases, un mapa de calor Grad-CAM y las clases que no se pueden descartar."),
]


def how():
    steps = [html.Div([
        html.Div([html.Div(icon(ic, 26), className=f"step-ic {alt}"), html.Span(f"{i:02d}", className="disp num step-num")],
                 className="step-top"),
        html.Span(str(i), className=f"disp num step-sq {alt}", **{"aria-hidden": "true"}),
        html.Div([html.H3(title, className="disp"), html.P(text)], className="step-body"),
    ], className="step") for i, (ic, alt, title, text) in enumerate(STEPS, 1)]
    return html.Section(html.Div([
        html.Div([html.Span("Cómo funciona", className="eyebrow"),
                  html.H2("De la foto a una respuesta explicable, en tres pasos.", className="disp sec-title")],
                 className="sec-head"),
        html.Div(steps, className="steps"),
    ], className="wrap wrap-1200 how"), className="band")


def _tile(ex, tint=None, fixed=False):
    cls = "photo" if ex else "derm"
    kids = [html.Img(src=f"/media/examples/{ex['file']}", alt="")] if ex else [html.Div(className="lesion")]
    if tint:
        kids.append(html.Div(className="tint", style={"background": tint[0], "opacity": tint[1]}))
    return html.Div(kids, className=cls + (" fixed" if fixed else ""))


def cci_section():
    ex = hero_example()
    o = data.ours()
    delta = (o["cci"]["ba"] - o["base"]["ba"]) if o["base"] and o["cci"]["ba"] is not None and o["base"]["ba"] is not None else None
    before = [html.Figure([_tile(ex, t), html.Figcaption(t[2])]) for t in TINTS]
    after = [html.Figure([_tile(ex, None, fixed=True), html.Figcaption(t[3])]) for t in TINTS]
    return html.Section([
        html.Div([
            html.Div([html.Span("La estrella: capa CCI", className="eyebrow eyebrow-accent"),
                      html.H2(["Tres cámaras. Tres colores.", html.Br(), " Un solo diagnóstico."], className="disp sec-title")],
                     className="sec-head"),
            html.P("Cada dermatoscopio ve el color a su manera. La capa de constancia de color se coloca delante de cualquier "
                   "modelo y elimina esa diferencia antes de que cuente.", className="cci-lead"),
        ], className="cci-head"),
        html.Div([
            html.Div([html.Span("Antes · lo que llega de cada equipo", className="lbl"), html.Div(before, className="tri")],
                     className="cci-col"),
            html.Div([html.Span(icon("arrow", 32)), html.B("CCI")], className="cci-arrow"),
            html.Div([html.Span("Después · lo que ve el modelo", className="lbl after"), html.Div(after, className="tri")],
                     className="cci-col"),
            html.P("Ilustración: los tintes simulan ganancias por canal (lo que cambia entre equipos). La capa es invariante a "
                   "esas ganancias por construcción, así que las tres versiones dan la misma entrada al modelo.",
                   className="cci-note"),
        ], className="card cci-card"),
        html.Div([
            html.Div([html.B("Delante de cualquier backbone"),
                      html.Span("Una capa ligera que se enchufa sin cambiar la arquitectura del modelo.")], className="arg"),
            html.Div([html.B("Medido, no prometido"),
                      html.Span(["Con CCI frente a sin CCI: ",
                                 html.Strong(fmt_signed(100 * delta)) if delta is not None else placeholder("[ Δ BA ]"),
                                 " puntos de exactitud balanceada."])], className="arg"),
            html.Div([html.B("Estable entre fuentes"),
                      html.Span("Probado en las 4 fuentes clínicas de HAM10000, con partición por lesión.")], className="arg"),
        ], className="args"),
    ], className="wrap wrap-1200 cci")


def closing():
    return html.Section([
        html.Div([html.Div([html.H2("Mira lo que ve el modelo.", className="disp"),
                            html.P("Prueba con una imagen de ejemplo, sin subir nada.")]),
                  dcc.Link("Abrir la demo en vivo", href="/demo", className="btn btn-accent")], className="cta"),
        disclaimer(),
    ], className="wrap wrap-1200 closing")


def layout(**_):
    return html.Div([hero(), stats(), how(), cci_section(), closing()])
