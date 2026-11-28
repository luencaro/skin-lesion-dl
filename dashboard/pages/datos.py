"""Explora los datos (design/Datos.dc.html): HAM10000 filtrable por clase y por fuente.

Todos los conteos se calculan con pandas sobre los metadatos exportados (eda/images.json, una fila
por imagen de HAM10000). Las frases de cada tarjeta se generan a partir de esos conteos.
"""

from __future__ import annotations

from fractions import Fraction

import dash
import numpy as np
import pandas as pd
from core import data
from core.components import (
    class_dot,
    dot,
    eyebrow,
    fmt_num,
    fmt_pct,
    insight_header,
    placeholder,
)
from core.config import (
    CLASS_COLORS,
    CLASS_NAMES,
    CLASS_ORDER,
    LOCALIZATION_NAMES,
    SEX_NAMES,
    SOURCE_NAMES,
    SOURCE_ORDER,
    SOURCE_VARS,
    TITLE,
)
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update

dash.register_page(__name__, path="/datos", name="Explora los datos", order=3, title=f"Explora los datos · {TITLE}",
                   redirect_from=["/eda"])

CLASS_PLURAL = {"mel": "melanomas", "nv": "nevus", "bcc": "carcinomas basocelulares", "akiec": "queratosis actínicas",
                "bkl": "lesiones queratósicas benignas", "df": "dermatofibromas", "vasc": "lesiones vasculares"}
CLASS_SINGULAR = {"mel": "melanoma", "nv": "nevus", "bcc": "carcinoma basocelular", "akiec": "queratosis actínica",
                  "bkl": "lesión queratósica benigna", "df": "dermatofibroma", "vasc": "lesión vascular"}
AGE_BINS = 18                       # tramos de 5 años: 0–4, ..., 85–89
GALLERY_SEED = 2026


def frame() -> pd.DataFrame:
    return data.images()


def apply(df: pd.DataFrame, classes: list[str], sources: list[str]) -> pd.DataFrame:
    if classes:
        df = df[df["dx"].isin(classes)]
    if sources:
        df = df[df["dataset"].isin(sources)]
    return df


def share_phrase(p: float) -> str:
    """0,669 -> "2 de cada 3" (fracción sencilla más cercana, denominador <= 10)."""
    f = Fraction(p).limit_denominator(10)
    if f.numerator == 0:
        return "casi ninguna de las"
    if f == 1:
        return "todas las"
    return f"{f.numerator} de cada {f.denominator}"


# ---------------------------------------------------------------------------
# Tarjetas
# ---------------------------------------------------------------------------


def class_card(df_src: pd.DataFrame, classes: list[str]):
    counts = df_src["dx"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    total = int(counts.sum())
    if total == 0:
        return html.Div(insight_header("Insight", "Sin imágenes con estos filtros."), className="card data-card")
    order = counts.sort_values(ascending=False)
    top, p = order.index[0], order.iloc[0] / total
    phrase = share_phrase(p)
    title = f"{phrase[0].upper()}{phrase[1:]} imágenes son {CLASS_PLURAL[top]}."
    sub = (f"Por eso medimos exactitud balanceada: un modelo que siempre dijera «{CLASS_SINGULAR[top]}» acertaría el "
           f"{fmt_num(100 * p, 0)} % sin aprender nada.")
    bars = []
    for c in order.index:
        n = int(order[c])
        bars.append(html.Div([
            html.Span([class_dot(c), CLASS_NAMES[c]], className="class-label"),
            html.Div(html.Span(style={"width": f"{100 * n / max(order.iloc[0], 1):.2f}%", "background": CLASS_COLORS[c]}),
                     className="track"),
            html.Span([html.Strong(fmt_num(n)), html.Span(f" · {fmt_pct(n / total)}")], className="num"),
        ], className="cbar", style={"opacity": 1 if not classes or c in classes else 0.3}))
    return html.Div([insight_header("Insight", title, sub), html.Div(bars, className="cbar-list")], className="card data-card")


def age_card(df: pd.DataFrame):
    ages = pd.to_numeric(df["age"], errors="coerce")
    missing = int(ages.isna().sum())
    a = ages.dropna()
    hist = np.bincount(np.clip((a // 5).astype(int), 0, AGE_BINS - 1), minlength=AGE_BINS) if len(a) else np.zeros(AGE_BINS, int)
    if len(a):
        lo, hi = (int(5 * round(q / 5)) for q in np.percentile(a, [10, 90]))
        title = f"La mayoría tiene entre {lo} y {hi} años."
    else:
        title = "Sin edades con estos filtros."
    top = max(int(hist.max()), 1)
    bars = [html.Div(style={"height": f"{100 * n / top:.1f}%"}, title=f"{5 * i}–{5 * i + 4} años: {fmt_num(n)}")
            for i, n in enumerate(hist)]
    labels = [html.Span(str(5 * i) if i % 2 == 0 else "", className="num") for i in range(AGE_BINS)]
    return html.Div([
        insight_header(None, title, f"Imágenes por edad del paciente (tramos de 5 años) · {fmt_num(missing)} sin edad · "
                                    "el 10 % y el 90 % de las edades, redondeados a 5 años", size=22),
        html.Div(bars, className="ages", style={"--n": AGE_BINS}, role="img",
                 **{"aria-label": "Histograma de edad: " + "; ".join(f"{5 * i}–{5 * i + 4}: {int(n)}" for i, n in enumerate(hist))}),
        html.Div(labels, className="ages-lab", style={"--n": AGE_BINS}, **{"aria-hidden": "true"}),
    ], className="card data-card", style={"flex": "2 1 520px", "minWidth": 0, "gap": 20})


def sex_card(df: pd.DataFrame):
    s = df["sex"].fillna("unknown").replace({"": "unknown"})
    counts = {k: int((s == k).sum()) for k in ("male", "female", "unknown")}
    total = max(sum(counts.values()), 1)
    m, f = counts["male"] / total, counts["female"] / total
    if not sum(counts.values()):
        title = "Sin datos con estos filtros."
    elif abs(m - f) < 0.1:
        title = "Casi parejo por sexo."
    else:
        title = f"Predominan los hombres: {fmt_pct(m)}." if m > f else f"Predominan las mujeres: {fmt_pct(f)}."
    colors = {"male": "var(--s1)", "female": "var(--s4)", "unknown": "var(--ink-3)"}
    return html.Div([
        insight_header(None, title, "Imágenes por sexo del paciente", size=22),
        html.Div([html.Span(style={"width": f"{100 * counts[k] / total:.2f}%", "background": colors[k]}, title=SEX_NAMES[k])
                  for k in counts if counts[k]], className="sexbar", role="img",
                 **{"aria-label": ", ".join(f"{SEX_NAMES[k]} {fmt_pct(counts[k] / total)}" for k in counts)}),
        html.Div([html.Span([html.Span([dot(colors[k]), SEX_NAMES[k]]),
                             html.Span([html.Strong(fmt_num(counts[k])), html.Span(f" · {fmt_pct(counts[k] / total)}")],
                                       className="num")]) for k in counts], className="legend"),
    ], className="card data-card", style={"flex": "1 1 300px", "minWidth": 0})


def loc_card(df: pd.DataFrame):
    counts = df["localization"].fillna("unknown").value_counts()
    total = max(int(counts.sum()), 1)
    if len(counts) >= 2:
        a, b = counts.index[:2]
        title = (f"{LOCALIZATION_NAMES.get(a, a)} y {LOCALIZATION_NAMES.get(b, b).lower()} concentran el "
                 f"{fmt_num(100 * (counts.iloc[0] + counts.iloc[1]) / total, 0)} %.")
    elif len(counts) == 1:
        title = f"Todas en {LOCALIZATION_NAMES.get(counts.index[0], counts.index[0]).lower()}."
    else:
        title = "Sin datos con estos filtros."
    top = max(int(counts.max()) if len(counts) else 1, 1)
    rows = [html.Div([html.Span(LOCALIZATION_NAMES.get(k, k)),
                      html.Div(html.Span(style={"width": f"{max(100 * n / top, 0.6):.2f}%", "background": "var(--s2)"}),
                               className="track", style={"height": 10}),
                      html.Span(fmt_num(n), className="num")], className="loc") for k, n in counts.items()]
    return html.Div([insight_header(None, title, "Imágenes por localización corporal", size=22),
                     html.Div(rows, className="locs")],
                    className="card data-card", style={"flex": "1.2 1 480px", "minWidth": 0, "gap": 18})


def source_card(df_cls: pd.DataFrame, sources: list[str]):
    counts = df_cls["dataset"].value_counts().reindex(SOURCE_ORDER).fillna(0).astype(int)
    top = max(int(counts.max()), 1)
    colors = data.source_color()
    rows = []
    for s in SOURCE_ORDER:
        c = colors.get(s)
        lab = (html.Span(f"L {fmt_num(c['lab'][0], 1)} · a {fmt_num(c['lab'][1], 1)} · b {fmt_num(c['lab'][2], 1)}", className="num")
               if c else placeholder("[ — ]"))
        swatch = (html.Div(className="src-swatch", style={"background": c["hex"]}, title=f"Color medio {c['hex']}") if c
                  else html.Div(html.Div(className="lesion"), className="derm src-swatch"))
        rows.append(html.Div([
            swatch,
            html.Div([html.Span([html.Strong(SOURCE_NAMES[s]), html.Span(fmt_num(counts[s]), className="num")]),
                      html.Div(html.Span(style={"width": f"{100 * counts[s] / top:.2f}%", "background": SOURCE_VARS[s]}),
                               className="track"),
                      html.Span(["Color medio (Lab): ", lab], className="src-lab")], className="src-body"),
        ], className="src-row", style={"opacity": 1 if not sources or s in sources else 0.35}))
    note = ("Color medio: media de los canales RGB de cada imagen de train de esa fuente (notebook 06), convertida a "
            "CIELAB (D65); no cambia con los filtros." if colors else
            "El color medio por fuente aparece cuando el EDA se exporta con las imágenes (color por imagen).")
    return html.Div([insight_header(None, "Cuatro fuentes, cuatro colores.",
                                    "Imágenes por fuente y su tono típico, lo que la capa CCI neutraliza", size=22),
                     *rows, html.Span(note, className="small-note")],
                    className="card data-card", style={"flex": "1 1 400px", "minWidth": 0, "gap": 18})


def gallery_card(df_all: pd.DataFrame, classes: list[str], sources: list[str]):
    gal = data.eda_summary().get("gallery") or {}
    src_of = df_all.set_index("image_id")["dataset"].to_dict() if len(df_all) else {}
    cols = []
    for c in CLASS_ORDER:
        ids = [i for i in gal.get(c, []) if not sources or src_of.get(i) in sources]
        rng = np.random.default_rng(GALLERY_SEED + CLASS_ORDER.index(c))
        pick = [ids[k] for k in rng.permutation(len(ids))[:3]] if ids else []

        def tile(i, small=False, c=c):
            if i is None:
                return html.Div(placeholder("—"), className="gimg" + (" sm" if small else ""),
                                style={"display": "flex", "alignItems": "center", "justifyContent": "center"})
            return html.Div(html.Img(src=f"/media/gallery/{i}.jpg", alt=f"{CLASS_NAMES[c]} ({i})"),
                            className="gimg" + (" sm" if small else ""), style={"--gc": CLASS_COLORS[c]})
        pick += [None] * (3 - len(pick))
        cols.append(html.Div([html.Span([class_dot(c), CLASS_NAMES[c]]), tile(pick[0]),
                              html.Div([tile(pick[1], True), tile(pick[2], True)], className="gpair")],
                             className="gcol", style={"opacity": 1 if not classes or c in classes else 0.3}))
    return html.Div([insight_header(None, "Galería por clase",
                                    "Imágenes reales de entrenamiento de HAM10000, elegidas al azar con semilla fija", size=22),
                     html.Div(cols, className="gallery")], className="card data-card", style={"gap": 20})


def data_sheet():
    card = data.eda_summary().get("data_card") or []
    statement = data.content("DATA_STATEMENT.md")
    return html.Details([
        html.Summary(html.Div([eyebrow("Ficha de datos"), html.H2("Qué hay en HAM10000 y qué no", className="disp")],
                              className="insight")),
        html.Div([
            html.Div(html.Table(html.Tbody([html.Tr([html.Th(r["campo"], scope="row"), html.Td(r["valor"])]) for r in card]),
                                className="tests"), style={"overflowX": "auto"}) if card else None,
            dcc.Markdown(statement, className="md") if statement else None,
        ], className="tech-body"),
    ], className="card tech")


def body(classes: list[str], sources: list[str]):
    df = frame()
    both = apply(df, classes, sources)
    return [
        class_card(apply(df, [], sources), classes),
        html.Div([age_card(both), sex_card(both)], className="duo"),
        html.Div([loc_card(both), source_card(apply(df, classes, []), sources)], className="duo"),
        gallery_card(df, classes, sources),
    ]


# ---------------------------------------------------------------------------
# Página y filtros
# ---------------------------------------------------------------------------


def chips(kind: str, keys: list[str], names: dict, colors: dict | None):
    out = [html.Button("Todas", id={"type": kind, "id": "all"}, n_clicks=0, className="chip on", **{"aria-pressed": "true"})]
    for k in keys:
        label = [dot(colors[k]), names[k]] if colors else [names[k]]
        out.append(html.Button(label, id={"type": kind, "id": k}, n_clicks=0, className="chip", **{"aria-pressed": "false"}))
    return out


def layout(**_):
    df = frame()
    s = data.eda_summary()
    q = s.get("quality") or {}
    n_src = df["dataset"].nunique() if len(df) else 0
    lead = f"{fmt_num(len(df))} imágenes de {n_src} fuentes clínicas."
    if q.get("lesiones_en_varias_particiones") == 0:
        lead += " Partición por lesión: ninguna lesión aparece a la vez en entrenamiento y en test."
    if not len(df):
        return html.Div([html.Div(html.Div([html.Span("Explora los datos", className="eyebrow"),
                                            html.H1("HAM10000, de cerca.", className="disp page-title"),
                                            html.P(["Faltan los metadatos exportados: ", placeholder("python -m src.export_dashboard")],
                                                   className="page-lead")]), className="page-head")], className="wrap page")
    return html.Div([
        html.Div([
            html.Div([html.Span("Explora los datos", className="eyebrow"), html.H1("HAM10000, de cerca.", className="disp page-title"),
                      html.P(lead, className="page-lead")]),
            html.Span(["Mostrando ", html.Strong(fmt_num(len(df)), id="data-count"), " imágenes"], className="num page-meta",
                      **{"aria-live": "polite"}),
        ], className="page-head"),
        html.Div([
            html.Div([html.Span("Clase", className="lbl"), *chips("cls-chip", CLASS_ORDER, CLASS_NAMES, CLASS_COLORS)],
                     className="filter-row", role="group", **{"aria-label": "Filtrar por clase"}),
            html.Div([html.Span("Fuente", className="lbl"), *chips("src-chip", SOURCE_ORDER, SOURCE_NAMES, None)],
                     className="filter-row", role="group", **{"aria-label": "Filtrar por fuente"}),
        ], className="card filters"),
        html.Div(body([], []), id="data-body", className="stack", style={"gap": 32}),
        data_sheet(),
        html.P("Fuente: metadatos de HAM10000 (Tschandl, Rosendahl y Kittler, 2018). Los filtros se aplican sobre los metadatos "
               "exportados por python -m src.export_dashboard.", className="data-source"),
        dcc.Store(id="data-filters", data={"classes": [], "sources": []}),
    ], className="wrap page", style={"gap": 32})


def _toggle(current: list[str], key: str, universe: list[str]) -> list[str]:
    if key == "all":
        return []
    sel = [k for k in current if k != key] if key in current else [*current, key]
    return [] if set(sel) == set(universe) else [k for k in universe if k in sel]


@callback(
    Output("data-filters", "data"),
    Output({"type": "cls-chip", "id": ALL}, "className"), Output({"type": "cls-chip", "id": ALL}, "aria-pressed"),
    Output({"type": "src-chip", "id": ALL}, "className"), Output({"type": "src-chip", "id": ALL}, "aria-pressed"),
    Output("data-body", "children"), Output("data-count", "children"),
    Input({"type": "cls-chip", "id": ALL}, "n_clicks"), Input({"type": "src-chip", "id": ALL}, "n_clicks"),
    State("data-filters", "data"), State({"type": "cls-chip", "id": ALL}, "id"), State({"type": "src-chip", "id": ALL}, "id"),
    prevent_initial_call=True,
)
def update_filters(_c, _s, filters, cls_ids, src_ids):
    trig = ctx.triggered_id
    if not isinstance(trig, dict):
        return (no_update,) * 7
    f = filters or {"classes": [], "sources": []}
    if trig["type"] == "cls-chip":
        f = {**f, "classes": _toggle(f["classes"], trig["id"], CLASS_ORDER)}
    else:
        f = {**f, "sources": _toggle(f["sources"], trig["id"], SOURCE_ORDER)}

    def states(ids, sel):
        on = [(i["id"] == "all" and not sel) or i["id"] in sel for i in ids]
        return ["chip on" if o else "chip" for o in on], ["true" if o else "false" for o in on]

    cc, ca = states(cls_ids, f["classes"])
    sc, sa = states(src_ids, f["sources"])
    n = len(apply(frame(), f["classes"], f["sources"]))
    return f, cc, ca, sc, sa, body(f["classes"], f["sources"]), fmt_num(n)
