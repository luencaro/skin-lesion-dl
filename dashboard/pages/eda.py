"""EDA esencial con filtros globales por clase y por fuente.

La página es un registro de secciones (SECTIONS): cada una declara su título, sus controles
propios, la clave de su interpretación (notebooks 05 y 06) y una función `render(df, summary, *valores)`
que recibe las imágenes ya filtradas. Para añadir una sección basta con decorar otra función con
@eda_section; el callback se registra solo y responde a los filtros globales.
"""


from collections.abc import Callable
from dataclasses import dataclass, field

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from core import data
from core.components import (
    error_alert,
    fmt,
    kpi,
    pending,
    section,
    static_graph,
    swatch,
)
from core.config import (
    CLASS_COLORS,
    CLASS_NAMES,
    CLASSES,
    OKABE_ITO,
    SOURCE_COLORS,
    SOURCES,
    TITLE,
    UI,
)
from core.stats import cramers_v, epsilon2
from dash import Input, Output, callback, dcc, html

dash.register_page(__name__, path="/eda", name="EDA", order=1, title=f"EDA · {TITLE}")

FILTERS = "eda-filters"


@dataclass
class Section:
    id: str
    title: str
    render: Callable
    lead: str | None = None
    interp: str | None = None
    controls: Callable | None = None                 # () -> componentes
    inputs: list = field(default_factory=list)       # [(id, propiedad)] de los controles


SECTIONS: list[Section] = []


def eda_section(sid: str, title: str, lead: str | None = None, interp: str | None = None,
                controls: Callable | None = None, inputs=()):
    def deco(fn):
        SECTIONS.append(Section(sid, title, fn, lead, interp, controls, list(inputs)))
        return fn
    return deco


def filtered(filters: dict | None) -> pd.DataFrame:
    df = data.images()
    f = filters or {}
    return df[df["dx"].isin(f.get("classes") or CLASSES) & df["dataset"].isin(f.get("sources") or SOURCES)]


def bar_labels(fig):
    fig.update_traces(textposition="outside", cliponaxis=False, textfont={"color": UI["dark"]})
    return fig


# ---------------------------------------------------------------------------
# Secciones
# ---------------------------------------------------------------------------


@eda_section("clases", "Distribución de clases y desbalance", interp="class_distribution",
             lead="Conteo de imágenes por clase con los filtros aplicados, y composición por fuente de cada clase.")
def render_classes(df: pd.DataFrame, summary: dict):
    counts = df["dx"].value_counts().reindex(CLASSES).fillna(0).astype(int)
    present = counts[counts > 0]
    ratio = present.max() / present.min() if len(present) > 1 else float("nan")
    p = present / present.sum() if present.sum() else present
    h = float(-(p * np.log2(p)).sum()) / np.log2(len(p)) if len(p) > 1 else float("nan")
    fig = go.Figure(go.Bar(x=CLASSES, y=counts.values, text=counts.values, marker_color=[CLASS_COLORS[c] for c in CLASSES],
                           customdata=[CLASS_NAMES[c] for c in CLASSES],
                           hovertemplate="%{x}: %{y} imágenes<br>%{customdata}<extra></extra>"))
    fig.update_layout(title="Imágenes por clase", yaxis_title="Imágenes", showlegend=False)
    bar_labels(fig)
    share = pd.crosstab(df["dx"], df["dataset"], normalize="index").reindex(index=CLASSES, columns=SOURCES).fillna(0)
    fig2 = go.Figure([go.Bar(y=CLASSES, x=100 * share[s], name=s, orientation="h", marker_color=SOURCE_COLORS[s],
                             hovertemplate=f"{s}: %{{x:.1f}} %<extra></extra>") for s in SOURCES])
    fig2.update_layout(barmode="stack", title="Fuente de las imágenes de cada clase (%)", xaxis_title="%",
                       yaxis={"autorange": "reversed"}, legend={"traceorder": "normal"})
    return [
        dbc.Row([dbc.Col(kpi(f"{len(df):,}".replace(",", " "), "imágenes seleccionadas"), xs=4),
                 dbc.Col(kpi(fmt(ratio, 1), "razón de desbalance (mayor / menor)", tone="accent"), xs=4),
                 dbc.Col(kpi(fmt(h, 3), "entropía normalizada (1 = equilibrio)"), xs=4)], className="g-2 mb-2"),
        dbc.Row([dbc.Col(static_graph(fig, 320), md=6), dbc.Col(static_graph(fig2, 320), md=6)], className="g-2"),
        _interp_extra("source_by_class"),
    ]


def _gallery_controls():
    return dbc.Row([
        dbc.Col(dbc.RadioItems(id="gallery-class", options=[{"label": swatch(c, CLASS_COLORS[c]), "value": c} for c in CLASSES],
                               value="mel", inline=True, className="touch-radio", inputClassName="btn-check",
                               labelClassName="btn btn-outline-primary btn-touch", labelCheckedClassName="active"), xs=12, md=9),
        dbc.Col(dbc.Button("Otra muestra", id="gallery-shuffle", n_clicks=0, color="primary", className="btn-touch w-100"),
                xs=12, md=3),
    ], className="g-2 align-items-center")


@eda_section("galeria", "Galería de muestras aleatorias por clase", interp="gallery", controls=_gallery_controls,
             inputs=[("gallery-class", "value"), ("gallery-shuffle", "n_clicks")],
             lead="Miniaturas de train elegidas al azar; respetan el filtro de fuente.")
def render_gallery(df: pd.DataFrame, summary: dict, dx: str, shuffle: int):
    ids = (summary.get("gallery") or {}).get(dx) or []
    meta = df.set_index("image_id")
    ids = [i for i in ids if i in meta.index]
    if not ids:
        return html.P("No hay miniaturas para esta combinación de clase y fuente.", className="muted")
    rng = np.random.default_rng(int(shuffle or 0))
    pick = rng.choice(ids, size=min(8, len(ids)), replace=False)
    cards = [html.Figure([html.Img(src=f"/media/gallery/{i}.jpg", alt=f"{dx} de {meta.loc[i, 'dataset']}",
                                   className="thumb"),
                          html.Figcaption([swatch(meta.loc[i, "dataset"], SOURCE_COLORS.get(meta.loc[i, "dataset"], UI["dark"]))])],
                         className="thumb-card fade-in") for i in pick]
    return html.Div(cards, className="thumb-grid", key=f"{dx}-{shuffle}")


def _color_controls():
    variables = data.eda_summary().get("color_variables") or {}
    return dcc.Dropdown(id="color-var", options=[{"label": v, "value": k} for k, v in variables.items()],
                        value="saturacion" if "saturacion" in variables else next(iter(variables), None),
                        clearable=False, searchable=False, className="touch-dropdown")


@eda_section("color", "Color por fuente frente a por clase", interp="color_source_vs_class", controls=_color_controls,
             inputs=[("color-var", "value")],
             lead="El hallazgo que motiva el modelo original: el color depende mucho más de la fuente de adquisición "
                  "que del diagnóstico. ε² de Kruskal–Wallis (0 = sin efecto, 1 = máximo) recalculado con los filtros.")
def render_color(df: pd.DataFrame, summary: dict, var: str | None):
    tr = df[df["split"] == "train"]
    if not var or var not in tr.columns or tr[var].isna().all():
        return pending("Sin datos de color", "Regenera el EDA con python -m src.export_dashboard (requiere las imágenes).")
    e_src, e_cls = epsilon2(tr[var], tr["dataset"]), epsilon2(tr[var], tr["dx"])
    label = (summary.get("color_variables") or {}).get(var, var)
    figs = []
    for group, order, colors, e, name in (("dataset", SOURCES, SOURCE_COLORS, e_src, "fuente"),
                                           ("dx", CLASSES, CLASS_COLORS, e_cls, "clase")):
        fig = go.Figure([go.Box(y=tr.loc[tr[group] == g, var], name=g, marker_color=colors[g], boxpoints=False,
                                fillcolor=colors[g], line_color=UI["dark"]) for g in order if (tr[group] == g).any()])
        fig.update_layout(title=f"Por {name}: ε² = {fmt(e)}", yaxis_title=label, showlegend=False)
        figs.append(fig)
    eps = pd.DataFrame(summary.get("epsilon2") or [])
    eps = eps[eps["variable"].isin(list((summary.get("color_variables") or {}).keys()))] if len(eps) else eps
    bars = go.Figure()
    if len(eps):
        for col, name, color in (("epsilon2_fuente", "Fuente", OKABE_ITO["vermillion"]), ("epsilon2_clase", "Clase", OKABE_ITO["blue"])):
            bars.add_bar(x=eps["etiqueta"], y=eps[col], name=name, marker_color=color, text=[fmt(v, 2) for v in eps[col]],
                         textfont={"size": 10})
        bars.update_layout(barmode="group", title="ε² por variable en todo train: fuente frente a clase", yaxis_title="ε²")
        bar_labels(bars)
    return [dbc.Row([dbc.Col(static_graph(figs[0], 330), md=6), dbc.Col(static_graph(figs[1], 330), md=6)], className="g-2"),
            static_graph(bars, 320), _interp_extra("color_by_class")]


@eda_section("calidad", "Calidad del conjunto de datos", interp="partition",
             lead="Duplicados por pHash, partición agrupada por lesión y validación adversarial entre particiones.")
def render_quality(df: pd.DataFrame, summary: dict):
    q = summary.get("quality") or {}
    kpis = dbc.Row([
        dbc.Col(kpi(str(q.get("duplicados_candidatos", "—")), f"pares candidatos (pHash ≤ {q.get('umbral_phash', 2)})"), xs=6, md=3),
        dbc.Col(kpi(str(q.get("duplicados_confirmados", "—")), "duplicados confirmados en la revisión visual"), xs=6, md=3),
        dbc.Col(kpi(str(q.get("lesiones_en_varias_particiones", "—")), "lesiones en train y test a la vez"), xs=6, md=3),
        dbc.Col(kpi(str(q.get("lesiones_en_varios_pliegues", "—")), "lesiones en más de un pliegue"), xs=6, md=3),
    ], className="g-2 mb-2")
    part = df.assign(particion=np.where(df["split"] == "test", "test", "pliegue " + df["fold"].fillna(-1).astype(int).astype(str)))
    order = [f"pliegue {i}" for i in range(5)] + ["test"]
    tab = pd.crosstab(part["particion"], part["dx"]).reindex(index=order, columns=CLASSES).fillna(0)
    share = tab.div(tab.sum(1).replace(0, np.nan), axis=0) * 100
    fig = go.Figure([go.Bar(x=order, y=share[c], name=c, marker_color=CLASS_COLORS[c],
                            hovertemplate=f"{c}: %{{y:.1f}} %<extra></extra>") for c in CLASSES])
    fig.update_layout(barmode="stack", title="Composición por clase de cada pliegue y de test (%)", yaxis_title="%")
    totals = tab.sum(1).astype(int)
    fig.add_trace(go.Scatter(x=order, y=[104] * len(order), text=[f"n = {v}" for v in totals], mode="text",
                             showlegend=False, hoverinfo="skip", textfont={"color": UI["dark"], "size": 11}))
    return [kpis, static_graph(fig, 340), _interp_extra("duplicates"), _interp_extra("adversarial")]


EXPLORER_VARS = {"age": ("Edad", "num"), "sex": ("Sexo", "cat"), "localization": ("Localización", "cat"),
                 "dx_type": ("Método diagnóstico", "cat"), "dataset": ("Fuente", "cat"), "dx": ("Clase", "cat")}
EXPLORER_INTERP = {"age": "age", "dataset": "source_covariates", "dx_type": "source_covariates"}


def _explorer_controls():
    options = [{"label": v[0], "value": k} for k, v in EXPLORER_VARS.items()]
    options += [{"label": v, "value": k} for k, v in (data.eda_summary().get("color_variables") or {}).items()]
    return dbc.Row([
        dbc.Col(dcc.Dropdown(id="explorer-var", options=options, value="age", clearable=False, searchable=False,
                             className="touch-dropdown"), xs=12, md=6),
        dbc.Col(dbc.RadioItems(id="explorer-by", options=[{"label": "Entre clases", "value": "dx"},
                                                           {"label": "Entre fuentes", "value": "dataset"}],
                               value="dx", inline=True, inputClassName="btn-check",
                               labelClassName="btn btn-outline-primary btn-touch", labelCheckedClassName="active"), xs=12, md=6),
    ], className="g-2 align-items-center")


@eda_section("explorador", "Explorador de metadatos", controls=_explorer_controls,
             inputs=[("explorer-var", "value"), ("explorer-by", "value")],
             lead="Elige una variable y compárala entre clases o entre fuentes. Numéricas: ε²; categóricas: V de Cramér.")
def render_explorer(df: pd.DataFrame, summary: dict, var: str, by: str):
    if var == by:
        return html.P("Elige una variable distinta del agrupador.", className="muted")
    order, colors = (CLASSES, CLASS_COLORS) if by == "dx" else (SOURCES, SOURCE_COLORS)
    kind = EXPLORER_VARS.get(var, (None, "num"))[1]
    label = EXPLORER_VARS.get(var, ((summary.get("color_variables") or {}).get(var, var),))[0]
    sub = df.dropna(subset=[var]) if var in df.columns else df.iloc[:0]
    sub = sub[~sub[var].isin(["unknown"])] if kind == "cat" else sub
    if sub.empty:
        return html.P("No hay datos para esta variable con los filtros aplicados.", className="muted")
    if kind == "num":
        e = epsilon2(sub[var], sub[by])
        fig = go.Figure([go.Box(y=sub.loc[sub[by] == g, var], name=g, marker_color=colors[g], fillcolor=colors[g],
                                line_color=UI["dark"], boxpoints=False) for g in order if (sub[by] == g).any()])
        fig.update_layout(title=f"{label} por {'clase' if by == 'dx' else 'fuente'}: ε² = {fmt(e)}", yaxis_title=label,
                          showlegend=False)
    else:
        v = cramers_v(sub[var], sub[by])
        tab = pd.crosstab(sub[by], sub[var], normalize="index").reindex(index=[g for g in order if g in set(sub[by])]) * 100
        cats = list(tab.columns)
        palette = (CLASS_COLORS if var == "dx" else SOURCE_COLORS if var == "dataset" else None)
        seq = [OKABE_ITO[k] for k in ("blue", "orange", "green", "vermillion", "purple", "sky", "yellow", "black")]
        fig = go.Figure([go.Bar(y=tab.index, x=tab[c], name=str(c), orientation="h",
                                marker_color=(palette or {}).get(c, seq[i % len(seq)]),
                                hovertemplate=f"{c}: %{{x:.1f}} %<extra></extra>") for i, c in enumerate(cats)])
        fig.update_layout(barmode="stack", title=f"{label} por {'clase' if by == 'dx' else 'fuente'}: V de Cramér = {fmt(v)}",
                          xaxis_title="%", yaxis={"autorange": "reversed"}, legend={"font": {"size": 11}})
    return [static_graph(fig, 380), _interp_extra(EXPLORER_INTERP.get(var) or ("color_source_vs_class" if var in
                                                                            (summary.get("color_variables") or {}) else None))]


def _interp_extra(key):
    from core.components import interpretation

    return interpretation(key) if key else None


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------


def filter_bar():
    return html.Div(dbc.Row([
        dbc.Col([html.Div("Clases", className="filter-label", id="flt-cls-label"),
                 dbc.Checklist(id="flt-classes", options=[{"label": swatch(c, CLASS_COLORS[c]), "value": c} for c in CLASSES],
                               value=CLASSES, inline=True, className="touch-check")],
                xs=12, lg=6),
        dbc.Col([html.Div("Fuentes", className="filter-label", id="flt-src-label"),
                 dbc.Checklist(id="flt-sources", options=[{"label": swatch(s, SOURCE_COLORS[s]), "value": s} for s in SOURCES],
                               value=SOURCES, inline=True, className="touch-check")],
                xs=12, lg=5),
        dbc.Col(dbc.Button("Todo", id="flt-reset", color="secondary", outline=True, className="btn-touch w-100"), xs=12, lg=1),
    ], className="g-2 align-items-end"), className="filter-bar")


def layout(**_):
    if data.images().empty:
        return pending("EDA sin datos", "Genera los datos con: python -m src.export_dashboard (desde la raíz del repositorio).")
    summary = data.eda_summary()
    sections = [section(s.id, s.title, html.Div(id=f"{s.id}-content"), s.lead, s.interp,
                        s.controls() if s.controls else None) for s in SECTIONS]
    return html.Div([
        html.H1("Análisis exploratorio", className="page-title"),
        html.P(f"{summary.get('n_images', 0):,} imágenes de {summary.get('n_lesions', 0):,} lesiones. Los filtros se "
               "aplican a todas las secciones.".replace(",", " "), className="page-lead"),
        dcc.Store(id=FILTERS, storage_type="memory", data={"classes": CLASSES, "sources": SOURCES}),
        filter_bar(),
        html.Div(sections, className="sections"),
    ])


@callback(Output(FILTERS, "data"), Input("flt-classes", "value"), Input("flt-sources", "value"))
def update_filters(classes, sources):
    return {"classes": classes or CLASSES, "sources": sources or SOURCES}


@callback(Output("flt-classes", "value"), Output("flt-sources", "value"), Input("flt-reset", "n_clicks"),
          prevent_initial_call=True)
def reset_filters(_):
    return CLASSES, SOURCES


def _register(sec: Section) -> None:
    @callback(Output(f"{sec.id}-content", "children"), Input(FILTERS, "data"),
              *[Input(cid, prop) for cid, prop in sec.inputs])
    def _update(filters, *values):
        try:
            return sec.render(filtered(filters), data.eda_summary(), *values)
        except Exception as e:  # noqa: BLE001 - se muestra al usuario en lugar de romper la página
            return error_alert(f"No se pudo dibujar esta sección: {e}")


for _s in SECTIONS:
    _register(_s)
