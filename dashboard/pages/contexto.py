"""Contexto: problema, cifra clave, preguntas, hipótesis, ficha de datos, pipeline, limitaciones, ética y créditos."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from core import data
from core.components import kpi, section
from core.config import KEY_FIGURE, SUBTITLE, TITLE
from dash import dcc, html

dash.register_page(__name__, path="/", name="Contexto", order=0, title=f"Contexto · {TITLE}")

QUESTIONS = [
    ("RQ1", "confirmatoria", "¿Supera alguno de los cinco modelos del estado del arte al mejor modelo base en al menos δ = 0,02 de exactitud balanceada?"),
    ("RQ2", "confirmatoria", "¿Supera cada modelo profundo a la regresión logística sobre embeddings congelados en al menos δ?"),
    ("RQ3", "exploratoria", "¿Depende el desempeño de la fuente de adquisición (vidir_molemax frente al resto)?"),
    ("R3", "Entregable 3", "¿Reduce la capa de constancia de color invariante (CCI) la dependencia de la fuente sin empeorar la exactitud balanceada global?"),
]

PREDICTIONS = [
    "Bajo ganancias por canal entre 0,5 y 2, la exactitud balanceada del modelo con la capa no cambia (salvo por saturación); la del backbone sin la capa cae.",
    "La ventaja desaparece al retirar la invarianza (pesos calculados sobre la imagen sin normalizar).",
    "La capa no aporta ventaja frente a desenfoque, ruido gaussiano o compresión JPEG.",
    "Un clasificador lineal predice peor la fuente a partir de las representaciones del modelo con la capa.",
]

PIPELINE = [
    ("HAM10000", "10 015 imágenes, 7 clases, 4 fuentes"),
    ("Partición", "agrupada por lesión: test 14 % y 5 pliegues"),
    ("Entrada", "224 × 224, RGB en [0, 255]"),
    ("Capa CCI", "iluminante por media de Minkowski ponderada, invariante a ganancias"),
    ("Backbone", "preentrenado en ImageNet, ajuste en dos fases"),
    ("Calibración", "temperatura y conjunto conforme"),
    ("Despliegue", "ONNX en CPU, este dashboard"),
]

LIMITATIONS = [
    "Un solo conjunto de datos de dos centros y poblaciones mayoritariamente de piel clara; el fototipo no está registrado.",
    "La invarianza es exacta frente a ganancias diagonales de canal, no frente al cambio de fuente real (saturación, gamma, viñeteado).",
    "No hay identificador de paciente: la partición agrupa por lesión, no por persona.",
    "El 47 % de las etiquetas no tiene confirmación histopatológica (seguimiento, consenso o confocal).",
    "La búsqueda de novedad usó una sola cadena en Scopus y GitHub (declarado en el capítulo 23).",
]

ETHICS = [
    "Uso exclusivamente académico: el modelo no está validado clínicamente y no debe orientar decisiones de biopsia o vigilancia.",
    "El desempeño se reporta desagregado por fuente, sexo, edad y localización para detectar desempeños desiguales.",
    "HAM10000 está anonimizado y se usa bajo CC BY-NC 4.0; no se redistribuye con fines comerciales.",
    "Las imágenes cargadas en el dashboard no se guardan ni se registran.",
]

CREDITS = [
    "Tschandl, P., Rosendahl, C. y Kittler, H. (2018). The HAM10000 dataset. Scientific Data, 5, 180161. doi: 10.1038/sdata.2018.161",
    "Pınar, A. et al. (2026). A reproducible and explainable deep learning framework for dermoscopic image analysis in skin cancer. Journal of Clinical Medicine, 15(19), 7422.",
    "Finlayson, G. D. y Trezzi, E. (2004). Shades of gray and colour constancy. Color and Imaging Conference, 37–41.",
    "Hu, Y., Wang, B. y Lin, S. (2017). FC4: Fully convolutional color constancy with confidence-weighted pooling. CVPR, 4085–4094.",
    "Barata, C., Celebi, M. E. y Marques, J. S. (2015). Improving dermoscopy image classification using color constancy. IEEE JBHI, 19(3), 1146–1152.",
    "Okabe, M. e Ito, K. (2008). Color Universal Design: paleta apta para daltonismo.",
]


def pipeline_diagram():
    steps = []
    for i, (name, desc) in enumerate(PIPELINE):
        cls = "pipe-step pipe-step-key" if name == "Capa CCI" else "pipe-step"
        steps.append(html.Li([html.Span(f"{i + 1}", className="pipe-num"), html.Strong(name), html.Span(desc)], className=cls))
    return html.Ol(steps, className="pipeline", **{"aria-label": "Pipeline del proyecto"})


def data_card_table():
    rows = (data.eda_summary().get("data_card") or [])
    if not rows:
        return html.P("La ficha se genera con python -m src.export_dashboard.", className="muted")
    return dbc.Table([html.Tbody([html.Tr([html.Th(r["campo"], scope="row"), html.Td(r["valor"])]) for r in rows])],
                     bordered=False, striped=True, responsive=True, className="data-card")


def _color_kpi(summary: dict):
    """ε² de la saturación según la fuente frente a según la clase (hallazgo que motiva el modelo)."""
    from core.components import fmt

    row = next((r for r in summary.get("epsilon2", []) if r["variable"] == "saturacion"), None)
    if row is None:
        return kpi("—", "ε² de la saturación por fuente y por clase", "python -m src.export_dashboard")
    return kpi(fmt(row["epsilon2_fuente"]), f"ε² de la saturación según la fuente, frente a {fmt(row['epsilon2_clase'])} "
               "según la clase", "Kruskal–Wallis en train (notebook 06)")


def layout(**_):
    s = data.eda_summary()
    hero = html.Header([
        html.H1(TITLE, className="hero-title"),
        html.P(SUBTITLE, className="hero-subtitle"),
        dbc.Row([
            dbc.Col(kpi(KEY_FIGURE["value"], KEY_FIGURE["label"], f"Fuente: {KEY_FIGURE['source']}", tone="alert"), xs=12, md=6),
            dbc.Col(kpi(f"{s.get('n_images', 10015):,}".replace(",", " "), "imágenes dermatoscópicas de "
                        f"{s.get('n_lesions', 7470):,}".replace(",", " ") + " lesiones en HAM10000",
                        "Tschandl et al. (2018)"), xs=6, md=3),
            dbc.Col(_color_kpi(s), xs=6, md=3),
        ], className="g-3 hero-kpis"),
    ], className="hero")

    problem = section("problema", "Problema", [
        html.P("La dermatoscopia permite detectar el melanoma y otras lesiones a tiempo, pero su precisión depende de la "
               "experiencia del observador. Los clasificadores profundos entrenados en HAM10000 reportan exactitudes "
               "altas, pero bajo protocolos estrictos rondan 0,70 de exactitud balanceada y caen al cambiar de "
               "institución. El análisis exploratorio muestra que, dentro de HAM10000, el color depende mucho más de "
               "la fuente de adquisición que de la clase: el modelo puede aprender el equipo en lugar de la lesión."),
    ])
    questions = section("preguntas", "Preguntas de investigación", html.Ul([
        html.Li([html.Strong(f"{q} "), html.Span(f"({kind}) ", className="muted"), text]) for q, kind, text in QUESTIONS
    ], className="list-spaced"))
    hypothesis = section("hipotesis", "Hipótesis del mecanismo", [
        html.P("La capa CCI reduce la dependencia del desempeño respecto de la fuente porque elimina por construcción las "
               "ganancias de canal propias de cada equipo, y reduce en al menos δ = 0,02 la diferencia de exactitud "
               "balanceada entre vidir_molemax y el resto de fuentes respecto del mismo backbone sin la capa."),
        html.P("Predicciones secundarias declaradas a priori:", className="mb-1"),
        html.Ol([html.Li(p) for p in PREDICTIONS], className="list-spaced"),
    ])
    card = section("ficha", "Ficha del conjunto de datos", data_card_table())
    pipe = section("pipeline", "Pipeline", pipeline_diagram(),
                   lead="De la imagen al resultado; la capa CCI es la contribución original.")
    lim = section("limitaciones", "Limitaciones", html.Ul([html.Li(x) for x in LIMITATIONS], className="list-spaced"))
    eth = section("etica", "Ética", html.Ul([html.Li(x) for x in ETHICS], className="list-spaced"))
    cred = section("creditos", "Créditos", [
        html.Ul([html.Li(x) for x in CREDITS], className="list-refs"),
        dcc.Markdown("Proyecto de Deep Learning, Universidad del Norte (2026). Código en MIT; datos bajo su propia licencia.",
                     className="muted"),
    ])
    return html.Div([
        hero,
        dbc.Row([dbc.Col(problem, lg=6), dbc.Col(questions, lg=6)], className="g-3"),
        pipe,
        hypothesis,
        dbc.Row([dbc.Col(card, lg=6), dbc.Col([lim, eth], lg=6)], className="g-3"),
        cred,
    ])
