"""Modelo original: capa de constancia de color invariante a las ganancias de canal (CCI).

La capa se antepone a cualquier modelo de src.models.REGISTRY (build_cci_model) y se entrena de
extremo a extremo con la pérdida de clasificación. Formulación (book/24-modelo-original-diseno.md):
sea x ∈ R_{>0}^{N×3} la imagen y x̄ = pool(x) su versión reducida a N̄ = 56 × 56 píxeles por
promedio de áreas; la capa calcula

    (cci-guia)       z_{i,c}(x̄) = log x̄_{i,c} − (1/N̄) Σ_j log x̄_{j,c}
    (cci-pesos)      w_i(x̄) = exp(g_φ(z)_i) / Σ_j exp(g_φ(z)_j),      p = 1 + softplus(ϑ)
    (cci-iluminante) e_c(x̄) = ( Σ_i w_i(x̄) x̄_{i,c}^p )^{1/p}
    (cci-correccion) h(x)_{i,c} = κ x_{i,c} / e_c(x̄)

con g_φ = Conv 3×3 (16, ReLU) → Conv 3×3 (16, ReLU) → Conv 1×1 (1). La media de Minkowski de
orden p es la de Shades of Gray (Finlayson y Trezzi, 2004, finlayson2004); los pesos espaciales
aprendidos, la ponderación por confianza de FC4 (Hu, Wang y Lin, 2017, hu2017), aquí sin
iluminantes etiquetados y calculados sobre z, que no cambia con las ganancias de canal. Por eso
h(Dx) = h(x) para toda D = diag(d_1, d_2, d_3) con d_c > 0 (teorema del capítulo 24).

Detalles numéricos (capítulo 24):
- Las intensidades se acotan por debajo con ε antes de reducir y de tomar logaritmos.
- El iluminante se calcula en el dominio logarítmico,
      log e_c = (1/p) logsumexp_i(log w_i + p log x̄_{i,c}),
  con log w = log_softmax(g_φ(z)) (sin pasar por w, que puede ser muy pequeño).
- ϑ se inicializa en softplus⁻¹(p_init − 1), de modo que p = 6, y la última convolución de g_φ en
  cero, de modo que los pesos empiezan uniformes: la capa empieza como Shades of Gray con p = 6.
- La capa calcula siempre en float32 (también con mixed_float16), como la capa de preprocesamiento.

Variantes de ablación del capítulo 24 (VARIANTS; flags de CCILayer):

    full            p aprendido, pesos aprendidos sobre z (modelo completo)
    fixed_p         p fijo en p_init (learn_p=False)
    uniform_weights pesos uniformes, sin g_φ (learn_weights=False)
    no_invariance   g_φ recibe la imagen reducida sin normalizar, x̄ / 255 (invariant=False):
                    mismos parámetros que full, pero los pesos cambian con las ganancias
    shades_of_gray  p fijo y pesos uniformes (Shades of Gray fijo, barata2015)

Uso:

    from src.original_model import build_cci_model
    model = build_cci_model("convnext_tiny", "full")      # o un keras.Model ya construido
    info = model.get_layer(CCI_NAME).explain(images)      # p, pesos espaciales e iluminante
"""

from __future__ import annotations

import math
from pathlib import Path

import keras
import numpy as np
import yaml
from keras import ops

CCI_NAME = "color_constancy"          # nombre de la capa dentro del modelo
THETA_NAME = "cci_theta"              # nombre de ϑ (excluido del decaimiento de pesos)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "cci.yaml"

VARIANTS = {
    "full": {"learn_p": True, "learn_weights": True, "invariant": True},
    "fixed_p": {"learn_p": False, "learn_weights": True, "invariant": True},
    "uniform_weights": {"learn_p": True, "learn_weights": False, "invariant": True},
    "no_invariance": {"learn_p": True, "learn_weights": True, "invariant": False},
    "shades_of_gray": {"learn_p": False, "learn_weights": False, "invariant": True},
}

# Claves de configs/<backbone>.yaml que el modelo CCI hereda si su YAML no las fija.
INHERITED_KEYS = ("gradcam_layer", "saliency", "keras_hub_preset")


def inverse_softplus(y: float) -> float:
    """ϑ tal que softplus(ϑ) = y (y > 0)."""
    if y <= 0:
        raise ValueError(f"softplus⁻¹ requiere y > 0, no {y}")
    return float(y + math.log(-math.expm1(-y)))


def load_cci_config(path: str | Path | None = None) -> dict:
    """Sección `cci` de configs/cci.yaml (parámetros de la capa)."""
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return dict((yaml.safe_load(f) or {}).get("cci") or {})


def area_pool_matrix(n_in: int, n_out: int) -> np.ndarray:
    """Matriz (n_out, n_in) del promedio de áreas en una dimensión: la fila i promedia el intervalo
    [i n_in / n_out, (i + 1) n_in / n_out) con pesos proporcionales a la parte de cada píxel que
    cae en él (lo mismo que tf.image.resize(method="area")). Lineal y por canal."""
    if n_out > n_in:
        raise ValueError(f"El promedio de áreas reduce: n_out = {n_out} > n_in = {n_in}")
    m = np.zeros((n_out, n_in), dtype=np.float64)
    scale = n_in / n_out
    for i in range(n_out):
        lo, hi = i * scale, (i + 1) * scale
        for j in range(math.floor(lo), math.ceil(hi)):
            m[i, j] = min(hi, j + 1) - max(lo, j)
    return m / m.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Capa
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="skin_lesion_dl")
class CCILayer(keras.layers.Layer):
    """Constancia de color aprendible e invariante a las ganancias de canal (capítulo 24).

    Entrada: imágenes (B, H, W, 3) en punto flotante con intensidades en `input_range` ([0, 255]).
    Salida: h(x) de la misma forma, ecuación (cci-correccion).

    Ecuaciones (x̄ = promedio de áreas de max(x, ε) a reduced_size × reduced_size):

        z_{i,c} = log x̄_{i,c} − mean_j log x̄_{j,c}                       (cci-guia)
        w = softmax_i(g_φ(z)_i),   p = 1 + softplus(ϑ)                      (cci-pesos)
        log e_c = (1/p) logsumexp_i(log w_i + p log x̄_{i,c})              (cci-iluminante)
        h_{i,c} = κ x_{i,c} / e_c                                           (cci-correccion)

    La media de Minkowski ponderada es la de Shades of Gray (finlayson2004) con p aprendible y
    pesos espaciales como el pooling ponderado por confianza de FC4 (hu2017), sin supervisión de
    iluminante. Flags de ablación: `learn_p` (False: p fijo en p_init), `learn_weights` (False:
    pesos uniformes y sin g_φ) e `invariant` (False: g_φ recibe la imagen reducida sin
    normalizar, x̄ / 255, de modo que los pesos dependen de las ganancias y la invarianza se
    pierde). No se usa log x̄ sin centrar como variante: una ganancia es entonces un desplazamiento
    constante por canal, que las convoluciones propagan casi como una constante espacial y la
    softmax anula, así que esa variante sería casi invariante y no pondría a prueba la predicción 2.

    Validación: el número de canales y el rango de la dimensión se comprueban al construir
    (ValueError); el tipo, en cada llamada (TypeError si no es de punto flotante); el rango de
    intensidades, en las llamadas en modo eager (ValueError fuera de `input_range`; en el grafo
    de entrenamiento no se comprueba, porque el pipeline ya recorta a [0, 255]).
    """

    def __init__(
        self,
        reduced_size: int = 56,
        filters: int = 16,
        kernel_size: int = 3,
        eps: float = 1e-3,
        kappa: float = 170.0,
        p_init: float = 6.0,
        learn_p: bool = True,
        learn_weights: bool = True,
        invariant: bool = True,
        input_range=(0.0, 255.0),
        **kwargs,
    ):
        kwargs.setdefault("dtype", "float32")
        super().__init__(**kwargs)
        if p_init < 1:
            raise ValueError(f"p_init debe ser >= 1 (p = 1 + softplus(ϑ)), no {p_init}")
        if eps <= 0 or kappa <= 0:
            raise ValueError("eps y kappa deben ser positivos")
        self.reduced_size = int(reduced_size)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.eps = float(eps)
        self.kappa = float(kappa)
        self.p_init = float(p_init)
        self.learn_p = bool(learn_p)
        self.learn_weights = bool(learn_weights)
        self.invariant = bool(invariant)
        self.input_range = (float(input_range[0]), float(input_range[1]))
        self.weight_net = None
        if self.learn_weights:
            policy = self.dtype_policy
            last = keras.layers.Conv2D(1, 1, kernel_initializer="zeros", bias_initializer="zeros",
                                       dtype=policy, name="g_out")
            self.weight_net = keras.Sequential([
                keras.layers.Conv2D(self.filters, self.kernel_size, padding="same", activation="relu",
                                    dtype=policy, name="g_conv1"),
                keras.layers.Conv2D(self.filters, self.kernel_size, padding="same", activation="relu",
                                    dtype=policy, name="g_conv2"),
                last,
            ], name="g_phi")

    # -- construcción ---------------------------------------------------------

    @staticmethod
    def _check_shape(shape) -> None:
        shape = tuple(shape)
        if len(shape) != 4:
            raise ValueError(f"CCILayer espera imágenes (B, H, W, 3); forma recibida {shape}")
        if shape[-1] != 3:
            raise ValueError(f"CCILayer espera 3 canales RGB; la entrada tiene {shape[-1]} (forma {shape})")

    def build(self, input_shape):
        self._check_shape(input_shape)
        h, w = input_shape[1], input_shape[2]
        if h is None or w is None:
            raise ValueError("CCILayer necesita alto y ancho estáticos para el promedio de áreas")
        r = self.reduced_size
        if h < r or w < r:
            raise ValueError(f"La entrada ({h} x {w}) es menor que reduced_size = {r}")
        self._spatial = (int(h), int(w))
        # Promedio de áreas: AveragePooling2D si el lado es múltiplo de reduced_size (caso del
        # proyecto: 224 = 4 x 56); si no, el producto por las matrices de area_pool_matrix.
        self._pool = None
        if h % r == 0 and w % r == 0:
            self._pool = keras.layers.AveragePooling2D(pool_size=(h // r, w // r), dtype=self.dtype_policy,
                                                       name="area_pool")
        else:
            self._pool_h = ops.convert_to_tensor(area_pool_matrix(h, r), dtype=self.compute_dtype)
            self._pool_w = ops.convert_to_tensor(area_pool_matrix(w, r), dtype=self.compute_dtype)
        self.theta = self.add_weight(
            name=THETA_NAME, shape=(),
            initializer=keras.initializers.Constant(inverse_softplus(self.p_init - 1.0)),
            trainable=self.learn_p,
        )
        if self.weight_net is not None:
            self.weight_net.build((None, r, r, 3))
        self.built = True

    # -- validación -----------------------------------------------------------

    def _validate(self, x):
        self._check_shape(x.shape)
        if self.built and tuple(x.shape[1:3]) != self._spatial:
            raise ValueError(f"La capa se construyó para {self._spatial[0]} x {self._spatial[1]} y recibió "
                             f"{x.shape[1]} x {x.shape[2]}")
        dtype = keras.backend.standardize_dtype(x.dtype)
        if "float" not in dtype:
            raise TypeError(f"CCILayer espera imágenes en punto flotante (float32) con intensidades en "
                            f"{list(self.input_range)}; recibió dtype {dtype}. Convierte con "
                            f"x.astype('float32') sin normalizar.")
        if _is_eager(x):
            arr = keras.ops.convert_to_numpy(x)
            lo, hi = self.input_range
            if not np.all(np.isfinite(arr)):
                raise ValueError("La entrada de CCILayer tiene valores no finitos (NaN o inf)")
            if arr.size and (arr.min() < lo or arr.max() > hi):
                raise ValueError(f"Intensidades fuera de {list(self.input_range)}: mínimo {arr.min():.4g}, "
                                 f"máximo {arr.max():.4g}. La capa espera la imagen en [0, 255], sin normalizar.")

    # -- paso hacia adelante --------------------------------------------------

    @property
    def p(self):
        """p = 1 + softplus(ϑ), ecuación (cci-pesos)."""
        return 1.0 + ops.softplus(ops.cast(self.theta, self.compute_dtype))

    def area_pool(self, x):
        """x̄ = pool(x): promedio de áreas a reduced_size × reduced_size (lineal y por canal)."""
        if self._pool is not None:
            return self._pool(x)
        return ops.einsum("ih,bhwc,jw->bijc", self._pool_h, x, self._pool_w)

    def guide(self, log_xbar):
        """Representación de entrada de g_φ: z (cci-guia) si `invariant`; si no, la imagen reducida
        sin normalizar en [0, 1], x̄ / max(input_range)."""
        if not self.invariant:
            return ops.exp(log_xbar) / self.input_range[1]
        return log_xbar - ops.mean(log_xbar, axis=(1, 2), keepdims=True)

    def log_weights(self, log_xbar):
        """log w (B, N̄): log_softmax espacial de g_φ(z), ecuación (cci-pesos); uniforme sin g_φ."""
        b = ops.shape(log_xbar)[0]
        n = self.reduced_size * self.reduced_size
        if self.weight_net is None:
            return ops.reshape(ops.zeros_like(log_xbar[..., 0]), (b, n)) - math.log(n)
        scores = ops.reshape(self.weight_net(self.guide(log_xbar)), (b, n))
        return scores - ops.logsumexp(scores, axis=1, keepdims=True)

    @staticmethod
    def log_illuminant(log_w, log_xbar_flat, p):
        """log e_c = (1/p) logsumexp_i(log w_i + p log x̄_{i,c}), ecuación (cci-iluminante) en el
        dominio logarítmico. log_w: (B, N̄); log_xbar_flat: (B, N̄, 3); p escalar. Devuelve (B, 3)."""
        return ops.logsumexp(log_w[:, :, None] + p * log_xbar_flat, axis=1) / p

    def _forward(self, x):
        x = ops.maximum(ops.cast(x, self.compute_dtype), self.eps)        # intensidades positivas
        log_xbar = ops.log(self.area_pool(x))                              # x̄ >= ε
        b = ops.shape(x)[0]
        log_w = self.log_weights(log_xbar)
        p = self.p
        log_e = self.log_illuminant(log_w, ops.reshape(log_xbar, (b, -1, 3)), p)
        h = self.kappa * x * ops.exp(-log_e)[:, None, None, :]             # (cci-correccion)
        return h, {"p": p, "log_weights": log_w, "log_illuminant": log_e}

    def call(self, inputs):
        self._validate(inputs)
        return self._forward(inputs)[0]

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    # -- visualización --------------------------------------------------------

    def explain(self, images) -> dict:
        """p, pesos espaciales, iluminante estimado e imagen corregida de `images` (eager).

        Devuelve un dict de arrays NumPy: p (escalar), weights (B, r, r) que suman 1 por imagen,
        illuminant (B, 3) = e_c en la escala de la entrada, corrected (B, H, W, 3) = h(x) y
        reduced (B, r, r, 3) = x̄.
        """
        x = ops.convert_to_tensor(np.asarray(images, dtype=np.float32))
        if len(x.shape) == 3:
            x = x[None]
        if not self.built:
            self.build(x.shape)
        self._validate(x)
        h, parts = self._forward(x)
        r = self.reduced_size
        xbar = self.area_pool(ops.maximum(x, self.eps))
        to_np = keras.ops.convert_to_numpy
        return {
            "p": float(to_np(parts["p"])),
            "weights": np.exp(to_np(parts["log_weights"])).reshape(-1, r, r),
            "illuminant": np.exp(to_np(parts["log_illuminant"])),
            "corrected": to_np(h),
            "reduced": to_np(xbar),
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            "reduced_size": self.reduced_size, "filters": self.filters, "kernel_size": self.kernel_size,
            "eps": self.eps, "kappa": self.kappa, "p_init": self.p_init, "learn_p": self.learn_p,
            "learn_weights": self.learn_weights, "invariant": self.invariant,
            "input_range": list(self.input_range),
        })
        return config


def _is_eager(x) -> bool:
    """True si `x` es un tensor concreto (no simbólico ni de un grafo en trazado)."""
    if isinstance(x, np.ndarray):
        return True
    if isinstance(x, keras.KerasTensor):
        return False
    try:
        import tensorflow as tf
    except ImportError:
        return False
    return tf.executing_eagerly() and hasattr(x, "numpy")


# ---------------------------------------------------------------------------
# Modelo completo
# ---------------------------------------------------------------------------


def cci_layer(variant: str = "full", layer_cfg: dict | None = None, **overrides) -> CCILayer:
    """CCILayer con los parámetros de configs/cci.yaml (o `layer_cfg`) y los flags de `variant`."""
    if variant not in VARIANTS:
        raise ValueError(f"Variante desconocida: {variant!r}. Disponibles: {sorted(VARIANTS)}")
    params = dict(load_cci_config() if layer_cfg is None else layer_cfg)
    params.update(VARIANTS[variant])
    params.update(overrides)
    return CCILayer(name=CCI_NAME, **params)


def _backbone_model(backbone, n_classes: int, dropout: float, weights) -> keras.Model:
    """Modelo del REGISTRY (por nombre) o un keras.Model ya construido."""
    if isinstance(backbone, keras.Model):
        return backbone
    if not isinstance(backbone, str):
        raise TypeError(f"backbone debe ser el nombre de un modelo del REGISTRY o un keras.Model, no {type(backbone)}")
    from src.models import REGISTRY
    from src.train import load_model_config

    if backbone not in REGISTRY or backbone.startswith("cci"):
        raise ValueError(f"Backbone {backbone!r} no válido. Disponibles: "
                         f"{sorted(m for m in REGISTRY if not m.startswith('cci'))}")
    bcfg = load_model_config(backbone)
    if weights != "default":
        bcfg = dict(bcfg, weights=weights)
    return REGISTRY[backbone](bcfg, n_classes, dropout)


def build_cci_model(
    backbone,
    variant: str = "full",
    *,
    n_classes: int = 7,
    dropout: float = 0.0,
    weights="default",
    layer_cfg: dict | None = None,
    name: str | None = None,
    **layer_overrides,
) -> keras.Model:
    """Antepone la capa CCI a un clasificador del REGISTRY sin modificarlo: f = softmax ∘ b_ψ ∘ h.

    `backbone`: nombre de un modelo del REGISTRY (se construye con configs/<nombre>.yaml, sus
    pesos de ImageNet, `n_classes` y `dropout`) o un keras.Model con la estructura de
    src/models/common.py (Input -> preprocess -> backbone -> cabeza). El modelo resultante es

        Input -> CCILayer -> [mismas capas del clasificador, mismos objetos y pesos]

    es decir, se vuelve a trazar la cadena lineal del clasificador sobre la salida de la capa, con
    el backbone llamado con training=False como en el original. Las capas se comparten (no se
    copian), así que get_backbone, set_backbone_trainable, head_layers y Grad-CAM funcionan igual
    y la fase A (backbone congelado) entrena la capa y la cabeza.
    """
    from src.models.common import BACKBONE_NAME

    base = _backbone_model(backbone, n_classes, dropout, weights)
    layers = [lyr for lyr in base.layers if not isinstance(lyr, keras.layers.InputLayer)]
    _check_linear_chain(base, layers)
    shape = tuple(base.inputs[0].shape[1:])
    inputs = keras.Input(shape=shape, dtype="float32", name="image")
    x = cci_layer(variant, layer_cfg, **layer_overrides)(inputs)
    for layer in layers:
        x = layer(x, training=False) if layer.name == BACKBONE_NAME else layer(x)
    return keras.Model(inputs, x, name=name or f"cci_{base.name}")


def _check_linear_chain(model: keras.Model, layers) -> None:
    """Error si el clasificador no es una cadena lineal Input -> capa -> ... -> salida."""
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ValueError(f"{model.name}: build_cci_model requiere una entrada y una salida")
    prev = model.inputs[0]
    for layer in layers:
        nodes = [n for n in layer._inbound_nodes if any(t is prev for t in n.input_tensors)]
        if len(nodes) != 1 or len(nodes[0].input_tensors) != 1:
            raise ValueError(f"{model.name}: la capa {layer.name} no continúa una cadena lineal; "
                             "build_cci_model solo admite clasificadores Input -> ... -> salida")
        prev = nodes[0].output_tensors[0]
    if prev is not model.outputs[0]:
        raise ValueError(f"{model.name}: la cadena de capas no termina en la salida del modelo")


def get_cci_layer(model: keras.Model) -> CCILayer:
    return model.get_layer(CCI_NAME)


def build_from_config(model_cfg: dict, n_classes: int, dropout: float) -> keras.Model:
    """Constructor del REGISTRY ("cci" y variantes): backbone, variante y parámetros de la capa de
    configs/cci*.yaml. `model_cfg["weights"]` (p. ej. None en src.gradcam o src.cost) se pasa al
    backbone."""
    backbone = model_cfg.get("backbone")
    if not backbone:
        raise ValueError(
            f"configs/{model_cfg.get('name', 'cci')}.yaml tiene backbone: null. Fíjalo con la regla del "
            "capítulo 24: el modelo del benchmark con mayor exactitud balanceada media en validación en el "
            "Entregable 2 (book/20-benchmark-e2.ipynb)."
        )
    model = build_cci_model(backbone, model_cfg.get("variant", "full"), n_classes=n_classes, dropout=dropout,
                            weights=model_cfg.get("weights", "imagenet"), layer_cfg=model_cfg.get("cci"),
                            name=model_cfg.get("name"))
    return model


def resolve_model_config(model_cfg: dict, load_config) -> dict:
    """Completa la config de un modelo CCI con las claves de su backbone (INHERITED_KEYS) que no
    fija. `load_config(nombre)` lee configs/<nombre>.yaml."""
    backbone = model_cfg.get("backbone")
    if not backbone:
        return model_cfg
    bcfg = load_config(backbone)
    out = dict(model_cfg)
    for key in INHERITED_KEYS:
        if out.get(key) is None and bcfg.get(key) is not None:
            out[key] = bcfg[key]
    if int(out.get("input_size", bcfg["input_size"])) != int(bcfg["input_size"]):
        raise ValueError(f"{out['name']}: input_size {out['input_size']} != {bcfg['input_size']} de {backbone}")
    return out
