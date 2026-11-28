"""Auditoría de fuga de información y contrato de datos (book/21-auditoria-fuga.ipynb).

Cada verificación devuelve una fila con su criterio de aprobación, el valor observado y el
resultado («aprobado», «fallido» o «no disponible» si faltan datos), además de sus tablas de
detalle. Parámetros: sección `audit` de base.yaml y configs/data_contract.yaml.

Verificaciones:
1. lesion_id (y group_id) sin intersección entre train y validación de cada pliegue y entre el
   conjunto de entrenamiento y test.
2. SHA-256 de los archivos de imagen sin colisiones entre particiones.
3. Vecinos casi duplicados de test en train por pHash (distancia de Hamming <= phash_threshold)
   y por similitud coseno de embeddings (>= audit.cosine_threshold): proporción de imágenes de
   test con vecino y desempeño de los modelos con y sin esos casos.
4. AUC univariado de cada metadato frente a la clase (uno contra el resto, macro), con la
   relación categoría-clase estimada en train y evaluada en test.
5. Validación adversarial train frente a test (src.eda.adversarial_validation) sobre embeddings.
6. Prueba de sanidad con etiquetas permutadas sobre la regresión logística.
7. Pruebas con datos sintéticos de que ninguna transformación se ajusta fuera de train.
8. Contrato de datos: esquema, diccionario de atributos y conteo de observaciones por población.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import data as D
from src.report import Missing, is_missing, tolerant

APROBADO, FALLIDO, NO_DISPONIBLE = "aprobado", "fallido", "no disponible"


def check(name: str, criterion: str, value, passed: bool | None, detail: str = "") -> dict:
    return {"verificacion": name, "criterio": criterion, "valor": value,
            "resultado": NO_DISPONIBLE if passed is None else (APROBADO if passed else FALLIDO), "detalle": detail}


def summary(checks) -> pd.DataFrame:
    """Tabla de verificaciones (una fila por dict de `check`; los Missing pasan a «no disponible»)."""
    rows = []
    for c in checks:
        if isinstance(c, Missing):
            rows.append(check("—", "—", None, None, c.message))
        elif isinstance(c, list):
            rows += c
        else:
            rows.append(c)
    return pd.DataFrame(rows).set_index("verificacion")


# ---------------------------------------------------------------------------
# 1. Intersección de lesiones
# ---------------------------------------------------------------------------


def lesion_overlap(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Lesiones y grupos compartidos entre train y val de cada pliegue y entre train y test."""
    s = D.load_splits(cfg["paths"]["splits_csv"])
    rows = []
    pool, test = s[s["split"] == "train"], s[s["split"] == "test"]
    for col in ("lesion_id", "group_id"):
        rows.append({"comparacion": "entrenamiento (todos los pliegues) - test", "columna": col,
                     "compartidas": len(set(pool[col]) & set(test[col]))})
        for f in sorted(pool["fold"].unique()):
            tr, va = pool[pool["fold"] != f], pool[pool["fold"] == f]
            rows.append({"comparacion": f"train - validación (pliegue {f})", "columna": col,
                         "compartidas": len(set(tr[col]) & set(va[col]))})
    t = pd.DataFrame(rows).set_index(["comparacion", "columna"])
    total = int(t["compartidas"].sum())
    return t, check("Lesiones sin intersección entre particiones", "0 lesion_id y 0 group_id compartidos",
                    total, total == 0)


# ---------------------------------------------------------------------------
# 2. SHA-256 por imagen
# ---------------------------------------------------------------------------


def image_hashes(cfg: dict, compute: bool = False) -> pd.DataFrame | Missing:
    """SHA-256 de cada archivo de imagen (cache en paths.image_sha256_csv; se calcula con `compute`)."""
    path = Path(cfg["paths"]["image_sha256_csv"])
    s = D.load_splits(cfg["paths"]["splits_csv"])
    if path.exists():
        h = pd.read_csv(path)
        if set(s["image_id"]) <= set(h["image_id"]):
            return h
    if not compute:
        return Missing(f"no existe {path.name}; pon CALCULAR_SHA256 = True (lee las {len(s)} imágenes).")
    from tqdm import tqdm

    rows = [{"image_id": i, "sha256": D.sha256_file(D.image_path(cfg, i))} for i in tqdm(s["image_id"], desc="SHA-256")]
    h = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    h.to_csv(path, index=False)
    return h


@tolerant()
def sha256_collisions(cfg: dict, hashes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    s = D.load_splits(cfg["paths"]["splits_csv"]).merge(hashes, on="image_id")
    s["particion"] = np.where(s["split"] == "test", "test", "train_f" + s["fold"].astype(str))
    dup = s[s.duplicated("sha256", keep=False)].sort_values("sha256")
    cross = dup.groupby("sha256")["particion"].nunique()
    cross_n = int((cross > 1).sum())
    detail = dup[["image_id", "lesion_id", "particion", "sha256"]].reset_index(drop=True)
    return detail, check("SHA-256 sin colisiones entre particiones", "0 hashes compartidos entre particiones",
                         cross_n, cross_n == 0, f"{dup['sha256'].nunique()} hashes repetidos en total")


# ---------------------------------------------------------------------------
# 3. Vecinos casi duplicados (pHash y embeddings)
# ---------------------------------------------------------------------------


def _test_train(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = D.load_splits(cfg["paths"]["splits_csv"])
    return s[s["split"] == "test"].sort_values("image_id"), s[s["split"] == "train"].sort_values("image_id")


def phash_neighbors(cfg: dict) -> pd.DataFrame | Missing:
    """Por imagen de test: distancia de Hamming mínima de pHash a train y su vecino."""
    path = cfg["paths"]["phash_csv"]
    if not Path(path).exists():
        return Missing(f"no existe {path}; se genera en book/05-eda_metadatos_particion.ipynb.")
    ph = pd.read_csv(path, dtype={"image_id": str, "phash": str}).set_index("image_id")["phash"]
    te, tr = _test_train(cfg)
    a = D._hex_to_bytes(ph.reindex(te["image_id"]).tolist())
    b = D._hex_to_bytes(ph.reindex(tr["image_id"]).tolist())
    best_d, best_j = [], []
    for start in range(0, len(a), 256):
        dist = np.bitwise_count(a[start:start + 256, None, :] ^ b[None, :, :]).sum(axis=2)
        best_j.append(dist.argmin(axis=1))
        best_d.append(dist.min(axis=1))
    j, d = np.concatenate(best_j), np.concatenate(best_d)
    return pd.DataFrame({"image_id": te["image_id"].to_numpy(), "vecino_train": tr["image_id"].to_numpy()[j],
                         "distancia": d}).set_index("image_id")


def embedding_neighbors(cfg: dict) -> pd.DataFrame | Missing:
    """Por imagen de test: similitud coseno máxima con train (embeddings de MobileNetV2) y su vecino."""
    from src.baselines import load_embeddings

    te, tr = _test_train(cfg)
    Xte, Xtr = load_embeddings(cfg, te["image_id"]), load_embeddings(cfg, tr["image_id"])
    if is_missing(Xte) or is_missing(Xtr):
        return Xte if is_missing(Xte) else Xtr
    Xte = Xte / np.linalg.norm(Xte, axis=1, keepdims=True)
    Xtr = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True)
    sim_max, idx = [], []
    for start in range(0, len(Xte), 512):
        sim = Xte[start:start + 512] @ Xtr.T
        idx.append(sim.argmax(axis=1))
        sim_max.append(sim.max(axis=1))
    j = np.concatenate(idx)
    return pd.DataFrame({"image_id": te["image_id"].to_numpy(), "vecino_train": tr["image_id"].to_numpy()[j],
                         "similitud": np.concatenate(sim_max)}).set_index("image_id")


@tolerant()
def neighbor_check(cfg: dict, neighbors: pd.DataFrame, kind: str) -> dict:
    """Proporción de imágenes de test con vecino en train según `kind` (phash o embeddings)."""
    a = cfg["audit"]
    if kind == "phash":
        flag = neighbors["distancia"] <= int(cfg["phash_threshold"])
        crit = f"distancia de Hamming <= {cfg['phash_threshold']}"
    else:
        flag = neighbors["similitud"] >= float(a["cosine_threshold"])
        crit = f"similitud coseno >= {a['cosine_threshold']}"
    frac = float(flag.mean())
    return check(f"Vecinos de test en train ({kind})", f"proporción con {crit} <= {a['max_neighbor_frac']}",
                 round(frac, 6), frac <= float(a["max_neighbor_frac"]), f"{int(flag.sum())} de {len(flag)} imágenes")


def neighbor_flags(cfg: dict, neighbors, kind: str) -> pd.Series | Missing:
    if is_missing(neighbors):
        return neighbors
    if kind == "phash":
        return neighbors["distancia"] <= int(cfg["phash_threshold"])
    return neighbors["similitud"] >= float(cfg["audit"]["cosine_threshold"])


@tolerant()
def performance_with_without(rs, cfg: dict, flags: pd.Series, kind: str) -> tuple[pd.DataFrame, dict]:
    """Exactitud balanceada de test (media entre corridas) por modelo con todas las imágenes y
    sin las que tienen vecino en train; criterio: la diferencia es menor que benchmark.delta."""
    from src import benchmark as B
    from src import stats as S

    td = B.test_data(rs, cfg)
    if is_missing(td):
        return td
    f = flags.reindex(td.image_ids).fillna(False).to_numpy(bool)
    delta = float(cfg["benchmark"]["delta"])
    rows = []
    one = np.ones((1, len(td.y)))
    for m in td.keys:
        c = td.correct(m)
        all_ = float(S.weighted_balanced_accuracy(one, td.y, c).mean())
        without = float(np.nanmean(S.weighted_balanced_accuracy(one, td.y, c, (~f).astype(float)))) if (~f).any() else np.nan
        only = float(np.nanmean(S.weighted_balanced_accuracy(one, td.y, c, f.astype(float)))) if f.any() else np.nan
        rows.append({"modelo": m, "todas": all_, "sin_vecinos": without, "solo_con_vecino": only,
                     "diferencia_todas_menos_sin": all_ - without})
    t = pd.DataFrame(rows).set_index("modelo")
    t.attrs["imagenes_con_vecino"] = int(f.sum())
    worst = float(t["diferencia_todas_menos_sin"].abs().max())
    return t, check(f"Desempeño sin vecinos de train ({kind})", f"|BA todas - BA sin vecinos| < {delta} en todos los modelos",
                    round(worst, 6), worst < delta, f"{int(f.sum())} imágenes con vecino")


# ---------------------------------------------------------------------------
# 4. AUC univariado de los metadatos
# ---------------------------------------------------------------------------


@tolerant()
def metadata_auc(cfg: dict, meta: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """AUC uno contra el resto de cada metadato para cada clase en test: en las categóricas, la
    puntuación de la clase k es su frecuencia en la categoría estimada en train; en la edad, la
    edad (se toma max(AUC, 1 - AUC)). AUC macro por variable y criterio audit.metadata_auc_max."""
    from sklearn.metrics import roc_auc_score

    te, tr = _test_train(cfg)
    m = meta.set_index("image_id") if "image_id" in meta.columns else meta
    classes = D.class_names(cfg)
    rows = {}
    for var in ("age", "sex", "localization", "dataset", "dx_type"):
        if var not in m.columns:
            continue
        row = {}
        for k in classes:
            yk = (te["dx"] == k).to_numpy()
            if yk.all() or not yk.any():
                continue
            if var == "age":
                s = m.loc[te["image_id"], var].astype(float).fillna(m.loc[tr["image_id"], var].median()).to_numpy()
                auc = roc_auc_score(yk, s)
                auc = max(auc, 1 - auc)
            else:
                train_cat = m.loc[tr["image_id"], var].astype(str).to_numpy()
                freq = pd.Series(tr["dx"].to_numpy() == k).groupby(train_cat).mean()
                s = pd.Series(m.loc[te["image_id"], var].astype(str).to_numpy()).map(freq).fillna(float((tr["dx"] == k).mean()))
                auc = roc_auc_score(yk, s.to_numpy())
            row[k] = float(auc)
        row["macro"] = float(np.mean(list(row.values())))
        rows[var] = row
    t = pd.DataFrame(rows).T
    thr = float(cfg["audit"]["metadata_auc_max"])
    checks = [check(f"AUC univariado de {v}", f"AUC macro < {thr}", round(float(t.loc[v, "macro"]), 4),
                    float(t.loc[v, "macro"]) < thr,
                    "el modelo no usa metadatos; un fallo señala un posible atajo si la imagen codifica el metadato")
              for v in t.index]
    return t, checks


# ---------------------------------------------------------------------------
# 5. Validación adversarial
# ---------------------------------------------------------------------------


@tolerant()
def adversarial_check(cfg: dict) -> tuple[dict, dict]:
    """AUC de una regresión logística que distingue train de test con los embeddings (CV estratificada)."""
    from src.baselines import load_embeddings
    from src.eda import adversarial_validation

    te, tr = _test_train(cfg)
    X = load_embeddings(cfg, list(tr["image_id"]) + list(te["image_id"]))
    if is_missing(X):
        return X
    y = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    res = adversarial_validation(X, y, int(cfg["adv_cv_folds"]), int(cfg["seed"]), int(cfg["adv_max_iter"]))
    thr = float(cfg["audit"]["adversarial_auc_max"])
    return res, check("Validación adversarial train frente a test", f"AUC media < {thr}",
                      round(res["auc_media"], 4), res["auc_media"] < thr, f"std {res['auc_std']:.4f}")


# ---------------------------------------------------------------------------
# 6. Etiquetas permutadas
# ---------------------------------------------------------------------------


@tolerant()
def permutation_check(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Regresión logística (C de outputs/tuning/logreg/best.json o el primero de C_grid) entrenada con
    las etiquetas de train permutadas: su exactitud balanceada en val y test debe quedar cerca del
    azar 1 / K (|media - 1 / K| <= audit.permutation.tolerance)."""
    from sklearn.metrics import balanced_accuracy_score

    from src import baselines as BL

    p = cfg["audit"]["permutation"]
    fold = int(cfg["train"]["tuning"]["val_fold"])
    frames = BL.split_frames(cfg, fold)
    if is_missing(BL.load_embeddings(cfg, frames["test"]["image_id"])):  # sin embeddings de test: solo validación
        frames.pop("test")
    X = BL._features(cfg, "logreg", frames, False)
    if is_missing(X):
        return X
    y = {s: BL.labels_of(cfg, f) for s, f in frames.items()}
    best = cfg["paths"]["tuning_dir"] / "logreg" / "best.json"
    C = D.load_json(best)["hyperparameters"]["C"] if best.exists() else cfg["baselines"]["logreg"]["C_grid"][0]
    k = len(D.class_names(cfg))
    rng = np.random.default_rng(int(p["seed"]))
    rows = []
    for i in range(-1, int(p["n_permutations"])):
        yt = y["train"] if i < 0 else rng.permutation(y["train"])
        proba, _ = BL.fit_predict(cfg, "logreg", {**X}, {**y, "train": yt}, k, {"C": C}, 0)
        rows.append({"permutacion": "original" if i < 0 else i,
                     **{f"ba_{s}": balanced_accuracy_score(y[s], proba[s].argmax(1)) for s in frames if s != "train"}})
    t = pd.DataFrame(rows).set_index("permutacion")
    perm = t.drop(index="original")
    cols = [c for c in ("ba_val", "ba_test") if c in t.columns]
    dev = float((perm[cols].mean() - 1 / k).abs().max())
    return t, check("Etiquetas permutadas (regresión logística)", f"|BA permutada media - 1/{k}| <= {p['tolerance']}",
                    round(dev, 4), dev <= float(p["tolerance"]),
                    "BA sin permutar: " + ", ".join(f"{c} {t.loc['original', c]:.4f}" for c in cols))


# ---------------------------------------------------------------------------
# 7. Ajustes solo con train (datos sintéticos)
# ---------------------------------------------------------------------------


def _synthetic_splits(rng, n: int = 600, n_folds: int = 5) -> pd.DataFrame:
    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    les = np.arange(n) // 2
    dx = np.array(classes)[rng.choice(7, n // 2, p=[0.05, 0.05, 0.1, 0.03, 0.1, 0.62, 0.05])][les]  # nv mayoritaria
    split = np.where(les % 7 == 0, "test", "train")
    fold = np.where(split == "test", -1, les % n_folds)
    return pd.DataFrame({"image_id": [f"ISIC_{i:07d}" for i in range(n)], "lesion_id": [f"HAM_{i:07d}" for i in les],
                         "group_id": [f"HAM_{i:07d}" for i in les], "dx": dx, "split": split, "fold": fold})


def synthetic_fit_checks(cfg: dict) -> tuple[pd.DataFrame, list]:
    """Pruebas con datos sintéticos: al cambiar arbitrariamente los datos de validación o test, no
    cambia nada de lo que se ajusta con train (partición, pesos de clase, regresión logística,
    prior del baseline trivial) ni lo que se ajusta con validación (temperatura y umbral
    conforme); y las estadísticas de color guardadas cubren solo train."""
    from src import baselines as BL
    from src import calibration as C

    rng = np.random.default_rng(int(cfg["audit"]["synthetic_seed"]))
    classes = D.class_names(cfg)
    k = len(classes)
    rows = []

    # Partición
    sp = _synthetic_splits(rng)
    tr, va, te = (D.select_split(sp, s, 0) for s in ("train", "val", "test"))
    disjoint = not (set(tr["lesion_id"]) & set(va["lesion_id"]) or set(tr["lesion_id"]) & set(te["lesion_id"])
                    or set(va["lesion_id"]) & set(te["lesion_id"]))
    rows.append(check("Sintético: select_split separa train, val y test por lesión", "conjuntos disjuntos",
                      disjoint, disjoint))

    # Pesos de clase: solo dependen de train
    y_tr = tr["dx"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    sp2 = sp.copy()
    sp2.loc[sp2["split"] == "test", "dx"] = "vasc"
    sp2.loc[(sp2["split"] == "train") & (sp2["fold"] == 0), "dx"] = "df"
    y_tr2 = D.select_split(sp2, "train", 0)["dx"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    same = D.class_weights(y_tr, k) == D.class_weights(y_tr2, k)
    rows.append(check("Sintético: pesos de clase solo con train", "iguales al cambiar las etiquetas de val y test",
                      same, same))

    # Regresión logística y baseline trivial
    X = {s: rng.normal(size=(len(f), 8)) for s, f in (("train", tr), ("val", va), ("test", te))}
    y = {s: f["dx"].map({c: i for i, c in enumerate(classes)}).to_numpy() for s, f in (("train", tr), ("val", va), ("test", te))}
    X2 = {**X, "val": X["val"] * 1000 + 50, "test": X["test"] - 1000}
    y2 = {**y, "val": np.zeros_like(y["val"]), "test": np.full_like(y["test"], 6)}
    for name in ("logreg", "trivial"):
        p1, _ = BL.fit_predict(cfg, name, X, y, k, {"C": 1.0}, 0)
        p2, _ = BL.fit_predict(cfg, name, X2, y2, k, {"C": 1.0}, 0)
        ok = bool(np.allclose(p1["train"], p2["train"]))
        rows.append(check(f"Sintético: {name} ajustado solo con train",
                          "predicciones de train iguales al cambiar X e y de val y test", ok, ok))
    model = BL.make_logreg(cfg, 1.0).fit(X["train"], y["train"])
    ok = bool(np.allclose(model[0].mean_, X["train"].mean(axis=0)))
    rows.append(check("Sintético: StandardScaler con la media de train", "media del escalado = media de train", ok, ok))

    # Temperatura y umbral conforme: solo validación (a través de las funciones del notebook)
    with tempfile.TemporaryDirectory() as tmp:
        from src.benchmark import RunSet

        def write(run_dir: Path, test_shift: float):
            run_dir.mkdir(parents=True, exist_ok=True)
            for s, n in (("val", 200), ("test", 150)):
                r = np.random.default_rng(1 if s == "val" else 2)
                lab = r.integers(0, k, n)
                logits = r.normal(size=(n, k)) + 2 * np.eye(k)[lab] + (test_shift if s == "test" else 0) * r.normal(size=(n, k))
                p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
                df = pd.DataFrame({"image_id": [f"x{i}" for i in range(n)], "label": [classes[i] for i in lab],
                                   "pred": [classes[i] for i in p.argmax(1)]})
                for i, c in enumerate(classes):
                    df[f"p_{c}"] = p[:, i]
                df.to_csv(run_dir / f"probabilities_{s}.csv", index=False)

        temps, qs = [], []
        for shift in (0.0, 5.0):
            d = Path(tmp) / f"shift{shift}"
            write(d, shift)
            rs = RunSet(label="x", models=["m"], expected_models=["m"], expected_pairs=[(0, 0)],
                        runs={"m": {(0, 0): d}}, metrics=pd.DataFrame(), info=pd.DataFrame())
            temps.append(float(C.temperature_table(rs, cfg, ["m"], classes)["T"].iloc[0]))
            e = C.run_predictions(rs, "m", classes)[(0, 0)]
            qs.append(C.conformal_threshold(e["val"][0], e["val"][1], 0.1))
    ok_t, ok_q = bool(np.isclose(temps[0], temps[1])), bool(np.isclose(qs[0], qs[1]))
    rows.append(check("Sintético: temperatura ajustada solo con validación", "T igual al cambiar test", ok_t, ok_t))
    rows.append(check("Sintético: umbral conforme calibrado solo con validación", "umbral igual al cambiar test", ok_q, ok_q))

    # Estadísticas de color del EDA
    p = cfg["paths"]["train_color_stats_json"]
    if Path(p).exists():
        n = int(D.load_json(p).get("n_imagenes", -1))
        n_train = int((D.load_splits(cfg["paths"]["splits_csv"])["split"] == "train").sum())
        rows.append(check("Estadísticas de color calculadas solo con train", f"n_imagenes = {n_train}", n, n == n_train))
    t = pd.DataFrame(rows).set_index("verificacion")
    return t, rows


# ---------------------------------------------------------------------------
# 8. Contrato de datos
# ---------------------------------------------------------------------------


def load_contract(cfg: dict) -> dict:
    with open(cfg["paths"]["data_contract"], encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_metadata(meta: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """Una fila por regla del contrato (columna, regla, infracciones, resultado)."""
    rows = []

    def add(col, rule, bad, extra=""):
        rows.append({"columna": col, "regla": rule, "infracciones": int(bad),
                     "resultado": APROBADO if bad == 0 else FALLIDO, "detalle": extra})

    if contract.get("rows") is not None:
        add("(tabla)", f"filas = {contract['rows']}", int(len(meta) != int(contract["rows"])), f"{len(meta)} filas")
    for col, spec in contract["columns"].items():
        if col not in meta.columns:
            add(col, "columna presente", 1)
            continue
        s = meta[col]
        if not spec.get("nullable", True):
            add(col, "sin faltantes", int(s.isna().sum()))
        if spec.get("unique"):
            add(col, "valores únicos", int(s.duplicated().sum()))
        if "pattern" in spec:
            add(col, f"patrón {spec['pattern']}", int((~s.dropna().astype(str).str.match(spec["pattern"])).sum()))
        if "allowed" in spec:
            bad = ~s.dropna().astype(str).isin([str(v) for v in spec["allowed"]])
            add(col, "valores permitidos", int(bad.sum()), ", ".join(sorted(s.dropna()[bad].astype(str).unique())[:5]))
        if spec.get("dtype") == "float":
            num = pd.to_numeric(s, errors="coerce")
            add(col, "numérica", int((num.isna() & s.notna()).sum()))
            if "min" in spec or "max" in spec:
                lo, hi = spec.get("min", -np.inf), spec.get("max", np.inf)
                add(col, f"rango [{lo}, {hi}]", int(((num < lo) | (num > hi)).sum()))
    key = contract.get("unique_key")
    for col in (contract.get("consistency") or {}).get("lesion_constant", []):
        if col in meta.columns:
            add(col, "constante dentro de cada lesion_id", int((meta.groupby("lesion_id")[col].nunique() > 1).sum()))
    out = pd.DataFrame(rows)
    out.attrs["clave"] = key
    return out


def build_dictionary(cfg: dict, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """Diccionario de atributos a partir del contrato (y de los metadatos, si se pasan: faltantes,
    valores únicos y ejemplo). Tokens de faltante: missing_tokens de base.yaml."""
    contract = load_contract(cfg)
    rows = []
    tokens = cfg.get("missing_tokens", [])
    for col, spec in contract["columns"].items():
        if "allowed" in spec:
            dom = ", ".join(map(str, spec["allowed"]))
        elif "min" in spec or "max" in spec:
            dom = f"[{spec.get('min')}, {spec.get('max')}]"
        else:
            dom = f"patrón {spec.get('pattern', '')}"
        row = {"atributo": col, "tipo": spec["dtype"], "descripcion": spec["description"], "dominio": dom,
               "admite_faltantes": bool(spec.get("nullable", True)), "unico": bool(spec.get("unique", False)),
               "fuente": spec.get("source", "")}
        if meta is not None and col in meta.columns:
            s = meta[col]
            miss = s.isna() | s.astype(str).isin(tokens)
            row.update({"faltantes": int(miss.sum()), "pct_faltantes": round(100 * float(miss.mean()), 2),
                        "valores_unicos": int(s.nunique()), "ejemplo": str(s.dropna().iloc[0])})
        rows.append(row)
    return pd.DataFrame(rows)


def contract_check(cfg: dict, meta: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    v = validate_metadata(meta, load_contract(cfg))
    bad = int((v["resultado"] == FALLIDO).sum())
    return v, check("Contrato de datos de los metadatos", "todas las reglas aprobadas", f"{bad} reglas fallidas", bad == 0)


def population_counts(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Imágenes y lesiones por población (total, train completo, train y validación de cada
    pliegue, test) y por clase; criterio: train + test = total y cada pliegue suma train."""
    s = D.load_splits(cfg["paths"]["splits_csv"])
    classes = D.class_names(cfg)
    pops = {"total": s, "entrenamiento (todos los pliegues)": s[s["split"] == "train"], "test": s[s["split"] == "test"]}
    for f in sorted(s.loc[s["split"] == "train", "fold"].unique()):
        pops[f"train de la corrida (pliegue {f} fuera)"] = D.select_split(s, "train", f)
        pops[f"validación (pliegue {f})"] = D.select_split(s, "val", f)
    rows = {}
    for name, d in pops.items():
        rows[name] = {"imagenes": len(d), "lesiones": d["lesion_id"].nunique(),
                      **{c: int((d["dx"] == c).sum()) for c in classes}}
    t = pd.DataFrame(rows).T
    ok = t.loc["entrenamiento (todos los pliegues)", "imagenes"] + t.loc["test", "imagenes"] == t.loc["total", "imagenes"]
    folds = [k for k in t.index if k.startswith("validación")]
    ok &= int(t.loc[folds, "imagenes"].sum()) == int(t.loc["entrenamiento (todos los pliegues)", "imagenes"])
    return t, check("Conteo de observaciones por población", "train + test = total y los pliegues de validación suman train",
                    int(t.loc["total", "imagenes"]), bool(ok))


# ---------------------------------------------------------------------------
# Todas las verificaciones
# ---------------------------------------------------------------------------


def run_audit(cfg: dict, rs=None, compute_sha: bool = False) -> dict:
    """Ejecuta todas las verificaciones con los datos disponibles. Devuelve {"resumen": tabla de
    verificaciones, "detalle": {nombre: tabla}}. `rs` (src.benchmark.load_runs_e2) permite medir
    el desempeño con y sin vecinos de train; `compute_sha` calcula los SHA-256 si no están en cache."""
    checks, detail = [], {}

    def add(name, res):
        if is_missing(res):
            checks.append(check(name, "—", None, None, res.message))
            return
        tab, chk = res
        detail[name] = tab
        checks.append(chk) if isinstance(chk, dict) else checks.extend(chk)

    add("lesiones", lesion_overlap(cfg))
    add("sha256", sha256_collisions(cfg, image_hashes(cfg, compute_sha)))
    for kind, fn in (("phash", phash_neighbors), ("embeddings", embedding_neighbors)):
        nb = fn(cfg)
        res = neighbor_check(cfg, nb, kind)
        if is_missing(res):
            checks.append(check(f"Vecinos de test en train ({kind})", "—", None, None, res.message))
        else:
            detail[f"vecinos_{kind}"] = nb
            checks.append(res)
            if rs is not None:
                add(f"desempeno_sin_vecinos_{kind}", performance_with_without(rs, cfg, neighbor_flags(cfg, nb, kind), kind))
    meta = D.load_metadata(cfg) if Path(cfg["paths"]["metadata_csv"]).exists() else Missing("faltan los metadatos.")
    add("contrato", contract_check(cfg, meta) if not is_missing(meta) else meta)
    add("auc_metadatos", metadata_auc(cfg, meta))
    adv = adversarial_check(cfg)
    if is_missing(adv):
        checks.append(check("Validación adversarial train frente a test", "—", None, None, adv.message))
    else:
        detail["adversarial"] = pd.Series({k: v for k, v in adv[0].items() if k != "auc_por_pliegue"}, name="valor").to_frame()
        checks.append(adv[1])
    add("permutacion", permutation_check(cfg))
    add("sinteticas", synthetic_fit_checks(cfg))
    add("poblaciones", population_counts(cfg))
    return {"resumen": summary(checks), "detalle": detail}


def detail(result):
    """Tabla de detalle de un resultado de verificación: el primer elemento de (detalle, criterio),
    con los dicts de valores convertidos en tabla; los Missing y las tablas pasan igual."""
    obj = result[0] if isinstance(result, tuple) else result
    if isinstance(obj, dict) and "verificacion" in obj:
        return pd.DataFrame([obj]).set_index("verificacion")
    if isinstance(obj, dict):
        return pd.Series({k: v for k, v in obj.items() if not isinstance(v, (list, dict))}, name="valor").to_frame()
    return obj


def collect(results) -> pd.DataFrame:
    """Resumen de verificaciones a partir de resultados de las funciones de este módulo: tuplas
    (detalle, criterio o lista de criterios), criterios sueltos o Missing (no disponible)."""
    checks = []
    for r in results:
        if isinstance(r, tuple):
            r = r[1]
        checks.append(r)
    return summary(checks)
