"""Prueba de humo del dashboard: visita cada página, llama a sus callbacks y lanza una predicción por
modelo disponible.

    cd dashboard && python smoke_test.py                  # en proceso, con el cliente de pruebas de Flask
    python smoke_test.py --url https://<app>.onrender.com # contra un despliegue (mismas peticiones por HTTP)

Por cada página registrada pide la ruta (200) y renderiza su layout; llama por HTTP a los
callbacks (/_dash-update-component, como el navegador): filtros de Datos, orden de la tabla,
dispersión, matriz de confusión y calibración de Modelos, y, por cada modelo con artefactos, la
predicción de la Demo con un ejemplo (y con una carga y la foto no dermatoscópica si se sirve en
vivo). Comprueba también que ninguna predicción escribe archivos en disco. Sale con código 1 si
algo falla. Usa DASHBOARD_DATA_DIR para apuntar a otros datos.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


class HttpClient:
    """Interfaz mínima de flask.testing.FlaskClient sobre urllib (para --url)."""

    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def _req(self, method, path, body=None):
        import urllib.error
        import urllib.request

        req = urllib.request.Request(self.base + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return _Resp(r.status, r.read())
        except urllib.error.HTTPError as e:
            return _Resp(e.code, e.read())

    def get(self, path):
        return self._req("GET", path)

    def post(self, path, json=None):
        return self._req("POST", path, json)


class _Resp:
    def __init__(self, status, data):
        self.status_code, self.data = status, data

    def get_json(self):
        return json.loads(self.data)


# ---------------------------------------------------------------------------
# Llamadas a callbacks con el mismo formato que el navegador
# ---------------------------------------------------------------------------


def _key(i) -> str:
    return json.dumps(i, sort_keys=True, separators=(",", ":")) if isinstance(i, dict) else i


class Callbacks:
    """Construye peticiones /_dash-update-component a partir de /_dash-dependencies.

    `values` = {(id, prop): valor}, con id = "texto" o el dict de un patrón concreto; para los
    patrones ALL se pasa {(tipo, prop): [(id_concreto, valor), ...]}."""

    def __init__(self, client):
        self.client = client
        r = client.get("/_dash-dependencies")
        assert r.status_code == 200, f"/_dash-dependencies: HTTP {r.status_code}"
        self.deps = r.get_json()

    def find(self, needle: str) -> dict:
        for d in self.deps:
            if d.get("clientside_function") is None and needle in d["output"]:
                return d
        raise KeyError(f"no hay callback de servidor con salida {needle}")

    @staticmethod
    def _parse(idstr: str):
        if idstr.startswith("{"):
            obj = json.loads(idstr)
            return obj, any(v == ["ALL"] for v in obj.values())
        return idstr, False

    def _expand(self, spec: dict, values: dict, with_value: bool):
        obj, wild = self._parse(spec["id"])
        prop = spec["property"]
        if wild:
            items = values.get((obj["type"], prop))
            if items is None:
                items = [(i, None) for i, _ in values.get((obj["type"], "__ids__"), [])]
            return [{"id": i, "property": prop, **({"value": v} if with_value else {})} for i, v in items]
        return {"id": obj, "property": prop, **({"value": values.get((obj, prop))} if with_value else {})}

    def call(self, needle: str, values: dict, changed: str, ids: dict | None = None):
        d = self.find(needle)
        out = d["output"]
        multi = out.startswith("..")
        parts = out.strip(".").split("...") if multi else [out]
        id_values = {(t, "__ids__"): [(i, None) for i in lst] for t, lst in (ids or {}).items()}
        outputs = []
        for p in parts:
            idstr, prop = p.rsplit(".", 1)
            outputs.append(self._expand({"id": idstr, "property": prop}, id_values, False))
        payload = {
            "output": out, "outputs": outputs if multi else outputs[0],
            "inputs": [self._expand(i, values, True) for i in d["inputs"]],
            "state": [self._expand(s, values, True) for s in d.get("state", [])],
            "changedPropIds": [changed],
        }
        r = self.client.post("/_dash-update-component", json=payload)
        if r.status_code == 204:
            return {}
        if r.status_code != 200:
            raise RuntimeError(f"{needle}: HTTP {r.status_code} {r.data[:300]!r}")
        return r.get_json()["response"]


def _text(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def page_layout(cb: Callbacks, path: str):
    d = next(x for x in cb.deps if "_pages_content.children" in x["output"])
    vals = {}
    for i in d["inputs"] + d.get("state", []):
        vals[(i["id"], i["property"])] = path if i["property"] == "pathname" else ""
    payload = {"output": d["output"], "outputs": [cb._expand({"id": s.rsplit(".", 1)[0], "property": s.rsplit(".", 1)[1]}, {}, False)
                                                    for s in d["output"].strip(".").split("...")] if d["output"].startswith("..")
               else cb._expand({"id": d["output"].rsplit(".", 1)[0], "property": d["output"].rsplit(".", 1)[1]}, {}, False),
               "inputs": [cb._expand(i, vals, True) for i in d["inputs"]],
               "state": [cb._expand(s, vals, True) for s in d.get("state", [])],
               "changedPropIds": ["_pages_location.pathname"]}
    r = cb.client.post("/_dash-update-component", json=payload)
    assert r.status_code == 200, f"layout {path}: HTTP {r.status_code}"
    return r.get_json()["response"]


def ids_of(layout, kind: str) -> list[dict]:
    return [n["props"]["id"] for n in _walk(layout) if isinstance(n.get("props"), dict)
            and isinstance(n["props"].get("id"), dict) and n["props"]["id"].get("type") == kind]


def pills(layout) -> list[dict]:
    out = []
    for n in _walk(layout):
        p = n.get("props") if isinstance(n, dict) else None
        if isinstance(p, dict) and isinstance(p.get("id"), dict) and p["id"].get("type") == "model-pill":
            out.append({"model": p["id"]["id"], "available": not p.get("disabled"), "live": "ejemplos" not in _text(p.get("children"))})
    return out


# ---------------------------------------------------------------------------
# Pruebas
# ---------------------------------------------------------------------------


def run(client, local: bool) -> list[str]:
    errors: list[str] = []

    def check(name, fn):
        t0 = time.perf_counter()
        try:
            fn()
            print(f"  ok   {name} ({1000 * (time.perf_counter() - t0):.0f} ms)")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
            print(f"  FAIL {name}: {e}")

    print("Páginas")
    for path in ("/", "/demo", "/modelos", "/datos", "/_dash-layout", "/_dash-dependencies", "/assets/cromia.css",
                 "/assets/clientside.js"):
        def get(path=path):
            r = client.get(path)
            assert r.status_code == 200, f"HTTP {r.status_code}"
        check(f"GET {path}", get)
    if local:
        import dash
        import plotly

        for page in dash.page_registry.values():
            def render(page=page):
                lay = page["layout"]() if callable(page["layout"]) else page["layout"]
                json.dumps(lay.to_plotly_json(), cls=plotly.utils.PlotlyJSONEncoder)
            check(f"layout {page['path']}", render)

    cb = Callbacks(client)
    layouts = {}
    for path in ("/", "/demo", "/modelos", "/datos"):
        def lay(path=path):
            layouts[path] = page_layout(cb, path)
            txt = _text(layouts[path])
            if path in ("/", "/demo"):
                assert "No es un dispositivo médico" in txt, "falta el aviso legal"
        check(f"layout por HTTP {path}", lay)

    print("Datos")
    cls_ids = ids_of(layouts.get("/datos"), "cls-chip")
    src_ids = ids_of(layouts.get("/datos"), "src-chip")
    for name, ctype, key in (("clase mel", "cls-chip", "mel"), ("fuente vidir_molemax", "src-chip", "vidir_molemax"),
                             ("todas", "cls-chip", "all")):
        def filt(ctype=ctype, key=key):
            vals = {("cls-chip", "n_clicks"): [(i, 1 if (ctype == "cls-chip" and i["id"] == key) else 0) for i in cls_ids],
                    ("src-chip", "n_clicks"): [(i, 1 if (ctype == "src-chip" and i["id"] == key) else 0) for i in src_ids],
                    ("data-filters", "data"): {"classes": [], "sources": []},
                    ("cls-chip", "id"): [(i, i) for i in cls_ids], ("src-chip", "id"): [(i, i) for i in src_ids]}
            out = cb.call("data-filters.data", vals, _key({"id": key, "type": ctype}) + ".n_clicks",
                          ids={"cls-chip": cls_ids, "src-chip": src_ids})
            assert "data-body" in out and "data-count" in out, _text(out)[:300]
            if local and key == "all":
                from core import data
                assert out["data-count"]["children"] == f"{len(data.images()):,}".replace(",", " "), out["data-count"]
        check(f"filtro {name}", filt)

    print("Modelos")
    sort_ids = ids_of(layouts.get("/modelos"), "lb-sort")
    def sort():
        out = cb.call("lb-body.children", {("lb-sort", "n_clicks"): [(i, 1 if i["id"] == "gflops" else 0) for i in sort_ids],
                                           ("lb-sort", "id"): [(i, i) for i in sort_ids]},
                      _key({"id": "gflops", "type": "lb-sort"}) + ".n_clicks", ids={"lb-sort": sort_ids})
        assert "lb-body" in out
    check("orden de la tabla por GFLOPs", sort)
    for theme_name in ("light", "dark"):
        def scatter(theme_name=theme_name):
            out = cb.call("scatter-graph.figure", {("sc-x-store", "data"): "latency_ms", ("theme", "data"): theme_name},
                          "theme.data")
            fig = out["scatter-graph"]["figure"]
            assert "layout" in fig and fig["layout"].get("template"), "figura sin plantilla"
        check(f"dispersión ({theme_name})", scatter)
        def calib(theme_name=theme_name):
            model = next((n["props"]["data"] for n in _walk(layouts.get("/modelos")) if isinstance(n.get("props"), dict)
                          and n["props"].get("id") == "cal-model"), None)
            out = cb.call("calibration-graph.figure", {("cal-model", "data"): model, ("theme", "data"): theme_name}, "theme.data")
            assert out["calibration-graph"]["figure"]["data"], "calibración vacía"
        check(f"calibración ({theme_name})", calib)
    cell_ids = ids_of(layouts.get("/modelos"), "cm-cell")
    mode_ids = ids_of(layouts.get("/modelos"), "cm-mode")
    state = next((n["props"]["data"] for n in _walk(layouts.get("/modelos")) if isinstance(n.get("props"), dict)
                  and n["props"].get("id") == "cm-state"), None)
    if state and state.get("model"):
        def cm():
            vals = {("cm-cell", "n_clicks"): [(i, 1 if (i["i"], i["j"]) == (0, 1) else 0) for i in cell_ids],
                    ("cm-mode", "n_clicks"): [(i, 0) for i in mode_ids], ("cm-state", "data"): state}
            out = cb.call("cm-detail.children", vals, _key({"i": 0, "j": 1, "type": "cm-cell"}) + ".n_clicks",
                          ids={"cm-cell": cell_ids, "cm-mode": mode_ids})
            assert "imágenes" in _text(out.get("cm-detail")), _text(out)[:300]
        check(f"matriz de confusión ({state['model']})", cm)

    print("Demo")
    models = [m for m in pills(layouts.get("/demo")) if m["available"]]
    if not models:
        print("  (sin modelos exportados: estado pendiente)")
    samples = ids_of(layouts.get("/demo"), "sample")
    examples = [s["id"] for s in samples if s["id"].startswith("ISIC_")]

    def predict(model, source, contents=None):
        return _text(cb.call("demo-output.children", {("demo-source", "data"): source, ("demo-model", "data"): model,
                                                      ("demo-upload", "contents"): contents, ("demo-opacity-store", "data"): 65},
                             "demo-source.data"))

    before = _snapshot(local)
    for m in models:
        def predict_example(m=m):
            assert examples, "no hay ejemplos en la galería"
            txt = predict(m["model"], {"kind": "sample", "id": examples[0], "t": 1})
            assert "alert-error" not in txt, txt[:400]
            assert "Probabilidad por clase" in txt, txt[:300]
        check(f"predicción {m['model']} (ejemplo)", predict_example)
        if m["live"] and examples:
            def predict_upload(m=m):
                raw = _example_bytes(client, examples[0])
                txt = predict(m["model"], {"kind": "upload", "t": 2}, "data:image/jpeg;base64," + base64.b64encode(raw).decode())
                assert "Probabilidad por clase" in txt, txt[:300]
            check(f"predicción {m['model']} (carga)", predict_upload)
            if any(s["id"] == "foto_normal" for s in samples):
                def predict_photo(m=m):
                    txt = predict(m["model"], {"kind": "sample", "id": "foto_normal", "t": 3})
                    assert "Probabilidad por clase" in txt, txt[:300]
                check(f"predicción {m['model']} (foto no dermatoscópica)", predict_photo)

        def bad_upload(m=m):
            txt = predict(m["model"], {"kind": "upload", "t": 4},
                          "data:image/png;base64," + base64.b64encode(b"esto no es una imagen").decode())
            assert "alert-error" in txt, "una carga inválida debe mostrar un aviso"
        check(f"carga inválida {m['model']}", bad_upload)
    if local and models:
        def no_disk():
            after = _snapshot(local)
            new = sorted(after - before)
            assert not new, f"la inferencia escribió archivos: {new[:5]}"
        check("ninguna imagen se escribe en disco", no_disk)
    return errors


def _snapshot(local: bool) -> set[str]:
    """Archivos de la carpeta del dashboard y de sus datos, e imágenes del directorio temporal."""
    if not local:
        return set()
    from core.config import APP_DIR, DATA_DIR

    roots = {APP_DIR, DATA_DIR, Path(tempfile.gettempdir())}
    out = set()
    for root in roots:
        for dirpath, dirnames, files in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            out.update(os.path.join(dirpath, f) for f in files
                       if root != Path(tempfile.gettempdir()) or f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    return out


def _example_bytes(client, image_id: str) -> bytes:
    r = client.get(f"/media/examples/{image_id}.jpg")
    assert r.status_code == 200
    return r.data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prueba de humo del dashboard.")
    parser.add_argument("--url", default=None, help="URL de un despliegue; sin ella, se usa la app en proceso")
    args = parser.parse_args(argv)
    if args.url:
        client, local = HttpClient(args.url), False
    else:
        from app import server

        client, local = server.test_client(), True
    errors = run(client, local)
    print(f"\n{'OK' if not errors else f'{len(errors)} fallos'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
