"""Prueba de humo del dashboard: visita cada página y lanza una predicción por modelo disponible.

    cd dashboard && python smoke_test.py                  # en proceso, con el cliente de pruebas de Flask
    python smoke_test.py --url https://<app>.onrender.com # contra un despliegue (mismas peticiones por HTTP)

Por cada página registrada pide la ruta (200) y renderiza su layout; llama por HTTP a los
callbacks de las secciones del EDA (/_dash-update-component, como el navegador) y, por cada modelo
con artefactos, a la predicción con el primer ejemplo de la galería (y con una carga si se sirve en
vivo). Sale con código 1 si algo falla. Usa DASHBOARD_DATA_DIR para apuntar a otros datos.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
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


def call(client, output: str, inputs: list[tuple[str, str, object]], state=(), changed=None) -> dict:
    """Llama a un callback como lo hace el navegador. `output` = "id.prop"; inputs/state = [(id, prop, valor)]."""
    oid, oprop = output.rsplit(".", 1)
    payload = {
        "output": output, "outputs": {"id": oid, "property": oprop},
        "inputs": [{"id": i, "property": p, "value": v} for i, p, v in inputs],
        "state": [{"id": i, "property": p, "value": v} for i, p, v in state],
        "changedPropIds": changed or [f"{inputs[0][0]}.{inputs[0][1]}"],
    }
    r = client.post("/_dash-update-component", json=payload)
    if r.status_code == 204:
        return {}
    if r.status_code != 200:
        raise RuntimeError(f"{output}: HTTP {r.status_code} {r.data[:300]!r}")
    return r.get_json()["response"][oid][oprop]


def _text(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


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
    for path in ("/", "/eda", "/modelos", "/_dash-layout", "/_dash-dependencies", "/assets/style.css"):
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

    print("EDA")
    filters = {"classes": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"], "sources": ["rosendahl", "vidir_modern", "vidir_molemax", "vienna_dias"]}
    sections = {"clases": [], "calidad": [], "galeria": [("gallery-class", "value", "mel"), ("gallery-shuffle", "n_clicks", 1)],
                "color": [("color-var", "value", "saturacion")], "explorador": [("explorer-var", "value", "age"), ("explorer-by", "value", "dx")]}
    for sid, extra in sections.items():
        def section(sid=sid, extra=extra):
            out = call(client, f"{sid}-content.children", [("eda-filters", "data", filters), *extra])
            assert "No se pudo dibujar" not in _text(out), _text(out)[:300]
        check(f"sección {sid}", section)

    print("Modelos")
    models, examples = _available(client, local)
    if not models:
        print("  (sin modelos exportados: estado pendiente)")
    for m in models:
        ex = examples[0]["image_id"] if examples else None

        def predict_example(m=m, ex=ex):
            assert ex, "no hay ejemplos en la galería"
            out = call(client, "pred-output.children", [("pred-source", "data", {"kind": "example", "id": ex}),
                                                        ("pred-model", "value", m["model"]), ("pred-alpha", "value", "0.1"),
                                                        ("pred-cam", "value", True)], state=[("pred-upload", "contents", None)])
            txt = _text(out)
            assert "No se pudo" not in txt and "error-alert" not in txt, txt[:300]
            assert "Probabilidad calibrada" in txt
        check(f"predicción {m['model']} (ejemplo)", predict_example)
        if m.get("live") and examples:
            def predict_upload(m=m):
                raw = _example_bytes(client, examples[0]["file"])
                contents = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
                out = call(client, "pred-output.children", [("pred-source", "data", {"kind": "upload"}),
                                                            ("pred-model", "value", m["model"]), ("pred-alpha", "value", "0.05"),
                                                            ("pred-cam", "value", False)], state=[("pred-upload", "contents", contents)])
                assert "Probabilidad calibrada" in _text(out), _text(out)[:300]
            check(f"predicción {m['model']} (carga)", predict_upload)

        def bad_upload(m=m):
            contents = "data:image/png;base64," + base64.b64encode(b"esto no es una imagen").decode()
            out = call(client, "pred-output.children", [("pred-source", "data", {"kind": "upload"}),
                                                        ("pred-model", "value", m["model"]), ("pred-alpha", "value", "0.1"),
                                                        ("pred-cam", "value", True)], state=[("pred-upload", "contents", contents)])
            assert "error-alert" in _text(out), "una carga inválida debe mostrar un aviso"
        check(f"carga inválida {m['model']}", bad_upload)
    return errors


def _available(client, local: bool):
    if local:
        from core import data

        return data.available_models(), data.examples()
    # remoto: lee las opciones del selector de la página de modelos
    out = call(client, "_pages_content.children", [("_pages_location", "pathname", "/modelos"), ("_pages_location", "search", "")])
    txt = _text(out)
    models = []
    if '"pred-model"' in txt:
        import re

        for value in re.findall(r'"value":\s*"([^"]+)"', txt.split('"pred-model"', 1)[1].split("]", 1)[0]):
            models.append({"model": value, "live": True})
    import re

    examples = [{"image_id": i, "file": f"{i}.jpg"} for i in dict.fromkeys(re.findall(r"/media/examples/(ISIC_\d+)\.jpg", txt))]
    return models, examples


def _example_bytes(client, name: str) -> bytes:
    r = client.get(f"/media/examples/{name}")
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
