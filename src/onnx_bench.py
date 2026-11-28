"""Latencia y memoria pico de un modelo ONNX en CPU, en un proceso aparte (sin TensorFlow).

src.export lo ejecuta como subproceso para que la memoria medida sea la de un proceso que solo
hace inferencia con ONNX Runtime, como el dashboard desplegado:

    python -m src.onnx_bench artifacts/cci/model.onnx --repeats 100 --warmup 10 --threads 2

Imprime un JSON con la mediana y el p95 de la latencia de una imagen (ms), la memoria residente
pico del proceso (MB, VmHWM en Linux), la memoria antes de cargar el modelo y el tamaño del archivo.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time


def _rss_peak_mb() -> float:
    """Memoria residente pico de este proceso. En Linux, VmHWM de /proc/self/status (se reinicia al
    ejecutar el programa; ru_maxrss, en cambio, conserva el pico del proceso padre a través de exec)."""
    status = "/proc/self/status"
    if os.path.exists(status):
        with open(status, encoding="ascii") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 2 ** 10                        # KiB -> MiB
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2 ** 20 if sys.platform == "darwin" else peak / 2 ** 10  # bytes en macOS, KiB en Linux


def bench(path: str, repeats: int = 100, warmup: int = 10, threads: int = 2, seed: int = 0) -> dict:
    import numpy as np
    import onnxruntime as ort

    before = _rss_peak_mb()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(threads)
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [1 if not isinstance(d, int) else d for d in inp.shape]
    x = np.random.default_rng(seed).uniform(1.0, 254.0, size=shape).astype(np.float32)
    for _ in range(int(warmup)):
        sess.run(None, {inp.name: x})
    times = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})
        times.append(1000 * (time.perf_counter() - t0))
    t = np.asarray(times)
    return {
        "latency_ms_median": float(np.median(t)),
        "latency_ms_p95": float(np.percentile(t, 95)),
        "latency_ms_mean": float(t.mean()),
        "repeats": int(repeats),
        "warmup": int(warmup),
        "intra_op_threads": int(threads),
        "peak_rss_mb": _rss_peak_mb(),
        "rss_before_load_mb": before,
        "file_size_mb": os.path.getsize(path) / 2 ** 20,
        "onnxruntime": ort.__version__,
        "provider": "CPUExecutionProvider",
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    print(json.dumps(bench(args.path, args.repeats, args.warmup, args.threads)))


if __name__ == "__main__":
    main()
