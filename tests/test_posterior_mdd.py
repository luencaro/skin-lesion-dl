"""Diferencia mínima detectable a posteriori (capítulo 7, «Número de corridas»)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.stats as ss

from src import benchmark as B
from src import stats as S


def test_formula_matches_chapter_7():
    sd, n, ratio = 0.012, 5, 0.25
    expected = (ss.t.ppf(0.975, 4) + ss.t.ppf(0.80, 4)) * sd * np.sqrt(1 / 5 + 0.25)
    assert S.minimum_detectable_difference_nb(sd, n, 0.05, 0.80, ratio) == pytest.approx(expected)
    assert np.isnan(S.minimum_detectable_difference_nb(sd, 1, 0.05, 0.80, ratio))


def test_posterior_mdd_table_from_runs():
    a = np.array([0.70, 0.72, 0.69, 0.71, 0.73])
    b = np.array([0.68, 0.71, 0.69, 0.68, 0.70])
    rows = [{"model": m, "seed": 0, "val_fold": f, "split": "test", "metric": "balanced_accuracy", "value": v}
            for m, vals in (("m1", a), ("m2", b)) for f, v in enumerate(vals)]
    info = pd.DataFrame({"model": ["m1"] * 5 + ["m2"] * 5, "n_val": 1717, "n_train": 6867})
    rs = B.RunSet("final", ["m1", "m2"], ["m1", "m2"], [(0, f) for f in range(5)], {}, pd.DataFrame(rows), info)
    cfg = {"benchmark": {"split": "test", "primary_metric": "balanced_accuracy", "delta": 0.02, "alpha": 0.05,
                         "power": 0.80}}
    t = B.posterior_mdd_table(rs, cfg)
    d = a - b
    expected = S.minimum_detectable_difference_nb(d.std(ddof=1), 5, 0.05, 0.80, 1717 / 6867)
    row = t.loc[("m1", "m2")]
    assert row["corridas_pareadas"] == 5
    assert row["sigma_d"] == pytest.approx(d.std(ddof=1))
    assert row["dif_minima_detectable"] == pytest.approx(expected)
    assert row["supera_delta"] == (expected > 0.02)
