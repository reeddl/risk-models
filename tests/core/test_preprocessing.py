import numpy as np
import pandas as pd
import pytest

from riskmodels.core.preprocessing import (
    clean_descriptor,
    combine_descriptors,
    fill_missing,
    orthogonalize,
    standardize,
    winsorize,
)

rng = np.random.default_rng(0)


def _sample(n=500):
    idx = pd.Index([f"A{i}" for i in range(n)])
    x = pd.Series(rng.standard_normal(n), index=idx)
    cap = pd.Series(np.exp(rng.normal(22, 1.5, n)), index=idx)
    return x, cap


def test_standardize_cap_weighted_mean_and_equal_weighted_std():
    x, cap = _sample()
    z = standardize(x * 3 + 7, cap)
    assert np.average(z, weights=cap) == pytest.approx(0.0, abs=1e-12)
    assert z.std(ddof=0) == pytest.approx(1.0, abs=1e-12)


def test_standardize_keeps_nan():
    x, cap = _sample()
    x.iloc[:10] = np.nan
    z = standardize(x, cap)
    assert z.iloc[:10].isna().all() and z.iloc[10:].notna().all()


def test_winsorize_bounds_and_preserves_order():
    x, cap = _sample()
    x.iloc[0] = 50.0
    x.iloc[1] = -40.0
    w = winsorize(x, cap, n_std=3.0)
    mu = np.average(w, weights=cap)
    sd = w.std(ddof=0)
    assert (w - mu).abs().max() <= 3.0 * sd + 1e-9
    assert w.iloc[0] == w.max() and w.iloc[1] == w.min()
    assert w.rank().corr(x.rank()) > 0.999


def test_clean_descriptor_has_no_extreme_tail():
    x, cap = _sample()
    x.iloc[:5] = 1e6
    z = clean_descriptor(x, cap)
    assert z.abs().max() < 4.0


def test_combine_descriptors_renormalizes_over_available():
    z = pd.DataFrame({"a": [1.0, np.nan, np.nan], "b": [3.0, 2.0, np.nan]})
    out = combine_descriptors(z, {"a": 0.25, "b": 0.75})
    assert out.iloc[0] == pytest.approx(0.25 * 1 + 0.75 * 3)
    assert out.iloc[1] == pytest.approx(2.0)
    assert np.isnan(out.iloc[2])


def test_fill_missing_uses_industry_cap_weighted_mean_then_zero():
    z = pd.Series([1.0, 3.0, np.nan, np.nan, 5.0, np.nan])
    groups = pd.Series(["a", "a", "a", "b", "c", None])
    cap = pd.Series([1.0, 3.0, 2.0, 1.0, 1.0, 1.0])
    out = fill_missing(z, groups, cap)
    assert out.iloc[2] == pytest.approx((1 * 1 + 3 * 3) / 4)  # group a
    assert out.iloc[3] == 0.0                                 # group b has no data
    assert out.iloc[5] == 0.0                                 # no group
    assert out.iloc[4] == 5.0
    assert z.isna().sum() == 3                                # input untouched


def test_orthogonalize_removes_weighted_correlation():
    x, cap = _sample()
    beta = pd.Series(rng.standard_normal(len(x)), index=x.index)
    y = 0.8 * beta + 0.3 * x
    w = np.sqrt(cap)
    r = orthogonalize(y, beta.to_frame("beta"), w)
    assert abs(np.average(r * beta, weights=w)) < 1e-10
    assert abs(np.average(r, weights=w)) < 1e-10
