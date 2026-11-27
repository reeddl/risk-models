import numpy as np
import pandas as pd
import pytest

from riskmodels.factor_model.exposures import exposure_matrix, industry_factor
from riskmodels.factor_model.regression import constrained_wls, cross_sectional_regression


def _problem(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.Index([f"A{i}" for i in range(n)])
    inds = ["X", "Y", "Z"]
    ind = pd.Series(rng.choice(inds, n), index=idx)
    styles = pd.DataFrame(rng.standard_normal((n, 2)), index=idx, columns=["size", "value"])
    X = exposure_matrix(styles, ind, inds)
    cap = pd.Series(np.exp(rng.normal(22, 1.5, n)), index=idx)
    return X, cap, ind


def test_recovers_factor_returns_exactly_without_noise():
    X, cap, ind = _problem()
    ind_cap = np.array([cap[ind == k].sum() for k in ["X", "Y", "Z"]])
    g = np.array([0.01, -0.02, 0.0])
    g = g - (ind_cap @ g) / ind_cap.sum()           # satisfy the cap-weighted constraint
    true = pd.Series([0.005, *g, 0.003, -0.001], index=X.columns)
    r = X @ true
    fit = cross_sectional_regression(r, X, cap)
    assert np.allclose(fit.factor_returns, true, atol=1e-12)
    assert fit.weighted_r2 == pytest.approx(1.0)


def test_industry_constraint_makes_cap_weighted_industry_return_zero():
    """The cap-weighted market portfolio has no net industry return, so its factor return is
    the market factor plus its (near-zero) style tilt."""
    X, cap, ind = _problem(seed=1)
    rng = np.random.default_rng(9)
    r = pd.Series(rng.standard_normal(len(X)) * 0.02, index=X.index)
    f = cross_sectional_regression(r, X, cap).factor_returns
    ind_cols = [industry_factor(k) for k in ["X", "Y", "Z"]]
    w = cap / cap.sum()
    assert float((X[ind_cols].T @ w) @ f[ind_cols]) == pytest.approx(0.0, abs=1e-14)
    assert f[ind_cols].abs().max() > 1e-4      # the industries themselves still move


def test_weights_are_sqrt_cap():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((50, 2))
    y = rng.standard_normal(50)
    w = rng.uniform(1, 100, 50)
    f = constrained_wls(y, X, w)
    W = np.diag(w)
    assert np.allclose(f, np.linalg.solve(X.T @ W @ X, X.T @ W @ y))


def test_empty_industry_is_dropped_not_singular():
    X, cap, ind = _problem(seed=2)
    keep = ind != "Z"
    r = pd.Series(np.random.default_rng(1).standard_normal(keep.sum()), index=X.index[keep])
    fit = cross_sectional_regression(r, X[keep], cap[keep])
    assert np.isnan(fit.factor_returns[industry_factor("Z")])
    assert fit.factor_returns.drop(industry_factor("Z")).notna().all()
