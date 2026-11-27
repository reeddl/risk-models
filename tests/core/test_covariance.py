import numpy as np
import pandas as pd
import pytest

from riskmodels.core.covariance import (
    ewma_weights,
    factor_covariance,
    nearest_psd,
    newey_west_covariance,
    shrink_correlation,
    specific_variance,
)


def test_ewma_weights_half_life():
    w = ewma_weights(200, 20)
    assert w.sum() == pytest.approx(1.0)
    assert w[-21] / w[-1] == pytest.approx(0.5)
    assert np.all(np.diff(w) > 0)


def test_ewma_cov_recovers_true_covariance():
    rng = np.random.default_rng(1)
    true = np.array([[4.0, 1.2], [1.2, 1.0]]) * 1e-4
    F = rng.multivariate_normal([0, 0], true, size=20000)
    est = newey_west_covariance(F, half_life=np.inf, lags=0)
    assert np.allclose(est, true, rtol=0.05, atol=2e-6)


def test_newey_west_raises_variance_for_positive_autocorrelation():
    rng = np.random.default_rng(2)
    e = rng.standard_normal(5000)
    ar = np.zeros_like(e)
    for t in range(1, len(e)):
        ar[t] = 0.4 * ar[t - 1] + e[t]
    plain = newey_west_covariance(ar, np.inf, 0)[0, 0]
    adj = newey_west_covariance(ar, np.inf, 5)[0, 0]
    assert adj > 1.5 * plain


def test_pairwise_missing_does_not_bias_variance():
    rng = np.random.default_rng(3)
    F = rng.standard_normal((4000, 2)) * 0.01
    F[:3000, 1] = np.nan   # a factor that only exists for the last 1000 periods
    est = newey_west_covariance(F, np.inf, 0)
    assert est[1, 1] == pytest.approx(1e-4, rel=0.1)


def test_shrinkage_and_psd():
    corr = np.array([[1.0, 0.99, -0.99], [0.99, 1.0, 0.99], [-0.99, 0.99, 1.0]])
    assert np.linalg.eigvalsh(corr).min() < 0
    shrunk = shrink_correlation(corr, 0.2)
    assert np.allclose(np.diag(shrunk), 1.0)
    assert shrunk[0, 1] == pytest.approx(0.8 * 0.99)
    fixed = nearest_psd(corr)
    assert np.linalg.eigvalsh(fixed).min() >= -1e-12


def test_factor_covariance_uses_separate_half_lives():
    rng = np.random.default_rng(4)
    n = 2000
    F = rng.standard_normal((n, 3)) * 0.01
    F[-100:] *= 3.0          # recent volatility spike
    df = pd.DataFrame(F, columns=list("abc"))
    fast = factor_covariance(df, vol_half_life=20, corr_half_life=500)
    slow = factor_covariance(df, vol_half_life=500, corr_half_life=500)
    assert np.sqrt(fast.loc["a", "a"]) > 1.5 * np.sqrt(slow.loc["a", "a"])
    assert np.linalg.eigvalsh(fast.to_numpy()).min() > 0
    assert np.allclose(fast.to_numpy(), fast.to_numpy().T)


def test_specific_variance_counts_and_floor():
    rng = np.random.default_rng(5)
    U = pd.DataFrame(rng.standard_normal((300, 3)) * 0.02, columns=list("xyz"))
    U.iloc[:250, 2] = np.nan
    alt = np.where(np.arange(300) % 2 == 0, 1.0, -1.0) * 0.02    # strong negative autocorr
    U["w"] = alt
    var, n = specific_variance(U, half_life=np.inf, lags=5)
    assert n["z"] == 50 and n["x"] == 300
    assert var["x"] == pytest.approx(4e-4, rel=0.25)
    assert var["w"] == pytest.approx(0.5 * 4e-4)                 # floored at half lag-0
