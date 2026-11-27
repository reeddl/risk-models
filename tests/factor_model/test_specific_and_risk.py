import numpy as np
import pandas as pd
import pytest

from riskmodels.factor_model.risk import RiskModel
from riskmodels.factor_model.specific_risk import bayesian_shrinkage, size_buckets


def test_size_buckets_are_deciles():
    cap = pd.Series(np.arange(1, 101, dtype=float))
    b = size_buckets(cap, 10)
    assert b.value_counts().eq(10).all()
    assert b.iloc[0] == 0 and b.iloc[-1] == 9


def test_bayesian_shrinkage_pulls_outliers_toward_bucket_mean():
    rng = np.random.default_rng(0)
    n = 500
    cap = pd.Series(np.exp(rng.normal(22, 1.5, n)))
    sigma = pd.Series(0.3 + 0.05 * rng.standard_normal(n)).abs()
    sigma.iloc[0] = 2.0
    n_obs = pd.Series(300, index=sigma.index)
    n_obs.iloc[1] = 5
    out = bayesian_shrinkage(sigma, cap, n_obs, min_obs=63)
    b = size_buckets(cap, 10)
    good = (b == b.iloc[0]) & (n_obs >= 63)
    mean0 = np.average(sigma[good], weights=cap[good])
    assert mean0 < out.iloc[0] < sigma.iloc[0]
    # thin history takes the bucket mean
    good1 = (b == b.iloc[1]) & (n_obs >= 63)
    assert out.iloc[1] == pytest.approx(np.average(sigma[good1], weights=cap[good1]))
    # more extreme estimates are shrunk more
    dev_before = (sigma - mean0).abs()[good]
    dev_after = (out - mean0).abs()[good]
    assert (dev_after <= dev_before + 1e-12).all()


def _toy_model():
    rng = np.random.default_rng(1)
    assets = [f"A{i}" for i in range(30)]
    factors = ["market", "industry:X", "size", "value"]
    X = pd.DataFrame(rng.standard_normal((30, 4)), index=assets, columns=factors)
    X["market"] = 1.0
    X["industry:X"] = (rng.random(30) < 0.5).astype(float)
    L = rng.standard_normal((4, 4)) * 0.1
    F = pd.DataFrame(L @ L.T + np.eye(4) * 0.01, index=factors, columns=factors)
    d = pd.Series(rng.uniform(0.01, 0.09, 30), index=assets)
    return RiskModel(pd.Timestamp("2024-01-02"), X, F, d, d)


def test_portfolio_risk_matches_full_covariance_and_contributions_add_up():
    rm = _toy_model()
    rng = np.random.default_rng(2)
    w = pd.Series(rng.standard_normal(30) / 30, index=rm.exposures.index)   # long/short
    rep = rm.portfolio_risk(w)
    V = rm.asset_covariance()
    assert rep.total_risk == pytest.approx(np.sqrt(w @ V @ w))
    assert rep.total_risk ** 2 == pytest.approx(rep.factor_risk ** 2 + rep.specific_risk ** 2)
    assert rep.by_position["contribution"].sum() == pytest.approx(rep.total_risk)
    assert rep.by_position["pct_of_variance"].sum() == pytest.approx(1.0)
    assert rep.by_factor["contribution"].sum() == pytest.approx(rep.factor_risk ** 2 / rep.total_risk)
    assert rep.by_group["contribution"].sum() == pytest.approx(rep.total_risk)
    assert rep.by_position["factor_contribution"].sum() == pytest.approx(
        rep.by_factor["contribution"].sum())


def test_marginal_contribution_is_the_gradient():
    rm = _toy_model()
    w = pd.Series(1 / 30, index=rm.exposures.index)
    rep = rm.portfolio_risk(w)
    eps = 1e-6
    i = w.index[3]
    bumped = w.copy()
    bumped[i] += eps
    numeric = (rm.portfolio_risk(bumped).total_risk - rep.total_risk) / eps
    assert rep.by_position.loc[i, "marginal"] == pytest.approx(numeric, rel=1e-4)


def test_active_risk_against_benchmark():
    rm = _toy_model()
    idx = rm.exposures.index
    port = pd.Series(1 / 15, index=idx[:15])
    bench = pd.Series(1 / 30, index=idx)
    rep = rm.portfolio_risk(port, benchmark=bench)
    active = port.reindex(idx, fill_value=0) - bench
    V = rm.asset_covariance()
    assert rep.total_risk == pytest.approx(np.sqrt(active @ V @ active))
    assert rep.by_factor.loc["market", "exposure"] == pytest.approx(0.0, abs=1e-12)


def test_unknown_asset_raises():
    rm = _toy_model()
    with pytest.raises(KeyError):
        rm.portfolio_risk(pd.Series({"NOPE": 1.0}))
