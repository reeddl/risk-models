import math

import numpy as np
import pandas as pd
import pytest

from riskmodels.factor_model.risk import RiskModel
from riskmodels.position_sizing import PositionSizer, cap_weighted_benchmark
from riskmodels.var import (
    HistoricalScenarios,
    VaRConfig,
    VaRLimits,
    VaRModel,
    backtest,
    backtest_report,
    binom_cdf,
    check_var_limits,
    chi2_sf,
    christoffersen_test,
    kupiec_test,
    norm_pdf,
    norm_ppf,
    parametric_var,
    rolling_forecasts,
    scenario_weighting,
    split_by_tail,
    tail_estimate,
    traffic_light,
)

Z99 = 2.3263478740408408


# ---------------------------------------------------------------- helpers

def toy_risk_model(n=12, seed=3) -> RiskModel:
    rng = np.random.default_rng(seed)
    assets = pd.Index([f"S{i:02d}" for i in range(n)], name="asset")
    ind = np.where(np.arange(n) % 2 == 0, "TECH", "BANK")
    X = pd.DataFrame({
        "market": 1.0,
        "industry:TECH": (ind == "TECH").astype(float),
        "industry:BANK": (ind == "BANK").astype(float),
        "beta": rng.normal(0, 1, n),
        "size": rng.normal(0, 1, n),
    }, index=assets)
    vols = np.array([0.16, 0.08, 0.07, 0.05, 0.04])
    corr = np.eye(5)
    corr[0, 3] = corr[3, 0] = 0.3
    F = pd.DataFrame(np.outer(vols, vols) * corr, index=X.columns, columns=X.columns)
    spec = pd.Series(rng.uniform(0.15, 0.40, n) ** 2, index=assets)
    return RiskModel(pd.Timestamp("2026-11-27"), X, F, spec, spec)


def toy_weights(rm, seed=5):
    rng = np.random.default_rng(seed)
    n = len(rm.exposures)
    return pd.Series(np.where(np.arange(n) < n // 2, 1, -1) * rng.uniform(0.02, 0.08, n),
                     index=rm.exposures.index)


@pytest.fixture(scope="module")
def book(daily_fit):
    rm = daily_fit.risk_model()
    rng = np.random.default_rng(11)
    names = rm.exposures.index[:40]
    w = pd.Series(rng.choice([-1, 1], len(names)) * rng.uniform(0.01, 0.04, len(names)),
                  index=names)
    return rm, w


@pytest.fixture(scope="module")
def report(daily_fit, synthetic, book):
    data, _ = synthetic
    rm, w = book
    return VaRModel(VaRConfig()).run(w, daily_fit, data, nav=100e6, risk_model=rm)


# ---------------------------------------------------------------- distributions

def test_norm_ppf_matches_reference_values():
    assert norm_ppf(0.99) == pytest.approx(Z99, abs=1e-12)
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-14)
    assert norm_ppf(0.001) == pytest.approx(-3.090232306167813, abs=1e-12)
    assert norm_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-12)
    with pytest.raises(ValueError):
        norm_ppf(1.0)


def test_chi2_sf_at_known_critical_values():
    assert chi2_sf(3.841458820694124, 1) == pytest.approx(0.05, abs=1e-12)
    assert chi2_sf(5.991464547107979, 2) == pytest.approx(0.05, abs=1e-12)
    assert chi2_sf(7.814727903251178, 3) == pytest.approx(0.05, abs=1e-10)
    assert chi2_sf(11.070497693516351, 5) == pytest.approx(0.05, abs=1e-10)
    assert chi2_sf(0.5, 4) == pytest.approx(0.9735009788392561, abs=1e-10)
    assert chi2_sf(0.0, 1) == 1.0


def test_binomial_and_traffic_light_zones():
    assert binom_cdf(4, 250, 0.01) == pytest.approx(0.8922, abs=1e-4)
    zones = [traffic_light(x, 250, 0.99) for x in range(12)]
    assert zones[:5] == ["green"] * 5
    assert zones[5:10] == ["yellow"] * 5
    assert zones[10:] == ["red"] * 2


# ---------------------------------------------------------------- tail estimator

def test_tail_estimate_equal_weights_by_hand():
    pnl = pd.Series(-np.arange(1, 101) / 100.0)          # losses 0.01 .. 1.00
    t = tail_estimate(pnl, alpha=0.05)
    assert t.var == pytest.approx(0.96)
    assert t.cvar == pytest.approx(np.mean([1.0, 0.99, 0.98, 0.97, 0.96]))
    assert t.cvar_weights.sum() == pytest.approx(1.0)
    assert t.effective_scenarios == pytest.approx(100.0)


def test_tail_estimate_weighted_with_fractional_boundary():
    pnl = pd.Series([-5.0, -4.0, -3.0, 0.0])
    p = pd.Series([0.02, 0.02, 0.06, 0.90])
    t = tail_estimate(pnl, alpha=0.05, probabilities=p)
    assert t.var == pytest.approx(3.0)               # cum 0.02, 0.04, 0.10 -> third scenario
    assert t.cvar == pytest.approx((0.02 * 5 + 0.02 * 4 + 0.01 * 3) / 0.05)


def test_split_by_tail_adds_up_exactly():
    rng = np.random.default_rng(0)
    comp = pd.DataFrame(rng.normal(0, 0.01, (500, 4)), columns=list("abcd"))
    t = tail_estimate(comp.sum(axis=1), 0.01, band=0.005)
    parts = split_by_tail(comp, t)
    assert parts["var"].sum() == pytest.approx(t.var)
    assert parts["cvar"].sum() == pytest.approx(t.cvar)


# ---------------------------------------------------------------- parametric

def test_parametric_var_matches_formula_and_euler_split():
    rm = toy_risk_model()
    w = toy_weights(rm)
    c = VaRConfig()
    par = parametric_var(rm, w, c)
    V = rm.asset_covariance().loc[w.index, w.index].to_numpy()
    sigma = np.sqrt(w.to_numpy() @ V @ w.to_numpy() / 252)
    assert par.sigma == pytest.approx(sigma)
    assert par.var == pytest.approx(Z99 * sigma)
    assert par.cvar == pytest.approx(sigma * norm_pdf(Z99) / 0.01)
    bp = par.by_position
    assert bp["var"].sum() == pytest.approx(par.var)
    assert (bp["factor_var"] + bp["specific_var"]).to_numpy() == pytest.approx(bp["var"].to_numpy())
    assert (bp["marginal_var"] * w).sum() == pytest.approx(par.var)
    assert par.by_factor["var"].sum() == pytest.approx(par.var)
    assert par.by_factor["cvar"].sum() == pytest.approx(par.cvar)


def test_marginal_var_is_the_derivative():
    rm = toy_risk_model()
    w = toy_weights(rm)
    c = VaRConfig()
    base = parametric_var(rm, w, c)
    eps = 1e-6
    name = w.index[3]
    bumped = w.copy()
    bumped[name] += eps
    fd = (parametric_var(rm, bumped, c).var - base.var) / eps
    assert base.by_position.loc[name, "marginal_var"] == pytest.approx(fd, rel=1e-4)


# ---------------------------------------------------------------- historical

def test_historical_matches_parametric_for_normal_scenarios():
    rng = np.random.default_rng(42)
    n, k = 20000, 5
    sig = np.array([0.01, 0.015, 0.02, 0.012, 0.008])
    R = pd.DataFrame(rng.normal(0, 1, (n, k)) * sig, columns=list("abcde"),
                     index=pd.bdate_range("2000-01-01", periods=n))
    w = pd.Series([0.3, -0.2, 0.25, -0.1, 0.15], index=list("abcde"))
    sc = HistoricalScenarios.from_returns(R)
    t = tail_estimate(sc.pnl(w), 0.01)
    s = math.sqrt(((w * sig) ** 2).sum())
    assert t.var == pytest.approx(Z99 * s, rel=0.04)
    assert t.cvar == pytest.approx(s * norm_pdf(Z99) / 0.01, rel=0.05)


def test_age_weights_favor_recent_days_and_volatility_rescales():
    idx = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(1)
    pnl = pd.Series(np.r_[rng.normal(0, 0.005, 200), rng.normal(0, 0.02, 100)], index=idx)
    p, s = scenario_weighting(pnl, VaRConfig(hs_weighting="age", hs_half_life=50))
    assert p.iloc[-1] > p.iloc[0]
    assert p.iloc[-51] == pytest.approx(p.iloc[-1] / 2, rel=1e-9)
    assert (s == 1).all()
    p, s = scenario_weighting(pnl, VaRConfig(hs_weighting="volatility", vol_half_life=10))
    assert p.nunique() == 1
    assert s.iloc[:150].mean() > 2.0        # calm days scaled up to today's higher vol
    # Recent volatility counts more: both weightings raise VaR above the unweighted one.
    flat = tail_estimate(pnl, 0.01).var
    for scheme in ("age", "volatility"):
        p, s = scenario_weighting(pnl, VaRConfig(hs_weighting=scheme, hs_half_life=50))
        assert tail_estimate(pnl * s, 0.01, p).var > flat


# ---------------------------------------------------------------- full report

def test_report_components_add_up(report):
    s = report.summary
    bp, bs, bf, side = report.by_position, report.by_sector, report.by_factor, report.by_side
    for col, total in (("parametric_var", s["parametric_var_1d"]),
                       ("historical_var", s["historical_var_1d"]),
                       ("historical_cvar", s["historical_cvar_1d"])):
        assert bp[col].sum() == pytest.approx(total)
        assert bs[col].sum() == pytest.approx(total)
        assert bf[col].sum() == pytest.approx(total)
    assert side.loc[["long", "short"], "component_parametric_var"].sum() == pytest.approx(
        s["parametric_var_1d"])
    assert side.loc[["long", "short"], "component_historical_cvar"].sum() == pytest.approx(
        s["historical_cvar_1d"])
    # Subadditivity of the normal VaR: long plus short alone is at least the book.
    assert side.loc["total", "diversification_parametric"] >= -1e-12
    assert s["historical_cvar_1d"] >= s["historical_var_1d"]


def test_report_horizons_and_amounts(report):
    s = report.summary
    assert s["parametric_var_10d"] == pytest.approx(s["parametric_var_1d"] * math.sqrt(10))
    assert s["historical_cvar_10d"] == pytest.approx(s["historical_cvar_1d"] * math.sqrt(10))
    assert s["parametric_var_1d_amount"] == pytest.approx(s["parametric_var_1d"] * 100e6)
    assert report.horizons.loc[("historical", "var"), "10d"] == pytest.approx(
        s["historical_var_10d"])
    amt = report.to_amount(report.by_sector)
    assert amt["parametric_var"].sum() == pytest.approx(s["parametric_var_1d_amount"])
    assert s["var_utilization"] == pytest.approx(s["parametric_var_1d"] / 0.02)


def test_methods_agree_on_synthetic_normal_data(report):
    # The synthetic market is close to normal, so the cross-check should be quiet.
    assert abs(report.summary["method_gap"]) < 0.25
    assert "method_gap" not in set(report.breaches["kind"])


def test_scenarios_use_observed_returns_and_proxy_gaps(report, synthetic):
    data, _ = synthetic
    sc = report.scenarios
    obs = ~sc.proxied
    raw = data.returns.reindex(index=sc.dates, columns=sc.asset_returns.columns)
    if data.risk_free is not None:
        raw = raw.sub(data.risk_free.reindex(sc.dates), axis=0)
    m = obs.to_numpy()
    assert m.sum() > 0.9 * m.size
    assert np.allclose(sc.asset_returns.to_numpy()[m], raw.to_numpy()[m])
    assert raw.to_numpy()[~m].size == 0 or np.isnan(raw.to_numpy()[~m]).all()
    w = report.weights
    fp = sc.factor_pnl(w)
    assert np.allclose(fp.sum(axis=1), sc.pnl(w))


def test_accepts_position_sizing_result(daily_fit, synthetic):
    data, _ = synthetic
    rm = daily_fit.risk_model()
    names = rm.exposures.index[:30]
    signals = pd.DataFrame({
        "side": ["long" if i % 2 == 0 else "short" for i in range(len(names))],
        "conviction": [(i % 5) + 1 for i in range(len(names))],
    }, index=names)
    sized = PositionSizer().size(signals, rm, nav=250e6,
                                 benchmark=cap_weighted_benchmark(daily_fit))
    rep = VaRModel().run(sized, daily_fit, data, risk_model=rm, nav=250e6)
    assert rep.by_position.index.equals(sized.weights[sized.weights != 0].index)
    assert set(rep.by_sector.index) <= set(sized.sector.unique())


# ---------------------------------------------------------------- limits

def _summary(par, hist=None, n=500, cvar=None):
    hist = par if hist is None else hist
    return {"parametric_var_1d": par, "historical_var_1d": hist,
            "historical_cvar_1d": cvar if cvar is not None else hist * 1.2,
            "method_gap": (hist - par) / par, "n_scenarios": n, "proxied_share": 0.0}


@pytest.mark.parametrize("var,expected", [
    (0.010, None), (0.0145, "watch"), (0.0165, "alert"), (0.020, "breach"), (0.025, "breach"),
])
def test_alert_levels(var, expected):
    rows = check_var_limits(_summary(var), VaRConfig())
    kinds = {r["kind"]: r["severity"] for r in rows}
    assert kinds.get("var_limit") == expected


def test_limit_basis_cvar_limit_and_warnings():
    c = VaRConfig(limits=VaRLimits(var_limit=0.03, limit_basis="max", cvar_limit=0.03),
                  min_history_days=250)
    rows = check_var_limits(_summary(0.01, hist=0.022, n=100, cvar=0.031), c)
    sev = {r["kind"]: r["severity"] for r in rows}
    assert sev["var_limit"] == "watch"         # max(0.01, 0.022) / 0.03 = 73%
    assert sev["cvar_limit"] == "breach"
    assert sev["method_gap"] == "warning"
    assert sev["short_history"] == "warning"


def test_config_validation():
    with pytest.raises(ValueError):
        VaRConfig(hs_weighting="bootstrap")
    with pytest.raises(ValueError):
        VaRConfig(horizon_days=10)
    with pytest.raises(ValueError):
        VaRLimits(alert_levels={"watch": 1.2})
    with pytest.raises(ValueError):
        VaRLimits(limit_basis="average")


# ---------------------------------------------------------------- backtests

def test_kupiec_reference_values():
    r = kupiec_test(10, 250, 0.99)
    assert r["lr_uc"] == pytest.approx(12.955491062356018)
    assert r["p_uc"] < 0.001
    assert kupiec_test(0, 250, 0.99)["lr_uc"] == pytest.approx(-2 * 250 * math.log(0.99))
    assert kupiec_test(3, 300, 0.99)["lr_uc"] == pytest.approx(0.0, abs=1e-12)


def test_christoffersen_detects_clustering():
    n = 500
    spread = pd.Series(False, index=range(n))
    spread[[50, 150, 250, 350, 450]] = True
    clustered = pd.Series(False, index=range(n))
    clustered[[200, 201, 202, 203, 204]] = True
    a = christoffersen_test(spread, 0.99)
    b = christoffersen_test(clustered, 0.99)
    assert (a["n01"], a["n11"]) == (5, 0)
    assert (b["n01"], b["n11"]) == (1, 4)
    assert a["p_ind"] > 0.5
    assert b["p_ind"] < 0.001
    assert b["p_cc"] < 0.001


def test_backtest_passes_a_correct_model_and_fails_an_understated_one():
    rng = np.random.default_rng(3)
    n, sigma = 5000, 0.01
    idx = pd.bdate_range("2000-01-03", periods=n)
    pnl = pd.Series(rng.normal(0, sigma, n), index=idx)
    good = pd.Series(Z99 * sigma, index=idx)
    res = backtest(good, pnl, cvar=pd.Series(sigma * norm_pdf(Z99) / 0.01, index=idx))
    assert res.passed
    assert res.summary["n_exceedances"] == pytest.approx(50, abs=20)
    assert res.summary["cvar_ratio"] == pytest.approx(1.0, abs=0.1)
    bad = backtest(good / 2, pnl)
    assert not bad.passed
    assert bad.summary["traffic_light"] == "red"


def test_rolling_forecasts_align_forecast_and_next_day_pnl(daily_fit, synthetic, book):
    data, _ = synthetic
    _, w = book
    start = daily_fit.exposure_dates[-25]
    fc = rolling_forecasts(daily_fit, data, w, VaRConfig(history_days=250), start=start)
    assert len(fc) == 24                       # the last exposure date has no next day
    assert (fc.index > fc["forecast_date"]).all()
    day = fc.index[5]
    held = w[w.index.intersection(daily_fit.risk_model(fc["forecast_date"].iloc[5]).exposures.index)]
    rf = data.risk_free.loc[day] if data.risk_free is not None else 0.0
    expected = (held * (data.returns.loc[day, held.index].fillna(0.0) - rf)).sum()
    assert fc.loc[day, "hypothetical_pnl"] == pytest.approx(expected)

    # A date x name book: the last row on or before each forecast date is used.
    book_df = pd.DataFrame([w, w * 2], index=[daily_fit.exposure_dates[0], start])
    fc2 = rolling_forecasts(daily_fit, data, book_df, VaRConfig(history_days=250),
                            start=start, end=daily_fit.exposure_dates[-20])
    assert fc2["parametric_var"].to_numpy() == pytest.approx(
        2 * fc["parametric_var"].iloc[:len(fc2)].to_numpy())

    actual = fc["hypothetical_pnl"] * 1.1
    table = backtest_report(fc, actual_pnl=actual)
    assert set(table.index) == {("parametric", "hypothetical"), ("parametric", "actual"),
                                ("historical", "hypothetical"), ("historical", "actual")}
    assert (table["n_obs"] == 24).all()
