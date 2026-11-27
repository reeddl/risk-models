"""End-to-end checks on synthetic data with a known factor structure."""

import numpy as np
import pandas as pd
import pytest

from riskmodels.factor_model import FactorModelConfig, regression_periods


def _corr(a, b):
    a, b = a.align(b, join="inner")
    return float(np.corrcoef(a, b)[0, 1])


def test_recovers_true_factor_returns(daily_fit, synthetic):
    _, truth = synthetic
    fr = daily_fit.factor_returns
    assert _corr(fr["market"], truth.market) > 0.85
    for s in ("size", "value", "quality"):
        assert _corr(fr[s], truth.style_returns[s]) > 0.6, s
    for k in truth.industry_returns.columns:
        assert _corr(fr[f"industry:{k}"], truth.industry_returns[k]) > 0.6, k


def test_exposures_are_standardized_and_complete(daily_fit):
    date = daily_fit.exposure_dates[-1]
    X = daily_fit.exposures(date)
    cap = daily_fit.market_cap.loc[date, X.index]
    styles = daily_fit.config.style_factors
    assert X[styles].notna().all().all()
    for s in styles:
        assert np.average(X[s], weights=cap) == pytest.approx(0.0, abs=1e-6), s
        assert X[s].std(ddof=0) == pytest.approx(1.0, abs=0.05), s
    ind_cols = [c for c in X.columns if c.startswith("industry:")]
    assert (X[ind_cols].sum(axis=1) == 1).all()
    assert (X["market"] == 1).all()


def test_residual_volatility_is_orthogonal_to_beta(daily_fit):
    date = daily_fit.exposure_dates[-1]
    X = daily_fit.exposures(date)
    w = np.sqrt(daily_fit.market_cap.loc[date, X.index])
    for other in ("beta", "size"):
        c = np.average(X["residual_volatility"] * X[other], weights=w)
        assert abs(c) < 0.05, other


def test_exposures_use_only_past_data(synthetic):
    """Changing returns after a date must not change exposures on that date."""
    from riskmodels.core.data import MarketData
    from riskmodels.factor_model.descriptors import DescriptorEngine
    data, _ = synthetic
    t = 350
    future = data.returns.copy()
    future.iloc[t + 1:] = future.iloc[t + 1:] * 0.5 + 0.001
    bumped = MarketData(future, data.market_cap, data.dollar_volume,
                        data.security_master, data.industries, data.fundamentals)
    cfg = FactorModelConfig.daily()
    a = DescriptorEngine(data, cfg).raw(t)
    b = DescriptorEngine(bumped, cfg).raw(t)
    pd.testing.assert_frame_equal(a, b)


def test_delisted_names_stay_in_history(daily_fit, synthetic):
    data, _ = synthetic
    sm = data.security_master
    dead = sm[sm["end_date"].notna() & (sm["end_date"] > daily_fit.factor_returns.index[5])]
    a, end = dead.iloc[0]["asset"], dead.iloc[0]["end_date"]
    u = daily_fit.specific_returns[a]
    assert u.loc[:end].notna().sum() > 20
    assert u.loc[end + pd.Timedelta(days=1):].isna().all()
    rm = daily_fit.risk_model()
    assert a not in rm.exposures.index


def test_risk_model_is_well_formed(daily_fit, synthetic):
    _, truth = synthetic
    rm = daily_fit.risk_model()
    F = rm.factor_covariance.to_numpy()
    assert np.allclose(F, F.T)
    assert np.linalg.eigvalsh(F).min() > 0
    assert rm.specific_variance.notna().all() and (rm.specific_variance > 0).all()
    true_spec = truth.specific_vol.reindex(rm.exposures.index) * np.sqrt(252)
    assert rm.specific_risk.median() == pytest.approx(true_spec.median(), rel=0.2)
    # smaller names carry more specific risk in the synthetic data; the forecast agrees
    cap = daily_fit.market_cap.loc[rm.as_of, rm.exposures.index]
    assert np.corrcoef(np.log(cap), rm.specific_risk)[0, 1] < -0.3


def test_portfolio_report_on_fitted_model(daily_fit):
    rm = daily_fit.risk_model()
    names = rm.exposures.index
    w = pd.Series(0.0, index=names)
    w.iloc[:20] = 0.05
    w.iloc[-20:] = -0.05                     # dollar-neutral long/short book
    rep = rm.portfolio_risk(w)
    s = rep.summary()
    assert s["factor_share_of_variance"] + s["specific_share_of_variance"] == pytest.approx(1.0)
    assert rep.by_position["contribution"].sum() == pytest.approx(rep.total_risk)
    assert rep.by_factor.loc["market", "exposure"] == pytest.approx(0.0, abs=1e-12)


def test_risk_model_as_of_uses_only_history_to_that_date(daily_fit):
    mid = daily_fit.exposure_dates[len(daily_fit.exposure_dates) // 2]
    rm = daily_fit.risk_model(as_of=mid)
    assert rm.as_of == mid
    later = daily_fit.factor_returns[daily_fit.factor_returns.index > mid]
    assert len(later) > 0


def test_weekly_periods_compound_and_fit(weekly_fit, daily_fit):
    dates = pd.bdate_range("2021-01-04", periods=20)
    p = regression_periods(dates, "W", 0)
    assert all(dates[b].dayofweek == 4 or b == len(dates) - 1 for _, b in p)
    fr_w = weekly_fit.factor_returns["market"]
    fr_d = daily_fit.factor_returns["market"]
    # compounding daily market factor returns over each week tracks the weekly estimate
    weekly_from_daily = (1 + fr_d).groupby(fr_d.index.to_period("W-FRI")).prod() - 1
    fr_w = fr_w.set_axis(fr_w.index.to_period("W-FRI"))
    assert _corr(fr_w, weekly_from_daily) > 0.95
    rm = weekly_fit.risk_model()
    assert np.linalg.eigvalsh(rm.factor_covariance.to_numpy()).min() > 0
