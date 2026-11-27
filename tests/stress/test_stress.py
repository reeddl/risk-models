import json
import math

import numpy as np
import pandas as pd
import pytest

from riskmodels.factor_model.risk import RiskModel
from riskmodels.liquidity import adv_from_data, market_cap_from_data
from riskmodels.stress import (
    Scenario,
    ScenarioLibrary,
    SqueezeConfig,
    StressConfig,
    StressLimits,
    StressTester,
    ad_hoc_stress_report,
    apply_scenario,
    calibrate_from_factor_returns,
    calibrate_from_reference,
    daily_liquidity_report,
    default_library,
    full_shock,
    identify_crowded,
    latest_short_interest,
    liquidity_overlay,
    map_shocks,
    periodic_stress_report,
    recalibrate_library,
    reverse_stress,
    short_squeeze,
)
from riskmodels.var.stats import norm_ppf

STYLES = {"size", "beta", "momentum", "residual_volatility", "value", "quality", "growth",
          "leverage", "liquidity", "dividend_yield"}
FF12 = {"NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq", "Telcm", "Utils", "Shops",
        "Hlth", "Money", "Other"}
SYNTH_MAP = {"Money": "IND0", "BusEq": "IND1", "Enrgy": "IND2", "Hlth": "IND3",
             "Utils": "IND4", "Shops": "IND5"}


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
def synth_book(synthetic, daily_fit):
    data, _ = synthetic
    rm = daily_fit.risk_model()
    names = rm.exposures.index[:50]
    rng = np.random.default_rng(11)
    w = pd.Series(np.where(np.arange(50) < 25, 1, -1) * rng.uniform(0.01, 0.03, 50), index=names)
    adv = adv_from_data(data, rm.as_of)
    cap = market_cap_from_data(data, rm.as_of)
    return data, rm, w, adv, cap


# ---------------------------------------------------------------- library

def test_default_library_shape_and_keys():
    lib = default_library()
    frame = lib.to_frame()
    assert (frame["kind"] == "historical").sum() == 8
    assert (frame["kind"] == "hypothetical").sum() == 8
    allowed = {"market", *STYLES} | {f"industry:{i}" for i in FF12}
    for s in lib:
        assert set(s.shocks) <= allowed, s.name
        assert set(s.basis) == set(s.shocks), s.name
        assert s.source and s.horizon_days >= 1
        if s.kind == "historical":
            assert s.window is not None
            assert s.basis["market"] == "index"
            assert {v for k, v in s.basis.items() if k != "market"} <= {"judgment"}
    assert len(lib.select(key=True)) >= 4
    assert "liquidity_crunch" in lib.select(tags=["liquidity"])


def test_library_json_round_trip(tmp_path):
    lib = default_library()
    path = tmp_path / "lib.json"
    lib.to_json(path)
    back = ScenarioLibrary.from_json(path)
    assert back.names == lib.names
    assert back["covid_crash_2020"] == lib["covid_crash_2020"]
    assert json.loads(lib.to_json())["scenarios"][0]["name"] == lib.names[0]


def test_scenario_variants_and_validation():
    s = Scenario("x", "hypothetical", {"market": -0.1, "beta": 0.02}, horizon_days=5)
    assert s.scaled(2).shocks == {"market": -0.2, "beta": 0.04}
    assert s.with_shocks(size=0.01).shocks["size"] == 0.01
    with pytest.raises(ValueError):
        Scenario("bad", "imaginary", {"market": -0.1})


# ---------------------------------------------------------------- engine

def test_factor_pnl_and_stressed_residual_by_hand():
    rm = toy_risk_model()
    w = toy_weights(rm)
    sc = Scenario("t", "hypothetical", {"market": -0.10, "industry:TECH": -0.05, "beta": 0.02},
                  horizon_days=10, vol_multiplier=2.0)
    cfg = StressConfig(propagation="zero", specific_confidence=0.99)
    r = apply_scenario(w, rm, sc, cfg)
    f = pd.Series({"market": -0.10, "industry:TECH": -0.05, "industry:BANK": 0.0,
                   "beta": 0.02, "size": 0.0})
    x = rm.exposures.T @ w
    assert r.summary["factor_pnl"] == pytest.approx(float(x @ f))
    assert r.summary["market_pnl"] == pytest.approx(x["market"] * -0.10)
    spec = norm_ppf(0.99) * 2.0 * math.sqrt(10 / 252 * float((w ** 2 * rm.specific_variance).sum()))
    assert r.summary["specific_stress"] == pytest.approx(spec)
    assert r.summary["pnl"] == pytest.approx(r.summary["factor_pnl"] - spec)
    # parts add up
    assert r.by_position["pnl"].sum() == pytest.approx(r.summary["pnl"])
    assert r.by_position["specific_stress"].sum() == pytest.approx(spec)
    assert r.by_factor["pnl"].sum() == pytest.approx(r.summary["factor_pnl"])
    assert r.summary["long_pnl"] + r.summary["short_pnl"] == pytest.approx(r.summary["pnl"])


def test_specific_horizon_is_capped():
    rm = toy_risk_model()
    w = toy_weights(rm)
    long = Scenario("long", "hypothetical", {"market": -0.1}, horizon_days=500)
    month = Scenario("month", "hypothetical", {"market": -0.1}, horizon_days=21)
    cfg = StressConfig(max_specific_horizon_days=21)
    a = apply_scenario(w, rm, long, cfg).summary["specific_stress"]
    b = apply_scenario(w, rm, month, cfg).summary["specific_stress"]
    assert a == pytest.approx(b)


def test_conditional_propagation():
    rm = toy_risk_model()
    F = rm.factor_covariance
    out = full_shock({"market": -0.2}, F, "conditional")
    assert out.loc["market", "source"] == "given"
    assert out.loc["beta", "shock"] == pytest.approx(F.loc["beta", "market"] / F.loc["market", "market"] * -0.2)
    assert out.loc["size", "shock"] == pytest.approx(0.0)
    zero = full_shock({"market": -0.2}, F, "zero")
    assert zero.loc["beta", "shock"] == 0.0 and zero.loc["beta", "source"] == "zero"


def test_map_shocks_drops_and_maps():
    factors = pd.Index(["market", "beta", "industry:TECH"])
    given, dropped = map_shocks({"market": -0.1, "momentum": 0.02, "industry:BusEq": -0.05},
                                factors, {"BusEq": "TECH"})
    assert given == {"market": -0.1, "industry:TECH": -0.05}
    assert dropped == ["momentum"]


def test_tester_report_order_and_limits():
    rm = toy_risk_model()
    w = toy_weights(rm).clip(lower=0) * 3        # long-only, so a crash hurts
    lib = ScenarioLibrary([
        Scenario("small", "hypothetical", {"market": -0.02}, horizon_days=1),
        Scenario("big", "hypothetical", {"market": -0.30}, horizon_days=5),
        Scenario("up", "hypothetical", {"market": 0.10}, horizon_days=5),
    ])
    rep = StressTester(StressConfig(limits=StressLimits(max_loss=0.10)), lib).run(w, rm, nav=1e8)
    assert list(rep.summary.index)[0] == "big"
    assert rep.summary["pnl"].is_monotonic_increasing
    assert rep.worst.scenario.name == "big"
    b = rep.breaches
    assert (b[(b["kind"] == "stress_loss") & (b["name"] == "big")]["severity"] == "breach").all()
    assert not rep.ok
    assert "market" in rep.by_factor.index
    assert list(rep.by_position.columns) == list(rep.results)


def test_dropped_shocks_are_reported(synth_book):
    _, rm, w, _, _ = synth_book
    rep = StressTester().run(w, rm, nav=1e8, scenarios=["gfc_lehman_2008"])
    assert rep.results["gfc_lehman_2008"].dropped_shocks == ["industry:Money"]
    assert "dropped_shocks" in set(rep.breaches["kind"])
    mapped = StressTester(StressConfig(industry_map=SYNTH_MAP)).run(
        w, rm, nav=1e8, scenarios=["gfc_lehman_2008"])
    assert mapped.results["gfc_lehman_2008"].dropped_shocks == []
    assert "approximate_scenarios" in set(mapped.breaches["kind"])


# ---------------------------------------------------------------- calibration

def test_calibrate_from_factor_returns():
    idx = pd.bdate_range("2020-01-01", periods=30)
    fr = pd.DataFrame({"market": 0.01, "beta": -0.002}, index=idx)
    s = calibrate_from_factor_returns(fr, idx[4], idx[14], name="w")
    # returns after the close of start through end: 10 days
    assert s.shocks["market"] == pytest.approx(0.10)
    assert s.shocks["beta"] == pytest.approx(-0.02)
    assert s.horizon_days == 10
    assert set(s.basis.values()) == {"calibrated"}
    with pytest.raises(ValueError):
        calibrate_from_factor_returns(fr, "1990-01-01", "1990-02-01")


def test_calibrate_from_reference_and_recalibrate():
    idx = pd.bdate_range("1998-07-01", "1998-09-30")
    ref = pd.DataFrame({"MKT": -0.005, "SMB": 0.001, "UMD": 0.002}, index=idx)
    base = default_library()["ltcm_russia_1998"]
    s = calibrate_from_reference(ref, *base.window, base=base)
    n = len(ref.loc[(ref.index > pd.Timestamp(base.window[0])) & (ref.index <= pd.Timestamp(base.window[1]))])
    assert s.shocks["market"] == pytest.approx(-0.005 * n)
    assert s.shocks["size"] == pytest.approx(-0.5 * 0.001 * n)
    assert s.basis["market"] == "calibrated"
    assert s.shocks["industry:Money"] == base.shocks["industry:Money"]   # kept, unmapped
    lib, report = recalibrate_library(default_library(), reference=ref)
    assert lib["ltcm_russia_1998"].shocks["market"] == pytest.approx(s.shocks["market"])
    assert lib["covid_crash_2020"] == default_library()["covid_crash_2020"]   # not covered
    assert "ltcm_russia_1998" in report.index


# ---------------------------------------------------------------- overlay

def test_liquidity_overlay(synth_book):
    _, rm, w, adv, cap = synth_book
    rep = StressTester(StressConfig(industry_map=SYNTH_MAP)).run(w, rm, nav=5e8)
    ov = liquidity_overlay(rep, adv, 5e8, rm, market_cap=cap)
    keys = {s.name for s in default_library() if s.key}
    assert set(ov.summary.index) == keys
    s = ov.summary
    assert (s["cut_cost_stressed"] >= s["cut_cost_normal"] - 1e-12).all()
    assert (s["cut_days_stressed"] >= s["cut_days_normal"] - 1e-12).all()
    assert np.allclose(s["total_pnl"], s["scenario_pnl"] - s["cut_cost_stressed"])
    assert (s["liquidatable_5d_stressed"] <= s["liquidatable_5d_normal"] + 1e-12).all()
    crunch = s.loc["liquidity_crunch"]
    assert crunch["volume_multiplier"] == 0.3


# ---------------------------------------------------------------- squeeze

def test_squeeze_by_hand():
    w = pd.Series({"L": 0.10, "S1": -0.05, "S2": -0.04})
    adv = pd.Series({"L": 1e8, "S1": 1e6, "S2": 1e8})
    sd = pd.DataFrame({"short_interest": [0.30, 0.02], "days_to_cover": [8.0, 1.0]},
                      index=["S1", "S2"])
    cfg = SqueezeConfig(moves=(0.30,), recall_days=3, max_participation=0.25)
    res = short_squeeze(w, adv, 1e7, short_data=sd, config=cfg,
                        spread_bps=pd.Series(10.0, index=w.index))
    assert list(res.crowded.index[res.crowded["crowded"]]) == ["S1"]
    row = res.summary.loc[0.30]
    assert row["move_pnl"] == pytest.approx(-0.05 * 0.30)
    assert row["cover_cost"] > 0
    assert row["pnl"] == pytest.approx(row["move_pnl"] - row["cover_cost"])
    # recalled short worth 6.5% of NAV = $650k against $1m ADV in 3 days: 22% a day
    pos = res.positions[0.30].loc["S1"]
    assert pos["participation_needed"] == pytest.approx(650e3 / 3 / 1e6)
    assert not pos["buy_in_risk"]
    # net after, as a share of NAV after the squeeze
    assert row["net_after"] == pytest.approx((0.10 - 0.04) / (1 + row["pnl"]))
    tight = short_squeeze(w, adv, 1e7, short_data=sd, config=cfg.with_overrides(recall_days=1))
    assert tight.positions[0.30].loc["S1", "buy_in_risk"]
    assert "buy_in_risk" in set(tight.breaches["kind"])


def test_squeeze_without_short_data_treats_all_shorts_as_crowded():
    w = pd.Series({"L": 0.10, "S1": -0.05, "S2": -0.04})
    adv = pd.Series(1e8, index=w.index)
    res = short_squeeze(w, adv, 1e7)
    assert res.crowded["crowded"].all()
    assert "squeeze_fallback" in set(res.breaches["kind"])
    assert res.summary.loc[0.50, "move_pnl"] == pytest.approx(-0.09 * 0.50)
    with pytest.raises(ValueError):
        short_squeeze(w.clip(lower=0), adv, 1e7)


def test_crowded_override_and_finra_point_in_time():
    w = pd.Series({"S1": -0.05, "S2": -0.04})
    sd = pd.DataFrame({"days_to_cover": [1.0, 1.0], "crowded": [True, np.nan]}, index=["S1", "S2"])
    c = identify_crowded(w, sd, SqueezeConfig())
    assert bool(c.loc["S1", "crowded"]) and not bool(c.loc["S2", "crowded"])
    finra = pd.DataFrame({
        "symbol": ["AAA", "AAA", "AAA"],
        "settlement_date": pd.to_datetime(["2026-10-15", "2026-10-31", "2026-11-14"]),
        "available_date": pd.to_datetime(["2026-10-24", "2026-11-11", "2026-11-25"]),
        "short_interest": [1e6, 2e6, 3e6],
        "days_to_cover": [2.0, 4.0, 6.0],
    })
    latest = latest_short_interest(finra, "2026-11-20", {"AAA": "A1"})
    assert latest.loc["A1", "days_to_cover"] == 4.0           # the Nov 14 data isn't out yet


# ---------------------------------------------------------------- reverse stress

def test_reverse_stress_most_likely_scenario_loses_target():
    rm = toy_risk_model()
    w = toy_weights(rm)
    rv = reverse_stress(w, rm, target_loss=0.05, horizon_days=21)
    assert rv.most_likely_shocks["pnl"].sum() == pytest.approx(-0.05)
    S = rm.factor_covariance * 21 / 252
    x = rm.exposures.T @ w
    assert rv.most_likely["distance_factors"] == pytest.approx(0.05 / math.sqrt(float(x @ S @ x)))
    assert rv.most_likely["distance_with_specific"] < rv.most_likely["distance_factors"]
    top = rv.factors.index[0]
    assert rv.factors.loc[top, "move_isolated"] * x[top] == pytest.approx(-0.05)
    for name, r in rv.positions.iterrows():
        assert r["move_needed"] * r["weight"] == pytest.approx(-0.05)
        if r["weight"] > 0 and r["weight"] < 0.05:
            assert not r["reachable"]


def test_reverse_stress_scenario_multiples(synth_book):
    _, rm, w, _, _ = synth_book
    long_only = w.clip(lower=0)
    rep = StressTester(StressConfig(industry_map=SYNTH_MAP)).run(long_only, rm, nav=1e8)
    rv = reverse_stress(long_only, rm, 0.20, stress_report=rep)
    r = rv.scenarios.loc["covid_crash_2020"]
    assert r["specific_stress"] + r["multiple_needed"] * r["factor_loss"] == pytest.approx(0.20)


# ---------------------------------------------------------------- reports

def test_daily_liquidity_report(synth_book):
    data, rm, w, _, _ = synth_book
    rep = daily_liquidity_report(w, 5e8, data=data, risk_model=rm)
    assert rep.cadence == "daily_liquidity"
    assert {"buckets", "profile"} <= set(rep.tables)
    assert "source" in rep.breaches.columns
    assert "liquidatable_5d_gross" in rep.headline
    assert "daily_liquidity report" in rep.text()


def test_periodic_stress_report(synth_book):
    data, rm, w, _, _ = synth_book
    rep = periodic_stress_report(w, rm, 5e8, data=data, frequency="monthly",
                                 stress_config=StressConfig(industry_map=SYNTH_MAP))
    assert rep.cadence == "monthly_stress"
    assert {"scenarios", "liquidity overlay", "short squeeze",
            "reverse stress: top factors"} <= set(rep.tables)
    assert set(rep.breaches["source"]) >= {"stress", "squeeze"}
    sev = rep.breaches["severity"].map({"breach": 0, "alert": 1, "watch": 2, "warning": 3, "info": 4})
    assert sev.is_monotonic_increasing
    with pytest.raises(ValueError):
        periodic_stress_report(w, rm, 5e8, data=data, frequency="daily")


def test_ad_hoc_report_before_and_after(synth_book):
    data, rm, w, _, _ = synth_book
    trade = pd.Series({w.index[0]: 0.05})
    rep = ad_hoc_stress_report(w, rm, 5e8, trades=trade, tags=["crash"], data=data,
                               stress_config=StressConfig(industry_map=SYNTH_MAP))
    cmp = rep.tables["scenarios before and after"]
    assert np.allclose(cmp["change"], cmp["pnl_after"] - cmp["pnl_before"])
    assert set(cmp.index) == set(default_library().select(tags=["crash"]).names)
    assert "after" in rep.tables["liquidity"].columns
    event = ad_hoc_stress_report(w, rm, 5e8, tags=["rates"], reason="FOMC")
    assert event.headline["reason"] == "FOMC"
    with pytest.raises(ValueError):
        ad_hoc_stress_report(w, rm, 5e8, trades=trade, proposed=w)
    with pytest.raises(ValueError):
        ad_hoc_stress_report(w, rm, 5e8, tags=["no_such_tag"])
