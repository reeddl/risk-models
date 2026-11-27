import math

import numpy as np
import pandas as pd
import pytest

from riskmodels.liquidity import (
    NO_ADV,
    LiquidityConfig,
    LiquidityLimits,
    LiquidityModel,
    adv_from_data,
    bucket,
    days_to_liquidate,
    days_to_liquidate_share,
    liquidation_profile,
    market_cap_from_data,
    reduce_gross,
    resolve_inputs,
    spreads_from_market_cap,
    trade_cost,
)

NAV = 10e6


@pytest.fixture
def toy():
    # A: 10% of NAV, ADV $1m -> 1e6 / (0.2 x 1e6) = 5 days
    # B: -5% of NAV, ADV $5m -> 5e5 / 1e6 = 0.5 days
    # C: 2% of NAV, no ADV
    # D: 3% of NAV, ADV $250k -> 3e5 / 5e4 = 6 days
    w = pd.Series({"A": 0.10, "B": -0.05, "C": 0.02, "D": 0.03})
    adv = pd.Series({"A": 1e6, "B": 5e6, "C": np.nan, "D": 2.5e5})
    return w, adv


def test_days_to_liquidate_formula(toy):
    w, adv = toy
    d = days_to_liquidate(w, adv, NAV, 0.20)
    assert d["A"] == pytest.approx(5.0)
    assert d["B"] == pytest.approx(0.5)
    assert math.isinf(d["C"])
    assert d["D"] == pytest.approx(6.0)
    # half the volume, twice the days
    assert days_to_liquidate(w, adv, NAV, 0.20, volume_multiplier=0.5)["A"] == pytest.approx(10.0)


def test_buckets():
    cfg = LiquidityConfig()
    b = bucket(pd.Series({"a": 0.5, "b": 3.0, "c": 7.0, "d": 12.0, "e": np.inf}), cfg)
    assert list(b) == ["<1d", "1-5d", "5-10d", ">10d", NO_ADV]
    assert cfg.bucket_labels == ["<1d", "1-5d", "5-10d", ">10d"]


def test_liquidation_profile_by_hand(toy):
    w, adv = toy
    cfg = LiquidityConfig()
    prof = liquidation_profile(w, adv, NAV, cfg, days=(1, 5, 10))
    gross = w.abs().sum()
    # one day: A 2%, B 5% (all of it), C 0, D 0.5%
    assert prof.loc[1, "gross"] == pytest.approx(0.02 + 0.05 + 0.005)
    assert prof.loc[1, "share_of_gross"] == pytest.approx(0.075 / gross)
    # five days: A 10%, B 5%, D 2.5%
    assert prof.loc[5, "gross"] == pytest.approx(0.175)
    # ten days: everything except C
    assert prof.loc[10, "gross"] == pytest.approx(0.18)
    assert prof.loc[10, "share_of_nav"] == pytest.approx(0.18)
    assert (prof["share_of_gross"].diff().dropna() >= 0).all()


def test_days_to_exit_share(toy):
    w, adv = toy
    assert math.isinf(days_to_liquidate_share(w, adv, NAV, 1.0, 0.20))   # C has no ADV
    d = days_to_liquidate_share(w.drop("C"), adv, NAV, 1.0, 0.20)
    assert d == pytest.approx(6.0, rel=1e-6)                              # D is slowest


def test_trade_cost_half_spread_plus_square_root_impact():
    cfg = LiquidityConfig(impact_coefficient=1.0)
    inputs = resolve_inputs(pd.Index(["A"]), pd.Series({"A": 1e6}), cfg,
                            spread_bps=pd.Series({"A": 20.0}), daily_vol=pd.Series({"A": 0.02}))
    trade = pd.Series({"A": -0.025})   # $250k against $1m ADV
    tc = trade_cost(trade, inputs, NAV, cfg)
    rate = 0.0010 + 0.02 * math.sqrt(0.25)
    assert tc.loc["A", "cost_rate"] == pytest.approx(rate)
    assert tc.loc["A", "cost"] == pytest.approx(0.025 * rate)
    stressed = trade_cost(trade, inputs, NAV, cfg, volume_multiplier=0.25, spread_multiplier=2,
                          vol_multiplier=2)
    assert stressed.loc["A", "cost_rate"] == pytest.approx(0.0020 + 0.04 * 1.0)


def test_spread_tiers_and_defaults():
    cfg = LiquidityConfig()
    s = spreads_from_market_cap(pd.Series({"mega": 50e9, "mid": 5e9, "small": 1e9, "micro": 1e8}),
                                cfg.spread_tiers)
    assert list(s) == [4, 10, 25, 60]
    inputs = resolve_inputs(pd.Index(["x", "y"]), pd.Series({"x": 1e6, "y": 1e6}), cfg,
                            market_cap=pd.Series({"x": 50e9}))
    assert inputs.loc["x", "spread_source"] == "cap_tier"
    assert inputs.loc["y", "spread_source"] == "default"
    assert inputs.loc["y", "spread_bps"] == cfg.default_spread_bps


def test_pro_rata_and_fastest_reduction(toy):
    w, adv = toy
    cfg = LiquidityConfig()
    inputs = resolve_inputs(w.index, adv, cfg, spread_bps=pd.Series(10.0, index=w.index),
                            daily_vol=pd.Series(0.02, index=w.index))
    pr = reduce_gross(w, inputs, NAV, cfg, target=0.30, method="pro_rata")
    # C can't trade, so the cut falls short of 30%
    assert pr.summary["achieved_cut"] == pytest.approx(0.30 * 0.18 / 0.20)
    assert pr.summary["untradable_names"] == 1
    # every tradable name cut by 30%: D's 0.9% takes 0.9e5 / 5e4 = 1.8 days, A's 3% takes 1.5
    assert pr.summary["days_to_complete"] == pytest.approx(1.8)
    fast = reduce_gross(w.drop("C"), inputs, NAV, cfg, target=0.30, method="fastest")
    assert fast.summary["achieved_cut"] == pytest.approx(0.30)
    slow = reduce_gross(w.drop("C"), inputs, NAV, cfg, target=0.30, method="pro_rata")
    assert fast.summary["days_to_complete"] <= slow.summary["days_to_complete"] + 1e-12
    stressed = reduce_gross(w.drop("C"), inputs, NAV, cfg, target=0.30, volume_multiplier=0.5,
                            spread_multiplier=3)
    assert stressed.summary["cost"] > slow.summary["cost"]
    assert stressed.summary["days_to_complete"] == pytest.approx(2 * slow.summary["days_to_complete"])


def test_report_and_limits(toy):
    w, adv = toy
    rep = LiquidityModel().run(w, adv, NAV)
    assert rep.positions.loc["A", "bucket"] in ("1-5d", "5-10d")
    assert rep.positions.loc["C", "bucket"] == NO_ADV
    assert rep.buckets["gross"].sum() == pytest.approx(w.abs().sum())
    assert rep.summary["n_no_adv"] == 1
    kinds = set(rep.breaches["kind"])
    assert "no_adv" in kinds
    # A is exactly 1.0x ADV: at the limit is a watch, not a breach
    pa = rep.breaches[rep.breaches["kind"] == "position_adv"].set_index("name")
    assert pa.loc["A", "severity"] == "watch"
    # D is 1.2x ADV: a breach
    assert pa.loc["D", "severity"] == "breach"
    assert not rep.ok


def test_gross_basis_is_stricter_than_nav_for_a_levered_book():
    # 200% gross book: 1.2 of NAV liquid in a day, 0.8 stuck (no ADV)
    w = pd.Series({"liq_long": 0.6, "liq_short": -0.6, "stuck_long": 0.4, "stuck_short": -0.4})
    adv = pd.Series({"liq_long": 1e12, "liq_short": 1e12, "stuck_long": np.nan, "stuck_short": np.nan})
    gross_cfg = LiquidityConfig(limits=LiquidityLimits(min_liquidatable={5: 0.80}))
    nav_cfg = LiquidityConfig(limits=LiquidityLimits(min_liquidatable={5: 0.80}, basis="nav"))
    g = LiquidityModel(gross_cfg).run(w, adv, NAV).breaches
    n = LiquidityModel(nav_cfg).run(w, adv, NAV).breaches
    assert (g[g["kind"] == "liquidatable"]["severity"] == "breach").any()   # 60% of gross
    assert not (n[n["kind"] == "liquidatable"]["severity"] == "breach").any()  # 120% of NAV


def test_config_validation_and_overrides():
    cfg = LiquidityConfig().with_overrides(participation=0.10)
    assert cfg.participation == 0.10
    with pytest.raises(ValueError):
        LiquidityConfig(participation=0)
    with pytest.raises(ValueError):
        LiquidityConfig(reduction_method="random")


def test_synthetic_book(synthetic, daily_fit):
    data, _ = synthetic
    rm = daily_fit.risk_model()
    adv = adv_from_data(data, rm.as_of)
    cap = market_cap_from_data(data, rm.as_of)
    assert (adv.dropna() > 0).all() and len(adv.dropna()) > 150
    names = rm.exposures.index[:40]
    rng = np.random.default_rng(3)
    w = pd.Series(rng.choice([-1, 1], 40) * rng.uniform(0.005, 0.03, 40), index=names)
    rep = LiquidityModel().run(w, adv, 500e6, market_cap=cap, risk_model=rm)
    assert rep.buckets["share_of_gross"].sum() == pytest.approx(1.0)
    assert 0 < rep.summary["reduction_cost"] < rep.summary["full_exit_cost"]
    assert rep.positions["days_to_liquidate"].is_monotonic_decreasing
