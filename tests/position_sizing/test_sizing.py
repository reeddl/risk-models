import numpy as np
import pandas as pd
import pytest

from riskmodels.factor_model.risk import RiskModel
from riskmodels.position_sizing import (
    PositionSizer,
    SizingConfig,
    apply_caps,
    cap_weighted_benchmark,
    conviction_multiplier,
    drift_weights,
    liquidity_cap,
    predicted_betas,
    short_trim_factor,
    trade_list,
)

LOOSE = dict(max_long_weight=1.0, max_short_weight=1.0, max_sector_gross=None,
             max_sector_net=None, max_gross=100.0, min_net=-10.0, max_net=10.0)


def toy_risk_model(n=20, seed=3) -> RiskModel:
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


def toy_signals(rm, n_long=10, n_short=10) -> pd.DataFrame:
    names = rm.exposures.index[: n_long + n_short]
    side = ["long"] * n_long + ["short"] * n_short
    conviction = np.resize([1, 2, 3, 4, 5], len(names))
    return pd.DataFrame({"side": side, "conviction": conviction}, index=names)


@pytest.fixture
def rm():
    return toy_risk_model()


@pytest.fixture
def bench(rm):
    return pd.Series(1.0 / len(rm.exposures), index=rm.exposures.index)


# ---------------------------------------------------------------- steps 1-3

def test_conviction_multiplier_interpolates_and_rejects_out_of_range():
    m = conviction_multiplier(pd.Series([1, 2, 3, 4, 5, 3.5]), {1: 0.5, 3: 1.0, 5: 1.5})
    assert m.tolist() == pytest.approx([0.5, 0.75, 1.0, 1.25, 1.5, 1.125])
    with pytest.raises(ValueError):
        conviction_multiplier(pd.Series([6]), {1: 0.5, 3: 1.0, 5: 1.5})
    with pytest.raises(ValueError):
        conviction_multiplier(pd.Series([np.nan]), {1: 0.5, 5: 1.5})


@pytest.mark.parametrize("vol", ["specific", "total"])
def test_raw_weight_is_risk_budget_over_vol(rm, vol):
    cfg = SizingConfig(volatility=vol, neutralize=None, target_volatility=None, **LOOSE)
    sig = toy_signals(rm)
    out = PositionSizer(cfg).size(sig, rm, nav=1e8)
    sigma = rm.specific_risk if vol == "specific" else rm.total_risk()
    mult = conviction_multiplier(sig["conviction"], cfg.conviction_multipliers)
    side = np.where(sig["side"] == "long", 1.0, -cfg.short_risk_multiplier)
    expected = side * cfg.base_risk * mult / sigma.reindex(sig.index)
    pd.testing.assert_series_equal(out.weights, expected, check_names=False)
    assert out.positions["volatility"].equals(sigma.reindex(sig.index))


def test_short_trim_steps_and_cover():
    cfg = SizingConfig(short_trim_step=0.10, short_trim_fraction=0.25)
    move = pd.Series([0.0, 0.05, 0.10, 0.25, 0.39])
    assert short_trim_factor(move, cfg).tolist() == pytest.approx(
        [1.0, 1.0, 0.75, 0.5625, 0.421875])
    cfg = cfg.with_overrides(short_cover_move=0.30)
    assert short_trim_factor(move, cfg).iloc[-1] == 0.0


def test_trim_only_applies_to_shorts_moving_against_you(rm):
    cfg = SizingConfig(neutralize=None, target_volatility=None, **LOOSE)
    sig = toy_signals(rm, 2, 2)
    sig["price"] = [120.0, 80.0, 125.0, 80.0]
    sig["entry_price"] = 100.0
    out = PositionSizer(cfg).size(sig, rm, nav=1e8)
    trim = out.positions["trim"]
    # long up 20% and long down 20%: untouched; short up 25%: two steps; short down: none
    assert trim.tolist() == pytest.approx([1.0, 1.0, 0.75 ** 2, 1.0])
    assert (out.breaches["kind"] == "short_trim").sum() == 1


# ---------------------------------------------------------------- step 4

def test_liquidity_cap_formula():
    cfg = SizingConfig(adv_participation=0.2, liquidation_days=5)
    adv_shares, price, nav = 1_000_000, 50.0, 1e9
    cap = liquidity_cap(pd.Series([adv_shares * price]), nav, cfg)
    assert cap.iloc[0] == pytest.approx(0.2 * adv_shares * 5 * price / nav)


def test_single_name_and_liquidity_caps_bind(rm):
    cfg = SizingConfig(neutralize=None, target_volatility=None, base_risk=0.05,
                       max_sector_gross=None, max_sector_net=None, max_gross=100.0,
                       min_net=-10.0, max_net=10.0)
    sig = toy_signals(rm, 2, 2)
    sig["price"] = 10.0
    sig["adv"] = [1e9, 1e3, 1e9, 1e9]  # second long is illiquid
    out = PositionSizer(cfg).size(sig, rm, nav=1e8)
    w = out.weights
    liq = 0.2 * 1e3 * 10.0 * 5 / 1e8
    assert w.iloc[0] == pytest.approx(cfg.max_long_weight)
    assert w.iloc[1] == pytest.approx(liq)
    assert w.iloc[2] == pytest.approx(-cfg.max_short_weight)
    assert "liquidity" in out.positions["binding"].iloc[1]
    assert "single_name" in out.positions["binding"].iloc[0]
    assert out.positions["days_to_liquidate"].iloc[1] == pytest.approx(5.0)
    assert out.ok


def test_sector_gross_net_and_book_caps():
    cfg = SizingConfig(max_long_weight=1, max_short_weight=1, max_sector_gross=0.5,
                       max_sector_net=0.1, max_gross=0.8, min_net=-0.05, max_net=0.05)
    w = pd.Series([0.4, 0.3, -0.1, 0.2, -0.05], index=list("abcde"))
    sector = pd.Series(["X", "X", "X", "Y", "Y"], index=w.index)
    caps = pd.Series(1.0, index=w.index)
    out, binding = apply_caps(w, caps, sector, cfg)
    by = out.groupby(sector)
    assert (by.apply(lambda s: s.abs().sum()) <= 0.5 + 1e-9).all()
    assert (by.sum().abs() <= 0.1 + 1e-9).all()
    assert out.abs().sum() <= 0.8 + 1e-9
    assert -0.05 - 1e-9 <= out.sum() <= 0.05 + 1e-9
    assert (np.abs(out) <= np.abs(w) + 1e-12).all()  # caps only shrink
    assert (np.sign(out[out != 0]) == np.sign(w[out != 0])).all()
    assert "sector_gross" in binding["a"]


# ---------------------------------------------------------------- steps 5-6

def test_predicted_beta_matches_covariance(rm, bench):
    V = rm.asset_covariance()
    b = bench.reindex(V.index)
    expected = V.to_numpy() @ b.to_numpy() / (b.to_numpy() @ V.to_numpy() @ b.to_numpy())
    assert predicted_betas(rm, bench).to_numpy() == pytest.approx(expected)
    assert float(predicted_betas(rm, bench) @ b) == pytest.approx(1.0)


@pytest.mark.parametrize("target", [0.0, 0.1, -0.05])
def test_beta_and_vol_targets_hit_when_caps_allow(rm, bench, target):
    cfg = SizingConfig(target_exposure=target, target_volatility=0.06, **LOOSE)
    out = PositionSizer(cfg).size(toy_signals(rm), rm, nav=1e8, benchmark=bench)
    beta = predicted_betas(rm, bench).reindex(out.weights.index)
    assert float(out.weights @ beta) == pytest.approx(target, abs=1e-3)
    assert out.risk.total_risk == pytest.approx(0.06, rel=1e-3)
    assert out.summary["converged"]
    assert not out.breaches["kind"].isin(["beta_target", "vol_target"]).any()


def test_net_target_without_vol_target_shrinks_excess_side(rm):
    cfg = SizingConfig(neutralize="net", target_exposure=0.0, target_volatility=None, **LOOSE)
    sig = toy_signals(rm, 10, 10)
    sizer = PositionSizer(cfg)
    raw = sizer.with_config(neutralize=None).size(sig, rm, nav=1e8).weights
    out = sizer.size(sig, rm, nav=1e8)
    assert out.weights.sum() == pytest.approx(0.0, abs=1e-3)
    # "shrink": one book is untouched, the other only gets smaller
    longs, shorts = raw > 0, raw < 0
    long_ratio = (out.weights[longs] / raw[longs]).unique()
    short_ratio = (out.weights[shorts] / raw[shorts]).unique()
    assert max(long_ratio.max(), short_ratio.max()) == pytest.approx(1.0)
    assert min(long_ratio.min(), short_ratio.min()) < 1.0


def test_caps_hold_after_vol_scaling_up(rm, bench):
    # A vol target far above what the caps allow: caps must still hold, miss is reported.
    cfg = SizingConfig(target_volatility=0.50, max_long_weight=0.05, max_short_weight=0.03)
    out = PositionSizer(cfg).size(toy_signals(rm), rm, nav=1e8, benchmark=bench)
    w = out.weights
    assert w[w > 0].max() <= 0.05 + 1e-9
    assert (-w[w < 0]).max() <= 0.03 + 1e-9
    assert w.abs().sum() <= cfg.max_gross + 1e-9
    assert out.ok
    assert (out.breaches["kind"] == "vol_target").any()
    assert not out.summary["converged"]


def test_beta_target_beats_vol_target_when_both_cannot_hold(rm, bench):
    cfg = SizingConfig(target_volatility=0.50, max_long_weight=0.05, max_short_weight=0.03)
    out = PositionSizer(cfg).size(toy_signals(rm), rm, nav=1e8, benchmark=bench)
    beta = predicted_betas(rm, bench).reindex(out.weights.index)
    assert float(out.weights @ beta) == pytest.approx(0.0, abs=1e-3)


def test_factor_bets_are_flagged(rm, bench):
    # Long the highest-size names, short the lowest: a big size bet.
    size = rm.exposures["size"].sort_values()
    names = list(size.index[-5:]) + list(size.index[:5])
    sig = pd.DataFrame({"side": ["long"] * 5 + ["short"] * 5, "conviction": 3}, index=names)
    out = PositionSizer(SizingConfig(**LOOSE)).size(sig, rm, nav=1e8, benchmark=bench)
    flagged = out.breaches.query("kind == 'style_exposure'")["name"].tolist()
    assert "size" in flagged


# ---------------------------------------------------------------- step 7

def test_trade_list_exits_and_min_trade():
    target = pd.Series({"a": 0.05, "b": -0.03, "c": 0.0201})
    current = pd.Series({"a": 0.02, "c": 0.02, "d": 0.01, "e": -0.02})
    t = trade_list(target, current, nav=1e6, price=pd.Series(10.0, index=list("abcde")),
                   adv_dollars=None, min_trade=0.001)
    assert set(t.index) == {"a", "b", "d", "e"}  # c's trade is below min_trade
    assert t.loc["a", "action"] == "buy" and t.loc["a", "shares"] == 3000
    assert t.loc["b", "action"] == "short"
    assert t.loc["d", "action"] == "sell" and t.loc["d", "target_weight"] == 0
    assert t.loc["e", "action"] == "cover"


def test_drift_weights():
    w = pd.Series({"a": 0.5, "b": -0.5})
    out = drift_weights(w, pd.Series({"a": 0.10, "b": 0.10}))
    assert out["a"] == pytest.approx(0.55) and out["b"] == pytest.approx(-0.55)


def test_monitor_flags_breaches_and_drift(rm, bench):
    cfg = SizingConfig(**{**LOOSE, "max_long_weight": 0.05})
    sizer = PositionSizer(cfg)
    sig = toy_signals(rm)
    target = sizer.size(sig, rm, nav=1e8, benchmark=bench).weights
    current = target.copy()
    current.iloc[0] = 0.08  # over the single-name cap and far from target
    mon = sizer.monitor(current, sig, rm, nav=1e8, benchmark=bench)
    kinds = set(mon.breaches["kind"])
    assert {"single_name_long", "drift"} <= kinds
    assert not mon.ok
    assert mon.resized.trades.loc[current.index[0], "trade_weight"] < 0


def test_missing_inputs_raise(rm):
    with pytest.raises(ValueError, match="benchmark"):
        PositionSizer().size(toy_signals(rm), rm, nav=1e8)
    with pytest.raises(KeyError):
        sig = toy_signals(rm)
        sig.index = ["X"] + list(sig.index[1:])
        PositionSizer(SizingConfig(neutralize=None)).size(sig, rm, nav=1e8)
    with pytest.raises(ValueError):
        SizingConfig(volatility="realized")


# ---------------------------------------------------------------- synthetic end to end

def test_end_to_end_on_synthetic_fit(synthetic, daily_fit):
    data, _ = synthetic
    rm = daily_fit.risk_model()
    bench = cap_weighted_benchmark(daily_fit)
    rng = np.random.default_rng(11)
    names = rm.exposures.index[rng.permutation(len(rm.exposures))[:80]]
    sig = pd.DataFrame({
        "side": np.where(np.arange(80) % 2 == 0, "long", "short"),
        "conviction": rng.integers(1, 6, 80),
        "price": 50.0,
        "adv_dollars": data.dollar_volume.iloc[-20:].mean().reindex(names).to_numpy(),
    }, index=names)
    cfg = SizingConfig(target_volatility=0.04)
    out = PositionSizer(cfg).size(sig, rm, nav=5e8, benchmark=bench)

    w = out.weights
    assert out.ok, out.breaches
    assert w[w > 0].max() <= cfg.max_long_weight + 1e-9
    assert (-w[w < 0]).max() <= cfg.max_short_weight + 1e-9
    assert (w.abs() <= out.positions["liquidity_cap"] + 1e-9).all()
    assert cfg.min_net - 1e-9 <= w.sum() <= cfg.max_net + 1e-9
    assert w.abs().sum() <= cfg.max_gross + 1e-9
    assert abs(out.summary["beta"]) <= 1e-3
    # predicted vol matches the factor model's own decomposition
    assert out.risk.total_risk == pytest.approx(rm.portfolio_risk(w).total_risk)
    assert out.risk.total_risk == pytest.approx(0.04, rel=1e-3) or not out.summary["vol_reachable"]
    report = out.risk_report()
    assert report["by_sector"]["net"].abs().max() <= cfg.max_sector_net + 1e-9
    assert len(out.trades) == (w != 0).sum()
