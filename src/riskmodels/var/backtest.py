"""VaR backtests: exceedance count, Kupiec, Christoffersen, traffic light, P&L feeds.

Timing: a VaR forecast made at the close of day t is compared with the P&L over day t+1.
Every series here is indexed by the P&L date, with the forecast that applied to it.

Two P&L feeds:

``hypothetical``  the book held at the close of t, revalued on day t+1's returns (no
                  trading). This tests the model alone. :func:`rolling_forecasts` builds it.
                  Pass a fixed weight Series to replay today's positions on past returns,
                  or a date x name DataFrame for the book as it actually stood each day.
``actual``        realized P&L from your books, passed in as a Series of fractions of NAV
                  (divide by the prior day's NAV). It includes intraday trading, fees and
                  financing, so an exceedance on actual P&L that is absent on hypothetical
                  P&L points at trading rather than the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.data import MarketData
from ..factor_model.model import FactorModelResult
from .config import VaRConfig
from .stats import binom_cdf, chi2_sf, xlogy


def exceedances(var: pd.Series, pnl: pd.Series) -> pd.Series:
    """True on days whose loss exceeded the VaR forecast for that day."""
    df = pd.concat({"var": var, "pnl": pnl}, axis=1).dropna()
    return (-df["pnl"] > df["var"]).rename("exceedance")


def kupiec_test(n_exceedances: int, n_obs: int, confidence: float) -> dict:
    """Kupiec (1995) proportion-of-failures test: is the exceedance rate equal to
    1 - confidence? Likelihood ratio, chi-square with 1 degree of freedom."""
    p = 1.0 - confidence
    x, n = int(n_exceedances), int(n_obs)
    if n <= 0:
        raise ValueError("no observations")
    phat = x / n
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(n - x, 1 - phat) + xlogy(x, phat)
    lr = max(-2.0 * (ll0 - ll1), 0.0)
    return {"lr_uc": lr, "p_uc": chi2_sf(lr, 1), "expected": n * p, "rate": phat}


def christoffersen_test(hits: pd.Series, confidence: float) -> dict:
    """Christoffersen (1998) independence test (are exceedances clustered?) and the joint
    conditional-coverage test (rate and independence together).

    LR_ind is chi-square(1); LR_cc = LR_uc + LR_ind is chi-square(2).
    """
    h = hits.astype(bool).to_numpy()
    prev, curr = h[:-1], h[1:]
    n00 = int((~prev & ~curr).sum())
    n01 = int((~prev & curr).sum())
    n10 = int((prev & ~curr).sum())
    n11 = int((prev & curr).sum())
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - p01) + xlogy(n01, p01) + xlogy(n10, 1 - p11) + xlogy(n11, p11)
    lr_ind = max(-2.0 * (ll0 - ll1), 0.0)
    uc = kupiec_test(int(h.sum()), len(h), confidence)
    lr_cc = uc["lr_uc"] + lr_ind
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "lr_ind": lr_ind, "p_ind": chi2_sf(lr_ind, 1),
            "lr_cc": lr_cc, "p_cc": chi2_sf(lr_cc, 2)}


def traffic_light(n_exceedances: int, n_obs: int, confidence: float) -> str:
    """Basel-style zone from the binomial probability of seeing this many exceedances or
    fewer: green below 95%, yellow below 99.99%, red otherwise. At 99% over 250 days this
    is green for 0-4, yellow for 5-9, red for 10 or more."""
    cum = binom_cdf(int(n_exceedances), int(n_obs), 1.0 - confidence)
    if cum < 0.95:
        return "green"
    if cum < 0.9999:
        return "yellow"
    return "red"


@dataclass
class BacktestResult:
    """One VaR series tested against one P&L series.

    ``passed`` is True when neither the Kupiec nor the conditional-coverage test rejects at
    the configured significance. Revisit parameters when it is False.
    """

    summary: dict
    hits: pd.Series
    frame: pd.DataFrame

    @property
    def passed(self) -> bool:
        return bool(self.summary["passed"])


def backtest(
    var: pd.Series,
    pnl: pd.Series,
    confidence: float = 0.99,
    significance: float = 0.05,
    cvar: pd.Series | None = None,
) -> BacktestResult:
    """Test a VaR forecast series against realized P&L (both fractions of NAV, aligned on
    the P&L date; VaR positive, P&L gains positive).

    With ``cvar`` given, also reports the mean loss on exceedance days against the mean
    CVaR forecast on those days (``cvar_ratio`` near 1 is good). This is a diagnostic,
    not a formal test: CVaR is harder to backtest than VaR.
    """
    cols = {"var": var, "pnl": pnl}
    if cvar is not None:
        cols["cvar"] = cvar
    frame = pd.concat(cols, axis=1).dropna(subset=["var", "pnl"])
    n = len(frame)
    if n == 0:
        raise ValueError("VaR and P&L share no dates")
    frame["loss"] = -frame["pnl"]
    frame["exceedance"] = frame["loss"] > frame["var"]
    hits = frame["exceedance"]
    x = int(hits.sum())
    uc = kupiec_test(x, n, confidence)
    cc = christoffersen_test(hits, confidence)
    summary = {
        "n_obs": n,
        "n_exceedances": x,
        "expected_exceedances": uc["expected"],
        "exceedance_rate": uc["rate"],
        "kupiec_lr": uc["lr_uc"],
        "kupiec_p": uc["p_uc"],
        "independence_lr": cc["lr_ind"],
        "independence_p": cc["p_ind"],
        "conditional_coverage_lr": cc["lr_cc"],
        "conditional_coverage_p": cc["p_cc"],
        "consecutive_exceedances": cc["n11"],
        "traffic_light": traffic_light(x, n, confidence),
        "mean_var": float(frame["var"].mean()),
        "worst_loss": float(frame["loss"].max()),
        "mean_exceedance_loss": float(frame.loc[hits, "loss"].mean()) if x else np.nan,
    }
    if "cvar" in frame and x:
        summary["mean_cvar_on_exceedances"] = float(frame.loc[hits, "cvar"].mean())
        summary["cvar_ratio"] = summary["mean_exceedance_loss"] / summary["mean_cvar_on_exceedances"]
    summary["passed"] = bool(uc["p_uc"] >= significance and cc["p_cc"] >= significance)
    return BacktestResult(summary=summary, hits=hits, frame=frame)


def rolling_forecasts(
    fit: FactorModelResult,
    data: MarketData,
    weights,
    config: VaRConfig | None = None,
    start=None,
    end=None,
    every: int = 1,
) -> pd.DataFrame:
    """Out-of-sample VaR forecasts and hypothetical P&L, one row per P&L date.

    For each exposure date t (every ``every``-th one between ``start`` and ``end``) the
    factor model's risk forecast as of t and the historical scenarios up to t give VaR and
    CVaR; the hypothetical P&L is the book at t revalued on day t+1's excess returns.

    ``weights`` is a Series (the same book every day, for example today's positions) or a
    DataFrame date x name (the book at each close; the last row on or before t is used).
    Names the risk model does not cover on t are dropped and counted in ``n_dropped``.
    """
    from .model import VaRModel

    c = config or VaRConfig()
    vm = VaRModel(c)
    dates = fit.exposure_dates
    trading = data.returns.index
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    dates = dates[::every]
    rf = data.risk_free if data.risk_free is not None else pd.Series(0.0, index=trading)

    rows = []
    for t in dates:
        pos = trading.searchsorted(t, side="right")
        if pos >= len(trading):
            continue
        nxt = trading[pos]
        if isinstance(weights, pd.DataFrame):
            hist = weights[weights.index <= t]
            if hist.empty:
                continue
            w = hist.iloc[-1]
        else:
            w = weights
        w = w.astype(float)
        w = w[w != 0]
        rm = fit.risk_model(t)
        covered = w.index.intersection(rm.exposures.index)
        dropped = len(w) - len(covered)
        w = w[covered]
        if w.empty:
            continue
        rep = vm.run(w, fit, data, risk_model=rm, decompose=False)
        r = data.returns.loc[nxt, w.index].astype(float).fillna(0.0) - float(rf.get(nxt, 0.0))
        rows.append({
            "date": nxt,
            "forecast_date": rm.as_of,
            "parametric_var": rep.summary["parametric_var_1d"],
            "parametric_cvar": rep.summary["parametric_cvar_1d"],
            "historical_var": rep.summary["historical_var_1d"],
            "historical_cvar": rep.summary["historical_cvar_1d"],
            "hypothetical_pnl": float((w * r).sum()),
            "n_names": len(w),
            "n_dropped": dropped,
        })
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def backtest_report(
    forecasts: pd.DataFrame,
    actual_pnl: pd.Series | None = None,
    config: VaRConfig | None = None,
) -> pd.DataFrame:
    """Backtest both VaR methods against hypothetical and (if given) actual P&L.

    ``forecasts`` is the output of :func:`rolling_forecasts`. Returns one row per
    (method, P&L feed) with the summary of :func:`backtest`.
    """
    c = config or VaRConfig()
    feeds = {"hypothetical": forecasts["hypothetical_pnl"]}
    if actual_pnl is not None:
        feeds["actual"] = actual_pnl
    rows = {}
    for method in ("parametric", "historical"):
        for feed, pnl in feeds.items():
            res = backtest(forecasts[f"{method}_var"], pnl, c.confidence, c.significance,
                           cvar=forecasts.get(f"{method}_cvar"))
            rows[(method, feed)] = res.summary
    out = pd.DataFrame(rows).T
    out.index.names = ["method", "pnl"]
    return out
