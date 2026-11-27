"""VaR / CVaR for a long/short book: parametric (primary) and historical (cross-check).

Example
-------
>>> fit = FactorModel(FactorModelConfig.daily()).fit(market_data)
>>> sized = PositionSizer().size(signals, fit.risk_model(), nav=250e6)
>>> report = VaRModel(VaRConfig()).run(sized, fit, market_data, nav=250e6)
>>> report.summary["parametric_var_1d"], report.breaches
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.data import MarketData
from ..factor_model.model import FactorModelResult
from ..factor_model.risk import RiskModel
from ..position_sizing.analytics import sector_from_industries
from .config import VaRConfig
from .historical import HistoricalScenarios, build_scenarios, scenario_weighting
from .limits import breach_frame, check_var_limits, limit_var
from .parametric import ParametricVaR, parametric_var, standalone_parametric_var
from .tail import TailEstimate, split_by_tail, tail_estimate


@dataclass
class VaRReport:
    """Everything one VaR run produced. Numbers are fractions of NAV, losses positive.

    Attributes
    ----------
    summary : dict
        Headline numbers: 1-day and scaled-horizon VaR and CVaR for both methods, the gap
        between methods, limit utilization, scenario counts. Currency amounts are added
        (``*_amount``) when ``nav`` is given.
    horizons : DataFrame
        VaR and CVaR by method for the 1-day and scaled horizons.
    by_position, by_sector, by_factor, by_side : DataFrame
        Component VaR (and CVaR) for each method.
    breaches : DataFrame
        Limit, alert and model-health rows.
    parametric : ParametricVaR
    historical : TailEstimate
    scenarios : HistoricalScenarios
    pnl : Series
        Historical scenario P&L of the book after weighting scale (date).
    probabilities : Series
        Probability of each scenario.
    """

    as_of: pd.Timestamp
    config: VaRConfig
    nav: float | None
    weights: pd.Series
    summary: dict
    horizons: pd.DataFrame
    by_position: pd.DataFrame
    by_sector: pd.DataFrame
    by_factor: pd.DataFrame
    by_side: pd.DataFrame
    breaches: pd.DataFrame
    parametric: ParametricVaR = field(repr=False)
    historical: TailEstimate = field(repr=False)
    scenarios: HistoricalScenarios = field(repr=False)
    pnl: pd.Series = field(repr=False)
    probabilities: pd.Series = field(repr=False)

    @property
    def ok(self) -> bool:
        """True when no limit is breached."""
        return not (self.breaches["severity"] == "breach").any()

    def to_amount(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Any table above in currency (needs ``nav``)."""
        if self.nav is None:
            raise ValueError("no nav was given")
        num = frame.select_dtypes("number")
        return frame.assign(**{c: num[c] * self.nav for c in num.columns
                               if c not in ("exposure",)})


def _unpack(portfolio, sector):
    """Weights and sector from a Series or a position sizing result."""
    if hasattr(portfolio, "weights") and hasattr(portfolio, "sector"):
        w = portfolio.weights
        sector = portfolio.sector if sector is None else sector
    else:
        w = portfolio
    w = pd.Series(w, dtype=float)
    return w[w != 0], sector


class VaRModel:
    def __init__(self, config: VaRConfig | None = None) -> None:
        self.config = config or VaRConfig()

    def with_config(self, **overrides) -> "VaRModel":
        return VaRModel(self.config.with_overrides(**overrides))

    def run(
        self,
        portfolio,
        fit: FactorModelResult,
        data: MarketData,
        as_of=None,
        nav: float | None = None,
        sector: pd.Series | None = None,
        risk_model: RiskModel | None = None,
        decompose: bool = True,
    ) -> VaRReport:
        """VaR / CVaR of ``portfolio`` on ``as_of``.

        Parameters
        ----------
        portfolio : Series or SizingResult
            Weights as fractions of NAV (long positive, short negative), or the output of
            :meth:`PositionSizer.size`, whose weights and sectors are used.
        fit, data : the fitted factor model and the data it was fitted on.
        as_of : date, optional. Defaults to the last exposure date.
        nav : float, optional. Adds currency amounts to the summary.
        sector : Series, optional. Name -> sector; defaults to the factor model industry.
        risk_model : RiskModel, optional. Reuse a forecast already built for ``as_of``.
        decompose : bool. False skips the component tables (faster, for backtests).
        """
        c = self.config
        alpha = c.tail_probability
        w, sector = _unpack(portfolio, sector)
        if w.empty:
            raise ValueError("portfolio has no non-zero weights")
        rm = risk_model if risk_model is not None else fit.risk_model(as_of)

        par = parametric_var(rm, w, c)
        scen = build_scenarios(fit, data, w.index, rm, c.history_days)
        raw_pnl = scen.pnl(w)
        probs, scale = scenario_weighting(raw_pnl, c)
        pnl = raw_pnl * scale
        hist = tail_estimate(pnl, alpha, probs, c.component_var_band)

        long_w, short_w = w[w > 0], w[w < 0]
        side_tails = {}
        for side, ws in (("long", long_w), ("short", short_w)):
            if ws.empty:
                side_tails[side] = None
                continue
            sp = scen.pnl(ws)
            sprob, sscale = scenario_weighting(sp, c)
            side_tails[side] = tail_estimate(sp * sscale, alpha, sprob, c.component_var_band)

        h = c.horizon_scale
        gross = float(w.abs().sum())
        proxied_share = float((scen.proxied[w.index].mul(w.abs(), axis=1).sum(axis=1)).mean()
                              / gross) if gross > 0 else 0.0
        summary = {
            "as_of": rm.as_of,
            "confidence": c.confidence,
            "scaled_horizon_days": c.scaled_horizon_days,
            "predicted_vol_1d": par.sigma,
            "parametric_var_1d": par.var,
            "parametric_cvar_1d": par.cvar,
            "historical_var_1d": hist.var,
            "historical_cvar_1d": hist.cvar,
            f"parametric_var_{c.scaled_horizon_days}d": par.var * h,
            f"parametric_cvar_{c.scaled_horizon_days}d": par.cvar * h,
            f"historical_var_{c.scaled_horizon_days}d": hist.var * h,
            f"historical_cvar_{c.scaled_horizon_days}d": hist.cvar * h,
            "method_gap": (hist.var - par.var) / par.var if par.var > 0 else np.nan,
            "hs_weighting": c.hs_weighting,
            "n_scenarios": hist.n_scenarios,
            "effective_scenarios": hist.effective_scenarios,
            "history_start": scen.dates[0],
            "history_end": scen.dates[-1],
            "proxied_share": proxied_share,
            "gross": gross,
            "net": float(w.sum()),
        }
        lim = c.limits
        summary["limit_basis"] = lim.limit_basis
        summary["var_limit"] = lim.var_limit
        summary["var_utilization"] = limit_var(summary, lim) / lim.var_limit
        if lim.cvar_limit is not None:
            summary["cvar_limit"] = lim.cvar_limit
            summary["cvar_utilization"] = hist.cvar / lim.cvar_limit
        if nav is not None:
            for key in [k for k in summary if ("_var_" in k or "_cvar_" in k or k == "predicted_vol_1d")]:
                summary[f"{key}_amount"] = summary[key] * nav

        horizons = pd.DataFrame(
            {"1d": [par.var, par.cvar, hist.var, hist.cvar]},
            index=pd.MultiIndex.from_tuples(
                [("parametric", "var"), ("parametric", "cvar"),
                 ("historical", "var"), ("historical", "cvar")], names=["method", "measure"]),
        )
        horizons[f"{c.scaled_horizon_days}d"] = horizons["1d"] * h

        if decompose:
            sec = (sector if sector is not None else sector_from_industries(rm)).reindex(w.index)
            sec = sec.fillna("unknown")
            by_position, by_sector, by_factor, by_side = _decompose(
                w, sec, par, hist, scen, scale, rm, c, side_tails)
        else:
            by_position = by_sector = by_factor = by_side = pd.DataFrame()

        breaches = breach_frame(check_var_limits(summary, c))
        return VaRReport(
            as_of=rm.as_of, config=c, nav=nav, weights=w, summary=summary, horizons=horizons,
            by_position=by_position, by_sector=by_sector, by_factor=by_factor, by_side=by_side,
            breaches=breaches, parametric=par, historical=hist, scenarios=scen, pnl=pnl,
            probabilities=probs,
        )


def _decompose(w, sec, par, hist, scen, scale, rm, c, side_tails):
    side = pd.Series(np.where(w > 0, "long", "short"), index=w.index)

    pos_hist = split_by_tail(scen.position_pnl(w).mul(scale, axis=0), hist)
    by_position = pd.DataFrame({
        "weight": w,
        "side": side,
        "sector": sec,
        "parametric_var": par.by_position["var"],
        "parametric_factor_var": par.by_position["factor_var"],
        "parametric_specific_var": par.by_position["specific_var"],
        "marginal_var": par.by_position["marginal_var"],
        "historical_var": pos_hist["var"],
        "historical_cvar": pos_hist["cvar"],
    })
    by_position["pct_of_parametric_var"] = by_position["parametric_var"] / par.var

    num = ["parametric_var", "historical_var", "historical_cvar"]
    g = by_position.groupby("sector")
    by_sector = pd.DataFrame({
        "long": g["weight"].apply(lambda s: s.clip(lower=0).sum()),
        "short": g["weight"].apply(lambda s: s.clip(upper=0).sum()),
    })
    by_sector["net"] = by_sector["long"] + by_sector["short"]
    by_sector["gross"] = by_sector["long"] - by_sector["short"]
    by_sector = by_sector.join(g[num].sum())
    by_sector["pct_of_parametric_var"] = by_sector["parametric_var"] / par.var

    fac_hist = split_by_tail(scen.factor_pnl(w).mul(scale, axis=0), hist)
    by_factor = par.by_factor[["group", "exposure"]].copy()
    by_factor["parametric_var"] = par.by_factor["var"]
    by_factor = by_factor.join(fac_hist.rename(columns={"var": "historical_var",
                                                        "cvar": "historical_cvar"}))
    by_factor["pct_of_parametric_var"] = by_factor["parametric_var"] / par.var

    gs = by_position.groupby("side")
    rows = {}
    for s in ("long", "short"):
        ws = w[side == s]
        t = side_tails[s]
        rows[s] = {
            "weight": float(ws.sum()),
            "component_parametric_var": float(gs["parametric_var"].sum().get(s, 0.0)),
            "standalone_parametric_var": standalone_parametric_var(rm, ws, c),
            "component_historical_var": float(gs["historical_var"].sum().get(s, 0.0)),
            "standalone_historical_var": t.var if t is not None else 0.0,
            "component_historical_cvar": float(gs["historical_cvar"].sum().get(s, 0.0)),
            "standalone_historical_cvar": t.cvar if t is not None else 0.0,
        }
    by_side = pd.DataFrame(rows).T
    total = {
        "weight": float(w.sum()),
        "component_parametric_var": par.var,
        "standalone_parametric_var": par.var,
        "component_historical_var": hist.var,
        "standalone_historical_var": hist.var,
        "component_historical_cvar": hist.cvar,
        "standalone_historical_cvar": hist.cvar,
    }
    by_side.loc["total"] = total
    by_side["diversification_parametric"] = np.nan
    by_side.loc["total", "diversification_parametric"] = (
        by_side.loc[["long", "short"], "standalone_parametric_var"].sum() - par.var)
    by_side["diversification_historical"] = np.nan
    by_side.loc["total", "diversification_historical"] = (
        by_side.loc[["long", "short"], "standalone_historical_var"].sum() - hist.var)
    return by_position, by_sector, by_factor, by_side
