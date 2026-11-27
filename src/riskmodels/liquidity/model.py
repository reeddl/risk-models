"""Liquidity risk for a long/short book: days to liquidate, buckets, limits, cost to cut.

Example
-------
>>> adv = adv_from_data(market_data)
>>> rep = LiquidityModel().run(sized, adv, nav=250e6,
...                            market_cap=market_cap_from_data(market_data), risk_model=rm)
>>> rep.buckets, rep.profile, rep.reduction.summary, rep.breaches
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..factor_model.risk import RiskModel
from ..var.limits import breach_frame
from ..var.model import _unpack
from .config import LiquidityConfig
from .inputs import resolve_inputs
from .liquidation import (
    GrossReduction,
    bucket,
    bucket_table,
    days_to_liquidate,
    days_to_liquidate_share,
    liquidation_profile,
    reduce_gross,
    trade_cost,
)
from .limits import check_liquidity_limits


@dataclass
class LiquidityReport:
    """Everything one liquidity run produced. Weights and costs are fractions of NAV.

    Attributes
    ----------
    summary : dict
        Gross, net, liquidatable shares at each profile horizon, days to exit 50%, 90% and
        100% of gross, gross-weighted average days to liquidate, cost of a full exit and of
        the gross reduction.
    positions : DataFrame
        Per name: weight, value, ADV, position as a multiple of ADV, days to liquidate,
        bucket, spread, daily vol, cost of a full exit.
    buckets : DataFrame
        Long, short, gross and share of gross per bucket.
    profile : DataFrame
        Share of gross (and of NAV) exitable within each horizon.
    reduction : GrossReduction
        Trades, time and cost to cut gross by ``config.reduction_target``.
    breaches : DataFrame
        Limit, alert and data-quality rows.
    """

    as_of: pd.Timestamp | None
    config: LiquidityConfig
    nav: float
    summary: dict
    positions: pd.DataFrame
    buckets: pd.DataFrame
    profile: pd.DataFrame
    reduction: GrossReduction
    breaches: pd.DataFrame

    @property
    def ok(self) -> bool:
        return not (self.breaches["severity"] == "breach").any()


class LiquidityModel:
    def __init__(self, config: LiquidityConfig | None = None) -> None:
        self.config = config or LiquidityConfig()

    def with_config(self, **overrides) -> "LiquidityModel":
        return LiquidityModel(self.config.with_overrides(**overrides))

    def run(
        self,
        portfolio,
        adv: pd.Series,
        nav: float,
        market_cap: pd.Series | None = None,
        spread_bps: pd.Series | None = None,
        daily_vol: pd.Series | None = None,
        risk_model: RiskModel | None = None,
        sector: pd.Series | None = None,
        as_of=None,
        volume_multiplier: float = 1.0,
        spread_multiplier: float = 1.0,
        vol_multiplier: float = 1.0,
    ) -> LiquidityReport:
        """Liquidity of ``portfolio``.

        Parameters
        ----------
        portfolio : Series or SizingResult
            Weights as fractions of NAV, or the position sizer's output.
        adv : Series
            Average daily traded value per name, same currency as ``nav``
            (see :func:`adv_from_data`).
        nav : float
        market_cap, spread_bps : Series, optional
            Quoted spreads in basis points; names without one get a spread from the market
            cap schedule, and failing that ``config.default_spread_bps``.
        daily_vol, risk_model : optional
            Daily volatility per name for the impact cost; taken from the risk model's total
            risk when not given.
        sector : Series, optional. Adds the sector to the position table.
        volume_multiplier, spread_multiplier, vol_multiplier : float
            Stress multipliers (ADV x 0.5, spreads x 2, ...). 1.0 is normal conditions.
        """
        c = self.config
        w, sector = _unpack(portfolio, sector)
        if w.empty:
            raise ValueError("portfolio has no non-zero weights")
        if not nav > 0:
            raise ValueError("nav must be positive")
        inputs = resolve_inputs(w.index, adv, c, market_cap, spread_bps, daily_vol, risk_model)

        dtl = days_to_liquidate(w, inputs["adv"], nav, c.participation, volume_multiplier)
        bk = bucket(dtl, c)
        exit_cost = trade_cost(w, inputs, nav, c, volume_multiplier, spread_multiplier, vol_multiplier)
        positions = pd.DataFrame({
            "weight": w,
            "value": w * nav,
            "adv": inputs["adv"] * volume_multiplier,
            "pct_adv": (w.abs() * nav / (inputs["adv"] * volume_multiplier)).where(inputs["adv"] > 0),
            "days_to_liquidate": dtl,
            "bucket": bk,
            "spread_bps": inputs["spread_bps"] * spread_multiplier,
            "spread_source": inputs["spread_source"],
            "daily_vol": inputs["daily_vol"] * vol_multiplier,
            "exit_cost_rate": exit_cost["cost_rate"],
            "exit_cost": exit_cost["cost"],
        })
        if sector is not None:
            positions.insert(0, "sector", sector.reindex(w.index).fillna("unknown"))
        positions = positions.sort_values("days_to_liquidate", ascending=False)

        profile = liquidation_profile(w, inputs["adv"], nav, c, volume_multiplier=volume_multiplier)
        buckets = bucket_table(w, bk, nav, c)
        reduction = reduce_gross(w, inputs, nav, c, volume_multiplier=volume_multiplier,
                                 spread_multiplier=spread_multiplier, vol_multiplier=vol_multiplier)

        gross = float(w.abs().sum())
        finite = np.isfinite(dtl)
        summary = {
            "as_of": as_of if as_of is None else pd.Timestamp(as_of),
            "nav": nav,
            "gross": gross,
            "net": float(w.sum()),
            "long": float(w[w > 0].sum()),
            "short": float(w[w < 0].sum()),
            "n_positions": int(len(w)),
            "n_no_adv": int((~finite).sum()),
            "participation": c.participation,
            "avg_days_to_liquidate": float((w.abs()[finite] * dtl[finite]).sum() / w.abs()[finite].sum())
            if finite.any() else np.inf,
            "max_days_to_liquidate": float(dtl.max()),
        }
        for share in (0.5, 0.9, 1.0):
            summary[f"days_to_exit_{int(share * 100)}pct"] = float(days_to_liquidate_share(
                w, inputs["adv"], nav, share, c.participation, volume_multiplier))
        for t, r in profile.iterrows():
            summary[f"liquidatable_{t:g}d_gross"] = float(r["share_of_gross"])
            summary[f"liquidatable_{t:g}d_nav"] = float(r["share_of_nav"])
        summary["full_exit_cost"] = float(exit_cost["cost"].sum(min_count=1))
        summary["reduction_cut"] = reduction.summary["target_cut"]
        summary["reduction_days"] = reduction.summary["days_to_complete"]
        summary["reduction_cost"] = reduction.summary["cost"]
        summary["volume_multiplier"] = volume_multiplier
        summary["spread_multiplier"] = spread_multiplier

        breaches = breach_frame(check_liquidity_limits(positions, profile, summary, c))
        return LiquidityReport(as_of=summary["as_of"], config=c, nav=nav, summary=summary,
                               positions=positions, buckets=buckets, profile=profile,
                               reduction=reduction, breaches=breaches)
