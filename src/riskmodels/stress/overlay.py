"""Liquidity overlay: repeat key scenarios with stressed volumes and spreads, and cost the
reduction of the book to a target.

For each key scenario:

1. Apply the scenario's factor shock to the book. Each name's value moves by its factor
   return, and NAV moves by the scenario P&L, giving the post-shock weights.
2. Measure liquidity of the post-shock book with ADV x ``volume_multiplier``, spreads x
   ``spread_multiplier`` and volatility x ``vol_multiplier`` (all from the scenario).
3. Cut gross by the target (30% by default) with the liquidity model's reduction method
   and cost it with the stressed inputs.
4. Report the scenario loss, the cost of the cut and the total, next to the same cut in
   normal conditions on today's book.

The cost is a fraction of today's NAV so it can be added to the scenario loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..factor_model.risk import RiskModel
from ..liquidity.config import LiquidityConfig
from ..liquidity.inputs import resolve_inputs
from ..liquidity.liquidation import liquidation_profile, reduce_gross
from ..var.limits import _row, breach_frame
from .engine import StressReport


@dataclass
class OverlayReport:
    """Stressed-liquidity versions of the key scenarios.

    Attributes
    ----------
    summary : DataFrame
        One row per scenario: scenario P&L, liquidatable share in normal and stressed
        conditions, days and cost to cut gross by the target in normal and stressed
        conditions, and the total P&L including the stressed cost of the cut.
    reductions : dict
        Scenario name -> (normal, stressed) :class:`GrossReduction` with per-name trades.
    breaches : DataFrame
    """

    target: float
    method: str
    summary: pd.DataFrame
    reductions: dict
    breaches: pd.DataFrame


def liquidity_overlay(
    report: StressReport,
    adv: pd.Series,
    nav: float,
    risk_model: RiskModel | None = None,
    market_cap: pd.Series | None = None,
    spread_bps: pd.Series | None = None,
    liquidity_config: LiquidityConfig | None = None,
    scenarios: list[str] | None = None,
    target: float | None = None,
    method: str | None = None,
) -> OverlayReport:
    """Overlay stressed liquidity on ``report``'s key scenarios (or the ``scenarios`` named)."""
    lc = liquidity_config or LiquidityConfig()
    target = lc.reduction_target if target is None else target
    method = lc.reduction_method if method is None else method
    names = scenarios if scenarios is not None else [
        n for n, r in report.results.items() if r.scenario.key]
    if not names:
        raise ValueError("no key scenarios in the report; pass scenarios=[...]")
    w = report.weights
    inputs = resolve_inputs(w.index, adv, lc, market_cap, spread_bps, None, risk_model)
    horizon = min(lc.limits.min_liquidatable) if lc.limits.min_liquidatable else 5.0

    normal = reduce_gross(w, inputs, nav, lc, target, method)
    normal_profile = liquidation_profile(w, inputs["adv"], nav, lc, days=(horizon,))
    rows, reductions, breaches = [], {}, []
    for name in names:
        res = report.results[name]
        sc = res.scenario
        pnl = res.summary["pnl"]
        nav_after = nav * (1 + pnl)
        if nav_after <= 0:
            raise ValueError(f"{name}: the scenario wipes out NAV; the overlay is undefined")
        w_after = w * (1 + res.by_position["factor_return"]) / (1 + pnl)
        stressed = reduce_gross(w_after, inputs, nav_after, lc, target, method,
                                volume_multiplier=sc.volume_multiplier,
                                spread_multiplier=sc.spread_multiplier,
                                vol_multiplier=sc.vol_multiplier)
        prof = liquidation_profile(w_after, inputs["adv"], nav_after, lc, days=(horizon,),
                                   volume_multiplier=sc.volume_multiplier)
        cost_today_nav = stressed.summary["cost"] * nav_after / nav
        reductions[name] = (normal, stressed)
        row = {
            "scenario": name,
            "volume_multiplier": sc.volume_multiplier,
            "spread_multiplier": sc.spread_multiplier,
            "scenario_pnl": pnl,
            f"liquidatable_{horizon:g}d_normal": float(normal_profile["share_of_gross"].iloc[0]),
            f"liquidatable_{horizon:g}d_stressed": float(prof["share_of_gross"].iloc[0]),
            "cut_days_normal": normal.summary["days_to_complete"],
            "cut_days_stressed": stressed.summary["days_to_complete"],
            "cut_cost_normal": normal.summary["cost"],
            "cut_cost_stressed": cost_today_nav,
            "cut_cost_bps_stressed": stressed.summary["cost_bps_of_traded"],
            "achieved_cut_stressed": stressed.summary["achieved_cut"],
            "total_pnl": pnl - cost_today_nav,
        }
        row["total_loss"] = -row["total_pnl"]
        rows.append(row)

        lim = report.config.limits
        if lim.max_loss is not None:
            u = row["total_loss"] / lim.max_loss
            sev = "breach" if u >= 1 else None
            if sev is None:
                crossed = [(lvl, n) for n, lvl in lim.alert_levels.items() if u >= lvl]
                sev = max(crossed)[1] if crossed else None
            if sev:
                breaches.append(_row(sev, "stress_loss_with_liquidity", name, row["total_loss"],
                                     lim.max_loss,
                                     f"{name} with stressed liquidity: loss plus the cost of "
                                     f"cutting gross {target:.0%} is {row['total_loss']:.2%} of NAV "
                                     f"({u:.0%} of the {lim.max_loss:.0%} limit)"))
        for days, need in lc.limits.min_liquidatable.items():
            if days == horizon and lc.limits.basis == "gross":
                share = row[f"liquidatable_{horizon:g}d_stressed"]
                if share < need:
                    breaches.append(_row("warning", "stressed_liquidity", name, share, need,
                                         f"{name}: only {share:.1%} of gross liquidatable in "
                                         f"{days:g} days at {sc.volume_multiplier:.0%} of normal "
                                         f"volume (normal-conditions minimum {need:.0%})"))
        if stressed.summary["achieved_cut"] < target - 1e-9:
            breaches.append(_row("warning", "cut_short", name, stressed.summary["achieved_cut"],
                                 target, f"{name}: the {target:.0%} cut reaches only "
                                         f"{stressed.summary['achieved_cut']:.1%} (names without ADV)"))
    summary = pd.DataFrame(rows).set_index("scenario")
    summary["total_pnl_amount"] = summary["total_pnl"] * nav
    summary["cut_cost_stressed_amount"] = summary["cut_cost_stressed"] * nav
    summary = summary.sort_values("total_pnl", kind="stable")
    return OverlayReport(target=target, method=method, summary=summary,
                         reductions=reductions, breaches=breach_frame(breaches))
