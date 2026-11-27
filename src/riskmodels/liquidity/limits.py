"""Liquidity limit checks.

Rows have the same columns and severities as the VaR alerts (severity, kind, name, value,
limit, message), so the daily report can stack liquidity, VaR and sizing alerts together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..var.limits import _row
from .config import LiquidityConfig
from .liquidation import NO_ADV, daily_capacity, liquidatable


def _level(utilization: float, levels: dict) -> str | None:
    # at the limit is allowed; only above it is a breach
    if utilization > 1.0 + 1e-9:
        return "breach"
    crossed = [(lvl, name) for name, lvl in levels.items() if utilization >= lvl]
    return max(crossed)[1] if crossed else None


def check_liquidity_limits(
    positions: pd.DataFrame,
    profile: pd.DataFrame,
    summary: dict,
    config: LiquidityConfig,
) -> list[dict]:
    lim = config.limits
    rows: list[dict] = []

    for days, need in lim.min_liquidatable.items():
        if days in profile.index:
            share = float(profile.loc[days, "share_of_gross" if lim.basis == "gross" else "share_of_nav"])
        else:  # a horizon not in the profile table: compute directly
            w = positions["weight"]
            cap = daily_capacity(positions["adv"], summary["nav"], config.participation)
            q = float(liquidatable(w, cap, days).sum())
            share = q / summary["gross"] if lim.basis == "gross" else q
        if lim.basis == "gross":
            util = (1 - share) / (1 - need) if need < 1 else (np.inf if share < 1 else 0.0)
        else:
            util = need / share if share > 0 else np.inf
        sev = _level(util, lim.alert_levels)
        if sev:
            basis = "gross" if lim.basis == "gross" else "NAV"
            rows.append(_row(sev, "liquidatable", f"{days:g}d liquidatable", share, need,
                             f"{share:.1%} of {basis} liquidatable in {days:g} days "
                             f"(minimum {need:.0%}, {util:.0%} utilization)"))

    if lim.max_position_adv is not None:
        pct = positions["pct_adv"].dropna()
        for name, v in pct[pct >= min(lim.alert_levels.values(), default=1.0) * lim.max_position_adv].items():
            sev = _level(v / lim.max_position_adv, lim.alert_levels)
            if sev:
                rows.append(_row(sev, "position_adv", name, float(v), lim.max_position_adv,
                                 f"{name} is {v:.2f}x ADV (limit {lim.max_position_adv:.2f}x)"))

    if lim.max_slow_share is not None:
        slow_label = config.bucket_labels[-1]
        slow = float(positions.loc[positions["bucket"] == slow_label, "weight"].abs().sum())
        share = slow / summary["gross"] if summary["gross"] > 0 else 0.0
        sev = _level(share / lim.max_slow_share, lim.alert_levels)
        if sev:
            rows.append(_row(sev, "slow_bucket", slow_label, share, lim.max_slow_share,
                             f"{share:.1%} of gross takes over {config.bucket_edges[-1]:g} days "
                             f"to exit (limit {lim.max_slow_share:.0%})"))

    no_adv = positions[positions["bucket"] == NO_ADV]
    if len(no_adv):
        g = float(no_adv["weight"].abs().sum())
        rows.append(_row("warning", "no_adv", f"{len(no_adv)} names", g, None,
                         f"{len(no_adv)} positions ({g:.1%} of NAV) have no ADV and count as "
                         f"not liquidatable"))
    defaults = positions["spread_source"].eq("default").sum()
    if defaults:
        rows.append(_row("info", "spread_default", f"{defaults} names", int(defaults), None,
                         f"{defaults} names use the default {config.default_spread_bps:g}bp spread"))
    return rows
