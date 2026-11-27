"""VaR limit utilization, alert levels and model warnings.

Rows use the same columns as the position sizing alerts (severity, kind, name, value,
limit, message) so one daily report can stack them. Severities, most serious first:

``breach``  VaR (or CVaR) at or above its limit.
``alert``   utilization at or above an alert level (named in ``VaRLimits.alert_levels``;
            the highest level crossed is reported).
``watch``   the same, for the lower default level.
``warning`` the two VaR methods disagree by more than ``max_method_gap``, or the
            historical sample is short.
``info``    something worth knowing, such as names filled with the factor proxy.
"""

from __future__ import annotations

import pandas as pd

from .config import VaRConfig, VaRLimits

BREACH_COLUMNS = ["severity", "kind", "name", "value", "limit", "message"]
SEVERITY_ORDER = {"breach": 0, "alert": 1, "watch": 2, "warning": 3, "info": 4}


def _row(severity, kind, name, value, limit, message) -> dict:
    return {"severity": severity, "kind": kind, "name": name, "value": value,
            "limit": limit, "message": message}


def limit_var(summary: dict, limits: VaRLimits) -> float:
    """The VaR number held against the limit, per ``limits.limit_basis``."""
    p, h = summary["parametric_var_1d"], summary["historical_var_1d"]
    if limits.limit_basis == "parametric":
        return p
    if limits.limit_basis == "historical":
        return h
    return max(p, h)


def _utilization_row(kind: str, value: float, limit: float, levels: dict, label: str) -> dict | None:
    u = value / limit
    if u >= 1.0:
        return _row("breach", kind, label, value, limit,
                    f"{label} {value:.2%} of NAV is {u:.0%} of its {limit:.2%} limit")
    crossed = [(lvl, name) for name, lvl in levels.items() if u >= lvl]
    if crossed:
        lvl, name = max(crossed)
        return _row(name, kind, label, value, limit,
                    f"{label} {value:.2%} of NAV is {u:.0%} of its {limit:.2%} limit "
                    f"({name} level {lvl:.0%})")
    return None


def check_var_limits(summary: dict, config: VaRConfig) -> list[dict]:
    """Limit, alert and model-health rows for one VaR summary."""
    lim = config.limits
    rows: list[dict] = []
    value = limit_var(summary, lim)
    r = _utilization_row("var_limit", value, lim.var_limit, lim.alert_levels,
                         f"1-day {config.confidence:.0%} VaR ({lim.limit_basis})")
    if r:
        rows.append(r)
    if lim.cvar_limit is not None:
        r = _utilization_row("cvar_limit", summary["historical_cvar_1d"], lim.cvar_limit,
                             lim.alert_levels, f"1-day {config.confidence:.0%} CVaR (historical)")
        if r:
            rows.append(r)

    gap = summary.get("method_gap")
    if lim.max_method_gap is not None and gap is not None and abs(gap) > lim.max_method_gap:
        direction = "above" if gap > 0 else "below"
        rows.append(_row("warning", "method_gap", "historical vs parametric", gap,
                         lim.max_method_gap,
                         f"historical VaR is {abs(gap):.0%} {direction} parametric VaR: check "
                         f"tails and skew, or a stale covariance"))
    n = summary.get("n_scenarios", 0)
    if n < config.min_history_days:
        rows.append(_row("warning", "short_history", "historical scenarios", n,
                         config.min_history_days,
                         f"only {n} historical scenarios (minimum {config.min_history_days})"))
    share = summary.get("proxied_share", 0.0)
    if share > 0:
        rows.append(_row("info", "proxied_returns", "historical scenarios", share, None,
                         f"{share:.1%} of gross-weighted position-days use the factor proxy "
                         f"(name had no return that day)"))
    return rows


def breach_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=BREACH_COLUMNS)
    if len(df):
        df = df.sort_values("severity", key=lambda s: s.map(SEVERITY_ORDER), kind="stable")
    return df.reset_index(drop=True)
