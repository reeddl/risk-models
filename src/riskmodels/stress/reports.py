"""Report functions for each cadence, for a scheduler to call.

- :func:`daily_liquidity_report`: every day. Days to liquidate, buckets, limits and the cost
  of cutting gross, stacked with the VaR and sizing alerts if you pass those reports.
- :func:`periodic_stress_report`: weekly or monthly. The full scenario library, the
  liquidity overlay on the key scenarios, the short squeeze and reverse stress tests.
- :func:`ad_hoc_stress_report`: before a large trade or a macro event. The book before and
  after the trade (or just the book, for an event) under the chosen scenarios.

Nothing here schedules itself. Each function returns a :class:`ScheduledReport` whose
``breaches`` table uses the same columns as the VaR and sizing alerts, plus ``source``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.data import MarketData
from ..factor_model.risk import RiskModel
from ..liquidity.config import LiquidityConfig
from ..liquidity.inputs import adv_from_data, market_cap_from_data
from ..liquidity.model import LiquidityModel
from ..var.limits import BREACH_COLUMNS, SEVERITY_ORDER, _row
from ..var.model import _unpack
from .config import StressConfig
from .engine import StressTester
from .overlay import liquidity_overlay
from .reverse import reverse_stress
from .scenario import ScenarioLibrary, default_library
from .squeeze import SqueezeConfig, short_squeeze


@dataclass
class ScheduledReport:
    """A report for one cadence.

    Attributes
    ----------
    cadence : ``"daily_liquidity"``, ``"weekly_stress"``, ``"monthly_stress"`` or ``"ad_hoc"``
    as_of : Timestamp or None
    headline : dict
        The few numbers that go at the top of the report.
    tables : dict
        Name -> DataFrame, in the order they should be shown.
    breaches : DataFrame
        Every alert from every model run, with a ``source`` column, most severe first.
    details : dict
        The underlying model reports, for drill-down.
    """

    cadence: str
    as_of: pd.Timestamp | None
    headline: dict
    tables: dict
    breaches: pd.DataFrame
    details: dict = field(default_factory=dict, repr=False)

    @property
    def ok(self) -> bool:
        return not (self.breaches["severity"] == "breach").any()

    def text(self, max_rows: int = 10) -> str:
        """A plain-text rendering (for a log, an email body or a terminal)."""
        lines = [f"{self.cadence} report" + (f" as of {self.as_of.date()}" if self.as_of is not None else "")]
        for k, v in self.headline.items():
            lines.append(f"  {k}: {_fmt(v)}")
        if len(self.breaches):
            lines.append("")
            lines.append("Alerts:")
            for _, r in self.breaches.head(max_rows * 2).iterrows():
                lines.append(f"  [{r['severity']}] {r['source']}: {r['message']}")
        for name, t in self.tables.items():
            lines.append("")
            lines.append(f"{name}:")
            lines.append(t.head(max_rows).to_string())
        return "\n".join(lines)


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return str(v)
        return f"{v:.4g}"
    return str(v)


def stack_breaches(**frames: pd.DataFrame) -> pd.DataFrame:
    """Stack alert tables from several models, adding a ``source`` column."""
    parts = []
    for source, df in frames.items():
        if df is None or not len(df):
            continue
        parts.append(df[BREACH_COLUMNS].assign(source=source))
    if not parts:
        return pd.DataFrame(columns=["source", *BREACH_COLUMNS])
    out = pd.concat(parts, ignore_index=True)[["source", *BREACH_COLUMNS]]
    out = out.sort_values("severity", key=lambda s: s.map(SEVERITY_ORDER).fillna(9), kind="stable")
    return out.reset_index(drop=True)


def _inputs(data, adv, market_cap, as_of, lc):
    if adv is None and data is not None:
        adv = adv_from_data(data, as_of, lc.adv_window, lc.adv_method)
    if market_cap is None and data is not None:
        market_cap = market_cap_from_data(data, as_of)
    return adv, market_cap


def daily_liquidity_report(
    portfolio,
    nav: float,
    data: MarketData | None = None,
    adv: pd.Series | None = None,
    market_cap: pd.Series | None = None,
    spread_bps: pd.Series | None = None,
    risk_model: RiskModel | None = None,
    as_of=None,
    config: LiquidityConfig | None = None,
    var_report=None,
    sizing_result=None,
) -> ScheduledReport:
    """Daily liquidity report. ADV and market cap come from ``data`` unless passed in.

    Pass ``var_report`` (a VaR report) and ``sizing_result`` (a sizing or monitor result)
    to put their alerts in the same table.
    """
    lc = config or LiquidityConfig()
    adv, market_cap = _inputs(data, adv, market_cap, as_of, lc)
    if adv is None:
        raise ValueError("pass data or adv")
    if as_of is None and risk_model is not None:
        as_of = risk_model.as_of
    rep = LiquidityModel(lc).run(portfolio, adv, nav, market_cap=market_cap, spread_bps=spread_bps,
                                 risk_model=risk_model, as_of=as_of)
    s = rep.summary
    headline = {
        "gross": s["gross"], "net": s["net"],
        **{k: v for k, v in s.items() if k.startswith("liquidatable_") and k.endswith("_gross")},
        "days_to_exit_90pct": s["days_to_exit_90pct"],
        "max_days_to_liquidate": s["max_days_to_liquidate"],
        f"cut_gross_{lc.reduction_target:.0%}_days": s["reduction_days"],
        f"cut_gross_{lc.reduction_target:.0%}_cost": s["reduction_cost"],
        "breaches": int((rep.breaches["severity"] == "breach").sum()),
    }
    breaches = stack_breaches(
        liquidity=rep.breaches,
        var=getattr(var_report, "breaches", None),
        sizing=getattr(sizing_result, "breaches", None),
    )
    tables = {
        "buckets": rep.buckets,
        "profile": rep.profile,
        "slowest positions": rep.positions.head(20),
    }
    if var_report is not None:
        tables["var"] = var_report.horizons
    details = {"liquidity": rep}
    if var_report is not None:
        details["var"] = var_report
    return ScheduledReport("daily_liquidity", None if as_of is None else pd.Timestamp(as_of),
                           headline, tables, breaches, details)


def periodic_stress_report(
    portfolio,
    risk_model: RiskModel,
    nav: float,
    data: MarketData | None = None,
    adv: pd.Series | None = None,
    market_cap: pd.Series | None = None,
    spread_bps: pd.Series | None = None,
    short_data: pd.DataFrame | None = None,
    library: ScenarioLibrary | None = None,
    stress_config: StressConfig | None = None,
    liquidity_config: LiquidityConfig | None = None,
    squeeze_config: SqueezeConfig | None = None,
    reverse_target_loss: float | None = None,
    reverse_horizon_days: int = 21,
    frequency: str = "weekly",
) -> ScheduledReport:
    """Weekly or monthly stress pack: library, liquidity overlay, squeeze, reverse stress.

    The overlay and squeeze run when ADV is available (from ``data`` or ``adv``). The
    reverse stress target defaults to the stress loss limit.
    """
    if frequency not in ("weekly", "monthly"):
        raise ValueError("frequency must be 'weekly' or 'monthly'")
    sc = stress_config or StressConfig()
    lc = liquidity_config or LiquidityConfig()
    lib = library if library is not None else default_library()
    adv, market_cap = _inputs(data, adv, market_cap, risk_model.as_of, lc)
    w, sector = _unpack(portfolio, None)

    tester = StressTester(sc, lib)
    stress = tester.run(portfolio, risk_model, nav=nav)
    overlay = squeeze = None
    if adv is not None and any(s.key for s in lib):
        overlay = liquidity_overlay(stress, adv, nav, risk_model, market_cap, spread_bps, lc)
    if adv is not None and (w < 0).any():
        squeeze = short_squeeze(w, adv, nav, short_data, squeeze_config, risk_model,
                                market_cap, spread_bps, lc, sc.limits)
    target = reverse_target_loss or sc.limits.max_loss or 0.10
    reverse = reverse_stress(w, risk_model, target, reverse_horizon_days, stress_report=stress,
                             trading_days_per_year=sc.trading_days_per_year)

    worst = stress.summary.iloc[0]
    headline = {
        "worst_scenario": stress.summary.index[0],
        "worst_loss": worst["loss"],
        "worst_loss_amount": worst["loss"] * nav,
        "scenarios_run": len(stress.summary),
        "historical_worst": stress.summary[stress.summary["kind"] == "historical"]["loss"].max(),
        "hypothetical_worst": stress.summary[stress.summary["kind"] == "hypothetical"]["loss"].max(),
    }
    tables = {"scenarios": stress.summary[[
        "kind", "horizon_days", "market_pnl", "industry_pnl", "style_pnl",
        "specific_stress", "pnl", "long_pnl", "short_pnl", "worst_sector", "worst_position"]]}
    if overlay is not None:
        headline["worst_with_liquidity"] = overlay.summary.index[0]
        headline["worst_loss_with_liquidity"] = overlay.summary["total_loss"].iloc[0]
        tables["liquidity overlay"] = overlay.summary
    if squeeze is not None:
        headline["squeeze_worst_loss"] = float(squeeze.summary["loss"].max())
        tables["short squeeze"] = squeeze.summary
    headline[f"reverse_{target:.0%}_loss_sigmas"] = reverse.most_likely["distance_with_specific"]
    tables["reverse stress: top factors"] = reverse.factors
    tables["reverse stress: top positions"] = reverse.positions
    if reverse.scenarios is not None:
        tables["reverse stress: scenario multiples"] = reverse.scenarios

    rev_rows = []
    d = reverse.most_likely["distance_with_specific"]
    if np.isfinite(d) and d < 2.0:
        rev_rows.append(_row("warning", "reverse_stress", f"{target:.0%} loss", d, 2.0,
                             f"a {target:.0%} loss over {reverse_horizon_days} days is only "
                             f"{d:.1f} standard deviations away"))
    breaches = stack_breaches(
        stress=stress.breaches,
        liquidity_overlay=None if overlay is None else overlay.breaches,
        squeeze=None if squeeze is None else squeeze.breaches,
        reverse=pd.DataFrame(rev_rows, columns=BREACH_COLUMNS),
    )
    details = {"stress": stress, "overlay": overlay, "squeeze": squeeze, "reverse": reverse}
    return ScheduledReport(f"{frequency}_stress", risk_model.as_of, headline, tables, breaches,
                           details)


def ad_hoc_stress_report(
    portfolio,
    risk_model: RiskModel,
    nav: float,
    trades: pd.Series | None = None,
    proposed: pd.Series | None = None,
    scenarios: list[str] | None = None,
    tags: list[str] | None = None,
    data: MarketData | None = None,
    adv: pd.Series | None = None,
    market_cap: pd.Series | None = None,
    spread_bps: pd.Series | None = None,
    short_data: pd.DataFrame | None = None,
    library: ScenarioLibrary | None = None,
    stress_config: StressConfig | None = None,
    liquidity_config: LiquidityConfig | None = None,
    squeeze_config: SqueezeConfig | None = None,
    reason: str = "",
) -> ScheduledReport:
    """Ad hoc stress before a large trade or a macro event.

    Pass ``trades`` (weight changes) or ``proposed`` (the full book after the trade) to
    compare before and after; pass neither to stress the current book for an event.
    ``scenarios`` (names) or ``tags`` (e.g. ``["rates"]`` before a central bank meeting)
    pick the scenarios; the default is the whole library.
    """
    sc = stress_config or StressConfig()
    lc = liquidity_config or LiquidityConfig()
    lib = library if library is not None else default_library()
    if scenarios is not None or tags is not None:
        lib = lib.select(names=scenarios, tags=tags)
    if not len(lib):
        raise ValueError("no scenarios selected")
    adv, market_cap = _inputs(data, adv, market_cap, risk_model.as_of, lc)
    before, _ = _unpack(portfolio, None)
    if trades is not None and proposed is not None:
        raise ValueError("pass trades or proposed, not both")
    after = None
    if trades is not None:
        idx = before.index.union(trades.index)
        after = before.reindex(idx, fill_value=0.0) + trades.astype(float).reindex(idx, fill_value=0.0)
    elif proposed is not None:
        after = pd.Series(proposed, dtype=float)
    if after is not None:
        after = after[after.abs() > 1e-12]

    tester = StressTester(sc, lib)
    s_before = tester.run(before, risk_model, nav=nav)
    tables, details, frames = {}, {"stress_before": s_before}, {"stress": s_before.breaches}
    headline = {"reason": reason or ("pre-trade" if after is not None else "event"),
                "scenarios_run": len(lib),
                "worst_scenario": s_before.summary.index[0],
                "worst_loss_before": s_before.summary["loss"].iloc[0]}

    if after is not None:
        s_after = tester.run(after, risk_model, nav=nav)
        details["stress_after"] = s_after
        cmp = pd.DataFrame({"pnl_before": s_before.summary["pnl"],
                            "pnl_after": s_after.summary["pnl"]})
        cmp["change"] = cmp["pnl_after"] - cmp["pnl_before"]
        cmp = cmp.sort_values("pnl_after")
        tables["scenarios before and after"] = cmp
        headline["worst_scenario_after"] = s_after.summary.index[0]
        headline["worst_loss_after"] = s_after.summary["loss"].iloc[0]
        headline["worst_loss_change"] = headline["worst_loss_after"] - headline["worst_loss_before"]
        frames["stress_after"] = s_after.breaches
        if headline["worst_loss_change"] > 0:
            frames["trade"] = pd.DataFrame([_row(
                "warning", "trade_worsens_stress", headline["worst_scenario_after"],
                headline["worst_loss_after"], headline["worst_loss_before"],
                f"the trade raises the worst stress loss from {headline['worst_loss_before']:.2%} "
                f"to {headline['worst_loss_after']:.2%} of NAV")], columns=BREACH_COLUMNS)
    else:
        tables["scenarios"] = s_before.summary[["kind", "horizon_days", "factor_pnl",
                                                "specific_stress", "pnl"]]

    if adv is not None:
        lm = LiquidityModel(lc)
        l_before = lm.run(before, adv, nav, market_cap=market_cap, spread_bps=spread_bps,
                          risk_model=risk_model)
        details["liquidity_before"] = l_before
        key = [k for k in l_before.summary if k.startswith("liquidatable_") and k.endswith("_gross")]
        liq = pd.DataFrame({"before": pd.Series({k: l_before.summary[k] for k in
                                                 key + ["days_to_exit_90pct", "reduction_cost"]})})
        if after is not None:
            l_after = lm.run(after, adv, nav, market_cap=market_cap, spread_bps=spread_bps,
                             risk_model=risk_model)
            details["liquidity_after"] = l_after
            liq["after"] = pd.Series({k: l_after.summary[k] for k in liq.index})
            frames["liquidity_after"] = l_after.breaches
        else:
            frames["liquidity"] = l_before.breaches
        tables["liquidity"] = liq
        book = after if after is not None else before
        if (book < 0).any():
            sq = short_squeeze(book, adv, nav, short_data, squeeze_config, risk_model,
                               market_cap, spread_bps, lc, sc.limits)
            details["squeeze"] = sq
            tables["short squeeze"] = sq.summary
            frames["squeeze"] = sq.breaches
    breaches = stack_breaches(**frames)
    return ScheduledReport("ad_hoc", risk_model.as_of, headline, tables, breaches, details)
