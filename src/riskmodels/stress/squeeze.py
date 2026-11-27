"""Short squeeze test: crowded shorts gap up and borrow is recalled.

Mechanics, for each squeeze move M (30% and 50% by default):

1. **Which shorts are crowded.** A short is crowded when any available criterion is met:
   short interest at or above ``min_short_interest`` of float, days to cover (short
   interest / average daily volume) at or above ``min_days_to_cover``, or an annual borrow
   fee at or above ``min_borrow_fee``. A ``crowded`` column in the short data overrides
   the criteria for a name. With no short data at all, every short is treated as crowded
   (the whole short book is squeezed) and the report says so.
2. **Mark to market.** Crowded shorts rise M; other shorts rise ``other_short_move`` and
   longs move ``long_move`` (both 0 by default). P&L is w x move, so a short loses.
3. **Forced covering.** Lenders recall ``recall_fraction`` of each crowded short (all of it
   by default), now worth |w| x (1 + M), and it must be bought back within
   ``recall_days``. The buy is costed with the square-root impact model at stressed
   spreads and volatility. Names that would need more than ``max_participation`` of
   their volume per day to finish in time, or have no ADV, are flagged as buy-in risk:
   the lender may buy them in at a worse price than modelled.
4. **After.** The report gives the book's net and gross after the squeeze and the covering
   (as shares of NAV after the squeeze),
   since a forced cover leaves the book longer than intended.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ..factor_model.risk import RiskModel
from ..liquidity.config import LiquidityConfig
from ..liquidity.inputs import resolve_inputs
from ..liquidity.liquidation import trade_cost
from ..var.limits import _row, breach_frame
from ..var.model import _unpack
from .config import StressLimits

CRITERIA = {
    "short_interest": "min_short_interest",
    "days_to_cover": "min_days_to_cover",
    "borrow_fee": "min_borrow_fee",
}


@dataclass(frozen=True)
class SqueezeConfig:
    """Settings for :func:`short_squeeze`. The defaults are the example numbers from the spec.

    Attributes
    ----------
    moves : tuple
        Price rises applied to crowded shorts (0.30 and 0.50).
    min_short_interest, min_days_to_cover, min_borrow_fee : float or None
        Crowding thresholds (20% of float, 5 days, 5% a year). None turns one off.
    recall_fraction : float
        Share of each crowded short whose borrow is recalled (1.0 = all, "recalls forced").
    recall_days : float
        Trading days allowed to buy back a recalled short before a buy-in (3).
    max_participation : float
        Above this share of daily volume, covering in time is flagged as buy-in risk.
    other_short_move, long_move : float
        Moves for the non-crowded shorts and for the longs (0 by default).
    volume_multiplier, spread_multiplier, vol_multiplier : float
        Trading conditions while covering. Volume in a squeeze is usually high, so volume
        is left at normal; spreads and volatility are stressed.
    """

    moves: tuple[float, ...] = (0.30, 0.50)
    min_short_interest: float | None = 0.20
    min_days_to_cover: float | None = 5.0
    min_borrow_fee: float | None = 0.05
    recall_fraction: float = 1.0
    recall_days: float = 3.0
    max_participation: float = 0.25
    other_short_move: float = 0.0
    long_move: float = 0.0
    volume_multiplier: float = 1.0
    spread_multiplier: float = 3.0
    vol_multiplier: float = 3.0

    def __post_init__(self) -> None:
        if not self.moves or any(m <= 0 for m in self.moves):
            raise ValueError("moves must be positive price rises")
        if not 0 <= self.recall_fraction <= 1:
            raise ValueError("recall_fraction must be in [0, 1]")
        if not self.recall_days > 0 or not 0 < self.max_participation <= 1:
            raise ValueError("recall_days must be positive and max_participation in (0, 1]")

    def with_overrides(self, **overrides) -> "SqueezeConfig":
        return replace(self, **overrides)


def latest_short_interest(short_interest: pd.DataFrame, as_of, symbol_to_asset: dict | None = None) -> pd.DataFrame:
    """Latest FINRA short interest known on ``as_of`` per name (point in time).

    ``short_interest`` is the output of ``PublicDataLoader.short_interest`` (columns
    ``symbol, settlement_date, short_interest, days_to_cover, available_date``). Returns a
    frame indexed by asset with ``days_to_cover`` and ``short_interest`` (shares). Add a
    ``short_interest`` share-of-float column yourself if you have float data; FINRA
    reports shares, so this column is renamed ``short_interest_shares``.
    """
    si = short_interest[pd.to_datetime(short_interest["available_date"]) <= pd.Timestamp(as_of)]
    si = si.sort_values("settlement_date").groupby("symbol").tail(1).set_index("symbol")
    out = si[[c for c in ("days_to_cover", "short_interest") if c in si.columns]].copy()
    out = out.rename(columns={"short_interest": "short_interest_shares"})
    if symbol_to_asset:
        out.index = out.index.map(lambda s: symbol_to_asset.get(s, s))
    out.index.name = "asset"
    return out


def identify_crowded(weights: pd.Series, short_data: pd.DataFrame | None, config: SqueezeConfig) -> pd.DataFrame:
    """One row per short: the crowding criteria, whether it is crowded and why."""
    w = weights.astype(float)
    shorts = w[w < 0]
    out = pd.DataFrame({"weight": shorts})
    if short_data is None or short_data.empty:
        out["crowded"] = True
        out["reason"] = "no short data: all shorts treated as crowded"
        return out
    sd = short_data.reindex(shorts.index)
    flags = pd.DataFrame(index=shorts.index)
    for col, attr in CRITERIA.items():
        thr = getattr(config, attr)
        if col in sd.columns:
            out[col] = sd[col].astype(float)
            if thr is not None:
                flags[col] = sd[col].astype(float) >= thr
    crowded = flags.any(axis=1) if len(flags.columns) else pd.Series(False, index=shorts.index)
    if "crowded" in sd.columns:
        given = sd["crowded"]
        crowded = given.where(given.notna(), crowded).astype(bool)
    out["crowded"] = crowded.astype(bool)
    reasons = []
    for name in shorts.index:
        r = [c for c in flags.columns if bool(flags.loc[name, c])]
        if "crowded" in sd.columns and pd.notna(sd.loc[name, "crowded"]):
            r = ["flagged"] if bool(sd.loc[name, "crowded"]) else []
        reasons.append(",".join(r))
    out["reason"] = reasons
    return out


@dataclass
class SqueezeResult:
    """Output of :func:`short_squeeze`. P&L and weights are fractions of today's NAV.

    Attributes
    ----------
    summary : DataFrame
        One row per move: P&L from the move, cost of forced covering, total, number of
        crowded shorts, buy-in risk names, net and gross before and after.
    crowded : DataFrame
        Every short with its crowding criteria and flag.
    positions : dict
        Move -> per-name table (move, P&L, recalled amount, days and participation needed,
        cover cost, buy-in risk).
    breaches : DataFrame
    """

    config: SqueezeConfig
    summary: pd.DataFrame
    crowded: pd.DataFrame
    positions: dict = field(repr=False)
    breaches: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def worst(self) -> pd.Series:
        return self.summary.sort_values("pnl").iloc[0]


def short_squeeze(
    portfolio,
    adv: pd.Series,
    nav: float,
    short_data: pd.DataFrame | None = None,
    config: SqueezeConfig | None = None,
    risk_model: RiskModel | None = None,
    market_cap: pd.Series | None = None,
    spread_bps: pd.Series | None = None,
    liquidity_config: LiquidityConfig | None = None,
    limits: StressLimits | None = None,
) -> SqueezeResult:
    """Short-book squeeze: crowded shorts up 30-50% with borrow recalls forced.

    ``short_data`` is indexed by name with any of ``short_interest`` (share of float),
    ``days_to_cover``, ``borrow_fee`` (annual, 0.05 = 5%) and ``crowded`` (bool override).
    """
    c = config or SqueezeConfig()
    lc = liquidity_config or LiquidityConfig()
    lim = limits or StressLimits()
    w, _ = _unpack(portfolio, None)
    if not (w < 0).any():
        raise ValueError("the book has no shorts")
    crowded = identify_crowded(w, short_data, c)
    cset = crowded.index[crowded["crowded"]]
    inputs = resolve_inputs(w.index, adv, lc, market_cap, spread_bps, None, risk_model)
    adv_s = inputs["adv"] * c.volume_multiplier

    rows, positions, breaches = [], {}, []
    for M in c.moves:
        move = pd.Series(np.where(w > 0, c.long_move, c.other_short_move), index=w.index)
        move[cset] = M
        mtm = w * move
        after = w * (1 + move)
        recalled = (after.abs() * c.recall_fraction).where(w.index.isin(cset), 0.0)
        cost = trade_cost(recalled, inputs, nav, lc, c.volume_multiplier, c.spread_multiplier,
                          c.vol_multiplier)
        days_needed = (recalled * nav / (lc.participation * adv_s)).where(adv_s > 0, np.inf)
        part_needed = (recalled * nav / (c.recall_days * adv_s)).where(adv_s > 0, np.inf)
        buy_in = (recalled > 0) & ((part_needed > c.max_participation) | ~np.isfinite(part_needed))
        cover_cost = cost["cost"].fillna(0.0)
        final = after + recalled  # buying back a short adds to its (negative) weight
        pnl_total = float(mtm.sum() - cover_cost.sum())
        nav_after = 1 + pnl_total
        pos = pd.DataFrame({
            "weight": w, "crowded": w.index.isin(cset), "move": move, "pnl": mtm,
            "recalled": recalled, "days_to_cover_at_participation": days_needed.where(recalled > 0),
            "participation_needed": part_needed.where(recalled > 0),
            "cover_cost": cover_cost, "buy_in_risk": buy_in, "weight_after": final,
        })
        positions[M] = pos[(pos["weight"] < 0) | (pos["move"] != 0)]
        rows.append({
            "move": M,
            "n_crowded": int(len(cset)),
            "crowded_gross": float(w[cset].abs().sum()),
            "move_pnl": float(mtm.sum()),
            "cover_cost": float(cover_cost.sum()),
            "pnl": pnl_total,
            "loss": -pnl_total,
            "pnl_amount": pnl_total * nav,
            "n_buy_in_risk": int(buy_in.sum()),
            "max_participation_needed": float(part_needed[recalled > 0].max()) if (recalled > 0).any() else 0.0,
            "net_before": float(w.sum()),
            "gross_before": float(w.abs().sum()),
            "net_after": float(final.sum() / nav_after) if nav_after > 0 else np.nan,
            "gross_after": float(final.abs().sum() / nav_after) if nav_after > 0 else np.nan,
        })
        for name in buy_in[buy_in].index:
            breaches.append(_row("warning", "buy_in_risk", name, float(part_needed[name]),
                                 c.max_participation,
                                 f"{name}: covering the recalled short within {c.recall_days:g} "
                                 f"days needs {part_needed[name]:.0%} of daily volume after a "
                                 f"{M:.0%} squeeze" if np.isfinite(part_needed[name]) else
                                 f"{name}: no ADV, so the recalled short cannot be costed or timed"))
    summary = pd.DataFrame(rows).set_index("move")

    if lim.max_squeeze_loss is not None:
        loss = float(summary["loss"].max())
        u = loss / lim.max_squeeze_loss
        sev = "breach" if u >= 1 else None
        if sev is None:
            crossed = [(lvl, n) for n, lvl in lim.alert_levels.items() if u >= lvl]
            sev = max(crossed)[1] if crossed else None
        if sev:
            worst = float(summary["loss"].idxmax())
            breaches.append(_row(sev, "squeeze_loss", f"{worst:.0%} squeeze", loss,
                                 lim.max_squeeze_loss,
                                 f"a {worst:.0%} squeeze of crowded shorts loses {loss:.2%} of "
                                 f"NAV ({u:.0%} of the {lim.max_squeeze_loss:.0%} limit)"))
    if short_data is None or short_data.empty:
        breaches.append(_row("info", "squeeze_fallback", "all shorts", len(cset), None,
                             "no short interest or borrow data given: every short was squeezed"))
    # one buy-in row per name (the worst move), not per move
    bf = breach_frame(breaches)
    if len(bf):
        bf = bf.drop_duplicates(subset=["kind", "name"], keep="last").reset_index(drop=True)
        bf = breach_frame(bf.to_dict("records"))
    return SqueezeResult(config=c, summary=summary, crowded=crowded, positions=positions,
                         breaches=bf)
