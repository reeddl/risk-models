"""Step 7: trade list, limit checks and breach alerts.

Alerts have three severities:

``breach``  a hard limit is broken (single name, liquidity, sector, gross, net).
``warning`` a target was missed, an unintended factor bet was found, or an input is missing.
``info``    something the model did on purpose that you should know about (a short trimmed
            or covered after moving against you).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..factor_model.risk import RiskDecomposition
from .config import SizingConfig

BREACH_COLUMNS = ["severity", "kind", "name", "value", "limit", "message"]
_TOL = 1e-9


def _row(severity, kind, name, value, limit, message) -> dict:
    return {"severity": severity, "kind": kind, "name": name, "value": value,
            "limit": limit, "message": message}


def check_limits(
    weights: pd.Series,
    caps: pd.DataFrame,
    sector: pd.Series,
    config: SizingConfig,
) -> list[dict]:
    """Hard-limit breaches for a weight vector (target or current)."""
    rows: list[dict] = []
    w = weights.astype(float)
    liq_cap = caps["liquidity_cap"].reindex(w.index)
    for name, x in w.items():
        if x == 0:
            continue
        limit = config.max_long_weight if x > 0 else config.max_short_weight
        if abs(x) > limit + _TOL:
            kind = "single_name_long" if x > 0 else "single_name_short"
            rows.append(_row("breach", kind, name, x, limit,
                             f"{name} is {abs(x):.2%} of NAV, over the {limit:.2%} cap"))
        lc = liq_cap.get(name, np.inf)
        lc = np.inf if pd.isna(lc) else lc
        if np.isfinite(lc) and abs(x) > lc + _TOL:
            rows.append(_row("breach", "liquidity", name, x, lc,
                             f"{name} is {abs(x):.2%} of NAV, over its {lc:.2%} liquidity cap"))

    sec = sector.reindex(w.index).fillna("unknown")
    by_sec = pd.DataFrame({"w": w, "sector": sec}).groupby("sector")["w"]
    if config.max_sector_gross is not None:
        for g, gross in by_sec.apply(lambda s: s.abs().sum()).items():
            if gross > config.max_sector_gross + _TOL:
                rows.append(_row("breach", "sector_gross", g, gross, config.max_sector_gross,
                                 f"sector {g} gross {gross:.2%} is over {config.max_sector_gross:.2%}"))
    if config.max_sector_net is not None:
        for g, net in by_sec.sum().items():
            if abs(net) > config.max_sector_net + _TOL:
                rows.append(_row("breach", "sector_net", g, net, config.max_sector_net,
                                 f"sector {g} net {net:+.2%} is outside +/-{config.max_sector_net:.2%}"))

    gross = float(w.abs().sum())
    if gross > config.max_gross + _TOL:
        rows.append(_row("breach", "gross", "book", gross, config.max_gross,
                         f"gross {gross:.2%} is over {config.max_gross:.2%}"))
    net = float(w.sum())
    if net > config.max_net + _TOL or net < config.min_net - _TOL:
        rows.append(_row("breach", "net", "book", net, (config.min_net, config.max_net),
                         f"net {net:+.2%} is outside [{config.min_net:+.2%}, {config.max_net:+.2%}]"))
    return rows


def check_targets(
    exposure_value: float | None,
    predicted_vol: float,
    config: SizingConfig,
) -> list[dict]:
    """Warnings when the beta/net target or the vol target was not reached."""
    rows: list[dict] = []
    if config.neutralize is not None and exposure_value is not None:
        miss = exposure_value - config.target_exposure
        if abs(miss) > max(config.tolerance, 1e-6) * 10:
            rows.append(_row("warning", f"{config.neutralize}_target", "book", exposure_value,
                             config.target_exposure,
                             f"portfolio {config.neutralize} is {exposure_value:+.3f}, target "
                             f"{config.target_exposure:+.3f}; the caps keep it from getting closer"))
    if config.target_volatility is not None:
        rel = predicted_vol / config.target_volatility - 1.0
        if abs(rel) > max(config.tolerance, 1e-6) * 10:
            rows.append(_row("warning", "vol_target", "book", predicted_vol,
                             config.target_volatility,
                             f"predicted vol {predicted_vol:.2%} vs target "
                             f"{config.target_volatility:.2%}; the caps keep it from getting closer"))
    return rows


def check_factors(risk: RiskDecomposition, config: SizingConfig) -> list[dict]:
    """Flag unintended factor bets: large style or industry exposures and risk shares."""
    rows: list[dict] = []
    bf = risk.by_factor
    for f, r in bf.iterrows():
        if r["group"] == "style" and abs(r["exposure"]) > config.max_style_exposure:
            rows.append(_row("warning", "style_exposure", f, r["exposure"], config.max_style_exposure,
                             f"{f} exposure {r['exposure']:+.2f} is beyond "
                             f"+/-{config.max_style_exposure:.2f}"))
        if r["group"] == "industry" and abs(r["exposure"]) > config.max_industry_exposure:
            rows.append(_row("warning", "industry_exposure", f, r["exposure"],
                             config.max_industry_exposure,
                             f"{f} net exposure {r['exposure']:+.2%} is beyond "
                             f"+/-{config.max_industry_exposure:.2%}"))
        if r["group"] in ("style", "industry") and r["pct_of_variance"] > config.max_factor_risk_share:
            rows.append(_row("warning", "factor_risk_share", f, r["pct_of_variance"],
                             config.max_factor_risk_share,
                             f"{f} explains {r['pct_of_variance']:.1%} of predicted variance"))
    return rows


def breach_frame(rows: list[dict]) -> pd.DataFrame:
    order = {"breach": 0, "warning": 1, "info": 2}
    df = pd.DataFrame(rows, columns=BREACH_COLUMNS)
    if len(df):
        df = df.sort_values("severity", key=lambda s: s.map(order), kind="stable")
    return df.reset_index(drop=True)


def trade_list(
    target: pd.Series,
    current: pd.Series | None,
    nav: float,
    price: pd.Series | None,
    adv_dollars: pd.Series | None,
    min_trade: float,
) -> pd.DataFrame:
    """Trades that move ``current`` weights to ``target``.

    Names held now but not in the target are sold or covered in full. Trades smaller than
    ``min_trade`` of NAV are dropped, except full exits.
    """
    cur = pd.Series(dtype=float) if current is None else current.astype(float)
    idx = target.index.union(cur.index)
    t = target.reindex(idx, fill_value=0.0)
    c = cur.reindex(idx, fill_value=0.0)
    trade = t - c
    exit_ = (t == 0) & (c != 0)
    keep = (trade.abs() >= min_trade) | exit_
    trade = trade[keep & (trade != 0)]
    idx = trade.index

    out = pd.DataFrame({
        "current_weight": c.reindex(idx),
        "target_weight": t.reindex(idx),
        "trade_weight": trade,
        "trade_value": trade * nav,
    }, index=idx)
    out["action"] = np.select(
        [(out.current_weight >= 0) & (out.trade_weight > 0),
         (out.current_weight > 0) & (out.trade_weight < 0),
         (out.current_weight <= 0) & (out.trade_weight < 0),
         (out.current_weight < 0) & (out.trade_weight > 0)],
        ["buy", "sell", "short", "cover"], default="")
    # Crossing zero is a sell-and-short (or cover-and-buy); label by where it ends up.
    flip_short = (out.current_weight > 0) & (out.target_weight < 0)
    flip_long = (out.current_weight < 0) & (out.target_weight > 0)
    out.loc[flip_short, "action"] = "sell+short"
    out.loc[flip_long, "action"] = "cover+buy"
    if price is not None:
        p = price.reindex(idx)
        out["price"] = p
        out["shares"] = np.round(out["trade_value"] / p)
    if adv_dollars is not None:
        out["pct_of_adv"] = out["trade_value"].abs() / adv_dollars.reindex(idx)
    out.index.name = "asset"
    return out.sort_values("trade_value", key=np.abs, ascending=False)


def drift_weights(weights: pd.Series, returns: pd.Series) -> pd.Series:
    """Weights after one period of price moves, with no trading.

    w_i' = w_i (1 + r_i) / (1 + sum_j w_j r_j). Missing returns count as 0.
    """
    w = weights.astype(float)
    r = returns.reindex(w.index).fillna(0.0)
    pnl = float((w * r).sum())
    return w * (1.0 + r) / (1.0 + pnl)
