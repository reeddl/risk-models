"""Steps 1-3: conviction multipliers, risk-budget weights and the short adjustment.

    w_i = side_i x base_risk x multiplier(conviction_i) x short_adj_i x trim_i / sigma_i

``sigma_i`` is the factor model's specific or total volatility (one choice for every name),
``short_adj_i`` is ``short_risk_multiplier`` for shorts and 1 for longs, and ``trim_i`` is
the trim factor for shorts that have moved against you since entry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SizingConfig


def conviction_multiplier(scores: pd.Series, anchors: dict[float, float]) -> pd.Series:
    """Map conviction scores to multipliers, interpolating linearly between anchors.

    Raises if a score is missing or outside the anchor range, so a typo never silently
    becomes a full-size position.
    """
    keys = np.array(sorted(anchors), dtype=float)
    vals = np.array([anchors[k] for k in sorted(anchors)], dtype=float)
    s = pd.to_numeric(scores, errors="coerce").astype(float)
    bad = s.isna() | (s < keys[0]) | (s > keys[-1])
    if bad.any():
        raise ValueError(
            f"conviction outside [{keys[0]:g}, {keys[-1]:g}] or missing for "
            f"{list(s.index[bad][:10])}"
        )
    return pd.Series(np.interp(s.to_numpy(), keys, vals), index=scores.index)


def parse_side(side: pd.Series) -> pd.Series:
    """+1 for longs, -1 for shorts. Accepts +1/-1 or 'long'/'short' (any case)."""
    def one(v) -> float:
        if isinstance(v, str):
            t = v.strip().lower()
            if t in ("long", "l", "buy"):
                return 1.0
            if t in ("short", "s", "sell"):
                return -1.0
        elif v in (1, -1):
            return float(v)
        raise ValueError(f"side must be long/short or +1/-1, got {v!r}")
    return side.map(one).astype(float)


def adverse_move(side: pd.Series, price: pd.Series, entry_price: pd.Series) -> pd.Series:
    """How far each short has moved against you since entry (0.2 = price up 20%).

    Zero for longs and for shorts with no entry or current price.
    """
    move = (price / entry_price - 1.0).where((side < 0) & (entry_price > 0))
    return move.clip(lower=0.0).fillna(0.0)


def short_trim_factor(move: pd.Series, config: SizingConfig) -> pd.Series:
    """Fraction of the untrimmed short target to keep after an adverse move."""
    steps = np.floor(move.to_numpy() / config.short_trim_step + 1e-12)
    keep = (1.0 - config.short_trim_fraction) ** steps
    if config.short_cover_move is not None:
        keep = np.where(move.to_numpy() >= config.short_cover_move, 0.0, keep)
    return pd.Series(keep, index=move.index)


def risk_budget_weights(
    side: pd.Series,
    multiplier: pd.Series,
    sigma: pd.Series,
    trim: pd.Series,
    config: SizingConfig,
) -> pd.Series:
    """Signed weights before caps, book scaling and vol targeting."""
    if (sigma <= 0).any() or sigma.isna().any():
        bad = sigma.index[(sigma <= 0) | sigma.isna()]
        raise ValueError(f"no usable volatility for {list(bad[:10])}")
    short_adj = np.where(side < 0, config.short_risk_multiplier, 1.0)
    w = side * config.base_risk * multiplier * short_adj * trim / sigma
    return w.astype(float)
