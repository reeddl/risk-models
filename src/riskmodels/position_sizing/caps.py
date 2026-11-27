"""Step 4: hard caps.

Every cap here only ever shrinks positions, which is what lets the sizer apply them last
in each iteration and guarantee they hold in its output:

1. Single name and liquidity: clip |w_i| to min(side cap, liquidity cap).
2. Sector gross: scale a sector's names down pro rata until its gross fits.
3. Sector net: scale down the heavier side of the sector (its longs if net long).
4. Book gross: scale every name down pro rata.
5. Book net: scale down the heavier side of the book.

Step 5 can push a sector back over its net limit, so the pass repeats until nothing moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SizingConfig

SINGLE_NAME = "single_name"
LIQUIDITY = "liquidity"
SECTOR_GROSS = "sector_gross"
SECTOR_NET = "sector_net"
GROSS = "gross"
NET = "net"


def liquidity_cap(adv_dollars: pd.Series, nav: float, config: SizingConfig) -> pd.Series:
    """Largest |weight| exitable in ``liquidation_days`` at ``adv_participation`` of volume.

    ``adv_dollars`` is ADV in shares times price. Names without ADV get no liquidity cap
    (infinity); the sizer reports them.
    """
    cap = config.adv_participation * adv_dollars * config.liquidation_days / nav
    return cap.where(adv_dollars.notna() & (adv_dollars >= 0), np.inf).astype(float)


def name_caps(side: pd.Series, liq_cap: pd.Series, config: SizingConfig) -> pd.DataFrame:
    """Per-name |weight| cap and which limit sets it."""
    side_cap = pd.Series(np.where(side > 0, config.max_long_weight, config.max_short_weight),
                         index=side.index)
    liq = liq_cap.reindex(side.index).fillna(np.inf)
    cap = np.minimum(side_cap, liq)
    source = np.where(liq < side_cap, LIQUIDITY, SINGLE_NAME)
    return pd.DataFrame({"side_cap": side_cap, "liquidity_cap": liq, "cap": cap,
                         "cap_source": source}, index=side.index)


def _shrink_side(w: np.ndarray, mask: np.ndarray, net: float, limit_lo: float,
                 limit_hi: float) -> tuple[np.ndarray, bool]:
    """Shrink the heavier side of ``w[mask]`` so its net lands inside [lo, hi]."""
    long_ = mask & (w > 0)
    short = mask & (w < 0)
    L = w[long_].sum()
    S = -w[short].sum()
    if net > limit_hi and L > 0:
        a = np.clip((limit_hi + S) / L, 0.0, 1.0)
        w[long_] *= a
        return w, a < 1.0
    if net < limit_lo and S > 0:
        a = np.clip((L - limit_lo) / S, 0.0, 1.0)
        w[short] *= a
        return w, a < 1.0
    return w, False


def apply_caps(
    weights: pd.Series,
    caps: pd.Series,
    sector: pd.Series,
    config: SizingConfig,
    cap_source: pd.Series | None = None,
    max_passes: int = 50,
) -> tuple[pd.Series, pd.Series]:
    """Apply every hard limit. Returns (capped weights, binding limits per name).

    ``caps`` is each name's |weight| cap and ``cap_source`` names the limit behind it
    (``single_name`` or ``liquidity``). ``binding`` is a comma-separated list of the limits
    that cut each name ('' if none).
    """
    w = weights.to_numpy(dtype=float).copy()
    names = weights.index
    cap = caps.reindex(names).to_numpy(dtype=float)
    sec = sector.reindex(names).fillna("unknown").to_numpy()
    groups = {g: sec == g for g in pd.unique(sec)}
    binding = [set() for _ in range(len(w))]

    def mark(mask: np.ndarray, label: str) -> None:
        for i in np.flatnonzero(mask & (w != 0)):
            binding[i].add(label)

    tol = 1e-12
    for _ in range(max_passes):
        before = w.copy()

        over = np.abs(w) > cap + tol
        mark(over, "cap")
        w = np.sign(w) * np.minimum(np.abs(w), cap)

        if config.max_sector_gross is not None:
            for g, m in groups.items():
                gross = np.abs(w[m]).sum()
                if gross > config.max_sector_gross + tol:
                    mark(m, SECTOR_GROSS)
                    w[m] *= config.max_sector_gross / gross

        if config.max_sector_net is not None:
            lim = config.max_sector_net
            for g, m in groups.items():
                net = w[m].sum()
                if abs(net) > lim + tol:
                    w, _ = _shrink_side(w, m, net, -lim, lim)
                    side = (w > 0) if net > 0 else (w < 0)
                    mark(m & side, SECTOR_NET)

        gross = np.abs(w).sum()
        if gross > config.max_gross + tol:
            mark(np.ones_like(w, dtype=bool), GROSS)
            w *= config.max_gross / gross

        net = w.sum()
        if net > config.max_net + tol or net < config.min_net - tol:
            side = (w > 0) if net > config.max_net else (w < 0)
            all_ = np.ones_like(w, dtype=bool)
            w, _ = _shrink_side(w, all_, net, config.min_net, config.max_net)
            mark(side, NET)

        if np.max(np.abs(w - before), initial=0.0) <= tol:
            break

    # Report the name cap under the limit that set it.
    source = (cap_source.reindex(names).fillna(SINGLE_NAME).to_numpy()
              if cap_source is not None else np.full(len(w), SINGLE_NAME, dtype=object))
    labels = []
    for i, b in enumerate(binding):
        if "cap" in b:
            b.discard("cap")
            b.add(str(source[i]))
        labels.append(",".join(sorted(b)))
    return pd.Series(w, index=names), pd.Series(labels, index=names)
