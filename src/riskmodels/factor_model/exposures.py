"""Turn raw descriptors into factor exposures.

Pipeline on each date, over the assets the model covers:

1. Winsorize each raw descriptor at +/- ``winsor_n_std`` and standardize it
   (cap-weighted mean 0, equal-weighted std 1).
2. Average the standardized descriptors into a style score using the composite weights,
   renormalizing over the descriptors an asset has.
3. Re-standardize the score, then fill any missing score with the cap-weighted mean
   score of the asset's industry (0 if the whole industry is missing).
4. Orthogonalize where configured (residual volatility on beta and size, liquidity on
   size) with a sqrt-cap weighted regression, then standardize again.

The market factor has exposure 1 for every asset and industries are 0/1 dummies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.preprocessing import (
    clean_descriptor,
    combine_descriptors,
    fill_missing,
    orthogonalize,
    standardize,
)
from .config import FactorModelConfig

MARKET = "market"
INDUSTRY_PREFIX = "industry:"


def industry_factor(name: str) -> str:
    return f"{INDUSTRY_PREFIX}{name}"


def style_exposures(
    raw: pd.DataFrame,
    cap: pd.Series,
    industry: pd.Series,
    config: FactorModelConfig,
) -> pd.DataFrame:
    """Standardized style exposures (asset x style) for one date."""
    n_std = config.winsor_n_std
    needed = {d for comp in config.style_composites.values() for d in comp}
    z = pd.DataFrame(
        {d: clean_descriptor(raw[d], cap, n_std) for d in sorted(needed) if d in raw.columns},
        index=raw.index,
    )
    styles = {}
    for factor, comp in config.style_composites.items():
        score = combine_descriptors(z, comp)
        score = standardize(score, cap)
        styles[factor] = fill_missing(score, industry, cap)
    out = pd.DataFrame(styles, index=raw.index)

    weights = np.sqrt(cap.where(cap > 0))
    for factor, on in config.orthogonalize.items():
        if factor not in out.columns:
            continue
        regressors = [f for f in on if f in out.columns]
        if not regressors:
            continue
        resid = orthogonalize(out[factor], out[regressors], weights)
        out[factor] = standardize(resid, cap).fillna(0.0)
    return out


def exposure_matrix(
    styles: pd.DataFrame,
    industry: pd.Series,
    industries: list[str],
) -> pd.DataFrame:
    """Full exposure matrix [market | industry dummies | styles] for one date."""
    ind = industry.reindex(styles.index)
    dummies = pd.DataFrame(
        {industry_factor(k): (ind == k).astype(float) for k in industries},
        index=styles.index,
    )
    market = pd.DataFrame({MARKET: 1.0}, index=styles.index)
    return pd.concat([market, dummies, styles], axis=1)


def factor_groups(columns: pd.Index) -> pd.Series:
    """Map each factor name to 'market', 'industry' or 'style'."""
    def group(c: str) -> str:
        if c == MARKET:
            return "market"
        if c.startswith(INDUSTRY_PREFIX):
            return "industry"
        return "style"
    return pd.Series([group(c) for c in columns], index=columns)
