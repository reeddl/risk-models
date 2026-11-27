"""Specific (idiosyncratic) risk with Bayesian shrinkage toward size-bucket averages.

Time-series estimates of specific volatility are noisy and mean-revert, and the error is
largest for the most extreme estimates. Following Barra USE4, each asset's estimate is
shrunk toward the cap-weighted average of its market-cap decile:

    sigma_shrunk = v * sigma_bucket + (1 - v) * sigma_hat
    v = q |sigma_hat - sigma_bucket| / (Delta + q |sigma_hat - sigma_bucket|)
    Delta = sqrt(mean over the bucket of (sigma_hat - sigma_bucket)^2)

An asset with too little residual history takes the bucket average outright.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def size_buckets(cap: pd.Series, n_buckets: int) -> pd.Series:
    """Market-cap quantile bucket per asset (0 = smallest)."""
    valid = cap.notna() & (cap > 0)
    out = pd.Series(np.nan, index=cap.index, dtype=float)
    if valid.sum() == 0:
        return out
    n = min(n_buckets, int(valid.sum()))
    ranks = cap[valid].rank(method="first")
    out[valid] = np.floor((ranks - 1) / valid.sum() * n).astype(float)
    return out


def bayesian_shrinkage(
    sigma_hat: pd.Series,
    cap: pd.Series,
    n_obs: pd.Series | None = None,
    min_obs: int = 0,
    n_buckets: int = 10,
    q: float = 0.1,
) -> pd.Series:
    """Shrink specific volatilities toward their size-bucket cap-weighted mean."""
    cap = cap.reindex(sigma_hat.index)
    buckets = size_buckets(cap, n_buckets)
    reliable = sigma_hat.notna() & np.isfinite(sigma_hat)
    if n_obs is not None:
        reliable &= n_obs.reindex(sigma_hat.index).fillna(0) >= min_obs

    out = pd.Series(np.nan, index=sigma_hat.index, dtype=float)
    fallback = (
        np.average(sigma_hat[reliable], weights=cap[reliable])
        if reliable.any() else np.nan
    )
    for b in buckets.dropna().unique():
        members = buckets == b
        good = members & reliable
        if good.any():
            mean_b = float(np.average(sigma_hat[good], weights=cap[good]))
            dev = sigma_hat[good] - mean_b
            delta = float(np.sqrt((dev ** 2).mean()))
            absdev = dev.abs()
            v = q * absdev / (delta + q * absdev) if delta > 0 else absdev * 0.0
            out[good] = v * mean_b + (1 - v) * sigma_hat[good]
        else:
            mean_b = fallback
        out[members & ~reliable] = mean_b
    return out
