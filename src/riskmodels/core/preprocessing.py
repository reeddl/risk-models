"""Cross-sectional cleaning: winsorize, standardize, fill and orthogonalize.

Conventions follow Barra USE4: a descriptor is standardized to a cap-weighted mean of zero
and an equal-weighted standard deviation of one. Cap-weighting the mean makes the
cap-weighted market portfolio have zero exposure to every style, so the market factor
carries the market return. Equal-weighting the dispersion stops a few mega caps from
setting the scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _valid(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & np.isfinite(w) & (w > 0)


def _arrays(x: pd.Series, w: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    return (x.to_numpy(dtype=float, na_value=np.nan),
            w.reindex(x.index).to_numpy(dtype=float, na_value=np.nan))


def _moments(x: np.ndarray, w: np.ndarray) -> tuple[float, float] | None:
    """(cap-weighted mean, equal-weighted std) over valid entries, or None if < 2."""
    m = _valid(x, w)
    if m.sum() < 2:
        return None
    xm = x[m]
    return float(np.dot(xm, w[m]) / w[m].sum()), float(xm.std())


def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    xv, wv = _arrays(x, w)
    m = _valid(xv, wv)
    if not m.any():
        return float("nan")
    return float(np.dot(xv[m], wv[m]) / wv[m].sum())


def _standardize(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan)
    mom = _moments(x, w)
    if mom is None:
        return out
    mu, sd = mom
    ok = np.isfinite(x)
    if not np.isfinite(sd) or sd <= 0:
        out[ok] = 0.0
        return out
    out[ok] = (x[ok] - mu) / sd
    return out


def _robust_preclip(y: np.ndarray, k: float = 10.0) -> np.ndarray:
    """Clip gross errors at median +/- k robust std (1.4826 MAD) before the mean/std passes.

    A handful of data errors (a cap off by 1e6) inflate the std so much that clipping on
    mean +/- 3 std alone needs dozens of passes. The pre-clip is wide enough that it never
    binds on genuine data.
    """
    v = y[np.isfinite(y)]
    if v.size < 3:
        return y
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med)))
    if mad <= 0:
        return y
    return np.clip(y, med - k * mad, med + k * mad)


def _winsorize(x: np.ndarray, w: np.ndarray, n_std: float, max_iter: int) -> np.ndarray:
    y = _robust_preclip(np.where(np.isfinite(x), x, np.nan))
    for _ in range(max_iter):
        mom = _moments(y, w)
        if mom is None:
            return y
        mu, sd = mom
        if not np.isfinite(sd) or sd <= 0:
            return y
        clipped = np.clip(y, mu - n_std * sd, mu + n_std * sd)
        if np.nanmax(np.abs(clipped - y), initial=0.0) <= 1e-12 * max(sd, 1.0):
            return clipped
        y = clipped
    return y


def standardize(x: pd.Series, cap: pd.Series) -> pd.Series:
    """(x - cap-weighted mean) / equal-weighted std. Returns NaN where x is NaN."""
    xv, wv = _arrays(x, cap)
    return pd.Series(_standardize(xv, wv), index=x.index, dtype=float)


def winsorize(x: pd.Series, cap: pd.Series, n_std: float = 3.0, max_iter: int = 50) -> pd.Series:
    """Clip x at cap-weighted mean +/- n_std equal-weighted std, repeated until stable.

    Gross errors are first clipped at median +/- 10 robust std. Each mean/std pass then
    shrinks the std, so the bounds tighten until no value moves (or ``max_iter`` passes).
    Values are clipped, not dropped, so ranks are preserved.
    """
    xv, wv = _arrays(x, cap)
    return pd.Series(_winsorize(xv, wv, n_std, max_iter), index=x.index, dtype=float)


def clean_descriptor(x: pd.Series, cap: pd.Series, n_std: float = 3.0) -> pd.Series:
    """Winsorize then standardize one raw descriptor."""
    xv, wv = _arrays(x, cap)
    return pd.Series(_standardize(_winsorize(xv, wv, n_std, 50), wv), index=x.index, dtype=float)


def combine_descriptors(z: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weighted average of standardized descriptors, renormalizing over those available.

    An asset missing one descriptor is scored on the rest rather than dropped.
    """
    cols = [c for c in weights if c in z.columns]
    if not cols:
        return pd.Series(np.nan, index=z.index, dtype=float)
    w = np.array([weights[c] for c in cols], dtype=float)
    vals = z[cols].to_numpy(dtype=float)
    present = np.isfinite(vals)
    num = np.where(present, vals, 0.0) @ w
    den = present.astype(float) @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return pd.Series(out, index=z.index, dtype=float)


def fill_missing(z: pd.Series, groups: pd.Series, cap: pd.Series) -> pd.Series:
    """Fill missing standardized exposures with the cap-weighted mean of the asset's group.

    Groups are industries. If a whole group is missing the fill is 0, the cap-weighted market
    average, so an asset with no information is neutral on that factor.
    """
    zv, wv = _arrays(z, cap)
    miss = ~np.isfinite(zv)
    if not miss.any():
        return z.astype(float)
    codes, _ = pd.factorize(groups.reindex(z.index), use_na_sentinel=True)
    ok = _valid(zv, wv) & (codes >= 0)
    n_groups = int(codes.max()) + 1 if codes.size and codes.max() >= 0 else 0
    num = np.bincount(codes[ok], weights=zv[ok] * wv[ok], minlength=n_groups)
    den = np.bincount(codes[ok], weights=wv[ok], minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    fill = np.where(codes >= 0, means[np.clip(codes, 0, None)] if n_groups else 0.0, 0.0)
    out = np.where(miss, fill, zv)
    return pd.Series(out, index=z.index, dtype=float)


def orthogonalize(y: pd.Series, X: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Residual of a weighted regression of y on [1, X].

    Used to strip, for example, beta and size out of residual volatility so the two factors
    do not fight over the same return.
    """
    weights = weights.reindex(y.index)
    X = X.reindex(y.index)
    A = np.column_stack([np.ones(len(y)), X.to_numpy(dtype=float)])
    yv = y.to_numpy(dtype=float)
    wv = weights.to_numpy(dtype=float)
    fit_rows = np.isfinite(yv) & np.isfinite(A).all(axis=1) & np.isfinite(wv) & (wv > 0)
    out = pd.Series(np.nan, index=y.index, dtype=float)
    if fit_rows.sum() <= A.shape[1]:
        return y.astype(float)
    sw = np.sqrt(wv[fit_rows])
    coef, *_ = np.linalg.lstsq(A[fit_rows] * sw[:, None], yv[fit_rows] * sw, rcond=None)
    pred_rows = np.isfinite(yv) & np.isfinite(A).all(axis=1)
    out[pred_rows] = yv[pred_rows] - A[pred_rows] @ coef
    return out
