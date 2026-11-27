"""Covariance estimators: EWMA, Newey-West, correlation shrinkage and PSD repair.

All estimators are NaN-aware. A missing observation (an industry with no members that
period, a stock before its IPO) is dropped pairwise and the EWMA weights are renormalized
over the observations that exist, so short histories do not drag estimates toward zero.

Factor and specific returns are treated as zero-mean by default, the standard choice in
risk models: sample means of daily factor returns are mostly noise and subtracting them
understates risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_weights(n: int, half_life: float) -> np.ndarray:
    """Weights for n observations, oldest first, summing to one.

    The newest observation has weight proportional to 1 and one ``half_life`` back has 1/2.
    """
    if n <= 0:
        return np.empty(0)
    if half_life is None or not np.isfinite(half_life):
        w = np.ones(n)
    else:
        age = np.arange(n - 1, -1, -1, dtype=float)
        w = 0.5 ** (age / float(half_life))
    return w / w.sum()


def _lagged_moment(F: np.ndarray, w: np.ndarray, lag: int) -> np.ndarray:
    """Weighted E[f_t f_{t-lag}'] over pairwise-complete observations (P x P)."""
    M = np.isfinite(F)
    F0 = np.where(M, F, 0.0)
    if lag == 0:
        A, B, MA, MB, wa = F0, F0, M, M, w
    else:
        A, B = F0[lag:], F0[:-lag]
        MA, MB = M[lag:], M[:-lag]
        wa = w[lag:]
    num = (A * wa[:, None]).T @ B
    den = (MA * wa[:, None]).T.astype(float) @ MB.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, 0.0)
    return out


def newey_west_covariance(
    returns: np.ndarray,
    half_life: float,
    lags: int = 0,
    demean: bool = False,
) -> np.ndarray:
    """EWMA covariance with a Newey-West (Bartlett) correction for serial correlation.

    Sigma = Gamma_0 + sum_{l=1..L} (1 - l/(L+1)) (Gamma_l + Gamma_l')

    Daily factor returns are autocorrelated (stale prices, slow information diffusion), so
    the one-day covariance scaled up by time misstates longer-horizon risk. The Bartlett
    weights keep the result positive semi-definite in population.
    """
    F = np.asarray(returns, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    w = ewma_weights(F.shape[0], half_life)
    if demean:
        M = np.isfinite(F)
        F0 = np.where(M, F, 0.0)
        sw = (M * w[:, None]).sum(0)
        mu = np.where(sw > 0, (F0 * w[:, None]).sum(0) / np.where(sw > 0, sw, 1), 0.0)
        F = F - mu
    cov = _lagged_moment(F, w, 0)
    for lag in range(1, int(lags) + 1):
        if lag >= F.shape[0]:
            break
        g = _lagged_moment(F, w, lag)
        cov = cov + (1.0 - lag / (lags + 1.0)) * (g + g.T)
    return 0.5 * (cov + cov.T)


def cov_to_corr(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    inv = np.where(sd > 0, 1.0 / np.where(sd > 0, sd, 1.0), 0.0)
    corr = cov * inv[:, None] * inv[None, :]
    np.fill_diagonal(corr, 1.0)
    return corr, sd


def shrink_correlation(corr: np.ndarray, intensity: float, target: str = "identity") -> np.ndarray:
    """Linear shrinkage of a correlation matrix toward a structured target.

    target="identity" pulls every correlation toward zero; target="constant" pulls them toward
    the average off-diagonal correlation. Variances are untouched.
    """
    if intensity <= 0:
        return corr
    p = corr.shape[0]
    if target == "identity":
        T = np.eye(p)
    elif target == "constant":
        off = corr[~np.eye(p, dtype=bool)]
        rho = float(off.mean()) if off.size else 0.0
        T = np.full((p, p), rho)
        np.fill_diagonal(T, 1.0)
    else:
        raise ValueError(f"unknown shrinkage target {target!r}")
    return (1.0 - intensity) * corr + intensity * T


def nearest_psd(mat: np.ndarray, min_eigenvalue_ratio: float = 1e-8) -> np.ndarray:
    """Clip eigenvalues so the matrix is positive definite, keeping it symmetric."""
    sym = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(sym)
    floor = max(float(vals.max()), 0.0) * min_eigenvalue_ratio
    if vals.min() >= floor:
        return sym
    vals = np.clip(vals, floor, None)
    out = (vecs * vals) @ vecs.T
    return 0.5 * (out + out.T)


def _psd_correlation(corr: np.ndarray) -> np.ndarray:
    c = nearest_psd(corr)
    d = np.sqrt(np.clip(np.diag(c), 1e-300, None))
    c = c / d[:, None] / d[None, :]
    np.fill_diagonal(c, 1.0)
    return c


def factor_covariance(
    factor_returns: pd.DataFrame,
    vol_half_life: float,
    corr_half_life: float,
    vol_lags: int = 0,
    corr_lags: int = 0,
    shrinkage: float = 0.0,
    shrinkage_target: str = "identity",
    window: int | None = None,
    demean: bool = False,
) -> pd.DataFrame:
    """Factor covariance with separate horizons for volatilities and correlations.

    Volatilities react quickly to regime changes, so they use a short half-life. Correlations
    need many more observations to estimate a P x P matrix stably, so they use a long one.
    The two are combined as D C D, the correlation matrix is shrunk, and the result is made
    positive definite.
    """
    F = factor_returns
    if window is not None:
        F = F.iloc[-int(window):]
    arr = F.to_numpy(dtype=float)
    vol_cov = newey_west_covariance(arr, vol_half_life, vol_lags, demean)
    corr_cov = newey_west_covariance(arr, corr_half_life, corr_lags, demean)
    sd = np.sqrt(np.clip(np.diag(vol_cov), 0.0, None))
    corr, _ = cov_to_corr(corr_cov)
    corr = shrink_correlation(corr, shrinkage, shrinkage_target)
    corr = _psd_correlation(corr)
    cov = corr * sd[:, None] * sd[None, :]
    cov = nearest_psd(cov)
    return pd.DataFrame(cov, index=F.columns, columns=F.columns)


def specific_variance(
    residuals: pd.DataFrame,
    half_life: float,
    lags: int = 0,
    window: int | None = None,
    floor_ratio: float = 0.5,
) -> tuple[pd.Series, pd.Series]:
    """Per-asset EWMA variance of residual returns with a diagonal Newey-West correction.

    Returns (variance per period, number of observations). The Newey-West term is floored at
    ``floor_ratio`` times the lag-0 variance: strong negative autocorrelation in a thin
    history (bid-ask bounce) should not drive a forecast toward zero.
    """
    U = residuals
    if window is not None:
        U = U.iloc[-int(window):]
    arr = U.to_numpy(dtype=float)
    M = np.isfinite(arr)
    A0 = np.where(M, arr, 0.0)
    w = ewma_weights(arr.shape[0], half_life)

    def moment(lag: int) -> np.ndarray:
        if lag == 0:
            num = (A0 * A0 * w[:, None]).sum(0)
            den = (M * w[:, None]).sum(0)
        else:
            num = (A0[lag:] * A0[:-lag] * w[lag:, None]).sum(0)
            den = ((M[lag:] & M[:-lag]) * w[lag:, None]).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan if lag == 0 else 0.0)

    g0 = moment(0)
    var = g0.copy()
    for lag in range(1, int(lags) + 1):
        if lag >= arr.shape[0]:
            break
        var = var + 2.0 * (1.0 - lag / (lags + 1.0)) * moment(lag)
    var = np.where(np.isfinite(g0), np.maximum(var, floor_ratio * g0), np.nan)
    return (
        pd.Series(var, index=U.columns, dtype=float),
        pd.Series(M.sum(0), index=U.columns, dtype=int),
    )
