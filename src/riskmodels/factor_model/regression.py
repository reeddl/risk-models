"""Constrained weighted cross-sectional regression.

Each period solves

    min_f  sum_i w_i (r_i - x_i' f)^2   subject to   sum_k c_k f_k = 0

with w_i = sqrt(market cap). Regression weights proportional to sqrt(cap) are the Barra
choice: specific variance falls roughly with sqrt(size), so this is close to GLS without
letting the largest names dominate.

The market column equals the sum of the industry dummies, so the unconstrained problem is
singular. The constraint c (cap weights of the industries, zero on everything else) makes
the cap-weighted industry returns sum to zero. The market factor then equals the
cap-weighted return of the estimation universe (net of style tilts) and each industry
factor is that industry's return relative to the market. The problem is solved exactly
through its KKT system, so no industry is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .exposures import INDUSTRY_PREFIX


@dataclass
class CrossSectionFit:
    factor_returns: pd.Series
    weighted_r2: float
    n_assets: int


def constrained_wls(
    y: np.ndarray,
    X: np.ndarray,
    w: np.ndarray,
    C: np.ndarray | None = None,
) -> np.ndarray:
    """Solve min ||sqrt(w)(y - X f)||^2 s.t. C f = 0 via the KKT system."""
    XtW = X.T * w
    A = XtW @ X
    b = XtW @ y
    if C is None or C.size == 0:
        f, *_ = np.linalg.lstsq(A, b, rcond=None)
        return f
    C = np.atleast_2d(C)
    k, p = C.shape
    K = np.block([[A, C.T], [C, np.zeros((k, k))]])
    rhs = np.concatenate([b, np.zeros(k)])
    try:
        sol = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
    return sol[:p]


def cross_sectional_regression(
    returns: pd.Series,
    exposures: pd.DataFrame,
    cap: pd.Series,
) -> CrossSectionFit:
    """One period's factor returns from the estimation universe.

    ``returns``, ``exposures`` and ``cap`` are indexed by the estimation-universe assets.
    Industries with no members are dropped for this period and get a NaN return.
    """
    X = exposures.astype(float)
    ind_cols = [c for c in X.columns if c.startswith(INDUSTRY_PREFIX)]
    empty = [c for c in ind_cols if X[c].sum() == 0]
    used = X.drop(columns=empty)
    used_ind = [c for c in ind_cols if c not in empty]

    capv = cap.to_numpy(dtype=float)
    w = np.sqrt(capv)
    y = returns.to_numpy(dtype=float)

    C = np.zeros(used.shape[1])
    if used_ind:
        ind_cap = used[used_ind].to_numpy().T @ capv
        pos = [used.columns.get_loc(c) for c in used_ind]
        C[pos] = ind_cap / ind_cap.sum()
        f = constrained_wls(y, used.to_numpy(), w, C[None, :])
    else:
        f = constrained_wls(y, used.to_numpy(), w)

    resid = y - used.to_numpy() @ f
    sst = float((w * y * y).sum())
    r2 = 1.0 - float((w * resid * resid).sum()) / sst if sst > 0 else float("nan")

    out = pd.Series(np.nan, index=X.columns, dtype=float)
    out[used.columns] = f
    return CrossSectionFit(out, r2, len(y))
