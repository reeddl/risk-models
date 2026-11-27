"""Weighted VaR / CVaR of a P&L distribution, and the tail weights that split them.

Every historical number in the module comes from here, so liquidity and stress models can
reuse the same estimator on their own scenario sets.

Conventions: P&L is in fractions of NAV, gains positive. VaR and CVaR are reported as
positive losses. Scenario probabilities need not be equal (EWMA age weighting).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TailEstimate:
    """VaR and CVaR of one P&L distribution plus the weights behind them.

    Attributes
    ----------
    var, cvar : float
        Positive losses at the tail probability ``alpha``.
    alpha : float
    var_weights : Series (scenario)
        Weights, summing to one, of the scenarios averaged around the VaR quantile. Applied
        to component P&L they split VaR (rescaled so the parts sum exactly to ``var``).
    cvar_weights : Series (scenario)
        Weights, summing to one, of the scenarios beyond VaR (with the fractional
        boundary scenario). Applied to component P&L they split CVaR exactly.
    n_scenarios : int
    effective_scenarios : float
        Kish effective sample size of the scenario probabilities.
    """

    var: float
    cvar: float
    alpha: float
    var_weights: pd.Series
    cvar_weights: pd.Series
    n_scenarios: int
    effective_scenarios: float


def tail_estimate(
    pnl: pd.Series,
    alpha: float,
    probabilities: pd.Series | np.ndarray | None = None,
    band: float = 0.0,
) -> TailEstimate:
    """VaR and CVaR at tail probability ``alpha`` from (weighted) P&L scenarios.

    VaR is the loss at which the cumulative probability of worse-or-equal losses first
    reaches ``alpha``. CVaR is the probability-weighted mean of the worst ``alpha`` of the
    distribution, with the boundary scenario counted fractionally so CVaR is exact:

        CVaR = ( sum_{worse than VaR} p_i L_i + (alpha - sum p_worse) VaR ) / alpha
    """
    pnl = pnl.dropna().astype(float)
    n = len(pnl)
    if n == 0:
        raise ValueError("no P&L scenarios")
    if probabilities is None:
        p = np.full(n, 1.0 / n)
    else:
        if isinstance(probabilities, pd.Series):
            p = probabilities.reindex(pnl.index).to_numpy(dtype=float)
        else:
            p = np.asarray(probabilities, dtype=float)
        if len(p) != n or not np.all(np.isfinite(p)) or (p < 0).any() or p.sum() <= 0:
            raise ValueError("probabilities must be finite, non-negative and align with pnl")
        p = p / p.sum()
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    loss = -pnl.to_numpy()
    order = np.argsort(-loss, kind="stable")        # worst first
    ls, ps = loss[order], p[order]
    cum = np.cumsum(ps)
    k = int(np.searchsorted(cum, alpha - 1e-12))      # first index with cum >= alpha
    k = min(k, n - 1)
    var = float(ls[k])
    before = cum[k - 1] if k > 0 else 0.0

    q = np.zeros(n)
    q[:k] = ps[:k]
    q[k] = max(alpha - before, 0.0)
    cvar = float((q * ls).sum() / alpha)
    cvar_w = np.zeros(n)
    cvar_w[order] = q / q.sum()

    # Scenarios near the quantile: those whose probability mid-point lies within
    # ``band`` of alpha. The boundary scenario is always included.
    mid = cum - ps / 2.0
    near = np.abs(mid - alpha) <= band
    near[k] = True
    vw = np.where(near, ps, 0.0)
    var_w = np.zeros(n)
    var_w[order] = vw / vw.sum()

    ess = float(1.0 / (p ** 2).sum())
    return TailEstimate(
        var=var,
        cvar=cvar,
        alpha=alpha,
        var_weights=pd.Series(var_w, index=pnl.index),
        cvar_weights=pd.Series(cvar_w, index=pnl.index),
        n_scenarios=n,
        effective_scenarios=ess,
    )


def split_by_tail(components: pd.DataFrame, tail: TailEstimate) -> pd.DataFrame:
    """Split VaR and CVaR across the columns of ``components`` (scenario x component P&L
    whose rows sum to the portfolio P&L).

    CVaR parts are conditional expected losses in the tail and add up to CVaR exactly. VaR
    parts are expected losses in the scenarios around the quantile, rescaled so they add up
    to VaR (when that average is zero the unscaled average is returned).
    """
    comp = components.reindex(tail.cvar_weights.index).fillna(0.0)
    cvar_part = -(comp.mul(tail.cvar_weights, axis=0)).sum()
    raw_var = -(comp.mul(tail.var_weights, axis=0)).sum()
    total = raw_var.sum()
    var_part = raw_var * (tail.var / total) if abs(total) > 1e-15 else raw_var
    return pd.DataFrame({"var": var_part, "cvar": cvar_part})
