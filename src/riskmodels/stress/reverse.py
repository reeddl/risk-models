"""Reverse stress tests: how far would the top risks have to move to lose a target amount?

All moves are measured against today's factor model over ``horizon_days``. With Sigma the
factor covariance over the horizon (annual / 252 x horizon), book exposures x and a target
loss L (fraction of NAV):

**Most likely factor scenario.** The factor move with the smallest Mahalanobis distance
that loses L is

    f* = -L Sigma x / (x' Sigma x),     distance d = L / sqrt(x' Sigma x)

d is in standard deviations of the book's factor P&L; under a normal model the chance of
a loss at least that large from factors alone is 1 - Phi(d). ``with_specific`` repeats the
calculation with specific risk included (total book volatility).

**Top factor risks.** For each of the factors contributing most to variance, the move in
that factor alone that loses L, two ways:

- ``conditional``: the other factors move with it by their conditional expectation
  (Sigma e_k / Sigma_kk per unit), so f_k = -L Sigma_kk / (Sigma x)_k;
- ``isolated``: nothing else moves, so f_k = -L / x_k.

Each is also given in the factor's own standard deviations over the horizon.

**Top positions.** For each of the names contributing most to risk, the stock-specific move
that loses L on its own, -L / w_i, in its own specific standard deviations. A long cannot
lose more than its weight, so a long smaller than L is marked unreachable.

**Scenario multiples.** For each library scenario, the multiple of its factor shocks that
loses L, keeping its specific stress: k = (L - specific stress) / factor loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..factor_model.exposures import factor_groups
from ..factor_model.risk import RiskModel
from ..var.model import _unpack
from ..var.stats import norm_cdf
from .engine import StressReport


@dataclass
class ReverseStressReport:
    """Output of :func:`reverse_stress`.

    Attributes
    ----------
    target_loss, horizon_days : the question asked.
    most_likely : dict
        Distance (in standard deviations) and normal-model probability of the most likely
        factor scenario, factors only and with specific risk.
    most_likely_shocks : DataFrame
        The most likely factor scenario: shock per factor, in return and in its own
        standard deviations, and the P&L it causes. Largest P&L first.
    factors : DataFrame
        Top factor risks: move needed (conditional and isolated) and in sigmas.
    positions : DataFrame
        Top positions: specific move needed and in sigmas.
    scenarios : DataFrame or None
        Multiple of each library scenario that loses the target.
    """

    target_loss: float
    horizon_days: int
    most_likely: dict
    most_likely_shocks: pd.DataFrame
    factors: pd.DataFrame
    positions: pd.DataFrame
    scenarios: pd.DataFrame | None


def reverse_stress(
    portfolio,
    risk_model: RiskModel,
    target_loss: float = 0.10,
    horizon_days: int = 21,
    top_n: int = 5,
    stress_report: StressReport | None = None,
    trading_days_per_year: int = 252,
) -> ReverseStressReport:
    """Reverse stress ``portfolio`` for a loss of ``target_loss`` (fraction of NAV)."""
    if not target_loss > 0:
        raise ValueError("target_loss must be positive")
    w, _ = _unpack(portfolio, None)
    missing = w.index.difference(risk_model.exposures.index)
    if len(missing):
        raise KeyError(f"no risk forecast for {len(missing)} names: {list(missing[:10])}")
    L = float(target_loss)
    h = float(horizon_days)
    factors = risk_model.factors
    S = risk_model.factor_covariance.loc[factors, factors].to_numpy() * h / trading_days_per_year
    X = risk_model.exposures.loc[w.index, factors].to_numpy()
    x = X.T @ w.to_numpy()
    Sx = S @ x
    fvar = float(x @ Sx)
    d_spec = risk_model.specific_variance.loc[w.index].to_numpy() * h / trading_days_per_year
    svar = float((w.to_numpy() ** 2 * d_spec).sum())
    fvol = float(np.sqrt(fvar))
    tvol = float(np.sqrt(fvar + svar))
    sd = np.sqrt(np.clip(np.diag(S), 0, None))

    # most likely factor scenario
    if fvar > 0:
        f_star = -L * Sx / fvar
        d_f = float(L / fvol)
    else:
        f_star = np.zeros(len(factors))
        d_f = np.inf
    most_likely = {
        "factor_vol": fvol,
        "total_vol": tvol,
        "distance_factors": d_f,
        "probability_factors": float(1 - norm_cdf(d_f)) if np.isfinite(d_f) else 0.0,
        "distance_with_specific": float(L / tvol) if tvol > 0 else np.inf,
        "probability_with_specific": float(1 - norm_cdf(L / tvol)) if tvol > 0 else 0.0,
    }
    shocks = pd.DataFrame({
        "group": factor_groups(factors),
        "exposure": x,
        "shock": f_star,
        "shock_sigmas": np.divide(f_star, sd, out=np.full_like(f_star, np.nan), where=sd > 0),
        "pnl": x * f_star,
    }, index=factors).sort_values("pnl")

    # top factor risks by contribution to factor variance
    contrib = pd.Series(x * Sx, index=factors)
    top = contrib.abs().sort_values(ascending=False).index[:top_n]
    rows = []
    for k in top:
        i = factors.get_loc(k)
        cond = -L * S[i, i] / Sx[i] if Sx[i] != 0 else np.inf
        iso = -L / x[i] if x[i] != 0 else np.inf
        rows.append({
            "factor": k, "group": factor_groups(pd.Index([k])).iloc[0],
            "exposure": x[i], "pct_of_factor_variance": contrib[k] / fvar if fvar > 0 else np.nan,
            "sigma": sd[i],
            "move_conditional": cond, "sigmas_conditional": cond / sd[i] if sd[i] > 0 else np.nan,
            "move_isolated": iso, "sigmas_isolated": iso / sd[i] if sd[i] > 0 else np.nan,
        })
    factor_table = pd.DataFrame(rows).set_index("factor") if rows else pd.DataFrame()

    # top positions by contribution to total variance
    V_h = X @ Sx + d_spec * w.to_numpy()
    pc = pd.Series(w.to_numpy() * V_h, index=w.index)
    topn = pc.abs().sort_values(ascending=False).index[:top_n]
    prow = []
    for name in topn:
        wi = float(w[name])
        move = -L / wi
        s_i = float(np.sqrt(risk_model.specific_variance[name] * h / trading_days_per_year))
        reachable = not (wi > 0 and move < -1.0)
        prow.append({
            "asset": name, "weight": wi,
            "pct_of_variance": pc[name] / (fvar + svar) if fvar + svar > 0 else np.nan,
            "specific_sigma": s_i, "move_needed": move,
            "sigmas_needed": move / s_i if s_i > 0 else np.nan,
            "reachable": reachable,
        })
    position_table = pd.DataFrame(prow).set_index("asset") if prow else pd.DataFrame()

    scen = None
    if stress_report is not None:
        srows = []
        for name, r in stress_report.summary.iterrows():
            floss = -r["factor_pnl"]
            spec = r["specific_stress"]
            if spec >= L:
                k = 0.0
            elif floss > 0:
                k = (L - spec) / floss
            else:
                k = np.inf
            srows.append({"scenario": name, "kind": r["kind"], "loss": r["loss"],
                          "factor_loss": floss, "specific_stress": spec, "multiple_needed": k})
        scen = pd.DataFrame(srows).set_index("scenario").sort_values("multiple_needed")

    return ReverseStressReport(target_loss=L, horizon_days=int(horizon_days),
                               most_likely=most_likely, most_likely_shocks=shocks,
                               factors=factor_table, positions=position_table, scenarios=scen)
