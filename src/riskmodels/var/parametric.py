"""Parametric (variance-covariance) VaR from the factor model.

The primary daily number. With the factor model's annualized asset covariance
V = X F X' + D and weights h (fractions of NAV):

    sigma_1d  = sqrt(h' V h / periods_per_year)
    VaR       = z * sigma_1d                  z = normal quantile at the confidence level
    CVaR      = sigma_1d * phi(z) / alpha     (normal; for reference, the headline CVaR
                                               is historical)

Component VaR is the Euler split of VaR, which is z / sqrt(periods_per_year) times the
factor model's volatility contributions, so it adds up exactly:

    position i   h_i (V h)_i / sigma * k      (split into factor and specific parts)
    factor k     x_k (F x)_k / sigma * k      plus one ``specific`` line
    marginal     d VaR / d h_i = (V h)_i / sigma * k

Returns are treated as zero-mean over one day, as in the factor model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..factor_model.risk import RiskDecomposition, RiskModel
from .config import VaRConfig
from .stats import norm_pdf, norm_ppf


@dataclass
class ParametricVaR:
    """Parametric 1-day VaR and its Euler decomposition, fractions of NAV.

    Attributes
    ----------
    sigma : float
        Predicted 1-day volatility of the book.
    var, cvar : float
        Normal VaR and CVaR at the configured confidence (positive losses).
    z : float
    risk : RiskDecomposition
        The factor model's annualized decomposition this is scaled from.
    by_position : DataFrame
        ``var``, ``factor_var``, ``specific_var``, ``marginal_var``, ``cvar`` per name.
    by_factor : DataFrame
        ``group``, ``exposure``, ``var``, ``cvar`` per factor, plus a ``specific`` row.
    """

    sigma: float
    var: float
    cvar: float
    z: float
    risk: RiskDecomposition
    by_position: pd.DataFrame
    by_factor: pd.DataFrame


def parametric_var(risk_model: RiskModel, weights: pd.Series, config: VaRConfig) -> ParametricVaR:
    alpha = config.tail_probability
    z = norm_ppf(config.confidence)
    cvar_ratio = norm_pdf(z) / (alpha * z)        # CVaR / VaR under normality
    k = z / np.sqrt(config.periods_per_year)

    w = weights.astype(float)
    w = w[w != 0]
    risk = risk_model.portfolio_risk(w)
    sigma = risk.total_risk / np.sqrt(config.periods_per_year)

    bp = risk.by_position
    by_position = pd.DataFrame({
        "var": bp["contribution"] * k,
        "factor_var": bp["factor_contribution"] * k,
        "specific_var": bp["specific_contribution"] * k,
        "marginal_var": bp["marginal"] * k,
    })
    by_position["cvar"] = by_position["var"] * cvar_ratio

    bf = risk.by_factor
    by_factor = pd.DataFrame({
        "group": bf["group"],
        "exposure": bf["exposure"],
        "var": bf["contribution"] * k,
    })
    spec = risk.specific_risk ** 2 / risk.total_risk * k if risk.total_risk > 0 else 0.0
    by_factor.loc["specific"] = ["specific", np.nan, spec]
    by_factor["cvar"] = by_factor["var"] * cvar_ratio

    var = z * sigma
    return ParametricVaR(sigma=float(sigma), var=float(var), cvar=float(var * cvar_ratio),
                         z=float(z), risk=risk, by_position=by_position, by_factor=by_factor)


def standalone_parametric_var(risk_model: RiskModel, weights: pd.Series, config: VaRConfig) -> float:
    """Parametric VaR of ``weights`` on their own (0 for an empty book)."""
    w = weights.astype(float)
    w = w[w != 0]
    if w.empty:
        return 0.0
    sigma = risk_model.portfolio_risk(w).total_risk / np.sqrt(config.periods_per_year)
    return float(norm_ppf(config.confidence) * sigma)
