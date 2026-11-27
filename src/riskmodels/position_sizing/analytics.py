"""Risk numbers the sizer needs, taken from the factor model's :class:`RiskModel`.

Nothing here re-estimates risk: volatilities, betas and exposures all come from
V = X F X' + D in :mod:`riskmodels.factor_model.risk`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..factor_model.model import FactorModelResult
from ..factor_model.risk import RiskModel


def cap_weighted_benchmark(result: FactorModelResult, as_of=None) -> pd.Series:
    """Cap-weighted portfolio of every name the factor model covers on ``as_of``.

    The default market portfolio for predicted betas when you do not pass your own
    benchmark (for example S&P 500 weights).
    """
    rm_dates = result.exposure_dates
    as_of = rm_dates[-1] if as_of is None else pd.Timestamp(as_of)
    pos = rm_dates.searchsorted(as_of, side="right") - 1
    if pos < 0:
        raise ValueError(f"no exposures on or before {as_of.date()}")
    date = rm_dates[pos]
    names = result.exposures(date).index
    cap = result.market_cap.loc[date].reindex(names)
    cap = cap[cap > 0].dropna()
    return cap / cap.sum()


def predicted_betas(risk_model: RiskModel, benchmark: pd.Series) -> pd.Series:
    """Predicted beta of every covered name to ``benchmark``: (V b)_i / (b' V b)."""
    b = benchmark.astype(float)
    b = b[b != 0]
    missing = b.index.difference(risk_model.exposures.index)
    if len(missing):
        raise KeyError(f"benchmark holds {len(missing)} names with no risk forecast: "
                       f"{list(missing[:10])}")
    f = risk_model.factors
    X = risk_model.exposures[f]
    F = risk_model.factor_covariance.loc[f, f].to_numpy()
    xb = X.loc[b.index].to_numpy().T @ b.to_numpy()
    Vb = X.to_numpy() @ (F @ xb)
    spec = risk_model.specific_variance.reindex(X.index).to_numpy()
    bfull = b.reindex(X.index, fill_value=0.0).to_numpy()
    Vb = Vb + spec * bfull
    var_b = float(bfull @ Vb)
    if not var_b > 0:
        raise ValueError("benchmark has zero predicted variance")
    return pd.Series(Vb / var_b, index=X.index, name="beta")


def sector_from_industries(risk_model: RiskModel) -> pd.Series:
    """Each name's industry from the risk model's industry dummies ('industry:' columns)."""
    cols = [c for c in risk_model.factors if str(c).startswith("industry:")]
    if not cols:
        return pd.Series("unknown", index=risk_model.exposures.index, name="sector")
    D = risk_model.exposures[cols]
    has = D.to_numpy().max(axis=1) > 0
    label = D.idxmax(axis=1).str.removeprefix("industry:")
    return label.where(has, "unknown").rename("sector")


class FastRisk:
    """Predicted volatility for many candidate weight vectors on a fixed set of names.

    Precomputes X, F and D for the names once so the sizer's iterations stay cheap. Gives
    the same numbers as ``RiskModel.portfolio_risk(w).total_risk``.
    """

    def __init__(self, risk_model: RiskModel, names: pd.Index) -> None:
        missing = pd.Index(names).difference(risk_model.exposures.index)
        if len(missing):
            raise KeyError(f"no risk forecast for {len(missing)} names: {list(missing[:10])}")
        f = risk_model.factors
        self.names = pd.Index(names)
        self.X = risk_model.exposures.loc[self.names, f].to_numpy(dtype=float)
        self.F = risk_model.factor_covariance.loc[f, f].to_numpy(dtype=float)
        self.d = risk_model.specific_variance.loc[self.names].to_numpy(dtype=float)

    def volatility(self, w: np.ndarray) -> float:
        x = self.X.T @ w
        return float(np.sqrt(max(x @ self.F @ x + (w * w * self.d).sum(), 0.0)))
