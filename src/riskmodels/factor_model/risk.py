"""Risk forecasts and portfolio risk decomposition.

Asset covariance is V = X F X' + D. For portfolio weights h (or active weights h - b):

    exposure      x = X' h
    factor var    x' F x
    specific var  sum_i h_i^2 d_i
    total vol     sigma = sqrt(factor var + specific var)

Contributions are Euler allocations of sigma, so they add up exactly:

    by position   h_i * (V h)_i / sigma      (split into factor and specific parts)
    by factor     x_k * (F x)_k / sigma      (sums to factor var / sigma)

Marginal contributions are d sigma / d h_i = (V h)_i / sigma and d sigma / d x_k = (F x)_k / sigma.
All risk numbers are annualized volatilities in return units (0.10 = 10%).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .exposures import factor_groups


@dataclass
class RiskDecomposition:
    total_risk: float
    factor_risk: float
    specific_risk: float
    by_factor: pd.DataFrame
    by_position: pd.DataFrame
    by_group: pd.DataFrame

    def summary(self) -> dict[str, float]:
        tv = self.total_risk ** 2
        return {
            "total_risk": self.total_risk,
            "factor_risk": self.factor_risk,
            "specific_risk": self.specific_risk,
            "factor_share_of_variance": self.factor_risk ** 2 / tv if tv > 0 else np.nan,
            "specific_share_of_variance": self.specific_risk ** 2 / tv if tv > 0 else np.nan,
        }


@dataclass
class RiskModel:
    """A factor risk forecast on one date.

    Attributes
    ----------
    as_of : Timestamp
    exposures : DataFrame (asset x factor)
    factor_covariance : DataFrame (factor x factor), annualized
    specific_variance : Series (asset), annualized
    specific_variance_raw : Series (asset), annualized, before size-bucket shrinkage
    """

    as_of: pd.Timestamp
    exposures: pd.DataFrame
    factor_covariance: pd.DataFrame
    specific_variance: pd.Series
    specific_variance_raw: pd.Series

    @property
    def factors(self) -> pd.Index:
        return self.factor_covariance.columns

    @property
    def specific_risk(self) -> pd.Series:
        return np.sqrt(self.specific_variance)

    def asset_covariance(self, assets=None) -> pd.DataFrame:
        X = self.exposures if assets is None else self.exposures.loc[list(assets)]
        Xv = X[self.factors].to_numpy()
        V = Xv @ self.factor_covariance.to_numpy() @ Xv.T
        V[np.diag_indices_from(V)] += self.specific_variance.loc[X.index].to_numpy()
        return pd.DataFrame(V, index=X.index, columns=X.index)

    def total_risk(self) -> pd.Series:
        """Annualized total volatility per asset."""
        Xv = self.exposures[self.factors].to_numpy()
        fvar = np.einsum("ij,jk,ik->i", Xv, self.factor_covariance.to_numpy(), Xv)
        return pd.Series(np.sqrt(fvar + self.specific_variance.loc[self.exposures.index]),
                         index=self.exposures.index)

    def portfolio_risk(
        self,
        weights: pd.Series,
        benchmark: pd.Series | None = None,
    ) -> RiskDecomposition:
        """Decompose the risk of a portfolio, or of active weights if a benchmark is given.

        Weights are fractions of capital (long/short allowed); assets absent from one side
        count as zero weight there.
        """
        h = weights.astype(float)
        if benchmark is not None:
            idx = h.index.union(benchmark.index)
            h = h.reindex(idx, fill_value=0.0) - benchmark.astype(float).reindex(idx, fill_value=0.0)
        h = h[h != 0]
        missing = h.index.difference(self.exposures.index)
        if len(missing):
            raise KeyError(f"no risk forecast for {len(missing)} assets: {list(missing[:10])}")

        X = self.exposures.loc[h.index, self.factors]
        F = self.factor_covariance.to_numpy()
        d = self.specific_variance.loc[h.index].to_numpy()
        hv = h.to_numpy()

        x = X.to_numpy().T @ hv
        Fx = F @ x
        fvar = float(x @ Fx)
        svar = float((hv * hv * d).sum())
        tvar = fvar + svar
        sigma = np.sqrt(tvar)
        inv = 1.0 / sigma if sigma > 0 else 0.0

        by_factor = pd.DataFrame({
            "exposure": x,
            "volatility": np.sqrt(np.clip(np.diag(F), 0, None)),
            "marginal": Fx * inv,
            "contribution": x * Fx * inv,
            "pct_of_variance": x * Fx / tvar if tvar > 0 else np.nan,
        }, index=self.factors)
        by_factor.insert(0, "group", factor_groups(self.factors))

        common = X.to_numpy() @ Fx
        by_position = pd.DataFrame({
            "weight": hv,
            "marginal": (common + d * hv) * inv,
            "factor_contribution": hv * common * inv,
            "specific_contribution": hv * hv * d * inv,
        }, index=h.index)
        by_position["contribution"] = (
            by_position["factor_contribution"] + by_position["specific_contribution"]
        )
        by_position["pct_of_variance"] = by_position["contribution"] * inv

        by_group = by_factor.groupby("group")[["contribution", "pct_of_variance"]].sum()
        spec = pd.DataFrame(
            {"contribution": [svar * inv], "pct_of_variance": [svar / tvar if tvar > 0 else np.nan]},
            index=["specific"],
        )
        by_group = pd.concat([by_group, spec])

        return RiskDecomposition(
            total_risk=float(sigma),
            factor_risk=float(np.sqrt(fvar)),
            specific_risk=float(np.sqrt(svar)),
            by_factor=by_factor,
            by_position=by_position,
            by_group=by_group,
        )
