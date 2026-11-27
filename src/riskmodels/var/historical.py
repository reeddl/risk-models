"""Historical simulation: today's positions revalued on past daily returns.

Scenario P&L on day t for weights h (fractions of NAV) is

    pnl_t = sum_i h_i r_it

with r the excess return of each name on that day. A name with no return on a day (it had
not listed yet, or did not trade) gets the factor model's proxy X_i f_t, using today's
exposures X_i and that day's factor returns f_t, with zero specific return. Every such
fill is counted in ``HistoricalScenarios.proxied``.

The factor split of each scenario uses today's exposures: factor k contributes
x_k f_kt with x = X' h, and the remainder of the scenario is specific P&L.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.covariance import ewma_weights
from ..core.data import MarketData
from ..factor_model.model import FactorModelResult
from ..factor_model.risk import RiskModel
from .config import VaRConfig


@dataclass
class HistoricalScenarios:
    """Daily return scenarios for a fixed set of names.

    Attributes
    ----------
    asset_returns : DataFrame (date x name)
        Excess returns, with gaps filled by the factor proxy.
    proxied : DataFrame (date x name), bool
        True where the return is the factor proxy rather than observed.
    factor_returns : DataFrame (date x factor) or None
    exposures : DataFrame (name x factor) or None
        Exposures used for the proxy and for the factor split (today's).
    """

    asset_returns: pd.DataFrame
    proxied: pd.DataFrame
    factor_returns: pd.DataFrame | None = None
    exposures: pd.DataFrame | None = None

    @classmethod
    def from_returns(cls, returns: pd.DataFrame) -> "HistoricalScenarios":
        """Scenarios from a plain return panel (no factor split, no proxy)."""
        r = returns.astype(float)
        return cls(asset_returns=r.fillna(0.0), proxied=r.isna())

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.asset_returns.index)

    def _weights(self, weights: pd.Series) -> pd.Series:
        w = weights.astype(float)
        missing = w.index.difference(self.asset_returns.columns)
        if len(missing):
            raise KeyError(f"no scenarios for {len(missing)} names: {list(missing[:10])}")
        return w

    def position_pnl(self, weights: pd.Series) -> pd.DataFrame:
        """Scenario P&L by position (date x name), fractions of NAV."""
        w = self._weights(weights)
        return self.asset_returns[w.index].mul(w, axis=1)

    def pnl(self, weights: pd.Series) -> pd.Series:
        """Scenario P&L of the whole book (date), fractions of NAV."""
        w = self._weights(weights)
        return pd.Series(self.asset_returns[w.index].to_numpy() @ w.to_numpy(),
                         index=self.asset_returns.index, name="pnl")

    def factor_pnl(self, weights: pd.Series) -> pd.DataFrame:
        """Scenario P&L by factor plus a ``specific`` column (date x component). Rows add
        up to :meth:`pnl`."""
        if self.factor_returns is None or self.exposures is None:
            raise ValueError("these scenarios have no factor returns to split by")
        w = self._weights(weights)
        f = self.factor_returns.columns
        x = self.exposures.loc[w.index, f].to_numpy().T @ w.to_numpy()
        out = self.factor_returns[f].mul(x, axis=1)
        out["specific"] = self.pnl(w) - out.sum(axis=1)
        return out


def build_scenarios(
    fit: FactorModelResult,
    data: MarketData,
    names,
    risk_model: RiskModel,
    history_days: int,
) -> HistoricalScenarios:
    """Scenarios for ``names`` over the last ``history_days`` factor-return dates on or
    before the risk model's date."""
    names = pd.Index(names)
    missing = names.difference(risk_model.exposures.index)
    if len(missing):
        raise KeyError(f"no exposures for {len(missing)} names: {list(missing[:10])}")
    fr = fit.factor_returns[fit.factor_returns.index <= risk_model.as_of]
    fr = fr.iloc[-history_days:].fillna(0.0)
    dates = fr.index
    raw = data.returns.reindex(index=dates, columns=names).astype(float)
    if data.risk_free is not None:
        raw = raw.sub(data.risk_free.reindex(dates).fillna(0.0), axis=0)
    X = risk_model.exposures.loc[names, risk_model.factors]
    proxy = pd.DataFrame(fr[X.columns].to_numpy() @ X.to_numpy().T, index=dates, columns=names)
    proxied = raw.isna()
    returns = raw.where(~proxied, proxy)
    return HistoricalScenarios(asset_returns=returns, proxied=proxied,
                               factor_returns=fr[X.columns], exposures=X)


def scenario_weighting(pnl: pd.Series, config: VaRConfig) -> tuple[pd.Series, pd.Series]:
    """Probabilities and scale factors for each scenario.

    Returns ``(probabilities, scale)``. The weighted distribution is P&L ``pnl * scale``
    with probabilities ``probabilities``.

    ``"age"``: EWMA probabilities, half-life ``hs_half_life``, scale 1.
    ``"volatility"``: equal probabilities; scale_t = sigma_now / sigma_t where sigma_t is
    the EWMA volatility of the book's P&L known before day t (half-life
    ``vol_half_life``) and sigma_now includes the last day.
    ``"none"``: equal probabilities, scale 1.
    """
    n = len(pnl)
    idx = pnl.index
    if config.hs_weighting == "age":
        return pd.Series(ewma_weights(n, config.hs_half_life), index=idx), pd.Series(1.0, index=idx)
    if config.hs_weighting == "none":
        return pd.Series(1.0 / n, index=idx), pd.Series(1.0, index=idx)

    lam = 0.5 ** (1.0 / config.vol_half_life)
    x = pnl.to_numpy(dtype=float)
    seed = max(min(20, n), 1)
    var = float(np.mean(x[:seed] ** 2))
    sig = np.empty(n)
    for t in range(n):
        sig[t] = np.sqrt(var)                    # known before day t
        var = lam * var + (1.0 - lam) * x[t] ** 2
    sigma_now = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(sig > 0, sigma_now / sig, 1.0)
    return pd.Series(1.0 / n, index=idx), pd.Series(scale, index=idx)
