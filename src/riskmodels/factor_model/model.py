"""Fit the factor model end to end and produce risk forecasts.

Timing: exposures are measured at the close of t0 and explain returns over (t0, t1]. With
``frequency="D"`` each period is one trading day; with ``"W"`` it is the trading days
between the last trading day of consecutive weeks, and returns are compounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.covariance import factor_covariance, specific_variance
from ..core.data import MarketData
from .config import FactorModelConfig
from .descriptors import DescriptorEngine
from .exposures import exposure_matrix, industry_factor, style_exposures
from .regression import cross_sectional_regression
from .risk import RiskModel
from .specific_risk import bayesian_shrinkage


def regression_periods(
    dates: pd.DatetimeIndex, frequency: str, first: int
) -> list[tuple[int, int]]:
    """(t0, t1) date positions for each regression period, starting at or after ``first``."""
    n = len(dates)
    if frequency == "D":
        return [(t, t + 1) for t in range(first, n - 1)]
    if frequency == "W":
        week = dates.to_period("W-FRI")
        ends = np.flatnonzero(np.r_[week[1:] != week[:-1], True])
        if ends[-1] != n - 1:  # the final, partial week
            ends = np.r_[ends, n - 1]
        ends = ends[ends >= first]
        return list(zip(ends[:-1].tolist(), ends[1:].tolist()))
    raise ValueError(f"frequency must be 'D' or 'W', got {frequency!r}")


@dataclass
class FactorModelResult:
    """Everything the fit produced, plus methods to build risk forecasts."""

    config: FactorModelConfig
    industries: list[str]
    style_exposures: pd.DataFrame       # (date, asset) x style
    industry: pd.Series                 # (date, asset) -> industry
    market_cap: pd.DataFrame            # date x asset, at exposure dates
    factor_returns: pd.DataFrame        # period end x factor
    specific_returns: pd.DataFrame      # period end x asset
    regression_stats: pd.DataFrame      # period end: weighted_r2, n_assets
    exposure_dates: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))

    @property
    def factors(self) -> pd.Index:
        return self.factor_returns.columns

    def exposures(self, date) -> pd.DataFrame:
        """Full exposure matrix [market | industries | styles] on an exposure date."""
        date = pd.Timestamp(date)
        styles = self.style_exposures.xs(date, level="date")
        ind = self.industry.xs(date, level="date")
        return exposure_matrix(styles, ind, self.industries)[self.factors]

    def risk_model(self, as_of=None) -> RiskModel:
        """Risk forecast using exposures on the last exposure date on or before ``as_of``
        and factor/specific returns realized by then."""
        c = self.config
        dates = self.exposure_dates
        as_of = dates[-1] if as_of is None else pd.Timestamp(as_of)
        pos = dates.searchsorted(as_of, side="right") - 1
        if pos < 0:
            raise ValueError(f"no exposures on or before {as_of.date()}")
        exp_date = dates[pos]

        fr = self.factor_returns[self.factor_returns.index <= exp_date]
        if len(fr) < 2:
            raise ValueError("not enough factor return history before as_of")
        F = factor_covariance(
            fr,
            vol_half_life=c.vol_half_life,
            corr_half_life=c.corr_half_life,
            vol_lags=c.vol_nw_lags,
            corr_lags=c.corr_nw_lags,
            shrinkage=c.corr_shrinkage,
            shrinkage_target=c.corr_shrinkage_target,
            window=c.factor_cov_window,
        ) * c.periods_per_year

        X = self.exposures(exp_date)
        U = self.specific_returns[self.specific_returns.index <= exp_date]
        var, n_obs = specific_variance(U, c.specific_half_life, c.specific_nw_lags)
        sigma_hat = np.sqrt(var.reindex(X.index) * c.periods_per_year)
        cap = self.market_cap.loc[exp_date].reindex(X.index)
        sigma = bayesian_shrinkage(
            sigma_hat,
            cap,
            n_obs=n_obs.reindex(X.index).fillna(0),
            min_obs=c.specific_min_obs,
            n_buckets=c.specific_size_buckets,
            q=c.specific_shrinkage_q,
        )
        return RiskModel(
            as_of=exp_date,
            exposures=X,
            factor_covariance=F,
            specific_variance=sigma ** 2,
            specific_variance_raw=sigma_hat ** 2,
        )


class FactorModel:
    """Barra/Axioma-style fundamental equity factor model.

    Example
    -------
    >>> model = FactorModel(FactorModelConfig.daily())
    >>> result = model.fit(market_data)
    >>> risk = result.risk_model()
    >>> report = risk.portfolio_risk(weights)
    """

    def __init__(self, config: FactorModelConfig | None = None) -> None:
        self.config = config or FactorModelConfig()

    def fit(
        self,
        data: MarketData,
        start=None,
        end=None,
        validate: bool = True,
    ) -> FactorModelResult:
        c = self.config
        if validate:
            data.validate()
        engine = DescriptorEngine(data, c)
        dates = data.dates
        assets = data.assets
        industries = sorted(data.industries["industry"].dropna().unique().tolist())
        factor_names = (["market"] + [industry_factor(k) for k in industries]
                        + c.style_factors)

        first = c.warmup_days
        if start is not None:
            first = max(first, int(dates.searchsorted(pd.Timestamp(start))))
        last = len(dates) - 1
        if end is not None:
            last = min(last, int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1)
        periods = [(a, b) for a, b in regression_periods(dates[: last + 1], c.frequency, first)]
        if not periods:
            raise ValueError("not enough history: no regression periods after warm-up")

        exposure_pos = sorted({a for a, _ in periods} | {last})
        styles_by_date: dict[int, pd.DataFrame] = {}
        ind_by_date: dict[int, pd.Series] = {}
        for t in exposure_pos:
            cov = engine.covered(t)
            names = assets[cov]
            cap = pd.Series(engine.cap[t, cov], index=names)
            ind = engine.industry.iloc[t][cov]
            raw = engine.raw(t).loc[names]
            styles_by_date[t] = style_exposures(raw, cap, ind, c)
            ind_by_date[t] = ind

        rf = engine.rf
        R = data.returns.to_numpy(dtype=float)
        f_rows, u_rows, stats, labels = [], [], [], []
        for t0, t1 in periods:
            block = R[t0 + 1: t1 + 1]
            has = np.isfinite(block).any(0)
            ret = np.expm1(np.nansum(np.log1p(block), axis=0))
            ret = np.where(has, ret, np.nan) - np.expm1(np.log1p(rf[t0 + 1: t1 + 1]).sum())

            styles = styles_by_date[t0]
            ind = ind_by_date[t0]
            X = exposure_matrix(styles, ind, industries)[factor_names]
            r = pd.Series(ret, index=assets).reindex(X.index)
            cap = pd.Series(engine.cap[t0], index=assets).reindex(X.index)
            est = r.notna() & (cap >= c.min_market_cap)
            if est.sum() <= X.shape[1]:
                continue
            fit = cross_sectional_regression(r[est], X[est], cap[est])
            f = fit.factor_returns
            u = r - X.to_numpy() @ f.fillna(0.0).to_numpy()
            f_rows.append(f.to_numpy())
            u_rows.append(u.reindex(assets).to_numpy())
            stats.append((fit.weighted_r2, fit.n_assets))
            labels.append(dates[t1])

        idx = pd.DatetimeIndex(labels, name="date")
        factor_returns = pd.DataFrame(np.array(f_rows), index=idx, columns=factor_names)
        specific_returns = pd.DataFrame(np.array(u_rows), index=idx, columns=assets)
        regression_stats = pd.DataFrame(stats, index=idx, columns=["weighted_r2", "n_assets"])

        exp_dates = dates[exposure_pos]
        style_long = pd.concat(
            [styles_by_date[t] for t in exposure_pos], keys=exp_dates, names=["date", "asset"]
        )
        ind_long = pd.concat(
            [ind_by_date[t] for t in exposure_pos], keys=exp_dates, names=["date", "asset"]
        )
        caps = data.market_cap.iloc[exposure_pos]

        return FactorModelResult(
            config=c,
            industries=industries,
            style_exposures=style_long,
            industry=ind_long,
            market_cap=caps,
            factor_returns=factor_returns,
            specific_returns=specific_returns,
            regression_stats=regression_stats,
            exposure_dates=pd.DatetimeIndex(exp_dates),
        )
