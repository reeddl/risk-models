"""Every tunable choice in the factor model, with its default and why."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

# Style factor -> {descriptor: weight}. A style score is the weighted average of its
# standardized descriptors, renormalized over the descriptors an asset has.
DEFAULT_STYLE_COMPOSITES: dict[str, dict[str, float]] = {
    "size": {"log_market_cap": 1.0},
    "beta": {"beta": 1.0},
    "momentum": {"momentum": 1.0},
    # USE4 residual-volatility weights.
    "residual_volatility": {"dastd": 0.74, "cmra": 0.16, "hsigma": 0.10},
    "value": {"book_to_price": 0.5, "earnings_to_price": 0.5},
    "quality": {"roe": 0.5, "gross_profitability": 0.5},
    "growth": {"sales_growth": 0.5, "earnings_growth": 0.5},
    "leverage": {"market_leverage": 0.5, "debt_to_assets": 0.5},
    # USE4 liquidity weights on monthly, quarterly and annual share turnover.
    "liquidity": {"stom": 0.35, "stoq": 0.35, "stoa": 0.30},
    "dividend_yield": {"dividend_yield": 1.0},
}

# Applied in order. Residual volatility is highly correlated with beta and with small size,
# and turnover with size, so each is replaced by its residual on the listed factors.
DEFAULT_ORTHOGONALIZE: dict[str, tuple[str, ...]] = {
    "residual_volatility": ("beta", "size"),
    "liquidity": ("size",),
}


@dataclass(frozen=True)
class FactorModelConfig:
    """Settings for :class:`riskmodels.factor_model.FactorModel`.

    Descriptor windows are in trading days whatever the regression frequency. Covariance
    half-lives, Newey-West lags and minimum observation counts are in regression periods
    (days for ``frequency="D"``, weeks for ``"W"``); use :meth:`weekly` for weekly defaults.
    """

    frequency: Literal["D", "W"] = "D"
    periods_per_year: int = 252

    # Estimation universe. Regression weights are sqrt(market cap).
    min_market_cap: float = 0.0

    # Cross-sectional cleaning.
    winsor_n_std: float = 3.0

    # Price-based descriptor windows (trading days).
    beta_window: int = 252
    beta_half_life: float = 63
    beta_min_obs: int = 126
    dastd_window: int = 252
    dastd_half_life: float = 42
    cmra_months: int = 12
    momentum_window: int = 252
    momentum_lag: int = 21
    momentum_half_life: float = 126
    momentum_min_obs: int = 126
    liquidity_min_obs: int = 15

    # Fundamentals.
    fundamental_max_age_days: int = 548     # ignore a value whose period ended > 18 months ago
    growth_tolerance_days: int = 45         # how far "one year earlier" may drift

    style_composites: dict[str, dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_STYLE_COMPOSITES.items()}
    )
    orthogonalize: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_ORTHOGONALIZE)
    )

    # Factor covariance (regression periods).
    vol_half_life: float = 84
    corr_half_life: float = 504
    vol_nw_lags: int = 5
    corr_nw_lags: int = 2
    factor_cov_window: int | None = None    # None uses all history
    corr_shrinkage: float = 0.10
    corr_shrinkage_target: Literal["identity", "constant"] = "identity"

    # Specific risk (regression periods).
    specific_half_life: float = 84
    specific_nw_lags: int = 5
    specific_min_obs: int = 63              # fewer than this: use the size-bucket average
    specific_size_buckets: int = 10
    specific_shrinkage_q: float = 0.1       # USE4 Bayesian shrinkage intensity

    @property
    def warmup_days(self) -> int:
        """Trading days of history needed before the first regression."""
        return max(
            self.beta_min_obs,
            self.momentum_min_obs + self.momentum_lag,
        )

    @property
    def style_factors(self) -> list[str]:
        return list(self.style_composites)

    @classmethod
    def daily(cls, **overrides) -> "FactorModelConfig":
        return replace(cls(), **overrides)

    @classmethod
    def weekly(cls, **overrides) -> "FactorModelConfig":
        base = cls(
            frequency="W",
            periods_per_year=52,
            vol_half_life=26,
            corr_half_life=104,
            vol_nw_lags=2,
            corr_nw_lags=1,
            specific_half_life=26,
            specific_nw_lags=1,
            specific_min_obs=26,
        )
        return replace(base, **overrides)
