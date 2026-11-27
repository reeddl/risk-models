"""Parameters for VaR / CVaR. Defaults follow the framework spec; limits are illustrative."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class VaRLimits:
    """Risk-appetite settings. The defaults are illustrative: set them from your own
    risk appetite.

    Attributes
    ----------
    var_limit : float
        1-day VaR limit as a fraction of NAV (0.02 = 2% NAV).
    alert_levels : dict
        Utilization (VaR / limit) at which an alert is raised, by severity name. A
        utilization of 1.0 or more is always a ``breach``.
    limit_basis : str
        Which VaR is held against the limit: ``"parametric"`` (the primary number),
        ``"historical"``, or ``"max"`` of the two.
    cvar_limit : float or None
        Optional 1-day CVaR limit as a fraction of NAV, checked against historical CVaR.
    max_method_gap : float or None
        Warn when historical and parametric VaR differ by more than this fraction of the
        parametric number (0.25 = 25%). A large gap means fat tails or skew the normal
        model misses, or a stale covariance.
    """

    var_limit: float = 0.02
    alert_levels: dict = field(default_factory=lambda: {"watch": 0.70, "alert": 0.80})
    limit_basis: str = "parametric"
    cvar_limit: float | None = None
    max_method_gap: float | None = 0.25

    def __post_init__(self) -> None:
        if self.var_limit <= 0:
            raise ValueError("var_limit must be positive")
        if self.limit_basis not in ("parametric", "historical", "max"):
            raise ValueError("limit_basis must be 'parametric', 'historical' or 'max'")
        for name, level in self.alert_levels.items():
            if not 0 < level < 1:
                raise ValueError(f"alert level {name!r} must be in (0, 1), got {level}")


@dataclass(frozen=True)
class VaRConfig:
    """VaR / CVaR settings.

    Attributes
    ----------
    confidence : float
        VaR confidence level (0.99). The tail probability is ``1 - confidence``.
    horizon_days : int
        Base horizon in trading days. Only 1 is estimated directly.
    scaled_horizon_days : int
        Longer horizon reported by square-root-of-time scaling (10). This is an
        approximation: it assumes i.i.d. returns and a constant book.
    periods_per_year : int
        Annualization used by the factor model's covariance (252 for daily).
    history_days : int
        Trading days of history for historical simulation (756 is about 3 years; use
        504 for 2). For the parametric number, set the factor model's
        ``factor_cov_window`` to match if you want the same window there.
    min_history_days : int
        Fewer scenarios than this raises a warning in the report.
    hs_weighting : str
        How historical scenarios are weighted so recent volatility counts more:
        ``"age"`` (EWMA probability weights, Boudoukh-Richardson-Whitelaw),
        ``"volatility"`` (Hull-White: each day's P&L rescaled by current / then EWMA
        volatility, equal probabilities) or ``"none"``.
    hs_half_life : float
        Half-life in days of the age weights (``"age"``).
    vol_half_life : float
        Half-life in days of the EWMA volatility used by ``"volatility"`` weighting.
    component_var_band : float
        Probability mass on each side of the VaR quantile whose scenarios are averaged to
        split historical VaR by position, sector and factor (see README).
    significance : float
        Significance level for the backtest tests (0.05).
    limits : VaRLimits
    """

    confidence: float = 0.99
    horizon_days: int = 1
    scaled_horizon_days: int = 10
    periods_per_year: int = 252
    history_days: int = 756
    min_history_days: int = 250
    hs_weighting: str = "age"
    hs_half_life: float = 252.0
    vol_half_life: float = 21.0
    component_var_band: float = 0.005
    significance: float = 0.05
    limits: VaRLimits = field(default_factory=VaRLimits)

    def __post_init__(self) -> None:
        if not 0.5 < self.confidence < 1:
            raise ValueError("confidence must be in (0.5, 1)")
        if self.horizon_days != 1:
            raise ValueError("only a 1-day base horizon is estimated; longer horizons are "
                             "reported by scaling (scaled_horizon_days)")
        if self.hs_weighting not in ("age", "volatility", "none"):
            raise ValueError("hs_weighting must be 'age', 'volatility' or 'none'")
        if self.history_days < 2:
            raise ValueError("history_days must be at least 2")

    @property
    def tail_probability(self) -> float:
        return 1.0 - self.confidence

    @property
    def horizon_scale(self) -> float:
        return float(self.scaled_horizon_days) ** 0.5

    def with_overrides(self, **overrides) -> "VaRConfig":
        return replace(self, **overrides)
