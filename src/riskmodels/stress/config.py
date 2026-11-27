"""Settings for stress testing. Limits are illustrative: set them from your risk appetite."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal


@dataclass(frozen=True)
class StressLimits:
    """Stress loss limits.

    Attributes
    ----------
    max_loss : float or None
        Largest acceptable loss in any one scenario, as a fraction of NAV (0.10 = 10%).
        Includes the stressed specific loss, and the liquidation cost where the overlay
        adds one. None turns it off.
    max_squeeze_loss : float or None
        The same for the short squeeze test (worst move tested).
    alert_levels : dict
        Utilization (loss / limit) at which an alert is raised, by severity name.
    """

    max_loss: float | None = 0.10
    max_squeeze_loss: float | None = 0.05
    alert_levels: dict = field(default_factory=lambda: {"watch": 0.70, "alert": 0.80})

    def __post_init__(self) -> None:
        for name in ("max_loss", "max_squeeze_loss"):
            v = getattr(self, name)
            if v is not None and not v > 0:
                raise ValueError(f"{name} must be positive or None")
        for name, level in self.alert_levels.items():
            if not 0 < level < 1:
                raise ValueError(f"alert level {name!r} must be in (0, 1), got {level}")


@dataclass(frozen=True)
class StressConfig:
    """Settings for :class:`riskmodels.stress.StressTester`.

    Attributes
    ----------
    propagation : ``"conditional"`` or ``"zero"``
        How factors a scenario does not mention are set. ``"conditional"`` (default) uses
        their expected return given the shocked factors under today's factor covariance,
        E[f_u | f_s] = F_us F_ss^-1 f_s, so a market crash also moves the factors that
        usually move with the market. ``"zero"`` leaves them at zero.
    specific_confidence : float
        Confidence level for the stressed stock-specific loss (0.99 adds a 2.33 standard
        deviation adverse specific move of the book).
    max_specific_horizon_days : int or None
        Cap on the horizon used for specific risk (21). Stock-specific losses over a
        multi-year scenario are assumed to be managed by trading within about a month;
        None uses the full scenario horizon.
    trading_days_per_year : int
        Converts the risk model's annualized variances to daily (252).
    industry_map : dict or None
        Scenario industry label -> model industry label (or list of labels), for when the
        model uses a different industry scheme from the library (the library uses Fama-French
        12 names). Several scenario labels mapped to one model label are averaged.
    limits : StressLimits
    """

    propagation: Literal["conditional", "zero"] = "conditional"
    specific_confidence: float = 0.99
    max_specific_horizon_days: int | None = 21
    trading_days_per_year: int = 252
    industry_map: dict | None = None
    limits: StressLimits = field(default_factory=StressLimits)

    def __post_init__(self) -> None:
        if self.propagation not in ("conditional", "zero"):
            raise ValueError("propagation must be 'conditional' or 'zero'")
        if not 0.5 <= self.specific_confidence < 1:
            raise ValueError("specific_confidence must be in [0.5, 1)")
        if self.max_specific_horizon_days is not None and self.max_specific_horizon_days < 1:
            raise ValueError("max_specific_horizon_days must be at least 1 or None")

    def with_overrides(self, **overrides) -> "StressConfig":
        return replace(self, **overrides)
