"""Every tunable choice in the position sizing model, with its default and why.

All weights are fractions of NAV (0.05 = 5% of NAV, negative for shorts). All risk numbers
are annualized volatilities in return units, matching :mod:`riskmodels.factor_model`.
The defaults are the example numbers from the spec; change them per portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

# Conviction score -> multiplier. Scores between anchors are linearly interpolated, so with
# these anchors 2 -> 0.75x and 4 -> 1.25x. Scores outside the anchor range are rejected.
DEFAULT_CONVICTION_MULTIPLIERS: dict[float, float] = {1: 0.5, 3: 1.0, 5: 1.5}


@dataclass(frozen=True)
class SizingConfig:
    """Settings for :class:`riskmodels.position_sizing.PositionSizer`.

    Risk budget
    -----------
    base_risk
        Annualized volatility each 1x-conviction position contributes on its own, as a
        fraction of NAV. 0.005 means a 1x name sized so its own risk is 50bp of NAV a year;
        a name with 25% vol then gets 0.005 / 0.25 = 2% of NAV before caps and scaling.
    volatility
        ``"specific"`` sizes on idiosyncratic volatility from the factor model, ``"total"``
        on total volatility. Used consistently for every name. Specific is the default
        because conviction is a view on the stock, and the market and style risk that total
        vol adds is managed at the book level by the beta target and factor checks.
    conviction_multipliers
        Anchor points mapping conviction scores to multipliers (see above).

    Shorts
    ------
    short_risk_multiplier
        Multiplier on ``base_risk`` for shorts (0.5-0.75 in the spec), because a short's
        loss is unbounded and its weight grows as it moves against you.
    short_trim_step, short_trim_fraction
        Trim rule. For every full ``short_trim_step`` the price has risen above the entry
        price, the short's target is cut by ``short_trim_fraction``: with 0.10 and 0.25 a
        short up 25% against you is sized at 0.75**2 = 56% of its untrimmed target.
    short_cover_move
        Adverse move at which the short's target goes to zero (cover). None turns this off,
        and the trim steps keep shrinking it instead.

    Caps (hard limits; they always hold in the output)
    ----
    max_long_weight, max_short_weight
        Largest single long and short, as positive fractions of NAV.
    adv_participation, liquidation_days
        Liquidity cap: |w| <= participation x ADV x days x price / NAV, i.e. the position can
        be exited within ``liquidation_days`` trading at ``adv_participation`` of volume.
    max_sector_gross, max_sector_net
        Per-sector limits on gross (sum |w|) and |net| (sum w). None turns a limit off.
    max_gross, min_net, max_net
        Book-level limits on gross and net exposure.

    Targets (hit as closely as the caps allow; misses are reported)
    -------
    neutralize, target_exposure
        Step 5 target. ``"beta"`` scales the long or short book until the predicted beta to
        the benchmark equals ``target_exposure``; ``"net"`` does the same for net exposure;
        None skips the step.
    book_scaling
        Which book moves to hit the target. ``"shrink"`` (default) scales down whichever
        side has the excess, so hitting the target never adds gross. ``"long"`` or
        ``"short"`` always scales that book (up or down). Only matters when
        ``target_volatility`` is None: with a vol target the overall level is set by the
        vol step, and only the long/short ratio the target needs is left.
    target_volatility
        Predicted annualized portfolio volatility the final weights are scaled to.
        None skips vol targeting.
        When the caps make both targets impossible at once, the beta/net target wins and
        vol ends at the highest level that still allows it; the miss is reported.

    Factor checks (flags only; nothing is hedged automatically)
    -------------
    max_style_exposure
        Flag any style factor whose portfolio exposure (in standard deviations of the
        cross-section) is larger than this in absolute value.
    max_factor_risk_share
        Flag any single style or industry factor that explains more than this share of
        predicted portfolio variance.
    max_industry_exposure
        Flag any industry with net exposure above this (same as sector net, but measured
        through the risk model's industry factors).

    Solver and monitoring
    ---------------------
    tolerance
        How close the solver gets beta (or net) to its target, and predicted vol to its
        target as a relative error, before it stops.
    min_trade
        Trades smaller than this fraction of NAV are dropped from the trade list.
    drift_tolerance
        In :meth:`PositionSizer.monitor`, flag a name whose current weight is more than
        this fraction of NAV away from its target.
    """

    base_risk: float = 0.005
    volatility: Literal["specific", "total"] = "specific"
    conviction_multipliers: dict[float, float] = field(
        default_factory=lambda: dict(DEFAULT_CONVICTION_MULTIPLIERS)
    )

    short_risk_multiplier: float = 0.6
    short_trim_step: float = 0.10
    short_trim_fraction: float = 0.25
    short_cover_move: float | None = None

    max_long_weight: float = 0.06
    max_short_weight: float = 0.04
    adv_participation: float = 0.20
    liquidation_days: float = 5.0
    max_sector_gross: float | None = 0.40
    max_sector_net: float | None = 0.10
    max_gross: float = 2.0
    min_net: float = -0.10
    max_net: float = 0.10

    neutralize: Literal["beta", "net"] | None = "beta"
    target_exposure: float = 0.0
    book_scaling: Literal["shrink", "long", "short"] = "shrink"
    target_volatility: float | None = 0.08

    max_style_exposure: float = 0.25
    max_factor_risk_share: float = 0.10
    max_industry_exposure: float = 0.10

    tolerance: float = 1e-4
    min_trade: float = 0.0005
    drift_tolerance: float = 0.005

    def __post_init__(self) -> None:
        if self.volatility not in ("specific", "total"):
            raise ValueError(f"volatility must be 'specific' or 'total', got {self.volatility!r}")
        if self.neutralize not in ("beta", "net", None):
            raise ValueError(f"neutralize must be 'beta', 'net' or None, got {self.neutralize!r}")
        if self.book_scaling not in ("shrink", "long", "short"):
            raise ValueError(f"book_scaling must be 'shrink', 'long' or 'short', got {self.book_scaling!r}")
        if len(self.conviction_multipliers) < 1:
            raise ValueError("conviction_multipliers needs at least one anchor")
        if any(m < 0 for m in self.conviction_multipliers.values()):
            raise ValueError("conviction multipliers must be non-negative")
        positive = {
            "base_risk": self.base_risk,
            "max_long_weight": self.max_long_weight,
            "max_short_weight": self.max_short_weight,
            "adv_participation": self.adv_participation,
            "liquidation_days": self.liquidation_days,
            "max_gross": self.max_gross,
            "short_trim_step": self.short_trim_step,
        }
        for name, value in positive.items():
            if not value > 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0 <= self.short_trim_fraction < 1:
            raise ValueError("short_trim_fraction must be in [0, 1)")
        if not 0 < self.short_risk_multiplier:
            raise ValueError("short_risk_multiplier must be positive")
        if self.min_net > self.max_net:
            raise ValueError("min_net must not exceed max_net")
        if self.neutralize == "net" and not self.min_net <= self.target_exposure <= self.max_net:
            raise ValueError("a net target outside [min_net, max_net] can never be hit")
        if self.target_volatility is not None and not self.target_volatility > 0:
            raise ValueError("target_volatility must be positive or None")

    def with_overrides(self, **overrides) -> "SizingConfig":
        return replace(self, **overrides)
