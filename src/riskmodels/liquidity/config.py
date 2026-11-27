"""Every tunable choice in the liquidity model, with its default and why.

Weights are fractions of NAV (negative for shorts) and ADV is average daily traded value in
the same currency as NAV. The limit defaults are illustrative: set them from the fund's
redemption terms (see the README).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

# (minimum market cap, quoted bid-ask spread in basis points). Used only when no spread is
# passed in: a rough schedule for US-listed common stock, from large caps at a few basis
# points to micro caps at tens. Replace it with measured spreads when you have them.
DEFAULT_SPREAD_TIERS: tuple[tuple[float, float], ...] = (
    (10e9, 4.0),
    (2e9, 10.0),
    (300e6, 25.0),
    (0.0, 60.0),
)


@dataclass(frozen=True)
class LiquidityLimits:
    """Liquidity limits. Illustrative defaults; set them from the redemption terms.

    Attributes
    ----------
    min_liquidatable : dict
        Days -> minimum share of the book that must be exitable within that many days,
        trading at ``participation`` of ADV in every name at once. ``{5: 0.80}`` means at
        least 80% liquidatable in 5 days.
    basis : ``"gross"`` or ``"nav"``
        What the share in ``min_liquidatable`` is a share of. ``"gross"`` (default) is the
        share of gross exposure; ``"nav"`` is liquidatable gross divided by NAV, which is
        easier to meet the more levered the book is (a 200% gross book that can exit 40% of
        its gross in 5 days has 80% of NAV liquidatable).
    max_position_adv : float or None
        Largest position as a multiple of its ADV (1.0 = one day's volume). With the default
        20% participation, a 1.0x position takes 5 days to exit.
    max_slow_share : float or None
        Largest share of gross allowed in the slowest bucket (over 10 days by default).
    alert_levels : dict
        Utilization at which an alert is raised, by severity name. Utilization is value /
        limit for the ADV and slow-bucket caps, and illiquid share / allowed illiquid share
        for ``min_liquidatable`` (so 80% required and 88% liquid is 12% / 20% = 60%).
    """

    min_liquidatable: dict = field(default_factory=lambda: {5: 0.80})
    basis: Literal["gross", "nav"] = "gross"
    max_position_adv: float | None = 1.0
    max_slow_share: float | None = 0.10
    alert_levels: dict = field(default_factory=lambda: {"watch": 0.80})

    def __post_init__(self) -> None:
        if self.basis not in ("gross", "nav"):
            raise ValueError("basis must be 'gross' or 'nav'")
        for days, share in self.min_liquidatable.items():
            if not days > 0:
                raise ValueError(f"min_liquidatable days must be positive, got {days}")
            hi = 1.0 if self.basis == "gross" else float("inf")
            if not 0 < share <= hi:
                raise ValueError(f"min_liquidatable share for {days}d must be in (0, {hi}]")
        for name in ("max_position_adv", "max_slow_share"):
            v = getattr(self, name)
            if v is not None and not v > 0:
                raise ValueError(f"{name} must be positive or None")
        for name, level in self.alert_levels.items():
            if not 0 < level < 1:
                raise ValueError(f"alert level {name!r} must be in (0, 1), got {level}")


@dataclass(frozen=True)
class LiquidityConfig:
    """Settings for :class:`riskmodels.liquidity.LiquidityModel`.

    Days to liquidate
    -----------------
    participation
        Share of each name's ADV you are willing to trade per day (0.20). The same number
        as the position sizer's ``adv_participation``, so the sizer's liquidity cap and
        this model agree.
    adv_window, adv_method
        ADV is the ``"median"`` (default, robust to one-off volume spikes) or ``"mean"`` of
        daily traded value over the last ``adv_window`` trading days (20).
    bucket_edges
        Days-to-liquidate bucket edges: (1, 5, 10) gives under 1 day, 1-5, 5-10 and over 10.
    profile_days
        Horizons, in days, for the liquidation profile table.

    Trading cost (used for the cost of cutting gross and for the stress overlay)
    ------------
    impact_coefficient
        Y in the square-root impact law, cost = Y x daily vol x sqrt(order / ADV). Published
        estimates are roughly 0.5 to 1; 1.0 is the conservative end.
    spread_tiers
        Market cap -> quoted spread schedule used when no spreads are passed in.
    default_spread_bps
        Spread for a name with neither a spread nor a market cap.
    default_daily_vol
        Daily volatility for a name with no risk forecast (0.02 = 2%).

    Gross reduction
    ---------------
    reduction_target
        Share of gross to cut in the reduction test (0.30 = cut gross by 30%).
    reduction_method
        ``"pro_rata"`` (default) cuts every position by the same share, so the book keeps
        its shape and exposures; it is as slow as the slowest name. ``"fastest"`` trades
        every name at full participation until the target is reached, which finishes
        sooner but leaves the book tilted toward its least liquid names.
    """

    participation: float = 0.20
    adv_window: int = 20
    adv_method: Literal["median", "mean"] = "median"
    bucket_edges: tuple[float, ...] = (1.0, 5.0, 10.0)
    profile_days: tuple[float, ...] = (1.0, 5.0, 10.0, 20.0)

    impact_coefficient: float = 1.0
    spread_tiers: tuple[tuple[float, float], ...] = DEFAULT_SPREAD_TIERS
    default_spread_bps: float = 25.0
    default_daily_vol: float = 0.02

    reduction_target: float = 0.30
    reduction_method: Literal["pro_rata", "fastest"] = "pro_rata"

    limits: LiquidityLimits = field(default_factory=LiquidityLimits)

    def __post_init__(self) -> None:
        if not 0 < self.participation <= 1:
            raise ValueError("participation must be in (0, 1]")
        if self.adv_window < 1:
            raise ValueError("adv_window must be at least 1")
        if self.adv_method not in ("median", "mean"):
            raise ValueError("adv_method must be 'median' or 'mean'")
        edges = tuple(self.bucket_edges)
        if not edges or any(e <= 0 for e in edges) or list(edges) != sorted(set(edges)):
            raise ValueError("bucket_edges must be positive and strictly increasing")
        if not 0 < self.reduction_target <= 1:
            raise ValueError("reduction_target must be in (0, 1]")
        if self.reduction_method not in ("pro_rata", "fastest"):
            raise ValueError("reduction_method must be 'pro_rata' or 'fastest'")
        if self.impact_coefficient < 0:
            raise ValueError("impact_coefficient must be non-negative")

    @property
    def bucket_labels(self) -> list[str]:
        e = [_fmt(x) for x in self.bucket_edges]
        labels = [f"<{e[0]}d"]
        labels += [f"{a}-{b}d" for a, b in zip(e[:-1], e[1:])]
        labels.append(f">{e[-1]}d")
        return labels

    def with_overrides(self, **overrides) -> "LiquidityConfig":
        return replace(self, **overrides)


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"
