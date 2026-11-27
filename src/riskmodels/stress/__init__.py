"""Stress testing: a scenario library of factor shocks run through the factor model, with
stressed residual risk, a liquidity overlay, a short squeeze test, reverse stress tests and
report functions for each cadence.
"""

from .calibrate import (
    DEFAULT_REFERENCE_MAP,
    calibrate_from_factor_returns,
    calibrate_from_reference,
    recalibrate_library,
    scales_from_comparison,
)
from .config import StressConfig, StressLimits
from .engine import (
    StressReport,
    StressResult,
    StressTester,
    apply_scenario,
    full_shock,
    map_shocks,
)
from .overlay import OverlayReport, liquidity_overlay
from .reports import (
    ScheduledReport,
    ad_hoc_stress_report,
    daily_liquidity_report,
    periodic_stress_report,
    stack_breaches,
)
from .reverse import ReverseStressReport, reverse_stress
from .scenario import Scenario, ScenarioLibrary, default_library
from .squeeze import (
    SqueezeConfig,
    SqueezeResult,
    identify_crowded,
    latest_short_interest,
    short_squeeze,
)

__all__ = [
    "DEFAULT_REFERENCE_MAP",
    "OverlayReport",
    "ReverseStressReport",
    "Scenario",
    "ScenarioLibrary",
    "ScheduledReport",
    "SqueezeConfig",
    "SqueezeResult",
    "StressConfig",
    "StressLimits",
    "StressReport",
    "StressResult",
    "StressTester",
    "ad_hoc_stress_report",
    "apply_scenario",
    "calibrate_from_factor_returns",
    "calibrate_from_reference",
    "daily_liquidity_report",
    "default_library",
    "full_shock",
    "identify_crowded",
    "latest_short_interest",
    "liquidity_overlay",
    "map_shocks",
    "periodic_stress_report",
    "recalibrate_library",
    "reverse_stress",
    "scales_from_comparison",
    "short_squeeze",
    "stack_breaches",
]
