"""VaR / CVaR: parametric VaR from the factor model as the primary daily number,
EWMA-weighted historical simulation as a cross-check and the source of CVaR, component
VaR by position, sector, factor and side, limits with alert levels, and backtests.
"""

from .backtest import (
    BacktestResult,
    backtest,
    backtest_report,
    christoffersen_test,
    exceedances,
    kupiec_test,
    rolling_forecasts,
    traffic_light,
)
from .config import VaRConfig, VaRLimits
from .historical import HistoricalScenarios, build_scenarios, scenario_weighting
from .limits import check_var_limits
from .model import VaRModel, VaRReport
from .parametric import ParametricVaR, parametric_var, standalone_parametric_var
from .stats import binom_cdf, chi2_sf, norm_cdf, norm_pdf, norm_ppf
from .tail import TailEstimate, split_by_tail, tail_estimate

__all__ = [
    "BacktestResult",
    "HistoricalScenarios",
    "ParametricVaR",
    "TailEstimate",
    "VaRConfig",
    "VaRLimits",
    "VaRModel",
    "VaRReport",
    "backtest",
    "backtest_report",
    "binom_cdf",
    "build_scenarios",
    "check_var_limits",
    "chi2_sf",
    "christoffersen_test",
    "exceedances",
    "kupiec_test",
    "norm_cdf",
    "norm_pdf",
    "norm_ppf",
    "parametric_var",
    "rolling_forecasts",
    "scenario_weighting",
    "split_by_tail",
    "standalone_parametric_var",
    "tail_estimate",
    "traffic_light",
]
