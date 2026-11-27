# VaR / CVaR

Daily Value at Risk and Conditional VaR (expected shortfall) for the long/short book, on
top of the factor model and the position sizing output.

- **Primary number:** parametric VaR from the factor model's covariance
  V = X F X' + D, so VaR agrees with every other risk number in the framework.
- **Cross-check:** historical simulation of today's positions on past returns, EWMA
  weighted. A large gap between the two methods raises a warning.
- **CVaR:** taken from the historical distribution. A normal CVaR is shown next to it
  for reference.
- **Breakdown:** component VaR (and CVaR) by position, sector, factor and long vs short.
- **Limits:** a VaR limit as % of NAV with alert levels.
- **Backtests:** exceedance count, Kupiec, Christoffersen, a traffic light, on
  hypothetical and actual P&L.

Status: tested on synthetic data and small hand-built cases only. It has not been run on
real data yet.

## Quick start

```python
from riskmodels.factor_model import FactorModel, FactorModelConfig
from riskmodels.position_sizing import PositionSizer
from riskmodels.var import VaRModel, VaRConfig, VaRLimits, rolling_forecasts, backtest_report

fit = FactorModel(FactorModelConfig.daily(factor_cov_window=756)).fit(market_data)
rm = fit.risk_model()
sized = PositionSizer().size(signals, rm, nav=250e6, benchmark=bench)

vm = VaRModel(VaRConfig(limits=VaRLimits(var_limit=0.02)))
rep = vm.run(sized, fit, market_data, nav=250e6, risk_model=rm)   # or a weight Series

rep.summary        # 1-day and 10-day VaR / CVaR, both methods, gap, limit utilization
rep.horizons       # the same as a table
rep.by_position    # component VaR per name: parametric (factor / specific), historical, CVaR
rep.by_sector      # long, short, net, gross, component VaR and CVaR
rep.by_factor      # component VaR per factor plus a specific line
rep.by_side        # long vs short: component and standalone VaR, diversification
rep.breaches       # breach / alert / watch / warning / info rows
rep.to_amount(rep.by_sector)   # any table in currency

# Backtest: today's book replayed over the last year, plus your actual P&L
fc = rolling_forecasts(fit, market_data, sized.weights, start="2025-11-27")
backtest_report(fc, actual_pnl=actual_pnl_pct_nav)
```

All numbers are fractions of NAV, with losses positive (0.012 = a 1.2% of NAV loss).

## Parameters

| Setting | Default | Where |
|---|---|---|
| Confidence | 99% | `VaRConfig.confidence` |
| Base horizon | 1 day | `VaRConfig.horizon_days` (only 1 is estimated) |
| Longer horizon | 10 days, by √10 scaling | `VaRConfig.scaled_horizon_days` |
| History | 756 days (about 3 years; 504 for 2) | `VaRConfig.history_days` |
| Historical weighting | EWMA age weights, half-life 252 days | `hs_weighting`, `hs_half_life` |
| Parametric covariance | the factor model's EWMA (vol half-life 84, correlation 504) | `FactorModelConfig` |
| VaR limit | 2% of NAV, 1-day 99% | `VaRLimits.var_limit` |
| Alert levels | watch at 70%, alert at 80% utilization | `VaRLimits.alert_levels` |
| Limit held against | parametric VaR | `VaRLimits.limit_basis` (`parametric`, `historical`, `max`) |
| CVaR limit | off | `VaRLimits.cvar_limit` |
| Method gap warning | historical vs parametric differ by more than 25% | `VaRLimits.max_method_gap` |

The 2% limit and the 70-80% alert levels are illustrative. Set them from your risk
appetite.

**The 10-day figure is an approximation.** It is the 1-day number times √10, which
assumes returns are independent and identically distributed over the 10 days and that the
book does not change. Autocorrelation, volatility clustering and trading all break that.

To use the same 2-3 year window for the parametric number, set the factor model's
`factor_cov_window` (756 or 504). Its default uses all history, where the EWMA half-lives
already put most of the weight on the last year or two.

## Design choices

### Parametric VaR

VaR = z × σ, with σ the book's 1-day predicted volatility from the factor model
(annualized volatility / √252) and z the normal quantile (2.326 at 99%). Daily returns are
treated as zero-mean, as in the factor model.

Component VaR is the Euler split: each position's VaR is z/√252 times its contribution to
volatility, h_i (Vh)_i / σ, so the parts add up exactly to VaR. Each position is also split
into a factor part and a specific part. Factor components are x_k (Fx)_k / σ scaled the
same way, plus one `specific` line. Sector and long/short components are sums of position
components. `marginal_var` is dVaR/dh_i: the VaR added per unit of extra weight.

### Historical simulation

Each scenario is one past day: today's weights times that day's excess returns
(return minus the risk-free rate, the same as the factor model). Today's book is held
fixed, so this is the same hypothetical P&L the backtest uses.

When a name has no return on a past day (it had not listed, or did not trade), its return
is filled with the factor model's proxy, X_i f_t: today's exposures times that day's factor
returns, with zero specific return. The share of gross-weighted position-days filled this
way is in `summary["proxied_share"]` and in an `info` row.

### How EWMA weighting applies to historical simulation

Plain historical simulation gives every day in the window equal weight, so a calm year
and a volatile month count the same. Two ways to make recent volatility count more are
available:

- **`"age"` (default):** EWMA probability weights (Boudoukh, Richardson and Whitelaw).
  The newest day has the most weight and a day one half-life back (252 days by default)
  has half as much. VaR is the weighted quantile and CVaR the weighted mean beyond it.
  Scenario returns are left as they were.
- **`"volatility"`:** Hull-White filtered simulation. Every day has equal weight, but its
  P&L is rescaled by today's EWMA volatility of the book divided by the EWMA volatility
  known before that day (half-life `vol_half_life`, 21 days). Old crashes are kept but
  scaled to today's volatility level.
- **`"none"`:** equal weights.

With age weighting, the half-life sets a trade-off: a short half-life reacts faster but
leaves fewer effective scenarios for a 1% tail. `summary["effective_scenarios"]` reports
the effective sample size (about 565 of 756 at the defaults, which leaves roughly 6 effective
days beyond 99% VaR).

VaR is the loss at which the cumulative probability of equal-or-worse days first reaches
1%. CVaR is the probability-weighted mean of the worst 1%, with the boundary day counted
fractionally so CVaR is exact for any weights.

### How component VaR is split for historical vs parametric

- **Parametric:** the Euler split above, exact.
- **Historical CVaR:** each component's expected loss in the tail scenarios, using the same
  weights that define CVaR. These add up exactly to CVaR and are the most stable
  historical breakdown.
- **Historical VaR:** VaR is a single quantile, so the loss of each component on the one
  VaR day is very noisy. Instead, each component's loss is averaged over the scenarios
  whose probability lies within `component_var_band` (0.5%) of the quantile (with 756 days,
  roughly the 4th to 11th worst), and the parts are rescaled so they add up to historical
  VaR. This is a smoothed version of Hallerbach's scenario method.
- **By factor:** each scenario is split with today's factor exposures: factor k contributes
  x_k f_kt, and the rest of the scenario's P&L is the specific line.
- **Long vs short:** components sum the positions on each side. `by_side` also gives each
  side's standalone VaR (the long book or the short book alone) and the diversification
  between them (standalone long + standalone short − book VaR).

### Limits and alerts

`var_utilization` is the VaR held against the limit divided by the limit. Rows in
`breaches`:

| Severity | When |
|---|---|
| `breach` | utilization at or above 100% (VaR, or CVaR when a CVaR limit is set) |
| `alert` | at or above 80% |
| `watch` | at or above 70% |
| `warning` | historical and parametric VaR differ by more than 25%, or fewer than 250 scenarios |
| `info` | names filled with the factor proxy |

The columns match the position sizing alerts, so the two can be stacked in one report.

### How hypothetical and actual P&L are fed in

A VaR forecast made at the close of day t is compared with the P&L of day t+1. All
backtest series are indexed by the P&L date.

- **Hypothetical P&L** is built by `rolling_forecasts`: the book held at the close of t,
  revalued on day t+1's excess returns, with no trading. Pass a weight Series to replay
  today's positions over the past, or a date × name DataFrame for the book as it stood
  each day (the last row on or before t is used). Each forecast uses only information up
  to t: the factor model's risk forecast as of t and historical scenarios ending at t.
  Names the model does not cover on t are dropped and counted.
- **Actual P&L** comes from your own books, as a Series of daily P&L in fractions of the
  prior day's NAV. Pass it to `backtest_report(forecasts, actual_pnl=...)`.

Test both. Hypothetical P&L tests the model alone. Actual P&L includes intraday trading,
fees and financing, so an exceedance on actual P&L that does not show on hypothetical
P&L points at trading rather than the model.

### Backtests

- **Exceedances:** at 99% over 250 days, about 2.5 are expected.
- **Kupiec:** tests whether the exceedance rate equals 1%. Likelihood ratio, χ²(1).
- **Christoffersen:** tests whether exceedances are independent or clustered
  (χ²(1)), and the joint conditional-coverage test of rate and independence (χ²(2)).
- **Traffic light:** Basel-style zone from the binomial probability. At 99% over 250
  days: green 0-4, yellow 5-9, red 10 or more.
- **CVaR diagnostic:** mean loss on exceedance days divided by the mean CVaR forecast on
  those days (`cvar_ratio`, near 1 is good). It is not a formal test.

`passed` is true when neither Kupiec nor conditional coverage rejects at 5%. If it fails,
revisit the parameters: the history window, the EWMA half-lives, the factor model's
covariance settings, or the limit basis (`"max"` holds the higher of the two methods
against the limit).

The chi-square, normal and binomial functions are implemented in `stats.py`, so the
package still depends on numpy and pandas only.

## Known limits

- **VaR says nothing about losses beyond the threshold.** It is the loss exceeded on 1%
  of days, not how bad those days are. CVaR helps, but it is harder to backtest: the
  `cvar_ratio` above is a diagnostic, not a test with a known distribution.
- **It assumes you can exit at current prices, so it ignores liquidity.** A position
  that takes days to unwind can lose more than its 1-day VaR. The liquidity and stress
  modules cover that.
- **L/S funds have short-squeeze risk that volatility-based VaR captures poorly.** A
  crowded short can gap up far beyond anything its volatility implies, and the
  historical window may hold no squeeze for that name at all. Stress testing should pick
  that up.
- **Parametric VaR is normal.** Fat tails and skew show up only in the historical
  cross-check; that is why the method gap is monitored.
- **The 10-day figure is a √10 approximation** (see above).
- **Historical scenarios inherit the data's gaps.** Names with short histories are partly
  factor proxies, and the data sources' known biases (see
  [`sources/README.md`](../sources/README.md)) carry through.
- **The book is held fixed** over the horizon in every number here.

## Reuse by the liquidity and stress modules

- `HistoricalScenarios` holds asset return scenarios and splits any book's P&L by
  position or by factor. `HistoricalScenarios.from_returns` wraps any return panel,
  for example stressed or simulated returns.
- `tail_estimate` and `split_by_tail` give VaR, CVaR and their component splits for any
  weighted P&L distribution.
- `parametric_var` and `standalone_parametric_var` take any `RiskModel`, so a stressed
  covariance plugs straight in.
- `check_var_limits` and `breach_frame` produce alert rows in the shared format.
