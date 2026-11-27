# Stress testing

What the book would lose in a set of named scenarios, what it would cost to cut risk in
stressed markets, what a short squeeze would do, and how far the top risks would have to
move to lose a given amount.

- **Scenario library:** 8 historical and 8 hypothetical scenarios, stored as factor-return
  shocks in [`data/scenarios.json`](data/scenarios.json).
- **Engine:** shocks applied to factor returns, multiplied by current exposures, plus a
  stressed residual for stock-specific losses.
- **Liquidity overlay:** the key scenarios repeated with stressed volume, spreads and
  volatility, with the cost of cutting gross by 30%.
- **Short squeeze:** crowded shorts up 30% and 50%, with borrow recalled and covered.
- **Reverse stress:** the most likely factor move that loses a target, and the move needed
  in each top factor and top position.
- **Reports by cadence:** daily liquidity, weekly or monthly stress, and ad hoc before a
  large trade or a macro event.

Status: tested on synthetic data and small hand-built cases only. It has not been run on
real data yet, and the historical scenarios are approximations (see below).

## Quick start

```python
from riskmodels.stress import (
    StressTester, StressConfig, default_library, liquidity_overlay, short_squeeze,
    reverse_stress, daily_liquidity_report, periodic_stress_report, ad_hoc_stress_report,
)

rep = StressTester().run(sized, rm, nav=250e6)      # whole library
rep.summary          # one row per scenario, worst first: factor, specific, long, short P&L
rep.by_factor        # scenario x factor P&L
rep.by_position      # scenario x name P&L
rep.breaches

ov = liquidity_overlay(rep, adv, 250e6, rm, market_cap=cap)
sq = short_squeeze(sized, adv, 250e6, short_data=finra_si, risk_model=rm)
rv = reverse_stress(sized, rm, target_loss=0.10, horizon_days=21, stress_report=rep)

daily = daily_liquidity_report(sized, 250e6, data=market_data, risk_model=rm, var_report=var)
weekly = periodic_stress_report(sized, rm, 250e6, data=market_data, short_data=finra_si)
pre = ad_hoc_stress_report(sized, rm, 250e6, trades=proposed_trades, data=market_data)
fomc = ad_hoc_stress_report(sized, rm, 250e6, tags=["rates"], reason="FOMC")
print(weekly.text())
```

## Scenario library

Each scenario holds cumulative factor returns over its horizon, keyed by the factor
model's names: `market` as a fraction, style factors per one standard deviation of
exposure, industries (`industry:<name>`) relative to the market. Industry keys use the
Fama-French 12 names the public loader assigns. A different industry scheme is mapped with
`StressConfig(industry_map={...})`; shocks naming factors the model doesn't have are
dropped and reported.

| Historical | Window | Hypothetical |
|---|---|---|
| Black Monday 1987 | Oct 14-19 1987 | Equity crash -20% (key) |
| LTCM / Russia 1998 | Jul 17-Aug 31 1998 | Rates up 100bp |
| Dot-com unwind | Mar 2000-Oct 2002 | Momentum reversal (key) |
| Quant quake (key) | Aug 6-9 2007 | Credit recession |
| GFC / Lehman (key) | Sep 12-Oct 10 2008 | Oil spike |
| Momentum crash 2009 | Mar 9-May 29 2009 | Tech de-rating |
| Covid crash (key) | Feb 19-Mar 23 2020 | Liquidity crunch (key) |
| Rate shock 2022 | Jan 3-Oct 12 2022 | Quant de-grossing (key) |

**The historical numbers are approximations, not measurements.** In each scenario the
market shock is a rounded move of the S&P 500 over the window (marked `basis: index`).
Style and industry shocks are marked `basis: judgment`: their signs follow the documented
episode (cited in each scenario's `source`), and their sizes are judgments. None of them
was measured with this factor model. The stress report adds an info row saying so.

To replace them with measured numbers:

- `calibrate_from_factor_returns(fit.factor_returns, start, end)` sums the model's own
  factor returns over a window, once the model has real data over it;
- `calibrate_from_reference(aqr_or_french_returns, start, end)` maps AQR or Ken French
  factor returns (MKT, SMB, HML Devil, UMD, QMJ, BAB) onto the model's factors, for
  windows before the model's history. The mapping (`DEFAULT_REFERENCE_MAP`) uses signs and
  rough scales; `scales_from_comparison` fits the scales from the factor model's
  benchmarking against AQR;
- `recalibrate_library` does both for every historical scenario and reports what changed.

Each scenario also carries stress multipliers (volume, spread, volatility) for the
liquidity overlay and the stressed residual, a horizon in trading days and a `key` flag for
the scenarios the overlay and reports repeat. Add scenarios with `ScenarioLibrary.add` or
by editing the JSON; `scaled` and `with_shocks` make variants.

## Design choices

**Factor P&L.** For book exposures x = X'w and shock f, factor P&L = x . f, split by
factor, by industry/style/market group, by position (w_i x X_i . f), by sector and by long
vs short.

**Stressed residual.** Stock-specific losses are added as
z x m x sqrt(H x sum w_i^2 d_i): the book's specific volatility over the horizon, scaled by
the scenario's volatility multiplier m and the 99% normal quantile z, always counted as a
loss and split across names by Euler allocation. H is capped at 21 days
(`max_specific_horizon_days`), so a two-year scenario like the dot-com unwind isn't
dominated by two years of diversifiable noise. This is a deliberately adverse residual,
not an expected one.

**Unshocked factors.** By default (`propagation="conditional"`) factors a scenario doesn't
name move by their expectation given the shocked ones under today's factor covariance,
F_us F_ss^-1 f_s, so a market crash also moves the factors correlated with it.
`propagation="zero"` leaves them at zero. Both shocked and derived values are shown.

**Liquidity overlay.** For each key scenario: apply the factor shock, so names and NAV move;
then measure liquidity and cost a 30% gross cut on the post-shock book with the scenario's
volume, spread and volatility multipliers. The cut's cost is restated in today's NAV and
added to the scenario loss, next to the same cut in normal markets.

**Short squeeze.** A short is crowded when short interest is at least 20% of float, days
to cover at least 5, or borrow fee at least 5% a year (FINRA short interest, point in
time on its publication date). With no short data every short is treated as crowded, and
the report says so. Crowded shorts rise 30% and 50%; lenders recall all of the borrow, to
be covered within 3 days at stressed spreads and volatility. Names needing over 25% of
daily volume to cover in time are flagged as buy-in risk. The report shows net and gross
after covering, since a forced cover leaves the book longer.

**Reverse stress.** Over the horizon, the most likely factor move that loses L is
f* = -L Σx / (x'Σx), at a distance of L / sqrt(x'Σx) standard deviations. For the factors
contributing most to risk, the move needed alone (others still, or others moving with it);
for the largest positions, the stock-specific move needed; for each library scenario, the
multiple of its shocks that would lose L. The default target is the stress loss limit.

**Limits.** Stress loss limit 10% of NAV with watch at 70% and alert at 80%; squeeze loss
limit 5% of NAV (`StressLimits`). Like the liquidity limits, these are examples to set
from the fund's own risk appetite and redemption terms.

**Reports by cadence.** The report functions return a `ScheduledReport` (headline numbers,
tables, one alert table across all models, and `text()` for an email or log). Nothing here
schedules itself: call `daily_liquidity_report` daily, `periodic_stress_report` weekly or
monthly, and `ad_hoc_stress_report` before a large trade (pass `trades` or `proposed` for a
before and after) or a macro event (pass `tags` or `scenarios`), from whatever scheduler
the fund uses.

## What is not modelled

- Non-linear instruments (options, convertibles): every position is linear in its return.
- Second-round effects inside a scenario (margin calls, other funds selling).
- Correlation breakdown beyond what the conditional propagation and the shocks imply.
