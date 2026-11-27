# Fundamental equity factor model

A Barra USE4 / Axioma-style model that splits each stock's risk into a factor part,
`X F X'`, and a specific part, `D`.

```python
from riskmodels.factor_model import FactorModel, FactorModelConfig

result = FactorModel(FactorModelConfig.daily()).fit(market_data)   # or .weekly()
risk = result.risk_model()                     # latest date; risk_model(as_of=...) for history
report = risk.portfolio_risk(weights)          # weights: Series of capital weights, L/S allowed
report.summary()       # total, factor and specific risk (annualized vol) and variance shares
report.by_factor       # exposure, factor vol, marginal and contribution per factor
report.by_position     # marginal and contribution per position, split factor vs specific
report.by_group        # market / industry / style / specific
```

`result.factor_returns` and `result.specific_returns` are the regression outputs;
`risk.factor_covariance`, `risk.specific_variance` and `risk.asset_covariance()` are the
forecasts the later models build on.

## Factors

- **Market:** exposure 1 for every stock.
- **Industries:** 0/1 dummies from the effective-dated classification.
- **Styles** (each the weighted average of standardized descriptors):

| Factor | Descriptors (weights) |
|---|---|
| size | log market cap |
| beta | EWMA beta to the cap-weighted market, 252d window, 63d half-life |
| momentum | EWMA sum of log excess returns, 252d window skipping the last 21d, 126d half-life |
| residual_volatility | DASTD 0.74, CMRA 0.16, HSIGMA 0.10; **orthogonalized to beta and size** |
| value | book/price 0.5, earnings/price 0.5 |
| quality | ROE 0.5, gross profit/assets 0.5 |
| growth | YoY TTM sales growth 0.5, YoY TTM earnings growth 0.5 |
| leverage | (cap + debt)/cap 0.5, debt/assets 0.5 |
| liquidity | log turnover over 1m 0.35, 3m 0.35, 12m 0.30; **orthogonalized to size** |
| dividend_yield | TTM dividends / cap |

Growth uses `(now - year ago) / mean(|now|, |year ago|)`, which is bounded and defined when
earnings are negative. "Year ago" is the period about 365 days (±45) before the latest
reported period, as known on the exposure date.

## Estimation, step by step

1. **Coverage:** on each date, stocks that are listed, have a positive cap and an
   industry. Delisted names drop out after their last day and stay in history.
2. **Descriptors:** computed only from data known at that close (prices through t,
   fundamentals with `available_date <= t`, ignored if the period ended more than 18 months
   earlier).
3. **Cleaning:** gross errors clipped at median ± 10 robust std, then winsorized at
   cap-weighted mean ± 3 equal-weighted std, repeated until stable; then z-scored with a
   **cap-weighted mean and equal-weighted std**.
4. **Composites:** weighted average over the descriptors a stock has, re-standardized.
5. **Missing values:** filled with the cap-weighted mean exposure of the stock's
   industry, or 0 (market-neutral) if the whole industry is missing.
6. **Orthogonalization:** sqrt-cap weighted regression residual, then re-standardized.
7. **Regression:** each period (daily, or weekly with compounded returns), excess returns
   over (t0, t1] on exposures at t0, weighted by **sqrt(market cap)**, subject to the
   cap-weighted industry returns summing to zero. Solved exactly via the KKT system, so no
   industry is dropped and the market factor is the cap-weighted market. Outputs factor
   returns, specific returns for every covered stock, and weighted R².
8. **Factor covariance:** EWMA with a short half-life for volatilities and a long one for
   correlations, each with a Newey-West (Bartlett) correction for autocorrelation,
   combined as `D C D`; correlations shrunk toward the identity; eigenvalues floored so
   the matrix is positive definite.
9. **Specific risk:** EWMA of squared residuals with a Newey-West term (floored at half the
   lag-0 variance), then Bayesian shrinkage toward the cap-weighted average of the stock's
   **size decile** (USE4 form). Stocks with fewer than the minimum observations take the
   bucket average.
10. **Portfolio risk:** total, factor (`x'Fx`) and specific (`Σ h²d`) risk, with Euler
    contributions that add up exactly, by position and by factor, plus marginal risk.

## Defaults

All live in `FactorModelConfig`. Covariance settings are in regression periods.

| Setting | Daily | Weekly |
|---|---|---|
| Factor vol half-life | 84 | 26 |
| Factor correlation half-life | 504 | 104 |
| Newey-West lags (vol / corr) | 5 / 2 | 2 / 1 |
| Correlation shrinkage | 10% toward identity | same |
| Specific risk half-life | 84 | 26 |
| Specific Newey-West lags | 5 | 1 |
| Specific min observations | 63 | 26 |
| Size buckets / shrinkage q | 10 / 0.1 | same |
| Winsorization | ±3 std | same |
| Annualization | 252 | 52 |

Descriptor windows are in trading days for both frequencies. Warm-up before the first
regression is 147 trading days (momentum's minimum observations plus its 21-day skip).

## Not yet included

Things Barra and Axioma add on top of this that are worth considering next: eigenfactor
risk adjustment and a volatility regime multiplier for the factor covariance, structural
specific-risk models for thin histories, a nonlinear size factor, thin-industry priors,
and country/currency factors for a global universe.
