# risk-models

A collection of portfolio risk models for a long/short equity book, built as one Python
package so every model shares the same data contracts and estimation code.

| Model | Folder | Status |
|---|---|---|
| Fundamental equity factor model (Barra/Axioma style) | [`src/riskmodels/factor_model`](src/riskmodels/factor_model/README.md) | v0.1 |
| Position sizing | `src/riskmodels/position_sizing` | planned |
| VaR / CVaR | `src/riskmodels/var` | planned |
| Liquidity risk | `src/riskmodels/liquidity` | planned |
| Stress testing | `src/riskmodels/stress` | planned |

## Layout

```
src/riskmodels/
  core/            shared by every model
    data.py          MarketData bundle, DataLoader protocol, PIT/survivorship validation
    pit.py           point-in-time fundamentals and effective-dated industries
    preprocessing.py winsorize, standardize, fill, orthogonalize
    covariance.py    EWMA, Newey-West, correlation shrinkage, PSD repair
    synthetic.py     synthetic market with a known factor structure (tests, demos)
  factor_model/    one folder per model, each with its own README
tests/
  core/            tests for the shared code
  factor_model/    tests for the factor model
```

Why one package with a folder per model rather than independent projects: the later
models are consumers of the factor model. Position sizing needs the asset covariance,
VaR/CVaR needs factor and specific volatilities, stress testing shocks factor returns
through the exposures, and liquidity reuses the same point-in-time universe and volume
data. Keeping them in one package means one data loader, one definition of "point in
time", and one covariance toolkit, and each model imports `riskmodels.factor_model`
instead of copying it.

## Data

Models consume a `riskmodels.core.MarketData` bundle. A data source (CRSP/Compustat,
a vendor feed, your own files) plugs in by implementing `DataLoader.load(start, end)`.
The contract every loader must meet is documented at the top of
[`core/data.py`](src/riskmodels/core/data.py) and checked by `MarketData.validate()`:

- **Survivorship-free:** every security that traded in the window, including delisted
  names, with the delisting return included on the last day.
- **Point-in-time fundamentals:** long format with the date each number became public
  (`available_date`); restatements are new rows, originals are kept.
- **Effective-dated industries:** a reclassification is only visible from its date.

No real data loader ships yet. The test suite and examples use
`riskmodels.core.synthetic.generate_market_data`.

## Development

```
pip install -e ".[dev]"
pytest
```

Dependencies are numpy and pandas only.
