# risk-models

A collection of portfolio risk models for a long/short equity book, built as one Python
package so every model shares the same data contracts and estimation code.

| Model | Folder | Status |
|---|---|---|
| Fundamental equity factor model (Barra/Axioma style) | [`src/riskmodels/factor_model`](src/riskmodels/factor_model/README.md) | v0.1 |
| Position sizing (conviction, risk budget, caps, beta and vol targets) | [`src/riskmodels/position_sizing`](src/riskmodels/position_sizing/README.md) | v0.1 |
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
  sources/         public data loaders (Yahoo, EDGAR, FRED, AQR, FINRA)
  factor_model/    one folder per model, each with its own README
  position_sizing/ conviction scores to capped, beta- and vol-targeted weights
tests/
  core/            tests for the shared code
  factor_model/    tests for the factor model
  position_sizing/ tests for the sizing model
  sources/         offline tests for the data loaders
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

`riskmodels.sources.PublicDataLoader` builds the bundle from free public data: Yahoo
Finance prices, SEC EDGAR fundamentals, SIC industries and filer history, FRED rates, AQR
benchmark factors and FINRA short interest. It approximates the contract above; what stays
biased (mainly delisted names, which Yahoo does not carry) is in
[`sources/README.md`](src/riskmodels/sources/README.md). The test suite uses
`riskmodels.core.synthetic.generate_market_data` and recorded-format sample responses, so it
runs offline.

## Development

```
pip install -e ".[dev]"
pytest
```

Dependencies are numpy and pandas only. Reading AQR's Excel files needs the optional
`aqr` extra (`pip install -e ".[aqr]"`, which adds openpyxl).
