# Public data sources

`PublicDataLoader` builds a `MarketData` bundle from free public data only:

| Data | Source | Module |
|---|---|---|
| Prices, total returns, volume, splits, dividends | Yahoo Finance chart API | `yahoo.py` |
| Fundamentals (point in time) and shares outstanding | SEC EDGAR XBRL company facts | `edgar.py` |
| Industries | SEC EDGAR SIC codes, mapped to Fama-French 12 (or SIC 2-digit / divisions) | `edgar.py` |
| Universe | SEC EDGAR quarterly filing indexes (`master.idx`) | `edgar.py` |
| Risk-free rate, macro and rates | FRED (`fredgraph.csv`, default DTB3) | `fred.py` |
| Benchmark factor returns | AQR data library workbooks | `aqr.py`, `factor_model/benchmark.py` |
| Short interest | FINRA consolidated short interest | `finra.py` |

```python
from riskmodels.sources import PublicDataLoader
from riskmodels.factor_model import FactorModel, FactorModelConfig

loader = PublicDataLoader(limit=300)          # try a few hundred names first
data = loader.load("2018-01-01", "2024-12-31")
print(loader.report.summary())                # what was found and what was left out
result = FactorModel(FactorModelConfig.daily()).fit(data)
```

Start the window at least 147 trading days before the first date you want factor returns
for: that is the descriptor warm-up.

## Being a good client

Every request goes through `HttpClient`, which caches responses on disk
(`$RISKMODELS_CACHE`, default `~/.cache/riskmodels`), spaces requests per host and retries
429 and 5xx responses with backoff. The SEC limit is 10 requests a second; the client
stays under 9.

The SEC asks automated clients to declare who they are in the User-Agent, ideally with a
contact email. The default here is a generic name. Before a full pull, set your own:

```
export RISKMODELS_USER_AGENT="Your Name you@example.com"
```

A full universe is several thousand companies and three to four requests each, so the
first run takes a while (roughly an hour) and later runs come from the cache.

## How close this gets to the data contract

The models assume a survivorship-free universe, point-in-time fundamentals and
effective-dated industries (`core/data.py`). With these sources:

**Universe: survivorship-free list, survivor-biased prices.** The list of names comes from
every company that filed a 10-K or 10-Q in the window, alive or dead, so the *universe* is
survivorship-free. But prices come from Yahoo, which mostly drops delisted symbols and
keys everything by today's ticker. A company that was acquired or went bankrupt usually
has no current ticker in EDGAR and no Yahoo history, so it is excluded and listed in
`loader.report.excluded` with the reason. What that leaves:

- The loaded universe is close to "companies still trading today", plus whatever dead names
  Yahoo still serves. Expect this to overstate the returns of small, distressed and value
  names and understate the volatility of the size and value factors, most for windows that
  start many years back.
- Ticker reuse: a ticker that changed hands maps to its current company. EDGAR's CIK is
  the identity here, so the fundamentals and SIC are right for the current holder, but the
  older part of a reused ticker's Yahoo history can belong to someone else.
- No delisting returns. Pass `delisting_return=-0.3` (a common assumption for performance
  delistings) to apply one on the last day of names whose filings show a Form 25 or 15;
  the default is none.
- `ticker_overrides={cik: "SYMBOL"}` lets you add back a dead company you can still price.

Fixing this properly needs a source with delisted prices and delisting returns (CRSP,
or a vendor feed). The loader interface stays the same.

**Fundamentals: point in time.** Every XBRL fact keeps the accession of the filing it came
from and becomes available at that filing's SEC acceptance time (same day if accepted
before 16:00, else next business day). Restated comparatives in later filings are kept as
new versions, so the model only sees a restatement once it was filed. Remaining gaps:

- XBRL starts in 2009 to 2011 depending on company size, so fundamental factors have no
  data before then.
- Concept choice varies by company. The mapping (`FLOW_CONCEPTS`, `INSTANT_CONCEPTS`,
  `DEBT_CONCEPTS` in `edgar.py`) covers the common tags; some companies, banks and insurers
  above all, use others and will show missing values, which the model fills with the
  industry average.
- Total debt is long-term debt plus short-term borrowings and commercial paper; leases are
  not included.
- Shares outstanding come from the 10-K/10-Q cover page, so market cap moves with each
  filing rather than daily buybacks or issuance.

**Industries: not point in time.** EDGAR only gives each company's current SIC code, so
every company is in its current industry for its whole history. Reclassifications are rare
but this does leak a little. SIC is also coarser and older than GICS; Fama-French 12 is the
default to keep each industry well populated.

**Prices.** Yahoo's adjusted close is used for total returns and its split-adjusted close
times split-rebased share counts for market cap. Yahoo occasionally revises history and has
gaps for thin names; names with fewer than `min_price_days` prices are excluded.

**FRED.** `fredgraph.csv` returns the latest vintage. That is fine for T-bill rates, which
are not revised, but revised macro series are not point in time.

**AQR.** Download links change when AQR updates its site, so pass the workbook URL from
AQR's data-set page, or a local copy. Reading .xlsx needs `pip install -e ".[aqr]"`.
AQR's factors are portfolios, ours are regression factor returns, so expect correlation,
not equality. `compare_factor_returns` reports correlation, vols and slope per pair.

**FINRA.** Short interest is stamped available 8 business days after settlement. It is not
one of the factor model's factors; it is there for the liquidity model and as an optional
descriptor.

## Not yet run against the live sources

These loaders were written in an environment that could not reach Yahoo, the SEC, FRED,
AQR or FINRA. The parsers are tested on hand-built responses in the documented formats
(`tests/sources`), but the first live run may turn up format details to fix. FINRA's API
in particular may need credentials for this data set.
