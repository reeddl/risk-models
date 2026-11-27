# Position sizing

Turns a list of names with a side and a conviction score into target weights, a trade
list, a risk report and breach alerts. All risk numbers (volatility, beta, factor
exposures, risk contributions) come from the factor model's `RiskModel`; nothing is
re-estimated here.

Status: tested on synthetic data and small hand-built cases only. It has not been run on
real data yet.

## Quick start

```python
from riskmodels.factor_model import FactorModel, FactorModelConfig
from riskmodels.position_sizing import PositionSizer, SizingConfig, cap_weighted_benchmark

fit = FactorModel(FactorModelConfig.daily()).fit(market_data)
rm = fit.risk_model()                      # risk forecast on the latest date
bench = cap_weighted_benchmark(fit)        # or your own benchmark weights

signals = pd.DataFrame({
    "side":        ["long", "short", ...],
    "conviction":  [5, 2, ...],
    "price":       [...],      # optional: shares in the trade list, trim rule
    "adv":         [...],      # optional: ADV in shares (or pass "adv_dollars")
    "entry_price": [...],      # optional: short trim rule
    "sector":      [...],      # optional: defaults to the factor model's industry
}, index=[...])

sizer = PositionSizer(SizingConfig())      # defaults below; override any field
out = sizer.size(signals, rm, nav=250e6, benchmark=bench, current=current_weights)

out.weights        # target weights, fractions of NAV
out.trades         # buy / sell / short / cover, weight, value, shares, % of ADV
out.breaches       # severity, kind, name, value, limit, message
out.risk_report()  # summary, by sector, by factor, by position
```

After a price move or a conviction change:

```python
current = drift_weights(current, todays_returns)   # weights after the move, no trading
mon = sizer.monitor(current, signals, rm, nav, benchmark=bench)
mon.breaches         # limits broken by the book as it stands, drift from target, trims
mon.resized.trades   # trades that bring it back
```

## The steps

| Step | What it does | Where |
|---|---|---|
| 1 Conviction | Score to multiplier, linear between anchors (default 1 = 0.5x, 3 = 1x, 5 = 1.5x). Scores outside the anchors are rejected, not clipped. | `budget.conviction_multiplier` |
| 2 Risk budget | `w = base_risk x multiplier / sigma`, one volatility choice for every name. | `budget.risk_budget_weights` |
| 3 Shorts | Shorts get `short_risk_multiplier` (default 0.6) on base risk, plus the trim rule. | `budget.short_trim_factor` |
| 4 Caps | Single name (6% long, 4% short), liquidity, sector gross and net, book gross and net. | `caps.apply_caps` |
| 5 Beta / factors | Scales the long or short book to the beta (or net) target, then flags style and industry bets. | `sizer.PositionSizer._solve`, `report.check_factors` |
| 6 Vol target | Scales the book so predicted vol equals the target (default 8%). | `sizer.PositionSizer._solve` |
| 7 Output | Trade list, risk report, breach alerts, and `monitor` for re-runs. | `report`, `sizer` |

Every default is a field on `SizingConfig`, documented in [`config.py`](config.py). The
defaults are the example numbers from the spec, not recommendations.

## Design choices

**Volatility: specific by default.** `volatility="specific"` sizes on the factor model's
idiosyncratic vol; `"total"` uses total vol. Specific is the default because a conviction
score is a view on the stock, and the market and style risk that total vol would add is
handled at the book level by the beta target and the factor checks. Sizing on total vol
instead makes high-beta names smaller, which double-counts once the book is beta-hedged.
Whichever you pick is used for every name.

**`base_risk` is per-position risk, not a weight.** 0.005 means a 1x name is sized so that
its own vol is 50bp of NAV a year. A 25%-vol name gets 2% of NAV before caps and scaling.
Because the vol target rescales the whole book in step 6, `base_risk` mostly sets the
starting point; what survives is the relative sizing between names.

**Trim rule: stepped, from entry.** For each full 10% a short has risen above its entry
price, its target is cut by 25% (`short_trim_step`, `short_trim_fraction`). A short up 25%
is sized at 0.75 x 0.75 = 56% of its untrimmed target. This is on top of the weight drift
you already have: a short that rises grows as a share of NAV, and the trade list brings it
back to the (smaller) target. `short_cover_move` sets an optional point where the target
goes to zero; it is off by default. Every trimmed or covered short is listed as an `info`
alert. Without `price` and `entry_price` the rule does nothing.

**Liquidity cap.** `|w| <= participation x ADV x days x price / NAV`, defaults 20% and 5
days. Pass ADV in shares with `price`, or `adv_dollars` directly. Names with no ADV get no
liquidity cap and a warning. `days_to_liquidate` is reported for every name.

**Caps only shrink, and are applied last.** Each cap scales positions down and never flips
a sign: clip names to their cap, then scale a sector down pro rata if its gross is over,
then scale down the heavier side of a sector whose net is over, then the book's gross, then
the heavier side of the book if net is outside its range. Shrinking one side of the book
can push a sector back over its net limit, so the pass repeats until nothing moves. Since
the caps are applied inside every evaluation of the search below, the output always meets
them. A `breach` in a sizing result therefore means the limits contradict each other (for
example a net floor that no shrinking can reach), not a solver miss.

**Beta and vol targets: one search, caps inside.** The weights are
`caps(k x (a_long(rho) x raw_long + a_short(rho) x raw_short))`, where `rho` is the
short-to-long balance and `k` the overall level. For a given level the sizer bisects on
`rho` until beta hits its target (or net, with `neutralize="net"`), then bisects on `k`
until predicted vol hits its target. Doing it this way rather than "beta, then vol, then
caps" in sequence matters because the steps undo each other: scaling up for vol pushes
names into their caps, which moves beta, and so on. An earlier version that looped the
three steps oscillated when caps were binding.

**When the caps make both targets impossible, beta wins.** If the book cannot reach the
vol target and the beta target together (typical when vol is set high relative to the
number of names and the caps), the beta target is kept and vol ends at the highest level
that still allows it. The vol miss is a `warning`. If vol cannot be reached even ignoring
beta, the book ends at the most the caps allow. Beta is the mandate and vol the scaling,
so this order seemed right; it is the one place you might want it the other way.

**Which book moves (`book_scaling`).** Only matters with `target_volatility=None`, because
the vol step sets the overall level otherwise. `"shrink"` (default) scales down whichever
side has the excess beta, so the target never adds gross. `"long"` or `"short"` always
moves that book, up or down.

**Beta is predicted, from the factor model.** `beta_i = (V b)_i / (b' V b)` with
`V = X F X' + D` and `b` the benchmark weights. `cap_weighted_benchmark(fit)` builds the
cap-weighted universe; pass your own benchmark weights or your own `betas` to override.

**Factor bets are flagged, not hedged.** After sizing, any style exposure beyond
+/-0.25 (in cross-sectional standard deviations), any industry net exposure beyond 10%, and
any single style or industry factor explaining more than 10% of predicted variance is a
`warning`. Hedging them away would change the names' relative sizes, which is your call.

**Trades.** `trade_list` covers every name in the target or current book. Held names not in
`signals` are exited in full. Trades under `min_trade` (5bp of NAV) are dropped, except full
exits. Crossing zero is labelled `sell+short` or `cover+buy`.

## Interface for later models

`SizingResult.weights` (a Series of fractions of NAV) is the hand-off. VaR/CVaR and stress
testing take it with the same `RiskModel`; the liquidity model can take it with the
`positions` table, which already carries ADV-based caps and days to liquidate.

## Known gaps

- Not run on real data. Expect real ADV, prices and sector labels to need format work.
- Sector defaults to the factor model's industry (EDGAR SIC buckets with the public loader),
  which is coarse. Pass a `sector` column for GICS-style sectors.
- The trim rule needs an entry price per short; partial fills and adds are not tracked, so
  "entry" is whatever price you pass.
- No transaction costs or turnover penalty; the trade list moves straight to target.
- Speed: about 0.2-0.5s for 60-80 names when caps bind, mostly in the cap pass. Fine for
  re-running after each move; a few hundred names with many sectors will be slower.
