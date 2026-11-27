# Liquidity risk

How long it would take to get out of the book, and what it would cost.

- **Days to liquidate** for every position, at a fixed share of ADV per day.
- **Buckets:** under 1 day, 1-5, 5-10, over 10 days, plus a "no ADV" bucket.
- **Liquidation profile:** the share of gross (and of NAV) that can be exited in 1, 5, 10
  and 20 days, and the days needed to exit 50%, 90% and 100% of gross.
- **Limits:** a minimum share liquidatable within a horizon, a cap on position size
  relative to ADV, and a cap on the share of gross in the slowest bucket.
- **Cost of cutting gross** by a target (30% by default), pro rata or fastest names first,
  with half spread plus square-root market impact.

Status: tested on synthetic data and small hand-built cases only. It has not been run on
real data yet.

## Quick start

```python
from riskmodels.liquidity import LiquidityModel, LiquidityConfig, adv_from_data, market_cap_from_data

adv = adv_from_data(market_data, as_of)          # 20-day median dollar volume
cap = market_cap_from_data(market_data, as_of)   # for spread tiers
rep = LiquidityModel().run(sized, adv, nav=250e6, market_cap=cap, risk_model=rm)

rep.summary      # avg / max days to liquidate, liquidatable shares, cost of a 30% cut
rep.buckets      # gross, long, short per bucket
rep.profile      # share of gross and NAV exitable per horizon
rep.positions    # per name: weight, value, ADV, %ADV, days, bucket, spread, cost to exit
rep.reduction    # the gross cut: per-name trades, days and cost
rep.breaches     # same columns as the VaR and sizing alerts
```

`riskmodels.stress.daily_liquidity_report` wraps this for the daily report and stacks the
VaR and sizing alerts with it.

## Design choices

**Execution model.** Every name trades at `participation` of its ADV per day (20% by
default, the same rate the sizing model uses for its ADV cap), all names in parallel.
Days to liquidate = |w| x NAV / (participation x ADV). Longs and shorts are treated alike.
Over t days the book can exit sum_i min(|w_i|, participation x ADV_i x t / NAV).

**ADV.** 20-day median dollar volume by default (`adv_method="mean"` is available). The
median keeps one block trade or index rebalance from overstating liquidity. Names with
fewer than 5 observations get no ADV and land in the "no ADV" bucket, counted as
unliquidatable, with a warning.

**Trading cost.** Per unit traded, half the quoted spread plus Y x sigma x sqrt(Q / ADV),
the square-root impact law, with Y = 1 (`impact_coefficient`). Spreads come from the
caller where known; otherwise from market-cap tiers (4 bps above $10bn, 10 bps above $2bn,
25 bps above $300m, 60 bps below), or 25 bps with no cap. Daily volatility comes from the
factor model's total risk. These defaults are rough and should be fitted to the fund's own
execution data.

**Gross reduction.** `pro_rata` cuts every position by the same fraction, so the book's
shape is kept and the time is set by the slowest name. `fastest` cuts the most liquid
names first, which is quicker and cheaper but leaves a less liquid, differently shaped
book. Names without ADV can't be cut; the achieved cut is reported next to the target.

**Limits.** The defaults are examples, not recommendations:

| Limit | Default | Config |
|---|---|---|
| Minimum liquidatable | 80% in 5 days | `LiquidityLimits.min_liquidatable={5: 0.80}` |
| Basis | gross | `LiquidityLimits.basis="gross"` or `"nav"` |
| Max position | 1.0x ADV | `LiquidityLimits.max_position_adv` |
| Max share in slowest bucket | 10% of gross | `LiquidityLimits.max_slow_share` |
| Alert level | watch at 80% of a limit | `LiquidityLimits.alert_levels` |

A position exactly at a limit is allowed; above it is a breach.

**Set the limits from the fund's redemption terms.** The horizon and share should match
what investors can take out and how fast. A fund with monthly liquidity, 30 days' notice
and a 25% investor-level gate needs to raise at most 25% of NAV within roughly a month;
a common rule is to hold enough liquidity to meet the largest plausible redemption
inside the notice period with room to spare, and to keep the position-vs-ADV cap tight
enough that no single name blocks it. Quarterly terms with a long notice period can run
a less liquid book. Pass the fund's own numbers, for example
`LiquidityLimits(min_liquidatable={5: 0.50, 20: 0.90})`.

**Why the default basis is gross.** The spec asks for a share of NAV. For a levered
long/short book, "X% of NAV liquidatable" is easy to meet even when much of the book is
stuck (a 200% gross book meets 80% of NAV with 60% of its gross unsellable). The share of
gross measures how much of the book can actually be unwound, so it is the default.
`basis="nav"` gives the NAV reading.

## What is not modelled

- Volume that dries up as you trade, or other funds selling the same names at the same
  time (use the stressed volumes in the stress overlay for that).
- Intraday timing, auctions, block trades and dark liquidity.
- Borrow recalls (see the short squeeze test in `riskmodels.stress`).
