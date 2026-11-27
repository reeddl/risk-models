"""Synthetic market data with a known factor structure, for tests and demos.

The generated bundle obeys the same contract as real data: names list and delist over the
window (with a delisting return), some names change industry, and fundamentals arrive with
a reporting lag and occasional restatements. Returns follow

    r_it = beta_i * m_t + g_{ind(i,t),t} + sum_k z_ik f_kt + e_it

where z are latent size, value and quality traits that the fundamentals encode, so a model
built on the fundamentals should recover the true factor returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import MarketData


@dataclass
class SyntheticTruth:
    market: pd.Series                 # true market factor return m_t
    industry_returns: pd.DataFrame    # true g_{k,t}
    style_returns: pd.DataFrame       # true f_{k,t} for size, value, quality
    traits: pd.DataFrame              # latent traits per asset
    beta: pd.Series                   # true market beta
    specific_vol: pd.Series           # true daily specific volatility


def generate_market_data(
    n_assets: int = 200,
    n_days: int = 520,
    n_industries: int = 6,
    seed: int = 0,
    start: str = "2020-01-01",
    birth_fraction: float = 0.15,
    death_fraction: float = 0.15,
    reclass_fraction: float = 0.03,
    restatement_fraction: float = 0.05,
) -> tuple[MarketData, SyntheticTruth]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    assets = pd.Index([f"A{i:04d}" for i in range(n_assets)], name="asset")
    industries = [f"IND{k}" for k in range(n_industries)]

    # Listing windows: most names exist throughout, some IPO late and some die early.
    start_idx = np.zeros(n_assets, dtype=int)
    born = rng.random(n_assets) < birth_fraction
    start_idx[born] = rng.integers(60, max(61, n_days - 160), born.sum())
    end_idx = np.full(n_assets, n_days - 1)
    dies = (rng.random(n_assets) < death_fraction) & (start_idx < n_days - 160)
    for i in np.where(dies)[0]:
        end_idx[i] = rng.integers(start_idx[i] + 120, n_days - 20)
    still_listed = end_idx == n_days - 1

    traits = pd.DataFrame(
        rng.standard_normal((n_assets, 4)),
        index=assets,
        columns=["value", "quality", "growth", "leverage"],
    )
    payer = rng.random(n_assets) < 0.6
    beta = pd.Series(np.clip(1.0 + 0.3 * rng.standard_normal(n_assets), 0.3, 2.0), index=assets)
    log_cap0 = np.log(5e9) + 1.2 * rng.standard_normal(n_assets)
    size_z = (log_cap0 - log_cap0.mean()) / log_cap0.std()
    spec_vol = pd.Series(0.018 * np.exp(-0.25 * size_z), index=assets)

    # Industry assignments, with a few reclassifications.
    ind0 = rng.integers(0, n_industries, n_assets)
    ind_rows = []
    ind_idx = np.tile(ind0, (n_days, 1))
    for i in range(n_assets):
        a, s, e = assets[i], start_idx[i], end_idx[i]
        if rng.random() < reclass_fraction and e - s > 100:
            cut = int(rng.integers(s + 40, e - 40))
            new = (ind0[i] + 1 + rng.integers(0, n_industries - 1)) % n_industries
            ind_rows.append((a, industries[ind0[i]], dates[s], dates[cut - 1]))
            ind_rows.append((a, industries[new], dates[cut], pd.NaT))
            ind_idx[cut:, i] = new
        else:
            ind_rows.append((a, industries[ind0[i]], dates[s], pd.NaT))
    industries_df = pd.DataFrame(ind_rows, columns=["asset", "industry", "effective_from", "effective_to"])

    # Factor returns.
    m = 0.0003 + 0.010 * rng.standard_normal(n_days)
    g = 0.006 * rng.standard_normal((n_days, n_industries))
    style_names = ["size", "value", "quality"]
    f = np.column_stack([
        -0.0001 + 0.0025 * rng.standard_normal(n_days),
        0.0001 + 0.0030 * rng.standard_normal(n_days),
        0.0001 + 0.0020 * rng.standard_normal(n_days),
    ])
    Z = np.column_stack([size_z, traits["value"], traits["quality"]])

    eps = rng.standard_t(6, (n_days, n_assets)) * np.sqrt(4 / 6) * spec_vol.to_numpy()[None, :]
    ind_ret = np.take_along_axis(g, ind_idx, axis=1)
    R = beta.to_numpy()[None, :] * m[:, None] + ind_ret + f @ Z.T + eps

    alive = np.zeros((n_days, n_assets), dtype=bool)
    for i in range(n_assets):
        alive[start_idx[i]:end_idx[i] + 1, i] = True
    R = np.where(alive, R, np.nan)
    for i in np.where(~still_listed)[0]:
        e = end_idx[i]
        R[e, i] = (1 + R[e, i]) * (1 - 0.3) - 1  # delisting return
    R = np.clip(R, -0.95, None)

    growth = np.where(alive, np.nan_to_num(np.log1p(R)), 0.0)
    log_cap = log_cap0[None, :] + np.cumsum(growth, axis=0)
    cap = np.where(alive, np.exp(log_cap), np.nan)

    base_turn = np.exp(np.log(0.004) + 0.5 * rng.standard_normal(n_assets))
    turn = base_turn[None, :] * np.exp(0.3 * rng.standard_normal((n_days, n_assets)))
    dvol = cap * turn

    security_master = pd.DataFrame({
        "asset": assets,
        "start_date": dates[start_idx],
        "end_date": [pd.NaT if s else dates[e] for s, e in zip(still_listed, end_idx)],
    })

    fundamentals = _fundamentals(rng, dates, assets, start_idx, end_idx, np.exp(log_cap),
                                 traits, payer, restatement_fraction)

    data = MarketData(
        returns=pd.DataFrame(R, index=dates, columns=assets),
        market_cap=pd.DataFrame(cap, index=dates, columns=assets),
        dollar_volume=pd.DataFrame(dvol, index=dates, columns=assets),
        security_master=security_master,
        industries=industries_df,
        fundamentals=fundamentals,
    )
    truth = SyntheticTruth(
        market=pd.Series(m, index=dates),
        industry_returns=pd.DataFrame(g, index=dates, columns=industries),
        style_returns=pd.DataFrame(f, index=dates, columns=style_names),
        traits=traits.assign(payer=payer, size=size_z),
        beta=beta,
        specific_vol=spec_vol,
    )
    return data, truth


def _fundamentals(rng, dates, assets, start_idx, end_idx, cap_path, traits, payer, restate_frac):
    quarter_ends = pd.date_range(dates[0] - pd.Timedelta(days=500), dates[-1], freq="QE")
    rows = []
    for i, a in enumerate(assets):
        first = dates[start_idx[i]] - pd.Timedelta(days=400)
        last = dates[end_idx[i]]
        tv, tq, tg, tl = traits.loc[a, ["value", "quality", "growth", "leverage"]]
        sales0 = None
        for pe in quarter_ends:
            if pe < first or pe > last:
                continue
            avail = pe + pd.Timedelta(days=int(rng.integers(30, 60)))
            if avail > dates[-1]:
                continue
            k = int(np.clip(dates.searchsorted(pe), start_idx[i], end_idx[i]))
            cap_ref = cap_path[k, i]
            book = cap_ref * np.exp(-0.5 + 0.5 * tv + 0.05 * rng.standard_normal())
            ni = book * (0.08 + 0.04 * tq + 0.01 * rng.standard_normal())
            years = (pe - quarter_ends[0]).days / 365.25
            if sales0 is None:
                sales0 = book * 0.9
            sales = sales0 * np.exp(0.12 * tg * years + 0.02 * rng.standard_normal())
            gp = sales * (0.30 + 0.06 * tq)
            debt = book * 0.6 * np.exp(0.5 * tl)
            assets_tot = book + debt + 0.2 * book
            div = 0.4 * max(ni, 0.0) if payer[i] else 0.0
            vals = {
                "book_equity": book, "net_income_ttm": ni, "sales_ttm": sales,
                "gross_profit_ttm": gp, "total_assets": assets_tot, "total_debt": debt,
                "dividends_ttm": div,
            }
            restated = rng.random() < restate_frac
            for fld, v in vals.items():
                rows.append((a, fld, v, pe, avail))
                if restated:
                    later = avail + pd.Timedelta(days=120)
                    if later <= dates[-1]:
                        rows.append((a, fld, v * 1.05, pe, later))
    return pd.DataFrame(rows, columns=["asset", "field", "value", "period_end", "available_date"])
