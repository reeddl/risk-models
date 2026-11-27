"""Raw descriptors: the measurable characteristics behind each style factor.

Every descriptor for date t uses only information known at the close of t: prices and
volumes through t, fundamentals whose ``available_date`` is on or before t, and the
industry assignment in force on t.

Price-based descriptors (Barra USE4 definitions, all on excess returns)
    beta            EWMA-weighted time-series beta to the cap-weighted market
    hsigma          volatility of the residual from that beta regression
    dastd           EWMA volatility of daily excess returns
    cmra            range of cumulative log excess returns over the last 12 months
    momentum        EWMA-weighted sum of log excess returns, skipping the last month
    stom/stoq/stoa  log of monthly share turnover averaged over 1, 3 and 12 months
    log_market_cap  natural log of market cap

Fundamental descriptors
    book_to_price, earnings_to_price, roe, gross_profitability, sales_growth,
    earnings_growth, market_leverage, debt_to_assets, dividend_yield
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.covariance import ewma_weights
from ..core.data import MarketData, alive_mask
from ..core.pit import PointInTimeFundamentals, industry_panel
from .config import FactorModelConfig

TRADING_DAYS_PER_MONTH = 21


def _window_weights(start: int, end: int, length: int, half_life: float) -> np.ndarray:
    """EWMA weights for rows [start, end), truncated if the window ran past the data start."""
    w = ewma_weights(length, half_life)
    return w[length - (end - start):]


def _sym_growth(now: pd.Series, before: pd.Series) -> pd.Series:
    """(now - before) / mean(|now|, |before|): bounded in [-2, 2] and defined for losses."""
    den = 0.5 * (now.abs() + before.abs())
    return (now - before).where(den > 0) / den


class DescriptorEngine:
    """Computes raw descriptors on any date of a :class:`MarketData` bundle."""

    def __init__(self, data: MarketData, config: FactorModelConfig) -> None:
        self.config = config
        self.dates = data.dates
        self.assets = data.assets
        R = data.returns.to_numpy(dtype=float)
        rf = (data.risk_free.to_numpy(dtype=float) if data.risk_free is not None
              else np.zeros(len(self.dates)))
        self.rf = rf
        self.excess = R - rf[:, None]
        self.log_excess = np.log1p(R) - np.log1p(rf)[:, None]
        self.cap = data.market_cap.to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.turnover = data.dollar_volume.to_numpy(dtype=float) / self.cap
        self.alive = alive_mask(data.security_master, self.dates, self.assets).to_numpy()
        self.industry = industry_panel(data.industries, self.dates, self.assets)
        self.market_excess = self._market_return(R) - rf
        self.pit = PointInTimeFundamentals(
            data.fundamentals,
            max_age_days=config.fundamental_max_age_days,
            prior_year_tolerance_days=config.growth_tolerance_days,
        )

    def _market_return(self, R: np.ndarray) -> np.ndarray:
        """Cap-weighted return of listed names, weighted by the previous close's cap."""
        lag_cap = np.vstack([self.cap[:1], self.cap[:-1]])
        w = np.where(np.isfinite(R) & np.isfinite(lag_cap) & self.alive, lag_cap, 0.0)
        num = (np.where(w > 0, R, 0.0) * w).sum(1)
        den = w.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, 0.0)

    # ------------------------------------------------------------------ price based
    def _beta_hsigma(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        c = self.config
        s, e = max(0, t - c.beta_window + 1), t + 1
        Y = self.excess[s:e]
        x = self.market_excess[s:e]
        w = _window_weights(s, e, c.beta_window, c.beta_half_life)
        M = np.isfinite(Y)
        ww = w[:, None] * M
        sw = ww.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            X0 = np.where(M, x[:, None], 0.0)
            Y0 = np.where(M, Y, 0.0)
            mx = (ww * X0).sum(0) / sw
            my = (ww * Y0).sum(0) / sw
            dx = np.where(M, X0 - mx, 0.0)
            dy = np.where(M, Y0 - my, 0.0)
            vxx = (ww * dx * dx).sum(0) / sw
            vxy = (ww * dx * dy).sum(0) / sw
            beta = vxy / vxx
            resid = dy - beta * dx
            hsigma = np.sqrt((ww * resid * resid).sum(0) / sw)
        short = M.sum(0) < c.beta_min_obs
        beta[short] = np.nan
        hsigma[short] = np.nan
        return beta, hsigma

    def _dastd(self, t: int) -> np.ndarray:
        c = self.config
        s, e = max(0, t - c.dastd_window + 1), t + 1
        Y = self.excess[s:e]
        w = _window_weights(s, e, c.dastd_window, c.dastd_half_life)
        M = np.isfinite(Y)
        ww = w[:, None] * M
        sw = ww.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            Y0 = np.where(M, Y, 0.0)
            mu = (ww * Y0).sum(0) / sw
            d = np.where(M, Y0 - mu, 0.0)
            out = np.sqrt((ww * d * d).sum(0) / sw)
        out[M.sum(0) < c.beta_min_obs] = np.nan
        return out

    def _cmra(self, t: int) -> np.ndarray:
        months = self.config.cmra_months
        n = months * TRADING_DAYS_PER_MONTH
        s, e = max(0, t - n + 1), t + 1
        L = self.log_excess[s:e]
        M = np.isfinite(L)
        L0 = np.where(M, L, 0.0)[::-1]           # newest first
        cum = np.cumsum(L0, axis=0)
        ends = [min(k * TRADING_DAYS_PER_MONTH, cum.shape[0]) - 1 for k in range(1, months + 1)]
        Z = cum[ends]
        out = Z.max(0) - Z.min(0)
        out[M.sum(0) < n // 2] = np.nan
        return out

    def _momentum(self, t: int) -> np.ndarray:
        c = self.config
        end = t + 1 - c.momentum_lag
        s = max(0, end - c.momentum_window)
        if end <= s:
            return np.full(len(self.assets), np.nan)
        L = self.log_excess[s:end]
        w = _window_weights(s, end, c.momentum_window, c.momentum_half_life)
        M = np.isfinite(L)
        out = (np.where(M, L, 0.0) * w[:, None]).sum(0)
        out[M.sum(0) < c.momentum_min_obs] = np.nan
        return out

    def _turnover(self, t: int, days: int) -> np.ndarray:
        s, e = max(0, t - days + 1), t + 1
        T = self.turnover[s:e]
        M = np.isfinite(T) & (T >= 0)
        n = M.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(M, T, 0.0).sum(0) / n
            out = np.where(mean > 0, np.log(mean * TRADING_DAYS_PER_MONTH), np.nan)
        out[n < min(self.config.liquidity_min_obs, days)] = np.nan
        return out

    # ------------------------------------------------------------------ fundamentals
    def _fundamental(self, t: int, cap: pd.Series) -> pd.DataFrame:
        snap = self.pit.snapshot(self.dates[t])
        cur = snap.current.reindex(index=self.assets)
        pri = snap.prior_year.reindex(index=self.assets)

        def col(df: pd.DataFrame, name: str) -> pd.Series:
            if name in df.columns:
                return df[name].astype(float)
            return pd.Series(np.nan, index=self.assets, dtype=float)

        book, ni = col(cur, "book_equity"), col(cur, "net_income_ttm")
        sales, gp = col(cur, "sales_ttm"), col(cur, "gross_profit_ttm")
        ta, debt = col(cur, "total_assets"), col(cur, "total_debt")
        div = col(cur, "dividends_ttm")
        cap = cap.where(cap > 0)
        return pd.DataFrame({
            "book_to_price": book / cap,
            "earnings_to_price": ni / cap,
            "roe": ni / book.where(book > 0),
            "gross_profitability": gp / ta.where(ta > 0),
            "sales_growth": _sym_growth(sales, col(pri, "sales_ttm")),
            "earnings_growth": _sym_growth(ni, col(pri, "net_income_ttm")),
            "market_leverage": (cap + debt) / cap,
            "debt_to_assets": debt / ta.where(ta > 0),
            "dividend_yield": div / cap,
        }, index=self.assets)

    # ------------------------------------------------------------------ public
    def raw(self, t: int) -> pd.DataFrame:
        """All raw descriptors on date position t (asset x descriptor, unstandardized)."""
        cap = pd.Series(self.cap[t], index=self.assets)
        beta, hsigma = self._beta_hsigma(t)
        price = pd.DataFrame({
            "log_market_cap": np.log(cap.where(cap > 0)),
            "beta": beta,
            "hsigma": hsigma,
            "dastd": self._dastd(t),
            "cmra": self._cmra(t),
            "momentum": self._momentum(t),
            "stom": self._turnover(t, TRADING_DAYS_PER_MONTH),
            "stoq": self._turnover(t, 3 * TRADING_DAYS_PER_MONTH),
            "stoa": self._turnover(t, 12 * TRADING_DAYS_PER_MONTH),
        }, index=self.assets)
        return pd.concat([price, self._fundamental(t, cap)], axis=1)

    def covered(self, t: int) -> np.ndarray:
        """Assets the model can forecast on date t: listed, with a cap and an industry."""
        ind = self.industry.iloc[t].notna().to_numpy()
        cap = self.cap[t]
        return self.alive[t] & np.isfinite(cap) & (cap > 0) & ind
