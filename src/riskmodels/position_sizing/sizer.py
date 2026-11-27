"""Turn conviction scores into capped, beta-targeted, vol-targeted weights.

Pipeline (see README.md for the reasoning behind each choice):

1. Conviction multiplier from the score.
2. Risk-budget weight  w = base_risk x multiplier / sigma  (specific or total vol).
3. Short adjustment and trim rule.
4-6. Search for the long/short book multipliers that hit the beta (or net) target and
     the vol target, with the hard caps applied inside every evaluation. Caps therefore
     always hold; targets are hit as closely as the caps allow and any miss is reported.
5'. Factor check on the result: style and industry exposures and risk shares.
7. Trade list against current holdings, risk report and breach alerts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..factor_model.risk import RiskDecomposition, RiskModel
from .analytics import FastRisk, predicted_betas, sector_from_industries
from .budget import (
    adverse_move,
    conviction_multiplier,
    parse_side,
    risk_budget_weights,
    short_trim_factor,
)
from .caps import apply_caps, liquidity_cap, name_caps
from .config import SizingConfig
from .report import (
    _row,
    breach_frame,
    check_factors,
    check_limits,
    check_targets,
    trade_list,
)

REQUIRED_COLUMNS = ("side", "conviction")


@dataclass
class SizingResult:
    """Output of :meth:`PositionSizer.size`.

    Attributes
    ----------
    weights : Series
        Target weight per name as a fraction of NAV (0 for names sized to nothing, such
        as a covered short). This is what VaR, liquidity and stress models consume.
    positions : DataFrame
        One row per name with every intermediate step: multiplier, volatility used, raw
        weight, trim, caps, binding limits, beta, sector, days to liquidate.
    trades : DataFrame
        Trades from current holdings to ``weights``.
    risk : RiskDecomposition
        The factor model's decomposition of the target portfolio.
    summary : dict
        Book-level numbers: gross, net, long, short, beta, predicted vol, solver info.
    breaches : DataFrame
        Alerts, columns severity, kind, name, value, limit, message.
    """

    as_of: pd.Timestamp
    nav: float
    config: SizingConfig
    weights: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    risk: RiskDecomposition
    summary: dict
    breaches: pd.DataFrame
    sector: pd.Series = field(repr=False)
    caps: pd.DataFrame = field(repr=False)
    exposure: pd.Series | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        """True when no hard limit is breached."""
        return not (self.breaches["severity"] == "breach").any()

    def risk_report(self) -> dict[str, pd.DataFrame | dict]:
        """Everything a daily risk report needs, as tables."""
        w = self.weights[self.weights != 0]
        sec = self.sector.reindex(w.index)
        by_sector = pd.DataFrame({
            "long": w.clip(lower=0).groupby(sec).sum(),
            "short": w.clip(upper=0).groupby(sec).sum(),
        })
        by_sector["net"] = by_sector["long"] + by_sector["short"]
        by_sector["gross"] = by_sector["long"] - by_sector["short"]
        rc = self.risk.by_position["contribution"]
        by_sector["risk_contribution"] = rc.groupby(sec).sum()
        return {
            "summary": self.summary,
            "by_sector": by_sector,
            "by_factor": self.risk.by_factor,
            "by_group": self.risk.by_group,
            "by_position": self.positions.join(self.risk.by_position[
                ["marginal", "contribution", "pct_of_variance"]], how="left"),
            "breaches": self.breaches,
        }


@dataclass
class MonitorResult:
    """Output of :meth:`PositionSizer.monitor`: alerts on the book as it stands now."""

    breaches: pd.DataFrame
    current_summary: dict
    drift: pd.DataFrame
    resized: SizingResult

    @property
    def ok(self) -> bool:
        return not (self.breaches["severity"] == "breach").any()


def _book_summary(w: pd.Series, risk: RiskDecomposition, exposure: pd.Series | None) -> dict:
    w = w.astype(float)
    out = {
        "gross": float(w.abs().sum()),
        "net": float(w.sum()),
        "long": float(w.clip(lower=0).sum()),
        "short": float(w.clip(upper=0).sum()),
        "n_long": int((w > 0).sum()),
        "n_short": int((w < 0).sum()),
        "predicted_vol": risk.total_risk,
        "factor_vol": risk.factor_risk,
        "specific_vol": risk.specific_risk,
    }
    if exposure is not None:
        out["beta"] = float((w * exposure.reindex(w.index).fillna(0.0)).sum())
    return out


class PositionSizer:
    """Size a long/short book from conviction scores using the factor model's risk.

    Example
    -------
    >>> result = FactorModel(FactorModelConfig.daily()).fit(data)
    >>> rm = result.risk_model()
    >>> bench = cap_weighted_benchmark(result)
    >>> sizing = PositionSizer(SizingConfig()).size(signals, rm, nav=100e6,
    ...                                            benchmark=bench, current=holdings)
    >>> sizing.trades, sizing.breaches
    """

    def __init__(self, config: SizingConfig | None = None) -> None:
        self.config = config or SizingConfig()

    def with_config(self, **overrides) -> "PositionSizer":
        """A sizer with some settings changed, e.g. ``sizer.with_config(base_risk=0.004)``."""
        return PositionSizer(self.config.with_overrides(**overrides))

    # ------------------------------------------------------------------ inputs
    def _prepare(self, signals: pd.DataFrame, risk_model: RiskModel, nav: float,
                 benchmark: pd.Series | None, betas: pd.Series | None) -> dict:
        c = self.config
        if not nav > 0:
            raise ValueError("nav must be positive")
        missing_cols = [k for k in REQUIRED_COLUMNS if k not in signals.columns]
        if missing_cols:
            raise ValueError(f"signals is missing columns {missing_cols}")
        if signals.index.has_duplicates:
            raise ValueError("signals has duplicate names")
        names = signals.index
        uncovered = names.difference(risk_model.exposures.index)
        if len(uncovered):
            raise KeyError(f"no risk forecast for {len(uncovered)} names: {list(uncovered[:10])}")

        side = parse_side(signals["side"])
        mult = conviction_multiplier(signals["conviction"], c.conviction_multipliers)
        if c.volatility == "specific":
            sigma = risk_model.specific_risk.reindex(names)
        else:
            sigma = risk_model.total_risk().reindex(names)

        price = signals["price"].astype(float) if "price" in signals else None
        if "adv_dollars" in signals:
            adv_dollars = signals["adv_dollars"].astype(float)
        elif "adv" in signals and price is not None:
            adv_dollars = signals["adv"].astype(float) * price
        else:
            adv_dollars = None

        if "entry_price" in signals and price is not None:
            move = adverse_move(side, price, signals["entry_price"].astype(float))
        else:
            move = pd.Series(0.0, index=names)
        trim = short_trim_factor(move, c)

        if "sector" in signals:
            sector = signals["sector"].astype(object).fillna("unknown")
        else:
            sector = sector_from_industries(risk_model).reindex(names).fillna("unknown")

        liq = (liquidity_cap(adv_dollars, nav, c) if adv_dollars is not None
               else pd.Series(np.inf, index=names))
        caps = name_caps(side, liq, c)

        exposure = None
        if c.neutralize == "net":
            exposure = pd.Series(1.0, index=names)
        if betas is not None:
            beta = betas.reindex(names).astype(float)
            if beta.isna().any():
                raise KeyError(f"no beta for {list(beta.index[beta.isna()][:10])}")
        elif benchmark is not None:
            beta = predicted_betas(risk_model, benchmark).reindex(names)
        else:
            beta = None
        if c.neutralize == "beta":
            if beta is None:
                raise ValueError("a beta target needs a benchmark (or betas)")
            exposure = beta

        raw = risk_budget_weights(side, mult, sigma, trim, c)
        return dict(names=names, side=side, mult=mult, sigma=sigma, price=price,
                    adv_dollars=adv_dollars, move=move, trim=trim, sector=sector,
                    caps=caps, beta=beta, exposure=exposure, raw=raw)

    # ------------------------------------------------------------------ solve
    def _solve(self, raw: pd.Series, caps: pd.DataFrame, sector: pd.Series,
               exposure: pd.Series | None, fast: FastRisk) -> tuple[pd.Series, pd.Series, dict]:
        """Book multipliers that hit the beta (or net) target and the vol target after caps.

        Weights are caps(k * (a_long(rho) * raw_long + a_short(rho) * raw_short)), where
        rho = s_short / s_long is the long/short balance and k the overall level.

        Inner search: for a given level k, bisect on rho until beta (or net) hits target.
        Outer search: bisect on k until predicted vol hits target. A level at which the
        caps make the beta target impossible counts as too high, so when both targets
        cannot hold at once the beta target is kept and vol ends at the highest level that
        allows it. Caps are applied inside every evaluation, so they hold by construction.
        """
        c = self.config
        long_ = (raw > 0).to_numpy()
        r = raw.to_numpy(dtype=float)
        e = None if exposure is None else exposure.reindex(raw.index).to_numpy(dtype=float)
        two_sided = bool(long_.any() and (~long_).any())
        evals = 0

        def build(k: float, rho: float) -> tuple[pd.Series, pd.Series]:
            nonlocal evals
            evals += 1
            if c.book_scaling == "long":
                s_l, s_s = 1.0 / rho, 1.0
            elif c.book_scaling == "short":
                s_l, s_s = 1.0, rho
            else:  # shrink whichever side has the excess
                s_l, s_s = (1.0, rho) if rho <= 1.0 else (1.0 / rho, 1.0)
            scaled = pd.Series(k * np.where(long_, s_l, s_s) * r, index=raw.index)
            return apply_caps(scaled, caps["cap"], sector, c, cap_source=caps["cap_source"])

        def exposure_gap(k: float, rho: float) -> float:
            w, _ = build(k, rho)
            return float(w.to_numpy() @ e) - c.target_exposure

        def balance(k: float) -> tuple[float, bool]:
            """rho that hits the exposure target at level k, and whether it was reachable."""
            if e is None:
                return 1.0, True
            if not two_sided:
                return 1.0, False
            x, ok = _bisect(lambda lr: exposure_gap(k, float(np.exp(lr))), -12.0, 12.0,
                            c.tolerance)
            return float(np.exp(x)), ok

        exposure_ok = True
        vol_ok = True
        if c.target_volatility is None:
            k = 1.0
            rho, exposure_ok = balance(k)
        else:
            keep_exposure = e is not None and two_sided
            if e is not None and not keep_exposure:
                exposure_ok = False  # one-sided book: nothing to balance

            def vol_gap(log_k: float) -> float:
                k = float(np.exp(log_k))
                rho, ok = balance(k) if keep_exposure else (1.0, True)
                if not ok:
                    # Either the book is too small to reach the target (a nonzero beta
                    # needs some size) or the caps at this size forbid it. Doubling the
                    # level tells which: if that gets closer, the book is too small.
                    now = abs(exposure_gap(k, rho))
                    bigger = abs(exposure_gap(2.0 * k, balance(2.0 * k)[0]))
                    return -1.0 if bigger < now - 1e-14 else 1.0
                w, _ = build(k, rho)
                return fast.volatility(w.to_numpy()) / c.target_volatility - 1.0

            log_k, vol_ok = _bisect(vol_gap, -12.0, 12.0, c.tolerance)
            k = float(np.exp(log_k))
            rho, ok = balance(k) if keep_exposure else (1.0, True)
            exposure_ok = exposure_ok and ok

        w, binding = build(k, rho)
        exp_err = 0.0 if e is None else abs(float(w.to_numpy() @ e) - c.target_exposure)
        vol_err = 0.0 if c.target_volatility is None else abs(
            fast.volatility(w.to_numpy()) / c.target_volatility - 1.0)
        vol_ok = vol_err <= 10 * c.tolerance
        exposure_ok = exp_err <= 10 * c.tolerance
        info = {
            "evaluations": evals,
            "converged": bool(exposure_ok and vol_ok),
            "level": k,
            "short_long_ratio": rho,
            "exposure_reachable": bool(exposure_ok),
            "vol_reachable": bool(vol_ok),
        }
        return w, binding, info

    # ------------------------------------------------------------------ public
    def size(
        self,
        signals: pd.DataFrame,
        risk_model: RiskModel,
        nav: float,
        benchmark: pd.Series | None = None,
        betas: pd.Series | None = None,
        current: pd.Series | None = None,
    ) -> SizingResult:
        """Size every name in ``signals``.

        Parameters
        ----------
        signals : DataFrame indexed by asset
            Required: ``side`` (long/short or +1/-1) and ``conviction``. Optional:
            ``price``; ``adv`` (shares, used with price) or ``adv_dollars`` for the
            liquidity cap; ``entry_price`` for the short trim rule; ``sector`` (defaults
            to the factor model's industry).
        risk_model : RiskModel
            ``FactorModelResult.risk_model(as_of)``.
        nav : float
            Net asset value, in the same currency as ``adv_dollars``.
        benchmark : Series, optional
            Benchmark weights for predicted betas (see ``cap_weighted_benchmark``).
            Required for a beta target unless ``betas`` is given.
        betas : Series, optional
            Your own betas, used instead of the factor model's predicted betas.
        current : Series, optional
            Current weights (fractions of NAV) for the trade list. Names held but not in
            ``signals`` are exited.
        """
        c = self.config
        p = self._prepare(signals, risk_model, nav, benchmark, betas)
        names = p["names"]
        fast = FastRisk(risk_model, names)
        w, binding, info = self._solve(p["raw"], p["caps"], p["sector"], p["exposure"], fast)
        w = w.rename("weight")

        risk = risk_model.portfolio_risk(w) if (w != 0).any() else _empty_risk(risk_model)
        summary = _book_summary(w, risk, p["beta"])
        summary.update(info)
        if p["exposure"] is not None:
            summary[f"{c.neutralize}_exposure"] = float((w * p["exposure"]).sum())

        liq_den = (c.adv_participation * p["adv_dollars"] if p["adv_dollars"] is not None
                   else pd.Series(np.nan, index=names))
        positions = pd.DataFrame({
            "side": np.where(p["side"] > 0, "long", "short"),
            "conviction": signals["conviction"],
            "multiplier": p["mult"],
            "volatility": p["sigma"],
            "adverse_move": p["move"],
            "trim": p["trim"],
            "raw_weight": p["raw"],
            "cap": p["caps"]["cap"],
            "liquidity_cap": p["caps"]["liquidity_cap"],
            "weight": w,
            "value": w * nav,
            "beta": p["beta"] if p["beta"] is not None else np.nan,
            "sector": p["sector"],
            "binding": binding,
            "days_to_liquidate": (w.abs() * nav / liq_den).replace([np.inf], np.nan),
        }, index=names)
        positions.index.name = "asset"

        rows = check_limits(w, p["caps"], p["sector"], c)
        rows += check_targets(summary.get(f"{c.neutralize}_exposure"), risk.total_risk, c)
        if (w != 0).any():
            rows += check_factors(risk, c)
        rows += _input_notes(p, c)

        trades = trade_list(w, current, nav, p["price"], p["adv_dollars"], c.min_trade)
        return SizingResult(
            as_of=risk_model.as_of, nav=nav, config=c, weights=w, positions=positions,
            trades=trades, risk=risk, summary=summary, breaches=breach_frame(rows),
            sector=p["sector"], caps=p["caps"], exposure=p["exposure"],
        )

    def monitor(
        self,
        current: pd.Series,
        signals: pd.DataFrame,
        risk_model: RiskModel,
        nav: float,
        benchmark: pd.Series | None = None,
        betas: pd.Series | None = None,
    ) -> MonitorResult:
        """Check the book as it stands now and re-size it.

        Run after every price move (pass drifted weights, see ``drift_weights``) and every
        conviction change. Alerts cover hard-limit breaches on the current book, names that
        have drifted more than ``drift_tolerance`` from their fresh target, shorts that hit a
        trim step, and factor bets in the current book. ``resized`` holds the fresh targets
        and the trade list that fixes everything.
        """
        c = self.config
        resized = self.size(signals, risk_model, nav, benchmark=benchmark, betas=betas,
                            current=current)
        cur = current.astype(float)
        cur = cur[cur != 0]
        rows = check_limits(cur, resized.caps, resized.sector.reindex(cur.index)
                            .fillna("unknown"), c)

        covered = cur.index.intersection(risk_model.exposures.index)
        risk = (risk_model.portfolio_risk(cur.loc[covered]) if len(covered)
                else _empty_risk(risk_model))
        beta = None
        if betas is not None:
            beta = betas
        elif benchmark is not None:
            beta = predicted_betas(risk_model, benchmark)
        summary = _book_summary(cur, risk, beta)
        if len(covered):
            rows += check_factors(risk, c)
        uncovered = cur.index.difference(risk_model.exposures.index)
        if len(uncovered):
            rows.append(_row("warning", "no_risk_forecast", ",".join(map(str, uncovered[:10])),
                             len(uncovered), None,
                             f"{len(uncovered)} held names have no risk forecast"))

        idx = resized.weights.index.union(cur.index)
        drift = pd.DataFrame({
            "current_weight": cur.reindex(idx, fill_value=0.0),
            "target_weight": resized.weights.reindex(idx, fill_value=0.0),
        })
        drift["drift"] = drift["current_weight"] - drift["target_weight"]
        for name, d in drift["drift"].items():
            if abs(d) > c.drift_tolerance:
                rows.append(_row("warning", "drift", name, d, c.drift_tolerance,
                                 f"{name} is {d:+.2%} of NAV away from its target"))
        rows += [r for r in resized.breaches.to_dict("records")
                 if r["kind"] in ("short_trim", "short_cover")]
        return MonitorResult(breaches=breach_frame(rows), current_summary=summary,
                             drift=drift, resized=resized)


def _bisect(f, lo: float, hi: float, tol: float, max_iter: int = 60) -> tuple[float, bool]:
    """Root of f on [lo, hi] by bisection, stopping once |f| <= tol.

    Bisection rather than a faster secant method because the caps make f flat in places
    and can make it step. With no sign change in the bracket, the endpoint closest to zero
    is returned with ok=False (target unreachable).
    """
    flo, fhi = f(lo), f(hi)
    if abs(flo) <= tol:
        return lo, True
    if abs(fhi) <= tol:
        return hi, True
    if np.sign(flo) == np.sign(fhi):
        return (lo, False) if abs(flo) < abs(fhi) else (hi, False)
    x, fx = lo, flo
    for _ in range(max_iter):
        x = 0.5 * (lo + hi)
        fx = f(x)
        if abs(fx) <= tol or hi - lo < 1e-10:
            break
        if np.sign(fx) == np.sign(flo):
            lo, flo = x, fx
        else:
            hi = x
    if abs(fx) > tol:  # stepped across the root: take the side that satisfies f <= 0
        x = lo
    return x, abs(f(x)) <= 10 * tol


def _input_notes(p: dict, c: SizingConfig) -> list[dict]:
    rows = []
    if p["adv_dollars"] is None:
        rows.append(_row("warning", "no_adv", "all", None, None,
                         "no ADV given, so no liquidity caps were applied"))
    else:
        miss = p["adv_dollars"].index[p["adv_dollars"].isna()]
        if len(miss):
            rows.append(_row("warning", "no_adv", ",".join(map(str, miss[:10])), len(miss), None,
                             f"{len(miss)} names have no ADV and no liquidity cap"))
    for name in p["trim"].index[p["trim"] < 1]:
        move = p["move"][name]
        if p["trim"][name] == 0:
            rows.append(_row("info", "short_cover", name, move, c.short_cover_move,
                             f"short {name} is up {move:.1%} since entry, sized to zero (cover)"))
        else:
            rows.append(_row("info", "short_trim", name, move, c.short_trim_step,
                             f"short {name} is up {move:.1%} since entry, trimmed to "
                             f"{p['trim'][name]:.0%} of its untrimmed size"))
    return rows


def _empty_risk(risk_model: RiskModel) -> RiskDecomposition:
    from ..factor_model.exposures import factor_groups
    f = risk_model.factors
    by_factor = pd.DataFrame({"group": factor_groups(f), "exposure": 0.0, "volatility": np.nan,
                              "marginal": 0.0, "contribution": 0.0, "pct_of_variance": 0.0},
                             index=f)
    by_position = pd.DataFrame(columns=["weight", "marginal", "factor_contribution",
                                        "specific_contribution", "contribution",
                                        "pct_of_variance"], dtype=float)
    by_group = pd.DataFrame({"contribution": [0.0], "pct_of_variance": [0.0]}, index=["specific"])
    return RiskDecomposition(0.0, 0.0, 0.0, by_factor, by_position, by_group)
