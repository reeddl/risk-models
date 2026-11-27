"""Run stress scenarios through the factor model.

For a book with weights h (fractions of NAV), exposures X and a scenario shock vector f
(cumulative factor returns over the scenario):

    factor P&L of name i      h_i x (X_i . f)
    factor P&L of the book    x . f,   x = X' h
    by factor                 x_k f_k

Stock-specific losses are added as a stressed residual: with daily specific variances d_i,
scenario horizon H (capped at ``max_specific_horizon_days``), the scenario's
``vol_multiplier`` m and the normal quantile z at ``specific_confidence``,

    specific stress = z x m x sqrt(H x sum_i h_i^2 d_i)

an adverse diversified move of the book's specific risk, always counted as a loss. It is
split across names by its Euler allocation, z m H h_i^2 d_i / sqrt(H sum h^2 d), so the parts add up.

    scenario P&L = factor P&L - specific stress
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..factor_model.exposures import INDUSTRY_PREFIX, factor_groups
from ..factor_model.risk import RiskModel
from ..position_sizing.analytics import sector_from_industries
from ..var.limits import _row, breach_frame
from ..var.model import _unpack
from ..var.stats import norm_ppf
from .config import StressConfig
from .scenario import Scenario, ScenarioLibrary, default_library


def map_shocks(shocks: dict, factors: pd.Index, industry_map: dict | None) -> tuple[dict, list[str]]:
    """Translate scenario shocks onto the model's factor names.

    Returns the shocks the model has and the names it does not.
    """
    collected: dict[str, list[float]] = {}
    dropped: list[str] = []
    known = set(factors)
    for k, v in shocks.items():
        targets = [k]
        if industry_map and k.startswith(INDUSTRY_PREFIX):
            label = k[len(INDUSTRY_PREFIX):]
            if label in industry_map:
                m = industry_map[label]
                targets = [f"{INDUSTRY_PREFIX}{t}" for t in ([m] if isinstance(m, str) else m)]
        hit = [t for t in targets if t in known]
        if not hit:
            dropped.append(k)
        for t in hit:
            collected.setdefault(t, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in collected.items()}, dropped


def full_shock(given: dict, factor_cov: pd.DataFrame, propagation: str) -> pd.DataFrame:
    """Shock for every factor: the given ones, the rest conditional on them (or zero).

    Returns columns ``shock`` and ``source`` (``given`` or ``propagated``).
    """
    factors = factor_cov.columns
    f = pd.Series(0.0, index=factors)
    s = [k for k in factors if k in given]
    u = [k for k in factors if k not in given]
    for k in s:
        f[k] = given[k]
    if propagation == "conditional" and s and u:
        F = factor_cov.to_numpy()
        si = [factors.get_loc(k) for k in s]
        ui = [factors.get_loc(k) for k in u]
        Fss = F[np.ix_(si, si)]
        Fus = F[np.ix_(ui, si)]
        ridge = 1e-12 * np.trace(Fss) / len(si)
        beta = np.linalg.solve(Fss + ridge * np.eye(len(si)), f[s].to_numpy())
        f[u] = Fus @ beta
    src = pd.Series(np.where(f.index.isin(s), "given", "propagated"), index=factors)
    if propagation == "zero":
        src[u] = "zero"
    return pd.DataFrame({"shock": f, "source": src})


@dataclass
class StressResult:
    """One scenario applied to one book. P&L in fractions of NAV, losses negative."""

    scenario: Scenario
    summary: dict
    by_position: pd.DataFrame
    by_factor: pd.DataFrame
    by_sector: pd.DataFrame
    dropped_shocks: list[str] = field(default_factory=list)

    @property
    def pnl(self) -> float:
        return self.summary["pnl"]


def apply_scenario(
    weights: pd.Series,
    risk_model: RiskModel,
    scenario: Scenario,
    config: StressConfig,
    sector: pd.Series | None = None,
    vol_multiplier: float | None = None,
) -> StressResult:
    """P&L of ``weights`` under ``scenario`` (see the module docstring)."""
    w = weights.astype(float)
    w = w[w != 0]
    missing = w.index.difference(risk_model.exposures.index)
    if len(missing):
        raise KeyError(f"no risk forecast for {len(missing)} names: {list(missing[:10])}")
    factors = risk_model.factors
    given, dropped = map_shocks(scenario.shocks, factors, config.industry_map)
    shock = full_shock(given, risk_model.factor_covariance, config.propagation)
    f = shock["shock"]

    X = risk_model.exposures.loc[w.index, factors]
    asset_ret = pd.Series(X.to_numpy() @ f.to_numpy(), index=w.index)
    factor_pnl_i = w * asset_ret
    x = pd.Series(X.to_numpy().T @ w.to_numpy(), index=factors)

    m = scenario.vol_multiplier if vol_multiplier is None else vol_multiplier
    H = scenario.horizon_days
    if config.max_specific_horizon_days is not None:
        H = min(H, config.max_specific_horizon_days)
    d = risk_model.specific_variance.loc[w.index] / config.trading_days_per_year
    z = norm_ppf(config.specific_confidence)
    spec_var = float((w * w * d).sum()) * H
    spec_sd = np.sqrt(spec_var) * m
    spec_total = z * spec_sd
    spec_i = (z * m * m * H * w * w * d / spec_sd) if spec_sd > 0 else w * 0.0

    pnl_i = factor_pnl_i - spec_i
    by_position = pd.DataFrame({
        "weight": w,
        "factor_return": asset_ret,
        "factor_pnl": factor_pnl_i,
        "specific_stress": spec_i,
        "pnl": pnl_i,
    })
    by_factor = pd.DataFrame({
        "group": factor_groups(factors),
        "exposure": x,
        "shock": f,
        "source": shock["source"],
        "pnl": x * f,
    })
    sec = (sector if sector is not None else sector_from_industries(risk_model))
    sec = sec.reindex(w.index).fillna("unknown")
    by_sector = by_position.groupby(sec)[["weight", "factor_pnl", "specific_stress", "pnl"]].sum()
    by_sector.index.name = "sector"

    groups = by_factor.groupby("group")["pnl"].sum()
    long_, short = w > 0, w < 0
    factor_pnl = float(factor_pnl_i.sum())
    summary = {
        "scenario": scenario.name,
        "kind": scenario.kind,
        "calibration": scenario.calibration,
        "horizon_days": scenario.horizon_days,
        "factor_pnl": factor_pnl,
        "market_pnl": float(groups.get("market", 0.0)),
        "industry_pnl": float(groups.get("industry", 0.0)),
        "style_pnl": float(groups.get("style", 0.0)),
        "specific_stress": float(spec_total),
        "pnl": factor_pnl - float(spec_total),
        "long_pnl": float(pnl_i[long_].sum()),
        "short_pnl": float(pnl_i[short].sum()),
        "worst_position": pnl_i.idxmin() if len(pnl_i) else None,
        "worst_position_pnl": float(pnl_i.min()) if len(pnl_i) else 0.0,
        "worst_sector": by_sector["pnl"].idxmin() if len(by_sector) else None,
        "worst_sector_pnl": float(by_sector["pnl"].min()) if len(by_sector) else 0.0,
        "n_given": len(given),
        "n_dropped": len(dropped),
    }
    summary["loss"] = -summary["pnl"]
    return StressResult(scenario=scenario, summary=summary, by_position=by_position,
                        by_factor=by_factor, by_sector=by_sector, dropped_shocks=dropped)


@dataclass
class StressReport:
    """Every scenario in a library applied to one book.

    Attributes
    ----------
    summary : DataFrame
        One row per scenario: factor, market, industry, style and specific P&L, total P&L
        and loss, long and short P&L, worst name and sector, limit utilization. Sorted worst
        first. Amounts (``pnl_amount``, ``loss_amount``) are added when ``nav`` is given.
    shocks : DataFrame
        Factor x scenario shock actually applied (given and propagated).
    results : dict
        Scenario name -> :class:`StressResult` with position, factor and sector detail.
    breaches : DataFrame
    """

    as_of: pd.Timestamp
    config: StressConfig
    nav: float | None
    weights: pd.Series
    summary: pd.DataFrame
    shocks: pd.DataFrame
    results: dict
    breaches: pd.DataFrame

    @property
    def ok(self) -> bool:
        return not (self.breaches["severity"] == "breach").any()

    @property
    def worst(self) -> StressResult:
        return self.results[self.summary.index[0]]

    @property
    def by_position(self) -> pd.DataFrame:
        """Name x scenario P&L table."""
        return pd.DataFrame({k: r.by_position["pnl"] for k, r in self.results.items()})

    @property
    def by_factor(self) -> pd.DataFrame:
        """Factor x scenario P&L table."""
        return pd.DataFrame({k: r.by_factor["pnl"] for k, r in self.results.items()})

    @property
    def by_sector(self) -> pd.DataFrame:
        """Sector x scenario P&L table."""
        return pd.DataFrame({k: r.by_sector["pnl"] for k, r in self.results.items()}).fillna(0.0)


def _limit_rows(summary: pd.DataFrame, results: dict, config: StressConfig) -> list[dict]:
    lim = config.limits
    rows: list[dict] = []
    if lim.max_loss is not None:
        for name, r in summary.iterrows():
            loss = r["loss"]
            u = loss / lim.max_loss
            sev = "breach" if u >= 1 else None
            if sev is None:
                crossed = [(lvl, n) for n, lvl in lim.alert_levels.items() if u >= lvl]
                sev = max(crossed)[1] if crossed else None
            if sev:
                rows.append(_row(sev, "stress_loss", name, loss, lim.max_loss,
                                 f"{name}: loss {loss:.2%} of NAV is {u:.0%} of the "
                                 f"{lim.max_loss:.0%} stress limit"))
    for name, res in results.items():
        if res.dropped_shocks:
            rows.append(_row("info", "dropped_shocks", name, len(res.dropped_shocks), None,
                             f"{name}: {len(res.dropped_shocks)} shocks name factors the model "
                             f"does not have ({', '.join(res.dropped_shocks[:5])}); set "
                             f"industry_map or edit the scenario"))
    approx = [n for n, r in results.items() if r.scenario.calibration == "approximate"
              and r.scenario.kind == "historical"]
    if approx:
        rows.append(_row("info", "approximate_scenarios", f"{len(approx)} historical", len(approx),
                         None, f"{len(approx)} historical scenarios use approximate shocks, "
                               f"not measured factor returns (see calibrate_from_factor_returns)"))
    return rows


class StressTester:
    """Applies a scenario library to a book through the factor model.

    Example
    -------
    >>> rep = StressTester().run(sized, fit.risk_model(), nav=250e6)
    >>> rep.summary[["pnl", "factor_pnl", "specific_stress"]]
    """

    def __init__(self, config: StressConfig | None = None,
                 library: ScenarioLibrary | None = None) -> None:
        self.config = config or StressConfig()
        self.library = library if library is not None else default_library()

    def with_config(self, **overrides) -> "StressTester":
        return StressTester(self.config.with_overrides(**overrides), self.library)

    def run(
        self,
        portfolio,
        risk_model: RiskModel,
        nav: float | None = None,
        scenarios: ScenarioLibrary | list[Scenario] | list[str] | None = None,
        sector: pd.Series | None = None,
    ) -> StressReport:
        """Apply ``scenarios`` (default: the whole library) to ``portfolio``.

        ``portfolio`` is a weight Series or a position sizing result. ``scenarios`` may be
        a library, a list of scenarios, or a list of names from this tester's library.
        """
        c = self.config
        w, sector = _unpack(portfolio, sector)
        if w.empty:
            raise ValueError("portfolio has no non-zero weights")
        lib = self._resolve(scenarios)
        results = {s.name: apply_scenario(w, risk_model, s, c, sector) for s in lib}
        summary = pd.DataFrame([r.summary for r in results.values()]).set_index("scenario")
        if c.limits.max_loss is not None:
            summary["limit_utilization"] = summary["loss"] / c.limits.max_loss
        if nav is not None:
            summary["pnl_amount"] = summary["pnl"] * nav
            summary["loss_amount"] = summary["loss"] * nav
        summary = summary.sort_values("pnl", kind="stable")
        shocks = pd.DataFrame({k: r.by_factor["shock"] for k, r in results.items()})
        breaches = breach_frame(_limit_rows(summary, results, c))
        return StressReport(as_of=risk_model.as_of, config=c, nav=nav, weights=w,
                            summary=summary, shocks=shocks, results=results, breaches=breaches)

    def _resolve(self, scenarios) -> ScenarioLibrary:
        if scenarios is None:
            return self.library
        if isinstance(scenarios, ScenarioLibrary):
            return scenarios
        items = list(scenarios)
        if items and all(isinstance(s, str) for s in items):
            return self.library.select(names=items)
        return ScenarioLibrary(items)
