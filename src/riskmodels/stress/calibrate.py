"""Calibrate historical scenarios from measured factor returns.

The shipped historical scenarios are approximate. Once real data exists, measure them:

- :func:`calibrate_from_factor_returns` sums this factor model's own factor returns over
  the scenario's window. This is the right source for every window the model covers
  (EDGAR fundamentals start around 2009, so roughly 2009 onward).
- :func:`calibrate_from_reference` uses published long/short factor returns (AQR, Ken
  French) for older windows, scaled to this model's units.

Factor returns over a window are summed, not compounded: a book with constant exposures x
earns sum_t x . f_t, so the sum is the shock that reproduces that P&L in the linear
stress engine.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .scenario import Scenario, ScenarioLibrary

# Model factor -> (reference column, scale). Reference names follow the AQR data library and
# Ken French's library. A long/short portfolio such as HML holds roughly the top 30% minus
# the bottom 30% of names, about 2 cross-sectional standard deviations apart, so one unit of
# its return is about 0.5 of a pure factor return per standard deviation. The size factor
# here is long large caps, so it is minus SMB; beta is long high beta, so it is minus BAB.
# Replace the scales with measured slopes (see scales_from_comparison).
DEFAULT_REFERENCE_MAP: dict[str, tuple[str, float]] = {
    "market": ("MKT", 1.0),
    "size": ("SMB", -0.5),
    "value": ("HML Devil", 0.5),
    "momentum": ("UMD", 0.5),
    "quality": ("QMJ", 0.5),
    "beta": ("BAB", -0.5),
}


def _window_slice(returns: pd.DataFrame, start, end) -> pd.DataFrame:
    """Returns realized after the close of ``start`` up to and including ``end``."""
    idx = pd.DatetimeIndex(returns.index)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if e <= s:
        raise ValueError("window end must be after its start")
    return returns.loc[(idx > s) & (idx <= e)]


def _covers(returns: pd.DataFrame, start, end) -> bool:
    idx = pd.DatetimeIndex(returns.index)
    return len(idx) > 0 and idx[0] <= pd.Timestamp(start) + pd.Timedelta(days=7) \
        and idx[-1] >= pd.Timestamp(end)


def calibrate_from_factor_returns(
    factor_returns: pd.DataFrame,
    start,
    end,
    name: str | None = None,
    base: Scenario | None = None,
    factors: list[str] | None = None,
    min_abs_shock: float = 0.0,
) -> Scenario:
    """A historical scenario whose shocks are this model's factor returns over a window.

    Parameters
    ----------
    factor_returns : DataFrame (period end x factor), e.g. ``FactorModelResult.factor_returns``.
    start, end : the window; returns after the close of ``start`` through ``end`` count.
    name : str, optional. Defaults to ``base.name`` or ``"window_<start>_<end>"``.
    base : Scenario, optional. Keep its description, tags and liquidity multipliers and
        replace its shocks.
    factors : list, optional. Only these factors (default all).
    min_abs_shock : float. Drop shocks smaller than this in absolute value.
    """
    block = _window_slice(factor_returns, start, end)
    if block.empty:
        raise ValueError(f"no factor returns between {start} and {end}")
    if factors is not None:
        block = block[[f for f in factors if f in block.columns]]
    shocks = block.sum(min_count=1).dropna()
    shocks = shocks[shocks.abs() >= min_abs_shock]
    horizon = int(np.busday_count(pd.Timestamp(start).date(), pd.Timestamp(end).date()))
    horizon = max(horizon, 1)
    basis = {k: "calibrated" for k in shocks.index}
    window = (str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()))
    source = (f"Calibrated: sum of the factor model's factor returns from "
              f"{window[0]} to {window[1]} ({len(block)} periods).")
    if base is not None:
        return replace(base, name=name or base.name, shocks=shocks.to_dict(), basis=basis,
                       window=window, horizon_days=horizon, source=source)
    return Scenario(name=name or f"window_{window[0]}_{window[1]}", kind="historical",
                    shocks=shocks.to_dict(), horizon_days=horizon, window=window,
                    basis=basis, source=source,
                    description=f"Factor returns from {window[0]} to {window[1]}.")


def calibrate_from_reference(
    reference: pd.DataFrame,
    start,
    end,
    mapping: dict[str, tuple[str, float]] | None = None,
    name: str | None = None,
    base: Scenario | None = None,
    keep_unmapped: bool = True,
) -> Scenario:
    """A historical scenario from published factor returns (AQR, Ken French).

    ``reference`` is period end x column, daily or monthly (a monthly series can only
    calibrate windows that span whole months). Each model factor in ``mapping`` gets
    scale x the summed reference return over the window. With ``base`` and
    ``keep_unmapped``, the base scenario's other shocks (industries, say) are kept.
    """
    mapping = mapping or DEFAULT_REFERENCE_MAP
    block = _window_slice(reference, start, end)
    if block.empty:
        raise ValueError(f"no reference returns between {start} and {end}")
    shocks, basis = {}, {}
    if base is not None and keep_unmapped:
        shocks.update(base.shocks)
        basis.update(base.basis)
    used = []
    for factor, (col, scale) in mapping.items():
        if col in block.columns and block[col].notna().any():
            shocks[factor] = float(scale * block[col].sum(min_count=1))
            basis[factor] = "calibrated"
            used.append(f"{factor} = {scale:g} x {col}")
    if not used:
        raise ValueError("none of the mapped reference columns are in the data")
    window = (str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()))
    horizon = max(int(np.busday_count(pd.Timestamp(start).date(), pd.Timestamp(end).date())), 1)
    source = (f"Calibrated from reference factor returns {window[0]} to {window[1]}: "
              + "; ".join(used) + ".")
    if base is not None:
        if keep_unmapped and base.source:
            source += " Other shocks as before: " + base.source
        return replace(base, name=name or base.name, shocks=shocks, basis=basis,
                       window=window, horizon_days=horizon, source=source)
    return Scenario(name=name or f"reference_{window[0]}_{window[1]}", kind="historical",
                    shocks=shocks, horizon_days=horizon, window=window, basis=basis,
                    source=source)


def scales_from_comparison(comparison: pd.DataFrame) -> dict[str, tuple[str, float]]:
    """Mapping for :func:`calibrate_from_reference` from measured slopes.

    ``comparison`` is the output of :func:`riskmodels.factor_model.compare_factor_returns`;
    its ``slope`` is our factor return per unit of the reference return, which is exactly
    the scale wanted. Rows with no slope are skipped.
    """
    out = {}
    for factor, row in comparison.iterrows():
        if pd.notna(row.get("slope")):
            out[str(factor)] = (str(row["reference"]), float(row["slope"]))
    return out


def recalibrate_library(
    library: ScenarioLibrary,
    factor_returns: pd.DataFrame | None = None,
    reference: pd.DataFrame | None = None,
    mapping: dict[str, tuple[str, float]] | None = None,
) -> tuple[ScenarioLibrary, pd.DataFrame]:
    """Recalibrate every historical scenario the data covers.

    The model's own factor returns are preferred; the reference is used for windows the
    model does not cover. Returns the new library and one row per historical scenario
    saying what was done.
    """
    out = ScenarioLibrary(meta=library.meta)
    rows = []
    for s in library:
        if s.kind != "historical" or s.window is None:
            out.add(s)
            continue
        start, end = s.window
        if factor_returns is not None and _covers(factor_returns, start, end):
            out.add(calibrate_from_factor_returns(factor_returns, start, end, base=s))
            rows.append((s.name, "factor_model"))
        elif reference is not None and _covers(reference, start, end):
            out.add(calibrate_from_reference(reference, start, end, mapping, base=s))
            rows.append((s.name, "reference"))
        else:
            out.add(s)
            rows.append((s.name, "kept_approximate"))
    report = pd.DataFrame(rows, columns=["scenario", "calibrated_from"]).set_index("scenario")
    return out, report
