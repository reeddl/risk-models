"""Compare estimated factor returns with published factor returns (for example AQR's).

Our style factors are pure factor returns from a cross-sectional regression, while AQR's
are long/short portfolios (QMJ, BAB, HML Devil, UMD). They should be clearly correlated
but not identical. Expected sign: quality vs QMJ +, value vs HML Devil +, momentum vs UMD +,
beta vs BAB - (BAB is long low beta).
"""

from __future__ import annotations

import pandas as pd

DEFAULT_PAIRS = {"quality": "QMJ", "value": "HML Devil", "momentum": "UMD", "beta": "BAB"}


def compound(returns: pd.DataFrame | pd.Series, freq: str = "ME") -> pd.DataFrame | pd.Series:
    """Compound simple returns to a lower frequency ("ME" month end, "W-FRI" weekly)."""
    return (1 + returns).resample(freq).prod(min_count=1) - 1


def compare_factor_returns(
    ours: pd.DataFrame,
    reference: pd.DataFrame,
    pairs: dict[str, str] | None = None,
    freq: str = "ME",
) -> pd.DataFrame:
    """One row per pair: correlation, observations, annualized vols and the regression slope.

    ``ours`` is ``FactorModelResult.factor_returns`` (daily or weekly). ``reference`` holds the
    published series with one column per name in ``pairs`` values, at ``freq`` or higher.
    """
    pairs = pairs or {k: v for k, v in DEFAULT_PAIRS.items() if k in ours and v in reference}
    a = compound(ours[list(pairs)], freq)
    b = compound(reference[list(pairs.values())], freq)
    periods = {"M": 12, "W": 52}.get(freq[0], 12)
    rows = []
    for mine, theirs in pairs.items():
        j = pd.concat([a[mine], b[theirs]], axis=1, keys=["ours", "ref"]).dropna()
        if len(j) < 3:
            rows.append((mine, theirs, float("nan"), len(j), float("nan"), float("nan"), float("nan")))
            continue
        slope = j["ours"].cov(j["ref"]) / j["ref"].var()
        rows.append((mine, theirs, j["ours"].corr(j["ref"]), len(j),
                     j["ours"].std() * periods ** 0.5, j["ref"].std() * periods ** 0.5, slope))
    return pd.DataFrame(rows, columns=["factor", "reference", "correlation", "n_periods",
                                       "vol_ours", "vol_reference", "slope"]).set_index("factor")
