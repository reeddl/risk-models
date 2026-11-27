"""AQR data library: published factor returns for benchmarking our factors.

AQR posts its data sets (QMJ, BAB, HML Devil, Momentum, Value and Momentum Everywhere, ...)
as Excel workbooks. Each sheet has a few lines of notes, then a header row whose first cell
is ``DATE``, then one row per date with a column per country or region ("USA", "Global", ...).
Returns are in decimals.

The download links change when AQR updates its site, so this module does not hard-code them.
Pass the workbook's URL as it appears on AQR's data-set page, or a local copy.
Reading .xlsx needs ``openpyxl`` (``pip install -e ".[aqr]"``).
AQR's terms of use apply to the data.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .http import Fetcher


def parse_aqr_sheet(raw: pd.DataFrame) -> pd.DataFrame:
    """A sheet read with ``header=None`` -> returns indexed by date, one column per region."""
    first = raw.iloc[:, 0].astype(str).str.strip().str.upper()
    hits = first.index[first == "DATE"]
    if len(hits) == 0:
        raise ValueError("no header row starting with DATE")
    h = hits[0]
    header = [str(c).strip() for c in raw.iloc[h]]
    body = raw.iloc[h + 1:].copy()
    body.columns = header
    body = body.loc[:, [c for c in header if c and c.lower() != "nan"]]
    dates = pd.to_datetime(body.iloc[:, 0], errors="coerce")
    body = body.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    body.index = pd.DatetimeIndex(dates, name="date")
    body = body[body.index.notna()]
    return body.dropna(how="all").sort_index()


class AqrLibrary:
    def __init__(self, client: Fetcher | None = None) -> None:
        self.client = client

    def _bytes(self, source: str | Path) -> bytes:
        s = str(source)
        if s.startswith("https://"):
            if self.client is None:
                raise ValueError("a client is needed to download from a URL")
            return self.client.get(s, max_age_days=30)
        return Path(s).read_bytes()

    def sheet_names(self, source: str | Path) -> list[str]:
        return list(pd.ExcelFile(io.BytesIO(self._bytes(source))).sheet_names)

    def load(self, source: str | Path, sheet: str) -> pd.DataFrame:
        """One sheet of an AQR workbook, e.g. ``load(url, "QMJ Factors")``."""
        raw = pd.read_excel(io.BytesIO(self._bytes(source)), sheet_name=sheet, header=None)
        return parse_aqr_sheet(raw)
