"""Stress scenarios stored as data: factor-return shocks keyed to the factor model's names.

A scenario says how much each factor returns over its horizon. The default library lives
in ``data/scenarios.json``; load it with :func:`default_library`, or keep your own JSON in
the same format and load it with :meth:`ScenarioLibrary.from_json`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from importlib import resources
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

KINDS = ("historical", "hypothetical")
BASES = ("index", "judgment", "calibrated")


@dataclass(frozen=True)
class Scenario:
    """One stress scenario.

    Attributes
    ----------
    name, description : str
    kind : ``"historical"`` or ``"hypothetical"``
    shocks : dict
        Factor name -> cumulative factor return over the scenario (``"market"``, a style
        name such as ``"momentum"``, or ``"industry:<name>"``). Factors not listed are
        filled in by the engine (conditional on these, or zero; see ``StressConfig``).
    horizon_days : int
        Trading days the scenario unfolds over. Sets the size of the stressed specific
        risk and is reported next to the loss.
    window : (start, end) or None
        Calendar window of a historical episode; used by the calibration functions.
    basis : dict
        Factor name -> where its number came from: ``"index"`` (a rounded published index
        move), ``"judgment"`` or ``"calibrated"`` (measured by a calibration function).
    source : str
        Plain-language note on where the numbers came from.
    vol_multiplier : float
        Stock-specific volatility multiplier in this scenario (also scales the impact cost
        in the liquidity overlay).
    volume_multiplier, spread_multiplier : float
        Stressed ADV and spreads for the liquidity overlay (0.5 = half of normal volume).
    key : bool
        Repeated with stressed liquidity in the liquidity overlay.
    tags : tuple of str
        Free labels for selecting scenarios ("rates", "crash", "liquidity", ...).
    """

    name: str
    kind: Literal["historical", "hypothetical"]
    shocks: dict
    horizon_days: int = 1
    description: str = ""
    window: tuple[str, str] | None = None
    basis: dict = field(default_factory=dict)
    source: str = ""
    vol_multiplier: float = 2.0
    volume_multiplier: float = 0.5
    spread_multiplier: float = 2.0
    key: bool = False
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"scenario {self.name!r}: kind must be one of {KINDS}")
        if not self.shocks:
            raise ValueError(f"scenario {self.name!r} has no shocks")
        if self.horizon_days < 1:
            raise ValueError(f"scenario {self.name!r}: horizon_days must be at least 1")
        for k, v in self.basis.items():
            if v not in BASES:
                raise ValueError(f"scenario {self.name!r}: basis for {k!r} must be one of {BASES}")
        for name in ("vol_multiplier", "volume_multiplier", "spread_multiplier"):
            if not getattr(self, name) > 0:
                raise ValueError(f"scenario {self.name!r}: {name} must be positive")
        object.__setattr__(self, "shocks", {str(k): float(v) for k, v in self.shocks.items()})
        object.__setattr__(self, "tags", tuple(self.tags))
        if self.window is not None:
            object.__setattr__(self, "window", tuple(self.window))

    @property
    def calibration(self) -> str:
        """``"calibrated"`` if every shock was measured, else ``"approximate"``."""
        if self.basis and all(v == "calibrated" for v in self.basis.values()):
            return "calibrated"
        return "approximate"

    def scaled(self, multiple: float, name: str | None = None) -> "Scenario":
        """The same scenario with every shock multiplied by ``multiple``."""
        return replace(self, name=name or f"{self.name}_x{multiple:g}",
                       shocks={k: v * multiple for k, v in self.shocks.items()})

    def with_shocks(self, **shocks: float) -> "Scenario":
        """A copy with some shocks replaced or added (basis set to judgment)."""
        new = dict(self.shocks)
        basis = dict(self.basis)
        for k, v in shocks.items():
            new[k] = v
            basis[k] = "judgment"
        return replace(self, shocks=new, basis=basis)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["window"] = list(self.window) if self.window else None
        d["tags"] = list(self.tags)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


class ScenarioLibrary:
    """An ordered collection of scenarios with lookup and selection."""

    def __init__(self, scenarios: Iterable[Scenario] = (), meta: dict | None = None) -> None:
        self._items: dict[str, Scenario] = {}
        self.meta = dict(meta or {})
        for s in scenarios:
            self.add(s)

    def add(self, scenario: Scenario, replace_existing: bool = False) -> "ScenarioLibrary":
        if scenario.name in self._items and not replace_existing:
            raise ValueError(f"scenario {scenario.name!r} is already in the library")
        self._items[scenario.name] = scenario
        return self

    def __getitem__(self, name: str) -> Scenario:
        return self._items[name]

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __iter__(self):
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    @property
    def names(self) -> list[str]:
        return list(self._items)

    def select(self, names: Iterable[str] | None = None, kind: str | None = None,
               tags: Iterable[str] | None = None, key: bool | None = None) -> "ScenarioLibrary":
        """Scenarios matching every filter given (``tags`` matches any of the tags)."""
        tags = set(tags) if tags is not None else None
        wanted = list(names) if names is not None else None
        if wanted is not None:
            missing = [n for n in wanted if n not in self._items]
            if missing:
                raise KeyError(f"no scenarios named {missing}")
        out = []
        for s in self._items.values():
            if wanted is not None and s.name not in wanted:
                continue
            if kind is not None and s.kind != kind:
                continue
            if tags is not None and not tags & set(s.tags):
                continue
            if key is not None and s.key != key:
                continue
            out.append(s)
        return ScenarioLibrary(out, self.meta)

    def shock_frame(self) -> pd.DataFrame:
        """Factor x scenario table of the stored shocks (NaN where not set)."""
        return pd.DataFrame({s.name: pd.Series(s.shocks) for s in self}).sort_index()

    def to_frame(self) -> pd.DataFrame:
        """One row per scenario with its settings (not the shocks)."""
        rows = []
        for s in self:
            rows.append({
                "name": s.name, "kind": s.kind, "horizon_days": s.horizon_days,
                "window_start": s.window[0] if s.window else None,
                "window_end": s.window[1] if s.window else None,
                "calibration": s.calibration, "n_shocks": len(s.shocks),
                "vol_multiplier": s.vol_multiplier, "volume_multiplier": s.volume_multiplier,
                "spread_multiplier": s.spread_multiplier, "key": s.key,
                "tags": ",".join(s.tags), "description": s.description,
            })
        return pd.DataFrame(rows).set_index("name")

    def to_json(self, path: str | Path | None = None) -> str:
        payload = {**self.meta, "scenarios": [s.to_dict() for s in self]}
        text = json.dumps(payload, indent=2)
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_json(cls, source: str | Path) -> "ScenarioLibrary":
        """Load from a JSON file path or a JSON string."""
        text = str(source)
        if not text.lstrip().startswith("{"):
            text = Path(source).read_text()
        payload = json.loads(text)
        meta = {k: v for k, v in payload.items() if k != "scenarios"}
        return cls([Scenario.from_dict(d) for d in payload["scenarios"]], meta)


def default_library() -> ScenarioLibrary:
    """The shipped library: 8 historical and 8 hypothetical scenarios (approximate)."""
    text = resources.files("riskmodels.stress").joinpath("data/scenarios.json").read_text()
    return ScenarioLibrary.from_json(text)
