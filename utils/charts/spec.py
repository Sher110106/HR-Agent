from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any

ChartType = Literal[
    "bar",
    "line",
    "scatter",
    "histogram",
    "box",
    "violin",
    "pie",
]


@dataclass
class ChartSpec:
    """Declarative specification for a chart request.

    Only a few fields are compulsory; the rest are optional and get sensible
    defaults inside the dispatcher.  Extra parameters specific to a chart can
    be supplied via *extras* so we don't need to constantly extend the base
    dataclass.
    """

    chart_type: ChartType
    x: str
    y: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None  # scatter bubble size
    title: Optional[str] = None
    palette: str = "primary"
    theme: str = "professional"
    extras: Dict[str, Any] = field(default_factory=dict)

    def with_updates(self, **updates) -> "ChartSpec":
        """Return a new *ChartSpec* with selected fields replaced."""
        data = {**self.__dict__, **updates}
        return ChartSpec(**data)