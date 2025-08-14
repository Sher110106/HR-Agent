from __future__ import annotations

from typing import Mapping

import pandas as pd
import plotly.graph_objects as go

from .spec import ChartSpec
from .bar import create_bar_chart
from .line import create_line_chart
from .scatter import create_scatter_chart
from .histogram import create_histogram_chart
from .box import create_box_chart
from .violin import create_violin_chart
from .pie import create_pie_chart


# ---------------------------------------------------------------------------
# Mapping of chart_type -> concrete creator function
# ---------------------------------------------------------------------------

_CREATOR_MAP: Mapping[str, callable] = {
    "bar": create_bar_chart,
    "line": create_line_chart,
    "scatter": create_scatter_chart,
    "histogram": create_histogram_chart,
    "box": create_box_chart,
    "violin": create_violin_chart,
    "pie": create_pie_chart,
}


def create_chart(df: pd.DataFrame, spec: ChartSpec) -> go.Figure:
    """Dispatch to the appropriate chart-creation helper.

    Raises
    ------
    KeyError
        If *spec.chart_type* is unknown.
    """
    if spec.chart_type not in _CREATOR_MAP:
        raise KeyError(f"Unsupported chart_type '{spec.chart_type}'. Available: {list(_CREATOR_MAP)}")

    creator = _CREATOR_MAP[spec.chart_type]

    if spec.chart_type == "bar":
        return creator(
            df=df,
            x=spec.x,
            y=spec.y or "count",
            color=spec.color,
            title=spec.title,
            palette=spec.palette,
            theme=spec.theme,
            **spec.extras,
        )
    elif spec.chart_type == "line":
        return creator(
            df=df,
            x=spec.x,
            y=spec.y or "value",
            color=spec.color,
            title=spec.title,
            palette=spec.palette,
            markers=spec.extras.get("markers", True),
            theme=spec.theme,
        )
    elif spec.chart_type == "scatter":
        return creator(
            df=df,
            x=spec.x,
            y=spec.y,
            color=spec.color,
            size=spec.size,
            title=spec.title,
            palette=spec.palette,
            trendline=spec.extras.get("trendline", False),
            theme=spec.theme,
        )
    elif spec.chart_type == "histogram":
        return creator(
            df=df,
            x=spec.x,
            color=spec.color,
            title=spec.title,
            palette=spec.palette,
            bins=spec.extras.get("bins", 30),
            theme=spec.theme,
        )
    elif spec.chart_type in ["box", "violin"]:
        return creator(
            df=df,
            x=spec.x,
            y=spec.y,
            color=spec.color,
            title=spec.title,
            palette=spec.palette,
            theme=spec.theme,
        )
    elif spec.chart_type == "pie":
        return creator(
            df=df,
            names=spec.x,
            values=spec.y,
            title=spec.title,
            palette=spec.palette,
            theme=spec.theme,
        )
    else:
        # Fallback – simply pass through extras
        return creator(df=df, x=spec.x, y=spec.y, title=spec.title, **spec.extras)
