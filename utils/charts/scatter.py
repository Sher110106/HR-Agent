"""Scatter chart wrapper using Plotly Express."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.colors import get_palette
from utils.style import apply_theme


def create_scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    size: Optional[str] = None,
    title: str | None = None,
    palette: str = "primary",
    theme: str = "professional",
    trendline: bool = False,
) -> go.Figure:
    """Return a Plotly scatter plot.

    If *trendline* is True, a OLS trendline is added when both axes are numeric.
    """
    n_colors = df[color].nunique() if color else 1
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        size=size,
        title=title,
        color_discrete_sequence=get_palette(palette, n_colors),
        trendline="ols" if trendline else None,
    )
    fig.update_layout(title_x=0.5)
    apply_theme(fig, theme)
    return fig

__all__ = ["create_scatter_chart"]