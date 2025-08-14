"""Line chart wrapper using Plotly Express."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.colors import get_palette
from utils.style import apply_theme


def create_line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: str | None = None,
    palette: str = "primary",
    markers: bool = True,
    theme: str = "professional",
) -> go.Figure:
    """Return a plotly line chart with sensible defaults."""
    if color is None:
        fig = px.line(
            df,
            x=x,
            y=y,
            title=title,
            markers=markers,
            color_discrete_sequence=get_palette(palette, 1),
        )
    else:
        n_groups = df[color].nunique()
        fig = px.line(
            df,
            x=x,
            y=y,
            color=color,
            title=title,
            markers=markers,
            color_discrete_sequence=get_palette(palette, n_groups),
        )
    fig.update_layout(title_x=0.5)
    apply_theme(fig, theme)
    return fig

__all__ = ["create_line_chart"]