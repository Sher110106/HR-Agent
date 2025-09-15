"""Histogram chart wrapper using Plotly Express."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.colors import get_palette
from utils.style import apply_theme


def create_histogram_chart(
    df: pd.DataFrame,
    x: str,
    color: Optional[str] = None,
    title: str | None = None,
    palette: str = "primary",
    bins: int = 30,
    theme: str = "professional",
) -> go.Figure:
    """Return a plotly histogram chart."""
    n_colors = df[color].nunique() if color else 1
    fig = px.histogram(
        df,
        x=x,
        color=color,
        title=title,
        nbins=bins,
        color_discrete_sequence=get_palette(palette, n_colors),
    )
    fig.update_layout(title_x=0.5)
    apply_theme(fig, theme)
    return fig

__all__ = ["create_histogram_chart"]