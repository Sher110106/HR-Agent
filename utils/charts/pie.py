"""Pie chart wrapper using Plotly Express."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.colors import get_palette
from utils.style import apply_theme


def create_pie_chart(
    df: pd.DataFrame,
    names: str,
    values: str,
    title: str | None = None,
    palette: str = "primary",
    theme: str = "professional",
) -> go.Figure:
    """Return a plotly pie chart."""
    n_colors = df[names].nunique()
    fig = px.pie(
        df,
        names=names,
        values=values,
        title=title,
        color_discrete_sequence=get_palette(palette, n_colors),
    )
    fig.update_layout(title_x=0.5)
    apply_theme(fig, theme)
    return fig

__all__ = ["create_pie_chart"]