"""Box plot wrapper using Plotly Express."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.colors import get_palette
from utils.style import apply_theme


def create_box_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: str | None = None,
    palette: str = "primary",
    theme: str = "professional",
) -> go.Figure:
    """Return a plotly box plot."""
    n_colors = df[color].nunique() if color else 1
    fig = px.box(
        df,
        x=x,
        y=y,
        color=color,
        title=title,
        color_discrete_sequence=get_palette(palette, n_colors),
    )
    fig.update_layout(title_x=0.5)
    apply_theme(fig, theme)
    return fig

__all__ = ["create_box_chart"]