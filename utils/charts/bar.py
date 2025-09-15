"""Bar chart wrapper using Plotly Express.

The function is intentionally lightweight – pass a tidy *pandas.DataFrame*
and specify which columns to map to *x* and *y* (plus optional *color*).
Styling (corporate colours, fonts, etc.) is delegated to *utils.colors* and the
caller can further tweak the returned figure if required.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.colors import get_palette


def create_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: str | None = None,
    palette: str = "primary",
    show_total_in_legend: bool = True,
    theme: str = "professional",
) -> go.Figure:
    """Return a *plotly* bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    x, y : str
        Column names for the x-axis (categorical) and y-axis (numeric).
    color : str | None
        Optional column for grouping.  If *None* a single-trace bar is drawn.
    title : str | None
        Figure title.
    palette : str
        Name of colour palette to use – see ``utils.colors.get_palette``.
    show_total_in_legend : bool
        If *True* and *color* is provided, append category totals in legend
        labels ("Group A (Total: 123)").
    """
    if color is None:
        colours = get_palette(palette, 1)
        fig = px.bar(
            df,
            x=x,
            y=y,
            title=title,
            color_discrete_sequence=colours,
        )
    else:
        # Determine number of unique groups to pick palette length.
        n_groups = df[color].nunique()
        colours = get_palette(palette, n_groups)
        fig = px.bar(
            df,
            x=x,
            y=y,
            color=color,
            title=title,
            color_discrete_sequence=colours,
        )
        if show_total_in_legend:
            # Compute totals per group and update legend names.
            totals = df.groupby(color)[y].sum()
            new_names = {
                str(group): f"{group} (Total: {totals[group]})" for group in totals.index
            }
            fig.for_each_trace(lambda t: t.update(name=new_names.get(t.name, t.name)))

    # A bit of sensible default styling --------------------------------------
    from utils.style import apply_theme
    fig.update_layout(
        legend_title_text=color if color else None,
        title_x=0.5,
    )
    apply_theme(fig, theme)

    # Rotate x-tick labels if they are long.
    max_label_len = df[x].astype(str).str.len().max()
    if max_label_len > 12:
        fig.update_xaxes(tickangle=-45)

    return fig


__all__ = ["create_bar_chart"]