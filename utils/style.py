"""Styling helpers for Plotly figures.

These wrappers centralise common layout tweaks so that each chart module
can call them in one line instead of repeating boilerplate settings.
"""
from __future__ import annotations

import plotly.graph_objects as go

_DEFAULT_FONT_FAMILY = "Arial, Helvetica, sans-serif"

_THEMES = {
    "professional": dict(
        template="plotly_white",
        font=dict(family=_DEFAULT_FONT_FAMILY, size=12, color="#333"),
        title=dict(font=dict(size=16, color="#111")),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(bgcolor="white"),
    ),
    "minimal": dict(
        template="plotly_white",
        font=dict(family=_DEFAULT_FONT_FAMILY, size=11, color="#444"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(bgcolor="white"),
    ),
    "dark": dict(
        template="plotly_dark",
        paper_bgcolor="#2F2F2F",
        plot_bgcolor="#2F2F2F",
    ),
}


def apply_theme(fig: go.Figure, theme: str = "professional") -> go.Figure:
    """Apply a named theme to *fig* and return it (chain-able)."""
    config = _THEMES.get(theme, _THEMES["professional"])
    fig.update_layout(**config)
    return fig

__all__ = ["apply_theme"]