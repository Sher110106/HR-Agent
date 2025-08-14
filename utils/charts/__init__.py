"""Chart wrappers that use Plotly Express for rendering.

Each sub-module implements a single chart type and returns a *plotly.graph_objects.Figure*.
The wrappers deliberately keep their surface minimal – the calling code is
expected to handle Streamlit display, file export, etc.
"""
from __future__ import annotations

from .bar import create_bar_chart

from .line import create_line_chart
from .scatter import create_scatter_chart
from .histogram import create_histogram_chart
from .box import create_box_chart
from .violin import create_violin_chart
from .pie import create_pie_chart

from .spec import ChartSpec
from .dispatcher import create_chart

__all__ = [
    "create_bar_chart",
    "create_line_chart",
    "create_scatter_chart",
    "create_histogram_chart",
    "create_box_chart",
    "create_violin_chart",
    "create_pie_chart",
    "ChartSpec",
    "create_chart",
]