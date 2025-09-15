"""Migration shims to bridge old matplotlib calls to new Plotly wrappers.

This module provides compatibility functions that keep the same signature as
the old `create_clean_*` helpers but return Plotly figures instead of
matplotlib objects. This allows existing code to work while we migrate
to the new chart system.
"""
from __future__ import annotations

from typing import Optional, Any, Union, List
import pandas as pd
import plotly.graph_objects as go
import warnings

from utils.charts import ChartSpec, create_chart


def create_clean_bar_chart(
    ax: Any,  # Ignored - we return Plotly Figure instead
    data_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: Optional[str] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend_totals: bool = True,
    theme: str = "professional",
    add_annotations: bool = True,
    **kwargs
) -> go.Figure:
    """Create bar chart using new Plotly system."""
    spec = ChartSpec(
        chart_type="bar",
        x=x_col,
        y=y_col,
        color=hue_col,
        title=title,
        palette=kwargs.get("palette", "primary"),
        theme=theme,
        extras={"show_total_in_legend": legend_totals}
    )
    return create_chart(data_df, spec)


def create_clean_line_chart(
    ax: Any,
    data_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: Optional[str] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    show_markers: bool = True,
    line_style: str = '-',
    theme: str = "professional",
    **kwargs
) -> go.Figure:
    """Create line chart using new Plotly system."""
    spec = ChartSpec(
        chart_type="line",
        x=x_col,
        y=y_col,
        color=hue_col,
        title=title,
        palette=kwargs.get("palette", "primary"),
        theme=theme,
        extras={"markers": show_markers}
    )
    return create_chart(data_df, spec)


def create_clean_scatter_plot(
    ax: Any,
    data_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: Optional[str] = None,
    size_col: Optional[str] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    add_trendline: bool = False,
    theme: str = "professional",
    **kwargs
) -> go.Figure:
    """Create scatter plot using new Plotly system."""
    spec = ChartSpec(
        chart_type="scatter",
        x=x_col,
        y=y_col,
        color=hue_col,
        size=size_col,
        title=title,
        palette=kwargs.get("palette", "primary"),
        theme=theme,
        extras={"trendline": add_trendline}
    )
    return create_chart(data_df, spec)


def create_clean_histogram(
    ax: Any,
    data_df: pd.DataFrame,
    col: str,
    bins: int = 30,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Frequency",
    show_stats: bool = True,
    theme: str = "professional",
    **kwargs
) -> go.Figure:
    """Create histogram using new Plotly system."""
    spec = ChartSpec(
        chart_type="histogram",
        x=col,
        title=title,
        palette=kwargs.get("palette", "primary"),
        theme=theme,
        extras={"bins": bins}
    )
    return create_chart(data_df, spec)


def create_clean_box_plot(
    ax: Any,
    data_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    show_outliers: bool = True,
    theme: str = "professional",
    **kwargs
) -> go.Figure:
    """Create box plot using new Plotly system."""
    spec = ChartSpec(
        chart_type="box",
        x=x_col,
        y=y_col,
        title=title,
        palette=kwargs.get("palette", "primary"),
        theme=theme
    )
    return create_chart(data_df, spec)


def create_clean_violin_plot(
    ax: Any,
    data_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    theme: str = "professional",
    show_points: bool = False,
    **kwargs
) -> go.Figure:
    """Create violin plot using new Plotly system."""
    spec = ChartSpec(
        chart_type="violin",
        x=x_col,
        y=y_col,
        title=title,
        palette=kwargs.get("palette", "primary"),
        theme=theme
    )
    return create_chart(data_df, spec)


def create_clean_pie_chart(
    ax: Any,
    data_df: pd.DataFrame,
    col: str,
    title: str = "",
    show_percentages: bool = True,
    explode_max: bool = True,
    theme: str = "professional",
    **kwargs
) -> go.Figure:
    """Create pie chart using new Plotly system."""
    # For pie charts, we need to aggregate the data first
    value_counts = data_df[col].value_counts().reset_index()
    value_counts.columns = [col, 'count']
    
    spec = ChartSpec(
        chart_type="pie",
        x=col,
        y='count',
        title=title,
        palette=kwargs.get("palette", "primary"),
        theme=theme
    )
    return create_chart(value_counts, spec)


# =========================================================================
#  Matplotlib-style Helper Functions (for compatibility)
# =========================================================================

def get_professional_colors(palette: str = 'primary') -> dict:
    """Get professional color palettes for charts."""
    from utils.colors import get_palette
    
    palettes = {
        'primary': {
            'colors': get_palette('primary'),
            'single': get_palette('primary')[0]
        },
        'secondary': {
            'colors': ['#5D6D7E', '#85929E', '#AEB6BF', '#D5DBDB', '#F8F9FA'],
            'single': '#5D6D7E'
        },
        'categorical': {
            'colors': get_palette('categorical'),
            'single': get_palette('categorical')[0]
        },
        'hr_specific': {
            'colors': get_palette('hr_specific'),
            'single': get_palette('hr_specific')[0]
        }
    }
    
    return palettes.get(palette, palettes['primary'])


def safe_color_access(colors, index):
    """Safely access colors with cycling if index is out of range."""
    if not colors:
        return '#2E86C1'  # Default fallback color
    return colors[index % len(colors)]


def create_category_palette(categories, palette_name='primary'):
    """Create a seaborn-compatible palette dictionary for specific categories."""
    colors = get_professional_colors(palette_name)['colors']
    palette = {}
    
    for i, category in enumerate(categories):
        palette[category] = colors[i % len(colors)]
    
    return palette


def apply_professional_styling(ax: Any, title: str = "", xlabel: str = "", ylabel: str = "", theme: str = "professional") -> None:
    """Apply professional styling - no-op for Plotly compatibility."""
    # This is a no-op since Plotly handles styling internally
    # Kept for compatibility with existing matplotlib code
    pass


def format_axis_labels(ax: Any, x_rotation: int = 0, y_rotation: int = 0) -> None:
    """Format axis labels - no-op for Plotly compatibility."""
    # This is a no-op since Plotly handles axis formatting internally
    # Kept for compatibility with existing matplotlib code
    pass


def optimize_figure_size(ax: Any) -> None:
    """Optimize figure size - no-op for Plotly compatibility."""
    # This is a no-op since Plotly handles sizing automatically
    # Kept for compatibility with existing matplotlib code
    pass


def add_value_labels(ax: Any, bars: Any = None, format_str: str = '{:.0f}', **kwargs) -> None:
    """Add value labels - no-op for Plotly compatibility."""
    # This is a no-op since Plotly can handle value labels internally
    # Kept for compatibility with existing matplotlib code
    pass


def handle_seaborn_warnings():
    """Suppress harmless seaborn warnings."""
    warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
    warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


def safe_binning(data: pd.Series, bins: Union[int, List], labels: List = None, 
                method: str = 'cut', **kwargs) -> pd.Series:
    """Safely create bins using pd.cut or pd.qcut with automatic validation."""
    try:
        if method == 'cut':
            if isinstance(bins, int):
                # Automatic binning without labels
                return pd.cut(data, bins=bins, **kwargs)
            else:
                # Custom bins - validate labels
                if labels is not None and len(labels) != len(bins) - 1:
                    # Fix labels to match bins
                    if len(labels) > len(bins) - 1:
                        labels = labels[:len(bins) - 1]
                    else:
                        # Use automatic labels instead
                        return pd.cut(data, bins=bins, **kwargs)
                return pd.cut(data, bins=bins, labels=labels, **kwargs)
        elif method == 'qcut':
            return pd.qcut(data, q=bins, labels=labels, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    except Exception as e:
        # Fallback to simple binning
        warnings.warn(f"Binning failed with error: {e}. Using automatic binning.")
        return pd.cut(data, bins=5)  # Simple fallback