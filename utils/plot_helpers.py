"""
Plot helper utilities for enhanced data visualization.
Used by the AI agents to create professional charts with value labels and proper formatting.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import textwrap
from typing import Union, List, Optional


# =========================================================================
#  add_value_labels
# -------------------------------------------------------------------------
# Adds numeric labels to bar, barh and line charts.
# New in v0.2:
#   •   `label_mode` parameter – 'auto' (default), 'bar', 'line', 'none'.
#       In auto-mode the helper will *skip* heat-maps / QuadMesh objects so
#       that seaborn/matplotlib heat-maps are not polluted with small dashes.
#   •   Uses matplotlib ≥3.4 `bar_label` when available for cleaner text
#       placement on vertical bars.
# =========================================================================

def add_value_labels(
    ax: plt.Axes,
    format_string: str = "{:.2f}",
    fontsize: int = 9,
    offset: float = 0.01,
    label_mode: str = "auto",
    min_display: float = 1e-4,  # skip bars with abs(height) below this
    place: str = "auto",  # 'auto'|'center'|'top' – where to place label on bar
) -> None:
    """
    Add value labels to bars in a matplotlib plot.
    
    Args:
        ax: The matplotlib axes object
        format_string: Format string for the values (default: "{:.2f}")
        fontsize: Font size for the labels (default: 9)
        offset: Vertical offset for label positioning (default: 0.01)
        label_mode: Label mode ('auto', 'bar', 'line', 'none')
        min_display: Minimum value to display label for (default: 1e-4)
        place: Placement of label ('auto', 'center', 'top')
    """
    # --------------------------------------------------
    # 0. Early-exit / skip modes
    # --------------------------------------------------
    if label_mode == "none":
        return

    # Auto-detect heat-map / QuadMesh (e.g., seaborn heatmap)
    if label_mode == "auto":
        from matplotlib.collections import QuadMesh
        if ax.collections and isinstance(ax.collections[0], QuadMesh):
            # Skip labelling heat-maps automatically
            return

    # --------------------------------------------------
    # 1. Bar / Barh charts
    # --------------------------------------------------
    for patch in ax.patches:
        # We only want actual bars (patches from bar/barh) – exclude spines etc.
        if not isinstance(patch, patches.Rectangle):
            continue

        height = patch.get_height()
        width = patch.get_width()

        # Heuristic to determine orientation (vertical vs horizontal)
        is_horizontal = width > height * 3  # barh has long width, small height

        # Skip zero-length bars
        bar_value = width if is_horizontal else height
        if np.isnan(bar_value) or abs(bar_value) < min_display:
            continue

        # ------------------------------
        # Numeric label formatting helper
        # ------------------------------
        def _fmt(val: float) -> str:
            if abs(val) >= 1000:
                return f"{val:,.0f}"
            elif abs(val) >= 1:
                return format_string.format(val)
            else:
                return f"{val:.3f}"

        label = _fmt(bar_value)

        if is_horizontal:
            # Horizontal bar ‑ place label at end of bar
            x_pos = patch.get_x() + bar_value + (ax.get_xlim()[1] - ax.get_xlim()[0]) * offset * (1 if bar_value > 0 else -1)
            y_pos = patch.get_y() + height / 2
            ha = 'left' if bar_value > 0 else 'right'
            if place == "center":
                va = 'center'
            else:  # 'top' or auto default
                y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * offset
                y_pos = bar_value + y_offset if bar_value > 0 else bar_value - y_offset
                va = 'bottom' if bar_value > 0 else 'top'
        else:
            # Vertical bar
            x_pos = patch.get_x() + width / 2
            if place == "center":
                y_pos = bar_value / 2
                va = 'center'
            else:  # 'top' or auto default
                y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * offset
                y_pos = bar_value + y_offset if bar_value > 0 else bar_value - y_offset
                va = 'bottom' if bar_value > 0 else 'top'
            ha = 'center'

        # Try using bar_label (Matplotlib >=3.4) for vertical bars
        used_bar_label = False
        if not is_horizontal:
            try:
                import matplotlib
                if matplotlib.__version__ >= "3.4":
                    ax.bar_label([patch], labels=[label], padding=3,
                                 fontsize=fontsize, weight='bold')
                    used_bar_label = True
            except Exception:
                used_bar_label = False

        if not used_bar_label:
            ax.text(x_pos, y_pos, label,
                    ha=ha, va=va,
                    fontsize=fontsize, fontweight='bold')

    # ------------------------------
    # Line plots (with markers)
    # ------------------------------
    if label_mode not in ("auto", "line", "both", "all"):
        return  # Skip line plotting if mode excludes it

    for line in ax.get_lines():
        if line.get_marker() != 'None':
            x_data = line.get_xdata(orig=False)
            y_data = line.get_ydata(orig=False)

            for x_val, y_val in zip(x_data, y_data):
                if np.isnan(y_val):
                    continue

                # Format label
                if abs(y_val) >= 1000:
                    label = f"{y_val:,.0f}"
                elif abs(y_val) >= 1:
                    label = format_string.format(y_val)
                else:
                    label = f"{y_val:.3f}"

                y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * offset
                ax.text(x_val, y_val + y_offset,
                        label,
                        ha='center', va='bottom',
                        fontsize=fontsize-1, fontweight='bold')


def format_axis_labels(ax: plt.Axes, x_rotation: float = 45, 
                      wrap_length: int = 15, max_labels: int = 20) -> None:
    """
    Format axis labels for better readability.
    
    Args:
        ax: The matplotlib axes object
        x_rotation: Rotation angle for x-axis labels (default: 45)
        wrap_length: Maximum characters per line for wrapping (default: 15)
        max_labels: Maximum number of labels to show (default: 20)
    """
    # Format x-axis labels
    x_labels = ax.get_xticklabels()
    if len(x_labels) > max_labels:
        # If too many labels, show every nth label
        step = len(x_labels) // max_labels + 1
        for i, label in enumerate(x_labels):
            if i % step != 0:
                label.set_visible(False)
    
    # Wrap long labels
    wrapped_labels = []
    for label in x_labels:
        text = label.get_text()
        if len(text) > wrap_length:
            wrapped_text = '\n'.join(textwrap.wrap(text, wrap_length))
            wrapped_labels.append(wrapped_text)
        else:
            wrapped_labels.append(text)
    
    # Apply formatting
    if wrapped_labels:
        ax.set_xticklabels(wrapped_labels, rotation=x_rotation, ha='right')
    
    # Format y-axis labels for readability
    y_labels = ax.get_yticklabels()
    for label in y_labels:
        text = label.get_text()
        try:
            # Format large numbers with commas
            if text and float(text) >= 1000:
                formatted = f"{float(text):,.0f}"
                label.set_text(formatted)
        except (ValueError, TypeError):
            pass


def apply_professional_styling(ax: plt.Axes, title: str = "", 
                              xlabel: str = "", ylabel: str = "") -> None:
    """
    Apply professional styling to the plot.
    
    Args:
        ax: The matplotlib axes object
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    # Set title and labels
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    
    # Apply grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make remaining spines less prominent
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    
    # Set background color
    ax.set_facecolor('#fafafa')


def get_professional_colors(palette: str = 'primary') -> dict:
    """
    Get professional color palettes for charts.
    
    Args:
        palette: Color palette name ('primary', 'secondary', 'categorical')
        
    Returns:
        Dictionary with color arrays
    """
    palettes = {
        'primary': {
            'colors': ['#2E86C1', '#28B463', '#F39C12', '#E74C3C', '#8E44AD', '#17A2B8'],
            'single': '#2E86C1'
        },
        'secondary': {
            'colors': ['#5D6D7E', '#85929E', '#AEB6BF', '#D5DBDB', '#F8F9FA', '#E8EAED'],
            'single': '#5D6D7E'
        },
        'categorical': {
            'colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'],
            'single': '#FF6B6B'
        }
    }
    return palettes.get(palette, palettes['primary'])