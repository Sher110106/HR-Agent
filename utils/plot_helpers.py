"""
Plot helper utilities for enhanced data visualization.
Used by the AI agents to create professional charts with value labels and proper formatting.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import textwrap
from typing import Union, List, Optional
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import warnings
from datetime import datetime
import pandas as pd

# Import Phase 3 quality system
try:
    from .plot_quality_system import PlotQualitySystem
    PHASE3_AVAILABLE = True
except ImportError:
    PHASE3_AVAILABLE = False
    PlotQualitySystem = None

# =========================================================================
#  PHASE 1: Enhanced Color Systems & Modern Aesthetics
# =========================================================================

def get_hr_specific_colors():
    """
    Get HR-specific color palettes for contextual visualization.
    
    Returns:
        dict: Dictionary with HR-specific color schemes
    """
    return {
        'attrition': ['#e74c3c', '#c0392b', '#a93226', '#922b21', '#7b241c'],
        'retention': ['#27ae60', '#229954', '#1e8449', '#196f3d', '#145a32'],
        'performance': ['#f39c12', '#e67e22', '#d35400', '#ba4a00', '#a04000'],
        'neutral': ['#3498db', '#2980b9', '#1f618d', '#154360', '#0e2f44'],
        'gradient': ['#3498db', '#2980b9', '#1f618d', '#154360', '#0e2f44'],
        'accessibility': ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd'],
        'modern': ['#2c3e50', '#34495e', '#3498db', '#2980b9', '#1abc9c', '#16a085', '#f39c12', '#e67e22']
    }

def get_gradient_colormap(colors: List[str], name: str = 'custom') -> LinearSegmentedColormap:
    """
    Create a gradient colormap for continuous data visualization.
    
    Args:
        colors: List of color codes for the gradient
        name: Name for the colormap
    Returns:
        LinearSegmentedColormap: Custom gradient colormap
    """
    return LinearSegmentedColormap.from_list(name, colors)

def get_contextual_colors(data_type: str, values: List = None) -> List[str]:
    """
    Get contextual colors based on data type and values.
    
    Args:
        data_type: Type of data ('attrition', 'retention', 'performance', 'neutral')
        values: Optional list of values to determine color intensity
    Returns:
        List[str]: Contextual color palette
    """
    hr_colors = get_hr_specific_colors()
    
    if data_type in hr_colors:
        base_colors = hr_colors[data_type]
    else:
        base_colors = hr_colors['neutral']
    
    if values and len(values) > len(base_colors):
        # Extend palette if needed
        while len(base_colors) < len(values):
            base_colors.extend(base_colors[:len(values) - len(base_colors)])
    
    return base_colors[:len(values)] if values else base_colors

def apply_modern_typography(ax: plt.Axes, font_family: str = 'system') -> None:
    """
    Apply modern typography with responsive font sizing.
    
    Args:
        ax: Matplotlib axes object
        font_family: Font family to use ('system', 'inter', 'roboto')
    """
    # Set modern font family
    if font_family == 'inter':
        plt.rcParams['font.family'] = 'Inter'
    elif font_family == 'roboto':
        plt.rcParams['font.family'] = 'Roboto'
    else:
        # Use system fonts
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    # Responsive font sizing based on figure dimensions
    fig = ax.get_figure()
    fig_width, fig_height = fig.get_size_inches()
    
    # Calculate responsive font sizes
    base_title_size = max(10, min(16, int(fig_width * 1.2)))
    base_label_size = max(8, min(12, int(fig_width * 0.9)))
    base_tick_size = max(6, min(10, int(fig_width * 0.7)))
    
    # Apply to existing elements
    title = ax.get_title()
    if title:
        ax.set_title(title, fontsize=base_title_size, fontweight='bold', pad=20)
    
    xlabel = ax.get_xlabel()
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=base_label_size, fontweight='bold')
    
    ylabel = ax.get_ylabel()
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=base_label_size, fontweight='bold')
    
    # Update tick label sizes
    ax.tick_params(axis='both', labelsize=base_tick_size)

def apply_golden_ratio_spacing(ax: plt.Axes) -> None:
    """
    Apply golden ratio spacing principles for better visual balance.
    """
    # Golden ratio constant
    phi = 1.618033988749895
    
    # Adjust margins based on golden ratio
    fig = ax.get_figure()
    fig_width, fig_height = fig.get_size_inches()
    
    # Calculate optimal margins
    total_width = fig_width
    total_height = fig_height
    
    # Apply golden ratio to spacing
    left_margin = total_width / (phi * 2)
    right_margin = total_width / (phi * 2)
    bottom_margin = total_height / (phi * 2)
    top_margin = total_height / (phi * 2)
    
    # Normalize margins
    left_norm = left_margin / total_width
    right_norm = right_margin / total_width
    bottom_norm = bottom_margin / total_height
    top_norm = top_margin / total_height
    
    # Apply margins
    ax.set_position([left_norm, bottom_norm, 1 - left_norm - right_norm, 1 - bottom_norm - top_norm])

def apply_modern_styling(ax: plt.Axes, theme: str = 'professional') -> None:
    """
    Apply modern styling to matplotlib axes.
    
    Args:
        ax: Matplotlib axes object
        theme: Styling theme ('professional', 'modern', 'minimal', 'elegant')
    """
    # Set theme colors
    if theme == 'professional':
        colors = get_professional_colors('primary')
    elif theme == 'modern':
        colors = get_professional_colors('modern')
    elif theme == 'minimal':
        colors = get_professional_colors('minimal')
    elif theme == 'elegant':
        colors = get_professional_colors('elegant')
    else:
        colors = get_professional_colors('primary')
    
    # Apply typography
    apply_modern_typography(ax)
    
    # Apply spacing
    apply_golden_ratio_spacing(ax)
    
    # Hide spines (except for polar plots)
    if not isinstance(ax, plt.PolarAxes):
        if 'top' in ax.spines:
            ax.spines['top'].set_visible(False)
        if 'right' in ax.spines:
            ax.spines['right'].set_visible(False)
        if 'left' in ax.spines:
            ax.spines['left'].set_alpha(0.3)
        if 'bottom' in ax.spines:
            ax.spines['bottom'].set_alpha(0.3)
    
    # Set grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set background color with fallback
    background_color = colors.get('background', '#f8f9fa')
    ax.set_facecolor(background_color)
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=1)
    ax.tick_params(axis='both', which='minor', labelsize=8, length=3, width=0.5)

def add_smart_annotations(ax: plt.Axes, data_df: 'pd.DataFrame', highlight_insights: bool = True) -> None:
    """
    Add contextual annotations based on data patterns.
    
    Args:
        ax: Matplotlib axes object
        data_df: DataFrame with data
        highlight_insights: Whether to highlight key insights
    """
    if not highlight_insights:
        return
    
    # Get the plot type and data
    children = ax.get_children()
    bars = [child for child in children if hasattr(child, 'get_height')]
    
    if bars:
        # Bar chart annotations
        heights = [bar.get_height() for bar in bars]
        if heights:
            max_height = max(heights)
            min_height = min(heights)
            
            # Annotate maximum value
            max_bar = bars[heights.index(max_height)]
            x_pos = max_bar.get_x() + max_bar.get_width() / 2
            y_pos = max_height + max_height * 0.05
            
            ax.annotate(f'Highest: {max_height:.1f}',
                       (x_pos, y_pos),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       va='bottom',
                       fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#e74c3c', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', alpha=0.6))
            
            # Annotate minimum value if significantly different
            if max_height > min_height * 1.5:
                min_bar = bars[heights.index(min_height)]
                x_pos = min_bar.get_x() + min_bar.get_width() / 2
                y_pos = min_height + min_height * 0.05
                
                ax.annotate(f'Lowest: {min_height:.1f}',
                           (x_pos, y_pos),
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center',
                           va='bottom',
                           fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='#3498db', alpha=0.8),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', alpha=0.6))

def create_gradient_background(ax: plt.Axes, color_start: str = '#ffffff', color_end: str = '#f8f9fa') -> None:
    """
    Create a subtle gradient background for depth.
    
    Args:
        ax: Matplotlib axes object
        color_start: Starting color for gradient
        color_end: Ending color for gradient
    """
    fig = ax.get_figure()
    
    # Create gradient background
    gradient = np.linspace(0, 1, 100)
    gradient = np.vstack((gradient, gradient))
    
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list('background_gradient', [color_start, color_end])
    
    # Apply gradient to figure background
    fig.patch.set_facecolor(color_start)
    ax.set_facecolor(color_end)

# =========================================================================
#  Enhanced Professional Colors (Updated)
# =========================================================================

def get_professional_colors(palette: str = 'primary') -> dict:
    """
    Get professional color palettes for charts with enhanced options.

    Args:
        palette (str): Color palette name ('primary', 'secondary', 'categorical', 'hr_specific').
    Returns:
        dict: Dictionary with color arrays.
    """
    palettes = {
        'primary': {
            'colors': ['#2E86C1', '#28B463', '#F39C12', '#E74C3C', '#8E44AD', '#17A2B8', '#FFC107', '#6C757D', '#20C997', '#FD7E14'],
            'single': '#2E86C1'
        },
        'secondary': {
            'colors': ['#5D6D7E', '#85929E', '#AEB6BF', '#D5DBDB', '#F8F9FA', '#E8EAED', '#CED4DA', '#6C757D', '#495057', '#343A40'],
            'single': '#5D6D7E'
        },
        'categorical': {
            'colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#FFB6C1', '#98FB98', '#F0E68C', '#DEB887'],
            'single': '#FF6B6B'
        },
        'hr_specific': {
            'colors': ['#2E86C1', '#28B463', '#F39C12', '#E74C3C', '#8E44AD', '#17A2B8', '#FFC107', '#6C757D', '#20C997', '#FD7E14'],
            'single': '#2E86C1'
        }
    }
    
    # Add HR-specific colors if requested
    if palette == 'hr_specific':
        hr_colors = get_hr_specific_colors()
        palettes['hr_specific']['colors'] = hr_colors['modern']
    
    return palettes.get(palette, palettes['primary'])

# =========================================================================
#  Original Functions (Enhanced with Phase 1 Features)
# =========================================================================

def add_value_labels(
    ax: plt.Axes,
    format_string: str = "{:.1f}",
    fontsize: int = 8,
    offset: float = 0.02,
    label_mode: str = "minimal",
    min_display: float = 0.1,
    place: str = "auto",
) -> None:
    """
    Add minimal, clean value labels to charts - only when truly helpful.
    
    Args:
        ax: The matplotlib axes object
        format_string: Format for numeric values (default: 1 decimal place)
        fontsize: Small font size for minimal impact (default: 8)
        label_mode: 'minimal' (default), 'selective', or 'none'
        min_display: Minimum value to display labels for
        place: Label placement - 'auto', 'center', or 'top'
    """
    if label_mode == "none":
        return
    
    # Get all bar/patch objects
    bars = [child for child in ax.get_children() 
            if hasattr(child, 'get_height') and hasattr(child, 'get_width')]
    
    if not bars:
        return
    
    # For minimal mode, only label significant bars or outliers
    if label_mode == "minimal":
        heights = [bar.get_height() for bar in bars if bar.get_height() > min_display]
        if not heights:
            return
        
        # Only label if we have fewer than 8 bars to avoid clutter
        if len(bars) > 8:
            return
        
        # Only label the highest and lowest values, or all if <= 4 bars
        if len(bars) <= 4:
            bars_to_label = bars
        else:
            # Label only max and min values
            max_bar = max(bars, key=lambda b: b.get_height())
            min_bar = min(bars, key=lambda b: b.get_height())
            bars_to_label = [max_bar, min_bar]
    else:
        # Selective mode - label all significant values
        bars_to_label = [bar for bar in bars if bar.get_height() > min_display]
    
    # Add labels to selected bars
    for bar in bars_to_label:
        height = bar.get_height()
        if height < min_display:
            continue
            
        # Format the value
        if abs(height) >= 1000:
            label = f"{height/1000:.1f}k"
        elif abs(height) >= 1:
            label = format_string.format(height)
        else:
            label = f"{height:.2f}"
        
        # Determine label position
        x = bar.get_x() + bar.get_width() / 2
        if place == "center":
            y = height / 2
            va = 'center'
        else:  # "auto" or "top"
            y = height + offset * ax.get_ylim()[1]
            va = 'bottom'
        
        # Add the label with minimal styling
        ax.text(x, y, label, 
               ha='center', va=va,
               fontsize=fontsize,
               color='#333333',
               weight='normal',
               alpha=0.8)

def format_axis_labels(
    ax: plt.Axes, 
    x_rotation: float = 45, 
    wrap_length: int = 15, 
    max_labels: int = 20
) -> None:
    """
    Format axis labels for better readability.
    """
    x_labels = ax.get_xticklabels()
    if len(x_labels) > max_labels:
        step = len(x_labels) // max_labels + 1
        for i, label in enumerate(x_labels):
            if i % step != 0:
                label.set_visible(False)
    wrapped_labels = []
    for label in x_labels:
        text = label.get_text()
        if len(text) > wrap_length:
            wrapped_text = '\n'.join(textwrap.wrap(text, wrap_length))
            wrapped_labels.append(wrapped_text)
        else:
            wrapped_labels.append(text)
    if wrapped_labels:
        # Set ticks before setting ticklabels to avoid warning
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(wrapped_labels)
        # Prefer tick_params for rotation
        ax.tick_params(axis='x', rotation=x_rotation)
    y_labels = ax.get_yticklabels()
    for label in y_labels:
        text = label.get_text()
        try:
            if text and float(text) >= 1000:
                formatted = f"{float(text):,.0f}"
                label.set_text(formatted)
        except (ValueError, TypeError):
            pass

def apply_professional_styling(
    ax: plt.Axes, 
    title: str = "", 
    xlabel: str = "", 
    ylabel: str = "",
    theme: str = "professional"
) -> None:
    """
    Apply professional styling to the plot with Phase 1 enhancements.

    Args:
        ax (plt.Axes): The matplotlib axes object.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        theme (str): Styling theme ('professional', 'modern', 'minimal', 'elegant')
    """
    # Use the new modern styling function
    apply_modern_styling(ax, theme)
    
    # Set title and labels with enhanced typography
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    
    # Apply tight layout for better spacing
    try:
        ax.figure.tight_layout()
    except Exception:
        # Fallback if tight_layout fails
        pass


def safe_color_access(colors, index):
    """
    Safely access colors with cycling if index is out of range.
    
    Args:
        colors: List of color codes
        index: Index to access
    Returns:
        Color code string
    """
    if not colors:
        return '#2E86C1'  # Default fallback color
    return colors[index % len(colors)]


def create_category_palette(categories, palette_name='primary'):
    """
    Create a seaborn-compatible palette dictionary for specific categories.
    
    Args:
        categories: List of category names (e.g., ['Active', 'Inactive'])
        palette_name: Name of the color palette ('primary', 'secondary', 'categorical')
    Returns:
        dict: Dictionary mapping category names to colors
    """
    colors = get_professional_colors(palette_name)['colors']
    palette = {}
    
    for i, category in enumerate(categories):
        palette[category] = colors[i % len(colors)]
    
    return palette


def optimize_figure_size(ax: plt.Axes) -> None:
    """
    Adjust the figure size based on the number and length of tick labels.
    Expands width for many or long x-tick labels, and height for many y-tick labels.

    Args:
        ax (plt.Axes): The matplotlib axes object.
    """
    fig = ax.get_figure()
    xlabels = [label.get_text() for label in ax.get_xticklabels() if label.get_text()]
    ylabels = [label.get_text() for label in ax.get_yticklabels() if label.get_text()]
    
    # Heuristics for width and height
    base_width, base_height = 12, 7
    max_label_len = max([len(lbl) for lbl in xlabels], default=0)
    n_xticks = len(xlabels)
    n_yticks = len(ylabels)

    # Increase width for many or long x-tick labels
    width = base_width + 0.2 * max(0, n_xticks - 10) + 0.1 * max(0, max_label_len - 10)
    # Increase height for many y-tick labels
    height = base_height + 0.15 * max(0, n_yticks - 10)

    # Set new size if different
    if (abs(fig.get_figwidth() - width) > 0.1) or (abs(fig.get_figheight() - height) > 0.1):
        fig.set_size_inches(width, height, forward=True)


def _plot_grouped_bar_chart(
    ax: plt.Axes,
    data_df: 'pd.DataFrame',
    x_col: str,
    y_col: str,
    hue_col: str,
    legend_totals: bool,
    colors: list
) -> None:
    """
    Internal helper to plot grouped bar chart.
    """
    groups = data_df[hue_col].unique()
    x_positions = data_df[x_col].unique()
    bar_width = 0.8 / len(groups)
    x_indices = range(len(x_positions))
    legend_labels = []
    # Slice colors to match number of groups to avoid palette warning
    colors = colors[:len(groups)]
    for i, group in enumerate(groups):
        group_data = data_df[data_df[hue_col] == group]
        group_values = [group_data[group_data[x_col] == x][y_col].sum() if len(group_data[group_data[x_col] == x]) > 0 else 0 for x in x_positions]
        x_pos = [x + bar_width * i for x in x_indices]
        ax.bar(x_pos, group_values, bar_width, 
               color=colors[i % len(colors)], 
               label=group, alpha=0.8, edgecolor='white', linewidth=0.5)
        if legend_totals:
            total = sum(group_values)
            legend_labels.append(f"{group} (Total: {total})")
        else:
            legend_labels.append(group)
    ax.set_xticks([x + bar_width * (len(groups) - 1) / 2 for x in x_indices])
    ax.set_xticklabels(x_positions, rotation=45, ha='right')
    if legend_totals:
        ax.legend(legend_labels, loc='upper right', frameon=True, fancybox=True, shadow=True)
    else:
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)


def _plot_simple_bar_chart(
    ax: plt.Axes,
    data_df: 'pd.DataFrame',
    x_col: str,
    y_col: str,
    legend_totals: bool,
    colors: list
) -> None:
    """
    Internal helper to plot simple bar chart.
    """
    # Aggregate data properly
    if len(data_df) > len(data_df[x_col].unique()):
        values = data_df.groupby(x_col)[y_col].sum()
    else:
        values = data_df.set_index(x_col)[y_col]
    
    # Create numeric positions for bars
    x_positions = range(len(values))
    bars = ax.bar(x_positions, values.values, 
                 color=colors[0], alpha=0.8, 
                 edgecolor='white', linewidth=0.5)
    
    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(values.index, rotation=45, ha='right')
    
    if legend_totals:
        total = values.sum()
        ax.legend([f"Total: {total}"], loc='upper right', frameon=True, fancybox=True, shadow=True)


def handle_seaborn_warnings():
    """Suppress harmless seaborn warnings."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
    warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


def safe_binning(data: pd.Series, bins: Union[int, List], labels: List = None, 
                method: str = 'cut', **kwargs) -> pd.Series:
    """
    Safely create bins using pd.cut or pd.qcut with automatic validation.
    
    Args:
        data: Input data series
        bins: Number of bins (int) or bin edges (list)
        labels: Optional labels for bins
        method: 'cut' for equal-width bins, 'qcut' for equal-frequency bins
        **kwargs: Additional arguments for pd.cut/pd.qcut
    
    Returns:
        pd.Series: Binned data with proper labels
    """
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
            if isinstance(bins, int):
                return pd.qcut(data, q=bins, **kwargs)
            else:
                # qcut doesn't support custom bins, use cut instead
                return safe_binning(data, bins, labels, method='cut', **kwargs)
        else:
            raise ValueError(f"Unknown binning method: {method}")
    except Exception as e:
        # Fallback to automatic binning
        try:
            if method == 'cut':
                return pd.cut(data, bins=5, **kwargs)
            else:
                return pd.qcut(data, q=5, **kwargs)
        except Exception:
            # Last resort - return original data
            return data


def smart_categorical_plot(
    ax: plt.Axes,
    data_df: 'pd.DataFrame', 
    x_col: str,
    y_col: str,
    plot_type: str = "auto",
    **kwargs
) -> None:
    """
    Intelligently choose the best categorical plot type based on data characteristics.
    
    Args:
        ax: Matplotlib axes
        data_df: DataFrame with data
        x_col: Column for x-axis (categorical)
        y_col: Column for y-axis (numeric) 
        plot_type: 'auto', 'box', 'violin', 'strip', 'swarm', 'bar'
        **kwargs: Additional arguments for seaborn plots
    """
    import seaborn as sns
    
    # Suppress seaborn warnings
    handle_seaborn_warnings()
    
    # Count data points per category
    counts_per_category = data_df.groupby(x_col).size()
    max_points = counts_per_category.max()
    n_categories = len(counts_per_category)
    
    # Auto-select plot type based on data characteristics
    if plot_type == "auto":
        if max_points > 100:
            plot_type = "box"  # Too many points for swarm/strip
        elif max_points > 50:
            plot_type = "violin"  # Medium density
        elif max_points > 20:
            plot_type = "strip"  # Some density, use strip instead of swarm
        else:
            plot_type = "swarm"  # Low density, swarm is fine
    
    # Create the appropriate plot
    if plot_type == "box":
        sns.boxplot(data=data_df, x=x_col, y=y_col, ax=ax, **kwargs)
    elif plot_type == "violin":
        sns.violinplot(data=data_df, x=x_col, y=y_col, ax=ax, **kwargs)
    elif plot_type == "strip":
        sns.stripplot(data=data_df, x=x_col, y=y_col, ax=ax, size=3, alpha=0.7, **kwargs)
    elif plot_type == "swarm":
        sns.swarmplot(data=data_df, x=x_col, y=y_col, ax=ax, size=3, **kwargs)
    elif plot_type == "bar":
        # For bar plots, aggregate the data first
        agg_data = data_df.groupby(x_col)[y_col].mean().reset_index()
        ax.bar(agg_data[x_col], agg_data[y_col], **kwargs)
    
    return plot_type


def create_clean_bar_chart(
    ax: plt.Axes,
    data_df: 'pd.DataFrame',
    x_col: str,
    y_col: str,
    hue_col: str = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend_totals: bool = True,
    theme: str = "professional",
    add_annotations: bool = True
) -> None:
    """
    Create a clean bar chart with Phase 1 enhancements.
    
    Args:
        ax: Matplotlib axes object
        data_df: DataFrame with data
        x_col: Column for x-axis
        y_col: Column for y-axis
        hue_col: Optional grouping column
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        legend_totals: Whether to show totals in legend
        theme: Styling theme ('professional', 'modern', 'minimal', 'elegant')
        add_annotations: Whether to add smart annotations
    """
    import pandas as pd
    
    # Use contextual colors based on data type
    colors = get_professional_colors('hr_specific')['colors']
    
    # Slice colors to match number of categories/groups
    if hue_col and hue_col in data_df.columns:
        n_groups = data_df[hue_col].nunique()
        colors = colors[:n_groups]
        _plot_grouped_bar_chart(ax, data_df, x_col, y_col, hue_col, legend_totals, colors)
    else:
        n_cats = data_df[x_col].nunique()
        colors = colors[:n_cats]
        _plot_simple_bar_chart(ax, data_df, x_col, y_col, legend_totals, colors)
    
    # Apply enhanced styling
    apply_professional_styling(ax, title=title, xlabel=xlabel, ylabel=ylabel, theme=theme)
    
    # Add smart annotations if requested
    if add_annotations:
        add_smart_annotations(ax, data_df, highlight_insights=True)
    
    # Apply gradient background for depth
    create_gradient_background(ax)
    
    # Don't call format_axis_labels here since _plot_simple_bar_chart already handles it
    optimize_figure_size(ax)


def create_clean_line_chart(
    ax: plt.Axes,
    data_df: 'pd.DataFrame',
    x_col: str,
    y_col: str,
    hue_col: str = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    show_markers: bool = True,
    line_style: str = '-',
    theme: str = "professional"
) -> None:
    """
    Create a clean line chart with Phase 1 enhancements.
    """
    import pandas as pd
    colors = get_professional_colors('hr_specific')['colors']
    
    if hue_col and hue_col in data_df.columns:
        # Multi-line chart
        groups = data_df[hue_col].unique()
        for i, group in enumerate(groups):
            group_data = data_df[data_df[hue_col] == group].sort_values(x_col)
            marker = 'o' if show_markers else None
            ax.plot(group_data[x_col], group_data[y_col], 
                   color=colors[i % len(colors)], 
                   label=group, linewidth=2.5, 
                   marker=marker, markersize=6, 
                   linestyle=line_style, alpha=0.9)
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    else:
        # Single line chart
        data_sorted = data_df.sort_values(x_col)
        marker = 'o' if show_markers else None
        ax.plot(data_sorted[x_col], data_sorted[y_col], 
               color=colors[0], linewidth=3, 
               marker=marker, markersize=7, 
               linestyle=line_style, alpha=0.9)
    
    apply_professional_styling(ax, title=title, xlabel=xlabel, ylabel=ylabel, theme=theme)
    format_axis_labels(ax, x_rotation=45)
    create_gradient_background(ax)
    optimize_figure_size(ax)


def create_clean_scatter_plot(
    ax: plt.Axes,
    data_df: 'pd.DataFrame',
    x_col: str,
    y_col: str,
    hue_col: str = None,
    size_col: str = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    add_trendline: bool = False,
    theme: str = "professional"
) -> None:
    """
    Create a clean scatter plot with Phase 1 enhancements.
    """
    import pandas as pd
    import numpy as np
    colors = get_professional_colors('hr_specific')['colors']
    
    # Clean data first - remove rows with NaN in essential columns
    essential_cols = [x_col, y_col]
    if hue_col and hue_col in data_df.columns:
        essential_cols.append(hue_col)
    if size_col and size_col in data_df.columns:
        essential_cols.append(size_col)
    
    clean_data = data_df[essential_cols].dropna()
    
    if len(clean_data) == 0:
        ax.text(0.5, 0.5, 'No valid data to plot', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12, color='red')
        return
    
    # Handle size scaling - ensure reasonable point sizes
    if size_col and size_col in clean_data.columns:
        size_values = clean_data[size_col]
        # Normalize sizes to reasonable range (30-300)
        min_size, max_size = 30, 300
        if size_values.max() > size_values.min():
            normalized_sizes = min_size + (size_values - size_values.min()) / (size_values.max() - size_values.min()) * (max_size - min_size)
        else:
            normalized_sizes = pd.Series([100] * len(size_values), index=size_values.index)
    else:
        normalized_sizes = pd.Series([80] * len(clean_data), index=clean_data.index)
    
    if hue_col and hue_col in clean_data.columns:
        # Colored scatter by category
        hue_values = clean_data[hue_col]
        
        # Handle continuous vs categorical hue
        if pd.api.types.is_numeric_dtype(hue_values):
            # Continuous color mapping
            scatter = ax.scatter(clean_data[x_col], clean_data[y_col], 
                               c=hue_values, s=normalized_sizes, 
                               alpha=0.7, edgecolors='white', linewidth=0.5,
                               cmap='viridis')
            # Add colorbar for continuous values
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(hue_col, rotation=270, labelpad=15)
        else:
            # Categorical color mapping
            unique_groups = hue_values.unique()
            unique_groups = unique_groups[pd.notna(unique_groups)]  # Remove NaN groups
            
            for i, group in enumerate(unique_groups):
                group_mask = hue_values == group
                group_data = clean_data[group_mask]
                group_sizes = normalized_sizes[group_mask]
                
                ax.scatter(group_data[x_col], group_data[y_col], 
                          c=colors[i % len(colors)], label=str(group), 
                          s=group_sizes, alpha=0.7, edgecolors='white', linewidth=0.5)
            
            # Only add legend if we have reasonable number of groups
            if len(unique_groups) <= 10:
                ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    else:
        # Single color scatter
        ax.scatter(clean_data[x_col], clean_data[y_col], 
                  c=colors[0], s=normalized_sizes, alpha=0.7, 
                  edgecolors='white', linewidth=0.5)
    
    # Add trend line if requested and we have enough data
    if add_trendline and len(clean_data) > 1:
        try:
            # Calculate correlation to decide if trend line is meaningful
            correlation = clean_data[x_col].corr(clean_data[y_col])
            if abs(correlation) > 0.1:  # Only show trend if there's some correlation
                z = np.polyfit(clean_data[x_col], clean_data[y_col], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(clean_data[x_col].min(), clean_data[x_col].max(), 100)
                ax.plot(x_trend, p(x_trend), 
                       color='red', linestyle='--', linewidth=2, alpha=0.8, 
                       label=f'Trend (r={correlation:.2f})')
                ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        except Exception as e:
            # Skip trend line if calculation fails
            pass
    
    apply_professional_styling(ax, title=title, xlabel=xlabel, ylabel=ylabel, theme=theme)
    format_axis_labels(ax, x_rotation=0)  # Don't rotate for scatter plots
    create_gradient_background(ax)
    optimize_figure_size(ax)


def smart_annotate_points(ax, data_df, x_col, y_col, label_col, max_annotations=5):
    """
    Add smart annotations to scatter plot points, avoiding overlap.
    
    Args:
        ax: matplotlib axes
        data_df: DataFrame with data
        x_col: x-axis column name
        y_col: y-axis column name  
        label_col: column to use for labels
        max_annotations: maximum number of annotations to add
    """
    import pandas as pd
    import numpy as np
    
    # Get extreme points (highest/lowest on both axes)
    extreme_points = []
    
    # Add highest and lowest y values
    max_y_idx = data_df[y_col].idxmax()
    min_y_idx = data_df[y_col].idxmin()
    extreme_points.extend([max_y_idx, min_y_idx])
    
    # Add highest and lowest x values if different
    max_x_idx = data_df[x_col].idxmax()
    min_x_idx = data_df[x_col].idxmin()
    if max_x_idx not in extreme_points:
        extreme_points.append(max_x_idx)
    if min_x_idx not in extreme_points:
        extreme_points.append(min_x_idx)
    
    # Limit to max_annotations
    extreme_points = extreme_points[:max_annotations]
    
    # Add annotations with smart positioning
    for i, idx in enumerate(extreme_points):
        if idx in data_df.index:
            row = data_df.loc[idx]
            x_pos = row[x_col]
            y_pos = row[y_col]
            label = str(row[label_col])
            
            # Smart positioning to avoid overlap
            if i % 2 == 0:
                xytext = (10, 10)  # Top-right
                ha = 'left'
            else:
                xytext = (-10, -10)  # Bottom-left  
                ha = 'right'
            
            ax.annotate(
                label,
                (x_pos, y_pos),
                xytext=xytext,
                textcoords='offset points',
                ha=ha,
                va='center',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', alpha=0.6)
            )


def create_clean_histogram(
    ax: plt.Axes,
    data_df: 'pd.DataFrame',
    col: str,
    bins: int = 30,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Frequency",
    show_stats: bool = True,
    theme: str = "professional"
) -> None:
    """
    Create a clean histogram with Phase 1 enhancements.
    """
    import pandas as pd
    import numpy as np
    colors = get_professional_colors('hr_specific')['colors']
    
    # Create histogram
    n, bins_used, patches = ax.hist(data_df[col].dropna(), bins=bins, 
                                   color=colors[0], alpha=0.7, 
                                   edgecolor='white', linewidth=0.5)
    
    # Add statistics text if requested
    if show_stats:
        mean_val = data_df[col].mean()
        std_val = data_df[col].std()
        stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nCount: {len(data_df[col].dropna())}'
        ax.text(0.75, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='white', alpha=0.8))
    
    apply_professional_styling(ax, title=title, xlabel=xlabel, ylabel=ylabel, theme=theme)
    format_axis_labels(ax, x_rotation=45)
    create_gradient_background(ax)
    optimize_figure_size(ax)


def create_clean_box_plot(
    ax: plt.Axes,
    data_df: 'pd.DataFrame',
    x_col: str,
    y_col: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    show_outliers: bool = True,
    theme: str = "professional"
) -> None:
    """
    Create a clean box plot with Phase 1 enhancements.
    """
    import seaborn as sns
    colors = get_professional_colors('hr_specific')['colors']
    
    # Suppress seaborn warnings
    handle_seaborn_warnings()
    
    # Create box plot
    n_categories = data_df[x_col].nunique()
    palette = colors[:n_categories] if n_categories <= len(colors) else colors
    sns.boxplot(data=data_df, x=x_col, y=y_col, ax=ax, 
               palette=palette, showfliers=show_outliers)
    
    apply_professional_styling(ax, title=title, xlabel=xlabel, ylabel=ylabel, theme=theme)
    format_axis_labels(ax, x_rotation=45)
    create_gradient_background(ax)
    optimize_figure_size(ax)


def create_clean_heatmap(
    ax: plt.Axes,
    data_df: 'pd.DataFrame',
    title: str = "",
    cmap: str = 'RdYlBu_r',
    show_values: bool = False,
    fmt: str = '.2f',
    theme: str = "professional"
) -> None:
    """
    Create a clean correlation heatmap with Phase 1 enhancements.
    """
    import seaborn as sns
    import numpy as np
    
    # Calculate correlation matrix if not already provided
    if data_df.shape[1] > 1:
        corr_matrix = data_df.select_dtypes(include=[np.number]).corr()
    else:
        corr_matrix = data_df
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=show_values, cmap=cmap, center=0,
               square=True, ax=ax, fmt=fmt, cbar_kws={'shrink': 0.8})
    
    apply_professional_styling(ax, title=title, xlabel="", ylabel="", theme=theme)
    # Don't rotate labels for heatmaps
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    create_gradient_background(ax)
    optimize_figure_size(ax)


def create_clean_pie_chart(
    ax: plt.Axes,
    data_df: 'pd.DataFrame',
    col: str,
    title: str = "",
    show_percentages: bool = True,
    explode_max: bool = True,
    theme: str = "professional"
) -> None:
    """
    Create a clean pie chart with Phase 1 enhancements.
    """
    import pandas as pd
    colors = get_professional_colors('hr_specific')['colors']
    
    # Aggregate data
    value_counts = data_df[col].value_counts()
    
    # Explode the largest slice slightly
    explode = None
    if explode_max and len(value_counts) > 1:
        explode = [0.05 if i == 0 else 0 for i in range(len(value_counts))]
    
    # Create pie chart
    autopct = '%1.1f%%' if show_percentages else None
    wedges, texts, autotexts = ax.pie(value_counts.values, labels=value_counts.index,
                                     autopct=autopct, colors=colors[:len(value_counts)],
                                     explode=explode, shadow=True, startangle=90)
    
    # Style the text
    for autotext in autotexts or []:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('equal')  # Equal aspect ratio ensures pie is circular

# =========================================================================
#  PHASE 2: Advanced Chart Types & Interactive Elements
# =========================================================================

def create_clean_violin_plot(
    ax: plt.Axes,
    data_df: 'pd.DataFrame',
    x_col: str,
    y_col: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    theme: str = "professional",
    show_points: bool = False
) -> None:
    """
    Create a clean violin plot with Phase 2 enhancements.
    
    Args:
        ax: Matplotlib axes object
        data_df: DataFrame with data
        x_col: Column for x-axis (categorical)
        y_col: Column for y-axis (numeric)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        theme: Styling theme
        show_points: Whether to overlay individual points
    """
    import seaborn as sns
    colors = get_professional_colors('hr_specific')['colors']
    
    # Suppress seaborn warnings
    handle_seaborn_warnings()
    
    # Create violin plot
    n_categories = data_df[x_col].nunique()
    palette = colors[:n_categories] if n_categories <= len(colors) else colors
    
    if show_points:
        # Violin plot with individual points
        sns.violinplot(data=data_df, x=x_col, y=y_col, ax=ax, 
                      palette=palette, inner='box')
        sns.stripplot(data=data_df, x=x_col, y=y_col, ax=ax,
                     color='black', size=3, alpha=0.3)
    else:
        # Standard violin plot
        sns.violinplot(data=data_df, x=x_col, y=y_col, ax=ax, 
                      palette=palette, inner='box')
    
    apply_professional_styling(ax, title=title, xlabel=xlabel, ylabel=ylabel, theme=theme)
    format_axis_labels(ax, x_rotation=45)
    create_gradient_background(ax)
    optimize_figure_size(ax)

def create_clean_swarm_plot(
    ax: plt.Axes,
    data_df: 'pd.DataFrame',
    x_col: str,
    y_col: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    theme: str = "professional",
    hue_col: str = None
) -> None:
    """
    Create a clean swarm plot with Phase 2 enhancements.
    
    Args:
        ax: Matplotlib axes object
        data_df: DataFrame with data
        x_col: Column for x-axis (categorical)
        y_col: Column for y-axis (numeric)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        theme: Styling theme
        hue_col: Optional grouping column
    """
    import seaborn as sns
    colors = get_professional_colors('hr_specific')['colors']
    
    # Suppress seaborn warnings
    handle_seaborn_warnings()
    
    # Create swarm plot
    if hue_col and hue_col in data_df.columns:
        sns.swarmplot(data=data_df, x=x_col, y=y_col, hue=hue_col, ax=ax,
                     palette=colors, size=4, alpha=0.7)
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    else:
        sns.swarmplot(data=data_df, x=x_col, y=y_col, ax=ax,
                     color=colors[0], size=4, alpha=0.7)
    
    apply_professional_styling(ax, title=title, xlabel=xlabel, ylabel=ylabel, theme=theme)
    format_axis_labels(ax, x_rotation=45)
    create_gradient_background(ax)
    optimize_figure_size(ax)

def create_clean_waterfall_chart(
    ax: plt.Axes,
    data_df: 'pd.DataFrame',
    x_col: str,
    y_col: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    theme: str = "professional",
    show_connectors: bool = True
) -> None:
    """
    Create a clean waterfall chart with Phase 2 enhancements.
    
    Args:
        ax: Matplotlib axes object
        data_df: DataFrame with data
        x_col: Column for x-axis (categories)
        y_col: Column for y-axis (values)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        theme: Styling theme
        show_connectors: Whether to show connecting lines
    """
    import numpy as np
    
    # Prepare data
    categories = data_df[x_col].values
    values = data_df[y_col].values
    
    # Calculate cumulative values for waterfall
    cumulative = np.cumsum(np.insert(values, 0, 0))
    
    # Create bars
    bar_width = 0.8
    x_positions = np.arange(len(categories))
    
    # Color bars based on positive/negative values
    colors = get_professional_colors('hr_specific')['colors']
    bar_colors = []
    for val in values:
        if val >= 0:
            bar_colors.append(colors[1])  # Green for positive
        else:
            bar_colors.append(colors[3])  # Red for negative
    
    # Create bars
    bars = ax.bar(x_positions, values, bar_width, 
                  bottom=cumulative[:-1], color=bar_colors, alpha=0.8,
                  edgecolor='white', linewidth=0.5)
    
    # Add connectors if requested
    if show_connectors:
        for i in range(len(x_positions) - 1):
            ax.plot([x_positions[i] + bar_width/2, x_positions[i+1] - bar_width/2],
                   [cumulative[i+1], cumulative[i+1]], 
                   color='gray', alpha=0.5, linewidth=1)
    
    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        y_pos = bar.get_y() + height/2 if height >= 0 else bar.get_y() + height/2
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
               f'{value:.1f}', ha='center', va='center',
               fontsize=8, fontweight='bold')
    
    apply_professional_styling(ax, title=title, xlabel=xlabel, ylabel=ylabel, theme=theme)
    create_gradient_background(ax)
    optimize_figure_size(ax)

def create_clean_ridge_plot(
    ax: plt.Axes,
    data_df: 'pd.DataFrame',
    x_col: str,
    y_col: str,
    group_col: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    theme: str = "professional"
) -> None:
    """
    Create a clean ridge plot with Phase 2 enhancements.
    
    Args:
        ax: Matplotlib axes object
        data_df: DataFrame with data
        x_col: Column for x-axis (numeric)
        y_col: Column for y-axis (density)
        group_col: Column for grouping
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        theme: Styling theme
    """
    import seaborn as sns
    from scipy import stats
    
    # Suppress seaborn warnings
    handle_seaborn_warnings()
    
    # Create ridge plot using seaborn
    sns.kdeplot(data=data_df, x=x_col, hue=group_col, ax=ax,
                palette=get_professional_colors('hr_specific')['colors'])
    
    apply_professional_styling(ax, title=title, xlabel=xlabel, ylabel=ylabel, theme=theme)
    create_gradient_background(ax)
    optimize_figure_size(ax)

def create_clean_sankey_diagram(
    ax: plt.Axes,
    source: list,
    target: list,
    value: list,
    title: str = "",
    theme: str = "professional"
) -> None:
    """
    Create a clean Sankey diagram with Phase 2 enhancements.
    
    Args:
        ax: Matplotlib axes object
        source: List of source nodes
        target: List of target nodes
        value: List of flow values
        title: Plot title
        theme: Styling theme
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        
        # Create Sankey diagram using plotly
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=list(set(source + target)),
                color=get_professional_colors('hr_specific')['colors'][:len(set(source + target))]
            ),
            link=dict(
                source=[list(set(source + target)).index(s) for s in source],
                target=[list(set(source + target)).index(t) for t in target],
                value=value
            )
        )])
        
        fig.update_layout(title_text=title, font_size=10)
        
        # Convert plotly figure to matplotlib
        # Note: This is a simplified conversion - in practice, you might want to use plotly directly
        ax.text(0.5, 0.5, 'Sankey Diagram\n(Use plotly for full functionality)', 
               transform=ax.transAxes, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
    except ImportError:
        # Fallback if plotly is not available
        ax.text(0.5, 0.5, 'Sankey Diagram\n(Install plotly for full functionality)', 
               transform=ax.transAxes, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    apply_professional_styling(ax, title=title, theme=theme)
    create_gradient_background(ax)
    optimize_figure_size(ax)

def add_interactive_elements(ax: plt.Axes, fig: plt.Figure, data_df: 'pd.DataFrame') -> None:
    """
    Add interactive elements to plots (for HTML export).
    
    Args:
        ax: Matplotlib axes object
        fig: Matplotlib figure object
        data_df: DataFrame with data
    """
    # Add hover tooltips (for HTML export)
    # This is a placeholder for future interactive features
    pass

def detect_insights(data_df: 'pd.DataFrame', x_col: str = None, y_col: str = None) -> dict:
    """
    Automatically detect insights from data.
    
    Args:
        data_df: DataFrame with data
        x_col: Optional x-axis column
        y_col: Optional y-axis column
    Returns:
        dict: Dictionary with detected insights
    """
    insights = {
        'outliers': [],
        'trends': [],
        'patterns': [],
        'correlations': [],
        'summary_stats': {}
    }
    
    if y_col and y_col in data_df.columns:
        # Detect outliers
        Q1 = data_df[y_col].quantile(0.25)
        Q3 = data_df[y_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = data_df[(data_df[y_col] < Q1 - 1.5 * IQR) | 
                          (data_df[y_col] > Q3 + 1.5 * IQR)]
        
        if len(outliers) > 0:
            insights['outliers'] = outliers[y_col].tolist()
        
        # Detect trends
        if x_col and x_col in data_df.columns:
            # Check if both columns are numeric
            if pd.api.types.is_numeric_dtype(data_df[x_col]) and pd.api.types.is_numeric_dtype(data_df[y_col]):
                try:
                    correlation = data_df[x_col].corr(data_df[y_col])
                    if not pd.isna(correlation):
                        if abs(correlation) > 0.7:
                            trend = "strong positive" if correlation > 0 else "strong negative"
                            insights['trends'].append(f"{trend} correlation ({correlation:.2f})")
                        elif abs(correlation) > 0.3:
                            trend = "moderate positive" if correlation > 0 else "moderate negative"
                            insights['trends'].append(f"{trend} correlation ({correlation:.2f})")
                except Exception:
                    # Skip correlation if calculation fails
                    pass
        
        # Summary statistics
        insights['summary_stats'] = {
            'mean': data_df[y_col].mean(),
            'median': data_df[y_col].median(),
            'std': data_df[y_col].std(),
            'min': data_df[y_col].min(),
            'max': data_df[y_col].max()
        }
    
    return insights

def add_insight_annotations(ax: plt.Axes, insights: dict) -> None:
    """
    Add insight annotations to plots.
    
    Args:
        ax: Matplotlib axes object
        insights: Dictionary with insights from detect_insights()
    """
    if not insights:
        return
    
    # Add summary statistics
    if 'summary_stats' in insights and insights['summary_stats']:
        stats = insights['summary_stats']
        stats_text = f"Mean: {stats['mean']:.2f}\nStd: {stats['std']:.2f}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='white', alpha=0.8), fontsize=8)
    
    # Add trend annotations
    if insights.get('trends'):
        for i, trend in enumerate(insights['trends']):
            ax.text(0.02, 0.85 - i*0.05, f"📈 {trend}", transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='lightgreen', alpha=0.8), fontsize=8)

def create_enhanced_chart_with_insights(
    ax: plt.Axes,
    data_df: 'pd.DataFrame',
    chart_type: str,
    x_col: str,
    y_col: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    theme: str = "professional",
    add_insights: bool = True,
    **kwargs
) -> None:
    """
    Create an enhanced chart with automatic insight detection.
    
    Args:
        ax: Matplotlib axes object
        data_df: DataFrame with data
        chart_type: Type of chart to create
        x_col: Column for x-axis
        y_col: Column for y-axis
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        theme: Styling theme
        add_insights: Whether to add insight annotations
        **kwargs: Additional arguments for specific chart types
    """
    # Create the base chart
    if chart_type == 'bar':
        create_clean_bar_chart(ax, data_df, x_col, y_col, title, xlabel, ylabel, theme=theme, **kwargs)
    elif chart_type == 'line':
        create_clean_line_chart(ax, data_df, x_col, y_col, title, xlabel, ylabel, theme=theme, **kwargs)
    elif chart_type == 'scatter':
        create_clean_scatter_plot(ax, data_df, x_col, y_col, title, xlabel, ylabel, theme=theme, **kwargs)
    elif chart_type == 'histogram':
        create_clean_histogram(ax, data_df, y_col, title=title, xlabel=xlabel, ylabel=ylabel, theme=theme, **kwargs)
    elif chart_type == 'box':
        create_clean_box_plot(ax, data_df, x_col, y_col, title, xlabel, ylabel, theme=theme, **kwargs)
    elif chart_type == 'violin':
        create_clean_violin_plot(ax, data_df, x_col, y_col, title, xlabel, ylabel, theme=theme, **kwargs)
    elif chart_type == 'swarm':
        create_clean_swarm_plot(ax, data_df, x_col, y_col, title, xlabel, ylabel, theme=theme, **kwargs)
    elif chart_type == 'waterfall':
        create_clean_waterfall_chart(ax, data_df, x_col, y_col, title, xlabel, ylabel, theme=theme, **kwargs)
    elif chart_type == 'radar':
        # For radar, expect categories, values, group_labels in kwargs
        create_clean_radar_chart(ax, kwargs['categories'], kwargs['values'], kwargs.get('group_labels'), title, theme=theme)
    elif chart_type == 'treemap':
        # For treemap, expect labels and sizes in kwargs
        create_clean_treemap(ax, kwargs['labels'], kwargs['sizes'], kwargs.get('colors'), title, theme=theme)
    elif chart_type == 'gantt':
        # For gantt, expect task_col, start_col, end_col in kwargs
        create_clean_gantt_chart(ax, data_df, kwargs['task_col'], kwargs['start_col'], kwargs['end_col'], kwargs.get('color_col'), title, theme=theme)
    else:
        # Default to bar chart
        create_clean_bar_chart(ax, data_df, x_col, y_col, title, xlabel, ylabel, theme=theme, **kwargs)
    
    # Add insights if requested
    if add_insights:
        insights = detect_insights(data_df, x_col, y_col)
        add_insight_annotations(ax, insights)

# =========================================================================
#  Plot Memory System for Phase 2
# =========================================================================

class PlotMemory:
    """Memory system for tracking plots and enabling modifications."""
    
    def __init__(self):
        self.plots = []  # List of plot objects
        self.plot_metadata = []  # List of metadata dicts
        self.plot_data = []  # List of data used for plots
        self.plot_context = []  # List of context strings
    
    def add_plot(self, fig, data_df, context, chart_type, styling):
        """Store plot with complete metadata."""
        plot_info = {
            'figure': fig,
            'data': data_df,
            'context': context,
            'chart_type': chart_type,
            'styling': styling,
            'timestamp': datetime.now(),
            'plot_id': len(self.plots)
        }
        self.plots.append(plot_info)
        return len(self.plots) - 1
    
    def get_plot_by_reference(self, reference):
        """Get plot by natural language reference."""
        # Handle references like "the above plot", "previous plot", "last chart"
        if reference.lower() in ["above plot", "previous plot", "last plot", "the plot"]:
            return self.plots[-1] if self.plots else None
        return None
    
    def get_plot_by_id(self, plot_id):
        """Get plot by ID."""
        if 0 <= plot_id < len(self.plots):
            return self.plots[plot_id]
        return None
    
    def list_plots(self):
        """List all stored plots with metadata."""
        return [(i, plot['chart_type'], plot['context']) for i, plot in enumerate(self.plots)]

def is_plot_modification_request(query: str) -> bool:
    """
    Detect if a query is requesting plot modification.
    
    Args:
        query: User query string
    Returns:
        bool: True if modification request detected
    """
    modification_keywords = [
        "change", "modify", "edit", "update", "adjust", "alter",
        "color", "colors", "style", "title", "labels", "size",
        "add", "remove", "include", "exclude", "make", "switch"
    ]
    
    return any(keyword in query.lower() for keyword in modification_keywords)

def generate_plot_modification_code(query: str, target_plot: dict, df: pd.DataFrame) -> str:
    """
    Generate code to modify existing plot.
    
    Args:
        query: User modification request
        target_plot: Plot metadata from PlotMemory
        df: Current DataFrame
    Returns:
        str: Python code for plot modification
    """
    prompt = f"""
You are modifying an existing plot. Here are the details:

ORIGINAL PLOT:
- Chart Type: {target_plot['chart_type']}
- Data Used: {target_plot['data'].columns.tolist()}
- Context: {target_plot['context']}
- Current Styling: {target_plot['styling']}

USER REQUEST: "{query}"

REQUIREMENTS:
- Modify the existing plot based on the user's request
- Keep the same chart type unless explicitly requested to change
- Use the same data unless new data is requested
- Apply the requested modifications (colors, titles, labels, etc.)
- Return the modified plot in the same format: (fig, data_df)

AVAILABLE MODIFICATIONS:
- Change colors: Use different color palettes
- Update titles: Modify plot title, axis labels
- Adjust styling: Change grid, background, fonts
- Add elements: Trend lines, annotations, legends
- Remove elements: Simplify the plot
- Change size: Modify figure dimensions
- Add data: Include additional data series

EXAMPLE:
```python
# Get the original plot data
original_data = {target_plot['data'].to_dict()}

# Create modified plot
fig, ax = plt.subplots(figsize=(10, 6))
# ... modification code based on user request ...
result = (fig, modified_data)
```
"""
    
    return prompt

# =========================================================================
#  PHASE 3: Enhanced Plot Functions with Quality System
# =========================================================================

def create_enhanced_plot_with_quality(
    data_df: pd.DataFrame,
    x_col: str,
    y_col: str = None,
    hue_col: str = None,
    chart_type: str = 'auto',
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    theme: str = "professional",
    add_quality_assessment: bool = True
) -> tuple:
    """
    Create an enhanced plot with Phase 3 quality assessment.
    
    Args:
        data_df: DataFrame with data
        x_col: Column for x-axis
        y_col: Optional column for y-axis
        hue_col: Optional grouping column
        chart_type: Type of chart ('auto', 'bar', 'line', 'scatter', 'histogram', 'box', 'violin')
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        theme: Styling theme
        add_quality_assessment: Whether to add quality assessment
        
    Returns:
        tuple: (fig, ax, quality_report) where quality_report is None if Phase 3 not available
    """
    if not PHASE3_AVAILABLE:
        # Fallback to basic plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if chart_type == 'auto':
            if pd.api.types.is_numeric_dtype(data_df[x_col]) and y_col and pd.api.types.is_numeric_dtype(data_df[y_col]):
                chart_type = 'scatter'
            elif pd.api.types.is_numeric_dtype(data_df[x_col]):
                chart_type = 'histogram'
            else:
                chart_type = 'bar'
        
        if chart_type == 'bar':
            create_clean_bar_chart(ax, data_df, x_col, y_col or 'count', hue_col, title, xlabel, ylabel, theme=theme)
        elif chart_type == 'line':
            create_clean_line_chart(ax, data_df, x_col, y_col, hue_col, title, xlabel, ylabel, theme=theme)
        elif chart_type == 'scatter':
            create_clean_scatter_plot(ax, data_df, x_col, y_col, hue_col, title, xlabel, ylabel, theme=theme)
        elif chart_type == 'histogram':
            create_clean_histogram(ax, data_df, x_col, title=title, xlabel=xlabel, ylabel=ylabel, theme=theme)
        elif chart_type == 'box':
            create_clean_box_plot(ax, data_df, x_col, y_col, title, xlabel, ylabel, theme=theme)
        elif chart_type == 'violin':
            create_clean_violin_plot(ax, data_df, x_col, y_col, title, xlabel, ylabel, theme=theme)
        
        return fig, ax, None
    
    # Use Phase 3 quality system
    quality_system = PlotQualitySystem()
    
    # Generate plot code
    if chart_type == 'auto':
        # Get recommendation from quality system
        recommendation = quality_system.chart_recommender.recommend_chart_type(data_df, x_col, y_col, hue_col)
        chart_type = recommendation['primary_chart']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if chart_type == 'bar_plot':
        create_clean_bar_chart(ax, data_df, x_col, y_col or 'count', hue_col, title, xlabel, ylabel, theme=theme)
    elif chart_type == 'line_plot':
        create_clean_line_chart(ax, data_df, x_col, y_col, hue_col, title, xlabel, ylabel, theme=theme)
    elif chart_type == 'scatter_plot':
        create_clean_scatter_plot(ax, data_df, x_col, y_col, hue_col, title, xlabel, ylabel, theme=theme)
    elif chart_type == 'histogram':
        create_clean_histogram(ax, data_df, x_col, title=title, xlabel=xlabel, ylabel=ylabel, theme=theme)
    elif chart_type == 'box_plot':
        create_clean_box_plot(ax, data_df, x_col, y_col, title, xlabel, ylabel, theme=theme)
    elif chart_type == 'violin_plot':
        create_clean_violin_plot(ax, data_df, x_col, y_col, title, xlabel, ylabel, theme=theme)
    else:
        # Default to bar chart
        create_clean_bar_chart(ax, data_df, x_col, y_col or 'count', hue_col, title, xlabel, ylabel, theme=theme)
    
    # Add smart legend if grouping is used
    if hue_col and hue_col in data_df.columns:
        quality_system.legend_system.create_smart_legend(ax, data_df, hue_col, chart_type)
    
    # Assess quality if requested
    quality_report = None
    if add_quality_assessment:
        quality_metrics = quality_system.assess_final_plot(fig, data_df)
        quality_report = {
            'metrics': quality_metrics,
            'recommendation': quality_system.chart_recommender.recommend_chart_type(data_df, x_col, y_col, hue_col),
            'data_validation': quality_system.data_validator.validate_plot_data(data_df, x_col, y_col, hue_col)
        }
    
    return fig, ax, quality_report


def validate_and_improve_plot_code(
    code: str,
    data_df: pd.DataFrame,
    x_col: str,
    y_col: str = None,
    hue_col: str = None
) -> dict:
    """
    Validate and improve plot code using Phase 3 quality system.
    
    Args:
        code: Generated plot code
        data_df: DataFrame with data
        x_col: Column for x-axis
        y_col: Optional column for y-axis
        hue_col: Optional grouping column
        
    Returns:
        dict: Validation and improvement results
    """
    if not PHASE3_AVAILABLE:
        return {
            'is_valid': True,
            'improved_code': code,
            'issues': [],
            'suggestions': [],
            'quality_score': 1.0
        }
    
    quality_system = PlotQualitySystem()
    result = quality_system.process_plot_request(code, data_df, x_col, y_col, hue_col)
    
    return {
        'is_valid': result['validation']['is_valid'],
        'improved_code': result['improved_code'],
        'issues': result['issues'],
        'suggestions': result['suggestions'],
        'quality_score': result['data_validation']['quality_score'] if result['data_validation'] else 1.0,
        'chart_recommendation': result['chart_recommendation'],
        'quality_report': quality_system.get_quality_report(result)
    }


def get_plot_quality_metrics(fig: plt.Figure, data_df: pd.DataFrame) -> dict:
    """
    Get quality metrics for a plot using Phase 3 assessment.
    
    Args:
        fig: Matplotlib figure
        data_df: DataFrame used in the plot
        
    Returns:
        dict: Quality metrics
    """
    if not PHASE3_AVAILABLE:
        return {
            'overall_score': 0.8,
            'quality_level': 'Good',
            'readability': 0.7,
            'information_density': 0.6,
            'aesthetic_appeal': 0.8,
            'data_accuracy': 1.0
        }
    
    quality_system = PlotQualitySystem()
    return quality_system.assess_final_plot(fig, data_df)


def recommend_chart_type(
    data_df: pd.DataFrame,
    x_col: str,
    y_col: str = None,
    hue_col: str = None
) -> dict:
    """
    Recommend the best chart type based on data characteristics.
    
    Args:
        data_df: DataFrame with data
        x_col: Column for x-axis
        y_col: Optional column for y-axis
        hue_col: Optional grouping column
        
    Returns:
        dict: Chart recommendation
    """
    if not PHASE3_AVAILABLE:
        # Basic recommendation logic
        x_numeric = pd.api.types.is_numeric_dtype(data_df[x_col])
        y_numeric = y_col and pd.api.types.is_numeric_dtype(data_df[y_col])
        
        if x_numeric and y_numeric:
            return {'primary_chart': 'scatter_plot', 'confidence_score': 0.8}
        elif x_numeric:
            return {'primary_chart': 'histogram', 'confidence_score': 0.9}
        else:
            return {'primary_chart': 'bar_plot', 'confidence_score': 0.8}
    
    quality_system = PlotQualitySystem()
    return quality_system.chart_recommender.recommend_chart_type(data_df, x_col, y_col, hue_col)


def validate_plot_data(
    data_df: pd.DataFrame,
    x_col: str,
    y_col: str = None,
    hue_col: str = None
) -> dict:
    """
    Validate data for plotting and suggest improvements.
    
    Args:
        data_df: DataFrame with data
        x_col: Column for x-axis
        y_col: Optional column for y-axis
        hue_col: Optional grouping column
        
    Returns:
        dict: Validation results
    """
    if not PHASE3_AVAILABLE:
        return {
            'is_valid': True,
            'warnings': [],
            'suggestions': [],
            'quality_score': 1.0,
            'cleaned_data': data_df.copy()
        }
    
    quality_system = PlotQualitySystem()
    return quality_system.data_validator.validate_plot_data(data_df, x_col, y_col, hue_col)


def create_smart_legend_enhanced(
    ax: plt.Axes,
    data_df: pd.DataFrame,
    hue_col: str = None,
    chart_type: str = 'auto'
) -> None:
    """
    Create enhanced smart legend with Phase 3 improvements.
    
    Args:
        ax: Matplotlib axes object
        data_df: DataFrame with data
        hue_col: Optional grouping column
        chart_type: Type of chart
    """
    if not PHASE3_AVAILABLE:
        # Basic legend creation
        if hue_col and hue_col in data_df.columns:
            ax.legend(title=hue_col)
        return
    
    quality_system = PlotQualitySystem()
    quality_system.legend_system.create_smart_legend(ax, data_df, hue_col, chart_type)


def get_hr_chart_template(chart_type: str, data_type: str) -> dict:
    """
    Get HR-specific chart template with best practices.
    
    Args:
        chart_type: Type of chart
        data_type: Type of data analysis
        
    Returns:
        dict: Chart template
    """
    if not PHASE3_AVAILABLE:
        return {
            'chart_type': 'bar_plot',
            'color_palette': 'primary',
            'annotations': True,
            'insights': False,
            'title_template': 'Analysis by {dimension}',
            'best_practices': ['Use clear labels', 'Include value labels']
        }
    
    quality_system = PlotQualitySystem()
    return quality_system.chart_recommender.get_hr_chart_template(chart_type, data_type)


# =========================================================================
#  Phase 3 Availability Check
# =========================================================================

def is_phase3_available() -> bool:
    """
    Check if Phase 3 quality system is available.
    
    Returns:
        bool: True if Phase 3 is available
    """
    return PHASE3_AVAILABLE


def get_phase3_features() -> dict:
    """
    Get information about available Phase 3 features.
    
    Returns:
        dict: Feature information
    """
    if not PHASE3_AVAILABLE:
        return {
            'available': False,
            'features': [],
            'message': 'Phase 3 quality system not available'
        }
    
    return {
        'available': True,
        'features': [
            'Code validation and automatic fixing',
            'Smart legend system with statistics',
            'Data validation and quality assessment',
            'Intelligent chart selection',
            'Quality metrics and scoring',
            'HR-specific chart templates'
        ],
        'message': 'Phase 3 quality system is available and active'
    }

def create_clean_radar_chart(
    ax: plt.Axes,
    data_df: 'pd.DataFrame',
    categories: list,
    values: list,
    group_labels: list = None,
    title: str = "",
    theme: str = "professional"
) -> None:
    """
    Create a clean radar (spider) chart with Phase 4 enhancements.
    Args:
        ax: Matplotlib axes object (polar=True)
        data_df: DataFrame with data (wide format: rows=groups, cols=categories)
        categories: List of variable names (axes)
        values: List of lists, each sublist is values for a group
        group_labels: List of group names
        title: Plot title
        theme: Styling theme
    """
    import numpy as np
    # Ensure categories are closed (first = last)
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    # Plot each group
    for idx, group_vals in enumerate(values):
        vals = group_vals + group_vals[:1]
        label = group_labels[idx] if group_labels else None
        ax.plot(angles, vals, linewidth=2, linestyle='solid', label=label)
        ax.fill(angles, vals, alpha=0.25)
    ax.set_title(title, size=14, pad=20)
    if group_labels:
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    ax.grid(True)
    # Apply styling
    apply_professional_styling(ax, title=title, theme=theme)
    create_gradient_background(ax)
    optimize_figure_size(ax)


def create_clean_treemap(
    ax: plt.Axes,
    labels: list,
    sizes: list,
    colors: list = None,
    title: str = "",
    theme: str = "professional"
) -> None:
    """
    Create a clean treemap chart with Phase 4 enhancements.
    Args:
        ax: Matplotlib axes object
        labels: List of category labels
        sizes: List of values (areas)
        colors: List of colors (optional)
        title: Plot title
        theme: Styling theme
    """
    import squarify
    if colors is None:
        colors = get_professional_colors('hr_specific')['colors'] * (len(sizes) // 4 + 1)
    squarify.plot(sizes=sizes, label=labels, color=colors[:len(sizes)], alpha=0.8, ax=ax)
    ax.set_title(title, size=14, pad=20)
    ax.axis('off')
    # Apply styling
    create_gradient_background(ax)
    optimize_figure_size(ax)


def create_clean_gantt_chart(
    ax: plt.Axes,
    data_df: 'pd.DataFrame',
    task_col: str,
    start_col: str,
    end_col: str,
    color_col: str = None,
    title: str = "",
    theme: str = "professional"
) -> None:
    """
    Create a clean Gantt-style chart with Phase 4 enhancements.
    Args:
        ax: Matplotlib axes object
        data_df: DataFrame with columns for tasks, start, end
        task_col: Column name for task/category
        start_col: Column name for start date/time
        end_col: Column name for end date/time
        color_col: Optional column for color/grouping
        title: Plot title
        theme: Styling theme
    """
    import matplotlib.dates as mdates
    import pandas as pd
    df = data_df.copy()
    df = df.sort_values(by=start_col)
    y_pos = range(len(df))
    starts = pd.to_datetime(df[start_col])
    ends = pd.to_datetime(df[end_col])
    durations = (ends - starts).dt.total_seconds() / 3600.0  # hours
    colors = get_professional_colors('hr_specific')['colors']
    for i, (task, start, duration) in enumerate(zip(df[task_col], starts, durations)):
        color = colors[i % len(colors)]
        ax.barh(i, duration, left=start, color=color, edgecolor='black', alpha=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df[task_col])
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set_title(title, size=14, pad=20)
    ax.set_xlabel('Time')
    ax.set_ylabel('Task')
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    # Apply styling
    create_gradient_background(ax)
    optimize_figure_size(ax)

def clean_plotly_figure_for_serialization(fig):
    """
    Clean Plotly figure data to ensure JSON serialization compatibility.
    Converts pandas Interval objects and other non-serializable types to strings.
    
    Args:
        fig: Plotly figure object
        
    Returns:
        Plotly figure object with cleaned data
    """
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    
    if not hasattr(fig, 'data') or not hasattr(fig, 'layout'):
        return fig
    
    # Create a copy of the figure to avoid modifying the original
    cleaned_fig = go.Figure(fig)
    
    # Clean each trace in the figure
    for trace in cleaned_fig.data:
        # Convert pandas Interval objects to strings in x and y data
        if hasattr(trace, 'x') and trace.x is not None:
            if isinstance(trace.x, pd.Series):
                trace.x = trace.x.astype(str)
            elif isinstance(trace.x, (list, np.ndarray)):
                trace.x = [str(x) if hasattr(x, 'left') and hasattr(x, 'right') else x for x in trace.x]
        
        if hasattr(trace, 'y') and trace.y is not None:
            if isinstance(trace.y, pd.Series):
                trace.y = trace.y.astype(str)
            elif isinstance(trace.y, (list, np.ndarray)):
                trace.y = [str(y) if hasattr(y, 'left') and hasattr(y, 'right') else y for y in trace.y]
        
        # Clean customdata if it exists
        if hasattr(trace, 'customdata') and trace.customdata is not None:
            if isinstance(trace.customdata, (list, np.ndarray)):
                trace.customdata = [
                    [str(item) if hasattr(item, 'left') and hasattr(item, 'right') else item 
                     for item in row] if isinstance(row, (list, np.ndarray)) else row
                    for row in trace.customdata
                ]
        
        # Clean text data if it exists
        if hasattr(trace, 'text') and trace.text is not None:
            if isinstance(trace.text, pd.Series):
                trace.text = trace.text.astype(str)
            elif isinstance(trace.text, (list, np.ndarray)):
                trace.text = [str(t) if hasattr(t, 'left') and hasattr(t, 'right') else t for t in trace.text]
    
    return cleaned_fig


def safe_plotly_to_html(fig, **kwargs):
    """
    Safely convert Plotly figure to HTML, handling non-serializable objects.
    
    Args:
        fig: Plotly figure object
        **kwargs: Additional arguments for plotly.offline.plot
        
    Returns:
        str: HTML content
    """
    import plotly.offline as pyo
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Clean the figure before serialization
    cleaned_fig = clean_plotly_figure_for_serialization(fig)
    
    try:
        # Try to convert to HTML
        html_content = pyo.plot(cleaned_fig, output_type='div', include_plotlyjs=True, **kwargs)
        return html_content
    except TypeError as e:
        # If still fails, try with a more aggressive cleaning
        if "not JSON serializable" in str(e):
            logger.warning(f"🔧 JSON serialization failed, attempting minimal figure creation: {e}")
            
            # Create a minimal figure with just the essential data
            import plotly.graph_objects as go
            
            minimal_fig = go.Figure()
            for trace in cleaned_fig.data:
                try:
                    # Create a new trace with only basic data
                    new_trace = go.Scatter(
                        x=trace.x if hasattr(trace, 'x') else None,
                        y=trace.y if hasattr(trace, 'y') else None,
                        mode=trace.mode if hasattr(trace, 'mode') else 'lines+markers',
                        name=trace.name if hasattr(trace, 'name') else None,
                        type=trace.type if hasattr(trace, 'type') else 'scatter'
                    )
                    minimal_fig.add_trace(new_trace)
                except Exception as trace_error:
                    logger.warning(f"🔧 Failed to add trace to minimal figure: {trace_error}")
                    continue
            
            # Copy layout
            if hasattr(cleaned_fig, 'layout'):
                minimal_fig.layout = cleaned_fig.layout
            
            try:
                return pyo.plot(minimal_fig, output_type='div', include_plotlyjs=True, **kwargs)
            except Exception as minimal_error:
                logger.error(f"❌ Even minimal figure creation failed: {minimal_error}")
                # Last resort: return a simple error message as HTML
                return f"""
                <div style="padding: 20px; border: 1px solid #ccc; background: #f9f9f9;">
                    <h3>⚠️ Plot Display Error</h3>
                    <p>The plot could not be displayed due to data serialization issues.</p>
                    <p><strong>Error:</strong> {str(e)}</p>
                    <p>This typically happens when the plot contains complex data types that cannot be serialized to JSON.</p>
                </div>
                """
        else:
            raise e
    except Exception as e:
        logger.error(f"❌ Unexpected error in safe_plotly_to_html: {e}")
        # Return a simple error message as HTML
        return f"""
        <div style="padding: 20px; border: 1px solid #ccc; background: #f9f9f9;">
            <h3>⚠️ Plot Display Error</h3>
            <p>The plot could not be displayed due to an unexpected error.</p>
            <p><strong>Error:</strong> {str(e)}</p>
        </div>
        """


def safe_plotly_figure_for_streamlit(fig, theme: str = 'auto', variant: str = 'professional'):
    """
    Clean Plotly figure for safe display in Streamlit with proper theming.
    This function ensures the figure can be safely passed to st.plotly_chart().
    
    Args:
        fig: Plotly figure object
        theme: 'light', 'dark', 'auto'
        variant: 'professional', 'modern', 'minimal', 'colorful'
        
    Returns:
        Plotly figure object safe for Streamlit display with proper theming
    """
    import plotly.graph_objects as go
    import logging
    
    logger = logging.getLogger(__name__)
    
    if not hasattr(fig, 'data') or not hasattr(fig, 'layout'):
        return fig
    
    try:
        # Clean the figure using the existing function
        cleaned_fig = clean_plotly_figure_for_serialization(fig)
        
        # Apply proper theming to the cleaned figure
        apply_plotly_theme(cleaned_fig, theme, variant)
        apply_plotly_trace_colors(cleaned_fig, theme=theme, variant=variant)
        enhance_plotly_accessibility(cleaned_fig, theme)
        
        return cleaned_fig
    except Exception as e:
        logger.warning(f"🔧 Failed to clean figure for Streamlit: {e}")
        
        # Create a minimal safe figure as fallback with proper theming
        try:
            minimal_fig = go.Figure()
            for trace in fig.data:
                try:
                    # Create a new trace with only basic data
                    new_trace = go.Scatter(
                        x=trace.x if hasattr(trace, 'x') else None,
                        y=trace.y if hasattr(trace, 'y') else None,
                        mode=trace.mode if hasattr(trace, 'mode') else 'lines+markers',
                        name=trace.name if hasattr(trace, 'name') else None,
                        type=trace.type if hasattr(trace, 'type') else 'scatter'
                    )
                    minimal_fig.add_trace(new_trace)
                except Exception as trace_error:
                    logger.warning(f"🔧 Failed to add trace to minimal Streamlit figure: {trace_error}")
                    continue
            
            # Apply proper theming to the minimal figure
            apply_plotly_theme(minimal_fig, theme, variant)
            apply_plotly_trace_colors(minimal_fig, theme=theme, variant=variant)
            enhance_plotly_accessibility(minimal_fig, theme)
            
            return minimal_fig
        except Exception as fallback_error:
            logger.error(f"❌ Failed to create minimal figure for Streamlit: {fallback_error}")
            # Return a simple empty figure with proper theming as last resort
            empty_fig = go.Figure()
            apply_plotly_theme(empty_fig, theme, variant)
            return empty_fig

def get_plotly_theme_colors(theme: str = 'light', variant: str = 'professional') -> dict:
    """
    Get comprehensive Plotly theme colors for different backgrounds and themes.
    
    Args:
        theme: 'light', 'dark', 'auto' (detects system preference)
        variant: 'professional', 'modern', 'minimal', 'colorful'
        
    Returns:
        dict: Complete color scheme for Plotly charts
    """
    import plotly.graph_objects as go
    
    # Base themes
    themes = {
        'light': {
            'background': '#ffffff',
            'paper': '#ffffff',
            'text': '#2c3e50',
            'grid': '#ecf0f1',
            'axis': '#bdc3c7',
            'title': '#2c3e50',
            'subtitle': '#7f8c8d'
        },
        'dark': {
            'background': '#1a1a1a',
            'paper': '#2d2d2d',
            'text': '#ffffff',
            'grid': '#404040',
            'axis': '#666666',
            'title': '#ffffff',
            'subtitle': '#cccccc'
        },
        'auto': {
            'background': '#ffffff',  # Will be overridden by system detection
            'paper': '#ffffff',
            'text': '#2c3e50',
            'grid': '#ecf0f1',
            'axis': '#bdc3c7',
            'title': '#2c3e50',
            'subtitle': '#7f8c8d'
        }
    }
    
    # Color palettes for different variants
    palettes = {
        'professional': {
            'primary': ['#3498db', '#2980b9', '#1f618d', '#154360'],
            'secondary': ['#e74c3c', '#c0392b', '#a93226', '#922b21'],
            'accent': ['#f39c12', '#e67e22', '#d35400', '#ba4a00'],
            'success': ['#27ae60', '#229954', '#1e8449', '#196f3d'],
            'neutral': ['#95a5a6', '#7f8c8d', '#6c7b7d', '#5d6d7e']
        },
        'modern': {
            'primary': ['#2c3e50', '#34495e', '#3498db', '#2980b9'],
            'secondary': ['#e74c3c', '#c0392b', '#a93226', '#922b21'],
            'accent': ['#1abc9c', '#16a085', '#138d75', '#117a65'],
            'success': ['#27ae60', '#229954', '#1e8449', '#196f3d'],
            'neutral': ['#95a5a6', '#7f8c8d', '#6c7b7d', '#5d6d7e']
        },
        'minimal': {
            'primary': ['#2c3e50', '#34495e', '#5d6d7e', '#85929e'],
            'secondary': ['#e74c3c', '#c0392b', '#a93226', '#922b21'],
            'accent': ['#f39c12', '#e67e22', '#d35400', '#ba4a00'],
            'success': ['#27ae60', '#229954', '#1e8449', '#196f3d'],
            'neutral': ['#bdc3c7', '#aeb6bf', '#85929e', '#6c7b7d']
        },
        'colorful': {
            'primary': ['#3498db', '#e74c3c', '#f39c12', '#27ae60'],
            'secondary': ['#9b59b6', '#e67e22', '#1abc9c', '#34495e'],
            'accent': ['#f1c40f', '#e74c3c', '#3498db', '#2ecc71'],
            'success': ['#27ae60', '#2ecc71', '#16a085', '#138d75'],
            'neutral': ['#95a5a6', '#7f8c8d', '#6c7b7d', '#5d6d7e']
        }
    }
    
    # Get base theme colors
    base_colors = themes.get(theme, themes['light'])
    palette = palettes.get(variant, palettes['professional'])
    
    # Auto theme detection (placeholder for future implementation)
    if theme == 'auto':
        # For now, default to light theme
        # In the future, this could detect system dark mode preference
        base_colors = themes['light']
    
    return {
        'background': base_colors['background'],
        'paper': base_colors['paper'],
        'text': base_colors['text'],
        'grid': base_colors['grid'],
        'axis': base_colors['axis'],
        'title': base_colors['title'],
        'subtitle': base_colors['subtitle'],
        'palette': palette
    }


def apply_plotly_theme(fig, theme: str = 'light', variant: str = 'professional', 
                      title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """
    Apply comprehensive theme to Plotly figure with proper contrast and colors.
    
    Args:
        fig: Plotly figure object
        theme: 'light', 'dark', 'auto'
        variant: 'professional', 'modern', 'minimal', 'colorful'
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    import plotly.graph_objects as go
    
    # Get theme colors
    colors = get_plotly_theme_colors(theme, variant)
    
    # Apply layout theme
    layout_kwargs = dict(
        # Background and paper colors
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['paper'],
        
        # Font family for all text
        font={
            'family': 'Arial, sans-serif',
            'color': colors['text'],
            'size': 12
        },
        
        # Margins for better spacing
        margin=dict(l=60, r=60, t=80, b=60),
        
        # Legend styling
        legend={
            'bgcolor': colors['paper'],
            'bordercolor': colors['axis'],
            'borderwidth': 1,
            'font': {'color': colors['text']}
        },
        
        # Hover template styling
        hoverlabel={
            'bgcolor': colors['paper'],
            'bordercolor': colors['axis'],
            'font': {'color': colors['text']}
        }
    )

    # Only set title if provided (avoid overwriting existing meaningful titles)
    if title:
        layout_kwargs['title'] = {
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {
                'size': 18,
                'color': colors['title'],
                'family': 'Arial, sans-serif'
            }
        }

    fig.update_layout(**layout_kwargs)
    
    # Apply axis styling
    xaxis_kwargs = dict(
        tickfont={'color': colors['text'], 'size': 11},
        gridcolor=colors['grid'],
        zerolinecolor=colors['axis'],
        linecolor=colors['axis'],
        showgrid=True,
        gridwidth=1
    )
    if xlabel:
        xaxis_kwargs['title'] = {'text': xlabel, 'font': {'color': colors['text'], 'size': 14}}
    fig.update_xaxes(**xaxis_kwargs)
    
    yaxis_kwargs = dict(
        tickfont={'color': colors['text'], 'size': 11},
        gridcolor=colors['grid'],
        zerolinecolor=colors['axis'],
        linecolor=colors['axis'],
        showgrid=True,
        gridwidth=1
    )
    if ylabel:
        yaxis_kwargs['title'] = {'text': ylabel, 'font': {'color': colors['text'], 'size': 14}}
    fig.update_yaxes(**yaxis_kwargs)


def create_plotly_color_palette(theme: str = 'light', variant: str = 'professional', 
                               n_colors: int = 8) -> list:
    """
    Create a color palette for Plotly charts with proper contrast.
    
    Args:
        theme: 'light', 'dark', 'auto'
        variant: 'professional', 'modern', 'minimal', 'colorful'
        n_colors: Number of colors needed
        
    Returns:
        list: Color palette optimized for the theme
    """
    colors = get_plotly_theme_colors(theme, variant)
    palette = colors['palette']
    
    # Flatten the palette and cycle if needed
    flat_palette = []
    for category in ['primary', 'secondary', 'accent', 'success', 'neutral']:
        flat_palette.extend(palette[category])
    
    # Ensure we have enough colors
    while len(flat_palette) < n_colors:
        flat_palette.extend(flat_palette[:n_colors - len(flat_palette)])
    
    return flat_palette[:n_colors]


def apply_plotly_trace_colors(fig, colors: list = None, theme: str = 'light', 
                             variant: str = 'professional') -> None:
    """
    Apply colors to Plotly figure traces with proper contrast.
    
    Args:
        fig: Plotly figure object
        colors: Optional list of colors to use
        theme: 'light', 'dark', 'auto'
        variant: 'professional', 'modern', 'minimal', 'colorful'
    """
    if colors is None:
        colors = create_plotly_color_palette(theme, variant, len(fig.data))
    
    # Apply colors to each trace
    for i, trace in enumerate(fig.data):
        color = colors[i % len(colors)]
        
        # Apply color based on trace type
        if hasattr(trace, 'marker'):
            trace.marker.color = color
        if hasattr(trace, 'line'):
            trace.line.color = color
        if hasattr(trace, 'fillcolor'):
            trace.fillcolor = color


def create_plotly_figure_with_theme(chart_type: str = 'scatter', theme: str = 'light', 
                                   variant: str = 'professional') -> object:
    """
    Create a Plotly figure with proper theme and styling.
    
    Args:
        chart_type: Type of chart ('scatter', 'bar', 'line', etc.)
        theme: 'light', 'dark', 'auto'
        variant: 'professional', 'modern', 'minimal', 'colorful'
        
    Returns:
        go.Figure: Pre-styled Plotly figure
    """
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Apply base theme
    apply_plotly_theme(fig, theme, variant)
    
    return fig


def enhance_plotly_accessibility(fig, theme: str = 'light') -> None:
    """
    Enhance Plotly figure accessibility with proper contrast and labels.
    
    Args:
        fig: Plotly figure object
        theme: 'light', 'dark', 'auto'
    """
    colors = get_plotly_theme_colors(theme)
    
    # Add better hover templates
    for trace in fig.data:
        if hasattr(trace, 'hovertemplate') and trace.hovertemplate is not None:
            # Ensure hover templates have good contrast
            trace.hovertemplate = trace.hovertemplate.replace(
                '<b>', f'<b style="color: {colors["title"]};">'
            )
    
    # Ensure all text has proper contrast
    fig.update_layout(
        font={'color': colors['text']},
        title={'font': {'color': colors['title']}},
        legend={'font': {'color': colors['text']}}
    )


def get_plotly_theme_for_streamlit() -> str:
    """
    Detect the best theme for Streamlit display.
    
    Returns:
        str: Recommended theme ('light' or 'dark')
    """
    # For now, return 'light' as default
    # In the future, this could detect Streamlit's theme setting
    return 'light'


def create_robust_plotly_chart(data_df: pd.DataFrame, x_col: str, y_col: str = None,
                               chart_type: str = 'auto', theme: str = 'auto',
                               variant: str = 'professional', title: str = "",
                               xlabel: str = "", ylabel: str = "") -> object:
    """
    Create a robust Plotly chart with proper theming and contrast.
    
    Args:
        data_df: DataFrame with data
        x_col: X-axis column
        y_col: Y-axis column (optional for some chart types)
        chart_type: 'auto', 'scatter', 'bar', 'line', 'histogram', 'box'
        theme: 'light', 'dark', 'auto'
        variant: 'professional', 'modern', 'minimal', 'colorful'
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        
    Returns:
        go.Figure: Styled Plotly figure
    """
    import plotly.graph_objects as go
    import plotly.express as px
    
    # Auto-detect theme if needed
    if theme == 'auto':
        theme = get_plotly_theme_for_streamlit()
    
    # Clean data to handle pandas Interval objects
    clean_df = data_df.copy()
    for col in clean_df.columns:
        if clean_df[col].dtype.name == 'category' and hasattr(clean_df[col].iloc[0], 'left'):
            # Convert Interval objects to strings
            clean_df[col] = clean_df[col].astype(str)
    
    # Auto-detect chart type if needed
    if chart_type == 'auto':
        if y_col is None:
            chart_type = 'histogram'
        elif clean_df[x_col].dtype == 'object' and clean_df[y_col].dtype in ['int64', 'float64']:
            chart_type = 'bar'
        else:
            chart_type = 'scatter'
    
    # Create figure based on chart type
    if chart_type == 'scatter':
        fig = px.scatter(clean_df, x=x_col, y=y_col, title=title)
    elif chart_type == 'bar':
        fig = px.bar(clean_df, x=x_col, y=y_col, title=title)
    elif chart_type == 'line':
        fig = px.line(clean_df, x=x_col, y=y_col, title=title)
    elif chart_type == 'histogram':
        fig = px.histogram(clean_df, x=x_col, title=title)
    elif chart_type == 'box':
        fig = px.box(clean_df, x=x_col, y=y_col, title=title)
    else:
        # Default to scatter
        fig = px.scatter(clean_df, x=x_col, y=y_col, title=title)
    
    # Apply comprehensive theming
    apply_plotly_theme(fig, theme, variant, title, xlabel, ylabel)
    apply_plotly_trace_colors(fig, theme=theme, variant=variant)
    enhance_plotly_accessibility(fig, theme)
    
    # Enhance legends and labels
    enhance_plotly_legends_and_labels(fig, data_df, x_col, y_col, chart_type, theme)
    
    return fig

def enhance_plotly_legends_and_labels(fig, data_df: pd.DataFrame, x_col: str, y_col: str, 
                                     chart_type: str, theme: str = 'light') -> None:
    """
    Enhance Plotly figure with better legends and labels.
    
    Args:
        fig: Plotly figure object
        data_df: DataFrame with data
        x_col: X-axis column name
        y_col: Y-axis column name
        chart_type: Type of chart
        theme: Theme for colors
    """
    colors = get_plotly_theme_colors(theme)
    
    # Force axis labels if missing; otherwise enhance if they're just raw column names
    if not getattr(fig.layout.xaxis.title, 'text', None):
        fig.update_xaxes(title_text=(x_col or 'X'))
    # Enhance axis labels if they're generic
    if fig.layout.xaxis.title.text == x_col:
        # Make x-axis label more descriptive
        x_label = x_col.replace('_', ' ').title()
        if x_col.lower() in ['date', 'time', 'datetime']:
            x_label = 'Date'
        elif x_col.lower() in ['age', 'years']:
            x_label = 'Age'
        elif x_col.lower() in ['salary', 'income', 'pay']:
            x_label = 'Salary'
        elif x_col.lower() in ['department', 'dept']:
            x_label = 'Department'
        elif x_col.lower() in ['category', 'cat']:
            x_label = 'Category'
        
        fig.update_xaxes(title_text=x_label)
    
    if not getattr(fig.layout.yaxis.title, 'text', None):
        fig.update_yaxes(title_text=(y_col or 'Y'))
    if y_col and fig.layout.yaxis.title.text == y_col:
        # Make y-axis label more descriptive
        y_label = y_col.replace('_', ' ').title()
        if y_col.lower() in ['count', 'frequency']:
            y_label = 'Count'
        elif y_col.lower() in ['salary', 'income', 'pay']:
            y_label = 'Salary'
        elif y_col.lower() in ['age', 'years']:
            y_label = 'Age'
        elif y_col.lower() in ['value', 'amount']:
            y_label = 'Value'
        
        fig.update_yaxes(title_text=y_label)
    
    # Enhance legends
    if fig.data:
        for i, trace in enumerate(fig.data):
            # Add better trace names for legends
            if not trace.name or trace.name == 'trace 0':
                if chart_type == 'scatter':
                    trace.name = f'{y_col} vs {x_col}'
                elif chart_type == 'bar':
                    trace.name = f'{y_col} by {x_col}'
                elif chart_type == 'line':
                    trace.name = f'{y_col} over {x_col}'
                elif chart_type == 'histogram':
                    trace.name = f'Distribution of {x_col}'
                elif chart_type == 'box':
                    trace.name = f'{y_col} by {x_col}'
                else:
                    trace.name = f'Chart {i+1}'
            
            # Add hover template with better formatting
            if hasattr(trace, 'hovertemplate') and trace.hovertemplate:
                # Enhance hover template
                if chart_type == 'scatter':
                    trace.hovertemplate = f'<b>{x_col.title()}</b>: %{{x}}<br>' + \
                                        f'<b>{y_col.title()}</b>: %{{y}}<br>' + \
                                        '<extra></extra>'
                elif chart_type == 'bar':
                    trace.hovertemplate = f'<b>{x_col.title()}</b>: %{{x}}<br>' + \
                                        f'<b>{y_col.title()}</b>: %{{y}}<br>' + \
                                        '<extra></extra>'
                elif chart_type == 'histogram':
                    trace.hovertemplate = f'<b>{x_col.title()}</b>: %{{x}}<br>' + \
                                        f'<b>Count</b>: %{{y}}<br>' + \
                                        '<extra></extra>'
    
    # Update legend styling
    fig.update_layout(
        legend=dict(
            bgcolor=colors['paper'],
            bordercolor=colors['axis'],
            borderwidth=1,
            font=dict(color=colors['text'], size=12),
            title=dict(text='Legend', font=dict(color=colors['title'], size=14))
        )
    )


def create_enhanced_multi_plotly_charts(data_df: pd.DataFrame, chart_configs: list, 
                                       theme: str = 'auto', variant: str = 'professional') -> list:
    """
    Create enhanced multiple Plotly charts with better labels, legends, and titles.
    
    Args:
        data_df: DataFrame with data
        chart_configs: List of chart configurations
        theme: 'light', 'dark', 'auto'
        variant: 'professional', 'modern', 'minimal', 'colorful'
        
    Returns:
        list: List of Plotly figures
    """
    import plotly.graph_objects as go
    import plotly.express as px
    
    figures = []
    
    for i, config in enumerate(chart_configs):
        try:
            # Extract configuration
            chart_type = config.get('type', 'auto')
            x_col = config.get('x')
            y_col = config.get('y')
            title = config.get('title', f'Chart {i+1}')
            xlabel = config.get('xlabel', x_col)
            ylabel = config.get('ylabel', y_col)
            
            # Create enhanced robust chart
            fig = create_robust_plotly_chart(
                data_df, x_col, y_col,
                chart_type=chart_type,
                theme=theme,
                variant=variant,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel
            )
            
            # Additional enhancements for multi-graph
            enhance_multi_graph_titles(fig, i+1, len(chart_configs), title)
            
            figures.append(fig)
            
        except Exception as e:
            # Create error figure
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"Error creating chart {i+1}: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="red")
            )
            error_fig.update_layout(
                title=f"Error - Chart {i+1}",
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            figures.append(error_fig)
    
    return figures


def enhance_multi_graph_titles(fig, chart_number: int, total_charts: int, title: str) -> None:
    """
    Enhance titles for multi-graph displays.
    
    Args:
        fig: Plotly figure object
        chart_number: Current chart number (1-based)
        total_charts: Total number of charts
        title: Original title
    """
    # Add chart number to title if there are multiple charts
    if total_charts > 1:
        enhanced_title = f"Chart {chart_number} of {total_charts}: {title}"
        fig.update_layout(title_text=enhanced_title)
    
    # Add subtitle with chart type if not present
    if not fig.layout.title.text:
        fig.update_layout(title_text=f"Chart {chart_number}: {title}")

def create_multi_plotly_charts(data_df: pd.DataFrame, chart_configs: list, 
                              theme: str = 'auto', variant: str = 'professional') -> list:
    """
    Create multiple Plotly charts based on configuration with enhanced labels and legends.
    
    Args:
        data_df: DataFrame with data
        chart_configs: List of chart configurations
        theme: 'light', 'dark', 'auto'
        variant: 'professional', 'modern', 'minimal', 'colorful'
        
    Returns:
        list: List of Plotly figures
    """
    # Use the enhanced version for better labels and legends
    return create_enhanced_multi_plotly_charts(data_df, chart_configs, theme, variant)


def create_multi_matplotlib_charts(data_df: pd.DataFrame, chart_configs: list,
                                 theme: str = 'professional') -> list:
    """
    Create multiple matplotlib charts based on configuration with enhanced labels and titles.
    
    Args:
        data_df: DataFrame with data
        chart_configs: List of chart configurations
        theme: 'professional', 'modern', 'minimal'
        
    Returns:
        list: List of matplotlib figures
    """
    import matplotlib.pyplot as plt
    
    figures = []
    
    for i, config in enumerate(chart_configs):
        try:
            # Extract configuration
            chart_type = config.get('type', 'auto')
            x_col = config.get('x')
            y_col = config.get('y')
            title = config.get('title', f'Chart {i+1}')
            xlabel = config.get('xlabel', x_col)
            ylabel = config.get('ylabel', y_col)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create chart based on type
            if chart_type == 'bar':
                create_clean_bar_chart(ax, data_df, x_col, y_col, title=title, xlabel=xlabel, ylabel=ylabel, theme=theme)
            elif chart_type == 'line':
                create_clean_line_chart(ax, data_df, x_col, y_col, title=title, xlabel=xlabel, ylabel=ylabel, theme=theme)
            elif chart_type == 'scatter':
                create_clean_scatter_plot(ax, data_df, x_col, y_col, title=title, xlabel=xlabel, ylabel=ylabel, theme=theme)
            elif chart_type == 'histogram':
                create_clean_histogram(ax, data_df, x_col, title=title, xlabel=xlabel, ylabel=ylabel, theme=theme)
            elif chart_type == 'box':
                create_clean_box_plot(ax, data_df, x_col, y_col, title=title, xlabel=xlabel, ylabel=ylabel, theme=theme)
            else:
                # Default to bar chart
                create_clean_bar_chart(ax, data_df, x_col, y_col, title=title, xlabel=xlabel, ylabel=ylabel, theme=theme)
            
            # Enhance title for multi-graph display
            if len(chart_configs) > 1:
                enhanced_title = f"Chart {i+1} of {len(chart_configs)}: {title}"
                ax.set_title(enhanced_title, fontsize=14, fontweight='bold', pad=20)
            
            # Enhance axis labels if they're generic
            enhance_matplotlib_labels(ax, x_col, y_col, chart_type)
            
            # Ensure legend is visible
            if ax.get_legend():
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            
            figures.append(fig)
            
        except Exception as e:
            # Create error figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Error creating chart {i+1}: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=12, color='red')
            ax.set_title(f'Error - Chart {i+1}')
            figures.append(fig)
    
    return figures


def enhance_matplotlib_labels(ax, x_col: str, y_col: str, chart_type: str) -> None:
    """
    Enhance matplotlib axis labels to be more descriptive.
    
    Args:
        ax: Matplotlib axes object
        x_col: X-axis column name
        y_col: Y-axis column name
        chart_type: Type of chart
    """
    # Enhance x-axis label
    if ax.get_xlabel() == x_col:
        x_label = x_col.replace('_', ' ').title()
        if x_col.lower() in ['date', 'time', 'datetime']:
            x_label = 'Date'
        elif x_col.lower() in ['age', 'years']:
            x_label = 'Age'
        elif x_col.lower() in ['salary', 'income', 'pay']:
            x_label = 'Salary'
        elif x_col.lower() in ['department', 'dept']:
            x_label = 'Department'
        elif x_col.lower() in ['category', 'cat']:
            x_label = 'Category'
        
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    
    # Enhance y-axis label
    if y_col and ax.get_ylabel() == y_col:
        y_label = y_col.replace('_', ' ').title()
        if y_col.lower() in ['count', 'frequency']:
            y_label = 'Count'
        elif y_col.lower() in ['salary', 'income', 'pay']:
            y_label = 'Salary'
        elif y_col.lower() in ['age', 'years']:
            y_label = 'Age'
        elif y_col.lower() in ['value', 'amount']:
            y_label = 'Value'
        
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')


def detect_multi_graph_result(result) -> tuple:
    """
    Detect if result contains multiple graphs and return appropriate format.
    
    Args:
        result: Execution result from code
        
    Returns:
        tuple: (is_multi_graph, figures, data_df)
    """
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    
    # Check if result is a list of figures
    if isinstance(result, list):
        # Check if all items are figures
        if all(isinstance(item, (plt.Figure, go.Figure)) for item in result):
            return True, result, None
        # Check if it's a list of tuples (fig, data)
        elif all(isinstance(item, tuple) and len(item) == 2 and 
                isinstance(item[0], (plt.Figure, go.Figure)) and 
                isinstance(item[1], pd.DataFrame) for item in result):
            figures = [item[0] for item in result]
            data_dfs = [item[1] for item in result]
            return True, figures, data_dfs
    
    # Check if result is a tuple with multiple figures
    elif isinstance(result, tuple):
        if len(result) == 2:
            fig, data = result
            # Check if fig is a list of figures
            if isinstance(fig, list) and all(isinstance(f, (plt.Figure, go.Figure)) for f in fig):
                return True, fig, data
            # Check if data is a list of dataframes
            elif isinstance(data, list) and all(isinstance(d, pd.DataFrame) for d in data):
                return True, fig, data
    
    return False, None, None


def format_multi_graph_response(figures: list, data_dfs: list = None, 
                              plot_engine: str = 'plotly') -> dict:
    """
    Format multiple graphs for display in the application.
    
    Args:
        figures: List of figures (matplotlib or Plotly)
        data_dfs: Optional list of DataFrames
        plot_engine: 'plotly' or 'matplotlib'
        
    Returns:
        dict: Formatted response for display
    """
    response = {
        'type': 'multi_graph',
        'plot_engine': plot_engine,
        'count': len(figures),
        'figures': figures,
        'data_dfs': data_dfs or [None] * len(figures)
    }
    
    # Add metadata for each figure
    for i, fig in enumerate(figures):
        if hasattr(fig, 'layout') and hasattr(fig, 'data'):
            # Plotly figure
            response[f'plotly_figure_{i}'] = fig
        else:
            # Matplotlib figure
            response[f'matplotlib_figure_{i}'] = fig
    
    return response


def create_multi_graph_code_template(chart_configs: list, plot_engine: str = 'plotly') -> str:
    """
    Generate code template for creating multiple graphs.
    
    Args:
        chart_configs: List of chart configurations
        plot_engine: 'plotly' or 'matplotlib'
        
    Returns:
        str: Code template
    """
    if plot_engine == 'plotly':
        code = """
# Create multiple Plotly charts
figures = []

"""
        for i, config in enumerate(chart_configs):
            chart_type = config.get('type', 'auto')
            x_col = config.get('x', 'x_column')
            y_col = config.get('y', 'y_column')
            title = config.get('title', f'Chart {i+1}')
            
            code += f"""
# Chart {i+1}: {title}
fig{i+1} = create_robust_plotly_chart(
    df, '{x_col}', '{y_col}',
    chart_type='{chart_type}',
    theme='auto',
    variant='professional',
    title='{title}'
)
figures.append(fig{i+1})

"""
        
        code += """
# Return multiple figures
result = figures
"""
    else:
        code = """
# Create multiple matplotlib charts
figures = []

"""
        for i, config in enumerate(chart_configs):
            chart_type = config.get('type', 'auto')
            x_col = config.get('x', 'x_column')
            y_col = config.get('y', 'y_column')
            title = config.get('title', f'Chart {i+1}')
            
            code += f"""
# Chart {i+1}: {title}
fig{i+1}, ax{i+1} = plt.subplots(figsize=(10, 6))
create_clean_{chart_type}_chart(ax{i+1}, df, '{x_col}', '{y_col}', title='{title}')
figures.append(fig{i+1})

"""
        
        code += """
# Return multiple figures
result = figures
"""
    
    return code


def validate_multi_graph_config(chart_configs: list) -> tuple:
    """
    Validate multi-graph configuration.
    
    Args:
        chart_configs: List of chart configurations
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(chart_configs, list):
        return False, "Chart configurations must be a list"
    
    if len(chart_configs) == 0:
        return False, "At least one chart configuration is required"
    
    for i, config in enumerate(chart_configs):
        if not isinstance(config, dict):
            return False, f"Chart configuration {i+1} must be a dictionary"
        
        required_fields = ['x']
        for field in required_fields:
            if field not in config:
                return False, f"Chart configuration {i+1} missing required field: {field}"
    
    return True, ""


def create_multi_graph_from_dataframe(df: pd.DataFrame, columns: list = None,
                                    chart_types: list = None, plot_engine: str = 'plotly') -> list:
    """
    Automatically create multiple graphs from DataFrame columns.
    
    Args:
        df: DataFrame with data
        columns: List of column names to plot (default: all numeric)
        chart_types: List of chart types (default: auto-detect)
        plot_engine: 'plotly' or 'matplotlib'
        
    Returns:
        list: List of figures
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    if chart_types is None:
        # Auto-detect chart types
        chart_types = ['histogram'] * len(columns)
    
    # Create configurations
    configs = []
    for i, col in enumerate(columns):
        chart_type = chart_types[i] if i < len(chart_types) else 'histogram'
        configs.append({
            'type': chart_type,
            'x': col,
            'title': f'Distribution of {col}'
        })
    
    # Create charts
    if plot_engine == 'plotly':
        return create_multi_plotly_charts(df, configs)
    else:
        return create_multi_matplotlib_charts(df, configs)