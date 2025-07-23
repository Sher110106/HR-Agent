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
#  add_value_labels  –  MINIMAL MODE
# -------------------------------------------------------------------------
#  Provides clean, minimal value labels only when truly helpful.
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


# -------------------------------------------------------------------------
#  Everything below is unchanged – already produces professional charts
# -------------------------------------------------------------------------
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
    ylabel: str = ""
) -> None:
    """
    Apply professional styling to the plot.

    Args:
        ax (plt.Axes): The matplotlib axes object.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
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
    
    # Apply tight layout for better spacing
    try:
        ax.figure.tight_layout()
    except Exception:
        # Fallback if tight_layout fails
        pass


def get_professional_colors(palette: str = 'primary') -> dict:
    """
    Get professional color palettes for charts.

    Args:
        palette (str): Color palette name ('primary', 'secondary', 'categorical').
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
        }
    }
    return palettes.get(palette, palettes['primary'])


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
    """
    Suppress common seaborn warnings that don't affect functionality.
    """
    import warnings
    warnings.filterwarnings('ignore', message='.*points cannot be placed.*', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*you may want to decrease the size.*', category=UserWarning)


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
    legend_totals: bool = True
) -> None:
    """
    Create a clean bar chart without value labels, with proper legend.
    """
    import pandas as pd
    colors = get_professional_colors()['colors']
    # Slice colors to match number of categories/groups
    if hue_col and hue_col in data_df.columns:
        n_groups = data_df[hue_col].nunique()
        colors = colors[:n_groups]
        _plot_grouped_bar_chart(ax, data_df, x_col, y_col, hue_col, legend_totals, colors)
    else:
        n_cats = data_df[x_col].nunique()
        colors = colors[:n_cats]
        _plot_simple_bar_chart(ax, data_df, x_col, y_col, legend_totals, colors)
    apply_professional_styling(ax, title=title, xlabel=xlabel, ylabel=ylabel)
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
    line_style: str = '-'
) -> None:
    """
    Create a clean line chart with professional styling.
    """
    import pandas as pd
    colors = get_professional_colors()['colors']
    
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
    
    apply_professional_styling(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    format_axis_labels(ax, x_rotation=45)
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
    add_trendline: bool = False
) -> None:
    """
    Create a clean scatter plot with optional trend line.
    """
    import pandas as pd
    import numpy as np
    colors = get_professional_colors()['colors']
    
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
    
    apply_professional_styling(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    format_axis_labels(ax, x_rotation=0)  # Don't rotate for scatter plots
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
    show_stats: bool = True
) -> None:
    """
    Create a clean histogram with optional statistics.
    """
    import pandas as pd
    import numpy as np
    colors = get_professional_colors()['colors']
    
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
    
    apply_professional_styling(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    format_axis_labels(ax, x_rotation=45)
    optimize_figure_size(ax)


def create_clean_box_plot(
    ax: plt.Axes,
    data_df: 'pd.DataFrame',
    x_col: str,
    y_col: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    show_outliers: bool = True
) -> None:
    """
    Create a clean box plot with professional styling.
    """
    import seaborn as sns
    colors = get_professional_colors()['colors']
    
    # Suppress seaborn warnings
    handle_seaborn_warnings()
    
    # Create box plot
    n_categories = data_df[x_col].nunique()
    palette = colors[:n_categories] if n_categories <= len(colors) else colors
    sns.boxplot(data=data_df, x=x_col, y=y_col, ax=ax, 
               palette=palette, showfliers=show_outliers)
    
    apply_professional_styling(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    format_axis_labels(ax, x_rotation=45)
    optimize_figure_size(ax)


def create_clean_heatmap(
    ax: plt.Axes,
    data_df: 'pd.DataFrame',
    title: str = "",
    cmap: str = 'RdYlBu_r',
    show_values: bool = False,
    fmt: str = '.2f'
) -> None:
    """
    Create a clean correlation heatmap.
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
    
    apply_professional_styling(ax, title=title, xlabel="", ylabel="")
    # Don't rotate labels for heatmaps
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    optimize_figure_size(ax)


def create_clean_pie_chart(
    ax: plt.Axes,
    data_df: 'pd.DataFrame',
    col: str,
    title: str = "",
    show_percentages: bool = True,
    explode_max: bool = True
) -> None:
    """
    Create a clean pie chart with professional styling.
    """
    import pandas as pd
    colors = get_professional_colors()['colors']
    
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