"""
Utility modules for the data analysis agent.
"""

from .plot_helpers import (
    format_axis_labels,
    apply_professional_styling,
    get_professional_colors,
    safe_color_access,
    create_category_palette,
    optimize_figure_size,
    create_clean_bar_chart,
    create_clean_line_chart,
    create_clean_scatter_plot,
    create_clean_histogram,
    create_clean_box_plot,
    create_clean_heatmap,
    create_clean_pie_chart,
    add_value_labels,
    smart_categorical_plot,
    handle_seaborn_warnings,
    smart_annotate_points
)

from .docx_utils import (
    text_to_docx,
    dataframe_to_docx_table,
    analysis_to_docx
)

__all__ = [
    'format_axis_labels',
    'apply_professional_styling', 
    'get_professional_colors',
    'safe_color_access',
    'create_category_palette',
    'optimize_figure_size',
    'create_clean_bar_chart',
    'create_clean_line_chart',
    'create_clean_scatter_plot',
    'create_clean_histogram',
    'create_clean_box_plot',
    'create_clean_heatmap',
    'create_clean_pie_chart',
    'add_value_labels',
    'smart_categorical_plot',
    'handle_seaborn_warnings',
    'smart_annotate_points',
    'text_to_docx',
    'dataframe_to_docx_table',
    'analysis_to_docx'
]