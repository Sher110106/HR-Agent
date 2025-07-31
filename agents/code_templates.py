"""
Code generation templates for consistent and maintainable code generation.

This module provides a template-based system for generating pandas/matplotlib
code with consistent patterns and professional styling.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ChartType(Enum):
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    PIE = "pie"
    HEATMAP = "heatmap"
    AREA = "area"

class AnalysisType(Enum):
    AGGREGATION = "aggregation"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"
    TREND = "trend"
    COMPARISON = "comparison"
    SUMMARY = "summary"

@dataclass
class CodeTemplate:
    """Template for generating specific types of code."""
    name: str
    template: str
    variables: List[str]
    description: str
    chart_type: Optional[ChartType] = None
    analysis_type: Optional[AnalysisType] = None

class CodeTemplateManager:
    """Manages code generation templates for different scenarios."""
    
    def __init__(self):
        self.templates: Dict[str, CodeTemplate] = {}
        self._initialize_templates()
        logger.info("📝 CodeTemplateManager initialized")
    
    def _initialize_templates(self):
        """Initialize all code generation templates."""
        
        # Bar Chart Template
        self.templates['bar_chart'] = CodeTemplate(
            name="bar_chart",
            template="""
# Create professional bar chart
fig, ax = plt.subplots(figsize=optimize_figure_size({x_count}, {y_count}))

# Prepare data
{data_preparation}

# Create bar chart with professional styling
bars = ax.bar({x_values}, {y_values}, 
               color=get_professional_colors()['primary'][0],
               edgecolor='white', linewidth=0.8, alpha=0.8)

# Add value labels
add_value_labels(ax, bars, {y_values})

# Apply professional styling
apply_professional_styling(ax, 
    title='{title}',
    xlabel='{xlabel}', 
    ylabel='{ylabel}')

# Return dual output
result = (fig, {data_df})
""",
            variables=['x_count', 'y_count', 'data_preparation', 'x_values', 'y_values', 
                     'title', 'xlabel', 'ylabel', 'data_df'],
            description="Professional bar chart with value labels",
            chart_type=ChartType.BAR
        )
        
        # Line Chart Template
        self.templates['line_chart'] = CodeTemplate(
            name="line_chart",
            template="""
# Create professional line chart
fig, ax = plt.subplots(figsize=optimize_figure_size({x_count}, {y_count}))

# Prepare data
{data_preparation}

# Create line chart with professional styling
line = ax.plot({x_values}, {y_values}, 
               color=get_professional_colors()['primary'][0],
               linewidth=2.5, marker='o', markersize=6)

# Add value labels
smart_annotate_points(ax, {x_values}, {y_values})

# Apply professional styling
apply_professional_styling(ax, 
    title='{title}',
    xlabel='{xlabel}', 
    ylabel='{ylabel}')

# Return dual output
result = (fig, {data_df})
""",
            variables=['x_count', 'y_count', 'data_preparation', 'x_values', 'y_values',
                     'title', 'xlabel', 'ylabel', 'data_df'],
            description="Professional line chart with markers and annotations",
            chart_type=ChartType.LINE
        )
        
        # Scatter Plot Template
        self.templates['scatter_plot'] = CodeTemplate(
            name="scatter_plot",
            template="""
# Create professional scatter plot
fig, ax = plt.subplots(figsize=optimize_figure_size({x_count}, {y_count}))

# Prepare data
{data_preparation}

# Create scatter plot with professional styling
scatter = ax.scatter({x_values}, {y_values}, 
                     c=get_professional_colors()['primary'][0],
                     alpha=0.7, s=50)

# Add trend line if correlation is strong
if len({x_values}) > 2:
    z = np.polyfit({x_values}, {y_values}, 1)
    p = np.poly1d(z)
    ax.plot({x_values}, p({x_values}), "--", 
            color=get_professional_colors()['secondary'][0], 
            alpha=0.8, linewidth=1.5)

# Apply professional styling
apply_professional_styling(ax, 
    title='{title}',
    xlabel='{xlabel}', 
    ylabel='{ylabel}')

# Return dual output
result = (fig, {data_df})
""",
            variables=['x_count', 'y_count', 'data_preparation', 'x_values', 'y_values',
                     'title', 'xlabel', 'ylabel', 'data_df'],
            description="Professional scatter plot with optional trend line",
            chart_type=ChartType.SCATTER
        )
        
        # Aggregation Analysis Template
        self.templates['aggregation_analysis'] = CodeTemplate(
            name="aggregation_analysis",
            template="""
# Perform aggregation analysis
{data_preparation}

# Group and aggregate data
result = {dataframe}.groupby({group_by}).agg({aggregations}).round(2)

# Sort by specified column if provided
if '{sort_by}' in result.columns:
    result = result.sort_values('{sort_by}', ascending={ascending})

# Return result
result
""",
            variables=['data_preparation', 'dataframe', 'group_by', 'aggregations', 
                     'sort_by', 'ascending'],
            description="Standard aggregation analysis with grouping",
            analysis_type=AnalysisType.AGGREGATION
        )
        
        # Correlation Analysis Template
        self.templates['correlation_analysis'] = CodeTemplate(
            name="correlation_analysis",
            template="""
# Perform correlation analysis
{data_preparation}

# Calculate correlation matrix
correlation_matrix = {dataframe}[{columns}].corr()

# Create heatmap
fig, ax = plt.subplots(figsize=optimize_figure_size(len({columns}), len({columns})))

# Create heatmap with professional styling
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='RdYlBu_r', 
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'shrink': 0.8})

# Apply professional styling
apply_professional_styling(ax, 
    title='{title}',
    xlabel='Variables', 
    ylabel='Variables')

# Return dual output
result = (fig, correlation_matrix)
""",
            variables=['data_preparation', 'dataframe', 'columns', 'title'],
            description="Correlation analysis with heatmap visualization",
            analysis_type=AnalysisType.CORRELATION,
            chart_type=ChartType.HEATMAP
        )
    
    def get_template(self, template_name: str) -> Optional[CodeTemplate]:
        """Get template by name."""
        return self.templates.get(template_name)
    
    def get_templates_by_type(self, chart_type: ChartType = None, 
                            analysis_type: AnalysisType = None) -> List[CodeTemplate]:
        """Get templates filtered by type."""
        templates = []
        for template in self.templates.values():
            if chart_type and template.chart_type == chart_type:
                templates.append(template)
            elif analysis_type and template.analysis_type == analysis_type:
                templates.append(template)
        return templates
    
    def render_template(self, template_name: str, variables: Dict[str, Any]) -> str:
        """Render a template with provided variables."""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Validate required variables
        missing_vars = [var for var in template.variables if var not in variables]
        if missing_vars:
            raise ValueError(f"Missing required variables for template '{template_name}': {missing_vars}")
        
        # Render template
        rendered = template.template
        for var_name, var_value in variables.items():
            placeholder = '{' + var_name + '}'
            rendered = rendered.replace(placeholder, str(var_value))
        
        logger.debug(f"📝 Rendered template '{template_name}' with {len(variables)} variables")
        return rendered
    
    def suggest_template(self, query: str, data_info: Dict[str, Any]) -> Optional[str]:
        """Suggest appropriate template based on query and data."""
        query_lower = query.lower()
        
        # Chart type detection
        if any(word in query_lower for word in ['bar', 'column', 'compare']):
            return 'bar_chart'
        elif any(word in query_lower for word in ['line', 'trend', 'time', 'over time']):
            return 'line_chart'
        elif any(word in query_lower for word in ['scatter', 'correlation', 'relationship']):
            return 'scatter_plot'
        elif any(word in query_lower for word in ['correlation', 'correlate']):
            return 'correlation_analysis'
        elif any(word in query_lower for word in ['group', 'aggregate', 'sum', 'count', 'average']):
            return 'aggregation_analysis'
        
        # Default to aggregation for analysis queries
        return 'aggregation_analysis'

# Global template manager
_template_manager = None

def get_template_manager() -> CodeTemplateManager:
    """Get the global template manager."""
    global _template_manager
    if _template_manager is None:
        _template_manager = CodeTemplateManager()
    return _template_manager 