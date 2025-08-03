"""
Phase 3: Plot Quality & Intelligence Improvements

This module implements the enhanced plot quality system including:
- Code validation and automatic fixing
- Smart legend system
- Data validation and quality assessment
- Intelligent chart selection
- Quality metrics and scoring
"""

import re
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings

class PlotCodeValidator:
    """Validate and fix plot code for common issues."""
    
    def __init__(self):
        self.common_issues = {
            'missing_imports': [],
            'undefined_functions': [],
            'legend_inconsistencies': [],
            'data_validation_missing': [],
            'inappropriate_binning': [],
            'color_palette_issues': []
        }
    
    def validate_plot_code(self, code: str, data_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate generated plot code for common issues.
        
        Args:
            code: Python code string
            data_df: DataFrame used in the plot
            
        Returns:
            Dict with validation results
        """
        issues = []
        suggestions = []
        fixed_code = code
        
        # Check for missing imports
        if 'seaborn' in code and 'import seaborn' not in code:
            issues.append("Missing seaborn import")
            suggestions.append("Add: import seaborn as sns")
            fixed_code = "import seaborn as sns\n" + fixed_code
        
        # Check for undefined functions
        if 'create_category_palette' in code and 'def create_category_palette' not in code:
            issues.append("Missing function definition: create_category_palette")
            suggestions.append("Use get_professional_colors() instead")
            fixed_code = fixed_code.replace(
                'create_category_palette',
                'get_professional_colors()["colors"]'
            )
        
        # Check for legend inconsistencies
        if 'legend=False' in code and 'ax.legend(' in code:
            issues.append("Inconsistent legend handling")
            suggestions.append("Remove legend=False or use automatic legend")
            fixed_code = fixed_code.replace('legend=False,', '')
        
        # Check for data validation
        if 'dropna()' not in code and data_df.isnull().any().any():
            issues.append("No data validation for missing values")
            suggestions.append("Add data validation before plotting")
            fixed_code = "# Data validation\n" + fixed_code
        
        # Check for appropriate binning
        if 'bins=30' in code and len(data_df) < 100:
            issues.append("Too many bins for small dataset")
            suggestions.append("Use fewer bins or automatic binning")
            fixed_code = fixed_code.replace('bins=30', 'bins="auto"')
        
        # Check for color palette issues
        if 'palette=' in code and 'get_professional_colors' not in code:
            issues.append("Non-standard color palette")
            suggestions.append("Use get_professional_colors() for consistency")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'suggestions': suggestions,
            'fixed_code': fixed_code,
            'severity': 'high' if len(issues) > 3 else 'medium' if len(issues) > 1 else 'low'
        }
    
    def fix_plot_code(self, code: str, issues: List[str], suggestions: List[str]) -> str:
        """
        Automatically fix common plot code issues.
        
        Args:
            code: Original code string
            issues: List of detected issues
            suggestions: List of suggestions
            
        Returns:
            Fixed code string
        """
        fixed_code = code
        
        # Apply fixes based on issues
        for issue in issues:
            if "Missing seaborn import" in issue:
                fixed_code = "import seaborn as sns\n" + fixed_code
            
            elif "create_category_palette" in issue:
                fixed_code = fixed_code.replace(
                    'create_category_palette',
                    'get_professional_colors()["colors"]'
                )
            
            elif "legend handling" in issue:
                fixed_code = fixed_code.replace('legend=False,', '')
                fixed_code = fixed_code.replace('ax.legend(', '# ax.legend(')
            
            elif "data validation" in issue:
                fixed_code = "# Data validation\n" + fixed_code
            
            elif "binning" in issue:
                fixed_code = fixed_code.replace('bins=30', 'bins="auto"')
        
        return fixed_code


class SmartLegendSystem:
    """Create intelligent legends with proper color mapping and statistics."""
    
    def __init__(self):
        self.legend_templates = {
            'bar_chart': '{category} (n={count}, {percentage:.1f}%)',
            'line_chart': '{category} (n={count})',
            'scatter_plot': '{category} (n={count})',
            'histogram': 'Count: {count}, Mean: {mean:.2f}',
            'box_plot': '{category} (n={count}, median={median:.2f})'
        }
    
    def create_smart_legend(self, ax: plt.Axes, data_df: pd.DataFrame, 
                           hue_col: str = None, chart_type: str = 'auto') -> None:
        """
        Create intelligent legend with proper color mapping and statistics.
        
        Args:
            ax: Matplotlib axes object
            data_df: DataFrame with data
            hue_col: Optional grouping column
            chart_type: Type of chart for legend template
        """
        if not hue_col or hue_col not in data_df.columns:
            return
        
        # Get unique categories and their counts
        categories = data_df[hue_col].value_counts()
        
        # Get colors from the plot
        handles, labels = ax.get_legend_handles_labels()
        
        if not handles:
            return
        
        # Create proper legend labels with statistics
        legend_labels = []
        for i, (category, count) in enumerate(categories.items()):
            percentage = (count / len(data_df)) * 100
            
            if chart_type == 'bar_chart':
                label = self.legend_templates['bar_chart'].format(
                    category=category, count=count, percentage=percentage
                )
            elif chart_type == 'line_chart':
                label = self.legend_templates['line_chart'].format(
                    category=category, count=count
                )
            elif chart_type == 'scatter_plot':
                label = self.legend_templates['scatter_plot'].format(
                    category=category, count=count
                )
            else:
                label = f"{category} (n={count}, {percentage:.1f}%)"
            
            legend_labels.append(label)
        
        # Position legend intelligently
        if len(categories) <= 3:
            ax.legend(handles, legend_labels, title=hue_col, loc='upper right')
        else:
            ax.legend(handles, legend_labels, title=hue_col, 
                     loc='center left', bbox_to_anchor=(1, 0.5))
    
    def add_statistical_legend(self, ax: plt.Axes, data_df: pd.DataFrame, 
                              y_col: str, chart_type: str = 'auto') -> None:
        """
        Add statistical information to legend.
        
        Args:
            ax: Matplotlib axes object
            data_df: DataFrame with data
            y_col: Column for statistical calculations
            chart_type: Type of chart
        """
        if y_col not in data_df.columns:
            return
        
        # Calculate statistics
        mean_val = data_df[y_col].mean()
        median_val = data_df[y_col].median()
        count_val = len(data_df[y_col].dropna())
        
        # Create statistical text
        if chart_type == 'histogram':
            stats_text = f"Count: {count_val}, Mean: {mean_val:.2f}"
        elif chart_type == 'box_plot':
            stats_text = f"n={count_val}, median={median_val:.2f}"
        else:
            stats_text = f"n={count_val}, mean={mean_val:.2f}"
        
        # Add to legend or as text annotation
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='white', alpha=0.8), fontsize=8)


class DataValidator:
    """Validate data for plotting and suggest improvements."""
    
    def __init__(self):
        self.validation_rules = {
            'missing_values': True,
            'outliers': True,
            'data_types': True,
            'sample_size': True,
            'value_ranges': True
        }
    
    def validate_plot_data(self, data_df: pd.DataFrame, x_col: str, 
                          y_col: str = None, hue_col: str = None) -> Dict[str, Any]:
        """
        Validate data for plotting and suggest improvements.
        
        Args:
            data_df: DataFrame with data
            x_col: Column for x-axis
            y_col: Optional column for y-axis
            hue_col: Optional grouping column
            
        Returns:
            Dict with validation results
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'suggestions': [],
            'cleaned_data': data_df.copy(),
            'quality_score': 1.0
        }
        
        # Check for missing values
        plot_cols = [x_col] + ([y_col] if y_col else []) + ([hue_col] if hue_col else [])
        missing_data = data_df[plot_cols].isnull().sum()
        
        if missing_data.any():
            validation['warnings'].append(f"Missing values detected: {missing_data.to_dict()}")
            validation['suggestions'].append("Consider handling missing values before plotting")
            validation['quality_score'] *= 0.9
        
        # Check data types
        if not pd.api.types.is_numeric_dtype(data_df[x_col]):
            validation['warnings'].append(f"X-axis column '{x_col}' is not numeric")
            validation['suggestions'].append("Consider converting to numeric or using categorical plot")
        
        if y_col and not pd.api.types.is_numeric_dtype(data_df[y_col]):
            validation['warnings'].append(f"Y-axis column '{y_col}' is not numeric")
            validation['suggestions'].append("Consider converting to numeric for better visualization")
        
        # Check for appropriate binning
        if pd.api.types.is_numeric_dtype(data_df[x_col]):
            data_range = data_df[x_col].max() - data_df[x_col].min()
            suggested_bins = min(30, max(5, int(len(data_df) / 20)))
            validation['suggestions'].append(f"Consider using {suggested_bins} bins for better visualization")
        
        # Check for outliers
        if pd.api.types.is_numeric_dtype(data_df[x_col]):
            outliers = self._detect_outliers(data_df[x_col])
            if len(outliers) > 0:
                validation['warnings'].append(f"Outliers detected in '{x_col}': {len(outliers)} points")
                validation['suggestions'].append("Consider outlier handling or robust statistics")
                validation['quality_score'] *= 0.95
        
        if y_col and pd.api.types.is_numeric_dtype(data_df[y_col]):
            outliers = self._detect_outliers(data_df[y_col])
            if len(outliers) > 0:
                validation['warnings'].append(f"Outliers detected in '{y_col}': {len(outliers)} points")
                validation['suggestions'].append("Consider outlier handling or robust statistics")
                validation['quality_score'] *= 0.95
        
        # Check sample size
        if len(data_df) < 10:
            validation['warnings'].append("Small sample size may affect plot reliability")
            validation['suggestions'].append("Consider collecting more data or using different visualization")
            validation['quality_score'] *= 0.8
        
        # Clean data if needed
        if validation['warnings']:
            validation['cleaned_data'] = self._clean_data(data_df, plot_cols)
        
        return validation
    
    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series[(series < lower_bound) | (series > upper_bound)]
    
    def _clean_data(self, data_df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Clean data by handling missing values and outliers."""
        cleaned_df = data_df.copy()
        
        # Handle missing values
        for col in columns:
            if col in cleaned_df.columns and cleaned_df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    # Fill with median for numeric columns
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                else:
                    # Fill with mode for categorical columns
                    mode_val = cleaned_df[col].mode().iloc[0] if not cleaned_df[col].mode().empty else 'Unknown'
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val)
        
        return cleaned_df


class ChartRecommender:
    """Recommend the best chart type based on data characteristics."""
    
    def __init__(self):
        self.chart_rules = {
            'numeric_numeric': ['scatter_plot', 'line_plot', 'regression_plot'],
            'numeric_categorical': ['histogram', 'density_plot', 'box_plot'],
            'categorical_numeric': ['bar_plot', 'violin_plot', 'swarm_plot'],
            'categorical_categorical': ['count_plot', 'pie_chart', 'bar_plot'],
            'time_series': ['line_plot', 'area_plot', 'scatter_plot'],
            'distribution': ['histogram', 'density_plot', 'violin_plot'],
            'correlation': ['scatter_plot', 'heatmap', 'regression_plot']
        }
    
    def recommend_chart_type(self, data_df: pd.DataFrame, x_col: str, 
                            y_col: str = None, hue_col: str = None) -> Dict[str, Any]:
        """
        Recommend the best chart type based on data characteristics.
        
        Args:
            data_df: DataFrame with data
            x_col: Column for x-axis
            y_col: Optional column for y-axis
            hue_col: Optional grouping column
            
        Returns:
            Dict with chart recommendations
        """
        recommendations = {
            'primary_chart': None,
            'alternative_charts': [],
            'reasoning': [],
            'data_insights': [],
            'confidence_score': 0.0
        }
        
        # Analyze data characteristics
        x_numeric = pd.api.types.is_numeric_dtype(data_df[x_col])
        y_numeric = y_col and pd.api.types.is_numeric_dtype(data_df[y_col])
        hue_categorical = hue_col and not pd.api.types.is_numeric_dtype(data_df[hue_col])
        
        # Determine primary chart type
        if x_numeric and y_numeric:
            if hue_categorical:
                recommendations['primary_chart'] = 'violin_plot'
                recommendations['alternative_charts'] = ['box_plot', 'swarm_plot']
                recommendations['confidence_score'] = 0.9
            else:
                # Check for correlation
                correlation = abs(data_df[x_col].corr(data_df[y_col]))
                if correlation > 0.7:
                    recommendations['primary_chart'] = 'scatter_plot'
                    recommendations['alternative_charts'] = ['line_plot', 'regression_plot']
                else:
                    recommendations['primary_chart'] = 'scatter_plot'
                    recommendations['alternative_charts'] = ['line_plot', 'hexbin_plot']
                recommendations['confidence_score'] = 0.8
        
        elif x_numeric and not y_numeric:
            recommendations['primary_chart'] = 'histogram'
            recommendations['alternative_charts'] = ['density_plot', 'box_plot']
            recommendations['confidence_score'] = 0.9
        
        elif not x_numeric and y_numeric:
            if len(data_df[x_col].unique()) <= 10:
                recommendations['primary_chart'] = 'bar_plot'
                recommendations['alternative_charts'] = ['violin_plot', 'swarm_plot']
            else:
                recommendations['primary_chart'] = 'violin_plot'
                recommendations['alternative_charts'] = ['box_plot', 'swarm_plot']
            recommendations['confidence_score'] = 0.8
        
        else:
            recommendations['primary_chart'] = 'count_plot'
            recommendations['alternative_charts'] = ['pie_chart', 'bar_plot']
            recommendations['confidence_score'] = 0.7
        
        # Radar: Recommend if wide data (many columns, all numeric, few rows)
        if data_df.shape[1] >= 4 and all(pd.api.types.is_numeric_dtype(data_df[c]) for c in data_df.columns) and data_df.shape[0] <= 10:
            recommendations['primary_chart'] = 'radar'
            recommendations['alternative_charts'] = ['bar_plot', 'line_plot']
            recommendations['confidence_score'] = 0.85
            recommendations['reasoning'].append('Wide numeric data, suitable for radar chart.')
        # Treemap: Recommend if single categorical + value column, many categories
        elif len(data_df.columns) == 2 and pd.api.types.is_numeric_dtype(data_df.iloc[:,1]) and data_df.iloc[:,0].nunique() > 5:
            recommendations['primary_chart'] = 'treemap'
            recommendations['alternative_charts'] = ['bar_plot', 'pie_chart']
            recommendations['confidence_score'] = 0.8
            recommendations['reasoning'].append('Hierarchical or categorical data, suitable for treemap.')
        # Gantt: Recommend if columns match task, start, end (date/time)
        elif set(['task','start','end']).issubset(set(data_df.columns)):
            recommendations['primary_chart'] = 'gantt'
            recommendations['alternative_charts'] = ['bar_plot']
            recommendations['confidence_score'] = 0.8
            recommendations['reasoning'].append('Task scheduling data, suitable for Gantt chart.')
        
        # Add reasoning
        recommendations['reasoning'].append(f"X-axis: {'numeric' if x_numeric else 'categorical'}")
        if y_col:
            recommendations['reasoning'].append(f"Y-axis: {'numeric' if y_numeric else 'categorical'}")
        if hue_col:
            recommendations['reasoning'].append(f"Grouping: {'categorical' if hue_categorical else 'numeric'}")
        
        # Add data insights
        if y_col and y_numeric:
            mean_val = data_df[y_col].mean()
            std_val = data_df[y_col].std()
            recommendations['data_insights'].append(f"Mean: {mean_val:.2f}, Std: {std_val:.2f}")
        
        return recommendations
    
    def get_hr_chart_template(self, chart_type: str, data_type: str) -> Dict[str, Any]:
        """
        Get HR-specific chart templates with best practices.
        
        Args:
            chart_type: Type of chart
            data_type: Type of data analysis
            
        Returns:
            Dict with chart template
        """
        templates = {
            'attrition_analysis': {
                'chart_type': 'stacked_bar',
                'color_palette': 'attrition',
                'annotations': True,
                'insights': True,
                'title_template': 'Employee Attrition Analysis by {dimension}',
                'best_practices': [
                    'Use red for attrition, green for retention',
                    'Include percentages in annotations',
                    'Add trend lines for time series'
                ]
            },
            'salary_analysis': {
                'chart_type': 'violin_plot',
                'color_palette': 'performance',
                'annotations': True,
                'insights': True,
                'title_template': 'Salary Distribution by {dimension}',
                'best_practices': [
                    'Show median and quartiles',
                    'Highlight outliers',
                    'Include sample sizes'
                ]
            },
            'tenure_analysis': {
                'chart_type': 'histogram',
                'color_palette': 'neutral',
                'annotations': True,
                'insights': True,
                'title_template': 'Employee Tenure Distribution',
                'best_practices': [
                    'Use appropriate bin sizes',
                    'Show cumulative distribution',
                    'Highlight retention milestones'
                ]
            },
            'performance_analysis': {
                'chart_type': 'box_plot',
                'color_palette': 'performance',
                'annotations': True,
                'insights': True,
                'title_template': 'Performance Distribution by {dimension}',
                'best_practices': [
                    'Show quartiles and outliers',
                    'Include statistical significance',
                    'Compare across groups'
                ]
            }
        }
        
        return templates.get(chart_type, {})


class PlotQualityAssessor:
    """Assess the quality of generated plots."""
    
    def __init__(self):
        self.quality_metrics = {
            'readability': 0.0,
            'information_density': 0.0,
            'aesthetic_appeal': 0.0,
            'data_accuracy': 0.0,
            'overall_score': 0.0
        }
    
    def assess_plot_quality(self, fig: plt.Figure, data_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess the quality of a generated plot.
        
        Args:
            fig: Matplotlib figure object
            data_df: DataFrame used in the plot
            
        Returns:
            Dict with quality assessment
        """
        quality_metrics = {
            'readability': 0.0,
            'information_density': 0.0,
            'aesthetic_appeal': 0.0,
            'data_accuracy': 0.0,
            'overall_score': 0.0,
            'issues': [],
            'improvements': []
        }
        
        # Check readability
        axes = fig.get_axes()
        for ax in axes:
            # Check title clarity
            if ax.get_title():
                quality_metrics['readability'] += 0.2
            
            # Check axis labels
            if ax.get_xlabel() and ax.get_ylabel():
                quality_metrics['readability'] += 0.2
            
            # Check legend quality
            if ax.get_legend():
                quality_metrics['readability'] += 0.2
            
            # Check grid readability
            if ax.grid():
                quality_metrics['readability'] += 0.1
            
            # Check for overlapping elements
            if self._check_overlapping_elements(ax):
                quality_metrics['issues'].append("Overlapping elements detected")
                quality_metrics['improvements'].append("Adjust element positioning")
        
        # Check information density
        data_points = len(data_df)
        if data_points > 0:
            quality_metrics['information_density'] = min(1.0, data_points / 1000)
        
        # Check aesthetic appeal
        quality_metrics['aesthetic_appeal'] = self._assess_aesthetics(fig)
        
        # Check data accuracy
        quality_metrics['data_accuracy'] = self._assess_data_accuracy(fig, data_df)
        
        # Calculate overall score
        quality_metrics['overall_score'] = (
            quality_metrics['readability'] * 0.3 +
            quality_metrics['information_density'] * 0.2 +
            quality_metrics['aesthetic_appeal'] * 0.3 +
            quality_metrics['data_accuracy'] * 0.2
        )
        
        # Add quality level
        if quality_metrics['overall_score'] >= 0.8:
            quality_metrics['quality_level'] = 'Excellent'
        elif quality_metrics['overall_score'] >= 0.6:
            quality_metrics['quality_level'] = 'Good'
        elif quality_metrics['overall_score'] >= 0.4:
            quality_metrics['quality_level'] = 'Fair'
        else:
            quality_metrics['quality_level'] = 'Poor'
        
        return quality_metrics
    
    def _check_overlapping_elements(self, ax: plt.Axes) -> bool:
        """Check for overlapping elements in the plot."""
        # This is a simplified check - in practice, you'd need more sophisticated detection
        children = ax.get_children()
        text_elements = [child for child in children if hasattr(child, 'get_text')]
        return len(text_elements) > 5  # Simplified heuristic
    
    def _assess_aesthetics(self, fig: plt.Figure) -> float:
        """Assess the aesthetic appeal of the plot."""
        score = 0.8  # Base score for professional styling
        
        # Check for modern styling
        axes = fig.get_axes()
        for ax in axes:
            # Check for removed top/right spines
            if not ax.spines['top'].get_visible() and not ax.spines['right'].get_visible():
                score += 0.1
            
            # Check for subtle grid
            if ax.grid():
                score += 0.05
            
            # Check for good color usage
            if len(ax.get_children()) > 0:
                score += 0.05
        
        return min(1.0, score)
    
    def _assess_data_accuracy(self, fig: plt.Figure, data_df: pd.DataFrame) -> float:
        """Assess the accuracy of data representation."""
        # This is a simplified assessment - in practice, you'd need more sophisticated validation
        return 1.0  # Assume accurate if no errors detected


class PlotQualitySystem:
    """Main class for Phase 3 plot quality improvements."""
    
    def __init__(self):
        self.validator = PlotCodeValidator()
        self.legend_system = SmartLegendSystem()
        self.data_validator = DataValidator()
        self.chart_recommender = ChartRecommender()
        self.quality_assessor = PlotQualityAssessor()
    
    def process_plot_request(self, code: str, data_df: pd.DataFrame, 
                           x_col: str, y_col: str = None, hue_col: str = None) -> Dict[str, Any]:
        """
        Process a plot request with full quality validation.
        
        Args:
            code: Generated plot code
            data_df: DataFrame with data
            x_col: Column for x-axis
            y_col: Optional column for y-axis
            hue_col: Optional grouping column
            
        Returns:
            Dict with processing results
        """
        results = {
            'original_code': code,
            'validation': None,
            'data_validation': None,
            'chart_recommendation': None,
            'quality_assessment': None,
            'improved_code': code,
            'success': True,
            'issues': [],
            'suggestions': []
        }
        
        # Step 1: Validate plot code
        validation = self.validator.validate_plot_code(code, data_df)
        results['validation'] = validation
        results['issues'].extend(validation['issues'])
        results['suggestions'].extend(validation['suggestions'])
        
        # Step 2: Validate data
        data_validation = self.data_validator.validate_plot_data(data_df, x_col, y_col, hue_col)
        results['data_validation'] = data_validation
        results['issues'].extend(data_validation['warnings'])
        results['suggestions'].extend(data_validation['suggestions'])
        
        # Step 3: Get chart recommendations
        chart_recommendation = self.chart_recommender.recommend_chart_type(data_df, x_col, y_col, hue_col)
        results['chart_recommendation'] = chart_recommendation
        
        # Step 4: Improve code if needed
        if not validation['is_valid']:
            results['improved_code'] = validation['fixed_code']
        
        # Step 5: Assess quality (if plot was created)
        # This would be called after the plot is actually created
        
        return results
    
    def assess_final_plot(self, fig: plt.Figure, data_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess the quality of a final plot.
        
        Args:
            fig: Generated matplotlib figure
            data_df: DataFrame used in the plot
            
        Returns:
            Dict with quality assessment
        """
        return self.quality_assessor.assess_plot_quality(fig, data_df)
    
    def get_quality_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive quality report.
        
        Args:
            results: Results from process_plot_request
            
        Returns:
            Formatted quality report string
        """
        report = "📊 **Plot Quality Report**\n\n"
        
        # Validation summary
        if results['validation']:
            report += f"**Code Validation:** {'✅ Passed' if results['validation']['is_valid'] else '❌ Issues Found'}\n"
            if results['validation']['issues']:
                report += f"- Issues: {len(results['validation']['issues'])}\n"
                report += f"- Suggestions: {len(results['validation']['suggestions'])}\n"
        
        # Data validation summary
        if results['data_validation']:
            quality_score = results['data_validation']['quality_score']
            report += f"**Data Quality:** {quality_score:.1%}\n"
            if results['data_validation']['warnings']:
                report += f"- Warnings: {len(results['data_validation']['warnings'])}\n"
        
        # Chart recommendation
        if results['chart_recommendation']:
            confidence = results['chart_recommendation']['confidence_score']
            report += f"**Chart Recommendation:** {results['chart_recommendation']['primary_chart']} (Confidence: {confidence:.1%})\n"
        
        # Overall assessment
        if results['issues']:
            report += f"\n**Issues Found:** {len(results['issues'])}\n"
            for issue in results['issues'][:3]:  # Show first 3 issues
                report += f"- {issue}\n"
        
        if results['suggestions']:
            report += f"\n**Suggestions:** {len(results['suggestions'])}\n"
            for suggestion in results['suggestions'][:3]:  # Show first 3 suggestions
                report += f"- {suggestion}\n"
        
        return report 