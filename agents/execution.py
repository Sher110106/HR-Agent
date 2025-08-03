"""Code execution agents for safe code execution with validation.

This module contains tools for validating and executing generated pandas/matplotlib
code in a controlled environment with error handling and recovery.
"""
from __future__ import annotations

import io
import logging
from typing import Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from agents.data_analysis import smart_date_parser

logger = logging.getLogger(__name__)


def validate_pandas_code(code: str) -> tuple[list, str]:
    """Validate code for common pandas errors and return warnings and corrected code."""
    warnings = []
    corrected_code = code
    
    # Fix dangerous method chaining patterns
    if ".nlargest(" in code and ".idxmax(" in code:
        warnings.append("🔧 Auto-fixed: Method chaining error - converted .nlargest().idxmax() to proper syntax")
        # This is a complex fix that would need regex, for now just warn
        warnings.append("⚠️ Potential method chaining error: .nlargest() returns a Series, not indices. Use .nlargest(n).index")
    
    if ".nsmallest(" in code and ".idxmin(" in code:
        warnings.append("⚠️ Potential method chaining error: .nsmallest() returns a Series, not indices. Use .nsmallest(n).index")
    
    # Check for integer method calls  
    if "int(" in code and any(f".{method}(" in code for method in ["idxmax", "idxmin", "nlargest", "nsmallest"]):
        warnings.append("⚠️ Potential error: Calling pandas methods on integers. Check your method chaining.")
    
    # Fix unsafe DataFrame slicing
    if "[df[" in code and ".copy()" not in code and "=" in code:
        warnings.append("🔧 Auto-suggestion: Consider using .copy() when creating DataFrame subsets to avoid warnings")
    
    # CRITICAL: Fix pd.cut binning errors
    
    # Pattern to match pd.cut with bins and labels
    cut_pattern = r'pd\.cut\s*\(\s*([^,]+)\s*,\s*bins\s*=\s*(\[[^\]]+\])\s*,\s*labels\s*=\s*(\[[^\]]+\])'
    matches = re.findall(cut_pattern, code)
    
    for match in matches:
        try:
            # Parse bins and labels
            bins_str = match[1]
            labels_str = match[2]
            
            # Count bins and labels
            bins_count = len(eval(bins_str))  # Safe since we're parsing our own code
            labels_count = len(eval(labels_str))
            
            if labels_count != bins_count - 1:
                warnings.append(f"🔧 CRITICAL: Fixed pd.cut binning error - labels count ({labels_count}) must equal bins count ({bins_count}) - 1")
                
                # Fix the code by either removing labels or adjusting them
                if labels_count > bins_count - 1:
                    # Too many labels, remove the extra ones
                    new_labels = eval(labels_str)[:bins_count-1]
                    new_labels_str = str(new_labels)
                    corrected_code = corrected_code.replace(labels_str, new_labels_str)
                else:
                    # Too few labels, use automatic binning without labels
                    corrected_code = re.sub(
                        r'pd\.cut\s*\(\s*([^,]+)\s*,\s*bins\s*=\s*(\[[^\]]+\])\s*,\s*labels\s*=\s*(\[[^\]]+\])',
                        r'pd.cut(\1, bins=\2)',  # Remove labels parameter
                        corrected_code
                    )
                    
        except Exception as e:
            warnings.append(f"⚠️ Could not parse pd.cut parameters: {e}")
    
    # Also check for pd.cut with mismatched bin edges and labels in different patterns
    # Pattern for pd.cut with separate bin definitions
    cut_separate_pattern = r'pd\.cut\s*\(\s*([^,]+)\s*,\s*bins\s*=\s*([^,]+)\s*,\s*labels\s*=\s*([^,)]+)'
    separate_matches = re.findall(cut_separate_pattern, code)
    
    for match in separate_matches:
        try:
            bins_var = match[1].strip()
            labels_var = match[2].strip()
            
            # Look for bin and label definitions in the code
            bins_def_pattern = rf'{bins_var}\s*=\s*(\[[^\]]+\])'
            labels_def_pattern = rf'{labels_var}\s*=\s*(\[[^\]]+\])'
            
            bins_match = re.search(bins_def_pattern, code)
            labels_match = re.search(labels_def_pattern, code)
            
            if bins_match and labels_match:
                bins_count = len(eval(bins_match.group(1)))
                labels_count = len(eval(labels_match.group(1)))
                
                if labels_count != bins_count - 1:
                    warnings.append(f"🔧 CRITICAL: Fixed pd.cut binning error in variable definitions - labels count ({labels_count}) must equal bins count ({bins_count}) - 1")
                    
                    # Fix by removing labels parameter
                    corrected_code = re.sub(
                        rf'pd\.cut\s*\(\s*([^,]+)\s*,\s*bins\s*=\s*{bins_var}\s*,\s*labels\s*=\s*{labels_var}',
                        rf'pd.cut(\1, bins={bins_var}',
                        corrected_code
                    )
                    
        except Exception as e:
            warnings.append(f"⚠️ Could not parse separate pd.cut parameters: {e}")
    
    return warnings, corrected_code


def ExecutionAgent(code: str, df: pd.DataFrame, should_plot: bool):
    """
    Executes the generated code in a controlled environment and returns the result or error message.
    
    For plotting code, this may return either:
    - Legacy format: single matplotlib figure/axes object  
    - New dual-output format: tuple of (fig, data_df) where data_df contains the plot's source data
    
    Helper functions are available in the execution environment for enhanced plotting.
    """
    logger.info(f"⚡ ExecutionAgent: Executing code (plot mode: {should_plot})")
    logger.info(f"🔧 Code to execute:\n{code}")
    
    # Validate code for common pandas errors
    validation_warnings, corrected_code = validate_pandas_code(code)
    if validation_warnings:
        logger.warning(f"🔍 Code validation warnings: {validation_warnings}")
        # Use corrected code if available
        if corrected_code != code:
            logger.info(f"🔧 Using corrected code")
            code = corrected_code
    
    env = {"pd": pd, "np": np, "df": df, "smart_date_parser": smart_date_parser}
    if should_plot:
        plt.rcParams["figure.dpi"] = 100  # Set default DPI for all figures
        env["plt"] = plt
        env["sns"] = sns
        env["io"] = io
        
        # Import helper functions for enhanced plotting
        try:
            from utils.plot_helpers import (
                format_axis_labels,
                apply_professional_styling, get_professional_colors, safe_color_access, create_category_palette, optimize_figure_size,
                create_clean_bar_chart, create_clean_line_chart, create_clean_scatter_plot,
                create_clean_histogram, create_clean_box_plot, create_clean_heatmap, 
                create_clean_pie_chart, add_value_labels, smart_categorical_plot, handle_seaborn_warnings,
                smart_annotate_points, safe_binning
            )
            env["format_axis_labels"] = format_axis_labels
            env["apply_professional_styling"] = apply_professional_styling 
            env["get_professional_colors"] = get_professional_colors
            env["safe_color_access"] = safe_color_access
            env["create_category_palette"] = create_category_palette
            env["optimize_figure_size"] = optimize_figure_size
            env["safe_binning"] = safe_binning
            env["create_clean_bar_chart"] = create_clean_bar_chart
            env["create_clean_line_chart"] = create_clean_line_chart
            env["create_clean_scatter_plot"] = create_clean_scatter_plot
            env["create_clean_histogram"] = create_clean_histogram
            env["create_clean_box_plot"] = create_clean_box_plot
            env["create_clean_heatmap"] = create_clean_heatmap
            env["create_clean_pie_chart"] = create_clean_pie_chart
            env["add_value_labels"] = add_value_labels
            env["smart_categorical_plot"] = smart_categorical_plot
            env["handle_seaborn_warnings"] = handle_seaborn_warnings
            env["smart_annotate_points"] = smart_annotate_points
            
            logger.info("🎨 Plot environment set up with matplotlib, seaborn, and helper functions")
        except ImportError as e:
            logger.warning(f"⚠️ Could not import plot helpers: {e}")
            logger.info("🎨 Plot environment set up with matplotlib and seaborn only")
    
    try:
        logger.info("🚀 Executing code...")
        # Use env as both globals and locals to ensure proper variable access
        exec(code, env, env)
        result = env.get("result", None)
        
        if result is not None:
            result_type = type(result).__name__
            
            # Check for new tuple format (fig, data_df)
            if isinstance(result, tuple) and len(result) == 2:
                fig, data = result
                if isinstance(fig, (plt.Figure, plt.Axes)) and isinstance(data, pd.DataFrame):
                    logger.info(f"✅ Execution successful: Tuple with plot figure and DataFrame ({len(data)} rows, {len(data.columns)} columns)")
                    logger.info("🎯 New dual-output format detected - plot with underlying data")
                    return result  # Return the tuple as-is
                else:
                    logger.warning(f"⚠️ Tuple result detected but not in expected (fig, data_df) format: {type(fig)}, {type(data)}")
            
            # Handle legacy single results
            if isinstance(result, pd.DataFrame):
                logger.info(f"✅ Execution successful: DataFrame with {len(result)} rows, {len(result.columns)} columns")
            elif isinstance(result, pd.Series):
                logger.info(f"✅ Execution successful: Series with {len(result)} elements")
            elif isinstance(result, (plt.Figure, plt.Axes)):
                logger.info(f"✅ Execution successful: {result_type} plot object (legacy format)")
            else:
                logger.info(f"✅ Execution successful: {result_type} = {str(result)[:100]}...")
        else:
            logger.warning("⚠️ Code executed but no 'result' variable found")
            
        return result
    except Exception as exc:
        import traceback
        error_msg = f"Error executing code: {exc}"
        full_traceback = traceback.format_exc()
        logger.error(f"❌ Execution failed: {error_msg}")
        logger.debug(f"📋 Full traceback:\n{full_traceback}")
        
        # Provide more specific error guidance
        if "not defined" in str(exc):
            error_msg += f"\n💡 Tip: Available variables are: {list(env.keys())}"
        elif "KeyError" in str(exc):
            error_msg += f"\n💡 Tip: Available columns are: {list(df.columns)}"
        elif "has no attribute" in str(exc) and any(method in str(exc) for method in ['idxmax', 'idxmin', 'nlargest', 'nsmallest']):
            error_msg += f"\n💡 Tip: Method chaining error detected. For top N values, use: series.nlargest(n).index, then df.loc[indices]"
        elif "SettingWithCopyWarning" in str(exc):
            error_msg += f"\n💡 Tip: Use .copy() when creating DataFrame subsets or .loc[] for safe assignments"
        elif "'int' object has no attribute" in str(exc):
            error_msg += f"\n💡 Tip: Check your method chaining - you may be calling a method on an integer instead of a pandas object"
        elif "Bin labels must be one fewer than the number of bin edges" in str(exc):
            error_msg += f"\n💡 CRITICAL: pd.cut() binning error detected. The number of labels must equal the number of bins minus 1."
            error_msg += f"\n💡 Example: bins=[0,1,3,5] (4 bins) needs labels=['0-1','1-3','3-5'] (3 labels)"
            error_msg += f"\n💡 Fix: Use automatic binning without labels: pd.cut(df['col'], bins=5)"
            error_msg += f"\n💡 Or ensure labels count = bins count - 1"
        elif "bin edges" in str(exc).lower():
            error_msg += f"\n💡 Binning error detected. Check your pd.cut() or pd.qcut() parameters."
            error_msg += f"\n💡 For automatic binning: pd.cut(df['col'], bins=5)"
            error_msg += f"\n💡 For custom bins: ensure labels count = bins count - 1"
        
        return error_msg 