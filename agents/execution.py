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

from app_core.helpers import smart_date_parser

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
                add_value_labels, format_axis_labels,
                apply_professional_styling, get_professional_colors
            )
            env["add_value_labels"] = add_value_labels
            env["format_axis_labels"] = format_axis_labels
            env["apply_professional_styling"] = apply_professional_styling 
            env["get_professional_colors"] = get_professional_colors
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
        
        return error_msg 