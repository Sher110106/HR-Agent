"""Enhanced Execution Agent for multi-sheet Excel analysis.

This module provides specialized execution capabilities for handling
multi-sheet Excel data with proper DataFrame management and error handling.
Enhanced with Phase 3: Resilience & Polish features.
"""

import logging
import io
import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from agents.excel_agents import SheetPlan, ColumnIndexerAgent
from agents.data_analysis import smart_date_parser
from utils.excel_error_handling import ExcelErrorHandler
from utils.excel_performance import PerformanceMonitor, performance_decorator

logger = logging.getLogger(__name__)


class ExcelExecutionAgent:
    """Enhanced execution agent for multi-sheet Excel analysis."""
    
    def __init__(self, column_indexer_agent: ColumnIndexerAgent):
        self.column_indexer_agent = column_indexer_agent
        self.sheet_catalog = column_indexer_agent.sheet_catalog
        self.error_handler = ExcelErrorHandler()
        self.performance_monitor = PerformanceMonitor()
        
    def create_execution_environment(self, sheet_plan: SheetPlan) -> Dict[str, Any]:
        """Create the execution environment with all required DataFrames."""
        logger.info(f"🔧 Creating execution environment for plan: {sheet_plan}")
        
        env = {
            "pd": pd,
            "np": np,
            "smart_date_parser": smart_date_parser
        }
        
        # Add all DataFrames from the sheet plan
        for sheet_name in sheet_plan.primary_sheets:
            if sheet_name in self.sheet_catalog:
                env[sheet_name] = self.sheet_catalog[sheet_name].copy()
                logger.info(f"📊 Added DataFrame '{sheet_name}' to environment")
        
        # Add plotting libraries if needed
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
                smart_annotate_points
            )
            env["format_axis_labels"] = format_axis_labels
            env["apply_professional_styling"] = apply_professional_styling 
            env["get_professional_colors"] = get_professional_colors
            env["safe_color_access"] = safe_color_access
            env["create_category_palette"] = create_category_palette
            env["optimize_figure_size"] = optimize_figure_size
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
        
        # Set matplotlib parameters
        plt.rcParams["figure.dpi"] = 100
        
        return env
    
    @performance_decorator
    def execute_code(self, code: str, sheet_plan: SheetPlan) -> Tuple[Any, str]:
        """
        Execute the generated code in a controlled environment.
        Enhanced with performance monitoring and error handling.
        
        Args:
            code: The pandas code to execute
            sheet_plan: The sheet plan being executed
            
        Returns:
            Tuple of (result, error_message) where error_message is None if successful
        """
        logger.info(f"⚡ Executing code for plan: {sheet_plan}")
        logger.debug(f"🔧 Code to execute:\n{code}")
        
        start_time = time.time()
        
        # Create execution environment
        env = self.create_execution_environment(sheet_plan)
        
        try:
            logger.info("🚀 Executing code...")
            # Use env as both globals and locals to ensure proper variable access
            exec(code, env, env)
            result = env.get("result", None)
            
            if result is not None:
                result_type = type(result).__name__
                
                # Check for new tuple format (fig, data_df) or (fig, dict_of_dataframes)
                if isinstance(result, tuple) and len(result) == 2:
                    fig, data = result
                    if isinstance(fig, (plt.Figure, plt.Axes)):
                        if isinstance(data, pd.DataFrame):
                            logger.info(f"✅ Execution successful: Tuple with plot figure and DataFrame ({len(data)} rows, {len(data.columns)} columns)")
                            logger.info("🎯 New dual-output format detected - plot with underlying data")
                        elif isinstance(data, dict):
                            # Handle dictionary of DataFrames
                            df_count = sum(1 for v in data.values() if isinstance(v, pd.DataFrame))
                            total_rows = sum(len(v) for v in data.values() if isinstance(v, pd.DataFrame))
                            logger.info(f"✅ Execution successful: Tuple with plot figure and dictionary of {df_count} DataFrames ({total_rows} total rows)")
                            logger.info("🎯 New dual-output format detected - plot with multiple underlying datasets")
                        else:
                            logger.warning(f"⚠️ Tuple result detected but not in expected (fig, data_df) format: {type(fig)}, {type(data)}")
                        
                        # Record performance metrics
                        execution_time = time.time() - start_time
                        self.performance_monitor.record_query_time("execute_code", execution_time)
                        self.performance_monitor.record_memory_usage()
                        
                        return result, None
                    else:
                        logger.warning(f"⚠️ Tuple result detected but not in expected (fig, data_df) format: {type(fig)}, {type(data)}")
                
                # Handle legacy single results
                if isinstance(result, pd.DataFrame):
                    logger.info(f"✅ Execution successful: DataFrame with {len(result)} rows, {len(result.columns)} columns")
                elif isinstance(result, pd.Series):
                    logger.info(f"✅ Execution successful: Series with {len(result)} elements")
                elif isinstance(result, (plt.Figure, plt.Axes)):
                    logger.info(f"✅ Execution successful: {result_type} plot object (legacy format)")
                elif isinstance(result, (int, float, str, bool)) and result == 0:
                    # Handle case where result is 0 (likely an error in code generation)
                    logger.warning(f"⚠️ Result is 0 - likely missing proper result assignment in generated code")
                    error_msg = "Code executed but returned 0. This usually means the analysis didn't assign a proper result. Please check the generated code."
                    return None, error_msg
                else:
                    logger.info(f"✅ Execution successful: {result_type} = {str(result)[:100]}...")
            else:
                logger.warning("⚠️ Code executed but no 'result' variable found")
                error_msg = "Code executed but no 'result' variable was assigned. Please ensure your analysis assigns the final result to the 'result' variable."
                return None, error_msg
            
            # Record performance metrics
            execution_time = time.time() - start_time
            self.performance_monitor.record_query_time("execute_code", execution_time)
            self.performance_monitor.record_memory_usage()
                
            return result, None
            
        except Exception as exc:
            import traceback
            error_msg = f"Error executing code: {exc}"
            full_traceback = traceback.format_exc()
            logger.error(f"❌ Execution failed: {error_msg}")
            logger.debug(f"📋 Full traceback:\n{full_traceback}")
            
            # Record error in performance monitor
            self.performance_monitor.record_error(exc)
            
            # Provide more specific error guidance
            if "not defined" in str(exc):
                available_vars = list(env.keys())
                error_msg += f"\n💡 Tip: Available variables are: {available_vars}"
            elif "KeyError" in str(exc):
                # Show available columns for the DataFrames
                available_columns = {}
                for sheet_name in sheet_plan.primary_sheets:
                    if sheet_name in self.sheet_catalog:
                        available_columns[sheet_name] = list(self.sheet_catalog[sheet_name].columns)
                error_msg += f"\n💡 Tip: Available columns are: {available_columns}"
            elif "has no attribute" in str(exc) and any(method in str(exc) for method in ['idxmax', 'idxmin', 'nlargest', 'nsmallest']):
                error_msg += f"\n💡 Tip: Method chaining error detected. For top N values, use: series.nlargest(n).index, then df.loc[indices]"
            elif "SettingWithCopyWarning" in str(exc):
                error_msg += f"\n💡 Tip: Use .copy() when creating DataFrame subsets or .loc[] for safe assignments"
            elif "'int' object has no attribute" in str(exc):
                error_msg += f"\n💡 Tip: Check your method chaining - you may be calling a method on an integer instead of a pandas object"
            elif "merge" in str(exc).lower():
                error_msg += f"\n💡 Tip: Check that the join keys exist in both DataFrames and have compatible data types"
            elif "concat" in str(exc).lower():
                error_msg += f"\n💡 Tip: Check that the DataFrames have compatible column structures for concatenation"
            
            return None, error_msg
    
    def validate_execution_environment(self, sheet_plan: SheetPlan) -> Tuple[bool, List[str]]:
        """Validate that the execution environment can be created successfully."""
        errors = []
        
        # Check if all required sheets exist
        for sheet_name in sheet_plan.primary_sheets:
            if sheet_name not in self.sheet_catalog:
                errors.append(f"Required sheet '{sheet_name}' not found in catalog")
        
        # Check if sheets have data
        for sheet_name in sheet_plan.primary_sheets:
            if sheet_name in self.sheet_catalog:
                df = self.sheet_catalog[sheet_name]
                if df.empty:
                    errors.append(f"Sheet '{sheet_name}' is empty")
        
        # Check join keys if using join strategy
        if sheet_plan.join_strategy == 'join' and sheet_plan.join_keys:
            for key in sheet_plan.join_keys:
                refs = self.column_indexer_agent.get_column_refs(key)
                if not refs:
                    errors.append(f"Join key '{key}' not found in any sheet")
                elif len(refs) < 2:
                    errors.append(f"Join key '{key}' only found in {len(refs)} sheet(s), need at least 2")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def get_execution_summary(self, sheet_plan: SheetPlan) -> Dict[str, Any]:
        """Get a summary of the execution environment."""
        summary = {
            'sheets': [],
            'total_rows': 0,
            'total_columns': 0,
            'join_strategy': sheet_plan.join_strategy,
            'join_keys': sheet_plan.join_keys
        }
        
        for sheet_name in sheet_plan.primary_sheets:
            if sheet_name in self.sheet_catalog:
                df = self.sheet_catalog[sheet_name]
                sheet_info = {
                    'name': sheet_name,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': list(df.columns)
                }
                summary['sheets'].append(sheet_info)
                summary['total_rows'] += len(df)
                summary['total_columns'] += len(df.columns)
        
        return summary 