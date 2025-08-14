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
import plotly.graph_objects as go

from agents.excel_agents import SheetPlan, ColumnIndexerAgent
from agents.data_analysis import smart_date_parser
from utils.excel_error_handling import ExcelErrorHandler, ExcelErrorType
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
        
        # Add all DataFrames from the sheet plan with categorical handling
        for sheet_name in sheet_plan.primary_sheets:
            if sheet_name in self.sheet_catalog:
                # Create a copy and handle categorical columns
                df = self.sheet_catalog[sheet_name].copy()
                df = self._prepare_dataframe_for_execution(df, sheet_name)
                env[sheet_name] = df
                logger.info(f"📊 Added DataFrame '{sheet_name}' to environment")
        
        # Add plotting libraries if needed
        env["plt"] = plt
        env["sns"] = sns
        env["io"] = io
        
        # Import helper functions for enhanced plotting
        try:
            # Import from plot_migration_shims for Plotly-based functions
            from utils.plot_migration_shims import (
                create_clean_bar_chart, create_clean_line_chart, create_clean_scatter_plot,
                create_clean_histogram, create_clean_box_plot, create_clean_pie_chart, 
                create_clean_violin_plot, apply_professional_styling, format_axis_labels,
                get_professional_colors, safe_color_access, create_category_palette, 
                optimize_figure_size, add_value_labels, handle_seaborn_warnings, safe_binning
            )
            # Import remaining functions from plot_helpers that aren't in shims
            from utils.plot_helpers import (
                create_clean_heatmap, smart_categorical_plot, smart_annotate_points
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
            env["create_clean_violin_plot"] = create_clean_violin_plot
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
    
    def _prepare_dataframe_for_execution(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Prepare DataFrame for execution by handling categorical columns."""
        df_prepared = df.copy()
        
        # Handle categorical columns that might cause issues
        categorical_columns = []
        for col in df_prepared.columns:
            if df_prepared[col].dtype.name == 'category':
                categorical_columns.append(col)
                # Convert categorical to string to avoid issues
                df_prepared[col] = df_prepared[col].astype(str)
                logger.info(f"🔄 Converted categorical column '{col}' to string in '{sheet_name}'")
        
        if categorical_columns:
            logger.info(f"📊 Prepared {len(categorical_columns)} categorical columns in '{sheet_name}': {categorical_columns}")
        
        return df_prepared
    
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
                    if isinstance(fig, (plt.Figure, plt.Axes, go.Figure)):
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
                elif isinstance(result, (plt.Figure, plt.Axes, go.Figure)):
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
            
            # Handle specific error types
            error_str = str(exc).lower()
            
            # Handle categorical errors
            if "categorical" in error_str and "new category" in error_str:
                # Extract column name from error message
                column_name = self._extract_column_name_from_error(str(exc))
                sheet_name = sheet_plan.primary_sheets[0] if sheet_plan.primary_sheets else "unknown"
                
                categorical_error = self.error_handler.handle_categorical_error(exc, column_name, sheet_name)
                self.error_handler.log_error(categorical_error)
                
                error_msg = f"Categorical data issue: {categorical_error.message}\n\nSuggestion: {categorical_error.recovery_suggestion}"
                
                # Try automatic recovery
                recovered_result = self._attempt_categorical_recovery(code, sheet_plan, column_name)
                if recovered_result is not None:
                    logger.info("✅ Automatic categorical recovery successful")
                    return recovered_result, None
            
            # Handle other specific errors
            elif "not defined" in error_str:
                available_vars = list(env.keys())
                error_msg += f"\n💡 Tip: Available variables are: {available_vars}"
            elif "KeyError" in error_str:
                # Show available columns for the DataFrames
                available_columns = {}
                for sheet_name in sheet_plan.primary_sheets:
                    if sheet_name in self.sheet_catalog:
                        available_columns[sheet_name] = list(self.sheet_catalog[sheet_name].columns)
                error_msg += f"\n💡 Tip: Available columns are: {available_columns}"
            elif "has no attribute" in error_str and any(method in error_str for method in ['idxmax', 'idxmin', 'nlargest', 'nsmallest']):
                error_msg += f"\n💡 Tip: Method chaining error detected. For top N values, use: series.nlargest(n).index, then df.loc[indices]"
            elif "SettingWithCopyWarning" in error_str:
                error_msg += f"\n💡 Tip: Use .copy() when creating DataFrame subsets or .loc[] for safe assignments"
            elif "'int' object has no attribute" in error_str:
                error_msg += f"\n💡 Tip: Check your method chaining - you may be calling a method on an integer instead of a pandas object"
            
            # Record error in performance monitor
            self.performance_monitor.record_error(exc)
            
            return None, error_msg
    
    def _extract_column_name_from_error(self, error_message: str) -> str:
        """Extract column name from categorical error message."""
        # Common patterns in categorical error messages
        import re
        
        # Pattern for "Cannot setitem on a Categorical with a new category (Unknown), set the categories first"
        pattern = r"Cannot setitem on a Categorical with a new category \((\w+)\)"
        match = re.search(pattern, error_message)
        if match:
            return match.group(1)
        
        # Try to extract from other patterns
        patterns = [
            r"column '(\w+)'",
            r"in column '(\w+)'",
            r"for column '(\w+)'"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_message)
            if match:
                return match.group(1)
        
        return "unknown_column"
    
    def _attempt_categorical_recovery(self, code: str, sheet_plan: SheetPlan, column_name: str) -> Optional[Any]:
        """Attempt to recover from categorical errors by modifying the code."""
        try:
            logger.info(f"🔄 Attempting categorical recovery for column '{column_name}'")
            
            # Create a modified version of the code that handles categorical columns
            modified_code = self._modify_code_for_categorical_recovery(code, column_name)
            
            if modified_code != code:
                logger.info("🔄 Using modified code for categorical recovery")
                
                # Execute the modified code
                env = self.create_execution_environment(sheet_plan)
                exec(modified_code, env, env)
                result = env.get("result", None)
                
                if result is not None:
                    logger.info("✅ Categorical recovery successful")
                    return result
            
        except Exception as recovery_error:
            logger.warning(f"⚠️ Categorical recovery failed: {recovery_error}")
        
        return None
    
    def _modify_code_for_categorical_recovery(self, code: str, column_name: str) -> str:
        """Modify code to handle categorical columns properly."""
        # Add categorical handling before the main code
        categorical_handling = f"""
# Handle categorical columns to prevent issues
for df_name in ['Active_Employees', 'Inactive_Employees']:
    if df_name in locals():
        df = locals()[df_name]
        for col in df.columns:
            if df[col].dtype.name == 'category':
                df[col] = df[col].astype(str)
                print(f"Converted categorical column {{col}} to string in {{df_name}}")

"""
        
        return categorical_handling + code
    
    def validate_execution_environment(self, sheet_plan: SheetPlan) -> Tuple[bool, List[str]]:
        """Validate that the execution environment is properly set up."""
        validation_errors = []
        
        # Check if all required DataFrames are available
        for sheet_name in sheet_plan.primary_sheets:
            if sheet_name not in self.sheet_catalog:
                validation_errors.append(f"Required DataFrame '{sheet_name}' not found in sheet catalog")
        
        # Check for categorical columns that might cause issues
        for sheet_name in sheet_plan.primary_sheets:
            if sheet_name in self.sheet_catalog:
                df = self.sheet_catalog[sheet_name]
                categorical_columns = [col for col in df.columns if df[col].dtype.name == 'category']
                if categorical_columns:
                    validation_errors.append(f"Found categorical columns in '{sheet_name}': {categorical_columns}")
        
        return len(validation_errors) == 0, validation_errors
    
    def get_execution_summary(self, sheet_plan: SheetPlan) -> Dict[str, Any]:
        """Get summary of execution environment and performance."""
        validation_passed, validation_errors = self.validate_execution_environment(sheet_plan)
        
        return {
            "validation_passed": validation_passed,
            "validation_errors": validation_errors,
            "sheets_loaded": len(sheet_plan.primary_sheets),
            "total_rows": sum(len(self.sheet_catalog[sheet]) for sheet in sheet_plan.primary_sheets if sheet in self.sheet_catalog),
            "performance_metrics": self.performance_monitor.get_summary()
        } 