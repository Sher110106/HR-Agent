"""
Advanced query engine for Excel operations.

This module provides sophisticated query capabilities for multi-sheet Excel analysis,
including complex joins, aggregations, time-series analysis, and data validation.
"""

import logging
import pandas as pd
import numpy as np
import io
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import re

from .excel_error_handling import ExcelErrorHandler, DataTypeConverter, ExcelError, ExcelErrorType

logger = logging.getLogger(__name__)


class ExcelQueryEngine:
    """Advanced query engine for multi-sheet Excel analysis."""
    
    def __init__(self, sheet_catalog: Dict[str, pd.DataFrame], error_handler: ExcelErrorHandler):
        self.sheet_catalog = sheet_catalog
        self.error_handler = error_handler
        self.data_converter = DataTypeConverter()
        self.query_cache = {}
    
    def execute_complex_join(self, 
                           primary_sheets: List[str], 
                           join_keys: List[str], 
                           join_type: str = 'inner',
                           additional_columns: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Execute complex multi-sheet joins with validation and error handling.
        
        Args:
            primary_sheets: List of sheet names to join
            join_keys: List of columns to join on
            join_type: Type of join ('inner', 'left', 'right', 'outer')
            additional_columns: Optional mapping for status labels
            
        Returns:
            Tuple of (joined DataFrame, list of warnings)
        """
        warnings = []
        
        if len(primary_sheets) < 2:
            raise ValueError("At least 2 sheets required for join operation")
        
        # Validate sheets exist
        for sheet in primary_sheets:
            if sheet not in self.sheet_catalog:
                raise ValueError(f"Sheet '{sheet}' not found in catalog")
        
        # Start with the first sheet
        result_df = self.sheet_catalog[primary_sheets[0]].copy()
        
        # Add status column if specified
        if additional_columns and primary_sheets[0] in additional_columns:
            result_df[additional_columns[primary_sheets[0]]] = primary_sheets[0]
        
        # Join with remaining sheets
        for i, sheet_name in enumerate(primary_sheets[1:], 1):
            current_df = self.sheet_catalog[sheet_name].copy()
            
            # Add status column if specified
            if additional_columns and sheet_name in additional_columns:
                current_df[additional_columns[sheet_name]] = sheet_name
            
            # Validate join compatibility
            for join_key in join_keys:
                if join_key not in result_df.columns or join_key not in current_df.columns:
                    error = self.error_handler.handle_join_error(
                        join_keys, [primary_sheets[0], sheet_name], 
                        Exception(f"Join key '{join_key}' missing")
                    )
                    self.error_handler.log_error(error)
                    raise ValueError(f"Join key '{join_key}' not found in both sheets")
                
                # Check data type compatibility
                compatible, message = self.data_converter.validate_join_compatibility(
                    result_df, current_df, join_key
                )
                if not compatible:
                    warnings.append(f"Data type warning for {join_key}: {message}")
            
            # Perform the join
            try:
                if join_type == 'inner':
                    result_df = result_df.merge(current_df, on=join_keys, how='inner')
                elif join_type == 'left':
                    result_df = result_df.merge(current_df, on=join_keys, how='left')
                elif join_type == 'right':
                    result_df = result_df.merge(current_df, on=join_keys, how='right')
                elif join_type == 'outer':
                    result_df = result_df.merge(current_df, on=join_keys, how='outer')
                else:
                    raise ValueError(f"Unsupported join type: {join_type}")
                
                logger.info(f"Successfully joined {sheet_name} ({len(current_df)} rows) -> {len(result_df)} total rows")
                
            except Exception as e:
                error = self.error_handler.handle_join_error(join_keys, [primary_sheets[0], sheet_name], e)
                self.error_handler.log_error(error)
                raise
        
        return result_df, warnings
    
    def execute_union_operation(self, 
                              primary_sheets: List[str], 
                              union_columns: Optional[List[str]] = None,
                              additional_columns: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Execute union operation across multiple sheets.
        
        Args:
            primary_sheets: List of sheet names to union
            union_columns: Optional list of columns to include (if None, uses common columns)
            additional_columns: Optional mapping for status labels
            
        Returns:
            Tuple of (unioned DataFrame, list of warnings)
        """
        warnings = []
        
        if len(primary_sheets) < 2:
            raise ValueError("At least 2 sheets required for union operation")
        
        # Validate sheets exist
        for sheet in primary_sheets:
            if sheet not in self.sheet_catalog:
                raise ValueError(f"Sheet '{sheet}' not found in catalog")
        
        # Determine columns to use
        if union_columns is None:
            # Find common columns across all sheets
            common_columns = set(self.sheet_catalog[primary_sheets[0]].columns)
            for sheet in primary_sheets[1:]:
                common_columns = common_columns.intersection(set(self.sheet_catalog[sheet].columns))
            
            if not common_columns:
                raise ValueError("No common columns found across sheets for union operation")
            
            union_columns = list(common_columns)
            warnings.append(f"Using common columns: {', '.join(union_columns)}")
        
        # Prepare DataFrames for union
        union_dfs = []
        for sheet_name in primary_sheets:
            df = self.sheet_catalog[sheet_name][union_columns].copy()
            
            # Add status column if specified
            if additional_columns and sheet_name in additional_columns:
                df[additional_columns[sheet_name]] = sheet_name
            
            union_dfs.append(df)
        
        # Perform union
        try:
            result_df = pd.concat(union_dfs, ignore_index=True, sort=False)
            logger.info(f"Successfully unioned {len(primary_sheets)} sheets -> {len(result_df)} total rows")
            
        except Exception as e:
            error = ExcelError(
                error_type=ExcelErrorType.EXECUTION_ERROR,
                message="Failed to union sheets",
                details=str(e),
                recovery_suggestion="Check that all sheets have compatible column structures",
                severity="high",
                affected_sheets=primary_sheets
            )
            self.error_handler.log_error(error)
            raise
        
        return result_df, warnings
    
    def execute_aggregation_query(self, 
                                df: pd.DataFrame, 
                                group_by_columns: List[str], 
                                agg_functions: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Execute aggregation queries with multiple functions.
        
        Args:
            df: Input DataFrame
            group_by_columns: Columns to group by
            agg_functions: Dictionary mapping column names to list of aggregation functions
            
        Returns:
            Aggregated DataFrame
        """
        try:
            # Validate columns exist
            for col in group_by_columns:
                if col not in df.columns:
                    raise ValueError(f"Group by column '{col}' not found")
            
            for col in agg_functions.keys():
                if col not in df.columns:
                    raise ValueError(f"Aggregation column '{col}' not found")
            
            # Perform aggregation
            result = df.groupby(group_by_columns).agg(agg_functions).reset_index()
            
            # Flatten column names if multi-level
            if isinstance(result.columns, pd.MultiIndex):
                result.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in result.columns]
            
            logger.info(f"Aggregation completed: {len(result)} groups")
            return result
            
        except Exception as e:
            error = ExcelError(
                error_type=ExcelErrorType.EXECUTION_ERROR,
                message="Failed to execute aggregation query",
                details=str(e),
                recovery_suggestion="Check column names and aggregation functions",
                severity="medium"
            )
            self.error_handler.log_error(error)
            raise
    
    def execute_time_series_analysis(self, 
                                   df: pd.DataFrame, 
                                   time_column: str, 
                                   value_columns: List[str],
                                   freq: str = 'D') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute time series analysis with resampling and trend detection.
        
        Args:
            df: Input DataFrame
            time_column: Column containing datetime values
            value_columns: Columns to analyze
            freq: Resampling frequency ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            Tuple of (resampled DataFrame, analysis results)
        """
        try:
            # Ensure time column is datetime
            if time_column not in df.columns:
                raise ValueError(f"Time column '{time_column}' not found")
            
            df_copy = df.copy()
            df_copy[time_column] = pd.to_datetime(df_copy[time_column], errors='coerce')
            
            # Remove rows with invalid dates
            invalid_dates = df_copy[time_column].isna()
            if invalid_dates.any():
                logger.warning(f"Removed {invalid_dates.sum()} rows with invalid dates")
                df_copy = df_copy.dropna(subset=[time_column])
            
            # Set time column as index
            df_copy = df_copy.set_index(time_column).sort_index()
            
            # Validate value columns
            for col in value_columns:
                if col not in df_copy.columns:
                    raise ValueError(f"Value column '{col}' not found")
            
            # Perform resampling
            resampled_data = {}
            analysis_results = {}
            
            for col in value_columns:
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    # Resample numeric columns
                    resampled = df_copy[col].resample(freq).agg(['mean', 'sum', 'count', 'std'])
                    resampled_data[col] = resampled
                    
                    # Calculate trends
                    if len(resampled) > 1:
                        trend = np.polyfit(range(len(resampled)), resampled['mean'].fillna(0), 1)[0]
                        analysis_results[col] = {
                            'trend_slope': trend,
                            'trend_direction': 'increasing' if trend > 0 else 'decreasing' if trend < 0 else 'stable',
                            'mean_value': resampled['mean'].mean(),
                            'total_sum': resampled['sum'].sum(),
                            'data_points': resampled['count'].sum()
                        }
            
            # Combine results
            if resampled_data:
                result_df = pd.concat(resampled_data.values(), axis=1, keys=resampled_data.keys())
                # Flatten column names
                if isinstance(result_df.columns, pd.MultiIndex):
                    result_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in result_df.columns]
                result_df = result_df.reset_index()
            else:
                result_df = pd.DataFrame()
            
            logger.info(f"Time series analysis completed for {len(value_columns)} columns")
            return result_df, analysis_results
            
        except Exception as e:
            error = ExcelError(
                error_type=ExcelErrorType.EXECUTION_ERROR,
                message="Failed to execute time series analysis",
                details=str(e),
                recovery_suggestion="Check time column format and value column data types",
                severity="medium"
            )
            self.error_handler.log_error(error)
            raise
    
    def validate_data_quality(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """
        Perform comprehensive data quality validation.
        
        Args:
            df: DataFrame to validate
            sheet_name: Name of the sheet for reporting
            
        Returns:
            Dictionary containing quality metrics and issues
        """
        quality_report = {
            'sheet_name': sheet_name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'null_counts': {},
            'duplicate_rows': 0,
            'data_type_issues': [],
            'outliers': {},
            'quality_score': 0.0
        }
        
        # Check for null values
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_percentage = (null_count / len(df)) * 100
            quality_report['null_counts'][col] = {
                'count': null_count,
                'percentage': null_percentage
            }
        
        # Check for duplicate rows
        quality_report['duplicate_rows'] = df.duplicated().sum()
        
        # Check data types and potential issues
        for col in df.columns:
            col_issues = []
            
            # Check for mixed data types in object columns
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(20)
                if len(sample_values) > 0:
                    # Try to detect mixed types
                    numeric_count = 0
                    date_count = 0
                    
                    for val in sample_values:
                        try:
                            float(val)
                            numeric_count += 1
                        except (ValueError, TypeError):
                            pass
                        
                        try:
                            pd.to_datetime(val)
                            date_count += 1
                        except (ValueError, TypeError):
                            pass
                    
                    if numeric_count > 0 and date_count > 0:
                        col_issues.append("Mixed numeric and date values")
                    elif numeric_count > len(sample_values) * 0.8:
                        col_issues.append("Primarily numeric values in object column")
            
            # Check for outliers in numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                if len(outliers) > 0:
                    quality_report['outliers'][col] = {
                        'count': len(outliers),
                        'percentage': (len(outliers) / len(df)) * 100,
                        'min_outlier': outliers.min(),
                        'max_outlier': outliers.max()
                    }
            
            if col_issues:
                quality_report['data_type_issues'].append({
                    'column': col,
                    'issues': col_issues
                })
        
        # Calculate quality score (0-100)
        score = 100.0
        
        # Deduct for null values
        total_null_percentage = sum(q['percentage'] for q in quality_report['null_counts'].values())
        score -= min(total_null_percentage * 0.5, 30)  # Max 30 points deduction for nulls
        
        # Deduct for duplicates
        duplicate_percentage = (quality_report['duplicate_rows'] / quality_report['total_rows']) * 100
        score -= min(duplicate_percentage * 0.3, 20)  # Max 20 points deduction for duplicates
        
        # Deduct for data type issues
        score -= len(quality_report['data_type_issues']) * 5  # 5 points per issue
        
        # Deduct for outliers
        outlier_penalty = sum(q['percentage'] for q in quality_report['outliers'].values()) * 0.1
        score -= min(outlier_penalty, 15)  # Max 15 points deduction for outliers
        
        quality_report['quality_score'] = max(0.0, score)
        
        return quality_report
    
    def export_results(self, 
                      df: pd.DataFrame, 
                      format: str = 'csv', 
                      filename: str = None) -> Tuple[bytes, str]:
        """
        Export results in various formats.
        
        Args:
            df: DataFrame to export
            format: Export format ('csv', 'excel', 'json')
            filename: Optional filename
            
        Returns:
            Tuple of (file content as bytes, suggested filename)
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"excel_analysis_{timestamp}.{format}"
            
            if format.lower() == 'csv':
                output = io.BytesIO()
                df.to_csv(output, index=False, encoding='utf-8')
                output.seek(0)
                return output.getvalue(), filename
            
            elif format.lower() == 'excel':
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Analysis_Results', index=False)
                output.seek(0)
                return output.getvalue(), filename.replace('.excel', '.xlsx')
            
            elif format.lower() == 'json':
                output = io.BytesIO()
                df.to_json(output, orient='records', indent=2)
                output.seek(0)
                return output.getvalue(), filename
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            error = ExcelError(
                error_type=ExcelErrorType.EXECUTION_ERROR,
                message="Failed to export results",
                details=str(e),
                recovery_suggestion="Try a different export format or check data compatibility",
                severity="medium"
            )
            self.error_handler.log_error(error)
            raise
    
    def get_query_cache_key(self, operation: str, **kwargs) -> str:
        """Generate cache key for query operations."""
        # Create a deterministic key based on operation and parameters
        key_parts = [operation]
        
        # Sort kwargs for deterministic ordering
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (list, tuple)):
                key_parts.append(f"{k}:{','.join(sorted(str(x) for x in v))}")
            else:
                key_parts.append(f"{k}:{v}")
        
        return "|".join(key_parts)
    
    def cache_query_result(self, cache_key: str, result: Any) -> None:
        """Cache query result for future use."""
        self.query_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now(),
            'access_count': 0
        }
    
    def get_cached_result(self, cache_key: str, max_age_minutes: int = 30) -> Optional[Any]:
        """Get cached result if available and not expired."""
        if cache_key not in self.query_cache:
            return None
        
        cache_entry = self.query_cache[cache_key]
        age = datetime.now() - cache_entry['timestamp']
        
        if age.total_seconds() > max_age_minutes * 60:
            # Remove expired cache entry
            del self.query_cache[cache_key]
            return None
        
        # Update access count
        cache_entry['access_count'] += 1
        return cache_entry['result']
    
    def clear_cache(self) -> None:
        """Clear all cached query results."""
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.query_cache:
            return {'total_entries': 0, 'total_accesses': 0}
        
        total_accesses = sum(entry['access_count'] for entry in self.query_cache.values())
        
        return {
            'total_entries': len(self.query_cache),
            'total_accesses': total_accesses,
            'avg_access_per_entry': total_accesses / len(self.query_cache) if self.query_cache else 0
        } 