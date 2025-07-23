"""
Enhanced error handling and recovery for Excel operations.

This module provides comprehensive error handling for Excel file processing,
including specific error types, recovery strategies, and user-friendly messages.
"""

import logging
import pandas as pd
import io
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ExcelErrorType(Enum):
    """Types of Excel-related errors."""
    FILE_CORRUPTED = "file_corrupted"
    SHEET_EMPTY = "sheet_empty"
    INCOMPATIBLE_SCHEMA = "incompatible_schema"
    MEMORY_LIMIT = "memory_limit"
    ENCODING_ISSUE = "encoding_issue"
    PERMISSION_DENIED = "permission_denied"
    INVALID_FORMAT = "invalid_format"
    DATA_TYPE_CONFLICT = "data_type_conflict"
    JOIN_KEY_MISSING = "join_key_missing"
    EXECUTION_ERROR = "execution_error"


@dataclass
class ExcelError:
    """Structured error information for Excel operations."""
    error_type: ExcelErrorType
    message: str
    details: str
    recovery_suggestion: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    affected_sheets: Optional[List[str]] = None
    affected_columns: Optional[List[str]] = None


class ExcelErrorHandler:
    """Comprehensive error handler for Excel operations."""
    
    def __init__(self):
        self.error_history: List[ExcelError] = []
        self.recovery_attempts: Dict[str, int] = {}
    
    def handle_file_reading_error(self, error: Exception, filename: str) -> ExcelError:
        """Handle errors during Excel file reading."""
        error_msg = str(error).lower()
        
        if "corrupt" in error_msg or "damaged" in error_msg:
            return ExcelError(
                error_type=ExcelErrorType.FILE_CORRUPTED,
                message="The Excel file appears to be corrupted or damaged",
                details=f"Error: {str(error)}",
                recovery_suggestion="Try opening the file in Excel and saving it again, or use a backup copy",
                severity="critical"
            )
        
        elif "permission" in error_msg or "access" in error_msg:
            return ExcelError(
                error_type=ExcelErrorType.PERMISSION_DENIED,
                message="Cannot access the Excel file due to permission issues",
                details=f"Error: {str(error)}",
                recovery_suggestion="Check file permissions or try copying the file to a different location",
                severity="high"
            )
        
        elif "format" in error_msg or "extension" in error_msg:
            return ExcelError(
                error_type=ExcelErrorType.INVALID_FORMAT,
                message="The file format is not supported",
                details=f"Error: {str(error)}",
                recovery_suggestion="Ensure the file is a valid .xlsx or .xls format",
                severity="medium"
            )
        
        else:
            return ExcelError(
                error_type=ExcelErrorType.INVALID_FORMAT,
                message="Unable to read the Excel file",
                details=f"Error: {str(error)}",
                recovery_suggestion="Try opening the file in Excel and saving it in .xlsx format",
                severity="high"
            )
    
    def handle_sheet_processing_error(self, error: Exception, sheet_name: str) -> ExcelError:
        """Handle errors during individual sheet processing."""
        error_msg = str(error).lower()
        
        if "empty" in error_msg or "no data" in error_msg:
            return ExcelError(
                error_type=ExcelErrorType.SHEET_EMPTY,
                message=f"Sheet '{sheet_name}' is empty or contains no data",
                details=f"Error: {str(error)}",
                recovery_suggestion="Check if the sheet contains data or skip this sheet",
                severity="low",
                affected_sheets=[sheet_name]
            )
        
        elif "memory" in error_msg or "size" in error_msg:
            return ExcelError(
                error_type=ExcelErrorType.MEMORY_LIMIT,
                message=f"Sheet '{sheet_name}' is too large to process",
                details=f"Error: {str(error)}",
                recovery_suggestion="Consider splitting the sheet into smaller parts or using data sampling",
                severity="high",
                affected_sheets=[sheet_name]
            )
        
        else:
            return ExcelError(
                error_type=ExcelErrorType.EXECUTION_ERROR,
                message=f"Error processing sheet '{sheet_name}'",
                details=f"Error: {str(error)}",
                recovery_suggestion="Try processing the sheet individually or check for data format issues",
                severity="medium",
                affected_sheets=[sheet_name]
            )
    
    def handle_data_type_conflict(self, column_name: str, sheet_name: str, expected_type: str, actual_type: str) -> ExcelError:
        """Handle data type conflicts in columns."""
        return ExcelError(
            error_type=ExcelErrorType.DATA_TYPE_CONFLICT,
            message=f"Data type conflict in column '{column_name}'",
            details=f"Expected {expected_type} but found {actual_type} in sheet '{sheet_name}'",
            recovery_suggestion="Consider data type conversion or cleaning the column data",
            severity="medium",
            affected_sheets=[sheet_name],
            affected_columns=[column_name]
        )
    
    def handle_join_error(self, join_keys: List[str], sheets: List[str], error: Exception) -> ExcelError:
        """Handle errors during sheet joining operations."""
        return ExcelError(
            error_type=ExcelErrorType.JOIN_KEY_MISSING,
            message="Unable to join sheets using the specified keys",
            details=f"Join keys: {join_keys}, Sheets: {sheets}, Error: {str(error)}",
            recovery_suggestion="Check if join keys exist in all sheets and have compatible data types",
            severity="medium",
            affected_sheets=sheets,
            affected_columns=join_keys
        )
    
    def validate_dataframe(self, df: pd.DataFrame, sheet_name: str) -> List[ExcelError]:
        """Validate a DataFrame for common issues."""
        errors = []
        
        # Check for empty DataFrame
        if df.empty:
            errors.append(ExcelError(
                error_type=ExcelErrorType.SHEET_EMPTY,
                message=f"Sheet '{sheet_name}' is empty",
                details="DataFrame contains no rows or columns",
                recovery_suggestion="Check if the sheet contains data or skip this sheet",
                severity="low",
                affected_sheets=[sheet_name]
            ))
        
        # Check for memory usage
        memory_usage = df.memory_usage(deep=True).sum()
        if memory_usage > 100 * 1024 * 1024:  # 100MB limit
            errors.append(ExcelError(
                error_type=ExcelErrorType.MEMORY_LIMIT,
                message=f"Sheet '{sheet_name}' uses excessive memory",
                details=f"Memory usage: {memory_usage / 1024 / 1024:.1f} MB",
                recovery_suggestion="Consider data sampling or column selection to reduce memory usage",
                severity="medium",
                affected_sheets=[sheet_name]
            ))
        
        # Check for data type issues
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed data types in object columns
                sample_values = df[col].dropna().head(10)
                if len(sample_values) > 0:
                    try:
                        pd.to_numeric(sample_values, errors='raise')
                    except (ValueError, TypeError):
                        # This is expected for text columns
                        pass
        
        return errors
    
    def get_user_friendly_message(self, error: ExcelError) -> str:
        """Convert error to user-friendly message."""
        severity_icons = {
            'low': 'ℹ️',
            'medium': '⚠️',
            'high': '🚨',
            'critical': '💥'
        }
        
        icon = severity_icons.get(error.severity, '❌')
        
        message = f"{icon} **{error.message}**\n\n"
        message += f"**Details:** {error.details}\n\n"
        message += f"**Suggestion:** {error.recovery_suggestion}"
        
        if error.affected_sheets:
            message += f"\n\n**Affected sheets:** {', '.join(error.affected_sheets)}"
        
        if error.affected_columns:
            message += f"\n\n**Affected columns:** {', '.join(error.affected_columns)}"
        
        return message
    
    def log_error(self, error: ExcelError) -> None:
        """Log error for debugging and monitoring."""
        self.error_history.append(error)
        logger.error(f"Excel Error [{error.error_type.value}]: {error.message}")
        logger.error(f"Details: {error.details}")
        logger.error(f"Severity: {error.severity}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        if not self.error_history:
            return {"total_errors": 0, "by_severity": {}, "by_type": {}}
        
        by_severity = {}
        by_type = {}
        
        for error in self.error_history:
            by_severity[error.severity] = by_severity.get(error.severity, 0) + 1
            by_type[error.error_type.value] = by_type.get(error.error_type.value, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "by_severity": by_severity,
            "by_type": by_type,
            "recent_errors": [error.message for error in self.error_history[-5:]]
        }


class DataTypeConverter:
    """Handles data type conversion and validation."""
    
    @staticmethod
    def infer_and_convert_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Infer and convert data types automatically."""
        conversion_log = []
        
        for col in df.columns:
            original_dtype = df[col].dtype
            
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Try to convert to numeric
            try:
                # Handle percentage strings
                if df[col].dtype == 'object':
                    sample_values = df[col].dropna().head(10)
                    if len(sample_values) > 0:
                        # Check if values contain percentage signs
                        if any('%' in str(val) for val in sample_values):
                            # Convert percentage strings to float
                            df[col] = df[col].astype(str).str.rstrip('%').astype('float64') / 100.0
                            conversion_log.append(f"Converted {col} from percentage strings to float")
                            continue
                
                # Try numeric conversion
                converted = pd.to_numeric(df[col], errors='coerce')
                if not converted.isna().all():  # Only convert if we get some valid numbers
                    df[col] = converted
                    conversion_log.append(f"Converted {col} from {original_dtype} to numeric")
                    
            except (ValueError, TypeError):
                # Try date conversion
                try:
                    converted = pd.to_datetime(df[col], errors='coerce')
                    if not converted.isna().all():
                        df[col] = converted
                        conversion_log.append(f"Converted {col} from {original_dtype} to datetime")
                except (ValueError, TypeError):
                    # Keep as object/string
                    pass
        
        return df, conversion_log
    
    @staticmethod
    def validate_join_compatibility(df1: pd.DataFrame, df2: pd.DataFrame, join_key: str) -> Tuple[bool, str]:
        """Validate if two DataFrames can be joined on a key."""
        if join_key not in df1.columns or join_key not in df2.columns:
            return False, f"Join key '{join_key}' not found in both DataFrames"
        
        # Check data types
        dtype1 = df1[join_key].dtype
        dtype2 = df2[join_key].dtype
        
        if dtype1 != dtype2:
            # Try to convert for compatibility
            try:
                if pd.api.types.is_numeric_dtype(dtype1) and pd.api.types.is_numeric_dtype(dtype2):
                    return True, f"Compatible numeric types: {dtype1} and {dtype2}"
                elif pd.api.types.is_string_dtype(dtype1) or pd.api.types.is_string_dtype(dtype2):
                    return True, f"Compatible string types: {dtype1} and {dtype2}"
                else:
                    return False, f"Incompatible data types: {dtype1} and {dtype2}"
            except Exception:
                return False, f"Cannot convert between types: {dtype1} and {dtype2}"
        
        return True, f"Compatible data types: {dtype1}"


class MemoryOptimizer:
    """Handles memory optimization for large Excel files."""
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Optimize DataFrame memory usage."""
        original_memory = df.memory_usage(deep=True).sum()
        optimization_log = {}
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype('uint8')
                    optimization_log[col] = 'int64 -> uint8'
                elif col_max < 65535:
                    df[col] = df[col].astype('uint16')
                    optimization_log[col] = 'int64 -> uint16'
                elif col_max < 4294967295:
                    df[col] = df[col].astype('uint32')
                    optimization_log[col] = 'int64 -> uint32'
            else:
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype('int8')
                    optimization_log[col] = 'int64 -> int8'
                elif col_min > -32768 and col_max < 32767:
                    df[col] = df[col].astype('int16')
                    optimization_log[col] = 'int64 -> int16'
                elif col_min > -2147483648 and col_max < 2147483647:
                    df[col] = df[col].astype('int32')
                    optimization_log[col] = 'int64 -> int32'
        
        # Optimize float columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
            optimization_log[col] = 'float64 -> float32'
        
        # Optimize object columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df[col]) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
                optimization_log[col] = 'object -> category'
        
        optimized_memory = df.memory_usage(deep=True).sum()
        memory_saved = original_memory - optimized_memory
        
        return df, {
            'original_memory_mb': original_memory / 1024 / 1024,
            'optimized_memory_mb': optimized_memory / 1024 / 1024,
            'memory_saved_mb': memory_saved / 1024 / 1024,
            'optimization_log': optimization_log
        }
    
    @staticmethod
    def sample_large_dataframe(df: pd.DataFrame, max_rows: int = 10000) -> pd.DataFrame:
        """Sample a large DataFrame to reduce memory usage."""
        if len(df) <= max_rows:
            return df
        
        # Use stratified sampling if possible
        if len(df.columns) > 0:
            # Sample based on the first categorical column if available
            cat_cols = df.select_dtypes(include=['category', 'object']).columns
            if len(cat_cols) > 0:
                sample_col = cat_cols[0]
                sample_ratio = min(max_rows / len(df), 1.0)
                df_sample = df.groupby(sample_col, group_keys=False).apply(
                    lambda x: x.sample(frac=sample_ratio, random_state=42)
                )
                # Ensure we get exactly max_rows
                if len(df_sample) > max_rows:
                    return df_sample.head(max_rows)
                elif len(df_sample) < max_rows:
                    # If we don't have enough rows, add more random samples
                    remaining_rows = max_rows - len(df_sample)
                    additional_sample = df[~df.index.isin(df_sample.index)].sample(
                        n=min(remaining_rows, len(df) - len(df_sample)), 
                        random_state=42
                    )
                    return pd.concat([df_sample, additional_sample], ignore_index=True)
                else:
                    return df_sample
        
        # Simple random sampling
        return df.sample(n=max_rows, random_state=42) 