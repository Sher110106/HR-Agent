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
    CATEGORICAL_ERROR = "categorical_error"


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
                severity="medium"
            )
        
        elif "memory" in error_msg or "out of memory" in error_msg:
            return ExcelError(
                error_type=ExcelErrorType.MEMORY_LIMIT,
                message=f"Sheet '{sheet_name}' is too large to process",
                details=f"Error: {str(error)}",
                recovery_suggestion="Try processing a smaller subset of the data or split the sheet",
                severity="high"
            )
        
        else:
            return ExcelError(
                error_type=ExcelErrorType.EXECUTION_ERROR,
                message=f"Error processing sheet '{sheet_name}'",
                details=f"Error: {str(error)}",
                recovery_suggestion="Check the sheet format and try again",
                severity="medium"
            )
    
    def handle_categorical_error(self, error: Exception, column_name: str, sheet_name: str) -> ExcelError:
        """Handle categorical data type errors."""
        error_msg = str(error).lower()
        
        if "categorical" in error_msg and "new category" in error_msg:
            return ExcelError(
                error_type=ExcelErrorType.CATEGORICAL_ERROR,
                message=f"Categorical data issue in column '{column_name}' of sheet '{sheet_name}'",
                details=f"Error: {str(error)}",
                recovery_suggestion="Convert categorical column to string type before processing: df['column'] = df['column'].astype(str)",
                severity="medium",
                affected_columns=[column_name],
                affected_sheets=[sheet_name]
            )
        
        elif "categorical" in error_msg:
            return ExcelError(
                error_type=ExcelErrorType.CATEGORICAL_ERROR,
                message=f"Categorical data type issue in column '{column_name}' of sheet '{sheet_name}'",
                details=f"Error: {str(error)}",
                recovery_suggestion="Handle categorical data by converting to string or using proper categorical operations",
                severity="medium",
                affected_columns=[column_name],
                affected_sheets=[sheet_name]
            )
        
        else:
            return ExcelError(
                error_type=ExcelErrorType.DATA_TYPE_CONFLICT,
                message=f"Data type issue in column '{column_name}' of sheet '{sheet_name}'",
                details=f"Error: {str(error)}",
                recovery_suggestion="Check data types and convert incompatible columns",
                severity="medium",
                affected_columns=[column_name],
                affected_sheets=[sheet_name]
            )
    
    def handle_data_type_conflict(self, column_name: str, sheet_name: str, expected_type: str, actual_type: str) -> ExcelError:
        """Handle data type conflicts between sheets."""
        return ExcelError(
            error_type=ExcelErrorType.DATA_TYPE_CONFLICT,
            message=f"Data type mismatch in column '{column_name}'",
            details=f"Expected {expected_type} in sheet '{sheet_name}', but found {actual_type}",
            recovery_suggestion="Convert columns to compatible types before joining or unioning sheets",
            severity="medium",
            affected_columns=[column_name],
            affected_sheets=[sheet_name]
        )
    
    def handle_join_error(self, join_keys: List[str], sheets: List[str], error: Exception) -> ExcelError:
        """Handle errors during sheet joining operations."""
        return ExcelError(
            error_type=ExcelErrorType.JOIN_KEY_MISSING,
            message="Cannot join sheets due to missing keys. Check your data structure.",
            details=f"Join keys: {join_keys}, Sheets: {sheets}, Error: {str(error)}",
            recovery_suggestion="Check that join keys exist in all sheets and have compatible data types",
            severity="high",
            affected_sheets=sheets
        )
    
    def validate_dataframe(self, df: pd.DataFrame, sheet_name: str) -> List[ExcelError]:
        """Validate DataFrame for common issues."""
        errors = []
        
        # Check for empty DataFrame
        if df.empty:
            errors.append(ExcelError(
                error_type=ExcelErrorType.SHEET_EMPTY,
                message=f"Sheet '{sheet_name}' is empty",
                details="DataFrame has no rows or columns",
                recovery_suggestion="Check if the sheet contains data",
                severity="medium",
                affected_sheets=[sheet_name]
            ))
        
        # Check for categorical columns that might cause issues
        categorical_columns = []
        for col in df.columns:
            if df[col].dtype.name == 'category':
                categorical_columns.append(col)
        
        if categorical_columns:
            logger.warning(f"⚠️ Found categorical columns in {sheet_name}: {categorical_columns}")
            # Don't add as error, just log for awareness
        
        # Check for memory usage
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        if memory_usage > 100:  # 100MB threshold
            errors.append(ExcelError(
                error_type=ExcelErrorType.MEMORY_LIMIT,
                message=f"Sheet '{sheet_name}' uses {memory_usage:.1f}MB of memory",
                details="Large memory usage may cause performance issues",
                recovery_suggestion="Consider sampling the data or optimizing data types",
                severity="low",
                affected_sheets=[sheet_name]
            ))
        
        return errors
    
    def get_user_friendly_message(self, error: ExcelError) -> str:
        """Convert error to user-friendly message."""
        messages = {
            ExcelErrorType.FILE_CORRUPTED: "📁 The Excel file appears to be corrupted. Please try opening it in Excel and saving it again.",
            ExcelErrorType.SHEET_EMPTY: "📄 One or more sheets are empty. Please check your Excel file.",
            ExcelErrorType.MEMORY_LIMIT: "💾 The file is too large to process efficiently. Consider splitting the data.",
            ExcelErrorType.PERMISSION_DENIED: "🔒 Cannot access the file. Check file permissions.",
            ExcelErrorType.INVALID_FORMAT: "📋 The file format is not supported. Please use .xlsx or .xls files.",
            ExcelErrorType.DATA_TYPE_CONFLICT: "🔧 Data type conflicts detected. The system will attempt to resolve them.",
            ExcelErrorType.CATEGORICAL_ERROR: "📊 Categorical data issue detected. Converting to string format.",
            ExcelErrorType.JOIN_KEY_MISSING: "🔗 Cannot join sheets due to missing keys. Check your data structure.",
            ExcelErrorType.EXECUTION_ERROR: "⚡ An error occurred during processing. Please try again."
        }
        
        return messages.get(error.error_type, f"❌ {error.message}")
    
    def log_error(self, error: ExcelError) -> None:
        """Log error for debugging."""
        self.error_history.append(error)
        logger.error(f"❌ {error.error_type.value}: {error.message}")
        logger.debug(f"📋 Error details: {error.details}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors."""
        error_counts = {}
        for error in self.error_history:
            error_type = error.error_type.value
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "error_counts": error_counts,
            "recovery_attempts": self.recovery_attempts
        }


class DataTypeConverter:
    """Handles data type conversion and optimization."""
    
    @staticmethod
    def infer_and_convert_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Infer and convert data types, handling categorical issues."""
        conversion_log = []
        df_copy = df.copy()
        
        for column in df_copy.columns:
            try:
                # Handle categorical columns that might cause issues
                if df_copy[column].dtype.name == 'category':
                    # Convert categorical to string to avoid issues
                    df_copy[column] = df_copy[column].astype(str)
                    conversion_log.append(f"Converted categorical column '{column}' to string")
                    logger.info(f"🔄 Converted categorical column '{column}' to string to prevent issues")
                
                # Handle object columns that might be dates
                elif df_copy[column].dtype == 'object':
                    # Try to convert to datetime, if it fails, keep as object
                    try:
                        pd.to_datetime(df_copy[column], errors='raise')
                        df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')
                        conversion_log.append(f"Converted column '{column}' to datetime")
                    except (ValueError, TypeError):
                        # Keep as object if not a date
                        pass
                
                # Optimize numeric columns
                elif pd.api.types.is_numeric_dtype(df_copy[column]):
                    # Downcast numeric types to save memory
                    if df_copy[column].dtype == 'int64':
                        df_copy[column] = pd.to_numeric(df_copy[column], downcast='integer')
                    elif df_copy[column].dtype == 'float64':
                        df_copy[column] = pd.to_numeric(df_copy[column], downcast='float')
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to convert column '{column}': {e}")
                conversion_log.append(f"Failed to convert column '{column}': {e}")
        
        return df_copy, conversion_log
    
    @staticmethod
    def validate_join_compatibility(df1: pd.DataFrame, df2: pd.DataFrame, join_key: str) -> Tuple[bool, str]:
        """Validate if two DataFrames can be joined on a key."""
        if join_key not in df1.columns:
            return False, f"Join key '{join_key}' not found in first DataFrame"
        
        if join_key not in df2.columns:
            return False, f"Join key '{join_key}' not found in second DataFrame"
        
        # Check for categorical compatibility
        if df1[join_key].dtype.name == 'category' and df2[join_key].dtype.name == 'category':
            # Check if categories are compatible
            cat1 = set(df1[join_key].cat.categories)
            cat2 = set(df2[join_key].cat.categories)
            if not cat1.issubset(cat2) and not cat2.issubset(cat1):
                return False, f"Categorical join key '{join_key}' has incompatible categories"
        
        return True, "Join compatibility validated"


class MemoryOptimizer:
    """Optimizes DataFrame memory usage."""
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Optimize DataFrame memory usage."""
        original_memory = df.memory_usage(deep=True).sum()
        df_optimized = df.copy()
        
        optimizations = {}
        
        for column in df_optimized.columns:
            col_type = df_optimized[column].dtype
            
            # Handle categorical columns
            if col_type.name == 'category':
                # Convert to string to avoid categorical issues
                df_optimized[column] = df_optimized[column].astype(str)
                optimizations[column] = "categorical_to_string"
            
            # Optimize numeric columns
            elif pd.api.types.is_numeric_dtype(col_type):
                if col_type == 'int64':
                    df_optimized[column] = pd.to_numeric(df_optimized[column], downcast='integer')
                    optimizations[column] = "downcasted_integer"
                elif col_type == 'float64':
                    df_optimized[column] = pd.to_numeric(df_optimized[column], downcast='float')
                    optimizations[column] = "downcasted_float"
            
            # Optimize object columns
            elif col_type == 'object':
                # Check if it's actually a string column
                if df_optimized[column].dtype == 'object':
                    # Convert to string to avoid potential issues
                    df_optimized[column] = df_optimized[column].astype(str)
                    optimizations[column] = "object_to_string"
        
        optimized_memory = df_optimized.memory_usage(deep=True).sum()
        memory_saved = original_memory - optimized_memory
        
        return df_optimized, {
            "original_memory_mb": original_memory / 1024 / 1024,
            "optimized_memory_mb": optimized_memory / 1024 / 1024,
            "memory_saved_mb": memory_saved / 1024 / 1024,
            "optimizations": optimizations
        }
    
    @staticmethod
    def sample_large_dataframe(df: pd.DataFrame, max_rows: int = 10000) -> pd.DataFrame:
        """Sample large DataFrames to improve performance."""
        if len(df) > max_rows:
            logger.info(f"📊 Sampling DataFrame from {len(df)} to {max_rows} rows for performance")
            return df.sample(n=max_rows, random_state=42)
        return df 