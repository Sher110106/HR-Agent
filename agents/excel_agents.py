"""Excel-specific agents for multi-sheet analysis.

This module contains agents specialized for handling Excel files with multiple sheets,
including sheet cataloging, column indexing, and semantic layer management.
Enhanced with Phase 3: Resilience & Polish features.
"""

import logging
import pandas as pd
import re
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from io import BytesIO

from utils.excel_error_handling import ExcelErrorHandler, DataTypeConverter, MemoryOptimizer
from utils.excel_query_engine import ExcelQueryEngine
from utils.excel_performance import PerformanceMonitor, AdvancedCache, LazyDataLoader

logger = logging.getLogger(__name__)


@dataclass
class ColumnRef:
    """Reference to a column in a specific sheet."""
    sheet_name: str
    column_name: str
    data_type: str
    unique_count: int
    null_count: int
    sample_values: List[str]
    
    def __str__(self):
        return f"{self.sheet_name}.{self.column_name} ({self.data_type})"


@dataclass
class SheetPlan:
    """Plan for how to combine and analyze multiple sheets."""
    primary_sheets: List[str]
    join_strategy: Optional[str] = None  # 'inner', 'left', 'right', 'outer', 'union'
    join_keys: Optional[List[str]] = None
    sheet_aliases: Optional[Dict[str, str]] = None
    union_columns: Optional[List[str]] = None
    additional_columns: Optional[Dict[str, str]] = None  # sheet_name -> column_name for status labels
    
    def __str__(self):
        return f"SheetPlan(primary_sheets={self.primary_sheets}, strategy={self.join_strategy})"


class SheetCatalogAgent:
    """Agent responsible for reading and cataloging Excel sheets."""
    
    def __init__(self):
        self.sheet_catalog: Dict[str, pd.DataFrame] = {}
        self.original_sheet_names: Dict[str, str] = {}
        self.error_handler = ExcelErrorHandler()
        self.performance_monitor = PerformanceMonitor()
        self.cache = AdvancedCache(max_size_mb=50)
        self.data_converter = DataTypeConverter()
        
    def sanitize_sheet_name(self, sheet_name: str) -> str:
        """Convert sheet name to a valid Python variable name."""
        # Remove or replace invalid characters
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', sheet_name)
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = f"sheet_{sanitized}"
        # Ensure it's not empty
        if not sanitized:
            sanitized = "unnamed_sheet"
        # Ensure it's not a Python keyword
        python_keywords = {
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del',
            'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if',
            'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass',
            'raise', 'return', 'try', 'while', 'with', 'yield'
        }
        if sanitized in python_keywords:
            sanitized = f"sheet_{sanitized}"
        return sanitized
    
    def read_excel_file(self, file_content: BytesIO, filename: str) -> Dict[str, pd.DataFrame]:
        """
        Read all sheets from an Excel file and return a catalog of DataFrames.
        Enhanced with error handling, data type conversion, and memory optimization.
        
        Args:
            file_content: BytesIO object containing the Excel file
            filename: Original filename for logging
            
        Returns:
            Dictionary mapping sanitized sheet names to DataFrames
        """
        logger.info(f"📁 SheetCatalogAgent: Reading Excel file: {filename}")
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"excel_file_{filename}_{file_content.getbuffer().nbytes}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info(f"📋 Using cached Excel file: {filename}")
                self.performance_monitor.record_cache_hit()
                return cached_result
            
            self.performance_monitor.record_cache_miss()
            
            # Read all sheets into a dictionary
            all_sheets = pd.read_excel(
                file_content, 
                sheet_name=None,  # Read all sheets
                engine='openpyxl'
            )
            
            logger.info(f"📊 Found {len(all_sheets)} sheets in {filename}")
            
            # Process each sheet with enhanced error handling
            for original_name, df in all_sheets.items():
                try:
                    # Validate DataFrame
                    validation_errors = self.error_handler.validate_dataframe(df, original_name)
                    for error in validation_errors:
                        self.error_handler.log_error(error)
                        if error.severity == 'critical':
                            raise ValueError(f"Critical error in sheet {original_name}: {error.message}")
                    
                    if df.empty:
                        logger.warning(f"⚠️ Sheet '{original_name}' is empty, skipping")
                        continue
                    
                    # Sanitize sheet name for variable safety
                    sanitized_name = self.sanitize_sheet_name(original_name)
                    
                    # Handle duplicate sanitized names
                    counter = 1
                    original_sanitized = sanitized_name
                    while sanitized_name in self.sheet_catalog:
                        sanitized_name = f"{original_sanitized}_{counter}"
                        counter += 1
                    
                    # Convert data types automatically
                    df, conversion_log = self.data_converter.infer_and_convert_types(df)
                    if conversion_log:
                        logger.info(f"🔄 Data type conversions for {original_name}: {', '.join(conversion_log)}")
                    
                    # Optimize memory usage
                    df, optimization_info = MemoryOptimizer.optimize_dataframe(df)
                    if optimization_info['memory_saved_mb'] > 0:
                        logger.info(f"💾 Memory optimization for {original_name}: saved {optimization_info['memory_saved_mb']:.1f} MB")
                    
                    # Store the DataFrame and mapping
                    self.sheet_catalog[sanitized_name] = df
                    self.original_sheet_names[sanitized_name] = original_name
                    
                    logger.info(f"📋 Sheet '{original_name}' → '{sanitized_name}' ({len(df)} rows, {len(df.columns)} columns)")
                    
                except Exception as e:
                    error = self.error_handler.handle_sheet_processing_error(e, original_name)
                    self.error_handler.log_error(error)
                    logger.error(f"❌ Failed to process sheet {original_name}: {e}")
                    # Continue with other sheets instead of failing completely
                    continue
            
            if not self.sheet_catalog:
                raise ValueError("No valid sheets found in Excel file")
            
            # Cache the result
            self.cache.set(cache_key, self.sheet_catalog, ttl_seconds=3600)
            
            # Record performance metrics
            load_time = time.time() - start_time
            self.performance_monitor.record_query_time("read_excel_file", load_time)
            self.performance_monitor.record_memory_usage()
            
            logger.info(f"✅ Successfully cataloged {len(self.sheet_catalog)} sheets in {load_time:.2f}s")
            return self.sheet_catalog
            
        except Exception as e:
            error = self.error_handler.handle_file_reading_error(e, filename)
            self.error_handler.log_error(error)
            self.performance_monitor.record_error(e)
            logger.error(f"❌ Failed to read Excel file {filename}: {e}")
            raise
    
    def get_sheet_info(self) -> Dict[str, Dict[str, Any]]:
        """Get summary information about all sheets."""
        info = {}
        for sanitized_name, df in self.sheet_catalog.items():
            original_name = self.original_sheet_names.get(sanitized_name, sanitized_name)
            info[sanitized_name] = {
                'original_name': original_name,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'data_types': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'null_counts': df.isnull().sum().to_dict()
            }
        return info
    
    def get_sheet_by_name(self, sheet_name: str) -> Optional[pd.DataFrame]:
        """Get a DataFrame by sanitized sheet name."""
        return self.sheet_catalog.get(sheet_name)
    
    def get_original_name(self, sanitized_name: str) -> Optional[str]:
        """Get the original sheet name from sanitized name."""
        return self.original_sheet_names.get(sanitized_name)


class ColumnIndexerAgent:
    """Agent responsible for indexing columns across all sheets and building semantic layer."""
    
    def __init__(self, sheet_catalog: Dict[str, pd.DataFrame]):
        self.sheet_catalog = sheet_catalog
        self.column_index: Dict[str, List[ColumnRef]] = {}
        self.semantic_layer: Dict[str, Any] = {}
        self.error_handler = ExcelErrorHandler()
        self.performance_monitor = PerformanceMonitor()
        self.query_engine = ExcelQueryEngine(sheet_catalog, self.error_handler)
        
    def build_column_index(self) -> Dict[str, List[ColumnRef]]:
        """
        Build an index of all columns across all sheets.
        
        Returns:
            Dictionary mapping lowercase column names to lists of ColumnRef objects
        """
        logger.info("🔍 ColumnIndexerAgent: Building column index across all sheets")
        
        self.column_index.clear()
        
        for sheet_name, df in self.sheet_catalog.items():
            logger.debug(f"📋 Indexing columns for sheet: {sheet_name}")
            
            for col_name in df.columns:
                col_lower = col_name.lower().strip()
                
                # Create ColumnRef object
                col_ref = ColumnRef(
                    sheet_name=sheet_name,
                    column_name=col_name,
                    data_type=str(df[col_name].dtype),
                    unique_count=df[col_name].nunique(),
                    null_count=df[col_name].isnull().sum(),
                    sample_values=self._get_sample_values(df[col_name])
                )
                
                # Add to index
                if col_lower not in self.column_index:
                    self.column_index[col_lower] = []
                self.column_index[col_lower].append(col_ref)
        
        logger.info(f"✅ Built column index with {len(self.column_index)} unique column names")
        return self.column_index
    
    def _get_sample_values(self, series: pd.Series, max_samples: int = 5) -> List[str]:
        """Get sample values from a series for display purposes."""
        try:
            # Get non-null values
            non_null = series.dropna()
            if len(non_null) == 0:
                return []
            
            # Get sample values
            samples = non_null.head(max_samples).astype(str).tolist()
            return samples
        except Exception as e:
            logger.warning(f"⚠️ Error getting sample values: {e}")
            return []
    
    def find_common_columns(self, min_sheets: int = 2) -> Dict[str, List[ColumnRef]]:
        """Find columns that appear in multiple sheets."""
        common_columns = {}
        for col_name, refs in self.column_index.items():
            if len(refs) >= min_sheets:
                common_columns[col_name] = refs
        return common_columns
    
    def find_potential_join_keys(self) -> List[str]:
        """Find columns that could serve as join keys between sheets."""
        potential_keys = []
        
        for col_name, refs in self.column_index.items():
            if len(refs) < 2:
                continue
                
            # Check if this column could be a good join key
            is_good_key = True
            total_rows = 0
            total_unique = 0
            
            for ref in refs:
                # Check for reasonable data type
                if ref.data_type not in ['object', 'int64', 'float64', 'int32', 'float32']:
                    is_good_key = False
                    break
                
                total_rows += ref.unique_count + ref.null_count
                total_unique += ref.unique_count
            
            # Calculate overall uniqueness ratio
            if total_rows > 0:
                overall_uniqueness = total_unique / total_rows
                # More lenient uniqueness requirement (50% instead of 80%)
                if overall_uniqueness < 0.5:
                    is_good_key = False
            
            # Additional checks for common join key patterns
            col_lower = col_name.lower()
            if any(pattern in col_lower for pattern in ['id', 'key', 'code', 'number', 'ref']):
                is_good_key = True  # Force include common ID patterns
            
            if is_good_key:
                potential_keys.append(col_name)
        
        # Sort by likelihood of being a good join key
        def sort_key(col_name):
            refs = self.column_index[col_name]
            # Higher score for more sheets, better data types, and common ID patterns
            score = len(refs) * 10
            col_lower = col_name.lower()
            if any(pattern in col_lower for pattern in ['id', 'key', 'code']):
                score += 50
            if 'employee' in col_lower:
                score += 30
            return score
        
        potential_keys.sort(key=sort_key, reverse=True)
        return potential_keys
    
    def add_semantic_tag(self, tag_name: str, value: Any) -> None:
        """Add a semantic tag to the semantic layer."""
        self.semantic_layer[tag_name] = value
        logger.info(f"🏷️ Added semantic tag: {tag_name} = {value}")
    
    def get_semantic_tag(self, tag_name: str) -> Any:
        """Get a semantic tag from the semantic layer."""
        return self.semantic_layer.get(tag_name)
    
    def get_column_refs(self, column_name: str) -> List[ColumnRef]:
        """Get all ColumnRef objects for a given column name (case-insensitive)."""
        return self.column_index.get(column_name.lower(), [])
    
    def get_sheet_columns(self, sheet_name: str) -> List[str]:
        """Get all column names for a specific sheet."""
        if sheet_name not in self.sheet_catalog:
            return []
        return list(self.sheet_catalog[sheet_name].columns)
    
    def get_column_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get a summary of all columns for display purposes."""
        summary = {}
        
        for col_name, refs in self.column_index.items():
            summary[col_name] = {
                'sheets': [ref.sheet_name for ref in refs],
                'total_occurrences': len(refs),
                'data_types': list(set(ref.data_type for ref in refs)),
                'avg_unique_ratio': sum(ref.unique_count / max(1, ref.unique_count + ref.null_count) for ref in refs) / len(refs),
                'sample_values': refs[0].sample_values if refs else []
            }
        
        return summary 