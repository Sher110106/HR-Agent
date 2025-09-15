"""
Performance optimization for Excel operations.

This module provides advanced performance optimization features including
caching, lazy loading, parallel processing, and memory management.
"""

import logging
import pandas as pd
import numpy as np
import threading
import time
import psutil
import gc
from typing import Dict, List, Optional, Tuple, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps, lru_cache
import hashlib
import pickle
import os
from datetime import datetime, timedelta

from .excel_error_handling import ExcelErrorHandler, MemoryOptimizer

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics = {
            'query_times': [],
            'memory_usage': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_operations': 0,
            'errors': []
        }
        self.start_time = time.time()
    
    def record_query_time(self, operation: str, duration: float) -> None:
        """Record query execution time."""
        self.metrics['query_times'].append({
            'operation': operation,
            'duration': duration,
            'timestamp': datetime.now()
        })
    
    def record_memory_usage(self) -> None:
        """Record current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        self.metrics['memory_usage'].append({
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'timestamp': datetime.now()
        })
    
    def record_cache_hit(self) -> None:
        """Record cache hit."""
        self.metrics['cache_hits'] += 1
    
    def record_cache_miss(self) -> None:
        """Record cache miss."""
        self.metrics['cache_misses'] += 1
    
    def record_parallel_operation(self) -> None:
        """Record parallel operation."""
        self.metrics['parallel_operations'] += 1
    
    def record_error(self, error: Exception) -> None:
        """Record error occurrence."""
        self.metrics['errors'].append({
            'error': str(error),
            'timestamp': datetime.now()
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics['query_times']:
            return {'status': 'No queries recorded'}
        
        query_times = [q['duration'] for q in self.metrics['query_times']]
        
        summary = {
            'total_queries': len(self.metrics['query_times']),
            'avg_query_time': np.mean(query_times),
            'median_query_time': np.median(query_times),
            'min_query_time': min(query_times),
            'max_query_time': max(query_times),
            'total_runtime': time.time() - self.start_time,
            'cache_hit_rate': self.metrics['cache_hits'] / max(1, self.metrics['cache_hits'] + self.metrics['cache_misses']),
            'parallel_operations': self.metrics['parallel_operations'],
            'error_count': len(self.metrics['errors'])
        }
        
        if self.metrics['memory_usage']:
            memory_values = [m['rss_mb'] for m in self.metrics['memory_usage']]
            summary.update({
                'avg_memory_usage_mb': np.mean(memory_values),
                'max_memory_usage_mb': max(memory_values),
                'current_memory_usage_mb': memory_values[-1] if memory_values else 0
            })
        
        return summary


class AdvancedCache:
    """Advanced caching system with persistence and intelligent eviction."""
    
    def __init__(self, max_size_mb: int = 100, cache_dir: str = ".cache"):
        self.max_size_mb = max_size_mb
        self.cache_dir = cache_dir
        self.memory_cache = {}
        self.cache_metadata = {}
        self.lock = threading.Lock()
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing cache metadata
        self._load_cache_metadata()
    
    def _get_cache_key(self, key: str) -> str:
        """Generate cache key hash."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> str:
        """Get cache file path."""
        return os.path.join(self.cache_dir, f"{key}.pkl")
    
    def _load_cache_metadata(self) -> None:
        """Load cache metadata from disk."""
        metadata_path = os.path.join(self.cache_dir, "metadata.pkl")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'rb') as f:
                    self.cache_metadata = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                self.cache_metadata = {}
    
    def _save_cache_metadata(self) -> None:
        """Save cache metadata to disk."""
        metadata_path = os.path.join(self.cache_dir, "metadata.pkl")
        try:
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.cache_metadata, f)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _get_cache_size_mb(self) -> float:
        """Get current cache size in MB."""
        total_size = 0
        for key in self.cache_metadata:
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                total_size += os.path.getsize(cache_path)
        return total_size / 1024 / 1024
    
    def _evict_oldest(self) -> None:
        """Evict oldest cache entries."""
        if not self.cache_metadata:
            return
        
        # Sort by access time and remove oldest
        sorted_entries = sorted(
            self.cache_metadata.items(),
            key=lambda x: x[1]['last_access']
        )
        
        # Remove oldest entries until we're under the limit
        current_size = self._get_cache_size_mb()
        for key, metadata in sorted_entries:
            if current_size <= self.max_size_mb * 0.8:  # Leave 20% buffer
                break
            
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                file_size = os.path.getsize(cache_path) / 1024 / 1024
                try:
                    os.remove(cache_path)
                    current_size -= file_size
                    del self.cache_metadata[key]
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_path}: {e}")
        
        self._save_cache_metadata()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        cache_key = self._get_cache_key(key)
        
        with self.lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                self.cache_metadata[cache_key]['last_access'] = datetime.now()
                self.cache_metadata[cache_key]['access_count'] += 1
                return self.memory_cache[cache_key]
            
            # Check disk cache
            cache_path = self._get_cache_path(cache_key)
            if os.path.exists(cache_path) and cache_key in self.cache_metadata:
                try:
                    with open(cache_path, 'rb') as f:
                        value = pickle.load(f)
                    
                    # Update metadata
                    self.cache_metadata[cache_key]['last_access'] = datetime.now()
                    self.cache_metadata[cache_key]['access_count'] += 1
                    
                    # Move to memory cache if small enough
                    if len(pickle.dumps(value)) < 1024 * 1024:  # 1MB limit for memory cache
                        self.memory_cache[cache_key] = value
                    
                    return value
                    
                except Exception as e:
                    logger.warning(f"Failed to load from cache {cache_path}: {e}")
                    # Remove corrupted cache entry
                    if cache_key in self.cache_metadata:
                        del self.cache_metadata[cache_key]
                    if os.path.exists(cache_path):
                        try:
                            os.remove(cache_path)
                        except:
                            pass
        
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Set value in cache."""
        cache_key = self._get_cache_key(key)
        
        with self.lock:
            # Check cache size and evict if necessary
            if self._get_cache_size_mb() > self.max_size_mb:
                self._evict_oldest()
            
            # Store in memory cache if small enough
            value_size = len(pickle.dumps(value))
            if value_size < 1024 * 1024:  # 1MB limit for memory cache
                self.memory_cache[cache_key] = value
            
            # Store metadata
            self.cache_metadata[cache_key] = {
                'size_bytes': value_size,
                'created': datetime.now(),
                'last_access': datetime.now(),
                'access_count': 0,
                'ttl_seconds': ttl_seconds
            }
            
            # Store on disk
            cache_path = self._get_cache_path(cache_key)
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
                self._save_cache_metadata()
            except Exception as e:
                logger.warning(f"Failed to save to cache {cache_path}: {e}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.memory_cache.clear()
            self.cache_metadata.clear()
            
            # Remove all cache files
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl') and filename != 'metadata.pkl':
                    try:
                        os.remove(os.path.join(self.cache_dir, filename))
                    except Exception as e:
                        logger.warning(f"Failed to remove cache file {filename}: {e}")
            
            self._save_cache_metadata()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_entries = len(self.cache_metadata)
            memory_entries = len(self.memory_cache)
            total_size_mb = self._get_cache_size_mb()
            
            if total_entries == 0:
                return {
                    'total_entries': 0,
                    'memory_entries': 0,
                    'total_size_mb': 0,
                    'hit_rate': 0
                }
            
            total_accesses = sum(meta['access_count'] for meta in self.cache_metadata.values())
            avg_accesses = total_accesses / total_entries if total_entries > 0 else 0
            
            # Calculate hit rate
            total_requests = total_accesses + len(self.cache_metadata)  # Rough estimate
            hit_rate = total_accesses / max(1, total_requests)
            
            return {
                'total_entries': total_entries,
                'memory_entries': memory_entries,
                'total_size_mb': total_size_mb,
                'max_size_mb': self.max_size_mb,
                'utilization_percent': (total_size_mb / self.max_size_mb) * 100,
                'total_accesses': total_accesses,
                'avg_accesses_per_entry': avg_accesses,
                'hit_rate': hit_rate
            }


class LazyDataLoader:
    """Lazy loading system for large Excel files."""
    
    def __init__(self, file_path: str, sheet_names: Optional[List[str]] = None):
        self.file_path = file_path
        self.sheet_names = sheet_names
        self.loaded_sheets = {}
        self.loading_locks = {}
        self.monitor = PerformanceMonitor()
    
    def get_sheet(self, sheet_name: str) -> pd.DataFrame:
        """Get sheet data, loading it if necessary."""
        if sheet_name not in self.loaded_sheets:
            with self._get_loading_lock(sheet_name):
                # Double-check after acquiring lock
                if sheet_name not in self.loaded_sheets:
                    self._load_sheet(sheet_name)
        
        return self.loaded_sheets[sheet_name]
    
    def _get_loading_lock(self, sheet_name: str) -> threading.Lock:
        """Get or create loading lock for a sheet."""
        if sheet_name not in self.loading_locks:
            self.loading_locks[sheet_name] = threading.Lock()
        return self.loading_locks[sheet_name]
    
    def _load_sheet(self, sheet_name: str) -> None:
        """Load a specific sheet from the Excel file."""
        start_time = time.time()
        
        try:
            logger.info(f"Loading sheet: {sheet_name}")
            
            # Load the sheet
            df = pd.read_excel(
                self.file_path,
                sheet_name=sheet_name,
                engine='openpyxl'
            )
            
            # Optimize memory usage
            df, optimization_info = MemoryOptimizer.optimize_dataframe(df)
            
            # Store the loaded sheet
            self.loaded_sheets[sheet_name] = df
            
            # Record performance metrics
            load_time = time.time() - start_time
            self.monitor.record_query_time(f"load_sheet_{sheet_name}", load_time)
            self.monitor.record_memory_usage()
            
            logger.info(f"Loaded sheet {sheet_name}: {len(df)} rows, {len(df.columns)} columns, "
                       f"{optimization_info['optimized_memory_mb']:.1f} MB, {load_time:.2f}s")
            
        except Exception as e:
            self.monitor.record_error(e)
            logger.error(f"Failed to load sheet {sheet_name}: {e}")
            raise
    
    def preload_sheets(self, sheet_names: List[str]) -> None:
        """Preload multiple sheets in parallel."""
        if not sheet_names:
            return
        
        logger.info(f"Preloading {len(sheet_names)} sheets")
        
        with ThreadPoolExecutor(max_workers=min(len(sheet_names), 4)) as executor:
            # Submit all loading tasks
            future_to_sheet = {
                executor.submit(self.get_sheet, sheet_name): sheet_name
                for sheet_name in sheet_names
            }
            
            # Wait for completion
            for future in as_completed(future_to_sheet):
                sheet_name = future_to_sheet[future]
                try:
                    future.result()  # This will raise any exceptions
                    logger.info(f"Preloaded sheet: {sheet_name}")
                except Exception as e:
                    logger.error(f"Failed to preload sheet {sheet_name}: {e}")
    
    def unload_sheet(self, sheet_name: str) -> None:
        """Unload a sheet to free memory."""
        if sheet_name in self.loaded_sheets:
            del self.loaded_sheets[sheet_name]
            logger.info(f"Unloaded sheet: {sheet_name}")
    
    def unload_all_sheets(self) -> None:
        """Unload all sheets to free memory."""
        self.loaded_sheets.clear()
        logger.info("Unloaded all sheets")
    
    def get_loaded_sheets(self) -> List[str]:
        """Get list of currently loaded sheets."""
        return list(self.loaded_sheets.keys())
    
    def get_memory_usage(self) -> float:
        """Get total memory usage of loaded sheets in MB."""
        total_memory = 0
        for df in self.loaded_sheets.values():
            total_memory += df.memory_usage(deep=True).sum()
        return total_memory / 1024 / 1024


class ParallelProcessor:
    """Parallel processing utilities for Excel operations."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.monitor = PerformanceMonitor()
    
    def parallel_apply(self, 
                      df: pd.DataFrame, 
                      func: Callable, 
                      column: str,
                      chunk_size: int = 1000) -> pd.Series:
        """
        Apply function to DataFrame column in parallel chunks.
        
        Args:
            df: Input DataFrame
            func: Function to apply
            column: Column to apply function to
            chunk_size: Size of chunks for parallel processing
            
        Returns:
            Series with results
        """
        if len(df) <= chunk_size:
            # For small DataFrames, process directly
            return df[column].apply(func)
        
        # Split DataFrame into chunks
        chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit chunk processing tasks
            future_to_chunk = {
                executor.submit(chunk[column].apply, func): i
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results in order
            chunk_results = [None] * len(chunks)
            for future in as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                try:
                    chunk_results[chunk_index] = future.result()
                    self.monitor.record_parallel_operation()
                except Exception as e:
                    self.monitor.record_error(e)
                    logger.error(f"Error processing chunk {chunk_index}: {e}")
                    raise
        
        # Combine results
        return pd.concat(chunk_results, ignore_index=True)
    
    def parallel_groupby(self, 
                        df: pd.DataFrame, 
                        group_cols: List[str], 
                        agg_funcs: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Perform groupby operations in parallel.
        
        Args:
            df: Input DataFrame
            group_cols: Columns to group by
            agg_funcs: Aggregation functions to apply
            
        Returns:
            Aggregated DataFrame
        """
        # For small DataFrames, process directly
        if len(df) < 10000:
            return df.groupby(group_cols).agg(agg_funcs).reset_index()
        
        # Split by groups for parallel processing
        unique_groups = df[group_cols].drop_duplicates()
        group_chunks = np.array_split(unique_groups, self.max_workers)
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit group processing tasks
            future_to_chunk = {
                executor.submit(self._process_group_chunk, df, chunk, group_cols, agg_funcs): i
                for i, chunk in enumerate(group_chunks)
            }
            
            # Collect results
            for future in as_completed(future_to_chunk):
                try:
                    chunk_result = future.result()
                    if chunk_result is not None and not chunk_result.empty:
                        results.append(chunk_result)
                    self.monitor.record_parallel_operation()
                except Exception as e:
                    self.monitor.record_error(e)
                    logger.error(f"Error in parallel groupby: {e}")
                    raise
        
        # Combine results
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _process_group_chunk(self, 
                           df: pd.DataFrame, 
                           group_chunk: pd.DataFrame, 
                           group_cols: List[str], 
                           agg_funcs: Dict[str, List[str]]) -> pd.DataFrame:
        """Process a chunk of groups."""
        # Filter DataFrame to include only the groups in this chunk
        merged = df.merge(group_chunk, on=group_cols, how='inner')
        if merged.empty:
            return pd.DataFrame()
        
        # Perform aggregation
        return merged.groupby(group_cols).agg(agg_funcs).reset_index()


def performance_decorator(func: Callable) -> Callable:
    """Decorator to automatically track performance metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Try to get monitor from args or create new one
            monitor = None
            for arg in args:
                if hasattr(arg, 'monitor'):
                    monitor = arg.monitor
                    break
            
            if monitor is None:
                monitor = PerformanceMonitor()
            
            monitor.record_query_time(func.__name__, duration)
            monitor.record_memory_usage()
            
            return result
            
        except Exception as e:
            # Try to get monitor from args
            monitor = None
            for arg in args:
                if hasattr(arg, 'monitor'):
                    monitor = arg.monitor
                    break
            
            if monitor:
                monitor.record_error(e)
            
            raise
    
    return wrapper


def memory_cleanup() -> None:
    """Perform memory cleanup and garbage collection."""
    # Force garbage collection
    gc.collect()
    
    # Get memory info
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    
    logger.info(f"Memory cleanup completed. Current usage: {memory_mb:.1f} MB")
    
    return memory_mb 