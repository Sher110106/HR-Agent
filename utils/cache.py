"""
Intelligent caching system for code snippets, analysis results, and API responses.
Improves performance by storing and retrieving successful patterns.
"""

import hashlib
import json
import time
import logging
import pickle
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from threading import Lock
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int]
    metadata: Dict[str, Any]
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    def update_access(self):
        """Update access metadata."""
        self.last_accessed = datetime.now()
        self.access_count += 1

class IntelligentCache:
    """High-performance caching system with LRU eviction and TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self._lock = Lock()
        self.hits = 0
        self.misses = 0
        logger.info(f"💾 IntelligentCache initialized: max_size={max_size}, default_ttl={default_ttl}s")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check if expired
            if entry.is_expired():
                del self.cache[key]
                self.misses += 1
                logger.debug(f"💾 Cache entry expired: {key}")
                return None
            
            # Update access metadata
            entry.update_access()
            self.hits += 1
            logger.debug(f"💾 Cache hit: {key}")
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None, metadata: Optional[Dict] = None):
        """Store value in cache."""
        with self._lock:
            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                ttl_seconds=ttl,
                metadata=metadata or {}
            )
            
            # Add to cache
            self.cache[key] = entry
            
            # Evict if over size limit
            self._evict_if_needed()
            
            logger.debug(f"💾 Cache stored: {key} (TTL: {ttl}s)")
    
    def _evict_if_needed(self):
        """Evict least recently used entries if cache is full."""
        while len(self.cache) > self.max_size:
            # Find LRU entry
            lru_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].last_accessed
            )
            del self.cache[lru_key]
            logger.debug(f"💾 Cache evicted LRU: {lru_key}")
    
    def invalidate(self, key: str) -> bool:
        """Remove specific key from cache."""
        with self._lock:
            if key in self.cache:
                del self.cache[key]
                logger.debug(f"💾 Cache invalidated: {key}")
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            logger.info("💾 Cache cleared")
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        with self._lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                logger.info(f"💾 Cleaned up {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "memory_usage_estimate": self._estimate_memory_usage()
            }
    
    def _estimate_memory_usage(self) -> Dict[str, int]:
        """Estimate memory usage of cache."""
        total_size = 0
        for entry in self.cache.values():
            try:
                entry_size = len(pickle.dumps(entry.value))
                total_size += entry_size
            except:
                total_size += 1024  # Rough estimate for unpicklable objects
        
        return {
            "total_bytes": total_size,
            "total_mb": total_size / (1024 * 1024),
            "avg_entry_bytes": total_size / len(self.cache) if self.cache else 0
        }

class CodeSnippetCache(IntelligentCache):
    """Specialized cache for code snippets and execution results."""
    
    def __init__(self):
        super().__init__(max_size=500, default_ttl=1800)  # 30 minutes
        self.successful_patterns: Dict[str, int] = {}  # Pattern -> success count
    
    def cache_code_result(self, query: str, data_columns: List[str], code: str, 
                         result: Any, success: bool = True, metadata: Optional[Dict] = None):
        """Cache code generation and execution result."""
        # Create cache key based on query and data structure
        key_data = {
            'query': query.lower().strip(),
            'columns': sorted(data_columns),
            'type': 'code_result'
        }
        key = self._generate_key(key_data)
        
        cache_metadata = {
            'query': query,
            'columns': data_columns,
            'code': code,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            **(metadata or {})
        }
        
        self.put(key, result, metadata=cache_metadata)
        
        if success:
            # Track successful patterns
            pattern_key = self._extract_pattern(query)
            self.successful_patterns[pattern_key] = self.successful_patterns.get(pattern_key, 0) + 1
            logger.debug(f"💾 Cached successful code result for pattern: {pattern_key}")
    
    def get_similar_code_result(self, query: str, data_columns: List[str]) -> Optional[Tuple[str, Any]]:
        """Get cached result for similar query."""
        # Try exact match first
        key_data = {
            'query': query.lower().strip(),
            'columns': sorted(data_columns),
            'type': 'code_result'
        }
        key = self._generate_key(key_data)
        result = self.get(key)
        
        if result is not None:
            # Get the code from metadata
            with self._lock:
                if key in self.cache:
                    code = self.cache[key].metadata.get('code', '')
                    return code, result
        
        # Try pattern-based matching
        pattern = self._extract_pattern(query)
        for cache_key, entry in self.cache.items():
            if (entry.metadata.get('success', False) and
                self._extract_pattern(entry.metadata.get('query', '')) == pattern and
                self._columns_similar(entry.metadata.get('columns', []), data_columns)):
                
                logger.debug(f"💾 Found similar cached result for pattern: {pattern}")
                entry.update_access()
                return entry.metadata.get('code', ''), entry.value
        
        return None
    
    def _extract_pattern(self, query: str) -> str:
        """Extract query pattern for matching."""
        # Simple pattern extraction - could be enhanced with NLP
        import re
        
        # Remove specific values and normalize
        pattern = query.lower()
        pattern = re.sub(r'\b\d+\b', 'NUM', pattern)  # Replace numbers
        pattern = re.sub(r"'[^']*'", 'STR', pattern)  # Replace strings
        pattern = re.sub(r'"[^"]*"', 'STR', pattern)  # Replace strings
        pattern = re.sub(r'\s+', ' ', pattern).strip()  # Normalize whitespace
        
        # Extract key action words
        action_words = ['plot', 'chart', 'graph', 'show', 'display', 'analyze', 'filter', 'group', 'sort', 'count', 'sum', 'average', 'mean']
        found_actions = [word for word in action_words if word in pattern]
        
        return ' '.join(found_actions) if found_actions else pattern[:50]
    
    def _columns_similar(self, cols1: List[str], cols2: List[str], threshold: float = 0.7) -> bool:
        """Check if column lists are similar enough."""
        if not cols1 or not cols2:
            return False
        
        # Calculate Jaccard similarity
        set1, set2 = set(cols1), set(cols2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold
    
    def get_popular_patterns(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most successful query patterns."""
        return sorted(
            self.successful_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

class AnalysisCache(IntelligentCache):
    """Specialized cache for data analysis results."""
    
    def __init__(self):
        super().__init__(max_size=200, default_ttl=3600)  # 1 hour
    
    def cache_analysis(self, data_hash: str, analysis_type: str, result: Any, metadata: Optional[Dict] = None):
        """Cache analysis result."""
        key = f"{analysis_type}:{data_hash}"
        
        cache_metadata = {
            'analysis_type': analysis_type,
            'data_hash': data_hash,
            'timestamp': datetime.now().isoformat(),
            **(metadata or {})
        }
        
        self.put(key, result, metadata=cache_metadata)
        logger.debug(f"💾 Cached analysis result: {analysis_type}")
    
    def get_analysis(self, data_hash: str, analysis_type: str) -> Optional[Any]:
        """Get cached analysis result."""
        key = f"{analysis_type}:{data_hash}"
        return self.get(key)

class PersistentCache:
    """Persistent cache that survives application restarts."""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._lock = Lock()
        logger.info(f"💾 PersistentCache initialized: {self.cache_dir}")
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        safe_key = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from persistent cache."""
        file_path = self._get_file_path(key)
        
        try:
            with self._lock:
                if not file_path.exists():
                    return None
                
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Check expiration
                if 'expires_at' in data and datetime.now() > data['expires_at']:
                    file_path.unlink()
                    return None
                
                logger.debug(f"💾 Persistent cache hit: {key}")
                return data['value']
        
        except Exception as e:
            logger.warning(f"💾 Failed to read persistent cache {key}: {e}")
            return None
    
    def put(self, key: str, value: Any, ttl_hours: int = 24):
        """Store value in persistent cache."""
        file_path = self._get_file_path(key)
        
        data = {
            'value': value,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(hours=ttl_hours),
            'key': key
        }
        
        try:
            with self._lock:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                
                logger.debug(f"💾 Persistent cache stored: {key}")
        
        except Exception as e:
            logger.warning(f"💾 Failed to write persistent cache {key}: {e}")
    
    def cleanup_expired(self) -> int:
        """Remove expired cache files."""
        removed_count = 0
        
        try:
            with self._lock:
                for file_path in self.cache_dir.glob("*.cache"):
                    try:
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                        
                        if 'expires_at' in data and datetime.now() > data['expires_at']:
                            file_path.unlink()
                            removed_count += 1
                    
                    except Exception:
                        # Remove corrupted files
                        file_path.unlink()
                        removed_count += 1
            
            if removed_count > 0:
                logger.info(f"💾 Cleaned up {removed_count} expired persistent cache files")
        
        except Exception as e:
            logger.warning(f"💾 Failed to cleanup persistent cache: {e}")
        
        return removed_count

# Global cache instances
code_cache = CodeSnippetCache()
analysis_cache = AnalysisCache()
persistent_cache = PersistentCache()

def get_cache_stats() -> Dict[str, Any]:
    """Get statistics for all caches."""
    return {
        "code_cache": code_cache.get_stats(),
        "analysis_cache": analysis_cache.get_stats(),
        "popular_patterns": code_cache.get_popular_patterns()
    }

def cleanup_all_caches():
    """Cleanup expired entries in all caches."""
    code_expired = code_cache.cleanup_expired()
    analysis_expired = analysis_cache.cleanup_expired()
    persistent_expired = persistent_cache.cleanup_expired()
    
    logger.info(f"💾 Cache cleanup completed: {code_expired + analysis_expired + persistent_expired} entries removed")

def clear_all_caches():
    """Clear all cache instances."""
    code_cache.clear()
    analysis_cache.clear()
    logger.info("💾 All caches cleared")