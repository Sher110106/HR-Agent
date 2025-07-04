"""
Performance metrics tracking system for the Data Analysis Agent.
Tracks response times, token usage, error rates, and system health.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from threading import Lock
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class MetricEvent:
    """Individual metric event record."""
    timestamp: datetime
    event_type: str  # 'api_call', 'code_execution', 'error', 'system_health'
    duration_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    success: bool = True
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class MetricsCollector:
    """Thread-safe metrics collection and reporting system."""
    
    def __init__(self, max_events: int = 1000):
        self.events: List[MetricEvent] = []
        self.max_events = max_events
        self._lock = Lock()
        self._session_start = datetime.now()
        logger.info("📊 MetricsCollector initialized")
    
    def record_api_call(self, duration_ms: float, tokens_used: int, success: bool = True, 
                       error_type: Optional[str] = None, metadata: Optional[Dict] = None):
        """Record an API call metric."""
        event = MetricEvent(
            timestamp=datetime.now(),
            event_type='api_call',
            duration_ms=duration_ms,
            tokens_used=tokens_used,
            success=success,
            error_type=error_type,
            metadata=metadata or {}
        )
        self._add_event(event)
        logger.debug(f"📊 API call recorded: {duration_ms:.1f}ms, {tokens_used} tokens, success={success}")
    
    def record_code_execution(self, duration_ms: float, success: bool = True, 
                            error_type: Optional[str] = None, metadata: Optional[Dict] = None):
        """Record a code execution metric."""
        event = MetricEvent(
            timestamp=datetime.now(),
            event_type='code_execution',
            duration_ms=duration_ms,
            success=success,
            error_type=error_type,
            metadata=metadata or {}
        )
        self._add_event(event)
        logger.debug(f"📊 Code execution recorded: {duration_ms:.1f}ms, success={success}")
    
    def record_error(self, error_type: str, metadata: Optional[Dict] = None):
        """Record an error event."""
        event = MetricEvent(
            timestamp=datetime.now(),
            event_type='error',
            success=False,
            error_type=error_type,
            metadata=metadata or {}
        )
        self._add_event(event)
        logger.warning(f"📊 Error recorded: {error_type}")
    
    def record_system_health(self, cpu_usage: float, memory_usage: float, 
                           active_sessions: int, metadata: Optional[Dict] = None):
        """Record system health metrics."""
        event = MetricEvent(
            timestamp=datetime.now(),
            event_type='system_health',
            metadata={
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'active_sessions': active_sessions,
                **(metadata or {})
            }
        )
        self._add_event(event)
        logger.debug(f"📊 System health recorded: CPU={cpu_usage:.1f}%, MEM={memory_usage:.1f}%")
    
    def _add_event(self, event: MetricEvent):
        """Thread-safe event addition with size limit."""
        with self._lock:
            self.events.append(event)
            if len(self.events) > self.max_events:
                # Remove oldest events
                self.events = self.events[-self.max_events:]
    
    def get_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get performance summary for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        with self._lock:
            recent_events = [e for e in self.events if e.timestamp >= cutoff_time]
        
        if not recent_events:
            return {"message": "No events in specified time period"}
        
        # Calculate metrics
        api_calls = [e for e in recent_events if e.event_type == 'api_call']
        code_executions = [e for e in recent_events if e.event_type == 'code_execution']
        errors = [e for e in recent_events if e.event_type == 'error']
        
        summary = {
            "time_period_hours": hours_back,
            "session_uptime_minutes": (datetime.now() - self._session_start).total_seconds() / 60,
            "total_events": len(recent_events),
            "api_calls": {
                "count": len(api_calls),
                "success_rate": sum(1 for e in api_calls if e.success) / len(api_calls) if api_calls else 0,
                "avg_response_time_ms": sum(e.duration_ms for e in api_calls if e.duration_ms) / len(api_calls) if api_calls else 0,
                "total_tokens": sum(e.tokens_used for e in api_calls if e.tokens_used),
                "avg_tokens_per_call": sum(e.tokens_used for e in api_calls if e.tokens_used) / len(api_calls) if api_calls else 0
            },
            "code_executions": {
                "count": len(code_executions),
                "success_rate": sum(1 for e in code_executions if e.success) / len(code_executions) if code_executions else 0,
                "avg_execution_time_ms": sum(e.duration_ms for e in code_executions if e.duration_ms) / len(code_executions) if code_executions else 0
            },
            "errors": {
                "count": len(errors),
                "error_rate": len(errors) / len(recent_events) if recent_events else 0,
                "top_error_types": self._get_top_errors(errors)
            }
        }
        
        return summary
    
    def _get_top_errors(self, errors: List[MetricEvent], top_n: int = 5) -> List[Dict[str, Any]]:
        """Get the most frequent error types."""
        error_counts = {}
        for error in errors:
            error_type = error.error_type or "unknown"
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"type": error_type, "count": count} for error_type, count in sorted_errors[:top_n]]
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        with self._lock:
            data = {
                "session_start": self._session_start.isoformat(),
                "export_time": datetime.now().isoformat(),
                "events": [
                    {
                        "timestamp": e.timestamp.isoformat(),
                        "event_type": e.event_type,
                        "duration_ms": e.duration_ms,
                        "tokens_used": e.tokens_used,
                        "success": e.success,
                        "error_type": e.error_type,
                        "metadata": e.metadata
                    }
                    for e in self.events
                ]
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"📊 Metrics exported to {filepath}")

class MetricsTimer:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, operation_type: str, metadata: Optional[Dict] = None):
        self.collector = collector
        self.operation_type = operation_type
        self.metadata = metadata or {}
        self.start_time = None
        self.success = True
        self.error_type = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        
        if exc_type is not None:
            self.success = False
            self.error_type = exc_type.__name__
        
        if self.operation_type == 'api_call':
            tokens = self.metadata.get('tokens_used', 0)
            self.collector.record_api_call(
                duration_ms=duration_ms,
                tokens_used=tokens,
                success=self.success,
                error_type=self.error_type,
                metadata=self.metadata
            )
        elif self.operation_type == 'code_execution':
            self.collector.record_code_execution(
                duration_ms=duration_ms,
                success=self.success,
                error_type=self.error_type,
                metadata=self.metadata
            )
    
    def set_tokens_used(self, tokens: int):
        """Set token usage for API calls."""
        self.metadata['tokens_used'] = tokens

# Global metrics collector instance
metrics_collector = MetricsCollector()

def get_metrics_summary(hours_back: int = 24) -> Dict[str, Any]:
    """Get performance metrics summary."""
    return metrics_collector.get_summary(hours_back)

def api_call_timer(metadata: Optional[Dict] = None) -> MetricsTimer:
    """Context manager for timing API calls."""
    return MetricsTimer(metrics_collector, 'api_call', metadata)

def code_execution_timer(metadata: Optional[Dict] = None) -> MetricsTimer:
    """Context manager for timing code execution."""
    return MetricsTimer(metrics_collector, 'code_execution', metadata)

def record_error(error_type: str, metadata: Optional[Dict] = None):
    """Record an error event."""
    metrics_collector.record_error(error_type, metadata)

def export_metrics(filepath: str = "metrics_export.json"):
    """Export metrics to file."""
    metrics_collector.export_metrics(filepath)