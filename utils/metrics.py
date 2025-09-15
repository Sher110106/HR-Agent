"""
Enhanced metrics collection for comprehensive system monitoring.

This module provides detailed metrics collection for agent operations,
performance monitoring, and system health tracking.
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
from threading import Lock
import psutil
import os

logger = logging.getLogger(__name__)

@dataclass
class AgentMetrics:
    """Metrics for individual agent operations."""
    agent_name: str
    operation: str
    session_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    success: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self, success: bool, error_type: Optional[str] = None, 
                error_message: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Complete the metric recording."""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.success = success
        self.error_type = error_type
        self.error_message = error_message
        if metadata:
            self.metadata.update(metadata)

class MetricsCollector:
    """Comprehensive metrics collection system."""
    
    def __init__(self):
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.system_metrics: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, float] = {}
        self._lock = Lock()
        self._initialize_metrics()
        logger.info("📊 MetricsCollector initialized")
    
    def _initialize_metrics(self):
        """Initialize default metrics."""
        self.system_metrics = {
            'total_agent_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'peak_memory_usage': 0.0,
            'current_memory_usage': 0.0,
            'cpu_usage': 0.0,
            'active_sessions': 0,
            'cache_hit_rate': 0.0,
            'circuit_breaker_trips': 0,
            'last_updated': datetime.now().isoformat()
        }
    
    def record_agent_operation(self, agent_name: str, operation: str, 
                              session_id: Optional[str] = None) -> str:
        """Record the start of an agent operation."""
        metric_id = f"{agent_name}_{operation}_{int(time.time() * 1000)}"
        
        with self._lock:
            metric = AgentMetrics(
                agent_name=agent_name,
                operation=operation,
                session_id=session_id
            )
            self.agent_metrics[metric_id] = metric
            self.system_metrics['total_agent_operations'] += 1
        
        logger.debug(f"📊 Started recording metrics for {agent_name}.{operation}")
        return metric_id
    
    def complete_agent_operation(self, metric_id: str, success: bool, 
                               error_type: Optional[str] = None,
                               error_message: Optional[str] = None,
                               metadata: Optional[Dict[str, Any]] = None):
        """Complete the recording of an agent operation."""
        with self._lock:
            if metric_id in self.agent_metrics:
                metric = self.agent_metrics[metric_id]
                metric.complete(success, error_type, error_message, metadata)
                
                # Update system metrics
                if success:
                    self.system_metrics['successful_operations'] += 1
                else:
                    self.system_metrics['failed_operations'] += 1
                
                if metric.duration:
                    self.system_metrics['total_execution_time'] += metric.duration
                    self.system_metrics['average_execution_time'] = (
                        self.system_metrics['total_execution_time'] / 
                        self.system_metrics['total_agent_operations']
                    )
                
                self.system_metrics['last_updated'] = datetime.now().isoformat()
                
                logger.debug(f"📊 Completed metrics for {metric.agent_name}.{metric.operation} "
                           f"({'success' if success else 'failed'})")
    
    def record_performance_metric(self, metric_name: str, value: float):
        """Record a performance metric."""
        with self._lock:
            self.performance_metrics[metric_name] = value
            logger.debug(f"📊 Recorded performance metric: {metric_name} = {value}")
    
    def record_system_health(self):
        """Record current system health metrics."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            with self._lock:
                self.system_metrics['current_memory_usage'] = memory_info.rss / 1024 / 1024  # MB
                self.system_metrics['peak_memory_usage'] = max(
                    self.system_metrics['peak_memory_usage'],
                    self.system_metrics['current_memory_usage']
                )
                self.system_metrics['cpu_usage'] = process.cpu_percent()
                self.system_metrics['last_updated'] = datetime.now().isoformat()
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to record system health: {e}")
    
    def record_cache_metrics(self, hit_rate: float):
        """Record cache performance metrics."""
        with self._lock:
            self.system_metrics['cache_hit_rate'] = hit_rate
    
    def record_circuit_breaker_trip(self):
        """Record a circuit breaker trip."""
        with self._lock:
            self.system_metrics['circuit_breaker_trips'] += 1
    
    def get_agent_metrics(self, agent_name: Optional[str] = None, 
                         session_id: Optional[str] = None) -> Dict[str, AgentMetrics]:
        """Get metrics filtered by agent name and/or session ID."""
        with self._lock:
            filtered_metrics = {}
            for metric_id, metric in self.agent_metrics.items():
                if agent_name and metric.agent_name != agent_name:
                    continue
                if session_id and metric.session_id != session_id:
                    continue
                filtered_metrics[metric_id] = metric
            return filtered_metrics
    
    def get_agent_summary(self, agent_name: str) -> Dict[str, Any]:
        """Get summary statistics for a specific agent."""
        agent_metrics = self.get_agent_metrics(agent_name=agent_name)
        
        if not agent_metrics:
            return {}
        
        total_operations = len(agent_metrics)
        successful_operations = sum(1 for m in agent_metrics.values() if m.success)
        failed_operations = total_operations - successful_operations
        total_duration = sum(m.duration or 0 for m in agent_metrics.values())
        avg_duration = total_duration / total_operations if total_operations > 0 else 0
        
        return {
            'agent_name': agent_name,
            'total_operations': total_operations,
            'successful_operations': successful_operations,
            'failed_operations': failed_operations,
            'success_rate': successful_operations / total_operations if total_operations > 0 else 0,
            'total_duration': total_duration,
            'average_duration': avg_duration,
            'operations': list(agent_metrics.values())
        }
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary."""
        with self._lock:
            # Update system health
            self.record_system_health()
            
            return {
                'system_metrics': dict(self.system_metrics),
                'performance_metrics': dict(self.performance_metrics),
                'total_agent_metrics': len(self.agent_metrics),
                'last_updated': datetime.now().isoformat()
            }
    
    def cleanup_old_metrics(self, max_age_hours: int = 24):
        """Clean up old metrics to prevent memory bloat."""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        with self._lock:
            old_metrics = [
                metric_id for metric_id, metric in self.agent_metrics.items()
                if metric.start_time.timestamp() < cutoff_time
            ]
            
            for metric_id in old_metrics:
                del self.agent_metrics[metric_id]
            
            if old_metrics:
                logger.info(f"🧹 Cleaned up {len(old_metrics)} old metrics")

# Global metrics collector
_metrics_collector = None

def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector

@contextmanager
def agent_operation_timer(agent_name: str, operation: str, session_id: Optional[str] = None):
    """Context manager for timing agent operations."""
    collector = get_metrics_collector()
    metric_id = collector.record_agent_operation(agent_name, operation, session_id)
    
    try:
        yield metric_id
        collector.complete_agent_operation(metric_id, success=True)
    except Exception as e:
        collector.complete_agent_operation(
            metric_id, 
            success=False, 
            error_type=type(e).__name__,
            error_message=str(e)
        )
        raise

def record_agent_interaction(agent_name: str, session_id: str, operation: str, 
                           success: bool, duration: float):
    """Record agent interaction for metrics."""
    collector = get_metrics_collector()
    metric_id = f"{agent_name}_{operation}_{int(time.time() * 1000)}"
    
    collector.complete_agent_operation(
        metric_id,
        success=success,
        metadata={'duration': duration, 'session_id': session_id}
    )

# Legacy timer functions for backward compatibility
@contextmanager
def api_call_timer():
    """Timer for API calls."""
    start_time = time.time()
    timer = type('Timer', (), {
        'set_tokens_used': lambda self, tokens: setattr(self, 'tokens_used', tokens),
        'tokens_used': 0
    })()
    
    try:
        yield timer
    finally:
        duration = time.time() - start_time
        get_metrics_collector().record_performance_metric('api_call_duration', duration)

@contextmanager
def code_execution_timer():
    """Timer for code execution."""
    start_time = time.time()
    timer = type('Timer', (), {
        'metadata': {},
        'set_tokens_used': lambda self, tokens: setattr(self, 'tokens_used', tokens),
        'tokens_used': 0
    })()
    
    try:
        yield timer
    finally:
        duration = time.time() - start_time
        get_metrics_collector().record_performance_metric('code_execution_duration', duration)

def record_error(error_type: str, metadata: Dict[str, Any]):
    """Record an error for metrics."""
    get_metrics_collector().record_performance_metric(f'error_{error_type}', 1)

# Legacy functions for backward compatibility
def get_metrics_summary(hours_back: int = 24) -> Dict[str, Any]:
    """Get performance metrics summary (legacy function for backward compatibility)."""
    collector = get_metrics_collector()
    system_summary = collector.get_system_summary()
    
    # Format for backward compatibility
    return {
        "time_period_hours": hours_back,
        "session_uptime_minutes": 0,  # Not tracked in new system
        "total_events": system_summary['total_agent_metrics'],
        "api_calls": {
            "count": system_summary['system_metrics'].get('total_agent_operations', 0),
            "success_rate": system_summary['system_metrics'].get('successful_operations', 0) / 
                          max(system_summary['system_metrics'].get('total_agent_operations', 1), 1),
            "avg_response_time_ms": system_summary['system_metrics'].get('average_execution_time', 0) * 1000,
            "total_tokens": 0,  # Not tracked in new system
            "avg_tokens_per_call": 0  # Not tracked in new system
        },
        "code_executions": {
            "count": system_summary['system_metrics'].get('total_agent_operations', 0),
            "success_rate": system_summary['system_metrics'].get('successful_operations', 0) / 
                          max(system_summary['system_metrics'].get('total_agent_operations', 1), 1),
            "avg_execution_time_ms": system_summary['system_metrics'].get('average_execution_time', 0) * 1000
        },
        "errors": {
            "count": system_summary['system_metrics'].get('failed_operations', 0),
            "error_rate": system_summary['system_metrics'].get('failed_operations', 0) / 
                        max(system_summary['system_metrics'].get('total_agent_operations', 1), 1),
            "top_error_types": []  # Not tracked in new system
        }
    }

# Legacy global instance for backward compatibility
metrics_collector = get_metrics_collector()