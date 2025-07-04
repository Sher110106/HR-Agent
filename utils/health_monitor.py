"""
Health monitoring system for the Data Analysis Agent.
Provides system health checks, resource monitoring, and diagnostic endpoints.
"""

import os
import sys
import time
import psutil
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from threading import Thread, Lock
import json
import threading
import gc

logger = logging.getLogger(__name__)

@dataclass
class HealthStatus:
    """Health status for a component."""
    name: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    message: str
    timestamp: datetime
    response_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class HealthMonitor:
    """System health monitoring and diagnostics."""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.health_checks: Dict[str, HealthStatus] = {}
        self._lock = Lock()
        self._monitoring_active = False
        self._monitor_thread = None
        self.start_time = datetime.now()
        logger.info("🏥 HealthMonitor initialized")
    
    def start_monitoring(self):
        """Start background health monitoring."""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("🏥 Health monitoring started")
    
    def stop_monitoring(self):
        """Stop background health monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("🏥 Health monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                self.perform_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"🏥 Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def perform_health_checks(self):
        """Perform all health checks."""
        checks = [
            self._check_system_resources,
            self._check_memory_usage,
            self._check_disk_space,
            self._check_process_health,
            self._check_api_connectivity
        ]
        
        for check in checks:
            try:
                status = check()
                with self._lock:
                    self.health_checks[status.name] = status
            except Exception as e:
                logger.error(f"🏥 Health check failed for {check.__name__}: {e}")
                with self._lock:
                    self.health_checks[check.__name__] = HealthStatus(
                        name=check.__name__,
                        status='unhealthy',
                        message=f"Check failed: {str(e)}",
                        timestamp=datetime.now()
                    )
    
    def _check_system_resources(self) -> HealthStatus:
        """Check CPU and memory usage."""
        start_time = time.time()
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        response_time = (time.time() - start_time) * 1000
        
        # Determine status based on thresholds
        if cpu_percent > 90 or memory.percent > 90:
            status = 'unhealthy'
            message = f"High resource usage: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%"
        elif cpu_percent > 70 or memory.percent > 70:
            status = 'degraded'
            message = f"Moderate resource usage: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%"
        else:
            status = 'healthy'
            message = f"Normal resource usage: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%"
        
        return HealthStatus(
            name="system_resources",
            status=status,
            message=message,
            timestamp=datetime.now(),
            response_time_ms=response_time,
            metadata={
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "memory_total_gb": memory.total / (1024**3)
            }
        )
    
    def _check_memory_usage(self) -> HealthStatus:
        """Check Python process memory usage and garbage collection."""
        start_time = time.time()
        
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # Garbage collection stats
        gc_stats = gc.get_stats()
        gc_counts = gc.get_count()
        
        response_time = (time.time() - start_time) * 1000
        
        if memory_percent > 15:  # High memory usage for a single process
            status = 'degraded'
            message = f"High process memory usage: {memory_percent:.1f}%"
        else:
            status = 'healthy'
            message = f"Normal process memory usage: {memory_percent:.1f}%"
        
        return HealthStatus(
            name="memory_usage",
            status=status,
            message=message,
            timestamp=datetime.now(),
            response_time_ms=response_time,
            metadata={
                "rss_mb": memory_info.rss / (1024**2),
                "vms_mb": memory_info.vms / (1024**2),
                "memory_percent": memory_percent,
                "gc_counts": gc_counts,
                "gc_collections": sum(stat['collections'] for stat in gc_stats)
            }
        )
    
    def _check_disk_space(self) -> HealthStatus:
        """Check available disk space."""
        start_time = time.time()
        
        disk_usage = psutil.disk_usage('.')
        free_percent = (disk_usage.free / disk_usage.total) * 100
        
        response_time = (time.time() - start_time) * 1000
        
        if free_percent < 5:
            status = 'unhealthy'
            message = f"Critical disk space: {free_percent:.1f}% free"
        elif free_percent < 15:
            status = 'degraded'
            message = f"Low disk space: {free_percent:.1f}% free"
        else:
            status = 'healthy'
            message = f"Sufficient disk space: {free_percent:.1f}% free"
        
        return HealthStatus(
            name="disk_space",
            status=status,
            message=message,
            timestamp=datetime.now(),
            response_time_ms=response_time,
            metadata={
                "total_gb": disk_usage.total / (1024**3),
                "free_gb": disk_usage.free / (1024**3),
                "used_gb": disk_usage.used / (1024**3),
                "free_percent": free_percent
            }
        )
    
    def _check_process_health(self) -> HealthStatus:
        """Check process health metrics."""
        start_time = time.time()
        
        process = psutil.Process()
        
        # Count active threads
        thread_count = threading.active_count()
        
        # Process uptime
        uptime = datetime.now() - self.start_time
        
        response_time = (time.time() - start_time) * 1000
        
        if thread_count > 50:
            status = 'degraded'
            message = f"High thread count: {thread_count}"
        else:
            status = 'healthy'
            message = f"Normal process health: {thread_count} threads"
        
        return HealthStatus(
            name="process_health",
            status=status,
            message=message,
            timestamp=datetime.now(),
            response_time_ms=response_time,
            metadata={
                "thread_count": thread_count,
                "uptime_minutes": uptime.total_seconds() / 60,
                "pid": process.pid,
                "cpu_times": process.cpu_times()._asdict(),
                "num_fds": process.num_fds() if hasattr(process, 'num_fds') else None
            }
        )
    
    def _check_api_connectivity(self) -> HealthStatus:
        """Check API connectivity (mock check)."""
        start_time = time.time()
        
        # This would normally test actual API connectivity
        # For now, we'll simulate based on environment variable
        api_key = os.environ.get("NVIDIA_API_KEY")
        
        response_time = (time.time() - start_time) * 1000
        
        if not api_key:
            status = 'unhealthy'
            message = "API key not configured"
        else:
            status = 'healthy'
            message = "API connectivity appears normal"
        
        return HealthStatus(
            name="api_connectivity",
            status=status,
            message=message,
            timestamp=datetime.now(),
            response_time_ms=response_time,
            metadata={
                "api_key_present": bool(api_key),
                "api_key_length": len(api_key) if api_key else 0
            }
        )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        with self._lock:
            if not self.health_checks:
                return {
                    "overall_status": "unknown",
                    "message": "No health checks performed yet",
                    "timestamp": datetime.now().isoformat(),
                    "uptime_minutes": (datetime.now() - self.start_time).total_seconds() / 60
                }
            
            # Determine overall status
            statuses = [check.status for check in self.health_checks.values()]
            if 'unhealthy' in statuses:
                overall_status = 'unhealthy'
            elif 'degraded' in statuses:
                overall_status = 'degraded'
            else:
                overall_status = 'healthy'
            
            return {
                "overall_status": overall_status,
                "timestamp": datetime.now().isoformat(),
                "uptime_minutes": (datetime.now() - self.start_time).total_seconds() / 60,
                "checks": {
                    name: {
                        "status": check.status,
                        "message": check.message,
                        "timestamp": check.timestamp.isoformat(),
                        "response_time_ms": check.response_time_ms,
                        "metadata": check.metadata
                    }
                    for name, check in self.health_checks.items()
                }
            }
    
    def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health information including system info."""
        summary = self.get_health_summary()
        
        # Add system information
        summary["system_info"] = {
            "python_version": sys.version,
            "platform": sys.platform,
            "cpu_count": psutil.cpu_count(),
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            "process_start_time": self.start_time.isoformat()
        }
        
        return summary
    
    def export_health_report(self, filepath: str):
        """Export detailed health report to file."""
        report = self.get_detailed_health()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"🏥 Health report exported to {filepath}")

# Global health monitor instance
health_monitor = HealthMonitor()

def start_health_monitoring():
    """Start the global health monitor."""
    health_monitor.start_monitoring()

def stop_health_monitoring():
    """Stop the global health monitor."""
    health_monitor.stop_monitoring()

def get_health_status() -> Dict[str, Any]:
    """Get current health status."""
    return health_monitor.get_health_summary()

def get_detailed_health_status() -> Dict[str, Any]:
    """Get detailed health status."""
    return health_monitor.get_detailed_health()

def perform_health_check():
    """Perform immediate health check."""
    health_monitor.perform_health_checks()
    return health_monitor.get_health_summary()

def export_health_report(filepath: str = "health_report.json"):
    """Export health report to file."""
    health_monitor.export_health_report(filepath)