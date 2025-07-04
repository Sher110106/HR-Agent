"""
Circuit breaker pattern implementation for LLM API calls.
Provides resilience and fail-fast behavior for external API dependencies.
"""

import time
import logging
from enum import Enum
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass
from threading import Lock
import asyncio
import functools
import threading, signal, concurrent.futures

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: float = 30.0  # Seconds before attempting recovery
    success_threshold: int = 2  # Successes needed to close from half-open
    timeout: float = 10.0  # Request timeout in seconds
    
class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass

class CircuitBreaker:
    """Circuit breaker implementation for external API calls."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self._lock = Lock()
        logger.info(f"🔧 CircuitBreaker '{name}' initialized with config: {self.config}")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if not self._can_execute():
            raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN")
        
        start_time = time.time()
        try:
            # Set timeout for the operation
            result = self._execute_with_timeout(func, *args, **kwargs)
            self._on_success()
            return result
        
        except Exception as e:
            execution_time = time.time() - start_time
            self._on_failure(e, execution_time)
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection."""
        if not self._can_execute():
            raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN")
        
        start_time = time.time()
        try:
            # Set timeout for the async operation
            result = await asyncio.wait_for(
                func(*args, **kwargs), 
                timeout=self.config.timeout
            )
            self._on_success()
            return result
        
        except Exception as e:
            execution_time = time.time() - start_time
            self._on_failure(e, execution_time)
            raise
    
    def _can_execute(self) -> bool:
        """Check if the circuit breaker allows execution."""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            
            elif self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if (self.last_failure_time and 
                    time.time() - self.last_failure_time >= self.config.recovery_timeout):
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"🔧 Circuit breaker '{self.name}' moving to HALF_OPEN state")
                    return True
                return False
            
            elif self.state == CircuitState.HALF_OPEN:
                return True
            
            return False
    
    def _execute_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with timeout.

        Uses `signal.alarm` when running in the main thread on POSIX systems. If the current
        thread is *not* the main interpreter thread – which is the case for many
        web-framework worker threads (e.g., Streamlit, FastAPI) – or if running on Windows,
        `signal` cannot be used. In those scenarios we fall back to executing the call inside a
        `concurrent.futures.ThreadPoolExecutor` and utilise its built-in timeout
        support. This prevents the "signal only works in main thread of the main
        interpreter" ValueError seen under Streamlit and platform compatibility issues.
        """
        # -------------------------------------------------------------
        # 1. Main thread + POSIX → safe to use signal.SIGALRM (efficient, no pool)
        # -------------------------------------------------------------
        if (threading.current_thread() is threading.main_thread() and 
            hasattr(signal, 'SIGALRM')):
            def timeout_handler(signum, frame):  # noqa: D401 – simple callback
                raise TimeoutError(
                    f"Function execution timed out after {self.config.timeout}s"
                )

            # Set up timeout alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.config.timeout))
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Cancel the alarm
                signal.signal(signal.SIGALRM, old_handler)  # Restore

        # -------------------------------------------------------------
        # 2. Non-main thread or Windows → fall back to ThreadPoolExecutor
        # -------------------------------------------------------------
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=self.config.timeout)
            except concurrent.futures.TimeoutError:
                # Cancel the future and raise a TimeoutError consistent with
                # the signal-based branch for callers to handle uniformly.
                future.cancel()
                raise TimeoutError(
                    f"Function execution timed out after {self.config.timeout}s"
                )
    
    def _on_success(self):
        """Handle successful execution."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"🔧 Circuit breaker '{self.name}' CLOSED after recovery")
            
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0
    
    def _on_failure(self, exception: Exception, execution_time: float):
        """Handle failed execution."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            logger.warning(
                f"🔧 Circuit breaker '{self.name}' failure #{self.failure_count}: "
                f"{type(exception).__name__} after {execution_time:.2f}s"
            )
            
            if (self.state == CircuitState.CLOSED and 
                self.failure_count >= self.config.failure_threshold):
                self.state = CircuitState.OPEN
                logger.error(f"🔧 Circuit breaker '{self.name}' OPENED after {self.failure_count} failures")
            
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning(f"🔧 Circuit breaker '{self.name}' back to OPEN state")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "success_threshold": self.config.success_threshold,
                    "timeout": self.config.timeout
                }
            }
    
    def reset(self):
        """Manually reset the circuit breaker to CLOSED state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            logger.info(f"🔧 Circuit breaker '{self.name}' manually reset to CLOSED")

class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self._lock = Lock()
    
    def get_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker by name."""
        with self._lock:
            if name not in self.breakers:
                self.breakers[name] = CircuitBreaker(name, config)
            return self.breakers[name]
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        with self._lock:
            return {name: breaker.get_stats() for name, breaker in self.breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self.breakers.values():
                breaker.reset()
            logger.info("🔧 All circuit breakers reset")

# Global registry instance
_registry = CircuitBreakerRegistry()

def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get a circuit breaker instance by name."""
    return _registry.get_breaker(name, config)

def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator for protecting functions with circuit breaker."""
    def decorator(func):
        breaker = get_circuit_breaker(name, config)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await breaker.call_async(func, *args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator

def get_all_circuit_breaker_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all circuit breakers."""
    return _registry.get_all_stats()

def reset_all_circuit_breakers():
    """Reset all circuit breakers to CLOSED state."""
    _registry.reset_all()

# Pre-configured circuit breakers for common operations
llm_api_breaker = get_circuit_breaker("llm_api", CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=60.0,
    success_threshold=2,
    timeout=30.0
))

code_execution_breaker = get_circuit_breaker("code_execution", CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=10.0,
    success_threshold=1,
    timeout=15.0
))