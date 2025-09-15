"""
Base Agent class providing common functionality for all agents.

This module defines the BaseAgent class that provides shared patterns
for error handling, performance monitoring, caching, and coordination.
"""

import logging
import time
import functools
from typing import Any, Dict, Optional, Callable, TypeVar, Generic
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from .coordinator import get_coordinator, AgentState
from utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from utils.metrics import record_agent_interaction
from utils.cache import IntelligentCache

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class AgentResult(Generic[T]):
    """Standardized result structure for all agents."""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    execution_time: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseAgent(ABC):
    """Base class for all agents providing common functionality."""
    
    def __init__(self, agent_name: str, session_id: Optional[str] = None):
        self.agent_name = agent_name
        self.session_id = session_id
        self.coordinator = get_coordinator()
        
        # Initialize circuit breaker for this agent
        self.circuit_breaker = CircuitBreaker(
            name=f"{agent_name}_breaker",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=60.0,
                success_threshold=2,
                timeout=30.0
            )
        )
        
        # Get agent-specific cache
        if session_id:
            self.cache = self.coordinator.get_agent_cache(agent_name, session_id)
        else:
            self.cache = IntelligentCache(max_size=50, default_ttl=1800)
        
        logger.info(f"🤖 {agent_name} agent initialized")
    
    def execute_with_monitoring(self, operation: str, func: Callable, *args, **kwargs) -> AgentResult:
        """Execute agent operation with comprehensive monitoring."""
        start_time = time.time()
        success = False
        error = None
        data = None
        
        try:
            # Update session state
            if self.session_id:
                self.coordinator.update_session_state(
                    self.session_id, 
                    AgentState.PROCESSING,
                    {'current_operation': operation}
                )
            
            # Execute with circuit breaker protection
            data = self.circuit_breaker.call(func, *args, **kwargs)
            success = True
            
            logger.info(f"✅ {self.agent_name} {operation} completed successfully")
            
        except Exception as e:
            error = str(e)
            logger.error(f"❌ {self.agent_name} {operation} failed: {error}")
            
        finally:
            execution_time = time.time() - start_time
            
            # Record interaction
            if self.session_id:
                self.coordinator.record_agent_interaction(
                    self.agent_name,
                    self.session_id,
                    operation,
                    success,
                    execution_time
                )
            
            # Update session state
            if self.session_id:
                final_state = AgentState.SUCCESS if success else AgentState.ERROR
                self.coordinator.update_session_state(
                    self.session_id,
                    final_state,
                    {'last_operation': operation, 'last_error': error}
                )
        
        return AgentResult(
            success=success,
            data=data,
            error=error,
            execution_time=execution_time,
            metadata={'operation': operation, 'agent': self.agent_name}
        )
    
    def cache_result(self, key: str, result: Any, ttl: Optional[int] = None):
        """Cache agent result with optional TTL."""
        self.cache.put(key, result, ttl=ttl)
        logger.debug(f"💾 {self.agent_name} cached result for key: {key}")
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Retrieve cached result."""
        result = self.cache.get(key)
        if result:
            logger.debug(f"💾 {self.agent_name} cache hit for key: {key}")
        return result
    
    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters. Override in subclasses."""
        return True
    
    def preprocess_input(self, **kwargs) -> Dict[str, Any]:
        """Preprocess input parameters. Override in subclasses."""
        return kwargs
    
    def postprocess_result(self, result: Any) -> Any:
        """Postprocess result. Override in subclasses."""
        return result
    
    @abstractmethod
    def process(self, **kwargs) -> AgentResult:
        """Main processing method. Must be implemented by subclasses."""
        pass
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent-specific statistics."""
        return {
            'agent_name': self.agent_name,
            'session_id': self.session_id,
            'circuit_breaker_stats': self.circuit_breaker.get_stats(),
            'cache_stats': self.cache.get_stats()
        }

def agent_monitor(func: Callable) -> Callable:
    """Decorator for monitoring agent operations."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not isinstance(self, BaseAgent):
            return func(self, *args, **kwargs)
        
        operation = func.__name__
        return self.execute_with_monitoring(operation, func, *args, **kwargs)
    
    return wrapper

class AgentRegistry:
    """Registry for managing all agent instances."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self._lock = threading.Lock()
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent instance."""
        with self._lock:
            self.agents[agent.agent_name] = agent
            logger.info(f"📝 Registered agent: {agent.agent_name}")
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Get agent by name."""
        return self.agents.get(agent_name)
    
    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """Get all registered agents."""
        return dict(self.agents)
    
    def get_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all agents."""
        return {
            name: agent.get_agent_stats() 
            for name, agent in self.agents.items()
        }

# Global agent registry
_agent_registry = None

def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry."""
    global _agent_registry
    if _agent_registry is None:
        _agent_registry = AgentRegistry()
    return _agent_registry 