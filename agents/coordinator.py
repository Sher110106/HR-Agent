"""
Agent Coordinator for unified state management and coordination.

This module provides a centralized coordinator that manages agent interactions,
state persistence, and ensures consistent behavior across all agents.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import threading
from enum import Enum

from .memory import ColumnMemoryAgent, SystemPromptMemoryAgent
from .excel_agents import SheetCatalogAgent, ColumnIndexerAgent
from utils.cache import IntelligentCache
from utils.metrics import record_agent_interaction

logger = logging.getLogger(__name__)

class AgentState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    SUCCESS = "success"

@dataclass
class AgentContext:
    """Context shared across all agents for a single analysis session."""
    session_id: str
    data_type: str  # 'csv' or 'excel'
    file_hash: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    state: AgentState = AgentState.IDLE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()

class AgentCoordinator:
    """Centralized coordinator for managing agent interactions and state."""
    
    def __init__(self):
        self.contexts: Dict[str, AgentContext] = {}
        self.agent_cache = IntelligentCache(max_size=100, default_ttl=3600)
        self._lock = threading.Lock()
        self.active_sessions = 0
        
        # Initialize core agents
        self.memory_agent = ColumnMemoryAgent()
        self.system_prompt_agent = SystemPromptMemoryAgent()
        
        logger.info("🎯 AgentCoordinator initialized")
    
    def create_session(self, session_id: str, data_type: str, file_hash: str) -> AgentContext:
        """Create a new analysis session with shared context."""
        with self._lock:
            context = AgentContext(
                session_id=session_id,
                data_type=data_type,
                file_hash=file_hash
            )
            self.contexts[session_id] = context
            self.active_sessions += 1
            
            logger.info(f"🎯 Created session {session_id} for {data_type} analysis")
            return context
    
    def get_context(self, session_id: str) -> Optional[AgentContext]:
        """Get the context for a session."""
        return self.contexts.get(session_id)
    
    def update_session_state(self, session_id: str, state: AgentState, metadata: Dict[str, Any] = None):
        """Update session state and metadata."""
        context = self.get_context(session_id)
        if context:
            context.state = state
            context.update_activity()
            if metadata:
                context.metadata.update(metadata)
            
            logger.debug(f"🎯 Session {session_id} state updated to {state}")
    
    def get_agent_cache(self, agent_name: str, session_id: str) -> IntelligentCache:
        """Get or create a cache instance for a specific agent."""
        cache_key = f"{agent_name}_{session_id}"
        cache = self.agent_cache.get(cache_key)
        if not cache:
            cache = IntelligentCache(max_size=50, default_ttl=1800)
            self.agent_cache.put(cache_key, cache)
        return cache
    
    def record_agent_interaction(self, agent_name: str, session_id: str, 
                               operation: str, success: bool, duration: float):
        """Record agent interaction for metrics and monitoring."""
        record_agent_interaction(agent_name, session_id, operation, success, duration)
        
        # Update session metadata
        context = self.get_context(session_id)
        if context:
            if 'agent_interactions' not in context.metadata:
                context.metadata['agent_interactions'] = []
            
            context.metadata['agent_interactions'].append({
                'agent': agent_name,
                'operation': operation,
                'success': success,
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            })
    
    def cleanup_session(self, session_id: str):
        """Clean up session resources."""
        with self._lock:
            if session_id in self.contexts:
                del self.contexts[session_id]
                self.active_sessions -= 1
                logger.info(f"🎯 Cleaned up session {session_id}")
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of a session."""
        context = self.get_context(session_id)
        if not context:
            return {}
        
        return {
            'session_id': context.session_id,
            'data_type': context.data_type,
            'state': context.state.value,
            'created_at': context.created_at.isoformat(),
            'last_activity': context.last_activity.isoformat(),
            'duration': (context.last_activity - context.created_at).total_seconds(),
            'metadata': context.metadata
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics."""
        with self._lock:
            return {
                'active_sessions': self.active_sessions,
                'total_contexts': len(self.contexts),
                'cache_stats': self.agent_cache.get_stats()
            }

# Global coordinator instance
_coordinator = None

def get_coordinator() -> AgentCoordinator:
    """Get the global agent coordinator instance."""
    global _coordinator
    if _coordinator is None:
        _coordinator = AgentCoordinator()
    return _coordinator 