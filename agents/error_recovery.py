"""
Error recovery system for robust agent operations.

This module provides centralized error handling, recovery strategies,
and fallback mechanisms for all agents.
"""

import logging
import traceback
from typing import Any, Dict, Optional, Callable, List, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from .base_agent import AgentResult

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    ABORT = "abort"

@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_type: str
    error_message: str
    severity: ErrorSeverity
    operation: str
    agent_name: str
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ErrorRecoveryManager:
    """Manages error recovery strategies and fallback mechanisms."""
    
    def __init__(self):
        self.recovery_strategies: Dict[str, Callable] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        self.error_history: List[ErrorContext] = []
        self._initialize_strategies()
        logger.info("🛡️ ErrorRecoveryManager initialized")
    
    def _initialize_strategies(self):
        """Initialize default recovery strategies."""
        
        # LLM API Error Recovery
        self.recovery_strategies['llm_api_error'] = self._handle_llm_api_error
        self.recovery_strategies['code_execution_error'] = self._handle_code_execution_error
        self.recovery_strategies['data_processing_error'] = self._handle_data_processing_error
        self.recovery_strategies['memory_error'] = self._handle_memory_error
        
        # Fallback handlers
        self.fallback_handlers['query_understanding'] = self._fallback_query_understanding
        self.fallback_handlers['code_generation'] = self._fallback_code_generation
        self.fallback_handlers['reasoning'] = self._fallback_reasoning
    
    def handle_error(self, error_context: ErrorContext, 
                    original_func: Callable, *args, **kwargs) -> AgentResult:
        """Handle error with appropriate recovery strategy."""
        
        # Log error
        logger.error(f"❌ {error_context.agent_name} {error_context.operation} failed: "
                    f"{error_context.error_type}: {error_context.error_message}")
        
        # Store error context
        self.error_history.append(error_context)
        
        # Determine recovery strategy
        strategy = self._determine_recovery_strategy(error_context)
        
        # Execute recovery
        if strategy == RecoveryStrategy.RETRY:
            return self._execute_retry_strategy(error_context, original_func, *args, **kwargs)
        elif strategy == RecoveryStrategy.FALLBACK:
            return self._execute_fallback_strategy(error_context, *args, **kwargs)
        elif strategy == RecoveryStrategy.DEGRADE:
            return self._execute_degrade_strategy(error_context, *args, **kwargs)
        else:  # ABORT
            return AgentResult(
                success=False,
                error=f"Operation aborted due to {error_context.error_type}: {error_context.error_message}",
                metadata={'error_context': error_context.__dict__}
            )
    
    def _determine_recovery_strategy(self, error_context: ErrorContext) -> RecoveryStrategy:
        """Determine appropriate recovery strategy based on error context."""
        
        # Critical errors should abort
        if error_context.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.ABORT
        
        # High severity errors should use fallback
        if error_context.severity == ErrorSeverity.HIGH:
            return RecoveryStrategy.FALLBACK
        
        # Medium severity errors should retry
        if error_context.severity == ErrorSeverity.MEDIUM:
            return RecoveryStrategy.RETRY
        
        # Low severity errors should degrade
        return RecoveryStrategy.DEGRADE
    
    def _execute_retry_strategy(self, error_context: ErrorContext, 
                              original_func: Callable, *args, **kwargs) -> AgentResult:
        """Execute retry strategy with exponential backoff."""
        
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                delay = base_delay * (2 ** attempt)
                logger.info(f"🔄 Retrying {error_context.operation} (attempt {attempt + 1}/{max_retries})")
                
                if delay > 0:
                    time.sleep(delay)
                
                result = original_func(*args, **kwargs)
                logger.info(f"✅ Retry successful on attempt {attempt + 1}")
                return AgentResult(success=True, data=result)
                
            except Exception as e:
                logger.warning(f"⚠️ Retry attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    # Final attempt failed, try fallback
                    return self._execute_fallback_strategy(error_context, *args, **kwargs)
        
        return AgentResult(success=False, error="All retry attempts failed")
    
    def _execute_fallback_strategy(self, error_context: ErrorContext, 
                                 *args, **kwargs) -> AgentResult:
        """Execute fallback strategy based on operation type."""
        
        fallback_handler = self.fallback_handlers.get(error_context.operation)
        if fallback_handler:
            try:
                result = fallback_handler(error_context, *args, **kwargs)
                logger.info(f"🔄 Fallback strategy executed for {error_context.operation}")
                return AgentResult(success=True, data=result)
            except Exception as e:
                logger.error(f"❌ Fallback strategy failed: {str(e)}")
                return AgentResult(success=False, error=f"Fallback failed: {str(e)}")
        else:
            return AgentResult(success=False, error=f"No fallback handler for {error_context.operation}")
    
    def _execute_degrade_strategy(self, error_context: ErrorContext, 
                                *args, **kwargs) -> AgentResult:
        """Execute degraded operation with reduced functionality."""
        
        logger.info(f"📉 Executing degraded operation for {error_context.operation}")
        
        # Return minimal but functional result
        degraded_result = {
            'status': 'degraded',
            'message': f'Operation completed with reduced functionality due to {error_context.error_type}',
            'original_error': error_context.error_message
        }
        
        return AgentResult(success=True, data=degraded_result, metadata={'degraded': True})
    
    def _handle_llm_api_error(self, error_context: ErrorContext) -> RecoveryStrategy:
        """Handle LLM API errors."""
        if "timeout" in error_context.error_message.lower():
            return RecoveryStrategy.RETRY
        elif "rate limit" in error_context.error_message.lower():
            return RecoveryStrategy.RETRY
        elif "authentication" in error_context.error_message.lower():
            return RecoveryStrategy.ABORT
        else:
            return RecoveryStrategy.FALLBACK
    
    def _handle_code_execution_error(self, error_context: ErrorContext) -> RecoveryStrategy:
        """Handle code execution errors."""
        if "syntax" in error_context.error_message.lower():
            return RecoveryStrategy.RETRY
        elif "name" in error_context.error_message.lower():
            return RecoveryStrategy.RETRY
        elif "memory" in error_context.error_message.lower():
            return RecoveryStrategy.DEGRADE
        else:
            return RecoveryStrategy.FALLBACK
    
    def _handle_data_processing_error(self, error_context: ErrorContext) -> RecoveryStrategy:
        """Handle data processing errors."""
        if "missing" in error_context.error_message.lower():
            return RecoveryStrategy.DEGRADE
        elif "type" in error_context.error_message.lower():
            return RecoveryStrategy.RETRY
        else:
            return RecoveryStrategy.FALLBACK
    
    def _handle_memory_error(self, error_context: ErrorContext) -> RecoveryStrategy:
        """Handle memory-related errors."""
        return RecoveryStrategy.DEGRADE
    
    def _fallback_query_understanding(self, error_context: ErrorContext, 
                                    query: str, conversation_context: str = "") -> bool:
        """Fallback for query understanding using keyword-based approach."""
        logger.info("🔄 Using keyword-based query understanding fallback")
        
        visualization_keywords = [
            'plot', 'chart', 'graph', 'visualize', 'show', 'display',
            'trend', 'compare', 'distribution', 'relationship', 'correlation'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in visualization_keywords)
    
    def _fallback_code_generation(self, error_context: ErrorContext, 
                                 query: str, df, **kwargs) -> str:
        """Fallback for code generation using template-based approach."""
        logger.info("🔄 Using template-based code generation fallback")
        
        from .code_templates import get_template_manager
        template_manager = get_template_manager()
        
        # Suggest template based on query
        template_name = template_manager.suggest_template(query, {'columns': list(df.columns)})
        
        if template_name:
            # Generate basic template with minimal variables
            variables = {
                'dataframe': 'df',
                'data_preparation': '# Using fallback template',
                'title': f'Analysis of {query}',
                'xlabel': 'Categories',
                'ylabel': 'Values',
                'data_df': 'df.head()'
            }
            
            return template_manager.render_template(template_name, variables)
        else:
            return "# Fallback analysis\nresult = df.describe()\nresult"
    
    def _fallback_reasoning(self, error_context: ErrorContext, 
                           query: str, result: Any) -> str:
        """Fallback for reasoning using template-based approach."""
        logger.info("🔄 Using template-based reasoning fallback")
        
        if hasattr(result, 'to_string'):
            # DataFrame result
            return f"""Analysis Results for: {query}

The analysis returned a data table with the following summary:
{result.to_string(max_rows=10, max_cols=5)}

Key observations:
- Data contains {len(result)} rows
- Analysis completed successfully
- Results are ready for further interpretation"""
        else:
            return f"""Analysis Results for: {query}

The analysis completed successfully and returned results.
Please review the output for insights and next steps."""

# Global error recovery manager
_error_recovery_manager = None

def get_error_recovery_manager() -> ErrorRecoveryManager:
    """Get the global error recovery manager."""
    global _error_recovery_manager
    if _error_recovery_manager is None:
        _error_recovery_manager = ErrorRecoveryManager()
    return _error_recovery_manager

def with_error_recovery(operation: str, agent_name: str):
    """Decorator for adding error recovery to agent methods."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = ErrorContext(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    severity=ErrorSeverity.MEDIUM,
                    operation=operation,
                    agent_name=agent_name
                )
                
                recovery_manager = get_error_recovery_manager()
                return recovery_manager.handle_error(error_context, func, *args, **kwargs)
        
        return wrapper
    return decorator 