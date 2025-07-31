"""
Agent Factory for centralized agent creation and configuration.

This module provides a factory pattern for creating and configuring
agents with proper initialization and dependency injection.
"""

import logging
from typing import Dict, Any, Optional, Type
from dataclasses import dataclass
from enum import Enum

from .base_agent import BaseAgent
from .coordinator import get_coordinator
from .memory import ColumnMemoryAgent, SystemPromptMemoryAgent
from .excel_agents import SheetCatalogAgent, ColumnIndexerAgent
from .sheet_selection import SheetSelectionAgent
from .excel_code_generation import ExcelCodeGenerationAgent
from .excel_execution import ExcelExecutionAgent
from .code_generation import CodeGenerationAgent
from .execution import ExecutionAgent
from .reasoning import ReasoningAgent
from .data_analysis import DataInsightAgent

logger = logging.getLogger(__name__)

class AgentType(Enum):
    MEMORY = "memory"
    EXCEL_CATALOG = "excel_catalog"
    EXCEL_INDEXER = "excel_indexer"
    SHEET_SELECTION = "sheet_selection"
    EXCEL_CODE_GEN = "excel_code_generation"
    EXCEL_EXECUTION = "excel_execution"
    CODE_GENERATION = "code_generation"
    EXECUTION = "execution"
    REASONING = "reasoning"
    DATA_INSIGHT = "data_insight"

@dataclass
class AgentConfig:
    """Configuration for agent creation."""
    agent_type: AgentType
    session_id: Optional[str] = None
    dependencies: Dict[str, Any] = None
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = {}
        if self.config is None:
            self.config = {}

class AgentFactory:
    """Factory for creating and configuring agents."""
    
    def __init__(self):
        self.agent_registry: Dict[AgentType, Type[BaseAgent]] = {}
        self.coordinator = get_coordinator()
        self._register_agents()
        logger.info("🏭 AgentFactory initialized")
    
    def _register_agents(self):
        """Register all available agent types."""
        # Note: Most agents are currently function-based, so we'll create wrapper classes
        # for the factory pattern. In a future refactor, these could be converted to classes.
        
        # Register agent types that can be instantiated
        self.agent_registry[AgentType.MEMORY] = ColumnMemoryAgent
        self.agent_registry[AgentType.EXCEL_CATALOG] = SheetCatalogAgent
        self.agent_registry[AgentType.EXCEL_INDEXER] = ColumnIndexerAgent
        self.agent_registry[AgentType.SHEET_SELECTION] = SheetSelectionAgent
        self.agent_registry[AgentType.EXCEL_CODE_GEN] = ExcelCodeGenerationAgent
        self.agent_registry[AgentType.EXCEL_EXECUTION] = ExcelExecutionAgent
        self.agent_registry[AgentType.CODE_GENERATION] = CodeGenerationAgent
        self.agent_registry[AgentType.EXECUTION] = ExecutionAgent
        self.agent_registry[AgentType.REASONING] = ReasoningAgent
        self.agent_registry[AgentType.DATA_INSIGHT] = DataInsightAgent
    
    def create_agent(self, config: AgentConfig) -> Any:
        """Create an agent based on configuration."""
        
        agent_type = config.agent_type
        session_id = config.session_id
        dependencies = config.dependencies
        agent_config = config.config
        
        logger.info(f"🏭 Creating agent: {agent_type.value}")
        
        try:
            if agent_type == AgentType.MEMORY:
                return self._create_memory_agent(session_id, agent_config)
            
            elif agent_type == AgentType.EXCEL_CATALOG:
                return self._create_excel_catalog_agent(session_id, agent_config)
            
            elif agent_type == AgentType.EXCEL_INDEXER:
                sheet_catalog = dependencies.get('sheet_catalog')
                if not sheet_catalog:
                    raise ValueError("SheetCatalogAgent required for ColumnIndexerAgent")
                return self._create_excel_indexer_agent(sheet_catalog, session_id, agent_config)
            
            elif agent_type == AgentType.SHEET_SELECTION:
                column_indexer = dependencies.get('column_indexer')
                sheet_catalog = dependencies.get('sheet_catalog')
                if not column_indexer:
                    raise ValueError("ColumnIndexerAgent required for SheetSelectionAgent")
                return self._create_sheet_selection_agent(column_indexer, sheet_catalog, session_id, agent_config)
            
            elif agent_type == AgentType.EXCEL_CODE_GEN:
                column_indexer = dependencies.get('column_indexer')
                if not column_indexer:
                    raise ValueError("ColumnIndexerAgent required for ExcelCodeGenerationAgent")
                return self._create_excel_code_generation_agent(column_indexer, session_id, agent_config)
            
            elif agent_type == AgentType.EXCEL_EXECUTION:
                column_indexer = dependencies.get('column_indexer')
                if not column_indexer:
                    raise ValueError("ColumnIndexerAgent required for ExcelExecutionAgent")
                return self._create_excel_execution_agent(column_indexer, session_id, agent_config)
            
            elif agent_type == AgentType.CODE_GENERATION:
                return self._create_code_generation_agent(session_id, agent_config)
            
            elif agent_type == AgentType.EXECUTION:
                return self._create_execution_agent(session_id, agent_config)
            
            elif agent_type == AgentType.REASONING:
                return self._create_reasoning_agent(session_id, agent_config)
            
            elif agent_type == AgentType.DATA_INSIGHT:
                return self._create_data_insight_agent(session_id, agent_config)
            
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
                
        except Exception as e:
            logger.error(f"❌ Failed to create agent {agent_type.value}: {str(e)}")
            raise
    
    def _create_memory_agent(self, session_id: Optional[str], config: Dict[str, Any]) -> ColumnMemoryAgent:
        """Create memory agent."""
        agent = ColumnMemoryAgent()
        logger.debug(f"✅ Created ColumnMemoryAgent")
        return agent
    
    def _create_excel_catalog_agent(self, session_id: Optional[str], config: Dict[str, Any]) -> SheetCatalogAgent:
        """Create Excel catalog agent."""
        agent = SheetCatalogAgent()
        logger.debug(f"✅ Created SheetCatalogAgent")
        return agent
    
    def _create_excel_indexer_agent(self, sheet_catalog: Dict[str, Any], 
                                   session_id: Optional[str], config: Dict[str, Any]) -> ColumnIndexerAgent:
        """Create Excel indexer agent."""
        agent = ColumnIndexerAgent(sheet_catalog)
        logger.debug(f"✅ Created ColumnIndexerAgent")
        return agent
    
    def _create_sheet_selection_agent(self, column_indexer: ColumnIndexerAgent, 
                                     sheet_catalog: Optional[SheetCatalogAgent],
                                     session_id: Optional[str], config: Dict[str, Any]) -> SheetSelectionAgent:
        """Create sheet selection agent."""
        agent = SheetSelectionAgent(column_indexer, sheet_catalog)
        logger.debug(f"✅ Created SheetSelectionAgent")
        return agent
    
    def _create_excel_code_generation_agent(self, column_indexer: ColumnIndexerAgent,
                                           session_id: Optional[str], config: Dict[str, Any]) -> ExcelCodeGenerationAgent:
        """Create Excel code generation agent."""
        agent = ExcelCodeGenerationAgent(column_indexer)
        logger.debug(f"✅ Created ExcelCodeGenerationAgent")
        return agent
    
    def _create_excel_execution_agent(self, column_indexer: ColumnIndexerAgent,
                                     session_id: Optional[str], config: Dict[str, Any]) -> ExcelExecutionAgent:
        """Create Excel execution agent."""
        agent = ExcelExecutionAgent(column_indexer)
        logger.debug(f"✅ Created ExcelExecutionAgent")
        return agent
    
    def _create_code_generation_agent(self, session_id: Optional[str], config: Dict[str, Any]):
        """Create code generation agent (function-based)."""
        # CodeGenerationAgent is currently function-based, so we return the function
        from .code_generation import CodeGenerationAgent
        logger.debug(f"✅ Created CodeGenerationAgent function")
        return CodeGenerationAgent
    
    def _create_execution_agent(self, session_id: Optional[str], config: Dict[str, Any]):
        """Create execution agent (function-based)."""
        # ExecutionAgent is currently function-based, so we return the function
        from .execution import ExecutionAgent
        logger.debug(f"✅ Created ExecutionAgent function")
        return ExecutionAgent
    
    def _create_reasoning_agent(self, session_id: Optional[str], config: Dict[str, Any]):
        """Create reasoning agent (function-based)."""
        # ReasoningAgent is currently function-based, so we return the function
        from .reasoning import ReasoningAgent
        logger.debug(f"✅ Created ReasoningAgent function")
        return ReasoningAgent
    
    def _create_data_insight_agent(self, session_id: Optional[str], config: Dict[str, Any]):
        """Create data insight agent (function-based)."""
        # DataInsightAgent is currently function-based, so we return the function
        from .data_analysis import DataInsightAgent
        logger.debug(f"✅ Created DataInsightAgent function")
        return DataInsightAgent
    
    def create_excel_analysis_chain(self, session_id: str, sheet_catalog: Dict[str, Any]) -> Dict[str, Any]:
        """Create a complete Excel analysis agent chain."""
        logger.info(f"🏭 Creating Excel analysis chain for session {session_id}")
        
        # Create agents in dependency order
        catalog_agent = self.create_agent(AgentConfig(
            agent_type=AgentType.EXCEL_CATALOG,
            session_id=session_id
        ))
        
        indexer_agent = self.create_agent(AgentConfig(
            agent_type=AgentType.EXCEL_INDEXER,
            session_id=session_id,
            dependencies={'sheet_catalog': sheet_catalog}
        ))
        
        selection_agent = self.create_agent(AgentConfig(
            agent_type=AgentType.SHEET_SELECTION,
            session_id=session_id,
            dependencies={
                'column_indexer': indexer_agent,
                'sheet_catalog': catalog_agent
            }
        ))
        
        code_gen_agent = self.create_agent(AgentConfig(
            agent_type=AgentType.EXCEL_CODE_GEN,
            session_id=session_id,
            dependencies={'column_indexer': indexer_agent}
        ))
        
        execution_agent = self.create_agent(AgentConfig(
            agent_type=AgentType.EXCEL_EXECUTION,
            session_id=session_id,
            dependencies={'column_indexer': indexer_agent}
        ))
        
        reasoning_agent = self.create_agent(AgentConfig(
            agent_type=AgentType.REASONING,
            session_id=session_id
        ))
        
        return {
            'catalog': catalog_agent,
            'indexer': indexer_agent,
            'selection': selection_agent,
            'code_generation': code_gen_agent,
            'execution': execution_agent,
            'reasoning': reasoning_agent
        }
    
    def create_csv_analysis_chain(self, session_id: str) -> Dict[str, Any]:
        """Create a complete CSV analysis agent chain."""
        logger.info(f"🏭 Creating CSV analysis chain for session {session_id}")
        
        # Create agents
        memory_agent = self.create_agent(AgentConfig(
            agent_type=AgentType.MEMORY,
            session_id=session_id
        ))
        
        code_gen_agent = self.create_agent(AgentConfig(
            agent_type=AgentType.CODE_GENERATION,
            session_id=session_id
        ))
        
        execution_agent = self.create_agent(AgentConfig(
            agent_type=AgentType.EXECUTION,
            session_id=session_id
        ))
        
        reasoning_agent = self.create_agent(AgentConfig(
            agent_type=AgentType.REASONING,
            session_id=session_id
        ))
        
        data_insight_agent = self.create_agent(AgentConfig(
            agent_type=AgentType.DATA_INSIGHT,
            session_id=session_id
        ))
        
        return {
            'memory': memory_agent,
            'code_generation': code_gen_agent,
            'execution': execution_agent,
            'reasoning': reasoning_agent,
            'data_insight': data_insight_agent
        }

# Global agent factory
_agent_factory = None

def get_agent_factory() -> AgentFactory:
    """Get the global agent factory."""
    global _agent_factory
    if _agent_factory is None:
        _agent_factory = AgentFactory()
    return _agent_factory 