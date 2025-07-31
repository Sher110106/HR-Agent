"""Agent modules for the data-analysis application.

This module provides a comprehensive agent system with:
- Base agent classes and coordination
- Code generation and execution
- Memory and reasoning capabilities
- Excel-specific analysis agents
- Error recovery and resilience
- Factory pattern for agent creation
"""

# Core agent components
from .coordinator import get_coordinator, AgentState, AgentContext
from .base_agent import BaseAgent, AgentResult, agent_monitor, get_agent_registry
from .error_recovery import (
    get_error_recovery_manager, 
    with_error_recovery, 
    ErrorSeverity, 
    RecoveryStrategy, 
    ErrorContext
)
from .factory import get_agent_factory, AgentType, AgentConfig

# Memory and context management
from .memory import (
    ColumnMemoryAgent, 
    ConversationMemoryTool, 
    SystemPromptMemoryAgent,
    enhance_prompt_with_context
)

# Code generation and execution
from .code_generation import (
    QueryUnderstandingTool,
    PlotCodeGeneratorTool,
    CodeWritingTool,
    CodeGenerationAgent,
)
from .execution import (
    validate_pandas_code,
    ExecutionAgent,
)

# Reasoning and analysis
from .reasoning import (
    ReasoningAgent,
    ReasoningCurator,
)
from .data_analysis import (
    DataFrameSummaryTool,
    DataInsightAgent,
    ColumnAnalysisAgent,
    AnalyzeColumnBatch,
    AnalyzeAllColumnsAgent,
    smart_date_parser,
    extract_first_code_block,
)

# Excel-specific agents
from .excel_agents import (
    SheetCatalogAgent,
    ColumnIndexerAgent,
    ColumnRef,
    SheetPlan,
)
from .sheet_selection import (
    SheetSelectionAgent,
    DisambiguationQuestion,
)
from .excel_code_generation import (
    ExcelCodeGenerationAgent,
)
from .excel_execution import (
    ExcelExecutionAgent,
)

# Code templates
from .code_templates import (
    get_template_manager,
    ChartType,
    AnalysisType,
    CodeTemplate,
    CodeTemplateManager
)

__all__ = [
    # Core components
    "get_coordinator",
    "AgentState", 
    "AgentContext",
    "BaseAgent",
    "AgentResult",
    "agent_monitor",
    "get_agent_registry",
    "get_error_recovery_manager",
    "with_error_recovery",
    "ErrorSeverity",
    "RecoveryStrategy",
    "ErrorContext",
    "get_agent_factory",
    "AgentType",
    "AgentConfig",
    
    # Memory and context
    "ColumnMemoryAgent",
    "ConversationMemoryTool",
    "SystemPromptMemoryAgent",
    "enhance_prompt_with_context",
    
    # Code generation and execution
    "QueryUnderstandingTool",
    "PlotCodeGeneratorTool",
    "CodeWritingTool",
    "CodeGenerationAgent",
    "validate_pandas_code",
    "ExecutionAgent",
    
    # Reasoning and analysis
    "ReasoningAgent",
    "ReasoningCurator",
    "DataFrameSummaryTool",
    "DataInsightAgent",
    "ColumnAnalysisAgent",
    "AnalyzeColumnBatch",
    "AnalyzeAllColumnsAgent",
    "smart_date_parser",
    "extract_first_code_block",
    
    # Excel-specific agents
    "SheetCatalogAgent",
    "ColumnIndexerAgent",
    "ColumnRef",
    "SheetPlan",
    "SheetSelectionAgent",
    "DisambiguationQuestion",
    "ExcelCodeGenerationAgent",
    "ExcelExecutionAgent",
    
    # Code templates
    "get_template_manager",
    "ChartType",
    "AnalysisType",
    "CodeTemplate",
    "CodeTemplateManager",
] 