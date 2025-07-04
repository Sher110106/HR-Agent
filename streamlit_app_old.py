import os, io, re, logging
import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from datetime import datetime
import chardet
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import time

# Import new utilities
from utils.metrics import (
    metrics_collector, api_call_timer, code_execution_timer, 
    record_error, get_metrics_summary
)
from utils.circuit_breaker import (
    llm_api_breaker, code_execution_breaker, CircuitBreakerError,
    get_all_circuit_breaker_stats
)
from utils.health_monitor import (
    health_monitor, start_health_monitoring, get_health_status,
    get_detailed_health_status, perform_health_check
)
from utils.cache import (
    code_cache, analysis_cache, persistent_cache,
    get_cache_stats, cleanup_all_caches
)
from utils.retry_utils import perform_with_retries

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('data_analysis_agent.log')  # File output
    ]
)
logger = logging.getLogger(__name__)

# === Configuration ===
api_key = os.environ.get("NVIDIA_API_KEY")

# Initialize monitoring systems
start_health_monitoring()
logger.info("🚀 Enhanced Data Analysis Agent starting with monitoring systems")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# === ENHANCED API CALLS WITH MONITORING ===
def make_llm_call(messages: List[Dict], model: str = "nvidia/llama-3.1-nemotron-ultra-253b-v1", 
                  temperature: float = 0.2, max_tokens: int = 4000, stream: bool = False):
    """Make LLM API call with circuit breaker protection and metrics tracking."""
    
    def api_call():
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
    
    try:
        with api_call_timer() as timer:
            response = llm_api_breaker.call(api_call)
            
            # Calculate token usage (approximate)
            prompt_tokens = sum(len(str(msg.get('content', ''))) for msg in messages) // 4  # Rough estimate
            
            if stream:
                # For streaming responses, we can't get exact tokens until complete
                # Set an estimated token count that will be updated when streaming finishes
                timer.set_tokens_used(prompt_tokens)
                logger.debug(f"🤖 LLM streaming call started: ~{prompt_tokens} prompt tokens, model: {model}")
            else:
                if hasattr(response, 'usage') and response.usage:
                    total_tokens = response.usage.total_tokens
                else:
                    total_tokens = prompt_tokens + (max_tokens // 4)  # Rough estimate
                
                timer.set_tokens_used(total_tokens)
                logger.debug(f"🤖 LLM call successful: {total_tokens} tokens, model: {model}")
            
            return response
            
    except CircuitBreakerError as e:
        record_error("circuit_breaker_open", {"model": model, "error": str(e)})
        logger.error(f"🚫 Circuit breaker blocked LLM call: {e}")
        raise
    except Exception as e:
        record_error("llm_api_error", {"model": model, "error": str(e)})
        logger.error(f"❌ LLM API call failed: {e}")
        raise

def execute_code_safely(code: str, local_vars: Dict[str, Any]) -> Tuple[bool, Any, str]:
    """Execute code with circuit breaker protection and metrics tracking."""
    
    def code_execution():
        exec(code, {}, local_vars)
        return local_vars.get('result')
    
    try:
        with code_execution_timer() as timer:
            result = code_execution_breaker.call(code_execution)
            timer.metadata['code_length'] = len(code)
            logger.debug(f"✅ Code execution successful: {len(code)} chars")
            return True, result, ""
            
    except CircuitBreakerError as e:
        record_error("circuit_breaker_open", {"operation": "code_execution", "error": str(e)})
        logger.error(f"🚫 Circuit breaker blocked code execution: {e}")
        return False, None, f"Circuit breaker is open: {e}"
    except Exception as e:
        record_error("code_execution_error", {"error": str(e), "code_length": len(code)})
        logger.error(f"❌ Code execution failed: {e}")
        return False, None, str(e)

# === MEMORY SYSTEM =====================================================

class ColumnMemoryAgent:
    """Memory system for storing AI-generated column descriptions."""
    
    def __init__(self):
        self.column_descriptions = {}
        logger.info("🧠 ColumnMemoryAgent initialized")
    
    def store_column_description(self, column_name: str, description: str):
        """Store description for a specific column."""
        self.column_descriptions[column_name] = description
        logger.info(f"💾 Stored description for column: {column_name}")
    
    def get_column_description(self, column_name: str) -> str:
        """Retrieve description for a specific column."""
        return self.column_descriptions.get(column_name, "")
    
    def get_all_descriptions(self) -> Dict[str, str]:
        """Get all stored column descriptions."""
        return self.column_descriptions.copy()
    
    def has_descriptions(self) -> bool:
        """Check if any column descriptions are stored."""
        return len(self.column_descriptions) > 0
    
    def clear_descriptions(self):
        """Clear all stored descriptions."""
        self.column_descriptions.clear()
        logger.info("🗑️ Cleared all column descriptions")

# === MEMORY TOOLS =====================================================

def ConversationMemoryTool(messages: List[Dict], max_history: int = 4) -> str:
    """Extract relevant conversation history for context in follow-up questions."""
    logger.info(f"🧠 ConversationMemoryTool: Processing {len(messages)} messages (max_history: {max_history})")
    
    if not messages:
        logger.info("💭 No conversation history available")
        return ""
    
    # Get the last few exchanges (user-assistant pairs)
    relevant_messages = messages[-max_history:] if len(messages) > max_history else messages
    
    # Build context string from recent conversation
    context_parts = []
    for msg in relevant_messages:
        role = msg["role"]
        content = msg["content"]
        
        # For assistant messages, extract just the main explanation (no HTML)
        if role == "assistant":
            # Remove HTML thinking sections and keep only the explanation
            import re
            clean_content = re.sub(r'<details[^>]*>.*?</details>', '', content, flags=re.DOTALL)
            clean_content = clean_content.strip()
            if clean_content:
                content = clean_content[:200] + "..." if len(clean_content) > 200 else clean_content
        
        context_parts.append(f"{role.capitalize()}: {content}")
    
    context = "\n".join(context_parts)
    logger.info(f"🧠 Generated conversation context: {len(context)} characters")
    logger.debug(f"Context preview: {context[:300]}...")
    return context

# === CodeGeneration TOOLS ============================================

# ------------------  QueryUnderstandingTool ---------------------------
def QueryUnderstandingTool(query: str, conversation_context: str = "") -> bool:
    """Return True if the query seems to request a visualisation based on keywords and context."""
    logger.info(f"🔍 QueryUnderstandingTool: Analyzing query: '{query}'")
    
    # Enhanced LLM prompt that includes conversation context
    system_prompt = """detailed thinking off. You are a highly specialized query classification assistant. Your sole purpose is to determine if a user's request, including the full conversation context, necessitates a data visualization.

Your analysis must consider:
1.  **Explicit Keywords**: Identify direct requests for a 'plot', 'chart', 'graph', 'diagram', 'visualize', 'show me', etc.
2.  **Implicit Intent**: Infer visualization needs from phrases that imply visual comparison or trend analysis, such as "compare sales across regions," "show the trend over time," "what does the distribution look like," or "can you illustrate the relationship between X and Y?"
3.  **Follow-up Context**: Analyze conversation history. If a user says "make that a bar chart" or "what about for Q3?", you must recognize this refers to a prior analysis or visualization.

Respond with only 'true' for any explicit or implicit visualization request. For all other requests (e.g., data summaries, statistical calculations, data transformations), respond with 'false'. Your output must be a single boolean value."""
    
    user_prompt = f"Query: {query}"
    if conversation_context:
        user_prompt = f"Previous conversation:\n{conversation_context}\n\nCurrent query: {query}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    logger.info("📤 Sending query understanding request to LLM...")
    response = make_llm_call(
        messages=messages,
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1", 
        temperature=0.1,
        max_tokens=1000
    )
    
    # Extract the response and convert to boolean
    intent_response = response.choices[0].message.content.strip().lower()
    result = intent_response == "true"
    logger.info(f"📥 LLM response: '{intent_response}' → Visualization needed: {result}")
    
    return result

# ------------------  PlotCodeGeneratorTool ---------------------------
def PlotCodeGeneratorTool(cols: List[str], query: str, df: pd.DataFrame, conversation_context: str = "", memory_agent: ColumnMemoryAgent = None) -> str:
    """
    Generate a prompt for the LLM to write pandas+matplotlib code for a plot based on the query and columns.
    
    The generated code will return a tuple (fig, data_df) where:
    - fig: matplotlib figure object with professional styling and value labels
    - data_df: pandas DataFrame containing the aggregated data used to create the plot
    
    This enables dual output for enhanced analysis and data export capabilities.
    """
    logger.info(f"📊 PlotCodeGeneratorTool: Generating plot prompt for columns: {cols}")
    
    # Get data types and sample values for better context
    data_info = []
    date_columns = []
    
    for col in cols:
        dtype = str(df[col].dtype)
        
        # Check if we have AI-generated column description
        if memory_agent and memory_agent.has_descriptions():
            stored_description = memory_agent.get_column_description(col)
            if stored_description:
                data_info.append(f"{col} ({dtype}): {stored_description[:200]}...")
                # Check if description mentions date/time
                if any(keyword in stored_description.lower() for keyword in ['date', 'time', 'temporal', 'timestamp']):
                    date_columns.append(col)
                continue
        
        # Fallback to basic analysis if no stored description
        # Check if this might be a date/time column
        is_likely_date = False
        if df[col].dtype == 'object':
            # Look for date-like patterns in the first few non-null values
            sample_values = df[col].dropna().head(3).astype(str).tolist()
            for val in sample_values:
                if any(pattern in val.lower() for pattern in ['am', 'pm', '/', '-', ':', 'time', 'date']):
                    is_likely_date = True
                    break
        
        if is_likely_date:
            date_columns.append(col)
            sample_vals = df[col].dropna().head(3).tolist()
            data_info.append(f"{col} ({dtype}): DATE/TIME column with sample values: {sample_vals}")
        elif df[col].dtype == 'object' or df[col].nunique() < 10:
            unique_vals = df[col].unique()[:5]  # First 5 unique values
            data_info.append(f"{col} ({dtype}): {list(unique_vals)}")
        else:
            data_info.append(f"{col} ({dtype}): numeric range {df[col].min()}-{df[col].max()}")
    
    data_context = "\n".join(data_info)
    logger.debug(f"Data context: {data_context[:300]}...")
    
    # Special instructions for date handling
    date_instructions = ""
    if date_columns:
        logger.info(f"📅 Detected potential date columns: {date_columns}")
        date_instructions = f"""
    
    IMPORTANT - Date/Time Handling:
    - Detected date/time columns: {', '.join(date_columns)}
    - For date parsing, ALWAYS use: pd.to_datetime(df['column_name'], errors='coerce', infer_datetime_format=True)
    - This handles various formats like "5/13/25 12:00 AM", "2025-05-13", etc.
    - For seasonal analysis, extract month: df['Month'] = pd.to_datetime(df['date_col'], errors='coerce').dt.month
    - For seasonal grouping: df['Season'] = df['Month'].map({{12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall'}})
    """
    
    # Build prompt with optional conversation context
    context_section = ""
    if conversation_context:
        context_section = f"""
    Previous conversation context:
    {conversation_context}
    
    """
    
    # Add note about enhanced column information if available
    enhancement_note = ""
    if memory_agent and memory_agent.has_descriptions():
        enhancement_note = """
    
    📈 ENHANCED MODE: Detailed AI-generated column descriptions are available above. 
    Use this rich context to create more sophisticated and contextually relevant visualizations.
    Consider the business meaning, data quality insights, and suggested use cases for each column.
    """
    
    prompt = f"""
    Given DataFrame `df` with columns and data types:
    {data_context}
    {context_section}{date_instructions}{enhancement_note}
    Write Python code using pandas **and matplotlib** (as plt) to answer:
    "{query}"

    CRITICAL NEW REQUIREMENTS - DUAL OUTPUT PLOTS
    ============================================
    Your code MUST return BOTH the plot figure AND the underlying data as a tuple: `result = (fig, data_df)`
    
    1. **Create the figure**: `fig, ax = plt.subplots(figsize=(12,7))`
    2. **Prepare plot data**: Create a DataFrame called `data_df` containing the aggregated/processed data used for the plot
    3. **Build the visualization**: Use ax.bar(), ax.scatter(), ax.plot(), etc.
    4. **Add automatic value labels**: Call helper functions to add value labels to every bar/point
    5. **Format axes**: Apply proper axis formatting and rotation for readability
    6. **Apply professional styling**: Use helper functions for consistent appearance
    7. **Return tuple**: `result = (fig, data_df)` where data_df contains the plot's source data

    Available Helper Functions
    -------------------------
    You have access to these pre-built helper functions (import them as needed):
    - `add_value_labels(ax)` - Automatically adds value labels to bars and points
    - `format_axis_labels(ax, x_rotation=45)` - Rotates and wraps long axis labels
    - `apply_professional_styling(ax, title="", xlabel="", ylabel="")` - Applies consistent styling
    - `get_professional_colors()['colors']` - Returns professional color palette
    - `optimize_figure_size(ax)` - Adjusts figure size based on content

    Mandatory Code Structure
    -----------------------
    ```python
    # 1. Data preparation and aggregation
    data_df = df.groupby('category')['value'].sum().reset_index()  # Example - adapt to your query
    
    # 2. Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 3. Create the plot
    colors = get_professional_colors()['colors']
    ax.bar(data_df['category'], data_df['value'], color=colors[0])
    
    # 4. Add value labels (MANDATORY)
    add_value_labels(ax)
    
    # 5. Format axes (MANDATORY)  
    format_axis_labels(ax, x_rotation=45)
    
    # 6. Apply styling (MANDATORY)
    apply_professional_styling(ax, title="Chart Title", xlabel="X Label", ylabel="Y Label")
    
    # 7. Return both figure and data (MANDATORY)
    result = (fig, data_df)
    ```

    Rules & Available Tools
    ----------------------
    1. Use pandas, matplotlib.pyplot (as plt), and seaborn (as sns) - `pd`, `np`, `df`, `plt`, `sns` are all available in scope.
    2. For date/time columns, prefer smart_date_parser(df, 'column_name') for robust parsing.
    3. For categorical columns, convert to numeric: df['col'].map({{'Yes': 1, 'No': 0}}).
    4. **CRITICAL**: Always return tuple `result = (fig, data_df)` - NOT just fig!
    5. Always call `add_value_labels(ax)` after creating the plot.
    6. Always call `format_axis_labels(ax)` for proper axis formatting.
    7. Always call `apply_professional_styling(ax, title, xlabel, ylabel)`.
    8. Handle missing values with .dropna() before plotting if needed.
    9. Use professional color palette: `colors = get_professional_colors()['colors']`
    10. Wrap code in ```python fence with no explanations.

    Enhanced Plotting Examples:
    - Bar chart: Create data_df → ax.bar() → add_value_labels(ax) → format_axis_labels(ax) → return (fig, data_df)
    - Scatter plot: Create data_df → ax.scatter() → add_value_labels(ax) → format_axis_labels(ax) → return (fig, data_df)  
    - Line chart: Create data_df → ax.plot() → add_value_labels(ax) → format_axis_labels(ax) → return (fig, data_df)
    - Heatmap: Create data_df → sns.heatmap() → format_axis_labels(ax) → return (fig, data_df)
    """
    logger.debug(f"Generated plot prompt: {prompt[:200]}...")
    return prompt

# ------------------  CodeWritingTool ---------------------------------
def CodeWritingTool(cols: List[str], query: str, df: pd.DataFrame, conversation_context: str = "", memory_agent: ColumnMemoryAgent = None) -> str:
    """Generate a prompt for the LLM to write pandas-only code for a data query (no plotting)."""
    logger.info(f"📝 CodeWritingTool: Generating code prompt for columns: {cols}")
    
    # Get data types and sample values for better context
    data_info = []
    date_columns = []
    
    for col in cols:
        dtype = str(df[col].dtype)
        
        # Check if we have AI-generated column description
        if memory_agent and memory_agent.has_descriptions():
            stored_description = memory_agent.get_column_description(col)
            if stored_description:
                data_info.append(f"{col} ({dtype}): {stored_description[:200]}...")
                # Check if description mentions date/time
                if any(keyword in stored_description.lower() for keyword in ['date', 'time', 'temporal', 'timestamp']):
                    date_columns.append(col)
                continue
        
        # Fallback to basic analysis if no stored description
        # Check if this might be a date/time column
        is_likely_date = False
        if df[col].dtype == 'object':
            # Look for date-like patterns in the first few non-null values
            sample_values = df[col].dropna().head(3).astype(str).tolist()
            for val in sample_values:
                if any(pattern in val.lower() for pattern in ['am', 'pm', '/', '-', ':', 'time', 'date']):
                    is_likely_date = True
                    break
        
        if is_likely_date:
            date_columns.append(col)
            sample_vals = df[col].dropna().head(3).tolist()
            data_info.append(f"{col} ({dtype}): DATE/TIME column with sample values: {sample_vals}")
        elif df[col].dtype == 'object' or df[col].nunique() < 10:
            unique_vals = df[col].unique()[:5]  # First 5 unique values
            data_info.append(f"{col} ({dtype}): {list(unique_vals)}")
        else:
            data_info.append(f"{col} ({dtype}): numeric range {df[col].min()}-{df[col].max()}")
    
    data_context = "\n".join(data_info)
    logger.debug(f"Data context: {data_context[:300]}...")
    
    # Special instructions for date handling
    date_instructions = ""
    if date_columns:
        logger.info(f"📅 Detected potential date columns: {date_columns}")
        date_instructions = f"""
    
    IMPORTANT - Date/Time Handling:
    - Detected date/time columns: {', '.join(date_columns)}
    - For date parsing, ALWAYS use: pd.to_datetime(df['column_name'], errors='coerce', infer_datetime_format=True)
    - This handles various formats like "5/13/25 12:00 AM", "2025-05-13", etc.
    - For seasonal analysis, extract month: df['Month'] = pd.to_datetime(df['date_col'], errors='coerce').dt.month
    - For seasonal grouping: df['Season'] = df['Month'].map({{12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall'}})
    """
    
    # Build prompt with optional conversation context
    context_section = ""
    if conversation_context:
        context_section = f"""
    Previous conversation context:
    {conversation_context}
    
    """
    
    # Add note about enhanced column information if available
    enhancement_note = ""
    if memory_agent and memory_agent.has_descriptions():
        enhancement_note = """
    
    📈 ENHANCED MODE: Detailed AI-generated column descriptions are available above. 
    Use this rich context to create more sophisticated and contextually relevant analysis.
    Consider the business meaning, data quality insights, and suggested use cases for each column.
    """
    
    prompt = f"""
    Given DataFrame `df` with columns and data types:
    {data_context}
    {context_section}{date_instructions}{enhancement_note}
    Write Python code (pandas **only**, no plotting) to answer:
    "{query}"

    Rules & Available Tools
    ----------------------
         1. Use pandas operations on `df` only - `pd`, `np`, and `df` are available in scope.
    2. For date/time columns, prefer smart_date_parser(df, 'column_name') for robust parsing.
    3. For categorical columns (like Yes/No), convert to numeric first: df['col'].map({{'Yes': 1, 'No': 0}}).
    4. For correlation analysis, use df[['col1', 'col2']].corr().iloc[0, 1] for cleaner results.
    5. For groupby operations, handle missing values with .dropna() if needed.
    6. Always assign the final result to `result` variable.
    7. Ensure result is a clear, interpretable value (float, dict, or small DataFrame).
    8. Wrap code in a single ```python fence with no explanations.

    Example Patterns:
    - Correlation: result = df[['col1', 'col2']].corr().iloc[0, 1]
    - Seasonal analysis: df['Season'] = df['Month'].map({{...}})
    - Summary stats: result = df.groupby('category')['value'].agg(['mean', 'std'])
    - Top N values: top_indices = series.nlargest(n).index; top_rows = df.loc[top_indices]
    - Avoid chaining errors: Use intermediate variables for complex operations
    - Safe DataFrame slicing: df_copy = df[condition].copy() to avoid SettingWithCopyWarning
    """
    logger.debug(f"Generated code prompt: {prompt[:200]}...")
    return prompt

# === CodeGenerationAgent ==============================================

def CodeGenerationAgent(query: str, df: pd.DataFrame, chat_history: List[Dict] = None, memory_agent: ColumnMemoryAgent = None, retry_context: str = None):
    """Selects the appropriate code generation tool and gets code from the LLM for the user's query."""
    logger.info(f"🤖 CodeGenerationAgent: Processing query: '{query}'")
    logger.info(f"📊 DataFrame info: {len(df)} rows, {len(df.columns)} columns")
    
    if retry_context:
        logger.info(f"🔄 Retry mode activated with context: {retry_context[:200]}...")
    
    # Extract conversation context for better understanding
    conversation_context = ""
    if chat_history:
        conversation_context = ConversationMemoryTool(chat_history)
        logger.info(f"🧠 Using conversation context: {len(conversation_context)} characters")
    
    # Add column descriptions context if available
    if memory_agent and memory_agent.has_descriptions():
        column_context = f"\n\nAI-Generated Column Descriptions:\n"
        for col in df.columns:
            desc = memory_agent.get_column_description(col)
            if desc:
                column_context += f"- {col}: {desc[:150]}...\n"
        conversation_context += column_context
        logger.info(f"🧠 Enhanced context with column descriptions: {len(conversation_context)} characters")
    
    # -------------------------------------------------------------
    # 0. Try cache: if a similar query has already produced working code,
    #    reuse it to avoid an expensive LLM round-trip.
    # -------------------------------------------------------------
    cached_entry = None
    if retry_context is None:  # only use cache on first attempt
        cached_entry = code_cache.get_similar_code_result(query, df.columns.tolist())
        if cached_entry:
            cached_code, _cached_result = cached_entry
            logger.info("♻️ Retrieved cached code for similar query – skipping LLM generation")
            # We still need to know whether this query likely needs a plot so the
            # caller can set up the execution environment correctly.
            should_plot = QueryUnderstandingTool(query, conversation_context)
            return cached_code, should_plot, ""
    
    # -------------------------------------------------------------
    # 1. No cache (or we are in retry mode) → proceed with normal LLM generation
    # -------------------------------------------------------------
    
    should_plot = QueryUnderstandingTool(query, conversation_context)
    
    # Add retry context if available
    if retry_context:
        conversation_context += f"\n\nPREVIOUS ERROR TO AVOID:\n{retry_context}\n\nPlease fix this error and generate corrected code."
    
    prompt = PlotCodeGeneratorTool(df.columns.tolist(), query, df, conversation_context, memory_agent) if should_plot else CodeWritingTool(df.columns.tolist(), query, df, conversation_context, memory_agent)

    messages = [
        {"role": "system", "content": """detailed thinking off. You are a senior data scientist with expertise in pandas, matplotlib, and seaborn for statistical analysis and visualization. Write clean, efficient, production-ready code. Focus on:

1. CORRECTNESS: Ensure proper variable scoping (pd, np, df, plt, sns are available)
2. ROBUSTNESS: Handle missing values and edge cases
3. CLARITY: Use descriptive variable names and clear logic
4. EFFICIENCY: Prefer vectorized operations over loops
5. BEST PRACTICES: Follow pandas conventions and leverage seaborn for enhanced visualizations
6. AESTHETICS: Use seaborn's statistical plotting capabilities for professional-looking charts

Output ONLY a properly-closed ```python code block. Use smart_date_parser() for date parsing. Assign final result to 'result' variable."""},
        {"role": "user", "content": prompt}
    ]

    logger.info("📤 Sending code generation request to LLM...")
    response = make_llm_call(
        messages=messages,
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        temperature=0.2,
        max_tokens=4000
    )

    full_response = response.choices[0].message.content
    logger.info(f"📥 LLM full response length: {len(full_response)} characters")
    
    code = extract_first_code_block(full_response)
    logger.info(f"✂️ Extracted code block length: {len(code)} characters")
    logger.info(f"💻 Generated code:\n{code}")
    
    return code, should_plot, ""

# === ExecutionAgent ====================================================

def validate_pandas_code(code: str) -> tuple[list, str]:
    """Validate code for common pandas errors and return warnings and corrected code."""
    warnings = []
    corrected_code = code
    
    # Fix dangerous method chaining patterns
    if ".nlargest(" in code and ".idxmax(" in code:
        warnings.append("🔧 Auto-fixed: Method chaining error - converted .nlargest().idxmax() to proper syntax")
        # This is a complex fix that would need regex, for now just warn
        warnings.append("⚠️ Potential method chaining error: .nlargest() returns a Series, not indices. Use .nlargest(n).index")
    
    if ".nsmallest(" in code and ".idxmin(" in code:
        warnings.append("⚠️ Potential method chaining error: .nsmallest() returns a Series, not indices. Use .nsmallest(n).index")
    
    # Check for integer method calls  
    if "int(" in code and any(f".{method}(" in code for method in ["idxmax", "idxmin", "nlargest", "nsmallest"]):
        warnings.append("⚠️ Potential error: Calling pandas methods on integers. Check your method chaining.")
    
    # Fix unsafe DataFrame slicing
    if "[df[" in code and ".copy()" not in code and "=" in code:
        warnings.append("🔧 Auto-suggestion: Consider using .copy() when creating DataFrame subsets to avoid warnings")
    
    return warnings, corrected_code

def ExecutionAgent(code: str, df: pd.DataFrame, should_plot: bool):
    """
    Executes the generated code in a controlled environment and returns the result or error message.
    
    For plotting code, this may return either:
    - Legacy format: single matplotlib figure/axes object  
    - New dual-output format: tuple of (fig, data_df) where data_df contains the plot's source data
    
    Helper functions are available in the execution environment for enhanced plotting.
    """
    logger.info(f"⚡ ExecutionAgent: Executing code (plot mode: {should_plot})")
    logger.info(f"🔧 Code to execute:\n{code}")
    
    # Validate code for common pandas errors
    validation_warnings, corrected_code = validate_pandas_code(code)
    if validation_warnings:
        logger.warning(f"🔍 Code validation warnings: {validation_warnings}")
        # Use corrected code if available
        if corrected_code != code:
            logger.info(f"🔧 Using corrected code")
            code = corrected_code
    
    env = {"pd": pd, "np": np, "df": df, "smart_date_parser": smart_date_parser}
    if should_plot:
        plt.rcParams["figure.dpi"] = 100  # Set default DPI for all figures
        env["plt"] = plt
        env["sns"] = sns
        env["io"] = io
        
        # Import helper functions for enhanced plotting
        try:
            from utils.plot_helpers import (
                add_value_labels, format_axis_labels,
                apply_professional_styling, get_professional_colors
            )
            env["add_value_labels"] = add_value_labels
            env["format_axis_labels"] = format_axis_labels
            env["apply_professional_styling"] = apply_professional_styling 
            env["get_professional_colors"] = get_professional_colors
            logger.info("🎨 Plot environment set up with matplotlib, seaborn, and helper functions")
        except ImportError as e:
            logger.warning(f"⚠️ Could not import plot helpers: {e}")
            logger.info("🎨 Plot environment set up with matplotlib and seaborn only")
    
    try:
        logger.info("🚀 Executing code...")
        # Use env as both globals and locals to ensure proper variable access
        exec(code, env, env)
        result = env.get("result", None)
        
        if result is not None:
            result_type = type(result).__name__
            
            # Check for new tuple format (fig, data_df)
            if isinstance(result, tuple) and len(result) == 2:
                fig, data = result
                if isinstance(fig, (plt.Figure, plt.Axes)) and isinstance(data, pd.DataFrame):
                    logger.info(f"✅ Execution successful: Tuple with plot figure and DataFrame ({len(data)} rows, {len(data.columns)} columns)")
                    logger.info("🎯 New dual-output format detected - plot with underlying data")
                    return result  # Return the tuple as-is
                else:
                    logger.warning(f"⚠️ Tuple result detected but not in expected (fig, data_df) format: {type(fig)}, {type(data)}")
            
            # Handle legacy single results
            if isinstance(result, pd.DataFrame):
                logger.info(f"✅ Execution successful: DataFrame with {len(result)} rows, {len(result.columns)} columns")
            elif isinstance(result, pd.Series):
                logger.info(f"✅ Execution successful: Series with {len(result)} elements")
            elif isinstance(result, (plt.Figure, plt.Axes)):
                logger.info(f"✅ Execution successful: {result_type} plot object (legacy format)")
            else:
                logger.info(f"✅ Execution successful: {result_type} = {str(result)[:100]}...")
        else:
            logger.warning("⚠️ Code executed but no 'result' variable found")
            
        return result
    except Exception as exc:
        import traceback
        error_msg = f"Error executing code: {exc}"
        full_traceback = traceback.format_exc()
        logger.error(f"❌ Execution failed: {error_msg}")
        logger.debug(f"📋 Full traceback:\n{full_traceback}")
        
        # Provide more specific error guidance
        if "not defined" in str(exc):
            error_msg += f"\n💡 Tip: Available variables are: {list(env.keys())}"
        elif "KeyError" in str(exc):
            error_msg += f"\n💡 Tip: Available columns are: {list(df.columns)}"
        elif "has no attribute" in str(exc) and any(method in str(exc) for method in ['idxmax', 'idxmin', 'nlargest', 'nsmallest']):
            error_msg += f"\n💡 Tip: Method chaining error detected. For top N values, use: series.nlargest(n).index, then df.loc[indices]"
        elif "SettingWithCopyWarning" in str(exc):
            error_msg += f"\n💡 Tip: Use .copy() when creating DataFrame subsets or .loc[] for safe assignments"
        elif "'int' object has no attribute" in str(exc):
            error_msg += f"\n💡 Tip: Check your method chaining - you may be calling a method on an integer instead of a pandas object"
        
        return error_msg

# === ReasoningCurator TOOL =========================================
def ReasoningCurator(query: str, result: Any) -> str:
    """Builds and returns the LLM prompt for reasoning about the result."""
    logger.info(f"🧠 ReasoningCurator: Creating reasoning prompt for result type: {type(result).__name__}")
    
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))
    is_dual_output = isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], (plt.Figure, plt.Axes)) and isinstance(result[1], pd.DataFrame)

    if is_dual_output:
        # Handle new dual-output format (fig, data_df)
        fig, data_df = result
        
        # Get plot description
        title = ""
        if isinstance(fig, plt.Figure):
            title = fig._suptitle.get_text() if fig._suptitle else ""
        elif isinstance(fig, plt.Axes):
            title = fig.get_title()
        
        plot_desc = f"[Enhanced Plot with Data: {title or 'Professional Chart'}]"
        
        # Get data description with actual values
        if len(data_df) <= 20:
            data_desc = f"Source Data ({len(data_df)} rows, {len(data_df.columns)} columns):\n{data_df.to_string()}"
        else:
            data_desc = f"Source Data ({len(data_df)} rows, {len(data_df.columns)} columns):\n{data_df.head(10).to_string()}\n... [showing first 10 rows]"
        
        desc = f"{plot_desc}\n\n{data_desc}"
        logger.info(f"📊 Result is dual-output: plot with {len(data_df)} data rows")
        
    elif is_error:
        desc = result
        result_summary = "Error occurred during execution"
        logger.info("❌ Result is an error")
    elif is_plot:
        title = ""
        if isinstance(result, plt.Figure):
            title = result._suptitle.get_text() if result._suptitle else ""
        elif isinstance(result, plt.Axes):
            title = result.get_title()
        desc = f"[Plot Object: {title or 'Chart'}]"
        result_summary = desc
        logger.info(f"📊 Result is a plot: {desc}")
    else:
        # For data results, provide both full and summary descriptions
        full_desc = str(result)
        if len(full_desc) > 1000:
            desc = full_desc[:1000] + "... [truncated]"
        else:
            desc = full_desc
        
        # Create a more detailed summary for non-plot results
        if isinstance(result, list):
            result_summary = f"List with {len(result)} items: {result[:10]}{'...' if len(result) > 10 else ''}"
        elif isinstance(result, dict):
            result_summary = f"Dictionary with {len(result)} keys: {list(result.keys())[:5]}{'...' if len(result) > 5 else ''}"
        elif isinstance(result, (pd.DataFrame, pd.Series)):
            if isinstance(result, pd.DataFrame):
                result_summary = f"DataFrame with {len(result)} rows and {len(result.columns)} columns:\n{result.head().to_string()}"
            else:
                result_summary = f"Series with {len(result)} values:\n{result.head().to_string()}"
        else:
            result_summary = str(result)
        
        logger.info(f"📄 Result description: {desc[:100]}...")

    if is_dual_output:
        prompt = f'''
        The user asked: "{query}".
        
        ENHANCED DUAL-OUTPUT ANALYSIS RESULT:
        {desc}
        
        CRITICAL INSTRUCTIONS FOR DUAL-OUTPUT:
        1. You have BOTH a professional visualization AND the complete source data
        2. Your response MUST include analysis of BOTH components:
           - Describe what the chart/visualization shows visually
           - Present and analyze the actual numerical data from the source table
        3. Provide comprehensive analysis using the specific data values shown
        4. Cross-reference between the visual patterns and the underlying numbers
        
        Your response must be structured and include:
        
        **VISUALIZATION ANALYSIS:**
        [Describe what the chart shows visually - trends, patterns, comparisons]
        
        **DATA INSIGHTS:**
        [Present the actual numerical findings from the source data table with specific values]
        
        **COMBINED INTERPRETATION:**
        [Explain how the visual and numerical data work together to tell the story]
        
        **BUSINESS RECOMMENDATIONS:**
        [Provide actionable insights based on both the chart patterns and specific data values]
        
        **NEXT STEPS:**
        [Suggest follow-up analyses leveraging both visual and data components]
        
        Remember: Always reference the specific data values from the source table in your analysis.'''
    elif is_plot:
        prompt = f'''
        The user asked: "{query}".
        Below is a description of the plot result:
        {desc}
        Explain in 2–3 concise sentences what the chart shows (no code talk).'''
    else:
        prompt = f'''
        The user asked: "{query}".
        
        EXECUTION RESULT:
        {desc}
        
        CRITICAL INSTRUCTIONS:
        1. Your final response MUST include the complete actual results from the analysis
        2. If it's a list (like employee IDs), show the full list of IDs in your response
        3. If it's numbers/calculations, show the actual values
        4. If it's a DataFrame, describe the key data points and patterns
        5. Be comprehensive and detailed - provide thorough analysis, not brief summaries
        
        Your response must be structured and include:
        
        **ANALYSIS RESULTS:**
        [Present the actual findings with specific values, lists, numbers - the raw results]
        
        **BUSINESS INTERPRETATION:**
        [Explain what these specific results mean in business context]
        
        **ACTIONABLE RECOMMENDATIONS:**
        [Provide concrete recommendations based on the actual findings]
        
        **SUGGESTED NEXT STEPS:**
        [Propose specific follow-up actions or analyses]
        
        Remember: Always include the actual data/results in your response, not just descriptions of what type of result it is.'''
    
    logger.debug(f"Generated reasoning prompt: {prompt[:200]}...")
    return prompt

# === ReasoningAgent (streaming) =========================================
def ReasoningAgent(query: str, result: Any):
    """Streams the LLM's reasoning about the result (plot or value) and extracts model 'thinking' and final explanation."""
    logger.info(f"🧠 ReasoningAgent: Starting reasoning for query: '{query}'")
    
    prompt = ReasoningCurator(query, result)
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))

    logger.info("📤 Sending reasoning request to LLM (streaming)...")
    # Streaming LLM call
    messages = [
        {"role": "system", "content": """detailed thinking on. You are an expert data analyst and business strategist. Your primary role is to translate raw code outputs (plots, data, or errors) into comprehensive, actionable business insights for decision-makers.

Follow this structured reasoning process:

1.  **INTERPRET THE RESULT**:
    *   **What is it?** Start by identifying the type of output and INCLUDE THE ACTUAL RESULTS/VALUES in your analysis.
    *   **What does it show?** Present the specific findings with actual numbers, lists, or data points. Don't just describe the type - show the content.

2.  **CONTEXTUALIZE THE FINDINGS**:
    *   **Why is this important?** Connect the specific results back to the user's original query and business context.
    *   **Leverage Business Context**: Explain what these concrete findings mean for the business using the AI-generated column descriptions and dataset context.

3.  **IDENTIFY ACTIONABLE INSIGHTS**:
    *   **What patterns emerge?** Identify clear trends, correlations, outliers, or relationships that are apparent in the data.
    *   **What are the implications?** Discuss the business implications of these findings.

4.  **PROVIDE NEXT STEPS**:
    *   **What questions should be asked next?** Suggest logical follow-up analyses that would build upon these findings.
    *   **What actions should be taken?** Recommend specific business actions based on the insights.

**CRITICAL REQUIREMENTS**:
- ALWAYS include the actual data values, numbers, and results in your analysis - not just descriptions
- Reference the specific columns and their business meanings using the enhanced column descriptions
- Connect findings to practical business outcomes and decisions
- Be comprehensive yet actionable
- If there are errors, explain what went wrong and suggest alternatives

# SPECIAL INSTRUCTIONS FOR DUAL-OUTPUT PLOTS:
When analyzing tuple results (fig, data_df), you have access to both the visual representation AND the underlying aggregated data. Your analysis should:
1. Describe what the plot shows visually
2. Reference the specific numbers from the data_df (the source data)
3. Explain both the visual patterns AND the numerical findings
4. Compare the visual impression with the actual data values

For example: "The bar chart shows Product A leading in sales, and the underlying data confirms this with Product A generating $125,000 compared to Product B's $89,000 and Product C's $67,000."

-- Include specific recommendations based on the actual data found

Begin by streaming your detailed analytical process within `<think>...</think>` tags. After your thought process is complete, provide a comprehensive, detailed explanation to the user outside the tags that includes the actual results."""},
        {"role": "user", "content": prompt}
    ]
    
    response = make_llm_call(
        messages=messages,
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        temperature=0.3,
        max_tokens=5000,
        stream=True
    )

    # Stream and display thinking
    thinking_placeholder = st.empty()
    full_response = ""
    thinking_content = ""
    in_think = False
    token_count = 0

    logger.info("📥 Starting to receive streaming response...")
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            token_count += 1
            full_response += token

            # Simple state machine to extract <think>...</think> as it streams
            if "<think>" in token:
                in_think = True
                token = token.split("<think>", 1)[1]
                logger.debug("🤔 Started thinking section")
            if "</think>" in token:
                token = token.split("</think>", 1)[0]
                in_think = False
                logger.debug("🤔 Ended thinking section")
            if in_think or ("<think>" in full_response and not "</think>" in full_response):
                thinking_content += token
                thinking_placeholder.markdown(
                    f'<details class="thinking" open><summary>🤔 Model Thinking</summary><pre>{thinking_content}</pre></details>',
                    unsafe_allow_html=True
                )

    logger.info(f"📥 Streaming complete: {token_count} tokens received")
    logger.info(f"🧠 Thinking content length: {len(thinking_content)} characters")

    # After streaming, extract final reasoning (outside <think>...</think>)
    cleaned = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
    logger.info(f"📄 Final reasoning length: {len(cleaned)} characters")
    logger.debug(f"Final reasoning: {cleaned[:200]}...")
    
    return thinking_content, cleaned

# === DataFrameSummary TOOL (pandas only) =========================================
def DataFrameSummaryTool(df: pd.DataFrame) -> str:
    """Generate a summary prompt string for the LLM based on the DataFrame."""
    logger.info(f"📊 DataFrameSummaryTool: Summarizing DataFrame with {len(df)} rows, {len(df.columns)} columns")
    
    missing_values = df.isnull().sum().to_dict()
    data_types = df.dtypes.to_dict()
    
    logger.info(f"📋 Data types: {data_types}")
    logger.info(f"❓ Missing values: {missing_values}")
    
    prompt = f"""
        Given a dataset with {len(df)} rows and {len(df.columns)} columns:
        Columns: {', '.join(df.columns)}
        Data types: {data_types}
        Missing values: {missing_values}

        Provide:
        1. A brief description of what this dataset contains
        2. 3-4 possible data analysis questions that could be explored
        Keep it concise and focused."""
    return prompt

# === DataInsightAgent (upload-time only) ===============================

def DataInsightAgent(df: pd.DataFrame) -> str:
    """Uses the LLM to generate a brief summary and possible questions for the uploaded dataset."""
    logger.info(f"💡 DataInsightAgent: Generating insights for uploaded dataset")
    
    prompt = DataFrameSummaryTool(df)
    try:
        logger.info("📤 Sending dataset insight request to LLM...")
        response = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
            messages=[
                {"role": "system", "content": """detailed thinking off. You are an automated data exploration assistant. Your task is to analyze the metadata of a newly uploaded dataset and provide a concise, structured, and actionable "first look" summary to orient the user.

Using the provided dataset metadata (row/column counts, column names, data types, missing values), generate the following in a clear, structured format:

1.  **Dataset Overview**:
    *   A one-sentence summary of what the dataset likely contains based on its column names and structure. (e.g., "This appears to be a customer sales dataset tracking transactions, products, and demographic information.").

2.  **Key Characteristics & Data Quality Snapshot**:
    *   Provide a bulleted list of key facts: Total Rows, Total Columns.
    *   Highlight any immediate data quality concerns, such as columns with a high percentage of missing values. (e.g., "Data Quality Note: The 'Region' column is missing 45% of its values, which may impact regional analysis.").

3.  **Suggested Analysis Questions (Starters)**:
    *   Provide 3-4 specific, relevant data analysis questions that are tailored to the dataset's structure and likely business context. These should inspire the user.
    *   Example Questions:
        *   "What is the overall sales trend over time?"
        *   "Which product category generates the most revenue?"
        *   "Is there a correlation between customer age and purchase amount?"

4.  **Recommended First Steps**:
    *   Suggest 1-2 immediate actions the user could take, such as analyzing a key column or handling missing data. (e.g., "Recommendation: Start by analyzing the 'Sales' column to understand its distribution, or ask to visualize the 'Transaction_Date' to see trends.").

Keep the entire response concise, business-focused, and free of technical jargon."""},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4000
        )
        insights = response.choices[0].message.content
        logger.info(f"📥 Generated insights length: {len(insights)} characters")
        logger.debug(f"Insights: {insights[:200]}...")
        return insights
    except Exception as exc:
        error_msg = f"Error generating dataset insights: {exc}"
        logger.error(f"❌ DataInsightAgent failed: {error_msg}")
        return error_msg

# === COLUMN ANALYSIS TOOLS ============================================

def ColumnAnalysisAgent(df: pd.DataFrame, column_name: str) -> str:
    """Generate AI-powered analysis and description for a specific column."""
    logger.info(f"🔍 ColumnAnalysisAgent: Analyzing column '{column_name}'")
    
    col_data = df[column_name]
    dtype = str(col_data.dtype)
    
    # Gather comprehensive column statistics
    stats = {
        'total_rows': len(col_data),
        'non_null_count': col_data.count(),
        'null_count': col_data.isnull().sum(),
        'null_percentage': (col_data.isnull().sum() / len(col_data)) * 100,
        'unique_count': col_data.nunique(),
        'data_type': dtype
    }
    
    # Get sample values (handle different data types)
    if col_data.dtype == 'object':
        # For text/categorical data
        value_counts = col_data.value_counts().head(10)
        sample_values = value_counts.index.tolist()
        value_distribution = dict(value_counts)
    else:
        # For numeric data
        sample_values = col_data.dropna().head(10).tolist()
        if col_data.dtype in ['int64', 'float64']:
            stats.update({
                'min_value': col_data.min(),
                'max_value': col_data.max(),
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std_dev': col_data.std()
            })
        value_distribution = {}
    
    # Check for potential date/time patterns
    is_likely_date = False
    if col_data.dtype == 'object':
        sample_str_values = col_data.dropna().head(5).astype(str).tolist()
        date_patterns = ['/', '-', ':', 'am', 'pm', 'time', 'date', '20', '19']
        for val in sample_str_values:
            if any(pattern in val.lower() for pattern in date_patterns):
                is_likely_date = True
                break
    
    # Build comprehensive prompt for AI analysis
    prompt = f"""
    Analyze this column from a dataset and provide a comprehensive description:
    
    Column Name: {column_name}
    Data Type: {dtype}
    Total Rows: {stats['total_rows']}
    Non-null Values: {stats['non_null_count']} ({100 - stats['null_percentage']:.1f}%)
    Missing Values: {stats['null_count']} ({stats['null_percentage']:.1f}%)
    Unique Values: {stats['unique_count']}
    
    Sample Values: {sample_values[:10]}
    
    {f"Statistical Summary: Min={stats.get('min_value', 'N/A')}, Max={stats.get('max_value', 'N/A')}, Mean={stats.get('mean', 'N/A'):.2f}, Median={stats.get('median', 'N/A')}" if 'mean' in stats else ""}
    
    {f"Value Distribution (top categories): {value_distribution}" if value_distribution else ""}
    
    {"IMPORTANT: This appears to be a date/time column based on sample values." if is_likely_date else ""}
    
    Please provide a detailed analysis including:
    1. What this column likely represents (business context)
    2. Data quality assessment (completeness, consistency)
    3. Potential use cases for analysis (what insights could be derived)
    4. Any data preprocessing recommendations
    5. Relationships this column might have with other typical business metrics
    
    Keep the response comprehensive but concise (3-4 paragraphs).
    """
    
    try:
        logger.info("📤 Sending column analysis request to LLM...")
        response = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
            messages=[
                {"role": "system", "content": """You are a senior data analyst and data steward, with deep expertise in business intelligence, data quality assessment, and data modeling. Your task is to perform a comprehensive, actionable analysis of a single data column and output it in a structured format suitable for being stored in a memory system.

For the given column, use the provided metadata (name, data type, sample values, missing values) and conversation context to generate the following analysis:

1.  **Business Context & Definition**:
    *   Describe what this column most likely represents in a business context. Be specific. (e.g., "The 'Employee_Satisfaction_Score' column likely represents a rating from 1 to 5 provided by employees during their annual performance review.").

2.  **Data Quality Assessment**:
    *   **Completeness**: State the percentage of missing values.
    *   **Consistency & Validity**: Assess the format and uniformity of the data. Are there mixed types? Unexpected values? (e.g., "Values are consistently integers from 1-5, with no invalid entries detected.").
    *   **Potential Issues**: Note any potential problems for analysis. (e.g., "A high concentration of scores at '3' may indicate central tendency bias.").

3.  **Analytical Use Cases**:
    *   Suggest 2-3 specific analytical applications for this column. (e.g., "1. Trend Analysis: Track satisfaction scores over time to measure the impact of HR initiatives. 2. Segmentation: Compare scores across different departments or job levels.").

4.  **Preprocessing Recommendations**:
    *   Provide specific, actionable data cleaning or transformation steps. (e.g., "For analysis, this categorical column should be one-hot encoded. Missing values could be imputed with the median score of '3' if necessary.").

5.  **Potential Relationships & Insights**:
    *   Hypothesize potential relationships with other columns or key business metrics. (e.g., "This column could be negatively correlated with 'Employee_Turnover_Rate' and positively correlated with 'Productivity_Score'.").

Output the analysis in a clear, structured format (e.g., using Markdown headings for each section) to ensure it is easily parsable for storage in the ColumnMemoryAgent."""},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=3000
        )
        
        analysis = response.choices[0].message.content
        logger.info(f"📥 Generated column analysis: {len(analysis)} characters")
        return analysis
        
    except Exception as exc:
        error_msg = f"Error analyzing column '{column_name}': {exc}"
        logger.error(f"❌ Column analysis failed: {error_msg}")
        return error_msg

def _analyze_single_column(df: pd.DataFrame, column: str) -> Tuple[str, str]:
    """Helper that actually calls the LLM for column analysis."""
    description = ColumnAnalysisAgent(df, column)
    return column, description

def AnalyzeColumnBatch(df: pd.DataFrame, column: str) -> Tuple[str, str]:
    """Single column analysis with shared retry/back-off handling."""

    try:
        return perform_with_retries(
            _analyze_single_column,
            df,
            column,
            max_retries=2,
            base_delay=1.0,
            retry_exceptions=(Exception,),  # broaden; ColumnAnalysisAgent already filters
        )
    except Exception as e:
        logger.error("❌ Failed to analyze column %s: %s", column, e)
        return column, f"Error analyzing column: {e}"

def AnalyzeAllColumnsAgent(df: pd.DataFrame, memory_agent: ColumnMemoryAgent) -> str:
    """Analyze all columns in the dataset in parallel and store descriptions in memory."""
    import time
    start_time = time.time()
    
    logger.info(f"📊 AnalyzeAllColumnsAgent: Starting PARALLEL analysis of {len(df.columns)} columns")
    
    columns = df.columns.tolist()
    total_columns = len(columns)
    
    # Create progress tracking
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    completed_count = 0
    analysis_results = {}
    
    # Use ThreadPoolExecutor for parallel API calls
    # Intelligent worker count: balance speed vs rate limits
    import os
    cpu_cores = os.cpu_count() or 4
    if total_columns <= 5:
        max_workers = total_columns  # Small datasets: full parallelism
    elif total_columns <= 20:
        max_workers = min(6, total_columns)  # Medium datasets: moderate parallelism
    else:
        # Large datasets: bound by CPU cores to avoid oversubscription
        max_workers = min(cpu_cores, 8)
    
    logger.info(f"🚀 Using {max_workers} parallel workers for {total_columns} columns")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all column analysis tasks
        future_to_column = {
            executor.submit(AnalyzeColumnBatch, df, column): column 
            for column in columns
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_column):
            column = future_to_column[future]
            completed_count += 1
            
            try:
                column_name, description = future.result()
                analysis_results[column_name] = description
                
                # Store in memory
                memory_agent.store_column_description(column_name, description)
                
                # Update progress
                progress = completed_count / total_columns
                progress_placeholder.progress(
                    progress, 
                    text=f"Analyzed {completed_count}/{total_columns} columns"
                )
                
                # Show current status
                status_placeholder.info(f"✅ Completed: **{column_name}** ({completed_count}/{total_columns})")
                
                logger.info(f"✅ Completed analysis for column: {column_name} ({completed_count}/{total_columns})")
                
            except Exception as e:
                logger.error(f"❌ Error processing future for column {column}: {e}")
                analysis_results[column] = f"Error: {str(e)}"
    
    # Clear progress indicators
    progress_placeholder.empty()
    status_placeholder.empty()
    
    # Calculate performance metrics
    end_time = time.time()
    total_time = end_time - start_time
    successful_analyses = sum(1 for desc in analysis_results.values() if not desc.startswith("Error"))
    failed_analyses = total_columns - successful_analyses
    
    # Estimate sequential time (rough calculation)
    estimated_sequential_time = total_time * max_workers
    time_saved = estimated_sequential_time - total_time
    speedup_factor = estimated_sequential_time / total_time if total_time > 0 else 1
    
    # Create clean, concise summary
    successful_columns = [col for col in columns if col in analysis_results and not analysis_results[col].startswith("Error")]
    failed_columns = [col for col in columns if col in analysis_results and analysis_results[col].startswith("Error")]
    
    summary = f"""🧠 **Column Analysis Complete!**

✅ **Successfully analyzed {successful_analyses}/{total_columns} columns** in {total_time:.1f}s

{f"⚠️ **Failed columns:** {', '.join([f'`{col}`' for col in failed_columns])}" if failed_columns else ""}

🚀 **Enhanced AI mode enabled!** The chat can now provide deeper insights using detailed column understanding."""
    
    logger.info(f"✅ PARALLEL column analysis complete: {successful_analyses} successful, {failed_analyses} failed in {total_time:.1f}s ({speedup_factor:.1f}x speedup)")
    return summary

# === Helpers ===========================================================

def smart_date_parser(df, column_name):
    """
    Smart date parsing function that handles various date formats robustly.
    This function is made available in the execution environment.
    """
    try:
        # First try pandas' automatic inference
        parsed = pd.to_datetime(df[column_name], errors='coerce', infer_datetime_format=True)
        
        # If that fails for too many values, try some common formats
        if parsed.isna().sum() > len(df) * 0.1:  # If more than 10% failed
            common_formats = [
                '%m/%d/%y %I:%M %p',    # 5/13/25 12:00 AM
                '%m/%d/%Y %I:%M %p',    # 5/13/2025 12:00 AM
                '%m-%d-%y %H:%M',       # 5-13-25 12:00
                '%m-%d-%Y %H:%M',       # 5-13-2025 12:00
                '%Y-%m-%d %H:%M:%S',    # 2025-05-13 12:00:00
                '%m/%d/%y',             # 5/13/25
                '%m/%d/%Y',             # 5/13/2025
            ]
            
            for fmt in common_formats:
                try:
                    test_parsed = pd.to_datetime(df[column_name], format=fmt, errors='coerce')
                    if test_parsed.isna().sum() < parsed.isna().sum():
                        parsed = test_parsed
                        break
                except:
                    continue
        
        return parsed
    except:
        # Fallback to original column if all parsing fails
        return df[column_name]

def extract_first_code_block(text: str) -> str:
    """Extracts the first Python code block from a markdown-formatted string."""
    logger.debug("✂️ Extracting first code block from response")
    start = text.find("```python")
    if start == -1:
        logger.warning("⚠️ No ```python code block found")
        return ""
    start += len("```python")
    end = text.find("```", start)
    if end == -1:
        logger.warning("⚠️ Unclosed code block found")
        return ""
    code = text[start:end].strip()
    logger.debug(f"✂️ Extracted code: {len(code)} characters")
    return code

# === Main Streamlit App ===============================================

def main():
    """Main application function with navigation."""
    st.set_page_config(
        page_title="Data Analysis Agent",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Data Analysis", "🔧 System Monitoring"]
    )
    
    if page == "📊 Data Analysis":
        data_analysis_page()
    elif page == "🔧 System Monitoring":
        monitoring_dashboard()

def data_analysis_page():
    """Main data analysis page."""
    # Initialize session state for authentication
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    # Authentication page
    if not st.session_state.authenticated:
        st.title("🔐 Business Analysis HR Agent - Login")
        st.markdown("---")
        
        # Center the login form
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### Please enter your credentials")
            
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                submit_button = st.form_submit_button("Login", use_container_width=True)
                
                if submit_button:
                    # Check credentials
                    if username == "Plaksha-HR" and password == "AgentHR1":
                        st.session_state.authenticated = True
                        st.success("✅ Login successful! Redirecting...")
                        logger.info(f"🔓 Successful login for user: {username}")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password. Please try again.")
                        logger.warning(f"🔒 Failed login attempt for user: {username}")
        
        # Add some styling and info
        st.markdown("---")
        st.info("💡 Please contact your administrator if you need access credentials.")
        return  # Exit early if not authenticated
    
    # Main application (only accessible after authentication)
    if "plots" not in st.session_state:
        st.session_state.plots = []
    if "plot_data" not in st.session_state:
        st.session_state.plot_data = []

    logger.info("🚀 Starting Streamlit app - User authenticated")

    # Add logout button in the sidebar
    with st.sidebar:
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            logger.info("🔓 User logged out")
            st.rerun()

    left, right = st.columns([3,7])

    with left:
        st.header("Business Analysis HR Agent")
        st.markdown("<medium>Powered by <a href='https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1'>NVIDIA Llama-3.1-Nemotron-Ultra-253B-v1</a></medium>", unsafe_allow_html=True)
        file = st.file_uploader("Choose CSV", type=["csv"])
        if file:
            if ("df" not in st.session_state) or (st.session_state.get("current_file") != file.name):
                logger.info(f"📁 New file uploaded: {file.name}")
                
                # Try to detect encoding automatically, then fallback to common encodings
                df_loaded = False
                detected_encoding = None
                
                try:
                    # First, try to detect encoding using chardet
                    file.seek(0)
                    raw_data = file.read(10000)  # Read first 10KB for detection
                    file.seek(0)  # Reset file pointer
                    
                    if raw_data:
                        detection_result = chardet.detect(raw_data)
                        detected_encoding = detection_result.get('encoding')
                        confidence = detection_result.get('confidence', 0)
                        
                        if detected_encoding and confidence > 0.7:
                            logger.info(f"🔍 Detected encoding: {detected_encoding} (confidence: {confidence:.2f})")
                        else:
                            logger.warning(f"⚠️ Low confidence encoding detection: {detected_encoding} (confidence: {confidence:.2f})")
                except Exception as e:
                    logger.warning(f"⚠️ Encoding detection failed: {e}")
                
                # List of encodings to try (put detected encoding first if available)
                encodings_to_try = []
                if detected_encoding:
                    encodings_to_try.append(detected_encoding)
                
                # Add common fallback encodings
                common_encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
                for enc in common_encodings:
                    if enc not in encodings_to_try:
                        encodings_to_try.append(enc)
                
                # Try each encoding
                for encoding in encodings_to_try:
                    try:
                        file.seek(0)  # Reset file pointer
                        st.session_state.df = pd.read_csv(file, encoding=encoding)
                        df_loaded = True
                        logger.info(f"📊 Successfully loaded with {encoding} encoding")
                        
                        # Show success message to user if non-UTF-8 encoding was used
                        if encoding.lower() != 'utf-8':
                            st.success(f"✅ File loaded successfully using {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        logger.warning(f"⚠️ Failed to load with {encoding} encoding, trying next...")
                        continue
                    except Exception as e:
                        logger.error(f"❌ Error loading with {encoding}: {e}")
                        continue
                
                if not df_loaded:
                    st.error("""
                    ❌ **Unable to read the CSV file due to encoding issues.**
                    
                    **Possible solutions:**
                    - Save your CSV file with UTF-8 encoding
                    - If using Excel, use "Save As" → "CSV UTF-8 (Comma delimited)"
                    - Try opening the file in a text editor and saving with UTF-8 encoding
                    - Check if the file is actually a CSV format
                    """)
                    logger.error(f"❌ Failed to load file {file.name} with any supported encoding")
                    return
                
                st.session_state.current_file = file.name
                st.session_state.messages = []
                logger.info(f"📊 Loaded DataFrame: {len(st.session_state.df)} rows, {len(st.session_state.df.columns)} columns")
                with st.spinner("Generating dataset insights …"):
                    st.session_state.insights = DataInsightAgent(st.session_state.df)
            st.dataframe(st.session_state.df.head())
            st.markdown("### Dataset Insights")
            st.markdown(st.session_state.insights)
        else:
            st.info("Upload a CSV to begin chatting with your data.")

    with right:
        st.header("Chat with your data")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Separator between dataset insights and chat area
        if file and "df" in st.session_state:
            st.markdown("---")

        chat_container = st.container()
        with chat_container:
            for i, msg in enumerate(st.session_state.messages):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)
                    
                    # For assistant messages, add download options
                    if msg["role"] == "assistant":
                        # Check if this is dual-output (has both plot and data)
                        has_data = msg.get("data_index") is not None
                        has_plot = msg.get("plot_index") is not None
                        
                        if has_data and has_plot:
                            # Dual output: 4 columns for text, data CSV, plot PNG, and spacing
                            col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
                        else:
                            # Legacy: 3 columns
                            col1, col2, col3 = st.columns([2, 1, 1])
                            col4 = None
                        
                        with col1 if not has_data else col2:
                            # Download text response
                            # Extract clean text from the response (remove HTML)
                            import re
                            clean_text = re.sub(r'<[^>]+>', '', msg["content"])
                            clean_text = re.sub(r'\n+', '\n', clean_text).strip()
                            
                            if clean_text:
                                st.download_button(
                                    label="📄 Download Text",
                                    data=clean_text,
                                    file_name=f"analysis_response_{i+1}.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                        
                        # Download CSV data if available (dual-output only)
                        if has_data and col4:
                            with col3:
                                data_idx = msg["data_index"]
                                if 0 <= data_idx < len(st.session_state.get("plot_data", [])):
                                    data_df = st.session_state.plot_data[data_idx]
                                    
                                    # Convert DataFrame to CSV
                                    csv_buffer = io.StringIO()
                                    data_df.to_csv(csv_buffer, index=False)
                                    csv_data = csv_buffer.getvalue()
                                    
                                    st.download_button(
                                        label="📊 Download Data (CSV)",
                                        data=csv_data,
                                        file_name=f"plot_data_{i+1}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                        
                        # Download plot if available
                        plot_col = col4 if has_data else col3
                        if has_plot and plot_col:
                            with plot_col:
                                plot_idx = msg["plot_index"]
                                if 0 <= plot_idx < len(st.session_state.plots):
                                    # Create download button for plot
                                    fig = st.session_state.plots[plot_idx]
                                    
                                    # Save plot to bytes buffer
                                    img_buffer = io.BytesIO()
                                    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                                    img_buffer.seek(0)
                                    
                                    st.download_button(
                                        label="🖼️ Download Plot",
                                        data=img_buffer.getvalue(),
                                        file_name=f"plot_{i+1}.png",
                                        mime="image/png",
                                        use_container_width=True
                                    )
                    
                    # Display plot
                    if msg.get("plot_index") is not None:
                        idx = msg["plot_index"]
                        if 0 <= idx < len(st.session_state.plots):
                            # Display plot at fixed size
                            st.pyplot(st.session_state.plots[idx], use_container_width=False)
                    
                    # Display data table for dual-output
                    if msg.get("data_index") is not None:
                        data_idx = msg["data_index"]
                        if 0 <= data_idx < len(st.session_state.get("plot_data", [])):
                            data_df = st.session_state.plot_data[data_idx]
                            
                            # Show data table with expandable section
                            with st.expander(f"📊 View Source Data ({len(data_df)} rows, {len(data_df.columns)} columns)", expanded=False):
                                st.dataframe(
                                    data_df, 
                                    use_container_width=True,
                                    height=min(400, len(data_df) * 35 + 40)  # Adaptive height
                                )
                                
                                # Add summary statistics for numeric columns
                                numeric_cols = data_df.select_dtypes(include=[np.number]).columns
                                if len(numeric_cols) > 0:
                                    st.markdown("**Summary Statistics:**")
                                    summary_stats = data_df[numeric_cols].describe()
                                    st.dataframe(summary_stats, use_container_width=True)
                    
                    # Display code in a proper expander for assistant messages
                    if msg.get("code") and msg["role"] == "assistant":
                        with st.expander("View code", expanded=False):
                            st.code(msg["code"], language="python")

        if file:  # only allow chat after upload
            if user_q := st.chat_input("Ask about your data…"):
                logger.info(f"💬 User query received: '{user_q}'")
                st.session_state.messages.append({"role": "user", "content": user_q})
                
                with st.spinner("Working …"):
                    start_time = datetime.now()
                    logger.info(f"⏱️ Processing started at {start_time}")
                    
                    # Pass chat history and column memory to enable enhanced analysis
                    code, should_plot_flag, code_thinking = CodeGenerationAgent(user_q, st.session_state.df, st.session_state.messages)
                    result_obj = ExecutionAgent(code, st.session_state.df, should_plot_flag)
                    
                    # Auto-retry logic for common pandas errors
                    if isinstance(result_obj, str) and result_obj.startswith("Error executing code:"):
                        logger.warning("🔄 Code execution failed, attempting automatic retry with error context")
                        error_context = result_obj
                        
                        # Try once more with error context
                        try:
                            code_retry, should_plot_flag_retry, _ = CodeGenerationAgent(
                                user_q, st.session_state.df, st.session_state.messages, 
                                retry_context=error_context
                            )
                            result_obj_retry = ExecutionAgent(code_retry, st.session_state.df, should_plot_flag_retry)
                            
                            # If retry succeeds, use the retry result
                            if not (isinstance(result_obj_retry, str) and result_obj_retry.startswith("Error executing code:")):
                                logger.info("✅ Retry successful, using corrected result")
                                code = code_retry
                                should_plot_flag = should_plot_flag_retry
                                result_obj = result_obj_retry
                            else:
                                logger.warning("⚠️ Retry also failed, using original error")
                        except Exception as e:
                            logger.error(f"❌ Retry attempt failed: {e}")
                    
                    raw_thinking, reasoning_txt = ReasoningAgent(user_q, result_obj)
                    reasoning_txt = reasoning_txt.replace("`", "")

                    end_time = datetime.now()
                    processing_time = (end_time - start_time).total_seconds()
                    logger.info(f"⏱️ Total processing time: {processing_time:.2f} seconds")

                # Build assistant response - handle dual output format
                is_dual_output = isinstance(result_obj, tuple) and len(result_obj) == 2 and isinstance(result_obj[0], (plt.Figure, plt.Axes)) and isinstance(result_obj[1], pd.DataFrame)
                is_plot = isinstance(result_obj, (plt.Figure, plt.Axes))
                plot_idx = None
                data_idx = None
                
                if is_dual_output:
                    # Handle new dual-output format (fig, data_df)
                    fig, data_df = result_obj
                    
                    # Store plot
                    st.session_state.plots.append(fig)
                    plot_idx = len(st.session_state.plots) - 1
                    
                    # Store data (create data storage if it doesn't exist)
                    if "plot_data" not in st.session_state:
                        st.session_state.plot_data = []
                    st.session_state.plot_data.append(data_df)
                    data_idx = len(st.session_state.plot_data) - 1
                    
                    header = "Here is your enhanced visualization with underlying data:"
                    logger.info(f"📊 Dual-output added: plot at index {plot_idx}, data at index {data_idx} ({len(data_df)} rows)")
                    
                elif is_plot:
                    # Handle legacy single plot format
                    fig = result_obj.figure if isinstance(result_obj, plt.Axes) else result_obj
                    st.session_state.plots.append(fig)
                    plot_idx = len(st.session_state.plots) - 1
                    header = "Here is the visualization you requested:"
                    logger.info(f"📊 Legacy plot added to session state at index {plot_idx}")
                    
                elif isinstance(result_obj, (pd.DataFrame, pd.Series)):
                    header = f"Result: {len(result_obj)} rows" if isinstance(result_obj, pd.DataFrame) else "Result series"
                    logger.info(f"📄 Data result: {header}")
                else:
                    header = f"Result: {result_obj}"
                    logger.info(f"📄 Scalar result: {header}")

                # Show only reasoning thinking in Model Thinking (collapsed by default)
                thinking_html = ""
                if raw_thinking:
                    thinking_html = (
                        '<details class="thinking">'
                        '<summary>🧠 Reasoning</summary>'
                        f'<pre>{raw_thinking}</pre>'
                        '</details>'
                    )

                # Show model explanation directly 
                explanation_html = reasoning_txt

                # Store code separately for proper display
                # Combine thinking and explanation
                assistant_msg = f"{thinking_html}{explanation_html}"

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_msg,
                    "plot_index": plot_idx,
                    "data_index": data_idx,  # Store data index for dual-output
                    "code": code  # Store code separately
                })
                
                logger.info("✅ Response added to chat history, rerunning app")
                st.rerun()

# === SYSTEM MONITORING DASHBOARD ===
def monitoring_dashboard():
    """System monitoring and health dashboard."""
    st.title("🔧 System Monitoring Dashboard")
    
    # Health Status
    st.header("🏥 System Health")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Health Check"):
            perform_health_check()
    
    with col2:
        if st.button("📊 Export Health Report"):
            export_health_report("health_report.json")
            st.success("Health report exported to health_report.json")
    
    with col3:
        if st.button("🗑️ Clear Cache"):
            cleanup_all_caches()
            st.success("Cache cleanup completed")
    
    # Get health status
    health_status = get_health_status()
    
    # Overall status
    status_color = {
        "healthy": "🟢",
        "degraded": "🟡", 
        "unhealthy": "🔴",
        "unknown": "⚪"
    }
    
    overall_status = health_status.get("overall_status", "unknown")
    st.metric(
        "Overall System Status",
        f"{status_color.get(overall_status, '⚪')} {overall_status.upper()}",
        f"Uptime: {health_status.get('uptime_minutes', 0):.1f} min"
    )
    
    # Health checks
    if "checks" in health_status:
        st.subheader("📋 Health Checks")
        for check_name, check_data in health_status["checks"].items():
            with st.expander(f"{status_color.get(check_data['status'], '⚪')} {check_name.replace('_', ' ').title()}"):
                st.write(f"**Status:** {check_data['status']}")
                st.write(f"**Message:** {check_data['message']}")
                if check_data.get('response_time_ms'):
                    st.write(f"**Response Time:** {check_data['response_time_ms']:.1f}ms")
                if check_data.get('metadata'):
                    st.json(check_data['metadata'])
    
    # Performance Metrics
    st.header("📈 Performance Metrics")
    
    # Get metrics summary
    metrics = get_metrics_summary(hours_back=1)
    
    if metrics.get("total_events", 0) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "API Calls",
                metrics["api_calls"]["count"],
                f"{metrics['api_calls']['success_rate']:.1%} success rate"
            )
        
        with col2:
            st.metric(
                "Avg Response Time",
                f"{metrics['api_calls']['avg_response_time_ms']:.0f}ms",
                f"{metrics['api_calls']['total_tokens']} tokens"
            )
        
        with col3:
            st.metric(
                "Code Executions", 
                metrics["code_executions"]["count"],
                f"{metrics['code_executions']['success_rate']:.1%} success rate"
            )
        
        with col4:
            st.metric(
                "Error Rate",
                f"{metrics['errors']['error_rate']:.1%}",
                f"{metrics['errors']['count']} errors"
            )
        
        # Error breakdown
        if metrics["errors"]["count"] > 0:
            st.subheader("❌ Error Breakdown")
            for error in metrics["errors"]["top_error_types"]:
                st.write(f"• **{error['type']}**: {error['count']} occurrences")
    
    else:
        st.info("No performance metrics available yet. Start using the application to see metrics.")
    
    # Circuit Breaker Status
    st.header("🔧 Circuit Breaker Status")
    
    breaker_stats = get_all_circuit_breaker_stats()
    if breaker_stats:
        for name, stats in breaker_stats.items():
            state_color = {
                "closed": "🟢",
                "half_open": "🟡",
                "open": "🔴"
            }
            
            with st.expander(f"{state_color.get(stats['state'], '⚪')} {name.replace('_', ' ').title()}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**State:** {stats['state']}")
                    st.write(f"**Failure Count:** {stats['failure_count']}")
                    st.write(f"**Success Count:** {stats['success_count']}")
                
                with col2:
                    st.write("**Configuration:**")
                    st.json(stats['config'])
    
    # Cache Statistics
    st.header("💾 Cache Statistics")
    
    cache_stats = get_cache_stats()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Code Cache")
        code_stats = cache_stats["code_cache"]
        st.metric("Cache Size", f"{code_stats['size']}/{code_stats['max_size']}")
        st.metric("Hit Rate", f"{code_stats['hit_rate']:.1%}")
        st.metric("Total Requests", code_stats['total_requests'])
        
        if code_stats.get('memory_usage_estimate'):
            st.metric("Memory Usage", f"{code_stats['memory_usage_estimate']['total_mb']:.1f} MB")
    
    with col2:
        st.subheader("Analysis Cache") 
        analysis_stats = cache_stats["analysis_cache"]
        st.metric("Cache Size", f"{analysis_stats['size']}/{analysis_stats['max_size']}")
        st.metric("Hit Rate", f"{analysis_stats['hit_rate']:.1%}")
        st.metric("Total Requests", analysis_stats['total_requests'])
        
        if analysis_stats.get('memory_usage_estimate'):
            st.metric("Memory Usage", f"{analysis_stats['memory_usage_estimate']['total_mb']:.1f} MB")
    
    # Popular Patterns
    if cache_stats.get("popular_patterns"):
        st.subheader("🔥 Popular Query Patterns")
        for pattern, count in cache_stats["popular_patterns"]:
            st.write(f"• **{pattern}**: {count} uses")

if __name__ == "__main__":
    main() 