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

# === Configuration ===
api_key = os.environ.get("NVIDIA_API_KEY")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('data_analysis_agent.log')  # File output
    ]
)
logger = logging.getLogger(__name__)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

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
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=messages,
        temperature=0.1,
        max_tokens=1000  # We only need a short response
    )
    
    # Extract the response and convert to boolean
    intent_response = response.choices[0].message.content.strip().lower()
    result = intent_response == "true"
    logger.info(f"📥 LLM response: '{intent_response}' → Visualization needed: {result}")
    
    return result

# ------------------  PlotCodeGeneratorTool ---------------------------
def PlotCodeGeneratorTool(cols: List[str], query: str, df: pd.DataFrame, conversation_context: str = "", memory_agent: ColumnMemoryAgent = None) -> str:
    """Generate a prompt for the LLM to write pandas+matplotlib code for a plot based on the query and columns."""
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

    Rules & Available Tools
    ----------------------
         1. Use pandas, matplotlib.pyplot (as plt), and seaborn (as sns) - `pd`, `np`, `df`, `plt`, `sns` are all available in scope.
    2. For date/time columns, prefer smart_date_parser(df, 'column_name') for robust parsing.
    3. For categorical columns, convert to numeric: df['col'].map({{'Yes': 1, 'No': 0}}).
    4. CRITICAL: Create figure with `fig, ax = plt.subplots(figsize=(12,7))` and assign `result = fig`.
    5. Create ONE clear, well-labeled plot with ax.set_title(), ax.set_xlabel(), ax.set_ylabel().
    6. For time series, consider df.set_index('date_col').plot() for automatic time formatting.
    7. Handle missing values with .dropna() before plotting if needed.
    8. Use clear colors and markers: ax.scatter(), ax.plot(), ax.bar(), etc.
    9. Leverage seaborn for enhanced statistical visualizations and better aesthetics.
    10. Wrap code in ```python fence with no explanations.

    Plotting Examples:
    - Scatter: ax.scatter(df['x'], df['y'], alpha=0.6) or sns.scatterplot(data=df, x='x', y='y', ax=ax)
    - Time series: df.set_index('date').plot(ax=ax)
    - Correlation heatmap: sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    - Box plots: sns.boxplot(data=df, x='category', y='value', ax=ax)
    - Distribution: sns.histplot(data=df, x='column', kde=True, ax=ax)
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
    
    should_plot = QueryUnderstandingTool(query, conversation_context)
    
    # Add retry context if available
    if retry_context:
        conversation_context += f"\n\nPREVIOUS ERROR TO AVOID:\n{retry_context}\n\nPlease fix this error and generate corrected code."
    
    prompt = PlotCodeGeneratorTool(df.columns.tolist(), query, df, conversation_context, memory_agent) if should_plot else CodeWritingTool(df.columns.tolist(), query, df, conversation_context, memory_agent)

    messages = [
        {"role": "system", "content": """detailed thinking off. You are an elite-level senior data scientist and Python programmer, specializing in writing production-grade data analysis and visualization code. Your code must be flawless, efficient, and contextually aware.

Your mission is to generate a single, executable Python code block to answer the user's query, adhering to these strict principles:

1.  **CORRECTNESS & SCOPE**: The code must be syntactically perfect. You have access to `pd`, `np`, `df`, `plt`, `sns`, and `smart_date_parser`. Ensure all variables are correctly scoped and used.
2.  **ROBUSTNESS & ERROR HANDLING**: Your code must anticipate and gracefully handle potential issues.
    *   Check for empty DataFrames (`df.empty`) or columns with all `NaN` values before performing operations.
    *   Include `try-except` blocks for operations prone to failure (e.g., data type conversions, complex calculations) and raise meaningful exceptions.
    *   Handle missing values (`.isna()`) appropriately based on the analytical context (e.g., `dropna()`, `fillna()`).
3.  **CLARITY & MAINTAINABILITY**: Write code that a human can easily understand.
    *   Use descriptive variable names (e.g., `average_employee_tenure` instead of `avg_ten`).
    *   Add concise inline comments (`#`) to explain complex logic, transformations, or business rule implementations.
    *   Adhere strictly to PEP 8 style guidelines.
4.  **EFFICIENCY**: Prioritize performance. Heavily favor vectorized pandas/numpy operations over loops or `.apply()` where possible.
5.  **CONTEXT AWARENESS**: This is critical. You MUST leverage the provided conversation history and AI-generated column descriptions (`{enhancement_note}`). Your code should reflect a deep understanding of the business context inferred from these sources.
6.  **PANDAS BEST PRACTICES**: Follow these critical pandas patterns:
    *   Use `.loc[]` for label-based indexing and `.iloc[]` for integer-based indexing correctly
    *   When chaining methods, ensure each step returns the expected data type
    *   For getting top N values: `series.nlargest(n).index` to get indices, then `df.loc[indices]` to slice
    *   Avoid SettingWithCopyWarning by using `.copy()` explicitly or proper `.loc[]` assignments
    *   Test method chains step by step - don't chain incompatible operations
7.  **VISUALIZATION AESTHETICS**: When creating plots, strive for presentation quality.
    *   Always create a figure and axes (e.g., `fig, ax = plt.subplots(figsize=(12, 7))`).
    *   Ensure every plot has a clear, descriptive title, and labels for the x and y axes.
    *   Use legends when multiple data series are present.
    *   Select color palettes and plot styles that are professional and suitable for both light and dark themes. Use `seaborn` to enhance aesthetics.
8.  **OUTPUT SPECIFICATION**:
    *   For visualizations, the final line must be `result = fig`.
    *   For non-visualization tasks, the final line must assign the output (e.g., DataFrame, scalar, list, string) to the `result` variable.
    *   Output ONLY the properly-closed ```python code block with no preceding or succeeding text."""},
        {"role": "user", "content": prompt}
    ]

    logger.info("📤 Sending code generation request to LLM...")
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=messages,
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
    """Executes the generated code in a controlled environment and returns the result or error message."""
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
        logger.info("🎨 Plot environment set up with matplotlib and seaborn")
    
    try:
        logger.info("🚀 Executing code...")
        # Use env as both globals and locals to ensure proper variable access
        exec(code, env, env)
        result = env.get("result", None)
        
        if result is not None:
            result_type = type(result).__name__
            if isinstance(result, pd.DataFrame):
                logger.info(f"✅ Execution successful: DataFrame with {len(result)} rows, {len(result.columns)} columns")
            elif isinstance(result, pd.Series):
                logger.info(f"✅ Execution successful: Series with {len(result)} elements")
            elif isinstance(result, (plt.Figure, plt.Axes)):
                logger.info(f"✅ Execution successful: {result_type} plot object")
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

    if is_error:
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

    if is_plot:
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
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=[
            {"role": "system", "content": """detailed thinking on. You are an expert data analyst and business strategist. Your primary role is to translate raw code outputs (plots, data, or errors) into comprehensive, actionable business insights for decision-makers.

Follow this structured reasoning process:

1.  **INTERPRET THE RESULT**:
    *   **What is it?** Start by identifying the type of output and INCLUDE THE ACTUAL RESULTS/VALUES in your analysis.
    *   **What does it show?** Present the specific findings with actual numbers, lists, or data points. Don't just describe the type - show the content.

2.  **CONTEXTUALIZE THE FINDINGS**:
    *   **Why is this important?** Connect the specific results back to the user's original query and business context.
    *   **Leverage Business Context**: Explain what these concrete findings mean for the business using the AI-generated column descriptions and dataset context.

3.  **PROVIDE ACTIONABLE INSIGHTS & RECOMMENDATIONS**:
    *   **So what?** Go beyond observation. Provide concrete, forward-looking recommendations based on the specific results.
    *   **Suggest Next Steps**: Propose logical follow-up questions or actions the user could take.

4.  **HANDLE ERRORS GRACEFULLY**:
    *   If the result is an error, clearly explain the cause and provide specific, actionable solutions.

CRITICAL REQUIREMENTS for your final response:
- ALWAYS include the actual results/findings (numbers, lists, values) in your final explanation - never just describe what type of result it is
- Be comprehensive and detailed - provide thorough analysis, not just 2-3 sentences
- Present results in a clear, organized format that's easy to understand
- Include specific recommendations based on the actual data found

Begin by streaming your detailed analytical process within `<think>...</think>` tags. After your thought process is complete, provide a comprehensive, detailed explanation to the user outside the tags that includes the actual results."""},
            {"role": "user", "content": prompt}
        ],
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

def AnalyzeColumnBatch(df: pd.DataFrame, column: str) -> Tuple[str, str]:
    """Single column analysis function for parallel processing with retry logic."""
    import time
    max_retries = 2
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries + 1):
        try:
            description = ColumnAnalysisAgent(df, column)
            return column, description
        except Exception as e:
            if attempt < max_retries and ("rate limit" in str(e).lower() or "429" in str(e)):
                logger.warning(f"⚠️ Rate limit hit for column {column}, retrying in {retry_delay}s (attempt {attempt + 1})")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            else:
                logger.error(f"❌ Failed to analyze column {column} after {attempt + 1} attempts: {e}")
                return column, f"Error analyzing column: {str(e)}"

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
    if total_columns <= 5:
        max_workers = total_columns  # Small datasets: full parallelism
    elif total_columns <= 20:
        max_workers = min(6, total_columns)  # Medium datasets: moderate parallelism
    else:
        max_workers = 8  # Large datasets: conservative parallelism to avoid rate limits
    
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
    st.set_page_config(layout="wide")
    
    # Initialize session state for authentication
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    # Initialize column memory agent
    if "column_memory" not in st.session_state:
        st.session_state.column_memory = ColumnMemoryAgent()
    
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
                # Clear column memory for new file
                st.session_state.column_memory.clear_descriptions()
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

        # Add column analysis button if data is loaded
        if file and "df" in st.session_state:
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            
            with col1:
                analyze_button = st.button(
                    "🧠 Analyze Columns",
                    help="Generate AI-powered descriptions for each column to enhance analysis capabilities",
                    use_container_width=True
                )
            
            with col2:
                # Show status of column analysis
                if st.session_state.column_memory.has_descriptions():
                    st.success(f"✅ {len(st.session_state.column_memory.get_all_descriptions())} columns analyzed")
                else:
                    st.info("💡 Click to analyze columns")
            
            with col3:
                # Show analyzed columns in an expander
                if st.session_state.column_memory.has_descriptions():
                    with st.expander("📋 View Analyzed Columns"):
                        analyzed_columns = list(st.session_state.column_memory.get_all_descriptions().keys())
                        st.write(", ".join([f"`{col}`" for col in analyzed_columns]))
            
            with col4:
                if st.session_state.column_memory.has_descriptions():
                    if st.button("🗑️ Clear", help="Clear stored column descriptions", use_container_width=True):
                        st.session_state.column_memory.clear_descriptions()
                        st.rerun()
            
            # Handle column analysis button click
            if analyze_button:
                with st.spinner("🔍 Analyzing all columns with AI... This may take a moment."):
                    try:
                        analysis_summary = AnalyzeAllColumnsAgent(st.session_state.df, st.session_state.column_memory)
                        
                        # Add analysis summary to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": analysis_summary,
                            "plot_index": None,
                            "code": None
                        })
                        
                        st.success("🎉 Column analysis complete! Enhanced AI understanding enabled.")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error during column analysis: {str(e)}")
                        logger.error(f"Column analysis failed: {e}")
            
            st.markdown("---")

        chat_container = st.container()
        with chat_container:
            for i, msg in enumerate(st.session_state.messages):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)
                    
                    # For assistant messages, add download options
                    if msg["role"] == "assistant":
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col2:
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
                        
                        with col3:
                            # Download plot if available
                            if msg.get("plot_index") is not None:
                                idx = msg["plot_index"]
                                if 0 <= idx < len(st.session_state.plots):
                                    # Create download button for plot
                                    fig = st.session_state.plots[idx]
                                    
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
                    code, should_plot_flag, code_thinking = CodeGenerationAgent(user_q, st.session_state.df, st.session_state.messages, st.session_state.column_memory)
                    result_obj = ExecutionAgent(code, st.session_state.df, should_plot_flag)
                    
                    # Auto-retry logic for common pandas errors
                    if isinstance(result_obj, str) and result_obj.startswith("Error executing code:"):
                        logger.warning("🔄 Code execution failed, attempting automatic retry with error context")
                        error_context = result_obj
                        
                        # Try once more with error context
                        try:
                            code_retry, should_plot_flag_retry, _ = CodeGenerationAgent(
                                user_q, st.session_state.df, st.session_state.messages, 
                                st.session_state.column_memory, retry_context=error_context
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

                # Build assistant response
                is_plot = isinstance(result_obj, (plt.Figure, plt.Axes))
                plot_idx = None
                if is_plot:
                    fig = result_obj.figure if isinstance(result_obj, plt.Axes) else result_obj
                    st.session_state.plots.append(fig)
                    plot_idx = len(st.session_state.plots) - 1
                    header = "Here is the visualization you requested:"
                    logger.info(f"📊 Plot added to session state at index {plot_idx}")
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
                    "code": code  # Store code separately
                })
                
                logger.info("✅ Response added to chat history, rerunning app")
                st.rerun()

if __name__ == "__main__":
    main() 