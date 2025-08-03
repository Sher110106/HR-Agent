"""Code generation agents and tools for data analysis queries.

This module contains tools for understanding user queries, generating appropriate
code (plotting or analysis), and orchestrating the code generation process.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any
import pandas as pd

from .memory import ConversationMemoryTool, ColumnMemoryAgent, SystemPromptMemoryAgent, enhance_prompt_with_context
from app_core.api import make_llm_call
from agents.data_analysis import extract_first_code_block
from utils.cache import code_cache

logger = logging.getLogger(__name__)


def QueryUnderstandingTool(query: str, conversation_context: str = "") -> bool:
    """Return True if the query seems to request a visualisation based on keywords and context."""
    logger.info(f"🔍 QueryUnderstandingTool: Analyzing query: '{query}'")
    
    # Get system prompt context
    system_prompt_agent = SystemPromptMemoryAgent()
    
    # Enhanced LLM prompt that includes conversation context
    base_system_prompt = """detailed thinking off. You are a highly specialized query classification assistant. Your sole purpose is to determine if a user's request, including the full conversation context, necessitates a data visualization.

Your analysis must consider:
1.  **Explicit Keywords**: Identify direct requests for a 'plot', 'chart', 'graph', 'diagram', 'visualize', 'show me', etc.
2.  **Implicit Intent**: Infer visualization needs from phrases that imply visual comparison or trend analysis, such as "compare sales across regions," "show the trend over time," "what does the distribution look like," or "can you illustrate the relationship between X and Y?"
3.  **Follow-up Context**: Analyze conversation history. If a user says "make that a bar chart" or "what about for Q3?", you must recognize this refers to a prior analysis or visualization.

Respond with only 'true' for any explicit or implicit visualization request. For all other requests (e.g., data summaries, statistical calculations, data transformations), respond with 'false'. Your output must be a single boolean value."""
    
    # Apply system prompt if active
    system_prompt = system_prompt_agent.apply_system_prompt(base_system_prompt)
    
    user_prompt = f"Query: {query}"
    if conversation_context:
        user_prompt = f"Previous conversation:\n{conversation_context}\n\nCurrent query: {query}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    logger.info("📤 Sending query understanding request to LLM...")
    try:
        response = make_llm_call(
            messages=messages,
            temperature=0.1,
            max_tokens=1000
        )
    except Exception as e:
        # Graceful degradation – if the LLM is unreachable, fall back to heuristic
        logger.error(f"❌ QueryUnderstandingTool failed – falling back to heuristic: {e}")
        keyword_triggers = [
            "plot", "chart", "graph", "diagram", "visualise", "visualize",
            "show me", "trend", "compare", "distribution", "relationship"
        ]
        heuristic = any(kw in query.lower() for kw in keyword_triggers)
        logger.info(f"🔍 Heuristic visualization detection result: {heuristic}")
        return heuristic
    
    # Extract the response and convert to boolean
    intent_response = response.choices[0].message.content.strip().lower()
    result = intent_response == "true"
    logger.info(f"📥 LLM response: '{intent_response}' → Visualization needed: {result}")
    
    return result


def PlotCodeGeneratorTool(cols: List[str], query: str, df: pd.DataFrame, conversation_context: str = "", memory_agent: ColumnMemoryAgent = None) -> str:
    """
    Generate a prompt for the LLM to write pandas+matplotlib code for a plot based on the query and columns.
    The generated code must return a tuple (fig, data_df):
      - fig: matplotlib figure object with professional styling
      - data_df: pandas DataFrame containing the aggregated data used to create the plot
    Value labels and text annotations are not allowed.
    """
    logger.info(f"📊 PlotCodeGeneratorTool: Generating plot prompt for columns: {cols}")
    
    # Get data types and sample values for better context
    data_info = []
    date_columns = []
    
    for col in cols:
        dtype = str(df[col].dtype)
        if memory_agent and memory_agent.has_descriptions():
            stored_description = memory_agent.get_column_description(col)
            if stored_description:
                data_info.append(f"{col} ({dtype}): {stored_description[:200]}...")
                if any(keyword in stored_description.lower() for keyword in ['date', 'time', 'temporal', 'timestamp']):
                    date_columns.append(col)
                continue
        is_likely_date = False
        if df[col].dtype == 'object':
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
            unique_vals = df[col].unique()[:5]
            data_info.append(f"{col} ({dtype}): {list(unique_vals)}")
        else:
            data_info.append(f"{col} ({dtype}): numeric range {df[col].min()}-{df[col].max()}")
    
    data_context = "\n".join(data_info)
    logger.debug(f"Data context: {data_context[:300]}...")
    
    date_instructions = ""
    if date_columns:
        logger.info(f"📅 Detected potential date columns: {date_columns}")
        date_instructions = f"""
    IMPORTANT - Date/Time Handling:
    - Detected date/time columns: {', '.join(date_columns)}
    - For date parsing, use: pd.to_datetime(df['column_name'], errors='coerce')
    """
    
    context_section = f"\nPrevious conversation context:\n{conversation_context}\n" if conversation_context else ""
    enhancement_note = "\n\nAI-generated column descriptions are available above. Use this context for more relevant visualizations." if memory_agent and memory_agent.has_descriptions() else ""
    
    prompt = f"""
Given DataFrame `df` with columns and data types:
{data_context}
{context_section}{date_instructions}{enhancement_note}
Write Python code using pandas, matplotlib (as plt) and seaborn (as sns) to answer:
"{query}"

REQUIREMENTS
- Return a tuple (fig, data_df) where:
    – fig is a matplotlib Figure with **professional styling**
    – data_df is the tidy DataFrame used to create the plot
- Use MINIMAL value labels only when truly helpful (avoid clutter)
- **DO NOT** use plt.table / ax.table
- Use ONLY these helper utilities:
    – apply_professional_styling(ax, title, xlabel, ylabel)
    – format_axis_labels(ax, x_rotation=45)
    – get_professional_colors()['colors']
    – safe_color_access(colors, index) # Safe color access with cycling
    – create_category_palette(categories, palette_name='primary') # For seaborn category-specific palettes
    – optimize_figure_size(ax)
    – add_value_labels(ax, label_mode="minimal") # Use sparingly for key insights only
    – handle_seaborn_warnings() # Call at start to suppress harmless seaborn warnings
    – safe_binning(data, bins, labels=None, method='cut') # Safe binning with automatic validation
    
    CRITICAL - PANDAS CUT/BINNING VALIDATION:
    - When using pd.cut(), ALWAYS ensure labels length = bins length - 1
    - Example: bins=[0,1,3,5], labels=['0-1','1-3','3-5'] (3 labels for 4 bins)
    - For automatic binning without labels: pd.cut(df['col'], bins=5) # No labels parameter
    - For custom bins with labels: pd.cut(df['col'], bins=[0,1,3,5], labels=['0-1','1-3','3-5'])
    - ALWAYS validate: len(labels) == len(bins) - 1 before using pd.cut()
    - If unsure, use automatic binning: pd.cut(df['col'], bins=5) without labels
    - RECOMMENDED: Use safe_binning() instead of pd.cut() for automatic error prevention
    - SAFE BINNING EXAMPLES:
      * Automatic: df['bins'] = safe_binning(df['col'], bins=5)
      * Custom bins: df['bins'] = safe_binning(df['col'], bins=[0,1,3,5], labels=['0-1','1-3','3-5'])
      * Equal frequency: df['bins'] = safe_binning(df['col'], bins=5, method='qcut')
    
    SEABORN PALETTE GUIDANCE:
    - For category-specific colors: palette = create_category_palette(df['category_col'].unique())
    - For general seaborn plots: palette = get_professional_colors()['colors'][:n_categories]
    - Always slice colors to match the number of categories to avoid warnings
    - When using seaborn with palette, assign the categorical column to 'hue' and set legend=False to avoid deprecation warnings
    - Example: sns.barplot(data=df, x='x_col', y='y_col', hue='category_col', palette=palette, legend=False)
    
    PANDAS BEST PRACTICES:
    - Use observed=True in groupby operations: df.groupby('col', observed=True)
    - Handle missing values with .dropna() before operations
    - Use .copy() to avoid SettingWithCopyWarning
    - For concatenation with different column structures, use pd.concat([df1, df2], ignore_index=True, sort=False)
    - Always assign your final result to the variable 'result' - this is critical!
    
    CHART TYPE HELPERS (choose the most appropriate):
    – create_clean_bar_chart(ax, data_df, x_col, y_col, hue_col=None, title="", xlabel="", ylabel="", legend_totals=True)
    – create_clean_line_chart(ax, data_df, x_col, y_col, hue_col=None, title="", xlabel="", ylabel="", show_markers=True)
    – create_clean_scatter_plot(ax, data_df, x_col, y_col, hue_col=None, size_col=None, title="", xlabel="", ylabel="", add_trendline=False)
    – create_clean_histogram(ax, data_df, col, bins=30, title="", xlabel="", ylabel="Frequency", show_stats=True)
    – create_clean_box_plot(ax, data_df, x_col, y_col, title="", xlabel="", ylabel="", show_outliers=True)
    – create_clean_heatmap(ax, data_df, title="", cmap='RdYlBu_r', show_values=False)
    – create_clean_pie_chart(ax, data_df, col, title="", show_percentages=True, explode_max=True)
    – smart_categorical_plot(ax, data_df, x_col, y_col, plot_type="auto") # For categorical data with many points
    – smart_annotate_points(ax, data_df, x_col, y_col, label_col, max_annotations=5) # Add smart annotations to scatter plots
- Ensure the figure is aesthetically pleasing: grid, spines removed, legend with totals, tight layout
- Assign the final tuple to `result`
- Wrap code in a single ```python block with no extra text
"""
    logger.debug(f"Generated plot prompt: {prompt[:200]}...")
    return prompt


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
    - For date parsing, ALWAYS use: pd.to_datetime(df['column_name'], errors='coerce')
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
    5. For groupby operations, use observed=True: df.groupby('col', observed=True) and handle missing values with .dropna() if needed.
    6. **CRITICAL**: Always assign the final result to `result` variable - this is required!
    7. Ensure result is a clear, interpretable value (float, dict, or small DataFrame).
    8. For concatenation with different column structures, use pd.concat([df1, df2], ignore_index=True, sort=False)
    9. When using seaborn, only use 'palette' if 'hue' is specified. If you want to color by a single variable, assign it to 'hue' and set 'legend=False'.
    10. When providing a palette, slice it to match the number of unique categories in the data.
    11. When setting tick labels, always set the ticks first or use ax.tick_params for rotation instead of set_ticklabels.
    12. Use .copy() to avoid SettingWithCopyWarning when creating filtered DataFrames.
    13. Wrap code in a single ```python fence with no explanations.

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


def CodeGenerationAgent(query: str, df: pd.DataFrame, chat_history: List[Dict] = None, memory_agent: ColumnMemoryAgent = None, retry_context: str = None):
    """Selects the appropriate code generation tool and gets code from the LLM for the user's query."""
    logger.info(f"🤖 CodeGenerationAgent: Processing query: '{query}'")
    logger.info(f"📊 DataFrame info: {len(df)} rows, {len(df.columns)} columns")
    
    if retry_context:
        logger.info(f"🔄 Retry mode activated with context: {retry_context[:200]}...")
    
    # Get system prompt context
    system_prompt_agent = SystemPromptMemoryAgent()
    
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

    # Base system prompt for code generation
    base_system_prompt = """detailed thinking off. You are a senior data scientist with expertise in pandas, matplotlib, and seaborn for statistical analysis and visualization. Write clean, efficient, production-ready code. Focus on:

1. CORRECTNESS: Ensure proper variable scoping (pd, np, df, plt, sns are available)
2. ROBUSTNESS: Handle missing values and edge cases
3. CLARITY: Use descriptive variable names and clear logic
4. EFFICIENCY: Prefer vectorized operations over loops
5. BEST PRACTICES: Follow pandas conventions and leverage seaborn for enhanced visualizations
6. AESTHETICS: Use seaborn's statistical plotting capabilities for professional-looking charts

Output ONLY a properly-closed ```python code block. Use smart_date_parser() for date parsing. Assign final result to 'result' variable."""
    
    # Apply system prompt if active
    system_prompt = system_prompt_agent.apply_system_prompt(base_system_prompt)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    logger.info("📤 Sending code generation request to LLM...")
    try:
        response = make_llm_call(
            messages=messages,
            temperature=0.2,
            max_tokens=4000
        )
    except Exception as e:
        logger.error(f"❌ Code generation LLM call failed: {e}")
        # Propagate a structured error so the UI can show meaningful message
        error_msg = f"Error generating code: {e}"
        return error_msg, False, ""

    full_response = response.choices[0].message.content
    logger.info(f"📥 LLM full response length: {len(full_response)} characters")
    
    code = extract_first_code_block(full_response)
    logger.info(f"✂️ Extracted code block length: {len(code)} characters")
    logger.info(f"💻 Generated code:\n{code}")
    
    return code, should_plot, "" 