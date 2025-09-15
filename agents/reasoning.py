"""
Reasoning agents for data analysis interpretation and insights.

This module contains the ReasoningAgent for providing comprehensive analysis
and the ReasoningCurator for crafting effective prompts.
"""
from __future__ import annotations

import logging
import re
import streamlit as st
from typing import Any
import matplotlib.pyplot as plt

from app_core.api import make_llm_call
from .memory import SystemPromptMemoryAgent

logger = logging.getLogger(__name__)


def ReasoningCurator(query: str, result: Any) -> str:
    """
    Create a detailed prompt for the LLM to reason about the query result.
    Considers different types of results (plots, data, errors) and structures
    the prompt to extract meaningful business insights.
    """
    logger.info(f"🎯 ReasoningCurator: Crafting prompt for query: '{query}'")
    
    # Import multi-graph detection
    from utils.plot_helpers import detect_multi_graph_result
    
    # Check if this is a multi-graph result first
    is_multi_graph, figures, data_dfs = detect_multi_graph_result(result)
    
    if is_multi_graph:
        result_type = "multi-graph visualization"
        result_context = f"""This is a multi-graph result containing {len(figures)} visualizations.

MULTI-GRAPH ANALYSIS:
You have access to {len(figures)} different charts/plots that were created to answer the user's query.

UNDERLYING DATA FOR EACH CHART:"""
        
        # Add data for each figure
        for i, fig in enumerate(figures):
            data_df = data_dfs[i] if data_dfs and i < len(data_dfs) else None
            
            if data_df is not None and hasattr(data_df, 'to_string'):
                result_context += f"""

CHART {i+1} DATA:
{data_df.to_string(max_rows=15, max_cols=8)}"""
            else:
                # Extract data from Plotly figure if available
                extracted_data = []
                if hasattr(fig, 'data') and fig.data:
                    for j, trace in enumerate(fig.data):
                        trace_data = {}
                        if hasattr(trace, 'x') and trace.x is not None:
                            trace_data['x'] = list(trace.x)
                        if hasattr(trace, 'y') and trace.y is not None:
                            trace_data['y'] = list(trace.y)
                        if hasattr(trace, 'name') and trace.name:
                            trace_data['name'] = trace.name
                        if hasattr(trace, 'type'):
                            trace_data['type'] = trace.type
                        
                        if trace_data:
                            extracted_data.append(trace_data)
                
                if extracted_data:
                    result_context += f"""

CHART {i+1} EXTRACTED DATA:
Chart Type: {extracted_data[0].get('type', 'unknown')}
Data Points: {len(extracted_data[0].get('x', []))} values"""
                    
                    # Add sample data values
                    for j, trace_data in enumerate(extracted_data):
                        if 'x' in trace_data and 'y' in trace_data:
                            result_context += f"""
Trace {j+1} ({trace_data.get('type', 'unknown')}):
X values: {trace_data['x'][:5]}...
Y values: {trace_data['y'][:5]}..."""
                        elif 'x' in trace_data:
                            result_context += f"""
Trace {j+1} ({trace_data.get('type', 'unknown')}):
Values: {trace_data['x'][:5]}..."""
                else:
                    result_context += f"""

CHART {i+1}: No specific data available, but you can analyze the visual patterns."""
        
        result_context += f"""

ANALYSIS REQUIREMENTS:
- Analyze each chart individually and explain what it shows
- Identify relationships between the different charts
- Provide insights based on the combined multi-graph analysis
- Reference specific data values from each chart's underlying data
- Explain how the multiple visualizations work together to answer the user's query"""
        
        logger.info(f"🎯 Multi-graph result detected: {len(figures)} figures with data")
    
    elif isinstance(result, str) and result.startswith("Error executing code"):
        result_type = "execution error"
        result_context = f"The code execution failed with the following error:\n{result}"
        logger.info("🎯 Error result detected")
    
    elif isinstance(result, tuple) and len(result) == 2:
        # New dual-output format: (fig, data_df) or (fig, dict_of_dataframes)
        fig, data_content = result
        if isinstance(fig, (plt.Figure, plt.Axes)):
            if hasattr(data_content, 'to_string'):  # DataFrame
                result_type = "dual output (plot + data)"
                result_context = f"""This is a dual-output result containing both a visualization and the underlying data.

PLOT: A {type(fig).__name__} showing the visualization requested by the user.

UNDERLYING DATA:
{data_content.to_string(max_rows=20, max_cols=10)}

You have access to both the visual representation AND the specific numerical values. Use both in your analysis."""
                logger.info(f"🎯 Dual-output result detected: plot + data with {len(data_content)} rows")
            elif isinstance(data_content, dict):
                # Dictionary of DataFrames
                result_type = "dual output (plot + multiple datasets)"
                data_summary = []
                data_details = []
                for key, df in data_content.items():
                    if hasattr(df, 'to_string'):
                        data_summary.append(f"{key}: {len(df)} rows")
                        data_details.append(f"\n{key.upper()} DATASET:\n{df.to_string(max_rows=20, max_cols=10)}")
                
                result_context = f"""This is a dual-output result containing both a visualization and multiple underlying datasets.

PLOT: A {type(fig).__name__} showing the visualization requested by the user.

UNDERLYING DATASETS SUMMARY:
{chr(10).join(data_summary)}

DETAILED DATASET CONTENTS:
{chr(10).join(data_details)}

You have access to both the visual representation AND the specific numerical values from multiple datasets. Use both in your analysis."""
                logger.info(f"🎯 Dual-output result detected: plot + multiple datasets")
            else:
                result_type = "unknown tuple"
                result_context = f"Tuple result with types: {type(result[0])}, {type(result[1])}"
                logger.warning("🎯 Unexpected tuple format detected")
        else:
            result_type = "unknown tuple"
            result_context = f"Tuple result with types: {type(result[0])}, {type(result[1])}"
            logger.warning("🎯 Unexpected tuple format detected")
    
    elif isinstance(result, (plt.Figure, plt.Axes)):
        result_type = "visualization"
        result_context = f"A {type(result).__name__} plot/chart showing the requested visualization."
        logger.info("🎯 Visualization result detected")
    
    elif hasattr(result, 'to_string'):  # DataFrame or Series
        result_type = "data table"
        result_context = f"""The analysis returned a {type(result).__name__} with the following data:

{result.to_string(max_rows=20, max_cols=10)}"""
        logger.info(f"🎯 Data table result detected: {type(result).__name__}")
    
    elif isinstance(result, (int, float)):
        result_type = "numerical value"
        result_context = f"The analysis returned a single numerical result: {result}"
        logger.info("🎯 Numerical result detected")
    
    elif isinstance(result, dict):
        result_type = "dictionary data"
        result_context = f"The analysis returned a dictionary with the following key-value pairs:\n{str(result)}"
        logger.info("🎯 Dictionary result detected")
    
    elif isinstance(result, list):
        result_type = "list data"
        result_context = f"The analysis returned a list with {len(result)} items: {str(result)[:500]}..."
        logger.info("🎯 List result detected")
    
    else:
        result_type = "other"
        result_context = f"The analysis returned: {str(result)[:500]}..."
        logger.info("🎯 Other result type detected")
    
    prompt = f"""You are analyzing the result of a data analysis query. Here are the details:

**USER'S ORIGINAL QUERY:** "{query}"

**RESULT TYPE:** {result_type}

**ANALYSIS RESULT:**
{result_context}

Your task is to provide comprehensive business insights and actionable recommendations based on this result. Follow this approach:

1. **INTERPRET THE SPECIFIC FINDINGS**: What exactly do these results show? Include actual numbers, values, and concrete findings.

2. **BUSINESS CONTEXT**: What do these findings mean for business operations, strategy, or decision-making?

3. **KEY INSIGHTS**: What are the most important patterns, trends, or discoveries revealed by this analysis?

4. **ACTIONABLE RECOMMENDATIONS**: What specific actions should stakeholders take based on these findings?

5. **FOLLOW-UP QUESTIONS**: What additional analyses would provide deeper insights or address related business questions?

Focus on being specific, actionable, and business-oriented in your response. Reference the actual data values and findings in your analysis."""

    logger.info(f"🎯 Generated reasoning prompt: {len(prompt)} characters")
    return prompt


def ReasoningAgent(query: str, result: Any):
    """Streams the LLM's reasoning about the result (plot or value) and extracts model 'thinking' and final explanation.
    
    Note: This function has UI dependencies (streamlit) that should be refactored for better separation of concerns.
    """
    logger.info(f"🧠 ReasoningAgent: Starting reasoning for query: '{query}'")
    logger.info(f"🧠 Result type: {type(result)}")
    logger.info(f"🧠 Result content preview: {str(result)[:200]}...")
    
    # Get system prompt context
    system_prompt_agent = SystemPromptMemoryAgent()
    
    prompt = ReasoningCurator(query, result)
    logger.info(f"🧠 Generated prompt length: {len(prompt)} characters")
    
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))

    logger.info("📤 Sending reasoning request to LLM (streaming)...")
    
    # Base system prompt for reasoning
    base_system_prompt = """detailed thinking on. You are an expert data analyst and business strategist. Your primary role is to translate raw code outputs (plots, data, or errors) into comprehensive, actionable business insights for decision-makers.

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

# SPECIAL INSTRUCTIONS FOR MULTI-GRAPH RESULTS:
When analyzing multi-graph results, you have access to multiple visualizations and their underlying data. Your analysis should:
1. Analyze each chart individually with specific data values
2. Identify relationships between the different charts
3. Reference actual numbers from each chart's data
4. Explain how the multiple visualizations work together
5. Provide insights based on the combined analysis

For example: "The bar chart shows Product A leading in sales ($125,000), the scatter plot reveals a strong correlation between age and salary, and the histogram shows the age distribution with most employees between 25-35 years old."

-- Include specific recommendations based on the actual data found

Begin by streaming your detailed analytical process within `<think>...</think>` tags. After your thought process is complete, provide a comprehensive, detailed explanation to the user outside the tags that includes the actual results."""
    
    # Apply system prompt if active
    system_prompt = system_prompt_agent.apply_system_prompt(base_system_prompt)
    logger.info(f"📤 System prompt length: {len(system_prompt)} characters")
    logger.info(f"📤 User prompt length: {len(prompt)} characters")
    
    # Streaming LLM call
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    response = make_llm_call(
        messages=messages,
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
        # Robust handling for streaming chunks that may be malformed or empty
        try:
            # Skip empty chunks
            if not chunk:
                continue

            # Ensure the chunk has choices and at least one choice
            choices = getattr(chunk, "choices", None)
            if not choices or len(choices) == 0:
                continue

            # Take the first choice (OpenAI returns one choice for streaming)
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            token = getattr(delta, "content", None)
            if token is None:
                continue
        except Exception as e:
            # Log and skip any malformed chunk to prevent entire reasoning from crashing
            logger.warning(f"⚠️ Skipping malformed stream chunk: {e}")
            continue

        # If we reach here, we have a valid token to process
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