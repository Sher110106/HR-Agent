# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, io, re, logging
import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from datetime import datetime
import chardet

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
    system_prompt = "detailed thinking off. You are an assistant that determines if a query is requesting a data visualization. Consider the conversation context to understand follow-up requests. Respond with only 'true' if the query is asking for a plot, chart, graph, or any visual representation of data. Otherwise, respond with 'false'."
    
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
        max_tokens=5  # We only need a short response
    )
    
    # Extract the response and convert to boolean
    intent_response = response.choices[0].message.content.strip().lower()
    result = intent_response == "true"
    logger.info(f"📥 LLM response: '{intent_response}' → Visualization needed: {result}")
    
    return result

# ------------------  PlotCodeGeneratorTool ---------------------------
def PlotCodeGeneratorTool(cols: List[str], query: str, df: pd.DataFrame, conversation_context: str = "") -> str:
    """Generate a prompt for the LLM to write pandas+matplotlib code for a plot based on the query and columns."""
    logger.info(f"📊 PlotCodeGeneratorTool: Generating plot prompt for columns: {cols}")
    
    # Get data types and sample values for better context
    data_info = []
    date_columns = []
    
    for col in cols:
        dtype = str(df[col].dtype)
        
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
    
    prompt = f"""
    Given DataFrame `df` with columns and data types:
    {data_context}
    {context_section}{date_instructions}
    Write Python code using pandas **and matplotlib** (as plt) to answer:
    "{query}"

    Rules & Available Tools
    ----------------------
         1. Use pandas and matplotlib.pyplot (as plt) - `pd`, `np`, `df`, `plt` are all available in scope.
    2. For date/time columns, prefer smart_date_parser(df, 'column_name') for robust parsing.
    3. For categorical columns, convert to numeric: df['col'].map({{'Yes': 1, 'No': 0}}).
    4. CRITICAL: Create figure with `fig, ax = plt.subplots(figsize=(8,5))` and assign `result = fig`.
    5. Create ONE clear, well-labeled plot with ax.set_title(), ax.set_xlabel(), ax.set_ylabel().
    6. For time series, consider df.set_index('date_col').plot() for automatic time formatting.
    7. Handle missing values with .dropna() before plotting if needed.
    8. Use clear colors and markers: ax.scatter(), ax.plot(), ax.bar(), etc.
    9. Wrap code in ```python fence with no explanations.

    Plotting Examples:
    - Scatter: ax.scatter(df['x'], df['y'], alpha=0.6)
    - Time series: df.set_index('date').plot(ax=ax)
    - Correlation heatmap: sns.heatmap(df.corr(), ax=ax) (if seaborn available)
    """
    logger.debug(f"Generated plot prompt: {prompt[:200]}...")
    return prompt

# ------------------  CodeWritingTool ---------------------------------
def CodeWritingTool(cols: List[str], query: str, df: pd.DataFrame, conversation_context: str = "") -> str:
    """Generate a prompt for the LLM to write pandas-only code for a data query (no plotting)."""
    logger.info(f"📝 CodeWritingTool: Generating code prompt for columns: {cols}")
    
    # Get data types and sample values for better context
    data_info = []
    date_columns = []
    
    for col in cols:
        dtype = str(df[col].dtype)
        
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
    
    prompt = f"""
    Given DataFrame `df` with columns and data types:
    {data_context}
    {context_section}{date_instructions}
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
    """
    logger.debug(f"Generated code prompt: {prompt[:200]}...")
    return prompt

# === CodeGenerationAgent ==============================================

def CodeGenerationAgent(query: str, df: pd.DataFrame, chat_history: List[Dict] = None):
    """Selects the appropriate code generation tool and gets code from the LLM for the user's query."""
    logger.info(f"🤖 CodeGenerationAgent: Processing query: '{query}'")
    logger.info(f"📊 DataFrame info: {len(df)} rows, {len(df.columns)} columns")
    
    # Extract conversation context for better understanding
    conversation_context = ""
    if chat_history:
        conversation_context = ConversationMemoryTool(chat_history)
        logger.info(f"🧠 Using conversation context: {len(conversation_context)} characters")
    
    should_plot = QueryUnderstandingTool(query, conversation_context)
    prompt = PlotCodeGeneratorTool(df.columns.tolist(), query, df, conversation_context) if should_plot else CodeWritingTool(df.columns.tolist(), query, df, conversation_context)

    messages = [
        {"role": "system", "content": "detailed thinking off. You are a senior data scientist with expertise in pandas and statistical analysis. Write clean, efficient, production-ready code. Focus on:\n\n1. CORRECTNESS: Ensure proper variable scoping (pd, df are available)\n2. ROBUSTNESS: Handle missing values and edge cases\n3. CLARITY: Use descriptive variable names and clear logic\n4. EFFICIENCY: Prefer vectorized operations over loops\n5. BEST PRACTICES: Follow pandas conventions and data analysis patterns\n\nOutput ONLY a properly-closed ```python code block. Use smart_date_parser() for date parsing. Assign final result to 'result' variable."},
        {"role": "user", "content": prompt}
    ]

    logger.info("📤 Sending code generation request to LLM...")
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=messages,
        temperature=0.2,
        max_tokens=1024
    )

    full_response = response.choices[0].message.content
    logger.info(f"📥 LLM full response length: {len(full_response)} characters")
    
    code = extract_first_code_block(full_response)
    logger.info(f"✂️ Extracted code block length: {len(code)} characters")
    logger.info(f"💻 Generated code:\n{code}")
    
    return code, should_plot, ""

# === ExecutionAgent ====================================================

def ExecutionAgent(code: str, df: pd.DataFrame, should_plot: bool):
    """Executes the generated code in a controlled environment and returns the result or error message."""
    logger.info(f"⚡ ExecutionAgent: Executing code (plot mode: {should_plot})")
    logger.info(f"🔧 Code to execute:\n{code}")
    
    env = {"pd": pd, "np": np, "df": df, "smart_date_parser": smart_date_parser}
    if should_plot:
        plt.rcParams["figure.dpi"] = 100  # Set default DPI for all figures
        env["plt"] = plt
        env["io"] = io
        logger.info("🎨 Plot environment set up with matplotlib")
    
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
        
        return error_msg

# === ReasoningCurator TOOL =========================================
def ReasoningCurator(query: str, result: Any) -> str:
    """Builds and returns the LLM prompt for reasoning about the result."""
    logger.info(f"🧠 ReasoningCurator: Creating reasoning prompt for result type: {type(result).__name__}")
    
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))

    if is_error:
        desc = result
        logger.info("❌ Result is an error")
    elif is_plot:
        title = ""
        if isinstance(result, plt.Figure):
            title = result._suptitle.get_text() if result._suptitle else ""
        elif isinstance(result, plt.Axes):
            title = result.get_title()
        desc = f"[Plot Object: {title or 'Chart'}]"
        logger.info(f"📊 Result is a plot: {desc}")
    else:
        desc = str(result)[:300]
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
        The result value is: {desc}
        Explain in 2–3 concise sentences what this tells about the data (no mention of charts).'''
    
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
            {"role": "system", "content": "detailed thinking on. You are an insightful data analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1024,
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
                {"role": "system", "content": "detailed thinking off. You are a data analyst providing brief, focused insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=512
        )
        insights = response.choices[0].message.content
        logger.info(f"📥 Generated insights length: {len(insights)} characters")
        logger.debug(f"Insights: {insights[:200]}...")
        return insights
    except Exception as exc:
        error_msg = f"Error generating dataset insights: {exc}"
        logger.error(f"❌ DataInsightAgent failed: {error_msg}")
        return error_msg

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

        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)
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
                    
                    # Pass chat history to enable conversational memory
                    code, should_plot_flag, code_thinking = CodeGenerationAgent(user_q, st.session_state.df, st.session_state.messages)
                    result_obj = ExecutionAgent(code, st.session_state.df, should_plot_flag)
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