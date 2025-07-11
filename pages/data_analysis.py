import io
import logging
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import chardet
from datetime import datetime
from typing import Dict, Any

from agents import (
    ColumnMemoryAgent, CodeGenerationAgent, ExecutionAgent, 
    ReasoningAgent, DataInsightAgent
)
from agents.memory import SystemPromptMemoryAgent
from utils.system_prompts import get_prompt_manager

logger = logging.getLogger(__name__)

def render_system_prompt_sidebar():
    """Render system prompt controls in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 AI Behavior")
    
    prompt_manager = get_prompt_manager()
    system_prompt_agent = SystemPromptMemoryAgent()
    
    # Get current active prompt
    active_prompt = prompt_manager.get_active_prompt()
    current_selection = active_prompt.name if active_prompt else "Default"
    
    # Get all available prompts
    all_prompts = prompt_manager.list_prompts()
    prompt_options = ["Default"] + [p.name for p in all_prompts]
    
    # Create selectbox for quick prompt switching
    selected_prompt = st.sidebar.selectbox(
        "Analysis Style:",
        options=prompt_options,
        index=prompt_options.index(current_selection) if current_selection in prompt_options else 0,
        help="Choose how the AI should approach your analysis"
    )
    
    # Apply button
    if st.sidebar.button("🔄 Apply Style", use_container_width=True):
        if selected_prompt == "Default":
            system_prompt_agent.clear_active_prompt()
            st.sidebar.success("✅ Using default AI behavior")
        else:
            success = system_prompt_agent.set_active_prompt(selected_prompt)
            if success:
                st.sidebar.success(f"✅ Applied: {selected_prompt}")
            else:
                st.sidebar.error(f"❌ Failed to apply: {selected_prompt}")
        st.rerun()
    
    # Show active prompt info
    if active_prompt:
        with st.sidebar.expander("📋 Active Style Details", expanded=False):
            st.markdown(f"**{active_prompt.name}**")
            st.markdown(f"*{active_prompt.description}*")
            st.markdown(f"**Category:** {active_prompt.category}")
            if active_prompt.tags:
                st.markdown(f"**Tags:** {', '.join(active_prompt.tags)}")
    else:
        st.sidebar.info("ℹ️ Using default AI behavior")
    
    # Quick link to full prompt manager
    if st.sidebar.button("🎯 Manage All Prompts", use_container_width=True):
        st.sidebar.info("💡 Use the main navigation to access the System Prompt Manager page")

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

    # Add system prompt controls and logout button in the sidebar
    render_system_prompt_sidebar()
    
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
        st.markdown("<medium>Powered by Azure OpenAI GPT-4.1</medium>", unsafe_allow_html=True)
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
                    from app_core.api import make_llm_call
                    st.session_state.insights = DataInsightAgent(st.session_state.df, make_llm_call)
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