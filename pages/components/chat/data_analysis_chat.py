"""Chat interface for data analysis."""

import logging
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from agents import CodeGenerationAgent, ExecutionAgent, ReasoningAgent
from utils.plot_helpers import (
    PlotMemory, is_plot_modification_request, generate_plot_modification_code,
    detect_multi_graph_result
)

logger = logging.getLogger(__name__)

def initialize_chat_session_state():
    """Initialize session state for chat."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "plots" not in st.session_state:
        st.session_state.plots = []
    if "plot_data" not in st.session_state:
        st.session_state.plot_data = []
    if "plot_memory" not in st.session_state:
        st.session_state.plot_memory = PlotMemory()

def process_user_query(user_q, df):
    """Process user query and generate response."""
    logger.info(f"💬 User query received: '{user_q}'")
    st.session_state.messages.append({"role": "user", "content": user_q})
    
    # Enhanced loading state
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown("""
        <div class="loading-container">
            <div class="loading-spinner"></div>
            <div class="loading-text">🧠 Analyzing your data...</div>
        </div>
        """, unsafe_allow_html=True)
    
    start_time = datetime.now()
    logger.info(f"⏱️ Processing started at {start_time}")
    
    try:
        # Check if this is a plot modification request
        if is_plot_modification_request(user_q) and st.session_state.plot_memory.plots:
            logger.info("🔄 Detected plot modification request")
            target_plot = st.session_state.plot_memory.get_plot_by_reference(user_q)
            if target_plot:
                # Generate modification code
                modification_prompt = generate_plot_modification_code(user_q, target_plot, df)
                code, should_plot_flag, code_thinking = CodeGenerationAgent(
                    f"Modify the previous plot: {user_q}", 
                    df, 
                    st.session_state.messages,
                    plot_engine=st.session_state.get("plot_engine", "plotly")
                )
            else:
                # Fall back to normal processing
                code, should_plot_flag, code_thinking = CodeGenerationAgent(
                    user_q, df, st.session_state.messages,
                    plot_engine=st.session_state.get("plot_engine", "plotly")
                )
        else:
            # Normal processing
            code, should_plot_flag, code_thinking = CodeGenerationAgent(
                user_q, df, st.session_state.messages,
                plot_engine=st.session_state.get("plot_engine", "plotly")
            )
        
        result_obj = ExecutionAgent(code, df, should_plot_flag)
        
        # Auto-retry logic for common pandas errors
        if isinstance(result_obj, str) and result_obj.startswith("Error executing code:"):
            logger.warning("🔄 Code execution failed, attempting automatic retry with error context")
            error_context = result_obj
            
            # Try once more with error context
            try:
                code_retry, should_plot_flag_retry, _ = CodeGenerationAgent(
                    user_q, df, st.session_state.messages, 
                    retry_context=error_context,
                    plot_engine=st.session_state.get("plot_engine", "plotly")
                )
                result_obj_retry = ExecutionAgent(code_retry, df, should_plot_flag_retry)
                
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

    except Exception as e:
        # Clear loading state on error
        loading_placeholder.empty()
        logger.error(f"❌ Error processing query: {e}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ **Error processing your request:**\n\n{str(e)}\n\nPlease try again or check your data.",
            "code": "# Error occurred during processing"
        })
        st.rerun()
        return

    # Build assistant response - handle multi-graph and dual output format
    is_multi_graph, figures, data_dfs = detect_multi_graph_result(result_obj)
    is_dual_output = isinstance(result_obj, tuple) and len(result_obj) == 2 and isinstance(result_obj[0], (plt.Figure, plt.Axes, go.Figure)) and isinstance(result_obj[1], pd.DataFrame)
    is_plot = isinstance(result_obj, (plt.Figure, plt.Axes, go.Figure))
    plot_idx = None
    data_idx = None
    
    if is_dual_output:
        # Handle new dual-output format (fig, data_df)
        fig, data_df = result_obj
        
        # Store plot in session state
        st.session_state.plots.append(fig)
        plot_idx = len(st.session_state.plots) - 1
        
        # Store data (create data storage if it doesn't exist)
        if "plot_data" not in st.session_state:
            st.session_state.plot_data = []
        st.session_state.plot_data.append(data_df)
        data_idx = len(st.session_state.plot_data) - 1
        
        # Store in plot memory system for Phase 2 enhancements
        plot_theme = getattr(st.session_state, 'plot_theme', 'professional')
        show_insights = getattr(st.session_state, 'show_insights', True)
        
        # Detect chart type from the plot
        chart_type = 'bar'  # Default, could be enhanced with better detection
        if hasattr(fig, 'data') and hasattr(fig, 'layout'):
            # Plotly figure - detect type from first trace
            if fig.data and len(fig.data) > 0:
                trace_type = fig.data[0].type
                if trace_type == 'histogram':
                    chart_type = 'histogram'
                elif trace_type == 'box':
                    chart_type = 'box'
                elif trace_type == 'violin':
                    chart_type = 'violin'
                elif trace_type == 'scatter':
                    chart_type = 'scatter'
                elif trace_type == 'bar':
                    chart_type = 'bar'
                else:
                    chart_type = trace_type
        elif hasattr(fig, 'axes') and fig.axes:
            # Matplotlib figure - original detection logic
            ax = fig.axes[0]
            if len(ax.get_children()) > 0:
                # Simple chart type detection
                children = ax.get_children()
                if any(isinstance(child, matplotlib.patches.Rectangle) for child in children):
                    chart_type = 'bar'
                elif any(isinstance(child, matplotlib.lines.Line2D) for child in children):
                    chart_type = 'line'
                elif any(isinstance(child, matplotlib.collections.PathCollection) for child in children):
                    chart_type = 'scatter'
        
        # Add to plot memory
        memory_idx = st.session_state.plot_memory.add_plot(
            fig=fig,
            data_df=data_df,
            context=user_q,
            chart_type=chart_type,
            styling={'theme': plot_theme, 'insights': show_insights}
        )
        
        header = "Here is your enhanced visualization with underlying data:"
        logger.info(f"📊 Enhanced plot added: plot at index {plot_idx}, data at index {data_idx}, memory at {memory_idx} ({len(data_df)} rows)")
        
    elif is_multi_graph:
        # Handle multi-graph format
        plot_indices = []
        data_indices = []
        
        for i, fig in enumerate(figures):
            # Store plot in session state
            st.session_state.plots.append(fig)
            plot_idx = len(st.session_state.plots) - 1
            plot_indices.append(plot_idx)
            
            # Store data if available
            data_df = data_dfs[i] if data_dfs and i < len(data_dfs) else df
            if "plot_data" not in st.session_state:
                st.session_state.plot_data = []
            st.session_state.plot_data.append(data_df)
            data_idx = len(st.session_state.plot_data) - 1
            data_indices.append(data_idx)
            
            # Add to plot memory
            memory_idx = st.session_state.plot_memory.add_plot(
                fig=fig,
                data_df=data_df,
                context=user_q,
                chart_type='multi_graph',
                styling={'theme': 'professional', 'insights': False}
            )
        
        header = f"Here are your {len(figures)} visualizations:"
        logger.info(f"📊 Multi-graph added: {len(figures)} plots at indices {plot_indices}, data at indices {data_indices}")
        
    elif is_plot:
        # Handle legacy single plot format
        fig = result_obj.figure if isinstance(result_obj, plt.Axes) else result_obj
        st.session_state.plots.append(fig)
        plot_idx = len(st.session_state.plots) - 1
        
        # Also store in plot memory
        memory_idx = st.session_state.plot_memory.add_plot(
            fig=fig,
            data_df=df,  # Use current dataframe
            context=user_q,
            chart_type='unknown',
            styling={'theme': 'professional', 'insights': False}
        )
        
        header = "Here is the visualization you requested:"
        logger.info(f"📊 Legacy plot added to session state at index {plot_idx}, memory at {memory_idx}")
        
    elif isinstance(result_obj, (pd.DataFrame, pd.Series)):
        header = f"Result: {len(result_obj)} rows" if isinstance(result_obj, pd.DataFrame) else "Result series"
        logger.info(f"📄 Data result: {header}")
    else:
        header = f"Result: {result_obj}"
        logger.info(f"📄 Scalar result: {header}")

    # Build content with collapsible reasoning like test.py
    thinking_html = ""
    if raw_thinking:
        thinking_html = (
            '<details class="thinking">'
            '<summary>🧠 Reasoning</summary>'
            f'<pre>{raw_thinking}</pre>'
            '</details>'
        )

    explanation_html = reasoning_txt

    # Visible content mirrors test.py (thinking block + explanation)
    detailed_reasoning = explanation_html  # keep full explanation for downloads
    visible_content = f"{thinking_html}{explanation_html}"

    # Store message with appropriate plot indices
    message_data = {
        "role": "assistant",
        "content": visible_content,
        "detailed_reasoning": detailed_reasoning,
        "code": code
    }
    
    if is_multi_graph:
        message_data["plot_indices"] = plot_indices
        message_data["data_indices"] = data_indices
    else:
        message_data["plot_index"] = plot_idx
        message_data["data_index"] = data_idx
    
    st.session_state.messages.append(message_data)
    
    # Clear loading state
    loading_placeholder.empty()
    
    logger.info("✅ Response added to chat history, rerunning app")
    st.rerun()
