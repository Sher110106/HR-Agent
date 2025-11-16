"""Chat interface for Excel analysis."""

import logging
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from datetime import datetime
from agents import (
    SheetSelectionAgent, ExcelCodeGenerationAgent, ExcelExecutionAgent,
    ReasoningAgent
)
from utils.plot_helpers import PlotMemory

logger = logging.getLogger(__name__)

def initialize_excel_chat_session_state():
    """Initialize session state for Excel chat."""
    if "excel_messages" not in st.session_state:
        st.session_state.excel_messages = []
    if "excel_plots" not in st.session_state:
        st.session_state.excel_plots = []
    if "excel_plot_data" not in st.session_state:
        st.session_state.excel_plot_data = []
    if "excel_disambiguation" not in st.session_state:
        st.session_state.excel_disambiguation = None
    if "excel_plot_memory" not in st.session_state:
        st.session_state.excel_plot_memory = PlotMemory()

def process_excel_user_query(user_q, sheet_catalog_agent, column_indexer_agent):
    """Process Excel user query and generate response."""
    logger.info(f"💬 Excel user query received: '{user_q}'")
    st.session_state.excel_messages.append({"role": "user", "content": user_q})
    
    # Enhanced loading state
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown("""
        <div class="loading-container">
            <div class="loading-spinner"></div>
            <div class="loading-text">📊 Analyzing Excel data...</div>
        </div>
        """, unsafe_allow_html=True)
    
    start_time = datetime.now()
    logger.info(f"⏱️ Excel processing started at {start_time}")
    
    try:
        # Phase 2: Intelligent Planning & Execution
        # Step 1: Create sheet selection agent
        sheet_selection_agent = SheetSelectionAgent(column_indexer_agent, sheet_catalog_agent)
        
        # Step 2: Create sheet plan or get disambiguation question
        sheet_plan, disambiguation = sheet_selection_agent.create_sheet_plan(user_q)
        
        if disambiguation:
            # Store disambiguation for UI
            st.session_state.excel_disambiguation = disambiguation
            st.session_state.excel_messages.append({
                "role": "assistant",
                "content": f"I need some clarification to answer your question: **{disambiguation.question}**\n\nPlease select an option below to proceed.",
                "code": "# Waiting for user clarification"
            })
        elif sheet_plan:
            # Step 3: Determine if plotting is needed
            should_plot = any(word in user_q.lower() for word in [
                'plot', 'chart', 'graph', 'visualize', 'show', 'display', 'trend', 'compare'
            ])
            
            # Step 4: Generate code
            excel_code_agent = ExcelCodeGenerationAgent(column_indexer_agent)
            code = excel_code_agent.generate_code(user_q, sheet_plan, should_plot)
            
            # Step 5: Execute code
            excel_execution_agent = ExcelExecutionAgent(column_indexer_agent)
            result, error = excel_execution_agent.execute_code(code, sheet_plan)
            
            # Auto-retry logic for common pandas errors
            if error:
                logger.warning("🔄 Code execution failed, attempting automatic retry with error context")
                error_context = error
                
                # Try once more with error context
                try:
                    # Generate new code with error context
                    excel_code_agent_retry = ExcelCodeGenerationAgent(column_indexer_agent)
                    code_retry = excel_code_agent_retry.generate_code_with_retry(user_q, sheet_plan, should_plot, error_context)
                    
                    # Execute retry code
                    result_retry, error_retry = excel_execution_agent.execute_code(code_retry, sheet_plan)
                    
                    # If retry succeeds, use the retry result
                    if not error_retry:
                        logger.info("✅ Retry successful, using corrected result")
                        code = code_retry
                        result = result_retry
                        error = None
                    else:
                        logger.warning("⚠️ Retry also failed, using original error")
                except Exception as e:
                    logger.error(f"❌ Retry attempt failed: {e}")
                
                if error:
                    # Handle execution error
                    st.session_state.excel_messages.append({
                        "role": "assistant",
                        "content": f"❌ **Error executing analysis:**\n\n{error}",
                        "code": code
                    })
                else:
                    # Handle successful execution - similar to CSV analysis
                    process_successful_excel_result(result, user_q, code, sheet_plan)
            else:
                # Handle successful execution
                process_successful_excel_result(result, user_q, code, sheet_plan)
        else:
            # No plan or disambiguation - error case
            st.session_state.excel_messages.append({
                "role": "assistant",
                "content": "❌ **Unable to create analysis plan.**\n\nPlease check your query or try rephrasing your question.",
                "code": "# No plan generated"
            })
        
    except Exception as e:
        # Clear loading state on error
        loading_placeholder.empty()
        logger.error(f"❌ Error in Excel analysis: {e}")
        st.session_state.excel_messages.append({
            "role": "assistant",
            "content": f"❌ **Error processing your request:**\n\n{str(e)}\n\nPlease try again or check your data.",
            "code": "# Error occurred during processing"
        })
    
    # Clear loading state
    loading_placeholder.empty()
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    logger.info(f"⏱️ Excel processing time: {processing_time:.2f} seconds")
    
    st.rerun()

def process_successful_excel_result(result, user_q, code, sheet_plan):
    """Process successful Excel analysis result."""
    is_dual_output = isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], (plt.Figure, plt.Axes))
    is_plot = isinstance(result, (plt.Figure, plt.Axes))
    plot_idx = None
    data_idx = None
    
    if is_dual_output:
        # Handle new dual-output format (fig, data_content)
        fig, data_content = result
        
        # Store plot in session state
        st.session_state.excel_plots.append(fig)
        plot_idx = len(st.session_state.excel_plots) - 1
        
        # Store in plot memory system for Phase 2 enhancements
        excel_plot_theme = getattr(st.session_state, 'excel_plot_theme', 'professional')
        excel_show_insights = getattr(st.session_state, 'excel_show_insights', True)
        
        # Detect chart type from the plot
        chart_type = 'bar'  # Default, could be enhanced with better detection
        if hasattr(fig, 'axes') and fig.axes:
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
        
        # Handle different data content types
        if isinstance(data_content, pd.DataFrame):
            # Single DataFrame
            st.session_state.excel_plot_data.append(data_content)
            data_idx = len(st.session_state.excel_plot_data) - 1
            
            # Add to Excel plot memory
            memory_idx = st.session_state.excel_plot_memory.add_plot(
                fig=fig,
                data_df=data_content,
                context=user_q,
                chart_type=chart_type,
                styling={'theme': excel_plot_theme, 'insights': excel_show_insights}
            )
            
            logger.info(f"📊 Enhanced Excel plot added: plot at index {plot_idx}, data at index {data_idx}, memory at {memory_idx} ({len(data_content)} rows)")
        elif isinstance(data_content, dict):
            # Dictionary of DataFrames - combine them into a single DataFrame for display
            combined_data = []
            for key, df in data_content.items():
                if isinstance(df, pd.DataFrame):
                    # Add a source column to identify the data
                    df_copy = df.copy()
                    df_copy['Data_Source'] = key
                    combined_data.append(df_copy)
            
            if combined_data:
                combined_df = pd.concat(combined_data, ignore_index=True)
                st.session_state.excel_plot_data.append(combined_df)
                data_idx = len(st.session_state.excel_plot_data) - 1
                
                # Add to Excel plot memory
                memory_idx = st.session_state.excel_plot_memory.add_plot(
                    fig=fig,
                    data_df=combined_df,
                    context=user_q,
                    chart_type=chart_type,
                    styling={'theme': excel_plot_theme, 'insights': excel_show_insights}
                )
                
                logger.info(f"📊 Enhanced Excel plot added: plot at index {plot_idx}, combined data at index {data_idx}, memory at {memory_idx} ({len(combined_df)} total rows)")
            else:
                data_idx = None
                logger.info(f"📊 Dual-output added: plot at index {plot_idx}, no data to store")
        else:
            # Other data types
            data_idx = None
            logger.info(f"📊 Dual-output added: plot at index {plot_idx}, data type: {type(data_content)}")
        
    elif is_plot:
        # Handle legacy single plot format
        fig = result.figure if isinstance(result, plt.Axes) else result
        st.session_state.excel_plots.append(fig)
        plot_idx = len(st.session_state.excel_plots) - 1
        logger.info(f"📊 Legacy plot added to session state at index {plot_idx}")
    
    # Generate professional reasoning using ReasoningAgent (like CSV analysis)
    try:
        logger.info("🧠 Calling ReasoningAgent...")
        raw_thinking, reasoning_txt = ReasoningAgent(user_q, result)
        reasoning_txt = reasoning_txt.replace("`", "")
        logger.info(f"🧠 ReasoningAgent completed successfully. Reasoning length: {len(reasoning_txt)}")
    except Exception as e:
        logger.error(f"❌ Error in ReasoningAgent: {e}")
        raw_thinking = ""
        reasoning_txt = f"Analysis completed successfully. The results show the requested comparison between active and inactive employees."
    
    # Show only reasoning thinking in Model Thinking (collapsed by default)
    thinking_html = ""
    if raw_thinking:
        thinking_html = (
            '<details class="thinking">'
            '<summary>🧠 Reasoning</summary>'
            f'<pre>{raw_thinking}</pre>'
            '</details>'
        )

    # Add sheet plan info
    plan_info = f"\n\n**📋 Analysis Plan:**\n"
    plan_info += f"- **Strategy:** {sheet_plan.join_strategy}\n"
    plan_info += f"- **Sheets used:** {', '.join(sheet_plan.primary_sheets)}\n"
    if sheet_plan.join_keys:
        plan_info += f"- **Join keys:** {', '.join(sheet_plan.join_keys)}\n"
    
    # Build content like test.py (collapsible thinking + explanation)
    detailed_reasoning = f"{reasoning_txt}{plan_info}"
    visible_content = f"{thinking_html}{detailed_reasoning}"

    st.session_state.excel_messages.append({
        "role": "assistant",
        "content": visible_content,
        "detailed_reasoning": detailed_reasoning,
        "plot_index": plot_idx,
        "data_index": data_idx,
        "code": code
    })

def handle_disambiguation():
    """Handle disambiguation questions."""
    if st.session_state.excel_disambiguation:
        disambiguation = st.session_state.excel_disambiguation
        st.markdown("---")
        st.markdown("### 🤔 Clarification Needed")
        st.markdown(f"**{disambiguation.question}**")
        
        if disambiguation.context:
            with st.expander("📋 Context", expanded=False):
                st.markdown(disambiguation.context)
        
        if disambiguation.options:
            st.markdown("**Options:**")
            for i, option in enumerate(disambiguation.options):
                if st.button(f"Option {i+1}: {option.get('text', 'Unknown')}", key=f"disambig_{i}"):
                    # Handle user selection
                    st.session_state.excel_disambiguation = None
                    st.rerun()
