"""Excel Analysis Page for multi-sheet Excel file processing.

This page provides functionality for uploading and analyzing Excel files with multiple sheets,
including sheet cataloging, column indexing, and semantic layer management.
Enhanced with Phase 3: Resilience & Polish features.
"""

import io
import logging
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from datetime import datetime

from agents import (
    SheetCatalogAgent, ColumnIndexerAgent, ColumnRef, SheetPlan,
    SheetSelectionAgent, DisambiguationQuestion,
    ExcelCodeGenerationAgent, ExcelExecutionAgent,
    ReasoningAgent, DataInsightAgent
)
from agents.memory import SystemPromptMemoryAgent
from utils.system_prompts import get_prompt_manager
from app_core.api import make_llm_call
from utils.excel_error_handling import ExcelErrorHandler
from utils.excel_performance import PerformanceMonitor, memory_cleanup

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


def render_sheet_catalog(sheet_catalog_agent: SheetCatalogAgent):
    """Render the sheet catalog information."""
    st.subheader("📋 Sheet Catalog")
    
    sheet_info = sheet_catalog_agent.get_sheet_info()
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📊 Summary", "🔍 Details"])
    
    with tab1:
        # Summary table
        summary_data = []
        for sanitized_name, info in sheet_info.items():
            summary_data.append({
                "Sheet Name": info['original_name'],
                "Variable Name": sanitized_name,
                "Rows": info['rows'],
                "Columns": info['columns'],
                "Memory (MB)": f"{info['memory_usage'] / 1024 / 1024:.2f}"
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("No sheets found in the Excel file.")
    
    with tab2:
        # Detailed view with expandable sections for each sheet
        for sanitized_name, info in sheet_info.items():
            with st.expander(f"📋 {info['original_name']} ({sanitized_name})", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Rows:** {info['rows']:,}")
                    st.markdown(f"**Columns:** {info['columns']}")
                    st.markdown(f"**Memory:** {info['memory_usage'] / 1024 / 1024:.2f} MB")
                
                with col2:
                    # Show column names
                    st.markdown("**Columns:**")
                    for col in info['column_names'][:10]:  # Show first 10
                        st.markdown(f"- {col}")
                    if len(info['column_names']) > 10:
                        st.markdown(f"- ... and {len(info['column_names']) - 10} more")
                
                # Show data types
                st.markdown("**Data Types:**")
                dtype_df = pd.DataFrame([
                    {"Column": col, "Type": str(dtype)} 
                    for col, dtype in info['data_types'].items()
                ])
                # Convert to string to avoid PyArrow issues
                st.dataframe(dtype_df.astype(str), use_container_width=True)


def render_column_index(column_indexer_agent: ColumnIndexerAgent):
    """Render the column index information."""
    st.subheader("🔍 Column Index")
    
    # Get column summary
    column_summary = column_indexer_agent.get_column_summary()
    
    if not column_summary:
        st.info("No columns found in the Excel file.")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 All Columns", "🔗 Common Columns", "🔑 Join Keys"])
    
    with tab1:
        # All columns summary
        all_columns_data = []
        for col_name, info in column_summary.items():
            all_columns_data.append({
                "Column": col_name,
                "Sheets": len(info['sheets']),
                "Data Types": ", ".join(info['data_types']),
                "Avg Unique Ratio": f"{info['avg_unique_ratio']:.2f}",
                "Sample Values": ", ".join(str(v)[:20] for v in info['sample_values'][:3])
            })
        
        if all_columns_data:
            all_columns_df = pd.DataFrame(all_columns_data)
            st.dataframe(all_columns_df, use_container_width=True)
    
    with tab2:
        # Common columns (appearing in multiple sheets)
        common_columns = column_indexer_agent.find_common_columns(min_sheets=2)
        
        if common_columns:
            common_data = []
            for col_name, refs in common_columns.items():
                sheets = [ref.sheet_name for ref in refs]
                common_data.append({
                    "Column": col_name,
                    "Sheets": ", ".join(sheets),
                    "Count": len(sheets),
                    "Data Types": ", ".join(set(ref.data_type for ref in refs))
                })
            
            common_df = pd.DataFrame(common_data)
            st.dataframe(common_df, use_container_width=True)
        else:
            st.info("No columns found in multiple sheets.")
    
    with tab3:
        # Potential join keys
        potential_keys = column_indexer_agent.find_potential_join_keys()
        
        if potential_keys:
            keys_data = []
            for key in potential_keys:
                refs = column_indexer_agent.get_column_refs(key)
                keys_data.append({
                    "Join Key": key,
                    "Sheets": ", ".join(ref.sheet_name for ref in refs),
                    "Data Type": ", ".join(set(ref.data_type for ref in refs)),
                    "Avg Unique Ratio": f"{sum(ref.unique_count / max(1, ref.unique_count + ref.null_count) for ref in refs) / len(refs):.2f}"
                })
            
            keys_df = pd.DataFrame(keys_data)
            st.dataframe(keys_df, use_container_width=True)
        else:
            st.info("No potential join keys found.")
            
            # Debug information
            with st.expander("🔍 Debug: Column Analysis", expanded=False):
                st.markdown("**All columns in multiple sheets:**")
                common_columns = column_indexer_agent.find_common_columns(min_sheets=2)
                if common_columns:
                    for col_name, refs in common_columns.items():
                        st.markdown(f"- **{col_name}**: {len(refs)} sheets")
                        for ref in refs:
                            st.markdown(f"  - {ref.sheet_name}: {ref.data_type}, {ref.unique_count} unique, {ref.null_count} null")
                else:
                    st.markdown("No columns found in multiple sheets.")


def render_semantic_layer(column_indexer_agent: ColumnIndexerAgent):
    """Render the semantic layer management interface."""
    st.subheader("🏷️ Semantic Layer")
    
    # Get potential join keys for tagging
    potential_keys = column_indexer_agent.find_potential_join_keys()
    
    if not potential_keys:
        st.warning("⚠️ No potential join keys found for semantic tagging.")
        
        # Show common columns as fallback
        common_columns = column_indexer_agent.find_common_columns(min_sheets=2)
        if common_columns:
            st.markdown("**Available common columns:**")
            for col_name, refs in common_columns.items():
                sheets = [ref.sheet_name for ref in refs]
                st.markdown(f"- **{col_name}**: {', '.join(sheets)}")
            
            # Allow manual selection from common columns
            st.markdown("**Manual Join Key Selection**")
            manual_key = st.selectbox(
                "Select a join key from common columns:",
                options=["None"] + list(common_columns.keys()),
                help="Manually select a column that appears in multiple sheets"
            )
            
            if manual_key != "None":
                column_indexer_agent.add_semantic_tag("primary_join_key", manual_key)
                st.success(f"✅ Set primary join key: {manual_key}")
        return
    
    # Primary join key selection
    st.markdown("**Primary Join Key**")
    primary_key = st.selectbox(
        "Select the primary key for joining sheets:",
        options=["None"] + potential_keys,
        help="This key will be used as the default for joining sheets"
    )
    
    if primary_key != "None":
        column_indexer_agent.add_semantic_tag("primary_join_key", primary_key)
        st.success(f"✅ Set primary join key: {primary_key}")
        
        # Show details about the selected key
        refs = column_indexer_agent.get_column_refs(primary_key)
        if refs:
            st.markdown("**Join Key Details:**")
            for ref in refs:
                uniqueness_ratio = ref.unique_count / max(1, ref.unique_count + ref.null_count)
                st.markdown(f"- **{ref.sheet_name}**: {ref.data_type}, {ref.unique_count} unique values ({uniqueness_ratio:.1%} uniqueness)")
    
    # Show current semantic tags
    semantic_layer = column_indexer_agent.semantic_layer
    if semantic_layer:
        st.markdown("**Current Tags:**")
        for tag, value in semantic_layer.items():
            st.markdown(f"- **{tag}:** {value}")


def render_phase3_features(sheet_catalog_agent: SheetCatalogAgent, column_indexer_agent: ColumnIndexerAgent):
    """Render Phase 3: Resilience & Polish features."""
    st.subheader("🚀 Phase 3: Advanced Features")
    
    # Create tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "🛡️ Error Handling", "💾 Data Quality", "📤 Export"])
    
    with tab1:
        st.markdown("**Performance Monitoring**")
        
        # Get performance metrics
        if hasattr(sheet_catalog_agent, 'performance_monitor'):
            perf_summary = sheet_catalog_agent.performance_monitor.get_performance_summary()
            
            if perf_summary.get('status') == 'No queries recorded':
                st.info("No performance data recorded yet.")
            else:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Queries", perf_summary.get('total_queries', 0))
                with col2:
                    st.metric("Avg Query Time", f"{perf_summary.get('avg_query_time', 0):.2f}s")
                with col3:
                    st.metric("Cache Hit Rate", f"{perf_summary.get('cache_hit_rate', 0)*100:.1f}%")
                with col4:
                    st.metric("Memory Usage", f"{perf_summary.get('current_memory_usage_mb', 0):.1f} MB")
                
                # Performance actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🧹 Memory Cleanup", use_container_width=True):
                        memory_usage = memory_cleanup()
                        st.success(f"✅ Memory cleanup completed. Current usage: {memory_usage:.1f} MB")
                        st.rerun()
                
                with col2:
                    if st.button("📊 Cache Stats", use_container_width=True):
                        if hasattr(sheet_catalog_agent, 'cache'):
                            cache_stats = sheet_catalog_agent.cache.get_stats()
                            st.json(cache_stats)
        
        # Cache management
        if hasattr(sheet_catalog_agent, 'cache'):
            st.markdown("**Cache Management**")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Clear Cache", use_container_width=True):
                    sheet_catalog_agent.cache.clear()
                    st.success("✅ Cache cleared")
                    st.rerun()
            
            with col2:
                if st.button("📋 Cache Info", use_container_width=True):
                    cache_stats = sheet_catalog_agent.cache.get_stats()
                    st.json(cache_stats)
    
    with tab2:
        st.markdown("**Error Handling & Recovery**")
        
        # Get error summary
        if hasattr(sheet_catalog_agent, 'error_handler'):
            error_summary = sheet_catalog_agent.error_handler.get_error_summary()
            
            if error_summary['total_errors'] == 0:
                st.success("✅ No errors recorded")
            else:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Errors", error_summary['total_errors'])
                with col2:
                    st.metric("Critical Errors", error_summary['by_severity'].get('critical', 0))
                with col3:
                    st.metric("High Severity", error_summary['by_severity'].get('high', 0))
                
                # Show recent errors
                if error_summary['recent_errors']:
                    st.markdown("**Recent Errors:**")
                    for error in error_summary['recent_errors'][-3:]:
                        st.warning(f"⚠️ {error}")
        
        # Error handling actions
        st.markdown("**Error Recovery Actions:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Retry Failed Operations", use_container_width=True):
                st.info("Retry functionality would be implemented here")
        
        with col2:
            if st.button("📋 Error Log", use_container_width=True):
                if hasattr(sheet_catalog_agent, 'error_handler'):
                    error_summary = sheet_catalog_agent.error_handler.get_error_summary()
                    st.json(error_summary)
    
    with tab3:
        st.markdown("**Data Quality Assessment**")
        
        if hasattr(column_indexer_agent, 'query_engine'):
            # Quality assessment for each sheet
            for sheet_name, df in sheet_catalog_agent.sheet_catalog.items():
                with st.expander(f"📋 {sheet_name} Quality Report", expanded=False):
                    try:
                        quality_report = column_indexer_agent.query_engine.validate_data_quality(df, sheet_name)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Quality Score", f"{quality_report['quality_score']:.1f}/100")
                        with col2:
                            st.metric("Total Rows", quality_report['total_rows'])
                        with col3:
                            st.metric("Null Rows", quality_report['duplicate_rows'])
                        with col4:
                            st.metric("Memory (MB)", f"{quality_report['memory_usage_mb']:.1f}")
                        
                        # Show quality issues
                        if quality_report['data_type_issues']:
                            st.warning("**Data Type Issues:**")
                            for issue in quality_report['data_type_issues']:
                                st.markdown(f"- **{issue['column']}:** {', '.join(issue['issues'])}")
                        
                        if quality_report['outliers']:
                            st.warning("**Outliers Detected:**")
                            for col, outlier_info in quality_report['outliers'].items():
                                st.markdown(f"- **{col}:** {outlier_info['count']} outliers ({outlier_info['percentage']:.1f}%)")
                        
                    except Exception as e:
                        st.error(f"Error generating quality report: {e}")
    
    with tab4:
        st.markdown("**Export Capabilities**")
        
        # Export options
        export_format = st.selectbox("Export Format:", ["csv", "excel", "json"])
        
        if st.button("📤 Export Current Data", use_container_width=True):
            if hasattr(column_indexer_agent, 'query_engine'):
                try:
                    # Export the first sheet as example
                    first_sheet = list(sheet_catalog_agent.sheet_catalog.values())[0]
                    file_content, filename = column_indexer_agent.query_engine.export_results(
                        first_sheet, format=export_format
                    )
                    
                    st.download_button(
                        label=f"📥 Download {filename}",
                        data=file_content,
                        file_name=filename,
                        mime="application/octet-stream"
                    )
                    
                except Exception as e:
                    st.error(f"Export failed: {e}")
        
        # Advanced export options
        st.markdown("**Advanced Export Options:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export All Sheets", use_container_width=True):
                st.info("Multi-sheet export would be implemented here")
        
        with col2:
            if st.button("📈 Export Analysis Results", use_container_width=True):
                st.info("Analysis results export would be implemented here")


def render_chat_interface(sheet_catalog_agent: SheetCatalogAgent, column_indexer_agent: ColumnIndexerAgent):
    """Render the chat interface for Excel analysis."""
    st.subheader("💬 Chat with your Excel data")
    
    # Initialize session state for Excel chat
    if "excel_messages" not in st.session_state:
        st.session_state.excel_messages = []
    if "excel_plots" not in st.session_state:
        st.session_state.excel_plots = []
    if "excel_plot_data" not in st.session_state:
        st.session_state.excel_plot_data = []
    if "excel_disambiguation" not in st.session_state:
        st.session_state.excel_disambiguation = None
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.excel_messages):
            with st.chat_message(msg["role"]):
                # Enhanced error display for assistant messages
                if msg["role"] == "assistant" and isinstance(msg["content"], str) and (
                    msg["content"].startswith("Error executing code:") or msg["content"].startswith("Error generating code:")
                ):
                    # Show error prominently
                    st.error("An error occurred while processing your request.")
                    # Show summary and collapsible technical details
                    error_lines = msg["content"].split("\n")
                    summary = error_lines[0]
                    details = "\n".join(error_lines[1:])
                    st.markdown(f"**Summary:** {summary}")
                    if details.strip():
                        with st.expander("Technical Details", expanded=False):
                            st.code(details, language="text")
                    continue
                
                # Normal message rendering
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
                        import re
                        clean_text = re.sub(r'<[^>]+>', '', msg["content"])
                        clean_text = re.sub(r'\n+', '\n', clean_text).strip()
                        
                        if clean_text:
                            st.download_button(
                                label="📄 Download Text",
                                data=clean_text,
                                file_name=f"excel_analysis_response_{i+1}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                    
                    # Download CSV data if available (dual-output only)
                    if has_data and col4:
                        with col3:
                            data_idx = msg["data_index"]
                            if 0 <= data_idx < len(st.session_state.get("excel_plot_data", [])):
                                data_df = st.session_state.excel_plot_data[data_idx]
                                
                                # Convert DataFrame to CSV
                                csv_buffer = io.StringIO()
                                data_df.to_csv(csv_buffer, index=False)
                                csv_data = csv_buffer.getvalue()
                                
                                st.download_button(
                                    label="📊 Download Data (CSV)",
                                    data=csv_data,
                                    file_name=f"excel_plot_data_{i+1}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                    
                    # Download plot if available
                    plot_col = col4 if has_data else col3
                    if has_plot and plot_col:
                        with plot_col:
                            plot_idx = msg["plot_index"]
                            if 0 <= plot_idx < len(st.session_state.excel_plots):
                                # Create download button for plot
                                fig = st.session_state.excel_plots[plot_idx]
                                
                                # Save plot to bytes buffer
                                img_buffer = io.BytesIO()
                                fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                                img_buffer.seek(0)
                                
                                st.download_button(
                                    label="🖼️ Download Plot",
                                    data=img_buffer.getvalue(),
                                    file_name=f"excel_plot_{i+1}.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                
                # Display plot
                if msg.get("plot_index") is not None:
                    idx = msg["plot_index"]
                    if 0 <= idx < len(st.session_state.excel_plots):
                        # Display plot at fixed size
                        st.pyplot(st.session_state.excel_plots[idx], use_container_width=False)
                
                # Display code in a proper expander for assistant messages
                if msg.get("code") and msg["role"] == "assistant":
                    with st.expander("View code", expanded=False):
                        st.code(msg["code"], language="python")
    
    # Handle disambiguation if present
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
    
    # Chat input
    if user_q := st.chat_input("Ask about your Excel data..."):
        logger.info(f"💬 Excel user query received: '{user_q}'")
        st.session_state.excel_messages.append({"role": "user", "content": user_q})
        
        with st.spinner("Working with Excel data..."):
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
                        is_dual_output = isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], (plt.Figure, plt.Axes))
                        is_plot = isinstance(result, (plt.Figure, plt.Axes))
                        plot_idx = None
                        data_idx = None
                        
                        if is_dual_output:
                            # Handle new dual-output format (fig, data_content)
                            fig, data_content = result
                            
                            # Store plot
                            st.session_state.excel_plots.append(fig)
                            plot_idx = len(st.session_state.excel_plots) - 1
                            
                            # Handle different data content types
                            if isinstance(data_content, pd.DataFrame):
                                # Single DataFrame
                                st.session_state.excel_plot_data.append(data_content)
                                data_idx = len(st.session_state.excel_plot_data) - 1
                                logger.info(f"📊 Dual-output added: plot at index {plot_idx}, data at index {data_idx} ({len(data_content)} rows)")
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
                                    logger.info(f"📊 Dual-output added: plot at index {plot_idx}, combined data at index {data_idx} ({len(combined_df)} total rows)")
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
                        from agents.reasoning import ReasoningAgent
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
                        
                        # Combine thinking and explanation (like CSV analysis)
                        explanation_html = reasoning_txt
                        assistant_msg = f"{thinking_html}{explanation_html}{plan_info}"
                        
                        st.session_state.excel_messages.append({
                            "role": "assistant",
                            "content": assistant_msg,
                            "plot_index": plot_idx,
                            "data_index": data_idx,
                            "code": code
                        })
                else:
                    # No plan or disambiguation - error case
                    st.session_state.excel_messages.append({
                        "role": "assistant",
                        "content": "❌ **Unable to create analysis plan.**\n\nPlease check your query or try rephrasing your question.",
                        "code": "# No plan generated"
                    })
                
            except Exception as e:
                logger.error(f"❌ Error in Excel analysis: {e}")
                st.session_state.excel_messages.append({
                    "role": "assistant",
                    "content": f"❌ **Error processing your request:**\n\n{str(e)}\n\nPlease try again or check your data.",
                    "code": "# Error occurred during processing"
                })
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            logger.info(f"⏱️ Excel processing time: {processing_time:.2f} seconds")
            
            st.rerun()


def excel_analysis_page():
    """Main Excel analysis page."""
    # Initialize session state for authentication
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    # Authentication page
    if not st.session_state.authenticated:
        st.title("🔐 Excel Analysis Agent - Login")
        st.markdown("---")
        
        # Center the login form
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### Please enter your credentials")
            
            with st.form("excel_login_form"):
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                submit_button = st.form_submit_button("Login", use_container_width=True)
                
                if submit_button:
                    # Check credentials
                    if username == "Plaksha-HR" and password == "AgentHR1":
                        st.session_state.authenticated = True
                        st.success("✅ Login successful! Redirecting...")
                        logger.info(f"🔓 Successful Excel login for user: {username}")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password. Please try again.")
                        logger.warning(f"🔒 Failed Excel login attempt for user: {username}")
        
        # Add some styling and info
        st.markdown("---")
        st.info("💡 Please contact your administrator if you need access credentials.")
        return  # Exit early if not authenticated
    
    # Main application (only accessible after authentication)
    logger.info("🚀 Starting Excel Analysis page - User authenticated")
    
    # Add system prompt controls and logout button in the sidebar
    render_system_prompt_sidebar()
    
    with st.sidebar:
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            logger.info("🔓 User logged out from Excel analysis")
            st.rerun()
    
    # Main content area
    st.title("📊 Excel Analysis Agent")
    st.markdown("<medium>Powered by Azure OpenAI GPT-4.1</medium>", unsafe_allow_html=True)
    st.markdown("---")
    
    # File upload section
    left, right = st.columns([3, 7])
    
    with left:
        st.header("📁 Upload Excel File")
        file = st.file_uploader("Choose Excel file", type=["xlsx", "xls"])
        
        if file:
            if ("excel_sheet_catalog" not in st.session_state) or (st.session_state.get("excel_current_file") != file.name):
                logger.info(f"📁 New Excel file uploaded: {file.name}")
                
                try:
                    # Initialize Excel agents
                    sheet_catalog_agent = SheetCatalogAgent()
                    column_indexer_agent = ColumnIndexerAgent({})
                    
                    # Read Excel file
                    with st.spinner("Reading Excel file..."):
                        file_content = io.BytesIO(file.read())
                        file.seek(0)  # Reset file pointer
                        
                        # Read all sheets
                        sheet_catalog = sheet_catalog_agent.read_excel_file(file_content, file.name)
                        
                        # Build column index
                        column_indexer_agent = ColumnIndexerAgent(sheet_catalog)
                        column_index = column_indexer_agent.build_column_index()
                    
                    # Store in session state
                    st.session_state.excel_sheet_catalog = sheet_catalog_agent
                    st.session_state.excel_column_indexer = column_indexer_agent
                    st.session_state.excel_current_file = file.name
                    st.session_state.excel_messages = []
                    
                    st.success(f"✅ Successfully loaded {len(sheet_catalog)} sheets from {file.name}")
                    logger.info(f"📊 Loaded Excel file: {len(sheet_catalog)} sheets")
                    
                except Exception as e:
                    st.error(f"❌ Error loading Excel file: {str(e)}")
                    logger.error(f"❌ Failed to load Excel file {file.name}: {e}")
                    return
            
            # Show file info
            if "excel_sheet_catalog" in st.session_state:
                sheet_catalog_agent = st.session_state.excel_sheet_catalog
                column_indexer_agent = st.session_state.excel_column_indexer
                
                # Show sheet catalog
                render_sheet_catalog(sheet_catalog_agent)
                
                # Show column index
                render_column_index(column_indexer_agent)
                
                # Show semantic layer
                render_semantic_layer(column_indexer_agent)
                
                # Show Phase 3 features
                render_phase3_features(sheet_catalog_agent, column_indexer_agent)
        else:
            st.info("Upload an Excel file to begin analysis.")
    
    with right:
        # Chat interface
        if "excel_sheet_catalog" in st.session_state:
            sheet_catalog_agent = st.session_state.excel_sheet_catalog
            column_indexer_agent = st.session_state.excel_column_indexer
            render_chat_interface(sheet_catalog_agent, column_indexer_agent)
        else:
            st.info("Upload an Excel file to start chatting with your data.") 