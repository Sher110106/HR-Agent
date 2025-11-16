"""Message display components for chat interface."""

import io
import logging
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils.plot_helpers import safe_plotly_figure_for_streamlit, safe_plotly_to_html
from utils.docx_utils import text_to_docx

logger = logging.getLogger(__name__)

def display_chat_messages(messages, plots_key="plots", plot_data_key="plot_data"):
    """Display chat messages with enhanced error handling and download options."""
    for i, msg in enumerate(messages):
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
                # Suggest retry if likely transient
                if "network" in msg["content"].lower() or "timeout" in msg["content"].lower() or "LLM" in msg["content"]:
                    st.info("You may try again or check your network connection.")
                continue
            
            # Normal message rendering
            st.markdown(msg["content"], unsafe_allow_html=True)
            
            # For assistant messages, add download options
            if msg["role"] == "assistant":
                display_download_options(msg, i, plots_key, plot_data_key)
            
            # Display plot(s)
            display_plots(msg, i, plots_key)
            
            # Display code in a proper expander for assistant messages
            if msg.get("code") and msg["role"] == "assistant":
                with st.expander("View code", expanded=False):
                    st.code(msg["code"], language="python")

def display_download_options(msg, message_index, plots_key="plots", plot_data_key="plot_data"):
    """Display download options for assistant messages."""
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
        # Download text response based on on-screen content
        # Extract clean text from the response (remove HTML)
        import re
        source_text = msg.get("detailed_reasoning", msg["content"]) or ""
        clean_text = re.sub(r'<[^>]+>', '', source_text)
        clean_text = re.sub(r'\n+', '\n', clean_text).strip()
        
        if clean_text:
            # Convert to DOCX
            docx_data = text_to_docx(clean_text, title=f"Analysis Response {message_index+1}")
            st.download_button(
                label="📝 DOCX",
                data=docx_data,
                file_name=f"analysis_response_{message_index+1}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                width='stretch'
            )
    
    # Download CSV data if available (dual-output only)
    if has_data and col4:
        with col3:
            data_idx = msg["data_index"]
            if 0 <= data_idx < len(st.session_state.get(plot_data_key, [])):
                data_df = st.session_state[plot_data_key][data_idx]
                
                # Convert DataFrame to CSV
                csv_buffer = io.StringIO()
                data_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📊 CSV",
                    data=csv_data,
                    file_name=f"plot_data_{message_index+1}.csv",
                    mime="text/csv",
                    width='stretch'
                )
    
    # Download plot if available
    plot_col = col4 if has_data else col3
    if has_plot and plot_col:
        with plot_col:
            plot_idx = msg["plot_index"]
            if 0 <= plot_idx < len(st.session_state.get(plots_key, [])):
                # Create download button for plot
                fig = st.session_state[plots_key][plot_idx]
                
                # Save plot to bytes buffer
                img_buffer = io.BytesIO()
                if hasattr(fig, 'data') and hasattr(fig, 'layout'):
                    # Plotly figure - export as HTML (reliable)
                    html_content = safe_plotly_to_html(fig)
                    img_buffer.write(html_content.encode('utf-8'))
                else:
                    # Matplotlib figure - PNG export (reliable)
                    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                
                # Determine file format and MIME type
                if hasattr(fig, 'data') and hasattr(fig, 'layout'):
                    file_name = f"plot_{message_index+1}.html"
                    mime_type = "text/html"
                    label = "📄 Download Plot (HTML)"
                else:
                    file_name = f"plot_{message_index+1}.png"
                    mime_type = "image/png"
                    label = "🖼️ Download Plot (PNG)"
                
                st.download_button(
                    label=label,
                    data=img_buffer.getvalue(),
                    file_name=file_name,
                    mime=mime_type,
                    width='stretch'
                )

def display_plots(msg, message_index, plots_key="plots"):
    """Display plots for a message."""
    if msg.get("plot_index") is not None:
        # Single plot display
        idx = msg["plot_index"]
        if 0 <= idx < len(st.session_state.get(plots_key, [])):
            fig = st.session_state[plots_key][idx]
            # Check if it's a Plotly figure or matplotlib
            if hasattr(fig, 'data') and hasattr(fig, 'layout'):
                # Plotly figure - clean for safe Streamlit display with proper theming
                safe_fig = safe_plotly_figure_for_streamlit(fig, theme='auto', variant='professional')
                st.plotly_chart(safe_fig, width='stretch', key=f"plotly_chart_single_{message_index}_{idx}")
            else:
                # Matplotlib figure (legacy)
                st.pyplot(fig, width='content')
    
    elif msg.get("plot_indices") is not None:
        # Multi-plot display
        indices = msg["plot_indices"]
        st.markdown(f"**📊 {len(indices)} Visualizations:**")
        
        for j, idx in enumerate(indices):
            if 0 <= idx < len(st.session_state.get(plots_key, [])):
                fig = st.session_state[plots_key][idx]
                
                # Add chart number
                st.markdown(f"**Chart {j+1}:**")
                
                # Check if it's a Plotly figure or matplotlib
                if hasattr(fig, 'data') and hasattr(fig, 'layout'):
                    # Plotly figure - clean for safe Streamlit display with proper theming
                    safe_fig = safe_plotly_figure_for_streamlit(fig, theme='auto', variant='professional')
                    st.plotly_chart(safe_fig, width='stretch', key=f"plotly_chart_multi_{j}_{idx}")
                else:
                    # Matplotlib figure (legacy)
                    st.pyplot(fig, width='content')
                
                # Add separator between charts
                if j < len(indices) - 1:
                    st.markdown("---")
