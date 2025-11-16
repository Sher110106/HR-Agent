"""Refactored Excel Analysis Page using modular components."""

import logging
import streamlit as st

# Import modular components
from pages.components.ui.authentication import render_auth_page, check_authentication
from pages.components.ui.layout import add_global_css, render_main_header, render_upload_section, render_chat_section, close_container, render_sheet_info_container
from pages.components.ui.sidebar import render_system_prompt_sidebar, render_excel_plot_settings, render_logout_button
from pages.components.data.excel_processing import process_excel_file, analyze_excel_structure, display_excel_file_info, get_excel_agents_from_session
from pages.components.excel.excel_components import render_sheet_catalog, render_column_index, render_semantic_layer, render_phase3_features
from pages.components.chat.message_display import display_chat_messages
from pages.components.chat.excel_chat import initialize_excel_chat_session_state, process_excel_user_query, handle_disambiguation

logger = logging.getLogger(__name__)

def excel_analysis_page():
    """Main Excel analysis page using modular components."""
    # Check authentication
    if not check_authentication():
        render_auth_page("📊 Excel Agent", "Advanced Excel Analysis Platform")
        return
    
    # Initialize session state
    initialize_excel_chat_session_state()
    
    logger.info("🚀 Starting Excel Analysis page - User authenticated")
    
    # Add global CSS styling
    add_global_css()
    
    # Add system prompt controls and logout button in the sidebar
    render_system_prompt_sidebar()
    render_excel_plot_settings()
    render_logout_button()
    
    # Main header
    render_main_header("Excel Analysis Agent", "Powered by Azure OpenAI GPT-4.1")
    
    # File upload section
    left, right = st.columns([3, 7])
    
    with left:
        # Upload section
        upload_container = render_upload_section("Upload Excel File")
        with upload_container:
            file = st.file_uploader("Choose Excel file", type=["xlsx", "xls"], help="Upload your Excel file for analysis")
        close_container()  # Close upload-section div
        
        if file:
            # Check if this is a new file
            if ("excel_file_bytes" not in st.session_state) or (st.session_state.get("excel_current_file") != file.name):
                logger.info(f"New Excel file uploaded: {file.name}")
                
                # Process the Excel file
                file_bytes, filename = process_excel_file(file)
                st.rerun()
            
            # Show file info and analysis
            if "excel_file_bytes" in st.session_state and "excel_current_file" in st.session_state:
                # Analyze Excel structure
                sheet_catalog_agent, column_indexer_agent = analyze_excel_structure(
                    st.session_state.excel_file_bytes, 
                    st.session_state.excel_current_file
                )
                
                if sheet_catalog_agent and column_indexer_agent:
                    # Display file info
                    display_excel_file_info(file, sheet_catalog_agent)
                    
                    # Show sheet catalog
                    with render_sheet_info_container():
                        render_sheet_catalog(sheet_catalog_agent)
                    close_container()
                    
                    # Show column index
                    with render_sheet_info_container():
                        render_column_index(column_indexer_agent)
                    close_container()
                    
                    # Show semantic layer
                    with render_sheet_info_container():
                        render_semantic_layer(column_indexer_agent)
                    close_container()
                    
                    # Show Phase 3 features
                    with render_sheet_info_container():
                        render_phase3_features(sheet_catalog_agent, column_indexer_agent)
                    close_container()
    
    with right:
        # Chat interface
        if "excel_file_bytes" in st.session_state and "excel_current_file" in st.session_state:
            # Chat section
            chat_container = render_chat_section("Chat with your Excel data")
            with chat_container:
                # Get agents from session state
                sheet_catalog_agent, column_indexer_agent = get_excel_agents_from_session()
                
                if sheet_catalog_agent and column_indexer_agent:
                    # Display chat messages
                    display_chat_messages(st.session_state.excel_messages, "excel_plots", "excel_plot_data")
                    
                    # Handle disambiguation
                    handle_disambiguation()
                    
                    # Chat input
                    if user_q := st.chat_input("Ask about your Excel data..."):
                        process_excel_user_query(user_q, sheet_catalog_agent, column_indexer_agent)
            close_container()  # Close chat-section div
