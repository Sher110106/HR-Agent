"""Refactored Data Analysis Page using modular components."""

import logging
import streamlit as st

# Import modular components
from pages.components.ui.authentication import render_auth_page, check_authentication
from pages.components.ui.layout import add_global_css, render_main_header, render_upload_section, render_chat_section, close_container
from pages.components.ui.sidebar import render_system_prompt_sidebar, render_plot_settings, render_logout_button
from pages.components.data.file_processing import process_csv_file, generate_data_insights, display_file_info, display_data_insights
from pages.components.chat.message_display import display_chat_messages
from pages.components.chat.data_analysis_chat import initialize_chat_session_state, process_user_query

logger = logging.getLogger(__name__)

def data_analysis_page():
    """Main data analysis page using modular components."""
    # Check authentication
    if not check_authentication():
        render_auth_page("HR Agent", "Business Analysis Platform")
        return
    
    # Initialize session state
    initialize_chat_session_state()
    
    logger.info("🚀 Starting Streamlit app - User authenticated")

    # Add global CSS styling
    add_global_css()
    
    # Add system prompt controls and logout button in the sidebar
    render_system_prompt_sidebar()
    render_plot_settings()
    render_logout_button()

    # Main header
    render_main_header("Business Analysis HR Agent", "Powered by Azure OpenAI GPT-4.1")

    # Main layout
    left, right = st.columns([3, 7])

    with left:
        # Upload section
        upload_container = render_upload_section("Data Upload")
        with upload_container:
            file = st.file_uploader("Choose CSV", type=["csv"], help="Upload your CSV file for analysis")
        close_container()  # Close upload-section div
        
        if file:
            # Check if this is a new file
            if ("df" not in st.session_state) or (st.session_state.get("current_file") != file.name):
                logger.info(f"New file uploaded: {file.name}")
                
                # Process the CSV file
                df = process_csv_file(file)
                if df is not None:
                    st.session_state.df = df
                    st.session_state.current_file = file.name
                    st.session_state.messages = []
                    logger.info(f"Loaded DataFrame: {len(df)} rows, {len(df.columns)} columns")
                    
                    # Generate insights
                    insights = generate_data_insights(df)
                    st.session_state.insights = insights
            
            # Display file info and insights
            if "df" in st.session_state:
                display_file_info(file, st.session_state.df)
                display_data_insights(st.session_state.insights)

    with right:
        # Chat section
        chat_container = render_chat_section("Chat with your data")
        with chat_container:
            # Display chat messages
            display_chat_messages(st.session_state.messages)
            
            # Chat input
            if file and "df" in st.session_state:  # only allow chat after upload
                if user_q := st.chat_input("Ask about your data..."):
                    process_user_query(user_q, st.session_state.df)
        
        close_container()  # Close chat-section div
