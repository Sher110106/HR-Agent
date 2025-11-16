"""Sidebar components for HR Agent pages."""

import streamlit as st
import logging
from app_core.api import get_available_models
from agents.memory import SystemPromptMemoryAgent
from utils.system_prompts import get_prompt_manager

logger = logging.getLogger(__name__)

def add_sidebar_css():
    """Add sophisticated minimal CSS for sidebar components."""
    st.sidebar.markdown("""
    <style>
    .sidebar-section {
        background: transparent;
        padding: 0;
        margin: 1rem 0;
        border: none;
        box-shadow: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: visible;
    }
    
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 300;
        color: #000000;
        margin-bottom: 1rem;
        letter-spacing: -0.3px;
    }
    
    .sidebar-subtitle {
        font-size: 0.9rem;
        font-weight: 300;
        color: #666666;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-button {
        background: #000000;
        color: #FFFFFF;
        border: none;
        border-radius: 0;
        padding: 0.6rem 1rem;
        font-size: 0.9rem;
        font-weight: 300;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .sidebar-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.6s ease;
    }
    
    .sidebar-button:hover::before {
        left: 100%;
    }
    
    .sidebar-button:hover {
        background: #333333;
        transform: translateY(-1px);
    }
    
    .sidebar-info {
        background: transparent;
        padding: 1rem;
        border-left: none;
        font-size: 0.85rem;
        color: #666666;
        font-weight: 300;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)

def render_system_prompt_sidebar():
    """Render sophisticated system prompt controls in the sidebar."""
    add_sidebar_css()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### AI Behavior")
    
    # Model Selection
    st.sidebar.markdown("**AI Model**")
    available_models = get_available_models()
    
    if not available_models:
        st.sidebar.error("No AI models available. Please check your API configuration.")
        return
    
    # Initialize model selection in session state
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = list(available_models.keys())[0]  # Default to first available
    
    # Model selection dropdown
    model_options = list(available_models.keys())
    model_labels = [available_models[model] for model in model_options]
    
    current_model_index = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
    
    selected_model_key = st.sidebar.selectbox(
        "Choose AI Model:",
        options=model_options,
        format_func=lambda x: available_models[x],
        index=current_model_index,
        help="Select which AI model to use for analysis"
    )
    
    # Update session state if selection changed
    if selected_model_key != st.session_state.selected_model:
        st.session_state.selected_model = selected_model_key
        st.sidebar.success(f"Switched to {available_models[selected_model_key]}")
        st.rerun()
    
    # Show current model info
    current_model_name = available_models.get(st.session_state.selected_model, "Unknown")
    st.sidebar.info(f"Active Model: **{current_model_name}**")
    
    st.sidebar.markdown("---")
    
    # System Prompt Selection
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
    if st.sidebar.button("Apply Style", width='stretch'):
        if selected_prompt == "Default":
            system_prompt_agent.clear_active_prompt()
            st.sidebar.success("Using default AI behavior")
        else:
            success = system_prompt_agent.set_active_prompt(selected_prompt)
            if success:
                st.sidebar.success(f"Applied: {selected_prompt}")
            else:
                st.sidebar.error(f"Failed to apply: {selected_prompt}")
        st.rerun()
    
    # Show active prompt info
    if active_prompt:
        with st.sidebar.expander("Active Style Details", expanded=False):
            st.markdown(f"**{active_prompt.name}**")
            st.markdown(f"*{active_prompt.description}*")
            st.markdown(f"**Category:** {active_prompt.category}")
            if active_prompt.tags:
                st.markdown(f"**Tags:** {', '.join(active_prompt.tags)}")
    else:
        st.sidebar.info("Using default AI behavior")
    
    # Quick link to full prompt manager
    if st.sidebar.button("Manage All Prompts", width='stretch'):
        st.sidebar.info("Use the main navigation to access the System Prompt Manager page")

def render_plot_settings():
    """Render sophisticated plot settings in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Plot Settings")
    plot_engine = st.sidebar.selectbox(
        "Choose Plotting Engine:",
        options=["Plotly (Interactive)", "Matplotlib (Static PNG)"],
        index=0,
        help="Plotly: Interactive plots, HTML downloads\nMatplotlib: Static plots, PNG downloads"
    )
    
    # Store the selection in session state
    if "plot_engine" not in st.session_state:
        st.session_state.plot_engine = "plotly"
    
    if "Plotly" in plot_engine:
        st.session_state.plot_engine = "plotly"
    else:
        st.session_state.plot_engine = "matplotlib"

def render_excel_plot_settings():
    """Render sophisticated Excel-specific plot settings in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Excel Plot Enhancements")
    
    # Theme selection
    excel_plot_theme = st.sidebar.selectbox(
        "Excel Plot Theme:",
        options=['professional', 'modern', 'minimal', 'elegant'],
        index=0,
        help="Choose the visual style for your Excel plots"
    )
    
    # Insight detection toggle
    excel_show_insights = st.sidebar.checkbox(
        "Auto-detect Excel Insights",
        value=True,
        help="Automatically detect and highlight key insights in Excel plots"
    )
    
    # Store settings in session state
    st.session_state.excel_plot_theme = excel_plot_theme
    st.session_state.excel_show_insights = excel_show_insights

def render_logout_button():
    """Render sophisticated logout button in the sidebar."""
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", width='stretch'):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        logger.info("User logged out")
        st.rerun()
