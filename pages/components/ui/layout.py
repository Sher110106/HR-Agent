"""Layout and styling components for HR Agent pages."""

import streamlit as st

def add_global_css():
    """Add sophisticated minimal CSS with smooth animations and modern design."""
    st.markdown("""
    <style>
    /* Modern Minimal Design System - Pure White, Black, Gray */
    
    /* Global Reset and Base Styles */
    * {
        box-sizing: border-box;
    }
    
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #FAFAFA;
        margin: 0;
        padding: 0;
    }
    
    /* Remove Streamlit's default white boxes */
    div[data-testid="stForm"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }
    
    div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stForm"]) {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    .stTextInput > div > div {
        background: transparent !important;
    }
    
    .stTextInput label {
        color: #CCCCCC !important;
    }
    
    .stTextInput input {
        color: #FFFFFF !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    
    p, span, div {
        color: #CCCCCC;
    }
    
    .block-container {
        background: transparent !important;
        padding-top: 2rem !important;
    }
    
    /* Remove white backgrounds from all Streamlit containers */
    div[class*="stMarkdown"] {
        background: transparent !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }
    
    /* Sidebar text colors */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6 {
        color: #FFFFFF !important;
    }
    
    section[data-testid="stSidebar"] label {
        color: #CCCCCC !important;
    }
    
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #CCCCCC !important;
    }
    
    /* Remove file uploader white box */
    div[data-testid="stFileUploader"] {
        background: transparent !important;
        border: 1px dashed #555555 !important;
        padding: 1rem !important;
    }
    
    div[data-testid="stFileUploader"] > div {
        background: transparent !important;
    }
    
    div[data-testid="stFileUploader"] label {
        color: #CCCCCC !important;
    }
    
    /* Elegant Header */
    .main-header {
        background: transparent;
        padding: 2rem 0 1rem;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: visible;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 200;
        margin-bottom: 0.8rem;
        color: #FFFFFF;
        letter-spacing: -1px;
        opacity: 0;
        animation: titleFadeIn 1s ease-out 0.3s forwards;
    }
    
    @keyframes titleFadeIn {
        from { 
            opacity: 0; 
            transform: translateY(30px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    
    .main-subtitle {
        font-size: 1rem;
        color: #CCCCCC;
        margin-bottom: 0;
        font-weight: 300;
        opacity: 0;
        animation: subtitleFadeIn 1s ease-out 0.6s forwards;
    }
    
    @keyframes subtitleFadeIn {
        from { 
            opacity: 0; 
            transform: translateY(20px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    
    /* Modern Upload Section */
    .upload-section {
        background: transparent;
        padding: 0;
        border: none;
        border-radius: 0;
        margin-bottom: 2rem;
        box-shadow: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: visible;
    }
    
    /* Elegant Chat Section */
    .chat-section {
        background: transparent;
        padding: 0;
        border: none;
        border-radius: 0;
        min-height: 600px;
        box-shadow: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    /* Sophisticated Loading States */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 5rem 2rem;
        background: transparent;
        border: none;
        box-shadow: none;
        position: relative;
        overflow: visible;
    }
    
    .loading-spinner {
        width: 32px;
        height: 32px;
        border: 2px solid #F0F0F0;
        border-top: 2px solid #000000;
        border-radius: 50%;
        animation: elegantSpin 1.2s cubic-bezier(0.4, 0, 0.2, 1) infinite;
        margin-bottom: 1.5rem;
        position: relative;
        z-index: 1;
    }
    
    @keyframes elegantSpin {
        0% { 
            transform: rotate(0deg); 
            border-top-color: #000000;
        }
        50% { 
            transform: rotate(180deg); 
            border-top-color: #666666;
        }
        100% { 
            transform: rotate(360deg); 
            border-top-color: #000000;
        }
    }
    
    .loading-text {
        color: #333333;
        font-size: 0.95rem;
        font-weight: 300;
        text-align: center;
        position: relative;
        z-index: 1;
        opacity: 0;
        animation: textFadeIn 0.8s ease-out 0.5s forwards;
    }
    
    @keyframes textFadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Modern Plot Containers */
    .plot-container {
        background: transparent;
        padding: 0;
        border: none;
        border-radius: 0;
        margin: 1.5rem 0;
        box-shadow: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: visible;
    }
    
    /* Elegant Message Containers */
    .message-container {
        background: transparent;
        padding: 0;
        border: none;
        border-left: none;
        margin: 1.5rem 0;
        box-shadow: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    /* Modern Sheet Info Containers */
    .sheet-info-container {
        background: transparent;
        padding: 0;
        border: none;
        border-radius: 0;
        margin: 1.5rem 0;
        box-shadow: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    /* Sophisticated Animations */
    .fade-in {
        opacity: 0;
        animation: elegantFadeIn 0.6s cubic-bezier(0.4, 0, 0.2, 1) forwards;
    }
    
    @keyframes elegantFadeIn {
        from { 
            opacity: 0; 
            transform: translateY(20px) scale(0.98); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0) scale(1); 
        }
    }
    
    .slide-up {
        opacity: 0;
        animation: elegantSlideUp 0.7s cubic-bezier(0.4, 0, 0.2, 1) forwards;
    }
    
    @keyframes elegantSlideUp {
        from { 
            opacity: 0; 
            transform: translateY(30px) scale(0.95); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0) scale(1); 
        }
    }
    
    /* Smooth Hover Effects */
    .hover-lift {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .hover-lift:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            padding: 3rem 1rem 2rem;
        }
        .main-title {
            font-size: 2.2rem;
        }
        .main-subtitle {
            font-size: 0.9rem;
        }
        .loading-container {
            padding: 4rem 1rem;
        }
    }
    
    @media (max-width: 480px) {
        .main-title {
            font-size: 1.8rem;
        }
        .main-subtitle {
            font-size: 0.85rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def render_main_header(title, subtitle):
    """Render the main application header."""
    st.markdown(f"""
    <div class="main-header">
        <div class="main-title">{title}</div>
        <div class="main-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def render_upload_section(title="Data Upload"):
    """Render the upload section container."""
    st.markdown(f"### {title}")
    return st.container()

def render_chat_section(title="Chat with your data"):
    """Render the chat section container."""
    st.markdown(f"### {title}")
    return st.container()

def render_loading_state(text="Analyzing your data"):
    """Render a loading state with spinner."""
    st.markdown(f"""
    <div class="loading-container slide-up">
        <div class="loading-spinner"></div>
        <div class="loading-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)

def render_plot_container():
    """Render a plot container."""
    return st.container()

def render_message_container():
    """Render a message container."""
    return st.container()

def render_sheet_info_container():
    """Render a sheet info container."""
    return st.container()

def close_container():
    """Close the current container - no-op now that we removed divs."""
    pass
