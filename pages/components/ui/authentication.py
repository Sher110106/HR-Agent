"""Authentication UI components for HR Agent pages."""

import streamlit as st
import logging

logger = logging.getLogger(__name__)

def render_auth_page(page_title="HR Agent", page_subtitle="Business Analysis Platform"):
    """Render the authentication page with sophisticated minimal styling."""
    
    # Sophisticated minimal CSS for authentication page
    st.markdown("""
    <style>
    .auth-container {
        max-width: 420px;
        margin: 6rem auto;
        padding: 0;
        background: transparent;
        border: none;
        box-shadow: none;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: visible;
    }
    
    .auth-header {
        text-align: center;
        margin-bottom: 4rem;
    }
    
    .auth-title {
        font-size: 2.2rem;
        font-weight: 200;
        margin-bottom: 0.8rem;
        color: #FFFFFF;
        letter-spacing: -1px;
        opacity: 0;
        animation: authTitleFadeIn 1s ease-out 0.3s forwards;
    }
    
    @keyframes authTitleFadeIn {
        from { 
            opacity: 0; 
            transform: translateY(30px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    
    .auth-subtitle {
        font-size: 1rem;
        color: #CCCCCC;
        margin-bottom: 0;
        font-weight: 300;
        opacity: 0;
        animation: authSubtitleFadeIn 1s ease-out 0.6s forwards;
    }
    
    @keyframes authSubtitleFadeIn {
        from { 
            opacity: 0; 
            transform: translateY(20px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    
    .auth-form {
        background: transparent;
        padding: 0;
        position: relative;
    }
    
    .auth-input {
        background: transparent;
        border: none;
        border-bottom: 1px solid #555555;
        border-radius: 0;
        padding: 1rem 0;
        margin-bottom: 2rem;
        font-size: 1rem;
        font-weight: 300;
        color: #FFFFFF;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        position: relative;
    }
    
    .auth-input:focus {
        outline: none;
        border-bottom-color: #FFFFFF;
        transform: translateY(-1px);
    }
    
    .auth-input::placeholder {
        color: #888888;
        font-weight: 300;
    }
    
    .auth-button {
        background: #000000;
        color: #FFFFFF;
        border: none;
        border-radius: 0;
        padding: 1rem 0;
        font-size: 1rem;
        font-weight: 300;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .auth-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.6s ease;
    }
    
    .auth-button:hover::before {
        left: 100%;
    }
    
    .auth-button:hover {
        background: #333333;
        transform: translateY(-1px);
    }
    
    /* Remove Streamlit's default form styling */
    div[data-testid="stForm"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }
    
    div[data-testid="stForm"] h3 {
        color: #FFFFFF !important;
    }
    
    /* Remove white background from Streamlit elements */
    .stTextInput > div > div {
        background: transparent !important;
    }
    
    .stTextInput input {
        background: transparent !important;
        color: #FFFFFF !important;
        border: none !important;
        border-bottom: 1px solid #555555 !important;
        border-radius: 0 !important;
    }
    
    .stTextInput input:focus {
        border-bottom-color: #FFFFFF !important;
    }
    
    .stTextInput input::placeholder {
        color: #888888 !important;
    }
    
    .stTextInput label {
        color: #CCCCCC !important;
    }
    
    /* Style the main block container */
    .block-container {
        background: transparent !important;
    }
    
    /* Remove any remaining white backgrounds */
    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }
    
    .auth-info {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem 0;
        background: transparent;
        border-left: none;
        font-size: 0.85rem;
        color: #AAAAAA;
        font-weight: 300;
        position: relative;
        overflow: visible;
    }
    
    /* Responsive Design for Auth */
    @media (max-width: 768px) {
        .auth-container {
            max-width: 90%;
            padding: 0;
            margin: 4rem auto;
        }
        .auth-title {
            font-size: 1.8rem;
        }
        .auth-subtitle {
            font-size: 0.9rem;
        }
        .auth-header {
            margin-bottom: 3rem;
        }
    }
    
    @media (max-width: 480px) {
        .auth-container {
            max-width: 95%;
            padding: 0;
            margin: 2rem auto;
        }
        .auth-title {
            font-size: 1.5rem;
        }
        .auth-subtitle {
            font-size: 0.85rem;
        }
        .auth-header {
            margin-bottom: 2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main container
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    
    # Header section
    st.markdown(f"""
    <div class="auth-header">
        <div class="auth-title">{page_title}</div>
        <div class="auth-subtitle">{page_subtitle}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Login form
    st.markdown('<div class="auth-form">', unsafe_allow_html=True)
    
    with st.form("login_form"):
        st.markdown("### Please enter your credentials")
        
        username = st.text_input(
            "Username", 
            placeholder="Enter your username",
            key="auth_username",
            help="Enter your assigned username"
        )
        
        password = st.text_input(
            "Password", 
            type="password", 
            placeholder="Enter your password",
            key="auth_password",
            help="Enter your assigned password"
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
                submit_button = st.form_submit_button(
                    "Login to Platform", 
                    width='stretch',
                    type="primary"
                )
        
        if submit_button:
            # Check credentials
            if username == "Plaksha-HR" and password == "AgentHR1":
                st.session_state.authenticated = True
                with st.spinner("Authenticating..."):
                    import time
                    time.sleep(1)  # Brief loading for better UX
                st.success("Login successful! Redirecting to dashboard...")
                logger.info(f"Successful login for user: {username}")
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")
                logger.warning(f"Failed login attempt for user: {username}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Info section
    st.markdown("""
    <div class="auth-info">
        <strong>Need Access?</strong> Please contact your administrator for credentials.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add some spacing
    st.markdown("<br><br>", unsafe_allow_html=True)

def check_authentication():
    """Check if user is authenticated, initialize session state if needed."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    return st.session_state.authenticated
