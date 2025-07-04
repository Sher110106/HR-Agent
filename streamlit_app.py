import streamlit as st
import traceback
from utils.logging_config import setup_logging, get_logger
from utils.navigation import get_navigation_registry

# Set up centralized logging configuration
setup_logging()
logger = get_logger(__name__)

# Global flag to prevent duplicate health monitoring threads
_health_monitoring_started = False

def render_error_page(error_message: str, error_details: str = None):
    """Render a fallback error page when navigation fails."""
    st.error("🚨 Application Error")
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"**Error:** {error_message}")
        if error_details:
            with st.expander("🔍 Technical Details"):
                st.code(error_details, language="text")
    
    with col2:
        if st.button("🔄 Refresh App"):
            st.rerun()
    
    st.markdown("---")
    st.info("💡 **Troubleshooting Tips:**")
    st.markdown("""
    - Check the application logs for more details
    - Try refreshing the application
    - Verify all required dependencies are installed
    - Contact support if the issue persists
    """)

def safe_initialize_monitoring():
    """Safely initialize health monitoring with error handling."""
    global _health_monitoring_started
    
    try:
        if not _health_monitoring_started:
            from utils.health_monitor import start_health_monitoring
            start_health_monitoring()
            _health_monitoring_started = True
            logger.info("🚀 Enhanced Data Analysis Agent starting with monitoring systems")
            return True
    except Exception as e:
        logger.error(f"Failed to initialize health monitoring: {e}")
        st.warning("⚠️ Health monitoring could not be initialized. Some features may be limited.")
        return False
    return True

def safe_get_navigation():
    """Safely get navigation registry with error handling."""
    try:
        nav_registry = get_navigation_registry()
        page_titles = nav_registry.get_page_titles()
        
        if not page_titles:
            raise ValueError("No pages registered in navigation system")
        
        return nav_registry, page_titles
    except Exception as e:
        logger.error(f"Failed to get navigation registry: {e}")
        raise

def main():
    """Main application function with comprehensive error handling."""
    try:
        # Configure Streamlit page
        st.set_page_config(
            page_title="Data Analysis Agent",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Hide Streamlit's built-in multipage navigation (appears automatically when a "pages/" folder is present)
        st.markdown(
            """
            <style>
            /* Hide the entire auto-generated navigation container */
            div[data-testid="stSidebarNav"] {
                display: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Initialize monitoring systems with error handling
        safe_initialize_monitoring()

        # Get navigation with error handling
        try:
            nav_registry, page_titles = safe_get_navigation()
        except Exception as e:
            render_error_page(
                "Navigation system initialization failed",
                f"Unable to load page registry: {str(e)}\n\n{traceback.format_exc()}"
            )
            return

        # Add navigation sidebar
        st.sidebar.title("🎯 Navigation")
        
        # Create selectbox with error handling
        try:
            page_options = list(page_titles.values())
            if not page_options:
                raise ValueError("No pages available")
                
            selected_title = st.sidebar.selectbox("Choose a page:", page_options)
            
            # Add app status indicator
            with st.sidebar:
                st.markdown("---")
                if _health_monitoring_started:
                    st.success("✅ Monitoring Active")
                else:
                    st.warning("⚠️ Limited Mode")
        
        except Exception as e:
            logger.error(f"Navigation UI error: {e}")
            render_error_page(
                "Navigation interface error",
                f"Unable to create navigation UI: {str(e)}"
            )
            return
        
        # Find the page_id for the selected title
        selected_page_id = None
        for page_id, title in page_titles.items():
            if title == selected_title:
                selected_page_id = page_id
                break
        
        # Render the selected page with comprehensive error handling
        if selected_page_id:
            try:
                success = nav_registry.render_page(selected_page_id)
                if not success:
                    # Get page config for better error messages
                    page_config = nav_registry.get_page_config(selected_page_id)
                    page_name = page_config.title if page_config else selected_title
                    
                    render_error_page(
                        f"Failed to load page: {page_name}",
                        f"Page ID: {selected_page_id}\nCheck logs for import or execution errors."
                    )
            except Exception as e:
                logger.error(f"Unexpected error rendering page {selected_page_id}: {e}")
                render_error_page(
                    f"Unexpected error in page: {selected_title}",
                    f"Error: {str(e)}\n\n{traceback.format_exc()}"
                )
        else:
            render_error_page(
                "Page selection error",
                f"Unable to find page for title: {selected_title}"
            )

    except Exception as e:
        # Catch-all error handler for any unhandled exceptions
        logger.critical(f"Critical application error: {e}")
        
        # Fallback UI when everything else fails
        st.error("🚨 Critical Application Error")
        st.markdown("The application encountered a critical error and cannot continue.")
        
        with st.expander("🔍 Error Details"):
            st.code(f"Error: {str(e)}\n\n{traceback.format_exc()}", language="text")
        
        if st.button("🔄 Restart Application"):
            st.rerun()

if __name__ == "__main__":
    main() 