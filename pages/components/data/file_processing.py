"""File processing components for data analysis."""

import io
import logging
import pandas as pd
import chardet
import streamlit as st
from agents import DataInsightAgent
from app_core.api import get_available_models

logger = logging.getLogger(__name__)

def get_llm_call_with_selected_model():
    """Get make_llm_call function with the selected model from session state."""
    from app_core.api import make_llm_call
    
    def llm_call_wrapper(messages, **kwargs):
        # Get selected model from session state, fallback to default
        selected_model = getattr(st.session_state, 'selected_model', 'gpt-4.1')
        return make_llm_call(messages, model=selected_model, **kwargs)
    
    return llm_call_wrapper

def process_csv_file(file):
    """Process uploaded CSV file with encoding detection and DataFrame loading."""
    logger.info(f"📁 Processing CSV file: {file.name}")
    
    with st.spinner("🔄 Processing file..."):
        # Try to detect encoding automatically, then fallback to common encodings
        df_loaded = False
        detected_encoding = None
        
        try:
            # First, try to detect encoding using chardet
            file.seek(0)
            raw_data = file.read(10000)  # Read first 10KB for detection
            file.seek(0)  # Reset file pointer
            
            if raw_data:
                detection_result = chardet.detect(raw_data)
                detected_encoding = detection_result.get('encoding')
                confidence = detection_result.get('confidence', 0)
                
                if detected_encoding and confidence > 0.7:
                    logger.info(f"🔍 Detected encoding: {detected_encoding} (confidence: {confidence:.2f})")
                else:
                    logger.warning(f"⚠️ Low confidence encoding detection: {detected_encoding} (confidence: {confidence:.2f})")
        except Exception as e:
            logger.warning(f"⚠️ Encoding detection failed: {e}")
        
        # List of encodings to try (put detected encoding first if available)
        encodings_to_try = []
        if detected_encoding:
            encodings_to_try.append(detected_encoding)
        
        # Add common fallback encodings
        common_encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        for enc in common_encodings:
            if enc not in encodings_to_try:
                encodings_to_try.append(enc)
        
        # Try each encoding
        for encoding in encodings_to_try:
            try:
                file.seek(0)  # Reset file pointer
                df = pd.read_csv(file, encoding=encoding)
                df_loaded = True
                logger.info(f"📊 Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                logger.warning(f"⚠️ Failed to load with {encoding} encoding, trying next...")
                continue
            except Exception as e:
                logger.error(f"❌ Error loading with {encoding}: {e}")
                continue
        
        if not df_loaded:
            st.error("""
            ❌ **Unable to read the CSV file due to encoding issues.**
            
            **Possible solutions:**
            - Save your CSV file with UTF-8 encoding
            - If using Excel, use "Save As" → "CSV UTF-8 (Comma delimited)"
            - Try opening the file in a text editor and saving with UTF-8 encoding
            - Check if the file is actually a CSV format
            """)
            logger.error(f"❌ Failed to load file {file.name} with any supported encoding")
            return None
        
        return df

def generate_data_insights(df):
    """Generate insights for the DataFrame."""
    logger.info("🧠 Generating dataset insights...")
    
    with st.spinner("🧠 Generating dataset insights..."):
        llm_call_func = get_llm_call_with_selected_model()
        insights = DataInsightAgent(df, llm_call_func)
    
    return insights

def display_file_info(file, df):
    """Display file information and data preview."""
    st.markdown("### 📊 Data Preview")
    st.dataframe(df.head(), width='stretch')
    
    # Show dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("File Size", f"{file.size / 1024:.1f} KB")

def display_data_insights(insights):
    """Display the generated insights."""
    st.markdown("### 🔍 Dataset Insights")
    st.markdown(insights)
