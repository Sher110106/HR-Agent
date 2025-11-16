"""Excel file processing components."""

import io
import logging
import pandas as pd
import streamlit as st
from agents import SheetCatalogAgent, ColumnIndexerAgent

logger = logging.getLogger(__name__)

def process_excel_file(file):
    """Process uploaded Excel file and return agents."""
    logger.info(f"📁 Processing Excel file: {file.name}")
    
    with st.spinner("🔄 Processing Excel file..."):
        # Store file bytes and metadata
        file_bytes = file.read()
        filename = file.name
        
        # Store in session state
        st.session_state.excel_file_bytes = file_bytes
        st.session_state.excel_current_file = filename
        st.session_state.excel_messages = []
    
    st.success(f"✅ Successfully uploaded {filename}")
    return file_bytes, filename

def analyze_excel_structure(file_bytes, filename):
    """Analyze Excel file structure and return agents."""
    logger.info("📊 Analyzing Excel structure...")
    
    with st.spinner("📊 Analyzing Excel structure..."):
        # Create file content buffer
        file_content = io.BytesIO(file_bytes)
        file_content.seek(0)
        
        # Create agents
        sheet_catalog_agent = SheetCatalogAgent()
        try:
            sheet_catalog = sheet_catalog_agent.read_excel_file(file_content, filename)
        except Exception as e:
            st.error(f"❌ Error loading Excel file: {str(e)}")
            logger.error(f"❌ Failed to load Excel file {filename}: {e}")
            return None, None
        
        column_indexer_agent = ColumnIndexerAgent(sheet_catalog)
        column_indexer_agent.build_column_index()
    
    return sheet_catalog_agent, column_indexer_agent

def display_excel_file_info(file, sheet_catalog_agent):
    """Display Excel file information."""
    st.markdown("### 📊 File Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File Size", f"{file.size / 1024:.1f} KB")
    with col2:
        st.metric("Sheets", len(sheet_catalog_agent.sheet_catalog))
    with col3:
        total_rows = sum(len(df) for df in sheet_catalog_agent.sheet_catalog.values())
        st.metric("Total Rows", total_rows)

def get_excel_agents_from_session():
    """Get Excel agents from session state."""
    if "excel_file_bytes" not in st.session_state or "excel_current_file" not in st.session_state:
        return None, None
    
    file_content = io.BytesIO(st.session_state.excel_file_bytes)
    file_content.seek(0)
    filename = st.session_state.excel_current_file
    
    sheet_catalog_agent = SheetCatalogAgent()
    try:
        sheet_catalog = sheet_catalog_agent.read_excel_file(file_content, filename)
    except Exception as e:
        st.error(f"❌ Error loading Excel file: {str(e)}")
        logger.error(f"❌ Failed to load Excel file {filename}: {e}")
        return None, None
    
    column_indexer_agent = ColumnIndexerAgent(sheet_catalog)
    column_indexer_agent.build_column_index()
    
    return sheet_catalog_agent, column_indexer_agent
