"""
DOCX utility functions for converting text to Microsoft Word documents.
"""

import io
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.shared import OxmlElement, qn
from typing import Optional, List, Dict, Any


def text_to_docx(text: str, title: Optional[str] = None, author: Optional[str] = None) -> bytes:
    """
    Convert text content to a DOCX document.
    
    Args:
        text: The text content to convert
        title: Optional document title
        author: Optional document author
        
    Returns:
        DOCX document as bytes
    """
    # Create a new document
    doc = Document()
    
    # Add title if provided
    if title:
        title_paragraph = doc.add_heading(title, level=0)
        title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add author if provided
    if author:
        author_paragraph = doc.add_paragraph(f"Author: {author}")
        author_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph()  # Add spacing
    
    # Split text into paragraphs and add to document
    paragraphs = text.split('\n\n')
    
    for paragraph_text in paragraphs:
        if paragraph_text.strip():
            # Clean up the paragraph text
            clean_text = paragraph_text.strip()
            
            # Check if this looks like a heading (starts with # or is short and all caps)
            if clean_text.startswith('#') or (len(clean_text) < 100 and clean_text.isupper()):
                # Remove # symbols and create heading
                heading_text = clean_text.lstrip('#').strip()
                if heading_text:
                    doc.add_heading(heading_text, level=1)
            else:
                # Regular paragraph
                p = doc.add_paragraph(clean_text)
                
                # Add some basic formatting for better readability
                if len(clean_text) > 200:  # Long paragraphs get justified alignment
                    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    # Save to bytes buffer
    docx_buffer = io.BytesIO()
    doc.save(docx_buffer)
    docx_buffer.seek(0)
    
    return docx_buffer.getvalue()


def dataframe_to_docx_table(df, title: Optional[str] = None) -> bytes:
    """
    Convert a pandas DataFrame to a DOCX document with a formatted table.
    
    Args:
        df: Pandas DataFrame to convert
        title: Optional document title
        
    Returns:
        DOCX document as bytes
    """
    doc = Document()
    
    # Add title if provided
    if title:
        doc.add_heading(title, level=0)
    
    # Create table
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = 'Table Grid'
    
    # Add header row
    hdr_cells = table.rows[0].cells
    for i, col_name in enumerate(df.columns):
        hdr_cells[i].text = str(col_name)
    
    # Add data rows
    for _, row in df.iterrows():
        row_cells = table.add_row().cells
        for i, value in enumerate(row):
            row_cells[i].text = str(value)
    
    # Save to bytes buffer
    docx_buffer = io.BytesIO()
    doc.save(docx_buffer)
    docx_buffer.seek(0)
    
    return docx_buffer.getvalue()


def analysis_to_docx(text: str, data_df=None, title: Optional[str] = None) -> bytes:
    """
    Create a comprehensive DOCX document with analysis text and optional data table.
    
    Args:
        text: Analysis text content
        data_df: Optional pandas DataFrame to include as table
        title: Optional document title
        
    Returns:
        DOCX document as bytes
    """
    doc = Document()
    
    # Add title
    if title:
        doc.add_heading(title, level=0)
    else:
        doc.add_heading("Data Analysis Report", level=0)
    
    # Add analysis text
    doc.add_heading("Analysis", level=1)
    
    # Split text into paragraphs
    paragraphs = text.split('\n\n')
    for paragraph_text in paragraphs:
        if paragraph_text.strip():
            clean_text = paragraph_text.strip()
            
            # Check if this looks like a heading
            if clean_text.startswith('#') or (len(clean_text) < 100 and clean_text.isupper()):
                heading_text = clean_text.lstrip('#').strip()
                if heading_text:
                    doc.add_heading(heading_text, level=2)
            else:
                p = doc.add_paragraph(clean_text)
                if len(clean_text) > 200:
                    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    # Add data table if provided
    if data_df is not None and not data_df.empty:
        doc.add_heading("Data Summary", level=1)
        
        # Create table
        table = doc.add_table(rows=1, cols=len(data_df.columns))
        table.style = 'Table Grid'
        
        # Add header row
        hdr_cells = table.rows[0].cells
        for i, col_name in enumerate(data_df.columns):
            hdr_cells[i].text = str(col_name)
        
        # Add data rows (limit to first 50 rows to keep document manageable)
        for _, row in data_df.head(50).iterrows():
            row_cells = table.add_row().cells
            for i, value in enumerate(row):
                row_cells[i].text = str(value)
        
        if len(data_df) > 50:
            doc.add_paragraph(f"Note: Showing first 50 rows of {len(data_df)} total rows.")
    
    # Save to bytes buffer
    docx_buffer = io.BytesIO()
    doc.save(docx_buffer)
    docx_buffer.seek(0)
    
    return docx_buffer.getvalue() 