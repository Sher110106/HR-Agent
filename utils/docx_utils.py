"""
DOCX utility functions for converting text to Microsoft Word documents with
improved formatting and readability.
"""

import io
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.shared import OxmlElement, qn
from typing import Optional, List, Dict, Any


def _apply_document_defaults(doc: Document, title: Optional[str] = None, author: Optional[str] = None) -> None:
    """Apply sensible defaults: margins, base font, and core properties."""
    # Margins
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Base font: Calibri 11 (fallback to default if unavailable)
    try:
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Calibri'
        font.size = Pt(11)
    except Exception:
        # If Normal style is not accessible, ignore silently
        pass

    # Core properties
    if title:
        doc.core_properties.title = title
    if author:
        doc.core_properties.author = author


def _add_inline_formatted_runs(paragraph, text: str) -> None:
    """Add runs supporting simple markdown-like inline formatting: **bold**, *italic*, `code`.

    This is not a full markdown parser; it handles non-nested, common cases cleanly.
    """
    import re

    # Tokenize by code spans first to avoid styling conflicts
    parts = re.split(r"(`[^`]+`)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith('`') and part.endswith('`') and len(part) >= 2:
            code_text = part[1:-1]
            run = paragraph.add_run(code_text)
            run.font.name = 'Courier New'
            run.font.size = Pt(10)
            # Subtle visual distinction
            run.italic = False
            run.bold = False
            continue

        # Now handle bold and italic within non-code text
        idx = 0
        pattern = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*)")
        for m in pattern.finditer(part):
            if m.start() > idx:
                paragraph.add_run(part[idx:m.start()])
            token = m.group(0)
            if token.startswith('**') and token.endswith('**'):
                run = paragraph.add_run(token[2:-2])
                run.bold = True
            elif token.startswith('*') and token.endswith('*'):
                run = paragraph.add_run(token[1:-1])
                run.italic = True
            idx = m.end()
        if idx < len(part):
            paragraph.add_run(part[idx:])


def _add_markdown_like_text(doc: Document, text: str) -> None:
    """Convert simple markdown-like text to a formatted DOCX document structure.

    Supports:
    - #, ##, ### headings
    - Bullet lists (-, *, •) and numbered lists (1., 1)
    - Inline styles: **bold**, *italic*, `code`
    - Code blocks delimited by ``` fences
    - Paragraph justification for longer text
    """
    import re

    lines = text.splitlines()
    in_code_block = False

    for raw_line in lines:
        line = raw_line.rstrip('\n')

        # Code block fence
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            if in_code_block:
                # Start an explicit code paragraph for separation
                p = doc.add_paragraph()
                run = p.add_run()
                run.add_break()
            else:
                # End of code block separation
                p = doc.add_paragraph()
            continue

        if in_code_block:
            # Monospace runs for each code line
            p = doc.add_paragraph()
            run = p.add_run(line if line else " ")
            run.font.name = 'Courier New'
            run.font.size = Pt(10)
            continue

        if not line.strip():
            doc.add_paragraph("")
            continue

        # Headings
        m_h = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m_h:
            hashes, h_text = m_h.groups()
            level = min(len(hashes), 3)  # use up to Heading 3 for consistency
            doc.add_heading(h_text.strip(), level=level)
            continue

        # Numbered list
        if re.match(r"^\s*\d+[\.)]\s+", line):
            text_only = re.sub(r"^\s*\d+[\.)]\s+", "", line)
            p = doc.add_paragraph(style='List Number')
            _add_inline_formatted_runs(p, text_only)
            continue

        # Bulleted list
        if re.match(r"^\s*[-*•]\s+", line):
            text_only = re.sub(r"^\s*[-*•]\s+", "", line)
            p = doc.add_paragraph(style='List Bullet')
            _add_inline_formatted_runs(p, text_only)
            continue

        # Normal paragraph with inline formatting
        p = doc.add_paragraph()
        _add_inline_formatted_runs(p, line)
        if len(line) > 200:
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

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
    # Create a new document and apply defaults
    doc = Document()
    _apply_document_defaults(doc, title=title, author=author)

    # Title and author heading
    if title:
        title_paragraph = doc.add_heading(title, level=0)
        title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if author:
        author_paragraph = doc.add_paragraph(f"Author: {author}")
        author_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph("")

    # Body content with markdown-like formatting
    _add_markdown_like_text(doc, text)
    
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
    _apply_document_defaults(doc, title=title)

    # Add title if provided
    if title:
        doc.add_heading(title, level=0)
        doc.add_paragraph("")

    # Create table with a clean style
    table = doc.add_table(rows=1, cols=len(df.columns))
    # Prefer a built-in light style if available; fall back to Table Grid
    try:
        table.style = 'Light List Accent 1'
    except Exception:
        table.style = 'Table Grid'

    table.autofit = True

    # Header styling
    hdr_cells = table.rows[0].cells
    for i, col_name in enumerate(df.columns):
        p = hdr_cells[i].paragraphs[0]
        run = p.add_run(str(col_name))
        run.bold = True
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add data rows (limit very large frames for performance)
    max_rows = 1000
    for _, row in df.head(max_rows).iterrows():
        row_cells = table.add_row().cells
        for i, value in enumerate(row):
            row_cells[i].text = str(value)

    if len(df) > max_rows:
        doc.add_paragraph(f"Note: Showing first {max_rows} rows of {len(df)} total rows.")
    
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
    _apply_document_defaults(doc, title=title)

    # Add title
    if title:
        doc.add_heading(title, level=0)
    else:
        doc.add_heading("Data Analysis Report", level=0)

    # Analysis section
    doc.add_heading("Analysis", level=1)
    _add_markdown_like_text(doc, text)

    # Add data table if provided
    if data_df is not None and hasattr(data_df, 'empty') and not data_df.empty:
        doc.add_paragraph("")
        doc.add_heading("Data Summary", level=1)

        # Create table with styling
        table = doc.add_table(rows=1, cols=len(data_df.columns))
        try:
            table.style = 'Light List Accent 1'
        except Exception:
            table.style = 'Table Grid'
        table.autofit = True

        # Header
        hdr_cells = table.rows[0].cells
        for i, col_name in enumerate(data_df.columns):
            p = hdr_cells[i].paragraphs[0]
            run = p.add_run(str(col_name))
            run.bold = True
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Rows (limit to 200 for a concise report)
        max_rows = 200
        for _, row in data_df.head(max_rows).iterrows():
            row_cells = table.add_row().cells
            for i, value in enumerate(row):
                row_cells[i].text = str(value)

        if len(data_df) > max_rows:
            doc.add_paragraph(f"Note: Showing first {max_rows} rows of {len(data_df)} total rows.")

    # Save to bytes buffer
    docx_buffer = io.BytesIO()
    doc.save(docx_buffer)
    docx_buffer.seek(0)

    return docx_buffer.getvalue()