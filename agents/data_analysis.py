import logging
import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

from utils.retry_utils import perform_with_retries

logger = logging.getLogger(__name__)

def DataFrameSummaryTool(df: pd.DataFrame) -> str:
    """Generate a summary prompt string for the LLM based on the DataFrame."""
    logger.info(f"📊 DataFrameSummaryTool: Summarizing DataFrame with {len(df)} rows, {len(df.columns)} columns")
    
    missing_values = df.isnull().sum().to_dict()
    data_types = df.dtypes.to_dict()
    
    logger.info(f"📋 Data types: {data_types}")
    logger.info(f"❓ Missing values: {missing_values}")
    
    prompt = f"""
        Given a dataset with {len(df)} rows and {len(df.columns)} columns:
        Columns: {', '.join(df.columns)}
        Data types: {data_types}
        Missing values: {missing_values}

        Provide:
        1. A brief description of what this dataset contains
        2. 3-4 possible data analysis questions that could be explored
        Keep it concise and focused."""
    return prompt

def DataInsightAgent(df: pd.DataFrame, make_llm_call_func) -> str:
    """Uses the LLM to generate a brief summary and possible questions for the uploaded dataset."""
    logger.info(f"💡 DataInsightAgent: Generating insights for uploaded dataset")
    
    prompt = DataFrameSummaryTool(df)
    try:
        logger.info("📤 Sending dataset insight request to LLM...")
        response = make_llm_call_func([
            {"role": "system", "content": """detailed thinking off. You are an automated data exploration assistant. Your task is to analyze the metadata of a newly uploaded dataset and provide a concise, structured, and actionable "first look" summary to orient the user.

Using the provided dataset metadata (row/column counts, column names, data types, missing values), generate the following in a clear, structured format:

1.  **Dataset Overview**:
    *   A one-sentence summary of what the dataset likely contains based on its column names and structure. (e.g., "This appears to be a customer sales dataset tracking transactions, products, and demographic information.").

2.  **Key Characteristics & Data Quality Snapshot**:
    *   Provide a bulleted list of key facts: Total Rows, Total Columns.
    *   Highlight any immediate data quality concerns, such as columns with a high percentage of missing values. (e.g., "Data Quality Note: The 'Region' column is missing 45% of its values, which may impact regional analysis.").

3.  **Suggested Analysis Questions (Starters)**:
    *   Provide 3-4 specific, relevant data analysis questions that are tailored to the dataset's structure and likely business context. These should inspire the user.
    *   Example Questions:
        *   "What is the overall sales trend over time?"
        *   "Which product category generates the most revenue?"
        *   "Is there a correlation between customer age and purchase amount?"

4.  **Recommended First Steps**:
    *   Suggest 1-2 immediate actions the user could take, such as analyzing a key column or handling missing data. (e.g., "Recommendation: Start by analyzing the 'Sales' column to understand its distribution, or ask to visualize the 'Transaction_Date' to see trends.").

Keep the entire response concise, business-focused, and free of technical jargon."""},
            {"role": "user", "content": prompt}
        ])
        insights = response.choices[0].message.content
        logger.info(f"📥 Generated insights length: {len(insights)} characters")
        logger.debug(f"Insights: {insights[:200]}...")
        return insights
    except Exception as exc:
        error_msg = f"Error generating dataset insights: {exc}"
        logger.error(f"❌ DataInsightAgent failed: {error_msg}")
        return error_msg

def ColumnAnalysisAgent(df: pd.DataFrame, column_name: str, make_llm_call_func) -> str:
    """Generate AI-powered analysis and description for a specific column."""
    logger.info(f"🔍 ColumnAnalysisAgent: Analyzing column '{column_name}'")
    
    col_data = df[column_name]
    dtype = str(col_data.dtype)
    
    # Gather comprehensive column statistics
    stats = {
        'total_rows': len(col_data),
        'non_null_count': col_data.count(),
        'null_count': col_data.isnull().sum(),
        'null_percentage': (col_data.isnull().sum() / len(col_data)) * 100,
        'unique_count': col_data.nunique(),
        'data_type': dtype
    }
    
    # Get sample values (handle different data types)
    if col_data.dtype == 'object':
        # For text/categorical data
        value_counts = col_data.value_counts().head(10)
        sample_values = value_counts.index.tolist()
        value_distribution = dict(value_counts)
    else:
        # For numeric data
        sample_values = col_data.dropna().head(10).tolist()
        if col_data.dtype in ['int64', 'float64']:
            stats.update({
                'min_value': col_data.min(),
                'max_value': col_data.max(),
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std_dev': col_data.std()
            })
        value_distribution = {}
    
    # Check for potential date/time patterns
    is_likely_date = False
    if col_data.dtype == 'object':
        sample_str_values = col_data.dropna().head(5).astype(str).tolist()
        date_patterns = ['/', '-', ':', 'am', 'pm', 'time', 'date', '20', '19']
        for val in sample_str_values:
            if any(pattern in val.lower() for pattern in date_patterns):
                is_likely_date = True
                break
    
    # Build comprehensive prompt for AI analysis
    prompt = f"""
    Analyze this column from a dataset and provide a comprehensive description:
    
    Column Name: {column_name}
    Data Type: {dtype}
    Total Rows: {stats['total_rows']}
    Non-null Values: {stats['non_null_count']} ({100 - stats['null_percentage']:.1f}%)
    Missing Values: {stats['null_count']} ({stats['null_percentage']:.1f}%)
    Unique Values: {stats['unique_count']}
    
    Sample Values: {sample_values[:10]}
    
    {f"Statistical Summary: Min={stats.get('min_value', 'N/A')}, Max={stats.get('max_value', 'N/A')}, Mean={stats.get('mean', 'N/A'):.2f}, Median={stats.get('median', 'N/A')}" if 'mean' in stats else ""}
    
    {f"Value Distribution (top categories): {value_distribution}" if value_distribution else ""}
    
    {"IMPORTANT: This appears to be a date/time column based on sample values." if is_likely_date else ""}
    
    Please provide a detailed analysis including:
    1. What this column likely represents (business context)
    2. Data quality assessment (completeness, consistency)
    3. Potential use cases for analysis (what insights could be derived)
    4. Any data preprocessing recommendations
    5. Relationships this column might have with other typical business metrics
    
    Keep the response comprehensive but concise (3-4 paragraphs).
    """
    
    try:
        logger.info("📤 Sending column analysis request to LLM...")
        response = make_llm_call_func([
            {"role": "system", "content": """You are a senior data analyst and data steward, with deep expertise in business intelligence, data quality assessment, and data modeling. Your task is to perform a comprehensive, actionable analysis of a single data column and output it in a structured format suitable for being stored in a memory system.

For the given column, use the provided metadata (name, data type, sample values, missing values) and conversation context to generate the following analysis:

1.  **Business Context & Definition**:
    *   Describe what this column most likely represents in a business context. Be specific. (e.g., "The 'Employee_Satisfaction_Score' column likely represents a rating from 1 to 5 provided by employees during their annual performance review.").

2.  **Data Quality Assessment**:
    *   **Completeness**: State the percentage of missing values.
    *   **Consistency & Validity**: Assess the format and uniformity of the data. Are there mixed types? Unexpected values? (e.g., "Values are consistently integers from 1-5, with no invalid entries detected.").
    *   **Potential Issues**: Note any potential problems for analysis. (e.g., "A high concentration of scores at '3' may indicate central tendency bias.").

3.  **Analytical Use Cases**:
    *   Suggest 2-3 specific analytical applications for this column. (e.g., "1. Trend Analysis: Track satisfaction scores over time to measure the impact of HR initiatives. 2. Segmentation: Compare scores across different departments or job levels.").

4.  **Preprocessing Recommendations**:
    *   Provide specific, actionable data cleaning or transformation steps. (e.g., "For analysis, this categorical column should be one-hot encoded. Missing values could be imputed with the median score of '3' if necessary.").

5.  **Potential Relationships & Insights**:
    *   Hypothesize potential relationships with other columns or key business metrics. (e.g., "This column could be negatively correlated with 'Employee_Turnover_Rate' and positively correlated with 'Productivity_Score'.").

Output the analysis in a clear, structured format (e.g., using Markdown headings for each section) to ensure it is easily parsable for storage in the ColumnMemoryAgent."""},
            {"role": "user", "content": prompt}
        ])
        
        analysis = response.choices[0].message.content
        logger.info(f"📥 Generated column analysis: {len(analysis)} characters")
        return analysis
        
    except Exception as exc:
        error_msg = f"Error analyzing column '{column_name}': {exc}"
        logger.error(f"❌ Column analysis failed: {error_msg}")
        return error_msg

def _analyze_single_column(df: pd.DataFrame, column: str, make_llm_call_func) -> Tuple[str, str]:
    """Helper that actually calls the LLM for column analysis."""
    description = ColumnAnalysisAgent(df, column, make_llm_call_func)
    return column, description

def AnalyzeColumnBatch(df: pd.DataFrame, column: str, make_llm_call_func) -> Tuple[str, str]:
    """Single column analysis with shared retry/back-off handling."""

    try:
        return perform_with_retries(
            _analyze_single_column,
            df,
            column,
            make_llm_call_func,
            max_retries=2,
            base_delay=1.0,
            retry_exceptions=(Exception,),  # broaden; ColumnAnalysisAgent already filters
        )
    except Exception as e:
        logger.error("❌ Failed to analyze column %s: %s", column, e)
        return column, f"Error analyzing column: {e}"

def AnalyzeAllColumnsAgent(df: pd.DataFrame, memory_agent, make_llm_call_func) -> str:
    """Analyze all columns in the dataset in parallel and store descriptions in memory."""
    start_time = time.time()
    
    logger.info(f"📊 AnalyzeAllColumnsAgent: Starting PARALLEL analysis of {len(df.columns)} columns")
    
    columns = df.columns.tolist()
    total_columns = len(columns)
    
    # Create progress tracking
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    completed_count = 0
    analysis_results = {}
    
    # Use ThreadPoolExecutor for parallel API calls
    # Intelligent worker count: balance speed vs rate limits
    cpu_cores = os.cpu_count() or 4
    if total_columns <= 5:
        max_workers = total_columns  # Small datasets: full parallelism
    elif total_columns <= 20:
        max_workers = min(6, total_columns)  # Medium datasets: moderate parallelism
    else:
        # Large datasets: bound by CPU cores to avoid oversubscription
        max_workers = min(cpu_cores, 8)
    
    logger.info(f"🚀 Using {max_workers} parallel workers for {total_columns} columns")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all column analysis tasks
        future_to_column = {
            executor.submit(AnalyzeColumnBatch, df, column, make_llm_call_func): column 
            for column in columns
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_column):
            column = future_to_column[future]
            completed_count += 1
            
            try:
                column_name, description = future.result()
                analysis_results[column_name] = description
                
                # Store in memory
                memory_agent.store_column_description(column_name, description)
                
                # Update progress
                progress = completed_count / total_columns
                progress_placeholder.progress(
                    progress, 
                    text=f"Analyzed {completed_count}/{total_columns} columns"
                )
                
                # Show current status
                status_placeholder.info(f"✅ Completed: **{column_name}** ({completed_count}/{total_columns})")
                
                logger.info(f"✅ Completed analysis for column: {column_name} ({completed_count}/{total_columns})")
                
            except Exception as e:
                logger.error(f"❌ Error processing future for column {column}: {e}")
                analysis_results[column] = f"Error: {str(e)}"
    
    # Clear progress indicators
    progress_placeholder.empty()
    status_placeholder.empty()
    
    # Calculate performance metrics
    end_time = time.time()
    total_time = end_time - start_time
    successful_analyses = sum(1 for desc in analysis_results.values() if not desc.startswith("Error"))
    failed_analyses = total_columns - successful_analyses
    
    # Estimate sequential time (rough calculation)
    estimated_sequential_time = total_time * max_workers
    time_saved = estimated_sequential_time - total_time
    speedup_factor = estimated_sequential_time / total_time if total_time > 0 else 1
    
    # Create clean, concise summary
    successful_columns = [col for col in columns if col in analysis_results and not analysis_results[col].startswith("Error")]
    failed_columns = [col for col in columns if col in analysis_results and analysis_results[col].startswith("Error")]
    
    summary = f"""🧠 **Column Analysis Complete!**

✅ **Successfully analyzed {successful_analyses}/{total_columns} columns** in {total_time:.1f}s

{f"⚠️ **Failed columns:** {', '.join([f'`{col}`' for col in failed_columns])}" if failed_columns else ""}

🚀 **Enhanced AI mode enabled!** The chat can now provide deeper insights using detailed column understanding."""
    
    logger.info(f"✅ PARALLEL column analysis complete: {successful_analyses} successful, {failed_analyses} failed in {total_time:.1f}s ({speedup_factor:.1f}x speedup)")
    return summary

def smart_date_parser(df, column_name):
    """
    Smart date parsing function that handles various date formats robustly.
    This function is made available in the execution environment.
    """
    try:
        # First try pandas' automatic inference
        parsed = pd.to_datetime(df[column_name], errors='coerce', infer_datetime_format=True)
        
        # If that fails for too many values, try some common formats
        if parsed.isna().sum() > len(df) * 0.1:  # If more than 10% failed
            common_formats = [
                '%m/%d/%y %I:%M %p',    # 5/13/25 12:00 AM
                '%m/%d/%Y %I:%M %p',    # 5/13/2025 12:00 AM
                '%m-%d-%y %H:%M',       # 5-13-25 12:00
                '%m-%d-%Y %H:%M',       # 5-13-2025 12:00
                '%Y-%m-%d %H:%M:%S',    # 2025-05-13 12:00:00
                '%m/%d/%y',             # 5/13/25
                '%m/%d/%Y',             # 5/13/2025
            ]
            
            for fmt in common_formats:
                try:
                    test_parsed = pd.to_datetime(df[column_name], format=fmt, errors='coerce')
                    if test_parsed.isna().sum() < parsed.isna().sum():
                        parsed = test_parsed
                        break
                except:
                    continue
        
        return parsed
    except:
        # Fallback to original column if all parsing fails
        return df[column_name]

def extract_first_code_block(text: str) -> str:
    """Extracts the first Python code block from a markdown-formatted string."""
    logger.debug("✂️ Extracting first code block from response")
    start = text.find("```python")
    if start == -1:
        logger.warning("⚠️ No ```python code block found")
        return ""
    start += len("```python")
    end = text.find("```", start)
    if end == -1:
        logger.warning("⚠️ Unclosed code block found")
        return ""
    code = text[start:end].strip()
    logger.debug(f"✂️ Extracted code: {len(code)} characters")
    return code 