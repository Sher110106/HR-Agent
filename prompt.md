# Data Analysis Agent - System Prompts Documentation

This document provides a comprehensive overview of all system prompts used in the Business Analysis HR Agent application, their locations, and detailed explanations of their intended purposes.

## Overview

The data analysis agent uses multiple specialized AI agents, each with carefully crafted system prompts designed for specific tasks. The system leverages the NVIDIA Llama-3.1-Nemotron-Ultra-253B-v1 model for all interactions.

---

## 1. QueryUnderstandingTool System Prompt

**Location:** `streamlit_app.py` - Line 109-113  
**Function:** `QueryUnderstandingTool()`  
**Model Call:** Lines 114-125

### System Prompt:
```
detailed thinking off. You are an assistant that determines if a query is requesting a data visualization. Consider the conversation context to understand follow-up requests. Respond with only 'true' if the query is asking for a plot, chart, graph, or any visual representation of data. Otherwise, respond with 'false'.
```

### Purpose:
- **Intent Classification**: Determines whether a user query requires visualization or data analysis
- **Context Awareness**: Considers conversation history to understand follow-up requests 
- **Binary Decision**: Returns a simple true/false to route the query to appropriate code generation tool
- **Follow-up Support**: Handles cases where users refer to previous visualizations with phrases like "make it bigger" or "show that as a bar chart"

### Usage Flow:
1. User submits a query
2. This prompt analyzes the query + conversation context
3. Returns boolean to determine if `PlotCodeGeneratorTool` or `CodeWritingTool` should be used
4. Enables smart routing between visualization and analysis code generation

---

## 2. CodeGenerationAgent System Prompt

**Location:** `streamlit_app.py` - Lines 374-378  
**Function:** `CodeGenerationAgent()`  
**Model Call:** Lines 379-387

### System Prompt:
```
detailed thinking off. You are a senior data scientist with expertise in pandas, matplotlib, and seaborn for statistical analysis and visualization. Write clean, efficient, production-ready code. Focus on:

1. CORRECTNESS: Ensure proper variable scoping (pd, np, df, plt, sns are available)
2. ROBUSTNESS: Handle missing values and edge cases
3. CLARITY: Use descriptive variable names and clear logic
4. EFFICIENCY: Prefer vectorized operations over loops
5. BEST PRACTICES: Follow pandas conventions and leverage seaborn for enhanced visualizations
6. AESTHETICS: Use seaborn's statistical plotting capabilities for professional-looking charts

Output ONLY a properly-closed ```python code block. Use smart_date_parser() for date parsing. Assign final result to 'result' variable.
```

### Purpose:
- **Code Quality Standards**: Establishes expectations for production-ready, professional code
- **Technical Expertise**: Positions the AI as a senior data scientist with deep domain knowledge
- **Library Mastery**: Ensures proper use of pandas, matplotlib, and seaborn
- **Best Practices**: Enforces vectorized operations, proper error handling, and clean code patterns
- **Environment Awareness**: Reminds about available variables and functions in execution scope
- **Output Format**: Ensures consistent code block formatting for extraction

### Usage Flow:
1. User query is processed and routed to code generation
2. This prompt generates either visualization or analysis code
3. Code is extracted and executed in controlled environment
4. Results are passed to reasoning agent for explanation

---

## 3. ReasoningAgent System Prompt

**Location:** `streamlit_app.py` - Line 497  
**Function:** `ReasoningAgent()`  
**Model Call:** Lines 494-505

### System Prompt:
```
detailed thinking on. You are an insightful data analyst.
```

### Purpose:
- **Explanation Generation**: Provides human-readable interpretations of analysis results
- **Thinking Process**: Enables detailed reasoning with "thinking on" to show analytical process
- **User Communication**: Translates technical results into business insights
- **Context Understanding**: Interprets results in the context of the original user query

### Special Features:
- **Streaming Response**: Supports real-time thinking display to users
- **Thinking Extraction**: Separates `<think>...</think>` content from final explanation
- **Adaptive Reasoning**: Handles different result types (plots, dataframes, scalars, errors)

### Usage Flow:
1. Code execution produces a result (plot, data, or error)
2. `ReasoningCurator` builds appropriate prompt based on result type
3. This agent streams analytical reasoning back to user
4. Thinking process is displayed in expandable UI element

---

## 4. DataInsightAgent System Prompt

**Location:** `streamlit_app.py` - Line 580  
**Function:** `DataInsightAgent()`  
**Model Call:** Lines 575-585

### System Prompt:
```
detailed thinking off. You are a data analyst providing brief, focused insights.
```

### Purpose:
- **Dataset Overview**: Generates initial insights when CSV files are uploaded
- **Quick Analysis**: Provides immediate value without deep analysis
- **Exploration Guidance**: Suggests potential analysis questions for the dataset
- **User Onboarding**: Helps users understand their data structure and possibilities

### Generated Content:
1. Brief description of dataset contents
2. 3-4 possible data analysis questions
3. High-level data characteristics
4. Recommended exploration directions

### Usage Flow:
1. User uploads CSV file
2. `DataFrameSummaryTool` extracts dataset metadata
3. This agent generates user-friendly insights
4. Insights are displayed in sidebar to guide user queries

---

## 5. ColumnAnalysisAgent System Prompt

**Location:** `streamlit_app.py` - Line 677  
**Function:** `ColumnAnalysisAgent()`  
**Model Call:** Lines 673-685

### System Prompt:
```
You are a senior data analyst with expertise in business intelligence and data quality assessment. Provide insightful, actionable analysis of data columns.
```

### Purpose:
- **Deep Column Analysis**: Generates comprehensive descriptions for individual columns
- **Business Context**: Interprets columns from business intelligence perspective
- **Data Quality Assessment**: Evaluates completeness, consistency, and quality
- **Actionable Insights**: Provides recommendations for data preprocessing and analysis
- **Memory Enhancement**: Creates rich context stored in `ColumnMemoryAgent` for enhanced future analysis

### Analysis Components:
1. **Business Context**: What the column likely represents
2. **Data Quality**: Completeness and consistency assessment
3. **Use Cases**: Potential analytical applications
4. **Preprocessing Recommendations**: Data cleaning suggestions
5. **Relationship Insights**: How column might relate to other business metrics

### Usage Flow:
1. User clicks "Analyze Columns" button
2. Each column is analyzed in parallel using `AnalyzeAllColumnsAgent`
3. Rich descriptions are stored in memory system
4. Enhanced context enables more sophisticated analysis in subsequent queries

---

## Supporting Prompt Systems

### PlotCodeGeneratorTool Prompt Template

**Location:** `streamlit_app.py` - Lines 135-243  
**Function:** `PlotCodeGeneratorTool()`

**Dynamic Prompt Structure:**
```
Given DataFrame `df` with columns and data types:
{data_context}
{context_section}{date_instructions}{enhancement_note}
Write Python code using pandas **and matplotlib** (as plt) to answer:
"{query}"

Rules & Available Tools
----------------------
1. Use pandas, matplotlib.pyplot (as plt), and seaborn (as sns)
2. For date/time columns, prefer smart_date_parser(df, 'column_name')
3. For categorical columns, convert to numeric: df['col'].map({'Yes': 1, 'No': 0})
4. CRITICAL: Create figure with `fig, ax = plt.subplots(figsize=(8,5))` and assign `result = fig`
5. Create ONE clear, well-labeled plot with ax.set_title(), ax.set_xlabel(), ax.set_ylabel()
6. For time series, consider df.set_index('date_col').plot()
7. Handle missing values with .dropna() before plotting if needed
8. Use clear colors and markers: ax.scatter(), ax.plot(), ax.bar(), etc.
9. Leverage seaborn for enhanced statistical visualizations
10. Wrap code in ```python fence with no explanations
```

**Purpose:** Generates sophisticated visualization code with proper formatting, error handling, and aesthetic considerations.

### CodeWritingTool Prompt Template

**Location:** `streamlit_app.py` - Lines 244-349  
**Function:** `CodeWritingTool()`

**Dynamic Prompt Structure:**
```
Given DataFrame `df` with columns and data types:
{data_context}
{context_section}{date_instructions}{enhancement_note}
Write Python code (pandas **only**, no plotting) to answer:
"{query}"

Rules & Available Tools
----------------------
1. Use pandas operations on `df` only
2. For date/time columns, prefer smart_date_parser(df, 'column_name')
3. For categorical columns, convert to numeric first
4. For correlation analysis, use df[['col1', 'col2']].corr().iloc[0, 1]
5. Handle missing values with .dropna() if needed
6. Always assign the final result to `result` variable
7. Ensure result is a clear, interpretable value
8. Wrap code in a single ```python fence with no explanations
```

**Purpose:** Generates pure data analysis code for statistical computations, aggregations, and transformations.

### DataFrameSummaryTool Prompt Template

**Location:** `streamlit_app.py` - Lines 547-570  
**Function:** `DataFrameSummaryTool()`

**Template 3. For categorical columns, convert to numeric: df['col'].map({{'Yes': 1, 'No': 0}}).
    4. CRITICAL: Create figure with `fig, ax = plt.subplots(figsize=(8,5))` and assign `result = fig`.
    5. Create ONE clear, well-labeled plot with ax.set_title(), ax.set_xlabel(), ax.set_ylabel().
    6. For time series, consider df.set_index('date_col').plot() for automatic time formatting.
    7. Handle missing values with .dropna() before plotting if needed.
    8. Use clear colors and markers: ax.scatter(), ax.plot(), ax.bar(), etc.
    9. Leverage seaborn for enhanced statistical visualizations and better aesthetics.
    10. Wrap code in ```python fence with no explanations.
:**
```
Given a dataset with {len(df)} rows and {len(df.columns)} columns:
Columns: {', '.join(df.columns)}
Data types: {data_types}
Missing values: {missing_values}

Provide:
1. A brief description of what this dataset contains
2. 3-4 possible data analysis questions that could be explored
Keep it concise and focused.
```

**Purpose:** Creates structured prompts for initial dataset analysis and insight generation.

---

## Enhanced Features & Context Systems

### Memory Enhancement System

**Location:** Throughout application via `ColumnMemoryAgent`  
**Enhancement Note Template:**
```
📈 ENHANCED MODE: Detailed AI-generated column descriptions are available above. 
Use this rich context to create more sophisticated and contextually relevant visualizations.
Consider the business meaning, data quality insights, and suggested use cases for each column.
```

**Purpose:** 
- Leverages stored column analysis for more intelligent code generation
- Provides business context for technical analysis
- Enables sophisticated, contextually-aware visualizations
- Improves code quality through domain understanding

### Conversation Memory System

**Location:** `ConversationMemoryTool()` - Lines 67-102  
**Purpose:**
- Maintains context across multi-turn conversations
- Enables follow-up questions and refinements
- Provides conversation history to all analysis agents
- Supports natural dialogue flow

### Date Handling System

**Dynamic Date Instructions Template:**
```
IMPORTANT - Date/Time Handling:
- Detected date/time columns: {', '.join(date_columns)}
- For date parsing, ALWAYS use: pd.to_datetime(df['column_name'], errors='coerce', infer_datetime_format=True)
- This handles various formats like "5/13/25 12:00 AM", "2025-05-13", etc.
- For seasonal analysis, extract month: df['Month'] = pd.to_datetime(df['date_col'], errors='coerce').dt.month
- For seasonal grouping: df['Season'] = df['Month'].map({...})
```

**Purpose:** Provides robust date parsing instructions dynamically added to prompts when date columns are detected.

---

## Execution Environment & Error Handling

### Execution Agent

**Location:** `ExecutionAgent()` - Lines 398-446  
**Environment Setup:**
- Controlled execution environment with `pd`, `np`, `df`, `plt`, `sns`, `smart_date_parser`
- Proper error handling with descriptive messages
- Result validation and type checking
- Enhanced error guidance for common issues

### Error Guidance System

**Smart Error Messages:**
```python
if "not defined" in str(exc):
    error_msg += f"\n💡 Tip: Available variables are: {list(env.keys())}"
elif "KeyError" in str(exc):
    error_msg += f"\n💡 Tip: Available columns are: {list(df.columns)}"
```

**Purpose:** Provides actionable debugging information when code execution fails.

---

## Model Configuration

**Model:** `nvidia/llama-3.1-nemotron-ultra-253b-v1`  
**Common Parameters:**
- Temperature: 0.1-0.3 (depending on task)
- Max tokens: 512-1024 (depending on expected output length)
- Streaming: Enabled for reasoning agent

**Temperature Tuning by Task:**
- Query Understanding: 0.1 (precise classification)
- Code Generation: 0.2 (reliable code)
- Column Analysis: 0.3 (creative insights)
- Dataset Insights: 0.2 (focused summaries)

---

## Summary

The Business Analysis HR Agent employs a sophisticated multi-agent architecture where each agent has specialized prompts optimized for specific tasks:

1. **Classification Agents** determine the type of analysis needed
2. **Code Generation Agents** create executable analysis code
3. **Reasoning Agents** interpret and explain results
4. **Analysis Agents** provide deep insights about data structure
5. **Memory Systems** enhance context across interactions

This architecture ensures high-quality, contextually-aware analysis while maintaining code reliability and user experience. The system prompts are designed to work together as a cohesive analytical framework, with each agent contributing specialized expertise to the overall analysis workflow.
