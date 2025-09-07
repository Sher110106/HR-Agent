import io
import logging
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend to prevent plots from opening in new windows
import chardet
from datetime import datetime
from typing import Dict, Any
import os

# PandasAI imports
import pandasai as pai
try:
    from pandasai_openai import AzureOpenAI
    PANDASAI_OPENAI_AVAILABLE = True
except ImportError:
    AzureOpenAI = None
    PANDASAI_OPENAI_AVAILABLE = False

from agents import (
    ColumnMemoryAgent, CodeGenerationAgent, ExecutionAgent, 
    ReasoningAgent, DataInsightAgent
)
from agents.memory import SystemPromptMemoryAgent
from utils.system_prompts import get_prompt_manager
from app_core.api import get_available_models, make_llm_call
from env_config import (
    AZURE_API_KEY, AZURE_ENDPOINT, AZURE_API_VERSION, AZURE_DEPLOYMENT_NAME
)
# Import Phase 1 and Phase 2 plot enhancements
from utils.plot_helpers import (
    PlotMemory, is_plot_modification_request, generate_plot_modification_code,
    create_enhanced_chart_with_insights, detect_insights, add_insight_annotations,
    get_hr_specific_colors, get_contextual_colors, apply_modern_styling
)

logger = logging.getLogger(__name__)

def get_llm_call_with_selected_model():
    """Get make_llm_call function with the selected model from session state."""
    def llm_call_wrapper(messages, **kwargs):
        # Get selected model from session state, fallback to default
        selected_model = getattr(st.session_state, 'selected_model', 'gpt-4.1')
        return make_llm_call(messages, model=selected_model, **kwargs)
    
    return llm_call_wrapper

def initialize_pandasai():
    """Initialize PandasAI with Azure OpenAI."""
    try:
        # Check if pandasai_openai is available
        if not PANDASAI_OPENAI_AVAILABLE:
            st.warning("⚠️ PandasAI OpenAI integration not available. Smart analysis will use fallback mode.")
            return None
        
        # Check if Azure OpenAI is configured
        if not AZURE_API_KEY:
            st.error("❌ Azure OpenAI API key not found. Please add it to Streamlit secrets under [azure].")
            return None
        
        if not AZURE_ENDPOINT:
            st.error("❌ Azure OpenAI endpoint not found. Please add it to Streamlit secrets under [azure].")
            return None
        
        # Initialize Azure OpenAI LLM
        llm = AzureOpenAI(
            api_token=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
            deployment_name=AZURE_DEPLOYMENT_NAME
        )
        
        # Configure PandasAI
        pai.config.set({
            "llm": llm,
            "verbose": st.session_state.get('verbose_mode', False)
        })
        
        return llm
        
    except Exception as e:
        logger.error(f"Failed to initialize PandasAI with Azure OpenAI: {e}")
        st.error(f"❌ Failed to initialize PandasAI with Azure OpenAI: {str(e)}")
        return None

def is_plot_query(question):
    """Check if the question involves plotting or visualization."""
    plot_keywords = [
        'plot', 'chart', 'graph', 'visualize', 'histogram', 'bar chart', 
        'scatter plot', 'line chart', 'pie chart', 'box plot', 'heatmap',
        'distribution', 'show me', 'create a', 'draw', 'display'
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in plot_keywords)

def analyze_with_pandasai(df, question):
    """Analyze data using PandasAI with Azure OpenAI and reasoning agent, storing results in session state."""
    # Check if pandasai_openai is available
    if not PANDASAI_OPENAI_AVAILABLE:
        st.warning("⚠️ PandasAI OpenAI integration not available. Using fallback analysis mode.")
        # Use the existing agents for analysis instead
        from agents.code_generation import CodeGenerationAgent
        from agents.execution import ExecutionAgent
        
        # Generate code using the existing agent
        code, should_plot, error = CodeGenerationAgent(question, df)
        if error:
            st.error(f"❌ Error generating analysis code: {error}")
            return
        
        # Execute the code
        result = ExecutionAgent(code, df)
        if result:
            st.success("✅ Analysis completed using fallback mode")
            return result
        else:
            st.error("❌ Failed to execute analysis code")
            return
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            with st.spinner(f"🤖 PandasAI is analyzing your data... (Attempt {retry_count + 1}/{max_retries})"):
                # Initialize PandasAI with Azure OpenAI
                llm = initialize_pandasai()
                if llm is None:
                    return
                
                # Create PandasAI DataFrame
                pai_df = pai.DataFrame(df)
                
                # Modify question for retry attempts to be more specific
                modified_question = question
                if retry_count > 0:
                    if is_plot_query(question):
                        modified_question = f"{question} Please ensure you return plots with the correct format. You can return multiple plots if needed."
                    else:
                        modified_question = f"{question} Please provide a clear analysis with the correct output format."
                
                # Perform analysis with PandasAI
                response = pai_df.chat(modified_question)
                
                # Debug: Log the response type and attributes
                logger.info(f"PandasAI response type: {type(response)}")
                logger.info(f"PandasAI response attributes: {dir(response)}")
                if hasattr(response, 'value'):
                    logger.info(f"PandasAI response value: {response.value}")
                if hasattr(response, 'figure'):
                    logger.info(f"PandasAI response has figure: {response.figure is not None}")
                if hasattr(response, 'chart'):
                    logger.info(f"PandasAI response has chart: {response.chart is not None}")
                
                # Debug: Log the string representation of the response
                logger.info(f"PandasAI response string: {str(response)}")
                
                # Debug: Check if response is a string (might contain error info)
                if isinstance(response, str):
                    logger.info(f"PandasAI returned string response: {response}")
                    if "error" in response.lower() or "invalid" in response.lower():
                        logger.warning(f"PandasAI response contains error indicators: {response}")
                
                # Initialize session state for plots and messages if not exists
                if "plots" not in st.session_state:
                    st.session_state.plots = []
                if "plot_data" not in st.session_state:
                    st.session_state.plot_data = []
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                
                # Initialize plot memory if not exists
                if "plot_memory" not in st.session_state:
                    st.session_state.plot_memory = PlotMemory()
                
                # Store user message
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Process response and store in session state
                plot_idx = None
                data_idx = None
                code_content = None
                
                # Handle different types of PandasAI responses
                if hasattr(response, 'figure') and response.figure is not None:
                    # Direct matplotlib figure
                    st.session_state.plots.append(response.figure)
                    plot_idx = len(st.session_state.plots) - 1
                    st.session_state.plot_data.append(df)
                    data_idx = len(st.session_state.plot_data) - 1
                    
                elif hasattr(response, 'chart') and response.chart is not None:
                    # ChartResponse object
                    st.session_state.plots.append(response.chart)
                    plot_idx = len(st.session_state.plots) - 1
                    st.session_state.plot_data.append(df)
                    data_idx = len(st.session_state.plot_data) - 1
                    
                elif isinstance(response, str):
                    # PandasAI returned a string (might be error message or file path)
                    logger.info(f"PandasAI returned string response: {response}")
                    
                    if response.endswith('.png'):
                        # String contains a file path
                        try:
                            import matplotlib.image as mpimg
                            import matplotlib.pyplot as plt
                            
                            # Load the image
                            img = mpimg.imread(response)
                            
                            # Create a new figure and display the image
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.imshow(img)
                            ax.axis('off')
                            
                            st.session_state.plots.append(fig)
                            plot_idx = len(st.session_state.plots) - 1
                            st.session_state.plot_data.append(df)
                            data_idx = len(st.session_state.plot_data) - 1
                            
                        except Exception as e:
                            logger.warning(f"Could not load plot from string path: {e}")
                    else:
                        # String might contain error information
                        logger.warning(f"PandasAI returned string that might contain error: {response}")
                        
                elif hasattr(response, 'value') and isinstance(response.value, str) and response.value.endswith('.png'):
                    # PandasAI saved a plot file
                    try:
                        # Try to load the saved plot
                        import matplotlib.image as mpimg
                        import matplotlib.pyplot as plt
                        
                        # Load the image
                        img = mpimg.imread(response.value)
                        
                        # Create a new figure and display the image
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.imshow(img)
                        ax.axis('off')
                        
                        st.session_state.plots.append(fig)
                        plot_idx = len(st.session_state.plots) - 1
                        st.session_state.plot_data.append(df)
                        data_idx = len(st.session_state.plot_data) - 1
                        
                    except Exception as e:
                        logger.warning(f"Could not load saved plot: {e}")
                        
                elif hasattr(response, 'value') and isinstance(response.value, list):
                    # PandasAI returned multiple images (invalid format, but we can handle it)
                    try:
                        import matplotlib.image as mpimg
                        import matplotlib.pyplot as plt
                        
                        # Take the first image from the list
                        if response.value and len(response.value) > 0:
                            first_image_path = response.value[0]
                            if isinstance(first_image_path, str) and first_image_path.endswith('.png'):
                                # Load the image
                                img = mpimg.imread(first_image_path)
                                
                                # Create a new figure and display the image
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.imshow(img)
                                ax.axis('off')
                                
                                st.session_state.plots.append(fig)
                                plot_idx = len(st.session_state.plots) - 1
                                st.session_state.plot_data.append(df)
                                data_idx = len(st.session_state.plot_data) - 1
                                
                    except Exception as e:
                        logger.warning(f"Could not load multiple images: {e}")
                        
                elif hasattr(response, 'value') and isinstance(response.value, dict):
                    # PandasAI returned a dictionary (might contain image paths)
                    try:
                        if 'value' in response.value and isinstance(response.value['value'], list):
                            # Handle list of image paths
                            first_image_path = response.value['value'][0]
                            if isinstance(first_image_path, str) and first_image_path.endswith('.png'):
                                import matplotlib.image as mpimg
                                import matplotlib.pyplot as plt
                                
                                # Load the image
                                img = mpimg.imread(first_image_path)
                                
                                # Create a new figure and display the image
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.imshow(img)
                                ax.axis('off')
                                
                                st.session_state.plots.append(fig)
                                plot_idx = len(st.session_state.plots) - 1
                                st.session_state.plot_data.append(df)
                                data_idx = len(st.session_state.plot_data) - 1
                                
                    except Exception as e:
                        logger.warning(f"Could not load from dictionary response: {e}")
                
                # Add to plot memory if we have a plot
                if plot_idx is not None:
                    memory_idx = st.session_state.plot_memory.add_plot(
                        fig=st.session_state.plots[plot_idx],
                        data_df=df,
                        context=question,
                        chart_type='pandasai_generated',
                        styling={'theme': 'professional', 'insights': True}
                    )
                    
                    # Save chart if enabled
                    if st.session_state.get('save_charts', True):
                        try:
                            os.makedirs('exports/charts', exist_ok=True)
                            chart_filename = f"smart_analysis_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                            chart_path = f"exports/charts/{chart_filename}"
                            st.session_state.plots[plot_idx].savefig(chart_path, dpi=300, bbox_inches='tight')
                            logger.info(f"✅ Chart saved as {chart_path}")
                        except Exception as e:
                            logger.warning(f"⚠️ Could not save chart: {str(e)}")
                
                # If this is a plot query, make a second call to get numerical data for reasoning agent
                numerical_data_response = None
                if is_plot_query(question):
                    try:
                        with st.spinner("📊 Getting numerical data for analysis..."):
                            # Create a modified question to get numerical data
                            numerical_question = f"Provide numerical analysis for: {question}. Give me the key statistics, numbers, and insights without creating any plots."
                            numerical_data_response = pai_df.chat(numerical_question)
                            logger.info(f"Numerical data response: {numerical_data_response}")
                    except Exception as e:
                        logger.warning(f"Failed to get numerical data: {e}")
                
                # Use reasoning agent to enhance the PandasAI response
                # Create a result object that the reasoning agent can work with
                result_obj = response
                
                # If there's a plot, create a tuple format like other analysis modes
                if plot_idx is not None:
                    result_obj = (st.session_state.plots[plot_idx], df)
                
                # Combine responses for reasoning agent
                combined_response = str(response)
                if numerical_data_response:
                    combined_response += f"\n\n**📊 Numerical Analysis:**\n{str(numerical_data_response)}"
                
                # Get reasoning agent analysis
                raw_thinking, reasoning_txt = ReasoningAgent(question, result_obj)
                reasoning_txt = reasoning_txt.replace("`", "")
                
                # Create enhanced content combining PandasAI response and reasoning
                enhanced_content = f"""
**🤖 PandasAI Analysis:**
{combined_response}

**🧠 AI Reasoning:**
{reasoning_txt}
"""
                
                # Create assistant message with proper structure
                assistant_message = {
                    "role": "assistant",
                    "content": enhanced_content,
                    "plot_index": plot_idx,
                    "data_index": data_idx,
                    "code": code_content
                }
                
                st.session_state.messages.append(assistant_message)
                
                if retry_count > 0:
                    st.success(f"✅ Analysis complete! (Succeeded on attempt {retry_count + 1})")
                else:
                    st.success("✅ Analysis complete!")
                
                # Success - break out of retry loop
                break
                
        except Exception as e:
            retry_count += 1
            logger.error(f"PandasAI analysis error (attempt {retry_count}): {e}")
            
            # Check if this is a retryable error (output type issues)
            retryable_errors = [
                "Invalid output type: image",
                "Invalid output type: images", 
                "Invalid output type: chart",
                "InvalidOutputValueMismatch",
                "Result must be in the format of dictionary",
                "Invalid output type"
            ]
            
            is_retryable = any(error in str(e) for error in retryable_errors)
            
            # If this is the last attempt or not retryable, show error
            if retry_count >= max_retries or not is_retryable:
                # Provide specific error messages for common issues
                error_message = f"❌ Analysis failed: {str(e)}"
                
                if "Invalid output type" in str(e):
                    error_message = f"""
❌ **PandasAI Output Error**

The analysis failed because PandasAI tried to return an invalid output type. This usually happens when:
- Multiple plots are generated but not properly formatted (this is now supported)
- The response format doesn't match PandasAI's expected output types

**Valid PandasAI output types:**
- `{{"type": "plot", "value": "path/to/plot.png"}}` ✅ **CORRECT**
- `{{"type": "string", "value": "text response"}}`
- `{{"type": "number", "value": 42}}`
- `{{"type": "dataframe", "value": df}}`

**Common Mistakes:**
- ❌ `{{"type": "image", "value": "path.png"}}` - Use "plot" instead of "image"
- ❌ `{{"type": "images", "value": ["path1.png", "path2.png"]}}` - Use "plot" for single image
- ❌ `{{"type": "chart", "value": "path.png"}}` - Use "plot" instead of "chart"

**Try:**
- Rephrasing your question to be more specific
- Using simpler visualization requests
- Using the correct output type: `{{"type": "plot", "value": "path.png"}}`
"""
                elif "InvalidOutputValueMismatch" in str(e):
                    error_message = f"""
❌ **PandasAI Response Format Error**

The analysis failed because the generated code didn't return the expected format.

**Expected format:**
```python
result = {{"type": "plot", "value": "path/to/plot.png"}}
```

**Common Issues:**
- Missing `result =` assignment
- Wrong type name (use "plot", not "image" or "chart")
- Missing quotes around the value
- Incorrect file path format
- Missing closing brace or parenthesis
- Using `return` instead of `result =`

**Examples of WRONG formats:**
```python
# ❌ WRONG - Missing result assignment
{{"type": "plot", "value": "path.png"}}

# ❌ WRONG - Wrong type
result = {{"type": "image", "value": "path.png"}}

# ❌ WRONG - Missing quotes
result = {{"type": "plot", "value": path.png}}

# ❌ WRONG - Using return
return {{"type": "plot", "value": "path.png"}}
```

**Examples of CORRECT formats:**
```python
# ✅ CORRECT
result = {{"type": "plot", "value": "path/to/plot.png"}}

# ✅ CORRECT
result = {{"type": "string", "value": "Analysis complete"}}

# ✅ CORRECT
result = {{"type": "number", "value": 42}}
```

**Try:**
- Rephrasing your question
- Using simpler visualizations
- Using more specific data analysis questions
- Ensuring the code returns the correct format
"""
                elif "Result must be in the format of dictionary" in str(e):
                    error_message = f"""
❌ **PandasAI Code Generation Error**

The analysis failed because the generated code doesn't return the result in the correct format.

**The Problem:**
The generated code is missing the proper result assignment or format.

**Required Format:**
```python
# At the end of your code, you MUST have:
result = {{"type": "plot", "value": "path/to/plot.png"}}
```

**Common Mistakes:**
```python
# ❌ WRONG - Missing result assignment
{{"type": "plot", "value": "path.png"}}

# ❌ WRONG - Wrong variable name
output = {{"type": "plot", "value": "path.png"}}

# ❌ WRONG - Missing quotes
result = {{"type": "plot", "value": path.png}}

# ❌ WRONG - Using return
return {{"type": "plot", "value": "path.png"}}
```

**Correct Format:**
```python
# ✅ CORRECT
result = {{"type": "plot", "value": "path/to/plot.png"}}
```

**Try:**
- Rephrasing your question to be more specific
- Using simpler visualization requests
- The system will automatically retry with the correct format
"""
                elif "Invalid output type: image" in str(e):
                    error_message = f"""
❌ **PandasAI Type Error: "image" is not valid**

The analysis failed because the code used `"type": "image"` which is not a valid PandasAI output type.

**The Problem:**
```python
# ❌ WRONG
result = {{"type": "image", "value": "path.png"}}

# ✅ CORRECT  
result = {{"type": "plot", "value": "path.png"}}
```

**Valid PandasAI output types:**
- `{{"type": "plot", "value": "path/to/plot.png"}}` ✅ **Use this for images/plots**
- `{{"type": "string", "value": "text"}}`
- `{{"type": "number", "value": 42}}`
- `{{"type": "dataframe", "value": df}}`

**Try:**
- Rephrasing your question to be more specific
- The system will automatically retry with the correct format
- Use simpler visualization requests
"""
                
                # Store error message in session state
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                
                error_msg_obj = {
                    "role": "assistant",
                    "content": error_message,
                    "plot_index": None,
                    "data_index": None,
                    "code": None
                }
                st.session_state.messages.append(error_msg_obj)
                
                # Provide helpful error message
                st.markdown("**💡 Troubleshooting Tips:**")
                st.markdown("""
                - Make sure your question is clear and specific
                - Try rephrasing your question
                - Check that your data contains the columns you're asking about
                - For complex questions, try breaking them down into simpler parts
                - Ensure your Azure OpenAI API key and endpoint are properly configured
                - Try asking for a single plot instead of multiple plots
                """)
                
                # If this is a retryable error, suggest retry
                if is_retryable:
                    st.markdown("**🔄 This looks like a retryable error. You can try asking the same question again - the system may automatically fix the output format.**")
            else:
                # Show retry message
                st.warning(f"⚠️ Attempt {retry_count} failed. Retrying with corrected format...")
                continue

def render_system_prompt_sidebar():
    """Render system prompt controls in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 AI Behavior")
    
    # Model Selection
    st.sidebar.markdown("**🤖 AI Model:**")
    available_models = get_available_models()
    
    if not available_models:
        st.sidebar.error("❌ No AI models available. Please check your API configuration.")
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
        st.sidebar.success(f"✅ Switched to {available_models[selected_model_key]}")
        st.rerun()
    
    # Show current model info
    current_model_name = available_models.get(st.session_state.selected_model, "Unknown")
    st.sidebar.info(f"🤖 Active Model: **{current_model_name}**")
    
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
    
    # Update active prompt if changed
    if selected_prompt != "Default" and selected_prompt != current_selection:
        try:
            prompt_manager.set_active_prompt(selected_prompt)
            st.sidebar.success(f"✅ Switched to {selected_prompt} analysis style")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Failed to switch prompt: {str(e)}")
    
    # Advanced Settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Advanced Settings")
    
    # Verbose mode toggle
    verbose_mode = st.sidebar.checkbox(
        "Verbose Mode",
        value=st.session_state.get('verbose_mode', False),
        help="Show detailed processing information"
    )
    
    if verbose_mode != st.session_state.get('verbose_mode', False):
        st.session_state.verbose_mode = verbose_mode
        st.rerun()
    
    # Chart saving toggle
    save_charts = st.sidebar.checkbox(
        "Save Charts",
        value=st.session_state.get('save_charts', True),
        help="Automatically save generated charts to exports/charts/"
    )
    
    if save_charts != st.session_state.get('save_charts', True):
        st.session_state.save_charts = save_charts
        st.rerun()

def render_file_upload():
    """Render file upload section."""
    st.markdown("### 📁 Upload Your Data")
    
    uploaded_file = st.file_uploader(
        "Choose a data file",
        type=['csv', 'xlsx', 'xls', 'parquet'],
        help="Supported formats: CSV, Excel (.xlsx, .xls), Parquet"
    )
    
    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Try to detect encoding
                content = uploaded_file.read()
                result = chardet.detect(content)
                encoding = result['encoding'] if result['confidence'] > 0.7 else 'utf-8'
                
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Read CSV with detected encoding
                df = pd.read_csv(uploaded_file, encoding=encoding)
                
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
                
            elif file_extension == 'parquet':
                df = pd.read_parquet(uploaded_file)
                
            else:
                st.error(f"❌ Unsupported file type: {file_extension}")
                return None
            
            # Store in session state
            st.session_state.df = df
            st.success(f"✅ Successfully loaded {uploaded_file.name}")
            return df
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            logger.error(f"File upload error: {e}")
            return None
    
    return st.session_state.get('df', None)

def render_data_preview(df):
    """Render data preview section."""
    st.markdown("### 📊 Data Preview")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Rows", len(df))
    
    with col2:
        st.metric("Columns", len(df.columns))
    
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Data types
    st.markdown("**📋 Data Types:**")
    dtype_df = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum()
    })
    st.dataframe(dtype_df, use_container_width=True)
    
    # Sample data
    st.markdown("**📋 Sample Data:**")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Missing values
    if df.isnull().sum().sum() > 0:
        st.markdown("**⚠️ Missing Values:**")
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum(),
            'Missing Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        st.dataframe(missing_df, use_container_width=True)

def render_chat_interface():
    """Render the chat interface with standardized message display."""
    st.markdown("### 💬 Smart Analysis Chat")
    
    # Initialize session state if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "plots" not in st.session_state:
        st.session_state.plots = []
    if "plot_data" not in st.session_state:
        st.session_state.plot_data = []
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                # Enhanced error display for assistant messages
                if msg["role"] == "assistant" and isinstance(msg["content"], str) and (
                    msg["content"].startswith("❌ Analysis failed:") or msg["content"].startswith("Error:")
                ):
                    # Show error prominently
                    st.error("An error occurred while processing your request.")
                    # Show summary and collapsible technical details
                    error_lines = msg["content"].split("\n")
                    summary = error_lines[0]
                    details = "\n".join(error_lines[1:]) if len(error_lines) > 1 else ""
                    st.markdown(f"**Summary:** {summary}")
                    if details.strip():
                        with st.expander("Technical Details", expanded=False):
                            st.code(details, language="text")
                    continue
                
                # Normal message rendering
                st.markdown(msg["content"], unsafe_allow_html=True)
                
                # For assistant messages, add download options
                if msg["role"] == "assistant":
                    # Check if this is dual-output (has both plot and data)
                    has_data = msg.get("data_index") is not None
                    has_plot = msg.get("plot_index") is not None
                    
                    if has_data and has_plot:
                        # Dual output: 4 columns for text, data CSV, plot PNG, and spacing
                        col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
                    else:
                        # Legacy: 3 columns
                        col1, col2, col3 = st.columns([2, 1, 1])
                        col4 = None
                    
                    with col1 if not has_data else col2:
                        # Download text response
                        # Extract clean text from the response (remove HTML)
                        import re
                        clean_text = re.sub(r'<[^>]+>', '', msg["content"])
                        clean_text = re.sub(r'\n+', '\n', clean_text).strip()
                        
                        if clean_text:
                            # Convert to DOCX
                            from utils.docx_utils import text_to_docx
                            docx_data = text_to_docx(clean_text, title=f"Smart Analysis Response {i+1}")
                            st.download_button(
                                label="📝 DOCX",
                                data=docx_data,
                                file_name=f"smart_analysis_response_{i+1}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                use_container_width=True
                            )
                    
                    # Download CSV data if available (dual-output only)
                    if has_data and col4:
                        with col3:
                            data_idx = msg["data_index"]
                            if 0 <= data_idx < len(st.session_state.get("plot_data", [])):
                                data_df = st.session_state.plot_data[data_idx]
                                
                                # Convert DataFrame to CSV
                                csv_buffer = io.StringIO()
                                data_df.to_csv(csv_buffer, index=False)
                                csv_data = csv_buffer.getvalue()
                                
                                st.download_button(
                                    label="📊 CSV",
                                    data=csv_data,
                                    file_name=f"smart_analysis_data_{i+1}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                    
                    # Download plot if available
                    plot_col = col4 if has_data else col3
                    if has_plot and plot_col:
                        with plot_col:
                            plot_idx = msg["plot_index"]
                            if 0 <= plot_idx < len(st.session_state.plots):
                                # Create download button for plot
                                fig = st.session_state.plots[plot_idx]
                                
                                # Save plot to bytes buffer
                                img_buffer = io.BytesIO()
                                if hasattr(fig, 'data') and hasattr(fig, 'layout'):
                                    # Plotly figure - export as HTML (reliable)
                                    from utils.plot_helpers import safe_plotly_to_html
                                    html_content = safe_plotly_to_html(fig)
                                    img_buffer.write(html_content.encode('utf-8'))
                                else:
                                    # Matplotlib figure - PNG export (reliable)
                                    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                                img_buffer.seek(0)
                                
                                # Determine file format and MIME type
                                if hasattr(fig, 'data') and hasattr(fig, 'layout'):
                                    file_name = f"smart_analysis_plot_{i+1}.html"
                                    mime_type = "text/html"
                                    label = "📄 Download Plot (HTML)"
                                else:
                                    file_name = f"smart_analysis_plot_{i+1}.png"
                                    mime_type = "image/png"
                                    label = "🖼️ Download Plot (PNG)"
                                
                                st.download_button(
                                    label=label,
                                    data=img_buffer.getvalue(),
                                    file_name=file_name,
                                    mime=mime_type,
                                    use_container_width=True
                                )
                
                # Display plot
                if msg.get("plot_index") is not None:
                    idx = msg["plot_index"]
                    if 0 <= idx < len(st.session_state.plots):
                        fig = st.session_state.plots[idx]
                        # Check if it's a Plotly figure or matplotlib
                        if hasattr(fig, 'data') and hasattr(fig, 'layout'):
                            # Plotly figure - clean for safe Streamlit display with proper theming
                            from utils.plot_helpers import safe_plotly_figure_for_streamlit
                            safe_fig = safe_plotly_figure_for_streamlit(fig, theme='auto', variant='professional')
                            st.plotly_chart(safe_fig, use_container_width=True)
                        else:
                            # Matplotlib figure (legacy)
                            st.pyplot(fig, use_container_width=False)
                    
                    # Display multiple plots
                    elif msg.get("plot_indices") is not None:
                        # Multi-plot display
                        indices = msg["plot_indices"]
                        st.markdown(f"**📊 {len(indices)} Visualizations:**")
                        
                        for i, idx in enumerate(indices):
                            if 0 <= idx < len(st.session_state.plots):
                                fig = st.session_state.plots[idx]
                                
                                # Add chart number
                                st.markdown(f"**Chart {i+1}:**")
                                
                                # Check if it's a Plotly figure or matplotlib
                                if hasattr(fig, 'data') and hasattr(fig, 'layout'):
                                    # Plotly figure - clean for safe Streamlit display with proper theming
                                    from utils.plot_helpers import safe_plotly_figure_for_streamlit
                                    safe_fig = safe_plotly_figure_for_streamlit(fig, theme='auto', variant='professional')
                                    st.plotly_chart(safe_fig, use_container_width=True)
                                else:
                                    # Matplotlib figure (legacy)
                                    st.pyplot(fig, use_container_width=False)
                                
                                # Add separator between charts
                                if i < len(indices) - 1:
                                    st.markdown("---")
                
                # Display code in a proper expander for assistant messages
                if msg.get("code") and msg["role"] == "assistant":
                    with st.expander("View code", expanded=False):
                        st.code(msg["code"], language="python")

def render_smart_analysis_interface(df):
    """Render the Smart AI-powered analysis interface."""
    st.markdown("### 🤖 Smart Analysis")
    st.markdown("Ask questions about your data in natural language using PandasAI with Azure OpenAI!")
    
    # Chat interface
    st.markdown("#### 💬 Ask Questions About Your Data")
    
    # Question input
    question = st.text_area(
        "Ask a question about your data:",
        placeholder="e.g., What is the average salary by department? Plot the distribution of ages. Who are the top 5 employees by salary?",
        height=100,
        help="Ask questions in natural language. PandasAI will understand and analyze your data!"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("🚀 Analyze", type="primary"):
            if question.strip():
                analyze_with_pandasai(df, question)
                st.rerun()
            else:
                st.warning("Please enter a question to analyze.")
    
    with col2:
        if st.button("💡 Example Questions"):
            show_example_questions()

def show_example_questions():
    """Show example questions for users."""
    st.markdown("### 💡 Example Questions")
    
    examples = [
        "What is the average salary by department?",
        "Plot the distribution of employee ages",
        "Who are the top 5 employees by salary?",
        "Show me the correlation between age and salary",
        "Create a bar chart of employees by department",
        "What is the salary range for each department?",
        "Plot a histogram of salaries",
        "Show me employees with salaries above the average",
        "Create a scatter plot of age vs salary",
        "What is the total salary cost by department?"
    ]
    
    for i, example in enumerate(examples, 1):
        if st.button(f"Example {i}: {example}", key=f"example_{i}"):
            st.session_state.example_question = example
            st.rerun()

def smart_analysis_page():
    """Main Smart Analysis page using PandasAI with Azure OpenAI."""
    st.title("🤖 Smart Analysis")
    st.markdown("**Powered by PandasAI with Azure OpenAI - Natural Language Data Analysis**")
    
    # Render sidebar controls
    render_system_prompt_sidebar()
    
    # Initialize session state
    if "df" not in st.session_state:
        st.session_state.df = None
    
    # File upload section
    df = render_file_upload()
    
    if df is not None:
        # Data preview
        render_data_preview(df)
        
        # Smart analysis interface
        render_smart_analysis_interface(df)
        
        # Chat interface for displaying results
        render_chat_interface()
        
        # Traditional analysis option
        st.markdown("---")
        st.markdown("### 🔄 Switch to Traditional Analysis")
        if st.button("📊 Use Traditional Data Analysis"):
            st.session_state.switch_to_traditional = True
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 Smart Analysis powered by PandasAI with Azure OpenAI</p>
        <p>Ask questions in natural language and get instant insights!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    smart_analysis_page() 