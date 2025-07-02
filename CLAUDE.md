# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment (required)
export NVIDIA_API_KEY="your_api_key_here"

# Run the application
streamlit run streamlit_app.py
```

### Dependencies
Core dependencies are managed in `requirements.txt`:
- `streamlit>=1.32.0` - Web application framework
- `pandas>=2.2.0` - Data manipulation
- `matplotlib>=3.8.0` - Plotting
- `seaborn>=0.13.0` - Statistical visualization
- `openai>=1.12.0` - NVIDIA API client
- `watchdog>=3.0.0` - File monitoring
- `chardet>=5.0.0` - Encoding detection

## Architecture Overview

This is a Streamlit-based AI-powered data analysis application that processes CSV uploads and generates insights using NVIDIA's Llama-3.1-Nemotron-Ultra model.

### Key Components

**streamlit_app.py** - Main application entry point containing:
- Authentication system (hardcoded credentials: `Plaksha-HR`/`AgentHR1`)
- File upload and CSV processing with automatic encoding detection
- Chat interface for natural language data queries
- Session state management for plots and conversation history

**data_analysis_agent.py** - Core agent logic (currently empty but referenced in README)

### AI Agent Architecture

The application uses a multi-agent system with specialized roles:

1. **Memory System** (`ColumnMemoryAgent`):
   - Stores AI-generated column descriptions
   - Enhances subsequent analysis with contextual understanding
   - Parallel column analysis with intelligent worker scaling

2. **Code Generation Pipeline**:
   - `QueryUnderstandingTool` - Determines if query needs visualization
   - `PlotCodeGeneratorTool` - Generates matplotlib/seaborn plotting code
   - `CodeWritingTool` - Generates pandas analysis code
   - `CodeGenerationAgent` - Orchestrates the above tools

3. **Execution & Reasoning**:
   - `ExecutionAgent` - Safely executes generated code with validation
   - `ReasoningAgent` - Provides business insights and recommendations
   - Auto-retry mechanism for common pandas errors

### Data Flow

```
CSV Upload → DataInsightAgent → User Query → CodeGenerationAgent → ExecutionAgent → ReasoningAgent → Streamlit UI
```

### Key Features

- **Secure Authentication**: Username/password protection
- **Smart CSV Processing**: Automatic encoding detection with fallbacks
- **Enhanced Column Analysis**: AI-powered column understanding for better insights

#### Enhanced Visualization System (NEW)
- **Dual-Output Plots**: Returns both matplotlib figure AND underlying source data
- **Automatic Value Labels**: Every bar/point displays exact numeric values
- **Professional Styling**: Consistent color palettes and formatting via helper functions
- **Smart Axis Formatting**: Auto-rotation and wrapping of long labels
- **Data Table Display**: Interactive tables showing plot source data
- **Enhanced Export Options**: Download plots (PNG) and data (CSV) separately

#### Core Capabilities
- **Error Recovery**: Automatic retry with error context for failed code execution
- **Conversation Memory**: Context-aware follow-up questions
- **Backward Compatibility**: Legacy single-output plots still supported

## Important Implementation Details

### Error Handling
The application includes sophisticated error handling:
- Encoding detection for CSV files with fallback encodings
- Pandas code validation with common error patterns
- Automatic retry mechanism for failed code execution
- Specific error guidance for users

### AI Integration
- Uses NVIDIA's Llama-3.1-Nemotron-Ultra-253B-v1 model
- Streaming responses with visible "thinking" process
- Temperature settings: 0.1 for query understanding, 0.2-0.3 for generation
- Context-aware prompts with conversation history

### Session Management
- Authentication state persistence
- Plot storage in session state
- Conversation history with downloadable responses
- Column memory system for enhanced analysis

### Security Considerations
- Hardcoded authentication credentials (consider environment variables for production)
- Code execution in controlled environment with limited scope
- Session-based data isolation

## Development Notes

- The `data_analysis_agent.py` file is currently empty but likely intended for core logic
- All AI agent functionality is currently in `streamlit_app.py`
- No test files or CI/CD configuration detected
- Application is designed for Streamlit Cloud deployment with secrets management

## Recently Implemented Features

The following enhancements from `plan.md` have been successfully implemented:

### Dual-Output Plot System ✅
- **Plot + Data Tuples**: All visualization queries now return `(fig, data_df)` tuples
- **Enhanced UI**: Streamlit interface displays both plots and interactive data tables
- **Export Options**: Separate download buttons for PNG plots and CSV data

### Professional Chart Enhancements ✅  
- **Automatic Value Labels**: `add_value_labels()` function adds values to all bars/points
- **Smart Axis Formatting**: `format_axis_labels()` handles rotation and wrapping
- **Professional Styling**: `apply_professional_styling()` ensures consistent appearance
- **Color Palettes**: `get_professional_colors()` provides business-appropriate colors

### Helper Utilities ✅
- **Plot Helper Functions**: Located in `utils/plot_helpers.py`
- **Available in LLM Context**: AI can use helper functions in generated code
- **Backward Compatible**: Legacy plots continue to work alongside new format

### Architecture Updates ✅
- **ExecutionAgent**: Enhanced to detect and handle tuple results
- **ReasoningCurator**: Updated prompts for dual-output analysis  
- **UI Components**: Data table display with summary statistics
- **Session Management**: Separate storage for plots and data arrays

## Using the Enhanced Plotting System

### For Users
When requesting visualizations, you'll now get:
1. **Professional Charts**: Automatic value labels, proper formatting, and consistent styling
2. **Source Data Tables**: Interactive tables showing the exact data used to create plots
3. **Export Options**: Download both the chart (PNG) and underlying data (CSV)
4. **Enhanced Analysis**: AI reasoning includes both visual and numerical insights

### For Developers
The new system automatically:
1. **Generates Tuple Results**: `result = (fig, data_df)` for all plot queries
2. **Applies Helper Functions**: Value labels, axis formatting, and professional styling
3. **Handles Legacy Code**: Existing single-result plots continue to work
4. **Provides Rich Context**: AI has access to both visual and data components

### Example Generated Code Structure
```python
# Data preparation
data_df = df.groupby('category')['value'].sum().reset_index()

# Create enhanced plot
fig, ax = plt.subplots(figsize=(12, 7))
colors = get_professional_colors()['colors']
ax.bar(data_df['category'], data_df['value'], color=colors[0])

# Apply enhancements
add_value_labels(ax)
format_axis_labels(ax, x_rotation=45)
apply_professional_styling(ax, title="Title", xlabel="X", ylabel="Y")

# Return both components
result = (fig, data_df)
```