# User Manual – Business Analysis HR Agent

> **Welcome!** This guide will help you go from zero to actionable HR insights in minutes.
> 
> **📖 For a comprehensive user guide with detailed features and examples, see [User_Guide.md](User_Guide.md)**

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Logging In](#logging-in)
4. [Navigation & Analysis Modes](#navigation--analysis-modes)
5. [Uploading Your Data](#uploading-your-data)
6. [Asking Questions](#asking-questions)
7. [Interpreting Results](#interpreting-results)
8. [Downloading Visualisations & Data](#downloading-visualisations--data)
9. [Session Management](#session-management)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)
12. [Release Notes](#release-notes)
13. [Support](#support)

---

## System Requirements
* **Browser**: Latest Chrome, Firefox, or Edge.
* **Resolution**: 1366×768 or higher for best chart clarity.
* **Network**: Stable internet connection (>5 Mbps).

## Installation
You have two options:

### 1. Streamlit Cloud (no-install)
1. Visit the hosted URL provided by your admin.
2. Enter the login credentials.

### 2. Local Setup
```bash
# Clone repository
$ git clone https://github.com/Sher110106/HR-Agent.git
$ cd HR-Agent

# Install Python dependencies
$ pip install -r requirements.txt

# Export your API keys
$ export NVIDIA_API_KEY="<your-nvidia-key>"
$ export AZURE_API_KEY="<your-azure-key>"
$ export AZURE_ENDPOINT="<your-azure-endpoint>"

# Launch app
$ streamlit run streamlit_app.py
```

## Logging In
Enter the following credentials when prompted:
- **Username**: `Plaksha-HR`
- **Password**: `AgentHR1`

> **Tip:** You can change these defaults in `streamlit_app.py`.

## Navigation & Analysis Modes

The application now provides **five specialized analysis pages** accessible through the sidebar navigation:

### 📊 CSV Analysis
- **Purpose**: Traditional CSV file processing with enhanced plot quality
- **Best for**: Simple HR datasets in CSV format
- **Features**: Professional charts, dual-output contract, comprehensive exports

### 📈 Excel Analysis  
- **Purpose**: Multi-sheet Excel support with intelligent sheet selection
- **Best for**: Complex Excel files with multiple sheets
- **Features**: 
  - Intelligent sheet selection based on your queries
  - Column indexing for cross-sheet analysis
  - Sheet cataloging and semantic layer creation
  - Performance optimization for large files

### 🧠 Smart Analysis
- **Purpose**: PandasAI-powered enhanced reasoning and natural language exploration
- **Best for**: Advanced data exploration and complex queries
- **Features**:
  - Enhanced AI reasoning capabilities
  - More sophisticated query understanding
  - Multi-modal output (charts, tables, explanations)
  - Azure OpenAI integration for enterprise-grade AI

### ⚙️ System Prompt Manager
- **Purpose**: Dynamic prompt customization for different analysis styles
- **Best for**: Customizing AI behavior for specific analysis needs
- **Features**:
  - Context-aware prompts that adapt to data type
  - Template library for common scenarios
  - Real-time prompt switching

### 🔧 Monitoring Dashboard
- **Purpose**: Real-time system health, performance metrics, and cache management
- **Best for**: System administrators and power users
- **Features**:
  - Real-time health monitoring
  - Performance metrics and API status
  - Cache management and cleanup
  - Export detailed health reports

## Uploading Your Data

### CSV Files
1. Navigate to **"CSV Analysis"** in the sidebar.
2. Click **"Upload CSV"** in the file uploader.
3. Drag-and-drop or browse to your HR dataset.
4. Verify that the preview table looks correct.

### Excel Files
1. Navigate to **"Excel Analysis"** in the sidebar.
2. Click **"Upload Excel"** in the file uploader.
3. Select your Excel file with multiple sheets.
4. The system will automatically:
   - Catalog all sheets with descriptions
   - Create a column index for cross-sheet analysis
   - Present sheet selection options

### Supported Formats
- **CSV**: UTF-8 encoded files
- **Excel**: `.xlsx` and `.xls` files with multiple sheets

## Asking Questions
After uploading, type a natural language query into the **"Ask a question..."** box. Examples:

### For CSV Analysis:
* "Show employee distribution by department"
* "Plot salary vs experience correlation"
* "Average tenure by location"

### For Excel Analysis:
* "Compare sales performance across all sheets"
* "Show employee data from the 'HR Data' sheet"
* "Analyze trends in the 'Monthly Reports' sheet"

### For Smart Analysis:
* "What insights can you find about employee retention?"
* "Analyze the relationship between performance and compensation"
* "Create a comprehensive report on workforce diversity"

Click **Enter** or the **Send** button. The system will:
1. Think (spinner shown).
2. Display the generated chart + data table.
3. Stream an explanation of the findings.

> **How it works:**
> - The agent classifies your query and decides if a chart is needed.
> - If so, it generates both a professional-quality plot **and** the underlying data table (dual-output contract).
> - All results are paired with a business-friendly explanation.
> - For Excel files, the system intelligently selects relevant sheets based on your query.

## Interpreting Results
Each response includes:
1. **Professional-style chart** (high DPI, accessibility-friendly colors).
2. **Data table** with the exact numbers plotted.
3. **Reasoning panel** describing what the AI did and what it means for your business.
4. **Download buttons** for PNG (chart), CSV (data), TXT (explanation), and DOCX (formatted report).

> **Note:** The system always returns both chart and data for visual queries, ensuring transparency and easy export.

## Downloading Visualisations & Data
Use the buttons below the chart:
* **"Download PNG"** – high-resolution image.
* **"Download CSV"** – underlying data table.
* **"Download TXT"** – business explanation.
* **"Download DOCX"** – formatted Word document with analysis text and data tables.

> **New Feature:** All text and data downloads now include DOCX format options for professional document creation.

## Session Management
- Sessions auto-expire after 30 minutes of inactivity for security.
- Reloading the page will start a fresh session.
- All data is processed in-memory and never written to disk.
- Each analysis mode maintains its own session state.

## Troubleshooting
| Symptom | Possible Cause | Fix |
|---------|----------------|-----|
| "Invalid API key" error | Key missing/typo | Re-export API keys and restart app |
| Blank chart | No data uploaded | Upload a CSV/Excel first |
| Slow responses | Large dataset | Filter data prior to upload |
| "LLM timeout" | Upstream model busy or circuit breaker open | Retry after a minute |
| "Download button missing" | Non-visual query | Only chart queries provide PNG/CSV export |
| "Health warning" | System resource issue | Check health dashboard or contact admin |
| "Sheet not found" | Excel sheet name mismatch | Check sheet names in the catalog |
| "Memory error" | Large Excel file | Use smaller file or contact admin |

## FAQ
**Q:** *What file types are supported?*  
**A:** CSV (UTF-8) and Excel (.xlsx, .xls) files. Excel files can have multiple sheets.

**Q:** *Is my data stored?*  
**A:** All processing happens in-memory; nothing is written to disk.

**Q:** *How accurate are the insights?*  
**A:** The system performs standard statistical analysis; review results before making decisions.

**Q:** *What if I get an error?*  
**A:** The agent will display a friendly error message and suggest fixes. Most common issues are auto-retried or explained.

**Q:** *How do I download my results?*  
**A:** Use the download buttons below each chart for PNG, CSV, TXT, and DOCX exports. DOCX files are formatted Word documents perfect for reports and presentations.

**Q:** *How is my session secured?*  
**A:** Sessions are password-protected and auto-expire after inactivity.

**Q:** *What is health monitoring?*  
**A:** The app checks system, API, and resource health in the background. Admins can view detailed health reports.

**Q:** *Is there caching?*  
**A:** Yes, repeated queries are cached for faster results. Caches are cleared on restart or by admin.

**Q:** *What's the difference between analysis modes?*  
**A:** CSV Analysis is for simple CSV files, Excel Analysis handles multi-sheet Excel files with intelligent sheet selection, and Smart Analysis provides enhanced AI reasoning with PandasAI.

**Q:** *How does Excel sheet selection work?*  
**A:** The system analyzes your query and automatically selects the most relevant sheets based on content and column names.

**Q:** *Can I customize the AI behavior?*  
**A:** Yes, use the System Prompt Manager to select different analysis styles and customize AI behavior.

## Release Notes
See [Technical Manual](Technical.md#release-notes) for developer-oriented changelog. User-facing highlights:
* **v0.5** – Multi-page interface, Excel multi-sheet support, Smart Analysis with PandasAI, advanced monitoring, and system prompt management.
* **v0.4** – Health monitoring, advanced caching, metrics, and improved error handling.
* **v0.3** – Dual-output visualisations, improved styling.
* **v0.2** – Added reasoning panel.
* **v0.1** – Initial release.

## Support
Encountered an issue?  
* **Email**: support@example.com  
* **GitHub**: [Open an issue](https://github.com/Sher110106/HR-Agent/issues)  

---
> Documentation written following best-practice guidelines for user manuals [[source](https://document360.com/blog/technical-manual/)].
