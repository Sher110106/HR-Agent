# 📊 How Does the Agent Return **Both** Chart *and* Text?

The Streamlit-based HR Data-Analysis Agent is architected to always deliver **dual-modal answers** – a professional-quality chart **and** a concise narrative explanation – whenever a user's natural-language question implies that a visualisation is helpful. The system now supports **multiple analysis modes** through a sophisticated multi-page interface.

> **📖 For comprehensive user documentation and practical usage examples, see [User_Guide.md](User_Guide.md)**

## 🎯 Multi-Page Analysis Interface

The application now provides **five specialized analysis pages**:

1. **📊 CSV Analysis** – Traditional CSV file processing with enhanced plot quality
2. **📈 Excel Analysis** – Multi-sheet Excel support with intelligent sheet selection
3. **🧠 Smart Analysis** – PandasAI-powered enhanced reasoning and natural language exploration
4. **⚙️ System Prompt Manager** – Dynamic prompt customization for different analysis styles
5. **🔧 Monitoring Dashboard** – Real-time system health, performance metrics, and cache management

## 🔄 Enhanced Dual-Output Pipeline

The system maintains the **mandatory tuple contract** while adding new capabilities:

1. **Query Classification** – `QueryUnderstandingTool` checks the latest user prompt (and recent chat turns) with the LLM to decide whether a plot is required. It returns a Boolean flag `should_plot`.
2. **Code Generation** –
   • If `should_plot` is **True**, `PlotCodeGeneratorTool` is invoked. The LLM receives:
     – Column descriptions and context from memory agents.
     – Clear **template instructions** to build the figure **and** return it **together** with the aggregated data in a tuple:
     ```python
     result = (fig, data_df)  # <- CRITICAL!
     ```
   • If `should_plot` is **False**, `CodeWritingTool` generates pure pandas logic and returns a DataFrame/Series/scalar.
3. **Execution** – `ExecutionAgent` runs the generated Python safely inside an isolated namespace. It detects the **tuple** shape `(fig, data_df)` and hands it back to the UI layer. Code is validated for common pandas errors and auto-retried if needed.
4. **Narrative Synthesis** – `ReasoningAgent` feeds the raw output back into the LLM to curate a short, business-friendly explanation of what the chart or data means. Error messages are also explained and actionable tips are provided.
5. **UI Rendering** – In the respective page controllers, the app:
   ```python
   fig, data_df = result_obj              # dual output detected
   st.session_state.plots.append(fig)     # store for reuse / download
   st.pyplot(fig)
   st.markdown(reasoning_txt)             # text insight
   # Download buttons for PNG, CSV, TXT, DOCX
   ```
   The data table is shown in an *expander* and all four artefacts (plot PNG, CSV, analysis TXT, DOCX) get instant **download buttons**.

> **Bottom line:** The mandatory tuple contract plus an extra LLM pass for reasoning guarantees that every visual answer is paired with an explanatory narrative – zero additional work needed from the user.

---

# 🛠️ End-to-End Pipeline Explained

| Stage | Key Component(s) | Purpose |
|-------|------------------|---------|
| **1. Navigation** | `utils/navigation.py` → Multi-page interface | Routes users to specialized analysis pages based on data type and requirements. |
| **2. Ingestion** | `st.file_uploader` → Format-specific processors | Securely load CSV/Excel with intelligent format detection and validation. |
| **3. Dataset Insight** | `DataInsightAgent` | Generate a quick "first look" summary (row/column counts, missing values, sample analysis questions). |
| **4. Sheet Selection** | `SheetSelectionAgent` (Excel only) | Intelligently select relevant Excel sheets based on user queries and content analysis. |
| **5. Column Indexing** | `ColumnIndexerAgent` (Excel only) | Create semantic layer for Excel column mapping and cross-sheet analysis. |
| **6. Conversation Memory** | `ConversationMemoryTool` | Supplies the last 4 chat turns to every subsequent LLM call for context preservation. |
| **7. Intent Detection** | `QueryUnderstandingTool` | Fast Boolean → does the request need a visualisation? |
| **8. Code Generation** | (`PlotCodeGeneratorTool` *or* `CodeWritingTool`) inside `CodeGenerationAgent` | Drafts fully-formed, *executable* pandas/matplotlib code. Special prompts enforce best-practice styling, high-DPI, value labels, etc. |
| **9. Validation & Retry** | `validate_pandas_code`, `perform_with_retries` | AST lint to block disallowed imports or unsafe ops before execution. Auto-retry logic patches common pandas syntax errors. |
| **10. Execution** | `ExecutionAgent` | Runs the code, gracefully captures stdout/stderr, and returns either a tuple `(fig, data_df)` or plain data. |
| **11. Reasoning** | `ReasoningAgent` & `ReasoningCurator` | Produces a natural-language summary, surfacing key insights while hiding internal "thinking". |
| **12. Smart Analysis** | `PandasAI` integration (optional) | Enhanced reasoning and natural language data exploration with advanced AI capabilities. |
| **13. Streaming UI** | Streamlit page controllers | Renders chat bubbles, the figure, expandable data table, code (in an expander), and download buttons. Maintains per-session `plots` & `plot_data` arrays for history. |
| **14. Caching & Metrics** | `code_cache`, `metrics_collector` | Caches repeated queries and tracks API, code, and error events for performance. |
| **15. Health Monitoring** | `health_monitor` | Background checks for system, API, and resource health with real-time dashboard. |

## 🆕 New Features & Capabilities

### **Excel Multi-Sheet Support**
- **Intelligent Sheet Selection** – Automatically identifies relevant sheets based on user queries
- **Column Indexing** – Creates semantic layer for cross-sheet analysis
- **Sheet Cataloging** – Comprehensive catalog of all sheets with descriptions
- **Performance Optimization** – Memory-efficient processing of large Excel files

### **Smart Analysis with PandasAI**
- **Enhanced Reasoning** – Advanced AI-powered data exploration
- **Natural Language Queries** – More sophisticated query understanding
- **Multi-Modal Output** – Charts, tables, and explanations in one response
- **Azure OpenAI Integration** – Enterprise-grade AI model support

### **System Prompt Management**
- **Dynamic Prompts** – Customizable AI behavior for different analysis styles
- **Context-Aware** – Prompts adapt based on data type and user preferences
- **Template Library** – Pre-built prompts for common analysis scenarios

### **Advanced Monitoring**
- **Real-Time Health** – System health, API status, and resource monitoring
- **Performance Metrics** – API calls, response times, error rates, and success rates
- **Cache Management** – Intelligent caching with cleanup and optimization
- **Export Reports** – Detailed health and performance reports

## Additional Engineering Highlights

* **Multi-Page Architecture** – Modular design with specialized pages for different analysis types
* **Dual-Output Contract** – Enforced in the prompt & checked at runtime, enabling one-click export and further analysis
* **Professional Styling** – `utils/plot_helpers.py` provides helpers (`add_value_labels`, `apply_professional_styling`, …) so every chart is presentation-ready
* **Robust Error Recovery** – Automatic second-pass code regeneration slashes typical pandas syntax errors (>80% success in testing)
* **Intelligent Caching** – In-memory and persistent cache for code, results, and analysis
* **Circuit Breaker & Retry** – Resilient API and code execution with fail-fast and recovery
* **System Prompt Management** – Customizable LLM prompt templates for agent behavior
* **Health Monitoring** – Background checks for system, API, and resource health with real-time dashboard
* **Metrics Collection** – Tracks API, code, and error events for performance and debugging
* **Excel Performance** – Optimized memory usage and processing for large Excel files
* **Multi-Format Export** – PNG, CSV, TXT, and DOCX export options for all analysis results

---

### 📌 How to Ask for Chart + Text Yourself
1. **Choose your analysis mode** – CSV Analysis, Excel Analysis, or Smart Analysis
2. Upload your data (CSV or Excel).
3. Ask *any* question that *implies* a visual ("trend", "distribution", "compare by", "show over time"…).
4. The agent returns the chart, an explanatory paragraph, plus expandable source data – all downloadable in multiple formats.

That's it! The system's architecture does the heavy lifting so you can stay focused on insights, not tooling.
