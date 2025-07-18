# 📊 How Does the Agent Return **Both** Chart *and* Text?

The Streamlit-based HR Data-Analysis Agent is architected to always deliver **dual-modal answers** – a professional-quality chart **and** a concise narrative explanation – whenever a user's natural-language question implies that a visualisation is helpful.

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
5. **UI Rendering** – In `streamlit_app.py` the app:
   ```python
   fig, data_df = result_obj              # dual output detected
   st.session_state.plots.append(fig)     # store for reuse / download
   st.pyplot(fig)
   st.markdown(reasoning_txt)             # text insight
   # Download buttons for PNG, CSV, TXT
   ```
   The data table is shown in an *expander* and all three artefacts (plot PNG, CSV, analysis TXT) get instant **download buttons**.

> **Bottom line:** The mandatory tuple contract plus an extra LLM pass for reasoning guarantees that every visual answer is paired with an explanatory narrative – zero additional work needed from the user.

---

# 🛠️ End-to-End Pipeline Explained

| Stage | Key Component(s) | Purpose |
|-------|------------------|---------|
| **1. Ingestion** | `st.file_uploader` → `pd.read_csv` | Securely load the HR CSV. Basic type inference + encoding detection (`chardet`). |
| **2. Dataset Insight** | `DataInsightAgent` | Generate a quick "first look" summary (row/column counts, missing values, sample analysis questions). |
| **3. Conversation Memory** | `ConversationMemoryTool` | Supplies the last 4 chat turns to every subsequent LLM call for context preservation. |
| **4. Intent Detection** | `QueryUnderstandingTool` | Fast Boolean → does the request need a visualisation? |
| **5. Code Generation** | (`PlotCodeGeneratorTool` *or* `CodeWritingTool`) inside `CodeGenerationAgent` | Drafts fully-formed, *executable* pandas/matplotlib code. Special prompts enforce best-practice styling, high-DPI, value labels, etc. |
| **6. Validation & Retry** | `validate_pandas_code`, `perform_with_retries` | AST lint to block disallowed imports or unsafe ops before execution. Auto-retry logic patches common pandas syntax errors. |
| **7. Execution** | `ExecutionAgent` | Runs the code, gracefully captures stdout/stderr, and returns either a tuple `(fig, data_df)` or plain data. |
| **8. Reasoning** | `ReasoningAgent` & `ReasoningCurator` | Produces a natural-language summary, surfacing key insights while hiding internal "thinking". |
| **9. Streaming UI** | Streamlit main loop | Renders chat bubbles, the figure, expandable data table, code (in an expander), and download buttons. Maintains per-session `plots` & `plot_data` arrays for history. |
| **10. Caching & Metrics** | `code_cache`, `metrics_collector` | Caches repeated queries and tracks API, code, and error events for performance. |
| **11. Health Monitoring** | `health_monitor` | Background checks for system, API, and resource health. |

## Additional Engineering Highlights

* **Dual-Output Contract** – Enforced in the prompt & checked at runtime, enabling one-click export and further analysis.
* **Professional Styling** – `utils/plot_helpers.py` provides helpers (`add_value_labels`, `apply_professional_styling`, …) so every chart is presentation-ready.
* **Robust Error Recovery** – Automatic second-pass code regeneration slashes typical pandas syntax errors (>80% success in testing).
* **Intelligent Caching** – In-memory and persistent cache for code, results, and analysis.
* **Circuit Breaker & Retry** – Resilient API and code execution with fail-fast and recovery.
* **System Prompt Management** – Customizable LLM prompt templates for agent behavior.
* **Health Monitoring** – Background checks for system, API, and resource health.
* **Metrics Collection** – Tracks API, code, and error events for performance and debugging.

---

### 📌 How to Ask for Chart + Text Yourself
1. Upload a CSV.
2. Ask *any* question that *implies* a visual ("trend", "distribution", "compare by", "show over time"…).
3. The agent returns the chart, an explanatory paragraph, plus expandable source data – all downloadable.

That's it! The system's architecture does the heavy lifting so you can stay focused on insights, not tooling.
