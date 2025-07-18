# User Manual – Business Analysis HR Agent

> **Welcome!** This guide will help you go from zero to actionable HR insights in minutes.

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Logging In](#logging-in)
4. [Uploading Your Data](#uploading-your-data)
5. [Asking Questions](#asking-questions)
6. [Interpreting Results](#interpreting-results)
7. [Downloading Visualisations & Data](#downloading-visualisations--data)
8. [Session Management](#session-management)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)
11. [Release Notes](#release-notes)
12. [Support](#support)

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

# Export your NVIDIA API key
$ export NVIDIA_API_KEY="<your key>"

# Launch app
$ streamlit run streamlit_app.py
```

## Logging In
Enter the following credentials when prompted:
- **Username**: `Plaksha-HR`
- **Password**: `AgentHR1`

> **Tip:** You can change these defaults in `streamlit_app.py`.

## Uploading Your Data
1. Click **"Upload CSV"** in the sidebar.
2. Drag-and-drop or browse to your HR dataset.
3. Verify that the preview table looks correct.

Supported formats: **CSV** (UTF-8). Excel coming soon.

## Asking Questions
After uploading, type a natural language query into the **"Ask a question..."** box. Examples:
* "Show employee distribution by department"
* "Plot salary vs experience correlation"
* "Average tenure by location"

Click **Enter** or the **Send** button. The system will:
1. Think (spinner shown).
2. Display the generated chart + data table.
3. Stream an explanation of the findings.

> **How it works:**
> - The agent classifies your query and decides if a chart is needed.
> - If so, it generates both a professional-quality plot **and** the underlying data table (dual-output contract).
> - All results are paired with a business-friendly explanation.

## Interpreting Results
Each response includes:
1. **Professional-style chart** (high DPI, accessibility-friendly colors).
2. **Data table** with the exact numbers plotted.
3. **Reasoning panel** describing what the AI did and what it means for your business.
4. **Download buttons** for PNG (chart), CSV (data), and TXT (explanation).

> **Note:** The system always returns both chart and data for visual queries, ensuring transparency and easy export.

## Downloading Visualisations & Data
Use the buttons below the chart:
* **"Download PNG"** – high-resolution image.
* **"Download CSV"** – underlying data table.
* **"Download TXT"** – business explanation.

## Session Management
- Sessions auto-expire after 30 minutes of inactivity for security.
- Reloading the page will start a fresh session.
- All data is processed in-memory and never written to disk.

## Troubleshooting
| Symptom | Possible Cause | Fix |
|---------|----------------|-----|
| "Invalid API key" error | Key missing/typo | Re-export `NVIDIA_API_KEY` and restart app |
| Blank chart | No data uploaded | Upload a CSV first |
| Slow responses | Large dataset | Filter data prior to upload |
| "LLM timeout" | Upstream model busy or circuit breaker open | Retry after a minute |
| "Download button missing" | Non-visual query | Only chart queries provide PNG/CSV export |
| "Health warning" | System resource issue | Check health dashboard or contact admin |

## FAQ
**Q:** *What file types are supported?*  
**A:** Currently CSV only. Excel support is on the roadmap.

**Q:** *Is my data stored?*  
**A:** All processing happens in-memory; nothing is written to disk.

**Q:** *How accurate are the insights?*  
**A:** The system performs standard statistical analysis; review results before making decisions.

**Q:** *What if I get an error?*  
**A:** The agent will display a friendly error message and suggest fixes. Most common issues are auto-retried or explained.

**Q:** *How do I download my results?*  
**A:** Use the download buttons below each chart for PNG, CSV, and TXT exports.

**Q:** *How is my session secured?*  
**A:** Sessions are password-protected and auto-expire after inactivity.

**Q:** *What is health monitoring?*  
**A:** The app checks system, API, and resource health in the background. Admins can view detailed health reports.

**Q:** *Is there caching?*  
**A:** Yes, repeated queries are cached for faster results. Caches are cleared on restart or by admin.

## Release Notes
See [Technical Manual](Technical.md#release-notes) for developer-oriented changelog. User-facing highlights:
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
