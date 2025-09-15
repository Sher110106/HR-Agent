So what I want to do is to create a system that can take a user's query and then use the data in the excel file to answer the query.

The system should be able to handle multiple sheets in the excel file and should be able to handle complex queries.

The system should be able to handle complex queries that involve multiple sheets and should be able to handle complex queries that involve multiple sheets.

This sysyem should not interfere with the system that is already built for csv file handling.

Here is the complete, consolidated, and actionable plan for the Intelligent Multi-Sheet Excel Analysis Agent. It incorporates all previous discussions and your most beneficial recommendations into a single, grounded blueprint.

-----

### **Final Project Plan: Intelligent Multi-Sheet Excel Analysis Agent**

**Last Updated:** July 19, 2025
**Status:** Final Blueprint for Implementation

This document outlines the definitive architecture and implementation strategy for upgrading the HR agent into a multi-sheet analysis platform. It is designed to be a practical guide for development, ensuring the final product is intelligent, resilient, and user-centric.

#### **1. Core Philosophy: The Interactive Mandate**

The system's fundamental design principle is that it should **never guess when it can ask**. Traditional software fails silently or with cryptic errors. This agent will instead engage in a dialogue.

The `SheetSelectionAgent` will be explicitly instructed in its system prompt:

> "Your primary goal is to create a valid execution plan. If the user's request is ambiguous, if columns could be interpreted in multiple ways, or if a clear path to combine data is missing, **your first action is to formulate a clear, multiple-choice question for the user.** Do not proceed with a low-confidence plan. Present the options and await user input."

This approach delegates complex, context-dependent decisions to the user, making the system safer, more accurate, and more transparent.

#### **2. End-to-End Workflow**

The refined workflow integrates a hybrid planning model to balance deterministic speed with LLM intelligence.

```mermaid
graph TD
    A[User Uploads .xlsx] --> B(SheetCatalogAgent);
    B --> C{Reads all sheets into DataFrames};
    C --> D(ColumnIndexerAgent);
    D --> E{Builds column index &<br>accepts optional user tags};
    E --> F[User Submits NL Query];
    F --> G(SheetSelectionAgent - Stage 1: Heuristics);
    G --> H{Pre-filters relevant sheets<br>based on query keywords};
    H --> I(SheetSelectionAgent - Stage 2: LLM Planner);
    I --> J{LLM receives pre-filtered sheets<br>and devises a `SheetPlan`.<br>Asks user if ambiguous.};
    J --> K(CodeGenerationAgent);
    K --> L{Uses `SheetPlan` to generate<br>pandas code with a preamble};
    L --> M(ExecutionAgent);
    M --> N{Executes code in sandbox};
    N --> O[Display Results in UI<br>(Figure, Table, or Clarification Question)];

    subgraph "Phase 1: Ingestion & Semantic Indexing"
        B; C; D; E;
    end

    subgraph "Phase 2: Hybrid Query Planning"
        G; H; I; J;
    end

    subgraph "Phase 3: Code Generation & Execution"
        K; L; M; N;
    end
```

#### **3. Architectural Components & Implementation**

##### **3.1. Data Structures**

  * `session_state["sheet_catalog"]`: `dict[str, pd.DataFrame]` mapping sanitized sheet names to their DataFrame.
  * `session_state["column_index"]`: `dict[str, list[ColumnRef]]` mapping lowercase column names to their locations.
  * `session_state["semantic_layer"]`: `dict` storing user-defined metadata, e.g., `{'primary_join_key': 'employee_id'}`.

##### **3.2. Agent Breakdown**

1.  **SheetCatalogAgent**

      * **Action**: Reads the `.xlsx` file using `pd.read_excel(sheet_name=None)`. Sanitizes sheet names for variable safety.
      * **Result**: Populates `session_state["sheet_catalog"]`.

2.  **ColumnIndexerAgent**

      * **Action**: Iterates through all DataFrames to build the `column_index`.
      * **Enhanced Capability (Semantic Layer)**: After indexing, the UI will present key columns (e.g., those with unique values like 'employee\_id' or common names across sheets) and allow the user to **tag them**. For example, a user can tag `employee_id` as the **"Primary Join Key"**. This metadata is stored in `session_state["semantic_layer"]`.

3.  **SheetSelectionAgent (Hybrid Model)**

      * **Stage 1: Heuristic Pre-filtering**:

          * **Action**: Extracts nouns and keywords from the user's query. It performs a fast, deterministic search against the `column_index` and sheet names.
          * **Example**: For "Compare salaries of active vs attrited employees," it identifies keywords `salary`, `active`, `attrited`. It finds that `Active_Employees` and `Attrited_Employees` sheets contain a `salary` column and their names match the keywords.
          * **Result**: A small, highly relevant list of candidate sheets is passed to Stage 2.

      * **Stage 2: LLM Planner**:

          * **Action**: Receives the pre-filtered sheets and the user's query. Its prompt includes the semantic tags and the "Interactive Mandate."
          * **Result**: It outputs a structured `SheetPlan` (defining aliases, join/union strategies) or, if ambiguity remains, a structured question for the UI to render.

4.  **CodeGenerationAgent**

      * **Action**: Takes the final, validated `SheetPlan`. It constructs a prompt preamble that sets up the context for the code-generating LLM.
      * **Preamble Example (for a union plan)**:
        ```
        The following DataFrames are pre-loaded: df_active, df_attrited.
        Your first task is to combine them. Add a new column 'employment_status' to each, label them 'Active' and 'Attrited' respectively, and then concatenate them.
        After combining, proceed with the analysis of 'salary'.
        ```
      * **Result**: Clean, executable pandas code.

5.  **ExecutionAgent**

      * **Action**: Receives the code and a `locals` dictionary containing the required DataFrames (`{'df_active': df1, ...}`). Executes the code in a secure sandbox.
      * **Result**: A figure and/or a DataFrame to be displayed in the UI.

#### **4. UI/UX: The Disambiguation Interface**

When the LLM decides to ask a question, the system will not just print text. It will render a rich, interactive modal.

**Scenario**: The LLM is unsure whether to join or stack two sheets.

**The Modal will display:**

  * **The Question**: "How should I combine the `Active_Employees` and `Attrited_Employees` sheets to answer your query?"
  * **Visual Context**: A side-by-side preview showing the first few rows and key columns of each DataFrame. The potential join key (`employee_id`) would be highlighted.
  * **Clear Options**:
      * **Button 1**: `[ Stack Vertically (Union) ]` with a sub-text: "Best for comparing groups (e.g., active vs. attrited)."
      * **Button 2**: `[ Join by 'employee_id' ]` with a sub-text: "Best for analyzing individual employees across both datasets."

This transforms a moment of ambiguity into an empowering, context-rich choice for the user.

#### **5. Mini-Roadmap (4 Sprints)**

  * **Sprint 1: Foundational Ingestion**

      * Implement `SheetCatalogAgent` and UI file uploader.
      * Implement `ColumnIndexerAgent` and the backend logic for the **Semantic Layer**.
      * **Goal**: A user can upload an Excel file and tag key columns.

  * **Sprint 2: Intelligent Planning & Interaction**

      * Build the `SheetSelectionAgent`'s **Heuristic Pre-filter**.
      * Implement the **LLM Planner** with the "Interactive Mandate" prompt.
      * Develop the **Disambiguation UI Modal**.
      * **Goal**: The system can correctly plan for a query or ask a clear, interactive question if it's unsure.

  * **Sprint 3: Code Generation & Execution**

      * Enhance `CodeGenerationAgent` to use the `SheetPlan` and construct the prompt preamble.
      * Integrate the full pipeline from `SheetPlan` to `ExecutionAgent`.
      * **Goal**: The system can successfully execute a plan for a single-sheet or multi-sheet query.

  * **Sprint 4: Resilience & Polish**

      * Implement caching for `SheetPlan` objects.
      * Add comprehensive error handling for file parsing, code execution, etc.
      * Write end-to-end integration tests for join, union, and single-sheet query scenarios.
      * **Goal**: A robust, tested, and performant feature ready for release.