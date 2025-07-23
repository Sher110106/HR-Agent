"""Enhanced Code Generation Agent for multi-sheet Excel analysis.

This module provides specialized code generation capabilities for handling
multi-sheet Excel data with join, union, and single-sheet operations.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from agents.excel_agents import SheetPlan, ColumnIndexerAgent
from agents.memory import SystemPromptMemoryAgent
from app_core.api import make_llm_call

logger = logging.getLogger(__name__)


class ExcelCodeGenerationAgent:
    """Enhanced code generation agent for multi-sheet Excel analysis."""
    
    def __init__(self, column_indexer_agent: ColumnIndexerAgent):
        self.column_indexer_agent = column_indexer_agent
        self.sheet_catalog = column_indexer_agent.sheet_catalog
        
    def create_preamble_from_plan(self, sheet_plan: SheetPlan) -> str:
        """Create a preamble that sets up the context for code generation."""
        logger.info(f"📝 Creating preamble for sheet plan: {sheet_plan}")
        
        preamble = "# Multi-sheet Excel Analysis\n"
        preamble += "# Available DataFrames:\n"
        
        # List available DataFrames
        if sheet_plan.primary_sheets:
            # Use the sheets from the plan
            for sheet_name in sheet_plan.primary_sheets:
                if sheet_name in self.sheet_catalog:
                    df = self.sheet_catalog[sheet_name]
                    preamble += f"# - {sheet_name}: {len(df)} rows, {len(df.columns)} columns\n"
                    preamble += f"#   Columns: {', '.join(df.columns)}\n"
        else:
            # Fallback: list all available sheets when plan is empty
            logger.warning("⚠️ Sheet plan has no primary sheets, listing all available sheets")
            for sheet_name, df in self.sheet_catalog.items():
                preamble += f"# - {sheet_name}: {len(df)} rows, {len(df.columns)} columns\n"
                preamble += f"#   Columns: {', '.join(df.columns)}\n"
        
        # Add sheet aliases if provided
        if sheet_plan.sheet_aliases:
            preamble += f"\n# Sheet aliases: {sheet_plan.sheet_aliases}\n"
        
        # Add strategy-specific instructions
        if sheet_plan.join_strategy == 'join':
            preamble += self._create_join_preamble(sheet_plan)
        elif sheet_plan.join_strategy == 'union':
            preamble += self._create_union_preamble(sheet_plan)
        elif sheet_plan.join_strategy == 'single_sheet':
            preamble += self._create_single_sheet_preamble(sheet_plan)
        else:
            # Default strategy when none is specified
            preamble += self._create_single_sheet_preamble(sheet_plan)
        
        return preamble
    
    def _create_join_preamble(self, sheet_plan: SheetPlan) -> str:
        """Create preamble for join operations."""
        preamble = "\n# JOIN STRATEGY: Combining data from multiple sheets\n"
        
        if sheet_plan.join_keys:
            join_key = sheet_plan.join_keys[0]  # Use first join key
            preamble += f"# Join key: {join_key}\n"
            
            # Show which sheets have this join key
            refs = self.column_indexer_agent.get_column_refs(join_key)
            sheets_with_key = [ref.sheet_name for ref in refs]
            preamble += f"# Sheets with join key: {', '.join(sheets_with_key)}\n"
            
            # Create the join code
            if len(sheet_plan.primary_sheets) >= 2:
                sheet1, sheet2 = sheet_plan.primary_sheets[:2]
                preamble += f"\n# Combine the sheets:\n"
                preamble += f"combined_df = {sheet1}.merge({sheet2}, on='{join_key}', how='inner')\n"
                preamble += f"# Now use 'combined_df' for your analysis\n"
        
        return preamble
    
    def _create_union_preamble(self, sheet_plan: SheetPlan) -> str:
        """Create preamble for union operations."""
        preamble = "\n# UNION STRATEGY: Stacking data from multiple sheets\n"
        
        if len(sheet_plan.primary_sheets) >= 2:
            sheets = sheet_plan.primary_sheets
            preamble += f"# Stacking sheets: {', '.join(sheets)}\n"
            
            # Add status labels if provided
            if sheet_plan.additional_columns:
                preamble += f"# Adding status labels: {sheet_plan.additional_columns}\n"
                for sheet_name, label_value in sheet_plan.additional_columns.items():
                    if sheet_name in self.sheet_catalog:
                        # Extract the actual label from the value (e.g., "Status = Active" -> "Active")
                        if '=' in label_value:
                            label = label_value.split('=')[1].strip()
                        else:
                            label = label_value
                        # Use a safe column name
                        status_col = 'Status'
                        preamble += f"{sheet_name}['{status_col}'] = '{label}'\n"
            
            # Create the union code
            preamble += f"\n# Combine the sheets:\n"
            if len(sheets) == 2:
                preamble += f"combined_df = pd.concat([{sheets[0]}, {sheets[1]}], ignore_index=True, sort=False)\n"
            else:
                sheet_list = ", ".join(sheets)
                preamble += f"combined_df = pd.concat([{sheet_list}], ignore_index=True, sort=False)\n"
            preamble += f"# Now use 'combined_df' for your analysis\n"
        
        return preamble
    
    def _create_single_sheet_preamble(self, sheet_plan: SheetPlan) -> str:
        """Create preamble for single sheet operations."""
        if sheet_plan.primary_sheets:
            sheet_name = sheet_plan.primary_sheets[0]
            preamble = f"\n# SINGLE SHEET STRATEGY: Using {sheet_name}\n"
            preamble += f"# IMPORTANT: Use the DataFrame named '{sheet_name}' for your analysis\n"
            preamble += f"# DO NOT use any other DataFrame names like 'All_Employees' or 'employees'\n"
            preamble += f"# The correct DataFrame name is: {sheet_name}\n"
        else:
            # Suggest the most appropriate sheet based on available options
            available_sheets = list(self.sheet_catalog.keys())
            if available_sheets:
                # Try to find a sheet that might be most relevant
                suggested_sheet = None
                for sheet in available_sheets:
                    if 'active' in sheet.lower() or 'employee' in sheet.lower():
                        suggested_sheet = sheet
                        break
                
                if suggested_sheet:
                    preamble = f"\n# SINGLE SHEET STRATEGY: Suggested sheet {suggested_sheet}\n"
                    preamble += f"# IMPORTANT: Use the DataFrame named '{suggested_sheet}' for your analysis\n"
                    preamble += f"# DO NOT use any other DataFrame names\n"
                    preamble += f"# The correct DataFrame name is: {suggested_sheet}\n"
                else:
                    preamble = f"\n# SINGLE SHEET STRATEGY: Using {available_sheets[0]}\n"
                    preamble += f"# IMPORTANT: Use the DataFrame named '{available_sheets[0]}' for your analysis\n"
                    preamble += f"# DO NOT use any other DataFrame names\n"
                    preamble += f"# The correct DataFrame name is: {available_sheets[0]}\n"
            else:
                preamble = "\n# SINGLE SHEET STRATEGY: No sheets available\n"
                preamble += "# Please check your data source\n"
        
        return preamble
    
    def generate_code(self, query: str, sheet_plan: SheetPlan, should_plot: bool = False) -> str:
        """
        Generate pandas code for the given query and sheet plan.
        
        Args:
            query: User's analysis query
            sheet_plan: The sheet plan to follow
            should_plot: Whether to generate plotting code
            
        Returns:
            Generated pandas code as string
        """
        logger.info(f"🔧 Generating code for query: '{query}' with plan: {sheet_plan}")
        
        # Create preamble
        preamble = self.create_preamble_from_plan(sheet_plan)
        logger.debug(f"📋 Generated preamble:\n{preamble}")
        
        # Get system prompt context
        system_prompt_agent = SystemPromptMemoryAgent()
        
        # Create the main prompt
        if should_plot:
            prompt = self._create_plotting_prompt(query, preamble)
        else:
            prompt = self._create_analysis_prompt(query, preamble)
        
        # Apply system prompt if active
        system_prompt = system_prompt_agent.apply_system_prompt(prompt)
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}"}
            ]
            
            logger.info("📤 Sending code generation request to LLM...")
            response = make_llm_call(
                messages=messages,
                model="gpt-4.1",
                temperature=0.1,
                max_tokens=2000
            )
            
            code = response.choices[0].message.content.strip()
            logger.info(f"📥 Generated code: {len(code)} characters")
            logger.debug(f"💻 Raw generated code:\n{code}")
            
            # Extract code block if wrapped in markdown
            if "```python" in code:
                start = code.find("```python") + 9
                end = code.find("```", start)
                if end != -1:
                    code = code[start:end].strip()
            
            # Validate that the code assigns a result
            if "result =" not in code and "result=" not in code:
                logger.warning("⚠️ Generated code does not assign 'result' variable - adding fallback")
                if should_plot:
                    code += "\n\n# Ensure result is assigned for plotting\nif 'result' not in locals():\n    result = (fig, combined_df if 'combined_df' in locals() else Active_Employees)"
                else:
                    code += "\n\n# Ensure result is assigned for analysis\nif 'result' not in locals():\n    result = combined_df.describe() if 'combined_df' in locals() else 'Analysis completed but no specific result was returned'"
            
            # Validate that the code uses the correct DataFrame names
            expected_dfs = sheet_plan.primary_sheets
            missing_dfs = [df for df in expected_dfs if df not in code]
            if missing_dfs:
                logger.warning(f"⚠️ Generated code missing expected DataFrames: {missing_dfs}")
                logger.debug(f"Expected DataFrames: {expected_dfs}")
                logger.debug(f"Code content preview: {code[:500]}...")
            
            return code
            
        except Exception as e:
            logger.error(f"❌ Error in code generation: {e}")
            # Return fallback code
            return self._create_fallback_code(query, sheet_plan, should_plot)
    
    def _create_analysis_prompt(self, query: str, preamble: str) -> str:
        """Create prompt for data analysis (non-plotting) code."""
        return f"""
You are an expert data analyst specializing in pandas operations. Generate clean, efficient pandas code to answer the user's query.

{preamble}

USER QUERY: "{query}"

CRITICAL REQUIREMENTS:
- Use ONLY the DataFrame names specified in the preamble above
- Do NOT invent or use DataFrame names that are not listed in the preamble
- If the preamble specifies a specific DataFrame name, use that exact name
- Pay close attention to the DataFrame names in the "Available DataFrames" section

REQUIREMENTS:
- Use the available DataFrames as specified in the preamble
- Write clean, readable pandas code
- Include proper error handling and data validation
- **CRITICAL**: Always assign the final result to a variable called 'result'
- Use only pandas, numpy, and standard Python libraries
- Add comments to explain complex operations

PANDAS BEST PRACTICES:
- Use observed=True in groupby operations: df.groupby('col', observed=True)
- Handle missing values with .dropna() before operations
- Use .copy() to avoid SettingWithCopyWarning
- For concatenation with different column structures, use pd.concat([df1, df2], ignore_index=True, sort=False)

EXAMPLE OUTPUT FORMAT:
```python
# Your analysis code here
result = some_analysis_operation()  # CRITICAL: Always assign to 'result'
```

Focus on data manipulation, aggregation, filtering, and analysis operations.
"""
    
    def _create_plotting_prompt(self, query: str, preamble: str) -> str:
        """Create prompt for plotting code."""
        return f"""
You are an expert data analyst specializing in data visualization with pandas and matplotlib. Generate code to create a professional plot.

{preamble}

USER QUERY: "{query}"

REQUIREMENTS:
- Use the available DataFrames as specified in the preamble
- Create a professional, publication-ready visualization
- Use matplotlib and seaborn for plotting
- Return a tuple (fig, data_content) where:
  - fig is a matplotlib Figure with professional styling
  - data_content can be either:
    - A single DataFrame (data_df) used to create the plot
    - A dictionary of DataFrames if multiple datasets are involved (e.g., {{'summary': df1, 'details': df2}})
- **CRITICAL**: Always assign the final result to a variable called 'result'
- Use the helper functions available in the execution environment
- Add proper titles, labels, and legends
- Handle edge cases (empty data, missing values, etc.)

AVAILABLE HELPER FUNCTIONS:
- apply_professional_styling(ax, title, xlabel, ylabel)
- format_axis_labels(ax, x_rotation=45)
- get_professional_colors()
- safe_color_access(colors, index)
- create_category_palette(categories, palette_name='primary') # For seaborn category-specific palettes
- optimize_figure_size(ax)
- add_value_labels(ax, label_mode="minimal")
- create_clean_bar_chart(), create_clean_line_chart(), etc.

SEABORN PALETTE GUIDANCE:
- For category-specific colors: palette = create_category_palette(df['category_col'].unique())
- For general seaborn plots: palette = get_professional_colors()['colors'][:n_categories]
- Always slice colors to match the number of categories to avoid warnings
- When using seaborn with palette, assign the categorical column to 'hue' and set legend=False to avoid deprecation warnings

MATPLOTLIB BEST PRACTICES:
- Always use the 'ax' object for axis operations, NOT the 'fig' object
- For axis operations use: ax.get_xticklabels(), ax.set_xlabel(), ax.set_ylabel(), etc.
- For figure operations use: fig.tight_layout(), fig.savefig(), etc.
- Never call axis methods on the figure object

PANDAS BEST PRACTICES:
- Use observed=True in groupby operations: df.groupby('col', observed=True)
- Handle missing values with .dropna() before operations
- Use .copy() to avoid SettingWithCopyWarning
- For concatenation with different column structures, use pd.concat([df1, df2], ignore_index=True, sort=False)

EXAMPLE OUTPUT FORMATS:
```python
# For single dataset:
fig, ax = plt.subplots(figsize=(10, 6))
# ... plotting code ...
data_df = processed_data  # The data used for plotting
result = (fig, data_df)  # CRITICAL: Always assign to 'result'

# For multiple datasets:
fig, ax = plt.subplots(figsize=(10, 6))
# ... plotting code ...
data_dict = {{
    'summary': summary_df,
    'details': details_df
}}
result = (fig, data_dict)  # CRITICAL: Always assign to 'result'
```

Focus on creating clear, informative visualizations that answer the user's query.
"""
    
    def _create_fallback_code(self, query: str, sheet_plan: SheetPlan, should_plot: bool) -> str:
        """Create fallback code when LLM generation fails."""
        logger.warning("🔄 Using fallback code generation")
        
        if not sheet_plan.primary_sheets:
            return "# Error: No sheets available for analysis"
        
        # Handle union strategy
        if sheet_plan.join_strategy == 'union' and len(sheet_plan.primary_sheets) >= 2:
            sheets = sheet_plan.primary_sheets
            if should_plot:
                return f"""
# Fallback plotting code for union
import matplotlib.pyplot as plt

# Combine sheets
{sheets[0]}['Status'] = 'Active'
{sheets[1]}['Status'] = 'Inactive'
combined_df = pd.concat([{sheets[0]}, {sheets[1]}], ignore_index=True, sort=False)

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
combined_df.groupby('Status').size().plot(kind='bar', ax=ax)
plt.title('Employee Count by Status')
plt.xlabel('Status')
plt.ylabel('Count')
plt.tight_layout()

# Return both plot and data
data_dict = {{
    'summary': combined_df.groupby('Status').agg({{'Employee Count': 'count', 'Average Salary': 'mean'}}).round(2),
    'gender_distribution': combined_df.groupby('Status')['Gender'].value_counts().unstack(fill_value=0)
}}
result = (fig, data_dict)
"""
            else:
                return f"""
# Fallback analysis code for union
{sheets[0]}['Status'] = 'Active'
{sheets[1]}['Status'] = 'Inactive'
combined_df = pd.concat([{sheets[0]}, {sheets[1]}], ignore_index=True, sort=False)
result = combined_df.groupby('Status').agg(['count', 'mean']).round(2)
"""
        
        # Handle single sheet
        sheet_name = sheet_plan.primary_sheets[0]
        if should_plot:
            return f"""
# Fallback plotting code
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
{sheet_name}.plot(kind='bar', ax=ax)
plt.title('Analysis of {sheet_name}')
plt.xlabel('Index')
plt.ylabel('Values')
plt.tight_layout()

data_df = {sheet_name}
result = (fig, data_df)
"""
        else:
            return f"""
# Fallback analysis code
result = {sheet_name}.describe()
"""
    
    def validate_generated_code(self, code: str, sheet_plan: SheetPlan) -> Tuple[bool, List[str]]:
        """Validate generated code for safety and correctness."""
        errors = []
        
        # Check for dangerous operations
        dangerous_patterns = [
            'exec(', 'eval(', 'os.system(', 'subprocess.call(',
            'import os', 'import subprocess', 'import sys'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                errors.append(f"Dangerous operation detected: {pattern}")
        
        # Check if required DataFrames are referenced
        for sheet_name in sheet_plan.primary_sheets:
            if sheet_name not in code:
                errors.append(f"Required DataFrame '{sheet_name}' not used in code")
        
        # Check for result variable
        if 'result' not in code:
            errors.append("Code does not assign to 'result' variable")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def generate_code_with_retry(self, query: str, sheet_plan: SheetPlan, should_plot: bool, error_context: str) -> str:
        """Generate code with retry context for error recovery."""
        logger.info(f"🔄 Generating retry code with error context: {error_context[:100]}...")
        
        try:
            # Create preamble
            preamble = self.create_preamble_from_plan(sheet_plan)
            
            # Create retry-specific prompt
            if should_plot:
                prompt = self._create_plotting_prompt_with_retry(query, preamble, error_context)
            else:
                prompt = self._create_analysis_prompt_with_retry(query, preamble, error_context)
            
            # Get system prompt
            system_prompt_agent = SystemPromptMemoryAgent()
            system_prompt = system_prompt_agent.apply_system_prompt("")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            logger.info("📤 Sending retry code generation request to LLM...")
            response = make_llm_call(
                messages=messages,
                model="gpt-4.1",
                temperature=0.1,
                max_tokens=2000
            )
            
            code = response.choices[0].message.content.strip()
            logger.info(f"📥 Generated retry code: {len(code)} characters")
            
            # Extract code block if wrapped in markdown
            if "```python" in code:
                start = code.find("```python") + 9
                end = code.find("```", start)
                if end != -1:
                    code = code[start:end].strip()
            
            # Validate that the code assigns a result
            if "result =" not in code and "result=" not in code:
                logger.warning("⚠️ Retry code does not assign 'result' variable - adding fallback")
                if should_plot:
                    code += "\n\n# Ensure result is assigned for plotting\nif 'result' not in locals():\n    result = (fig, combined_df if 'combined_df' in locals() else Active_Employees)"
                else:
                    code += "\n\n# Ensure result is assigned for analysis\nif 'result' not in locals():\n    result = combined_df.describe() if 'combined_df' in locals() else 'Analysis completed but no specific result was returned'"
            
            return code
            
        except Exception as e:
            logger.error(f"❌ Error in retry code generation: {e}")
            # Return fallback code
            return self._create_fallback_code(query, sheet_plan, should_plot)
    
    def _create_analysis_prompt_with_retry(self, query: str, preamble: str, error_context: str) -> str:
        """Create prompt for data analysis with retry context."""
        return f"""
You are an expert data analyst specializing in pandas operations. The previous code generation failed with an error. Please generate corrected code.

{preamble}

USER QUERY: "{query}"

PREVIOUS ERROR:
{error_context}

CRITICAL REQUIREMENTS:
- Use ONLY the DataFrame names specified in the preamble above
- Do NOT invent or use DataFrame names that are not listed in the preamble
- If the preamble specifies a specific DataFrame name, use that exact name
- Pay close attention to the DataFrame names in the "Available DataFrames" section
- Fix the specific error mentioned above

REQUIREMENTS:
- Use the available DataFrames as specified in the preamble
- Write clean, readable pandas code
- Include proper error handling and data validation
- **CRITICAL**: Always assign the final result to a variable called 'result'
- Use only pandas, numpy, and standard Python libraries
- Add comments to explain complex operations

PANDAS BEST PRACTICES:
- Use observed=True in groupby operations: df.groupby('col', observed=True)
- Handle missing values with .dropna() before operations
- Use .copy() to avoid SettingWithCopyWarning
- For concatenation with different column structures, use pd.concat([df1, df2], ignore_index=True, sort=False)

EXAMPLE OUTPUT FORMAT:
```python
# Your analysis code here
result = some_analysis_operation()  # CRITICAL: Always assign to 'result'
```

Focus on data manipulation, aggregation, filtering, and analysis operations.
"""
    
    def _create_plotting_prompt_with_retry(self, query: str, preamble: str, error_context: str) -> str:
        """Create prompt for plotting code with retry context."""
        return f"""
You are an expert data analyst specializing in data visualization with pandas and matplotlib. The previous code generation failed with an error. Please generate corrected code.

{preamble}

USER QUERY: "{query}"

PREVIOUS ERROR:
{error_context}

REQUIREMENTS:
- Use the available DataFrames as specified in the preamble
- Create a professional, publication-ready visualization
- Use matplotlib and seaborn for plotting
- Return a tuple (fig, data_content) where:
  - fig is a matplotlib Figure with professional styling
  - data_content can be either:
    - A single DataFrame (data_df) used to create the plot
    - A dictionary of DataFrames if multiple datasets are involved (e.g., {{'summary': df1, 'details': df2}})
- **CRITICAL**: Always assign the final result to a variable called 'result'
- Use the helper functions available in the execution environment
- Add proper titles, labels, and legends
- Handle edge cases (empty data, missing values, etc.)
- Fix the specific error mentioned above

AVAILABLE HELPER FUNCTIONS:
- apply_professional_styling(ax, title, xlabel, ylabel)
- format_axis_labels(ax, x_rotation=45)
- get_professional_colors()
- safe_color_access(colors, index)
- create_category_palette(categories, palette_name='primary') # For seaborn category-specific palettes
- optimize_figure_size(ax)
- add_value_labels(ax, label_mode="minimal")
- create_clean_bar_chart(), create_clean_line_chart(), etc.

SEABORN PALETTE GUIDANCE:
- For category-specific colors: palette = create_category_palette(df['category_col'].unique())
- For general seaborn plots: palette = get_professional_colors()['colors'][:n_categories]
- Always slice colors to match the number of categories to avoid warnings
- When using seaborn with palette, assign the categorical column to 'hue' and set legend=False to avoid deprecation warnings

MATPLOTLIB BEST PRACTICES:
- Always use the 'ax' object for axis operations, NOT the 'fig' object
- For axis operations use: ax.get_xticklabels(), ax.set_xlabel(), ax.set_ylabel(), etc.
- For figure operations use: fig.tight_layout(), fig.savefig(), etc.
- Never call axis methods on the figure object

PANDAS BEST PRACTICES:
- Use observed=True in groupby operations: df.groupby('col', observed=True)
- Handle missing values with .dropna() before operations
- Use .copy() to avoid SettingWithCopyWarning
- For concatenation with different column structures, use pd.concat([df1, df2], ignore_index=True, sort=False)

EXAMPLE OUTPUT FORMATS:
```python
# For single dataset:
fig, ax = plt.subplots(figsize=(10, 6))
# ... plotting code ...
data_df = processed_data  # The data used for plotting
result = (fig, data_df)  # CRITICAL: Always assign to 'result'

# For multiple datasets:
fig, ax = plt.subplots(figsize=(10, 6))
# ... plotting code ...
data_dict = {{
    'summary': summary_df,
    'details': details_df
}}
result = (fig, data_dict)  # CRITICAL: Always assign to 'result'
```

Focus on creating clear, informative visualizations that answer the user's query.
""" 