"""Sheet Selection Agent for intelligent multi-sheet query planning.

This module implements the hybrid planning model with heuristic pre-filtering
and LLM-based sheet selection with interactive disambiguation capabilities.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass

from agents.excel_agents import ColumnIndexerAgent, SheetPlan, ColumnRef, SheetCatalogAgent
from app_core.api import make_llm_call

logger = logging.getLogger(__name__)


@dataclass
class DisambiguationQuestion:
    """Represents a question for user disambiguation."""
    question: str
    options: List[Dict[str, Any]]
    context: Dict[str, Any]
    question_type: str  # 'join_strategy', 'column_selection', 'sheet_combination'


class SheetSelectionAgent:
    """Agent responsible for intelligent sheet selection and planning."""
    
    def __init__(self, column_indexer_agent: ColumnIndexerAgent, sheet_catalog_agent: SheetCatalogAgent = None):
        self.column_indexer_agent = column_indexer_agent
        self.sheet_catalog = column_indexer_agent.sheet_catalog
        self.sheet_catalog_agent = sheet_catalog_agent  # Add reference to SheetCatalogAgent for original names
        
    def extract_query_keywords(self, query: str) -> Set[str]:
        """Extract relevant keywords from user query for heuristic pre-filtering."""
        logger.info(f"🔍 Extracting keywords from query: '{query}'")
        
        # Convert to lowercase and remove punctuation
        query_lower = re.sub(r'[^\w\s]', ' ', query.lower())
        
        # Extract words (excluding common stop words)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'show', 'me',
            'what', 'how', 'when', 'where', 'why', 'which', 'who', 'that', 'this', 'these',
            'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'i', 'you', 'he',
            'she', 'it', 'we', 'they', 'am', 'is', 'are', 'was', 'were', 'be', 'been'
        }
        
        words = set(query_lower.split())
        keywords = words - stop_words
        
        logger.info(f"🔍 Extracted keywords: {keywords}")
        return keywords
    
    def heuristic_pre_filter(self, query: str) -> List[str]:
        """
        Stage 1: Heuristic pre-filtering based on query keywords.
        
        Returns:
            List of candidate sheet names that are likely relevant to the query
        """
        logger.info("🎯 Stage 1: Heuristic pre-filtering")
        
        keywords = self.extract_query_keywords(query)
        candidate_sheets = set()
        
        # Get column index for searching
        column_index = self.column_indexer_agent.column_index
        
        # Search for keywords in column names
        for keyword in keywords:
            for col_name, refs in column_index.items():
                if keyword in col_name:
                    # Add all sheets that contain this column
                    for ref in refs:
                        candidate_sheets.add(ref.sheet_name)
        
        # Search for keywords in sheet names
        for sheet_name in self.sheet_catalog.keys():
            original_name = None
            if self.sheet_catalog_agent:
                original_name = self.sheet_catalog_agent.get_original_name(sheet_name)
            if original_name:
                for keyword in keywords:
                    if keyword in original_name.lower():
                        candidate_sheets.add(sheet_name)
        
        # If no candidates found, include all sheets
        if not candidate_sheets:
            logger.info("⚠️ No specific candidates found, including all sheets")
            candidate_sheets = set(self.sheet_catalog.keys())
        
        candidate_list = list(candidate_sheets)
        logger.info(f"✅ Pre-filtered candidates: {candidate_list}")
        return candidate_list
    
    def create_llm_prompt(self, query: str, candidate_sheets: List[str]) -> str:
        """Create the LLM prompt for sheet planning."""
        
        # Get semantic layer information
        semantic_layer = self.column_indexer_agent.semantic_layer
        primary_join_key = semantic_layer.get('primary_join_key')
        
        # Get column information for candidate sheets
        sheet_info = []
        for sheet_name in candidate_sheets:
            if sheet_name in self.sheet_catalog:
                df = self.sheet_catalog[sheet_name]
                columns = list(df.columns)
                sheet_info.append({
                    'name': sheet_name,
                    'rows': len(df),
                    'columns': columns,
                    'sample_data': df.head(3).to_dict('records')
                })
        
        # Get common columns and potential join keys
        common_columns = self.column_indexer_agent.find_common_columns(min_sheets=2)
        potential_keys = self.column_indexer_agent.find_potential_join_keys()
        
        prompt = f"""
You are an expert data analyst tasked with creating a plan for analyzing multi-sheet Excel data.

USER QUERY: "{query}"

AVAILABLE SHEETS:
"""
        
        for info in sheet_info:
            prompt += f"""
- {info['name']} ({info['rows']} rows, {len(info['columns'])} columns)
  Columns: {', '.join(info['columns'])}
  Sample data: {info['sample_data'][:2]}  # First 2 rows
"""
        
        prompt += f"""

COMMON COLUMNS ACROSS SHEETS:
"""
        for col_name, refs in common_columns.items():
            sheets = [ref.sheet_name for ref in refs]
            prompt += f"- {col_name}: {', '.join(sheets)}\n"
        
        prompt += f"""

POTENTIAL JOIN KEYS:
"""
        for key in potential_keys:
            refs = self.column_indexer_agent.get_column_refs(key)
            sheets = [ref.sheet_name for ref in refs]
            prompt += f"- {key}: {', '.join(sheets)}\n"
        
        if primary_join_key:
            prompt += f"\nPRIMARY JOIN KEY: {primary_join_key}\n"
        
        prompt += """

INSTRUCTIONS:
1. Analyze the user query and determine which sheets are needed
2. Decide on the best strategy to combine the sheets:
   - JOIN: Use when you need to combine data from different sheets based on a common key
   - UNION: Use when you need to stack similar data from different sheets
   - SINGLE SHEET: Use when the query can be answered from one sheet
3. If the strategy is unclear or multiple approaches are possible, ask for clarification

RESPONSE FORMAT:
If you can create a clear plan, respond with:
PLAN:
- strategy: [join|union|single_sheet]
- primary_sheets: [list of sheet names]
- join_keys: [list of join keys, if applicable]
- sheet_aliases: [optional mapping of sheet names to aliases]
- additional_columns: [optional mapping for status labels in unions]

If you need clarification, respond with:
CLARIFY:
- question: [clear question for the user]
- options: [list of possible approaches]
- context: [relevant information to help user decide]

Examples:
- "Compare salaries across departments" → JOIN on department
- "Show all employees and their performance" → JOIN on employee_id
- "Compare active vs inactive employees" → UNION with status labels
- "What's the average salary?" → SINGLE SHEET (Employees)
"""
        
        return prompt
    
    def parse_llm_response(self, response: str) -> Tuple[Optional[SheetPlan], Optional[DisambiguationQuestion]]:
        """Parse the LLM response to extract plan or clarification question."""
        
        response = response.strip()
        
        if response.startswith("PLAN:"):
            # Parse the plan
            plan_lines = response[5:].strip().split('\n')
            plan_data = {}
            
            for line in plan_lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().lstrip('-').strip()  # Remove leading dash and whitespace
                    value = value.strip()
                    
                    if key == 'strategy':
                        plan_data['join_strategy'] = value
                    elif key == 'primary_sheets':
                        # Parse list format - handle both [item1, item2] and [item] formats
                        if '[' in value and ']' in value:
                            # Extract content between brackets
                            start_bracket = value.find('[')
                            end_bracket = value.find(']')
                            content = value[start_bracket+1:end_bracket].strip()
                            if content:
                                # Split by comma and clean up
                                sheets = [s.strip() for s in content.split(',')]
                                plan_data['primary_sheets'] = sheets
                            else:
                                plan_data['primary_sheets'] = []
                        else:
                            # Fallback: treat as single value
                            plan_data['primary_sheets'] = [value.strip()]
                    elif key == 'join_keys':
                        # Parse list format - handle both [item1, item2] and [item] formats
                        if '[' in value and ']' in value:
                            # Extract content between brackets
                            content = value[value.find('[')+1:value.find(']')].strip()
                            if content:
                                # Split by comma and clean up
                                keys = [k.strip() for k in content.split(',')]
                                plan_data['join_keys'] = keys
                            else:
                                plan_data['join_keys'] = []
                        else:
                            # Fallback: treat as single value
                            plan_data['join_keys'] = [value.strip()]
                    elif key == 'sheet_aliases':
                        # Parse dictionary format
                        aliases = {}
                        alias_matches = re.findall(r'(\w+):\s*(\w+)', value)
                        for alias_match in alias_matches:
                            aliases[alias_match[0]] = alias_match[1]
                        plan_data['sheet_aliases'] = aliases
                    elif key == 'additional_columns':
                        # Parse dictionary format for status labels
                        additional = {}
                        # Handle more complex values with spaces and special characters
                        if '{' in value and '}' in value:
                            # Extract content between braces
                            start_brace = value.find('{')
                            end_brace = value.find('}')
                            content = value[start_brace+1:end_brace].strip()
                            if content:
                                # Parse key-value pairs more flexibly
                                pairs = content.split(',')
                                for pair in pairs:
                                    if ':' in pair:
                                        key_part, value_part = pair.split(':', 1)
                                        key_part = key_part.strip().strip("'\"")
                                        value_part = value_part.strip().strip("'\"")
                                        additional[key_part] = value_part
                        else:
                            # Fallback to simple regex for basic cases
                            additional_matches = re.findall(r'(\w+):\s*([^,}]+)', value)
                            for additional_match in additional_matches:
                                additional[additional_match[0]] = additional_match[1].strip()
                        plan_data['additional_columns'] = additional
            
            # Create SheetPlan object
            sheet_plan = SheetPlan(
                primary_sheets=plan_data.get('primary_sheets', []),
                join_strategy=plan_data.get('join_strategy'),
                join_keys=plan_data.get('join_keys'),
                sheet_aliases=plan_data.get('sheet_aliases'),
                additional_columns=plan_data.get('additional_columns')
            )
            
            logger.info(f"✅ Parsed sheet plan: {sheet_plan}")
            return sheet_plan, None
            
        elif response.startswith("CLARIFY:"):
            # Parse the clarification question
            clarify_lines = response[8:].strip().split('\n')
            question_data = {}
            
            for line in clarify_lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key == 'question':
                        question_data['question'] = value
                    elif key == 'options':
                        # Parse options list
                        options = []
                        option_matches = re.findall(r'\[(.*?)\]', value)
                        for option_match in option_matches:
                            options.append({'text': option_match.strip()})
                        question_data['options'] = options
                    elif key == 'context':
                        question_data['context'] = value
            
            # Create DisambiguationQuestion object
            disambiguation = DisambiguationQuestion(
                question=question_data.get('question', 'How should I proceed?'),
                options=question_data.get('options', []),
                context=question_data.get('context', {}),
                question_type='sheet_combination'
            )
            
            logger.info(f"❓ Parsed disambiguation question: {disambiguation.question}")
            return None, disambiguation
        
        else:
            logger.warning(f"⚠️ Unrecognized LLM response format: {response[:100]}...")
            return None, None
    
    def create_sheet_plan(self, query: str) -> Tuple[Optional[SheetPlan], Optional[DisambiguationQuestion]]:
        """
        Main method to create a sheet plan or request clarification.
        
        Returns:
            Tuple of (SheetPlan, None) if plan is clear, or (None, DisambiguationQuestion) if clarification needed
        """
        logger.info(f"🎯 Creating sheet plan for query: '{query}'")
        
        # Stage 1: Heuristic pre-filtering
        candidate_sheets = self.heuristic_pre_filter(query)
        
        if not candidate_sheets:
            logger.warning("⚠️ No candidate sheets found")
            return None, DisambiguationQuestion(
                question="No relevant sheets found for your query. Please check your data or rephrase your question.",
                options=[],
                context={},
                question_type='no_sheets'
            )
        
        # Stage 2: LLM planning
        try:
            prompt = self.create_llm_prompt(query, candidate_sheets)
            
            messages = [
                {"role": "system", "content": "You are an expert data analyst specializing in multi-sheet Excel analysis."},
                {"role": "user", "content": prompt}
            ]
            
            logger.info("📤 Sending sheet planning request to LLM...")
            response = make_llm_call(
                messages=messages,
                model="gpt-4.1",
                temperature=0.1,
                max_tokens=1000
            )
            
            llm_response = response.choices[0].message.content.strip()
            logger.info(f"📥 LLM response: {llm_response[:200]}...")
            
            # Parse the response
            sheet_plan, disambiguation = self.parse_llm_response(llm_response)
            
            if sheet_plan:
                logger.info(f"✅ Sheet plan created successfully: {sheet_plan}")
            elif disambiguation:
                logger.info(f"❓ Clarification needed: {disambiguation.question}")
            else:
                logger.warning("⚠️ Could not parse LLM response")
                # Fallback to simple plan
                sheet_plan = SheetPlan(
                    primary_sheets=candidate_sheets[:2],  # Use first 2 candidates
                    join_strategy='union' if len(candidate_sheets) > 1 else 'single_sheet'
                )
            
            return sheet_plan, disambiguation
            
        except Exception as e:
            logger.error(f"❌ Error in LLM planning: {e}")
            # Fallback to simple heuristic plan
            fallback_plan = SheetPlan(
                primary_sheets=candidate_sheets[:2],
                join_strategy='union' if len(candidate_sheets) > 1 else 'single_sheet'
            )
            logger.info(f"🔄 Using fallback plan: {fallback_plan}")
            return fallback_plan, None
    
    def validate_sheet_plan(self, sheet_plan: SheetPlan) -> Tuple[bool, List[str]]:
        """Validate a sheet plan for feasibility."""
        errors = []
        
        # Check if all primary sheets exist
        for sheet_name in sheet_plan.primary_sheets:
            if sheet_name not in self.sheet_catalog:
                errors.append(f"Sheet '{sheet_name}' not found in catalog")
        
        # Check join keys if using join strategy
        if sheet_plan.join_strategy == 'join' and sheet_plan.join_keys:
            for key in sheet_plan.join_keys:
                refs = self.column_indexer_agent.get_column_refs(key)
                if not refs:
                    errors.append(f"Join key '{key}' not found in any sheet")
                elif len(refs) < 2:
                    errors.append(f"Join key '{key}' only found in {len(refs)} sheet(s), need at least 2")
        
        # Check for union columns if using union strategy
        if sheet_plan.join_strategy == 'union' and sheet_plan.union_columns:
            for col in sheet_plan.union_columns:
                found = False
                for sheet_name in sheet_plan.primary_sheets:
                    if sheet_name in self.sheet_catalog:
                        if col in self.sheet_catalog[sheet_name].columns:
                            found = True
                            break
                if not found:
                    errors.append(f"Union column '{col}' not found in primary sheets")
        
        is_valid = len(errors) == 0
        return is_valid, errors 