"""
System prompt management for the Data Analysis Agent.
Handles storage, retrieval, and application of custom system prompts and templates.
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class SystemPrompt:
    """Configuration for a system prompt."""
    name: str
    prompt: str
    description: str
    category: str = "custom"
    created_at: str = ""
    last_used: str = ""
    usage_count: int = 0
    tags: List[str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []

class SystemPromptManager:
    """Manager for system prompts with persistence and templating."""
    
    def __init__(self, storage_file: str = "system_prompts.json"):
        self.storage_file = storage_file
        self.state_file = "system_prompt_state.json"
        self.prompts: Dict[str, SystemPrompt] = {}
        self.active_prompt: Optional[SystemPrompt] = None
        self._load_prompts()
        self._initialize_default_templates()
        self._load_active_prompt_state()
        logger.info("🎯 SystemPromptManager initialized")
    
    def _load_prompts(self):
        """Load prompts from storage file."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for name, prompt_data in data.items():
                        self.prompts[name] = SystemPrompt(**prompt_data)
                logger.info(f"🎯 Loaded {len(self.prompts)} system prompts")
            except Exception as e:
                logger.error(f"Failed to load system prompts: {e}")
    
    def _save_prompts(self):
        """Save prompts to storage file."""
        try:
            data = {name: asdict(prompt) for name, prompt in self.prompts.items()}
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug("🎯 System prompts saved")
        except Exception as e:
            logger.error(f"Failed to save system prompts: {e}")
    
    def _load_active_prompt_state(self):
        """Load the active prompt state from storage."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                    active_prompt_name = state_data.get("active_prompt")
                    if active_prompt_name:
                        self.active_prompt = self.get_prompt(active_prompt_name)
                        if self.active_prompt:
                            logger.info(f"🎯 Restored active prompt: {active_prompt_name}")
                        else:
                            logger.warning(f"🎯 Could not restore active prompt: {active_prompt_name}")
            except Exception as e:
                logger.error(f"Failed to load active prompt state: {e}")
    
    def _save_active_prompt_state(self):
        """Save the active prompt state to storage."""
        try:
            state_data = {
                "active_prompt": self.active_prompt.name if self.active_prompt else None
            }
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2)
            logger.debug("🎯 Active prompt state saved")
        except Exception as e:
            logger.error(f"Failed to save active prompt state: {e}")
    
    def _initialize_default_templates(self):
        """Initialize default prompt templates."""
        default_templates = {
            "analytical_expert": SystemPrompt(
                name="Analytical Expert",
                prompt="""You are an expert data analyst with deep knowledge in statistics, machine learning, and data visualization. 

Your approach:
- Break down complex problems systematically
- Use evidence-based reasoning
- Provide detailed explanations of methodology
- Suggest multiple analysis approaches when appropriate
- Always validate assumptions and highlight limitations
- Focus on actionable insights

Communication style: Professional, precise, and educational.""",
                description="Expert analyst focused on rigorous data analysis methodology",
                category="analysis",
                tags=["analysis", "statistics", "methodology"]
            ),
            
            "creative_problem_solver": SystemPrompt(
                name="Creative Problem Solver",
                prompt="""You are a creative problem solver who approaches challenges with innovative thinking and diverse perspectives.

Your approach:
- Think outside conventional boundaries
- Generate multiple creative solutions
- Use analogies and cross-domain insights
- Encourage experimentation and iteration
- Balance creativity with practicality
- Ask provocative questions to reframe problems

Communication style: Enthusiastic, inspiring, and open-minded.""",
                description="Creative approach to problem-solving with innovative thinking",
                category="creative",
                tags=["creativity", "innovation", "problem-solving"]
            ),
            
            "strategic_advisor": SystemPrompt(
                name="Strategic Advisor",
                prompt="""You are a strategic business advisor with expertise in decision-making and long-term planning.

Your approach:
- Consider multiple stakeholder perspectives
- Analyze short and long-term implications
- Identify risks and opportunities
- Provide clear recommendations with rationale
- Focus on business impact and ROI
- Consider market context and competitive landscape

Communication style: Authoritative, clear, and business-focused.""",
                description="Strategic business advisor for high-level decision making",
                category="strategy",
                tags=["strategy", "business", "decision-making"]
            ),
            
            "technical_architect": SystemPrompt(
                name="Technical Architect",
                prompt="""You are a senior technical architect with expertise in system design and engineering best practices.

Your approach:
- Focus on scalability, maintainability, and performance
- Consider security and compliance requirements
- Evaluate trade-offs between different approaches
- Provide detailed technical reasoning
- Suggest implementation patterns and best practices
- Think about system integration and future extensibility

Communication style: Technical, precise, and architecture-focused.""",
                description="Technical expert for system architecture and engineering decisions",
                category="technical",
                tags=["architecture", "engineering", "systems"]
            ),
            
            "hr_agent": SystemPrompt(
                name="HR Agent",
                prompt="""You are an AI assistant with expertise in human resources and data analysis, specializing in employee attrition. Your goal is to help users understand, calculate, and analyze employee attrition based on the provided formula and categories.

Attrition Formula:
- Numerator: (Number of Full-Time Employees Left) + (Number of Fixed-Term Employees who left before their contract expiry)
- Denominator: Average number of Full-Time Employees + Fixed-Term Employees during the specified period.

Types of Attrition:

Voluntary:
- Career Growth
- Compensation
- Personal Reasons
- Work-life Balance
- Relocation
- Family Reasons
- Health reasons
- Further Studies
- Management
- Other (requires specification)

Involuntary:
- Job Abandonment
- Disciplinary Issues
- Performance Issues""",
                description="HR specialist focusing on employee attrition analysis and insights",
                category="human_resources",
                tags=["hr", "attrition", "analysis"]
            )
        }
        
        # Add default templates if they don't exist
        for name, template in default_templates.items():
            if name not in self.prompts:
                self.prompts[name] = template
                logger.debug(f"🎯 Added default template: {name}")
        
        # Save after adding defaults
        self._save_prompts()
    
    def create_prompt(self, name: str, prompt: str, description: str, 
                     category: str = "custom", tags: List[str] = None) -> bool:
        """Create a new system prompt."""
        # Check if prompt with this name already exists (by storage key or display name)
        if self.get_prompt(name) is not None:
            logger.warning(f"🎯 Prompt '{name}' already exists")
            return False
        
        # Generate storage key from name (convert to lowercase, replace spaces with underscores)
        storage_key = name.lower().replace(" ", "_").replace("-", "_")
        
        # Make sure storage key is unique
        original_key = storage_key
        counter = 1
        while storage_key in self.prompts:
            storage_key = f"{original_key}_{counter}"
            counter += 1
        
        self.prompts[storage_key] = SystemPrompt(
            name=name,
            prompt=prompt,
            description=description,
            category=category,
            tags=tags or []
        )
        self._save_prompts()
        logger.info(f"🎯 Created new system prompt: {name}")
        return True
    
    def update_prompt(self, name: str, prompt: str = None, description: str = None,
                     category: str = None, tags: List[str] = None) -> bool:
        """Update an existing system prompt by storage key or display name."""
        existing = self.get_prompt(name)
        if not existing:
            logger.warning(f"🎯 Prompt '{name}' not found")
            return False
        if prompt is not None:
            existing.prompt = prompt
        if description is not None:
            existing.description = description
        if category is not None:
            existing.category = category
        if tags is not None:
            existing.tags = tags
        
        self._save_prompts()
        logger.info(f"🎯 Updated system prompt: {name}")
        return True
    
    def delete_prompt(self, name: str) -> bool:
        """Delete a system prompt by storage key or display name."""
        prompt = self.get_prompt(name)
        if not prompt:
            return False
        
        # Find the storage key for this prompt
        storage_key = None
        for key, stored_prompt in self.prompts.items():
            if stored_prompt == prompt:
                storage_key = key
                break
        
        if not storage_key:
            return False
        
        # Don't allow deletion of default templates
        if prompt.category in ["analysis", "creative", "strategy", "technical", "human_resources"]:
            logger.warning(f"🎯 Cannot delete default template: {name}")
            return False
        
        del self.prompts[storage_key]
        
        # Clear active prompt if it was deleted
        if self.active_prompt and self.active_prompt == prompt:
            self.active_prompt = None
        
        self._save_prompts()
        logger.info(f"🎯 Deleted system prompt: {name}")
        return True
    
    def get_prompt(self, name: str) -> Optional[SystemPrompt]:
        """Get a specific system prompt by storage key or display name."""
        # First try direct key lookup
        if name in self.prompts:
            return self.prompts[name]
        
        # If not found, try lookup by display name
        for prompt in self.prompts.values():
            if prompt.name == name:
                return prompt
        
        return None
    
    def list_prompts(self, category: str = None) -> List[SystemPrompt]:
        """List all prompts, optionally filtered by category."""
        prompts = list(self.prompts.values())
        if category:
            prompts = [p for p in prompts if p.category == category]
        return sorted(prompts, key=lambda p: p.name)
    
    def get_categories(self) -> List[str]:
        """Get all unique categories."""
        return sorted(list(set(p.category for p in self.prompts.values())))
    
    def set_active_prompt(self, name: str) -> bool:
        """Set the active system prompt by storage key or display name."""
        prompt = self.get_prompt(name)
        if not prompt:
            logger.warning(f"🎯 Prompt '{name}' not found")
            return False
        
        self.active_prompt = prompt
        
        # Update usage statistics
        self.active_prompt.last_used = datetime.now().isoformat()
        self.active_prompt.usage_count += 1
        self._save_prompts()
        self._save_active_prompt_state()
        
        logger.info(f"🎯 Activated system prompt: {name}")
        return True
    
    def clear_active_prompt(self):
        """Clear the active system prompt."""
        self.active_prompt = None
        self._save_active_prompt_state()
        logger.info("🎯 Cleared active system prompt")
    
    def get_active_prompt(self) -> Optional[SystemPrompt]:
        """Get the currently active system prompt."""
        return self.active_prompt
    
    def get_active_prompt_text(self) -> str:
        """Get the text of the active prompt, or empty string if none."""
        if self.active_prompt:
            return self.active_prompt.prompt
        return ""
    
    def search_prompts(self, query: str) -> List[SystemPrompt]:
        """Search prompts by name, description, or tags."""
        query = query.lower()
        results = []
        
        for prompt in self.prompts.values():
            if (query in prompt.name.lower() or 
                query in prompt.description.lower() or
                any(query in tag.lower() for tag in prompt.tags)):
                results.append(prompt)
        
        return sorted(results, key=lambda p: p.usage_count, reverse=True)
    
    def export_prompts(self, filepath: str) -> bool:
        """Export all prompts to a file."""
        try:
            data = {name: asdict(prompt) for name, prompt in self.prompts.items()}
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"🎯 Exported {len(self.prompts)} prompts to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export prompts: {e}")
            return False
    
    def import_prompts(self, filepath: str, overwrite: bool = False) -> int:
        """Import prompts from a file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            imported_count = 0
            for name, prompt_data in data.items():
                if name not in self.prompts or overwrite:
                    self.prompts[name] = SystemPrompt(**prompt_data)
                    imported_count += 1
            
            self._save_prompts()
            logger.info(f"🎯 Imported {imported_count} prompts from {filepath}")
            return imported_count
        except Exception as e:
            logger.error(f"Failed to import prompts: {e}")
            return 0
    
    def refresh(self):
        """Refresh prompts from storage file and defaults."""
        # Clear current prompts and reload
        self.prompts = {}
        self._load_prompts()
        self._initialize_default_templates()

# Global system prompt manager instance
_prompt_manager = SystemPromptManager()

def reinitialize_prompt_manager():
    """Reinitialize the global prompt manager to ensure all defaults are loaded."""
    global _prompt_manager
    _prompt_manager = SystemPromptManager()
    return _prompt_manager

def get_prompt_manager() -> SystemPromptManager:
    """Get the global system prompt manager instance."""
    return _prompt_manager

def get_active_prompt() -> str:
    """Get the active system prompt text."""
    return _prompt_manager.get_active_prompt_text()

def set_active_prompt(name: str) -> bool:
    """Set the active system prompt by name."""
    return _prompt_manager.set_active_prompt(name) 