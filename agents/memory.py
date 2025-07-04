"""Memory-related agent utilities.

This module houses the ColumnMemoryAgent (for storing AI-generated column
descriptions), ConversationMemoryTool helper, and SystemPromptMemoryAgent for
managing dynamic system prompts.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ColumnMemoryAgent:
    """Memory system for storing AI-generated column descriptions."""

    def __init__(self) -> None:
        self._column_descriptions: Dict[str, str] = {}
        logger.info("🧠 ColumnMemoryAgent initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def store_column_description(self, column_name: str, description: str) -> None:
        """Persist a description for *column_name*."""
        self._column_descriptions[column_name] = description
        logger.debug("💾 Stored description for column %s", column_name)

    def get_column_description(self, column_name: str) -> str:
        """Return the stored description for *column_name* (or empty str)."""
        return self._column_descriptions.get(column_name, "")

    def get_all_descriptions(self) -> Dict[str, str]:
        """Return **copy** of the full description mapping."""
        return dict(self._column_descriptions)

    def has_descriptions(self) -> bool:
        """Return ``True`` if at least one description is stored."""
        return bool(self._column_descriptions)

    def clear_descriptions(self) -> None:
        """Remove all stored descriptions."""
        self._column_descriptions.clear()
        logger.info("🗑️ Cleared all column descriptions")


class SystemPromptMemoryAgent:
    """Memory system for managing and applying dynamic system prompts."""
    
    def __init__(self):
        from utils.system_prompts import get_prompt_manager
        self.prompt_manager = get_prompt_manager()
        logger.info("🎯 SystemPromptMemoryAgent initialized")
    
    def get_active_prompt_context(self) -> str:
        """Get the active system prompt text for injection into LLM calls."""
        active_prompt = self.prompt_manager.get_active_prompt()
        if active_prompt:
            logger.debug(f"🎯 Retrieved active prompt: {active_prompt.name}")
            return active_prompt.prompt
        return ""
    
    def apply_system_prompt(self, base_system_prompt: str) -> str:
        """Apply the active system prompt to a base system prompt."""
        active_prompt = self.get_active_prompt_context()
        if active_prompt:
            # Combine the active prompt with the base prompt
            enhanced_prompt = f"""{active_prompt}

---

{base_system_prompt}"""
            logger.debug("🎯 Applied active system prompt to base prompt")
            return enhanced_prompt
        return base_system_prompt
    
    def get_prompt_summary(self) -> Dict[str, any]:
        """Get summary information about the active prompt."""
        active_prompt = self.prompt_manager.get_active_prompt()
        if active_prompt:
            return {
                "name": active_prompt.name,
                "description": active_prompt.description,
                "category": active_prompt.category,
                "usage_count": active_prompt.usage_count,
                "tags": active_prompt.tags
            }
        return {"name": "None", "description": "No active system prompt", "category": "none"}
    
    def set_active_prompt(self, prompt_name: str) -> bool:
        """Set the active system prompt by name."""
        success = self.prompt_manager.set_active_prompt(prompt_name)
        if success:
            logger.info(f"🎯 SystemPromptMemoryAgent activated prompt: {prompt_name}")
        else:
            logger.warning(f"🎯 Failed to activate prompt: {prompt_name}")
        return success
    
    def clear_active_prompt(self) -> None:
        """Clear the active system prompt."""
        self.prompt_manager.clear_active_prompt()
        logger.info("🎯 SystemPromptMemoryAgent cleared active prompt")
    
    def create_custom_prompt(self, name: str, prompt: str, description: str, 
                           category: str = "custom", tags: List[str] = None) -> bool:
        """Create a new custom system prompt."""
        success = self.prompt_manager.create_prompt(name, prompt, description, category, tags)
        if success:
            logger.info(f"🎯 Created custom prompt: {name}")
        return success
    
    def get_available_prompts(self) -> List[Dict[str, str]]:
        """Get list of available prompts with metadata."""
        prompts = self.prompt_manager.list_prompts()
        return [{
            "name": p.name,
            "description": p.description,
            "category": p.category,
            "usage_count": p.usage_count,
            "tags": ", ".join(p.tags) if p.tags else ""
        } for p in prompts]


# ----------------------------------------------------------------------
# Conversation memory helper
# ----------------------------------------------------------------------

def ConversationMemoryTool(messages: List[Dict[str, str]], max_history: int = 4) -> str:  # noqa: D401
    """Return the last *max_history* chat turns as a concise string.

    Only the most recent *max_history* messages are included. Assistant messages
    are stripped of HTML tags (e.g., the thinking `<details>` blocks) so the LLM
    sees clean content.
    """
    logger.info("🧠 ConversationMemoryTool: processing %s messages (max %s)", len(messages), max_history)

    if not messages:
        return ""

    relevant = messages[-max_history:]
    context_parts: list[str] = []
    for msg in relevant:
        role = msg["role"].capitalize()
        content = msg["content"]

        if role == "Assistant":  # strip HTML / details
            content = re.sub(r"<details[^>]*>.*?</details>", "", content, flags=re.DOTALL).strip()
            if len(content) > 200:
                content = content[:200] + "..."
        context_parts.append(f"{role}: {content}")

    return "\n".join(context_parts)


def enhance_prompt_with_context(base_prompt: str, system_prompt_agent: SystemPromptMemoryAgent) -> str:
    """Helper function to enhance any prompt with active system prompt context."""
    return system_prompt_agent.apply_system_prompt(base_prompt) 