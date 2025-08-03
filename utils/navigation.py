"""
Navigation registry system for the Data Analysis Agent.
Provides extensible page management and routing for Streamlit applications.
"""

import importlib
from typing import Dict, Callable, Any, Optional
from dataclasses import dataclass
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PageConfig:
    """Configuration for a page in the navigation system."""
    title: str
    icon: str
    module_path: str
    function_name: str
    description: Optional[str] = None
    requires_auth: bool = False
    order: int = 0


class NavigationRegistry:
    """Registry for managing application pages and navigation."""
    
    def __init__(self):
        self.pages: Dict[str, PageConfig] = {}
        self._loaded_functions: Dict[str, Callable] = {}
        self._initialize_default_pages()
        logger.info("🧭 NavigationRegistry initialized")
    
    def _initialize_default_pages(self):
        """Initialize default pages in the application."""
        default_pages = {
            "data_analysis": PageConfig(
                title="📊 Data Analysis",
                icon="📊",
                module_path="pages.data_analysis",
                function_name="data_analysis_page",
                description="Upload and analyze data with AI assistance",
                order=1
            ),
            "smart_analysis": PageConfig(
                title="🤖 Smart Analysis",
                icon="🤖",
                module_path="pages.smart_analysis",
                function_name="smart_analysis_page",
                description="Natural language data analysis powered by PandasAI",
                order=2
            ),
            "excel_analysis": PageConfig(
                title="📈 Excel Analysis",
                icon="📈",
                module_path="pages.excel_analysis",
                function_name="excel_analysis_page",
                description="Upload and analyze multi-sheet Excel files with AI assistance",
                order=3
            ),
            "system_prompts": PageConfig(
                title="🎯 System Prompts",
                icon="🎯",
                module_path="pages.system_prompt_manager",
                function_name="system_prompt_manager_page",
                description="Create and manage custom AI system prompts",
                order=4
            ),
            "monitoring": PageConfig(
                title="📈 Monitoring",
                icon="📈",
                module_path="pages.monitoring",
                function_name="monitoring_dashboard",
                description="Application health and performance monitoring",
                order=5
            )
        }
        
        for page_id, config in default_pages.items():
            self.pages[page_id] = config
            logger.debug(f"🧭 Registered default page: {page_id}")
    
    def register_page(self, page_id: str, config: PageConfig) -> bool:
        """Register a new page in the navigation system."""
        if page_id in self.pages:
            logger.warning(f"🧭 Page '{page_id}' already registered")
            return False
        
        self.pages[page_id] = config
        logger.info(f"🧭 Registered page: {page_id}")
        return True
    
    def unregister_page(self, page_id: str) -> bool:
        """Remove a page from the navigation system."""
        if page_id not in self.pages:
            return False
        
        del self.pages[page_id]
        
        # Also remove cached function
        if page_id in self._loaded_functions:
            del self._loaded_functions[page_id]
        
        logger.info(f"🧭 Unregistered page: {page_id}")
        return True
    
    def get_page_config(self, page_id: str) -> Optional[PageConfig]:
        """Get the configuration for a specific page."""
        return self.pages.get(page_id)
    
    def get_page_titles(self) -> Dict[str, str]:
        """Get all page IDs mapped to their display titles."""
        # Sort by order, then by title
        sorted_pages = sorted(
            self.pages.items(),
            key=lambda x: (x[1].order, x[1].title)
        )
        return {page_id: config.title for page_id, config in sorted_pages}
    
    def get_pages_by_category(self, category: str) -> Dict[str, PageConfig]:
        """Get all pages in a specific category (future enhancement)."""
        # For now, return empty dict as categories are not implemented
        return {}
    
    def _load_page_function(self, page_id: str) -> Optional[Callable]:
        """Lazy load a page function."""
        if page_id in self._loaded_functions:
            return self._loaded_functions[page_id]
        
        if page_id not in self.pages:
            logger.error(f"🧭 Page '{page_id}' not found in registry")
            return None
        
        config = self.pages[page_id]
        
        try:
            # Import the module
            module = importlib.import_module(config.module_path)
            
            # Get the function from the module
            if not hasattr(module, config.function_name):
                logger.error(f"🧭 Function '{config.function_name}' not found in module '{config.module_path}'")
                return None
            
            function = getattr(module, config.function_name)
            
            # Cache the loaded function
            self._loaded_functions[page_id] = function
            logger.debug(f"🧭 Loaded function for page: {page_id}")
            
            return function
            
        except Exception as e:
            logger.error(f"🧭 Failed to load page function for '{page_id}': {e}")
            return None
    
    def render_page(self, page_id: str) -> bool:
        """Render a specific page by calling its function."""
        function = self._load_page_function(page_id)
        
        if function is None:
            logger.error(f"🧭 Cannot render page '{page_id}' - function not available")
            return False
        
        try:
            # Call the page function
            function()
            logger.debug(f"🧭 Successfully rendered page: {page_id}")
            return True
            
        except Exception as e:
            logger.exception(f"🧭 Error rendering page '{page_id}': {e}")
            return False
    
    def list_pages(self) -> Dict[str, PageConfig]:
        """List all registered pages."""
        return self.pages.copy()
    
    def search_pages(self, query: str) -> Dict[str, PageConfig]:
        """Search pages by title or description."""
        query = query.lower()
        results = {}
        
        for page_id, config in self.pages.items():
            if (query in config.title.lower() or
                (config.description and query in config.description.lower())):
                results[page_id] = config
        
        return results


# Global navigation registry instance
_nav_registry = NavigationRegistry()

def get_navigation_registry() -> NavigationRegistry:
    """Get the global navigation registry instance."""
    return _nav_registry

def register_page(page_id: str, title: str, icon: str, module_path: str, 
                 function_name: str, description: str = None, order: int = 0) -> bool:
    """Convenience function to register a page."""
    config = PageConfig(
        title=title,
        icon=icon,
        module_path=module_path,
        function_name=function_name,
        description=description,
        order=order
    )
    return _nav_registry.register_page(page_id, config) 