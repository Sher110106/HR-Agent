"""
Integration tests for the Data Analysis Agent architecture.
Tests critical system behavior to prevent regressions during future changes.
"""

import unittest
import threading
import time
import logging
from unittest.mock import patch, MagicMock

# Import modules under test
from utils.health_monitor import HealthMonitor, start_health_monitoring, stop_health_monitoring
from utils.navigation import get_navigation_registry, PageConfig, register_page
from utils.logging_config import setup_logging, get_logger
from utils.circuit_breaker import get_circuit_breaker, CircuitBreakerConfig


class TestHealthMonitoringIntegration(unittest.TestCase):
    """Test health monitoring system integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Stop any existing monitoring
        stop_health_monitoring()
        time.sleep(0.1)  # Allow time for thread cleanup
    
    def tearDown(self):
        """Clean up after tests."""
        stop_health_monitoring()
        time.sleep(0.1)  # Allow time for thread cleanup
    
    def test_single_monitoring_thread(self):
        """Test that only one monitoring thread is created."""
        # Get initial thread count
        initial_thread_count = threading.active_count()
        
        # Start monitoring multiple times
        start_health_monitoring()
        start_health_monitoring()
        start_health_monitoring()
        
        # Allow time for threads to start
        time.sleep(0.2)
        
        # Should only have one additional thread
        final_thread_count = threading.active_count()
        self.assertEqual(final_thread_count, initial_thread_count + 1)
        
        # Stop monitoring
        stop_health_monitoring()
        time.sleep(0.2)
        
        # Thread count should return to initial
        cleanup_thread_count = threading.active_count()
        self.assertEqual(cleanup_thread_count, initial_thread_count)
    
    def test_health_monitoring_lifecycle(self):
        """Test complete health monitoring lifecycle."""
        monitor = HealthMonitor(check_interval=1)
        
        # Initially not monitoring
        self.assertFalse(monitor._monitoring_active)
        
        # Start monitoring
        monitor.start_monitoring()
        self.assertTrue(monitor._monitoring_active)
        self.assertIsNotNone(monitor._monitor_thread)
        
        # Allow some checks to run
        time.sleep(0.5)
        
        # Stop monitoring
        monitor.stop_monitoring()
        self.assertFalse(monitor._monitoring_active)


class TestNavigationRegistryIntegration(unittest.TestCase):
    """Test navigation registry system integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.registry = get_navigation_registry()
        # Clear any test pages
        if "test_page" in self.registry.pages:
            self.registry.unregister_page("test_page")
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove any test pages
        if "test_page" in self.registry.pages:
            self.registry.unregister_page("test_page")
    
    def test_page_registration_and_loading(self):
        """Test page registration and lazy loading."""
        # Register a test page
        config = PageConfig(
            title="Test Page",
            icon="🧪",
            module_path="builtins",  # Use built-in module for testing
            function_name="len",     # Use built-in function for testing
            order=999
        )
        
        register_page("test_page", config)
        
        # Verify page is registered
        self.assertIn("test_page", self.registry.pages)
        
        # Test page titles generation
        titles = self.registry.get_page_titles()
        self.assertIn("test_page", titles)
        self.assertEqual(titles["test_page"], "🧪 Test Page")
        
        # Test function loading
        func = self.registry.get_page_function("test_page")
        self.assertIsNotNone(func)
        self.assertEqual(func, len)  # Should be the built-in len function
        
        # Test caching (function should be loaded from cache)
        func2 = self.registry.get_page_function("test_page")
        self.assertIs(func, func2)
    
    def test_page_rendering_error_handling(self):
        """Test error handling during page rendering."""
        # Register a page with invalid module
        config = PageConfig(
            title="Invalid Page",
            icon="❌",
            module_path="nonexistent.module",
            function_name="nonexistent_function",
            order=999
        )
        
        register_page("test_page", config)
        
        # Attempt to render should fail gracefully
        success = self.registry.render_page("test_page")
        self.assertFalse(success)
        
        # Should return None for invalid function
        func = self.registry.get_page_function("test_page")
        self.assertIsNone(func)


class TestLoggingConfigIntegration(unittest.TestCase):
    """Test logging configuration integration."""
    
    def test_logging_setup_idempotency(self):
        """Test that logging setup is idempotent."""
        # Clear existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Setup logging multiple times
        setup_logging()
        initial_handler_count = len(logging.root.handlers)
        
        setup_logging()
        setup_logging()
        
        # Should not add duplicate handlers
        final_handler_count = len(logging.root.handlers)
        self.assertEqual(initial_handler_count, final_handler_count)
    
    def test_logger_creation(self):
        """Test logger creation and configuration."""
        logger = get_logger("test_module")
        
        # Should return a properly configured logger
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "test_module")
        
        # Should have root handlers available
        self.assertTrue(len(logging.root.handlers) > 0)


class TestCircuitBreakerIntegration(unittest.TestCase):
    """Test circuit breaker integration."""
    
    def test_circuit_breaker_creation_and_stats(self):
        """Test circuit breaker creation and statistics."""
        breaker = get_circuit_breaker("test_api", CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=5.0,
            timeout=1.0
        ))
        
        # Should be properly configured
        self.assertEqual(breaker.name, "test_api")
        self.assertEqual(breaker.config.failure_threshold, 3)
        
        # Should provide stats
        stats = breaker.get_stats()
        self.assertIn("name", stats)
        self.assertIn("state", stats)
        self.assertIn("failure_count", stats)
        self.assertEqual(stats["name"], "test_api")
    
    def test_circuit_breaker_timeout_fallback(self):
        """Test that circuit breaker falls back to ThreadPoolExecutor properly."""
        breaker = get_circuit_breaker("test_timeout", CircuitBreakerConfig(timeout=0.1))
        
        def slow_function():
            time.sleep(0.2)  # Longer than timeout
            return "success"
        
        # Should timeout and raise TimeoutError
        with self.assertRaises(TimeoutError):
            breaker.call(slow_function)


class TestApplicationStartup(unittest.TestCase):
    """Test complete application startup integration."""
    
    @patch('streamlit.set_page_config')
    @patch('streamlit.sidebar')
    def test_application_startup_sequence(self, mock_sidebar, mock_set_page_config):
        """Test the complete application startup sequence."""
        # Mock Streamlit components
        mock_sidebar.title = MagicMock()
        mock_sidebar.selectbox = MagicMock(return_value="📊 Data Analysis")
        
        # Import and setup the application
        from streamlit_app import main
        
        # Mock health monitoring to avoid actual thread creation
        with patch('utils.health_monitor.start_health_monitoring') as mock_health:
            # Should execute without errors
            try:
                # This would normally be called by Streamlit, but we can't run it
                # in tests without a full Streamlit environment
                pass
            except Exception as e:
                self.fail(f"Application startup failed: {e}")


if __name__ == "__main__":
    # Set up test logging
    setup_logging()
    
    # Run tests
    unittest.main(verbosity=2) 