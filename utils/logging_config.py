"""
Centralized logging configuration for the Data Analysis Agent.
Provides consistent logging setup across all modules with Streamlit secrets-based configuration.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
import streamlit as st


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    log_format: Optional[str] = None
) -> None:
    """
    Set up centralized logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (defaults to 'data_analysis_agent.log')
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        log_format: Custom log format string
    """
    # Get configuration from Streamlit secrets with fallbacks to environment variables
    def get_logging_config(key: str, default: str) -> str:
        """Get logging configuration from Streamlit secrets or environment variables."""
        try:
            return st.secrets["logging"][key]
        except (KeyError, AttributeError):
            return os.getenv(f'LOG_{key}', default)
    
    log_level = log_level or get_logging_config('LEVEL', 'INFO').upper()
    log_file = log_file or get_logging_config('FILE', 'data_analysis_agent.log')
    max_bytes = int(get_logging_config('MAX_BYTES', str(max_bytes)))
    backup_count = int(get_logging_config('BACKUP_COUNT', str(backup_count)))
    
    # Default log format
    default_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_format = log_format or get_logging_config('FORMAT', default_format)
    
    # Prevent duplicate handlers
    if logging.root.handlers:
        return
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Configure file handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Set up root logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        handlers=[console_handler, file_handler],
        force=True  # Override existing configuration
    )
    
    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(f"📝 Logging configured: level={log_level}, file={log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    Ensures logging is configured before returning the logger.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    # Ensure logging is set up
    if not logging.root.handlers:
        setup_logging()
    
    return logging.getLogger(name)


def configure_module_logger(module_name: str) -> logging.Logger:
    """
    Configure and return a logger for a specific module.
    
    Args:
        module_name: Name of the module (typically __name__)
        
    Returns:
        Configured logger for the module
    """
    return get_logger(module_name)


# Streamlit secrets documentation
SECRETS_DOC = """
Streamlit Secrets for Logging Configuration:
logging:
  level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) [default: INFO]
  file: Path to log file [default: data_analysis_agent.log]
  max_bytes: Maximum log file size before rotation [default: 10485760 (10MB)]
  backup_count: Number of backup files to keep [default: 5]
  format: Custom log format string [default: %(asctime)s - %(name)s - %(levelname)s - %(message)s]

Fallback Environment Variables (if secrets not available):
- LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) [default: INFO]
- LOG_FILE: Path to log file [default: data_analysis_agent.log]
- LOG_MAX_BYTES: Maximum log file size before rotation [default: 10485760 (10MB)]
- LOG_BACKUP_COUNT: Number of backup files to keep [default: 5]
- LOG_FORMAT: Custom log format string [default: %(asctime)s - %(name)s - %(levelname)s - %(message)s]
""" 