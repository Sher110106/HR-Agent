"""
Environment Configuration
========================

This file loads secrets from Streamlit secrets and maps them to environment variables.
All API keys and sensitive configuration are managed through Streamlit's secrets management.

Required Streamlit Secrets:
--------------------------

# Azure OpenAI Configuration
azure:
  api_key: your_azure_api_key_here
  endpoint: https://ai-sherpartap11019601ai587562462851.openai.azure.com
  api_version: 2025-01-01-preview
  deployment_name: gpt-4.1

# Google Gemini Configuration
gemini:
  api_key: your_gemini_api_key_here

# Other Configuration
nvidia:
  api_key: your_nvidia_api_key_here

Setup Instructions:
------------------
1. Configure secrets in Streamlit Cloud or local .streamlit/secrets.toml
2. Use the structure above for organizing your secrets
3. Secrets are automatically loaded and mapped to environment variables
"""

import os
import streamlit as st

def get_secret_or_env(key: str, default: str = None, required: bool = False) -> str:
    """Get secret from Streamlit secrets or environment variable with optional default and required validation."""
    # Try to get from Streamlit secrets first
    try:
        # Handle nested secrets like azure.api_key
        if '.' in key:
            parts = key.split('.')
            secret_value = st.secrets
            for part in parts:
                secret_value = secret_value[part]
            return secret_value
        else:
            return st.secrets[key]
    except (KeyError, AttributeError):
        # Fallback to environment variable
        value = os.getenv(key, default)
        if required and not value:
            raise EnvironmentError(f"Required secret/environment variable {key} is not set")
        return value

# Map secrets into environment variables for compatibility
try:
    # Azure OpenAI Configuration
    os.environ["AZURE_API_KEY"] = st.secrets["azure"]["api_key"]
    os.environ["AZURE_ENDPOINT"] = st.secrets["azure"]["endpoint"]
    os.environ["AZURE_API_VERSION"] = st.secrets["azure"]["api_version"]
    os.environ["AZURE_DEPLOYMENT_NAME"] = st.secrets["azure"]["deployment_name"]
    
    # Google Gemini Configuration
    os.environ["GEMINI_API_KEY"] = st.secrets["gemini"]["api_key"]
    
    # Other Configuration
    os.environ["NVIDIA_API_KEY"] = st.secrets["nvidia"]["api_key"]
    
except (KeyError, AttributeError) as e:
    # Fallback to environment variables if secrets are not available
    pass

# Azure OpenAI Configuration
AZURE_API_KEY = get_secret_or_env("AZURE_API_KEY")
AZURE_ENDPOINT = get_secret_or_env("AZURE_ENDPOINT", "https://ai-sherpartap11019601ai587562462851.openai.azure.com")
AZURE_API_VERSION = get_secret_or_env("AZURE_API_VERSION", "2025-01-01-preview")
AZURE_DEPLOYMENT_NAME = get_secret_or_env("AZURE_DEPLOYMENT_NAME", "gpt-4.1")

# Google Gemini Configuration
GEMINI_API_KEY = get_secret_or_env("GEMINI_API_KEY")

# Other Configuration
NVIDIA_API_KEY = get_secret_or_env("NVIDIA_API_KEY")