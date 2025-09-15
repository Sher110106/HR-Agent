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

_secrets_loaded = False

def _load_secrets_to_env():
    """Load secrets from Streamlit and map them to environment variables."""
    global _secrets_loaded
    if _secrets_loaded:
        return
        
    try:
        import streamlit as st

        # --- Azure OpenAI Configuration ---
        # Support both [azure] with snake_case keys and [azure_openai] with UPPERCASE keys
        if "azure" in st.secrets:
            azure_sec = st.secrets["azure"]
            os.environ["AZURE_API_KEY"] = azure_sec.get("api_key", os.environ.get("AZURE_API_KEY", ""))
            os.environ["AZURE_ENDPOINT"] = azure_sec.get("endpoint", os.environ.get("AZURE_ENDPOINT", ""))
            os.environ["AZURE_API_VERSION"] = azure_sec.get("api_version", os.environ.get("AZURE_API_VERSION", ""))
            os.environ["AZURE_DEPLOYMENT_NAME"] = azure_sec.get("deployment_name", os.environ.get("AZURE_DEPLOYMENT_NAME", ""))

        if "azure_openai" in st.secrets:
            azure_alt = st.secrets["azure_openai"]
            # Accept uppercase or snake_case keys
            os.environ["AZURE_API_KEY"] = azure_alt.get("AZURE_API_KEY", azure_alt.get("api_key", os.environ.get("AZURE_API_KEY", "")))
            os.environ["AZURE_ENDPOINT"] = azure_alt.get("AZURE_ENDPOINT", azure_alt.get("endpoint", os.environ.get("AZURE_ENDPOINT", "")))
            os.environ["AZURE_API_VERSION"] = azure_alt.get("AZURE_API_VERSION", azure_alt.get("api_version", os.environ.get("AZURE_API_VERSION", "")))
            os.environ["AZURE_DEPLOYMENT_NAME"] = azure_alt.get("AZURE_DEPLOYMENT_NAME", azure_alt.get("deployment_name", os.environ.get("AZURE_DEPLOYMENT_NAME", "")))

        # --- Google Gemini Configuration ---
        if "gemini" in st.secrets:
            gemini_sec = st.secrets["gemini"]
            os.environ["GEMINI_API_KEY"] = gemini_sec.get("api_key", os.environ.get("GEMINI_API_KEY", ""))

        if "google_gemini" in st.secrets:
            gemini_alt = st.secrets["google_gemini"]
            os.environ["GEMINI_API_KEY"] = gemini_alt.get("GEMINI_API_KEY", gemini_alt.get("api_key", os.environ.get("GEMINI_API_KEY", "")))

        # --- NVIDIA (optional) ---
        if "nvidia" in st.secrets:
            nvidia_sec = st.secrets["nvidia"]
            os.environ["NVIDIA_API_KEY"] = nvidia_sec.get("api_key", os.environ.get("NVIDIA_API_KEY", ""))

        _secrets_loaded = True
        
    except (ImportError, AttributeError, KeyError):
        # Streamlit not available or secrets not configured
        pass

def get_secret_or_env(key: str, default: str = None, required: bool = False) -> str:
    """Get secret from environment variable (which may have been loaded from Streamlit secrets)."""
    # Try to load secrets first
    _load_secrets_to_env()
    
    value = os.getenv(key, default)
    if required and not value:
        raise EnvironmentError(f"Required secret/environment variable {key} is not set")
    return value

# Azure OpenAI Configuration
AZURE_API_KEY = get_secret_or_env("AZURE_API_KEY")
AZURE_ENDPOINT = get_secret_or_env("AZURE_ENDPOINT", "https://ai-sherpartap11019601ai587562462851.openai.azure.com")
AZURE_API_VERSION = get_secret_or_env("AZURE_API_VERSION", "2025-01-01-preview")
AZURE_DEPLOYMENT_NAME = get_secret_or_env("AZURE_DEPLOYMENT_NAME", "gpt-4.1")

# Google Gemini Configuration
GEMINI_API_KEY = get_secret_or_env("GEMINI_API_KEY")

# Other Configuration
NVIDIA_API_KEY = get_secret_or_env("NVIDIA_API_KEY")