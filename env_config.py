"""
Environment Configuration
========================

Copy the .env.template file to .env and fill in your actual API keys.
This file shows you how to set up environment variables for all AI models.

Required Environment Variables:
------------------------------

# Azure OpenAI Configuration
AZURE_API_KEY=your_azure_api_key_here
AZURE_ENDPOINT=https://ai-sherpartap11019601ai587562462851.openai.azure.com
AZURE_API_VERSION=2025-01-01-preview
AZURE_DEPLOYMENT_NAME=gpt-4.1

# Google Gemini Configuration
GEMINI_API_KEY=

# Other Configuration
NVIDIA_API_KEY=your_nvidia_api_key_here

Setup Instructions:
------------------
1. Create a .env file in the project root
2. Copy the environment variables above into your .env file
3. Replace the placeholder values with your actual API keys
4. Make sure .env is in your .gitignore (it already is)
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_env_var(key: str, default: str = None, required: bool = False) -> str:
    """Get environment variable with optional default and required validation."""
    value = os.getenv(key, default)
    if required and not value:
        raise EnvironmentError(f"Required environment variable {key} is not set")
    return value

# Azure OpenAI Configuration
AZURE_API_KEY = get_env_var("AZURE_API_KEY")
AZURE_ENDPOINT = get_env_var("AZURE_ENDPOINT", "https://ai-sherpartap11019601ai587562462851.openai.azure.com")
AZURE_API_VERSION = get_env_var("AZURE_API_VERSION", "2025-01-01-preview")
AZURE_DEPLOYMENT_NAME = get_env_var("AZURE_DEPLOYMENT_NAME", "gpt-4.1")

# Google Gemini Configuration
GEMINI_API_KEY = get_env_var("GEMINI_API_KEY")

# Other Configuration
NVIDIA_API_KEY = get_env_var("NVIDIA_API_KEY")