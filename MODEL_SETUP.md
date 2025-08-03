# AI Model Configuration Guide

This guide explains how to set up and configure the supported AI models in the HR Agent application.

## Supported Models

The application now supports two AI models:

1. **GPT-4.1** (Azure OpenAI)
2. **Gemini 2.5 Pro** (Google)

## Environment Setup

### Step 1: Install Dependencies

First, install the required Python packages:

```bash
pip install -r requirements.txt
```

This will install the required dependencies:
- `python-dotenv` - Environment variable management

### Step 2: Create Environment File

Create a `.env` file in the project root directory:

```bash
# Copy the template and edit with your keys
cp env_config.py .env.template
# Then create your actual .env file
```

### Step 3: Configure API Keys

Add your API keys to the `.env` file:

```env
# Azure OpenAI Configuration
AZURE_API_KEY=your_azure_api_key_here
AZURE_ENDPOINT=https://ai-sherpartap11019601ai587562462851.openai.azure.com
AZURE_API_VERSION=2025-01-01-preview
AZURE_DEPLOYMENT_NAME=gpt-4.1

# Google Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Other Configuration
NVIDIA_API_KEY=your_nvidia_api_key_here
```

## Getting API Keys

### Azure OpenAI (GPT-4.1)
1. Go to [Azure Portal](https://portal.azure.com)
2. Create an Azure OpenAI resource
3. Get your API key from the resource's "Keys and Endpoint" section
4. Note your endpoint URL and deployment name

### Google Gemini
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

## Testing Your Setup

Run the integration test to verify all models work:

```bash
python test_integration.py
```

This will:
1. Check which models are available
2. Test each model with a simple query
3. Report any configuration issues

## Model Selection in the UI

Once configured, users can select between models in the application:

- **Sidebar**: Model selection dropdown appears in the AI Behavior section
- **Dynamic**: Only properly configured models will appear
- **Fallback**: If no model is selected, defaults to the first available model

## Troubleshooting

### Common Issues

1. **"No models available"**
   - Check your `.env` file exists and has correct keys
   - Verify API keys are valid and not expired

2. **"Azure OpenAI client not initialized"**
   - Check AZURE_API_KEY in `.env`
   - Verify endpoint URL is correct
   - Ensure deployment exists

3. **"Gemini API key not set"**
   - Add GEMINI_API_KEY to `.env`
   - Verify key is valid in Google AI Studio

### Debug Steps

1. Check environment loading:
   ```python
   from env_config import *
   print(f"Azure API Key: {'Set' if AZURE_API_KEY else 'Not set'}")
   print(f"Gemini API Key: {'Set' if GEMINI_API_KEY else 'Not set'}")
   ```

2. Test individual models:
   ```python
   from app_core.api import get_available_models
   models = get_available_models()
   print(f"Available models: {list(models.keys())}")
   ```

3. Check API connectivity:
   ```python
   from app_core.api import make_llm_call
   try:
       response = make_llm_call(
           messages=[{"role": "user", "content": "Hello"}],
           model="gpt-4.1"  # or "gemini-2.5-pro"
       )
       print("✅ API call successful")
   except Exception as e:
       print(f"❌ API call failed: {e}")
   ```

## Model Comparison

| Model | Provider | Cost | Best For |
|-------|----------|------|----------|
| **GPT-4.1** | Azure OpenAI | High | Complex reasoning, code generation |
| **Gemini 2.5 Pro** | Google | Medium | General tasks, analysis |

## Cost Optimization

- **Development**: Use Gemini 2.5 Pro for testing (lower cost)
- **Production**: Use GPT-4.1 for critical tasks (higher quality)
- **Monitoring**: Check usage in respective provider dashboards

## Security Notes

- Never commit API keys to version control
- Use environment variables for all sensitive data
- Regularly rotate API keys
- Monitor usage for unexpected activity