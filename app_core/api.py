import os
import logging
from typing import List, Dict, Any, Tuple
import time
from openai import AzureOpenAI
import google.generativeai as genai

from utils.metrics import api_call_timer, code_execution_timer, record_error
from utils.circuit_breaker import (
    llm_api_breaker, code_execution_breaker, CircuitBreakerError
)
from env_config import (
    AZURE_API_KEY, AZURE_ENDPOINT, AZURE_API_VERSION, AZURE_DEPLOYMENT_NAME,
    GEMINI_API_KEY, _load_secrets_to_env
)

logger = logging.getLogger(__name__)

# Model configuration
SUPPORTED_MODELS = {
    "gpt-4.1": "azure_openai",
    "models/gemini-2.5-flash": "google_genai"
}

# Global client variables
azure_client = None
gemini_configured = False

def initialize_clients():
    """Initialize API clients with current configuration."""
    global azure_client, gemini_configured
    
    # Load secrets into environment variables
    _load_secrets_to_env()
    
    # Re-import configuration after secrets are loaded
    from env_config import (
        AZURE_API_KEY, AZURE_ENDPOINT, AZURE_API_VERSION, AZURE_DEPLOYMENT_NAME,
        GEMINI_API_KEY
    )
    
    # Initialize Azure OpenAI client
    if AZURE_API_KEY:
        try:
            azure_client = AzureOpenAI(
                azure_endpoint=AZURE_ENDPOINT,
                api_key=AZURE_API_KEY,
                api_version=AZURE_API_VERSION
            )
            logger.info("✅ Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure OpenAI client: {e}")
    else:
        logger.warning("AZURE_API_KEY not available. Azure OpenAI will not be available.")

    # Initialize Google Generative AI client
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            # Perform a lightweight validation call to ensure the key/model works
            try:
                test_model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
                _ = test_model.generate_content(
                    "ping",
                    generation_config=genai.types.GenerationConfig(max_output_tokens=1, temperature=0)
                )
                gemini_configured = True
                logger.info("✅ Google Gemini client initialized successfully")
            except Exception as ge:
                # Treat HTTP 500 as transient; keep Gemini enabled but warn
                gemini_configured = True
                logger.error(f"❌ Google Gemini validation failed: {ge}")
                logger.warning("⚠️ Proceeding with Gemini enabled (validation failed, likely transient 5xx). Calls will retry with backoff.")
        except Exception as e:
            gemini_configured = False
            logger.error(f"❌ Failed to initialize Google Gemini client: {e}")
    else:
        logger.warning("GEMINI_API_KEY not available. Gemini will not be available.")

# Initialize clients on module load
initialize_clients()

def _make_azure_openai_call(messages: List[Dict], model: str, temperature: float, max_tokens: int, stream: bool):
    """Make Azure OpenAI API call."""
    if not azure_client:
        # Try to reinitialize clients in case secrets were loaded later
        initialize_clients()
        if not azure_client:
            raise EnvironmentError("Azure OpenAI client not initialized. Please configure Azure API key in Streamlit secrets.")
    
    return azure_client.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream
    )

def _make_gemini_call(messages: List[Dict], model: str, temperature: float, max_tokens: int, stream: bool):
    """Make Google Gemini API call."""
    try:
        # Convert OpenAI format messages to Gemini format
        gemini_messages = []
        system_instruction = None

        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')

            if role == 'system':
                system_instruction = content
            elif role == 'user':
                gemini_messages.append({
                    'role': 'user',
                    'parts': [{'text': content}]
                })
            elif role == 'assistant':
                gemini_messages.append({
                    'role': 'model',
                    'parts': [{'text': content}]
                })

        # Create Gemini model
        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_instruction
        )

        # Generate content
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # If we only have a single user message, pass it directly as a string for simpler handling
        if len(gemini_messages) == 1 and gemini_messages[0]['role'] == 'user':
            content_to_send = gemini_messages[0]['parts'][0]['text']
        else:
            content_to_send = gemini_messages

        # Exponential backoff for transient errors (e.g., 500)
        max_attempts = 3
        delay_seconds = 1.0
        attempt = 0
        while True:
            attempt += 1
            try:
                if stream:
                    response = gemini_model.generate_content(
                        content_to_send,
                        generation_config=generation_config,
                        stream=True
                    )
                else:
                    response = gemini_model.generate_content(
                        content_to_send,
                        generation_config=generation_config
                    )
                return response
            except Exception as inner_e:
                # Retry on likely transient server errors
                is_transient = any(s in str(inner_e).lower() for s in [
                    "internal error", "500", "temporarily unavailable", "deadline exceeded", "unavailable"
                ])
                if attempt < max_attempts and is_transient:
                    logger.warning(f"⚠️ Gemini attempt {attempt} failed (transient). Retrying in {delay_seconds:.1f}s: {inner_e}")
                    time.sleep(delay_seconds)
                    delay_seconds *= 2
                    continue
                raise

    except Exception as e:
        logger.error(f"❌ Gemini call failed (model={model}): {e}")
        raise

def make_llm_call(messages: List[Dict], model: str = None, 
                  temperature: float = 0.2, max_tokens: int = 4000, stream: bool = False):
    """Make LLM API call with circuit breaker protection and metrics tracking."""
    
    # If no model specified, try to get from session state (for Streamlit apps)
    if model is None:
        try:
            import streamlit as st
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'selected_model'):
                model = st.session_state.selected_model
                logger.debug(f"🎯 Using model from session state: {model}")
            else:
                model = AZURE_DEPLOYMENT_NAME  # fallback to default
                logger.debug(f"🎯 Using default model: {model}")
        except ImportError:
            model = AZURE_DEPLOYMENT_NAME  # fallback if not in Streamlit context
            logger.debug(f"🎯 Using default model (no Streamlit): {model}")
    
    # Determine which provider to use
    provider = SUPPORTED_MODELS.get(model)
    if not provider:
        raise ValueError(f"Unsupported model: {model}. Supported models: {list(SUPPORTED_MODELS.keys())}")
    
    def api_call():
        if provider == "azure_openai":
            return _make_azure_openai_call(messages, model, temperature, max_tokens, stream)
        elif provider == "google_genai":
            return _make_gemini_call(messages, model, temperature, max_tokens, stream)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    try:
        with api_call_timer() as timer:
            response = llm_api_breaker.call(api_call)
            
            # Calculate token usage (approximate)
            prompt_tokens = sum(len(str(msg.get('content', ''))) for msg in messages) // 4  # Rough estimate
            
            if stream:
                # For streaming responses, we can't get exact tokens until complete
                # Set an estimated token count that will be updated when streaming finishes
                timer.set_tokens_used(prompt_tokens)
                logger.debug(f"🤖 LLM streaming call started: ~{prompt_tokens} prompt tokens, model: {model}")
                
                # For Gemini streaming, we need to wrap the response
                if provider == "google_genai":
                    return _wrap_gemini_streaming_response(response)
                return response
            else:
                # Handle token counting based on provider
                if provider == "azure_openai":
                    if hasattr(response, 'usage') and response.usage:
                        total_tokens = response.usage.total_tokens
                    else:
                        total_tokens = prompt_tokens + (max_tokens // 4)  # Rough estimate
                elif provider == "google_genai":
                    # Gemini doesn't provide token usage in the same way
                    response_text = getattr(response, 'text', '')
                    total_tokens = prompt_tokens + len(response_text) // 4  # Rough estimate
                else:
                    total_tokens = prompt_tokens + (max_tokens // 4)  # Rough estimate
                
                timer.set_tokens_used(total_tokens)
                logger.debug(f"🤖 LLM call successful: {total_tokens} tokens, model: {model}")
                
                # For Gemini, we need to wrap the response to match OpenAI format
                if provider == "google_genai":
                    return _wrap_gemini_response(response)
                return response
            
    except CircuitBreakerError as e:
        record_error("circuit_breaker_open", {"model": model, "error": str(e)})
        logger.error(f"🚫 Circuit breaker blocked LLM call: {e}")
        raise
    except Exception as e:
        record_error("llm_api_error", {"model": model, "error": str(e)})
        logger.error(f"❌ LLM API call failed: {e}")
        raise

def _wrap_gemini_response(gemini_response):
    """Wrap Gemini response to match OpenAI format."""
    class MockChoice:
        def __init__(self, text):
            self.message = MockMessage(text)
    
    class MockMessage:
        def __init__(self, text):
            self.content = text
    
    class MockResponse:
        def __init__(self, text):
            self.choices = [MockChoice(text)]
    
    # Get text from Gemini response safely
    response_text = getattr(gemini_response, 'text', '')
    return MockResponse(response_text)

def _wrap_gemini_streaming_response(gemini_stream):
    """Wrap Gemini streaming response to match OpenAI format."""
    class MockStreamChoice:
        def __init__(self, text):
            self.delta = MockDelta(text)
    
    class MockDelta:
        def __init__(self, text):
            self.content = text
    
    class MockStreamResponse:
        def __init__(self, text):
            self.choices = [MockStreamChoice(text)]
    
    for chunk in gemini_stream:
        if chunk.text:
            yield MockStreamResponse(chunk.text)

def get_available_models():
    """Get list of available models with their display names."""
    available = {}

    # Ensure clients are initialized if this is called early in a rerun
    # This helps the UI sidebar reflect availability after secrets load
    if not azure_client and not gemini_configured:
        try:
            initialize_clients()
        except Exception:
            pass
    
    # Check Azure OpenAI availability
    if azure_client:
        available["gpt-4.1"] = "GPT-4.1 (Azure OpenAI)"
    
    # Check Gemini availability
    if gemini_configured:
        available["models/gemini-2.5-flash"] = "Gemini 2.5 Flash (Google)"
    
    return available

def execute_code_safely(code: str, local_vars: Dict[str, Any]) -> Tuple[bool, Any, str]:
    """Execute code with circuit breaker protection and metrics tracking."""
    
    def code_execution():
        exec(code, {}, local_vars)
        return local_vars.get('result')
    
    try:
        with code_execution_timer() as timer:
            result = code_execution_breaker.call(code_execution)
            timer.metadata['code_length'] = len(code)
            logger.debug(f"✅ Code execution successful: {len(code)} chars")
            return True, result, ""
            
    except CircuitBreakerError as e:
        record_error("circuit_breaker_open", {"operation": "code_execution", "error": str(e)})
        logger.error(f"🚫 Circuit breaker blocked code execution: {e}")
        return False, None, f"Circuit breaker is open: {e}"
    except Exception as e:
        record_error("code_execution_error", {"error": str(e), "code_length": len(code)})
        logger.error(f"❌ Code execution failed: {e}")
        return False, None, str(e) 