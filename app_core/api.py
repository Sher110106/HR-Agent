import os
import logging
from typing import List, Dict, Any, Tuple
from openai import AzureOpenAI

from utils.metrics import api_call_timer, code_execution_timer, record_error
from utils.circuit_breaker import (
    llm_api_breaker, code_execution_breaker, CircuitBreakerError
)

logger = logging.getLogger(__name__)

# Initialize client
api_key = os.environ.get("Azure_Key")
if not api_key:
    raise EnvironmentError("Azure_Key environment variable not set. Please export Azure_Key with your Azure OpenAI API key.")

AZURE_DEPLOYMENT_NAME = "gpt-4.1"  # Default Azure OpenAI deployment
AZURE_ENDPOINT = "https://ai-sherpartap11019601ai587562462851.openai.azure.com"  # No trailing slash after domain
AZURE_API_VERSION = "2025-01-01-preview"

# Use AzureOpenAI client which knows how to build resource URLs internally
client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=api_key,
    api_version=AZURE_API_VERSION
)

def make_llm_call(messages: List[Dict], model: str = AZURE_DEPLOYMENT_NAME, 
                  temperature: float = 0.2, max_tokens: int = 4000, stream: bool = False):
    """Make LLM API call with circuit breaker protection and metrics tracking."""
    
    def api_call():
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
    
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
            else:
                if hasattr(response, 'usage') and response.usage:
                    total_tokens = response.usage.total_tokens
                else:
                    total_tokens = prompt_tokens + (max_tokens // 4)  # Rough estimate
                
                timer.set_tokens_used(total_tokens)
                logger.debug(f"🤖 LLM call successful: {total_tokens} tokens, model: {model}")
            
            return response
            
    except CircuitBreakerError as e:
        record_error("circuit_breaker_open", {"model": model, "error": str(e)})
        logger.error(f"🚫 Circuit breaker blocked LLM call: {e}")
        raise
    except Exception as e:
        record_error("llm_api_error", {"model": model, "error": str(e)})
        logger.error(f"❌ LLM API call failed: {e}")
        raise

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