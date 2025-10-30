# Utility functions for the intro_cache package
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_key(model='gemini'):
    """
    Get API key(s) for the specified model from .env file
    
    Args:
        model (str): The model name ('gemini', 'claude', 'openai')
    
    Returns:
        list: List of API keys for the specified model (separated by |)
    
    Raises:
        ValueError: If the model is not supported or key is not found
    """
    key_mapping = {
        'gemini': 'GEMINI_API_KEY',
        'claude': 'CLAUDE_API_KEY', 
        'openai': 'OPENAI_API_KEY'
    }
    
    if model not in key_mapping:
        raise ValueError(f"Unsupported model: {model}. Supported models: {list(key_mapping.keys())}")
    
    env_var = key_mapping[model]
    key_string = os.getenv(env_var)
    
    if not key_string:
        raise ValueError(f"API key for {model} not found in environment variables. Please set {env_var} in your .env file.")
    
    # Split by | to support multiple keys
    keys = [k.strip() for k in key_string.split('|') if k.strip()]
    
    if not keys:
        raise ValueError(f"No valid API keys found for {model}. Please check {env_var} in your .env file.")
    
    return keys 
