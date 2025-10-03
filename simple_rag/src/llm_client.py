"""
LLM Client Module

This module provides a unified interface for different LLM providers (OpenAI, Ollama).
It abstracts the differences between various LLM APIs to provide a consistent interface.
"""

import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from config import config
from logging_config import get_logger

logger = get_logger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response using the LLM."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI LLM client."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """Initialize OpenAI client."""
        try:
            import openai
            self.client = openai
            self.client.api_key = api_key
            self.model = model
            logger.info(f"Initialized OpenAI client with model: {model}")
        except ImportError:
            raise ImportError("OpenAI library not found. Install with: pip install openai")
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', config.MAX_TOKENS),
                temperature=kwargs.get('temperature', config.TEMPERATURE)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        return {
            'provider': 'openai',
            'model': self.model,
            'type': 'cloud',
            'max_tokens': config.MAX_TOKENS
        }


class OllamaClient(LLMClient):
    """Ollama LLM client."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "deepseek-r1"):
        """Initialize Ollama client."""
        try:
            import ollama
            self.client = ollama.Client(host=base_url)
            self.model = model
            self.base_url = base_url
            logger.info(f"Initialized Ollama client with model: {model} at {base_url}")
            
            # Test connection and model availability
            self._test_connection()
            
        except ImportError:
            raise ImportError("Ollama library not found. Install with: pip install ollama")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {str(e)}")
            raise ConnectionError(f"Could not connect to Ollama at {base_url}: {str(e)}")
    
    def _test_connection(self):
        """Test connection to Ollama server and model availability."""
        try:
            # Check if Ollama is running
            models_response = self.client.list()
            
            # Handle different response formats
            if isinstance(models_response, dict) and 'models' in models_response:
                models_list = models_response['models']
            else:
                models_list = models_response
            
            # Extract model names safely
            available_models = []
            for model in models_list:
                if isinstance(model, dict):
                    if 'name' in model:
                        available_models.append(model['name'])
                    elif 'model' in model:
                        available_models.append(model['model'])
                else:
                    available_models.append(str(model))
            
            logger.info(f"Connected to Ollama. Available models: {available_models}")
            
            # Check if our target model is available
            if not any(self.model in model_name for model_name in available_models):
                logger.warning(f"Model '{self.model}' not found in available models. "
                             f"Available models: {available_models}")
                self._ensure_model_available()
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            raise
    
    def _ensure_model_available(self):
        """Ensure the model is available, pull if necessary."""
        logger.info(f"Attempting to pull model '{self.model}'...")
        try:
            self.client.pull(self.model)
            logger.info(f"Successfully pulled model '{self.model}'")
        except Exception as e:
            logger.error(f"Failed to pull model '{self.model}': {str(e)}")
            raise
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Ollama API."""
        try:
            # Convert OpenAI-style messages to Ollama format
            prompt = self._convert_messages_to_prompt(messages)
            
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': kwargs.get('temperature', config.TEMPERATURE),
                    'num_predict': kwargs.get('max_tokens', config.MAX_TOKENS)
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            # If the error is about model not found, try to pull it
            if "model" in str(e).lower() and "not found" in str(e).lower():
                self._ensure_model_available()
                # Retry the request
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        'temperature': kwargs.get('temperature', config.TEMPERATURE),
                        'num_predict': kwargs.get('max_tokens', config.MAX_TOKENS)
                    }
                )
                return response['response'].strip()
            raise
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to a single prompt for Ollama."""
        prompt_parts = []
        
        for message in messages:
            role = message['role']
            content = message['content']
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        # Add a final "Assistant:" to prompt the model to respond
        prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
        return prompt
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Ollama model information."""
        try:
            model_info = self.client.show(self.model)
            return {
                'provider': 'ollama',
                'model': self.model,
                'type': 'local',
                'base_url': self.base_url,
                'model_info': model_info
            }
        except Exception as e:
            logger.warning(f"Could not get model info: {str(e)}")
            return {
                'provider': 'ollama',
                'model': self.model,
                'type': 'local',
                'base_url': self.base_url,
                'error': str(e)
            }


class LLMFactory:
    """Factory class to create LLM clients based on configuration."""
    
    @staticmethod
    def create_client() -> LLMClient:
        """Create an LLM client based on the configuration."""
        provider = config.LLM_PROVIDER.lower()
        
        if provider == "openai":
            if not config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
            return OpenAIClient(config.OPENAI_API_KEY, config.LLM_MODEL)
        
        elif provider == "ollama":
            return OllamaClient(config.OLLAMA_BASE_URL, config.LLM_MODEL)
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. "
                           f"Supported providers: 'openai', 'ollama'")

