"""
Ollama interface for LLaMA 3 8B integration.
Provides LangChain-compatible LLM interface for local inference.
"""

from typing import List, Dict, Any, Optional, Iterator
import logging
import requests
import json
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import Generation, LLMResult
from pydantic import Field
from config.settings import config

logger = logging.getLogger(__name__)


class OllamaLLM(LLM):
    """
    Ollama LLM wrapper for LangChain compatibility.
    Interfaces with local Ollama server running LLaMA 3 8B.
    """
    
    model_name: str = Field(default_factory=lambda: config.llm.model_name)
    base_url: str = Field(default_factory=lambda: config.llm.base_url)
    temperature: float = Field(default_factory=lambda: config.llm.temperature)
    max_tokens: int = Field(default_factory=lambda: config.llm.max_tokens)
    timeout: int = Field(default=60)
    
    def __init__(self, **kwargs):
        """Initialize Ollama LLM."""
        super().__init__(**kwargs)
        self._verify_connection()
    
    @property
    def _llm_type(self) -> str:
        """Return identifier of llm type."""
        return "ollama"
    
    def _verify_connection(self) -> bool:
        """
        Verify connection to Ollama server.
        
        Returns:
            True if connection is successful
            
        Raises:
            ConnectionError: If unable to connect to Ollama
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            # Check if the model is available
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            if self.model_name not in model_names:
                logger.warning(f"Model {self.model_name} not found. Available models: {model_names}")
                logger.info(f"You may need to run: ollama pull {self.model_name}")
            else:
                logger.info(f"Connected to Ollama server. Model {self.model_name} is available.")
            
            return True
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to connect to Ollama server at {self.base_url}: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call the Ollama API to generate text.
        
        Args:
            prompt: Input prompt
            stop: Stop sequences
            run_manager: Callback manager
            **kwargs: Additional arguments
            
        Returns:
            Generated text
        """
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }
        
        if stop:
            payload["options"]["stop"] = stop
        
        try:
            # Make request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            generated_text = result.get("response", "")
            
            # Log token usage if available
            if "eval_count" in result:
                logger.debug(f"Generated {result['eval_count']} tokens")
            
            return generated_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Ollama response: {e}")
            raise
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Stream text generation from Ollama.
        
        Args:
            prompt: Input prompt
            stop: Stop sequences
            run_manager: Callback manager
            **kwargs: Additional arguments
            
        Yields:
            Generated text chunks
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }
        
        if stop:
            payload["options"]["stop"] = stop
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if "response" in chunk:
                            text = chunk["response"]
                            if text:
                                if run_manager:
                                    run_manager.on_llm_new_token(text)
                                yield text
                        
                        # Check if generation is done
                        if chunk.get("done", False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
                        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error streaming from Ollama API: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model_name},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting model info: {e}")
            return {}
    
    def pull_model(self) -> bool:
        """
        Pull the model if it's not available locally.
        
        Returns:
            True if successful
        """
        try:
            logger.info(f"Pulling model {self.model_name}...")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                timeout=300  # 5 minutes timeout for model download
            )
            response.raise_for_status()
            logger.info(f"Successfully pulled model {self.model_name}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error pulling model: {e}")
            return False


class OllamaRAGChain:
    """
    High-level RAG chain using Ollama LLM.
    Combines retrieval with generation using the specified prompt template.
    """
    
    def __init__(
        self,
        llm: OllamaLLM = None,
        prompt_template: str = None
    ):
        """
        Initialize RAG chain.
        
        Args:
            llm: Ollama LLM instance
            prompt_template: Template for RAG prompts
        """
        self.llm = llm or OllamaLLM()
        
        # Import here to avoid circular imports
        from llm.prompts import RAG_PROMPT_TEMPLATE
        self.prompt_template = prompt_template or RAG_PROMPT_TEMPLATE
    
    def generate_response(
        self,
        question: str,
        context_documents: List[str],
        **kwargs
    ) -> str:
        """
        Generate a response using retrieved context.
        
        Args:
            question: User question
            context_documents: List of relevant document texts
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        # Combine context documents
        context = "\n\n".join(context_documents)
        
        # Format prompt
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        # Generate response
        response = self.llm._call(prompt, **kwargs)
        
        return response.strip()
    
    def stream_response(
        self,
        question: str,
        context_documents: List[str],
        **kwargs
    ) -> Iterator[str]:
        """
        Stream a response using retrieved context.
        
        Args:
            question: User question
            context_documents: List of relevant document texts
            **kwargs: Additional generation parameters
            
        Yields:
            Response chunks
        """
        # Combine context documents
        context = "\n\n".join(context_documents)
        
        # Format prompt
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        # Stream response
        yield from self.llm._stream(prompt, **kwargs)


def create_ollama_llm(**kwargs) -> OllamaLLM:
    """
    Factory function to create an Ollama LLM instance.
    
    Args:
        **kwargs: Additional arguments for OllamaLLM
        
    Returns:
        OllamaLLM instance
    """
    return OllamaLLM(**kwargs)


def create_rag_chain(llm: OllamaLLM = None) -> OllamaRAGChain:
    """
    Factory function to create a RAG chain.
    
    Args:
        llm: Ollama LLM instance
        
    Returns:
        OllamaRAGChain instance
    """
    return OllamaRAGChain(llm=llm)


def check_ollama_status() -> Dict[str, Any]:
    """
    Check the status of the Ollama server and available models.
    
    Returns:
        Status information dictionary
    """
    base_url = config.llm.base_url
    
    try:
        # Check server status
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        response.raise_for_status()
        
        models_data = response.json()
        models = models_data.get("models", [])
        
        return {
            "server_running": True,
            "server_url": base_url,
            "available_models": [model["name"] for model in models],
            "total_models": len(models),
            "target_model": config.llm.model_name,
            "target_model_available": config.llm.model_name in [model["name"] for model in models]
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "server_running": False,
            "server_url": base_url,
            "error": str(e),
            "target_model": config.llm.model_name,
            "target_model_available": False
        }
