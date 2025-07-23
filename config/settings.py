"""
Configuration management for the RAG system using Pydantic.
"""

from typing import Optional
from pydantic import BaseModel, Field
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class EmbeddingConfig(BaseModel):
    """Configuration for embedding model."""
    model_name: str = Field(default="BAAI/bge-base-en-v1.5", description="Embedding model name")
    device: str = Field(default="cpu", description="Device to run embeddings on")
    batch_size: int = Field(default=32, description="Batch size for embedding generation")


class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""
    chunk_size: int = Field(default=512, description="Chunk size in tokens")
    chunk_overlap: int = Field(default=64, description="Overlap between chunks in tokens")
    max_overlap: int = Field(default=128, description="Maximum overlap between chunks")


class VectorStoreConfig(BaseModel):
    """Configuration for FAISS vector store."""
    index_path: str = Field(default="./vectorstore/faiss_index", description="Path to FAISS index")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold for retrieval")
    top_k: int = Field(default=5, description="Number of top documents to retrieve")


class LLMConfig(BaseModel):
    """Configuration for LLM (Ollama)."""
    model_name: str = Field(default="llama3:8b", description="Ollama model name")
    base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    temperature: float = Field(default=0.1, description="Temperature for generation")
    max_tokens: int = Field(default=1000, description="Maximum tokens to generate")


class RAGConfig(BaseModel):
    """Main RAG system configuration."""
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    
    # Document processing
    supported_formats: list[str] = Field(
        default=["pdf", "txt", "md", "docx", "html", "json", "csv"],
        description="Supported document formats"
    )
    
    # System settings
    log_level: str = Field(default="INFO", description="Logging level")
    data_dir: str = Field(default="./data", description="Directory for storing documents")
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "RAG_"
        case_sensitive = False


def get_config() -> RAGConfig:
    """Get the RAG configuration instance."""
    return RAGConfig()


# Global configuration instance
config = get_config()
