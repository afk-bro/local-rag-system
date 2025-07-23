"""
Text chunking utilities for the RAG system.
Implements token-based chunking with configurable overlap.
"""

from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import tiktoken
from config.settings import config


class TokenBasedTextSplitter:
    """
    Token-based text splitter that respects token limits.
    Uses tiktoken for accurate token counting.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize the token-based text splitter.
        
        Args:
            chunk_size: Maximum tokens per chunk (default from config)
            chunk_overlap: Overlap between chunks in tokens (default from config)
            encoding_name: Tokenizer encoding to use
        """
        self.chunk_size = chunk_size or config.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunking.chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # Convert token counts to approximate character counts
        # Average ~4 characters per token for English text
        self.char_chunk_size = self.chunk_size * 4
        self.char_overlap = self.chunk_overlap * 4
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.char_chunk_size,
            chunk_overlap=self.char_overlap,
            length_function=self._token_length,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def _token_length(self, text: str) -> int:
        """Calculate the number of tokens in text."""
        return len(self.encoding.encode(text))
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on token count.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        return self.text_splitter.split_text(text)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks while preserving metadata.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of chunked Document objects
        """
        chunked_docs = []
        
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            
            for i, chunk in enumerate(chunks):
                # Create new document with chunk and enhanced metadata
                chunk_metadata = doc.metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": self._token_length(chunk),
                    "source_length": self._token_length(doc.page_content)
                })
                
                chunked_docs.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))
        
        return chunked_docs
    
    def get_chunk_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about the chunking process.
        
        Args:
            documents: List of chunked documents
            
        Returns:
            Dictionary with chunking statistics
        """
        if not documents:
            return {}
        
        chunk_sizes = [self._token_length(doc.page_content) for doc in documents]
        
        return {
            "total_chunks": len(documents),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "total_tokens": sum(chunk_sizes),
            "sources": len(set(doc.metadata.get("source", "") for doc in documents))
        }


class SemanticTextSplitter:
    """
    Semantic text splitter that tries to preserve meaning boundaries.
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """Initialize semantic text splitter."""
        self.chunk_size = chunk_size or config.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunking.chunk_overlap
        
        # Semantic separators in order of preference
        self.separators = [
            "\n\n\n",  # Multiple line breaks
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentence endings
            "! ",      # Exclamation sentences
            "? ",      # Question sentences
            "; ",      # Semicolons
            ", ",      # Commas
            " ",       # Spaces
            ""         # Character level
        ]
        
        self.base_splitter = TokenBasedTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Override separators for semantic splitting
        self.base_splitter.text_splitter.separators = self.separators
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using semantic boundaries."""
        return self.base_splitter.split_documents(documents)


def create_text_splitter(strategy: str = "token") -> TokenBasedTextSplitter:
    """
    Factory function to create text splitters.
    
    Args:
        strategy: Splitting strategy ("token" or "semantic")
        
    Returns:
        Text splitter instance
    """
    if strategy == "semantic":
        return SemanticTextSplitter()
    else:
        return TokenBasedTextSplitter()
