"""
Embedding pipeline using sentence-transformers with bge-base-en-v1.5 model.
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
import pickle
from pathlib import Path
import hashlib
from config.settings import config

logger = logging.getLogger(__name__)


class BGEEmbeddings(Embeddings):
    """
    BGE (BAAI General Embedding) model wrapper for LangChain compatibility.
    Uses sentence-transformers to load the bge-base-en-v1.5 model.
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        batch_size: int = None,
        cache_dir: str = "./embeddings_cache"
    ):
        """
        Initialize BGE embeddings.
        
        Args:
            model_name: Name of the BGE model to use
            device: Device to run the model on ('cpu' or 'cuda')
            batch_size: Batch size for embedding generation
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name or config.embedding.model_name
        self.device = device or config.embedding.device
        self.batch_size = batch_size or config.embedding.batch_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize the model
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.info(f"Embedding {len(texts)} documents")
        
        # Check cache first
        cached_embeddings = self._get_cached_embeddings(texts)
        if cached_embeddings:
            return cached_embeddings
        
        # Generate embeddings in batches
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=True if len(texts) > 10 else False
            )
            embeddings.extend(batch_embeddings.tolist())
        
        # Cache the embeddings
        self._cache_embeddings(texts, embeddings)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()
    
    def _get_text_hash(self, text: str) -> str:
        """Generate a hash for text content."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_cached_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Retrieve cached embeddings if available.
        
        Args:
            texts: List of texts to check for cached embeddings
            
        Returns:
            Cached embeddings if all texts are cached, None otherwise
        """
        cache_file = self.cache_dir / f"embeddings_{self.model_name.replace('/', '_')}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            
            # Check if all texts are in cache
            text_hashes = [self._get_text_hash(text) for text in texts]
            if all(hash_val in cache for hash_val in text_hashes):
                logger.info(f"Found cached embeddings for {len(texts)} texts")
                return [cache[hash_val] for hash_val in text_hashes]
        
        except Exception as e:
            logger.warning(f"Error loading embedding cache: {e}")
        
        return None
    
    def _cache_embeddings(self, texts: List[str], embeddings: List[List[float]]):
        """
        Cache embeddings for future use.
        
        Args:
            texts: List of texts
            embeddings: Corresponding embeddings
        """
        cache_file = self.cache_dir / f"embeddings_{self.model_name.replace('/', '_')}.pkl"
        
        # Load existing cache
        cache = {}
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cache = pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading existing cache: {e}")
        
        # Add new embeddings to cache
        for text, embedding in zip(texts, embeddings):
            text_hash = self._get_text_hash(text)
            cache[text_hash] = embedding
        
        # Save updated cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)
            logger.info(f"Cached {len(texts)} new embeddings")
        except Exception as e:
            logger.warning(f"Error saving embedding cache: {e}")
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding model."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_sequence_length": self.model.max_seq_length
        }


class EmbeddingPipeline:
    """
    High-level embedding pipeline for document processing.
    """
    
    def __init__(self, embeddings: BGEEmbeddings = None):
        """
        Initialize the embedding pipeline.
        
        Args:
            embeddings: BGE embeddings instance
        """
        self.embeddings = embeddings or BGEEmbeddings()
    
    def process_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Process documents and generate embeddings.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary containing documents, embeddings, and metadata
        """
        if not documents:
            return {"documents": [], "embeddings": [], "metadata": {}}
        
        logger.info(f"Processing {len(documents)} documents for embedding")
        
        # Extract text content
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(texts)
        
        # Prepare metadata
        metadata = {
            "total_documents": len(documents),
            "embedding_stats": self.embeddings.get_embedding_stats(),
            "avg_text_length": sum(len(text) for text in texts) / len(texts),
            "sources": list(set(doc.metadata.get("source", "unknown") for doc in documents))
        }
        
        return {
            "documents": documents,
            "embeddings": embeddings,
            "metadata": metadata
        }
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
        """
        return self.embeddings.embed_query(query)
    
    def similarity_search(
        self,
        query_embedding: List[float],
        document_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[int]:
        """
        Find most similar documents using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: List of document embedding vectors
            top_k: Number of top results to return
            
        Returns:
            List of indices of most similar documents
        """
        if not document_embeddings:
            return []
        
        # Convert to numpy arrays
        query_vec = np.array(query_embedding)
        doc_vecs = np.array(document_embeddings)
        
        # Calculate cosine similarities
        similarities = np.dot(doc_vecs, query_vec) / (
            np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return top_indices.tolist()


def create_embedding_pipeline() -> EmbeddingPipeline:
    """Factory function to create an embedding pipeline."""
    return EmbeddingPipeline()


def create_bge_embeddings() -> BGEEmbeddings:
    """Factory function to create BGE embeddings."""
    return BGEEmbeddings()
