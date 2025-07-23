"""
RAG orchestrator that coordinates document ingestion, retrieval, and generation.
Main pipeline for the RAG system.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
from langchain.schema import Document

# Import components
from config.settings import config
from ingest.document_loader import create_document_loader
from ingest.text_chunker import create_text_splitter
from embed.embedding_pipeline import create_bge_embeddings
from vectorstore.faiss_store import create_faiss_store
from llm.ollama_interface import create_ollama_llm, create_rag_chain

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """
    Main RAG orchestrator that coordinates all components.
    Provides high-level interface for document ingestion and querying.
    """
    
    def __init__(
        self,
        index_path: str = None,
        embedding_model: str = None,
        llm_model: str = None
    ):
        """
        Initialize RAG orchestrator.
        
        Args:
            index_path: Path for FAISS index storage
            embedding_model: Embedding model name
            llm_model: LLM model name
        """
        self.index_path = index_path or config.vectorstore.index_path
        
        # Initialize components
        logger.info("Initializing RAG components...")
        
        # Document processing
        self.document_loader = create_document_loader()
        self.text_splitter = create_text_splitter()
        
        # Embeddings and vector store
        self.embeddings = create_bge_embeddings()
        if embedding_model:
            self.embeddings.model_name = embedding_model
        
        self.vector_store = create_faiss_store(embeddings=self.embeddings)
        
        # LLM and RAG chain
        self.llm = create_ollama_llm()
        if llm_model:
            self.llm.model_name = llm_model
        
        self.rag_chain = create_rag_chain(llm=self.llm)
        
        logger.info("RAG orchestrator initialized successfully")
    
    def ingest_documents(
        self,
        file_paths: List[str] = None,
        directory_path: str = None,
        recursive: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest documents into the RAG system.
        
        Args:
            file_paths: List of specific file paths to ingest
            directory_path: Directory path to ingest all supported files
            recursive: Whether to search directory recursively
            
        Returns:
            Ingestion results and statistics
        """
        logger.info("Starting document ingestion...")
        
        documents = []
        
        # Load documents from file paths
        if file_paths:
            for file_path in file_paths:
                try:
                    docs = self.document_loader.load_document(file_path)
                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents from {file_path}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
        
        # Load documents from directory
        if directory_path:
            try:
                docs = self.document_loader.load_directory(directory_path, recursive=recursive)
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {directory_path}")
            except Exception as e:
                logger.error(f"Failed to load from directory {directory_path}: {e}")
        
        if not documents:
            logger.warning("No documents loaded")
            return {"status": "no_documents", "total_documents": 0}
        
        # Chunk documents
        logger.info(f"Chunking {len(documents)} documents...")
        chunked_documents = self.text_splitter.split_documents(documents)
        chunk_stats = self.text_splitter.get_chunk_stats(chunked_documents)
        
        logger.info(f"Created {len(chunked_documents)} chunks")
        
        # Add to vector store
        logger.info("Adding documents to vector store...")
        doc_ids = self.vector_store.add_documents(chunked_documents)
        
        # Save index
        self.vector_store.save_index()
        
        # Prepare results
        results = {
            "status": "success",
            "total_documents": len(documents),
            "total_chunks": len(chunked_documents),
            "document_ids": doc_ids,
            "chunk_stats": chunk_stats,
            "vector_store_stats": self.vector_store.get_stats()
        }
        
        logger.info(f"Successfully ingested {len(documents)} documents into {len(chunked_documents)} chunks")
        return results
    
    def ingest_youtube_transcript(self, video_url: str, language: str = 'en') -> Dict[str, Any]:
        """
        Ingest YouTube video transcript.
        
        Args:
            video_url: YouTube video URL
            language: Language code for transcript
            
        Returns:
            Ingestion results
        """
        logger.info(f"Ingesting YouTube transcript from {video_url}")
        
        try:
            # Load transcript
            documents = self.document_loader.load_youtube_transcript(video_url, language)
            
            # Chunk documents
            chunked_documents = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            doc_ids = self.vector_store.add_documents(chunked_documents)
            
            # Save index
            self.vector_store.save_index()
            
            results = {
                "status": "success",
                "video_url": video_url,
                "total_chunks": len(chunked_documents),
                "document_ids": doc_ids
            }
            
            logger.info(f"Successfully ingested YouTube transcript: {len(chunked_documents)} chunks")
            return results
            
        except Exception as e:
            logger.error(f"Failed to ingest YouTube transcript: {e}")
            return {"status": "error", "error": str(e)}
    
    def query(
        self,
        question: str,
        top_k: int = None,
        score_threshold: float = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system with score threshold filtering.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
            return_sources: Whether to return source documents
            
        Returns:
            Query response with answer and sources
        """
        logger.info(f"Processing query: {question}")

        # Use default threshold if not provided
        if score_threshold is None:
            score_threshold = config.vectorstore.similarity_threshold

        # Retrieve top_k results with scores
        results = self.vector_store.similarity_search_with_score(
            query=question,
            k=top_k or config.vectorstore.top_k
        )

        # Filter based on similarity score
        retrieved_docs = [
            doc for doc, score in results if score >= score_threshold
        ]

        if not retrieved_docs:
            logger.warning("No relevant documents found")
            return {
                "answer": "The answer is not in the provided documents.",
                "sources": [],
                "retrieved_documents": 0
            }

        logger.info(f"Retrieved {len(retrieved_docs)} documents above score threshold")

        # Extract context for generation
        context_texts = [doc.page_content for doc in retrieved_docs]

        # Generate answer
        answer = self.rag_chain.generate_response(
            question=question,
            context_documents=context_texts
        )

        response = {
            "answer": answer,
            "retrieved_documents": len(retrieved_docs)
        }

        if return_sources:
            sources = []
            for doc, score in results:
                if score >= score_threshold:
                    sources.append({
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": score
                    })
            response["sources"] = sources

        logger.info("Query processed successfully")
        return response
    
    def stream_query(
        self,
        question: str,
        top_k: int = None,
        score_threshold: float = None
    ):
        """
        Stream query response.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score
            
        Yields:
            Response chunks
        """
        logger.info(f"Streaming query: {question}")
        
        # Use default threshold if not provided
        if score_threshold is None:
            score_threshold = config.vectorstore.similarity_threshold
        
        # Retrieve relevant documents with scores
        results = self.vector_store.similarity_search_with_score(
            query=question,
            k=top_k or config.vectorstore.top_k
        )
        
        # Filter based on similarity score
        retrieved_docs = [
            doc for doc, score in results if score >= score_threshold
        ]
        
        if not retrieved_docs:
            yield "The answer is not in the provided documents."
            return
        
        # Extract context
        context_texts = [doc.page_content for doc in retrieved_docs]
        
        # Stream response
        yield from self.rag_chain.stream_response(
            question=question,
            context_documents=context_texts
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            System status information
        """
        from llm.ollama_interface import check_ollama_status
        
        return {
            "vector_store": self.vector_store.get_stats(),
            "embedding_model": self.embeddings.get_embedding_stats(),
            "ollama_status": check_ollama_status(),
            "configuration": {
                "chunk_size": config.chunking.chunk_size,
                "chunk_overlap": config.chunking.chunk_overlap,
                "top_k": config.vectorstore.top_k,
                "similarity_threshold": config.vectorstore.similarity_threshold
            }
        }
    
    def reset_index(self) -> bool:
        """
        Reset the vector store index.
        
        Returns:
            True if successful
        """
        try:
            # Clear vector store
            self.vector_store.documents = []
            self.vector_store.vectorstore = None
            
            # Remove index files
            index_path = Path(self.index_path)
            for suffix in ['.faiss', '.pkl']:
                file_path = index_path.with_suffix(suffix)
                if file_path.exists():
                    file_path.unlink()
            
            logger.info("Vector store index reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting index: {e}")
            return False
    
    def add_documents_from_text(self, texts: List[str], metadatas: List[Dict] = None) -> Dict[str, Any]:
        """
        Add documents directly from text strings.
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            Addition results
        """
        if not texts:
            return {"status": "no_texts", "total_documents": 0}
        
        # Create Document objects
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            metadata.setdefault("source", f"text_{i}")
            
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        
        # Chunk documents
        chunked_documents = self.text_splitter.split_documents(documents)
        
        # Add to vector store
        doc_ids = self.vector_store.add_documents(chunked_documents)
        
        # Save index
        self.vector_store.save_index()
        
        return {
            "status": "success",
            "total_documents": len(documents),
            "total_chunks": len(chunked_documents),
            "document_ids": doc_ids
        }


def create_rag_orchestrator(**kwargs) -> RAGOrchestrator:
    """
    Factory function to create a RAG orchestrator.
    
    Args:
        **kwargs: Additional arguments for RAGOrchestrator
        
    Returns:
        RAGOrchestrator instance
    """
    return RAGOrchestrator(**kwargs)
