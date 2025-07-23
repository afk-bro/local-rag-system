"""
Main entry point for the Local RAG System.
Demonstrates minimal working flow and provides CLI access.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rag.orchestrator import create_rag_orchestrator
from cli.interface import cli
from config.settings import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def demo_workflow():
    """
    Demonstrate the complete RAG workflow with sample data.
    This function shows how to use the system programmatically.
    """
    print("🚀 Local RAG System Demo")
    print("=" * 50)
    
    try:
        # Initialize the RAG orchestrator
        print("\n1. Initializing RAG system...")
        rag = create_rag_orchestrator()
        print("✅ RAG system initialized successfully!")
        
        # Check system status
        print("\n2. Checking system status...")
        status = rag.get_system_status()
        
        print(f"   📊 Vector Store: {status['vector_store']['total_documents']} documents")
        print(f"   🧠 Embedding Model: {status['embedding_model']['model_name']}")
        print(f"   🤖 LLM Status: {'✅ Connected' if status['ollama_status']['server_running'] else '❌ Disconnected'}")
        
        # Sample documents for demonstration
        sample_texts = [
            """
            Artificial Intelligence (AI) is a branch of computer science that aims to create 
            intelligent machines that work and react like humans. Some of the activities 
            computers with artificial intelligence are designed for include speech recognition, 
            learning, planning, and problem solving.
            """,
            """
            Machine Learning is a subset of artificial intelligence (AI) that provides systems 
            the ability to automatically learn and improve from experience without being 
            explicitly programmed. Machine learning focuses on the development of computer 
            programs that can access data and use it to learn for themselves.
            """,
            """
            Natural Language Processing (NLP) is a subfield of linguistics, computer science, 
            and artificial intelligence concerned with the interactions between computers and 
            human language, in particular how to program computers to process and analyze 
            large amounts of natural language data.
            """,
            """
            Deep Learning is part of a broader family of machine learning methods based on 
            artificial neural networks with representation learning. Learning can be supervised, 
            semi-supervised or unsupervised. Deep learning architectures such as deep neural 
            networks, deep belief networks, and recurrent neural networks have been applied 
            to fields including computer vision, speech recognition, natural language processing, 
            and bioinformatics.
            """
        ]
        
        # Add sample documents if vector store is empty
        if status['vector_store']['total_documents'] == 0:
            print("\n3. Adding sample documents...")
            
            metadatas = [
                {"source": "ai_overview.txt", "topic": "artificial_intelligence"},
                {"source": "ml_basics.txt", "topic": "machine_learning"},
                {"source": "nlp_intro.txt", "topic": "natural_language_processing"},
                {"source": "dl_guide.txt", "topic": "deep_learning"}
            ]
            
            result = rag.add_documents_from_text(sample_texts, metadatas)
            print(f"✅ Added {result['total_documents']} documents ({result['total_chunks']} chunks)")
        else:
            print(f"\n3. Using existing documents ({status['vector_store']['total_documents']} documents)")
        
        # Demonstrate querying
        print("\n4. Demonstrating queries...")
        
        sample_questions = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are the applications of deep learning?",
            "What is the relationship between AI and NLP?"
        ]
        
        for i, question in enumerate(sample_questions, 1):
            print(f"\n   Query {i}: {question}")
            
            try:
                response = rag.query(question, return_sources=False)
                print(f"   Answer: {response['answer'][:200]}...")
                print(f"   Retrieved: {response['retrieved_documents']} documents")
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
        
        print("\n5. Demo completed successfully! 🎉")
        print("\nTo use the CLI interface, run:")
        print("   python main.py --help")
        print("\nExample CLI commands:")
        print("   python main.py ingest --path ./documents")
        print("   python main.py query 'What is machine learning?'")
        print("   python main.py status")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        logger.error(f"Demo error: {str(e)}", exc_info=True)


def check_dependencies():
    """
    Check if all required dependencies are available.
    """
    missing_deps = []
    
    try:
        import sentence_transformers
    except ImportError:
        missing_deps.append("sentence-transformers")
    
    try:
        import faiss
    except ImportError:
        missing_deps.append("faiss-cpu")
    
    try:
        import langchain
    except ImportError:
        missing_deps.append("langchain")
    
    try:
        import click
    except ImportError:
        missing_deps.append("click")
    
    try:
        import rich
    except ImportError:
        missing_deps.append("rich")
    
    if missing_deps:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nPlease install missing dependencies:")
        print("   pip install -r requirements.txt")
        return False
    
    return True


def main():
    """
    Main entry point.
    If no CLI arguments are provided, run the demo.
    Otherwise, use the CLI interface.
    """
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # If no arguments provided, run demo
    if len(sys.argv) == 1:
        demo_workflow()
    else:
        # Use CLI interface
        cli()


if __name__ == "__main__":
    main()
