# Local RAG System

A modular, production-ready Retrieval-Augmented Generation (RAG) system using Python with fully local inference.

## Features

- **🤖 Local LLM**: LLaMA 3 8B via Ollama (no cloud APIs)
- **🧠 Embeddings**: BGE-base-en-v1.5 via sentence-transformers
- **🗄️ Vector Database**: FAISS for efficient similarity search
- **📚 Multi-format Support**: PDF, DOCX, HTML, TXT, MD, JSON, CSV, YouTube transcripts
- **⚡ Fast Chunking**: Token-based chunking (512 tokens, 64-128 overlap)
- **🎯 Configurable**: Pydantic-based configuration management
- **🖥️ CLI Interface**: Rich CLI with progress bars and formatted output
- **🔧 Modular Design**: Clean separation of concerns for easy extension
- **🔄 Auto-Recovery**: Automatic FAISS index recovery and recreation
- **💾 Smart Caching**: Embedding caching for improved performance

## Architecture

```
local-rag-system/
├── config/          # Configuration management
├── ingest/          # Document loading and chunking
├── embed/           # Embedding pipeline (BGE)
├── vectorstore/     # FAISS vector storage
├── llm/             # Ollama LLM interface
├── rag/             # RAG orchestrator
├── cli/             # Command-line interface
└── main.py          # Entry point and demo
```

## Quick Start

### Prerequisites

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Pull LLaMA 3 8B**:
   ```bash
   ollama pull llama3:8b
   ```

### Installation

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd local-rag-system
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run demo**:
   ```bash
   python main.py
   ```

### CLI Usage

The system provides a rich command-line interface with the following commands:

#### Document Ingestion

```bash
# Ingest documents from a directory
python -m cli.interface ingest --path ./documents

# Ingest documents recursively
python -m cli.interface ingest --path ./documents --recursive

# Ingest specific file formats
python -m cli.interface ingest --path ./documents --formats pdf,docx,txt

# Ingest YouTube transcript
python -m cli.interface ingest-youtube --url "https://youtube.com/watch?v=..." --language en
```

#### Querying

```bash
# Basic query
python -m cli.interface query "What is machine learning?"

# Query with custom parameters
python -m cli.interface query "Explain deep learning" --top-k 10 --threshold 0.6

# Stream response for real-time output
python -m cli.interface query "What are neural networks?" --stream

# Query without showing sources
python -m cli.interface query "Define AI" --no-sources

# Combine parameters
python -m cli.interface query "How does backpropagation work?" --top-k 3 --threshold 0.7 --stream
```

#### System Management

```bash
# Check system status
python -m cli.interface status

# Reset the vector store index
python -m cli.interface reset

# Show current configuration
python -m cli.interface config-show

# Save configuration to file
python -m cli.interface config-show --output config.json
```

### Query Parameters

The query command supports several parameters to fine-tune retrieval:

- **`--top-k, -k`**: Number of documents to retrieve (default: 5)
  - Higher values provide more context but may include less relevant information
  - Lower values focus on the most relevant documents

- **`--threshold, -t`**: Similarity threshold for filtering results (default: 0.7)
  - Higher values (0.8-0.9) return only very similar documents
  - Lower values (0.5-0.6) include more diverse but potentially less relevant results

- **`--stream, -s`**: Stream the response in real-time
  - Useful for long responses or interactive sessions

- **`--no-sources`**: Hide source document excerpts
  - Cleaner output when you only need the answer

## Configuration

The system uses Pydantic for type-safe configuration. Key settings:

```python
# Embedding settings
embedding.model_name = "BAAI/bge-base-en-v1.5"
embedding.device = "cpu"
embedding.batch_size = 32

# Chunking settings
chunking.chunk_size = 512  # tokens
chunking.chunk_overlap = 64  # tokens

# Vector store settings
vectorstore.top_k = 5  # Default number of documents to retrieve
vectorstore.similarity_threshold = 0.7  # Default similarity threshold

# LLM settings
llm.model_name = "llama3:8b"
llm.base_url = "http://localhost:11434"
llm.temperature = 0.1
```

Environment variables can override defaults using `RAG_` prefix:
```bash
export RAG_LLM_TEMPERATURE=0.2
export RAG_CHUNKING_CHUNK_SIZE=1024
export RAG_VECTORSTORE_TOP_K=10
export RAG_VECTORSTORE_SIMILARITY_THRESHOLD=0.8
```

## Supported Document Formats

- **PDF**: Via PyPDF2
- **DOCX**: Via python-docx
- **HTML**: Via BeautifulSoup
- **Text/Markdown**: Direct loading
- **JSON**: Structured data parsing
- **CSV**: Tabular data via pandas
- **YouTube**: Transcript via youtube-transcript-api

## Programmatic Usage

```python
from rag.orchestrator import create_rag_orchestrator

# Initialize system
rag = create_rag_orchestrator()

# Ingest documents
result = rag.ingest_documents(directory_path="./documents")
print(f"Ingested {result['total_documents']} documents into {result['total_chunks']} chunks")

# Query system with custom parameters
response = rag.query(
    question="What is the main topic?",
    top_k=10,
    score_threshold=0.6,
    return_sources=True
)
print(f"Answer: {response['answer']}")
print(f"Retrieved {response['retrieved_documents']} relevant documents")

# Stream response
for chunk in rag.stream_query(
    question="Explain the concept",
    top_k=5,
    score_threshold=0.7
):
    print(chunk, end="")

# Check system status
status = rag.get_system_status()
print(f"Vector store contains {status['vector_store']['total_documents']} documents")
```

## Advanced Features

### Automatic Index Recovery

The system includes robust error handling for FAISS index loading:

- **Automatic Recovery**: If the FAISS index becomes corrupted or incompatible, the system automatically recreates it from stored document metadata
- **Embedding Caching**: Previously computed embeddings are cached and reused during index recreation
- **Seamless Operation**: Recovery happens transparently without user intervention

### Custom Prompt Template

The system uses an optimized prompt format:

```
You are an expert assistant helping answer questions based on provided context.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
Give a concise and factual answer using only the information from the context.
If the answer cannot be found in the context, respond with: "The answer is not in the provided documents."
```

### Embedding Caching

- Embeddings are automatically cached in the `embeddings_cache/` directory
- Significantly improves performance on repeated ingestion
- Cache is automatically managed and cleaned up

### Metadata Preservation

Document metadata is preserved through the entire pipeline:
- Source file information
- Chunk indices and statistics
- Custom metadata fields
- Similarity scores for retrieved documents

### Smart Retrieval

The system implements intelligent document retrieval:
- **Score-based filtering**: Only documents above the similarity threshold are used
- **Configurable top-k**: Retrieve the most relevant documents
- **Source attribution**: Each answer includes source document references with similarity scores

## Performance Tuning

### Embedding Performance
- Adjust `batch_size` for your hardware (32 for CPU, 64+ for GPU)
- Use GPU if available: `device = "cuda"`
- Monitor memory usage with large document sets

### Chunking Optimization
- **Larger chunks (1024 tokens)**: Better context preservation, slower retrieval
- **Smaller chunks (256 tokens)**: Faster retrieval, may lose context
- **Overlap adjustment**: Higher overlap (128 tokens) for better continuity

### Retrieval Tuning
- **Higher top-k (10-20)**: More comprehensive answers, slower processing
- **Lower top-k (3-5)**: Faster responses, focused answers
- **Threshold tuning**:
  - 0.8-0.9: High precision, may miss relevant information
  - 0.6-0.7: Balanced precision and recall
  - 0.4-0.5: High recall, may include less relevant information

## System Status

Use `python -m cli.interface status` to get comprehensive system information:

```
Vector Store
├── Total Documents: 370
├── Index Size: 370
├── Embedding Dimension: 768
└── Sources: 1

Embedding Model
├── Model Name: BAAI/bge-base-en-v1.5
├── Dimension: 768
├── Device: cpu
└── Batch Size: 32

Ollama LLM
├── Server Running: ✅ Yes
├── Server URL: http://localhost:11434
├── Target Model: llama3:8b
├── Model Available: ✅ Yes
└── Total Models: 3

Configuration
├── Chunk Size: 512 tokens
├── Chunk Overlap: 64 tokens
├── Top K: 5
└── Similarity Threshold: 0.7
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**:
   ```bash
   # Check if Ollama is running
   ollama list
   
   # Start Ollama service
   ollama serve
   
   # Verify model is available
   ollama pull llama3:8b
   ```

2. **FAISS Index Loading Issues**:
   - The system automatically recovers from corrupted indices
   - Check logs for "Recreating FAISS index from existing documents"
   - If issues persist, use `python -m cli.interface reset` to start fresh

3. **Memory Issues**:
   - Reduce embedding batch size: `RAG_EMBEDDING_BATCH_SIZE=16`
   - Use smaller chunk sizes: `RAG_CHUNKING_CHUNK_SIZE=256`
   - Process documents in smaller batches

4. **Slow Performance**:
   - Enable GPU for embeddings: `RAG_EMBEDDING_DEVICE=cuda`
   - Increase batch sizes for GPU: `RAG_EMBEDDING_BATCH_SIZE=64`
   - Use SSD storage for vector store and cache

5. **No Relevant Documents Found**:
   - Lower the similarity threshold: `--threshold 0.5`
   - Increase top-k: `--top-k 10`
   - Check if documents were properly ingested: `python -m cli.interface status`

### Logging

Enable verbose logging:
```bash
python -m cli.interface --verbose status
```

Or set log level in configuration:
```python
log_level = "DEBUG"
```

### Performance Monitoring

Monitor system performance:
```bash
# Check embedding cache usage
ls -la embeddings_cache/

# Monitor vector store size
python -m cli.interface status

# Test query performance
time python -m cli.interface query "test question" --threshold 0.7
```

## Development

### Project Structure

- **Modular Design**: Each component is independently testable
- **Factory Pattern**: Easy component swapping
- **Type Hints**: Full type annotation for better IDE support
- **Pydantic Models**: Type-safe configuration
- **Rich CLI**: Beautiful command-line interface
- **Error Recovery**: Robust error handling and automatic recovery

### Extending the System

1. **Add New Document Loaders**:
   ```python
   # In ingest/document_loader.py
   def _load_custom_format(self, file_path: str) -> List[Document]:
       # Your custom loader logic
       pass
   ```

2. **Custom Embedding Models**:
   ```python
   # In embed/embedding_pipeline.py
   class CustomEmbeddings(Embeddings):
       # Your custom embedding implementation
       pass
   ```

3. **Alternative Vector Stores**:
   ```python
   # Create new module in vectorstore/
   class CustomVectorStore(VectorStore):
       # Your vector store implementation
       pass
   ```

### Testing

```bash
# Test document ingestion
python -m cli.interface ingest --path ./test_documents

# Test query functionality
python -m cli.interface query "test query" --top-k 3 --threshold 0.6

# Test system status
python -m cli.interface status

# Test streaming
python -m cli.interface query "test streaming" --stream
```

## License

[Your License Here]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review system status: `python -m cli.interface status`
- Enable verbose logging for debugging: `--verbose`
- Check the logs for automatic recovery messages
- Verify Ollama is running and model is available

## Recent Updates

- ✅ Fixed FAISS index loading with automatic recovery
- ✅ Added `--threshold` parameter for similarity filtering
- ✅ Added `--top-k` parameter for retrieval control
- ✅ Implemented embedding caching for better performance
- ✅ Enhanced error handling and system robustness
- ✅ Improved CLI interface with rich formatting
