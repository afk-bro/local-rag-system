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
   pip install -r requirements.txt
   ```

2. **Run demo**:
   ```bash
   python main.py
   ```

### CLI Usage

```bash
# Ingest documents
python main.py ingest --path ./documents

# Ingest YouTube transcript
python main.py ingest-youtube --url "https://youtube.com/watch?v=..."

# Query the system
python main.py query "What is machine learning?"

# Stream response
python main.py query "Explain deep learning" --stream

# Check system status
python main.py status

# Reset index
python main.py reset

# Show configuration
python main.py config-show
```

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
vectorstore.top_k = 5
vectorstore.similarity_threshold = 0.7

# LLM settings
llm.model_name = "llama3:8b"
llm.base_url = "http://localhost:11434"
llm.temperature = 0.1
```

Environment variables can override defaults using `RAG_` prefix:
```bash
export RAG_LLM_TEMPERATURE=0.2
export RAG_CHUNKING_CHUNK_SIZE=1024
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

# Query system
response = rag.query("What is the main topic?")
print(response["answer"])

# Stream response
for chunk in rag.stream_query("Explain the concept"):
    print(chunk, end="")
```

## Advanced Features

### Custom Prompt Template

The system uses your specified prompt format:

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

Embeddings are automatically cached to improve performance on repeated ingestion.

### Metadata Preservation

Document metadata is preserved through the entire pipeline:
- Source file information
- Chunk indices and statistics
- Custom metadata fields

### Error Handling

Robust error handling with detailed logging:
- Connection retry logic for Ollama
- Graceful handling of unsupported formats
- Index corruption recovery

## Performance Tuning

### Embedding Performance
- Adjust `batch_size` for your hardware
- Use GPU if available: `device = "cuda"`

### Chunking Optimization
- Larger chunks (1024 tokens) for better context
- Smaller chunks (256 tokens) for precise retrieval
- Adjust overlap based on document structure

### Vector Store Tuning
- Increase `top_k` for more comprehensive retrieval
- Adjust `similarity_threshold` for precision/recall balance

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**:
   ```bash
   # Check if Ollama is running
   ollama list
   
   # Start Ollama service
   ollama serve
   ```

2. **Model Not Found**:
   ```bash
   # Pull the required model
   ollama pull llama3:8b
   ```

3. **Memory Issues**:
   - Reduce embedding batch size
   - Use smaller chunk sizes
   - Process documents in smaller batches

4. **Slow Performance**:
   - Enable GPU for embeddings
   - Increase batch sizes
   - Use SSD for vector store

### Logging

Enable verbose logging:
```bash
python main.py --verbose status
```

Or set log level in configuration:
```python
log_level = "DEBUG"
```

## Development

### Project Structure

- **Modular Design**: Each component is independently testable
- **Factory Pattern**: Easy component swapping
- **Type Hints**: Full type annotation for better IDE support
- **Pydantic Models**: Type-safe configuration
- **Rich CLI**: Beautiful command-line interface

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

## License

[Your License Here]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review system status: `python main.py status`
- Enable verbose logging for debugging
