"""
Command-line interface for the RAG system.
Provides user-friendly commands for document ingestion and querying.
"""

import click
import logging
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown
import json

from rag.orchestrator import create_rag_orchestrator
from config.settings import config

# Initialize rich console
console = Console()

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """Local RAG System - Document ingestion and querying with LLaMA 3 8B."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option('--path', '-p', required=True, help='Path to document or directory')
@click.option('--recursive', '-r', is_flag=True, default=True, help='Search directory recursively')
@click.option('--formats', '-f', help='Comma-separated list of file formats to include')
def ingest(path, recursive, formats):
    """Ingest documents into the RAG system."""
    console.print(f"\n[bold blue]📚 Ingesting documents from: {path}[/bold blue]")
    
    try:
        # Initialize orchestrator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing RAG system...", total=None)
            rag = create_rag_orchestrator()
            progress.update(task, description="RAG system initialized ✓")
        
        # Determine if path is file or directory
        path_obj = Path(path)
        if not path_obj.exists():
            console.print(f"[red]❌ Path does not exist: {path}[/red]")
            return
        
        # Ingest documents
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing documents...", total=None)
            
            if path_obj.is_file():
                results = rag.ingest_documents(file_paths=[str(path_obj)])
            else:
                results = rag.ingest_documents(directory_path=str(path_obj), recursive=recursive)
            
            progress.update(task, description="Documents processed ✓")
        
        # Display results
        if results["status"] == "success":
            console.print(f"\n[green]✅ Successfully ingested documents![/green]")
            
            # Create results table
            table = Table(title="Ingestion Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total Documents", str(results["total_documents"]))
            table.add_row("Total Chunks", str(results["total_chunks"]))
            table.add_row("Average Chunk Size", f"{results['chunk_stats']['avg_chunk_size']:.0f} tokens")
            table.add_row("Sources", str(results['chunk_stats']['sources']))
            
            console.print(table)
            
        else:
            console.print(f"[red]❌ Ingestion failed: {results.get('status', 'Unknown error')}[/red]")
    
    except Exception as e:
        console.print(f"[red]❌ Error during ingestion: {str(e)}[/red]")


@cli.command()
@click.option('--url', '-u', required=True, help='YouTube video URL')
@click.option('--language', '-l', default='en', help='Language code for transcript')
def ingest_youtube(url, language):
    """Ingest YouTube video transcript."""
    console.print(f"\n[bold blue]🎥 Ingesting YouTube transcript from: {url}[/bold blue]")
    
    try:
        # Initialize orchestrator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing RAG system...", total=None)
            rag = create_rag_orchestrator()
            progress.update(task, description="RAG system initialized ✓")
        
        # Ingest transcript
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Downloading transcript...", total=None)
            results = rag.ingest_youtube_transcript(url, language)
            progress.update(task, description="Transcript processed ✓")
        
        # Display results
        if results["status"] == "success":
            console.print(f"\n[green]✅ Successfully ingested YouTube transcript![/green]")
            console.print(f"📊 Created {results['total_chunks']} chunks")
        else:
            console.print(f"[red]❌ Failed to ingest transcript: {results.get('error', 'Unknown error')}[/red]")
    
    except Exception as e:
        console.print(f"[red]❌ Error during ingestion: {str(e)}[/red]")


@cli.command()
@click.argument('question')
@click.option('--top-k', '-k', default=None, type=int, help='Number of documents to retrieve')
@click.option('--threshold', '-t', default=None, type=float, help='Similarity threshold')
@click.option('--no-sources', is_flag=True, help='Hide source documents')
@click.option('--stream', '-s', is_flag=True, help='Stream the response')
def query(question, top_k, threshold, no_sources, stream):
    """Query the RAG system."""
    console.print(f"\n[bold blue]🤔 Question: {question}[/bold blue]")
    
    try:
        # Initialize orchestrator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing RAG system...", total=None)
            rag = create_rag_orchestrator()
            progress.update(task, description="RAG system initialized ✓")
        
        if stream:
            # Stream response
            console.print(f"\n[bold green]🤖 Answer:[/bold green]")
            console.print()
            
            for chunk in rag.stream_query(question, top_k=top_k, score_threshold=threshold):
                console.print(chunk, end="")
            
            console.print("\n")
        
        else:
            # Get complete response
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Processing query...", total=None)
                response = rag.query(
                    question=question,
                    top_k=top_k,
                    score_threshold=threshold,
                    return_sources=not no_sources
                )
                progress.update(task, description="Query processed ✓")
            
            # Display answer
            console.print(f"\n[bold green]🤖 Answer:[/bold green]")
            answer_panel = Panel(
                Markdown(response["answer"]),
                title="Response",
                border_style="green"
            )
            console.print(answer_panel)
            
            # Display sources if requested
            if not no_sources and response.get("sources"):
                console.print(f"\n[bold cyan]📚 Sources ({len(response['sources'])} documents):[/bold cyan]")
                
                for i, source in enumerate(response["sources"], 1):
                    source_panel = Panel(
                        f"[dim]{source['content']}[/dim]\n\n"
                        f"[bold]Source:[/bold] {source['metadata'].get('source', 'Unknown')}\n"
                        f"[bold]Similarity:[/bold] {source['similarity_score']:.3f}",
                        title=f"Source {i}",
                        border_style="cyan"
                    )
                    console.print(source_panel)
    
    except Exception as e:
        console.print(f"[red]❌ Error during query: {str(e)}[/red]")


@cli.command()
def status():
    """Show system status and statistics."""
    console.print(f"\n[bold blue]📊 RAG System Status[/bold blue]")
    
    try:
        # Initialize orchestrator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Checking system status...", total=None)
            rag = create_rag_orchestrator()
            status_info = rag.get_system_status()
            progress.update(task, description="Status retrieved ✓")
        
        # Vector Store Status
        vs_stats = status_info["vector_store"]
        vs_table = Table(title="Vector Store")
        vs_table.add_column("Metric", style="cyan")
        vs_table.add_column("Value", style="green")
        
        vs_table.add_row("Total Documents", str(vs_stats["total_documents"]))
        vs_table.add_row("Index Size", str(vs_stats["index_size"]))
        vs_table.add_row("Embedding Dimension", str(vs_stats["embedding_dimension"]))
        vs_table.add_row("Sources", str(len(vs_stats["sources"])))
        
        console.print(vs_table)
        
        # Embedding Model Status
        embed_stats = status_info["embedding_model"]
        embed_table = Table(title="Embedding Model")
        embed_table.add_column("Metric", style="cyan")
        embed_table.add_column("Value", style="green")
        
        embed_table.add_row("Model Name", embed_stats["model_name"])
        embed_table.add_row("Dimension", str(embed_stats["embedding_dimension"]))
        embed_table.add_row("Device", embed_stats["device"])
        embed_table.add_row("Batch Size", str(embed_stats["batch_size"]))
        
        console.print(embed_table)
        
        # Ollama Status
        ollama_stats = status_info["ollama_status"]
        ollama_table = Table(title="Ollama LLM")
        ollama_table.add_column("Metric", style="cyan")
        ollama_table.add_column("Value", style="green" if ollama_stats["server_running"] else "red")
        
        ollama_table.add_row("Server Running", "✅ Yes" if ollama_stats["server_running"] else "❌ No")
        ollama_table.add_row("Server URL", ollama_stats["server_url"])
        ollama_table.add_row("Target Model", ollama_stats["target_model"])
        ollama_table.add_row("Model Available", "✅ Yes" if ollama_stats["target_model_available"] else "❌ No")
        ollama_table.add_row("Total Models", str(ollama_stats.get("total_models", 0)))
        
        console.print(ollama_table)
        
        # Configuration
        config_info = status_info["configuration"]
        config_table = Table(title="Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")
        
        config_table.add_row("Chunk Size", f"{config_info['chunk_size']} tokens")
        config_table.add_row("Chunk Overlap", f"{config_info['chunk_overlap']} tokens")
        config_table.add_row("Top K", str(config_info['top_k']))
        config_table.add_row("Similarity Threshold", str(config_info['similarity_threshold']))
        
        console.print(config_table)
    
    except Exception as e:
        console.print(f"[red]❌ Error getting status: {str(e)}[/red]")


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to reset the index?')
def reset():
    """Reset the vector store index."""
    console.print(f"\n[bold yellow]🔄 Resetting vector store index...[/bold yellow]")
    
    try:
        rag = create_rag_orchestrator()
        success = rag.reset_index()
        
        if success:
            console.print(f"[green]✅ Index reset successfully![/green]")
        else:
            console.print(f"[red]❌ Failed to reset index[/red]")
    
    except Exception as e:
        console.print(f"[red]❌ Error resetting index: {str(e)}[/red]")


@cli.command()
@click.option('--output', '-o', help='Output file for configuration')
def config_show(output):
    """Show current configuration."""
    config_dict = {
        "embedding": {
            "model_name": config.embedding.model_name,
            "device": config.embedding.device,
            "batch_size": config.embedding.batch_size
        },
        "chunking": {
            "chunk_size": config.chunking.chunk_size,
            "chunk_overlap": config.chunking.chunk_overlap,
            "max_overlap": config.chunking.max_overlap
        },
        "vectorstore": {
            "index_path": config.vectorstore.index_path,
            "similarity_threshold": config.vectorstore.similarity_threshold,
            "top_k": config.vectorstore.top_k
        },
        "llm": {
            "model_name": config.llm.model_name,
            "base_url": config.llm.base_url,
            "temperature": config.llm.temperature,
            "max_tokens": config.llm.max_tokens
        }
    }
    
    if output:
        with open(output, 'w') as f:
            json.dump(config_dict, f, indent=2)
        console.print(f"[green]✅ Configuration saved to {output}[/green]")
    else:
        console.print(Panel(
            json.dumps(config_dict, indent=2),
            title="Current Configuration",
            border_style="blue"
        ))


if __name__ == '__main__':
    cli()
