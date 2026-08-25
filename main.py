from pathlib import Path
import logging
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import settings
from src.ingest.pipeline import DocumentIngestPipeline
from src.graph.extractor import EntityRelationExtractor
from src.graph.deduplication import EntityDeduplicator
from src.graph.kuzu_store import KuzuGraphStore
from src.vector.vector_store import VectorStoreManager
from src.retrieval.hybrid_engine import HybridRetrievalEngine
from src.generator.synthesizer import AnswerSynthesizer

app = typer.Typer(
    name="GraphRAG Corporate Brain",
    help="Production-grade GraphRAG Pipeline with Obsidian Vault, KùzuDB Property Graph & Chroma Vector Index",
    add_completion=False
)
console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@app.command()
def ingest(
    raw_dir: str = typer.Option(None, "--raw-dir", "-r", help="Path to directory containing raw files"),
    vault_dir: str = typer.Option(None, "--vault-dir", "-v", help="Path to target Obsidian vault directory")
):
    """Ingest multi-format raw files (PDF, DOCX, PPTX) into Obsidian markdown vault."""
    console.print(Panel.fit("[bold blue]Ingesting Raw Enterprise Documents[/bold blue]"))
    pipeline = DocumentIngestPipeline(
        raw_dir=Path(raw_dir) if raw_dir else None,
        vault_dir=Path(vault_dir) if vault_dir else None
    )
    outputs = pipeline.run_all()
    console.print(f"[bold green]Successfully ingested {len(outputs)} documents into Obsidian vault![/bold green]")
    for path in outputs:
        console.print(f"  - [cyan]{path.name}[/cyan]")


@app.command()
def index(
    vault_dir: str = typer.Option(None, "--vault-dir", "-v", help="Path to Obsidian vault directory")
):
    """Extract entities & relations, deduplicate ontology, build KùzuDB graph & Chroma vector index."""
    console.print(Panel.fit("[bold magenta]Building Graph & Vector Index[/bold magenta]"))
    
    target_vault = Path(vault_dir) if vault_dir else settings.data_vault_dir
    extractor = EntityRelationExtractor()
    deduplicator = EntityDeduplicator()
    graph_store = KuzuGraphStore()
    vector_store = VectorStoreManager()

    md_files = list(target_vault.glob("*.md"))
    if not md_files:
        console.print("[yellow]No markdown files found in vault directory. Run 'python main.py ingest' first.[/yellow]")
        return

    console.print(f"Processing [bold]{len(md_files)}[/bold] vault files...")
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        
        # 1. Add to Vector Store
        vector_store.add_documents(doc_id=md_file.stem, content=content, metadata={"source": md_file.name})

        # 2. Extract Entities & Relations
        raw_extraction = extractor.extract_from_text(content, source_doc=md_file.name)
        
        # 3. Deduplicate Entities
        dedup_extraction = deduplicator.deduplicate(raw_extraction)

        # 4. Store in KùzuDB Graph Store
        graph_store.add_extraction_result(dedup_extraction)

        # 5. Generate Atomic Concept Notes with Wikilinks
        extractor.create_atomic_concept_notes(dedup_extraction, source_doc=md_file.name, vault_dir=target_vault)

    stats = graph_store.get_stats()
    console.print(f"[bold green]Indexing completed successfully![/bold green]")
    console.print(f"  - Knowledge Graph Nodes: [bold cyan]{stats['node_count']}[/bold cyan]")
    console.print(f"  - Knowledge Graph Edges: [bold cyan]{stats['rel_count']}[/bold cyan]")


@app.command()
def query(
    question: str = typer.Argument(..., help="User question to answer via GraphRAG pipeline")
):
    """Query GraphRAG engine using hybrid vector + multi-hop graph retrieval."""
    console.print(Panel.fit(f"[bold green]Question:[/bold green] {question}"))

    hybrid_engine = HybridRetrievalEngine()
    synthesizer = AnswerSynthesizer()

    # Retrieve & Rerank
    retrieved = hybrid_engine.retrieve(question)

    # Synthesize Grounded Answer
    result = synthesizer.synthesize(question, retrieved)

    console.print("\n[bold yellow]--- Grounded Answer ---[/bold yellow]\n")
    console.print(result["answer"])

    if result.get("citations"):
        console.print("\n[bold cyan]--- Citations Traceability ---[/bold cyan]")
        table = Table(title="Retrieved Citations & Graph Paths")
        table.add_column("Ref #", style="dim", width=6)
        table.add_column("Source ID", style="cyan")
        table.add_column("Type", style="magenta")
        for cite in result["citations"]:
            table.add_row(str(cite["ref_num"]), str(cite["source"]), str(cite["type"]))
        console.print(table)


@app.command()
def stats():
    """Display statistics for Vault files, KùzuDB Graph, and Chroma Vector Index."""
    graph_store = KuzuGraphStore()
    vector_store = VectorStoreManager()
    
    vault_files = list(settings.data_vault_dir.glob("*.md"))
    g_stats = graph_store.get_stats()
    c_count = vector_store.collection.count() if vector_store.collection else 0

    table = Table(title="GraphRAG Corporate Brain Statistics")
    table.add_column("Component", style="cyan")
    table.add_column("Metric", style="green")
    table.add_column("Value", style="bold yellow")

    table.add_row("Obsidian Vault", "Markdown Files", str(len(vault_files)))
    table.add_row("KùzuDB Property Graph", "Node Count", str(g_stats.get("node_count", 0)))
    table.add_row("KùzuDB Property Graph", "Edge Count", str(g_stats.get("rel_count", 0)))
    table.add_row("ChromaDB Vector Store", "Embedded Chunks", str(c_count))

    console.print(table)


if __name__ == "__main__":
    app()
