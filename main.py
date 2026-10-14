from pathlib import Path
from typing import Optional
import logging
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import settings
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
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

# Silence third-party network & library debug logs
for logger_name in ["httpx", "httpcore", "transformers", "sentence_transformers", "litellm", "urllib3"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


from src.ingestion.universal_parser import UniversalDocumentParser
from src.cache.semantic_cache import SemanticCache
from src.retrieval.router import RetrievalStrategy


@app.command()
def ingest(
    input_dir: str = typer.Option(None, "--input-dir", "-i", help="Path to input directory containing PDF, DOCX, PPTX, MD files"),
    raw_dir: str = typer.Option(None, "--raw-dir", "-r", help="Alias for --input-dir")
):
    """Ingest multi-format files (PDF, DOCX, PPTX, MD) into standardized Markdown & Obsidian vault."""
    target_dir = Path(input_dir or raw_dir or settings.data_raw_dir)
    console.print(Panel.fit(f"[bold blue]Universal Ingestion from: {target_dir.resolve()}[/bold blue]"))
    
    if not target_dir.exists():
        console.print(f"[bold red]Directory '{target_dir}' does not exist.[/bold red]")
        return

    parser = UniversalDocumentParser()
    files = [f for f in target_dir.iterdir() if f.is_file() and f.suffix.lower() in [".pdf", ".docx", ".pptx", ".md", ".txt"]]

    if not files:
        console.print(f"[yellow]No supported documents found in {target_dir}.[/yellow]")
        return

    console.print(f"Discovered [bold]{len(files)}[/bold] documents. Parsing...")
    for idx, f in enumerate(files, 1):
        proc_doc = parser.parse_and_export(f)
        console.print(f"  [{idx}/{len(files)}] [green][OK][/green] Ingested [cyan]{f.name}[/cyan] ({proc_doc.source_type.value.upper()}) -> {len(proc_doc.chunks)} chunks")

    console.print(f"\n[bold green]Successfully ingested {len(files)} documents into data/markdown/ and Obsidian vault![/bold green]")


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

    md_files = [f for f in target_vault.glob("*.md") if not f.name.startswith("concept_")]
    if not md_files:
        console.print("[yellow]No markdown files found in vault directory. Run 'python main.py ingest' first.[/yellow]")
        return

    import time
    console.print(f"Processing [bold]{len(md_files)}[/bold] enterprise documents...")
    for idx, md_file in enumerate(md_files, 1):
        console.print(f"  [{idx}/{len(md_files)}] Indexing [cyan]{md_file.name}[/cyan]...")
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
        
        # Safe pause (4.5s) to guarantee <= 12 requests/minute under Free Tier
        time.sleep(4.5)

    stats = graph_store.get_stats()
    console.print(f"[bold green]Indexing completed successfully![/bold green]")
    console.print(f"  - Knowledge Graph Nodes: [bold cyan]{stats['node_count']}[/bold cyan]")
    console.print(f"  - Knowledge Graph Edges: [bold cyan]{stats['rel_count']}[/bold cyan]")


@app.command()
def query(
    question: str = typer.Argument(..., help="User question to answer via GraphRAG pipeline"),
    strategy: str = typer.Option(None, "--strategy", "-s", help="Force retrieval strategy: naive_vector | local_graph | global_community | hybrid_adaptive"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass semantic cache")
):
    """Query GraphRAG engine using semantic cache, adaptive routing, and grounded synthesis."""
    console.print(Panel.fit(f"[bold green]Question:[/bold green] {question}"))

    forced_strategy = None
    if strategy:
        try:
            forced_strategy = RetrievalStrategy(strategy.lower())
        except ValueError:
            console.print(f"[bold red]Invalid strategy '{strategy}'.[/bold red] Options: {[s.value for s in RetrievalStrategy]}")
            return

    # Check Semantic Cache
    cache = SemanticCache(threshold=0.90)
    if not no_cache and not forced_strategy:
        cache_hit = cache.get(question)
        if cache_hit:
            console.print(f"[bold green][Strategy]: SEMANTIC_CACHE[/bold green] [dim](Hit in {cache_hit['latency_ms']:.1f}ms, $0.00, Match: '{cache_hit['matched_query']}') [/dim]\n")
            console.print("\n[bold yellow]--- Grounded Answer ---[/bold yellow]\n")
            console.print(cache_hit["answer"])

            if cache_hit.get("citations"):
                console.print("\n[bold cyan]--- Citations Traceability ---[/bold cyan]")
                table = Table(title="Cached Citations & Graph Paths")
                table.add_column("Ref #", style="dim", width=6)
                table.add_column("Source ID", style="cyan")
                table.add_column("Type", style="magenta")
                for cite in cache_hit["citations"]:
                    table.add_row(str(cite["ref_num"]), str(cite["source"]), str(cite["type"]))
                console.print(table)
            return

    hybrid_engine = HybridRetrievalEngine()
    synthesizer = AnswerSynthesizer()

    # Retrieve & Rerank with Router
    retrieved, chosen_strategy = hybrid_engine.retrieve(question, strategy=forced_strategy)
    
    # Display routing decision
    override_note = " (Manual Override)" if forced_strategy else " (Auto-Routed)"
    console.print(f"[bold magenta][Strategy]:[/bold magenta] [bold cyan]{chosen_strategy.value.upper()}[/bold cyan]{override_note}\n")

    # Synthesize Grounded Answer
    result = synthesizer.synthesize(question, retrieved)

    # Store in cache
    if not no_cache and result.get("answer"):
        cache.set(question, result["answer"], result.get("citations", []), chosen_strategy.value)

    console.print("\n[bold yellow]--- Grounded Answer ---[/bold yellow]\n")
    console.print(result["answer"])

    if result.get("citations"):
        console.print("\n[bold cyan]--- Citations Traceability ---[/bold cyan]")
        table = Table(title=f"Retrieved Citations & Graph Paths ({chosen_strategy.value})")
        table.add_column("Ref #", style="dim", width=6)
        table.add_column("Source ID", style="cyan")
        table.add_column("Type", style="magenta")
        for cite in result["citations"]:
            table.add_row(str(cite["ref_num"]), str(cite["source"]), str(cite["type"]))
        console.print(table)


@app.command()
def ui(
    port: int = typer.Option(8501, "--port", "-p", help="Port to run Streamlit app on"),
    host: str = typer.Option("localhost", "--host", "-h", help="Host address for Streamlit app")
):
    """Launch the interactive Streamlit Web UI."""
    import sys
    import subprocess
    console.print(Panel.fit(f"[bold green]Launching GraphRAG Enterprise Web UI on http://{host}:{port}[/bold green]"))
    cmd = [
        sys.executable, "-m", "streamlit", "run", "src/ui/app.py",
        f"--server.port={port}", f"--server.address={host}",
        "--server.fileWatcherType=none", "--server.headless=true"
    ]
    subprocess.run(cmd)


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


@app.command("download-crag")
def download_crag(
    sample_size: int = typer.Option(50, "--sample-size", "-s", help="Number of balanced stratified samples for eval_dataset.json"),
    full_path: str = typer.Option("data/crag_benchmark_full.json", "--full-path", "-f", help="Output path for full CRAG benchmark"),
    eval_path: str = typer.Option("data/eval_dataset.json", "--eval-path", "-e", help="Output path for stratified evaluation benchmark"),
    corpus_path: str = typer.Option("data/raw/crag_corpus.json", "--corpus-path", "-c", help="Output path for deduplicated document corpus")
):
    """Download and prepare the NeurIPS 2024 CRAG dataset, search snippets, and document corpus."""
    from src.evaluation.crag_loader import CRAGBenchmarkLoader
    console.print(Panel.fit("[bold blue]Downloading NeurIPS 2024 CRAG Benchmark & Reference Corpus from Hugging Face...[/bold blue]"))
    loader = CRAGBenchmarkLoader()
    res = loader.download_and_save(full_qa_path=full_path, eval_path=eval_path, corpus_path=corpus_path, eval_sample_size=sample_size)
    console.print(f"[bold green]CRAG Benchmark and Reference Corpus ready:[/bold green]")
    console.print(f"  - Total QA queries: [cyan]{res['total_qa_queries']}[/cyan]")
    console.print(f"  - Deduplicated Corpus Documents: [cyan]{res['total_corpus_documents']}[/cyan]")
    console.print(f"  - Stratified Evaluation Samples: [cyan]{res['eval_sample_size']}[/cyan]")
    console.print(f"  - Full benchmark saved to: [cyan]{res['full_qa_path']}[/cyan]")
    console.print(f"  - Document corpus saved to: [cyan]{res['corpus_path']}[/cyan]")
    console.print(f"  - Evaluation dataset saved to: [cyan]{res['eval_dataset_path']}[/cyan]")


@app.command("ingest-crag")
def ingest_crag(
    corpus_path: str = typer.Option("data/raw/crag_corpus.json", "--corpus-path", "-c", help="Path to CRAG corpus JSON"),
    eval_path: str = typer.Option("data/eval_dataset.json", "--eval-path", "-e", help="Path to evaluation dataset JSON"),
    max_docs: Optional[int] = typer.Option(None, "--max-docs", "-m", help="Maximum documents to index"),
    only_gold: bool = typer.Option(True, "--only-gold/--all-docs", help="Index gold reference documents for eval questions"),
    delay: Optional[float] = typer.Option(None, "--delay", "-d", help="Delay in seconds between LLM calls (default 0.0s for local Ollama, 4.0s for Gemini)"),
    model: Optional[str] = typer.Option(None, "--model", "-M", help="LLM model string (e.g. 'ollama/gemma:2b', 'gemini/gemini-2.5-flash')")
):
    """Index CRAG document corpus into ChromaDB Vector Store and KuzuDB Knowledge Graph."""
    from src.evaluation.crag_loader import CRAGBenchmarkLoader
    active_model = model or getattr(settings, "llm_model", "ollama/gemma:2b")
    effective_delay = delay if delay is not None else (0.0 if active_model.startswith("ollama/") else 4.0)
    console.print(Panel.fit(f"[bold magenta]Ingesting CRAG Reference Corpus into ChromaDB & KuzuDB ({active_model}, delay={effective_delay}s)[/bold magenta]"))
    loader = CRAGBenchmarkLoader()
    res = loader.ingest_crag_corpus(
        corpus_path=corpus_path,
        eval_dataset_path=eval_path,
        max_docs=max_docs,
        only_eval_gold_docs=only_gold,
        delay_seconds=effective_delay,
        model=active_model
    )
    console.print(f"\n[bold green]CRAG Corpus indexing completed successfully![/bold green]")


@app.command("filter-dataset")
def filter_dataset(
    dataset: str = typer.Option("data/eval_dataset.json", "--dataset", help="Evaluation dataset to filter"),
    output: str = typer.Option("data/eval_dataset_filtered.json", "--output", "-o", help="Where to write the filtered dataset"),
    samples: Optional[int] = typer.Option(None, "--samples", "-n", help="Keep at most N balanced questions"),
    index_missing: bool = typer.Option(False, "--index-missing", help="Index the evidence documents of selected questions that are not indexed yet (uses the local LLM)"),
):
    """Keep only questions whose answer is present in gold documents indexed in Markdown, ChromaDB and KùzuDB."""
    from collections import Counter
    from src.evaluation.dataset_filter import build_filtered_dataset

    selected = build_filtered_dataset(dataset, output, samples, index_missing)
    console.print(f"[bold green]{len(selected)} questions with verified evidence coverage -> {output}[/bold green]")
    table = Table(title="Coverage by typology / domain")
    table.add_column("Typology", style="cyan")
    table.add_column("Domain", style="magenta")
    table.add_column("n", style="bold yellow")
    for (group, domain), n in sorted(Counter((i["type_group"], i["domain"]) for i in selected).items()):
        table.add_row(group, domain, str(n))
    console.print(table)


@app.command()
def evaluate(
    dataset: str = typer.Option("data/eval_dataset.json", "--dataset", help="Evaluation dataset JSON (see filter-dataset)"),
    samples: Optional[int] = typer.Option(None, "--samples", "-n", help="Number of samples to evaluate (default: all)"),
    output: str = typer.Option("reports/evaluation_report.md", "--output", "-o", help="Path to output markdown report"),
    delay: Optional[float] = typer.Option(None, "--delay", "-d", help="Delay in seconds between LLM calls"),
    model: Optional[str] = typer.Option(None, "--model", "-M", help="LLM model string for generation (e.g. 'ollama/gemma:2b')"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model", "-J", help="LLM model string for evaluation judge (e.g. 'gemini/gemini-3.8-flash')"),
    no_checkpoint: bool = typer.Option(False, "--no-checkpoint", "--force", "-f", help="Bypass and delete existing checkpoint to force a fresh evaluation from scratch")
):
    """Run automated paired benchmark evaluation (Naive Vector vs Hybrid Adaptive) with statistical inference."""
    from src.evaluation.evaluator import RAGEvaluator
    from src.evaluation.statistics import StatisticalAnalyzer

    active_model = model or getattr(settings, "llm_model", "ollama/gemma:2b")
    evaluator = RAGEvaluator(model_name=active_model, judge_model_name=judge_model)
    analyzer = StatisticalAnalyzer()

    console.print(Panel.fit(f"[bold green]Starting Automated RAGAS Paired Evaluation (Generation: {active_model} | Judge: {evaluator.judge_model_name})[/bold green]"))

    results = evaluator.run_paired_benchmark(dataset_path=dataset, max_samples=samples, delay_seconds=delay, no_checkpoint=no_checkpoint)

    console.print(f"\n[bold green]Computing inferential statistics for {len(results)} paired observations...[/bold green]")
    report_data = analyzer.generate_full_report(results)
    markdown_report = analyzer.format_markdown_report(report_data)

    out_file = Path(output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(markdown_report, encoding="utf-8")

    console.print(f"[bold cyan]Evaluation report saved to: {out_file.resolve()}[/bold cyan]\n")
    console.print(markdown_report)


if __name__ == "__main__":
    app()
