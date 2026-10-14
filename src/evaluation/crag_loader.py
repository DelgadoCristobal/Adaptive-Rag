from typing import List, Dict, Any, Optional
import json
import time
from pathlib import Path
import logging
import random
from collections import defaultdict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import datasets

from src.config import settings

logger = logging.getLogger(__name__)
console = Console()


class CRAGBenchmarkLoader:
    """Loader and corpus ingestion pipeline for the official NeurIPS 2024 CRAG benchmark.
    
    Extracts question-answer pairs, ground-truth search snippets, and deduplicated
    reference documents across all 5 benchmark domains (Finance, Movie, Music, Open, Sports).
    Indexes the resulting corpus into ChromaDB (Vector Store) and KuzuDB (Knowledge Graph).
    """

    HF_DATASET_NAME = "DataRobot-Research/crag"
    HF_FULL_DATASET_NAME = "Quivr/CRAG"
    DOMAINS = ["finance", "movie", "music", "open", "sports"]
    SPLITS = ["train", "test", "holdout", "sample"]

    def __init__(self):
        pass

    def load_qa_and_corpus_from_huggingface(self) -> Dict[str, Any]:
        """Downloads QA pairs with associated search results and extracts deduplicated corpus documents."""
        logger.info("Fetching CRAG benchmark and reference search snippets from Hugging Face...")
        all_qa_items: List[Dict[str, Any]] = []
        hash_to_doc: Dict[str, Dict[str, Any]] = {}

        for domain in self.DOMAINS:
            cfg_name = f"qapairs_{domain}"
            for split in self.SPLITS:
                try:
                    ds = datasets.load_dataset(self.HF_DATASET_NAME, cfg_name, split=split)
                except Exception as e:
                    logger.debug("Split %s not found for %s: %s", split, cfg_name, e)
                    continue

                for row in ds:
                    search_results = row.get("search_results") or []
                    gold_doc_ids = []
                    snippets_meta = []

                    for sr in search_results:
                        doc_hash = sr.get("_hash")
                        if not doc_hash:
                            continue
                        
                        doc_id = f"crag_doc_{doc_hash[:16]}"
                        gold_doc_ids.append(doc_id)
                        
                        title = sr.get("page_name") or f"Document {doc_hash[:8]}"
                        url = sr.get("page_url") or ""
                        snippet_text = sr.get("page_snippet") or ""

                        clean_markdown = f"# {title}\n\n**Source URL:** {url}\n**Domain:** {domain}\n**Document ID:** {doc_id}\n\n{snippet_text}\n"

                        snippets_meta.append({
                            "doc_id": doc_id,
                            "hash": doc_hash,
                            "title": title,
                            "url": url,
                            "snippet": snippet_text
                        })

                        if doc_hash not in hash_to_doc:
                            hash_to_doc[doc_hash] = {
                                "doc_id": doc_id,
                                "hash": doc_hash,
                                "domain": domain,
                                "title": title,
                                "url": url,
                                "text": clean_markdown,
                                "raw_snippet": snippet_text,
                                "metadata": {
                                    "source": doc_id,
                                    "title": title,
                                    "domain": domain,
                                    "url": url,
                                    "hash": doc_hash
                                }
                            }

                    q_id = f"crag_{str(row['interaction_id'])[:8]}"
                    expected_source = gold_doc_ids[0] if gold_doc_ids else ""

                    qa_item = {
                        "id": q_id,
                        "interaction_id": row["interaction_id"],
                        "question": row["query"],
                        "query": row["query"],
                        "ground_truth": row["answer"],
                        "answer": row["answer"],
                        "domain": domain,
                        "type": row.get("question_type", "general"),
                        "question_type": row.get("question_type", "general"),
                        "static_or_dynamic": row.get("static_or_dynamic", "static"),
                        "query_time": row.get("query_time", ""),
                        "gold_document_ids": gold_doc_ids,
                        "expected_source": expected_source,
                        "search_snippets": snippets_meta[:5]
                    }
                    all_qa_items.append(qa_item)

        logger.info("Extracted %d QA pairs and %d unique corpus documents.", len(all_qa_items), len(hash_to_doc))
        return {
            "qa_items": all_qa_items,
            "corpus": list(hash_to_doc.values())
        }

    def create_stratified_sample(
        self,
        qa_items: List[Dict[str, Any]],
        sample_size: int = 50,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """Creates a balanced, stratified benchmark subset across domains and question types."""
        valid_items = [
            item for item in qa_items
            if item.get("ground_truth")
            and item.get("ground_truth").lower() != "i don't know"
            and item.get("gold_document_ids")
        ]

        strata = defaultdict(list)
        for item in valid_items:
            key = (item["domain"], item["question_type"])
            strata[key].append(item)

        rng = random.Random(seed)
        sampled: List[Dict[str, Any]] = []

        strata_keys = list(strata.keys())
        rng.shuffle(strata_keys)

        strata_pools = {k: list(v) for k, v in strata.items()}
        for k in strata_pools:
            rng.shuffle(strata_pools[k])

        current_idx = 0
        while len(sampled) < sample_size and strata_pools:
            key = strata_keys[current_idx % len(strata_keys)]
            if strata_pools[key]:
                sampled.append(strata_pools[key].pop(0))
            else:
                strata_keys.remove(key)
                if not strata_keys:
                    break
            current_idx += 1

        sampled.sort(key=lambda x: (x["domain"], x["question_type"]))
        return sampled

    def download_and_save(
        self,
        full_qa_path: str = "data/crag_benchmark_full.json",
        eval_path: str = "data/eval_dataset.json",
        corpus_path: str = "data/raw/crag_corpus.json",
        eval_sample_size: int = 50
    ) -> Dict[str, Any]:
        """Downloads benchmark QA pairs and extracts deduplicated document corpus to disk."""
        data = self.load_qa_and_corpus_from_huggingface()
        qa_items = data["qa_items"]
        corpus = data["corpus"]

        # 1. Save Full QA benchmark
        full_file = Path(full_qa_path)
        full_file.parent.mkdir(parents=True, exist_ok=True)
        with open(full_file, "w", encoding="utf-8") as f:
            json.dump(qa_items, f, indent=2, ensure_ascii=False)

        # 2. Save Deduplicated Document Corpus (.json and .jsonl)
        corpus_file = Path(corpus_path)
        corpus_file.parent.mkdir(parents=True, exist_ok=True)
        with open(corpus_file, "w", encoding="utf-8") as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False)

        jsonl_file = corpus_file.with_suffix(".jsonl")
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for doc in corpus:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        # 3. Save Stratified Evaluation Dataset with Gold Document References
        eval_sample = self.create_stratified_sample(qa_items, sample_size=eval_sample_size)
        eval_file = Path(eval_path)
        eval_file.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(eval_sample, f, indent=2, ensure_ascii=False)

        # Also write stratified subset documents into data/markdown/ for instant vault visibility
        md_dir = settings.data_markdown_dir
        md_dir.mkdir(parents=True, exist_ok=True)
        relevant_gold_hashes = {
            doc_id for item in eval_sample for doc_id in item.get("gold_document_ids", [])
        }
        for doc in corpus:
            if doc["doc_id"] in relevant_gold_hashes:
                doc_md_path = md_dir / f"{doc['doc_id']}.md"
                doc_md_path.write_text(doc["text"], encoding="utf-8")

        return {
            "total_qa_queries": len(qa_items),
            "total_corpus_documents": len(corpus),
            "eval_sample_size": len(eval_sample),
            "full_qa_path": str(full_file.resolve()),
            "corpus_path": str(corpus_file.resolve()),
            "eval_dataset_path": str(eval_file.resolve())
        }

    @staticmethod
    def index_document(doc_id, content, metadata, vector_store, extractor, deduplicator, graph_store, vault_dir) -> None:
        """Index one document into ChromaDB, the Obsidian vault and (via LLM extraction) the KùzuDB graph."""
        vector_store.add_documents(doc_id=doc_id, content=content, metadata=metadata)
        (vault_dir / f"{doc_id}.md").write_text(content, encoding="utf-8")
        try:
            raw_extraction = extractor.extract_from_text(content[:1800], source_doc=f"{doc_id}.md")
            dedup_extraction = deduplicator.deduplicate(raw_extraction)
            graph_store.add_extraction_result(dedup_extraction)
            extractor.create_atomic_concept_notes(dedup_extraction, source_doc=f"{doc_id}.md", vault_dir=vault_dir)
        except Exception as e:
            logger.warning("Graph extraction failed for %s: %s", doc_id, e)

    def ingest_crag_corpus(
        self,
        corpus_path: str = "data/raw/crag_corpus.json",
        eval_dataset_path: str = "data/eval_dataset.json",
        max_docs: Optional[int] = None,
        only_eval_gold_docs: bool = True,
        delay_seconds: Optional[float] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ingests the CRAG corpus into ChromaDB Vector Index and KuzuDB Knowledge Graph.
        
        Args:
            corpus_path: Path to deduplicated crag_corpus.json.
            eval_dataset_path: Path to stratified eval_dataset.json.
            max_docs: Optional cap on documents to ingest.
            only_eval_gold_docs: If True, prioritizes indexing the gold reference documents
                                 for the active evaluation benchmark.
            delay_seconds: Pause between LLM entity extractions to respect rate limits.
            model: Optional model identifier (e.g. 'ollama/gemma:2b', 'gemini/gemini-2.5-flash').
        """
        from src.vector.vector_store import VectorStoreManager
        from src.graph.extractor import EntityRelationExtractor
        from src.graph.deduplication import EntityDeduplicator
        from src.graph.kuzu_store import KuzuGraphStore

        corpus_file = Path(corpus_path)
        if not corpus_file.exists():
            console.print(f"[bold yellow]Corpus file '{corpus_file}' not found. Downloading first...[/bold yellow]")
            self.download_and_save(corpus_path=corpus_path, eval_path=eval_dataset_path)

        with open(corpus_file, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        target_docs = []
        if only_eval_gold_docs:
            eval_file = Path(eval_dataset_path)
            if eval_file.exists():
                with open(eval_file, "r", encoding="utf-8") as f:
                    eval_items = json.load(f)
                
                corpus_dict = {d["doc_id"]: d for d in corpus}
                eval_slice = eval_items[:max_docs] if max_docs else eval_items
                seen_doc_ids = set()

                # Priority 1: Primary expected_source for every eval question
                for item in eval_slice:
                    exp = item.get("expected_source")
                    if exp and exp in corpus_dict and exp not in seen_doc_ids:
                        target_docs.append(corpus_dict[exp])
                        seen_doc_ids.add(exp)

                # Priority 2: Additional top gold_document_ids for richer multi-hop grounding
                for item in eval_slice:
                    for gid in item.get("gold_document_ids", []):
                        if max_docs and len(target_docs) >= max_docs:
                            break
                        if gid in corpus_dict and gid not in seen_doc_ids:
                            target_docs.append(corpus_dict[gid])
                            seen_doc_ids.add(gid)
                    if max_docs and len(target_docs) >= max_docs:
                        break

        if not target_docs:
            target_docs = corpus[:max_docs] if max_docs else corpus

        active_model = model or getattr(settings, "llm_model", "ollama/gemma:2b")
        effective_delay = delay_seconds if delay_seconds is not None else (0.0 if active_model.startswith("ollama/") else 4.0)
        console.print(f"[bold blue]Starting CRAG Corpus Ingestion ({len(target_docs)} documents using {active_model}, delay: {effective_delay}s)...[/bold blue]")

        vector_store = VectorStoreManager()
        extractor = EntityRelationExtractor(model=active_model)
        deduplicator = EntityDeduplicator()
        graph_store = KuzuGraphStore()
        vault_dir = settings.data_vault_dir
        vault_dir.mkdir(parents=True, exist_ok=True)

        ingested_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[{task.completed}/{task.total}]"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Indexing CRAG Documents...", total=len(target_docs))

            for doc in target_docs:
                doc_id = doc["doc_id"]
                content = doc["text"]
                metadata = doc.get("metadata", {"source": doc_id, "title": doc.get("title", ""), "domain": doc.get("domain", "")})

                self.index_document(doc_id, content, metadata, vector_store, extractor, deduplicator, graph_store, vault_dir)

                ingested_count += 1
                progress.update(task, advance=1)
                if effective_delay > 0:
                    time.sleep(effective_delay)

        stats = graph_store.get_stats()
        c_count = vector_store.collection.count() if vector_store.collection else 0

        console.print(f"\n[bold green]CRAG Corpus Ingestion Completed Successfully:[/bold green]")
        console.print(f"  - Ingested Documents: [cyan]{ingested_count}[/cyan]")
        console.print(f"  - ChromaDB Embedded Chunks: [cyan]{c_count}[/cyan]")
        console.print(f"  - KuzuDB Graph Nodes: [cyan]{stats.get('node_count', 0)}[/cyan]")
        console.print(f"  - KuzuDB Graph Edges: [cyan]{stats.get('rel_count', 0)}[/cyan]")

        return {
            "ingested_docs": ingested_count,
            "vector_chunks": c_count,
            "graph_nodes": stats.get("node_count", 0),
            "graph_edges": stats.get("rel_count", 0)
        }
