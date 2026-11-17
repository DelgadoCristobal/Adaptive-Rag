"""Single ingestion path shared by the CLI, the CRAG loader and the UI."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class IndexReport:
    """Outcome of indexing one document; `error` is set instead of raising when a step failed."""
    doc_id: str
    n_entities: int = 0
    n_relations: int = 0
    chunks_total: int = 0
    chunks_failed: int = 0
    error: Optional[str] = None


def source_doc_name(doc_id: str) -> str:
    """Canonical provenance name for a document: its stem plus a single '.md' suffix."""
    stem = doc_id[:-3] if doc_id.lower().endswith(".md") else doc_id
    return f"{stem}.md"


def index_document(
    doc_id: str,
    content: str,
    *,
    metadata: Optional[dict],
    vector_store,
    extractor,
    deduplicator,
    graph_store,
    vault_dir: Path,
    write_vault_copy: bool = False,
) -> IndexReport:
    """Vector add -> optional vault copy -> chunked extraction -> dedup -> graph upsert -> concept notes."""
    report = IndexReport(doc_id=doc_id)
    source_doc = source_doc_name(doc_id)
    vault_dir = Path(vault_dir)
    try:
        vector_store.add_documents(doc_id=doc_id, content=content, metadata=metadata)
        if write_vault_copy:
            vault_dir.mkdir(parents=True, exist_ok=True)
            (vault_dir / source_doc).write_text(content, encoding="utf-8")

        raw_extraction = extractor.extract_from_text(content, source_doc=source_doc)
        report.chunks_total = raw_extraction.chunks_total
        report.chunks_failed = raw_extraction.chunks_failed

        extraction = deduplicator.deduplicate(raw_extraction)
        report.n_entities = len(extraction.entities)
        report.n_relations = len(extraction.relations)
        graph_store.add_extraction_result(extraction)
        extractor.create_atomic_concept_notes(extraction, source_doc=source_doc, vault_dir=vault_dir)
    except Exception as e:
        report.error = f"{type(e).__name__}: {e}"
        logger.warning("Indexing failed for %s: %s", doc_id, report.error)
    return report
