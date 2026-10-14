from typing import Any, Dict, List, Optional, Set
import ast
import json
import re
import logging
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

# CRAG question types grouped into the typologies reported in the study (others, e.g. false_premise, are excluded).
TYPE_GROUPS = {
    "factoid": {"simple", "simple_w_condition"},
    "multi-hop": {"multi-hop"},
    "aggregation": {"aggregation"},
    "comparison": {"comparison"},
}


def _normalize(text: str) -> str:
    """Lowercase and drop thousands separators / punctuation so '1,889,100' matches '1889100'."""
    text = re.sub(r"(?<=\d)[,_](?=\d{3})", "", str(text).lower())
    return re.sub(r"[^\w.%$ ]+", " ", text)


def answer_in_text(answer: Any, text: str) -> bool:
    """True when the (normalized) answer appears as a whole token sequence inside `text`."""
    ans = " ".join(_normalize(answer).split())
    if len(ans) < 3 or ans in {"invalid question", "i don't know"}:  # yes/no and refusals are not extractive
        return False
    return re.search(rf"(?<![\w.]){re.escape(ans)}(?![\w])", " ".join(_normalize(text).split())) is not None


def _gold_ids(item: Dict[str, Any]) -> List[str]:
    ids = item.get("gold_document_ids") or []
    if isinstance(ids, str):  # tolerate list serialized as a string
        ids = ast.literal_eval(ids)
    return list(dict.fromkeys(ids))


def graph_indexed_docs(vault_dir: Optional[Path] = None) -> Set[str]:
    """Source documents that produced at least one KùzuDB entity (provenance lives in the vault concept notes)."""
    concepts = (vault_dir or settings.data_vault_dir) / "concepts"
    docs: Set[str] = set()
    for note in concepts.glob("*.md"):
        match = re.search(r"^source:\s*['\"]?(.*?)['\"]?\s*$", note.read_text(encoding="utf-8", errors="ignore"), re.MULTILINE)
        if match:
            docs.add(match.group(1).replace(".md", "").strip())
    return docs


def vector_indexed(collection, doc_id: str) -> bool:
    return bool(collection.get(where={"doc_id": doc_id}, limit=1)["ids"])


def evidence_documents(
    item: Dict[str, Any], markdown_dir: Path, collection, graph_docs: Set[str], require_index: bool = True
) -> List[str]:
    """Gold documents that exist on disk and contain the answer (and, if required, are indexed in ChromaDB and KùzuDB)."""
    evidence = []
    for doc_id in _gold_ids(item):
        md_file = markdown_dir / f"{doc_id}.md"
        if not md_file.exists():
            continue
        if require_index and not (doc_id in graph_docs and vector_indexed(collection, doc_id)):
            continue
        if answer_in_text(item.get("ground_truth") or item.get("answer"), md_file.read_text(encoding="utf-8")):
            evidence.append(doc_id)
    return evidence


def index_missing_evidence(selected: List[Dict[str, Any]], collection, graph_docs: Set[str], max_docs_per_question: int = 2) -> int:
    """Index (ChromaDB + vault + KùzuDB) the evidence documents of `selected` that are not yet fully indexed."""
    from src.evaluation.crag_loader import CRAGBenchmarkLoader
    from src.graph.deduplication import EntityDeduplicator
    from src.graph.extractor import EntityRelationExtractor
    from src.graph.kuzu_store import KuzuGraphStore
    from src.vector.vector_store import VectorStoreManager

    pending = [
        doc_id
        for item in selected
        for doc_id in item["evidence_document_ids"][:max_docs_per_question]
        if not (doc_id in graph_docs and vector_indexed(collection, doc_id))
    ]
    pending = list(dict.fromkeys(pending))
    if not pending:
        return 0

    vector_store = VectorStoreManager()
    extractor, deduplicator, graph_store = EntityRelationExtractor(), EntityDeduplicator(), KuzuGraphStore()
    vault_dir = settings.data_vault_dir
    vault_dir.mkdir(parents=True, exist_ok=True)
    for doc_id in pending:
        content = (settings.data_markdown_dir / f"{doc_id}.md").read_text(encoding="utf-8")
        CRAGBenchmarkLoader.index_document(
            doc_id, content, {"source": doc_id}, vector_store, extractor, deduplicator, graph_store, vault_dir
        )
        logger.info("Indexed evidence document %s", doc_id)
    return len(pending)


def build_filtered_dataset(
    dataset_path: str = "data/eval_dataset.json",
    output_path: str = "data/eval_dataset_filtered.json",
    max_samples: Optional[int] = None,
    index_missing: bool = False,
) -> List[Dict[str, Any]]:
    """Keep only questions whose answer is verifiably present in indexed gold documents, balanced by typology.

    With `index_missing`, answer-bearing gold documents of the selected questions that are not yet indexed are
    indexed first (instead of dropping the question), so the index covers exactly the evaluated subset.

    The output is ordered round-robin over typologies (and domains within each), so any prefix of it
    (e.g. `evaluate --samples 5`) stays as balanced as the available coverage allows.
    """
    from src.vector.vector_store import VectorStoreManager

    collection = VectorStoreManager().collection
    if collection is None:
        raise RuntimeError("ChromaDB collection unavailable; cannot verify index coverage.")
    graph_docs = graph_indexed_docs()
    markdown_dir = settings.data_markdown_dir

    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    kept = 0
    for item in dataset:
        group = next((g for g, types in TYPE_GROUPS.items() if item.get("type") in types), None)
        if group is None:
            continue
        evidence = evidence_documents(item, markdown_dir, collection, graph_docs, require_index=not index_missing)
        if evidence:
            kept += 1
            by_group[group].append({**item, "evidence_document_ids": evidence, "type_group": group})

    # Round-robin over domains inside each group, then over groups.
    queues = []
    for group in TYPE_GROUPS:
        by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in by_group[group]:
            by_domain[item["domain"]].append(item)
        queues.append([i for tier in zip_longest(*by_domain.values()) for i in tier if i])
    selected = [i for tier in zip_longest(*queues) for i in tier if i]
    if max_samples:
        selected = selected[:max_samples]

    if index_missing:
        n_indexed = index_missing_evidence(selected, collection, graph_docs)
        graph_docs = graph_indexed_docs()
        # Re-verify coverage after indexing; drop questions whose graph/vector extraction failed.
        selected = [i for i in selected if evidence_documents(i, markdown_dir, collection, graph_docs)]
        logger.info("Indexed %d evidence documents; %d questions remain fully covered.", n_indexed, len(selected))

    logger.info("Coverage filter kept %d/%d questions (%d after balancing).", kept, len(dataset), len(selected))
    out = Path(output_path)
    out.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    return selected
