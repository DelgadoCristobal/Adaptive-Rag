from typing import List, Dict, Any, Optional
import logging
from src.config import settings
from src.graph.kuzu_store import KuzuGraphStore
from src.graph.schema import parse_note_frontmatter, parse_note_sources

logger = logging.getLogger(__name__)

_GLOBAL_ENTITY_DOC_CACHE: Optional[Dict[str, List[str]]] = None


def reset_entity_doc_cache() -> None:
    """Forget the cached entity -> source documents map so the next retrieval re-reads the vault."""
    global _GLOBAL_ENTITY_DOC_CACHE
    _GLOBAL_ENTITY_DOC_CACHE = None


class GraphRetriever:
    """Multi-hop entity and relationship retriever querying KùzuDB property graph."""

    def __init__(self, graph_store: Optional[KuzuGraphStore] = None):
        self.graph_store = graph_store or KuzuGraphStore()

    def _load_entity_doc_map(self) -> Dict[str, List[str]]:
        """Maps entity identifiers to their source doc IDs (in note order) from Obsidian concept notes."""
        global _GLOBAL_ENTITY_DOC_CACHE
        if _GLOBAL_ENTITY_DOC_CACHE is not None:
            return _GLOBAL_ENTITY_DOC_CACHE

        mapping: Dict[str, List[str]] = {}
        concepts_dir = settings.data_vault_dir / "concepts"
        if concepts_dir.exists():
            for f in concepts_dir.glob("*.md"):
                try:
                    text = f.read_text(encoding="utf-8", errors="ignore")
                except OSError as e:
                    logger.warning("Skipping unreadable concept note %s: %s", f.name, e)
                    continue
                docs = [s.replace(".md", "").strip() for s in parse_note_sources(text)]
                docs = [d for d in dict.fromkeys(docs) if d]
                ent_id = str(parse_note_frontmatter(text).get("id") or f.stem.replace("concept_", "")).strip()
                if docs and ent_id:
                    for key in (ent_id.lower(), ent_id.replace(" ", "_").lower()):
                        mapping[key] = list(dict.fromkeys(mapping.get(key, []) + docs))
        _GLOBAL_ENTITY_DOC_CACHE = mapping
        return mapping

    def retrieve(self, query: str, hops: int = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve candidate entities and multi-hop paths from KùzuDB for a query.

        Args:
            query: User input query.
            hops: Max traversal depth (default from settings).
            limit: Maximum candidates to extract.

        Returns:
            List of candidate dictionaries with linearized natural language text.
        """
        hops = hops or settings.graph_hops
        candidates = []
        doc_map = self._load_entity_doc_map()

        # 1. Match seed entities using single batch keyword query
        keywords = [query] + [w for w in query.split() if len(w) > 3]
        matched_entities = self.graph_store.search_entities_by_keywords(keywords)

        seen_entities = set()
        for ent in matched_entities:
            ent_id = ent.get("id")
            if not ent_id or ent_id in seen_entities:
                continue
            seen_entities.add(ent_id)

            # Linearize entity node to natural sentence
            desc = ent.get("description", "")
            type_label = ent.get("type", "CONCEPT")
            node_text = f"Entity '{ent.get('name')}' is a {type_label}. {desc}".strip()

            source_docs = doc_map.get(str(ent_id).lower()) or doc_map.get(str(ent_id).replace(" ", "_").lower()) or []
            source_doc = source_docs[0] if source_docs else ""

            candidates.append({
                "type": "graph_node",
                "id": ent_id,
                "text": node_text,
                "source": source_doc or f"graph_node:{ent_id}",
                "source_doc_id": source_doc,
                "source_doc_ids": list(source_docs),
                "entity_id": ent_id,
                "score": 0.9
            })

            # K-hop traversal
            neighborhood = self.graph_store.query_k_hop_neighborhood(ent_id, hops=hops)
            for path in neighborhood:
                target_desc = path.get("target_desc", "")
                target_name = path.get("target_name", path.get("target"))
                target_type = path.get("target_type", "CONCEPT")
                
                src_docs = doc_map.get(str(path['source']).lower()) or []
                tgt_docs = doc_map.get(str(path['target']).lower()) or []
                path_source_docs = src_docs or tgt_docs or source_docs
                path_source_doc = path_source_docs[0] if path_source_docs else ""

                # Linearize graph edge path into clear natural language sentence
                path_text = f"The entity '{ent.get('name')}' is connected to '{target_name}' ({target_type}). {target_desc}".strip()
                candidates.append({
                    "type": "graph_path",
                    "id": f"{path['source']}->{path['target']}",
                    "text": path_text,
                    "source": path_source_doc or f"graph_path:{path['source']}->{path['target']}",
                    "source_doc_id": path_source_doc,
                    "source_doc_ids": list(path_source_docs),
                    "score": 0.85
                })

            if len(candidates) >= limit:
                break

        return candidates[:limit]
