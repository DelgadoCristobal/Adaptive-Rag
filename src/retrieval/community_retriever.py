from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from src.config import settings
from src.graph.kuzu_store import KuzuGraphStore

logger = logging.getLogger(__name__)


class CommunityRetriever:
    """Retriever leveraging macro concept syntheses and community topic summaries."""

    def __init__(
        self,
        concepts_dir: Optional[Path] = None,
        graph_store: Optional[KuzuGraphStore] = None
    ):
        self.concepts_dir = concepts_dir or (settings.data_vault_dir / "concepts")
        self.graph_store = graph_store or KuzuGraphStore()
        self._cached_concept_notes: Optional[List[Dict[str, Any]]] = None

    def _load_concept_notes(self) -> List[Dict[str, Any]]:
        """Loads and caches concept notes in memory to eliminate repeated disk I/O."""
        if self._cached_concept_notes is not None:
            return self._cached_concept_notes

        notes = []
        if self.concepts_dir.exists():
            for cf in self.concepts_dir.glob("*.md"):
                try:
                    text = cf.read_text(encoding="utf-8", errors="ignore")
                    notes.append({
                        "type": "community_summary",
                        "id": cf.stem,
                        "text": f"Concept Summary [{cf.stem.replace('concept_', '')}]:\n{text[:600]}",
                        "source": f"concept_note:{cf.name}",
                        "score": 0.88
                    })
                except Exception as e:
                    logger.debug(f"Failed to read concept note {cf}: {e}")
        self._cached_concept_notes = notes
        return notes

    def retrieve(self, query: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Fetch high-level concept notes and global community clusters.

        Args:
            query: Input user query.
            limit: Maximum community summaries to retrieve.

        Returns:
            List of candidate dictionaries formatted for reranking.
        """
        candidates = []

        # 1. Filter relevant concept notes from cached in-memory pool
        all_notes = self._load_concept_notes()
        q_words = [w for w in query.lower().split() if len(w) > 3]
        if q_words:
            matched_notes = [n for n in all_notes if any(w in n["text"].lower() for w in q_words)]
            candidates.extend(matched_notes[:limit] if matched_notes else all_notes[:limit])
        else:
            candidates.extend(all_notes[:limit])

        # 2. Extract prominent entity clusters from KuzuDB
        if len(candidates) < limit and self.graph_store and self.graph_store.conn:
            try:
                res = self.graph_store.conn.execute(
                    "MATCH (e:EntityNode) WHERE length(e.description) > 10 RETURN e.id, e.name, e.type, e.description LIMIT 20"
                )
                while res.has_next():
                    row = res.get_next()
                    candidates.append({
                        "type": "community_cluster",
                        "id": row[0],
                        "text": f"Knowledge Cluster: {row[1]} ({row[2]}) - {row[3]}",
                        "source": f"cluster:{row[0]}",
                        "score": 0.85
                    })
            except Exception as e:
                logger.debug(f"Cluster extraction notice: {e}")

        return candidates[:limit]
