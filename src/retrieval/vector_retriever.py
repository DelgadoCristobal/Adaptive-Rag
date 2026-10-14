from typing import List, Dict, Any, Optional
import logging
from src.config import settings
from src.vector.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Dense vector retriever using ChromaDB for semantic similarity search."""

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.vector_store = vector_store or VectorStoreManager()

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieves top_k most similar document chunks for a query.

        Args:
            query: Input user question or query string.
            top_k: Number of chunks to retrieve.

        Returns:
            List of candidate dictionaries for reranker.
        """
        top_k = top_k or settings.vector_top_k
        results = self.vector_store.similarity_search(query, top_k=top_k)
        
        candidates = []
        for res in results:
            meta = res.get("metadata", {}) or {}
            # Small-to-Big Retrieval: Use rich parent section context if available, otherwise chunk content
            context_text = meta.get("parent_text") or res.get("content", "")
            source_doc = meta.get("source_doc_id") or meta.get("doc_id", "unknown")

            candidates.append({
                "type": "text_chunk",
                "id": res.get("id"),
                "text": context_text,
                "child_chunk": res.get("content", ""),
                "section": meta.get("section", ""),
                "source": source_doc,
                "source_doc_id": source_doc,
                "score": 1.0 - res.get("distance", 0.5)
            })
        return candidates
