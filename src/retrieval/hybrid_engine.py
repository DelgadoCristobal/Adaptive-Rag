from typing import List, Dict, Any
import logging
from src.config import settings
from src.graph.kuzu_store import KuzuGraphStore
from src.vector.vector_store import VectorStoreManager
from .reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class HybridRetrievalEngine:
    """Hybrid Retriever combining Dense Vector Search + Cypher Multi-Hop Graph Traversal."""

    def __init__(
        self,
        graph_store: KuzuGraphStore = None,
        vector_store: VectorStoreManager = None,
        reranker: CrossEncoderReranker = None,
    ):
        self.graph_store = graph_store or KuzuGraphStore()
        self.vector_store = vector_store or VectorStoreManager()
        self.reranker = reranker or CrossEncoderReranker()

    def retrieve(self, query: str, top_k_vector: int = None, hops: int = None) -> List[Dict[str, Any]]:
        """Retrieve candidate contexts via vector search + graph traversal, then rerank."""
        top_k_vector = top_k_vector or settings.vector_top_k
        hops = hops or settings.graph_hops

        candidates = []

        # 1. Vector Search
        vector_results = self.vector_store.similarity_search(query, top_k=top_k_vector)
        for res in vector_results:
            candidates.append({
                "type": "text_chunk",
                "id": res.get("id"),
                "text": res.get("content", ""),
                "source": res.get("metadata", {}).get("doc_id", "unknown"),
                "score": 1.0 - res.get("distance", 0.5)
            })

        # 2. Graph Traversal: Find seed entities in query
        matched_entities = self.graph_store.search_entities_by_keyword(query)
        # If no direct match, split words and query entities
        if not matched_entities:
            for word in query.split():
                if len(word) > 3:
                    matched_entities.extend(self.graph_store.search_entities_by_keyword(word))

        # Deduplicate seed entity IDs
        seen_entity_ids = set()
        for ent in matched_entities:
            ent_id = ent.get("id")
            if ent_id and ent_id not in seen_entity_ids:
                seen_entity_ids.add(ent_id)

                # Add seed entity node description
                candidates.append({
                    "type": "graph_node",
                    "id": ent_id,
                    "text": f"Entity: {ent.get('name')} (Type: {ent.get('type')}) - {ent.get('description')}",
                    "source": f"graph_node:{ent_id}",
                    "score": 0.9
                })

                # Perform K-hop neighborhood expansion
                neighborhood = self.graph_store.query_k_hop_neighborhood(ent_id, hops=hops)
                for path in neighborhood:
                    candidates.append({
                        "type": "graph_path",
                        "id": f"{path['source']}->{path['target']}",
                        "text": f"Graph Relation Path: [{path['source']}] is connected to [{path['target_name']} ({path['target_type']})]: {path['target_desc']}",
                        "source": f"graph_path:{path['source']}->{path['target']}",
                        "score": 0.85
                    })

        # 3. Cross-Encoder Re-ranking
        reranked_results = self.reranker.rerank(query, candidates, top_k=settings.rerank_top_k)
        return reranked_results
