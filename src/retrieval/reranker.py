from typing import List, Dict, Any, Tuple
import logging
from src.config import settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder scoring module using sentence-transformers / bge-reranker."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.reranker_model
        self.encoder = None
        self._init_encoder()

    def _init_encoder(self):
        """Initialize sentence-transformers CrossEncoder if installed."""
        try:
            from sentence_transformers import CrossEncoder
            self.encoder = CrossEncoder(self.model_name)
            logger.info(f"Initialized CrossEncoder model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize CrossEncoder '{self.model_name}': {e}. Using fallback reranker.")

    def rerank(
        self, query: str, candidate_contexts: List[Dict[str, Any]], top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Score candidate context items against query and return top_k reranked items."""
        top_k = top_k or settings.rerank_top_k
        if not candidate_contexts:
            return []

        if self.encoder:
            try:
                pairs = [(query, item.get("text", "")) for item in candidate_contexts]
                scores = self.encoder.predict(pairs)
                
                scored_items = []
                for item, score in zip(candidate_contexts, scores):
                    item_copy = dict(item)
                    item_copy["rerank_score"] = float(score)
                    scored_items.append(item_copy)

                scored_items.sort(key=lambda x: x["rerank_score"], reverse=True)
                return scored_items[:top_k]
            except Exception as e:
                logger.warning(f"CrossEncoder reranking error: {e}")

        # Fallback keyword overlap scoring
        import re
        scored_items = []
        q_words = set(re.findall(r'\w+', query.lower()))
        for item in candidate_contexts:
            text = item.get("text", "").lower()
            t_words = set(re.findall(r'\w+', text))
            overlap = len(q_words.intersection(t_words))
            item_copy = dict(item)
            item_copy["rerank_score"] = float(overlap)
            scored_items.append(item_copy)

        scored_items.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_items[:top_k]
