from typing import List, Dict, Any
import logging
import re
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
            import torch
            from sentence_transformers import CrossEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.encoder = CrossEncoder(self.model_name, device=device)
            logger.info(f"Initialized CrossEncoder model: {self.model_name} on {device}")
        except Exception as e:
            logger.warning(f"Failed to initialize CrossEncoder '{self.model_name}': {e}. Using fallback reranker.")

    @staticmethod
    def _top_distinct(scored_items: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Top-k by score, skipping candidates with identical text (Small-to-Big children share one parent section)."""
        scored_items.sort(key=lambda x: x["rerank_score"], reverse=True)
        distinct = list({item.get("text", ""): item for item in reversed(scored_items)}.values())
        distinct.sort(key=lambda x: x["rerank_score"], reverse=True)
        return distinct[:top_k]

    def rerank(
        self, query: str, candidate_contexts: List[Dict[str, Any]], top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Score candidate context items against query and return top_k reranked items."""
        top_k = top_k or settings.rerank_top_k
        if not candidate_contexts:
            return []

        if self.encoder:
            try:
                # Deduplicate and cap texts to 512 chars to minimize cross-encoder inference cost
                truncated_texts = [item.get("text", "")[:512] for item in candidate_contexts]
                unique_texts = list(dict.fromkeys(truncated_texts))
                pairs = [(query, t) for t in unique_texts]
                scores = self.encoder.predict(pairs, batch_size=32)
                score_map = {t: float(s) for t, s in zip(unique_texts, scores)}

                scored_items = []
                for item, t in zip(candidate_contexts, truncated_texts):
                    item_copy = dict(item)
                    item_copy["rerank_score"] = score_map.get(t, 0.0)
                    scored_items.append(item_copy)

                top = self._top_distinct(scored_items, top_k)
                # Drop weak candidates so a larger pool cannot pad the context with noise; always keep the best one.
                return [c for c in top if c["rerank_score"] >= settings.rerank_min_score] or top[:1]
            except Exception as e:
                logger.warning(f"CrossEncoder reranking error: {e}")

        # Fallback keyword overlap scoring
        scored_items = []
        q_words = set(re.findall(r'\w+', query.lower()))
        for item in candidate_contexts:
            text = item.get("text", "").lower()
            t_words = set(re.findall(r'\w+', text))
            overlap = len(q_words.intersection(t_words))
            item_copy = dict(item)
            item_copy["rerank_score"] = float(overlap)
            scored_items.append(item_copy)

        return self._top_distinct(scored_items, top_k)
