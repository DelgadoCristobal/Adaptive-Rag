from typing import List, Dict, Any, Optional
import json
import time
from pathlib import Path
import logging
import numpy as np
from src.config import settings

logger = logging.getLogger(__name__)


class SemanticCache:
    """High-performance Semantic Query Cache using local dense embeddings and cosine similarity."""

    def __init__(
        self,
        threshold: float = 0.92,
        cache_file: Optional[Path] = None,
        model_name: Optional[str] = None,
    ):
        self.threshold = threshold
        self.cache_file = cache_file or (settings.data_cache_dir / "semantic_cache.json")
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name or settings.embedding_model
        self.encoder = None
        self.cache_entries: List[Dict[str, Any]] = []
        self._matrix: Optional[np.ndarray] = None
        self._load_cache()

    def _init_encoder(self):
        """Lazy loads sentence-transformers model to save memory until needed."""
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.model_name)

    def _rebuild_matrix(self):
        """Builds normalized 2D numpy matrix for instant vectorized cosine similarity."""
        if not self.cache_entries:
            self._matrix = None
            return
        embs = [e["embedding"] for e in self.cache_entries if e.get("embedding")]
        if not embs:
            self._matrix = None
            return
        arr = np.array(embs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = arr / norms

    def _load_cache(self):
        """Loads persisted semantic cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache_entries = json.load(f)
                self._rebuild_matrix()
                logger.info(f"Loaded {len(self.cache_entries)} entries from semantic cache.")
            except Exception as e:
                logger.warning(f"Failed to load cache file: {e}")
                self.cache_entries = []
                self._matrix = None

    def _save_cache(self):
        """Persists in-memory cache to disk."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache_entries, f, indent=2)
            self._rebuild_matrix()
        except Exception as e:
            logger.error(f"Failed to save semantic cache: {e}")

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Searches semantic cache for matches with cosine similarity >= threshold."""
        if self._matrix is None or not self.cache_entries:
            return None

        start_time = time.perf_counter()
        self._init_encoder()

        query_emb = self.encoder.encode(query, convert_to_numpy=True)
        norm_q = np.linalg.norm(query_emb)
        if norm_q == 0:
            return None
        query_emb_norm = (query_emb / norm_q).astype(np.float32)

        # Single vectorized BLAS dot product across all cached embeddings
        scores = np.dot(self._matrix, query_emb_norm)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_entry = self.cache_entries[best_idx]

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if best_score >= self.threshold:
            logger.info(
                f"[CACHE_HIT] Match: '{best_entry['query']}' (Similarity: {best_score:.3f}, Latency: {latency_ms:.1f}ms, Cost: $0.00)"
            )
            return {
                "answer": best_entry["answer"],
                "citations": best_entry.get("citations", []),
                "strategy": "SEMANTIC_CACHE",
                "matched_query": best_entry["query"],
                "similarity": round(best_score, 4),
                "latency_ms": round(latency_ms, 2),
                "cost_usd": 0.00,
            }

        logger.info(f"[CACHE_MISS] Max Similarity: {best_score:.3f} < Threshold: {self.threshold}")
        return None

    def find_similar(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.20,
        fast_threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """Finds top-k similar queries from cache history for real-time recommendation and fast retrieval."""
        if self._matrix is None or not self.cache_entries:
            return []

        start_time = time.perf_counter()
        self._init_encoder()

        query_emb = self.encoder.encode(query, convert_to_numpy=True)
        norm_q = np.linalg.norm(query_emb)
        if norm_q == 0:
            return []
        query_emb_norm = (query_emb / norm_q).astype(np.float32)

        scores = np.dot(self._matrix, query_emb_norm)
        top_indices = np.argsort(scores)[::-1]

        scored_entries = []
        for idx in top_indices:
            sim = float(scores[idx])
            if sim < min_similarity:
                break
            entry = self.cache_entries[idx]
            scored_entries.append({
                "query": entry.get("query", ""),
                "similarity": round(sim, 4),
                "similarity_pct": int(round(sim * 100)),
                "answer": entry.get("answer", ""),
                "citations": entry.get("citations", []),
                "strategy": entry.get("strategy", "HYBRID_ADAPTIVE"),
                "timestamp": entry.get("timestamp", 0),
                "eligible_for_fast_cache": sim >= fast_threshold
            })
            if len(scored_entries) >= top_k:
                break

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(f"find_similar found {len(scored_entries)} matches in {latency_ms:.1f}ms")
        return scored_entries

    def set(self, query: str, answer: str, citations: List[Dict[str, Any]], strategy: str):
        """Stores new query, embedding, and generated answer in the cache."""
        self._init_encoder()
        query_emb = self.encoder.encode(query, convert_to_numpy=True).tolist()

        entry = {
            "query": query,
            "embedding": query_emb,
            "answer": answer,
            "citations": citations,
            "strategy": strategy,
            "timestamp": time.time(),
        }

        self.cache_entries.append(entry)
        self._save_cache()
        logger.info(f"Stored query '{query}' in semantic cache (Total entries: {len(self.cache_entries)})")

    def clear(self):
        """Invalidates and empties the entire semantic cache."""
        self.cache_entries = []
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Semantic cache cleared.")

    def get_stats(self) -> Dict[str, Any]:
        """Returns cache health and usage statistics."""
        return {
            "total_entries": len(self.cache_entries),
            "threshold": self.threshold,
            "cache_file": str(self.cache_file),
            "model": self.model_name,
        }
