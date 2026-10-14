from typing import List, Dict, Any, Optional, Tuple
import logging
from src.config import settings
from .router import AdaptiveRouter, RetrievalStrategy
from .vector_retriever import VectorRetriever
from .graph_retriever import GraphRetriever
from .community_retriever import CommunityRetriever
from .reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class HybridRetrievalEngine:
    """Intelligent Multi-Strategy Retrieval Engine with Semantic Routing & Cross-Encoder Reranking."""

    def __init__(
        self,
        router: Optional[AdaptiveRouter] = None,
        vector_retriever: Optional[VectorRetriever] = None,
        graph_retriever: Optional[GraphRetriever] = None,
        community_retriever: Optional[CommunityRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.router = router or AdaptiveRouter()
        self.vector_retriever = vector_retriever or VectorRetriever()
        self.graph_retriever = graph_retriever or GraphRetriever()
        self.community_retriever = community_retriever or CommunityRetriever()
        self.reranker = reranker or CrossEncoderReranker()

    def retrieve(
        self,
        query: str,
        strategy: Optional[RetrievalStrategy] = None,
        top_k: int = None
    ) -> Tuple[List[Dict[str, Any]], RetrievalStrategy]:
        """Retrieve candidates using selected or automatically routed strategy, then rerank.

        Args:
            query: Input user question.
            strategy: Optional explicit retrieval strategy to override router.
            top_k: Number of final reranked results to return.

        Returns:
            Tuple of (reranked candidate list, strategy used).
        """
        chosen_strategy = strategy or self.router.route_query(query)
        top_k = top_k or settings.rerank_top_k
        candidates: List[Dict[str, Any]] = []

        if chosen_strategy == RetrievalStrategy.NAIVE_VECTOR:
            candidates = self.vector_retriever.retrieve(query, top_k=settings.vector_top_k)

        elif chosen_strategy == RetrievalStrategy.LOCAL_GRAPH:
            candidates = self.graph_retriever.retrieve(query, hops=settings.graph_hops)
            # If graph has no matching seed entities, fallback to vector chunks to prevent empty answers
            if not candidates:
                candidates = self.vector_retriever.retrieve(query, top_k=settings.vector_top_k)

        elif chosen_strategy == RetrievalStrategy.GLOBAL_COMMUNITY:
            candidates = self.community_retriever.retrieve(query, limit=15)
            # Include top vector hits for high-relevance grounding
            candidates.extend(self.vector_retriever.retrieve(query, top_k=2))

        elif chosen_strategy == RetrievalStrategy.HYBRID_ADAPTIVE:
            vec_candidates = self.vector_retriever.retrieve(query, top_k=settings.vector_top_k)
            graph_candidates = self.graph_retriever.retrieve(query, hops=settings.graph_hops)
            candidates = vec_candidates + graph_candidates

        # Re-rank unified candidate pool with CrossEncoder
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        return reranked, chosen_strategy
