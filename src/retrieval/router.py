from typing import Optional
from enum import Enum
import logging
from src.utils.llm_client import LLMClient, qualify_model

logger = logging.getLogger(__name__)


class RetrievalStrategy(str, Enum):
    NAIVE_VECTOR = "naive_vector"          # Specific factual queries, definitions, config lookups
    LOCAL_GRAPH = "local_graph"            # Multi-hop relationships between specific entities, dependencies
    GLOBAL_COMMUNITY = "global_community"  # Holistic questions, high-level summaries, macro themes
    HYBRID_ADAPTIVE = "hybrid_adaptive"    # Complex questions requiring both dense facts and topology


ROUTER_PROMPT = """Classify the following query into exactly ONE retrieval strategy:

1. NAIVE_VECTOR: Specific facts, numbers, single-item definitions, configurations, or direct document lookups.
2. LOCAL_GRAPH: Relationships between entities, architecture dependencies, team/role connections, structural paths.
3. GLOBAL_COMMUNITY: Macro summaries, broad overviews, cross-document themes, high-level synthesis across all docs.
4. HYBRID_ADAPTIVE: Complex questions comparing components, analyzing causes, or requiring both factual metrics and system connections.

Query: "{query}"

Respond ONLY with the strategy name: NAIVE_VECTOR, LOCAL_GRAPH, GLOBAL_COMMUNITY, or HYBRID_ADAPTIVE.
Strategy:"""


class AdaptiveRouter:
    """Dynamically classifies and routes user queries to the optimal retrieval strategy with low latency."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = qualify_model(model_name)
        self.llm_client = LLMClient(default_model=self.model_name)

    def route_query(self, query: str) -> RetrievalStrategy:
        """Analyzes query complexity and intent to determine the optimal retrieval path.

        Args:
            query: Input user question.

        Returns:
            Selected RetrievalStrategy enum.
        """
        # Fast Heuristic Check (Zero-cost / instant path for common patterns)
        query_lower = query.lower()
        if any(word in query_lower for word in ["summarize all", "overview of all", "general theme", "resumen general", "panorama global"]):
            logger.info("Heuristic routed query '%s' -> GLOBAL_COMMUNITY", query)
            return RetrievalStrategy.GLOBAL_COMMUNITY

        # LLM Semantic Classification (ultra-compact output)
        try:
            response = self.llm_client.complete(
                prompt=ROUTER_PROMPT.format(query=query),
                system_prompt="You are a strict query intent classifier. Return only the strategy identifier.",
                model=self.model_name,
                temperature=0.0,
                max_tokens=8,
            )
            cleaned = response.strip().upper()
            for strategy in RetrievalStrategy:
                if strategy.name in cleaned:
                    logger.info("LLM routed query '%s' -> [%s]", query, strategy.value)
                    return strategy
        except Exception as e:
            logger.warning(f"Router LLM classification notice ({e}). Using heuristic fallback.")

        # Heuristic fallback
        if any(word in query_lower for word in ["relacion", "conectado", "impacta", "dependencia", "relationship", "connect", "between", "how does", "role of"]):
            return RetrievalStrategy.LOCAL_GRAPH
        if any(word in query_lower for word in ["compara", "analiza", "multi-hop", "compare", "why", "difference", "impact"]):
            return RetrievalStrategy.HYBRID_ADAPTIVE
        if any(word in query_lower for word in ["resumen", "temas", "summary", "overview"]):
            return RetrievalStrategy.GLOBAL_COMMUNITY

        # Default to NAIVE_VECTOR for minimal cost and standard factual precision
        return RetrievalStrategy.NAIVE_VECTOR
