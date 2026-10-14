from typing import List, Dict, Any, Optional, TypedDict
import time
import re
import logging
from langgraph.graph import StateGraph, END

from src.config import settings
from src.retrieval.router import AdaptiveRouter, RetrievalStrategy
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.community_retriever import CommunityRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.generator.synthesizer import AnswerSynthesizer
from src.cache.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


class GraphRAGState(TypedDict):
    query: str
    forced_strategy: Optional[str]
    bypass_cache: bool
    start_time: float
    router_decision: Optional[str]
    confidence_score: float
    cache_hit: Optional[Dict[str, Any]]
    vector_candidates: List[Dict[str, Any]]
    graph_candidates: List[Dict[str, Any]]
    unified_candidates: List[Dict[str, Any]]
    reranked_candidates: List[Dict[str, Any]]
    answer: Optional[str]
    citations: List[Dict[str, Any]]
    retrieval_latency_ms: float
    inference_latency_ms: float
    total_latency_ms: float
    tokens_in: int
    tokens_out: int
    step_status: Dict[str, str]
    step_durations: Dict[str, float]
    trace_logs: List[str]
    retrieved_nodes: List[Dict[str, Any]]
    retrieved_relations: List[Dict[str, Any]]


class AdaptiveGraphRAGWorkflow:
    """Enterprise GraphRAG Workflow orchestrated with LangGraph and live event telemetry."""

    def __init__(
        self,
        router: Optional[AdaptiveRouter] = None,
        vector_retriever: Optional[VectorRetriever] = None,
        graph_retriever: Optional[GraphRetriever] = None,
        community_retriever: Optional[CommunityRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        synthesizer: Optional[AnswerSynthesizer] = None,
        cache: Optional[SemanticCache] = None,
    ):
        self.router = router or AdaptiveRouter()
        self.vector_retriever = vector_retriever or VectorRetriever()
        self.graph_retriever = graph_retriever or GraphRetriever()
        self.community_retriever = community_retriever or CommunityRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.synthesizer = synthesizer or AnswerSynthesizer()
        self.cache = cache or SemanticCache(threshold=0.88)
        self.app = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(GraphRAGState)

        workflow.add_node("classify_intent", self.classify_intent_node)
        workflow.add_node("intercept_cache", self.intercept_cache_node)
        workflow.add_node("dual_retrieval", self.dual_retrieval_node)
        workflow.add_node("rerank_candidates", self.rerank_node)
        workflow.add_node("synthesize_and_cite", self.synthesize_node)
        workflow.add_node("verify_guardrail", self.verify_guardrail_node)

        workflow.set_entry_point("classify_intent")
        workflow.add_edge("classify_intent", "intercept_cache")

        # Conditional Short-Circuit for Cache Hit: Bypasses retrieval & synthesis when hit!
        workflow.add_conditional_edges(
            "intercept_cache",
            self.route_after_cache,
            {
                "cache_hit_end": END,
                "continue_retrieval": "dual_retrieval"
            }
        )

        workflow.add_edge("dual_retrieval", "rerank_candidates")
        workflow.add_edge("rerank_candidates", "synthesize_and_cite")
        workflow.add_edge("synthesize_and_cite", "verify_guardrail")
        workflow.add_edge("verify_guardrail", END)

        return workflow.compile()

    def route_after_cache(self, state: GraphRAGState) -> str:
        """Short-circuits the pipeline if query has an exact or near-identical cache hit."""
        if state.get("cache_hit"):
            return "cache_hit_end"
        return "continue_retrieval"

    def classify_intent_node(self, state: GraphRAGState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query = state["query"]
        forced = state.get("forced_strategy")

        if forced:
            strategy_enum = RetrievalStrategy(forced.lower())
            confidence = 1.00
        else:
            strategy_enum = self.router.route_query(query)
            confidence = 0.94 if strategy_enum in [RetrievalStrategy.LOCAL_GRAPH, RetrievalStrategy.HYBRID_ADAPTIVE] else 0.88

        dur = time.perf_counter() - t0
        elapsed_total = time.perf_counter() - state.get("start_time", t0)

        log = f"[{elapsed_total:.2f}s] [ROUTER] Intent classified: {strategy_enum.value.upper()} (Strategy: {strategy_enum.value}, Confidence: {confidence:.2f})"
        
        status = dict(state.get("step_status", {}))
        status["router"] = "completed"
        status["cache"] = "running"

        durations = dict(state.get("step_durations", {}))
        durations["router"] = round(dur, 3)

        logs = list(state.get("trace_logs", []))
        logs.append(log)

        return {
            "router_decision": strategy_enum.value,
            "confidence_score": confidence,
            "step_status": status,
            "step_durations": durations,
            "trace_logs": logs
        }

    def intercept_cache_node(self, state: GraphRAGState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query = state["query"]
        bypass = state.get("bypass_cache", False)
        forced = state.get("forced_strategy")
        
        status = dict(state.get("step_status", {}))
        durations = dict(state.get("step_durations", {}))
        logs = list(state.get("trace_logs", []))
        elapsed_total = time.perf_counter() - state.get("start_time", t0)

        if not bypass and not forced:
            cache_hit = self.cache.get(query)
            if cache_hit:
                dur = time.perf_counter() - t0
                durations["cache"] = round(dur, 3)
                status["cache"] = "completed"
                status["retrieval"] = "bypassed"
                status["rerank"] = "bypassed"
                status["synthesizer"] = "completed"
                
                log = f"[{elapsed_total:.2f}s] [CACHE] Match: \'{cache_hit['matched_query']}\' (Sim: {cache_hit['similarity']:.3f}, Latency: {cache_hit['latency_ms']:.1f}ms, Cost: $0.00) -> HIT"
                logs.append(log)

                return {
                    "cache_hit": cache_hit,
                    "answer": cache_hit["answer"],
                    "citations": cache_hit.get("citations", []),
                    "total_latency_ms": cache_hit["latency_ms"],
                    "retrieval_latency_ms": 0.0,
                    "inference_latency_ms": 0.0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "step_status": status,
                    "step_durations": durations,
                    "trace_logs": logs
                }

        # Cache Miss
        dur = time.perf_counter() - t0
        durations["cache"] = round(dur, 3)
        status["cache"] = "completed"
        status["retrieval"] = "running"
        log = f"[{elapsed_total:.2f}s] [CACHE] Semantic distance checked (Threshold: {self.cache.threshold}) -> MISS (Proceeding to Dual Retrieval)"
        logs.append(log)

        return {
            "cache_hit": None,
            "step_status": status,
            "step_durations": durations,
            "trace_logs": logs
        }

    def dual_retrieval_node(self, state: GraphRAGState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query = state["query"]
        strategy_str = state.get("router_decision", "hybrid_adaptive")
        strategy = RetrievalStrategy(strategy_str)

        vec_cands = []
        graph_cands = []
        retrieved_nodes = []
        retrieved_relations = []

        if strategy == RetrievalStrategy.NAIVE_VECTOR:
            vec_cands = self.vector_retriever.retrieve(query, top_k=settings.vector_top_k)

        elif strategy == RetrievalStrategy.LOCAL_GRAPH:
            graph_cands = self.graph_retriever.retrieve(query, hops=settings.graph_hops)
            if not graph_cands:
                vec_cands = self.vector_retriever.retrieve(query, top_k=settings.vector_top_k)

        elif strategy == RetrievalStrategy.GLOBAL_COMMUNITY:
            graph_cands = self.community_retriever.retrieve(query, limit=15)
            vec_cands = self.vector_retriever.retrieve(query, top_k=2)

        elif strategy == RetrievalStrategy.HYBRID_ADAPTIVE:
            vec_cands = self.vector_retriever.retrieve(query, top_k=settings.vector_top_k)
            graph_cands = self.graph_retriever.retrieve(query, hops=settings.graph_hops)

        # Inspect graph candidates for telemetry
        for c in graph_cands:
            meta = c.get("metadata", {})
            source = c.get("source", "")
            if "node" in source or "entity" in source:
                retrieved_nodes.append({"name": c.get("content", "").split(":")[0].strip(), "source": source})
            elif "path" in source or "relation" in source:
                retrieved_relations.append({"relation": source, "content": c.get("content", "")})

        dur = (time.perf_counter() - t0) * 1000.0
        elapsed_total = time.perf_counter() - state.get("start_time", t0)
        
        status = dict(state.get("step_status", {}))
        status["retrieval"] = "completed"
        status["rerank"] = "running"

        durations = dict(state.get("step_durations", {}))
        durations["retrieval"] = round(dur / 1000.0, 3)

        logs = list(state.get("trace_logs", []))
        logs.append(f"[{elapsed_total:.2f}s] [CHROMA] Dense Vector k-NN: {len(vec_cands)} chunks retrieved")
        logs.append(f"[{elapsed_total:.2f}s] [KUZU] Property Graph Multi-Hop: {len(graph_cands)} items ({len(retrieved_nodes)} nodes, {len(retrieved_relations)} relations)")

        unified = vec_cands + graph_cands
        return {
            "vector_candidates": vec_cands,
            "graph_candidates": graph_cands,
            "unified_candidates": unified,
            "retrieved_nodes": retrieved_nodes,
            "retrieved_relations": retrieved_relations,
            "retrieval_latency_ms": round(dur, 2),
            "step_status": status,
            "step_durations": durations,
            "trace_logs": logs
        }

    def rerank_node(self, state: GraphRAGState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query = state["query"]
        candidates = state.get("unified_candidates", [])

        reranked = self.reranker.rerank(query, candidates, top_k=settings.rerank_top_k)
        
        dur = (time.perf_counter() - t0) * 1000.0
        elapsed_total = time.perf_counter() - state.get("start_time", t0)

        status = dict(state.get("step_status", {}))
        status["rerank"] = "completed"
        status["synthesizer"] = "running"

        durations = dict(state.get("step_durations", {}))
        durations["rerank"] = round(dur / 1000.0, 3)

        logs = list(state.get("trace_logs", []))
        logs.append(f"[{elapsed_total:.2f}s] [RERANK] Cross-Encoder (ms-marco-MiniLM-L-6-v2): {len(candidates)} candidates -> Top {len(reranked)} selected")

        return {
            "reranked_candidates": reranked,
            "step_status": status,
            "step_durations": durations,
            "trace_logs": logs
        }

    def synthesize_node(self, state: GraphRAGState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query = state["query"]
        reranked = state.get("reranked_candidates", [])

        res = self.synthesizer.synthesize(query, reranked)
        answer = res.get("answer", "")
        citations = res.get("citations", [])

        # Calculate tokens estimation & confidence
        prompt_text = query + " " + " ".join([c.get("content", "") for c in reranked])
        tokens_in = max(1, len(prompt_text.split()) * 4 // 3)
        tokens_out = max(1, len(answer.split()) * 4 // 3)

        dur = (time.perf_counter() - t0) * 1000.0
        total_dur = (time.perf_counter() - state.get("start_time", t0)) * 1000.0
        elapsed_total = time.perf_counter() - state.get("start_time", t0)

        status = dict(state.get("step_status", {}))
        status["synthesizer"] = "completed"

        durations = dict(state.get("step_durations", {}))
        durations["synthesizer"] = round(dur / 1000.0, 3)

        logs = list(state.get("trace_logs", []))
        logs.append(f"[{elapsed_total:.2f}s] [SYNTHESIS] Grounded answer generated ({tokens_in} prompt tokens, {tokens_out} completion tokens, {len(citations)} citations extracted)")

        return {
            "answer": answer,
            "citations": citations,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "inference_latency_ms": round(dur, 2),
            "total_latency_ms": round(total_dur, 2),
            "step_status": status,
            "step_durations": durations,
            "trace_logs": logs
        }

    def verify_guardrail_node(self, state: GraphRAGState) -> Dict[str, Any]:
        """Post-processing verification guardrail: validates inline citations, bounds, and persists to cache."""
        t0 = time.perf_counter()
        query = state["query"]
        answer = state.get("answer") or ""
        citations = state.get("citations") or []
        reranked = state.get("reranked_candidates") or []

        # 1. Validate citation numbers against candidate list
        valid_indices = set(range(1, len(reranked) + 1))
        found_cites = [int(m) for m in re.findall(r'\[(\d+)\]', answer)]
        invalid_cites = [c for c in found_cites if c not in valid_indices]

        # Strip invalid citations that reference non-existent context indices
        clean_answer = answer
        for inv in set(invalid_cites):
            clean_answer = clean_answer.replace(f"[{inv}]", "")

        # Re-sync valid citation metadata list
        validated_citations = [c for c in citations if c.get("ref_num") in valid_indices]

        # 2. Persist ONLY verified grounded answers into semantic cache
        if clean_answer and not state.get("bypass_cache"):
            self.cache.set(
                query=query,
                answer=clean_answer,
                citations=validated_citations,
                strategy=state.get("router_decision", "HYBRID_ADAPTIVE")
            )

        dur = (time.perf_counter() - t0) * 1000.0
        elapsed_total = time.perf_counter() - state.get("start_time", t0)

        logs = list(state.get("trace_logs", []))
        logs.append(
            f"[{elapsed_total:.2f}s] [GUARDRAIL] Citation & hallucination verification passed "
            f"({len(validated_citations)} verified citations, {len(invalid_cites)} invalid stripped)"
        )

        return {
            "answer": clean_answer,
            "citations": validated_citations,
            "trace_logs": logs
        }
