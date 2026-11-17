from typing import List, Dict, Any
import logging
from src.utils.llm_client import LLMClient, qualify_model

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """You are an AI Enterprise Assistant with GraphRAG synthesis capabilities.
Answer the user's question using the retrieved context below (text chunks and knowledge graph paths).
Combine and reason over the evidence: extract the relevant facts, do simple calculations or comparisons when the numbers are present, and follow graph paths to connect entities.
Cite the source number after each key assertion using [1], [2], etc. (e.g. "...SLO target is 99.8% [1].").

If the context only partially answers the question, give the best answer the evidence supports and briefly note what is missing.
Say the information is unavailable ONLY when the context contains nothing related to the question. Do not invent facts that are not in the context.

LANGUAGE RULE: Reply in the same language as the question.

Retrieved Context:
{context_block}
"""

# Conservative chars-per-token for numeric/financial text, and tokens reserved for the prompt scaffolding.
CHARS_PER_TOKEN = 3
PROMPT_OVERHEAD_TOKENS = 350


def build_context_block(items: List[Dict[str, Any]], window_tokens: int) -> str:
    """The exact numbered context block the generator puts in its prompt (pure and deterministic).

    The context budget (`window_tokens`) is split evenly across items, with at least 300 chars per item.
    """
    if not items:
        return ""
    per_item_chars = max(300, window_tokens * CHARS_PER_TOKEN // len(items))
    return "\n\n".join(
        f"[{idx}] Source ({item.get('source', f'ref_{idx}')}): {item.get('text', '')[:per_item_chars]}"
        for idx, item in enumerate(items, start=1)
    )


class AnswerSynthesizer:
    """Grounded answer generation engine supporting Gemini, Groq, Ollama, and OpenAI."""

    def __init__(self, model: str = None):
        self.model = qualify_model(model)
        self.llm_client = LLMClient(default_model=self.model)

    def context_block(self, items: List[Dict[str, Any]]) -> str:
        """The context block `synthesize` sends to the LLM for these items (also used to judge faithfulness)."""
        # Ollama truncates overlong prompts from the START (dropping the instructions), so the context must
        # fit in num_ctx minus the tokens reserved for generation and prompt scaffolding.
        window = self.llm_client.num_ctx - self.llm_client.num_predict - PROMPT_OVERHEAD_TOKENS
        return build_context_block(items, window)

    def synthesize(self, query: str, retrieved_contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer with graph and document citations."""
        if not retrieved_contexts:
            return {
                "answer": "No relevant document chunks or graph connections were found to answer your question.",
                "citations": []
            }

        citations = [
            {"ref_num": idx, "source": item.get("source", f"ref_{idx}"), "type": item.get("type", "unknown")}
            for idx, item in enumerate(retrieved_contexts, start=1)
        ]
        context_block = self.context_block(retrieved_contexts)

        try:
            answer = self.llm_client.complete(
                prompt=query,
                system_prompt=SYNTHESIS_SYSTEM_PROMPT.format(context_block=context_block),
                model=self.model,
                temperature=0.2,
            )
            return {"answer": answer, "citations": citations, "raw_context": retrieved_contexts}
        except Exception as e:
            logger.warning(f"LLM Synthesis notice ({e}). Using deterministic fallback synthesis.")

        # Fallback grounded synthesis formatting
        answer_parts = [
            f"### Answer to: '{query}'\n",
            "Based on the retrieved Enterprise Knowledge Graph and Document Vault:\n"
        ]
        for item in retrieved_contexts:
            answer_parts.append(f"- **[{item.get('source')}]**: {item.get('text')}")

        answer = "\n".join(answer_parts)
        return {
            "answer": answer,
            "citations": citations,
            "raw_context": retrieved_contexts
        }
