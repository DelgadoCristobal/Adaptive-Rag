from typing import List, Dict, Any
import logging
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
from src.config import settings

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """You are an AI Enterprise Assistant with GraphRAG synthesis capabilities.
Answer the user's question grounded STRICTLY on the retrieved context below (text chunks and knowledge graph paths).
Every key assertion MUST include a explicit citation trace back to the retrieved context using [Citation: SourceID].

If the retrieved context does not contain enough information, clearly state what is known and what is missing.

Retrieved Context:
{context_block}
"""


class AnswerSynthesizer:
    """Grounded answer generation engine with citation traceability."""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.llm_model

        if self.api_key and self.api_key != "mock-key":
            self.client = OpenAI(api_key=self.api_key, base_url=settings.openai_base_url)
        else:
            self.client = None

    def synthesize(self, query: str, retrieved_contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer with graph and document citations."""
        if not retrieved_contexts:
            return {
                "answer": "No relevant document chunks or graph connections were found to answer your question.",
                "citations": []
            }

        # Build context block with explicit source IDs
        context_lines = []
        citations = []
        for idx, item in enumerate(retrieved_contexts, start=1):
            source_id = item.get("source", f"ref_{idx}")
            text = item.get("text", "")
            context_lines.append(f"[{idx}] Source ({source_id}): {text}")
            citations.append({"ref_num": idx, "source": source_id, "type": item.get("type", "unknown")})

        context_block = "\n\n".join(context_lines)

        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT.format(context_block=context_block)},
                        {"role": "user", "content": query}
                    ],
                    temperature=0.2,
                )
                answer = response.choices[0].message.content
                return {
                    "answer": answer,
                    "citations": citations,
                    "raw_context": retrieved_contexts
                }
            except Exception as e:
                logger.warning(f"LLM Synthesis failed: {e}. Using deterministic fallback synthesis.")

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
