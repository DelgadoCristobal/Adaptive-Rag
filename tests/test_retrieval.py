from src.retrieval.reranker import CrossEncoderReranker
from src.generator.synthesizer import AnswerSynthesizer


def test_cross_encoder_reranker():
    reranker = CrossEncoderReranker()
    query = "What is GraphRAG?"
    candidates = [
        {"id": "1", "text": "Cooking recipe for apple pie", "source": "pie.md"},
        {"id": "2", "text": "GraphRAG integrates KùzuDB Knowledge Graphs with dense vector search", "source": "architecture.md"}
    ]

    reranked = reranker.rerank(query, candidates, top_k=2)
    assert len(reranked) == 2
    assert "GraphRAG" in reranked[0]["text"]


def test_answer_synthesizer():
    synthesizer = AnswerSynthesizer()
    query = "Explain GraphRAG architecture"
    contexts = [
        {"id": "1", "text": "GraphRAG uses KùzuDB for graph traversal and Chroma for vector retrieval.", "source": "doc1.md", "type": "text_chunk"}
    ]

    result = synthesizer.synthesize(query, contexts)
    assert "answer" in result
    assert len(result["citations"]) == 1
    assert result["citations"][0]["source"] == "doc1.md"
