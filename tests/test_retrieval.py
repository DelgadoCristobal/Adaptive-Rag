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
    assert "GraphRAG" in reranked[0]["text"]
    assert all("pie" not in c["text"] for c in reranked)  # irrelevant candidate falls below the score floor


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


class _FakeEncoder:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs, batch_size=32):
        return [self.scores[text] for _, text in pairs]


def _reranker_with(scores):
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.encoder = _FakeEncoder(scores)
    return reranker


def test_reranker_drops_candidates_below_score_floor():
    reranker = _reranker_with({"good": 5.0, "ok": 1.0, "noise": -10.0})
    candidates = [{"text": t} for t in ("noise", "ok", "good")]
    assert [c["text"] for c in reranker.rerank("q", candidates, top_k=3)] == ["good", "ok"]


def test_reranker_keeps_best_candidate_when_all_below_floor():
    reranker = _reranker_with({"a": -3.0, "b": -9.0})
    reranked = reranker.rerank("q", [{"text": "b"}, {"text": "a"}], top_k=3)
    assert [c["text"] for c in reranked] == ["a"]
