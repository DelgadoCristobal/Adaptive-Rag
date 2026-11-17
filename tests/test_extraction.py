import json
import logging

import pytest

from src.config import settings
from src.graph.extractor import (
    EXTRACTION_PROMPT,
    EntityRelationExtractor,
    effective_chunk_chars,
    split_into_chunks,
)

PROMPT_MARKER = "Text to analyze:\n"


def _chunk_of(prompt: str) -> str:
    return prompt.split(PROMPT_MARKER, 1)[1][:-1]


def _payload(entities=(), relations=()):
    return json.dumps({"entities": list(entities), "relationships": list(relations)})


class StubClient:
    """LLM client double: returns (or raises) one scripted response per call."""

    def __init__(self, responses, num_ctx=2048):
        self.responses = list(responses)
        self.num_ctx = num_ctx
        self.calls = []

    def complete(self, prompt, system_prompt=None, model=None, temperature=0.1, max_tokens=None):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        response = self.responses.pop(0) if self.responses else _payload()
        if isinstance(response, Exception):
            raise response
        return response


def _extractor(client, model="ollama/test-model"):
    extractor = EntityRelationExtractor.__new__(EntityRelationExtractor)
    extractor.model = model
    extractor.llm_client = client
    return extractor


@pytest.fixture
def small_chunks(monkeypatch):
    monkeypatch.setattr(settings, "extraction_chunk_chars", 200)
    monkeypatch.setattr(settings, "extraction_max_tokens", 256)


def _long_text(paragraphs=6):
    sentence = "Alpha Corp builds Beta engines in Gamma city for many years now."
    return "\n\n".join(" ".join([sentence] * 2) for _ in range(paragraphs))


def test_split_into_chunks_respects_limit_and_boundaries():
    text = _long_text()
    chunks = split_into_chunks(text, 200)
    assert len(chunks) > 1
    assert all(0 < len(c) <= 200 for c in chunks)
    assert all(c.endswith(".") for c in chunks)  # paragraph/sentence boundaries, not mid-word cuts


def test_split_into_chunks_hard_splits_oversized_sentence():
    chunks = split_into_chunks("x" * 450, 200)
    assert [len(c) for c in chunks] == [200, 200, 50]


def test_long_text_is_extracted_in_multiple_chunks_within_limit(small_chunks):
    client = StubClient([])
    extractor = _extractor(client)

    result = extractor.extract_from_text(_long_text(), source_doc="doc.md")

    assert len(client.calls) > 1
    assert all(len(_chunk_of(call["prompt"])) <= 200 for call in client.calls)
    assert all(call["max_tokens"] == 256 for call in client.calls)
    assert result.chunks_total == len(client.calls)
    assert result.chunks_failed == 0


def test_failing_chunk_contributes_nothing_and_is_logged(small_chunks, caplog):
    good = _payload(
        [{"id": "Alpha", "name": "Alpha", "type": "ORG", "description": "A company"},
         {"id": "Beta", "name": "Beta", "type": "PRODUCT", "description": "An engine"}],
        [{"source": "Alpha", "target": "Beta", "relation_type": "BUILDS"}],
    )
    client = StubClient([good, RuntimeError("model timed out"), "no json here"])
    extractor = _extractor(client)

    with caplog.at_level(logging.WARNING, logger="src.graph.extractor"):
        result = extractor.extract_from_text(_long_text(3), source_doc="doc.md")

    assert result.chunks_total == 3
    assert result.chunks_failed == 2
    assert {e.id for e in result.entities} == {"Alpha", "Beta"}
    assert [(r.source, r.target, r.relation_type) for r in result.relations] == [("Alpha", "Beta", "BUILDS")]
    assert not any("Extracted concept from" in e.description for e in result.entities)
    assert not any(r.relation_type == "ASSOCIATED_WITH" for r in result.relations)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("doc.md" in m and "chunk 2/3" in m and "model timed out" in m for m in warnings)
    assert any("doc.md" in m and "chunk 3/3" in m for m in warnings)


def test_all_chunks_failing_returns_empty_result(small_chunks, caplog):
    client = StubClient([RuntimeError("down")] * 10)
    extractor = _extractor(client)

    with caplog.at_level(logging.WARNING, logger="src.graph.extractor"):
        result = extractor.extract_from_text(_long_text(3), source_doc="doc.md")

    assert result.entities == [] and result.relations == []
    assert result.chunks_failed == result.chunks_total == 3
    assert any("all 3 chunks failed" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


def test_merge_dedups_entities_and_relations_across_chunks(small_chunks):
    first = _payload(
        [{"id": "Alpha", "name": "Alpha", "type": "ORG", "description": "Short", "aliases": ["A"]},
         {"id": "Beta", "name": "Beta", "type": "PRODUCT", "description": ""}],
        [{"source": "Alpha", "target": "Beta", "relation_type": "BUILDS", "description": "x", "weight": 0.4}],
    )
    second = _payload(
        [{"id": "alpha", "name": "Alpha Corp", "type": "ORG", "description": "A much longer description", "aliases": ["AC"]},
         {"id": "Beta", "name": "Beta", "type": "PRODUCT", "description": "Engine"}],
        [{"source": "Alpha", "target": "Beta", "relation_type": "BUILDS", "description": "longer one", "weight": 0.9}],
    )
    client = StubClient([first, second])
    extractor = _extractor(client)

    result = extractor.extract_from_text(_long_text(2), source_doc="doc.md")

    assert result.chunks_total == 2
    by_id = {e.id: e for e in result.entities}
    assert set(by_id) == {"Alpha", "Beta"}
    assert by_id["Alpha"].name == "Alpha"
    assert by_id["Alpha"].description == "A much longer description"
    assert set(by_id["Alpha"].aliases) == {"A", "AC"}
    assert by_id["Beta"].description == "Engine"
    assert len(result.relations) == 1
    assert result.relations[0].description == "longer one"
    assert result.relations[0].weight == 0.9


def test_empty_text_makes_no_calls():
    client = StubClient([])
    result = _extractor(client).extract_from_text("   ", source_doc="doc.md")
    assert client.calls == [] and result.entities == [] and result.chunks_total == 0


def test_effective_chunk_chars_keeps_configured_size_when_it_fits():
    assert effective_chunk_chars(1500, 1024, num_ctx=2048, prompt_chars=len(EXTRACTION_PROMPT)) == 1500


def test_effective_chunk_chars_shrinks_and_warns_once(small_chunks, monkeypatch, caplog):
    monkeypatch.setattr(settings, "extraction_chunk_chars", 4000)
    monkeypatch.setattr(settings, "extraction_max_tokens", 1024)
    shrunk = effective_chunk_chars(4000, 1024, num_ctx=2048, prompt_chars=1400)
    assert shrunk == (2048 - 1024 - 1400 // 4) * 4
    assert effective_chunk_chars(4000, 2048, num_ctx=2048, prompt_chars=1400) >= 200  # floor, never zero

    client = StubClient([], num_ctx=2048)
    extractor = _extractor(client)
    with caplog.at_level(logging.WARNING, logger="src.graph.extractor"):
        extractor.extract_from_text(_long_text(1), source_doc="a.md")
        extractor.extract_from_text(_long_text(1), source_doc="b.md")
    budget_warnings = [r for r in caplog.records if "num_ctx" in r.getMessage()]
    assert len(budget_warnings) == 1


def test_budget_is_not_applied_to_non_ollama_models(monkeypatch):
    monkeypatch.setattr(settings, "extraction_chunk_chars", 4000)
    client = StubClient([], num_ctx=2048)
    extractor = _extractor(client, model="gemini/gemini-2.0-flash")
    extractor.extract_from_text("word " * 1500, source_doc="doc.md")
    assert max(len(_chunk_of(c["prompt"])) for c in client.calls) > 2048
