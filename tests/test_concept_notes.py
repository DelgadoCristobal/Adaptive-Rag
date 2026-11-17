import re

import pytest

import src.retrieval.graph_retriever as graph_retriever
from src.evaluation.dataset_filter import graph_indexed_docs
from src.evaluation.evaluator import RAGEvaluator
from src.graph.extractor import EntityRelationExtractor
from src.graph.schema import Entity, ExtractionResult, Relation, concept_note_path, parse_note_sources


LEGACY_NOTE = """---
id: Legacy_Entity
type: CONCEPT
tags:
- concept
- concept
source: legacy_doc.md
aliases: []
---

# Legacy Entity
"""


@pytest.fixture
def extractor():
    return EntityRelationExtractor.__new__(EntityRelationExtractor)  # note writing needs no LLM client


@pytest.fixture(autouse=True)
def fresh_doc_cache():
    graph_retriever.reset_entity_doc_cache()
    yield
    graph_retriever.reset_entity_doc_cache()


def _write(extractor, vault, doc, entity, relations=()):
    extractor.create_atomic_concept_notes(
        ExtractionResult(entities=[entity], relations=list(relations)), source_doc=doc, vault_dir=vault
    )


def test_note_path_is_filesystem_safe_and_hashed(tmp_path):
    path = concept_note_path(tmp_path, "A/B:C*D?E|F..G")
    assert path.parent == tmp_path / "concepts"
    assert re.fullmatch(r"concept_[a-z0-9_]{1,60}_[0-9a-f]{8}\.md", path.name)
    assert concept_note_path(tmp_path, "x" * 300).name.startswith("concept_" + "x" * 60 + "_")


def test_note_paths_do_not_collide_on_case_or_punctuation(tmp_path):
    ids = ["Apple", "apple", "A/B", "A:B", "A_B", "Ünïcode", "Unicode"]
    names = {concept_note_path(tmp_path, i).name for i in ids}
    assert len(names) == len(ids)


def test_same_entity_from_two_docs_accumulates_sources(tmp_path, extractor):
    _write(extractor, tmp_path, "doc_a.md", Entity(id="Acme", name="Acme", description="A rich description of Acme"),
           [Relation(source="Acme", target="Widget", relation_type="MAKES")])
    _write(extractor, tmp_path, "doc_b.md", Entity(id="Acme", name="Acme", description="Short"),
           [Relation(source="Acme", target="Gadget", relation_type="SELLS")])
    _write(extractor, tmp_path, "doc_a.md", Entity(id="Acme", name="Acme", aliases=["ACME Inc"]))

    notes = list((tmp_path / "concepts").glob("*.md"))
    assert len(notes) == 1
    text = notes[0].read_text(encoding="utf-8")
    assert parse_note_sources(text) == ["doc_a.md", "doc_b.md"]
    assert "sources: [doc_a.md, doc_b.md]" in text
    assert "A rich description of Acme" in text
    assert "[[Widget|MAKES]]" in text and "[[Gadget|SELLS]]" in text
    assert "ACME Inc" in text


def test_parse_note_sources_accepts_legacy_source_field():
    assert parse_note_sources(LEGACY_NOTE) == ["legacy_doc.md"]
    assert parse_note_sources("no frontmatter") == []


def test_graph_indexed_docs_unions_new_and_legacy_notes(tmp_path, extractor):
    _write(extractor, tmp_path, "doc_a.md", Entity(id="Acme", name="Acme"))
    _write(extractor, tmp_path, "doc_b.md", Entity(id="Acme", name="Acme"))
    (tmp_path / "concepts" / "concept_Legacy_Entity.md").write_text(LEGACY_NOTE, encoding="utf-8")

    assert graph_indexed_docs(tmp_path) == {"doc_a", "doc_b", "legacy_doc"}


class _FakeStore:
    def search_entities_by_keywords(self, keywords):
        return [{"id": "Acme", "name": "Acme", "type": "ORGANIZATION", "description": "maker"}]

    def query_k_hop_neighborhood(self, entity_id, hops=2):
        return [{"source": "Acme", "target": "Legacy_Entity", "target_name": "Legacy", "target_type": "CONCEPT", "target_desc": ""}]


def test_graph_candidates_expose_all_source_docs(tmp_path, extractor, monkeypatch):
    _write(extractor, tmp_path, "doc_a.md", Entity(id="Acme", name="Acme"))
    _write(extractor, tmp_path, "doc_b.md", Entity(id="Acme", name="Acme"))
    (tmp_path / "concepts" / "concept_Legacy_Entity.md").write_text(LEGACY_NOTE, encoding="utf-8")
    monkeypatch.setattr(graph_retriever.settings, "data_vault_dir", tmp_path)

    node, path = graph_retriever.GraphRetriever(graph_store=_FakeStore()).retrieve("Acme", hops=1)

    assert node["source_doc_id"] == "doc_a"
    assert node["source_doc_ids"] == ["doc_a", "doc_b"]
    assert path["source_doc_id"] == "doc_a"
    assert path["source_doc_ids"] == ["doc_a", "doc_b", "legacy_doc"]  # union of both endpoints' docs


def _evaluator():
    return RAGEvaluator.__new__(RAGEvaluator)  # context precision needs no LLM client


def test_context_precision_matches_any_source_doc():
    retrieved = [
        {"source_doc_id": "crag_doc_other", "source": "crag_doc_other"},
        {"source_doc_id": "crag_doc_aaa", "source_doc_ids": ["crag_doc_aaa", "crag_doc_gold"], "source": "crag_doc_aaa"},
    ]
    assert _evaluator().evaluate_context_precision(["crag_doc_gold.md"], retrieved) == 0.5


def test_context_precision_uses_exact_document_ids():
    retrieved = [{"source_doc_id": "crag_doc_gold_extra", "source": "crag_doc_gold_extra"}]
    assert _evaluator().evaluate_context_precision("crag_doc_gold", retrieved) == 0.0
    chunk = [{"source_doc_id": "crag_doc_gold", "id": "crag_doc_gold_chunk_3"}]
    assert _evaluator().evaluate_context_precision("crag_doc_gold", chunk) == 1.0
