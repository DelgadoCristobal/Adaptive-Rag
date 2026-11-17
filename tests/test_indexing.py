import json
import logging

import pytest

import src.retrieval.graph_retriever as graph_retriever
from src.config import settings
from src.graph.deduplication import EntityDeduplicator
from src.graph.extractor import EntityRelationExtractor
from src.graph.indexing import IndexReport, index_document, source_doc_name
from src.graph.kuzu_store import KuzuGraphStore
from src.graph.schema import concept_note_path, parse_note_sources
from src.retrieval.community_retriever import CommunityRetriever
from src.retrieval.graph_retriever import GraphRetriever

PAYLOAD = json.dumps({
    "entities": [
        {"id": "Alpha", "name": "Alpha", "type": "ORG", "description": "A company"},
        {"id": "Beta", "name": "Beta", "type": "PRODUCT", "description": "An engine"},
    ],
    "relationships": [{"source": "Alpha", "target": "Beta", "relation_type": "BUILDS"}],
})


class StubClient:
    num_ctx = 2048

    def __init__(self, response=PAYLOAD):
        self.response = response

    def complete(self, prompt, system_prompt=None, model=None, temperature=0.1, max_tokens=None):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeVectorStore:
    def __init__(self, fail=False):
        self.fail = fail
        self.added = []

    def add_documents(self, doc_id, content, metadata=None):
        if self.fail:
            raise RuntimeError("vector store offline")
        self.added.append((doc_id, content, metadata))


def _extractor(response=PAYLOAD):
    extractor = EntityRelationExtractor.__new__(EntityRelationExtractor)
    extractor.model = "ollama/test-model"
    extractor.llm_client = StubClient(response)
    return extractor


@pytest.fixture(autouse=True)
def fresh_doc_cache():
    graph_retriever.reset_entity_doc_cache()
    yield
    graph_retriever.reset_entity_doc_cache()


@pytest.fixture
def graph_store(tmp_path):
    db_path = tmp_path / "kuzu_db"
    store = KuzuGraphStore(db_path=db_path)
    yield store
    # Each Kuzu database reserves a huge virtual address range and the store caches instances
    # forever; release it so later tests can still open their own databases.
    KuzuGraphStore._instances.pop(str(db_path), None)
    for handle in (store.conn, store.db):
        if handle is not None:
            handle.close()


def _index(doc_id, vault, graph_store, *, vector_store=None, extractor=None, **kwargs):
    return index_document(
        doc_id, "Alpha builds Beta.", metadata={"source": doc_id},
        vector_store=vector_store or FakeVectorStore(), extractor=extractor or _extractor(),
        deduplicator=EntityDeduplicator(), graph_store=graph_store, vault_dir=vault, **kwargs,
    )


def test_source_doc_name_normalizes_md_suffix():
    assert source_doc_name("report") == "report.md"
    assert source_doc_name("report.md") == "report.md"
    assert source_doc_name("report.MD") == "report.md"


def test_index_document_reports_counts_and_writes_everything(tmp_path, graph_store):
    vault, vectors = tmp_path / "vault", FakeVectorStore()

    report = _index("doc1", vault, graph_store, vector_store=vectors)

    assert report == IndexReport(doc_id="doc1", n_entities=2, n_relations=1, chunks_total=1, chunks_failed=0, error=None)
    assert vectors.added == [("doc1", "Alpha builds Beta.", {"source": "doc1"})]
    assert graph_store.get_stats() == {"node_count": 2, "rel_count": 1}
    assert not (vault / "doc1.md").exists()  # vault copy is opt-in


@pytest.mark.parametrize("doc_id", ["report", "report.md"])
def test_notes_use_normalized_source_doc(tmp_path, graph_store, doc_id):
    vault = tmp_path / "vault"
    _index(doc_id, vault, graph_store)
    note = concept_note_path(vault, "Alpha").read_text(encoding="utf-8")
    assert parse_note_sources(note) == ["report.md"]


def test_write_vault_copy_toggles_the_copy(tmp_path, graph_store):
    vault = tmp_path / "vault"
    _index("doc1", vault, graph_store, write_vault_copy=True)
    assert (vault / "doc1.md").read_text(encoding="utf-8") == "Alpha builds Beta."


def test_step_failure_is_reported_not_raised(tmp_path, graph_store, caplog):
    with caplog.at_level(logging.WARNING, logger="src.graph.indexing"):
        report = _index("doc1", tmp_path / "vault", graph_store, vector_store=FakeVectorStore(fail=True))

    assert report.error and "vector store offline" in report.error
    assert any("doc1" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


def test_failed_chunks_are_reported(tmp_path, graph_store):
    report = _index("doc1", tmp_path / "vault", graph_store, extractor=_extractor(RuntimeError("down")))
    assert (report.chunks_total, report.chunks_failed, report.n_entities, report.error) == (1, 1, 0, None)
    assert graph_store.get_stats() == {"node_count": 0, "rel_count": 0}


def test_community_label_uses_frontmatter_id(tmp_path, graph_store):
    vault = tmp_path / "vault"
    _index("doc1", vault, graph_store)
    notes = CommunityRetriever(concepts_dir=vault / "concepts", graph_store=graph_store)._load_concept_notes()
    labels = sorted(n["text"].split("]", 1)[0] for n in notes)
    assert labels == ["Concept Summary [Alpha", "Concept Summary [Beta"]


def test_graph_path_sources_union_both_endpoints(tmp_path, graph_store, monkeypatch):
    vault = tmp_path / "vault"
    monkeypatch.setattr(settings, "data_vault_dir", vault)
    extractor = _extractor()
    extractor.create_atomic_concept_notes(
        EntityDeduplicator().deduplicate(extractor.extract_from_text("x", "a.md")), "a.md", vault_dir=vault
    )
    graph_store.add_extraction_result(extractor.extract_from_text("x", "a.md"))
    # Beta also appears in a second document; Alpha does not
    extractor.llm_client = StubClient(json.dumps({"entities": [{"id": "Beta", "name": "Beta"}]}))
    extractor.create_atomic_concept_notes(extractor.extract_from_text("x", "b.md"), "b.md", vault_dir=vault)

    candidates = GraphRetriever(graph_store=graph_store).retrieve("Alpha", hops=1)

    paths = [c for c in candidates if c["type"] == "graph_path" and c["id"] == "Alpha->Beta"]
    assert paths, candidates
    assert paths[0]["source_doc_ids"] == ["a", "b"]
    assert paths[0]["source_doc_id"] == "a"
