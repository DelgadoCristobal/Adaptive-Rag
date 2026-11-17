import logging

import pytest

from src.graph.kuzu_store import KuzuGraphStore
from src.graph.schema import Entity, Relation, ExtractionResult


@pytest.fixture
def store(tmp_path):
    return KuzuGraphStore(db_path=tmp_path / "kuzu_db")


def _edges(store):
    res = store.conn.execute(
        "MATCH (a:EntityNode)-[r:RELATION]->(b:EntityNode) "
        "RETURN a.id, b.id, r.relation_type, r.description, r.weight ORDER BY r.relation_type"
    )
    rows = []
    while res.has_next():
        rows.append(res.get_next())
    return rows


def _node(store, node_id):
    res = store.conn.execute(
        "MATCH (e:EntityNode) WHERE e.id = $id RETURN e.name, e.type, e.description", {"id": node_id}
    )
    return res.get_next() if res.has_next() else None


def _pair(rel):
    return ExtractionResult(
        entities=[Entity(id="A", name="A"), Entity(id="B", name="B")],
        relations=[rel],
    )


def test_relation_description_with_special_characters_round_trips(store):
    desc = "it's a \"quoted\" back\\slash; DROP\nsecond line"
    store.add_extraction_result(_pair(Relation(source="A", target="B", relation_type="USES", description=desc)))

    edges = _edges(store)
    assert len(edges) == 1
    assert edges[0][3] == desc


def test_same_relation_inserted_twice_creates_one_edge(store):
    rel = Relation(source="A", target="B", relation_type="USES", description="first")
    store.add_extraction_result(_pair(rel))
    store.add_extraction_result(_pair(Relation(source="A", target="B", relation_type="USES", description="second")))

    edges = _edges(store)
    assert len(edges) == 1
    assert edges[0][3] == "first"
    assert store.get_stats() == {"node_count": 2, "rel_count": 1}


def test_different_relation_types_create_separate_edges(store):
    store.add_extraction_result(_pair(Relation(source="A", target="B", relation_type="USES")))
    store.add_extraction_result(_pair(Relation(source="A", target="B", relation_type="owned by")))

    assert [e[2] for e in _edges(store)] == ["OWNED_BY", "USES"]


def test_non_finite_weight_defaults_to_one(store):
    rel = Relation(source="A", target="B", relation_type="USES")
    rel.weight = float("nan")
    store.add_extraction_result(_pair(rel))

    assert _edges(store)[0][4] == 1.0


def test_node_upsert_keeps_richer_description(store):
    rich = Entity(id="Alpha", name="Alpha Project", type="PROJECT", description="A long, detailed description")
    store.add_extraction_result(ExtractionResult(entities=[rich]))

    store.add_extraction_result(ExtractionResult(entities=[Entity(id="Alpha", name="Alpha", type="PROJECT", description="Short")]))
    assert _node(store, "Alpha")[2] == "A long, detailed description"

    store.add_extraction_result(ExtractionResult(entities=[Entity(id="Alpha", name="Alpha", type="PROJECT", description="")]))
    assert _node(store, "Alpha")[2] == "A long, detailed description"

    longer = "A long, detailed description with even more context"
    store.add_extraction_result(ExtractionResult(entities=[Entity(id="Alpha", name="Alpha", type="PROJECT", description=longer)]))
    assert _node(store, "Alpha")[2] == longer


def test_relation_endpoint_placeholder_does_not_erase_existing_node(store):
    store.add_extraction_result(ExtractionResult(entities=[Entity(id="A", name="Alpha", type="PROJECT", description="desc")]))
    store.add_extraction_result(ExtractionResult(relations=[Relation(source="A", target="Z", relation_type="USES")]))

    assert _node(store, "A") == ["Alpha", "PROJECT", "desc"]
    assert store.get_stats() == {"node_count": 2, "rel_count": 1}


def test_failed_edge_write_logs_warning(store, caplog, monkeypatch):
    store.add_extraction_result(ExtractionResult(entities=[Entity(id="A", name="A"), Entity(id="B", name="B")]))
    real_execute = store.conn.execute

    def failing_execute(query, *args, **kwargs):
        if "RELATION {" in query:
            raise RuntimeError("boom")
        return real_execute(query, *args, **kwargs)

    monkeypatch.setattr(store.conn, "execute", failing_execute)
    with caplog.at_level(logging.WARNING, logger="src.graph.kuzu_store"):
        store.add_extraction_result(ExtractionResult(relations=[Relation(source="A", target="B", relation_type="USES")]))

    assert any("A" in r.getMessage() and "B" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.parametrize("hops", [0, -3, "2", 99])
def test_k_hop_clamps_hops(store, hops):
    store.add_extraction_result(_pair(Relation(source="A", target="B", relation_type="USES")))

    paths = store.query_k_hop_neighborhood("A", hops=hops)
    assert [p["target"] for p in paths] == ["B"]
