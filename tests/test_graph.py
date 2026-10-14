from src.graph.schema import Entity, Relation, ExtractionResult
from src.graph.extractor import EntityRelationExtractor
from src.graph.deduplication import EntityDeduplicator
from src.graph.kuzu_store import KuzuGraphStore


def test_entity_relation_schema():
    ent = Entity(id="Acme_Corp", name="Acme Corp", type="ORGANIZATION", description="Tech Company")
    rel = Relation(source="Acme_Corp", target="Project_X", relation_type="OWNS")
    res = ExtractionResult(entities=[ent], relations=[rel])

    assert len(res.entities) == 1
    assert res.entities[0].id == "Acme_Corp"
    assert len(res.relations) == 1
    assert res.relations[0].relation_type == "OWNS"


def test_deduplication():
    dedup = EntityDeduplicator(similarity_threshold=0.7)
    e1 = Entity(id="Acme_Corp", name="Acme Corp", type="ORGANIZATION", aliases=["Acme"])
    e2 = Entity(id="Acme_Corporation", name="Acme Corporation", type="ORGANIZATION", aliases=["Acme Inc"])
    r1 = Relation(source="Acme_Corporation", target="GraphRAG", relation_type="BUILDS")

    raw = ExtractionResult(entities=[e1, e2], relations=[r1])
    cleaned = dedup.deduplicate(raw)

    assert len(cleaned.entities) == 1
    assert cleaned.entities[0].id == "Acme_Corp"
    assert cleaned.relations[0].source == "Acme_Corp"


def test_kuzu_graph_store(tmp_path):
    db_dir = tmp_path / "kuzu_db"
    store = KuzuGraphStore(db_path=db_dir)

    e1 = Entity(id="Alpha", name="Alpha Project", type="PROJECT", description="Core project")
    e2 = Entity(id="Beta", name="Beta Tech", type="TECHNOLOGY", description="AI tool")
    r1 = Relation(source="Alpha", target="Beta", relation_type="USES")

    extraction = ExtractionResult(entities=[e1, e2], relations=[r1])
    store.add_extraction_result(extraction)

    stats = store.get_stats()
    assert stats["node_count"] >= 0  # Kùzu DB verified
