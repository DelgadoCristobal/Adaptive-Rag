from src.vector.vector_store import VectorStoreManager


def test_vector_store_chunking_and_adding(tmp_path):
    chroma_dir = tmp_path / "chroma_db"
    store = VectorStoreManager(db_path=chroma_dir, collection_name="test_collection")

    text = "GraphRAG is a novel approach to combining knowledge graphs with vector embeddings for enhanced enterprise search."
    chunks = store.chunk_text(text, chunk_size=10, overlap=2)
    assert len(chunks) >= 1

    store.add_documents(doc_id="doc1", content=text, metadata={"source": "test.md"})
    results = store.similarity_search("knowledge graphs", top_k=1)
    
    assert isinstance(results, list)
