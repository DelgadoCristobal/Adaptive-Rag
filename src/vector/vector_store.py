from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from src.config import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Dense Vector Manager using ChromaDB for document chunks and concept summaries."""

    def __init__(self, db_path: Path = None, collection_name: str = "corporate_brain"):
        self.db_path = Path(db_path) if db_path else settings.chroma_db_path
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.in_memory_docs = []  # Fallback store
        self._init_chroma()

    def _init_chroma(self):
        """Initialize ChromaDB Persistent Client."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            self.client = chromadb.PersistentClient(path=str(self.db_path))
            
            try:
                emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            except Exception:
                emb_fn = None

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=emb_fn
            )
            logger.info(f"Initialized ChromaDB collection '{self.collection_name}' at {self.db_path}")
        except Exception as e:
            logger.info(f"ChromaDB not available ({e}). Using in-memory vector storage.")

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split document text into overlapping chunks."""
        words = text.split()
        if not words:
            return []
        
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += (chunk_size - overlap)
        return chunks

    def add_documents(self, doc_id: str, content: str, metadata: Dict[str, Any] = None):
        """Chunk text content and add embeddings to ChromaDB or in-memory fallback."""
        chunks = self.chunk_text(content)
        if not chunks:
            return

        if not self.collection:
            for idx, chunk in enumerate(chunks):
                self.in_memory_docs.append({
                    "id": f"{doc_id}_chunk_{idx}",
                    "content": chunk,
                    "metadata": {"doc_id": doc_id, "chunk_index": idx, **(metadata or {})}
                })
            logger.info(f"Added {len(chunks)} chunks for {doc_id} to in-memory vector store.")
            return

        ids = [f"{doc_id}_chunk_{idx}" for idx in range(len(chunks))]
        metadatas = [
            {
                "doc_id": doc_id,
                "chunk_index": idx,
                **(metadata or {})
            }
            for idx in range(len(chunks))
        ]

        try:
            self.collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(chunks)} chunks for document {doc_id} to ChromaDB")
        except Exception as e:
            logger.error(f"Failed to add document {doc_id} to ChromaDB: {e}")

    def similarity_search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Perform dense vector k-NN search against text chunks."""
        top_k = top_k or settings.vector_top_k
        if not self.collection:
            # Fallback search over in-memory docs
            results = []
            q_words = set(query.lower().split())
            for item in self.in_memory_docs:
                text = item["content"].lower()
                t_words = set(text.split())
                score = len(q_words.intersection(t_words))
                results.append({
                    "id": item["id"],
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "distance": 1.0 - (score / (len(q_words) + 1))
                })
            results.sort(key=lambda x: x["distance"])
            return results[:top_k]

        results = []
        try:
            res = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            if res and "documents" in res and res["documents"]:
                docs = res["documents"][0]
                metas = res["metadatas"][0] if "metadatas" in res else [{}] * len(docs)
                ids = res["ids"][0] if "ids" in res else [""] * len(docs)
                distances = res["distances"][0] if "distances" in res and res["distances"] else [0.0] * len(docs)

                for doc, meta, doc_id, dist in zip(docs, metas, ids, distances):
                    results.append({
                        "id": doc_id,
                        "content": doc,
                        "metadata": meta,
                        "distance": dist
                    })
        except Exception as e:
            logger.warning(f"ChromaDB similarity search failed: {e}")

        return results
