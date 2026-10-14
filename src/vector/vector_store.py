from pathlib import Path
from typing import List, Dict, Any
import logging
from src.config import settings

import re

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Dense Vector Manager using ChromaDB with Semantic Chunking and Parent-Child Context Retrieval."""

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

    def chunk_text_semantic(self, text: str, max_chunk_chars: int = 500) -> List[Dict[str, str]]:
        """Split text semantically respecting markdown headers and paragraphs, returning child/parent pairs."""
        sections = re.split(r'(?m)(?=^#{1,4}\s+)', text)
        results = []
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue

            header_match = re.match(r'^(#{1,4}\s+[^\n]+)', sec)
            header = header_match.group(1).strip() if header_match else ""
            body = sec[len(header):].strip() if header else sec

            paragraphs = [p.strip() for p in re.split(r'\n\s*\n', body) if p.strip()]
            if not paragraphs:
                results.append({"child_text": sec[:max_chunk_chars], "parent_text": sec, "section": header})
                continue

            for p in paragraphs:
                if len(p) <= max_chunk_chars:
                    child = f"{header}\n{p}".strip() if header else p
                    results.append({"child_text": child, "parent_text": sec, "section": header})
                else:
                    sentences = re.split(r'(?<=[.!?])\s+', p)
                    curr_child = []
                    curr_len = 0
                    for s in sentences:
                        curr_child.append(s)
                        curr_len += len(s)
                        if curr_len >= max_chunk_chars:
                            c_text = " ".join(curr_child)
                            child = f"{header}\n{c_text}".strip() if header else c_text
                            results.append({"child_text": child, "parent_text": sec, "section": header})
                            curr_child = []
                            curr_len = 0
                    if curr_child:
                        c_text = " ".join(curr_child)
                        child = f"{header}\n{c_text}".strip() if header else c_text
                        results.append({"child_text": child, "parent_text": sec, "section": header})
        return results if results else [{"child_text": text[:max_chunk_chars], "parent_text": text, "section": ""}]

    def add_documents(self, doc_id: str, content: str, metadata: Dict[str, Any] = None):
        """Chunk text semantically with Parent-Child metadata and add embeddings to ChromaDB."""
        pairs = self.chunk_text_semantic(content)
        if not pairs:
            return

        documents = [p["child_text"] for p in pairs]
        ids = [f"{doc_id}_chunk_{idx}" for idx in range(len(pairs))]
        metadatas = [
            {
                "doc_id": doc_id,
                "source_doc_id": doc_id,
                "chunk_index": idx,
                "section": p.get("section", ""),
                "parent_text": p["parent_text"][:2000],
                **(metadata or {})
            }
            for idx, p in enumerate(pairs)
        ]

        if not self.collection:
            for c_id, doc_text, meta in zip(ids, documents, metadatas):
                self.in_memory_docs.append({
                    "id": c_id,
                    "content": doc_text,
                    "metadata": meta
                })
            logger.info(f"Added {len(documents)} semantic chunks for {doc_id} to in-memory vector store.")
            return

        try:
            if hasattr(self.collection, "upsert"):
                self.collection.upsert(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            else:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            logger.info(f"Upserted {len(documents)} semantic chunks for document {doc_id} to ChromaDB")
        except Exception as e:
            logger.error(f"Failed to upsert document {doc_id} to ChromaDB: {e}")

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
