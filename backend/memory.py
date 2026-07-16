"""
memory.py — Long-term semantic memory for the task-planning agent.

Uses ChromaDB (persistent, local) with a local sentence-transformers
embedding model, so remembering/recalling facts costs no API calls —
only the agent's reasoning calls the LLM.
"""
import time
import uuid

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "chroma_memory"
COLLECTION_NAME = "task_agent_memory"


class LongTermMemory:
    def __init__(self, path: str = CHROMA_PATH):
        self.client = chromadb.PersistentClient(path=path)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embed_fn,
        )

    def remember(self, text: str, kind: str = "note", metadata: dict | None = None) -> str:
        """Store a fact/goal/preference. kind: 'goal' | 'preference' | 'note' | 'reflection'."""
        mem_id = str(uuid.uuid4())
        meta = {"kind": kind, "timestamp": time.time()}
        if metadata:
            meta.update(metadata)
        self.collection.add(documents=[text], metadatas=[meta], ids=[mem_id])
        return mem_id

    def recall(self, query: str, k: int = 4) -> list[dict]:
        """Semantic search over everything remembered so far."""
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(query_texts=[query], n_results=min(k, count))
        memories = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            memories.append({"text": doc, "metadata": meta, "distance": dist})
        return memories
