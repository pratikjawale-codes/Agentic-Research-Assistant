import math
import re
from typing import List, Dict, Any, Optional
import numpy as np

from agentic_ai_research_assistant.config import settings

class LocalSemanticVectorizer:
    """
    High-efficiency deterministic semantic vectorizer.
    Enables instant vector indexing without requiring external API keys.
    When OpenAI or Gemini keys are added, hosted embeddings can be seamlessly plugged in.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_\-\$₹]+\b", text.lower())

    def embed_text(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.dimension, dtype=np.float32)

        vec = np.zeros(self.dimension, dtype=np.float32)
        for token in tokens:
            # Deterministic character-level hash projection
            h = hash(token)
            idx = abs(h) % self.dimension
            sign = 1.0 if (h >> 1) % 2 == 0 else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [self.embed_text(t) for t in texts]


class VectorStore:
    """
    Vector storage with metadata filtering by user_id and cosine similarity computation.
    """
    def __init__(self):
        self.vectorizer = LocalSemanticVectorizer()
        # Storage: chunk_id -> dict of data
        self.chunks: Dict[str, Dict[str, Any]] = {}
        # Storage: chunk_id -> np.ndarray vector
        self.vectors: Dict[str, np.ndarray] = {}

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        for ch in chunks:
            cid = ch["chunk_id"]
            self.chunks[cid] = ch
            self.vectors[cid] = self.vectorizer.embed_text(ch["text"])

    def delete_by_doc_id(self, user_id: int, doc_id: int):
        to_del = [cid for cid, ch in self.chunks.items() if ch["user_id"] == user_id and ch["doc_id"] == doc_id]
        for cid in to_del:
            self.chunks.pop(cid, None)
            self.vectors.pop(cid, None)

    def similarity_search_with_score(
        self, query: str, user_id: int, k: int = 20
    ) -> List[tuple[Dict[str, Any], float]]:
        """
        Cosine similarity search with strict user_id filtering.
        Returns list of (chunk, cosine_similarity_score).
        """
        query_vec = self.vectorizer.embed_text(query)
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return []

        results: List[tuple[Dict[str, Any], float]] = []

        for cid, ch in self.chunks.items():
            if ch["user_id"] != user_id:
                continue  # Strict multi-tenant isolation
            vec = self.vectors[cid]
            dot = float(np.dot(query_vec, vec))
            # Cosine similarity between normalized vectors is the dot product
            score = max(0.0, min(1.0, dot))
            results.append((ch, score))

        # Sort descending by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


# Global singleton vector store instance
global_vector_store = VectorStore()
