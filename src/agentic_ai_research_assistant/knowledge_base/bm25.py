import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

class UserBM25Index:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.chunks: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase, alphanumeric + rupee / dollar
        return re.findall(r"\b[a-zA-Z0-9_\-\$₹]+\b", text.lower())

    def update_index(self, user_chunks: List[Dict[str, Any]]):
        self.chunks = list(user_chunks)
        self.corpus_tokens = [self._tokenize(ch["text"]) for ch in self.chunks]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.chunks:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        scored_pairs = [(self.chunks[i], float(scores[i])) for i in range(len(self.chunks))]
        # Filter zero/negative scores and sort descending
        scored_pairs = [pair for pair in scored_pairs if pair[1] > 0]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return [pair[0] for pair in scored_pairs[:k]]


class BM25Manager:
    """Manages per-user BM25 indexes with in-memory caching and invalidation."""
    def __init__(self):
        self.indexes: Dict[int, UserBM25Index] = {}

    def get_or_build_index(self, user_id: int, all_user_chunks: List[Dict[str, Any]]) -> UserBM25Index:
        if user_id not in self.indexes:
            idx = UserBM25Index(user_id)
            idx.update_index(all_user_chunks)
            self.indexes[user_id] = idx
        return self.indexes[user_id]

    def invalidate(self, user_id: int):
        self.indexes.pop(user_id, None)

    def search(self, user_id: int, query: str, all_user_chunks: List[Dict[str, Any]], k: int = 20) -> List[Dict[str, Any]]:
        idx = self.get_or_build_index(user_id, all_user_chunks)
        return idx.search(query, k=k)


global_bm25_manager = BM25Manager()
