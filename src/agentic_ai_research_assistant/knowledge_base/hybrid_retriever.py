from typing import List, Dict, Any
from agentic_ai_research_assistant.config import settings
from agentic_ai_research_assistant.knowledge_base.vector_store import global_vector_store
from agentic_ai_research_assistant.knowledge_base.bm25 import global_bm25_manager

def rrf_fusion(ranked_lists: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF, Cormack et al., 2009).
    score = sum(1 / (k + rank))
    Combines disparate score distributions (BM25 vs Cosine) without requiring normalization.
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            cid = chunk["chunk_id"]
            chunk_map[cid] = chunk
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [chunk_map[cid] for cid in sorted_ids]


def hybrid_search(
    query: str,
    user_id: int,
    k_final: int = 5,
    min_sim: float = 0.35,
    candidate_k: int = 20
) -> Dict[str, Any]:
    """
    Executes hybrid retrieval:
    1. Vector semantic search (top 20) with user_id metadata filtering
    2. BM25 keyword search (top 20) on user index
    3. RRF fusion to merge rankings
    4. Anti-hallucination cosine score cutoff test on top candidate
    """
    # 1. Vector Search
    vector_results = global_vector_store.similarity_search_with_score(
        query=query,
        user_id=user_id,
        k=candidate_k
    )
    vector_chunks = [ch for ch, _ in vector_results]
    best_cosine_sim = max((score for _, score in vector_results), default=0.0)

    # 2. BM25 Search
    all_user_chunks = [ch for ch in global_vector_store.chunks.values() if ch["user_id"] == user_id]
    bm25_chunks = global_bm25_manager.search(
        user_id=user_id,
        query=query,
        all_user_chunks=all_user_chunks,
        k=candidate_k
    )

    if not vector_chunks and not bm25_chunks:
        return {
            "status": "empty",
            "message": "not found in uploaded documents",
            "data": [],
            "best_sim": 0.0
        }

    # 3. Fuse with RRF
    fused_chunks = rrf_fusion([vector_chunks, bm25_chunks], k=60)[:k_final]

    # 4. Anti-hallucination score cutoff
    # If the semantic similarity is below cutoff AND query didn't match strong BM25 keywords, trigger refusal
    has_strong_bm25 = bool(bm25_chunks and any(query.lower() in ch["text"].lower() for ch in bm25_chunks[:1]))
    if best_cosine_sim < min_sim and not has_strong_bm25:
        return {
            "status": "empty",
            "message": "not found in uploaded documents",
            "data": [],
            "best_sim": best_cosine_sim
        }

    # Formatted payload
    formatted_data = []
    for idx, ch in enumerate(fused_chunks, start=1):
        formatted_data.append({
            "citation_id": f"[S{idx}]",
            "filename": ch["filename"],
            "page": ch["page"],
            "chunk_id": ch["chunk_id"],
            "text": ch["text"]
        })

    return {
        "status": "ok",
        "message": f"Successfully retrieved {len(formatted_data)} chunks",
        "data": formatted_data,
        "best_sim": best_cosine_sim
    }
