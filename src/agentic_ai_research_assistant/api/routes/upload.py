import hashlib
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from agentic_ai_research_assistant.database.session import get_db
from agentic_ai_research_assistant.database.models import User, Document
from agentic_ai_research_assistant.auth.security import get_current_user
from agentic_ai_research_assistant.knowledge_base.parser import parse_document, DocumentParseError
from agentic_ai_research_assistant.knowledge_base.chunker import create_chunks
from agentic_ai_research_assistant.knowledge_base.vector_store import global_vector_store
from agentic_ai_research_assistant.knowledge_base.bm25 import global_bm25_manager
from agentic_ai_research_assistant.knowledge_base.hybrid_retriever import hybrid_search
from agentic_ai_research_assistant.config import settings

router = APIRouter(tags=["Knowledge Base & Retrieval"])

class SearchRequest(BaseModel):
    query: str = Field(description="Search text or query")
    k: int = Field(default=5, ge=1, le=20)

class SearchHit(BaseModel):
    citation_id: str
    filename: str
    page: int
    text: str

class SearchResponse(BaseModel):
    status: str
    message: str
    data: List[SearchHit]
    best_sim: float


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload and index private documents (PDF, DOCX, TXT).
    Extracts text, creates chunks with deterministic IDs, and updates hybrid index with user_id isolation.
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Max 20MB guard
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds maximum allowed size of 20MB.")

    file_hash = hashlib.sha256(content).hexdigest()

    # Check if doc exists for this user
    existing = await db.execute(
        select(Document).where(Document.user_id == current_user.id, Document.filename == file.filename)
    )
    doc_record = existing.scalars().first()

    if doc_record:
        if doc_record.file_hash == file_hash:
            return {"status": "skipped", "message": "Document already ingested with identical content hash."}
        # Re-upload with modified content: clear old chunks
        global_vector_store.delete_by_doc_id(current_user.id, doc_record.id)
        doc_record.file_hash = file_hash
        await db.commit()
    else:
        doc_record = Document(user_id=current_user.id, filename=file.filename, file_hash=file_hash)
        db.add(doc_record)
        await db.commit()
        await db.refresh(doc_record)

    try:
        pages = parse_document(content, file.filename)
    except DocumentParseError as pe:
        raise HTTPException(status_code=400, detail=str(pe))

    chunks = create_chunks(
        pages_data=pages,
        user_id=current_user.id,
        doc_id=doc_record.id,
        filename=file.filename,
        chunk_size=settings.RAG_CHUNK_SIZE,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP
    )

    # Upsert to vector store
    global_vector_store.add_chunks(chunks)

    # Invalidate and update BM25 index for this user
    all_user_chunks = [ch for ch in global_vector_store.chunks.values() if ch["user_id"] == current_user.id]
    global_bm25_manager.invalidate(current_user.id)
    global_bm25_manager.get_or_build_index(current_user.id, all_user_chunks)

    return {
        "status": "ready",
        "doc_id": doc_record.id,
        "filename": file.filename,
        "chunks_indexed": len(chunks),
        "message": f"Successfully parsed {len(pages)} pages and indexed {len(chunks)} chunks."
    }


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    req: SearchRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Search user's uploaded documents using hybrid BM25 + dense vector search and relevance cutoff.
    """
    result = hybrid_search(
        query=req.query,
        user_id=current_user.id,
        k_final=req.k,
        min_sim=settings.RELEVANCE_SCORE_CUTOFF,
        candidate_k=settings.RAG_CANDIDATE_K
    )
    return SearchResponse(
        status=result["status"],
        message=result["message"],
        data=[SearchHit(**item) for item in result["data"]],
        best_sim=result.get("best_sim", 0.0)
    )
