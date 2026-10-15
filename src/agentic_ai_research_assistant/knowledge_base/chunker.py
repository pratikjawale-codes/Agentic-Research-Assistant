import hashlib
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(
    pages_data: List[tuple[int, str]],
    user_id: int,
    doc_id: int,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Splits document pages into chunks with deterministic IDs, content hashes, and rich metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    chunk_counter = 0

    for page_num, text in pages_data:
        if not text.strip():
            continue
        page_chunks = splitter.split_text(text)
        for sub_text in page_chunks:
            if not sub_text.strip():
                continue
            content_hash = hashlib.sha256(sub_text.encode("utf-8")).hexdigest()
            chunk_id = f"{doc_id}:{chunk_counter}"
            chunks.append({
                "chunk_id": chunk_id,
                "user_id": user_id,
                "doc_id": doc_id,
                "filename": filename,
                "page": page_num,
                "chunk_index": chunk_counter,
                "text": sub_text,
                "content_hash": content_hash,
                "citation_tag": f"[S{chunk_counter + 1}]"
            })
            chunk_counter += 1

    return chunks
