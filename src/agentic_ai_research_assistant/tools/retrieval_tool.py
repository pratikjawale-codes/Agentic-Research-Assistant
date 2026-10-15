from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agentic_ai_research_assistant.knowledge_base.hybrid_retriever import hybrid_search
from agentic_ai_research_assistant.tools.validator import validated_tool_output
from agentic_ai_research_assistant.config import settings

class RetrieveArgs(BaseModel):
    query: str = Field(description="Search term or topic to look up in uploaded documents.")

@tool("retrieve_documents", args_schema=RetrieveArgs)
def retrieve_documents(query: str, config: RunnableConfig) -> dict:
    """
    Search the current user's uploaded PDF/DOCX/TXT internal documents.
    Uses hybrid BM25 + dense vector retrieval with an anti-hallucination score cutoff.
    """
    # Server-injected tenant user_id, hidden from LLM schema
    configurable = config.get("configurable", {})
    user_id_raw = configurable.get("user_id", 1)
    user_id = int(user_id_raw)

    result = hybrid_search(
        query=query,
        user_id=user_id,
        k_final=settings.RAG_TOP_K,
        min_sim=settings.RELEVANCE_SCORE_CUTOFF,
        candidate_k=settings.RAG_CANDIDATE_K
    )

    if result["status"] == "empty":
        return {
            "status": "empty",
            "data": [],
            "sources": [],
            "message": "not found in uploaded documents"
        }

    sources = [f"{item['citation_id']} {item['filename']} (p.{item['page']})" for item in result["data"]]
    return {
        "status": "ok",
        "data": result["data"],
        "sources": sources,
        "message": f"Found {len(result['data'])} relevant sections"
    }
