import pytest
import asyncio
from agentic_ai_research_assistant.knowledge_base.vector_store import global_vector_store
from agentic_ai_research_assistant.knowledge_base.bm25 import global_bm25_manager
from agentic_ai_research_assistant.knowledge_base.chunker import create_chunks
from agentic_ai_research_assistant.knowledge_base.hybrid_retriever import hybrid_search
from agentic_ai_research_assistant.tools.calculator_tool import cagr
from agentic_ai_research_assistant.tools.db_tool import get_market_stat
from agentic_ai_research_assistant.tools.search_tool import web_search
from agentic_ai_research_assistant.agents.graph import research_graph
from agentic_ai_research_assistant.database.session import init_db

@pytest.fixture(autouse=True)
def setup_database():
    asyncio.run(init_db())

def test_multi_user_data_isolation():
    """
    Security Test: User B must NEVER access or retrieve User A's private documents.
    """
    user_a_id = 101
    user_b_id = 102

    user_a_doc = [
        (1, "Confidential Strategy Document: Target acquisition budget for Project Titan is $245 million. Expected gross margin efficiency is 34.2%.")
    ]
    chunks = create_chunks(user_a_doc, user_id=user_a_id, doc_id=1, filename="confidential_strategy.pdf")
    global_vector_store.add_chunks(chunks)
    global_bm25_manager.get_or_build_index(user_a_id, chunks)

    # 1. User A queries their own document
    user_a_res = hybrid_search("Project Titan acquisition budget", user_id=user_a_id)
    assert user_a_res["status"] == "ok"
    assert len(user_a_res["data"]) > 0
    assert "$245 million" in user_a_res["data"][0]["text"]

    # 2. User B queries the exact same query - MUST RETURN EMPTY / NOT FOUND
    user_b_res = hybrid_search("Project Titan acquisition budget", user_id=user_b_id)
    assert user_b_res["status"] == "empty"
    assert len(user_b_res["data"]) == 0
    assert user_b_res["message"] == "not found in uploaded documents"


def test_anti_hallucination_score_cutoff():
    """
    Anti-Hallucination Test: Queries with no evidence must trigger 'not found' refusal.
    """
    user_id = 201
    doc_text = [(1, "Quarterly operational summary for North American logistics facilities.")]
    chunks = create_chunks(doc_text, user_id=user_id, doc_id=2, filename="logistics_summary.pdf")
    global_vector_store.add_chunks(chunks)
    global_bm25_manager.get_or_build_index(user_id, chunks)

    # Query for something completely absent from documents
    res = hybrid_search("astrophysical dark matter galactic rotational anomalies", user_id=user_id)
    assert res["status"] == "empty"
    assert "not found" in res["message"]


def test_tools_execution_and_validation():
    """
    Tool Layer Test: Checks calculator CAGR precision, DB query allow-listing, and schema validation.
    """
    # 1. CAGR calculation: from $500M to $1850M over 5 years
    cagr_out = cagr.invoke({"start_value": 500.0, "end_value": 1850.0, "years": 5.0})
    assert cagr_out["status"] == "ok"
    assert cagr_out["data"]["cagr_percent"] == "29.91%"
    assert cagr_out["data"]["inputs"]["start_value"] == 500.0

    # 2. DB Query tool
    db_out = get_market_stat.invoke({"metric": "baseline_market_valuation", "year": 2024})
    assert db_out["status"] == "ok"
    assert db_out["data"]["value"] == 500.0
    assert db_out["data"]["unit"] == "Million USD"

    # 3. Disallowed DB query rejected
    bad_db = get_market_stat.invoke({"metric": "DROP TABLE users;--", "year": 2024})
    assert bad_db["status"] == "error"

    # 4. Web search tool
    search_out = web_search.invoke({"query": "enterprise cloud software adoption growth"})
    assert search_out["status"] == "ok"
    assert len(search_out["data"]) > 0


def test_end_to_end_research_system():
    """
    End-to-End Scenario: Autonomous multi-agent research workflow.
    """
    user_id = 301
    doc_text = [
        (1, "Strategic Pricing Model: Pricing assumptions model an initial target contract value of $150,000 annually."),
        (2, "Competitive overview: Sector expansion shows accelerating digital infrastructure investments.")
    ]
    chunks = create_chunks(doc_text, user_id=user_id, doc_id=10, filename="strategic_pricing_model.pdf")
    global_vector_store.add_chunks(chunks)
    global_bm25_manager.get_or_build_index(user_id, chunks)

    query = "Analyze the commercial market expansion, evaluate against the pricing assumptions in my uploaded PDF, calculate 5-year CAGR, and produce a research report."

    initial_state = {
        "messages": [{"role": "user", "content": query}],
        "intent": {},
        "plan": [],
        "current_step": 0,
        "findings": {},
        "numbers": {},
        "report": "",
        "replans": 0,
        "errors": []
    }
    config = {
        "configurable": {
            "user_id": str(user_id),
            "thread_id": f"{user_id}:chat_test"
        }
    }

    result = research_graph.invoke(initial_state, config=config)

    # Assert supervisor formed plan and intent
    assert result["intent"]["intent"] == "research"
    assert len(result["plan"]) >= 3

    # Assert analyst computed quantitative metrics
    assert len(result["numbers"]) > 0

    # Assert report was written with citations
    report = result["report"]
    assert "Research & Intelligence Report" in report
    assert "CAGR" in report
    assert "[S1]" in report
