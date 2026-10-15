from typing import Literal, Dict, Any
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig

from agentic_ai_research_assistant.agents.state import AgentState
from agentic_ai_research_assistant.tools.retrieval_tool import retrieve_documents
from agentic_ai_research_assistant.tools.search_tool import web_search

def research_node(state: AgentState, config: RunnableConfig) -> Command[Literal["supervisor"]]:
    """
    Research Worker Agent:
    - Dedicated to evidence collection.
    - Bound strictly to tools: retrieve_documents and web_search.
    - Executes dynamically according to current step instruction.
    """
    plan = state.get("plan", [])
    step_idx = state.get("current_step", 0)
    current_step_info = plan[step_idx] if step_idx < len(plan) else {}
    desc = current_step_info.get("description", "").lower()

    # Get user original message
    user_query = ""
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break

    step_findings: Dict[str, Any] = {}

    # Check if retrieval is needed
    if any(k in desc for k in ["retrieve", "internal", "document", "assumption", "upload"]):
        # Extract query topic
        retrieval_query = user_query
        if "pricing" in user_query.lower():
            retrieval_query = "pricing assumptions"
        elif ":" in desc:
            retrieval_query = desc.split(":")[-1].strip()
        ret_result = retrieve_documents.invoke({"query": retrieval_query}, config=config)
        step_findings["document_retrieval"] = ret_result

    # Check if web search is needed
    if any(k in desc for k in ["search", "web", "market", "external", "trend", "gather"]):
        search_query = user_query if len(user_query) < 100 else user_query[:100]
        if ":" in desc:
            search_query = desc.split(":")[-1].strip()
        web_result = web_search.invoke({"query": search_query})
        step_findings["web_search"] = web_result

    if not step_findings:
        web_result = web_search.invoke({"query": user_query[:100] if user_query else "industry trends"})
        step_findings["web_search"] = web_result

    new_findings = {**state.get("findings", {}), f"step_{step_idx}": step_findings}
    summary_msg = f"Research Agent completed step {step_idx} ({len(step_findings)} evidence streams collected)."

    return Command(
        goto="supervisor",
        update={
            "messages": state.get("messages", []) + [{"role": "assistant", "content": summary_msg}],
            "findings": new_findings,
            "current_step": step_idx + 1
        }
    )
