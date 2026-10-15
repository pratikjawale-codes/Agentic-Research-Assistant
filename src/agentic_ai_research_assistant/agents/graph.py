from langgraph.graph import StateGraph, START, END
from agentic_ai_research_assistant.agents.state import AgentState
from agentic_ai_research_assistant.agents.supervisor import supervisor_node
from agentic_ai_research_assistant.agents.research_agent import research_node
from agentic_ai_research_assistant.agents.analyst_agent import analyst_node
from agentic_ai_research_assistant.agents.report_agent import report_node

def build_research_graph():
    """
    Compiles the Supervisor-Worker multi-agent LangGraph workflow.
    - START routes into supervisor
    - Supervisor coordinates Research, Analyst, and Report workers
    - Workers transition back to supervisor with last_message and structured state
    """
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("research", research_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("report", report_node)

    builder.add_edge(START, "supervisor")

    return builder.compile()

# Global compiled research graph
research_graph = build_research_graph()
