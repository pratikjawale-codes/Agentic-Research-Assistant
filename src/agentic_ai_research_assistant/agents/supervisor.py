from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langgraph.types import Command
from langgraph.graph import END

from agentic_ai_research_assistant.config import settings
from agentic_ai_research_assistant.agents.state import AgentState
from agentic_ai_research_assistant.agents.llm_factory import LLMFactory

class Intent(BaseModel):
    intent: Literal["research", "calculation", "doc_qa", "chitchat"]
    domain: str
    location: Optional[str] = None
    output_format: Literal["report", "answer", "table"] = "report"
    needs_documents: bool = True
    needs_web: bool = True
    needs_math: bool = False

class Step(BaseModel):
    id: int
    description: str
    agent: Literal["research", "analyst", "report"]
    depends_on: List[int] = Field(default_factory=list)

class Plan(BaseModel):
    steps: List[Step]


def parse_intent(user_query: str) -> Intent:
    """Dynamically parses and classifies user query intent across any research domain."""
    llm = LLMFactory.get_chat_model(temperature=0.0, role="supervisor")
    if llm:
        try:
            structured_llm = llm.with_structured_output(Intent)
            return structured_llm.invoke(f"Extract intent for query: {user_query}")
        except Exception:
            pass

    q = user_query.lower()
    if any(k in q for k in ["hi", "hello", "who are you", "what can you do", "help"]):
        return Intent(
            intent="chitchat",
            domain="general",
            output_format="answer",
            needs_documents=False,
            needs_web=False,
            needs_math=False
        )

    # Dynamic domain identification
    domain = "General Industry Research"
    for keyword, name in [
        ("pharma", "Pharmaceutical & Healthcare"),
        ("tech", "Technology & Software"),
        ("energy", "Energy & Infrastructure"),
        ("finance", "Financial Services"),
        ("retail", "Consumer & Retail"),
        ("market", "Market Intelligence"),
        ("ai", "Artificial Intelligence")
    ]:
        if keyword in q:
            domain = name
            break

    needs_docs = any(w in q for w in ["pdf", "doc", "document", "internal", "upload", "assumption", "file"])
    needs_math = any(w in q for w in ["cagr", "calculate", "growth", "percent", "metric", "rate", "compare"])
    has_web = not (q.startswith("only search my document") or q.startswith("from my file only"))

    return Intent(
        intent="research",
        domain=domain,
        location=None,
        output_format="report",
        needs_documents=needs_docs,
        needs_web=has_web,
        needs_math=needs_math
    )


def build_plan(user_query: str, intent: Intent) -> Plan:
    """Generates a dynamic multi-step execution plan tailored to the specific query."""
    llm = LLMFactory.get_chat_model(temperature=0.0, role="supervisor")
    if llm:
        try:
            structured_llm = llm.with_structured_output(Plan)
            prompt = f"Create an ordered execution plan for: '{user_query}'. Intent: {intent.model_dump_json()}."
            return structured_llm.invoke(prompt)
        except Exception:
            pass

    steps = []
    step_id = 0

    if intent.needs_documents:
        steps.append(Step(
            id=step_id,
            description=f"Retrieve internal assumptions and evidence regarding: {user_query[:60]}",
            agent="research",
            depends_on=[]
        ))
        step_id += 1

    if intent.needs_web:
        steps.append(Step(
            id=step_id,
            description=f"Gather external market trends, industry benchmarks, and public evidence regarding: {user_query[:60]}",
            agent="research",
            depends_on=[]
        ))
        step_id += 1

    if intent.needs_math or "cagr" in user_query.lower() or "calculate" in user_query.lower():
        steps.append(Step(
            id=step_id,
            description=f"Perform verified quantitative calculations and statistical comparisons for: {user_query[:60]}",
            agent="analyst",
            depends_on=[0, 1] if step_id >= 2 else []
        ))
        step_id += 1

    steps.append(Step(
        id=step_id,
        description="Synthesize all retrieved evidence, citations, and quantitative analysis into a final report",
        agent="report",
        depends_on=[step_id - 1] if step_id > 0 else []
    ))

    return Plan(steps=steps)


def supervisor_node(state: AgentState) -> Command[Literal["research", "analyst", "report", "__end__"]]:
    """
    Supervisor Agent Node:
    - Enforces step budget and iteration caps
    - Analyzes intent and constructs plan upfront
    - Routes deterministically from plan to worker nodes without unnecessary routing LLM calls
    """
    current_step = state.get("current_step", 0)

    if current_step >= settings.MAX_SUPERVISOR_STEPS:
        return Command(
            goto="report",
            update={"errors": state.get("errors", []) + ["Supervisor maximum step limit reached"]}
        )

    user_query = ""
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break

    if not state.get("intent"):
        intent_obj = parse_intent(user_query)
        if intent_obj.intent == "chitchat":
            return Command(
                goto=END,
                update={
                    "intent": intent_obj.model_dump(),
                    "report": "Hello! I am your Agentic AI Research Assistant. You can upload documents and assign high-level research tasks combining private documents, live web intelligence, and quantitative analysis."
                }
            )
        state["intent"] = intent_obj.model_dump()

    if not state.get("plan"):
        plan_obj = build_plan(user_query, Intent(**state["intent"]))
        return Command(
            goto="supervisor",
            update={
                "intent": state["intent"],
                "plan": [s.model_dump() for s in plan_obj.steps],
                "current_step": 0
            }
        )

    plan = state["plan"]

    if current_step >= len(plan):
        return Command(goto=END)

    target_agent = plan[current_step]["agent"]
    return Command(goto=target_agent)
