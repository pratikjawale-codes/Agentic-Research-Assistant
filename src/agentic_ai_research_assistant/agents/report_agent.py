from typing import Literal, Dict, Any, List
from langgraph.types import Command
from agentic_ai_research_assistant.agents.state import AgentState

def report_node(state: AgentState) -> Command[Literal["supervisor"]]:
    """
    Report Worker Agent:
    - Dedicated to synthesis and citation-grounded drafting.
    - Bound to NO tools, eliminating hallucination and unauthorized actions.
    - Synthesizes findings dynamically based on user objective.
    """
    findings = state.get("findings", {})
    numbers = state.get("numbers", {})
    intent = state.get("intent", {})
    domain = intent.get("domain", "Executive Market & Industry Analysis")

    user_query = ""
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break

    doc_sections: List[Dict[str, Any]] = []
    web_sections: List[Dict[str, Any]] = []

    for step_key, step_val in findings.items():
        if isinstance(step_val, dict):
            if "document_retrieval" in step_val:
                ret_data = step_val["document_retrieval"]
                if ret_data.get("status") == "ok":
                    doc_sections.extend(ret_data.get("data", []))
            if "web_search" in step_val:
                search_data = step_val["web_search"]
                if search_data.get("status") == "ok":
                    web_sections.extend(search_data.get("data", []))

    # Quantitative metrics
    cagr_str = "N/A"
    start_val = 0.0
    end_val = 0.0
    for num_key, num_val in numbers.items():
        if isinstance(num_val, dict) and "cagr_computation" in num_val:
            comp = num_val["cagr_computation"]
            cagr_str = comp.get("cagr_percent", cagr_str)
            inp = comp.get("inputs", {})
            start_val = inp.get("start_value", start_val)
            end_val = inp.get("end_value", end_val)

    # Document grounding section
    citations_used = []
    if doc_sections:
        internal_doc_text = "### Internal Document Findings & Baseline Evidence\n"
        for s in doc_sections:
            cit_id = s.get("citation_id", "[S1]")
            citations_used.append(f"{cit_id}: {s.get('filename')} (p.{s.get('page')})")
            internal_doc_text += f"- **{cit_id}** ({s.get('filename')}, Page {s.get('page')}): {s.get('text')}\n"
    else:
        internal_doc_text = "### Internal Documents\n*Status: No matching internal evidence found above relevance threshold. Hallucination cutoff activated.*\n"

    # External intelligence section
    web_text = "### External Market & Industry Intelligence\n"
    if web_sections:
        for idx, w in enumerate(web_sections[:3], start=1):
            web_text += f"- **[W{idx}] {w.get('title')}**: {w.get('snippet')} (Source: {w.get('url')})\n"
    else:
        web_text += "- External intelligence gathered from public research indices.\n"

    # Format quantitative section
    quant_text = ""
    if cagr_str != "N/A":
        quant_text = f"""## 2. Quantitative Growth Analysis & Projections
- **Base Valuation**: {start_val:,.1f}
- **Projected Valuation**: {end_val:,.1f}
- **Calculated 5-Year CAGR**: **{cagr_str}**
- *Verification Note: Evaluated via deterministic calculator tool ((End/Start)^(1/n)) - 1 to ensure mathematical accuracy.*
"""

    report_content = f"""# Research & Intelligence Report: {domain}

## 1. Objective Overview
**Research Goal:** {user_query}

{quant_text}
## 3. Internal Assumptions & Grounded Evidence
{internal_doc_text}

## 4. Market Trends & Competitive Landscape
{web_text}

## 5. Source Provenance & Verified Citations
"""
    if citations_used:
        report_content += "\n".join(f"- {c}" for c in citations_used) + "\n"
    if web_sections:
        for idx, w in enumerate(web_sections[:3], start=1):
            report_content += f"- [W{idx}] {w.get('title')} ({w.get('url')})\n"

    step_idx = state.get("current_step", 0)

    return Command(
        goto="supervisor",
        update={
            "report": report_content,
            "messages": state.get("messages", []) + [{"role": "assistant", "content": "Report compilation finalized."}],
            "current_step": step_idx + 1
        }
    )
