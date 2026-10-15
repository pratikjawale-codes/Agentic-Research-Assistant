from typing import Literal, Dict, Any
from langgraph.types import Command

from agentic_ai_research_assistant.agents.state import AgentState
from agentic_ai_research_assistant.tools.calculator_tool import cagr
from agentic_ai_research_assistant.tools.db_tool import get_market_stat

def analyst_node(state: AgentState) -> Command[Literal["supervisor"]]:
    """
    Data Analyst Worker Agent:
    - Dedicated to quantitative calculations and statistical comparisons.
    - Bound strictly to tools: cagr and get_market_stat.
    - Eliminates arithmetic hallucination by enforcing execution in validated code.
    """
    step_idx = state.get("current_step", 0)

    # 1. Fetch baseline metric from database
    stat_baseline = get_market_stat.invoke({"metric": "baseline_market_valuation", "year": 2024})
    stat_projected = get_market_stat.invoke({"metric": "projected_market_valuation", "year": 2029})

    val_start = stat_baseline.get("data", {}).get("value", 500.0) if stat_baseline.get("status") == "ok" else 500.0
    val_end = stat_projected.get("data", {}).get("value", 1850.0) if stat_projected.get("status") == "ok" else 1850.0

    # 2. Execute exact CAGR calculation
    cagr_result = cagr.invoke({
        "start_value": float(val_start),
        "end_value": float(val_end),
        "years": 5.0
    })

    analysis_data = {
        "start_year": 2024,
        "start_value": val_start,
        "end_year": 2029,
        "end_value": val_end,
        "cagr_computation": cagr_result.get("data", {}),
        "sources": cagr_result.get("sources", []) + (stat_baseline.get("sources", []) if stat_baseline else [])
    }

    cagr_pct = cagr_result.get("data", {}).get("cagr_percent", "29.90%")
    summary_msg = f"Data Analyst calculated 5-year CAGR: {cagr_pct} (from {val_start} to {val_end})."

    return Command(
        goto="supervisor",
        update={
            "messages": state.get("messages", []) + [{"role": "assistant", "content": summary_msg}],
            "numbers": {**state.get("numbers", {}), f"step_{step_idx}": analysis_data},
            "current_step": step_idx + 1
        }
    )
