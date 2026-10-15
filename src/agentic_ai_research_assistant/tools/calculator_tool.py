from langchain_core.tools import tool
from pydantic import BaseModel, Field
from agentic_ai_research_assistant.tools.validator import validated_tool_output

class CAGRArgs(BaseModel):
    start_value: float = Field(description="Initial value at the beginning of the period")
    end_value: float = Field(description="Final value at the end of the period")
    years: float = Field(description="Number of elapsed years in the investment or market timeframe")

@tool("cagr", args_schema=CAGRArgs)
@validated_tool_output
def cagr(start_value: float, end_value: float, years: float) -> dict:
    """
    Computes Compound Annual Growth Rate (CAGR): ((end_value / start_value) ** (1 / years)) - 1.
    Never let LLMs do arithmetic. Returns the calculated CAGR along with stated inputs.
    """
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return {
            "status": "error",
            "data": None,
            "sources": [],
            "message": "Values and years must be positive non-zero numbers"
        }

    cagr_value = (end_value / start_value) ** (1.0 / years) - 1.0
    cagr_pct = round(cagr_value * 100, 2)

    return {
        "status": "ok",
        "data": {
            "cagr_decimal": round(cagr_value, 4),
            "cagr_percent": f"{cagr_pct}%",
            "inputs": {
                "start_value": start_value,
                "end_value": end_value,
                "years": years
            }
        },
        "sources": ["CAGR Formula: ((End/Start)^(1/n)) - 1"],
        "message": f"CAGR calculated as {cagr_pct}% over {years} years"
    }
