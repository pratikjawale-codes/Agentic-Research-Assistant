from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sqlalchemy import select
from agentic_ai_research_assistant.database.session import get_sync_db
from agentic_ai_research_assistant.database.models import MetricRecord
from agentic_ai_research_assistant.tools.validator import validated_tool_output

ALLOWED_METRICS = {
    "baseline_market_valuation",
    "projected_market_valuation",
    "sector_growth_benchmark",
    "enterprise_retention_index"
}

class MetricQueryArgs(BaseModel):
    metric: str = Field(description=f"Standardized metric name to query. Available: {sorted(list(ALLOWED_METRICS))}")
    year: int = Field(description="Calendar or fiscal year of the required statistic")

@tool("get_market_stat", args_schema=MetricQueryArgs)
@validated_tool_output
def get_market_stat(metric: str, year: int) -> dict:
    """
    Look up verified industry and quantitative market statistics from the internal database.
    Enforces read-only parameterized execution and strict metric validation.
    """
    cleaned_metric = metric.strip().lower()
    if cleaned_metric not in ALLOWED_METRICS:
        return {
            "status": "error",
            "data": None,
            "sources": [],
            "message": f"Metric '{metric}' not recognized. Allowed metrics: {sorted(list(ALLOWED_METRICS))}"
        }

    with get_sync_db() as session:
        stmt = select(MetricRecord).where(MetricRecord.metric == cleaned_metric, MetricRecord.year == year)
        row = session.execute(stmt).scalars().first()
        if not row:
            return {
                "status": "empty",
                "data": None,
                "sources": [],
                "message": f"No metric record found for '{metric}' in year {year}"
            }

        data_dict = {
            "metric": row.metric,
            "year": row.year,
            "value": row.value,
            "unit": row.unit,
            "description": row.description
        }
        return {
            "status": "ok",
            "data": data_dict,
            "sources": [f"Database Record: metric={row.metric} (year={row.year})"],
            "message": f"Found metric: {row.value} {row.unit}"
        }
