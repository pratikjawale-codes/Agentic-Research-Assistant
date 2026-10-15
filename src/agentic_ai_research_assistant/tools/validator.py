import functools
from typing import Any, List, Optional, Callable
from pydantic import BaseModel, Field

class ToolOutputEnvelope(BaseModel):
    status: str = Field(description="'ok', 'empty', or 'error'")
    data: Optional[Any] = Field(default=None, description="Payload data")
    sources: List[str] = Field(default_factory=list, description="Source URLs, doc filenames, or citations")
    message: str = Field(default="", description="Descriptive status or error message")


def validated_tool_output(func: Callable) -> Callable:
    """
    Decorator enforcing that tool execution always returns a validated ToolOutputEnvelope dict.
    Never lets tools throw unhandled exceptions to crash the agent loop.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> dict:
        try:
            res = func(*args, **kwargs)
            if isinstance(res, dict) and "status" in res:
                envelope = ToolOutputEnvelope(
                    status=res.get("status", "ok"),
                    data=res.get("data"),
                    sources=res.get("sources", []),
                    message=res.get("message", "")
                )
            elif isinstance(res, ToolOutputEnvelope):
                envelope = res
            else:
                envelope = ToolOutputEnvelope(status="ok", data=res)
            return envelope.model_dump()
        except Exception as e:
            return ToolOutputEnvelope(
                status="error",
                data=None,
                sources=[],
                message=f"Tool {func.__name__} execution error: {type(e).__name__}: {str(e)}"
            ).model_dump()
    return wrapper
