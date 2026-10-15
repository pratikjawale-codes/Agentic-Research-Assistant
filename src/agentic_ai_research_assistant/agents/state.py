from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """
    Shared state that flows across nodes in the multi-agent graph.
    Only the last message of each worker is returned to prevent context bloating.
    """
    messages: List[Dict[str, str]]
    intent: Dict[str, Any]
    plan: List[Dict[str, Any]]
    current_step: int
    findings: Dict[str, Any]
    numbers: Dict[str, Any]
    report: str
    replans: int
    errors: List[str]
