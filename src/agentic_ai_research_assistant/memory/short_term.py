from typing import Dict, Any, List
from langchain_core.messages import BaseMessage, trim_messages

class ShortTermMemoryManager:
    """
    Manages session/thread working memory keyed by thread_id: '{user_id}:{chat_id}'.
    Enforces context window controls via sliding window message trimming.
    """
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self._threads: Dict[str, Dict[str, Any]] = {}

    def get_thread_state(self, thread_id: str) -> Dict[str, Any]:
        return self._threads.get(thread_id, {
            "messages": [],
            "findings": {},
            "numbers": {}
        })

    def save_thread_state(self, thread_id: str, state: Dict[str, Any]):
        # Sliding window over messages
        msgs = state.get("messages", [])
        if len(msgs) > self.max_messages:
            msgs = msgs[-self.max_messages:]
            state["messages"] = msgs
        self._threads[thread_id] = state

    def clear_thread(self, thread_id: str):
        self._threads.pop(thread_id, None)

global_short_term_memory = ShortTermMemoryManager()
