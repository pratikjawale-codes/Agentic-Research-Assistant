import json
from typing import List, Dict, Any
from sqlalchemy import select, desc
from agentic_ai_research_assistant.database.session import get_sync_db
from agentic_ai_research_assistant.database.models import MemoryStore

class LongTermMemoryManager:
    """
    Manages long-term cross-session memory partitioned by user_id.
    Stores user preferences, completed research summaries, and verified facts.
    """

    @staticmethod
    def get_user_memories(user_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        with get_sync_db() as session:
            stmt = (
                select(MemoryStore)
                .where(MemoryStore.user_id == user_id)
                .order_by(desc(MemoryStore.created_at))
                .limit(limit)
            )
            rows = session.execute(stmt).scalars().all()
            memories = []
            for r in rows:
                try:
                    meta = json.loads(r.metadata_json)
                except Exception:
                    meta = {}
                memories.append({
                    "id": r.id,
                    "type": r.memory_type,
                    "content": r.content,
                    "metadata": meta,
                    "created_at": r.created_at.isoformat() if r.created_at else None
                })
            return memories

    @staticmethod
    def save_memory(user_id: int, memory_type: str, content: str, metadata: Dict[str, Any] = None):
        with get_sync_db() as session:
            record = MemoryStore(
                user_id=user_id,
                memory_type=memory_type,
                content=content,
                metadata_json=json.dumps(metadata or {})
            )
            session.add(record)
            session.commit()

    @staticmethod
    def format_memories_for_prompt(user_id: int) -> str:
        memories = LongTermMemoryManager.get_user_memories(user_id)
        if not memories:
            return "No previous long-term memories found."
        lines = ["Background context from previous sessions (may be outdated):"]
        for m in memories:
            lines.append(f"- [{m['type'].upper()}]: {m['content']}")
        return "\n".join(lines)
