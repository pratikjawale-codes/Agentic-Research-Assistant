from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from agentic_ai_research_assistant.database.session import get_db
from agentic_ai_research_assistant.database.models import User, Chat, Message
from agentic_ai_research_assistant.auth.security import get_current_user
from agentic_ai_research_assistant.memory.short_term import global_short_term_memory
from agentic_ai_research_assistant.memory.long_term import LongTermMemoryManager
from agentic_ai_research_assistant.agents.graph import research_graph

router = APIRouter(tags=["Multi-Agent Chat & Orchestration"])

class CreateChatRequest(BaseModel):
    title: str = Field(default="GLP-1 Market Research", max_length=255)

class ChatItem(BaseModel):
    id: int
    title: str
    created_at: str

class ChatMessageRequest(BaseModel):
    chat_id: int
    message: str = Field(min_length=1)

class ChatMessageResponse(BaseModel):
    chat_id: int
    intent: Dict[str, Any]
    plan: List[Dict[str, Any]]
    report: str
    reply: str


@router.get("/chats", response_model=List[ChatItem])
async def list_chats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Chat).where(Chat.user_id == current_user.id))
    chats = result.scalars().all()
    return [
        ChatItem(id=c.id, title=c.title, created_at=c.created_at.isoformat() if c.created_at else "")
        for c in chats
    ]


@router.post("/chats", response_model=ChatItem, status_code=status.HTTP_201_CREATED)
async def create_chat(
    req: CreateChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    new_chat = Chat(user_id=current_user.id, title=req.title)
    db.add(new_chat)
    await db.commit()
    await db.refresh(new_chat)
    return ChatItem(
        id=new_chat.id,
        title=new_chat.title,
        created_at=new_chat.created_at.isoformat() if new_chat.created_at else ""
    )


@router.post("/chat", response_model=ChatMessageResponse)
async def chat_endpoint(
    req: ChatMessageRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # Verify chat ownership (prevents BOLA / IDOR)
    chat_res = await db.execute(select(Chat).where(Chat.id == req.chat_id, Chat.user_id == current_user.id))
    chat = chat_res.scalars().first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found or access denied")

    # Persist incoming user message
    user_msg_record = Message(chat_id=chat.id, role="user", content=req.message)
    db.add(user_msg_record)
    await db.commit()

    thread_id = f"{current_user.id}:{chat.id}"
    thread_state = global_short_term_memory.get_thread_state(thread_id)

    # Inject long-term memory background context
    lt_notes = LongTermMemoryManager.format_memories_for_prompt(current_user.id)

    # Prepare inputs for LangGraph
    messages_history = thread_state.get("messages", []) + [{"role": "user", "content": req.message}]

    initial_state = {
        "messages": messages_history,
        "intent": {},
        "plan": [],
        "current_step": 0,
        "findings": thread_state.get("findings", {}),
        "numbers": thread_state.get("numbers", {}),
        "report": "",
        "replans": 0,
        "errors": []
    }

    # Pass server-side injected user_id in RunnableConfig
    config = {
        "configurable": {
            "user_id": str(current_user.id),
            "thread_id": thread_id
        }
    }

    # Execute LangGraph workflow
    final_output = research_graph.invoke(initial_state, config=config)

    report_text = final_output.get("report", "")
    assistant_reply = report_text if report_text else "Research process finished."

    # Save to short term memory
    global_short_term_memory.save_thread_state(thread_id, {
        "messages": final_output.get("messages", []),
        "findings": final_output.get("findings", {}),
        "numbers": final_output.get("numbers", {})
    })

    # Persist assistant message in DB
    asst_msg_record = Message(chat_id=chat.id, role="assistant", content=assistant_reply)
    db.add(asst_msg_record)
    await db.commit()

    # Background write: Save task summary to long-term memory
    if report_text:
        LongTermMemoryManager.save_memory(
            user_id=current_user.id,
            memory_type="task_summary",
            content=f"Completed research for chat '{chat.title}'. Query: {req.message[:120]}",
            metadata={"chat_id": chat.id, "has_report": True}
        )

    return ChatMessageResponse(
        chat_id=chat.id,
        intent=final_output.get("intent", {}),
        plan=final_output.get("plan", []),
        report=report_text,
        reply=assistant_reply
    )
