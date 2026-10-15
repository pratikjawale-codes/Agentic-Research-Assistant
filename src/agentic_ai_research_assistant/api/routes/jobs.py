import uuid
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from agentic_ai_research_assistant.database.session import get_db, get_sync_db
from agentic_ai_research_assistant.database.models import User, Job
from agentic_ai_research_assistant.auth.security import get_current_user
from agentic_ai_research_assistant.agents.graph import research_graph

router = APIRouter(prefix="/jobs", tags=["Background Research Jobs"])

class JobSubmitRequest(BaseModel):
    goal: str = Field(min_length=3, description="High-level research objective")

class JobResponse(BaseModel):
    id: str
    status: str
    input_goal: str
    output_report: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

def _run_background_agent_job(job_id: str, user_id: int, goal: str):
    """Executes the agent graph in a background thread with dedicated sync DB connection."""
    with get_sync_db() as session:
        job = session.execute(select(Job).where(Job.id == job_id)).scalars().first()
        if not job:
            return
        job.status = "running"
        job.updated_at = datetime.now(timezone.utc)
        session.commit()

        try:
            initial_state = {
                "messages": [{"role": "user", "content": goal}],
                "intent": {},
                "plan": [],
                "current_step": 0,
                "findings": {},
                "numbers": {},
                "report": "",
                "replans": 0,
                "errors": []
            }
            config = {
                "configurable": {
                    "user_id": str(user_id),
                    "thread_id": f"job:{job_id}"
                }
            }
            final_output = research_graph.invoke(initial_state, config=config)

            job.status = "completed"
            job.output_report = final_output.get("report", "")
            job.updated_at = datetime.now(timezone.utc)
            session.commit()
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.updated_at = datetime.now(timezone.utc)
            session.commit()


@router.post("", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_job(
    req: JobSubmitRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    job_id = str(uuid.uuid4())
    job = Job(id=job_id, user_id=current_user.id, status="queued", input_goal=req.goal)
    db.add(job)
    await db.commit()
    await db.refresh(job)

    background_tasks.add_task(_run_background_agent_job, job_id, current_user.id, req.goal)

    return JobResponse(
        id=job.id,
        status=job.status,
        input_goal=job.input_goal,
        created_at=job.created_at,
        updated_at=job.updated_at
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Job).where(Job.id == job_id, Job.user_id == current_user.id))
    job = result.scalars().first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or access denied")

    return JobResponse(
        id=job.id,
        status=job.status,
        input_goal=job.input_goal,
        output_report=job.output_report,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at
    )
