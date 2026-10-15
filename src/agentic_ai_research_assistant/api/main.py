from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from agentic_ai_research_assistant.config import settings
from agentic_ai_research_assistant.database.session import init_db, async_engine
from agentic_ai_research_assistant.api.routes import auth, upload, chat, jobs

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure tables and seeds exist
    await init_db()
    yield
    # Shutdown
    await async_engine.dispose()

app = FastAPI(
    title=settings.APP_NAME,
    description="Supervisor-led Multi-Agent AI Research System with Hybrid RAG, Multi-User Isolation, and Validated Tools",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health & Ready endpoints
@app.get("/health", tags=["Monitoring"])
async def health_check():
    return {"status": "ok", "app": settings.APP_NAME}

@app.get("/ready", tags=["Monitoring"])
async def readiness_check():
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "ready", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": str(e)}

# Include routers
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(upload.router, prefix=settings.API_V1_STR)
app.include_router(chat.router, prefix=settings.API_V1_STR)
app.include_router(jobs.router, prefix=settings.API_V1_STR)
