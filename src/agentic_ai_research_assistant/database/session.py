import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, Session

from agentic_ai_research_assistant.config import settings
from agentic_ai_research_assistant.database.models import Base, MetricRecord

async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

sync_engine = create_engine(
    settings.SYNC_DATABASE_URL,
    echo=False,
    future=True
)
SyncSessionLocal = sessionmaker(bind=sync_engine, autocommit=False, autoflush=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def get_sync_db() -> Session:
    return SyncSessionLocal()


async def init_db():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed generic statistical metrics if empty
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(MetricRecord))
        records = result.scalars().all()
        if not records:
            seed_data = [
                MetricRecord(
                    metric="baseline_market_valuation",
                    year=2024,
                    value=500.0,
                    unit="Million USD",
                    description="Baseline market valuation across monitored commercial segments"
                ),
                MetricRecord(
                    metric="projected_market_valuation",
                    year=2029,
                    value=1850.0,
                    unit="Million USD",
                    description="Forecasted market valuation under standard adoption curves"
                ),
                MetricRecord(
                    metric="sector_growth_benchmark",
                    year=2024,
                    value=14.5,
                    unit="Percent",
                    description="Industry sector annual growth rate benchmark"
                ),
                MetricRecord(
                    metric="enterprise_retention_index",
                    year=2024,
                    value=88.2,
                    unit="Index",
                    description="Standard enterprise customer retention index"
                )
            ]
            session.add_all(seed_data)
            await session.commit()
