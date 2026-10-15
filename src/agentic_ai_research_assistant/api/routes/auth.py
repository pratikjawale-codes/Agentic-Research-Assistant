from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from agentic_ai_research_assistant.database.session import get_db
from agentic_ai_research_assistant.database.models import User, ApiKey
from agentic_ai_research_assistant.auth.security import (
    hash_password,
    verify_password,
    create_access_token,
    generate_api_key,
    get_current_user
)

router = APIRouter(prefix="/auth", tags=["Authentication & API Keys"])

class UserRegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=6)

class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    email: str

class ApiKeyCreateRequest(BaseModel):
    name: str = Field(default="My API Key", max_length=100)

class ApiKeyCreateResponse(BaseModel):
    id: int
    name: str
    raw_key: str
    key_prefix: str
    created_at: datetime
    message: str = "Please store this raw key safely. It will not be shown again."

class ApiKeyItem(BaseModel):
    id: int
    name: str
    key_prefix: str
    revoked: bool
    created_at: datetime
    last_used_at: Optional[datetime] = None


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(req: UserRegisterRequest, db: AsyncSession = Depends(get_db)):
    # Check existing email
    existing = await db.execute(select(User).where(User.email == req.email))
    if existing.scalars().first():
        raise HTTPException(status_code=400, detail="Email already registered")

    pwd_hash = hash_password(req.password)
    new_user = User(name=req.name, email=req.email, password_hash=pwd_hash)
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    token = create_access_token({"sub": str(new_user.id), "email": new_user.email})
    return TokenResponse(access_token=token, user_id=new_user.id, email=new_user.email)


@router.post("/login", response_model=TokenResponse)
async def login(req: UserLoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == req.email))
    user = result.scalars().first()
    # Constant-time message to avoid email enumeration
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    token = create_access_token({"sub": str(user.id), "email": user.email})
    return TokenResponse(access_token=token, user_id=user.id, email=user.email)


@router.post("/api-keys", response_model=ApiKeyCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_new_api_key(
    req: ApiKeyCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    raw_key, prefix, key_hash = generate_api_key()
    key_record = ApiKey(
        user_id=current_user.id,
        name=req.name,
        key_prefix=prefix,
        key_hash=key_hash
    )
    db.add(key_record)
    await db.commit()
    await db.refresh(key_record)

    return ApiKeyCreateResponse(
        id=key_record.id,
        name=key_record.name,
        raw_key=raw_key,
        key_prefix=prefix,
        created_at=key_record.created_at
    )


@router.get("/api-keys", response_model=List[ApiKeyItem])
async def list_api_keys(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(ApiKey).where(ApiKey.user_id == current_user.id))
    keys = result.scalars().all()
    return [
        ApiKeyItem(
            id=k.id,
            name=k.name,
            key_prefix=k.key_prefix,
            revoked=k.revoked,
            created_at=k.created_at,
            last_used_at=k.last_used_at
        ) for k in keys
    ]


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == current_user.id))
    key_rec = result.scalars().first()
    if not key_rec:
        raise HTTPException(status_code=404, detail="API key not found")
    key_rec.revoked = True
    await db.commit()
    return {"message": "API key revoked successfully"}
