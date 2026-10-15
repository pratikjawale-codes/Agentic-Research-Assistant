import secrets
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from typing import Optional
import jwt
import bcrypt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from agentic_ai_research_assistant.config import settings
from agentic_ai_research_assistant.database.session import get_db
from agentic_ai_research_assistant.database.models import User, ApiKey

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def generate_api_key() -> tuple[str, str, str]:
    """
    Generates a cryptographically secure API key.
    Returns: (raw_key, key_prefix, key_hash)
    Raw key is shown ONCE to the user.
    """
    raw_key = "rk_" + secrets.token_urlsafe(32)
    key_prefix = raw_key[:11]
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    return raw_key, key_prefix, key_hash


def verify_api_key_hash(raw_key: str, stored_hash: str) -> bool:
    candidate_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    return hmac.compare_digest(candidate_hash, stored_hash)


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Security(api_key_header),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Dual-auth dependency: verifies either JWT Bearer token or X-API-Key.
    Injects current user_id safely without letting LLM control identity.
    """
    if token:
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            user_id = payload.get("sub")
            if user_id is None:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
            result = await db.execute(select(User).where(User.id == int(user_id)))
            user = result.scalars().first()
            if not user:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
            return user
        except jwt.PyJWTError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")

    if api_key:
        prefix = api_key[:11]
        result = await db.execute(select(ApiKey).where(ApiKey.key_prefix == prefix, ApiKey.revoked == False))
        key_record = result.scalars().first()
        if key_record and verify_api_key_hash(api_key, key_record.key_hash):
            key_record.last_used_at = datetime.now(timezone.utc)
            await db.commit()
            user_res = await db.execute(select(User).where(User.id == key_record.user_id))
            user = user_res.scalars().first()
            if user:
                return user
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated. Provide Bearer token or X-API-Key header."
    )
