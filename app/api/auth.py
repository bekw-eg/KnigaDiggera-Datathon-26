"""
Модуль авторизации для жюри Datathon 26 по заголовку X-API-Key.
"""

from __future__ import annotations

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(api_key_header),
) -> str:
    """
    Проверяет валидность ключа жюри в заголовке X-API-Key.
    Если ключ отсутствует или неверен -> 401 Unauthorized.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Отсутствует обязательный заголовок 'X-API-Key'",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный 'X-API-Key'. Доступ запрещен.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key
