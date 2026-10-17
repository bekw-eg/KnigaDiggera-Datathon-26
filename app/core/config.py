"""
Конфигурация бэкенда ForteBank Video AI Audit.
Разработчик №1 (Backend/API) | KnigaDiggera.
"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Глобальные настройки сервиса."""
    
    # Авторизация для жюри
    API_KEY: str = Field(
        default_factory=lambda: os.getenv("API_KEY", "fortebank-jury-secret"),
        description="Ключ авторизации для жюри в заголовке X-API-Key",
    )
    
    # База данных SQLite
    DB_PATH: Path = Field(
        default_factory=lambda: Path(os.getenv("DB_PATH", "data/jobs.db")),
        description="Путь к файлу SQLite базы данных",
    )
    
    # Директория для загружаемых видео
    UPLOAD_DIR: Path = Field(
        default_factory=lambda: Path(os.getenv("UPLOAD_DIR", "data/uploads")),
        description="Директория сохранения входящих видеозаписей касс",
    )
    
    # Лимиты выполнения
    TIMEOUT_SECONDS: int = Field(
        default=300,
        description="Жесткий таймаут анализа одного видео (5 минут по ТЗ)",
    )
    
    MAX_CONCURRENT_JOBS: int = Field(
        default=1,
        description="Максимальное число одновременно обрабатываемых видео (VRAM 6 ГБ)",
    )
    
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "ForteBank Cash Discipline Video Inspector"
    VERSION: str = "0.1.0"


settings = Settings()
