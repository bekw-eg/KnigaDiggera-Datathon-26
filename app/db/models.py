"""
Pydantic v2 models — строго по ТЗ ForteBank / Datathon 26.
Используются как JSON-ответы жюри и для внутренней валидации.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────

class VerdictEnum(str, Enum):
    """Итоговый вердикт по видео."""
    no_violation = "no_violation"
    violation = "violation"


class JobStatusEnum(str, Enum):
    """Статус задачи в очереди."""
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"
    error = "error"


# ── Evidence & Violations ───────────────────────────────────────────

class EvidenceFrame(BaseModel):
    """Кадр-доказательство нарушения."""
    timestamp_sec: float = Field(
        ...,
        description="Секунда видео, откуда извлечён кадр",
        examples=[12.5],
    )
    image_base64: str = Field(
        ...,
        description="Кадр в base64 (JPEG)",
    )


class Violation(BaseModel):
    """Одно конкретное нарушение, найденное на видео."""
    type: str = Field(
        ...,
        description="Тип нарушения (напр. 'cash_not_counted', 'no_greeting')",
        examples=["cash_not_counted"],
    )
    timecode_sec: float = Field(
        ...,
        description="Таймкод начала нарушения в секундах",
        examples=[34.0],
    )
    description: str = Field(
        ...,
        description="Человекочитаемое описание нарушения",
        examples=["Кассир не пересчитал купюры перед выдачей"],
    )
    evidence_frames: list[EvidenceFrame] = Field(
        default_factory=list,
        description="Кадры-доказательства",
    )


# ── Job result (то, что возвращаем жюри) ────────────────────────────

class JobResultResponse(BaseModel):
    """Полный ответ с результатом анализа видео."""
    job_id: str
    status: JobStatusEnum
    verdict: VerdictEnum | None = Field(
        None,
        description="Итоговый вердикт (null пока задача не завершена)",
    )
    violations: list[Violation] = Field(
        default_factory=list,
        description="Список обнаруженных нарушений (пуст если no_violation)",
    )
    video_path: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    error_message: str | None = None


# ── Request schemas ─────────────────────────────────────────────────

class JobCreateRequest(BaseModel):
    """Тело запроса на создание задачи (по пути или URL)."""
    video_url: str | None = Field(
        None,
        description="URL видеофайла для анализа",
    )
    video_path: str | None = Field(
        None,
        description="Локальный путь к видеофайлу на сервере",
    )


class JobStatusResponse(BaseModel):
    """Краткий ответ о статусе задачи (без тяжёлого result)."""
    job_id: str
    status: JobStatusEnum
    created_at: str | None = None
    updated_at: str | None = None
    error_message: str | None = None
