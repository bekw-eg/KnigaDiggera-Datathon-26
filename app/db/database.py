"""
Async SQLite database layer using aiosqlite.
Single-file DB — идеально для datathon, zero config.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import aiosqlite

from app.core.config import settings

DB_PATH = settings.DB_PATH

# ── helpers ──────────────────────────────────────────────────────────


def _now_iso() -> str:
    """UTC timestamp in ISO-8601."""
    return datetime.now(UTC).isoformat()


def _new_job_id() -> str:
    return uuid.uuid4().hex


# ── schema bootstrap ────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id         TEXT PRIMARY KEY,
    status         TEXT NOT NULL DEFAULT 'queued',   -- queued | processing | done | error
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    video_path     TEXT,
    result_json    TEXT,      -- сериализованный JSON с вердиктом
    error_message  TEXT
);
"""


async def init_db() -> None:
    """Create DB file + tables if they don't exist yet."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(_SCHEMA_SQL)
        await db.commit()


@asynccontextmanager
async def get_db() -> AsyncIterator[aiosqlite.Connection]:
    """Provide a dedicated connection context with row_factory enabled."""
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    try:
        yield db
    finally:
        await db.close()


# ── CRUD ─────────────────────────────────────────────────────────────


async def create_job(video_path: str) -> dict[str, Any]:
    """Insert a new job row and return it as dict."""
    job_id = _new_job_id()
    now = _now_iso()
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO jobs (job_id, status, created_at, updated_at, video_path)
            VALUES (?, 'queued', ?, ?, ?)
            """,
            (job_id, now, now, video_path),
        )
        await db.commit()
    return {
        "job_id": job_id,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "video_path": video_path,
        "result_json": None,
        "error_message": None,
    }


async def get_job(job_id: str) -> dict[str, Any] | None:
    """Fetch a single job by id. Returns None if not found."""
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)


async def update_job_status(
    job_id: str,
    status: str,
    *,
    error_message: str | None = None,
) -> None:
    """Set status (and optional error_message), bump updated_at."""
    now = _now_iso()
    async with get_db() as db:
        await db.execute(
            """
            UPDATE jobs
               SET status = ?, updated_at = ?, error_message = ?
             WHERE job_id = ?
            """,
            (status, now, error_message, job_id),
        )
        await db.commit()


async def save_job_result(job_id: str, result: dict[str, Any]) -> None:
    """Persist the final analysis JSON and mark the job as done."""
    now = _now_iso()
    async with get_db() as db:
        await db.execute(
            """
            UPDATE jobs
               SET status = 'done',
                   updated_at = ?,
                   result_json = ?
             WHERE job_id = ?
            """,
            (now, json.dumps(result, ensure_ascii=False), job_id),
        )
        await db.commit()


async def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent jobs (newest first)."""
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
