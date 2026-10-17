"""
Главная точка входа в сервис FastAPI для ForteBank / Datathon 26.
Разработчик №1 (Backend/API) | KnigaDiggera.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as jobs_router
from app.core.config import settings
from app.db.database import init_db
from app.services.worker import task_queue


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Жизненный цикл сервиса: инициализация БД и запуск фонового воркера."""
    # Startup
    await init_db()
    task_queue.start()
    yield
    # Shutdown
    await task_queue.stop()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Сервис аудита кассовой дисциплины ForteBank (Datathon 26, команда KnigaDiggera).",
    lifespan=lifespan,
)

# Настройка CORS для работы со Streamlit UI и внешними клиентами
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение маршрутов API
app.include_router(jobs_router)


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str | int]:
    """Проверка доступности сервиса и состояния очереди."""
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "queue_size": task_queue.queue_size,
        "version": settings.VERSION,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
