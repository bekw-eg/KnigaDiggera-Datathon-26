"""
FastAPI эндпоинты для жюри Datathon 26 / ForteBank.
Разработчик №1 (Backend/API) | KnigaDiggera.
"""

from __future__ import annotations

import json
import shutil
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.auth import verify_api_key
from app.core.config import settings
from app.db.database import create_job, get_job, list_jobs
from app.db.models import (
    JobCreateRequest,
    JobResultResponse,
    JobStatusEnum,
    JobStatusResponse,
    VerdictEnum,
    Violation,
)
from app.services.worker import task_queue

router = APIRouter(prefix=settings.API_V1_PREFIX, tags=["Jobs API"])


@router.post(
    "/jobs/upload",
    response_model=JobStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Загрузка видеофайла (.mov, .mp4) на анализ",
    dependencies=[Depends(verify_api_key)],
)
async def upload_job_video(
    file: Annotated[UploadFile, File(description="Видеозапись кассового узла (.mov, .mp4)")],
) -> JobStatusResponse:
    """
    Принимает видеофайл через multipart/form-data.
    Сохраняет в хранилище, регистрирует задачу в SQLite и ставит в очередь (1 видео за раз).
    """
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Создаем предварительную запись задачи в БД
    temp_path = str(settings.UPLOAD_DIR / f"incoming_{file.filename}")
    job_record = await create_job(video_path=temp_path)
    job_id = job_record["job_id"]

    # 2. Сохраняем файл с уникальным префиксом job_id
    safe_filename = f"{job_id}_{file.filename}"
    saved_path = settings.UPLOAD_DIR / safe_filename

    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось сохранить загруженный файл: {e}",
        ) from e
    finally:
        await file.close()

    # 3. Добавляем в очередь задач
    await task_queue.enqueue(job_id, str(saved_path))

    return JobStatusResponse(
        job_id=job_id,
        status=JobStatusEnum.queued,
        created_at=job_record.get("created_at"),
        updated_at=job_record.get("updated_at"),
    )


@router.post(
    "/jobs",
    response_model=JobStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Создание задачи по URL или локальному пути",
    dependencies=[Depends(verify_api_key)],
)
async def create_job_from_url(
    payload: JobCreateRequest,
) -> JobStatusResponse:
    """Создает задачу на анализ видео по указанному video_url."""
    video_target = payload.video_path or payload.video_url or "unknown_source.mov"
    job_record = await create_job(video_path=video_target)
    job_id = job_record["job_id"]

    await task_queue.enqueue(job_id, video_target)

    return JobStatusResponse(
        job_id=job_id,
        status=JobStatusEnum.queued,
        created_at=job_record.get("created_at"),
        updated_at=job_record.get("updated_at"),
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobResultResponse,
    summary="Получение статуса и результатов анализа задачи",
    dependencies=[Depends(verify_api_key)],
)
async def get_job_result(
    job_id: str,
) -> JobResultResponse:
    """
    Возвращает текущий статус задачи.
    Когда статус 'done', возвращает итоговый вердикт и все доказанные нарушения.
    """
    job = await get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Задача с job_id '{job_id}' не найдена",
        )

    verdict: VerdictEnum | None = None
    violations: list[Violation] = []

    # Десериализуем итоговый результат
    if job.get("result_json"):
        try:
            res_dict = json.loads(job["result_json"])
            raw_verdict = res_dict.get("verdict")
            if raw_verdict:
                verdict = VerdictEnum(raw_verdict)
            violations_data = res_dict.get("violations", [])
            violations = [Violation.model_validate(v) for v in violations_data]
        except Exception:
            # Если возникла ошибка парсинга сохраненного JSON
            pass

    return JobResultResponse(
        job_id=job["job_id"],
        status=JobStatusEnum(job["status"]),
        verdict=verdict,
        violations=violations,
        video_path=job.get("video_path"),
        created_at=job.get("created_at"),
        updated_at=job.get("updated_at"),
        error_message=job.get("error_message"),
    )


@router.get(
    "/jobs",
    response_model=list[JobStatusResponse],
    summary="Список последних задач",
    dependencies=[Depends(verify_api_key)],
)
async def get_recent_jobs() -> list[JobStatusResponse]:
    """Возвращает список последних 50 задач."""
    jobs = await list_jobs(limit=50)
    return [
        JobStatusResponse(
            job_id=j["job_id"],
            status=JobStatusEnum(j["status"]),
            created_at=j.get("created_at"),
            updated_at=j.get("updated_at"),
            error_message=j.get("error_message"),
        )
        for j in jobs
    ]
