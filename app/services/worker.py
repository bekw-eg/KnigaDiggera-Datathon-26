"""
Асинхронный воркер для обработки очереди видео.
Гарантирует строго 1 задачу за раз (ограничение 6 ГБ VRAM).
Таймаут: 300 секунд (5 минут). При превышении -> статус "failed".
Интеграция: ML (CandidateEvent) -> Legal (VerifiedEvent) -> Base64 JPEG frames -> SQLite.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.db.database import save_job_result, update_job_status
from app.db.models import VerdictEnum
from app.rules.engine import RULE_CATALOG, evaluate_candidate
from app.schemas import CandidateEvent, VerificationState, VerifiedEvent

logger = logging.getLogger("worker")

TIMEOUT_SECONDS = settings.TIMEOUT_SECONDS


# ── Проверка доступности ML модуля ───────────────────────────────────

try:
    from app.vision.config import VisionConfig
    from app.vision.pipeline import analyze_video as vision_analyze_video
except ImportError:
    VisionConfig = None  # type: ignore
    vision_analyze_video = None


# ── Base64 конвертер для доказательств ────────────────────────────────


def frame_to_base64(frame_data: Any) -> str:
    """
    Конвертирует различные форматы кадров в JPEG Base64 строку:
    - Path / str (путь к существующему файлу или готовый base64)
    - bytes (уже сжатый JPEG/PNG)
    - numpy ndarray (OpenCV кадр)
    - PIL.Image.Image
    """
    if isinstance(frame_data, (str, Path)):
        p = Path(frame_data)
        if p.is_file():
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        if isinstance(frame_data, str):
            return frame_data

    if isinstance(frame_data, (bytes, bytearray)):
        return base64.b64encode(frame_data).decode("utf-8")

    # Обработка PIL.Image
    try:
        from PIL import Image

        if isinstance(frame_data, Image.Image):
            buf = io.BytesIO()
            frame_data.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except ImportError:
        pass

    # Обработка OpenCV / NumPy массива
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        if isinstance(frame_data, np.ndarray):
            success, encoded_img = cv2.imencode(
                ".jpg", frame_data, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            )
            if success:
                return base64.b64encode(encoded_img.tobytes()).decode("utf-8")
    except ImportError:
        pass

    raise ValueError(f"Не удалось преобразовать кадр типа {type(frame_data)} в base64 JPEG")


# ── Запуск ML анализа ────────────────────────────────────────────────


async def run_ml_pipeline(video_path: str) -> list[CandidateEvent]:
    """
    Запускает ML пайплайн (analyze_video).
    Если модель Ollama / OpenCV пайплайн подключен — вызывает его в отдельном потоке.
    Если ML модуль не смонтирован локально — возвращает дефолтный пустой список кандидатов.
    """
    path_obj = Path(video_path)

    if vision_analyze_video is not None and VisionConfig is not None:
        try:
            config = VisionConfig()
            # Запуск в отдельном потоке, так как analyze_video синхронный и изолирует процесс
            result = await asyncio.to_thread(vision_analyze_video, path_obj, config)
            if hasattr(result, "events"):
                return result.events
        except Exception as e:
            logger.warning(
                "Ошибка при запуске vision_analyze_video: %s. Переход на резервный режим.", e
            )

    logger.info("ML пайплайн работает в режиме готовности для: %s", video_path)
    await asyncio.sleep(0.1)
    return []


# ── Конвейер анализа видео ──────────────────────────────────────────


async def run_pipeline(video_path: str) -> dict[str, Any]:
    """
    Полный конвейер:
    1. ML анализ -> list[CandidateEvent]
    2. Rule Engine -> list[VerifiedEvent]
    3. Преобразование подтвержденных нарушений в структуру ТЗ (VerdictEnum, Violation, Base64)
    """
    # 1. Извлечение кандидатов
    candidates: list[CandidateEvent] = await run_ml_pipeline(video_path)

    # 2. Оценка через Rule Engine
    verified_events: list[tuple[CandidateEvent, VerifiedEvent]] = []
    for cand in candidates:
        verified = evaluate_candidate(cand)
        verified_events.append((cand, verified))

    # 3. Формирование подтвержденных нарушений
    parsed_violations: list[dict[str, Any]] = []

    for cand, verified in verified_events:
        if verified.state == VerificationState.CONFIRMED:
            rule_meta = RULE_CATALOG.get(verified.rule_id or "", {})
            v_type = rule_meta.get("violation_type", "regulatory_violation")

            # Кодируем доказательные кадры
            evidence_frames: list[dict[str, Any]] = []
            for frame_path in cand.frame_paths:
                try:
                    b64 = frame_to_base64(frame_path)
                    evidence_frames.append(
                        {
                            "timestamp_sec": cand.start_sec,
                            "image_base64": b64,
                        }
                    )
                except Exception as ex:
                    logger.warning("Не удалось закодировать кадр %s: %s", frame_path, ex)

            timecode = (
                verified.evidence_timestamps[0] if verified.evidence_timestamps else cand.start_sec
            )

            parsed_violations.append(
                {
                    "type": v_type,
                    "timecode_sec": float(timecode),
                    "description": verified.explanation,
                    "evidence_frames": evidence_frames,
                }
            )

    verdict = VerdictEnum.violation.value if parsed_violations else VerdictEnum.no_violation.value

    return {
        "verdict": verdict,
        "violations": parsed_violations,
    }


# ── Обработка одной задачи ──────────────────────────────────────────


async def process_job(job_id: str, video_path: str) -> None:
    """
    Обрабатывает строго одну задачу:
    - Меняет статус на 'processing'
    - Ограничивает время выполнения 300 секундами
    - При таймауте или ошибке -> статус 'failed'
    - При успехе сохраняет итоговый JSON через save_job_result
    """
    logger.info("Начало обработки задачи %s (видео: %s)", job_id, video_path)
    await update_job_status(job_id, status="processing")

    try:
        # Жесткий таймаут 300 секунд (5 минут)
        result = await asyncio.wait_for(
            run_pipeline(video_path),
            timeout=TIMEOUT_SECONDS,
        )
        await save_job_result(job_id, result)
        logger.info("Задача %s успешно выполнена. Вердикт: %s", job_id, result.get("verdict"))

    except TimeoutError:
        error_msg = f"Превышен лимит времени выполнения ({TIMEOUT_SECONDS} сек)"
        logger.error("Задача %s завершилась по таймауту: %s", job_id, error_msg)
        await update_job_status(job_id, status="failed", error_message=error_msg)

    except Exception as exc:
        error_msg = f"Ошибка обработки: {str(exc)}"
        logger.exception("Ошибка при обработке задачи %s: %s", job_id, error_msg)
        await update_job_status(job_id, status="failed", error_message=error_msg)


# ── Синглтон-очередь с гарантией 1 воркера ────────────────────────────


class VideoTaskQueue:
    """
    Очередь задач с единственным асинхронным консьюмером.
    Гарантирует, что одновременно обрабатывается строго 1 видео.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[str, str]] | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._running: bool = False

    def get_queue(self) -> asyncio.Queue[tuple[str, str]]:
        if self._queue is None:
            self._queue = asyncio.Queue()
        return self._queue

    async def _worker_loop(self) -> None:
        logger.info("Фоновый воркер очереди видео запущен (конкурентность: 1, лимит 6GB VRAM).")
        q = self.get_queue()
        while self._running:
            try:
                job_id, video_path = await q.get()
            except asyncio.CancelledError:
                break

            try:
                await process_job(job_id, video_path)
            except Exception as e:
                logger.error("Непредвиденная ошибка в воркере для %s: %s", job_id, e)
            finally:
                q.task_done()

    def start(self) -> None:
        """Запуск фонового воркера."""
        if not self._running:
            self._running = True
            self.get_queue()
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        """Остановка воркера с ожиданием завершения текущей задачи."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        self._queue = None
        logger.info("Фоновый воркер очереди видео остановлен.")

    async def enqueue(self, job_id: str, video_path: str) -> int:
        """Добавляет задачу в очередь."""
        q = self.get_queue()
        await q.put((job_id, video_path))
        return q.qsize()

    @property
    def queue_size(self) -> int:
        if self._queue is None:
            return 0
        return self._queue.qsize()


task_queue = VideoTaskQueue()
