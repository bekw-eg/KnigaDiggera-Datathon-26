"""
Тесты бэкенда, очереди и Rule Engine для ForteBank Video Inspector.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.db import database
from app.main import app
from app.rules.engine import evaluate_candidate
from app.schemas import CandidateEvent, Observation, VerificationState
from app.services import worker


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Exercise real routes/SQLite while keeping all storage and ML test-local."""
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "jobs.db")
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "uploads")

    async def mock_ml_pipeline(video_path: str) -> list[CandidateEvent]:
        return []

    monkeypatch.setattr(worker, "run_ml_pipeline", mock_ml_pipeline)
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    """Проверка доступности эндпоинта /health."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "queue_size" in data


def test_auth_unauthorized(client):
    """Проверка отклонения запроса без X-API-Key."""
    response = client.get("/api/v1/jobs/nonexistent-id")
    assert response.status_code == 401
    assert "X-API-Key" in response.json()["detail"]


def test_auth_invalid_key(client):
    """Проверка отклонения с неверным ключом."""
    response = client.get(
        "/api/v1/jobs/nonexistent-id",
        headers={"X-API-Key": "wrong-secret-key"},
    )
    assert response.status_code == 401


def test_create_job_and_poll(client):
    """Создание задачи через API и опрос ее статуса."""
    headers = {"X-API-Key": settings.API_KEY}

    # 1. Создание задачи
    create_resp = client.post(
        "/api/v1/jobs",
        headers=headers,
        json={"video_url": "test_cashier_feed.mov"},
    )
    assert create_resp.status_code == 202
    job_data = create_resp.json()
    job_id = job_data["job_id"]
    assert job_data["status"] == "queued"

    # 2. Получение статуса задачи
    get_resp = client.get(f"/api/v1/jobs/{job_id}", headers=headers)
    assert get_resp.status_code == 200
    res = get_resp.json()
    assert res["job_id"] == job_id
    assert res["status"] in ("queued", "processing", "done")


def test_jobs_start_with_isolated_database(client):
    response = client.get("/api/v1/jobs", headers={"X-API-Key": settings.API_KEY})
    assert response.status_code == 200
    assert response.json() == []


def test_upload_file_is_required(client):
    response = client.post("/api/v1/jobs/upload", headers={"X-API-Key": settings.API_KEY})
    assert response.status_code == 422


def test_upload_uses_test_storage_and_mock_ml(client):
    response = client.post(
        "/api/v1/jobs/upload",
        headers={"X-API-Key": settings.API_KEY},
        files={"file": ("synthetic.mov", b"test-only-placeholder", "video/quicktime")},
    )
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "queued"
    saved = settings.UPLOAD_DIR / f"{job['job_id']}_synthetic.mov"
    assert saved.read_bytes() == b"test-only-placeholder"


def test_upload_write_error_stays_visible(client, monkeypatch):
    def fail_write(*args, **kwargs):
        raise OSError("synthetic write failure")

    monkeypatch.setattr("app.api.routes.shutil.copyfileobj", fail_write)
    response = client.post(
        "/api/v1/jobs/upload",
        headers={"X-API-Key": settings.API_KEY},
        files={"file": ("synthetic.mov", b"test-only-placeholder", "video/quicktime")},
    )
    assert response.status_code == 500
    assert "synthetic write failure" in response.json()["detail"]


def test_rule_engine_cash_not_counted():
    """Тест Rule Engine: обнаружение отсутствия полистного пересчета (п.31 Правил НБ РК)."""
    cand = CandidateEvent(
        event_id="cand-001",
        event_type="cash_handover",
        start_sec=12.0,
        end_sec=15.0,
        frame_paths=[],
        observations={"cash_counted": Observation.NO},
    )

    verified = evaluate_candidate(cand)
    assert verified.state == VerificationState.CONFIRMED
    assert verified.rule_id == "KZ-NB-232-P31"
    assert "Постановления НБ РК № 232" in verified.explanation


def test_rule_engine_clean_no_violation():
    """Тест Rule Engine: отсутствие нарушений при корректном обслуживании."""
    cand = CandidateEvent(
        event_id="cand-002",
        event_type="standard_service",
        start_sec=5.0,
        end_sec=10.0,
        frame_paths=[],
        observations={"cash_counted": Observation.YES, "phone_usage": Observation.NO},
    )

    verified = evaluate_candidate(cand)
    assert verified.state == VerificationState.REJECTED
    assert verified.rule_id is None
