"""
Тесты бэкенда, очереди и Rule Engine для ForteBank Video Inspector.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.rules.engine import evaluate_candidate
from app.schemas import CandidateEvent, Observation, VerificationState


@pytest.fixture
def client():
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
