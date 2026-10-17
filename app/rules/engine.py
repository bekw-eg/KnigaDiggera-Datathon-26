"""
Rule Engine — модуль проверки нарушений кассовой дисциплины (Законодательство РК).
Принимает CandidateEvent от ML и выносит VerifiedEvent строго по контракту app/schemas.py.
"""

from __future__ import annotations

import logging
from typing import Any

from app.schemas import CandidateEvent, Observation, VerificationState, VerifiedEvent

logger = logging.getLogger("rules")


# ── Справочник нормативных актов РК ──────────────────────────────────

RULE_CATALOG: dict[str, dict[str, str]] = {
    "KZ-NB-232-P31": {
        "title": "Постановление Правления НБ РК № 232 (Правила ведения кассовых операций)",
        "clause": "Глава 4, п. 31: «Работник кассы производит полистный и поштучный пересчет наличных денег в поле зрения клиента до завершения операции»",
        "url": "https://adilet.zan.kz/rus/docs/V1900019747",
        "violation_type": "cash_not_counted",
    },
    "KZ-BANK-SEC-R3": {
        "title": "Правила организации охраны и технической укрепленности банков РК",
        "clause": "Раздел 3: «Категорически запрещается оставление наличных денег, ценностей и печатей без надзора на кассовом узле»",
        "url": "https://adilet.zan.kz/rus/docs/V1900019747",
        "violation_type": "unattended_cash",
    },
    "FORTE-REG-SEC-4": {
        "title": "Правила режима безопасности на рабочих местах кассовых работников БВУ",
        "clause": "Инструкция: Запрет использования мобильных устройств кассиром в процессе расчетно-кассового обслуживания клиента",
        "url": "https://adilet.zan.kz/rus/docs/Z950002444_",
        "violation_type": "phone_usage",
    },
    "KZ-AML-ART5": {
        "title": "Закон РК «О противодействии легализации (отмыванию) доходов...» (ПОД/ФТ)",
        "clause": "Статья 5: «Обязательное проведение надлежащей проверки клиента до завершения кассовой операции»",
        "url": "https://adilet.zan.kz/rus/docs/Z090000191_",
        "violation_type": "no_client_id",
    },
    "KZ-BANK-GENERAL": {
        "title": "Закон РК «О банках и банковской деятельности в Республике Казахстан»",
        "clause": "Статья 34: «Обеспечение банками стандартов безопасности и кассовой дисциплины»",
        "url": "https://adilet.zan.kz/rus/docs/Z950002444_",
        "violation_type": "general_discipline_violation",
    },
}


def evaluate_candidate(candidate: CandidateEvent) -> VerifiedEvent:
    """
    Оценивает отдельное визуальное событие-кандидат на соответствие нормам РК.
    Возвращает строго типизированный VerifiedEvent.
    """
    event_type = candidate.event_type.lower()
    obs = candidate.observations
    
    # 1. Проверка полистного пересчета денег (п. 31 Постановления НБ РК №232)
    # Если событие связано с выдачей денег, но пересчет перед клиентом отсутствует
    if "cash" in event_type or "count" in event_type or obs.get("cash_counted") == Observation.NO:
        if obs.get("cash_counted") == Observation.NO or obs.get("handover_without_counting") == Observation.YES:
            return VerifiedEvent(
                event_id=candidate.event_id,
                state=VerificationState.CONFIRMED,
                rule_id="KZ-NB-232-P31",
                evidence_timestamps=[candidate.start_sec],
                explanation="Кассир выдал наличные денежные средства без полистного/поштучного пересчета в поле зрения клиента (п. 31 Постановления НБ РК № 232).",
            )
    
    # 2. Оставление денег без присмотра на кассовом столе
    if "unattended" in event_type or obs.get("unattended_cash") == Observation.YES:
        return VerifiedEvent(
            event_id=candidate.event_id,
            state=VerificationState.CONFIRMED,
            rule_id="KZ-BANK-SEC-R3",
            evidence_timestamps=[candidate.start_sec],
            explanation="Оставление денежных средств или ценностей на кассовом узле без надзора кассира (нарушение режима охраны БВУ).",
        )
        
    # 3. Использование личного телефона во время обслуживания
    if "phone" in event_type or obs.get("phone_usage") == Observation.YES or obs.get("teller_holding_phone") == Observation.YES:
        return VerifiedEvent(
            event_id=candidate.event_id,
            state=VerificationState.CONFIRMED,
            rule_id="FORTE-REG-SEC-4",
            evidence_timestamps=[candidate.start_sec],
            explanation="Использование кассиром мобильного телефона в рабочей кассовой кабине во время расчетно-кассовой операции.",
        )
        
    # 4. Непроведение идентификации клиента (ПОД/ФТ)
    if "no_id" in event_type or obs.get("client_identified") == Observation.NO:
        return VerifiedEvent(
            event_id=candidate.event_id,
            state=VerificationState.CONFIRMED,
            rule_id="KZ-AML-ART5",
            evidence_timestamps=[candidate.start_sec],
            explanation="Не проведена надлежащая проверка документа, удостоверяющего личность клиента (ст. 5 Закона РК ПОД/ФТ).",
        )
    
    # Неопределенность (если есть признаки, но недостаточно данных для подтверждения)
    if any(val == Observation.UNKNOWN for val in obs.values()):
        return VerifiedEvent(
            event_id=candidate.event_id,
            state=VerificationState.UNCERTAIN,
            rule_id=None,
            evidence_timestamps=[candidate.start_sec],
            explanation="Визуальных данных недостаточно для однозначной правовой квалификации инцидента.",
        )
        
    # Нарушений не выявлено
    return VerifiedEvent(
        event_id=candidate.event_id,
        state=VerificationState.REJECTED,
        rule_id=None,
        evidence_timestamps=[candidate.start_sec],
        explanation="Признаков нарушений кассовой дисциплины не зафиксировано.",
    )


def evaluate_candidates(candidates: list[CandidateEvent]) -> list[VerifiedEvent]:
    """Применяет правила ко всем событиям-кандидатам."""
    return [evaluate_candidate(c) for c in candidates]
