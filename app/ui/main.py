"""
Streamlit UI для ForteBank (Анализ видеокасс)
Команда: KnigaDiggera | Datathon 26
Автор: Yera-KAZ

Функционал:
- Загрузка видеофайла (.mov, .mp4)
- Отправка в FastAPI /api/v1/jobs
- Асинхронный поллинг статуса (queued -> processing -> done)
- Отображение вердикта («Нарушение не обнаружено» / «Обнаружены признаки возможных нарушений»)
- Список нарушений: таймкоды, описания, декодированные кадры-доказательства Base64
- Ссылки на нормативно-правовые акты РК (adilet.zan.kz)
"""

from __future__ import annotations

import base64
import io
import time
from typing import Any

import requests
import streamlit as st

# ── Нормативная база РК (adilet.zan.kz) ───────────────────────────────

LEGAL_REFERENCES: dict[str, dict[str, str]] = {
    "cash_not_counted": {
        "law": "Постановление Правления НБ РК № 232 (Правила кассовых операций)",
        "clause": "Глава 4, п. 31: «Работник кассы производит полистный и поштучный пересчет наличных денег в поле зрения клиента до завершения операции»",
        "url": "https://adilet.zan.kz/rus/docs/V1900019747",
    },
    "unattended_cash": {
        "law": "Правила организации охраны и технической укрепленности банков РК",
        "clause": "Раздел 3: «Категорически запрещается оставление наличных денег, бланков строгой отчетности и печатей без надзора на кассовом столе»",
        "url": "https://adilet.zan.kz/rus/docs/V1900019747",
    },
    "no_client_id": {
        "law": "Закон РК «О противодействии легализации (отмыванию) доходов...» (ПОД/ФТ)",
        "clause": "Статья 5: «Субъекты финансового мониторинга обязаны проводить надлежащую проверку клиента до проведения кассовых операций»",
        "url": "https://adilet.zan.kz/rus/docs/Z090000191_",
    },
    "no_greeting": {
        "law": "Закон РК «О банках и банковской деятельности в Республике Казахстан»",
        "clause": "Статья 34: «Обязанность банков по обеспечению стандартов добросовестного и профессионального поведения при обслуживании клиентов»",
        "url": "https://adilet.zan.kz/rus/docs/Z950002444_",
    },
    "phone_usage": {
        "law": "Правила безопасности и режима на рабочих местах кассовых работников БВУ",
        "clause": "Инструкция по безопасности: Запрет использования личных мобильных устройств в кассовой кабине во время расчетно-кассового обслуживания",
        "url": "https://adilet.zan.kz/rus/docs/Z950002444_",
    },
}

DEFAULT_LEGAL = {
    "law": "Закон РК «О банках и банковской деятельности в Республике Казахстан»",
    "clause": "Статья 34-1: Требования к внутренним правилам и регламентам проведения кассовых операций",
    "url": "https://adilet.zan.kz/rus/docs/Z950002444_",
}


def get_legal_info(violation_type: str) -> dict[str, str]:
    """Возвращает нормативную базу по типу нарушения."""
    v_clean = violation_type.lower().strip()
    return LEGAL_REFERENCES.get(v_clean, DEFAULT_LEGAL)


def format_seconds(seconds: float) -> str:
    """Форматирует секунды в MM:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d} ({seconds:.1f}с)"


# ── Стилизация страницы ──────────────────────────────────────────────

st.set_page_config(
    page_title="ForteBank • Анализ видеокасс",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ForteBank Brand Colors */
    :root {
        --forte-primary: #9B1138;
        --forte-primary-dark: #7A0A2B;
        --forte-bg: #FAFAFC;
    }
    
    .main-header {
        background: linear-gradient(135deg, #9B1138 0%, #4A0E17 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(155, 17, 56, 0.25);
    }
    .main-header h1 {
        color: white !important;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .main-header p {
        color: #F8D7DA;
        font-size: 1.05rem;
        margin: 0;
    }
    
    .verdict-card-clean {
        background: linear-gradient(135deg, #107C41 0%, #0B552C 100%);
        color: white;
        padding: 22px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.45rem;
        font-weight: 700;
        margin: 16px 0;
        box-shadow: 0 4px 12px rgba(16, 124, 65, 0.25);
    }
    
    .verdict-card-violation {
        background: linear-gradient(135deg, #D83B01 0%, #A80000 100%);
        color: white;
        padding: 22px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.45rem;
        font-weight: 700;
        margin: 16px 0;
        box-shadow: 0 4px 12px rgba(216, 59, 1, 0.25);
    }
    
    .violation-box {
        border-left: 5px solid #9B1138;
        background-color: #FDF4F6;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 16px;
    }
    
    .legal-tag {
        background-color: #EBF3FC;
        border: 1px solid #C7E0F4;
        color: #004578;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.9rem;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Сайдбар: Настройки подключения к API ──────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/42/ForteBank_logo.svg", width=180)
    st.markdown("### ⚙️ Настройки API")
    api_base_url = st.text_input("URL сервиса FastAPI", value="http://127.0.0.1:8000")
    api_key = st.text_input("X-API-Key (для жюри)", value="fortebank-jury-secret", type="password")
    
    st.markdown("---")
    st.markdown(
        """
        **Команда:** KnigaDiggera  
        **Разработчик:** Yera-KAZ  
        **Контест:** Datathon 26  
        **Кейс:** ForteBank Video AI Audit  
        """
    )
    
    # Проверка доступности API
    if st.button("Проверить соединение с API"):
        try:
            r = requests.get(f"{api_base_url.rstrip('/')}/health", timeout=3)
            if r.status_code == 200:
                st.success("API онлайн! ✅")
            else:
                st.warning(f"Статус: {r.status_code}")
        except Exception as e:
            st.error(f"Не удалось подключиться: {e}")


# ── Главная страница ─────────────────────────────────────────────────

st.markdown(
    """
    <div class="main-header">
        <h1>🏦 ForteBank • Контроль кассовой дисциплины</h1>
        <p>Автоматизированный видеомониторинг кассовых узлов и соблюдения регламентов НБ РК</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_upload, tab_history = st.tabs(["📤 Загрузка и анализ видео", "📋 Просмотр по Job ID"])

# ─────────────────────────────────────────────────────────────────────
# TAB 1: Загрузка и анализ видео
# ─────────────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown("### 1. Загрузите видеозапись кассы")
    uploaded_file = st.file_uploader(
        "Выберите видеозапись кассового обслуживания (поддерживается .mov, .mp4):",
        type=["mov", "mp4", "avi", "mkv"],
        help="Загрузите файл для проверки соблюдения регламентов пересчета наличных и протоколов безопасности.",
    )

    if uploaded_file is not None:
        col_preview, col_info = st.columns([2, 1])
        with col_preview:
            st.video(uploaded_file)
        with col_info:
            st.info(
                f"**Имя файла:** `{uploaded_file.name}`  \n"
                f"**Размер:** `{uploaded_file.size / (1024 * 1024):.2f} MB`  \n"
                f"**Тип:** `{uploaded_file.type}`"
            )

        start_btn = st.button("🚀 Начать анализ видео", type="primary", use_container_width=True)

        if start_btn:
            # 1. Отправка файла в API
            headers = {"X-API-Key": api_key}
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "video/quicktime")}
            
            with st.spinner("Отправка видеофайла в защищенный шлюз ForteBank..."):
                try:
                    submit_url = f"{api_base_url.rstrip('/')}/api/v1/jobs/upload"
                    resp = requests.post(submit_url, headers=headers, files=files, timeout=60)
                    
                    if resp.status_code not in (200, 201, 202):
                        # Попробуем альтернативный роут /api/v1/jobs если /upload не используется
                        submit_url_alt = f"{api_base_url.rstrip('/')}/api/v1/jobs"
                        resp = requests.post(submit_url_alt, headers=headers, files=files, timeout=60)
                    
                    if resp.status_code not in (200, 201, 202):
                        st.error(f"Ошибка API при отправке видео ({resp.status_code}): {resp.text}")
                        st.stop()
                    
                    data = resp.json()
                    job_id = data.get("job_id")
                    st.success(f"Задача успешно принята в очередь! Job ID: `{job_id}`")
                    
                except requests.RequestException as e:
                    st.error(f"Не удалось связаться с сервером API: {e}")
                    st.stop()

            # 2. Поллинг статуса задачи
            progress_bar = st.progress(0, text="Ожидание в очереди (VRAM 6 ГБ, 1 видео за раз)...")
            status_box = st.empty()
            
            poll_url = f"{api_base_url.rstrip('/')}/api/v1/jobs/{job_id}"
            completed = False
            result_data: dict[str, Any] = {}
            start_poll_time = time.time()
            timeout_limit = 320  # чуть больше воркерского таймаута 300с
            
            while not completed and (time.time() - start_poll_time < timeout_limit):
                try:
                    poll_resp = requests.get(poll_url, headers=headers, timeout=10)
                    if poll_resp.status_code == 200:
                        job_info = poll_resp.json()
                        current_status = job_info.get("status")
                        
                        if current_status == "queued":
                            progress_bar.progress(15, text="🕒 Задача в очереди. Ожидание освобождения GPU...")
                        elif current_status == "processing":
                            progress_bar.progress(55, text="⚡ Нейросеть анализирует видеокадры и действия кассира...")
                        elif current_status == "done":
                            progress_bar.progress(100, text="✅ Анализ завершен!")
                            result_data = job_info
                            completed = True
                            break
                        elif current_status in ("failed", "error"):
                            progress_bar.progress(100, text="❌ Обработка завершилась ошибкой")
                            st.error(f"Ошибка обработки: {job_info.get('error_message', 'Неизвестная ошибка')}")
                            completed = True
                            st.stop()
                    else:
                        status_box.warning(f"Опрос статуса ({poll_resp.status_code})...")
                except Exception as ex:
                    status_box.warning(f"Связь с сервером: {ex}")
                
                time.sleep(2.0)
            
            if not completed:
                st.error("Превышено время ожидания ответа от сервиса.")
                st.stop()

            # 3. Красивое отображение результатов
            st.markdown("---")
            st.markdown("## 📊 Результаты аудита видеозаписи")
            
            verdict = result_data.get("verdict")
            violations = result_data.get("violations", [])
            
            # Отображение вердикта строго по ТЗ
            if verdict == "no_violation":
                st.markdown(
                    """
                    <div class="verdict-card-clean">
                        🛡️ НАРУШЕНИЕ НЕ ОБНАРУЖЕНО
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.success("Все требования регламента и кассовой дисциплины соблюдены в полном объеме.")
            else:
                st.markdown(
                    """
                    <div class="verdict-card-violation">
                        ⚠️ ОБНАРУЖЕНЫ ПРИЗНАКИ ВОЗМОЖНЫХ НАРУШЕНИЙ
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.warning(f"Всего выявлено инцидентов: **{len(violations)}**")

            # Список нарушений
            if violations:
                st.markdown("### 🔍 Зафиксированные нарушения и доказательная база")
                
                for idx, v in enumerate(violations, 1):
                    v_type = v.get("type", "unknown")
                    timecode = v.get("timecode_sec", 0.0)
                    desc = v.get("description", "Описание отсутствует")
                    evidence_frames = v.get("evidence_frames", [])
                    legal = get_legal_info(v_type)
                    
                    with st.expander(f"Нарушение #{idx}: {desc} — ⏱️ {format_seconds(timecode)}", expanded=True):
                        col_v1, col_v2 = st.columns([1, 1])
                        
                        with col_v1:
                            st.markdown(f"**Тип нарушения:** `{v_type}`")
                            st.markdown(f"**Таймкод события:** `{format_seconds(timecode)}`")
                            st.markdown(f"**Детали инцидента:** {desc}")
                            
                            # Нормативная ссылка на Закон РК (adilet.zan.kz)
                            st.markdown("#### ⚖️ Нормативное основание РК:")
                            st.markdown(
                                f"""
                                <div class="legal-tag">
                                    <strong>📜 Нормативный акт:</strong> {legal['law']}<br>
                                    <strong>📖 Статья/пункт:</strong> {legal['clause']}<br>
                                    <strong>🔗 Источник:</strong> <a href="{legal['url']}" target="_blank" style="color: #0078D4; font-weight: bold;">Перейти в ИС «Әділет» (adilet.zan.kz) ↗</a>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        
                        with col_v2:
                            st.markdown("#### 📸 Кадры-доказательства (Evidence):")
                            if evidence_frames:
                                frame_cols = st.columns(min(len(evidence_frames), 2) or 1)
                                for f_idx, frame in enumerate(evidence_frames):
                                    f_ts = frame.get("timestamp_sec", timecode)
                                    b64_str = frame.get("image_base64", "")
                                    
                                    target_col = frame_cols[f_idx % len(frame_cols)]
                                    with target_col:
                                        if b64_str:
                                            try:
                                                img_bytes = base64.b64decode(b64_str)
                                                st.image(
                                                    img_bytes,
                                                    caption=f"Кадр-доказательство: {f_ts:.1f} сек",
                                                    use_container_width=True,
                                                )
                                            except Exception as img_err:
                                                st.error(f"Ошибка декодирования base64: {img_err}")
                                        else:
                                            st.caption("Кадр не прикреплен")
                            else:
                                st.info("Прямые фотодоказательства отсутствуют")


# ─────────────────────────────────────────────────────────────────────
# TAB 2: Просмотр по Job ID
# ─────────────────────────────────────────────────────────────────────
with tab_history:
    st.markdown("### 🔎 Поиск результата по Job ID")
    search_job_id = st.text_input("Введите идентификатор задачи (UUID / Hex):", placeholder="напр. 3a1f9e8...")
    
    if st.button("Запросить статус"):
        if not search_job_id.strip():
            st.warning("Пожалуйста, укажите Job ID")
        else:
            headers = {"X-API-Key": api_key}
            check_url = f"{api_base_url.rstrip('/')}/api/v1/jobs/{search_job_id.strip()}"
            try:
                r = requests.get(check_url, headers=headers, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    st.json(data)
                elif r.status_code == 404:
                    st.error("Задача с таким Job ID не найдена в базе данных.")
                else:
                    st.error(f"Ошибка сервера: {r.status_code} — {r.text}")
            except Exception as e:
                st.error(f"Не удалось отправить запрос: {e}")
