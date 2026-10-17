# Datathon 26 — ForteBank Cash Discipline AI

Локальный ML/CV-модуль участника №2: MOV → реальные PTS → sampling → движение →
визуальные наблюдения Ollama → CandidateEvent с путями к JPEG-доказательствам.
Юридические решения принимает отдельный Rule Engine. Здесь нет распознавания лиц,
идентификации личности, облачных AI API, backend, базы данных и UI.

Лучше вернуть `unknown`, чем придумать нарушение. Даже успешный технический запуск
не подтверждает точность модели или соблюдение банковских правил.

## Запуск в Windows PowerShell

Python 3.11+; локально проверен Python 3.14.6. Нужен работающий Ollama с установленной
Vision-моделью. Большие модели автоматически не скачиваются.

```powershell
py -3.14 -m venv .venv  # либо другой установленный Python >=3.11
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# Проверка сервиса и capability vision; без отправки кадров:
Invoke-RestMethod http://127.0.0.1:11434/api/tags
python -c "from app.vision.vlm import OllamaVisionClient,VLMConfig; OllamaVisionClient(VLMConfig()).check_ready(); print('vision ready')"

# Только если qwen2.5vl:3b отсутствует; загрузка около нескольких GB:
ollama pull qwen2.5vl:3b

# Из корня репозитория; путь ниже замените своим локальным MOV:
python -m scripts.test_vision "D:\local-videos\sample.mov" --config configs/vision.example.json

# CV без VLM: кандидаты остаются unknown, статус partial при их наличии:
python -m scripts.test_vision "D:\local-videos\sample.mov" --no-vlm

# Папка с 5–10 локальными JPEG, последовательные запросы:
python -m scripts.benchmark_vlm "D:\local-test-frames" --repeats 5
# Дополнительно выгрузить выбранную модель для измерения загрузки:
python -m scripts.benchmark_vlm "D:\local-test-frames" --repeats 5 --cold-start

python -m pytest -q
ruff check .
ruff format --check .
```

Если PowerShell запрещает активацию, используйте `.\.venv\Scripts\python.exe`
напрямую; менять системную execution policy не требуется. На машинах без `py`
подойдёт `python -m venv .venv` при подходящей версии Python.

CLI возвращает exit code 0 только для `complete`, 2 для неполной/ошибочной обработки.
Отчёты и JPEG остаются локально в игнорируемом `artifacts/`. CLI запускайте как
`python -m scripts.…` из корня репозитория; это обеспечивает импорт пакетов и
корректный Windows spawn.

## Интеграция backend

```python
from pathlib import Path
from app.vision.pipeline import VisionConfig, analyze_video


def process_job(local_video: Path):
    config = VisionConfig(
        output_dir=Path("artifacts/vision"),
        roi_path=Path("configs/roi.json"),
        camera_profile=None,  # задать только после настоящей калибровки
    )
    result = analyze_video(local_video, config)
    return result


if __name__ == "__main__":  # обязательно в самостоятельном Windows-скрипте
    result = process_job(Path("local-video.mov"))
    print(result.status, result.metrics)
```

Это синхронная функция для worker участника №1. Вызывайте её вне event loop API;
worker должен разрешать создание дочернего процесса (не daemon multiprocessing).
Не запускайте задания при импорте модуля. Процесс-supervisor ограничивает wall-clock
время; при зависании декодера/SDK прекращает дочерний процесс и возвращает последний
checkpoint. Завершение процесса может добавить до примерно 2 секунд к лимиту.
Это механизм остановки, а не обещание закончить анализ любого видео за 300 секунд.
Закрытие клиента не гарантирует мгновенную остановку внутренних вычислений Ollama.

Backend проверяет `result.status` и `diagnostics`, хранит `evidence_frames` и передаёт
`result.events` в независимый Rule Engine. Сериализация: `model_dump(mode="json")`
или `model_dump_json()`. Для итоговых evidence timestamps сопоставьте `frame_paths`
с `result.evidence_frames[*].path`: там есть относительное время, исходные PTS,
time_base и source_time_sec. Backend отвечает за доступ к файлам, срок хранения и
удаление каталога `result.output_dir` после завершения работы всех потребителей.

## Конфигурация и интерпретация

- Начальный профиль: 2 FPS, 300 секунд, максимум 1200 sampled frames, 24 кандидата,
  6 VLM-попыток, 1 изображение на запрос, `num_ctx=2048`, `num_predict=200`, temperature 0.
- Одно изображение не доказывает передачу объекта: `object_transfer` и тип
  переданного объекта принудительно остаются `unknown` в режиме одного кадра.
- Для temporal-проверки задайте `VLMConfig(max_images=3, num_ctx=4096)` через поле
  `VisionConfig.vlm`. На данной установке 3 кадра превысили контекст 2048; с 4096
  реальный запрос прошёл. Для другой сцены/разрешения этого может быть недостаточно.
  Переполнение возвращается как `vlm_context_exceeded`, без скрытого повтора.
- VLM получает уменьшенные JPEG, при активной ROI — crop с 10% контекста. Исходное
  доказательство — sampled full-frame JPEG с точным timestamp; видео не изменяется.
- `configs/roi.json` — **демонстрационные неоткалиброванные координаты**. Активны
  только `calibrated=true`, `demonstration=false` и совпадающий camera_profile.
  До калибровки движение считается по всему кадру; `roi_id=None`, ROI-классификация
  недоступна. `safe_area` добавляется только если сейф действительно виден.
- Motion score — доля изменившихся пикселей после фильтрации; score YOLO — выход
  детектора. Ни один не является вероятностью нарушения. Тени, свет и отражения
  тоже вызывают движение. `person` и `cell phone` не означают нарушение.
- `ReflectionGuard` помечает person/glass overlap как возможное отражение и убирает
  утверждение physical presence. Без геометрии/детекций присутствие также unknown.
- JSON VLM строго проверяется. Ошибки сервиса, отсутствие модели, invalid JSON,
  недопустимые enum, truncation и бюджет не превращаются в успешное отсутствие событий.
- В пределах машины lock сериализует тяжёлые VLM-запросы разных потоков/процессов.
  Поддерживаются только loopback HTTP, локальные теги моделей, без proxy/redirect.
- `.env.example` — ориентир для backend. Модуль **не читает .env автоматически**;
  используйте VisionConfig или JSON из `configs/vision.example.json`.

`complete` означает, что проход sampling завершён и все сохранённые кандидаты
обработаны в пределах бюджета. Это не анализ каждого исходного кадра VLM и не
гарантия обнаружения коротких действий. `partial`, `failed`, `timed_out` требуют
явной обработки backend. `decoded_fraction` измеряет проход декодера; временные
интервалы успешных VLM-вызовов перечислены отдельно в `coverage.vlm_intervals`.
Отсутствие движения может дать complete без VLM-вызова.

## Опциональный YOLO

По умолчанию отключён. При необходимости установите `pip install -e ".[yolo]"`,
положите отдельно полученные веса в `models/yolo11n.pt` и включите `yolo_enabled`.
Загрузка ленивая, при недоступном GPU/RuntimeError есть CPU fallback. Автоскачивания
весов нет. Стандартные COCO-классы не дают распознавания купюр. Настоящие веса YOLO
в этой работе не запускались; установка не нужна для P0.

## Тесты и локальные интеграционные проверки

Обычные pytest-тесты генерируют небольшие MOV/JPEG, используют mock Ollama и не
требуют GPU, модели или реальных банковских записей. CI проверяет Windows/Linux и
Python 3.11/3.14. Зависимости имеют совместимые диапазоны; полный lockfile не поставляется.

Реальная модель и явно указанные локальные файлы включаются отдельно:

```powershell
$env:VISION_INTEGRATION_JPEG_DIR="D:\local-test-frames"
$env:VISION_INTEGRATION_VIDEO="D:\local-videos\sample.mov"
python -m pytest -m integration -q
Remove-Item Env:VISION_INTEGRATION_JPEG_DIR
Remove-Item Env:VISION_INTEGRATION_VIDEO
```

Без переменных эти два теста пропускаются. Проверки модели с synthetic controls
и MOV выполнены локально; результаты и ограничения: [benchmark](docs/benchmark.md).
Не переносите их показатели на качество банковских сцен.

## Структура и зоны ответственности

```text
app/
  schemas.py               # CandidateEvent / VerifiedEvent / tri-state
  vision/
    video.py roi.py motion.py detector.py reflection.py vlm.py
    config.py result.py pipeline.py
  api/ core/ ui/           # только точки расширения участника №1
  rules/                   # только точка расширения участника №3
configs/                   # пример профиля и неактивная demo ROI
scripts/                   # smoke test и benchmark
tests/                     # synthetic unit + opt-in integration
docs/                      # архитектура, команда, фактические измерения
.github/workflows/         # CI
```

Согласование контракта, включая полное соответствие отдельному ТЗ v2.0, которое
не было предоставлено целиком, описано в [архитектуре](docs/architecture.md).
Роли аккаунтов не назначены по догадке: [TEAM](docs/TEAM.md).
Правила для агентов: [AGENTS.md](AGENTS.md).

Банковские видео, кадры, datasets, веса, токены и локальные отчёты не коммитить и
не прикладывать к Issues/PR. Сохраняются технические метаданные, без raw Base64,
текста документов или идентифицирующих описаний. Нужна дальнейшая проверка на
размеченных локальных банковских сценах, особенно отражениях и коротких действиях.
