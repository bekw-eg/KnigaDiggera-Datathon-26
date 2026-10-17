# Desk ROI и двухпроходный temporal-анализ

## Интерфейс и совместимость

Публичный вызов прежний: `analyze_video(Path, VisionConfig)` из `app.vision.pipeline`.
`temporal.enabled=false` по умолчанию сохраняет legacy merge/single-frame поведение.
Новый профиль — `configs/vision.temporal.example.json`. Поля `app/schemas.py` не менялись.
Расширен только ML-конверт `VisionAnalysisResult`: новые метрики, coverage и
`event_replay` с defaults, поэтому старый JSON этого конверта продолжает читаться.
Клиентам со строгой старой схемой конверта следует обновить её; Rule Engine получает
прежний `CandidateEvent`. Дополнительные temporal-признаки используют существующий
`observations: dict[str, Observation]`, без новых обязательных полей контракта.

```text
MOV → PyAV / каждый PTS валидируется → overview 2 FPS
 → Desk mask ∩ ROI masks → полная временная последовательность motion scores
 → hysteresis / паузы / локальные пики → короткие proposals
 → пространственно-временная дедупликация / приоритеты / лимит кандидатов
 → повторный последовательный decode → 8 FPS только внутри выбранных окон
 → повторная сегментация → измеренные quiet-before / peak / quiet-after
 → полные JPEG + роли / PTS / crop mapping → локальный Qwen → ReflectionGuard
 → CandidateEvent[] + EventReplay[] → независимый Rule Engine / UI участника №1
```

`desk.py` измеряет движение, `temporal.py` содержит чистые функции сегментации и
дедупликации, `evidence.py` — выбор/валидацию evidence и replay, `temporal_pipeline.py`
— оркестрацию. `video.py`, local-only клиент и межпроцессный inference lock общие.
Новых зависимостей, backend jobs, UI, хранения в БД или юридических правил нет.

## Desk и виртуальная граница

Используется прежний JSON `ROIConfig` из редактора PR #2. Нужны вручную проверенные
`calibrated=true`, `demonstration=false` и точно совпадающий `camera_profile`.
Нормализованные полигоны `desk`, `cash_area`, `transfer_zone`, `glass_partition`
сохраняют формат. Идентификатор Desk настраивается (`temporal.desk_roi_id`).
В temporal-режиме каждая активная зона пересекается с маской Desk. Изменения снаружи
не дают desk-кандидатов; пустые пересечения исключаются. Стекло сохраняет свою
геометрию для ReflectionGuard независимо от motion mask.

Без Desk используются прежние активные ROI, а без них — полный кадр. Выдаются
`desk_roi_unavailable_fallback` и, при отсутствии всех ROI,
`roi_classification_unavailable`. `roi_id=None` допустим. При этом temporal-передача
всегда остаётся `unknown`, даже при положительном ответе модели. Наличие рабочего
fallback не означает калибровку камеры.

`temporal.boundary=null` по умолчанию. Если граница нужна, вручную задайте объект
`enabled`, `camera_profile`, `points` (две разные пары координат 0..1), `band_width`
(ширина окрестности сегмента в нормализованных единицах). Координаты следует брать
с той же откалиброванной сцены; готовые банковские координаты не поставляются.
При несовпадении профиля или отсутствии активной Desk граница игнорируется с
`boundary_calibration_unavailable`.

`boundary_activity=true` означает лишь наличие изменённых пикселей по обе стороны
вблизи конечного сегмента. Это не отслеживание руки/предмета, направление движения,
пересечение объектом, идентичность объектов или подтверждение передачи наличных.

## Сегментация, дедупликация и приоритет

Motion score — доля изменившихся пикселей после порога/контурного фильтра, не
вероятность нарушения. Хранятся и нулевые значения. Эпизод должен достичь
`high_threshold`, продолжаться через `low_threshold`; низкие значения на протяжении
`pause_sec` отделяют эпизоды. Короткие провалы и фрагменты через `merge_gap_sec`
объединяются. Локальные максимумы дают реальные PTS пиков, а pre/post-roll — контекст.
Пороговые значения примера требуют настройки на размеченных локальных сценах.

Полезная длина активности = `max_window_sec - pre_roll_sec - post_roll_sec`.
Непрерывный длинный эпизод даёт ограниченные окна по реальным sampled парам и код
`continuous_activity_undersegmented`; окна имеют `segmentation_insufficient=true`.
Число таких окон не является числом действий. При очень редких PTS окно может
содержать только наблюдаемый endpoint с тем же признаком неопределённости.
После refinement возможна успешная локальная сегментация при реально наблюдённых
паузах; без них сохраняется `temporal_segmentation_insufficient` и `partial`.

Дедупликация требует пересечения активных временных диапазонов (доля не меньше
`dedup_overlap`) и пересечения полигонов. Сравниваются все участники группы:
цепочка пересечений не должна поглотить далёкие эпизоды. Главная ROI выбирается
по `roi_priorities` (словарь ID → целый приоритет, больше — выше), затем по motion
score и времени. Default-приоритеты равны и не зависят от имён зон.
`roi_id` остаётся единственным основным ID в CandidateEvent, все источники —
`event_replay[*].roi_ids`. Одновременные разные движения в перекрывающихся зонах
могут быть объединены: это известный риск предложения, без tracking/re-ID.

## Второй проход и evidence

Каждый проход декодирует последовательно от исходного origin. Это сохраняет
проверку PTS и не вводит непроверенный seek для длинных MOV. Во втором проходе
сохраняются/анализируются только кадры выбранных окон; через пропущенный промежуток
motion state сбрасывается. `refine_fps` настраивается, пример — 8 FPS. VFR остаётся
VFR: используются реальные frame.pts*time_base, никакого frame_index/fps.
Пустые/битые потоки, отсутствующие или не монотонные PTS дают явную ошибку.

После повторной сегментации выбирается during около измеренного пика, before —
последний наблюдённый тихий кадр перед активностью, after — первый тихий после.
Для evidence нужны три разных coarse grayscale fingerprint и возрастающие PTS.
Если пик совпадает с финальным состоянием, выбирается ближайший отличный кадр
внутри той же активности. Позднее постороннее движение не заменяет quiet-after.
Fingerprint проверяет повтор кадров, а не идентичность предметов.
Отсутствующие роли не заполняются. При неполной/недостаточно сегментированной
последовательности VLM получает только during, передача и её тип — `unknown`.
Если `vlm.max_images<3`, это также явный fallback с `temporal_vlm_image_budget`.

JPEG сохраняется целиком в ограниченном разрешении `max_frame_side`, без ROI crop.
Только VLM получает crop с контекстом 0.1. ReplayFrame содержит путь исходного JPEG,
timestamp, source_pts/time_base/source_time_sec, роль, fingerprint, ID и snapshot
полигона crop, контекст и `vlm_max_side`. Crop не заменяет оригинал и не хранится
Base64 в результате. `vlm_frame_paths` показывает фактически отправленные кадры.
Пиксели нужны только для одного текущего кадра; на диске максимум
`max_refined_frames` JPEG (default 768), а не все overview кадры.

## VLM, unknown и технические статусы

Temporal JSON строго требует дополнительные tri-state поля `object_movement`,
`hand_object_interaction`, `position_change_trackable`, `object_occluded`.
Свободные `observations` в temporal JSON должны быть `[]`: компактный ответ лучше
помещается в `num_predict=200` и не содержит идентифицирующих описаний.
Legacy-схема прежняя. Невалидные JSON/enum, лишние поля, truncation и context overflow
дают ошибку, а не исправленный успешный ответ или скрытый retry.

Перекрытие предмета, непрослеживаемое положение, неопределённые hand/object признаки
или отражение запрещают положительное temporal-утверждение о передаче. Белый
прямоугольник не равен банкноте; сходство до/после перекрытия не доказывает
идентичность. ReflectionGuard применяется и к новым ответам. `unknown` означает
недостаточность наблюдений, а не отсутствие события. Три JPEG и temperature 0
не гарантируют правильный вывод или детерминизм.

Для UI сопоставьте `event_replay[*].event_id` с `events[*].event_id`, покажите
`frames` в сохранённом порядке и диагностику. Статусы replay:

| Статус | Технический смысл |
| --- | --- |
| pending | промежуточный checkpoint до завершения анализа |
| verified | VLM вернула валидные наблюдения для полного triplet; юридического подтверждения нет |
| incomplete | нет полного temporal evidence или анализ прерван |
| skipped_budget | VLM не вызвана из-за лимита; наблюдения unknown |
| failed | ошибка модели/JSON/сервиса |

`complete` результата означает завершение выбранного технического плана, не
гарантию нахождения всех действий. `partial` означает ограничение/ошибку хотя бы
одной части; смотрите diagnostics и статусы каждого replay. `timed_out` — общий
deadline, с последним доступным checkpoint; `failed` — отсутствие рабочего анализа.
Успешный overview без движения может завершиться без Ollama и без replay.

## Бюджеты, coverage и benchmark

Default: 300 s wall-clock, 6 VLM-попыток, 24 кандидата, 1200 overview кадров,
100000 decoded кадров суммарно в двух проходах, 768 сохранённых refinement кадров,
256 proposals на этап сегментации. При превышении выдаются `sample_frame_limit`,
`decode_frame_limit`, `refined_frame_limit`, `proposal_limit`, `candidate_limit`,
`vlm_call_limit` и неполный статус. Кандидаты приоритизируются до VLM; пропущенные
не помечаются проверенными. Deadline включает запуск процесса; default резерв
остановки worker — 2 s. Supervisor завершает зависший worker, а worker проверяет
deadline даже между несэмплируемыми decoded кадрами. Остановка клиента не означает
мгновенную отмену уже отправленной операции внутри Ollama. Существующий OS/thread
inference lock остаётся общим для pipeline и benchmark.

`decoded_frames` = first_pass_decoded_frames + second_pass_decoded_frames.
`sampled_frames` — overview, `second_pass_frames` — сохранённые refined кадры.
`candidates_before_dedup` — все overview proposals до cap, `candidates_after_dedup`
— дедуплицированные сохранённые proposals до max_candidates, `refined_candidates`
— proposals после refinement до финального cap. `candidates_total` сохраняет смысл
числа обзорных кандидатов до cap. При `proposal_limit` after_dedup не описывает
отброшенные proposals. Legacy считает before/after одинаково, refinement = 0.

`coverage.decoded_*` относится только к overview, `refined_intervals` — полностью
декодированные выбранные окна, `vlm_intervals` — диапазоны предложений с валидным
ответом VLM, включая single-frame fallback. Последнее не означает просмотра каждого
момента интервала; полноту triplet проверяйте отдельно. Покрытие не равно доле
правильно обнаруженных действий.

PowerShell (в активированной `.venv`, пути и профиль заменить своими):

```powershell
python -m scripts.test_vision "D:\local-videos\X-1.mov" --config configs/vision.temporal.example.json --roi configs/camera.local.json --camera-profile camera-01 --report artifacts/temporal-report.json
python -m scripts.benchmark_temporal --video "D:\local-videos\X-1.mov" --roi configs/camera.local.json --camera-profile camera-01 --report artifacts/local-temporal-benchmark-report.json
python -m scripts.benchmark_temporal --synthetic --no-vlm
python -m scripts.benchmark_temporal --synthetic --report artifacts/temporal-ollama-benchmark-report.json
python -m pytest -q
ruff check .
ruff format --check .
git diff --check
```

Benchmark последовательно запускает legacy single-frame/2048 и temporal triplet/4096
на одном файле, возвращает counts, интервалы, времена, coverage, ошибки и sampled
device-wide VRAM. CLI сравнения возвращает 1 при failed/timed_out, а ожидаемый
CV-only partial записывает в отчёт без аварийного exit. Технический benchmark JSON
не содержит путей к исходному видео, JPEG или model free text. Полные локальные
evidence остаются в игнорируемом output_dir; backend владеет retention/доступом.

Обзор 2 FPS может пропустить короткое действие целиком; второй проход не восстанавливает
то, для чего не было выбрано окно. Генератор содержит видимые движения 1.5–2.25 и
4.5–5.25 s и импульс 3.40–3.45 s между samples. Тест проверяет пропуск последнего.
Это демонстрация sampling gap, не precision/recall передачи денег.
Реальные camera calibration, bank MOV, длинные записи, YOLO и качество на размеченных
сценах требуют отдельной локальной проверки. Освещение, тени, компрессия, мелкие
предметы, перекрытия и отражения остаются источниками false positives/пропусков.
