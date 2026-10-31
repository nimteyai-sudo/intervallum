# Intervallum

Адаптивный тренажёр по физике для подготовки к ЕГЭ. Репозиторий содержит ML-ядро, реальный банк заданий, reference-API и Docker-запуск.

## Что уже внутри

- **1 145 заданий ЕГЭ по физике** в `content/fipi_physics_2026.json`.
- **471 задание с рисунком/графиком** — страницы PDF в `content/media/ege_2026/`.
- **FSRS-6** — интервальное повторение и прогноз забывания.
- **Conjunctive 3PL IRT** — оценка навыка по нескольким Knowledge Units и распределение вины за ошибку.
- **Адаптивный выбор заданий** — ZPD, повторения, ценность для ЕГЭ, разнообразие, назначения учителя, квота повторений.
- **Прогноз готовности к ЕГЭ** — ожидаемый балл и 80% интервал.
- **Проверка ответов** — числовые, multi-select, коды `11/22/333`, измерения `450,012,5`.
- **Симулятор ученика** — знание, забывание, подсказки, угадывание, усталость, мотивация, dropout.
- **FastAPI** — `select`, `attempt`, `grade`, `readiness`.
- **Docker** — API поднимается одной командой.

## Быстрый старт без Docker

```bash
cd intervallum
python3 -m venv .venv
. .venv/bin/activate

pip install -e ".[dev,api]"

pytest -q
python -m intervallum_ml.cli --users 20 --kus 16 --tasks 80 --attempts 20
```

Запуск API:

```bash
uvicorn intervallum_ml.api:app --reload
```

Открыть:

```text
http://127.0.0.1:8000/docs
```

API автоматически загружает реальный банк заданий из `content/fipi_physics_2026.json`.

## Запуск через Docker

```bash
docker compose up --build
```

API будет доступен по адресу:

```text
http://127.0.0.1:8000/docs
```

Проверка:

```bash
curl http://127.0.0.1:8000/health
```

Ожидаемый ответ:

```json
{"status":"ok","knowledge_units":26,"tasks":1145}
```

Остановить:

```bash
docker compose down
```

## Основные команды

| Команда | Что делает |
|---|---|
| `pytest -q` | Все тесты |
| `python -m intervallum_ml.cli` | Демо: синтетические данные, выбор задания, калибровка, прогноз |
| `python scripts/audit.py` | Аудит инвариантов рекомендаций |
| `python scripts/stress_test.py` | Массовый стресс-тест на реальном банке |
| `python scripts/policy_benchmark.py` | Сравнение adaptive vs random |
| `python scripts/simulate_users.py` | Симуляция учеников на реальном банке |
| `python scripts/train.py` | Калибровка по логам попыток |
| `python scripts/import_fipi.py` | Импорт новых PDF в банк заданий |

## Как пользоваться API

### 1. Получить следующее задание

```bash
curl -X POST http://127.0.0.1:8000/v1/select \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo-user",
    "session_position": 0,
    "session_length": 10,
    "days_to_exam": 40,
    "seed": 42
  }'
```

В ответе будет:

- `task` — задание;
- `score` — из чего сложился выбор;
- `propensity` — вероятность выбора для offline-оценки;
- `ranked` — остальные кандидаты.

### 2. Проверить ответ

```bash
curl -X POST http://127.0.0.1:8000/v1/grade \
  -H "Content-Type: application/json" \
  -d '{"task_id": 100000, "answer": "3"}'
```

### 3. Отправить попытку

```bash
curl -X POST http://127.0.0.1:8000/v1/attempt \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo-user",
    "task_id": 100000,
    "answer": "3",
    "duration_ms": 60000,
    "self_grade": 3
  }'
```

Если передать `answer`, сервер сам посчитает `raw_score`. Если передать `raw_score`, будет использовано ваше значение.

### 4. Получить прогноз ЕГЭ

```bash
curl -X POST http://127.0.0.1:8000/v1/readiness \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo-user",
    "exam_date": "2027-06-01T00:00:00Z",
    "blueprint": [
      {"position": 1, "max_points": 1, "ku_ids": [101], "b_exam": 0, "a_exam": 1}
    ]
  }'
```

## Как устроено

### 1. Память — FSRS

Для каждой пары «ученик × Knowledge Unit» хранится:

- `stability` — сколько дней до падения вероятности вспоминания до 90%;
- `difficulty` — сложность;
- `retrievability` — вероятность вспомнить сейчас;
- `due_at` — когда повторить.

### 2. Навык — conjunctive 3PL IRT

Задание требует несколько Knowledge Units. Слабейшее звено тянет результат вниз:

```text
theta_task = softmin(theta_eff)
p_star = sigmoid(a * (theta_task - b))
p_correct = c + (1 - c) * p_star
```

При ошибке штраф распределяется по ответственности: слабый навык получает больше.

### 3. Выбор задания — педагогический scorer

Учитываются:

- ZPD: целевая вероятность успеха 0.72–0.85;
- срочность повторения;
- ценность темы для ЕГЭ;
- разнообразие механик;
- назначения учителя;
- квота повторений;
- контролируемое исследование.

### 4. Прогноз ЕГЭ

Считается по 26 позициям КИМ: ожидаемый первичный балл, дисперсия, 80% интервал и перевод в тестовый балл.

### 5. Калибровка

Офлайн-пайплайн подбирает:

- сложность заданий `b`;
- дискриминацию `a`;
- жёсткость softmin `tau`;
- влияние забывания `gamma`.

## Контент

Файл:

```text
content/fipi_physics_2026.json
```

Структура:

- `meta` — источник и статистика;
- `knowledge_units` — разделы и позиции ЕГЭ;
- `tasks` — задания, ответы, тип ответа, медиа, теги.

Добавить новые PDF:

```bash
python scripts/import_fipi.py \
  --pdf-dir "/путь/к/PDF" \
  --out content/fipi_physics_2026.json \
  --media-dir content/media/ege_2026
```

## Симуляция учеников

```bash
python scripts/simulate_users.py \
  --students 12 \
  --sessions 4 \
  --tasks 10 \
  --policies random adaptive easy hard topic_order
```

Симулятор моделирует знание, забывание, подсказки, угадывание, время ответа, усталость, мотивацию, тревожность и dropout.

Пример результата:

| Политика | Истинный прирост | Средний true p |
|---|---:|---:|
| random | 0.0096 | 0.544 |
| adaptive | 0.0161 | 0.730 |
| easy | 0.0143 | 0.846 |
| hard | 0.0025 | 0.342 |

## Структура репозитория

```text
src/intervallum_ml/
  types.py          типы и контракты
  config.py         все гиперпараметры
  telemetry.py      очистка сигнала, y_eff и g_sync
  memory.py         FSRS
  irt.py            conjunctive 3PL IRT
  selector.py       выбор задания
  readiness.py      прогноз ЕГЭ
  grading.py        проверка ответов
  content.py        загрузка и валидация контента
  calibration.py    калибровка заданий
  evaluation.py     метрики и off-policy оценка
  simulation.py     синтетический мир
  user_sim.py       поведенческий симулятор ученика
  api.py            FastAPI
  model_store.py    версии моделей
  state_store.py    сериализация состояния

content/            реальный банк заданий и медиа
scripts/            импорт, стресс-тесты, симуляция, калибровка
tests/              тесты
docs/               архитектура, математика, API, отчёт проверки
```

## Проверка

```text
83 теста
90% покрытия
mypy — чисто
ruff — чисто
```

Стресс-тест:

```text
2 000 выборов
200 сессий
2 000 попыток
200 проверок квоты
0 ошибок
```

## Источник заданий

Задания взяты из открытого банка ФИПИ.

## Лицензия

MIT.
