# Мониторинг (Prometheus + Grafana) и Langfuse

## Обзор

Система мониторинга включает:

- **LLM Performance метрики**: TTFT (prefill), ITL (decode), токены, стоимость
- **Бизнес-аналитика**: города, бюджеты, пассажиры, длительность поездок
- **Системные метрики**: CPU, RSS, латентность, коды ответов
- **Трейсинг**: Langfuse (опционально)

## Prometheus / Grafana

### Запуск

```bash
# UI + полный стек мониторинга
docker compose --profile monitoring up --build

# Только UI
docker compose up --build
```

### Endpoints

| Сервис | URL | Описание |
|--------|-----|----------|
| Streamlit UI | `http://localhost:8501` | Веб-интерфейс приложения |
| Prometheus metrics | `http://localhost:9090/metrics` | Метрики приложения (pull) |
| Prometheus UI | `http://localhost:9091` | Управление и запросы Prometheus |
| Grafana | `http://localhost:3000` | Дашборды (анонимный вход, PoC) |

### Дашборды Grafana

#### 1. Travel Agent — LLM & planning (`travel-monitoring.json`)

Базовый дашборд с техническими метриками:

- Planning requests rate (по outcome и HTTP-коду)
- Planning latency p50/p95
- LLM TTFT prefill vs decode phase (p95)
- LLM inter-token latency (ITL)
- Токены/сек (input + output)
- Стоимость LLM (USD/s)
- CPU и RSS процесса

#### 2. Travel Agent — Analytics Dashboard (`travel-analytics.json`)

Расширенный аналитический дашборд:

**📊 KPI Cards:**
- Всего запросов
- Средняя латентность
- Общая стоимость LLM (USD)
- Среднее кол-во токенов (output)
- Средний бюджет
- Success Rate (%)

**🌍 Travel Analytics:**
- Топ городов вылета (bar gauge)
- Топ городов назначения (bar gauge)
- Популярные маршруты (table)

**💰 Budget & Cost:**
- Распределение бюджетов по категориям (pie/donut)
- Итоговая стоимость поездок — гистограмма
- Бюджет vs Итоговая стоимость (timeseries)

**👥 Passengers & Trip Details:**
- Распределение по кол-ву пассажиров
- Длительность поездок (дни)
- Валюты запросов (pie)
- Тип поиска: flights/hotels/both (pie)

**⚡ LLM Performance:**
- TTFT prefill p50/p95
- ITL inter-token latency p95
- Токены/сек (input vs output)
- Стоимость LLM по stage
- Скорость генерации (токенов/сек)
- Кол-во токенов в ответе (гистограмма)

**🔄 System & Reliability:**
- Planning requests rate
- Planning latency p50/p95
- Guardrail Retries распределение
- CPU и RSS

## Метрики Prometheus

### Счётчики (Counters)

| Метрика | Описание | Labels |
|---------|----------|--------|
| `travel_planning_requests_total` | Запуски графа планирования | `outcome`, `http_code` |
| `travel_llm_input_tokens_total` | Входные токены LLM | `stage` |
| `travel_llm_output_tokens_total` | Выходные токены LLM | `stage` |
| `travel_llm_cost_usd_total` | Стоимость LLM (USD) | `stage` |
| `travel_origin_city_total` | Города вылета | `city` |
| `travel_destination_city_total` | Города назначения | `city` |
| `travel_outcome_with_budget_total` | Исходы по бюджету | `outcome`, `budget_range` |
| `travel_search_scope_total` | Тип поиска | `scope` (flights/hotels/both) |

### Гистограммы (Histograms)

| Метрика | Описание | Buckets |
|---------|----------|---------|
| `travel_planning_duration_seconds` | Полное время выполнения | до 120s |
| `travel_llm_wall_seconds` | Латентность LLM-вызова | до 120s |
| `travel_llm_prefill_seconds` | TTFT — время до первого токена | до 120s |
| `travel_llm_decode_phase_seconds` | Фаза decode | до 120s |
| `travel_llm_inter_token_latency_seconds` | ITL (задержка/токен) | микросек-сек |
| `travel_trip_passengers` | Кол-во пассажиров | 1-12 |
| `travel_trip_budget` | Бюджет поездки | 100-20000 |
| `travel_trip_duration_days` | Длительность (дни) | 1-60 |
| `travel_guardrail_retries` | Retry-попытки | 0-3 |
| `travel_total_trip_cost_usd` | Итоговая стоимость (USD) | 100-30000 |
| `travel_response_token_count` | Токены в ответе | 100-10000 |
| `travel_generation_tokens_per_sec` | Скорость генерации | 5-200 tps |

### Gauge (через _ProcessResourceCollector)

| Метрика | Описание |
|---------|----------|
| `travel_host_cpu_percent` | Загрузка CPU (%) |
| `travel_process_resident_memory_bytes` | RSS процесса (bytes) |

## Переменные окружения

| Переменная | Описание | Default |
|------------|----------|---------|
| `PROMETHEUS_METRICS_PORT` | Порт для /metrics endpoint | `9090` |
| `PROMETHEUS_ENABLE` | Включить метрики (1/true/yes) | - |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key | - |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key | - |
| `LANGFUSE_HOST` | Langfuse API URL | `https://cloud.langfuse.com` |
| `LLM_DEFAULT_INPUT_USD_PER_1K` | Цена input токенов (USD/1k) | `0.0` |
| `LLM_DEFAULT_OUTPUT_USD_PER_1K` | Цена output токенов (USD/1k) | `0.0` |

## Langfuse (расширенная трассировка)

### Активация

1. Создайте проект в [Langfuse Cloud](https://cloud.langfuse.com) или поднимите self-hosted
2. Задайте переменные окружения для сервиса `ui`:
   - `LANGFUSE_PUBLIC_KEY`
   - `LANGFUSE_SECRET_KEY`
   - `LANGFUSE_HOST` — базовый URL API (для EU Cloud обычно `https://cloud.langfuse.com`)

При отсутствии ключей интеграция не активируется.

### Возможности

#### 1. Метаданные трассировки

Каждый запрос планирования автоматически получает метаданные:
- `user_id` — ID пользователя (если задан)
- `session_id` / `thread_id` — идентификатор сессии
- `tags` — теги для фильтрации (`["travel_planning"]`)
- `metadata` — превью запроса, города вылета/назначения

В Langfuse UI можно фильтровать traces по этим полям.

#### 2. Score-метрики

После каждого успешного запроса автоматически выставляются оценки:

| Score | Описание | Диапазон |
|-------|----------|----------|
| `budget_compliance` | Насколько уложились в бюджет | 0-1 (1 = уложились) |
| `latency_rating` | Оценка скорости ответа | 1.0 (<30s), 0.5 (<60s), 0.2 (>60s) |
| `success_score` | Успех без retry | 1.0 (0 retry), 0.7 (1-2 retry), 0.4 (3+ retry) |

#### 3. Dataset-эксперименты

Все пары запрос/ответ сохраняются в Langfuse Dataset `travel_planning` для:
- A/B тестирования разных моделей
- Оценки качества генерации
- Анализа паттернов запросов

Поля каждого item:
```json
{
  "input": {
    "user_text": "Хочу в Париж на неделю...",
    "extracted_query": {"origin_city": "Москва", ...}
  },
  "expected_output": {
    "final_markdown": "## План поездки...",
    "total_cost_usd": 1500.0,
    "outcome": "ok"
  },
  "metadata": {
    "duration_sec": 45.2,
    "passengers": 2,
    "budget": 2000,
    "destination": "Париж"
  }
}
```

#### 4. Custom tracing spans

Каждый узел LangGraph-графа логируется как отдельный span:

| Span | Что трейсится |
|------|---------------|
| `extract_intent` | Извлечение TripQuery: города, бюджет, пассажиры |
| `validate_constraints` | Валидация бюджета: valid=True/False, days, budget |
| `fetch_data` | Поиск данных: scope, количество найденных рейсов/отелей/достопримечательностей |
| `generate_itinerary` | Генерация маршрута: destination, days, длина ответа |
| `budget_guardrail` | Проверка бюджета: passed, total_estimated, flight/hotel parts |

В Langfuse UI видна иерархия: **Trace → Spans → Generations**

#### 5. Ссылка на Langfuse trace в Grafana

Результат запроса содержит `langfuse_trace_url` — прямую ссылку на trace в Langfuse.
Это позволяет перейти из приложения прямо в детализацию трейса.

### API Langfuse SDK

Проект предоставляет следующие функции (импорт из `backend.langfuse_tracing`):

```python
# Создание кореневого trace
trace_id = create_langfuse_trace_for_planning(
    user_id="user123",
    session_id="session_abc",
    input_text="Хочу в Париж..."
)

# Custom span
with langfuse_span("my_operation", metadata={"key": "value"}) as span:
    # ... код ...
    span["result"] = {"status": "ok"}

# Score-метрики
score_travel_quality(
    trace_id=trace_id,
    budget=2000,
    total_cost_usd=1500,
    retry_count=0,
    duration_sec=45.0
)

# Сохранение в dataset
save_to_langfuse_dataset(
    user_input="Хочу в Париж...",
    result={"final_markdown": "..."},
    query={"destination_city": "Париж"},
    total_cost_usd=1500,
    duration_sec=45.0
)

# Получение URL trace
url = get_langfuse_trace_url(trace_id)
# → "https://cloud.langfuse.com/project/.../traces/..."
```

### Структура Trace в Langfuse

```
travel_planning (Trace)
├── extract_intent (Span)
│   └── trip_extraction (Generation) — LLM вызов извлечения TripQuery
├── validate_constraints (Span)
├── fetch_data (Span)
│   └── city_attractions (Generation) — LLM вызов достопримечательностей
├── generate_itinerary (Span)
│   └── travel_itinerary (Generation) — LLM вызов генерации маршрута
└── budget_guardrail (Span)

Scores:
├── budget_compliance: 0.85
├── latency_rating: 1.0
└── success_score: 1.0
```

## Безопасность

> ⚠️ Grafana настроена с анонимным доступом (`GF_AUTH_ANONYMOUS_ENABLED=true`) только для PoC.
> Не выставляйте это в интернет без reverse proxy и ограничений!
