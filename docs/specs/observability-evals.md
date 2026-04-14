# Наблюдаемость и оценки качества

## LLM Observability (`backend/llm_observability.py`)

Для всех LLM-вызовов через `stream_plain_text` / `stream_structured_output` автоматически собираются:

| Метрика | Описание | Формула |
|---------|----------|---------|
| **TTFT** (`ttft_sec`) | Время до первого токена (prefill) | `t_first_signal − t_start` |
| **Decode** (`decode_sec`) | Фаза генерации после первого токена | `t_end − t_first_signal` |
| **ITL/TPOT** (`itl_sec`) | Средняя задержка на выходной токен | `decode_sec / max(1, completion_tokens)` |
| **Токены** | Input, output, total | Из `response_metadata.token_usage` |
| **Стоимость** (`cost_usd`) | Оценка стоимости | Из metadata шлюза или env: `LLM_DEFAULT_INPUT_USD_PER_1K`, `LLM_DEFAULT_OUTPUT_USD_PER_1K` |

Метрики пишутся в лог (`INFO`), попадают в ответы и в состояние графа (`itinerary_llm_metrics`).

### Stages LLM-вызовов

| Stage | Где вызывается | Модель (default) |
|-------|----------------|------------------|
| `trip_extraction` | Извлечение TripQuery | gpt-4o-mini → gpt-4o |
| `city_attractions` | Подбор достопримечательностей | gpt-4o-mini |
| `travel_itinerary` | Генерация маршрута Markdown | gpt-4o-mini |

---

## Prometheus метрики (`backend/prometheus_metrics.py`)

Экспорт на `/metrics` (порт `PROMETHEUS_METRICS_PORT`, default 9090).

### Counters

| Метрика | Labels | Описание |
|---------|--------|----------|
| `travel_planning_requests_total` | `outcome`, `http_code` | Запуски графа |
| `travel_llm_input_tokens_total` | `stage` | Входные токены |
| `travel_llm_output_tokens_total` | `stage` | Выходные токены |
| `travel_llm_cost_usd_total` | `stage` | Стоимость LLM (USD) |
| `travel_origin_city_total` | `city` | Города вылета |
| `travel_destination_city_total` | `city` | Города назначения |
| `travel_outcome_with_budget_total` | `outcome`, `budget_range` | Исходы по бюджету |
| `travel_search_scope_total` | `scope` | Тип поиска (flights/hotels/both) |

### Histograms

| Метрика | Buckets | Описание |
|---------|---------|----------|
| `travel_planning_duration_seconds` | до 120s | Полное время выполнения |
| `travel_llm_wall_seconds` | до 120s | Латентность LLM-вызова |
| `travel_llm_prefill_seconds` | до 120s | TTFT (prefill) |
| `travel_llm_decode_phase_seconds` | до 120s | Фаза decode |
| `travel_llm_inter_token_latency_seconds` | микросек-сек | ITL |
| `travel_trip_passengers` | 1-12 | Кол-во пассажиров |
| `travel_trip_budget` | 100-20000 | Бюджет (по валютам) |
| `travel_trip_duration_days` | 1-60 | Длительность поездки |
| `travel_guardrail_retries` | 0-3 | Retry-попытки |
| `travel_total_trip_cost_usd` | 100-30000 | Итоговая стоимость |
| `travel_response_token_count` | 100-10000 | Токены в ответе |
| `travel_generation_tokens_per_sec` | 5-200 | Скорость генерации |

### Gauges (custom collector)

| Метрика | Описание |
|---------|----------|
| `travel_host_cpu_percent` | Загрузка CPU (%) |
| `travel_process_resident_memory_bytes` | RSS процесса |

---

## Langfuse трассировка (`backend/langfuse_tracing.py`)

При заданных `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` автоматически:

### Trace на каждый запрос

```
travel_planning (Trace)
├── extract_intent (Span) → trip_extraction (Generation)
├── validate_constraints (Span)
├── fetch_data (Span) → city_attractions (Generation)
├── generate_itinerary (Span) → travel_itinerary (Generation)
└── budget_guardrail (Span)

Scores: budget_compliance, latency_rating, success_score
```

### Метаданные trace

| Поле | Значение |
|------|----------|
| `user_id` | ID пользователя (из UI) |
| `session_id` | thread_id сессии |
| `tags` | `["travel_planning"]` |
| `metadata` | Превью запроса, города, бюджет |

### Score-метрики (автоматические)

| Score | Формула | Диапазон |
|-------|---------|----------|
| `budget_compliance` | `min(1.0, budget / actual_cost)` | 0-1 |
| `latency_rating` | <30s → 1.0, <60s → 0.5, >60s → 0.2 | 0.2-1.0 |
| `success_score` | 0 retries → 1.0, 1-2 → 0.7, 3+ → 0.4 | 0.4-1.0 |

### Dataset-эксперименты

Каждый успешный запрос сохраняется в Langfuse Dataset `travel_planning`:

```json
{
  "input": { "user_text": "...", "extracted_query": {...} },
  "expected_output": { "final_markdown": "...", "total_cost_usd": 1500, "outcome": "ok" },
  "metadata": { "duration_sec": 45, "passengers": 2, "budget": 2000, "destination": "Париж" }
}
```

Используется для A/B тестирования моделей и оценки качества.

### Custom Spans

Контекст-менеджер `langfuse_span()` для обёртки произвольных операций:

```python
with langfuse_span("my_op", metadata={"key": "value"}) as span:
    # ... код ...
    span["result"] = {"status": "ok"}
```

Все узлы графа используют этот механизм для трейсинга.

---

## Grafana дашборды

Запуск: `docker compose --profile monitoring up --build`

### travel-monitoring.json (технический)

- Planning requests rate (ops)
- Planning latency p50/p95
- LLM TTFT prefill vs decode (p95)
- LLM inter-token latency (p95)
- Токены/сек
- Стоимость LLM (USD/s)
- CPU и RSS

### travel-analytics.json (бизнес-аналитика)

- 📊 KPI Cards: запросы, латентность, стоимость, токены, бюджет, success rate
- 🌍 Топ городов вылета/назначения, маршруты
- 💰 Распределение бюджетов, итоговая стоимость, бюджет vs реальная
- 👥 Пассажиры, длительность, валюты, тип поиска
- ⚡ LLM: TTFT, ITL, скорость генерации, токены ответа
- 🔄 Reliability: requests rate, latency, retries, CPU/RSS

Подробнее: [monitoring/README.md](../../monitoring/README.md)

---

## Идеи для evals (CI/CD)

1. **Набор сценариев:** крайние бюджеты, длинные маршруты, неоднозначные города
2. **Проверки:**
   - Бюджет согласован с guardrail
   - Ответ — валидный Markdown с разбивкой по дням
   - Ссылки на реальные `hotel_id` / `flight_id` из тулов
3. **Langfuse Datasets:** использование сохранённых пар input/output для регрессионного тестирования
4. **Score trending:** мониторинг `budget_compliance` и `latency_rating` во времени
