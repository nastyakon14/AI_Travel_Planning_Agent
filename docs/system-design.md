## Архитектура AI Travel Planning Agent (PoC)

### 1. Ключевые решения

- **Оркестрация:** [LangGraph](https://langchain-ai.github.io/langgraph/) — явный граф узлов (`backend/agent_graph.py`), условные рёбра, ретраи при превышении бюджета.
- **Паттерн:** stateful workflow: извлечение намерения → валидация → сбор данных из API → генерация маршрута → guardrail по бюджету.
- **LLM:** [LangChain `ChatOpenAI`](https://python.langchain.com/docs/integrations/chat/openai/) с `base_url=os.getenv("AGENTPLATFORM_API_BASE")` — OpenAI-совместимый HTTP API. Имена моделей задаются переменными `TRIP_EXTRACTION_*`, `TRIP_ATTRACTIONS_MODEL` и т.д. Пакет **LiteLLM в коде не используется**.
- **Память сессии:** встроенный [`MemorySaver`](https://langchain-ai.github.io/langgraph/concepts/persistence/) LangGraph (in-memory PoC).
- **Достопримечательности:** LLM (`suggest_city_attractions`), Nominatim geocoding + Wikipedia thumbnails для визуализации.
- **Наблюдаемость:** Prometheus метрики, Langfuse трассировка (traces, spans, scores, datasets), LLM observability (TTFT, ITL, токены, cost).

### 2. Модули

| Модуль | Файл | Назначение |
|--------|------|------------|
| **UI** | `streamlit_app.py` | Streamlit: чат, фильтры, дашборды, графики, карты |
| **Facade** | `backend/travel_facade.py` | Обёртка графа: Langfuse trace, Prometheus метрики, business metrics |
| **Оркестратор** | `backend/agent_graph.py` | LangGraph StateGraph, 8 узлов, MemorySaver |
| **Travel Agent** | `backend/travel_agent.py` | TripQuery (Pydantic), извлечение (cascade fast→strong), даты, достопримечательности |
| **Инструменты** | `backend/agent_tools.py` | 7 LangChain `@tool` функций |
| **Avia API** | `backend/aviatickets.py` | Travelpayouts Aviasales клиент |
| **Hotels API** | `backend/hotels.py` | Travelpayouts Hotellook клиент (3-tier search) |
| **Guardrails** | `backend/guardrails.py` | Санитизация: injection EN/RU, secrets, XSS |
| **Attractions UI** | `backend/attractions_ui.py` | Folium map, Nominatim, Wikipedia thumbnails |
| **LLM Observability** | `backend/llm_observability.py` | TTFT, TPOT, токены, cost estimation, stream helpers |
| **Prometheus** | `backend/prometheus_metrics.py` | Counters, histograms, gauges: LLM, planning, business metrics |
| **Langfuse** | `backend/langfuse_tracing.py` | Traces, spans, scores, datasets, custom span context manager |
| **Context Memory** | `backend/context_memory.py` | Форматирование истории диалога |
| **Auth** | `backend/auth_streamlit.py` | Простая парольная аутентификация |
| **Serving** | `backend/serving/model_registry.py` | Метаданные моделей для UI |

### 3. Поток выполнения

```
User Input → sanitize → extract TripQuery → validate budget
  → detect search scope (flights/hotels/both)
  → fetch: Travelpayouts (flights, hotels) + LLM (attractions)
  → generate itinerary Markdown
  → guardrail: flight + hotel×nights vs budget
     → pass → finalize
     → fail, retries < N → budget×0.85 → fetch again
     → fail, retries = N → finalize with warning
```

### 4. LangGraph Workflow

| Узел | Вход | Выход | Langfuse Span |
|------|------|-------|---------------|
| `extract` | user_input | requirements (TripQuery dict), extraction_meta | ✅ `extract_intent` |
| `validate` | requirements | early_exit flag + message | ✅ `validate_constraints` |
| `fetch_data` | requirements, budget_multiplier | flights_result, hotels_result, attractions_result | ✅ `fetch_data` |
| `generate` | requirements, fetch results | itinerary_md, itinerary_model, itinerary_llm_metrics | ✅ `generate_itinerary` |
| `guardrail` | requirements, fetch results | guardrail_pass, budget_check dict | ✅ `budget_guardrail` |
| `retry_patch` | — | guardrail_retries+1, budget_multiplier×0.85 | — |
| `finalize` | itinerary_md, budget_check | final_markdown, query, total_cost_usd, retry_count | — |

### 5. Состояние (TravelPlanningState)

```python
class TravelPlanningState(TypedDict, total=False):
    user_input: str                       # Исходный запрос
    conversation_context: str | None      # Контекст диалога
    user_id: str | None                   # ID пользователя
    requirements: dict                    # TripQuery как dict
    extraction_meta: dict                 # Метаданные LLM извлечения
    itinerary_llm_metrics: dict | None    # Метрики генерации маршрута
    default_origin_city: str | None
    default_destination_city: str | None
    default_origin_iata: str | None
    max_results: int                      # Лимит результатов поиска
    early_exit: bool                      # Флаг раннего выхода
    early_exit_message: str
    error: str
    flights_result: dict                  # Рейсы из Travelpayouts
    hotels_result: dict                   # Отели из Travelpayouts
    attractions_result: dict              # Достопримечательности (LLM)
    itinerary_md: str                     # Сгенерированный маршрут
    itinerary_model: str | None
    guardrail_pass: bool
    budget_check: dict                    # total_estimated, budget, flight/hotel parts
    guardrail_retries: int                # Счётчик retry-попыток
    budget_multiplier: float              # Множитель бюджета (1.0 → 0.85 → ...)
    max_guardrail_retries: int            # Максимум retry (default=3)
    final_markdown: str                   # Итоговый ответ
    query: dict                           # TripQuery для бизнес-метрик
    total_cost_usd: float | None          # Итоговая стоимость
    retry_count: int                      # Кол-во retries
    llm_metrics: list                     # Список метрик LLM-вызовов
```

Идентификатор потока: `thread_id` → `configurable: {"thread_id": ...}` → MemorySaver.

### 6. Наблюдаемость

#### Prometheus метрики

| Категория | Метрики |
|-----------|---------|
| Planning | `travel_planning_requests_total`, `travel_planning_duration_seconds` |
| LLM | `travel_llm_prefill_seconds`, `travel_llm_decode_phase_seconds`, `travel_llm_inter_token_latency_seconds`, `travel_llm_input_tokens_total`, `travel_llm_output_tokens_total`, `travel_llm_cost_usd_total` |
| Business | `travel_trip_passengers`, `travel_trip_budget`, `travel_trip_duration_days`, `travel_origin_city_total`, `travel_destination_city_total`, `travel_total_trip_cost_usd`, `travel_response_token_count`, `travel_generation_tokens_per_sec` |
| System | `travel_host_cpu_percent`, `travel_process_resident_memory_bytes` |

#### Langfuse трассировка

| Элемент | Описание |
|---------|----------|
| **Trace** | Корневой trace `travel_planning` на каждый запрос |
| **Spans** | `extract_intent`, `validate_constraints`, `fetch_data`, `generate_itinerary`, `budget_guardrail` |
| **Generations** | LLM-вызовы: `trip_extraction`, `city_attractions`, `travel_itinerary` |
| **Scores** | `budget_compliance` (0-1), `latency_rating`, `success_score` |
| **Datasets** | Dataset `travel_planning` — пары input/output для A/B тестов |
| **Metadata** | `user_id`, `session_id`, `thread_id`, tags, preview |

#### LLM Observability

| Метрика | Описание |
|---------|----------|
| TTFT | Время до первого токена (prefill) |
| ITL/TPOT | Средняя задержка на выходной токен (decode) |
| Tokens | Input, output, total |
| Cost | USD estimation (из metadata или env prices) |

### 7. Grafana дашборды

**travel-monitoring.json** — технический мониторинг:
- Planning requests rate (ops) по outcome и HTTP-коду
- Planning latency p50/p95 (сек)
- LLM TTFT prefill vs decode phase (p95)
- LLM inter-token latency (p95)
- Токены/сек (input + output)
- Стоимость LLM (USD/s по stage)
- CPU хоста и RSS процесса

**travel-analytics.json** — бизнес-аналитика:
- 📊 KPI Cards: запросы, латентность, стоимость LLM, токены, бюджет, success rate
- 🌍 Travel Analytics: топ городов вылета/назначения, маршруты
- 💰 Budget & Cost: распределение бюджетов (pie), гистограмма стоимости, бюджет vs реальная стоимость
- 👥 Passengers & Trip Details: распределение пассажиров, длительность, валюты, тип поиска
- ⚡ LLM Performance: TTFT p50/p95, ITL p95, токены/сек, скорость генерации
- 🔄 System & Reliability: requests rate, latency, guardrail retries, CPU/RSS

### 8. Сбои и ограничения

- Таймауты HTTP к LLM и внешним API через env (`LLM_TIMEOUT_SEC` и др.)
- При ошибках Travelpayouts тулы возвращают пустые списки с диагностикой
- Задержки и стоимость зависят от шлюза LLM; SLA не фиксированы

### 9. Ограничения PoC

- Нет реального бронирования и оплаты
- Оценки бюджета ориентировочные
- Покрытие городов зависит от Travelpayouts
