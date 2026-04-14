# AI Travel Planning Agent

PoC агентной системы для планирования поездок: извлечение параметров из текста (LLM), поиск рейсов и отелей ([Travelpayouts](https://www.travelpayouts.com/) / Aviasales API), подсказки по достопримечательностям, генерация маршрута в Markdown и проверка бюджета. Оркестрация — **LangGraph** (`backend/agent_graph.py`); LLM — **LangChain `ChatOpenAI`** к OpenAI-совместимому HTTP API (`AGENTPLATFORM_API_BASE`), без пакета LiteLLM.

---

## Схема работы агента (LangGraph)

Граф собирается в `build_graph()`: узлы и условные переходы совпадают с реализацией.

```mermaid
flowchart TD
    START([START]) --> extract[extract: извлечение TripQuery + guardrails]
    extract --> validate[validate: эвристика бюджета / длительности]
    validate -->|early_exit| finalize[finalize]
    validate -->|ok| fetch[fetch_data: рейсы, отели, достопримечательности]
    fetch --> generate[generate: generate_travel_itinerary → Markdown]
    generate --> guardrail[guardrail: сравнение оценки с бюджетом]
    guardrail -->|pass| finalize
    guardrail -->|fail, retries left| retry_patch[retry_patch: budget_multiplier × 0.85]
    retry_patch --> fetch
    guardrail -->|max retries| finalize_warn[finalize_warn]
    finalize --> END([END])
    finalize_warn --> END
```

Кратко по шагам:

1. **extract** — `extract_trip_query()`: структура `TripQuery` из текста; при нарушении guardrails — выход.
2. **validate** — если бюджет задан и выглядит несовместимым с длительностью — `early_exit` с сообщением.
3. **fetch_data** — по области поиска (`detect_search_scope`): Travelpayouts для рейсов/отелей; достопримечательности — `suggest_city_attractions()` (LLM).
4. **generate** — `generate_travel_itinerary`: маршрут по дням в Markdown.
5. **guardrail** — оценка «перелёт + отель×ночи» vs бюджет; при превышении — **до N** попторов (`TRAVEL_GUARDRAIL_MAX_RETRIES`, по умолчанию 3) с ужесточением лимита отелей через `budget_multiplier`.

Состояние сессии в PoC: **MemorySaver** (in-process), ключ потока `thread_id` в `run_travel_planning_graph`.

---

## Инструменты агента (`backend/agent_tools.py`)

Инструменты оформлены как LangChain `@tool` и собраны в `AGENT_TOOL_FUNCTIONS`. Дополнительно `get_extended_tool_list()` добавляет NL-обёртки из `travel_agent` для совместимости с UI.

| Инструмент | Назначение |
|------------|------------|
| `search_flights` | Авиапоиск по городам и датам (через `search_routes_from_extracted` / Travelpayouts). |
| `search_hotels` | Поиск отелей по городу и датам заезда/выезда. |
| `search_attractions` | Идеи мест в городе (обёртка над логикой достопримечательностей). |
| `extract_travel_requirements` | Извлечение `TripQuery` из свободного текста. |
| `check_travel_budget` | Сводка и проверка бюджета по извлечённым данным. |
| `generate_travel_itinerary` | Генерация Markdown-маршрута через LLM. |
| `validate_travel_constraints` | Лёгкая эвристика «бюджет vs дни» без внешних API. |

Расширенный список: `search_routes_from_text`, `search_hotels_from_text`, `search_travel_from_text`.

---

## Запуск

### Требования

- Python 3.10+
- Ключи: **`AGENTPLATFORM_API_KEY`** (или `OPENAI_API_KEY`), **`TRAVELPAYOUTS_API_TOKEN`**

### Локально (Streamlit)

```bash
pip install -r requirements.txt
```

Создайте `.env` в корне (или экспортируйте переменные):

```env
AGENTPLATFORM_API_KEY=...
AGENTPLATFORM_API_BASE=https://litellm.tokengate.ru/v1
TRAVELPAYOUTS_API_TOKEN=...
# Имена моделей — как у провайдера / GET {AGENTPLATFORM_API_BASE}/v1/models
TRIP_EXTRACTION_FAST_MODEL=gpt-4o-mini
TRIP_EXTRACTION_STRONG_MODEL=gpt-4o
```

```bash
streamlit run streamlit_app.py
```

Откройте URL из вывода (обычно `http://localhost:8501`).

### Docker

```bash
docker compose up --build
```

- UI: `http://localhost:8501` (`UI_PORT`). Один контейнер: Streamlit + LangGraph в одном процессе.

Переменные — из `.env` рядом с `docker-compose.yml`.

---

## Документация в репозитории

| Файл | Содержание |
|------|------------|
| [docs/architecture-microservices.md](docs/architecture-microservices.md) | Один процесс, Docker |
| [docs/system-design.md](docs/system-design.md) | Архитектура PoC (актуализировано под код) |
| [docs/specs/tools-APIs.md](docs/specs/tools-APIs.md) | Контракты тулов и таймауты |
| [docs/specs/agent-orchestrator.md](docs/specs/agent-orchestrator.md) | Узлы LangGraph |
| [docs/diagrams/workflow.mmd](docs/diagrams/workflow.mmd) | Диаграмма (Mermaid-совместимая схема) |
| [docs/product-proposal.md](docs/product-proposal.md) | Продуктовое видение PoC |
| [docs/governance.md](docs/governance.md) | Риски и политика данных (общая) |
| [docs/specs/observability-evals.md](docs/specs/observability-evals.md) | Метрики и evals |

Диаграммы C4 и поток данных: [docs/diagrams/](docs/diagrams/) (`c4-context.mmd`, `c4-container.mmd`, `data-flow.mmd`).

---

## Ссылки на технологии

- [LangGraph](https://langchain-ai.github.io/langgraph/) — граф состояний и чекпоинты  
- [LangChain OpenAI](https://python.langchain.com/docs/integrations/chat/openai/) — `ChatOpenAI` + `base_url`  
- [Streamlit](https://docs.streamlit.io/) — UI  
- [Travelpayouts / Aviasales API](https://support.travelpayouts.com/hc/en-us/articles/203956163-Aviasales-Travelpayouts-API) — рейсы и отели (read-only, без бронирования)

---

## Ограничения PoC

Нет реального бронирования и оплаты; оценки бюджета ориентировочные; покрытие городов и API зависит от Travelpayouts и настроек ключа.
