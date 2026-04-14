## Архитектура AI Travel Planning Agent (PoC)

### 1. Ключевые решения

- **Оркестрация:** [LangGraph](https://langchain-ai.github.io/langgraph/) — явный граф узлов (`backend/agent_graph.py`), условные рёбра, ретраи при превышении бюджета.
- **Паттерн:** stateful workflow: извлечение намерения → валидация → сбор данных из API → генерация маршрута → guardrail по бюджету.
- **LLM:** [LangChain `ChatOpenAI`](https://python.langchain.com/docs/integrations/chat/openai/) с `base_url=os.getenv("AGENTPLATFORM_API_BASE")` — OpenAI-совместимый HTTP API. Имена моделей задаются переменными `TRIP_EXTRACTION_*`, `TRIP_ATTRACTIONS_MODEL` и т.д. (задаются через `TRIP_EXTRACTION_*` и список моделей шлюза (`/v1/models`)). Пакет **LiteLLM в коде не используется**.
- **Память сессии:** встроенный [`MemorySaver`](https://langchain-ai.github.io/langgraph/concepts/persistence/) LangGraph (in-memory PoC). PostgreSQL/Redis для персистентных чекпоинтов — не подключены.
- **Достопримечательности:** эвристики и LLM (`suggest_city_attractions`), **отдельного векторного RAG в коде нет**.

### 2. Модули

- **REST API** (`services/api/main.py`): извлечение, полный прогон графа, поиск рейсов/отелей, метрики Prometheus, `/v1/serving/info`.
- **Streamlit** (`streamlit_app.py`): чат, ручные фильтры, опционально полный цикл LangGraph в том же процессе.
- **Инструменты** (`backend/agent_tools.py`): обёртки над `travel_agent` и Travelpayouts.
- **Guardrails** (`backend/guardrails.py`): санитизация ввода до LLM.

### 3. Поток выполнения (упрощённо)

1. Пользовательский текст → **извлечение** `TripQuery` (structured output).
2. **Валидация** бюджета относительно длительности (эвристика).
3. **Поиск** рейсов/отелей (Travelpayouts), достопримечательности (LLM).
4. **Генерация** Markdown-маршрута (`generate_travel_itinerary`).
5. **Guardrail:** сумма оценки vs бюджет; при провале — повтор с понижением «эффективного» бюджета для отелей (`budget_multiplier`).

### 4. Состояние

Тип `TravelPlanningState` в `agent_graph.py`: `requirements`, результаты поиска, `itinerary_md`, счётчики guardrail, `final_markdown`. Идентификатор потока — `thread_id` в `run_travel_planning_graph`.

### 5. Сбои и ограничения

- Таймауты HTTP к LLM и внешним API настраиваются переменными (`LLM_TIMEOUT_SEC` и т.д.).
- При ошибках Travelpayouts тулы возвращают диагностику и пустые списки, чтобы сценарий мог продолжиться с пояснением в ответе.

### 6. Ограничения PoC

Задержки и стоимость зависят от шлюза LLM и объёма запросов; точные SLA не фиксированы в коде.
