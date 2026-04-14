# State, Memory & Context

## Роль

Состояние одного прогона графа и изоляция сессий через `thread_id`.

## Реализация (PoC)

- Чекпоинтер: `MemorySaver` при компиляции графа (`get_compiled_travel_graph()`).
- Конфигурация вызова: `configurable.thread_id` в `run_travel_planning_graph`.

## Схема состояния

Фактический тип — `TravelPlanningState` в `backend/agent_graph.py` (`TypedDict`), в т.ч.:

- `user_input`, `conversation_context`, `user_id`
- `requirements` (словарь из `TripQuery`)
- `extraction_meta`, `flights_result`, `hotels_result`, `attractions_result`
- `itinerary_md`, `guardrail_pass`, `budget_check`, `guardrail_retries`, `budget_multiplier`
- `final_markdown`, `early_exit`, `error`

Долгосрочная память пользователя и векторное хранилище в этом репозитории **не реализованы**.
