# Agent & Orchestrator (LangGraph)

## Роль

Ядро — скомпилированный граф в `backend/agent_graph.py` (`build_graph()` → `compile(checkpointer=MemorySaver())`).

## Узлы (реализация)

| Узел | Функция | Содержание |
|------|---------|------------|
| `extract` | `node_extract` | Guardrails на ввод → `extract_trip_query()` → `requirements` + `extraction_meta`. |
| `validate` | `node_validate` | Эвристика: бюджет слишком мал для длительности → `early_exit`. |
| `fetch_data` | `node_fetch_data` | По `detect_search_scope`: рейсы (`search_routes_from_extracted`), отели (`search_hotels_from_extracted`), достопримечательности (`suggest_city_attractions`). Учитывает `budget_multiplier` при ретраях. |
| `generate` | `node_generate` | `generate_travel_itinerary.invoke(...)` → `itinerary_md`. |
| `guardrail` | `node_guardrail` | Оценка перелёт + отель×ночи vs `budget`; `guardrail_pass`. |
| `retry_patch` | `node_retry_patch` | `guardrail_retries += 1`, `budget_multiplier *= 0.85`. |
| `finalize` / `finalize_warn` | `node_finalize` | Сборка `final_markdown`, предупреждение при исчерпании ретраев. |

## Условные переходы

- После `validate`: при `early_exit` → `finalize`, иначе → `fetch_data`.
- После `guardrail`: успех → `finalize`; провал и retries < max → `retry_patch` → снова `fetch_data`; провал и retries ≥ max → `finalize_warn`.

## Лимит ретраев

`TRAVEL_GUARDRAIL_MAX_RETRIES` (по умолчанию 3) — см. `run_travel_planning_graph`.
