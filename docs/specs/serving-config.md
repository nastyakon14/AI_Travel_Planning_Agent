# Serving и конфигурация

## Компоненты

- **UI:** Streamlit, `streamlit run streamlit_app.py` (см. `Dockerfile` в корне репозитория).
- **Compose:** `docker compose up` поднимает только сервис **`ui`**; LangGraph и LLM выполняются в процессе Streamlit.

Персистентность графа в PoC — in-memory (`MemorySaver`).

## Переменные окружения (основные)

| Переменная | Назначение |
|------------|------------|
| `AGENTPLATFORM_API_KEY` / `OPENAI_API_KEY` | Ключ к LLM API |
| `AGENTPLATFORM_API_BASE` | Базовый URL OpenAI-compatible API (`.../v1`) |
| `TRAVELPAYOUTS_API_TOKEN` | Токен Travelpayouts |
| `TRIP_EXTRACTION_FAST_MODEL`, `TRIP_EXTRACTION_STRONG_MODEL` | Имена моделей на шлюзе |
| `EXTRACTION_COMPLEX_FIRST` | `1` — сначала «сильная» модель для сложных запросов |
| `TRAVEL_GUARDRAIL_MAX_RETRIES` | Лимит ретраев бюджета в графе |

Секреты не коммитить; используйте `.env` локально.

## Версии моделей

Имена моделей задаются через env и должны совпадать с ответом `GET …/v1/models` у вашего шлюза.
