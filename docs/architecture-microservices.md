# Развёртывание (один процесс)

Приложение — **Streamlit** (`streamlit_app.py`), в том же процессе импортируется **`backend`**: LangGraph (`agent_graph`), инструменты (`agent_tools`), клиенты Travelpayouts.

LLM вызывается через **LangChain `ChatOpenAI`** с `base_url = AGENTPLATFORM_API_BASE` и ключом `AGENTPLATFORM_API_KEY` (см. `backend/travel_agent.py`).

Docker: `docker compose up` — один сервис **`ui`** ([`Dockerfile`](../Dockerfile)), без отдельного REST API.
