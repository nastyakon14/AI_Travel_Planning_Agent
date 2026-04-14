# Retrieval (RAG) — не реализовано в текущем PoC

В коде **нет** отдельного векторного индекса (Qdrant/Chroma и т.д.) для достопримечательностей.

Подбор и формулировки для мест в городе выполняются через **`suggest_city_attractions`** в `backend/travel_agent.py` (LLM + эвристики по названию города).

Если понадобится RAG: вынести статические гиды в индекс, подключить эмбеддинги и retrieval-узел перед генерацией маршрута — см. [LangGraph documentation](https://langchain-ai.github.io/langgraph/).
