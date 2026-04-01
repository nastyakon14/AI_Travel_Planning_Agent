# State, Memory & Context Handling

## Роль
Управление состоянием графа агента и историей диалога без превышения лимитов токенов LLM.

## Session State (Состояние сессии)
- Хранится через `MemorySaver` (встроенный чекпоинтер LangGraph) 
- Ключ сессии — `thread_id`.
- **Схема состояния (`TypedDict`):**

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages] # История переписки и вызовов тулов
    requirements: dict  # {"city": "", "budget": 0.0, "days": 0}
    found_flights: list # Сохраненные билеты из API
    found_hotels: list  # Сохраненные отели из API
    current_cost: float # Текущий рассчитанный бюджет
    final_plan: str     # Итоговый маркдаун
