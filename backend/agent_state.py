"""
Схема состояния агента (см. docs/specs/memory-context.md).

Для LangGraph можно обернуть в TypedDict с Annotated[..., add_messages] после установки langgraph.
Здесь — переносимая структура без обязательной зависимости от графа.
"""

from __future__ import annotations

from typing import Any, TypedDict


class TravelAgentState(TypedDict, total=False):
    """Минимальное состояние сессии планирования поездки."""

    thread_id: str
    messages: list[Any]
    requirements: dict[str, Any]
    found_flights: list[dict[str, Any]]
    found_hotels: list[dict[str, Any]]
    found_attractions: list[dict[str, Any]]
    current_cost: float
    final_plan: str
    guardrail_retries: int
