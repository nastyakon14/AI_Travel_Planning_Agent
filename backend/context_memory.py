"""Форматирование истории диалога для подсказок LLM (context engineering)."""

from __future__ import annotations

from typing import Any


def format_conversation_for_extraction(
    messages: list[dict[str, Any]],
    *,
    max_turns: int = 10,
    max_chars_per_message: int = 2000,
    exclude_last_user: bool = True,
) -> str | None:
    """
    Собирает краткий контекст из списка сообщений вида {"role": "user"|"assistant", "content": str}.
    Последнее сообщение пользователя обычно дублируется в основном промпте — его можно исключить.
    """
    if not messages:
        return None
    slice_msgs = messages[:-1] if exclude_last_user and messages else messages
    slice_msgs = slice_msgs[-max_turns:]
    lines: list[str] = []
    for m in slice_msgs:
        role = str(m.get("role") or "unknown")
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        if len(content) > max_chars_per_message:
            content = content[: max_chars_per_message - 3] + "..."
        lines.append(f"{role.upper()}: {content}")
    if not lines:
        return None
    return "\n\n".join(lines)
