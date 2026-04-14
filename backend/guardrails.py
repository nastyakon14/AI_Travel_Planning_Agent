"""Входные/выходные ограничения для LLM и инструментов (PoC guardrails)."""

from __future__ import annotations

import os
import re
from typing import Any

# Максимум символов в одном пользовательском сообщении (защита от переполнения контекста).
MAX_USER_INPUT_CHARS = int(os.getenv("GUARDRAIL_MAX_USER_CHARS", "24000"))

_INJECTION_HINTS = re.compile(
    r"(?i)\b(ignore (all )?previous|system prompt|you are now|jailbreak|"
    r"disregard (the )?above|override instructions)\b"
)


class GuardrailViolation(Exception):
    """Пользовательский ввод отклонён политикой безопасности."""


def sanitize_user_input(text: str) -> str:
    """
    Удаляет NUL, схлопывает экстремальные повторы, обрезает по длине.
    При явных признаках prompt-injection выбрасывает GuardrailViolation.
    """
    if text is None:
        raise GuardrailViolation("Пустой запрос.")
    raw = str(text).replace("\x00", "").strip()
    if not raw:
        raise GuardrailViolation("Пустой запрос.")
    if len(raw) > MAX_USER_INPUT_CHARS:
        raise GuardrailViolation(
            f"Слишком длинное сообщение (>{MAX_USER_INPUT_CHARS} символов). Сократите текст."
        )
    # Один «спам»-символ подряд
    raw = re.sub(r"(.)\1{120,}", r"\1\1\1…[truncated]", raw)
    if _INJECTION_HINTS.search(raw) and os.getenv("GUARDRAIL_STRICT_INJECTION", "0").lower() in (
        "1",
        "true",
        "yes",
    ):
        raise GuardrailViolation(
            "Запрос отклонён: обнаружены фразы, характерные для обхода инструкций модели."
        )
    return raw.strip()


def guardrail_tool_args(name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Лёгкая проверка аргументов инструментов перед вызовом API."""
    out = dict(kwargs)
    if name in ("search_flights", "search_hotels", "search_attractions", "generate_travel_itinerary"):
        for key in list(out.keys()):
            v = out[key]
            if isinstance(v, str) and len(v) > 8000:
                out[key] = v[:7997] + "..."
    return out


def clip_llm_markdown(text: str | None, max_chars: int | None = None) -> str:
    """Ограничение длины ответа ассистента для UI."""
    limit = max_chars or int(os.getenv("GUARDRAIL_MAX_ASSISTANT_CHARS", "120000"))
    if not text:
        return ""
    s = str(text)
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 20)] + "\n\n… [обрезано по лимиту guardrail]"
