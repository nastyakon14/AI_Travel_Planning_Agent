"""
Guardrails: политика входа для LLM и инструментов (PoC).

Слои:
  1. Санитизация — NUL, экстремальные повторы, лимит длины.
  2. Prompt-injection — эвристики по типовым фразам обхода инструкций (RU/EN).
  3. Утечки секретов — ключи API, PEM, токены в стиле OpenAI/GitHub/AWS и т.п.
  4. Прочее — явные XSS/скрипт-вставки в пользовательском тексте.

Отключение эвристик 2–4 (только длина/NUL): GUARDRAIL_CONTENT_FILTERS=0
Узкая настройка: GUARDRAIL_ENABLE_INJECTION, GUARDRAIL_ENABLE_SECRETS, GUARDRAIL_ENABLE_MARKUP (см. ниже).
"""

from __future__ import annotations

import os
import re
from typing import Any

# Максимум символов в одном пользовательском сообщении (защита от переполнения контекста).
MAX_USER_INPUT_CHARS = int(os.getenv("GUARDRAIL_MAX_USER_CHARS", "24000"))

# Мастер-выключатель эвристик injection/secrets/markup (1 = включено по умолчанию).
_CONTENT_FILTERS = os.getenv("GUARDRAIL_CONTENT_FILTERS", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
_ENABLE_INJECTION = _CONTENT_FILTERS and os.getenv("GUARDRAIL_ENABLE_INJECTION", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
_ENABLE_SECRETS = _CONTENT_FILTERS and os.getenv("GUARDRAIL_ENABLE_SECRETS", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
_ENABLE_MARKUP = _CONTENT_FILTERS and os.getenv("GUARDRAIL_ENABLE_MARKUP", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)


class GuardrailViolation(Exception):
    """Пользовательский ввод отклонён политикой безопасности."""

    def __init__(self, message: str, *, code: str = "policy") -> None:
        super().__init__(message)
        self.code = code


# --- Prompt injection (мультиязычные эвристики; не заменяют серверные политики модели) ---

_INJECTION_EN = re.compile(
    r"(?is)"
    r"\b("
    r"ignore (all )?(previous|prior) (instructions|prompts?)|"
    r"disregard (the )?(above|instructions)|"
    r"system prompt|override (the )?instructions|"
    r"you are now (a |an )?|"
    r"jailbreak|DAN mode|developer mode|"
    r"reveal (your )?(hidden |system )?(prompt|instructions)|"
    r"print(_| )?your (system )?prompt|"
    r"repeat (the )?(words )?above (back|verbatim)"
    r")\b"
)

_INJECTION_RU = re.compile(
    r"(?isu)"
    r"("
    r"игнорируй(те)?\s+(все\s+)?предыдущ\w*\s+инструкц\w*|"
    r"игнорируй(те)?\s+ранее\s+[\w\s,]*инструкц\w*|"
    r"забудь(те)?\s+(все\s+)?инструкц\w*|"
    r"новые\s+правила\s*:|"
    r"ты\s+теперь\s+(\w+\s+)?(ассистент|бот|модель)|"
    r"раскрой(те)?\s+(системн(ый|ое)\s+)?(промпт|сообщени\w*)|"
    r"покажи(те)?\s+(мне\s+)?(системн(ый|ое)\s+)?промпт|"
    r"выйди(те)?\s+из\s+роли|"
    r"обойди(те)?\s+ограничени\w*|"
    r"скопируй(те)?\s+(весь\s+)?(текст\s+)?выше"
    r")"
)

_INJECTION_MARKERS = re.compile(
    r"(?is)"
    r"(\[INST\]|<\|im_start\|>assistant\||<\|system\|>|###\s*System\b|javascript\s*:)"
)


# --- Утечки секретов (консервативные паттерны; цель — блокировать явные вставки ключей) ---

# OpenAI / похожие префиксы ключей
_SECRET_SK = re.compile(r"(?i)\bsk-(?:proj-)?[a-zA-Z0-9\-]{16,}\b")
# GitHub PAT
_SECRET_GH = re.compile(r"(?i)\bghp_[a-zA-Z0-9]{20,}\b|\bgithub_pat_[a-zA-Z0-9_]{20,}\b")
# AWS Access Key ID
_SECRET_AWS = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
# Slack, Telegram bot-like
_SECRET_BOT = re.compile(r"(?i)\bxox[baprs]-[0-9]+-[a-zA-Z0-9\-]{10,}\b")
# PEM блок
_SECRET_PEM = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
# Назначение: длинный токен после api_key / token / secret (одна строка «подставь ключ»)
_SECRET_INLINE = re.compile(
    r"(?i)(?:api[_-]?key|apikey|auth[_-]?token|access[_-]?token|secret|password)\s*[=:]\s*"
    r"['\"]?([a-zA-Z0-9_+/=\-]{24,})"
)
# Authorization: Bearer <jwt-like>
_SECRET_BEARER = re.compile(
    r"(?i)\bBearer\s+[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\b"
)


def _reject_injection(text: str) -> None:
    if _INJECTION_EN.search(text):
        raise GuardrailViolation(
            "Запрос отклонён: обнаружены формулировки, характерные для обхода инструкций модели (EN).",
            code="prompt_injection",
        )
    if _INJECTION_RU.search(text):
        raise GuardrailViolation(
            "Запрос отклонён: обнаружены формулировки, характерные для обхода инструкций модели (RU).",
            code="prompt_injection",
        )
    if _INJECTION_MARKERS.search(text):
        raise GuardrailViolation(
            "Запрос отклонён: обнаружены подозрительные технические маркеры (инъекция контекста / скрипт).",
            code="prompt_injection",
        )


def _reject_secrets(text: str) -> None:
    if _SECRET_SK.search(text):
        raise GuardrailViolation(
            "Запрос отклонён: похоже на фрагмент API-ключа. Не вставляйте секреты в чат.",
            code="secret_leak",
        )
    if _SECRET_GH.search(text):
        raise GuardrailViolation(
            "Запрос отклонён: похоже на токен доступа GitHub. Не публикуйте секреты в чате.",
            code="secret_leak",
        )
    if _SECRET_AWS.search(text):
        raise GuardrailViolation(
            "Запрос отклонён: похоже на идентификатор ключа AWS. Не передавайте секреты в запросе.",
            code="secret_leak",
        )
    if _SECRET_BOT.search(text):
        raise GuardrailViolation(
            "Запрос отклонён: похоже на токен бота/интеграции. Не вставляйте секреты в чат.",
            code="secret_leak",
        )
    if _SECRET_PEM.search(text):
        raise GuardrailViolation(
            "Запрос отклонён: обнаружен блок приватного ключа (PEM). Не передавайте ключи в чате.",
            code="secret_leak",
        )
    if _SECRET_INLINE.search(text):
        raise GuardrailViolation(
            "Запрос отклонён: похоже на строку с секретом (api_key / token / password). "
            "Не вставляйте учётные данные в сообщение.",
            code="secret_leak",
        )
    if _SECRET_BEARER.search(text):
        raise GuardrailViolation(
            "Запрос отклонён: похоже на Bearer/JWT в полном виде. Не передавайте токены авторизации в чате.",
            code="secret_leak",
        )


def _reject_markup(text: str) -> None:
    # Только явные теги скрипта; обычный HTML в путешествиях редок и рискован для рендеринга
    if re.search(r"(?is)<\s*script\b", text):
        raise GuardrailViolation(
            "Запрос отклонён: обнаружен тег script. Опишите задачу обычным текстом.",
            code="markup_policy",
        )


def sanitize_user_input(text: str) -> str:
    """
    Проверяет ввод по политике Guardrails, нормализует текст.

    Порядок: пустота → длина → повторы → (опц.) injection → секреты → (опц.) markup.
    """
    if text is None:
        raise GuardrailViolation("Пустой запрос.", code="empty")
    raw = str(text).replace("\x00", "").strip()
    if not raw:
        raise GuardrailViolation("Пустой запрос.", code="empty")
    if len(raw) > MAX_USER_INPUT_CHARS:
        raise GuardrailViolation(
            f"Слишком длинное сообщение (>{MAX_USER_INPUT_CHARS} символов). Сократите текст.",
            code="length",
        )
    raw = re.sub(r"(.)\1{120,}", r"\1\1\1…[truncated]", raw)

    if _ENABLE_INJECTION:
        _reject_injection(raw)
    if _ENABLE_SECRETS:
        _reject_secrets(raw)
    if _ENABLE_MARKUP:
        _reject_markup(raw)

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
