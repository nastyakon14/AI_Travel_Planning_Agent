"""
Расширенная трассировка Langfuse для LangChain / LangGraph.

Возможности:
  1. Метаданные трассировки (user_id, session_id, thread_id, теги)
  2. Score-метрики (quality score, budget compliance, latency rating)
  3. Dataset-эксперименты (сохранение запросов/ответов)
  4. Custom tracing spans для узлов графа
  5. Интеграция с Langfuse Python SDK (не только CallbackHandler)

Требуется: pip install langfuse
Переменные (или Langfuse Cloud):
  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
  LANGFUSE_HOST — базовый URL API (например https://cloud.langfuse.com для EU)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

_handler = None
_langfuse_client = None


def _ensure_langfuse_env() -> None:
    host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")
    if host and not os.getenv("LANGFUSE_BASE_URL"):
        os.environ["LANGFUSE_BASE_URL"] = host.rstrip("/")


def get_langfuse_handler():
    """Один CallbackHandler на процесс; None если ключи не заданы или пакет не установлен."""
    global _handler
    if _handler is not None:
        return _handler
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        return None
    try:
        _ensure_langfuse_env()
        from langfuse.langchain import CallbackHandler

        _handler = CallbackHandler()
        return _handler
    except ImportError:
        logger.warning("langfuse не установлен; трассировка отключена")
        return None
    except Exception as exc:  # pragma: no cover
        logger.warning("Langfuse CallbackHandler недоступен: %s", exc)
        return None


def get_langfuse_client():
    """
    Возвращает Langfuse SDK клиент для продвинутой работы (scores, datasets, spans).
    Инициализируется один раз на процесс.
    """
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        return None
    try:
        _ensure_langfuse_env()
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        )
        return _langfuse_client
    except ImportError:
        logger.warning("langfuse SDK не установлен")
        return None
    except Exception as exc:
        logger.warning("Langfuse клиент недоступен: %s", exc)
        return None


# ============================================================================
# 1. Метаданные трассировки (user_id, session_id, thread_id)
# ============================================================================

def build_langfuse_config(
    *,
    thread_id: str = "default",
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Строит config с Langfuse callbacks и метаданными для trace.

    Параметры:
      thread_id — идентификатор сессии LangGraph (будет session_id в Langfuse)
      user_id — ID пользователя
      session_id — явный ID сессии (если не задан, используется thread_id)
      tags — теги для фильтрации в Langfuse (например: ["travel", "budget_2000"])
      metadata — произвольные метаданные (город, бюджет, и т.д.)
    """
    cfg: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    h = get_langfuse_handler()
    if not h:
        return cfg

    cbs = [h]
    cfg["callbacks"] = cbs

    # Метаданные для Langfuse через callbacks metadata
    langfuse_meta: dict[str, Any] = {}
    if user_id:
        langfuse_meta["user_id"] = user_id
    langfuse_meta["session_id"] = session_id or thread_id
    langfuse_meta["thread_id"] = thread_id
    if tags:
        langfuse_meta["tags"] = tags
    if metadata:
        langfuse_meta.update(metadata)

    if langfuse_meta:
        cfg["langfuse_tags"] = langfuse_meta.get("tags", [])
        cfg["langfuse_metadata"] = langfuse_meta

    return cfg


def update_langfuse_trace_metadata(
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Обновляет метаданные текущего trace (через Langfuse SDK).
    Вызывать внутри обработчика запроса.
    """
    client = get_langfuse_client()
    if not client:
        return
    # Примечание: обновление метаданных активного trace требует trace_id.
    # В LangChain интеграции это происходит автоматически через CallbackHandler.
    # Здесь мы можем только залогировать для отладки.
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Langfuse trace metadata: user_id=%s session_id=%s tags=%s",
            user_id,
            session_id,
            tags,
        )


# ============================================================================
# 2. Score-метрики (quality score, budget compliance, latency rating)
# ============================================================================

def score_travel_quality(
    *,
    trace_id: str,
    query: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    total_cost_usd: float | None = None,
    budget: float | None = None,
    retry_count: int = 0,
    duration_sec: float = 0,
) -> None:
    """
    Выставляет оценки качества для trace в Langfuse.

    Scores:
      budget_compliance — насколько уложились в бюджет (0-1)
      latency_rating — оценка скорости (0-1, <30s=1, <60s=0.5, >60s=0.2)
      success_score — успех без retry (1), с retry (0.7), ошибка (0)
    """
    client = get_langfuse_client()
    if not client:
        return

    # Budget compliance
    if budget is not None and budget > 0 and total_cost_usd is not None:
        compliance = min(1.0, budget / max(total_cost_usd, 0.01))
        try:
            client.score(
                trace_id=trace_id,
                name="budget_compliance",
                value=round(compliance, 3),
                comment="ratio=budget/actual_cost",
            )
        except Exception as exc:
            logger.debug("Langfuse score (budget) failed: %s", exc)

    # Latency rating
    if duration_sec > 0:
        if duration_sec < 30:
            latency_score = 1.0
        elif duration_sec < 60:
            latency_score = 0.5
        else:
            latency_score = 0.2
        try:
            client.score(
                trace_id=trace_id,
                name="latency_rating",
                value=latency_score,
                comment=f"duration={duration_sec:.1f}s",
            )
        except Exception as exc:
            logger.debug("Langfuse score (latency) failed: %s", exc)

    # Success score
    if retry_count == 0:
        success_score = 1.0
    elif retry_count <= 2:
        success_score = 0.7
    else:
        success_score = 0.4
    try:
        client.score(
            trace_id=trace_id,
            name="success_score",
            value=success_score,
            comment=f"retries={retry_count}",
        )
    except Exception as exc:
        logger.debug("Langfuse score (success) failed: %s", exc)


# ============================================================================
# 3. Dataset-эксперименты (сохранение запросов/ответов)
# ============================================================================

def save_to_langfuse_dataset(
    *,
    dataset_name: str = "travel_planning",
    user_input: str,
    result: dict[str, Any],
    query: dict[str, Any] | None = None,
    total_cost_usd: float | None = None,
    duration_sec: float = 0,
    outcome: str = "ok",
) -> None:
    """
    Сохраняет пару запрос/ответ в Langfuse Dataset для A/B тестирования.
    """
    client = get_langfuse_client()
    if not client:
        return

    try:
        # Создаём item в dataset
        client.create_dataset_item(
            dataset_name=dataset_name,
            input={"user_text": user_input, "extracted_query": query},
            expected_output={
                "final_markdown": result.get("final_markdown", ""),
                "total_cost_usd": total_cost_usd,
                "outcome": outcome,
            },
            metadata={
                "duration_sec": round(duration_sec, 2),
                "passengers": (query or {}).get("passengers"),
                "budget": (query or {}).get("budget"),
                "currency": (query or {}).get("currency"),
                "destination": (query or {}).get("destination_city"),
                "origin": (query or {}).get("origin_city"),
            },
        )
        logger.debug("Saved to Langfuse dataset: %s", dataset_name)
    except Exception as exc:
        logger.debug("Langfuse dataset save failed: %s", exc)


# ============================================================================
# 4. Custom tracing spans для узлов графа
# ============================================================================

@contextmanager
def langfuse_span(
    name: str,
    *,
    parent_observation_id: str | None = None,
    trace_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Context manager для создания custom span в Langfuse.

    Использование:
        with langfuse_span("fetch_data", metadata={"scope": "both"}) as span_info:
            # ... код узла ...
            span_info["result"] = {"flights": 5, "hotels": 3}
    """
    client = get_langfuse_client()
    span_obj = None
    start_time = time.perf_counter()

    span_info: dict[str, Any] = {"name": name, "start": start_time}

    if client:
        try:
            span_obj = client.span(
                name=name,
                trace_id=trace_id,
                parent_observation_id=parent_observation_id,
                metadata=metadata,
                tags=tags or [],
            )
            span_info["span_id"] = span_obj.id
        except Exception as exc:
            logger.debug("Langfuse span create failed: %s", exc)

    try:
        yield span_info
    except Exception as exc:
        span_info["error"] = str(exc)
        if span_obj:
            try:
                span_obj.end(
                    status_message="error",
                    metadata={"error": str(exc)},
                )
            except Exception:
                pass
        raise
    else:
        duration = time.perf_counter() - start_time
        span_info["duration"] = duration
        if span_obj:
            try:
                span_obj.end(
                    status_message="success",
                    metadata=span_info.get("result"),
                    output=span_info.get("result"),
                )
            except Exception:
                pass


def create_langfuse_trace_for_planning(
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    input_text: str = "",
) -> str | None:
    """
    Создаёт корневой trace для запроса планирования.
    Возвращает trace_id или None если Langfuse не активен.
    """
    client = get_langfuse_client()
    if not client:
        return None

    try:
        trace = client.trace(
            name="travel_planning",
            user_id=user_id,
            session_id=session_id,
            input={"user_text": input_text},
            tags=["travel", "planning"],
            metadata={
                "source": "streamlit_ui",
                "user_id": user_id,
                "session_id": session_id,
            },
        )
        return trace.id
    except Exception as exc:
        logger.debug("Langfuse trace create failed: %s", exc)
        return None


# ============================================================================
# 5. Хелперы для Grafana интеграции (ссылки на Langfuse trace)
# ============================================================================

def get_langfuse_trace_url(trace_id: str) -> str | None:
    """Возвращает URL для просмотра trace в Langfuse UI."""
    client = get_langfuse_client()
    if not client:
        return None
    # Langfuse Cloud URL
    base = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").rstrip("/")
    return f"{base}/project/{client.public_key}/traces/{trace_id}"


# ============================================================================
# Совместимость: старые функции
# ============================================================================

def get_langfuse_callbacks_list() -> list | None:
    h = get_langfuse_handler()
    return [h] if h else None


def merge_langchain_callbacks(base: dict[str, Any] | None) -> dict[str, Any]:
    """Добавляет Langfuse в config.callbacks для stream/invoke."""
    out = dict(base or {})
    h = get_langfuse_handler()
    if not h:
        return out
    cbs = list(out.get("callbacks") or [])
    cbs.append(h)
    out["callbacks"] = cbs
    return out
