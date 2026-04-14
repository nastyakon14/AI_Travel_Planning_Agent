"""
Точка входа для UI: только in-process — LangGraph / travel_agent в том же процессе, что и Streamlit.

LLM: LangChain `ChatOpenAI` с `base_url` = `AGENTPLATFORM_API_BASE` (см. `travel_agent.OPENAI_URL`).
"""

from __future__ import annotations

import time
from typing import Any

from backend.guardrails import GuardrailViolation
from backend.travel_agent import TripQuery


def extract_trip_query(
    user_text: str,
    return_metadata: bool = False,
    *,
    conversation_context: str | None = None,
    user_id: str | None = None,
) -> TripQuery | tuple[TripQuery, dict[str, Any]]:
    from backend.travel_agent import extract_trip_query as _impl

    return _impl(
        user_text,
        return_metadata=return_metadata,
        conversation_context=conversation_context,
        user_id=user_id,
    )


def run_travel_planning_graph(
    user_request: str,
    *,
    default_origin_city: str | None = None,
    default_destination_city: str | None = None,
    default_origin_iata: str | None = None,
    max_results: int = 5,
    thread_id: str = "default",
    conversation_context: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    from backend.agent_graph import run_travel_planning_graph as _impl
    from backend.langfuse_tracing import (
        build_langfuse_config,
        create_langfuse_trace_for_planning,
        get_langfuse_callbacks_list,
        get_langfuse_trace_url,
        save_to_langfuse_dataset,
        score_travel_quality,
    )

    try:
        from backend.prometheus_metrics import budget_to_range, outcome_to_http, record_planning_run, record_trip_business_metrics
    except ImportError:

        def record_planning_run(**_: Any) -> None:
            return None

        def outcome_to_http(_: str) -> str:
            return "200"

        def record_trip_business_metrics(**__: Any) -> None:
            return None

        def budget_to_range(b, c=None):
            return "unknown"

    # Создаём корневой trace в Langfuse
    session_id = thread_id
    trace_id = create_langfuse_trace_for_planning(
        user_id=user_id,
        session_id=session_id,
        input_text=user_request,
    )

    # Строим config с метаданными Langfuse
    langfuse_cbs = get_langfuse_callbacks_list()
    cfg = build_langfuse_config(
        thread_id=thread_id,
        user_id=user_id,
        session_id=session_id,
        tags=["travel_planning"],
        metadata={
            "user_request_preview": user_request[:200],
            "default_origin": default_origin_city,
            "default_destination": default_destination_city,
        },
    )
    if langfuse_cbs:
        cfg["callbacks"] = langfuse_cbs

    t0 = time.perf_counter()
    try:
        out = _impl(
            user_request,
            default_origin_city=default_origin_city,
            default_destination_city=default_destination_city,
            default_origin_iata=default_origin_iata,
            max_results=max_results,
            thread_id=thread_id,
            conversation_context=conversation_context,
            user_id=user_id,
            langchain_callbacks=langfuse_cbs,
        )
        dt = time.perf_counter() - t0
        record_planning_run(outcome="ok", duration_sec=dt, http_code=outcome_to_http("ok"))

        # Записываем бизнес-метрики из результата
        _record_business_metrics_from_result(out, user_request, dt)

        # Langfuse: Score-метрики
        if trace_id:
            query = out.get("query") or {}
            _langfuse_scores(
                trace_id=trace_id,
                query=query,
                result=out,
                duration_sec=dt,
            )

        # Langfuse: сохраняем в dataset
        query = out.get("query") or {}
        save_to_langfuse_dataset(
            user_input=user_request,
            result=out,
            query=query,
            total_cost_usd=out.get("total_cost_usd"),
            duration_sec=dt,
            outcome="ok",
        )

        # Добавляем ссылку на Langfuse trace в результат (для UI)
        trace_url = get_langfuse_trace_url(trace_id) if trace_id else None
        if trace_url:
            out["langfuse_trace_url"] = trace_url
            out["langfuse_trace_id"] = trace_id

        return out
    except GuardrailViolation:
        dt = time.perf_counter() - t0
        record_planning_run(
            outcome="guardrail",
            duration_sec=dt,
            http_code=outcome_to_http("guardrail"),
        )
        raise
    except Exception:
        dt = time.perf_counter() - t0
        record_planning_run(outcome="error", duration_sec=dt, http_code=outcome_to_http("error"))
        raise


def _langfuse_scores(
    *,
    trace_id: str,
    query: dict[str, Any],
    result: dict[str, Any],
    duration_sec: float,
) -> None:
    """Выставляет Score-метрики в Langfuse после успешного выполнения."""
    try:
        from backend.langfuse_tracing import score_travel_quality

        budget = query.get("budget")
        total_cost = result.get("total_cost_usd")
        retry_count = result.get("retry_count", 0)

        score_travel_quality(
            trace_id=trace_id,
            query=query,
            result=result,
            total_cost_usd=total_cost,
            budget=budget,
            retry_count=retry_count,
            duration_sec=duration_sec,
        )
    except Exception:
        pass


def _record_business_metrics_from_result(out: dict[str, Any], user_request: str, duration_sec: float) -> None:
    """Извлекает данные из результата и записывает бизнес-метрики."""
    try:
        from backend.prometheus_metrics import record_trip_business_metrics

        query = out.get("query") or {}
        passengers = query.get("passengers")
        budget = query.get("budget")
        currency = query.get("currency")
        trip_days = query.get("trip_days")
        origin = query.get("origin_city")
        destination = query.get("destination_city")

        # Итоговая стоимость из результата
        total_cost = out.get("total_cost_usd")
        retry_count = out.get("retry_count")

        # Токены из LLM-метрик
        llm_metrics = out.get("llm_metrics", [])
        total_output_tokens = sum(m.get("output_tokens", 0) or 0 for m in llm_metrics)
        total_latency = sum(m.get("latency_sec", 0) or 0 for m in llm_metrics)
        generation_speed = total_output_tokens / total_latency if total_latency > 0 else None

        budget_range = budget_to_range(budget, currency)

        record_trip_business_metrics(
            passengers=passengers,
            budget=budget,
            currency=currency,
            trip_days=trip_days,
            origin_city=origin,
            destination_city=destination,
            outcome="ok",
            budget_range=budget_range,
            retry_count=retry_count,
            total_cost_usd=total_cost,
            response_tokens=total_output_tokens,
            generation_speed_tps=generation_speed,
        )
    except Exception:
        pass


def search_routes_from_extracted(
    user_request: str,
    extracted: TripQuery,
    extraction_meta: dict[str, Any] | None = None,
    default_origin_city: str | None = None,
    default_origin_iata: str | None = None,
    max_results: int = 5,
) -> dict[str, Any]:
    from backend.travel_agent import search_routes_from_extracted as _impl

    return _impl(
        user_request,
        extracted,
        extraction_meta=extraction_meta,
        default_origin_city=default_origin_city,
        default_origin_iata=default_origin_iata,
        max_results=max_results,
    )


def search_hotels_from_extracted(
    user_request: str,
    extracted: TripQuery,
    extraction_meta: dict[str, Any] | None = None,
    default_destination_city: str | None = None,
    max_results: int = 5,
) -> dict[str, Any]:
    from backend.travel_agent import search_hotels_from_extracted as _impl

    return _impl(
        user_request,
        extracted,
        extraction_meta=extraction_meta,
        default_destination_city=default_destination_city,
        max_results=max_results,
    )


def suggest_city_attractions(
    city: str,
    country: str | None = None,
    *,
    max_items: int = 8,
    model: str | None = None,
) -> dict[str, Any]:
    from backend.travel_agent import suggest_city_attractions as _impl

    return _impl(city, country, max_items=max_items, model=model)
