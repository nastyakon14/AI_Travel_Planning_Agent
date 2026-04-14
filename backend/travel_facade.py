"""
Точка входа для UI: только in-process — LangGraph / travel_agent в том же процессе, что и Streamlit.

LLM: LangChain `ChatOpenAI` с `base_url` = `AGENTPLATFORM_API_BASE` (см. `travel_agent.OPENAI_URL`).
"""

from __future__ import annotations

from typing import Any

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

    return _impl(
        user_request,
        default_origin_city=default_origin_city,
        default_destination_city=default_destination_city,
        default_origin_iata=default_origin_iata,
        max_results=max_results,
        thread_id=thread_id,
        conversation_context=conversation_context,
        user_id=user_id,
    )


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
