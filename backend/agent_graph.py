"""
LangGraph-оркестратор по docs/specs/agent-orchestrator.md и diagrams/workflow.mmd.

Узлы: ExtractIntent → ValidateConstraints → FetchData → GenerateItinerary → BudgetGuardrail
→ (до 3 повторов с budget_multiplier *= 0.85) → Finalize.

Чекпоинтер: MemorySaver (PoC). Для PostgreSQL/Redis см. документацию LangGraph persistence.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

try:
    from .agent_tools import generate_travel_itinerary
    from .guardrails import GuardrailViolation, sanitize_user_input
    from .travel_agent import (
        OPENAI_API_KEY,
        TripQuery,
        destination_label_for_attractions,
        detect_search_scope,
        extract_trip_query,
        search_hotels_from_extracted,
        search_routes_from_extracted,
        suggest_city_attractions,
    )
except ImportError:  # pragma: no cover
    from agent_tools import generate_travel_itinerary
    from guardrails import GuardrailViolation, sanitize_user_input
    from travel_agent import (
        OPENAI_API_KEY,
        TripQuery,
        destination_label_for_attractions,
        detect_search_scope,
        extract_trip_query,
        search_hotels_from_extracted,
        search_routes_from_extracted,
        suggest_city_attractions,
    )


class TravelPlanningState(TypedDict, total=False):
    user_input: str
    conversation_context: str | None
    user_id: str | None
    requirements: dict[str, Any]
    extraction_meta: dict[str, Any]
    default_origin_city: str | None
    default_destination_city: str | None
    default_origin_iata: str | None
    max_results: int
    early_exit: bool
    early_exit_message: str
    error: str
    flights_result: dict[str, Any]
    hotels_result: dict[str, Any]
    attractions_result: dict[str, Any]
    itinerary_md: str
    itinerary_model: str | None
    itinerary_error: str
    guardrail_pass: bool
    budget_check: dict[str, Any]
    guardrail_retries: int
    budget_multiplier: float
    max_guardrail_retries: int
    final_markdown: str


def _nights_from_query(q: TripQuery) -> int:
    if q.trip_days and int(q.trip_days) > 0:
        return int(q.trip_days)
    d0, d1 = q.departure_date, q.return_date
    if d0 and d1:
        try:
            a = datetime.strptime(str(d0).strip()[:10], "%Y-%m-%d").date()
            b = datetime.strptime(str(d1).strip()[:10], "%Y-%m-%d").date()
            return max(1, (b - a).days)
        except ValueError:
            pass
    return 3


def _apply_budget_multiplier(rq: TripQuery, multiplier: float) -> TripQuery:
    if rq.budget is None or rq.budget <= 0:
        return rq
    return rq.model_copy(update={"budget": float(rq.budget) * multiplier})


def node_extract(state: TravelPlanningState) -> dict[str, Any]:
    if not OPENAI_API_KEY:
        return {
            "error": "Нет AGENTPLATFORM_API_KEY / OPENAI_API_KEY",
            "early_exit": True,
            "early_exit_message": "Ключ LLM не настроен.",
        }
    try:
        raw_in = sanitize_user_input(state["user_input"].strip())
    except GuardrailViolation as exc:
        return {
            "error": str(exc),
            "early_exit": True,
            "early_exit_message": str(exc),
        }
    try:
        extracted, meta = extract_trip_query(
            raw_in,
            return_metadata=True,
            conversation_context=state.get("conversation_context"),
            user_id=state.get("user_id"),
        )
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "early_exit": True,
            "early_exit_message": f"Ошибка извлечения: {exc}",
        }
    return {
        "requirements": extracted.model_dump(),
        "extraction_meta": meta,
        "early_exit": False,
    }


def node_validate(state: TravelPlanningState) -> dict[str, Any]:
    if state.get("early_exit") or not state.get("requirements"):
        return {}
    rq = TripQuery.model_validate(state["requirements"])
    days = max(1, _nights_from_query(rq))
    budget = float(rq.budget or 0)
    dest = (rq.destination_city or rq.destination_country or "город").strip()
    currency = (rq.currency or "EUR").upper()
    min_suggested = 30.0 * days if currency == "EUR" else 3000.0 * days
    ok = budget <= 0 or budget >= min_suggested * 0.12
    if not ok:
        return {
            "early_exit": True,
            "early_exit_message": (
                f"Бюджет {budget} {currency} выглядит слишком низким для поездки ~{days} дн. в «{dest}». "
                "Увеличьте бюджет или сократите срок."
            ),
        }
    return {"early_exit": False}


def node_fetch_data(state: TravelPlanningState) -> dict[str, Any]:
    if state.get("early_exit"):
        return {}
    rq = TripQuery.model_validate(state["requirements"])
    bm = float(state.get("budget_multiplier") or 1.0)
    rq = _apply_budget_multiplier(rq, bm)

    user_request = state["user_input"]
    em = state.get("extraction_meta") or {}
    scope = detect_search_scope(user_request)
    max_r = int(state.get("max_results") or 5)
    default_origin = state.get("default_origin_city")
    default_dest = state.get("default_destination_city")
    default_oi = state.get("default_origin_iata")

    out: dict[str, Any] = {}

    if scope in ("flights", "both"):
        try:
            out["flights_result"] = search_routes_from_extracted(
                user_request=user_request,
                extracted=rq,
                extraction_meta=em,
                default_origin_city=default_origin,
                default_origin_iata=default_oi,
                max_results=max_r,
            )
        except Exception as exc:
            out["flights_result"] = {"routes": [], "error": str(exc), "route_not_found_message": str(exc)}

    if scope in ("hotels", "both"):
        try:
            out["hotels_result"] = search_hotels_from_extracted(
                user_request=user_request,
                extracted=rq,
                extraction_meta=em,
                default_destination_city=default_dest,
                max_results=max_r,
            )
        except Exception as exc:
            out["hotels_result"] = {"hotels": [], "error": str(exc)}

    label, country = destination_label_for_attractions(rq, default_dest)
    if label:
        out["attractions_result"] = suggest_city_attractions(label, country, max_items=8)

    return out


def node_generate(state: TravelPlanningState) -> dict[str, Any]:
    if state.get("early_exit"):
        return {}
    rq = TripQuery.model_validate(state["requirements"])
    dest = (
        rq.destination_city
        or state.get("default_destination_city")
        or rq.destination_country
        or "город"
    )
    days = max(1, min(_nights_from_query(rq), 21))
    names: list[str] = []
    ar = state.get("attractions_result") or {}
    for a in ar.get("attractions") or []:
        if isinstance(a, dict) and a.get("name"):
            names.append(str(a["name"]))
    notes_parts = []
    fr = state.get("flights_result") or {}
    hr = state.get("hotels_result") or {}
    if fr.get("routes"):
        r0 = fr["routes"][0]
        notes_parts.append(
            f"Ориентир перелёт: {r0.get('origin')}→{r0.get('destination')}, "
            f"от {float(r0.get('price') or 0):.0f} {r0.get('currency', '')}"
        )
    if hr.get("hotels"):
        h0 = hr["hotels"][0]
        if not h0.get("is_search_portal_only"):
            notes_parts.append(
                f"Ориентир отель: {h0.get('name')}, "
                f"{float(h0.get('price_per_night') or 0):.0f} {h0.get('currency', '')}/ночь"
            )
    user_notes = "; ".join(notes_parts)

    try:
        raw = generate_travel_itinerary.invoke(
            {
                "destination_city": str(dest),
                "days": days,
                "attractions": names[:12],
                "user_notes": user_notes,
            }
        )
    except Exception as exc:
        return {"itinerary_md": f"(Ошибка генерации маршрута: {exc})", "itinerary_error": str(exc)}

    md = raw.get("itinerary_markdown") if isinstance(raw, dict) else None
    return {
        "itinerary_md": md or "",
        "itinerary_model": raw.get("model") if isinstance(raw, dict) else None,
    }


def node_guardrail(state: TravelPlanningState) -> dict[str, Any]:
    if state.get("early_exit"):
        return {"guardrail_pass": True}
    rq = TripQuery.model_validate(state["requirements"])
    budget = rq.budget
    if budget is None or budget <= 0:
        return {"guardrail_pass": True, "budget_check": {"skipped": True}}

    nights = _nights_from_query(rq)
    fr = state.get("flights_result") or {}
    hr = state.get("hotels_result") or {}
    routes = fr.get("routes") or []
    hotels = hr.get("hotels") or []

    fp = float(routes[0].get("price") or 0) if routes else 0.0
    hp = 0.0
    for h in hotels:
        if h.get("is_search_portal_only"):
            continue
        p = float(h.get("price_per_night") or 0)
        if p > 0:
            hp = p
            break

    scope = detect_search_scope(state["user_input"])
    if scope == "flights":
        total = fp
    elif scope == "hotels":
        total = hp * nights
    else:
        total = fp + hp * nights

    passed = total <= float(budget) * 1.03
    return {
        "guardrail_pass": passed,
        "budget_check": {
            "total_estimated": round(total, 2),
            "budget": float(budget),
            "currency": rq.currency,
            "nights": nights,
            "flight_part": fp,
            "hotel_part": round(hp * nights, 2),
        },
    }


def node_retry_patch(state: TravelPlanningState) -> dict[str, Any]:
    retries = int(state.get("guardrail_retries") or 0) + 1
    bm = float(state.get("budget_multiplier") or 1.0) * 0.85
    return {"guardrail_retries": retries, "budget_multiplier": bm}


def node_finalize(state: TravelPlanningState) -> dict[str, Any]:
    parts: list[str] = []
    if state.get("early_exit"):
        parts.append("## Остановка\n\n" + str(state.get("early_exit_message") or state.get("error") or ""))
        return {"final_markdown": "\n\n".join(parts).strip()}

    parts.append("## План поездки\n\n" + str(state.get("itinerary_md") or ""))

    bc = state.get("budget_check") or {}
    if bc and not bc.get("skipped"):
        req = state.get("requirements") or {}
        parts.append(
            f"\n\n### Бюджет\n\nОриентировочная сумма: **{bc.get('total_estimated')}** "
            f"{req.get('currency', '')} "
            f"(лимит: **{bc.get('budget')}**)."
        )

    gp = state.get("guardrail_pass")
    retries = int(state.get("guardrail_retries") or 0)
    max_r = int(state.get("max_guardrail_retries") or 3)
    if gp is False and retries >= max_r:
        parts.append(
            "\n\n> **Внимание:** не удалось уложиться в заданный бюджет за отведённые попытки. "
            "Показан лучший доступный вариант — уточните даты или бюджет."
        )
    elif gp is False:
        parts.append("\n\n> Бюджет превышен; выполнена повторная попытка с более жёстким лимитом по отелям.")

    return {"final_markdown": "\n\n".join(parts).strip()}


def route_after_validate(state: TravelPlanningState) -> Literal["fetch", "end"]:
    if state.get("early_exit"):
        return "end"
    return "fetch"


def route_after_guardrail(state: TravelPlanningState) -> Literal["finalize", "retry", "finalize_warn"]:
    if state.get("guardrail_pass"):
        return "finalize"
    if int(state.get("guardrail_retries") or 0) >= int(state.get("max_guardrail_retries") or 3):
        return "finalize_warn"
    return "retry"


def build_graph() -> StateGraph:
    g = StateGraph(TravelPlanningState)
    g.add_node("extract", node_extract)
    g.add_node("validate", node_validate)
    g.add_node("fetch_data", node_fetch_data)
    g.add_node("generate", node_generate)
    g.add_node("guardrail", node_guardrail)
    g.add_node("retry_patch", node_retry_patch)
    g.add_node("finalize", node_finalize)
    g.add_node("finalize_warn", node_finalize)

    g.add_edge(START, "extract")
    g.add_edge("extract", "validate")
    g.add_conditional_edges("validate", route_after_validate, {"fetch": "fetch_data", "end": "finalize"})
    g.add_edge("fetch_data", "generate")
    g.add_edge("generate", "guardrail")
    g.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"finalize": "finalize", "retry": "retry_patch", "finalize_warn": "finalize_warn"},
    )
    g.add_edge("retry_patch", "fetch_data")
    g.add_edge("finalize", END)
    g.add_edge("finalize_warn", END)
    return g


_compiled = None


def get_compiled_travel_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph().compile(checkpointer=MemorySaver())
    return _compiled


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
    """Запуск полного графа планирования (см. thread_id для изоляции сессий в MemorySaver)."""
    graph = get_compiled_travel_graph()
    init: dict[str, Any] = {
        "user_input": user_request,
        "conversation_context": conversation_context,
        "user_id": user_id,
        "default_origin_city": default_origin_city,
        "default_destination_city": default_destination_city,
        "default_origin_iata": default_origin_iata,
        "max_results": max_results,
        "guardrail_retries": 0,
        "budget_multiplier": 1.0,
        "max_guardrail_retries": int(os.getenv("TRAVEL_GUARDRAIL_MAX_RETRIES", "3")),
    }
    cfg = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(init, cfg)
