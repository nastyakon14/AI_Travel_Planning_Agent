"""
Инструменты агента по схемам из docs/specs/tools-APIs.md и workflow (diagrams/workflow.mmd).

Контракты:
- search_flights → упрощённый список рейсов (в т.ч. price_eur для сравнения).
- search_hotels → топ дешёвых отелей с id, name, price_per_night, rating.
- search_attractions → места в городе.
- extract_travel_requirements → структурированный TripQuery.
- check_travel_budget → guardrail по бюджету.
- generate_travel_itinerary → Markdown-маршрут по дням.

Длинные ответы API урезаются до top-N (по умолчанию 5) согласно docs/specs/tools-APIs.md.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

try:
    from .llm_observability import stream_plain_text
    from .travel_agent import (
        FAST_EXTRACTION_MODEL,
        OPENAI_API_KEY,
        OPENAI_URL,
        TripQuery,
        extract_trip_query,
        search_hotels_from_extracted,
        search_routes_from_extracted,
        suggest_city_attractions,
    )
except ImportError:  # pragma: no cover
    from llm_observability import stream_plain_text
    from travel_agent import (
        FAST_EXTRACTION_MODEL,
        OPENAI_API_KEY,
        OPENAI_URL,
        TripQuery,
        extract_trip_query,
        search_hotels_from_extracted,
        search_routes_from_extracted,
        suggest_city_attractions,
    )

# Грубые курсы к EUR для PoC (информативное сравнение; не для финансов).
_ROUGH_EUR_RATE: dict[str, float] = {
    "EUR": 1.0,
    "RUB": 0.0105,
    "USD": 0.92,
    "GBP": 1.17,
}

_DEFAULT_TOP_N = 5
_API_TIMEOUT_HINT = (
    "API недоступен или таймаут. Используйте ориентировочные значения и продолжайте план."
)


def _to_eur(amount: float, currency: str | None) -> float:
    c = (currency or "EUR").strip().upper()
    rate = _ROUGH_EUR_RATE.get(c, 1.0)
    return float(amount) * rate


def _flights_contract_rows(routes: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Схема tools-APIs: flight_id, airline, price_eur, time_dep, time_arr."""
    out: list[dict[str, Any]] = []
    for r in routes[:limit]:
        price = float(r.get("price") or 0)
        cur = str(r.get("currency") or "EUR")
        out.append(
            {
                "flight_id": r.get("flight_id"),
                "airline": r.get("airline"),
                "price_eur": round(_to_eur(price, cur), 2),
                "price_original": price,
                "currency_original": cur,
                "time_dep": r.get("departure_at"),
                "time_arr": r.get("return_at"),
                "transfers": int(r.get("transfers") or 0),
            }
        )
    return out


def _hotels_contract_rows(hotels: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Схема tools-APIs: hotel_id, name, price_per_night, rating (+ урезание полей)."""
    rows: list[dict[str, Any]] = []
    for h in hotels[:limit]:
        if h.get("is_search_portal_only"):
            continue
        rows.append(
            {
                "hotel_id": h.get("hotel_id"),
                "name": h.get("name"),
                "price_per_night": float(h.get("price_per_night") or 0),
                "currency": str(h.get("currency") or ""),
                "rating": h.get("rating"),
                "stars": h.get("stars"),
            }
        )
    return rows


@tool
def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str | None = None,
    currency: str = "EUR",
    max_results: int = _DEFAULT_TOP_N,
) -> dict[str, Any]:
    """
    Поиск авиабилетов через Travelpayouts по уже известным городам и датам.

    Используй, когда в запросе пользователя уже есть конкретные origin/destination и даты,
    или после `extract_travel_requirements`, если нужен отдельный вызов по структурированным полям.
    Не вызывай для свободного текста без параметров — сначала извлеки требования.

    Args:
        origin: Город или регион вылета (как в поиске).
        destination: Город назначения.
        departure_date: Дата вылета YYYY-MM-DD.
        return_date: Дата обратно; None или пусто = перелёт в одну сторону.
        currency: ISO валюта для цен в ответе API.
        max_results: Ограничение выдачи (1–10).

    Returns:
        Словарь с ключом `flights`: список объектов контракта (flight_id, airline, price_eur,
        time_dep, time_arr, transfers, …) и диагностическими полями при отсутствии маршрутов.
    """
    max_results = max(1, min(int(max_results), 10))
    user_request = (
        f"Перелёт {origin} — {destination}, вылет {departure_date}"
        + (f", обратно {return_date}" if return_date else ", в одну сторону")
    )
    extracted = TripQuery(
        origin_city=origin.strip(),
        destination_city=destination.strip(),
        departure_date=departure_date.strip(),
        return_date=return_date.strip() if return_date else None,
        currency=currency.strip().upper() if currency else "EUR",
        one_way=return_date is None or not str(return_date).strip(),
    )
    meta = {"source": "agent_tool.search_flights", "used_model": "structured_params"}
    try:
        raw = search_routes_from_extracted(
            user_request=user_request,
            extracted=extracted,
            extraction_meta=meta,
            default_origin_city=None,
            default_origin_iata=None,
            max_results=max_results,
        )
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "hint": _API_TIMEOUT_HINT,
            "flights": [],
            "mock_example_eur": 150.0,
        }

    routes = raw.get("routes") or []
    return {
        "flights": _flights_contract_rows(routes, max_results),
        "raw_routes_count": len(routes),
        "route_not_found_reason": raw.get("route_not_found_reason"),
        "route_not_found_message": raw.get("route_not_found_message"),
        "notices": raw.get("notices") or [],
    }


@tool
def search_hotels(
    city: str,
    check_in: str,
    check_out: str,
    currency: str = "EUR",
    max_results: int = _DEFAULT_TOP_N,
) -> dict[str, Any]:
    """
    Поиск отелей в городе на диапазон дат (Travelpayouts / партнёрский API).

    Применяй, когда известны город назначения и даты проживания. Для разрешения только страны
    или IATA сначала уточни город через пользователя или извлечённые поля TripQuery.

    Args:
        city: Название города проживания.
        check_in / check_out: YYYY-MM-DD.
        currency: Валюта цен.
        max_results: Сколько отелей вернуть (1–10).

    Returns:
        `hotels`: список с hotel_id, name, price_per_night, currency, rating, stars;
        при ошибках API — `error`, `hint`, пустой список.
    """
    max_results = max(1, min(int(max_results), 10))
    user_request = f"Отель в {city.strip()} с {check_in.strip()} по {check_out.strip()}"
    extracted = TripQuery(
        destination_city=city.strip(),
        departure_date=check_in.strip(),
        return_date=check_out.strip(),
        currency=currency.strip().upper() if currency else "EUR",
    )
    meta = {"source": "agent_tool.search_hotels", "used_model": "structured_params"}
    try:
        raw = search_hotels_from_extracted(
            user_request=user_request,
            extracted=extracted,
            extraction_meta=meta,
            default_destination_city=None,
            max_results=max_results,
        )
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "hint": _API_TIMEOUT_HINT,
            "hotels": [],
            "mock_example": {"hotel_id": "mock", "name": "Example Hotel", "price_per_night": 80.0, "rating": 8.0},
        }

    hotels = raw.get("hotels") or []
    return {
        "hotels": _hotels_contract_rows(hotels, max_results),
        "raw_hotels_count": len(hotels),
        "hotel_not_found_reason": raw.get("hotel_not_found_reason"),
        "hotel_not_found_message": raw.get("hotel_not_found_message"),
        "notices": raw.get("notices") or [],
    }


@tool
def search_attractions(
    city: str,
    country: str | None = None,
    max_items: int = _DEFAULT_TOP_N,
) -> dict[str, Any]:
    """
    Идеи достопримечательностей через LLM (без бронирований и без гарантии актуальности часов работы).

    Вызывай после того, как известен город (и при необходимости страна). Результаты — ориентир;
    предупреди пользователя перепроверить данные. Ошибки сети/LLM приходят в поле `error`.

    Args:
        city: Город или крупный регион для подсказок.
        country: Уточнение страны, если город неоднозначен.
        max_items: Число пунктов (3–15).

    Returns:
        `attractions`: attraction_id, name, description (summary); `model` — использованная модель.
    """
    max_items = max(3, min(int(max_items), 15))
    raw = suggest_city_attractions(city.strip(), country, max_items=max_items)
    items: list[dict[str, Any]] = []
    for i, a in enumerate(raw.get("attractions") or [], start=1):
        items.append(
            {
                "attraction_id": f"att-{i}",
                "name": a.get("name"),
                "description": a.get("summary"),
            }
        )
    return {
        "destination": raw.get("destination"),
        "country": raw.get("country"),
        "attractions": items[:max_items],
        "error": raw.get("error"),
        "model": raw.get("model"),
    }


@tool
def extract_travel_requirements(user_request: str) -> dict[str, Any]:
    """
    Извлекает из текста структурированный TripQuery (параметры перелёта/отеля) через LiteLLM.

    Передай сюда **текущую реплику пользователя**; если важна история диалога, заранее включи
    краткое резюме предыдущих реплик в строку запроса или вызывай `extract_trip_query` из кода
    оркестратора с аргументом `conversation_context`. В ответе — `requirements` и `extraction_meta`.
    """
    if not OPENAI_API_KEY:
        return {
            "error": "AGENTPLATFORM_API_KEY (or OPENAI_API_KEY) is not set",
            "requirements": None,
        }
    try:
        query, meta = extract_trip_query(user_request.strip(), return_metadata=True)
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "requirements": None,
        }
    return {
        "requirements": query.model_dump(),
        "extraction_meta": meta,
    }


@tool
def check_travel_budget(
    total_budget: float,
    flight_price: float,
    hotel_price_per_night: float,
    nights: int,
    currency: str = "EUR",
    passengers: int = 1,
) -> dict[str, Any]:
    """
    Guardrail по бюджету: сравнивает ориентировочную сумму перелёта + ночей в отеле с лимитом.

    Используй после получения цен из `search_flights` / `search_hotels`. `flight_price` — как в API
    (часто на всех пассажиров сразу); `hotel_price_per_night` умножается на `nights`.
    При превышении возвращает `retry_hint` для повторного поиска более дешёвых вариантов.

    Returns:
        within_budget, guardrail_pass, estimated_total, remaining, breakdown.
    """
    nights = max(1, int(nights))
    passengers = max(1, int(passengers))
    hotel_total = float(hotel_price_per_night) * nights
    # flight_price в выдаче часто уже на всех; если агент передаёт за человека — он укажет в notes
    estimated = float(flight_price) + hotel_total
    remaining = float(total_budget) - estimated
    within = estimated <= float(total_budget) + 1e-6
    return {
        "currency": currency.strip().upper(),
        "total_budget": float(total_budget),
        "estimated_total": round(estimated, 2),
        "remaining": round(remaining, 2),
        "within_budget": within,
        "guardrail_pass": within,
        "breakdown": {
            "flight_price": float(flight_price),
            "hotel_total": round(hotel_total, 2),
            "nights": nights,
            "passengers_note": passengers,
        },
        "retry_hint": (
            None
            if within
            else "Бюджет превышен. Найди отели с рейтингом ниже, но дешевле или более дешёвый перелёт."
        ),
    }


@tool
def generate_travel_itinerary(
    destination_city: str,
    days: int,
    attractions: list[str],
    user_notes: str = "",
) -> dict[str, Any]:
    """
    Генерирует туристический маршрут по дням (Markdown) через OpenAI-compatible API (ChatOpenAI).

    Вызывай в конце цепочки, когда известны город, длительность и (по возможности) список мест.
    Не подставляй выдуманные цены билетов/входов; предупреждай проверить часы работы локально.

    Args:
        destination_city: Город пребывания.
        days: Число дней (1–21).
        attractions: Названия точек из `search_attractions` или пустой список.
        user_notes: Доп. пожелания (темп, тип отдыха).

    Returns:
        itinerary_markdown, model, error при сбое.
    """
    if not OPENAI_API_KEY:
        return {"error": "API key not set", "itinerary_markdown": None}

    days = max(1, min(int(days), 21))
    model_name = os.getenv("TRIP_ITINERARY_MODEL", FAST_EXTRACTION_MODEL)
    att = ", ".join(attractions) if attractions else "без жёсткого списка"
    prompt = (
        f"Составь туристический маршрут на {days} дн. для города «{destination_city.strip()}». "
        f"Включи по возможности места: {att}. "
        f"{user_notes.strip() if user_notes else ''}\n"
        "Формат ответа: Markdown с заголовками ## День 1, ## День 2, … и маркированными списками дел. "
        "Без выдуманных цен и бронирований; укажи, что часы работы нужно проверить на месте."
    )
    try:
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.35,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_URL,
            timeout=float(os.getenv("LLM_TIMEOUT_SEC", "120")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
        )
        text, llm_metrics = stream_plain_text(
            llm, prompt, stage="travel_itinerary"
        )
        return {
            "itinerary_markdown": text,
            "model": model_name,
            "error": None,
            "llm_metrics": llm_metrics,
        }
    except Exception as exc:
        return {
            "itinerary_markdown": None,
            "model": model_name,
            "error": f"{type(exc).__name__}: {exc}",
        }


@tool
def validate_travel_constraints(
    destination_city: str,
    budget: float,
    days: int,
    currency: str = "EUR",
) -> dict[str, Any]:
    """
    Лёгкая проверка запроса до дорогих вызовов (без внешних API): бюджет vs длительность.

    Используй в начале графа для early exit, если сумма заведомо несоизмерима с числом дней
    (эвристика PoC, не юридическая или финансовая гарантия).

    Returns:
        ok, early_exit, message — при `early_exit=True` можно остановить сценарий и запросить уточнение.
    """
    days = max(1, int(days))
    budget = float(budget)
    min_suggested = 30.0 * days if currency.upper() == "EUR" else 3000.0 * days
    ok = budget >= min_suggested * 0.15
    return {
        "ok": ok,
        "destination_city": destination_city.strip(),
        "budget": budget,
        "currency": currency.upper(),
        "days": days,
        "early_exit": not ok,
        "message": (
            None
            if ok
            else (
                "Заявленный бюджет выглядит недостаточным для поездки такой длительности "
                f"в «{destination_city}». Увеличьте бюджет или сократите число дней."
            )
        ),
    }


# Список инструментов для bind_tools / LangGraph
AGENT_TOOL_FUNCTIONS = [
    search_flights,
    search_hotels,
    search_attractions,
    extract_travel_requirements,
    check_travel_budget,
    generate_travel_itinerary,
    validate_travel_constraints,
]


def get_extended_tool_list() -> list[Any]:
    """
    Контрактные тулы + NL-обёртки из travel_agent (search_routes_from_text и др.)
    для совместимости с существующим Streamlit / демо.
    """
    try:
        from .travel_agent import (
            search_hotels_from_text,
            search_routes_from_text,
            search_travel_from_text,
        )
    except ImportError:  # pragma: no cover
        from travel_agent import (
            search_hotels_from_text,
            search_routes_from_text,
            search_travel_from_text,
        )

    return [
        *AGENT_TOOL_FUNCTIONS,
        search_routes_from_text,
        search_hotels_from_text,
        search_travel_from_text,
    ]
