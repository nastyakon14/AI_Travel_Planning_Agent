"""LLM-powered extraction + Aviasales route search tool"""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from .guardrails import sanitize_user_input
    from .llm_observability import stream_structured_output
except ImportError:  # pragma: no cover
    from guardrails import sanitize_user_input
    from llm_observability import stream_structured_output

try:
    from .aviatickets import TravelPayoutsClient, filter_routes_by_budget
    from .hotels import (
        TravelPayoutsHotelsClient,
        filter_hotels_by_budget,
        sort_hotels_by_price_then_rating,
    )
except ImportError:  # pragma: no cover
    from aviatickets import TravelPayoutsClient, filter_routes_by_budget
    from hotels import (
        TravelPayoutsHotelsClient,
        filter_hotels_by_budget,
        sort_hotels_by_price_then_rating,
    )

# месяцы для маппинга месяца и его порядкового номера
MONTHS_RU: dict[str, int] = {
    "январ": 1,
    "феврал": 2,
    "март": 3,
    "апрел": 4,
    "мая": 5,
    "май": 5,
    # предложный/дательный падежи («в мае», «к маю») — иначе месяц не находился по подстроке
    "мае": 5,
    "маю": 5,
    "июн": 6,
    "июл": 7,
    "август": 8,
    "сентябр": 9,
    "октябр": 10,
    "ноябр": 11,
    "декабр": 12,
}

MONTHS_EN: dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

# маппинг символов валют в ISO коды валют
SYMBOL_TO_CURRENCY = {"€": "EUR", "$": "USD", "₽": "RUB"}

# Каскад моделей: имена должны совпадать с тем, что отдаёт ваш шлюз (см. GET {base}/v1/models).
# Дефолты — типичные имена OpenAI; при необходимости задайте TRIP_EXTRACTION_* в .env (например алиас шлюза).
FAST_EXTRACTION_MODEL = os.getenv("TRIP_EXTRACTION_FAST_MODEL", "openai/gpt-4o-mini")
STRONG_EXTRACTION_MODEL = os.getenv("TRIP_EXTRACTION_STRONG_MODEL", "openai/gpt-4o")
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "120"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
EXTRACTION_QUALITY_THRESHOLD = float(os.getenv("TRIP_EXTRACTION_QUALITY_THRESHOLD", "0.7"))
OPENAI_URL = os.getenv("AGENTPLATFORM_API_BASE", "https://litellm.tokengate.ru/v1")
OPENAI_API_KEY = (
    os.getenv("AGENTPLATFORM_API_KEY")
    or os.getenv("AGENTPLATFORM_KEY")
    or os.getenv("OPENAI_API_KEY")
)


def _make_chat_llm(model: str, *, temperature: float = 0.0) -> ChatOpenAI:
    """OpenAI-compatible HTTP API (AGENTPLATFORM_API_BASE), без LiteLLM."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_URL,
        timeout=LLM_TIMEOUT_SEC,
        max_retries=LLM_MAX_RETRIES,
    )


def _complexity_hint(user_text: str, context: str | None) -> float:
    """0..1 — эвристика сложности запроса (каскад моделей)."""
    t = f"{user_text}\n{context or ''}"
    score = 0.0
    if len(t) > 1500:
        score += 0.35
    if t.count(",") > 8 or t.count(";") > 4:
        score += 0.2
    if re.search(
        r"\b(виза|visa|шенген|schengen|нескольк\w* город|multi-?city|стоповер|пересадк\w* в)\b",
        t,
        re.I,
    ):
        score += 0.35
    if re.search(r"\b\d+\s*(дн|ноч|day|night)", t, re.I) and len(t) > 600:
        score += 0.15
    return min(1.0, score)


def choose_extraction_model(user_text: str, context: str | None) -> str:
    if os.getenv("EXTRACTION_COMPLEX_FIRST", "0").lower() in ("1", "true", "yes"):
        if _complexity_hint(user_text, context) >= 0.45:
            return STRONG_EXTRACTION_MODEL
    return FAST_EXTRACTION_MODEL


def extraction_model_order(user_text: str, context: str | None) -> list[str]:
    if FAST_EXTRACTION_MODEL == STRONG_EXTRACTION_MODEL:
        return [FAST_EXTRACTION_MODEL]
    if os.getenv("EXTRACTION_COMPLEX_FIRST", "0").lower() in ("1", "true", "yes"):
        if choose_extraction_model(user_text, context) == STRONG_EXTRACTION_MODEL:
            return [STRONG_EXTRACTION_MODEL, FAST_EXTRACTION_MODEL]
    return [FAST_EXTRACTION_MODEL, STRONG_EXTRACTION_MODEL]


class TripQuery(BaseModel):
    """Structured user intent extracted from a natural language request."""

    origin_city: str | None = Field(default=None, description="Origin city in text form")
    origin_country: str | None = Field(default=None, description="Origin country if city is absent")
    origin_iata: str | None = Field(default=None, description="Origin IATA code if explicit")
    destination_city: str | None = Field(default=None, description="Destination city")
    destination_country: str | None = Field(default=None, description="Destination country")
    destination_iata: str | None = Field(default=None, description="Destination IATA if explicit")
    departure_date: str | None = Field(default=None, description="YYYY-MM-DD if explicit")
    return_date: str | None = Field(default=None, description="YYYY-MM-DD if explicit")
    departure_month: int | None = Field(default=None, ge=1, le=12, description="Month number if only month is specified")
    departure_year: int | None = Field(default=None, ge=2024, le=2100, description="Year if known")
    trip_days: int | None = Field(default=None, ge=1, le=60, description="Trip duration in days")
    passengers: int = Field(default=1, ge=1, le=9)
    adults: int | None = Field(default=None, ge=1, le=9)
    children: int | None = Field(default=0, ge=0, le=8)
    infants: int | None = Field(default=0, ge=0, le=8)
    budget: float | None = Field(default=None, ge=0, description="Total budget in the selected currency")
    currency: str = Field(default="RUB", description="ISO currency code")
    direct_only: bool | None = Field(default=None, description="True if no layovers are allowed")
    max_stops: int | None = Field(default=None, ge=0, le=3)
    cabin_class: str | None = Field(default=None, description="Economy, Business, First")
    include_baggage: bool | None = Field(default=None)
    one_way: bool | None = Field(default=None)
    flexible_dates: bool | None = Field(default=None)
    preferred_departure_time: str | None = Field(
        default=None, description="morning/day/evening/night if requested"
    )
    preferred_airlines: list[str] | None = Field(default=None)
    excluded_airlines: list[str] | None = Field(default=None)

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency_symbol(cls, value: Any) -> str:
        if not value:
            return "RUB"
        clean = str(value).strip().upper()
        return SYMBOL_TO_CURRENCY.get(clean, clean)

    @field_validator("origin_iata", "destination_iata", mode="before")
    @classmethod
    def normalize_iata(cls, value: Any) -> str | None:
        if value is None:
            return None
        clean = str(value).strip().upper()
        return clean or None

    @field_validator(
        "origin_city",
        "origin_country",
        "destination_city",
        "destination_country",
        "cabin_class",
        "preferred_departure_time",
        mode="before",
    )
    @classmethod
    def strip_text_fields(cls, value: Any) -> str | None:
        if value is None:
            return None
        clean = str(value).strip()
        return clean or None

    @model_validator(mode="after")
    def apply_consistency_rules(self) -> "TripQuery":
        if self.max_stops is not None and self.max_stops == 0:
            self.direct_only = True
        if self.adults is not None:
            self.passengers = max(
                1, self.adults + (self.children or 0) + (self.infants or 0)
            )
        if self.one_way is True:
            self.return_date = None
        return self


class AttractionItem(BaseModel):
    """Одна достопримечательность или место для прогулки."""

    name: str = Field(..., description="Название места")
    summary: str = Field(
        ...,
        description="1–2 предложения по-русски: что это и почему стоит посетить",
    )


class CityAttractions(BaseModel):
    """Список мест в городе/регионе для туриста."""

    items: list[AttractionItem] = Field(default_factory=list)


def destination_label_for_attractions(
    query: TripQuery,
    default_destination_city: str | None,
) -> tuple[str | None, str | None]:
    """
    Город/направление для подсказок «куда сходить».
    Возвращает (подпись места, страна или None).
    """
    if query.destination_city and str(query.destination_city).strip():
        return str(query.destination_city).strip(), query.destination_country
    if default_destination_city and str(default_destination_city).strip():
        return str(default_destination_city).strip(), query.destination_country
    if query.destination_country and str(query.destination_country).strip():
        return str(query.destination_country).strip(), query.destination_country
    if query.destination_iata and str(query.destination_iata).strip():
        return str(query.destination_iata).strip(), query.destination_country
    return None, None


def suggest_city_attractions(
    city: str,
    country: str | None = None,
    *,
    max_items: int = 8,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Подбор популярных достопримечательностей через LLM (структурированный ответ).
    Без отдельного внешнего API карт; данные нужно перепроверять перед визитом.
    """
    if os.getenv("SKIP_CITY_ATTRACTIONS", "").strip().lower() in ("1", "true", "yes"):
        return {
            "destination": city,
            "country": country,
            "attractions": [],
            "model": None,
            "skipped": True,
            "error": "disabled_by_SKIP_CITY_ATTRACTIONS",
        }

    if not OPENAI_API_KEY:
        return {
            "destination": city,
            "country": country,
            "attractions": [],
            "model": None,
            "error": "AGENTPLATFORM_API_KEY (or OPENAI_API_KEY) is not set",
        }

    max_items = max(3, min(int(max_items), 15))
    model_name = model or os.getenv("TRIP_ATTRACTIONS_MODEL", FAST_EXTRACTION_MODEL)

    location = city.strip()
    if country and str(country).strip():
        location = f"{city.strip()}, {str(country).strip()}"

    prompt = (
        f"Ты эксперт по туризму. Для места «{location}» составь ровно {max_items} пунктов: "
        "популярные достопримечательности и куда можно сходить туристу "
        "(музеи, парки, исторические кварталы, смотровые точки, храмы, рынки и т.п.). "
        "Указывай только реально существующие объекты, без выдуманных названий. "
        "Каждый пункт: поле name — короткое название, поле summary — 1–2 предложения по-русски, "
        "зачем туда идти.\n"
        f"Место: {location}"
    )
    system = (
        "Ответь одним JSON-объектом по схеме CityAttractions: поле items — массив объектов "
        "{name: string, summary: string}. Без markdown и пояснений вне JSON."
    )

    try:
        llm = _make_chat_llm(model_name, temperature=0.25)
        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        parsed, llm_metrics = stream_structured_output(
            llm,
            CityAttractions,
            messages,
            stage="city_attractions",
        )
        if not isinstance(parsed, CityAttractions):
            parsed = CityAttractions.model_validate(parsed)
        items_out = [
            {"name": it.name.strip(), "summary": it.summary.strip()}
            for it in parsed.items
            if it.name and it.summary
        ][:max_items]
        return {
            "destination": city,
            "country": country,
            "attractions": items_out,
            "model": model_name,
            "error": None,
            "llm_metrics": llm_metrics,
        }
    except Exception as exc:  # pragma: no cover - сеть / провайдер
        return {
            "destination": city,
            "country": country,
            "attractions": [],
            "model": model_name,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _normalize_currency(value: str | None) -> str:
    if not value:
        return "RUB"
    clean = value.strip().upper()
    if clean in SYMBOL_TO_CURRENCY:
        return SYMBOL_TO_CURRENCY[clean]
    if len(clean) == 1 and clean in SYMBOL_TO_CURRENCY:
        return SYMBOL_TO_CURRENCY[clean]
    if clean == "RUR":
        return "RUB"
    return clean


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _detect_month_from_text(text: str) -> int | None:
    lower = text.lower()
    for token, month in MONTHS_RU.items():
        if token in lower:
            return month
    for token, month in MONTHS_EN.items():
        if token in lower:
            return month
    return None


def _next_month_first_day(today: date) -> date:
    if today.month == 12:
        return date(today.year + 1, 1, 1)
    return date(today.year, today.month + 1, 1)


def _end_of_month(today: date) -> date:
    if today.month == 12:
        return date(today.year, 12, 31)
    first_day_next_month = date(today.year, today.month + 1, 1)
    return first_day_next_month - timedelta(days=1)


def _next_weekend(today: date, prefer_next: bool = False) -> tuple[date, date]:
    weekday = today.weekday()  # Monday=0 ... Sunday=6
    if weekday == 5:
        saturday = today
    elif weekday == 6:
        saturday = today + timedelta(days=6)
    else:
        saturday = today + timedelta(days=(5 - weekday))

    if prefer_next:
        saturday = saturday + timedelta(days=7)

    return saturday, saturday + timedelta(days=1)


def _upcoming_annual_date(today: date, month: int, day: int) -> date:
    this_year = date(today.year, month, day)
    return this_year if this_year >= today else date(today.year + 1, month, day)


def _detect_season(text: str) -> str | None:
    season_tokens = {
        "winter": ("зима", "зимой", "winter"),
        "spring": ("весна", "весной", "spring"),
        "summer": ("лето", "летом", "summer"),
        "autumn": ("осень", "осенью", "autumn", "fall"),
    }
    for season, tokens in season_tokens.items():
        if any(token in text for token in tokens):
            return season
    return None


def _upcoming_season_date(today: date, season: str, force_next: bool = False) -> date:
    season_config = {
        "winter": {"months": (12, 1, 2), "start_month": 12},
        "spring": {"months": (3, 4, 5), "start_month": 3},
        "summer": {"months": (6, 7, 8), "start_month": 6},
        "autumn": {"months": (9, 10, 11), "start_month": 9},
    }
    cfg = season_config[season]
    in_season = today.month in cfg["months"]
    if in_season and not force_next:
        return today

    start_month = cfg["start_month"]
    target_year = today.year if today.month < start_month else today.year + 1
    return date(target_year, start_month, 1)


def _infer_departure_from_relative_text(
    text: str,
    today: date,
    one_way: bool,
) -> tuple[date | None, date | None]:
    departure: date | None = None
    return_date: date | None = None

    if "послезавтра" in text or "day after tomorrow" in text:
        departure = today + timedelta(days=2)
    elif re.search(r"\bзавтра\b|\btomorrow\b", text):
        departure = today + timedelta(days=1)
    elif re.search(r"\bсегодня\b|\btoday\b", text):
        departure = today

    in_days_match = re.search(
        r"через\s+(\d+)\s*(?:дн|дня|дней)\b|in\s+(\d+)\s+days?\b",
        text,
    )
    if departure is None and in_days_match:
        days = int(in_days_match.group(1) or in_days_match.group(2))
        departure = today + timedelta(days=max(0, days))

    if departure is None and re.search(r"\bчерез\s+недел\w*\b|\bin a week\b", text):
        departure = today + timedelta(days=7)

    if departure is None and re.search(r"\bмайск\w*\b|may holidays", text):
        may_start = _upcoming_annual_date(today, 5, 1)
        departure = may_start
        if not one_way:
            return_date = may_start + timedelta(days=8)

    if departure is None and re.search(r"\bновогодн\w*\b|new year holidays", text):
        current_year_window_start = date(today.year, 12, 30)
        if today <= current_year_window_start:
            departure = current_year_window_start
            if not one_way:
                return_date = date(today.year + 1, 1, 7)
        else:
            departure = date(today.year + 1, 12, 30)
            if not one_way:
                return_date = date(today.year + 2, 1, 7)

    weekend_next = bool(re.search(r"следующ\w*\s+выходн\w*|next weekend", text))
    weekend_any = bool(re.search(r"\bвыходн\w*\b|\bweekend\b", text))
    if departure is None and (weekend_next or weekend_any):
        saturday, sunday = _next_weekend(today, prefer_next=weekend_next)
        departure = saturday
        if not one_way:
            return_date = sunday

    if departure is None and re.search(r"следующ\w*\s+месяц\w*|next month", text):
        departure = _next_month_first_day(today)

    if departure is None and re.search(r"в\s+конце\s+месяц\w*|end of (the )?month", text):
        departure = _end_of_month(today)

    if departure is None and re.search(r"следующ\w*\s+недел\w*|next week", text):
        days_until_next_monday = (7 - today.weekday()) % 7
        if days_until_next_monday == 0:
            days_until_next_monday = 7
        departure = today + timedelta(days=days_until_next_monday)

    if departure is None:
        season = _detect_season(text)
        if season:
            force_next = bool(
                re.search(
                    r"следующ\w*\s+(?:зим|весн|лет|осен)\w*|next\s+(?:winter|spring|summer|autumn|fall)",
                    text,
                )
            )
            departure = _upcoming_season_date(today, season=season, force_next=force_next)

    return departure, return_date


def _infer_trip_days_from_text(text: str) -> int | None:
    """Длительность поездки/проживания из текста, если LLM не заполнил trip_days."""
    t = text.lower()
    if re.search(r"\bна\s+недел\w*\b|\bна\s+7\s+дн", t) or re.search(r"\bone\s*week\b", t):
        return 7
    m = re.search(r"\bна\s+(\d{1,2})\s*(?:дн|дня|дней)\b", t)
    if m:
        return max(1, min(60, int(m.group(1))))
    return None


def _infer_dates(user_text: str, query: TripQuery) -> tuple[date | None, date | None]:
    today = date.today()
    lower = user_text.lower()

    departure = _parse_iso_date(query.departure_date)
    return_date = _parse_iso_date(query.return_date)

    if departure is None:
        relative_departure, relative_return = _infer_departure_from_relative_text(
            text=lower,
            today=today,
            one_way=bool(query.one_way),
        )
        if relative_departure:
            departure = relative_departure
        if relative_return and return_date is None:
            return_date = relative_return

    if departure is None:
        month = query.departure_month or _detect_month_from_text(user_text)
        if month:
            year = query.departure_year or today.year
            if year == today.year and month < today.month:
                year += 1
            departure = date(year, month, 1)

    trip_days_effective = query.trip_days or _infer_trip_days_from_text(user_text)
    if return_date is None and departure and trip_days_effective:
        return_date = departure + timedelta(days=trip_days_effective)

    return departure, return_date


def _mentions_budget(text: str) -> bool:
    return bool(
        re.search(r"\b\d[\d\s,.]*\s?(?:eur|usd|rub|р|руб|доллар|евро)\b", text)
        or re.search(r"[\$€₽]\s?\d", text)
        or re.search(r"\d\s?[\$€₽]", text)
    )


def _mentions_duration(text: str) -> bool:
    return bool(re.search(r"\b\d+\s?(?:дн(?:я|ей)?|day|days)\b", text))


def _mentions_date(text: str) -> bool:
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", text):
        return True
    if re.search(r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b", text):
        return True
    if re.search(
        r"\b(сегодня|завтра|послезавтра|today|tomorrow|day after tomorrow|weekend|выходн|"
        r"next month|следующ\w*\s+месяц|next week|следующ\w*\s+недел|"
        r"через\s+\d+\s*(?:дн|дня|дней)|in\s+\d+\s+days?|через\s+недел\w*|in a week|"
        r"майск\w*|may holidays|новогодн\w*|new year holidays|"
        r"в\s+конце\s+месяц\w*|end of (the )?month)\b",
        text,
    ):
        return True
    return any(token in text for token in MONTHS_RU) or any(token in text for token in MONTHS_EN)


def _score_extraction_quality(user_text: str, query: TripQuery) -> float:
    # Легкая эвристика: насколько извлечение покрывает явные требования пользователя.
    lower = user_text.lower()
    checks: list[bool] = []

    # Направление поездки считаем обязательным минимумом.
    checks.append(bool(query.destination_city or query.destination_iata or query.destination_country))

    if re.search(r"\b(из|from|origin)\b", lower):
        checks.append(bool(query.origin_city or query.origin_iata or query.origin_country))
    if _mentions_budget(lower):
        checks.append(query.budget is not None)
    if _mentions_date(lower):
        checks.append(bool(query.departure_date or query.departure_month))
    if _mentions_duration(lower):
        checks.append(bool(query.trip_days or query.return_date))
    if re.search(r"\b(без пересадок|без пересад|прямой|direct)\b", lower):
        checks.append(bool(query.direct_only))
    if re.search(r"\b(в одну сторону|one[- ]?way)\b", lower):
        checks.append(query.one_way is True)

    populated = sum(
        bool(value)
        for value in (
            query.origin_city,
            query.origin_iata,
            query.destination_city,
            query.destination_iata,
            query.destination_country,
            query.departure_date,
            query.departure_month,
            query.trip_days,
            query.budget,
            query.direct_only,
            query.one_way,
        )
    )

    if not checks:
        return round(min(1.0, populated / 3), 2)

    score = sum(checks) / len(checks)
    # Защита от "формально валидного", но почти пустого результата.
    if populated < 2:
        score = min(score, 0.4)
    return round(score, 2)


def _invoke_extractor_model(
    user_text: str,
    model: str,
    *,
    conversation_context: str | None = None,
    user_id: str | None = None,
) -> tuple[TripQuery, dict[str, Any]]:
    """Извлечение TripQuery через OpenAI-compatible API (ChatOpenAI → AGENTPLATFORM_API_BASE)."""
    _ = user_id  # зарезервировано для трейсинга / метаданных
    llm = _make_chat_llm(model, temperature=0)
    user_block = user_text.strip()
    if conversation_context:
        user_block = (
            "Previous conversation (for reference only; current request takes priority):\n"
            f"{conversation_context[:8000]}\n\n---\nCurrent user message:\n{user_text.strip()}"
        )
    system = (
        "Extract complete flight-search parameters from the user text into schema fields. "
        "Use ISO dates YYYY-MM-DD when concrete dates are explicitly present. "
        "If data is missing, keep null. "
        "Do not invent origin city or dates. "
        "Parse passengers, budget, currency, layover preferences, and date flexibility when present."
    )
    messages = [SystemMessage(content=system), HumanMessage(content=user_block)]
    result, llm_metrics = stream_structured_output(
        llm,
        TripQuery,
        messages,
        stage="trip_extraction",
    )
    query = result if isinstance(result, TripQuery) else TripQuery.model_validate(result)
    return query, {"model": model, "llm_metrics": llm_metrics}


def extract_trip_query(
    user_text: str,
    return_metadata: bool = False,
    *,
    conversation_context: str | None = None,
    user_id: str | None = None,
) -> TripQuery | tuple[TripQuery, dict[str, Any]]:
    """Извлечение TripQuery через ChatOpenAI (OpenAI-compatible URL) с каскадом моделей."""
    if not OPENAI_API_KEY:
        raise ValueError(
            "AGENTPLATFORM_API_KEY (or AGENTPLATFORM_KEY / OPENAI_API_KEY fallback) is not set."
        )
    user_text = sanitize_user_input(user_text)

    attempts: list[dict[str, Any]] = []
    models = extraction_model_order(user_text, conversation_context)

    def run_attempt(model: str) -> dict[str, Any]:
        try:
            query, _llm_meta = _invoke_extractor_model(
                user_text,
                model=model,
                conversation_context=conversation_context,
                user_id=user_id,
            )
            score = _score_extraction_quality(user_text, query)
            return {
                "model": model,
                "query": query,
                "score": score,
                "error": None,
                "llm_metrics": _llm_meta.get("llm_metrics"),
            }
        except Exception as exc:  # pragma: no cover - network/provider failures
            return {"model": model, "query": None, "score": 0.0, "error": str(exc)}

    selected: dict[str, Any] | None = None
    fallback_reason: str | None = None

    for i, model in enumerate(models):
        att = run_attempt(model)
        attempts.append(att)
        if att["error"]:
            if i == 0:
                fallback_reason = f"model_failed_{att['model']}"
            continue
        assert att["query"] is not None
        if att["score"] >= EXTRACTION_QUALITY_THRESHOLD:
            selected = att
            break
        if selected is None or att["score"] > selected["score"]:
            selected = att
        if i == 0 and att["score"] < EXTRACTION_QUALITY_THRESHOLD:
            fallback_reason = f"low_quality_{att['score']:.2f}"

    if not selected or selected.get("query") is None:
        errors = "; ".join(
            f"{attempt['model']}: {attempt['error']}" for attempt in attempts if attempt["error"]
        )
        raise ValueError(f"Could not extract trip query. Errors: {errors}")

    sel_metrics = selected.get("llm_metrics")
    metadata = {
        "used_model": selected["model"],
        "used_fallback": len(attempts) > 1,
        "fallback_reason": fallback_reason,
        "extraction_llm_metrics": sel_metrics,
        "attempts": [
            {
                "model": attempt["model"],
                "score": attempt["score"],
                "error": attempt["error"],
                "llm_metrics": attempt.get("llm_metrics"),
            }
            for attempt in attempts
        ],
    }

    if return_metadata:
        return selected["query"], metadata
    return selected["query"]


def _resolve_iata(
    client: TravelPayoutsClient,
    iata: str | None,
    city: str | None,
    country: str | None = None,
    fallback_country: str | None = None,
) -> tuple[str | None, str | None]:
    if iata:
        return iata.strip().upper(), None

    if city:
        resolved = client.resolve_place(city, locale="ru")
        if resolved and resolved.get("iata"):
            return str(resolved["iata"]).upper(), None

    if country:
        resolved = client.resolve_place(country, locale="en")
        if resolved and resolved.get("iata"):
            return str(resolved["iata"]).upper(), (
                "IATA was resolved from country name. Confirm city for better precision."
            )

    if fallback_country:
        resolved = client.resolve_place(fallback_country, locale="en")
        if resolved and resolved.get("iata"):
            return str(resolved["iata"]).upper(), (
                "Destination was resolved from country name. "
                "Confirm destination city for better accuracy."
            )

    return None, "Could not resolve IATA code."


def _resolve_hotel_destination_text(
    query: TripQuery,
    default_destination_city: str | None,
) -> tuple[str | None, str | None]:
    if query.destination_city:
        return query.destination_city, None
    if default_destination_city:
        return default_destination_city, None
    if query.destination_country:
        return query.destination_country, (
            "Destination city is missing, country is used for hotel search (less precise)."
        )
    if query.destination_iata:
        return query.destination_iata, (
            "Destination city is missing, IATA code is used for hotel search."
        )
    return None, "Destination city is not provided."


def detect_search_scope(user_request: str) -> str:
    """Infer whether user asks for flights, hotels, or a package/tour (both)."""
    lower = user_request.lower()

    tour_requested = bool(
        re.search(
            r"\b(тур|туры|турпакет|package tour|package trip|vacation package)\b",
            lower,
        )
    )
    flight_requested = bool(
        re.search(
            r"\b(билет\w*|авиабилет\w*|рейс\w*|перелет\w*|flight\w*|ticket\w*|airfare)\b",
            lower,
        )
    )
    hotel_requested = bool(
        re.search(
            r"\b(отел\w*|гостин\w*|апартамент\w*|жиль\w*|"
            r"hotel\w*|accommodation|stay|"
            r"проживан\w*|ночлег\w*|хостел\w*|квартир\w*)\b",
            lower,
        )
    )

    if tour_requested or (flight_requested and hotel_requested):
        return "both"
    if hotel_requested:
        return "hotels"
    return "flights"


def search_routes_from_extracted(
    user_request: str,
    extracted: TripQuery,
    extraction_meta: dict[str, Any] | None = None,
    default_origin_city: str | None = None,
    default_origin_iata: str | None = None,
    max_results: int = 5,
) -> dict[str, Any]:
    """Run Travelpayouts search using already extracted TripQuery."""
    api_token = os.getenv("TRAVELPAYOUTS_API_TOKEN") or os.getenv("TRAVELPAYOUTS_API_KEY")
    if not api_token:
        raise ValueError("TRAVELPAYOUTS_API_TOKEN (or TRAVELPAYOUTS_API_KEY) is not set.")

    extraction_meta = extraction_meta or {}
    currency = _normalize_currency(extracted.currency)
    passengers = max(1, extracted.passengers)
    departure_date, return_date = _infer_dates(user_request, extracted)

    client = TravelPayoutsClient(api_token=api_token)
    notices: list[str] = []

    origin_iata, origin_warning = _resolve_iata(
        client=client,
        iata=extracted.origin_iata or default_origin_iata,
        city=extracted.origin_city or default_origin_city,
        country=extracted.origin_country,
    )
    if origin_warning:
        notices.append(f"origin: {origin_warning}")

    destination_iata, destination_warning = _resolve_iata(
        client=client,
        iata=extracted.destination_iata,
        city=extracted.destination_city,
        country=extracted.destination_country,
        fallback_country=extracted.destination_country,
    )
    if destination_warning:
        notices.append(f"destination: {destination_warning}")

    if not origin_iata:
        notices.append("origin: not found in request, pass default_origin_city/default_origin_iata.")
    if not destination_iata:
        notices.append("destination: could not be resolved.")
    if not departure_date:
        notices.append("departure_date: not found, API will search closest available prices.")
    if not origin_iata or not destination_iata:
        if not origin_iata and not destination_iata:
            not_found_message = (
                "Не удалось сопоставить города вылета и прилёта с кодами аэропортов IATA "
                "(трёхбуквенные коды вроде MOW, LED). Без них поиск билетов не запускается. "
                "Напишите полные названия городов и стран или укажите IATA явно; при необходимости "
                "задайте город вылета в настройках по умолчанию."
            )
        elif not origin_iata:
            not_found_message = (
                "Не удалось определить аэропорт вылета (IATA). "
                "Уточните город и страну отправления или укажите код аэропорта (например DME, SVO)."
            )
        else:
            not_found_message = (
                "Не удалось определить аэропорт прилёта (IATA). "
                "Уточните город и страну назначения или укажите код аэропорта."
            )
        return {
            "user_request": user_request,
            "extracted": extracted.model_dump(),
            "extraction_meta": extraction_meta,
            "route_not_found_reason": "missing_iata",
            "route_not_found_message": not_found_message,
            "notices": notices,
            "routes": [],
        }

    direct_only = extracted.direct_only
    one_way = extracted.one_way if extracted.one_way is not None else return_date is None

    routes = client.search_flights(
        origin_iata=origin_iata,
        destination_iata=destination_iata,
        departure_date=departure_date,
        return_date=return_date,
        currency=currency,
        direct_only=direct_only,
        one_way=one_way,
        limit=max(10, max_results * 4),
    )

    route_not_found_reason: str | None = None
    route_not_found_message: str | None = None
    if not routes:
        route_not_found_reason = "no_offers_from_api"
        route_not_found_message = (
            "Сервис цен на авиабилеты вернул пустой список: для этого направления и дат "
            "сейчас нет предложений в базе. Так бывает при редких маршрутах, отсутствии рейсов в выбранные дни "
            "или если дата слишком далеко вперёд. Что попробовать: сдвинуть даты на несколько дней, "
            "поискать вылет из крупного хаба рядом, проверить «туда-обратно» и «в одну сторону», "
            "или ослабить ограничения в запросе (например только прямой рейс или лимит по цене)."
        )

    if direct_only:
        direct_routes = [route for route in routes if int(route.get("transfers") or 0) == 0]
        if routes and not direct_routes:
            route_not_found_reason = "no_direct_flights"
            route_not_found_message = (
                "В выдаче есть рейсы с пересадками, но прямых рейсов на эти даты нет "
                "(или они не попали в ответ API). Снимите ограничение «только прямой рейс» "
                "или выберите другие даты — тогда можно увидеть варианты с пересадкой."
            )
        routes = direct_routes

    routes_before_budget_filter = len(routes)
    routes = filter_routes_by_budget(routes, extracted.budget, passengers)
    if (
        routes_before_budget_filter > 0
        and extracted.budget is not None
        and len(routes) == 0
    ):
        route_not_found_reason = "budget_too_low"
        route_not_found_message = (
            "Подходящие по датам и маршруту варианты есть, но цена каждого выше заданного бюджета. "
            "Бюджет обычно сравнивается с полной стоимостью на всех пассажиров и на выбранный тип билета "
            "(туда-обратно или в одну сторону). Попробуйте увеличить лимит или убрать ограничение по бюджету."
        )

    routes = routes[: max(1, max_results)]
    if routes:
        route_not_found_reason = None
        route_not_found_message = None
    elif route_not_found_reason is None:
        route_not_found_reason = "no_routes_after_filters"
        route_not_found_message = (
            "Подходящих маршрутов в итоге не осталось при текущих датах, направлении и ограничениях "
            "(в т.ч. «только прямой» и бюджет). Проверьте текст запроса и фильтры слева: сдвиньте даты, "
            "увеличьте бюджет или разрешите рейсы с пересадкой."
        )

    return {
        "user_request": user_request,
        "extracted": extracted.model_dump(),
        "extraction_meta": extraction_meta,
        "route_not_found_reason": route_not_found_reason,
        "route_not_found_message": route_not_found_message,
        "search_params": {
            "origin_iata": origin_iata,
            "destination_iata": destination_iata,
            "departure_date": departure_date.isoformat() if departure_date else None,
            "return_date": return_date.isoformat() if return_date else None,
            "currency": currency,
            "passengers": passengers,
            "adults": extracted.adults,
            "children": extracted.children,
            "infants": extracted.infants,
            "direct_only": direct_only,
            "max_stops": extracted.max_stops,
            "one_way": one_way,
            "flexible_dates": extracted.flexible_dates,
            "cabin_class": extracted.cabin_class,
            "preferred_departure_time": extracted.preferred_departure_time,
            "preferred_airlines": extracted.preferred_airlines,
            "excluded_airlines": extracted.excluded_airlines,
            "budget": extracted.budget,
        },
        "notices": notices,
        "routes": routes,
    }


def search_hotels_from_extracted(
    user_request: str,
    extracted: TripQuery,
    extraction_meta: dict[str, Any] | None = None,
    default_destination_city: str | None = None,
    max_results: int = 5,
) -> dict[str, Any]:
    """Run hotel search using already extracted TripQuery."""
    api_token = os.getenv("TRAVELPAYOUTS_API_TOKEN") or os.getenv("TRAVELPAYOUTS_API_KEY")
    marker = os.getenv("TRAVELPAYOUTS_MARKER") or os.getenv("TRAVELPAYOUTS_HOTEL_MARKER")

    extraction_meta = extraction_meta or {}
    destination_text, destination_warning = _resolve_hotel_destination_text(
        query=extracted,
        default_destination_city=default_destination_city,
    )
    check_in, check_out = _infer_dates(user_request, extracted)

    notices: list[str] = []
    if destination_warning:
        notices.append(f"destination: {destination_warning}")
    if not check_in:
        notices.append("check_in: could not infer check-in date from request.")

    if not destination_text or not check_in:
        reason = "missing_destination_or_dates"
        if not destination_text and not check_in:
            message = (
                "Недостаточно данных для поиска отелей: не указаны город (или регион) назначения "
                "и дата заезда. Опишите их в сообщении или задайте в фильтрах слева."
            )
        elif not destination_text:
            message = (
                "Не удалось понять, в каком городе или регионе искать отели. "
                "Укажите название явно (например «Сочи», «Париж, Франция»)."
            )
        else:
            message = (
                "Дата заезда не определена из текста. Укажите дату заезда в запросе "
                "или выберите её в фильтрах."
            )
        return {
            "user_request": user_request,
            "extracted": extracted.model_dump(),
            "extraction_meta": extraction_meta,
            "hotel_not_found_reason": reason,
            "hotel_not_found_message": message,
            "notices": notices,
            "hotels": [],
        }

    if check_out is None or check_out <= check_in:
        check_out = check_in + timedelta(days=max(1, extracted.trip_days or 1))

    nights = max(1, (check_out - check_in).days)
    currency = _normalize_currency(extracted.currency)
    adults = extracted.adults or max(1, extracted.passengers)
    children = extracted.children or 0

    hotel_client = TravelPayoutsHotelsClient(api_token=api_token or "", marker=marker)
    city_resolution = hotel_client.resolve_city(destination_text, lang="ru")
    search_location = (
        city_resolution.get("location_id")
        if city_resolution and city_resolution.get("location_id")
        else destination_text
    )
    if city_resolution and city_resolution.get("city_name"):
        notices.append(f"destination_resolved: {city_resolution['city_name']}")

    hotels = hotel_client.search_hotels(
        location_query=str(search_location),
        check_in=check_in,
        check_out=check_out,
        currency=currency,
        adults=adults,
        children=children,
        limit=max(10, max_results * 3),
        language="ru",
    )

    degraded_to_search_links = False
    if not hotels:
        if not api_token:
            notices.append("hotel_api: TRAVELPAYOUTS_API_TOKEN не задан — список отелей из Hotellook недоступен.")
        else:
            notices.append(
                "hotel_api: виджет, кеш и статический список не вернули отели "
                "(проверьте токен, даты и доступность сервисов Hotellook)."
            )
        fallback_hotels = hotel_client.build_redirect_hotel_options(
            location_query=destination_text,
            check_in=check_in,
            check_out=check_out,
            currency=currency,
            adults=adults,
            children=children,
        )[: max(1, max_results)]
        if fallback_hotels:
            hotels = fallback_hotels
            degraded_to_search_links = True
            notices.append("hotel_fallback: показаны только ссылки на общий поиск Aviasales (не конкретные отели).")

    hotel_not_found_reason: str | None = None
    hotel_not_found_message: str | None = None
    if not hotels:
        hotel_not_found_reason = "no_hotels_from_api"
        hotel_not_found_message = (
            "По выбранному месту и датам заезда сервис не вернул ни одного доступного варианта. "
            "Проверьте написание города, попробуйте соседние даты или более крупный населённый пункт рядом."
        )
    elif degraded_to_search_links:
        hotel_not_found_reason = "degraded_to_aviasales_search"
        hotel_not_found_message = (
            "Конкретные отели из API не получены; ниже — ссылки на общий поиск по городу на Aviasales "
            "(разные сортировки)."
        )

    hotels_before_budget = len(hotels)
    hotels = filter_hotels_by_budget(hotels, extracted.budget, nights=nights)
    if hotels_before_budget > 0 and extracted.budget is not None and not hotels:
        hotel_not_found_reason = "budget_too_low"
        hotel_not_found_message = (
            "Отели в выдаче есть, но ни один не укладывается в указанный бюджет "
            "(учитывается стоимость проживания на весь срок). Увеличьте лимит или снимите ограничение по бюджету."
        )

    if hotels:
        sort_hotels_by_price_then_rating(hotels)
    hotels = hotels[: max(1, max_results)]
    if hotels and not degraded_to_search_links:
        hotel_not_found_reason = None
        hotel_not_found_message = None
    elif hotel_not_found_reason is None:
        hotel_not_found_reason = "no_hotels_after_filters"
        hotel_not_found_message = (
            "После отбора по бюджету и лимиту числа результатов подходящих отелей не осталось. "
            "Ослабьте фильтры или измените даты и город поиска."
        )

    return {
        "user_request": user_request,
        "extracted": extracted.model_dump(),
        "extraction_meta": extraction_meta,
        "hotel_not_found_reason": hotel_not_found_reason,
        "hotel_not_found_message": hotel_not_found_message,
        "search_params": {
            "destination": destination_text,
            "check_in": check_in.isoformat(),
            "check_out": check_out.isoformat(),
            "nights": nights,
            "currency": currency,
            "adults": adults,
            "children": children,
            "budget": extracted.budget,
        },
        "notices": notices,
        "hotels": hotels,
    }


@tool
def search_routes_from_text(
    user_request: str,
    default_origin_city: str | None = None,
    default_origin_iata: str | None = None,
    max_results: int = 5,
) -> dict[str, Any]:
    """
    Universal flight-search tool:
    1) LLM extracts parameters from text.
    2) Parameters are normalized.
    3) Travelpayouts API is queried.
    4) Results are filtered by budget/stops and returned.
    """
    extracted, extraction_meta = extract_trip_query(user_request, return_metadata=True)
    return search_routes_from_extracted(
        user_request=user_request,
        extracted=extracted,
        extraction_meta=extraction_meta,
        default_origin_city=default_origin_city,
        default_origin_iata=default_origin_iata,
        max_results=max_results,
    )


@tool
def search_hotels_from_text(
    user_request: str,
    default_destination_city: str | None = None,
    max_results: int = 5,
) -> dict[str, Any]:
    """
    Hotel-search tool:
    1) LLM extracts destination/dates/preferences from text.
    2) Hotellook/Travelpayouts API is queried.
    3) Results are filtered by budget and normalized.
    """
    extracted, extraction_meta = extract_trip_query(user_request, return_metadata=True)
    return search_hotels_from_extracted(
        user_request=user_request,
        extracted=extracted,
        extraction_meta=extraction_meta,
        default_destination_city=default_destination_city,
        max_results=max_results,
    )


@tool
def search_travel_from_text(
    user_request: str,
    default_origin_city: str | None = None,
    default_origin_iata: str | None = None,
    default_destination_city: str | None = None,
    max_results: int = 5,
) -> dict[str, Any]:
    """
    Combined travel-search tool with intent routing:
    - flights only,
    - hotels only,
    - both (package/tour).
    """
    scope = detect_search_scope(user_request)
    extracted, extraction_meta = extract_trip_query(user_request, return_metadata=True)

    flights_result: dict[str, Any] | None = None
    hotels_result: dict[str, Any] | None = None
    errors: dict[str, str] = {}

    if scope in ("flights", "both"):
        try:
            flights_result = search_routes_from_extracted(
                user_request=user_request,
                extracted=extracted,
                extraction_meta=extraction_meta,
                default_origin_city=default_origin_city,
                default_origin_iata=default_origin_iata,
                max_results=max_results,
            )
        except Exception as exc:  # pragma: no cover
            errors["flights"] = f"{type(exc).__name__}: {exc}"

    if scope in ("hotels", "both"):
        try:
            hotels_result = search_hotels_from_extracted(
                user_request=user_request,
                extracted=extracted,
                extraction_meta=extraction_meta,
                default_destination_city=default_destination_city,
                max_results=max_results,
            )
        except Exception as exc:  # pragma: no cover
            errors["hotels"] = f"{type(exc).__name__}: {exc}"

    attractions_result: dict[str, Any] | None = None
    dest_label, dest_country = destination_label_for_attractions(extracted, default_destination_city)
    if dest_label:
        attractions_result = suggest_city_attractions(dest_label, dest_country, max_items=8)

    return {
        "user_request": user_request,
        "scope": scope,
        "extracted": extracted.model_dump(),
        "extraction_meta": extraction_meta,
        "flights": flights_result,
        "hotels": hotels_result,
        "attractions": attractions_result,
        "errors": errors,
    }


if __name__ == "__main__":
    sample = "Хочу поездку в Рим на 5 дней в июне, бюджет 1200€"
    result = search_routes_from_text.invoke(
        {
            "user_request": sample,
            "default_origin_city": "Moscow",
            "max_results": 3,
        }
    )
    print(result)
