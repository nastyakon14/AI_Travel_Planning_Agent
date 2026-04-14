"""Утилиты для работы с API Travelpayouts (Aviasales)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

import requests

AUTOCOMPLETE_URL = "https://autocomplete.travelpayouts.com/places2"   # API для автокомплита городов и аэропортов
PRICES_FOR_DATES_URL = "https://api.travelpayouts.com/aviasales/v3/prices_for_dates"   # API для поиска авиабилетов

@dataclass
class TravelPayoutsClient:
    """Класс для поиска авиабилетов по API Travelpayouts (Aviasales)"""

    api_token: str
    timeout_seconds: int = 20 # таймаут в секундах

    def _get(self, url: str, params: Any) -> Any:
        response = requests.get(url, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def resolve_place(self, query: str, locale: str = "en") -> dict[str, Any] | None:
        """Resolve free-form city/airport text into an IATA code."""
        if not query:
            return None

        clean = query.strip()
        if re.fullmatch(r"[A-Za-z]{3}", clean):
            return {"iata": clean.upper(), "name": clean.upper(), "type": "iata"}

        params = [("term", clean), ("locale", locale), ("types[]", "city"), ("types[]", "airport")]
        payload = self._get(AUTOCOMPLETE_URL, params)

        if not isinstance(payload, list):
            return None

        for item in payload:
            code = str(item.get("code", "")).upper()
            if len(code) == 3:
                return {
                    "iata": code,
                    "name": item.get("name"),
                    "country_name": item.get("country_name"),
                    "type": item.get("type"),
                }
        return None

    def search_flights(
        self,
        origin_iata: str,
        destination_iata: str,
        departure_date: date | None = None,
        return_date: date | None = None,
        currency: str = "RUB",
        direct_only: bool | None = None,
        one_way: bool | None = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Fetch flight options from `prices_for_dates` and normalize fields."""
        if not self.api_token:
            raise ValueError("TRAVELPAYOUTS_API_TOKEN (or TRAVELPAYOUTS_API_KEY) is not set.")

        params: dict[str, Any] = {
            "token": self.api_token,
            "origin": origin_iata.upper(),
            "destination": destination_iata.upper(),
            "currency": currency.upper(),
            "sorting": "price",
            "page": 1,
            "limit": max(1, min(limit, 100)),
        }

        if departure_date:
            params["departure_at"] = departure_date.isoformat()
        if return_date:
            params["return_at"] = return_date.isoformat()
        if direct_only is not None:
            params["direct"] = "true" if direct_only else "false"
        if one_way is not None:
            params["one_way"] = "true" if one_way else "false"

        payload = self._get(PRICES_FOR_DATES_URL, params)
        rows = payload.get("data", []) if isinstance(payload, dict) else []

        normalized: list[dict[str, Any]] = []
        for row in rows:
            price = row.get("price")
            if price is None:
                continue

            normalized.append(
                {
                    "flight_id": (
                        f'{origin_iata.upper()}-{destination_iata.upper()}-'
                        f'{row.get("departure_at", "unknown")}-{row.get("flight_number", "NA")}'
                    ),
                    "origin": row.get("origin", origin_iata.upper()),
                    "destination": row.get("destination", destination_iata.upper()),
                    "departure_at": row.get("departure_at"),
                    "return_at": row.get("return_at"),
                    "airline": row.get("airline"),
                    "flight_number": row.get("flight_number"),
                    "transfers": row.get("transfers", 0),
                    "duration_to": row.get("duration_to"),
                    "duration_back": row.get("duration_back"),
                    "link": row.get("link"),
                    "price": float(price),
                    "currency": currency.upper(),
                }
            )

        normalized.sort(key=lambda flight: flight["price"])
        return normalized


def filter_routes_by_budget(
    routes: list[dict[str, Any]],
    budget: float | None,
    passengers: int,
) -> list[dict[str, Any]]:
    """Keep routes whose estimated total price is within budget."""
    if budget is None:
        return routes
    return [
        route
        for route in routes
        if (route.get("price", 0.0) * max(passengers, 1)) <= budget
    ]
