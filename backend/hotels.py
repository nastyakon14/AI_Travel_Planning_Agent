"""Hotel search utilities for Travelpayouts/Hotellook API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import urlencode

import requests

HOTEL_LOOKUP_URL = "https://engine.hotellook.com/api/v2/lookup.json"
HOTEL_CACHE_URL = "https://engine.hotellook.com/api/v2/cache.json"
HOTEL_STATIC_HOTELS_URL = "https://engine.hotellook.com/api/v2/static/hotels.json"
HOTEL_REDIRECT_URL = "https://hotels-api.aviasales.ru/v1/tp/redirect"

# Виджет выдачи (документация Travelpayouts); пробуем несколько схем хоста.
WIDGET_LOCATION_DUMP_URLS = (
    "https://yasen.hotellook.com/tp/public/widget_location_dump.json",
    "http://yasen.hotellook.com/tp/public/widget_location_dump.json",
)

DEFAULT_REQUEST_HEADERS = {
    "User-Agent": "AI-Travel-Planning-Agent/1.0 (+https://github.com/)",
    "Accept": "application/json",
}


def normalize_hotel_guest_rating(value: Any) -> float:
    """
    Единая шкала 0–10 для сравнения и сортировки.
    В ответах Hotellook встречаются баллы 0–10 и целые 20–95 (условно «из 100»).
    """
    if value is None:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v <= 0:
        return 0.0
    if v <= 10:
        return v
    # чуть выше 10 на «десятибалльной» шкале
    if v <= 11:
        return min(10.0, v)
    if v <= 100:
        return min(10.0, v / 10.0)
    return 10.0


def sort_hotels_by_price_then_rating(hotels: list[dict[str, Any]]) -> None:
    """Сначала дешевле, при равной цене — выше нормализованный гостевой рейтинг."""
    hotels.sort(
        key=lambda h: (
            float(h.get("price_per_night") or 0),
            -normalize_hotel_guest_rating(h.get("rating")),
        )
    )


class HotelsAPIError(RuntimeError):
    """Raised when hotel API is unavailable or returns invalid response."""


@dataclass
class TravelPayoutsHotelsClient:
    """Read-only wrapper around Hotellook endpoints."""

    api_token: str
    marker: str | None = None
    timeout_seconds: int = 20

    def _get(self, url: str, params: dict[str, Any]) -> Any:
        try:
            response = requests.get(
                url,
                params=params,
                timeout=self.timeout_seconds,
                headers=DEFAULT_REQUEST_HEADERS,
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            body = (exc.response.text[:200] if exc.response is not None else "").replace("\n", " ")
            raise HotelsAPIError(
                f"Hotel API HTTP {status} for {url}. Response: {body}"
            ) from exc
        except requests.RequestException as exc:
            raise HotelsAPIError(f"Hotel API request failed for {url}: {exc}") from exc

    def _get_optional(self, url: str, params: dict[str, Any]) -> Any | None:
        try:
            return self._get(url, params)
        except HotelsAPIError:
            return None

    def _hotel_search_portal_link(self, hotel_id: int | str, language: str = "ru") -> str:
        q: dict[str, str] = {"language": language, "hotelId": str(hotel_id)}
        if self.marker:
            q["marker"] = str(self.marker)
        return f"https://search.hotellook.com/?{urlencode(q)}"

    @staticmethod
    def _pick_hotel_name(name_field: Any, lang: str) -> str:
        if isinstance(name_field, str) and name_field.strip():
            return name_field.strip()
        if isinstance(name_field, dict):
            for key in (lang, "ru", "en"):
                val = name_field.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            for val in name_field.values():
                if isinstance(val, str) and val.strip():
                    return val.strip()
        return "Отель"

    def resolve_city(self, query: str, lang: str = "ru") -> dict[str, Any] | None:
        if not query:
            return None

        params: dict[str, Any] = {
            "query": query.strip(),
            "lang": lang,
            "lookFor": "city",
            "limit": 7,
        }
        if self.api_token:
            params["token"] = self.api_token

        payload = self._get_optional(HOTEL_LOOKUP_URL, params)
        if not isinstance(payload, list):
            return None

        for row in payload:
            if not isinstance(row, dict):
                continue
            location_id = row.get("locationId") or row.get("cityId") or row.get("id")
            if location_id is None:
                continue
            return {
                "location_id": str(location_id),
                "city_name": row.get("cityName") or row.get("name") or query,
                "country_name": row.get("countryName"),
            }
        return None

    def _normalize_cache_rows(
        self,
        rows: list[dict[str, Any]],
        location_label: str,
        currency: str,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            raw_price = row.get("priceFrom") or row.get("price") or row.get("minPrice")
            if raw_price is None:
                continue
            try:
                price_per_night = float(raw_price)
            except (TypeError, ValueError):
                continue

            hotel_id = row.get("hotelId") or row.get("id") or row.get("hotel_id")
            name = row.get("hotelName") or row.get("name") or row.get("label")
            link = row.get("url") or row.get("deepLink") or row.get("link")
            if hotel_id is not None and not link:
                link = self._hotel_search_portal_link(hotel_id)

            normalized.append(
                {
                    "hotel_id": str(hotel_id) if hotel_id is not None else None,
                    "name": name,
                    "location": row.get("locationName") or location_label,
                    "price_per_night": price_per_night,
                    "currency": currency.upper(),
                    "stars": row.get("stars"),
                    "rating": row.get("rating"),
                    "link": link,
                    "source": "hotellook_cache",
                }
            )

        normalized.sort(
            key=lambda hotel: (
                hotel["price_per_night"],
                -normalize_hotel_guest_rating(hotel.get("rating")),
            )
        )
        return normalized

    def _fetch_cache_hotels(
        self,
        location_param: str,
        check_in: date,
        check_out: date,
        currency: str,
        adults: int,
        children: int,
        limit: int,
        location_label: str,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "location": location_param,
            "checkIn": check_in.isoformat(),
            "checkOut": check_out.isoformat(),
            "adultsCount": max(1, adults),
            "childrenCount": max(0, children),
            "currency": currency.upper(),
            "limit": max(1, min(limit, 50)),
        }
        if self.api_token:
            params["token"] = self.api_token
        if self.marker:
            params["marker"] = self.marker

        payload = self._get_optional(HOTEL_CACHE_URL, params)
        if payload is None:
            return []

        rows: list[dict[str, Any]]
        if isinstance(payload, dict):
            raw_rows = payload.get("data") or payload.get("hotels") or []
            rows = raw_rows if isinstance(raw_rows, list) else []
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []

        return self._normalize_cache_rows(rows, location_label, currency)

    def _fetch_widget_hotels(
        self,
        location_id: int,
        check_in: date,
        check_out: date,
        currency: str,
        language: str,
        limit: int,
        location_label: str,
        collection_type: str = "popularity",
    ) -> list[dict[str, Any]]:
        if not self.api_token:
            return []

        base_params: dict[str, Any] = {
            "id": location_id,
            "type": collection_type,
            "check_in": check_in.isoformat(),
            "check_out": check_out.isoformat(),
            "currency": currency.lower(),
            "language": language,
            "limit": max(1, min(100, limit)),
            "token": self.api_token,
        }

        payload: Any = None
        for url in WIDGET_LOCATION_DUMP_URLS:
            payload = self._get_optional(url, base_params)
            if payload is not None:
                break

        if payload is None:
            return []

        rows: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            rows = payload.get(collection_type) or []
            if not rows:
                for value in payload.values():
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        rows = value
                        break
        elif isinstance(payload, list):
            rows = payload

        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            hid = row.get("hotel_id")
            if hid is None:
                continue
            lpi = row.get("last_price_info") if isinstance(row.get("last_price_info"), dict) else {}
            ppn = lpi.get("price_pn")
            if ppn is None:
                total = lpi.get("price")
                nights = lpi.get("nights") or 0
                try:
                    if total is not None and int(nights) > 0:
                        ppn = float(total) / float(int(nights))
                    else:
                        ppn = 0.0
                except (TypeError, ValueError, ZeroDivisionError):
                    ppn = 0.0
            else:
                try:
                    ppn = float(ppn)
                except (TypeError, ValueError):
                    ppn = 0.0

            rating = row.get("rating")
            stars = row.get("stars")
            name = row.get("name") or f"Отель {hid}"

            out.append(
                {
                    "hotel_id": str(hid),
                    "name": name,
                    "location": location_label,
                    "price_per_night": max(0.0, ppn),
                    "currency": currency.upper(),
                    "stars": stars,
                    "rating": rating,
                    "link": self._hotel_search_portal_link(hid, language=language),
                    "source": "yasen_widget_dump",
                }
            )

        out.sort(
            key=lambda h: (
                h["price_per_night"] if h["price_per_night"] > 0 else 1e12,
                -normalize_hotel_guest_rating(h.get("rating")),
            )
        )
        return out

    def _fetch_static_hotels(
        self,
        location_id: int,
        currency: str,
        limit: int,
        location_label: str,
        language: str,
    ) -> list[dict[str, Any]]:
        if not self.api_token:
            return []

        params = {"locationId": location_id, "token": self.api_token}
        payload = self._get_optional(HOTEL_STATIC_HOTELS_URL, params)
        if not isinstance(payload, dict):
            return []

        raw_hotels = payload.get("hotels")
        if not isinstance(raw_hotels, list):
            return []

        scored: list[dict[str, Any]] = []
        for row in raw_hotels:
            if not isinstance(row, dict):
                continue
            hid = row.get("id")
            if hid is None:
                continue
            popularity = int(row.get("popularity") or 0)
            try:
                pfrom = float(row.get("pricefrom") or 0.0)
            except (TypeError, ValueError):
                pfrom = 0.0
            scored.append((popularity, -pfrom, row))

        scored.sort(key=lambda t: (-t[0], t[1]))

        out: list[dict[str, Any]] = []
        for _pop, _neg_price, row in scored[: max(1, limit)]:
            hid = row.get("id")
            name = self._pick_hotel_name(row.get("name"), language)
            link = row.get("link")
            if link and not str(link).startswith("http"):
                link = f"https://www.hotellook.com{link}"
            elif not link:
                link = self._hotel_search_portal_link(hid, language=language)

            try:
                ppn = float(row.get("pricefrom") or 0.0)
            except (TypeError, ValueError):
                ppn = 0.0

            rating = row.get("rating")
            if rating == 0:
                rating = None

            out.append(
                {
                    "hotel_id": str(hid),
                    "name": name,
                    "location": location_label,
                    "price_per_night": max(0.0, ppn),
                    "currency": currency.upper(),
                    "stars": row.get("stars"),
                    "rating": rating,
                    "link": link,
                    "source": "hotellook_static_list",
                }
            )

        return out

    def search_hotels(
        self,
        location_query: str,
        check_in: date,
        check_out: date,
        currency: str = "RUB",
        adults: int = 1,
        children: int = 0,
        limit: int = 10,
        language: str = "ru",
    ) -> list[dict[str, Any]]:
        """
        Цепочка источников: виджет yasen (актуальные цены) → кеш cache.json →
        статический список static/hotels.json (реальные названия и ориентировочная цена).
        """
        raw_q = str(location_query).strip()
        location_label = raw_q
        location_id: int | None = None

        if raw_q.isdigit():
            location_id = int(raw_q)
        else:
            resolved = self.resolve_city(raw_q, lang=language)
            if resolved and resolved.get("location_id"):
                try:
                    location_id = int(resolved["location_id"])
                except (TypeError, ValueError):
                    location_id = None
                if resolved.get("city_name"):
                    location_label = str(resolved["city_name"])

        fetch_limit = max(10, limit * 3)

        if location_id is not None:
            hotels = self._fetch_widget_hotels(
                location_id=location_id,
                check_in=check_in,
                check_out=check_out,
                currency=currency,
                language=language,
                limit=fetch_limit,
                location_label=location_label,
            )
            if hotels:
                return hotels[:limit]

        cache_loc = str(location_id) if location_id is not None else raw_q
        hotels = self._fetch_cache_hotels(
            location_param=cache_loc,
            check_in=check_in,
            check_out=check_out,
            currency=currency,
            adults=adults,
            children=children,
            limit=fetch_limit,
            location_label=location_label,
        )
        if hotels:
            return hotels[:limit]

        if location_id is not None:
            hotels = self._fetch_static_hotels(
                location_id=location_id,
                currency=currency,
                limit=fetch_limit,
                location_label=location_label,
                language=language,
            )
            if hotels:
                return hotels[:limit]

        return []

    def build_redirect_hotel_options(
        self,
        location_query: str,
        check_in: date,
        check_out: date,
        currency: str = "RUB",
        adults: int = 1,
        children: int = 0,
    ) -> list[dict[str, Any]]:
        """Резерв: только ссылки на поиск Aviasales (не выдача конкретных отелей)."""
        base_params = {
            "destination": location_query,
            "checkIn": check_in.isoformat(),
            "checkOut": check_out.isoformat(),
            "adults": max(1, adults),
            "children": max(0, children),
            "selected_currency": currency.upper(),
        }
        variants = [
            ("recommended", "Поиск Aviasales — по популярности"),
            ("price", "Поиск Aviasales — сначала дешевле"),
            ("class", "Поиск Aviasales — по рейтингу"),
        ]
        options: list[dict[str, Any]] = []
        for idx, (order, title) in enumerate(variants, start=1):
            params = {**base_params, "order": order}
            prepared = requests.Request("GET", HOTEL_REDIRECT_URL, params=params).prepare()
            options.append(
                {
                    "hotel_id": f"search-{idx}",
                    "name": title,
                    "location": location_query,
                    "price_per_night": 0.0,
                    "currency": currency.upper(),
                    "stars": None,
                    "rating": None,
                    "link": prepared.url,
                    "source": "aviasales_redirect_fallback",
                    "is_search_portal_only": True,
                }
            )
        return options


def filter_hotels_by_budget(
    hotels: list[dict[str, Any]],
    budget: float | None,
    nights: int,
) -> list[dict[str, Any]]:
    """Keep hotels whose estimated stay price is within total budget."""
    if budget is None:
        return hotels

    stay_nights = max(1, nights)
    return [
        hotel
        for hotel in hotels
        if (hotel.get("price_per_night", 0.0) * stay_nights) <= budget
    ]
