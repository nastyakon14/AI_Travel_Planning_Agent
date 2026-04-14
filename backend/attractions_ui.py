"""
Обогащение списка достопримечательностей для UI: координаты (Nominatim), превью (Wikipedia).

Nominatim: https://operations.osmfoundation.org/policies/nominatim/ — не чаще ~1 запрос/сек.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

_NOMINATIM_DELAY = float(os.getenv("NOMINATIM_DELAY_SEC", "1.1"))
_USER_AGENT = os.getenv(
    "NOMINATIM_USER_AGENT",
    "AI-Travel-Planning-Agent/1.0 (https://github.com/; travel demo)",
)
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": _USER_AGENT, "Accept-Language": "ru,en;q=0.9"})


def _geocode_nominatim(query: str) -> tuple[float, float] | None:
    if not query.strip():
        return None
    params = {"q": query, "format": "json", "limit": 1}
    try:
        r = _SESSION.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
        return lat, lon
    except (requests.RequestException, KeyError, ValueError, TypeError, IndexError):
        return None


def _wikipedia_thumbnail_ru(search_query: str) -> str | None:
    """Миниатюра статьи Википедии (ru), если есть."""
    if not search_query.strip():
        return None
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": search_query,
        "gsrlimit": "1",
        "gsrnamespace": "0",
        "prop": "pageimages",
        "piprop": "thumbnail",
        "pithumbsize": "400",
        "format": "json",
        "origin": "*",
    }
    try:
        r = _SESSION.get(
            "https://ru.wikipedia.org/w/api.php",
            params=params,
            timeout=12,
        )
        r.raise_for_status()
        data = r.json()
        pages = (data.get("query") or {}).get("pages") or {}
        for _pid, page in pages.items():
            if not isinstance(page, dict):
                continue
            thumb = page.get("thumbnail")
            if isinstance(thumb, dict) and thumb.get("source"):
                return str(thumb["source"])
    except (requests.RequestException, TypeError, ValueError):
        return None
    return None


def enrich_attractions_for_ui(
    items: list[dict[str, Any]],
    city: str,
    country: str | None,
    *,
    geocode: bool = True,
    images: bool = True,
) -> list[dict[str, Any]]:
    """
    Добавляет к каждому элементу lat, lon, image_url (по возможности).
    """
    if os.getenv("SKIP_ATTRACTION_ENRICHMENT", "").strip().lower() in ("1", "true", "yes"):
        return [dict(x) for x in items]

    out: list[dict[str, Any]] = []
    loc_ctx = ", ".join(x for x in (city.strip(), (country or "").strip()) if x)

    for i, raw in enumerate(items):
        row = dict(raw)
        name = str(row.get("name") or "").strip()
        summary = str(row.get("summary") or "").strip()
        row.setdefault("lat", None)
        row.setdefault("lon", None)
        row.setdefault("image_url", None)

        if geocode and name:
            time.sleep(_NOMINATIM_DELAY if i > 0 else 0)
            q = f"{name}, {loc_ctx}" if loc_ctx else name
            coords = _geocode_nominatim(q)
            if coords:
                row["lat"], row["lon"] = coords

        if images and name and not row.get("image_url"):
            row["image_url"] = _wikipedia_thumbnail_ru(f"{name} {city}")

        out.append(row)

    return out


def build_attractions_folium_map(
    enriched: list[dict[str, Any]],
) -> Any | None:
    """Folium-карта с маркерами и ломаной «маршрут» по порядку списка. Возвращает None, если нет координат."""
    try:
        import folium
    except ImportError:
        return None

    points: list[tuple[float, float, str]] = []
    for i, row in enumerate(enriched):
        lat, lon = row.get("lat"), row.get("lon")
        if lat is None or lon is None:
            continue
        try:
            points.append((float(lat), float(lon), str(row.get("name") or f"Точка {i + 1}")))
        except (TypeError, ValueError):
            continue

    if not points:
        return None

    mid_lat = sum(p[0] for p in points) / len(points)
    mid_lon = sum(p[1] for p in points) / len(points)
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=13, tiles="OpenStreetMap")

    for i, (lat, lon, name) in enumerate(points):
        folium.Marker(
            [lat, lon],
            popup=f"{i + 1}. {name}",
            tooltip=f"{i + 1}. {name}",
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(m)

    if len(points) >= 2:
        folium.PolyLine(
            [[p[0], p[1]] for p in points],
            color="#2563eb",
            weight=4,
            opacity=0.75,
            dash_array="10, 10",
        ).add_to(m)

    return m
