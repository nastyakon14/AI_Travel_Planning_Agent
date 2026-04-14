"""Streamlit web UI for flight-route search from natural language."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.auth_streamlit import ensure_session_user
from backend.context_memory import format_conversation_for_extraction
from backend.guardrails import GuardrailViolation, sanitize_user_input
from backend.hotels import normalize_hotel_guest_rating
from backend.travel_agent import (
    FAST_EXTRACTION_MODEL,
    OPENAI_API_KEY,
    OPENAI_URL,
    STRONG_EXTRACTION_MODEL,
    TripQuery,
    destination_label_for_attractions,
    detect_search_scope,
)
from backend.travel_facade import (
    extract_trip_query,
    run_travel_planning_graph,
    search_hotels_from_extracted,
    search_routes_from_extracted,
    suggest_city_attractions,
)

_log = logging.getLogger(__name__)
if os.getenv("PROMETHEUS_METRICS_PORT", "").strip() or os.getenv("PROMETHEUS_ENABLE", "").lower() in (
    "1",
    "true",
    "yes",
):
    try:
        from backend.prometheus_metrics import start_metrics_server

        _mp = int(os.getenv("PROMETHEUS_METRICS_PORT", "9090"))
        start_metrics_server(_mp)
    except Exception as exc:  # pragma: no cover
        _log.warning("Prometheus /metrics not started: %s", exc)

DEFAULT_MAX_RESULTS = max(1, min(int(os.getenv("MAX_RESULTS_DEFAULT", "5")), 10))


def _none_if_blank(value: str) -> str | None:
    clean = str(value or "").strip()
    return clean or None


def _csv_to_list(value: str) -> list[str] | None:
    clean = [item.strip() for item in (value or "").split(",") if item.strip()]
    return clean or None


def _to_optional_bool(value: str) -> bool | None:
    mapping: dict[str, bool | None] = {
        "Не важно": None,
        "Да": True,
        "Нет": False,
        "Только прямые": True,
        "Можно с пересадками": False,
    }
    return mapping.get(value)


def _build_manual_trip_query(filters: dict[str, Any]) -> TripQuery:
    max_stops_raw = filters.get("max_stops")
    max_stops = None if max_stops_raw == "Не важно" else int(max_stops_raw)
    cabin_class_raw = filters.get("cabin_class")
    cabin_class = None if cabin_class_raw == "Не важно" else cabin_class_raw
    pref_dep_raw = filters.get("preferred_departure_time")
    preferred_departure_time = None if pref_dep_raw == "Не важно" else pref_dep_raw

    return TripQuery(
        origin_city=_none_if_blank(filters.get("origin_city")),
        origin_iata=_none_if_blank(filters.get("origin_iata")),
        destination_city=_none_if_blank(filters.get("destination_city")),
        destination_iata=_none_if_blank(filters.get("destination_iata")),
        departure_date=_none_if_blank(filters.get("departure_date")),
        return_date=_none_if_blank(filters.get("return_date")),
        trip_days=(int(filters["trip_days"]) if int(filters["trip_days"]) > 0 else None),
        passengers=max(1, int(filters["passengers"])),
        adults=(int(filters["adults"]) if int(filters["adults"]) > 0 else None),
        children=int(filters["children"]),
        infants=int(filters["infants"]),
        budget=(float(filters["budget"]) if float(filters["budget"]) > 0 else None),
        currency=str(filters["currency"]),
        direct_only=_to_optional_bool(filters["direct_only"]),
        one_way=_to_optional_bool(filters["one_way"]),
        max_stops=max_stops,
        cabin_class=cabin_class,
        include_baggage=_to_optional_bool(filters["include_baggage"]),
        flexible_dates=_to_optional_bool(filters["flexible_dates"]),
        preferred_departure_time=preferred_departure_time,
        preferred_airlines=_csv_to_list(filters.get("preferred_airlines")),
        excluded_airlines=_csv_to_list(filters.get("excluded_airlines")),
    )


def _build_manual_user_text(filters: dict[str, Any], scope: str) -> str:
    parts = []
    if _none_if_blank(filters.get("origin_city")):
        parts.append(f"из {filters['origin_city']}")
    if _none_if_blank(filters.get("destination_city")):
        parts.append(f"в {filters['destination_city']}")
    if _none_if_blank(filters.get("departure_date")):
        parts.append(f"вылет {filters['departure_date']}")
    if _none_if_blank(filters.get("return_date")):
        parts.append(f"обратно {filters['return_date']}")
    if int(filters["passengers"]) > 1:
        parts.append(f"пассажиров: {int(filters['passengers'])}")
    if float(filters["budget"]) > 0:
        parts.append(f"бюджет: {float(filters['budget']):.0f} {filters['currency']}")
    details = ", ".join(parts) if parts else "без уточненных параметров"
    return f"Поиск по ручным фильтрам ({scope}): {details}"


def _normalize_ticket_link(raw_link: Any) -> str | None:
    if not raw_link:
        return None
    link = str(raw_link).strip()
    if not link:
        return None
    if link.startswith(("http://", "https://")):
        return link
    if link.startswith("/"):
        return f"https://www.aviasales.ru{link}"
    return f"https://www.aviasales.ru/{link}"


def _normalize_hotel_link(raw_link: Any) -> str | None:
    if not raw_link:
        return None
    link = str(raw_link).strip()
    if not link:
        return None
    if link.startswith(("http://", "https://")):
        return link
    if link.startswith("/"):
        return f"https://www.hotellook.com{link}"
    return f"https://www.hotellook.com/{link}"


def _format_hotel_rating(value: Any) -> str:
    n = normalize_hotel_guest_rating(value)
    if n <= 0:
        return "н/д"
    return f"{n:g}/10"


def _render_llm_metrics_expanders(
    *,
    extraction: dict[str, Any] | None = None,
    attractions: dict[str, Any] | None = None,
    itinerary: dict[str, Any] | None = None,
) -> None:
    """Показывает TTFT, TPOT, токены и оценку стоимости по этапам (если есть)."""
    bundle: dict[str, Any] = {}
    if extraction:
        bundle["extraction"] = extraction
    if attractions:
        bundle["attractions"] = attractions
    if itinerary:
        bundle["itinerary"] = itinerary
    if not bundle:
        return
    with st.expander("LLM: TTFT / TPOT / токены / стоимость"):
        st.json(bundle)


def _format_hotel_stars(value: Any) -> str:
    if value is None:
        return "н/д"
    try:
        s = int(value)
    except (TypeError, ValueError):
        return str(value)
    if s <= 0:
        return "н/д"
    return str(s)


def _infer_stay_nights(extracted: TripQuery) -> int:
    if extracted.trip_days and int(extracted.trip_days) > 0:
        return int(extracted.trip_days)
    d0, d1 = extracted.departure_date, extracted.return_date
    if d0 and d1:
        try:
            a = datetime.strptime(str(d0).strip()[:10], "%Y-%m-%d").date()
            b = datetime.strptime(str(d1).strip()[:10], "%Y-%m-%d").date()
            n = (b - a).days
            return max(1, n)
        except ValueError:
            pass
    return 3


def _best_route_index(routes: list[dict[str, Any]]) -> int | None:
    if not routes:
        return None

    def sort_key(i: int) -> tuple[int, float]:
        r = routes[i]
        transfers = int(r.get("transfers") or 0)
        price = float(r.get("price") or 1e12)
        return (transfers, price)

    return min(range(len(routes)), key=sort_key)


def _best_hotel_index(hotels: list[dict[str, Any]]) -> int | None:
    """Среди цен в «полосе» у минимума (~+15 %) выбираем максимальный гостевой рейтинг (шкала 0–10)."""
    if not hotels:
        return None
    candidates = [
        i
        for i, h in enumerate(hotels)
        if not h.get("is_search_portal_only") and float(h.get("price_per_night") or 0) > 0
    ]
    if not candidates:
        candidates = [i for i, h in enumerate(hotels) if not h.get("is_search_portal_only")]
    if not candidates:
        return None

    prices = {i: float(hotels[i].get("price_per_night") or 0) for i in candidates}
    positive = [prices[i] for i in candidates if prices[i] > 0]
    if not positive:
        return max(
            candidates,
            key=lambda i: normalize_hotel_guest_rating(hotels[i].get("rating")),
        )

    min_p = min(positive)
    # «Близкие цены» к минимальной: не выше чем на 15 % (минимум +50 единиц валюты для малых сумм)
    band = max(min_p * 0.15, 50.0)
    close_max = min_p + band
    in_band = [i for i in candidates if 0 < prices[i] <= close_max]
    pool = in_band if in_band else list(candidates)

    return max(
        pool,
        key=lambda i: (
            normalize_hotel_guest_rating(hotels[i].get("rating")),
            -prices[i],
        ),
    )


def _routes_comparison_df(routes: list[dict[str, Any]], best_idx: int | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, r in enumerate(routes):
        label = f"{i + 1}"
        rows.append(
            {
                "№": label,
                "Маршрут": f"{r.get('origin', '?')} → {r.get('destination', '?')}",
                "Цена": float(r.get("price") or 0),
                "Валюта": str(r.get("currency") or ""),
                "Пересадки": int(r.get("transfers") or 0),
                "Авиалиния": str(r.get("airline") or "—"),
                "Вылет": str(r.get("departure_at") or "—"),
                "Обратно": str(r.get("return_at") or "—"),
                "Оценка": "★ рекомендуем" if best_idx is not None and i == best_idx else "",
            }
        )
    return pd.DataFrame(rows)


def _hotels_comparison_df(hotels: list[dict[str, Any]], best_idx: int | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, h in enumerate(hotels):
        if h.get("is_search_portal_only"):
            rows.append(
                {
                    "№": str(i + 1),
                    "Название": str(h.get("name") or "—"),
                    "Цена/ночь": None,
                    "Валюта": str(h.get("currency") or ""),
                    "Рейтинг": "—",
                    "Звёзды": "—",
                    "Оценка": "",
                }
            )
            continue
        p = float(h.get("price_per_night") or 0)
        rows.append(
            {
                "№": str(i + 1),
                "Название": str(h.get("name") or "—"),
                "Цена/ночь": p if p > 0 else None,
                "Валюта": str(h.get("currency") or ""),
                "Рейтинг": _format_hotel_rating(h.get("rating")),
                "Звёзды": _format_hotel_stars(h.get("stars")),
                "Оценка": "★ рекомендуем" if best_idx is not None and i == best_idx else "",
            }
        )
    return pd.DataFrame(rows)


def render_flights_dashboard(routes: list[dict[str, Any]]) -> int | None:
    """Таблица, график цен, метрики и рекомендация по маршрутам."""
    if not routes:
        return None

    best_idx = _best_route_index(routes)
    prices = [float(r.get("price") or 0) for r in routes]
    transfers = [int(r.get("transfers") or 0) for r in routes]
    cur = str(routes[0].get("currency") or "")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Вариантов", len(routes))
    with c2:
        st.metric("Мин. цена", f"{min(prices):.0f} {cur}")
    with c3:
        st.metric("Макс. цена", f"{max(prices):.0f} {cur}")
    with c4:
        st.metric("Мин. пересадок", min(transfers))

    st.caption(
        "Рекомендация: сначала минимум пересадок, при равном числе пересадок — минимальная цена."
    )
    if best_idx is not None:
        br = routes[best_idx]
        st.success(
            f"**Оптимальный маршрут — вариант №{best_idx + 1}**: "
            f"{br.get('origin')} → {br.get('destination')}, "
            f"**{float(br.get('price') or 0):.0f} {cur}**, "
            f"пересадок: {int(br.get('transfers') or 0)}."
        )

    df = _routes_comparison_df(routes, best_idx)
    tab_t, tab_g = st.tabs(["Таблица", "График цен"])
    with tab_t:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Цена": st.column_config.NumberColumn("Цена", format="%.0f"),
                "Пересадки": st.column_config.NumberColumn("Пересадки", step=1),
            },
        )
    with tab_g:
        chart_df = pd.DataFrame(
            {
                "Вариант": [f"#{i + 1}" for i in range(len(routes))],
                "Цена": prices,
                "Пересадки": [str(t) for t in transfers],
            }
        )
        fig = px.bar(
            chart_df,
            x="Вариант",
            y="Цена",
            color="Пересадки",
            text="Цена",
        )
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig.update_layout(
            yaxis_title=f"Цена, {cur}",
            xaxis_title="Маршрут",
            height=420,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    return best_idx


def render_hotels_dashboard(hotels: list[dict[str, Any]]) -> int | None:
    """Таблица, график цен за ночь, метрики и рекомендация по отелям."""
    if not hotels:
        return None

    real = [h for h in hotels if not h.get("is_search_portal_only")]
    if not real:
        st.info("Доступны только ссылки на общий поиск — сравнительная таблица по отелям недоступна.")
        return None

    best_idx = _best_hotel_index(hotels)
    priced = [float(h.get("price_per_night") or 0) for h in real if float(h.get("price_per_night") or 0) > 0]
    cur = str(real[0].get("currency") or "")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Вариантов с данными", len(real))
    with c2:
        st.metric(
            "Мин. цена / ночь",
            f"{min(priced):.0f} {cur}" if priced else "—",
        )
    with c3:
        st.metric(
            "Макс. цена / ночь",
            f"{max(priced):.0f} {cur}" if priced else "—",
        )

    st.caption(
        "Рекомендация: среди вариантов с ценой не выше минимальной более чем примерно на 15 % "
        "(но не меньше +50 к минимуму в валюте) выбирается отель с **наивысшим** гостевым рейтингом (0–10); "
        "при равном рейтинге — дешевле. Рейтинг в таблице приведён к шкале /10."
    )
    if best_idx is not None:
        bh = hotels[best_idx]
        pp = float(bh.get("price_per_night") or 0)
        st.success(
            f"**Оптимальный отель — вариант №{best_idx + 1}**: **{bh.get('name', '—')}**, "
            f"{f'{pp:.0f} {cur} / ночь' if pp > 0 else 'цена по ссылке'}, "
            f"рейтинг {_format_hotel_rating(bh.get('rating'))}, звёзд {_format_hotel_stars(bh.get('stars'))}."
        )

    df = _hotels_comparison_df(hotels, best_idx)
    tab_t, tab_g = st.tabs(["Таблица", "Цена за ночь"])
    with tab_t:
        st.dataframe(df, use_container_width=True, hide_index=True)
    with tab_g:
        chart_rows = []
        for i, h in enumerate(hotels):
            if h.get("is_search_portal_only"):
                continue
            p = float(h.get("price_per_night") or 0)
            if p <= 0:
                continue
            chart_rows.append({"Отель": f"#{i + 1} {str(h.get('name'))[:28]}", "Цена за ночь": p})
        if chart_rows:
            cdf = pd.DataFrame(chart_rows)
            fig = px.bar(cdf, x="Цена за ночь", y="Отель", orientation="h", text="Цена за ночь")
            fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
            fig.update_layout(height=max(320, 40 * len(chart_rows)), margin=dict(l=20, r=80, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Нет числовых цен для графика — откройте ссылки для уточнения.")

    return best_idx


def render_package_summary(
    routes: list[dict[str, Any]],
    hotels: list[dict[str, Any]],
    extracted: TripQuery,
    best_route_i: int | None,
    best_hotel_i: int | None,
) -> None:
    """Ориентировочная сумма лучшего перелёта + проживания для тура."""
    if best_route_i is None or best_hotel_i is None:
        return
    hr = hotels[best_hotel_i]
    if hr.get("is_search_portal_only"):
        return
    hp = float(hr.get("price_per_night") or 0)
    if hp <= 0:
        return
    rr = routes[best_route_i]
    fp = float(rr.get("price") or 0)
    cur_f = str(rr.get("currency") or "")
    cur_h = str(hr.get("currency") or "")
    nights = _infer_stay_nights(extracted)
    if cur_f and cur_h and cur_f != cur_h:
        st.warning(
            f"Валюты перелёта ({cur_f}) и отеля ({cur_h}) различаются — сумма ниже ориентировочная."
        )
    cur = cur_f or cur_h
    total = fp + hp * nights
    st.subheader("Сводка тура (ориентировочно)")
    st.info(
        f"Лучший выбранный перелёт (**{fp:.0f} {cur_f}**) + **{nights}** ноч. "
        f"в рекомендуемом отеле (**{hp:.0f} {cur_h}** / ночь) ≈ **{total:.0f}** "
        f"({cur}) без учёта сборов, страховок и акций."
    )


def _render_routes_markdown(routes: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, route in enumerate(routes, start=1):
        airline = route.get("airline") or "н/д"
        ticket_link = _normalize_ticket_link(route.get("link"))
        link_line = (
            f"Ссылка на билет: [Открыть билет]({ticket_link})"
            if ticket_link
            else "Ссылка на билет: `нет в ответе API`"
        )
        lines.append(
            (
                f"**{idx}. {route.get('origin', '?')} -> {route.get('destination', '?')}**  \n"
                f"Цена: **{float(route.get('price', 0.0)):.0f} {route.get('currency', '')}**  \n"
                f"Вылет: `{route.get('departure_at', 'n/a')}`  \n"
                f"Обратно: `{route.get('return_at', 'n/a')}`  \n"
                f"Пересадки: `{route.get('transfers', 0)}`  \n"
                f"Авиалиния: `{airline}`  \n"
                f"{link_line}"
            )
        )
    return "\n\n".join(lines)


def _render_hotels_markdown(hotels: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, hotel in enumerate(hotels, start=1):
        hotel_link = _normalize_hotel_link(hotel.get("link"))
        raw_price = float(hotel.get("price_per_night", 0.0))
        if hotel.get("is_search_portal_only"):
            link_line = (
                f"[Перейти к поиску]({hotel_link})"
                if hotel_link
                else "`ссылка недоступна`"
            )
            lines.append(
                f"**{idx}. {hotel.get('name', 'Поиск')}**  \n"
                f"_{link_line}_"
            )
            continue

        price_line = (
            f"Цена за ночь: **{raw_price:.0f} {hotel.get('currency', '')}**"
            if raw_price > 0
            else "Цена за ночь: `уточняется по ссылке`"
        )
        link_line = (
            f"Ссылка на отель: [Открыть вариант]({hotel_link})"
            if hotel_link
            else "Ссылка на отель: `нет в ответе API`"
        )
        lines.append(
            (
                f"**{idx}. {hotel.get('name', 'Unknown hotel')}**  \n"
                f"Локация: `{hotel.get('location', 'n/a')}`  \n"
                f"{price_line}  \n"
                f"Рейтинг: `{_format_hotel_rating(hotel.get('rating'))}` | "
                f"Звёзды: `{_format_hotel_stars(hotel.get('stars'))}`  \n"
                f"{link_line}"
            )
        )
    return "\n\n".join(lines)


def _render_routes_widgets(routes: list[dict[str, Any]]) -> None:
    for idx, route in enumerate(routes, start=1):
        airline = route.get("airline") or "н/д"
        ticket_link = _normalize_ticket_link(route.get("link"))
        with st.container(border=True):
            st.markdown(f"**{idx}. {route.get('origin', '?')} -> {route.get('destination', '?')}**")
            st.write(
                f"Цена: {float(route.get('price', 0.0)):.0f} {route.get('currency', '')} | "
                f"Авиалиния: {airline} | Пересадки: {route.get('transfers', 0)}"
            )
            st.write(
                f"Вылет: {route.get('departure_at', 'n/a')} | "
                f"Обратно: {route.get('return_at', 'n/a')}"
            )
            if ticket_link:
                st.link_button("Открыть билет", ticket_link, use_container_width=False)
            else:
                st.caption("Ссылка на билет не пришла из API.")


def _render_hotels_widgets(hotels: list[dict[str, Any]]) -> None:
    for idx, hotel in enumerate(hotels, start=1):
        hotel_link = _normalize_hotel_link(hotel.get("link"))
        raw_price = float(hotel.get("price_per_night", 0.0))
        with st.container(border=True):
            st.markdown(f"**{idx}. {hotel.get('name', 'Unknown hotel')}**")
            if hotel.get("is_search_portal_only"):
                st.caption("Список отелей с API не получен — ссылка ведёт на общий поиск.")
                if hotel_link:
                    st.link_button("Открыть поиск", hotel_link, use_container_width=False)
                continue

            if raw_price > 0:
                price_info = f"{raw_price:.0f} {hotel.get('currency', '')}"
            else:
                price_info = "уточняется по ссылке"
            st.write(
                f"Цена за ночь: {price_info} | "
                f"Рейтинг: {_format_hotel_rating(hotel.get('rating'))} | "
                f"Звёзды: {_format_hotel_stars(hotel.get('stars'))}"
            )
            st.write(f"Локация: {hotel.get('location', 'n/a')}")
            if hotel.get("source"):
                st.caption(f"Источник: {hotel.get('source')}")
            if hotel_link:
                st.link_button("Открыть отель", hotel_link, use_container_width=False)
            else:
                st.caption("Ссылка на отель не пришла из API.")


def _build_flights_text(result: dict[str, Any]) -> str:
    routes = result.get("routes") or []
    notices = result.get("notices") or []
    not_found_reason = result.get("route_not_found_reason")
    not_found_message = result.get("route_not_found_message")

    if not routes:
        base = "Не удалось найти релевантные маршруты по вашему запросу."
        reason_block = ""
        if not_found_message:
            reason_block = f"\n\nПричина: {not_found_message}"
            if not_found_reason:
                reason_block += f"\nКод причины: `{not_found_reason}`"
        if notices:
            return (
                f"{base}{reason_block}\n\nПричины/заметки:\n- " + "\n- ".join(notices)
            )
        return f"{base}{reason_block}"

    header = f"Найдено релевантных маршрутов: **{len(routes)}**"
    body = _render_routes_markdown(routes)
    if notices:
        body = f"{body}\n\nЗаметки:\n- " + "\n- ".join(notices)
    return f"{header}\n\n{body}"


def _render_attractions_map_and_cards(
    items: list[dict[str, Any]],
    dest_label: str,
    dest_country: str | None,
) -> None:
    """Карта OSM (схема порядка точек) + картинки из Википедии при наличии."""
    try:
        from backend.attractions_ui import (
            build_attractions_folium_map,
            enrich_attractions_for_ui,
        )
    except ImportError:
        for i, a in enumerate(items, 1):
            st.markdown(f"**{i}. {a.get('name', '—')}**  \n{a.get('summary', '')}")
        return

    with st.spinner("Координаты (OpenStreetMap) и превью (Википедия)…"):
        enriched = enrich_attractions_for_ui(items, dest_label, dest_country)

    st.caption(
        "Карта: условный маршрут по порядку списка (не навигация). "
        "Данные © OpenStreetMap. Фото — Википедия при наличии статьи."
    )
    m = build_attractions_folium_map(enriched)
    if m is not None:
        try:
            from streamlit_folium import st_folium

            st_folium(m, width=None, height=440, returned_objects=[])
        except ImportError:
            st.info("Для карты установите: `pip install streamlit-folium folium`")
    else:
        st.caption("Координаты не получены — карта недоступна (проверьте сеть и лимиты Nominatim).")

    for i, a in enumerate(enriched, 1):
        name = a.get("name", "—")
        summary = a.get("summary", "")
        img = a.get("image_url")
        if img:
            c_img, c_txt = st.columns((1, 2))
            with c_img:
                st.image(img, use_container_width=True)
            with c_txt:
                st.markdown(f"**{i}. {name}**")
                st.write(summary)
        else:
            st.markdown(f"**{i}. {name}**")
            st.write(summary)


def _build_attractions_plain_text(att: dict[str, Any]) -> str:
    if not att or att.get("skipped"):
        return ""
    items = att.get("attractions") or []
    if not items:
        err = att.get("error")
        return f"(достопримечательности: не удалось — {err})" if err else ""
    lines: list[str] = []
    for i, a in enumerate(items, 1):
        name = a.get("name") or "—"
        summary = a.get("summary") or ""
        lines.append(f"{i}. {name}: {summary}")
    return "\n".join(lines)


def _build_hotels_text(result: dict[str, Any]) -> str:
    hotels = result.get("hotels") or []
    notices = result.get("notices") or []
    not_found_reason = result.get("hotel_not_found_reason")
    not_found_message = result.get("hotel_not_found_message")

    if not hotels:
        base = "Не удалось найти релевантные отели по вашему запросу."
        reason_block = ""
        if not_found_message:
            reason_block = f"\n\nПричина: {not_found_message}"
            if not_found_reason:
                reason_block += f"\nКод причины: `{not_found_reason}`"
        if notices:
            return (
                f"{base}{reason_block}\n\nПричины/заметки:\n- " + "\n- ".join(notices)
            )
        return f"{base}{reason_block}"

    header = f"Найдено релевантных отелей: **{len(hotels)}**"
    body = _render_hotels_markdown(hotels)
    if notices:
        body = f"{body}\n\nЗаметки:\n- " + "\n- ".join(notices)
    return f"{header}\n\n{body}"


def _build_extraction_text(extracted: dict[str, Any]) -> str:
    important_fields = [
        ("origin_city", "Город вылета"),
        ("origin_iata", "IATA вылета"),
        ("destination_city", "Город назначения"),
        ("destination_iata", "IATA назначения"),
        ("departure_date", "Дата вылета"),
        ("return_date", "Дата обратно"),
        ("departure_month", "Месяц вылета"),
        ("trip_days", "Длительность"),
        ("budget", "Бюджет"),
        ("currency", "Валюта"),
        ("passengers", "Пассажиры"),
        ("direct_only", "Только прямые"),
        ("one_way", "В одну сторону"),
    ]
    rows = []
    for field, title in important_fields:
        value = extracted.get(field)
        if value is not None and value != "":
            rows.append(f"- {title}: `{value}`")

    if not rows:
        return "LLM не извлек явные параметры (почти все поля пустые)."
    return "LLM извлек параметры для поиска:\n" + "\n".join(rows)


def main() -> None:
    st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")
    st.title("AI Travel Planner")
    st.caption(
        "Запрос в чат или ручные фильтры → извлечение параметров (LLM) → поиск билетов и отелей. "
        "Ниже — **таблицы сравнения**, **графики цен** и **рекомендуемые** варианты по простым правилам."
    )
    user_id = ensure_session_user()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("Настройки поиска")
        default_origin_city = st.text_input("Город вылета по умолчанию", value="Moscow")
        default_origin_iata = st.text_input("IATA вылета (опционально)", value="")
        default_destination_city = st.text_input("Город назначения по умолчанию", value="")
        max_results = st.slider("Количество вариантов", min_value=1, max_value=10, value=DEFAULT_MAX_RESULTS)
        use_langgraph = st.checkbox(
            "LangGraph: полный цикл (extract → поиск → маршрут → guardrail, до 3 ретраев)",
            value=False,
            help="См. backend/agent_graph.py — оркестратор из docs/specs/agent-orchestrator.md",
        )

        st.divider()
        st.subheader("Ручные фильтры (опционально)")
        manual_scope_label = st.selectbox(
            "Режим поиска по фильтрам",
            ["Авто (туры)", "Только билеты", "Только отели", "Туры (билеты + отели)"],
            index=0,
        )
        with st.expander("Открыть фильтры"):
            manual_origin_city = st.text_input("Фильтр: город вылета", value="")
            manual_origin_iata = st.text_input("Фильтр: IATA вылета", value="")
            manual_destination_city = st.text_input("Фильтр: город назначения", value="")
            manual_destination_iata = st.text_input("Фильтр: IATA назначения", value="")
            manual_departure_date = st.text_input("Фильтр: дата отправки (YYYY-MM-DD)", value="")
            manual_return_date = st.text_input("Фильтр: дата обратного прилета (YYYY-MM-DD)", value="")
            manual_trip_days = st.number_input("Фильтр: длительность (дней)", min_value=0, max_value=60, value=0)
            manual_passengers = st.number_input("Фильтр: кол-во человек", min_value=1, max_value=9, value=1)
            manual_adults = st.number_input("Фильтр: взрослых", min_value=0, max_value=9, value=0)
            manual_children = st.number_input("Фильтр: детей", min_value=0, max_value=8, value=0)
            manual_infants = st.number_input("Фильтр: младенцев", min_value=0, max_value=8, value=0)
            manual_budget = st.number_input("Фильтр: бюджет (0 = не учитывать)", min_value=0.0, value=0.0)
            manual_currency = st.selectbox("Фильтр: валюта", ["RUB", "EUR", "USD"], index=0)
            manual_direct_only = st.selectbox(
                "Фильтр: только прямые билеты",
                ["Не важно", "Только прямые", "Можно с пересадками"],
                index=0,
            )
            manual_one_way = st.selectbox("Фильтр: билет обратно", ["Не важно", "Да", "Нет"], index=0)
            manual_max_stops = st.selectbox("Фильтр: макс. пересадок", ["Не важно", "0", "1", "2", "3"], index=0)
            manual_cabin_class = st.selectbox(
                "Фильтр: класс обслуживания",
                ["Не важно", "Economy", "Business", "First"],
                index=0,
            )
            manual_include_baggage = st.selectbox("Фильтр: багаж", ["Не важно", "Да", "Нет"], index=0)
            manual_flexible_dates = st.selectbox("Фильтр: гибкие даты", ["Не важно", "Да", "Нет"], index=0)
            manual_pref_dep_time = st.selectbox(
                "Фильтр: время вылета",
                ["Не важно", "morning", "day", "evening", "night"],
                index=0,
            )
            manual_preferred_airlines = st.text_input(
                "Фильтр: предпочитаемые авиалинии (через запятую)",
                value="",
            )
            manual_excluded_airlines = st.text_input(
                "Фильтр: исключить авиалинии (через запятую)",
                value="",
            )

        run_filter_search = st.button("Искать по фильтрам", use_container_width=True)
        st.divider()
        st.caption(f"Extraction fast model: `{FAST_EXTRACTION_MODEL}`")
        st.caption(f"Extraction strong model: `{STRONG_EXTRACTION_MODEL}`")
        st.caption(f"LLM: `ChatOpenAI` → `{OPENAI_URL}` · ключ: `{bool(OPENAI_API_KEY)}`")
        if st.button("Очистить диалог", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("meta"):
                st.caption(message["meta"])

    user_text = st.chat_input("Напишите запрос, например: Хочу в Рим на 5 дней в июне, бюджет 1200€")
    text_mode = bool(user_text and user_text.strip())
    filters_mode = bool(run_filter_search) and not text_mode
    if not text_mode and not filters_mode:
        return

    if text_mode and run_filter_search:
        st.info("Обнаружен текстовый запрос: ручные фильтры проигнорированы (текст в приоритете).")

    manual_scope_map = {
        "Авто (туры)": "both",
        "Только билеты": "flights",
        "Только отели": "hotels",
        "Туры (билеты + отели)": "both",
    }
    manual_scope = manual_scope_map.get(manual_scope_label, "both")

    manual_filters: dict[str, Any] = {
        "origin_city": manual_origin_city,
        "origin_iata": manual_origin_iata,
        "destination_city": manual_destination_city,
        "destination_iata": manual_destination_iata,
        "departure_date": manual_departure_date,
        "return_date": manual_return_date,
        "trip_days": manual_trip_days,
        "passengers": manual_passengers,
        "adults": manual_adults,
        "children": manual_children,
        "infants": manual_infants,
        "budget": manual_budget,
        "currency": manual_currency,
        "direct_only": manual_direct_only,
        "one_way": manual_one_way,
        "max_stops": manual_max_stops,
        "cabin_class": manual_cabin_class,
        "include_baggage": manual_include_baggage,
        "flexible_dates": manual_flexible_dates,
        "preferred_departure_time": manual_pref_dep_time,
        "preferred_airlines": manual_preferred_airlines,
        "excluded_airlines": manual_excluded_airlines,
    }

    effective_user_text = user_text if text_mode else _build_manual_user_text(manual_filters, manual_scope_label)
    try:
        effective_user_text = sanitize_user_input(effective_user_text)
    except GuardrailViolation as exc:
        st.warning(str(exc))
        return

    conversation_context = (
        format_conversation_for_extraction(st.session_state.messages) if text_mode else None
    )
    st.session_state.messages.append({"role": "user", "content": effective_user_text})
    with st.chat_message("user"):
        st.markdown(effective_user_text)

    if use_langgraph:
        try:
            with st.chat_message("assistant"):
                with st.spinner("LangGraph: полный цикл планирования..."):
                    lg_result = run_travel_planning_graph(
                        effective_user_text,
                        default_origin_city=default_origin_city or None,
                        default_destination_city=default_destination_city or None,
                        default_origin_iata=default_origin_iata or None,
                        max_results=max_results,
                        thread_id=str(uuid.uuid4()),
                        conversation_context=conversation_context,
                        user_id=user_id,
                    )
                st.markdown(lg_result.get("final_markdown") or "_Нет текста результата_")
                em = lg_result.get("extraction_meta") or {}
                _render_llm_metrics_expanders(
                    extraction=em.get("extraction_llm_metrics"),
                    attractions=(lg_result.get("attractions_result") or {}).get("llm_metrics"),
                    itinerary=lg_result.get("itinerary_llm_metrics"),
                )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": lg_result.get("final_markdown") or "",
                    "meta": "langgraph | agent_graph.run_travel_planning_graph",
                }
            )
        except GuardrailViolation as exc:
            with st.chat_message("assistant"):
                st.warning(str(exc))
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": str(exc),
                    "meta": f"guardrail ({getattr(exc, 'code', '')})",
                }
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            with st.chat_message("assistant"):
                st.error(err)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"Ошибка LangGraph: `{err}`",
                    "meta": "langgraph error",
                }
            )
        return

    if text_mode:
        try:
            with st.chat_message("assistant"):
                with st.spinner("LLM извлекает параметры поездки..."):
                    extracted, extraction_meta = extract_trip_query(
                        effective_user_text,
                        return_metadata=True,
                        conversation_context=conversation_context,
                        user_id=user_id,
                    )
                    extraction_text = _build_extraction_text(extracted.model_dump())
                    st.markdown(extraction_text)
                    st.caption(
                        "LLM extraction model: "
                        f"{extraction_meta.get('used_model', 'unknown')} | "
                        f"fallback: {extraction_meta.get('used_fallback', False)} | "
                        f"reason: {extraction_meta.get('fallback_reason')}"
                    )
                    _render_llm_metrics_expanders(
                        extraction=extraction_meta.get("extraction_llm_metrics"),
                    )
                    # with st.expander("JSON извлеченных сущностей"):
                    #     st.json(extracted.model_dump())
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": extraction_text,
                            "meta": (
                                "extraction model: "
                                f"{extraction_meta.get('used_model', 'unknown')}, "
                                f"fallback: {extraction_meta.get('used_fallback', False)}"
                            ),
                        }
                    )
        except GuardrailViolation as exc:
            error_text = str(exc)
            with st.chat_message("assistant"):
                st.warning(error_text)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_text,
                    "meta": "guardrail",
                }
            )
            return
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            with st.chat_message("assistant"):
                st.error(error_text)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"Ошибка LLM-извлечения перед поиском маршрутов: `{error_text}`",
                    "meta": "extraction error",
                }
            )
            return
        scope = detect_search_scope(effective_user_text)
    else:
        try:
            extracted = _build_manual_trip_query(manual_filters)
            extraction_meta = {
                "used_model": "manual_filters",
                "used_fallback": False,
                "fallback_reason": "filters_mode",
            }
            with st.chat_message("assistant"):
                extraction_text = _build_extraction_text(extracted.model_dump())
                st.markdown(extraction_text)
                st.caption("Источник параметров: ручные фильтры (LLM не вызывался).")
                # with st.expander("JSON примененных фильтров"):
                #     st.json(extracted.model_dump())
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": extraction_text,
                        "meta": "extraction source: manual filters",
                    }
                )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            with st.chat_message("assistant"):
                st.error(error_text)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"Ошибка в ручных фильтрах: `{error_text}`",
                    "meta": "filters error",
                }
            )
            return
        scope = manual_scope

    with st.chat_message("assistant"):
        with st.spinner("Ищу релевантные варианты по API..."):
            scope_label = {
                "flights": "только билеты",
                "hotels": "только отели",
                "both": "туры (билеты + отели)",
            }.get(scope, scope)
            st.caption(f"Определен режим запроса: **{scope_label}**")

            response_parts: list[str] = []
            errors: list[str] = []
            flights_result: dict[str, Any] | None = None
            hotels_result: dict[str, Any] | None = None
            best_route_i: int | None = None
            best_hotel_i: int | None = None

            if scope in ("flights", "both"):
                try:
                    flights_result = search_routes_from_extracted(
                        user_request=effective_user_text,
                        extracted=extracted,
                        extraction_meta=extraction_meta,
                        default_origin_city=default_origin_city or None,
                        default_origin_iata=default_origin_iata or None,
                        max_results=max_results,
                    )
                    flights_text = _build_flights_text(flights_result)
                    st.markdown("### Билеты")
                    st.markdown(flights_text)
                    routes = flights_result.get("routes") or []
                    if routes:
                        best_route_i = render_flights_dashboard(routes)
                        with st.expander("Карточки маршрутов (ссылки на билеты)"):
                            _render_routes_widgets(routes)
                    # with st.expander("JSON ответа по билетам"):
                    #     st.json(flights_result)
                    response_parts.append("Билеты:\n" + flights_text)
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"
                    errors.append(f"flights: {error_text}")
                    st.error(f"Ошибка при поиске билетов: {error_text}")

            if scope in ("hotels", "both"):
                try:
                    hotels_result = search_hotels_from_extracted(
                        user_request=effective_user_text,
                        extracted=extracted,
                        extraction_meta=extraction_meta,
                        default_destination_city=default_destination_city or None,
                        max_results=max_results,
                    )
                    hotels_text = _build_hotels_text(hotels_result)
                    st.markdown("### Отели")
                    st.markdown(hotels_text)
                    hotels = hotels_result.get("hotels") or []
                    if hotels:
                        best_hotel_i = render_hotels_dashboard(hotels)
                        with st.expander("Карточки отелей (ссылки на бронирование)"):
                            _render_hotels_widgets(hotels)
                    # with st.expander("JSON ответа по отелям"):
                    #     st.json(hotels_result)
                    response_parts.append("Отели:\n" + hotels_text)
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"
                    errors.append(f"hotels: {error_text}")
                    st.error(f"Ошибка при поиске отелей: {error_text}")

            if (
                scope == "both"
                and flights_result
                and hotels_result
                and best_route_i is not None
                and best_hotel_i is not None
            ):
                render_package_summary(
                    flights_result.get("routes") or [],
                    hotels_result.get("hotels") or [],
                    extracted,
                    best_route_i,
                    best_hotel_i,
                )

            dest_label, dest_country = destination_label_for_attractions(
                extracted,
                default_destination_city or None,
            )
            if dest_label:
                with st.spinner("Подбираю достопримечательности..."):
                    att_result = suggest_city_attractions(
                        dest_label,
                        dest_country,
                        max_items=8,
                    )
                st.markdown("### Куда сходить")
                if att_result.get("skipped"):
                    st.caption("Список достопримечательностей отключён (`SKIP_CITY_ATTRACTIONS`).")
                elif att_result.get("error") and not (att_result.get("attractions") or []):
                    st.warning(
                        "Не удалось сформировать список достопримечательностей: "
                        f"{att_result['error']}"
                    )
                else:
                    items = att_result.get("attractions") or []
                    if items:
                        st.caption(
                            "Подсказки носят информационный характер; перед поездкой уточняйте часы работы и билеты."
                        )
                        _render_attractions_map_and_cards(items, dest_label, dest_country)
                        if att_result.get("model"):
                            st.caption(f"Модель подбора: `{att_result['model']}`")
                    elif not att_result.get("skipped"):
                        st.caption("Список мест пуст.")
                att_plain = _build_attractions_plain_text(att_result)
                if att_plain:
                    response_parts.append("Достопримечательности:\n" + att_plain)

            if not response_parts and errors:
                combined_text = "Ошибки при поиске:\n- " + "\n- ".join(errors)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": combined_text,
                        "meta": f"api errors | scope: {scope}",
                    }
                )
            else:
                combined_text = "\n\n".join(response_parts)
                if errors:
                    combined_text += "\n\nОшибки:\n- " + "\n- ".join(errors)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": combined_text,
                        "meta": f"api response | scope: {scope}",
                    }
                )


if __name__ == "__main__":
    main()

