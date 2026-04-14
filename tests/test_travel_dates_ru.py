"""Регрессия: русские даты и длительность для отелей (мае, на неделю)."""

from backend.travel_agent import (
    TripQuery,
    _detect_month_from_text,
    _infer_dates,
    _infer_trip_days_from_text,
)


def test_mae_month_prepositional() -> None:
    assert _detect_month_from_text("в Кисловодске в мае") == 5


def test_na_nedelyu_duration() -> None:
    assert _infer_trip_days_from_text("на семерых на неделю") == 7


def test_infer_may_plus_week() -> None:
    q = TripQuery(destination_city="Кисловодск", passengers=7)
    d0, d1 = _infer_dates("отели в Кисловодске в мае на семерых на неделю", q)
    assert d0 is not None and d1 is not None
    assert (d1 - d0).days == 7
