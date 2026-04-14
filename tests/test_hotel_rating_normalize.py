"""Нормализация гостевого рейтинга отелей (0–10 vs 0–100)."""

from backend.hotels import normalize_hotel_guest_rating, sort_hotels_by_price_then_rating


def test_normalize_ten_scale() -> None:
    assert abs(normalize_hotel_guest_rating(8.4) - 8.4) < 1e-6


def test_normalize_hundred_scale() -> None:
    assert abs(normalize_hotel_guest_rating(87) - 8.7) < 1e-6


def test_sort_secondary_rating() -> None:
    hotels = [
        {"price_per_night": 100.0, "rating": 80.0},
        {"price_per_night": 100.0, "rating": 9.0},
    ]
    sort_hotels_by_price_then_rating(hotels)
    # при одной цене выше нормализованный рейтинг (9.0 vs 8.0)
    assert hotels[0]["rating"] == 9.0
