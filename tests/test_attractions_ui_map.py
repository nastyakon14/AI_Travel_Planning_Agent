import os

from backend.attractions_ui import build_attractions_folium_map, enrich_attractions_for_ui


def test_build_map_two_points() -> None:
    m = build_attractions_folium_map(
        [
            {"name": "A", "lat": 55.75, "lon": 37.62},
            {"name": "B", "lat": 55.76, "lon": 37.63},
        ],
    )
    assert m is not None


def test_enrich_skipped(monkeypatch) -> None:
    monkeypatch.setenv("SKIP_ATTRACTION_ENRICHMENT", "1")
    out = enrich_attractions_for_ui(
        [{"name": "X", "summary": "y"}],
        "City",
        None,
    )
    assert out[0].get("lat") is None
