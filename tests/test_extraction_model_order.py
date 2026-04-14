"""Каскад моделей извлечения (ChatOpenAI, без LiteLLM)."""

from backend import travel_agent as ta


def test_extraction_model_order_same_model(monkeypatch) -> None:
    monkeypatch.setattr(ta, "FAST_EXTRACTION_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(ta, "STRONG_EXTRACTION_MODEL", "gpt-4o-mini")
    assert ta.extraction_model_order("anything", None) == ["gpt-4o-mini"]


def test_complexity_hint_long_text() -> None:
    t = "x" * 2000 + " виза"
    assert ta._complexity_hint(t, None) >= 0.45


def test_extraction_model_order_complex_first_strong_then_fast(monkeypatch) -> None:
    monkeypatch.setattr(ta, "FAST_EXTRACTION_MODEL", "fast-m")
    monkeypatch.setattr(ta, "STRONG_EXTRACTION_MODEL", "strong-m")
    monkeypatch.setenv("EXTRACTION_COMPLEX_FIRST", "1")
    long_text = "x" * 1600 + " виза"
    assert ta.extraction_model_order(long_text, None) == ["strong-m", "fast-m"]


def test_extraction_model_order_default_fast_then_strong(monkeypatch) -> None:
    monkeypatch.setattr(ta, "FAST_EXTRACTION_MODEL", "fast-m")
    monkeypatch.setattr(ta, "STRONG_EXTRACTION_MODEL", "strong-m")
    monkeypatch.delenv("EXTRACTION_COMPLEX_FIRST", raising=False)
    assert ta.extraction_model_order("коротко", None) == ["fast-m", "strong-m"]


def test_extraction_model_order_complex_first_but_simple_query(monkeypatch) -> None:
    monkeypatch.setattr(ta, "FAST_EXTRACTION_MODEL", "fast-m")
    monkeypatch.setattr(ta, "STRONG_EXTRACTION_MODEL", "strong-m")
    monkeypatch.setenv("EXTRACTION_COMPLEX_FIRST", "1")
    assert ta.extraction_model_order("коротко", None) == ["fast-m", "strong-m"]
