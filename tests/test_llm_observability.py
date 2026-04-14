"""Unit tests for LLM observability helpers (no live API)."""

from backend.llm_observability import build_metrics_dict, estimate_cost_usd


def test_normalize_usage_via_build_metrics():
    m = build_metrics_dict(
        stage="t",
        model="m",
        t_start=0.0,
        t_first_signal=None,
        t_end=1.0,
        meta={"token_usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}},
    )
    assert m["input_tokens"] == 10 and m["output_tokens"] == 20 and m["total_tokens"] == 30


def test_estimate_cost_from_env(monkeypatch):
    monkeypatch.setenv("LLM_DEFAULT_INPUT_USD_PER_1K", "0.001")
    monkeypatch.setenv("LLM_DEFAULT_OUTPUT_USD_PER_1K", "0.002")
    c = estimate_cost_usd(
        prompt_tokens=1000,
        completion_tokens=500,
        model="gpt-test",
        meta={},
    )
    assert c is not None
    assert abs(c - (0.001 + 0.001)) < 1e-9


def test_build_metrics_dict_tpot():
    meta = {"token_usage": {"prompt_tokens": 1, "completion_tokens": 4, "total_tokens": 5}}
    m = build_metrics_dict(
        stage="x",
        model="m",
        t_start=0.0,
        t_first_signal=1.0,
        t_end=3.0,
        meta=meta,
    )
    assert m["ttft_sec"] == 1.0
    assert m["latency_sec"] == 3.0
    assert m["input_tokens"] == 1
    assert m["output_tokens"] == 4
    # decode window 2.0 / 4 = 0.5
    assert m["tpot_sec"] is not None and abs(m["tpot_sec"] - 0.5) < 1e-6


def test_cost_from_metadata():
    c = estimate_cost_usd(
        prompt_tokens=None,
        completion_tokens=None,
        model=None,
        meta={"cost": 0.042},
    )
    assert c == 0.042
