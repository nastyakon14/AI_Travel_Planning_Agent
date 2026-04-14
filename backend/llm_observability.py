"""
Метрики LLM: TTFT, TPOT, токены и оценка стоимости (наблюдаемость PoC).

TTFT — время от начала запроса до первого «сигнала» в потоке (контент или tool_calls delta).
TPOT — (время после первого сигнала) / max(1, completion_tokens); при отсутствии токенов — None.

Стоимость: если провайдер не отдаёт cost в metadata, можно задать цены через env
LLM_DEFAULT_INPUT_USD_PER_1K / LLM_DEFAULT_OUTPUT_USD_PER_1K (USD за 1000 токенов).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _env_float(name: str, default: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _chunk_has_signal(chunk: Any) -> bool:
    """Первый осмысленный фрагмент в потоке Chat Completions (текст или tool_calls)."""
    if getattr(chunk, "content", None):
        return True
    ak = getattr(chunk, "additional_kwargs", None) or {}
    if ak.get("tool_calls"):
        return True
    return False


def _normalize_usage(meta: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    """Извлекает prompt/completion/total tokens из response_metadata LangChain / OpenAI."""
    if not meta:
        return None, None, None
    usage = meta.get("token_usage") or meta.get("usage")
    if not isinstance(usage, dict):
        return None, None, None
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    tt = usage.get("total_tokens")
    pin = int(pt) if pt is not None else None
    cin = int(ct) if ct is not None else None
    tin = int(tt) if tt is not None else None
    return pin, cin, tin


def _cost_from_metadata(meta: dict[str, Any]) -> float | None:
    """Некоторые шлюзы (например LiteLLM) пробрасывают cost в metadata."""
    for key in ("cost", "response_cost", "total_cost"):
        v = meta.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    # langchain / litellm иногда кладут вложенные структуры
    hidden = meta.get("_hidden_params")
    if isinstance(hidden, dict):
        for key in ("response_cost", "cost"):
            v = hidden.get(key)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def estimate_cost_usd(
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    model: str | None,
    meta: dict[str, Any],
) -> float | None:
    c = _cost_from_metadata(meta)
    if c is not None:
        return round(c, 6)
    pin_per_1k = _env_float("LLM_DEFAULT_INPUT_USD_PER_1K", 0.0)
    pout_per_1k = _env_float("LLM_DEFAULT_OUTPUT_USD_PER_1K", 0.0)
    if pin_per_1k <= 0 and pout_per_1k <= 0:
        return None
    pt = int(prompt_tokens or 0)
    ct = int(completion_tokens or 0)
    return round((pt / 1000.0) * pin_per_1k + (ct / 1000.0) * pout_per_1k, 6)


def build_metrics_dict(
    *,
    stage: str,
    model: str | None,
    t_start: float,
    t_first_signal: float | None,
    t_end: float,
    meta: dict[str, Any],
) -> dict[str, Any]:
    pt, ct, tt = _normalize_usage(meta)
    latency = max(0.0, t_end - t_start)
    ttft = (t_first_signal - t_start) if t_first_signal is not None else None
    decode_window = (
        (t_end - t_first_signal) if t_first_signal is not None else None
    )
    tpot = None
    if decode_window is not None and ct is not None and ct > 0:
        tpot = decode_window / float(ct)
    cost = estimate_cost_usd(
        prompt_tokens=pt,
        completion_tokens=ct,
        model=model,
        meta=meta,
    )
    decode_sec = round(decode_window, 6) if decode_window is not None else None
    return {
        "stage": stage,
        "model": model,
        "latency_sec": round(latency, 6),
        "ttft_sec": round(ttft, 6) if ttft is not None else None,
        "decode_sec": decode_sec,
        "tpot_sec": round(tpot, 8) if tpot is not None else None,
        "itl_sec": round(tpot, 8) if tpot is not None else None,
        "input_tokens": pt,
        "output_tokens": ct,
        "total_tokens": tt,
        "cost_usd": cost,
    }


def log_llm_metrics(metrics: dict[str, Any]) -> None:
    try:
        from backend.prometheus_metrics import record_llm_metrics as _prom_llm

        _prom_llm(metrics)
    except Exception:
        pass
    if not logger.isEnabledFor(logging.INFO):
        return
    logger.info(
        "llm_metrics stage=%s model=%s latency_sec=%s ttft_sec=%s decode_sec=%s itl_sec=%s "
        "in=%s out=%s total=%s cost_usd=%s",
        metrics.get("stage"),
        metrics.get("model"),
        metrics.get("latency_sec"),
        metrics.get("ttft_sec"),
        metrics.get("decode_sec"),
        metrics.get("itl_sec"),
        metrics.get("input_tokens"),
        metrics.get("output_tokens"),
        metrics.get("total_tokens"),
        metrics.get("cost_usd"),
    )


def _merge_run_config(run_config: dict[str, Any] | None) -> dict[str, Any]:
    try:
        from backend.langfuse_tracing import merge_langchain_callbacks
    except ImportError:
        return dict(run_config or {})
    return merge_langchain_callbacks(run_config)


def stream_plain_text(
    llm: ChatOpenAI,
    prompt: str,
    *,
    stage: str,
    run_config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Потоковая генерация текста (одно пользовательское сообщение)."""
    messages: list[BaseMessage] = [HumanMessage(content=prompt)]
    cfg = _merge_run_config(run_config)
    t0 = time.perf_counter()
    first_at: float | None = None
    acc: Any = None
    for chunk in llm.stream(messages, config=cfg):
        if first_at is None and _chunk_has_signal(chunk):
            first_at = time.perf_counter()
        acc = chunk if acc is None else acc + chunk
    t1 = time.perf_counter()
    if acc is None:
        raise ValueError("empty LLM stream")
    text = str(getattr(acc, "content", "") or "")
    meta = getattr(acc, "response_metadata", None) or {}
    m = build_metrics_dict(
        stage=stage,
        model=getattr(llm, "model_name", None),
        t_start=t0,
        t_first_signal=first_at,
        t_end=t1,
        meta=meta if isinstance(meta, dict) else {},
    )
    log_llm_metrics(m)
    return text, m


def stream_structured_output(
    llm: ChatOpenAI,
    schema: type[T],
    messages: list[BaseMessage],
    *,
    stage: str,
    run_config: dict[str, Any] | None = None,
) -> tuple[T, dict[str, Any]]:
    """
    Структурированный вывод через with_structured_output(..., include_raw=True).

    Сначала stream (для TTFT); если parse отсутствует — fallback на invoke (часто нужен для LiteLLM / шлюзов).
    """
    chain = llm.with_structured_output(schema, include_raw=True)
    cfg = _merge_run_config(run_config)
    t0 = time.perf_counter()
    first_at: float | None = None
    last: dict[str, Any] | None = None
    direct: T | None = None
    for part in chain.stream(messages, config=cfg):
        if isinstance(part, dict):
            last = part
        elif isinstance(part, schema):
            direct = part
        if first_at is None and isinstance(part, dict):
            raw = part.get("raw")
            if raw is not None and _chunk_has_signal(raw):
                first_at = time.perf_counter()
    t_stream_end = time.perf_counter()

    parsed: Any = None
    raw_msg: Any = None
    if direct is not None:
        parsed = direct
    elif last and isinstance(last, dict):
        parsed = last.get("parsed")
        raw_msg = last.get("raw")

    if parsed is None:
        logger.info(
            "structured stream без parsed; fallback invoke (stage=%s)",
            stage,
        )
        t_inv0 = time.perf_counter()
        inv = chain.invoke(messages, config=cfg)
        t_inv1 = time.perf_counter()
        if isinstance(inv, dict):
            parsed = inv.get("parsed")
            raw_msg = inv.get("raw") or raw_msg
            err = inv.get("parsing_error")
            if parsed is None and os.getenv(
                "LLM_STRUCTURED_FALLBACK_FUNCTION_CALLING", "1"
            ).strip().lower() not in ("0", "false", "no", "off"):
                logger.info(
                    "retrying structured output with method=function_calling (stage=%s)",
                    stage,
                )
                chain_fc = llm.with_structured_output(
                    schema, include_raw=True, method="function_calling"
                )
                inv2 = chain_fc.invoke(messages, config=cfg)
                t_inv1 = time.perf_counter()
                if isinstance(inv2, dict):
                    parsed = inv2.get("parsed")
                    raw_msg = inv2.get("raw") or raw_msg
                    err = inv2.get("parsing_error") or err
                elif isinstance(inv2, schema):
                    parsed = inv2
                if parsed is None:
                    raise ValueError(
                        f"structured output parsing failed: {err!s}"
                        if err
                        else "structured output parsing failed (invoke returned no parsed object)"
                    )
            elif parsed is None:
                raise ValueError(
                    f"structured output parsing failed: {err!s}"
                    if err
                    else "structured output parsing failed (invoke returned no parsed object)"
                )
        elif isinstance(inv, schema):
            parsed = inv
        else:
            parsed = schema.model_validate(inv)
        meta_fb: dict[str, Any] = {}
        if raw_msg is not None:
            meta_fb = getattr(raw_msg, "response_metadata", None) or {}
            if not isinstance(meta_fb, dict):
                meta_fb = {}
        m = build_metrics_dict(
            stage=stage,
            model=getattr(llm, "model_name", None),
            t_start=t_inv0,
            t_first_signal=None,
            t_end=t_inv1,
            meta=meta_fb,
        )
        log_llm_metrics(m)
        out = parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
        return out, m

    if not isinstance(parsed, schema):
        parsed = schema.model_validate(parsed)  # type: ignore[assignment]
    meta: dict[str, Any] = {}
    if raw_msg is not None:
        meta = getattr(raw_msg, "response_metadata", None) or {}
        if not isinstance(meta, dict):
            meta = {}
    m = build_metrics_dict(
        stage=stage,
        model=getattr(llm, "model_name", None),
        t_start=t0,
        t_first_signal=first_at,
        t_end=t_stream_end,
        meta=meta,
    )
    log_llm_metrics(m)
    return parsed, m
