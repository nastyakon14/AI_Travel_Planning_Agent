"""
Экспорт метрик в Prometheus (pull /metrics). Включается переменной PROMETHEUS_METRICS_PORT.

Счётчики: запросы планирования, токены, стоимость, «коды» исходов.
Гистограммы: полная латентность LLM, TTFT (prefill), фаза decode, ITL (между токенами), длительность графа.
CPU/RSS: gauge через psutil (если установлен).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_METRICS_STARTED = False
_PROCESS_COLLECTOR_REGISTERED = False
_METRICS_LOCK = threading.Lock()

# Buckets до ~2 мин — LLM и граф
_LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
)

try:
    from prometheus_client import Counter, Histogram, REGISTRY, start_http_server
    from prometheus_client.core import GaugeMetricFamily

    _PROM = True
except ImportError:  # pragma: no cover
    _PROM = False
    Counter = Histogram = None  # type: ignore[misc, assignment]
    REGISTRY = None
    start_http_server = None


def _safe_stage(stage: str | None) -> str:
    s = (stage or "unknown").replace('"', "")[:64]
    return s


if _PROM:
    planning_requests = Counter(
        "travel_planning_requests_total",
        "Запуски графа планирования",
        ["outcome", "http_code"],
    )

    planning_duration = Histogram(
        "travel_planning_duration_seconds",
        "Полное время выполнения run_travel_planning_graph",
        ["outcome"],
        buckets=_LATENCY_BUCKETS,
    )

    llm_wall_seconds = Histogram(
        "travel_llm_wall_seconds",
        "Полная латентность одного LLM-вызова (wall clock)",
        ["stage"],
        buckets=_LATENCY_BUCKETS,
    )

    llm_prefill_seconds = Histogram(
        "travel_llm_prefill_seconds",
        "TTFT — время до первого токена/чанка (prefill)",
        ["stage"],
        buckets=_LATENCY_BUCKETS,
    )

    llm_decode_phase_seconds = Histogram(
        "travel_llm_decode_phase_seconds",
        "Фаза decode: от первого токена до конца генерации",
        ["stage"],
        buckets=_LATENCY_BUCKETS,
    )

    llm_inter_token_latency_seconds = Histogram(
        "travel_llm_inter_token_latency_seconds",
        "ITL — средняя задержка на выходной токен (decode / completion_tokens)",
        ["stage"],
        buckets=(
            1e-5,
            5e-5,
            1e-4,
            5e-4,
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
            0.5,
            1.0,
            5.0,
        ),
    )

    llm_input_tokens = Counter(
        "travel_llm_input_tokens_total",
        "Входные токены LLM",
        ["stage"],
    )

    llm_output_tokens = Counter(
        "travel_llm_output_tokens_total",
        "Выходные токены LLM",
        ["stage"],
    )

    llm_cost_usd = Counter(
        "travel_llm_cost_usd_total",
        "Накопленная оценка стоимости LLM (USD)",
        ["stage"],
    )

    # ---- Бизнес-метрики (Travel Analytics) ----
    trip_passengers = Histogram(
        "travel_trip_passengers",
        "Количество пассажиров в запросе",
        buckets=(1, 2, 3, 4, 5, 6, 7, 8, 10, 12),
    )

    trip_budget = Histogram(
        "travel_trip_budget",
        "Бюджет поездки (в валюте запроса)",
        ["currency"],
        buckets=(100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 20000),
    )

    trip_duration_days = Histogram(
        "travel_trip_duration_days",
        "Длительность поездки (дней)",
        buckets=(1, 2, 3, 4, 5, 7, 10, 14, 21, 30, 45, 60),
    )

    origin_city_counter = Counter(
        "travel_origin_city_total",
        "Города вылета (топ направлений)",
        ["city"],
    )

    destination_city_counter = Counter(
        "travel_destination_city_total",
        "Города назначения (топ направлений)",
        ["city"],
    )

    trip_outcome_with_budget = Counter(
        "travel_outcome_with_budget_total",
        "Исходы запросов с разбивкой по бюджету",
        ["outcome", "budget_range"],
    )

    search_scope_counter = Counter(
        "travel_search_scope_total",
        "Тип поиска: flights_only, hotels_only, both",
        ["scope"],
    )

    retry_count_histogram = Histogram(
        "travel_guardrail_retries",
        "Количество retry-попыток guardrail",
        buckets=(0, 1, 2, 3),
    )

    total_trip_cost_usd = Histogram(
        "travel_total_trip_cost_usd",
        "Итоговая стоимость поездки (USD)",
        buckets=(100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000),
    )

    response_token_count = Histogram(
        "travel_response_token_count",
        "Количество токенов в итоговом ответе (маршрут)",
        buckets=(100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 8000, 10000),
    )

    generation_tokens_per_sec = Histogram(
        "travel_generation_tokens_per_sec",
        "Скорость генерации: токенов в секунду",
        buckets=(5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200),
    )


class _ProcessResourceCollector:
    """CPU (все ядра) и RSS текущего процесса."""

    def collect(self):
        if not _PROM:
            return
        try:
            import psutil

            cpu = psutil.cpu_percent(interval=None)
            rss = psutil.Process(os.getpid()).memory_info().rss
            g1 = GaugeMetricFamily(
                "travel_host_cpu_percent",
                "Загрузка CPU хоста (все ядра), процент",
                value=float(cpu),
            )
            g2 = GaugeMetricFamily(
                "travel_process_resident_memory_bytes",
                "RSS процесса приложения",
                value=float(rss),
            )
            yield g1
            yield g2
        except Exception:  # pragma: no cover
            return


def start_metrics_server(port: int | None = None) -> None:
    """Запускает HTTP /metrics в фоновом потоке (идемпотентно)."""
    global _METRICS_STARTED, _PROCESS_COLLECTOR_REGISTERED
    if not _PROM:
        logger.warning("prometheus_client не установлен; метрики отключены")
        return
    p = port or int(os.getenv("PROMETHEUS_METRICS_PORT", "9090"))
    with _METRICS_LOCK:
        if _METRICS_STARTED:
            return
        if not _PROCESS_COLLECTOR_REGISTERED:
            try:
                REGISTRY.register(_ProcessResourceCollector())
                _PROCESS_COLLECTOR_REGISTERED = True
            except (ValueError, AttributeError):
                _PROCESS_COLLECTOR_REGISTERED = True
        start_http_server(p)
        _METRICS_STARTED = True
        logger.info("Prometheus metrics listening on :%s/metrics", p)


def record_llm_metrics(m: dict[str, Any]) -> None:
    """Записывает метрики одного LLM-вызова (словарь из llm_observability.build_metrics_dict)."""
    if not _PROM:
        return
    stage = _safe_stage(m.get("stage"))

    lat = m.get("latency_sec")
    if isinstance(lat, (int, float)) and lat >= 0:
        llm_wall_seconds.labels(stage=stage).observe(float(lat))

    ttft = m.get("ttft_sec")
    if isinstance(ttft, (int, float)) and ttft >= 0:
        llm_prefill_seconds.labels(stage=stage).observe(float(ttft))

    dec = m.get("decode_sec")
    if isinstance(dec, (int, float)) and dec >= 0:
        llm_decode_phase_seconds.labels(stage=stage).observe(float(dec))

    itl = m.get("itl_sec") or m.get("tpot_sec")
    if isinstance(itl, (int, float)) and itl >= 0:
        llm_inter_token_latency_seconds.labels(stage=stage).observe(float(itl))

    it = m.get("input_tokens")
    if isinstance(it, int) and it > 0:
        llm_input_tokens.labels(stage=stage).inc(it)

    ot = m.get("output_tokens")
    if isinstance(ot, int) and ot > 0:
        llm_output_tokens.labels(stage=stage).inc(ot)

    cost = m.get("cost_usd")
    if isinstance(cost, (int, float)) and cost > 0:
        llm_cost_usd.labels(stage=stage).inc(float(cost))


def record_planning_run(
    *,
    outcome: str,
    duration_sec: float,
    http_code: str = "200",
) -> None:
    """
    outcome: ok | error | guardrail
    http_code: условный код ответа (200 / 403 / 500) для дашбордов.
    """
    if not _PROM:
        return
    planning_requests.labels(outcome=outcome, http_code=http_code).inc()
    if duration_sec >= 0:
        planning_duration.labels(outcome=outcome).observe(duration_sec)


def record_trip_business_metrics(
    *,
    passengers: int | None = None,
    budget: float | None = None,
    currency: str | None = None,
    trip_days: int | None = None,
    origin_city: str | None = None,
    destination_city: str | None = None,
    outcome: str = "ok",
    budget_range: str | None = None,
    search_scope: str | None = None,
    retry_count: int | None = None,
    total_cost_usd: float | None = None,
    response_tokens: int | None = None,
    generation_speed_tps: float | None = None,
) -> None:
    """Записывает бизнес-метрики одной сессии планирования."""
    if not _PROM:
        return

    if passengers is not None and passengers > 0:
        trip_passengers.observe(passengers)

    if budget is not None and budget > 0:
        trip_budget.labels(currency=currency or "unknown").observe(budget)

    if trip_days is not None and trip_days > 0:
        trip_duration_days.observe(trip_days)

    if origin_city:
        origin_city_counter.labels(city=origin_city[:64]).inc()

    if destination_city:
        destination_city_counter.labels(city=destination_city[:64]).inc()

    if budget_range:
        trip_outcome_with_budget.labels(outcome=outcome, budget_range=budget_range).inc()

    if search_scope:
        search_scope_counter.labels(scope=search_scope).inc()

    if retry_count is not None and retry_count >= 0:
        retry_count_histogram.observe(retry_count)

    if total_cost_usd is not None and total_cost_usd > 0:
        total_trip_cost_usd.observe(total_cost_usd)

    if response_tokens is not None and response_tokens > 0:
        response_token_count.observe(response_tokens)

    if generation_speed_tps is not None and generation_speed_tps > 0:
        generation_tokens_per_sec.observe(generation_speed_tps)


def budget_to_range(budget: float | None, currency: str | None = None) -> str:
    """Преобразует бюджет в категорию для дашборда."""
    if budget is None:
        return "unknown"
    # Нормализуем приблизительно к USD (грубо)
    rate = 1.0
    c = (currency or "USD").upper()
    if c == "EUR":
        rate = 1.08
    elif c == "RUB":
        rate = 0.011
    elif c == "GBP":
        rate = 1.27
    usd_equiv = budget * rate
    if usd_equiv < 500:
        return "<500"
    elif usd_equiv < 1000:
        return "500-1k"
    elif usd_equiv < 2000:
        return "1k-2k"
    elif usd_equiv < 5000:
        return "2k-5k"
    elif usd_equiv < 10000:
        return "5k-10k"
    else:
        return "10k+"


def outcome_to_http(outcome: str) -> str:
    if outcome == "ok":
        return "200"
    if outcome == "guardrail":
        return "403"
    return "500"
