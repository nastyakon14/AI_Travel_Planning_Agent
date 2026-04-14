"""
Реестр моделей и билд-метаданные для воспроизводимости и наблюдаемости.

Правило: в production фиксируйте модели через env (без «плавающих» алиасов), см. docs/specs/serving-config.md.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ServingInfo:
    service_name: str
    build_version: str
    git_sha: str
    environment: str
    extraction_fast_model: str
    extraction_strong_model: str
    itinerary_model: str
    attractions_model: str
    llm_api_base: str  # без ключа; только host/путь для диагностики

    def to_public_dict(self) -> dict:
        d = asdict(self)
        return d


def _safe_base(url: str | None) -> str:
    if not url:
        return ""
    u = str(url).strip()
    return u.split("?")[0][:120]


def get_serving_info() -> ServingInfo:
    return ServingInfo(
        service_name=os.getenv("SERVING_SERVICE_NAME", "travel-planning-api"),
        build_version=os.getenv("BUILD_VERSION", os.getenv("DOCKER_TAG", "dev")),
        git_sha=os.getenv("GIT_SHA", os.getenv("SOURCE_COMMIT", "unknown"))[:40],
        environment=os.getenv("DEPLOY_ENV", os.getenv("ENV", "development")),
        extraction_fast_model=os.getenv("TRIP_EXTRACTION_FAST_MODEL", "gpt-4o-mini"),
        extraction_strong_model=os.getenv("TRIP_EXTRACTION_STRONG_MODEL", "gpt-4o"),
        itinerary_model=os.getenv("TRIP_ITINERARY_MODEL", os.getenv("TRIP_EXTRACTION_FAST_MODEL", "gpt-4o-mini")),
        attractions_model=os.getenv("TRIP_ATTRACTIONS_MODEL", os.getenv("TRIP_EXTRACTION_FAST_MODEL", "gpt-4o-mini")),
        llm_api_base=_safe_base(os.getenv("AGENTPLATFORM_API_BASE")),
    )
