"""Serving: реестр моделей и метаданные деплоя (MLOps)."""

from .model_registry import ServingInfo, get_serving_info

__all__ = ["ServingInfo", "get_serving_info"]
