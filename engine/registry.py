"""Champion/challenger registry — the enforcement point for the accuracy gates.

No model reaches a user except through this registry. Models register via
the `register` decorator; only classes that pass test_contract_compliance.py
may be registered in practice, but that enforcement is a test-suite concern,
not a runtime one — this module only holds the mapping.
"""
from __future__ import annotations

from typing import Type

from engine.models.base import ForecastModel

REGISTRY: dict[str, Type[ForecastModel]] = {}


def register(model_cls: Type[ForecastModel]) -> Type[ForecastModel]:
    model_id = getattr(model_cls, "model_id", None)
    if not model_id:
        raise ValueError(f"{model_cls.__name__} must define model_id before registration")
    if model_id in REGISTRY:
        raise ValueError(f"model_id '{model_id}' is already registered")
    REGISTRY[model_id] = model_cls
    return model_cls
