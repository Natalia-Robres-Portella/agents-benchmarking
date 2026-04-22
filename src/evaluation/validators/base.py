"""
Validator registry — maps validator names from EvaluationConfig to TaskValidator instances.
Concrete validators (exact_match, fuzzy_match, etc.) register themselves here.
"""
from __future__ import annotations

from typing import Dict, Type

from src.tasks.base import TaskValidator


class ValidatorRegistry:
    def __init__(self) -> None:
        self._validators: Dict[str, Type[TaskValidator]] = {}

    def register(self, validator_cls: Type[TaskValidator]) -> None:
        self._validators[validator_cls().name] = validator_cls

    def get(self, name: str) -> TaskValidator:
        if name not in self._validators:
            raise KeyError(
                f"Validator '{name}' not registered. Available: {list(self._validators)}"
            )
        return self._validators[name]()

    def list_validators(self) -> list[str]:
        return list(self._validators)


VALIDATOR_REGISTRY = ValidatorRegistry()
