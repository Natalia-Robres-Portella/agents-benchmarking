"""Exact match validator — case-insensitive string equality."""
from __future__ import annotations

from src.schema import TaskInstance
from src.tasks.base import TaskValidator


class ExactMatchValidator(TaskValidator):
    @property
    def name(self) -> str:
        return "exact_match"

    def validate(self, prediction, gold, task: TaskInstance) -> float:
        return 1.0 if str(prediction).strip().lower() == str(gold).strip().lower() else 0.0
