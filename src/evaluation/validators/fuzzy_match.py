"""
Fuzzy match validator — token-level F1 score.

Mirrors the SQuAD / HotPotQA evaluation metric (Rajpurkar et al., 2016).
Both prediction and gold answer are tokenised, stop words removed, and
the token multiset overlap is computed as F1.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Counter as CounterType

from src.schema import TaskInstance
from src.tasks.base import TaskValidator

_STOP = frozenset({"a", "an", "the", "of", "in", "is", "it", "to", "and", "or", "was"})
_PUNCT = re.compile(r"[^\w\s]")


def _tokenise(text: str) -> CounterType[str]:
    tokens = _PUNCT.sub("", text.lower()).split()
    return Counter(t for t in tokens if t not in _STOP)


class FuzzyMatchValidator(TaskValidator):
    """
    Token-level F1 (0–1).  Empty prediction on non-empty gold → 0.
    Exact token match → 1.  Partial overlap → F1 of precision and recall.
    """

    @property
    def name(self) -> str:
        return "fuzzy_match"

    def validate(self, prediction, gold, task: TaskInstance) -> float:
        pred_c = _tokenise(str(prediction))
        gold_c = _tokenise(str(gold))

        if not pred_c and not gold_c:
            return 1.0
        if not pred_c or not gold_c:
            return 0.0

        common = sum((pred_c & gold_c).values())
        if common == 0:
            return 0.0
        precision = common / sum(pred_c.values())
        recall = common / sum(gold_c.values())
        return 2 * precision * recall / (precision + recall)
