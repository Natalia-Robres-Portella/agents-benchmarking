"""HotPotQA task loader — HuggingFace Hub with offline fixture fallback."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.schema import TaskInstance
from src.tasks.base import TaskLoader

_FIXTURE_PATH = (
    Path(__file__).parent.parent.parent.parent / "fixtures" / "hotpotqa_sample.json"
)


class HotPotQALoader(TaskLoader):
    """
    Loads HotPotQA tasks.  Falls back to the bundled fixture when the
    HuggingFace Hub is unreachable (CI, offline experiments).

    The same seed produces the same task list every time — required for
    reproducibility across agent comparisons.
    """

    @property
    def name(self) -> str:
        return "hotpotqa"

    def load(
        self,
        split: str = "validation",
        n_samples: int = 10,
        seed: int = 42,
        filter_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[TaskInstance]:
        items = self._load_raw(split)

        if filter_kwargs and filter_kwargs.get("difficulty"):
            allowed = set(filter_kwargs["difficulty"])
            items = [i for i in items if i.get("level") in allowed]

        rng = random.Random(seed)
        rng.shuffle(items)
        return [self._to_instance(item) for item in items[:n_samples]]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_raw(self, split: str) -> List[Dict[str, Any]]:
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]

            ds = load_dataset(
                "hotpot_qa", "fullwiki", split=split, trust_remote_code=True
            )
            return list(ds)
        except Exception:
            return self._load_fixture()

    @staticmethod
    def _load_fixture() -> List[Dict[str, Any]]:
        if not _FIXTURE_PATH.exists():
            raise FileNotFoundError(
                f"HotPotQA fixture missing at {_FIXTURE_PATH}. "
                "Commit fixtures/hotpotqa_sample.json to the repo."
            )
        with open(_FIXTURE_PATH) as fh:
            return json.load(fh)

    @staticmethod
    def _to_instance(item: Dict[str, Any]) -> TaskInstance:
        return TaskInstance(
            task_id=f"hotpotqa_{item['id']}",
            input=item["question"],
            gold=item["answer"],
            metadata={
                "type": item.get("type", "bridge"),
                "level": item.get("level", "medium"),
            },
        )
