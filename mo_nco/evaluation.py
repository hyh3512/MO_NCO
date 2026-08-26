from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .instance import MultiObjectiveTSPInstance
from .moves import two_opt_at
from .types import ObjectiveVector, Tour


class EvaluationLimitExceeded(RuntimeError):
    """Raised when an algorithm attempts to evaluate beyond its budget."""


@dataclass
class CountingTSPInstance:
    """Count objective evaluations while preserving the instance interface."""

    base: MultiObjectiveTSPInstance
    max_evaluations: Optional[int] = None
    evaluations: int = 0

    @property
    def name(self) -> str:
        return self.base.name

    @property
    def num_cities(self) -> int:
        return self.base.num_cities

    @property
    def num_objectives(self) -> int:
        return self.base.num_objectives

    @property
    def distance_matrices(self):  # type: ignore[no-untyped-def]
        return self.base.distance_matrices

    @property
    def symmetric_objectives(self):  # type: ignore[no-untyped-def]
        return self.base.symmetric_objectives

    @property
    def exact_two_opt_delta_objectives(self):  # type: ignore[no-untyped-def]
        return getattr(self.base, "exact_two_opt_delta_objectives", ())

    @property
    def exact_two_opt_delta_in_binary64(self) -> bool:
        return bool(
            getattr(
                self.base,
                "exact_two_opt_delta_in_binary64",
                False,
            )
        )

    @property
    def objective_scale_estimates(self) -> ObjectiveVector:
        return self.base.objective_scale_estimates

    def evaluate(self, tour: Tour) -> ObjectiveVector:
        self._charge_evaluation()
        return self.base.evaluate(tour)

    def evaluate_two_opt(
        self,
        tour: Tour,
        current_objectives: ObjectiveVector,
        i: int,
        j: int,
    ) -> ObjectiveVector:
        self._charge_evaluation()
        method = getattr(self.base, "evaluate_two_opt", None)
        if callable(method):
            return method(tour, current_objectives, i, j)
        return self.base.evaluate(two_opt_at(tour, i, j))

    def validate_tour(self, tour: Tour) -> None:
        self.base.validate_tour(tour)

    def remaining_evaluations(self) -> Optional[int]:
        if self.max_evaluations is None:
            return None
        return max(0, self.max_evaluations - self.evaluations)

    def can_evaluate(self, count: int = 1) -> bool:
        remaining = self.remaining_evaluations()
        return remaining is None or remaining >= count

    def charge_evaluations(self, count: int) -> None:
        """Atomically charge a compiled/batched evaluation block."""
        if count < 0:
            raise ValueError("evaluation count must be nonnegative.")
        if self.max_evaluations is not None and self.evaluations + count > self.max_evaluations:
            raise EvaluationLimitExceeded("Evaluation budget exhausted.")
        self.evaluations += count

    def _charge_evaluation(self) -> None:
        self.charge_evaluations(1)


def evaluation_count(instance: object) -> int:
    return int(getattr(instance, "evaluations", 0))


def can_evaluate(instance: object, count: int = 1) -> bool:
    method = getattr(instance, "can_evaluate", None)
    if callable(method):
        return bool(method(count))
    return True


def remaining_evaluations(instance: object) -> Optional[int]:
    method = getattr(instance, "remaining_evaluations", None)
    if callable(method):
        return method()
    return None
