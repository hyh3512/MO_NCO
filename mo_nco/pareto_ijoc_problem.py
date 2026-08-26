from __future__ import annotations

"""Problem interfaces used to test IJOC-level algorithmic generality."""

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
import hashlib
import json
import math
import random
from typing import Protocol, Sequence, Tuple, runtime_checkable

from .instance import MultiObjectiveTSPInstance, instance_sha256
from .moves import random_tour, sample_two_opt_indices, two_opt_at
from .pareto_smc_spec import analytic_objective_box
from .types import ObjectiveVector

Solution = Tuple[int, ...]


@runtime_checkable
class MultiObjectiveCombinatorialProblem(Protocol):
    name: str

    @property
    def num_objectives(self) -> int: ...

    @property
    def solution_size(self) -> int: ...

    @property
    def objective_lower_bounds(self) -> ObjectiveVector: ...

    @property
    def objective_upper_bounds(self) -> ObjectiveVector: ...

    @property
    def symmetric_proposal_contract(self) -> str: ...

    def random_solution(self, rng: random.Random) -> Solution: ...

    def propose(self, solution: Solution, rng: random.Random) -> Solution: ...

    def proposal_probability(self, source: Solution, target: Solution) -> float: ...

    def evaluate(self, solution: Solution) -> ObjectiveVector: ...

    def validate_solution(self, solution: Solution) -> None: ...

    def canonical_payload(self) -> object: ...


def problem_sha256(problem: MultiObjectiveCombinatorialProblem) -> str:
    payload = problem.canonical_payload()
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MultiObjectiveTSPProblemAdapter:
    """Problem-generic wrapper around the fixed-origin MOTSP state space."""

    instance: MultiObjectiveTSPInstance

    @property
    def name(self) -> str:
        return self.instance.name

    @property
    def num_objectives(self) -> int:
        return self.instance.num_objectives

    @property
    def solution_size(self) -> int:
        return self.instance.num_cities

    @property
    def objective_lower_bounds(self) -> ObjectiveVector:
        lower, _ = analytic_objective_box(self.instance)
        return lower

    @property
    def objective_upper_bounds(self) -> ObjectiveVector:
        _, upper = analytic_objective_box(self.instance)
        return upper

    @property
    def symmetric_proposal_contract(self) -> str:
        return "uniform_fixed_origin_two_opt_involution_v1"

    def random_solution(self, rng: random.Random) -> Solution:
        return random_tour(self.instance.num_cities, rng)

    def propose(self, solution: Solution, rng: random.Random) -> Solution:
        self.validate_solution(solution)
        i, j = sample_two_opt_indices(self.instance.num_cities, rng)
        return two_opt_at(solution, i, j)

    def proposal_probability(self, source: Solution, target: Solution) -> float:
        self.validate_solution(source)
        self.validate_solution(target)
        differences = [
            index for index, (left, right) in enumerate(zip(source, target))
            if left != right
        ]
        if not differences:
            return 0.0
        first, last = differences[0], differences[-1]
        if first <= 0 or last <= first:
            return 0.0
        candidate = list(source)
        candidate[first : last + 1] = reversed(candidate[first : last + 1])
        if tuple(candidate) != target:
            return 0.0
        return 1.0 / math.comb(self.instance.num_cities - 1, 2)

    def evaluate(self, solution: Solution) -> ObjectiveVector:
        return self.instance.evaluate(solution)

    def validate_solution(self, solution: Solution) -> None:
        self.instance.validate_tour(solution)

    def canonical_payload(self) -> object:
        return {
            "family": "fixed_origin_multiobjective_tsp_v1",
            "instance_sha256": instance_sha256(self.instance),
            "name": self.instance.name,
        }


@dataclass(frozen=True)
class MultiObjectiveKnapsackInstance:
    """Finite multiobjective 0--1 knapsack in minimization convention.

    Profits are maximized in the usual formulation.  The public objective
    vector is the negative profit vector, so every objective is minimized.
    The one-bit feasible-toggle proposal is symmetric on the feasible state
    graph: whenever two distinct feasible solutions differ in one bit, both
    proposal probabilities equal ``1 / n``; infeasible additions become a
    self-loop.
    """

    item_weights: Tuple[int, ...]
    profits_by_objective: Tuple[Tuple[int, ...], ...]
    capacity: int
    name: str = "multiobjective_knapsack"

    def __post_init__(self) -> None:
        if not self.item_weights:
            raise ValueError("At least one knapsack item is required.")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.item_weights):
            raise ValueError("Item weights must be positive integers.")
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int) or self.capacity <= 0:
            raise ValueError("capacity must be a positive integer.")
        if not self.profits_by_objective:
            raise ValueError("At least one profit objective is required.")
        n = len(self.item_weights)
        for profits in self.profits_by_objective:
            if len(profits) != n:
                raise ValueError("Every profit objective must contain one value per item.")
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in profits):
                raise ValueError("Profits must be nonnegative integers.")

    @property
    def num_objectives(self) -> int:
        return len(self.profits_by_objective)

    @property
    def solution_size(self) -> int:
        return len(self.item_weights)

    @property
    def objective_lower_bounds(self) -> ObjectiveVector:
        return tuple(-float(sum(profits)) for profits in self.profits_by_objective)

    @property
    def objective_upper_bounds(self) -> ObjectiveVector:
        return tuple(0.0 for _ in self.profits_by_objective)

    @property
    def symmetric_proposal_contract(self) -> str:
        return "uniform_one_bit_feasible_toggle_with_infeasible_self_loop_v1"

    def total_weight(self, solution: Solution) -> int:
        return sum(weight for selected, weight in zip(solution, self.item_weights) if selected)

    def validate_solution(self, solution: Solution) -> None:
        if len(solution) != self.solution_size:
            raise ValueError("Knapsack solution has the wrong length.")
        if any(value not in (0, 1) for value in solution):
            raise ValueError("Knapsack solutions must be binary.")
        if self.total_weight(solution) > self.capacity:
            raise ValueError("Knapsack solution violates the capacity constraint.")

    def evaluate(self, solution: Solution) -> ObjectiveVector:
        self.validate_solution(solution)
        return tuple(
            -float(sum(value * profit for value, profit in zip(solution, profits)))
            for profits in self.profits_by_objective
        )

    @cached_property
    def _uniform_feasible_suffix_counts(self) -> Tuple[Tuple[int, ...], ...]:
        """Count feasible suffix completions for exact uniform initialization."""

        capacity = self.capacity
        rows = [[1] * (capacity + 1)]
        for weight in reversed(self.item_weights):
            next_row = rows[-1]
            row = [
                next_row[remaining]
                + (
                    next_row[remaining - weight]
                    if weight <= remaining
                    else 0
                )
                for remaining in range(capacity + 1)
            ]
            rows.append(row)
        rows.reverse()
        return tuple(tuple(row) for row in rows)

    @property
    def feasible_solution_count(self) -> int:
        """Exact number of feasible binary solutions."""

        return self._uniform_feasible_suffix_counts[0][self.capacity]

    def uniform_solution_probability(self, solution: Solution) -> Fraction:
        """Exact initializer mass of a feasible solution."""

        self.validate_solution(solution)
        return Fraction(1, self.feasible_solution_count)

    def random_solution(self, rng: random.Random) -> Solution:
        """Sample exactly uniformly from all feasible binary solutions.

        The suffix-count recursion avoids the nonuniform random-greedy
        initializer that would break the counting-measure beta-zero target.
        """

        counts = self._uniform_feasible_suffix_counts
        remaining = self.capacity
        solution = []
        for item, weight in enumerate(self.item_weights):
            exclude_count = counts[item + 1][remaining]
            include_count = (
                counts[item + 1][remaining - weight]
                if weight <= remaining
                else 0
            )
            total = exclude_count + include_count
            include = include_count > 0 and rng.randrange(total) < include_count
            solution.append(int(include))
            if include:
                remaining -= weight
        sampled = tuple(solution)
        self.validate_solution(sampled)
        return sampled

    def propose(self, solution: Solution, rng: random.Random) -> Solution:
        self.validate_solution(solution)
        item = rng.randrange(self.solution_size)
        candidate = list(solution)
        candidate[item] = 1 - candidate[item]
        proposed = tuple(candidate)
        if self.total_weight(proposed) > self.capacity:
            return solution
        return proposed

    def proposal_probability(self, source: Solution, target: Solution) -> float:
        """Exact O(n) proposal probability under uniform one-bit selection."""
        self.validate_solution(source)
        self.validate_solution(target)
        differences = [
            index
            for index, (left, right) in enumerate(zip(source, target))
            if left != right
        ]
        if len(differences) == 1:
            return 1.0 / self.solution_size
        if differences:
            return 0.0
        current_weight = self.total_weight(source)
        infeasible_additions = sum(
            1
            for selected, weight in zip(source, self.item_weights)
            if selected == 0 and current_weight + weight > self.capacity
        )
        return infeasible_additions / self.solution_size

    def canonical_payload(self) -> object:
        return {
            "family": "multiobjective_0_1_knapsack_v1",
            "name": self.name,
            "item_weights": self.item_weights,
            "profits_by_objective": self.profits_by_objective,
            "capacity": self.capacity,
        }

    @staticmethod
    def random_instance(
        num_items: int,
        *,
        num_objectives: int = 2,
        seed: int = 0,
    ) -> "MultiObjectiveKnapsackInstance":
        if num_items <= 0 or num_objectives <= 0:
            raise ValueError("num_items and num_objectives must be positive.")
        rng = random.Random(seed)
        weights = tuple(rng.randint(1, 30) for _ in range(num_items))
        profits = tuple(
            tuple(rng.randint(1, 50) for _ in range(num_items))
            for _ in range(num_objectives)
        )
        capacity = max(1, int(0.35 * sum(weights)))
        return MultiObjectiveKnapsackInstance(
            item_weights=weights,
            profits_by_objective=profits,
            capacity=capacity,
            name=f"mokp_n{num_items}_m{num_objectives}_s{seed}",
        )
