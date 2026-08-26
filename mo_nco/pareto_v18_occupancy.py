"""Exact categorical occupancy certificates and allocation for Pareto-SMC v18.

For type ``r`` let ``h[r,j]`` be simultaneous lower bounds on the probabilities
of mutually disjoint frozen cells.  The row sum must not exceed one.  The true
categorical endpoint law can be coupled with the lower law

    P(Y=j)=h[r,j],  P(Y=outside)=1-sum_j h[r,j]

so that every certified hit of ``Y`` is also a hit of the true endpoint.  Hence
an exact inclusion--exclusion calculation under the lower law is a rigorous
lower bound on simultaneous coverage under the true law.

The exact planner minimizes a positive rational cost over the integer lattice.
It uses Dijkstra's algorithm and therefore certifies global optimality when the
search completes.  Exhausting the node cap is fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import heapq
from functools import lru_cache
from typing import Sequence

from .pareto_v17_regeneration import as_fraction
from .pareto_v17_multitype_confirm import MultiTypeConfirmProblem, greedy_feasible_allocation


class OccupancyCertificateError(ValueError):
    pass


def _validate_matrix(
    raw: Sequence[Sequence[Fraction | int | str]],
) -> tuple[tuple[Fraction, ...], ...]:
    matrix = tuple(tuple(as_fraction(value) for value in row) for row in raw)
    if not matrix or not matrix[0]:
        raise OccupancyCertificateError("endpoint lower matrix must be nonempty")
    if any(len(row) != len(matrix[0]) for row in matrix):
        raise OccupancyCertificateError("endpoint lower matrix is ragged")
    for r, row in enumerate(matrix):
        if any(value < 0 or value > 1 for value in row):
            raise OccupancyCertificateError("cell lower probabilities must lie in [0,1]")
        if sum(row, Fraction(0, 1)) > 1:
            raise OccupancyCertificateError(
                f"type row {r} is not a dominated categorical lower vector"
            )
    for j in range(len(matrix[0])):
        if all(matrix[r][j] == 0 for r in range(len(matrix))):
            raise OccupancyCertificateError(f"cell {j} has no positive hit lower bound")
    return matrix


def exact_all_cells_hit_lower(
    endpoint_lower: Sequence[Sequence[Fraction | int | str]],
    counts: Sequence[int],
    *,
    max_cells: int = 14,
) -> Fraction:
    """Exact inclusion--exclusion lower bound for simultaneous cell coverage."""

    matrix = _validate_matrix(endpoint_lower)
    r_count = len(matrix)
    j_count = len(matrix[0])
    if len(counts) != r_count:
        raise OccupancyCertificateError("allocation dimension mismatch")
    if any(not isinstance(value, int) or value < 0 for value in counts):
        raise OccupancyCertificateError("allocation counts must be nonnegative integers")
    if j_count > max_cells:
        raise OccupancyCertificateError(
            f"exact occupancy has 2^{j_count} subsets, exceeding max_cells={max_cells}"
        )

    subset_base: list[tuple[Fraction, ...]] = []
    parity: list[int] = []
    for mask in range(1 << j_count):
        values: list[Fraction] = []
        for r in range(r_count):
            subset_mass = sum(
                (matrix[r][j] for j in range(j_count) if mask & (1 << j)),
                Fraction(0, 1),
            )
            values.append(Fraction(1, 1) - subset_mass)
        subset_base.append(tuple(values))
        parity.append(-1 if mask.bit_count() % 2 else 1)

    total = Fraction(0, 1)
    for sign, bases in zip(parity, subset_base, strict=True):
        term = Fraction(1, 1)
        for r, count in enumerate(counts):
            term *= bases[r] ** count
        total += sign * term
    if total < 0 or total > 1:
        raise AssertionError("exact occupancy lower bound escaped [0,1]")
    return total


def exact_simultaneous_miss_upper(
    endpoint_lower: Sequence[Sequence[Fraction | int | str]],
    counts: Sequence[int],
    *,
    max_cells: int = 14,
) -> Fraction:
    return Fraction(1, 1) - exact_all_cells_hit_lower(
        endpoint_lower,
        counts,
        max_cells=max_cells,
    )


def union_miss_upper(
    endpoint_lower: Sequence[Sequence[Fraction | int | str]],
    counts: Sequence[int],
) -> Fraction:
    matrix = _validate_matrix(endpoint_lower)
    if len(counts) != len(matrix):
        raise OccupancyCertificateError("allocation dimension mismatch")
    total = Fraction(0, 1)
    for j in range(len(matrix[0])):
        miss = Fraction(1, 1)
        for r, count in enumerate(counts):
            if not isinstance(count, int) or count < 0:
                raise OccupancyCertificateError("allocation counts must be nonnegative integers")
            miss *= (Fraction(1, 1) - matrix[r][j]) ** count
        total += miss
    return min(Fraction(1, 1), total)


@dataclass(frozen=True)
class MultiTypeOccupancyProblem:
    endpoint_lower: tuple[tuple[Fraction, ...], ...]
    costs: tuple[Fraction, ...]
    delta: Fraction
    max_cells_exact: int = 14
    max_subset_terms: int = 1 << 14
    _subset_bases: tuple[tuple[Fraction, ...], ...] = field(init=False, repr=False, compare=False)
    _subset_signs: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        matrix = _validate_matrix(self.endpoint_lower)
        costs = tuple(as_fraction(value) for value in self.costs)
        delta = as_fraction(self.delta)
        object.__setattr__(self, "endpoint_lower", matrix)
        object.__setattr__(self, "costs", costs)
        object.__setattr__(self, "delta", delta)
        if len(costs) != len(matrix) or any(value <= 0 for value in costs):
            raise OccupancyCertificateError("one positive rational cost is required per type")
        if not (Fraction(0, 1) < delta < Fraction(1, 1)):
            raise OccupancyCertificateError("delta must lie in (0,1)")
        if len(matrix[0]) > self.max_cells_exact:
            raise OccupancyCertificateError("exact occupancy cell cap exceeded")
        if (1 << len(matrix[0])) > self.max_subset_terms:
            raise OccupancyCertificateError(
                "exact occupancy subset-term budget exceeded"
            )
        bases: list[tuple[Fraction, ...]] = []
        signs: list[int] = []
        for mask in range(1 << len(matrix[0])):
            bases.append(
                tuple(
                    Fraction(1, 1)
                    - sum(
                        (matrix[r][j] for j in range(len(matrix[0])) if mask & (1 << j)),
                        Fraction(0, 1),
                    )
                    for r in range(len(matrix))
                )
            )
            signs.append(-1 if mask.bit_count() % 2 else 1)
        object.__setattr__(self, "_subset_bases", tuple(bases))
        object.__setattr__(self, "_subset_signs", tuple(signs))

    @property
    def num_types(self) -> int:
        return len(self.endpoint_lower)

    @property
    def num_cells(self) -> int:
        return len(self.endpoint_lower[0])

    def validate_counts(self, counts: Sequence[int]) -> tuple[int, ...]:
        result = tuple(counts)
        if len(result) != self.num_types or any(
            not isinstance(value, int) or value < 0 for value in result
        ):
            raise OccupancyCertificateError("invalid allocation counts")
        return result

    def cost(self, counts: Sequence[int]) -> Fraction:
        state = self.validate_counts(counts)
        return sum(
            (self.costs[r] * state[r] for r in range(self.num_types)),
            Fraction(0, 1),
        )

    @lru_cache(maxsize=100_000)
    def risk_cached(self, state: tuple[int, ...]) -> Fraction:
        hit = Fraction(0, 1)
        for sign, bases in zip(self._subset_signs, self._subset_bases, strict=True):
            term = Fraction(1, 1)
            for r, count in enumerate(state):
                term *= bases[r] ** count
            hit += sign * term
        if hit < 0 or hit > 1:
            raise AssertionError("cached occupancy probability escaped [0,1]")
        return Fraction(1, 1) - hit

    def risk(self, counts: Sequence[int]) -> Fraction:
        return self.risk_cached(self.validate_counts(counts))

    def feasible(self, counts: Sequence[int]) -> bool:
        return self.risk(counts) <= self.delta

    def marginal_reduction(self, counts: Sequence[int], type_index: int) -> Fraction:
        state = self.validate_counts(counts)
        if type_index < 0 or type_index >= self.num_types:
            raise OccupancyCertificateError("type index out of range")
        child = list(state)
        child[type_index] += 1
        return self.risk(state) - self.risk(tuple(child))


@dataclass(frozen=True)
class OccupancyPlanCertificate:
    counts: tuple[int, ...]
    total_cost: Fraction
    exact_miss_upper: Fraction
    union_miss_upper: Fraction
    optimal: bool
    explored_nodes: int
    search_exhausted: bool
    incumbent_source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "risk_model": "dominated_categorical_exact_occupancy_v18",
            "counts": list(self.counts),
            "total_cost": str(self.total_cost),
            "exact_miss_upper": str(self.exact_miss_upper),
            "union_miss_upper": str(self.union_miss_upper),
            "optimal": self.optimal,
            "explored_nodes": self.explored_nodes,
            "search_exhausted": self.search_exhausted,
            "incumbent_source": self.incumbent_source,
        }


def _union_greedy_incumbent(problem: MultiTypeOccupancyProblem, max_steps: int) -> tuple[int, ...]:
    """Use the union-bound greedy design as a guaranteed feasible incumbent.

    The exact all-hit probability can have zero one-step marginal gain (for
    example, two cells and an empty allocation).  The union-bound objective has
    strictly positive progress whenever a cell has positive certified mass.
    Since exact miss risk is no larger than the union bound, every union-bound
    feasible design is an exact-occupancy feasible incumbent.
    """

    union_problem = MultiTypeConfirmProblem(
        endpoint_lower=problem.endpoint_lower,
        costs=problem.costs,
        delta=problem.delta,
    )
    return greedy_feasible_allocation(union_problem, max_steps=max_steps)


def exact_minimum_cost_occupancy_allocation(
    problem: MultiTypeOccupancyProblem,
    *,
    max_nodes: int = 2_000_000,
    max_greedy_steps: int = 1_000_000,
) -> OccupancyPlanCertificate:
    """Exact integer optimization of the dominated-categorical occupancy risk."""

    incumbent = _union_greedy_incumbent(problem, max_greedy_steps)
    incumbent_cost = problem.cost(incumbent)
    zero = tuple(0 for _ in range(problem.num_types))
    queue: list[tuple[Fraction, int, tuple[int, ...]]] = [(Fraction(0, 1), 0, zero)]
    seen = {zero}
    explored = 0

    while queue:
        cost, l1, state = heapq.heappop(queue)
        explored += 1
        if explored > max_nodes:
            return OccupancyPlanCertificate(
                counts=incumbent,
                total_cost=incumbent_cost,
                exact_miss_upper=problem.risk(incumbent),
                union_miss_upper=union_miss_upper(problem.endpoint_lower, incumbent),
                optimal=False,
                explored_nodes=explored,
                search_exhausted=True,
                incumbent_source="union_bound_greedy",
            )
        if cost > incumbent_cost:
            break
        if problem.feasible(state):
            return OccupancyPlanCertificate(
                counts=state,
                total_cost=cost,
                exact_miss_upper=problem.risk(state),
                union_miss_upper=union_miss_upper(problem.endpoint_lower, state),
                optimal=True,
                explored_nodes=explored,
                search_exhausted=False,
                incumbent_source="dijkstra_first_feasible",
            )
        for r in range(problem.num_types):
            child = list(state)
            child[r] += 1
            child_t = tuple(child)
            if child_t in seen:
                continue
            child_cost = cost + problem.costs[r]
            if child_cost > incumbent_cost:
                continue
            seen.add(child_t)
            heapq.heappush(queue, (child_cost, l1 + 1, child_t))

    return OccupancyPlanCertificate(
        counts=incumbent,
        total_cost=incumbent_cost,
        exact_miss_upper=problem.risk(incumbent),
        union_miss_upper=union_miss_upper(problem.endpoint_lower, incumbent),
        optimal=True,
        explored_nodes=explored,
        search_exhausted=False,
        incumbent_source="greedy_after_all_lower_cost_states_exhausted",
    )


__all__ = [
    "MultiTypeOccupancyProblem",
    "OccupancyCertificateError",
    "OccupancyPlanCertificate",
    "exact_all_cells_hit_lower",
    "exact_minimum_cost_occupancy_allocation",
    "exact_simultaneous_miss_upper",
    "union_miss_upper",
]
