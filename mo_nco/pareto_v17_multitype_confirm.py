"""Exact multi-type confirm allocation for Pareto-SMC v17.

The certified optimization problem is

    minimize   sum_r c_r m_r
    subject to sum_j prod_r (1-h[r,j])**m_r <= delta,
               m_r in N_0.

All certified comparisons are exact rational arithmetic.  A Dijkstra search in
integer allocation space returns the first feasible cost level and therefore an
exact minimum-cost allocation.  A node cap is fail-closed: exhausting the cap
never emits an optimality certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import heapq
import math
from typing import Iterable, Sequence

from .pareto_v17_regeneration import RegenerationCertificateError, as_fraction


class ConfirmPlannerError(RegenerationCertificateError):
    pass


@dataclass(frozen=True)
class MultiTypeConfirmProblem:
    endpoint_lower: tuple[tuple[Fraction, ...], ...]  # type x cell
    costs: tuple[Fraction, ...]
    delta: Fraction

    def __post_init__(self) -> None:
        matrix = tuple(tuple(as_fraction(x) for x in row) for row in self.endpoint_lower)
        costs = tuple(as_fraction(x) for x in self.costs)
        delta = as_fraction(self.delta)
        object.__setattr__(self, "endpoint_lower", matrix)
        object.__setattr__(self, "costs", costs)
        object.__setattr__(self, "delta", delta)
        if not matrix or not matrix[0]:
            raise ConfirmPlannerError("endpoint lower matrix must be nonempty")
        if any(len(row) != len(matrix[0]) for row in matrix):
            raise ConfirmPlannerError("endpoint lower matrix is ragged")
        if len(costs) != len(matrix):
            raise ConfirmPlannerError("one positive cost is required per type")
        if any(x <= 0 for x in costs):
            raise ConfirmPlannerError("all type costs must be strictly positive")
        if delta <= 0 or delta >= 1:
            raise ConfirmPlannerError("delta must lie strictly between zero and one")
        if any(x < 0 or x > 1 for row in matrix for x in row):
            raise ConfirmPlannerError("endpoint probability lower bounds must lie in [0,1]")
        for j in range(len(matrix[0])):
            if all(matrix[r][j] == 0 for r in range(len(matrix))):
                raise ConfirmPlannerError(f"cell {j} has no positive certified hit probability")

    @property
    def num_types(self) -> int:
        return len(self.endpoint_lower)

    @property
    def num_cells(self) -> int:
        return len(self.endpoint_lower[0])

    def cost(self, counts: Sequence[int]) -> Fraction:
        self._validate_counts(counts)
        return sum((self.costs[r] * counts[r] for r in range(self.num_types)), Fraction(0, 1))

    def cell_miss(self, counts: Sequence[int], cell: int) -> Fraction:
        self._validate_counts(counts)
        if cell < 0 or cell >= self.num_cells:
            raise ConfirmPlannerError("cell index out of range")
        out = Fraction(1, 1)
        for r in range(self.num_types):
            out *= (Fraction(1, 1) - self.endpoint_lower[r][cell]) ** counts[r]
        return out

    def raw_union_risk(self, counts: Sequence[int]) -> Fraction:
        self._validate_counts(counts)
        return sum((self.cell_miss(counts, j) for j in range(self.num_cells)), Fraction(0, 1))

    def risk(self, counts: Sequence[int]) -> Fraction:
        """Probability-scale union bound, capped at one for reporting only."""
        return min(Fraction(1, 1), self.raw_union_risk(counts))

    def feasible(self, counts: Sequence[int]) -> bool:
        # delta<1, so using the uncapped sum is equivalent for feasibility and
        # preserves strict marginal progress while the union sum exceeds one.
        return self.raw_union_risk(counts) <= self.delta

    def marginal_reduction(self, counts: Sequence[int], type_index: int) -> Fraction:
        self._validate_counts(counts)
        if type_index < 0 or type_index >= self.num_types:
            raise ConfirmPlannerError("type index out of range")
        before = self.raw_union_risk(counts)
        next_counts = list(counts)
        next_counts[type_index] += 1
        return before - self.raw_union_risk(next_counts)

    def _validate_counts(self, counts: Sequence[int]) -> None:
        if len(counts) != self.num_types:
            raise ConfirmPlannerError("allocation dimension mismatch")
        if any(not isinstance(x, int) or x < 0 for x in counts):
            raise ConfirmPlannerError("allocation counts must be nonnegative integers")


@dataclass(frozen=True)
class ConfirmPlanCertificate:
    counts: tuple[int, ...]
    total_cost: Fraction
    exact_union_risk: Fraction
    optimal: bool
    explored_nodes: int
    incumbent_source: str
    search_exhausted: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "counts": list(self.counts),
            "total_cost": str(self.total_cost),
            "exact_union_risk": str(self.exact_union_risk),
            "optimal": self.optimal,
            "explored_nodes": self.explored_nodes,
            "incumbent_source": self.incumbent_source,
            "search_exhausted": self.search_exhausted,
        }


def greedy_feasible_allocation(
    problem: MultiTypeConfirmProblem,
    *,
    max_steps: int = 1_000_000,
) -> tuple[int, ...]:
    """Produce a deterministic feasible incumbent.

    At each step choose the largest exact risk reduction per unit cost.  This is
    an incumbent heuristic only; no optimality claim is attached to it.
    """

    counts = [0] * problem.num_types
    for _ in range(max_steps + 1):
        if problem.feasible(counts):
            return tuple(counts)
        scores: list[tuple[Fraction, int]] = []
        for r in range(problem.num_types):
            reduction = problem.marginal_reduction(counts, r)
            scores.append((reduction / problem.costs[r], r))
        best_score = max(score for score, _ in scores)
        if best_score <= 0:
            raise ConfirmPlannerError("risk cannot be reduced under the supplied probability matrix")
        chosen = min(r for score, r in scores if score == best_score)
        counts[chosen] += 1
    raise ConfirmPlannerError("greedy incumbent exceeded max_steps")


def exact_minimum_cost_allocation(
    problem: MultiTypeConfirmProblem,
    *,
    max_nodes: int = 2_000_000,
) -> ConfirmPlanCertificate:
    """Return an exact minimum-cost allocation or fail closed.

    The state graph has vertices ``m in N_0^R`` and edges ``m -> m+e_r`` with
    positive cost ``c_r``.  Dijkstra's algorithm pops states in nondecreasing
    total cost.  Therefore the first feasible state popped has globally minimum
    cost.  The greedy incumbent supplies a finite search ceiling.
    """

    incumbent = greedy_feasible_allocation(problem)
    incumbent_cost = problem.cost(incumbent)
    zero = tuple(0 for _ in range(problem.num_types))
    heap: list[tuple[Fraction, int, tuple[int, ...]]] = [(Fraction(0, 1), 0, zero)]
    seen: set[tuple[int, ...]] = {zero}
    explored = 0

    while heap:
        cost, l1_count, state = heapq.heappop(heap)
        explored += 1
        if explored > max_nodes:
            return ConfirmPlanCertificate(
                counts=incumbent,
                total_cost=incumbent_cost,
                exact_union_risk=problem.risk(incumbent),
                optimal=False,
                explored_nodes=explored,
                incumbent_source="greedy",
                search_exhausted=True,
            )
        if cost > incumbent_cost:
            break
        if problem.feasible(state):
            return ConfirmPlanCertificate(
                counts=state,
                total_cost=cost,
                exact_union_risk=problem.risk(state),
                optimal=True,
                explored_nodes=explored,
                incumbent_source="dijkstra_first_feasible",
                search_exhausted=False,
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
            heapq.heappush(heap, (child_cost, l1_count + 1, child_t))

    # The greedy incumbent itself is feasible.  Reaching this branch means all
    # strictly cheaper states were exhausted and no feasible state was found;
    # hence the incumbent cost is optimal.  We still search same-cost states only
    # if they are reached before the queue exceeds the incumbent ceiling.
    return ConfirmPlanCertificate(
        counts=incumbent,
        total_cost=incumbent_cost,
        exact_union_risk=problem.risk(incumbent),
        optimal=True,
        explored_nodes=explored,
        incumbent_source="greedy_after_lower_cost_exhaustion",
        search_exhausted=False,
    )


def continuous_union_risk(
    endpoint_lower: Sequence[Sequence[float]],
    allocation: Sequence[float],
) -> float:
    """Evaluate the convex continuous relaxation in log-safe arithmetic."""

    if not endpoint_lower or len(endpoint_lower) != len(allocation):
        raise ConfirmPlannerError("continuous relaxation dimension mismatch")
    j_count = len(endpoint_lower[0])
    if any(len(row) != j_count for row in endpoint_lower):
        raise ConfirmPlannerError("ragged endpoint lower matrix")
    if any(x < 0 for x in allocation):
        raise ConfirmPlannerError("continuous allocation must be nonnegative")
    total = 0.0
    for j in range(j_count):
        exponent = 0.0
        impossible = False
        for r, x in enumerate(allocation):
            h = float(endpoint_lower[r][j])
            if h < 0.0 or h > 1.0:
                raise ConfirmPlannerError("probability outside [0,1]")
            if h == 1.0 and x > 0.0:
                impossible = True
                break
            if h > 0.0:
                exponent += x * math.log1p(-h)
        total += 0.0 if impossible else math.exp(exponent)
    return min(1.0, total)


def ceiling_rounding_certificate(
    problem: MultiTypeConfirmProblem,
    relaxation_point: Sequence[float],
) -> dict[str, object]:
    """Verify the ceiling-rounding theorem for a supplied feasible relaxation.

    The function does not claim that the supplied point is the continuous
    optimum.  It checks feasibility numerically, rounds upward, and returns the
    exact rational risk/cost of the integer point.  The mathematical theorem
    supplies the additive ``sum(costs)`` gap when the input is a true continuous
    optimum.
    """

    if len(relaxation_point) != problem.num_types or any(x < 0 for x in relaxation_point):
        raise ConfirmPlannerError("invalid relaxation point")
    float_matrix = [[float(x) for x in row] for row in problem.endpoint_lower]
    relaxed_risk = continuous_union_risk(float_matrix, relaxation_point)
    rounded = tuple(math.ceil(x) for x in relaxation_point)
    return {
        "relaxation_risk_float": relaxed_risk,
        "relaxation_feasible_float": relaxed_risk <= float(problem.delta),
        "rounded_counts": list(rounded),
        "rounded_exact_risk": str(problem.risk(rounded)),
        "rounded_exact_cost": str(problem.cost(rounded)),
        "rounded_exact_feasible": problem.feasible(rounded),
        "additive_cost_gap_upper": str(sum(problem.costs, Fraction(0, 1))),
    }
