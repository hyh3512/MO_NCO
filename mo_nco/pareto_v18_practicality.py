"""Instance-dependent practicality and information-obstruction gates for v18.

No concentration theorem can make a cell with endpoint probability ``q`` cheap:
any sequence of independent or predictably chosen endpoint laws whose hit
probability is at most ``q`` needs at least

    min{m : (1-q)^m <= delta}

endpoints to make that cell's miss probability at most ``delta``.  This module
computes the exact integer obstruction with rational arithmetic and compares it
with the certified upper design cost.  The result is an instance-dependent
feasibility statement, not a universal scalability theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from .pareto_v17_regeneration import as_fraction


class PracticalityCertificateError(ValueError):
    pass


def minimum_independent_endpoints_for_cell(
    hit_probability_upper: Fraction | int | str,
    miss_probability_target: Fraction | int | str,
) -> int:
    """Exact smallest ``m`` with ``(1-q)^m <= delta``.

    ``q`` is an upper bound on every endpoint's conditional hit probability.
    The lower-bound theorem remains valid under predictable adaptive type
    selection because each conditional miss probability is at least ``1-q``.
    """

    q = as_fraction(hit_probability_upper)
    delta = as_fraction(miss_probability_target)
    if q < 0 or q > 1:
        raise PracticalityCertificateError("hit probability upper bound must lie in [0,1]")
    if not (Fraction(0, 1) < delta < Fraction(1, 1)):
        raise PracticalityCertificateError("miss target must lie in (0,1)")
    if q == 0:
        raise PracticalityCertificateError("a zero hit upper bound makes the target impossible")
    if q == 1:
        return 1
    base = Fraction(1, 1) - q
    # Exponential search followed by exact binary search avoids floating logs.
    lo = 0
    hi = 1
    while base**hi > delta:
        lo = hi
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if base**mid <= delta:
            hi = mid
        else:
            lo = mid
    return hi


def binomial_upper_tail(n: int, threshold: int, q: Fraction) -> Fraction:
    """Exact ``P[Bin(n,q) >= threshold]``."""

    if n < 0 or threshold < 0 or q < 0 or q > 1:
        raise PracticalityCertificateError("invalid binomial-tail arguments")
    if threshold <= 0:
        return Fraction(1, 1)
    if threshold > n:
        return Fraction(0, 1)
    from math import comb

    return sum(
        (
            Fraction(comb(n, k), 1) * q**k * (Fraction(1, 1) - q) ** (n - k)
            for k in range(threshold, n + 1)
        ),
        Fraction(0, 1),
    )


def minimum_endpoints_for_subset_hits(
    subset_size: int,
    per_endpoint_subset_hit_upper: Fraction | int | str,
    failure_target: Fraction | int | str,
) -> int:
    """Necessary endpoint count to see every cell of a subset.

    Covering ``s`` distinct cells requires at least ``s`` endpoint outcomes in
    their union.  If every predictably chosen endpoint has conditional union-hit
    probability at most ``q``, the hit count is stochastically dominated by
    ``Bin(n,q)``.  Therefore success probability cannot exceed the binomial tail.
    """

    if not isinstance(subset_size, int) or subset_size <= 0:
        raise PracticalityCertificateError("subset_size must be positive")
    q = as_fraction(per_endpoint_subset_hit_upper)
    delta = as_fraction(failure_target)
    if q < 0 or q > 1 or not (Fraction(0, 1) < delta < Fraction(1, 1)):
        raise PracticalityCertificateError("invalid subset probability or failure target")
    if q == 0:
        raise PracticalityCertificateError("a zero union-hit upper bound makes subset coverage impossible")
    target = Fraction(1, 1) - delta
    lo = subset_size - 1
    hi = max(1, subset_size)
    while binomial_upper_tail(hi, subset_size, q) < target:
        lo = hi
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if binomial_upper_tail(mid, subset_size, q) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def subset_coverage_endpoint_lower(
    endpoint_probability_upper: Sequence[Sequence[Fraction | int | str]],
    failure_target: Fraction | int | str,
    *,
    max_cells_exact: int = 14,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Maximize the exact subset-count obstruction over all nonempty cell sets."""

    matrix = tuple(tuple(as_fraction(value) for value in row) for row in endpoint_probability_upper)
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise PracticalityCertificateError("invalid endpoint probability upper matrix")
    j_count = len(matrix[0])
    if j_count > max_cells_exact:
        raise PracticalityCertificateError("subset practicality cell cap exceeded")
    best = 0
    witnesses: list[tuple[int, int]] = []
    for mask in range(1, 1 << j_count):
        size = mask.bit_count()
        q = max(
            sum((matrix[r][j] for j in range(j_count) if mask & (1 << j)), Fraction(0, 1))
            for r in range(len(matrix))
        )
        q = min(Fraction(1, 1), q)
        required = minimum_endpoints_for_subset_hits(size, q, failure_target)
        if required > best:
            best = required
            witnesses = [(mask, required)]
        elif required == best:
            witnesses.append((mask, required))
    return best, tuple(witnesses)


@dataclass(frozen=True)
class PracticalityCertificate:
    endpoint_upper_by_type_cell: tuple[tuple[Fraction, ...], ...]
    per_cell_endpoint_lower: tuple[int, ...]
    universal_endpoint_lower: int
    subset_endpoint_lower: int
    combined_endpoint_lower: int
    subset_lower_witnesses: tuple[tuple[int, int], ...]
    minimum_endpoint_cost: Fraction
    certified_upper_endpoint_count: int
    certified_upper_cost: Fraction
    upper_to_lower_cost_ratio: Fraction | None
    evaluation_budget: Fraction
    budget_feasible: bool
    lower_bound_nontrivial: bool
    verdict: str

    def to_dict(self) -> dict[str, object]:
        return {
            "semantics": "instance_dependent_endpoint_information_obstruction_v18",
            "endpoint_upper_by_type_cell": [
                [str(value) for value in row]
                for row in self.endpoint_upper_by_type_cell
            ],
            "per_cell_endpoint_lower": list(self.per_cell_endpoint_lower),
            "universal_endpoint_lower": self.universal_endpoint_lower,
            "subset_endpoint_lower": self.subset_endpoint_lower,
            "combined_endpoint_lower": self.combined_endpoint_lower,
            "subset_lower_witnesses": [list(item) for item in self.subset_lower_witnesses],
            "minimum_endpoint_cost": str(self.minimum_endpoint_cost),
            "certified_upper_endpoint_count": self.certified_upper_endpoint_count,
            "certified_upper_cost": str(self.certified_upper_cost),
            "upper_to_lower_cost_ratio": (
                None if self.upper_to_lower_cost_ratio is None else str(self.upper_to_lower_cost_ratio)
            ),
            "evaluation_budget": str(self.evaluation_budget),
            "budget_feasible": self.budget_feasible,
            "lower_bound_nontrivial": self.lower_bound_nontrivial,
            "verdict": self.verdict,
        }


def build_practicality_certificate(
    endpoint_probability_upper: Sequence[Sequence[Fraction | int | str]],
    costs: Sequence[Fraction | int | str],
    certified_counts: Sequence[int],
    *,
    simultaneous_failure_target: Fraction | int | str,
    evaluation_budget: Fraction | int | str,
    max_cells_exact: int = 14,
) -> PracticalityCertificate:
    """Compare an exact information lower bound with a certified upper plan."""

    matrix = tuple(tuple(as_fraction(value) for value in row) for row in endpoint_probability_upper)
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise PracticalityCertificateError("invalid endpoint probability upper matrix")
    if any(value < 0 or value > 1 for row in matrix for value in row):
        raise PracticalityCertificateError("endpoint probability upper bounds must lie in [0,1]")
    cost_vector = tuple(as_fraction(value) for value in costs)
    counts = tuple(certified_counts)
    if len(cost_vector) != len(matrix) or len(counts) != len(matrix):
        raise PracticalityCertificateError("type dimensions differ")
    if any(value <= 0 for value in cost_vector):
        raise PracticalityCertificateError("costs must be positive")
    if any(not isinstance(value, int) or value < 0 for value in counts):
        raise PracticalityCertificateError("counts must be nonnegative integers")
    delta = as_fraction(simultaneous_failure_target)
    budget = as_fraction(evaluation_budget)
    if not (Fraction(0, 1) < delta < Fraction(1, 1)) or budget <= 0:
        raise PracticalityCertificateError("invalid failure target or budget")

    per_cell = []
    for j in range(len(matrix[0])):
        qmax = max(matrix[r][j] for r in range(len(matrix)))
        per_cell.append(minimum_independent_endpoints_for_cell(qmax, delta))
    universal = max(per_cell)
    subset_lower, subset_witnesses = subset_coverage_endpoint_lower(
        matrix, delta, max_cells_exact=max_cells_exact
    )
    combined_lower = max(universal, subset_lower, len(matrix[0]))
    min_cost = min(cost_vector) * combined_lower
    upper_count = sum(counts)
    upper_cost = sum(
        (cost_vector[r] * counts[r] for r in range(len(counts))),
        Fraction(0, 1),
    )
    ratio = None if min_cost == 0 else upper_cost / min_cost
    feasible = upper_cost <= budget
    verdict = (
        "PASS_WITHIN_DECLARED_BUDGET"
        if feasible
        else "INFEASIBLE_UNDER_DECLARED_BUDGET"
    )
    return PracticalityCertificate(
        endpoint_upper_by_type_cell=matrix,
        per_cell_endpoint_lower=tuple(per_cell),
        universal_endpoint_lower=universal,
        subset_endpoint_lower=subset_lower,
        combined_endpoint_lower=combined_lower,
        subset_lower_witnesses=subset_witnesses,
        minimum_endpoint_cost=min_cost,
        certified_upper_endpoint_count=upper_count,
        certified_upper_cost=upper_cost,
        upper_to_lower_cost_ratio=ratio,
        evaluation_budget=budget,
        budget_feasible=feasible,
        lower_bound_nontrivial=combined_lower > 0,
        verdict=verdict,
    )


__all__ = [
    "PracticalityCertificate",
    "PracticalityCertificateError",
    "binomial_upper_tail",
    "build_practicality_certificate",
    "minimum_endpoints_for_subset_hits",
    "minimum_independent_endpoints_for_cell",
    "subset_coverage_endpoint_lower",
]
