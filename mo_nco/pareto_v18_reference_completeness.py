"""Exact finite-state reference-completeness certificates for Pareto-SMC v18.

The executable true-front branch is intentionally narrow: it enumerates every
fixed-origin Hamiltonian tour of a small explicit multi-objective TSP instance,
recomputes the exact Pareto objective set, and verifies that a frozen reference
set covers that set.  Large instances without a separately verified
problem-specific proof remain reference-relative.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import json
import math
from typing import Sequence

from .pareto_v17_regeneration import as_fraction


class ReferenceCompletenessError(ValueError):
    pass


Point = tuple[Fraction, ...]
Tour = tuple[int, ...]


def _validate_matrices(
    raw: Sequence[Sequence[Sequence[Fraction | int | str]]],
) -> tuple[tuple[tuple[Fraction, ...], ...], ...]:
    matrices = tuple(
        tuple(tuple(as_fraction(value) for value in row) for row in matrix)
        for matrix in raw
    )
    if not matrices or not matrices[0]:
        raise ReferenceCompletenessError("objective matrices must be nonempty")
    n = len(matrices[0])
    if n < 3:
        raise ReferenceCompletenessError("TSP enumeration requires at least three cities")
    for matrix in matrices:
        if len(matrix) != n or any(len(row) != n for row in matrix):
            raise ReferenceCompletenessError("objective matrices must be square and share a dimension")
        if any(value < 0 for row in matrix for value in row):
            raise ReferenceCompletenessError("formal enumeration requires nonnegative exact costs")
    return matrices


def evaluate_tour_exact(
    matrices: Sequence[Sequence[Sequence[Fraction]]],
    tour: Tour,
) -> Point:
    n = len(tour)
    if sorted(tour) != list(range(n)) or tour[0] != 0:
        raise ReferenceCompletenessError("tour must be a fixed-origin permutation")
    return tuple(
        sum(
            (matrix[tour[i]][tour[(i + 1) % n]] for i in range(n)),
            Fraction(0, 1),
        )
        for matrix in matrices
    )


def weakly_dominates(left: Point, right: Point) -> bool:
    return all(a <= b for a, b in zip(left, right, strict=True))


def strictly_dominates(left: Point, right: Point) -> bool:
    return weakly_dominates(left, right) and any(
        a < b for a, b in zip(left, right, strict=True)
    )


def nondominated(points: Sequence[Point]) -> tuple[Point, ...]:
    unique = sorted(set(points))
    front = [
        point
        for i, point in enumerate(unique)
        if not any(strictly_dominates(other, point) for j, other in enumerate(unique) if i != j)
    ]
    return tuple(front)


def additive_cover_holds(reference: Sequence[Point], targets: Sequence[Point], eta: Point) -> bool:
    return all(
        any(
            all(q_i <= p_i + eta_i for q_i, p_i, eta_i in zip(q, p, eta, strict=True))
            for q in reference
        )
        for p in targets
    )


def directed_metric_radius(reference: Sequence[Point], targets: Sequence[Point], p: str) -> Fraction:
    def distance(left: Point, right: Point) -> Fraction:
        diffs = tuple(abs(a - b) for a, b in zip(left, right, strict=True))
        if p == "1":
            return sum(diffs, Fraction(0, 1))
        if p == "infinity":
            return max(diffs)
        if p == "2_squared":
            return sum((value * value for value in diffs), Fraction(0, 1))
        raise ReferenceCompletenessError("metric must be '1', 'infinity', or '2_squared'")

    return max(min(distance(target, q) for q in reference) for target in targets)


@dataclass(frozen=True)
class ExactTSPReferenceCompletenessCertificate:
    city_count: int
    objective_count: int
    enumerated_tour_count: int
    exact_pareto_front: tuple[Point, ...]
    frozen_reference: tuple[Point, ...]
    additive_eta: Point
    additive_cover_verified: bool
    directed_metric_radius_l1: Fraction
    directed_metric_radius_linf: Fraction
    objective_table_sha256: str
    true_front_sha256: str
    scope: str = "exact_fixed_origin_tsp_true_front_v18"

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "city_count": self.city_count,
            "objective_count": self.objective_count,
            "enumerated_tour_count": self.enumerated_tour_count,
            "exact_pareto_front": [[str(value) for value in point] for point in self.exact_pareto_front],
            "frozen_reference": [[str(value) for value in point] for point in self.frozen_reference],
            "additive_eta": [str(value) for value in self.additive_eta],
            "additive_cover_verified": self.additive_cover_verified,
            "directed_metric_radius_l1": str(self.directed_metric_radius_l1),
            "directed_metric_radius_linf": str(self.directed_metric_radius_linf),
            "objective_table_sha256": self.objective_table_sha256,
            "true_front_sha256": self.true_front_sha256,
        }


def certify_exact_tsp_reference_completeness(
    objective_matrices: Sequence[Sequence[Sequence[Fraction | int | str]]],
    frozen_reference: Sequence[Sequence[Fraction | int | str]],
    additive_eta: Sequence[Fraction | int | str],
    *,
    max_tours: int = 2_000_000,
) -> ExactTSPReferenceCompletenessCertificate:
    matrices = _validate_matrices(objective_matrices)
    n = len(matrices[0])
    total_tours = math.factorial(n - 1)
    if total_tours > max_tours:
        raise ReferenceCompletenessError(
            f"exact enumeration needs {total_tours} tours, exceeding max_tours={max_tours}"
        )
    reference = tuple(tuple(as_fraction(value) for value in point) for point in frozen_reference)
    if not reference or any(len(point) != len(matrices) for point in reference):
        raise ReferenceCompletenessError("frozen reference has the wrong dimension")
    eta = tuple(as_fraction(value) for value in additive_eta)
    if len(eta) != len(matrices) or any(value < 0 for value in eta):
        raise ReferenceCompletenessError("additive eta has the wrong dimension or a negative entry")

    objective_rows: list[tuple[Tour, Point]] = []
    for suffix in itertools.permutations(range(1, n)):
        tour = (0, *suffix)
        objective_rows.append((tour, evaluate_tour_exact(matrices, tour)))
    all_points = tuple(point for _, point in objective_rows)
    point_set = set(all_points)
    if any(point not in point_set for point in reference):
        raise ReferenceCompletenessError("every frozen reference point must have an exact feasible witness")
    front = nondominated(all_points)
    cover = additive_cover_holds(reference, front, eta)
    if not cover:
        raise ReferenceCompletenessError("frozen reference fails the declared true-front additive cover")

    table_payload = [
        {"tour": list(tour), "objective": [str(value) for value in point]}
        for tour, point in objective_rows
    ]
    table_sha = hashlib.sha256(
        json.dumps(table_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    front_payload = [[str(value) for value in point] for point in front]
    front_sha = hashlib.sha256(
        json.dumps(front_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ExactTSPReferenceCompletenessCertificate(
        city_count=n,
        objective_count=len(matrices),
        enumerated_tour_count=total_tours,
        exact_pareto_front=front,
        frozen_reference=reference,
        additive_eta=eta,
        additive_cover_verified=True,
        directed_metric_radius_l1=directed_metric_radius(reference, front, "1"),
        directed_metric_radius_linf=directed_metric_radius(reference, front, "infinity"),
        objective_table_sha256=table_sha,
        true_front_sha256=front_sha,
    )


def compose_additive_error(
    reference_eta: Sequence[Fraction | int | str],
    algorithm_epsilon: Sequence[Fraction | int | str],
) -> tuple[Fraction, ...]:
    eta = tuple(as_fraction(value) for value in reference_eta)
    eps = tuple(as_fraction(value) for value in algorithm_epsilon)
    if len(eta) != len(eps) or any(value < 0 for value in (*eta, *eps)):
        raise ReferenceCompletenessError("additive vectors have incompatible dimensions")
    return tuple(a + b for a, b in zip(eta, eps, strict=True))


__all__ = [
    "ExactTSPReferenceCompletenessCertificate",
    "ReferenceCompletenessError",
    "additive_cover_holds",
    "certify_exact_tsp_reference_completeness",
    "compose_additive_error",
    "evaluate_tour_exact",
    "nondominated",
]
