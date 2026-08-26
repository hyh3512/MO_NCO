from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from .archive import dominates
from .types import ObjectiveVector


def nondominated_points(points: Sequence[ObjectiveVector]) -> Tuple[ObjectiveVector, ...]:
    front: List[ObjectiveVector] = []
    for idx, point in enumerate(points):
        if any(j != idx and dominates(other, point) for j, other in enumerate(points)):
            continue
        if point not in front:
            front.append(point)
    return tuple(sorted(front))


def normalize_points(
    points: Sequence[ObjectiveVector],
    ideal: ObjectiveVector,
    nadir: ObjectiveVector,
) -> Tuple[ObjectiveVector, ...]:
    normalized = []
    for point in points:
        normalized.append(
            tuple((value - lo) / max(1e-12, hi - lo) for value, lo, hi in zip(point, ideal, nadir))
        )
    return tuple(normalized)


def empirical_reference_front(fronts: Sequence[Sequence[ObjectiveVector]]) -> Tuple[ObjectiveVector, ...]:
    return nondominated_points([point for front in fronts for point in front])


def ideal_nadir(points: Sequence[ObjectiveVector]) -> Tuple[ObjectiveVector, ObjectiveVector]:
    if not points:
        raise ValueError("Cannot compute ideal/nadir for an empty point set.")
    dim = len(points[0])
    ideal = tuple(min(point[i] for point in points) for i in range(dim))
    nadir = tuple(max(point[i] for point in points) for i in range(dim))
    return ideal, nadir


def igd_plus(approximation: Sequence[ObjectiveVector], reference: Sequence[ObjectiveVector]) -> float:
    """IGD+ for minimization on already normalized points."""
    if not approximation or not reference:
        return float("inf")
    total = 0.0
    for ref in reference:
        best = float("inf")
        for point in approximation:
            dist2 = sum(max(point_i - ref_i, 0.0) ** 2 for point_i, ref_i in zip(point, ref))
            best = min(best, math.sqrt(dist2))
        total += best
    return total / len(reference)


def additive_epsilon(approximation: Sequence[ObjectiveVector], reference: Sequence[ObjectiveVector]) -> float:
    """Additive epsilon indicator for minimization on normalized points."""
    if not approximation or not reference:
        return float("inf")
    worst = -float("inf")
    for ref in reference:
        best_for_ref = float("inf")
        for point in approximation:
            best_for_ref = min(best_for_ref, max(point_i - ref_i for point_i, ref_i in zip(point, ref)))
        worst = max(worst, best_for_ref)
    return worst


def spacing(approximation: Sequence[ObjectiveVector]) -> float:
    if len(approximation) <= 2:
        return 0.0
    nearest = []
    for idx, point in enumerate(approximation):
        best = float("inf")
        for j, other in enumerate(approximation):
            if idx == j:
                continue
            distance = sum(abs(a - b) for a, b in zip(point, other))
            best = min(best, distance)
        nearest.append(best)
    avg = sum(nearest) / len(nearest)
    return math.sqrt(sum((value - avg) ** 2 for value in nearest) / (len(nearest) - 1))
