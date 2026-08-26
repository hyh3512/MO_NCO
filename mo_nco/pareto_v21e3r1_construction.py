from __future__ import annotations

"""Shared family-aware construction operators for V21e3r1 diagnostics.

The production C0--C3 hybrid and any development-only seeded baseline must use
these pure functions when they claim matched family-aware initialization.  This
module contains no ledger, archive, or random-state mutation.
"""

from typing import Sequence

from .moves import two_opt_at
from .pareto_ijoc_problem import (
    MultiObjectiveCombinatorialProblem,
    MultiObjectiveKnapsackInstance,
    MultiObjectiveTSPProblemAdapter,
    Solution,
)


def motsp_weighted_nearest_neighbor(
    problem: MultiObjectiveTSPProblemAdapter,
    direction: Sequence[float],
    variant: int,
) -> Solution:
    """Construct a fixed-origin tour and apply a deterministic 2-opt variant."""

    if variant < 0:
        raise ValueError("variant must be nonnegative")
    if len(direction) != problem.num_objectives:
        raise ValueError("direction dimension mismatch")
    remaining = set(range(1, problem.instance.num_cities))
    tour = [0]
    while remaining:
        current = tour[-1]
        following = min(
            remaining,
            key=lambda city: (
                sum(
                    float(weight) * matrix[current][city]
                    for weight, matrix in zip(
                        direction,
                        problem.instance.distance_matrices,
                    )
                ),
                city,
            ),
        )
        tour.append(following)
        remaining.remove(following)
    output = tuple(tour)
    if variant > 0 and len(output) >= 3:
        pairs = tuple(
            (left, right)
            for left in range(1, len(output) - 1)
            for right in range(left + 1, len(output))
        )
        if pairs:
            left, right = pairs[(variant - 1) % len(pairs)]
            output = two_opt_at(output, left, right)
    problem.validate_solution(output)
    return output


def mokp_directional_densities(
    problem: MultiObjectiveKnapsackInstance,
    direction: Sequence[float],
) -> tuple[float, ...]:
    """Return deterministic directional profit/weight scores."""

    if len(direction) != problem.num_objectives:
        raise ValueError("direction dimension mismatch")
    return tuple(
        sum(
            float(weight) * profits[index]
            for weight, profits in zip(direction, problem.profits_by_objective)
        )
        / problem.item_weights[index]
        for index in range(problem.solution_size)
    )


def mokp_repair(
    problem: MultiObjectiveKnapsackInstance,
    solution: Sequence[int],
    direction: Sequence[float],
    *,
    refill: bool,
) -> Solution:
    """Direction-aware deterministic capacity repair."""

    child = [int(value) for value in solution]
    if len(child) != problem.solution_size:
        raise ValueError("MOKP solution size mismatch")
    if any(value not in (0, 1) for value in child):
        raise ValueError("MOKP solution must be binary")
    densities = mokp_directional_densities(problem, direction)
    weight = sum(
        item_weight
        for selected, item_weight in zip(child, problem.item_weights)
        if selected
    )
    for index in sorted(
        (index for index, value in enumerate(child) if value),
        key=lambda index: (densities[index], index),
    ):
        if weight <= problem.capacity:
            break
        child[index] = 0
        weight -= problem.item_weights[index]
    if refill:
        for index in sorted(
            (index for index, value in enumerate(child) if not value),
            key=lambda index: (-densities[index], index),
        ):
            if weight + problem.item_weights[index] <= problem.capacity:
                child[index] = 1
                weight += problem.item_weights[index]
    output = tuple(child)
    problem.validate_solution(output)
    return output


def mokp_profit_density_construction(
    problem: MultiObjectiveKnapsackInstance,
    direction: Sequence[float],
    variant: int,
) -> Solution:
    """Strong deterministic greedy seed with an indexed drop/swap variant."""

    if variant < 0:
        raise ValueError("variant must be nonnegative")
    base = mokp_repair(
        problem,
        (0,) * problem.solution_size,
        direction,
        refill=True,
    )
    if variant == 0:
        return base
    densities = mokp_directional_densities(problem, direction)
    selected = sorted(
        (index for index, value in enumerate(base) if value),
        key=lambda index: (densities[index], index),
    )
    unselected = sorted(
        (index for index, value in enumerate(base) if not value),
        key=lambda index: (-densities[index], index),
    )
    child = list(base)
    if selected:
        child[selected[(variant - 1) % len(selected)]] = 0
    current_weight = sum(
        weight
        for value, weight in zip(child, problem.item_weights)
        if value
    )
    if unselected:
        start = (variant - 1) % len(unselected)
        for offset in range(len(unselected)):
            candidate = unselected[(start + offset) % len(unselected)]
            if current_weight + problem.item_weights[candidate] <= problem.capacity:
                child[candidate] = 1
                break
    return mokp_repair(problem, child, direction, refill=False)


def family_aware_initial_solution(
    problem: MultiObjectiveCombinatorialProblem,
    direction: Sequence[float],
    variant: int,
) -> tuple[Solution, str]:
    """Return the frozen family-aware seed and its semantic operator label."""

    if isinstance(problem, MultiObjectiveKnapsackInstance):
        return (
            mokp_profit_density_construction(problem, direction, variant),
            "mokp_profit_density_construction_repair_v21e3r1_shared_v1",
        )
    if isinstance(problem, MultiObjectiveTSPProblemAdapter):
        return (
            motsp_weighted_nearest_neighbor(problem, direction, variant),
            "motsp_weighted_nearest_neighbor_construction_v21e3r1_shared_v1",
        )
    raise TypeError(
        "Family-aware construction is implemented only for MOTSP and MOKP."
    )


__all__ = [
    "family_aware_initial_solution",
    "mokp_directional_densities",
    "mokp_profit_density_construction",
    "mokp_repair",
    "motsp_weighted_nearest_neighbor",
]


