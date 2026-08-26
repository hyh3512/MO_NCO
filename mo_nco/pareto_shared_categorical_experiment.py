from __future__ import annotations

"""Executable shared-categorical balanced pilot for the v16 theorem branch."""

from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Callable, Sequence

from .pareto_adaptive_type_cell import anytime_hoeffding_radius_upper
from .pareto_independent_replica_certificate import (
    canonical_rational_string,
    parse_canonical_probability,
)
from .pareto_shared_categorical_design import (
    SharedConfirmAllocationCertificate,
    exact_shared_confirm_allocation,
)

SHARED_CATEGORICAL_PILOT_SCHEMA_V16 = "pareto_shared_categorical_pilot_result_v16_1"


class SharedCategoricalPilotError(ValueError):
    pass


@dataclass(frozen=True)
class SharedCategoricalPilotCellResult:
    cell_id: str
    selected_type: str
    elimination_winner: str
    stopping_round: int
    empirical_means: tuple[tuple[str, str], ...]
    anytime_mass_lower_bounds: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SharedCategoricalPilotResult:
    schema: str
    type_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    familywise_error: str
    total_endpoint_replicas: int
    final_round: int
    final_anytime_radius_upper: str
    cell_results: tuple[SharedCategoricalPilotCellResult, ...]
    observation_model: str
    selection_rule: str
    optional_stopping_safe: bool

    def to_jsonable(self) -> dict[str, object]:
        payload = asdict(self)
        payload["cell_results"] = [asdict(item) for item in self.cell_results]
        return payload


def run_shared_categorical_successive_elimination(
    *,
    type_ids: Sequence[str],
    cell_ids: Sequence[str],
    sample_endpoint_cell: Callable[[str, int], str | None],
    familywise_error: Fraction | str,
    max_rounds: int,
) -> SharedCategoricalPilotResult:
    """Sample one categorical endpoint per type and round.

    ``sample_endpoint_cell(type_id, index)`` returns one frozen observable cell
    ID or ``None`` for an endpoint outside the certified subfamily.  For each
    type, calls over increasing ``index`` are assumed iid.  Cross-cell
    independence is neither true nor required.

    The reported mass lower bounds are ``max(0, p_hat-c_n)`` from the same
    all-time confidence event used by elimination.  They remain valid at the
    data-dependent stopping round.  Fixed-horizon Clopper--Pearson intervals
    are deliberately not used after optional stopping.
    """
    types = tuple(sorted(type_ids))
    cells = tuple(sorted(cell_ids))
    if len(types) < 2 or not cells:
        raise SharedCategoricalPilotError("Need at least two types and one cell.")
    if len(set(types)) != len(types) or len(set(cells)) != len(cells):
        raise SharedCategoricalPilotError("Type and cell IDs must be unique.")
    alpha = parse_canonical_probability(familywise_error, label="familywise_error")
    if not (0 < alpha < 1):
        raise SharedCategoricalPilotError("familywise_error must lie in (0,1).")
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds <= 0:
        raise SharedCategoricalPilotError("max_rounds must be a positive integer.")
    active = {cell: set(types) for cell in cells}
    stopped: dict[str, int] = {}
    counts = {type_id: 0 for type_id in types}
    successes = {(type_id, cell): 0 for type_id in types for cell in cells}
    final_round = 0
    for round_index in range(1, max_rounds + 1):
        final_round = round_index
        for type_id in types:
            outcome = sample_endpoint_cell(type_id, counts[type_id])
            if outcome is not None and outcome not in cells:
                raise SharedCategoricalPilotError(
                    f"Endpoint returned undeclared cell {outcome!r}."
                )
            counts[type_id] += 1
            if outcome is not None:
                successes[(type_id, outcome)] += 1
        radius = anytime_hoeffding_radius_upper(
            round_index,
            type_count=len(types),
            cell_count=len(cells),
            familywise_error=alpha,
        )
        for cell in cells:
            if cell in stopped:
                continue
            means = {
                type_id: Fraction(successes[(type_id, cell)], counts[type_id])
                for type_id in active[cell]
            }
            max_lower = max(means[type_id] - radius for type_id in active[cell])
            active[cell].difference_update(
                {
                    type_id
                    for type_id in active[cell]
                    if means[type_id] + radius < max_lower
                }
            )
            if len(active[cell]) == 1:
                stopped[cell] = round_index
        if len(stopped) == len(cells):
            break
    if len(stopped) != len(cells):
        raise SharedCategoricalPilotError("Pilot did not resolve every cell.")
    final_radius = anytime_hoeffding_radius_upper(
        final_round,
        type_count=len(types),
        cell_count=len(cells),
        familywise_error=alpha,
    )
    results: list[SharedCategoricalPilotCellResult] = []
    for cell in cells:
        means = {
            type_id: Fraction(successes[(type_id, cell)], counts[type_id])
            for type_id in types
        }
        lower = {
            type_id: max(Fraction(0), means[type_id] - final_radius)
            for type_id in types
        }
        selected = min(types, key=lambda t: (-lower[t], t))
        results.append(
            SharedCategoricalPilotCellResult(
                cell_id=cell,
                selected_type=selected,
                elimination_winner=min(active[cell]),
                stopping_round=stopped[cell],
                empirical_means=tuple(
                    (type_id, canonical_rational_string(means[type_id]))
                    for type_id in types
                ),
                anytime_mass_lower_bounds=tuple(
                    (type_id, canonical_rational_string(lower[type_id]))
                    for type_id in types
                ),
            )
        )
    return SharedCategoricalPilotResult(
        schema=SHARED_CATEGORICAL_PILOT_SCHEMA_V16,
        type_ids=types,
        cell_ids=cells,
        familywise_error=canonical_rational_string(alpha),
        total_endpoint_replicas=len(types) * final_round,
        final_round=final_round,
        final_anytime_radius_upper=canonical_rational_string(final_radius),
        cell_results=tuple(results),
        observation_model="one_iid_categorical_endpoint_sequence_per_type",
        selection_rule="shared_successive_elimination_then_anytime_lower_bound_max",
        optional_stopping_safe=True,
    )


def pilot_lower_bound_matrix(
    result: SharedCategoricalPilotResult,
) -> dict[str, dict[str, Fraction]]:
    """Transpose a pilot result into the exact type--cell lower-bound matrix."""
    matrix = {type_id: {} for type_id in result.type_ids}
    for cell_result in result.cell_results:
        for type_id, value in cell_result.anytime_mass_lower_bounds:
            matrix[type_id][cell_result.cell_id] = Fraction(value)
    return matrix


def plan_shared_confirm_from_pilot(
    result: SharedCategoricalPilotResult,
    *,
    union_miss_budget: Fraction | str,
    max_assignments: int = 1_000_000,
    max_total_replicas: int = 10_000_000,
) -> SharedConfirmAllocationCertificate:
    """Plan confirm counts from optional-stopping-safe pilot lower bounds."""
    return exact_shared_confirm_allocation(
        pilot_lower_bound_matrix(result),
        union_miss_budget=union_miss_budget,
        max_assignments=max_assignments,
        max_total_replicas=max_total_replicas,
    )


__all__ = [
    "SHARED_CATEGORICAL_PILOT_SCHEMA_V16",
    "SharedCategoricalPilotCellResult",
    "SharedCategoricalPilotError",
    "SharedCategoricalPilotResult",
    "pilot_lower_bound_matrix",
    "plan_shared_confirm_from_pilot",
    "run_shared_categorical_successive_elimination",
]
