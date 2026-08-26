from __future__ import annotations

"""Shared categorical pilot and confirm design certificates.

One endpoint from type ``r`` is a categorical observation over the frozen cell
family.  The same endpoint therefore updates every cell indicator for that
``r``.  This is the correct finite-reference statistical object; it avoids the
artificial duplication of one independent run per (type, cell) pair.
"""

from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from typing import Mapping, Sequence

from .pareto_adaptive_type_cell import anytime_hoeffding_radius_upper
from .pareto_independent_replica_certificate import (
    canonical_rational_string,
    parse_canonical_probability,
)

SHARED_IDENTIFICATION_SCHEMA_V16 = "pareto_shared_categorical_identification_v16"
SHARED_CONFIRM_SCHEMA_V16 = "pareto_shared_categorical_confirm_allocation_v16"
RATIONAL_TRANSPORT_SCHEMA_V16 = "pareto_rational_transport_lower_bound_v16"


class SharedCategoricalDesignError(ValueError):
    pass


def _probability(value: Fraction | str | int, *, label: str) -> Fraction:
    result = parse_canonical_probability(value, label=label)
    if result < 0 or result > 1:
        raise SharedCategoricalDesignError(f"{label} must lie in [0,1].")
    return result


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SharedCategoricalDesignError(f"{label} must be a positive integer.")
    return value


def _matrix(
    probabilities: Mapping[str, Mapping[str, Fraction | str | int]],
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, dict[str, Fraction]]]:
    if not probabilities:
        raise SharedCategoricalDesignError("At least one type is required.")
    type_ids = tuple(sorted(probabilities))
    cell_ids: tuple[str, ...] | None = None
    matrix: dict[str, dict[str, Fraction]] = {}
    for type_id in type_ids:
        if not isinstance(type_id, str) or not type_id:
            raise SharedCategoricalDesignError("Type IDs must be nonempty strings.")
        row = probabilities[type_id]
        if not row:
            raise SharedCategoricalDesignError("Every type must declare at least one cell.")
        observed_cells = tuple(sorted(row))
        if cell_ids is None:
            cell_ids = observed_cells
        elif observed_cells != cell_ids:
            raise SharedCategoricalDesignError("Every type must declare the same cell family.")
        resolved = {
            cell: _probability(row[cell], label=f"p[{type_id},{cell}]")
            for cell in observed_cells
        }
        if sum(resolved.values(), Fraction(0)) > 1:
            raise SharedCategoricalDesignError(
                f"Categorical cell probabilities for type {type_id!r} exceed one."
            )
        matrix[type_id] = resolved
    assert cell_ids is not None
    if len(type_ids) < 2:
        raise SharedCategoricalDesignError("Identification requires at least two types.")
    return type_ids, cell_ids, matrix


@dataclass(frozen=True)
class SharedCellIdentificationBound:
    cell_id: str
    best_type: str
    best_probability: str
    second_probability: str
    gap: str
    stopping_round_upper: int


@dataclass(frozen=True)
class SharedCategoricalIdentificationCertificate:
    schema: str
    type_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    familywise_error: str
    cell_bounds: tuple[SharedCellIdentificationBound, ...]
    total_pilot_replicas_upper: int
    largest_stopping_round_upper: int
    cost_improvement_over_cell_separated_upper: int
    unique_best_required: bool
    observation_model: str
    theorem_scope: str

    def to_jsonable(self) -> dict[str, object]:
        payload = asdict(self)
        payload["cell_bounds"] = [asdict(item) for item in self.cell_bounds]
        return payload


def certify_shared_categorical_identification_upper_bound(
    probabilities: Mapping[str, Mapping[str, Fraction | str | int]], *,
    familywise_error: Fraction | str,
    max_rounds: int = 1_000_000,
) -> SharedCategoricalIdentificationCertificate:
    """Balanced shared-endpoint successive-elimination upper bound.

    One endpoint is sampled from every type in each round and all frozen cell
    indicators are updated.  On the simultaneous confidence event, every cell
    resolves by the first round with ``4*c_n`` smaller than its best-v-second
    gap.  Total pilot runs are therefore ``R * max_j N_j``.
    """
    type_ids, cell_ids, matrix = _matrix(probabilities)
    alpha = _probability(familywise_error, label="familywise_error")
    if not (0 < alpha < 1):
        raise SharedCategoricalDesignError("familywise_error must lie in (0,1).")
    maximum = _positive_integer(max_rounds, label="max_rounds")
    bounds: list[SharedCellIdentificationBound] = []
    stopping_rounds: list[int] = []
    for cell in cell_ids:
        ranking = sorted(
            ((matrix[type_id][cell], type_id) for type_id in type_ids),
            key=lambda item: (-item[0], item[1]),
        )
        best_probability, best_type = ranking[0]
        second_probability = ranking[1][0]
        gap = best_probability - second_probability
        if gap <= 0:
            raise SharedCategoricalDesignError(
                f"Cell {cell!r} does not have a unique best type."
            )
        stop: int | None = None
        for round_index in range(1, maximum + 1):
            radius = anytime_hoeffding_radius_upper(
                round_index,
                type_count=len(type_ids),
                cell_count=len(cell_ids),
                familywise_error=alpha,
            )
            if 4 * radius < gap:
                stop = round_index
                break
        if stop is None:
            raise SharedCategoricalDesignError(
                f"No certified stopping round through {maximum} for cell {cell!r}."
            )
        stopping_rounds.append(stop)
        bounds.append(
            SharedCellIdentificationBound(
                cell_id=cell,
                best_type=best_type,
                best_probability=canonical_rational_string(best_probability),
                second_probability=canonical_rational_string(second_probability),
                gap=canonical_rational_string(gap),
                stopping_round_upper=stop,
            )
        )
    largest = max(stopping_rounds)
    shared_cost = len(type_ids) * largest
    separated_cost = len(type_ids) * sum(stopping_rounds)
    return SharedCategoricalIdentificationCertificate(
        schema=SHARED_IDENTIFICATION_SCHEMA_V16,
        type_ids=type_ids,
        cell_ids=cell_ids,
        familywise_error=canonical_rational_string(alpha),
        cell_bounds=tuple(bounds),
        total_pilot_replicas_upper=shared_cost,
        largest_stopping_round_upper=largest,
        cost_improvement_over_cell_separated_upper=separated_cost - shared_cost,
        unique_best_required=True,
        observation_model="iid_categorical_endpoint_sequence_per_type",
        theorem_scope=(
            "frozen_finite_cell_family;cross_cell_independence_not_required;"
            "within_type_endpoint_sequence_iid"
        ),
    )


@dataclass(frozen=True)
class SharedConfirmAllocationCertificate:
    schema: str
    type_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    assignment_type_by_cell: tuple[str, ...]
    replicas_by_type: tuple[int, ...]
    cell_miss_upper: tuple[str, ...]
    total_union_miss_upper: str
    target_union_miss_budget: str
    total_replicas: int
    assignments_enumerated: int
    exact_single_type_assignment_optimum: bool
    allocation_rule: str
    theorem_scope: str

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


def _fixed_assignment_greedy(
    *,
    type_ids: tuple[str, ...],
    cell_ids: tuple[str, ...],
    matrix: dict[str, dict[str, Fraction]],
    assignment: tuple[int, ...],
    delta: Fraction,
    max_total_replicas: int,
) -> tuple[int, tuple[int, ...], tuple[Fraction, ...], Fraction]:
    replicas = [0 for _ in type_ids]
    miss = [Fraction(1) for _ in cell_ids]
    risk = Fraction(len(cell_ids))
    total = 0
    while risk > delta:
        if total >= max_total_replicas:
            raise SharedCategoricalDesignError(
                "Shared confirm allocation exceeds max_total_replicas."
            )
        aggregate_marginals: list[Fraction] = []
        for type_index, type_id in enumerate(type_ids):
            value = Fraction(0)
            for cell_index, cell_id in enumerate(cell_ids):
                if assignment[cell_index] == type_index:
                    q = matrix[type_id][cell_id]
                    value += q * miss[cell_index]
            aggregate_marginals.append(value)
        chosen = max(
            range(len(type_ids)),
            key=lambda index: (aggregate_marginals[index], -index),
        )
        if aggregate_marginals[chosen] <= 0:
            raise SharedCategoricalDesignError(
                "The assigned cells have zero certified mass under every remaining type."
            )
        old_risk = risk
        type_id = type_ids[chosen]
        for cell_index, cell_id in enumerate(cell_ids):
            if assignment[cell_index] == chosen:
                old = miss[cell_index]
                miss[cell_index] *= 1 - matrix[type_id][cell_id]
                risk -= old - miss[cell_index]
        if risk >= old_risk:
            raise AssertionError("A positive shared-confirm marginal did not reduce risk.")
        replicas[chosen] += 1
        total += 1
    return total, tuple(replicas), tuple(miss), risk


def exact_shared_confirm_allocation(
    hit_probability_lower_bounds: Mapping[str, Mapping[str, Fraction | str | int]], *,
    union_miss_budget: Fraction | str,
    max_assignments: int = 1_000_000,
    max_total_replicas: int = 10_000_000,
) -> SharedConfirmAllocationCertificate:
    """Exact assignment-plus-count optimum for the union-miss proxy.

    For a fixed cell-to-type assignment, adding one endpoint from type ``r``
    yields the nonincreasing aggregate marginal

        sum_{j:a(j)=r} q[r,j] (1-q[r,j])**m_r.

    Largest-marginal greedy is exact for every fixed total count.  Enumerating
    all assignments and selecting the first-crossing minimum is therefore the
    exact global integer optimum over the frozen finite assignment family.
    """
    type_ids, cell_ids, matrix = _matrix(hit_probability_lower_bounds)
    delta = _probability(union_miss_budget, label="union_miss_budget")
    if not (0 < delta < 1):
        raise SharedCategoricalDesignError("union_miss_budget must lie in (0,1).")
    assignment_cap = _positive_integer(max_assignments, label="max_assignments")
    replica_cap = _positive_integer(max_total_replicas, label="max_total_replicas")
    assignment_count = len(type_ids) ** len(cell_ids)
    if assignment_count > assignment_cap:
        raise SharedCategoricalDesignError(
            f"Assignment count {assignment_count} exceeds max_assignments={assignment_cap}."
        )
    best: tuple[
        int,
        Fraction,
        tuple[int, ...],
        tuple[int, ...],
        tuple[Fraction, ...],
    ] | None = None
    for encoded in range(assignment_count):
        value = encoded
        assignment_values = []
        for _ in cell_ids:
            assignment_values.append(value % len(type_ids))
            value //= len(type_ids)
        assignment = tuple(assignment_values)
        if any(matrix[type_ids[assignment[j]]][cell_ids[j]] <= 0 for j in range(len(cell_ids))):
            continue
        total, replicas, misses, risk = _fixed_assignment_greedy(
            type_ids=type_ids,
            cell_ids=cell_ids,
            matrix=matrix,
            assignment=assignment,
            delta=delta,
            max_total_replicas=replica_cap,
        )
        candidate = (total, risk, assignment, replicas, misses)
        if best is None or candidate[:4] < best[:4]:
            best = candidate
    if best is None:
        raise SharedCategoricalDesignError("No positive-mass cell-to-type assignment exists.")
    total, risk, assignment, replicas, misses = best
    return SharedConfirmAllocationCertificate(
        schema=SHARED_CONFIRM_SCHEMA_V16,
        type_ids=type_ids,
        cell_ids=cell_ids,
        assignment_type_by_cell=tuple(type_ids[index] for index in assignment),
        replicas_by_type=replicas,
        cell_miss_upper=tuple(canonical_rational_string(value) for value in misses),
        total_union_miss_upper=canonical_rational_string(risk),
        target_union_miss_budget=canonical_rational_string(delta),
        total_replicas=total,
        assignments_enumerated=assignment_count,
        exact_single_type_assignment_optimum=True,
        allocation_rule=(
            "enumerate_cell_to_type_assignments_then_exact_largest_aggregate_marginal_greedy"
        ),
        theorem_scope="union_bound_proxy_with_shared_iid_categorical_endpoints_per_type",
    )


@dataclass(frozen=True)
class RationalTransportationLowerBound:
    schema: str
    cell_id: str
    best_type: str
    challenger_type: str
    best_probability: str
    challenger_probability: str
    error_probability: str
    log_series_terms: int
    binary_decision_kl_lower: str
    midpoint_arm_kl_upper: str
    expected_total_samples_lower: str
    theorem_scope: str

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


def rational_pairwise_transportation_lower_bound(
    *,
    cell_id: str,
    best_type: str,
    challenger_type: str,
    best_probability: Fraction | str | int,
    challenger_probability: Fraction | str | int,
    error_probability: Fraction | str,
    log_series_terms: int = 64,
) -> RationalTransportationLowerBound:
    """Exact rational lower bound implied by the transportation inequality.

    With y=1-2*alpha,

        kl(1-alpha, alpha) = 2 sum_{k>=0} y^(2k+2)/(2k+1).

    Truncation is a lower bound.  For midpoint m and gap Delta,

        kl(p,m) <= (p-m)^2/[m(1-m)]
                 = Delta^2/[4m(1-m)].

    Dividing the rational lower numerator by this rational upper denominator
    gives a machine-checkable conservative sample lower bound.
    """
    p_star = _probability(best_probability, label="best_probability")
    p_other = _probability(challenger_probability, label="challenger_probability")
    alpha = _probability(error_probability, label="error_probability")
    terms = _positive_integer(log_series_terms, label="log_series_terms")
    if not (0 < alpha < Fraction(1, 2)):
        raise SharedCategoricalDesignError("error_probability must lie in (0,1/2).")
    if not p_star > p_other:
        raise SharedCategoricalDesignError(
            "best_probability must strictly exceed challenger_probability."
        )
    y = 1 - 2 * alpha
    decision_lower = Fraction(0)
    for index in range(terms):
        decision_lower += Fraction(2, 2 * index + 1) * y ** (2 * index + 2)
    midpoint = (p_star + p_other) / 2
    gap = p_star - p_other
    arm_upper = gap * gap / (4 * midpoint * (1 - midpoint))
    if arm_upper <= 0:
        raise AssertionError("Midpoint KL upper bound must be positive.")
    lower = decision_lower / arm_upper
    return RationalTransportationLowerBound(
        schema=RATIONAL_TRANSPORT_SCHEMA_V16,
        cell_id=cell_id,
        best_type=best_type,
        challenger_type=challenger_type,
        best_probability=canonical_rational_string(p_star),
        challenger_probability=canonical_rational_string(p_other),
        error_probability=canonical_rational_string(alpha),
        log_series_terms=terms,
        binary_decision_kl_lower=canonical_rational_string(decision_lower),
        midpoint_arm_kl_upper=canonical_rational_string(arm_upper),
        expected_total_samples_lower=canonical_rational_string(lower),
        theorem_scope=(
            "uniform_alpha_correct_identifier;almost_surely_finite_stopping;"
            "ideal_iid_Bernoulli_observations"
        ),
    )


__all__ = [
    "RATIONAL_TRANSPORT_SCHEMA_V16",
    "SHARED_CONFIRM_SCHEMA_V16",
    "SHARED_IDENTIFICATION_SCHEMA_V16",
    "RationalTransportationLowerBound",
    "SharedCategoricalDesignError",
    "SharedCategoricalIdentificationCertificate",
    "SharedCellIdentificationBound",
    "SharedConfirmAllocationCertificate",
    "certify_shared_categorical_identification_upper_bound",
    "exact_shared_confirm_allocation",
    "rational_pairwise_transportation_lower_bound",
]
