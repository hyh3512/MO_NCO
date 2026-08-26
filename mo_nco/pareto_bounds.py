from __future__ import annotations

"""Auditable finite-particle to Pareto-indicator geometry certificates.

The certificate is deliberately conditional.  It consumes:

* a finite, externally fixed feasible objective set;
* an externally fixed normalization box and normalized epsilon-cell widths;
* final Feynman--Kac target masses for the resulting Pareto cells; and
* an externally proved cellwise mean-square particle-error constant ``B_L``.

It never estimates ``p_min`` or ``B_L`` from ESS.  Under

    E[(eta_L^M(C) - eta_L(C))^2] <= B_L / M

for every Pareto cell, a union--Chebyshev argument gives

    P(any Pareto cell is empty) <= K B_L / (M p_min^2).

On the cell-hit event, same-cell witnesses give componentwise additive
coverage, IGD_p bounds, and a fixed-reference minimization hypervolume deficit
bound.  Cells are half-open in every coordinate, except that the global upper
box boundary is closed.
"""

import math
from collections import Counter
from typing import Any, Mapping, Sequence


SCHEMA = "pareto_smc_geometric_bound_certificate_v2"
LEGACY_SUPERSEDED_SCHEMAS = frozenset(
    {"pareto_smc_geometric_bound_certificate_v1"}
)

Point = tuple[float, ...]
Cell = tuple[int, ...]


class OutOfBoxError(ValueError):
    """Raised rather than silently clipping an objective outside the frozen box."""


def _product(values: Sequence[float]) -> float:
    result = 1.0
    for value in values:
        result *= value
    return result


def _lp_norm(values: Sequence[float], p: float) -> float:
    if math.isinf(p):
        return max(values, default=0.0)
    return sum(value**p for value in values) ** (1.0 / p)


def _coerce_point(point: Sequence[float], dimension: int, label: str) -> Point:
    if len(point) != dimension:
        raise ValueError(
            f"{label} has dimension {len(point)}; expected {dimension}."
        )
    result = tuple(float(value) for value in point)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} contains a non-finite coordinate.")
    return result


def _normalize_points(
    points: Sequence[Sequence[float]],
    *,
    lower: Point,
    upper: Point,
    label: str,
) -> tuple[Point, ...]:
    normalized: list[Point] = []
    spans = tuple(hi - lo for lo, hi in zip(lower, upper))
    for point_index, raw_point in enumerate(points):
        point = _coerce_point(raw_point, len(lower), f"{label}[{point_index}]")
        for coordinate, (value, lo, hi) in enumerate(zip(point, lower, upper)):
            if value < lo or value > hi:
                side = "below" if value < lo else "above"
                raise OutOfBoxError(
                    f"{label}[{point_index}][{coordinate}]={value!r} is {side} "
                    f"the frozen objective box [{lo!r}, {hi!r}]."
                )
        normalized.append(
            tuple((value - lo) / span for value, lo, span in zip(point, lower, spans))
        )
    return tuple(normalized)


def _dominates(left: Point, right: Point) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(
        a < b for a, b in zip(left, right)
    )


def nondominated_points(points: Sequence[Point]) -> tuple[Point, ...]:
    """Return unique minimization-nondominated points in lexicographic order."""

    unique = tuple(sorted(set(points)))
    return tuple(
        point
        for index, point in enumerate(unique)
        if not any(
            other_index != index and _dominates(other, point)
            for other_index, other in enumerate(unique)
        )
    )


def normalized_cell_index(
    point: Sequence[float],
    normalized_cell_widths: Sequence[float],
) -> Cell:
    """Index a normalized point using half-open cells and a closed global upper."""

    if len(point) != len(normalized_cell_widths):
        raise ValueError("Point and cell widths have different dimensions.")
    indices: list[int] = []
    for coordinate, (raw_value, raw_width) in enumerate(
        zip(point, normalized_cell_widths)
    ):
        value = float(raw_value)
        width = float(raw_width)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise OutOfBoxError(
                f"normalized point coordinate {coordinate}={value!r} "
                "is outside [0, 1]."
            )
        if not math.isfinite(width) or width <= 0.0 or width > 1.0:
            raise ValueError("Normalized cell widths must lie in (0, 1].")
        cell_count = math.ceil(1.0 / width)
        if value == 1.0:
            indices.append(cell_count - 1)
        else:
            indices.append(math.floor(value / width))
    return tuple(indices)


def _igd_p(
    approximation: Sequence[Point],
    reference: Sequence[Point],
    p: float,
) -> float:
    if not approximation or not reference:
        return math.inf
    distances = [
        min(
            _lp_norm(tuple(abs(a - r) for a, r in zip(point, ref)), p)
            for point in approximation
        )
        for ref in reference
    ]
    # ``p`` selects the local objective-space norm only.  Standard IGD then
    # takes the arithmetic mean over reference points.  The v1 certificate
    # incorrectly reused ``p`` for a second, cross-reference power mean; v2
    # intentionally changes that metric contract and supersedes v1 artifacts.
    return sum(distances) / len(distances)


def _igd_plus_p(
    approximation: Sequence[Point],
    reference: Sequence[Point],
    p: float,
) -> float:
    """One-sided minimization IGD+ with a local p-norm and arithmetic mean."""

    if not approximation or not reference:
        return math.inf
    distances = [
        min(
            _lp_norm(
                tuple(max(a - r, 0.0) for a, r in zip(point, ref)),
                p,
            )
            for point in approximation
        )
        for ref in reference
    ]
    return sum(distances) / len(distances)


def hypervolume_minimization(
    points: Sequence[Point],
    reference: Sequence[float],
) -> float:
    """Exact fixed-reference dominated hypervolume by recursive axis slicing."""

    ref = tuple(float(value) for value in reference)
    if not ref or not all(math.isfinite(value) for value in ref):
        raise ValueError("Hypervolume reference must be finite and nonempty.")
    coerced = tuple(
        _coerce_point(point, len(ref), f"hypervolume_points[{index}]")
        for index, point in enumerate(points)
    )
    relevant = nondominated_points(
        tuple(point for point in coerced if all(x < r for x, r in zip(point, ref)))
    )

    def recurse(active: tuple[Point, ...], local_ref: Point) -> float:
        if not active:
            return 0.0
        if len(local_ref) == 1:
            return max(0.0, local_ref[0] - min(point[0] for point in active))
        levels = sorted({point[0] for point in active if point[0] < local_ref[0]})
        levels.append(local_ref[0])
        volume = 0.0
        for left, right in zip(levels, levels[1:]):
            if right <= left:
                continue
            projection = tuple(
                point[1:] for point in active if point[0] <= left
            )
            volume += (right - left) * recurse(
                nondominated_points(projection),
                local_ref[1:],
            )
        return volume

    return recurse(relevant, ref)


def _cell_key(cell: Cell) -> str:
    return ",".join(str(index) for index in cell)


def certify_pareto_bounds(
    feasible_objectives: Sequence[Sequence[float]],
    particle_objectives: Sequence[Sequence[float]],
    *,
    particle_weights: Sequence[float] | None = None,
    objective_lower: Sequence[float],
    objective_upper: Sequence[float],
    normalized_cell_widths: Sequence[float],
    target_pareto_cell_probabilities: Mapping[Sequence[int], float],
    declared_p_min: float,
    cellwise_mse_constant_B_L: float,
    confidence_delta: float,
    igd_p: float = 2.0,
    normalized_hv_reference: Sequence[float] | None = None,
    declared_cellwise_error_radius: float | None = None,
    declared_error_failure_probability: float | None = None,
) -> dict[str, Any]:
    """Build a conditional finite-particle Pareto geometry certificate.

    ``feasible_objectives`` must enumerate the finite feasible objective set,
    not an empirical front.  Every particle objective must occur in that set.
    Target cell masses, ``declared_p_min``, and ``B_L`` are external theorem
    inputs and are checked for internal consistency but not learned here.
    """

    if not objective_lower:
        raise ValueError("The frozen objective box must have positive dimension.")
    dimension = len(objective_lower)
    lower = _coerce_point(objective_lower, dimension, "objective_lower")
    upper = _coerce_point(objective_upper, dimension, "objective_upper")
    if any(hi <= lo for lo, hi in zip(lower, upper)):
        raise ValueError("Every objective upper bound must exceed its lower bound.")
    widths = _coerce_point(
        normalized_cell_widths,
        dimension,
        "normalized_cell_widths",
    )
    if any(width <= 0.0 or width > 1.0 for width in widths):
        raise ValueError("Normalized cell widths must lie in (0, 1].")
    if not feasible_objectives:
        raise ValueError("The finite feasible objective set cannot be empty.")
    if not particle_objectives:
        raise ValueError("At least one particle is required.")
    if particle_weights is None:
        normalized_particle_weights = tuple(
            1.0 / len(particle_objectives)
            for _ in particle_objectives
        )
        particle_weight_source = "uniform_default"
    else:
        raw_particle_weights = tuple(float(value) for value in particle_weights)
        if len(raw_particle_weights) != len(particle_objectives):
            raise ValueError(
                "particle_weights must have one value per particle objective."
            )
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in raw_particle_weights
        ):
            raise ValueError(
                "particle_weights must be finite and strictly positive."
            )
        total_particle_weight = sum(raw_particle_weights)
        normalized_particle_weights = tuple(
            value / total_particle_weight for value in raw_particle_weights
        )
        particle_weight_source = "externally_supplied_normalized"
    if not math.isfinite(igd_p) or igd_p < 1.0:
        raise ValueError("igd_p must be finite and at least 1.")
    p_min = float(declared_p_min)
    B_L = float(cellwise_mse_constant_B_L)
    delta = float(confidence_delta)
    if not math.isfinite(p_min) or p_min <= 0.0 or p_min > 1.0:
        raise ValueError("declared_p_min must lie in (0, 1].")
    if not math.isfinite(B_L) or B_L < 0.0:
        raise ValueError("cellwise_mse_constant_B_L must be finite and nonnegative.")
    if not math.isfinite(delta) or delta <= 0.0 or delta >= 1.0:
        raise ValueError("confidence_delta must lie in (0, 1).")
    if (declared_cellwise_error_radius is None) != (
        declared_error_failure_probability is None
    ):
        raise ValueError(
            "declared_cellwise_error_radius and "
            "declared_error_failure_probability must be supplied together."
        )
    external_radius: float | None = None
    external_failure: float | None = None
    if declared_cellwise_error_radius is not None:
        external_radius = float(declared_cellwise_error_radius)
        external_failure = float(declared_error_failure_probability)
        if not math.isfinite(external_radius) or external_radius < 0.0:
            raise ValueError(
                "declared_cellwise_error_radius must be finite and nonnegative."
            )
        if (
            not math.isfinite(external_failure)
            or external_failure < 0.0
            or external_failure >= 1.0
        ):
            raise ValueError(
                "declared_error_failure_probability must lie in [0, 1)."
            )

    feasible_raw = tuple(
        _coerce_point(point, dimension, f"feasible_objectives[{index}]")
        for index, point in enumerate(feasible_objectives)
    )
    particles_raw = tuple(
        _coerce_point(point, dimension, f"particle_objectives[{index}]")
        for index, point in enumerate(particle_objectives)
    )
    feasible_normalized = _normalize_points(
        feasible_raw,
        lower=lower,
        upper=upper,
        label="feasible_objectives",
    )
    particles_normalized = _normalize_points(
        particles_raw,
        lower=lower,
        upper=upper,
        label="particle_objectives",
    )
    feasible_set = set(feasible_raw)
    missing_particles = [
        index for index, point in enumerate(particles_raw) if point not in feasible_set
    ]
    if missing_particles:
        raise ValueError(
            "Every particle objective must occur in the enumerated feasible set; "
            f"missing particle indices: {missing_particles}."
        )

    pareto_normalized = nondominated_points(feasible_normalized)
    pareto_raw = nondominated_points(feasible_raw)
    pareto_cells = tuple(
        sorted(
            {
                normalized_cell_index(point, widths)
                for point in pareto_normalized
            }
        )
    )
    supplied_probabilities: dict[Cell, float] = {}
    for raw_cell, raw_probability in target_pareto_cell_probabilities.items():
        if len(raw_cell) != dimension:
            raise ValueError(
                "Every target Pareto-cell key must match the objective dimension."
            )
        converted: list[int] = []
        for raw_index in raw_cell:
            numeric_index = float(raw_index)
            if (
                not math.isfinite(numeric_index)
                or numeric_index < 0.0
                or not numeric_index.is_integer()
            ):
                raise ValueError(
                    "Target Pareto-cell indices must be nonnegative integers."
                )
            converted.append(int(numeric_index))
        cell = tuple(converted)
        if cell in supplied_probabilities:
            raise ValueError("Duplicate target Pareto-cell key after normalization.")
        supplied_probabilities[cell] = float(raw_probability)
    if set(supplied_probabilities) != set(pareto_cells):
        missing = sorted(set(pareto_cells) - set(supplied_probabilities))
        extra = sorted(set(supplied_probabilities) - set(pareto_cells))
        raise ValueError(
            "Target probabilities must be declared for exactly the frozen Pareto "
            f"cells; missing={missing}, extra={extra}."
        )
    if any(
        not math.isfinite(probability) or probability <= 0.0
        for probability in supplied_probabilities.values()
    ):
        raise ValueError("Every target Pareto-cell probability must be positive.")
    if sum(supplied_probabilities.values()) > 1.0 + 1e-12:
        raise ValueError("Declared Pareto-cell probabilities cannot sum above one.")
    actual_p_min = min(supplied_probabilities.values())
    if p_min > actual_p_min:
        raise ValueError(
            "declared_p_min exceeds an externally declared Pareto-cell mass."
        )

    particle_cells = tuple(
        normalized_cell_index(point, widths) for point in particles_normalized
    )
    counts = Counter(particle_cells)
    M = len(particles_normalized)
    empirical_masses = {
        cell: sum(
            weight
            for particle_cell, weight in zip(
                particle_cells,
                normalized_particle_weights,
            )
            if particle_cell == cell
        )
        for cell in pareto_cells
    }
    cellwise_errors = {
        cell: abs(empirical_masses[cell] - supplied_probabilities[cell])
        for cell in pareto_cells
    }
    all_cells_hit = all(counts[cell] > 0 for cell in pareto_cells)

    point_cells = tuple(
        normalized_cell_index(point, widths) for point in pareto_normalized
    )
    support_additive_verified = all(
        any(
            particle_cell == pareto_cell
            and all(
                particle_value <= pareto_value + width
                for particle_value, pareto_value, width in zip(
                    particle,
                    pareto_point,
                    widths,
                )
            )
            for particle, particle_cell in zip(
                particles_normalized,
                particle_cells,
            )
        )
        for pareto_point, pareto_cell in zip(pareto_normalized, point_cells)
    )
    archive_normalized = nondominated_points(particles_normalized)
    archive_raw = nondominated_points(particles_raw)
    archive_additive_verified = all(
        any(
            all(
                archive_value <= pareto_value + width
                for archive_value, pareto_value, width in zip(
                    archive_point,
                    pareto_point,
                    widths,
                )
            )
            for archive_point in archive_normalized
        )
        for pareto_point in pareto_normalized
    )

    spans = tuple(hi - lo for lo, hi in zip(lower, upper))
    additive_original = tuple(span * width for span, width in zip(spans, widths))
    support_ordinary_igd_normalized = _igd_p(
        particles_normalized,
        pareto_normalized,
        igd_p,
    )
    support_ordinary_igd_original = _igd_p(particles_raw, pareto_raw, igd_p)
    archive_ordinary_igd_normalized = _igd_p(
        archive_normalized,
        pareto_normalized,
        igd_p,
    )
    archive_ordinary_igd_original = _igd_p(archive_raw, pareto_raw, igd_p)
    archive_igd_plus_normalized = _igd_plus_p(
        archive_normalized,
        pareto_normalized,
        igd_p,
    )
    archive_igd_plus_original = _igd_plus_p(archive_raw, pareto_raw, igd_p)
    igd_bound_normalized = _lp_norm(widths, igd_p)
    igd_bound_original = _lp_norm(additive_original, igd_p)

    normalized_reference = (
        tuple(1.0 for _ in range(dimension))
        if normalized_hv_reference is None
        else _coerce_point(
            normalized_hv_reference,
            dimension,
            "normalized_hv_reference",
        )
    )
    if any(value < 1.0 for value in normalized_reference):
        raise ValueError(
            "The fixed normalized hypervolume reference must be componentwise "
            "at or above the frozen box upper boundary."
        )
    original_reference = tuple(
        lo + ref * span
        for lo, ref, span in zip(lower, normalized_reference, spans)
    )
    normalized_lengths = normalized_reference
    original_lengths = tuple(
        ref - lo for ref, lo in zip(original_reference, lower)
    )
    hv_bound_normalized = min(
        _product(normalized_lengths),
        sum(
            widths[i]
            * _product(
                tuple(
                    normalized_lengths[j]
                    for j in range(dimension)
                    if j != i
                )
            )
            for i in range(dimension)
        ),
    )
    hv_bound_original = min(
        _product(original_lengths),
        sum(
            additive_original[i]
            * _product(
                tuple(
                    original_lengths[j]
                    for j in range(dimension)
                    if j != i
                )
            )
            for i in range(dimension)
        ),
    )
    actual_hv_deficit_normalized = max(
        0.0,
        hypervolume_minimization(pareto_normalized, normalized_reference)
        - hypervolume_minimization(archive_normalized, normalized_reference),
    )
    actual_hv_deficit_original = max(
        0.0,
        hypervolume_minimization(pareto_raw, original_reference)
        - hypervolume_minimization(archive_raw, original_reference),
    )

    K = len(pareto_cells)
    direct_empty_cell_failure_bound = min(
        1.0,
        sum(
            B_L / (M * probability * probability)
            for probability in supplied_probabilities.values()
        ),
    )
    direct_empty_cell_failure_bound_coarse = min(
        1.0,
        K * B_L / (M * p_min * p_min),
    )
    # This is the deliberately conservative route from a uniform cellwise MSE
    # premise to a simultaneous sup-cell error event.  Choosing t=p_min/2
    # leaves a strict margin before the cell-hit threshold p_min.
    mse_error_threshold = p_min / 2.0
    mse_failure_bound = min(
        1.0,
        K * B_L / (M * mse_error_threshold * mse_error_threshold),
    )
    mse_confidence_gate = mse_failure_bound <= delta
    external_radius_gate = (
        external_radius is not None
        and external_failure is not None
        and external_radius < p_min
        and external_failure <= delta
    )
    selected_failure_bound = min(
        mse_failure_bound,
        (
            external_failure
            if external_radius is not None
            and external_failure is not None
            and external_radius < p_min
            else 1.0
        ),
    )
    confidence_gate = selected_failure_bound <= delta
    observed_error = max(cellwise_errors.values())
    observed_error_implies_hits = observed_error < p_min
    box_diameter_normalized = _lp_norm(tuple(1.0 for _ in range(dimension)), igd_p)
    box_diameter_original = _lp_norm(spans, igd_p)
    box_hv_normalized = _product(normalized_lengths)
    box_hv_original = _product(original_lengths)

    geometry_consistent = (
        support_additive_verified
        and archive_additive_verified
        and support_ordinary_igd_normalized <= igd_bound_normalized + 1e-12
        and support_ordinary_igd_original <= igd_bound_original + 1e-12
        and archive_igd_plus_normalized <= igd_bound_normalized + 1e-12
        and archive_igd_plus_original <= igd_bound_original + 1e-12
        and actual_hv_deficit_normalized <= hv_bound_normalized + 1e-12
        and actual_hv_deficit_original <= hv_bound_original + 1e-9
    )
    if not all_cells_hit:
        verdict = "FAIL_OBSERVED_COVERAGE"
    elif not geometry_consistent:
        verdict = "FAIL_GEOMETRY_CHECK"
    elif not confidence_gate:
        verdict = "FAIL_PARTICLE_ERROR_GATE"
    else:
        verdict = "PASS"
    return {
        "schema": SCHEMA,
        "supersedes_schemas": sorted(LEGACY_SUPERSEDED_SCHEMAS),
        "metric_semantics": {
            "ordinary_igd": (
                "arithmetic_mean_over_reference_points_of_nearest_local_lp_distance"
            ),
            "igd_plus": (
                "arithmetic_mean_over_reference_points_of_nearest_local_lp_"
                "positive_part_distance"
            ),
            "p_scope": "local_objective_space_norm_only",
        },
        "verdict": verdict,
        "implication_chain": [
            {
                "step": "finite_particle_error_to_cell_hits",
                "gate": confidence_gate,
                "failure_probability_bound": selected_failure_bound,
            },
            {
                "step": "observed_cell_hits_to_epsilon_coverage",
                "gate": all_cells_hit and support_additive_verified,
            },
            {
                "step": "epsilon_coverage_to_igd_hv_bounds",
                "gate": geometry_consistent,
            },
        ],
        "claim_scope": (
            "finite enumerated feasible set; frozen exogenous box/cells/target; "
            "conditional on the externally proved cellwise MSE assumption"
        ),
        "box": {
            "objective_lower_original": list(lower),
            "objective_upper_original": list(upper),
            "objective_spans_original": list(spans),
            "normalized_lower": [0.0] * dimension,
            "normalized_upper": [1.0] * dimension,
            "normalized_cell_widths": list(widths),
            "cell_convention": (
                "half-open [k*h,(k+1)*h) in each coordinate; only the global "
                "normalized upper boundary 1 is closed"
            ),
        },
        "pareto": {
            "num_feasible_points": len(feasible_raw),
            "num_unique_pareto_points": len(pareto_raw),
            "points_original": [list(point) for point in pareto_raw],
            "points_normalized": [list(point) for point in pareto_normalized],
            "num_cells": K,
            "cells": [list(cell) for cell in pareto_cells],
            "target_cell_probabilities": {
                _cell_key(cell): supplied_probabilities[cell]
                for cell in pareto_cells
            },
            "declared_p_min": p_min,
            "minimum_supplied_cell_probability": actual_p_min,
        },
        "finite_particle_assumption": {
            "form": (
                "for each Pareto cell C: "
                "E[(eta_L^M(C)-eta_L(C))^2] <= B_L/M"
            ),
            "B_L": B_L,
            "M": M,
            "confidence_delta": delta,
            "not_estimated_from_ess": True,
            "mse_route": {
                "error_threshold_t": mse_error_threshold,
                "threshold_choice": "t = declared_p_min / 2",
                "union_chebyshev_failure_bound_K_B_L_over_M_t_sq": (
                    mse_failure_bound
                ),
                "mse_requested_confidence_gate": mse_confidence_gate,
            },
            "mse_requested_confidence_gate": mse_confidence_gate,
            "direct_empty_cell_failure_bound": direct_empty_cell_failure_bound,
            "direct_empty_cell_failure_bound_using_p_min": (
                direct_empty_cell_failure_bound_coarse
            ),
            "external_radius_route": {
                "declared_cellwise_error_radius": external_radius,
                "declared_error_failure_probability": external_failure,
                "radius_strictly_below_p_min": (
                    external_radius is not None and external_radius < p_min
                ),
                "requested_confidence_gate": external_radius_gate,
            },
            "external_radius_gate": external_radius_gate,
            "selected_failure_probability_bound": selected_failure_bound,
            "requested_confidence_gate": confidence_gate,
            "coverage_probability_lower_bound": 1.0 - selected_failure_bound,
        },
        "observed": {
            "particle_count": M,
            "particle_weight_source": particle_weight_source,
            "particle_weight_min": min(normalized_particle_weights),
            "particle_weight_max": max(normalized_particle_weights),
            "particle_weights_sum": sum(normalized_particle_weights),
            "pareto_cell_counts": {
                _cell_key(cell): counts[cell] for cell in pareto_cells
            },
            "pareto_cell_empirical_masses": {
                _cell_key(cell): empirical_masses[cell] for cell in pareto_cells
            },
            "pareto_cell_absolute_errors": {
                _cell_key(cell): cellwise_errors[cell] for cell in pareto_cells
            },
            "maximum_cellwise_absolute_error": observed_error,
            "observed_error_strictly_below_declared_p_min": observed_error_implies_hits,
            "all_pareto_cells_hit": all_cells_hit,
            "additive_componentwise_coverage_verified": (
                support_additive_verified
            ),
            "support_additive_componentwise_coverage_verified": (
                support_additive_verified
            ),
            "archive_additive_componentwise_coverage_verified": (
                archive_additive_verified
            ),
            "archive_size": len(archive_normalized),
            "support_ordinary_igd_p_normalized": (
                support_ordinary_igd_normalized
            ),
            "support_ordinary_igd_p_original": support_ordinary_igd_original,
            "archive_ordinary_igd_p_normalized": (
                archive_ordinary_igd_normalized
            ),
            "archive_ordinary_igd_p_original": archive_ordinary_igd_original,
            "archive_igd_plus_p_normalized": archive_igd_plus_normalized,
            "archive_igd_plus_p_original": archive_igd_plus_original,
            "igd_p_normalized": support_ordinary_igd_normalized,
            "igd_p_original": support_ordinary_igd_original,
            "hv_deficit_normalized": actual_hv_deficit_normalized,
            "hv_deficit_original": actual_hv_deficit_original,
            "geometry_bound_check_passed_on_hit_event": geometry_consistent,
        },
        "geometry_bounds": {
            "igd_p": igd_p,
            "additive_widths_normalized": list(widths),
            "additive_widths_original": list(additive_original),
            "igd_p_normalized": igd_bound_normalized,
            "igd_p_original": igd_bound_original,
            "normalized_hv_reference": list(normalized_reference),
            "original_hv_reference": list(original_reference),
            "hv_side_lengths_normalized": list(normalized_lengths),
            "hv_side_lengths_original": list(original_lengths),
            "hv_deficit_formula": "sum_i h_i * product_{j != i} L_j",
            "hv_deficit_normalized": hv_bound_normalized,
            "hv_deficit_original": hv_bound_original,
        },
        "high_probability_bounds": {
            "probability_at_least": 1.0 - selected_failure_bound,
            "requested_probability_at_least": 1.0 - delta,
            "requested_probability_certified": confidence_gate,
            "on_that_event": {
                "all_pareto_cells_hit": True,
                "support_additive_componentwise_widths_normalized": list(
                    widths
                ),
                "support_additive_componentwise_widths_original": list(
                    additive_original
                ),
                "archive_additive_componentwise_widths_normalized": list(
                    widths
                ),
                "archive_additive_componentwise_widths_original": list(
                    additive_original
                ),
                "support_ordinary_igd_p_normalized_at_most": (
                    igd_bound_normalized
                ),
                "support_ordinary_igd_p_original_at_most": igd_bound_original,
                "archive_igd_plus_p_normalized_at_most": igd_bound_normalized,
                "archive_igd_plus_p_original_at_most": igd_bound_original,
                "hv_deficit_normalized_at_most": hv_bound_normalized,
                "hv_deficit_original_at_most": hv_bound_original,
            },
        },
        "expectation_bounds": {
            "failure_probability_used": selected_failure_bound,
            "box_diameter_normalized": box_diameter_normalized,
            "box_diameter_original": box_diameter_original,
            "box_hv_volume_normalized": box_hv_normalized,
            "box_hv_volume_original": box_hv_original,
            "igd_p_normalized_simple": (
                igd_bound_normalized
                + selected_failure_bound * box_diameter_normalized
            ),
            "igd_p_original_simple": (
                igd_bound_original
                + selected_failure_bound * box_diameter_original
            ),
            "hv_deficit_normalized_simple": (
                hv_bound_normalized
                + selected_failure_bound * box_hv_normalized
            ),
            "hv_deficit_original_simple": (
                hv_bound_original
                + selected_failure_bound * box_hv_original
            ),
            "support_ordinary_igd_p_normalized_simple": (
                igd_bound_normalized
                + selected_failure_bound * box_diameter_normalized
            ),
            "support_ordinary_igd_p_original_simple": (
                igd_bound_original
                + selected_failure_bound * box_diameter_original
            ),
            "archive_igd_plus_p_normalized_simple": (
                igd_bound_normalized
                + selected_failure_bound * box_diameter_normalized
            ),
            "archive_igd_plus_p_original_simple": (
                igd_bound_original
                + selected_failure_bound * box_diameter_original
            ),
            "derivation": (
                "split on the all-Pareto-cells-hit event and add the failure "
                "probability times the frozen-box diameter or HV volume"
            ),
            "igd_scope": (
                "ordinary IGD_p for terminal particle support; IGD+_p for the "
                "nondominated archive"
            ),
        },
        "claim_limit": (
            "This is one certificate at the supplied finite M and declared "
            "inputs, not a continuous interval claim and not evidence that "
            "p_min or B_L follows from ESS. Nondominated filtering preserves "
            "additive coverage, IGD+ and HV, but not an ordinary archive-IGD "
            "bound."
        ),
    }


def shifted_front_hv_deficit_bound(
    pareto_points: Sequence[Sequence[float]],
    *,
    additive_widths: Sequence[float],
    reference: Sequence[float],
) -> float:
    """Exact shifted-front upper bound for additive Pareto coverage.

    If an approximation contains, for every ``p`` in the supplied full Pareto
    front, a feasible point ``a <= p + h``, then its fixed-reference HV deficit
    is at most ``HV(P) - HV(min(reference, P+h))``.  This is never larger than
    the global box-slab bound and is usually substantially sharper.
    """

    ref = tuple(float(value) for value in reference)
    widths = tuple(float(value) for value in additive_widths)
    if len(ref) != len(widths) or not ref:
        raise ValueError("reference and additive_widths must have the same positive dimension.")
    if any(not math.isfinite(value) for value in ref + widths):
        raise ValueError("reference and additive widths must be finite.")
    if any(width < 0.0 for width in widths):
        raise ValueError("additive widths must be nonnegative.")
    points = tuple(
        _coerce_point(point, len(ref), f"pareto_points[{index}]")
        for index, point in enumerate(pareto_points)
    )
    if not points:
        raise ValueError("pareto_points cannot be empty.")
    shifted = tuple(
        tuple(min(limit, value + width) for value, width, limit in zip(point, widths, ref))
        for point in points
    )
    return max(
        0.0,
        hypervolume_minimization(points, ref)
        - hypervolume_minimization(shifted, ref),
    )


def certify_independent_cell_probe_bounds(
    feasible_objectives: Sequence[Sequence[float]],
    probe_objectives: Sequence[Sequence[float]],
    *,
    declared_cells: Sequence[Sequence[int]],
    objective_lower: Sequence[float],
    objective_upper: Sequence[float],
    cell_widths_original: Sequence[float],
    source_bound_failure_probability: float,
    requested_confidence_delta: float,
    igd_p: float = 2.0,
    hv_reference: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Audit the explicit independent-probe coverage and geometry theorem."""

    if not objective_lower:
        raise ValueError("The objective box must have positive dimension.")
    dimension = len(objective_lower)
    lower = _coerce_point(objective_lower, dimension, "objective_lower")
    upper = _coerce_point(objective_upper, dimension, "objective_upper")
    widths_original = _coerce_point(
        cell_widths_original,
        dimension,
        "cell_widths_original",
    )
    if any(high <= low for low, high in zip(lower, upper)):
        raise ValueError("Every objective upper bound must exceed its lower bound.")
    if any(width <= 0.0 or width > high - low for width, low, high in zip(widths_original, lower, upper)):
        raise ValueError("Each original-unit width must lie in (0, box span].")
    reference = (
        upper
        if hv_reference is None
        else _coerce_point(hv_reference, dimension, "hv_reference")
    )
    if any(
        reference_value < upper_value
        for reference_value, upper_value in zip(reference, upper)
    ):
        raise ValueError(
            "hv_reference must be coordinatewise no better than objective_upper."
        )
    failure = float(source_bound_failure_probability)
    delta = float(requested_confidence_delta)
    if not math.isfinite(failure) or not (0.0 <= failure <= 1.0):
        raise ValueError("source_bound_failure_probability must lie in [0, 1].")
    if not math.isfinite(delta) or not (0.0 < delta < 1.0):
        raise ValueError("requested_confidence_delta must lie in (0, 1).")
    feasible_raw = tuple(
        _coerce_point(point, dimension, f"feasible_objectives[{index}]")
        for index, point in enumerate(feasible_objectives)
    )
    probes_raw = tuple(
        _coerce_point(point, dimension, f"probe_objectives[{index}]")
        for index, point in enumerate(probe_objectives)
    )
    if not feasible_raw or not probes_raw:
        raise ValueError("Feasible and probe objective sets must be nonempty.")
    normalized_widths = tuple(
        width / (high - low)
        for width, low, high in zip(widths_original, lower, upper)
    )
    cell_counts = tuple(
        max(1, int(math.ceil((high - low) / width)))
        for low, high, width in zip(lower, upper, widths_original)
    )
    pareto_raw = nondominated_points(feasible_raw)
    # The metric box is allowed to be tighter than the target safety box.  Its
    # source-bound obligation is to contain the Pareto/reference set used by
    # the metric theorem, not every dominated feasible state.
    pareto_normalized = _normalize_points(
        pareto_raw,
        lower=lower,
        upper=upper,
        label="pareto_objectives",
    )
    pareto_cells = tuple(
        sorted(
            {
                normalized_cell_index(point, normalized_widths)
                for point in pareto_normalized
            }
        )
    )
    declared_values: list[Cell] = []
    for index, cell in enumerate(declared_cells):
        if len(cell) != dimension or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in cell
        ):
            raise ValueError(
                f"declared_cells[{index}] must contain {dimension} "
                "nonnegative integers."
            )
        declared_cell = tuple(cell)
        if any(
            coordinate >= count
            for coordinate, count in zip(declared_cell, cell_counts)
        ):
            raise ValueError(
                f"declared_cells[{index}] lies outside the objective grid."
            )
        declared_values.append(declared_cell)
    declared = tuple(sorted(set(declared_values)))
    completeness = set(pareto_cells).issubset(declared)
    probe_cells: tuple[Cell | None, ...] = tuple(
        (
            normalized_cell_index(
                tuple(
                    (value - low) / (high - low)
                    for value, low, high in zip(point, lower, upper)
                ),
                normalized_widths,
            )
            if all(
                low <= value <= high
                for value, low, high in zip(point, lower, upper)
            )
            else None
        )
        for point in probes_raw
    )
    counts = Counter(probe_cells)
    all_hit = all(counts[cell] > 0 for cell in pareto_cells)
    cell_representatives: list[Point] = []
    for cell in pareto_cells:
        for point, point_cell in zip(probes_raw, probe_cells):
            if point_cell == cell:
                cell_representatives.append(point)
                break
    archive_raw = nondominated_points(probes_raw)
    additive_verified = True
    for pareto_point in pareto_raw:
        if not any(
            all(a <= p + width + 1e-12 for a, p, width in zip(candidate, pareto_point, widths_original))
            for candidate in cell_representatives
        ):
            additive_verified = False
            break
    igd_support = _igd_p(cell_representatives, pareto_raw, igd_p)
    igd_plus_archive = _igd_plus_p(archive_raw, pareto_raw, igd_p)
    igd_bound = _lp_norm(widths_original, igd_p)
    pareto_hv = hypervolume_minimization(pareto_raw, reference)
    archive_hv = hypervolume_minimization(archive_raw, reference)
    actual_hv_deficit = max(0.0, pareto_hv - archive_hv)
    shifted_bound = shifted_front_hv_deficit_bound(
        pareto_raw,
        additive_widths=widths_original,
        reference=reference,
    )
    slab_bound = sum(
        width * _product(
            tuple(
                reference[other] - lower[other]
                for other in range(dimension)
                if other != coordinate
            )
        )
        for coordinate, width in enumerate(widths_original)
    )
    design_gate = bool(completeness and failure <= delta)
    geometry_gate = bool(
        all_hit
        and additive_verified
        and igd_support <= igd_bound + 1e-10
        and igd_plus_archive <= igd_bound + 1e-10
        and actual_hv_deficit <= shifted_bound + 1e-10
        and shifted_bound <= slab_bound + 1e-10
    )
    return {
        "schema": "pareto_independent_cell_probe_certificate_v1",
        "design_verdict": "PASS" if design_gate else "FAIL",
        "realized_geometry_verdict": "PASS" if geometry_gate else "FAIL",
        "probability_at_least": 1.0 - failure,
        "requested_probability_at_least": 1.0 - delta,
        "source_bound_failure_probability": failure,
        "cell_completeness": completeness,
        "true_pareto_cells": [list(cell) for cell in pareto_cells],
        "declared_cells": [list(cell) for cell in declared],
        "missing_pareto_cells": [
            list(cell) for cell in pareto_cells if cell not in set(declared)
        ],
        "observed_counts": {_cell_key(cell): counts[cell] for cell in pareto_cells},
        "probe_points_outside_metric_box": sum(
            cell is None for cell in probe_cells
        ),
        "all_true_pareto_cells_observed": all_hit,
        "additive_coverage_verified": additive_verified,
        "igd_p": igd_p,
        "igd_bound_original": igd_bound,
        "cell_representative_igd_original": igd_support,
        "nondominated_archive_igd_plus_original": igd_plus_archive,
        "actual_hv_deficit_original": actual_hv_deficit,
        "shifted_front_hv_deficit_bound_original": shifted_bound,
        "global_slab_hv_deficit_bound_original": slab_bound,
        "shifted_bound_not_worse_than_slab": shifted_bound <= slab_bound + 1e-10,
        "objective_lower": list(lower),
        "objective_upper": list(upper),
        "hv_reference": list(reference),
        "cell_widths_original": list(widths_original),
        "claim_scope": (
            "source-bound complete cell set; independent finite-step probe chains; "
            "original-unit additive widths; full finite Pareto front"
        ),
    }
