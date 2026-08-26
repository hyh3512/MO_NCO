from __future__ import annotations

"""Pilot-confirm metric certificates for fixed-schedule Pareto-SMC.

The reference front is a finite feasible objective set frozen independently of
both SMC streams.  The certificate is benchmark-relative: it does not assert
that the supplied reference front is the unknown complete Pareto front.
"""

import hashlib
import json
import math
from typing import Any, Mapping, Sequence, Tuple

from .contracts import ClaimLevel
from .pareto_bounds import (
    hypervolume_minimization,
    nondominated_points,
    shifted_front_hv_deficit_bound,
)
from .pareto_fk_certificate import make_contraction_aware_fk_plan
from .pareto_regeneration_certificate import (
    confirm_cell_certificate,
    pilot_target_mass_lower_bound,
    target_normalizer_lower_bound,
    terminal_residual_weight,
)
from .sampler import OptimizationResult
from .types import ObjectiveVector

Cell = Tuple[int, ...]


class FixedScheduleCertificateError(ValueError):
    """Raised when a pilot-confirm contract is not auditable."""


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise FixedScheduleCertificateError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return value


def _finite_probability(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not (0.0 < result < 1.0):
        raise FixedScheduleCertificateError(
            f"{label} must lie in (0, 1)."
        )
    return result


def _lp_norm(values: Sequence[float], p: float) -> float:
    if math.isinf(p):
        return max(values)
    return sum(value**p for value in values) ** (1.0 / p)


def _distance(
    reference: ObjectiveVector,
    candidate: ObjectiveVector,
    *,
    p: float,
    plus: bool,
) -> float:
    differences = tuple(
        max(0.0, candidate_value - reference_value)
        if plus
        else abs(candidate_value - reference_value)
        for reference_value, candidate_value in zip(reference, candidate)
    )
    return _lp_norm(differences, p)


def _igd(
    reference: Sequence[ObjectiveVector],
    approximation: Sequence[ObjectiveVector],
    *,
    p: float,
    plus: bool,
) -> float:
    if not reference or not approximation:
        raise FixedScheduleCertificateError(
            "Reference and approximation sets must be nonempty."
        )
    return sum(
        min(
            _distance(point, candidate, p=p, plus=plus)
            for candidate in approximation
        )
        for point in reference
    ) / len(reference)


def _cell_index(
    objective: ObjectiveVector,
    *,
    lower: ObjectiveVector,
    upper: ObjectiveVector,
    widths: ObjectiveVector,
    counts: Tuple[int, ...],
) -> Cell:
    cell = []
    for coordinate, (value, low, high, width, count) in enumerate(
        zip(objective, lower, upper, widths, counts)
    ):
        tolerance = 1e-12 * max(1.0, abs(low), abs(high))
        if value < low - tolerance or value > high + tolerance:
            raise FixedScheduleCertificateError(
                f"reference objective coordinate {coordinate} leaves the "
                "frozen objective box."
            )
        if value >= high - tolerance:
            cell.append(count - 1)
        else:
            cell.append(
                min(
                    count - 1,
                    max(0, int(math.floor((value - low) / width))),
                )
            )
    return tuple(cell)


def _metadata_tuple(
    metadata: Mapping[str, object],
    key: str,
) -> tuple[Any, ...]:
    value = metadata.get(key)
    if not isinstance(value, (list, tuple)):
        raise FixedScheduleCertificateError(
            f"metadata.{key} must be an array."
        )
    return tuple(value)


def _validate_fixed_schedule_pair(
    pilot: OptimizationResult,
    confirm: OptimizationResult,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    pilot_metadata = pilot.metadata
    confirm_metadata = confirm.metadata
    if not isinstance(pilot_metadata, dict) or not isinstance(
        confirm_metadata,
        dict,
    ):
        raise FixedScheduleCertificateError(
            "Both results must contain metadata dictionaries."
        )
    for label, metadata in (
        ("pilot", pilot_metadata),
        ("confirm", confirm_metadata),
    ):
        if metadata.get("resampling_policy") != "always":
            raise FixedScheduleCertificateError(
                f"{label} must resample at every positive stage."
            )
        if metadata.get("bootstrap_mutations_by_stage") is None:
            raise FixedScheduleCertificateError(
                f"{label} lacks fixed mutation counts."
            )
        if metadata.get(
            "contraction_aware_fixed_schedule_certificate"
        ) is None:
            raise FixedScheduleCertificateError(
                f"{label} lacks the contraction-aware certificate."
            )

    matched_keys = (
        "algorithm_contract",
        "instance_sha256",
        "context_hash",
        "reporting_context_hash",
        "objective_lower_bounds",
        "objective_upper_bounds",
        "epsilon",
        "epsilon_cell_counts",
        "reference_directions",
        "num_reference_types",
        "beta_schedule",
        "chebyshev_rho",
        "global_refresh_probability",
        "bootstrap_mutations_by_stage",
    )
    mismatched = [
        key
        for key in matched_keys
        if pilot_metadata.get(key) != confirm_metadata.get(key)
    ]
    if mismatched:
        raise FixedScheduleCertificateError(
            "Pilot and confirm contracts differ: "
            + ", ".join(mismatched)
        )
    if pilot_metadata.get("seed") == confirm_metadata.get("seed"):
        raise FixedScheduleCertificateError(
            "Pilot and confirm seeds must be distinct."
        )
    return pilot_metadata, confirm_metadata


def _terminal_support_sha256(
    result: OptimizationResult,
    *,
    label: str,
) -> str:
    if len(result.particles) != len(result.objectives):
        raise FixedScheduleCertificateError(
            f"{label} terminal particles and objectives have different sizes."
        )
    particles = tuple(
        tuple(int(city) for city in tour)
        for tour in result.particles
    )
    objectives = tuple(
        tuple(float(value) for value in objective)
        for objective in result.objectives
    )
    if any(
        not math.isfinite(value)
        for objective in objectives
        for value in objective
    ):
        raise FixedScheduleCertificateError(
            f"{label} terminal objectives must be finite."
        )
    return _payload_sha256(
        {
            "particles": particles,
            "objectives": objectives,
        }
    )


def _cell_mass_maps(
    result: OptimizationResult,
    *,
    label: str,
) -> Tuple[Mapping[Cell, float], ...]:
    metadata = result.metadata
    raw_groups = _metadata_tuple(
        metadata,
        "final_epsilon_cell_masses_by_reference",
    )
    expected_groups = int(metadata["num_reference_types"])
    if len(raw_groups) != expected_groups:
        raise FixedScheduleCertificateError(
            "The final cell-mass ledger has the wrong number of types."
        )
    groups = []
    for type_index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, (list, tuple)):
            raise FixedScheduleCertificateError(
                f"Cell-mass group {type_index} is invalid."
            )
        masses: dict[Cell, float] = {}
        for record in raw_group:
            if not isinstance(record, dict):
                raise FixedScheduleCertificateError(
                    "Every cell-mass record must be an object."
                )
            raw_cell = record.get("epsilon_cell")
            if not isinstance(raw_cell, (list, tuple)):
                raise FixedScheduleCertificateError(
                    "Every cell-mass record must contain a cell."
                )
            cell = tuple(raw_cell)
            mass = float(record.get("mass"))
            if (
                any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in cell
                )
                or not math.isfinite(mass)
                or mass <= 0.0
            ):
                raise FixedScheduleCertificateError(
                    "Cell-mass records must contain valid cells and positive masses."
                )
            if cell in masses:
                raise FixedScheduleCertificateError(
                    "Cell-mass ledgers must contain at most one record per cell."
                )
            masses[cell] = mass
        if not math.isclose(
            sum(masses.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise FixedScheduleCertificateError(
                f"Cell masses for type {type_index} do not sum to one."
            )
        groups.append(masses)
    claimed_groups = tuple(groups)

    particles_per_reference = int(metadata["particles_per_reference"])
    if particles_per_reference <= 0:
        raise FixedScheduleCertificateError(
            f"{label} particles_per_reference must be positive."
        )
    if len(result.objectives) != expected_groups * particles_per_reference:
        raise FixedScheduleCertificateError(
            f"{label} terminal objectives do not match the typed layout."
        )
    raw_weight_groups = _metadata_tuple(
        metadata,
        "final_normalized_weights_by_reference",
    )
    if len(raw_weight_groups) != expected_groups:
        raise FixedScheduleCertificateError(
            f"{label} terminal weights do not match the typed layout."
        )
    lower = tuple(float(value) for value in metadata["objective_lower_bounds"])
    upper = tuple(float(value) for value in metadata["objective_upper_bounds"])
    widths = tuple(float(value) for value in metadata["epsilon"])
    counts = tuple(int(value) for value in metadata["epsilon_cell_counts"])
    if not (
        lower
        and len(lower) == len(upper) == len(widths) == len(counts)
    ):
        raise FixedScheduleCertificateError(
            f"{label} objective-box metadata has inconsistent dimensions."
        )

    recomputed_groups = []
    for type_index in range(expected_groups):
        start = type_index * particles_per_reference
        stop = start + particles_per_reference
        raw_weights = raw_weight_groups[type_index]
        if (
            not isinstance(raw_weights, (list, tuple))
            or len(raw_weights) != particles_per_reference
        ):
            raise FixedScheduleCertificateError(
                f"{label} terminal weight group {type_index} has the "
                "wrong size."
            )
        weights = tuple(float(value) for value in raw_weights)
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in weights
        ):
            raise FixedScheduleCertificateError(
                f"{label} terminal weights must be finite and positive."
            )
        recomputed: dict[Cell, float] = {}
        for objective, weight in zip(
            result.objectives[start:stop],
            weights,
        ):
            cell = _cell_index(
                tuple(float(value) for value in objective),
                lower=lower,
                upper=upper,
                widths=widths,
                counts=counts,
            )
            recomputed[cell] = recomputed.get(cell, 0.0) + weight
        if claimed_groups[type_index] != recomputed:
            raise FixedScheduleCertificateError(
                f"{label} terminal cell-mass ledger does not match the "
                f"terminal objective payload for type {type_index}."
            )
        recomputed_groups.append(recomputed)
    return tuple(recomputed_groups)


def _validate_uniform_terminal_weights(
    metadata: Mapping[str, object],
    *,
    label: str,
) -> None:
    raw_groups = _metadata_tuple(
        metadata,
        "final_normalized_weights_by_reference",
    )
    expected_groups = int(metadata["num_reference_types"])
    particles_per_reference = int(metadata["particles_per_reference"])
    if len(raw_groups) != expected_groups:
        raise FixedScheduleCertificateError(
            f"{label} terminal weights have the wrong number of types."
        )
    target_weight = 1.0 / particles_per_reference
    for type_index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, (list, tuple)) or len(
            raw_group
        ) != particles_per_reference:
            raise FixedScheduleCertificateError(
                f"{label} terminal weight group {type_index} has the wrong size."
            )
        weights = tuple(float(value) for value in raw_group)
        if any(
            not math.isfinite(value)
            or not math.isclose(
                value,
                target_weight,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for value in weights
        ):
            raise FixedScheduleCertificateError(
                f"{label} terminal weights are not uniform after fixed resampling."
            )


def certify_fixed_schedule_reference_metrics(
    pilot: OptimizationResult,
    confirm: OptimizationResult,
    *,
    reference_objectives: Sequence[Sequence[float]],
    reference_source: str,
    reference_artifact_sha256: str,
    pilot_failure_budget: float,
    confirm_failure_budget: float,
    igd_p: float,
    hv_reference: Sequence[float],
    max_igd_bound: float,
    max_hv_deficit_bound: float,
    certificate_mode: str = "published",
) -> dict[str, object]:
    """Certify practical metrics for a frozen feasible reference front.

    The pilot stream is simultaneous over every type-cell pair.  It selects a
    type for each frozen reference cell using lower confidence bounds.  The
    independent confirm stream then supplies the cell hits.  The design gate
    never depends on whether those hits happened in the realized confirm run.
    """

    pilot_metadata, confirm_metadata = _validate_fixed_schedule_pair(
        pilot,
        confirm,
    )
    _validate_uniform_terminal_weights(
        pilot_metadata,
        label="pilot",
    )
    _validate_uniform_terminal_weights(
        confirm_metadata,
        label="confirm",
    )
    reference_hash = _validate_sha256(
        reference_artifact_sha256,
        "reference_artifact_sha256",
    )
    allowed_sources = {
        "independent_exact_solver",
        "frozen_external_archive",
        "public_benchmark_reference",
    }
    if reference_source not in allowed_sources:
        raise FixedScheduleCertificateError(
            "reference_source is not independent of the certified streams."
        )
    delta_pilot = _finite_probability(
        pilot_failure_budget,
        "pilot_failure_budget",
    )
    delta_confirm = _finite_probability(
        confirm_failure_budget,
        "confirm_failure_budget",
    )
    if delta_pilot + delta_confirm >= 1.0:
        raise FixedScheduleCertificateError(
            "The pilot and confirm failure budgets must sum to less than one."
        )
    allowed_certificate_modes = {
        "published",
        "regeneration",
        "published_or_regeneration",
    }
    if certificate_mode not in allowed_certificate_modes:
        raise FixedScheduleCertificateError(
            "certificate_mode must be one of: "
            + ", ".join(sorted(allowed_certificate_modes))
        )
    # An OR rule selects between two theorem families after the pilot. Split
    # both familywise budgets before observing the pilot so the aggregate
    # false-PASS bound remains delta_pilot + delta_confirm.
    theorem_family_count = (
        2 if certificate_mode == "published_or_regeneration" else 1
    )
    published_delta_pilot = delta_pilot / theorem_family_count
    published_delta_confirm = delta_confirm / theorem_family_count
    regeneration_delta_pilot = delta_pilot / theorem_family_count
    regeneration_delta_confirm = delta_confirm / theorem_family_count

    lower = tuple(
        float(value)
        for value in _metadata_tuple(
            pilot_metadata,
            "objective_lower_bounds",
        )
    )
    upper = tuple(
        float(value)
        for value in _metadata_tuple(
            pilot_metadata,
            "objective_upper_bounds",
        )
    )
    widths = tuple(
        float(value)
        for value in _metadata_tuple(pilot_metadata, "epsilon")
    )
    counts = tuple(
        int(value)
        for value in _metadata_tuple(
            pilot_metadata,
            "epsilon_cell_counts",
        )
    )
    dimension = len(lower)
    if dimension == 0 or not (
        len(upper) == len(widths) == len(counts) == dimension
    ):
        raise FixedScheduleCertificateError(
            "The frozen objective grid has inconsistent dimensions."
        )

    reference = tuple(
        tuple(float(value) for value in point)
        for point in reference_objectives
    )
    if not reference or any(
        len(point) != dimension
        or any(not math.isfinite(value) for value in point)
        for point in reference
    ):
        raise FixedScheduleCertificateError(
            "reference_objectives must be a nonempty finite objective set."
        )
    if len(set(reference)) != len(reference):
        raise FixedScheduleCertificateError(
            "reference_objectives must be unique."
        )
    if set(nondominated_points(reference)) != set(reference):
        raise FixedScheduleCertificateError(
            "reference_objectives must be mutually nondominated."
        )
    canonical_reference = tuple(sorted(reference))
    reference_payload_hash = _payload_sha256(canonical_reference)
    if reference_hash != reference_payload_hash:
        raise FixedScheduleCertificateError(
            "reference_artifact_sha256 must hash the canonical sorted "
            "reference-objective payload."
        )
    reference_cells = tuple(
        sorted(
            {
                _cell_index(
                    point,
                    lower=lower,
                    upper=upper,
                    widths=widths,
                    counts=counts,
                )
                for point in canonical_reference
            }
        )
    )

    p_value = float(igd_p)
    if not (
        math.isinf(p_value)
        or (math.isfinite(p_value) and p_value >= 1.0)
    ):
        raise FixedScheduleCertificateError("igd_p must lie in [1, infinity].")
    hv_ref = tuple(float(value) for value in hv_reference)
    if len(hv_ref) != dimension or any(
        not math.isfinite(reference_value)
        or reference_value < upper_value
        for reference_value, upper_value in zip(hv_ref, upper)
    ):
        raise FixedScheduleCertificateError(
            "hv_reference must be finite and no better than the objective-box upper endpoint."
        )
    igd_tolerance = float(max_igd_bound)
    hv_tolerance = float(max_hv_deficit_bound)
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (igd_tolerance, hv_tolerance)
    ):
        raise FixedScheduleCertificateError(
            "Metric tolerances must be finite and nonnegative."
        )

    reference_type_count = int(pilot_metadata["num_reference_types"])
    pilot_masses = _cell_mass_maps(pilot, label="pilot")
    confirm_masses = _cell_mass_maps(confirm, label="confirm")
    beta_schedule = tuple(
        float(value)
        for value in _metadata_tuple(pilot_metadata, "beta_schedule")
    )
    mutation_steps = tuple(
        int(value)
        for value in _metadata_tuple(
            pilot_metadata,
            "bootstrap_mutations_by_stage",
        )
    )
    potential_upper = 1.0 + float(pilot_metadata["chebyshev_rho"])
    gamma = float(pilot_metadata["global_refresh_probability"])
    pilot_plan = make_contraction_aware_fk_plan(
        beta_schedule,
        potential_upper_bound=potential_upper,
        global_refresh_probability=gamma,
        mutation_steps_by_stage=mutation_steps,
        particle_count=int(pilot_metadata["particles_per_reference"]),
        observable_count=reference_type_count * len(reference_cells),
        failure_budget=published_delta_pilot,
    )
    confirm_plan = make_contraction_aware_fk_plan(
        beta_schedule,
        potential_upper_bound=potential_upper,
        global_refresh_probability=gamma,
        mutation_steps_by_stage=mutation_steps,
        particle_count=int(confirm_metadata["particles_per_reference"]),
        observable_count=len(reference_cells),
        failure_budget=published_delta_confirm,
    )
    radius_pilot = pilot_plan.simultaneous_error_radius
    radius_confirm = confirm_plan.simultaneous_error_radius
    published_concentration_gate = bool(
        pilot_plan.published_concentration_gate
        and confirm_plan.published_concentration_gate
    )

    # Direct final-stage target-regeneration certificate.  The published
    # empirical-measure theorem remains an independently selectable path.
    final_normalizer_lower_bound = target_normalizer_lower_bound(
        beta_schedule[-1],
        potential_upper,
    )
    final_residual_weight = terminal_residual_weight(
        global_refresh_probability=gamma,
        normalizer_lower_bound=final_normalizer_lower_bound,
        mutation_steps=mutation_steps[-1],
    )
    pilot_pair_failure_budget = (
        regeneration_delta_pilot
        / (reference_type_count * len(reference_cells))
    )
    regeneration_assignments = []
    for cell in reference_cells:
        candidates = []
        for type_index in range(reference_type_count):
            pilot_mass = pilot_masses[type_index].get(cell, 0.0)
            pilot_certificate = pilot_target_mass_lower_bound(
                empirical_terminal_mass=pilot_mass,
                pilot_particles=int(
                    pilot_metadata["particles_per_reference"]
                ),
                pilot_failure_budget=pilot_pair_failure_budget,
                pilot_residual_weight=final_residual_weight,
            )
            candidates.append(
                (
                    pilot_certificate.target_mass_lower_bound,
                    pilot_mass,
                    -type_index,
                    type_index,
                    pilot_certificate,
                )
            )
        (
            target_mass_lower,
            pilot_mass,
            _,
            selected_type,
            pilot_certificate,
        ) = max(candidates)
        confirm_certificate = confirm_cell_certificate(
            target_mass_lower_bound=target_mass_lower,
            confirm_particles=int(
                confirm_metadata["particles_per_reference"]
            ),
            confirm_residual_weight=final_residual_weight,
        )
        confirm_mass = confirm_masses[selected_type].get(cell, 0.0)
        regeneration_assignments.append(
            {
                "cell": cell,
                "selected_reference_type": selected_type,
                "pilot_empirical_mass": pilot_mass,
                "pilot_hoeffding_radius": (
                    pilot_certificate.pilot_hoeffding_radius
                ),
                "pilot_target_mass_lower_bound": target_mass_lower,
                "final_target_normalizer_lower_bound": (
                    final_normalizer_lower_bound
                ),
                "final_residual_weight": final_residual_weight,
                "confirm_per_particle_hit_lower_bound": (
                    confirm_certificate.per_particle_hit_lower_bound
                ),
                "confirm_cell_miss_probability_upper_bound": (
                    confirm_certificate.cell_miss_probability_upper_bound
                ),
                "confirm_empirical_mass": confirm_mass,
                "observed_confirm_hit": confirm_mass > 0.0,
            }
        )
    regeneration_confirm_miss_upper_bound = min(
        1.0,
        math.nextafter(
            math.fsum(
                float(
                    record[
                        "confirm_cell_miss_probability_upper_bound"
                    ]
                )
                for record in regeneration_assignments
            ),
            math.inf,
        ),
    )
    regeneration_hit_design_gate = bool(
        all(
            float(record["pilot_target_mass_lower_bound"]) > 0.0
            for record in regeneration_assignments
        )
        and regeneration_confirm_miss_upper_bound
        <= regeneration_delta_confirm
    )

    assignments = []
    for cell in reference_cells:
        candidates = []
        for type_index in range(reference_type_count):
            pilot_mass = pilot_masses[type_index].get(cell, 0.0)
            lower_bound = max(0.0, pilot_mass - radius_pilot)
            candidates.append(
                (lower_bound, pilot_mass, -type_index, type_index)
            )
        lower_bound, pilot_mass, _, selected_type = max(candidates)
        confirm_mass = confirm_masses[selected_type].get(cell, 0.0)
        margin = lower_bound - radius_confirm
        assignments.append(
            {
                "cell": cell,
                "selected_reference_type": selected_type,
                "pilot_empirical_mass": pilot_mass,
                "pilot_target_mass_lower_bound": lower_bound,
                "confirm_error_radius": radius_confirm,
                "strict_hit_margin": margin,
                "design_hit_gate": "PASS" if margin > 0.0 else "FAIL",
                "confirm_empirical_mass": confirm_mass,
                "observed_confirm_hit": confirm_mass > 0.0,
            }
        )

    igd_bound = math.nextafter(
        _lp_norm(widths, p_value),
        math.inf,
    )
    shifted_hv_bound = math.nextafter(
        shifted_front_hv_deficit_bound(
            canonical_reference,
            additive_widths=widths,
            reference=hv_ref,
        ),
        math.inf,
    )
    box_diameter = _lp_norm(
        tuple(high - low for low, high in zip(lower, upper)),
        p_value,
    )
    reference_box_volume = math.prod(
        reference_value - low
        for reference_value, low in zip(hv_ref, lower)
    )
    tolerances_nontrivial = bool(
        igd_tolerance < box_diameter
        and hv_tolerance < reference_box_volume
    )
    metric_gate = bool(
        tolerances_nontrivial
        and igd_bound <= igd_tolerance
        and shifted_hv_bound <= hv_tolerance
    )
    hit_design_gate = bool(
        published_concentration_gate
        and all(
            record["design_hit_gate"] == "PASS"
            for record in assignments
        )
    )
    published_design_pass = bool(hit_design_gate and metric_gate)
    regeneration_design_pass = bool(
        regeneration_hit_design_gate and metric_gate
    )
    if certificate_mode == "published":
        active_certificate_basis = "published"
        certified_assignments = assignments
        design_pass = published_design_pass
        active_claim_level = (
            ClaimLevel.PARETO_SMC_FIXED_REFERENCE_BOUND.value
            if design_pass
            else ClaimLevel.PARETO_SMC_BOOTSTRAP_BOUND.value
        )
    elif certificate_mode == "regeneration":
        active_certificate_basis = "regeneration"
        certified_assignments = regeneration_assignments
        design_pass = regeneration_design_pass
        active_claim_level = (
            ClaimLevel.PARETO_SMC_REGENERATION_REFERENCE_BOUND.value
            if design_pass
            else ClaimLevel.PARETO_SMC_BOOTSTRAP_BOUND.value
        )
    elif regeneration_design_pass:
        active_certificate_basis = "regeneration"
        certified_assignments = regeneration_assignments
        design_pass = True
        active_claim_level = (
            ClaimLevel.PARETO_SMC_REGENERATION_REFERENCE_BOUND.value
        )
    elif published_design_pass:
        active_certificate_basis = "published"
        certified_assignments = assignments
        design_pass = True
        active_claim_level = (
            ClaimLevel.PARETO_SMC_FIXED_REFERENCE_BOUND.value
        )
    else:
        active_certificate_basis = "none"
        certified_assignments = regeneration_assignments
        design_pass = False
        active_claim_level = (
            ClaimLevel.PARETO_SMC_BOOTSTRAP_BOUND.value
        )

    terminal_support = tuple(
        tuple(float(value) for value in point)
        for point in confirm.objectives
    )
    nondominated_terminal = nondominated_points(terminal_support)
    actual_igd = _igd(
        canonical_reference,
        terminal_support,
        p=p_value,
        plus=False,
    )
    actual_igd_plus = _igd(
        canonical_reference,
        nondominated_terminal,
        p=p_value,
        plus=True,
    )
    reference_hv = hypervolume_minimization(
        canonical_reference,
        hv_ref,
    )
    terminal_hv = hypervolume_minimization(
        nondominated_terminal,
        hv_ref,
    )
    actual_hv_deficit = max(0.0, reference_hv - terminal_hv)
    published_observed_all_hits = all(
        bool(record["observed_confirm_hit"])
        for record in assignments
    )
    regeneration_observed_all_hits = all(
        bool(record["observed_confirm_hit"])
        for record in regeneration_assignments
    )
    observed_all_hits = all(
        bool(record["observed_confirm_hit"])
        for record in certified_assignments
    )
    realized_metric_pass = bool(
        observed_all_hits
        and actual_igd <= igd_bound
        and actual_igd_plus <= igd_bound
        and actual_hv_deficit <= shifted_hv_bound
    )

    particles_per_reference = int(
        confirm_metadata["particles_per_reference"]
    )
    if len(confirm.particles) != len(confirm.objectives) or len(
        confirm.particles
    ) != particles_per_reference * reference_type_count:
        raise FixedScheduleCertificateError(
            "Confirm terminal particles do not match the frozen typed layout."
        )
    cell_cover_entries = []
    # The bounded certificate archive must use the same pilot-selected type
    # assignment as the active theorem path.
    for assignment in certified_assignments:
        cell = tuple(assignment["cell"])
        selected_type = int(assignment["selected_reference_type"])
        start = selected_type * particles_per_reference
        stop = start + particles_per_reference
        candidates = []
        for flat_index in range(start, stop):
            objectives = tuple(confirm.objectives[flat_index])
            if (
                _cell_index(
                    objectives,
                    lower=lower,
                    upper=upper,
                    widths=widths,
                    counts=counts,
                )
                == cell
            ):
                candidates.append(
                    (
                        objectives,
                        tuple(confirm.particles[flat_index]),
                        flat_index,
                    )
                )
        if candidates:
            objectives, tour, flat_index = min(candidates)
            cell_cover_entries.append(
                {
                    "epsilon_cell": cell,
                    "selected_reference_type": selected_type,
                    "terminal_flat_index": flat_index,
                    "tour": tour,
                    "objectives": objectives,
                }
            )
    cell_cover_complete = (
        len(cell_cover_entries) == len(reference_cells)
        and len(
            {
                tuple(entry["epsilon_cell"])
                for entry in cell_cover_entries
            }
        )
        == len(reference_cells)
    )
    if cell_cover_complete:
        cell_cover_support = tuple(
            tuple(entry["objectives"])
            for entry in cell_cover_entries
        )
        nondominated_cell_cover = nondominated_points(
            cell_cover_support
        )
        cell_cover_igd = _igd(
            canonical_reference,
            cell_cover_support,
            p=p_value,
            plus=False,
        )
        cell_cover_igd_plus = _igd(
            canonical_reference,
            nondominated_cell_cover,
            p=p_value,
            plus=True,
        )
        cell_cover_hv = hypervolume_minimization(
            nondominated_cell_cover,
            hv_ref,
        )
        cell_cover_hv_deficit = max(0.0, reference_hv - cell_cover_hv)
        cell_cover_metric_gate = bool(
            cell_cover_igd <= igd_bound
            and cell_cover_igd_plus <= igd_bound
            and cell_cover_hv_deficit <= shifted_hv_bound
        )
    else:
        cell_cover_support = ()
        nondominated_cell_cover = ()
        cell_cover_igd = None
        cell_cover_igd_plus = None
        cell_cover_hv_deficit = None
        cell_cover_metric_gate = False

    pilot_terminal_support_sha256 = _terminal_support_sha256(
        pilot,
        label="pilot",
    )
    confirm_terminal_support_sha256 = _terminal_support_sha256(
        confirm,
        label="confirm",
    )
    pair_payload = {
        "pilot_run_contract_hash": pilot_metadata.get("run_contract_hash"),
        "confirm_run_contract_hash": confirm_metadata.get(
            "run_contract_hash"
        ),
        "pilot_terminal_support_sha256": (
            pilot_terminal_support_sha256
        ),
        "confirm_terminal_support_sha256": (
            confirm_terminal_support_sha256
        ),
        "reference_artifact_sha256": reference_hash,
        "reference_payload_sha256": reference_payload_hash,
        "reference_cells": reference_cells,
        "certificate_mode": certificate_mode,
        "active_certificate_basis": active_certificate_basis,
        "pilot_failure_budget": delta_pilot,
        "confirm_failure_budget": delta_confirm,
        "theorem_family_count": theorem_family_count,
        "igd_p": p_value,
        "hv_reference": hv_ref,
        "max_igd_bound": igd_tolerance,
        "max_hv_deficit_bound": hv_tolerance,
    }
    return {
        "schema": "pareto_smc_fixed_reference_pilot_confirm_v2",
        "claim_level": active_claim_level,
        "scientific_design_gate": "PASS" if design_pass else "FAIL",
        "realized_metric_gate": (
            "PASS" if realized_metric_pass else "FAIL"
        ),
        "claim_scope": (
            "fixed feasible reference front frozen independently of pilot "
            "and confirm; not the unknown complete Pareto front"
        ),
        "reference_feasibility_assumed_from_external_source": True,
        "reference_feasibility_verified_by_runtime": False,
        "reference_source": reference_source,
        "reference_artifact_sha256": reference_hash,
        "reference_payload_sha256": reference_payload_hash,
        "reference_objectives": canonical_reference,
        "reference_cells": reference_cells,
        "pilot_seed": pilot_metadata.get("seed"),
        "confirm_seed": confirm_metadata.get("seed"),
        "pilot_terminal_support_sha256": (
            pilot_terminal_support_sha256
        ),
        "confirm_terminal_support_sha256": (
            confirm_terminal_support_sha256
        ),
        "distinct_seed_gate": (
            pilot_metadata.get("seed") != confirm_metadata.get("seed")
        ),
        "ideal_product_stream_assumption": True,
        "pair_signature_sha256": _payload_sha256(pair_payload),
        "requested_certificate_mode": certificate_mode,
        "active_certificate_basis": active_certificate_basis,
        "theorem_family_count": theorem_family_count,
        "published_pilot_failure_budget": published_delta_pilot,
        "published_confirm_failure_budget": published_delta_confirm,
        "regeneration_pilot_failure_budget": regeneration_delta_pilot,
        "regeneration_confirm_failure_budget": regeneration_delta_confirm,
        "pilot_plan": pilot_plan.__dict__,
        "confirm_plan": confirm_plan.__dict__,
        "published_concentration_gate": (
            "PASS" if published_concentration_gate else "FAIL"
        ),
        "cell_assignments": tuple(assignments),
        "hit_design_gate": "PASS" if hit_design_gate else "FAIL",
        "published_design_gate": (
            "PASS" if published_design_pass else "FAIL"
        ),
        "regeneration_cell_assignments": tuple(
            regeneration_assignments
        ),
        "regeneration_final_target_normalizer_lower_bound": (
            final_normalizer_lower_bound
        ),
        "regeneration_final_residual_weight": final_residual_weight,
        "regeneration_confirm_miss_probability_upper_bound": (
            regeneration_confirm_miss_upper_bound
        ),
        "regeneration_hit_design_gate": (
            "PASS" if regeneration_hit_design_gate else "FAIL"
        ),
        "regeneration_design_gate": (
            "PASS" if regeneration_design_pass else "FAIL"
        ),
        "certified_cell_assignments": tuple(certified_assignments),
        "certified_cell_assignment_contract": (
            "terminal archive construction uses the same pilot-selected "
            "type per cell as the active certificate basis"
        ),
        "simultaneous_confidence_event_probability_at_least": (
            1.0 - delta_pilot
        ),
        "false_pass_probability_upper_bound": (
            delta_pilot + delta_confirm
        ),
        "false_pass_event": (
            "scientific_design_gate_PASS_and_confirm_cell_coverage_failure"
        ),
        "conditional_coverage_probability_given_pass_claimed": False,
        "probability_at_least": None,
        "probability_at_least_deprecation_reason": (
            "The pilot margin gate is random; the valid statement is a "
            "false-PASS probability bound, not a conditional coverage "
            "probability given PASS."
        ),
        "metric_igd_p": p_value,
        "metric_igd_aggregation": (
            "arithmetic_mean_of_nearest_l_p_distances"
        ),
        "metric_cell_widths": widths,
        "metric_igd_bound": igd_bound,
        "metric_igd_tolerance": igd_tolerance,
        "metric_hv_reference": hv_ref,
        "metric_hv_deficit_bound": shifted_hv_bound,
        "metric_hv_deficit_tolerance": hv_tolerance,
        "metric_tolerances_predeclared": True,
        "metric_tolerances_nontrivial": tolerances_nontrivial,
        "metric_nonvacuity_gate": "PASS" if metric_gate else "FAIL",
        "certified_output_scope": "nondominated_confirm_terminal_support",
        "compressed_certified_output_scope": (
            "deterministic_one_terminal_representative_per_hit_reference_cell"
        ),
        "cell_cover_archive_policy": (
            "deterministic_lexicographic_terminal_representative_v1"
        ),
        "cell_cover_archive_entries": tuple(cell_cover_entries),
        "cell_cover_archive_size": len(cell_cover_entries),
        "cell_cover_archive_size_bound": len(reference_cells),
        "cell_cover_archive_complete_on_observed_hit_event": (
            cell_cover_complete
        ),
        "cell_cover_archive_construction_gate": (
            "PASS" if cell_cover_complete else "FAIL"
        ),
        "cell_cover_archive_metric_preservation_gate": (
            "PASS" if cell_cover_metric_gate else "FAIL"
        ),
        "cell_cover_archive_ordinary_igd": cell_cover_igd,
        "cell_cover_archive_nondominated_igd_plus": (
            cell_cover_igd_plus
        ),
        "cell_cover_archive_nondominated_hv_deficit": (
            cell_cover_hv_deficit
        ),
        "cell_cover_archive_keeps_dominated_same_cell_witnesses": True,
        "cell_cover_archive_nondominated_view_size": len(
            nondominated_cell_cover
        ),
        "observed_all_assigned_cells_hit": observed_all_hits,
        "observed_all_published_assigned_cells_hit": (
            published_observed_all_hits
        ),
        "observed_all_regeneration_assigned_cells_hit": (
            regeneration_observed_all_hits
        ),
        "observed_terminal_support_igd": actual_igd,
        "observed_nondominated_terminal_igd_plus": actual_igd_plus,
        "observed_nondominated_terminal_hv_deficit": actual_hv_deficit,
        "terminal_support_size": len(terminal_support),
        "nondominated_terminal_support_size": len(
            nondominated_terminal
        ),
        "pilot_evaluations": int(pilot_metadata["evaluations_used"]),
        "confirm_evaluations": int(confirm_metadata["evaluations_used"]),
        "total_certificate_evaluations": (
            int(pilot_metadata["evaluations_used"])
            + int(confirm_metadata["evaluations_used"])
        ),
        "adaptive_ess_branch_covered": False,
    }


def build_regeneration_pilot_plan_commitment_from_spec(
    pilot: OptimizationResult,
    specification: object,
    *,
    confirm_particles_per_reference: int,
    run_seed: int = 0,
) -> dict[str, object]:
    """Freeze the pilot-selected regeneration plan before confirm executes."""

    from .pareto_fixed_reference_spec import (
        FixedReferenceCertificateSpecification,
    )

    if not isinstance(
        specification,
        FixedReferenceCertificateSpecification,
    ):
        raise FixedScheduleCertificateError(
            "specification must be a validated fixed-reference specification."
        )
    if (
        isinstance(confirm_particles_per_reference, bool)
        or not isinstance(confirm_particles_per_reference, int)
        or confirm_particles_per_reference <= 0
    ):
        raise FixedScheduleCertificateError(
            "confirm_particles_per_reference must be a positive integer."
        )
    metadata = pilot.metadata
    if not isinstance(metadata, dict):
        raise FixedScheduleCertificateError(
            "The pilot result must contain a metadata dictionary."
        )
    if metadata.get("resampling_policy") != "always":
        raise FixedScheduleCertificateError(
            "The regeneration pilot must resample at every positive stage."
        )
    if metadata.get("bootstrap_mutations_by_stage") is None:
        raise FixedScheduleCertificateError(
            "The regeneration pilot lacks fixed mutation counts."
        )
    _validate_uniform_terminal_weights(metadata, label="pilot")
    pilot_seed, confirm_seed = specification.stream_seeds(run_seed)
    if metadata.get("instance_sha256") != specification.instance_sha256:
        raise FixedScheduleCertificateError(
            "Pilot instance hash does not match the frozen specification."
        )
    if metadata.get("seed") != pilot_seed:
        raise FixedScheduleCertificateError(
            "Pilot seed does not match the frozen specification."
        )
    if (
        metadata.get("external_specification_sha256")
        != specification.pareto_smc_specification_sha256
    ):
        raise FixedScheduleCertificateError(
            "Pilot Pareto-SMC specification hash is not bound."
        )

    lower = tuple(
        float(value)
        for value in _metadata_tuple(metadata, "objective_lower_bounds")
    )
    upper = tuple(
        float(value)
        for value in _metadata_tuple(metadata, "objective_upper_bounds")
    )
    widths = tuple(
        float(value) for value in _metadata_tuple(metadata, "epsilon")
    )
    counts = tuple(
        int(value)
        for value in _metadata_tuple(metadata, "epsilon_cell_counts")
    )
    canonical_reference = tuple(
        sorted(
            tuple(float(value) for value in point)
            for point in specification.reference_objectives
        )
    )
    if (
        _payload_sha256(canonical_reference)
        != specification.reference_artifact_sha256
    ):
        raise FixedScheduleCertificateError(
            "The committed reference payload hash does not match."
        )
    reference_cells = tuple(
        sorted(
            {
                _cell_index(
                    point,
                    lower=lower,
                    upper=upper,
                    widths=widths,
                    counts=counts,
                )
                for point in canonical_reference
            }
        )
    )
    pilot_masses = _cell_mass_maps(pilot, label="pilot")
    type_count = int(metadata["num_reference_types"])
    beta_schedule = tuple(
        float(value)
        for value in _metadata_tuple(metadata, "beta_schedule")
    )
    mutation_steps = tuple(
        int(value)
        for value in _metadata_tuple(
            metadata,
            "bootstrap_mutations_by_stage",
        )
    )
    normalizer_lower = target_normalizer_lower_bound(
        beta_schedule[-1],
        1.0 + float(metadata["chebyshev_rho"]),
    )
    residual = terminal_residual_weight(
        global_refresh_probability=float(
            metadata["global_refresh_probability"]
        ),
        normalizer_lower_bound=normalizer_lower,
        mutation_steps=mutation_steps[-1],
    )
    pair_failure_budget = (
        specification.pilot_failure_budget
        / (type_count * len(reference_cells))
    )
    assignments = []
    for cell in reference_cells:
        candidates = []
        for type_index in range(type_count):
            empirical = pilot_masses[type_index].get(cell, 0.0)
            mass_certificate = pilot_target_mass_lower_bound(
                empirical_terminal_mass=empirical,
                pilot_particles=int(
                    metadata["particles_per_reference"]
                ),
                pilot_failure_budget=pair_failure_budget,
                pilot_residual_weight=residual,
            )
            candidates.append(
                (
                    mass_certificate.target_mass_lower_bound,
                    empirical,
                    -type_index,
                    type_index,
                    mass_certificate,
                )
            )
        lower_mass, empirical, _, selected_type, mass_certificate = max(
            candidates
        )
        confirm_certificate = confirm_cell_certificate(
            target_mass_lower_bound=lower_mass,
            confirm_particles=confirm_particles_per_reference,
            confirm_residual_weight=residual,
        )
        assignments.append(
            {
                "cell": cell,
                "selected_reference_type": selected_type,
                "pilot_empirical_mass": empirical,
                "pilot_hoeffding_radius": (
                    mass_certificate.pilot_hoeffding_radius
                ),
                "pilot_target_mass_lower_bound": lower_mass,
                "confirm_per_particle_hit_lower_bound": (
                    confirm_certificate.per_particle_hit_lower_bound
                ),
                "confirm_cell_miss_probability_upper_bound": (
                    confirm_certificate.cell_miss_probability_upper_bound
                ),
            }
        )
    miss_upper = min(
        1.0,
        math.nextafter(
            math.fsum(
                float(
                    row["confirm_cell_miss_probability_upper_bound"]
                )
                for row in assignments
            ),
            math.inf,
        ),
    )
    payload = {
        "schema": "pareto_smc_regeneration_pilot_plan_commitment_v1",
        "certificate_mode": "regeneration",
        "selection_rule": (
            "max_target_mass_lower_then_empirical_then_lowest_type_v1"
        ),
        "certificate_specification_sha256": specification.sha256,
        "pareto_smc_specification_sha256": (
            specification.pareto_smc_specification_sha256
        ),
        "instance_sha256": specification.instance_sha256,
        "reference_artifact_sha256": (
            specification.reference_artifact_sha256
        ),
        "reference_cells": reference_cells,
        "run_seed": run_seed,
        "pilot_seed": pilot_seed,
        "confirm_seed": confirm_seed,
        "pilot_run_contract_hash": metadata.get("run_contract_hash"),
        "pilot_terminal_support_sha256": _terminal_support_sha256(
            pilot,
            label="pilot",
        ),
        "pilot_evaluations": int(metadata["evaluations_used"]),
        "pilot_particles_per_reference": int(
            metadata["particles_per_reference"]
        ),
        "confirm_particles_per_reference": (
            confirm_particles_per_reference
        ),
        "pilot_failure_budget": specification.pilot_failure_budget,
        "confirm_failure_budget": specification.confirm_failure_budget,
        "pilot_pair_failure_budget": pair_failure_budget,
        "final_target_normalizer_lower_bound": normalizer_lower,
        "final_residual_weight": residual,
        "cell_assignments": tuple(assignments),
        "confirm_miss_probability_upper_bound": miss_upper,
        "regeneration_hit_design_gate": (
            "PASS"
            if (
                all(
                    float(row["pilot_target_mass_lower_bound"]) > 0.0
                    for row in assignments
                )
                and miss_upper
                <= specification.confirm_failure_budget
            )
            else "FAIL"
        ),
    }
    return {
        **payload,
        "commitment_sha256": _payload_sha256(payload),
    }


def certify_fixed_schedule_reference_metrics_from_spec(
    pilot: OptimizationResult,
    confirm: OptimizationResult,
    specification: object,
    *,
    run_seed: int = 0,
    certificate_mode: str = "published",
    pilot_plan_commitment: Mapping[str, object] | None = None,
    pilot_plan_commitment_preconfirm_order_attested_by_runner: bool = False,
) -> dict[str, object]:
    """Bind a certificate to a pre-run fixed-reference specification.

    The generic verifier can check commitment content, but a post-hoc pair
    artifact cannot prove when that content was produced.  The official
    pilot-confirm runner therefore supplies a separate control-flow
    attestation after it creates the commitment before launching confirm.
    This is not an independently timestamped or externally signed receipt.
    """

    from .pareto_fixed_reference_spec import (
        FixedReferenceCertificateSpecification,
    )

    if not isinstance(
        specification,
        FixedReferenceCertificateSpecification,
    ):
        raise FixedScheduleCertificateError(
            "specification must be a validated fixed-reference specification."
        )
    if not isinstance(
        pilot_plan_commitment_preconfirm_order_attested_by_runner,
        bool,
    ):
        raise FixedScheduleCertificateError(
            "pilot_plan_commitment_preconfirm_order_attested_by_runner "
            "must be boolean."
        )
    if (
        pilot_plan_commitment_preconfirm_order_attested_by_runner
        and (
            certificate_mode != "regeneration"
            or pilot_plan_commitment is None
        )
    ):
        raise FixedScheduleCertificateError(
            "A pre-confirm runner-order attestation is valid only for a "
            "provided regeneration commitment."
        )
    pilot_seed, confirm_seed = specification.stream_seeds(run_seed)
    for label, result, expected_seed in (
        ("pilot", pilot, pilot_seed),
        ("confirm", confirm, confirm_seed),
    ):
        metadata = result.metadata
        if metadata.get("instance_sha256") != specification.instance_sha256:
            raise FixedScheduleCertificateError(
                f"{label} instance hash does not match the frozen specification."
            )
        if metadata.get("seed") != expected_seed:
            raise FixedScheduleCertificateError(
                f"{label} seed does not match the frozen specification."
            )
        if (
            metadata.get("external_specification_sha256")
            != specification.pareto_smc_specification_sha256
        ):
            raise FixedScheduleCertificateError(
                f"{label} Pareto-SMC specification hash is not bound."
            )
    certificate = certify_fixed_schedule_reference_metrics(
        pilot,
        confirm,
        reference_objectives=specification.reference_objectives,
        reference_source=specification.reference_source,
        reference_artifact_sha256=(
            specification.reference_artifact_sha256
        ),
        pilot_failure_budget=specification.pilot_failure_budget,
        confirm_failure_budget=specification.confirm_failure_budget,
        igd_p=specification.igd_p,
        hv_reference=specification.hv_reference,
        max_igd_bound=specification.max_igd_bound,
        max_hv_deficit_bound=(
            specification.max_hv_deficit_bound
        ),
        certificate_mode=certificate_mode,
    )
    certificate["certificate_specification_path"] = str(
        specification.path
    )
    certificate["certificate_specification_sha256"] = (
        specification.sha256
    )
    certificate["pareto_smc_specification_sha256"] = (
        specification.pareto_smc_specification_sha256
    )
    certificate["run_seed"] = run_seed
    certificate["resolved_pilot_seed"] = pilot_seed
    certificate["resolved_confirm_seed"] = confirm_seed
    if certificate_mode == "regeneration":
        if pilot_plan_commitment is None:
            pilot_commitment_content_gate = "MISSING"
            pilot_commitment_sha256 = None
        else:
            expected_commitment = (
                build_regeneration_pilot_plan_commitment_from_spec(
                    pilot,
                    specification,
                    confirm_particles_per_reference=int(
                        confirm.metadata["particles_per_reference"]
                    ),
                    run_seed=run_seed,
                )
            )
            provided_commitment = dict(pilot_plan_commitment)
            provided_digest = provided_commitment.pop(
                "commitment_sha256",
                None,
            )
            if (
                provided_digest
                != _payload_sha256(provided_commitment)
                or provided_digest
                != expected_commitment["commitment_sha256"]
            ):
                raise FixedScheduleCertificateError(
                    "The pilot plan commitment does not match the "
                    "pilot-selected regeneration plan."
                )
            pilot_commitment_content_gate = "PASS"
            pilot_commitment_sha256 = expected_commitment[
                "commitment_sha256"
            ]
    elif certificate_mode == "published":
        pilot_commitment_content_gate = (
            "NOT_REQUIRED_LEGACY_PUBLISHED_MODE"
        )
        pilot_commitment_sha256 = None
    else:
        pilot_commitment_content_gate = (
            "UNSUPPORTED_FOR_FORMAL_OR_MODE"
        )
        pilot_commitment_sha256 = None
    if certificate_mode == "regeneration":
        if pilot_commitment_content_gate != "PASS":
            pilot_commitment_gate = pilot_commitment_content_gate
            preconfirm_order_gate = "NOT_EVALUATED"
        elif (
            pilot_plan_commitment_preconfirm_order_attested_by_runner
        ):
            pilot_commitment_gate = "PASS"
            preconfirm_order_gate = "PASS_RUNNER_CONTROL_FLOW_ATTESTED"
        else:
            pilot_commitment_gate = (
                "MISSING_PRECONFIRM_RUNNER_ORDER_ATTESTATION"
            )
            preconfirm_order_gate = "MISSING"
    else:
        pilot_commitment_gate = pilot_commitment_content_gate
        preconfirm_order_gate = pilot_commitment_content_gate
    certificate["pilot_plan_commitment_content_gate"] = (
        pilot_commitment_content_gate
    )
    certificate["pilot_plan_commitment_gate"] = pilot_commitment_gate
    certificate["pilot_plan_commitment_sha256"] = (
        pilot_commitment_sha256
    )
    certificate["pilot_plan_commitment_preconfirm_order_gate"] = (
        preconfirm_order_gate
    )
    certificate[
        "pilot_plan_commitment_preconfirm_order_attested_by_runner"
    ] = (
        pilot_plan_commitment_preconfirm_order_attested_by_runner
        and pilot_commitment_content_gate == "PASS"
    )
    certificate[
        "pilot_plan_commitment_preconfirm_timing_independently_verified"
    ] = False
    certificate["pilot_plan_committed_before_confirm"] = None
    certificate[
        "pilot_plan_committed_before_confirm_deprecation_reason"
    ] = (
        "Commitment content does not prove creation time. The official "
        "runner can attest its control-flow order, but independent timing "
        "requires an external append-only or signed pre-confirm receipt."
    )
    certificate["formal_pair_signature_sha256"] = _payload_sha256(
        {
            "pair_signature_sha256": certificate[
                "pair_signature_sha256"
            ],
            "certificate_specification_sha256": specification.sha256,
            "certificate_mode": certificate_mode,
            "pilot_plan_commitment_sha256": (
                pilot_commitment_sha256
            ),
            "pilot_plan_commitment_preconfirm_order_attested_by_runner": (
                certificate[
                    "pilot_plan_commitment_preconfirm_order_attested_by_runner"
                ]
            ),
        }
    )
    certificate["reference_feasibility_gate"] = (
        "PASS"
        if specification.reference_feasibility_verified_by_runtime
        else "EXTERNAL_ASSERTION_ONLY"
    )
    certificate["reference_feasibility_verified_by_runtime"] = (
        specification.reference_feasibility_verified_by_runtime
    )
    certificate["reference_feasibility_assumed_from_external_source"] = (
        not specification.reference_feasibility_verified_by_runtime
    )
    certificate["reference_witness_payload_sha256"] = (
        specification.reference_witness_payload_sha256
    )
    certificate["reference_witness_max_abs_error"] = (
        specification.reference_witness_max_abs_error
    )
    certificate["reference_witness_equivalence_contract"] = (
        specification.reference_witness_equivalence_contract
    )
    certificate["certified_archive_policy"] = (
        specification.certified_archive_policy
    )
    certificate["certified_archive_max_size"] = (
        specification.certified_archive_max_size
    )
    if specification.certified_archive_policy is not None:
        assert specification.certified_archive_max_size is not None
        reference_cell_count = len(certificate["reference_cells"])
        if specification.certified_archive_max_size < reference_cell_count:
            raise FixedScheduleCertificateError(
                "The certified archive cap is smaller than the number of "
                "distinct frozen reference cells."
            )
        observed_size = int(certificate["cell_cover_archive_size"])
        size_gate = (
            observed_size <= specification.certified_archive_max_size
        )
        certificate["certified_archive_size_gate"] = (
            "PASS" if size_gate else "FAIL"
        )
        certificate["certified_archive_gate"] = (
            "PASS"
            if (
                size_gate
                and certificate[
                    "cell_cover_archive_construction_gate"
                ]
                == "PASS"
                and certificate[
                    "cell_cover_archive_metric_preservation_gate"
                ]
                == "PASS"
            )
            else "FAIL"
        )
        certificate["certified_archive_theorem_scope"] = (
            "conditional on the simultaneous assigned-cell hit event; one "
            "same-cell terminal witness per distinct frozen reference cell "
            "preserves the stated ordinary IGD, IGD+, and HV-deficit bounds"
        )
        certificate["certified_output_scope"] = certificate[
            "compressed_certified_output_scope"
        ]
    formal_packet_pass = bool(
        certificate["scientific_design_gate"] == "PASS"
        and certificate["reference_feasibility_gate"] == "PASS"
        and certificate.get("certified_archive_gate") == "PASS"
        and (
            certificate_mode == "published"
            or certificate["pilot_plan_commitment_gate"] == "PASS"
        )
    )
    certificate["formal_packet_gate"] = (
        "PASS" if formal_packet_pass else "FAIL"
    )
    if certificate_mode == "regeneration":
        certificate[
            "external_preconfirm_commitment_receipt_gate"
        ] = "NOT_IMPLEMENTED"
        certificate[
            "independently_auditable_preconfirm_freeze_gate"
        ] = "FAIL"
        certificate["publication_certificate_packet_gate"] = "FAIL"
    elif certificate_mode == "published":
        certificate[
            "external_preconfirm_commitment_receipt_gate"
        ] = "NOT_REQUIRED_BY_LEGACY_MODE"
        certificate[
            "independently_auditable_preconfirm_freeze_gate"
        ] = "NOT_CLAIMED"
        certificate["publication_certificate_packet_gate"] = (
            certificate["formal_packet_gate"]
        )
    else:
        certificate[
            "external_preconfirm_commitment_receipt_gate"
        ] = "UNSUPPORTED_FOR_FORMAL_OR_MODE"
        certificate[
            "independently_auditable_preconfirm_freeze_gate"
        ] = "FAIL"
        certificate["publication_certificate_packet_gate"] = "FAIL"
    certificate["formal_packet_claim_scope"] = (
        "witness-bound fixed feasible reference set and bounded same-cell "
        "archive under trusted runner control flow; not independent timing "
        "provenance and not the unknown complete Pareto front"
    )
    return certificate
