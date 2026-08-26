from __future__ import annotations

"""Fail-closed loader for source-bound finite-step cell-probe manifests."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

from .pareto_cell_certification import CertifiedCellType
from .types import ObjectiveVector

CELL_SPEC_SCHEMA = "pareto_cell_source_bound_spec_v4"
LEGACY_CELL_SPEC_SCHEMAS = {
    "pareto_cell_source_bound_spec_v2",
    "pareto_cell_source_bound_spec_v3",
}


@dataclass(frozen=True)
class ParetoCellCertificationSpecification:
    path: Path
    sha256: str
    instance_sha256: str
    target_safety_lower_bounds: ObjectiveVector
    target_safety_upper_bounds: ObjectiveVector
    metric_lower_bounds: ObjectiveVector
    metric_upper_bounds: ObjectiveVector
    cell_widths: ObjectiveVector
    target_safety_box_source: str
    target_safety_box_proof_sha256: str
    metric_box_source: str
    metric_box_proof_sha256: str
    metric_igd_p: float
    max_igd_bound: float | None
    hv_reference: ObjectiveVector
    max_hv_deficit_bound: float | None
    cell_completeness_proof_sha256: str
    beta: float
    chebyshev_rho: float
    confidence_delta: float
    cell_types: Tuple[CertifiedCellType, ...]
    archive_max_size: int | None


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _float_tuple(value: object, label: str, dimension: int) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != dimension:
        raise ValueError(f"{label} must be an array of length {dimension}.")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite values.")
    return result


def _int_tuple(value: object, label: str, dimension: int) -> Tuple[int, ...]:
    if not isinstance(value, list) or len(value) != dimension:
        raise ValueError(f"{label} must be an integer array of length {dimension}.")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValueError(f"{label} must contain nonnegative integers.")
    return tuple(value)


def load_pareto_cell_certification_specification(
    path: str | Path,
    *,
    objective_dimension: int,
    num_cities: int,
    expected_instance_sha256: str,
) -> ParetoCellCertificationSpecification:
    del num_cities  # source-bound contracts contain no in-cell anchor tours
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Cell certification specification is missing: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Cell certification specification is not valid UTF-8 JSON."
        ) from error
    root = _mapping(payload, "cell certification specification")
    schema = root.get("schema")
    common_keys = {
        "schema",
        "instance_sha256",
        "cell_grid",
        "target",
        "confidence_delta",
        "cell_completeness",
        "cell_types",
        "reporting",
    }
    if schema == CELL_SPEC_SCHEMA:
        expected_keys = common_keys | {
            "target_safety_box",
            "metric_box",
            "metric_nonvacuity",
        }
    elif schema == "pareto_cell_source_bound_spec_v3":
        expected_keys = common_keys | {"target_safety_box", "metric_box"}
    elif schema == "pareto_cell_source_bound_spec_v2":
        expected_keys = common_keys | {"objective_box"}
    else:
        raise ValueError(
            f"schema must be {CELL_SPEC_SCHEMA!r} or one of "
            f"{sorted(LEGACY_CELL_SPEC_SCHEMAS)!r}."
        )
    if set(root) != expected_keys:
        raise ValueError(
            "Cell certification specification has an unexpected top-level shape; "
            f"missing={sorted(expected_keys - set(root))}, "
            f"extra={sorted(set(root) - expected_keys)}."
        )
    bound_instance = _sha256(root.get("instance_sha256"), "instance_sha256")
    if bound_instance != expected_instance_sha256:
        raise ValueError("Cell certification specification is bound to another instance.")

    if schema == "pareto_cell_source_bound_spec_v2":
        target_box = _mapping(root.get("objective_box"), "objective_box")
        metric_box = target_box
        target_box_label = "objective_box"
        metric_box_label = "objective_box"
    else:
        target_box = _mapping(
            root.get("target_safety_box"),
            "target_safety_box",
        )
        metric_box = _mapping(root.get("metric_box"), "metric_box")
        target_box_label = "target_safety_box"
        metric_box_label = "metric_box"

    expected_box_keys = {
        "source",
        "lower",
        "upper",
        "proof_artifact_sha256",
        "archive_independent",
    }
    if set(target_box) != expected_box_keys:
        raise ValueError(f"{target_box_label} has an unexpected shape.")
    if set(metric_box) != expected_box_keys:
        raise ValueError(f"{metric_box_label} has an unexpected shape.")
    if target_box.get("archive_independent") is not True:
        raise ValueError(f"{target_box_label}.archive_independent must be true.")
    if metric_box.get("archive_independent") is not True:
        raise ValueError(f"{metric_box_label}.archive_independent must be true.")
    target_lower = _float_tuple(
        target_box.get("lower"),
        f"{target_box_label}.lower",
        objective_dimension,
    )
    target_upper = _float_tuple(
        target_box.get("upper"),
        f"{target_box_label}.upper",
        objective_dimension,
    )
    metric_lower = _float_tuple(
        metric_box.get("lower"),
        f"{metric_box_label}.lower",
        objective_dimension,
    )
    metric_upper = _float_tuple(
        metric_box.get("upper"),
        f"{metric_box_label}.upper",
        objective_dimension,
    )
    if any(high <= low for low, high in zip(target_lower, target_upper)):
        raise ValueError(
            "Every target safety-box upper endpoint must exceed its lower endpoint."
        )
    if any(high <= low for low, high in zip(metric_lower, metric_upper)):
        raise ValueError(
            "Every metric-box upper endpoint must exceed its lower endpoint."
        )
    if any(
        metric_low < target_low or metric_high > target_high
        for target_low, target_high, metric_low, metric_high in zip(
            target_lower,
            target_upper,
            metric_lower,
            metric_upper,
        )
    ):
        raise ValueError(
            "The metric box must be contained in the target safety box."
        )
    allowed_sources = {
        "exact_enumeration",
        "problem_specific_theorem",
        "unused_calibration_manifest_with_holdout_proof",
        "external_verified_manifest",
    }
    target_box_source = str(target_box.get("source"))
    metric_box_source = str(metric_box.get("source"))
    if target_box_source not in allowed_sources:
        raise ValueError(
            f"{target_box_label}.source is not an allowed source-bound value."
        )
    if metric_box_source not in allowed_sources:
        raise ValueError(
            f"{metric_box_label}.source is not an allowed source-bound value."
        )
    target_box_proof = _sha256(
        target_box.get("proof_artifact_sha256"),
        f"{target_box_label}.proof_artifact_sha256",
    )
    metric_box_proof = _sha256(
        metric_box.get("proof_artifact_sha256"),
        f"{metric_box_label}.proof_artifact_sha256",
    )

    if schema == CELL_SPEC_SCHEMA:
        nonvacuity = _mapping(
            root.get("metric_nonvacuity"),
            "metric_nonvacuity",
        )
        expected_nonvacuity_keys = {
            "igd_p",
            "max_igd_bound",
            "hv_reference",
            "max_hv_deficit_bound",
        }
        if set(nonvacuity) != expected_nonvacuity_keys:
            raise ValueError("metric_nonvacuity has an unexpected shape.")
        metric_igd_p = float(nonvacuity.get("igd_p"))
        if not (
            math.isinf(metric_igd_p)
            or (math.isfinite(metric_igd_p) and metric_igd_p >= 1.0)
        ):
            raise ValueError("metric_nonvacuity.igd_p must lie in [1, infinity].")
        max_igd_bound = float(nonvacuity.get("max_igd_bound"))
        max_hv_deficit_bound = float(
            nonvacuity.get("max_hv_deficit_bound")
        )
        if (
            not math.isfinite(max_igd_bound)
            or max_igd_bound < 0.0
            or not math.isfinite(max_hv_deficit_bound)
            or max_hv_deficit_bound < 0.0
        ):
            raise ValueError(
                "metric nonvacuity tolerances must be finite and nonnegative."
            )
        hv_reference = _float_tuple(
            nonvacuity.get("hv_reference"),
            "metric_nonvacuity.hv_reference",
            objective_dimension,
        )
        if any(
            reference < upper
            for reference, upper in zip(hv_reference, metric_upper)
        ):
            raise ValueError(
                "metric_nonvacuity.hv_reference must be coordinatewise no "
                "better than metric_box.upper."
            )
    else:
        metric_igd_p = 2.0
        max_igd_bound = None
        hv_reference = metric_upper
        max_hv_deficit_bound = None

    grid = _mapping(root.get("cell_grid"), "cell_grid")
    if set(grid) != {
        "coordinate_system",
        "widths",
        "archive_independent",
    }:
        raise ValueError("cell_grid has an unexpected shape.")
    if grid.get("coordinate_system") != "original_objective_units":
        raise ValueError(
            "cell_grid.coordinate_system must be 'original_objective_units'."
        )
    if grid.get("archive_independent") is not True:
        raise ValueError("cell_grid.archive_independent must be true.")
    widths = _float_tuple(
        grid.get("widths"),
        "cell_grid.widths",
        objective_dimension,
    )
    if any(
        width <= 0.0 or width > high - low
        for width, low, high in zip(widths, metric_lower, metric_upper)
    ):
        raise ValueError(
            "Each original-unit cell width must lie in (0, box span]."
        )
    metric_cell_counts = tuple(
        max(1, int(math.ceil((high - low) / width)))
        for width, low, high in zip(widths, metric_lower, metric_upper)
    )

    target = _mapping(root.get("target"), "target")
    if set(target) != {"beta", "chebyshev_rho", "family", "base_measure"}:
        raise ValueError("target has an unexpected shape.")
    if target.get("family") != "uniform_base_cell_penalized_augmented_tchebycheff":
        raise ValueError("target.family is invalid.")
    if target.get("base_measure") != "uniform_fixed_zero_tours":
        raise ValueError("target.base_measure is invalid.")
    beta = float(target.get("beta"))
    rho = float(target.get("chebyshev_rho"))
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("target.beta must be finite and nonnegative.")
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError("target.chebyshev_rho must be finite and positive.")

    confidence_delta = float(root.get("confidence_delta"))
    if not math.isfinite(confidence_delta) or not (0.0 < confidence_delta < 1.0):
        raise ValueError("confidence_delta must lie in (0, 1).")

    completeness = _mapping(root.get("cell_completeness"), "cell_completeness")
    if set(completeness) != {
        "claimed_cells",
        "proof_artifact_sha256",
        "source",
    }:
        raise ValueError("cell_completeness has an unexpected shape.")
    completeness_proof = _sha256(
        completeness.get("proof_artifact_sha256"),
        "cell_completeness.proof_artifact_sha256",
    )
    if completeness.get("source") not in {
        "exact_enumeration",
        "external_verified_manifest",
        "problem_specific_theorem",
    }:
        raise ValueError("cell_completeness.source is not source-bound.")
    claimed_cells_raw = completeness.get("claimed_cells")
    if not isinstance(claimed_cells_raw, list) or not claimed_cells_raw:
        raise ValueError("cell_completeness.claimed_cells must be nonempty.")
    claimed_cells = tuple(
        _int_tuple(
            cell,
            f"cell_completeness.claimed_cells[{index}]",
            objective_dimension,
        )
        for index, cell in enumerate(claimed_cells_raw)
    )
    if len(set(claimed_cells)) != len(claimed_cells):
        raise ValueError("cell_completeness.claimed_cells must be unique.")
    for cell_index, cell in enumerate(claimed_cells):
        if any(
            coordinate >= count
            for coordinate, count in zip(cell, metric_cell_counts)
        ):
            raise ValueError(
                "cell_completeness.claimed_cells["
                f"{cell_index}] lies outside the declared metric grid."
            )

    types_raw = root.get("cell_types")
    if not isinstance(types_raw, list) or not types_raw:
        raise ValueError("cell_types must be a nonempty array.")
    contracts = []
    for index, raw_contract in enumerate(types_raw):
        item = _mapping(raw_contract, f"cell_types[{index}]")
        expected_item_keys = {
            "cell",
            "reference_direction",
            "base_cell_mass_lower_bound",
            "base_mass_proof_sha256",
            "outside_cell_penalty",
            "global_refresh_probability",
            "mutation_steps",
            "particle_count",
            "failure_budget",
        }
        if set(item) != expected_item_keys:
            raise ValueError(f"cell_types[{index}] has an unexpected shape.")
        cell = _int_tuple(
            item.get("cell"),
            f"cell_types[{index}].cell",
            objective_dimension,
        )
        direction = _float_tuple(
            item.get("reference_direction"),
            f"cell_types[{index}].reference_direction",
            objective_dimension,
        )
        if any(weight <= 0.0 for weight in direction) or not math.isclose(
            sum(direction),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"cell_types[{index}].reference_direction must be strictly "
                "positive and sum to one."
            )
        kappa = float(item.get("base_cell_mass_lower_bound"))
        if not math.isfinite(kappa) or not (0.0 < kappa <= 1.0):
            raise ValueError("base_cell_mass_lower_bound must lie in (0, 1].")
        mass_proof = _sha256(
            item.get("base_mass_proof_sha256"),
            f"cell_types[{index}].base_mass_proof_sha256",
        )
        mutation_steps = item.get("mutation_steps")
        particle_count = item.get("particle_count")
        if (
            isinstance(mutation_steps, bool)
            or not isinstance(mutation_steps, int)
            or mutation_steps < 0
        ):
            raise ValueError("mutation_steps must be a nonnegative integer.")
        if (
            isinstance(particle_count, bool)
            or not isinstance(particle_count, int)
            or particle_count <= 0
        ):
            raise ValueError("particle_count must be a positive integer.")
        penalty = float(item.get("outside_cell_penalty"))
        gamma = float(item.get("global_refresh_probability"))
        failure_budget = float(item.get("failure_budget"))
        if not math.isfinite(penalty) or penalty < 0.0:
            raise ValueError(
                "outside_cell_penalty must be finite and nonnegative."
            )
        if not math.isfinite(gamma) or not (0.0 < gamma <= 1.0):
            raise ValueError(
                "global_refresh_probability must lie in (0, 1]."
            )
        if not math.isfinite(failure_budget) or not (
            0.0 < failure_budget < 1.0
        ):
            raise ValueError("failure_budget must lie in (0, 1).")
        contracts.append(
            CertifiedCellType(
                cell=cell,
                reference_direction=direction,
                base_cell_mass_lower_bound=kappa,
                base_mass_proof_sha256=mass_proof,
                outside_cell_penalty=penalty,
                global_refresh_probability=gamma,
                mutation_steps=mutation_steps,
                particle_count=particle_count,
                failure_budget=failure_budget,
            )
        )
    if {contract.cell for contract in contracts} != set(claimed_cells):
        raise ValueError(
            "The cell_types cells must equal the source-bound completeness cell set."
        )
    if sum(contract.failure_budget for contract in contracts) > confidence_delta + 1e-15:
        raise ValueError(
            "The sum of cell-type failure budgets exceeds confidence_delta."
        )

    reporting = _mapping(root.get("reporting"), "reporting")
    if set(reporting) != {"archive_max_size", "archive_role"}:
        raise ValueError("reporting has an unexpected shape.")
    if reporting.get("archive_role") != "reporting_only":
        raise ValueError("reporting.archive_role must be 'reporting_only'.")
    raw_size = reporting.get("archive_max_size")
    if raw_size is None:
        archive_size = None
    elif isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size <= 0:
        raise ValueError("reporting.archive_max_size must be null or positive.")
    else:
        archive_size = raw_size

    return ParetoCellCertificationSpecification(
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        instance_sha256=bound_instance,
        target_safety_lower_bounds=target_lower,
        target_safety_upper_bounds=target_upper,
        metric_lower_bounds=metric_lower,
        metric_upper_bounds=metric_upper,
        cell_widths=widths,
        target_safety_box_source=target_box_source,
        target_safety_box_proof_sha256=target_box_proof,
        metric_box_source=metric_box_source,
        metric_box_proof_sha256=metric_box_proof,
        metric_igd_p=metric_igd_p,
        max_igd_bound=max_igd_bound,
        hv_reference=hv_reference,
        max_hv_deficit_bound=max_hv_deficit_bound,
        cell_completeness_proof_sha256=completeness_proof,
        beta=beta,
        chebyshev_rho=rho,
        confidence_delta=confidence_delta,
        cell_types=tuple(contracts),
        archive_max_size=archive_size,
    )
