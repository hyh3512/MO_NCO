from __future__ import annotations

"""Strict loader for the archive-independent Pareto-SMC run specification."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from .instance import MultiObjectiveTSPInstance
from .types import ObjectiveVector


SPEC_SCHEMA = "annealed_pareto_smc_spec_v1"
EXACT_INCREMENTAL_TWO_OPT_CONTRACT = (
    "exact_incremental_two_opt_on_verified_integer_domain_else_"
    "full_tour_v1"
)


@dataclass(frozen=True)
class ParetoSMCSpecification:
    """Validated pre-run settings that are independent of a run archive."""

    path: Path
    sha256: str
    beta_schedule: Tuple[float, ...]
    reference_directions: Tuple[ObjectiveVector, ...]
    normalized_cell_widths: ObjectiveVector
    ess_threshold_fraction: float
    chebyshev_rho: float
    mutation_proposal: str
    mutation_objective_evaluation: str
    global_refresh_probability: float
    mutation_steps_by_stage: Tuple[int, ...] | None
    archive_max_size: int | None


def _reject_duplicate_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"Duplicate JSON field is forbidden: {key!r}."
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}.")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} has an unexpected shape; "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}."
        )


def _finite_tuple(
    value: object,
    *,
    label: str,
    dimension: int | None = None,
) -> Tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array.")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in value
    ):
        raise ValueError(f"{label} must contain JSON numbers.")
    result = tuple(float(item) for item in value)
    if dimension is not None and len(result) != dimension:
        raise ValueError(f"{label} must have length {dimension}.")
    if not result or any(not math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite values.")
    return result


def load_pareto_smc_specification(
    path: str | Path,
    *,
    objective_dimension: int,
) -> ParetoSMCSpecification:
    """Load and fail-closed validate a Pareto-SMC manifest.

    The manifest binds the policy used to construct the objective grid.  The
    actual objective-box endpoints and original-unit widths remain
    instance-specific and are separately included in the optimizer hashes.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Pareto-SMC specification is missing: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            "Pareto-SMC specification is not valid strict UTF-8 JSON: "
            f"{resolved}: {error}"
        ) from error
    root = _mapping(payload, "Pareto-SMC specification")
    if root.get("schema") != SPEC_SCHEMA:
        raise ValueError(
            f"Pareto-SMC specification schema must be {SPEC_SCHEMA!r}."
        )
    expected_top_level_keys = {
        "schema",
        "objective_box",
        "epsilon_cells",
        "reference_directions",
        "target",
        "resampling",
        "mutation",
        "particle_allocation",
        "reporting",
    }
    if set(root) != expected_top_level_keys:
        missing = sorted(expected_top_level_keys - set(root))
        extra = sorted(set(root) - expected_top_level_keys)
        raise ValueError(
            "Pareto-SMC specification has an unexpected top-level shape; "
            f"missing={missing}, extra={extra}."
        )

    objective_box = _mapping(root.get("objective_box"), "objective_box")
    _exact_keys(
        objective_box,
        {"source", "archive_independent"},
        "objective_box",
    )
    if objective_box.get("source") != "analytic_distance_matrix_box":
        raise ValueError(
            "objective_box.source must be 'analytic_distance_matrix_box' "
            "for this formal benchmark entry."
        )
    if objective_box.get("archive_independent") is not True:
        raise ValueError("objective_box.archive_independent must be true.")
    cells = _mapping(root.get("epsilon_cells"), "epsilon_cells")
    _exact_keys(
        cells,
        {
            "coordinate_system",
            "widths",
            "archive_independent",
            "role",
        },
        "epsilon_cells",
    )
    if cells.get("coordinate_system") != "normalized_frozen_objective_box":
        raise ValueError(
            "epsilon_cells.coordinate_system must be "
            "'normalized_frozen_objective_box'."
        )
    widths = _finite_tuple(
        cells.get("widths"),
        label="epsilon_cells.widths",
        dimension=objective_dimension,
    )
    if any(width <= 0.0 or width > 1.0 for width in widths):
        raise ValueError("Every normalized epsilon-cell width must lie in (0, 1].")
    if cells.get("archive_independent") is not True:
        raise ValueError("epsilon_cells.archive_independent must be true.")
    if cells.get("role") != "reporting_and_coverage_only":
        raise ValueError(
            "epsilon_cells.role must be 'reporting_and_coverage_only'."
        )

    target = _mapping(root.get("target"), "target")
    _exact_keys(
        target,
        {
            "family",
            "stage_frozen",
            "beta_schedule",
            "chebyshev_rho",
        },
        "target",
    )
    if target.get("family") != "typed_augmented_tchebycheff_gibbs":
        raise ValueError(
            "target.family must be 'typed_augmented_tchebycheff_gibbs'."
        )
    if target.get("stage_frozen") is not True:
        raise ValueError("target.stage_frozen must be true.")
    raw_rho = target.get("chebyshev_rho")
    if isinstance(raw_rho, bool) or not isinstance(raw_rho, (int, float)):
        raise ValueError("target.chebyshev_rho must be a JSON number.")
    rho = float(raw_rho)
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError("target.chebyshev_rho must be finite and strictly positive.")
    beta_schedule = _finite_tuple(
        target.get("beta_schedule"),
        label="target.beta_schedule",
    )
    if len(beta_schedule) < 2 or beta_schedule[0] != 0.0:
        raise ValueError("target.beta_schedule must start at 0 and contain a stage.")
    if any(
        right <= left
        for left, right in zip(beta_schedule, beta_schedule[1:])
    ):
        raise ValueError("target.beta_schedule must be strictly increasing.")

    references_raw = root.get("reference_directions")
    if not isinstance(references_raw, list) or not references_raw:
        raise ValueError("reference_directions must be a nonempty JSON array.")
    directions = tuple(
        _finite_tuple(
            direction,
            label=f"reference_directions[{index}]",
            dimension=objective_dimension,
        )
        for index, direction in enumerate(references_raw)
    )
    for direction in directions:
        if any(weight <= 0.0 for weight in direction):
            raise ValueError("Reference-direction weights must be strictly positive.")
        if not math.isclose(sum(direction), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Every reference direction must sum to one.")

    resampling = _mapping(root.get("resampling"), "resampling")
    _exact_keys(
        resampling,
        {
            "method",
            "scope",
            "ess_threshold_fraction",
            "ess_is_not_a_coverage_certificate",
        },
        "resampling",
    )
    if resampling.get("method") != "multinomial":
        raise ValueError("resampling.method must be 'multinomial'.")
    if resampling.get("scope") != "within_reference_type":
        raise ValueError("resampling.scope must be 'within_reference_type'.")
    if resampling.get("ess_is_not_a_coverage_certificate") is not True:
        raise ValueError(
            "resampling.ess_is_not_a_coverage_certificate must be true."
        )
    raw_ess_threshold = resampling.get("ess_threshold_fraction")
    if (
        isinstance(raw_ess_threshold, bool)
        or not isinstance(raw_ess_threshold, (int, float))
    ):
        raise ValueError(
            "resampling.ess_threshold_fraction must be a JSON number."
        )
    ess_threshold = float(raw_ess_threshold)
    if not math.isfinite(ess_threshold) or not (0.0 < ess_threshold <= 1.0):
        raise ValueError(
            "resampling.ess_threshold_fraction must lie in (0, 1]."
        )

    mutation = _mapping(root.get("mutation"), "mutation")
    required_mutation_keys = {
        "proposal",
        "acceptance",
        "objective_evaluation",
    }
    allowed_mutation_keys = required_mutation_keys | {
        "global_refresh_probability",
        "steps_per_stage",
    }
    if (
        not required_mutation_keys.issubset(mutation)
        or not set(mutation).issubset(allowed_mutation_keys)
    ):
        raise ValueError(
            "mutation has an unexpected shape; "
            f"missing={sorted(required_mutation_keys - set(mutation))}, "
            f"extra={sorted(set(mutation) - allowed_mutation_keys)}."
        )
    if mutation.get("acceptance") != "exact_log_domain_mh":
        raise ValueError(
            "mutation.acceptance must be 'exact_log_domain_mh'."
        )
    objective_evaluation = mutation.get("objective_evaluation")
    if objective_evaluation not in {
        "full_tour",
        EXACT_INCREMENTAL_TWO_OPT_CONTRACT,
    }:
        raise ValueError(
            "mutation.objective_evaluation must be 'full_tour' or "
            f"{EXACT_INCREMENTAL_TWO_OPT_CONTRACT!r}."
        )
    proposal = mutation.get("proposal")
    if proposal not in {
        "uniform_symmetric_two_opt",
        "local_two_opt_plus_uniform_global_refresh",
    }:
        raise ValueError("mutation.proposal is invalid.")
    if proposal == "uniform_symmetric_two_opt":
        if "global_refresh_probability" in mutation:
            raise ValueError(
                "The pure two-opt proposal must not declare "
                "mutation.global_refresh_probability."
            )
        global_refresh_probability = 0.0
    else:
        if "global_refresh_probability" not in mutation:
            raise ValueError(
                "The mixture proposal must explicitly declare "
                "mutation.global_refresh_probability."
            )
        raw_refresh_probability = mutation.get(
            "global_refresh_probability"
        )
        if (
            isinstance(raw_refresh_probability, bool)
            or not isinstance(raw_refresh_probability, (int, float))
        ):
            raise ValueError(
                "mutation.global_refresh_probability must be a JSON number."
            )
        global_refresh_probability = float(raw_refresh_probability)
        if (
            not math.isfinite(global_refresh_probability)
            or global_refresh_probability <= 0.0
            or global_refresh_probability > 1.0
        ):
            raise ValueError(
                "mutation.global_refresh_probability must lie in (0, 1]."
            )
    raw_steps = mutation.get("steps_per_stage")
    if raw_steps is None:
        mutation_steps_by_stage = None
    else:
        if not isinstance(raw_steps, list) or len(raw_steps) != (
            len(beta_schedule) - 1
        ):
            raise ValueError(
                "mutation.steps_per_stage must contain one entry per "
                "positive beta stage."
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in raw_steps
        ):
            raise ValueError(
                "mutation.steps_per_stage must contain nonnegative integers."
            )
        mutation_steps_by_stage = tuple(raw_steps)

    allocation = _mapping(root.get("particle_allocation"), "particle_allocation")
    _exact_keys(
        allocation,
        {"policy"},
        "particle_allocation",
    )
    if (
        allocation.get("policy")
        != "split_cli_population_equally_across_reference_types"
    ):
        raise ValueError(
            "particle_allocation.policy must split the CLI population equally "
            "across reference types."
        )

    reporting = _mapping(root.get("reporting"), "reporting")
    _exact_keys(
        reporting,
        {"archive_role", "archive_max_size", "cell_ledger"},
        "reporting",
    )
    if reporting.get("archive_role") != "reporting_only":
        raise ValueError("reporting.archive_role must be 'reporting_only'.")
    if (
        reporting.get("cell_ledger")
        != "untruncated_first_evaluated_representative_per_cell"
    ):
        raise ValueError(
            "reporting.cell_ledger must preserve the untruncated first "
            "evaluated representative per cell."
        )
    raw_archive_size = reporting.get("archive_max_size")
    if raw_archive_size is None:
        archive_size = None
    else:
        if (
            isinstance(raw_archive_size, bool)
            or not isinstance(raw_archive_size, int)
            or raw_archive_size <= 0
        ):
            raise ValueError("reporting.archive_max_size must be null or positive.")
        archive_size = raw_archive_size

    return ParetoSMCSpecification(
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        beta_schedule=beta_schedule,
        reference_directions=directions,
        normalized_cell_widths=widths,
        ess_threshold_fraction=ess_threshold,
        chebyshev_rho=rho,
        mutation_proposal=str(proposal),
        mutation_objective_evaluation=str(objective_evaluation),
        global_refresh_probability=global_refresh_probability,
        mutation_steps_by_stage=mutation_steps_by_stage,
        archive_max_size=archive_size,
    )


def analytic_objective_box(
    instance: MultiObjectiveTSPInstance,
) -> Tuple[ObjectiveVector, ObjectiveVector]:
    """Return the archive-independent, outward-rounded Pareto-SMC box.

    A tour objective is accumulated by repeated binary64 additions.  Even
    though ``n * edge_extremum`` is the analytic finite-domain bound, its
    rounded product can be one ulp inside a repeatedly accumulated endpoint.
    The runtime and every certificate constructor therefore share this single
    outward-rounded contract.
    """

    matrices = getattr(instance, "distance_matrices", None)
    if matrices is None:
        raise ValueError(
            "The analytic Pareto-SMC box requires explicit distance matrices."
        )
    lower_values = []
    upper_values = []
    n = instance.num_cities
    for matrix in matrices:
        off_diagonal = [
            float(matrix[i][j])
            for i in range(n)
            for j in range(n)
            if i != j
        ]
        lower = n * min(off_diagonal)
        upper = n * max(off_diagonal)
        if upper <= lower:
            upper = lower + max(1e-12, 1e-12 * abs(lower))
        lower_values.append(math.nextafter(lower, -math.inf))
        upper_values.append(math.nextafter(upper, math.inf))
    return tuple(lower_values), tuple(upper_values)


def original_unit_cell_widths(
    instance: MultiObjectiveTSPInstance,
    specification: ParetoSMCSpecification,
) -> ObjectiveVector:
    """Convert the frozen normalized widths using the analytic instance box."""

    lower, upper = analytic_objective_box(instance)
    return tuple(
        width * (high - low)
        for width, low, high in zip(
            specification.normalized_cell_widths,
            lower,
            upper,
        )
    )
