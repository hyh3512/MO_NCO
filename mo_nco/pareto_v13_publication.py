from __future__ import annotations

"""Staged v13 pilot-freeze-authorize-confirm publication protocol.

The orchestration deliberately has two public execution calls.  Pilot output
is frozen into a hash-addressed envelope first.  Confirm cannot be launched by
this module until an Ed25519 receipt over that exact envelope has been
verified.  The signature authenticates external authorization; it does not,
by itself, establish an independently timestamped wall-clock ordering.
"""

import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from .archive import ArchiveEntry, ParetoArchive
from .instance import MultiObjectiveTSPInstance, instance_sha256
from .pareto_execution_contract import (
    DOMAIN_SEPARATED_SEED_SCHEMA_V1,
    FullTypeSweepCheckpointVerification,
    verify_full_type_sweep_checkpoints,
)
from .pareto_fixed_reference_spec import (
    FIXED_REFERENCE_SPEC_SCHEMA_V2,
    FixedReferenceCertificateSpecification,
    load_fixed_reference_certificate_specification,
)
from .pareto_fixed_schedule_certificate import (
    build_regeneration_pilot_plan_commitment_from_spec,
    certify_fixed_schedule_reference_metrics_from_spec,
)
from .pareto_fixed_schedule_experiment import (
    FixedScheduleExecutionPlan,
    prepare_fixed_schedule_execution,
    run_fixed_schedule_stream,
)
from .pareto_preconfirm_receipt import (
    PreconfirmReceiptBindings,
    VerifiedPreconfirmReceipt,
    verify_preconfirm_receipt,
)
from .pareto_regeneration_certificate import (
    AssignmentPilotNonemptinessPreflight,
    RefreshRequirement,
    assignment_pilot_nonemptiness_preflight,
    minimum_refresh_for_assigned_cells,
    target_normalizer_lower_bound,
)
from .pareto_sparse_compression_certificate import (
    SparseFiniteReferenceCompressionCertificate,
    certify_sparse_finite_reference_compression,
)
from .pareto_sparse_reference import SparseReferenceCover
from .pareto_smc_spec import (
    ParetoSMCSpecification,
    analytic_objective_box,
    load_pareto_smc_specification,
    original_unit_cell_widths,
)
from .pareto_v13_spec import (
    V13ProtocolSpecification,
    load_v13_protocol_specification,
)
from .sampler import Diagnostic, OptimizationResult


V13_FREEZE_ENVELOPE_SCHEMA = (
    "pareto_smc_v13_pilot_freeze_envelope_v2"
)
V13_CONFIRM_CONTRACT_SCHEMA = "pareto_smc_v13_confirm_contract_v1"
V13_CERTIFICATE_SCHEMA = "pareto_smc_v13_publication_certificate_v1"
V13_SEED_COMMITMENT_SCHEMA = (
    "pareto_smc_v13_confirm_seed_commitment_v1"
)
V13_PILOT_ARTIFACT_SCHEMA = "pareto_smc_v13_pilot_artifact_v2"


class V13PublicationProtocolError(ValueError):
    """Raised before confirm whenever one v13 obligation is not proved."""


def _require_source_reloaded_specifications(
    *,
    instance: MultiObjectiveTSPInstance,
    pareto_smc_specification: ParetoSMCSpecification,
    anchor_certificate_specification: (
        FixedReferenceCertificateSpecification
    ),
    full_reference_specification: (
        FixedReferenceCertificateSpecification
    ),
    protocol_specification: V13ProtocolSpecification,
) -> None:
    """Reject dataclass fields that are not backed by their recorded files."""

    try:
        reloaded_smc = load_pareto_smc_specification(
            pareto_smc_specification.path,
            objective_dimension=instance.num_objectives,
        )
        reloaded_anchor = (
            load_fixed_reference_certificate_specification(
                anchor_certificate_specification.path,
                objective_dimension=instance.num_objectives,
                instance=instance,
            )
        )
        reloaded_full = (
            load_fixed_reference_certificate_specification(
                full_reference_specification.path,
                objective_dimension=instance.num_objectives,
                instance=instance,
            )
        )
        reloaded_protocol = load_v13_protocol_specification(
            protocol_specification.path
        )
    except (OSError, TypeError, ValueError) as error:
        raise V13PublicationProtocolError(
            "A frozen specification could not be strictly reloaded from "
            "its recorded source path."
        ) from error
    for supplied, reloaded, label in (
        (
            pareto_smc_specification,
            reloaded_smc,
            "Pareto-SMC specification",
        ),
        (
            anchor_certificate_specification,
            reloaded_anchor,
            "anchor certificate specification",
        ),
        (
            full_reference_specification,
            reloaded_full,
            "full-reference certificate specification",
        ),
        (
            protocol_specification,
            reloaded_protocol,
            "v13 protocol specification",
        ),
    ):
        if supplied != reloaded:
            raise V13PublicationProtocolError(
                f"The supplied {label} is not exactly backed by its "
                "recorded source file."
            )


def _require_matching_reference_streams(
    *,
    anchor: FixedReferenceCertificateSpecification,
    full_reference: FixedReferenceCertificateSpecification,
    run_seed: int,
) -> None:
    """Require evaluation-reference stream declarations to match execution."""

    try:
        anchor_streams = anchor.stream_seeds(run_seed)
        full_reference_streams = full_reference.stream_seeds(run_seed)
    except (TypeError, ValueError) as error:
        raise V13PublicationProtocolError(
            "A fixed-reference stream declaration does not cover the "
            "requested paired run seed."
        ) from error
    if full_reference_streams != anchor_streams:
        raise V13PublicationProtocolError(
            "The full-reference pilot/confirm stream declaration differs "
            "from the execution-anchor stream declaration."
        )


@dataclass(frozen=True)
class V13PilotFreezeResult:
    """In-memory pilot artifact that can be externally authorized."""

    execution_plan: FixedScheduleExecutionPlan
    protocol_specification: V13ProtocolSpecification
    full_reference_specification: FixedReferenceCertificateSpecification
    sparse_compression_certificate: (
        SparseFiniteReferenceCompressionCertificate
    )
    pilot: OptimizationResult
    base_pilot_plan_commitment: Mapping[str, object]
    assignment_preflight: AssignmentPilotNonemptinessPreflight
    refresh_cost_certificate: "V13RefreshCostCertificate"
    pilot_checkpoint_verification: FullTypeSweepCheckpointVerification
    confirm_contract_sha256: str
    confirm_seed_commitment_sha256: str
    pilot_result_payload_sha256: str
    pilot_terminal_support_sha256: str
    freeze_envelope: Mapping[str, object]
    freeze_envelope_sha256: str
    receipt_bindings: PreconfirmReceiptBindings


@dataclass(frozen=True)
class V13PilotConfirmResult:
    pilot: OptimizationResult
    confirm: OptimizationResult
    certificate: Mapping[str, object]
    verified_receipt: VerifiedPreconfirmReceipt


@dataclass(frozen=True)
class V13RefreshCostCertificate:
    """Minimum refresh required by the frozen confirm-risk design.

    This is a joint certificate-risk/evaluation-cost result only.  It is not
    a theorem that the minimum-refresh point maximizes optimization quality.
    """

    requirement: RefreshRequirement
    configured_global_refresh_probability: float
    configured_refresh_sufficiency_gate: str
    certificate_cost_minimality_gate: str
    expected_global_refresh_proposals_per_stream: float
    minimum_expected_global_refresh_proposals_per_stream: float | None
    certificate_cost_joint_design_theorem: str
    optimizer_quality_joint_design_theorem_established: bool

    def to_jsonable(self) -> dict[str, object]:
        return {
            "requirement": asdict(self.requirement),
            "configured_global_refresh_probability": (
                self.configured_global_refresh_probability
            ),
            "configured_refresh_sufficiency_gate": (
                self.configured_refresh_sufficiency_gate
            ),
            "certificate_cost_minimality_gate": (
                self.certificate_cost_minimality_gate
            ),
            "expected_global_refresh_proposals_per_stream": (
                self.expected_global_refresh_proposals_per_stream
            ),
            "minimum_expected_global_refresh_proposals_per_stream": (
                self.minimum_expected_global_refresh_proposals_per_stream
            ),
            "certificate_cost_joint_design_theorem": (
                self.certificate_cost_joint_design_theorem
            ),
            "optimizer_quality_joint_design_theorem_established": False,
        }


def _canonical_sha256(payload: object) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise V13PublicationProtocolError(
            "A v13 certificate payload is not canonical-JSON serializable."
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_json_pairs(
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


def _terminal_support_sha256(result: OptimizationResult) -> str:
    if len(result.particles) != len(result.objectives):
        raise V13PublicationProtocolError(
            "Terminal particles and objectives have different sizes."
        )
    particles = tuple(
        tuple(int(city) for city in tour) for tour in result.particles
    )
    objectives = tuple(
        tuple(float(value) for value in point)
        for point in result.objectives
    )
    if any(
        not math.isfinite(value)
        for point in objectives
        for value in point
    ):
        raise V13PublicationProtocolError(
            "Terminal objectives must be finite."
        )
    return _canonical_sha256(
        {"particles": particles, "objectives": objectives}
    )


def _sparse_certificate_sha256(
    certificate: SparseFiniteReferenceCompressionCertificate,
) -> str:
    if not isinstance(
        certificate,
        SparseFiniteReferenceCompressionCertificate,
    ):
        raise V13PublicationProtocolError(
            "sparse_compression_certificate has the wrong type."
        )
    return _canonical_sha256(certificate.to_jsonable())


def _require_v2_runtime_witnesses(
    specification: FixedReferenceCertificateSpecification,
    *,
    label: str,
) -> None:
    if not isinstance(
        specification,
        FixedReferenceCertificateSpecification,
    ):
        raise V13PublicationProtocolError(
            f"{label} must be a loaded fixed-reference specification."
        )
    if specification.schema != FIXED_REFERENCE_SPEC_SCHEMA_V2:
        raise V13PublicationProtocolError(
            f"{label} must use the witness-bound v2 schema."
        )
    if (
        not specification.reference_feasibility_verified_by_runtime
        or not specification.reference_witnesses
        or specification.reference_witness_payload_sha256 is None
    ):
        raise V13PublicationProtocolError(
            f"{label} must have exact runtime-verified witness tours."
        )


def _require_full_reference_metric_nonvacuity(
    instance: MultiObjectiveTSPInstance,
    specification: FixedReferenceCertificateSpecification,
) -> None:
    lower, upper = analytic_objective_box(instance)
    spans = tuple(
        high - low for low, high in zip(lower, upper)
    )
    p_value = specification.igd_p
    box_diameter = sum(
        span**p_value for span in spans
    ) ** (1.0 / p_value)
    hv_box_volume = math.prod(
        reference_value - low
        for reference_value, low in zip(
            specification.hv_reference,
            lower,
        )
    )
    if (
        box_diameter <= 0.0
        or hv_box_volume <= 0.0
        or specification.max_igd_bound >= box_diameter
        or specification.max_hv_deficit_bound >= hv_box_volume
    ):
        raise V13PublicationProtocolError(
            "The full-reference metric tolerances are vacuous relative to "
            "the frozen analytic objective box."
        )


def _checkpoint_verification(
    result: OptimizationResult,
    *,
    execution_plan: FixedScheduleExecutionPlan,
) -> FullTypeSweepCheckpointVerification:
    metadata = result.metadata
    stage_ledger = metadata.get("stage_ledger")
    if not isinstance(stage_ledger, (list, tuple)):
        raise V13PublicationProtocolError(
            "The stream is missing its stage ledger."
        )
    diagnostics = tuple(
        diagnostic.iteration for diagnostic in result.diagnostics
    )
    verification = verify_full_type_sweep_checkpoints(
        stage_ledger=stage_ledger,
        num_reference_types=execution_plan.reference_count,
        particles_per_reference=(
            execution_plan.particles_per_reference
        ),
        total_evaluations=execution_plan.evaluations_per_stream,
        checkpoint_period=execution_plan.anytime_checkpoint_period,
        diagnostic_iterations=diagnostics,
    )
    if verification.gate != "PASS":
        raise V13PublicationProtocolError(
            "The requested checkpoint grid is not a complete-type-sweep "
            f"grid: {verification.reasons!r}."
        )
    if (
        metadata.get("stage_ledger_hash")
        != verification.stage_ledger_sha256
    ):
        raise V13PublicationProtocolError(
            "The published stage-ledger hash does not match the verified "
            "ledger."
        )
    return verification


def _verify_terminal_result_against_instance(
    result: OptimizationResult,
    *,
    instance: MultiObjectiveTSPInstance,
    execution_plan: FixedScheduleExecutionPlan,
) -> None:
    expected_total = execution_plan.total_particles_per_stream
    if (
        len(result.particles) != expected_total
        or len(result.objectives) != expected_total
    ):
        raise V13PublicationProtocolError(
            "Terminal support does not match the frozen typed particle count."
        )
    for index, (tour, objective) in enumerate(
        zip(result.particles, result.objectives)
    ):
        if instance.evaluate(tour) != tuple(objective):
            raise V13PublicationProtocolError(
                f"Terminal objective {index} does not exactly match its tour."
            )
    for index, entry in enumerate(result.archive.entries):
        if instance.evaluate(entry.tour) != tuple(entry.objectives):
            raise V13PublicationProtocolError(
                f"Archive objective {index} does not exactly match its tour."
            )

    metadata = result.metadata
    particles_per_type = execution_plan.particles_per_reference
    type_count = execution_plan.reference_count
    if (
        metadata.get("particles_per_reference") != particles_per_type
        or metadata.get("num_reference_types") != type_count
        or metadata.get("evaluations_used")
        != execution_plan.evaluations_per_stream
    ):
        raise V13PublicationProtocolError(
            "Terminal metadata differs from the frozen typed budget."
        )
    lower = tuple(float(value) for value in metadata.get(
        "objective_lower_bounds",
        (),
    ))
    widths = tuple(float(value) for value in metadata.get("epsilon", ()))
    counts = tuple(
        int(value) for value in metadata.get("epsilon_cell_counts", ())
    )
    if not lower or not (
        len(lower) == len(widths) == len(counts)
    ):
        raise V13PublicationProtocolError(
            "Terminal metadata has a malformed epsilon-cell contract."
        )
    raw_weights = metadata.get(
        "final_normalized_weights_by_reference"
    )
    raw_cells = metadata.get("final_epsilon_cells_by_reference")
    raw_masses = metadata.get(
        "final_epsilon_cell_masses_by_reference"
    )
    if not all(
        isinstance(value, (list, tuple))
        and len(value) == type_count
        for value in (raw_weights, raw_cells, raw_masses)
    ):
        raise V13PublicationProtocolError(
            "Terminal metadata lacks typed weights, cells, or masses."
        )

    computed_cells = []
    computed_masses = []
    for type_index in range(type_count):
        start = type_index * particles_per_type
        stop = start + particles_per_type
        objectives = result.objectives[start:stop]
        weights = tuple(float(value) for value in raw_weights[type_index])
        if (
            len(weights) != particles_per_type
            or any(
                not math.isfinite(value) or value < 0.0
                for value in weights
            )
        ):
            raise V13PublicationProtocolError(
                "Terminal typed weights are malformed."
            )
        group_cells = []
        mass_by_cell: dict[tuple[int, ...], float] = {}
        for objective, weight in zip(objectives, weights):
            cell = tuple(
                min(
                    count - 1,
                    int(
                        math.floor(
                            max(0.0, value - low) / width
                        )
                    ),
                )
                for value, low, width, count in zip(
                    objective,
                    lower,
                    widths,
                    counts,
                )
            )
            group_cells.append(cell)
            mass_by_cell[cell] = mass_by_cell.get(cell, 0.0) + weight
        computed_cells.append(tuple(group_cells))
        computed_masses.append(
            tuple(
                {"epsilon_cell": cell, "mass": mass}
                for cell, mass in sorted(mass_by_cell.items())
            )
        )
    if _canonical_sha256(tuple(computed_cells)) != _canonical_sha256(
        raw_cells
    ) or _canonical_sha256(tuple(computed_masses)) != _canonical_sha256(
        raw_masses
    ):
        raise V13PublicationProtocolError(
            "Terminal typed cell or mass metadata does not match the "
            "terminal objectives and weights."
        )


def _confirm_contract(
    *,
    execution_plan: FixedScheduleExecutionPlan,
    protocol: V13ProtocolSpecification,
    full_reference: FixedReferenceCertificateSpecification,
    sparse_certificate_sha256: str,
    refresh_cost_certificate: V13RefreshCostCertificate,
) -> dict[str, object]:
    seed_contract = execution_plan.confirm_seed_contract
    if seed_contract is None:
        raise V13PublicationProtocolError(
            "The confirm stream lacks a domain-separated seed contract."
        )
    return {
        "schema": V13_CONFIRM_CONTRACT_SCHEMA,
        "v13_protocol_specification_sha256": protocol.sha256,
        "instance_sha256": protocol.instance_sha256,
        "pareto_smc_specification_sha256": (
            protocol.pareto_smc_specification_sha256
        ),
        "anchor_certificate_specification_sha256": (
            protocol.anchor_certificate_specification_sha256
        ),
        "full_reference_certificate_specification_sha256": (
            full_reference.sha256
        ),
        "sparse_compression_certificate_sha256": (
            sparse_certificate_sha256
        ),
        "run_seed": execution_plan.run_seed,
        "particles_per_reference": (
            execution_plan.particles_per_reference
        ),
        "reference_type_count": execution_plan.reference_count,
        "evaluations_per_stream": (
            execution_plan.evaluations_per_stream
        ),
        "requested_full_sweep_checkpoints": (
            protocol.requested_full_sweep_checkpoints
        ),
        "refresh_cost_certificate_sha256": _canonical_sha256(
            refresh_cost_certificate.to_jsonable()
        ),
        "confirm_seed_contract": seed_contract.metadata(),
    }


def _refresh_cost_certificate(
    *,
    protocol: V13ProtocolSpecification,
    execution_plan: FixedScheduleExecutionPlan,
) -> V13RefreshCostCertificate:
    anchor = execution_plan.certificate_specification
    budgets = protocol.confirm_failure_budgets_by_anchor_cell
    allocated_budget_exact = sum(
        (Fraction.from_float(value) for value in budgets),
        Fraction(0),
    )
    if allocated_budget_exact > Fraction.from_float(
        anchor.confirm_failure_budget
    ):
        raise V13PublicationProtocolError(
            "The per-anchor confirm failure budgets must sum to no more than "
            "the base certificate confirm failure budget."
        )
    smc = execution_plan.pareto_smc_specification
    mutation_steps = smc.mutation_steps_by_stage
    if mutation_steps is None or not mutation_steps:
        raise V13PublicationProtocolError(
            "The refresh-cost certificate requires fixed positive stages."
        )
    normalizer_lower = target_normalizer_lower_bound(
        smc.beta_schedule[-1],
        1.0 + smc.chebyshev_rho,
    )
    requirement = minimum_refresh_for_assigned_cells(
        target_mass_lower_bounds=(
            protocol.desired_target_mass_lower_bounds_by_anchor_cell
        ),
        cell_failure_budgets=budgets,
        particles=execution_plan.particles_per_reference,
        terminal_steps=mutation_steps[-1],
        normalizer_lower_bound=normalizer_lower,
    )
    configured = smc.global_refresh_probability
    minimum = requirement.minimum_global_refresh_probability
    sufficient = bool(
        requirement.feasible
        and minimum is not None
        and configured >= minimum
    )
    if not sufficient:
        raise V13PublicationProtocolError(
            "The configured global-refresh probability is insufficient for "
            "the frozen desired masses and per-cell confirm risks."
        )
    mutation_attempts = (
        execution_plan.total_particles_per_stream * sum(mutation_steps)
    )
    configured_expected = configured * mutation_attempts
    minimum_expected = (
        None if minimum is None else minimum * mutation_attempts
    )
    cost_minimal = configured == minimum
    return V13RefreshCostCertificate(
        requirement=requirement,
        configured_global_refresh_probability=configured,
        configured_refresh_sufficiency_gate="PASS",
        certificate_cost_minimality_gate=(
            "PASS" if cost_minimal else "NOT_MINIMAL"
        ),
        expected_global_refresh_proposals_per_stream=configured_expected,
        minimum_expected_global_refresh_proposals_per_stream=(
            minimum_expected
        ),
        certificate_cost_joint_design_theorem=(
            "among fixed schedules with the same desired target-mass lower "
            "bounds, per-cell confirm risks, terminal steps, particles, and "
            "normalizer lower bound, the returned binary64 gamma is the "
            "smallest value satisfying the direct regeneration miss "
            "certificates and therefore minimizes expected refresh proposals"
        ),
        optimizer_quality_joint_design_theorem_established=False,
    )


def _assignment_preflight_from_commitment(
    *,
    base_commitment: Mapping[str, object],
    protocol: V13ProtocolSpecification,
    execution_plan: FixedScheduleExecutionPlan,
) -> AssignmentPilotNonemptinessPreflight:
    assignments_raw = base_commitment.get("cell_assignments")
    if not isinstance(assignments_raw, (list, tuple)):
        raise V13PublicationProtocolError(
            "The base pilot commitment lacks cell assignments."
        )
    assignments = tuple(assignments_raw)
    if any(not isinstance(row, Mapping) for row in assignments):
        raise V13PublicationProtocolError(
            "The base pilot commitment has malformed cell assignments."
        )
    cell_count = len(assignments)
    if (
        cell_count == 0
        or len(
            protocol.desired_target_mass_lower_bounds_by_anchor_cell
        )
        != cell_count
        or len(protocol.pilot_failure_budgets_by_anchor_cell)
        != cell_count
    ):
        raise V13PublicationProtocolError(
            "The assignment-preflight vectors do not match the distinct "
            "anchor-cell count."
        )
    type_count = execution_plan.reference_count
    anchor = execution_plan.certificate_specification
    expected_pair_budget = (
        anchor.pilot_failure_budget / (type_count * cell_count)
    )
    if any(
        not math.isclose(
            budget,
            expected_pair_budget,
            rel_tol=0.0,
            abs_tol=1e-18,
        )
        for budget in protocol.pilot_failure_budgets_by_anchor_cell
    ):
        raise V13PublicationProtocolError(
            "Every v13 preflight pilot budget must equal the base theorem's "
            "per-type/per-cell union-budget allocation."
        )
    assigned_types = tuple(
        int(row["selected_reference_type"]) for row in assignments
    )
    residual = float(base_commitment["final_residual_weight"])
    preflight = assignment_pilot_nonemptiness_preflight(
        desired_target_mass_lower_bounds_by_cell=(
            protocol.desired_target_mass_lower_bounds_by_anchor_cell
        ),
        assigned_type_by_cell=assigned_types,
        pilot_particles_by_type=tuple(
            execution_plan.particles_per_reference
            for _ in range(type_count)
        ),
        pilot_failure_budgets_by_cell=(
            protocol.pilot_failure_budgets_by_anchor_cell
        ),
        pilot_residual_weights_by_type=tuple(
            residual for _ in range(type_count)
        ),
        mutually_exclusive_cells=(
            protocol.mutually_exclusive_anchor_cells
        ),
    )
    if not preflight.feasible:
        raise V13PublicationProtocolError(
            "The adaptive pilot assignment fails its cell/type simplex "
            f"preflight: {preflight.gate}."
        )
    if any(
        float(row["pilot_target_mass_lower_bound"]) < desired
        for row, desired in zip(
            assignments,
            protocol.desired_target_mass_lower_bounds_by_anchor_cell,
        )
    ):
        raise V13PublicationProtocolError(
            "The frozen pilot assignment did not attain every predeclared "
            "target-mass lower bound."
        )
    return preflight


def _freeze_payload(
    *,
    protocol: V13ProtocolSpecification,
    base_commitment: Mapping[str, object],
    pilot_result_payload_sha256: str,
    pilot_terminal_sha256: str,
    confirm_contract_sha256: str,
    confirm_seed_commitment_sha256: str,
    sparse_certificate_sha256: str,
    full_reference: FixedReferenceCertificateSpecification,
    pilot_checkpoint: FullTypeSweepCheckpointVerification,
    assignment_preflight: AssignmentPilotNonemptinessPreflight,
    refresh_cost_certificate: V13RefreshCostCertificate,
) -> dict[str, object]:
    return {
        "schema": V13_FREEZE_ENVELOPE_SCHEMA,
        "v13_protocol_specification_sha256": protocol.sha256,
        "base_pilot_plan_commitment_sha256": base_commitment[
            "commitment_sha256"
        ],
        "pilot_result_payload_sha256": pilot_result_payload_sha256,
        "pilot_terminal_support_sha256": pilot_terminal_sha256,
        "confirm_contract_sha256": confirm_contract_sha256,
        "confirm_seed_commitment_sha256": (
            confirm_seed_commitment_sha256
        ),
        "sparse_compression_certificate_sha256": (
            sparse_certificate_sha256
        ),
        "full_reference_witness_payload_sha256": (
            full_reference.reference_witness_payload_sha256
        ),
        "pilot_checkpoint_verification": pilot_checkpoint.metadata(),
        "assignment_preflight": assignment_preflight.to_jsonable(),
        "refresh_cost_certificate": (
            refresh_cost_certificate.to_jsonable()
        ),
        "adaptive_assignment_contract": (
            "pilot-selected assignments and their simplex ledger are frozen "
            "inside this envelope before receipt authorization"
        ),
    }


def run_v13_pilot_freeze(
    instance: MultiObjectiveTSPInstance,
    *,
    pareto_smc_specification: ParetoSMCSpecification,
    anchor_certificate_specification: (
        FixedReferenceCertificateSpecification
    ),
    full_reference_specification: (
        FixedReferenceCertificateSpecification
    ),
    protocol_specification: V13ProtocolSpecification,
    sparse_cover: SparseReferenceCover,
    sparse_compression_certificate: (
        SparseFiniteReferenceCompressionCertificate
    ),
    particles_per_reference: int,
    run_seed: int = 0,
) -> V13PilotFreezeResult:
    """Run pilot and freeze every adaptive choice before authorization."""

    protocol = protocol_specification
    if not isinstance(protocol, V13ProtocolSpecification):
        raise V13PublicationProtocolError(
            "protocol_specification must be a loaded v13 specification."
        )
    anchor = anchor_certificate_specification
    full_reference = full_reference_specification
    _require_source_reloaded_specifications(
        instance=instance,
        pareto_smc_specification=pareto_smc_specification,
        anchor_certificate_specification=anchor,
        full_reference_specification=full_reference,
        protocol_specification=protocol,
    )
    _require_matching_reference_streams(
        anchor=anchor,
        full_reference=full_reference,
        run_seed=run_seed,
    )
    _require_v2_runtime_witnesses(anchor, label="anchor specification")
    _require_v2_runtime_witnesses(
        full_reference,
        label="full-reference specification",
    )
    _require_full_reference_metric_nonvacuity(
        instance,
        full_reference,
    )
    observed_instance_sha256 = instance_sha256(instance)
    required_bindings = (
        (
            observed_instance_sha256,
            protocol.instance_sha256,
            "instance",
        ),
        (
            pareto_smc_specification.sha256,
            protocol.pareto_smc_specification_sha256,
            "Pareto-SMC specification",
        ),
        (
            anchor.sha256,
            protocol.anchor_certificate_specification_sha256,
            "anchor certificate specification",
        ),
        (
            full_reference.sha256,
            protocol.full_reference_certificate_specification_sha256,
            "full-reference certificate specification",
        ),
    )
    for observed, expected, label in required_bindings:
        if observed != expected:
            raise V13PublicationProtocolError(
                f"The v13 protocol is bound to a different {label}."
            )
    for label, specification in (
        ("anchor", anchor),
        ("full-reference", full_reference),
    ):
        if specification.instance_sha256 != observed_instance_sha256:
            raise V13PublicationProtocolError(
                f"The {label} specification has a different instance hash."
            )
        if (
            specification.pareto_smc_specification_sha256
            != pareto_smc_specification.sha256
        ):
            raise V13PublicationProtocolError(
                f"The {label} specification has a different SMC hash."
            )
    if protocol.seed_derivation_schema != DOMAIN_SEPARATED_SEED_SCHEMA_V1:
        raise V13PublicationProtocolError(
            "The v13 seed derivation schema is unsupported."
        )

    expected_widths = original_unit_cell_widths(
        instance,
        pareto_smc_specification,
    )
    recomputed_sparse = certify_sparse_finite_reference_compression(
        full_reference.reference_objectives,
        sparse_cover,
        cell_width_vector=expected_widths,
        igd_p=full_reference.igd_p,
        hv_reference=full_reference.hv_reference,
    )
    if recomputed_sparse != sparse_compression_certificate:
        raise V13PublicationProtocolError(
            "The sparse compression certificate does not match the full "
            "witness-bound reference set, cover, or metric contract."
        )
    sparse_sha256 = _sparse_certificate_sha256(recomputed_sparse)
    if sparse_sha256 != protocol.sparse_compression_certificate_sha256:
        raise V13PublicationProtocolError(
            "The v13 protocol is bound to a different sparse certificate."
        )
    if tuple(sorted(recomputed_sparse.anchor_reference_set)) != tuple(
        sorted(anchor.reference_objectives)
    ):
        raise V13PublicationProtocolError(
            "The execution anchor specification is not the canonical sparse "
            "cover anchor set."
        )
    if (
        recomputed_sparse.metric_p_norm != full_reference.igd_p
        or recomputed_sparse.hv_reference != full_reference.hv_reference
    ):
        raise V13PublicationProtocolError(
            "Sparse and full-reference metric contracts differ."
        )
    if (
        recomputed_sparse.ordinary_igd_bound
        > full_reference.max_igd_bound
        or recomputed_sparse.igd_plus_bound
        > full_reference.max_igd_bound
        or recomputed_sparse.shifted_front_hv_deficit_bound
        > full_reference.max_hv_deficit_bound
    ):
        raise V13PublicationProtocolError(
            "The sparse bridge exceeds the frozen full-reference metric "
            "tolerances."
        )

    checkpoint_period = (
        protocol.requested_full_sweep_checkpoints[0]
    )
    execution_plan = prepare_fixed_schedule_execution(
        instance,
        pareto_smc_specification=pareto_smc_specification,
        certificate_specification=anchor,
        particles_per_reference=particles_per_reference,
        run_seed=run_seed,
        anytime_checkpoint_period=checkpoint_period,
        certificate_mode="regeneration",
        v13_case_identity=protocol.case_id,
    )
    expected_grid = tuple(
        range(
            checkpoint_period,
            execution_plan.evaluations_per_stream,
            checkpoint_period,
        )
    ) + (execution_plan.evaluations_per_stream,)
    if (
        protocol.requested_full_sweep_checkpoints != expected_grid
        or expected_grid[-1] != execution_plan.evaluations_per_stream
    ):
        raise V13PublicationProtocolError(
            "The frozen checkpoint list must be exactly the positive "
            "periodic grid through the per-stream endpoint."
        )

    refresh_cost_certificate = _refresh_cost_certificate(
        protocol=protocol,
        execution_plan=execution_plan,
    )
    confirm_contract = _confirm_contract(
        execution_plan=execution_plan,
        protocol=protocol,
        full_reference=full_reference,
        sparse_certificate_sha256=sparse_sha256,
        refresh_cost_certificate=refresh_cost_certificate,
    )
    confirm_contract_sha256 = _canonical_sha256(confirm_contract)
    confirm_seed_contract = execution_plan.confirm_seed_contract
    assert confirm_seed_contract is not None
    confirm_seed_commitment_sha256 = _canonical_sha256(
        {
            "schema": V13_SEED_COMMITMENT_SCHEMA,
            "confirm_seed_contract": confirm_seed_contract.metadata(),
        }
    )

    pilot = run_fixed_schedule_stream(execution_plan, stream="pilot")
    _verify_terminal_result_against_instance(
        pilot,
        instance=instance,
        execution_plan=execution_plan,
    )
    pilot_checkpoint = _checkpoint_verification(
        pilot,
        execution_plan=execution_plan,
    )
    if (
        pilot_checkpoint.requested_checkpoints
        != protocol.requested_full_sweep_checkpoints
    ):
        raise V13PublicationProtocolError(
            "The executed pilot checkpoint grid differs from the frozen grid."
        )
    base_commitment = (
        build_regeneration_pilot_plan_commitment_from_spec(
            pilot,
            anchor,
            confirm_particles_per_reference=particles_per_reference,
            run_seed=run_seed,
        )
    )
    assignment_preflight = _assignment_preflight_from_commitment(
        base_commitment=base_commitment,
        protocol=protocol,
        execution_plan=execution_plan,
    )

    pilot_result_payload_sha256 = _canonical_sha256(
        _pilot_result_payload(pilot)
    )
    pilot_terminal_sha256 = _terminal_support_sha256(pilot)
    freeze_payload = _freeze_payload(
        protocol=protocol,
        base_commitment=base_commitment,
        pilot_result_payload_sha256=pilot_result_payload_sha256,
        pilot_terminal_sha256=pilot_terminal_sha256,
        confirm_contract_sha256=confirm_contract_sha256,
        confirm_seed_commitment_sha256=(
            confirm_seed_commitment_sha256
        ),
        sparse_certificate_sha256=sparse_sha256,
        full_reference=full_reference,
        pilot_checkpoint=pilot_checkpoint,
        assignment_preflight=assignment_preflight,
        refresh_cost_certificate=refresh_cost_certificate,
    )
    freeze_sha256 = _canonical_sha256(freeze_payload)
    receipt_bindings = PreconfirmReceiptBindings(
        pilot_plan_commitment_sha256=freeze_sha256,
        pilot_result_payload_sha256=pilot_result_payload_sha256,
        pilot_terminal_support_sha256=pilot_terminal_sha256,
        certificate_specification_sha256=protocol.sha256,
        run_id=protocol.run_id,
        case_id=protocol.case_id,
        algorithm_id=protocol.algorithm_id,
        pilot_stream_id=f"{protocol.run_id}:pilot:v1",
        confirm_stream_id=f"{protocol.run_id}:confirm:v1",
        confirm_contract_sha256=confirm_contract_sha256,
        confirm_seed_commitment_sha256=(
            confirm_seed_commitment_sha256
        ),
    )
    return V13PilotFreezeResult(
        execution_plan=execution_plan,
        protocol_specification=protocol,
        full_reference_specification=full_reference,
        sparse_compression_certificate=recomputed_sparse,
        pilot=pilot,
        base_pilot_plan_commitment=dict(base_commitment),
        assignment_preflight=assignment_preflight,
        refresh_cost_certificate=refresh_cost_certificate,
        pilot_checkpoint_verification=pilot_checkpoint,
        confirm_contract_sha256=confirm_contract_sha256,
        confirm_seed_commitment_sha256=(
            confirm_seed_commitment_sha256
        ),
        pilot_result_payload_sha256=pilot_result_payload_sha256,
        pilot_terminal_support_sha256=pilot_terminal_sha256,
        freeze_envelope=freeze_payload,
        freeze_envelope_sha256=freeze_sha256,
        receipt_bindings=receipt_bindings,
    )


def _deep_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_deep_tuple(item) for item in value)
    if isinstance(value, dict):
        return {key: _deep_tuple(item) for key, item in value.items()}
    return value


def _binary64_bits(value: float) -> str:
    """Return an exact canonical JSON-safe encoding of one binary64."""

    return struct.pack(">d", float(value)).hex()


def _binary64_from_bits(value: object, *, field: str) -> float:
    if (
        not isinstance(value, str)
        or len(value) != 16
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise V13PublicationProtocolError(
            f"{field} must be exactly 16 lowercase binary64 hex digits."
        )
    return struct.unpack(">d", bytes.fromhex(value))[0]


def _diagnostic_payload(diagnostic: Diagnostic) -> dict[str, object]:
    return {
        "iteration": diagnostic.iteration,
        "temperature_binary64": _binary64_bits(diagnostic.temperature),
        "acceptance_rate_binary64": _binary64_bits(
            diagnostic.acceptance_rate
        ),
        "archive_size": diagnostic.archive_size,
        "hypervolume_2d_binary64": _binary64_bits(
            diagnostic.hypervolume_2d
        ),
        "empirical_energy_binary64": _binary64_bits(
            diagnostic.empirical_energy
        ),
        "positive_archive_jump_binary64": _binary64_bits(
            diagnostic.positive_archive_jump
        ),
        "front_binary64": tuple(
            tuple(_binary64_bits(value) for value in point)
            for point in diagnostic.front
        ),
        "elapsed_seconds_binary64": _binary64_bits(
            diagnostic.elapsed_seconds
        ),
        "replacement_attempts": diagnostic.replacement_attempts,
        "accepted_replacements": diagnostic.accepted_replacements,
        "rejected_replacements": diagnostic.rejected_replacements,
        "rejection_rate_binary64": _binary64_bits(
            diagnostic.rejection_rate
        ),
        "current_rejection_streak": (
            diagnostic.current_rejection_streak
        ),
        "max_rejection_streak": diagnostic.max_rejection_streak,
    }


def _pilot_result_payload(result: OptimizationResult) -> dict[str, object]:
    return {
        "particles": result.particles,
        "objectives": result.objectives,
        "archive_entries": tuple(
            {
                "tour": entry.tour,
                "objectives": entry.objectives,
            }
            for entry in result.archive.entries
        ),
        "diagnostics": tuple(
            _diagnostic_payload(diagnostic)
            for diagnostic in result.diagnostics
        ),
        "metadata": result.metadata,
    }


def _pilot_result_from_payload(
    payload: object,
) -> OptimizationResult:
    if not isinstance(payload, Mapping) or set(payload) != {
        "particles",
        "objectives",
        "archive_entries",
        "diagnostics",
        "metadata",
    }:
        raise V13PublicationProtocolError(
            "The pilot result artifact has an unexpected shape."
        )
    restored = _deep_tuple(dict(payload))
    particles = restored["particles"]
    objectives = restored["objectives"]
    archive_entries = restored["archive_entries"]
    diagnostic_rows = restored["diagnostics"]
    metadata = restored["metadata"]
    if (
        not isinstance(particles, tuple)
        or not isinstance(objectives, tuple)
        or len(particles) != len(objectives)
        or not isinstance(archive_entries, tuple)
        or not isinstance(diagnostic_rows, tuple)
        or not isinstance(metadata, dict)
    ):
        raise V13PublicationProtocolError(
            "The pilot result artifact contains malformed sequences."
        )
    archive_max_size = metadata.get("archive_max_size")
    if archive_max_size is not None and (
        isinstance(archive_max_size, bool)
        or not isinstance(archive_max_size, int)
        or archive_max_size <= 0
    ):
        raise V13PublicationProtocolError(
            "The persisted archive_max_size is malformed."
        )
    archive = ParetoArchive(max_size=archive_max_size)
    try:
        archive.update(
            ArchiveEntry(
                tuple(int(city) for city in row["tour"]),
                tuple(float(value) for value in row["objectives"]),
            )
            for row in archive_entries
        )
        diagnostics = tuple(
            Diagnostic(
                iteration=int(row["iteration"]),
                temperature=_binary64_from_bits(
                    row["temperature_binary64"],
                    field="diagnostic temperature",
                ),
                acceptance_rate=_binary64_from_bits(
                    row["acceptance_rate_binary64"],
                    field="diagnostic acceptance rate",
                ),
                archive_size=int(row["archive_size"]),
                hypervolume_2d=_binary64_from_bits(
                    row["hypervolume_2d_binary64"],
                    field="diagnostic hypervolume",
                ),
                empirical_energy=_binary64_from_bits(
                    row["empirical_energy_binary64"],
                    field="diagnostic empirical energy",
                ),
                positive_archive_jump=_binary64_from_bits(
                    row["positive_archive_jump_binary64"],
                    field="diagnostic positive archive jump",
                ),
                front=tuple(
                    tuple(
                        _binary64_from_bits(
                            value,
                            field="diagnostic front objective",
                        )
                        for value in point
                    )
                    for point in row["front_binary64"]
                ),
                elapsed_seconds=_binary64_from_bits(
                    row["elapsed_seconds_binary64"],
                    field="diagnostic elapsed seconds",
                ),
                replacement_attempts=int(
                    row["replacement_attempts"]
                ),
                accepted_replacements=int(
                    row["accepted_replacements"]
                ),
                rejected_replacements=int(
                    row["rejected_replacements"]
                ),
                rejection_rate=_binary64_from_bits(
                    row["rejection_rate_binary64"],
                    field="diagnostic rejection rate",
                ),
                current_rejection_streak=int(
                    row["current_rejection_streak"]
                ),
                max_rejection_streak=int(
                    row["max_rejection_streak"]
                ),
            )
            for row in diagnostic_rows
        )
        result = OptimizationResult(
            particles=tuple(
                tuple(int(city) for city in tour) for tour in particles
            ),
            objectives=tuple(
                tuple(float(value) for value in point)
                for point in objectives
            ),
            archive=archive,
            diagnostics=diagnostics,
            metadata=metadata,
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise V13PublicationProtocolError(
            "The persisted pilot result cannot be reconstructed."
        ) from error
    if _canonical_sha256(_pilot_result_payload(result)) != _canonical_sha256(
        payload
    ):
        raise V13PublicationProtocolError(
            "The reconstructed pilot result differs from its artifact."
        )
    return result


def write_v13_pilot_artifact(
    pilot_freeze: V13PilotFreezeResult,
    path: str | Path,
) -> str:
    """Persist a canonical, self-hashed pilot artifact for process handoff."""

    if not isinstance(pilot_freeze, V13PilotFreezeResult):
        raise V13PublicationProtocolError(
            "pilot_freeze must be a V13PilotFreezeResult."
        )
    observed_pilot_result_sha256 = _canonical_sha256(
        _pilot_result_payload(pilot_freeze.pilot)
    )
    if (
        observed_pilot_result_sha256
        != pilot_freeze.pilot_result_payload_sha256
    ):
        raise V13PublicationProtocolError(
            "The pilot result changed after its freeze."
        )
    plan = pilot_freeze.execution_plan
    payload = {
        "v13_protocol_specification_sha256": (
            pilot_freeze.protocol_specification.sha256
        ),
        "pareto_smc_specification_sha256": (
            plan.pareto_smc_specification.sha256
        ),
        "anchor_certificate_specification_sha256": (
            plan.certificate_specification.sha256
        ),
        "full_reference_certificate_specification_sha256": (
            pilot_freeze.full_reference_specification.sha256
        ),
        "sparse_compression_certificate_sha256": (
            _sparse_certificate_sha256(
                pilot_freeze.sparse_compression_certificate
            )
        ),
        "particles_per_reference": plan.particles_per_reference,
        "run_seed": plan.run_seed,
        "pilot_result_payload_sha256": (
            pilot_freeze.pilot_result_payload_sha256
        ),
        "pilot_result": _pilot_result_payload(pilot_freeze.pilot),
        "base_pilot_plan_commitment": dict(
            pilot_freeze.base_pilot_plan_commitment
        ),
        "freeze_envelope": dict(pilot_freeze.freeze_envelope),
        "freeze_envelope_sha256": (
            pilot_freeze.freeze_envelope_sha256
        ),
        "receipt_bindings": pilot_freeze.receipt_bindings.as_dict(),
    }
    envelope = {
        "schema": V13_PILOT_ARTIFACT_SCHEMA,
        "payload": payload,
        "payload_sha256": _canonical_sha256(payload),
    }
    encoded = json.dumps(
        envelope,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def load_v13_pilot_artifact(
    path: str | Path,
    *,
    instance: MultiObjectiveTSPInstance,
    pareto_smc_specification: ParetoSMCSpecification,
    anchor_certificate_specification: (
        FixedReferenceCertificateSpecification
    ),
    full_reference_specification: (
        FixedReferenceCertificateSpecification
    ),
    protocol_specification: V13ProtocolSpecification,
    sparse_cover: SparseReferenceCover,
    sparse_compression_certificate: (
        SparseFiniteReferenceCompressionCertificate
    ),
) -> V13PilotFreezeResult:
    """Reload and fully revalidate a pilot artifact without rerunning pilot."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise V13PublicationProtocolError(
            f"The v13 pilot artifact is missing: {resolved}"
        )
    raw = resolved.read_bytes()
    try:
        envelope = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise V13PublicationProtocolError(
            "The v13 pilot artifact is not valid strict UTF-8 JSON."
        ) from error
    if not isinstance(envelope, dict) or set(envelope) != {
        "schema",
        "payload",
        "payload_sha256",
    }:
        raise V13PublicationProtocolError(
            "The v13 pilot artifact has an unexpected envelope."
        )
    if envelope["schema"] != V13_PILOT_ARTIFACT_SCHEMA:
        raise V13PublicationProtocolError(
            "The v13 pilot artifact schema is unsupported."
        )
    canonical_raw = json.dumps(
        envelope,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if canonical_raw != raw:
        raise V13PublicationProtocolError(
            "The v13 pilot artifact must use exact canonical JSON."
        )
    payload = envelope["payload"]
    if not isinstance(payload, dict) or envelope[
        "payload_sha256"
    ] != _canonical_sha256(payload):
        raise V13PublicationProtocolError(
            "The v13 pilot artifact payload hash is invalid."
        )
    expected_payload_keys = {
        "v13_protocol_specification_sha256",
        "pareto_smc_specification_sha256",
        "anchor_certificate_specification_sha256",
        "full_reference_certificate_specification_sha256",
        "sparse_compression_certificate_sha256",
        "particles_per_reference",
        "run_seed",
        "pilot_result_payload_sha256",
        "pilot_result",
        "base_pilot_plan_commitment",
        "freeze_envelope",
        "freeze_envelope_sha256",
        "receipt_bindings",
    }
    if set(payload) != expected_payload_keys:
        raise V13PublicationProtocolError(
            "The v13 pilot artifact payload has an unexpected shape."
        )
    protocol = protocol_specification
    anchor = anchor_certificate_specification
    full_reference = full_reference_specification
    _require_source_reloaded_specifications(
        instance=instance,
        pareto_smc_specification=pareto_smc_specification,
        anchor_certificate_specification=anchor,
        full_reference_specification=full_reference,
        protocol_specification=protocol,
    )
    sparse_sha256 = _sparse_certificate_sha256(
        sparse_compression_certificate
    )
    hash_bindings = (
        (
            payload["v13_protocol_specification_sha256"],
            protocol.sha256,
        ),
        (
            payload["pareto_smc_specification_sha256"],
            pareto_smc_specification.sha256,
        ),
        (
            payload["anchor_certificate_specification_sha256"],
            anchor.sha256,
        ),
        (
            payload[
                "full_reference_certificate_specification_sha256"
            ],
            full_reference.sha256,
        ),
        (
            payload["sparse_compression_certificate_sha256"],
            sparse_sha256,
        ),
    )
    if any(observed != expected for observed, expected in hash_bindings):
        raise V13PublicationProtocolError(
            "A supplied frozen specification differs from the pilot artifact."
        )
    protocol_bindings = (
        (
            protocol.pareto_smc_specification_sha256,
            pareto_smc_specification.sha256,
        ),
        (
            protocol.anchor_certificate_specification_sha256,
            anchor.sha256,
        ),
        (
            protocol.full_reference_certificate_specification_sha256,
            full_reference.sha256,
        ),
        (
            protocol.sparse_compression_certificate_sha256,
            sparse_sha256,
        ),
    )
    if any(observed != expected for observed, expected in protocol_bindings):
        raise V13PublicationProtocolError(
            "The v13 protocol bindings differ from the supplied artifacts."
        )
    if protocol.seed_derivation_schema != DOMAIN_SEPARATED_SEED_SCHEMA_V1:
        raise V13PublicationProtocolError(
            "The v13 seed derivation schema is unsupported."
        )
    _require_v2_runtime_witnesses(anchor, label="anchor specification")
    _require_v2_runtime_witnesses(
        full_reference,
        label="full-reference specification",
    )
    _require_full_reference_metric_nonvacuity(
        instance,
        full_reference,
    )
    if instance_sha256(instance) != protocol.instance_sha256:
        raise V13PublicationProtocolError(
            "The supplied instance differs from the v13 protocol."
        )
    for label, specification in (
        ("anchor", anchor),
        ("full-reference", full_reference),
    ):
        if (
            specification.instance_sha256 != protocol.instance_sha256
            or specification.pareto_smc_specification_sha256
            != pareto_smc_specification.sha256
        ):
            raise V13PublicationProtocolError(
                f"The supplied {label} specification has inconsistent "
                "instance or SMC bindings."
            )
    widths = original_unit_cell_widths(
        instance,
        pareto_smc_specification,
    )
    recomputed_sparse = certify_sparse_finite_reference_compression(
        full_reference.reference_objectives,
        sparse_cover,
        cell_width_vector=widths,
        igd_p=full_reference.igd_p,
        hv_reference=full_reference.hv_reference,
    )
    if (
        recomputed_sparse != sparse_compression_certificate
        or sparse_sha256
        != protocol.sparse_compression_certificate_sha256
        or tuple(sorted(recomputed_sparse.anchor_reference_set))
        != tuple(sorted(anchor.reference_objectives))
    ):
        raise V13PublicationProtocolError(
            "The supplied sparse/full-reference chain differs from the "
            "pilot artifact."
        )
    if (
        recomputed_sparse.ordinary_igd_bound
        > full_reference.max_igd_bound
        or recomputed_sparse.igd_plus_bound
        > full_reference.max_igd_bound
        or recomputed_sparse.shifted_front_hv_deficit_bound
        > full_reference.max_hv_deficit_bound
    ):
        raise V13PublicationProtocolError(
            "The sparse bridge exceeds the frozen full-reference metric "
            "tolerances."
        )
    particles_per_reference = payload["particles_per_reference"]
    run_seed = payload["run_seed"]
    if (
        isinstance(particles_per_reference, bool)
        or not isinstance(particles_per_reference, int)
        or isinstance(run_seed, bool)
        or not isinstance(run_seed, int)
    ):
        raise V13PublicationProtocolError(
            "The persisted particle count or run seed is malformed."
        )
    _require_matching_reference_streams(
        anchor=anchor,
        full_reference=full_reference,
        run_seed=run_seed,
    )
    checkpoint_period = protocol.requested_full_sweep_checkpoints[0]
    plan = prepare_fixed_schedule_execution(
        instance,
        pareto_smc_specification=pareto_smc_specification,
        certificate_specification=anchor,
        particles_per_reference=particles_per_reference,
        run_seed=run_seed,
        anytime_checkpoint_period=checkpoint_period,
        certificate_mode="regeneration",
        v13_case_identity=protocol.case_id,
    )
    pilot = _pilot_result_from_payload(payload["pilot_result"])
    pilot_result_payload_sha256 = _canonical_sha256(
        _pilot_result_payload(pilot)
    )
    if (
        pilot_result_payload_sha256
        != payload["pilot_result_payload_sha256"]
    ):
        raise V13PublicationProtocolError(
            "The persisted pilot-result hash does not match its exact "
            "canonical payload."
        )
    _verify_terminal_result_against_instance(
        pilot,
        instance=instance,
        execution_plan=plan,
    )
    checkpoint = _checkpoint_verification(pilot, execution_plan=plan)
    base_commitment = (
        build_regeneration_pilot_plan_commitment_from_spec(
            pilot,
            anchor,
            confirm_particles_per_reference=particles_per_reference,
            run_seed=run_seed,
        )
    )
    if _canonical_sha256(base_commitment) != _canonical_sha256(
        payload["base_pilot_plan_commitment"]
    ):
        raise V13PublicationProtocolError(
            "The persisted pilot commitment does not match the pilot result."
        )
    preflight = _assignment_preflight_from_commitment(
        base_commitment=base_commitment,
        protocol=protocol,
        execution_plan=plan,
    )
    refresh_cost_certificate = _refresh_cost_certificate(
        protocol=protocol,
        execution_plan=plan,
    )
    confirm_contract_sha256 = _canonical_sha256(
        _confirm_contract(
            execution_plan=plan,
            protocol=protocol,
            full_reference=full_reference,
            sparse_certificate_sha256=sparse_sha256,
            refresh_cost_certificate=refresh_cost_certificate,
        )
    )
    assert plan.confirm_seed_contract is not None
    seed_commitment_sha256 = _canonical_sha256(
        {
            "schema": V13_SEED_COMMITMENT_SCHEMA,
            "confirm_seed_contract": (
                plan.confirm_seed_contract.metadata()
            ),
        }
    )
    terminal_sha256 = _terminal_support_sha256(pilot)
    expected_freeze = _freeze_payload(
        protocol=protocol,
        base_commitment=base_commitment,
        pilot_result_payload_sha256=pilot_result_payload_sha256,
        pilot_terminal_sha256=terminal_sha256,
        confirm_contract_sha256=confirm_contract_sha256,
        confirm_seed_commitment_sha256=seed_commitment_sha256,
        sparse_certificate_sha256=sparse_sha256,
        full_reference=full_reference,
        pilot_checkpoint=checkpoint,
        assignment_preflight=preflight,
        refresh_cost_certificate=refresh_cost_certificate,
    )
    expected_freeze_sha256 = _canonical_sha256(expected_freeze)
    if (
        expected_freeze_sha256 != payload["freeze_envelope_sha256"]
        or _canonical_sha256(payload["freeze_envelope"])
        != expected_freeze_sha256
    ):
        raise V13PublicationProtocolError(
            "The persisted freeze envelope does not match reconstructed "
            "pilot obligations."
        )
    bindings = PreconfirmReceiptBindings(
        pilot_plan_commitment_sha256=expected_freeze_sha256,
        pilot_result_payload_sha256=pilot_result_payload_sha256,
        pilot_terminal_support_sha256=terminal_sha256,
        certificate_specification_sha256=protocol.sha256,
        run_id=protocol.run_id,
        case_id=protocol.case_id,
        algorithm_id=protocol.algorithm_id,
        pilot_stream_id=f"{protocol.run_id}:pilot:v1",
        confirm_stream_id=f"{protocol.run_id}:confirm:v1",
        confirm_contract_sha256=confirm_contract_sha256,
        confirm_seed_commitment_sha256=seed_commitment_sha256,
    )
    if bindings.as_dict() != payload["receipt_bindings"]:
        raise V13PublicationProtocolError(
            "The persisted receipt bindings do not match the freeze."
        )
    return V13PilotFreezeResult(
        execution_plan=plan,
        protocol_specification=protocol,
        full_reference_specification=full_reference,
        sparse_compression_certificate=recomputed_sparse,
        pilot=pilot,
        base_pilot_plan_commitment=base_commitment,
        assignment_preflight=preflight,
        refresh_cost_certificate=refresh_cost_certificate,
        pilot_checkpoint_verification=checkpoint,
        confirm_contract_sha256=confirm_contract_sha256,
        confirm_seed_commitment_sha256=seed_commitment_sha256,
        pilot_result_payload_sha256=pilot_result_payload_sha256,
        pilot_terminal_support_sha256=terminal_sha256,
        freeze_envelope=expected_freeze,
        freeze_envelope_sha256=expected_freeze_sha256,
        receipt_bindings=bindings,
    )


def run_v13_confirm_from_signed_receipt(
    pilot_freeze: V13PilotFreezeResult,
    *,
    signed_receipt: bytes,
    external_signer_key_not_held_by_runner: bool,
) -> V13PilotConfirmResult:
    """Verify external authorization and only then launch confirm."""

    if not isinstance(pilot_freeze, V13PilotFreezeResult):
        raise V13PublicationProtocolError(
            "pilot_freeze must be a V13PilotFreezeResult."
        )
    protocol = pilot_freeze.protocol_specification
    fresh_pilot_result_payload_sha256 = _canonical_sha256(
        _pilot_result_payload(pilot_freeze.pilot)
    )
    if (
        fresh_pilot_result_payload_sha256
        != pilot_freeze.pilot_result_payload_sha256
    ):
        raise V13PublicationProtocolError(
            "The exact pilot result changed after pilot freeze."
        )
    if (
        _canonical_sha256(dict(pilot_freeze.freeze_envelope))
        != pilot_freeze.freeze_envelope_sha256
    ):
        raise V13PublicationProtocolError(
            "The in-memory pilot freeze envelope was modified."
        )
    if (
        _terminal_support_sha256(pilot_freeze.pilot)
        != pilot_freeze.pilot_terminal_support_sha256
    ):
        raise V13PublicationProtocolError(
            "The frozen pilot terminal support was modified."
        )
    original_plan = pilot_freeze.execution_plan
    frozen_instance = original_plan.optimizer_arguments.get("instance")
    if not isinstance(frozen_instance, MultiObjectiveTSPInstance):
        raise V13PublicationProtocolError(
            "The frozen execution plan has no valid instance."
        )
    _require_source_reloaded_specifications(
        instance=frozen_instance,
        pareto_smc_specification=(
            original_plan.pareto_smc_specification
        ),
        anchor_certificate_specification=(
            original_plan.certificate_specification
        ),
        full_reference_specification=(
            pilot_freeze.full_reference_specification
        ),
        protocol_specification=protocol,
    )
    _require_matching_reference_streams(
        anchor=original_plan.certificate_specification,
        full_reference=pilot_freeze.full_reference_specification,
        run_seed=original_plan.run_seed,
    )
    fresh_plan = prepare_fixed_schedule_execution(
        frozen_instance,
        pareto_smc_specification=(
            original_plan.pareto_smc_specification
        ),
        certificate_specification=(
            original_plan.certificate_specification
        ),
        particles_per_reference=(
            original_plan.particles_per_reference
        ),
        run_seed=original_plan.run_seed,
        anytime_checkpoint_period=(
            original_plan.anytime_checkpoint_period
        ),
        certificate_mode="regeneration",
        v13_case_identity=protocol.case_id,
    )
    fresh_refresh = _refresh_cost_certificate(
        protocol=protocol,
        execution_plan=fresh_plan,
    )
    if (
        _canonical_sha256(fresh_refresh.to_jsonable())
        != _canonical_sha256(
            pilot_freeze.refresh_cost_certificate.to_jsonable()
        )
    ):
        raise V13PublicationProtocolError(
            "The refresh-cost design changed after pilot freeze."
        )
    fresh_confirm_contract_sha256 = _canonical_sha256(
        _confirm_contract(
            execution_plan=fresh_plan,
            protocol=protocol,
            full_reference=pilot_freeze.full_reference_specification,
            sparse_certificate_sha256=_sparse_certificate_sha256(
                pilot_freeze.sparse_compression_certificate
            ),
            refresh_cost_certificate=fresh_refresh,
        )
    )
    if (
        fresh_confirm_contract_sha256
        != pilot_freeze.confirm_contract_sha256
    ):
        raise V13PublicationProtocolError(
            "The confirm execution contract changed after pilot freeze."
        )
    _verify_terminal_result_against_instance(
        pilot_freeze.pilot,
        instance=frozen_instance,
        execution_plan=fresh_plan,
    )
    fresh_pilot_checkpoint = _checkpoint_verification(
        pilot_freeze.pilot,
        execution_plan=fresh_plan,
    )
    fresh_base_commitment = (
        build_regeneration_pilot_plan_commitment_from_spec(
            pilot_freeze.pilot,
            fresh_plan.certificate_specification,
            confirm_particles_per_reference=(
                fresh_plan.particles_per_reference
            ),
            run_seed=fresh_plan.run_seed,
        )
    )
    if _canonical_sha256(fresh_base_commitment) != _canonical_sha256(
        pilot_freeze.base_pilot_plan_commitment
    ):
        raise V13PublicationProtocolError(
            "The pilot commitment changed after pilot freeze."
        )
    fresh_preflight = _assignment_preflight_from_commitment(
        base_commitment=fresh_base_commitment,
        protocol=protocol,
        execution_plan=fresh_plan,
    )
    if fresh_plan.confirm_seed_contract is None:
        raise V13PublicationProtocolError(
            "The reconstructed confirm seed contract is missing."
        )
    fresh_seed_commitment_sha256 = _canonical_sha256(
        {
            "schema": V13_SEED_COMMITMENT_SCHEMA,
            "confirm_seed_contract": (
                fresh_plan.confirm_seed_contract.metadata()
            ),
        }
    )
    if (
        fresh_seed_commitment_sha256
        != pilot_freeze.confirm_seed_commitment_sha256
    ):
        raise V13PublicationProtocolError(
            "The confirm seed commitment changed after pilot freeze."
        )
    fresh_freeze_payload = _freeze_payload(
        protocol=protocol,
        base_commitment=fresh_base_commitment,
        pilot_result_payload_sha256=(
            fresh_pilot_result_payload_sha256
        ),
        pilot_terminal_sha256=(
            pilot_freeze.pilot_terminal_support_sha256
        ),
        confirm_contract_sha256=fresh_confirm_contract_sha256,
        confirm_seed_commitment_sha256=fresh_seed_commitment_sha256,
        sparse_certificate_sha256=_sparse_certificate_sha256(
            pilot_freeze.sparse_compression_certificate
        ),
        full_reference=pilot_freeze.full_reference_specification,
        pilot_checkpoint=fresh_pilot_checkpoint,
        assignment_preflight=fresh_preflight,
        refresh_cost_certificate=fresh_refresh,
    )
    if (
        _canonical_sha256(fresh_freeze_payload)
        != pilot_freeze.freeze_envelope_sha256
        or _canonical_sha256(pilot_freeze.freeze_envelope)
        != pilot_freeze.freeze_envelope_sha256
    ):
        raise V13PublicationProtocolError(
            "The reconstructed pilot freeze changed before receipt "
            "verification."
        )
    fresh_receipt_bindings = PreconfirmReceiptBindings(
        pilot_plan_commitment_sha256=(
            pilot_freeze.freeze_envelope_sha256
        ),
        pilot_result_payload_sha256=(
            fresh_pilot_result_payload_sha256
        ),
        pilot_terminal_support_sha256=(
            pilot_freeze.pilot_terminal_support_sha256
        ),
        certificate_specification_sha256=protocol.sha256,
        run_id=protocol.run_id,
        case_id=protocol.case_id,
        algorithm_id=protocol.algorithm_id,
        pilot_stream_id=f"{protocol.run_id}:pilot:v1",
        confirm_stream_id=f"{protocol.run_id}:confirm:v1",
        confirm_contract_sha256=fresh_confirm_contract_sha256,
        confirm_seed_commitment_sha256=(
            fresh_seed_commitment_sha256
        ),
    )
    if (
        fresh_receipt_bindings.as_dict()
        != pilot_freeze.receipt_bindings.as_dict()
    ):
        raise V13PublicationProtocolError(
            "The receipt bindings changed after pilot freeze."
        )

    # This call is deliberately the last operation before confirm launch.
    # Passing True for the order fact is justified by this control-flow seam;
    # key separation remains an external fact supplied by orchestration.
    verified_receipt = verify_preconfirm_receipt(
        signed_receipt,
        frozen_signer_public_key_raw=protocol.signer_public_key_raw,
        expected_bindings=fresh_receipt_bindings,
        expected_signer_key_id=protocol.signer_key_id,
        external_signer_key_not_held_by_runner=(
            external_signer_key_not_held_by_runner
        ),
        receipt_verified_before_confirm_start=True,
    )
    confirm = run_fixed_schedule_stream(
        fresh_plan,
        stream="confirm",
    )
    _verify_terminal_result_against_instance(
        confirm,
        instance=frozen_instance,
        execution_plan=fresh_plan,
    )
    confirm_checkpoint = _checkpoint_verification(
        confirm,
        execution_plan=fresh_plan,
    )
    if (
        confirm_checkpoint.requested_checkpoints
        != protocol.requested_full_sweep_checkpoints
    ):
        raise V13PublicationProtocolError(
            "The executed confirm checkpoint grid differs from the frozen "
            "grid."
        )

    base_certificate = (
        certify_fixed_schedule_reference_metrics_from_spec(
            pilot_freeze.pilot,
            confirm,
            fresh_plan.certificate_specification,
            run_seed=fresh_plan.run_seed,
            certificate_mode="regeneration",
            pilot_plan_commitment=(
                pilot_freeze.base_pilot_plan_commitment
            ),
            pilot_plan_commitment_preconfirm_order_attested_by_runner=True,
        )
    )
    sparse = pilot_freeze.sparse_compression_certificate
    full_reference = pilot_freeze.full_reference_specification
    sparse_metric_gate = bool(
        sparse.ordinary_igd_bound <= full_reference.max_igd_bound
        and sparse.igd_plus_bound <= full_reference.max_igd_bound
        and sparse.shifted_front_hv_deficit_bound
        <= full_reference.max_hv_deficit_bound
    )
    output_size = int(base_certificate["cell_cover_archive_size"])
    sparse_archive_gate = bool(
        base_certificate.get("certified_archive_gate") == "PASS"
        and output_size <= sparse.archive_cardinality_bound
    )
    v13_conditional_content_pass = bool(
        base_certificate.get("formal_packet_gate") == "PASS"
        and verified_receipt.authorization_gate == "PASS"
        and pilot_freeze.assignment_preflight.feasible
        and pilot_freeze.refresh_cost_certificate
        .configured_refresh_sufficiency_gate
        == "PASS"
        and pilot_freeze.pilot_checkpoint_verification.gate == "PASS"
        and confirm_checkpoint.gate == "PASS"
        and full_reference.reference_feasibility_verified_by_runtime
        and sparse_metric_gate
        and sparse_archive_gate
    )

    certificate = dict(base_certificate)
    certificate.update(
        {
            "schema": V13_CERTIFICATE_SCHEMA,
            "base_certificate_schema": base_certificate.get("schema"),
            "base_formal_packet_gate": base_certificate.get(
                "formal_packet_gate"
            ),
            "v13_protocol_specification_path": str(protocol.path),
            "v13_protocol_specification_sha256": protocol.sha256,
            "v13_freeze_envelope": dict(
                pilot_freeze.freeze_envelope
            ),
            "v13_freeze_envelope_sha256": (
                pilot_freeze.freeze_envelope_sha256
            ),
            "cryptographic_preconfirm_receipt_verification_gate": "PASS",
            "external_preconfirm_commitment_receipt_gate": (
                "CONDITIONAL_ON_DECLARED_KEY_SEPARATION_AND_NO_PREVIEW"
            ),
            "preconfirm_receipt_sha256": (
                verified_receipt.receipt_sha256
            ),
            "preconfirm_receipt_signed_payload_sha256": (
                verified_receipt.signed_payload_sha256
            ),
            "preconfirm_receipt_signer_public_key_sha256": (
                verified_receipt.signer_public_key_sha256
            ),
            "preconfirm_receipt_issued_at_is_metadata_only": True,
            "pilot_plan_commitment_preconfirm_timing_"
            "independently_verified": False,
            "independent_wall_clock_timing_proof_gate": "NOT_ESTABLISHED",
            "independently_auditable_preconfirm_freeze_gate": (
                "CONDITIONAL_ON_DECLARED_KEY_SEPARATION_NO_PREVIEW_AND_"
                "VERIFY_BEFORE_LAUNCH"
            ),
            "external_authorization_assumptions": (
                "the signer private key is outside runner control and this "
                "function verifies the receipt immediately before confirm; "
                "the runner has not previewed the frozen confirm stream; "
                "the signed timestamp is not an independent time proof"
            ),
            "v13_domain_separated_seed_gate": "PASS",
            "v13_seed_derivation_schema": (
                protocol.seed_derivation_schema
            ),
            "pilot_full_type_sweep_checkpoint": (
                pilot_freeze.pilot_checkpoint_verification.metadata()
            ),
            "confirm_full_type_sweep_checkpoint": (
                confirm_checkpoint.metadata()
            ),
            "formal_full_type_sweep_checkpoint_gate": "PASS",
            "assignment_pilot_nonemptiness_preflight": (
                pilot_freeze.assignment_preflight.to_jsonable()
            ),
            "assignment_pilot_nonemptiness_preflight_gate": "PASS",
            "global_refresh_certificate_cost_design": (
                pilot_freeze.refresh_cost_certificate.to_jsonable()
            ),
            "global_refresh_certificate_sufficiency_gate": "PASS",
            "global_refresh_certificate_cost_minimality_gate": (
                pilot_freeze.refresh_cost_certificate
                .certificate_cost_minimality_gate
            ),
            "global_refresh_optimizer_quality_joint_theorem_gate": (
                "NOT_ESTABLISHED"
            ),
            "full_reference_certificate_specification_path": str(
                full_reference.path
            ),
            "full_reference_certificate_specification_sha256": (
                full_reference.sha256
            ),
            "full_reference_witness_payload_sha256": (
                full_reference.reference_witness_payload_sha256
            ),
            "full_reference_feasibility_gate": "PASS",
            "reference_feasibility_gate": "PASS",
            "reference_feasibility_verified_by_runtime": True,
            "sparse_compression_certificate": sparse.to_jsonable(),
            "sparse_compression_certificate_sha256": (
                _sparse_certificate_sha256(sparse)
            ),
            "sparse_full_reference_metric_gate": (
                "PASS" if sparse_metric_gate else "FAIL"
            ),
            "sparse_certified_archive_gate": (
                "PASS" if sparse_archive_gate else "FAIL"
            ),
            "certified_output_scope": (
                "retained feasible same-cell witness support before "
                "nondominated filtering for ordinary IGD; its nondominated "
                "view for IGD+ and shifted-front HV deficit"
            ),
            "certified_archive_cardinality_bound": (
                sparse.archive_cardinality_bound
            ),
            "reference_count_independent_universal_compression_claimed": (
                False
            ),
            "claim_scope": (
                "typed Pareto-SMC with a hash-frozen adaptive pilot plan, "
                "externally authenticated pre-confirm authorization, exact "
                "witness-bound finite reference set, and direct sparse "
                "finite-reference metric certificate; not the unknown "
                "complete Pareto front"
            ),
            "v13_conditional_certificate_content_gate": (
                "PASS" if v13_conditional_content_pass else "FAIL"
            ),
            "v13_formal_packet_gate": "NOT_ESTABLISHED",
            "formal_packet_gate": "FAIL",
            "publication_certificate_packet_gate": "NOT_ESTABLISHED",
            "publication_certificate_packet_blockers": (
                (
                    "conditional certificate content did not pass",
                )
                if not v13_conditional_content_pass
                else (
                    "independent pilot-receipt-confirm ordering proof is "
                    "not established",
                    "absence of confirm-seed preview or selective rerun is "
                    "not independently established",
                    "external signer key separation is a declared fact, "
                    "not a runner-verifiable fact",
                )
            ),
            "unconditional_preconfirm_false_selection_control_claimed": (
                False
            ),
            "algorithm_competitiveness_gate": "NOT_RUN",
            "top_tier_submission_gate": "HOLD",
            "scalable_claim_authorized": False,
            "state_of_the_art_claim_authorized": False,
        }
    )
    certificate["v13_certificate_sha256"] = _canonical_sha256(
        certificate
    )
    return V13PilotConfirmResult(
        pilot=pilot_freeze.pilot,
        confirm=confirm,
        certificate=certificate,
        verified_receipt=verified_receipt,
    )


__all__ = [
    "V13_CERTIFICATE_SCHEMA",
    "V13_FREEZE_ENVELOPE_SCHEMA",
    "V13PublicationProtocolError",
    "V13PilotConfirmResult",
    "V13PilotFreezeResult",
    "V13RefreshCostCertificate",
    "load_v13_pilot_artifact",
    "run_v13_confirm_from_signed_receipt",
    "run_v13_pilot_freeze",
    "write_v13_pilot_artifact",
]
