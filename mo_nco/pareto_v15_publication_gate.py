from __future__ import annotations

"""Component claim-hygiene gate with the composed P0 gate held closed.

Caller-created dataclasses and Booleans are not accepted as raw-artifact,
external-control, formalization, or competitive-evidence verification.
"""

from dataclasses import asdict, dataclass
from typing import Sequence

from .pareto_archive_cap_certificate import (
    ARCHIVE_CAP_CERTIFICATE_SCHEMA_V15,
    ArchiveCapMetricCertificate,
)
from .pareto_frozen_cells import (
    METRIC_SEMANTICS_V15,
    OBJECTIVE_ARITHMETIC_V15,
    PROBABILITY_SEMANTICS_V15,
)
from .pareto_independent_replica_certificate import (
    FALSE_PASS_BOUND_SEMANTICS,
    FALSE_PASS_EVENT_LABEL,
    FalsePassCertificate,
    PilotPowerCertificate,
)
from .pareto_independent_replica_runner import (
    ACCEPTANCE_SEMANTICS_V15,
    ENDPOINT_SUM_SEMANTICS_V15,
    INDEPENDENT_REPLICA_ALGORITHM_ID_V15,
    INDEPENDENT_REPLICA_RESULT_SCHEMA_V15,
    IndependentReplicaBatchResult,
)
from .pareto_reference_fidelity import (
    REFERENCE_FIDELITY_SCHEMA_V15,
    ReferenceFidelityCertificate,
)
from .pareto_v15_context import (
    V15CertificateContext,
    V15CertificateContextError,
    verify_v15_context_sha256,
)


V15_PUBLICATION_GATE_SCHEMA = "pareto_v15_publication_gate_v1"
CURRENT_GEOMETRIC_METRIC_SCHEMA = (
    "pareto_smc_geometric_bound_certificate_v2"
)
SUPERSEDED_METRIC_SCHEMAS = frozenset(
    {"pareto_smc_geometric_bound_certificate_v1"}
)


@dataclass(frozen=True)
class V15PublicationGate:
    schema: str
    component_contract_gate: bool
    p0_correctness_gate: bool
    p1_theory_gate: bool
    operational_authorization_gate: bool
    study_level_commitment_gate: bool
    machine_formalization_gate: bool
    competitive_evidence_gate: bool
    p0_issues: tuple[str, ...]
    p1_issues: tuple[str, ...]
    unresolved_publication_obligations: tuple[str, ...]
    algorithm_identity: str
    certificate_scope: str
    metric_semantics: str
    exactness_scope: str
    false_pass_scope: str
    formalization_status: str
    competitive_evidence: str
    submission_verdict: str

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


def evaluate_v15_publication_gate(
    *,
    certificate_context: V15CertificateContext,
    geometric_metric_schema: str,
    runner_result: IndependentReplicaBatchResult,
    false_pass_certificate: FalsePassCertificate,
    pilot_power_certificates: Sequence[PilotPowerCertificate],
    archive_cap_certificate: ArchiveCapMetricCertificate,
    reference_fidelity_certificate: ReferenceFidelityCertificate | None,
    true_front_coverage_claimed: bool,
    interacting_smc_certificate_transfer_claimed: bool,
    external_future_beacon_verified: bool,
    study_matrix_commitment_verified: bool,
    lean_probability_core_compiled_zero_sorry: bool,
    competitive_evidence_complete: bool,
) -> V15PublicationGate:
    """Evaluate claim hygiene without turning engineering tests into evidence."""

    p0: list[str] = []
    p1: list[str] = []
    publication: list[str] = []
    try:
        verify_v15_context_sha256(
            certificate_context,
            certificate_context.context_sha256,
        )
    except V15CertificateContextError:
        p0.append("certificate_context_is_not_canonical")
    if geometric_metric_schema in SUPERSEDED_METRIC_SCHEMAS:
        p0.append("legacy_power_mean_igd_schema_is_superseded")
    elif geometric_metric_schema != CURRENT_GEOMETRIC_METRIC_SCHEMA:
        p0.append("unknown_geometric_metric_schema")
    if runner_result.schema != INDEPENDENT_REPLICA_RESULT_SCHEMA_V15:
        p0.append("independent_replica_result_schema_mismatch")
    if runner_result.context_sha256 != certificate_context.context_sha256:
        p0.append("runner_result_context_hash_mismatch")
    if runner_result.instance_sha256 != certificate_context.instance_sha256:
        p0.append("runner_result_instance_hash_mismatch")
    if (
        runner_result.configuration_sha256
        != certificate_context.configuration_sha256
    ):
        p0.append("runner_result_configuration_hash_mismatch")
    if (
        runner_result.cell_manifest_sha256
        != certificate_context.cell_manifest_sha256
    ):
        p0.append("runner_result_cell_manifest_hash_mismatch")
    if runner_result.algorithm_id != INDEPENDENT_REPLICA_ALGORITHM_ID_V15:
        p0.append("replica_algorithm_identity_mismatch")
    if runner_result.stream_role != "confirm":
        p0.append("publication_gate_requires_confirm_stream_result")
    if not runner_result.endpoints:
        p0.append("independent_replica_endpoint_batch_is_empty")
    if runner_result.exact_total_evaluations <= 0:
        p0.append("independent_replica_evaluation_count_is_nonpositive")
    elif runner_result.exact_total_evaluations != sum(
        endpoint.evaluations for endpoint in runner_result.endpoints
    ):
        p0.append("independent_replica_evaluation_ledger_mismatch")
    endpoint_keys = tuple(
        (endpoint.type_id, endpoint.replica_index)
        for endpoint in runner_result.endpoints
    )
    if len(endpoint_keys) != len(set(endpoint_keys)):
        p0.append("duplicate_independent_replica_endpoint_identity")
    hit_cells = tuple(cell for cell, _ in runner_result.hit_counts)
    hit_total = sum(count for _, count in runner_result.hit_counts)
    if (
        len(hit_cells) != len(set(hit_cells))
        or any(count < 0 for _, count in runner_result.hit_counts)
        or hit_total
        != sum(
            endpoint.observable_cell_hit
            for endpoint in runner_result.endpoints
        )
    ):
        p0.append("independent_replica_hit_ledger_mismatch")
    if runner_result.population_interaction_present:
        p0.append("independent_replica_branch_reports_population_interaction")
    if runner_result.resampling_performed:
        p0.append("independent_replica_branch_performed_resampling")
    if runner_result.probability_semantics != PROBABILITY_SEMANTICS_V15:
        p0.append("independent_replica_probability_semantics_mismatch")
    if runner_result.acceptance_semantics != ACCEPTANCE_SEMANTICS_V15:
        p0.append("mh_acceptance_exactness_scope_mismatch")
    if runner_result.endpoint_sum_semantics != ENDPOINT_SUM_SEMANTICS_V15:
        p0.append("dyadic_edge_sum_contract_mismatch")
    if (
        runner_result.endpoint_classification_semantics
        != OBJECTIVE_ARITHMETIC_V15
    ):
        p0.append("endpoint_cell_arithmetic_contract_mismatch")
    if false_pass_certificate.to_jsonable()["event"] != FALSE_PASS_EVENT_LABEL:
        p0.append("false_pass_event_mismatch")
    if (
        false_pass_certificate.to_jsonable()["semantics"]
        != FALSE_PASS_BOUND_SEMANTICS
    ):
        p0.append("false_pass_joint_probability_was_relabelled")
    if archive_cap_certificate.schema != ARCHIVE_CAP_CERTIFICATE_SCHEMA_V15:
        p0.append("archive_cap_schema_mismatch")
    if not archive_cap_certificate.passed:
        p0.append("archive_cap_metric_tolerance_failed")
    if interacting_smc_certificate_transfer_claimed:
        p0.append("unproved_certificate_transfer_to_interacting_smc")

    powers = tuple(pilot_power_certificates)
    if not powers:
        p1.append("pilot_power_certificate_missing")
    elif any(not certificate.power_gate for certificate in powers):
        p1.append("predeclared_pilot_power_gate_failed")
    if true_front_coverage_claimed:
        p1.append(
            "verified_true_front_completeness_artifact_not_implemented"
        )
        if reference_fidelity_certificate is None:
            p1.append("true_front_claim_without_reference_fidelity")
        scope = "true_front_claim_rejected_unverified_completeness"
    else:
        scope = (
            "supplied_front_relative_conditional_composition"
            if reference_fidelity_certificate is not None
            else "frozen_reference_relative_only"
        )
    if reference_fidelity_certificate is not None and (
        reference_fidelity_certificate.schema
        != REFERENCE_FIDELITY_SCHEMA_V15
        or not reference_fidelity_certificate.composed_cover_verified
        or reference_fidelity_certificate.true_front_coverage_claimed
        or reference_fidelity_certificate.external_true_front_completeness_verified
    ):
        p0.append("reference_fidelity_scope_or_schema_mismatch")
    p1.extend(
        (
            "adaptive_type_cell_allocation_upper_lower_bound_missing",
            "finite_menu_out_of_sample_oracle_inequality_missing",
            "intrinsic_dimension_instance_family_certificate_missing",
        )
    )

    publication.extend(
        (
            "external_future_beacon_controls_not_verified_by_raw_artifact_gate",
            "study_level_merkle_matrix_not_verified_by_raw_artifact_gate",
            "lean_probability_core_not_machine_verified_by_this_gate",
            "matched_competitive_evidence_not_verified_by_this_gate",
        )
    )
    if external_future_beacon_verified:
        publication.append("caller_beacon_boolean_is_not_accepted_as_evidence")
    if study_matrix_commitment_verified:
        publication.append(
            "caller_study_commitment_boolean_is_not_accepted_as_evidence"
        )
    if lean_probability_core_compiled_zero_sorry:
        publication.append("caller_lean_boolean_is_not_accepted_as_evidence")
    if competitive_evidence_complete:
        publication.append(
            "caller_competitive_boolean_is_not_accepted_as_evidence"
        )

    component_gate = not p0
    p0.append(
        "end_to_end_context_bound_raw_artifact_reverification_not_implemented"
    )
    p0_gate = False
    # The three explicitly listed missing contribution theorems keep the
    # top-tier theory gate closed even though the finite-reference composition
    # and pilot-power lemmas are now implemented.
    p1_gate = not p1
    all_gates = False
    return V15PublicationGate(
        schema=V15_PUBLICATION_GATE_SCHEMA,
        component_contract_gate=component_gate,
        p0_correctness_gate=p0_gate,
        p1_theory_gate=p1_gate,
        operational_authorization_gate=False,
        study_level_commitment_gate=False,
        machine_formalization_gate=False,
        competitive_evidence_gate=False,
        p0_issues=tuple(p0),
        p1_issues=tuple(p1),
        unresolved_publication_obligations=tuple(publication),
        algorithm_identity=INDEPENDENT_REPLICA_ALGORITHM_ID_V15,
        certificate_scope=scope,
        metric_semantics=METRIC_SEMANTICS_V15,
        exactness_scope=(
            "exact_dyadic_edge_sum_and_exact_frozen_endpoint_cell_only;"
            "binary64_energy_and_mh_not_machine_exact"
        ),
        false_pass_scope=FALSE_PASS_BOUND_SEMANTICS,
        formalization_status="NOT_PERFORMED_FOR_V15_PROBABILITY_CORE",
        competitive_evidence="NOT_RUN",
        submission_verdict=(
            "READY_FOR_EXTERNAL_REVIEW" if all_gates else "HOLD"
        ),
    )


__all__ = [
    "CURRENT_GEOMETRIC_METRIC_SCHEMA",
    "SUPERSEDED_METRIC_SCHEMAS",
    "V15_PUBLICATION_GATE_SCHEMA",
    "V15PublicationGate",
    "evaluate_v15_publication_gate",
]
