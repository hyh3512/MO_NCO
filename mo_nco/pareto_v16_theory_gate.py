from __future__ import annotations

"""Fail-closed v16 theory gate driven only by canonical raw packets."""

from dataclasses import asdict, dataclass
from pathlib import Path

from .pareto_frozen_cells import parse_canonical_fraction
from .pareto_v16_artifact_bundle import V16ComposedArtifactCertificate
from .pareto_v16_theory_packet import (
    V16TheoryParameterCertificate,
    verify_v16_theory_packet,
)

V16_THEORY_GATE_SCHEMA = "pareto_v16_theory_gate_v2"


@dataclass(frozen=True)
class V16TheoryGate:
    schema: str
    p0_correctness_gate: bool
    p1_main_theory_gate: bool
    p2_mathematical_contribution_gate: bool
    literature_novelty_gate: bool
    operational_gate: bool
    machine_formalization_gate: bool
    competitive_evidence_gate: bool
    submission_verdict: str
    p0_certificate_sha256: str
    p1_certificate_sha256: str
    p1_scope: str
    p2_scope: str
    remaining_obligations: tuple[str, ...]

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


def evaluate_v16_theory_gate(
    *,
    composed_bundle_path: str | Path,
    theory_packet_path: str | Path,
) -> tuple[
    V16TheoryGate,
    V16ComposedArtifactCertificate,
    V16TheoryParameterCertificate,
]:
    theory, composed = verify_v16_theory_packet(
        theory_packet_path,
        composed_bundle_path=composed_bundle_path,
    )
    p0 = composed.p0_correctness_gate and composed.canonical_raw_artifacts_recomputed
    p1 = (
        theory.all_children_recomputed_from_raw_exact_inputs
        and theory.shared_identification.unique_best_required
        and theory.shared_confirm_allocation.exact_single_type_assignment_optimum
        and bool(theory.transportation_lower_bounds)
        and all(
            parse_canonical_fraction(
                item.expected_total_samples_lower,
                label="expected_total_samples_lower",
            ) > 0
            for item in theory.transportation_lower_bounds
        )
        and theory.intrinsic_dimension.pairwise_bilipschitz_verified
        and theory.intrinsic_dimension.tau_net_verified
        and theory.composed_packet_sha256 == composed.packet_sha256
        and theory.context_sha256 == composed.context_sha256
    )
    # P2 is a mathematical candidate package, not a literature-novelty claim.
    p2 = (
        p1
        and theory.shared_identification.total_pilot_replicas_upper > 0
        and theory.shared_confirm_allocation.total_replicas > 0
        and all(
            parse_canonical_fraction(
                item.expected_total_samples_lower,
                label="expected_total_samples_lower",
            ) > 0
            for item in theory.transportation_lower_bounds
        )
    )
    obligations = (
        "systematic_literature_novelty_review_not_completed",
        "probability_matrix_requires_theorem_status_or_independent_source_certificate",
        "ideal_iid_categorical_endpoint_streams_not_machine_verified",
        "study_level_future_beacon_and_non_equivocating_log_not_verified",
        "lean_probability_adaptive_allocation_and_metric_core_not_compiled_zero_sorry",
        "matched_competitive_matrix_not_run",
        "true_Pareto_front_completeness_not_claimed_by_supplied_reference_packet",
        "interacting_Pareto_SMC_does_not_inherit_the_independent_endpoint_bounds",
    )
    return (
        V16TheoryGate(
            schema=V16_THEORY_GATE_SCHEMA,
            p0_correctness_gate=p0,
            p1_main_theory_gate=p0 and p1,
            p2_mathematical_contribution_gate=p0 and p2,
            literature_novelty_gate=False,
            operational_gate=False,
            machine_formalization_gate=False,
            competitive_evidence_gate=False,
            submission_verdict="HOLD",
            p0_certificate_sha256=composed.packet_sha256,
            p1_certificate_sha256=theory.packet_sha256,
            p1_scope=(
                "closed_for_canonical_supplied_reference_independent_categorical_endpoint_branch"
                if p0 and p1
                else "FAIL"
            ),
            p2_scope=(
                "shared_categorical_gap_upper_plus_rational_transport_lower_plus_exact_joint_confirm_allocation"
                if p0 and p2
                else "FAIL"
            ),
            remaining_obligations=obligations,
        ),
        composed,
        theory,
    )


__all__ = ["V16_THEORY_GATE_SCHEMA", "V16TheoryGate", "evaluate_v16_theory_gate"]
