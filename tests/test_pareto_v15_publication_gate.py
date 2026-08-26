from __future__ import annotations

import unittest
from dataclasses import replace
from fractions import Fraction

from mo_nco.pareto_archive_cap_certificate import certify_archive_cap
from mo_nco.pareto_frozen_cells import (
    OBJECTIVE_ARITHMETIC_V15,
    PROBABILITY_SEMANTICS_V15,
)
from mo_nco.pareto_independent_replica_certificate import (
    build_false_pass_certificate,
    certify_pilot_power,
)
from mo_nco.pareto_independent_replica_runner import (
    ACCEPTANCE_SEMANTICS_V15,
    ENDPOINT_SUM_SEMANTICS_V15,
    INDEPENDENT_REPLICA_ALGORITHM_ID_V15,
    INDEPENDENT_REPLICA_RESULT_SCHEMA_V15,
    IndependentReplicaBatchResult,
    ReplicaEndpoint,
)
from mo_nco.pareto_v15_publication_gate import (
    CURRENT_GEOMETRIC_METRIC_SCHEMA,
    evaluate_v15_publication_gate,
)
from mo_nco.pareto_v15_context import V15CertificateContext


class PublicationGateV15Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = V15CertificateContext(
            case_id="publication-gate-test",
            instance_sha256="1" * 64,
            configuration_sha256="3" * 64,
            cell_manifest_sha256="2" * 64,
            reference_sha256="4" * 64,
            type_cell_plan_sha256="5" * 64,
            pilot_plan_sha256="6" * 64,
            confirm_plan_sha256="7" * 64,
        )
        self.runner = IndependentReplicaBatchResult(
            schema=INDEPENDENT_REPLICA_RESULT_SCHEMA_V15,
            algorithm_id=INDEPENDENT_REPLICA_ALGORITHM_ID_V15,
            instance_sha256="1" * 64,
            cell_manifest_sha256="2" * 64,
            configuration_sha256="3" * 64,
            context_sha256=self.context.context_sha256,
            stream_role="confirm",
            endpoints=(
                ReplicaEndpoint(
                    type_id="balanced",
                    replica_index=0,
                    derived_seed=7,
                    tour=(0, 1, 2),
                    exact_objective=("1/1", "1/1"),
                    binary64_objective=(1.0, 1.0),
                    frozen_cell=(0, 0),
                    observable_cell_hit=True,
                    evaluations=1,
                    accepted_mutations=0,
                    mutation_attempts=0,
                ),
            ),
            hit_counts=(((0, 0), 1),),
            exact_total_evaluations=1,
            population_interaction_present=False,
            resampling_performed=False,
            probability_semantics=PROBABILITY_SEMANTICS_V15,
            acceptance_semantics=ACCEPTANCE_SEMANTICS_V15,
            endpoint_sum_semantics=ENDPOINT_SUM_SEMANTICS_V15,
            endpoint_classification_semantics=OBJECTIVE_ARITHMETIC_V15,
        )
        self.false_pass = build_false_pass_certificate(
            Fraction(1, 100),
            Fraction(1, 100),
        )
        self.power = certify_pilot_power(
            10,
            Fraction(1, 10),
            Fraction(1, 2),
            Fraction(1, 20),
            minimum_acceptable_pass_probability=Fraction(1, 2),
        )
        self.cap = certify_archive_cap(
            reference_points=((0, 0),),
            witnesses=((0, 0),),
            reference_to_witness=(0,),
            cap=1,
            p="1",
            ordinary_igd_base_upper=0,
            additive_base_vector=(0, 0),
            hv_reference=(2, 2),
            max_ordinary_igd=0,
            max_igd_plus=0,
            max_hv_deficit=0,
        )

    def _evaluate(self, **overrides):  # type: ignore[no-untyped-def]
        arguments = {
            "certificate_context": self.context,
            "geometric_metric_schema": CURRENT_GEOMETRIC_METRIC_SCHEMA,
            "runner_result": self.runner,
            "false_pass_certificate": self.false_pass,
            "pilot_power_certificates": (self.power,),
            "archive_cap_certificate": self.cap,
            "reference_fidelity_certificate": None,
            "true_front_coverage_claimed": False,
            "interacting_smc_certificate_transfer_claimed": False,
            "external_future_beacon_verified": False,
            "study_matrix_commitment_verified": False,
            "lean_probability_core_compiled_zero_sorry": False,
            "competitive_evidence_complete": False,
        }
        arguments.update(overrides)
        return evaluate_v15_publication_gate(**arguments)

    def test_component_repairs_do_not_turn_hold_into_p0_or_publication_pass(self) -> None:
        gate = self._evaluate()

        self.assertTrue(gate.component_contract_gate)
        self.assertFalse(gate.p0_correctness_gate)
        self.assertFalse(gate.p1_theory_gate)
        self.assertEqual(gate.competitive_evidence, "NOT_RUN")
        self.assertEqual(gate.submission_verdict, "HOLD")
        self.assertEqual(gate.certificate_scope, "frozen_reference_relative_only")
        self.assertIn(
            "adaptive_type_cell_allocation_upper_lower_bound_missing",
            gate.p1_issues,
        )
        self.assertIn(
            "end_to_end_context_bound_raw_artifact_reverification_not_implemented",
            gate.p0_issues,
        )

    def test_legacy_metric_or_smc_transfer_reopens_p0(self) -> None:
        gate = self._evaluate(
            geometric_metric_schema=(
                "pareto_smc_geometric_bound_certificate_v1"
            ),
            interacting_smc_certificate_transfer_claimed=True,
        )

        self.assertFalse(gate.p0_correctness_gate)
        self.assertFalse(gate.component_contract_gate)
        self.assertIn(
            "legacy_power_mean_igd_schema_is_superseded",
            gate.p0_issues,
        )
        self.assertIn(
            "unproved_certificate_transfer_to_interacting_smc",
            gate.p0_issues,
        )

    def test_true_front_wording_requires_reference_fidelity_certificate(self) -> None:
        gate = self._evaluate(true_front_coverage_claimed=True)

        self.assertIn(
            "true_front_claim_without_reference_fidelity",
            gate.p1_issues,
        )
        self.assertEqual(
            gate.certificate_scope,
            "true_front_claim_rejected_unverified_completeness",
        )

    def test_empty_or_semantically_relabelled_replica_result_reopens_p0(self) -> None:
        self.runner = replace(
            self.runner,
            endpoints=(),
            hit_counts=(),
            exact_total_evaluations=0,
            probability_semantics="empirical_replay_treated_as_iid",
        )

        gate = self._evaluate()

        self.assertFalse(gate.p0_correctness_gate)
        self.assertFalse(gate.component_contract_gate)
        self.assertIn(
            "independent_replica_endpoint_batch_is_empty",
            gate.p0_issues,
        )
        self.assertIn(
            "independent_replica_probability_semantics_mismatch",
            gate.p0_issues,
        )

    def test_caller_booleans_cannot_open_external_or_formal_gates(self) -> None:
        gate = self._evaluate(
            external_future_beacon_verified=True,
            study_matrix_commitment_verified=True,
            lean_probability_core_compiled_zero_sorry=True,
            competitive_evidence_complete=True,
        )

        self.assertFalse(gate.operational_authorization_gate)
        self.assertFalse(gate.study_level_commitment_gate)
        self.assertFalse(gate.machine_formalization_gate)
        self.assertFalse(gate.competitive_evidence_gate)
        self.assertEqual(
            gate.formalization_status,
            "NOT_PERFORMED_FOR_V15_PROBABILITY_CORE",
        )
        self.assertEqual(gate.competitive_evidence, "NOT_RUN")
        self.assertIn(
            "caller_lean_boolean_is_not_accepted_as_evidence",
            gate.unresolved_publication_obligations,
        )


if __name__ == "__main__":
    unittest.main()

