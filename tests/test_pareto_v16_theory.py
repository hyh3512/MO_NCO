from __future__ import annotations

from fractions import Fraction
import tempfile
from pathlib import Path
import unittest

from mo_nco.instance import MultiObjectiveTSPInstance, instance_sha256
from mo_nco.pareto_frozen_cells import canonical_manifest_payload
from mo_nco.pareto_independent_replica_runner import (
    ReplicaTypeConfiguration,
    replica_configuration_sha256,
    replica_stream_plan_sha256,
)
from mo_nco.pareto_shared_categorical_design import (
    certify_shared_categorical_identification_upper_bound,
    exact_shared_confirm_allocation,
    rational_pairwise_transportation_lower_bound,
)
from mo_nco.pareto_v15_context import V15CertificateContext
from mo_nco.pareto_v16_artifact_bundle import (
    V16_COMPOSED_BUNDLE_SCHEMA,
    V16_INSTANCE_ARTIFACT_SCHEMA,
    V16_REFERENCE_PLAN_SCHEMA,
    V16_REPLICA_CONFIGURATION_SCHEMA,
    V16_STREAM_PLAN_SCHEMA,
    V16_TYPE_CELL_PLAN_SCHEMA,
    V16ArtifactError,
    canonical_sha256,
    verify_v16_composed_bundle,
    write_canonical_v16_bundle,
)
from mo_nco.pareto_v16_theory_gate import evaluate_v16_theory_gate
from mo_nco.pareto_v16_theory_packet import (
    V16_THEORY_PACKET_SCHEMA,
    V16TheoryPacketError,
    verify_v16_theory_packet,
    write_canonical_v16_theory_packet,
)


class ParetoV16TheoryTests(unittest.TestCase):
    def _bundle(self) -> dict[str, object]:
        matrix = tuple(
            tuple(0.0 if i == j else 1.0 for j in range(4))
            for i in range(4)
        )
        instance = MultiObjectiveTSPInstance.from_distance_matrices(
            (matrix, matrix), name="all-one-four-city"
        )
        instance_payload = {
            "schema": V16_INSTANCE_ARTIFACT_SCHEMA,
            "name": "all-one-four-city",
            "distance_matrices_hex": [
                [[value.hex() for value in row] for row in matrix],
                [[value.hex() for value in row] for row in matrix],
            ],
        }
        cells = canonical_manifest_payload(
            lower=(0, 0),
            upper=(10, 10),
            widths=(10, 10),
            observable_cells=((0, 0),),
            local_norm_p="1",
        )
        configs = (
            ReplicaTypeConfiguration(
                type_id="A",
                reference_direction=(0.6, 0.4),
                beta_schedule=(0.0, 1.0),
                mutation_steps_by_stage=(0,),
                replica_count=8,
                chebyshev_rho=0.05,
                global_refresh_probability=0.0,
            ),
            ReplicaTypeConfiguration(
                type_id="B",
                reference_direction=(0.4, 0.6),
                beta_schedule=(0.0, 1.0),
                mutation_steps_by_stage=(0,),
                replica_count=8,
                chebyshev_rho=0.05,
                global_refresh_probability=0.0,
            ),
        )
        config_payload = [
            {
                "schema": V16_REPLICA_CONFIGURATION_SCHEMA,
                "type_id": config.type_id,
                "reference_direction_hex": [
                    value.hex() for value in config.reference_direction
                ],
                "beta_schedule_hex": [
                    value.hex() for value in config.beta_schedule
                ],
                "mutation_steps_by_stage": list(config.mutation_steps_by_stage),
                "replica_count": config.replica_count,
                "chebyshev_rho_hex": config.chebyshev_rho.hex(),
                "global_refresh_probability_hex": (
                    config.global_refresh_probability.hex()
                ),
            }
            for config in configs
        ]
        reference = {
            "schema": V16_REFERENCE_PLAN_SCHEMA,
            "reference_points": [["4", "5"], ["5", "4"]],
            "local_norm_p": "1",
            "archive_cap": 1,
            "hv_reference": ["10", "10"],
            "max_ordinary_igd": "1",
            "max_igd_plus": "0",
            "max_hv_deficit": "0",
        }
        plan = {
            "schema": V16_TYPE_CELL_PLAN_SCHEMA,
            "selection_rule": "max_cp_lower_then_lexicographic_type_id",
            "cells": [
                {
                    "cell": [0, 0],
                    "pilot_alpha_by_type": {"A": "1/200", "B": "1/200"},
                    "target_probability": "1/4",
                    "true_probability_lower_bound": "3/4",
                    "minimum_pilot_power": "1/2",
                    "confirm_failure_budget": "1/100",
                }
            ],
        }
        cell_hash = canonical_sha256(cells)
        pilot_seed = 11
        confirm_seed = 29
        context = V15CertificateContext(
            case_id="v16-all-one",
            instance_sha256=instance_sha256(instance),
            configuration_sha256=replica_configuration_sha256(configs),
            cell_manifest_sha256=cell_hash,
            reference_sha256=canonical_sha256(reference),
            type_cell_plan_sha256=canonical_sha256(plan),
            pilot_plan_sha256=replica_stream_plan_sha256(
                configs,
                master_seed=pilot_seed,
                stream_role="pilot",
                cell_manifest_sha256=cell_hash,
            ),
            confirm_plan_sha256=replica_stream_plan_sha256(
                configs,
                master_seed=confirm_seed,
                stream_role="confirm",
                cell_manifest_sha256=cell_hash,
            ),
        )
        return {
            "schema": V16_COMPOSED_BUNDLE_SCHEMA,
            "context": context.to_jsonable(),
            "instance": instance_payload,
            "cell_manifest": cells,
            "replica_configurations": config_payload,
            "pilot_stream": {
                "schema": V16_STREAM_PLAN_SCHEMA,
                "stream_role": "pilot",
                "master_seed": pilot_seed,
            },
            "confirm_stream": {
                "schema": V16_STREAM_PLAN_SCHEMA,
                "stream_role": "confirm",
                "master_seed": confirm_seed,
            },
            "reference_plan": reference,
            "type_cell_plan": plan,
        }

    def _theory_packet(
        self, *, composed_sha256: str, context_sha256: str
    ) -> dict[str, object]:
        return {
            "schema": V16_THEORY_PACKET_SCHEMA,
            "composed_packet_sha256": composed_sha256,
            "context_sha256": context_sha256,
            "probability_matrix": {
                "A": {"0,0": "4/5"},
                "B": {"0,0": "2/5"},
            },
            "identification_error": "1/20",
            "max_rounds": 100000,
            "transport_error": "1/20",
            "transport_log_series_terms": 64,
            "confirm_union_miss_budget": "1/20",
            "max_assignments": 1000,
            "max_total_confirm_replicas": 100000,
            "finite_menu": {
                "confidence_error": "1/20",
                "losses_by_design": {
                    "d1": ["0", "1/10"],
                    "d2": ["1", "1"],
                },
            },
            "intrinsic": {
                "reference_points": [["4", "5"], ["5", "4"]],
                "lower_constant": "1",
                "upper_constant": "1",
                "tau": "1",
            },
            "probability_matrix_status": "theorem_parameter",
        }

    def test_composed_raw_packet_opens_p0(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            write_canonical_v16_bundle(path, self._bundle())
            certificate = verify_v16_composed_bundle(path)
        self.assertTrue(certificate.p0_correctness_gate)
        self.assertTrue(certificate.archive_cap_certificate.passed)

    def test_noncanonical_packet_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            write_canonical_v16_bundle(path, self._bundle())
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(V16ArtifactError, "canonical"):
                verify_v16_composed_bundle(path)

    def test_shared_upper_lower_and_joint_confirm_allocation(self):
        probabilities = {
            "A": {"left": Fraction(3, 5), "right": Fraction(1, 5)},
            "B": {"left": Fraction(1, 5), "right": Fraction(3, 5)},
        }
        upper = certify_shared_categorical_identification_upper_bound(
            probabilities,
            familywise_error="1/20",
            max_rounds=100000,
        )
        self.assertEqual({item.best_type for item in upper.cell_bounds}, {"A", "B"})
        self.assertLess(
            upper.total_pilot_replicas_upper,
            2 * sum(item.stopping_round_upper for item in upper.cell_bounds),
        )
        lower = rational_pairwise_transportation_lower_bound(
            cell_id="left",
            best_type="A",
            challenger_type="B",
            best_probability="3/5",
            challenger_probability="1/5",
            error_probability="1/20",
        )
        self.assertGreater(Fraction(lower.expected_total_samples_lower), 0)
        allocation = exact_shared_confirm_allocation(
            probabilities,
            union_miss_budget="1/20",
        )
        self.assertTrue(allocation.exact_single_type_assignment_optimum)
        self.assertLessEqual(Fraction(allocation.total_union_miss_upper), Fraction(1, 20))

    def test_raw_theory_packet_and_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle_path = Path(directory) / "bundle.json"
            write_canonical_v16_bundle(bundle_path, self._bundle())
            composed = verify_v16_composed_bundle(bundle_path)
            theory_path = Path(directory) / "theory.json"
            write_canonical_v16_theory_packet(
                theory_path,
                self._theory_packet(
                    composed_sha256=composed.packet_sha256,
                    context_sha256=composed.context_sha256,
                ),
            )
            theory, _ = verify_v16_theory_packet(
                theory_path,
                composed_bundle_path=bundle_path,
            )
            gate, _, _ = evaluate_v16_theory_gate(
                composed_bundle_path=bundle_path,
                theory_packet_path=theory_path,
            )
        self.assertTrue(theory.all_children_recomputed_from_raw_exact_inputs)
        self.assertTrue(gate.p0_correctness_gate)
        self.assertTrue(gate.p1_main_theory_gate)
        self.assertTrue(gate.p2_mathematical_contribution_gate)
        self.assertFalse(gate.literature_novelty_gate)
        self.assertEqual(gate.submission_verdict, "HOLD")

    def test_theory_packet_binding_tamper_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle_path = Path(directory) / "bundle.json"
            write_canonical_v16_bundle(bundle_path, self._bundle())
            composed = verify_v16_composed_bundle(bundle_path)
            payload = self._theory_packet(
                composed_sha256="0" * 64,
                context_sha256=composed.context_sha256,
            )
            theory_path = Path(directory) / "theory.json"
            write_canonical_v16_theory_packet(theory_path, payload)
            with self.assertRaisesRegex(V16TheoryPacketError, "different composed"):
                verify_v16_theory_packet(
                    theory_path,
                    composed_bundle_path=bundle_path,
                )


if __name__ == "__main__":
    unittest.main()

