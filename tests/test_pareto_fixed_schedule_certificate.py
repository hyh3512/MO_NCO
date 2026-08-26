from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from mo_nco.contracts import ClaimLevel
from mo_nco.instance import MultiObjectiveTSPInstance
from mo_nco.instance import instance_sha256
from mo_nco.pareto_bounds import (
    nondominated_points,
    shifted_front_hv_deficit_bound,
)
from mo_nco.pareto_fixed_schedule_certificate import (
    FixedScheduleCertificateError,
    build_regeneration_pilot_plan_commitment_from_spec,
    certify_fixed_schedule_reference_metrics,
    certify_fixed_schedule_reference_metrics_from_spec,
)
from mo_nco.pareto_fixed_reference_spec import (
    load_fixed_reference_certificate_specification,
)
from mo_nco.pareto_fixed_schedule_experiment import (
    run_fixed_schedule_pilot_confirm,
)
from mo_nco.pareto_smc_spec import (
    analytic_objective_box,
    load_pareto_smc_specification,
    original_unit_cell_widths,
)
from mo_nco.pareto_smc import AnnealedParetoSMCOptimizer


class FixedScheduleReferenceCertificateTests(unittest.TestCase):
    def test_fixed_reference_duplicate_json_fields_fail_closed(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        raw = (
            root
            / "benchmarks"
            / "pareto_smc_v10_fixed_reference_smoke.json"
        ).read_text(encoding="utf-8")
        duplicate = raw.replace(
            '"schema":',
            '"schema":"forged","schema":',
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(duplicate, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
                load_fixed_reference_certificate_specification(
                    path,
                    objective_dimension=2,
                )

    def test_fixed_reference_numeric_strings_and_booleans_fail_closed(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        baseline = json.loads(
            (
                root
                / "benchmarks"
                / "pareto_smc_v10_fixed_reference_smoke.json"
            ).read_text(encoding="utf-8")
        )
        variants = []
        boolean_norm = copy.deepcopy(baseline)
        boolean_norm["metrics"]["igd_p"] = True
        variants.append(boolean_norm)
        string_budget = copy.deepcopy(baseline)
        string_budget["failure_budgets"]["pilot"] = "0.025"
        variants.append(string_budget)
        boolean_reference = copy.deepcopy(baseline)
        boolean_reference["metrics"]["hv_reference"][0] = False
        variants.append(boolean_reference)
        with tempfile.TemporaryDirectory() as directory:
            for index, payload in enumerate(variants):
                path = Path(directory) / f"numeric-{index}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_fixed_reference_certificate_specification(
                        path,
                        objective_dimension=2,
                    )

    def test_v2_tolerance_only_witness_is_diagnostic_not_formal(
        self,
    ) -> None:
        matrix = (
            (0.0, 1.1, 2.2, 3.3),
            (1.1, 0.0, 4.4, 5.5),
            (2.2, 4.4, 0.0, 6.6),
            (3.3, 5.5, 6.6, 0.0),
        )
        instance = MultiObjectiveTSPInstance.from_distance_matrices(
            (matrix,),
            name="floating_witness",
        )
        self.assertFalse(instance.exact_two_opt_delta_in_binary64)
        tour = (0, 1, 2, 3)
        local = instance.evaluate(tour)
        approximate = ((local[0] + 5e-13,),)
        witnesses = (
            {
                "tour": tour,
                "objectives": approximate[0],
            },
        )
        payload = {
            "schema": "pareto_smc_fixed_reference_certificate_spec_v2",
            "instance_sha256": instance_sha256(instance),
            "pareto_smc_specification_sha256": "a" * 64,
            "reference_front": {
                "source": "independent_exact_solver",
                "artifact_sha256": self._reference_hash(approximate),
                "objectives": approximate,
                "witnesses": witnesses,
                "witness_payload_sha256": self._witness_hash(witnesses),
            },
            "streams": {
                "pilot_seed": 1,
                "confirm_seed": 2,
                "independence_model": "ideal_product_random_streams",
            },
            "failure_budgets": {"pilot": 0.025, "confirm": 0.025},
            "metrics": {
                "igd_p": 1.0,
                "hv_reference": [local[0] + 1.0],
                "max_igd_bound": 1.0,
                "max_hv_deficit_bound": 1.0,
            },
            "certified_archive": {
                "policy": "deterministic_reference_coverage_v1",
                "max_size": 1,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "approximate_floating_witness.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            specification = (
                load_fixed_reference_certificate_specification(
                    path,
                    objective_dimension=1,
                    instance=instance,
                )
            )
        self.assertGreater(
            specification.reference_witness_max_abs_error,
            0.0,
        )
        self.assertFalse(
            specification.reference_feasibility_verified_by_runtime
        )
        self.assertIn(
            "diagnostic_only",
            specification.reference_witness_equivalence_contract,
        )

    def test_v2_integer_witness_requires_exact_objective_equality(
        self,
    ) -> None:
        matrix = (
            (0.0, 100_000_000.0, 200_000_000.0, 300_000_000.0),
            (100_000_000.0, 0.0, 400_000_000.0, 500_000_000.0),
            (200_000_000.0, 400_000_000.0, 0.0, 600_000_000.0),
            (300_000_000.0, 500_000_000.0, 600_000_000.0, 0.0),
        )
        instance = MultiObjectiveTSPInstance.from_distance_matrices(
            (matrix,),
            name="large_integer",
        )
        tour = (0, 1, 2, 3)
        local = instance.evaluate(tour)
        tampered_reference = ((local[0] + 1e-4,),)
        witnesses = (
            {
                "tour": tour,
                "objectives": tampered_reference[0],
            },
        )
        payload = {
            "schema": "pareto_smc_fixed_reference_certificate_spec_v2",
            "instance_sha256": instance_sha256(instance),
            "pareto_smc_specification_sha256": "a" * 64,
            "reference_front": {
                "source": "independent_exact_solver",
                "artifact_sha256": self._reference_hash(
                    tampered_reference
                ),
                "objectives": tampered_reference,
                "witnesses": witnesses,
                "witness_payload_sha256": self._witness_hash(
                    witnesses
                ),
            },
            "streams": {
                "pilot_seed": 1,
                "confirm_seed": 2,
                "independence_model": "ideal_product_random_streams",
            },
            "failure_budgets": {"pilot": 0.025, "confirm": 0.025},
            "metrics": {
                "igd_p": 1.0,
                "hv_reference": [local[0] + 1.0],
                "max_igd_bound": 1.0,
                "max_hv_deficit_bound": 1.0,
            },
            "certified_archive": {
                "policy": "deterministic_reference_coverage_v1",
                "max_size": 1,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered_integer_witness.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "does not match local",
            ):
                load_fixed_reference_certificate_specification(
                    path,
                    objective_dimension=1,
                    instance=instance,
                )

    @staticmethod
    def _reference_hash(reference):  # type: ignore[no-untyped-def]
        canonical = tuple(sorted(reference))
        encoded = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _reference_witnesses(instance, reference):  # type: ignore[no-untyped-def]
        by_objective = {}
        for tail in itertools.permutations(range(1, instance.num_cities)):
            tour = (0,) + tail
            objective = instance.evaluate(tour)
            by_objective.setdefault(objective, tour)
        return tuple(
            {
                "tour": by_objective[objective],
                "objectives": objective,
            }
            for objective in sorted(reference)
        )

    @staticmethod
    def _witness_hash(witnesses):  # type: ignore[no-untyped-def]
        canonical = tuple(
            sorted(
                (
                    {
                        "tour": tuple(record["tour"]),
                        "objectives": tuple(record["objectives"]),
                    }
                    for record in witnesses
                ),
                key=lambda record: (
                    record["objectives"],
                    record["tour"],
                ),
            )
        )
        encoded = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _make_pair(self):  # type: ignore[no-untyped-def]
        instance = MultiObjectiveTSPInstance.random_biobjective(4, seed=410)
        tours = tuple(
            (0,) + tail
            for tail in itertools.permutations(range(1, 4))
        )
        objectives = tuple(instance.evaluate(tour) for tour in tours)
        reference = nondominated_points(objectives)
        lower = tuple(
            min(point[index] for point in objectives)
            for index in range(2)
        )
        upper = tuple(
            max(point[index] for point in objectives)
            for index in range(2)
        )
        widths = tuple(
            (upper[index] - lower[index]) / 4.0
            for index in range(2)
        )
        kwargs = {
            "instance": instance,
            "particles_per_reference": 4096,
            "evaluations": 12288,
            "beta_schedule": (0.0, 0.1),
            "reference_directions": ((0.5, 0.5),),
            "objective_lower_bounds": lower,
            "objective_upper_bounds": upper,
            "epsilon": widths,
            "resampling_policy": "always",
            "mutation_steps_by_stage": (2,),
            "global_refresh_probability": 1.0,
            "archive_max_size": 1,
        }
        pilot = AnnealedParetoSMCOptimizer(seed=411, **kwargs).run()
        confirm = AnnealedParetoSMCOptimizer(seed=412, **kwargs).run()
        igd_bound = math.sqrt(sum(width * width for width in widths))
        hv_bound = shifted_front_hv_deficit_bound(
            reference,
            additive_widths=widths,
            reference=upper,
        )
        return (
            pilot,
            confirm,
            reference,
            upper,
            igd_bound,
            hv_bound,
        )

    def test_pilot_confirm_certificate_is_nonvacuous_and_metric_direct(
        self,
    ) -> None:
        (
            pilot,
            confirm,
            reference,
            upper,
            igd_bound,
            hv_bound,
        ) = self._make_pair()
        certificate = certify_fixed_schedule_reference_metrics(
            pilot,
            confirm,
            reference_objectives=reference,
            reference_source="independent_exact_solver",
            reference_artifact_sha256=self._reference_hash(reference),
            pilot_failure_budget=0.025,
            confirm_failure_budget=0.025,
            igd_p=2.0,
            hv_reference=upper,
            max_igd_bound=1.1 * igd_bound,
            max_hv_deficit_bound=1.1 * hv_bound,
        )
        self.assertEqual(certificate["scientific_design_gate"], "PASS")
        self.assertEqual(certificate["realized_metric_gate"], "PASS")
        self.assertEqual(
            certificate["claim_level"],
            ClaimLevel.PARETO_SMC_FIXED_REFERENCE_BOUND.value,
        )
        self.assertIsNone(certificate["probability_at_least"])
        self.assertAlmostEqual(
            certificate["false_pass_probability_upper_bound"],
            0.05,
        )
        self.assertFalse(
            certificate[
                "conditional_coverage_probability_given_pass_claimed"
            ]
        )
        self.assertTrue(certificate["metric_tolerances_nontrivial"])
        self.assertTrue(
            all(
                row["strict_hit_margin"] > 0.0
                for row in certificate["cell_assignments"]
            )
        )
        self.assertEqual(certificate["total_certificate_evaluations"], 24576)
        self.assertEqual(
            certificate["certified_output_scope"],
            "nondominated_confirm_terminal_support",
        )
        combined = certify_fixed_schedule_reference_metrics(
            pilot,
            confirm,
            reference_objectives=reference,
            reference_source="independent_exact_solver",
            reference_artifact_sha256=self._reference_hash(reference),
            pilot_failure_budget=0.025,
            confirm_failure_budget=0.025,
            igd_p=2.0,
            hv_reference=upper,
            max_igd_bound=1.1 * igd_bound,
            max_hv_deficit_bound=1.1 * hv_bound,
            certificate_mode="published_or_regeneration",
        )
        self.assertEqual(combined["theorem_family_count"], 2)
        self.assertAlmostEqual(
            combined["published_pilot_failure_budget"],
            0.0125,
        )
        self.assertAlmostEqual(
            combined["regeneration_pilot_failure_budget"],
            0.0125,
        )
        self.assertAlmostEqual(
            combined["false_pass_probability_upper_bound"],
            0.05,
        )
        self.assertNotEqual(
            combined["pair_signature_sha256"],
            certificate["pair_signature_sha256"],
        )

    def test_metric_tolerance_boundary_fails_closed(self) -> None:
        (
            pilot,
            confirm,
            reference,
            upper,
            igd_bound,
            hv_bound,
        ) = self._make_pair()
        strict_tolerance = math.nextafter(igd_bound, 0.0)
        certificate = certify_fixed_schedule_reference_metrics(
            pilot,
            confirm,
            reference_objectives=reference,
            reference_source="independent_exact_solver",
            reference_artifact_sha256=self._reference_hash(reference),
            pilot_failure_budget=0.025,
            confirm_failure_budget=0.025,
            igd_p=2.0,
            hv_reference=upper,
            max_igd_bound=strict_tolerance,
            max_hv_deficit_bound=1.1 * hv_bound,
        )
        self.assertGreater(
            certificate["metric_igd_bound"],
            certificate["metric_igd_tolerance"],
        )
        self.assertEqual(
            certificate["metric_nonvacuity_gate"],
            "FAIL",
        )
        self.assertEqual(certificate["scientific_design_gate"], "FAIL")

    def test_reference_payload_and_independent_streams_fail_closed(self) -> None:
        (
            pilot,
            confirm,
            reference,
            upper,
            igd_bound,
            hv_bound,
        ) = self._make_pair()
        with self.assertRaisesRegex(
            FixedScheduleCertificateError,
            "canonical sorted",
        ):
            certify_fixed_schedule_reference_metrics(
                pilot,
                confirm,
                reference_objectives=reference,
                reference_source="independent_exact_solver",
                reference_artifact_sha256="0" * 64,
                pilot_failure_budget=0.025,
                confirm_failure_budget=0.025,
                igd_p=2.0,
                hv_reference=upper,
                max_igd_bound=1.1 * igd_bound,
                max_hv_deficit_bound=1.1 * hv_bound,
            )

        confirm.metadata["seed"] = pilot.metadata["seed"]
        with self.assertRaisesRegex(
            FixedScheduleCertificateError,
            "seeds must be distinct",
        ):
            certify_fixed_schedule_reference_metrics(
                pilot,
                confirm,
                reference_objectives=reference,
                reference_source="independent_exact_solver",
                reference_artifact_sha256=self._reference_hash(reference),
                pilot_failure_budget=0.025,
                confirm_failure_budget=0.025,
                igd_p=2.0,
                hv_reference=upper,
                max_igd_bound=1.1 * igd_bound,
                max_hv_deficit_bound=1.1 * hv_bound,
            )

    def test_nonuniform_terminal_weights_fail_closed(self) -> None:
        (
            pilot,
            confirm,
            reference,
            upper,
            igd_bound,
            hv_bound,
        ) = self._make_pair()
        tampered = copy.deepcopy(confirm)
        weights = list(
            tampered.metadata["final_normalized_weights_by_reference"][0]
        )
        weights[0] = 0.0
        tampered.metadata["final_normalized_weights_by_reference"] = (
            tuple(weights),
        )
        with self.assertRaisesRegex(
            FixedScheduleCertificateError,
            "not uniform",
        ):
            certify_fixed_schedule_reference_metrics(
                pilot,
                tampered,
                reference_objectives=reference,
                reference_source="independent_exact_solver",
                reference_artifact_sha256=self._reference_hash(reference),
                pilot_failure_budget=0.025,
                confirm_failure_budget=0.025,
                igd_p=2.0,
                hv_reference=upper,
                max_igd_bound=1.1 * igd_bound,
                max_hv_deficit_bound=1.1 * hv_bound,
            )

    def test_predeclared_certificate_spec_binds_both_runs(self) -> None:
        (
            pilot,
            confirm,
            reference,
            upper,
            igd_bound,
            hv_bound,
        ) = self._make_pair()
        smc_hash = "a" * 64
        pilot.metadata["external_specification_sha256"] = smc_hash
        confirm.metadata["external_specification_sha256"] = smc_hash
        payload = {
            "schema": "pareto_smc_fixed_reference_certificate_spec_v1",
            "instance_sha256": pilot.metadata["instance_sha256"],
            "pareto_smc_specification_sha256": smc_hash,
            "reference_front": {
                "source": "independent_exact_solver",
                "artifact_sha256": self._reference_hash(reference),
                "objectives": reference,
            },
            "streams": {
                "pilot_seed": pilot.metadata["seed"],
                "confirm_seed": confirm.metadata["seed"],
                "independence_model": "ideal_product_random_streams",
            },
            "failure_budgets": {
                "pilot": 0.025,
                "confirm": 0.025,
            },
            "metrics": {
                "igd_p": 2.0,
                "hv_reference": upper,
                "max_igd_bound": 1.1 * igd_bound,
                "max_hv_deficit_bound": 1.1 * hv_bound,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "certificate.json"
            path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            specification = (
                load_fixed_reference_certificate_specification(
                    path,
                    objective_dimension=2,
                )
            )
            pilot_commitment = (
                build_regeneration_pilot_plan_commitment_from_spec(
                    pilot,
                    specification,
                    confirm_particles_per_reference=int(
                        confirm.metadata["particles_per_reference"]
                    ),
                )
            )
            certificate = (
                certify_fixed_schedule_reference_metrics_from_spec(
                    pilot,
                    confirm,
                    specification,
                    certificate_mode="regeneration",
                    pilot_plan_commitment=pilot_commitment,
                    pilot_plan_commitment_preconfirm_order_attested_by_runner=(
                        True
                    ),
                )
            )
        self.assertEqual(certificate["scientific_design_gate"], "PASS")
        self.assertEqual(
            certificate["pareto_smc_specification_sha256"],
            smc_hash,
        )
        self.assertEqual(
            len(certificate["certificate_specification_sha256"]),
            64,
        )

    def test_v2_reference_witnesses_are_verified_against_instance(
        self,
    ) -> None:
        (
            pilot,
            confirm,
            reference,
            upper,
            igd_bound,
            hv_bound,
        ) = self._make_pair()
        instance = MultiObjectiveTSPInstance.random_biobjective(
            4,
            seed=410,
        )
        witnesses = self._reference_witnesses(instance, reference)
        smc_hash = "b" * 64
        pilot.metadata["external_specification_sha256"] = smc_hash
        confirm.metadata["external_specification_sha256"] = smc_hash
        payload = {
            "schema": "pareto_smc_fixed_reference_certificate_spec_v2",
            "instance_sha256": instance_sha256(instance),
            "pareto_smc_specification_sha256": smc_hash,
            "reference_front": {
                "source": "independent_exact_solver",
                "artifact_sha256": self._reference_hash(reference),
                "objectives": reference,
                "witnesses": witnesses,
                "witness_payload_sha256": self._witness_hash(witnesses),
            },
            "streams": {
                "pilot_seed": pilot.metadata["seed"],
                "confirm_seed": confirm.metadata["seed"],
                "independence_model": "ideal_product_random_streams",
            },
            "failure_budgets": {
                "pilot": 0.025,
                "confirm": 0.025,
            },
            "metrics": {
                "igd_p": 2.0,
                "hv_reference": upper,
                "max_igd_bound": 1.1 * igd_bound,
                "max_hv_deficit_bound": 1.1 * hv_bound,
            },
            "certified_archive": {
                "policy": "deterministic_reference_coverage_v1",
                "max_size": len(reference),
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "certificate_v2.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            specification = (
                load_fixed_reference_certificate_specification(
                    path,
                    objective_dimension=2,
                    instance=instance,
                )
            )
            pilot_commitment = (
                build_regeneration_pilot_plan_commitment_from_spec(
                    pilot,
                    specification,
                    confirm_particles_per_reference=int(
                        confirm.metadata["particles_per_reference"]
                    ),
                )
            )
            certificate = (
                certify_fixed_schedule_reference_metrics_from_spec(
                    pilot,
                    confirm,
                    specification,
                    certificate_mode="regeneration",
                    pilot_plan_commitment=pilot_commitment,
                    pilot_plan_commitment_preconfirm_order_attested_by_runner=(
                        True
                    ),
                )
            )
        self.assertEqual(
            certificate["reference_feasibility_gate"],
            "PASS",
        )
        self.assertTrue(
            certificate["reference_feasibility_verified_by_runtime"]
        )
        self.assertEqual(
            certificate["reference_witness_payload_sha256"],
            self._witness_hash(witnesses),
        )
        self.assertLessEqual(
            certificate["reference_witness_max_abs_error"],
            1e-12,
        )
        self.assertFalse(
            certificate[
                "reference_feasibility_assumed_from_external_source"
            ]
        )
        self.assertEqual(certificate["certified_archive_gate"], "PASS")
        self.assertEqual(certificate["formal_packet_gate"], "PASS")
        self.assertEqual(
            certificate["claim_level"],
            ClaimLevel.PARETO_SMC_REGENERATION_REFERENCE_BOUND.value,
        )
        self.assertEqual(
            certificate["pilot_plan_commitment_gate"],
            "PASS",
        )
        self.assertEqual(
            certificate["pilot_plan_commitment_content_gate"],
            "PASS",
        )
        self.assertTrue(
            certificate[
                "pilot_plan_commitment_preconfirm_order_attested_by_runner"
            ]
        )
        self.assertEqual(
            certificate["pilot_plan_commitment_preconfirm_order_gate"],
            "PASS_RUNNER_CONTROL_FLOW_ATTESTED",
        )
        self.assertFalse(
            certificate[
                "pilot_plan_commitment_preconfirm_timing_independently_verified"
            ]
        )
        self.assertIsNone(
            certificate["pilot_plan_committed_before_confirm"]
        )
        self.assertEqual(
            certificate["pilot_plan_commitment_sha256"],
            pilot_commitment["commitment_sha256"],
        )
        self.assertEqual(
            pilot_commitment["pilot_terminal_support_sha256"],
            certificate["pilot_terminal_support_sha256"],
        )
        self.assertEqual(
            len(certificate["confirm_terminal_support_sha256"]),
            64,
        )
        tampered_pilot = replace(
            pilot,
            objectives=tuple(
                tuple(pilot.metadata["objective_lower_bounds"])
                for _ in pilot.objectives
            ),
        )
        with self.assertRaisesRegex(
            FixedScheduleCertificateError,
            "does not match",
        ):
            build_regeneration_pilot_plan_commitment_from_spec(
                tampered_pilot,
                specification,
                confirm_particles_per_reference=int(
                    confirm.metadata["particles_per_reference"]
                ),
            )
        uncommitted = certify_fixed_schedule_reference_metrics_from_spec(
            pilot,
            confirm,
            specification,
            certificate_mode="regeneration",
        )
        self.assertEqual(
            uncommitted["pilot_plan_commitment_gate"],
            "MISSING",
        )
        self.assertEqual(uncommitted["formal_packet_gate"], "FAIL")
        posthoc_content_only = (
            certify_fixed_schedule_reference_metrics_from_spec(
                pilot,
                confirm,
                specification,
                certificate_mode="regeneration",
                pilot_plan_commitment=pilot_commitment,
            )
        )
        self.assertEqual(
            posthoc_content_only["pilot_plan_commitment_content_gate"],
            "PASS",
        )
        self.assertEqual(
            posthoc_content_only["pilot_plan_commitment_gate"],
            "MISSING_PRECONFIRM_RUNNER_ORDER_ATTESTATION",
        )
        self.assertEqual(
            posthoc_content_only["formal_packet_gate"],
            "FAIL",
        )
        tampered_commitment = dict(pilot_commitment)
        tampered_commitment["confirm_particles_per_reference"] = 1
        with self.assertRaisesRegex(
            FixedScheduleCertificateError,
            "commitment does not match",
        ):
            certify_fixed_schedule_reference_metrics_from_spec(
                pilot,
                confirm,
                specification,
                certificate_mode="regeneration",
                pilot_plan_commitment=tampered_commitment,
            )
        self.assertLessEqual(
            certificate["cell_cover_archive_size"],
            payload["certified_archive"]["max_size"],
        )
        self.assertEqual(
            certificate[
                "cell_cover_archive_metric_preservation_gate"
            ],
            "PASS",
        )
        self.assertTrue(
            certificate[
                "cell_cover_archive_keeps_dominated_same_cell_witnesses"
            ]
        )

        tampered_objective = copy.deepcopy(payload)
        altered = list(
            tampered_objective["reference_front"]["witnesses"][0][
                "objectives"
            ]
        )
        altered[0] += 0.1
        tampered_objective["reference_front"]["witnesses"][0][
            "objectives"
        ] = altered
        tampered_hash = copy.deepcopy(payload)
        tampered_hash["reference_front"][
            "witness_payload_sha256"
        ] = "0" * 64
        tampered_instance = copy.deepcopy(payload)
        tampered_instance["instance_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            for index, malformed in enumerate(
                (tampered_objective, tampered_hash, tampered_instance)
            ):
                bad_path = Path(directory) / f"bad_{index}.json"
                bad_path.write_text(
                    json.dumps(malformed),
                    encoding="utf-8",
                )
                with self.assertRaises(ValueError):
                    load_fixed_reference_certificate_specification(
                        bad_path,
                        objective_dimension=2,
                        instance=instance,
                    )

    def test_end_to_end_runner_charges_both_streams(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(4, seed=410)
        objectives = tuple(
            instance.evaluate((0,) + tail)
            for tail in itertools.permutations(range(1, 4))
        )
        reference = nondominated_points(objectives)
        smc = load_pareto_smc_specification(
            Path(__file__).parents[1]
            / "benchmarks"
            / "pareto_smc_v10_fixed_smoke_spec.json",
            objective_dimension=2,
        )
        _, upper = analytic_objective_box(instance)
        widths = original_unit_cell_widths(instance, smc)
        igd_bound = math.sqrt(sum(width * width for width in widths))
        hv_bound = shifted_front_hv_deficit_bound(
            reference,
            additive_widths=widths,
            reference=upper,
        )
        payload = {
            "schema": "pareto_smc_fixed_reference_certificate_spec_v1",
            "instance_sha256": instance_sha256(instance),
            "pareto_smc_specification_sha256": smc.sha256,
            "reference_front": {
                "source": "independent_exact_solver",
                "artifact_sha256": self._reference_hash(reference),
                "objectives": reference,
            },
            "streams": {
                "pilot_seed": 411,
                "confirm_seed": 412,
                "independence_model": "ideal_product_random_streams",
            },
            "failure_budgets": {
                "pilot": 0.025,
                "confirm": 0.025,
            },
            "metrics": {
                "igd_p": 2.0,
                "hv_reference": upper,
                "max_igd_bound": 1.1 * igd_bound,
                "max_hv_deficit_bound": 1.1 * hv_bound,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "certificate.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            certificate_spec = (
                load_fixed_reference_certificate_specification(
                    path,
                    objective_dimension=2,
                )
            )
            pair = run_fixed_schedule_pilot_confirm(
                instance,
                pareto_smc_specification=smc,
                certificate_specification=certificate_spec,
                particles_per_reference=4096,
            )
        self.assertEqual(
            pair.certificate["scientific_design_gate"],
            "PASS",
        )
        self.assertEqual(pair.certificate["evaluations_per_stream"], 12288)
        self.assertEqual(
            pair.certificate["total_certificate_evaluations"],
            24576,
        )


if __name__ == "__main__":
    unittest.main()

