from __future__ import annotations

import itertools
import hashlib
import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from mo_nco.contracts import ClaimLevel
from mo_nco.evaluation import CountingTSPInstance
from mo_nco.instance import MultiObjectiveTSPInstance, instance_sha256
from mo_nco.pareto_bounds import (
    certify_independent_cell_probe_bounds,
    nondominated_points,
)
from mo_nco.pareto_cell_certification import (
    CertifiedCellType,
    CellCertifiedParetoSampler,
    augmented_tchebycheff_energy,
    doeblin_minorization_constant,
    mixing_tv_radius,
    original_cell_index,
    original_cell_index_or_none,
    plan_cell_type,
    target_cell_mass_lower_bound,
)
from mo_nco.pareto_fk_certificate import (
    make_bootstrap_fk_plan,
    make_contraction_aware_fk_plan,
    recommend_mutation_steps_for_stage_contraction,
)
from mo_nco.pareto_smc import AnnealedParetoSMCOptimizer
from mo_nco.pareto_cell_spec import (
    CELL_SPEC_SCHEMA,
    load_pareto_cell_certification_specification,
)
from mo_nco.run_pareto_cell_certified_audit import run_audit


class ParetoCellCertificationTests(unittest.TestCase):
    def test_mass_lower_bound_is_valid_on_finite_examples(self) -> None:
        beta = 0.7
        u_max = 1.3
        penalty = 0.8
        for n in range(2, 9):
            for cell_count in range(1, n + 1):
                kappa = cell_count / n
                bound = target_cell_mass_lower_bound(
                    base_cell_mass_lower_bound=kappa,
                    beta=beta,
                    base_energy_upper=u_max,
                    outside_cell_penalty=penalty,
                )
                inside = [u_max * (index + 1) / cell_count for index in range(cell_count)]
                outside = [0.0 for _ in range(n - cell_count)]
                numerator = sum(math.exp(-beta * value) for value in inside)
                denominator = numerator + sum(
                    math.exp(-beta * (value + penalty)) for value in outside
                )
                exact = numerator / denominator
                self.assertGreaterEqual(exact + 1e-12, bound)

    def test_plan_instantiates_finite_step_radius(self) -> None:
        proof = "0" * 64
        contract = CertifiedCellType(
            cell=(0, 0),
            reference_direction=(0.5, 0.5),
            base_cell_mass_lower_bound=0.1,
            base_mass_proof_sha256=proof,
            outside_cell_penalty=0.0,
            global_refresh_probability=1.0,
            mutation_steps=2,
            particle_count=55,
            failure_budget=0.05,
        )
        plan = plan_cell_type(contract, beta=0.1, chebyshev_rho=0.03)
        self.assertGreater(plan.doeblin_minorization, 0.9)
        self.assertLess(plan.mutation_tv_radius, 0.011)
        self.assertGreater(plan.endpoint_cell_hit_lower_bound, 0.0)
        self.assertLessEqual(
            plan.cell_miss_probability_bound,
            contract.failure_budget,
        )

    def test_energy_range_and_box_fail_closed(self) -> None:
        energy = augmented_tchebycheff_energy(
            (5.0, 7.0),
            lower=(0.0, 0.0),
            upper=(10.0, 10.0),
            direction=(0.5, 0.5),
            rho=0.03,
        )
        self.assertGreaterEqual(energy, 0.0)
        self.assertLessEqual(energy, 1.03)
        with self.assertRaises(ValueError):
            augmented_tchebycheff_energy(
                (11.0, 7.0),
                lower=(0.0, 0.0),
                upper=(10.0, 10.0),
                direction=(0.5, 0.5),
                rho=0.03,
            )

    def test_runtime_emits_source_bound_not_exact_claim(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(4, seed=11)
        tours = tuple((0,) + tail for tail in itertools.permutations(range(1, 4)))
        objectives = tuple(instance.evaluate(tour) for tour in tours)
        lower = tuple(min(point[i] for point in objectives) for i in range(2))
        upper = tuple(max(point[i] for point in objectives) for i in range(2))
        widths = tuple((upper[i] - lower[i]) / 2 for i in range(2))
        cell = original_cell_index(
            objectives[0], lower=lower, upper=upper, widths=widths
        )
        count = sum(
            original_cell_index(point, lower=lower, upper=upper, widths=widths) == cell
            for point in objectives
        )
        proof = "1" * 64
        contract = CertifiedCellType(
            cell=cell,
            reference_direction=(0.5, 0.5),
            base_cell_mass_lower_bound=count / len(tours),
            base_mass_proof_sha256=proof,
            outside_cell_penalty=0.0,
            global_refresh_probability=1.0,
            mutation_steps=2,
            particle_count=60,
            failure_budget=0.05,
        )
        counted = CountingTSPInstance(instance, max_evaluations=180)
        result = CellCertifiedParetoSampler(
            counted,
            cell_types=(contract,),
            objective_lower_bounds=lower,
            objective_upper_bounds=upper,
            cell_widths=widths,
            beta=0.1,
            chebyshev_rho=0.03,
            confidence_delta=0.05,
            cell_completeness_proof_sha256=proof,
            objective_box_proof_sha256=proof,
            metric_box_proof_sha256=proof,
            metric_igd_p=2.0,
            max_igd_bound=math.hypot(*widths),
            hv_reference=upper,
            max_hv_deficit_bound=(
                widths[0] * (upper[1] - lower[1])
                + widths[1] * (upper[0] - lower[0])
            ),
        ).run()
        self.assertEqual(
            result.metadata["claim_level"],
            ClaimLevel.PARETO_CELL_SOURCE_BOUND.value,
        )
        self.assertFalse(result.metadata["proof_truth_verified_by_runtime"])
        self.assertEqual(counted.evaluations, 180)

    def test_v3_spec_loader(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(4, seed=2)
        proof = "2" * 64
        payload = {
            "schema": CELL_SPEC_SCHEMA,
            "instance_sha256": instance_sha256(instance),
            "target_safety_box": {
                "source": "problem_specific_theorem",
                "lower": [0.0, 0.0],
                "upper": [20.0, 20.0],
                "proof_artifact_sha256": proof,
                "archive_independent": True,
            },
            "metric_box": {
                "source": "unused_calibration_manifest_with_holdout_proof",
                "lower": [2.0, 2.0],
                "upper": [10.0, 10.0],
                "proof_artifact_sha256": proof,
                "archive_independent": True,
            },
            "metric_nonvacuity": {
                "igd_p": 2.0,
                "max_igd_bound": math.sqrt(2.0),
                "hv_reference": [10.0, 10.0],
                "max_hv_deficit_bound": 16.0,
            },
            "cell_grid": {
                "coordinate_system": "original_objective_units",
                "widths": [1.0, 1.0],
                "archive_independent": True,
            },
            "target": {
                "beta": 0.1,
                "chebyshev_rho": 0.03,
                "family": "uniform_base_cell_penalized_augmented_tchebycheff",
                "base_measure": "uniform_fixed_zero_tours",
            },
            "confidence_delta": 0.05,
            "cell_completeness": {
                "claimed_cells": [[0, 0]],
                "proof_artifact_sha256": proof,
                "source": "problem_specific_theorem",
            },
            "cell_types": [
                {
                    "cell": [0, 0],
                    "reference_direction": [0.5, 0.5],
                    "base_cell_mass_lower_bound": 0.1,
                    "base_mass_proof_sha256": proof,
                    "outside_cell_penalty": 0.0,
                    "global_refresh_probability": 1.0,
                    "mutation_steps": 2,
                    "particle_count": 55,
                    "failure_budget": 0.05,
                }
            ],
            "reporting": {
                "archive_max_size": None,
                "archive_role": "reporting_only",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            spec = load_pareto_cell_certification_specification(
                path,
                objective_dimension=2,
                num_cities=4,
                expected_instance_sha256=instance_sha256(instance),
            )
            invalid_direction = json.loads(json.dumps(payload))
            invalid_direction["cell_types"][0]["reference_direction"] = [
                0.6,
                0.6,
            ]
            path.write_text(
                json.dumps(invalid_direction),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sum to one"):
                load_pareto_cell_certification_specification(
                    path,
                    objective_dimension=2,
                    num_cities=4,
                    expected_instance_sha256=instance_sha256(instance),
                )

            invalid_box = json.loads(json.dumps(payload))
            invalid_box["metric_box"]["upper"] = [21.0, 10.0]
            path.write_text(json.dumps(invalid_box), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contained"):
                load_pareto_cell_certification_specification(
                    path,
                    objective_dimension=2,
                    num_cities=4,
                    expected_instance_sha256=instance_sha256(instance),
                )
        self.assertEqual(spec.cell_types[0].base_cell_mass_lower_bound, 0.1)
        self.assertEqual(spec.target_safety_upper_bounds, (20.0, 20.0))
        self.assertEqual(spec.metric_lower_bounds, (2.0, 2.0))

    def test_metric_box_can_be_tighter_than_target_box(self) -> None:
        self.assertIsNone(
            original_cell_index_or_none(
                (1.0, 1.0),
                lower=(2.0, 2.0),
                upper=(10.0, 10.0),
                widths=(1.0, 1.0),
            )
        )

    def test_geometry_metric_box_need_not_contain_dominated_states(self) -> None:
        certificate = certify_independent_cell_probe_bounds(
            feasible_objectives=((0.0, 10.0), (10.0, 0.0), (100.0, 100.0)),
            probe_objectives=((0.2, 10.0), (10.0, 0.2), (100.0, 100.0)),
            declared_cells=((0, 9), (9, 0)),
            objective_lower=(0.0, 0.0),
            objective_upper=(10.0, 10.0),
            cell_widths_original=(1.0, 1.0),
            source_bound_failure_probability=0.05,
            requested_confidence_delta=0.05,
        )
        self.assertEqual(certificate["design_verdict"], "PASS")
        self.assertEqual(certificate["realized_geometry_verdict"], "PASS")
        self.assertEqual(certificate["probe_points_outside_metric_box"], 1)
        self.assertEqual(
            original_cell_index_or_none(
                (2.5, 3.5),
                lower=(2.0, 2.0),
                upper=(10.0, 10.0),
                widths=(1.0, 1.0),
            ),
            (0, 1),
        )

    def test_geometry_certificate_uses_the_declared_hv_reference(self) -> None:
        certificate = certify_independent_cell_probe_bounds(
            feasible_objectives=((0.0, 1.0), (1.0, 0.0)),
            probe_objectives=((0.0, 1.0), (1.0, 0.0)),
            declared_cells=((0, 1), (1, 0)),
            objective_lower=(0.0, 0.0),
            objective_upper=(1.0, 1.0),
            cell_widths_original=(0.5, 0.5),
            source_bound_failure_probability=0.05,
            requested_confidence_delta=0.05,
            hv_reference=(2.0, 2.0),
        )
        self.assertEqual(certificate["hv_reference"], [2.0, 2.0])
        self.assertAlmostEqual(
            certificate["global_slab_hv_deficit_bound_original"],
            2.0,
        )

    def test_bootstrap_fk_constant_is_explicit(self) -> None:
        plan = make_bootstrap_fk_plan(
            (0.0, 0.05, 0.1),
            potential_upper_bound=1.03,
            particle_count=128,
            failure_budget=0.05,
            target_cell_mass_lower_bound=0.2,
        )
        expected_ratios = tuple(
            math.exp((0.1 - beta) * 1.03)
            for beta in (0.0, 0.05, 0.1)
        )
        for actual, expected in zip(
            plan.backward_oscillation_ratios,
            expected_ratios,
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(
            plan.finite_particle_mse_constant,
            0.25 * sum(expected_ratios) ** 2,
        )

    def test_published_concentration_gate_rejects_missing_refresh(
        self,
    ) -> None:
        plan = make_contraction_aware_fk_plan(
            (0.0, 0.05, 0.1),
            potential_upper_bound=1.03,
            global_refresh_probability=0.0,
            mutation_steps_by_stage=(5, 7),
            particle_count=128,
            observable_count=4,
            failure_budget=0.05,
        )
        self.assertFalse(plan.published_concentration_gate)
        self.assertIsNone(plan.theorem_a)
        self.assertIsNone(plan.simultaneous_error_radius_raw)
        self.assertEqual(plan.simultaneous_error_radius, 1.0)

    def test_contraction_aware_step_planner_meets_published_gate(
        self,
    ) -> None:
        schedule = (0.0, 0.1, 0.2)
        steps = recommend_mutation_steps_for_stage_contraction(
            schedule,
            potential_upper_bound=1.03,
            global_refresh_probability=1.0,
            maximum_mixing_product=0.1,
        )
        plan = make_contraction_aware_fk_plan(
            schedule,
            potential_upper_bound=1.03,
            global_refresh_probability=1.0,
            mutation_steps_by_stage=steps,
            particle_count=512,
            observable_count=8,
            failure_budget=0.05,
        )
        self.assertTrue(plan.published_concentration_gate)
        self.assertLessEqual(plan.regularity_product, 0.1 + 1e-15)
        self.assertIsNotNone(plan.theorem_a)
        self.assertLess(plan.simultaneous_error_radius, 1.0)

    def test_integer_counts_and_grid_coordinates_fail_closed(self) -> None:
        proof = "3" * 64
        base_contract = {
            "cell": (0, 0),
            "reference_direction": (0.5, 0.5),
            "base_cell_mass_lower_bound": 0.1,
            "base_mass_proof_sha256": proof,
            "outside_cell_penalty": 0.0,
            "global_refresh_probability": 1.0,
            "mutation_steps": 2,
            "particle_count": 8,
            "failure_budget": 0.05,
        }
        with self.assertRaisesRegex(Exception, "nonnegative integer"):
            mixing_tv_radius(0.5, 2.5)  # type: ignore[arg-type]
        with self.assertRaisesRegex(Exception, "positive integer"):
            make_bootstrap_fk_plan(
                (0.0, 0.1),
                potential_upper_bound=1.03,
                particle_count=8.5,  # type: ignore[arg-type]
                failure_budget=0.05,
            )
        with self.assertRaisesRegex(Exception, "positive integer"):
            plan_cell_type(
                CertifiedCellType(
                    **{
                        **base_contract,
                        "particle_count": 8.5,
                    }
                ),
                beta=0.1,
                chebyshev_rho=0.03,
            )
        with self.assertRaisesRegex(ValueError, "nonnegative integers"):
            certify_independent_cell_probe_bounds(
                feasible_objectives=((0.0, 1.0), (1.0, 0.0)),
                probe_objectives=((0.0, 1.0), (1.0, 0.0)),
                declared_cells=((0.0, 1),),  # type: ignore[arg-type]
                objective_lower=(0.0, 0.0),
                objective_upper=(1.0, 1.0),
                cell_widths_original=(0.5, 0.5),
                source_bound_failure_probability=0.05,
                requested_confidence_delta=0.05,
            )

        instance = MultiObjectiveTSPInstance.random_biobjective(4, seed=23)
        for replacement in (
            {"mutation_steps": 2.5},
            {"particle_count": 8.5},
            {"cell": (2, 0)},
        ):
            payload = dict(base_contract)
            payload.update(replacement)
            with self.assertRaises(Exception):
                CellCertifiedParetoSampler(
                    instance,
                    cell_types=(CertifiedCellType(**payload),),
                    objective_lower_bounds=(0.0, 0.0),
                    objective_upper_bounds=(20.0, 20.0),
                    metric_lower_bounds=(0.0, 0.0),
                    metric_upper_bounds=(2.0, 2.0),
                    cell_widths=(1.0, 1.0),
                    beta=0.1,
                    chebyshev_rho=0.03,
                    confidence_delta=0.05,
                )

    def test_always_resampling_branch_emits_B_L(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(4, seed=17)
        result = AnnealedParetoSMCOptimizer(
            instance,
            particles_per_reference=2,
            evaluations=12,
            seed=3,
            beta_schedule=(0.0, 0.1, 0.2),
            reference_directions=((0.25, 0.75), (0.75, 0.25)),
            num_reference_types=2,
            ess_threshold=0.5,
            resampling_policy="always",
            chebyshev_rho=0.03,
            global_refresh_probability=1.0,
        ).run()
        self.assertEqual(
            result.metadata["claim_level"],
            ClaimLevel.PARETO_SMC_BOOTSTRAP_BOUND.value,
        )
        certificate = result.metadata[
            "bootstrap_feynman_kac_finite_particle_certificate"
        ]
        self.assertIsNotNone(certificate)
        self.assertGreater(certificate["per_type_mse_constant_B_L_2"], 0.0)
        self.assertEqual(result.metadata["bootstrap_mutations_by_stage"], (1, 1))
        contraction = result.metadata[
            "contraction_aware_fixed_schedule_certificate"
        ]
        self.assertIsNotNone(contraction)
        self.assertEqual(contraction["mutation_steps_by_stage"], (1, 1))
        self.assertTrue(contraction["published_concentration_gate"])
        self.assertLess(contraction["regularity_product"], 0.5)

    def test_predeclared_stage_mutation_counts_bind_the_exact_budget(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(4, seed=31)
        result = AnnealedParetoSMCOptimizer(
            instance,
            particles_per_reference=2,
            evaluations=6,
            seed=32,
            beta_schedule=(0.0, 0.1, 0.2),
            reference_directions=((0.5, 0.5),),
            resampling_policy="always",
            mutation_steps_by_stage=(0, 2),
            global_refresh_probability=1.0,
        ).run()
        self.assertEqual(result.metadata["bootstrap_mutations_by_stage"], (0, 2))
        self.assertEqual(
            result.metadata["mutation_steps_by_stage_predeclared"],
            (0, 2),
        )
        self.assertEqual(result.metadata["evaluations_used"], 6)

        with self.assertRaisesRegex(ValueError, "require 6 evaluations"):
            AnnealedParetoSMCOptimizer(
                instance,
                particles_per_reference=2,
                evaluations=8,
                seed=32,
                beta_schedule=(0.0, 0.1, 0.2),
                reference_directions=((0.5, 0.5),),
                resampling_policy="always",
                mutation_steps_by_stage=(0, 2),
                global_refresh_probability=1.0,
            )

    def test_exact_audit_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proof_path = Path(directory) / "proof.json"
            report = run_audit(
                output=Path(directory) / "audit.json",
                proof_output=proof_path,
                cities=4,
                instance_seed=20260726,
                algorithm_seed=0,
                confidence_delta=0.05,
            )
            proof_file_hash = hashlib.sha256(proof_path.read_bytes()).hexdigest()
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(
            report["verified_claim_level"],
            ClaimLevel.PARETO_CELL_CERTIFIED.value,
        )
        self.assertEqual(
            report["proof_artifact_hash_contract"],
            "sha256_of_canonical_utf8_json_no_trailing_newline",
        )
        self.assertEqual(
            proof_file_hash,
            report["proof_artifact_sha256"],
        )


if __name__ == "__main__":
    unittest.main()

