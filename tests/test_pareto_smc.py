from __future__ import annotations

import json
import math
import random
import unittest

from mo_nco.evaluation import CountingTSPInstance, evaluation_count
from mo_nco.contracts import ClaimLevel
from mo_nco.instance import MultiObjectiveTSPInstance
from mo_nco.pareto_smc import (
    AnnealedParetoSMCOptimizer,
    ObjectiveBoundsViolation,
)


def _integer_symmetric_instance(
    num_cities: int,
    num_objectives: int,
    seed: int,
) -> MultiObjectiveTSPInstance:
    rng = random.Random(seed)
    matrices = []
    for _ in range(num_objectives):
        matrix = [[0.0] * num_cities for _ in range(num_cities)]
        for left in range(num_cities):
            for right in range(left + 1, num_cities):
                value = float(rng.randint(1, 1000))
                matrix[left][right] = value
                matrix[right][left] = value
        matrices.append(tuple(tuple(row) for row in matrix))
    return MultiObjectiveTSPInstance.from_distance_matrices(
        tuple(matrices),
        name=f"integer_symmetric_{num_cities}_{num_objectives}_{seed}",
    )


class AnnealedParetoSMCTests(unittest.TestCase):
    def test_local_two_opt_uses_incremental_interface_without_changing_budget(self) -> None:
        base = _integer_symmetric_instance(8, 2, 420)

        class AuditedIncrementalInstance:
            num_cities = base.num_cities
            num_objectives = base.num_objectives
            objective_scale_estimates = base.objective_scale_estimates
            distance_matrices = base.distance_matrices
            symmetric_objectives = base.symmetric_objectives
            exact_two_opt_delta_in_binary64 = (
                base.exact_two_opt_delta_in_binary64
            )

            def __init__(self) -> None:
                self.full_calls = 0
                self.incremental_calls = 0

            def evaluate(self, tour):  # type: ignore[no-untyped-def]
                self.full_calls += 1
                return base.evaluate(tour)

            def evaluate_two_opt(  # type: ignore[no-untyped-def]
                self,
                tour,
                current_objectives,
                i,
                j,
            ):
                self.incremental_calls += 1
                return base.evaluate_two_opt(tour, current_objectives, i, j)

        instance = AuditedIncrementalInstance()
        result = AnnealedParetoSMCOptimizer(
            instance,  # type: ignore[arg-type]
            particles_per_reference=3,
            evaluations=18,
            seed=421,
            beta_schedule=(0.0, 1.0),
            reference_directions=((0.5, 0.5),),
            epsilon=(0.1, 0.1),
        ).run()

        self.assertEqual(result.metadata["implementation_version"], "0.6.0")
        self.assertTrue(result.metadata["local_two_opt_incremental_enabled"])
        self.assertEqual(instance.full_calls, 3)
        self.assertEqual(instance.incremental_calls, 15)
        self.assertEqual(result.metadata["evaluations_used"], 18)
        self.assertEqual(
            result.metadata["local_two_opt_incremental_evaluations"],
            15,
        )
        self.assertEqual(
            result.metadata["local_two_opt_full_fallback_evaluations"],
            0,
        )
        self.assertEqual(result.metadata["local_two_opt_proposal_evaluations"], 15)
        self.assertEqual(result.metadata["global_refresh_proposal_evaluations"], 0)
        self.assertTrue(
            result.metadata["mutation_evaluation_path_accounting_complete"]
        )

    def test_local_incremental_objectives_match_full_multiobjective_tours_randomly(self) -> None:
        for instance_seed in range(5):
            instance = _integer_symmetric_instance(
                10,
                3,
                430 + instance_seed,
            )
            result = AnnealedParetoSMCOptimizer(
                instance,
                particles_per_reference=3,
                evaluations=36,
                seed=440 + instance_seed,
                beta_schedule=(0.0, 0.4, 1.2),
                reference_directions=((0.2, 0.3, 0.5),),
                epsilon=(0.1, 0.1, 0.1),
            ).run()

            mutations = tuple(
                mutation
                for stage in result.metadata["stage_ledger"][1:]
                for reference in stage["references"]
                for mutation in reference["mutations"]
            )
            self.assertEqual(len(mutations), 33)
            for mutation in mutations:
                expected = instance.evaluate(mutation["proposed_tour"])
                for observed, target in zip(
                    mutation["proposed_objective"],
                    expected,
                ):
                    self.assertTrue(
                        math.isclose(
                            observed,
                            target,
                            rel_tol=1e-13,
                            abs_tol=1e-13,
                        )
                    )
                self.assertEqual(
                    mutation["objective_evaluation_kind"],
                    "exact_incremental_two_opt",
                )

    def test_incremental_and_full_local_evaluation_preserve_seeded_state_exactly(self) -> None:
        matrix_rng = random.Random(450)
        matrices = []
        for _ in range(2):
            matrix = [[0.0] * 12 for _ in range(12)]
            for left in range(12):
                for right in range(left + 1, 12):
                    distance = float(matrix_rng.randint(1, 100))
                    matrix[left][right] = distance
                    matrix[right][left] = distance
            matrices.append(tuple(tuple(row) for row in matrix))
        base = MultiObjectiveTSPInstance.from_distance_matrices(
            tuple(matrices),
            name="exact_integer_biobjective",
        )

        kwargs = {
            "particles_per_reference": 4,
            "evaluations": 96,
            "seed": 451,
            "beta_schedule": (0.0, 0.5, 1.5),
            "reference_directions": ((0.7, 0.3), (0.3, 0.7)),
            "epsilon": (0.05, 0.05),
            "global_refresh_probability": 0.25,
        }
        incremental = AnnealedParetoSMCOptimizer(base, **kwargs).run()
        full = AnnealedParetoSMCOptimizer(
            base,
            enable_exact_incremental_two_opt=False,
            **kwargs,
        ).run()

        self.assertEqual(incremental.particles, full.particles)
        self.assertEqual(incremental.objectives, full.objectives)
        self.assertEqual(
            incremental.metadata["final_normalized_weights_by_reference"],
            full.metadata["final_normalized_weights_by_reference"],
        )
        self.assertEqual(
            incremental.metadata[
                "final_log_normalizer_estimates_by_reference"
            ],
            full.metadata["final_log_normalizer_estimates_by_reference"],
        )
        self.assertEqual(
            incremental.metadata["accepted_mutations"],
            full.metadata["accepted_mutations"],
        )
        self.assertEqual(incremental.archive.entries, full.archive.entries)

        def mutation_records(result):  # type: ignore[no-untyped-def]
            return tuple(
                mutation
                for stage in result.metadata["stage_ledger"][1:]
                for reference in stage["references"]
                for mutation in reference["mutations"]
            )

        incremental_mutations = mutation_records(incremental)
        full_mutations = mutation_records(full)
        self.assertEqual(
            tuple(
                (
                    mutation["proposal_kind"],
                    mutation["proposed_tour"],
                    mutation["log_uniform"],
                    mutation["accepted"],
                )
                for mutation in incremental_mutations
            ),
            tuple(
                (
                    mutation["proposal_kind"],
                    mutation["proposed_tour"],
                    mutation["log_uniform"],
                    mutation["accepted"],
                )
                for mutation in full_mutations
            ),
        )
        for incremental_mutation, full_mutation in zip(
            incremental_mutations,
            full_mutations,
        ):
            self.assertEqual(
                incremental_mutation["log_alpha"],
                full_mutation["log_alpha"],
            )
        self.assertGreater(
            incremental.metadata["local_two_opt_incremental_evaluations"],
            0,
        )
        self.assertGreater(
            full.metadata["local_two_opt_full_fallback_evaluations"],
            0,
        )
        self.assertEqual(
            full.metadata["objective_evaluation_contract"],
            "full_tour_all_proposals_v1",
        )
        self.assertFalse(
            full.metadata["exact_incremental_two_opt_requested"]
        )

    def test_global_refresh_keeps_full_tour_evaluation_contract(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(8, seed=460)

        class AuditedGlobalInstance:
            num_cities = base.num_cities
            num_objectives = base.num_objectives
            objective_scale_estimates = base.objective_scale_estimates
            distance_matrices = base.distance_matrices
            symmetric_objectives = base.symmetric_objectives
            exact_two_opt_delta_in_binary64 = (
                base.exact_two_opt_delta_in_binary64
            )

            def __init__(self) -> None:
                self.full_calls = 0
                self.incremental_calls = 0

            def evaluate(self, tour):  # type: ignore[no-untyped-def]
                self.full_calls += 1
                return base.evaluate(tour)

            def evaluate_two_opt(  # type: ignore[no-untyped-def]
                self,
                tour,
                current_objectives,
                i,
                j,
            ):
                self.incremental_calls += 1
                return base.evaluate_two_opt(tour, current_objectives, i, j)

        instance = AuditedGlobalInstance()
        result = AnnealedParetoSMCOptimizer(
            instance,  # type: ignore[arg-type]
            particles_per_reference=3,
            evaluations=18,
            seed=461,
            beta_schedule=(0.0, 1.0),
            reference_directions=((0.5, 0.5),),
            epsilon=(0.1, 0.1),
            global_refresh_probability=1.0,
        ).run()

        self.assertEqual(instance.full_calls, 18)
        self.assertEqual(instance.incremental_calls, 0)
        self.assertEqual(result.metadata["global_refresh_full_tour_evaluations"], 15)
        self.assertEqual(result.metadata["global_refresh_proposal_evaluations"], 15)
        self.assertEqual(result.metadata["local_two_opt_incremental_evaluations"], 0)
        self.assertEqual(result.metadata["full_tour_evaluations"], 18)
        self.assertTrue(
            result.metadata["mutation_evaluation_path_accounting_complete"]
        )
        self.assertTrue(
            all(
                mutation["objective_evaluation_kind"]
                == "full_tour_global_refresh"
                for stage in result.metadata["stage_ledger"][1:]
                for reference in stage["references"]
                for mutation in reference["mutations"]
            )
        )

    def test_nonsymmetric_local_moves_fail_safe_to_counted_full_evaluation(self) -> None:
        symmetric = (
            (0.0, 1.0, 2.0, 3.0, 4.0),
            (1.0, 0.0, 1.5, 2.5, 3.5),
            (2.0, 1.5, 0.0, 1.0, 2.0),
            (3.0, 2.5, 1.0, 0.0, 1.5),
            (4.0, 3.5, 2.0, 1.5, 0.0),
        )
        asymmetric = (
            (0.0, 2.0, 4.0, 6.0, 8.0),
            (1.0, 0.0, 3.0, 5.0, 7.0),
            (9.0, 2.0, 0.0, 4.0, 6.0),
            (7.0, 5.0, 3.0, 0.0, 2.0),
            (5.0, 4.0, 3.0, 2.0, 0.0),
        )
        base = MultiObjectiveTSPInstance.from_distance_matrices(
            (symmetric, asymmetric),
            name="mixed_symmetry",
        )
        counted = CountingTSPInstance(base, max_evaluations=18)
        result = AnnealedParetoSMCOptimizer(
            counted,  # type: ignore[arg-type]
            particles_per_reference=3,
            evaluations=18,
            seed=471,
            beta_schedule=(0.0, 1.0),
            reference_directions=((0.5, 0.5),),
            epsilon=(1.0, 1.0),
        ).run()

        self.assertEqual(counted.evaluations, 18)
        self.assertFalse(result.metadata["local_two_opt_incremental_enabled"])
        self.assertEqual(result.metadata["local_two_opt_incremental_evaluations"], 0)
        self.assertEqual(
            result.metadata["local_two_opt_full_fallback_evaluations"],
            15,
        )
        self.assertEqual(result.metadata["full_tour_evaluations"], 18)
        self.assertTrue(result.metadata["evaluation_path_accounting_complete"])
        for tour, objective in zip(result.particles, result.objectives):
            self.assertEqual(objective, base.evaluate(tour))

    def test_counting_wrapper_reports_full_fallback_when_base_lacks_delta_interface(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(8, seed=480)

        class FullEvaluationOnlyBase:
            num_cities = base.num_cities
            num_objectives = base.num_objectives
            objective_scale_estimates = base.objective_scale_estimates
            distance_matrices = base.distance_matrices
            symmetric_objectives = base.symmetric_objectives

            @staticmethod
            def evaluate(tour):  # type: ignore[no-untyped-def]
                return base.evaluate(tour)

            @staticmethod
            def validate_tour(tour):  # type: ignore[no-untyped-def]
                base.validate_tour(tour)

        counted = CountingTSPInstance(
            FullEvaluationOnlyBase(),  # type: ignore[arg-type]
            max_evaluations=18,
        )
        result = AnnealedParetoSMCOptimizer(
            counted,  # type: ignore[arg-type]
            particles_per_reference=3,
            evaluations=18,
            seed=481,
            beta_schedule=(0.0, 1.0),
            reference_directions=((0.5, 0.5),),
            epsilon=(0.1, 0.1),
        ).run()

        self.assertEqual(counted.evaluations, 18)
        self.assertEqual(result.metadata["local_two_opt_incremental_evaluations"], 0)
        self.assertEqual(
            result.metadata["local_two_opt_full_fallback_evaluations"],
            15,
        )
        self.assertEqual(result.metadata["full_tour_evaluations"], 18)

    def test_counting_wrapper_falls_back_once_when_delta_interface_is_unsupported(self) -> None:
        base = _integer_symmetric_instance(8, 2, 490)

        class UnsupportedDeltaBase:
            num_cities = base.num_cities
            num_objectives = base.num_objectives
            objective_scale_estimates = base.objective_scale_estimates
            distance_matrices = base.distance_matrices
            symmetric_objectives = base.symmetric_objectives
            exact_two_opt_delta_in_binary64 = (
                base.exact_two_opt_delta_in_binary64
            )

            def __init__(self) -> None:
                self.delta_attempts = 0

            @staticmethod
            def evaluate(tour):  # type: ignore[no-untyped-def]
                return base.evaluate(tour)

            def evaluate_two_opt(self, *_args):  # type: ignore[no-untyped-def]
                self.delta_attempts += 1
                raise NotImplementedError

            @staticmethod
            def validate_tour(tour):  # type: ignore[no-untyped-def]
                base.validate_tour(tour)

        unsupported = UnsupportedDeltaBase()
        counted = CountingTSPInstance(unsupported, max_evaluations=18)  # type: ignore[arg-type]
        result = AnnealedParetoSMCOptimizer(
            counted,  # type: ignore[arg-type]
            particles_per_reference=3,
            evaluations=18,
            seed=491,
            beta_schedule=(0.0, 1.0),
            reference_directions=((0.5, 0.5),),
            epsilon=(0.1, 0.1),
        ).run()

        self.assertEqual(counted.evaluations, 18)
        self.assertEqual(unsupported.delta_attempts, 1)
        self.assertFalse(result.metadata["local_two_opt_incremental_enabled"])
        self.assertEqual(result.metadata["local_two_opt_incremental_evaluations"], 0)
        self.assertEqual(
            result.metadata["local_two_opt_full_fallback_evaluations"],
            15,
        )
        self.assertTrue(result.metadata["evaluation_path_accounting_complete"])

    def test_summary_trace_preserves_seeded_algorithm_state(self) -> None:
        def run(trace_level: str):  # type: ignore[no-untyped-def]
            instance = MultiObjectiveTSPInstance.random_biobjective(
                4,
                seed=410,
            )
            return AnnealedParetoSMCOptimizer(
                instance,
                particles_per_reference=32,
                evaluations=96,
                seed=411,
                beta_schedule=(0.0, 0.1),
                reference_directions=((0.5, 0.5),),
                epsilon=(1.0, 1.0),
                resampling_policy="always",
                mutation_steps_by_stage=(2,),
                global_refresh_probability=1.0,
                archive_max_size=1,
                audit_trace_level=trace_level,
            ).run()

        full = run("full")
        summary = run("summary")
        self.assertEqual(full.particles, summary.particles)
        self.assertEqual(full.objectives, summary.objectives)
        self.assertEqual(
            full.metadata["final_normalized_weights_by_reference"],
            summary.metadata["final_normalized_weights_by_reference"],
        )
        self.assertEqual(
            full.metadata["accepted_mutations"],
            summary.metadata["accepted_mutations"],
        )
        self.assertTrue(
            summary.metadata["stage_ledger"][1]["references"][0][
                "trace_compacted"
            ]
        )

    def test_strict_smc_consumes_run_local_budget_with_frozen_predeclared_context(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(8, seed=901)
        counted = CountingTSPInstance(base, max_evaluations=96, evaluations=7)
        result = AnnealedParetoSMCOptimizer(
            counted,  # type: ignore[arg-type]
            particles_per_reference=3,
            evaluations=72,
            seed=902,
            beta_schedule=(0.0, 0.5, 1.0, 2.0),
            reference_directions=((0.8, 0.2), (0.2, 0.8)),
            epsilon=(0.2, 0.2),
        ).run()

        self.assertEqual(evaluation_count(counted), 79)
        self.assertEqual(
            result.metadata["algorithm_contract"],
            "annealed_pareto_smc_feynman_kac_v4",
        )
        self.assertEqual(
            result.metadata["claim_level"],
            ClaimLevel.PARETO_SMC_MECHANICAL.value,
        )
        self.assertEqual(result.metadata["evaluation_counter_start"], 7)
        self.assertEqual(result.metadata["evaluation_budget"], 72)
        self.assertEqual(result.metadata["evaluations_used"], 72)
        self.assertEqual(
            result.metadata["initial_population_full_tour_evaluations"],
            6,
        )
        self.assertEqual(
            result.metadata["local_two_opt_incremental_evaluations"],
            0,
        )
        self.assertEqual(
            result.metadata["local_two_opt_full_fallback_evaluations"],
            66,
        )
        self.assertEqual(result.metadata["global_refresh_full_tour_evaluations"], 0)
        self.assertEqual(result.metadata["full_tour_evaluations"], 72)
        self.assertTrue(result.metadata["evaluation_path_accounting_complete"])
        self.assertTrue(result.metadata["context_frozen"])
        self.assertEqual(result.metadata["context_refresh_count"], 0)
        self.assertEqual(result.metadata["bounds_source"], "analytic_distance_matrix_box")
        self.assertFalse(result.metadata["archive_feedback"])
        self.assertEqual(result.metadata["archive_role"], "reporting_only_no_smc_feedback")
        self.assertFalse(result.metadata["machine_exact_detailed_balance_claimed"])
        self.assertEqual(
            result.metadata["proposal"],
            "uniform_symmetric_two_opt",
        )
        self.assertEqual(result.metadata["global_refresh_probability"], 0.0)
        self.assertIsNone(result.metadata["global_refresh_base_measure"])
        self.assertTrue(
            all(
                mutation["proposal_kind"] == "uniform_symmetric_two_opt"
                for stage in result.metadata["stage_ledger"][1:]
                for reference in stage["references"]
                for mutation in reference["mutations"]
            )
        )
        self.assertEqual(len(result.metadata["stage_ledger"]), 4)
        self.assertEqual(len(result.particles), 6)
        self.assertEqual(len(result.objectives), 6)
        for tour, objective in zip(result.particles, result.objectives):
            base.validate_tour(tour)
            for observed, expected in zip(objective, base.evaluate(tour)):
                self.assertAlmostEqual(observed, expected, places=12)

    def test_stage_ledger_recomputes_incremental_weights_ess_and_typed_resampling(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(9, seed=903)
        result = AnnealedParetoSMCOptimizer(
            instance,
            particles_per_reference=4,
            evaluations=24,
            seed=904,
            beta_schedule=(0.0, 100.0),
            reference_directions=((0.999, 0.001), (0.001, 0.999)),
            epsilon=(1e-6, 1e-6),
            ess_threshold=1.0,
        ).run()

        stage = result.metadata["stage_ledger"][1]
        self.assertEqual(stage["delta_beta"], 100.0)
        self.assertTrue(stage["target_frozen_during_stage"])
        self.assertEqual(len(stage["references"]), 2)
        self.assertGreaterEqual(result.metadata["resampling_events"], 1)
        for reference in stage["references"]:
            energies = reference["pre_weight_energies"]
            increments = reference["incremental_log_weights"]
            probabilities = reference["normalized_weights_before_resampling"]
            self.assertEqual(len(energies), 4)
            self.assertEqual(len(increments), 4)
            self.assertEqual(len(probabilities), 4)
            for energy, increment in zip(energies, increments):
                self.assertAlmostEqual(increment, -100.0 * energy, places=12)
            self.assertAlmostEqual(sum(probabilities), 1.0, places=12)
            expected_ess = 1.0 / sum(weight * weight for weight in probabilities)
            self.assertAlmostEqual(reference["ess_before_resampling"], expected_ess, places=12)
            if reference["resampled"]:
                self.assertEqual(reference["resampling_method"], "multinomial")
                self.assertTrue(
                    all(0 <= ancestor < 4 for ancestor in reference["ancestor_indices"])
                )
                self.assertTrue(
                    all(
                        math.isclose(weight, 0.25, abs_tol=1e-15)
                        for weight in reference["normalized_weights_after_resampling"]
                    )
                )

    def test_mutation_ledger_audits_full_fallback_log_domain_exact_mh(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(8, seed=905)

        class FullStateOnlyInstance:
            num_cities = base.num_cities
            num_objectives = base.num_objectives
            objective_scale_estimates = base.objective_scale_estimates
            distance_matrices = base.distance_matrices

            @staticmethod
            def evaluate(tour):  # type: ignore[no-untyped-def]
                return base.evaluate(tour)

            @staticmethod
            def evaluate_two_opt(*_args):  # type: ignore[no-untyped-def]
                raise AssertionError("strict SMC mutation must recompute the tour state function")

        result = AnnealedParetoSMCOptimizer(
            FullStateOnlyInstance(),  # type: ignore[arg-type]
            particles_per_reference=3,
            evaluations=18,
            seed=906,
            beta_schedule=(0.0, 2.0),
            reference_directions=((0.5, 0.5),),
            epsilon=(0.01, 0.01),
            ess_threshold=0.5,
        ).run()

        reference = result.metadata["stage_ledger"][1]["references"][0]
        self.assertEqual(len(reference["mutations"]), 15)
        for mutation in reference["mutations"]:
            self.assertEqual(
                mutation["proposed_objective"],
                base.evaluate(mutation["proposed_tour"]),
            )
            expected_delta = mutation["proposed_energy"] - mutation["current_energy"]
            self.assertAlmostEqual(mutation["delta_energy"], expected_delta, places=15)
            expected_log_alpha = min(0.0, -2.0 * expected_delta)
            self.assertAlmostEqual(mutation["log_alpha"], expected_log_alpha, places=15)
            expected_accepted = (
                mutation["log_uniform"] == "-inf"
                or mutation["log_uniform"] < mutation["log_alpha"]
            )
            self.assertEqual(mutation["accepted"], expected_accepted)
        self.assertEqual(
            result.metadata["objective_evaluation_contract"],
            (
                "initial_full_tour_local_exact_incremental_with_fail_safe_full_"
                "fallback_global_refresh_full_tour"
            ),
        )
        self.assertEqual(
            result.metadata["local_two_opt_evaluation_contract"],
            "symmetric_nonnegative_integer_binary64_safe_delta_else_full_tour",
        )
        self.assertEqual(
            result.metadata["local_two_opt_full_fallback_evaluations"],
            15,
        )
        self.assertLessEqual(
            result.metadata["db_max_abs_log_residual_real_arithmetic_identity"],
            1e-12,
        )

    def test_archive_capacity_cannot_change_the_smc_state_or_ledger(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=907)
        kwargs = {
            "particles_per_reference": 3,
            "evaluations": 48,
            "seed": 908,
            "beta_schedule": (0.0, 0.5, 1.5),
            "reference_directions": ((0.75, 0.25), (0.25, 0.75)),
            "epsilon": (0.05, 0.05),
            "ess_threshold": 0.9,
        }
        tiny_archive = AnnealedParetoSMCOptimizer(
            instance,
            archive_max_size=1,
            **kwargs,
        ).run()
        large_archive = AnnealedParetoSMCOptimizer(
            instance,
            archive_max_size=200,
            **kwargs,
        ).run()

        self.assertEqual(tiny_archive.particles, large_archive.particles)
        self.assertEqual(tiny_archive.objectives, large_archive.objectives)
        self.assertEqual(
            tiny_archive.metadata["stage_ledger"],
            large_archive.metadata["stage_ledger"],
        )
        self.assertEqual(
            tiny_archive.metadata["final_normalized_weights_by_reference"],
            large_archive.metadata["final_normalized_weights_by_reference"],
        )
        self.assertEqual(tiny_archive.metadata["archive_kernel_reads"], 0)
        self.assertEqual(large_archive.metadata["archive_kernel_reads"], 0)

    def test_reporting_cell_observer_retains_one_representative_for_every_queried_cell(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=909)
        result = AnnealedParetoSMCOptimizer(
            instance,
            particles_per_reference=3,
            evaluations=30,
            seed=910,
            beta_schedule=(0.0, 0.5, 1.0),
            reference_directions=((0.5, 0.5),),
            epsilon=(0.08, 0.08),
        ).run()
        metadata = result.metadata
        ledger = metadata["stage_ledger"]

        queried_cells = set()
        for reference in ledger[0]["references"]:
            queried_cells.update(reference["epsilon_cells"])
        for stage in ledger[1:]:
            for reference in stage["references"]:
                queried_cells.update(
                    mutation["proposed_epsilon_cell"]
                    for mutation in reference["mutations"]
                )

        representatives = metadata["cell_representatives"]
        representative_cells = {
            tuple(representative["epsilon_cell"])
            for representative in representatives
        }
        self.assertEqual(representative_cells, queried_cells)
        self.assertEqual(len(representatives), metadata["queried_epsilon_cell_count"])
        for representative in representatives:
            self.assertEqual(
                tuple(representative["epsilon_cell"]),
                self._cell_from_metadata(
                    representative["objectives"],
                    metadata,
                ),
            )
            instance.validate_tour(representative["tour"])
            for observed, expected in zip(
                representative["objectives"],
                instance.evaluate(tuple(representative["tour"])),
            ):
                self.assertAlmostEqual(observed, expected, places=12)
        self.assertFalse(metadata["cell_observer_feedback"])
        self.assertEqual(
            metadata["ordinary_igd_same_cell_support"],
            "terminal_weighted_support_or_reporting_cell_observer",
        )
        self.assertEqual(
            metadata["pareto_archive_metric_scope"],
            "additive_igd_plus_and_hypervolume_not_unconditional_ordinary_igd",
        )

    def test_explicit_objective_box_is_validated_on_every_query_and_fails_closed(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=911)
        with self.assertRaises(ObjectiveBoundsViolation):
            AnnealedParetoSMCOptimizer(
                instance,
                particles_per_reference=2,
                evaluations=4,
                seed=912,
                beta_schedule=(0.0, 1.0),
                reference_directions=((0.5, 0.5),),
                objective_lower_bounds=(0.0, 0.0),
                objective_upper_bounds=(0.1, 0.1),
                epsilon=(0.01, 0.01),
            )

        result = AnnealedParetoSMCOptimizer(
            instance,
            particles_per_reference=2,
            evaluations=4,
            seed=912,
            beta_schedule=(0.0, 1.0),
            reference_directions=((0.5, 0.5),),
            objective_lower_bounds=(0.0, 0.0),
            objective_upper_bounds=(100.0, 100.0),
            epsilon=(1.0, 1.0),
        ).run()
        self.assertEqual(result.metadata["bounds_source"], "explicit_predeclared_box")
        self.assertTrue(result.metadata["bounds_violations_fail_closed"])

    def test_formal_objective_path_rejects_one_ulp_box_escape_without_clamping(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=913)
        optimizer = AnnealedParetoSMCOptimizer(
            instance,
            particles_per_reference=1,
            evaluations=3,
            seed=914,
            beta_schedule=(0.0, 1.0),
            reference_directions=((0.5, 0.5),),
            objective_lower_bounds=(0.0, 0.0),
            objective_upper_bounds=(100.0, 100.0),
            epsilon=(1.0, 1.0),
        )

        with self.assertRaises(ObjectiveBoundsViolation):
            optimizer.target_energy(
                (math.nextafter(100.0, math.inf), 50.0),
                reference_index=0,
            )
        with self.assertRaises(ObjectiveBoundsViolation):
            optimizer.target_energy(
                (math.nextafter(0.0, -math.inf), 50.0),
                reference_index=0,
            )

        result = optimizer.run()
        self.assertEqual(
            result.metadata["algorithm_identity"],
            "typed_annealed_independent_mh_chain_per_type",
        )
        self.assertFalse(result.metadata["population_interaction_present"])
        self.assertEqual(
            result.metadata["normalized_objective_clipping_contract"],
            "disabled_fail_closed_v2",
        )

    def test_default_box_and_epsilon_are_archive_independent_analytic_constants(self) -> None:
        instance = MultiObjectiveTSPInstance.from_distance_matrices(
            (
                (
                    (0.0, 1.0, 2.0, 3.0),
                    (1.0, 0.0, 4.0, 2.0),
                    (2.0, 4.0, 0.0, 1.5),
                    (3.0, 2.0, 1.5, 0.0),
                ),
                (
                    (0.0, 2.0, 3.0, 4.0),
                    (2.0, 0.0, 5.0, 3.0),
                    (3.0, 5.0, 0.0, 2.5),
                    (4.0, 3.0, 2.5, 0.0),
                ),
            )
        )
        result = AnnealedParetoSMCOptimizer(
            instance,
            particles_per_reference=2,
            evaluations=4,
            seed=913,
            beta_schedule=(0.0, 1.0),
            reference_directions=((0.5, 0.5),),
        ).run()
        self.assertEqual(
            result.metadata["objective_lower_bounds"],
            (
                math.nextafter(4.0, -math.inf),
                math.nextafter(8.0, -math.inf),
            ),
        )
        self.assertEqual(
            result.metadata["objective_upper_bounds"],
            (
                math.nextafter(16.0, math.inf),
                math.nextafter(20.0, math.inf),
            ),
        )
        for value in result.metadata["epsilon"]:
            self.assertAlmostEqual(value, 0.6, places=15)
        self.assertEqual(
            result.metadata["analytic_box_formula"],
            (
                "outward_nextafter_of_n_times_min_and_max_offdiagonal_edge_"
                "per_objective_v2"
            ),
        )

    def test_beta_path_and_minimum_mutation_budget_fail_closed_before_sampling(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=914)
        common = {
            "particles_per_reference": 2,
            "reference_directions": ((0.5, 0.5),),
        }
        with self.assertRaisesRegex(ValueError, "start at beta_0"):
            AnnealedParetoSMCOptimizer(
                instance,
                evaluations=4,
                beta_schedule=(0.1, 1.0),
                **common,
            )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            AnnealedParetoSMCOptimizer(
                instance,
                evaluations=6,
                beta_schedule=(0.0, 1.0, 1.0),
                **common,
            )
        with self.assertRaisesRegex(ValueError, "minimum 6"):
            AnnealedParetoSMCOptimizer(
                instance,
                evaluations=5,
                beta_schedule=(0.0, 0.5, 1.0),
                **common,
            )

        counted = CountingTSPInstance(instance, max_evaluations=5)
        with self.assertRaisesRegex(ValueError, "smaller than the requested"):
            AnnealedParetoSMCOptimizer(
                counted,  # type: ignore[arg-type]
                evaluations=6,
                beta_schedule=(0.0, 0.5, 1.0),
                **common,
            )

        result = AnnealedParetoSMCOptimizer(
            instance,
            evaluations=6,
            beta_schedule=(0.0, 0.5, 1.0),
            **common,
        ).run()
        self.assertEqual(result.metadata["minimum_evaluation_budget"], 6)
        self.assertEqual(
            result.metadata["stage_mutation_budgets"],
            (2, 2),
        )

    def test_augmented_tchebycheff_rho_must_be_strictly_positive(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=920)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            AnnealedParetoSMCOptimizer(
                instance,
                particles_per_reference=2,
                evaluations=4,
                beta_schedule=(0.0, 1.0),
                reference_directions=((0.5, 0.5),),
                chebyshev_rho=0.0,
            )
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            AnnealedParetoSMCOptimizer(
                instance,
                particles_per_reference=2,
                evaluations=4,
                beta_schedule=(0.0, 1.0),
                reference_directions=((1.0, 0.0),),
            )
        result = AnnealedParetoSMCOptimizer(
            instance,
            particles_per_reference=2,
            evaluations=4,
            beta_schedule=(0.0, 1.0),
            reference_directions=((0.5, 0.5),),
        ).run()
        self.assertEqual(
            result.metadata["pareto_monotonicity_scope"],
            "strict_on_objective_vectors_under_componentwise_dominance",
        )

    def test_target_energy_is_continuous_in_objectives_and_not_tied_within_epsilon_cell(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=924)
        optimizer = AnnealedParetoSMCOptimizer(
            instance,
            particles_per_reference=2,
            evaluations=4,
            seed=925,
            beta_schedule=(0.0, 1.0),
            reference_directions=((0.5, 0.5),),
            objective_lower_bounds=(0.0, 0.0),
            objective_upper_bounds=(100.0, 100.0),
            epsilon=(10.0, 10.0),
        )
        dominating = (1.0, 1.0)
        dominated_same_cell = (1.1, 1.1)
        self.assertLess(
            optimizer.target_energy(dominating, reference_index=0),
            optimizer.target_energy(dominated_same_cell, reference_index=0),
        )
        result = optimizer.run()
        self.assertTrue(result.metadata["target_independent_of_epsilon_cells"])
        self.assertEqual(
            result.metadata["epsilon_cells_role"],
            "external_reporting_coverage_observer_no_target_feedback",
        )

    def test_predeclared_target_and_reporting_grid_mutation_fail_closed(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=926)
        target_mutated = AnnealedParetoSMCOptimizer(
            instance,
            particles_per_reference=2,
            evaluations=4,
            beta_schedule=(0.0, 1.0),
            reference_directions=((0.5, 0.5),),
        )
        target_mutated.chebyshev_rho = 0.04
        with self.assertRaisesRegex(RuntimeError, "target context changed"):
            target_mutated.run()

        grid_mutated = AnnealedParetoSMCOptimizer(
            instance,
            particles_per_reference=2,
            evaluations=4,
            beta_schedule=(0.0, 1.0),
            reference_directions=((0.5, 0.5),),
        )
        grid_mutated.epsilon = tuple(2.0 * value for value in grid_mutated.epsilon)
        with self.assertRaisesRegex(RuntimeError, "reporting context changed"):
            grid_mutated.run()

    def test_maximum_ess_does_not_claim_epsilon_cell_coverage(self) -> None:
        def constant_matrix(value: float):
            return tuple(
                tuple(0.0 if i == j else value for j in range(4))
                for i in range(4)
            )

        instance = MultiObjectiveTSPInstance.from_distance_matrices(
            (constant_matrix(1.0), constant_matrix(2.0))
        )
        result = AnnealedParetoSMCOptimizer(
            instance,
            particles_per_reference=4,
            evaluations=8,
            seed=921,
            beta_schedule=(0.0, 1.0),
            reference_directions=((0.5, 0.5),),
            ess_threshold=1.0,
        ).run()
        reference = result.metadata["stage_ledger"][1]["references"][0]
        self.assertEqual(reference["ess_before_resampling"], 4.0)
        self.assertFalse(reference["resampled"])
        self.assertEqual(reference["occupied_epsilon_cell_count_before_weighting"], 1)
        self.assertEqual(result.metadata["queried_epsilon_cell_count"], 1)
        self.assertTrue(result.metadata["ess_is_not_coverage_certificate"])
        self.assertEqual(
            result.metadata["ess_resampling_rule"],
            "resample_iff_ess_strictly_below_threshold_times_type_size",
        )

    def test_context_run_and_stage_hashes_are_stable_and_metadata_is_strict_json(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=915)
        kwargs = {
            "particles_per_reference": 2,
            "evaluations": 12,
            "beta_schedule": (0.0, 0.5, 1.0),
            "reference_directions": ((0.6, 0.4), (0.4, 0.6)),
            "epsilon": (0.1, 0.1),
        }
        first = AnnealedParetoSMCOptimizer(
            instance,
            seed=916,
            **kwargs,
        ).run()
        replay = AnnealedParetoSMCOptimizer(
            instance,
            seed=916,
            **kwargs,
        ).run()
        other_seed = AnnealedParetoSMCOptimizer(
            instance,
            seed=917,
            **kwargs,
        ).run()
        other_epsilon = AnnealedParetoSMCOptimizer(
            instance,
            particles_per_reference=2,
            evaluations=12,
            seed=916,
            beta_schedule=(0.0, 0.5, 1.0),
            reference_directions=((0.6, 0.4), (0.4, 0.6)),
            epsilon=(0.2, 0.1),
        ).run()

        self.assertEqual(first.metadata["context_hash"], replay.metadata["context_hash"])
        self.assertEqual(first.metadata["run_contract_hash"], replay.metadata["run_contract_hash"])
        self.assertEqual(first.metadata["stage_ledger_hash"], replay.metadata["stage_ledger_hash"])
        self.assertEqual(first.metadata["stage_ledger"], replay.metadata["stage_ledger"])
        self.assertEqual(first.metadata["context_hash"], other_seed.metadata["context_hash"])
        self.assertNotEqual(first.metadata["run_contract_hash"], other_seed.metadata["run_contract_hash"])
        self.assertEqual(first.metadata["context_hash"], other_epsilon.metadata["context_hash"])
        self.assertEqual(
            first.metadata["run_contract_hash"],
            other_epsilon.metadata["run_contract_hash"],
        )
        self.assertNotEqual(
            first.metadata["reporting_context_hash"],
            other_epsilon.metadata["reporting_context_hash"],
        )
        json.dumps(first.metadata, sort_keys=True, allow_nan=False)

    def test_incremental_weights_accumulate_when_ess_does_not_trigger_resampling(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=918)
        result = AnnealedParetoSMCOptimizer(
            instance,
            particles_per_reference=4,
            evaluations=20,
            seed=919,
            beta_schedule=(0.0, 0.4, 1.0),
            reference_directions=((0.5, 0.5),),
            epsilon=(0.01, 0.01),
            ess_threshold=1e-6,
        ).run()
        stages = result.metadata["stage_ledger"]
        first = stages[1]["references"][0]
        second = stages[2]["references"][0]
        self.assertFalse(first["resampled"])
        self.assertFalse(second["resampled"])
        self.assertEqual(
            tuple(second["normalized_weights_before_increment"]),
            tuple(first["normalized_weights_after_resampling"]),
        )
        raw = [
            math.log(old_weight) + incremental
            for old_weight, incremental in zip(
                second["normalized_weights_before_increment"],
                second["incremental_log_weights"],
            )
        ]
        maximum = max(raw)
        normalizer = maximum + math.log(
            sum(math.exp(value - maximum) for value in raw)
        )
        expected = tuple(math.exp(value - normalizer) for value in raw)
        for actual, target in zip(
            second["normalized_weights_before_resampling"],
            expected,
        ):
            self.assertAlmostEqual(actual, target, places=15)
        expected_log_z = sum(
            stage["references"][0]["log_normalizer_increment"]
            for stage in stages
        )
        self.assertAlmostEqual(
            result.metadata["final_log_normalizer_estimates_by_reference"][0],
            expected_log_z,
            places=15,
        )
        for reference_masses in result.metadata[
            "final_epsilon_cell_masses_by_reference"
        ]:
            self.assertAlmostEqual(
                sum(record["mass"] for record in reference_masses),
                1.0,
                places=15,
            )
            self.assertTrue(
                all(record["mass"] > 0.0 for record in reference_masses)
            )
        self.assertEqual(
            result.metadata["finite_particle_coverage_observable"],
            "positive_terminal_weight_mass_per_predeclared_epsilon_cell",
        )

    @staticmethod
    def _cell_from_metadata(objective, metadata):  # type: ignore[no-untyped-def]
        cells = []
        for value, lower, width, count in zip(
            objective,
            metadata["objective_lower_bounds"],
            metadata["epsilon"],
            metadata["epsilon_cell_counts"],
        ):
            raw = math.floor(max(0.0, value - lower) / width)
            cells.append(min(count - 1, raw))
        return tuple(cells)


if __name__ == "__main__":
    unittest.main()

