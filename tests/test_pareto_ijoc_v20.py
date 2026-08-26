from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from mo_nco.archive import ArchiveEntry, ParetoArchive, dominates
from mo_nco.benchmark import resolve_predeclared_algorithm_configuration, run_algorithm
from mo_nco.instance import MultiObjectiveTSPInstance
from mo_nco.pareto_ijoc_allocation import (
    Exp3TypeAllocator,
    SearchRewardWeights,
    derive_domain_separated_seed,
)
from mo_nco.pareto_ijoc_generic_search import GenericTypedArchiveSearch
from mo_nco.pareto_ijoc_generic_smc import GenericAnnealedParetoSMCOptimizer
from mo_nco.pareto_ijoc_preflight import audit_ijoc_competitive_study
from mo_nco.pareto_ijoc_problem import (
    MultiObjectiveKnapsackInstance,
    MultiObjectiveTSPProblemAdapter,
)
from mo_nco.pareto_ijoc_spec import load_ijoc_pareto_smc_specification
from mo_nco.pareto_smc import AnnealedParetoSMCOptimizer


def _instance(seed: int = 0, cities: int = 7) -> MultiObjectiveTSPInstance:
    rng = random.Random(seed)
    matrices = []
    for _ in range(2):
        matrix = [[0.0] * cities for _ in range(cities)]
        for i in range(cities):
            for j in range(i + 1, cities):
                value = float(rng.randint(1, 100))
                matrix[i][j] = value
                matrix[j][i] = value
        matrices.append(tuple(tuple(row) for row in matrix))
    return MultiObjectiveTSPInstance.from_distance_matrices(tuple(matrices))


def _base_spec(path: Path) -> Path:
    payload = {
        "schema": "annealed_pareto_smc_spec_v1",
        "objective_box": {
            "source": "analytic_distance_matrix_box",
            "archive_independent": True,
        },
        "epsilon_cells": {
            "coordinate_system": "normalized_frozen_objective_box",
            "widths": [0.1, 0.1],
            "archive_independent": True,
            "role": "reporting_and_coverage_only",
        },
        "reference_directions": [[0.75, 0.25], [0.25, 0.75]],
        "target": {
            "family": "typed_augmented_tchebycheff_gibbs",
            "stage_frozen": True,
            "beta_schedule": [0.0, 0.5, 1.0],
            "chebyshev_rho": 0.03,
        },
        "resampling": {
            "method": "multinomial",
            "scope": "within_reference_type",
            "ess_threshold_fraction": 0.5,
            "ess_is_not_a_coverage_certificate": True,
        },
        "mutation": {
            "proposal": "uniform_symmetric_two_opt",
            "acceptance": "exact_log_domain_mh",
            "objective_evaluation": "exact_incremental_two_opt_on_verified_integer_domain_else_full_tour_v1",
        },
        "particle_allocation": {
            "policy": "split_cli_population_equally_across_reference_types"
        },
        "reporting": {
            "archive_role": "reporting_only",
            "archive_max_size": 20,
            "cell_ledger": "untruncated_first_evaluated_representative_per_cell",
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _ijoc_spec(path: Path, base: Path, *, tail: int = 6, policy: str = "exp3") -> Path:
    payload = {
        "schema": "ijoc_typed_pareto_smc_spec_v2",
        "base_smc": {
            "path": base.name,
            "sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        },
        "adaptive_search": {
            "evaluations": tail,
            "allocation_policy": policy,
            "minimum_pulls_per_type": 1 if policy == "exp3" else 0,
            "exp3_exploration": 0.4 if policy == "exp3" else None,
            "reward_weights": {
                "hypervolume": 0.75,
                "new_cell": 0.20,
                "scalar_improvement": 0.05,
            },
        },
        "output": {
            "competitive_archive": "unbounded_all_evaluated_nondominated",
            "deployment_archive_max_size": 5,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


class IJOCV20Tests(unittest.TestCase):
    def test_zero_tolerance_is_not_interchangeable_for_neighboring_floats(
        self,
    ) -> None:
        entries = (
            ArchiveEntry((0,), (1.0, 1.0)),
            ArchiveEntry((1,), (1.0 + 5e-13, 1.0 - 5e-13)),
        )
        exact = ParetoArchive(tol=0.0)
        tolerant = ParetoArchive(tol=1e-12)
        for entry in entries:
            exact.update((entry,))
            tolerant.update((entry,))
        self.assertEqual(exact.entries, entries)
        self.assertEqual(tolerant.entries, entries[:1])

    def test_integer_2d_updates_match_default_tolerance_after_every_step(
        self,
    ) -> None:
        rng = random.Random(20260731)
        for _ in range(40):
            exact = ParetoArchive(tol=0.0)
            tolerant = ParetoArchive(tol=1e-12)
            for _step in range(100):
                entry = ArchiveEntry(
                    (rng.randrange(80),),
                    (
                        float(rng.randint(-1_000_000, 1_000_000)),
                        float(rng.randint(-1_000_000, 1_000_000)),
                    ),
                )
                exact.update((entry,))
                tolerant.update((entry,))
                self.assertEqual(exact.entries, tolerant.entries)

    def test_zero_tolerance_single_2d_updates_use_incremental_fast_path(
        self,
    ) -> None:
        archive = ParetoArchive(tol=0.0)
        entries = (
            ArchiveEntry((0,), (1.0, 4.0)),
            ArchiveEntry((1,), (2.0, 3.0)),
            ArchiveEntry((2,), (3.0, 2.0)),
        )
        with patch.object(
            archive,
            "_update_2d_exact_single",
            wraps=archive._update_2d_exact_single,
        ) as incremental:
            for entry in entries:
                archive.update((entry,))
        self.assertEqual(incremental.call_count, len(entries))
        self.assertEqual(archive.entries, entries)

    def test_tolerant_2d_archive_matches_bruteforce_dominance(self) -> None:
        # Regression counterexample for the old one-direction sweep: the second
        # point dominates the first because its x coordinate is within tol.
        archive = ParetoArchive(tol=1e-3)
        archive.update(
            (
                ArchiveEntry((0,), (0.0, 1.0)),
                ArchiveEntry((1,), (5e-4, 0.0)),
            )
        )
        self.assertEqual(archive.objectives(), ((5e-4, 0.0),))

        duplicate_archive = ParetoArchive(tol=1e-3)
        duplicate_archive.update(
            (
                ArchiveEntry((0,), (0.0, 0.0)),
                ArchiveEntry((1,), (5e-4, 5e-4)),
            )
        )
        self.assertEqual(duplicate_archive.objectives(), ((0.0, 0.0),))

        rng = random.Random(991)
        for _ in range(300):
            entries = [
                ArchiveEntry((index,), (rng.uniform(-2, 2), rng.uniform(-2, 2)))
                for index in range(30)
            ]
            fast = ParetoArchive(tol=1e-3)
            fast.update(entries)
            expected = tuple(
                sorted(
                    entry.objectives
                    for entry in entries
                    if not any(
                        other.tour != entry.tour
                        and dominates(other.objectives, entry.objectives, 1e-3)
                        for other in entries
                    )
                )
            )
            self.assertEqual(fast.objectives(), expected)

    def test_archive_exact_membership_uses_tour_objective_pairs(self) -> None:
        archive = ParetoArchive(tol=0.0)
        left = ArchiveEntry((0,), (0.0, 2.0))
        right = ArchiveEntry((1,), (2.0, 0.0))
        archive.update((left, right))
        self.assertTrue(archive.contains(left))
        self.assertTrue(archive.contains(right))
        self.assertFalse(archive.contains(ArchiveEntry((0,), (2.0, 0.0))))

    def test_exact_2d_single_insert_matches_bruteforce_after_every_step(self) -> None:
        rng = random.Random(20260730)
        for _ in range(40):
            archive = ParetoArchive(tol=0.0)
            observed = []
            for index in range(60):
                entry = ArchiveEntry(
                    (index,),
                    (float(rng.randint(-20, 20)), float(rng.randint(-20, 20))),
                )
                observed.append(entry)
                archive.update((entry,))
                unique = []
                seen_objectives = set()
                for candidate in observed:
                    if candidate.objectives in seen_objectives:
                        continue
                    seen_objectives.add(candidate.objectives)
                    unique.append(candidate)
                expected = tuple(
                    sorted(
                        candidate.objectives
                        for candidate in unique
                        if not any(
                            other.tour != candidate.tour
                            and dominates(
                                other.objectives, candidate.objectives, 0.0
                            )
                            for other in unique
                        )
                    )
                )
                self.assertEqual(archive.objectives(), expected)

    def test_exp3_floor_normalization_and_update_domain(self) -> None:
        allocator = Exp3TypeAllocator(7, exploration=0.35)
        rng = random.Random(4)
        floor = 0.35 / 7
        for _ in range(1000):
            probabilities = allocator.probabilities()
            self.assertLessEqual(abs(math.fsum(probabilities) - 1.0), 4 * math.ulp(1.0))
            self.assertTrue(all(value >= floor for value in probabilities))
            arm, probability = allocator.select(rng)
            allocator.observe(arm, rng.random(), probability)
        snapshot = allocator.snapshot()
        self.assertEqual(snapshot.rounds, 1000)
        self.assertEqual(sum(snapshot.pulls), 1000)
        pending = Exp3TypeAllocator(3, exploration=0.3)
        selected, probability = pending.select(random.Random(9))
        with self.assertRaises(RuntimeError):
            pending.select(random.Random(10))
        with self.assertRaises(RuntimeError):
            pending.observe((selected + 1) % 3, 0.5, probability)

    def test_domain_separated_adaptive_rng_roles_are_reproducible_and_distinct(self) -> None:
        left = derive_domain_separated_seed(17, context="abc", domain="selection")
        self.assertEqual(
            left,
            derive_domain_separated_seed(17, context="abc", domain="selection"),
        )
        self.assertNotEqual(
            left,
            derive_domain_separated_seed(17, context="abc", domain="environment"),
        )

    def test_reward_is_quality_aligned_and_bounded(self) -> None:
        weights = SearchRewardWeights(0.7, 0.2, 0.1)
        reward = weights.combine(
            normalized_hypervolume_gain=0.5,
            new_cell=True,
            normalized_scalar_improvement=0.25,
        )
        self.assertAlmostEqual(reward, 0.575)
        self.assertEqual(
            weights.combine(
                normalized_hypervolume_gain=100.0,
                new_cell=True,
                normalized_scalar_improvement=100.0,
            ),
            1.0,
        )

    def test_adaptive_tail_preserves_exact_budget_and_pre_tail_snapshot(self) -> None:
        result = AnnealedParetoSMCOptimizer(
            _instance(10),
            particles_per_reference=2,
            evaluations=22,
            seed=11,
            beta_schedule=(0.0, 0.5, 1.0),
            reference_directions=((0.75, 0.25), (0.25, 0.75)),
            epsilon=(20.0, 20.0),
            adaptive_search_evaluations=6,
            exp3_exploration=0.4,
            archive_tolerance=0.0,
            archive_max_size=2,
            audit_trace_level="summary",
        ).run()
        metadata = result.metadata
        self.assertEqual(metadata["evaluations_used"], 22)
        self.assertEqual(metadata["smc_core_evaluation_budget"], 16)
        self.assertEqual(metadata["adaptive_search_tail_attempts"], 6)
        self.assertEqual(
            metadata["certificate_snapshot_before_adaptive_tail"]["evaluation_end"],
            16,
        )
        self.assertTrue(
            metadata["certificate_snapshot_before_adaptive_tail"]["trace_compacted"]
        )
        self.assertTrue(metadata["certificate_snapshot_excludes_adaptive_search_tail"])
        self.assertTrue(
            metadata["certificate_cell_representatives_exclude_adaptive_tail"]
        )
        self.assertGreaterEqual(
            metadata["all_evaluated_epsilon_cell_count"],
            len(metadata["certificate_cell_representatives_before_adaptive_tail"]),
        )
        self.assertEqual(
            metadata["competitive_search_archive_dominance_tolerance"],
            0.0,
        )
        self.assertIsNone(result.archive.max_size)
        self.assertLessEqual(metadata["deployment_archive_size"], 2)
        self.assertEqual(metadata["adaptive_search_allocator"]["rounds"], 6)
        self.assertEqual(
            metadata["adaptive_search_counterfactual_reward_contract"],
            "private_per_type_random_tapes_sampled_before_arm_selection_v1",
        )
        self.assertEqual(
            metadata["adaptive_search_rng_domain_separation_contract"],
            "independent_mutable_rng_states_for_selection_and_environment_v1",
        )
        self.assertNotEqual(
            metadata["adaptive_search_selection_seed_sha256"],
            metadata["adaptive_search_environment_seed_sha256"],
        )

    def test_uniform_tail_has_no_exp3_claim(self) -> None:
        result = AnnealedParetoSMCOptimizer(
            _instance(12),
            particles_per_reference=2,
            evaluations=18,
            seed=13,
            beta_schedule=(0.0, 1.0),
            reference_directions=((0.75, 0.25), (0.25, 0.75)),
            epsilon=(20.0, 20.0),
            adaptive_search_evaluations=4,
            adaptive_allocation_policy="uniform",
            archive_max_size=None,
            audit_trace_level="summary",
        ).run()
        self.assertIsNone(result.metadata["adaptive_search_allocator"])
        self.assertEqual(result.metadata["adaptive_search_tail_attempts"], 4)

    def test_adaptive_tail_cannot_be_mixed_with_bootstrap_certificate(self) -> None:
        with self.assertRaises(ValueError):
            AnnealedParetoSMCOptimizer(
                _instance(15),
                particles_per_reference=2,
                evaluations=20,
                beta_schedule=(0.0, 1.0),
                reference_directions=((0.75, 0.25), (0.25, 0.75)),
                epsilon=(20.0, 20.0),
                resampling_policy="always",
                adaptive_search_evaluations=4,
            )

    def test_ijoc_spec_is_sha_bound_and_alias_consumes_exact_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = _base_spec(root / "base.json")
            spec = _ijoc_spec(root / "ijoc.json", base, tail=6)
            loaded = load_ijoc_pareto_smc_specification(
                spec,
                objective_dimension=2,
                total_evaluations=22,
            )
            self.assertEqual(loaded.adaptive_search_evaluations, 6)
            self.assertEqual(loaded.minimum_pulls_per_type, 1)
            with self.assertRaises(ValueError):
                load_ijoc_pareto_smc_specification(
                    spec,
                    objective_dimension=3,
                    total_evaluations=22,
                )
            env = {"MO_NCO_IJOC_PARETO_SMC_SPEC": str(spec)}
            with patch.dict(os.environ, env, clear=False):
                frozen = resolve_predeclared_algorithm_configuration(
                    case_name="smoke",
                    instance=_instance(20),
                    algorithm="ijoc-pareto-smc",
                    seed=0,
                    population=4,
                    iterations=22,
                    log_period=2,
                    archive_update_period=2,
                    output_archive_limit=10,
                    certified_traces=False,
                    anytime_checkpoint_period=11,
                )
                self.assertEqual(
                    frozen.payload["algorithm_specific"]["adaptive_search_evaluations"],
                    6,
                )
                self.assertEqual(
                    frozen.payload["algorithm_specific"][
                        "adaptive_minimum_pulls_per_type"
                    ],
                    1,
                )
                result = run_algorithm(
                    "ijoc-pareto-smc",
                    _instance(20),
                    seed=0,
                    population=4,
                    iterations=22,
                    log_period=2,
                    archive_update_period=2,
                    anytime_checkpoint_period=11,
                )
            self.assertEqual(result.metadata["evaluations_used"], 22)
            self.assertEqual(
                result.metadata["adaptive_search_uniform_prefix_evaluations"],
                2,
            )
            self.assertEqual(
                result.metadata["adaptive_search_allocator"]["rounds"],
                4,
            )
            self.assertEqual(result.metadata["ijoc_specification_sha256"], loaded.sha256)
            base.write_text(base.read_text() + " ", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_ijoc_pareto_smc_specification(spec, objective_dimension=2)

    def test_ijoc_spec_rejects_absolute_and_escaping_base_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inside = root / "inside"
            inside.mkdir()
            base = _base_spec(root / "base.json")
            digest = hashlib.sha256(base.read_bytes()).hexdigest()
            payload = {
                "schema": "ijoc_typed_pareto_smc_spec_v2",
                "base_smc": {"path": str(base), "sha256": digest},
                "adaptive_search": {
                    "evaluations": 4,
                    "allocation_policy": "uniform",
                    "minimum_pulls_per_type": 0,
                    "exp3_exploration": None,
                    "reward_weights": {
                        "hypervolume": 0.75,
                        "new_cell": 0.20,
                        "scalar_improvement": 0.05,
                    },
                },
                "output": {
                    "competitive_archive": "unbounded_all_evaluated_nondominated",
                    "deployment_archive_max_size": 5,
                },
            }
            spec = inside / "ijoc.json"
            spec.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be relative"):
                load_ijoc_pareto_smc_specification(spec, objective_dimension=2)
            payload["base_smc"]["path"] = "../base.json"
            spec.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_ijoc_pareto_smc_specification(spec, objective_dimension=2)

    def test_ijoc_anytime_grid_must_divide_budget_and_is_complete(self) -> None:
        instance = _instance(41)
        result = AnnealedParetoSMCOptimizer(
            instance,
            particles_per_reference=2,
            evaluations=22,
            seed=42,
            beta_schedule=(0.0, 0.5, 1.0),
            reference_directions=((0.75, 0.25), (0.25, 0.75)),
            epsilon=(20.0, 20.0),
            adaptive_search_evaluations=6,
            exp3_exploration=0.4,
            anytime_checkpoint_period=11,
            audit_trace_level="summary",
        ).run()
        self.assertEqual(
            result.metadata["expected_anytime_checkpoints"],
            (11, 22),
        )
        self.assertEqual(
            result.metadata["observed_anytime_checkpoints"],
            (11, 22),
        )
        self.assertTrue(result.metadata["anytime_checkpoint_grid_complete"])
        self.assertEqual(
            tuple(
                item["evaluation"]
                for item in result.metadata["checkpoint_solution_witnesses"]
            ),
            (11, 22),
        )
        for checkpoint in result.metadata["checkpoint_solution_witnesses"]:
            for entry in checkpoint["entries"]:
                self.assertEqual(
                    tuple(entry["objectives"]),
                    instance.evaluate(tuple(entry["tour"])),
                )
        with self.assertRaises(RuntimeError):
            AnnealedParetoSMCOptimizer(
                _instance(43),
                particles_per_reference=2,
                evaluations=22,
                seed=44,
                beta_schedule=(0.0, 0.5, 1.0),
                reference_directions=((0.75, 0.25), (0.25, 0.75)),
                epsilon=(20.0, 20.0),
                adaptive_search_evaluations=6,
                exp3_exploration=0.4,
                anytime_checkpoint_period=10,
                audit_trace_level="summary",
            ).run()

    def test_generic_annealed_smc_is_single_use_and_exact_archive(self) -> None:
        optimizer = GenericAnnealedParetoSMCOptimizer(
            MultiObjectiveKnapsackInstance.random_instance(8, seed=45),
            reference_directions=((0.75, 0.25), (0.25, 0.75)),
            particles_per_reference=2,
            evaluations=24,
            beta_schedule=(0.0, 0.5, 1.0),
            adaptive_search_evaluations=4,
            exp3_exploration=0.4,
            seed=46,
        )
        result = optimizer.run()
        self.assertEqual(
            result.metadata["competitive_search_archive_dominance_tolerance"],
            0.0,
        )
        with self.assertRaises(RuntimeError):
            optimizer.run()

    def test_generic_annealed_smc_supports_true_uniform_tail(self) -> None:
        problem = MultiObjectiveKnapsackInstance.random_instance(8, seed=245)
        result = GenericAnnealedParetoSMCOptimizer(
            problem,
            reference_directions=((0.75, 0.25), (0.25, 0.75)),
            particles_per_reference=2,
            evaluations=24,
            beta_schedule=(0.0, 0.5, 1.0),
            adaptive_search_evaluations=4,
            adaptive_allocation_policy="uniform",
            minimum_pulls_per_type=0,
            exp3_exploration=None,
            seed=246,
        ).run()
        self.assertEqual(result.metadata["adaptive_allocation_policy"], "uniform")
        self.assertIsNone(result.metadata["allocator"])
        with self.assertRaisesRegex(ValueError, "exp3_exploration"):
            GenericAnnealedParetoSMCOptimizer(
                problem,
                reference_directions=((0.75, 0.25), (0.25, 0.75)),
                particles_per_reference=2,
                evaluations=24,
                beta_schedule=(0.0, 0.5, 1.0),
                adaptive_search_evaluations=4,
                adaptive_allocation_policy="uniform",
                exp3_exploration=0.25,
                seed=247,
            )

    def test_generic_smc_emits_exact_checkpoint_solution_witnesses(self) -> None:
        problem = MultiObjectiveKnapsackInstance.random_instance(8, seed=145)
        result = GenericAnnealedParetoSMCOptimizer(
            problem,
            reference_directions=((0.75, 0.25), (0.25, 0.75)),
            particles_per_reference=2,
            evaluations=24,
            beta_schedule=(0.0, 0.5, 1.0),
            adaptive_search_evaluations=4,
            exp3_exploration=0.4,
            anytime_checkpoint_period=6,
            seed=146,
        ).run()
        self.assertEqual(
            result.metadata["expected_anytime_checkpoints"],
            (6, 12, 18, 24),
        )
        self.assertEqual(
            result.metadata["observed_anytime_checkpoints"],
            (6, 12, 18, 24),
        )
        self.assertEqual(
            tuple(
                item["evaluation"]
                for item in result.metadata["checkpoint_solution_witnesses"]
            ),
            (6, 12, 18, 24),
        )
        for checkpoint in result.metadata["checkpoint_solution_witnesses"]:
            for entry in checkpoint["entries"]:
                self.assertEqual(
                    tuple(entry["objectives"]),
                    problem.evaluate(tuple(entry["solution"])),
                )
        with self.assertRaisesRegex(ValueError, "positive divisor"):
            GenericAnnealedParetoSMCOptimizer(
                problem,
                reference_directions=((0.75, 0.25), (0.25, 0.75)),
                particles_per_reference=2,
                evaluations=24,
                beta_schedule=(0.0, 0.5, 1.0),
                adaptive_search_evaluations=4,
                anytime_checkpoint_period=5,
                seed=147,
            )

    def test_knapsack_proposal_is_symmetric_on_distinct_feasible_neighbors(self) -> None:
        problem = MultiObjectiveKnapsackInstance(
            item_weights=(1, 2, 3, 2, 1),
            profits_by_objective=((4, 3, 7, 1, 5), (1, 5, 2, 8, 3)),
            capacity=5,
        )
        feasible = []
        for mask in range(1 << problem.solution_size):
            solution = tuple((mask >> item) & 1 for item in range(problem.solution_size))
            try:
                problem.validate_solution(solution)
            except ValueError:
                continue
            feasible.append(solution)
        for left in feasible:
            for right in feasible:
                distance = sum(a != b for a, b in zip(left, right))
                if distance == 1:
                    self.assertEqual(
                        problem.proposal_probability(left, right),
                        problem.proposal_probability(right, left),
                    )

    def test_knapsack_initializer_is_exact_uniform_over_feasible_solutions(self) -> None:
        problem = MultiObjectiveKnapsackInstance(
            item_weights=(1, 2, 3, 2, 1),
            profits_by_objective=((4, 3, 7, 1, 5), (1, 5, 2, 8, 3)),
            capacity=5,
        )
        feasible = []
        for mask in range(1 << problem.solution_size):
            solution = tuple(
                (mask >> item) & 1 for item in range(problem.solution_size)
            )
            try:
                problem.validate_solution(solution)
            except ValueError:
                continue
            feasible.append(solution)
        self.assertEqual(problem.feasible_solution_count, len(feasible))
        probabilities = {
            problem.uniform_solution_probability(solution)
            for solution in feasible
        }
        self.assertEqual(len(probabilities), 1)
        self.assertEqual(
            sum(
                problem.uniform_solution_probability(solution)
                for solution in feasible
            ),
            1,
        )
        rng = random.Random(20260731)
        sampled = [problem.random_solution(rng) for _ in range(200)]
        self.assertTrue(set(sampled).issubset(set(feasible)))
        self.assertGreater(len(set(sampled)), 1)

    def test_generic_annealed_smc_runs_same_skeleton_on_tsp_and_knapsack(self) -> None:
        tsp_problem = MultiObjectiveTSPProblemAdapter(_instance(31))
        knapsack_problem = MultiObjectiveKnapsackInstance.random_instance(10, seed=32)
        for problem in (tsp_problem, knapsack_problem):
            result = GenericAnnealedParetoSMCOptimizer(
                problem,
                reference_directions=((0.75, 0.25), (0.25, 0.75)),
                particles_per_reference=2,
                evaluations=30,
                beta_schedule=(0.0, 0.5, 1.0),
                adaptive_search_evaluations=6,
                exp3_exploration=0.4,
                deployment_archive_max_size=3,
                seed=33,
            ).run()
            self.assertEqual(result.metadata["evaluations_used"], 30)
            self.assertEqual(result.metadata["core_evaluation_budget"], 24)
            self.assertEqual(result.metadata["allocator"]["rounds"], 6)
            self.assertIsNone(result.archive.max_size)
            self.assertLessEqual(result.metadata["deployment_archive_size"], 3)

    def test_generic_knapsack_search_uses_exact_budget_and_archive_split(self) -> None:
        problem = MultiObjectiveKnapsackInstance.random_instance(12, seed=21)
        result = GenericTypedArchiveSearch(
            problem,
            reference_directions=((0.75, 0.25), (0.25, 0.75)),
            population_per_type=2,
            evaluations=40,
            seed=22,
            deployment_archive_max_size=3,
        ).run()
        self.assertEqual(result.metadata["evaluations_used"], 40)
        self.assertEqual(result.metadata["allocator"]["rounds"], 36)
        self.assertIsNone(result.archive.max_size)
        self.assertEqual(
            result.metadata["competitive_search_archive_dominance_tolerance"],
            0.0,
        )
        self.assertEqual(
            result.metadata["deployment_archive_dominance_tolerance"],
            0.0,
        )
        self.assertLessEqual(result.metadata["deployment_archive_size"], 3)
        self.assertGreater(result.metadata["observed_cell_count"], 0)
        optimizer = GenericTypedArchiveSearch(
            problem,
            reference_directions=((0.75, 0.25), (0.25, 0.75)),
            population_per_type=2,
            evaluations=8,
            seed=23,
        )
        optimizer.run()
        with self.assertRaises(RuntimeError):
            optimizer.run()

    def test_generic_search_rejects_non_symmetric_proposal_transition(self) -> None:
        class AsymmetricProblem:
            name = "asymmetric-test"
            num_objectives = 2
            solution_size = 1
            objective_lower_bounds = (0.0, 0.0)
            objective_upper_bounds = (2.0, 2.0)
            symmetric_proposal_contract = "false_test_contract"

            def random_solution(self, rng):
                return (0,)

            def propose(self, solution, rng):
                return (1,)

            def proposal_probability(self, source, target):
                return 1.0 if source == (0,) and target == (1,) else 0.5

            def evaluate(self, solution):
                return (float(solution[0]), float(1 - solution[0]))

            def validate_solution(self, solution):
                if solution not in {(0,), (1,)}:
                    raise ValueError("invalid")

            def canonical_payload(self):
                return {"family": "asymmetric-test"}

        optimizer = GenericTypedArchiveSearch(
            AsymmetricProblem(),
            reference_directions=((0.5, 0.5),),
            population_per_type=1,
            evaluations=2,
            seed=77,
        )
        with self.assertRaises(RuntimeError):
            optimizer.run()

    def test_ijoc_study_preflight_requires_exact_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            motsp_cases = [f"tsp-{index:02d}" for index in range(15)]
            mokp_cases = [f"kp-{index:02d}" for index in range(15)]
            families = [
                {
                    "id": "motsp",
                    "cases": motsp_cases,
                    "algorithms": [
                        "ijoc-pareto-smc",
                        "paquete",
                        "tpls",
                        "pymoo-moead",
                    ],
                    "required_baselines": ["paquete", "tpls", "pymoo-moead"],
                },
                {
                    "id": "mokp",
                    "cases": mokp_cases,
                    "algorithms": [
                        "ijoc-pareto-smc",
                        "pymoo-nsga2",
                        "pymoo-moead",
                        "mokp-pls",
                    ],
                    "required_baselines": ["pymoo-nsga2", "pymoo-moead", "mokp-pls"],
                },
            ]
            seeds = list(range(10))
            budgets = [10, 20, 30]
            reference_points = [[0.0, 1.0], [1.0, 0.0]]
            reference_sha = hashlib.sha256(
                json.dumps(
                    [tuple(point) for point in reference_points],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            metric_cases = {}
            instance_files = []
            for case in motsp_cases + mokp_cases:
                source = root / f"{case}.calibration.json"
                source.write_text(json.dumps({"case": case, "role": "calibration"}) + "\n", encoding="utf-8")
                instance = root / f"{case}.instance.json"
                instance.write_text(json.dumps({"case": case, "instance": True}) + "\n", encoding="utf-8")
                metric_cases[case] = {
                    "source_artifact": {
                        "path": source.name,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    },
                    "source_role": "calibration_only_disjoint_from_current_arms",
                    "reference_sha256": reference_sha,
                    "reference_points": reference_points,
                    "ideal": [0.0, 0.0],
                    "nadir": [1.0, 1.0],
                    "hv_reference": [1.1, 1.1],
                }
                instance_files.append(
                    {
                        "case_id": case,
                        "path": instance.name,
                        "sha256": hashlib.sha256(instance.read_bytes()).hexdigest(),
                    }
                )
            metric = {
                "schema": "ijoc_metric_reference_manifest_v2",
                "cases": metric_cases,
            }
            metric_path = root / "metric.json"
            metric_path.write_text(json.dumps(metric) + "\n", encoding="utf-8")
            rows = []
            for family in families:
                for case in family["cases"]:
                    for algorithm in family["algorithms"]:
                        for seed in seeds:
                            for budget in budgets:
                                readable_configuration = {
                                    "case_id": case,
                                    "algorithm": algorithm,
                                    "seed": seed,
                                    "budget": budget,
                                }
                                configuration_sha = hashlib.sha256(
                                    json.dumps(
                                        readable_configuration,
                                        sort_keys=True,
                                        separators=(",", ":"),
                                        allow_nan=False,
                                    ).encode("utf-8")
                                ).hexdigest()
                                rows.append(
                                    {
                                        "case_id": case,
                                        "algorithm": algorithm,
                                        "seed": seed,
                                        "budget": budget,
                                        "configuration": readable_configuration,
                                        "configuration_sha256": configuration_sha,
                                    }
                                )
            config = {"schema": "ijoc_algorithm_configuration_matrix_v1", "rows": rows}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
            source_archive = root / "source.tar.gz"
            source_archive.write_bytes(b"frozen-source")
            dependency_lock = root / "requirements-lock.txt"
            dependency_lock.write_text("python==3.13\n", encoding="utf-8")
            baseline_bindings = []
            for algorithm in sorted(
                {baseline for family in families for baseline in family["required_baselines"]}
            ):
                artifact = root / f"{algorithm}.wrapper"
                artifact.write_text(f"wrapper for {algorithm}\n", encoding="utf-8")
                baseline_bindings.append(
                    {
                        "algorithm": algorithm,
                        "kind": "wrapper_script",
                        "version": "unit-test-v1",
                        "command": f"python {artifact.name} --input input.json --output output.json",
                        "artifact": {
                            "path": artifact.name,
                            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        },
                    }
                )
            reproducibility = {
                "schema": "ijoc_reproducibility_manifest_v2",
                "source_archive": {
                    "path": source_archive.name,
                    "sha256": hashlib.sha256(source_archive.read_bytes()).hexdigest(),
                },
                "instance_files": instance_files,
                "reproduction_commands": [
                    "python -m mo_nco.run_ijoc_preflight --study study.json --output audit.json"
                ],
                "baseline_bindings": baseline_bindings,
                "license": "MIT",
                "environment": {
                    "python_version": "3.13",
                    "dependency_lock": {
                        "path": dependency_lock.name,
                        "sha256": hashlib.sha256(
                            dependency_lock.read_bytes()
                        ).hexdigest(),
                    },
                },
            }
            reproducibility_path = root / "reproducibility.json"
            reproducibility_path.write_text(
                json.dumps(reproducibility) + "\n",
                encoding="utf-8",
            )
            study = {
                "schema": "ijoc_competitive_study_v3",
                "study_id": "unit",
                "problem_families": families,
                "seeds": seeds,
                "budgets": budgets,
                "anytime_checkpoint_period": 10,
                "metric_reference_manifest": {
                    "path": metric_path.name,
                    "sha256": hashlib.sha256(metric_path.read_bytes()).hexdigest(),
                },
                "algorithm_configuration_matrix": {
                    "path": config_path.name,
                    "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                },
                "artifact_release": {
                    "path": reproducibility_path.name,
                    "sha256": hashlib.sha256(
                        reproducibility_path.read_bytes()
                    ).hexdigest(),
                },
            }
            study_path = root / "study.json"
            study_path.write_text(json.dumps(study) + "\n", encoding="utf-8")
            result = audit_ijoc_competitive_study(study_path)
            self.assertEqual(result.submission_preflight_gate, "PASS")
            self.assertEqual(result.expected_run_count, len(rows))
            self.assertEqual(result.case_count, 30)
            self.assertEqual(result.evidence_status, "NOT_RUN")

            config["rows"].pop()
            config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
            study["algorithm_configuration_matrix"]["sha256"] = hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest()
            study_path.write_text(json.dumps(study) + "\n", encoding="utf-8")
            failed = audit_ijoc_competitive_study(study_path)
            self.assertEqual(failed.exact_matrix_gate, "FAIL")
            self.assertEqual(failed.submission_preflight_gate, "FAIL")

            study["algorithm_configuration_matrix"]["path"] = "../outside.json"
            study_path.write_text(json.dumps(study) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                audit_ijoc_competitive_study(study_path)


if __name__ == "__main__":
    unittest.main()

