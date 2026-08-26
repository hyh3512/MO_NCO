from __future__ import annotations

import csv
import itertools
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mo_nco.baselines import (
    MOEADOptimizer,
    MOTSPParetoLocalSearchOptimizer,
    NSGAIIOptimizer,
    RandomTwoOptOptimizer,
    order_crossover,
)
from mo_nco.instance import MultiObjectiveTSPInstance
from mo_nco.mature_baselines import (
    ExternalBaselineConfig,
    ExternalBaselineOptimizer,
    MatureBaselineUnavailable,
    load_external_baseline_from_env,
)


class BaselineTests(unittest.TestCase):
    def test_external_objectives_are_recomputed_and_mismatches_rejected(
        self,
    ) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(5, seed=17)
        optimizer = ExternalBaselineOptimizer(
            instance,
            ExternalBaselineConfig(command=("unused",)),
            population_size=4,
            evaluations=20,
            seed=3,
        )
        tour = (0, 1, 2, 3, 4)
        local = instance.evaluate(tour)
        row = {
            "objective_0": repr(local[0]),
            "objective_1": repr(local[1]),
        }
        self.assertEqual(
            optimizer._verified_local_objectives(row, tour),
            local,
        )
        row["objective_1"] = repr(local[1] + 1.0)
        with self.assertRaisesRegex(
            MatureBaselineUnavailable,
            "does not match local",
        ):
            optimizer._verified_local_objectives(row, tour)

    def test_external_budget_and_anytime_steps_fail_closed(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(5, seed=18)
        optimizer = ExternalBaselineOptimizer(
            instance,
            ExternalBaselineConfig(command=("unused",)),
            population_size=4,
            evaluations=20,
            seed=3,
        )
        tour = (0, 1, 2, 3, 4)
        objective = instance.evaluate(tour)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.csv"
            with output.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "tour",
                        "objective_0",
                        "objective_1",
                        "evaluations",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "tour": " ".join(map(str, tour)),
                        "objective_0": repr(objective[0]),
                        "objective_1": repr(objective[1]),
                        "evaluations": "",
                    }
                )
            with self.assertRaisesRegex(
                MatureBaselineUnavailable,
                "must report",
            ):
                optimizer._read_output(output)

            optimizer._external_evaluations = 20
            diagnostics = output.with_suffix(".diagnostics.csv")
            with diagnostics.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "evaluations",
                        "tour",
                        "objective_0",
                        "objective_1",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "evaluations": "1.5",
                        "tour": " ".join(map(str, tour)),
                        "objective_0": repr(objective[0]),
                        "objective_1": repr(objective[1]),
                    }
                )
            with self.assertRaisesRegex(
                MatureBaselineUnavailable,
                "invalid evaluation count",
            ):
                optimizer._read_diagnostics(diagnostics)

    def test_external_anytime_fronts_are_rebuilt_cumulatively(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(5, seed=91)
        optimizer = ExternalBaselineOptimizer(
            instance,
            ExternalBaselineConfig(command=("unused",)),
            population_size=4,
            evaluations=20,
            seed=3,
        )
        candidates = [
            ((0, *tail), instance.evaluate((0, *tail)))
            for tail in itertools.permutations(range(1, 5))
        ]
        pair = None
        for left_tour, left_objectives in candidates:
            for right_tour, right_objectives in candidates:
                if left_tour == right_tour:
                    continue
                left_dominates = all(
                    left <= right
                    for left, right in zip(
                        left_objectives,
                        right_objectives,
                    )
                )
                right_dominates = all(
                    right <= left
                    for left, right in zip(
                        left_objectives,
                        right_objectives,
                    )
                )
                if not left_dominates and not right_dominates:
                    pair = (
                        (left_tour, left_objectives),
                        (right_tour, right_objectives),
                    )
                    break
            if pair is not None:
                break
        self.assertIsNotNone(pair)
        assert pair is not None
        optimizer._external_evaluations = 20
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.diagnostics.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "evaluations",
                        "tour",
                        "objective_0",
                        "objective_1",
                    ),
                )
                writer.writeheader()
                for step, (tour, objectives) in zip((10, 20), pair):
                    writer.writerow(
                        {
                            "evaluations": step,
                            "tour": " ".join(map(str, tour)),
                            "objective_0": repr(objectives[0]),
                            "objective_1": repr(objectives[1]),
                        }
                    )
            diagnostics = optimizer._read_diagnostics(path)
        self.assertEqual(len(diagnostics), 2)
        self.assertIn(pair[0][1], diagnostics[-1].front)
        self.assertIn(pair[1][1], diagnostics[-1].front)

    def test_order_crossover_preserves_permutation(self) -> None:
        import random

        rng = random.Random(4)
        a = (0, 1, 2, 3, 4, 5)
        b = (0, 5, 4, 3, 2, 1)
        child = order_crossover(a, b, rng)
        self.assertEqual(child[0], 0)
        self.assertEqual(sorted(child), list(range(6)))

    def test_baselines_small_run(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(9, seed=5)
        for cls in [RandomTwoOptOptimizer, NSGAIIOptimizer, MOEADOptimizer, MOTSPParetoLocalSearchOptimizer]:
            if cls is RandomTwoOptOptimizer:
                opt = cls(instance, num_particles=8, iterations=24, seed=5, log_period=8)
            else:
                opt = cls(instance, population_size=8, evaluations=32, seed=5, log_period=8)
            result = opt.run()
            self.assertGreater(len(result.archive), 0)
            self.assertGreater(len(result.diagnostics), 0)

    def test_motsp_pls_exposes_explicit_archive_tolerance(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(7, seed=19)
        optimizer = MOTSPParetoLocalSearchOptimizer(
            instance,
            population_size=4,
            evaluations=8,
            seed=3,
            log_period=4,
            archive_max_size=None,
            archive_tolerance=0.0,
            neighborhood_sample=4,
            anytime_checkpoint_period=4,
        )
        result = optimizer.run()
        self.assertEqual(result.archive.tol, 0.0)
        self.assertEqual(result.metadata["archive_tolerance"], 0.0)

    def test_motsp_pls_restart_v2_makes_progress_after_stalled_expansion(
        self,
    ) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(7, seed=29)
        with patch.object(
            MOTSPParetoLocalSearchOptimizer,
            "_expand_parent",
            return_value=None,
        ):
            result = MOTSPParetoLocalSearchOptimizer(
                instance,
                population_size=4,
                evaluations=12,
                seed=11,
                log_period=4,
                archive_max_size=None,
                archive_tolerance=0.0,
                neighborhood_sample=4,
                anytime_checkpoint_period=4,
                stalled_expansion_policy="uniform-random-unvisited-v1",
            ).run()
        self.assertEqual(result.metadata["algorithm"], "motsp-pls-restart-native-v2")
        self.assertEqual(result.metadata["evaluations_used"], 12)
        self.assertEqual(result.metadata["restart_evaluations"], 8)
        self.assertEqual(result.metadata["stalled_expansions"], 8)
        self.assertEqual(result.metadata["liveness_gate"], "PASS")

    def test_motsp_pls_restart_v2_is_deterministic_with_fallback(self) -> None:
        def execute() -> tuple[object, ...]:
            instance = MultiObjectiveTSPInstance.random_biobjective(7, seed=31)
            with patch.object(
                MOTSPParetoLocalSearchOptimizer,
                "_expand_parent",
                return_value=None,
            ):
                result = MOTSPParetoLocalSearchOptimizer(
                    instance,
                    population_size=4,
                    evaluations=12,
                    seed=13,
                    log_period=4,
                    archive_max_size=None,
                    archive_tolerance=0.0,
                    neighborhood_sample=4,
                    anytime_checkpoint_period=4,
                    stalled_expansion_policy="uniform-random-unvisited-v1",
                    restart_random_attempts=0,
                ).run()
            return (
                result.particles,
                result.objectives,
                result.archive.entries,
                tuple(
                    (item.iteration, item.archive_size, item.front)
                    for item in result.diagnostics
                ),
                result.metadata["checkpoint_solution_witnesses"],
                result.metadata["restart_evaluations"],
                result.metadata["restart_random_draws"],
                result.metadata["restart_fallbacks"],
            )

        first = execute()
        second = execute()
        self.assertEqual(first, second)
        self.assertEqual(first[-3:], (8, 0, 8))

    def test_motsp_pls_restart_v2_fails_if_tour_space_is_exhausted(
        self,
    ) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(3, seed=37)
        optimizer = MOTSPParetoLocalSearchOptimizer(
            instance,
            population_size=1,
            evaluations=3,
            seed=17,
            log_period=1,
            archive_max_size=None,
            archive_tolerance=0.0,
            neighborhood_sample=1,
            anytime_checkpoint_period=1,
            stalled_expansion_policy="uniform-random-unvisited-v1",
            restart_random_attempts=0,
        )
        with self.assertRaisesRegex(RuntimeError, "exhausted every fixed-zero tour"):
            optimizer.run()

    def test_mature_solver_bridge_env_uses_protocol_adapter(self) -> None:
        old_direct = os.environ.pop("MO_NCO_BASELINE_PAQUETE", None)
        old_bridge = os.environ.get("MO_NCO_BRIDGE_PAQUETE")
        os.environ["MO_NCO_BRIDGE_PAQUETE"] = f'"{sys.executable}" --version'
        try:
            config = load_external_baseline_from_env("paquete")
            self.assertEqual(config.command[:3], [sys.executable, "-m", "mo_nco.external_motsp_bridge"])
            self.assertEqual(config.command[3], "paquete")
        finally:
            if old_direct is not None:
                os.environ["MO_NCO_BASELINE_PAQUETE"] = old_direct
            if old_bridge is None:
                os.environ.pop("MO_NCO_BRIDGE_PAQUETE", None)
            else:
                os.environ["MO_NCO_BRIDGE_PAQUETE"] = old_bridge


if __name__ == "__main__":
    unittest.main()

