from __future__ import annotations

import unittest

from mo_nco.ijoc_mokp_baselines import MOKP_BASELINES, run_mokp_baseline
from mo_nco.pareto_ijoc_problem import MultiObjectiveKnapsackInstance


class IJOCMOKPBaselineTests(unittest.TestCase):
    def problem(self) -> MultiObjectiveKnapsackInstance:
        return MultiObjectiveKnapsackInstance.random_instance(
            14,
            seed=20260731,
        )

    def test_all_native_baselines_are_budget_exact_and_replayable(self) -> None:
        for offset, algorithm in enumerate(sorted(MOKP_BASELINES)):
            with self.subTest(algorithm=algorithm):
                problem = self.problem()
                result = run_mokp_baseline(
                    algorithm,
                    problem,
                    evaluations=60,
                    seed=100 + offset,
                    anytime_checkpoint_period=10,
                )
                self.assertEqual(result.metadata["evaluations_used"], 60)
                self.assertEqual(result.metadata["exact_budget_gate"], "PASS")
                self.assertEqual(
                    result.metadata["observed_anytime_checkpoints"],
                    (10, 20, 30, 40, 50, 60),
                )
                self.assertEqual(
                    len(result.metadata["checkpoint_solution_witnesses"]),
                    6,
                )
                for checkpoint in result.metadata[
                    "checkpoint_solution_witnesses"
                ]:
                    for entry in checkpoint["entries"]:
                        solution = tuple(entry["solution"])
                        self.assertEqual(
                            tuple(entry["objectives"]),
                            problem.evaluate(solution),
                        )
                for entry in result.archive.entries:
                    self.assertEqual(
                        entry.objectives,
                        problem.evaluate(entry.tour),
                    )

    def test_unknown_baseline_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown MOKP baseline"):
            run_mokp_baseline(
                "not-an-algorithm",
                self.problem(),
                evaluations=60,
                seed=0,
                anytime_checkpoint_period=10,
            )


if __name__ == "__main__":
    unittest.main()

