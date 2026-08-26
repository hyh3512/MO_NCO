from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from mo_nco.run_pareto_smc_strict_analysis import analyze


FIELDNAMES = (
    "case",
    "algorithm",
    "seed",
    "case_relative_hypervolume_2d",
    "case_relative_anytime_hv_eval_auc",
    "igd_plus",
    "additive_epsilon",
    "runtime_seconds",
)


def _row(case: str, algorithm: str, seed: int, offset: float) -> dict[str, object]:
    return {
        "case": case,
        "algorithm": algorithm,
        "seed": seed,
        "case_relative_hypervolume_2d": 1.0 + offset,
        "case_relative_anytime_hv_eval_auc": 1.0 + offset,
        "igd_plus": 0.1 - offset,
        "additive_epsilon": 0.2 - offset,
        "runtime_seconds": 1.0 - offset,
    }


class ParetoSMCStrictAnalysisTests(unittest.TestCase):
    def _analyze(self, rows: list[dict[str, object]], expected_seeds: int):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = Path(directory) / "aggregate_runs.csv"
            with aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)
            return analyze(
                aggregate,
                anchor="annealed-pareto-smc",
                expected_cases=1,
                expected_seeds=expected_seeds,
                bootstrap_repetitions=20,
                randomization_repetitions=20,
                random_seed=7,
                igd_noninferiority_margin=0.01,
            )

    def test_accepts_complete_matched_nonzero_seed_ids(self) -> None:
        rows = []
        for seed in (10, 20):
            rows.append(_row("case-1", "annealed-pareto-smc", seed, 0.1))
            rows.append(_row("case-1", "control", seed, 0.0))
        report = self._analyze(rows, expected_seeds=2)
        self.assertEqual(report["seed_ids"], [10, 20])
        self.assertEqual(report["matched_pairs_per_comparator"], 2)

    def test_rejects_silently_ignored_extra_seed(self) -> None:
        rows = [
            _row("case-1", "annealed-pareto-smc", 0, 0.1),
            _row("case-1", "control", 0, 0.0),
            _row("case-1", "annealed-pareto-smc", 1, 0.1),
            _row("case-1", "control", 1, 0.0),
        ]
        with self.assertRaisesRegex(
            ValueError,
            "Expected 1 distinct seed IDs",
        ):
            self._analyze(rows, expected_seeds=1)

    def test_rejects_incomplete_case_algorithm_seed_matrix(self) -> None:
        rows = [
            _row("case-1", "annealed-pareto-smc", 0, 0.1),
            _row("case-1", "control", 0, 0.0),
            _row("case-1", "annealed-pareto-smc", 1, 0.1),
        ]
        with self.assertRaisesRegex(
            ValueError,
            "matrix is incomplete",
        ):
            self._analyze(rows, expected_seeds=2)


if __name__ == "__main__":
    unittest.main()

