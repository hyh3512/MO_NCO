from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from mo_nco.external_pymoo_baseline import run_pymoo


def _pymoo_available() -> bool:
    try:
        import pymoo  # noqa: F401

        return True
    except Exception:
        return False


@unittest.skipUnless(_pymoo_available(), "pymoo is optional")
class ExternalPymooBaselineTests(unittest.TestCase):
    @staticmethod
    def _payload(seed: int, evaluations: int = 20) -> dict[str, object]:
        first = (
            (0, 1, 4, 7, 3),
            (1, 0, 2, 6, 5),
            (4, 2, 0, 1, 8),
            (7, 6, 1, 0, 2),
            (3, 5, 8, 2, 0),
        )
        second = (
            (0, 8, 3, 2, 6),
            (8, 0, 5, 7, 1),
            (3, 5, 0, 4, 9),
            (2, 7, 4, 0, 5),
            (6, 1, 9, 5, 0),
        )
        return {
            "name": "tiny-exact-budget",
            "num_cities": 5,
            "num_objectives": 2,
            "population_size": 10,
            "evaluations": evaluations,
            "seed": seed,
            "distance_matrices": [first, second],
            "anytime_checkpoint_period": 5,
        }

    @staticmethod
    def _tradeoff_payload(seed: int) -> dict[str, object]:
        cities = 7
        first = [[0] * cities for _ in range(cities)]
        value = 1
        for left in range(cities):
            for right in range(left + 1, cities):
                weight = (
                    value * value * 37 + value * 11 + 17
                ) % 401 + 1
                first[left][right] = weight
                first[right][left] = weight
                value += 1
        second = [
            [
                0 if left == right else 500 - first[left][right]
                for right in range(cities)
            ]
            for left in range(cities)
        ]
        return {
            "name": "anti-correlated-all-history",
            "num_cities": cities,
            "num_objectives": 2,
            "population_size": 10,
            "evaluations": 100,
            "seed": seed,
            "distance_matrices": [first, second],
            "anytime_checkpoint_period": 10,
        }

    def test_exact_budget_for_nsga2_and_moead_across_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for algorithm in ("nsga2", "moead"):
                for seed in (0, 1, 2):
                    input_path = root / f"{algorithm}-{seed}.json"
                    output_path = root / f"{algorithm}-{seed}.csv"
                    input_path.write_text(
                        json.dumps(self._payload(seed)),
                        encoding="utf-8",
                    )
                    run_pymoo(algorithm, input_path, output_path)
                    with output_path.open(
                        newline="",
                        encoding="utf-8",
                    ) as handle:
                        final_rows = list(csv.DictReader(handle))
                    self.assertTrue(final_rows)
                    self.assertEqual(
                        {int(row["evaluations"]) for row in final_rows},
                        {20},
                    )
                    diagnostics_path = output_path.with_suffix(
                        ".diagnostics.csv"
                    )
                    with diagnostics_path.open(
                        newline="",
                        encoding="utf-8",
                    ) as handle:
                        diagnostics = list(csv.DictReader(handle))
                    self.assertTrue(diagnostics)
                    self.assertEqual(
                        max(
                            int(row["evaluations"])
                            for row in diagnostics
                        ),
                        20,
                    )
                    observed_steps = {
                        int(row["evaluations"])
                        for row in diagnostics
                    }
                    self.assertTrue(
                        {5, 10, 15, 20}.issubset(observed_steps)
                    )
                    self.assertTrue(
                        all(
                            float(row["elapsed_seconds"]) > 0.0
                            for row in diagnostics
                        )
                    )

    def test_nondivisible_budget_fails_instead_of_rounding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"
            output_path = root / "output.csv"
            input_path.write_text(
                json.dumps(self._payload(0, evaluations=21)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "divisible"):
                run_pymoo("nsga2", input_path, output_path)

    def test_final_archive_includes_survival_rejected_offspring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for algorithm in ("nsga2", "moead"):
                input_path = root / f"{algorithm}.json"
                output_path = root / f"{algorithm}.csv"
                input_path.write_text(
                    json.dumps(self._tradeoff_payload(seed=7)),
                    encoding="utf-8",
                )
                run_pymoo(algorithm, input_path, output_path)
                with output_path.open(
                    newline="",
                    encoding="utf-8",
                ) as handle:
                    final_rows = list(csv.DictReader(handle))
                with output_path.with_suffix(".diagnostics.csv").open(
                    newline="",
                    encoding="utf-8",
                ) as handle:
                    diagnostic_rows = list(csv.DictReader(handle))
                last_rows = [
                    row
                    for row in diagnostic_rows
                    if int(row["evaluations"]) == 100
                ]
                final_objectives = {
                    (
                        float(row["objective_0"]),
                        float(row["objective_1"]),
                    )
                    for row in final_rows
                }
                last_objectives = {
                    (
                        float(row["objective_0"]),
                        float(row["objective_1"]),
                    )
                    for row in last_rows
                }
                with self.subTest(algorithm=algorithm):
                    self.assertGreater(len(final_objectives), 10)
                    self.assertEqual(
                        final_objectives,
                        last_objectives,
                    )

    def test_moead_population_one_fails_before_evaluation(self) -> None:
        payload = self._payload(seed=0, evaluations=20)
        payload["population_size"] = 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"
            output_path = root / "output.csv"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, ">= 2"):
                run_pymoo("moead", input_path, output_path)
        self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()

