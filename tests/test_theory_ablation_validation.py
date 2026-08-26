from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from mo_nco.theory_ablation_validation import (
    ARM_CONTRACTS,
    analyze_case_cluster_contrast,
    load_suite_output,
    validate_v2_contract,
)


class TheoryAblationValidationTests(unittest.TestCase):
    def _write_valid_fixture(self, root: Path) -> None:
        algorithms = tuple(ARM_CONTRACTS)
        fields = (
            "case",
            "algorithm",
            "seed",
            "evaluations",
            "case_relative_hypervolume_2d",
            "case_relative_anytime_hv_eval_auc",
            "igd_plus",
        )
        rows = []
        for case_idx, case in enumerate(("case_a", "case_b")):
            case_dir = root / case
            case_dir.mkdir(parents=True)
            metadata_rows = []
            for seed in (100, 101):
                initial_hash = f"initial-{case}-{seed}"
                rng_hash = f"rng-{case}-{seed}"
                for arm_idx, algorithm in enumerate(algorithms):
                    rows.append(
                        {
                            "case": case,
                            "algorithm": algorithm,
                            "seed": seed,
                            "evaluations": 32,
                            "case_relative_hypervolume_2d": 1.0 + 0.01 * arm_idx + 0.001 * case_idx,
                            "case_relative_anytime_hv_eval_auc": 1.0 + 0.005 * arm_idx,
                            "igd_plus": 0.2 - 0.01 * arm_idx,
                        }
                    )
                    contract = ARM_CONTRACTS[algorithm]
                    mode = contract.rsplit(":", 1)[1]
                    scalar_hash = "scalar-hash" if mode in {"scalar", "full"} else ""
                    move_hash = "move-hash" if mode in {"move", "full"} else ""
                    metadata_rows.append(
                        {
                            "algorithm": algorithm,
                            "seed": seed,
                            "evaluations": 32,
                            "execution_order_contract": "seed-major-balanced-v1",
                            "metric_reference_contract": "frozen_external_v1",
                            "metric_reference_manifest_sha256": "reference-hash",
                            "metadata": {
                                "ablation_contract": contract,
                                "evaluation_budget": 32,
                                "evaluations_used": 32,
                                "prior_loading_rng_isolated": True,
                                "initial_population_sha256": initial_hash,
                                "base_rng_state_after_initialization_sha256": rng_hash,
                                "local_move_check_upper_bound": 999,
                                "accelerator_fallbacks": [],
                                "context_jump_accounting_complete": True,
                                "neural_prior_sha256": scalar_hash,
                                "learned_move_prior_sha256": move_hash,
                            },
                        }
                    )
            (case_dir / "run_metadata.jsonl").write_text(
                "\n".join(json.dumps(row) for row in metadata_rows) + "\n",
                encoding="utf-8",
            )
        with (root / "aggregate_runs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_valid_v2_contract_checks_cartesian_rng_priors_and_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_valid_fixture(root)
            rows, metadata = load_suite_output(root)
            report = validate_v2_contract(
                rows,
                metadata,
                expected_cases=2,
                expected_seeds=(100, 101),
                expected_evaluations=32,
                require_execution_order="seed-major-balanced-v1",
                require_metric_reference="frozen_external_v1",
            )
        self.assertTrue(report["passed"])
        self.assertTrue(all(check["passed"] for check in report["checks"]))

    def test_contract_fails_when_initial_population_hash_differs_between_arms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_valid_fixture(root)
            path = root / "case_a" / "run_metadata.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[1]["metadata"]["initial_population_sha256"] = "contaminated"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            aggregate, metadata = load_suite_output(root)
            report = validate_v2_contract(
                aggregate,
                metadata,
                expected_cases=2,
                expected_seeds=(100, 101),
                expected_evaluations=32,
            )
        self.assertFalse(report["passed"])
        failed = {check["name"] for check in report["checks"] if not check["passed"]}
        self.assertIn("matched_initial_population_hash", failed)

    def test_case_cluster_analysis_uses_cases_not_seed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_valid_fixture(root)
            rows, _ = load_suite_output(root)
            summary = analyze_case_cluster_contrast(
                rows,
                left="ips-neural-mv-jitgreedy-targetflow-theory-optimized",
                right="ips-theory-heavy-no-prior",
                metric="case_relative_hypervolume_2d",
                higher_is_better=True,
                bootstrap_draws=200,
                randomization_draws=200,
                random_seed=7,
            )
        self.assertEqual(summary["independent_case_units"], 2)
        self.assertEqual(summary["matched_seed_rows"], 4)
        self.assertAlmostEqual(summary["mean_delta"], 0.03)


if __name__ == "__main__":
    unittest.main()

