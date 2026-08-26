from __future__ import annotations

import unittest

from mo_nco.run_theorem_certificates import build_report


class TheoremCertificateTests(unittest.TestCase):
    @staticmethod
    def _run_row(algorithm: str) -> dict[str, str]:
        return {
            "algorithm": algorithm,
            "case": "toy",
            "seed": "0",
            "budget": "64",
            "case_relative_hypervolume_2d": "1.0",
            "case_relative_anytime_hv_eval_auc": "1.0",
            "case_relative_anytime_hv_time_auc": "1.0",
            "case_relative_hypervolume_per_second": "1.0",
        }

    def test_report_distinguishes_child_acceptance_from_neighbor_replacements(self) -> None:
        report = build_report(
            [self._run_row("method"), self._run_row("baseline")],
            [
                {
                    "algorithm": "method",
                    "metadata": {
                        "neural_generated_children": 2,
                        "neural_accepted_children": 1,
                        "neural_accepted_replacements": 4,
                    },
                }
            ],
            method="method",
            baselines=["baseline"],
            min_quality_margin=0.0,
            source="memory",
        )
        self.assertIn("accepted-child rate", report)
        self.assertIn("neighbor replacements/neural child", report)
        self.assertIn("| method | 1 |", report)
        self.assertIn("| 0.5 | 2 |", report)
        self.assertIn("can exceed one", report)

    def test_legacy_metadata_does_not_masquerade_as_zero_acceptance(self) -> None:
        report = build_report(
            [self._run_row("method"), self._run_row("baseline")],
            [
                {
                    "algorithm": "method",
                    "metadata": {
                        "neural_generated_children": 2,
                        "neural_accepted_replacements": 4,
                    },
                }
            ],
            method="method",
            baselines=["baseline"],
            min_quality_margin=0.0,
            source="memory",
        )
        method_line = next(line for line in report.splitlines() if line.startswith("| method | 1 |"))
        self.assertIn("| missing | 2 |", method_line)

    def test_theory_optimized_contract_gate_rejects_legacy_prior_metadata(self) -> None:
        method = "ips-neural-mv-jitgreedy-targetflow-theory-optimized"
        report = build_report(
            [self._run_row(method), self._run_row("baseline")],
            [
                {
                    "algorithm": method,
                    "metadata": {
                        "neural_generated_children": 1,
                        "neural_accepted_children": 1,
                        "neural_endpoint_contract_required": True,
                        "neural_endpoint_contract_satisfied": False,
                        "move_target_only_contract_required": True,
                        "move_target_only_contract_satisfied": True,
                    },
                }
            ],
            method=method,
            baselines=["baseline"],
            min_quality_margin=0.0,
            source="memory",
        )
        self.assertIn("endpoint/target-only runtime contract gate: FAIL", report)

    def test_theory_optimized_contract_gate_accepts_hashed_frozen_priors(self) -> None:
        method = "ips-neural-mv-jitgreedy-targetflow-theory-optimized"
        report = build_report(
            [self._run_row(method), self._run_row("baseline")],
            [
                {
                    "algorithm": method,
                    "metadata": {
                        "neural_endpoint_contract_required": True,
                        "neural_endpoint_contract_satisfied": True,
                        "move_target_only_contract_required": True,
                        "move_target_only_contract_satisfied": True,
                        "neural_prior_sha256": "a" * 64,
                        "learned_move_prior_sha256": "b" * 64,
                        "neural_online_training": False,
                        "learned_move_updates": 0,
                        "learned_move_runtime_flow_head_weight": 0,
                        "learned_move_runtime_mean_field_head_weight": 0,
                        "learned_move_runtime_conductance_head_weight": 0,
                    },
                }
            ],
            method=method,
            baselines=["baseline"],
            min_quality_margin=0.0,
            source="memory",
        )
        self.assertIn("endpoint/target-only runtime contract gate: PASS", report)


if __name__ == "__main__":
    unittest.main()

