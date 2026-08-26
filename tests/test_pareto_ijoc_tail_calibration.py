from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

from mo_nco.pareto_ijoc_freeze import _canonical_digest


_RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "ijoc_submission_v20"
    / "scripts"
    / "run_tail_calibration.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "ijoc_run_tail_calibration_contract", _RUNNER_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load the tail-calibration runner.")
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)


class TailCalibrationFreezeContractTests(unittest.TestCase):
    def test_decision_rule_digest_matches_formal_freezer_canonical_value(self) -> None:
        policy = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "ijoc_submission_v20"
                / "calibration"
                / "frozen"
                / "tail_policy_freeze.json"
            ).read_text(encoding="utf-8")
        )
        decision_rule = policy["decision_rule"]
        self.assertEqual(
            _RUNNER.canonical_value_sha256(decision_rule),
            _canonical_digest(decision_rule),
        )
        self.assertEqual(
            policy["decision_rule_sha256"],
            _RUNNER.canonical_value_sha256(decision_rule),
        )

    def test_resumed_pipeline_gate_is_hash_stable_across_launcher_timings(self) -> None:
        first = _RUNNER.build_pipeline_gate_result(
            row_count=24,
            resumed_row_count=24,
            launcher_elapsed_seconds=0.001,
            maximum_elapsed_seconds=3600,
        )
        second = _RUNNER.build_pipeline_gate_result(
            row_count=24,
            resumed_row_count=24,
            launcher_elapsed_seconds=9.999,
            maximum_elapsed_seconds=3600,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["launcher_elapsed_seconds"], 0.0)
        self.assertEqual(
            first["elapsed_interpretation"],
            "resume_validation_elapsed_not_remeasured",
        )


if __name__ == "__main__":
    unittest.main()

