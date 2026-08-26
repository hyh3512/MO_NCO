from __future__ import annotations

import hashlib
import io
from contextlib import redirect_stdout
from importlib import metadata as importlib_metadata
import json
from datetime import datetime
import os
from pathlib import Path
import platform
import subprocess
import tarfile
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from mo_nco.pareto_ijoc_cold_matrix import (
    _strict_json,
    _study_components,
    _validate_plan,
    run_cold_process_matrix,
)
from mo_nco.pareto_ijoc_freeze import freeze_ijoc_study
from mo_nco.pareto_ijoc_postrun import audit_ijoc_post_run
from mo_nco.pareto_ijoc_preflight import audit_ijoc_competitive_study


_DUMMY_ADAPTER = r'''
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from mo_nco.frozen_probe import SOURCE_MARKER

if SOURCE_MARKER != "frozen-runtime-only":
    raise RuntimeError("adapter imported mo_nco from an unfrozen source tree")

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("run", "replay"))
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--algorithm-result")
parser.add_argument("--fail", action="store_true")
parser.add_argument("--sleep", type=float, default=0.0)
args = parser.parse_args()
time.sleep(args.sleep)
if args.fail:
    sys.exit(7)
input_path = Path(args.input)
packet = json.loads(input_path.read_text(encoding="utf-8"))
output = Path(args.output)
if args.mode == "run":
    archive = output.parent / "archive.json"
    archive.write_text(
        json.dumps(
            {"run_key": packet["run_key"], "solutions": [[0, 1], [1, 0]]},
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    budget = packet["run_key"]["budget"]
    period = packet["anytime_checkpoint_period"]
    checkpoints = output.parent / "checkpoint_witnesses.json"
    checkpoints.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {"evaluation": value, "solutions": [[0, 1], [1, 0]]}
                    for value in range(period, budget + 1, period)
                ]
            },
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    checkpoint_sha = hashlib.sha256(checkpoints.read_bytes()).hexdigest()
    payload = {
        "schema": "ijoc_algorithm_result_v1",
        "run_key": packet["run_key"],
        "status": "SUCCESS",
        "evaluations_used": budget,
        "observed_checkpoints": list(range(period, budget + 1, period)),
        "archive_artifact": {"path": archive.name, "sha256": archive_sha},
        "checkpoint_artifact": {
            "path": checkpoints.name,
            "sha256": checkpoint_sha,
        },
        "metrics": {"dummy_metric": 0.0},
    }
else:
    result_path = Path(args.algorithm_result)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    archive = result_path.parent / result["archive_artifact"]["path"]
    checkpoints = result_path.parent / result["checkpoint_artifact"]["path"]
    instance = Path(packet["instance_artifact"]["path"])
    payload = {
        "schema": "ijoc_replay_receipt_v1",
        "run_key": packet["run_key"],
        "status": "PASS",
        "instance_sha256": hashlib.sha256(instance.read_bytes()).hexdigest(),
        "algorithm_result_sha256": hashlib.sha256(
            result_path.read_bytes()
        ).hexdigest(),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "checkpoint_artifact_sha256": hashlib.sha256(
            checkpoints.read_bytes()
        ).hexdigest(),
        "evaluations_used": result["evaluations_used"],
        "observed_checkpoints": result["observed_checkpoints"],
    }
output.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
'''


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _make_request(root: Path, *, behavior: str = "success") -> Path:
    archive_source = root / "archive-source" / "mo_nco"
    archive_source.mkdir(parents=True)
    (archive_source / "__init__.py").write_text(
        "from .frozen_probe import SOURCE_MARKER\n",
        encoding="utf-8",
    )
    (archive_source / "frozen_probe.py").write_text(
        'SOURCE_MARKER = "frozen-runtime-only"\n',
        encoding="utf-8",
    )
    with tarfile.open(root / "source.tar.gz", mode="w:gz") as archive:
        archive.add(archive_source, arcname="mo_nco")
    (root / "requirements-lock.txt").write_text(
        f"pip=={importlib_metadata.version('pip')}\n", encoding="utf-8"
    )
    (root / "dummy_adapter.py").write_text(
        textwrap.dedent(_DUMMY_ADAPTER), encoding="utf-8"
    )
    evaluation_code_sha = hashlib.sha256(
        (root / "dummy_adapter.py").read_bytes()
    ).hexdigest()
    metric_contract = {
        "objective_sense": ["minimize", "minimize"],
        "dominance_tolerance": 0.0,
        "normalization": "frozen_ideal_nadir_affine",
        "archive_semantics": "calibration_all_evaluated_nondominated",
        "evaluation_code_sha256": evaluation_code_sha,
    }

    tail_case_ids: list[str] = []
    tail_instance_artifacts: list[dict[str, object]] = []
    for family, count in (("MOTSP", 5), ("MOKP", 5)):
        for index in range(count):
            case_id = f"tail-{family.lower()}-{index:02d}"
            tail_case_ids.append(case_id)
            artifact_count = 2 if family == "MOTSP" else 1
            for artifact_index in range(artifact_count):
                suffix = (
                    f"-objective-{artifact_index + 1}.tsp"
                    if family == "MOTSP"
                    else ".json"
                )
                path = root / f"{case_id}{suffix}"
                path.write_bytes(
                    (
                        f"tail calibration {case_id} artifact "
                        f"{artifact_index}\n"
                    ).encode("utf-8")
                )
                tail_instance_artifacts.append(
                    {
                        "case_id": case_id,
                        "family": family,
                        "path": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size,
                    }
                )
    calibration_manifest = root / "calibration-artifact-manifest.json"
    _write_json(
        calibration_manifest,
        {
            "schema": "ijoc_tail_calibration_evidence_v1",
            "status": "COMPLETE",
            "calibration_instance_artifacts": tail_instance_artifacts,
        },
    )
    calibration_receipt = root / "calibration-suite-receipt.json"
    _write_json(
        calibration_receipt,
        {
            "schema": "ijoc_calibration_suite_receipt_v1",
            "suite_id": "unit-test-calibration",
            "status": "COMPLETE",
            "evidence_scope": "tail_policy_selection_only",
            "calibration_case_ids": tail_case_ids,
            "candidate_policy_ids": ["uniform", "exp3"],
            "seeds": list(range(5)),
            "artifact_manifest": {
                "path": calibration_manifest.name,
                "sha256": hashlib.sha256(
                    calibration_manifest.read_bytes()
                ).hexdigest(),
            },
            "instance_artifacts": tail_instance_artifacts,
        },
    )
    calibration_suite_sha = hashlib.sha256(
        calibration_receipt.read_bytes()
    ).hexdigest()
    decision_rule = {
        "primary": "paired_case_mean_hv_auc",
        "fallback": "uniform",
    }
    decision_rule_sha = hashlib.sha256(
        json.dumps(
            decision_rule, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    tail_policy = root / "tail-policy.json"
    _write_json(
        tail_policy,
        {
            "schema": "ijoc_tail_policy_freeze_v1",
            "status": "FROZEN",
            "policy_id": "uniform",
            "calibration_suite_sha256": calibration_suite_sha,
            "selection_gate": "FALLBACK",
            "decision_rule": decision_rule,
            "decision_rule_sha256": decision_rule_sha,
            "configuration": {
                "allocation_policy": "uniform",
                "adaptive_search_evaluations": 10,
            },
            "fallback_applied": True,
        },
    )
    tail_policy_sha = hashlib.sha256(tail_policy.read_bytes()).hexdigest()
    family_cases: dict[str, list[dict[str, object]]] = {
        "motsp": [],
        "mokp": [],
    }
    reference_precommit_cases: list[dict[str, object]] = []
    for family in family_cases:
        for index in range(15):
            case_id = f"{family}-{index:02d}"
            artifact_count = 2 if family == "motsp" else 1
            packet_artifacts: list[dict[str, str]] = []
            for artifact_index in range(artifact_count):
                suffix = (
                    f"-objective-{artifact_index + 1}.tsp"
                    if family == "motsp"
                    else "-instance.json"
                )
                child = root / f"{case_id}{suffix}"
                child.write_bytes(
                    (
                        f"formal {family} {case_id} child "
                        f"{artifact_index}\n"
                    ).encode("utf-8")
                )
                packet_artifacts.append(
                    {
                        "path": child.name,
                        "sha256": hashlib.sha256(
                            child.read_bytes()
                        ).hexdigest(),
                    }
                )
            packet = root / f"{case_id}.packet.json"
            _write_json(
                packet,
                {
                    "schema": "ijoc_case_instance_packet_v1",
                    "case_id": case_id,
                    "family": family.upper(),
                    "problem_sha256": hashlib.sha256(
                        f"problem:{case_id}".encode("utf-8")
                    ).hexdigest(),
                    "artifacts": packet_artifacts,
                },
            )
            reference_precommit_cases.append(
                {
                    "case_id": case_id,
                    "family": family.upper(),
                    "instance_artifact_sha256": [
                        artifact["sha256"] for artifact in packet_artifacts
                    ],
                }
            )
            family_cases[family].append(
                {
                    "id": case_id,
                    "instance_path": packet.name,
                    "metric_reference": {
                        "source_artifact_path": (
                            f"{case_id}.calibration-reference.json"
                        ),
                    },
                }
            )

    reference_precommit = root / "reference-calibration-precommit.json"
    _write_json(
        reference_precommit,
        {
            "schema": "ijoc_reference_calibration_precommit_v1",
            "suite_id": "unit-test-reference-calibration",
            "status": "PRECOMMITTED",
            "evidence_scope": "metric_reference_construction_only",
            "cases": reference_precommit_cases,
            "algorithms": ["reference-exact-enumerator-v1"],
            "seeds": [100],
            "budgets": [5],
            "metric_contract": metric_contract,
        },
    )
    reference_precommit_sha = hashlib.sha256(
        reference_precommit.read_bytes()
    ).hexdigest()
    reference_runs: list[dict[str, object]] = []
    case_outputs: list[dict[str, object]] = []
    for descriptor in reference_precommit_cases:
        case_id = str(descriptor["case_id"])
        run_source = root / f"{case_id}.reference-run.json"
        _write_json(
            run_source,
            {
                "case_id": case_id,
                "algorithm": "reference-exact-enumerator-v1",
                "seed": 100,
                "budget": 5,
                "archive": [[0.0, 1.0], [1.0, 0.0]],
            },
        )
        reference_runs.append(
            {
                "case_id": case_id,
                "algorithm": "reference-exact-enumerator-v1",
                "seed": 100,
                "budget": 5,
                "source_artifacts": [
                    {
                        "role": "nondominated_archive",
                        "path": run_source.name,
                        "sha256": hashlib.sha256(
                            run_source.read_bytes()
                        ).hexdigest(),
                        "bytes": run_source.stat().st_size,
                    }
                ],
            }
        )
        reference = root / f"{case_id}.calibration-reference.json"
        _write_json(
            reference,
            {
                "schema": "ijoc_calibration_reference_case_v1",
                "case_id": case_id,
                "source_role": (
                    "reference_calibration_precommitted_"
                    "disjoint_arms_and_seeds"
                ),
                "reference_calibration_precommit_sha256": (
                    reference_precommit_sha
                ),
                "metric_contract": metric_contract,
                "reference_points": [[0.0, 1.0], [1.0, 0.0]],
                "ideal": [0.0, 0.0],
                "nadir": [1.0, 1.0],
                "hv_reference": [1.1, 1.1],
            },
        )
        case_outputs.append(
            {
                "case_id": case_id,
                "path": reference.name,
                "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                "bytes": reference.stat().st_size,
            }
        )
    reference_evidence = root / "reference-calibration-evidence.json"
    _write_json(
        reference_evidence,
        {
            "schema": (
                "ijoc_reference_calibration_completion_evidence_v1"
            ),
            "status": "COMPLETE",
            "reference_calibration_precommit_sha256": (
                reference_precommit_sha
            ),
            "reference_runs": reference_runs,
            "case_outputs": case_outputs,
        },
    )
    reference_completion = (
        root / "reference-calibration-completion-receipt.json"
    )
    _write_json(
        reference_completion,
        {
            "schema": (
                "ijoc_reference_calibration_completion_receipt_v1"
            ),
            "suite_id": "unit-test-reference-calibration",
            "status": "COMPLETE",
            "evidence_scope": "metric_reference_construction_only",
            "reference_calibration_precommit_sha256": (
                reference_precommit_sha
            ),
            "reference_runs": reference_runs,
            "case_outputs": case_outputs,
            "artifact_manifest": {
                "path": reference_evidence.name,
                "sha256": hashlib.sha256(
                    reference_evidence.read_bytes()
                ).hexdigest(),
            },
        },
    )
    algorithm_ids = [
        "ijoc-pareto-smc",
        "dummy-baseline-a",
        "dummy-baseline-b",
        "dummy-baseline-c",
    ]
    algorithms = {}
    for algorithm_id in algorithm_ids:
        command_argv = [
            "{python_executable}",
            "{adapter_path}",
            "run",
            "--input",
            "{input_path}",
            "--output",
            "{result_path}",
            "--sleep",
            (
                "1"
                if behavior == "timeout"
                else "0.2"
                if behavior == "slow_success"
                else "0.02"
            ),
        ]
        if behavior == "failure":
            command_argv.append("--fail")
        algorithms[algorithm_id] = {
            "role": (
                "treatment" if algorithm_id == "ijoc-pareto-smc" else "baseline"
            ),
            "families": ["motsp", "mokp"],
            "kind": "wrapper_script",
            "version": "unit-test-only-v1",
            "adapter_artifact_path": "dummy_adapter.py",
            "command_argv": command_argv,
            "replay_verifier_artifact_path": "dummy_adapter.py",
            "replay_verifier_argv": [
                "{python_executable}",
                "{replay_verifier_path}",
                "replay",
                "--input",
                "{input_path}",
                "--algorithm-result",
                "{result_path}",
                "--output",
                "{replay_result_path}",
                "--sleep",
                "0.02",
            ],
            "configuration": {"environment": {"OMP_NUM_THREADS": "1"}},
        }
    formal_analysis_plan = root / "formal-analysis-plan.json"
    _write_json(
        formal_analysis_plan,
        {
            "schema": "ijoc_formal_analysis_plan_v1",
            "plan_id": "unit-test-formal-analysis",
            "status": "PRECOMMITTED_BEFORE_FORMAL_EXECUTION",
            "formal_evidence_status": "NOT_RUN",
            "families": ["MOKP", "MOTSP"],
            "treatment": "ijoc-pareto-smc",
            "required_baselines": {
                "MOKP": algorithm_ids[1:],
                "MOTSP": algorithm_ids[1:],
            },
            "formal_seeds": list(range(10)),
            "evaluation_budgets": [10, 20, 30],
            "anytime_checkpoint_period": 10,
            "primary_budget": 30,
            "primary_metric": {"name": "unit-test-hv-auc"},
            "secondary_metrics": [{"name": "unit-test-final-hv"}],
            "comparison_unit": "same_family_case_seed_budget",
            "cluster_unit": "case_id",
            "family_pooling": "forbidden",
            "budget_pooling": "forbidden",
            "paired_contrast_orientation": "positive_favors_treatment",
            "uncertainty": {"method": "unit-test-only"},
            "wins_ties_losses": {"unit": "paired_case_seed"},
            "primary_gate": {"decision": "unit-test-only"},
            "efficiency_claim_gate": {"decision": "unit-test-only"},
            "missing_or_failed_rows": {"action": "HOLD"},
            "reference_scope": "supplied-reference-relative-only",
            "randomness_scope": "pseudo-random-computational-experiment",
        },
    )
    request = {
        "schema": "ijoc_manifest_freeze_request_v1",
        "study_id": "unit-test-plumbing-only",
        "evidence_status": "NOT_RUN",
        "problem_families": [
            {
                "id": family,
                "cases": cases,
                "algorithms": algorithm_ids,
                "required_baselines": algorithm_ids[1:],
            }
            for family, cases in family_cases.items()
        ],
        "algorithms": algorithms,
        "seeds": list(range(10)),
        "budgets": [10, 20, 30],
        "anytime_checkpoint_period": 10,
        "source_archive_path": "source.tar.gz",
        "dependency_lock_path": "requirements-lock.txt",
        "tail_calibration_suite_receipt_path": calibration_receipt.name,
        "reference_calibration_precommit_path": (
            reference_precommit.name
        ),
        "reference_calibration_completion_receipt_path": (
            reference_completion.name
        ),
        "tail_policy_artifact_path": tail_policy.name,
        "formal_analysis_plan_path": formal_analysis_plan.name,
        "python_version": platform.python_version(),
        "license": "MIT",
        "reproduction_commands": [
            "python scripts/run_ijoc_cold_matrix.py --study study.json "
            "--execution-plan execution_plan.json --results-directory results "
            "--timeout-seconds 60"
        ],
    }
    request_path = root / "freeze_request.json"
    _write_json(request_path, request)
    return request_path


class IJOCExperimentInfrastructureTests(unittest.TestCase):
    def _freeze(self, root: Path, *, behavior: str = "success"):
        return freeze_ijoc_study(
            _make_request(root, behavior=behavior), root / "frozen"
        )

    def test_freezer_materializes_real_hash_bound_manifests_as_not_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self._freeze(root)
            self.assertEqual(frozen.expected_run_count, 30 * 4 * 10 * 3)
            self.assertEqual(frozen.evidence_status, "NOT_RUN")
            preflight = audit_ijoc_competitive_study(frozen.study_path)
            self.assertEqual(preflight.submission_preflight_gate, "PASS")
            self.assertEqual(preflight.evidence_status, "NOT_RUN")
            receipt = json.loads(frozen.receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["formal_evidence_status"], "NOT_RUN")
            plan = json.loads(
                frozen.execution_plan_path.read_text(encoding="utf-8")
            )
            for binding in plan["case_instances"].values():
                self.assertFalse(Path(binding["path"]).is_absolute())
            for case_id, expected_children in (
                ("motsp-00", 2),
                ("mokp-00", 1),
            ):
                packet_path = (
                    frozen.execution_plan_path.parent
                    / plan["case_instances"][case_id]["path"]
                )
                packet = json.loads(
                    packet_path.read_text(encoding="utf-8")
                )
                self.assertEqual(
                    packet["schema"], "ijoc_case_instance_packet_v1"
                )
                self.assertEqual(len(packet["artifacts"]), expected_children)
                for child in packet["artifacts"]:
                    child_path = packet_path.parent / child["path"]
                    self.assertTrue(child_path.is_file())
                    self.assertEqual(
                        hashlib.sha256(child_path.read_bytes()).hexdigest(),
                        child["sha256"],
                    )
            (root / "motsp-00.calibration-reference.json").write_text(
                "mutated\n", encoding="utf-8"
            )
            self.assertEqual(
                audit_ijoc_competitive_study(
                    frozen.study_path
                ).submission_preflight_gate,
                "PASS",
            )
            plan_bytes = frozen.execution_plan_path.read_bytes()
            frozen.execution_plan_path.write_bytes(plan_bytes + b" ")
            with self.assertRaisesRegex(ValueError, "[Ff]reeze receipt"):
                run_cold_process_matrix(
                    frozen.study_path,
                    frozen.execution_plan_path,
                    root / "must-not-run",
                    timeout_seconds=5,
                    plumbing_max_rows=1,
                )
            frozen.execution_plan_path.write_bytes(plan_bytes)
            study = json.loads(
                frozen.study_path.read_text(encoding="utf-8")
            )
            reproducibility = json.loads(
                (
                    frozen.study_path.parent
                    / study["artifact_release"]["path"]
                ).read_text(encoding="utf-8")
            )
            bound_source = (
                frozen.study_path.parent
                / reproducibility["source_archive"]["path"]
            )
            bound_source.write_bytes(bound_source.read_bytes() + b"x")
            with self.assertRaises(ValueError):
                audit_ijoc_competitive_study(frozen.study_path)

    def test_plumbing_prefix_is_resumable_and_never_becomes_formal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self._freeze(root)
            results = root / "results"
            with patch.dict(os.environ, {"PYTHONPATH": ""}, clear=False), patch(
                "mo_nco.pareto_ijoc_cold_matrix._sample_process_tree_rss",
                return_value=123456,
            ):
                summary = run_cold_process_matrix(
                    frozen.study_path,
                    frozen.execution_plan_path,
                    results,
                    timeout_seconds=5,
                    plumbing_max_rows=2,
                    sample_period_seconds=0.01,
                )
                resumed = run_cold_process_matrix(
                    frozen.study_path,
                    frozen.execution_plan_path,
                    results,
                    timeout_seconds=5,
                    resume=True,
                    plumbing_max_rows=2,
                    sample_period_seconds=0.01,
                )
            self.assertEqual(summary.success_count, 2)
            self.assertEqual(resumed.success_count, 2)
            self.assertEqual(summary.execution_scope, "plumbing_only")
            self.assertEqual(summary.formal_evidence_status, "NOT_RUN")
            attempts = list(results.glob("runs/*/attempts/*"))
            self.assertEqual(len(attempts), 2)
            audit = audit_ijoc_post_run(
                frozen.study_path, frozen.execution_plan_path, results
            )
            self.assertEqual(audit.formal_matched_matrix_gate, "FAIL")
            self.assertEqual(audit.evidence_status, "NOT_RUN")
            self.assertEqual(audit.missing_run_count, 3598)

    def test_stratified_plumbing_covers_every_family_algorithm_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self._freeze(root)
            results = root / "stratified-results"
            with patch.dict(os.environ, {"PYTHONPATH": ""}, clear=False), patch(
                "mo_nco.pareto_ijoc_cold_matrix._sample_process_tree_rss",
                return_value=123456,
            ):
                summary = run_cold_process_matrix(
                    frozen.study_path,
                    frozen.execution_plan_path,
                    results,
                    timeout_seconds=5,
                    plumbing_stratified=True,
                    sample_period_seconds=0.01,
                    workers=4,
                )
                resumed = run_cold_process_matrix(
                    frozen.study_path,
                    frozen.execution_plan_path,
                    results,
                    timeout_seconds=5,
                    resume=True,
                    plumbing_stratified=True,
                    sample_period_seconds=0.01,
                    workers=4,
                )
            self.assertEqual(summary.selected_run_count, 2 * 4 * 3)
            self.assertEqual(summary.success_count, 24)
            self.assertEqual(resumed.success_count, 24)
            self.assertEqual(summary.execution_scope, "plumbing_only")
            self.assertEqual(summary.formal_evidence_status, "NOT_RUN")
            invocation = json.loads(
                (results / "matrix_invocation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                invocation["selection"]["coverage"],
                "each_family_algorithm_budget",
            )
            study = json.loads(
                frozen.study_path.read_text(encoding="utf-8")
            )
            family_by_case = {
                case_id: family["id"]
                for family in study["problem_families"]
                for case_id in family["cases"]
            }
            observed = set()
            for terminal in results.glob("runs/*/terminal_receipt.json"):
                receipt = json.loads(terminal.read_text(encoding="utf-8"))
                key = receipt["run_key"]
                observed.add(
                    (
                        family_by_case[key["case_id"]],
                        key["algorithm"],
                        key["budget"],
                    )
                )
                self.assertEqual(receipt["formal_evidence_status"], "NOT_RUN")
            expected = {
                (family["id"], algorithm, budget)
                for family in study["problem_families"]
                for algorithm in family["algorithms"]
                for budget in (10, 20, 30)
            }
            self.assertEqual(observed, expected)
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                run_cold_process_matrix(
                    frozen.study_path,
                    frozen.execution_plan_path,
                    root / "must-not-run",
                    timeout_seconds=5,
                    plumbing_max_rows=1,
                    plumbing_stratified=True,
                )

    def test_parallel_workers_keep_cold_rows_atomic_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self._freeze(root, behavior="slow_success")
            results = root / "parallel-results"
            with patch(
                "mo_nco.pareto_ijoc_cold_matrix._sample_process_tree_rss",
                return_value=123456,
            ):
                summary = run_cold_process_matrix(
                    frozen.study_path,
                    frozen.execution_plan_path,
                    results,
                    timeout_seconds=5,
                    plumbing_max_rows=4,
                    sample_period_seconds=0.01,
                    workers=4,
                )
                resumed = run_cold_process_matrix(
                    frozen.study_path,
                    frozen.execution_plan_path,
                    results,
                    timeout_seconds=5,
                    resume=True,
                    plumbing_max_rows=4,
                    sample_period_seconds=0.01,
                    workers=4,
                )
            self.assertEqual(summary.success_count, 4)
            self.assertEqual(resumed.success_count, 4)
            invocation = json.loads(
                (results / "matrix_invocation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(invocation["workers"], 4)
            terminals = sorted(
                results.glob("runs/*/terminal_receipt.json")
            )
            self.assertEqual(len(terminals), 4)
            receipts = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in terminals
            ]
            self.assertTrue(
                all(receipt["status"] == "SUCCESS" for receipt in receipts)
            )
            started = [
                datetime.fromisoformat(
                    receipt["algorithm_process"]["started_utc"].replace(
                        "Z", "+00:00"
                    )
                )
                for receipt in receipts
            ]
            finished = [
                datetime.fromisoformat(
                    receipt["algorithm_process"]["finished_utc"].replace(
                        "Z", "+00:00"
                    )
                )
                for receipt in receipts
            ]
            self.assertLess(max(started), min(finished))
            self.assertEqual(
                len(list(results.glob("runs/*/attempts/*"))), 4
            )
            with self.assertRaisesRegex(ValueError, "Resume invocation"):
                run_cold_process_matrix(
                    frozen.study_path,
                    frozen.execution_plan_path,
                    results,
                    timeout_seconds=5,
                    resume=True,
                    plumbing_max_rows=4,
                    sample_period_seconds=0.01,
                    workers=2,
                )

    def test_parallel_interrupt_leaves_atomic_terminal_receipts_for_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self._freeze(root)
            results = root / "interrupted-results"
            interrupted_process = {
                "argv": ["unit-test-interrupt"],
                "started_utc": "2026-07-31T00:00:00.000000Z",
                "finished_utc": "2026-07-31T00:00:00.000001Z",
                "wall_time_seconds": 0.000001,
                "exit_code": -1,
                "timed_out": False,
                "interrupted": True,
                "spawn_error": None,
                "sampled_peak_process_tree_rss_bytes": 1,
                "resource_measurement_status": "PASS",
                "stdout": {"path": "unused", "sha256": "0" * 64, "bytes": 0},
                "stderr": {"path": "unused", "sha256": "0" * 64, "bytes": 0},
            }
            with patch(
                "mo_nco.pareto_ijoc_cold_matrix._execute_process",
                return_value=interrupted_process,
            ), self.assertRaises(KeyboardInterrupt):
                run_cold_process_matrix(
                    frozen.study_path,
                    frozen.execution_plan_path,
                    results,
                    timeout_seconds=5,
                    plumbing_max_rows=4,
                    workers=4,
                )
            terminals = sorted(
                results.glob("runs/*/terminal_receipt.json")
            )
            self.assertGreaterEqual(len(terminals), 1)
            self.assertTrue(
                all(
                    json.loads(path.read_text(encoding="utf-8"))["status"]
                    == "INTERRUPTED"
                    for path in terminals
                )
            )
            self.assertEqual(list(results.rglob("*.tmp")), [])
            attempts_before = {
                terminal.parent: len(
                    list((terminal.parent / "attempts").iterdir())
                )
                for terminal in terminals
            }
            with self.assertRaises(KeyboardInterrupt):
                run_cold_process_matrix(
                    frozen.study_path,
                    frozen.execution_plan_path,
                    results,
                    timeout_seconds=5,
                    resume=True,
                    plumbing_max_rows=4,
                    workers=4,
                )
            for run_directory, count in attempts_before.items():
                self.assertEqual(
                    len(list((run_directory / "attempts").iterdir())),
                    count,
                )

    def test_case_packet_seams_fail_closed(self) -> None:
        mutations = ("escape", "hash_drift", "tail_byte_overlap")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                request_path = _make_request(root)
                packet_path = root / "motsp-00.packet.json"
                packet = json.loads(packet_path.read_text(encoding="utf-8"))
                if mutation == "escape":
                    packet["artifacts"][0]["path"] = "../escape.tsp"
                    expected = "unsafe|escapes"
                elif mutation == "hash_drift":
                    packet["artifacts"][0]["sha256"] = "0" * 64
                    expected = "SHA-256 mismatch"
                else:
                    tail_receipt = json.loads(
                        (root / "calibration-suite-receipt.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    tail_artifact = next(
                        artifact
                        for artifact in tail_receipt["instance_artifacts"]
                        if artifact["family"] == "MOTSP"
                    )
                    formal_child = root / packet["artifacts"][0]["path"]
                    formal_child.write_bytes(
                        (root / tail_artifact["path"]).read_bytes()
                    )
                    packet["artifacts"][0]["sha256"] = hashlib.sha256(
                        formal_child.read_bytes()
                    ).hexdigest()
                    expected = "reuses calibration instance bytes"
                _write_json(packet_path, packet)
                with self.assertRaisesRegex(ValueError, expected):
                    freeze_ijoc_study(request_path, root / "must-not-freeze")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self._freeze(root)
            plan = json.loads(
                frozen.execution_plan_path.read_text(encoding="utf-8")
            )
            packet_path = (
                frozen.execution_plan_path.parent
                / plan["case_instances"]["motsp-00"]["path"]
            )
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
            child = packet_path.parent / packet["artifacts"][0]["path"]
            child.write_bytes(child.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "child hash mismatch"):
                run_cold_process_matrix(
                    frozen.study_path,
                    frozen.execution_plan_path,
                    root / "must-not-run",
                    timeout_seconds=5,
                    plumbing_max_rows=1,
                )

    def test_runtime_archive_dependency_and_analysis_seams_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path = _make_request(root)
            with tarfile.open(root / "source.tar.gz", mode="w:gz") as archive:
                body = b"escape"
                member = tarfile.TarInfo("../escape.py")
                member.size = len(body)
                archive.addfile(member, io.BytesIO(body))
            with self.assertRaisesRegex(ValueError, "unsafe"):
                freeze_ijoc_study(request_path, root / "unsafe-runtime")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path = _make_request(root)
            (root / "requirements-lock.txt").write_text(
                "pip==0.0-impossible\n", encoding="utf-8"
            )
            frozen = freeze_ijoc_study(request_path, root / "frozen")
            with self.assertRaisesRegex(ValueError, "version mismatch"):
                run_cold_process_matrix(
                    frozen.study_path,
                    frozen.execution_plan_path,
                    root / "must-not-run",
                    timeout_seconds=5,
                    plumbing_max_rows=1,
                )
            self.assertFalse((root / "must-not-run").exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path = _make_request(root)
            plan_path = root / "formal-analysis-plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["evaluation_budgets"] = [10, 20, 40]
            plan["primary_budget"] = 40
            _write_json(plan_path, plan)
            with self.assertRaisesRegex(ValueError, "analysis plan.*differs"):
                freeze_ijoc_study(request_path, root / "bad-analysis")

    def test_reference_precommit_completion_chain_is_acyclic_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self._freeze(root)
            plan = json.loads(
                frozen.execution_plan_path.read_text(encoding="utf-8")
            )
            self.assertNotEqual(
                plan["tail_calibration_suite_receipt"]["sha256"],
                plan["reference_calibration_precommit"]["sha256"],
            )
            completion_path = (
                frozen.execution_plan_path.parent
                / plan["reference_calibration_completion_receipt"]["path"]
            )
            completion = json.loads(
                completion_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                completion["reference_calibration_precommit_sha256"],
                plan["reference_calibration_precommit"]["sha256"],
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path = _make_request(root)
            request = json.loads(request_path.read_text(encoding="utf-8"))
            request["seeds"] = list(range(9)) + [100]
            _write_json(request_path, request)
            analysis_path = root / "formal-analysis-plan.json"
            analysis = json.loads(
                analysis_path.read_text(encoding="utf-8")
            )
            analysis["formal_seeds"] = list(range(9)) + [100]
            _write_json(analysis_path, analysis)
            with self.assertRaisesRegex(ValueError, "formal seeds overlap"):
                freeze_ijoc_study(request_path, root / "seed-overlap")

    def test_nonzero_process_and_timeout_each_leave_terminal_evidence(self) -> None:
        for mode in ("failure", "timeout"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                frozen = self._freeze(root, behavior=mode)
                timeout = 5 if mode == "failure" else 0.05
                with patch(
                    "mo_nco.pareto_ijoc_cold_matrix._sample_process_tree_rss",
                    return_value=123456,
                ):
                    summary = run_cold_process_matrix(
                        frozen.study_path,
                        frozen.execution_plan_path,
                        root / "results",
                        timeout_seconds=timeout,
                        plumbing_max_rows=1,
                        sample_period_seconds=0.01,
                    )
                self.assertEqual(summary.failure_count, 1)
                terminal = next(
                    (root / "results").glob("runs/*/terminal_receipt.json")
                )
                receipt = json.loads(terminal.read_text(encoding="utf-8"))
                self.assertEqual(
                    receipt["status"],
                    "PROCESS_FAILURE" if mode == "failure" else "TIMEOUT",
                )
                self.assertEqual(receipt["formal_evidence_status"], "NOT_RUN")
                self.assertTrue(
                    (terminal.parent / receipt["algorithm_process"]["stdout"]["path"])
                    .is_file()
                )
                self.assertTrue(
                    (terminal.parent / receipt["algorithm_process"]["stderr"]["path"])
                    .is_file()
                )

    def test_residual_process_tree_timeout_is_bounded_and_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self._freeze(root)
            results = root / "residual-tree-timeout-results"
            process = MagicMock()
            process.pid = 424242
            process.poll.return_value = None
            process.wait.side_effect = subprocess.TimeoutExpired(
                cmd=["residual-process-tree"], timeout=10
            )
            with patch(
                "mo_nco.pareto_ijoc_cold_matrix.subprocess.Popen",
                return_value=process,
            ), patch(
                "mo_nco.pareto_ijoc_cold_matrix._terminate_process_tree"
            ) as terminate_tree, patch(
                "mo_nco.pareto_ijoc_cold_matrix._sample_process_tree_rss",
                return_value=123456,
            ):
                summary = run_cold_process_matrix(
                    frozen.study_path,
                    frozen.execution_plan_path,
                    results,
                    timeout_seconds=0.000001,
                    plumbing_max_rows=1,
                    sample_period_seconds=0.000001,
                )
            self.assertEqual(summary.failure_count, 1)
            terminal = next(results.glob("runs/*/terminal_receipt.json"))
            terminal_bytes = terminal.read_bytes()
            receipt = json.loads(terminal_bytes)
            self.assertEqual(receipt["status"], "TIMEOUT")
            self.assertTrue(receipt["algorithm_process"]["timed_out"])
            self.assertIsNone(receipt["algorithm_process"]["exit_code"])
            self.assertGreaterEqual(terminate_tree.call_count, 2)
            process.kill.assert_called()
            self.assertGreaterEqual(process.wait.call_count, 2)
            self.assertTrue(
                all(
                    call.kwargs.get("timeout") is not None
                    for call in process.wait.call_args_list
                )
            )
            resumed = run_cold_process_matrix(
                frozen.study_path,
                frozen.execution_plan_path,
                results,
                timeout_seconds=0.000001,
                resume=True,
                plumbing_max_rows=1,
                sample_period_seconds=0.000001,
            )
            self.assertEqual(resumed.failure_count, 1)
            self.assertEqual(terminal.read_bytes(), terminal_bytes)

    def test_parallel_residual_process_trees_share_bounded_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self._freeze(root)
            results = root / "parallel-residual-tree-timeouts"
            processes = []

            def residual_process(*args, **kwargs):
                del args, kwargs
                process = MagicMock()
                process.pid = 500000 + len(processes)
                process.poll.return_value = None
                process.wait.side_effect = subprocess.TimeoutExpired(
                    cmd=["parallel-residual-process-tree"], timeout=10
                )
                processes.append(process)
                return process

            with patch(
                "mo_nco.pareto_ijoc_cold_matrix.subprocess.Popen",
                side_effect=residual_process,
            ), patch(
                "mo_nco.pareto_ijoc_cold_matrix._terminate_process_tree"
            ) as terminate_tree, patch(
                "mo_nco.pareto_ijoc_cold_matrix._sample_process_tree_rss",
                return_value=123456,
            ):
                summary = run_cold_process_matrix(
                    frozen.study_path,
                    frozen.execution_plan_path,
                    results,
                    timeout_seconds=0.000001,
                    plumbing_max_rows=2,
                    sample_period_seconds=0.000001,
                    workers=2,
                )
            self.assertEqual(summary.failure_count, 2)
            receipts = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in results.glob("runs/*/terminal_receipt.json")
            ]
            self.assertEqual(len(receipts), 2)
            self.assertTrue(
                all(receipt["status"] == "TIMEOUT" for receipt in receipts)
            )
            self.assertEqual(len(processes), 2)
            self.assertGreaterEqual(terminate_tree.call_count, 4)
            for process in processes:
                process.kill.assert_called()
                self.assertGreaterEqual(process.wait.call_count, 2)
                self.assertTrue(
                    all(
                        call.kwargs.get("timeout") is not None
                        for call in process.wait.call_args_list
                    )
                )

    def test_spawn_error_remains_explicit_terminal_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self._freeze(root)
            results = root / "spawn-error-results"
            with patch(
                "mo_nco.pareto_ijoc_cold_matrix.subprocess.Popen",
                side_effect=OSError("spawn-denied-sentinel"),
            ):
                summary = run_cold_process_matrix(
                    frozen.study_path,
                    frozen.execution_plan_path,
                    results,
                    timeout_seconds=5,
                    plumbing_max_rows=1,
                )
            self.assertEqual(summary.failure_count, 1)
            terminal = next(results.glob("runs/*/terminal_receipt.json"))
            receipt = json.loads(terminal.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "PROCESS_SPAWN_FAILURE")
            self.assertIn("spawn-denied-sentinel", receipt["reason"])
            self.assertIn(
                "spawn-denied-sentinel",
                receipt["algorithm_process"]["spawn_error"],
            )
            self.assertEqual(
                receipt["algorithm_process"]["termination"]["status"],
                "NOT_STARTED",
            )

    def test_cleanup_exceptions_are_structured_without_masking_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self._freeze(root)
            results = root / "cleanup-exception-results"
            process = MagicMock()
            process.pid = 616161
            process.poll.return_value = None
            process.wait.side_effect = subprocess.TimeoutExpired(
                cmd=["cleanup-exception-process"], timeout=10
            )
            process.kill.side_effect = RuntimeError("root-kill-sentinel")
            with patch(
                "mo_nco.pareto_ijoc_cold_matrix.subprocess.Popen",
                return_value=process,
            ), patch(
                "mo_nco.pareto_ijoc_cold_matrix._terminate_process_tree",
                side_effect=(
                    RuntimeError("tree-round-one-sentinel"),
                    RuntimeError("tree-round-two-sentinel"),
                ),
            ), patch(
                "mo_nco.pareto_ijoc_cold_matrix._sample_process_tree_rss",
                return_value=123456,
            ):
                summary = run_cold_process_matrix(
                    frozen.study_path,
                    frozen.execution_plan_path,
                    results,
                    timeout_seconds=0.000001,
                    plumbing_max_rows=1,
                    sample_period_seconds=0.000001,
                )
            self.assertEqual(summary.failure_count, 1)
            terminal = next(results.glob("runs/*/terminal_receipt.json"))
            receipt = json.loads(terminal.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "TIMEOUT")
            cleanup = receipt["algorithm_process"]["termination"]
            self.assertEqual(cleanup["status"], "BOUNDED_CLEANUP_EXHAUSTED")
            errors = "\n".join(cleanup["errors"])
            for sentinel in (
                "tree-round-one-sentinel",
                "tree-round-two-sentinel",
                "root-kill-sentinel",
            ):
                self.assertIn(sentinel, errors)

    def test_postrun_cli_exposes_retry_and_resource_verdicts(self) -> None:
        from scripts import audit_ijoc_postrun

        result = SimpleNamespace(
            audit_path=Path("post_run_audit.json"),
            expected_run_count=3600,
            observed_unique_run_count=3600,
            valid_run_count=3600,
            missing_run_count=0,
            duplicate_run_count=0,
            invalid_run_count=0,
            retry_run_count=4,
            prior_attempt_count=4,
            formal_matched_matrix_gate="PASS",
            resource_efficiency_gate="NOT_ESTABLISHED",
            evidence_status="REPORTED_ARCHIVE_MATRIX_INTEGRITY_ESTABLISHED",
            submission_verdict="HOLD_PENDING_METRIC_AND_STATISTICAL_AUDIT",
        )
        stdout = io.StringIO()
        arguments = [
            "audit_ijoc_postrun.py",
            "--study",
            "study.json",
            "--execution-plan",
            "execution_plan.json",
            "--results-directory",
            "formal_results",
        ]
        with patch.object(
            audit_ijoc_postrun,
            "audit_ijoc_post_run",
            return_value=result,
        ), patch("sys.argv", arguments), redirect_stdout(stdout):
            audit_ijoc_postrun.main()
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["retry_run_count"], 4)
        self.assertEqual(payload["prior_attempt_count"], 4)
        self.assertEqual(
            payload["resource_efficiency_gate"],
            "NOT_ESTABLISHED",
        )

    def test_postrun_accepts_complete_bound_receipts_and_detects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self._freeze(root)
            results = root / "results"
            with patch(
                "mo_nco.pareto_ijoc_cold_matrix._sample_process_tree_rss",
                return_value=123456,
            ):
                run_cold_process_matrix(
                    frozen.study_path,
                    frozen.execution_plan_path,
                    results,
                    timeout_seconds=5,
                    plumbing_max_rows=2,
                    sample_period_seconds=0.01,
                )
            study, study_sha, full_configuration, config_sha, instances = (
                _study_components(frozen.study_path)
            )
            _, plan_sha, algorithms = _validate_plan(
                frozen.execution_plan_path,
                study_sha=study_sha,
                configuration_sha=config_sha,
                instances=instances,
            )
            rows = full_configuration["rows"][:2]
            invocation_path = results / "matrix_invocation.json"
            invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
            invocation["execution_scope"] = "formal_candidate"
            invocation["selection"] = {"kind": "all"}
            invocation["selected_run_count"] = 2
            invocation["expected_run_count"] = 2
            _write_json(invocation_path, invocation)
            for terminal in results.glob("runs/*/terminal_receipt.json"):
                receipt = json.loads(terminal.read_text(encoding="utf-8"))
                receipt["execution_scope"] = "formal_candidate"
                receipt["formal_evidence_status"] = "PENDING_POST_RUN_AUDIT"
                _write_json(terminal, receipt)
            mocked_components = (
                study,
                study_sha,
                {"schema": full_configuration["schema"], "rows": rows},
                config_sha,
                instances,
            )
            mocked_plan = (
                {},
                plan_sha,
                {"ijoc-pareto-smc": algorithms["ijoc-pareto-smc"]},
            )
            freeze_receipt_sha = hashlib.sha256(
                frozen.receipt_path.read_bytes()
            ).hexdigest()
            with patch(
                "mo_nco.pareto_ijoc_postrun._study_components",
                return_value=mocked_components,
            ), patch(
                "mo_nco.pareto_ijoc_postrun._validate_plan",
                return_value=mocked_plan,
            ), patch(
                "mo_nco.pareto_ijoc_postrun._validate_freeze_receipt",
                return_value=freeze_receipt_sha,
            ), patch(
                "mo_nco.pareto_ijoc_postrun._validate_runtime_source_manifest",
                return_value=root,
            ):
                passed = audit_ijoc_post_run(
                    frozen.study_path, frozen.execution_plan_path, results
                )
                self.assertEqual(passed.formal_matched_matrix_gate, "PASS")
                self.assertEqual(
                    passed.evidence_status,
                    "REPORTED_ARCHIVE_MATRIX_INTEGRITY_ESTABLISHED",
                )
                passed_payload = json.loads(
                    passed.audit_path.read_text(encoding="utf-8")
                )
                self.assertEqual(
                    passed_payload["schema"], "ijoc_post_run_audit_v2"
                )
                postrun_schema = json.loads(
                    (
                        Path(__file__).parents[1]
                        / "ijoc_submission_v20"
                        / "protocol"
                        / "schemas"
                        / "ijoc_post_run_audit_v2.schema.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(
                    set(passed_payload), set(postrun_schema["required"])
                )
                self.assertEqual(
                    passed_payload["quality_estimand_scope"],
                    "reported_archive_relative",
                )
                self.assertEqual(
                    passed_payload["gates"][
                        "reported_archive_witness_self_consistency_gate"
                    ],
                    "PASS",
                )
                self.assertEqual(
                    passed_payload["gates"][
                        "all_evaluated_trace_completeness_gate"
                    ],
                    "NOT_ESTABLISHED",
                )

                # A hard-stopped runner can leave an attempt directory without
                # a terminal receipt.  Resuming must preserve quality auditing
                # for an identical-input, no-result prior attempt while making
                # the cold resource claim fail closed.
                terminal = next(results.glob("runs/*/terminal_receipt.json"))
                receipt = json.loads(terminal.read_text(encoding="utf-8"))
                run_directory = terminal.parent
                first_attempt = run_directory / "attempts" / "000001"
                second_attempt = run_directory / "attempts" / "000002"
                first_attempt.rename(second_attempt)
                first_attempt.mkdir()
                (first_attempt / "input.json").write_bytes(
                    (second_attempt / "input.json").read_bytes()
                )
                (first_attempt / "algorithm.stdout").write_bytes(b"")
                (first_attempt / "algorithm.stderr").write_bytes(b"")
                def remap_attempt(value: object) -> object:
                    if isinstance(value, dict):
                        return {
                            key: remap_attempt(item)
                            for key, item in value.items()
                        }
                    if isinstance(value, list):
                        return [remap_attempt(item) for item in value]
                    if isinstance(value, str):
                        return value.replace(
                            str(first_attempt), str(second_attempt)
                        ).replace(
                            "attempts/000001/", "attempts/000002/"
                        )
                    return value

                receipt = remap_attempt(receipt)
                assert isinstance(receipt, dict)
                receipt["attempt_number"] = 2
                _write_json(terminal, receipt)

                retried = audit_ijoc_post_run(
                    frozen.study_path, frozen.execution_plan_path, results
                )
                retried_payload = json.loads(
                    retried.audit_path.read_text(encoding="utf-8")
                )
                self.assertEqual(
                    retried.formal_matched_matrix_gate,
                    "PASS",
                    retried_payload,
                )
                self.assertEqual(retried.retry_run_count, 1)
                self.assertEqual(
                    retried.resource_efficiency_gate, "NOT_ESTABLISHED"
                )
                self.assertEqual(
                    retried_payload["gates"]["retry_quality_selection_gate"],
                    "PASS",
                )
                self.assertEqual(
                    retried_payload["gates"]["resource_efficiency_gate"],
                    "NOT_ESTABLISHED",
                )
                self.assertEqual(
                    retried_payload["attempt_audit"]["prior_attempt_count"], 1
                )
                prior = retried_payload["attempt_audit"]["histories"][0][
                    "prior_attempts"
                ][0]
                self.assertEqual(
                    prior["termination_reason"], "UNKNOWN_UNRECORDED"
                )
                self.assertTrue(prior["input_matches_terminal"])
                self.assertEqual(
                    prior["result_artifact_status"], "NO_RESULT_ARTIFACT"
                )

                (first_attempt / "algorithm.stderr").write_bytes(
                    b"partial failure output"
                )
                logged_retry = audit_ijoc_post_run(
                    frozen.study_path, frozen.execution_plan_path, results
                )
                logged_retry_payload = json.loads(
                    logged_retry.audit_path.read_text(encoding="utf-8")
                )
                self.assertEqual(
                    logged_retry.formal_matched_matrix_gate, "FAIL"
                )
                self.assertEqual(
                    logged_retry_payload["gates"][
                        "retry_quality_selection_gate"
                    ],
                    "FAIL",
                )
                (first_attempt / "algorithm.stderr").write_bytes(b"")

                archive = next(results.glob("runs/*/attempts/*/archive.json"))
                archive.write_bytes(archive.read_bytes() + b"tamper")
                failed = audit_ijoc_post_run(
                    frozen.study_path, frozen.execution_plan_path, results
                )
            self.assertEqual(failed.formal_matched_matrix_gate, "FAIL")
            self.assertGreaterEqual(failed.invalid_run_count, 1)


if __name__ == "__main__":
    unittest.main()

