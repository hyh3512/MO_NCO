from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "run_v21e3r1_recovery_bound_same_implementation_branch_replay_coverage.py"
)
CONTINUATION_HELPER_PATH = (
    ROOT
    / "artifacts"
    / "v21e3r1_v8_work_20260822"
    / "run_frozen_diagnostic_metric_timeout_recovery_continuation.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "v21e3r1_recovery_bound_coverage", RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _bound(core: dict[str, object], digest_field: str) -> dict[str, object]:
    return {
        **core,
        digest_field: hashlib.sha256(_canonical_bytes(core)).hexdigest(),
    }


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hold_fields() -> dict[str, object]:
    return {
        "runtime_authority": False,
        "scientific_authority": False,
        "selection_authority": False,
        "confirmation_authority": False,
        "formal_study_authority": False,
        "publication_status": "IJOC_HOLD",
    }


def _file_manifest(root: Path, paths: list[Path]) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix())
    ]


def _external_fixture(tmp_path: Path, runner):
    diagnostic = tmp_path / "diagnostic"
    diagnostic.mkdir()
    helper = tmp_path / "external-helper.py"
    helper.write_text("# frozen external helper\n", encoding="utf-8")
    helper_sha = runner.sha256_file(helper)
    plan_sha = "1" * 64
    source_sha = "2" * 64
    frozen_runner_sha = "3" * 64
    completed: list[dict[str, object]] = []
    row_ids: list[str] = []
    for ordinal in range(463, 505):
        row_id = f"synthetic-row-{ordinal:04d}"
        row_ids.append(row_id)
        attempt = diagnostic / "attempts" / row_id / "attempt-0001"
        spec_path = attempt / "worker.spec.json"
        spec = {"row_id": row_id, "ordinal": ordinal}
        spec_sha = _write_json(spec_path, spec)
        marker_path = diagnostic / "completed" / f"{row_id}.json"
        marker_sha = _write_json(
            marker_path,
            {"row_id": row_id, "attempt_directory": attempt.relative_to(diagnostic).as_posix()},
        )
        completed.append(
            {
                "ordinal": ordinal,
                "row_id": row_id,
                "path": marker_path.relative_to(diagnostic).as_posix(),
                "sha256": marker_sha,
                "attempt_directory": attempt.relative_to(diagnostic).as_posix(),
                "worker_spec_path": spec_path.relative_to(diagnostic).as_posix(),
                "worker_spec_sha256": spec_sha,
                "worker_spec_payload_sha256": hashlib.sha256(
                    _canonical_bytes(spec)
                ).hexdigest(),
            }
        )
    handoff_path = diagnostic / (
        "external-scheduling.v21e3-motsp-development-n500-s01."
        "main-driver-handoff.receipt.json"
    )
    handoff = _bound(
        {
            "schema": "v21e3r1_external_main_driver_handoff_receipt_v1",
            "status": "PASS_EXTERNAL_STOP_RECORDED_AND_NO_ORIGINAL_PROCESS_OBSERVED",
            "scope": "AUDIT_RECORD_ONLY_EXECUTION_MUST_RESCAN",
            "issued_at_utc": "synthetic",
            "output_root": diagnostic.as_posix(),
            "stopped_main_pid": 1,
            "stopped_main_command_line": "synthetic",
            "stopped_main_command_sha256": "4" * 64,
            "helper_sha256": helper_sha,
            "plan_sha256": plan_sha,
            "source_snapshot_sha256": source_sha,
            "interpreter_identity": {"status": "PASS"},
            "environment_receipt": {"status": "PASS"},
            "completed_prefix": {"completed_marker_count": 443},
            "process_scan": {"driver_processes": [], "worker_processes": []},
            "receipt_is_audit_record_not_trusted_liveness": True,
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
        },
        "receipt_payload_sha256",
    )
    handoff_sha = _write_json(handoff_path, handoff)
    claim_path = diagnostic / (
        "external-scheduling.v21e3-motsp-development-n500-s01."
        "helper-instance.claim.json"
    )
    worker_manifest = [
        {
            "ordinal": item["ordinal"],
            "row_id": item["row_id"],
            "worker_spec_payload_sha256": item["worker_spec_payload_sha256"],
        }
        for item in completed
    ]
    claim = _bound(
        {
            "schema": "v21e3r1_external_scheduling_helper_instance_claim_v2",
            "status": "SEALED_HELPER_INSTANCE_CLAIM_NOT_A_MAIN_RUNNER_LOCK",
            "scope": "EXTERNAL_SCHEDULING_ONLY_FROZEN_DEVELOPMENT_ROWS",
            "process_id": 2,
            "target_case_id": "v21e3-motsp-development-n500-s01",
            "target_row_count": 42,
            "target_row_ids": row_ids,
            "worker_spec_payload_manifest": worker_manifest,
            "worker_spec_payload_manifest_sha256": hashlib.sha256(
                _canonical_bytes(worker_manifest)
            ).hexdigest(),
            "jobs": 4,
            "helper_sha256": helper_sha,
            "frozen_runner_sha256": frozen_runner_sha,
            "handoff_receipt_path": handoff_path.name,
            "handoff_receipt_sha256": handoff_sha,
            "handoff_receipt_payload_sha256": handoff["receipt_payload_sha256"],
            "plan_sha256": plan_sha,
            "source_snapshot_sha256": source_sha,
            "interpreter_identity": {"status": "PASS"},
            "environment_receipt": {"status": "PASS"},
            "operational_quiescence_depends_on_external_stop_and_repeated_process_scan": True,
            "original_main_runner_honors_this_claim": False,
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
        },
        "claim_payload_sha256",
    )
    claim_sha = _write_json(claim_path, claim)
    receipt_path = diagnostic / (
        "external-scheduling.v21e3-motsp-development-n500-s01.receipt.json"
    )
    seal_path = diagnostic / (
        "external-scheduling.v21e3-motsp-development-n500-s01.receipt.seal.json"
    )
    receipt = _bound(
        {
            "schema": "v21e3r1_external_scheduling_only_receipt_v2",
            "status": "PASS_EXTERNAL_SCHEDULING_ONLY_TARGET_42",
            "scope": "EXTERNAL_SCHEDULING_ONLY_FROZEN_DEVELOPMENT_ROWS",
            "target_case_id": "v21e3-motsp-development-n500-s01",
            "target_row_count": 42,
            "completed_marker_count": 42,
            "completed_markers": completed,
            "full_plan_row_count": 504,
            "jobs": 4,
            "scheduling_policy": "THREAD_POOL_EXECUTOR_MAX_WORKERS_4",
            "worker_execution": "DELEGATED_TO_FROZEN_RUNNER_RUN_CHILD_AND_WORKER",
            "completed_marker_generation": "HELPER_MATERIALIZED_ORIGINAL_FORMAT_ONLY_AFTER_FROZEN_WORKER_RESULT_ARTIFACT_VALIDATION",
            "completed_marker_verification": "FROZEN_RUNNER_COMPLETED_PAYLOAD_REVALIDATED_PER_ROW_AND_FINAL",
            "plan_path": "diagnostic.plan.json",
            "plan_sha256": plan_sha,
            "source_snapshot_sha256": source_sha,
            "frozen_runner_path": "frozen.py",
            "frozen_runner_sha256": frozen_runner_sha,
            "helper_path": helper.as_posix(),
            "helper_sha256": helper_sha,
            "helper_instance_claim_path": claim_path.name,
            "helper_instance_claim_sha256": claim_sha,
            "handoff_receipt_path": handoff_path.name,
            "handoff_receipt_sha256": handoff_sha,
            "handoff_receipt_payload_sha256": handoff["receipt_payload_sha256"],
            "receipt_seal_path": seal_path.name,
            "interpreter_identity": {"status": "PASS"},
            "environment_receipt": {"status": "PASS"},
            "original_main_runner_honors_helper_instance_claim": False,
            "original_runner_resume_after_helper_success_only": True,
            "original_runner_resume_required": True,
            "original_runner_or_algorithm_sources_modified": False,
            "case_generation_performed": False,
            "generated_case_count": 0,
            "implementation_independence": False,
            "algorithm_execution_independence": False,
            "scientific_independence": False,
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
            "confirmation_authority": False,
            "formal_study_authority": False,
            "publication_status": "IJOC_HOLD",
        },
        "receipt_payload_sha256",
    )
    receipt_sha = _write_json(receipt_path, receipt)
    seal = _bound(
        {
            "schema": "v21e3r1_external_scheduling_receipt_file_seal_v1",
            "status": "PASS_SUCCESS_RECEIPT_FILE_DIGEST_SEALED",
            "receipt_path": receipt_path.name,
            "receipt_sha256": receipt_sha,
            "receipt_payload_sha256": receipt["receipt_payload_sha256"],
            "helper_instance_claim_sha256": claim_sha,
            "handoff_receipt_sha256": handoff_sha,
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
        },
        "seal_payload_sha256",
    )
    _write_json(seal_path, seal)
    return {
        "diagnostic": diagnostic,
        "helper": helper,
        "claim": claim_path,
        "handoff": handoff_path,
        "receipt": receipt_path,
        "seal": seal_path,
        "plan_sha": plan_sha,
        "source_sha": source_sha,
        "frozen_runner_sha": frozen_runner_sha,
    }


def _old_recovery_fixture(tmp_path: Path, runner):
    fixture = _external_fixture(tmp_path, runner)
    diagnostic: Path = fixture["diagnostic"]
    helper = tmp_path / "old-recovery-helper.py"
    helper.write_text("# frozen failed recovery helper\n", encoding="utf-8")
    helper_sha = runner.sha256_file(helper)
    metric_sha = "5" * 64
    # Production uses the same fixed process-guard/external-helper source.
    process_guard_sha = runner.sha256_file(fixture["helper"])
    target_rows = [f"synthetic-row-{ordinal:04d}" for ordinal in range(446, 463)]
    preexisting_root = (
        diagnostic
        / "attempts"
        / target_rows[0]
        / "attempt-0001"
    )
    preexisting_paths = [
        preexisting_root / "failure.receipt.json",
        preexisting_root / "terminal.receipt.json",
        preexisting_root / "trace.sqlite3",
        preexisting_root / "worker.spec.json",
    ]
    for path in preexisting_paths:
        _write_json(path, {"status": "FAILED_300_SECONDS", "path": path.name})
    preexisting_manifest = _file_manifest(diagnostic, preexisting_paths)
    worker_manifest: list[dict[str, object]] = []
    for ordinal, row_id in zip(range(446, 463), target_rows, strict=True):
        spec_payload = {"ordinal": ordinal, "row_id": row_id}
        worker_manifest.append(
            {
                "expected_attempt_number": 2 if ordinal == 446 else 1,
                "ordinal": ordinal,
                "row_id": row_id,
                "worker_spec_payload_sha256": hashlib.sha256(
                    _canonical_bytes(spec_payload)
                ).hexdigest(),
            }
        )
    external_receipt_sha = runner.sha256_file(fixture["receipt"])
    external_seal_sha = runner.sha256_file(fixture["seal"])
    claim_path = diagnostic / (
        "metric-timeout-recovery.v21e3-motsp-development-n500-s00."
        "helper-instance.claim.json"
    )
    claim = _bound(
        {
            "schema": "v21e3r1_metric_timeout_recovery_helper_instance_claim_v1",
            "status": "SEALED_HELPER_INSTANCE_CLAIM_NOT_A_MAIN_RUNNER_LOCK",
            "scope": "EXACT_ORDINALS_446_462_FROZEN_DEVELOPMENT_RECOVERY_ONLY",
            "process_id": 3,
            "target_case_id": "v21e3-motsp-development-n500-s00",
            "target_ordinals": list(range(446, 463)),
            "target_row_count": 17,
            "target_row_ids": target_rows,
            "worker_spec_payload_manifest": worker_manifest,
            "worker_spec_payload_manifest_sha256": hashlib.sha256(
                _canonical_bytes(worker_manifest)
            ).hexdigest(),
            "jobs": 4,
            "original_metric_timeout_seconds": 300,
            "operational_metric_timeout_seconds": 1200,
            "outer_row_timeout_seconds": 2400,
            "outer_timeout_margin_seconds": 1200,
            "recovery_semantics": "SAME_ALGORITHM_AND_METRIC_CODE_OPERATIONAL_TIMEOUT_OVERRIDE_ONLY",
            "fresh_full_algorithm_reruns_required": True,
            "metric_only_replay": False,
            "preexisting_failed_trace_reuse_authorized": False,
            "non_target_completed_marker_count": 487,
            "non_target_completed_marker_manifest_sha256": "7" * 64,
            "preexisting_failed_attempt_manifest": preexisting_manifest,
            "preexisting_failed_attempt_manifest_sha256": hashlib.sha256(
                _canonical_bytes(preexisting_manifest)
            ).hexdigest(),
            "preclaim_process_scan": {"matching_process_count": 0},
            "helper_sha256": helper_sha,
            "frozen_runner_sha256": fixture["frozen_runner_sha"],
            "independent_metric_source_sha256": metric_sha,
            "process_guard_sha256": process_guard_sha,
            "plan_sha256": fixture["plan_sha"],
            "source_snapshot_sha256": fixture["source_sha"],
            "upstream_scheduling_receipt_sha256": external_receipt_sha,
            "upstream_scheduling_seal_sha256": external_seal_sha,
            "interpreter_identity": {"status": "PASS"},
            "environment_receipt": {"status": "PASS"},
            "original_main_runner_honors_this_claim": False,
            "automatic_resume_authorized": False,
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
            "confirmation_authority": False,
            "formal_study_authority": False,
            "publication_status": "IJOC_HOLD",
        },
        "claim_payload_sha256",
    )
    claim_sha = _write_json(claim_path, claim)
    owned_attempts: list[dict[str, object]] = []
    owned_artifact_paths: list[Path] = []
    for ordinal, row_id in zip(range(446, 451), target_rows[:5], strict=True):
        attempt_number = 2 if ordinal == 446 else 1
        attempt = diagnostic / "attempts" / row_id / f"attempt-{attempt_number:04d}"
        artifacts: list[dict[str, object]] = []
        for name in (
            "diagnostic.json",
            "independent.metric.json",
            "operational.metric-timeout-override.receipt.json",
            "row.json",
            "terminal.receipt.json",
            "trace.sqlite3",
            "worker.result.json",
            "worker.spec.json",
        ):
            path = attempt / name
            if name.endswith(".json"):
                sha = _write_json(path, {"row_id": row_id, "name": name})
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((row_id + name).encode("utf-8"))
                sha = runner.sha256_file(path)
            owned_artifact_paths.append(path)
            artifacts.append(
                {
                    "path": path.relative_to(diagnostic).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha,
                }
            )
        owned_attempts.append(
            {
                "row_id": row_id,
                "attempt_directory": attempt.relative_to(diagnostic).as_posix(),
                "artifacts": sorted(artifacts, key=lambda item: item["path"]),
            }
        )
    failure_path = diagnostic / (
        "metric-timeout-recovery.v21e3-motsp-development-n500-s00."
        "failure.receipt.json"
    )
    seal_path = diagnostic / (
        "metric-timeout-recovery.v21e3-motsp-development-n500-s00."
        "failure.receipt.seal.json"
    )
    failure = _bound(
        {
            "schema": "v21e3r1_metric_timeout_recovery_failure_receipt_v1",
            "status": "HOLD_METRIC_TIMEOUT_RECOVERY_FAILURE_MANUAL_AUDIT_REQUIRED",
            "scope": "EXACT_ORDINALS_446_462_FROZEN_DEVELOPMENT_RECOVERY_ONLY",
            "failure_phase": "PARALLEL_RECOVERY_ROWS",
            "exception_type": "RuntimeError",
            "exception_message": "synthetic job accounting race",
            "target_ordinals": list(range(446, 463)),
            "target_row_count": 17,
            "owned_attempts": owned_attempts,
            "validated_completed_markers": [],
            "cleanup_events": [],
            "terminal_descendant_state": {
                "schema": "v21e3r1_recovery_descendant_zero_scan_v1",
                "terminal_matching_process_count": 0,
                "terminal_observed_recovery_processes": [],
                "worker_specs": [],
                "block_all_recovery_processes": True,
                "scan_count": 1,
                "scan_payload_sha256": "8" * 64,
                "original_process_scan_payload_sha256": "9" * 64,
            },
            "terminal_descendant_state_confirmed": True,
            "preexisting_failed_attempt_preservation_status": "REQUIRES_MANUAL_REVALIDATION_AFTER_FAILURE",
            "recovery_semantics": "SAME_ALGORITHM_AND_METRIC_CODE_OPERATIONAL_TIMEOUT_OVERRIDE_ONLY",
            "helper_sha256": helper_sha,
            "helper_instance_claim_path": claim_path.name,
            "helper_instance_claim_sha256": claim_sha,
            "frozen_runner_sha256": fixture["frozen_runner_sha"],
            "independent_metric_source_sha256": metric_sha,
            "plan_sha256": fixture["plan_sha"],
            "source_snapshot_sha256": fixture["source_sha"],
            "aggregate_materialized": False,
            "diagnostic_receipt_materialized": False,
            "automatic_retry_authorized": False,
            "main_runner_resume_authorized": False,
            "manual_audit_required": True,
            "implementation_independence": False,
            "algorithm_execution_independence": False,
            "scientific_independence": False,
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
            "confirmation_authority": False,
            "formal_study_authority": False,
            "publication_status": "IJOC_HOLD",
        },
        "receipt_payload_sha256",
    )
    failure_sha = _write_json(failure_path, failure)
    seal = _bound(
        {
            "schema": "v21e3r1_metric_timeout_recovery_failure_seal_v1",
            "status": "SEALED_DURABLE_FAILURE_RECEIPT",
            "failure_receipt_path": failure_path.name,
            "failure_receipt_sha256": failure_sha,
            "failure_receipt_payload_sha256": failure["receipt_payload_sha256"],
            "helper_instance_claim_sha256": claim_sha,
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
            "confirmation_authority": False,
            "formal_study_authority": False,
            "publication_status": "IJOC_HOLD",
        },
        "seal_payload_sha256",
    )
    _write_json(seal_path, seal)
    fixture.update(
        {
            "old_helper": helper,
            "old_claim": claim_path,
            "old_failure": failure_path,
            "old_seal": seal_path,
            "metric_sha": metric_sha,
            "process_guard_sha": process_guard_sha,
            "owned_artifact_paths": owned_artifact_paths,
        }
    )
    return fixture


def _continuation_fixture(tmp_path: Path, runner):
    fixture = _old_recovery_fixture(tmp_path, runner)
    diagnostic: Path = fixture["diagnostic"]
    external_chain = runner.validate_external_scheduling_chain(
        diagnostic_root=diagnostic,
        helper_path=fixture["helper"],
        claim_path=fixture["claim"],
        handoff_path=fixture["handoff"],
        receipt_path=fixture["receipt"],
        seal_path=fixture["seal"],
        expected_plan_sha256=fixture["plan_sha"],
        expected_source_sha256=fixture["source_sha"],
        expected_frozen_runner_sha256=fixture["frozen_runner_sha"],
    )
    old_chain = runner.validate_old_recovery_failure_chain(
        diagnostic_root=diagnostic,
        helper_path=fixture["old_helper"],
        claim_path=fixture["old_claim"],
        failure_path=fixture["old_failure"],
        seal_path=fixture["old_seal"],
        upstream_receipt_path=fixture["receipt"],
        upstream_seal_path=fixture["seal"],
        expected_plan_sha256=fixture["plan_sha"],
        expected_source_sha256=fixture["source_sha"],
        expected_frozen_runner_sha256=fixture["frozen_runner_sha"],
        expected_metric_sha256=fixture["metric_sha"],
        expected_process_guard_sha256=fixture["process_guard_sha"],
    )
    target_rows = [f"synthetic-row-{ordinal:04d}" for ordinal in range(446, 463)]
    for ordinal in range(1, 446):
        row_id = f"synthetic-row-{ordinal:04d}"
        _write_json(
            diagnostic / "completed" / f"{row_id}.json",
            {
                "row_id": row_id,
                "attempt_directory": f"attempts/{row_id}/attempt-0001",
            },
        )
    preserved_paths = list((diagnostic / "completed").iterdir())
    preserved_manifest = _file_manifest(diagnostic, preserved_paths)
    external_manifest = _file_manifest(
        diagnostic,
        [fixture["handoff"], fixture["claim"], fixture["receipt"], fixture["seal"]],
    )
    custody = _bound(
        {
            "schema": "v21e3r1_external_scheduling_s01_custody_binding_v1",
            "status": "PASS_HASH_BOUND_EXTERNAL_SCHEDULING_ONLY_NO_NEW_AUTHORITY",
            "target_case_id": "v21e3-motsp-development-n500-s01",
            "target_ordinals": list(range(463, 505)),
            "target_row_count": 42,
            "external_helper_sha256": external_chain["helper_sha256"],
            "handoff_path": Path(fixture["handoff"]).name,
            "handoff_sha256": external_chain["handoff_sha256"],
            "handoff_payload_sha256": external_chain["handoff_payload_sha256"],
            "claim_path": Path(fixture["claim"]).name,
            "claim_sha256": external_chain["claim_sha256"],
            "claim_payload_sha256": external_chain["claim_payload_sha256"],
            "receipt_path": Path(fixture["receipt"]).name,
            "receipt_sha256": external_chain["receipt_sha256"],
            "receipt_payload_sha256": external_chain["receipt_payload_sha256"],
            "seal_path": Path(fixture["seal"]).name,
            "seal_sha256": external_chain["seal_sha256"],
            "seal_payload_sha256": external_chain["seal_payload_sha256"],
            "completed_marker_count": 42,
            "completed_marker_manifest_sha256": hashlib.sha256(
                _canonical_bytes(external_chain["completed_markers"])
            ).hexdigest(),
            "external_evidence_manifest_sha256": hashlib.sha256(
                _canonical_bytes(external_manifest)
            ).hexdigest(),
            "implementation_independence": False,
            "algorithm_execution_independence": False,
            "scientific_independence": False,
            **_hold_fields(),
        },
        "custody_payload_sha256",
    )
    terminal_scan = _bound(
        {
            "schema": "v21e3r1_recovery_continuation_terminal_process_scan_v1",
            "original_process_scan_payload_sha256": "7" * 64,
            "matching_process_count": 0,
            "matching_processes": [],
        },
        "scan_payload_sha256",
    )
    complete_manifest = sorted(
        [
            dict(artifact)
            for attempt in old_chain["old_complete_attempts"]
            for artifact in attempt["artifacts"]
        ],
        key=lambda item: item["path"],
    )
    failed_manifest = list(old_chain["preexisting_failed_attempt_manifest"])
    incident = _bound(
        {
            "schema": "v21e3r1_metric_timeout_recovery_continuation_incident_v1",
            "predecessor_helper_sha256": old_chain["helper_sha256"],
            "continuation_helper_sha256": runner.FROZEN_CONTINUATION_HELPER_SHA256,
            "old_claim_sha256": old_chain["claim_sha256"],
            "old_failure_receipt_sha256": old_chain["failure_receipt_sha256"],
            "old_failure_seal_sha256": old_chain["failure_seal_sha256"],
            "preserved_marker_count": 487,
            "preserved_marker_manifest_sha256": hashlib.sha256(
                _canonical_bytes(preserved_manifest)
            ).hexdigest(),
            "incident_complete_attempt_count": 5,
            "incident_complete_attempt_adopted_count": 0,
            "incident_complete_attempts_not_adopted": True,
            "incident_complete_attempt_manifest_sha256": hashlib.sha256(
                _canonical_bytes(complete_manifest)
            ).hexdigest(),
            "predecessor_failed_attempt_manifest_sha256": hashlib.sha256(
                _canonical_bytes(failed_manifest)
            ).hexdigest(),
            "external_scheduling_custody": custody,
            "external_scheduling_manifest": external_manifest,
            "external_scheduling_manifest_sha256": hashlib.sha256(
                _canonical_bytes(external_manifest)
            ).hexdigest(),
            "missing_recovery_marker_count": 17,
            "fresh_full_algorithm_rerun_count": 17,
            "terminal_process_scan": terminal_scan,
            "old_success_absent": True,
            **_hold_fields(),
        },
        "incident_payload_sha256",
    )
    attempt_by_ordinal = {
        ordinal: (3 if ordinal == 446 else 2 if ordinal <= 450 else 1)
        for ordinal in range(446, 463)
    }
    row_staging: list[dict[str, object]] = []
    worker_manifest: list[dict[str, object]] = []
    for ordinal, row_id in zip(range(446, 463), target_rows, strict=True):
        attempt_number = attempt_by_ordinal[ordinal]
        attempt = (
            diagnostic
            / "attempts"
            / row_id
            / f"attempt-{attempt_number:04d}"
        )
        spec = {"schema": "synthetic_worker_spec_v1", "row_id": row_id}
        spec_path = attempt / "worker.spec.json"
        spec_sha = _write_json(spec_path, spec)
        spec_payload_sha = hashlib.sha256(_canonical_bytes(spec)).hexdigest()
        result_path = attempt / "worker.result.json"
        result_sha = _write_json(
            result_path,
            {"status": "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY", "row_id": row_id},
        )
        worker_manifest.append(
            {
                "ordinal": ordinal,
                "row_id": row_id,
                "expected_attempt_number": attempt_number,
                "worker_spec_payload_sha256": spec_payload_sha,
            }
        )
        row_staging.append(
            {
                "ordinal": ordinal,
                "row_id": row_id,
                "attempt_number": attempt_number,
                "attempt": attempt,
                "spec_path": spec_path,
                "spec_sha": spec_sha,
                "spec_payload_sha": spec_payload_sha,
                "result_path": result_path,
                "result_sha": result_sha,
            }
        )
    interpreter = {"schema": "synthetic_interpreter_identity_v1", "status": "PASS"}
    environment = {"schema": "synthetic_environment_receipt_v1", "status": "PASS"}
    claim = _bound(
        {
            "schema": "v21e3r1_metric_timeout_recovery_continuation_claim_v1",
            "status": "SEALED_APPEND_ONLY_CONTINUATION_INSTANCE_CLAIM",
            "scope": "EXACT_ORDINALS_446_462_FROZEN_DEVELOPMENT_CONTINUATION_ONLY",
            "continuation_semantics": "CHAINED_APPEND_ONLY_RECOVERY_5_COMPLETE_NOT_ADOPTED_FRESH_RERUN_17_V1",
            "process_id": 1234,
            "target_case_id": "v21e3-motsp-development-n500-s00",
            "target_ordinals": list(range(446, 463)),
            "target_row_count": 17,
            "old_complete_attempt_count": 5,
            "old_complete_attempt_adopted_count": 0,
            "old_complete_attempts_not_adopted": True,
            "fresh_full_algorithm_rerun_count": 17,
            "worker_spec_payload_manifest": worker_manifest,
            "worker_spec_payload_manifest_sha256": hashlib.sha256(
                _canonical_bytes(worker_manifest)
            ).hexdigest(),
            "incident_receipt": incident,
            "predecessor_helper_sha256": old_chain["helper_sha256"],
            "predecessor_claim_sha256": old_chain["claim_sha256"],
            "predecessor_failure_receipt_sha256": old_chain[
                "failure_receipt_sha256"
            ],
            "predecessor_failure_seal_sha256": old_chain["failure_seal_sha256"],
            "plan_sha256": fixture["plan_sha"],
            "source_snapshot_sha256": fixture["source_sha"],
            "frozen_runner_sha256": fixture["frozen_runner_sha"],
            "independent_metric_source_sha256": fixture["metric_sha"],
            "process_guard_sha256": fixture["process_guard_sha"],
            "helper_sha256": runner.FROZEN_CONTINUATION_HELPER_SHA256,
            "interpreter_identity": interpreter,
            "environment_receipt": environment,
            "preclaim_process_scan": terminal_scan,
            "jobs": 4,
            "original_metric_timeout_seconds": 300,
            "operational_metric_timeout_seconds": 1200,
            "outer_row_timeout_seconds": 2400,
            "accounting_grace_seconds": 30,
            "original_main_runner_honors_this_claim": False,
            "automatic_resume_authorized": False,
            **_hold_fields(),
        },
        "claim_payload_sha256",
    )
    claim_path = diagnostic / (
        "metric-timeout-recovery-continuation."
        "v21e3-motsp-development-n500-s00.helper-instance.claim.json"
    )
    claim_sha = _write_json(claim_path, claim)
    completed_rows: list[dict[str, object]] = []
    timeout_paths: list[Path] = []
    for staged in row_staging:
        ordinal = staged["ordinal"]
        row_id = staged["row_id"]
        attempt: Path = staged["attempt"]
        command = ["synthetic-independent-metric", row_id]
        original_kwargs = {
            "cwd": tmp_path.as_posix(),
            "text": True,
            "capture_output": True,
            "timeout": 300,
            "check": False,
        }
        effective_kwargs = {**original_kwargs, "timeout": 1200}
        timeout = _bound(
            {
                "schema": "v21e3r1_metric_timeout_override_witness_v1",
                "status": "PASS_EXACT_ONE_INDEPENDENT_METRIC_TIMEOUT_OVERRIDE",
                "scope": "EXACT_FROZEN_DEVELOPMENT_RECOVERY_ROW_ONLY",
                "recovery_semantics": "SAME_ALGORITHM_AND_METRIC_CODE_OPERATIONAL_TIMEOUT_OVERRIDE_ONLY",
                "fresh_full_algorithm_rerun": True,
                "preexisting_failed_trace_reused": False,
                "original_diagnostic_receipt_alone_insufficient": True,
                "row_id": row_id,
                "worker_spec_path": "worker.spec.json",
                "worker_spec_sha256": staged["spec_sha"],
                "worker_result_path": "worker.result.json",
                "worker_result_sha256": staged["result_sha"],
                "independent_metric_command": command,
                "independent_metric_command_sha256": hashlib.sha256(
                    _canonical_bytes(command)
                ).hexdigest(),
                "subprocess_call_count": 1,
                "subprocess_returncode": 0,
                "original_subprocess_kwargs": original_kwargs,
                "effective_subprocess_kwargs": effective_kwargs,
                "plan_sha256": fixture["plan_sha"],
                "source_snapshot_sha256": fixture["source_sha"],
                "frozen_runner_sha256": fixture["frozen_runner_sha"],
                "independent_metric_source_sha256": fixture["metric_sha"],
                "helper_sha256": runner.FROZEN_CONTINUATION_HELPER_SHA256,
                "interpreter_identity": interpreter,
                "implementation_independence": False,
                "algorithm_execution_independence": False,
                "scientific_independence": False,
                **_hold_fields(),
            },
            "receipt_payload_sha256",
        )
        timeout_path = attempt / "operational.metric-timeout-override.receipt.json"
        timeout_sha = _write_json(timeout_path, timeout)
        timeout_paths.append(timeout_path)
        terminal_worker_scan = _bound(
            {
                "schema": "v21e3r1_recovery_continuation_descendant_zero_v1",
                "worker_specs": [Path(staged["spec_path"]).resolve().as_posix()],
                "block_all": False,
                "scan_count": 1,
                "terminal_matching_process_count": 0,
                "original_process_scan_payload_sha256": "8" * 64,
            },
            "scan_payload_sha256",
        )
        job_control = {
            "schema": "v21e3r1_continuation_windows_job_witness_v1",
            "kill_on_job_close_limit": True,
            "job_limit_flags": 0x2000,
            "wrapper_pid": 10000 + ordinal,
            "job_assignment_verified_before_gate_release": True,
            "outer_timeout_seconds": 2400,
            "outer_timeout_fired": False,
            "wrapper_returncode": 0,
            "initial_active_processes_after_wrapper_exit": 0,
            "accounting_grace_seconds": 30,
            "accounting_lag_observed": False,
            "accounting_lag_drained_without_termination": False,
            "accounting_grace_expired": False,
            "accounting_wait_seconds": 0.0,
            "terminal_active_processes": 0,
            "terminal_process_scan": terminal_worker_scan,
        }
        attempt_relative = attempt.relative_to(diagnostic).as_posix()
        job = _bound(
            {
                "schema": "v21e3r1_continuation_row_windows_job_receipt_v1",
                "status": "PASS_CORRECTED_WINDOWS_JOB_CONTAINMENT_AND_TERMINAL_ZERO",
                "scope": "ONE_FRESH_FROZEN_DEVELOPMENT_RECOVERY_RERUN_ONLY",
                "continuation_semantics": "CHAINED_APPEND_ONLY_RECOVERY_5_COMPLETE_NOT_ADOPTED_FRESH_RERUN_17_V1",
                "ordinal": ordinal,
                "row_id": row_id,
                "attempt_directory": attempt_relative,
                "worker_spec_sha256": staged["spec_sha"],
                "worker_result_sha256": staged["result_sha"],
                "timeout_witness_sha256": timeout_sha,
                "job_control": job_control,
                "helper_sha256": runner.FROZEN_CONTINUATION_HELPER_SHA256,
                "helper_instance_claim_sha256": claim_sha,
                "predecessor_complete_attempt_adopted": False,
                **_hold_fields(),
            },
            "receipt_payload_sha256",
        )
        job_path = attempt / "continuation.windows-job.receipt.json"
        job_sha = _write_json(job_path, job)
        marker = {
            "attempt_directory": attempt_relative,
            "diagnostic_sha256": "a" * 64,
            "independent_metric_receipt_sha256": "b" * 64,
            "plan_sha256": fixture["plan_sha"],
            "row_id": row_id,
            "row_sha256": "c" * 64,
            "status": "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY",
            "terminal_receipt_sha256": "d" * 64,
            "trace_sha256": "e" * 64,
        }
        marker_path = diagnostic / "completed" / f"{row_id}.json"
        marker_sha = _write_json(marker_path, marker)
        completed_rows.append(
            {
                "ordinal": ordinal,
                "row_id": row_id,
                "attempt_directory": attempt_relative,
                "worker_spec_sha256": staged["spec_sha"],
                "worker_spec_payload_sha256": staged["spec_payload_sha"],
                "worker_result_sha256": staged["result_sha"],
                "timeout_witness_sha256": timeout_sha,
                "timeout_witness_payload_sha256": timeout[
                    "receipt_payload_sha256"
                ],
                "windows_job_receipt_path": job_path.relative_to(
                    diagnostic
                ).as_posix(),
                "windows_job_receipt_sha256": job_sha,
                "windows_job_receipt_payload_sha256": job[
                    "receipt_payload_sha256"
                ],
                "windows_job_receipt": job,
                "completed_marker_path": marker_path.relative_to(
                    diagnostic
                ).as_posix(),
                "completed_marker_sha256": marker_sha,
            }
        )
    incident_files = _file_manifest(
        diagnostic,
        [fixture["old_claim"], fixture["old_failure"], fixture["old_seal"]],
    )
    receipt = _bound(
        {
            "schema": "v21e3r1_metric_timeout_recovery_continuation_receipt_v1",
            "status": "PASS_CHAINED_FRESH_EXACT17_OPERATIONAL_TIMEOUT_CONTINUATION_ONLY",
            "scope": "EXACT_ORDINALS_446_462_FROZEN_DEVELOPMENT_CONTINUATION_ONLY",
            "continuation_semantics": "CHAINED_APPEND_ONLY_RECOVERY_5_COMPLETE_NOT_ADOPTED_FRESH_RERUN_17_V1",
            "original_diagnostic_receipt_alone_insufficient": True,
            "target_case_id": "v21e3-motsp-development-n500-s00",
            "target_ordinals": list(range(446, 463)),
            "target_row_count": 17,
            "jobs": 4,
            "original_metric_timeout_seconds": 300,
            "operational_metric_timeout_seconds": 1200,
            "outer_row_timeout_seconds": 2400,
            "accounting_grace_seconds": 30,
            "old_complete_attempt_count": 5,
            "old_complete_attempt_adopted_count": 0,
            "incident_complete_attempts_not_adopted": True,
            "fresh_full_algorithm_rerun_count": 17,
            "predecessor_incident_immutable": True,
            "predecessor_helper_sha256": old_chain["helper_sha256"],
            "predecessor_claim_sha256": old_chain["claim_sha256"],
            "predecessor_failure_receipt_sha256": old_chain[
                "failure_receipt_sha256"
            ],
            "predecessor_failure_seal_sha256": old_chain["failure_seal_sha256"],
            "incident_receipt": incident,
            "incident_file_manifest": incident_files,
            "incident_complete_attempt_manifest": complete_manifest,
            "predecessor_failed_attempt_manifest": failed_manifest,
            "external_scheduling_custody": custody,
            "external_scheduling_manifest": external_manifest,
            "plan_sha256": fixture["plan_sha"],
            "source_snapshot_sha256": fixture["source_sha"],
            "frozen_runner_sha256": fixture["frozen_runner_sha"],
            "independent_metric_source_sha256": fixture["metric_sha"],
            "process_guard_sha256": fixture["process_guard_sha"],
            "helper_sha256": runner.FROZEN_CONTINUATION_HELPER_SHA256,
            "helper_instance_claim_path": claim_path.name,
            "helper_instance_claim_sha256": claim_sha,
            "interpreter_identity": interpreter,
            "environment_receipt": environment,
            "completed_rows": completed_rows,
            "final_completed_marker_count": 504,
            "aggregate_materialized": False,
            "diagnostic_receipt_materialized": False,
            "original_runner_resume_required": True,
            "original_runner_resume_after_continuation_success_only": True,
            "implementation_independence": False,
            "algorithm_execution_independence": False,
            "scientific_independence": False,
            **_hold_fields(),
        },
        "receipt_payload_sha256",
    )
    receipt_path = diagnostic / (
        "metric-timeout-recovery-continuation."
        "v21e3-motsp-development-n500-s00.receipt.json"
    )
    receipt_sha = _write_json(receipt_path, receipt)
    seal = _bound(
        {
            "schema": "v21e3r1_metric_timeout_recovery_continuation_success_seal_v1",
            "status": "SEALED_CHAINED_CONTINUATION_SUCCESS_RECEIPT_FILE_DIGEST",
            "receipt_path": receipt_path.name,
            "receipt_sha256": receipt_sha,
            "receipt_payload_sha256": receipt["receipt_payload_sha256"],
            "helper_instance_claim_sha256": claim_sha,
            "predecessor_failure_receipt_sha256": old_chain[
                "failure_receipt_sha256"
            ],
            "external_scheduling_receipt_sha256": external_chain["receipt_sha256"],
            "external_scheduling_success_seal_sha256": external_chain["seal_sha256"],
            "external_scheduling_custody_payload_sha256": custody[
                "custody_payload_sha256"
            ],
            **_hold_fields(),
        },
        "seal_payload_sha256",
    )
    seal_path = diagnostic / (
        "metric-timeout-recovery-continuation."
        "v21e3-motsp-development-n500-s00.receipt.seal.json"
    )
    _write_json(seal_path, seal)
    fixture.update(
        {
            "external_chain": external_chain,
            "old_chain": old_chain,
            "continuation_claim": claim_path,
            "continuation_receipt": receipt_path,
            "continuation_seal": seal_path,
            "continuation_timeout_paths": timeout_paths,
        }
    )
    return fixture


def test_identity_mismatch_fails_before_output_creation(tmp_path: Path) -> None:
    runner = _load_runner()
    inner = tmp_path / "inner.py"
    inner.write_text("raise SystemExit(99)\n", encoding="utf-8")
    output = tmp_path / "coverage"

    with pytest.raises(runner.RecoveryBoundCoverageError, match="outer runner"):
        runner.validate_runtime_identities(
            outer_runner_path=RUNNER_PATH,
            inner_runner_path=inner,
            expected_outer_runner_sha256="0" * 64,
            expected_inner_runner_sha256=runner.sha256_file(inner),
            expected_python_path=Path(sys.executable),
            expected_python_sha256=runner.sha256_file(Path(sys.executable)),
        )

    assert not output.exists()


def test_external_scheduler_chain_detects_raw_receipt_drift(tmp_path: Path) -> None:
    runner = _load_runner()
    fixture = _external_fixture(tmp_path, runner)
    binding = runner.validate_external_scheduling_chain(
        diagnostic_root=fixture["diagnostic"],
        helper_path=fixture["helper"],
        claim_path=fixture["claim"],
        handoff_path=fixture["handoff"],
        receipt_path=fixture["receipt"],
        seal_path=fixture["seal"],
        expected_plan_sha256=fixture["plan_sha"],
        expected_source_sha256=fixture["source_sha"],
        expected_frozen_runner_sha256=fixture["frozen_runner_sha"],
    )
    assert binding["target_row_count"] == 42

    fixture["receipt"].write_bytes(fixture["receipt"].read_bytes() + b" ")
    with pytest.raises(runner.RecoveryBoundCoverageError, match="receipt file"):
        runner.validate_external_scheduling_chain(
            diagnostic_root=fixture["diagnostic"],
            helper_path=fixture["helper"],
            claim_path=fixture["claim"],
            handoff_path=fixture["handoff"],
            receipt_path=fixture["receipt"],
            seal_path=fixture["seal"],
            expected_plan_sha256=fixture["plan_sha"],
            expected_source_sha256=fixture["source_sha"],
            expected_frozen_runner_sha256=fixture["frozen_runner_sha"],
        )


def test_old_recovery_failure_chain_detects_owned_artifact_drift(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    fixture = _old_recovery_fixture(tmp_path, runner)
    binding = runner.validate_old_recovery_failure_chain(
        diagnostic_root=fixture["diagnostic"],
        helper_path=fixture["old_helper"],
        claim_path=fixture["old_claim"],
        failure_path=fixture["old_failure"],
        seal_path=fixture["old_seal"],
        upstream_receipt_path=fixture["receipt"],
        upstream_seal_path=fixture["seal"],
        expected_plan_sha256=fixture["plan_sha"],
        expected_source_sha256=fixture["source_sha"],
        expected_frozen_runner_sha256=fixture["frozen_runner_sha"],
        expected_metric_sha256=fixture["metric_sha"],
        expected_process_guard_sha256=fixture["process_guard_sha"],
    )
    assert binding["old_complete_attempt_count"] == 5
    assert binding["old_complete_attempt_adopted_count"] == 0

    victim = fixture["owned_artifact_paths"][0]
    victim.write_bytes(victim.read_bytes() + b"drift")
    with pytest.raises(runner.RecoveryBoundCoverageError, match="owned artifact"):
        runner.validate_old_recovery_failure_chain(
            diagnostic_root=fixture["diagnostic"],
            helper_path=fixture["old_helper"],
            claim_path=fixture["old_claim"],
            failure_path=fixture["old_failure"],
            seal_path=fixture["old_seal"],
            upstream_receipt_path=fixture["receipt"],
            upstream_seal_path=fixture["seal"],
            expected_plan_sha256=fixture["plan_sha"],
            expected_source_sha256=fixture["source_sha"],
            expected_frozen_runner_sha256=fixture["frozen_runner_sha"],
            expected_metric_sha256=fixture["metric_sha"],
            expected_process_guard_sha256=fixture["process_guard_sha"],
        )


def test_output_initialization_race_loser_performs_zero_writes(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    output = tmp_path / "coverage"
    plan = _bound(
        {
            "schema": "v21e3r1_recovery_bound_coverage_plan_v2",
            "status": "FROZEN_RECOVERY_BOUND_COVERAGE",
            "jobs": 4,
            "row_timeout_seconds": 2400,
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
            "confirmation_authority": False,
            "formal_study_authority": False,
            "publication_status": "IJOC_HOLD",
        },
        "plan_payload_sha256",
    )
    provenance = _bound(
        {
            "schema": "v21e3r1_recovered_diagnostic_provenance_binding_v1",
            "status": "PASS_CHAIN_VALIDATED_FOR_REPLAY_INPUT_ONLY",
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
            "confirmation_authority": False,
            "formal_study_authority": False,
            "publication_status": "IJOC_HOLD",
        },
        "provenance_payload_sha256",
    )
    runner.initialize_recovery_bound_output(
        output_root=output,
        plan=plan,
        provenance=provenance,
        resume=False,
    )
    before = runner.tree_manifest(output)

    with pytest.raises(FileExistsError):
        runner.initialize_recovery_bound_output(
            output_root=output,
            plan=plan,
            provenance=provenance,
            resume=False,
        )

    assert runner.tree_manifest(output) == before


def test_diagnostic_boundary_requires_and_revalidates_exact_504_rows(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    inner_path = tmp_path / "inner.py"
    inner_path.write_text("# synthetic exact504 verifier\n", encoding="utf-8")
    diagnostic = tmp_path / "diagnostic"
    diagnostic.mkdir()
    _write_json(diagnostic / "diagnostic.receipt.json", {"status": "PASS"})
    _write_json(diagnostic / "diagnostic.aggregate.json", {"rows": 504})
    row_specs = [
        {
            "ordinal": ordinal,
            "row_id": f"synthetic-row-{ordinal:04d}",
            "family": "MOTSP" if ordinal >= 253 else "MOKP",
            "size": 500 if ordinal >= 421 else 100,
            "budget": 2000,
            "case_path": tmp_path / f"case-{ordinal:04d}.json",
            "case_sha256": f"{ordinal % 16:x}" * 64,
        }
        for ordinal in range(1, 505)
    ]
    calls: list[str] = []
    plan_sha = _write_json(diagnostic / "diagnostic.plan.json", {"expected_rows": 504})
    source_sha = "b" * 64
    source_entries = [
        {
            "path": "ijoc_submission_v21e3r1/scripts/run_v21e3r1_development_diagnostics.py",
            "bytes": 1,
            "sha256": "c" * 64,
        },
        {
            "path": "independent_reproduction/recompute_v21e3r1_metrics.py",
            "bytes": 1,
            "sha256": "d" * 64,
        },
    ]

    def validate_row(**kwargs):
        row_spec = kwargs["row_spec"]
        calls.append(row_spec["row_id"])
        return {
            "marker": {
                "attempt_directory": f"attempts/{row_spec['row_id']}/attempt-0001",
                "trace_sha256": "e" * 64,
            },
            "marker_sha256": "f" * 64,
            "trace": diagnostic / f"{row_spec['row_id']}.sqlite3",
        }

    fake_inner = SimpleNamespace(
        _load_plan_contract=lambda **kwargs: {
            "exact_full": True,
            "expected_rows": 504,
            "plan_sha256": plan_sha,
            "source_root": source_sha,
            "source_entries": source_entries,
            "row_specs": row_specs,
        },
        _scan_marker_names=lambda *args, **kwargs: {item["row_id"] for item in row_specs},
        _validate_diagnostic_final=lambda **kwargs: (
            runner.sha256_file(diagnostic / "diagnostic.receipt.json"),
            runner.sha256_file(diagnostic / "diagnostic.aggregate.json"),
        ),
        _validate_diagnostic_completed=validate_row,
    )

    binding = runner.validate_exact_diagnostic_tree(
        project_root=tmp_path,
        diagnostic_root=diagnostic,
        inner_runner_path=inner_path,
        expected_inner_runner_sha256=runner.sha256_file(inner_path),
        inner_module=fake_inner,
    )

    assert binding["expected_rows"] == 504
    assert len(binding["rows"]) == len(calls) == 504
    assert calls[0] == "synthetic-row-0001"
    assert calls[-1] == "synthetic-row-0504"


def test_n500_preflight_success_is_operational_only_and_sealed(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    inner = tmp_path / "inner.py"
    inner.write_text("# pinned inner runner\n", encoding="utf-8")
    provenance = _bound(
        {
            "schema": "v21e3r1_recovered_diagnostic_provenance_binding_v1",
            "status": "PASS_CHAIN_VALIDATED_FOR_REPLAY_INPUT_ONLY",
            "plan_sha256": "1" * 64,
            "source_snapshot_sha256": "2" * 64,
            "expected_rows": 504,
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
            "confirmation_authority": False,
            "formal_study_authority": False,
            "publication_status": "IJOC_HOLD",
        },
        "provenance_payload_sha256",
    )
    row = {
        "ordinal": 446,
        "row_id": "synthetic-n500-row",
        "family": "MOTSP",
        "size": 500,
        "budget": 2000,
        "attempt_directory": "attempts/synthetic-n500-row/attempt-0003",
        "completed_marker_path": "completed/synthetic-n500-row.json",
        "completed_marker_sha256": "3" * 64,
        "trace_path": (tmp_path / "trace.sqlite3").as_posix(),
        "trace_sha256": "4" * 64,
        "case_path": (tmp_path / "case.json").as_posix(),
        "case_sha256": "5" * 64,
    }
    diagnostic_binding = {
        "schema": "v21e3r1_recovery_bound_exact504_diagnostic_binding_v1",
        "status": "PASS_EXACT504_DIAGNOSTIC_TREE_REVALIDATED_INPUT_ONLY",
        "expected_rows": 504,
        "rows": [row],
        "plan_sha256": "1" * 64,
        "source_snapshot_sha256": "2" * 64,
        "implementation_independence": False,
        "scientific_independence": False,
        "runtime_authority": False,
        "scientific_authority": False,
        "selection_authority": False,
        "confirmation_authority": False,
        "formal_study_authority": False,
        "publication_status": "IJOC_HOLD",
    }
    output = tmp_path / "n500-preflight"

    def synthetic_executor(**kwargs):
        branch = _bound(
            {
                "schema": "v21e3r1_same_implementation_branch_replay_v1",
                "status": "PASS_SAME_IMPLEMENTATION_FULL_BRANCH_REEXECUTION",
                "implementation_independence": False,
                "scientific_independence": False,
                "third_party_replication": False,
            },
            "receipt_payload_sha256",
        )
        branch_path = kwargs["output_root"] / "branch.replay.json"
        branch_sha = _write_json(branch_path, branch)
        return {
            "branch_replay_receipt_path": branch_path,
            "branch_replay_receipt_sha256": branch_sha,
            "branch_replay_payload_sha256": branch["receipt_payload_sha256"],
            "process_isolation": "SYNTHETIC_BOUNDARY_ONLY",
            "returncode": 0,
            "wall_time_seconds": 1.25,
        }

    receipt = runner.run_n500_operational_preflight(
        project_root=tmp_path,
        diagnostic_root=tmp_path,
        output_root=output,
        inner_runner_path=inner,
        expected_outer_runner_sha256=runner.sha256_file(RUNNER_PATH),
        expected_inner_runner_sha256=runner.sha256_file(inner),
        expected_python_path=Path(sys.executable),
        expected_python_sha256=runner.sha256_file(Path(sys.executable)),
        provenance=provenance,
        diagnostic_binding=diagnostic_binding,
        row_id=row["row_id"],
        row_timeout_seconds=2400,
        executor=synthetic_executor,
    )

    assert receipt["status"] == "PASS_N500_FULL_BUDGET_OPERATIONAL_PREFLIGHT_ONLY"
    assert receipt["charged_evaluation_budget"] == 2000
    assert receipt["runtime_authority"] is False
    assert receipt["scientific_authority"] is False
    assert receipt["publication_status"] == "IJOC_HOLD"
    seal = json.loads(
        (output / "n500_preflight.receipt.seal.json").read_text(encoding="utf-8")
    )
    assert seal["receipt_sha256"] == runner.sha256_file(
        output / "n500_preflight.receipt.json"
    )

    # Even a fully re-digested/co-tampered chain must not be able to replace
    # the frozen inner-v1 branch receipt contract with a self-declared schema.
    branch_path = output / "branch.replay.json"
    branch = json.loads(branch_path.read_text(encoding="utf-8"))
    branch.pop("receipt_payload_sha256")
    branch["schema"] = "tampered_self_declared_branch_schema_v1"
    branch = _bound(branch, "receipt_payload_sha256")
    branch_raw_sha = _write_json(branch_path, branch)

    receipt_path = output / "n500_preflight.receipt.json"
    tampered_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered_receipt.pop("receipt_payload_sha256")
    tampered_receipt["branch_replay_receipt_sha256"] = branch_raw_sha
    tampered_receipt["branch_replay_payload_sha256"] = branch[
        "receipt_payload_sha256"
    ]
    tampered_receipt = _bound(tampered_receipt, "receipt_payload_sha256")
    tampered_receipt_raw_sha = _write_json(receipt_path, tampered_receipt)

    seal_path = output / "n500_preflight.receipt.seal.json"
    tampered_seal = json.loads(seal_path.read_text(encoding="utf-8"))
    tampered_seal.pop("seal_payload_sha256")
    tampered_seal["receipt_sha256"] = tampered_receipt_raw_sha
    tampered_seal["receipt_payload_sha256"] = tampered_receipt[
        "receipt_payload_sha256"
    ]
    tampered_seal = _bound(tampered_seal, "seal_payload_sha256")
    _write_json(seal_path, tampered_seal)

    identities = runner.validate_runtime_identities(
        outer_runner_path=RUNNER_PATH,
        inner_runner_path=inner,
        expected_outer_runner_sha256=runner.sha256_file(RUNNER_PATH),
        expected_inner_runner_sha256=runner.sha256_file(inner),
        expected_python_path=Path(sys.executable),
        expected_python_sha256=runner.sha256_file(Path(sys.executable)),
    )
    with pytest.raises(runner.RecoveryBoundCoverageError, match="schema/status"):
        runner.validate_n500_preflight_evidence(
            receipt_path=receipt_path,
            seal_path=seal_path,
            provenance=provenance,
            diagnostic_binding=diagnostic_binding,
            identities=identities,
        )


def test_n500_preflight_executor_failure_is_durable_and_never_success(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    inner = tmp_path / "inner.py"
    inner.write_text("# pinned inner runner\n", encoding="utf-8")
    provenance = _bound(
        {
            "schema": "v21e3r1_recovered_diagnostic_provenance_binding_v1",
            "status": "PASS_CHAIN_VALIDATED_FOR_REPLAY_INPUT_ONLY",
            "plan_sha256": "1" * 64,
            "source_snapshot_sha256": "2" * 64,
            "expected_rows": 504,
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
            "confirmation_authority": False,
            "formal_study_authority": False,
            "publication_status": "IJOC_HOLD",
        },
        "provenance_payload_sha256",
    )
    row = {
        "ordinal": 450,
        "row_id": "synthetic-failing-n500-row",
        "family": "MOKP",
        "size": 500,
        "budget": 2000,
        "attempt_directory": "attempts/synthetic-failing-n500-row/attempt-0002",
        "completed_marker_path": "completed/synthetic-failing-n500-row.json",
        "completed_marker_sha256": "3" * 64,
        "trace_path": (tmp_path / "trace.sqlite3").as_posix(),
        "trace_sha256": "4" * 64,
        "case_path": (tmp_path / "case.json").as_posix(),
        "case_sha256": "5" * 64,
    }
    diagnostic_binding = {
        "schema": "v21e3r1_recovery_bound_exact504_diagnostic_binding_v1",
        "status": "PASS_EXACT504_DIAGNOSTIC_TREE_REVALIDATED_INPUT_ONLY",
        "expected_rows": 504,
        "rows": [row],
        "plan_sha256": "1" * 64,
        "source_snapshot_sha256": "2" * 64,
        "implementation_independence": False,
        "scientific_independence": False,
        "runtime_authority": False,
        "scientific_authority": False,
        "selection_authority": False,
        "confirmation_authority": False,
        "formal_study_authority": False,
        "publication_status": "IJOC_HOLD",
    }
    output = tmp_path / "failed-n500-preflight"

    def failing_executor(**kwargs):
        raise RuntimeError("synthetic preflight boom")

    with pytest.raises(
        runner.RecoveryBoundCoverageError, match="Preflight executor failed"
    ):
        runner.run_n500_operational_preflight(
            project_root=tmp_path,
            diagnostic_root=tmp_path,
            output_root=output,
            inner_runner_path=inner,
            expected_outer_runner_sha256=runner.sha256_file(RUNNER_PATH),
            expected_inner_runner_sha256=runner.sha256_file(inner),
            expected_python_path=Path(sys.executable),
            expected_python_sha256=runner.sha256_file(Path(sys.executable)),
            provenance=provenance,
            diagnostic_binding=diagnostic_binding,
            row_id=row["row_id"],
            row_timeout_seconds=2400,
            executor=failing_executor,
        )

    assert not (output / "n500_preflight.receipt.json").exists()
    failure_path = output / "n500_preflight.failure.receipt.json"
    seal_path = output / "n500_preflight.failure.receipt.seal.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert failure["status"] == "HOLD_N500_OPERATIONAL_PREFLIGHT_FAILED"
    assert failure["failure_phase"] == "EXECUTOR"
    assert failure["exception_type"] == "RuntimeError"
    assert failure["runtime_authority"] is False
    assert failure["publication_status"] == "IJOC_HOLD"
    assert seal["failure_receipt_sha256"] == runner.sha256_file(failure_path)


def test_n500_preflight_post_executor_rejection_is_durable(tmp_path: Path) -> None:
    runner = _load_runner()
    inner = tmp_path / "inner.py"
    inner.write_text("# pinned inner runner\n", encoding="utf-8")
    provenance = _bound(
        {
            "schema": "v21e3r1_recovered_diagnostic_provenance_binding_v1",
            "status": "PASS_CHAIN_VALIDATED_FOR_REPLAY_INPUT_ONLY",
            "plan_sha256": "1" * 64,
            "source_snapshot_sha256": "2" * 64,
            "expected_rows": 504,
            **_hold_fields(),
        },
        "provenance_payload_sha256",
    )
    row = {
        "ordinal": 446,
        "row_id": "synthetic-post-validation-n500",
        "family": "MOTSP",
        "size": 500,
        "budget": 2000,
        "attempt_directory": "attempts/synthetic-post-validation-n500/attempt-0003",
        "completed_marker_path": "completed/synthetic-post-validation-n500.json",
        "completed_marker_sha256": "3" * 64,
        "trace_path": "synthetic.sqlite3",
        "trace_sha256": "4" * 64,
        "case_path": "synthetic.json",
        "case_sha256": "5" * 64,
    }
    diagnostic_binding = {
        "schema": "v21e3r1_recovery_bound_exact504_diagnostic_binding_v1",
        "status": "PASS_EXACT504_DIAGNOSTIC_TREE_REVALIDATED_INPUT_ONLY",
        "expected_rows": 504,
        "rows": [row],
        "plan_sha256": "1" * 64,
        "source_snapshot_sha256": "2" * 64,
        "implementation_independence": False,
        "scientific_independence": False,
        **_hold_fields(),
    }
    output = tmp_path / "post-validation-failure"

    with pytest.raises(
        runner.RecoveryBoundCoverageError, match="Preflight result validation failed"
    ):
        runner.run_n500_operational_preflight(
            project_root=tmp_path,
            diagnostic_root=tmp_path,
            output_root=output,
            inner_runner_path=inner,
            expected_outer_runner_sha256=runner.sha256_file(RUNNER_PATH),
            expected_inner_runner_sha256=runner.sha256_file(inner),
            expected_python_path=Path(sys.executable),
            expected_python_sha256=runner.sha256_file(Path(sys.executable)),
            provenance=provenance,
            diagnostic_binding=diagnostic_binding,
            row_id=row["row_id"],
            row_timeout_seconds=2400,
            executor=lambda **kwargs: {"returncode": 9},
        )

    failure = json.loads(
        (output / "n500_preflight.failure.receipt.json").read_text(encoding="utf-8")
    )
    assert failure["failure_phase"] == "POST_EXECUTOR_VALIDATION"
    assert not (output / "n500_preflight.receipt.json").exists()


def test_production_preflight_executor_delegates_exact_one_n500_row(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    inner_path = tmp_path / "inner.py"
    inner_path.write_text("# pinned inner runner\n", encoding="utf-8")
    diagnostic = tmp_path / "diagnostic"
    diagnostic.mkdir()
    (diagnostic / "diagnostic.plan.json").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "preflight-output"
    output.mkdir()
    row = {
        "ordinal": 446,
        "row_id": "synthetic-production-preflight",
        "family": "MOTSP",
        "size": 500,
        "budget": 2000,
        "case_path": (tmp_path / "case.json").as_posix(),
        "case_sha256": "a" * 64,
    }
    row_spec = {
        **row,
        "charged_evaluation_budget": 2000,
    }
    calls: list[dict[str, object]] = []

    def process_one(**kwargs):
        calls.append(kwargs)
        attempt = (
            kwargs["coverage_root"]
            / "attempts"
            / row["row_id"]
            / "attempt-0001"
        )
        branch = _bound(
            {
                "schema": "v21e3r1_same_implementation_branch_replay_v1",
                "status": "PASS_SAME_IMPLEMENTATION_FULL_BRANCH_REEXECUTION",
                "implementation_independence": False,
                "scientific_independence": False,
                "third_party_replication": False,
            },
            "receipt_payload_sha256",
        )
        branch_sha = _write_json(attempt / "branch.replay.json", branch)
        return {
            "row_id": row["row_id"],
            "attempt_directory": attempt.relative_to(
                kwargs["coverage_root"]
            ).as_posix(),
            "branch_replay_receipt_sha256": branch_sha,
        }

    fake_inner = SimpleNamespace(
        _load_plan_contract=lambda **kwargs: {
            "exact_full": True,
            "expected_rows": 504,
            "plan_sha256": "1" * 64,
            "source_root": "2" * 64,
            "source_entries": [],
            "row_specs": [row_spec],
        },
        _process_one_row=process_one,
        _validate_coverage_completed=lambda **kwargs: ({"row_id": row["row_id"]}, "b" * 64),
    )
    identities = runner.validate_runtime_identities(
        outer_runner_path=RUNNER_PATH,
        inner_runner_path=inner_path,
        expected_outer_runner_sha256=runner.sha256_file(RUNNER_PATH),
        expected_inner_runner_sha256=runner.sha256_file(inner_path),
        expected_python_path=Path(sys.executable),
        expected_python_sha256=runner.sha256_file(Path(sys.executable)),
    )

    result = runner._execute_real_n500_preflight(
        project_root=tmp_path,
        diagnostic_root=diagnostic,
        output_root=output,
        row=row,
        row_timeout_seconds=2400,
        expected_plan_sha256="1" * 64,
        expected_source_sha256="2" * 64,
        identities=identities,
        inner_module=fake_inner,
    )

    assert result["returncode"] == 0
    assert len(calls) == 1
    assert calls[0]["jobs"] == 1
    assert calls[0]["timeout_seconds"] == 2400
    assert calls[0]["row_spec"]["size"] == 500
    assert calls[0]["row_spec"]["budget"] == 2000


def test_continuation_validator_rejects_helper_drift_before_evidence_io(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    helper = tmp_path / "continuation.py"
    helper.write_text("# stable-looking but wrong continuation helper\n", encoding="utf-8")

    with pytest.raises(
        runner.RecoveryBoundCoverageError,
        match="Continuation helper identity drifted",
    ):
        runner.validate_continuation_success_chain(
            diagnostic_root=tmp_path / "missing-diagnostic",
            helper_path=helper,
            expected_helper_sha256="0" * 64,
            claim_path=tmp_path / "must-not-be-read-claim.json",
            receipt_path=tmp_path / "must-not-be-read-receipt.json",
            seal_path=tmp_path / "must-not-be-read-seal.json",
            external_chain={},
            old_recovery_chain={},
            expected_plan_sha256="1" * 64,
            expected_source_sha256="2" * 64,
            expected_frozen_runner_sha256="3" * 64,
            expected_metric_sha256="4" * 64,
            expected_process_guard_sha256="5" * 64,
        )


def _validate_continuation_fixture(runner, fixture):
    return runner.validate_continuation_success_chain(
        diagnostic_root=fixture["diagnostic"],
        helper_path=CONTINUATION_HELPER_PATH,
        expected_helper_sha256=runner.FROZEN_CONTINUATION_HELPER_SHA256,
        claim_path=fixture["continuation_claim"],
        receipt_path=fixture["continuation_receipt"],
        seal_path=fixture["continuation_seal"],
        external_chain=fixture["external_chain"],
        old_recovery_chain=fixture["old_chain"],
        expected_plan_sha256=fixture["plan_sha"],
        expected_source_sha256=fixture["source_sha"],
        expected_frozen_runner_sha256=fixture["frozen_runner_sha"],
        expected_metric_sha256=fixture["metric_sha"],
        expected_process_guard_sha256=fixture["process_guard_sha"],
    )


def test_continuation_validator_binds_exact_fresh17_witness_chain(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    assert (
        runner.sha256_file(CONTINUATION_HELPER_PATH)
        == runner.FROZEN_CONTINUATION_HELPER_SHA256
    )
    fixture = _continuation_fixture(tmp_path, runner)

    binding = _validate_continuation_fixture(runner, fixture)

    assert binding["status"] == "PASS_FRESH17_CONTINUATION_CHAIN_BOUND_INPUT_ONLY"
    assert binding["fresh_full_algorithm_rerun_count"] == 17
    assert binding["final_completed_marker_count"] == 504
    assert binding["runtime_authority"] is False
    assert binding["scientific_authority"] is False
    assert binding["publication_status"] == "IJOC_HOLD"


def test_continuation_validator_rejects_one_fresh_timeout_witness_drift(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    fixture = _continuation_fixture(tmp_path, runner)
    timeout_path = fixture["continuation_timeout_paths"][0]
    timeout = json.loads(timeout_path.read_text(encoding="utf-8"))
    timeout["subprocess_call_count"] = 2
    _write_json(timeout_path, timeout)

    with pytest.raises(
        runner.RecoveryBoundCoverageError,
        match="Continuation fresh-row evidence drifted",
    ):
        _validate_continuation_fixture(runner, fixture)


def test_recovered_exact504_provenance_cross_binds_all_chain_segments(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    fixture = _continuation_fixture(tmp_path, runner)
    continuation = _validate_continuation_fixture(runner, fixture)
    external_by_ordinal = {
        entry["ordinal"]: entry
        for entry in fixture["external_chain"]["completed_markers"]
    }
    continuation_by_ordinal = {
        entry["ordinal"]: entry for entry in continuation["completed_rows"]
    }
    rows: list[dict[str, object]] = []
    for ordinal in range(1, 505):
        row_id = f"synthetic-row-{ordinal:04d}"
        marker_path = fixture["diagnostic"] / "completed" / f"{row_id}.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if ordinal in continuation_by_ordinal:
            attempt_directory = continuation_by_ordinal[ordinal]["attempt_directory"]
        elif ordinal in external_by_ordinal:
            attempt_directory = external_by_ordinal[ordinal]["attempt_directory"]
        else:
            attempt_directory = marker["attempt_directory"]
        rows.append(
            {
                "ordinal": ordinal,
                "row_id": row_id,
                "family": "MOTSP",
                "size": 500,
                "budget": 2000,
                "attempt_directory": attempt_directory,
                "completed_marker_path": f"completed/{row_id}.json",
                "completed_marker_sha256": runner.sha256_file(marker_path),
                "trace_path": f"synthetic/{row_id}.sqlite3",
                "trace_sha256": "9" * 64,
                "case_path": f"synthetic/{row_id}.json",
                "case_sha256": "a" * 64,
            }
        )
    diagnostic_binding = {
        "schema": "v21e3r1_recovery_bound_exact504_diagnostic_binding_v1",
        "status": "PASS_EXACT504_DIAGNOSTIC_TREE_REVALIDATED_INPUT_ONLY",
        "expected_rows": 504,
        "rows": rows,
        "plan_sha256": fixture["plan_sha"],
        "source_snapshot_sha256": fixture["source_sha"],
        "frozen_diagnostic_runner_sha256": fixture["frozen_runner_sha"],
        "independent_metric_source_sha256": fixture["metric_sha"],
        "diagnostic_receipt_sha256": "b" * 64,
        "diagnostic_aggregate_sha256": "c" * 64,
        "inner_coverage_runner_sha256": "d" * 64,
        "implementation_independence": False,
        "scientific_independence": False,
        **_hold_fields(),
    }

    provenance = runner.build_recovered_diagnostic_provenance_binding(
        external_chain=fixture["external_chain"],
        old_recovery_chain=fixture["old_chain"],
        continuation_chain=continuation,
        diagnostic_binding=diagnostic_binding,
    )

    assert provenance["status"] == "PASS_CHAIN_VALIDATED_FOR_REPLAY_INPUT_ONLY"
    assert provenance["expected_rows"] == 504
    assert provenance["continuation_row_count"] == 17
    assert provenance["external_scheduling_row_count"] == 42
    assert provenance["runtime_authority"] is False
    assert provenance["publication_status"] == "IJOC_HOLD"


def test_full_coverage_requires_sealed_preflight_and_seals_outer_receipt(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    inner = tmp_path / "inner.py"
    inner.write_text("# pinned inner runner\n", encoding="utf-8")
    rows = [
        {
            "ordinal": ordinal,
            "row_id": f"synthetic-full-row-{ordinal:04d}",
            "family": "MOTSP",
            "size": 500,
            "budget": 2000,
            "attempt_directory": f"attempts/synthetic-full-row-{ordinal:04d}/attempt-0001",
            "completed_marker_path": f"completed/synthetic-full-row-{ordinal:04d}.json",
            "completed_marker_sha256": "3" * 64,
            "trace_path": f"trace-{ordinal}.sqlite3",
            "trace_sha256": "4" * 64,
            "case_path": f"case-{ordinal}.json",
            "case_sha256": "5" * 64,
        }
        for ordinal in range(1, 505)
    ]
    diagnostic_binding = {
        "schema": "v21e3r1_recovery_bound_exact504_diagnostic_binding_v1",
        "status": "PASS_EXACT504_DIAGNOSTIC_TREE_REVALIDATED_INPUT_ONLY",
        "expected_rows": 504,
        "rows": rows,
        "plan_sha256": "1" * 64,
        "source_snapshot_sha256": "2" * 64,
        "diagnostic_receipt_sha256": "6" * 64,
        "diagnostic_aggregate_sha256": "7" * 64,
        "implementation_independence": False,
        "scientific_independence": False,
        **_hold_fields(),
    }
    diagnostic_digest = hashlib.sha256(
        _canonical_bytes(diagnostic_binding)
    ).hexdigest()
    provenance = _bound(
        {
            "schema": "v21e3r1_recovered_diagnostic_provenance_binding_v1",
            "status": "PASS_CHAIN_VALIDATED_FOR_REPLAY_INPUT_ONLY",
            "plan_sha256": "1" * 64,
            "source_snapshot_sha256": "2" * 64,
            "expected_rows": 504,
            "exact504_diagnostic_binding_sha256": diagnostic_digest,
            "exact504_diagnostic_binding": diagnostic_binding,
            "implementation_independence": False,
            "scientific_independence": False,
            **_hold_fields(),
        },
        "provenance_payload_sha256",
    )
    selected = rows[445]
    preflight_root = tmp_path / "n500-preflight"

    def preflight_executor(**kwargs):
        branch = _bound(
            {
                "schema": "v21e3r1_same_implementation_branch_replay_v1",
                "status": "PASS_SAME_IMPLEMENTATION_FULL_BRANCH_REEXECUTION",
                "implementation_independence": False,
                "scientific_independence": False,
                "third_party_replication": False,
            },
            "receipt_payload_sha256",
        )
        branch_path = kwargs["output_root"] / "branch.replay.json"
        branch_sha = _write_json(branch_path, branch)
        return {
            "branch_replay_receipt_path": branch_path,
            "branch_replay_receipt_sha256": branch_sha,
            "branch_replay_payload_sha256": branch["receipt_payload_sha256"],
            "process_isolation": "SYNTHETIC_BOUNDARY_ONLY",
            "returncode": 0,
            "wall_time_seconds": 0.25,
        }

    runner.run_n500_operational_preflight(
        project_root=tmp_path,
        diagnostic_root=tmp_path,
        output_root=preflight_root,
        inner_runner_path=inner,
        expected_outer_runner_sha256=runner.sha256_file(RUNNER_PATH),
        expected_inner_runner_sha256=runner.sha256_file(inner),
        expected_python_path=Path(sys.executable),
        expected_python_sha256=runner.sha256_file(Path(sys.executable)),
        provenance=provenance,
        diagnostic_binding=diagnostic_binding,
        row_id=selected["row_id"],
        row_timeout_seconds=2400,
        executor=preflight_executor,
    )
    calls: list[dict[str, object]] = []

    def full_executor(**kwargs):
        calls.append(kwargs)
        inner_root = kwargs["inner_output_root"]
        inner_receipt = {
            "schema": "v21e3r1_branch_replay_coverage_receipt_v1",
            "status": "PASS_SAME_IMPLEMENTATION_BRANCH_REPLAY_EXACT_504_DEVELOPMENT_ONLY",
            "completed_rows": 504,
            "expected_rows": 504,
            "exact_full_504_coverage": True,
            "verification_jobs": kwargs["jobs"],
            "row_timeout_seconds": kwargs["row_timeout_seconds"],
            "diagnostic_plan_sha256": "1" * 64,
            "source_snapshot_sha256": "2" * 64,
            "diagnostic_receipt_sha256": "6" * 64,
            "diagnostic_aggregate_sha256": "7" * 64,
            "implementation_independence": False,
            "scientific_independence": False,
            "third_party_replication": False,
            "selection_authorized": False,
            "confirmation_authorized": False,
            "formal_authorized": False,
            "runtime_efficiency_claims": False,
            "scientific_performance_claims": False,
            "ijoc_submission_status": "IJOC_HOLD",
        }
        receipt_path = inner_root / "branch_replay_coverage.receipt.json"
        receipt_sha = _write_json(receipt_path, inner_receipt)
        return {
            "inner_receipt_path": receipt_path,
            "inner_receipt_sha256": receipt_sha,
            "inner_receipt_payload_sha256": hashlib.sha256(
                _canonical_bytes(inner_receipt)
            ).hexdigest(),
            "returncode": 0,
            "wall_time_seconds": 2.0,
            "process_isolation": "SYNTHETIC_FULL_BOUNDARY_ONLY",
        }

    output = tmp_path / "full-coverage"
    receipt = runner.run_recovery_bound_coverage(
        project_root=tmp_path,
        diagnostic_root=tmp_path,
        output_root=output,
        inner_runner_path=inner,
        expected_outer_runner_sha256=runner.sha256_file(RUNNER_PATH),
        expected_inner_runner_sha256=runner.sha256_file(inner),
        expected_python_path=Path(sys.executable),
        expected_python_sha256=runner.sha256_file(Path(sys.executable)),
        provenance=provenance,
        diagnostic_binding=diagnostic_binding,
        preflight_receipt_path=preflight_root / "n500_preflight.receipt.json",
        preflight_seal_path=preflight_root / "n500_preflight.receipt.seal.json",
        jobs=4,
        row_timeout_seconds=2400,
        resume=False,
        executor=full_executor,
    )

    assert receipt["status"] == "PASS_RECOVERY_BOUND_SAME_IMPLEMENTATION_EXACT504_ONLY"
    assert len(calls) == 1
    assert calls[0]["jobs"] == 4
    assert calls[0]["row_timeout_seconds"] == 2400
    assert calls[0]["expected_plan_sha256"] == "1" * 64
    assert calls[0]["expected_source_sha256"] == "2" * 64
    assert calls[0]["expected_diagnostic_receipt_sha256"] == "6" * 64
    assert calls[0]["expected_diagnostic_aggregate_sha256"] == "7" * 64
    assert receipt["runtime_authority"] is False
    assert receipt["publication_status"] == "IJOC_HOLD"
    assert (output / "executions" / "execution-0001.claim.json").is_file()
    assert (output / "recovery_bound_coverage.receipt.seal.json").is_file()

    inner_receipt_path = output / "inner_v1" / "branch_replay_coverage.receipt.json"
    tampered_inner = json.loads(inner_receipt_path.read_text(encoding="utf-8"))
    tampered_inner["diagnostic_plan_sha256"] = "8" * 64
    tampered_inner_raw_sha = _write_json(inner_receipt_path, tampered_inner)
    with pytest.raises(runner.RecoveryBoundCoverageError, match="receipt drifted"):
        runner._validate_full_executor_result(
            result={
                "inner_receipt_path": inner_receipt_path,
                "inner_receipt_sha256": tampered_inner_raw_sha,
                "inner_receipt_payload_sha256": hashlib.sha256(
                    _canonical_bytes(tampered_inner)
                ).hexdigest(),
                "returncode": 0,
                "wall_time_seconds": 2.0,
                "process_isolation": "SYNTHETIC_FULL_BOUNDARY_ONLY",
            },
            output_root=output,
            inner_output_root=output / "inner_v1",
            jobs=4,
            row_timeout_seconds=2400,
            expected_plan_sha256="1" * 64,
            expected_source_sha256="2" * 64,
            expected_diagnostic_receipt_sha256="6" * 64,
            expected_diagnostic_aggregate_sha256="7" * 64,
        )


def _minimal_full_inputs():
    rows = [
        {"ordinal": ordinal, "row_id": f"row-{ordinal:04d}"}
        for ordinal in range(1, 505)
    ]
    diagnostic = {
        "schema": "v21e3r1_recovery_bound_exact504_diagnostic_binding_v1",
        "status": "PASS_EXACT504_DIAGNOSTIC_TREE_REVALIDATED_INPUT_ONLY",
        "expected_rows": 504,
        "rows": rows,
        "plan_sha256": "1" * 64,
        "source_snapshot_sha256": "2" * 64,
        "diagnostic_receipt_sha256": "6" * 64,
        "diagnostic_aggregate_sha256": "7" * 64,
        "implementation_independence": False,
        "scientific_independence": False,
        **_hold_fields(),
    }
    diagnostic_sha = hashlib.sha256(_canonical_bytes(diagnostic)).hexdigest()
    provenance = _bound(
        {
            "schema": "v21e3r1_recovered_diagnostic_provenance_binding_v1",
            "status": "PASS_CHAIN_VALIDATED_FOR_REPLAY_INPUT_ONLY",
            "expected_rows": 504,
            "plan_sha256": "1" * 64,
            "source_snapshot_sha256": "2" * 64,
            "exact504_diagnostic_binding_sha256": diagnostic_sha,
            "exact504_diagnostic_binding": diagnostic,
            "implementation_independence": False,
            "scientific_independence": False,
            **_hold_fields(),
        },
        "provenance_payload_sha256",
    )
    return provenance, diagnostic


def test_full_coverage_bad_preflight_fails_before_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    inner = tmp_path / "inner.py"
    inner.write_text("# pinned inner runner\n", encoding="utf-8")
    provenance, diagnostic = _minimal_full_inputs()
    called = False

    def reject_preflight(**kwargs):
        raise runner.RecoveryBoundCoverageError("synthetic preflight seal drift")

    def must_not_execute(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(runner, "validate_n500_preflight_evidence", reject_preflight)
    output = tmp_path / "must-not-exist"
    with pytest.raises(
        runner.RecoveryBoundCoverageError, match="preflight seal drift"
    ):
        runner.run_recovery_bound_coverage(
            project_root=tmp_path,
            diagnostic_root=tmp_path,
            output_root=output,
            inner_runner_path=inner,
            expected_outer_runner_sha256=runner.sha256_file(RUNNER_PATH),
            expected_inner_runner_sha256=runner.sha256_file(inner),
            expected_python_path=Path(sys.executable),
            expected_python_sha256=runner.sha256_file(Path(sys.executable)),
            provenance=provenance,
            diagnostic_binding=diagnostic,
            preflight_receipt_path=tmp_path / "missing.json",
            preflight_seal_path=tmp_path / "missing.seal.json",
            jobs=4,
            row_timeout_seconds=2400,
            resume=False,
            executor=must_not_execute,
        )
    assert not called
    assert not output.exists()


def test_full_coverage_failure_is_durable_and_resume_preserves_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    inner = tmp_path / "inner.py"
    inner.write_text("# pinned inner runner\n", encoding="utf-8")
    provenance, diagnostic = _minimal_full_inputs()
    preflight_binding = {
        "schema": "v21e3r1_recovery_bound_n500_preflight_binding_v1",
        "status": "PASS_SEALED_N500_PREFLIGHT_REQUIRED_FOR_FULL_COVERAGE",
        "receipt_sha256": "3" * 64,
        "seal_sha256": "4" * 64,
        "runtime_authority": False,
        "scientific_authority": False,
        "selection_authority": False,
        "confirmation_authority": False,
        "formal_study_authority": False,
        "publication_status": "IJOC_HOLD",
    }
    monkeypatch.setattr(
        runner,
        "validate_n500_preflight_evidence",
        lambda **kwargs: preflight_binding,
    )
    output = tmp_path / "resumable-full"
    common = dict(
        project_root=tmp_path,
        diagnostic_root=tmp_path,
        output_root=output,
        inner_runner_path=inner,
        expected_outer_runner_sha256=runner.sha256_file(RUNNER_PATH),
        expected_inner_runner_sha256=runner.sha256_file(inner),
        expected_python_path=Path(sys.executable),
        expected_python_sha256=runner.sha256_file(Path(sys.executable)),
        provenance=provenance,
        diagnostic_binding=diagnostic,
        preflight_receipt_path=tmp_path / "synthetic-preflight.json",
        preflight_seal_path=tmp_path / "synthetic-preflight.seal.json",
        jobs=4,
        row_timeout_seconds=2400,
    )
    with pytest.raises(runner.RecoveryBoundCoverageError, match="executor failed"):
        runner.run_recovery_bound_coverage(
            **common,
            resume=False,
            executor=lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("synthetic full failure")
            ),
        )
    failure_path = output / "executions" / "execution-0001.failure.receipt.json"
    failure_seal_path = (
        output / "executions" / "execution-0001.failure.receipt.seal.json"
    )
    assert failure_path.is_file() and failure_seal_path.is_file()
    failure_sha = runner.sha256_file(failure_path)

    def successful_resume(**kwargs):
        inner_receipt = {
            "schema": "v21e3r1_branch_replay_coverage_receipt_v1",
            "status": "PASS_SAME_IMPLEMENTATION_BRANCH_REPLAY_EXACT_504_DEVELOPMENT_ONLY",
            "completed_rows": 504,
            "expected_rows": 504,
            "exact_full_504_coverage": True,
            "verification_jobs": 4,
            "row_timeout_seconds": 2400,
            "diagnostic_plan_sha256": "1" * 64,
            "source_snapshot_sha256": "2" * 64,
            "diagnostic_receipt_sha256": "6" * 64,
            "diagnostic_aggregate_sha256": "7" * 64,
            "implementation_independence": False,
            "scientific_independence": False,
            "third_party_replication": False,
            "selection_authorized": False,
            "confirmation_authorized": False,
            "formal_authorized": False,
            "runtime_efficiency_claims": False,
            "scientific_performance_claims": False,
            "ijoc_submission_status": "IJOC_HOLD",
        }
        path = kwargs["inner_output_root"] / "branch_replay_coverage.receipt.json"
        raw_sha = _write_json(path, inner_receipt)
        return {
            "inner_receipt_path": path,
            "inner_receipt_sha256": raw_sha,
            "inner_receipt_payload_sha256": hashlib.sha256(
                _canonical_bytes(inner_receipt)
            ).hexdigest(),
            "returncode": 0,
            "wall_time_seconds": 1.0,
            "process_isolation": "SYNTHETIC_RESUME_BOUNDARY",
        }

    receipt = runner.run_recovery_bound_coverage(
        **common, resume=True, executor=successful_resume
    )
    assert receipt["status"] == "PASS_RECOVERY_BOUND_SAME_IMPLEMENTATION_EXACT504_ONLY"
    assert (output / "executions" / "execution-0002.claim.json").is_file()
    assert runner.sha256_file(failure_path) == failure_sha
    outer_seal_path = output / "recovery_bound_coverage.receipt.seal.json"
    outer_seal = json.loads(outer_seal_path.read_text(encoding="utf-8"))
    outer_seal["receipt_sha256"] = "0" * 64
    _write_json(outer_seal_path, outer_seal)
    with pytest.raises(
        runner.RecoveryBoundCoverageError, match="Existing outer success seal"
    ):
        runner.run_recovery_bound_coverage(
            **common,
            resume=True,
            executor=lambda **kwargs: pytest.fail("existing PASS must not execute"),
        )


def test_execution_claim_race_loser_writes_zero_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    output = tmp_path / "race-output"
    (output / "executions").mkdir(parents=True)
    identities = runner.RuntimeIdentities(
        outer_runner_path="outer.py",
        outer_runner_sha256="1" * 64,
        inner_runner_path="inner.py",
        inner_runner_sha256="2" * 64,
        python_path="python.exe",
        python_sha256="3" * 64,
        python_version="synthetic",
    )
    before = runner.tree_manifest(output)

    def lose_race(path, payload):
        raise FileExistsError(path)

    monkeypatch.setattr(runner, "_exclusive_json", lose_race)
    with pytest.raises(FileExistsError):
        runner._acquire_execution_claim(
            output_root=output,
            plan_sha256="4" * 64,
            provenance_payload_sha256="5" * 64,
            preflight_binding={"receipt_sha256": "6" * 64, "seal_sha256": "7" * 64},
            identities=identities,
            jobs=4,
            row_timeout_seconds=2400,
            inner_resume=False,
        )
    assert runner.tree_manifest(output) == before


def test_cli_exposes_explicit_frozen_identity_and_preflight_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _load_runner()
    parser = runner._build_argument_parser()

    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["n500-preflight", "--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--expected-continuation-helper-sha256" in help_text
    assert "--expected-outer-runner-sha256" in help_text
    assert "--expected-python-sha256" in help_text
    assert "--row-id" in help_text
    assert "--output-root" in help_text

