from __future__ import annotations

"""Recovery-bound outer gate for V21e3r1 same-implementation replay.

This module is deliberately separate from the frozen diagnostic producer and
the existing v1 branch-replay coverage runner.  It can only add an engineering
provenance gate; it never grants scientific, selection, confirmation, formal,
runtime-performance, publication, or implementation-independence authority.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence


class RecoveryBoundCoverageError(RuntimeError):
    """A source, provenance, execution, or authority gate failed closed."""


FROZEN_CONTINUATION_HELPER_SHA256 = (
    "448ecb77a011bd7f02862d3d5eb906c129d40cbe0d6fed814a9a8d9d06f63828"
)
_AUTHORITY_A6_KEYS = {
    "runtime_authority",
    "scientific_authority",
    "selection_authority",
    "confirmation_authority",
    "formal_study_authority",
    "publication_status",
}
_CONTINUATION_CUSTODY_KEYS = {
    "schema", "status", "target_case_id", "target_ordinals",
    "target_row_count", "external_helper_sha256", "handoff_path",
    "handoff_sha256", "handoff_payload_sha256", "claim_path",
    "claim_sha256", "claim_payload_sha256", "receipt_path",
    "receipt_sha256", "receipt_payload_sha256", "seal_path", "seal_sha256",
    "seal_payload_sha256", "completed_marker_count",
    "completed_marker_manifest_sha256", "external_evidence_manifest_sha256",
    "custody_payload_sha256", "implementation_independence",
    "algorithm_execution_independence", "scientific_independence",
} | _AUTHORITY_A6_KEYS
_CONTINUATION_INCIDENT_KEYS = {
    "schema", "predecessor_helper_sha256", "continuation_helper_sha256",
    "old_claim_sha256", "old_failure_receipt_sha256",
    "old_failure_seal_sha256", "preserved_marker_count",
    "preserved_marker_manifest_sha256", "incident_complete_attempt_count",
    "incident_complete_attempt_adopted_count",
    "incident_complete_attempts_not_adopted",
    "incident_complete_attempt_manifest_sha256",
    "predecessor_failed_attempt_manifest_sha256",
    "external_scheduling_custody", "external_scheduling_manifest",
    "external_scheduling_manifest_sha256", "missing_recovery_marker_count",
    "fresh_full_algorithm_rerun_count", "terminal_process_scan",
    "old_success_absent", "incident_payload_sha256",
} | _AUTHORITY_A6_KEYS
_CONTINUATION_CLAIM_KEYS = {
    "schema", "status", "scope", "continuation_semantics", "process_id",
    "target_case_id", "target_ordinals", "target_row_count",
    "old_complete_attempt_count", "old_complete_attempt_adopted_count",
    "old_complete_attempts_not_adopted", "fresh_full_algorithm_rerun_count",
    "worker_spec_payload_manifest", "worker_spec_payload_manifest_sha256",
    "incident_receipt", "predecessor_helper_sha256",
    "predecessor_claim_sha256", "predecessor_failure_receipt_sha256",
    "predecessor_failure_seal_sha256", "plan_sha256",
    "source_snapshot_sha256", "frozen_runner_sha256",
    "independent_metric_source_sha256", "process_guard_sha256",
    "helper_sha256", "interpreter_identity", "environment_receipt",
    "preclaim_process_scan", "jobs", "original_metric_timeout_seconds",
    "operational_metric_timeout_seconds", "outer_row_timeout_seconds",
    "accounting_grace_seconds", "original_main_runner_honors_this_claim",
    "automatic_resume_authorized", "claim_payload_sha256",
} | _AUTHORITY_A6_KEYS
_CONTINUATION_RECEIPT_KEYS = {
    "schema", "status", "scope", "continuation_semantics",
    "original_diagnostic_receipt_alone_insufficient", "target_case_id",
    "target_ordinals", "target_row_count", "jobs",
    "original_metric_timeout_seconds", "operational_metric_timeout_seconds",
    "outer_row_timeout_seconds", "accounting_grace_seconds",
    "old_complete_attempt_count", "old_complete_attempt_adopted_count",
    "incident_complete_attempts_not_adopted",
    "fresh_full_algorithm_rerun_count", "predecessor_incident_immutable",
    "predecessor_helper_sha256", "predecessor_claim_sha256",
    "predecessor_failure_receipt_sha256", "predecessor_failure_seal_sha256",
    "incident_receipt", "incident_file_manifest",
    "incident_complete_attempt_manifest", "predecessor_failed_attempt_manifest",
    "external_scheduling_custody", "external_scheduling_manifest",
    "plan_sha256", "source_snapshot_sha256", "frozen_runner_sha256",
    "independent_metric_source_sha256", "process_guard_sha256",
    "helper_sha256", "helper_instance_claim_path",
    "helper_instance_claim_sha256", "interpreter_identity",
    "environment_receipt", "completed_rows", "final_completed_marker_count",
    "aggregate_materialized", "diagnostic_receipt_materialized",
    "original_runner_resume_required",
    "original_runner_resume_after_continuation_success_only",
    "receipt_payload_sha256", "implementation_independence",
    "algorithm_execution_independence", "scientific_independence",
} | _AUTHORITY_A6_KEYS
_CONTINUATION_SEAL_KEYS = {
    "schema", "status", "receipt_path", "receipt_sha256",
    "receipt_payload_sha256", "helper_instance_claim_sha256",
    "predecessor_failure_receipt_sha256",
    "external_scheduling_receipt_sha256",
    "external_scheduling_success_seal_sha256",
    "external_scheduling_custody_payload_sha256", "seal_payload_sha256",
} | _AUTHORITY_A6_KEYS
_CONTINUATION_WORKER_MANIFEST_KEYS = {
    "expected_attempt_number", "ordinal", "row_id",
    "worker_spec_payload_sha256",
}
_CONTINUATION_COMPLETED_ROW_KEYS = {
    "ordinal", "row_id", "attempt_directory", "worker_spec_sha256",
    "worker_spec_payload_sha256", "worker_result_sha256",
    "timeout_witness_sha256", "timeout_witness_payload_sha256",
    "windows_job_receipt_path", "windows_job_receipt_sha256",
    "windows_job_receipt_payload_sha256", "windows_job_receipt",
    "completed_marker_path", "completed_marker_sha256",
}
_CONTINUATION_TIMEOUT_WITNESS_KEYS = {
    "schema", "status", "scope", "recovery_semantics",
    "fresh_full_algorithm_rerun", "preexisting_failed_trace_reused",
    "original_diagnostic_receipt_alone_insufficient", "row_id",
    "worker_spec_path", "worker_spec_sha256", "worker_result_path",
    "worker_result_sha256", "independent_metric_command",
    "independent_metric_command_sha256", "subprocess_call_count",
    "subprocess_returncode", "original_subprocess_kwargs",
    "effective_subprocess_kwargs", "plan_sha256", "source_snapshot_sha256",
    "frozen_runner_sha256", "independent_metric_source_sha256",
    "helper_sha256", "interpreter_identity", "receipt_payload_sha256",
    "implementation_independence", "algorithm_execution_independence",
    "scientific_independence",
} | _AUTHORITY_A6_KEYS
_CONTINUATION_JOB_RECEIPT_KEYS = {
    "schema", "status", "scope", "continuation_semantics", "ordinal",
    "row_id", "attempt_directory", "worker_spec_sha256",
    "worker_result_sha256", "timeout_witness_sha256", "job_control",
    "helper_sha256", "helper_instance_claim_sha256",
    "predecessor_complete_attempt_adopted", "receipt_payload_sha256",
} | _AUTHORITY_A6_KEYS
_CONTINUATION_JOB_CONTROL_KEYS = {
    "schema", "kill_on_job_close_limit", "job_limit_flags", "wrapper_pid",
    "job_assignment_verified_before_gate_release", "outer_timeout_seconds",
    "outer_timeout_fired", "wrapper_returncode",
    "initial_active_processes_after_wrapper_exit", "accounting_grace_seconds",
    "accounting_lag_observed", "accounting_lag_drained_without_termination",
    "accounting_grace_expired", "accounting_wait_seconds",
    "terminal_active_processes", "terminal_process_scan",
}
_CONTINUATION_TERMINAL_SCAN_KEYS = {
    "schema", "worker_specs", "block_all", "scan_count",
    "terminal_matching_process_count", "original_process_scan_payload_sha256",
    "scan_payload_sha256",
}
_CONTINUATION_INCIDENT_SCAN_KEYS = {
    "schema", "original_process_scan_payload_sha256", "matching_process_count",
    "matching_processes", "scan_payload_sha256",
}
_DIAGNOSTIC_COMPLETED_MARKER_KEYS = {
    "attempt_directory", "diagnostic_sha256", "independent_metric_receipt_sha256",
    "plan_sha256", "row_id", "row_sha256", "status",
    "terminal_receipt_sha256", "trace_sha256",
}
_SUBPROCESS_KWARGS_KEYS = {"cwd", "text", "capture_output", "timeout", "check"}
_N500_PREFLIGHT_RECEIPT_KEYS = {
    "schema", "status", "plan_path", "plan_sha256", "plan_payload_sha256",
    "selected_row", "charged_evaluation_budget", "verification_jobs",
    "row_timeout_seconds", "wall_time_seconds", "process_isolation",
    "branch_replay_receipt_path", "branch_replay_receipt_sha256",
    "branch_replay_payload_sha256", "diagnostic_provenance_payload_sha256",
    "diagnostic_binding_sha256", "runtime_identities_before",
    "runtime_identities_after", "preflight_required_for_full_coverage",
    "operational_only", "implementation_independence",
    "scientific_independence", "receipt_payload_sha256",
} | _AUTHORITY_A6_KEYS
_N500_PREFLIGHT_SEAL_KEYS = {
    "schema", "status", "receipt_path", "receipt_sha256",
    "receipt_payload_sha256", "plan_sha256", "seal_payload_sha256",
} | _AUTHORITY_A6_KEYS
_OUTER_COVERAGE_RECEIPT_KEYS = {
    "schema", "status", "expected_rows", "jobs", "row_timeout_seconds",
    "wall_time_seconds", "process_isolation", "plan_sha256",
    "plan_payload_sha256", "provenance_payload_sha256",
    "diagnostic_binding_sha256", "preflight_receipt_sha256",
    "preflight_seal_sha256", "execution_claim_path",
    "execution_claim_sha256", "execution_claim_payload_sha256",
    "inner_receipt_path", "inner_receipt_sha256",
    "inner_receipt_payload_sha256", "runtime_identities_before",
    "runtime_identities_after", "same_implementation_only",
    "implementation_independence", "algorithm_execution_independence",
    "scientific_independence", "receipt_payload_sha256",
} | _AUTHORITY_A6_KEYS
_OUTER_COVERAGE_SEAL_KEYS = {
    "schema", "status", "receipt_path", "receipt_sha256",
    "receipt_payload_sha256", "plan_sha256", "provenance_payload_sha256",
    "preflight_receipt_sha256", "inner_receipt_sha256",
    "seal_payload_sha256",
} | _AUTHORITY_A6_KEYS


_EXTERNAL_CLAIM_KEYS = {
    "schema",
    "status",
    "scope",
    "process_id",
    "target_case_id",
    "target_row_count",
    "target_row_ids",
    "worker_spec_payload_manifest",
    "worker_spec_payload_manifest_sha256",
    "jobs",
    "helper_sha256",
    "frozen_runner_sha256",
    "handoff_receipt_path",
    "handoff_receipt_sha256",
    "handoff_receipt_payload_sha256",
    "plan_sha256",
    "source_snapshot_sha256",
    "interpreter_identity",
    "environment_receipt",
    "operational_quiescence_depends_on_external_stop_and_repeated_process_scan",
    "original_main_runner_honors_this_claim",
    "runtime_authority",
    "scientific_authority",
    "selection_authority",
    "claim_payload_sha256",
}
_EXTERNAL_HANDOFF_KEYS = {
    "schema",
    "status",
    "scope",
    "issued_at_utc",
    "output_root",
    "stopped_main_pid",
    "stopped_main_command_line",
    "stopped_main_command_sha256",
    "helper_sha256",
    "plan_sha256",
    "source_snapshot_sha256",
    "interpreter_identity",
    "environment_receipt",
    "completed_prefix",
    "process_scan",
    "receipt_is_audit_record_not_trusted_liveness",
    "runtime_authority",
    "scientific_authority",
    "selection_authority",
    "receipt_payload_sha256",
}
_EXTERNAL_RECEIPT_KEYS = {
    "schema",
    "status",
    "scope",
    "target_case_id",
    "target_row_count",
    "completed_marker_count",
    "completed_markers",
    "full_plan_row_count",
    "jobs",
    "scheduling_policy",
    "worker_execution",
    "completed_marker_generation",
    "completed_marker_verification",
    "plan_path",
    "plan_sha256",
    "source_snapshot_sha256",
    "frozen_runner_path",
    "frozen_runner_sha256",
    "helper_path",
    "helper_sha256",
    "helper_instance_claim_path",
    "helper_instance_claim_sha256",
    "handoff_receipt_path",
    "handoff_receipt_sha256",
    "handoff_receipt_payload_sha256",
    "receipt_seal_path",
    "interpreter_identity",
    "environment_receipt",
    "original_main_runner_honors_helper_instance_claim",
    "original_runner_resume_after_helper_success_only",
    "original_runner_resume_required",
    "original_runner_or_algorithm_sources_modified",
    "case_generation_performed",
    "generated_case_count",
    "implementation_independence",
    "algorithm_execution_independence",
    "scientific_independence",
    "runtime_authority",
    "scientific_authority",
    "selection_authority",
    "confirmation_authority",
    "formal_study_authority",
    "publication_status",
    "receipt_payload_sha256",
}
_EXTERNAL_SEAL_KEYS = {
    "schema",
    "status",
    "receipt_path",
    "receipt_sha256",
    "receipt_payload_sha256",
    "helper_instance_claim_sha256",
    "handoff_receipt_sha256",
    "runtime_authority",
    "scientific_authority",
    "selection_authority",
    "seal_payload_sha256",
}
_EXTERNAL_COMPLETED_KEYS = {
    "ordinal",
    "row_id",
    "path",
    "sha256",
    "attempt_directory",
    "worker_spec_path",
    "worker_spec_sha256",
    "worker_spec_payload_sha256",
}
_EXTERNAL_WORKER_MANIFEST_KEYS = {
    "ordinal",
    "row_id",
    "worker_spec_payload_sha256",
}
_OLD_RECOVERY_CLAIM_KEYS = {
    "schema",
    "status",
    "scope",
    "process_id",
    "target_case_id",
    "target_ordinals",
    "target_row_count",
    "target_row_ids",
    "worker_spec_payload_manifest",
    "worker_spec_payload_manifest_sha256",
    "jobs",
    "original_metric_timeout_seconds",
    "operational_metric_timeout_seconds",
    "outer_row_timeout_seconds",
    "outer_timeout_margin_seconds",
    "recovery_semantics",
    "fresh_full_algorithm_reruns_required",
    "metric_only_replay",
    "preexisting_failed_trace_reuse_authorized",
    "non_target_completed_marker_count",
    "non_target_completed_marker_manifest_sha256",
    "preexisting_failed_attempt_manifest",
    "preexisting_failed_attempt_manifest_sha256",
    "preclaim_process_scan",
    "helper_sha256",
    "frozen_runner_sha256",
    "independent_metric_source_sha256",
    "process_guard_sha256",
    "plan_sha256",
    "source_snapshot_sha256",
    "upstream_scheduling_receipt_sha256",
    "upstream_scheduling_seal_sha256",
    "interpreter_identity",
    "environment_receipt",
    "original_main_runner_honors_this_claim",
    "automatic_resume_authorized",
    "runtime_authority",
    "scientific_authority",
    "selection_authority",
    "confirmation_authority",
    "formal_study_authority",
    "publication_status",
    "claim_payload_sha256",
}
_OLD_RECOVERY_FAILURE_KEYS = {
    "schema",
    "status",
    "scope",
    "failure_phase",
    "exception_type",
    "exception_message",
    "target_ordinals",
    "target_row_count",
    "owned_attempts",
    "validated_completed_markers",
    "cleanup_events",
    "terminal_descendant_state",
    "terminal_descendant_state_confirmed",
    "preexisting_failed_attempt_preservation_status",
    "recovery_semantics",
    "helper_sha256",
    "helper_instance_claim_path",
    "helper_instance_claim_sha256",
    "frozen_runner_sha256",
    "independent_metric_source_sha256",
    "plan_sha256",
    "source_snapshot_sha256",
    "aggregate_materialized",
    "diagnostic_receipt_materialized",
    "automatic_retry_authorized",
    "main_runner_resume_authorized",
    "manual_audit_required",
    "implementation_independence",
    "algorithm_execution_independence",
    "scientific_independence",
    "runtime_authority",
    "scientific_authority",
    "selection_authority",
    "confirmation_authority",
    "formal_study_authority",
    "publication_status",
    "receipt_payload_sha256",
}
_OLD_RECOVERY_SEAL_KEYS = {
    "schema",
    "status",
    "failure_receipt_path",
    "failure_receipt_sha256",
    "failure_receipt_payload_sha256",
    "helper_instance_claim_sha256",
    "runtime_authority",
    "scientific_authority",
    "selection_authority",
    "confirmation_authority",
    "formal_study_authority",
    "publication_status",
    "seal_payload_sha256",
}
_OLD_WORKER_MANIFEST_KEYS = {
    "expected_attempt_number",
    "ordinal",
    "row_id",
    "worker_spec_payload_sha256",
}
_FILE_MANIFEST_KEYS = {"path", "bytes", "sha256"}
_OWNED_ATTEMPT_KEYS = {"row_id", "attempt_directory", "artifacts"}


def sha256_file(path: str | Path) -> str:
    target = Path(path).resolve()
    if target.is_symlink() or not target.is_file():
        raise RecoveryBoundCoverageError(f"Unsafe or missing file: {target}")
    before = target.stat()
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    after = target.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise RecoveryBoundCoverageError(f"File changed while hashing: {target}")
    return digest.hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_file_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_json(path: Path, payload: object) -> str:
    raw = _json_file_bytes(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    digest = hashlib.sha256(raw).hexdigest()
    if sha256_file(path) != digest:
        raise RecoveryBoundCoverageError(f"Exclusive JSON drifted after write: {path}")
    return digest


def tree_manifest(root: str | Path) -> list[dict[str, object]]:
    base = Path(root).resolve()
    if not base.is_dir() or base.is_symlink():
        raise RecoveryBoundCoverageError("Manifest root is unsafe or absent")
    entries: list[dict[str, object]] = []
    for path in sorted(base.rglob("*"), key=lambda item: item.relative_to(base).as_posix()):
        if path.is_symlink():
            raise RecoveryBoundCoverageError(f"Manifest contains a symlink: {path}")
        relative = path.relative_to(base).as_posix()
        if path.is_dir():
            entries.append({"path": relative + "/", "kind": "directory"})
        elif path.is_file():
            entries.append(
                {
                    "path": relative,
                    "kind": "file",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        else:
            raise RecoveryBoundCoverageError(f"Manifest contains an unsafe node: {path}")
    return entries


def initialize_recovery_bound_output(
    *,
    output_root: str | Path,
    plan: Mapping[str, object],
    provenance: Mapping[str, object],
    resume: bool,
) -> None:
    if type(resume) is not bool:
        raise RecoveryBoundCoverageError("resume must be an exact boolean")
    output = Path(output_root).resolve()
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise RecoveryBoundCoverageError("Output parent is unsafe or absent")
    plan_path = output / "recovery_bound_coverage.plan.json"
    provenance_path = output / "diagnostic_provenance.binding.json"
    executions = output / "executions"
    if not resume:
        try:
            os.mkdir(output)
        except FileExistsError:
            # Atomic directory creation is the first possible write.  A race
            # loser reaches this branch without owning any path or file.
            raise
        _fsync_directory(output.parent)
        _exclusive_json(plan_path, dict(plan))
        _exclusive_json(provenance_path, dict(provenance))
        os.mkdir(executions)
        _fsync_directory(output)
        return
    if output.is_symlink() or not output.is_dir():
        raise RecoveryBoundCoverageError("Resume output root is unsafe or absent")
    existing_plan, _ = _load_json_object(plan_path, label="recovery-bound plan")
    existing_provenance, _ = _load_json_object(
        provenance_path, label="diagnostic provenance binding"
    )
    if existing_plan != dict(plan):
        raise RecoveryBoundCoverageError("Resume plan disagrees with frozen invocation")
    if existing_provenance != dict(provenance):
        raise RecoveryBoundCoverageError("Resume provenance disagrees with frozen inputs")
    if executions.is_symlink() or not executions.is_dir():
        raise RecoveryBoundCoverageError("Resume execution ledger is unsafe or absent")


def _require_exact_keys(
    payload: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    observed = set(payload)
    if observed != expected:
        raise RecoveryBoundCoverageError(
            f"{label} keys drifted: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _load_json_object(path: str | Path, *, label: str) -> tuple[dict[str, Any], str]:
    target = Path(path).resolve()
    raw_sha = sha256_file(target)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RecoveryBoundCoverageError(f"Cannot load {label}: {target}") from error
    if not isinstance(payload, dict):
        raise RecoveryBoundCoverageError(f"{label} is not a JSON object")
    if sha256_file(target) != raw_sha:
        raise RecoveryBoundCoverageError(f"{label} changed while loading")
    return payload, raw_sha


def _validate_bound_payload(
    path: str | Path,
    *,
    label: str,
    expected_keys: set[str],
    digest_field: str,
    schema: str,
    status: str,
) -> tuple[dict[str, Any], str]:
    payload, raw_sha = _load_json_object(path, label=label)
    _require_exact_keys(payload, expected_keys, label=label)
    if payload.get("schema") != schema or payload.get("status") != status:
        raise RecoveryBoundCoverageError(f"{label} schema/status drifted")
    declared = _require_sha256(payload.get(digest_field), label=f"{label} digest")
    core = dict(payload)
    del core[digest_field]
    if hashlib.sha256(_canonical_bytes(core)).hexdigest() != declared:
        raise RecoveryBoundCoverageError(f"{label} payload digest drifted")
    return payload, raw_sha


def _contained_path(root: Path, relative: object, *, label: str) -> Path:
    if type(relative) is not str or not relative:
        raise RecoveryBoundCoverageError(f"{label} is not a nonempty path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RecoveryBoundCoverageError(f"{label} escaped its root")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise RecoveryBoundCoverageError(f"{label} escaped its root") from error
    return resolved


def _require_hold_authority(payload: Mapping[str, object], *, label: str) -> None:
    false_fields = {
        "implementation_independence",
        "algorithm_execution_independence",
        "scientific_independence",
        "runtime_authority",
        "scientific_authority",
        "selection_authority",
        "confirmation_authority",
        "formal_study_authority",
        "selection_authorized",
        "confirmation_authorized",
        "formal_authorized",
        "third_party_replication",
        "runtime_efficiency_claims",
        "scientific_performance_claims",
    }
    for field in false_fields & set(payload):
        if payload[field] is not False:
            raise RecoveryBoundCoverageError(f"{label} expands authority at {field}")
    for field in ("publication_status", "ijoc_submission_status"):
        if field in payload and payload[field] != "IJOC_HOLD":
            raise RecoveryBoundCoverageError(f"{label} publication status drifted")


def _require_list(value: object, *, label: str, length: int) -> list[object]:
    if not isinstance(value, list) or len(value) != length:
        raise RecoveryBoundCoverageError(f"{label} must contain exact {length} items")
    return value


def _validate_file_manifest(
    root: Path,
    raw_manifest: object,
    *,
    label: str,
    expected_length: int | None = None,
) -> list[dict[str, object]]:
    if not isinstance(raw_manifest, list):
        raise RecoveryBoundCoverageError(f"{label} is not an array")
    if expected_length is not None and len(raw_manifest) != expected_length:
        raise RecoveryBoundCoverageError(
            f"{label} must contain exact {expected_length} entries"
        )
    normalized: list[dict[str, object]] = []
    observed_paths: list[str] = []
    for raw in raw_manifest:
        if not isinstance(raw, dict):
            raise RecoveryBoundCoverageError(f"{label} entry is not an object")
        _require_exact_keys(raw, _FILE_MANIFEST_KEYS, label=f"{label} entry")
        path = _contained_path(root, raw.get("path"), label=f"{label} path")
        if path.is_symlink() or not path.is_file():
            raise RecoveryBoundCoverageError(f"{label} file is unsafe or absent")
        size = raw.get("bytes")
        digest = _require_sha256(raw.get("sha256"), label=f"{label} SHA-256")
        if type(size) is not int or size < 0:
            raise RecoveryBoundCoverageError(f"{label} byte count drifted")
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise RecoveryBoundCoverageError(f"{label} file drifted: {path.name}")
        relative = path.relative_to(root).as_posix()
        if relative != raw.get("path"):
            raise RecoveryBoundCoverageError(f"{label} path is not canonical")
        observed_paths.append(relative)
        normalized.append({"path": relative, "bytes": size, "sha256": digest})
    if len(set(item.casefold() for item in observed_paths)) != len(observed_paths):
        raise RecoveryBoundCoverageError(f"{label} contains duplicate paths")
    return normalized


def validate_external_scheduling_chain(
    *,
    diagnostic_root: str | Path,
    helper_path: str | Path,
    claim_path: str | Path,
    handoff_path: str | Path,
    receipt_path: str | Path,
    seal_path: str | Path,
    expected_plan_sha256: str,
    expected_source_sha256: str,
    expected_frozen_runner_sha256: str,
) -> dict[str, object]:
    diagnostic = Path(diagnostic_root).resolve()
    if not diagnostic.is_dir():
        raise RecoveryBoundCoverageError("Diagnostic root is absent")
    plan_sha = _require_sha256(expected_plan_sha256, label="diagnostic plan SHA-256")
    source_sha = _require_sha256(expected_source_sha256, label="source SHA-256")
    frozen_sha = _require_sha256(
        expected_frozen_runner_sha256, label="frozen runner SHA-256"
    )
    helper = Path(helper_path).resolve()
    helper_sha = sha256_file(helper)
    handoff, handoff_sha = _validate_bound_payload(
        handoff_path,
        label="external handoff receipt",
        expected_keys=_EXTERNAL_HANDOFF_KEYS,
        digest_field="receipt_payload_sha256",
        schema="v21e3r1_external_main_driver_handoff_receipt_v1",
        status="PASS_EXTERNAL_STOP_RECORDED_AND_NO_ORIGINAL_PROCESS_OBSERVED",
    )
    claim, claim_sha = _validate_bound_payload(
        claim_path,
        label="external scheduler claim",
        expected_keys=_EXTERNAL_CLAIM_KEYS,
        digest_field="claim_payload_sha256",
        schema="v21e3r1_external_scheduling_helper_instance_claim_v2",
        status="SEALED_HELPER_INSTANCE_CLAIM_NOT_A_MAIN_RUNNER_LOCK",
    )
    receipt, receipt_sha = _validate_bound_payload(
        receipt_path,
        label="external scheduling receipt",
        expected_keys=_EXTERNAL_RECEIPT_KEYS,
        digest_field="receipt_payload_sha256",
        schema="v21e3r1_external_scheduling_only_receipt_v2",
        status="PASS_EXTERNAL_SCHEDULING_ONLY_TARGET_42",
    )
    seal, seal_sha = _validate_bound_payload(
        seal_path,
        label="external scheduling seal",
        expected_keys=_EXTERNAL_SEAL_KEYS,
        digest_field="seal_payload_sha256",
        schema="v21e3r1_external_scheduling_receipt_file_seal_v1",
        status="PASS_SUCCESS_RECEIPT_FILE_DIGEST_SEALED",
    )
    for label, payload in (
        ("external handoff receipt", handoff),
        ("external scheduler claim", claim),
        ("external scheduling receipt", receipt),
        ("external scheduling seal", seal),
    ):
        _require_hold_authority(payload, label=label)
    if (
        handoff.get("scope") != "AUDIT_RECORD_ONLY_EXECUTION_MUST_RESCAN"
        or handoff.get("receipt_is_audit_record_not_trusted_liveness") is not True
        or claim.get("scope") != "EXTERNAL_SCHEDULING_ONLY_FROZEN_DEVELOPMENT_ROWS"
        or claim.get("target_case_id") != "v21e3-motsp-development-n500-s01"
        or claim.get("target_row_count") != 42
        or claim.get("jobs") != 4
        or claim.get("operational_quiescence_depends_on_external_stop_and_repeated_process_scan")
        is not True
        or claim.get("original_main_runner_honors_this_claim") is not False
        or receipt.get("scope") != "EXTERNAL_SCHEDULING_ONLY_FROZEN_DEVELOPMENT_ROWS"
        or receipt.get("target_case_id") != "v21e3-motsp-development-n500-s01"
        or receipt.get("target_row_count") != 42
        or receipt.get("completed_marker_count") != 42
        or receipt.get("full_plan_row_count") != 504
        or receipt.get("jobs") != 4
        or receipt.get("original_main_runner_honors_helper_instance_claim") is not False
        or receipt.get("original_runner_resume_after_helper_success_only") is not True
        or receipt.get("original_runner_resume_required") is not True
        or receipt.get("original_runner_or_algorithm_sources_modified") is not False
        or receipt.get("case_generation_performed") is not False
        or receipt.get("generated_case_count") != 0
    ):
        raise RecoveryBoundCoverageError("External scheduling semantics drifted")
    for payload in (handoff, claim, receipt):
        if (
            payload.get("helper_sha256") != helper_sha
            or payload.get("plan_sha256") != plan_sha
            or payload.get("source_snapshot_sha256") != source_sha
        ):
            raise RecoveryBoundCoverageError("External scheduling source binding drifted")
    if (
        claim.get("frozen_runner_sha256") != frozen_sha
        or receipt.get("frozen_runner_sha256") != frozen_sha
        or claim.get("handoff_receipt_path") != Path(handoff_path).name
        or claim.get("handoff_receipt_sha256") != handoff_sha
        or claim.get("handoff_receipt_payload_sha256")
        != handoff.get("receipt_payload_sha256")
        or receipt.get("helper_instance_claim_path") != Path(claim_path).name
        or receipt.get("helper_instance_claim_sha256") != claim_sha
        or receipt.get("handoff_receipt_path") != Path(handoff_path).name
        or receipt.get("handoff_receipt_sha256") != handoff_sha
        or receipt.get("handoff_receipt_payload_sha256")
        != handoff.get("receipt_payload_sha256")
        or receipt.get("receipt_seal_path") != Path(seal_path).name
        or seal.get("receipt_path") != Path(receipt_path).name
        or seal.get("receipt_sha256") != receipt_sha
        or seal.get("receipt_payload_sha256") != receipt.get("receipt_payload_sha256")
        or seal.get("helper_instance_claim_sha256") != claim_sha
        or seal.get("handoff_receipt_sha256") != handoff_sha
    ):
        raise RecoveryBoundCoverageError("External scheduling receipt file chain drifted")

    raw_completed = _require_list(
        receipt.get("completed_markers"), label="external completed markers", length=42
    )
    raw_target_ids = _require_list(
        claim.get("target_row_ids"), label="external target row IDs", length=42
    )
    raw_worker_manifest = _require_list(
        claim.get("worker_spec_payload_manifest"),
        label="external worker-spec manifest",
        length=42,
    )
    if hashlib.sha256(_canonical_bytes(raw_worker_manifest)).hexdigest() != claim.get(
        "worker_spec_payload_manifest_sha256"
    ):
        raise RecoveryBoundCoverageError("External worker-spec manifest digest drifted")
    observed_ids: list[str] = []
    normalized_manifest: list[dict[str, object]] = []
    normalized_completed: list[dict[str, object]] = []
    for expected_ordinal, (raw_entry, raw_manifest) in enumerate(
        zip(raw_completed, raw_worker_manifest, strict=True), start=463
    ):
        if not isinstance(raw_entry, dict) or not isinstance(raw_manifest, dict):
            raise RecoveryBoundCoverageError("External row evidence is not an object")
        _require_exact_keys(
            raw_entry, _EXTERNAL_COMPLETED_KEYS, label="external completed marker entry"
        )
        _require_exact_keys(
            raw_manifest,
            _EXTERNAL_WORKER_MANIFEST_KEYS,
            label="external worker manifest entry",
        )
        row_id = raw_entry.get("row_id")
        if type(row_id) is not str or not row_id:
            raise RecoveryBoundCoverageError("External row ID drifted")
        marker = _contained_path(diagnostic, raw_entry.get("path"), label="marker path")
        spec = _contained_path(
            diagnostic, raw_entry.get("worker_spec_path"), label="worker spec path"
        )
        attempt = _contained_path(
            diagnostic, raw_entry.get("attempt_directory"), label="attempt path"
        )
        spec_payload, spec_sha = _load_json_object(spec, label="external worker spec")
        spec_payload_sha = hashlib.sha256(_canonical_bytes(spec_payload)).hexdigest()
        marker_sha = sha256_file(marker)
        if (
            raw_entry.get("ordinal") != expected_ordinal
            or marker.name != f"{row_id}.json"
            or marker.parent.name != "completed"
            or attempt not in spec.parents
            or raw_entry.get("sha256") != marker_sha
            or raw_entry.get("worker_spec_sha256") != spec_sha
            or raw_entry.get("worker_spec_payload_sha256") != spec_payload_sha
            or raw_manifest
            != {
                "ordinal": expected_ordinal,
                "row_id": row_id,
                "worker_spec_payload_sha256": spec_payload_sha,
            }
        ):
            raise RecoveryBoundCoverageError("External row evidence drifted")
        observed_ids.append(row_id)
        normalized_manifest.append(dict(raw_manifest))
        normalized_completed.append(dict(raw_entry))
    if observed_ids != raw_target_ids or len(set(observed_ids)) != 42:
        raise RecoveryBoundCoverageError("External target row order/set drifted")
    return {
        "schema": "v21e3r1_recovery_bound_external_scheduling_binding_v1",
        "status": "PASS_EXTERNAL_SCHEDULING_CHAIN_BOUND_INPUT_ONLY",
        "target_row_count": 42,
        "target_row_ids": observed_ids,
        "completed_markers": normalized_completed,
        "worker_spec_payload_manifest": normalized_manifest,
        "helper_sha256": helper_sha,
        "claim_sha256": claim_sha,
        "claim_payload_sha256": claim["claim_payload_sha256"],
        "handoff_sha256": handoff_sha,
        "handoff_payload_sha256": handoff["receipt_payload_sha256"],
        "receipt_sha256": receipt_sha,
        "receipt_payload_sha256": receipt["receipt_payload_sha256"],
        "seal_sha256": seal_sha,
        "seal_payload_sha256": seal["seal_payload_sha256"],
        "plan_sha256": plan_sha,
        "source_snapshot_sha256": source_sha,
        "frozen_runner_sha256": frozen_sha,
        "implementation_independence": False,
        "scientific_independence": False,
        "runtime_authority": False,
        "scientific_authority": False,
        "selection_authority": False,
        "confirmation_authority": False,
        "formal_study_authority": False,
        "publication_status": "IJOC_HOLD",
    }


def validate_old_recovery_failure_chain(
    *,
    diagnostic_root: str | Path,
    helper_path: str | Path,
    claim_path: str | Path,
    failure_path: str | Path,
    seal_path: str | Path,
    upstream_receipt_path: str | Path,
    upstream_seal_path: str | Path,
    expected_plan_sha256: str,
    expected_source_sha256: str,
    expected_frozen_runner_sha256: str,
    expected_metric_sha256: str,
    expected_process_guard_sha256: str,
) -> dict[str, object]:
    diagnostic = Path(diagnostic_root).resolve()
    plan_sha = _require_sha256(expected_plan_sha256, label="diagnostic plan SHA-256")
    source_sha = _require_sha256(expected_source_sha256, label="source SHA-256")
    frozen_sha = _require_sha256(
        expected_frozen_runner_sha256, label="frozen runner SHA-256"
    )
    metric_sha = _require_sha256(expected_metric_sha256, label="metric SHA-256")
    process_guard_sha = _require_sha256(
        expected_process_guard_sha256, label="process guard SHA-256"
    )
    helper_sha = sha256_file(helper_path)
    upstream_receipt_sha = sha256_file(upstream_receipt_path)
    upstream_seal_sha = sha256_file(upstream_seal_path)
    claim, claim_sha = _validate_bound_payload(
        claim_path,
        label="old recovery claim",
        expected_keys=_OLD_RECOVERY_CLAIM_KEYS,
        digest_field="claim_payload_sha256",
        schema="v21e3r1_metric_timeout_recovery_helper_instance_claim_v1",
        status="SEALED_HELPER_INSTANCE_CLAIM_NOT_A_MAIN_RUNNER_LOCK",
    )
    failure, failure_sha = _validate_bound_payload(
        failure_path,
        label="old recovery failure receipt",
        expected_keys=_OLD_RECOVERY_FAILURE_KEYS,
        digest_field="receipt_payload_sha256",
        schema="v21e3r1_metric_timeout_recovery_failure_receipt_v1",
        status="HOLD_METRIC_TIMEOUT_RECOVERY_FAILURE_MANUAL_AUDIT_REQUIRED",
    )
    seal, seal_sha = _validate_bound_payload(
        seal_path,
        label="old recovery failure seal",
        expected_keys=_OLD_RECOVERY_SEAL_KEYS,
        digest_field="seal_payload_sha256",
        schema="v21e3r1_metric_timeout_recovery_failure_seal_v1",
        status="SEALED_DURABLE_FAILURE_RECEIPT",
    )
    for label, payload in (
        ("old recovery claim", claim),
        ("old recovery failure receipt", failure),
        ("old recovery failure seal", seal),
    ):
        _require_hold_authority(payload, label=label)
    expected_ordinals = list(range(446, 463))
    target_ids = _require_list(
        claim.get("target_row_ids"), label="old recovery target rows", length=17
    )
    worker_manifest = _require_list(
        claim.get("worker_spec_payload_manifest"),
        label="old recovery worker manifest",
        length=17,
    )
    if hashlib.sha256(_canonical_bytes(worker_manifest)).hexdigest() != claim.get(
        "worker_spec_payload_manifest_sha256"
    ):
        raise RecoveryBoundCoverageError("Old recovery worker manifest digest drifted")
    for expected_ordinal, raw_entry, row_id in zip(
        expected_ordinals, worker_manifest, target_ids, strict=True
    ):
        if not isinstance(raw_entry, dict):
            raise RecoveryBoundCoverageError("Old recovery worker entry is not an object")
        _require_exact_keys(
            raw_entry, _OLD_WORKER_MANIFEST_KEYS, label="old recovery worker entry"
        )
        if (
            raw_entry.get("ordinal") != expected_ordinal
            or raw_entry.get("row_id") != row_id
            or raw_entry.get("expected_attempt_number")
            != (2 if expected_ordinal == 446 else 1)
        ):
            raise RecoveryBoundCoverageError("Old recovery worker manifest drifted")
        _require_sha256(
            raw_entry.get("worker_spec_payload_sha256"),
            label="old recovery worker-spec payload SHA-256",
        )
    preexisting_manifest = _validate_file_manifest(
        diagnostic,
        claim.get("preexisting_failed_attempt_manifest"),
        label="preexisting failed attempt manifest",
        expected_length=4,
    )
    if hashlib.sha256(_canonical_bytes(preexisting_manifest)).hexdigest() != claim.get(
        "preexisting_failed_attempt_manifest_sha256"
    ):
        raise RecoveryBoundCoverageError("Preexisting failed attempt digest drifted")
    if (
        claim.get("scope")
        != "EXACT_ORDINALS_446_462_FROZEN_DEVELOPMENT_RECOVERY_ONLY"
        or claim.get("target_case_id") != "v21e3-motsp-development-n500-s00"
        or claim.get("target_ordinals") != expected_ordinals
        or claim.get("target_row_count") != 17
        or len(set(target_ids)) != 17
        or claim.get("jobs") != 4
        or claim.get("original_metric_timeout_seconds") != 300
        or claim.get("operational_metric_timeout_seconds") != 1200
        or claim.get("outer_row_timeout_seconds") != 2400
        or claim.get("outer_timeout_margin_seconds") != 1200
        or claim.get("fresh_full_algorithm_reruns_required") is not True
        or claim.get("metric_only_replay") is not False
        or claim.get("preexisting_failed_trace_reuse_authorized") is not False
        or claim.get("original_main_runner_honors_this_claim") is not False
        or claim.get("automatic_resume_authorized") is not False
        or claim.get("upstream_scheduling_receipt_sha256") != upstream_receipt_sha
        or claim.get("upstream_scheduling_seal_sha256") != upstream_seal_sha
    ):
        raise RecoveryBoundCoverageError("Old recovery claim semantics drifted")
    for payload in (claim, failure):
        if (
            payload.get("helper_sha256") != helper_sha
            or payload.get("frozen_runner_sha256") != frozen_sha
            or payload.get("independent_metric_source_sha256") != metric_sha
            or payload.get("plan_sha256") != plan_sha
            or payload.get("source_snapshot_sha256") != source_sha
        ):
            raise RecoveryBoundCoverageError("Old recovery source binding drifted")
    if claim.get("process_guard_sha256") != process_guard_sha:
        raise RecoveryBoundCoverageError("Old recovery process guard drifted")
    terminal = failure.get("terminal_descendant_state")
    if (
        failure.get("scope")
        != "EXACT_ORDINALS_446_462_FROZEN_DEVELOPMENT_RECOVERY_ONLY"
        or failure.get("failure_phase") != "PARALLEL_RECOVERY_ROWS"
        or failure.get("target_ordinals") != expected_ordinals
        or failure.get("target_row_count") != 17
        or failure.get("validated_completed_markers") != []
        or failure.get("terminal_descendant_state_confirmed") is not True
        or not isinstance(terminal, dict)
        or terminal.get("schema") != "v21e3r1_recovery_descendant_zero_scan_v1"
        or terminal.get("terminal_matching_process_count") != 0
        or terminal.get("terminal_observed_recovery_processes") != []
        or terminal.get("block_all_recovery_processes") is not True
        or failure.get("aggregate_materialized") is not False
        or failure.get("diagnostic_receipt_materialized") is not False
        or failure.get("automatic_retry_authorized") is not False
        or failure.get("main_runner_resume_authorized") is not False
        or failure.get("manual_audit_required") is not True
        or failure.get("helper_instance_claim_path") != Path(claim_path).name
        or failure.get("helper_instance_claim_sha256") != claim_sha
    ):
        raise RecoveryBoundCoverageError("Old recovery failure semantics drifted")
    raw_owned = _require_list(
        failure.get("owned_attempts"), label="old recovery owned attempts", length=5
    )
    owned_by_row: dict[str, dict[str, Any]] = {}
    for raw_attempt in raw_owned:
        if not isinstance(raw_attempt, dict):
            raise RecoveryBoundCoverageError("Old owned attempt is not an object")
        _require_exact_keys(
            raw_attempt, _OWNED_ATTEMPT_KEYS, label="old owned attempt"
        )
        row_id_value = raw_attempt.get("row_id")
        if type(row_id_value) is not str or row_id_value in owned_by_row:
            raise RecoveryBoundCoverageError("Old owned attempt row set drifted")
        owned_by_row[row_id_value] = raw_attempt
    expected_owned_rows = list(target_ids[:5])
    if set(owned_by_row) != set(expected_owned_rows):
        raise RecoveryBoundCoverageError("Old owned attempt row set drifted")
    normalized_owned: list[dict[str, object]] = []
    for expected_ordinal, row_id in zip(
        range(446, 451), expected_owned_rows, strict=True
    ):
        raw_attempt = owned_by_row[str(row_id)]
        attempt = _contained_path(
            diagnostic, raw_attempt.get("attempt_directory"), label="owned attempt"
        )
        expected_name = "attempt-0002" if expected_ordinal == 446 else "attempt-0001"
        if attempt.name != expected_name or attempt.parent.name != row_id:
            raise RecoveryBoundCoverageError("Old owned attempt path drifted")
        try:
            artifacts = _validate_file_manifest(
                diagnostic,
                raw_attempt.get("artifacts"),
                label="old owned artifact",
                expected_length=8,
            )
        except RecoveryBoundCoverageError as error:
            raise RecoveryBoundCoverageError("Old owned artifact drifted") from error
        expected_names = {
            "diagnostic.json",
            "independent.metric.json",
            "operational.metric-timeout-override.receipt.json",
            "row.json",
            "terminal.receipt.json",
            "trace.sqlite3",
            "worker.result.json",
            "worker.spec.json",
        }
        if {Path(str(item["path"])).name for item in artifacts} != expected_names:
            raise RecoveryBoundCoverageError("Old owned artifact layout drifted")
        normalized_owned.append(
            {
                "row_id": row_id,
                "attempt_directory": attempt.relative_to(diagnostic).as_posix(),
                "artifacts": artifacts,
            }
        )
    if seal.get("failure_receipt_path") != Path(failure_path).name or (
        seal.get("failure_receipt_sha256") != failure_sha
        or seal.get("failure_receipt_payload_sha256")
        != failure.get("receipt_payload_sha256")
        or seal.get("helper_instance_claim_sha256") != claim_sha
    ):
        raise RecoveryBoundCoverageError("Old recovery failure receipt file chain drifted")
    return {
        "schema": "v21e3r1_recovery_bound_old_failure_binding_v1",
        "status": "PASS_OLD_RECOVERY_FAILURE_CHAIN_BOUND_NOT_ADOPTED",
        "target_ordinals": expected_ordinals,
        "target_row_ids": list(target_ids),
        "old_complete_attempt_count": 5,
        "old_complete_attempt_adopted_count": 0,
        "old_complete_attempts": normalized_owned,
        "preexisting_failed_attempt_manifest": preexisting_manifest,
        "helper_sha256": helper_sha,
        "claim_sha256": claim_sha,
        "claim_payload_sha256": claim["claim_payload_sha256"],
        "failure_receipt_sha256": failure_sha,
        "failure_receipt_payload_sha256": failure["receipt_payload_sha256"],
        "failure_seal_sha256": seal_sha,
        "failure_seal_payload_sha256": seal["seal_payload_sha256"],
        "upstream_scheduling_receipt_sha256": upstream_receipt_sha,
        "upstream_scheduling_seal_sha256": upstream_seal_sha,
        "plan_sha256": plan_sha,
        "source_snapshot_sha256": source_sha,
        "frozen_runner_sha256": frozen_sha,
        "independent_metric_source_sha256": metric_sha,
        "process_guard_sha256": process_guard_sha,
        "implementation_independence": False,
        "scientific_independence": False,
        "runtime_authority": False,
        "scientific_authority": False,
        "selection_authority": False,
        "confirmation_authority": False,
        "formal_study_authority": False,
        "publication_status": "IJOC_HOLD",
    }


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RecoveryBoundCoverageError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_exact_diagnostic_tree(
    *,
    project_root: str | Path,
    diagnostic_root: str | Path,
    inner_runner_path: str | Path,
    expected_inner_runner_sha256: str,
    inner_module: Any | None = None,
) -> dict[str, object]:
    project = Path(project_root).resolve()
    diagnostic = Path(diagnostic_root).resolve()
    inner_path = Path(inner_runner_path).resolve()
    expected_inner_sha = _require_sha256(
        expected_inner_runner_sha256, label="expected inner runner SHA-256"
    )
    if sha256_file(inner_path) != expected_inner_sha:
        raise RecoveryBoundCoverageError("Inner runner drifted before diagnostic validation")
    module = (
        inner_module
        if inner_module is not None
        else _load_module(inner_path, "_v21e3r1_recovery_bound_inner_v1")
    )
    required_functions = (
        "_load_plan_contract",
        "_scan_marker_names",
        "_validate_diagnostic_final",
        "_validate_diagnostic_completed",
    )
    if any(not callable(getattr(module, name, None)) for name in required_functions):
        raise RecoveryBoundCoverageError("Inner runner omits a diagnostic verifier")
    plan_path = diagnostic / "diagnostic.plan.json"
    contract = module._load_plan_contract(
        project_root=project,
        diagnostic_plan_path=plan_path,
        allow_smoke=False,
    )
    row_specs = list(contract.get("row_specs", []))
    if (
        contract.get("exact_full") is not True
        or contract.get("expected_rows") != 504
        or len(row_specs) != 504
        or contract.get("plan_sha256") != sha256_file(plan_path)
    ):
        raise RecoveryBoundCoverageError("Diagnostic tree is not the exact frozen 504 plan")
    plan_sha = _require_sha256(
        contract.get("plan_sha256"), label="diagnostic plan SHA-256"
    )
    source_sha = _require_sha256(
        contract.get("source_root"), label="diagnostic source SHA-256"
    )
    row_ids = [item.get("row_id") for item in row_specs]
    if (
        any(type(row_id) is not str or not row_id for row_id in row_ids)
        or len(set(row_ids)) != 504
    ):
        raise RecoveryBoundCoverageError("Diagnostic row IDs drifted")
    marker_names = module._scan_marker_names(
        diagnostic / "completed", expected=set(row_ids), label="diagnostic"
    )
    if marker_names != set(row_ids):
        raise RecoveryBoundCoverageError("Diagnostic tree does not contain exact 504 markers")
    diagnostic_receipt_sha, diagnostic_aggregate_sha = (
        module._validate_diagnostic_final(
            diagnostic_root=diagnostic, contract=contract
        )
    )
    if (
        diagnostic_receipt_sha
        != sha256_file(diagnostic / "diagnostic.receipt.json")
        or diagnostic_aggregate_sha
        != sha256_file(diagnostic / "diagnostic.aggregate.json")
    ):
        raise RecoveryBoundCoverageError("Diagnostic final artifacts drifted")
    rows: list[dict[str, object]] = []
    for expected_ordinal, row_spec in enumerate(row_specs, start=1):
        if row_spec.get("ordinal") != expected_ordinal:
            raise RecoveryBoundCoverageError("Diagnostic row ordinal drifted")
        validated = module._validate_diagnostic_completed(
            diagnostic_root=diagnostic,
            row_spec=row_spec,
            diagnostic_plan_sha256=plan_sha,
            source_root=source_sha,
        )
        marker = validated.get("marker")
        if not isinstance(marker, dict):
            raise RecoveryBoundCoverageError("Diagnostic marker validation is incomplete")
        marker_sha = _require_sha256(
            validated.get("marker_sha256"), label="diagnostic marker SHA-256"
        )
        trace_sha = _require_sha256(
            marker.get("trace_sha256"), label="diagnostic trace SHA-256"
        )
        attempt_directory = marker.get("attempt_directory")
        if type(attempt_directory) is not str or not attempt_directory:
            raise RecoveryBoundCoverageError("Diagnostic attempt binding drifted")
        case_path = Path(row_spec["case_path"]).resolve()
        rows.append(
            {
                "ordinal": expected_ordinal,
                "row_id": row_spec["row_id"],
                "family": row_spec["family"],
                "size": row_spec["size"],
                "budget": row_spec["budget"],
                "attempt_directory": attempt_directory,
                "completed_marker_path": (
                    Path("completed") / f"{row_spec['row_id']}.json"
                ).as_posix(),
                "completed_marker_sha256": marker_sha,
                "trace_path": Path(validated["trace"]).resolve().as_posix(),
                "trace_sha256": trace_sha,
                "case_path": case_path.as_posix(),
                "case_sha256": _require_sha256(
                    row_spec.get("case_sha256"), label="case artifact SHA-256"
                ),
            }
        )
    source_entries = contract.get("source_entries")
    if not isinstance(source_entries, list):
        raise RecoveryBoundCoverageError("Diagnostic source entries are absent")
    source_by_path = {
        entry.get("path"): entry
        for entry in source_entries
        if isinstance(entry, dict) and type(entry.get("path")) is str
    }
    producer_path = (
        "ijoc_submission_v21e3r1/scripts/"
        "run_v21e3r1_development_diagnostics.py"
    )
    metric_path = "independent_reproduction/recompute_v21e3r1_metrics.py"
    if producer_path not in source_by_path or metric_path not in source_by_path:
        raise RecoveryBoundCoverageError("Diagnostic source inventory omits producer/metric")
    producer_sha = _require_sha256(
        source_by_path[producer_path].get("sha256"),
        label="frozen diagnostic runner SHA-256",
    )
    metric_sha = _require_sha256(
        source_by_path[metric_path].get("sha256"),
        label="independent metric SHA-256",
    )
    if sha256_file(inner_path) != expected_inner_sha:
        raise RecoveryBoundCoverageError("Inner runner drifted after diagnostic validation")
    return {
        "schema": "v21e3r1_recovery_bound_exact504_diagnostic_binding_v1",
        "status": "PASS_EXACT504_DIAGNOSTIC_TREE_REVALIDATED_INPUT_ONLY",
        "expected_rows": 504,
        "rows": rows,
        "plan_sha256": plan_sha,
        "source_snapshot_sha256": source_sha,
        "frozen_diagnostic_runner_sha256": producer_sha,
        "independent_metric_source_sha256": metric_sha,
        "diagnostic_receipt_sha256": diagnostic_receipt_sha,
        "diagnostic_aggregate_sha256": diagnostic_aggregate_sha,
        "inner_coverage_runner_sha256": expected_inner_sha,
        "implementation_independence": False,
        "scientific_independence": False,
        "runtime_authority": False,
        "scientific_authority": False,
        "selection_authority": False,
        "confirmation_authority": False,
        "formal_study_authority": False,
        "publication_status": "IJOC_HOLD",
    }


def _validate_exact_bound_mapping(
    payload: object,
    *,
    label: str,
    expected_keys: set[str],
    digest_field: str,
    schema: str,
    status: str | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RecoveryBoundCoverageError(f"{label} is not an object")
    _require_exact_keys(payload, expected_keys, label=label)
    if payload.get("schema") != schema or (
        status is not None and payload.get("status") != status
    ):
        raise RecoveryBoundCoverageError(f"{label} schema/status drifted")
    declared = _require_sha256(payload.get(digest_field), label=f"{label} digest")
    core = dict(payload)
    del core[digest_field]
    if hashlib.sha256(_canonical_bytes(core)).hexdigest() != declared:
        raise RecoveryBoundCoverageError(f"{label} payload digest drifted")
    _require_hold_authority(payload, label=label)
    return payload


def _manifest_payload_sha256(manifest: object, *, label: str) -> str:
    if not isinstance(manifest, list):
        raise RecoveryBoundCoverageError(f"{label} is not an array")
    return hashlib.sha256(_canonical_bytes(manifest)).hexdigest()


def _snapshot_file_manifest(root: Path, paths: Sequence[Path]) -> list[dict[str, object]]:
    normalized: list[Path] = []
    for raw_path in paths:
        path = raw_path.resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise RecoveryBoundCoverageError("Snapshot file escaped diagnostic root") from error
        if path.is_symlink() or not path.is_file():
            raise RecoveryBoundCoverageError("Snapshot file is unsafe or absent")
        normalized.append(path)
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(
            normalized, key=lambda item: item.relative_to(root).as_posix()
        )
    ]


def _validate_continuation_incident_scan(payload: object, *, label: str) -> None:
    scan = _validate_exact_bound_mapping(
        payload,
        label=label,
        expected_keys=_CONTINUATION_INCIDENT_SCAN_KEYS,
        digest_field="scan_payload_sha256",
        schema="v21e3r1_recovery_continuation_terminal_process_scan_v1",
    )
    if (
        scan.get("matching_process_count") != 0
        or scan.get("matching_processes") != []
    ):
        raise RecoveryBoundCoverageError(f"{label} is not terminal-zero")
    _require_sha256(
        scan.get("original_process_scan_payload_sha256"),
        label=f"{label} original process scan SHA-256",
    )


def validate_continuation_success_chain(
    *,
    diagnostic_root: str | Path,
    helper_path: str | Path,
    expected_helper_sha256: str,
    claim_path: str | Path,
    receipt_path: str | Path,
    seal_path: str | Path,
    external_chain: Mapping[str, object],
    old_recovery_chain: Mapping[str, object],
    expected_plan_sha256: str,
    expected_source_sha256: str,
    expected_frozen_runner_sha256: str,
    expected_metric_sha256: str,
    expected_process_guard_sha256: str,
) -> dict[str, object]:
    """Bind the immutable failed incident to every fresh continuation witness."""

    expected_helper = _require_sha256(
        expected_helper_sha256, label="expected continuation helper SHA-256"
    )
    if (
        expected_helper != FROZEN_CONTINUATION_HELPER_SHA256
        or sha256_file(helper_path) != FROZEN_CONTINUATION_HELPER_SHA256
    ):
        raise RecoveryBoundCoverageError("Continuation helper identity drifted")
    diagnostic = Path(diagnostic_root).resolve()
    if diagnostic.is_symlink() or not diagnostic.is_dir():
        raise RecoveryBoundCoverageError("Continuation diagnostic root is unsafe")
    plan_sha = _require_sha256(expected_plan_sha256, label="diagnostic plan SHA-256")
    source_sha = _require_sha256(
        expected_source_sha256, label="source snapshot SHA-256"
    )
    frozen_sha = _require_sha256(
        expected_frozen_runner_sha256, label="frozen diagnostic runner SHA-256"
    )
    metric_sha = _require_sha256(
        expected_metric_sha256, label="independent metric SHA-256"
    )
    process_guard_sha = _require_sha256(
        expected_process_guard_sha256, label="process guard SHA-256"
    )
    if (
        external_chain.get("schema")
        != "v21e3r1_recovery_bound_external_scheduling_binding_v1"
        or external_chain.get("status")
        != "PASS_EXTERNAL_SCHEDULING_CHAIN_BOUND_INPUT_ONLY"
        or external_chain.get("target_row_count") != 42
        or external_chain.get("plan_sha256") != plan_sha
        or external_chain.get("source_snapshot_sha256") != source_sha
        or external_chain.get("frozen_runner_sha256") != frozen_sha
        or external_chain.get("helper_sha256") != process_guard_sha
    ):
        raise RecoveryBoundCoverageError("External scheduling binding drifted")
    _require_hold_authority(external_chain, label="external scheduling binding")
    if (
        old_recovery_chain.get("schema")
        != "v21e3r1_recovery_bound_old_failure_binding_v1"
        or old_recovery_chain.get("status")
        != "PASS_OLD_RECOVERY_FAILURE_CHAIN_BOUND_NOT_ADOPTED"
        or old_recovery_chain.get("target_ordinals") != list(range(446, 463))
        or old_recovery_chain.get("old_complete_attempt_count") != 5
        or old_recovery_chain.get("old_complete_attempt_adopted_count") != 0
        or old_recovery_chain.get("plan_sha256") != plan_sha
        or old_recovery_chain.get("source_snapshot_sha256") != source_sha
        or old_recovery_chain.get("frozen_runner_sha256") != frozen_sha
        or old_recovery_chain.get("independent_metric_source_sha256") != metric_sha
        or old_recovery_chain.get("process_guard_sha256") != process_guard_sha
        or old_recovery_chain.get("upstream_scheduling_receipt_sha256")
        != external_chain.get("receipt_sha256")
        or old_recovery_chain.get("upstream_scheduling_seal_sha256")
        != external_chain.get("seal_sha256")
    ):
        raise RecoveryBoundCoverageError("Old recovery binding drifted")
    _require_hold_authority(old_recovery_chain, label="old recovery binding")
    claim, claim_sha = _validate_bound_payload(
        claim_path,
        label="continuation claim",
        expected_keys=_CONTINUATION_CLAIM_KEYS,
        digest_field="claim_payload_sha256",
        schema="v21e3r1_metric_timeout_recovery_continuation_claim_v1",
        status="SEALED_APPEND_ONLY_CONTINUATION_INSTANCE_CLAIM",
    )
    receipt, receipt_sha = _validate_bound_payload(
        receipt_path,
        label="continuation success receipt",
        expected_keys=_CONTINUATION_RECEIPT_KEYS,
        digest_field="receipt_payload_sha256",
        schema="v21e3r1_metric_timeout_recovery_continuation_receipt_v1",
        status="PASS_CHAINED_FRESH_EXACT17_OPERATIONAL_TIMEOUT_CONTINUATION_ONLY",
    )
    seal, seal_sha = _validate_bound_payload(
        seal_path,
        label="continuation success seal",
        expected_keys=_CONTINUATION_SEAL_KEYS,
        digest_field="seal_payload_sha256",
        schema="v21e3r1_metric_timeout_recovery_continuation_success_seal_v1",
        status="SEALED_CHAINED_CONTINUATION_SUCCESS_RECEIPT_FILE_DIGEST",
    )
    for label, payload in (
        ("continuation claim", claim),
        ("continuation success receipt", receipt),
        ("continuation success seal", seal),
    ):
        _require_hold_authority(payload, label=label)
    if sha256_file(helper_path) != FROZEN_CONTINUATION_HELPER_SHA256:
        raise RecoveryBoundCoverageError("Continuation helper drifted while loading evidence")

    semantics = (
        "CHAINED_APPEND_ONLY_RECOVERY_5_COMPLETE_NOT_ADOPTED_FRESH_RERUN_17_V1"
    )
    target_ordinals = list(range(446, 463))
    target_ids = old_recovery_chain.get("target_row_ids")
    if (
        not isinstance(target_ids, list)
        or len(target_ids) != 17
        or len(set(target_ids)) != 17
        or any(type(row_id) is not str or not row_id for row_id in target_ids)
    ):
        raise RecoveryBoundCoverageError("Continuation target row IDs drifted")
    static_expected = {
        "continuation_semantics": semantics,
        "target_ordinals": target_ordinals,
        "target_row_count": 17,
        "jobs": 4,
        "original_metric_timeout_seconds": 300,
        "operational_metric_timeout_seconds": 1200,
        "outer_row_timeout_seconds": 2400,
        "accounting_grace_seconds": 30,
        "plan_sha256": plan_sha,
        "source_snapshot_sha256": source_sha,
        "frozen_runner_sha256": frozen_sha,
        "independent_metric_source_sha256": metric_sha,
        "process_guard_sha256": process_guard_sha,
        "helper_sha256": FROZEN_CONTINUATION_HELPER_SHA256,
        "predecessor_helper_sha256": old_recovery_chain.get("helper_sha256"),
        "predecessor_claim_sha256": old_recovery_chain.get("claim_sha256"),
        "predecessor_failure_receipt_sha256": old_recovery_chain.get(
            "failure_receipt_sha256"
        ),
        "predecessor_failure_seal_sha256": old_recovery_chain.get(
            "failure_seal_sha256"
        ),
    }
    for field, expected in static_expected.items():
        if claim.get(field) != expected or receipt.get(field) != expected:
            raise RecoveryBoundCoverageError(
                f"Continuation claim/receipt binding drifted at {field}"
            )
    if (
        claim.get("scope")
        != "EXACT_ORDINALS_446_462_FROZEN_DEVELOPMENT_CONTINUATION_ONLY"
        or receipt.get("scope")
        != "EXACT_ORDINALS_446_462_FROZEN_DEVELOPMENT_CONTINUATION_ONLY"
        or claim.get("target_case_id")
        != "v21e3-motsp-development-n500-s00"
        or receipt.get("target_case_id")
        != "v21e3-motsp-development-n500-s00"
        or claim.get("old_complete_attempt_count") != 5
        or claim.get("old_complete_attempt_adopted_count") != 0
        or claim.get("old_complete_attempts_not_adopted") is not True
        or claim.get("fresh_full_algorithm_rerun_count") != 17
        or claim.get("original_main_runner_honors_this_claim") is not False
        or claim.get("automatic_resume_authorized") is not False
        or receipt.get("old_complete_attempt_count") != 5
        or receipt.get("old_complete_attempt_adopted_count") != 0
        or receipt.get("incident_complete_attempts_not_adopted") is not True
        or receipt.get("fresh_full_algorithm_rerun_count") != 17
        or receipt.get("predecessor_incident_immutable") is not True
        or receipt.get("original_diagnostic_receipt_alone_insufficient") is not True
        or receipt.get("final_completed_marker_count") != 504
        or receipt.get("aggregate_materialized") is not False
        or receipt.get("diagnostic_receipt_materialized") is not False
        or receipt.get("original_runner_resume_required") is not True
        or receipt.get("original_runner_resume_after_continuation_success_only")
        is not True
        or claim.get("interpreter_identity") != receipt.get("interpreter_identity")
        or claim.get("environment_receipt") != receipt.get("environment_receipt")
    ):
        raise RecoveryBoundCoverageError("Continuation static semantics drifted")

    external_manifest = _validate_file_manifest(
        diagnostic,
        receipt.get("external_scheduling_manifest"),
        label="continuation external evidence manifest",
        expected_length=4,
    )
    if claim.get("incident_receipt") != receipt.get("incident_receipt"):
        raise RecoveryBoundCoverageError("Continuation incident copies disagree")
    incident = _validate_exact_bound_mapping(
        receipt.get("incident_receipt"),
        label="continuation incident receipt",
        expected_keys=_CONTINUATION_INCIDENT_KEYS,
        digest_field="incident_payload_sha256",
        schema="v21e3r1_metric_timeout_recovery_continuation_incident_v1",
    )
    if incident.get("external_scheduling_manifest") != external_manifest:
        raise RecoveryBoundCoverageError("Continuation external manifest copies disagree")
    external_manifest_sha = _manifest_payload_sha256(
        external_manifest, label="continuation external evidence manifest"
    )
    if (
        incident.get("external_scheduling_manifest_sha256") != external_manifest_sha
        or receipt.get("external_scheduling_manifest") != external_manifest
    ):
        raise RecoveryBoundCoverageError("Continuation external manifest digest drifted")
    custody = _validate_exact_bound_mapping(
        receipt.get("external_scheduling_custody"),
        label="continuation external custody",
        expected_keys=_CONTINUATION_CUSTODY_KEYS,
        digest_field="custody_payload_sha256",
        schema="v21e3r1_external_scheduling_s01_custody_binding_v1",
        status="PASS_HASH_BOUND_EXTERNAL_SCHEDULING_ONLY_NO_NEW_AUTHORITY",
    )
    if incident.get("external_scheduling_custody") != custody:
        raise RecoveryBoundCoverageError("Continuation custody copies disagree")
    external_names = {
        "handoff_path": (
            "external-scheduling.v21e3-motsp-development-n500-s01."
            "main-driver-handoff.receipt.json"
        ),
        "claim_path": (
            "external-scheduling.v21e3-motsp-development-n500-s01."
            "helper-instance.claim.json"
        ),
        "receipt_path": (
            "external-scheduling.v21e3-motsp-development-n500-s01.receipt.json"
        ),
        "seal_path": (
            "external-scheduling.v21e3-motsp-development-n500-s01."
            "receipt.seal.json"
        ),
    }
    custody_expected = {
        "target_case_id": "v21e3-motsp-development-n500-s01",
        "target_ordinals": list(range(463, 505)),
        "target_row_count": 42,
        "external_helper_sha256": process_guard_sha,
        "handoff_sha256": external_chain.get("handoff_sha256"),
        "handoff_payload_sha256": external_chain.get("handoff_payload_sha256"),
        "claim_sha256": external_chain.get("claim_sha256"),
        "claim_payload_sha256": external_chain.get("claim_payload_sha256"),
        "receipt_sha256": external_chain.get("receipt_sha256"),
        "receipt_payload_sha256": external_chain.get("receipt_payload_sha256"),
        "seal_sha256": external_chain.get("seal_sha256"),
        "seal_payload_sha256": external_chain.get("seal_payload_sha256"),
        "completed_marker_count": 42,
        "completed_marker_manifest_sha256": hashlib.sha256(
            _canonical_bytes(external_chain.get("completed_markers"))
        ).hexdigest(),
        "external_evidence_manifest_sha256": external_manifest_sha,
    }
    for field, expected in {**external_names, **custody_expected}.items():
        if custody.get(field) != expected:
            raise RecoveryBoundCoverageError(
                f"Continuation external custody drifted at {field}"
            )

    incident_manifest = _validate_file_manifest(
        diagnostic,
        receipt.get("incident_file_manifest"),
        label="continuation predecessor incident manifest",
        expected_length=3,
    )
    expected_incident_hashes = {
        old_recovery_chain.get("claim_sha256"),
        old_recovery_chain.get("failure_receipt_sha256"),
        old_recovery_chain.get("failure_seal_sha256"),
    }
    if {entry["sha256"] for entry in incident_manifest} != expected_incident_hashes:
        raise RecoveryBoundCoverageError("Continuation predecessor incident files drifted")
    incident_names = {Path(str(entry["path"])).name for entry in incident_manifest}
    if incident_names != {
        "metric-timeout-recovery.v21e3-motsp-development-n500-s00."
        "helper-instance.claim.json",
        "metric-timeout-recovery.v21e3-motsp-development-n500-s00."
        "failure.receipt.json",
        "metric-timeout-recovery.v21e3-motsp-development-n500-s00."
        "failure.receipt.seal.json",
    }:
        raise RecoveryBoundCoverageError("Continuation incident filenames drifted")
    complete_manifest = _validate_file_manifest(
        diagnostic,
        receipt.get("incident_complete_attempt_manifest"),
        label="continuation complete predecessor attempts",
        expected_length=40,
    )
    expected_complete_manifest = sorted(
        [
            dict(artifact)
            for attempt in old_recovery_chain.get("old_complete_attempts", [])
            for artifact in attempt.get("artifacts", [])
        ],
        key=lambda item: item["path"],
    )
    if complete_manifest != expected_complete_manifest:
        raise RecoveryBoundCoverageError("Continuation adopted/drifted old attempts")
    failed_manifest = _validate_file_manifest(
        diagnostic,
        receipt.get("predecessor_failed_attempt_manifest"),
        label="continuation predecessor failed attempt",
    )
    if failed_manifest != old_recovery_chain.get("preexisting_failed_attempt_manifest"):
        raise RecoveryBoundCoverageError("Continuation predecessor failure drifted")
    completed_root = diagnostic / "completed"
    if completed_root.is_symlink() or not completed_root.is_dir():
        raise RecoveryBoundCoverageError("Continuation completed marker root is unsafe")
    completed_paths = sorted(completed_root.iterdir(), key=lambda path: path.name)
    if (
        len(completed_paths) != 504
        or any(path.is_symlink() or not path.is_file() or path.suffix != ".json" for path in completed_paths)
        or len({path.stem.casefold() for path in completed_paths}) != 504
    ):
        raise RecoveryBoundCoverageError("Continuation marker set is not exact504")
    target_id_set = set(target_ids)
    preserved_paths = [path for path in completed_paths if path.stem not in target_id_set]
    if len(preserved_paths) != 487:
        raise RecoveryBoundCoverageError("Continuation preserved marker complement drifted")
    preserved_manifest = _snapshot_file_manifest(diagnostic, preserved_paths)
    incident_expected = {
        "predecessor_helper_sha256": old_recovery_chain.get("helper_sha256"),
        "continuation_helper_sha256": FROZEN_CONTINUATION_HELPER_SHA256,
        "old_claim_sha256": old_recovery_chain.get("claim_sha256"),
        "old_failure_receipt_sha256": old_recovery_chain.get(
            "failure_receipt_sha256"
        ),
        "old_failure_seal_sha256": old_recovery_chain.get("failure_seal_sha256"),
        "preserved_marker_count": 487,
        "preserved_marker_manifest_sha256": _manifest_payload_sha256(
            preserved_manifest, label="preserved marker manifest"
        ),
        "incident_complete_attempt_count": 5,
        "incident_complete_attempt_adopted_count": 0,
        "incident_complete_attempts_not_adopted": True,
        "incident_complete_attempt_manifest_sha256": _manifest_payload_sha256(
            complete_manifest, label="complete predecessor attempt manifest"
        ),
        "predecessor_failed_attempt_manifest_sha256": _manifest_payload_sha256(
            failed_manifest, label="failed predecessor attempt manifest"
        ),
        "external_scheduling_manifest_sha256": external_manifest_sha,
        "missing_recovery_marker_count": 17,
        "fresh_full_algorithm_rerun_count": 17,
        "old_success_absent": True,
    }
    for field, expected in incident_expected.items():
        if incident.get(field) != expected:
            raise RecoveryBoundCoverageError(
                f"Continuation incident drifted at {field}"
            )
    _validate_continuation_incident_scan(
        incident.get("terminal_process_scan"), label="continuation incident scan"
    )
    _validate_continuation_incident_scan(
        claim.get("preclaim_process_scan"), label="continuation preclaim scan"
    )

    worker_manifest = claim.get("worker_spec_payload_manifest")
    if not isinstance(worker_manifest, list) or len(worker_manifest) != 17:
        raise RecoveryBoundCoverageError("Continuation worker manifest is not exact17")
    if claim.get("worker_spec_payload_manifest_sha256") != hashlib.sha256(
        _canonical_bytes(worker_manifest)
    ).hexdigest():
        raise RecoveryBoundCoverageError("Continuation worker manifest digest drifted")
    completed_rows = receipt.get("completed_rows")
    if not isinstance(completed_rows, list) or len(completed_rows) != 17:
        raise RecoveryBoundCoverageError("Continuation completed rows are not exact17")
    normalized_rows: list[dict[str, object]] = []
    for index, (raw_manifest, raw_entry) in enumerate(
        zip(worker_manifest, completed_rows, strict=True)
    ):
        expected_ordinal = target_ordinals[index]
        expected_row_id = target_ids[index]
        expected_attempt_number = 3 if expected_ordinal == 446 else 2 if expected_ordinal <= 450 else 1
        try:
            if not isinstance(raw_manifest, dict) or not isinstance(raw_entry, dict):
                raise RecoveryBoundCoverageError("Continuation row entry is not an object")
            _require_exact_keys(
                raw_manifest,
                _CONTINUATION_WORKER_MANIFEST_KEYS,
                label="continuation worker manifest entry",
            )
            _require_exact_keys(
                raw_entry,
                _CONTINUATION_COMPLETED_ROW_KEYS,
                label="continuation completed row entry",
            )
            if (
                raw_manifest.get("ordinal") != expected_ordinal
                or raw_manifest.get("row_id") != expected_row_id
                or raw_manifest.get("expected_attempt_number")
                != expected_attempt_number
                or raw_entry.get("ordinal") != expected_ordinal
                or raw_entry.get("row_id") != expected_row_id
            ):
                raise RecoveryBoundCoverageError("Continuation row order drifted")
            expected_attempt_relative = (
                Path("attempts")
                / expected_row_id
                / f"attempt-{expected_attempt_number:04d}"
            ).as_posix()
            if raw_entry.get("attempt_directory") != expected_attempt_relative:
                raise RecoveryBoundCoverageError("Continuation attempt mapping drifted")
            attempt = _contained_path(
                diagnostic,
                expected_attempt_relative,
                label="continuation fresh attempt",
            )
            if attempt.is_symlink() or not attempt.is_dir():
                raise RecoveryBoundCoverageError("Continuation fresh attempt is unsafe")
            spec_path = attempt / "worker.spec.json"
            result_path = attempt / "worker.result.json"
            timeout_path = attempt / "operational.metric-timeout-override.receipt.json"
            job_path = attempt / "continuation.windows-job.receipt.json"
            spec_payload, spec_raw_sha = _load_json_object(
                spec_path, label="continuation worker spec"
            )
            result_payload, result_raw_sha = _load_json_object(
                result_path, label="continuation worker result"
            )
            del result_payload
            spec_payload_sha = hashlib.sha256(_canonical_bytes(spec_payload)).hexdigest()
            if (
                raw_entry.get("worker_spec_sha256") != spec_raw_sha
                or raw_entry.get("worker_spec_payload_sha256") != spec_payload_sha
                or raw_manifest.get("worker_spec_payload_sha256") != spec_payload_sha
                or raw_entry.get("worker_result_sha256") != result_raw_sha
            ):
                raise RecoveryBoundCoverageError("Continuation spec/result hashes drifted")
            timeout, timeout_raw_sha = _validate_bound_payload(
                timeout_path,
                label="continuation timeout witness",
                expected_keys=_CONTINUATION_TIMEOUT_WITNESS_KEYS,
                digest_field="receipt_payload_sha256",
                schema="v21e3r1_metric_timeout_override_witness_v1",
                status="PASS_EXACT_ONE_INDEPENDENT_METRIC_TIMEOUT_OVERRIDE",
            )
            _require_hold_authority(timeout, label="continuation timeout witness")
            original_kwargs = timeout.get("original_subprocess_kwargs")
            effective_kwargs = timeout.get("effective_subprocess_kwargs")
            if not isinstance(original_kwargs, dict) or not isinstance(effective_kwargs, dict):
                raise RecoveryBoundCoverageError("Continuation timeout kwargs are absent")
            _require_exact_keys(
                original_kwargs, _SUBPROCESS_KWARGS_KEYS, label="original metric kwargs"
            )
            _require_exact_keys(
                effective_kwargs, _SUBPROCESS_KWARGS_KEYS, label="effective metric kwargs"
            )
            original_without_timeout = dict(original_kwargs)
            effective_without_timeout = dict(effective_kwargs)
            original_timeout = original_without_timeout.pop("timeout")
            effective_timeout = effective_without_timeout.pop("timeout")
            command = timeout.get("independent_metric_command")
            if (
                timeout.get("scope")
                != "EXACT_FROZEN_DEVELOPMENT_RECOVERY_ROW_ONLY"
                or timeout.get("recovery_semantics")
                != "SAME_ALGORITHM_AND_METRIC_CODE_OPERATIONAL_TIMEOUT_OVERRIDE_ONLY"
                or not isinstance(command, list)
                or not command
                or any(type(argument) is not str or not argument for argument in command)
                or timeout.get("fresh_full_algorithm_rerun") is not True
                or timeout.get("preexisting_failed_trace_reused") is not False
                or timeout.get("original_diagnostic_receipt_alone_insufficient")
                is not True
                or timeout.get("row_id") != expected_row_id
                or timeout.get("worker_spec_path") != "worker.spec.json"
                or timeout.get("worker_spec_sha256") != spec_raw_sha
                or timeout.get("worker_result_path") != "worker.result.json"
                or timeout.get("worker_result_sha256") != result_raw_sha
                or timeout.get("independent_metric_command_sha256")
                != hashlib.sha256(_canonical_bytes(command)).hexdigest()
                or timeout.get("subprocess_call_count") != 1
                or timeout.get("subprocess_returncode") != 0
                or original_timeout != 300
                or effective_timeout != 1200
                or original_without_timeout != effective_without_timeout
                or original_kwargs.get("text") is not True
                or original_kwargs.get("capture_output") is not True
                or original_kwargs.get("check") is not False
                or timeout.get("plan_sha256") != plan_sha
                or timeout.get("source_snapshot_sha256") != source_sha
                or timeout.get("frozen_runner_sha256") != frozen_sha
                or timeout.get("independent_metric_source_sha256") != metric_sha
                or timeout.get("helper_sha256") != FROZEN_CONTINUATION_HELPER_SHA256
                or timeout.get("interpreter_identity")
                != receipt.get("interpreter_identity")
                or raw_entry.get("timeout_witness_sha256") != timeout_raw_sha
                or raw_entry.get("timeout_witness_payload_sha256")
                != timeout.get("receipt_payload_sha256")
            ):
                raise RecoveryBoundCoverageError("Continuation timeout semantics drifted")
            job, job_raw_sha = _validate_bound_payload(
                job_path,
                label="continuation Windows Job receipt",
                expected_keys=_CONTINUATION_JOB_RECEIPT_KEYS,
                digest_field="receipt_payload_sha256",
                schema="v21e3r1_continuation_row_windows_job_receipt_v1",
                status="PASS_CORRECTED_WINDOWS_JOB_CONTAINMENT_AND_TERMINAL_ZERO",
            )
            _require_hold_authority(job, label="continuation Windows Job receipt")
            if (
                raw_entry.get("windows_job_receipt_path")
                != job_path.relative_to(diagnostic).as_posix()
                or raw_entry.get("windows_job_receipt_sha256") != job_raw_sha
                or raw_entry.get("windows_job_receipt_payload_sha256")
                != job.get("receipt_payload_sha256")
                or raw_entry.get("windows_job_receipt") != job
                or job.get("ordinal") != expected_ordinal
                or job.get("row_id") != expected_row_id
                or job.get("attempt_directory") != expected_attempt_relative
                or job.get("worker_spec_sha256") != spec_raw_sha
                or job.get("worker_result_sha256") != result_raw_sha
                or job.get("timeout_witness_sha256") != timeout_raw_sha
                or job.get("helper_sha256") != FROZEN_CONTINUATION_HELPER_SHA256
                or job.get("helper_instance_claim_sha256") != claim_sha
                or job.get("predecessor_complete_attempt_adopted") is not False
                or job.get("continuation_semantics") != semantics
                or job.get("scope")
                != "ONE_FRESH_FROZEN_DEVELOPMENT_RECOVERY_RERUN_ONLY"
            ):
                raise RecoveryBoundCoverageError("Continuation Job receipt binding drifted")
            job_control = job.get("job_control")
            if not isinstance(job_control, dict):
                raise RecoveryBoundCoverageError("Continuation Job control is absent")
            _require_exact_keys(
                job_control,
                _CONTINUATION_JOB_CONTROL_KEYS,
                label="continuation Job control",
            )
            initial_active = job_control.get("initial_active_processes_after_wrapper_exit")
            wait_seconds = job_control.get("accounting_wait_seconds")
            if (
                job_control.get("schema")
                != "v21e3r1_continuation_windows_job_witness_v1"
                or job_control.get("kill_on_job_close_limit") is not True
                or job_control.get("job_limit_flags") != 0x2000
                or type(job_control.get("wrapper_pid")) is not int
                or job_control.get("wrapper_pid") <= 0
                or job_control.get("job_assignment_verified_before_gate_release")
                is not True
                or job_control.get("outer_timeout_seconds") != 2400
                or job_control.get("outer_timeout_fired") is not False
                or job_control.get("wrapper_returncode") != 0
                or type(initial_active) is not int
                or initial_active < 0
                or job_control.get("accounting_grace_seconds") != 30
                or job_control.get("accounting_grace_expired") is not False
                or type(wait_seconds) not in {int, float}
                or wait_seconds < 0
                or job_control.get("terminal_active_processes") != 0
                or (
                    initial_active > 0
                    and (
                        job_control.get("accounting_lag_observed") is not True
                        or job_control.get(
                            "accounting_lag_drained_without_termination"
                        )
                        is not True
                    )
                )
                or (
                    initial_active == 0
                    and (
                        job_control.get("accounting_lag_observed") is not False
                        or job_control.get(
                            "accounting_lag_drained_without_termination"
                        )
                        is not False
                    )
                )
            ):
                raise RecoveryBoundCoverageError("Continuation Job accounting drifted")
            terminal_scan = _validate_exact_bound_mapping(
                job_control.get("terminal_process_scan"),
                label="continuation terminal worker scan",
                expected_keys=_CONTINUATION_TERMINAL_SCAN_KEYS,
                digest_field="scan_payload_sha256",
                schema="v21e3r1_recovery_continuation_descendant_zero_v1",
            )
            if (
                terminal_scan.get("worker_specs") != [spec_path.resolve().as_posix()]
                or terminal_scan.get("block_all") is not False
                or type(terminal_scan.get("scan_count")) is not int
                or terminal_scan.get("scan_count") <= 0
                or terminal_scan.get("terminal_matching_process_count") != 0
            ):
                raise RecoveryBoundCoverageError("Continuation terminal scan drifted")
            _require_sha256(
                terminal_scan.get("original_process_scan_payload_sha256"),
                label="continuation original process scan SHA-256",
            )
            marker_path = _contained_path(
                diagnostic,
                raw_entry.get("completed_marker_path"),
                label="continuation completed marker",
            )
            marker, marker_raw_sha = _load_json_object(
                marker_path, label="continuation completed marker"
            )
            _require_exact_keys(
                marker,
                _DIAGNOSTIC_COMPLETED_MARKER_KEYS,
                label="continuation completed marker",
            )
            if (
                marker_path != completed_root / f"{expected_row_id}.json"
                or raw_entry.get("completed_marker_sha256") != marker_raw_sha
                or marker.get("row_id") != expected_row_id
                or marker.get("attempt_directory") != expected_attempt_relative
                or marker.get("plan_sha256") != plan_sha
                or marker.get("status") != "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY"
            ):
                raise RecoveryBoundCoverageError("Continuation marker binding drifted")
            for field in (
                "diagnostic_sha256",
                "independent_metric_receipt_sha256",
                "row_sha256",
                "terminal_receipt_sha256",
                "trace_sha256",
            ):
                _require_sha256(marker.get(field), label=f"continuation marker {field}")
            normalized_rows.append(
                {
                    "ordinal": expected_ordinal,
                    "row_id": expected_row_id,
                    "attempt_directory": expected_attempt_relative,
                    "completed_marker_path": marker_path.relative_to(
                        diagnostic
                    ).as_posix(),
                    "completed_marker_sha256": marker_raw_sha,
                    "timeout_witness_sha256": timeout_raw_sha,
                    "windows_job_receipt_sha256": job_raw_sha,
                }
            )
        except (OSError, ValueError, KeyError, RecoveryBoundCoverageError) as error:
            raise RecoveryBoundCoverageError(
                f"Continuation fresh-row evidence drifted: {expected_row_id}"
            ) from error
    if (
        seal.get("receipt_path") != Path(receipt_path).name
        or seal.get("receipt_sha256") != receipt_sha
        or seal.get("receipt_payload_sha256") != receipt.get("receipt_payload_sha256")
        or seal.get("helper_instance_claim_sha256") != claim_sha
        or seal.get("predecessor_failure_receipt_sha256")
        != old_recovery_chain.get("failure_receipt_sha256")
        or seal.get("external_scheduling_receipt_sha256")
        != external_chain.get("receipt_sha256")
        or seal.get("external_scheduling_success_seal_sha256")
        != external_chain.get("seal_sha256")
        or seal.get("external_scheduling_custody_payload_sha256")
        != custody.get("custody_payload_sha256")
    ):
        raise RecoveryBoundCoverageError("Continuation success seal chain drifted")
    if sha256_file(helper_path) != FROZEN_CONTINUATION_HELPER_SHA256:
        raise RecoveryBoundCoverageError("Continuation helper drifted after validation")
    return {
        "schema": "v21e3r1_recovery_bound_continuation_binding_v1",
        "status": "PASS_FRESH17_CONTINUATION_CHAIN_BOUND_INPUT_ONLY",
        "target_ordinals": target_ordinals,
        "target_row_ids": list(target_ids),
        "fresh_full_algorithm_rerun_count": 17,
        "completed_rows": normalized_rows,
        "final_completed_marker_count": 504,
        "helper_sha256": FROZEN_CONTINUATION_HELPER_SHA256,
        "claim_sha256": claim_sha,
        "claim_payload_sha256": claim["claim_payload_sha256"],
        "receipt_sha256": receipt_sha,
        "receipt_payload_sha256": receipt["receipt_payload_sha256"],
        "seal_sha256": seal_sha,
        "seal_payload_sha256": seal["seal_payload_sha256"],
        "external_scheduling_custody_payload_sha256": custody[
            "custody_payload_sha256"
        ],
        "plan_sha256": plan_sha,
        "source_snapshot_sha256": source_sha,
        "frozen_runner_sha256": frozen_sha,
        "independent_metric_source_sha256": metric_sha,
        "process_guard_sha256": process_guard_sha,
        "implementation_independence": False,
        "scientific_independence": False,
        "runtime_authority": False,
        "scientific_authority": False,
        "selection_authority": False,
        "confirmation_authority": False,
        "formal_study_authority": False,
        "publication_status": "IJOC_HOLD",
    }


def build_recovered_diagnostic_provenance_binding(
    *,
    external_chain: Mapping[str, object],
    old_recovery_chain: Mapping[str, object],
    continuation_chain: Mapping[str, object],
    diagnostic_binding: Mapping[str, object],
) -> dict[str, object]:
    """Cross-bind all recovery segments to the resumed exact504 tree."""

    expected_statuses = (
        (
            external_chain,
            "v21e3r1_recovery_bound_external_scheduling_binding_v1",
            "PASS_EXTERNAL_SCHEDULING_CHAIN_BOUND_INPUT_ONLY",
            "external scheduling binding",
        ),
        (
            old_recovery_chain,
            "v21e3r1_recovery_bound_old_failure_binding_v1",
            "PASS_OLD_RECOVERY_FAILURE_CHAIN_BOUND_NOT_ADOPTED",
            "old recovery binding",
        ),
        (
            continuation_chain,
            "v21e3r1_recovery_bound_continuation_binding_v1",
            "PASS_FRESH17_CONTINUATION_CHAIN_BOUND_INPUT_ONLY",
            "continuation binding",
        ),
        (
            diagnostic_binding,
            "v21e3r1_recovery_bound_exact504_diagnostic_binding_v1",
            "PASS_EXACT504_DIAGNOSTIC_TREE_REVALIDATED_INPUT_ONLY",
            "exact504 diagnostic binding",
        ),
    )
    for payload, schema, status, label in expected_statuses:
        if payload.get("schema") != schema or payload.get("status") != status:
            raise RecoveryBoundCoverageError(f"{label} schema/status drifted")
        _require_hold_authority(payload, label=label)
        if (
            payload.get("implementation_independence") is not False
            or payload.get("scientific_independence") is not False
        ):
            raise RecoveryBoundCoverageError(f"{label} expands independence")
    plan_sha = _require_sha256(
        diagnostic_binding.get("plan_sha256"), label="diagnostic plan SHA-256"
    )
    source_sha = _require_sha256(
        diagnostic_binding.get("source_snapshot_sha256"),
        label="diagnostic source SHA-256",
    )
    frozen_sha = _require_sha256(
        diagnostic_binding.get("frozen_diagnostic_runner_sha256"),
        label="frozen diagnostic runner SHA-256",
    )
    metric_sha = _require_sha256(
        diagnostic_binding.get("independent_metric_source_sha256"),
        label="independent metric SHA-256",
    )
    for label, payload in (
        ("external scheduling binding", external_chain),
        ("old recovery binding", old_recovery_chain),
        ("continuation binding", continuation_chain),
    ):
        if (
            payload.get("plan_sha256") != plan_sha
            or payload.get("source_snapshot_sha256") != source_sha
            or payload.get("frozen_runner_sha256") != frozen_sha
        ):
            raise RecoveryBoundCoverageError(f"{label} source identity drifted")
    if (
        old_recovery_chain.get("independent_metric_source_sha256") != metric_sha
        or continuation_chain.get("independent_metric_source_sha256") != metric_sha
        or continuation_chain.get("process_guard_sha256")
        != old_recovery_chain.get("process_guard_sha256")
        or continuation_chain.get("target_row_ids")
        != old_recovery_chain.get("target_row_ids")
    ):
        raise RecoveryBoundCoverageError("Recovery segment identity drifted")
    rows = diagnostic_binding.get("rows")
    if (
        diagnostic_binding.get("expected_rows") != 504
        or not isinstance(rows, list)
        or len(rows) != 504
    ):
        raise RecoveryBoundCoverageError("Diagnostic binding is not exact504")
    row_by_ordinal: dict[int, Mapping[str, object]] = {}
    for expected_ordinal, raw in enumerate(rows, start=1):
        if (
            not isinstance(raw, dict)
            or raw.get("ordinal") != expected_ordinal
            or type(raw.get("row_id")) is not str
            or not raw.get("row_id")
        ):
            raise RecoveryBoundCoverageError("Diagnostic row order drifted")
        for field in (
            "completed_marker_sha256",
            "trace_sha256",
            "case_sha256",
        ):
            _require_sha256(raw.get(field), label=f"diagnostic row {field}")
        row_by_ordinal[expected_ordinal] = raw
    if len({str(row["row_id"]).casefold() for row in rows}) != 504:
        raise RecoveryBoundCoverageError("Diagnostic row IDs are not unique")
    continuation_rows = continuation_chain.get("completed_rows")
    if not isinstance(continuation_rows, list) or len(continuation_rows) != 17:
        raise RecoveryBoundCoverageError("Continuation binding is not exact17")
    for expected_ordinal, entry in zip(
        range(446, 463), continuation_rows, strict=True
    ):
        diagnostic_row = row_by_ordinal[expected_ordinal]
        if (
            not isinstance(entry, dict)
            or entry.get("ordinal") != expected_ordinal
            or entry.get("row_id") != diagnostic_row.get("row_id")
            or entry.get("attempt_directory")
            != diagnostic_row.get("attempt_directory")
            or entry.get("completed_marker_path")
            != diagnostic_row.get("completed_marker_path")
            or entry.get("completed_marker_sha256")
            != diagnostic_row.get("completed_marker_sha256")
        ):
            raise RecoveryBoundCoverageError(
                "Continuation-to-diagnostic row binding drifted"
            )
    external_rows = external_chain.get("completed_markers")
    if not isinstance(external_rows, list) or len(external_rows) != 42:
        raise RecoveryBoundCoverageError("External scheduling binding is not exact42")
    for expected_ordinal, entry in zip(range(463, 505), external_rows, strict=True):
        diagnostic_row = row_by_ordinal[expected_ordinal]
        if (
            not isinstance(entry, dict)
            or entry.get("ordinal") != expected_ordinal
            or entry.get("row_id") != diagnostic_row.get("row_id")
            or entry.get("attempt_directory")
            != diagnostic_row.get("attempt_directory")
            or entry.get("path") != diagnostic_row.get("completed_marker_path")
            or entry.get("sha256") != diagnostic_row.get("completed_marker_sha256")
        ):
            raise RecoveryBoundCoverageError(
                "External-scheduling-to-diagnostic row binding drifted"
            )
    for field in (
        "diagnostic_receipt_sha256",
        "diagnostic_aggregate_sha256",
        "inner_coverage_runner_sha256",
    ):
        _require_sha256(diagnostic_binding.get(field), label=f"diagnostic {field}")
    external_digest = hashlib.sha256(_canonical_bytes(external_chain)).hexdigest()
    old_digest = hashlib.sha256(_canonical_bytes(old_recovery_chain)).hexdigest()
    continuation_digest = hashlib.sha256(
        _canonical_bytes(continuation_chain)
    ).hexdigest()
    diagnostic_digest = hashlib.sha256(
        _canonical_bytes(diagnostic_binding)
    ).hexdigest()
    return _bind_payload(
        {
            "schema": "v21e3r1_recovered_diagnostic_provenance_binding_v1",
            "status": "PASS_CHAIN_VALIDATED_FOR_REPLAY_INPUT_ONLY",
            "expected_rows": 504,
            "continuation_row_count": 17,
            "external_scheduling_row_count": 42,
            "plan_sha256": plan_sha,
            "source_snapshot_sha256": source_sha,
            "frozen_diagnostic_runner_sha256": frozen_sha,
            "independent_metric_source_sha256": metric_sha,
            "process_guard_sha256": continuation_chain[
                "process_guard_sha256"
            ],
            "continuation_helper_sha256": continuation_chain["helper_sha256"],
            "external_scheduling_binding_sha256": external_digest,
            "old_recovery_failure_binding_sha256": old_digest,
            "continuation_binding_sha256": continuation_digest,
            "exact504_diagnostic_binding_sha256": diagnostic_digest,
            "external_scheduling_binding": dict(external_chain),
            "old_recovery_failure_binding": dict(old_recovery_chain),
            "continuation_binding": dict(continuation_chain),
            "exact504_diagnostic_binding": dict(diagnostic_binding),
            "same_implementation_only": True,
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
        "provenance_payload_sha256",
    )


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RecoveryBoundCoverageError(f"{label} is not lowercase SHA-256")
    return value


@dataclass(frozen=True)
class RuntimeIdentities:
    outer_runner_path: str
    outer_runner_sha256: str
    inner_runner_path: str
    inner_runner_sha256: str
    python_path: str
    python_sha256: str
    python_version: str


def validate_runtime_identities(
    *,
    outer_runner_path: str | Path,
    inner_runner_path: str | Path,
    expected_outer_runner_sha256: str,
    expected_inner_runner_sha256: str,
    expected_python_path: str | Path,
    expected_python_sha256: str,
) -> RuntimeIdentities:
    outer = Path(outer_runner_path).resolve()
    inner = Path(inner_runner_path).resolve()
    expected_python = Path(expected_python_path).resolve()
    actual_python = Path(sys.executable).resolve()
    expected_outer_sha = _require_sha256(
        expected_outer_runner_sha256, label="expected outer runner SHA-256"
    )
    expected_inner_sha = _require_sha256(
        expected_inner_runner_sha256, label="expected inner runner SHA-256"
    )
    expected_python_sha = _require_sha256(
        expected_python_sha256, label="expected Python SHA-256"
    )
    actual_outer_sha = sha256_file(outer)
    if actual_outer_sha != expected_outer_sha:
        raise RecoveryBoundCoverageError("The outer runner identity drifted")
    actual_inner_sha = sha256_file(inner)
    if actual_inner_sha != expected_inner_sha:
        raise RecoveryBoundCoverageError("The inner runner identity drifted")
    if actual_python != expected_python:
        raise RecoveryBoundCoverageError("The active Python path drifted")
    actual_python_sha = sha256_file(actual_python)
    if actual_python_sha != expected_python_sha:
        raise RecoveryBoundCoverageError("The active Python identity drifted")
    return RuntimeIdentities(
        outer_runner_path=outer.as_posix(),
        outer_runner_sha256=actual_outer_sha,
        inner_runner_path=inner.as_posix(),
        inner_runner_sha256=actual_inner_sha,
        python_path=actual_python.as_posix(),
        python_sha256=actual_python_sha,
        python_version=sys.version,
    )


def _bind_payload(payload: Mapping[str, object], digest_field: str) -> dict[str, object]:
    if digest_field in payload:
        raise RecoveryBoundCoverageError(f"Duplicate payload digest field: {digest_field}")
    bound = dict(payload)
    bound[digest_field] = hashlib.sha256(_canonical_bytes(bound)).hexdigest()
    return bound


def _validate_in_memory_bound_payload(
    payload: Mapping[str, object],
    *,
    label: str,
    digest_field: str,
    schema: str,
    status: str,
) -> str:
    if payload.get("schema") != schema or payload.get("status") != status:
        raise RecoveryBoundCoverageError(f"{label} schema/status drifted")
    declared = _require_sha256(payload.get(digest_field), label=f"{label} digest")
    core = dict(payload)
    del core[digest_field]
    if hashlib.sha256(_canonical_bytes(core)).hexdigest() != declared:
        raise RecoveryBoundCoverageError(f"{label} payload digest drifted")
    _require_hold_authority(payload, label=label)
    return declared


def _validate_n500_preflight_inputs(
    *,
    provenance: Mapping[str, object],
    diagnostic_binding: Mapping[str, object],
    row_id: str,
    row_timeout_seconds: int,
) -> tuple[dict[str, object], str, str]:
    provenance_digest = _validate_in_memory_bound_payload(
        provenance,
        label="recovered diagnostic provenance",
        digest_field="provenance_payload_sha256",
        schema="v21e3r1_recovered_diagnostic_provenance_binding_v1",
        status="PASS_CHAIN_VALIDATED_FOR_REPLAY_INPUT_ONLY",
    )
    if provenance.get("expected_rows") != 504:
        raise RecoveryBoundCoverageError("Recovered diagnostic provenance is not exact504")
    if (
        diagnostic_binding.get("schema")
        != "v21e3r1_recovery_bound_exact504_diagnostic_binding_v1"
        or diagnostic_binding.get("status")
        != "PASS_EXACT504_DIAGNOSTIC_TREE_REVALIDATED_INPUT_ONLY"
        or diagnostic_binding.get("expected_rows") != 504
    ):
        raise RecoveryBoundCoverageError("Diagnostic binding is not exact504")
    _require_hold_authority(diagnostic_binding, label="exact504 diagnostic binding")
    plan_sha = _require_sha256(
        provenance.get("plan_sha256"), label="provenance plan SHA-256"
    )
    source_sha = _require_sha256(
        provenance.get("source_snapshot_sha256"),
        label="provenance source SHA-256",
    )
    if (
        diagnostic_binding.get("plan_sha256") != plan_sha
        or diagnostic_binding.get("source_snapshot_sha256") != source_sha
    ):
        raise RecoveryBoundCoverageError(
            "Diagnostic binding disagrees with recovered provenance"
        )
    if type(row_id) is not str or not row_id:
        raise RecoveryBoundCoverageError("Preflight row ID must be nonempty text")
    if type(row_timeout_seconds) is not int or row_timeout_seconds != 2400:
        raise RecoveryBoundCoverageError("Preflight timeout must be exactly 2400 seconds")
    rows = diagnostic_binding.get("rows")
    if not isinstance(rows, list):
        raise RecoveryBoundCoverageError("Diagnostic binding rows are absent")
    matches = [row for row in rows if isinstance(row, dict) and row.get("row_id") == row_id]
    if len(matches) != 1:
        raise RecoveryBoundCoverageError("Preflight row is not unique in diagnostic binding")
    row = dict(matches[0])
    if (
        row.get("size") != 500
        or row.get("budget") != 2000
        or row.get("family") not in {"MOKP", "MOTSP"}
        or type(row.get("ordinal")) is not int
    ):
        raise RecoveryBoundCoverageError(
            "Preflight row is not a supported n500 full-budget row"
        )
    for field in (
        "completed_marker_sha256",
        "trace_sha256",
        "case_sha256",
    ):
        _require_sha256(row.get(field), label=f"preflight row {field}")
    for field in (
        "attempt_directory",
        "completed_marker_path",
        "trace_path",
        "case_path",
    ):
        if type(row.get(field)) is not str or not row[field]:
            raise RecoveryBoundCoverageError(f"Preflight row {field} is invalid")
    diagnostic_digest = hashlib.sha256(_canonical_bytes(diagnostic_binding)).hexdigest()
    return row, provenance_digest, diagnostic_digest


def _write_n500_preflight_failure(
    *,
    output: Path,
    plan_path: Path,
    plan_raw_sha256: str,
    plan_payload_sha256: str,
    row: Mapping[str, object],
    provenance_payload_sha256: str,
    diagnostic_binding_sha256: str,
    identities_before: RuntimeIdentities,
    failure_phase: str,
    error: BaseException,
) -> None:
    failure_path = output / "n500_preflight.failure.receipt.json"
    seal_path = output / "n500_preflight.failure.receipt.seal.json"
    if failure_path.exists() or seal_path.exists():
        raise RecoveryBoundCoverageError(
            "A preflight failure record already owns this output root"
        ) from error
    partial_manifest = tree_manifest(output)
    failure = _bind_payload(
        {
            "schema": "v21e3r1_recovery_bound_n500_preflight_failure_receipt_v1",
            "status": "HOLD_N500_OPERATIONAL_PREFLIGHT_FAILED",
            "failure_phase": failure_phase,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "plan_path": plan_path.name,
            "plan_sha256": plan_raw_sha256,
            "plan_payload_sha256": plan_payload_sha256,
            "selected_row": dict(row),
            "diagnostic_provenance_payload_sha256": provenance_payload_sha256,
            "diagnostic_binding_sha256": diagnostic_binding_sha256,
            "runtime_identities_before": asdict(identities_before),
            "partial_output_manifest_before_failure_receipt": partial_manifest,
            "automatic_retry_authorized": False,
            "full_coverage_authorized": False,
            "implementation_independence": False,
            "scientific_independence": False,
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
            "confirmation_authority": False,
            "formal_study_authority": False,
            "publication_status": "IJOC_HOLD",
        },
        "failure_receipt_payload_sha256",
    )
    failure_raw_sha = _exclusive_json(failure_path, failure)
    seal = _bind_payload(
        {
            "schema": "v21e3r1_recovery_bound_n500_preflight_failure_seal_v1",
            "status": "SEALED_N500_OPERATIONAL_PREFLIGHT_FAILURE_RECEIPT",
            "failure_receipt_path": failure_path.name,
            "failure_receipt_sha256": failure_raw_sha,
            "failure_receipt_payload_sha256": failure[
                "failure_receipt_payload_sha256"
            ],
            "plan_sha256": plan_raw_sha256,
            "automatic_retry_authorized": False,
            "full_coverage_authorized": False,
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
            "confirmation_authority": False,
            "formal_study_authority": False,
            "publication_status": "IJOC_HOLD",
        },
        "seal_payload_sha256",
    )
    _exclusive_json(seal_path, seal)


def _validate_n500_preflight_result(
    *,
    result: object,
    output: Path,
    identities_before: RuntimeIdentities,
    inner_runner_path: str | Path,
    expected_outer_runner_sha256: str,
    expected_inner_runner_sha256: str,
    expected_python_path: str | Path,
    expected_python_sha256: str,
) -> tuple[float | int, str, str, str, str, RuntimeIdentities]:
    if not isinstance(result, dict):
        raise RecoveryBoundCoverageError("Preflight executor result is not an object")
    if result.get("returncode") != 0:
        raise RecoveryBoundCoverageError("Preflight replay did not exit successfully")
    wall_time = result.get("wall_time_seconds")
    if type(wall_time) not in {int, float} or wall_time < 0:
        raise RecoveryBoundCoverageError("Preflight wall time is invalid")
    process_isolation = result.get("process_isolation")
    if type(process_isolation) is not str or not process_isolation:
        raise RecoveryBoundCoverageError("Preflight process isolation is absent")
    branch_path_value = result.get("branch_replay_receipt_path")
    if not isinstance(branch_path_value, (str, Path)):
        raise RecoveryBoundCoverageError("Preflight branch receipt path is absent")
    branch_path = Path(branch_path_value).resolve()
    try:
        branch_relative = branch_path.relative_to(output).as_posix()
    except ValueError as error:
        raise RecoveryBoundCoverageError("Preflight branch receipt escaped output") from error
    branch_payload, branch_raw_sha = _load_json_object(
        branch_path, label="preflight branch replay receipt"
    )
    if branch_raw_sha != _require_sha256(
        result.get("branch_replay_receipt_sha256"),
        label="preflight branch receipt SHA-256",
    ):
        raise RecoveryBoundCoverageError("Preflight branch receipt raw digest drifted")
    branch_payload_sha = _validate_in_memory_bound_payload(
        branch_payload,
        label="preflight branch replay receipt",
        digest_field="receipt_payload_sha256",
        schema=str(branch_payload.get("schema")),
        status="PASS_SAME_IMPLEMENTATION_FULL_BRANCH_REEXECUTION",
    )
    if branch_payload_sha != _require_sha256(
        result.get("branch_replay_payload_sha256"),
        label="preflight branch payload SHA-256",
    ):
        raise RecoveryBoundCoverageError("Preflight branch payload digest drifted")
    identities_after = validate_runtime_identities(
        outer_runner_path=Path(__file__).resolve(),
        inner_runner_path=inner_runner_path,
        expected_outer_runner_sha256=expected_outer_runner_sha256,
        expected_inner_runner_sha256=expected_inner_runner_sha256,
        expected_python_path=expected_python_path,
        expected_python_sha256=expected_python_sha256,
    )
    if identities_after != identities_before:
        raise RecoveryBoundCoverageError("Preflight runtime identities changed")
    return (
        wall_time,
        process_isolation,
        branch_relative,
        branch_raw_sha,
        branch_payload_sha,
        identities_after,
    )


def _execute_real_n500_preflight(
    *,
    project_root: Path,
    diagnostic_root: Path,
    output_root: Path,
    row: Mapping[str, object],
    row_timeout_seconds: int,
    expected_plan_sha256: str,
    expected_source_sha256: str,
    identities: RuntimeIdentities,
    inner_module: Any | None = None,
) -> dict[str, object]:
    """Delegate exactly one n500/2000 row to the unchanged inner-v1 worker."""

    project = Path(project_root).resolve()
    diagnostic = Path(diagnostic_root).resolve()
    output = Path(output_root).resolve()
    inner_path = Path(identities.inner_runner_path).resolve()
    if (
        row_timeout_seconds != 2400
        or sha256_file(inner_path) != identities.inner_runner_sha256
    ):
        raise RecoveryBoundCoverageError("Production preflight execution identity drifted")
    module = (
        inner_module
        if inner_module is not None
        else _load_module(inner_path, "_v21e3r1_recovery_bound_preflight_inner_v1")
    )
    for name in (
        "_load_plan_contract",
        "_process_one_row",
        "_validate_coverage_completed",
    ):
        if not callable(getattr(module, name, None)):
            raise RecoveryBoundCoverageError("Inner runner omits preflight boundary")
    contract = module._load_plan_contract(
        project_root=project,
        diagnostic_plan_path=diagnostic / "diagnostic.plan.json",
        allow_smoke=False,
    )
    if (
        not isinstance(contract, dict)
        or contract.get("exact_full") is not True
        or contract.get("expected_rows") != 504
    ):
        raise RecoveryBoundCoverageError("Inner preflight contract is not exact504")
    plan_sha = _require_sha256(
        contract.get("plan_sha256"), label="inner preflight plan SHA-256"
    )
    source_sha = _require_sha256(
        contract.get("source_root"), label="inner preflight source SHA-256"
    )
    if (
        plan_sha
        != _require_sha256(
            expected_plan_sha256, label="expected preflight plan SHA-256"
        )
        or source_sha
        != _require_sha256(
            expected_source_sha256, label="expected preflight source SHA-256"
        )
    ):
        raise RecoveryBoundCoverageError(
            "Inner preflight plan/source disagrees with recovered provenance"
        )
    source_entries = contract.get("source_entries")
    row_specs = contract.get("row_specs")
    if not isinstance(source_entries, list) or not isinstance(row_specs, list):
        raise RecoveryBoundCoverageError("Inner preflight contract is incomplete")
    matches = [
        candidate
        for candidate in row_specs
        if isinstance(candidate, dict) and candidate.get("row_id") == row.get("row_id")
    ]
    if len(matches) != 1:
        raise RecoveryBoundCoverageError("Inner preflight row is not unique")
    row_spec = matches[0]
    case_path = Path(str(row_spec.get("case_path"))).resolve()
    expected_case_path = Path(str(row.get("case_path"))).resolve()
    if (
        row_spec.get("ordinal") != row.get("ordinal")
        or row_spec.get("family") != row.get("family")
        or row_spec.get("size") != 500
        or row_spec.get("budget") != 2000
        or row.get("budget") != 2000
        or case_path != expected_case_path
        or row_spec.get("case_sha256") != row.get("case_sha256")
    ):
        raise RecoveryBoundCoverageError("Inner preflight row binding drifted")
    branch_root = output / "inner_v1_n500_preflight"
    os.mkdir(branch_root)
    _fsync_directory(output)
    completed_root = branch_root / "completed"
    os.mkdir(completed_root)
    _fsync_directory(branch_root)
    source_manifest = {
        "schema": "v21e3r1_branch_replay_source_manifest_binding_v1",
        "source_root_sha256": source_sha,
        "entries": source_entries,
    }
    source_manifest_path = branch_root / "source.manifest.json"
    source_manifest_sha = _exclusive_json(source_manifest_path, source_manifest)
    started = time.monotonic()
    completed = module._process_one_row(
        project_root=project,
        diagnostic_root=diagnostic,
        coverage_root=branch_root,
        row_spec=row_spec,
        diagnostic_plan_sha256=plan_sha,
        source_root=source_sha,
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=source_manifest_sha,
        jobs=1,
        timeout_seconds=2400,
    )
    wall_time = time.monotonic() - started
    if not isinstance(completed, dict) or completed.get("row_id") != row.get("row_id"):
        raise RecoveryBoundCoverageError("Inner preflight completion drifted")
    module._validate_coverage_completed(
        coverage_root=branch_root,
        diagnostic_root=diagnostic,
        row_spec=row_spec,
        diagnostic_plan_sha256=plan_sha,
        source_root=source_sha,
        source_manifest_sha256=source_manifest_sha,
    )
    attempt = _contained_path(
        branch_root,
        completed.get("attempt_directory"),
        label="preflight inner attempt",
    )
    branch_receipt_path = attempt / "branch.replay.json"
    branch_receipt, branch_receipt_sha = _load_json_object(
        branch_receipt_path, label="preflight inner branch receipt"
    )
    branch_payload_sha = _validate_in_memory_bound_payload(
        branch_receipt,
        label="preflight inner branch receipt",
        digest_field="receipt_payload_sha256",
        schema="v21e3r1_same_implementation_branch_replay_v1",
        status="PASS_SAME_IMPLEMENTATION_FULL_BRANCH_REEXECUTION",
    )
    if sha256_file(inner_path) != identities.inner_runner_sha256:
        raise RecoveryBoundCoverageError("Inner runner drifted during preflight")
    return {
        "branch_replay_receipt_path": branch_receipt_path,
        "branch_replay_receipt_sha256": branch_receipt_sha,
        "branch_replay_payload_sha256": branch_payload_sha,
        "process_isolation": "INNER_V1_ISOLATED_PROCESS_BOUNDARY",
        "returncode": 0,
        "wall_time_seconds": wall_time,
    }


def run_n500_operational_preflight(
    *,
    project_root: str | Path,
    diagnostic_root: str | Path,
    output_root: str | Path,
    inner_runner_path: str | Path,
    expected_outer_runner_sha256: str,
    expected_inner_runner_sha256: str,
    expected_python_path: str | Path,
    expected_python_sha256: str,
    provenance: Mapping[str, object],
    diagnostic_binding: Mapping[str, object],
    row_id: str,
    row_timeout_seconds: int,
    executor: Any,
) -> dict[str, object]:
    """Run and seal one n500/2000 operational-only same-implementation replay.

    ``executor`` is an explicit boundary so focused tests never launch a real
    replay.  The command-line entry point will supply only the production
    executor; callers cannot select an arbitrary implementation there.
    """

    if not callable(executor):
        raise RecoveryBoundCoverageError("Preflight executor is not callable")
    project = Path(project_root).resolve()
    diagnostic = Path(diagnostic_root).resolve()
    output = Path(output_root).resolve()
    identities_before = validate_runtime_identities(
        outer_runner_path=Path(__file__).resolve(),
        inner_runner_path=inner_runner_path,
        expected_outer_runner_sha256=expected_outer_runner_sha256,
        expected_inner_runner_sha256=expected_inner_runner_sha256,
        expected_python_path=expected_python_path,
        expected_python_sha256=expected_python_sha256,
    )
    row, provenance_digest, diagnostic_digest = _validate_n500_preflight_inputs(
        provenance=provenance,
        diagnostic_binding=diagnostic_binding,
        row_id=row_id,
        row_timeout_seconds=row_timeout_seconds,
    )
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise RecoveryBoundCoverageError("Preflight output parent is unsafe or absent")
    plan = _bind_payload(
        {
            "schema": "v21e3r1_recovery_bound_n500_operational_preflight_plan_v1",
            "status": "FROZEN_N500_FULL_BUDGET_OPERATIONAL_PREFLIGHT_ONLY",
            "project_root": project.as_posix(),
            "diagnostic_root": diagnostic.as_posix(),
            "output_root": output.as_posix(),
            "selected_row": row,
            "charged_evaluation_budget": 2000,
            "verification_jobs": 1,
            "row_timeout_seconds": row_timeout_seconds,
            "diagnostic_provenance_payload_sha256": provenance_digest,
            "diagnostic_binding_sha256": diagnostic_digest,
            "runtime_identities": asdict(identities_before),
            "preflight_required_for_full_coverage": True,
            "implementation_independence": False,
            "scientific_independence": False,
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
            "confirmation_authority": False,
            "formal_study_authority": False,
            "publication_status": "IJOC_HOLD",
        },
        "plan_payload_sha256",
    )
    # Atomic directory creation is the first write.  A race loser owns no
    # filesystem object and therefore leaves exactly zero bytes behind.
    os.mkdir(output)
    _fsync_directory(output.parent)
    plan_path = output / "n500_preflight.plan.json"
    plan_raw_sha = _exclusive_json(plan_path, plan)

    try:
        result = executor(
            project_root=project,
            diagnostic_root=diagnostic,
            output_root=output,
            row=dict(row),
            row_timeout_seconds=row_timeout_seconds,
            expected_plan_sha256=provenance["plan_sha256"],
            expected_source_sha256=provenance["source_snapshot_sha256"],
            identities=identities_before,
        )
    except Exception as error:
        _write_n500_preflight_failure(
            output=output,
            plan_path=plan_path,
            plan_raw_sha256=plan_raw_sha,
            plan_payload_sha256=str(plan["plan_payload_sha256"]),
            row=row,
            provenance_payload_sha256=provenance_digest,
            diagnostic_binding_sha256=diagnostic_digest,
            identities_before=identities_before,
            failure_phase="EXECUTOR",
            error=error,
        )
        raise RecoveryBoundCoverageError("Preflight executor failed") from error
    try:
        (
            wall_time,
            process_isolation,
            branch_relative,
            branch_raw_sha,
            branch_payload_sha,
            identities_after,
        ) = _validate_n500_preflight_result(
            result=result,
            output=output,
            identities_before=identities_before,
            inner_runner_path=inner_runner_path,
            expected_outer_runner_sha256=expected_outer_runner_sha256,
            expected_inner_runner_sha256=expected_inner_runner_sha256,
            expected_python_path=expected_python_path,
            expected_python_sha256=expected_python_sha256,
        )
    except Exception as error:
        _write_n500_preflight_failure(
            output=output,
            plan_path=plan_path,
            plan_raw_sha256=plan_raw_sha,
            plan_payload_sha256=str(plan["plan_payload_sha256"]),
            row=row,
            provenance_payload_sha256=provenance_digest,
            diagnostic_binding_sha256=diagnostic_digest,
            identities_before=identities_before,
            failure_phase="POST_EXECUTOR_VALIDATION",
            error=error,
        )
        raise RecoveryBoundCoverageError(
            "Preflight result validation failed"
        ) from error
    receipt = _bind_payload(
        {
            "schema": "v21e3r1_recovery_bound_n500_operational_preflight_receipt_v1",
            "status": "PASS_N500_FULL_BUDGET_OPERATIONAL_PREFLIGHT_ONLY",
            "plan_path": plan_path.name,
            "plan_sha256": plan_raw_sha,
            "plan_payload_sha256": plan["plan_payload_sha256"],
            "selected_row": row,
            "charged_evaluation_budget": 2000,
            "verification_jobs": 1,
            "row_timeout_seconds": row_timeout_seconds,
            "wall_time_seconds": wall_time,
            "process_isolation": process_isolation,
            "branch_replay_receipt_path": branch_relative,
            "branch_replay_receipt_sha256": branch_raw_sha,
            "branch_replay_payload_sha256": branch_payload_sha,
            "diagnostic_provenance_payload_sha256": provenance_digest,
            "diagnostic_binding_sha256": diagnostic_digest,
            "runtime_identities_before": asdict(identities_before),
            "runtime_identities_after": asdict(identities_after),
            "preflight_required_for_full_coverage": True,
            "operational_only": True,
            "implementation_independence": False,
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
    receipt_path = output / "n500_preflight.receipt.json"
    receipt_raw_sha = _exclusive_json(receipt_path, receipt)
    seal = _bind_payload(
        {
            "schema": "v21e3r1_recovery_bound_n500_preflight_success_seal_v1",
            "status": "SEALED_N500_OPERATIONAL_PREFLIGHT_SUCCESS_RECEIPT",
            "receipt_path": receipt_path.name,
            "receipt_sha256": receipt_raw_sha,
            "receipt_payload_sha256": receipt["receipt_payload_sha256"],
            "plan_sha256": plan_raw_sha,
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
            "confirmation_authority": False,
            "formal_study_authority": False,
            "publication_status": "IJOC_HOLD",
        },
        "seal_payload_sha256",
    )
    _exclusive_json(output / "n500_preflight.receipt.seal.json", seal)
    return receipt


def validate_n500_preflight_evidence(
    *,
    receipt_path: str | Path,
    seal_path: str | Path,
    provenance: Mapping[str, object],
    diagnostic_binding: Mapping[str, object],
    identities: RuntimeIdentities,
) -> dict[str, object]:
    receipt_file = Path(receipt_path).resolve()
    seal_file = Path(seal_path).resolve()
    preflight_root = receipt_file.parent
    if seal_file.parent != preflight_root:
        raise RecoveryBoundCoverageError("Preflight receipt/seal roots disagree")
    receipt, receipt_raw_sha = _validate_bound_payload(
        receipt_file,
        label="n500 preflight receipt",
        expected_keys=_N500_PREFLIGHT_RECEIPT_KEYS,
        digest_field="receipt_payload_sha256",
        schema="v21e3r1_recovery_bound_n500_operational_preflight_receipt_v1",
        status="PASS_N500_FULL_BUDGET_OPERATIONAL_PREFLIGHT_ONLY",
    )
    seal, seal_raw_sha = _validate_bound_payload(
        seal_file,
        label="n500 preflight success seal",
        expected_keys=_N500_PREFLIGHT_SEAL_KEYS,
        digest_field="seal_payload_sha256",
        schema="v21e3r1_recovery_bound_n500_preflight_success_seal_v1",
        status="SEALED_N500_OPERATIONAL_PREFLIGHT_SUCCESS_RECEIPT",
    )
    _require_hold_authority(receipt, label="n500 preflight receipt")
    _require_hold_authority(seal, label="n500 preflight success seal")
    provenance_digest = _validate_in_memory_bound_payload(
        provenance,
        label="recovered diagnostic provenance",
        digest_field="provenance_payload_sha256",
        schema="v21e3r1_recovered_diagnostic_provenance_binding_v1",
        status="PASS_CHAIN_VALIDATED_FOR_REPLAY_INPUT_ONLY",
    )
    diagnostic_digest = hashlib.sha256(_canonical_bytes(diagnostic_binding)).hexdigest()
    selected = receipt.get("selected_row")
    rows = diagnostic_binding.get("rows")
    if not isinstance(selected, dict) or not isinstance(rows, list):
        raise RecoveryBoundCoverageError("Preflight selected row binding is absent")
    matching_rows = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("row_id") == selected.get("row_id")
    ]
    if (
        len(matching_rows) != 1
        or matching_rows[0] != selected
        or selected.get("size") != 500
        or selected.get("budget") != 2000
        or receipt.get("charged_evaluation_budget") != 2000
        or receipt.get("verification_jobs") != 1
        or receipt.get("row_timeout_seconds") != 2400
        or receipt.get("preflight_required_for_full_coverage") is not True
        or receipt.get("operational_only") is not True
        or receipt.get("diagnostic_provenance_payload_sha256")
        != provenance_digest
        or receipt.get("diagnostic_binding_sha256") != diagnostic_digest
        or receipt.get("runtime_identities_before") != asdict(identities)
        or receipt.get("runtime_identities_after") != asdict(identities)
    ):
        raise RecoveryBoundCoverageError("Preflight receipt invocation binding drifted")
    plan_relative = receipt.get("plan_path")
    plan_path = _contained_path(
        preflight_root, plan_relative, label="n500 preflight plan"
    )
    plan, plan_raw_sha = _load_json_object(plan_path, label="n500 preflight plan")
    plan_payload_sha = _validate_in_memory_bound_payload(
        plan,
        label="n500 preflight plan",
        digest_field="plan_payload_sha256",
        schema="v21e3r1_recovery_bound_n500_operational_preflight_plan_v1",
        status="FROZEN_N500_FULL_BUDGET_OPERATIONAL_PREFLIGHT_ONLY",
    )
    if (
        receipt.get("plan_sha256") != plan_raw_sha
        or receipt.get("plan_payload_sha256") != plan_payload_sha
        or plan.get("selected_row") != selected
        or plan.get("verification_jobs") != 1
        or plan.get("row_timeout_seconds") != 2400
        or plan.get("diagnostic_provenance_payload_sha256") != provenance_digest
        or plan.get("diagnostic_binding_sha256") != diagnostic_digest
        or plan.get("runtime_identities") != asdict(identities)
    ):
        raise RecoveryBoundCoverageError("Preflight plan binding drifted")
    branch_path = _contained_path(
        preflight_root,
        receipt.get("branch_replay_receipt_path"),
        label="n500 preflight branch receipt",
    )
    branch, branch_raw_sha = _load_json_object(
        branch_path, label="n500 preflight branch receipt"
    )
    branch_payload_sha = _validate_in_memory_bound_payload(
        branch,
        label="n500 preflight branch receipt",
        digest_field="receipt_payload_sha256",
        schema="v21e3r1_same_implementation_branch_replay_v1",
        status="PASS_SAME_IMPLEMENTATION_FULL_BRANCH_REEXECUTION",
    )
    if (
        receipt.get("branch_replay_receipt_sha256") != branch_raw_sha
        or receipt.get("branch_replay_payload_sha256") != branch_payload_sha
        or seal.get("receipt_path") != receipt_file.name
        or seal.get("receipt_sha256") != receipt_raw_sha
        or seal.get("receipt_payload_sha256") != receipt.get(
            "receipt_payload_sha256"
        )
        or seal.get("plan_sha256") != plan_raw_sha
    ):
        raise RecoveryBoundCoverageError("Preflight success file chain drifted")
    return {
        "schema": "v21e3r1_recovery_bound_n500_preflight_binding_v1",
        "status": "PASS_SEALED_N500_PREFLIGHT_REQUIRED_FOR_FULL_COVERAGE",
        "selected_row_id": selected["row_id"],
        "charged_evaluation_budget": 2000,
        "verification_jobs": 1,
        "row_timeout_seconds": 2400,
        "receipt_path": receipt_file.as_posix(),
        "receipt_sha256": receipt_raw_sha,
        "receipt_payload_sha256": receipt["receipt_payload_sha256"],
        "seal_path": seal_file.as_posix(),
        "seal_sha256": seal_raw_sha,
        "seal_payload_sha256": seal["seal_payload_sha256"],
        "plan_sha256": plan_raw_sha,
        "diagnostic_provenance_payload_sha256": provenance_digest,
        "diagnostic_binding_sha256": diagnostic_digest,
        "runtime_authority": False,
        "scientific_authority": False,
        "selection_authority": False,
        "confirmation_authority": False,
        "formal_study_authority": False,
        "publication_status": "IJOC_HOLD",
    }


def _validate_full_coverage_inputs(
    *,
    provenance: Mapping[str, object],
    diagnostic_binding: Mapping[str, object],
) -> tuple[str, str]:
    provenance_digest = _validate_in_memory_bound_payload(
        provenance,
        label="recovered diagnostic provenance",
        digest_field="provenance_payload_sha256",
        schema="v21e3r1_recovered_diagnostic_provenance_binding_v1",
        status="PASS_CHAIN_VALIDATED_FOR_REPLAY_INPUT_ONLY",
    )
    if (
        provenance.get("expected_rows") != 504
        or diagnostic_binding.get("schema")
        != "v21e3r1_recovery_bound_exact504_diagnostic_binding_v1"
        or diagnostic_binding.get("status")
        != "PASS_EXACT504_DIAGNOSTIC_TREE_REVALIDATED_INPUT_ONLY"
        or diagnostic_binding.get("expected_rows") != 504
    ):
        raise RecoveryBoundCoverageError("Full coverage input is not exact504")
    rows = diagnostic_binding.get("rows")
    if not isinstance(rows, list) or len(rows) != 504:
        raise RecoveryBoundCoverageError("Full coverage diagnostic rows are not exact504")
    if any(
        not isinstance(row, dict) or row.get("ordinal") != ordinal
        for ordinal, row in enumerate(rows, start=1)
    ):
        raise RecoveryBoundCoverageError("Full coverage diagnostic row order drifted")
    for field in (
        "plan_sha256",
        "source_snapshot_sha256",
        "diagnostic_receipt_sha256",
        "diagnostic_aggregate_sha256",
    ):
        _require_sha256(
            diagnostic_binding.get(field), label=f"full coverage {field}"
        )
    _require_hold_authority(provenance, label="recovered diagnostic provenance")
    _require_hold_authority(diagnostic_binding, label="exact504 diagnostic binding")
    diagnostic_digest = hashlib.sha256(_canonical_bytes(diagnostic_binding)).hexdigest()
    if (
        provenance.get("plan_sha256") != diagnostic_binding.get("plan_sha256")
        or provenance.get("source_snapshot_sha256")
        != diagnostic_binding.get("source_snapshot_sha256")
        or provenance.get("exact504_diagnostic_binding_sha256")
        != diagnostic_digest
        or provenance.get("exact504_diagnostic_binding") != diagnostic_binding
    ):
        raise RecoveryBoundCoverageError("Full coverage provenance cross-binding drifted")
    return provenance_digest, diagnostic_digest


def _acquire_execution_claim(
    *,
    output_root: Path,
    plan_sha256: str,
    provenance_payload_sha256: str,
    preflight_binding: Mapping[str, object],
    identities: RuntimeIdentities,
    jobs: int,
    row_timeout_seconds: int,
    inner_resume: bool,
) -> tuple[Path, dict[str, object], str, int]:
    executions = output_root / "executions"
    claims: list[int] = []
    pattern = re.compile(r"execution-(\d{4})\.claim\.json")
    for path in executions.iterdir():
        if path.is_symlink() or not path.is_file():
            raise RecoveryBoundCoverageError("Execution ledger contains unsafe entry")
        match = pattern.fullmatch(path.name)
        if match:
            claims.append(int(match.group(1)))
            continue
        if re.fullmatch(
            r"execution-\d{4}\.(?:failure\.receipt|failure\.receipt\.seal)\.json",
            path.name,
        ):
            continue
        raise RecoveryBoundCoverageError("Execution ledger contains unknown entry")
    execution_number = max(claims, default=0) + 1
    claim_path = executions / f"execution-{execution_number:04d}.claim.json"
    claim = _bind_payload(
        {
            "schema": "v21e3r1_recovery_bound_coverage_execution_claim_v1",
            "status": "SEALED_EXCLUSIVE_SAME_IMPLEMENTATION_EXECUTION_CLAIM",
            "execution_number": execution_number,
            "plan_sha256": plan_sha256,
            "provenance_payload_sha256": provenance_payload_sha256,
            "preflight_receipt_sha256": preflight_binding["receipt_sha256"],
            "preflight_seal_sha256": preflight_binding["seal_sha256"],
            "runtime_identities": asdict(identities),
            "jobs": jobs,
            "row_timeout_seconds": row_timeout_seconds,
            "inner_resume": inner_resume,
            "implementation_independence": False,
            "scientific_independence": False,
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
            "confirmation_authority": False,
            "formal_study_authority": False,
            "publication_status": "IJOC_HOLD",
        },
        "claim_payload_sha256",
    )
    claim_raw_sha = _exclusive_json(claim_path, claim)
    return claim_path, claim, claim_raw_sha, execution_number


def _write_full_coverage_failure(
    *,
    output_root: Path,
    claim_path: Path,
    claim: Mapping[str, object],
    claim_raw_sha256: str,
    execution_number: int,
    failure_phase: str,
    error: BaseException,
) -> None:
    failure_path = output_root / "executions" / (
        f"execution-{execution_number:04d}.failure.receipt.json"
    )
    seal_path = output_root / "executions" / (
        f"execution-{execution_number:04d}.failure.receipt.seal.json"
    )
    inner_root = output_root / "inner_v1"
    partial_manifest = tree_manifest(inner_root) if inner_root.is_dir() else []
    failure = _bind_payload(
        {
            "schema": "v21e3r1_recovery_bound_coverage_failure_receipt_v1",
            "status": "HOLD_RECOVERY_BOUND_COVERAGE_EXECUTION_FAILED",
            "failure_phase": failure_phase,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "execution_number": execution_number,
            "claim_path": claim_path.relative_to(output_root).as_posix(),
            "claim_sha256": claim_raw_sha256,
            "claim_payload_sha256": claim["claim_payload_sha256"],
            "partial_inner_tree_manifest": partial_manifest,
            "automatic_retry_authorized": False,
            "success_materialized": False,
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
            "confirmation_authority": False,
            "formal_study_authority": False,
            "publication_status": "IJOC_HOLD",
        },
        "failure_receipt_payload_sha256",
    )
    failure_raw_sha = _exclusive_json(failure_path, failure)
    seal = _bind_payload(
        {
            "schema": "v21e3r1_recovery_bound_coverage_failure_seal_v1",
            "status": "SEALED_RECOVERY_BOUND_COVERAGE_FAILURE_RECEIPT",
            "failure_receipt_path": failure_path.relative_to(output_root).as_posix(),
            "failure_receipt_sha256": failure_raw_sha,
            "failure_receipt_payload_sha256": failure[
                "failure_receipt_payload_sha256"
            ],
            "claim_sha256": claim_raw_sha256,
            "automatic_retry_authorized": False,
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
            "confirmation_authority": False,
            "formal_study_authority": False,
            "publication_status": "IJOC_HOLD",
        },
        "seal_payload_sha256",
    )
    _exclusive_json(seal_path, seal)


def _validate_full_executor_result(
    *,
    result: object,
    output_root: Path,
    inner_output_root: Path,
    jobs: int,
    row_timeout_seconds: int,
    expected_plan_sha256: str,
    expected_source_sha256: str,
    expected_diagnostic_receipt_sha256: str,
    expected_diagnostic_aggregate_sha256: str,
) -> tuple[dict[str, Any], str, str, float | int, str]:
    if not isinstance(result, dict) or result.get("returncode") != 0:
        raise RecoveryBoundCoverageError("Full coverage executor failed")
    wall_time = result.get("wall_time_seconds")
    isolation = result.get("process_isolation")
    if (
        type(wall_time) not in {int, float}
        or wall_time < 0
        or type(isolation) is not str
        or not isolation
    ):
        raise RecoveryBoundCoverageError("Full coverage executor accounting drifted")
    receipt_value = result.get("inner_receipt_path")
    if not isinstance(receipt_value, (str, Path)):
        raise RecoveryBoundCoverageError("Inner final receipt path is absent")
    receipt_path = Path(receipt_value).resolve()
    try:
        receipt_path.relative_to(inner_output_root.resolve())
    except ValueError as error:
        raise RecoveryBoundCoverageError("Inner final receipt escaped output") from error
    inner_receipt, inner_raw_sha = _load_json_object(
        receipt_path, label="inner-v1 final coverage receipt"
    )
    inner_payload_sha = hashlib.sha256(_canonical_bytes(inner_receipt)).hexdigest()
    if (
        receipt_path.name != "branch_replay_coverage.receipt.json"
        or result.get("inner_receipt_sha256") != inner_raw_sha
        or result.get("inner_receipt_payload_sha256") != inner_payload_sha
        or inner_receipt.get("schema")
        != "v21e3r1_branch_replay_coverage_receipt_v1"
        or inner_receipt.get("status")
        != "PASS_SAME_IMPLEMENTATION_BRANCH_REPLAY_EXACT_504_DEVELOPMENT_ONLY"
        or inner_receipt.get("completed_rows") != 504
        or inner_receipt.get("expected_rows") != 504
        or inner_receipt.get("exact_full_504_coverage") is not True
        or inner_receipt.get("verification_jobs") != jobs
        or inner_receipt.get("row_timeout_seconds") != row_timeout_seconds
        or inner_receipt.get("diagnostic_plan_sha256")
        != expected_plan_sha256
        or inner_receipt.get("source_snapshot_sha256")
        != expected_source_sha256
        or inner_receipt.get("diagnostic_receipt_sha256")
        != expected_diagnostic_receipt_sha256
        or inner_receipt.get("diagnostic_aggregate_sha256")
        != expected_diagnostic_aggregate_sha256
    ):
        raise RecoveryBoundCoverageError("Inner-v1 exact504 receipt drifted")
    _require_hold_authority(inner_receipt, label="inner-v1 final coverage receipt")
    for field in (
        "implementation_independence",
        "scientific_independence",
        "third_party_replication",
        "selection_authorized",
        "confirmation_authorized",
        "formal_authorized",
        "runtime_efficiency_claims",
        "scientific_performance_claims",
    ):
        if inner_receipt.get(field) is not False:
            raise RecoveryBoundCoverageError("Inner-v1 receipt expands authority")
    if inner_receipt.get("ijoc_submission_status") != "IJOC_HOLD":
        raise RecoveryBoundCoverageError("Inner-v1 publication status drifted")
    return inner_receipt, inner_raw_sha, inner_payload_sha, wall_time, isolation


def _execute_real_full_coverage(
    *,
    project_root: Path,
    diagnostic_root: Path,
    inner_output_root: Path,
    inner_runner_path: Path,
    expected_inner_runner_sha256: str,
    jobs: int,
    row_timeout_seconds: int,
    resume: bool,
    expected_plan_sha256: str,
    expected_source_sha256: str,
    expected_diagnostic_receipt_sha256: str,
    expected_diagnostic_aggregate_sha256: str,
) -> dict[str, object]:
    inner_path = Path(inner_runner_path).resolve()
    if sha256_file(inner_path) != expected_inner_runner_sha256:
        raise RecoveryBoundCoverageError("Inner runner drifted before full coverage")
    module = _load_module(inner_path, "_v21e3r1_recovery_bound_full_inner_v1")
    if not callable(getattr(module, "run_coverage", None)):
        raise RecoveryBoundCoverageError("Inner runner omits run_coverage")
    started = time.monotonic()
    receipt = module.run_coverage(
        project_root,
        diagnostic_root,
        inner_output_root,
        diagnostic_plan_path=diagnostic_root / "diagnostic.plan.json",
        allow_smoke=False,
        resume=resume,
        jobs=jobs,
        row_timeout_seconds=row_timeout_seconds,
    )
    if (
        receipt.get("diagnostic_plan_sha256") != expected_plan_sha256
        or receipt.get("source_snapshot_sha256") != expected_source_sha256
        or receipt.get("diagnostic_receipt_sha256")
        != expected_diagnostic_receipt_sha256
        or receipt.get("diagnostic_aggregate_sha256")
        != expected_diagnostic_aggregate_sha256
    ):
        raise RecoveryBoundCoverageError(
            "Inner full-coverage receipt disagrees with recovered provenance"
        )
    wall_time = time.monotonic() - started
    receipt_path = inner_output_root / "branch_replay_coverage.receipt.json"
    if sha256_file(inner_path) != expected_inner_runner_sha256:
        raise RecoveryBoundCoverageError("Inner runner drifted after full coverage")
    return {
        "inner_receipt_path": receipt_path,
        "inner_receipt_sha256": sha256_file(receipt_path),
        "inner_receipt_payload_sha256": hashlib.sha256(
            _canonical_bytes(receipt)
        ).hexdigest(),
        "returncode": 0,
        "wall_time_seconds": wall_time,
        "process_isolation": "INNER_V1_PER_ROW_ISOLATED_PROCESS_BOUNDARY",
    }


def _validate_existing_outer_coverage(
    *,
    output: Path,
    plan: Mapping[str, object],
    provenance_payload_sha256: str,
    diagnostic_binding_sha256: str,
    preflight_binding: Mapping[str, object],
    diagnostic_binding: Mapping[str, object],
    identities: RuntimeIdentities,
    jobs: int,
    row_timeout_seconds: int,
) -> dict[str, object]:
    receipt_path = output / "recovery_bound_coverage.receipt.json"
    seal_path = output / "recovery_bound_coverage.receipt.seal.json"
    receipt, receipt_raw_sha = _validate_bound_payload(
        receipt_path,
        label="existing recovery-bound receipt",
        expected_keys=_OUTER_COVERAGE_RECEIPT_KEYS,
        digest_field="receipt_payload_sha256",
        schema="v21e3r1_recovery_bound_coverage_receipt_v1",
        status="PASS_RECOVERY_BOUND_SAME_IMPLEMENTATION_EXACT504_ONLY",
    )
    seal, _ = _validate_bound_payload(
        seal_path,
        label="existing recovery-bound success seal",
        expected_keys=_OUTER_COVERAGE_SEAL_KEYS,
        digest_field="seal_payload_sha256",
        schema="v21e3r1_recovery_bound_coverage_success_seal_v1",
        status="SEALED_RECOVERY_BOUND_COVERAGE_SUCCESS_RECEIPT",
    )
    _require_hold_authority(receipt, label="existing recovery-bound receipt")
    _require_hold_authority(seal, label="existing recovery-bound success seal")
    plan_path = output / "recovery_bound_coverage.plan.json"
    plan_raw_sha = sha256_file(plan_path)
    if (
        receipt.get("expected_rows") != 504
        or receipt.get("jobs") != jobs
        or receipt.get("row_timeout_seconds") != row_timeout_seconds
        or receipt.get("plan_sha256") != plan_raw_sha
        or receipt.get("plan_payload_sha256") != plan.get("plan_payload_sha256")
        or receipt.get("provenance_payload_sha256")
        != provenance_payload_sha256
        or receipt.get("diagnostic_binding_sha256") != diagnostic_binding_sha256
        or receipt.get("preflight_receipt_sha256")
        != preflight_binding.get("receipt_sha256")
        or receipt.get("preflight_seal_sha256")
        != preflight_binding.get("seal_sha256")
        or receipt.get("runtime_identities_before") != asdict(identities)
        or receipt.get("runtime_identities_after") != asdict(identities)
        or receipt.get("same_implementation_only") is not True
    ):
        raise RecoveryBoundCoverageError("Existing outer receipt binding drifted")
    claim_path = _contained_path(
        output,
        receipt.get("execution_claim_path"),
        label="existing execution claim",
    )
    claim, claim_raw_sha = _load_json_object(
        claim_path, label="existing execution claim"
    )
    claim_payload_sha = _validate_in_memory_bound_payload(
        claim,
        label="existing execution claim",
        digest_field="claim_payload_sha256",
        schema="v21e3r1_recovery_bound_coverage_execution_claim_v1",
        status="SEALED_EXCLUSIVE_SAME_IMPLEMENTATION_EXECUTION_CLAIM",
    )
    if (
        receipt.get("execution_claim_sha256") != claim_raw_sha
        or receipt.get("execution_claim_payload_sha256") != claim_payload_sha
    ):
        raise RecoveryBoundCoverageError("Existing execution claim drifted")
    inner_path = _contained_path(
        output,
        receipt.get("inner_receipt_path"),
        label="existing inner-v1 receipt",
    )
    _validate_full_executor_result(
        result={
            "inner_receipt_path": inner_path,
            "inner_receipt_sha256": receipt.get("inner_receipt_sha256"),
            "inner_receipt_payload_sha256": receipt.get(
                "inner_receipt_payload_sha256"
            ),
            "returncode": 0,
            "wall_time_seconds": receipt.get("wall_time_seconds"),
            "process_isolation": receipt.get("process_isolation"),
        },
        output_root=output,
        inner_output_root=output / "inner_v1",
        jobs=jobs,
        row_timeout_seconds=row_timeout_seconds,
        expected_plan_sha256=str(diagnostic_binding["plan_sha256"]),
        expected_source_sha256=str(diagnostic_binding["source_snapshot_sha256"]),
        expected_diagnostic_receipt_sha256=str(
            diagnostic_binding["diagnostic_receipt_sha256"]
        ),
        expected_diagnostic_aggregate_sha256=str(
            diagnostic_binding["diagnostic_aggregate_sha256"]
        ),
    )
    if (
        seal.get("receipt_path") != receipt_path.name
        or seal.get("receipt_sha256") != receipt_raw_sha
        or seal.get("receipt_payload_sha256") != receipt.get(
            "receipt_payload_sha256"
        )
        or seal.get("plan_sha256") != plan_raw_sha
        or seal.get("provenance_payload_sha256") != provenance_payload_sha256
        or seal.get("preflight_receipt_sha256")
        != preflight_binding.get("receipt_sha256")
        or seal.get("inner_receipt_sha256") != receipt.get("inner_receipt_sha256")
    ):
        raise RecoveryBoundCoverageError("Existing outer success seal drifted")
    return receipt


def run_recovery_bound_coverage(
    *,
    project_root: str | Path,
    diagnostic_root: str | Path,
    output_root: str | Path,
    inner_runner_path: str | Path,
    expected_outer_runner_sha256: str,
    expected_inner_runner_sha256: str,
    expected_python_path: str | Path,
    expected_python_sha256: str,
    provenance: Mapping[str, object],
    diagnostic_binding: Mapping[str, object],
    preflight_receipt_path: str | Path,
    preflight_seal_path: str | Path,
    jobs: int,
    row_timeout_seconds: int,
    resume: bool,
    executor: Any,
) -> dict[str, object]:
    if type(jobs) is not int or jobs <= 0:
        raise RecoveryBoundCoverageError("Full coverage jobs must be positive")
    if type(row_timeout_seconds) is not int or row_timeout_seconds < 2400:
        raise RecoveryBoundCoverageError(
            "Full coverage row timeout must be at least 2400 seconds"
        )
    if type(resume) is not bool or not callable(executor):
        raise RecoveryBoundCoverageError("Full coverage invocation types drifted")
    project = Path(project_root).resolve()
    diagnostic = Path(diagnostic_root).resolve()
    output = Path(output_root).resolve()
    identities_before = validate_runtime_identities(
        outer_runner_path=Path(__file__).resolve(),
        inner_runner_path=inner_runner_path,
        expected_outer_runner_sha256=expected_outer_runner_sha256,
        expected_inner_runner_sha256=expected_inner_runner_sha256,
        expected_python_path=expected_python_path,
        expected_python_sha256=expected_python_sha256,
    )
    provenance_digest, diagnostic_digest = _validate_full_coverage_inputs(
        provenance=provenance, diagnostic_binding=diagnostic_binding
    )
    preflight_binding = validate_n500_preflight_evidence(
        receipt_path=preflight_receipt_path,
        seal_path=preflight_seal_path,
        provenance=provenance,
        diagnostic_binding=diagnostic_binding,
        identities=identities_before,
    )
    plan = _bind_payload(
        {
            "schema": "v21e3r1_recovery_bound_coverage_plan_v1",
            "status": "FROZEN_RECOVERY_BOUND_SAME_IMPLEMENTATION_EXACT504",
            "project_root": project.as_posix(),
            "diagnostic_root": diagnostic.as_posix(),
            "output_root": output.as_posix(),
            "expected_rows": 504,
            "jobs": jobs,
            "row_timeout_seconds": row_timeout_seconds,
            "provenance_payload_sha256": provenance_digest,
            "diagnostic_binding_sha256": diagnostic_digest,
            "preflight_binding": preflight_binding,
            "runtime_identities": asdict(identities_before),
            "implementation_independence": False,
            "scientific_independence": False,
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
            "confirmation_authority": False,
            "formal_study_authority": False,
            "publication_status": "IJOC_HOLD",
        },
        "plan_payload_sha256",
    )
    initialize_recovery_bound_output(
        output_root=output,
        plan=plan,
        provenance=provenance,
        resume=resume,
    )
    outer_receipt_path = output / "recovery_bound_coverage.receipt.json"
    if outer_receipt_path.exists():
        if not resume:
            raise RecoveryBoundCoverageError("Unexpected outer receipt on fresh run")
        try:
            return _validate_existing_outer_coverage(
                output=output,
                plan=plan,
                provenance_payload_sha256=provenance_digest,
                diagnostic_binding_sha256=diagnostic_digest,
                preflight_binding=preflight_binding,
                diagnostic_binding=diagnostic_binding,
                identities=identities_before,
                jobs=jobs,
                row_timeout_seconds=row_timeout_seconds,
            )
        except Exception as error:
            raise RecoveryBoundCoverageError(
                "Existing outer success seal or receipt drifted"
            ) from error
    inner_output = output / "inner_v1"
    inner_resume = inner_output.exists()
    plan_raw_sha = sha256_file(output / "recovery_bound_coverage.plan.json")
    try:
        claim_path, claim, claim_raw_sha, execution_number = _acquire_execution_claim(
            output_root=output,
            plan_sha256=plan_raw_sha,
            provenance_payload_sha256=provenance_digest,
            preflight_binding=preflight_binding,
            identities=identities_before,
            jobs=jobs,
            row_timeout_seconds=row_timeout_seconds,
            inner_resume=inner_resume,
        )
    except FileExistsError as error:
        raise RecoveryBoundCoverageError(
            "Execution claim race loser performed zero writes"
        ) from error
    try:
        result = executor(
            project_root=project,
            diagnostic_root=diagnostic,
            inner_output_root=inner_output,
            inner_runner_path=Path(inner_runner_path).resolve(),
            expected_inner_runner_sha256=expected_inner_runner_sha256,
            jobs=jobs,
            row_timeout_seconds=row_timeout_seconds,
            resume=inner_resume,
            expected_plan_sha256=str(diagnostic_binding["plan_sha256"]),
            expected_source_sha256=str(
                diagnostic_binding["source_snapshot_sha256"]
            ),
            expected_diagnostic_receipt_sha256=str(
                diagnostic_binding["diagnostic_receipt_sha256"]
            ),
            expected_diagnostic_aggregate_sha256=str(
                diagnostic_binding["diagnostic_aggregate_sha256"]
            ),
        )
    except Exception as error:
        _write_full_coverage_failure(
            output_root=output,
            claim_path=claim_path,
            claim=claim,
            claim_raw_sha256=claim_raw_sha,
            execution_number=execution_number,
            failure_phase="INNER_V1_EXECUTOR",
            error=error,
        )
        raise RecoveryBoundCoverageError("Full coverage executor failed") from error
    try:
        (
            inner_receipt,
            inner_receipt_sha,
            inner_payload_sha,
            wall_time,
            isolation,
        ) = _validate_full_executor_result(
            result=result,
            output_root=output,
            inner_output_root=inner_output,
            jobs=jobs,
            row_timeout_seconds=row_timeout_seconds,
            expected_plan_sha256=str(diagnostic_binding["plan_sha256"]),
            expected_source_sha256=str(
                diagnostic_binding["source_snapshot_sha256"]
            ),
            expected_diagnostic_receipt_sha256=str(
                diagnostic_binding["diagnostic_receipt_sha256"]
            ),
            expected_diagnostic_aggregate_sha256=str(
                diagnostic_binding["diagnostic_aggregate_sha256"]
            ),
        )
        del inner_receipt
        identities_after = validate_runtime_identities(
            outer_runner_path=Path(__file__).resolve(),
            inner_runner_path=inner_runner_path,
            expected_outer_runner_sha256=expected_outer_runner_sha256,
            expected_inner_runner_sha256=expected_inner_runner_sha256,
            expected_python_path=expected_python_path,
            expected_python_sha256=expected_python_sha256,
        )
        if identities_after != identities_before:
            raise RecoveryBoundCoverageError("Full coverage runtime identities changed")
    except Exception as error:
        _write_full_coverage_failure(
            output_root=output,
            claim_path=claim_path,
            claim=claim,
            claim_raw_sha256=claim_raw_sha,
            execution_number=execution_number,
            failure_phase="POST_EXECUTOR_VALIDATION",
            error=error,
        )
        raise RecoveryBoundCoverageError("Full coverage result validation failed") from error
    outer_receipt = _bind_payload(
        {
            "schema": "v21e3r1_recovery_bound_coverage_receipt_v1",
            "status": "PASS_RECOVERY_BOUND_SAME_IMPLEMENTATION_EXACT504_ONLY",
            "expected_rows": 504,
            "jobs": jobs,
            "row_timeout_seconds": row_timeout_seconds,
            "wall_time_seconds": wall_time,
            "process_isolation": isolation,
            "plan_sha256": plan_raw_sha,
            "plan_payload_sha256": plan["plan_payload_sha256"],
            "provenance_payload_sha256": provenance_digest,
            "diagnostic_binding_sha256": diagnostic_digest,
            "preflight_receipt_sha256": preflight_binding["receipt_sha256"],
            "preflight_seal_sha256": preflight_binding["seal_sha256"],
            "execution_claim_path": claim_path.relative_to(output).as_posix(),
            "execution_claim_sha256": claim_raw_sha,
            "execution_claim_payload_sha256": claim["claim_payload_sha256"],
            "inner_receipt_path": (
                inner_output / "branch_replay_coverage.receipt.json"
            ).relative_to(output).as_posix(),
            "inner_receipt_sha256": inner_receipt_sha,
            "inner_receipt_payload_sha256": inner_payload_sha,
            "runtime_identities_before": asdict(identities_before),
            "runtime_identities_after": asdict(identities_after),
            "same_implementation_only": True,
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
    outer_receipt_sha = _exclusive_json(outer_receipt_path, outer_receipt)
    outer_seal = _bind_payload(
        {
            "schema": "v21e3r1_recovery_bound_coverage_success_seal_v1",
            "status": "SEALED_RECOVERY_BOUND_COVERAGE_SUCCESS_RECEIPT",
            "receipt_path": outer_receipt_path.name,
            "receipt_sha256": outer_receipt_sha,
            "receipt_payload_sha256": outer_receipt["receipt_payload_sha256"],
            "plan_sha256": plan_raw_sha,
            "provenance_payload_sha256": provenance_digest,
            "preflight_receipt_sha256": preflight_binding["receipt_sha256"],
            "inner_receipt_sha256": inner_receipt_sha,
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
            "confirmation_authority": False,
            "formal_study_authority": False,
            "publication_status": "IJOC_HOLD",
        },
        "seal_payload_sha256",
    )
    _exclusive_json(output / "recovery_bound_coverage.receipt.seal.json", outer_seal)
    return outer_receipt


def _add_common_chain_arguments(parser: argparse.ArgumentParser) -> None:
    for name in (
        "project-root",
        "diagnostic-output-root",
        "inner-runner-path",
        "expected-outer-runner-sha256",
        "expected-inner-runner-sha256",
        "expected-python-path",
        "expected-python-sha256",
        "external-helper-path",
        "external-claim-path",
        "external-handoff-path",
        "external-receipt-path",
        "external-seal-path",
        "old-recovery-helper-path",
        "old-recovery-claim-path",
        "old-recovery-failure-path",
        "old-recovery-failure-seal-path",
        "continuation-helper-path",
        "expected-continuation-helper-sha256",
        "continuation-claim-path",
        "continuation-receipt-path",
        "continuation-seal-path",
        "expected-plan-sha256",
        "expected-source-sha256",
        "expected-frozen-runner-sha256",
        "expected-metric-sha256",
        "expected-process-guard-sha256",
    ):
        parser.add_argument(f"--{name}", required=True)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recovery-bound outer gate for exact504 same-implementation branch replay"
        )
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser("validate-chain")
    _add_common_chain_arguments(validate)
    preflight = subcommands.add_parser("n500-preflight")
    _add_common_chain_arguments(preflight)
    preflight.add_argument("--output-root", required=True)
    preflight.add_argument("--row-id", required=True)
    coverage = subcommands.add_parser("full-coverage")
    _add_common_chain_arguments(coverage)
    coverage.add_argument("--output-root", required=True)
    coverage.add_argument("--preflight-receipt", required=True)
    coverage.add_argument("--preflight-seal", required=True)
    coverage.add_argument("--jobs", type=int, default=4)
    coverage.add_argument("--row-timeout-seconds", type=int, default=2400)
    coverage.add_argument("--resume", action="store_true")
    return parser


def _build_live_recovered_provenance(
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
    external = validate_external_scheduling_chain(
        diagnostic_root=args.diagnostic_output_root,
        helper_path=args.external_helper_path,
        claim_path=args.external_claim_path,
        handoff_path=args.external_handoff_path,
        receipt_path=args.external_receipt_path,
        seal_path=args.external_seal_path,
        expected_plan_sha256=args.expected_plan_sha256,
        expected_source_sha256=args.expected_source_sha256,
        expected_frozen_runner_sha256=args.expected_frozen_runner_sha256,
    )
    old = validate_old_recovery_failure_chain(
        diagnostic_root=args.diagnostic_output_root,
        helper_path=args.old_recovery_helper_path,
        claim_path=args.old_recovery_claim_path,
        failure_path=args.old_recovery_failure_path,
        seal_path=args.old_recovery_failure_seal_path,
        upstream_receipt_path=args.external_receipt_path,
        upstream_seal_path=args.external_seal_path,
        expected_plan_sha256=args.expected_plan_sha256,
        expected_source_sha256=args.expected_source_sha256,
        expected_frozen_runner_sha256=args.expected_frozen_runner_sha256,
        expected_metric_sha256=args.expected_metric_sha256,
        expected_process_guard_sha256=args.expected_process_guard_sha256,
    )
    continuation = validate_continuation_success_chain(
        diagnostic_root=args.diagnostic_output_root,
        helper_path=args.continuation_helper_path,
        expected_helper_sha256=args.expected_continuation_helper_sha256,
        claim_path=args.continuation_claim_path,
        receipt_path=args.continuation_receipt_path,
        seal_path=args.continuation_seal_path,
        external_chain=external,
        old_recovery_chain=old,
        expected_plan_sha256=args.expected_plan_sha256,
        expected_source_sha256=args.expected_source_sha256,
        expected_frozen_runner_sha256=args.expected_frozen_runner_sha256,
        expected_metric_sha256=args.expected_metric_sha256,
        expected_process_guard_sha256=args.expected_process_guard_sha256,
    )
    diagnostic = validate_exact_diagnostic_tree(
        project_root=args.project_root,
        diagnostic_root=args.diagnostic_output_root,
        inner_runner_path=args.inner_runner_path,
        expected_inner_runner_sha256=args.expected_inner_runner_sha256,
    )
    provenance = build_recovered_diagnostic_provenance_binding(
        external_chain=external,
        old_recovery_chain=old,
        continuation_chain=continuation,
        diagnostic_binding=diagnostic,
    )
    return diagnostic, provenance


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    validate_runtime_identities(
        outer_runner_path=Path(__file__).resolve(),
        inner_runner_path=args.inner_runner_path,
        expected_outer_runner_sha256=args.expected_outer_runner_sha256,
        expected_inner_runner_sha256=args.expected_inner_runner_sha256,
        expected_python_path=args.expected_python_path,
        expected_python_sha256=args.expected_python_sha256,
    )
    diagnostic, provenance = _build_live_recovered_provenance(args)
    if args.command == "validate-chain":
        summary = {
            "status": provenance["status"],
            "expected_rows": provenance["expected_rows"],
            "provenance_payload_sha256": provenance[
                "provenance_payload_sha256"
            ],
            "runtime_authority": False,
            "scientific_authority": False,
            "publication_status": "IJOC_HOLD",
        }
    elif args.command == "n500-preflight":
        receipt = run_n500_operational_preflight(
            project_root=args.project_root,
            diagnostic_root=args.diagnostic_output_root,
            output_root=args.output_root,
            inner_runner_path=args.inner_runner_path,
            expected_outer_runner_sha256=args.expected_outer_runner_sha256,
            expected_inner_runner_sha256=args.expected_inner_runner_sha256,
            expected_python_path=args.expected_python_path,
            expected_python_sha256=args.expected_python_sha256,
            provenance=provenance,
            diagnostic_binding=diagnostic,
            row_id=args.row_id,
            row_timeout_seconds=2400,
            executor=_execute_real_n500_preflight,
        )
        summary = {
            "status": receipt["status"],
            "selected_row_id": receipt["selected_row"]["row_id"],
            "receipt_payload_sha256": receipt["receipt_payload_sha256"],
            "runtime_authority": False,
            "scientific_authority": False,
            "publication_status": "IJOC_HOLD",
        }
    elif args.command == "full-coverage":
        receipt = run_recovery_bound_coverage(
            project_root=args.project_root,
            diagnostic_root=args.diagnostic_output_root,
            output_root=args.output_root,
            inner_runner_path=args.inner_runner_path,
            expected_outer_runner_sha256=args.expected_outer_runner_sha256,
            expected_inner_runner_sha256=args.expected_inner_runner_sha256,
            expected_python_path=args.expected_python_path,
            expected_python_sha256=args.expected_python_sha256,
            provenance=provenance,
            diagnostic_binding=diagnostic,
            preflight_receipt_path=args.preflight_receipt,
            preflight_seal_path=args.preflight_seal,
            jobs=args.jobs,
            row_timeout_seconds=args.row_timeout_seconds,
            resume=args.resume,
            executor=_execute_real_full_coverage,
        )
        summary = {
            "status": receipt["status"],
            "receipt_payload_sha256": receipt["receipt_payload_sha256"],
            "runtime_authority": False,
            "scientific_authority": False,
            "publication_status": "IJOC_HOLD",
        }
    else:  # pragma: no cover - argparse makes this unreachable.
        raise RecoveryBoundCoverageError("Unknown recovery-bound command")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
