from __future__ import annotations

"""Append-only continuation of the sealed exact-17 recovery incident.

The predecessor recovery is immutable evidence.  This helper validates but
does not adopt its five complete attempts, then performs fresh full reruns for
all exact ordinals 446-462 with corrected per-row Job witnesses.
It never grants runtime, scientific, selection, confirmation, formal-study,
or publication authority.
"""

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import ModuleType
from typing import Callable, Mapping, NoReturn, Sequence


PROJECT_RELATIVE = Path(
    "artifacts/v21e3r1_v8_work_20260822/"
    "run_frozen_diagnostic_metric_timeout_recovery_continuation.py"
)
PREDECESSOR_RELATIVE = Path(
    "artifacts/v21e3r1_v8_work_20260822/"
    "run_frozen_diagnostic_metric_timeout_recovery.py"
)
OUTPUT_RELATIVE = Path(
    "outputs/v21e3r1_v7_exposed_development_diagnostics_20260823"
)
EXPECTED_PREDECESSOR_HELPER_SHA256 = (
    "c4d0c67fc22fbc45a6b73b9d94cc9ab73bd1767fd78d9cfb6d9d670b194aa122"
)
EXPECTED_PLAN_SHA256 = (
    "4408d10944cb6511e99ff0bd95ded256b9c230b91d8806a7bd5b962f10622886"
)
EXPECTED_SOURCE_ROOT_SHA256 = (
    "218bc398f04722d1da305928a9c206641f9b43d74b2afbc46c29ba1f08d6639b"
)
EXPECTED_RUNNER_SHA256 = (
    "70a45fd0e62d870702b29a92b66b38eef6c04952152d5defae89c115c6d85b7b"
)
EXPECTED_METRIC_SHA256 = (
    "587d4ed4d647d8293b36449c835109ee3afa6e9899fe155f917a492fdf303ea2"
)
EXPECTED_PROCESS_GUARD_SHA256 = (
    "f4536eb6cef95047fc02958d0362b2b4a1fa521d5a0bc4d172bb462bd660a697"
)
EXPECTED_OLD_CLAIM_SHA256 = (
    "52eb90cb1d10ae5fd81f934bf38916dc270994a53fc19089e5ac8b181b186a8b"
)
EXPECTED_OLD_FAILURE_SHA256 = (
    "914df3f81e99f2ad7392312d1ac5c033979619db5dafb6edf74ce7c4be64c5e6"
)
EXPECTED_OLD_FAILURE_SEAL_SHA256 = (
    "a6c5e77b2b28c6eeafd2a6914ace7ef311d945c1f9346d51b1f1ea9a9639d72d"
)
EXPECTED_EXTERNAL_HANDOFF_SHA256 = (
    "bd12fe96531acf7725c6cac3dddbd4d6dfb75c23f48ae999815651e8ce0587f6"
)
EXPECTED_EXTERNAL_CLAIM_SHA256 = (
    "ad157ce2678ed7dd02d8e8b95e2b6b1ef463df14b6b5cdb4b0cc2a633a3c8feb"
)
EXPECTED_EXTERNAL_RECEIPT_SHA256 = (
    "bfa1952ec5b585d5a6e192201090c9097119539b38ee63a21df6f3a4a2e9707c"
)
EXPECTED_EXTERNAL_SEAL_SHA256 = (
    "6113d749ab3a24c4955fc145bfef4627db597ce08f4bc2e5083893974f521c3a"
)
EXPECTED_INTERPRETER_PATH = Path(r"C:\miniconda3\python.exe")
EXPECTED_FULL_ROWS = 504
TARGET_CASE_ID = "v21e3-motsp-development-n500-s00"
RECOVERY_ORDINALS = tuple(range(446, 463))
INCIDENT_COMPLETE_ORDINALS = tuple(range(446, 451))
FRESH_RERUN_ORDINALS = RECOVERY_ORDINALS
JOBS = 4
ORIGINAL_METRIC_TIMEOUT_SECONDS = 300
OPERATIONAL_METRIC_TIMEOUT_SECONDS = 1200
OUTER_ROW_TIMEOUT_SECONDS = 2400
ACCOUNTING_GRACE_SECONDS = 30
DESCENDANT_ZERO_TIMEOUT_SECONDS = 30
JOB_START_GATE_ARGUMENT = "V21E3R1_CONTINUATION_WINDOWS_JOB_ASSIGNED_V1"
JOB_START_GATE_LINE = JOB_START_GATE_ARGUMENT + "\n"
CONTINUATION_SEMANTICS = (
    "CHAINED_APPEND_ONLY_RECOVERY_5_COMPLETE_NOT_ADOPTED_FRESH_RERUN_17_V1"
)

OLD_CLAIM_NAME = (
    "metric-timeout-recovery.v21e3-motsp-development-n500-s00."
    "helper-instance.claim.json"
)
OLD_FAILURE_NAME = (
    "metric-timeout-recovery.v21e3-motsp-development-n500-s00."
    "failure.receipt.json"
)
OLD_FAILURE_SEAL_NAME = (
    "metric-timeout-recovery.v21e3-motsp-development-n500-s00."
    "failure.receipt.seal.json"
)
OLD_QUARANTINE_NAME = (
    "metric-timeout-recovery.v21e3-motsp-development-n500-s00."
    "descendant-state.quarantine.json"
)
OLD_SUCCESS_NAME = (
    "metric-timeout-recovery.v21e3-motsp-development-n500-s00.receipt.json"
)
OLD_SUCCESS_SEAL_NAME = (
    "metric-timeout-recovery.v21e3-motsp-development-n500-s00."
    "receipt.seal.json"
)
CLAIM_NAME = (
    "metric-timeout-recovery-continuation.v21e3-motsp-development-n500-s00."
    "helper-instance.claim.json"
)
FAILURE_NAME = (
    "metric-timeout-recovery-continuation.v21e3-motsp-development-n500-s00."
    "failure.receipt.json"
)
FAILURE_SEAL_NAME = (
    "metric-timeout-recovery-continuation.v21e3-motsp-development-n500-s00."
    "failure.receipt.seal.json"
)
QUARANTINE_NAME = (
    "metric-timeout-recovery-continuation.v21e3-motsp-development-n500-s00."
    "descendant-state.quarantine.json"
)
RECEIPT_NAME = (
    "metric-timeout-recovery-continuation.v21e3-motsp-development-n500-s00."
    "receipt.json"
)
RECEIPT_SEAL_NAME = (
    "metric-timeout-recovery-continuation.v21e3-motsp-development-n500-s00."
    "receipt.seal.json"
)
JOB_WITNESS_NAME = "continuation.windows-job.receipt.json"
EXTERNAL_HANDOFF_NAME = (
    "external-scheduling.v21e3-motsp-development-n500-s01."
    "main-driver-handoff.receipt.json"
)
EXTERNAL_CLAIM_NAME = (
    "external-scheduling.v21e3-motsp-development-n500-s01."
    "helper-instance.claim.json"
)
EXTERNAL_RECEIPT_NAME = (
    "external-scheduling.v21e3-motsp-development-n500-s01.receipt.json"
)
EXTERNAL_SEAL_NAME = (
    "external-scheduling.v21e3-motsp-development-n500-s01.receipt.seal.json"
)


@dataclass(frozen=True)
class ContinuationRow:
    ordinal: int
    row_id: str
    expected_attempt_number: int
    worker_spec: dict[str, object]


@dataclass(frozen=True)
class ContinuationContext:
    project_root: Path
    output: Path
    plan_path: Path
    runner_path: Path
    metric_path: Path
    process_guard_path: Path
    predecessor_path: Path
    rows: tuple[ContinuationRow, ...]
    incident_complete_rows: tuple[ContinuationRow, ...]
    fresh_rows: tuple[ContinuationRow, ...]
    all_row_ids: tuple[str, ...]
    preserved_marker_manifest: tuple[dict[str, object], ...]
    incident_file_manifest: tuple[dict[str, object], ...]
    incident_complete_attempt_manifest: tuple[dict[str, object], ...]
    predecessor_failed_attempt_manifest: tuple[dict[str, object], ...]
    external_scheduling_manifest: tuple[dict[str, object], ...]


class DescendantTerminationUnconfirmed(RuntimeError):
    pass


class JobControlledProcessTimeout(RuntimeError):
    def __init__(self, message: str, witness: Mapping[str, object]) -> None:
        super().__init__(message)
        self.witness = dict(witness)


def _fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _exclusive_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _exact_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} is not an exact lowercase SHA-256")
    return value


def _exact_keys(
    payload: object, expected: set[str], *, label: str
) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != expected:
        actual = set(payload) if type(payload) is dict else type(payload).__name__
        _fail(f"{label} exact keys drifted: {actual}")
    return payload


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _fail(f"Cannot load fixed module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_predecessor(project_root: Path) -> ModuleType:
    path = (project_root / PREDECESSOR_RELATIVE).resolve()
    if path.is_symlink() or not path.is_file() or _sha256(path) != EXPECTED_PREDECESSOR_HELPER_SHA256:
        _fail("Immutable predecessor recovery helper drifted")
    return _load_module(path, "_v21e3r1_recovery_continuation_predecessor")


def _validate_expected_helper_identity(
    project_root: Path, expected_helper_sha256: object
) -> str:
    expected = _exact_sha256(
        expected_helper_sha256, label="externally supplied continuation helper SHA"
    )
    helper_path = (project_root / PROJECT_RELATIVE).resolve()
    if (
        helper_path.is_symlink()
        or not helper_path.is_file()
        or _sha256(helper_path) != expected
    ):
        _fail("Continuation helper bytes do not match externally supplied SHA")
    return expected


def _bound_payload(core: Mapping[str, object], *, digest_field: str) -> dict[str, object]:
    payload = dict(core)
    payload[digest_field] = hashlib.sha256(_canonical_bytes(core)).hexdigest()
    return payload


def _authority_hold_fields() -> dict[str, object]:
    return {
        "runtime_authority": False,
        "scientific_authority": False,
        "selection_authority": False,
        "confirmation_authority": False,
        "formal_study_authority": False,
        "publication_status": "IJOC_HOLD",
    }


def _file_manifest(root: Path, paths: Sequence[Path]) -> tuple[dict[str, object], ...]:
    entries: list[dict[str, object]] = []
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink() or not path.is_file():
            _fail(f"Unsafe or missing manifest file: {path}")
        relative = path.relative_to(root).as_posix()
        before = path.stat()
        digest = _sha256(path)
        after = path.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            _fail(f"File changed while hashing: {relative}")
        entries.append(
            {"path": relative, "bytes": after.st_size, "sha256": digest}
        )
    return tuple(entries)


def _manifest_sha256(entries: Sequence[Mapping[str, object]]) -> str:
    return hashlib.sha256(_canonical_bytes([dict(entry) for entry in entries])).hexdigest()


def _verify_manifest_unchanged(
    root: Path,
    entries: Sequence[Mapping[str, object]],
    *,
    label: str,
) -> None:
    paths = [root / str(entry["path"]) for entry in entries]
    current = _file_manifest(root, paths)
    if [dict(entry) for entry in current] != [dict(entry) for entry in entries]:
        _fail(f"Immutable {label} manifest drifted")


def _verify_incident_immutable(context: ContinuationContext) -> None:
    _verify_manifest_unchanged(
        context.output,
        context.preserved_marker_manifest,
        label="exact487 completed-marker",
    )
    _verify_manifest_unchanged(
        context.output,
        context.incident_file_manifest,
        label="predecessor claim/failure/seal",
    )
    _verify_manifest_unchanged(
        context.output,
        context.incident_complete_attempt_manifest,
        label="five complete-but-unadopted attempts",
    )
    _verify_manifest_unchanged(
        context.output,
        context.predecessor_failed_attempt_manifest,
        label="ordinal446 predecessor failed attempt",
    )
    _verify_manifest_unchanged(
        context.output,
        context.external_scheduling_manifest,
        label="external-scheduling s01 claim/handoff/receipt/seal",
    )
    for name in (
        OLD_QUARANTINE_NAME,
        OLD_SUCCESS_NAME,
        OLD_SUCCESS_SEAL_NAME,
        "diagnostic.aggregate.json",
        "diagnostic.receipt.json",
    ):
        if (context.output / name).exists() or (context.output / name).is_symlink():
            _fail(f"Forbidden predecessor/final artifact appeared: {name}")


def _validate_bound_payload(payload: Mapping[str, object], *, digest_field: str, label: str) -> None:
    core = dict(payload)
    digest = _exact_sha256(core.pop(digest_field), label=f"{label} payload digest")
    if digest != hashlib.sha256(_canonical_bytes(core)).hexdigest():
        _fail(f"{label} payload digest drifted")


def _build_rows(
    project_root: Path,
    output: Path,
    predecessor: ModuleType,
    runner: ModuleType,
) -> tuple[tuple[ContinuationRow, ...], tuple[str, ...]]:
    plan_path = output / predecessor.PLAN_NAME
    plan = _exact_keys(
        runner._load_json_object(plan_path),
        {
            "schema", "status", "scientific_scope", "case_ids", "seeds",
            "arms", "charged_evaluation_budget", "checkpoint_period",
            "expected_rows", "input_binding", "source_manifest",
            "row_timeout_seconds", "selection_entropy_release",
            "confirmation_materialization", "formal_materialization",
        },
        label="frozen diagnostic plan",
    )
    if (
        plan["schema"] != "v21e3r1_exposed_development_diagnostic_plan_v2"
        or plan["status"] != "FROZEN_FULL_504_DEVELOPMENT_DIAGNOSTIC"
        or plan["scientific_scope"] != runner.DIAGNOSTIC_SCOPE
        or plan["case_ids"] != list(runner.EXPECTED_CASE_IDS)
        or plan["seeds"] != list(runner.SEEDS)
        or plan["arms"] != list(runner.DIAGNOSTIC_ARMS)
        or plan["charged_evaluation_budget"] != runner.FULL_BUDGET
        or plan["checkpoint_period"] != runner.FULL_CHECKPOINT_PERIOD
        or plan["expected_rows"] != EXPECTED_FULL_ROWS
        or plan["row_timeout_seconds"] != predecessor.FROZEN_PLAN_ROW_TIMEOUT_SECONDS
        or plan["selection_entropy_release"] != "PROHIBITED"
        or plan["confirmation_materialization"] != "PROHIBITED"
        or plan["formal_materialization"] != "PROHIBITED"
    ):
        _fail("Frozen diagnostic plan semantic contract drifted")
    current_source = runner._source_manifest(project_root)
    if runner._canonical_json(plan["source_manifest"]) != runner._canonical_json(current_source):
        _fail("Frozen diagnostic source manifest drifted")
    if current_source.get("source_snapshot_sha256") != EXPECTED_SOURCE_ROOT_SHA256:
        _fail("Frozen diagnostic source root drifted")
    cases, bounds, directions, input_binding = runner._load_inputs(project_root)
    if runner._canonical_json(input_binding) != runner._canonical_json(plan["input_binding"]):
        _fail("Frozen diagnostic input binding drifted")
    target = [case for case in cases if case.get("case_id") == TARGET_CASE_ID]
    if len(target) != 1 or target[0].get("family") != "MOTSP" or target[0].get("size") != 500:
        _fail("Continuation target case drifted")
    case_path = runner._case_path(project_root, target[0])
    case_sha256 = runner._sha256(case_path)
    lower, upper = bounds[TARGET_CASE_ID]
    all_row_ids = tuple(
        f"{case_id}__seed-{seed}__arm-{arm.lower()}"
        for case_id in runner.EXPECTED_CASE_IDS
        for seed in runner.SEEDS
        for arm in runner.DIAGNOSTIC_ARMS
    )
    if len(all_row_ids) != EXPECTED_FULL_ROWS or len(set(all_row_ids)) != EXPECTED_FULL_ROWS:
        _fail("Frozen exact504 row-ID construction drifted")
    target_index = list(runner.EXPECTED_CASE_IDS).index(TARGET_CASE_ID)
    rows: list[ContinuationRow] = []
    local = 0
    for seed in runner.SEEDS:
        for arm in runner.DIAGNOSTIC_ARMS:
            local += 1
            ordinal = target_index * len(runner.SEEDS) * len(runner.DIAGNOSTIC_ARMS) + local
            if ordinal not in RECOVERY_ORDINALS:
                continue
            row_id = f"{TARGET_CASE_ID}__seed-{seed}__arm-{arm.lower()}"
            rows.append(
                ContinuationRow(
                    ordinal=ordinal,
                    row_id=row_id,
                    expected_attempt_number=(
                        3 if ordinal == 446 else 2 if ordinal <= 450 else 1
                    ),
                    worker_spec={
                        "schema": "v21e3r1_diagnostic_row_worker_spec_v1",
                        "project_root": str(project_root),
                        "case_id": TARGET_CASE_ID,
                        "family": "MOTSP",
                        "size": 500,
                        "case_path": str(case_path),
                        "case_artifact_sha256": case_sha256,
                        "objective_lower_bounds": lower,
                        "objective_upper_bounds": upper,
                        "reference_directions": directions,
                        "seed": seed,
                        "arm_id": arm,
                        "charged_evaluation_budget": runner.FULL_BUDGET,
                        "checkpoint_period": runner.FULL_CHECKPOINT_PERIOD,
                        "source_snapshot_sha256": EXPECTED_SOURCE_ROOT_SHA256,
                        "plan_sha256": EXPECTED_PLAN_SHA256,
                    },
                )
            )
    if (
        tuple(row.ordinal for row in rows) != RECOVERY_ORDINALS
        or rows[5].worker_spec["arm_id"] != "C0_NO_LS"
        or rows[-1].worker_spec["arm_id"] != "MOEAD_SEEDED_POP21"
    ):
        _fail("Exact continuation row construction drifted")
    return tuple(rows), all_row_ids


def _validate_incident(
    project_root: Path,
    output: Path,
    predecessor: ModuleType,
    runner: ModuleType,
) -> ContinuationContext:
    helper_path = (project_root / PROJECT_RELATIVE).resolve()
    predecessor_path = (project_root / PREDECESSOR_RELATIVE).resolve()
    plan_path = (output / predecessor.PLAN_NAME).resolve()
    runner_path = (project_root / predecessor.RUNNER_RELATIVE).resolve()
    metric_path = (project_root / predecessor.METRIC_RELATIVE).resolve()
    process_guard_path = (project_root / predecessor.PROCESS_GUARD_RELATIVE).resolve()
    for path, expected, label in (
        (predecessor_path, EXPECTED_PREDECESSOR_HELPER_SHA256, "predecessor helper"),
        (plan_path, EXPECTED_PLAN_SHA256, "diagnostic plan"),
        (runner_path, EXPECTED_RUNNER_SHA256, "frozen runner"),
        (metric_path, EXPECTED_METRIC_SHA256, "independent metric"),
        (process_guard_path, EXPECTED_PROCESS_GUARD_SHA256, "process guard"),
    ):
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
            _fail(f"Fixed {label} bytes drifted")
    if helper_path.is_symlink() or not helper_path.is_file():
        _fail("Continuation helper is missing or unsafe")
    rows, all_row_ids = _build_rows(project_root, output, predecessor, runner)
    row_by_id = {row.row_id: row for row in rows}
    old_paths = (
        output / OLD_CLAIM_NAME,
        output / OLD_FAILURE_NAME,
        output / OLD_FAILURE_SEAL_NAME,
    )
    incident_manifest = _file_manifest(output, old_paths)
    if [entry["sha256"] for entry in incident_manifest] != [
        EXPECTED_OLD_FAILURE_SHA256,
        EXPECTED_OLD_FAILURE_SEAL_SHA256,
        EXPECTED_OLD_CLAIM_SHA256,
    ]:
        by_name = {Path(str(entry["path"])).name: entry["sha256"] for entry in incident_manifest}
        if (
            by_name.get(OLD_CLAIM_NAME) != EXPECTED_OLD_CLAIM_SHA256
            or by_name.get(OLD_FAILURE_NAME) != EXPECTED_OLD_FAILURE_SHA256
            or by_name.get(OLD_FAILURE_SEAL_NAME) != EXPECTED_OLD_FAILURE_SEAL_SHA256
        ):
            _fail("Predecessor incident raw bytes drifted")
    external_paths = (
        output / EXTERNAL_HANDOFF_NAME,
        output / EXTERNAL_CLAIM_NAME,
        output / EXTERNAL_RECEIPT_NAME,
        output / EXTERNAL_SEAL_NAME,
    )
    external_manifest = _file_manifest(output, external_paths)
    external_by_name = {
        Path(str(entry["path"])).name: entry["sha256"]
        for entry in external_manifest
    }
    if external_by_name != {
        EXTERNAL_HANDOFF_NAME: EXPECTED_EXTERNAL_HANDOFF_SHA256,
        EXTERNAL_CLAIM_NAME: EXPECTED_EXTERNAL_CLAIM_SHA256,
        EXTERNAL_RECEIPT_NAME: EXPECTED_EXTERNAL_RECEIPT_SHA256,
        EXTERNAL_SEAL_NAME: EXPECTED_EXTERNAL_SEAL_SHA256,
    }:
        _fail("External-scheduling s01 fixed raw bytes drifted")
    old_claim = runner._load_json_object(output / OLD_CLAIM_NAME)
    old_failure = runner._load_json_object(output / OLD_FAILURE_NAME)
    old_seal = runner._load_json_object(output / OLD_FAILURE_SEAL_NAME)
    _validate_bound_payload(old_claim, digest_field="claim_payload_sha256", label="old claim")
    _validate_bound_payload(old_failure, digest_field="receipt_payload_sha256", label="old failure")
    _validate_bound_payload(old_seal, digest_field="seal_payload_sha256", label="old failure seal")
    if (
        old_claim.get("schema") != "v21e3r1_metric_timeout_recovery_helper_instance_claim_v1"
        or old_claim.get("status") != "SEALED_HELPER_INSTANCE_CLAIM_NOT_A_MAIN_RUNNER_LOCK"
        or old_claim.get("helper_sha256") != EXPECTED_PREDECESSOR_HELPER_SHA256
        or old_claim.get("target_ordinals") != list(RECOVERY_ORDINALS)
        or old_failure.get("schema") != "v21e3r1_metric_timeout_recovery_failure_receipt_v1"
        or old_failure.get("status") != "HOLD_METRIC_TIMEOUT_RECOVERY_FAILURE_MANUAL_AUDIT_REQUIRED"
        or old_failure.get("helper_instance_claim_sha256") != EXPECTED_OLD_CLAIM_SHA256
        or old_failure.get("helper_sha256") != EXPECTED_PREDECESSOR_HELPER_SHA256
        or old_failure.get("failure_phase") != "PARALLEL_RECOVERY_ROWS"
        or old_failure.get("terminal_descendant_state_confirmed") is not True
        or old_failure.get("main_runner_resume_authorized") is not False
        or old_seal.get("schema") != "v21e3r1_metric_timeout_recovery_failure_seal_v1"
        or old_seal.get("failure_receipt_sha256") != EXPECTED_OLD_FAILURE_SHA256
        or old_seal.get("helper_instance_claim_sha256") != EXPECTED_OLD_CLAIM_SHA256
    ):
        _fail("Predecessor incident semantic contract drifted")
    for name in (OLD_QUARANTINE_NAME, OLD_SUCCESS_NAME, OLD_SUCCESS_SEAL_NAME):
        if (output / name).exists() or (output / name).is_symlink():
            _fail(f"Unexpected predecessor terminal evidence exists: {name}")
    for name in (CLAIM_NAME, FAILURE_NAME, FAILURE_SEAL_NAME, QUARANTINE_NAME, RECEIPT_NAME, RECEIPT_SEAL_NAME):
        if (output / name).exists() or (output / name).is_symlink():
            _fail(f"Continuation evidence already exists: {name}")
    completed_root = output / "completed"
    observed_markers = sorted(completed_root.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() or not path.is_file() for path in observed_markers):
        _fail("Completed directory contains a non-regular entry")
    expected_missing = {row.row_id for row in rows}
    expected_marker_names = {
        f"{row_id}.json" for row_id in all_row_ids if row_id not in expected_missing
    }
    if {path.name for path in observed_markers} != expected_marker_names:
        _fail("Incident completed-marker set is not exact487 complement")
    marker_manifest = _file_manifest(output, observed_markers)
    owned = old_failure.get("owned_attempts")
    if type(owned) is not list or len(owned) != len(INCIDENT_COMPLETE_ORDINALS):
        _fail("Predecessor incident does not bind exact five owned attempts")
    owned_by_row = {
        item.get("row_id"): item for item in owned if type(item) is dict
    }
    if set(owned_by_row) != {row.row_id for row in rows[:5]}:
        _fail("Predecessor owned-attempt row set drifted")
    incident_complete_attempt_files: list[Path] = []
    for row in rows[:5]:
        item = owned_by_row[row.row_id]
        incident_attempt_number = 2 if row.ordinal == 446 else 1
        expected_relative = (
            Path("attempts") / row.row_id / f"attempt-{incident_attempt_number:04d}"
        ).as_posix()
        if item.get("attempt_directory") != expected_relative:
            _fail(f"Incident complete-attempt layout drifted: {row.row_id}")
        attempt = output / expected_relative
        artifacts = item.get("artifacts")
        if type(artifacts) is not list or len(artifacts) != 8:
            _fail(f"Incident complete-attempt manifest cardinality drifted: {row.row_id}")
        expected_artifacts = {entry.get("path"): entry for entry in artifacts if type(entry) is dict}
        children = sorted(attempt.iterdir(), key=lambda path: path.name)
        current = _file_manifest(output, children)
        if [dict(entry) for entry in current] != artifacts:
            _fail(f"Incident complete-attempt bytes drifted: {row.row_id}")
        if runner._canonical_json(runner._load_json_object(attempt / "worker.spec.json")) != runner._canonical_json(row.worker_spec):
            _fail(f"Incident complete worker spec drifted: {row.row_id}")
        predecessor._verify_worker_spec(runner, row, attempt / "worker.spec.json")
        validation_context = predecessor.RecoveryContext(
            project_root=project_root,
            output=output,
            plan_path=plan_path,
            runner_path=runner_path,
            metric_path=metric_path,
            process_guard_path=process_guard_path,
            rows=tuple(rows),
            all_row_ids=all_row_ids,
            non_target_marker_manifest=(),
            preexisting_failed_attempt_manifest=(),
        )
        predecessor._validate_recovery_result(
            validation_context,
            runner,
            row,
            attempt,
            helper_sha256=EXPECTED_PREDECESSOR_HELPER_SHA256,
        )
        incident_complete_attempt_files.extend(children)
    attempts_root = output / "attempts"
    observed_roots = sorted(attempts_root.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() or not path.is_dir() for path in observed_roots):
        _fail("Incident attempts root contains an unsafe entry")
    expected_attempt_roots = (
        set(all_row_ids) - {row.row_id for row in rows[1:]}
    ) | {row.row_id for row in rows[1:5]}
    if {path.name for path in observed_roots} != expected_attempt_roots:
        _fail("Incident attempt-root set is not exact492")
    for row in rows:
        row_root = attempts_root / row.row_id
        observed = [path.name for path in sorted(row_root.iterdir())] if row_root.exists() else []
        if row.ordinal == 446 and observed != ["attempt-0001", "attempt-0002"]:
            _fail("Ordinal 446 incident attempt layout drifted")
        if row.ordinal in range(447, 451) and observed != ["attempt-0001"]:
            _fail(f"Complete incident attempt layout drifted: {row.row_id}")
        if row.ordinal >= 451 and observed:
            _fail(f"Fresh continuation row already has an attempt: {row.row_id}")
    predecessor_failed_attempt_manifest = predecessor._validate_failed_attempt(
        output,
        runner,
        rows[0],
        output / "attempts" / rows[0].row_id / "attempt-0001",
    )
    return ContinuationContext(
        project_root=project_root,
        output=output,
        plan_path=plan_path,
        runner_path=runner_path,
        metric_path=metric_path,
        process_guard_path=process_guard_path,
        predecessor_path=predecessor_path,
        rows=tuple(rows),
        incident_complete_rows=tuple(rows[:5]),
        fresh_rows=tuple(rows),
        all_row_ids=all_row_ids,
        preserved_marker_manifest=marker_manifest,
        incident_file_manifest=incident_manifest,
        incident_complete_attempt_manifest=_file_manifest(
            output, incident_complete_attempt_files
        ),
        predecessor_failed_attempt_manifest=(
            predecessor_failed_attempt_manifest
        ),
        external_scheduling_manifest=external_manifest,
    )


def _validate_external_scheduling_evidence(
    context: ContinuationContext,
    process_guard: ModuleType,
    runner: ModuleType,
    *,
    interpreter_identity: Mapping[str, object],
    environment_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Validate the frozen external-scheduling s01 custody chain in place."""
    _verify_manifest_unchanged(
        context.output,
        context.external_scheduling_manifest,
        label="external-scheduling s01 claim/handoff/receipt/seal",
    )
    external_context = process_guard._validate_plan_and_build_rows(
        context.project_root, context.output, runner
    )
    if (
        len(external_context.rows) != 42
        or [row.ordinal for row in external_context.rows]
        != list(range(463, 505))
        or any(
            not row.row_id.startswith(
                "v21e3-motsp-development-n500-s01__"
            )
            for row in external_context.rows
        )
    ):
        _fail("External-scheduling exact42 row design drifted")
    handoff, handoff_sha256 = process_guard._validate_handoff_receipt(
        external_context,
        runner,
        interpreter_identity=interpreter_identity,
        environment_receipt=environment_receipt,
        require_current_prefix=False,
    )
    if handoff_sha256 != EXPECTED_EXTERNAL_HANDOFF_SHA256:
        _fail("External-scheduling handoff raw SHA drifted")

    claim_keys = {
        "schema", "status", "scope", "target_case_id", "target_row_count",
        "target_row_ids", "worker_spec_payload_manifest",
        "worker_spec_payload_manifest_sha256", "jobs", "process_id",
        "plan_sha256", "source_snapshot_sha256", "frozen_runner_sha256",
        "helper_sha256", "handoff_receipt_path", "handoff_receipt_sha256",
        "handoff_receipt_payload_sha256", "interpreter_identity",
        "environment_receipt", "original_main_runner_honors_this_claim",
        "operational_quiescence_depends_on_external_stop_and_repeated_process_scan",
        "runtime_authority", "scientific_authority", "selection_authority",
        "claim_payload_sha256",
    }
    claim, claim_sha256 = process_guard._validate_bound_payload_file(
        context.output / EXTERNAL_CLAIM_NAME,
        runner,
        expected_keys=claim_keys,
        payload_field="claim_payload_sha256",
        label="external-scheduling s01 claim",
    )
    expected_worker_manifest = [
        {
            "ordinal": row.ordinal,
            "row_id": row.row_id,
            "worker_spec_payload_sha256": hashlib.sha256(
                _canonical_bytes(row.worker_spec)
            ).hexdigest(),
        }
        for row in external_context.rows
    ]
    if (
        claim_sha256 != EXPECTED_EXTERNAL_CLAIM_SHA256
        or claim["schema"]
        != "v21e3r1_external_scheduling_helper_instance_claim_v2"
        or claim["status"]
        != "SEALED_HELPER_INSTANCE_CLAIM_NOT_A_MAIN_RUNNER_LOCK"
        or claim["scope"]
        != "EXTERNAL_SCHEDULING_ONLY_FROZEN_DEVELOPMENT_ROWS"
        or claim["target_case_id"]
        != "v21e3-motsp-development-n500-s01"
        or claim["target_row_count"] != 42
        or claim["target_row_ids"]
        != [row.row_id for row in external_context.rows]
        or claim["worker_spec_payload_manifest"] != expected_worker_manifest
        or claim["worker_spec_payload_manifest_sha256"]
        != hashlib.sha256(_canonical_bytes(expected_worker_manifest)).hexdigest()
        or claim["jobs"] != 4
        or type(claim["process_id"]) is not int
        or claim["process_id"] <= 0
        or claim["plan_sha256"] != EXPECTED_PLAN_SHA256
        or claim["source_snapshot_sha256"] != EXPECTED_SOURCE_ROOT_SHA256
        or claim["frozen_runner_sha256"] != EXPECTED_RUNNER_SHA256
        or claim["helper_sha256"] != EXPECTED_PROCESS_GUARD_SHA256
        or claim["handoff_receipt_path"] != EXTERNAL_HANDOFF_NAME
        or claim["handoff_receipt_sha256"] != handoff_sha256
        or claim["handoff_receipt_payload_sha256"]
        != handoff["receipt_payload_sha256"]
        or claim["interpreter_identity"] != dict(interpreter_identity)
        or claim["environment_receipt"] != dict(environment_receipt)
        or claim["original_main_runner_honors_this_claim"] is not False
        or claim[
            "operational_quiescence_depends_on_external_stop_and_repeated_process_scan"
        ] is not True
        or claim["runtime_authority"] is not False
        or claim["scientific_authority"] is not False
        or claim["selection_authority"] is not False
    ):
        _fail("External-scheduling s01 claim semantic/cross-hash drifted")

    receipt_keys = {
        "schema", "status", "scope", "scheduling_policy", "target_case_id",
        "full_plan_row_count", "target_row_count", "jobs", "plan_path",
        "plan_sha256", "source_snapshot_sha256", "frozen_runner_path",
        "frozen_runner_sha256", "helper_path", "helper_sha256",
        "handoff_receipt_path", "handoff_receipt_sha256",
        "handoff_receipt_payload_sha256", "helper_instance_claim_path",
        "helper_instance_claim_sha256", "interpreter_identity",
        "environment_receipt", "completed_marker_count", "completed_markers",
        "completed_marker_generation", "completed_marker_verification",
        "worker_execution", "original_main_runner_honors_helper_instance_claim",
        "original_runner_resume_required",
        "original_runner_resume_after_helper_success_only", "receipt_seal_path",
        "case_generation_performed", "generated_case_count",
        "original_runner_or_algorithm_sources_modified",
        "implementation_independence", "algorithm_execution_independence",
        "scientific_independence", "runtime_authority", "scientific_authority",
        "selection_authority", "confirmation_authority",
        "formal_study_authority", "publication_status", "receipt_payload_sha256",
    }
    receipt, receipt_sha256 = process_guard._validate_bound_payload_file(
        context.output / EXTERNAL_RECEIPT_NAME,
        runner,
        expected_keys=receipt_keys,
        payload_field="receipt_payload_sha256",
        label="external-scheduling s01 success receipt",
    )
    if (
        receipt_sha256 != EXPECTED_EXTERNAL_RECEIPT_SHA256
        or receipt["schema"] != "v21e3r1_external_scheduling_only_receipt_v2"
        or receipt["status"] != "PASS_EXTERNAL_SCHEDULING_ONLY_TARGET_42"
        or receipt["scope"]
        != "EXTERNAL_SCHEDULING_ONLY_FROZEN_DEVELOPMENT_ROWS"
        or receipt["scheduling_policy"]
        != "THREAD_POOL_EXECUTOR_MAX_WORKERS_4"
        or receipt["target_case_id"]
        != "v21e3-motsp-development-n500-s01"
        or receipt["full_plan_row_count"] != EXPECTED_FULL_ROWS
        or receipt["target_row_count"] != 42
        or receipt["jobs"] != 4
        or receipt["plan_path"] != "diagnostic.plan.json"
        or receipt["plan_sha256"] != EXPECTED_PLAN_SHA256
        or receipt["source_snapshot_sha256"] != EXPECTED_SOURCE_ROOT_SHA256
        or receipt["frozen_runner_path"]
        != process_guard.RUNNER_RELATIVE.as_posix()
        or receipt["frozen_runner_sha256"] != EXPECTED_RUNNER_SHA256
        or receipt["helper_path"]
        != process_guard.HELPER_RELATIVE.as_posix()
        or receipt["helper_sha256"] != EXPECTED_PROCESS_GUARD_SHA256
        or receipt["handoff_receipt_path"] != EXTERNAL_HANDOFF_NAME
        or receipt["handoff_receipt_sha256"] != handoff_sha256
        or receipt["handoff_receipt_payload_sha256"]
        != handoff["receipt_payload_sha256"]
        or receipt["helper_instance_claim_path"] != EXTERNAL_CLAIM_NAME
        or receipt["helper_instance_claim_sha256"] != claim_sha256
        or receipt["interpreter_identity"] != dict(interpreter_identity)
        or receipt["environment_receipt"] != dict(environment_receipt)
        or receipt["completed_marker_count"] != 42
        or receipt["completed_marker_generation"]
        != (
            "HELPER_MATERIALIZED_ORIGINAL_FORMAT_ONLY_AFTER_FROZEN_WORKER_"
            "RESULT_ARTIFACT_VALIDATION"
        )
        or receipt["completed_marker_verification"]
        != "FROZEN_RUNNER_COMPLETED_PAYLOAD_REVALIDATED_PER_ROW_AND_FINAL"
        or receipt["worker_execution"]
        != "DELEGATED_TO_FROZEN_RUNNER_RUN_CHILD_AND_WORKER"
        or receipt["original_main_runner_honors_helper_instance_claim"] is not False
        or receipt["original_runner_resume_required"] is not True
        or receipt["original_runner_resume_after_helper_success_only"] is not True
        or receipt["receipt_seal_path"] != EXTERNAL_SEAL_NAME
        or receipt["case_generation_performed"] is not False
        or receipt["generated_case_count"] != 0
        or receipt["original_runner_or_algorithm_sources_modified"] is not False
        or receipt["implementation_independence"] is not False
        or receipt["algorithm_execution_independence"] is not False
        or receipt["scientific_independence"] is not False
        or any(
            receipt[field] is not False
            for field in (
                "runtime_authority", "scientific_authority",
                "selection_authority", "confirmation_authority",
                "formal_study_authority",
            )
        )
        or receipt["publication_status"] != "IJOC_HOLD"
    ):
        _fail("External-scheduling s01 success receipt semantic/cross-hash drifted")
    markers = receipt["completed_markers"]
    if type(markers) is not list or len(markers) != 42:
        _fail("External-scheduling s01 completed-marker cardinality drifted")
    marker_keys = {
        "ordinal", "row_id", "path", "sha256", "attempt_directory",
        "worker_spec_path", "worker_spec_sha256",
        "worker_spec_payload_sha256",
    }
    for row, marker in zip(external_context.rows, markers, strict=True):
        marker = _exact_keys(
            marker, marker_keys, label=f"external marker {row.row_id}"
        )
        expected_attempt = (
            Path("attempts") / row.row_id / "attempt-0001"
        ).as_posix()
        expected_marker_path = (
            Path("completed") / f"{row.row_id}.json"
        ).as_posix()
        expected_spec_path = (
            Path(expected_attempt) / "worker.spec.json"
        ).as_posix()
        spec_path = context.output / expected_spec_path
        spec_raw, spec_payload = process_guard._verify_worker_spec(
            runner,
            row,
            spec_path,
            expected_raw_sha256=_exact_sha256(
                marker["worker_spec_sha256"],
                label=f"external spec SHA {row.row_id}",
            ),
        )
        marker_path = context.output / expected_marker_path
        marker_sha = _exact_sha256(
            marker["sha256"], label=f"external marker SHA {row.row_id}"
        )
        completed = runner._completed_payload(context.output, row.row_id)
        if (
            marker["ordinal"] != row.ordinal
            or marker["row_id"] != row.row_id
            or marker["path"] != expected_marker_path
            or marker["attempt_directory"] != expected_attempt
            or marker["worker_spec_path"] != expected_spec_path
            or marker["worker_spec_sha256"] != spec_raw
            or marker["worker_spec_payload_sha256"] != spec_payload
            or marker_path.is_symlink()
            or not marker_path.is_file()
            or runner._sha256(marker_path) != marker_sha
            or completed is None
            or completed.get("attempt_directory") != expected_attempt
            or completed.get("plan_sha256") != EXPECTED_PLAN_SHA256
        ):
            _fail(f"External-scheduling row/marker cross-hash drifted: {row.row_id}")

    seal_keys = {
        "schema", "status", "receipt_path", "receipt_sha256",
        "receipt_payload_sha256", "helper_instance_claim_sha256",
        "handoff_receipt_sha256", "runtime_authority", "scientific_authority",
        "selection_authority", "seal_payload_sha256",
    }
    seal, seal_sha256 = process_guard._validate_bound_payload_file(
        context.output / EXTERNAL_SEAL_NAME,
        runner,
        expected_keys=seal_keys,
        payload_field="seal_payload_sha256",
        label="external-scheduling s01 success seal",
    )
    if (
        seal_sha256 != EXPECTED_EXTERNAL_SEAL_SHA256
        or seal["schema"]
        != "v21e3r1_external_scheduling_receipt_file_seal_v1"
        or seal["status"] != "PASS_SUCCESS_RECEIPT_FILE_DIGEST_SEALED"
        or seal["receipt_path"] != EXTERNAL_RECEIPT_NAME
        or seal["receipt_sha256"] != receipt_sha256
        or seal["receipt_payload_sha256"] != receipt["receipt_payload_sha256"]
        or seal["helper_instance_claim_sha256"] != claim_sha256
        or seal["handoff_receipt_sha256"] != handoff_sha256
        or seal["runtime_authority"] is not False
        or seal["scientific_authority"] is not False
        or seal["selection_authority"] is not False
    ):
        _fail("External-scheduling s01 success seal semantic/cross-hash drifted")
    _verify_manifest_unchanged(
        context.output,
        context.external_scheduling_manifest,
        label="external-scheduling s01 claim/handoff/receipt/seal",
    )
    core = {
        "schema": "v21e3r1_external_scheduling_s01_custody_binding_v1",
        "status": "PASS_HASH_BOUND_EXTERNAL_SCHEDULING_ONLY_NO_NEW_AUTHORITY",
        "target_case_id": "v21e3-motsp-development-n500-s01",
        "target_ordinals": list(range(463, 505)),
        "target_row_count": 42,
        "external_helper_sha256": EXPECTED_PROCESS_GUARD_SHA256,
        "handoff_path": EXTERNAL_HANDOFF_NAME,
        "handoff_sha256": handoff_sha256,
        "handoff_payload_sha256": handoff["receipt_payload_sha256"],
        "claim_path": EXTERNAL_CLAIM_NAME,
        "claim_sha256": claim_sha256,
        "claim_payload_sha256": claim["claim_payload_sha256"],
        "receipt_path": EXTERNAL_RECEIPT_NAME,
        "receipt_sha256": receipt_sha256,
        "receipt_payload_sha256": receipt["receipt_payload_sha256"],
        "seal_path": EXTERNAL_SEAL_NAME,
        "seal_sha256": seal_sha256,
        "seal_payload_sha256": seal["seal_payload_sha256"],
        "completed_marker_count": 42,
        "completed_marker_manifest_sha256": hashlib.sha256(
            _canonical_bytes(markers)
        ).hexdigest(),
        "external_evidence_manifest_sha256": _manifest_sha256(
            context.external_scheduling_manifest
        ),
        "implementation_independence": False,
        "algorithm_execution_independence": False,
        "scientific_independence": False,
        **_authority_hold_fields(),
    }
    return _bound_payload(core, digest_field="custody_payload_sha256")


def _normalized_path_text(value: str) -> str:
    return os.path.normcase(os.path.normpath(value.strip('"')))


def _classify_recovery_command(
    command: str,
    context: ContinuationContext,
    process_guard: ModuleType,
) -> tuple[str, str | None] | None:
    argv = process_guard._windows_command_line_to_argv(command)
    normalized = [_normalized_path_text(value) for value in argv]
    current_helper = _normalized_path_text(
        str((context.project_root / PROJECT_RELATIVE).resolve())
    )
    predecessor_helper = _normalized_path_text(str(context.predecessor_path))
    metric = _normalized_path_text(str(context.metric_path))
    current_form = current_helper in normalized
    predecessor_form = predecessor_helper in normalized
    metric_form = metric in normalized
    if not current_form and not predecessor_form and not metric_form:
        recognized_basenames = {
            os.path.normcase(Path(current_helper).name),
            os.path.normcase(Path(predecessor_helper).name),
            os.path.normcase(Path(metric).name),
            "run_v21e3r1_development_diagnostics.py",
        }
        ambiguous_script = any(
            os.path.normcase(os.path.basename(value.strip('"')))
            in recognized_basenames
            for value in argv
        )
        module_forms = {
            "ijoc_submission_v21e3r1.scripts."
            "run_v21e3r1_development_diagnostics",
            "ijoc_submission_v21e3r1.scripts."
            "run_v21e3r1_development_diagnostics.py",
        }
        ambiguous_module = any(
            value in module_forms
            and index > 0
            and argv[index - 1] == "-m"
            for index, value in enumerate(argv)
        )
        if ambiguous_script or ambiguous_module:
            return ("unknown", None)
        return None
    if sum((current_form, predecessor_form, metric_form)) != 1:
        return ("unknown", None)
    if predecessor_form:
        return ("predecessor", None)
    if current_form:
        try:
            value = process_guard._option_value(
                argv, "--continuation-worker-spec"
            )
        except RuntimeError:
            return ("unknown", None)
        if type(value) is not str or not Path(value).is_absolute():
            return ("unknown", value)
        spec = Path(value).resolve()
        try:
            spec.relative_to((context.output / "attempts").resolve())
        except ValueError:
            return ("unknown", spec.as_posix())
        return ("wrapper", spec.as_posix())
    try:
        trace_value = process_guard._option_value(argv, "--trace")
        output_value = process_guard._option_value(argv, "--output")
    except RuntimeError:
        return ("unknown", None)
    if (
        type(trace_value) is not str
        or type(output_value) is not str
        or not Path(trace_value).is_absolute()
        or not Path(output_value).is_absolute()
    ):
        return ("unknown", None)
    trace = Path(trace_value).resolve()
    metric_output = Path(output_value).resolve()
    if (
        trace.name != "trace.sqlite3"
        or metric_output.name != "independent.metric.json"
        or trace.parent != metric_output.parent
    ):
        return ("unknown", trace.as_posix())
    try:
        trace.parent.relative_to((context.output / "attempts").resolve())
    except ValueError:
        return ("unknown", trace.as_posix())
    return ("metric", (trace.parent / "worker.spec.json").as_posix())


def _scan_live_recovery_processes(
    context: ContinuationContext,
    process_guard: ModuleType,
) -> list[dict[str, object]]:
    if os.name != "nt":
        _fail("Continuation process scan is authorized only on Windows")
    powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / (
        "System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    script = (
        "$ErrorActionPreference='Stop';"
        "$rows=@(Get-CimInstance Win32_Process -ErrorAction Stop | "
        "Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine);"
        "ConvertTo-Json -Compress -Depth 3 -InputObject $rows"
    )
    completed = subprocess.run(
        [str(powershell), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
        text=True,
        encoding="utf-8",
        errors="strict",
        capture_output=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        _fail("Continuation process scan failed")
    raw = json.loads(completed.stdout)
    if type(raw) is dict:
        raw = [raw]
    if type(raw) is not list:
        _fail("Continuation process scan did not return an array")
    records: list[dict[str, object]] = []
    saw_current = False
    for item in raw:
        if type(item) is not dict or type(item.get("ProcessId")) is not int:
            _fail("Continuation process row type drifted")
        pid = item["ProcessId"]
        if pid == os.getpid():
            saw_current = True
            continue
        command = item.get("CommandLine")
        executable = item.get("ExecutablePath")
        if command is None:
            continue
        if type(command) is not str or (executable is not None and type(executable) is not str):
            _fail("Continuation process command type drifted")
        classification = _classify_recovery_command(
            command, context, process_guard
        )
        if classification is None:
            continue
        kind, worker_spec = classification
        records.append(
            {
                "kind": kind,
                "pid": pid,
                "parent_pid": item.get("ParentProcessId"),
                "executable_path": executable,
                "worker_spec_path": worker_spec,
                "command_line_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
            }
        )
    if not saw_current:
        _fail("Continuation process scan omitted current helper")
    return records


def _assert_no_live_processes(
    context: ContinuationContext,
    process_guard: ModuleType,
) -> dict[str, object]:
    original = process_guard._assert_no_conflicting_original_processes(context.output)
    records = _scan_live_recovery_processes(context, process_guard)
    if records:
        _fail("Active predecessor/continuation wrapper or metric process remains")
    core = {
        "schema": "v21e3r1_recovery_continuation_terminal_process_scan_v1",
        "original_process_scan_payload_sha256": original["scan_payload_sha256"],
        "matching_process_count": 0,
        "matching_processes": [],
    }
    return _bound_payload(core, digest_field="scan_payload_sha256")


def _assert_process_boundary(
    context: ContinuationContext,
    process_guard: ModuleType,
    *,
    allowed_worker_specs: set[str] | None = None,
) -> dict[str, object]:
    original = process_guard._assert_no_conflicting_original_processes(
        context.output
    )
    allowed = {
        Path(value).resolve().as_posix()
        for value in (allowed_worker_specs or set())
    }
    records = _scan_live_recovery_processes(context, process_guard)
    wrappers = {
        int(record["pid"]): record
        for record in records
        if record["kind"] == "wrapper"
        and record["parent_pid"] == os.getpid()
        and record["worker_spec_path"] in allowed
        and type(record["executable_path"]) is str
        and Path(str(record["executable_path"])).resolve()
        == EXPECTED_INTERPRETER_PATH.resolve()
    }
    for record in records:
        if record["kind"] == "wrapper" and int(record["pid"]) in wrappers:
            continue
        if (
            record["kind"] == "metric"
            and record["worker_spec_path"] in allowed
            and int(record["parent_pid"]) in wrappers
            and type(record["executable_path"]) is str
            and Path(str(record["executable_path"])).resolve()
            == EXPECTED_INTERPRETER_PATH.resolve()
        ):
            continue
        _fail("Active or ambiguous predecessor/continuation process targets output")
    core = {
        "schema": "v21e3r1_recovery_continuation_process_boundary_v1",
        "original_process_scan_payload_sha256": original["scan_payload_sha256"],
        "allowed_worker_specs": sorted(allowed),
        "recovery_processes": records,
    }
    return _bound_payload(core, digest_field="scan_payload_sha256")


def _wait_for_worker_specs_zero(
    context: ContinuationContext,
    process_guard: ModuleType,
    worker_specs: set[str],
    *,
    block_all: bool,
    timeout_seconds: float = DESCENDANT_ZERO_TIMEOUT_SECONDS,
) -> dict[str, object]:
    normalized = {Path(value).resolve().as_posix() for value in worker_specs}
    deadline = time.monotonic() + timeout_seconds
    scans = 0
    while True:
        scans += 1
        original = process_guard._assert_no_conflicting_original_processes(
            context.output
        )
        records = _scan_live_recovery_processes(context, process_guard)
        blockers = [
            record
            for record in records
            if block_all
            or record["kind"] in {"unknown", "predecessor"}
            or record["worker_spec_path"] in normalized
        ]
        if not blockers:
            core = {
                "schema": "v21e3r1_recovery_continuation_descendant_zero_v1",
                "worker_specs": sorted(normalized),
                "block_all": block_all,
                "scan_count": scans,
                "terminal_matching_process_count": 0,
                "original_process_scan_payload_sha256": original[
                    "scan_payload_sha256"
                ],
            }
            return _bound_payload(core, digest_field="scan_payload_sha256")
        if time.monotonic() >= deadline:
            raise DescendantTerminationUnconfirmed(
                "Continuation descendant process state remained NOT_TERMINAL"
            )
        time.sleep(0.1)


def _run_in_windows_job_with_accounting_grace(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
    accounting_grace_seconds: float,
    start_gate_line: str,
    terminal_zero_check: Callable[[], Mapping[str, object]],
    job_factory: Callable[[], object],
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    if (
        type(command) not in (list, tuple)
        or not command
        or any(type(value) is not str or not value for value in command)
        or type(timeout_seconds) not in (int, float)
        or isinstance(timeout_seconds, bool)
        or timeout_seconds <= 0
        or type(accounting_grace_seconds) not in (int, float)
        or isinstance(accounting_grace_seconds, bool)
        or accounting_grace_seconds <= 0
        or type(start_gate_line) is not str
        or not start_gate_line
    ):
        _fail("Continuation Windows Job invocation contract drifted")
    job = job_factory()
    process: subprocess.Popen[str] | None = None
    assigned = False
    timed_out = False
    primary_error: BaseException | None = None
    stdout = ""
    stderr = ""
    returncode: int | None = None
    initial_active: int | None = None
    accounting_lag_observed = False
    accounting_lag_drained = False
    accounting_grace_expired = False
    accounting_wait_seconds = 0.0
    terminal_active: int | None = None
    terminal_scan: Mapping[str, object] | None = None
    cleanup_errors: list[str] = []
    try:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        job.assign_and_verify(process)
        assigned = True
        try:
            stdout, stderr = process.communicate(
                input=start_gate_line, timeout=timeout_seconds
            )
            returncode = process.returncode
            initial_active = job.active_processes()
            if initial_active > 0:
                accounting_lag_observed = True
                started = time.monotonic()
                try:
                    job.wait_active_zero(accounting_grace_seconds)
                    accounting_lag_drained = True
                except BaseException as error:
                    accounting_grace_expired = True
                    primary_error = RuntimeError(
                        "Continuation Job accounting/descendant state remained "
                        "active beyond the bounded grace period"
                    )
                    primary_error.__cause__ = error
                finally:
                    accounting_wait_seconds = time.monotonic() - started
        except subprocess.TimeoutExpired as error:
            timed_out = True
            primary_error = error
        except BaseException as error:
            primary_error = error
    except BaseException as error:
        primary_error = error
    finally:
        if process is not None:
            try:
                alive = process.poll() is None
                active = job.active_processes() if assigned else 0
                if alive or active != 0 or primary_error is not None:
                    if assigned:
                        job.terminate()
                    elif alive:
                        process.kill()
                try:
                    drained_stdout, drained_stderr = process.communicate(
                        timeout=DESCENDANT_ZERO_TIMEOUT_SECONDS
                    )
                    if not stdout:
                        stdout = drained_stdout
                    if not stderr:
                        stderr = drained_stderr
                    returncode = process.returncode
                except BaseException as error:
                    cleanup_errors.append("wrapper wait/drain failed: " + repr(error))
            except BaseException as error:
                cleanup_errors.append("wrapper termination failed: " + repr(error))
        try:
            terminal_active = job.wait_active_zero(
                DESCENDANT_ZERO_TIMEOUT_SECONDS
            )
        except BaseException as error:
            cleanup_errors.append("Job Object did not reach zero: " + repr(error))
        try:
            job.close()
        except BaseException as error:
            cleanup_errors.append("Job Object close failed: " + repr(error))
        try:
            terminal_scan = dict(terminal_zero_check())
        except BaseException as error:
            cleanup_errors.append("terminal process rescan failed: " + repr(error))
    if cleanup_errors or terminal_active != 0 or terminal_scan is None:
        raise DescendantTerminationUnconfirmed(
            "Continuation wrapper descendant termination is NOT_TERMINAL: "
            + "; ".join(cleanup_errors)
        ) from primary_error
    witness = {
        "schema": "v21e3r1_continuation_windows_job_witness_v1",
        "kill_on_job_close_limit": True,
        "job_limit_flags": 0x00002000,
        "wrapper_pid": process.pid if process is not None else None,
        "job_assignment_verified_before_gate_release": assigned,
        "outer_timeout_seconds": timeout_seconds,
        "outer_timeout_fired": timed_out,
        "wrapper_returncode": returncode,
        "initial_active_processes_after_wrapper_exit": initial_active,
        "accounting_grace_seconds": accounting_grace_seconds,
        "accounting_lag_observed": accounting_lag_observed,
        "accounting_lag_drained_without_termination": accounting_lag_drained,
        "accounting_grace_expired": accounting_grace_expired,
        "accounting_wait_seconds": accounting_wait_seconds,
        "terminal_active_processes": terminal_active,
        "terminal_process_scan": dict(terminal_scan),
    }
    if timed_out:
        raise JobControlledProcessTimeout(
            "Continuation wrapper exceeded the outer timeout after its Job tree reached zero",
            witness,
        ) from primary_error
    if primary_error is not None:
        raise RuntimeError(
            "Continuation wrapper failed after its Job tree reached zero: "
            + str(primary_error)
        ) from primary_error
    if returncode is None:
        _fail("Continuation wrapper return code was not observed")
    return (
        subprocess.CompletedProcess(list(command), returncode, stdout, stderr),
        witness,
    )


def preflight(
    project_root: str | Path,
    output_directory: str | Path,
    *,
    expected_helper_sha256: str,
) -> tuple[
    ContinuationContext,
    ModuleType,
    ModuleType,
    ModuleType,
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    root = Path(project_root).resolve()
    output = Path(output_directory).resolve()
    expected_helper_sha256 = _validate_expected_helper_identity(
        root, expected_helper_sha256
    )
    predecessor = _load_predecessor(root)
    predecessor._validate_production_paths(root, output)
    process_guard = predecessor._load_process_guard(root)
    interpreter_identity = predecessor._validate_interpreter(process_guard)
    environment_receipt = process_guard._execution_environment_receipt(
        root, interpreter_identity
    )
    runner = predecessor._load_frozen_runner(root)
    context = _validate_incident(root, output, predecessor, runner)
    if _sha256(root / PROJECT_RELATIVE) != expected_helper_sha256:
        _fail("Continuation helper drifted during preflight")
    external_scheduling_custody = _validate_external_scheduling_evidence(
        context,
        process_guard,
        runner,
        interpreter_identity=interpreter_identity,
        environment_receipt=environment_receipt,
    )
    process_scan = _assert_no_live_processes(context, process_guard)
    incident_core = {
        "schema": "v21e3r1_metric_timeout_recovery_continuation_incident_v1",
        "predecessor_helper_sha256": EXPECTED_PREDECESSOR_HELPER_SHA256,
        "continuation_helper_sha256": expected_helper_sha256,
        "old_claim_sha256": EXPECTED_OLD_CLAIM_SHA256,
        "old_failure_receipt_sha256": EXPECTED_OLD_FAILURE_SHA256,
        "old_failure_seal_sha256": EXPECTED_OLD_FAILURE_SEAL_SHA256,
        "preserved_marker_count": len(context.preserved_marker_manifest),
        "preserved_marker_manifest_sha256": _manifest_sha256(context.preserved_marker_manifest),
        "incident_complete_attempt_count": len(context.incident_complete_rows),
        "incident_complete_attempt_adopted_count": 0,
        "incident_complete_attempts_not_adopted": True,
        "incident_complete_attempt_manifest_sha256": _manifest_sha256(
            context.incident_complete_attempt_manifest
        ),
        "predecessor_failed_attempt_manifest_sha256": _manifest_sha256(
            context.predecessor_failed_attempt_manifest
        ),
        "external_scheduling_custody": external_scheduling_custody,
        "external_scheduling_manifest": list(
            context.external_scheduling_manifest
        ),
        "external_scheduling_manifest_sha256": _manifest_sha256(
            context.external_scheduling_manifest
        ),
        "missing_recovery_marker_count": len(context.rows),
        "fresh_full_algorithm_rerun_count": len(context.fresh_rows),
        "terminal_process_scan": process_scan,
        "old_success_absent": True,
        **_authority_hold_fields(),
    }
    incident_receipt = _bound_payload(incident_core, digest_field="incident_payload_sha256")
    _validate_expected_helper_identity(root, expected_helper_sha256)
    return (
        context,
        predecessor,
        runner,
        process_guard,
        interpreter_identity,
        environment_receipt,
        incident_receipt,
    )


def _claim_payload(
    context: ContinuationContext,
    *,
    helper_sha256: str,
    interpreter_identity: Mapping[str, object],
    environment_receipt: Mapping[str, object],
    incident_receipt: Mapping[str, object],
    preclaim_process_scan: Mapping[str, object],
) -> dict[str, object]:
    worker_manifest = [
        {
            "ordinal": row.ordinal,
            "row_id": row.row_id,
            "expected_attempt_number": row.expected_attempt_number,
            "worker_spec_payload_sha256": hashlib.sha256(
                _canonical_bytes(row.worker_spec)
            ).hexdigest(),
        }
        for row in context.fresh_rows
    ]
    core = {
        "schema": "v21e3r1_metric_timeout_recovery_continuation_claim_v1",
        "status": "SEALED_APPEND_ONLY_CONTINUATION_INSTANCE_CLAIM",
        "scope": "EXACT_ORDINALS_446_462_FROZEN_DEVELOPMENT_CONTINUATION_ONLY",
        "continuation_semantics": CONTINUATION_SEMANTICS,
        "process_id": os.getpid(),
        "target_case_id": TARGET_CASE_ID,
        "target_ordinals": list(RECOVERY_ORDINALS),
        "target_row_count": len(context.rows),
        "old_complete_attempt_count": len(context.incident_complete_rows),
        "old_complete_attempt_adopted_count": 0,
        "old_complete_attempts_not_adopted": True,
        "fresh_full_algorithm_rerun_count": len(context.fresh_rows),
        "worker_spec_payload_manifest": worker_manifest,
        "worker_spec_payload_manifest_sha256": hashlib.sha256(
            _canonical_bytes(worker_manifest)
        ).hexdigest(),
        "incident_receipt": dict(incident_receipt),
        "predecessor_helper_sha256": EXPECTED_PREDECESSOR_HELPER_SHA256,
        "predecessor_claim_sha256": EXPECTED_OLD_CLAIM_SHA256,
        "predecessor_failure_receipt_sha256": EXPECTED_OLD_FAILURE_SHA256,
        "predecessor_failure_seal_sha256": EXPECTED_OLD_FAILURE_SEAL_SHA256,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "source_snapshot_sha256": EXPECTED_SOURCE_ROOT_SHA256,
        "frozen_runner_sha256": EXPECTED_RUNNER_SHA256,
        "independent_metric_source_sha256": EXPECTED_METRIC_SHA256,
        "process_guard_sha256": EXPECTED_PROCESS_GUARD_SHA256,
        "helper_sha256": helper_sha256,
        "interpreter_identity": dict(interpreter_identity),
        "environment_receipt": dict(environment_receipt),
        "preclaim_process_scan": dict(preclaim_process_scan),
        "jobs": JOBS,
        "original_metric_timeout_seconds": ORIGINAL_METRIC_TIMEOUT_SECONDS,
        "operational_metric_timeout_seconds": OPERATIONAL_METRIC_TIMEOUT_SECONDS,
        "outer_row_timeout_seconds": OUTER_ROW_TIMEOUT_SECONDS,
        "accounting_grace_seconds": ACCOUNTING_GRACE_SECONDS,
        "original_main_runner_honors_this_claim": False,
        "automatic_resume_authorized": False,
        **_authority_hold_fields(),
    }
    return _bound_payload(core, digest_field="claim_payload_sha256")


def _validate_claim(
    context: ContinuationContext,
    predecessor: ModuleType,
    runner: ModuleType,
    *,
    expected_claim: Mapping[str, object],
    expected_sha256: str,
) -> None:
    actual = predecessor._validate_bound_json(
        context.output / CLAIM_NAME,
        runner,
        expected=expected_claim,
        digest_field="claim_payload_sha256",
        label="continuation claim",
    )
    if actual != expected_sha256:
        _fail("Continuation claim raw bytes drifted")


def _claim_is_owned(
    context: ContinuationContext,
    predecessor: ModuleType,
    runner: ModuleType,
    *,
    expected_claim: Mapping[str, object],
    expected_sha256: str,
) -> bool:
    try:
        _validate_claim(
            context,
            predecessor,
            runner,
            expected_claim=expected_claim,
            expected_sha256=expected_sha256,
        )
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError):
        return False
    return True


def _verify_static_boundary(
    context: ContinuationContext,
    predecessor: ModuleType,
    runner: ModuleType,
    process_guard: ModuleType,
    *,
    helper_sha256: str,
    interpreter_identity: Mapping[str, object],
    environment_receipt: Mapping[str, object],
) -> None:
    helper_path = (context.project_root / PROJECT_RELATIVE).resolve()
    if (
        _sha256(helper_path) != helper_sha256
        or _sha256(context.predecessor_path)
        != EXPECTED_PREDECESSOR_HELPER_SHA256
        or _sha256(context.plan_path) != EXPECTED_PLAN_SHA256
        or _sha256(context.runner_path) != EXPECTED_RUNNER_SHA256
        or _sha256(context.metric_path) != EXPECTED_METRIC_SHA256
        or _sha256(context.process_guard_path) != EXPECTED_PROCESS_GUARD_SHA256
        or runner._source_manifest(context.project_root).get(
            "source_snapshot_sha256"
        )
        != EXPECTED_SOURCE_ROOT_SHA256
        or predecessor._validate_interpreter(process_guard)
        != dict(interpreter_identity)
        or process_guard._execution_environment_receipt(
            context.project_root, interpreter_identity
        )
        != dict(environment_receipt)
    ):
        _fail("Continuation static source/interpreter boundary drifted")
    _verify_incident_immutable(context)


def _assert_continuation_state(
    context: ContinuationContext,
    runner: ModuleType,
    *,
    owned_attempts: Mapping[str, Path],
    sealed_rows: set[str],
) -> None:
    for row in context.rows:
        row_root = context.output / "attempts" / row.row_id
        marker = context.output / "completed" / f"{row.row_id}.json"
        owned = owned_attempts.get(row.row_id)
        predecessor_names = (
            {"attempt-0001", "attempt-0002"}
            if row.ordinal == 446
            else {"attempt-0001"}
            if row.ordinal <= 450
            else set()
        )
        expected_names = set(predecessor_names)
        if owned is not None:
            expected = row_root / f"attempt-{row.expected_attempt_number:04d}"
            if owned.resolve() != expected.resolve() or not owned.is_dir():
                _fail(f"Continuation owned attempt path drifted: {row.row_id}")
            expected_names.add(expected.name)
        observed_names = (
            {path.name for path in row_root.iterdir()} if row_root.exists() else set()
        )
        if observed_names != expected_names:
            _fail(f"Continuation attempt layout drifted: {row.row_id}")
        if marker.exists() or marker.is_symlink():
            if row.row_id not in sealed_rows or owned is None:
                _fail(f"Non-continuation marker appeared: {row.row_id}")
            completed = runner._completed_payload(context.output, row.row_id)
            if (
                completed is None
                or completed.get("row_id") != row.row_id
                or completed.get("attempt_directory")
                != owned.relative_to(context.output).as_posix()
                or completed.get("plan_sha256") != EXPECTED_PLAN_SHA256
            ):
                _fail(f"Continuation marker drifted: {row.row_id}")
        elif row.row_id in sealed_rows:
            _fail(f"Continuation marker disappeared: {row.row_id}")


def _validate_worker_claim(
    output: Path,
    runner: ModuleType,
    *,
    spec: Mapping[str, object],
    spec_path: Path,
    helper_sha256: str,
    claim_sha256: str,
) -> dict[str, object]:
    claim_path = output / CLAIM_NAME
    if runner._sha256(claim_path) != _exact_sha256(
        claim_sha256, label="continuation worker claim SHA"
    ):
        _fail("Continuation worker claim raw bytes drifted")
    claim = runner._load_json_object(claim_path)
    _validate_bound_payload(
        claim, digest_field="claim_payload_sha256", label="continuation claim"
    )
    row_id = (
        f"{spec.get('case_id')}__seed-{spec.get('seed')}__arm-"
        f"{str(spec.get('arm_id')).lower()}"
    )
    manifest = claim.get("worker_spec_payload_manifest")
    matches = [
        entry
        for entry in manifest
        if type(entry) is dict and entry.get("row_id") == row_id
    ] if type(manifest) is list else []
    if (
        claim.get("schema")
        != "v21e3r1_metric_timeout_recovery_continuation_claim_v1"
        or claim.get("status")
        != "SEALED_APPEND_ONLY_CONTINUATION_INSTANCE_CLAIM"
        or claim.get("continuation_semantics") != CONTINUATION_SEMANTICS
        or claim.get("helper_sha256") != helper_sha256
        or claim.get("target_ordinals") != list(RECOVERY_ORDINALS)
        or claim.get("fresh_full_algorithm_rerun_count") != 17
        or claim.get("old_complete_attempt_adopted_count") != 0
        or type(claim.get("process_id")) is not int
        or claim.get("process_id") != os.getppid()
        or len(matches) != 1
        or matches[0].get("worker_spec_payload_sha256")
        != hashlib.sha256(_canonical_bytes(dict(spec))).hexdigest()
    ):
        _fail("Continuation worker claim-to-spec binding drifted")
    expected_attempt = matches[0].get("expected_attempt_number")
    if (
        type(expected_attempt) is not int
        or spec_path.name != "worker.spec.json"
        or spec_path.parent.name != f"attempt-{expected_attempt:04d}"
        or spec_path.parent.parent.name != row_id
    ):
        _fail("Continuation worker attempt layout drifted")
    return claim


def run_continuation_worker(
    spec_path: str | Path,
    *,
    helper_sha256: str,
    claim_sha256: str,
) -> dict[str, object]:
    if (
        threading.active_count() != 1
        or threading.current_thread() is not threading.main_thread()
    ):
        _fail("Continuation monkeypatch is authorized only in a single-threaded child")
    helper_path = Path(__file__).resolve()
    project_root = helper_path.parents[2]
    output = (project_root / OUTPUT_RELATIVE).resolve()
    helper_sha256 = _exact_sha256(helper_sha256, label="continuation helper SHA")
    if _sha256(helper_path) != helper_sha256:
        _fail("Continuation helper source drifted in worker")
    predecessor = _load_predecessor(project_root)
    predecessor._validate_production_paths(project_root, output)
    process_guard = predecessor._load_process_guard(project_root)
    interpreter_identity = predecessor._validate_interpreter(process_guard)
    runner = predecessor._load_frozen_runner(project_root)
    spec_file = Path(spec_path).resolve()
    try:
        spec_file.relative_to((output / "attempts").resolve())
    except ValueError as error:
        raise RuntimeError("Continuation worker spec escaped attempts root") from error
    spec = _exact_keys(
        runner._load_json_object(spec_file),
        {
            "schema", "project_root", "case_id", "family", "size",
            "case_path", "case_artifact_sha256", "objective_lower_bounds",
            "objective_upper_bounds", "reference_directions", "seed",
            "arm_id", "charged_evaluation_budget", "checkpoint_period",
            "source_snapshot_sha256", "plan_sha256",
        },
        label="continuation worker spec",
    )
    if (
        spec["schema"] != "v21e3r1_diagnostic_row_worker_spec_v1"
        or Path(str(spec["project_root"])).resolve() != project_root
        or spec["case_id"] != TARGET_CASE_ID
        or spec["family"] != "MOTSP"
        or spec["size"] != 500
        or spec["charged_evaluation_budget"] != 2000
        or spec["checkpoint_period"] != 200
        or spec["source_snapshot_sha256"] != EXPECTED_SOURCE_ROOT_SHA256
        or spec["plan_sha256"] != EXPECTED_PLAN_SHA256
    ):
        _fail("Continuation worker spec semantic contract drifted")
    _validate_worker_claim(
        output,
        runner,
        spec=spec,
        spec_path=spec_file,
        helper_sha256=helper_sha256,
        claim_sha256=claim_sha256,
    )
    spec_sha256 = runner._sha256(spec_file)

    def verify_boundary() -> None:
        if (
            _sha256(helper_path) != helper_sha256
            or _sha256(project_root / PREDECESSOR_RELATIVE)
            != EXPECTED_PREDECESSOR_HELPER_SHA256
            or _sha256(output / "diagnostic.plan.json") != EXPECTED_PLAN_SHA256
            or _sha256(project_root / predecessor.RUNNER_RELATIVE)
            != EXPECTED_RUNNER_SHA256
            or _sha256(project_root / predecessor.METRIC_RELATIVE)
            != EXPECTED_METRIC_SHA256
            or _sha256(project_root / predecessor.PROCESS_GUARD_RELATIVE)
            != EXPECTED_PROCESS_GUARD_SHA256
            or runner._source_manifest(project_root).get("source_snapshot_sha256")
            != EXPECTED_SOURCE_ROOT_SHA256
            or runner._sha256(spec_file) != spec_sha256
            or runner._sha256(output / CLAIM_NAME) != claim_sha256
            or predecessor._validate_interpreter(process_guard)
            != interpreter_identity
            or _sha256(output / OLD_CLAIM_NAME) != EXPECTED_OLD_CLAIM_SHA256
            or _sha256(output / OLD_FAILURE_NAME) != EXPECTED_OLD_FAILURE_SHA256
            or _sha256(output / OLD_FAILURE_SEAL_NAME)
            != EXPECTED_OLD_FAILURE_SEAL_SHA256
        ):
            _fail("Continuation worker boundary drifted around metric subprocess")

    process_guard._assert_no_conflicting_original_processes(output)
    worker_result, witness = predecessor._execute_frozen_worker_with_timeout_override(
        runner,
        spec_file,
        project_root=project_root,
        helper_sha256=helper_sha256,
        interpreter_identity=interpreter_identity,
        verify_boundary=verify_boundary,
    )
    process_guard._assert_no_conflicting_original_processes(output)
    verify_boundary()
    return {
        "worker_result_sha256": runner._sha256(
            spec_file.parent / "worker.result.json"
        ),
        "timeout_witness_sha256": runner._sha256(
            spec_file.parent / predecessor.OVERRIDE_WITNESS_NAME
        ),
        "worker_status": worker_result["status"],
        "timeout_witness_status": witness["status"],
    }


def _run_continuation_child(
    context: ContinuationContext,
    predecessor: ModuleType,
    process_guard: ModuleType,
    spec_path: Path,
    *,
    helper_sha256: str,
    claim_sha256: str,
) -> dict[str, object]:
    command = [
        str(EXPECTED_INTERPRETER_PATH),
        str((context.project_root / PROJECT_RELATIVE).resolve()),
        "--continuation-worker-spec",
        str(spec_path.resolve()),
        "--helper-sha256",
        helper_sha256,
        "--claim-sha256",
        claim_sha256,
        "--job-start-gate",
        JOB_START_GATE_ARGUMENT,
    ]
    environment = dict(os.environ)
    current_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(context.project_root)
        if not current_path
        else str(context.project_root) + os.pathsep + current_path
    )
    completed, job_witness = _run_in_windows_job_with_accounting_grace(
        command,
        cwd=context.project_root,
        environment=environment,
        timeout_seconds=OUTER_ROW_TIMEOUT_SECONDS,
        accounting_grace_seconds=ACCOUNTING_GRACE_SECONDS,
        start_gate_line=JOB_START_GATE_LINE,
        terminal_zero_check=lambda: _wait_for_worker_specs_zero(
            context,
            process_guard,
            {spec_path.resolve().as_posix()},
            block_all=False,
        ),
        job_factory=predecessor._WindowsKillOnCloseJob,
    )
    if completed.returncode != 0:
        _fail("Continuation worker failed: " + completed.stderr[-2000:])
    if (
        job_witness.get("schema")
        != "v21e3r1_continuation_windows_job_witness_v1"
        or job_witness.get("job_assignment_verified_before_gate_release") is not True
        or job_witness.get("outer_timeout_fired") is not False
        or job_witness.get("wrapper_returncode") != 0
        or job_witness.get("terminal_active_processes") != 0
        or job_witness.get("accounting_grace_expired") is not False
    ):
        _fail("Continuation corrected Job witness drifted")
    return job_witness


def _write_quarantine(
    context: ContinuationContext,
    predecessor: ModuleType,
    runner: ModuleType,
    *,
    error: BaseException,
    phase: str,
    helper_sha256: str,
    claim: Mapping[str, object],
    claim_sha256: str,
    owned_attempts: Mapping[str, Path],
) -> dict[str, object]:
    path = context.output / QUARANTINE_NAME
    if path.exists() or path.is_symlink():
        _fail("Continuation quarantine already exists")
    _validate_claim(
        context,
        predecessor,
        runner,
        expected_claim=claim,
        expected_sha256=claim_sha256,
    )
    core = {
        "schema": "v21e3r1_metric_timeout_recovery_continuation_quarantine_v1",
        "status": "NOT_TERMINAL_DESCENDANT_TERMINATION_UNCONFIRMED",
        "scope": "EXACT_ORDINALS_446_462_FROZEN_DEVELOPMENT_CONTINUATION_ONLY",
        "continuation_semantics": CONTINUATION_SEMANTICS,
        "failure_phase": phase,
        "exception_type": type(error).__name__,
        "exception_message": str(error)[-2000:],
        "helper_sha256": helper_sha256,
        "helper_instance_claim_sha256": claim_sha256,
        "predecessor_failure_receipt_sha256": EXPECTED_OLD_FAILURE_SHA256,
        "owned_attempt_directories_without_terminal_hash_claim": [
            {
                "row_id": row_id,
                "attempt_directory": attempt.relative_to(context.output).as_posix(),
            }
            for row_id, attempt in sorted(owned_attempts.items())
        ],
        "old_complete_attempt_count": 5,
        "old_complete_attempt_adopted_count": 0,
        "terminal_descendant_state_confirmed": False,
        "durable_terminal_failure": False,
        "attempt_artifact_hashes_authorized": False,
        "automatic_resume_authorized": False,
        "main_runner_resume_authorized": False,
        "manual_process_and_artifact_audit_required": True,
        "aggregate_materialized": False,
        "diagnostic_receipt_materialized": False,
        **_authority_hold_fields(),
    }
    payload = _bound_payload(core, digest_field="quarantine_payload_sha256")
    runner._exclusive_json(path, payload)
    predecessor._fsync_file(path)
    predecessor._validate_bound_json(
        path,
        runner,
        expected=payload,
        digest_field="quarantine_payload_sha256",
        label="continuation descendant quarantine",
    )
    _validate_claim(
        context,
        predecessor,
        runner,
        expected_claim=claim,
        expected_sha256=claim_sha256,
    )
    return payload


def _write_failure(
    context: ContinuationContext,
    predecessor: ModuleType,
    runner: ModuleType,
    *,
    error: BaseException,
    phase: str,
    helper_sha256: str,
    claim: Mapping[str, object],
    claim_sha256: str,
    owned_attempts: Mapping[str, Path],
    sealed_entries: Mapping[str, Mapping[str, object]],
    terminal_process_scan: Mapping[str, object],
) -> dict[str, object]:
    path = context.output / FAILURE_NAME
    seal_path = context.output / FAILURE_SEAL_NAME
    if path.exists() or seal_path.exists() or path.is_symlink() or seal_path.is_symlink():
        _fail("Continuation failure evidence already exists")
    _validate_claim(
        context,
        predecessor,
        runner,
        expected_claim=claim,
        expected_sha256=claim_sha256,
    )
    _verify_incident_immutable(context)
    attempts: list[dict[str, object]] = []
    for row_id, attempt in sorted(owned_attempts.items()):
        children = sorted(attempt.iterdir(), key=lambda item: item.name)
        attempts.append(
            {
                "row_id": row_id,
                "attempt_directory": attempt.relative_to(context.output).as_posix(),
                "artifacts": list(_file_manifest(context.output, children)),
            }
        )
    core = {
        "schema": "v21e3r1_metric_timeout_recovery_continuation_failure_v1",
        "status": "HOLD_CONTINUATION_FAILURE_MANUAL_AUDIT_REQUIRED",
        "scope": "EXACT_ORDINALS_446_462_FROZEN_DEVELOPMENT_CONTINUATION_ONLY",
        "continuation_semantics": CONTINUATION_SEMANTICS,
        "failure_phase": phase,
        "exception_type": type(error).__name__,
        "exception_message": str(error)[-2000:],
        "target_ordinals": list(RECOVERY_ORDINALS),
        "helper_sha256": helper_sha256,
        "helper_instance_claim_sha256": claim_sha256,
        "predecessor_helper_sha256": EXPECTED_PREDECESSOR_HELPER_SHA256,
        "predecessor_claim_sha256": EXPECTED_OLD_CLAIM_SHA256,
        "predecessor_failure_receipt_sha256": EXPECTED_OLD_FAILURE_SHA256,
        "predecessor_failure_seal_sha256": EXPECTED_OLD_FAILURE_SEAL_SHA256,
        "old_complete_attempt_count": 5,
        "old_complete_attempt_adopted_count": 0,
        "owned_fresh_attempts": attempts,
        "validated_completed_markers": [
            dict(sealed_entries[row_id]) for row_id in sorted(sealed_entries)
        ],
        "terminal_process_scan": dict(terminal_process_scan),
        "terminal_descendant_state_confirmed": True,
        "automatic_retry_authorized": False,
        "main_runner_resume_authorized": False,
        "partial_markers_main_runner_resume_prohibited": True,
        "manual_audit_required": True,
        "aggregate_materialized": False,
        "diagnostic_receipt_materialized": False,
        **_authority_hold_fields(),
    }
    payload = _bound_payload(core, digest_field="receipt_payload_sha256")
    runner._exclusive_json(path, payload)
    predecessor._fsync_file(path)
    receipt_sha = predecessor._validate_bound_json(
        path,
        runner,
        expected=payload,
        digest_field="receipt_payload_sha256",
        label="continuation failure receipt",
    )
    _validate_claim(
        context,
        predecessor,
        runner,
        expected_claim=claim,
        expected_sha256=claim_sha256,
    )
    seal_core = {
        "schema": "v21e3r1_metric_timeout_recovery_continuation_failure_seal_v1",
        "status": "SEALED_CONTINUATION_DURABLE_FAILURE",
        "failure_receipt_path": FAILURE_NAME,
        "failure_receipt_sha256": receipt_sha,
        "failure_receipt_payload_sha256": payload["receipt_payload_sha256"],
        "helper_instance_claim_sha256": claim_sha256,
        **_authority_hold_fields(),
    }
    seal = _bound_payload(seal_core, digest_field="seal_payload_sha256")
    runner._exclusive_json(seal_path, seal)
    predecessor._fsync_file(seal_path)
    predecessor._validate_bound_json(
        seal_path,
        runner,
        expected=seal,
        digest_field="seal_payload_sha256",
        label="continuation failure seal",
    )
    _validate_claim(
        context,
        predecessor,
        runner,
        expected_claim=claim,
        expected_sha256=claim_sha256,
    )
    return payload


def _assert_success_side_evidence_absent(context: ContinuationContext) -> None:
    for name in (FAILURE_NAME, FAILURE_SEAL_NAME, QUARANTINE_NAME):
        path = context.output / name
        if path.exists() or path.is_symlink():
            _fail("Continuation failure/quarantine evidence appeared on success path")


def _write_row_job_witness(
    context: ContinuationContext,
    predecessor: ModuleType,
    runner: ModuleType,
    row: ContinuationRow,
    attempt: Path,
    *,
    helper_sha256: str,
    claim_sha256: str,
    job_witness: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    path = attempt / JOB_WITNESS_NAME
    if path.exists() or path.is_symlink():
        _fail(f"Continuation row Job witness already exists: {row.row_id}")
    core = {
        "schema": "v21e3r1_continuation_row_windows_job_receipt_v1",
        "status": "PASS_CORRECTED_WINDOWS_JOB_CONTAINMENT_AND_TERMINAL_ZERO",
        "scope": "ONE_FRESH_FROZEN_DEVELOPMENT_RECOVERY_RERUN_ONLY",
        "continuation_semantics": CONTINUATION_SEMANTICS,
        "ordinal": row.ordinal,
        "row_id": row.row_id,
        "attempt_directory": attempt.relative_to(context.output).as_posix(),
        "worker_spec_sha256": runner._sha256(attempt / "worker.spec.json"),
        "worker_result_sha256": runner._sha256(attempt / "worker.result.json"),
        "timeout_witness_sha256": runner._sha256(
            attempt / predecessor.OVERRIDE_WITNESS_NAME
        ),
        "job_control": dict(job_witness),
        "helper_sha256": helper_sha256,
        "helper_instance_claim_sha256": claim_sha256,
        "predecessor_complete_attempt_adopted": False,
        **_authority_hold_fields(),
    }
    payload = _bound_payload(core, digest_field="receipt_payload_sha256")
    runner._exclusive_json(path, payload)
    predecessor._fsync_file(path)
    raw_sha = predecessor._validate_bound_json(
        path,
        runner,
        expected=payload,
        digest_field="receipt_payload_sha256",
        label=f"continuation row Job witness {row.row_id}",
    )
    return payload, raw_sha


def execute_continuation(
    context: ContinuationContext,
    predecessor: ModuleType,
    runner: ModuleType,
    process_guard: ModuleType,
    *,
    interpreter_identity: Mapping[str, object],
    environment_receipt: Mapping[str, object],
    incident_receipt: Mapping[str, object],
    expected_helper_sha256: str,
) -> dict[str, object]:
    """Freshly rerun exact17 and append original-format completed markers.

    The five complete predecessor attempts are immutable incident evidence only.
    They are never adopted and never become completed markers.
    """
    if OUTER_ROW_TIMEOUT_SECONDS <= OPERATIONAL_METRIC_TIMEOUT_SECONDS:
        _fail("Continuation outer timeout has no margin over metric timeout")
    helper_sha256 = _validate_expected_helper_identity(
        context.project_root, expected_helper_sha256
    )
    expected_external_custody = incident_receipt.get(
        "external_scheduling_custody"
    )
    if type(expected_external_custody) is not dict:
        _fail("Continuation incident receipt lacks external-scheduling custody")

    def verify_external_scheduling_custody() -> None:
        current = _validate_external_scheduling_evidence(
            context,
            process_guard,
            runner,
            interpreter_identity=interpreter_identity,
            environment_receipt=environment_receipt,
        )
        if current != expected_external_custody:
            _fail("External-scheduling custody binding changed during continuation")

    _verify_static_boundary(
        context,
        predecessor,
        runner,
        process_guard,
        helper_sha256=helper_sha256,
        interpreter_identity=interpreter_identity,
        environment_receipt=environment_receipt,
    )
    verify_external_scheduling_custody()
    _assert_continuation_state(
        context, runner, owned_attempts={}, sealed_rows=set()
    )
    preclaim_scan = _assert_no_live_processes(context, process_guard)
    claim = _claim_payload(
        context,
        helper_sha256=helper_sha256,
        interpreter_identity=interpreter_identity,
        environment_receipt=environment_receipt,
        incident_receipt=incident_receipt,
        preclaim_process_scan=preclaim_scan,
    )
    claim_sha256 = hashlib.sha256(_exclusive_json_bytes(claim)).hexdigest()
    claim_path = context.output / CLAIM_NAME
    receipt_path = context.output / RECEIPT_NAME
    seal_path = context.output / RECEIPT_SEAL_NAME
    owned_attempts: dict[str, Path] = {}
    sealed_rows: set[str] = set()
    sealed_entries: dict[str, dict[str, object]] = {}
    running_specs: set[str] = set()
    cleanup_events: list[dict[str, object]] = []
    state_lock = threading.Lock()
    process_lock = threading.Lock()
    stop_event = threading.Event()
    phase = "CLAIM_EXCLUSIVE_CREATE"
    claim_acquired = False
    success_receipt: dict[str, object] | None = None
    success_receipt_sha256: str | None = None
    success_receipt_created = False
    success_seal: dict[str, object] | None = None
    success_seal_sha256: str | None = None
    success_seal_created = False

    def verify_operational_boundary() -> None:
        _verify_static_boundary(
            context,
            predecessor,
            runner,
            process_guard,
            helper_sha256=helper_sha256,
            interpreter_identity=interpreter_identity,
            environment_receipt=environment_receipt,
        )
        _validate_claim(
            context,
            predecessor,
            runner,
            expected_claim=claim,
            expected_sha256=claim_sha256,
        )
        with state_lock:
            allowed = set(running_specs)
        with process_lock:
            _assert_process_boundary(
                context, process_guard, allowed_worker_specs=allowed
            )

    def verify_row_evidence(
        row: ContinuationRow,
        attempt: Path,
        entry: Mapping[str, object],
    ) -> None:
        spec_path = attempt / "worker.spec.json"
        predecessor._verify_worker_spec(
            runner,
            row,
            spec_path,
            expected_sha256=str(entry["worker_spec_sha256"]),
        )
        predecessor._validate_recovery_result(
            context,
            runner,
            row,
            attempt,
            helper_sha256=helper_sha256,
        )
        job_path = attempt / JOB_WITNESS_NAME
        if (
            runner._sha256(attempt / "worker.result.json")
            != entry["worker_result_sha256"]
            or runner._sha256(attempt / predecessor.OVERRIDE_WITNESS_NAME)
            != entry["timeout_witness_sha256"]
            or runner._sha256(job_path) != entry["windows_job_receipt_sha256"]
        ):
            _fail(f"Continuation row evidence drifted: {row.row_id}")
        predecessor._validate_bound_json(
            job_path,
            runner,
            expected=entry["windows_job_receipt"],
            digest_field="receipt_payload_sha256",
            label=f"continuation row Job witness {row.row_id}",
        )
        marker_path = context.output / str(entry["completed_marker_path"])
        if runner._sha256(marker_path) != entry["completed_marker_sha256"]:
            _fail(f"Continuation completed marker drifted: {row.row_id}")
        completed = runner._completed_payload(context.output, row.row_id)
        if completed is None or completed.get("row_id") != row.row_id:
            _fail(f"Continuation completed verifier rejected row: {row.row_id}")

    def run_one(row: ContinuationRow) -> dict[str, object]:
        if stop_event.is_set():
            _fail("Continuation cancelled after another row failed")
        verify_operational_boundary()
        with state_lock:
            _assert_continuation_state(
                context,
                runner,
                owned_attempts=owned_attempts,
                sealed_rows=sealed_rows,
            )
            attempt = runner._next_attempt_directory(
                context.output, row.row_id
            ).resolve()
            expected_attempt = (
                context.output
                / "attempts"
                / row.row_id
                / f"attempt-{row.expected_attempt_number:04d}"
            ).resolve()
            if attempt != expected_attempt:
                _fail(f"Continuation attempt number drifted: {row.row_id}")
            if attempt.is_symlink() or not attempt.is_dir():
                _fail(f"Continuation attempt is unsafe: {row.row_id}")
            owned_attempts[row.row_id] = attempt
            spec_path = attempt / "worker.spec.json"
            runner._exclusive_json(spec_path, row.worker_spec)
            predecessor._fsync_file(spec_path)
            spec_sha256, spec_payload_sha256 = predecessor._verify_worker_spec(
                runner, row, spec_path
            )
            running_specs.add(spec_path.resolve().as_posix())
        verify_operational_boundary()
        child_error: BaseException | None = None
        job_witness: dict[str, object] | None = None
        try:
            job_witness = _run_continuation_child(
                context,
                predecessor,
                process_guard,
                spec_path,
                helper_sha256=helper_sha256,
                claim_sha256=claim_sha256,
            )
        except BaseException as error:
            child_error = error
        try:
            predecessor._verify_worker_spec(
                runner, row, spec_path, expected_sha256=spec_sha256
            )
            verify_operational_boundary()
        finally:
            with state_lock:
                running_specs.discard(spec_path.resolve().as_posix())
        if child_error is not None:
            raise child_error
        if type(job_witness) is not dict:
            _fail("Continuation child returned no corrected Job witness")
        result, timeout_witness = predecessor._validate_recovery_result(
            context,
            runner,
            row,
            attempt,
            helper_sha256=helper_sha256,
        )
        if stop_event.is_set():
            _fail("Continuation cancelled before Job receipt and marker sealing")
        verify_operational_boundary()
        predecessor._verify_worker_spec(
            runner, row, spec_path, expected_sha256=spec_sha256
        )
        with state_lock:
            _assert_continuation_state(
                context,
                runner,
                owned_attempts=owned_attempts,
                sealed_rows=sealed_rows,
            )
            job_receipt, job_receipt_sha256 = _write_row_job_witness(
                context,
                predecessor,
                runner,
                row,
                attempt,
                helper_sha256=helper_sha256,
                claim_sha256=claim_sha256,
                job_witness=job_witness,
            )
            completed = {
                **result,
                "row_id": row.row_id,
                "attempt_directory": attempt.relative_to(
                    context.output
                ).as_posix(),
                "plan_sha256": EXPECTED_PLAN_SHA256,
            }
            marker_path = context.output / "completed" / f"{row.row_id}.json"
            marker_sha256 = hashlib.sha256(
                _exclusive_json_bytes(completed)
            ).hexdigest()
            runner._exclusive_json(marker_path, completed)
            predecessor._fsync_file(marker_path)
            try:
                if runner._completed_payload(context.output, row.row_id) != completed:
                    _fail(
                        "Frozen completed verifier rejected continuation row: "
                        + row.row_id
                    )
            except BaseException:
                cleanup_status = predecessor._cleanup_exact_owned_file(
                    marker_path,
                    expected_sha256=marker_sha256,
                    expected_payload=completed,
                    runner=runner,
                )
                cleanup_events.append(
                    {
                        "row_id": row.row_id,
                        "path": marker_path.relative_to(
                            context.output
                        ).as_posix(),
                        "cleanup_status": cleanup_status,
                    }
                )
                raise
            sealed_rows.add(row.row_id)
            job_receipt_path = (attempt / JOB_WITNESS_NAME).relative_to(
                context.output
            ).as_posix()
            entry: dict[str, object] = {
                "ordinal": row.ordinal,
                "row_id": row.row_id,
                "attempt_directory": attempt.relative_to(
                    context.output
                ).as_posix(),
                "worker_spec_sha256": spec_sha256,
                "worker_spec_payload_sha256": spec_payload_sha256,
                "worker_result_sha256": runner._sha256(
                    attempt / "worker.result.json"
                ),
                "timeout_witness_sha256": runner._sha256(
                    attempt / predecessor.OVERRIDE_WITNESS_NAME
                ),
                "timeout_witness_payload_sha256": timeout_witness[
                    "receipt_payload_sha256"
                ],
                "windows_job_receipt_path": job_receipt_path,
                "windows_job_receipt_sha256": job_receipt_sha256,
                "windows_job_receipt_payload_sha256": job_receipt[
                    "receipt_payload_sha256"
                ],
                "windows_job_receipt": job_receipt,
                "completed_marker_path": marker_path.relative_to(
                    context.output
                ).as_posix(),
                "completed_marker_sha256": marker_sha256,
            }
            if not job_receipt_path.endswith("/" + JOB_WITNESS_NAME):
                _fail("Continuation row Job receipt path drifted")
            sealed_entries[row.row_id] = entry
            _assert_continuation_state(
                context,
                runner,
                owned_attempts=owned_attempts,
                sealed_rows=sealed_rows,
            )
            return entry

    try:
        try:
            runner._exclusive_json(claim_path, claim)
        except FileExistsError as error:
            raise RuntimeError(
                "Continuation claim race loser performed zero writes"
            ) from error
        predecessor._fsync_file(claim_path)
        _validate_claim(
            context,
            predecessor,
            runner,
            expected_claim=claim,
            expected_sha256=claim_sha256,
        )
        claim_acquired = True
        phase = "POST_CLAIM_PROCESS_AND_BOUNDARY"
        verify_operational_boundary()
        phase = "PARALLEL_FRESH_FULL_ALGORITHM_RERUN_17"
        first_error: BaseException | None = None
        futures: list[Future[dict[str, object]]] = []
        with ThreadPoolExecutor(
            max_workers=JOBS,
            thread_name_prefix="v21e3r1-metric-timeout-continuation",
        ) as executor:
            futures = [executor.submit(run_one, row) for row in context.fresh_rows]
            for future in as_completed(futures):
                try:
                    future.result()
                except BaseException as error:
                    stop_event.set()
                    if first_error is None:
                        first_error = error
                    for pending in futures:
                        pending.cancel()
        if first_error is not None:
            raise first_error

        phase = "FINAL_FRESH17_AND_EXACT504_VERIFICATION"
        ordered_entries = [sealed_entries[row.row_id] for row in context.fresh_rows]
        if len(ordered_entries) != len(RECOVERY_ORDINALS):
            _fail("Continuation did not seal exact fresh17 rows")
        verify_operational_boundary()
        _verify_incident_immutable(context)
        with state_lock:
            _assert_continuation_state(
                context,
                runner,
                owned_attempts=owned_attempts,
                sealed_rows=sealed_rows,
            )
        completed_paths = list((context.output / "completed").iterdir())
        if (
            any(path.is_symlink() or not path.is_file() for path in completed_paths)
            or {path.name for path in completed_paths}
            != {f"{row_id}.json" for row_id in context.all_row_ids}
        ):
            _fail("Continuation final completed-marker set is not exact504")
        if (context.output / "diagnostic.aggregate.json").exists() or (
            context.output / "diagnostic.receipt.json"
        ).exists():
            _fail("Continuation must not materialize aggregate/diagnostic receipt")
        for row, entry in zip(context.fresh_rows, ordered_entries, strict=True):
            verify_row_evidence(
                row, context.output / str(entry["attempt_directory"]), entry
            )
        verify_external_scheduling_custody()
        _assert_success_side_evidence_absent(context)

        phase = "SUCCESS_RECEIPT_MATERIALIZATION"
        core: dict[str, object] = {
            "schema": "v21e3r1_metric_timeout_recovery_continuation_receipt_v1",
            "status": "PASS_CHAINED_FRESH_EXACT17_OPERATIONAL_TIMEOUT_CONTINUATION_ONLY",
            "scope": "EXACT_ORDINALS_446_462_FROZEN_DEVELOPMENT_CONTINUATION_ONLY",
            "continuation_semantics": CONTINUATION_SEMANTICS,
            "original_diagnostic_receipt_alone_insufficient": True,
            "target_case_id": TARGET_CASE_ID,
            "target_ordinals": list(RECOVERY_ORDINALS),
            "target_row_count": len(context.rows),
            "jobs": JOBS,
            "original_metric_timeout_seconds": ORIGINAL_METRIC_TIMEOUT_SECONDS,
            "operational_metric_timeout_seconds": OPERATIONAL_METRIC_TIMEOUT_SECONDS,
            "outer_row_timeout_seconds": OUTER_ROW_TIMEOUT_SECONDS,
            "accounting_grace_seconds": ACCOUNTING_GRACE_SECONDS,
            "old_complete_attempt_count": len(context.incident_complete_rows),
            "old_complete_attempt_adopted_count": 0,
            "incident_complete_attempts_not_adopted": True,
            "fresh_full_algorithm_rerun_count": len(context.fresh_rows),
            "predecessor_incident_immutable": True,
            "predecessor_helper_sha256": EXPECTED_PREDECESSOR_HELPER_SHA256,
            "predecessor_claim_sha256": EXPECTED_OLD_CLAIM_SHA256,
            "predecessor_failure_receipt_sha256": EXPECTED_OLD_FAILURE_SHA256,
            "predecessor_failure_seal_sha256": EXPECTED_OLD_FAILURE_SEAL_SHA256,
            "incident_receipt": dict(incident_receipt),
            "incident_file_manifest": list(context.incident_file_manifest),
            "incident_complete_attempt_manifest": list(
                context.incident_complete_attempt_manifest
            ),
            "predecessor_failed_attempt_manifest": list(
                context.predecessor_failed_attempt_manifest
            ),
            "external_scheduling_custody": dict(expected_external_custody),
            "external_scheduling_manifest": list(
                context.external_scheduling_manifest
            ),
            "plan_sha256": EXPECTED_PLAN_SHA256,
            "source_snapshot_sha256": EXPECTED_SOURCE_ROOT_SHA256,
            "frozen_runner_sha256": EXPECTED_RUNNER_SHA256,
            "independent_metric_source_sha256": EXPECTED_METRIC_SHA256,
            "process_guard_sha256": EXPECTED_PROCESS_GUARD_SHA256,
            "helper_sha256": helper_sha256,
            "helper_instance_claim_path": CLAIM_NAME,
            "helper_instance_claim_sha256": claim_sha256,
            "interpreter_identity": dict(interpreter_identity),
            "environment_receipt": dict(environment_receipt),
            "completed_rows": ordered_entries,
            "final_completed_marker_count": EXPECTED_FULL_ROWS,
            "aggregate_materialized": False,
            "diagnostic_receipt_materialized": False,
            "original_runner_resume_required": True,
            "original_runner_resume_after_continuation_success_only": True,
            "implementation_independence": False,
            "algorithm_execution_independence": False,
            "scientific_independence": False,
            **_authority_hold_fields(),
        }
        receipt = _bound_payload(core, digest_field="receipt_payload_sha256")
        success_receipt = receipt
        success_receipt_sha256 = hashlib.sha256(
            _exclusive_json_bytes(receipt)
        ).hexdigest()
        runner._exclusive_json(receipt_path, receipt)
        success_receipt_created = True
        predecessor._fsync_file(receipt_path)
        actual_receipt_sha256 = predecessor._validate_bound_json(
            receipt_path,
            runner,
            expected=receipt,
            digest_field="receipt_payload_sha256",
            label="continuation success receipt",
        )
        if actual_receipt_sha256 != success_receipt_sha256:
            _fail("Continuation success receipt raw bytes drifted")
        seal_core = {
            "schema": "v21e3r1_metric_timeout_recovery_continuation_success_seal_v1",
            "status": "SEALED_CHAINED_CONTINUATION_SUCCESS_RECEIPT_FILE_DIGEST",
            "receipt_path": RECEIPT_NAME,
            "receipt_sha256": actual_receipt_sha256,
            "receipt_payload_sha256": receipt["receipt_payload_sha256"],
            "helper_instance_claim_sha256": claim_sha256,
            "predecessor_failure_receipt_sha256": EXPECTED_OLD_FAILURE_SHA256,
            "external_scheduling_receipt_sha256": (
                EXPECTED_EXTERNAL_RECEIPT_SHA256
            ),
            "external_scheduling_success_seal_sha256": (
                EXPECTED_EXTERNAL_SEAL_SHA256
            ),
            "external_scheduling_custody_payload_sha256": (
                expected_external_custody["custody_payload_sha256"]
            ),
            **_authority_hold_fields(),
        }
        success_seal = _bound_payload(seal_core, digest_field="seal_payload_sha256")
        success_seal_sha256 = hashlib.sha256(
            _exclusive_json_bytes(success_seal)
        ).hexdigest()
        runner._exclusive_json(seal_path, success_seal)
        success_seal_created = True
        predecessor._fsync_file(seal_path)
        actual_seal_sha256 = predecessor._validate_bound_json(
            seal_path,
            runner,
            expected=success_seal,
            digest_field="seal_payload_sha256",
            label="continuation success seal",
        )
        if (
            actual_seal_sha256 != success_seal_sha256
            or runner._sha256(receipt_path) != actual_receipt_sha256
        ):
            _fail("Continuation success receipt/seal digest drifted")
        _assert_success_side_evidence_absent(context)

        phase = "TERMINAL_POSTWRITE_REHASH"
        verify_operational_boundary()
        _verify_incident_immutable(context)
        with state_lock:
            _assert_continuation_state(
                context,
                runner,
                owned_attempts=owned_attempts,
                sealed_rows=sealed_rows,
            )
        for row, entry in zip(context.fresh_rows, ordered_entries, strict=True):
            verify_row_evidence(
                row, context.output / str(entry["attempt_directory"]), entry
            )
        verify_external_scheduling_custody()
        if (
            runner._sha256(receipt_path) != actual_receipt_sha256
            or runner._sha256(seal_path) != actual_seal_sha256
            or (context.output / "diagnostic.aggregate.json").exists()
            or (context.output / "diagnostic.receipt.json").exists()
        ):
            _fail("Continuation terminal evidence boundary drifted")
        _assert_success_side_evidence_absent(context)
        return receipt
    except BaseException as error:
        for path, payload, expected_sha256, created in (
            (seal_path, success_seal, success_seal_sha256, success_seal_created),
            (
                receipt_path,
                success_receipt,
                success_receipt_sha256,
                success_receipt_created,
            ),
        ):
            if created and path.is_file():
                cleanup_status = (
                    "RETAINED_NO_CONTINUATION_OWNERSHIP_WITNESS"
                    if payload is None or expected_sha256 is None
                    else predecessor._cleanup_exact_owned_file(
                        path,
                        expected_sha256=expected_sha256,
                        expected_payload=payload,
                        runner=runner,
                    )
                )
                cleanup_events.append(
                    {
                        "row_id": "__success_file__",
                        "path": path.name,
                        "cleanup_status": cleanup_status,
                    }
                )
        if not claim_acquired:
            raise RuntimeError(
                "Continuation stopped before exact claim acquisition; no owned "
                "failure/quarantine receipt was materialized: " + str(error)
            ) from error
        if not _claim_is_owned(
            context,
            predecessor,
            runner,
            expected_claim=claim,
            expected_sha256=claim_sha256,
        ):
            raise RuntimeError(
                "Continuation claim changed after acquisition; onsite evidence "
                "was retained and no owned failure/quarantine was materialized"
            ) from error
        terminal_error: BaseException | None = (
            error if isinstance(error, DescendantTerminationUnconfirmed) else None
        )
        terminal_scan: dict[str, object] | None = None
        if terminal_error is None:
            try:
                terminal_scan = _wait_for_worker_specs_zero(
                    context,
                    process_guard,
                    {
                        (attempt / "worker.spec.json").resolve().as_posix()
                        for attempt in owned_attempts.values()
                    },
                    block_all=True,
                )
            except BaseException as descendant_error:
                terminal_error = DescendantTerminationUnconfirmed(
                    "Continuation cleanup could not prove every wrapper/metric "
                    "descendant reached zero: " + str(descendant_error)
                )
        if terminal_error is not None:
            _write_quarantine(
                context,
                predecessor,
                runner,
                error=terminal_error,
                phase=phase,
                helper_sha256=helper_sha256,
                claim=claim,
                claim_sha256=claim_sha256,
                owned_attempts=owned_attempts,
            )
            raise RuntimeError(
                "Continuation is NOT_TERMINAL; quarantine retained and every "
                "resume path is prohibited"
            ) from terminal_error
        _write_failure(
            context,
            predecessor,
            runner,
            error=error,
            phase=phase,
            helper_sha256=helper_sha256,
            claim=claim,
            claim_sha256=claim_sha256,
            owned_attempts=owned_attempts,
            sealed_entries=sealed_entries,
            terminal_process_scan=terminal_scan,
        )
        raise RuntimeError(
            "Continuation stopped fail-closed; append-only attempts/markers and "
            "durable failure evidence retained; automatic resume prohibited"
        ) from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-directory")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--expected-helper-sha256")
    parser.add_argument("--continuation-worker-spec", help=argparse.SUPPRESS)
    parser.add_argument("--helper-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--claim-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--job-start-gate", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.continuation_worker_spec is not None:
            if (
                args.output_directory is not None
                or args.preflight_only
                or args.expected_helper_sha256 is not None
                or args.helper_sha256 is None
                or args.claim_sha256 is None
                or args.job_start_gate != JOB_START_GATE_ARGUMENT
            ):
                _fail("Malformed internal continuation-worker invocation")
            if sys.stdin.readline() != JOB_START_GATE_LINE:
                _fail("Continuation worker Job Object start gate drifted")
            result = run_continuation_worker(
                args.continuation_worker_spec,
                helper_sha256=args.helper_sha256,
                claim_sha256=args.claim_sha256,
            )
            print(json.dumps(result, sort_keys=True), flush=True)
            return 0
        if (
            args.output_directory is None
            or args.expected_helper_sha256 is None
            or args.helper_sha256 is not None
            or args.claim_sha256 is not None
            or args.job_start_gate is not None
        ):
            _fail("--output-directory is required for continuation parent mode")
        (
            context,
            predecessor,
            runner,
            process_guard,
            interpreter_identity,
            environment_receipt,
            incident_receipt,
        ) = preflight(
            args.project_root,
            args.output_directory,
            expected_helper_sha256=args.expected_helper_sha256,
        )
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "status": (
                            "PASS_PREFLIGHT_CHAINED_CONTINUATION_FRESH_"
                            "EXACT17_READY_NO_WRITES"
                        ),
                        "target_ordinals": list(RECOVERY_ORDINALS),
                        "old_complete_attempt_count": 5,
                        "old_complete_attempt_adopted_count": 0,
                        "fresh_full_algorithm_rerun_count": 17,
                        "plan_sha256": EXPECTED_PLAN_SHA256,
                        "source_snapshot_sha256": EXPECTED_SOURCE_ROOT_SHA256,
                        "continuation_helper_sha256": args.expected_helper_sha256,
                        "predecessor_failure_receipt_sha256": (
                            EXPECTED_OLD_FAILURE_SHA256
                        ),
                        "incident_payload_sha256": incident_receipt[
                            "incident_payload_sha256"
                        ],
                        "writes_performed": False,
                        **_authority_hold_fields(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0
        result = execute_continuation(
            context,
            predecessor,
            runner,
            process_guard,
            interpreter_identity=interpreter_identity,
            environment_receipt=environment_receipt,
            incident_receipt=incident_receipt,
            expected_helper_sha256=args.expected_helper_sha256,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        print(f"FAIL_CLOSED: {type(error).__name__}: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
