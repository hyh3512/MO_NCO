from __future__ import annotations

import importlib.util
import ctypes
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = PROJECT_ROOT / (
    "artifacts/v21e3r1_v8_work_20260822/"
    "run_frozen_diagnostic_metric_timeout_recovery_continuation.py"
)
OUTPUT = PROJECT_ROOT / (
    "outputs/v21e3r1_v7_exposed_development_diagnostics_20260823"
)


def _load_helper() -> ModuleType:
    name = "_test_v21e3r1_metric_timeout_recovery_continuation"
    spec = importlib.util.spec_from_file_location(name, HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _pid_alive(pid: int) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    handle = kernel32.OpenProcess(0x1000, 0, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_uint32()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return code.value == 259
    finally:
        kernel32.CloseHandle(handle)


def test_continuation_static_contract_fresh_reruns_exact17() -> None:
    helper = _load_helper()
    assert helper.EXPECTED_PREDECESSOR_HELPER_SHA256 == (
        "c4d0c67fc22fbc45a6b73b9d94cc9ab73bd1767fd78d9cfb6d9d670b194aa122"
    )
    assert helper.RECOVERY_ORDINALS == tuple(range(446, 463))
    assert helper.INCIDENT_COMPLETE_ORDINALS == tuple(range(446, 451))
    assert helper.FRESH_RERUN_ORDINALS == tuple(range(446, 463))
    assert helper.JOBS == 4
    assert helper.ORIGINAL_METRIC_TIMEOUT_SECONDS == 300
    assert helper.OPERATIONAL_METRIC_TIMEOUT_SECONDS == 1200
    assert helper.OUTER_ROW_TIMEOUT_SECONDS == 2400
    assert "5_COMPLETE_NOT_ADOPTED_FRESH_RERUN_17" in (
        helper.CONTINUATION_SEMANTICS
    )


def test_real_incident_preflight_is_read_only_and_requires_fresh_exact17() -> None:
    helper = _load_helper()
    before = {
        path.name: helper._sha256(path)
        for path in OUTPUT.iterdir()
        if path.is_file()
        and (
            path.name.startswith("metric-timeout-recovery.")
            or path.name.startswith("metric-timeout-recovery-continuation.")
        )
    }
    (
        context,
        predecessor,
        runner,
        process_guard,
        interpreter,
        environment,
        incident,
    ) = helper.preflight(
        PROJECT_ROOT,
        OUTPUT,
        expected_helper_sha256=helper._sha256(HELPER_PATH),
    )
    del predecessor, runner, process_guard, interpreter, environment

    assert [row.ordinal for row in context.incident_complete_rows] == list(
        range(446, 451)
    )
    assert [row.ordinal for row in context.fresh_rows] == list(range(446, 463))
    assert [row.expected_attempt_number for row in context.fresh_rows] == (
        [3] + [2] * 4 + [1] * 12
    )
    assert incident["incident_complete_attempt_count"] == 5
    assert incident["incident_complete_attempt_adopted_count"] == 0
    assert incident["incident_complete_attempts_not_adopted"] is True
    assert incident["fresh_full_algorithm_rerun_count"] == 17
    assert incident["preserved_marker_count"] == 487
    assert incident["old_success_absent"] is True
    custody = incident["external_scheduling_custody"]
    assert custody["status"] == (
        "PASS_HASH_BOUND_EXTERNAL_SCHEDULING_ONLY_NO_NEW_AUTHORITY"
    )
    assert custody["target_ordinals"] == list(range(463, 505))
    assert custody["target_row_count"] == 42
    assert custody["claim_sha256"] == helper.EXPECTED_EXTERNAL_CLAIM_SHA256
    assert custody["receipt_sha256"] == helper.EXPECTED_EXTERNAL_RECEIPT_SHA256
    assert custody["seal_sha256"] == helper.EXPECTED_EXTERNAL_SEAL_SHA256
    assert custody["runtime_authority"] is False
    assert incident["external_scheduling_manifest_sha256"] == (
        helper._manifest_sha256(context.external_scheduling_manifest)
    )
    assert incident["publication_status"] == "IJOC_HOLD"
    assert {
        path.name: helper._sha256(path)
        for path in OUTPUT.iterdir()
        if path.is_file()
        and (
            path.name.startswith("metric-timeout-recovery.")
            or path.name.startswith("metric-timeout-recovery-continuation.")
        )
    } == before


def test_wrong_externally_supplied_helper_sha_fails_before_any_write() -> None:
    helper = _load_helper()
    before = {
        path.name: helper._sha256(path)
        for path in OUTPUT.iterdir()
        if path.is_file()
    }
    with pytest.raises(RuntimeError, match="externally supplied SHA"):
        helper.preflight(
            PROJECT_ROOT,
            OUTPUT,
            expected_helper_sha256="0" * 64,
        )
    assert {
        path.name: helper._sha256(path)
        for path in OUTPUT.iterdir()
        if path.is_file()
    } == before


def _real_external_validation_inputs(helper: ModuleType):
    predecessor = helper._load_predecessor(PROJECT_ROOT)
    predecessor._validate_production_paths(PROJECT_ROOT, OUTPUT)
    process_guard = predecessor._load_process_guard(PROJECT_ROOT)
    interpreter = predecessor._validate_interpreter(process_guard)
    environment = process_guard._execution_environment_receipt(
        PROJECT_ROOT, interpreter
    )
    runner = predecessor._load_frozen_runner(PROJECT_ROOT)
    context = helper._validate_incident(
        PROJECT_ROOT, OUTPUT, predecessor, runner
    )
    return context, process_guard, runner, interpreter, environment


def test_external_scheduling_missing_bound_file_is_fail_closed() -> None:
    helper = _load_helper()
    context, process_guard, runner, interpreter, environment = (
        _real_external_validation_inputs(helper)
    )
    broken = replace(
        context,
        external_scheduling_manifest=(
            {
                "path": "external-scheduling.missing.json",
                "bytes": 1,
                "sha256": "0" * 64,
            },
        ),
    )
    with pytest.raises(RuntimeError, match="missing manifest file"):
        helper._validate_external_scheduling_evidence(
            broken,
            process_guard,
            runner,
            interpreter_identity=interpreter,
            environment_receipt=environment,
        )


def test_external_scheduling_raw_receipt_tamper_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    context, process_guard, runner, interpreter, environment = (
        _real_external_validation_inputs(helper)
    )
    original_sha256 = runner._sha256

    def drift_receipt_sha(path: Path) -> str:
        if Path(path).name == helper.EXTERNAL_RECEIPT_NAME:
            return "0" * 64
        return original_sha256(path)

    monkeypatch.setattr(runner, "_sha256", drift_receipt_sha)
    with pytest.raises(RuntimeError, match="receipt semantic/cross-hash drifted"):
        helper._validate_external_scheduling_evidence(
            context,
            process_guard,
            runner,
            interpreter_identity=interpreter,
            environment_receipt=environment,
        )


@pytest.mark.parametrize(
    ("target_label", "field", "replacement", "message"),
    [
        (
            "external-scheduling s01 success receipt",
            "schema",
            "drifted_schema",
            "receipt semantic/cross-hash drifted",
        ),
        (
            "external-scheduling s01 success receipt",
            "runtime_authority",
            True,
            "receipt semantic/cross-hash drifted",
        ),
        (
            "external-scheduling s01 claim",
            "handoff_receipt_sha256",
            "0" * 64,
            "claim semantic/cross-hash drifted",
        ),
        (
            "external-scheduling s01 success seal",
            "receipt_sha256",
            "0" * 64,
            "seal semantic/cross-hash drifted",
        ),
    ],
)
def test_external_scheduling_schema_authority_and_cross_hash_drift_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    target_label: str,
    field: str,
    replacement: object,
    message: str,
) -> None:
    helper = _load_helper()
    context, process_guard, runner, interpreter, environment = (
        _real_external_validation_inputs(helper)
    )
    original = process_guard._validate_bound_payload_file

    def mutate(path, child_runner, *, expected_keys, payload_field, label):
        payload, raw_sha = original(
            path,
            child_runner,
            expected_keys=expected_keys,
            payload_field=payload_field,
            label=label,
        )
        if label == target_label:
            payload = dict(payload)
            payload[field] = replacement
        return payload, raw_sha

    monkeypatch.setattr(
        process_guard, "_validate_bound_payload_file", mutate
    )
    with pytest.raises(RuntimeError, match=message):
        helper._validate_external_scheduling_evidence(
            context,
            process_guard,
            runner,
            interpreter_identity=interpreter,
            environment_receipt=environment,
        )


def _write_lag_wrapper(tmp_path: Path, *, child_sleep: float) -> tuple[Path, Path, Path]:
    wrapper = tmp_path / "accounting_lag_wrapper.py"
    pid_path = tmp_path / "descendant.pid"
    terminal_path = tmp_path / "descendant.terminal"
    wrapper.write_text(
        "import subprocess,sys\n"
        "if sys.stdin.readline() != sys.argv[4] + '\\n': raise SystemExit(91)\n"
        "code=\"import os,pathlib,sys,time; "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
        "time.sleep(float(sys.argv[3])); "
        "pathlib.Path(sys.argv[2]).write_text('terminal')\"\n"
        "subprocess.Popen([sys.executable,'-c',code,sys.argv[1],sys.argv[2],sys.argv[3]])\n",
        encoding="utf-8",
    )
    return wrapper, pid_path, terminal_path


def test_normal_job_accounting_lag_drains_to_zero_without_false_failure(
    tmp_path: Path,
) -> None:
    helper = _load_helper()
    predecessor = helper._load_predecessor(PROJECT_ROOT)
    wrapper, pid_path, terminal_path = _write_lag_wrapper(
        tmp_path, child_sleep=1.0
    )

    def terminal_zero() -> dict[str, object]:
        pid = int(pid_path.read_text(encoding="utf-8"))
        if _pid_alive(pid):
            raise RuntimeError("descendant remained alive")
        return {"schema": "synthetic_terminal_zero_v1", "pid": pid}

    completed, witness = helper._run_in_windows_job_with_accounting_grace(
        [
            sys.executable,
            str(wrapper),
            str(pid_path),
            str(terminal_path),
            "1.0",
            helper.JOB_START_GATE_ARGUMENT,
        ],
        cwd=tmp_path,
        environment=os.environ,
        timeout_seconds=5,
        accounting_grace_seconds=3,
        start_gate_line=helper.JOB_START_GATE_LINE,
        terminal_zero_check=terminal_zero,
        job_factory=predecessor._WindowsKillOnCloseJob,
    )

    assert completed.returncode == 0
    assert terminal_path.read_text(encoding="utf-8") == "terminal"
    assert witness["initial_active_processes_after_wrapper_exit"] > 0
    assert witness["accounting_lag_observed"] is True
    assert witness["accounting_lag_drained_without_termination"] is True
    assert witness["accounting_grace_expired"] is False
    assert witness["terminal_active_processes"] == 0


def test_true_persistent_descendant_is_killed_and_fails_after_grace(
    tmp_path: Path,
) -> None:
    helper = _load_helper()
    predecessor = helper._load_predecessor(PROJECT_ROOT)
    wrapper, pid_path, terminal_path = _write_lag_wrapper(
        tmp_path, child_sleep=60.0
    )

    def terminal_zero() -> dict[str, object]:
        pid = int(pid_path.read_text(encoding="utf-8"))
        if _pid_alive(pid):
            raise RuntimeError("descendant survived Job termination")
        return {"schema": "synthetic_terminal_zero_v1", "pid": pid}

    with pytest.raises(RuntimeError, match="bounded grace|Job tree reached zero"):
        helper._run_in_windows_job_with_accounting_grace(
            [
                sys.executable,
                str(wrapper),
                str(pid_path),
                str(terminal_path),
                "60.0",
                helper.JOB_START_GATE_ARGUMENT,
            ],
            cwd=tmp_path,
            environment=os.environ,
            timeout_seconds=5,
            accounting_grace_seconds=0.3,
            start_gate_line=helper.JOB_START_GATE_LINE,
            terminal_zero_check=terminal_zero,
            job_factory=predecessor._WindowsKillOnCloseJob,
        )

    descendant_pid = int(pid_path.read_text(encoding="utf-8"))
    assert not _pid_alive(descendant_pid)
    time.sleep(1.2)
    assert not terminal_path.exists()


def _empty_scan(helper: ModuleType) -> dict[str, object]:
    core = {
        "schema": "synthetic_continuation_process_scan_v1",
        "matching_process_count": 0,
        "matching_processes": [],
    }
    return helper._bound_payload(core, digest_field="scan_payload_sha256")


def _synthetic_context(tmp_path: Path, helper: ModuleType):
    output = (tmp_path / "matrix").resolve()
    (output / "attempts").mkdir(parents=True)
    (output / "completed").mkdir()
    rows = []
    for index, ordinal in enumerate(helper.RECOVERY_ORDINALS):
        seed = 31057 if index < 7 else 31059 if index < 14 else 31063
        arm = f"TEST_ARM_{ordinal}"
        row_id = f"{helper.TARGET_CASE_ID}__seed-{seed}__arm-{arm.lower()}"
        rows.append(
            helper.ContinuationRow(
                ordinal=ordinal,
                row_id=row_id,
                expected_attempt_number=(
                    3 if ordinal == 446 else 2 if ordinal <= 450 else 1
                ),
                worker_spec={
                    "schema": "v21e3r1_diagnostic_row_worker_spec_v1",
                    "project_root": str(PROJECT_ROOT),
                    "case_id": helper.TARGET_CASE_ID,
                    "family": "MOTSP",
                    "size": 500,
                    "case_path": str(tmp_path / "case.json"),
                    "case_artifact_sha256": "a" * 64,
                    "objective_lower_bounds": [0.0, 1.0],
                    "objective_upper_bounds": [2.0, 3.0],
                    "reference_directions": [[0.5, 0.5]],
                    "seed": seed,
                    "arm_id": arm,
                    "charged_evaluation_budget": 2000,
                    "checkpoint_period": 200,
                    "source_snapshot_sha256": helper.EXPECTED_SOURCE_ROOT_SHA256,
                    "plan_sha256": helper.EXPECTED_PLAN_SHA256,
                },
            )
        )
    for row in rows[:5]:
        numbers = (1, 2) if row.ordinal == 446 else (1,)
        for number in numbers:
            attempt = (
                output
                / "attempts"
                / row.row_id
                / f"attempt-{number:04d}"
            )
            attempt.mkdir(parents=True)
            (attempt / "immutable.incident").write_bytes(
                f"old-{row.ordinal}-{number}".encode("ascii")
            )
    return helper.ContinuationContext(
        project_root=PROJECT_ROOT,
        output=output,
        plan_path=output / "diagnostic.plan.json",
        runner_path=PROJECT_ROOT / (
            "ijoc_submission_v21e3r1/scripts/"
            "run_v21e3r1_development_diagnostics.py"
        ),
        metric_path=PROJECT_ROOT / (
            "independent_reproduction/recompute_v21e3r1_metrics.py"
        ),
        process_guard_path=PROJECT_ROOT / (
            "ijoc_submission_v21e3r1/scripts/"
            "v21e3r1_frozen_diagnostic_process_guard.py"
        ),
        predecessor_path=PROJECT_ROOT / helper.PREDECESSOR_RELATIVE,
        rows=tuple(rows),
        incident_complete_rows=tuple(rows[:5]),
        fresh_rows=tuple(rows),
        all_row_ids=tuple(row.row_id for row in rows),
        preserved_marker_manifest=(),
        incident_file_manifest=(),
        incident_complete_attempt_manifest=(),
        predecessor_failed_attempt_manifest=(),
        external_scheduling_manifest=(),
    )


def _install_synthetic_execution(
    monkeypatch: pytest.MonkeyPatch,
    helper: ModuleType,
    predecessor: ModuleType,
    runner: ModuleType,
    *,
    delay_seconds: float = 0.01,
) -> dict[str, int]:
    monkeypatch.setattr(helper, "_verify_static_boundary", lambda *a, **k: None)
    monkeypatch.setattr(helper, "_verify_incident_immutable", lambda *a, **k: None)
    synthetic_custody = helper._bound_payload(
        {
            "schema": "synthetic_external_scheduling_custody_v1",
            "runtime_authority": False,
        },
        digest_field="custody_payload_sha256",
    )
    monkeypatch.setattr(
        helper,
        "_validate_external_scheduling_evidence",
        lambda *a, **k: dict(synthetic_custody),
    )
    monkeypatch.setattr(
        helper, "_assert_no_live_processes", lambda *a, **k: _empty_scan(helper)
    )
    monkeypatch.setattr(
        helper, "_assert_process_boundary", lambda *a, **k: _empty_scan(helper)
    )
    monkeypatch.setattr(
        helper,
        "_wait_for_worker_specs_zero",
        lambda *a, **k: _empty_scan(helper),
    )
    state = {"active": 0, "max_active": 0, "calls": 0}
    state_lock = threading.Lock()

    def fake_child(
        child_context,
        child_predecessor,
        process_guard,
        spec_path: Path,
        *,
        helper_sha256: str,
        claim_sha256: str,
    ) -> dict[str, object]:
        del child_context, child_predecessor, process_guard, helper_sha256
        del claim_sha256
        spec = runner._load_json_object(spec_path)
        attempt = spec_path.parent
        with state_lock:
            state["active"] += 1
            state["calls"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        try:
            time.sleep(delay_seconds)
            row = {
                "status": "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY",
                "case_id": spec["case_id"],
                "family": spec["family"],
                "size": spec["size"],
                "seed": spec["seed"],
                "arm_id": spec["arm_id"],
                "charged_evaluation_budget": 2000,
                "checkpoint_period": 200,
                "case_artifact_sha256": spec["case_artifact_sha256"],
                "source_snapshot_sha256": helper.EXPECTED_SOURCE_ROOT_SHA256,
                "plan_sha256": helper.EXPECTED_PLAN_SHA256,
                "selection_entropy_release": "PROHIBITED",
                "confirmation_materialization": "PROHIBITED",
                "formal_materialization": "PROHIBITED",
            }
            runner._exclusive_json(attempt / "row.json", row)
            runner._exclusive_json(attempt / "diagnostic.json", {"row": row})
            (attempt / "trace.sqlite3").write_bytes(b"synthetic trace")
            runner._exclusive_json(
                attempt / "terminal.receipt.json", {"status": "PASS"}
            )
            runner._exclusive_json(
                attempt / "independent.metric.json", {"status": "PASS"}
            )
            result = {
                "status": "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY",
                "row_sha256": runner._sha256(attempt / "row.json"),
                "diagnostic_sha256": runner._sha256(attempt / "diagnostic.json"),
                "trace_sha256": runner._sha256(attempt / "trace.sqlite3"),
                "terminal_receipt_sha256": runner._sha256(
                    attempt / "terminal.receipt.json"
                ),
                "independent_metric_receipt_sha256": runner._sha256(
                    attempt / "independent.metric.json"
                ),
            }
            runner._exclusive_json(attempt / "worker.result.json", result)
            runner._exclusive_json(
                attempt / predecessor.OVERRIDE_WITNESS_NAME,
                {"receipt_payload_sha256": "f" * 64},
            )
        finally:
            with state_lock:
                state["active"] -= 1
        return {
            "schema": "v21e3r1_continuation_windows_job_witness_v1",
            "kill_on_job_close_limit": True,
            "job_limit_flags": 0x00002000,
            "wrapper_pid": 12345,
            "job_assignment_verified_before_gate_release": True,
            "outer_timeout_seconds": helper.OUTER_ROW_TIMEOUT_SECONDS,
            "outer_timeout_fired": False,
            "wrapper_returncode": 0,
            "initial_active_processes_after_wrapper_exit": 1,
            "accounting_grace_seconds": helper.ACCOUNTING_GRACE_SECONDS,
            "accounting_lag_observed": True,
            "accounting_lag_drained_without_termination": True,
            "accounting_grace_expired": False,
            "accounting_wait_seconds": 0.01,
            "terminal_active_processes": 0,
            "terminal_process_scan": _empty_scan(helper),
        }

    def fake_validate(
        child_context,
        child_runner,
        row,
        attempt: Path,
        *,
        helper_sha256: str,
    ):
        del child_context, row, helper_sha256
        return (
            child_runner._load_json_object(attempt / "worker.result.json"),
            child_runner._load_json_object(
                attempt / predecessor.OVERRIDE_WITNESS_NAME
            ),
        )

    monkeypatch.setattr(helper, "_run_continuation_child", fake_child)
    monkeypatch.setattr(predecessor, "_validate_recovery_result", fake_validate)
    return state


def _execute_synthetic(
    helper: ModuleType,
    context,
    predecessor: ModuleType,
    runner: ModuleType,
):
    external_custody = helper._validate_external_scheduling_evidence(
        context,
        object(),
        runner,
        interpreter_identity={},
        environment_receipt={},
    )
    return helper.execute_continuation(
        context,
        predecessor,
        runner,
        object(),
        interpreter_identity={},
        environment_receipt={},
        incident_receipt={
            "schema": "synthetic_incident_v1",
            "external_scheduling_custody": external_custody,
        },
        expected_helper_sha256=helper._sha256(HELPER_PATH),
    )


def test_incident_complete_attempt_manifest_corruption_is_fail_closed(
    tmp_path: Path,
) -> None:
    helper = _load_helper()
    root = tmp_path.resolve()
    artifact = root / "old-complete.bin"
    artifact.write_bytes(b"sealed")
    manifest = helper._file_manifest(root, [artifact])
    artifact.write_bytes(b"corrupt")

    with pytest.raises(RuntimeError, match="manifest drifted"):
        helper._verify_manifest_unchanged(
            root, manifest, label="five complete-but-unadopted attempts"
        )


def test_continuation_attempt_layout_drift_is_fail_closed(tmp_path: Path) -> None:
    helper = _load_helper()
    predecessor = helper._load_predecessor(PROJECT_ROOT)
    runner = predecessor._load_frozen_runner(PROJECT_ROOT)
    context = _synthetic_context(tmp_path, helper)
    drift = (
        context.output
        / "attempts"
        / context.rows[5].row_id
        / "attempt-0001"
    )
    drift.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="attempt layout drifted"):
        helper._assert_continuation_state(
            context, runner, owned_attempts={}, sealed_rows=set()
        )


@pytest.mark.parametrize(
    "command",
    [
        (
            r'C:\miniconda3\python.exe '
            r'artifacts\v21e3r1_v8_work_20260822\run_frozen_'
            r'diagnostic_metric_timeout_recovery_continuation.py '
            r'--output-directory outputs\relative'
        ),
        (
            r'C:\miniconda3\python.exe -m '
            r'ijoc_submission_v21e3r1.scripts.'
            r'run_v21e3r1_development_diagnostics '
            r'--output-directory outputs\relative'
        ),
    ],
)
def test_relative_or_module_recovery_process_is_ambiguous_and_blocking(
    tmp_path: Path, command: str,
) -> None:
    helper = _load_helper()
    predecessor = helper._load_predecessor(PROJECT_ROOT)
    process_guard = predecessor._load_process_guard(PROJECT_ROOT)
    context = _synthetic_context(tmp_path, helper)

    assert helper._classify_recovery_command(
        command, context, process_guard
    ) == ("unknown", None)


def test_synthetic_fresh17_uses_jobs4_correct_attempts_and_job_witnesses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    predecessor = helper._load_predecessor(PROJECT_ROOT)
    runner = predecessor._load_frozen_runner(PROJECT_ROOT)
    context = _synthetic_context(tmp_path, helper)
    old_files = {
        path: helper._sha256(path)
        for row in context.rows[:5]
        for path in (
            context.output / "attempts" / row.row_id
        ).rglob("immutable.incident")
    }
    state = _install_synthetic_execution(
        monkeypatch, helper, predecessor, runner
    )

    receipt = _execute_synthetic(
        helper, context, predecessor, runner
    )

    assert state["calls"] == 17
    assert 2 <= state["max_active"] <= 4
    assert receipt["old_complete_attempt_count"] == 5
    assert receipt["old_complete_attempt_adopted_count"] == 0
    assert receipt["fresh_full_algorithm_rerun_count"] == 17
    assert receipt["schema"] == (
        "v21e3r1_metric_timeout_recovery_continuation_receipt_v1"
    )
    assert receipt["status"] == (
        "PASS_CHAINED_FRESH_EXACT17_OPERATIONAL_TIMEOUT_CONTINUATION_ONLY"
    )
    assert receipt["original_diagnostic_receipt_alone_insufficient"] is True
    assert receipt["runtime_authority"] is False
    assert receipt["scientific_authority"] is False
    assert len(receipt["completed_rows"]) == 17
    assert len(list((context.output / "completed").glob("*.json"))) == 17
    assert {
        path: helper._sha256(path) for path in old_files
    } == old_files
    assert (
        context.output
        / "attempts"
        / context.rows[0].row_id
        / "attempt-0003"
        / helper.JOB_WITNESS_NAME
    ).is_file()
    assert {
        path.name
        for path in (
            context.output / "attempts" / context.rows[0].row_id
        ).iterdir()
    } == {"attempt-0001", "attempt-0002", "attempt-0003"}
    for row in context.rows[1:5]:
        assert (
            context.output
            / "attempts"
            / row.row_id
            / "attempt-0002"
            / helper.JOB_WITNESS_NAME
        ).is_file()
        assert {
            path.name
            for path in (context.output / "attempts" / row.row_id).iterdir()
        } == {"attempt-0001", "attempt-0002"}
    for row in context.rows[5:]:
        assert (
            context.output
            / "attempts"
            / row.row_id
            / "attempt-0001"
            / helper.JOB_WITNESS_NAME
        ).is_file()
        assert {
            path.name
            for path in (context.output / "attempts" / row.row_id).iterdir()
        } == {"attempt-0001"}
    for row, entry in zip(context.rows, receipt["completed_rows"], strict=True):
        completed = runner._completed_payload(context.output, row.row_id)
        assert completed is not None
        assert completed["attempt_directory"] == entry["attempt_directory"]
        job_receipt = runner._load_json_object(
            context.output / entry["windows_job_receipt_path"]
        )
        assert job_receipt["schema"] == (
            "v21e3r1_continuation_row_windows_job_receipt_v1"
        )
        assert job_receipt["status"] == (
            "PASS_CORRECTED_WINDOWS_JOB_CONTAINMENT_AND_TERMINAL_ZERO"
        )
        assert job_receipt["job_control"]["kill_on_job_close_limit"] is True
        assert job_receipt["job_control"]["terminal_active_processes"] == 0
        assert job_receipt["predecessor_complete_attempt_adopted"] is False
    success_file = context.output / helper.RECEIPT_NAME
    seal_file = context.output / helper.RECEIPT_SEAL_NAME
    assert runner._load_json_object(success_file) == receipt
    seal = runner._load_json_object(seal_file)
    assert seal["schema"] == (
        "v21e3r1_metric_timeout_recovery_continuation_success_seal_v1"
    )
    assert seal["status"] == (
        "SEALED_CHAINED_CONTINUATION_SUCCESS_RECEIPT_FILE_DIGEST"
    )
    assert seal["receipt_sha256"] == runner._sha256(success_file)
    assert seal["receipt_payload_sha256"] == receipt["receipt_payload_sha256"]
    assert not (context.output / "diagnostic.aggregate.json").exists()
    assert not (context.output / "diagnostic.receipt.json").exists()
    assert not (context.output / helper.FAILURE_NAME).exists()
    assert not (context.output / helper.QUARANTINE_NAME).exists()


def test_two_continuations_claim_race_has_one_pass_and_zero_write_loser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    predecessor = helper._load_predecessor(PROJECT_ROOT)
    runner = predecessor._load_frozen_runner(PROJECT_ROOT)
    context = _synthetic_context(tmp_path, helper)
    _install_synthetic_execution(
        monkeypatch, helper, predecessor, runner, delay_seconds=0.02
    )
    original_exclusive = runner._exclusive_json
    barrier = threading.Barrier(2)

    def racing_exclusive(path: Path, payload: object) -> None:
        target = context.output / helper.CLAIM_NAME
        if Path(path) == target and not target.exists():
            barrier.wait(timeout=10)
        original_exclusive(path, payload)

    monkeypatch.setattr(runner, "_exclusive_json", racing_exclusive)
    results: list[dict[str, object]] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def invoke() -> None:
        try:
            value = _execute_synthetic(helper, context, predecessor, runner)
            with lock:
                results.append(value)
        except BaseException as error:
            with lock:
                errors.append(error)

    threads = [threading.Thread(target=invoke) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert all(not thread.is_alive() for thread in threads)
    assert [item["status"] for item in results] == [
        "PASS_CHAINED_FRESH_EXACT17_OPERATIONAL_TIMEOUT_CONTINUATION_ONLY"
    ]
    assert len(errors) == 1
    assert "zero writes" in str(errors[0])
    assert not (context.output / helper.FAILURE_NAME).exists()
    assert not (context.output / helper.FAILURE_SEAL_NAME).exists()
    assert not (context.output / helper.QUARANTINE_NAME).exists()


def test_postclaim_row_failure_seals_durable_hold_and_prohibits_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    predecessor = helper._load_predecessor(PROJECT_ROOT)
    runner = predecessor._load_frozen_runner(PROJECT_ROOT)
    context = _synthetic_context(tmp_path, helper)
    _install_synthetic_execution(monkeypatch, helper, predecessor, runner)

    def fail_child(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic postclaim row failure")

    monkeypatch.setattr(helper, "_run_continuation_child", fail_child)
    with pytest.raises(RuntimeError, match="durable failure evidence"):
        _execute_synthetic(helper, context, predecessor, runner)

    failure = runner._load_json_object(context.output / helper.FAILURE_NAME)
    assert failure["status"] == "HOLD_CONTINUATION_FAILURE_MANUAL_AUDIT_REQUIRED"
    assert failure["terminal_descendant_state_confirmed"] is True
    assert failure["main_runner_resume_authorized"] is False
    assert failure["partial_markers_main_runner_resume_prohibited"] is True
    assert failure["old_complete_attempt_adopted_count"] == 0
    assert (context.output / helper.FAILURE_SEAL_NAME).is_file()
    assert not (context.output / helper.RECEIPT_NAME).exists()
    assert not (context.output / helper.QUARANTINE_NAME).exists()


def test_unconfirmed_descendant_state_writes_quarantine_not_terminal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    predecessor = helper._load_predecessor(PROJECT_ROOT)
    runner = predecessor._load_frozen_runner(PROJECT_ROOT)
    context = _synthetic_context(tmp_path, helper)
    _install_synthetic_execution(monkeypatch, helper, predecessor, runner)

    def nonterminal_child(*args, **kwargs):
        del args, kwargs
        raise helper.DescendantTerminationUnconfirmed(
            "synthetic descendant could not be proven zero"
        )

    monkeypatch.setattr(helper, "_run_continuation_child", nonterminal_child)
    with pytest.raises(RuntimeError, match="NOT_TERMINAL"):
        _execute_synthetic(helper, context, predecessor, runner)

    quarantine = runner._load_json_object(context.output / helper.QUARANTINE_NAME)
    assert quarantine["status"] == (
        "NOT_TERMINAL_DESCENDANT_TERMINATION_UNCONFIRMED"
    )
    assert quarantine["terminal_descendant_state_confirmed"] is False
    assert quarantine["durable_terminal_failure"] is False
    assert quarantine["automatic_resume_authorized"] is False
    assert quarantine["main_runner_resume_authorized"] is False
    assert not (context.output / helper.FAILURE_NAME).exists()
    assert not (context.output / helper.FAILURE_SEAL_NAME).exists()
    assert not (context.output / helper.RECEIPT_NAME).exists()
