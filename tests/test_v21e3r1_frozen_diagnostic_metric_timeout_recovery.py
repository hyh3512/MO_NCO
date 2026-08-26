from __future__ import annotations

import ast
import ctypes
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
import sys
import threading
import time
from types import ModuleType
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = (
    PROJECT_ROOT
    / "artifacts/v21e3r1_v8_work_20260822/"
    "run_frozen_diagnostic_metric_timeout_recovery.py"
)


def _load_helper() -> ModuleType:
    name = "_test_v21e3r1_metric_timeout_recovery"
    spec = importlib.util.spec_from_file_location(name, HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_context(tmp_path: Path, helper: ModuleType):
    output = (tmp_path / "matrix").resolve()
    (output / "attempts").mkdir(parents=True)
    (output / "completed").mkdir()
    rows = []
    for index, ordinal in enumerate(helper.RECOVERY_ORDINALS):
        seed = 31057 if index < 3 else 31059
        arm = f"TEST_ARM_{ordinal}"
        row_id = (
            f"{helper.TARGET_CASE_ID}__seed-{seed}__arm-{arm.lower()}"
        )
        rows.append(
            helper.RecoveryRow(
                ordinal=ordinal,
                row_id=row_id,
                expected_attempt_number=2 if index == 0 else 1,
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
    first_attempt = output / "attempts" / rows[0].row_id / "attempt-0001"
    first_attempt.mkdir(parents=True)
    return helper.RecoveryContext(
        project_root=PROJECT_ROOT,
        output=output,
        plan_path=output / helper.PLAN_NAME,
        runner_path=PROJECT_ROOT / helper.RUNNER_RELATIVE,
        metric_path=PROJECT_ROOT / helper.METRIC_RELATIVE,
        process_guard_path=PROJECT_ROOT / helper.PROCESS_GUARD_RELATIVE,
        rows=tuple(rows),
        all_row_ids=tuple(row.row_id for row in rows),
        non_target_marker_manifest=(),
        preexisting_failed_attempt_manifest=(),
    )


def _empty_scan(helper: ModuleType) -> dict[str, object]:
    core = {
        "schema": "synthetic_process_scan_v1",
        "original_runner_scan_payload_sha256": "a" * 64,
        "allowed_worker_specs": [],
        "recovery_processes": [],
    }
    result = dict(core)
    result["scan_payload_sha256"] = hashlib.sha256(
        helper._canonical_bytes(core)
    ).hexdigest()
    return result


def _windows_pid_is_alive(pid: int) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(0x1000, 0, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_uint32()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return exit_code.value == 259
    finally:
        kernel32.CloseHandle(handle)


def test_helper_is_static_operational_timeout_recovery_only() -> None:
    helper = _load_helper()
    source = HELPER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "mo_nco" not in imported_roots
    assert helper.EXPECTED_PLAN_SHA256 == (
        "4408d10944cb6511e99ff0bd95ded256b9c230b91d8806a7bd5b962f10622886"
    )
    assert helper.EXPECTED_SOURCE_ROOT_SHA256 == (
        "218bc398f04722d1da305928a9c206641f9b43d74b2afbc46c29ba1f08d6639b"
    )
    assert helper.EXPECTED_RUNNER_SHA256 == (
        "70a45fd0e62d870702b29a92b66b38eef6c04952152d5defae89c115c6d85b7b"
    )
    assert helper.EXPECTED_METRIC_SHA256 == (
        "587d4ed4d647d8293b36449c835109ee3afa6e9899fe155f917a492fdf303ea2"
    )
    assert helper.RECOVERY_ORDINALS == tuple(range(446, 463))
    assert helper.JOBS == 4
    assert helper.ORIGINAL_METRIC_TIMEOUT_SECONDS == 300
    assert helper.OPERATIONAL_METRIC_TIMEOUT_SECONDS == 1200
    assert helper.FROZEN_PLAN_ROW_TIMEOUT_SECONDS == 1800
    assert helper.OUTER_ROW_TIMEOUT_SECONDS == 2400
    assert helper.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE == 0x00002000
    assert "SAME_ALGORITHM_AND_METRIC_CODE_OPERATIONAL_TIMEOUT_OVERRIDE_ONLY" in source
    assert "runtime_authority\": False" in source
    assert "selection_authority\": False" in source


def test_metric_timeout_guard_allows_exact_command_once_and_only_changes_timeout(
    tmp_path: Path,
) -> None:
    helper = _load_helper()
    project_root = tmp_path.resolve()
    metric = project_root / "independent_reproduction/recompute_v21e3r1_metrics.py"
    trace = project_root / "attempt/trace.sqlite3"
    output = project_root / "attempt/independent.metric.json"
    expected_command = [
        str(helper.EXPECTED_INTERPRETER_PATH),
        str(metric),
        "--trace",
        str(trace),
        "--lower=0.0,1.0",
        "--upper=2.0,3.0",
        "--expected-evaluations",
        "2000",
        "--output",
        str(output),
    ]
    observed: list[tuple[list[str], dict[str, object]]] = []
    boundary_checks = 0

    def verify_boundary() -> None:
        nonlocal boundary_checks
        boundary_checks += 1

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.append((command, dict(kwargs)))
        return subprocess.CompletedProcess(command, 0, "ok", "")

    guard = helper.MetricTimeoutRunGuard(
        original_run=fake_run,
        expected_command=expected_command,
        project_root=project_root,
        verify_boundary=verify_boundary,
    )
    result = guard(
        list(expected_command),
        cwd=project_root,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    guard.assert_exactly_once()

    assert result.returncode == 0
    assert len(observed) == 1
    assert observed[0][0] == expected_command
    assert observed[0][1] == {
        "cwd": project_root,
        "text": True,
        "capture_output": True,
        "timeout": 1200,
        "check": False,
    }
    assert guard.original_timeout_seconds == 300
    assert guard.operational_timeout_seconds == 1200
    assert guard.completed_returncode == 0
    assert boundary_checks == 2


@pytest.mark.parametrize(
    "mutation",
    ["command", "timeout", "extra_kwarg"],
)
def test_metric_timeout_guard_rejects_command_or_frozen_kwargs_drift(
    tmp_path: Path, mutation: str
) -> None:
    helper = _load_helper()
    project_root = tmp_path.resolve()
    command = [str(helper.EXPECTED_INTERPRETER_PATH), "metric.py"]
    guard = helper.MetricTimeoutRunGuard(
        original_run=lambda *args, **kwargs: pytest.fail("drift reached subprocess"),
        expected_command=command,
        project_root=project_root,
        verify_boundary=lambda: None,
    )
    observed = list(command)
    kwargs: dict[str, object] = {
        "cwd": project_root,
        "text": True,
        "capture_output": True,
        "timeout": 300,
        "check": False,
    }
    if mutation == "command":
        observed.append("--attacker")
    elif mutation == "timeout":
        kwargs["timeout"] = 301
    else:
        kwargs["env"] = {}

    with pytest.raises(RuntimeError, match="drifted"):
        guard(observed, **kwargs)
    assert guard.call_count == 0


def test_metric_timeout_guard_rejects_second_subprocess_call(tmp_path: Path) -> None:
    helper = _load_helper()
    project_root = tmp_path.resolve()
    command = [str(helper.EXPECTED_INTERPRETER_PATH), "metric.py"]
    guard = helper.MetricTimeoutRunGuard(
        original_run=lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, "", ""),
        expected_command=command,
        project_root=project_root,
        verify_boundary=lambda: None,
    )
    kwargs = {
        "cwd": project_root,
        "text": True,
        "capture_output": True,
        "timeout": 300,
        "check": False,
    }
    guard(list(command), **kwargs)

    with pytest.raises(RuntimeError, match="more than one subprocess"):
        guard(list(command), **kwargs)
    assert guard.call_count == 1


def test_metric_source_drift_is_rejected_before_operational_subprocess(
    tmp_path: Path,
) -> None:
    helper = _load_helper()
    project_root = tmp_path.resolve()
    command = [str(helper.EXPECTED_INTERPRETER_PATH), "metric.py"]

    def reject_drift() -> None:
        raise RuntimeError("frozen metric source drifted")

    guard = helper.MetricTimeoutRunGuard(
        original_run=lambda *args, **kwargs: pytest.fail("drift reached subprocess"),
        expected_command=command,
        project_root=project_root,
        verify_boundary=reject_drift,
    )
    with pytest.raises(RuntimeError, match="metric source drifted"):
        guard(
            list(command),
            cwd=project_root,
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
    assert guard.call_count == 0


def test_windows_job_outer_timeout_kills_wrapper_and_grandchild_before_terminal(
    tmp_path: Path,
) -> None:
    helper = _load_helper()
    wrapper = tmp_path / "job_wrapper.py"
    child_pid_path = tmp_path / "grandchild.pid"
    escaped_write_path = tmp_path / "grandchild-escaped.txt"
    wrapper.write_text(
        "import pathlib, subprocess, sys, time\n"
        "gate = sys.stdin.readline()\n"
        "if gate != sys.argv[3] + '\\n': raise SystemExit(91)\n"
        "child = \"import os,pathlib,sys,time; "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
        "time.sleep(2); pathlib.Path(sys.argv[2]).write_text('escaped')\"\n"
        "subprocess.Popen([sys.executable, '-c', child, sys.argv[1], sys.argv[2]])\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    def terminal_zero() -> dict[str, object]:
        if not child_pid_path.is_file():
            raise RuntimeError("grandchild PID witness missing")
        pid = int(child_pid_path.read_text(encoding="utf-8"))
        if _windows_pid_is_alive(pid):
            raise RuntimeError("grandchild still alive")
        return {"schema": "synthetic_descendant_zero_v1", "grandchild_pid": pid}

    with pytest.raises(helper.JobControlledProcessTimeout) as caught:
        helper._run_in_windows_kill_on_close_job(
            [
                sys.executable,
                str(wrapper),
                str(child_pid_path),
                str(escaped_write_path),
                helper.JOB_START_GATE_ARGUMENT,
            ],
            cwd=tmp_path,
            environment=os.environ,
            timeout_seconds=1.0,
            start_gate_line=helper.JOB_START_GATE_LINE,
            terminal_zero_check=terminal_zero,
        )

    witness = caught.value.witness
    assert witness["job_assignment_verified_before_gate_release"] is True
    assert witness["outer_timeout_fired"] is True
    assert witness["terminal_active_processes"] == 0
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    assert not _windows_pid_is_alive(child_pid)
    time.sleep(2.2)
    assert not escaped_write_path.exists()


def test_windows_job_assignment_failure_never_releases_worker_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    wrapper = tmp_path / "gated_wrapper.py"
    gate_released = tmp_path / "gate-released.txt"
    wrapper.write_text(
        "import pathlib,sys,time\n"
        "line=sys.stdin.readline()\n"
        "pathlib.Path(sys.argv[1]).write_text(line)\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    def reject_assignment(self, process) -> None:
        del self, process
        raise RuntimeError("injected AssignProcessToJobObject failure")

    monkeypatch.setattr(
        helper._WindowsKillOnCloseJob,
        "assign_and_verify",
        reject_assignment,
    )

    with pytest.raises(RuntimeError, match="AssignProcessToJobObject"):
        helper._run_in_windows_kill_on_close_job(
            [sys.executable, str(wrapper), str(gate_released)],
            cwd=tmp_path,
            environment=os.environ,
            timeout_seconds=2,
            start_gate_line=helper.JOB_START_GATE_LINE,
            terminal_zero_check=lambda: {
                "schema": "synthetic_assignment_failure_zero_v1"
            },
        )

    assert not gate_released.exists()


def test_active_main_is_rejected_before_claim_with_zero_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    context = _synthetic_context(tmp_path, helper)
    monkeypatch.setattr(helper, "_verify_static_boundary", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper, "_verify_preserved_evidence", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper, "_assert_owned_recovery_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        helper,
        "_assert_process_boundary",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("Active original main driver")
        ),
    )

    with pytest.raises(RuntimeError, match="Active original main"):
        helper.execute_recovery(
            context,
            object(),
            object(),
            interpreter_identity={},
            environment_receipt={},
        )

    assert not (context.output / helper.CLAIM_NAME).exists()
    assert not (context.output / helper.FAILURE_NAME).exists()
    assert not (context.output / helper.FAILURE_SEAL_NAME).exists()


def test_claim_collision_loser_performs_zero_writes_and_never_emits_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    runner = helper._load_frozen_runner(PROJECT_ROOT)
    context = _synthetic_context(tmp_path, helper)
    claim_path = context.output / helper.CLAIM_NAME
    claim_path.write_bytes(b"winner-owned-claim\n")
    before = {
        path.relative_to(context.output).as_posix(): path.read_bytes()
        for path in context.output.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(helper, "_verify_static_boundary", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper, "_verify_preserved_evidence", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper, "_assert_owned_recovery_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        helper,
        "_assert_process_boundary",
        lambda *args, **kwargs: _empty_scan(helper),
    )

    with pytest.raises(RuntimeError, match="claim.*exists|claim.*race"):
        helper.execute_recovery(
            context,
            runner,
            object(),
            interpreter_identity={},
            environment_receipt={},
        )

    after = {
        path.relative_to(context.output).as_posix(): path.read_bytes()
        for path in context.output.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not (context.output / helper.FAILURE_NAME).exists()
    assert not (context.output / helper.FAILURE_SEAL_NAME).exists()


def test_claim_changed_before_acquisition_is_retained_without_owned_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    runner = helper._load_frozen_runner(PROJECT_ROOT)
    context = _synthetic_context(tmp_path, helper)
    monkeypatch.setattr(helper, "_verify_static_boundary", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper, "_verify_preserved_evidence", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper, "_assert_owned_recovery_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        helper,
        "_assert_process_boundary",
        lambda *args, **kwargs: _empty_scan(helper),
    )
    original_fsync = helper._fsync_file

    def mutate_claim_after_fsync(path: Path) -> None:
        original_fsync(path)
        if path == context.output / helper.CLAIM_NAME:
            path.write_bytes(b"externally-replaced-claim\n")

    monkeypatch.setattr(helper, "_fsync_file", mutate_claim_after_fsync)

    with pytest.raises(RuntimeError, match="claim"):
        helper.execute_recovery(
            context,
            runner,
            object(),
            interpreter_identity={},
            environment_receipt={},
        )

    assert (context.output / helper.CLAIM_NAME).read_bytes() == (
        b"externally-replaced-claim\n"
    )
    assert not (context.output / helper.FAILURE_NAME).exists()
    assert not (context.output / helper.FAILURE_SEAL_NAME).exists()


def test_claim_fsync_failure_never_enters_owned_failure_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    runner = helper._load_frozen_runner(PROJECT_ROOT)
    context = _synthetic_context(tmp_path, helper)
    monkeypatch.setattr(helper, "_verify_static_boundary", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper, "_verify_preserved_evidence", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper, "_assert_owned_recovery_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        helper,
        "_assert_process_boundary",
        lambda *args, **kwargs: _empty_scan(helper),
    )
    original_fsync = helper._fsync_file

    def fail_claim_fsync(path: Path) -> None:
        if path == context.output / helper.CLAIM_NAME:
            raise OSError("injected claim fsync failure")
        original_fsync(path)

    monkeypatch.setattr(helper, "_fsync_file", fail_claim_fsync)

    with pytest.raises(RuntimeError, match="before exact claim acquisition"):
        helper.execute_recovery(
            context,
            runner,
            object(),
            interpreter_identity={},
            environment_receipt={},
        )

    assert (context.output / helper.CLAIM_NAME).is_file()
    assert not (context.output / helper.FAILURE_NAME).exists()
    assert not (context.output / helper.FAILURE_SEAL_NAME).exists()


def test_claim_replaced_after_acquisition_is_not_mislabelled_as_owned_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    runner = helper._load_frozen_runner(PROJECT_ROOT)
    context = _synthetic_context(tmp_path, helper)
    monkeypatch.setattr(helper, "_verify_static_boundary", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper, "_verify_preserved_evidence", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper, "_assert_owned_recovery_state", lambda *args, **kwargs: None)
    calls = 0

    def replace_after_claim(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _empty_scan(helper)
        (context.output / helper.CLAIM_NAME).write_bytes(
            b"externally-replaced-after-acquisition\n"
        )
        raise RuntimeError("injected post-acquisition boundary failure")

    monkeypatch.setattr(helper, "_assert_process_boundary", replace_after_claim)

    with pytest.raises(RuntimeError, match="claim changed after acquisition"):
        helper.execute_recovery(
            context,
            runner,
            object(),
            interpreter_identity={},
            environment_receipt={},
        )

    assert (context.output / helper.CLAIM_NAME).read_bytes() == (
        b"externally-replaced-after-acquisition\n"
    )
    assert not (context.output / helper.FAILURE_NAME).exists()
    assert not (context.output / helper.FAILURE_SEAL_NAME).exists()
    assert not (context.output / helper.QUARANTINE_NAME).exists()


def test_postclaim_boundary_failure_writes_durable_hold_and_retains_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    runner = helper._load_frozen_runner(PROJECT_ROOT)
    context = _synthetic_context(tmp_path, helper)
    monkeypatch.setattr(helper, "_verify_static_boundary", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper, "_verify_preserved_evidence", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper, "_assert_owned_recovery_state", lambda *args, **kwargs: None)
    calls = 0

    def process_boundary(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _empty_scan(helper)
        raise RuntimeError("injected postclaim process failure")

    monkeypatch.setattr(helper, "_assert_process_boundary", process_boundary)
    monkeypatch.setattr(
        helper,
        "_wait_for_worker_specs_process_zero",
        lambda *args, **kwargs: {
            "schema": "synthetic_descendant_zero_v1",
            "terminal_matching_process_count": 0,
        },
    )

    with pytest.raises(RuntimeError, match="stopped fail-closed"):
        helper.execute_recovery(
            context,
            runner,
            object(),
            interpreter_identity={},
            environment_receipt={},
        )

    assert (context.output / helper.CLAIM_NAME).is_file()
    assert (context.output / helper.FAILURE_NAME).is_file()
    assert (context.output / helper.FAILURE_SEAL_NAME).is_file()
    failure = json.loads(
        (context.output / helper.FAILURE_NAME).read_text(encoding="utf-8")
    )
    assert failure["status"] == (
        "HOLD_METRIC_TIMEOUT_RECOVERY_FAILURE_MANUAL_AUDIT_REQUIRED"
    )
    assert failure["main_runner_resume_authorized"] is False
    assert failure["automatic_retry_authorized"] is False
    assert failure["owned_attempts"] == []


def test_unconfirmed_descendant_state_writes_quarantine_not_durable_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    runner = helper._load_frozen_runner(PROJECT_ROOT)
    context = _synthetic_context(tmp_path, helper)
    monkeypatch.setattr(helper, "_verify_static_boundary", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper, "_verify_preserved_evidence", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper, "_assert_owned_recovery_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        helper,
        "_assert_process_boundary",
        lambda *args, **kwargs: _empty_scan(helper),
    )

    def unconfirmed_child(*args: object, **kwargs: object) -> dict[str, object]:
        raise helper.DescendantTerminationUnconfirmed(
            "synthetic orphan process remained NOT_TERMINAL"
        )

    monkeypatch.setattr(helper, "_run_recovery_child", unconfirmed_child)

    with pytest.raises(RuntimeError, match="NOT_TERMINAL|quarantine"):
        helper.execute_recovery(
            context,
            runner,
            object(),
            interpreter_identity={},
            environment_receipt={},
        )

    quarantine = json.loads(
        (context.output / helper.QUARANTINE_NAME).read_text(encoding="utf-8")
    )
    assert quarantine["status"] == (
        "NOT_TERMINAL_DESCENDANT_TERMINATION_UNCONFIRMED"
    )
    assert quarantine["durable_terminal_failure"] is False
    assert quarantine["main_runner_resume_authorized"] is False
    assert not (context.output / helper.FAILURE_NAME).exists()
    assert not (context.output / helper.FAILURE_SEAL_NAME).exists()


@pytest.mark.parametrize(
    "mutation", ["ordinal446_attempt2", "ordinal447_attempt1", "ordinal447_marker"]
)
def test_recovery_attempt_layout_drift_is_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    helper = _load_helper()
    context = _synthetic_context(tmp_path, helper)
    if mutation == "ordinal446_attempt2":
        (
            context.output
            / "attempts"
            / context.rows[0].row_id
            / "attempt-0002"
        ).mkdir()
    elif mutation == "ordinal447_attempt1":
        (
            context.output
            / "attempts"
            / context.rows[1].row_id
            / "attempt-0001"
        ).mkdir(parents=True)
    else:
        marker = context.output / "completed" / f"{context.rows[1].row_id}.json"
        marker.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="attempt|marker"):
        helper._assert_owned_recovery_state(
            context,
            object(),
            owned_attempts={},
            sealed_rows=set(),
        )


@pytest.mark.parametrize("worker_mode", ["success", "second_call"])
def test_frozen_worker_patch_is_child_local_exact_once_and_always_restored(
    tmp_path: Path, worker_mode: str
) -> None:
    helper = _load_helper()
    frozen = helper._load_frozen_runner(PROJECT_ROOT)
    attempt = tmp_path / "attempt-0001"
    attempt.mkdir()
    spec_path = attempt / "worker.spec.json"
    spec = {
        "case_id": helper.TARGET_CASE_ID,
        "seed": 31057,
        "arm_id": "MOEAD_POP21",
        "objective_lower_bounds": [0.0, 1.0],
        "objective_upper_bounds": [2.0, 3.0],
        "charged_evaluation_budget": 2000,
    }
    frozen._exclusive_json(spec_path, spec)
    expected_command = helper._expected_metric_command(
        project_root=PROJECT_ROOT, spec_path=spec_path, spec=spec
    )
    original_calls: list[dict[str, object]] = []

    def original_run(command: list[str], **kwargs: object):
        original_calls.append(dict(kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    subprocess_binding = SimpleNamespace(run=original_run)
    fake_runner = SimpleNamespace(
        subprocess=subprocess_binding,
        _load_json_object=frozen._load_json_object,
        _exclusive_json=frozen._exclusive_json,
        _sha256=frozen._sha256,
    )

    def worker_run(path: Path) -> dict[str, object]:
        kwargs = {
            "cwd": PROJECT_ROOT,
            "text": True,
            "capture_output": True,
            "timeout": 300,
            "check": False,
        }
        fake_runner.subprocess.run(list(expected_command), **kwargs)
        if worker_mode == "second_call":
            fake_runner.subprocess.run(list(expected_command), **kwargs)
        return {
            "status": "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY",
            "row_sha256": "a" * 64,
            "diagnostic_sha256": "b" * 64,
            "trace_sha256": "c" * 64,
            "terminal_receipt_sha256": "d" * 64,
            "independent_metric_receipt_sha256": "e" * 64,
        }

    fake_runner._worker_run = worker_run
    if worker_mode == "success":
        helper._execute_frozen_worker_with_timeout_override(
            fake_runner,
            spec_path,
            project_root=PROJECT_ROOT,
            helper_sha256=helper._sha256(HELPER_PATH),
            interpreter_identity={},
            verify_boundary=lambda: None,
        )
        assert len(original_calls) == 1
        assert original_calls[0]["timeout"] == 1200
        assert (attempt / "worker.result.json").is_file()
        assert (attempt / helper.OVERRIDE_WITNESS_NAME).is_file()
    else:
        with pytest.raises(RuntimeError, match="more than one subprocess"):
            helper._execute_frozen_worker_with_timeout_override(
                fake_runner,
                spec_path,
                project_root=PROJECT_ROOT,
                helper_sha256=helper._sha256(HELPER_PATH),
                interpreter_identity={},
                verify_boundary=lambda: None,
            )
        assert len(original_calls) == 1
        assert not (attempt / "worker.result.json").exists()
    assert fake_runner.subprocess.run is original_run


def _install_synthetic_success_child(
    monkeypatch: pytest.MonkeyPatch,
    helper: ModuleType,
    runner: ModuleType,
    *,
    delay_seconds: float,
) -> None:
    def fake_child(
        child_context,
        spec_path: Path,
        *,
        helper_sha256: str,
        claim_sha256: str,
        process_guard,
    ) -> dict[str, object]:
        del child_context, helper_sha256, claim_sha256, process_guard
        spec = runner._load_json_object(spec_path)
        attempt = spec_path.parent
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
        runner._exclusive_json(attempt / "terminal.receipt.json", {"status": "PASS"})
        runner._exclusive_json(attempt / "independent.metric.json", {"status": "PASS"})
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
            attempt / helper.OVERRIDE_WITNESS_NAME,
            {"receipt_payload_sha256": "f" * 64},
        )
        return {
            "schema": "v21e3r1_windows_kill_on_close_job_witness_v1",
            "kill_on_job_close_limit": True,
            "job_limit_flags": helper.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            "wrapper_pid": 12345,
            "job_assignment_verified_before_gate_release": True,
            "outer_timeout_seconds": helper.OUTER_ROW_TIMEOUT_SECONDS,
            "outer_timeout_fired": False,
            "wrapper_returncode": 0,
            "active_processes_after_wrapper": 0,
            "terminal_active_processes": 0,
            "terminal_process_scan": {},
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
            child_runner._load_json_object(attempt / helper.OVERRIDE_WITNESS_NAME),
        )

    monkeypatch.setattr(helper, "_run_recovery_child", fake_child)
    monkeypatch.setattr(helper, "_validate_recovery_result", fake_validate)


def test_two_helpers_simultaneous_claim_has_one_pass_and_zero_write_loser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    runner = helper._load_frozen_runner(PROJECT_ROOT)
    context = _synthetic_context(tmp_path, helper)
    monkeypatch.setattr(helper, "_verify_static_boundary", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper, "_verify_preserved_evidence", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        helper,
        "_assert_process_boundary",
        lambda *args, **kwargs: _empty_scan(helper),
    )
    _install_synthetic_success_child(
        monkeypatch, helper, runner, delay_seconds=0.02
    )
    original_exclusive = runner._exclusive_json
    claim_barrier = threading.Barrier(2)

    def racing_exclusive(path: Path, payload: object) -> None:
        if Path(path) == context.output / helper.CLAIM_NAME and not Path(path).exists():
            claim_barrier.wait(timeout=10)
        original_exclusive(path, payload)

    monkeypatch.setattr(runner, "_exclusive_json", racing_exclusive)
    results: list[dict[str, object]] = []
    errors: list[BaseException] = []
    result_lock = threading.Lock()

    def invoke() -> None:
        try:
            value = helper.execute_recovery(
                context,
                runner,
                object(),
                interpreter_identity={},
                environment_receipt={},
            )
            with result_lock:
                results.append(value)
        except BaseException as error:
            with result_lock:
                errors.append(error)

    threads = [threading.Thread(target=invoke) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert all(not thread.is_alive() for thread in threads)
    assert [result["status"] for result in results] == [
        "PASS_EXACT_17_OPERATIONAL_TIMEOUT_RECOVERY_ONLY"
    ]
    assert len(errors) == 1
    assert "zero recovery writes" in str(errors[0])
    assert not (context.output / helper.FAILURE_NAME).exists()
    assert not (context.output / helper.FAILURE_SEAL_NAME).exists()
    assert not (context.output / helper.QUARANTINE_NAME).exists()


def test_synthetic_exact17_schedule_uses_expected_attempts_and_seals_original_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    runner = helper._load_frozen_runner(PROJECT_ROOT)
    context = _synthetic_context(tmp_path, helper)
    monkeypatch.setattr(helper, "_verify_static_boundary", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper, "_verify_preserved_evidence", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        helper,
        "_assert_process_boundary",
        lambda *args, **kwargs: _empty_scan(helper),
    )
    state = {"active": 0, "max_active": 0, "calls": 0}
    state_lock = threading.Lock()

    def fake_child(
        child_context,
        spec_path: Path,
        *,
        helper_sha256: str,
        claim_sha256: str,
        process_guard,
    ) -> dict[str, object]:
        del child_context, helper_sha256, claim_sha256, process_guard
        spec = runner._load_json_object(spec_path)
        attempt = spec_path.parent
        with state_lock:
            state["active"] += 1
            state["calls"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        try:
            time.sleep(0.01)
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
                attempt / helper.OVERRIDE_WITNESS_NAME,
                {"receipt_payload_sha256": "f" * 64},
            )
        finally:
            with state_lock:
                state["active"] -= 1
        return {
            "schema": "v21e3r1_windows_kill_on_close_job_witness_v1",
            "kill_on_job_close_limit": True,
            "job_limit_flags": helper.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            "wrapper_pid": 12345,
            "job_assignment_verified_before_gate_release": True,
            "outer_timeout_seconds": helper.OUTER_ROW_TIMEOUT_SECONDS,
            "outer_timeout_fired": False,
            "wrapper_returncode": 0,
            "active_processes_after_wrapper": 0,
            "terminal_active_processes": 0,
            "terminal_process_scan": {},
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
                attempt / helper.OVERRIDE_WITNESS_NAME
            ),
        )

    monkeypatch.setattr(helper, "_run_recovery_child", fake_child)
    monkeypatch.setattr(helper, "_validate_recovery_result", fake_validate)

    receipt = helper.execute_recovery(
        context,
        runner,
        object(),
        interpreter_identity={},
        environment_receipt={},
    )

    assert state["calls"] == 17
    assert 2 <= state["max_active"] <= 4
    assert receipt["status"] == "PASS_EXACT_17_OPERATIONAL_TIMEOUT_RECOVERY_ONLY"
    assert receipt["fresh_full_algorithm_rerun_count"] == 17
    assert receipt["failed_trace_reuse"] is False
    assert receipt["metric_only_replay"] is False
    assert receipt["runtime_authority"] is False
    assert receipt["selection_authority"] is False
    assert len(list((context.output / "completed").glob("*.json"))) == 17
    assert (
        context.output
        / "attempts"
        / context.rows[0].row_id
        / "attempt-0002"
    ).is_dir()
    for row in context.rows[1:]:
        assert (
            context.output
            / "attempts"
            / row.row_id
            / "attempt-0001"
        ).is_dir()
    assert not (context.output / "diagnostic.aggregate.json").exists()
    assert not (context.output / "diagnostic.receipt.json").exists()


def test_real_postincident_predecessor_preflight_fails_closed_and_is_read_only() -> None:
    output = PROJECT_ROOT / (
        "outputs/v21e3r1_v7_exposed_development_diagnostics_20260823"
    )
    helper = _load_helper()
    claim_path = output / helper.CLAIM_NAME
    failure_path = output / helper.FAILURE_NAME
    failure_seal_path = output / helper.FAILURE_SEAL_NAME
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    target_row_ids = claim["target_row_ids"]

    def immutable_snapshot() -> tuple[tuple[str, str, int | None, str | None], ...]:
        paths = list(output.glob("metric-timeout-recovery.*"))
        paths.extend((output / "completed").iterdir())
        for row_id in target_row_ids:
            row_root = output / "attempts" / row_id
            if row_root.exists():
                paths.append(row_root)
                paths.extend(row_root.rglob("*"))
        entries = []
        for path in sorted(set(paths), key=lambda item: item.as_posix()):
            relative = path.relative_to(output).as_posix()
            if path.is_symlink():
                entries.append((relative, "symlink", None, None))
            elif path.is_dir():
                entries.append((relative, "directory", None, None))
            elif path.is_file():
                entries.append(
                    (relative, "file", path.stat().st_size, helper._sha256(path))
                )
            else:
                entries.append((relative, "other", None, None))
        return tuple(entries)

    before = immutable_snapshot()
    completed = subprocess.run(
        [
            r"C:\miniconda3\python.exe",
            str(HELPER_PATH),
            "--project-root",
            str(PROJECT_ROOT),
            "--output-directory",
            str(output),
            "--preflight-only",
        ],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 3
    assert "FAIL_CLOSED" in completed.stderr
    assert "recovery evidence already exists" in completed.stderr.lower()
    assert immutable_snapshot() == before
    assert claim_path.is_file()
    assert failure_path.is_file()
    assert failure_seal_path.is_file()
    assert not (output / helper.RECEIPT_NAME).exists()
    assert not (output / helper.RECEIPT_SEAL_NAME).exists()
    assert not (output / helper.QUARANTINE_NAME).exists()
    assert not (output / "diagnostic.aggregate.json").exists()
    assert not (output / "diagnostic.receipt.json").exists()
