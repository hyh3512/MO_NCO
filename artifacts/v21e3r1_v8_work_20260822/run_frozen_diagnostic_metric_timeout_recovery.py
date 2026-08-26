from __future__ import annotations

"""Recover the exact metric-timeout gap without changing frozen science code.

This is an external operational wrapper.  It reruns the exact frozen worker in
an isolated process and changes only the independent metric subprocess timeout
from the frozen 300 seconds to 1200 seconds.  It never authorizes runtime,
scientific, selection, confirmation, formal-study, or publication claims.
"""

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import ctypes
from ctypes import wintypes
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
EXPECTED_INTERPRETER_PATH = Path(r"C:\miniconda3\python.exe")
EXPECTED_INTERPRETER_SHA256 = (
    "f77193cf0405ab440c39324bdb2f8864596321c1df888adbbe357f3d760f4716"
)
EXPECTED_FULL_ROWS = 504
RECOVERY_ORDINALS = tuple(range(446, 463))
EXPECTED_RECOVERY_ROWS = 17
JOBS = 4
ORIGINAL_METRIC_TIMEOUT_SECONDS = 300
OPERATIONAL_METRIC_TIMEOUT_SECONDS = 1200
FROZEN_PLAN_ROW_TIMEOUT_SECONDS = 1800
OUTER_ROW_TIMEOUT_SECONDS = 2400
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
JOB_START_GATE_ARGUMENT = "V21E3R1_WINDOWS_JOB_ASSIGNED_V1"
JOB_START_GATE_LINE = JOB_START_GATE_ARGUMENT + "\n"
DESCENDANT_ZERO_TIMEOUT_SECONDS = 30
TARGET_CASE_ID = "v21e3-motsp-development-n500-s00"
RUNNER_RELATIVE = Path(
    "ijoc_submission_v21e3r1/scripts/run_v21e3r1_development_diagnostics.py"
)
METRIC_RELATIVE = Path("independent_reproduction/recompute_v21e3r1_metrics.py")
PROCESS_GUARD_RELATIVE = Path(
    "artifacts/v21e3r1_v8_work_20260822/"
    "run_frozen_diagnostic_disjoint_case_parallel.py"
)
HELPER_RELATIVE = Path(
    "artifacts/v21e3r1_v8_work_20260822/"
    "run_frozen_diagnostic_metric_timeout_recovery.py"
)
OUTPUT_RELATIVE = Path(
    "outputs/v21e3r1_v7_exposed_development_diagnostics_20260823"
)
PLAN_NAME = "diagnostic.plan.json"
CLAIM_NAME = (
    "metric-timeout-recovery.v21e3-motsp-development-n500-s00."
    "helper-instance.claim.json"
)
FAILURE_NAME = (
    "metric-timeout-recovery.v21e3-motsp-development-n500-s00."
    "failure.receipt.json"
)
FAILURE_SEAL_NAME = (
    "metric-timeout-recovery.v21e3-motsp-development-n500-s00."
    "failure.receipt.seal.json"
)
QUARANTINE_NAME = (
    "metric-timeout-recovery.v21e3-motsp-development-n500-s00."
    "descendant-state.quarantine.json"
)
RECEIPT_NAME = (
    "metric-timeout-recovery.v21e3-motsp-development-n500-s00.receipt.json"
)
RECEIPT_SEAL_NAME = (
    "metric-timeout-recovery.v21e3-motsp-development-n500-s00."
    "receipt.seal.json"
)
OVERRIDE_WITNESS_NAME = "operational.metric-timeout-override.receipt.json"
RECOVERY_SEMANTICS = (
    "SAME_ALGORITHM_AND_METRIC_CODE_OPERATIONAL_TIMEOUT_OVERRIDE_ONLY"
)
EXPECTED_FAILED_ATTEMPT_SHA256 = {
    "worker.spec.json": "c9657675fa3cd2073b6a27476a4ef177141ccd7e3cb42e64b1d69144a7e995f3",
    "trace.sqlite3": "68f571e08665a7fa422fe06555b663fa991083b3fd50599d3a09ff446820cfd8",
    "terminal.receipt.json": "2624ba1610a74d7340b904a6afdaf9e072fe3b711942128a868b6908dc458e30",
    "failure.receipt.json": "49b968a071bb8efa26f224d2518b45a3ff9dd5a95cbf70cae795a7acf79b2da5",
}
UPSTREAM_SCHEDULING_RECEIPT_NAME = (
    "external-scheduling.v21e3-motsp-development-n500-s01.receipt.json"
)
UPSTREAM_SCHEDULING_RECEIPT_SHA256 = (
    "bfa1952ec5b585d5a6e192201090c9097119539b38ee63a21df6f3a4a2e9707c"
)
UPSTREAM_SCHEDULING_SEAL_NAME = (
    "external-scheduling.v21e3-motsp-development-n500-s01.receipt.seal.json"
)
UPSTREAM_SCHEDULING_SEAL_SHA256 = (
    "6113d749ab3a24c4955fc145bfef4627db597ce08f4bc2e5083893974f521c3a"
)


@dataclass(frozen=True)
class RecoveryRow:
    ordinal: int
    row_id: str
    expected_attempt_number: int
    worker_spec: dict[str, object]


@dataclass(frozen=True)
class RecoveryContext:
    project_root: Path
    output: Path
    plan_path: Path
    runner_path: Path
    metric_path: Path
    process_guard_path: Path
    rows: tuple[RecoveryRow, ...]
    all_row_ids: tuple[str, ...]
    non_target_marker_manifest: tuple[dict[str, object], ...]
    preexisting_failed_attempt_manifest: tuple[dict[str, object], ...]


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _JobObjectBasicAccountingInformation(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


class DescendantTerminationUnconfirmed(RuntimeError):
    """The helper cannot prove that a wrapper process tree is terminal."""


class JobControlledProcessTimeout(RuntimeError):
    """The outer row deadline fired after the Job Object tree was terminated."""

    def __init__(self, message: str, witness: Mapping[str, object]) -> None:
        super().__init__(message)
        self.witness = dict(witness)


class _WindowsKillOnCloseJob:
    def __init__(self) -> None:
        if os.name != "nt":
            _fail("Windows Job Object recovery is authorized only on Windows")
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self._kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        self._kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self._kernel32.SetInformationJobObject.restype = wintypes.BOOL
        self._kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
        ]
        self._kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        self._kernel32.IsProcessInJob.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.BOOL),
        ]
        self._kernel32.IsProcessInJob.restype = wintypes.BOOL
        self._kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        self._kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self._kernel32.TerminateJobObject.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        self._handle = self._kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self._closed = False
        limits = _JobObjectExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not self._kernel32.SetInformationJobObject(
            self._handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign_and_verify(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            _fail("Recovery wrapper exited before Job Object assignment")
        process_handle = wintypes.HANDLE(int(process._handle))
        if not self._kernel32.AssignProcessToJobObject(
            self._handle, process_handle
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        in_job = wintypes.BOOL()
        if not self._kernel32.IsProcessInJob(
            process_handle, self._handle, ctypes.byref(in_job)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if not bool(in_job.value):
            _fail("Recovery wrapper Job Object assignment was not verified")

    def active_processes(self) -> int:
        accounting = _JobObjectBasicAccountingInformation()
        returned = wintypes.DWORD()
        if not self._kernel32.QueryInformationJobObject(
            self._handle,
            JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(accounting),
            ctypes.sizeof(accounting),
            ctypes.byref(returned),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(accounting.ActiveProcesses)

    def terminate(self) -> None:
        if not self._kernel32.TerminateJobObject(self._handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def wait_active_zero(self, timeout_seconds: float) -> int:
        deadline = time.monotonic() + timeout_seconds
        while True:
            active = self.active_processes()
            if active == 0:
                return 0
            if time.monotonic() >= deadline:
                _fail(
                    f"Windows Job Object retained {active} active processes"
                )
            time.sleep(0.05)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._handle and not self._kernel32.CloseHandle(self._handle):
            raise ctypes.WinError(ctypes.get_last_error())


def _run_in_windows_kill_on_close_job(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
    start_gate_line: str,
    terminal_zero_check: Callable[[], Mapping[str, object]],
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    if (
        type(command) not in (list, tuple)
        or not command
        or any(type(value) is not str or not value for value in command)
        or type(timeout_seconds) not in (int, float)
        or isinstance(timeout_seconds, bool)
        or timeout_seconds <= 0
        or type(start_gate_line) is not str
        or not start_gate_line
    ):
        _fail("Windows Job Object invocation contract drifted")
    job = _WindowsKillOnCloseJob()
    process: subprocess.Popen[str] | None = None
    assigned = False
    timed_out = False
    primary_error: BaseException | None = None
    stdout = ""
    stderr = ""
    returncode: int | None = None
    active_after_wrapper: int | None = None
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
            active_after_wrapper = job.active_processes()
            if active_after_wrapper != 0:
                primary_error = RuntimeError(
                    "Recovery wrapper exited while Job Object descendants remained"
                )
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
                process_alive = process.poll() is None
                active = job.active_processes() if assigned else 0
                if process_alive or active != 0 or primary_error is not None:
                    if assigned:
                        job.terminate()
                    elif process_alive:
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
                    cleanup_errors.append(
                        "wrapper wait/drain failed: " + repr(error)
                    )
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
            "Recovery wrapper descendant termination is NOT_TERMINAL: "
            + "; ".join(cleanup_errors)
        ) from primary_error
    witness: dict[str, object] = {
        "schema": "v21e3r1_windows_kill_on_close_job_witness_v1",
        "kill_on_job_close_limit": True,
        "job_limit_flags": JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        "wrapper_pid": process.pid if process is not None else None,
        "job_assignment_verified_before_gate_release": assigned,
        "outer_timeout_seconds": timeout_seconds,
        "outer_timeout_fired": timed_out,
        "wrapper_returncode": returncode,
        "active_processes_after_wrapper": active_after_wrapper,
        "terminal_active_processes": terminal_active,
        "terminal_process_scan": dict(terminal_scan),
    }
    if timed_out:
        raise JobControlledProcessTimeout(
            "Recovery wrapper exceeded the operational outer row timeout after "
            "its Windows Job Object process tree reached zero",
            witness,
        ) from primary_error
    if primary_error is not None:
        raise RuntimeError(
            "Recovery wrapper failed after its Windows Job Object process tree "
            "reached zero: " + str(primary_error)
        ) from primary_error
    if returncode is None:
        _fail("Recovery wrapper return code was not observed")
    return (
        subprocess.CompletedProcess(list(command), returncode, stdout, stderr),
        witness,
    )


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


def _exact_keys(
    payload: object, expected: set[str], *, label: str
) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != expected:
        actual = set(payload) if type(payload) is dict else type(payload).__name__
        _fail(f"{label} exact keys drifted: {actual}")
    return payload


def _exact_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} is not an exact lowercase SHA-256")
    return value


def _load_module(path: Path, *, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        _fail(f"Cannot load fixed module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _load_process_guard(project_root: Path) -> ModuleType:
    path = (project_root / PROCESS_GUARD_RELATIVE).resolve()
    if not path.is_file() or _sha256(path) != EXPECTED_PROCESS_GUARD_SHA256:
        _fail("Audited original-process guard source drifted")
    return _load_module(path, name="_v21e3r1_metric_recovery_process_guard")


def _load_frozen_runner(project_root: Path) -> ModuleType:
    path = (project_root / RUNNER_RELATIVE).resolve()
    if not path.is_file() or _sha256(path) != EXPECTED_RUNNER_SHA256:
        _fail("Frozen diagnostic runner source drifted")
    return _load_module(path, name="_v21e3r1_metric_recovery_frozen_runner")


def _validate_production_paths(project_root: Path, output: Path) -> None:
    if Path(__file__).resolve() != (project_root / HELPER_RELATIVE).resolve():
        _fail("Metric-timeout recovery helper is not at its fixed repository path")
    if output != (project_root / OUTPUT_RELATIVE).resolve():
        _fail("Output is not the fixed frozen exact504 diagnostic directory")
    try:
        output.relative_to(project_root)
    except ValueError as error:
        raise RuntimeError("Output escaped the project root") from error


def _validate_interpreter(process_guard: ModuleType) -> dict[str, object]:
    identity = process_guard._exact_interpreter_identity()
    if (
        Path(str(identity.get("resolved_path"))).resolve()
        != EXPECTED_INTERPRETER_PATH.resolve()
        or identity.get("sha256") != EXPECTED_INTERPRETER_SHA256
    ):
        _fail("Recovery interpreter is not the exact historical base interpreter")
    return identity


def _file_manifest(root: Path, paths: Sequence[Path]) -> tuple[dict[str, object], ...]:
    result: list[dict[str, object]] = []
    for path in paths:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root.resolve()).as_posix()
        except ValueError as error:
            raise RuntimeError("Manifest path escaped its evidence root") from error
        if path.is_symlink() or not path.is_file():
            _fail(f"Manifest input is not a regular file: {relative}")
        result.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    return tuple(sorted(result, key=lambda item: str(item["path"])))


def _non_target_marker_manifest(
    output: Path,
    runner: ModuleType,
    *,
    all_row_ids: Sequence[str],
    recovery_row_ids: set[str],
) -> tuple[dict[str, object], ...]:
    completed_root = output / "completed"
    if completed_root.is_symlink() or not completed_root.is_dir():
        _fail("Frozen completed-marker directory is missing or unsafe")
    expected = [row_id for row_id in all_row_ids if row_id not in recovery_row_ids]
    observed_paths = sorted(completed_root.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() or not path.is_file() for path in observed_paths):
        _fail("Completed-marker directory contains a non-regular entry")
    if {path.name for path in observed_paths} != {f"{row_id}.json" for row_id in expected}:
        _fail("Completed-marker set is not the exact 487-row recovery complement")
    manifest: list[dict[str, object]] = []
    for ordinal, row_id in enumerate(all_row_ids, start=1):
        if row_id in recovery_row_ids:
            continue
        marker = completed_root / f"{row_id}.json"
        payload = runner._load_json_object(marker)
        if (
            payload.get("status") != "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY"
            or payload.get("row_id") != row_id
            or payload.get("plan_sha256") != EXPECTED_PLAN_SHA256
            or type(payload.get("attempt_directory")) is not str
        ):
            _fail(f"Non-target completed marker binding drifted: {row_id}")
        attempt = (output / str(payload["attempt_directory"])).resolve()
        try:
            attempt.relative_to((output / "attempts").resolve())
        except ValueError as error:
            raise RuntimeError("Non-target marker attempt escaped output") from error
        if not attempt.is_dir():
            _fail(f"Non-target completed attempt is missing: {row_id}")
        manifest.append(
            {
                "ordinal": ordinal,
                "row_id": row_id,
                "marker_path": marker.relative_to(output).as_posix(),
                "marker_sha256": _sha256(marker),
                "attempt_directory": str(payload["attempt_directory"]),
            }
        )
    if len(manifest) != EXPECTED_FULL_ROWS - EXPECTED_RECOVERY_ROWS:
        _fail("Non-target marker manifest cardinality drifted")
    return tuple(manifest)


def _validate_failed_attempt(
    context_root: Path,
    runner: ModuleType,
    row: RecoveryRow,
    attempt: Path,
) -> tuple[dict[str, object], ...]:
    expected_names = {
        "worker.spec.json",
        "trace.sqlite3",
        "terminal.receipt.json",
        "failure.receipt.json",
    }
    children = sorted(attempt.iterdir(), key=lambda path: path.name)
    if (
        attempt.is_symlink()
        or not attempt.is_dir()
        or any(path.is_symlink() or not path.is_file() for path in children)
        or {path.name for path in children} != expected_names
    ):
        _fail("Ordinal 446 failed attempt does not have the exact preserved layout")
    spec = runner._load_json_object(attempt / "worker.spec.json")
    if runner._canonical_json(spec) != runner._canonical_json(row.worker_spec):
        _fail("Ordinal 446 failed worker spec drifted")
    failure = _exact_keys(
        runner._load_json_object(attempt / "failure.receipt.json"),
        {"schema", "status", "returncode", "stdout_tail", "stderr_tail"},
        label="ordinal 446 failure receipt",
    )
    if (
        failure["schema"] != "v21e3r1_diagnostic_row_failure_v1"
        or failure["status"] != "FAIL_ROW_PROCESS"
        or type(failure["returncode"]) is not int
        or failure["returncode"] != 1
        or type(failure["stdout_tail"]) is not str
        or type(failure["stderr_tail"]) is not str
        or "subprocess.TimeoutExpired" not in failure["stderr_tail"]
        or "timed out after 300 seconds" not in failure["stderr_tail"]
        or METRIC_RELATIVE.as_posix().replace("/", "\\")
        not in failure["stderr_tail"].replace("\\\\", "\\")
    ):
        _fail("Ordinal 446 is not the exact independent-metric timeout failure")
    terminal = runner._load_json_object(attempt / "terminal.receipt.json")
    if (
        terminal.get("schema") != "v21e3_terminal_receipt_v1"
        or terminal.get("status") != "SUCCESS"
        or terminal.get("charged_evaluation_count") != 2000
        or terminal.get("decision_count") != 2000
        or terminal.get("problem") != TARGET_CASE_ID
    ):
        _fail("Ordinal 446 preserved terminal receipt is not exact SUCCESS")
    trace = attempt / "trace.sqlite3"
    if trace.stat().st_size <= 0:
        _fail("Ordinal 446 preserved trace is empty")
    manifest = _file_manifest(context_root, children)
    if {
        Path(str(item["path"])).name: item["sha256"] for item in manifest
    } != EXPECTED_FAILED_ATTEMPT_SHA256:
        _fail("Ordinal 446 preserved failed-attempt bytes drifted")
    return manifest


def _validate_plan_and_build_context(
    project_root: Path,
    output: Path,
    runner: ModuleType,
) -> RecoveryContext:
    plan_path = (output / PLAN_NAME).resolve()
    runner_path = (project_root / RUNNER_RELATIVE).resolve()
    metric_path = (project_root / METRIC_RELATIVE).resolve()
    process_guard_path = (project_root / PROCESS_GUARD_RELATIVE).resolve()
    for path, expected, label in (
        (plan_path, EXPECTED_PLAN_SHA256, "diagnostic plan"),
        (runner_path, EXPECTED_RUNNER_SHA256, "frozen runner"),
        (metric_path, EXPECTED_METRIC_SHA256, "independent metric"),
        (process_guard_path, EXPECTED_PROCESS_GUARD_SHA256, "process guard"),
    ):
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
            _fail(f"Fixed {label} bytes drifted")
    plan = _exact_keys(
        runner._load_json_object(plan_path),
        {
            "schema",
            "status",
            "scientific_scope",
            "case_ids",
            "seeds",
            "arms",
            "charged_evaluation_budget",
            "checkpoint_period",
            "expected_rows",
            "input_binding",
            "source_manifest",
            "row_timeout_seconds",
            "selection_entropy_release",
            "confirmation_materialization",
            "formal_materialization",
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
        or plan["row_timeout_seconds"] != FROZEN_PLAN_ROW_TIMEOUT_SECONDS
        or plan["selection_entropy_release"] != "PROHIBITED"
        or plan["confirmation_materialization"] != "PROHIBITED"
        or plan["formal_materialization"] != "PROHIBITED"
    ):
        _fail("Frozen diagnostic plan semantic contract drifted")
    manifest = _exact_keys(
        plan["source_manifest"],
        {"schema", "hash_rule", "entry_count", "entries", "source_snapshot_sha256"},
        label="frozen source manifest",
    )
    current_manifest = runner._source_manifest(project_root)
    if (
        manifest["schema"] != "v21e3r1_diagnostic_source_manifest_v1"
        or manifest["hash_rule"] != "sha256(canonical_json(sorted_entries))"
        or manifest["entry_count"] != 170
        or manifest["source_snapshot_sha256"] != EXPECTED_SOURCE_ROOT_SHA256
        or runner._canonical_json(manifest) != runner._canonical_json(current_manifest)
    ):
        _fail("Frozen diagnostic source manifest drifted")
    entries = manifest["entries"]
    if type(entries) is not list or len(entries) != 170:
        _fail("Frozen source inventory cardinality drifted")
    for relative, expected_sha, expected_path in (
        (RUNNER_RELATIVE.as_posix(), EXPECTED_RUNNER_SHA256, runner_path),
        (METRIC_RELATIVE.as_posix(), EXPECTED_METRIC_SHA256, metric_path),
    ):
        matches = [
            item
            for item in entries
            if type(item) is dict and item.get("path") == relative
        ]
        if (
            len(matches) != 1
            or matches[0].get("sha256") != expected_sha
            or matches[0].get("bytes") != expected_path.stat().st_size
        ):
            _fail(f"Frozen source manifest does not bind {relative}")
    cases, bounds, directions, input_binding = runner._load_inputs(project_root)
    if runner._canonical_json(input_binding) != runner._canonical_json(
        plan["input_binding"]
    ):
        _fail("Frozen diagnostic input binding drifted")
    target_cases = [case for case in cases if case.get("case_id") == TARGET_CASE_ID]
    if len(target_cases) != 1:
        _fail("Recovery target case is not unique")
    target_case = target_cases[0]
    if target_case.get("family") != "MOTSP" or target_case.get("size") != 500:
        _fail("Recovery target family or size drifted")
    case_path = runner._case_path(project_root, target_case)
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
    target_case_index = list(runner.EXPECTED_CASE_IDS).index(TARGET_CASE_ID)
    rows: list[RecoveryRow] = []
    local_ordinal = 0
    for seed in runner.SEEDS:
        for arm in runner.DIAGNOSTIC_ARMS:
            local_ordinal += 1
            ordinal = (
                target_case_index * len(runner.SEEDS) * len(runner.DIAGNOSTIC_ARMS)
                + local_ordinal
            )
            if ordinal not in RECOVERY_ORDINALS:
                continue
            row_id = f"{TARGET_CASE_ID}__seed-{seed}__arm-{arm.lower()}"
            spec = {
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
            }
            rows.append(
                RecoveryRow(
                    ordinal=ordinal,
                    row_id=row_id,
                    expected_attempt_number=2 if ordinal == RECOVERY_ORDINALS[0] else 1,
                    worker_spec=spec,
                )
            )
    if (
        tuple(row.ordinal for row in rows) != RECOVERY_ORDINALS
        or len(rows) != EXPECTED_RECOVERY_ROWS
        or rows[0].row_id
        != f"{TARGET_CASE_ID}__seed-31057__arm-moead_seeded"
        or rows[-1].row_id
        != f"{TARGET_CASE_ID}__seed-31059__arm-moead_seeded_pop21"
    ):
        _fail("Exact ordinal 446-462 recovery construction drifted")
    recovery_row_ids = {row.row_id for row in rows}
    if (output / "diagnostic.aggregate.json").exists() or (
        output / "diagnostic.receipt.json"
    ).exists():
        _fail("Diagnostic matrix is already finalized")
    for name in (
        CLAIM_NAME,
        FAILURE_NAME,
        FAILURE_SEAL_NAME,
        QUARANTINE_NAME,
        RECEIPT_NAME,
        RECEIPT_SEAL_NAME,
    ):
        if (output / name).exists():
            _fail(f"Metric-timeout recovery evidence already exists: {name}")
    for name, expected_sha256 in (
        (UPSTREAM_SCHEDULING_RECEIPT_NAME, UPSTREAM_SCHEDULING_RECEIPT_SHA256),
        (UPSTREAM_SCHEDULING_SEAL_NAME, UPSTREAM_SCHEDULING_SEAL_SHA256),
    ):
        artifact = output / name
        if artifact.is_symlink() or not artifact.is_file() or _sha256(artifact) != expected_sha256:
            _fail(f"Upstream disjoint-suffix scheduling evidence drifted: {name}")
    attempts_root = output / "attempts"
    if attempts_root.is_symlink() or not attempts_root.is_dir():
        _fail("Frozen attempts directory is missing or unsafe")
    expected_attempt_roots = set(all_row_ids) - {
        row.row_id for row in rows if row.ordinal != RECOVERY_ORDINALS[0]
    }
    observed_attempt_roots = sorted(attempts_root.iterdir(), key=lambda path: path.name)
    if (
        any(path.is_symlink() or not path.is_dir() for path in observed_attempt_roots)
        or {path.name for path in observed_attempt_roots} != expected_attempt_roots
    ):
        _fail("Attempt-root set is not the exact 488-row recovery boundary")
    for row in rows:
        marker = output / "completed" / f"{row.row_id}.json"
        if marker.exists():
            _fail(f"Recovery row already has a completed marker: {row.row_id}")
        row_root = attempts_root / row.row_id
        if row.ordinal == RECOVERY_ORDINALS[0]:
            children = sorted(row_root.iterdir(), key=lambda path: path.name)
            if [path.name for path in children] != ["attempt-0001"]:
                _fail("Ordinal 446 must preserve exactly failed attempt-0001")
        elif row_root.exists():
            _fail(f"Recovery row unexpectedly has an attempt: {row.row_id}")
    non_target_manifest = _non_target_marker_manifest(
        output,
        runner,
        all_row_ids=all_row_ids,
        recovery_row_ids=recovery_row_ids,
    )
    failed_attempt = attempts_root / rows[0].row_id / "attempt-0001"
    failed_manifest = _validate_failed_attempt(
        output, runner, rows[0], failed_attempt
    )
    return RecoveryContext(
        project_root=project_root,
        output=output,
        plan_path=plan_path,
        runner_path=runner_path,
        metric_path=metric_path,
        process_guard_path=process_guard_path,
        rows=tuple(rows),
        all_row_ids=all_row_ids,
        non_target_marker_manifest=non_target_manifest,
        preexisting_failed_attempt_manifest=failed_manifest,
    )


def _authority_hold_fields() -> dict[str, object]:
    return {
        "runtime_authority": False,
        "scientific_authority": False,
        "selection_authority": False,
        "confirmation_authority": False,
        "formal_study_authority": False,
        "publication_status": "IJOC_HOLD",
    }


def _verify_manifest_unchanged(
    root: Path,
    manifest: Sequence[Mapping[str, object]],
    *,
    label: str,
) -> None:
    for item in manifest:
        if set(item) not in (
            {"path", "bytes", "sha256"},
            {"ordinal", "row_id", "marker_path", "marker_sha256", "attempt_directory"},
        ):
            _fail(f"{label} manifest shape drifted")
        relative = item.get("path", item.get("marker_path"))
        expected_sha = item.get("sha256", item.get("marker_sha256"))
        if type(relative) is not str or type(expected_sha) is not str:
            _fail(f"{label} manifest types drifted")
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise RuntimeError(f"{label} manifest escaped its root") from error
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected_sha:
            _fail(f"{label} artifact drifted: {relative}")
        if "bytes" in item and path.stat().st_size != item["bytes"]:
            _fail(f"{label} artifact byte count drifted: {relative}")


def _verify_static_boundary(
    context: RecoveryContext,
    runner: ModuleType,
    process_guard: ModuleType,
    *,
    helper_sha256: str,
    interpreter_identity: Mapping[str, object],
    environment_receipt: Mapping[str, object],
) -> None:
    helper_path = Path(__file__).resolve()
    if (
        _sha256(context.plan_path) != EXPECTED_PLAN_SHA256
        or _sha256(context.runner_path) != EXPECTED_RUNNER_SHA256
        or _sha256(context.metric_path) != EXPECTED_METRIC_SHA256
        or _sha256(context.process_guard_path) != EXPECTED_PROCESS_GUARD_SHA256
        or _sha256(helper_path) != helper_sha256
        or runner._source_manifest(context.project_root).get("source_snapshot_sha256")
        != EXPECTED_SOURCE_ROOT_SHA256
        or _validate_interpreter(process_guard) != dict(interpreter_identity)
        or process_guard._execution_environment_receipt(
            context.project_root, interpreter_identity
        )
        != dict(environment_receipt)
    ):
        _fail("Recovery plan/source/helper/interpreter/environment boundary drifted")
    for name, expected in (
        (UPSTREAM_SCHEDULING_RECEIPT_NAME, UPSTREAM_SCHEDULING_RECEIPT_SHA256),
        (UPSTREAM_SCHEDULING_SEAL_NAME, UPSTREAM_SCHEDULING_SEAL_SHA256),
    ):
        path = context.output / name
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
            _fail("Upstream disjoint-suffix scheduling evidence drifted")


def _verify_preserved_evidence(context: RecoveryContext) -> None:
    _verify_manifest_unchanged(
        context.output,
        context.non_target_marker_manifest,
        label="non-target completed-marker",
    )
    _verify_manifest_unchanged(
        context.output,
        context.preexisting_failed_attempt_manifest,
        label="preexisting failed-attempt",
    )


def _classify_recovery_command(
    command_line: str,
    context: RecoveryContext,
    process_guard: ModuleType,
) -> tuple[str, str | None] | None:
    argv = process_guard._windows_command_line_to_argv(command_line)
    helper_text = process_guard._normalized_path_text(
        str((context.project_root / HELPER_RELATIVE).resolve())
    )
    metric_text = process_guard._normalized_path_text(str(context.metric_path))
    normalized = [process_guard._normalized_path_text(value) for value in argv]
    helper_form = helper_text in normalized
    metric_form = metric_text in normalized
    if not helper_form and not metric_form:
        return None
    if helper_form and metric_form:
        return ("unknown", None)
    if helper_form:
        try:
            spec_value = process_guard._option_value(argv, "--recovery-worker-spec")
        except RuntimeError:
            return ("unknown", None)
        if type(spec_value) is not str or not Path(spec_value).is_absolute():
            return ("unknown", spec_value)
        spec = Path(spec_value).resolve()
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


def _scan_recovery_processes(
    context: RecoveryContext,
    process_guard: ModuleType,
) -> list[dict[str, object]]:
    if os.name != "nt":
        _fail("Metric-timeout recovery process scan is only authorized on Windows")
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
        [
            str(powershell),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            script,
        ],
        text=True,
        encoding="utf-8",
        errors="strict",
        capture_output=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        _fail("Recovery wrapper/metric process scan failed")
    raw = json.loads(completed.stdout)
    if type(raw) is dict:
        raw = [raw]
    if type(raw) is not list:
        _fail("Recovery process scan did not return an array")
    records: list[dict[str, object]] = []
    saw_current = False
    for index, item in enumerate(raw):
        if type(item) is not dict:
            _fail(f"Recovery process row {index} is not an object")
        pid = item.get("ProcessId")
        parent_pid = item.get("ParentProcessId")
        command_line = item.get("CommandLine")
        executable = item.get("ExecutablePath")
        if type(pid) is not int or type(parent_pid) is not int:
            _fail(f"Recovery process row {index} has invalid PID types")
        if pid == os.getpid():
            saw_current = True
            continue
        if command_line is None:
            continue
        if type(command_line) is not str or (
            executable is not None and type(executable) is not str
        ):
            _fail(f"Recovery process row {index} has invalid command types")
        classification = _classify_recovery_command(
            command_line, context, process_guard
        )
        if classification is None:
            continue
        kind, spec_path = classification
        records.append(
            {
                "kind": kind,
                "pid": pid,
                "parent_pid": parent_pid,
                "executable_path": executable,
                "worker_spec_path": spec_path,
                "command_line_sha256": hashlib.sha256(
                    command_line.encode("utf-8")
                ).hexdigest(),
            }
        )
    if not saw_current:
        _fail("Recovery process scan omitted the current helper")
    return records


def _wait_for_worker_specs_process_zero(
    context: RecoveryContext,
    process_guard: ModuleType,
    worker_specs: set[str],
    *,
    block_all_recovery_processes: bool,
    timeout_seconds: float = DESCENDANT_ZERO_TIMEOUT_SECONDS,
) -> dict[str, object]:
    normalized_specs = {Path(value).resolve().as_posix() for value in worker_specs}
    deadline = time.monotonic() + timeout_seconds
    scans = 0
    last_records: list[dict[str, object]] = []
    last_original_scan_sha256: str | None = None
    while True:
        scans += 1
        original_scan = process_guard._assert_no_conflicting_original_processes(
            context.output
        )
        last_original_scan_sha256 = _exact_sha256(
            original_scan.get("scan_payload_sha256"),
            label="descendant-zero original-process scan",
        )
        records = _scan_recovery_processes(context, process_guard)
        last_records = records
        blockers = [
            record
            for record in records
            if block_all_recovery_processes
            or record["kind"] == "unknown"
            or record["worker_spec_path"] in normalized_specs
        ]
        if not blockers:
            core: dict[str, object] = {
                "schema": "v21e3r1_recovery_descendant_zero_scan_v1",
                "worker_specs": sorted(normalized_specs),
                "block_all_recovery_processes": block_all_recovery_processes,
                "scan_count": scans,
                "original_process_scan_payload_sha256": (
                    last_original_scan_sha256
                ),
                "terminal_matching_process_count": 0,
                "terminal_observed_recovery_processes": records,
            }
            return _bound_payload(core, digest_field="scan_payload_sha256")
        if time.monotonic() >= deadline:
            raise DescendantTerminationUnconfirmed(
                "Recovery descendant process scan remained NOT_TERMINAL: "
                + json.dumps(last_records, sort_keys=True)
            )
        time.sleep(0.1)


def _assert_process_boundary(
    context: RecoveryContext,
    process_guard: ModuleType,
    *,
    allowed_worker_specs: set[str] | None = None,
) -> dict[str, object]:
    original_scan = process_guard._assert_no_conflicting_original_processes(
        context.output
    )
    allowed = {
        Path(value).resolve().as_posix() for value in (allowed_worker_specs or set())
    }
    records = _scan_recovery_processes(context, process_guard)
    wrappers = {
        int(record["pid"]): record
        for record in records
        if record["kind"] == "wrapper"
        and record["parent_pid"] == os.getpid()
        and record["worker_spec_path"] in allowed
        and Path(str(record["executable_path"])).resolve()
        == EXPECTED_INTERPRETER_PATH.resolve()
    }
    for record in records:
        kind = record["kind"]
        if kind == "wrapper" and int(record["pid"]) in wrappers:
            continue
        if (
            kind == "metric"
            and record["worker_spec_path"] in allowed
            and int(record["parent_pid"]) in wrappers
            and Path(str(record["executable_path"])).resolve()
            == EXPECTED_INTERPRETER_PATH.resolve()
        ):
            continue
        _fail("Active or ambiguous recovery wrapper/metric process targets output")
    core: dict[str, object] = {
        "schema": "v21e3r1_metric_timeout_recovery_process_scan_v1",
        "original_runner_scan_payload_sha256": original_scan["scan_payload_sha256"],
        "allowed_worker_specs": sorted(allowed),
        "recovery_processes": records,
    }
    result = dict(core)
    result["scan_payload_sha256"] = hashlib.sha256(_canonical_bytes(core)).hexdigest()
    return result


def _bound_payload(
    core: Mapping[str, object], *, digest_field: str
) -> dict[str, object]:
    payload = dict(core)
    payload[digest_field] = hashlib.sha256(_canonical_bytes(core)).hexdigest()
    return payload


def _validate_bound_json(
    path: Path,
    runner: ModuleType,
    *,
    expected: Mapping[str, object],
    digest_field: str,
    label: str,
) -> str:
    before = runner._sha256(path)
    loaded = _exact_keys(
        runner._load_json_object(path), set(expected), label=label
    )
    if loaded != dict(expected):
        _fail(f"{label} exact payload drifted")
    core = dict(loaded)
    digest = _exact_sha256(core.pop(digest_field), label=f"{label} digest")
    if digest != hashlib.sha256(_canonical_bytes(core)).hexdigest():
        _fail(f"{label} payload digest drifted")
    after = runner._sha256(path)
    if before != after:
        _fail(f"{label} changed during verification")
    return before


def _claim_payload(
    context: RecoveryContext,
    *,
    helper_sha256: str,
    interpreter_identity: Mapping[str, object],
    environment_receipt: Mapping[str, object],
    preclaim_process_scan: Mapping[str, object],
) -> dict[str, object]:
    spec_manifest = [
        {
            "ordinal": row.ordinal,
            "row_id": row.row_id,
            "expected_attempt_number": row.expected_attempt_number,
            "worker_spec_payload_sha256": hashlib.sha256(
                _canonical_bytes(row.worker_spec)
            ).hexdigest(),
        }
        for row in context.rows
    ]
    non_target_manifest_sha256 = hashlib.sha256(
        _canonical_bytes(list(context.non_target_marker_manifest))
    ).hexdigest()
    failed_manifest_sha256 = hashlib.sha256(
        _canonical_bytes(list(context.preexisting_failed_attempt_manifest))
    ).hexdigest()
    core: dict[str, object] = {
        "schema": "v21e3r1_metric_timeout_recovery_helper_instance_claim_v1",
        "status": "SEALED_HELPER_INSTANCE_CLAIM_NOT_A_MAIN_RUNNER_LOCK",
        "scope": "EXACT_ORDINALS_446_462_FROZEN_DEVELOPMENT_RECOVERY_ONLY",
        "recovery_semantics": RECOVERY_SEMANTICS,
        "process_id": os.getpid(),
        "target_case_id": TARGET_CASE_ID,
        "target_ordinals": list(RECOVERY_ORDINALS),
        "target_row_count": EXPECTED_RECOVERY_ROWS,
        "target_row_ids": [row.row_id for row in context.rows],
        "worker_spec_payload_manifest": spec_manifest,
        "worker_spec_payload_manifest_sha256": hashlib.sha256(
            _canonical_bytes(spec_manifest)
        ).hexdigest(),
        "non_target_completed_marker_count": len(
            context.non_target_marker_manifest
        ),
        "non_target_completed_marker_manifest_sha256": non_target_manifest_sha256,
        "preexisting_failed_attempt_manifest": list(
            context.preexisting_failed_attempt_manifest
        ),
        "preexisting_failed_attempt_manifest_sha256": failed_manifest_sha256,
        "upstream_scheduling_receipt_sha256": UPSTREAM_SCHEDULING_RECEIPT_SHA256,
        "upstream_scheduling_seal_sha256": UPSTREAM_SCHEDULING_SEAL_SHA256,
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
        "outer_timeout_margin_seconds": (
            OUTER_ROW_TIMEOUT_SECONDS - OPERATIONAL_METRIC_TIMEOUT_SECONDS
        ),
        "fresh_full_algorithm_reruns_required": True,
        "preexisting_failed_trace_reuse_authorized": False,
        "metric_only_replay": False,
        "original_main_runner_honors_this_claim": False,
        "automatic_resume_authorized": False,
        **_authority_hold_fields(),
    }
    return _bound_payload(core, digest_field="claim_payload_sha256")


def _validate_claim(
    context: RecoveryContext,
    runner: ModuleType,
    *,
    expected_claim: Mapping[str, object],
    expected_sha256: str,
) -> None:
    actual = _validate_bound_json(
        context.output / CLAIM_NAME,
        runner,
        expected=expected_claim,
        digest_field="claim_payload_sha256",
        label="metric-timeout recovery claim",
    )
    if actual != expected_sha256:
        _fail("Metric-timeout recovery claim raw bytes drifted")


def _claim_is_exact_owned(
    context: RecoveryContext,
    runner: ModuleType,
    *,
    expected_claim: Mapping[str, object],
    expected_sha256: str,
) -> bool:
    try:
        _validate_claim(
            context,
            runner,
            expected_claim=expected_claim,
            expected_sha256=expected_sha256,
        )
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError):
        return False
    return True


def _assert_failure_evidence_absent(context: RecoveryContext) -> None:
    for name in (FAILURE_NAME, FAILURE_SEAL_NAME, QUARANTINE_NAME):
        path = context.output / name
        if path.exists() or path.is_symlink():
            _fail("Recovery failure evidence appeared on a success path")


class MetricTimeoutRunGuard:
    """One-shot replacement for the frozen runner's subprocess.run binding."""

    def __init__(
        self,
        *,
        original_run: Callable[..., subprocess.CompletedProcess[str]],
        expected_command: Sequence[str],
        project_root: Path,
        verify_boundary: Callable[[], None],
    ) -> None:
        if type(expected_command) not in (list, tuple) or not expected_command:
            _fail("Expected independent metric command must be an exact sequence")
        if any(type(value) is not str or not value for value in expected_command):
            _fail("Expected independent metric command contains invalid arguments")
        self._original_run = original_run
        self._expected_command = tuple(expected_command)
        self._project_root = project_root.resolve()
        self._verify_boundary = verify_boundary
        self._call_count = 0
        self._completed_returncode: int | None = None

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def original_timeout_seconds(self) -> int:
        return ORIGINAL_METRIC_TIMEOUT_SECONDS

    @property
    def operational_timeout_seconds(self) -> int:
        return OPERATIONAL_METRIC_TIMEOUT_SECONDS

    @property
    def command_sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(list(self._expected_command))).hexdigest()

    @property
    def completed_returncode(self) -> int | None:
        return self._completed_returncode

    def __call__(self, *popenargs: object, **kwargs: object):
        if self._call_count != 0:
            _fail("Frozen worker attempted more than one subprocess call")
        if len(popenargs) != 1 or type(popenargs[0]) is not list:
            _fail("Frozen worker subprocess call shape drifted")
        command = popenargs[0]
        if tuple(command) != self._expected_command:
            _fail("Frozen worker subprocess command drifted from independent metric replay")
        expected_keys = {"cwd", "text", "capture_output", "timeout", "check"}
        if set(kwargs) != expected_keys:
            _fail("Frozen independent metric subprocess kwargs drifted")
        cwd = kwargs["cwd"]
        if not isinstance(cwd, Path) or cwd.resolve() != self._project_root:
            _fail("Frozen independent metric subprocess cwd drifted")
        if (
            kwargs["text"] is not True
            or kwargs["capture_output"] is not True
            or type(kwargs["timeout"]) is not int
            or kwargs["timeout"] != ORIGINAL_METRIC_TIMEOUT_SECONDS
            or kwargs["check"] is not False
        ):
            _fail("Frozen independent metric subprocess operational contract drifted")
        self._verify_boundary()
        self._call_count = 1
        overridden = dict(kwargs)
        overridden["timeout"] = OPERATIONAL_METRIC_TIMEOUT_SECONDS
        try:
            result = self._original_run(command, **overridden)
            if type(result.returncode) is not int:
                _fail("Independent metric subprocess returncode type drifted")
            self._completed_returncode = result.returncode
            return result
        finally:
            self._verify_boundary()

    def assert_exactly_once(self) -> None:
        if self._call_count != 1:
            _fail("Frozen worker did not invoke the independent metric exactly once")


def _expected_metric_command(
    *,
    project_root: Path,
    spec_path: Path,
    spec: Mapping[str, object],
) -> list[str]:
    lower = tuple(float(value) for value in spec["objective_lower_bounds"])
    upper = tuple(float(value) for value in spec["objective_upper_bounds"])
    budget = spec.get("charged_evaluation_budget")
    if type(budget) is not int or budget != 2000:
        _fail("Recovery worker budget drifted")
    attempt = spec_path.resolve().parent
    return [
        str(EXPECTED_INTERPRETER_PATH),
        str((project_root / METRIC_RELATIVE).resolve()),
        "--trace",
        str(attempt / "trace.sqlite3"),
        "--lower=" + ",".join(repr(value) for value in lower),
        "--upper=" + ",".join(repr(value) for value in upper),
        "--expected-evaluations",
        str(budget),
        "--output",
        str(attempt / "independent.metric.json"),
    ]


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _execute_frozen_worker_with_timeout_override(
    runner: ModuleType,
    spec_path: Path,
    *,
    project_root: Path,
    helper_sha256: str,
    interpreter_identity: Mapping[str, object],
    verify_boundary: Callable[[], None],
) -> tuple[dict[str, object], dict[str, object]]:
    spec_path = spec_path.resolve()
    spec = runner._load_json_object(spec_path)
    expected_command = _expected_metric_command(
        project_root=project_root, spec_path=spec_path, spec=spec
    )
    original_run = runner.subprocess.run
    guard = MetricTimeoutRunGuard(
        original_run=original_run,
        expected_command=expected_command,
        project_root=project_root,
        verify_boundary=verify_boundary,
    )
    worker_result_path = spec_path.parent / "worker.result.json"
    witness_path = spec_path.parent / OVERRIDE_WITNESS_NAME
    if worker_result_path.exists() or witness_path.exists():
        _fail("Recovery worker output already exists")
    verify_boundary()
    runner.subprocess.run = guard
    try:
        worker_result = runner._worker_run(spec_path)
        guard.assert_exactly_once()
        if runner.subprocess.run is not guard:
            _fail("Frozen runner subprocess binding changed during recovery")
    finally:
        runner.subprocess.run = original_run
    if runner.subprocess.run is not original_run:
        _fail("Frozen runner subprocess binding was not restored")
    verify_boundary()
    worker_result = _exact_keys(
        worker_result,
        {
            "status",
            "row_sha256",
            "diagnostic_sha256",
            "trace_sha256",
            "terminal_receipt_sha256",
            "independent_metric_receipt_sha256",
        },
        label="frozen worker result",
    )
    if worker_result["status"] != "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY":
        _fail("Frozen worker did not return development PASS")
    runner._exclusive_json(worker_result_path, worker_result)
    _fsync_file(worker_result_path)
    worker_result_sha256 = runner._sha256(worker_result_path)
    witness_core: dict[str, object] = {
        "schema": "v21e3r1_metric_timeout_override_witness_v1",
        "status": "PASS_EXACT_ONE_INDEPENDENT_METRIC_TIMEOUT_OVERRIDE",
        "scope": "EXACT_FROZEN_DEVELOPMENT_RECOVERY_ROW_ONLY",
        "recovery_semantics": RECOVERY_SEMANTICS,
        "fresh_full_algorithm_rerun": True,
        "preexisting_failed_trace_reused": False,
        "original_diagnostic_receipt_alone_insufficient": True,
        "row_id": (
            f"{spec['case_id']}__seed-{spec['seed']}__arm-{str(spec['arm_id']).lower()}"
        ),
        "worker_spec_path": spec_path.name,
        "worker_spec_sha256": runner._sha256(spec_path),
        "worker_result_path": worker_result_path.name,
        "worker_result_sha256": worker_result_sha256,
        "independent_metric_command": expected_command,
        "independent_metric_command_sha256": guard.command_sha256,
        "subprocess_call_count": guard.call_count,
        "subprocess_returncode": guard.completed_returncode,
        "original_subprocess_kwargs": {
            "cwd": project_root.as_posix(),
            "text": True,
            "capture_output": True,
            "timeout": ORIGINAL_METRIC_TIMEOUT_SECONDS,
            "check": False,
        },
        "effective_subprocess_kwargs": {
            "cwd": project_root.as_posix(),
            "text": True,
            "capture_output": True,
            "timeout": OPERATIONAL_METRIC_TIMEOUT_SECONDS,
            "check": False,
        },
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "source_snapshot_sha256": EXPECTED_SOURCE_ROOT_SHA256,
        "frozen_runner_sha256": EXPECTED_RUNNER_SHA256,
        "independent_metric_source_sha256": EXPECTED_METRIC_SHA256,
        "helper_sha256": helper_sha256,
        "interpreter_identity": dict(interpreter_identity),
        "implementation_independence": False,
        "algorithm_execution_independence": False,
        "scientific_independence": False,
        **_authority_hold_fields(),
    }
    witness = dict(witness_core)
    witness["receipt_payload_sha256"] = hashlib.sha256(
        _canonical_bytes(witness_core)
    ).hexdigest()
    runner._exclusive_json(witness_path, witness)
    _fsync_file(witness_path)
    loaded_witness = runner._load_json_object(witness_path)
    if (
        loaded_witness != witness
        or runner._sha256(worker_result_path) != worker_result_sha256
    ):
        _fail("Recovery worker result or timeout witness drifted after write")
    verify_boundary()
    return worker_result, witness


def _verify_worker_spec(
    runner: ModuleType,
    row: RecoveryRow,
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[str, str]:
    before = runner._sha256(path)
    loaded = _exact_keys(
        runner._load_json_object(path), set(row.worker_spec), label="recovery worker spec"
    )
    if runner._canonical_json(loaded) != runner._canonical_json(row.worker_spec):
        _fail(f"Recovery worker spec semantic drift: {row.row_id}")
    after = runner._sha256(path)
    if before != after or (expected_sha256 is not None and before != expected_sha256):
        _fail(f"Recovery worker spec raw-byte drift: {row.row_id}")
    return before, hashlib.sha256(_canonical_bytes(row.worker_spec)).hexdigest()


def _validate_recovery_result(
    context: RecoveryContext,
    runner: ModuleType,
    row: RecoveryRow,
    attempt: Path,
    *,
    helper_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    result = _exact_keys(
        runner._load_json_object(attempt / "worker.result.json"),
        {
            "status",
            "row_sha256",
            "diagnostic_sha256",
            "trace_sha256",
            "terminal_receipt_sha256",
            "independent_metric_receipt_sha256",
        },
        label=f"recovery worker result {row.row_id}",
    )
    if result["status"] != "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY":
        _fail(f"Recovery frozen worker did not PASS: {row.row_id}")
    for name, field in (
        ("row.json", "row_sha256"),
        ("diagnostic.json", "diagnostic_sha256"),
        ("trace.sqlite3", "trace_sha256"),
        ("terminal.receipt.json", "terminal_receipt_sha256"),
        ("independent.metric.json", "independent_metric_receipt_sha256"),
    ):
        expected = _exact_sha256(result[field], label=f"worker result {field}")
        artifact = attempt / name
        if artifact.is_symlink() or not artifact.is_file() or runner._sha256(artifact) != expected:
            _fail(f"Recovery worker artifact drifted: {row.row_id}/{name}")
    row_payload = runner._load_json_object(attempt / "row.json")
    for field, expected in (
        ("status", "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY"),
        ("case_id", TARGET_CASE_ID),
        ("family", "MOTSP"),
        ("size", 500),
        ("seed", row.worker_spec["seed"]),
        ("arm_id", row.worker_spec["arm_id"]),
        ("charged_evaluation_budget", 2000),
        ("checkpoint_period", 200),
        ("case_artifact_sha256", row.worker_spec["case_artifact_sha256"]),
        ("source_snapshot_sha256", EXPECTED_SOURCE_ROOT_SHA256),
        ("plan_sha256", EXPECTED_PLAN_SHA256),
        ("selection_entropy_release", "PROHIBITED"),
        ("confirmation_materialization", "PROHIBITED"),
        ("formal_materialization", "PROHIBITED"),
    ):
        if row_payload.get(field) != expected:
            _fail(f"Recovery row payload drifted at {field}: {row.row_id}")
    witness_path = attempt / OVERRIDE_WITNESS_NAME
    witness = runner._load_json_object(witness_path)
    required_witness_keys = {
        "schema",
        "status",
        "scope",
        "recovery_semantics",
        "fresh_full_algorithm_rerun",
        "preexisting_failed_trace_reused",
        "original_diagnostic_receipt_alone_insufficient",
        "row_id",
        "worker_spec_path",
        "worker_spec_sha256",
        "worker_result_path",
        "worker_result_sha256",
        "independent_metric_command",
        "independent_metric_command_sha256",
        "subprocess_call_count",
        "subprocess_returncode",
        "original_subprocess_kwargs",
        "effective_subprocess_kwargs",
        "plan_sha256",
        "source_snapshot_sha256",
        "frozen_runner_sha256",
        "independent_metric_source_sha256",
        "helper_sha256",
        "interpreter_identity",
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
    _exact_keys(witness, required_witness_keys, label=f"timeout witness {row.row_id}")
    witness_core = dict(witness)
    witness_digest = _exact_sha256(
        witness_core.pop("receipt_payload_sha256"), label="timeout witness payload"
    )
    expected_command = _expected_metric_command(
        project_root=context.project_root,
        spec_path=attempt / "worker.spec.json",
        spec=row.worker_spec,
    )
    expected_original_kwargs = {
        "cwd": context.project_root.as_posix(),
        "text": True,
        "capture_output": True,
        "timeout": ORIGINAL_METRIC_TIMEOUT_SECONDS,
        "check": False,
    }
    expected_effective_kwargs = dict(expected_original_kwargs)
    expected_effective_kwargs["timeout"] = OPERATIONAL_METRIC_TIMEOUT_SECONDS
    if (
        witness_digest != hashlib.sha256(_canonical_bytes(witness_core)).hexdigest()
        or witness["schema"] != "v21e3r1_metric_timeout_override_witness_v1"
        or witness["status"]
        != "PASS_EXACT_ONE_INDEPENDENT_METRIC_TIMEOUT_OVERRIDE"
        or witness["recovery_semantics"] != RECOVERY_SEMANTICS
        or witness["fresh_full_algorithm_rerun"] is not True
        or witness["preexisting_failed_trace_reused"] is not False
        or witness["original_diagnostic_receipt_alone_insufficient"] is not True
        or witness["row_id"] != row.row_id
        or witness["worker_spec_sha256"] != runner._sha256(attempt / "worker.spec.json")
        or witness["worker_result_sha256"]
        != runner._sha256(attempt / "worker.result.json")
        or witness["independent_metric_command"] != expected_command
        or witness["independent_metric_command_sha256"]
        != hashlib.sha256(_canonical_bytes(expected_command)).hexdigest()
        or witness["subprocess_call_count"] != 1
        or witness["subprocess_returncode"] != 0
        or witness["original_subprocess_kwargs"] != expected_original_kwargs
        or witness["effective_subprocess_kwargs"] != expected_effective_kwargs
        or witness["plan_sha256"] != EXPECTED_PLAN_SHA256
        or witness["source_snapshot_sha256"] != EXPECTED_SOURCE_ROOT_SHA256
        or witness["frozen_runner_sha256"] != EXPECTED_RUNNER_SHA256
        or witness["independent_metric_source_sha256"] != EXPECTED_METRIC_SHA256
        or witness["helper_sha256"] != helper_sha256
        or witness["implementation_independence"] is not False
        or witness["algorithm_execution_independence"] is not False
        or witness["scientific_independence"] is not False
        or any(witness[field] is not False for field in (
            "runtime_authority",
            "scientific_authority",
            "selection_authority",
            "confirmation_authority",
            "formal_study_authority",
        ))
        or witness["publication_status"] != "IJOC_HOLD"
    ):
        _fail(f"Recovery timeout witness semantic drift: {row.row_id}")
    return result, witness


def _assert_owned_recovery_state(
    context: RecoveryContext,
    runner: ModuleType,
    *,
    owned_attempts: Mapping[str, Path],
    sealed_rows: set[str],
) -> None:
    for row in context.rows:
        row_root = context.output / "attempts" / row.row_id
        marker = context.output / "completed" / f"{row.row_id}.json"
        owned = owned_attempts.get(row.row_id)
        if owned is None:
            if row.ordinal == RECOVERY_ORDINALS[0]:
                if [path.name for path in sorted(row_root.iterdir())] != ["attempt-0001"]:
                    _fail("Ordinal 446 preexisting attempt layout drifted")
            elif row_root.exists():
                _fail(f"Unowned recovery attempt appeared: {row.row_id}")
        else:
            expected = row_root / f"attempt-{row.expected_attempt_number:04d}"
            if owned.resolve() != expected.resolve() or not owned.is_dir():
                _fail(f"Owned recovery attempt path drifted: {row.row_id}")
            expected_names = (
                {"attempt-0001", "attempt-0002"}
                if row.ordinal == RECOVERY_ORDINALS[0]
                else {"attempt-0001"}
            )
            if {path.name for path in row_root.iterdir()} != expected_names:
                _fail(f"Concurrent recovery attempt appeared: {row.row_id}")
        if marker.exists():
            if row.row_id not in sealed_rows or owned is None:
                _fail(f"Non-helper recovery marker appeared: {row.row_id}")
            completed = runner._completed_payload(context.output, row.row_id)
            if (
                completed is None
                or completed.get("row_id") != row.row_id
                or completed.get("attempt_directory")
                != owned.relative_to(context.output).as_posix()
                or completed.get("plan_sha256") != EXPECTED_PLAN_SHA256
            ):
                _fail(f"Helper-owned recovery marker drifted: {row.row_id}")
        elif row.row_id in sealed_rows:
            _fail(f"Helper-sealed recovery marker disappeared: {row.row_id}")


def _cleanup_exact_owned_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_payload: Mapping[str, object],
    runner: ModuleType,
) -> str:
    if path.is_symlink() or not path.is_file():
        return "RETAINED_NOT_REGULAR_OR_MISSING"
    try:
        current_sha = runner._sha256(path)
        current_payload = runner._load_json_object(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return "RETAINED_UNREADABLE_OR_EXTERNALLY_CHANGED"
    if current_sha != expected_sha256 or current_payload != dict(expected_payload):
        return "RETAINED_EXTERNALLY_CHANGED"
    path.unlink()
    return "REMOVED_EXACT_HELPER_OWNED_FALSE_PASS_FILE"


def _run_recovery_child(
    context: RecoveryContext,
    spec_path: Path,
    *,
    helper_sha256: str,
    claim_sha256: str,
    process_guard: ModuleType,
) -> dict[str, object]:
    command = [
        str(EXPECTED_INTERPRETER_PATH),
        str((context.project_root / HELPER_RELATIVE).resolve()),
        "--recovery-worker-spec",
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
    completed, job_witness = _run_in_windows_kill_on_close_job(
        command,
        cwd=context.project_root,
        environment=environment,
        timeout_seconds=OUTER_ROW_TIMEOUT_SECONDS,
        start_gate_line=JOB_START_GATE_LINE,
        terminal_zero_check=lambda: _wait_for_worker_specs_process_zero(
            context,
            process_guard,
            {spec_path.resolve().as_posix()},
            block_all_recovery_processes=False,
        ),
    )
    if completed.returncode != 0:
        _fail(
            "Metric-timeout recovery child failed: " + completed.stderr[-2000:]
        )
    job_witness = _exact_keys(
        job_witness,
        {
            "schema",
            "kill_on_job_close_limit",
            "job_limit_flags",
            "wrapper_pid",
            "job_assignment_verified_before_gate_release",
            "outer_timeout_seconds",
            "outer_timeout_fired",
            "wrapper_returncode",
            "active_processes_after_wrapper",
            "terminal_active_processes",
            "terminal_process_scan",
        },
        label="metric-timeout recovery Job Object witness",
    )
    if (
        job_witness["schema"]
        != "v21e3r1_windows_kill_on_close_job_witness_v1"
        or job_witness["kill_on_job_close_limit"] is not True
        or job_witness["job_limit_flags"]
        != JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        or type(job_witness["wrapper_pid"]) is not int
        or job_witness["job_assignment_verified_before_gate_release"] is not True
        or job_witness["outer_timeout_seconds"] != OUTER_ROW_TIMEOUT_SECONDS
        or job_witness["terminal_active_processes"] != 0
        or job_witness["active_processes_after_wrapper"] != 0
        or job_witness["outer_timeout_fired"] is not False
        or job_witness["wrapper_returncode"] != 0
        or type(job_witness["terminal_process_scan"]) is not dict
    ):
        _fail("Metric-timeout recovery Job Object witness drifted")
    return job_witness


def _validate_worker_claim(
    *,
    output: Path,
    spec_path: Path,
    spec: Mapping[str, object],
    runner: ModuleType,
    helper_sha256: str,
    claim_sha256: str,
) -> dict[str, object]:
    claim_path = output / CLAIM_NAME
    if runner._sha256(claim_path) != _exact_sha256(
        claim_sha256, label="worker claim raw SHA"
    ):
        _fail("Recovery worker claim raw bytes drifted")
    claim = runner._load_json_object(claim_path)
    core = dict(claim)
    digest = _exact_sha256(
        core.pop("claim_payload_sha256"), label="worker claim payload"
    )
    if digest != hashlib.sha256(_canonical_bytes(core)).hexdigest():
        _fail("Recovery worker claim payload digest drifted")
    row_id = (
        f"{spec.get('case_id')}__seed-{spec.get('seed')}__arm-"
        f"{str(spec.get('arm_id')).lower()}"
    )
    entries = claim.get("worker_spec_payload_manifest")
    matches = [
        entry
        for entry in entries
        if type(entry) is dict and entry.get("row_id") == row_id
    ] if type(entries) is list else []
    if (
        claim.get("schema")
        != "v21e3r1_metric_timeout_recovery_helper_instance_claim_v1"
        or claim.get("status")
        != "SEALED_HELPER_INSTANCE_CLAIM_NOT_A_MAIN_RUNNER_LOCK"
        or claim.get("helper_sha256") != helper_sha256
        or claim.get("target_ordinals") != list(RECOVERY_ORDINALS)
        or claim.get("target_row_count") != EXPECTED_RECOVERY_ROWS
        or type(claim.get("process_id")) is not int
        or claim.get("process_id") != os.getppid()
        or len(matches) != 1
        or matches[0].get("worker_spec_payload_sha256")
        != hashlib.sha256(_canonical_bytes(dict(spec))).hexdigest()
    ):
        _fail("Recovery worker claim-to-spec binding drifted")
    expected_attempt = matches[0].get("expected_attempt_number")
    if (
        type(expected_attempt) is not int
        or spec_path.name != "worker.spec.json"
        or spec_path.parent.name != f"attempt-{expected_attempt:04d}"
        or spec_path.parent.parent.name != row_id
    ):
        _fail("Recovery worker attempt layout drifted")
    return claim


def run_recovery_worker(
    spec_path: str | Path,
    *,
    helper_sha256: str,
    claim_sha256: str,
) -> dict[str, object]:
    if threading.active_count() != 1 or threading.current_thread() is not threading.main_thread():
        _fail("Recovery monkeypatch is authorized only in a single-threaded child")
    helper_path = Path(__file__).resolve()
    project_root = helper_path.parents[2]
    output = (project_root / OUTPUT_RELATIVE).resolve()
    _validate_production_paths(project_root, output)
    helper_sha256 = _exact_sha256(helper_sha256, label="worker helper SHA")
    if _sha256(helper_path) != helper_sha256:
        _fail("Recovery helper source drifted in child")
    process_guard = _load_process_guard(project_root)
    interpreter_identity = _validate_interpreter(process_guard)
    runner = _load_frozen_runner(project_root)
    metric_path = (project_root / METRIC_RELATIVE).resolve()
    plan_path = (output / PLAN_NAME).resolve()
    if (
        _sha256(metric_path) != EXPECTED_METRIC_SHA256
        or _sha256(plan_path) != EXPECTED_PLAN_SHA256
        or runner._source_manifest(project_root).get("source_snapshot_sha256")
        != EXPECTED_SOURCE_ROOT_SHA256
        or runner.subprocess is not subprocess
    ):
        _fail("Recovery worker frozen boundary drifted before monkeypatch")
    spec_file = Path(spec_path).resolve()
    try:
        spec_file.relative_to((output / "attempts").resolve())
    except ValueError as error:
        raise RuntimeError("Recovery worker spec escaped attempts root") from error
    spec = _exact_keys(
        runner._load_json_object(spec_file),
        {
            "schema",
            "project_root",
            "case_id",
            "family",
            "size",
            "case_path",
            "case_artifact_sha256",
            "objective_lower_bounds",
            "objective_upper_bounds",
            "reference_directions",
            "seed",
            "arm_id",
            "charged_evaluation_budget",
            "checkpoint_period",
            "source_snapshot_sha256",
            "plan_sha256",
        },
        label="recovery child worker spec",
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
        _fail("Recovery child worker spec semantic contract drifted")
    _validate_worker_claim(
        output=output,
        spec_path=spec_file,
        spec=spec,
        runner=runner,
        helper_sha256=helper_sha256,
        claim_sha256=claim_sha256,
    )
    spec_sha256 = runner._sha256(spec_file)

    def verify_boundary() -> None:
        if (
            _sha256(helper_path) != helper_sha256
            or _sha256(plan_path) != EXPECTED_PLAN_SHA256
            or _sha256((project_root / RUNNER_RELATIVE).resolve())
            != EXPECTED_RUNNER_SHA256
            or _sha256(metric_path) != EXPECTED_METRIC_SHA256
            or _sha256((project_root / PROCESS_GUARD_RELATIVE).resolve())
            != EXPECTED_PROCESS_GUARD_SHA256
            or runner._source_manifest(project_root).get("source_snapshot_sha256")
            != EXPECTED_SOURCE_ROOT_SHA256
            or runner._sha256(spec_file) != spec_sha256
            or runner._sha256(output / CLAIM_NAME) != claim_sha256
            or _validate_interpreter(process_guard) != interpreter_identity
        ):
            _fail("Recovery child boundary drifted around metric subprocess")

    process_guard._assert_no_conflicting_original_processes(output)
    worker_result, witness = _execute_frozen_worker_with_timeout_override(
        runner,
        spec_file,
        project_root=project_root,
        helper_sha256=helper_sha256,
        interpreter_identity=interpreter_identity,
        verify_boundary=verify_boundary,
    )
    process_guard._assert_no_conflicting_original_processes(output)
    if runner._sha256(spec_file) != spec_sha256:
        _fail("Recovery child worker spec drifted at terminal boundary")
    return {
        "worker_result_sha256": runner._sha256(
            spec_file.parent / "worker.result.json"
        ),
        "timeout_witness_sha256": runner._sha256(
            spec_file.parent / OVERRIDE_WITNESS_NAME
        ),
        "worker_status": worker_result["status"],
        "timeout_witness_status": witness["status"],
    }


def _write_descendant_quarantine(
    context: RecoveryContext,
    runner: ModuleType,
    *,
    error: BaseException,
    phase: str,
    helper_sha256: str,
    claim_sha256: str,
    expected_claim: Mapping[str, object],
    owned_attempts: Mapping[str, Path],
) -> dict[str, object]:
    path = context.output / QUARANTINE_NAME
    if path.exists() or path.is_symlink():
        _fail("Metric-timeout recovery descendant quarantine already exists")
    _validate_claim(
        context,
        runner,
        expected_claim=expected_claim,
        expected_sha256=claim_sha256,
    )
    core: dict[str, object] = {
        "schema": "v21e3r1_metric_timeout_recovery_descendant_quarantine_v1",
        "status": "NOT_TERMINAL_DESCENDANT_TERMINATION_UNCONFIRMED",
        "scope": "EXACT_ORDINALS_446_462_FROZEN_DEVELOPMENT_RECOVERY_ONLY",
        "recovery_semantics": RECOVERY_SEMANTICS,
        "failure_phase": phase,
        "exception_type": type(error).__name__,
        "exception_message": str(error)[-2000:],
        "target_ordinals": list(RECOVERY_ORDINALS),
        "target_row_count": EXPECTED_RECOVERY_ROWS,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "source_snapshot_sha256": EXPECTED_SOURCE_ROOT_SHA256,
        "frozen_runner_sha256": EXPECTED_RUNNER_SHA256,
        "independent_metric_source_sha256": EXPECTED_METRIC_SHA256,
        "helper_sha256": helper_sha256,
        "helper_instance_claim_path": CLAIM_NAME,
        "helper_instance_claim_sha256": claim_sha256,
        "owned_attempt_directories_without_terminal_hash_claim": [
            {
                "row_id": row_id,
                "attempt_directory": attempt.relative_to(context.output).as_posix(),
            }
            for row_id, attempt in sorted(owned_attempts.items())
        ],
        "terminal_descendant_state_confirmed": False,
        "durable_terminal_failure": False,
        "attempt_artifact_hashes_authorized": False,
        "automatic_retry_authorized": False,
        "main_runner_resume_authorized": False,
        "manual_process_and_artifact_audit_required": True,
        "aggregate_materialized": False,
        "diagnostic_receipt_materialized": False,
        **_authority_hold_fields(),
    }
    payload = _bound_payload(core, digest_field="quarantine_payload_sha256")
    runner._exclusive_json(path, payload)
    _fsync_file(path)
    _validate_bound_json(
        path,
        runner,
        expected=payload,
        digest_field="quarantine_payload_sha256",
        label="descendant-state quarantine",
    )
    _validate_claim(
        context,
        runner,
        expected_claim=expected_claim,
        expected_sha256=claim_sha256,
    )
    return payload


def _write_durable_failure(
    context: RecoveryContext,
    runner: ModuleType,
    *,
    error: BaseException,
    phase: str,
    helper_sha256: str,
    claim_sha256: str,
    expected_claim: Mapping[str, object],
    terminal_descendant_witness: Mapping[str, object],
    owned_attempts: Mapping[str, Path],
    sealed_entries: Mapping[str, Mapping[str, object]],
    cleanup_events: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failure_path = context.output / FAILURE_NAME
    seal_path = context.output / FAILURE_SEAL_NAME
    if failure_path.exists() or seal_path.exists():
        _fail("Durable metric-timeout recovery failure evidence already exists")
    _validate_claim(
        context,
        runner,
        expected_claim=expected_claim,
        expected_sha256=claim_sha256,
    )
    attempts: list[dict[str, object]] = []
    for row_id, attempt in sorted(owned_attempts.items()):
        files = [path for path in attempt.iterdir() if path.is_file() and not path.is_symlink()]
        attempts.append(
            {
                "row_id": row_id,
                "attempt_directory": attempt.relative_to(context.output).as_posix(),
                "artifacts": list(_file_manifest(context.output, files)),
            }
        )
    core: dict[str, object] = {
        "schema": "v21e3r1_metric_timeout_recovery_failure_receipt_v1",
        "status": "HOLD_METRIC_TIMEOUT_RECOVERY_FAILURE_MANUAL_AUDIT_REQUIRED",
        "scope": "EXACT_ORDINALS_446_462_FROZEN_DEVELOPMENT_RECOVERY_ONLY",
        "recovery_semantics": RECOVERY_SEMANTICS,
        "failure_phase": phase,
        "exception_type": type(error).__name__,
        "exception_message": str(error)[-2000:],
        "target_ordinals": list(RECOVERY_ORDINALS),
        "target_row_count": EXPECTED_RECOVERY_ROWS,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "source_snapshot_sha256": EXPECTED_SOURCE_ROOT_SHA256,
        "frozen_runner_sha256": EXPECTED_RUNNER_SHA256,
        "independent_metric_source_sha256": EXPECTED_METRIC_SHA256,
        "helper_sha256": helper_sha256,
        "helper_instance_claim_path": CLAIM_NAME,
        "helper_instance_claim_sha256": claim_sha256,
        "terminal_descendant_state": dict(terminal_descendant_witness),
        "terminal_descendant_state_confirmed": True,
        "owned_attempts": attempts,
        "validated_completed_markers": [
            dict(sealed_entries[row_id]) for row_id in sorted(sealed_entries)
        ],
        "cleanup_events": [dict(item) for item in cleanup_events],
        "preexisting_failed_attempt_preservation_status": (
            "REQUIRES_MANUAL_REVALIDATION_AFTER_FAILURE"
        ),
        "automatic_retry_authorized": False,
        "main_runner_resume_authorized": False,
        "manual_audit_required": True,
        "aggregate_materialized": False,
        "diagnostic_receipt_materialized": False,
        "implementation_independence": False,
        "algorithm_execution_independence": False,
        "scientific_independence": False,
        **_authority_hold_fields(),
    }
    receipt = _bound_payload(core, digest_field="receipt_payload_sha256")
    runner._exclusive_json(failure_path, receipt)
    _fsync_file(failure_path)
    receipt_sha256 = _validate_bound_json(
        failure_path,
        runner,
        expected=receipt,
        digest_field="receipt_payload_sha256",
        label="durable recovery failure receipt",
    )
    _validate_claim(
        context,
        runner,
        expected_claim=expected_claim,
        expected_sha256=claim_sha256,
    )
    seal_core: dict[str, object] = {
        "schema": "v21e3r1_metric_timeout_recovery_failure_seal_v1",
        "status": "SEALED_DURABLE_FAILURE_RECEIPT",
        "failure_receipt_path": FAILURE_NAME,
        "failure_receipt_sha256": receipt_sha256,
        "failure_receipt_payload_sha256": receipt["receipt_payload_sha256"],
        "helper_instance_claim_sha256": claim_sha256,
        **_authority_hold_fields(),
    }
    seal = _bound_payload(seal_core, digest_field="seal_payload_sha256")
    runner._exclusive_json(seal_path, seal)
    _fsync_file(seal_path)
    _validate_bound_json(
        seal_path,
        runner,
        expected=seal,
        digest_field="seal_payload_sha256",
        label="durable recovery failure seal",
    )
    _validate_claim(
        context,
        runner,
        expected_claim=expected_claim,
        expected_sha256=claim_sha256,
    )
    return receipt


def execute_recovery(
    context: RecoveryContext,
    runner: ModuleType,
    process_guard: ModuleType,
    *,
    interpreter_identity: Mapping[str, object],
    environment_receipt: Mapping[str, object],
) -> dict[str, object]:
    if OUTER_ROW_TIMEOUT_SECONDS <= OPERATIONAL_METRIC_TIMEOUT_SECONDS:
        _fail("Outer row timeout has no positive margin over metric timeout")
    helper_sha256 = _sha256(Path(__file__).resolve())
    _verify_static_boundary(
        context,
        runner,
        process_guard,
        helper_sha256=helper_sha256,
        interpreter_identity=interpreter_identity,
        environment_receipt=environment_receipt,
    )
    _verify_preserved_evidence(context)
    _assert_owned_recovery_state(
        context, runner, owned_attempts={}, sealed_rows=set()
    )
    preclaim_scan = _assert_process_boundary(context, process_guard)
    claim = _claim_payload(
        context,
        helper_sha256=helper_sha256,
        interpreter_identity=interpreter_identity,
        environment_receipt=environment_receipt,
        preclaim_process_scan=preclaim_scan,
    )
    claim_sha256 = hashlib.sha256(_exclusive_json_bytes(claim)).hexdigest()
    claim_path = context.output / CLAIM_NAME
    receipt_path = context.output / RECEIPT_NAME
    seal_path = context.output / RECEIPT_SEAL_NAME
    owned_attempts: dict[str, Path] = {}
    sealed_rows: set[str] = set()
    sealed_entries: dict[str, dict[str, object]] = {}
    cleanup_events: list[dict[str, object]] = []
    running_specs: set[str] = set()
    state_lock = threading.Lock()
    process_lock = threading.Lock()
    stop_event = threading.Event()
    phase = "CLAIM_EXCLUSIVE_CREATE"
    success_receipt: dict[str, object] | None = None
    success_receipt_sha256: str | None = None
    success_receipt_created = False
    success_seal: dict[str, object] | None = None
    success_seal_sha256: str | None = None
    success_seal_created = False
    claim_acquired = False

    def verify_operational_boundary() -> None:
        _verify_static_boundary(
            context,
            runner,
            process_guard,
            helper_sha256=helper_sha256,
            interpreter_identity=interpreter_identity,
            environment_receipt=environment_receipt,
        )
        _validate_claim(
            context,
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

    def run_one(row: RecoveryRow) -> dict[str, object]:
        if stop_event.is_set():
            _fail("Recovery cancelled after another row failed")
        verify_operational_boundary()
        with state_lock:
            _assert_owned_recovery_state(
                context,
                runner,
                owned_attempts=owned_attempts,
                sealed_rows=sealed_rows,
            )
            attempt = runner._next_attempt_directory(context.output, row.row_id).resolve()
            expected_attempt = (
                context.output
                / "attempts"
                / row.row_id
                / f"attempt-{row.expected_attempt_number:04d}"
            ).resolve()
            if attempt != expected_attempt:
                _fail(f"Recovery attempt number drifted: {row.row_id}")
            owned_attempts[row.row_id] = attempt
            spec_path = attempt / "worker.spec.json"
            runner._exclusive_json(spec_path, row.worker_spec)
            _fsync_file(spec_path)
            spec_sha256, spec_payload_sha256 = _verify_worker_spec(
                runner, row, spec_path
            )
            running_specs.add(spec_path.resolve().as_posix())
        verify_operational_boundary()
        child_error: BaseException | None = None
        job_witness: dict[str, object] | None = None
        try:
            job_witness = _run_recovery_child(
                context,
                spec_path,
                helper_sha256=helper_sha256,
                claim_sha256=claim_sha256,
                process_guard=process_guard,
            )
        except BaseException as error:
            child_error = error
        try:
            _verify_worker_spec(
                runner, row, spec_path, expected_sha256=spec_sha256
            )
            verify_operational_boundary()
        finally:
            with state_lock:
                running_specs.discard(spec_path.resolve().as_posix())
        if child_error is not None:
            raise child_error
        if type(job_witness) is not dict:
            _fail("Recovery child did not return a Job Object witness")
        result, witness = _validate_recovery_result(
            context,
            runner,
            row,
            attempt,
            helper_sha256=helper_sha256,
        )
        if stop_event.is_set():
            _fail("Recovery cancelled before marker materialization")
        verify_operational_boundary()
        _verify_worker_spec(runner, row, spec_path, expected_sha256=spec_sha256)
        with state_lock:
            _assert_owned_recovery_state(
                context,
                runner,
                owned_attempts=owned_attempts,
                sealed_rows=sealed_rows,
            )
            completed = {
                **result,
                "row_id": row.row_id,
                "attempt_directory": attempt.relative_to(context.output).as_posix(),
                "plan_sha256": EXPECTED_PLAN_SHA256,
            }
            marker_path = context.output / "completed" / f"{row.row_id}.json"
            marker_sha256 = hashlib.sha256(_exclusive_json_bytes(completed)).hexdigest()
            runner._exclusive_json(marker_path, completed)
            _fsync_file(marker_path)
            try:
                verified = runner._completed_payload(context.output, row.row_id)
                if verified != completed:
                    _fail(f"Frozen completed verifier rejected recovery row: {row.row_id}")
            except BaseException:
                cleanup_status = _cleanup_exact_owned_file(
                    marker_path,
                    expected_sha256=marker_sha256,
                    expected_payload=completed,
                    runner=runner,
                )
                cleanup_events.append(
                    {
                        "row_id": row.row_id,
                        "path": marker_path.relative_to(context.output).as_posix(),
                        "cleanup_status": cleanup_status,
                    }
                )
                raise
            sealed_rows.add(row.row_id)
            entry = {
                "ordinal": row.ordinal,
                "row_id": row.row_id,
                "attempt_directory": attempt.relative_to(context.output).as_posix(),
                "worker_spec_sha256": spec_sha256,
                "worker_spec_payload_sha256": spec_payload_sha256,
                "worker_result_sha256": runner._sha256(
                    attempt / "worker.result.json"
                ),
                "timeout_witness_sha256": runner._sha256(
                    attempt / OVERRIDE_WITNESS_NAME
                ),
                "timeout_witness_payload_sha256": witness[
                    "receipt_payload_sha256"
                ],
                "windows_job_object_witness": dict(job_witness),
                "completed_marker_path": marker_path.relative_to(
                    context.output
                ).as_posix(),
                "completed_marker_sha256": marker_sha256,
            }
            sealed_entries[row.row_id] = entry
            _assert_owned_recovery_state(
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
                "Recovery helper-instance claim already exists; claim race loser "
                "performed zero recovery writes"
            ) from error
        _fsync_file(claim_path)
        _validate_claim(
            context,
            runner,
            expected_claim=claim,
            expected_sha256=claim_sha256,
        )
        claim_acquired = True
        phase = "POST_CLAIM_PROCESS_AND_BOUNDARY"
        verify_operational_boundary()
        phase = "PARALLEL_RECOVERY_ROWS"
        first_error: BaseException | None = None
        futures: list[Future[dict[str, object]]] = []
        with ThreadPoolExecutor(
            max_workers=JOBS, thread_name_prefix="v21e3r1-metric-timeout-recovery"
        ) as executor:
            futures = [executor.submit(run_one, row) for row in context.rows]
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
        phase = "FINAL_17_ROW_VERIFICATION"
        ordered_entries = [sealed_entries[row.row_id] for row in context.rows]
        if len(ordered_entries) != EXPECTED_RECOVERY_ROWS:
            _fail("Recovery did not seal exact 17 rows")
        verify_operational_boundary()
        _verify_preserved_evidence(context)
        with state_lock:
            _assert_owned_recovery_state(
                context,
                runner,
                owned_attempts=owned_attempts,
                sealed_rows=sealed_rows,
            )
        completed_names = {
            path.name
            for path in (context.output / "completed").iterdir()
            if path.is_file() and not path.is_symlink()
        }
        if completed_names != {f"{row_id}.json" for row_id in context.all_row_ids}:
            _fail("Recovery final completed-marker set is not exact504")
        if (context.output / "diagnostic.aggregate.json").exists() or (
            context.output / "diagnostic.receipt.json"
        ).exists():
            _fail("Recovery must not materialize aggregate or diagnostic receipt")
        for row, entry in zip(context.rows, ordered_entries, strict=True):
            attempt = context.output / str(entry["attempt_directory"])
            _verify_worker_spec(
                runner,
                row,
                attempt / "worker.spec.json",
                expected_sha256=str(entry["worker_spec_sha256"]),
            )
            _validate_recovery_result(
                context,
                runner,
                row,
                attempt,
                helper_sha256=helper_sha256,
            )
            marker = context.output / str(entry["completed_marker_path"])
            if runner._sha256(marker) != entry["completed_marker_sha256"]:
                _fail(f"Recovery marker drifted before receipt: {row.row_id}")
            runner._completed_payload(context.output, row.row_id)
        _assert_failure_evidence_absent(context)
        phase = "SUCCESS_RECEIPT_MATERIALIZATION"
        core: dict[str, object] = {
            "schema": "v21e3r1_metric_timeout_recovery_receipt_v1",
            "status": "PASS_EXACT_17_OPERATIONAL_TIMEOUT_RECOVERY_ONLY",
            "scope": "EXACT_ORDINALS_446_462_FROZEN_DEVELOPMENT_RECOVERY_ONLY",
            "recovery_semantics": RECOVERY_SEMANTICS,
            "original_diagnostic_receipt_alone_insufficient": True,
            "target_case_id": TARGET_CASE_ID,
            "target_ordinals": list(RECOVERY_ORDINALS),
            "target_row_count": EXPECTED_RECOVERY_ROWS,
            "jobs": JOBS,
            "original_metric_timeout_seconds": ORIGINAL_METRIC_TIMEOUT_SECONDS,
            "operational_metric_timeout_seconds": OPERATIONAL_METRIC_TIMEOUT_SECONDS,
            "outer_row_timeout_seconds": OUTER_ROW_TIMEOUT_SECONDS,
            "outer_timeout_margin_seconds": (
                OUTER_ROW_TIMEOUT_SECONDS - OPERATIONAL_METRIC_TIMEOUT_SECONDS
            ),
            "fresh_full_algorithm_rerun_count": EXPECTED_RECOVERY_ROWS,
            "failed_trace_reuse": False,
            "metric_only_replay": False,
            "preexisting_failed_attempt_count": 1,
            "preexisting_failed_attempt_preserved": True,
            "preexisting_failed_attempt_manifest": list(
                context.preexisting_failed_attempt_manifest
            ),
            "plan_sha256": EXPECTED_PLAN_SHA256,
            "source_snapshot_sha256": EXPECTED_SOURCE_ROOT_SHA256,
            "frozen_runner_sha256": EXPECTED_RUNNER_SHA256,
            "independent_metric_source_sha256": EXPECTED_METRIC_SHA256,
            "process_guard_sha256": EXPECTED_PROCESS_GUARD_SHA256,
            "helper_sha256": helper_sha256,
            "helper_instance_claim_path": CLAIM_NAME,
            "helper_instance_claim_sha256": claim_sha256,
            "upstream_scheduling_receipt_sha256": UPSTREAM_SCHEDULING_RECEIPT_SHA256,
            "upstream_scheduling_seal_sha256": UPSTREAM_SCHEDULING_SEAL_SHA256,
            "interpreter_identity": dict(interpreter_identity),
            "completed_rows": ordered_entries,
            "final_completed_marker_count": EXPECTED_FULL_ROWS,
            "aggregate_materialized": False,
            "diagnostic_receipt_materialized": False,
            "original_runner_resume_required": True,
            "original_runner_resume_after_recovery_success_only": True,
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
        _fsync_file(receipt_path)
        actual_receipt_sha256 = _validate_bound_json(
            receipt_path,
            runner,
            expected=receipt,
            digest_field="receipt_payload_sha256",
            label="metric-timeout recovery success receipt",
        )
        if actual_receipt_sha256 != success_receipt_sha256:
            _fail("Recovery success receipt raw bytes drifted")
        seal_core: dict[str, object] = {
            "schema": "v21e3r1_metric_timeout_recovery_success_seal_v1",
            "status": "SEALED_SUCCESS_RECEIPT_FILE_DIGEST",
            "receipt_path": RECEIPT_NAME,
            "receipt_sha256": actual_receipt_sha256,
            "receipt_payload_sha256": receipt["receipt_payload_sha256"],
            "helper_instance_claim_sha256": claim_sha256,
            **_authority_hold_fields(),
        }
        seal = _bound_payload(seal_core, digest_field="seal_payload_sha256")
        success_seal = seal
        success_seal_sha256 = hashlib.sha256(_exclusive_json_bytes(seal)).hexdigest()
        runner._exclusive_json(seal_path, seal)
        success_seal_created = True
        _fsync_file(seal_path)
        actual_seal_sha256 = _validate_bound_json(
            seal_path,
            runner,
            expected=seal,
            digest_field="seal_payload_sha256",
            label="metric-timeout recovery success seal",
        )
        if (
            actual_seal_sha256 != success_seal_sha256
            or runner._sha256(receipt_path) != actual_receipt_sha256
        ):
            _fail("Recovery success receipt/seal terminal digest drifted")
        _assert_failure_evidence_absent(context)
        phase = "TERMINAL_POSTWRITE_REHASH"
        verify_operational_boundary()
        _verify_preserved_evidence(context)
        with state_lock:
            _assert_owned_recovery_state(
                context,
                runner,
                owned_attempts=owned_attempts,
                sealed_rows=sealed_rows,
            )
        for row, entry in zip(context.rows, ordered_entries, strict=True):
            attempt = context.output / str(entry["attempt_directory"])
            _verify_worker_spec(
                runner,
                row,
                attempt / "worker.spec.json",
                expected_sha256=str(entry["worker_spec_sha256"]),
            )
            if (
                runner._sha256(attempt / "worker.result.json")
                != entry["worker_result_sha256"]
                or runner._sha256(attempt / OVERRIDE_WITNESS_NAME)
                != entry["timeout_witness_sha256"]
                or runner._sha256(
                    context.output / str(entry["completed_marker_path"])
                )
                != entry["completed_marker_sha256"]
            ):
                _fail(f"Recovery row evidence drifted after receipt: {row.row_id}")
            runner._completed_payload(context.output, row.row_id)
        if (
            runner._sha256(receipt_path) != actual_receipt_sha256
            or runner._sha256(seal_path) != actual_seal_sha256
            or (context.output / "diagnostic.aggregate.json").exists()
            or (context.output / "diagnostic.receipt.json").exists()
        ):
            _fail("Recovery terminal receipt/evidence boundary drifted")
        _assert_failure_evidence_absent(context)
        return receipt
    except BaseException as error:
        for path, payload, expected_sha, created in (
            (
                seal_path,
                success_seal,
                success_seal_sha256,
                success_seal_created,
            ),
            (
                receipt_path,
                success_receipt,
                success_receipt_sha256,
                success_receipt_created,
            ),
        ):
            if created and path.is_file():
                cleanup_status = (
                    "RETAINED_NO_HELPER_OWNERSHIP_WITNESS"
                    if payload is None or expected_sha is None
                    else _cleanup_exact_owned_file(
                        path,
                        expected_sha256=expected_sha,
                        expected_payload=payload,
                        runner=runner,
                    )
                )
                cleanup_events.append(
                    {"row_id": "__success_file__", "path": path.name, "cleanup_status": cleanup_status}
                )
        if not claim_acquired:
            raise RuntimeError(
                "Metric-timeout recovery stopped before exact claim acquisition; "
                "no owned failure receipt was materialized: " + str(error)
            ) from error
        if not _claim_is_exact_owned(
            context,
            runner,
            expected_claim=claim,
            expected_sha256=claim_sha256,
        ):
            raise RuntimeError(
                "Metric-timeout recovery claim changed after acquisition; onsite "
                "evidence was retained and no owned failure receipt was materialized"
            ) from error
        terminal_error: BaseException | None = (
            error if isinstance(error, DescendantTerminationUnconfirmed) else None
        )
        terminal_descendant_witness: dict[str, object] | None = None
        if terminal_error is None:
            try:
                terminal_descendant_witness = _wait_for_worker_specs_process_zero(
                    context,
                    process_guard,
                    {
                        (attempt / "worker.spec.json").resolve().as_posix()
                        for attempt in owned_attempts.values()
                    },
                    block_all_recovery_processes=True,
                )
            except BaseException as descendant_error:
                terminal_error = DescendantTerminationUnconfirmed(
                    "Recovery exception cleanup could not prove all wrapper/metric "
                    "descendants reached zero: " + str(descendant_error)
                )
        if terminal_error is not None:
            _write_descendant_quarantine(
                context,
                runner,
                error=terminal_error,
                phase=phase,
                helper_sha256=helper_sha256,
                claim_sha256=claim_sha256,
                expected_claim=claim,
                owned_attempts=owned_attempts,
            )
            raise RuntimeError(
                "Metric-timeout recovery is NOT_TERMINAL; descendant-state "
                "quarantine was retained and every resume path is prohibited"
            ) from terminal_error
        _write_durable_failure(
            context,
            runner,
            error=error,
            phase=phase,
            helper_sha256=helper_sha256,
            claim_sha256=claim_sha256,
            expected_claim=claim,
            terminal_descendant_witness=terminal_descendant_witness,
            owned_attempts=owned_attempts,
            sealed_entries=sealed_entries,
            cleanup_events=cleanup_events,
        )
        raise RuntimeError(
            "Metric-timeout recovery stopped fail-closed; claim, attempts, markers, "
            "and durable failure evidence were retained; automatic resume is prohibited"
        ) from error


def preflight(
    project_root: str | Path,
    output_directory: str | Path,
) -> tuple[
    RecoveryContext,
    ModuleType,
    ModuleType,
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    root = Path(project_root).resolve()
    output = Path(output_directory).resolve()
    _validate_production_paths(root, output)
    process_guard = _load_process_guard(root)
    interpreter_identity = _validate_interpreter(process_guard)
    environment_receipt = process_guard._execution_environment_receipt(
        root, interpreter_identity
    )
    runner = _load_frozen_runner(root)
    context = _validate_plan_and_build_context(root, output, runner)
    helper_sha256 = _sha256(Path(__file__).resolve())
    _verify_static_boundary(
        context,
        runner,
        process_guard,
        helper_sha256=helper_sha256,
        interpreter_identity=interpreter_identity,
        environment_receipt=environment_receipt,
    )
    _verify_preserved_evidence(context)
    _assert_owned_recovery_state(
        context, runner, owned_attempts={}, sealed_rows=set()
    )
    process_scan = _assert_process_boundary(context, process_guard)
    return (
        context,
        runner,
        process_guard,
        interpreter_identity,
        environment_receipt,
        process_scan,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-directory")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--recovery-worker-spec", help=argparse.SUPPRESS)
    parser.add_argument("--helper-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--claim-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--job-start-gate", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.recovery_worker_spec is not None:
            if (
                args.output_directory is not None
                or args.preflight_only
                or args.helper_sha256 is None
                or args.claim_sha256 is None
                or args.job_start_gate != JOB_START_GATE_ARGUMENT
            ):
                _fail("Malformed internal recovery-worker invocation")
            if sys.stdin.readline() != JOB_START_GATE_LINE:
                _fail("Recovery worker Job Object start gate drifted")
            result = run_recovery_worker(
                args.recovery_worker_spec,
                helper_sha256=args.helper_sha256,
                claim_sha256=args.claim_sha256,
            )
            print(json.dumps(result, sort_keys=True), flush=True)
            return 0
        if (
            args.output_directory is None
            or args.helper_sha256 is not None
            or args.claim_sha256 is not None
            or args.job_start_gate is not None
        ):
            _fail("--output-directory is required for recovery parent mode")
        (
            context,
            runner,
            process_guard,
            interpreter_identity,
            environment_receipt,
            process_scan,
        ) = preflight(args.project_root, args.output_directory)
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "status": "PASS_PREFLIGHT_EXACT_17_RECOVERY_READY_NO_WRITES",
                        "target_ordinals": list(RECOVERY_ORDINALS),
                        "target_row_count": EXPECTED_RECOVERY_ROWS,
                        "plan_sha256": EXPECTED_PLAN_SHA256,
                        "source_snapshot_sha256": EXPECTED_SOURCE_ROOT_SHA256,
                        "frozen_runner_sha256": EXPECTED_RUNNER_SHA256,
                        "independent_metric_source_sha256": EXPECTED_METRIC_SHA256,
                        "process_scan_payload_sha256": process_scan[
                            "scan_payload_sha256"
                        ],
                        "writes_performed": False,
                        **_authority_hold_fields(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0
        result = execute_recovery(
            context,
            runner,
            process_guard,
            interpreter_identity=interpreter_identity,
            environment_receipt=environment_receipt,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        print(f"FAIL_CLOSED: {type(error).__name__}: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
