from __future__ import annotations

"""Externally schedule the final disjoint 42 rows of the frozen V7 diagnostic.

This helper owns scheduling only.  Every row is executed and verified by the
already-frozen V21e3r1 diagnostic runner.  It does not change algorithms,
metrics, analysis, phase authority, or the original runner.
"""

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import locale
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
from types import ModuleType
from typing import Mapping, NoReturn, Sequence


EXPECTED_PLAN_SHA256 = (
    "4408d10944cb6511e99ff0bd95ded256b9c230b91d8806a7bd5b962f10622886"
)
EXPECTED_SOURCE_ROOT_SHA256 = (
    "218bc398f04722d1da305928a9c206641f9b43d74b2afbc46c29ba1f08d6639b"
)
EXPECTED_RUNNER_SHA256 = (
    "70a45fd0e62d870702b29a92b66b38eef6c04952152d5defae89c115c6d85b7b"
)
EXPECTED_INTERPRETER_PATH = Path(r"C:\miniconda3\python.exe")
EXPECTED_INTERPRETER_SHA256 = (
    "f77193cf0405ab440c39324bdb2f8864596321c1df888adbbe357f3d760f4716"
)
EXPECTED_INTERPRETER_BYTES = 105288
EXPECTED_PYTHON_VERSION = (
    "3.13.12 | packaged by Anaconda, Inc. | (main, Feb 24 2026, 16:05:56) "
    "[MSC v.1942 64 bit (AMD64)]"
)
EXPECTED_VERSION_INFO = [3, 13, 12, "final", 0]
EXPECTED_FULL_ROWS = 504
EXPECTED_TARGET_ROWS = 42
JOBS = 4
TARGET_CASE_ID = "v21e3-motsp-development-n500-s01"
RUNNER_RELATIVE = Path(
    "ijoc_submission_v21e3r1/scripts/run_v21e3r1_development_diagnostics.py"
)
HELPER_RELATIVE = Path(
    "artifacts/v21e3r1_v8_work_20260822/"
    "run_frozen_diagnostic_disjoint_case_parallel.py"
)
OUTPUT_RELATIVE = Path(
    "outputs/v21e3r1_v7_exposed_development_diagnostics_20260823"
)
PLAN_NAME = "diagnostic.plan.json"
HANDOFF_NAME = f"external-scheduling.{TARGET_CASE_ID}.main-driver-handoff.receipt.json"
CLAIM_NAME = f"external-scheduling.{TARGET_CASE_ID}.helper-instance.claim.json"
FAILURE_NAME = f"external-scheduling.{TARGET_CASE_ID}.failure.receipt.json"
RECEIPT_NAME = f"external-scheduling.{TARGET_CASE_ID}.receipt.json"
RECEIPT_SEAL_NAME = f"external-scheduling.{TARGET_CASE_ID}.receipt.seal.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
UTC_RE = re.compile(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")


@dataclass(frozen=True)
class RowWork:
    ordinal: int
    row_id: str
    worker_spec: dict[str, object]


@dataclass(frozen=True)
class FrozenContext:
    project_root: Path
    output: Path
    plan_path: Path
    plan_sha256: str
    source_snapshot_sha256: str
    runner_path: Path
    runner_sha256: str
    row_timeout_seconds: int
    existing_completed_marker_count: int
    all_row_ids: tuple[str, ...]
    rows: tuple[RowWork, ...]


def _fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(
    value: object, expected: set[str], *, label: str
) -> dict[str, object]:
    if type(value) is not dict:
        _fail(f"{label} must be an exact JSON object")
    raw = dict(value)
    if set(raw) != expected:
        _fail(
            f"{label} key set drifted: missing={sorted(expected - set(raw))}, "
            f"extra={sorted(set(raw) - expected)}"
        )
    return raw


def _exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an exact integer >= {minimum}")
    return value


def _exact_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _exact_interpreter_identity() -> dict[str, object]:
    expected = EXPECTED_INTERPRETER_PATH.resolve()
    observed = Path(sys.executable).resolve()
    try:
        same_file = os.path.samefile(observed, expected)
    except OSError:
        same_file = False
    if (
        not same_file
        or os.path.normcase(str(observed)) != os.path.normcase(str(expected))
    ):
        _fail(
            "Helper must use the exact historical main-job interpreter "
            f"{expected}; observed {observed}"
        )
    if (
        not observed.is_file()
        or observed.stat().st_size != EXPECTED_INTERPRETER_BYTES
        or _raw_sha256(observed) != EXPECTED_INTERPRETER_SHA256
        or platform.python_implementation() != "CPython"
        or platform.architecture()[0] != "64bit"
        or sys.version != EXPECTED_PYTHON_VERSION
        or list(sys.version_info[:5]) != EXPECTED_VERSION_INFO
        or Path(sys.prefix).resolve() != Path(r"C:\miniconda3").resolve()
        or Path(sys.base_prefix).resolve() != Path(r"C:\miniconda3").resolve()
    ):
        _fail("Historical main-job interpreter bytes/version drifted")
    core: dict[str, object] = {
        "schema": "v21e3r1_external_scheduling_interpreter_identity_v1",
        "status": "PASS_EXACT_HISTORICAL_MAIN_JOB_INTERPRETER",
        "resolved_path": expected.as_posix(),
        "bytes": EXPECTED_INTERPRETER_BYTES,
        "sha256": EXPECTED_INTERPRETER_SHA256,
        "implementation": "CPython",
        "architecture_bits": 64,
        "prefix": Path(sys.prefix).resolve().as_posix(),
        "base_prefix": Path(sys.base_prefix).resolve().as_posix(),
        "python_version": EXPECTED_PYTHON_VERSION,
        "version_info": list(EXPECTED_VERSION_INFO),
    }
    identity = dict(core)
    identity["identity_payload_sha256"] = hashlib.sha256(
        runner_canonical_bytes(core)
    ).hexdigest()
    return identity


def _execution_environment_receipt(
    project_root: Path, interpreter_identity: Mapping[str, object]
) -> dict[str, object]:
    pythonpath_input = os.environ.get("PYTHONPATH")
    effective_worker_pythonpath = (
        str(project_root)
        if not pythonpath_input
        else str(project_root) + os.pathsep + pythonpath_input
    )
    core: dict[str, object] = {
        "schema": "v21e3r1_external_scheduling_environment_receipt_v1",
        "status": "PASS_MINIMAL_WORKER_ENVIRONMENT_RECORDED",
        "interpreter_identity_sha256": interpreter_identity[
            "identity_payload_sha256"
        ],
        "project_root": project_root.as_posix(),
        "worker_working_directory": project_root.as_posix(),
        "pythonpath_input": pythonpath_input,
        "effective_worker_pythonpath": effective_worker_pythonpath,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "utf8_mode": int(sys.flags.utf8_mode),
        "isolated_mode": int(sys.flags.isolated),
        "filesystem_encoding": sys.getfilesystemencoding(),
        "preferred_encoding": locale.getpreferredencoding(False),
        "os_name": os.name,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
    }
    receipt = dict(core)
    receipt["environment_payload_sha256"] = hashlib.sha256(
        runner_canonical_bytes(core)
    ).hexdigest()
    return receipt


def _option_value(argv: Sequence[str], option: str) -> str | None:
    matches: list[str] = []
    for index, value in enumerate(argv):
        if value == option:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                _fail(f"Malformed runner option: {option}")
            matches.append(argv[index + 1])
        prefix = option + "="
        if value.startswith(prefix):
            option_value = value[len(prefix) :]
            if not option_value:
                _fail(f"Malformed runner option: {option}")
            matches.append(option_value)
    if len(matches) > 1:
        _fail(f"Duplicate runner option: {option}")
    return matches[0] if matches else None


def _windows_command_line_to_argv(command_line: str) -> tuple[str, ...]:
    if os.name != "nt":
        _fail("Process command-line parsing is only authorized on Windows")
    import ctypes

    argc = ctypes.c_int()
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    shell32.CommandLineToArgvW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    argv_pointer = shell32.CommandLineToArgvW(command_line, ctypes.byref(argc))
    if not argv_pointer:
        _fail("Windows CommandLineToArgvW failed")
    try:
        return tuple(argv_pointer[index] for index in range(argc.value))
    finally:
        kernel32.LocalFree(argv_pointer)


def _normalized_path_text(value: str) -> str:
    return value.replace("/", "\\").rstrip("\\").casefold()


def _argument_targets_output(
    value: str | None,
    output: Path,
    *,
    process_cwd: Path | None,
) -> bool | None:
    if type(value) is not str or not value:
        return None
    argument_path = Path(value)
    if not argument_path.is_absolute():
        if process_cwd is None or not process_cwd.is_absolute():
            return None
        argument_path = process_cwd / argument_path
    try:
        normalized = _normalized_path_text(str(argument_path.resolve()))
        expected = _normalized_path_text(str(output.resolve()))
    except OSError:
        return None
    return normalized == expected


def _argument_is_frozen_runner(value: str) -> bool:
    normalized = _normalized_path_text(value)
    relative = _normalized_path_text(RUNNER_RELATIVE.as_posix())
    return normalized.endswith("\\" + relative) or normalized == relative


def _argument_is_frozen_runner_module(value: str) -> bool:
    normalized = value.strip().casefold()
    module_basename = RUNNER_RELATIVE.stem.casefold()
    return normalized == module_basename or normalized.endswith(
        "." + module_basename
    )


def _frozen_runner_module_arguments(argv: Sequence[str]) -> tuple[str, ...]:
    matches: list[str] = []
    for index, value in enumerate(argv):
        candidate: str | None = None
        if value == "-m" and index + 1 < len(argv):
            candidate = argv[index + 1]
        elif value.startswith("-m="):
            candidate = value[3:]
        elif value.startswith("-m") and len(value) > 2:
            candidate = value[2:]
        if candidate is not None and _argument_is_frozen_runner_module(candidate):
            matches.append(candidate)
    if len(matches) > 1:
        _fail("Duplicate frozen runner module invocation")
    return tuple(matches)


def _classify_runner_command(
    command_line: str,
    output: Path,
    *,
    process_cwd: Path | None = None,
) -> tuple[str, str | None] | None:
    argv = _windows_command_line_to_argv(command_line)
    path_form = any(_argument_is_frozen_runner(value) for value in argv)
    module_forms = _frozen_runner_module_arguments(argv)
    if not path_form and not module_forms:
        return None
    worker_spec = _option_value(argv, "--worker-spec")
    output_argument = _option_value(argv, "--output-directory")
    if worker_spec is not None and output_argument is not None:
        _fail("Runner command line mixes driver and worker modes")
    if worker_spec is not None:
        if not Path(worker_spec).is_absolute():
            return ("unknown", worker_spec)
        try:
            worker_path = Path(worker_spec).resolve()
            worker_path.relative_to((output / "attempts").resolve())
        except (OSError, ValueError):
            return ("unknown", worker_spec)
        return ("worker", worker_path.as_posix())
    targets_output = _argument_targets_output(
        output_argument, output, process_cwd=process_cwd
    )
    if targets_output is True:
        return ("driver", None)
    if targets_output is None:
        return ("unknown", None)
    if _normalized_path_text(str(output)) in _normalized_path_text(command_line):
        return ("unknown", None)
    return None


def _scan_relevant_processes(output: Path) -> dict[str, object]:
    if os.name != "nt":
        _fail("Exact original-process scan is only authorized on Windows")
    powershell = Path(
        os.environ.get("SystemRoot", r"C:\Windows")
    ) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    if not powershell.is_file():
        _fail("Windows PowerShell process scanner is unavailable")
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
        _fail(
            "Win32_Process scan failed: "
            + completed.stderr.strip()[-1000:]
        )
    try:
        raw_processes = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Win32_Process scan returned invalid JSON") from error
    if type(raw_processes) is dict:
        raw_processes = [raw_processes]
    if type(raw_processes) is not list:
        _fail("Win32_Process scan did not return an array")
    relevant: dict[str, list[dict[str, object]]] = {
        "driver_processes": [],
        "worker_processes": [],
        "unknown_runner_processes": [],
    }
    saw_current_pid = False
    for index, item in enumerate(raw_processes):
        if type(item) is not dict:
            _fail(f"Win32_Process row {index} is not an object")
        pid = item.get("ProcessId")
        parent_pid = item.get("ParentProcessId")
        command_line = item.get("CommandLine")
        executable = item.get("ExecutablePath")
        if type(pid) is not int or type(parent_pid) is not int:
            _fail(f"Win32_Process row {index} has invalid PID types")
        if pid == os.getpid():
            saw_current_pid = True
        if command_line is None:
            continue
        if type(command_line) is not str or (
            executable is not None and type(executable) is not str
        ):
            _fail(f"Win32_Process row {index} has invalid command fields")
        classification = _classify_runner_command(command_line, output)
        if classification is None:
            continue
        kind, worker_spec = classification
        record: dict[str, object] = {
            "pid": pid,
            "parent_pid": parent_pid,
            "executable_path": (
                None if executable is None else Path(executable).as_posix()
            ),
            "command_line_sha256": hashlib.sha256(
                command_line.encode("utf-8")
            ).hexdigest(),
        }
        if worker_spec is not None:
            record["worker_spec_path"] = worker_spec
        bucket = (
            "unknown_runner_processes" if kind == "unknown" else f"{kind}_processes"
        )
        relevant[bucket].append(record)
    if not saw_current_pid:
        _fail("Win32_Process scan did not contain the current helper process")
    for values in relevant.values():
        values.sort(key=lambda item: int(item["pid"]))
    core: dict[str, object] = {
        "schema": "v21e3r1_original_diagnostic_process_scan_v1",
        "method": "WIN32_PROCESS_CIM_COMMAND_LINE_AND_PARENT_PID",
        "output_root": output.as_posix(),
        **relevant,
    }
    scan = dict(core)
    scan["scan_payload_sha256"] = hashlib.sha256(
        runner_canonical_bytes(core)
    ).hexdigest()
    return scan


def _assert_no_conflicting_original_processes(
    output: Path,
    *,
    allowed_helper_worker_specs: set[str] | None = None,
) -> dict[str, object]:
    scan = _scan_relevant_processes(output)
    allowed = {
        Path(value).resolve().as_posix()
        for value in (allowed_helper_worker_specs or set())
    }
    if scan["driver_processes"] or scan["unknown_runner_processes"]:
        _fail("Active original main driver or unclassified runner targets this output root")
    conflicting_workers = []
    for worker in scan["worker_processes"]:
        if (
            worker.get("parent_pid") != os.getpid()
            or worker.get("worker_spec_path") not in allowed
            or _normalized_path_text(str(worker.get("executable_path")))
            != _normalized_path_text(EXPECTED_INTERPRETER_PATH.as_posix())
        ):
            conflicting_workers.append(worker)
    if conflicting_workers:
        _fail("Active original worker not owned by this helper targets this output root")
    return scan


def _load_frozen_runner(project_root: Path) -> ModuleType:
    runner_path = (project_root / RUNNER_RELATIVE).resolve()
    try:
        runner_path.relative_to(project_root)
    except ValueError as error:
        raise RuntimeError("Frozen runner escaped the project root") from error
    if not runner_path.is_file() or _raw_sha256(runner_path) != EXPECTED_RUNNER_SHA256:
        _fail("Frozen diagnostic runner bytes drifted")
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module_name = "_v21e3r1_frozen_development_diagnostic_runner"
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    if spec is None or spec.loader is None:
        _fail("Could not load the frozen diagnostic runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    if _raw_sha256(Path(str(module.__file__)).resolve()) != EXPECTED_RUNNER_SHA256:
        _fail("Frozen diagnostic runner changed during import")
    return module


def _validate_production_paths(project_root: Path, output: Path) -> None:
    expected_helper = (project_root / HELPER_RELATIVE).resolve()
    if Path(__file__).resolve() != expected_helper:
        _fail("Scheduling helper is not running from its fixed repository path")
    expected_output = (project_root / OUTPUT_RELATIVE).resolve()
    if output != expected_output:
        _fail("Output directory is not the fixed frozen exact504 evidence directory")
    try:
        output.relative_to(project_root)
    except ValueError as error:
        raise RuntimeError("Output directory escaped the project root") from error


def _validate_plan_and_build_rows(
    project_root: Path,
    output: Path,
    runner: ModuleType,
) -> FrozenContext:
    plan_path = (output / PLAN_NAME).resolve()
    if not plan_path.is_file():
        _fail("Frozen diagnostic plan is missing")
    plan_sha256 = _raw_sha256(plan_path)
    if plan_sha256 != EXPECTED_PLAN_SHA256:
        _fail("Frozen diagnostic plan SHA-256 drifted")
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
        or plan["row_timeout_seconds"] != 1800
        or plan["selection_entropy_release"] != "PROHIBITED"
        or plan["confirmation_materialization"] != "PROHIBITED"
        or plan["formal_materialization"] != "PROHIBITED"
    ):
        _fail("Frozen diagnostic plan semantic contract drifted")
    if (
        len(runner.EXPECTED_CASE_IDS)
        * len(runner.SEEDS)
        * len(runner.DIAGNOSTIC_ARMS)
        != EXPECTED_FULL_ROWS
        or runner.EXPECTED_CASE_IDS[-1] != TARGET_CASE_ID
    ):
        _fail("Frozen exact504 row construction contract drifted")

    current_source_manifest = runner._source_manifest(project_root)
    frozen_source_manifest = _exact_keys(
        plan["source_manifest"],
        {"schema", "hash_rule", "entry_count", "entries", "source_snapshot_sha256"},
        label="frozen source manifest",
    )
    if (
        frozen_source_manifest["schema"]
        != "v21e3r1_diagnostic_source_manifest_v1"
        or frozen_source_manifest["hash_rule"]
        != "sha256(canonical_json(sorted_entries))"
        or frozen_source_manifest["entry_count"] != 170
        or frozen_source_manifest["source_snapshot_sha256"]
        != EXPECTED_SOURCE_ROOT_SHA256
        or runner._canonical_json(current_source_manifest)
        != runner._canonical_json(frozen_source_manifest)
    ):
        _fail("Frozen diagnostic source root or manifest drifted")
    entries = frozen_source_manifest["entries"]
    if type(entries) is not list or len(entries) != 170:
        _fail("Frozen diagnostic source entry inventory drifted")
    runner_entries = [
        entry
        for entry in entries
        if type(entry) is dict
        and entry.get("path") == RUNNER_RELATIVE.as_posix()
    ]
    if (
        len(runner_entries) != 1
        or runner_entries[0].get("sha256") != EXPECTED_RUNNER_SHA256
        or runner_entries[0].get("bytes") != (project_root / RUNNER_RELATIVE).stat().st_size
    ):
        _fail("Frozen runner is not exactly bound by the source manifest")

    cases, bounds, directions, input_binding = runner._load_inputs(project_root)
    if runner._canonical_json(input_binding) != runner._canonical_json(
        plan["input_binding"]
    ):
        _fail("Frozen diagnostic input binding drifted")
    target_cases = [case for case in cases if case.get("case_id") == TARGET_CASE_ID]
    if len(target_cases) != 1:
        _fail("Target case is not uniquely present in the frozen input packet")
    target_case = target_cases[0]
    if target_case.get("family") != "MOTSP" or target_case.get("size") != 500:
        _fail("Target case family/size drifted")
    case_path = runner._case_path(project_root, target_case)
    case_sha256 = runner._sha256(case_path)
    lower, upper = bounds[TARGET_CASE_ID]
    case_index = list(runner.EXPECTED_CASE_IDS).index(TARGET_CASE_ID)
    rows: list[RowWork] = []
    all_row_ids = tuple(
        f"{case_id}__seed-{seed}__arm-{arm.lower()}"
        for case_id in runner.EXPECTED_CASE_IDS
        for seed in runner.SEEDS
        for arm in runner.DIAGNOSTIC_ARMS
    )
    if len(all_row_ids) != EXPECTED_FULL_ROWS or len(set(all_row_ids)) != EXPECTED_FULL_ROWS:
        _fail("Frozen exact504 row-ID construction drifted")
    local_ordinal = 0
    for seed in runner.SEEDS:
        for arm in runner.DIAGNOSTIC_ARMS:
            local_ordinal += 1
            ordinal = (
                case_index * len(runner.SEEDS) * len(runner.DIAGNOSTIC_ARMS)
                + local_ordinal
            )
            row_id = f"{TARGET_CASE_ID}__seed-{seed}__arm-{arm.lower()}"
            worker_spec = {
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
            rows.append(RowWork(ordinal=ordinal, row_id=row_id, worker_spec=worker_spec))
    if (
        len(rows) != EXPECTED_TARGET_ROWS
        or len({row.row_id for row in rows}) != EXPECTED_TARGET_ROWS
        or [row.ordinal for row in rows] != list(range(463, 505))
    ):
        _fail("Target row construction is not the exact disjoint 42-row suffix")

    return FrozenContext(
        project_root=project_root,
        output=output,
        plan_path=plan_path,
        plan_sha256=plan_sha256,
        source_snapshot_sha256=EXPECTED_SOURCE_ROOT_SHA256,
        runner_path=(project_root / RUNNER_RELATIVE).resolve(),
        runner_sha256=EXPECTED_RUNNER_SHA256,
        row_timeout_seconds=_exact_int(
            plan["row_timeout_seconds"], label="row_timeout_seconds", minimum=1
        ),
        existing_completed_marker_count=len(
            list((output / "completed").glob("*.json"))
        ),
        all_row_ids=all_row_ids,
        rows=tuple(rows),
    )


def _target_prefixed_entries(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        [item for item in root.iterdir() if item.name.startswith(TARGET_CASE_ID)],
        key=lambda item: item.name,
    )


def _completed_prefix_summary(
    context: FrozenContext, runner: ModuleType
) -> dict[str, object]:
    completed_root = context.output / "completed"
    marker_paths = sorted(completed_root.glob("*.json"), key=lambda path: path.name)
    observed_ids = {path.stem for path in marker_paths}
    expected_ids = set(context.all_row_ids)
    if not observed_ids.issubset(expected_ids):
        _fail("Completed directory contains a row outside the exact504 plan")
    prefix_count = 0
    for row_id in context.all_row_ids:
        if row_id in observed_ids:
            prefix_count += 1
        else:
            break
    if observed_ids != set(context.all_row_ids[:prefix_count]):
        _fail("Original main completed markers are not an exact contiguous plan prefix")
    non_target_row_count = EXPECTED_FULL_ROWS - EXPECTED_TARGET_ROWS
    if prefix_count > non_target_row_count:
        _fail("Original main entered the helper-owned target suffix before handoff")
    manifest: list[dict[str, object]] = []
    for ordinal, row_id in enumerate(context.all_row_ids[:prefix_count], start=1):
        marker = completed_root / f"{row_id}.json"
        completed = runner._completed_payload(context.output, row_id)
        if (
            completed is None
            or completed.get("row_id") != row_id
            or completed.get("plan_sha256") != EXPECTED_PLAN_SHA256
        ):
            _fail(f"Original-main completed prefix marker fails verification: {row_id}")
        manifest.append(
            {
                "ordinal": ordinal,
                "row_id": row_id,
                "marker_path": marker.relative_to(context.output).as_posix(),
                "marker_sha256": runner._sha256(marker),
                "attempt_directory": completed["attempt_directory"],
            }
        )
    last = manifest[-1] if manifest else None
    return {
        "completed_marker_count": prefix_count,
        "completed_prefix_first_ordinal": 1 if manifest else None,
        "completed_prefix_last_ordinal": prefix_count,
        "completed_prefix_manifest_sha256": hashlib.sha256(
            runner_canonical_bytes(manifest)
        ).hexdigest(),
        "last_completed_row_id": None if last is None else last["row_id"],
        "last_completed_marker_path": None if last is None else last["marker_path"],
        "last_completed_marker_sha256": None if last is None else last["marker_sha256"],
        "non_target_completed_row_count": prefix_count,
        "non_target_incomplete_row_ids": list(
            context.all_row_ids[prefix_count:non_target_row_count]
        ),
        "original_runner_resume_required": True,
        "original_runner_resume_after_helper_success_only": True,
    }


def _validate_handoff_command(command_line: str, context: FrozenContext) -> None:
    if type(command_line) is not str or not command_line.strip():
        _fail("Stopped main command line must be a nonempty exact string")
    classification = _classify_runner_command(command_line, context.output)
    if classification != ("driver", None):
        _fail("Stopped command is not the original driver for this exact output root")


def _handoff_exact_keys() -> set[str]:
    return {
        "schema",
        "status",
        "scope",
        "issued_at_utc",
        "stopped_main_pid",
        "stopped_main_command_line",
        "stopped_main_command_sha256",
        "output_root",
        "plan_sha256",
        "source_snapshot_sha256",
        "helper_sha256",
        "interpreter_identity",
        "environment_receipt",
        "process_scan",
        "completed_prefix",
        "receipt_is_audit_record_not_trusted_liveness",
        "runtime_authority",
        "scientific_authority",
        "selection_authority",
        "receipt_payload_sha256",
    }


def issue_external_handoff_receipt(
    context: FrozenContext,
    runner: ModuleType,
    *,
    stopped_main_pid: int,
    stopped_main_command_line: str,
    interpreter_identity: Mapping[str, object],
    environment_receipt: Mapping[str, object],
) -> dict[str, object]:
    stopped_main_pid = _exact_int(
        stopped_main_pid, label="stopped_main_pid", minimum=1
    )
    if stopped_main_pid == os.getpid():
        _fail("Stopped main PID cannot be the handoff issuer PID")
    _validate_handoff_command(stopped_main_command_line, context)
    handoff_path = context.output / HANDOFF_NAME
    if handoff_path.exists():
        _fail("Fixed external main-driver handoff receipt already exists")
    _assert_initial_target_unclaimed(context, runner, allow_handoff_absent=True)
    process_scan = _assert_no_conflicting_original_processes(context.output)
    all_observed_pids = {
        int(record["pid"])
        for field in (
            "driver_processes",
            "worker_processes",
            "unknown_runner_processes",
        )
        for record in process_scan[field]
    }
    if stopped_main_pid in all_observed_pids:
        _fail("Declared stopped main PID remains active")
    helper_sha256 = _raw_sha256(Path(__file__).resolve())
    core: dict[str, object] = {
        "schema": "v21e3r1_external_main_driver_handoff_receipt_v1",
        "status": "PASS_EXTERNAL_STOP_RECORDED_AND_NO_ORIGINAL_PROCESS_OBSERVED",
        "scope": "AUDIT_RECORD_ONLY_EXECUTION_MUST_RESCAN",
        "issued_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stopped_main_pid": stopped_main_pid,
        "stopped_main_command_line": stopped_main_command_line,
        "stopped_main_command_sha256": hashlib.sha256(
            stopped_main_command_line.encode("utf-8")
        ).hexdigest(),
        "output_root": context.output.as_posix(),
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "source_snapshot_sha256": EXPECTED_SOURCE_ROOT_SHA256,
        "helper_sha256": helper_sha256,
        "interpreter_identity": dict(interpreter_identity),
        "environment_receipt": dict(environment_receipt),
        "process_scan": process_scan,
        "completed_prefix": _completed_prefix_summary(context, runner),
        "receipt_is_audit_record_not_trusted_liveness": True,
        "runtime_authority": False,
        "scientific_authority": False,
        "selection_authority": False,
    }
    receipt = dict(core)
    receipt["receipt_payload_sha256"] = hashlib.sha256(
        runner_canonical_bytes(core)
    ).hexdigest()
    runner._exclusive_json(handoff_path, receipt)
    _validate_handoff_receipt(
        context,
        runner,
        interpreter_identity=interpreter_identity,
        environment_receipt=environment_receipt,
    )
    return receipt


def _validate_handoff_receipt(
    context: FrozenContext,
    runner: ModuleType,
    *,
    interpreter_identity: Mapping[str, object],
    environment_receipt: Mapping[str, object],
    require_current_prefix: bool = True,
) -> tuple[dict[str, object], str]:
    path = context.output / HANDOFF_NAME
    if not path.is_file():
        _fail(
            "Fixed external main-driver handoff receipt is required before helper claim"
        )
    raw_sha256 = runner._sha256(path)
    receipt = _exact_keys(
        runner._load_json_object(path),
        _handoff_exact_keys(),
        label="external main-driver handoff receipt",
    )
    core = dict(receipt)
    payload_sha256 = _exact_sha256(
        core.pop("receipt_payload_sha256"), label="handoff receipt payload"
    )
    if payload_sha256 != hashlib.sha256(runner_canonical_bytes(core)).hexdigest():
        _fail("External handoff receipt payload digest drifted")
    issued_at = receipt["issued_at_utc"]
    stopped_pid = _exact_int(
        receipt["stopped_main_pid"], label="handoff stopped_main_pid", minimum=1
    )
    stopped_command = receipt["stopped_main_command_line"]
    if (
        type(issued_at) is not str
        or UTC_RE.fullmatch(issued_at) is None
        or type(stopped_command) is not str
        or receipt["schema"] != "v21e3r1_external_main_driver_handoff_receipt_v1"
        or receipt["status"]
        != "PASS_EXTERNAL_STOP_RECORDED_AND_NO_ORIGINAL_PROCESS_OBSERVED"
        or receipt["scope"] != "AUDIT_RECORD_ONLY_EXECUTION_MUST_RESCAN"
        or stopped_pid == os.getpid()
        or receipt["stopped_main_command_sha256"]
        != hashlib.sha256(stopped_command.encode("utf-8")).hexdigest()
        or receipt["output_root"] != context.output.as_posix()
        or receipt["plan_sha256"] != EXPECTED_PLAN_SHA256
        or receipt["source_snapshot_sha256"] != EXPECTED_SOURCE_ROOT_SHA256
        or receipt["helper_sha256"] != _raw_sha256(Path(__file__).resolve())
        or receipt["interpreter_identity"] != dict(interpreter_identity)
        or receipt["environment_receipt"] != dict(environment_receipt)
        or receipt["receipt_is_audit_record_not_trusted_liveness"] is not True
        or receipt["runtime_authority"] is not False
        or receipt["scientific_authority"] is not False
        or receipt["selection_authority"] is not False
    ):
        _fail("External handoff receipt identity/status contract drifted")
    _validate_handoff_command(stopped_command, context)
    process_scan = _exact_keys(
        receipt["process_scan"],
        {
            "schema",
            "method",
            "output_root",
            "driver_processes",
            "worker_processes",
            "unknown_runner_processes",
            "scan_payload_sha256",
        },
        label="handoff process scan",
    )
    scan_core = dict(process_scan)
    scan_payload = _exact_sha256(
        scan_core.pop("scan_payload_sha256"), label="handoff process scan payload"
    )
    if (
        scan_payload != hashlib.sha256(runner_canonical_bytes(scan_core)).hexdigest()
        or process_scan["driver_processes"] != []
        or process_scan["worker_processes"] != []
        or process_scan["unknown_runner_processes"] != []
    ):
        _fail("Handoff did not bind an empty original-process scan")
    completed_prefix = _exact_keys(
        receipt["completed_prefix"],
        {
            "completed_marker_count",
            "completed_prefix_first_ordinal",
            "completed_prefix_last_ordinal",
            "completed_prefix_manifest_sha256",
            "last_completed_row_id",
            "last_completed_marker_path",
            "last_completed_marker_sha256",
            "non_target_completed_row_count",
            "non_target_incomplete_row_ids",
            "original_runner_resume_required",
            "original_runner_resume_after_helper_success_only",
        },
        label="handoff completed prefix",
    )
    if require_current_prefix and completed_prefix != _completed_prefix_summary(
        context, runner
    ):
        _fail("Completed-marker prefix changed after external handoff")
    return receipt, raw_sha256


def _assert_target_state(
    context: FrozenContext,
    runner: ModuleType,
    *,
    owned_attempts: Mapping[str, Path],
    sealed_rows: set[str],
) -> None:
    row_ids = {row.row_id for row in context.rows}
    attempt_root = context.output / "attempts"
    completed_root = context.output / "completed"
    for entry in _target_prefixed_entries(attempt_root):
        if entry.name not in row_ids:
            _fail(f"Unexpected target-case attempt root appeared: {entry.name}")
    for entry in _target_prefixed_entries(completed_root):
        if entry.suffix != ".json" or entry.stem not in row_ids:
            _fail(f"Unexpected target-case completed marker appeared: {entry.name}")
    for row_id in row_ids:
        row_root = attempt_root / row_id
        marker = completed_root / f"{row_id}.json"
        owned = owned_attempts.get(row_id)
        if owned is None:
            if row_root.exists():
                _fail(f"Non-helper target attempt state appeared: {row_id}")
        else:
            if not row_root.is_dir() or owned.parent != row_root or not owned.is_dir():
                _fail(f"Owned target attempt path drifted: {row_id}")
            children = sorted(row_root.iterdir(), key=lambda item: item.name)
            if children != [owned]:
                _fail(f"Concurrent or inconsistent target attempt appeared: {row_id}")
        if marker.exists():
            if row_id not in sealed_rows or owned is None:
                _fail(f"Non-helper target completed marker appeared: {row_id}")
            completed = runner._completed_payload(context.output, row_id)
            expected_attempt = owned.relative_to(context.output).as_posix()
            if (
                completed is None
                or completed.get("row_id") != row_id
                or completed.get("attempt_directory") != expected_attempt
                or completed.get("plan_sha256") != EXPECTED_PLAN_SHA256
            ):
                _fail(f"Helper-owned completed marker drifted: {row_id}")
        elif row_id in sealed_rows:
            _fail(f"Helper-sealed completed marker disappeared: {row_id}")


def _assert_initial_target_unclaimed(
    context: FrozenContext,
    runner: ModuleType,
    *,
    allow_handoff_absent: bool = False,
) -> None:
    if (context.output / "diagnostic.aggregate.json").exists() or (
        context.output / "diagnostic.receipt.json"
    ).exists():
        _fail("Main diagnostic is already finalized")
    for name in (CLAIM_NAME, FAILURE_NAME, RECEIPT_NAME, RECEIPT_SEAL_NAME):
        if (context.output / name).exists():
            _fail(f"External helper-instance evidence already exists: {name}")
    if not allow_handoff_absent and not (context.output / HANDOFF_NAME).is_file():
        _fail("Fixed external main-driver handoff receipt is absent")
    _assert_target_state(context, runner, owned_attempts={}, sealed_rows=set())


def _validate_worker_result(
    context: FrozenContext,
    runner: ModuleType,
    row: RowWork,
    attempt: Path,
) -> dict[str, object]:
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
        label=f"worker result {row.row_id}",
    )
    if result["status"] != "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY":
        _fail(f"Frozen worker did not PASS: {row.row_id}")
    for name, field in (
        ("row.json", "row_sha256"),
        ("diagnostic.json", "diagnostic_sha256"),
        ("trace.sqlite3", "trace_sha256"),
        ("terminal.receipt.json", "terminal_receipt_sha256"),
        ("independent.metric.json", "independent_metric_receipt_sha256"),
    ):
        expected = _exact_sha256(result[field], label=f"worker result {field}")
        artifact = attempt / name
        if not artifact.is_file() or runner._sha256(artifact) != expected:
            _fail(f"Frozen worker artifact verification failed: {row.row_id}/{name}")
    row_payload = runner._load_json_object(attempt / "row.json")
    spec = row.worker_spec
    for field, expected in (
        ("status", "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY"),
        ("case_id", TARGET_CASE_ID),
        ("family", "MOTSP"),
        ("size", 500),
        ("seed", spec["seed"]),
        ("arm_id", spec["arm_id"]),
        ("charged_evaluation_budget", spec["charged_evaluation_budget"]),
        ("checkpoint_period", spec["checkpoint_period"]),
        ("case_artifact_sha256", spec["case_artifact_sha256"]),
        ("source_snapshot_sha256", EXPECTED_SOURCE_ROOT_SHA256),
        ("plan_sha256", EXPECTED_PLAN_SHA256),
        ("selection_entropy_release", "PROHIBITED"),
        ("confirmation_materialization", "PROHIBITED"),
        ("formal_materialization", "PROHIBITED"),
    ):
        if row_payload.get(field) != expected:
            _fail(f"Frozen worker row payload drifted at {field}: {row.row_id}")
    return result


def _verify_worker_spec(
    runner: ModuleType,
    row: RowWork,
    spec_path: Path,
    *,
    expected_raw_sha256: str | None = None,
) -> tuple[str, str]:
    before_sha256 = runner._sha256(spec_path)
    loaded = _exact_keys(
        runner._load_json_object(spec_path),
        set(row.worker_spec),
        label=f"worker spec {row.row_id}",
    )
    if runner._canonical_json(loaded) != runner._canonical_json(row.worker_spec):
        _fail(f"Worker spec semantic payload drifted: {row.row_id}")
    after_sha256 = runner._sha256(spec_path)
    if before_sha256 != after_sha256:
        _fail(f"Worker spec changed during verification: {row.row_id}")
    if expected_raw_sha256 is not None and before_sha256 != expected_raw_sha256:
        _fail(f"Worker spec raw bytes drifted: {row.row_id}")
    payload_sha256 = hashlib.sha256(
        runner_canonical_bytes(row.worker_spec)
    ).hexdigest()
    return before_sha256, payload_sha256


def _verify_execution_boundary(
    context: FrozenContext,
    runner: ModuleType,
    *,
    helper_sha256: str,
    interpreter_identity: Mapping[str, object],
    environment_receipt: Mapping[str, object],
) -> None:
    if (
        _raw_sha256(context.plan_path) != EXPECTED_PLAN_SHA256
        or _raw_sha256(context.runner_path) != EXPECTED_RUNNER_SHA256
        or _raw_sha256(Path(__file__).resolve()) != helper_sha256
        or runner._source_manifest(context.project_root).get("source_snapshot_sha256")
        != EXPECTED_SOURCE_ROOT_SHA256
        or _exact_interpreter_identity() != dict(interpreter_identity)
        or _execution_environment_receipt(
            context.project_root, interpreter_identity
        )
        != dict(environment_receipt)
    ):
        _fail("Frozen plan/source/helper/interpreter/environment boundary drifted")


def _claim_payload(
    context: FrozenContext,
    helper_sha256: str,
    *,
    handoff_sha256: str,
    handoff_payload_sha256: str,
    interpreter_identity: Mapping[str, object],
    environment_receipt: Mapping[str, object],
) -> dict[str, object]:
    worker_spec_payload_manifest = [
        {
            "ordinal": row.ordinal,
            "row_id": row.row_id,
            "worker_spec_payload_sha256": hashlib.sha256(
                runner_canonical_bytes(row.worker_spec)
            ).hexdigest(),
        }
        for row in context.rows
    ]
    core: dict[str, object] = {
        "schema": "v21e3r1_external_scheduling_helper_instance_claim_v2",
        "status": "SEALED_HELPER_INSTANCE_CLAIM_NOT_A_MAIN_RUNNER_LOCK",
        "scope": "EXTERNAL_SCHEDULING_ONLY_FROZEN_DEVELOPMENT_ROWS",
        "target_case_id": TARGET_CASE_ID,
        "target_row_count": EXPECTED_TARGET_ROWS,
        "target_row_ids": [row.row_id for row in context.rows],
        "worker_spec_payload_manifest": worker_spec_payload_manifest,
        "worker_spec_payload_manifest_sha256": hashlib.sha256(
            runner_canonical_bytes(worker_spec_payload_manifest)
        ).hexdigest(),
        "jobs": JOBS,
        "process_id": os.getpid(),
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "source_snapshot_sha256": EXPECTED_SOURCE_ROOT_SHA256,
        "frozen_runner_sha256": EXPECTED_RUNNER_SHA256,
        "helper_sha256": helper_sha256,
        "handoff_receipt_path": HANDOFF_NAME,
        "handoff_receipt_sha256": handoff_sha256,
        "handoff_receipt_payload_sha256": handoff_payload_sha256,
        "interpreter_identity": dict(interpreter_identity),
        "environment_receipt": dict(environment_receipt),
        "original_main_runner_honors_this_claim": False,
        "operational_quiescence_depends_on_external_stop_and_repeated_process_scan": True,
        "runtime_authority": False,
        "scientific_authority": False,
        "selection_authority": False,
    }
    payload = dict(core)
    payload["claim_payload_sha256"] = hashlib.sha256(
        runner_canonical_bytes(core)
    ).hexdigest()
    return payload


def runner_canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def runner_exclusive_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _validate_bound_payload_file(
    path: Path,
    runner: ModuleType,
    *,
    expected_keys: set[str],
    payload_field: str,
    label: str,
) -> tuple[dict[str, object], str]:
    before_sha256 = runner._sha256(path)
    payload = _exact_keys(
        runner._load_json_object(path), expected_keys, label=label
    )
    core = dict(payload)
    payload_sha256 = _exact_sha256(
        core.pop(payload_field), label=f"{label} {payload_field}"
    )
    if payload_sha256 != hashlib.sha256(runner_canonical_bytes(core)).hexdigest():
        _fail(f"{label} payload digest drifted")
    after_sha256 = runner._sha256(path)
    if before_sha256 != after_sha256:
        _fail(f"{label} changed during verification")
    return payload, before_sha256


def _validate_claim_file(
    context: FrozenContext,
    runner: ModuleType,
    *,
    expected_claim: Mapping[str, object],
    expected_claim_sha256: str,
) -> None:
    path = context.output / CLAIM_NAME
    payload, raw_sha256 = _validate_bound_payload_file(
        path,
        runner,
        expected_keys=set(expected_claim),
        payload_field="claim_payload_sha256",
        label="helper-instance claim",
    )
    if payload != dict(expected_claim) or raw_sha256 != expected_claim_sha256:
        _fail("Helper-instance claim bytes or payload drifted")


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _cleanup_owned_marker_after_verifier_failure(
    marker_path: Path,
    *,
    owned_marker_sha256: str,
    owned_marker_payload: Mapping[str, object],
    runner: ModuleType,
) -> str:
    if marker_path.is_symlink() or not marker_path.is_file():
        return "RETAINED_NOT_REGULAR_OR_MISSING"
    try:
        current_sha256 = runner._sha256(marker_path)
        current_payload = runner._load_json_object(marker_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return "RETAINED_UNREADABLE_OR_EXTERNALLY_CHANGED"
    if (
        current_sha256 != owned_marker_sha256
        or runner._canonical_json(current_payload)
        != runner._canonical_json(dict(owned_marker_payload))
    ):
        return "RETAINED_EXTERNALLY_CHANGED"
    marker_path.unlink()
    return "REMOVED_EXACT_HELPER_OWNED_FALSE_MARKER"


def _write_durable_failure_receipt(
    context: FrozenContext,
    runner: ModuleType,
    *,
    error: BaseException,
    helper_sha256: str,
    claim_sha256: str,
    handoff_sha256: str,
    handoff_payload_sha256: str,
    interpreter_identity: Mapping[str, object],
    environment_receipt: Mapping[str, object],
    owned_attempts: Mapping[str, Path],
    marker_entries: Mapping[str, Mapping[str, object]],
    marker_cleanup_events: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    path = context.output / FAILURE_NAME
    if path.exists():
        _fail("Durable external scheduling failure receipt already exists")
    attempt_rows: list[dict[str, object]] = []
    for row_id, attempt in sorted(owned_attempts.items()):
        spec_path = attempt / "worker.spec.json"
        try:
            spec_sha256 = runner._sha256(spec_path) if spec_path.is_file() else None
        except OSError:
            spec_sha256 = None
        attempt_rows.append(
            {
                "row_id": row_id,
                "attempt_directory": attempt.relative_to(context.output).as_posix(),
                "worker_spec_sha256": spec_sha256,
            }
        )
    core: dict[str, object] = {
        "schema": "v21e3r1_external_scheduling_failure_receipt_v1",
        "status": "HOLD_EXTERNAL_SCHEDULING_FAILURE_MANUAL_AUDIT_REQUIRED",
        "scope": "EXTERNAL_SCHEDULING_ONLY_FROZEN_DEVELOPMENT_ROWS",
        "target_case_id": TARGET_CASE_ID,
        "target_row_count": EXPECTED_TARGET_ROWS,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "source_snapshot_sha256": EXPECTED_SOURCE_ROOT_SHA256,
        "frozen_runner_sha256": EXPECTED_RUNNER_SHA256,
        "helper_sha256": helper_sha256,
        "handoff_receipt_path": HANDOFF_NAME,
        "handoff_receipt_sha256": handoff_sha256,
        "handoff_receipt_payload_sha256": handoff_payload_sha256,
        "helper_instance_claim_path": CLAIM_NAME,
        "helper_instance_claim_sha256": claim_sha256,
        "interpreter_identity": dict(interpreter_identity),
        "environment_receipt": dict(environment_receipt),
        "exception_type": type(error).__name__,
        "exception_message": str(error)[-2000:],
        "owned_attempts": attempt_rows,
        "validated_completed_markers": [
            dict(marker_entries[row_id]) for row_id in sorted(marker_entries)
        ],
        "marker_cleanup_events": [dict(event) for event in marker_cleanup_events],
        "automatic_retry_authorized": False,
        "manual_audit_required": True,
        "runtime_authority": False,
        "scientific_authority": False,
        "selection_authority": False,
        "confirmation_authority": False,
        "formal_study_authority": False,
        "publication_status": "IJOC_HOLD",
    }
    receipt = dict(core)
    receipt["receipt_payload_sha256"] = hashlib.sha256(
        runner_canonical_bytes(core)
    ).hexdigest()
    runner._exclusive_json(path, receipt)
    _fsync_file(path)
    verified, _ = _validate_bound_payload_file(
        path,
        runner,
        expected_keys=set(receipt),
        payload_field="receipt_payload_sha256",
        label="durable external scheduling failure receipt",
    )
    if verified != receipt:
        _fail("Durable external scheduling failure receipt changed after write")
    return receipt


def execute_external_schedule(
    context: FrozenContext,
    runner: ModuleType,
    *,
    interpreter_identity: Mapping[str, object],
    environment_receipt: Mapping[str, object],
    handoff_receipt: Mapping[str, object],
    handoff_sha256: str,
) -> dict[str, object]:
    _assert_initial_target_unclaimed(context, runner)
    observed_handoff, observed_handoff_sha256 = _validate_handoff_receipt(
        context,
        runner,
        interpreter_identity=interpreter_identity,
        environment_receipt=environment_receipt,
    )
    if (
        observed_handoff != dict(handoff_receipt)
        or observed_handoff_sha256 != handoff_sha256
    ):
        _fail("Execution handoff binding drifted before helper-instance claim")
    _assert_no_conflicting_original_processes(context.output)
    helper_path = Path(__file__).resolve()
    helper_sha256 = _raw_sha256(helper_path)
    _verify_execution_boundary(
        context,
        runner,
        helper_sha256=helper_sha256,
        interpreter_identity=interpreter_identity,
        environment_receipt=environment_receipt,
    )
    claim_path = context.output / CLAIM_NAME
    receipt_path = context.output / RECEIPT_NAME
    receipt_seal_path = context.output / RECEIPT_SEAL_NAME
    claim = _claim_payload(
        context,
        helper_sha256,
        handoff_sha256=handoff_sha256,
        handoff_payload_sha256=str(handoff_receipt["receipt_payload_sha256"]),
        interpreter_identity=interpreter_identity,
        environment_receipt=environment_receipt,
    )
    claim_sha256 = hashlib.sha256(runner_exclusive_json_bytes(claim)).hexdigest()
    claim_write_may_have_materialized = False
    owned_attempts: dict[str, Path] = {}
    sealed_rows: set[str] = set()
    marker_entries: dict[str, dict[str, object]] = {}
    marker_cleanup_events: list[dict[str, object]] = []
    worker_spec_bindings: dict[str, dict[str, str]] = {}
    running_worker_specs: set[str] = set()
    state_lock = threading.Lock()
    process_scan_lock = threading.Lock()
    stop_event = threading.Event()
    success_receipt_payload: dict[str, object] | None = None
    success_receipt_owned_sha256: str | None = None
    success_seal_payload: dict[str, object] | None = None
    success_seal_owned_sha256: str | None = None

    def verify_operational_boundary() -> None:
        with process_scan_lock:
            _verify_execution_boundary(
                context,
                runner,
                helper_sha256=helper_sha256,
                interpreter_identity=interpreter_identity,
                environment_receipt=environment_receipt,
            )
            if runner._sha256(context.output / HANDOFF_NAME) != handoff_sha256:
                _fail("External handoff receipt bytes drifted during scheduling")
            with state_lock:
                allowed_specs = set(running_worker_specs)
            _assert_no_conflicting_original_processes(
                context.output, allowed_helper_worker_specs=allowed_specs
            )

    def run_one(row: RowWork) -> dict[str, object]:
        if stop_event.is_set():
            _fail("External scheduling was cancelled after another row failed")
        verify_operational_boundary()
        with state_lock:
            _assert_target_state(
                context,
                runner,
                owned_attempts=owned_attempts,
                sealed_rows=sealed_rows,
            )
            attempt = runner._next_attempt_directory(context.output, row.row_id).resolve()
            expected_attempt = context.output / "attempts" / row.row_id / "attempt-0001"
            if attempt != expected_attempt.resolve():
                _fail(f"Target row did not receive helper-owned attempt-0001: {row.row_id}")
            owned_attempts[row.row_id] = attempt
            spec_path = attempt / "worker.spec.json"
            runner._exclusive_json(spec_path, row.worker_spec)
            spec_raw_sha256, spec_payload_sha256 = _verify_worker_spec(
                runner, row, spec_path
            )
            worker_spec_bindings[row.row_id] = {
                "worker_spec_path": spec_path.relative_to(context.output).as_posix(),
                "worker_spec_sha256": spec_raw_sha256,
                "worker_spec_payload_sha256": spec_payload_sha256,
            }
            running_worker_specs.add(spec_path.resolve().as_posix())
        verify_operational_boundary()
        _verify_worker_spec(
            runner, row, spec_path, expected_raw_sha256=spec_raw_sha256
        )
        child_error: BaseException | None = None
        try:
            runner._run_child(
                spec_path,
                project_root=context.project_root,
                timeout_seconds=context.row_timeout_seconds,
            )
        except BaseException as error:
            child_error = error
        try:
            _verify_worker_spec(
                runner, row, spec_path, expected_raw_sha256=spec_raw_sha256
            )
            verify_operational_boundary()
        finally:
            with state_lock:
                running_worker_specs.discard(spec_path.resolve().as_posix())
        if child_error is not None:
            raise child_error
        result = _validate_worker_result(context, runner, row, attempt)
        if stop_event.is_set():
            _fail("External scheduling was cancelled before marker materialization")
        verify_operational_boundary()
        _verify_worker_spec(
            runner, row, spec_path, expected_raw_sha256=spec_raw_sha256
        )
        with state_lock:
            _assert_target_state(
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
            owned_marker_sha256 = hashlib.sha256(
                runner_exclusive_json_bytes(completed)
            ).hexdigest()
            runner._exclusive_json(marker_path, completed)
            try:
                verified = runner._completed_payload(context.output, row.row_id)
                if (
                    verified is None
                    or runner._canonical_json(verified)
                    != runner._canonical_json(completed)
                ):
                    _fail(f"Frozen completed-payload verifier rejected: {row.row_id}")
            except BaseException:
                cleanup_status = _cleanup_owned_marker_after_verifier_failure(
                    marker_path,
                    owned_marker_sha256=owned_marker_sha256,
                    owned_marker_payload=completed,
                    runner=runner,
                )
                marker_cleanup_events.append(
                    {
                        "row_id": row.row_id,
                        "marker_path": marker_path.relative_to(
                            context.output
                        ).as_posix(),
                        "owned_marker_sha256": owned_marker_sha256,
                        "cleanup_status": cleanup_status,
                    }
                )
                raise
            sealed_rows.add(row.row_id)
            spec_binding = worker_spec_bindings[row.row_id]
            marker_entry = {
                "ordinal": row.ordinal,
                "row_id": row.row_id,
                "path": marker_path.relative_to(context.output).as_posix(),
                "sha256": owned_marker_sha256,
                "attempt_directory": completed["attempt_directory"],
                **spec_binding,
            }
            marker_entries[row.row_id] = marker_entry
            _assert_target_state(
                context,
                runner,
                owned_attempts=owned_attempts,
                sealed_rows=sealed_rows,
            )
            return marker_entry

    try:
        # Set the expected-payload/hash ownership witness before the exclusive
        # create: the create itself may materialize bytes and then raise.
        claim_write_may_have_materialized = True
        runner._exclusive_json(claim_path, claim)
        _fsync_file(claim_path)
        _validate_claim_file(
            context,
            runner,
            expected_claim=claim,
            expected_claim_sha256=claim_sha256,
        )
        verify_operational_boundary()
        _assert_target_state(
            context, runner, owned_attempts=owned_attempts, sealed_rows=sealed_rows
        )
        first_error: BaseException | None = None
        futures: list[Future[dict[str, object]]] = []
        with ThreadPoolExecutor(
            max_workers=JOBS, thread_name_prefix="v21e3r1-external-scheduling"
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

        ordered_markers = [marker_entries[row.row_id] for row in context.rows]
        with state_lock:
            _assert_target_state(
                context,
                runner,
                owned_attempts=owned_attempts,
                sealed_rows=sealed_rows,
            )
        if (
            len(ordered_markers) != EXPECTED_TARGET_ROWS
            or len(sealed_rows) != EXPECTED_TARGET_ROWS
        ):
            _fail("External scheduling did not verify the exact target 42 markers")
        verify_operational_boundary()
        _validate_claim_file(
            context,
            runner,
            expected_claim=claim,
            expected_claim_sha256=claim_sha256,
        )
        _validate_handoff_receipt(
            context,
            runner,
            interpreter_identity=interpreter_identity,
            environment_receipt=environment_receipt,
            require_current_prefix=False,
        )
        for row, marker_entry in zip(context.rows, ordered_markers, strict=True):
            spec_path = context.output / str(marker_entry["worker_spec_path"])
            _verify_worker_spec(
                runner,
                row,
                spec_path,
                expected_raw_sha256=str(marker_entry["worker_spec_sha256"]),
            )
            marker_path = context.output / str(marker_entry["path"])
            if runner._sha256(marker_path) != marker_entry["sha256"]:
                _fail(f"Completed marker drifted before receipt: {row.row_id}")
            completed = runner._completed_payload(context.output, row.row_id)
            if (
                completed is None
                or completed.get("attempt_directory")
                != marker_entry["attempt_directory"]
                or completed.get("plan_sha256") != EXPECTED_PLAN_SHA256
            ):
                _fail(f"Completed marker failed final frozen verification: {row.row_id}")

        core: dict[str, object] = {
            "schema": "v21e3r1_external_scheduling_only_receipt_v2",
            "status": "PASS_EXTERNAL_SCHEDULING_ONLY_TARGET_42",
            "scope": "EXTERNAL_SCHEDULING_ONLY_FROZEN_DEVELOPMENT_ROWS",
            "scheduling_policy": "THREAD_POOL_EXECUTOR_MAX_WORKERS_4",
            "target_case_id": TARGET_CASE_ID,
            "full_plan_row_count": EXPECTED_FULL_ROWS,
            "target_row_count": EXPECTED_TARGET_ROWS,
            "jobs": JOBS,
            "plan_path": PLAN_NAME,
            "plan_sha256": EXPECTED_PLAN_SHA256,
            "source_snapshot_sha256": EXPECTED_SOURCE_ROOT_SHA256,
            "frozen_runner_path": RUNNER_RELATIVE.as_posix(),
            "frozen_runner_sha256": EXPECTED_RUNNER_SHA256,
            "helper_path": HELPER_RELATIVE.as_posix(),
            "helper_sha256": helper_sha256,
            "handoff_receipt_path": HANDOFF_NAME,
            "handoff_receipt_sha256": handoff_sha256,
            "handoff_receipt_payload_sha256": handoff_receipt[
                "receipt_payload_sha256"
            ],
            "helper_instance_claim_path": CLAIM_NAME,
            "helper_instance_claim_sha256": claim_sha256,
            "interpreter_identity": dict(interpreter_identity),
            "environment_receipt": dict(environment_receipt),
            "completed_marker_count": EXPECTED_TARGET_ROWS,
            "completed_markers": ordered_markers,
            "completed_marker_generation": (
                "HELPER_MATERIALIZED_ORIGINAL_FORMAT_ONLY_AFTER_FROZEN_WORKER_"
                "RESULT_ARTIFACT_VALIDATION"
            ),
            "completed_marker_verification": (
                "FROZEN_RUNNER_COMPLETED_PAYLOAD_REVALIDATED_PER_ROW_AND_FINAL"
            ),
            "worker_execution": "DELEGATED_TO_FROZEN_RUNNER_RUN_CHILD_AND_WORKER",
            "original_main_runner_honors_helper_instance_claim": False,
            "original_runner_resume_required": True,
            "original_runner_resume_after_helper_success_only": True,
            "receipt_seal_path": RECEIPT_SEAL_NAME,
            "case_generation_performed": False,
            "generated_case_count": 0,
            "original_runner_or_algorithm_sources_modified": False,
            "implementation_independence": False,
            "algorithm_execution_independence": False,
            "scientific_independence": False,
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
            "confirmation_authority": False,
            "formal_study_authority": False,
            "publication_status": "IJOC_HOLD",
        }
        receipt = dict(core)
        receipt["receipt_payload_sha256"] = hashlib.sha256(
            runner_canonical_bytes(core)
        ).hexdigest()
        owned_receipt_file_sha256 = hashlib.sha256(
            runner_exclusive_json_bytes(receipt)
        ).hexdigest()
        success_receipt_payload = receipt
        success_receipt_owned_sha256 = owned_receipt_file_sha256
        runner._exclusive_json(receipt_path, receipt)
        _fsync_file(receipt_path)
        verified_receipt, receipt_file_sha256 = _validate_bound_payload_file(
            receipt_path,
            runner,
            expected_keys=set(receipt),
            payload_field="receipt_payload_sha256",
            label="external scheduling success receipt",
        )
        if verified_receipt != receipt:
            _fail("External scheduling success receipt payload changed after write")
        if receipt_file_sha256 != owned_receipt_file_sha256:
            _fail("External scheduling success receipt raw bytes are not helper-owned")
        seal_core: dict[str, object] = {
            "schema": "v21e3r1_external_scheduling_receipt_file_seal_v1",
            "status": "PASS_SUCCESS_RECEIPT_FILE_DIGEST_SEALED",
            "receipt_path": RECEIPT_NAME,
            "receipt_sha256": receipt_file_sha256,
            "receipt_payload_sha256": receipt["receipt_payload_sha256"],
            "helper_instance_claim_sha256": claim_sha256,
            "handoff_receipt_sha256": handoff_sha256,
            "runtime_authority": False,
            "scientific_authority": False,
            "selection_authority": False,
        }
        seal = dict(seal_core)
        seal["seal_payload_sha256"] = hashlib.sha256(
            runner_canonical_bytes(seal_core)
        ).hexdigest()
        success_seal_payload = seal
        success_seal_owned_sha256 = hashlib.sha256(
            runner_exclusive_json_bytes(seal)
        ).hexdigest()
        runner._exclusive_json(receipt_seal_path, seal)
        _fsync_file(receipt_seal_path)
        verified_seal, _ = _validate_bound_payload_file(
            receipt_seal_path,
            runner,
            expected_keys=set(seal),
            payload_field="seal_payload_sha256",
            label="external scheduling success receipt seal",
        )
        if verified_seal != seal or runner._sha256(receipt_path) != receipt_file_sha256:
            _fail("External scheduling success receipt/seal post-write digest drifted")
        return receipt
    except BaseException as error:
        if not claim_write_may_have_materialized:
            raise
        if receipt_seal_path.is_file():
            try:
                if success_seal_payload is None or success_seal_owned_sha256 is None:
                    seal_cleanup = "RETAINED_NO_HELPER_OWNERSHIP_WITNESS"
                else:
                    seal_cleanup = _cleanup_owned_marker_after_verifier_failure(
                        receipt_seal_path,
                        owned_marker_sha256=success_seal_owned_sha256,
                        owned_marker_payload=success_seal_payload,
                        runner=runner,
                    )
            except BaseException:
                seal_cleanup = "RETAINED_UNREADABLE_SUCCESS_RECEIPT_SEAL"
            marker_cleanup_events.append(
                {
                    "row_id": "__success_receipt_seal__",
                    "marker_path": RECEIPT_SEAL_NAME,
                    "owned_marker_sha256": success_seal_owned_sha256,
                    "cleanup_status": seal_cleanup,
                }
            )
        if receipt_path.is_file():
            try:
                if (
                    success_receipt_payload is None
                    or success_receipt_owned_sha256 is None
                ):
                    cleanup = "RETAINED_NO_HELPER_OWNERSHIP_WITNESS"
                else:
                    cleanup = _cleanup_owned_marker_after_verifier_failure(
                        receipt_path,
                        owned_marker_sha256=success_receipt_owned_sha256,
                        owned_marker_payload=success_receipt_payload,
                        runner=runner,
                    )
            except BaseException:
                cleanup = "RETAINED_UNREADABLE_SUCCESS_RECEIPT"
            marker_cleanup_events.append(
                {
                    "row_id": "__success_receipt__",
                    "marker_path": RECEIPT_NAME,
                    "owned_marker_sha256": None,
                    "cleanup_status": cleanup,
                }
            )
        _write_durable_failure_receipt(
            context,
            runner,
            error=error,
            helper_sha256=helper_sha256,
            claim_sha256=claim_sha256,
            handoff_sha256=handoff_sha256,
            handoff_payload_sha256=str(
                handoff_receipt["receipt_payload_sha256"]
            ),
            interpreter_identity=interpreter_identity,
            environment_receipt=environment_receipt,
            owned_attempts=owned_attempts,
            marker_entries=marker_entries,
            marker_cleanup_events=marker_cleanup_events,
        )
        raise RuntimeError(
            "External scheduling stopped fail-closed; helper-instance claim, durable "
            "failure receipt, and append-only attempt evidence were retained; automatic "
            "resume is prohibited"
        ) from error


def preflight(
    project_root: str | Path,
    output_directory: str | Path,
    *,
    require_handoff: bool,
) -> tuple[
    FrozenContext,
    ModuleType,
    dict[str, object],
    dict[str, object],
    dict[str, object] | None,
    str | None,
]:
    interpreter_identity = _exact_interpreter_identity()
    root = Path(project_root).resolve()
    output = Path(output_directory).resolve()
    _validate_production_paths(root, output)
    environment_receipt = _execution_environment_receipt(
        root, interpreter_identity
    )
    runner = _load_frozen_runner(root)
    context = _validate_plan_and_build_rows(root, output, runner)
    _assert_initial_target_unclaimed(
        context, runner, allow_handoff_absent=not require_handoff
    )
    _assert_no_conflicting_original_processes(context.output)
    if require_handoff:
        handoff_receipt, handoff_sha256 = _validate_handoff_receipt(
            context,
            runner,
            interpreter_identity=interpreter_identity,
            environment_receipt=environment_receipt,
        )
    else:
        handoff_receipt, handoff_sha256 = None, None
    return (
        context,
        runner,
        interpreter_identity,
        environment_receipt,
        handoff_receipt,
        handoff_sha256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(default_root))
    parser.add_argument(
        "--output-directory", default=str(default_root / OUTPUT_RELATIVE)
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the frozen boundary and target absence without writing or running rows.",
    )
    parser.add_argument(
        "--issue-handoff-receipt",
        action="store_true",
        help=(
            "After an external operator stops the original main driver, rescan and "
            "exclusively issue the fixed audit-only handoff receipt."
        ),
    )
    parser.add_argument("--stopped-main-pid", type=int)
    parser.add_argument("--stopped-main-command-line")
    args = parser.parse_args(argv)
    if args.preflight_only and args.issue_handoff_receipt:
        parser.error("--preflight-only and --issue-handoff-receipt are exclusive")
    if args.issue_handoff_receipt:
        if args.stopped_main_pid is None or args.stopped_main_command_line is None:
            parser.error(
                "--issue-handoff-receipt requires --stopped-main-pid and "
                "--stopped-main-command-line"
            )
    elif args.stopped_main_pid is not None or args.stopped_main_command_line is not None:
        parser.error("Stopped-main fields are only valid with --issue-handoff-receipt")
    (
        context,
        runner,
        interpreter_identity,
        environment_receipt,
        handoff_receipt,
        handoff_sha256,
    ) = preflight(
        args.project_root,
        args.output_directory,
        require_handoff=not (args.preflight_only or args.issue_handoff_receipt),
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PASS_EXTERNAL_SCHEDULING_PREFLIGHT_ONLY_NO_WRITES",
                    "plan_sha256": context.plan_sha256,
                    "source_snapshot_sha256": context.source_snapshot_sha256,
                    "full_plan_row_count": EXPECTED_FULL_ROWS,
                    "target_case_id": TARGET_CASE_ID,
                    "target_row_count": len(context.rows),
                    "jobs": JOBS,
                    "interpreter_identity": interpreter_identity,
                    "environment_receipt": environment_receipt,
                    "existing_completed_marker_count": (
                        context.existing_completed_marker_count
                    ),
                    "runtime_authority": False,
                    "scientific_authority": False,
                    "selection_authority": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.issue_handoff_receipt:
        receipt = issue_external_handoff_receipt(
            context,
            runner,
            stopped_main_pid=args.stopped_main_pid,
            stopped_main_command_line=args.stopped_main_command_line,
            interpreter_identity=interpreter_identity,
            environment_receipt=environment_receipt,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if handoff_receipt is None or handoff_sha256 is None:
        _fail("Execution preflight did not return the fixed handoff binding")
    receipt = execute_external_schedule(
        context,
        runner,
        interpreter_identity=interpreter_identity,
        environment_receipt=environment_receipt,
        handoff_receipt=handoff_receipt,
        handoff_sha256=handoff_sha256,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"HOLD_EXTERNAL_SCHEDULING_INTEGRITY_ERROR: {error}", file=sys.stderr)
        raise SystemExit(3) from error
