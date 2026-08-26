from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = (
    PROJECT_ROOT
    / "artifacts/v21e3r1_v8_work_20260822/"
    "run_frozen_diagnostic_disjoint_case_parallel.py"
)
REAL_PLAN = (
    PROJECT_ROOT
    / "outputs/v21e3r1_v7_exposed_development_diagnostics_20260823/"
    "diagnostic.plan.json"
)


def _load_helper() -> ModuleType:
    name = "_test_v21e3r1_external_disjoint_scheduler"
    spec = importlib.util.spec_from_file_location(name, HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _context(tmp_path: Path, helper: ModuleType, runner: ModuleType):
    output = tmp_path / "matrix"
    output.mkdir()
    plan_path = output / "diagnostic.plan.json"
    plan_path.write_bytes(REAL_PLAN.read_bytes())
    all_row_ids = tuple(
        f"{case_id}__seed-{seed}__arm-{arm.lower()}"
        for case_id in runner.EXPECTED_CASE_IDS
        for seed in runner.SEEDS
        for arm in runner.DIAGNOSTIC_ARMS
    )
    rows = []
    ordinal = 462
    for seed in runner.SEEDS:
        for arm in runner.DIAGNOSTIC_ARMS:
            ordinal += 1
            row_id = (
                f"{helper.TARGET_CASE_ID}__seed-{seed}__arm-{arm.lower()}"
            )
            rows.append(
                helper.RowWork(
                    ordinal=ordinal,
                    row_id=row_id,
                    worker_spec={
                        "schema": "v21e3r1_diagnostic_row_worker_spec_v1",
                        "project_root": str(PROJECT_ROOT),
                        "case_id": helper.TARGET_CASE_ID,
                        "family": "MOTSP",
                        "size": 500,
                        "case_path": str(tmp_path / "case.json"),
                        "case_artifact_sha256": "a" * 64,
                        "objective_lower_bounds": [0.0, 0.0],
                        "objective_upper_bounds": [1.0, 1.0],
                        "reference_directions": [[0.5, 0.5]],
                        "seed": seed,
                        "arm_id": arm,
                        "charged_evaluation_budget": 2000,
                        "checkpoint_period": 200,
                        "source_snapshot_sha256": (
                            helper.EXPECTED_SOURCE_ROOT_SHA256
                        ),
                        "plan_sha256": helper.EXPECTED_PLAN_SHA256,
                    },
                )
            )
    assert ordinal == 504 and len(rows) == 42
    return helper.FrozenContext(
        project_root=PROJECT_ROOT,
        output=output.resolve(),
        plan_path=plan_path.resolve(),
        plan_sha256=helper.EXPECTED_PLAN_SHA256,
        source_snapshot_sha256=helper.EXPECTED_SOURCE_ROOT_SHA256,
        runner_path=(PROJECT_ROOT / helper.RUNNER_RELATIVE).resolve(),
        runner_sha256=helper.EXPECTED_RUNNER_SHA256,
        row_timeout_seconds=1800,
        existing_completed_marker_count=0,
        all_row_ids=all_row_ids,
        rows=tuple(rows),
    )


def _install_fake_child(
    runner: ModuleType,
    *,
    corrupt_result: bool = False,
    inject_intruder: bool = False,
    tamper_spec_after_child: bool = False,
) -> dict[str, int]:
    state = {"active": 0, "max_active": 0, "calls": 0}
    lock = threading.Lock()
    intruder_written = False

    def fake_run_child(
        spec_path: Path, *, project_root: Path, timeout_seconds: int
    ) -> None:
        nonlocal intruder_written
        assert project_root == PROJECT_ROOT
        assert timeout_seconds == 1800
        spec = runner._load_json_object(spec_path)
        attempt = spec_path.parent
        with lock:
            state["active"] += 1
            state["calls"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            if inject_intruder and not intruder_written:
                intruder_written = True
                (attempt.parent / "attempt-0002").mkdir()
        try:
            time.sleep(0.01)
            row = {
                "schema": "v21e3r1_exposed_development_diagnostic_row_v2",
                "status": "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY",
                "case_id": spec["case_id"],
                "family": spec["family"],
                "size": spec["size"],
                "seed": spec["seed"],
                "arm_id": spec["arm_id"],
                "charged_evaluation_budget": spec["charged_evaluation_budget"],
                "checkpoint_period": spec["checkpoint_period"],
                "case_artifact_sha256": spec["case_artifact_sha256"],
                "source_snapshot_sha256": spec["source_snapshot_sha256"],
                "plan_sha256": spec["plan_sha256"],
                "selection_entropy_release": "PROHIBITED",
                "confirmation_materialization": "PROHIBITED",
                "formal_materialization": "PROHIBITED",
            }
            runner._exclusive_json(attempt / "row.json", row)
            runner._exclusive_json(attempt / "diagnostic.json", {"row": row})
            (attempt / "trace.sqlite3").write_bytes(b"fake sealed trace")
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
            if corrupt_result:
                result["row_sha256"] = "f" * 64
            runner._exclusive_json(attempt / "worker.result.json", result)
            if tamper_spec_after_child:
                spec["seed"] = int(spec["seed"]) + 1
                spec_path.write_text(json.dumps(spec), encoding="utf-8")
        finally:
            with lock:
                state["active"] -= 1

    runner._run_child = fake_run_child
    return state


def _empty_process_scan(helper: ModuleType, output: Path) -> dict[str, object]:
    core: dict[str, object] = {
        "schema": "v21e3r1_original_diagnostic_process_scan_v1",
        "method": "WIN32_PROCESS_CIM_COMMAND_LINE_AND_PARENT_PID",
        "output_root": output.as_posix(),
        "driver_processes": [],
        "worker_processes": [],
        "unknown_runner_processes": [],
    }
    scan = dict(core)
    scan["scan_payload_sha256"] = hashlib.sha256(
        helper.runner_canonical_bytes(core)
    ).hexdigest()
    return scan


def _conflicting_process_scan(
    helper: ModuleType, output: Path, *, kind: str
) -> dict[str, object]:
    scan = _empty_process_scan(helper, output)
    record: dict[str, object] = {
        "pid": 12345,
        "parent_pid": 54321,
        "executable_path": helper.EXPECTED_INTERPRETER_PATH.as_posix(),
        "command_line_sha256": "b" * 64,
    }
    if kind == "worker":
        record["worker_spec_path"] = (
            output / "attempts/foreign/attempt-0001/worker.spec.json"
        ).resolve().as_posix()
        scan["worker_processes"] = [record]
    elif kind == "driver":
        scan["driver_processes"] = [record]
    else:
        scan["unknown_runner_processes"] = [record]
    core = dict(scan)
    core.pop("scan_payload_sha256")
    scan["scan_payload_sha256"] = hashlib.sha256(
        helper.runner_canonical_bytes(core)
    ).hexdigest()
    return scan


def _execution_bindings(
    helper: ModuleType,
    runner: ModuleType,
    context,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], str]:
    monkeypatch.setattr(
        helper,
        "_scan_relevant_processes",
        lambda output: _empty_process_scan(helper, output),
    )
    monkeypatch.setattr(helper, "_verify_execution_boundary", lambda *args, **kwargs: None)
    interpreter = helper._exact_interpreter_identity()
    environment = helper._execution_environment_receipt(PROJECT_ROOT, interpreter)
    command = (
        f'"{helper.EXPECTED_INTERPRETER_PATH}" '
        f'"{PROJECT_ROOT / helper.RUNNER_RELATIVE}" '
        f'--project-root "{PROJECT_ROOT}" '
        f'--output-directory "{context.output}" --resume'
    )
    handoff = helper.issue_external_handoff_receipt(
        context,
        runner,
        stopped_main_pid=999_999,
        stopped_main_command_line=command,
        interpreter_identity=interpreter,
        environment_receipt=environment,
    )
    handoff_sha256 = runner._sha256(context.output / helper.HANDOFF_NAME)
    return interpreter, environment, handoff, handoff_sha256


def test_helper_is_static_external_scheduling_only() -> None:
    source = HELPER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "mo_nco" not in imported_roots
    assert "ThreadPoolExecutor(" in source
    assert "max_workers=JOBS" in source
    assert "JOBS = 4" in source
    assert "_run_child(" in source
    assert "_completed_payload(" in source
    assert "HELPER_INSTANCE_CLAIM_NOT_A_MAIN_RUNNER_LOCK" in source
    assert "Get-CimInstance Win32_Process" in source
    assert "EXPECTED_INTERPRETER_PATH = Path(r\"C:\\miniconda3\\python.exe\")" in source
    assert "runtime_authority\": False" in source
    assert "selection_authority\": False" in source


def test_wrong_ssm_interpreter_fails_before_any_evidence_write() -> None:
    output = (
        PROJECT_ROOT
        / "outputs/v21e3r1_v7_exposed_development_diagnostics_20260823"
    )
    before = sorted(path.name for path in output.glob("external-scheduling.*"))
    completed = subprocess.run(
        [
            r"C:\miniconda3\envs\ssm_env\python.exe",
            str(HELPER_PATH),
            "--project-root",
            str(PROJECT_ROOT),
            "--output-directory",
            str(output),
            "--preflight-only",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 3
    assert "exact historical main-job interpreter" in completed.stderr
    assert sorted(path.name for path in output.glob("external-scheduling.*")) == before


@pytest.mark.parametrize("field", ["sha256", "version"])
def test_interpreter_byte_or_version_drift_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    helper = _load_helper()
    if field == "sha256":
        monkeypatch.setattr(helper, "EXPECTED_INTERPRETER_SHA256", "0" * 64)
    else:
        monkeypatch.setattr(helper, "EXPECTED_PYTHON_VERSION", "3.13.12-attacker")

    with pytest.raises(RuntimeError, match="bytes/version drifted"):
        helper._exact_interpreter_identity()


@pytest.mark.parametrize("kind", ["driver", "worker", "unknown"])
def test_active_original_process_blocks_before_claim_with_zero_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    helper = _load_helper()
    runner = helper._load_frozen_runner(PROJECT_ROOT)
    context = _context(tmp_path, helper, runner)
    monkeypatch.setattr(
        helper,
        "_scan_relevant_processes",
        lambda output: _conflicting_process_scan(helper, output, kind=kind),
    )

    with pytest.raises(RuntimeError, match="Active original"):
        helper._assert_no_conflicting_original_processes(context.output)

    assert list(context.output.glob("external-scheduling.*")) == []
    assert not (context.output / "attempts").exists()


def test_process_classifier_accepts_relative_runner_with_absolute_output_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    helper = _load_helper()
    output = (tmp_path / "matrix").resolve()
    command = (
        f'"{helper.EXPECTED_INTERPRETER_PATH}" '
        f'"{helper.RUNNER_RELATIVE.as_posix()}" '
        f'--output-directory "{output}" --resume'
    )
    assert helper._classify_runner_command(command, output) == ("driver", None)
    duplicate = command + f' --output-directory "{output}"'
    with pytest.raises(RuntimeError, match="Duplicate runner option"):
        helper._classify_runner_command(duplicate, output)


def test_process_classifier_resolves_relative_output_only_with_explicit_process_cwd(
    tmp_path: Path,
) -> None:
    helper = _load_helper()
    process_cwd = (tmp_path / "original-driver-cwd").resolve()
    output = (process_cwd / "matrix").resolve()
    command = (
        f'"{helper.EXPECTED_INTERPRETER_PATH}" '
        f'"{PROJECT_ROOT / helper.RUNNER_RELATIVE}" '
        '--output-directory "matrix" --resume'
    )

    assert helper._classify_runner_command(
        command, output, process_cwd=process_cwd
    ) == ("driver", None)
    assert helper._classify_runner_command(command, output) == ("unknown", None)
    assert (
        helper._classify_runner_command(
            command,
            output,
            process_cwd=(tmp_path / "proven-other-cwd").resolve(),
        )
        is None
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "ijoc_submission_v21e3r1.scripts.run_v21e3r1_development_diagnostics",
        "run_v21e3r1_development_diagnostics",
    ],
)
def test_process_classifier_recognizes_real_frozen_module_driver(
    tmp_path: Path, module_name: str
) -> None:
    helper = _load_helper()
    output = (tmp_path / "matrix").resolve()
    command = (
        f'"{helper.EXPECTED_INTERPRETER_PATH}" -m {module_name} '
        f'--output-directory "{output}" --resume'
    )

    assert helper._classify_runner_command(command, output) == ("driver", None)


@pytest.mark.parametrize("kind", ["attempt", "marker", "claim"])
def test_preexisting_target_or_claim_is_refused(
    tmp_path: Path, kind: str
) -> None:
    helper = _load_helper()
    runner = helper._load_frozen_runner(PROJECT_ROOT)
    context = _context(tmp_path, helper, runner)
    row = context.rows[0]
    if kind == "attempt":
        (context.output / "attempts" / row.row_id / "attempt-0001").mkdir(
            parents=True
        )
    elif kind == "marker":
        marker = context.output / "completed" / f"{row.row_id}.json"
        marker.parent.mkdir(parents=True)
        marker.write_text("{}", encoding="utf-8")
    else:
        (context.output / helper.CLAIM_NAME).write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="target|claim|marker"):
        helper._assert_initial_target_unclaimed(
            context, runner, allow_handoff_absent=True
        )


def test_missing_handoff_is_refused_before_helper_instance_claim(
    tmp_path: Path,
) -> None:
    helper = _load_helper()
    runner = helper._load_frozen_runner(PROJECT_ROOT)
    context = _context(tmp_path, helper, runner)

    with pytest.raises(RuntimeError, match="handoff receipt is absent"):
        helper._assert_initial_target_unclaimed(context, runner)

    assert not (context.output / helper.CLAIM_NAME).exists()


@pytest.mark.parametrize("mutation", ["extra_key", "stale_prefix", "payload_tamper"])
def test_handoff_extra_stale_or_tampered_payload_is_integrity_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    helper = _load_helper()
    runner = helper._load_frozen_runner(PROJECT_ROOT)
    context = _context(tmp_path, helper, runner)
    interpreter, environment, _, _ = _execution_bindings(
        helper, runner, context, monkeypatch
    )
    handoff_path = context.output / helper.HANDOFF_NAME
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if mutation == "extra_key":
        handoff["attacker"] = False
    elif mutation == "stale_prefix":
        handoff["completed_prefix"]["completed_marker_count"] = 1
    else:
        handoff["stopped_main_pid"] = 123
    if mutation != "payload_tamper":
        core = dict(handoff)
        core.pop("receipt_payload_sha256")
        handoff["receipt_payload_sha256"] = hashlib.sha256(
            helper.runner_canonical_bytes(core)
        ).hexdigest()
    handoff_path.write_bytes(helper.runner_exclusive_json_bytes(handoff))

    with pytest.raises(RuntimeError):
        helper._validate_handoff_receipt(
            context,
            runner,
            interpreter_identity=interpreter,
            environment_receipt=environment,
        )

    assert not (context.output / helper.CLAIM_NAME).exists()


def test_small_fake_42_row_schedule_uses_four_workers_and_seals_original_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    runner = helper._load_frozen_runner(PROJECT_ROOT)
    context = _context(tmp_path, helper, runner)
    interpreter, environment, handoff, handoff_sha256 = _execution_bindings(
        helper, runner, context, monkeypatch
    )
    state = _install_fake_child(runner)

    receipt = helper.execute_external_schedule(
        context,
        runner,
        interpreter_identity=interpreter,
        environment_receipt=environment,
        handoff_receipt=handoff,
        handoff_sha256=handoff_sha256,
    )

    assert state["calls"] == 42
    assert 2 <= state["max_active"] <= 4
    assert receipt["status"] == "PASS_EXTERNAL_SCHEDULING_ONLY_TARGET_42"
    assert receipt["jobs"] == 4
    assert receipt["completed_marker_count"] == 42
    assert [item["ordinal"] for item in receipt["completed_markers"]] == list(
        range(463, 505)
    )
    assert receipt["runtime_authority"] is False
    assert receipt["scientific_authority"] is False
    assert receipt["selection_authority"] is False
    assert (context.output / helper.CLAIM_NAME).is_file()
    assert (context.output / helper.RECEIPT_NAME).is_file()
    assert (context.output / helper.RECEIPT_SEAL_NAME).is_file()
    assert len(list((context.output / "completed").glob("*.json"))) == 42
    core = dict(receipt)
    payload_sha256 = core.pop("receipt_payload_sha256")
    assert payload_sha256 == hashlib.sha256(
        helper.runner_canonical_bytes(core)
    ).hexdigest()
    for row in context.rows:
        completed = runner._completed_payload(context.output, row.row_id)
        assert completed is not None
        assert completed["plan_sha256"] == helper.EXPECTED_PLAN_SHA256


def test_driver_appearing_after_claim_blocks_before_any_attempt_and_writes_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    runner = helper._load_frozen_runner(PROJECT_ROOT)
    context = _context(tmp_path, helper, runner)
    interpreter, environment, handoff, handoff_sha256 = _execution_bindings(
        helper, runner, context, monkeypatch
    )
    calls = 0

    def race_scan(output: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _empty_process_scan(helper, output)
        return _conflicting_process_scan(helper, output, kind="driver")

    monkeypatch.setattr(helper, "_scan_relevant_processes", race_scan)

    with pytest.raises(RuntimeError, match="stopped fail-closed"):
        helper.execute_external_schedule(
            context,
            runner,
            interpreter_identity=interpreter,
            environment_receipt=environment,
            handoff_receipt=handoff,
            handoff_sha256=handoff_sha256,
        )

    assert (context.output / helper.CLAIM_NAME).is_file()
    assert (context.output / helper.FAILURE_NAME).is_file()
    assert not (context.output / "attempts").exists()
    assert not (context.output / helper.RECEIPT_NAME).exists()


def test_claim_fsync_failure_retains_claim_and_writes_durable_failure_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    runner = helper._load_frozen_runner(PROJECT_ROOT)
    context = _context(tmp_path, helper, runner)
    interpreter, environment, handoff, handoff_sha256 = _execution_bindings(
        helper, runner, context, monkeypatch
    )
    original_fsync_file = helper._fsync_file

    def fail_claim_fsync(path: Path) -> None:
        if Path(path).resolve() == (context.output / helper.CLAIM_NAME).resolve():
            raise OSError("injected helper-instance claim fsync failure")
        original_fsync_file(path)

    monkeypatch.setattr(helper, "_fsync_file", fail_claim_fsync)

    with pytest.raises(RuntimeError, match="stopped fail-closed"):
        helper.execute_external_schedule(
            context,
            runner,
            interpreter_identity=interpreter,
            environment_receipt=environment,
            handoff_receipt=handoff,
            handoff_sha256=handoff_sha256,
        )

    claim_path = context.output / helper.CLAIM_NAME
    failure_path = context.output / helper.FAILURE_NAME
    assert claim_path.is_file()
    assert failure_path.is_file()
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["exception_type"] == "OSError"
    assert failure["manual_audit_required"] is True
    assert failure["helper_instance_claim_sha256"] == runner._sha256(claim_path)
    assert failure["owned_attempts"] == []
    assert not (context.output / "attempts").exists()
    assert not (context.output / helper.RECEIPT_NAME).exists()
    assert not (context.output / helper.RECEIPT_SEAL_NAME).exists()


def test_invalid_worker_result_never_materializes_completed_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    runner = helper._load_frozen_runner(PROJECT_ROOT)
    context = _context(tmp_path, helper, runner)
    interpreter, environment, handoff, handoff_sha256 = _execution_bindings(
        helper, runner, context, monkeypatch
    )
    _install_fake_child(runner, corrupt_result=True)

    with pytest.raises(RuntimeError, match="stopped fail-closed"):
        helper.execute_external_schedule(
            context,
            runner,
            interpreter_identity=interpreter,
            environment_receipt=environment,
            handoff_receipt=handoff,
            handoff_sha256=handoff_sha256,
        )

    assert (context.output / helper.CLAIM_NAME).is_file()
    assert (context.output / helper.FAILURE_NAME).is_file()
    assert not (context.output / helper.RECEIPT_NAME).exists()
    assert list((context.output / "completed").glob("*.json")) == []


def test_worker_spec_tamper_after_child_is_detected_before_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    runner = helper._load_frozen_runner(PROJECT_ROOT)
    context = _context(tmp_path, helper, runner)
    interpreter, environment, handoff, handoff_sha256 = _execution_bindings(
        helper, runner, context, monkeypatch
    )
    _install_fake_child(runner, tamper_spec_after_child=True)

    with pytest.raises(RuntimeError, match="stopped fail-closed"):
        helper.execute_external_schedule(
            context,
            runner,
            interpreter_identity=interpreter,
            environment_receipt=environment,
            handoff_receipt=handoff,
            handoff_sha256=handoff_sha256,
        )

    assert (context.output / helper.FAILURE_NAME).is_file()
    assert not (context.output / helper.RECEIPT_NAME).exists()
    assert list((context.output / "completed").glob("*.json")) == []


@pytest.mark.parametrize("externally_change_marker", [False, True])
def test_completed_verifier_failure_only_removes_exact_helper_owned_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    externally_change_marker: bool,
) -> None:
    helper = _load_helper()
    runner = helper._load_frozen_runner(PROJECT_ROOT)
    context = _context(tmp_path, helper, runner)
    interpreter, environment, handoff, handoff_sha256 = _execution_bindings(
        helper, runner, context, monkeypatch
    )
    _install_fake_child(runner)
    original_completed_payload = runner._completed_payload

    def reject_target_marker(output: Path, row_id: str):
        marker = output / "completed" / f"{row_id}.json"
        if row_id.startswith(helper.TARGET_CASE_ID) and marker.is_file():
            if externally_change_marker:
                marker.write_text('{"external_change":true}', encoding="utf-8")
            raise RuntimeError("injected frozen completed verifier failure")
        return original_completed_payload(output, row_id)

    monkeypatch.setattr(runner, "_completed_payload", reject_target_marker)

    with pytest.raises(RuntimeError, match="stopped fail-closed"):
        helper.execute_external_schedule(
            context,
            runner,
            interpreter_identity=interpreter,
            environment_receipt=environment,
            handoff_receipt=handoff,
            handoff_sha256=handoff_sha256,
        )

    failure = json.loads(
        (context.output / helper.FAILURE_NAME).read_text(encoding="utf-8")
    )
    cleanup_statuses = {
        item["cleanup_status"] for item in failure["marker_cleanup_events"]
    }
    if externally_change_marker:
        assert "RETAINED_EXTERNALLY_CHANGED" in cleanup_statuses
        assert list((context.output / "completed").glob("*.json"))
    else:
        assert "REMOVED_EXACT_HELPER_OWNED_FALSE_MARKER" in cleanup_statuses
        assert list((context.output / "completed").glob("*.json")) == []
    assert not (context.output / helper.RECEIPT_NAME).exists()


def test_concurrent_non_helper_attempt_is_detected_and_no_receipt_is_issued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    runner = helper._load_frozen_runner(PROJECT_ROOT)
    context = _context(tmp_path, helper, runner)
    interpreter, environment, handoff, handoff_sha256 = _execution_bindings(
        helper, runner, context, monkeypatch
    )
    _install_fake_child(runner, inject_intruder=True)

    with pytest.raises(RuntimeError, match="stopped fail-closed"):
        helper.execute_external_schedule(
            context,
            runner,
            interpreter_identity=interpreter,
            environment_receipt=environment,
            handoff_receipt=handoff,
            handoff_sha256=handoff_sha256,
        )

    assert (context.output / helper.CLAIM_NAME).is_file()
    assert (context.output / helper.FAILURE_NAME).is_file()
    assert not (context.output / helper.RECEIPT_NAME).exists()
    intruders = list(
        (context.output / "attempts").glob(
            f"{helper.TARGET_CASE_ID}*/attempt-0002"
        )
    )
    assert len(intruders) == 1


@pytest.mark.parametrize("mutation", ["extra", "missing", "bad_payload", "truncated"])
def test_postwrite_payload_validator_rejects_receipt_shape_or_digest_drift(
    tmp_path: Path, mutation: str
) -> None:
    helper = _load_helper()
    runner = helper._load_frozen_runner(PROJECT_ROOT)
    path = tmp_path / "receipt.json"
    core = {"schema": "example", "status": "PASS"}
    receipt = dict(core)
    receipt["receipt_payload_sha256"] = hashlib.sha256(
        helper.runner_canonical_bytes(core)
    ).hexdigest()
    if mutation == "extra":
        receipt["extra"] = False
    elif mutation == "missing":
        receipt.pop("status")
    elif mutation == "bad_payload":
        receipt["status"] = "MUTATED"
    path.write_bytes(helper.runner_exclusive_json_bytes(receipt))
    if mutation == "truncated":
        path.write_bytes(path.read_bytes()[:10])

    with pytest.raises((RuntimeError, json.JSONDecodeError)):
        helper._validate_bound_payload_file(
            path,
            runner,
            expected_keys={"schema", "status", "receipt_payload_sha256"},
            payload_field="receipt_payload_sha256",
            label="test receipt",
        )

