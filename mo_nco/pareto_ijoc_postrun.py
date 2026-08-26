from __future__ import annotations

"""Independent, fail-closed post-run audit for an IJOC matched matrix."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .pareto_ijoc_cold_matrix import (
    INPUT_SCHEMA,
    INVOCATION_SCHEMA,
    RUN_RECEIPT_SCHEMA,
    _bound_path,
    _canonical_digest,
    _dependency_environment_gate,
    _exact_keys,
    _expand_argv,
    _file_digest,
    _run_key,
    _strict_json,
    _study_components,
    _validate_freeze_receipt,
    _validate_plan,
    _validate_runtime_source_manifest,
    _write_json_atomic,
    validate_algorithm_result,
    validate_replay_result,
)


POST_RUN_AUDIT_SCHEMA = "ijoc_post_run_audit_v2"

_INVOCATION_KEYS = {
    "schema",
    "study_sha256",
    "configuration_matrix_sha256",
    "execution_plan_sha256",
    "freeze_receipt_sha256",
    "tail_calibration_suite_receipt",
    "reference_calibration_precommit",
    "reference_calibration_completion_receipt",
    "tail_policy_artifact",
    "dependency_environment_gate",
    "execution_scope",
    "formal_evidence_status",
    "timeout_seconds",
    "sample_period_seconds",
    "workers",
    "selected_run_count",
    "expected_run_count",
    "selection",
    "python_executable",
}
_RUN_RECEIPT_KEYS = {
    "schema",
    "run_key",
    "run_key_sha256",
    "study_sha256",
    "configuration_matrix_sha256",
    "execution_plan_sha256",
    "freeze_receipt_sha256",
    "tail_calibration_suite_receipt",
    "reference_calibration_precommit",
    "reference_calibration_completion_receipt",
    "execution_scope",
    "formal_evidence_status",
    "attempt_number",
    "status",
    "reason",
    "input_artifact",
    "algorithm_process",
    "algorithm_result",
    "replay_process",
    "replay_result",
}
_PROCESS_KEYS = {
    "argv",
    "started_utc",
    "finished_utc",
    "wall_time_seconds",
    "exit_code",
    "timed_out",
    "interrupted",
    "spawn_error",
    "sampled_peak_process_tree_rss_bytes",
    "resource_measurement_status",
    "stdout",
    "stderr",
}
_FILE_RECEIPT_KEYS = {"path", "sha256", "bytes"}
_INPUT_KEYS = {
    "schema",
    "study_sha256",
    "configuration_matrix_sha256",
    "execution_plan_sha256",
    "freeze_receipt_sha256",
    "tail_calibration_suite_receipt",
    "reference_calibration_precommit",
    "reference_calibration_completion_receipt",
    "tail_policy_artifact",
    "run_key",
    "run_key_sha256",
    "configuration",
    "configuration_sha256",
    "instance_artifact",
    "anytime_checkpoint_period",
}


@dataclass(frozen=True)
class IJOCPostRunResult:
    audit_path: Path
    expected_run_count: int
    observed_unique_run_count: int
    valid_run_count: int
    missing_run_count: int
    duplicate_run_count: int
    invalid_run_count: int
    retry_run_count: int
    prior_attempt_count: int
    formal_matched_matrix_gate: str
    resource_efficiency_gate: str
    evidence_status: str
    submission_verdict: str


_INCOMPLETE_ATTEMPT_FILENAMES = {
    "input.json",
    "algorithm.stdout",
    "algorithm.stderr",
}


def _audit_attempt_history(
    *,
    result_root: Path,
    run_directory: Path,
    run_key_sha: str,
    terminal_attempt_number: int,
    terminal_input_path: Path,
    terminal_input_sha: str,
) -> dict[str, object]:
    attempts_root = (run_directory / "attempts").resolve(strict=True)
    terminal_attempt = (
        attempts_root / f"{terminal_attempt_number:06d}"
    ).resolve(strict=True)
    if terminal_input_path.parent != terminal_attempt:
        raise ValueError("Terminal input is not stored in the terminal attempt.")

    observed_numbers: list[int] = []
    unexpected_entries: list[str] = []
    attempt_directories: dict[int, Path] = {}
    for child in sorted(attempts_root.iterdir(), key=lambda item: item.name):
        relative = child.relative_to(result_root).as_posix()
        if (
            not child.is_dir()
            or len(child.name) != 6
            or not child.name.isascii()
            or not child.name.isdigit()
            or int(child.name) <= 0
        ):
            unexpected_entries.append(relative)
            continue
        number = int(child.name)
        observed_numbers.append(number)
        attempt_directories[number] = child.resolve(strict=True)

    expected_numbers = list(range(1, terminal_attempt_number + 1))
    structural_errors: list[str] = []
    if unexpected_entries:
        structural_errors.append("unexpected_attempt_entries")
    if observed_numbers != expected_numbers:
        structural_errors.append("noncontiguous_or_extra_attempt_numbers")
    if attempt_directories.get(terminal_attempt_number) != terminal_attempt:
        structural_errors.append("terminal_attempt_directory_mismatch")

    prior_attempts: list[dict[str, object]] = []
    for number in observed_numbers:
        if number >= terminal_attempt_number:
            continue
        attempt = attempt_directories[number]
        artifacts: list[dict[str, object]] = []
        unexpected_subdirectories: list[str] = []
        for artifact in sorted(attempt.rglob("*")):
            if artifact.is_dir():
                unexpected_subdirectories.append(
                    artifact.relative_to(result_root).as_posix()
                )
                continue
            if not artifact.is_file():
                unexpected_subdirectories.append(
                    artifact.relative_to(result_root).as_posix()
                )
                continue
            artifacts.append(
                {
                    "path": artifact.relative_to(result_root).as_posix(),
                    "sha256": _file_digest(artifact),
                    "bytes": artifact.stat().st_size,
                }
            )
        input_path = attempt / "input.json"
        input_sha = _file_digest(input_path) if input_path.is_file() else None
        input_matches = input_sha == terminal_input_sha
        artifact_by_name = {
            Path(str(artifact["path"])).name: artifact
            for artifact in artifacts
        }
        exact_incomplete_artifact_set = (
            set(artifact_by_name) == _INCOMPLETE_ATTEMPT_FILENAMES
        )
        empty_algorithm_logs = (
            exact_incomplete_artifact_set
            and artifact_by_name["algorithm.stdout"]["bytes"] == 0
            and artifact_by_name["algorithm.stderr"]["bytes"] == 0
        )
        result_artifacts = [
            artifact
            for artifact in artifacts
            if Path(str(artifact["path"])).name
            not in _INCOMPLETE_ATTEMPT_FILENAMES
        ]
        if unexpected_subdirectories:
            result_artifacts.extend(
                {
                    "path": path,
                    "sha256": None,
                    "bytes": None,
                }
                for path in unexpected_subdirectories
            )
        prior_attempts.append(
            {
                "attempt_number": number,
                "path": attempt.relative_to(result_root).as_posix(),
                "termination_reason": "UNKNOWN_UNRECORDED",
                "input_artifact": (
                    None
                    if input_sha is None
                    else {
                        "path": input_path.relative_to(result_root).as_posix(),
                        "sha256": input_sha,
                        "bytes": input_path.stat().st_size,
                    }
                ),
                "input_matches_terminal": input_matches,
                "exact_incomplete_artifact_set": (
                    exact_incomplete_artifact_set
                ),
                "empty_algorithm_logs": empty_algorithm_logs,
                "result_artifact_status": (
                    "NO_RESULT_ARTIFACT"
                    if not result_artifacts
                    else "UNBOUND_RESULT_OR_UNKNOWN_ARTIFACT_PRESENT"
                ),
                "artifacts": artifacts,
                "unexpected_subdirectories": unexpected_subdirectories,
            }
        )

    quality_eligible = (
        not structural_errors
        and all(
            bool(item["input_matches_terminal"])
            and bool(item["exact_incomplete_artifact_set"])
            and bool(item["empty_algorithm_logs"])
            and item["result_artifact_status"] == "NO_RESULT_ARTIFACT"
            for item in prior_attempts
        )
    )
    return {
        "run_key_sha256": run_key_sha,
        "terminal_attempt_number": terminal_attempt_number,
        "observed_attempt_numbers": observed_numbers,
        "prior_attempt_count": len(prior_attempts),
        "prior_attempts": prior_attempts,
        "unexpected_attempt_entries": unexpected_entries,
        "structural_errors": structural_errors,
        "retry_quality_eligible": quality_eligible,
    }


def _file_binding(
    root: Path,
    value: object,
    *,
    label: str,
    with_size: bool,
) -> tuple[Path, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    expected = _FILE_RECEIPT_KEYS if with_size else {"path", "sha256"}
    _exact_keys(value, expected, label)
    path, digest = _bound_path(
        root,
        {"path": value.get("path"), "sha256": value.get("sha256")},
        label=label,
    )
    if with_size:
        size = value.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"{label}.bytes must be a nonnegative integer.")
        if path.stat().st_size != size:
            raise ValueError(f"{label}.bytes does not match the file.")
    return path, digest


def _validate_process_record(
    value: object,
    *,
    run_directory: Path,
    expected_attempt_directory: Path,
    expected_argv: list[str],
    label: str,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is missing.")
    _exact_keys(value, _PROCESS_KEYS, label)
    if value.get("argv") != expected_argv:
        raise ValueError(f"{label} argv differs from the frozen execution plan.")
    if value.get("exit_code") != 0:
        raise ValueError(f"{label} did not exit successfully.")
    if value.get("timed_out") is not False:
        raise ValueError(f"{label} timed out.")
    if value.get("interrupted") is not False:
        raise ValueError(f"{label} was interrupted.")
    if value.get("spawn_error") is not None:
        raise ValueError(f"{label} has a spawn error.")
    wall = value.get("wall_time_seconds")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or wall < 0:
        raise ValueError(f"{label} wall time is invalid.")
    peak = value.get("sampled_peak_process_tree_rss_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak <= 0:
        raise ValueError(f"{label} lacks a positive process-tree peak RSS.")
    if value.get("resource_measurement_status") != "PASS":
        raise ValueError(f"{label} resource measurement did not PASS.")
    for timestamp_key in ("started_utc", "finished_utc"):
        timestamp = value.get(timestamp_key)
        if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
            raise ValueError(f"{label} {timestamp_key} is invalid.")
    stdout_path, _ = _file_binding(
        run_directory,
        value.get("stdout"),
        label=f"{label} stdout",
        with_size=True,
    )
    stderr_path, _ = _file_binding(
        run_directory,
        value.get("stderr"),
        label=f"{label} stderr",
        with_size=True,
    )
    if (
        stdout_path.parent != expected_attempt_directory
        or stderr_path.parent != expected_attempt_directory
    ):
        raise ValueError(f"{label} logs are not stored in the terminal attempt.")


def _validate_input(
    input_path: Path,
    *,
    run_key: Mapping[str, object],
    run_key_sha: str,
    study_sha: str,
    config_sha: str,
    plan_sha: str,
    freeze_receipt_sha: str,
    configuration: Mapping[str, Any],
    instance_path: Path,
    instance_sha: str,
    tail_calibration_receipt_path: Path,
    tail_calibration_receipt_sha: str,
    reference_precommit_path: Path,
    reference_precommit_sha: str,
    reference_completion_path: Path,
    reference_completion_sha: str,
    tail_policy_path: Path,
    tail_policy_sha: str,
    checkpoint_period: int,
) -> None:
    payload, _, _ = _strict_json(input_path)
    _exact_keys(payload, _INPUT_KEYS, "cold-process input")
    if payload.get("schema") != INPUT_SCHEMA:
        raise ValueError("Cold-process input schema mismatch.")
    if payload.get("study_sha256") != study_sha:
        raise ValueError("Cold-process input study hash mismatch.")
    if payload.get("configuration_matrix_sha256") != config_sha:
        raise ValueError("Cold-process input matrix hash mismatch.")
    if payload.get("execution_plan_sha256") != plan_sha:
        raise ValueError("Cold-process input plan hash mismatch.")
    if payload.get("freeze_receipt_sha256") != freeze_receipt_sha:
        raise ValueError("Cold-process input freeze-receipt hash mismatch.")
    for key, expected_path, expected_sha in (
        (
            "tail_calibration_suite_receipt",
            tail_calibration_receipt_path,
            tail_calibration_receipt_sha,
        ),
        (
            "reference_calibration_precommit",
            reference_precommit_path,
            reference_precommit_sha,
        ),
        (
            "reference_calibration_completion_receipt",
            reference_completion_path,
            reference_completion_sha,
        ),
        ("tail_policy_artifact", tail_policy_path, tail_policy_sha),
    ):
        binding = payload.get(key)
        if not isinstance(binding, dict):
            raise ValueError(f"Cold-process input {key} is missing.")
        _exact_keys(binding, {"path", "sha256"}, f"input {key}")
        try:
            observed_path = Path(str(binding.get("path"))).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError(f"Cold-process input {key} path is invalid.") from error
        if observed_path != expected_path or binding.get("sha256") != expected_sha:
            raise ValueError(f"Cold-process input {key} binding mismatch.")
        if _file_digest(observed_path) != expected_sha:
            raise ValueError(f"Cold-process input {key} bytes changed.")
    if payload.get("run_key") != run_key:
        raise ValueError("Cold-process input run key mismatch.")
    if payload.get("run_key_sha256") != run_key_sha:
        raise ValueError("Cold-process input run-key hash mismatch.")
    readable = payload.get("configuration")
    if readable != configuration:
        raise ValueError("Cold-process input readable configuration mismatch.")
    if payload.get("configuration_sha256") != _canonical_digest(configuration):
        raise ValueError("Cold-process input configuration hash mismatch.")
    instance = payload.get("instance_artifact")
    if not isinstance(instance, dict):
        raise ValueError("Cold-process input instance binding is missing.")
    _exact_keys(instance, {"path", "sha256"}, "input instance_artifact")
    try:
        bound_input_path = Path(str(instance.get("path"))).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("Cold-process input instance path is invalid.") from error
    if bound_input_path != instance_path or instance.get("sha256") != instance_sha:
        raise ValueError("Cold-process input instance binding mismatch.")
    if _file_digest(bound_input_path) != instance_sha:
        raise ValueError("Cold-process input instance bytes changed.")
    if payload.get("anytime_checkpoint_period") != checkpoint_period:
        raise ValueError("Cold-process input checkpoint period mismatch.")


def audit_ijoc_post_run(
    study_path: str | Path,
    execution_plan_path: str | Path,
    results_directory: str | Path,
    *,
    output_path: str | Path | None = None,
) -> IJOCPostRunResult:
    """Audit a full matrix.  Partial/plumbing invocations always fail closed."""

    study_file = Path(study_path).expanduser().resolve()
    plan_file = Path(execution_plan_path).expanduser().resolve()
    result_root = Path(results_directory).expanduser().resolve()
    study, study_sha, configuration, config_sha, instances = _study_components(
        study_file
    )
    dependency_environment_gate = _dependency_environment_gate(
        study_file, study
    )
    validated_plan, plan_sha, algorithms = _validate_plan(
        plan_file,
        study_sha=study_sha,
        configuration_sha=config_sha,
        instances=instances,
    )
    _validate_runtime_source_manifest(plan_file, validated_plan)
    plan_payload, _, _ = _strict_json(plan_file)
    tail_calibration_receipt_path, tail_calibration_receipt_sha = _bound_path(
        plan_file.parent,
        plan_payload.get("tail_calibration_suite_receipt"),
        label="tail calibration suite receipt",
    )
    reference_precommit_path, reference_precommit_sha = _bound_path(
        plan_file.parent,
        plan_payload.get("reference_calibration_precommit"),
        label="reference calibration precommit",
    )
    reference_completion_path, reference_completion_sha = _bound_path(
        plan_file.parent,
        plan_payload.get("reference_calibration_completion_receipt"),
        label="reference calibration completion receipt",
    )
    tail_policy_path, tail_policy_sha = _bound_path(
        plan_file.parent,
        plan_payload.get("tail_policy_artifact"),
        label="tail policy artifact",
    )
    invocation_path = result_root / "matrix_invocation.json"
    invocation, _, invocation_sha = _strict_json(invocation_path)
    _exact_keys(invocation, _INVOCATION_KEYS, "matrix invocation")
    if invocation.get("schema") != INVOCATION_SCHEMA:
        raise ValueError("Matrix invocation schema mismatch.")
    if invocation.get("dependency_environment_gate") != (
        dependency_environment_gate
    ):
        raise ValueError("Matrix invocation dependency environment gate mismatch.")
    for key, expected in (
        ("study_sha256", study_sha),
        ("configuration_matrix_sha256", config_sha),
        ("execution_plan_sha256", plan_sha),
    ):
        if invocation.get(key) != expected:
            raise ValueError(f"Matrix invocation {key} mismatch.")
    for key, expected_path, expected_sha in (
        (
            "tail_calibration_suite_receipt",
            tail_calibration_receipt_path,
            tail_calibration_receipt_sha,
        ),
        (
            "reference_calibration_precommit",
            reference_precommit_path,
            reference_precommit_sha,
        ),
        (
            "reference_calibration_completion_receipt",
            reference_completion_path,
            reference_completion_sha,
        ),
        ("tail_policy_artifact", tail_policy_path, tail_policy_sha),
    ):
        binding = invocation.get(key)
        if not isinstance(binding, dict):
            raise ValueError(f"Matrix invocation {key} is missing.")
        _exact_keys(binding, {"path", "sha256"}, f"invocation {key}")
        try:
            observed_path = Path(str(binding.get("path"))).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError(f"Matrix invocation {key} path is invalid.") from error
        if observed_path != expected_path or binding.get("sha256") != expected_sha:
            raise ValueError(f"Matrix invocation {key} binding mismatch.")
        if _file_digest(observed_path) != expected_sha:
            raise ValueError(f"Matrix invocation {key} bytes changed.")

    rows_raw = configuration.get("rows")
    if not isinstance(rows_raw, list):
        raise ValueError("Configuration matrix rows must be an array.")
    expected_rows: dict[str, Mapping[str, Any]] = {}
    expected_keys: dict[str, dict[str, object]] = {}
    for raw in rows_raw:
        if not isinstance(raw, dict):
            raise ValueError("Configuration matrix row must be an object.")
        run_key = _run_key(raw)
        run_key_sha = _canonical_digest(run_key)
        if run_key_sha in expected_rows:
            raise ValueError("Configuration matrix contains duplicate run keys.")
        expected_rows[run_key_sha] = raw
        expected_keys[run_key_sha] = run_key
    if {
        str(run_key["algorithm"]) for run_key in expected_keys.values()
    } != set(algorithms):
        raise ValueError(
            "Execution-plan algorithms do not exactly match the matrix algorithms."
        )
    freeze_receipt_sha = _validate_freeze_receipt(
        study_path=study_file,
        plan_path=plan_file,
        study_sha=study_sha,
        configuration_sha=config_sha,
        plan_sha=plan_sha,
        expected_run_count=len(expected_rows),
    )
    if invocation.get("freeze_receipt_sha256") != freeze_receipt_sha:
        raise ValueError("Matrix invocation freeze-receipt hash mismatch.")

    checkpoint_period = study.get("anytime_checkpoint_period")
    if (
        isinstance(checkpoint_period, bool)
        or not isinstance(checkpoint_period, int)
        or checkpoint_period <= 0
    ):
        raise ValueError("Study checkpoint period is invalid.")
    python_executable = invocation.get("python_executable")
    if not isinstance(python_executable, str) or not python_executable:
        raise ValueError("Invocation python_executable is invalid.")
    workers = invocation.get("workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("Invocation workers is invalid.")

    observed: dict[str, list[str]] = {}
    valid_run_keys: set[str] = set()
    invalid_runs: list[dict[str, str]] = []
    unexpected_receipts: list[str] = []
    retry_run_keys: set[str] = set()
    quality_retry_failures: list[dict[str, object]] = []
    retry_histories: list[dict[str, object]] = []
    prior_attempt_count = 0
    receipt_paths = sorted(result_root.rglob("terminal_receipt.json"))
    for receipt_path in receipt_paths:
        relative_receipt = receipt_path.relative_to(result_root).as_posix()
        run_key_sha_for_error = "UNPARSEABLE"
        try:
            receipt, _, _ = _strict_json(receipt_path)
            _exact_keys(receipt, _RUN_RECEIPT_KEYS, "terminal receipt")
            if receipt.get("schema") != RUN_RECEIPT_SCHEMA:
                raise ValueError("Terminal receipt schema mismatch.")
            run_key = receipt.get("run_key")
            if not isinstance(run_key, dict):
                raise ValueError("Terminal receipt run_key is missing.")
            run_key_sha = _canonical_digest(run_key)
            run_key_sha_for_error = run_key_sha
            observed.setdefault(run_key_sha, []).append(relative_receipt)
            if run_key_sha not in expected_rows:
                unexpected_receipts.append(relative_receipt)
                continue
            if receipt.get("run_key_sha256") != run_key_sha:
                raise ValueError("Terminal receipt run-key hash mismatch.")
            expected_directory = result_root / "runs" / run_key_sha
            if receipt_path.parent != expected_directory:
                raise ValueError("Terminal receipt is stored under the wrong run path.")
            expected_key = expected_keys[run_key_sha]
            if run_key != expected_key:
                raise ValueError("Terminal receipt run key differs from the matrix.")
            for key, expected in (
                ("study_sha256", study_sha),
                ("configuration_matrix_sha256", config_sha),
                ("execution_plan_sha256", plan_sha),
                ("freeze_receipt_sha256", freeze_receipt_sha),
            ):
                if receipt.get(key) != expected:
                    raise ValueError(f"Terminal receipt {key} mismatch.")
            for key, expected_path, expected_sha in (
                (
                    "tail_calibration_suite_receipt",
                    tail_calibration_receipt_path,
                    tail_calibration_receipt_sha,
                ),
                (
                    "reference_calibration_precommit",
                    reference_precommit_path,
                    reference_precommit_sha,
                ),
                (
                    "reference_calibration_completion_receipt",
                    reference_completion_path,
                    reference_completion_sha,
                ),
            ):
                binding = receipt.get(key)
                if not isinstance(binding, dict):
                    raise ValueError(f"Terminal receipt {key} is missing.")
                _exact_keys(binding, {"path", "sha256"}, f"terminal {key}")
                observed_path = Path(str(binding.get("path"))).resolve(
                    strict=True
                )
                if (
                    observed_path != expected_path
                    or binding.get("sha256") != expected_sha
                    or _file_digest(observed_path) != expected_sha
                ):
                    raise ValueError(
                        f"Terminal receipt {key} binding mismatch."
                    )
            if receipt.get("execution_scope") != "formal_candidate":
                raise ValueError("Terminal receipt is plumbing-only, not formal.")
            if receipt.get("formal_evidence_status") != "PENDING_POST_RUN_AUDIT":
                raise ValueError("Terminal receipt has no pending formal audit state.")
            if receipt.get("status") != "SUCCESS":
                raise ValueError(
                    f"Terminal status is {receipt.get('status')!r}, not SUCCESS."
                )
            if receipt.get("reason") is not None:
                raise ValueError("Successful terminal receipt contains a failure reason.")
            attempt_number = receipt.get("attempt_number")
            if (
                isinstance(attempt_number, bool)
                or not isinstance(attempt_number, int)
                or attempt_number <= 0
            ):
                raise ValueError("Terminal receipt attempt number is invalid.")
            attempt = (
                expected_directory / "attempts" / f"{attempt_number:06d}"
            ).resolve()
            if not attempt.is_dir():
                raise ValueError("Terminal receipt attempt directory is missing.")

            input_path, input_sha = _file_binding(
                expected_directory,
                receipt.get("input_artifact"),
                label="terminal input artifact",
                with_size=False,
            )
            row = expected_rows[run_key_sha]
            readable = row.get("configuration")
            if not isinstance(readable, dict):
                raise ValueError("Matrix readable configuration is invalid.")
            instance_path, instance_sha = instances[str(expected_key["case_id"])]
            _validate_input(
                input_path,
                run_key=expected_key,
                run_key_sha=run_key_sha,
                study_sha=study_sha,
                config_sha=config_sha,
                plan_sha=plan_sha,
                freeze_receipt_sha=freeze_receipt_sha,
                configuration=readable,
                instance_path=instance_path,
                instance_sha=instance_sha,
                tail_calibration_receipt_path=tail_calibration_receipt_path,
                tail_calibration_receipt_sha=tail_calibration_receipt_sha,
                reference_precommit_path=reference_precommit_path,
                reference_precommit_sha=reference_precommit_sha,
                reference_completion_path=reference_completion_path,
                reference_completion_sha=reference_completion_sha,
                tail_policy_path=tail_policy_path,
                tail_policy_sha=tail_policy_sha,
                checkpoint_period=checkpoint_period,
            )
            if input_sha != _file_digest(input_path):
                raise ValueError("Input artifact receipt hash mismatch.")
            attempt_history = _audit_attempt_history(
                result_root=result_root,
                run_directory=expected_directory,
                run_key_sha=run_key_sha,
                terminal_attempt_number=attempt_number,
                terminal_input_path=input_path,
                terminal_input_sha=input_sha,
            )
            observed_attempt_numbers = attempt_history[
                "observed_attempt_numbers"
            ]
            if attempt_number > 1 or len(observed_attempt_numbers) > 1:
                retry_run_keys.add(run_key_sha)
                retry_histories.append(attempt_history)
                prior_attempt_count += int(
                    attempt_history["prior_attempt_count"]
                )
            if attempt_history["retry_quality_eligible"] is not True:
                quality_retry_failures.append(
                    {
                        "run_key_sha256": run_key_sha,
                        "structural_errors": attempt_history[
                            "structural_errors"
                        ],
                        "prior_attempt_count": attempt_history[
                            "prior_attempt_count"
                        ],
                    }
                )

            algorithm_id = str(expected_key["algorithm"])
            algorithm = algorithms[algorithm_id]
            result_record = receipt.get("algorithm_result")
            if not isinstance(result_record, dict):
                raise ValueError("Terminal algorithm_result binding is missing.")
            _exact_keys(
                result_record,
                {
                    "path",
                    "sha256",
                    "archive_path",
                    "archive_sha256",
                    "checkpoint_path",
                    "checkpoint_sha256",
                },
                "terminal algorithm_result",
            )
            result_path, result_sha = _file_binding(
                expected_directory,
                {
                    "path": result_record.get("path"),
                    "sha256": result_record.get("sha256"),
                },
                label="terminal algorithm result",
                with_size=False,
            )
            if result_path.parent != attempt:
                raise ValueError(
                    "Algorithm result is not stored in the terminal attempt."
                )
            validated_result = validate_algorithm_result(
                result_path,
                expected_run_key=expected_key,
                checkpoint_period=checkpoint_period,
            )
            if result_sha != validated_result["sha256"]:
                raise ValueError("Algorithm result binding hash mismatch.")
            archive_path, archive_sha = _file_binding(
                expected_directory,
                {
                    "path": result_record.get("archive_path"),
                    "sha256": result_record.get("archive_sha256"),
                },
                label="terminal archive",
                with_size=False,
            )
            if archive_path.parent != attempt:
                raise ValueError("Archive is not stored in the terminal attempt.")
            if (
                archive_path != validated_result["archive_path"]
                or archive_sha != validated_result["archive_sha256"]
            ):
                raise ValueError("Terminal archive binding disagrees with the result.")
            checkpoint_path, checkpoint_sha = _file_binding(
                expected_directory,
                {
                    "path": result_record.get("checkpoint_path"),
                    "sha256": result_record.get("checkpoint_sha256"),
                },
                label="terminal checkpoint witnesses",
                with_size=False,
            )
            if checkpoint_path.parent != attempt:
                raise ValueError(
                    "Checkpoint witnesses are not stored in the terminal attempt."
                )
            if (
                checkpoint_path != validated_result["checkpoint_path"]
                or checkpoint_sha != validated_result["checkpoint_sha256"]
            ):
                raise ValueError(
                    "Terminal checkpoint binding disagrees with the result."
                )

            replay_record = receipt.get("replay_result")
            if not isinstance(replay_record, dict):
                raise ValueError("Terminal replay_result binding is missing.")
            replay_path, replay_sha = _file_binding(
                expected_directory,
                replay_record,
                label="terminal replay result",
                with_size=False,
            )
            if replay_path.parent != attempt:
                raise ValueError(
                    "Replay receipt is not stored in the terminal attempt."
                )
            validated_replay = validate_replay_result(
                replay_path,
                expected_run_key=expected_key,
                checkpoint_period=checkpoint_period,
                instance_sha256=instance_sha,
                algorithm_result_sha256=result_sha,
                archive_sha256=archive_sha,
                checkpoint_artifact_sha256=checkpoint_sha,
            )
            if replay_sha != validated_replay["sha256"]:
                raise ValueError("Replay result binding hash mismatch.")

            substitutions = {
                "python_executable": python_executable,
                "adapter_path": algorithm["_adapter_path"],
                "replay_verifier_path": algorithm["_replay_path"],
                "input_path": input_path,
                "configuration_path": input_path,
                "result_path": result_path,
                "replay_result_path": replay_path,
                "instance_path": instance_path,
                **expected_key,
                "checkpoint_period": checkpoint_period,
                "tail_policy_path": tail_policy_path,
            }
            expected_algorithm_argv = _expand_argv(
                algorithm["command_argv"], substitutions
            )
            expected_replay_argv = _expand_argv(
                algorithm["replay_verifier_argv"], substitutions
            )
            _validate_process_record(
                receipt.get("algorithm_process"),
                run_directory=expected_directory,
                expected_attempt_directory=attempt,
                expected_argv=expected_algorithm_argv,
                label="algorithm process",
            )
            _validate_process_record(
                receipt.get("replay_process"),
                run_directory=expected_directory,
                expected_attempt_directory=attempt,
                expected_argv=expected_replay_argv,
                label="replay process",
            )
            valid_run_keys.add(run_key_sha)
        except (OSError, ValueError, KeyError, TypeError) as error:
            invalid_runs.append(
                {
                    "run_key_sha256": run_key_sha_for_error,
                    "receipt": relative_receipt,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    duplicate_keys = {
        key: paths for key, paths in observed.items() if len(paths) > 1
    }
    missing_keys = sorted(set(expected_rows) - set(observed))
    unexpected_keys = sorted(set(observed) - set(expected_rows))
    invocation_is_full = (
        invocation.get("execution_scope") == "formal_candidate"
        and invocation.get("selection") == {"kind": "all"}
        and invocation.get("selected_run_count") == len(expected_rows)
        and invocation.get("expected_run_count") == len(expected_rows)
        and invocation.get("formal_evidence_status") == "NOT_RUN"
    )
    row_validation_passed = (
        len(valid_run_keys) == len(expected_rows) and not invalid_runs
    )
    retry_quality_gate = (
        "PASS"
        if row_validation_passed and not quality_retry_failures
        else "FAIL"
    )
    single_attempt_resource_cleanliness_gate = (
        "PASS"
        if row_validation_passed and not retry_run_keys
        else "NOT_ESTABLISHED"
        if row_validation_passed
        else "FAIL"
    )
    # The V20 invocation runs four rows concurrently in fixed matrix order.
    # Even a retry-free row set would therefore support descriptive timing and
    # sampled working-set summaries only, not an efficiency comparison gate.
    resource_design_balance_gate = (
        "NOT_ESTABLISHED" if row_validation_passed else "FAIL"
    )
    resource_efficiency_gate = (
        "NOT_ESTABLISHED" if row_validation_passed else "FAIL"
    )
    gates = {
        "frozen_preflight_gate": "PASS",
        "full_invocation_gate": "PASS" if invocation_is_full else "FAIL",
        "complete_row_set_gate": (
            "PASS"
            if not missing_keys
            and not unexpected_keys
            and not duplicate_keys
            and len(observed) == len(expected_rows)
            else "FAIL"
        ),
        "terminal_success_gate": (
            "PASS" if row_validation_passed else "FAIL"
        ),
        "budget_checkpoint_gate": (
            "PASS" if row_validation_passed else "FAIL"
        ),
        "hash_binding_gate": (
            "PASS" if row_validation_passed else "FAIL"
        ),
        "reported_archive_witness_self_consistency_gate": (
            "PASS" if row_validation_passed else "FAIL"
        ),
        # The V20 replay receipts independently re-evaluate only the entries
        # that the adapter reports.  They do not contain an ordered witness for
        # every charged objective call, so completeness of the all-evaluated
        # archive is deliberately not promoted from source-level intent to an
        # independently replayed fact.
        "all_evaluated_trace_completeness_gate": "NOT_ESTABLISHED",
        "terminal_process_resource_measurement_gate": (
            "PASS" if row_validation_passed else "FAIL"
        ),
        "attempt_history_enumeration_gate": (
            "PASS" if row_validation_passed else "FAIL"
        ),
        "retry_quality_selection_gate": retry_quality_gate,
        "single_attempt_resource_cleanliness_gate": (
            single_attempt_resource_cleanliness_gate
        ),
        "resource_design_balance_gate": resource_design_balance_gate,
        "resource_efficiency_gate": resource_efficiency_gate,
        "frozen_command_gate": (
            "PASS" if row_validation_passed else "FAIL"
        ),
    }
    quality_gate_names = (
        "frozen_preflight_gate",
        "full_invocation_gate",
        "complete_row_set_gate",
        "terminal_success_gate",
        "budget_checkpoint_gate",
        "hash_binding_gate",
        "reported_archive_witness_self_consistency_gate",
        "attempt_history_enumeration_gate",
        "retry_quality_selection_gate",
        "frozen_command_gate",
    )
    formal_gate = (
        "PASS"
        if all(gates[name] == "PASS" for name in quality_gate_names)
        else "FAIL"
    )
    evidence_status = (
        "REPORTED_ARCHIVE_MATRIX_INTEGRITY_ESTABLISHED"
        if formal_gate == "PASS"
        else "NOT_RUN"
        if invocation.get("execution_scope") == "plumbing_only"
        else "NOT_ESTABLISHED"
    )
    submission_verdict = (
        "HOLD_PENDING_METRIC_AND_STATISTICAL_AUDIT"
        if formal_gate == "PASS"
        else "HOLD"
    )
    audit_payload = {
        "schema": POST_RUN_AUDIT_SCHEMA,
        "audit_implementation": {
            "scope": (
                "posthoc_fail_closed_amendment_not_frozen_algorithm_runtime"
            ),
            "postrun_source_sha256": _file_digest(
                Path(__file__).resolve(strict=True)
            ),
            "frozen_algorithm_modified": False,
            "formal_results_modified": False,
        },
        "study_sha256": study_sha,
        "configuration_matrix_sha256": config_sha,
        "execution_plan_sha256": plan_sha,
        "freeze_receipt_sha256": freeze_receipt_sha,
        "matrix_invocation_sha256": invocation_sha,
        "expected_run_count": len(expected_rows),
        "terminal_receipt_count": len(receipt_paths),
        "observed_unique_run_count": len(observed),
        "valid_run_count": len(valid_run_keys),
        "missing_run_count": len(missing_keys),
        "duplicate_run_count": len(duplicate_keys),
        "unexpected_run_count": len(unexpected_keys),
        "invalid_run_count": len(invalid_runs),
        "missing_run_key_sha256": missing_keys,
        "duplicate_receipts": duplicate_keys,
        "unexpected_run_key_sha256": unexpected_keys,
        "unexpected_receipts": sorted(unexpected_receipts),
        "invalid_runs": invalid_runs,
        "attempt_audit": {
            "retry_run_count": len(retry_run_keys),
            "prior_attempt_count": prior_attempt_count,
            "retry_run_key_sha256": sorted(retry_run_keys),
            "quality_retry_failures": quality_retry_failures,
            "histories": sorted(
                retry_histories,
                key=lambda item: str(item["run_key_sha256"]),
            ),
        },
        "gates": gates,
        "quality_estimand_scope": "reported_archive_relative",
        "all_evaluated_archive_claim_status": "NOT_ESTABLISHED",
        "resource_estimand_scope": "descriptive_terminal_attempt_only",
        "resource_efficiency_claim_status": resource_efficiency_gate,
        "formal_matched_matrix_gate": formal_gate,
        "evidence_status": evidence_status,
        "competitive_superiority_status": "NOT_EVALUATED_BY_THIS_AUDIT",
        "submission_verdict": submission_verdict,
    }
    audit_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else result_root / "post_run_audit.json"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(audit_path, audit_payload)
    return IJOCPostRunResult(
        audit_path=audit_path,
        expected_run_count=len(expected_rows),
        observed_unique_run_count=len(observed),
        valid_run_count=len(valid_run_keys),
        missing_run_count=len(missing_keys),
        duplicate_run_count=len(duplicate_keys),
        invalid_run_count=len(invalid_runs),
        retry_run_count=len(retry_run_keys),
        prior_attempt_count=prior_attempt_count,
        formal_matched_matrix_gate=formal_gate,
        resource_efficiency_gate=resource_efficiency_gate,
        evidence_status=evidence_status,
        submission_verdict=submission_verdict,
    )
