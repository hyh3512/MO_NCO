from __future__ import annotations

"""Cold-process execution skeleton for a frozen IJOC matrix.

Each selected matrix row is executed in a fresh process and receives an
immutable terminal receipt whether it succeeds, times out, or fails validation.
The runner never upgrades a run to formal evidence; only the independent
post-run audit can issue a formal matrix gate.
"""

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import ctypes
from ctypes import wintypes
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

from .pareto_ijoc_preflight import audit_ijoc_competitive_study


EXECUTION_PLAN_SCHEMA = "ijoc_cold_process_execution_plan_v1"
INPUT_SCHEMA = "ijoc_cold_process_input_v1"
ALGORITHM_RESULT_SCHEMA = "ijoc_algorithm_result_v1"
REPLAY_RESULT_SCHEMA = "ijoc_replay_receipt_v1"
RUN_RECEIPT_SCHEMA = "ijoc_cold_process_run_receipt_v1"
INVOCATION_SCHEMA = "ijoc_cold_process_matrix_invocation_v1"
SUMMARY_SCHEMA = "ijoc_cold_process_matrix_summary_v1"
FREEZE_RECEIPT_SCHEMA = "ijoc_manifest_freeze_receipt_v1"

_PLAN_KEYS = {
    "schema",
    "study_sha256",
    "configuration_matrix_sha256",
    "request_sha256",
    "execution_scope",
    "formal_evidence_status",
    "runtime_source_manifest",
    "formal_analysis_plan",
    "tail_calibration_suite_receipt",
    "tail_calibration_artifact_manifest",
    "tail_calibration_instance_bindings",
    "reference_calibration_precommit",
    "reference_calibration_completion_receipt",
    "reference_calibration_artifact_manifest",
    "reference_calibration_bindings",
    "tail_policy_artifact",
    "case_instances",
    "algorithms",
}
_PLAN_ALGORITHM_KEYS = {
    "role",
    "families",
    "kind",
    "version",
    "adapter_artifact",
    "command_argv",
    "replay_verifier_artifact",
    "replay_verifier_argv",
    "configuration",
}
_RESULT_KEYS = {
    "schema",
    "run_key",
    "status",
    "evaluations_used",
    "observed_checkpoints",
    "archive_artifact",
    "checkpoint_artifact",
    "metrics",
}
_REPLAY_KEYS = {
    "schema",
    "run_key",
    "status",
    "instance_sha256",
    "algorithm_result_sha256",
    "archive_sha256",
    "checkpoint_artifact_sha256",
    "evaluations_used",
    "observed_checkpoints",
}
_RUN_KEY_KEYS = {"case_id", "algorithm", "seed", "budget"}
_FREEZE_RECEIPT_KEYS = {
    "schema",
    "status",
    "formal_evidence_status",
    "expected_run_count",
    "artifacts",
}
_FREEZE_ARTIFACT_KEYS = {
    "study",
    "metric_reference_manifest",
    "algorithm_configuration_matrix",
    "reproducibility_manifest",
    "execution_plan",
    "runtime_source_manifest",
    "formal_analysis_plan",
    "tail_calibration_suite_receipt",
    "tail_calibration_artifact_manifest",
    "tail_calibration_instance_bindings",
    "reference_calibration_precommit",
    "reference_calibration_completion_receipt",
    "reference_calibration_artifact_manifest",
    "reference_calibration_bindings",
    "tail_policy_artifact",
}


@dataclass(frozen=True)
class ColdMatrixSummary:
    results_directory: Path
    expected_run_count: int
    selected_run_count: int
    terminal_run_count: int
    success_count: int
    failure_count: int
    execution_scope: str
    formal_evidence_status: str
    summary_path: Path


@dataclass(frozen=True)
class _RowExecutionContext:
    algorithms: Mapping[str, Mapping[str, object]]
    instances: Mapping[str, tuple[Path, str]]
    result_root: Path
    study_sha: str
    configuration_sha: str
    plan_sha: str
    freeze_receipt_sha: str
    execution_scope: str
    checkpoint_period: int
    timeout_seconds: float
    sample_period_seconds: float
    tail_calibration_receipt_path: Path
    tail_calibration_receipt_sha: str
    reference_precommit_path: Path
    reference_precommit_sha: str
    reference_completion_path: Path
    reference_completion_sha: str
    tail_policy_path: Path
    tail_policy_sha: str
    runtime_python_path: Path
    cancel_event: threading.Event


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _reject_duplicate_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field is forbidden: {key!r}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}.")


def _strict_json(path: Path) -> tuple[Mapping[str, Any], bytes, str]:
    if not path.is_file():
        raise ValueError(f"Required JSON artifact is missing: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Artifact is not strict UTF-8 JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact root must be a JSON object: {path}")
    return payload, raw, hashlib.sha256(raw).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} has an unexpected shape; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}."
        )


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> str:
    raw = _canonical_bytes(value)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)
    return hashlib.sha256(raw).hexdigest()


def _bound_path(
    parent: Path,
    binding: object,
    *,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} must be a JSON object.")
    _exact_keys(binding, {"path", "sha256"}, label)
    raw_path = binding.get("path")
    digest = binding.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}.path must be nonempty.")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise ValueError(f"{label}.path must be relative.")
    root = parent.resolve()
    path = (root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label}.path escapes its artifact directory.") from error
    if not path.is_file():
        raise ValueError(f"{label} file is missing: {path}")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{label}.sha256 must be a digest.")
    actual = _file_digest(path)
    if actual != digest:
        raise ValueError(f"{label} SHA-256 mismatch.")
    return path, actual


def _validate_case_packet_children(
    packet_path: Path,
    *,
    case_id: str,
    family_id: str,
) -> None:
    packet, _, _ = _strict_json(packet_path)
    _exact_keys(
        packet,
        {"schema", "case_id", "family", "problem_sha256", "artifacts"},
        f"case packet {case_id}",
    )
    if (
        packet.get("schema") != "ijoc_case_instance_packet_v1"
        or packet.get("case_id") != case_id
        or str(packet.get("family")).upper() != family_id.upper()
    ):
        raise ValueError(f"Case packet identity mismatch for {case_id!r}.")
    problem_sha = packet.get("problem_sha256")
    if not isinstance(problem_sha, str) or len(problem_sha) != 64:
        raise ValueError(f"Case packet problem hash is invalid for {case_id!r}.")
    raw_artifacts = packet.get("artifacts")
    expected_count = 2 if family_id.upper() == "MOTSP" else 1
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != expected_count:
        raise ValueError(
            f"Case packet {case_id!r} has the wrong child-artifact count."
        )
    packet_root = packet_path.parent.resolve()
    child_hashes: set[str] = set()
    child_paths: set[str] = set()
    for index, raw in enumerate(raw_artifacts):
        if not isinstance(raw, dict):
            raise ValueError(f"Case packet child {index} must be an object.")
        _exact_keys(raw, {"path", "sha256"}, f"case packet child {index}")
        raw_path = raw.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("Case packet child path must be nonempty.")
        relative = Path(raw_path)
        normalized = relative.as_posix()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or normalized in child_paths
        ):
            raise ValueError("Case packet child path is unsafe or duplicate.")
        child_paths.add(normalized)
        path = (packet_root / relative).resolve()
        try:
            path.relative_to(packet_root)
        except ValueError as error:
            raise ValueError("Case packet child escapes its packet.") from error
        digest = raw.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not path.is_file()
            or _file_digest(path) != digest
        ):
            raise ValueError("Case packet child hash mismatch.")
        child_hashes.add(digest)
    if len(child_hashes) != expected_count:
        raise ValueError("Case packet repeats child bytes.")


def _study_components(
    study_path: Path,
) -> tuple[
    Mapping[str, Any],
    str,
    Mapping[str, Any],
    str,
    dict[str, tuple[Path, str]],
]:
    preflight = audit_ijoc_competitive_study(study_path)
    if preflight.submission_preflight_gate != "PASS":
        raise ValueError(
            "Cold-process execution requires a PASS preflight packet: "
            + "; ".join(preflight.reasons)
        )
    study, _, study_sha = _strict_json(study_path)
    config_binding = study.get("algorithm_configuration_matrix")
    config_path, config_sha = _bound_path(
        study_path.parent,
        config_binding,
        label="algorithm_configuration_matrix",
    )
    configuration, _, _ = _strict_json(config_path)
    reproduction_path, _ = _bound_path(
        study_path.parent,
        study.get("artifact_release"),
        label="artifact_release",
    )
    reproducibility, _, _ = _strict_json(reproduction_path)
    family_by_case: dict[str, str] = {}
    for raw_family in study.get("problem_families", []):
        if not isinstance(raw_family, dict):
            raise ValueError("Study family must be an object.")
        family_id = raw_family.get("id")
        raw_cases = raw_family.get("cases")
        if not isinstance(family_id, str) or not isinstance(raw_cases, list):
            raise ValueError("Study family identity is invalid.")
        for case_id in raw_cases:
            if not isinstance(case_id, str) or case_id in family_by_case:
                raise ValueError("Study case-family bindings are invalid.")
            family_by_case[case_id] = family_id
    instances: dict[str, tuple[Path, str]] = {}
    for raw in reproducibility.get("instance_files", []):
        if not isinstance(raw, dict):
            raise ValueError("instance_files entry must be an object.")
        case_id = raw.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("instance_files entry has no case_id.")
        path, digest = _bound_path(
            reproduction_path.parent,
            {"path": raw.get("path"), "sha256": raw.get("sha256")},
            label=f"instance {case_id}",
        )
        if case_id not in family_by_case:
            raise ValueError("Reproducibility instance names an unknown case.")
        _validate_case_packet_children(
            path,
            case_id=case_id,
            family_id=family_by_case[case_id],
        )
        instances[case_id] = (path, digest)
    if set(instances) != set(family_by_case):
        raise ValueError("Reproducibility instances do not cover every case.")
    return study, study_sha, configuration, config_sha, instances


def _select_stratified_plumbing_rows(
    study: Mapping[str, Any],
    instances: Mapping[str, tuple[Path, str]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], dict[str, object]]:
    """Select a deterministic capacity gate across every family/algorithm/budget.

    Each family uses the case whose hash-bound packet has the largest total
    child-artifact byte size. Within every family/algorithm/budget stratum, the
    minimum available seed is selected. The subset remains plumbing-only.
    """

    family_specs = study.get("problem_families")
    if not isinstance(family_specs, list) or not family_specs:
        raise ValueError("Study problem_families must be a nonempty array.")
    family_by_case: dict[str, str] = {}
    algorithms_by_family: dict[str, tuple[str, ...]] = {}
    cases_by_family: dict[str, tuple[str, ...]] = {}
    for raw_family in family_specs:
        if not isinstance(raw_family, dict):
            raise ValueError("Study family must be an object.")
        family_id = raw_family.get("id")
        raw_cases = raw_family.get("cases")
        raw_algorithms = raw_family.get("algorithms")
        if (
            not isinstance(family_id, str)
            or not family_id
            or not isinstance(raw_cases, list)
            or not raw_cases
            or not isinstance(raw_algorithms, list)
            or not raw_algorithms
        ):
            raise ValueError("Study family is invalid for stratified plumbing.")
        cases: list[str] = []
        for case_id in raw_cases:
            if (
                not isinstance(case_id, str)
                or not case_id
                or case_id in family_by_case
                or case_id not in instances
            ):
                raise ValueError(
                    "Study case-family bindings are invalid for stratified plumbing."
                )
            family_by_case[case_id] = family_id
            cases.append(case_id)
        algorithms: list[str] = []
        for algorithm_id in raw_algorithms:
            if (
                not isinstance(algorithm_id, str)
                or not algorithm_id
                or algorithm_id in algorithms
            ):
                raise ValueError(
                    "Study family algorithms are invalid for stratified plumbing."
                )
            algorithms.append(algorithm_id)
        cases_by_family[family_id] = tuple(cases)
        algorithms_by_family[family_id] = tuple(algorithms)

    packet_child_bytes: dict[str, int] = {}
    for case_id, (packet_path, _) in instances.items():
        packet, _, _ = _strict_json(packet_path)
        artifacts = packet.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(
                f"Case packet {case_id!r} has no child artifacts for plumbing."
            )
        packet_root = packet_path.parent.resolve()
        total_bytes = 0
        for raw_artifact in artifacts:
            if not isinstance(raw_artifact, dict):
                raise ValueError("Case packet child must be an object.")
            raw_path = raw_artifact.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError("Case packet child path must be nonempty.")
            child_path = (packet_root / raw_path).resolve()
            try:
                child_path.relative_to(packet_root)
            except ValueError as error:
                raise ValueError("Case packet child escapes its packet.") from error
            if not child_path.is_file():
                raise ValueError("Case packet child is missing.")
            total_bytes += child_path.stat().st_size
        packet_child_bytes[case_id] = total_bytes

    representative_cases = {
        family_id: max(
            cases,
            key=lambda case_id: (packet_child_bytes[case_id], case_id),
        )
        for family_id, cases in cases_by_family.items()
    }
    indexed: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    budgets: set[int] = set()
    for row in rows:
        run_key = _run_key(row)
        case_id = str(run_key["case_id"])
        family_id = family_by_case.get(case_id)
        if family_id is None:
            raise ValueError("Matrix row names a case outside the study families.")
        budget = int(run_key["budget"])
        budgets.add(budget)
        if case_id == representative_cases[family_id]:
            indexed.setdefault(
                (family_id, str(run_key["algorithm"]), budget), []
            ).append(row)
    if not budgets:
        raise ValueError("Cannot stratify an empty matrix.")

    selected: list[Mapping[str, Any]] = []
    for family_id in sorted(algorithms_by_family):
        for algorithm_id in sorted(algorithms_by_family[family_id]):
            for budget in sorted(budgets):
                candidates = indexed.get((family_id, algorithm_id, budget), [])
                if not candidates:
                    raise ValueError(
                        "Matrix lacks a family/algorithm/budget plumbing stratum: "
                        f"{family_id}/{algorithm_id}/{budget}."
                    )
                selected.append(
                    min(
                        candidates,
                        key=lambda row: (
                            int(_run_key(row)["seed"]),
                            _canonical_digest(_run_key(row)),
                        ),
                    )
                )
    selection = {
        "kind": "stratified_largest_bound_instance_bytes",
        "coverage": "each_family_algorithm_budget",
        "case_by_family": {
            family_id: representative_cases[family_id]
            for family_id in sorted(representative_cases)
        },
        "case_child_bytes_by_family": {
            family_id: packet_child_bytes[representative_cases[family_id]]
            for family_id in sorted(representative_cases)
        },
        "budgets": sorted(budgets),
        "seed_rule": "minimum_available_seed_per_stratum",
    }
    return selected, selection


def _dependency_environment_gate(
    study_path: Path,
    study: Mapping[str, Any],
) -> dict[str, object]:
    reproduction_path, _ = _bound_path(
        study_path.parent,
        study.get("artifact_release"),
        label="artifact_release",
    )
    reproduction, _, _ = _strict_json(reproduction_path)
    environment = reproduction.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("Reproducibility environment must be an object.")
    declared_python = environment.get("python_version")
    observed_python = platform.python_version()
    if declared_python != observed_python:
        raise ValueError(
            "Frozen Python version does not match the executing interpreter: "
            f"declared={declared_python!r}, observed={observed_python!r}."
        )
    lock_path, lock_sha = _bound_path(
        reproduction_path.parent,
        environment.get("dependency_lock"),
        label="dependency lock",
    )
    try:
        lines = lock_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("Dependency lock is not UTF-8 text.") from error
    requirement_pattern = re.compile(
        r"^([A-Za-z0-9][A-Za-z0-9._-]*)=="
        r"([A-Za-z0-9][A-Za-z0-9._+!-]*)$"
    )
    requirements: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for line_number, raw in enumerate(lines, start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        match = requirement_pattern.fullmatch(text)
        if match is None:
            raise ValueError(
                "Dependency lock accepts only exact name==version lines; "
                f"invalid line {line_number}."
            )
        name, expected_version = match.groups()
        normalized_name = re.sub(r"[-_.]+", "-", name).lower()
        if normalized_name in seen_names:
            raise ValueError("Dependency lock repeats a distribution.")
        seen_names.add(normalized_name)
        try:
            observed_version = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError as error:
            raise ValueError(
                f"Dependency distribution is not installed: {name!r}."
            ) from error
        if observed_version != expected_version:
            raise ValueError(
                f"Dependency version mismatch for {name!r}: "
                f"declared={expected_version!r}, observed={observed_version!r}."
            )
        requirements.append({"name": name, "version": observed_version})
    if not requirements:
        raise ValueError("Dependency lock must contain at least one requirement.")
    return {
        "status": "PASS",
        "python_executable": sys.executable,
        "python_version": observed_python,
        "dependency_lock_sha256": lock_sha,
        "requirements": requirements,
    }


def _validate_plan(
    plan_path: Path,
    *,
    study_sha: str,
    configuration_sha: str,
    instances: Mapping[str, tuple[Path, str]],
) -> tuple[Mapping[str, Any], str, dict[str, dict[str, object]]]:
    plan, _, plan_sha = _strict_json(plan_path)
    _exact_keys(plan, _PLAN_KEYS, "execution plan")
    if plan.get("schema") != EXECUTION_PLAN_SCHEMA:
        raise ValueError(f"execution plan schema must be {EXECUTION_PLAN_SCHEMA!r}.")
    if plan.get("study_sha256") != study_sha:
        raise ValueError("Execution plan is not bound to the selected study bytes.")
    if plan.get("configuration_matrix_sha256") != configuration_sha:
        raise ValueError("Execution plan is not bound to the configuration matrix.")
    if plan.get("execution_scope") != "formal_candidate":
        raise ValueError("Frozen execution_plan must declare formal_candidate scope.")
    if plan.get("formal_evidence_status") != "NOT_RUN":
        raise ValueError("Frozen execution_plan must begin in NOT_RUN state.")
    for label in (
        "runtime_source_manifest",
        "formal_analysis_plan",
        "tail_calibration_suite_receipt",
        "tail_calibration_artifact_manifest",
        "tail_calibration_instance_bindings",
        "reference_calibration_precommit",
        "reference_calibration_completion_receipt",
        "reference_calibration_artifact_manifest",
        "reference_calibration_bindings",
        "tail_policy_artifact",
    ):
        _bound_path(plan_path.parent, plan.get(label), label=f"plan {label}")
    raw_case_instances = plan.get("case_instances")
    if not isinstance(raw_case_instances, dict):
        raise ValueError("case_instances must be an object.")
    if set(raw_case_instances) != set(instances):
        raise ValueError("Plan case_instances do not exactly match the study.")
    for case_id, binding in raw_case_instances.items():
        path, digest = _bound_path(
            plan_path.parent, binding, label=f"plan instance {case_id}"
        )
        expected_path, expected_digest = instances[case_id]
        if path != expected_path or digest != expected_digest:
            raise ValueError(f"Plan instance binding disagrees for {case_id!r}.")

    raw_algorithms = plan.get("algorithms")
    if not isinstance(raw_algorithms, dict) or not raw_algorithms:
        raise ValueError("Plan algorithms must be a nonempty object.")
    algorithms: dict[str, dict[str, object]] = {}
    for algorithm_id, raw in raw_algorithms.items():
        if not isinstance(algorithm_id, str) or not algorithm_id:
            raise ValueError("Plan algorithm id must be nonempty.")
        if not isinstance(raw, dict):
            raise ValueError(f"Plan algorithm {algorithm_id!r} must be an object.")
        _exact_keys(raw, _PLAN_ALGORITHM_KEYS, f"plan algorithm {algorithm_id}")
        adapter_path, adapter_sha = _bound_path(
            plan_path.parent,
            raw.get("adapter_artifact"),
            label=f"algorithm {algorithm_id} adapter",
        )
        replay_path, replay_sha = _bound_path(
            plan_path.parent,
            raw.get("replay_verifier_artifact"),
            label=f"algorithm {algorithm_id} replay verifier",
        )
        for key in ("command_argv", "replay_verifier_argv"):
            argv = raw.get(key)
            if (
                not isinstance(argv, list)
                or not argv
                or any(not isinstance(token, str) or not token for token in argv)
            ):
                raise ValueError(f"Algorithm {algorithm_id!r} {key} is invalid.")
        configuration = raw.get("configuration")
        if not isinstance(configuration, dict):
            raise ValueError(f"Algorithm {algorithm_id!r} configuration is invalid.")
        algorithms[algorithm_id] = {
            **raw,
            "_adapter_path": adapter_path,
            "_adapter_sha256": adapter_sha,
            "_replay_path": replay_path,
            "_replay_sha256": replay_sha,
        }
    return plan, plan_sha, algorithms


def _validate_runtime_source_manifest(
    plan_path: Path,
    plan: Mapping[str, Any],
) -> Path:
    manifest_path, _ = _bound_path(
        plan_path.parent,
        plan.get("runtime_source_manifest"),
        label="runtime source manifest",
    )
    manifest, _, _ = _strict_json(manifest_path)
    _exact_keys(
        manifest,
        {"schema", "source_archive_sha256", "python_path_root", "files"},
        "runtime source manifest",
    )
    if manifest.get("schema") != "ijoc_frozen_runtime_source_manifest_v1":
        raise ValueError("Runtime source manifest schema mismatch.")
    archive_sha = manifest.get("source_archive_sha256")
    if not isinstance(archive_sha, str) or len(archive_sha) != 64:
        raise ValueError("Runtime source archive SHA-256 is invalid.")
    raw_python_root = manifest.get("python_path_root")
    if not isinstance(raw_python_root, str) or not raw_python_root:
        raise ValueError("Runtime source python_path_root is invalid.")
    candidate = Path(raw_python_root)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Runtime source python_path_root must be safe and relative.")
    packet_root = plan_path.parent.resolve()
    python_root = (packet_root / candidate).resolve()
    try:
        python_root.relative_to(packet_root)
    except ValueError as error:
        raise ValueError("Runtime Python path escapes the frozen packet.") from error
    if not python_root.is_dir() or not (python_root / "mo_nco" / "__init__.py").is_file():
        raise ValueError("Frozen runtime does not contain importable mo_nco bytes.")

    runtime_root = packet_root / "runtime" / "source"
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("Runtime source manifest files must be nonempty.")
    declared_paths: set[str] = set()
    for index, raw in enumerate(raw_files):
        if not isinstance(raw, dict):
            raise ValueError(f"Runtime source file {index} must be an object.")
        _exact_keys(raw, {"path", "sha256", "bytes"}, f"runtime source file {index}")
        raw_path = raw.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("Runtime source file path must be nonempty.")
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Runtime source file path is unsafe.")
        normalized = relative.as_posix()
        if normalized in declared_paths:
            raise ValueError("Runtime source file paths must be unique.")
        declared_paths.add(normalized)
        path = (runtime_root / relative).resolve()
        try:
            path.relative_to(runtime_root.resolve())
        except ValueError as error:
            raise ValueError("Runtime source file escapes its tree.") from error
        if not path.is_file() or _file_digest(path) != raw.get("sha256"):
            raise ValueError("Runtime source file hash mismatch.")
        size = raw.get("bytes")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or path.stat().st_size != size
        ):
            raise ValueError("Runtime source file byte count mismatch.")
    observed_paths = {
        path.relative_to(runtime_root).as_posix()
        for path in runtime_root.rglob("*")
        if path.is_file()
    }
    if observed_paths != declared_paths:
        raise ValueError("Frozen runtime source tree differs from its manifest.")
    return python_root


def _validate_freeze_receipt(
    *,
    study_path: Path,
    plan_path: Path,
    study_sha: str,
    configuration_sha: str,
    plan_sha: str,
    expected_run_count: int,
) -> str:
    if study_path.parent != plan_path.parent:
        raise ValueError("Study and execution plan must share one frozen directory.")
    plan_payload, _, observed_plan_sha = _strict_json(plan_path)
    if observed_plan_sha != plan_sha:
        raise ValueError("Observed execution-plan bytes changed during validation.")
    receipt_path = study_path.parent / "freeze_receipt.json"
    receipt, _, receipt_sha = _strict_json(receipt_path)
    _exact_keys(receipt, _FREEZE_RECEIPT_KEYS, "freeze receipt")
    if receipt.get("schema") != FREEZE_RECEIPT_SCHEMA:
        raise ValueError("Freeze receipt schema mismatch.")
    if receipt.get("status") != "FROZEN":
        raise ValueError("Freeze receipt is not terminally FROZEN.")
    if receipt.get("formal_evidence_status") != "NOT_RUN":
        raise ValueError("Freeze receipt must precede all formal execution.")
    if receipt.get("expected_run_count") != expected_run_count:
        raise ValueError("Freeze receipt expected-run count mismatch.")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Freeze receipt artifacts must be an object.")
    _exact_keys(artifacts, _FREEZE_ARTIFACT_KEYS, "freeze receipt artifacts")
    expected_paths_and_hashes = {
        "study": (study_path, study_sha),
        "algorithm_configuration_matrix": (
            study_path.parent / "algorithm_configuration_matrix.json",
            configuration_sha,
        ),
        "execution_plan": (plan_path, plan_sha),
    }
    for label, (expected_path, expected_sha) in expected_paths_and_hashes.items():
        path, digest = _bound_path(
            receipt_path.parent,
            artifacts.get(label),
            label=f"freeze receipt {label}",
        )
        if path != expected_path.resolve() or digest != expected_sha:
            raise ValueError(f"Freeze receipt {label} binding mismatch.")
    for label in ("metric_reference_manifest", "reproducibility_manifest"):
        _bound_path(
            receipt_path.parent,
            artifacts.get(label),
            label=f"freeze receipt {label}",
        )
    for label in (
        "runtime_source_manifest",
        "formal_analysis_plan",
        "tail_calibration_suite_receipt",
        "tail_calibration_artifact_manifest",
        "tail_calibration_instance_bindings",
        "reference_calibration_precommit",
        "reference_calibration_completion_receipt",
        "reference_calibration_artifact_manifest",
        "reference_calibration_bindings",
        "tail_policy_artifact",
    ):
        receipt_path_bound, receipt_digest = _bound_path(
            receipt_path.parent,
            artifacts.get(label),
            label=f"freeze receipt {label}",
        )
        plan_path_bound, plan_digest = _bound_path(
            plan_path.parent,
            plan_payload.get(label),
            label=f"execution plan {label}",
        )
        if (
            receipt_path_bound != plan_path_bound
            or receipt_digest != plan_digest
        ):
            raise ValueError(f"Freeze receipt {label} disagrees with the plan.")
    return receipt_sha


def _run_key(row: Mapping[str, Any]) -> dict[str, object]:
    result = {
        "case_id": row.get("case_id"),
        "algorithm": row.get("algorithm"),
        "seed": row.get("seed"),
        "budget": row.get("budget"),
    }
    if (
        not isinstance(result["case_id"], str)
        or not result["case_id"]
        or not isinstance(result["algorithm"], str)
        or not result["algorithm"]
        or isinstance(result["seed"], bool)
        or not isinstance(result["seed"], int)
        or result["seed"] < 0
        or isinstance(result["budget"], bool)
        or not isinstance(result["budget"], int)
        or result["budget"] <= 0
    ):
        raise ValueError("Configuration row has an invalid run key.")
    return result


def _expected_checkpoints(budget: int, period: int) -> list[int]:
    if budget % period:
        raise ValueError("Budget is not divisible by the checkpoint period.")
    return list(range(period, budget + 1, period))


def _validate_run_key(value: object, expected: Mapping[str, object], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    _exact_keys(value, _RUN_KEY_KEYS, label)
    if value != expected:
        raise ValueError(f"{label} does not match its frozen matrix row.")


def validate_algorithm_result(
    result_path: Path,
    *,
    expected_run_key: Mapping[str, object],
    checkpoint_period: int,
) -> dict[str, object]:
    payload, _, result_sha = _strict_json(result_path)
    _exact_keys(payload, _RESULT_KEYS, "algorithm result")
    if payload.get("schema") != ALGORITHM_RESULT_SCHEMA:
        raise ValueError("Algorithm result schema mismatch.")
    _validate_run_key(payload.get("run_key"), expected_run_key, "result run_key")
    if payload.get("status") != "SUCCESS":
        raise ValueError("Algorithm result does not declare SUCCESS.")
    budget = int(expected_run_key["budget"])
    if payload.get("evaluations_used") != budget:
        raise ValueError("Algorithm result does not use the exact frozen budget.")
    expected_checkpoints = _expected_checkpoints(budget, checkpoint_period)
    if payload.get("observed_checkpoints") != expected_checkpoints:
        raise ValueError("Algorithm result has an incomplete checkpoint grid.")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Algorithm result metrics must be an object.")
    archive_path, archive_sha = _bound_path(
        result_path.parent,
        payload.get("archive_artifact"),
        label="algorithm result archive",
    )
    checkpoint_path, checkpoint_sha = _bound_path(
        result_path.parent,
        payload.get("checkpoint_artifact"),
        label="algorithm checkpoint witnesses",
    )
    return {
        "path": result_path,
        "sha256": result_sha,
        "archive_path": archive_path,
        "archive_sha256": archive_sha,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha,
        "payload": payload,
    }


def validate_replay_result(
    replay_path: Path,
    *,
    expected_run_key: Mapping[str, object],
    checkpoint_period: int,
    instance_sha256: str,
    algorithm_result_sha256: str,
    archive_sha256: str,
    checkpoint_artifact_sha256: str,
) -> dict[str, object]:
    payload, _, replay_sha = _strict_json(replay_path)
    _exact_keys(payload, _REPLAY_KEYS, "replay result")
    if payload.get("schema") != REPLAY_RESULT_SCHEMA:
        raise ValueError("Replay result schema mismatch.")
    _validate_run_key(payload.get("run_key"), expected_run_key, "replay run_key")
    if payload.get("status") != "PASS":
        raise ValueError("Replay verifier did not report PASS.")
    if payload.get("instance_sha256") != instance_sha256:
        raise ValueError("Replay receipt is bound to the wrong instance bytes.")
    if payload.get("algorithm_result_sha256") != algorithm_result_sha256:
        raise ValueError("Replay receipt is bound to the wrong result bytes.")
    if payload.get("archive_sha256") != archive_sha256:
        raise ValueError("Replay receipt is bound to the wrong archive bytes.")
    if payload.get("checkpoint_artifact_sha256") != checkpoint_artifact_sha256:
        raise ValueError(
            "Replay receipt is bound to the wrong checkpoint-witness bytes."
        )
    budget = int(expected_run_key["budget"])
    if payload.get("evaluations_used") != budget:
        raise ValueError("Replay receipt does not confirm the frozen budget.")
    if payload.get("observed_checkpoints") != _expected_checkpoints(
        budget, checkpoint_period
    ):
        raise ValueError("Replay receipt does not confirm the checkpoint grid.")
    return {"path": replay_path, "sha256": replay_sha, "payload": payload}


def _linux_process_tree_rss(root_pid: int) -> int | None:
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return None
    parents: dict[int, int] = {}
    rss: dict[int, int] = {}
    page_size = os.sysconf("SC_PAGE_SIZE")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            tail = stat[stat.rfind(")") + 2 :].split()
            parents[pid] = int(tail[1])
            statm = (entry / "statm").read_text(encoding="ascii").split()
            rss[pid] = int(statm[1]) * page_size
        except (FileNotFoundError, PermissionError, IndexError, ValueError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    values = [rss[pid] for pid in descendants if pid in rss]
    return sum(values) if values else None


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_void_p),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]


class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def _windows_process_tree_rss(root_pid: int) -> int | None:
    if os.name != "nt":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_PROCESSENTRY32W),
    ]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_PROCESSENTRY32W),
    ]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.OpenProcess.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_PROCESS_MEMORY_COUNTERS_EX),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot == invalid_handle:
        return None
    parents: dict[int, int] = {}
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        success = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while success:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            success = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    process_query_limited_information = 0x1000
    process_vm_read = 0x0010
    total = 0
    observed = False
    for pid in descendants:
        handle = kernel32.OpenProcess(
            process_query_limited_information | process_vm_read,
            False,
            pid,
        )
        if not handle:
            continue
        try:
            counters = _PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(counters)
            if psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                total += int(counters.WorkingSetSize)
                observed = True
        finally:
            kernel32.CloseHandle(handle)
    return total if observed else None


def _sample_process_tree_rss(root_pid: int) -> int | None:
    if os.name == "nt":
        return _windows_process_tree_rss(root_pid)
    if sys.platform.startswith("linux"):
        return _linux_process_tree_rss(root_pid)
    return None


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()


def _terminate_and_reap_process_bounded(
    process: subprocess.Popen[bytes],
    *,
    wait_timeout_seconds: float = 10.0,
) -> tuple[int | None, dict[str, object]]:
    """Best-effort tree cleanup without an unbounded ``wait`` seam.

    A first tree termination is followed by a bounded wait.  If the process
    still cannot be reaped, a second tree termination plus a direct root kill
    is attempted and waited for once more with the same finite bound.  Cleanup
    failures are evidence returned to the caller; they never prevent the row
    runner from writing its terminal receipt.
    """

    errors: list[str] = []
    wait_attempts = 0
    for cleanup_round in (1, 2):
        try:
            _terminate_process_tree(process)
        except Exception as error:
            errors.append(
                f"tree_termination_round_{cleanup_round}: "
                f"{type(error).__name__}: {error}"
            )
        if cleanup_round == 2:
            try:
                process.kill()
            except Exception as error:
                errors.append(
                    f"root_kill_round_2: {type(error).__name__}: {error}"
                )
        wait_attempts += 1
        try:
            exit_code = process.wait(timeout=wait_timeout_seconds)
        except Exception as error:
            errors.append(
                f"bounded_wait_round_{cleanup_round}: "
                f"{type(error).__name__}: {error}"
            )
            continue
        return exit_code, {
            "status": "REAPED",
            "rounds": cleanup_round,
            "wait_attempts": wait_attempts,
            "wait_timeout_seconds": wait_timeout_seconds,
            "errors": errors,
        }

    try:
        exit_code = process.poll()
    except Exception as error:
        errors.append(f"final_poll: {type(error).__name__}: {error}")
        exit_code = None
    return exit_code, {
        "status": (
            "EXIT_OBSERVED_AFTER_BOUNDED_CLEANUP"
            if exit_code is not None
            else "BOUNDED_CLEANUP_EXHAUSTED"
        ),
        "rounds": 2,
        "wait_attempts": wait_attempts,
        "wait_timeout_seconds": wait_timeout_seconds,
        "errors": errors,
    }


def _file_receipt(path: Path, relative_to: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": _file_digest(path),
        "bytes": path.stat().st_size,
    }


def _execute_process(
    argv: Sequence[str],
    *,
    working_directory: Path,
    timeout_seconds: float,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    sample_period_seconds: float,
    cancel_event: threading.Event | None = None,
) -> dict[str, object]:
    started_utc = _utc_now()
    started = time.monotonic()
    peak_rss = 0
    observed_rss = False
    timed_out = False
    interrupted = False
    spawn_error: str | None = None
    exit_code: int | None = None
    termination: dict[str, object] | None = None
    creationflags = (
        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    )
    with stdout_path.open("wb") as stdout_handle, stderr_path.open(
        "wb"
    ) as stderr_handle:
        try:
            process = subprocess.Popen(
                list(argv),
                cwd=working_directory,
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                shell=False,
                start_new_session=(os.name != "nt"),
                creationflags=creationflags,
            )
        except OSError as error:
            spawn_error = f"{type(error).__name__}: {error}"
            process = None
            termination = {
                "status": "NOT_STARTED",
                "rounds": 0,
                "wait_attempts": 0,
                "wait_timeout_seconds": 10.0,
                "errors": [],
            }
        if process is not None:
            try:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        interrupted = True
                        exit_code, termination = (
                            _terminate_and_reap_process_bounded(process)
                        )
                        break
                    sample = _sample_process_tree_rss(process.pid)
                    if sample is not None:
                        observed_rss = True
                        peak_rss = max(peak_rss, sample)
                    exit_code = process.poll()
                    if exit_code is not None:
                        break
                    if time.monotonic() - started >= timeout_seconds:
                        timed_out = True
                        exit_code, termination = (
                            _terminate_and_reap_process_bounded(process)
                        )
                        break
                    time.sleep(sample_period_seconds)
            except KeyboardInterrupt:
                interrupted = True
                exit_code, termination = _terminate_and_reap_process_bounded(
                    process
                )
            finally:
                sample = _sample_process_tree_rss(process.pid)
                if sample is not None:
                    observed_rss = True
                    peak_rss = max(peak_rss, sample)
    finished_utc = _utc_now()
    result = {
        "argv": list(argv),
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "wall_time_seconds": round(time.monotonic() - started, 9),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "interrupted": interrupted,
        "spawn_error": spawn_error,
        "sampled_peak_process_tree_rss_bytes": peak_rss if observed_rss else None,
        "resource_measurement_status": "PASS" if observed_rss else "UNAVAILABLE",
        "stdout": _file_receipt(stdout_path, working_directory.parent.parent),
        "stderr": _file_receipt(stderr_path, working_directory.parent.parent),
    }
    if termination is not None:
        result["termination"] = termination
    return result


def _expand_argv(
    template: Sequence[str],
    substitutions: Mapping[str, object],
) -> list[str]:
    text_substitutions = {key: str(value) for key, value in substitutions.items()}
    try:
        return [token.format_map(text_substitutions) for token in template]
    except (KeyError, ValueError) as error:
        raise ValueError(f"Command template expansion failed: {error}") from error


def _validate_environment_overrides(
    configuration: Mapping[str, Any],
    *,
    seed: int,
    run_key_sha: str,
    runtime_python_path: Path,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    raw = configuration.get("environment", {})
    if not isinstance(raw, dict):
        raise ValueError("configuration.environment must be an object.")
    overrides: dict[str, str] = {}
    for key, value in raw.items():
        if (
            not isinstance(key, str)
            or not key
            or "=" in key
            or "\x00" in key
            or not isinstance(value, str)
            or "\x00" in value
        ):
            raise ValueError("configuration.environment must contain safe strings.")
        if key.upper() in {"PYTHONPATH", "PYTHONHOME"}:
            raise ValueError(
                "configuration.environment may not override the frozen "
                "Python runtime."
            )
        overrides[key] = value
    overrides["PYTHONHASHSEED"] = str(seed)
    overrides["IJOC_RUN_KEY_SHA256"] = run_key_sha
    overrides["PYTHONPATH"] = str(runtime_python_path)
    overrides["PYTHONNOUSERSITE"] = "1"
    overrides["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.update(overrides)
    return environment


def _validate_terminal_receipt_identity(
    receipt_path: Path,
    *,
    run_key: Mapping[str, object],
    run_key_sha: str,
    study_sha: str,
    plan_sha: str,
    freeze_receipt_sha: str,
    tail_calibration_receipt_path: Path,
    tail_calibration_receipt_sha: str,
    reference_precommit_path: Path,
    reference_precommit_sha: str,
    reference_completion_path: Path,
    reference_completion_sha: str,
) -> Mapping[str, Any]:
    receipt, _, _ = _strict_json(receipt_path)
    if receipt.get("schema") != RUN_RECEIPT_SCHEMA:
        raise ValueError(f"Existing terminal receipt has wrong schema: {receipt_path}")
    if receipt.get("run_key") != run_key:
        raise ValueError(f"Existing terminal receipt has wrong run key: {receipt_path}")
    if receipt.get("run_key_sha256") != run_key_sha:
        raise ValueError(f"Existing terminal receipt has wrong key hash: {receipt_path}")
    if receipt.get("study_sha256") != study_sha:
        raise ValueError(f"Existing terminal receipt has wrong study hash: {receipt_path}")
    if receipt.get("execution_plan_sha256") != plan_sha:
        raise ValueError(f"Existing terminal receipt has wrong plan hash: {receipt_path}")
    if receipt.get("freeze_receipt_sha256") != freeze_receipt_sha:
        raise ValueError(
            f"Existing terminal receipt has wrong freeze receipt hash: {receipt_path}"
        )
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
            raise ValueError(f"Existing terminal receipt has no {key}.")
        _exact_keys(binding, {"path", "sha256"}, f"terminal {key}")
        try:
            observed_path = Path(str(binding.get("path"))).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError(
                f"Existing terminal receipt {key} path is invalid."
            ) from error
        if (
            observed_path != expected_path
            or binding.get("sha256") != expected_sha
            or _file_digest(observed_path) != expected_sha
        ):
            raise ValueError(
                f"Existing terminal receipt {key} binding mismatch."
            )
    return receipt


def _execute_matrix_row(
    row: Mapping[str, Any],
    context: _RowExecutionContext,
) -> str:
    if context.cancel_event.is_set():
        return "CANCELLED_BEFORE_START"
    run_key = _run_key(row)
    run_key_sha = _canonical_digest(run_key)
    run_directory = context.result_root / "runs" / run_key_sha
    terminal_path = run_directory / "terminal_receipt.json"
    if terminal_path.exists():
        receipt = _validate_terminal_receipt_identity(
            terminal_path,
            run_key=run_key,
            run_key_sha=run_key_sha,
            study_sha=context.study_sha,
            plan_sha=context.plan_sha,
            freeze_receipt_sha=context.freeze_receipt_sha,
            tail_calibration_receipt_path=(
                context.tail_calibration_receipt_path
            ),
            tail_calibration_receipt_sha=(
                context.tail_calibration_receipt_sha
            ),
            reference_precommit_path=context.reference_precommit_path,
            reference_precommit_sha=context.reference_precommit_sha,
            reference_completion_path=context.reference_completion_path,
            reference_completion_sha=context.reference_completion_sha,
        )
        return str(receipt.get("status"))

    run_directory.mkdir(parents=True, exist_ok=True)
    attempts = run_directory / "attempts"
    attempts.mkdir(exist_ok=True)
    existing_attempts = sorted(
        child for child in attempts.iterdir() if child.is_dir()
    )
    attempt_number = len(existing_attempts) + 1
    attempt = attempts / f"{attempt_number:06d}"
    attempt.mkdir()

    algorithm_id = str(run_key["algorithm"])
    algorithm = context.algorithms.get(algorithm_id)
    status = "INFRASTRUCTURE_FAILURE"
    reason: str | None = None
    algorithm_process: dict[str, object] | None = None
    replay_process: dict[str, object] | None = None
    input_binding: dict[str, object] | None = None
    result_binding: dict[str, object] | None = None
    replay_binding: dict[str, object] | None = None
    interrupted = False
    try:
        if algorithm is None:
            raise ValueError(f"No execution-plan algorithm for {algorithm_id!r}.")
        configuration_payload = row.get("configuration")
        if not isinstance(configuration_payload, dict):
            raise ValueError("Configuration row readable payload is invalid.")
        if _canonical_digest(configuration_payload) != row.get(
            "configuration_sha256"
        ):
            raise ValueError("Configuration row hash mismatch.")
        instance_path, instance_sha = context.instances[str(run_key["case_id"])]
        input_payload = {
            "schema": INPUT_SCHEMA,
            "study_sha256": context.study_sha,
            "configuration_matrix_sha256": context.configuration_sha,
            "execution_plan_sha256": context.plan_sha,
            "freeze_receipt_sha256": context.freeze_receipt_sha,
            "tail_calibration_suite_receipt": {
                "path": str(context.tail_calibration_receipt_path),
                "sha256": context.tail_calibration_receipt_sha,
            },
            "reference_calibration_precommit": {
                "path": str(context.reference_precommit_path),
                "sha256": context.reference_precommit_sha,
            },
            "reference_calibration_completion_receipt": {
                "path": str(context.reference_completion_path),
                "sha256": context.reference_completion_sha,
            },
            "tail_policy_artifact": {
                "path": str(context.tail_policy_path),
                "sha256": context.tail_policy_sha,
            },
            "run_key": run_key,
            "run_key_sha256": run_key_sha,
            "configuration": configuration_payload,
            "configuration_sha256": row["configuration_sha256"],
            "instance_artifact": {
                "path": str(instance_path),
                "sha256": instance_sha,
            },
            "anytime_checkpoint_period": context.checkpoint_period,
        }
        input_path = attempt / "input.json"
        input_sha = _write_json_atomic(input_path, input_payload)
        input_binding = {
            "path": input_path.relative_to(run_directory).as_posix(),
            "sha256": input_sha,
        }
        result_path = attempt / "algorithm_result.json"
        replay_result_path = attempt / "replay_result.json"
        substitutions = {
            "python_executable": sys.executable,
            "adapter_path": algorithm["_adapter_path"],
            "replay_verifier_path": algorithm["_replay_path"],
            "input_path": input_path,
            "configuration_path": input_path,
            "result_path": result_path,
            "replay_result_path": replay_result_path,
            "instance_path": instance_path,
            **run_key,
            "checkpoint_period": context.checkpoint_period,
            "tail_policy_path": context.tail_policy_path,
        }
        environment = _validate_environment_overrides(
            configuration_payload,
            seed=int(run_key["seed"]),
            run_key_sha=run_key_sha,
            runtime_python_path=context.runtime_python_path,
        )
        algorithm_argv = _expand_argv(
            algorithm["command_argv"], substitutions
        )
        algorithm_process = _execute_process(
            algorithm_argv,
            working_directory=attempt,
            timeout_seconds=context.timeout_seconds,
            environment=environment,
            stdout_path=attempt / "algorithm.stdout",
            stderr_path=attempt / "algorithm.stderr",
            sample_period_seconds=context.sample_period_seconds,
            cancel_event=context.cancel_event,
        )
        interrupted = bool(algorithm_process["interrupted"])
        if interrupted:
            status = "INTERRUPTED"
            reason = "Algorithm process was interrupted."
        elif algorithm_process["spawn_error"] is not None:
            status = "PROCESS_SPAWN_FAILURE"
            reason = str(algorithm_process["spawn_error"])
        elif algorithm_process["timed_out"]:
            status = "TIMEOUT"
            reason = "Algorithm process exceeded the per-row timeout."
        elif algorithm_process["exit_code"] != 0:
            status = "PROCESS_FAILURE"
            reason = (
                f"Algorithm process exited with {algorithm_process['exit_code']}."
            )
        elif algorithm_process["resource_measurement_status"] != "PASS":
            status = "RESOURCE_MEASUREMENT_FAILURE"
            reason = "Process-tree RSS measurement was unavailable."
        else:
            validated = validate_algorithm_result(
                result_path,
                expected_run_key=run_key,
                checkpoint_period=context.checkpoint_period,
            )
            result_binding = {
                "path": result_path.relative_to(run_directory).as_posix(),
                "sha256": validated["sha256"],
                "archive_path": Path(validated["archive_path"])
                .relative_to(run_directory)
                .as_posix(),
                "archive_sha256": validated["archive_sha256"],
                "checkpoint_path": Path(validated["checkpoint_path"])
                .relative_to(run_directory)
                .as_posix(),
                "checkpoint_sha256": validated["checkpoint_sha256"],
            }
            replay_argv = _expand_argv(
                algorithm["replay_verifier_argv"], substitutions
            )
            replay_process = _execute_process(
                replay_argv,
                working_directory=attempt,
                timeout_seconds=context.timeout_seconds,
                environment=environment,
                stdout_path=attempt / "replay.stdout",
                stderr_path=attempt / "replay.stderr",
                sample_period_seconds=context.sample_period_seconds,
                cancel_event=context.cancel_event,
            )
            interrupted = bool(replay_process["interrupted"])
            if interrupted:
                status = "INTERRUPTED"
                reason = "Replay process was interrupted."
            elif replay_process["spawn_error"] is not None:
                status = "REPLAY_SPAWN_FAILURE"
                reason = str(replay_process["spawn_error"])
            elif replay_process["timed_out"]:
                status = "REPLAY_TIMEOUT"
                reason = "Replay process exceeded the per-row timeout."
            elif replay_process["exit_code"] != 0:
                status = "REPLAY_PROCESS_FAILURE"
                reason = (
                    f"Replay process exited with {replay_process['exit_code']}."
                )
            elif replay_process["resource_measurement_status"] != "PASS":
                status = "REPLAY_RESOURCE_MEASUREMENT_FAILURE"
                reason = "Replay process-tree RSS measurement was unavailable."
            else:
                replay_validated = validate_replay_result(
                    replay_result_path,
                    expected_run_key=run_key,
                    checkpoint_period=context.checkpoint_period,
                    instance_sha256=instance_sha,
                    algorithm_result_sha256=str(validated["sha256"]),
                    archive_sha256=str(validated["archive_sha256"]),
                    checkpoint_artifact_sha256=str(
                        validated["checkpoint_sha256"]
                    ),
                )
                replay_binding = {
                    "path": replay_result_path.relative_to(
                        run_directory
                    ).as_posix(),
                    "sha256": replay_validated["sha256"],
                }
                status = "SUCCESS"
    except (OSError, ValueError, KeyError, TypeError) as error:
        status = (
            "RESULT_INVALID"
            if algorithm_process is not None
            and algorithm_process.get("exit_code") == 0
            and replay_process is None
            else "REPLAY_INVALID"
            if replay_process is not None
            and replay_process.get("exit_code") == 0
            else "INFRASTRUCTURE_FAILURE"
        )
        reason = f"{type(error).__name__}: {error}"

    receipt = {
        "schema": RUN_RECEIPT_SCHEMA,
        "run_key": run_key,
        "run_key_sha256": run_key_sha,
        "study_sha256": context.study_sha,
        "configuration_matrix_sha256": context.configuration_sha,
        "execution_plan_sha256": context.plan_sha,
        "freeze_receipt_sha256": context.freeze_receipt_sha,
        "tail_calibration_suite_receipt": {
            "path": str(context.tail_calibration_receipt_path),
            "sha256": context.tail_calibration_receipt_sha,
        },
        "reference_calibration_precommit": {
            "path": str(context.reference_precommit_path),
            "sha256": context.reference_precommit_sha,
        },
        "reference_calibration_completion_receipt": {
            "path": str(context.reference_completion_path),
            "sha256": context.reference_completion_sha,
        },
        "execution_scope": context.execution_scope,
        "formal_evidence_status": (
            "PENDING_POST_RUN_AUDIT"
            if context.execution_scope == "formal_candidate"
            and status == "SUCCESS"
            else "NOT_RUN"
            if context.execution_scope == "plumbing_only"
            else "NOT_ESTABLISHED"
        ),
        "attempt_number": attempt_number,
        "status": status,
        "reason": reason,
        "input_artifact": input_binding,
        "algorithm_process": algorithm_process,
        "algorithm_result": result_binding,
        "replay_process": replay_process,
        "replay_result": replay_binding,
    }
    _write_json_atomic(terminal_path, receipt)
    return status


def run_cold_process_matrix(
    study_path: str | Path,
    execution_plan_path: str | Path,
    results_directory: str | Path,
    *,
    timeout_seconds: float,
    resume: bool = False,
    plumbing_max_rows: int | None = None,
    plumbing_stratified: bool = False,
    sample_period_seconds: float = 0.05,
    workers: int = 1,
) -> ColdMatrixSummary:
    """Execute all rows, or an explicitly plumbing-only deterministic subset."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    if sample_period_seconds <= 0 or sample_period_seconds > 1:
        raise ValueError("sample_period_seconds must be in (0, 1].")
    if plumbing_max_rows is not None and plumbing_max_rows <= 0:
        raise ValueError("plumbing_max_rows must be positive.")
    if plumbing_max_rows is not None and plumbing_stratified:
        raise ValueError(
            "plumbing_max_rows and plumbing_stratified are mutually exclusive."
        )
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer.")

    study_file = Path(study_path).expanduser().resolve()
    plan_file = Path(execution_plan_path).expanduser().resolve()
    study, study_sha, configuration, config_sha, instances = _study_components(
        study_file
    )
    dependency_environment_gate = _dependency_environment_gate(
        study_file, study
    )
    plan, plan_sha, algorithms = _validate_plan(
        plan_file,
        study_sha=study_sha,
        configuration_sha=config_sha,
        instances=instances,
    )
    runtime_python_path = _validate_runtime_source_manifest(plan_file, plan)
    rows_raw = configuration.get("rows")
    if not isinstance(rows_raw, list):
        raise ValueError("Configuration matrix rows must be an array.")
    rows: list[Mapping[str, Any]] = []
    for raw in rows_raw:
        if not isinstance(raw, dict):
            raise ValueError("Every configuration row must be an object.")
        rows.append(raw)
    matrix_algorithms = {str(_run_key(row)["algorithm"]) for row in rows}
    if matrix_algorithms != set(algorithms):
        raise ValueError(
            "Execution-plan algorithms do not exactly match the matrix algorithms."
        )
    freeze_receipt_sha = _validate_freeze_receipt(
        study_path=study_file,
        plan_path=plan_file,
        study_sha=study_sha,
        configuration_sha=config_sha,
        plan_sha=plan_sha,
        expected_run_count=len(rows),
    )
    tail_policy_path, tail_policy_sha = _bound_path(
        plan_file.parent,
        plan.get("tail_policy_artifact"),
        label="tail policy artifact",
    )
    tail_calibration_receipt_path, tail_calibration_receipt_sha = _bound_path(
        plan_file.parent,
        plan.get("tail_calibration_suite_receipt"),
        label="tail calibration suite receipt",
    )
    reference_precommit_path, reference_precommit_sha = _bound_path(
        plan_file.parent,
        plan.get("reference_calibration_precommit"),
        label="reference calibration precommit",
    )
    reference_completion_path, reference_completion_sha = _bound_path(
        plan_file.parent,
        plan.get("reference_calibration_completion_receipt"),
        label="reference calibration completion receipt",
    )
    if plumbing_stratified:
        selected_rows, selection = _select_stratified_plumbing_rows(
            study, instances, rows
        )
    elif plumbing_max_rows is not None:
        selected_rows = rows[: min(plumbing_max_rows, len(rows))]
        selection = {
            "kind": "deterministic_prefix",
            "maximum_rows": plumbing_max_rows,
        }
    else:
        selected_rows = rows
        selection = {"kind": "all"}
    execution_scope = (
        "plumbing_only"
        if plumbing_max_rows is not None or plumbing_stratified
        else "formal_candidate"
    )

    result_root = Path(results_directory).expanduser().resolve()
    invocation_path = result_root / "matrix_invocation.json"
    invocation = {
        "schema": INVOCATION_SCHEMA,
        "study_sha256": study_sha,
        "configuration_matrix_sha256": config_sha,
        "execution_plan_sha256": plan_sha,
        "freeze_receipt_sha256": freeze_receipt_sha,
        "tail_calibration_suite_receipt": {
            "path": str(tail_calibration_receipt_path),
            "sha256": tail_calibration_receipt_sha,
        },
        "reference_calibration_precommit": {
            "path": str(reference_precommit_path),
            "sha256": reference_precommit_sha,
        },
        "reference_calibration_completion_receipt": {
            "path": str(reference_completion_path),
            "sha256": reference_completion_sha,
        },
        "tail_policy_artifact": {
            "path": str(tail_policy_path),
            "sha256": tail_policy_sha,
        },
        "dependency_environment_gate": dependency_environment_gate,
        "execution_scope": execution_scope,
        "formal_evidence_status": "NOT_RUN",
        "timeout_seconds": float(timeout_seconds),
        "sample_period_seconds": float(sample_period_seconds),
        "workers": workers,
        "selected_run_count": len(selected_rows),
        "expected_run_count": len(rows),
        "selection": selection,
        "python_executable": sys.executable,
    }
    if result_root.exists():
        if not resume:
            raise FileExistsError(
                f"Results directory already exists; use resume explicitly: {result_root}"
            )
        existing, _, _ = _strict_json(invocation_path)
        if existing != invocation:
            raise ValueError("Resume invocation does not match the frozen prior invocation.")
    else:
        result_root.mkdir(parents=True)
        (result_root / "runs").mkdir()
        _write_json_atomic(invocation_path, invocation)

    checkpoint_period = study.get("anytime_checkpoint_period")
    if (
        isinstance(checkpoint_period, bool)
        or not isinstance(checkpoint_period, int)
        or checkpoint_period <= 0
    ):
        raise ValueError("Study checkpoint period is invalid.")
    statuses: list[str] = []
    sequential_rows = selected_rows
    if workers > 1:
        cancel_event = threading.Event()
        context = _RowExecutionContext(
            algorithms=algorithms,
            instances=instances,
            result_root=result_root,
            study_sha=study_sha,
            configuration_sha=config_sha,
            plan_sha=plan_sha,
            freeze_receipt_sha=freeze_receipt_sha,
            execution_scope=execution_scope,
            checkpoint_period=checkpoint_period,
            timeout_seconds=timeout_seconds,
            sample_period_seconds=sample_period_seconds,
            tail_calibration_receipt_path=tail_calibration_receipt_path,
            tail_calibration_receipt_sha=tail_calibration_receipt_sha,
            reference_precommit_path=reference_precommit_path,
            reference_precommit_sha=reference_precommit_sha,
            reference_completion_path=reference_completion_path,
            reference_completion_sha=reference_completion_sha,
            tail_policy_path=tail_policy_path,
            tail_policy_sha=tail_policy_sha,
            runtime_python_path=runtime_python_path,
            cancel_event=cancel_event,
        )
        run_key_hashes = [
            _canonical_digest(_run_key(row)) for row in selected_rows
        ]
        if len(run_key_hashes) != len(set(run_key_hashes)):
            raise ValueError("Selected matrix rows contain duplicate run keys.")
        interrupted_parallel = False
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="ijoc-cold-row",
        ) as executor:
            futures = [
                executor.submit(_execute_matrix_row, row, context)
                for row in selected_rows
            ]
            try:
                for future in as_completed(futures):
                    status = future.result()
                    statuses.append(status)
                    if status == "INTERRUPTED":
                        interrupted_parallel = True
                        cancel_event.set()
            except KeyboardInterrupt:
                interrupted_parallel = True
                cancel_event.set()
                for future in futures:
                    future.cancel()
        if interrupted_parallel:
            raise KeyboardInterrupt
        sequential_rows = []
    for row in sequential_rows:
        run_key = _run_key(row)
        run_key_sha = _canonical_digest(run_key)
        run_directory = result_root / "runs" / run_key_sha
        terminal_path = run_directory / "terminal_receipt.json"
        if terminal_path.exists():
            receipt = _validate_terminal_receipt_identity(
                terminal_path,
                run_key=run_key,
                run_key_sha=run_key_sha,
                study_sha=study_sha,
                plan_sha=plan_sha,
                freeze_receipt_sha=freeze_receipt_sha,
                tail_calibration_receipt_path=tail_calibration_receipt_path,
                tail_calibration_receipt_sha=tail_calibration_receipt_sha,
                reference_precommit_path=reference_precommit_path,
                reference_precommit_sha=reference_precommit_sha,
                reference_completion_path=reference_completion_path,
                reference_completion_sha=reference_completion_sha,
            )
            statuses.append(str(receipt.get("status")))
            continue
        run_directory.mkdir(parents=True, exist_ok=True)
        attempts = run_directory / "attempts"
        attempts.mkdir(exist_ok=True)
        existing_attempts = sorted(
            child for child in attempts.iterdir() if child.is_dir()
        )
        attempt_number = len(existing_attempts) + 1
        attempt = attempts / f"{attempt_number:06d}"
        attempt.mkdir()

        algorithm_id = str(run_key["algorithm"])
        algorithm = algorithms.get(algorithm_id)
        status = "INFRASTRUCTURE_FAILURE"
        reason: str | None = None
        algorithm_process: dict[str, object] | None = None
        replay_process: dict[str, object] | None = None
        input_binding: dict[str, object] | None = None
        result_binding: dict[str, object] | None = None
        replay_binding: dict[str, object] | None = None
        interrupted = False
        try:
            if algorithm is None:
                raise ValueError(f"No execution-plan algorithm for {algorithm_id!r}.")
            configuration_payload = row.get("configuration")
            if not isinstance(configuration_payload, dict):
                raise ValueError("Configuration row readable payload is invalid.")
            if _canonical_digest(configuration_payload) != row.get(
                "configuration_sha256"
            ):
                raise ValueError("Configuration row hash mismatch.")
            instance_path, instance_sha = instances[str(run_key["case_id"])]
            input_payload = {
                "schema": INPUT_SCHEMA,
                "study_sha256": study_sha,
                "configuration_matrix_sha256": config_sha,
                "execution_plan_sha256": plan_sha,
                "freeze_receipt_sha256": freeze_receipt_sha,
                "tail_calibration_suite_receipt": {
                    "path": str(tail_calibration_receipt_path),
                    "sha256": tail_calibration_receipt_sha,
                },
                "reference_calibration_precommit": {
                    "path": str(reference_precommit_path),
                    "sha256": reference_precommit_sha,
                },
                "reference_calibration_completion_receipt": {
                    "path": str(reference_completion_path),
                    "sha256": reference_completion_sha,
                },
                "tail_policy_artifact": {
                    "path": str(tail_policy_path),
                    "sha256": tail_policy_sha,
                },
                "run_key": run_key,
                "run_key_sha256": run_key_sha,
                "configuration": configuration_payload,
                "configuration_sha256": row["configuration_sha256"],
                "instance_artifact": {
                    "path": str(instance_path),
                    "sha256": instance_sha,
                },
                "anytime_checkpoint_period": checkpoint_period,
            }
            input_path = attempt / "input.json"
            input_sha = _write_json_atomic(input_path, input_payload)
            input_binding = {
                "path": input_path.relative_to(run_directory).as_posix(),
                "sha256": input_sha,
            }
            result_path = attempt / "algorithm_result.json"
            replay_result_path = attempt / "replay_result.json"
            substitutions = {
                "python_executable": sys.executable,
                "adapter_path": algorithm["_adapter_path"],
                "replay_verifier_path": algorithm["_replay_path"],
                "input_path": input_path,
                "configuration_path": input_path,
                "result_path": result_path,
                "replay_result_path": replay_result_path,
                "instance_path": instance_path,
                **run_key,
                "checkpoint_period": checkpoint_period,
                "tail_policy_path": tail_policy_path,
            }
            environment = _validate_environment_overrides(
                configuration_payload,
                seed=int(run_key["seed"]),
                run_key_sha=run_key_sha,
                runtime_python_path=runtime_python_path,
            )
            algorithm_argv = _expand_argv(
                algorithm["command_argv"], substitutions
            )
            algorithm_process = _execute_process(
                algorithm_argv,
                working_directory=attempt,
                timeout_seconds=timeout_seconds,
                environment=environment,
                stdout_path=attempt / "algorithm.stdout",
                stderr_path=attempt / "algorithm.stderr",
                sample_period_seconds=sample_period_seconds,
            )
            interrupted = bool(algorithm_process["interrupted"])
            if interrupted:
                status = "INTERRUPTED"
                reason = "Algorithm process was interrupted."
            elif algorithm_process["spawn_error"] is not None:
                status = "PROCESS_SPAWN_FAILURE"
                reason = str(algorithm_process["spawn_error"])
            elif algorithm_process["timed_out"]:
                status = "TIMEOUT"
                reason = "Algorithm process exceeded the per-row timeout."
            elif algorithm_process["exit_code"] != 0:
                status = "PROCESS_FAILURE"
                reason = (
                    f"Algorithm process exited with {algorithm_process['exit_code']}."
                )
            elif algorithm_process["resource_measurement_status"] != "PASS":
                status = "RESOURCE_MEASUREMENT_FAILURE"
                reason = "Process-tree RSS measurement was unavailable."
            else:
                validated = validate_algorithm_result(
                    result_path,
                    expected_run_key=run_key,
                    checkpoint_period=checkpoint_period,
                )
                result_binding = {
                    "path": result_path.relative_to(run_directory).as_posix(),
                    "sha256": validated["sha256"],
                    "archive_path": Path(validated["archive_path"])
                    .relative_to(run_directory)
                    .as_posix(),
                    "archive_sha256": validated["archive_sha256"],
                    "checkpoint_path": Path(validated["checkpoint_path"])
                    .relative_to(run_directory)
                    .as_posix(),
                    "checkpoint_sha256": validated["checkpoint_sha256"],
                }
                replay_argv = _expand_argv(
                    algorithm["replay_verifier_argv"], substitutions
                )
                replay_process = _execute_process(
                    replay_argv,
                    working_directory=attempt,
                    timeout_seconds=timeout_seconds,
                    environment=environment,
                    stdout_path=attempt / "replay.stdout",
                    stderr_path=attempt / "replay.stderr",
                    sample_period_seconds=sample_period_seconds,
                )
                interrupted = bool(replay_process["interrupted"])
                if interrupted:
                    status = "INTERRUPTED"
                    reason = "Replay process was interrupted."
                elif replay_process["spawn_error"] is not None:
                    status = "REPLAY_SPAWN_FAILURE"
                    reason = str(replay_process["spawn_error"])
                elif replay_process["timed_out"]:
                    status = "REPLAY_TIMEOUT"
                    reason = "Replay process exceeded the per-row timeout."
                elif replay_process["exit_code"] != 0:
                    status = "REPLAY_PROCESS_FAILURE"
                    reason = (
                        f"Replay process exited with {replay_process['exit_code']}."
                    )
                elif replay_process["resource_measurement_status"] != "PASS":
                    status = "REPLAY_RESOURCE_MEASUREMENT_FAILURE"
                    reason = "Replay process-tree RSS measurement was unavailable."
                else:
                    replay_validated = validate_replay_result(
                        replay_result_path,
                        expected_run_key=run_key,
                        checkpoint_period=checkpoint_period,
                        instance_sha256=instance_sha,
                        algorithm_result_sha256=str(validated["sha256"]),
                        archive_sha256=str(validated["archive_sha256"]),
                        checkpoint_artifact_sha256=str(
                            validated["checkpoint_sha256"]
                        ),
                    )
                    replay_binding = {
                        "path": replay_result_path.relative_to(
                            run_directory
                        ).as_posix(),
                        "sha256": replay_validated["sha256"],
                    }
                    status = "SUCCESS"
        except (OSError, ValueError, KeyError, TypeError) as error:
            status = (
                "RESULT_INVALID"
                if algorithm_process is not None
                and algorithm_process.get("exit_code") == 0
                and replay_process is None
                else "REPLAY_INVALID"
                if replay_process is not None
                and replay_process.get("exit_code") == 0
                else "INFRASTRUCTURE_FAILURE"
            )
            reason = f"{type(error).__name__}: {error}"

        receipt = {
            "schema": RUN_RECEIPT_SCHEMA,
            "run_key": run_key,
            "run_key_sha256": run_key_sha,
            "study_sha256": study_sha,
            "configuration_matrix_sha256": config_sha,
            "execution_plan_sha256": plan_sha,
            "freeze_receipt_sha256": freeze_receipt_sha,
            "tail_calibration_suite_receipt": {
                "path": str(tail_calibration_receipt_path),
                "sha256": tail_calibration_receipt_sha,
            },
            "reference_calibration_precommit": {
                "path": str(reference_precommit_path),
                "sha256": reference_precommit_sha,
            },
            "reference_calibration_completion_receipt": {
                "path": str(reference_completion_path),
                "sha256": reference_completion_sha,
            },
            "execution_scope": execution_scope,
            "formal_evidence_status": (
                "PENDING_POST_RUN_AUDIT"
                if execution_scope == "formal_candidate" and status == "SUCCESS"
                else "NOT_RUN"
                if execution_scope == "plumbing_only"
                else "NOT_ESTABLISHED"
            ),
            "attempt_number": attempt_number,
            "status": status,
            "reason": reason,
            "input_artifact": input_binding,
            "algorithm_process": algorithm_process,
            "algorithm_result": result_binding,
            "replay_process": replay_process,
            "replay_result": replay_binding,
        }
        _write_json_atomic(terminal_path, receipt)
        statuses.append(status)
        if interrupted:
            raise KeyboardInterrupt

    success_count = sum(status == "SUCCESS" for status in statuses)
    failure_count = len(statuses) - success_count
    formal_evidence_status = (
        "NOT_RUN"
        if execution_scope == "plumbing_only"
        else "PENDING_POST_RUN_AUDIT"
        if len(statuses) == len(rows) and failure_count == 0
        else "NOT_ESTABLISHED"
    )
    summary_payload = {
        "schema": SUMMARY_SCHEMA,
        "study_sha256": study_sha,
        "configuration_matrix_sha256": config_sha,
        "execution_plan_sha256": plan_sha,
        "freeze_receipt_sha256": freeze_receipt_sha,
        "execution_scope": execution_scope,
        "formal_evidence_status": formal_evidence_status,
        "expected_run_count": len(rows),
        "selected_run_count": len(selected_rows),
        "terminal_run_count": len(statuses),
        "success_count": success_count,
        "failure_count": failure_count,
        "workers": workers,
        "submission_verdict": "HOLD_PENDING_POST_RUN_AUDIT",
        "platform": platform.platform(),
        "python_executable": sys.executable,
    }
    summary_path = result_root / "matrix_summary.json"
    _write_json_atomic(summary_path, summary_payload)
    return ColdMatrixSummary(
        results_directory=result_root,
        expected_run_count=len(rows),
        selected_run_count=len(selected_rows),
        terminal_run_count=len(statuses),
        success_count=success_count,
        failure_count=failure_count,
        execution_scope=execution_scope,
        formal_evidence_status=formal_evidence_status,
        summary_path=summary_path,
    )
