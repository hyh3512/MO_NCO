from __future__ import annotations

"""Strict batch coverage for V21e3r1 same-implementation branch replay.

The input is an append-only V21e3r1 development-diagnostic output tree.  This
program revalidates every completed diagnostic row and invokes the existing
``reexecute_and_compare`` implementation in an isolated child process.  It is
verification-only: it never establishes implementation/scientific independence
and it never authorizes selection, confirmation, or formal materialization.

Only the exact frozen 12-case x 3-seed x 14-arm (504-row) plan can emit the
full-coverage PASS.  A smoke plan can be exercised explicitly for engineering
tests, but its terminal receipt remains HOLD.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
from typing import Any, Mapping, Sequence


FULL_CASE_IDS = (
    "v21e3-mokp-development-n100-s00",
    "v21e3-mokp-development-n100-s01",
    "v21e3-mokp-development-n200-s00",
    "v21e3-mokp-development-n200-s01",
    "v21e3-mokp-development-n500-s00",
    "v21e3-mokp-development-n500-s01",
    "v21e3-motsp-development-n100-s00",
    "v21e3-motsp-development-n100-s01",
    "v21e3-motsp-development-n200-s00",
    "v21e3-motsp-development-n200-s01",
    "v21e3-motsp-development-n500-s00",
    "v21e3-motsp-development-n500-s01",
)
FULL_SEEDS = (31051, 31057, 31059)
FULL_ARMS = (
    "C0_STANDARD",
    "C0_RANDOM",
    "C0_NO_LS",
    "C0_RANDOM_NO_LS",
    "C0_SELF_REPLACE",
    "C0_POP_MATCH",
    "NSGAII_STANDARD",
    "NSGAII_SEEDED",
    "NSGAII_POP21",
    "NSGAII_SEEDED_POP21",
    "MOEAD_STANDARD",
    "MOEAD_SEEDED",
    "MOEAD_POP21",
    "MOEAD_SEEDED_POP21",
)
FULL_BUDGET = 2000
FULL_CHECKPOINT_PERIOD = 200
FULL_ROW_COUNT = 504
DIAGNOSTIC_SCOPE = "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE"
FULL_PLAN_STATUS = "FROZEN_FULL_504_DEVELOPMENT_DIAGNOSTIC"
SMOKE_PLAN_STATUS = "FROZEN_DIAGNOSTIC_SMOKE_ONLY"
FULL_DIAGNOSTIC_STATUS = "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY"
SMOKE_DIAGNOSTIC_STATUS = "PASS_DIAGNOSTIC_SMOKE_ONLY"
ROW_PASS_STATUS = "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY"
BRANCH_PASS_STATUS = "PASS_SAME_IMPLEMENTATION_FULL_BRANCH_REEXECUTION"
FULL_COVERAGE_STATUS = (
    "PASS_SAME_IMPLEMENTATION_BRANCH_REPLAY_EXACT_504_DEVELOPMENT_ONLY"
)
SMOKE_COVERAGE_STATUS = "HOLD_COMPLETE_SMOKE_COVERAGE_ONLY_NOT_EXACT_504"
INCOMPLETE_STATUS = "HOLD_INCOMPLETE_SAME_IMPLEMENTATION_BRANCH_REPLAY_COVERAGE"

INPUT_MANIFEST_SHA256 = {
    "ijoc_submission_v21e3/development_partitions_v1/case_manifest.json": (
        "1970361ba557aadd26de38aed008de11d11d158c797c00db1036cc4616cbdc8c"
    ),
    "ijoc_submission_v21e3/development_manifests_v1/reference_manifest_development.json": (
        "86336403c3e098f0e5022c796db1778552c0d92ca40d85953dc341eb534a4402"
    ),
    "ijoc_submission_v21e3/development_manifests_v1/config_manifest_development.json": (
        "d33ba2d83909af4fecff85f4663791b7c63b5ed56738a67f0eec6ccfd6336d4e"
    ),
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ATTEMPT_RE = re.compile(r"attempt-([0-9]{4,})\Z")
_PLAN_KEYS = {
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
}
_DIAGNOSTIC_COMPLETED_KEYS = {
    "attempt_directory",
    "diagnostic_sha256",
    "independent_metric_receipt_sha256",
    "plan_sha256",
    "row_id",
    "row_sha256",
    "status",
    "terminal_receipt_sha256",
    "trace_sha256",
}
_COVERAGE_COMPLETED_KEYS = {
    "schema",
    "status",
    "row_id",
    "plan_ordinal",
    "attempt_directory",
    "diagnostic_completed_marker_sha256",
    "diagnostic_plan_sha256",
    "diagnostic_trace_sha256",
    "source_snapshot_sha256",
    "branch_replay_receipt_sha256",
    "worker_spec_sha256",
    "worker_result_sha256",
    "verification_jobs",
    "implementation_independence",
    "scientific_independence",
    "third_party_replication",
    "selection_authorized",
    "confirmation_authorized",
    "formal_authorized",
}


class CoverageError(RuntimeError):
    """A strict input, replay, or coverage gate failed."""


class IsolatedProcessTimeout(CoverageError):
    """An isolated worker exceeded its deadline after its process tree was killed."""

    def __init__(
        self,
        message: str,
        *,
        timeout_seconds: int,
        stdout: str,
        stderr: str,
        isolation: str,
    ) -> None:
        super().__init__(message)
        self.timeout_seconds = timeout_seconds
        self.stdout = stdout
        self.stderr = stderr
        self.isolation = isolation


class ProcessTreeTerminationUnconfirmed(CoverageError):
    """The deadline fired but OS evidence cannot prove whole-tree termination."""

    def __init__(
        self,
        message: str,
        *,
        timeout_seconds: int,
        stdout: str,
        stderr: str,
        isolation: str,
    ) -> None:
        super().__init__(message)
        self.timeout_seconds = timeout_seconds
        self.stdout = stdout
        self.stderr = stderr
        self.isolation = isolation


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CoverageError(f"Duplicate JSON key is prohibited: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise CoverageError(f"Non-finite JSON constant is prohibited: {value}")


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise CoverageError(f"Required JSON artifact is absent: {path}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CoverageError(f"JSON artifact is not UTF-8: {path}") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise CoverageError(f"JSON artifact is malformed: {path}") from error
    if not isinstance(value, dict):
        raise CoverageError(f"Expected a JSON object: {path}")
    return value, _sha256_bytes(raw)


def _exclusive_json(path: Path, payload: object) -> str:
    raw = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256_bytes(raw)


def _exclusive_canonical_json(path: Path, payload: object) -> str:
    """Create a canonical JSON artifact for strict downstream consumers."""
    raw = _canonical_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256_bytes(raw)


def _require_exact_keys(
    payload: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    observed = set(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise CoverageError(f"{label} key set drifted; missing={missing}, extra={extra}")


def _require_string(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise CoverageError(f"{label} must be an exact nonempty string.")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CoverageError(f"{label} must be an exact integer >= {minimum}.")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise CoverageError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _require_string_list(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise CoverageError(f"{label} must be a nonempty JSON array.")
    result = tuple(value)
    if any(type(item) is not str or not item for item in result):
        raise CoverageError(f"{label} must contain exact nonempty strings.")
    if len(set(result)) != len(result):
        raise CoverageError(f"{label} must not contain duplicates.")
    return result


def _require_int_list(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise CoverageError(f"{label} must be a nonempty JSON array.")
    result = tuple(value)
    if any(type(item) is not int or item < 0 for item in result):
        raise CoverageError(f"{label} must contain exact nonnegative integers.")
    if len(set(result)) != len(result):
        raise CoverageError(f"{label} must not contain duplicates.")
    return result


def _canonical_relative_path(value: object, *, label: str) -> PurePosixPath:
    raw = _require_string(value, label=label)
    path = PurePosixPath(raw)
    if (
        "\\" in raw
        or path.is_absolute()
        or path.as_posix() != raw
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in path.parts[0]
    ):
        raise CoverageError(f"{label} is not a canonical contained POSIX path.")
    return path


def _contained_path(root: Path, relative: object, *, label: str) -> Path:
    posix = _canonical_relative_path(relative, label=label)
    candidate = (root / Path(*posix.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise CoverageError(f"{label} escapes its root.") from error
    return candidate


def _assert_disjoint_outputs(diagnostic_root: Path, coverage_root: Path) -> None:
    try:
        coverage_root.relative_to(diagnostic_root)
    except ValueError:
        pass
    else:
        raise CoverageError("Coverage output must not be inside the diagnostic tree.")
    try:
        diagnostic_root.relative_to(coverage_root)
    except ValueError:
        pass
    else:
        raise CoverageError("Coverage output must not contain the diagnostic tree.")


def _validate_source_entries(
    source_manifest: Mapping[str, object],
    *,
    project_root: Path,
    normalized: bool,
    expected_root: str | None = None,
) -> tuple[list[dict[str, object]], str]:
    expected_keys = (
        {"schema", "source_root_sha256", "entries"}
        if normalized
        else {
            "schema",
            "hash_rule",
            "entry_count",
            "entries",
            "source_snapshot_sha256",
        }
    )
    _require_exact_keys(source_manifest, expected_keys, label="source manifest")
    expected_schema = (
        "v21e3r1_branch_replay_source_manifest_binding_v1"
        if normalized
        else "v21e3r1_diagnostic_source_manifest_v1"
    )
    if source_manifest.get("schema") != expected_schema:
        raise CoverageError("Source-manifest schema drifted.")
    if not normalized and source_manifest.get("hash_rule") != (
        "sha256(canonical_json(sorted_entries))"
    ):
        raise CoverageError("Diagnostic source-manifest hash rule drifted.")
    raw_entries = source_manifest.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise CoverageError("Source manifest must contain a nonempty entries array.")
    entries: list[dict[str, object]] = []
    paths: list[str] = []
    for ordinal, raw_entry in enumerate(raw_entries, start=1):
        if not isinstance(raw_entry, dict):
            raise CoverageError(f"Source entry {ordinal} is not an object.")
        _require_exact_keys(
            raw_entry, {"path", "bytes", "sha256"}, label=f"source entry {ordinal}"
        )
        relative = _canonical_relative_path(
            raw_entry.get("path"), label=f"source entry {ordinal} path"
        ).as_posix()
        declared_bytes = _require_int(
            raw_entry.get("bytes"), label=f"source entry {ordinal} bytes", minimum=1
        )
        declared_sha = _require_sha256(
            raw_entry.get("sha256"), label=f"source entry {ordinal} sha256"
        )
        path = _contained_path(project_root, relative, label="source entry path")
        if not path.is_file():
            raise CoverageError(f"Frozen source file is absent: {relative}")
        if path.stat().st_size != declared_bytes or _sha256_file(path) != declared_sha:
            raise CoverageError(f"Frozen source file drifted: {relative}")
        entries.append(
            {"path": relative, "bytes": declared_bytes, "sha256": declared_sha}
        )
        paths.append(relative)
    if len(set(path.casefold() for path in paths)) != len(paths):
        raise CoverageError("Source manifest contains duplicate paths.")
    if paths != sorted(paths, key=lambda item: item.casefold()):
        raise CoverageError("Source-manifest entries are not in frozen sorted order.")
    declared_root = _require_sha256(
        source_manifest.get(
            "source_root_sha256" if normalized else "source_snapshot_sha256"
        ),
        label="source snapshot root",
    )
    observed_root = _sha256_bytes(_canonical_bytes(entries))
    if declared_root != observed_root:
        raise CoverageError("Source snapshot root does not match its inventory.")
    if expected_root is not None and declared_root != expected_root:
        raise CoverageError("Source snapshot root disagrees with the expected root.")
    if not normalized and source_manifest.get("entry_count") != len(entries):
        raise CoverageError("Source-manifest entry_count drifted.")
    required = {
        "mo_nco/pareto_v21e3_hybrid.py",
        "mo_nco/pareto_v21e3_baselines.py",
        "mo_nco/pareto_v21e3r1_branch_replay.py",
    }
    if not required.issubset(paths):
        raise CoverageError("Source manifest omits an executing replay module.")
    return entries, declared_root


def _validate_input_and_cases(
    project_root: Path,
    plan: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    binding = plan.get("input_binding")
    if not isinstance(binding, dict):
        raise CoverageError("Diagnostic plan omits input_binding.")
    _require_exact_keys(
        binding,
        {"schema", "manifest_sha256", "case_ids"},
        label="diagnostic input binding",
    )
    if binding.get("schema") != "v21e3r1_exposed_development_input_binding_v1":
        raise CoverageError("Diagnostic input-binding schema drifted.")
    if binding.get("manifest_sha256") != INPUT_MANIFEST_SHA256:
        raise CoverageError("Frozen development-manifest binding drifted.")
    if binding.get("case_ids") != list(FULL_CASE_IDS):
        raise CoverageError("Input binding does not bind all frozen development cases.")
    for relative, expected_sha in INPUT_MANIFEST_SHA256.items():
        path = _contained_path(project_root, relative, label="input manifest path")
        if not path.is_file() or _sha256_file(path) != expected_sha:
            raise CoverageError(f"Frozen input manifest drifted: {relative}")

    manifest_path = project_root / (
        "ijoc_submission_v21e3/development_partitions_v1/case_manifest.json"
    )
    case_manifest, _ = _load_json_object(manifest_path)
    if (
        case_manifest.get("split") != "development"
        or case_manifest.get("formal_confirmatory_eligibility") is not False
    ):
        raise CoverageError("Case manifest is not exposed development-only evidence.")
    raw_cases = case_manifest.get("cases")
    if not isinstance(raw_cases, list) or not all(
        isinstance(item, dict) for item in raw_cases
    ):
        raise CoverageError("Case manifest has an invalid cases array.")
    if tuple(item.get("case_id") for item in raw_cases) != FULL_CASE_IDS:
        raise CoverageError("Case-manifest IDs/order drifted.")
    cases: dict[str, dict[str, object]] = {}
    manifest_root = manifest_path.parent.resolve()
    for raw_case in raw_cases:
        case_id = _require_string(raw_case.get("case_id"), label="case_id")
        family = _require_string(raw_case.get("family"), label=f"{case_id}.family")
        size = _require_int(raw_case.get("size"), label=f"{case_id}.size", minimum=1)
        if family not in {"MOKP", "MOTSP"}:
            raise CoverageError(f"Unsupported frozen family: {family}")
        artifact = raw_case.get("artifact")
        if not isinstance(artifact, dict):
            raise CoverageError(f"Case omits artifact binding: {case_id}")
        _require_exact_keys(
            artifact, {"path", "bytes", "sha256"}, label=f"{case_id} artifact"
        )
        path = _contained_path(
            manifest_root, artifact.get("path"), label=f"{case_id} artifact path"
        )
        declared_bytes = _require_int(
            artifact.get("bytes"), label=f"{case_id} artifact bytes", minimum=1
        )
        declared_sha = _require_sha256(
            artifact.get("sha256"), label=f"{case_id} artifact sha256"
        )
        if (
            not path.is_file()
            or path.stat().st_size != declared_bytes
            or _sha256_file(path) != declared_sha
        ):
            raise CoverageError(f"Frozen case artifact drifted: {case_id}")
        cases[case_id] = {
            "family": family,
            "size": size,
            "path": path,
            "sha256": declared_sha,
            "bytes": declared_bytes,
        }
    return cases


def _load_plan_contract(
    *,
    project_root: Path,
    diagnostic_plan_path: Path,
    allow_smoke: bool,
) -> dict[str, object]:
    plan, plan_sha256 = _load_json_object(diagnostic_plan_path)
    _require_exact_keys(plan, _PLAN_KEYS, label="diagnostic plan")
    if plan.get("schema") != "v21e3r1_exposed_development_diagnostic_plan_v2":
        raise CoverageError("Diagnostic-plan schema drifted.")
    if plan.get("scientific_scope") != DIAGNOSTIC_SCOPE:
        raise CoverageError("Diagnostic-plan scientific scope drifted.")
    for field in (
        "selection_entropy_release",
        "confirmation_materialization",
        "formal_materialization",
    ):
        if plan.get(field) != "PROHIBITED":
            raise CoverageError(f"Diagnostic plan does not prohibit {field}.")
    cases = _require_string_list(plan.get("case_ids"), label="plan.case_ids")
    seeds = _require_int_list(plan.get("seeds"), label="plan.seeds")
    arms = _require_string_list(plan.get("arms"), label="plan.arms")
    if any(case_id not in FULL_CASE_IDS for case_id in cases):
        raise CoverageError("Plan includes a non-development case.")
    if tuple(case_id for case_id in FULL_CASE_IDS if case_id in cases) != cases:
        raise CoverageError("Plan case IDs do not preserve frozen order.")
    if any(arm not in FULL_ARMS for arm in arms):
        raise CoverageError("Plan includes an unknown diagnostic arm.")
    budget = _require_int(
        plan.get("charged_evaluation_budget"),
        label="plan.charged_evaluation_budget",
        minimum=1,
    )
    checkpoint = _require_int(
        plan.get("checkpoint_period"), label="plan.checkpoint_period", minimum=1
    )
    if budget % checkpoint != 0:
        raise CoverageError("Diagnostic budget is not divisible by checkpoint period.")
    _require_int(plan.get("row_timeout_seconds"), label="plan.row_timeout_seconds", minimum=1)
    expected_rows = _require_int(
        plan.get("expected_rows"), label="plan.expected_rows", minimum=1
    )
    if expected_rows != len(cases) * len(seeds) * len(arms):
        raise CoverageError("Diagnostic expected_rows disagrees with its Cartesian plan.")
    exact_full = (
        plan.get("status") == FULL_PLAN_STATUS
        and cases == FULL_CASE_IDS
        and seeds == FULL_SEEDS
        and arms == FULL_ARMS
        and budget == FULL_BUDGET
        and checkpoint == FULL_CHECKPOINT_PERIOD
        and expected_rows == FULL_ROW_COUNT
    )
    if plan.get("status") == FULL_PLAN_STATUS and not exact_full:
        raise CoverageError("A declared full plan is not the exact frozen 504-row plan.")
    if not exact_full:
        if plan.get("status") != SMOKE_PLAN_STATUS:
            raise CoverageError("Diagnostic plan has an unsupported status.")
        if not allow_smoke:
            raise CoverageError(
                "Only the exact full 504-row plan is accepted unless allow_smoke=True."
            )

    source_manifest = plan.get("source_manifest")
    if not isinstance(source_manifest, dict):
        raise CoverageError("Diagnostic plan omits its source manifest.")
    entries, source_root = _validate_source_entries(
        source_manifest, project_root=project_root, normalized=False
    )
    case_map = _validate_input_and_cases(project_root, plan)
    row_specs: list[dict[str, object]] = []
    ordinal = 0
    for case_id in cases:
        case = case_map[case_id]
        for seed in seeds:
            for arm in arms:
                ordinal += 1
                row_id = f"{case_id}__seed-{seed}__arm-{arm.lower()}"
                _canonical_relative_path(row_id, label="row_id")
                row_specs.append(
                    {
                        "row_id": row_id,
                        "ordinal": ordinal,
                        "case_id": case_id,
                        "family": case["family"],
                        "size": case["size"],
                        "case_path": case["path"],
                        "case_sha256": case["sha256"],
                        "seed": seed,
                        "arm_id": arm,
                        "budget": budget,
                        "checkpoint": checkpoint,
                    }
                )
    if len(row_specs) != expected_rows:
        raise AssertionError("Validated row construction disagrees with plan cardinality.")
    return {
        "plan": plan,
        "plan_sha256": plan_sha256,
        "exact_full": exact_full,
        "matrix_mode": "FULL_504" if exact_full else "SMOKE_ONLY",
        "expected_rows": expected_rows,
        "source_entries": entries,
        "source_root": source_root,
        "row_specs": row_specs,
    }


def _validate_independent_metric(
    payload: Mapping[str, object], *, trace_sha256: str, budget: int
) -> None:
    if (
        payload.get("schema") != "v21e3r1_independent_metric_reimplementation_v2"
        or payload.get("status") != "PASS_INDEPENDENT_METRIC_IMPLEMENTATION"
        or payload.get("trace_sha256") != trace_sha256
        or payload.get("evaluation_count") != budget
        or payload.get("decision_count") != budget
        or payload.get("terminal_accounting_gate") != "PASS"
        or payload.get("algorithm_execution_independence") is not False
        or payload.get("scientific_independence") is not False
        or payload.get("implementation_independence_from_project_metrics") is not True
    ):
        raise CoverageError("Independent metric receipt fails strict row gates.")


def _validate_terminal_receipt(
    payload: Mapping[str, object], *, case_id: str, family: str, budget: int
) -> None:
    gates = payload.get("finalization_gates")
    if not isinstance(gates, dict):
        raise CoverageError("Terminal receipt omits finalization_gates.")
    if (
        payload.get("schema") != "v21e3_terminal_receipt_v1"
        or payload.get("status") != "SUCCESS"
        or payload.get("problem") != case_id
        or payload.get("family") != family
        or payload.get("charged_evaluation_count") != budget
        or payload.get("decision_count") != budget
        or payload.get("physical_call_started_count") != budget
        or payload.get("unresolved_decision_count") != 0
        or gates.get("sqlite_integrity") != "ok"
        or gates.get("expected_charged_evaluations") != budget
        or gates.get("persisted_evaluations") != budget
        or gates.get("expected_decisions") != budget
        or gates.get("persisted_decisions") != budget
        or gates.get("nonterminal_attempts") != 0
    ):
        raise CoverageError("Terminal receipt fails strict accounting gates.")


def _validate_diagnostic_completed(
    *,
    diagnostic_root: Path,
    row_spec: Mapping[str, object],
    diagnostic_plan_sha256: str,
    source_root: str,
) -> dict[str, object]:
    row_id = str(row_spec["row_id"])
    marker_path = diagnostic_root / "completed" / f"{row_id}.json"
    marker, marker_sha256 = _load_json_object(marker_path)
    _require_exact_keys(
        marker, _DIAGNOSTIC_COMPLETED_KEYS, label=f"diagnostic marker {row_id}"
    )
    if (
        marker.get("status") != ROW_PASS_STATUS
        or marker.get("row_id") != row_id
        or marker.get("plan_sha256") != diagnostic_plan_sha256
    ):
        raise CoverageError(f"Diagnostic completed marker binding failed: {row_id}")
    for field in (
        "row_sha256",
        "diagnostic_sha256",
        "trace_sha256",
        "terminal_receipt_sha256",
        "independent_metric_receipt_sha256",
    ):
        _require_sha256(marker.get(field), label=f"{row_id}.{field}")
    attempt_relative = _canonical_relative_path(
        marker.get("attempt_directory"), label=f"{row_id}.attempt_directory"
    )
    expected_prefix = PurePosixPath("attempts") / row_id
    if (
        len(attempt_relative.parts) != 3
        or PurePosixPath(*attempt_relative.parts[:2]) != expected_prefix
        or _ATTEMPT_RE.fullmatch(attempt_relative.parts[2]) is None
    ):
        raise CoverageError(f"Diagnostic attempt path is not canonical: {row_id}")
    attempt = _contained_path(
        diagnostic_root, attempt_relative.as_posix(), label="diagnostic attempt"
    )
    artifact_fields = {
        "row.json": "row_sha256",
        "diagnostic.json": "diagnostic_sha256",
        "trace.sqlite3": "trace_sha256",
        "terminal.receipt.json": "terminal_receipt_sha256",
        "independent.metric.json": "independent_metric_receipt_sha256",
    }
    for name, field in artifact_fields.items():
        artifact = attempt / name
        if not artifact.is_file() or _sha256_file(artifact) != marker[field]:
            raise CoverageError(f"Diagnostic artifact drifted: {row_id}/{name}")

    row, _ = _load_json_object(attempt / "row.json")
    required_row_values = {
        "schema": "v21e3r1_exposed_development_diagnostic_row_v2",
        "status": ROW_PASS_STATUS,
        "scientific_scope": DIAGNOSTIC_SCOPE,
        "case_id": row_spec["case_id"],
        "family": row_spec["family"],
        "size": row_spec["size"],
        "seed": row_spec["seed"],
        "arm_id": row_spec["arm_id"],
        "charged_evaluation_budget": row_spec["budget"],
        "checkpoint_period": row_spec["checkpoint"],
        "case_artifact_sha256": row_spec["case_sha256"],
        "source_snapshot_sha256": source_root,
        "plan_sha256": diagnostic_plan_sha256,
        "trace_database_path": "trace.sqlite3",
        "trace_database_sha256": marker["trace_sha256"],
        "terminal_receipt_path": "terminal.receipt.json",
        "terminal_receipt_sha256": marker["terminal_receipt_sha256"],
        "independent_metric_receipt_path": "independent.metric.json",
        "independent_metric_receipt_sha256": marker[
            "independent_metric_receipt_sha256"
        ],
        "selection_entropy_release": "PROHIBITED",
        "confirmation_materialization": "PROHIBITED",
        "formal_materialization": "PROHIBITED",
    }
    for field, expected in required_row_values.items():
        if row.get(field) != expected:
            raise CoverageError(f"Diagnostic row binding drifted: {row_id}/{field}")
    if not isinstance(row.get("algorithm_config"), dict):
        raise CoverageError(f"Diagnostic row omits algorithm_config: {row_id}")
    diagnostic, _ = _load_json_object(attempt / "diagnostic.json")
    if (
        diagnostic.get("schema") != "v21e3r1_existing_trace_diagnostic_v1"
        or diagnostic.get("status") != ROW_PASS_STATUS
        or diagnostic.get("scientific_scope") != DIAGNOSTIC_SCOPE
        or diagnostic.get("case_id") != row_spec["case_id"]
        or diagnostic.get("family") != row_spec["family"]
        or diagnostic.get("size") != row_spec["size"]
        or diagnostic.get("seed") != row_spec["seed"]
        or diagnostic.get("arm_id") != row_spec["arm_id"]
        or diagnostic.get("budget") != row_spec["budget"]
    ):
        raise CoverageError(f"Diagnostic analysis receipt drifted: {row_id}")
    independent, _ = _load_json_object(attempt / "independent.metric.json")
    _validate_independent_metric(
        independent,
        trace_sha256=str(marker["trace_sha256"]),
        budget=int(row_spec["budget"]),
    )
    if row.get("independent_metric_replay") != independent:
        raise CoverageError(f"Embedded independent metric receipt drifted: {row_id}")
    terminal, _ = _load_json_object(attempt / "terminal.receipt.json")
    _validate_terminal_receipt(
        terminal,
        case_id=str(row_spec["case_id"]),
        family=str(row_spec["family"]),
        budget=int(row_spec["budget"]),
    )
    return {
        "marker": marker,
        "marker_path": marker_path,
        "marker_sha256": marker_sha256,
        "attempt": attempt,
        "trace": attempt / "trace.sqlite3",
    }


def _job_object_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: int,
) -> dict[str, object]:
    """Run a Windows child suspended, bind its full tree to a kill-on-close job."""

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
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

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    ntdll.NtResumeProcess.restype = ctypes.c_long

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise CoverageError(
            f"CreateJobObjectW failed: WinError {ctypes.get_last_error()}"
        )
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        job, 9, ctypes.byref(info), ctypes.sizeof(info)
    ):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        raise CoverageError(f"SetInformationJobObject failed: WinError {error}")

    process: subprocess.Popen[str] | None = None
    job_is_open = True

    def terminate_assigned_tree(exit_code: int) -> str | None:
        """Return the proven tree-termination mechanism, or ``None``."""

        nonlocal job_is_open
        if kernel32.TerminateJobObject(job, exit_code):
            return "TERMINATE_JOB_OBJECT"
        # KILL_ON_JOB_CLOSE is an independent kernel-enforced fallback.  The
        # handle is non-inheritable, so this process owns the last open handle.
        if kernel32.CloseHandle(job):
            job_is_open = False
            return "KILL_ON_JOB_CLOSE"
        return None

    try:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            # CPython does not export CREATE_SUSPENDED on every Windows build.
            # These are the stable Win32 CreateProcess flag values.
            creationflags=0x00000004 | 0x00000200,
        )
        process_handle = wintypes.HANDLE(int(process._handle))  # type: ignore[attr-defined]
        if not kernel32.AssignProcessToJobObject(job, process_handle):
            error = ctypes.get_last_error()
            # The process was deliberately created suspended, so it cannot
            # have spawned descendants.  Because assignment itself failed it
            # is not yet covered by TerminateJobObject; kill it explicitly.
            try:
                process.kill()
                process.communicate(timeout=10)
            except Exception as termination_error:
                raise ProcessTreeTerminationUnconfirmed(
                    "Job assignment failed and the suspended worker could not be reaped.",
                    timeout_seconds=timeout_seconds,
                    stdout="",
                    stderr=str(termination_error),
                    isolation="WINDOWS_SUSPENDED_WORKER_TERMINATION_UNCONFIRMED_V1",
                ) from termination_error
            if process.poll() is None:
                raise ProcessTreeTerminationUnconfirmed(
                    "Job assignment failed and suspended-worker termination is unconfirmed.",
                    timeout_seconds=timeout_seconds,
                    stdout="",
                    stderr="",
                    isolation="WINDOWS_SUSPENDED_WORKER_TERMINATION_UNCONFIRMED_V1",
                )
            raise CoverageError(f"AssignProcessToJobObject failed: WinError {error}")
        status = ntdll.NtResumeProcess(process_handle)
        if status != 0:
            termination = terminate_assigned_tree(0xE0000003)
            if termination is None:
                process.kill()
                process.communicate(timeout=10)
                raise ProcessTreeTerminationUnconfirmed(
                    "NtResumeProcess failed and assigned-tree termination is unconfirmed.",
                    timeout_seconds=timeout_seconds,
                    stdout="",
                    stderr="",
                    isolation="WINDOWS_JOB_TREE_TERMINATION_UNCONFIRMED_V1",
                )
            process.communicate(timeout=10)
            raise CoverageError(f"NtResumeProcess failed: NTSTATUS 0x{status & 0xffffffff:08x}")
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            termination = terminate_assigned_tree(0xE0000001)
            if termination is None:
                # Kill/reap the known parent as hygiene, but do not mislabel
                # that weaker action as proof that the whole tree was killed.
                process.kill()
                stdout, stderr = process.communicate(timeout=15)
                raise ProcessTreeTerminationUnconfirmed(
                    "Worker timed out and Windows could not confirm whole-tree termination.",
                    timeout_seconds=timeout_seconds,
                    stdout=(error.stdout or "") + (stdout or ""),
                    stderr=(error.stderr or "") + (stderr or ""),
                    isolation="WINDOWS_JOB_TREE_TERMINATION_UNCONFIRMED_V1",
                ) from error
            try:
                stdout, stderr = process.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            raise IsolatedProcessTimeout(
                "Worker exceeded its timeout; the Windows Job Object was terminated.",
                timeout_seconds=timeout_seconds,
                stdout=(error.stdout or "") + (stdout or ""),
                stderr=(error.stderr or "") + (stderr or ""),
                isolation=(
                    "WINDOWS_SUSPENDED_KILL_ON_CLOSE_JOB_OBJECT_V1:"
                    + termination
                ),
            ) from error
        return {
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "isolation": "WINDOWS_SUSPENDED_KILL_ON_CLOSE_JOB_OBJECT_V1",
        }
    finally:
        # Closing a kill-on-close job also removes any descendant left after a
        # nominal worker exit.  The worker process itself has already been read.
        if job_is_open:
            closed = kernel32.CloseHandle(job)
            if not closed and sys.exc_info()[0] is None:
                raise CoverageError(
                    "CloseHandle failed; descendant cleanup cannot be confirmed."
                )


def _posix_process_group(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: int,
) -> dict[str, object]:
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        raise IsolatedProcessTimeout(
            "Worker exceeded its timeout; the POSIX process group was terminated.",
            timeout_seconds=timeout_seconds,
            stdout=(error.stdout or "") + (stdout or ""),
            stderr=(error.stderr or "") + (stderr or ""),
            isolation="POSIX_NEW_SESSION_PROCESS_GROUP_V1",
        ) from error
    return {
        "returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "isolation": "POSIX_NEW_SESSION_PROCESS_GROUP_V1",
    }


def run_isolated_process(
    command: Sequence[str],
    *,
    cwd: str | Path,
    env: Mapping[str, str] | None = None,
    timeout_seconds: int,
) -> dict[str, object]:
    """Run one command under a deadline that kills the entire process tree."""

    if (
        not isinstance(command, (list, tuple))
        or not command
        or any(type(item) is not str or not item for item in command)
    ):
        raise CoverageError("command must contain exact nonempty strings.")
    timeout = _require_int(timeout_seconds, label="timeout_seconds", minimum=1)
    directory = Path(cwd).resolve()
    if not directory.is_dir():
        raise CoverageError(f"Worker cwd is not a directory: {directory}")
    environment = dict(os.environ if env is None else env)
    if os.name == "nt":
        return _job_object_process(
            command, cwd=directory, env=environment, timeout_seconds=timeout
        )
    return _posix_process_group(
        command, cwd=directory, env=environment, timeout_seconds=timeout
    )


def _validate_branch_receipt(
    receipt: Mapping[str, object],
    *,
    trace_sha256: str,
    case_sha256: str,
    source_root: str,
    source_manifest_sha256: str,
) -> None:
    checks = receipt.get("checks")
    artifacts = receipt.get("artifacts")
    problem = receipt.get("problem_binding")
    source = receipt.get("source_binding")
    if not all(isinstance(item, dict) for item in (checks, artifacts, problem, source)):
        raise CoverageError("Branch replay receipt omits strict binding objects.")
    assert isinstance(checks, dict)
    assert isinstance(artifacts, dict)
    assert isinstance(problem, dict)
    assert isinstance(source, dict)
    if not checks or any(value is not True for value in checks.values()):
        raise CoverageError("Branch replay semantic comparison is not all-true.")
    if (
        receipt.get("schema") != "v21e3r1_same_implementation_branch_replay_v1"
        or receipt.get("status") != BRANCH_PASS_STATUS
        or receipt.get("implementation_independence") is not False
        or receipt.get("scientific_independence") is not False
        or receipt.get("third_party_replication") is not False
        or receipt.get("scope")
        != "same_source_stochastic_program_reexecution_not_independent_replication"
        or artifacts.get("original_database_sha256") != trace_sha256
        or problem.get("problem_artifact_sha256") != case_sha256
        or source.get("context_source_sha256") != source_root
        or source.get("source_manifest_sha256") != source_manifest_sha256
        or source.get("replay_verified") is not True
    ):
        raise CoverageError("Branch replay receipt fails strict same-source gates.")
    declared_payload_sha = _require_sha256(
        receipt.get("receipt_payload_sha256"), label="branch receipt payload digest"
    )
    unsigned = dict(receipt)
    del unsigned["receipt_payload_sha256"]
    if _sha256_bytes(_canonical_bytes(unsigned)) != declared_payload_sha:
        raise CoverageError("Branch replay receipt payload digest drifted.")


def _worker_run(spec_path: str | Path) -> dict[str, object]:
    spec_file = Path(spec_path).resolve()
    spec, _ = _load_json_object(spec_file)
    expected_keys = {
        "schema",
        "project_root",
        "diagnostic_output_root",
        "coverage_output_root",
        "row_id",
        "plan_ordinal",
        "trace_path",
        "trace_sha256",
        "case_path",
        "case_sha256",
        "source_manifest_path",
        "source_manifest_sha256",
        "source_snapshot_sha256",
        "diagnostic_plan_sha256",
        "verification_jobs",
    }
    _require_exact_keys(spec, expected_keys, label="branch replay worker spec")
    if spec.get("schema") != "v21e3r1_branch_replay_coverage_worker_spec_v1":
        raise CoverageError("Branch replay worker-spec schema drifted.")
    project_root = Path(_require_string(spec.get("project_root"), label="project_root")).resolve()
    diagnostic_root = Path(
        _require_string(spec.get("diagnostic_output_root"), label="diagnostic_output_root")
    ).resolve()
    coverage_root = Path(
        _require_string(spec.get("coverage_output_root"), label="coverage_output_root")
    ).resolve()
    _assert_disjoint_outputs(diagnostic_root, coverage_root)
    trace = Path(_require_string(spec.get("trace_path"), label="trace_path")).resolve()
    case = Path(_require_string(spec.get("case_path"), label="case_path")).resolve()
    source_manifest_path = Path(
        _require_string(spec.get("source_manifest_path"), label="source_manifest_path")
    ).resolve()
    try:
        trace.relative_to(diagnostic_root)
        case.relative_to(project_root)
        source_manifest_path.relative_to(coverage_root)
        spec_file.relative_to(coverage_root)
    except ValueError as error:
        raise CoverageError("Worker input path escaped its declared root.") from error
    trace_sha = _require_sha256(spec.get("trace_sha256"), label="trace_sha256")
    case_sha = _require_sha256(spec.get("case_sha256"), label="case_sha256")
    source_manifest_sha = _require_sha256(
        spec.get("source_manifest_sha256"), label="source_manifest_sha256"
    )
    source_root = _require_sha256(
        spec.get("source_snapshot_sha256"), label="source_snapshot_sha256"
    )
    if (
        not trace.is_file()
        or _sha256_file(trace) != trace_sha
        or not case.is_file()
        or _sha256_file(case) != case_sha
        or not source_manifest_path.is_file()
        or _sha256_file(source_manifest_path) != source_manifest_sha
    ):
        raise CoverageError("Worker pre-replay artifact hash gate failed.")
    source_manifest, _ = _load_json_object(source_manifest_path)
    _validate_source_entries(
        source_manifest,
        project_root=project_root,
        normalized=True,
        expected_root=source_root,
    )

    from mo_nco.pareto_v21e3r1_branch_replay import reexecute_and_compare

    output_receipt = spec_file.parent / "branch.replay.json"
    receipt = reexecute_and_compare(
        original_database=trace,
        problem_artifact=case,
        output_receipt=output_receipt,
        source_manifest_path=source_manifest_path,
    )
    if (
        _sha256_file(trace) != trace_sha
        or _sha256_file(case) != case_sha
        or _sha256_file(source_manifest_path) != source_manifest_sha
    ):
        raise CoverageError("Worker input changed during branch replay.")
    source_manifest_after, _ = _load_json_object(source_manifest_path)
    _validate_source_entries(
        source_manifest_after,
        project_root=project_root,
        normalized=True,
        expected_root=source_root,
    )
    _validate_branch_receipt(
        receipt,
        trace_sha256=trace_sha,
        case_sha256=case_sha,
        source_root=source_root,
        source_manifest_sha256=source_manifest_sha,
    )
    return {
        "schema": "v21e3r1_branch_replay_coverage_worker_result_v1",
        "status": BRANCH_PASS_STATUS,
        "row_id": spec["row_id"],
        "plan_ordinal": spec["plan_ordinal"],
        "diagnostic_plan_sha256": spec["diagnostic_plan_sha256"],
        "diagnostic_trace_sha256": trace_sha,
        "case_artifact_sha256": case_sha,
        "source_snapshot_sha256": source_root,
        "source_manifest_sha256": source_manifest_sha,
        "branch_replay_receipt_sha256": _sha256_file(output_receipt),
        "branch_replay_payload_sha256": receipt["receipt_payload_sha256"],
        "verification_jobs": spec["verification_jobs"],
        "implementation_independence": False,
        "scientific_independence": False,
        "third_party_replication": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_authorized": False,
        "runtime_efficiency_claims": False,
        "scientific_performance_claims": False,
    }


def _next_attempt(coverage_root: Path, row_id: str) -> Path:
    root = coverage_root / "attempts" / row_id
    root.mkdir(parents=True, exist_ok=True)
    used: list[int] = []
    for path in root.iterdir():
        if not path.is_dir():
            raise CoverageError(f"Unexpected non-directory in attempt ledger: {path}")
        match = _ATTEMPT_RE.fullmatch(path.name)
        if match is None:
            raise CoverageError(f"Unexpected attempt directory name: {path}")
        used.append(int(match.group(1)))
    attempt = root / f"attempt-{max(used, default=0) + 1:04d}"
    attempt.mkdir()
    return attempt


def _failure_payload(
    *, status: str, row_id: str, result: Mapping[str, object] | None, detail: str
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "v21e3r1_branch_replay_coverage_failure_v1",
        "status": status,
        "row_id": row_id,
        "detail": detail[-4000:],
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
    if result is not None:
        payload.update(
            {
                "returncode": result.get("returncode"),
                "stdout_tail": str(result.get("stdout", ""))[-4000:],
                "stderr_tail": str(result.get("stderr", ""))[-4000:],
                "process_tree_isolation": result.get("isolation"),
            }
        )
    return payload


def _validate_worker_result(
    *,
    attempt: Path,
    row_spec: Mapping[str, object],
    diagnostic: Mapping[str, object],
    diagnostic_plan_sha256: str,
    source_root: str,
    source_manifest_sha256: str,
    jobs: int,
) -> tuple[dict[str, object], str, str]:
    result, result_sha = _load_json_object(attempt / "worker.result.json")
    expected_keys = {
        "schema",
        "status",
        "row_id",
        "plan_ordinal",
        "diagnostic_plan_sha256",
        "diagnostic_trace_sha256",
        "case_artifact_sha256",
        "source_snapshot_sha256",
        "source_manifest_sha256",
        "branch_replay_receipt_sha256",
        "branch_replay_payload_sha256",
        "verification_jobs",
        "implementation_independence",
        "scientific_independence",
        "third_party_replication",
        "selection_authorized",
        "confirmation_authorized",
        "formal_authorized",
        "runtime_efficiency_claims",
        "scientific_performance_claims",
    }
    _require_exact_keys(result, expected_keys, label="branch replay worker result")
    expected = {
        "schema": "v21e3r1_branch_replay_coverage_worker_result_v1",
        "status": BRANCH_PASS_STATUS,
        "row_id": row_spec["row_id"],
        "plan_ordinal": row_spec["ordinal"],
        "diagnostic_plan_sha256": diagnostic_plan_sha256,
        "diagnostic_trace_sha256": diagnostic["marker"]["trace_sha256"],
        "case_artifact_sha256": row_spec["case_sha256"],
        "source_snapshot_sha256": source_root,
        "source_manifest_sha256": source_manifest_sha256,
        "verification_jobs": jobs,
        "implementation_independence": False,
        "scientific_independence": False,
        "third_party_replication": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_authorized": False,
        "runtime_efficiency_claims": False,
        "scientific_performance_claims": False,
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise CoverageError(f"Worker-result binding drifted at {field}.")
    receipt_sha = _require_sha256(
        result.get("branch_replay_receipt_sha256"),
        label="branch replay receipt sha256",
    )
    _require_sha256(
        result.get("branch_replay_payload_sha256"),
        label="branch replay payload sha256",
    )
    receipt_path = attempt / "branch.replay.json"
    if not receipt_path.is_file() or _sha256_file(receipt_path) != receipt_sha:
        raise CoverageError("Branch replay receipt file drifted.")
    receipt, _ = _load_json_object(receipt_path)
    _validate_branch_receipt(
        receipt,
        trace_sha256=str(diagnostic["marker"]["trace_sha256"]),
        case_sha256=str(row_spec["case_sha256"]),
        source_root=source_root,
        source_manifest_sha256=source_manifest_sha256,
    )
    if receipt.get("receipt_payload_sha256") != result.get(
        "branch_replay_payload_sha256"
    ):
        raise CoverageError("Worker and branch receipt payload digests disagree.")
    return result, result_sha, receipt_sha


def _process_one_row(
    *,
    project_root: Path,
    diagnostic_root: Path,
    coverage_root: Path,
    row_spec: Mapping[str, object],
    diagnostic_plan_sha256: str,
    source_root: str,
    source_manifest_path: Path,
    source_manifest_sha256: str,
    jobs: int,
    timeout_seconds: int,
) -> dict[str, object]:
    row_id = str(row_spec["row_id"])
    diagnostic = _validate_diagnostic_completed(
        diagnostic_root=diagnostic_root,
        row_spec=row_spec,
        diagnostic_plan_sha256=diagnostic_plan_sha256,
        source_root=source_root,
    )
    attempt = _next_attempt(coverage_root, row_id)
    spec = {
        "schema": "v21e3r1_branch_replay_coverage_worker_spec_v1",
        "project_root": str(project_root),
        "diagnostic_output_root": str(diagnostic_root),
        "coverage_output_root": str(coverage_root),
        "row_id": row_id,
        "plan_ordinal": row_spec["ordinal"],
        "trace_path": str(diagnostic["trace"]),
        "trace_sha256": diagnostic["marker"]["trace_sha256"],
        "case_path": str(row_spec["case_path"]),
        "case_sha256": row_spec["case_sha256"],
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": source_manifest_sha256,
        "source_snapshot_sha256": source_root,
        "diagnostic_plan_sha256": diagnostic_plan_sha256,
        "verification_jobs": jobs,
    }
    spec_path = attempt / "worker.spec.json"
    spec_sha = _exclusive_json(spec_path, spec)
    environment = dict(os.environ)
    old_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(project_root)
        if not old_path
        else str(project_root) + os.pathsep + old_path
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-spec",
        str(spec_path),
    ]
    try:
        process_result = run_isolated_process(
            command,
            cwd=project_root,
            env=environment,
            timeout_seconds=timeout_seconds,
        )
    except ProcessTreeTerminationUnconfirmed as error:
        failure = _failure_payload(
            status="FAIL_ROW_TIMEOUT_PROCESS_TREE_TERMINATION_UNCONFIRMED",
            row_id=row_id,
            result={
                "returncode": None,
                "stdout": error.stdout,
                "stderr": error.stderr,
                "isolation": error.isolation,
            },
            detail=str(error),
        )
        failure["timeout_seconds"] = error.timeout_seconds
        _exclusive_json(attempt / "failure.receipt.json", failure)
        raise
    except IsolatedProcessTimeout as error:
        failure = _failure_payload(
            status="FAIL_ROW_TIMEOUT_PROCESS_TREE_TERMINATED",
            row_id=row_id,
            result={
                "returncode": None,
                "stdout": error.stdout,
                "stderr": error.stderr,
                "isolation": error.isolation,
            },
            detail=str(error),
        )
        failure["timeout_seconds"] = error.timeout_seconds
        _exclusive_json(attempt / "failure.receipt.json", failure)
        raise
    except CoverageError as error:
        _exclusive_json(
            attempt / "failure.receipt.json",
            _failure_payload(
                status="FAIL_ROW_PROCESS_ISOLATION_SETUP",
                row_id=row_id,
                result=None,
                detail=str(error),
            ),
        )
        raise
    except Exception as error:
        _exclusive_json(
            attempt / "failure.receipt.json",
            _failure_payload(
                status="FAIL_ROW_PROCESS_LAUNCH_OR_ISOLATION",
                row_id=row_id,
                result=None,
                detail=str(error),
            ),
        )
        raise
    if process_result["returncode"] != 0:
        _exclusive_json(
            attempt / "failure.receipt.json",
            _failure_payload(
                status="FAIL_ROW_PROCESS",
                row_id=row_id,
                result=process_result,
                detail="Same-implementation replay worker exited nonzero.",
            ),
        )
        raise CoverageError(
            f"Branch replay worker failed for {row_id}: "
            + str(process_result.get("stderr", ""))[-2000:]
        )
    # Repeat all upstream artifact gates after replay, not only in the child.
    diagnostic_after = _validate_diagnostic_completed(
        diagnostic_root=diagnostic_root,
        row_spec=row_spec,
        diagnostic_plan_sha256=diagnostic_plan_sha256,
        source_root=source_root,
    )
    if diagnostic_after["marker_sha256"] != diagnostic["marker_sha256"]:
        raise CoverageError(f"Diagnostic marker changed during replay: {row_id}")
    result, result_sha, receipt_sha = _validate_worker_result(
        attempt=attempt,
        row_spec=row_spec,
        diagnostic=diagnostic_after,
        diagnostic_plan_sha256=diagnostic_plan_sha256,
        source_root=source_root,
        source_manifest_sha256=source_manifest_sha256,
        jobs=jobs,
    )
    completed = {
        "schema": "v21e3r1_branch_replay_coverage_completed_row_v1",
        "status": BRANCH_PASS_STATUS,
        "row_id": row_id,
        "plan_ordinal": row_spec["ordinal"],
        "attempt_directory": attempt.relative_to(coverage_root).as_posix(),
        "diagnostic_completed_marker_sha256": diagnostic["marker_sha256"],
        "diagnostic_plan_sha256": diagnostic_plan_sha256,
        "diagnostic_trace_sha256": diagnostic["marker"]["trace_sha256"],
        "source_snapshot_sha256": source_root,
        "branch_replay_receipt_sha256": receipt_sha,
        "worker_spec_sha256": spec_sha,
        "worker_result_sha256": result_sha,
        "verification_jobs": jobs,
        "implementation_independence": False,
        "scientific_independence": False,
        "third_party_replication": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_authorized": False,
    }
    _exclusive_json(coverage_root / "completed" / f"{row_id}.json", completed)
    return completed


def _validate_coverage_completed(
    *,
    coverage_root: Path,
    diagnostic_root: Path,
    row_spec: Mapping[str, object],
    diagnostic_plan_sha256: str,
    source_root: str,
    source_manifest_sha256: str,
) -> tuple[dict[str, object], str]:
    row_id = str(row_spec["row_id"])
    path = coverage_root / "completed" / f"{row_id}.json"
    marker, marker_sha = _load_json_object(path)
    _require_exact_keys(marker, _COVERAGE_COMPLETED_KEYS, label="coverage marker")
    fixed = {
        "schema": "v21e3r1_branch_replay_coverage_completed_row_v1",
        "status": BRANCH_PASS_STATUS,
        "row_id": row_id,
        "plan_ordinal": row_spec["ordinal"],
        "diagnostic_plan_sha256": diagnostic_plan_sha256,
        "source_snapshot_sha256": source_root,
        "implementation_independence": False,
        "scientific_independence": False,
        "third_party_replication": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_authorized": False,
    }
    for field, expected in fixed.items():
        if marker.get(field) != expected:
            raise CoverageError(f"Coverage marker drifted at {row_id}/{field}")
    jobs = _require_int(
        marker.get("verification_jobs"), label="coverage verification_jobs", minimum=1
    )
    del jobs
    for field in (
        "diagnostic_completed_marker_sha256",
        "diagnostic_trace_sha256",
        "branch_replay_receipt_sha256",
        "worker_spec_sha256",
        "worker_result_sha256",
    ):
        _require_sha256(marker.get(field), label=f"coverage marker {field}")
    attempt_relative = _canonical_relative_path(
        marker.get("attempt_directory"), label="coverage attempt_directory"
    )
    if (
        len(attempt_relative.parts) != 3
        or attempt_relative.parts[0] != "attempts"
        or attempt_relative.parts[1] != row_id
        or _ATTEMPT_RE.fullmatch(attempt_relative.parts[2]) is None
    ):
        raise CoverageError(f"Coverage attempt path drifted: {row_id}")
    attempt = _contained_path(
        coverage_root, attempt_relative.as_posix(), label="coverage attempt"
    )
    spec_path = attempt / "worker.spec.json"
    result_path = attempt / "worker.result.json"
    receipt_path = attempt / "branch.replay.json"
    if (
        not spec_path.is_file()
        or _sha256_file(spec_path) != marker["worker_spec_sha256"]
        or not result_path.is_file()
        or _sha256_file(result_path) != marker["worker_result_sha256"]
        or not receipt_path.is_file()
        or _sha256_file(receipt_path) != marker["branch_replay_receipt_sha256"]
    ):
        raise CoverageError(f"Coverage attempt artifact drifted: {row_id}")
    diagnostic = _validate_diagnostic_completed(
        diagnostic_root=diagnostic_root,
        row_spec=row_spec,
        diagnostic_plan_sha256=diagnostic_plan_sha256,
        source_root=source_root,
    )
    if (
        marker["diagnostic_completed_marker_sha256"]
        != diagnostic["marker_sha256"]
        or marker["diagnostic_trace_sha256"]
        != diagnostic["marker"]["trace_sha256"]
    ):
        raise CoverageError(f"Coverage marker upstream binding drifted: {row_id}")
    result, _, receipt_sha = _validate_worker_result(
        attempt=attempt,
        row_spec=row_spec,
        diagnostic=diagnostic,
        diagnostic_plan_sha256=diagnostic_plan_sha256,
        source_root=source_root,
        source_manifest_sha256=source_manifest_sha256,
        jobs=int(marker["verification_jobs"]),
    )
    del result
    if receipt_sha != marker["branch_replay_receipt_sha256"]:
        raise CoverageError(f"Coverage receipt binding drifted: {row_id}")
    return marker, marker_sha


def _scan_marker_names(root: Path, *, expected: set[str], label: str) -> set[str]:
    if not root.exists():
        return set()
    if not root.is_dir():
        raise CoverageError(f"{label} marker path is not a directory: {root}")
    observed: set[str] = set()
    for path in root.iterdir():
        if not path.is_file() or path.suffix != ".json":
            raise CoverageError(f"Unexpected entry in {label} marker directory: {path}")
        row_id = path.stem
        if row_id not in expected:
            raise CoverageError(f"Unexpected {label} row key: {row_id}")
        observed.add(row_id)
    return observed


def _validate_diagnostic_final(
    *,
    diagnostic_root: Path,
    contract: Mapping[str, object],
) -> tuple[str, str]:
    receipt_path = diagnostic_root / "diagnostic.receipt.json"
    aggregate_path = diagnostic_root / "diagnostic.aggregate.json"
    receipt, receipt_sha = _load_json_object(receipt_path)
    _require_exact_keys(
        receipt,
        {
            "schema",
            "status",
            "scientific_scope",
            "matrix_mode",
            "completed_rows",
            "expected_rows",
            "plan_sha256",
            "source_snapshot_sha256",
            "aggregate_sha256",
            "selection_entropy_release",
            "confirmation_materialization",
            "formal_materialization",
        },
        label="diagnostic final receipt",
    )
    expected_status = (
        FULL_DIAGNOSTIC_STATUS if contract["exact_full"] else SMOKE_DIAGNOSTIC_STATUS
    )
    expected = {
        "schema": "v21e3r1_exposed_development_diagnostic_receipt_v2",
        "status": expected_status,
        "scientific_scope": DIAGNOSTIC_SCOPE,
        "matrix_mode": contract["matrix_mode"],
        "completed_rows": contract["expected_rows"],
        "expected_rows": contract["expected_rows"],
        "plan_sha256": contract["plan_sha256"],
        "source_snapshot_sha256": contract["source_root"],
        "selection_entropy_release": "PROHIBITED",
        "confirmation_materialization": "PROHIBITED",
        "formal_materialization": "PROHIBITED",
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise CoverageError(f"Diagnostic final receipt drifted at {field}.")
    if not aggregate_path.is_file():
        raise CoverageError("Diagnostic aggregate is absent.")
    aggregate_sha = _sha256_file(aggregate_path)
    if receipt.get("aggregate_sha256") != aggregate_sha:
        raise CoverageError("Diagnostic aggregate hash binding drifted.")
    return receipt_sha, aggregate_sha


def _base_authority_fields() -> dict[str, object]:
    return {
        "implementation_independence": False,
        "scientific_independence": False,
        "third_party_replication": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_authorized": False,
        "selection_entropy_release": "PROHIBITED",
        "confirmation_materialization": "PROHIBITED",
        "formal_materialization": "PROHIBITED",
        "runtime_efficiency_claims": False,
        "scientific_performance_claims": False,
        "ijoc_submission_status": "IJOC_HOLD",
    }


def _provisional_result(
    *,
    contract: Mapping[str, object],
    diagnostic_completed: int,
    replay_completed: int,
    jobs: int,
) -> dict[str, object]:
    return {
        "schema": "v21e3r1_branch_replay_coverage_progress_v1",
        "status": INCOMPLETE_STATUS,
        "matrix_mode": contract["matrix_mode"],
        "expected_rows": contract["expected_rows"],
        "diagnostic_completed_rows": diagnostic_completed,
        "completed_rows": replay_completed,
        "verification_jobs": jobs,
        "parallel_execution_semantics": (
            "VERIFICATION_ONLY_PLAN_ORDERED_FINALIZATION_NO_RUNTIME_OR_PERFORMANCE_CLAIM"
        ),
        **_base_authority_fields(),
    }


def _build_row_seals(
    *,
    coverage_root: Path,
    diagnostic_root: Path,
    contract: Mapping[str, object],
    source_manifest_sha256: str,
) -> tuple[list[dict[str, object]], list[int]]:
    seals: list[dict[str, object]] = []
    observed_jobs: list[int] = []
    for row_spec in contract["row_specs"]:
        marker, marker_sha = _validate_coverage_completed(
            coverage_root=coverage_root,
            diagnostic_root=diagnostic_root,
            row_spec=row_spec,
            diagnostic_plan_sha256=str(contract["plan_sha256"]),
            source_root=str(contract["source_root"]),
            source_manifest_sha256=source_manifest_sha256,
        )
        observed_jobs.append(int(marker["verification_jobs"]))
        seals.append(
            {
                "row_id": marker["row_id"],
                "plan_ordinal": marker["plan_ordinal"],
                "coverage_completed_marker_sha256": marker_sha,
                "diagnostic_completed_marker_sha256": marker[
                    "diagnostic_completed_marker_sha256"
                ],
                "diagnostic_trace_sha256": marker["diagnostic_trace_sha256"],
                "branch_replay_receipt_sha256": marker[
                    "branch_replay_receipt_sha256"
                ],
            }
        )
    return seals, sorted(set(observed_jobs))


def _validate_existing_final(
    *,
    receipt_path: Path,
    coverage_root: Path,
    diagnostic_root: Path,
    contract: Mapping[str, object],
    source_manifest_sha256: str,
) -> dict[str, object]:
    receipt, _ = _load_json_object(receipt_path)
    _require_exact_keys(
        receipt,
        {
            "schema",
            "status",
            "scientific_scope",
            "matrix_mode",
            "completed_rows",
            "expected_rows",
            "exact_full_504_coverage",
            "diagnostic_plan_sha256",
            "diagnostic_receipt_sha256",
            "diagnostic_aggregate_sha256",
            "source_snapshot_sha256",
            "source_manifest_path",
            "source_manifest_sha256",
            "row_order_rule",
            "row_seals",
            "row_seals_sha256",
            "verification_jobs",
            "verification_jobs_observed",
            "row_timeout_seconds",
            "parallel_execution_semantics",
            "implementation_independence",
            "scientific_independence",
            "third_party_replication",
            "selection_authorized",
            "confirmation_authorized",
            "formal_authorized",
            "selection_entropy_release",
            "confirmation_materialization",
            "formal_materialization",
            "runtime_efficiency_claims",
            "scientific_performance_claims",
            "ijoc_submission_status",
        },
        label="final coverage receipt",
    )
    diagnostic_receipt_sha, diagnostic_aggregate_sha = _validate_diagnostic_final(
        diagnostic_root=diagnostic_root, contract=contract
    )
    seals, observed_jobs = _build_row_seals(
        coverage_root=coverage_root,
        diagnostic_root=diagnostic_root,
        contract=contract,
        source_manifest_sha256=source_manifest_sha256,
    )
    expected_status = FULL_COVERAGE_STATUS if contract["exact_full"] else SMOKE_COVERAGE_STATUS
    if (
        receipt.get("schema") != "v21e3r1_branch_replay_coverage_receipt_v1"
        or receipt.get("status") != expected_status
        or receipt.get("scientific_scope") != DIAGNOSTIC_SCOPE
        or receipt.get("matrix_mode") != contract["matrix_mode"]
        or receipt.get("completed_rows") != contract["expected_rows"]
        or receipt.get("expected_rows") != contract["expected_rows"]
        or receipt.get("exact_full_504_coverage") != bool(contract["exact_full"])
        or receipt.get("diagnostic_plan_sha256") != contract["plan_sha256"]
        or receipt.get("diagnostic_receipt_sha256") != diagnostic_receipt_sha
        or receipt.get("diagnostic_aggregate_sha256") != diagnostic_aggregate_sha
        or receipt.get("source_snapshot_sha256") != contract["source_root"]
        or receipt.get("source_manifest_path") != "source.manifest.json"
        or receipt.get("source_manifest_sha256") != source_manifest_sha256
        or receipt.get("row_order_rule")
        != "FROZEN_DIAGNOSTIC_PLAN_CASE_SEED_ARM_ORDER"
        or receipt.get("row_seals") != seals
        or receipt.get("row_seals_sha256") != _sha256_bytes(_canonical_bytes(seals))
        or receipt.get("verification_jobs_observed") != observed_jobs
    ):
        raise CoverageError("Existing final coverage receipt fails strict validation.")
    _require_int(receipt.get("verification_jobs"), label="final verification_jobs", minimum=1)
    _require_int(
        receipt.get("row_timeout_seconds"),
        label="final row_timeout_seconds",
        minimum=1,
    )
    authority = _base_authority_fields()
    if any(receipt.get(field) != value for field, value in authority.items()):
        raise CoverageError("Existing final receipt would expand scientific authority.")
    if receipt.get("parallel_execution_semantics") != (
        "VERIFICATION_ONLY_PLAN_ORDERED_FINALIZATION_NO_RUNTIME_OR_PERFORMANCE_CLAIM"
    ):
        raise CoverageError("Existing final receipt parallel semantics drifted.")
    return receipt


def run_coverage(
    project_root: str | Path,
    diagnostic_output_root: str | Path,
    coverage_output_root: str | Path,
    *,
    diagnostic_plan_path: str | Path | None = None,
    allow_smoke: bool = False,
    resume: bool = False,
    jobs: int = 1,
    row_timeout_seconds: int = 1800,
) -> dict[str, object]:
    """Verify and replay every currently completed row in plan order."""

    if type(allow_smoke) is not bool or type(resume) is not bool:
        raise CoverageError("allow_smoke and resume must be exact booleans.")
    jobs = _require_int(jobs, label="jobs", minimum=1)
    timeout = _require_int(
        row_timeout_seconds, label="row_timeout_seconds", minimum=1
    )
    project = Path(project_root).resolve()
    diagnostic_root = Path(diagnostic_output_root).resolve()
    coverage_root = Path(coverage_output_root).resolve()
    if not project.is_dir():
        raise CoverageError(f"Project root is not a directory: {project}")
    if not diagnostic_root.is_dir():
        raise CoverageError(f"Diagnostic output root is not a directory: {diagnostic_root}")
    _assert_disjoint_outputs(diagnostic_root, coverage_root)
    plan_path = (
        diagnostic_root / "diagnostic.plan.json"
        if diagnostic_plan_path is None
        else Path(diagnostic_plan_path).resolve()
    )
    contract = _load_plan_contract(
        project_root=project,
        diagnostic_plan_path=plan_path,
        allow_smoke=allow_smoke,
    )
    row_specs = list(contract["row_specs"])
    expected_ids = {str(item["row_id"]) for item in row_specs}
    frozen_coverage_plan = {
        "schema": "v21e3r1_branch_replay_coverage_plan_v1",
        "status": "FROZEN_SAME_IMPLEMENTATION_REPLAY_COVERAGE",
        "scientific_scope": DIAGNOSTIC_SCOPE,
        "project_root": str(project),
        "diagnostic_output_root": str(diagnostic_root),
        "diagnostic_plan_path": str(plan_path),
        "diagnostic_plan_sha256": contract["plan_sha256"],
        "matrix_mode": contract["matrix_mode"],
        "expected_rows": contract["expected_rows"],
        "row_keys_in_plan_order": [item["row_id"] for item in row_specs],
        "source_snapshot_sha256": contract["source_root"],
        **{
            key: value
            for key, value in _base_authority_fields().items()
            if key
            in {
                "implementation_independence",
                "scientific_independence",
                "third_party_replication",
                "selection_authorized",
                "confirmation_authorized",
                "formal_authorized",
            }
        },
    }
    normalized_source = {
        "schema": "v21e3r1_branch_replay_source_manifest_binding_v1",
        "source_root_sha256": contract["source_root"],
        "entries": contract["source_entries"],
    }
    coverage_plan_path = coverage_root / "branch_replay_coverage.plan.json"
    source_manifest_path = coverage_root / "source.manifest.json"
    if coverage_root.exists():
        if not resume:
            raise FileExistsError(coverage_root)
        existing_plan, _ = _load_json_object(coverage_plan_path)
        if existing_plan != frozen_coverage_plan:
            raise CoverageError("Resume coverage plan disagrees with frozen inputs.")
        existing_source, source_manifest_sha = _load_json_object(source_manifest_path)
        if existing_source != normalized_source:
            raise CoverageError("Resume source manifest disagrees with frozen source.")
        _validate_source_entries(
            existing_source,
            project_root=project,
            normalized=True,
            expected_root=str(contract["source_root"]),
        )
    else:
        coverage_root.mkdir(parents=True)
        _exclusive_json(coverage_plan_path, frozen_coverage_plan)
        source_manifest_sha = _exclusive_canonical_json(
            source_manifest_path, normalized_source
        )

    final_path = coverage_root / "branch_replay_coverage.receipt.json"
    diagnostic_names = _scan_marker_names(
        diagnostic_root / "completed", expected=expected_ids, label="diagnostic"
    )
    coverage_names = _scan_marker_names(
        coverage_root / "completed", expected=expected_ids, label="coverage"
    )
    diagnostic_final_path = diagnostic_root / "diagnostic.receipt.json"
    if diagnostic_final_path.exists():
        if diagnostic_names != expected_ids:
            raise CoverageError(
                "Diagnostic final receipt claims completion without exact row markers."
            )
        _validate_diagnostic_final(diagnostic_root=diagnostic_root, contract=contract)
    # Validate all resumed rows before trusting them or returning an existing final.
    spec_by_id = {str(item["row_id"]): item for item in row_specs}
    for row_id in coverage_names:
        if row_id not in diagnostic_names:
            raise CoverageError(f"Coverage row lacks a diagnostic completed row: {row_id}")
        _validate_coverage_completed(
            coverage_root=coverage_root,
            diagnostic_root=diagnostic_root,
            row_spec=spec_by_id[row_id],
            diagnostic_plan_sha256=str(contract["plan_sha256"]),
            source_root=str(contract["source_root"]),
            source_manifest_sha256=source_manifest_sha,
        )
    if final_path.exists():
        if coverage_names != expected_ids or diagnostic_names != expected_ids:
            raise CoverageError("Final receipt exists without exact row-key coverage.")
        _validate_diagnostic_final(diagnostic_root=diagnostic_root, contract=contract)
        return _validate_existing_final(
            receipt_path=final_path,
            coverage_root=coverage_root,
            diagnostic_root=diagnostic_root,
            contract=contract,
            source_manifest_sha256=source_manifest_sha,
        )

    pending = [
        item
        for item in row_specs
        if item["row_id"] in diagnostic_names and item["row_id"] not in coverage_names
    ]
    failures: list[tuple[str, Exception]] = []
    if jobs == 1:
        for item in pending:
            try:
                _process_one_row(
                    project_root=project,
                    diagnostic_root=diagnostic_root,
                    coverage_root=coverage_root,
                    row_spec=item,
                    diagnostic_plan_sha256=str(contract["plan_sha256"]),
                    source_root=str(contract["source_root"]),
                    source_manifest_path=source_manifest_path,
                    source_manifest_sha256=source_manifest_sha,
                    jobs=jobs,
                    timeout_seconds=timeout,
                )
                print(
                    f"replayed {item['ordinal']}/{contract['expected_rows']} {item['row_id']}",
                    flush=True,
                )
            except Exception as error:  # preserve append-only attempts, then fail
                failures.append((str(item["row_id"]), error))
                break
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    _process_one_row,
                    project_root=project,
                    diagnostic_root=diagnostic_root,
                    coverage_root=coverage_root,
                    row_spec=item,
                    diagnostic_plan_sha256=str(contract["plan_sha256"]),
                    source_root=str(contract["source_root"]),
                    source_manifest_path=source_manifest_path,
                    source_manifest_sha256=source_manifest_sha,
                    jobs=jobs,
                    timeout_seconds=timeout,
                ): item
                for item in pending
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    future.result()
                    print(
                        f"replayed {item['ordinal']}/{contract['expected_rows']} {item['row_id']}",
                        flush=True,
                    )
                except Exception as error:
                    failures.append((str(item["row_id"]), error))
    if failures:
        row_id, error = failures[0]
        raise CoverageError(
            f"{len(failures)} branch replay row(s) failed; first={row_id}: {error}"
        ) from error

    diagnostic_names = _scan_marker_names(
        diagnostic_root / "completed", expected=expected_ids, label="diagnostic"
    )
    coverage_names = _scan_marker_names(
        coverage_root / "completed", expected=expected_ids, label="coverage"
    )
    if diagnostic_names != expected_ids or coverage_names != expected_ids:
        return _provisional_result(
            contract=contract,
            diagnostic_completed=len(diagnostic_names),
            replay_completed=len(coverage_names),
            jobs=jobs,
        )

    # The diagnostic runner publishes its aggregate before its exclusive final
    # receipt.  Exact rows observed during that narrow window remain HOLD and
    # can be finalized by a later --resume invocation.
    if not diagnostic_final_path.is_file():
        return _provisional_result(
            contract=contract,
            diagnostic_completed=len(diagnostic_names),
            replay_completed=len(coverage_names),
            jobs=jobs,
        )

    diagnostic_receipt_sha, diagnostic_aggregate_sha = _validate_diagnostic_final(
        diagnostic_root=diagnostic_root, contract=contract
    )
    seals, observed_jobs = _build_row_seals(
        coverage_root=coverage_root,
        diagnostic_root=diagnostic_root,
        contract=contract,
        source_manifest_sha256=source_manifest_sha,
    )
    if [seal["row_id"] for seal in seals] != [
        item["row_id"] for item in row_specs
    ]:
        raise AssertionError("Final row seals are not sorted in frozen plan order.")
    final = {
        "schema": "v21e3r1_branch_replay_coverage_receipt_v1",
        "status": FULL_COVERAGE_STATUS if contract["exact_full"] else SMOKE_COVERAGE_STATUS,
        "scientific_scope": DIAGNOSTIC_SCOPE,
        "matrix_mode": contract["matrix_mode"],
        "completed_rows": contract["expected_rows"],
        "expected_rows": contract["expected_rows"],
        "exact_full_504_coverage": bool(contract["exact_full"]),
        "diagnostic_plan_sha256": contract["plan_sha256"],
        "diagnostic_receipt_sha256": diagnostic_receipt_sha,
        "diagnostic_aggregate_sha256": diagnostic_aggregate_sha,
        "source_snapshot_sha256": contract["source_root"],
        "source_manifest_path": "source.manifest.json",
        "source_manifest_sha256": source_manifest_sha,
        "row_order_rule": "FROZEN_DIAGNOSTIC_PLAN_CASE_SEED_ARM_ORDER",
        "row_seals": seals,
        "row_seals_sha256": _sha256_bytes(_canonical_bytes(seals)),
        "verification_jobs": jobs,
        "verification_jobs_observed": observed_jobs,
        "row_timeout_seconds": timeout,
        "parallel_execution_semantics": (
            "VERIFICATION_ONLY_PLAN_ORDERED_FINALIZATION_NO_RUNTIME_OR_PERFORMANCE_CLAIM"
        ),
        **_base_authority_fields(),
    }
    _exclusive_json(final_path, final)
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--diagnostic-output-root")
    parser.add_argument("--diagnostic-plan")
    parser.add_argument("--coverage-output-root")
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--row-timeout-seconds", type=int, default=1800)
    parser.add_argument("--worker-spec", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_spec:
        result = _worker_run(args.worker_spec)
        _exclusive_json(
            Path(args.worker_spec).resolve().parent / "worker.result.json", result
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if not args.diagnostic_output_root:
        parser.error("--diagnostic-output-root is required")
    if not args.coverage_output_root:
        parser.error("--coverage-output-root is required")
    result = run_coverage(
        project_root=args.project_root,
        diagnostic_output_root=args.diagnostic_output_root,
        diagnostic_plan_path=args.diagnostic_plan,
        coverage_output_root=args.coverage_output_root,
        allow_smoke=args.allow_smoke,
        resume=args.resume,
        jobs=args.jobs,
        row_timeout_seconds=args.row_timeout_seconds,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 3


if __name__ == "__main__":
    raise SystemExit(main())
