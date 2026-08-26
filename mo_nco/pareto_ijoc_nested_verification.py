from __future__ import annotations

"""Independent, fail-closed verification of nested IJOC evidence manifests."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tarfile
import time
from typing import BinaryIO, Mapping, Sequence


RECEIPT_SCHEMA = "ijoc_nested_evidence_verification_receipt_v1"
CONSUMED_SCHEMA = "ijoc_formal_analysis_consumed_artifacts_v1"
SOURCE_MANIFEST_SCHEMA = "ijoc_source_file_manifest_v1"

TOP_LEVEL_FROZEN = frozenset(
    {
        "algorithm_configuration_matrix",
        "execution_plan",
        "formal_analysis_plan",
        "freeze_receipt",
        "metric_reference_manifest",
        "study",
    }
)
TOP_LEVEL_RESULTS = frozenset({"matrix_invocation", "post_run_audit"})
TOP_LEVEL_INPUTS = TOP_LEVEL_FROZEN | TOP_LEVEL_RESULTS
ROW_ARTIFACTS = frozenset(
    {
        "algorithm_result",
        "all_evaluated_archive",
        "checkpoint_witnesses",
        "replay_receipt",
        "terminal_receipt",
    }
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class NestedEvidenceError(ValueError):
    """Raised when verification cannot safely produce a receipt."""


@dataclass(frozen=True)
class NestedEvidenceExpectations:
    """Frozen cardinalities for the current formal evidence packet."""

    row_entries: int = 3600
    row_artifacts: int = 18000
    top_level_inputs: int = 8
    metric_reference_sources: int = 30
    source_files: int = 690

    def __post_init__(self) -> None:
        for name in (
            "row_entries",
            "row_artifacts",
            "top_level_inputs",
            "metric_reference_sources",
            "source_files",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise NestedEvidenceError(
                    f"Expected count {name} must be a nonnegative integer."
                )


@dataclass(frozen=True)
class NestedEvidenceVerification:
    receipt_path: Path
    receipt_sha256: str
    status: str
    issue_count: int
    elapsed_seconds: float


@dataclass(frozen=True)
class _HashTask:
    category: str
    identity: str
    path: Path
    declared: Mapping[str, object]


@dataclass(frozen=True)
class _HashResult:
    category: str
    identity: str
    path: Path
    bytes: int | None
    sha256: str | None
    error: str | None


def canonical_json_bytes(
    value: object, *, trailing_newline: bool = True
) -> bytes:
    """Encode canonical compact UTF-8 JSON used by receipts and set digests."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise NestedEvidenceError(
            "Value is not canonical-JSON serializable."
        ) from error
    if trailing_newline:
        text += "\n"
    return text.encode("utf-8")


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        canonical_json_bytes(value, trailing_newline=False)
    ).hexdigest()


def _pretty_manifest_bytes(value: object) -> bytes:
    try:
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
    except (TypeError, ValueError) as error:
        raise NestedEvidenceError("Manifest is not JSON serializable.") from error


def _strict_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise NestedEvidenceError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise NestedEvidenceError(f"Non-finite JSON constant {value!r}.")


def _read_strict_json(path: Path, *, label: str) -> tuple[object, bytes]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise NestedEvidenceError(f"Cannot read {label}: {path}.") from error
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, NestedEvidenceError) as error:
        raise NestedEvidenceError(f"{label} is not strict UTF-8 JSON: {error}") from error
    return value, raw


def _issue(
    issues: list[dict[str, object]],
    code: str,
    location: str,
    message: str,
    *,
    expected: object | None = None,
    observed: object | None = None,
) -> None:
    entry: dict[str, object] = {
        "code": code,
        "location": location,
        "message": message,
    }
    if expected is not None:
        entry["expected"] = expected
    if observed is not None:
        entry["observed"] = observed
    issues.append(entry)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _safe_relative(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise NestedEvidenceError(f"{label} must be nonempty text.")
    if "\\" in value:
        raise NestedEvidenceError(f"{label} must use forward slashes.")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise NestedEvidenceError(f"{label} must be relative.")
    if value != posix.as_posix() or any(
        part in {"", ".", ".."} for part in posix.parts
    ):
        raise NestedEvidenceError(f"{label} is not a canonical relative path.")
    return value


def _resolve_inside(root: Path, relative: str, *, label: str) -> Path:
    base = root.resolve(strict=True)
    try:
        candidate = (
            base / Path(*PurePosixPath(relative).parts)
        ).resolve(strict=True)
    except OSError as error:
        raise NestedEvidenceError(f"{label} is missing: {relative}.") from error
    try:
        candidate.relative_to(base)
    except ValueError as error:
        raise NestedEvidenceError(f"{label} escapes its declared root.") from error
    if not candidate.is_file():
        raise NestedEvidenceError(f"{label} is not a regular file.")
    return candidate


def _resolve_input(root: Path, value: str | Path, *, label: str) -> Path:
    base = root.resolve(strict=True)
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        candidate = candidate.resolve(strict=True)
    except OSError as error:
        raise NestedEvidenceError(f"{label} is missing: {value}.") from error
    try:
        candidate.relative_to(base)
    except ValueError as error:
        raise NestedEvidenceError(f"{label} escapes packet root.") from error
    return candidate


def _resolve_directory(root: Path, value: str | Path, *, label: str) -> Path:
    candidate = _resolve_input(root, value, label=label)
    if not candidate.is_dir():
        raise NestedEvidenceError(f"{label} is not a directory.")
    return candidate


def _relative_to_packet(path: Path, packet_root: Path) -> str:
    return path.resolve(strict=True).relative_to(
        packet_root.resolve(strict=True)
    ).as_posix()


def _binding_shape(
    value: object,
    *,
    label: str,
    extra_keys: frozenset[str] = frozenset(),
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise NestedEvidenceError(f"{label} must be an object.")
    expected = {"bytes", "path", "sha256"} | set(extra_keys)
    if set(value) != expected:
        raise NestedEvidenceError(
            f"{label} fields differ: expected={sorted(expected)}, "
            f"observed={sorted(value)}."
        )
    size = value["bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise NestedEvidenceError(f"{label}.bytes is invalid.")
    _safe_relative(value["path"], label=f"{label}.path")
    if not _is_sha256(value["sha256"]):
        raise NestedEvidenceError(f"{label}.sha256 is invalid.")
    return value


def _hash_stream(handle: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        size += len(block)
        digest.update(block)
    return size, digest.hexdigest()


def _hash_path(path: Path) -> tuple[int, str]:
    with path.open("rb") as handle:
        return _hash_stream(handle)


def _run_hash_task(task: _HashTask) -> _HashResult:
    try:
        size, digest = _hash_path(task.path)
    except OSError as error:
        return _HashResult(
            task.category,
            task.identity,
            task.path,
            None,
            None,
            f"{type(error).__name__}: {error}",
        )
    return _HashResult(
        task.category, task.identity, task.path, size, digest, None
    )


def _hash_tasks(
    tasks: Sequence[_HashTask], *, workers: int
) -> list[_HashResult]:
    if not tasks:
        return []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_run_hash_task, tasks))


def _actual_binding(result: _HashResult, declared: Mapping[str, object]):
    if result.bytes is None or result.sha256 is None:
        return None
    return {
        "bytes": result.bytes,
        "path": declared["path"],
        "sha256": result.sha256,
    }


def _record_hash_result(
    issues: list[dict[str, object]],
    result: _HashResult,
    declared: Mapping[str, object],
) -> bool:
    if result.error is not None:
        _issue(
            issues,
            "FILE_READ_ERROR",
            result.identity,
            result.error,
        )
        return False
    matched = True
    if result.bytes != declared["bytes"]:
        _issue(
            issues,
            "FILE_SIZE_MISMATCH",
            result.identity,
            "Observed byte count differs from the declared binding.",
            expected=declared["bytes"],
            observed=result.bytes,
        )
        matched = False
    if result.sha256 != declared["sha256"]:
        _issue(
            issues,
            "FILE_SHA256_MISMATCH",
            result.identity,
            "Observed SHA-256 differs from the declared binding.",
            expected=declared["sha256"],
            observed=result.sha256,
        )
        matched = False
    return matched


def _verify_consumed(
    *,
    packet_root: Path,
    consumed_path: Path,
    frozen_root: Path,
    results_root: Path,
    expectations: NestedEvidenceExpectations,
    workers: int,
    issues: list[dict[str, object]],
) -> dict[str, object]:
    manifest_value, manifest_raw = _read_strict_json(
        consumed_path, label="consumed-artifact manifest"
    )
    manifest_size = len(manifest_raw)
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if not isinstance(manifest_value, dict):
        raise NestedEvidenceError("Consumed-artifact manifest root must be an object.")
    if manifest_raw != _pretty_manifest_bytes(manifest_value):
        _issue(
            issues,
            "CONSUMED_MANIFEST_ENCODING_DRIFT",
            _relative_to_packet(consumed_path, packet_root),
            "Manifest is not the canonical indented UTF-8 encoding.",
        )
    expected_manifest_keys = {
        "consumed_row_artifact_set_sha256",
        "metric_reference_sources",
        "row_artifact_count",
        "row_artifacts",
        "schema",
        "terminal_receipt_set_sha256",
        "top_level_inputs",
    }
    if set(manifest_value) != expected_manifest_keys:
        raise NestedEvidenceError(
            "Consumed-artifact manifest field set drifted."
        )
    if manifest_value["schema"] != CONSUMED_SCHEMA:
        raise NestedEvidenceError("Consumed-artifact manifest schema mismatch.")

    top = manifest_value["top_level_inputs"]
    references = manifest_value["metric_reference_sources"]
    rows = manifest_value["row_artifacts"]
    if not isinstance(top, dict):
        raise NestedEvidenceError("top_level_inputs must be an object.")
    if not isinstance(references, list):
        raise NestedEvidenceError("metric_reference_sources must be an array.")
    if not isinstance(rows, list):
        raise NestedEvidenceError("row_artifacts must be an array.")

    observed_counts = {
        "metric_reference_sources": len(references),
        "row_entries": len(rows),
        "top_level_inputs": len(top),
    }
    for name, expected in (
        ("top_level_inputs", expectations.top_level_inputs),
        ("metric_reference_sources", expectations.metric_reference_sources),
        ("row_entries", expectations.row_entries),
    ):
        if observed_counts[name] != expected:
            _issue(
                issues,
                "CARDINALITY_MISMATCH",
                f"consumed_artifacts.{name}",
                "Observed inventory cardinality differs from the frozen expectation.",
                expected=expected,
                observed=observed_counts[name],
            )
    if set(top) != set(TOP_LEVEL_INPUTS):
        _issue(
            issues,
            "TOP_LEVEL_INPUT_SET_MISMATCH",
            "consumed_artifacts.top_level_inputs",
            "Top-level input role set drifted.",
            expected=sorted(TOP_LEVEL_INPUTS),
            observed=sorted(str(key) for key in top),
        )

    tasks: list[_HashTask] = []
    task_declared: dict[str, Mapping[str, object]] = {}
    top_identities: dict[str, str] = {}
    seen_paths: set[tuple[str, str]] = set()
    for name in sorted(top):
        label = f"top_level_inputs.{name}"
        try:
            binding = _binding_shape(top[name], label=label)
            binding_root = (
                frozen_root if name in TOP_LEVEL_FROZEN else results_root
            )
            if name not in TOP_LEVEL_INPUTS:
                continue
            relative = str(binding["path"])
            duplicate_key = (str(binding_root), relative)
            if duplicate_key in seen_paths:
                raise NestedEvidenceError(f"{label}.path is duplicated.")
            seen_paths.add(duplicate_key)
            path = _resolve_inside(binding_root, relative, label=label)
        except NestedEvidenceError as error:
            _issue(issues, "INVALID_BINDING", label, str(error))
            continue
        identity = f"consumed:{label}"
        top_identities[name] = identity
        task_declared[identity] = binding
        tasks.append(_HashTask("top_level", identity, path, binding))

    reference_identities: list[str | None] = []
    seen_cases: set[str] = set()
    for index, value in enumerate(references):
        label = f"metric_reference_sources[{index}]"
        identity: str | None = None
        try:
            binding = _binding_shape(
                value, label=label, extra_keys=frozenset({"case_id"})
            )
            case_id = binding["case_id"]
            if not isinstance(case_id, str) or not case_id or case_id in seen_cases:
                raise NestedEvidenceError(f"{label}.case_id is invalid or duplicated.")
            seen_cases.add(case_id)
            relative = str(binding["path"])
            duplicate_key = (str(frozen_root), relative)
            if duplicate_key in seen_paths:
                raise NestedEvidenceError(f"{label}.path is duplicated.")
            seen_paths.add(duplicate_key)
            path = _resolve_inside(frozen_root, relative, label=label)
            identity = f"consumed:{label}"
            task_declared[identity] = binding
            tasks.append(_HashTask("metric_reference", identity, path, binding))
        except NestedEvidenceError as error:
            _issue(issues, "INVALID_BINDING", label, str(error))
        reference_identities.append(identity)

    row_identities: list[dict[str, str]] = []
    seen_run_keys: set[str] = set()
    ordered_run_keys: list[str] = []
    artifact_inventory_count = 0
    for row_index, value in enumerate(rows):
        identities: dict[str, str] = {}
        row_identities.append(identities)
        label = f"row_artifacts[{row_index}]"
        if not isinstance(value, dict) or set(value) != {
            "artifacts",
            "run_key",
            "run_key_sha256",
        }:
            _issue(
                issues,
                "INVALID_ROW",
                label,
                "Row field set is invalid.",
            )
            continue
        run_key = value["run_key"]
        run_sha = value["run_key_sha256"]
        if not isinstance(run_key, dict) or set(run_key) != {
            "algorithm",
            "budget",
            "case_id",
            "seed",
        }:
            _issue(issues, "INVALID_RUN_KEY", label, "Run-key shape is invalid.")
            continue
        valid_values = (
            isinstance(run_key["algorithm"], str)
            and bool(run_key["algorithm"])
            and isinstance(run_key["case_id"], str)
            and bool(run_key["case_id"])
            and isinstance(run_key["budget"], int)
            and not isinstance(run_key["budget"], bool)
            and run_key["budget"] > 0
            and isinstance(run_key["seed"], int)
            and not isinstance(run_key["seed"], bool)
        )
        if not valid_values or run_sha != _canonical_digest(run_key):
            _issue(
                issues,
                "INVALID_RUN_KEY",
                label,
                "Run-key values or canonical digest are invalid.",
            )
            continue
        if run_sha in seen_run_keys:
            _issue(issues, "DUPLICATE_RUN_KEY", label, "Run key is duplicated.")
            continue
        seen_run_keys.add(str(run_sha))
        ordered_run_keys.append(str(run_sha))
        artifacts = value["artifacts"]
        if not isinstance(artifacts, dict) or set(artifacts) != set(ROW_ARTIFACTS):
            _issue(
                issues,
                "ROW_ARTIFACT_SET_MISMATCH",
                label,
                "Row artifact role set drifted.",
            )
            continue
        artifact_inventory_count += len(artifacts)
        for name in sorted(artifacts):
            artifact_label = f"{label}.artifacts.{name}"
            try:
                binding = _binding_shape(artifacts[name], label=artifact_label)
                relative = str(binding["path"])
                duplicate_key = (str(results_root), relative)
                if duplicate_key in seen_paths:
                    raise NestedEvidenceError(
                        f"{artifact_label}.path is duplicated."
                    )
                seen_paths.add(duplicate_key)
                path = _resolve_inside(
                    results_root, relative, label=artifact_label
                )
            except NestedEvidenceError as error:
                _issue(issues, "INVALID_BINDING", artifact_label, str(error))
                continue
            identity = f"consumed:{artifact_label}"
            identities[name] = identity
            task_declared[identity] = binding
            tasks.append(_HashTask("row_artifact", identity, path, binding))

    if ordered_run_keys != sorted(ordered_run_keys):
        _issue(
            issues,
            "ROW_ORDER_MISMATCH",
            "consumed_artifacts.row_artifacts",
            "Rows are not in canonical run-key digest order.",
        )
    observed_counts["row_artifacts"] = artifact_inventory_count
    declared_artifact_count = manifest_value["row_artifact_count"]
    if (
        isinstance(declared_artifact_count, bool)
        or not isinstance(declared_artifact_count, int)
        or declared_artifact_count != artifact_inventory_count
    ):
        _issue(
            issues,
            "DECLARED_ROW_ARTIFACT_COUNT_MISMATCH",
            "consumed_artifacts.row_artifact_count",
            "Declared row-artifact count does not match the inventory.",
            expected=artifact_inventory_count,
            observed=declared_artifact_count,
        )
    if artifact_inventory_count != expectations.row_artifacts:
        _issue(
            issues,
            "CARDINALITY_MISMATCH",
            "consumed_artifacts.row_artifacts",
            "Observed inventory cardinality differs from the frozen expectation.",
            expected=expectations.row_artifacts,
            observed=artifact_inventory_count,
        )

    declared_row_digest = manifest_value["consumed_row_artifact_set_sha256"]
    declared_terminal_digest = manifest_value["terminal_receipt_set_sha256"]
    recomputed_manifest_row_digest = _canonical_digest(rows)
    terminal_bindings = [
        row["artifacts"]["terminal_receipt"]
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("artifacts"), dict)
        and "terminal_receipt" in row["artifacts"]
    ]
    recomputed_manifest_terminal_digest = _canonical_digest(terminal_bindings)
    for field, declared, observed in (
        (
            "consumed_row_artifact_set_sha256",
            declared_row_digest,
            recomputed_manifest_row_digest,
        ),
        (
            "terminal_receipt_set_sha256",
            declared_terminal_digest,
            recomputed_manifest_terminal_digest,
        ),
    ):
        if not _is_sha256(declared) or declared != observed:
            _issue(
                issues,
                "INTERNAL_SET_DIGEST_MISMATCH",
                f"consumed_artifacts.{field}",
                "Stored set digest differs from the canonical manifest inventory.",
                expected=declared,
                observed=observed,
            )

    hash_results = _hash_tasks(tasks, workers=workers)
    results_by_identity = {result.identity: result for result in hash_results}
    matched_by_identity: dict[str, bool] = {}
    for result in hash_results:
        declared = task_declared[result.identity]
        matched_by_identity[result.identity] = _record_hash_result(
            issues, result, declared
        )

    verified_top = sum(
        1
        for identity in top_identities.values()
        if matched_by_identity.get(identity, False)
    )
    verified_references = sum(
        1
        for identity in reference_identities
        if identity is not None and matched_by_identity.get(identity, False)
    )
    verified_row_artifacts = sum(
        1
        for identities in row_identities
        for identity in identities.values()
        if matched_by_identity.get(identity, False)
    )
    verified_rows = sum(
        1
        for identities in row_identities
        if set(identities) == set(ROW_ARTIFACTS)
        and all(matched_by_identity.get(identity, False) for identity in identities.values())
    )

    verified_bytes = {
        "metric_reference_sources": sum(
            int(results_by_identity[identity].bytes or 0)
            for identity in reference_identities
            if identity is not None and matched_by_identity.get(identity, False)
        ),
        "row_artifacts": sum(
            int(results_by_identity[identity].bytes or 0)
            for identities in row_identities
            for identity in identities.values()
            if matched_by_identity.get(identity, False)
        ),
        "top_level_inputs": sum(
            int(results_by_identity[identity].bytes or 0)
            for identity in top_identities.values()
            if matched_by_identity.get(identity, False)
        ),
    }
    verified_bytes["total"] = sum(verified_bytes.values())

    actual_top: dict[str, object] = {}
    for name, identity in top_identities.items():
        actual = _actual_binding(results_by_identity[identity], task_declared[identity])
        if actual is not None:
            actual_top[name] = actual
    actual_references: list[dict[str, object]] = []
    for index, identity in enumerate(reference_identities):
        if identity is None:
            continue
        actual = _actual_binding(results_by_identity[identity], task_declared[identity])
        if actual is not None:
            actual["case_id"] = references[index]["case_id"]
            actual_references.append(actual)
    actual_rows: list[dict[str, object]] = []
    actual_terminals: list[dict[str, object]] = []
    actual_rows_complete = True
    for index, identities in enumerate(row_identities):
        if set(identities) != set(ROW_ARTIFACTS):
            actual_rows_complete = False
            continue
        actual_artifacts: dict[str, object] = {}
        for name, identity in identities.items():
            actual = _actual_binding(
                results_by_identity[identity], task_declared[identity]
            )
            if actual is None:
                actual_rows_complete = False
                break
            actual_artifacts[name] = actual
        else:
            row = rows[index]
            actual_row = {
                "artifacts": actual_artifacts,
                "run_key": row["run_key"],
                "run_key_sha256": row["run_key_sha256"],
            }
            actual_rows.append(actual_row)
            actual_terminals.append(actual_artifacts["terminal_receipt"])

    actual_row_digest = (
        _canonical_digest(actual_rows)
        if actual_rows_complete and len(actual_rows) == len(rows)
        else None
    )
    actual_terminal_digest = (
        _canonical_digest(actual_terminals)
        if actual_rows_complete and len(actual_terminals) == len(rows)
        else None
    )
    set_digests = {
        "consumed_row_artifact_set": {
            "declared": declared_row_digest,
            "recomputed_from_files": actual_row_digest,
            "recomputed_from_manifest": recomputed_manifest_row_digest,
        },
        "metric_reference_source_set": {
            "declared_inventory": _canonical_digest(references),
            "recomputed_from_files": (
                _canonical_digest(actual_references)
                if len(actual_references) == len(references)
                else None
            ),
        },
        "terminal_receipt_set": {
            "declared": declared_terminal_digest,
            "recomputed_from_files": actual_terminal_digest,
            "recomputed_from_manifest": recomputed_manifest_terminal_digest,
        },
        "top_level_input_set": {
            "declared_inventory": _canonical_digest(top),
            "recomputed_from_files": (
                _canonical_digest(actual_top)
                if len(actual_top) == len(top)
                else None
            ),
        },
    }
    for name, entry in set_digests.items():
        values = [value for value in entry.values() if value is not None]
        if values and any(value != values[0] for value in values[1:]):
            _issue(
                issues,
                "FILE_SET_DIGEST_MISMATCH",
                f"consumed_artifacts.set_digests.{name}",
                "File-derived set digest differs from the declared inventory.",
            )

    return {
        "manifest": {
            "bytes": manifest_size,
            "path": _relative_to_packet(consumed_path, packet_root),
            "sha256": manifest_sha,
        },
        "inventory_counts": observed_counts,
        "verified_counts": {
            "metric_reference_sources": verified_references,
            "row_artifacts": verified_row_artifacts,
            "row_entries": verified_rows,
            "top_level_inputs": verified_top,
        },
        "verified_bytes": verified_bytes,
        "set_digests": set_digests,
    }


def _verify_source_archive(
    *,
    packet_root: Path,
    source_manifest_path: Path,
    source_archive_path: Path,
    expectations: NestedEvidenceExpectations,
    issues: list[dict[str, object]],
) -> dict[str, object]:
    manifest_value, manifest_raw = _read_strict_json(
        source_manifest_path, label="external source-file manifest"
    )
    manifest_binding = {
        "bytes": len(manifest_raw),
        "path": _relative_to_packet(source_manifest_path, packet_root),
        "sha256": hashlib.sha256(manifest_raw).hexdigest(),
    }
    if manifest_raw != canonical_json_bytes(manifest_value, trailing_newline=True):
        _issue(
            issues,
            "SOURCE_MANIFEST_ENCODING_DRIFT",
            manifest_binding["path"],
            "External source manifest is not canonical compact UTF-8 JSON.",
        )
    if not isinstance(manifest_value, dict) or set(manifest_value) != {
        "archive_prefix",
        "file_count",
        "files",
        "schema",
    }:
        raise NestedEvidenceError("External source manifest field set drifted.")
    if manifest_value["schema"] != SOURCE_MANIFEST_SCHEMA:
        raise NestedEvidenceError("External source manifest schema mismatch.")
    prefix = _safe_relative(
        manifest_value["archive_prefix"], label="source manifest archive_prefix"
    )
    if len(PurePosixPath(prefix).parts) != 1:
        raise NestedEvidenceError("Source archive prefix must be one path segment.")
    files = manifest_value["files"]
    if not isinstance(files, list):
        raise NestedEvidenceError("Source manifest files must be an array.")
    declared_count = manifest_value["file_count"]
    if (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count != len(files)
    ):
        _issue(
            issues,
            "DECLARED_SOURCE_FILE_COUNT_MISMATCH",
            "source_archive.file_count",
            "Declared source-file count differs from the manifest inventory.",
            expected=len(files),
            observed=declared_count,
        )
    if len(files) != expectations.source_files:
        _issue(
            issues,
            "CARDINALITY_MISMATCH",
            "source_archive.source_files",
            "Source-file inventory differs from the frozen expectation.",
            expected=expectations.source_files,
            observed=len(files),
        )

    declared_by_member: dict[str, Mapping[str, object]] = {}
    ordered_paths: list[str] = []
    for index, value in enumerate(files):
        label = f"source_manifest.files[{index}]"
        try:
            binding = _binding_shape(value, label=label)
        except NestedEvidenceError as error:
            _issue(issues, "INVALID_SOURCE_BINDING", label, str(error))
            continue
        relative = str(binding["path"])
        member_name = f"{prefix}/{relative}"
        if member_name in declared_by_member:
            _issue(
                issues,
                "DUPLICATE_SOURCE_PATH",
                label,
                "Source path is duplicated.",
            )
            continue
        declared_by_member[member_name] = binding
        ordered_paths.append(relative)
    if ordered_paths != sorted(ordered_paths):
        _issue(
            issues,
            "SOURCE_ORDER_MISMATCH",
            "source_manifest.files",
            "Source-file entries are not in canonical path order.",
        )

    archive_size, archive_sha = _hash_path(source_archive_path)
    archive_binding = {
        "bytes": archive_size,
        "path": _relative_to_packet(source_archive_path, packet_root),
        "sha256": archive_sha,
    }
    internal_name = f"{prefix}/source_file_manifest.json"
    internal_binding: dict[str, object] | None = None
    actual_files: list[dict[str, object]] = []
    verified_files = 0
    verified_source_bytes = 0
    archive_member_count = 0
    try:
        with tarfile.open(source_archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            archive_member_count = len(members)
            member_by_name: dict[str, tarfile.TarInfo] = {}
            for member in members:
                try:
                    safe_name = _safe_relative(
                        member.name, label="source archive member"
                    )
                except NestedEvidenceError as error:
                    _issue(
                        issues,
                        "UNSAFE_ARCHIVE_MEMBER",
                        str(member.name),
                        str(error),
                    )
                    continue
                if not member.isfile():
                    _issue(
                        issues,
                        "NONREGULAR_ARCHIVE_MEMBER",
                        safe_name,
                        "Every archive member must be a regular file.",
                    )
                    continue
                if safe_name in member_by_name:
                    _issue(
                        issues,
                        "DUPLICATE_ARCHIVE_MEMBER",
                        safe_name,
                        "Archive member name is duplicated.",
                    )
                    continue
                member_by_name[safe_name] = member

            expected_members = set(declared_by_member) | {internal_name}
            observed_members = set(member_by_name)
            if expected_members != observed_members:
                _issue(
                    issues,
                    "ARCHIVE_MEMBER_SET_MISMATCH",
                    archive_binding["path"],
                    "Archive members differ from the external manifest.",
                    expected=sorted(expected_members),
                    observed=sorted(observed_members),
                )

            internal_member = member_by_name.get(internal_name)
            if internal_member is not None:
                handle = archive.extractfile(internal_member)
                if handle is None:
                    _issue(
                        issues,
                        "ARCHIVE_MEMBER_READ_ERROR",
                        internal_name,
                        "Cannot read the internal source manifest.",
                    )
                else:
                    with handle:
                        internal_raw = handle.read()
                    internal_binding = {
                        "bytes": len(internal_raw),
                        "path": internal_name,
                        "sha256": hashlib.sha256(internal_raw).hexdigest(),
                    }
                    if internal_raw != manifest_raw:
                        _issue(
                            issues,
                            "INTERNAL_EXTERNAL_MANIFEST_MISMATCH",
                            internal_name,
                            "Internal source manifest bytes differ from the external manifest.",
                            expected=manifest_binding["sha256"],
                            observed=internal_binding["sha256"],
                        )

            for member_name in sorted(declared_by_member):
                declared = declared_by_member[member_name]
                member = member_by_name.get(member_name)
                if member is None:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    _issue(
                        issues,
                        "ARCHIVE_MEMBER_READ_ERROR",
                        member_name,
                        "Cannot read source archive member.",
                    )
                    continue
                with handle:
                    size, digest = _hash_stream(handle)
                actual = {
                    "bytes": size,
                    "path": declared["path"],
                    "sha256": digest,
                }
                actual_files.append(actual)
                matched = True
                if size != declared["bytes"]:
                    _issue(
                        issues,
                        "SOURCE_FILE_SIZE_MISMATCH",
                        member_name,
                        "Archived source byte count differs from the manifest.",
                        expected=declared["bytes"],
                        observed=size,
                    )
                    matched = False
                if digest != declared["sha256"]:
                    _issue(
                        issues,
                        "SOURCE_FILE_SHA256_MISMATCH",
                        member_name,
                        "Archived source SHA-256 differs from the manifest.",
                        expected=declared["sha256"],
                        observed=digest,
                    )
                    matched = False
                if matched:
                    verified_files += 1
                    verified_source_bytes += size
    except (tarfile.TarError, OSError) as error:
        _issue(
            issues,
            "SOURCE_ARCHIVE_READ_ERROR",
            archive_binding["path"],
            f"{type(error).__name__}: {error}",
        )

    declared_source_digest = _canonical_digest(files)
    actual_source_digest = (
        _canonical_digest(actual_files)
        if len(actual_files) == len(files)
        else None
    )
    if (
        actual_source_digest is not None
        and actual_source_digest != declared_source_digest
    ):
        _issue(
            issues,
            "SOURCE_FILE_SET_DIGEST_MISMATCH",
            "source_archive.set_digests.source_file_set",
            "Archive-derived source-file set digest differs from the external manifest.",
        )

    return {
        "archive": archive_binding,
        "archive_member_count": archive_member_count,
        "external_manifest": manifest_binding,
        "internal_manifest": internal_binding,
        "inventory_counts": {"source_files": len(files)},
        "set_digests": {
            "source_file_set": {
                "declared_inventory": declared_source_digest,
                "recomputed_from_archive": actual_source_digest,
            }
        },
        "verified_bytes": {
            "source_files": verified_source_bytes,
            "total_with_archive_and_manifests": (
                verified_source_bytes
                + archive_size
                + len(manifest_raw)
                + (int(internal_binding["bytes"]) if internal_binding else 0)
            ),
        },
        "verified_counts": {
            "internal_external_manifest_matches": int(
                internal_binding is not None
                and internal_binding["sha256"] == manifest_binding["sha256"]
                and internal_binding["bytes"] == manifest_binding["bytes"]
            ),
            "source_files": verified_files,
        },
    }


def _output_path(
    packet_root: Path,
    output: str | Path,
    *,
    protected_roots: Sequence[Path],
    protected_files: Sequence[Path],
) -> Path:
    base = packet_root.resolve(strict=True)
    candidate = Path(output)
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        parent = candidate.parent.resolve(strict=True)
    except OSError as error:
        raise NestedEvidenceError(
            f"Receipt parent directory is missing: {candidate.parent}."
        ) from error
    candidate = parent / candidate.name
    try:
        candidate.relative_to(base)
    except ValueError as error:
        raise NestedEvidenceError("Receipt output escapes packet root.") from error
    for protected in protected_roots:
        try:
            candidate.relative_to(protected.resolve(strict=True))
        except ValueError:
            continue
        raise NestedEvidenceError(
            "Receipt output may not be written inside a protected evidence root."
        )
    if any(candidate == path.resolve(strict=True) for path in protected_files):
        raise NestedEvidenceError("Receipt output aliases an input evidence file.")
    if candidate.exists():
        raise NestedEvidenceError(
            f"Refusing to replace an existing nested-evidence receipt: {candidate}."
        )
    return candidate


def verify_nested_evidence(
    *,
    packet_root: str | Path,
    consumed_manifest: str | Path,
    frozen_root: str | Path,
    results_root: str | Path,
    source_manifest: str | Path,
    source_archive: str | Path,
    receipt_output: str | Path,
    expectations: NestedEvidenceExpectations | None = None,
    workers: int = 4,
) -> NestedEvidenceVerification:
    """Rehash all declared nested evidence and write one immutable receipt.

    Every input is opened read-only.  A receipt is emitted even when evidence
    mismatches are found; its status is then ``FAIL`` and callers must reject it.
    Unsafe paths, unreadable top-level inputs, or output collisions raise before
    any receipt bytes are written.
    """

    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise NestedEvidenceError("workers must be a positive integer.")
    expected = expectations or NestedEvidenceExpectations()
    started = time.perf_counter()
    root = Path(packet_root).resolve(strict=True)
    if not root.is_dir():
        raise NestedEvidenceError("packet_root must be a directory.")
    consumed_path = _resolve_input(
        root, consumed_manifest, label="consumed manifest"
    )
    frozen = _resolve_directory(root, frozen_root, label="frozen root")
    results = _resolve_directory(root, results_root, label="results root")
    source_manifest_path = _resolve_input(
        root, source_manifest, label="source manifest"
    )
    source_archive_path = _resolve_input(
        root, source_archive, label="source archive"
    )
    for label, path in (
        ("consumed manifest", consumed_path),
        ("source manifest", source_manifest_path),
        ("source archive", source_archive_path),
    ):
        if not path.is_file():
            raise NestedEvidenceError(f"{label} must be a regular file.")
    receipt_path = _output_path(
        root,
        receipt_output,
        protected_roots=(frozen, results),
        protected_files=(
            consumed_path,
            source_manifest_path,
            source_archive_path,
        ),
    )

    issues: list[dict[str, object]] = []
    consumed = _verify_consumed(
        packet_root=root,
        consumed_path=consumed_path,
        frozen_root=frozen,
        results_root=results,
        expectations=expected,
        workers=workers,
        issues=issues,
    )
    source = _verify_source_archive(
        packet_root=root,
        source_manifest_path=source_manifest_path,
        source_archive_path=source_archive_path,
        expectations=expected,
        issues=issues,
    )
    elapsed = round(time.perf_counter() - started, 6)
    status = "PASS" if not issues else "FAIL"
    overall_set_digest = None
    row_digest = consumed["set_digests"]["consumed_row_artifact_set"][
        "recomputed_from_files"
    ]
    source_digest = source["set_digests"]["source_file_set"][
        "recomputed_from_archive"
    ]
    if row_digest is not None and source_digest is not None:
        overall_set_digest = _canonical_digest(
            {
                "consumed_manifest_sha256": consumed["manifest"]["sha256"],
                "consumed_row_artifact_set_sha256": row_digest,
                "metric_reference_source_set_sha256": consumed["set_digests"][
                    "metric_reference_source_set"
                ]["recomputed_from_files"],
                "source_archive_sha256": source["archive"]["sha256"],
                "source_file_set_sha256": source_digest,
                "terminal_receipt_set_sha256": consumed["set_digests"][
                    "terminal_receipt_set"
                ]["recomputed_from_files"],
                "top_level_input_set_sha256": consumed["set_digests"][
                    "top_level_input_set"
                ]["recomputed_from_files"],
            }
        )
    receipt = {
        "consumed_artifacts": consumed,
        "elapsed_seconds": elapsed,
        "expectations": {
            "metric_reference_sources": expected.metric_reference_sources,
            "row_artifacts": expected.row_artifacts,
            "row_entries": expected.row_entries,
            "source_files": expected.source_files,
            "top_level_inputs": expected.top_level_inputs,
        },
        "issues": issues,
        "nested_evidence_set_sha256": overall_set_digest,
        "schema": RECEIPT_SCHEMA,
        "source_archive": source,
        "status": status,
    }
    raw = canonical_json_bytes(receipt, trailing_newline=True)
    try:
        with receipt_path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise NestedEvidenceError(
            f"Refusing to replace an existing nested-evidence receipt: {receipt_path}."
        ) from error
    return NestedEvidenceVerification(
        receipt_path=receipt_path,
        receipt_sha256=hashlib.sha256(raw).hexdigest(),
        status=status,
        issue_count=len(issues),
        elapsed_seconds=elapsed,
    )
