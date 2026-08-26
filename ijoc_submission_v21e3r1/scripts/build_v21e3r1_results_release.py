from __future__ import annotations

"""Build a deterministic, data-only V21e3r1 development-results release.

The executable code archive is deliberately not nested in this archive.  It is
bound by bytes and SHA-256 in the external release index, while its manifest,
checksum, and clean-room receipt are included as independently inspectable
supporting evidence.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
from typing import Mapping, Sequence
import zipfile


ROW_FILES = frozenset(
    {
        "trace.sqlite3",
        "terminal.receipt.json",
        "row.preverification.json",
        "objective_archive_replay.receipt.json",
        "metric_replay.receipt.json",
        "row.json",
    }
)
ROW_PATH_SEMANTICS = "row_directory_relative_posix_v1"
MATRIX_PATH_SEMANTICS = "matrix_directory_relative_posix_v1"
MATRIX_SCIENTIFIC_SCOPE = "authors_generated_development_only_not_formal_evidence"
MATRIX_SEEDS = (31051, 31057, 31059)
MATRIX_ARMS = ("V21E3_C0", "NSGAII", "MOEAD")
TARGET_PATH_SEMANTICS = "repo_root_relative_posix_v1"
TARGET_RECEIPT_NAME = "V21E3R1_TARGET_SIZE_EXECUTION_RECEIPT_V1.json"
TARGET_SCIENTIFIC_SCOPE = (
    "target_size_small_budget_structure_and_objective_archive_replay_"
    "not_performance_evidence"
)
TARGET_ARMS = ("V21E3_C0", "NSGAII", "MOEAD")
TARGET_FAMILIES = ("MOTSP", "MOKP")
DEFAULT_PREFIX = "ijoc_v21e3r1_results_release"
CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class ReleaseEntry:
    source_path: Path
    archive_path: str
    role: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class CapturedFile:
    live_path: Path
    staged_path: Path
    bytes: int
    sha256: str
    live_mtime_ns: int
    live_device: int
    live_inode: int


@dataclass(frozen=True)
class CapturedTree:
    live_root: Path
    staged_root: Path
    relative_files: tuple[str, ...]
    relative_directories: tuple[str, ...]


@dataclass(frozen=True)
class CapturedPacket:
    root: Path
    paths: Mapping[str, Path]
    files: tuple[CapturedFile, ...]
    trees: tuple[CapturedTree, ...]


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _stream_binding(path: Path, *, chunk_size: int = CHUNK_SIZE) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def _capture_entry(path: str | Path, *, archive_path: str, role: str) -> ReleaseEntry:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"Release input is not a regular non-symlink file: {source}")
    before = source.stat()
    observed_bytes, observed_sha256 = _stream_binding(source)
    after = source.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or observed_bytes != after.st_size
    ):
        raise RuntimeError(f"Release input changed during preflight hashing: {source.name}")
    _require_archive_path(archive_path, field="archive_path")
    return ReleaseEntry(
        source_path=source,
        archive_path=archive_path,
        role=str(role),
        bytes=observed_bytes,
        sha256=observed_sha256,
    )


def _is_link(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(is_junction is not None and is_junction())


def _capture_regular_file(live_path: Path, staged_path: Path) -> CapturedFile:
    if _is_link(live_path) or not live_path.is_file():
        raise RuntimeError(f"Packet input is not a regular non-link file: {live_path.name}")
    before = live_path.stat()
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    with live_path.open("rb") as source, staged_path.open("xb") as destination:
        while True:
            chunk = source.read(CHUNK_SIZE)
            if not chunk:
                break
            destination.write(chunk)
            digest.update(chunk)
            total += len(chunk)
        destination.flush()
        os.fsync(destination.fileno())
    after = live_path.stat()
    identity_before = (before.st_dev, before.st_ino)
    identity_after = (after.st_dev, after.st_ino)
    if (
        _is_link(live_path)
        or not live_path.is_file()
        or identity_before != identity_after
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or total != before.st_size
        or staged_path.stat().st_size != total
    ):
        raise RuntimeError(f"TOCTOU: packet input changed during staging: {live_path.name}")
    return CapturedFile(
        live_path=live_path,
        staged_path=staged_path,
        bytes=total,
        sha256=digest.hexdigest(),
        live_mtime_ns=before.st_mtime_ns,
        live_device=before.st_dev,
        live_inode=before.st_ino,
    )


def _scan_tree_inventory(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if _is_link(root) or not root.is_dir():
        raise RuntimeError(f"Packet tree is not a regular directory: {root.name}")
    files: list[str] = []
    directories: list[str] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if _is_link(child):
                raise RuntimeError(f"Packet tree contains a link: {child.name}")
            relative = child.relative_to(root).as_posix()
            if child.is_dir():
                directories.append(relative)
                pending.append(child)
            elif child.is_file():
                files.append(relative)
            else:
                raise RuntimeError(f"Packet tree contains a special file: {child.name}")
    return tuple(sorted(files)), tuple(sorted(directories))


def _capture_tree(
    live_root: Path, staged_root: Path
) -> tuple[CapturedTree, list[CapturedFile]]:
    relative_files, relative_directories = _scan_tree_inventory(live_root)
    staged_root.mkdir(parents=True, exist_ok=False)
    for relative in relative_directories:
        (staged_root / Path(*PurePosixPath(relative).parts)).mkdir()
    captured: list[CapturedFile] = []
    for relative in relative_files:
        parts = PurePosixPath(relative).parts
        captured.append(
            _capture_regular_file(live_root.joinpath(*parts), staged_root.joinpath(*parts))
        )
    if _scan_tree_inventory(live_root) != (
        relative_files,
        relative_directories,
    ):
        raise RuntimeError(f"TOCTOU: packet tree changed during staging: {live_root.name}")
    return (
        CapturedTree(
            live_root=live_root,
            staged_root=staged_root,
            relative_files=relative_files,
            relative_directories=relative_directories,
        ),
        captured,
    )


def _verify_captured_packet(packet: CapturedPacket) -> None:
    for captured in packet.files:
        if (
            _is_link(captured.live_path)
            or not captured.live_path.is_file()
            or _is_link(captured.staged_path)
            or not captured.staged_path.is_file()
        ):
            raise RuntimeError("TOCTOU: captured packet file type changed.")
        live_stat = captured.live_path.stat()
        if (
            live_stat.st_dev != captured.live_device
            or live_stat.st_ino != captured.live_inode
            or live_stat.st_mtime_ns != captured.live_mtime_ns
            or live_stat.st_size != captured.bytes
        ):
            raise RuntimeError(
                f"TOCTOU: live packet identity changed: {captured.live_path.name}"
            )
        live_bytes, live_sha = _stream_binding(captured.live_path)
        staged_bytes, staged_sha = _stream_binding(captured.staged_path)
        if (
            live_bytes != captured.bytes
            or live_sha != captured.sha256
            or staged_bytes != captured.bytes
            or staged_sha != captured.sha256
        ):
            raise RuntimeError(
                f"TOCTOU: captured packet bytes changed: {captured.live_path.name}"
            )
    for tree in packet.trees:
        expected = (tree.relative_files, tree.relative_directories)
        if _scan_tree_inventory(tree.live_root) != expected:
            raise RuntimeError(f"TOCTOU: live packet tree changed: {tree.live_root.name}")
        if _scan_tree_inventory(tree.staged_root) != expected:
            raise RuntimeError(f"TOCTOU: staged packet tree changed: {tree.live_root.name}")


def _stage_release_packet(
    staging_root: Path, input_paths: Mapping[str, str | Path]
) -> CapturedPacket:
    tree_keys = {"matrix_directory", "target_execution_directory"}
    paths: dict[str, Path] = {}
    files: list[CapturedFile] = []
    trees: list[CapturedTree] = []
    for key in sorted(input_paths):
        supplied = Path(input_paths[key])
        if _is_link(supplied):
            raise RuntimeError(f"Packet input root must not be a link: {key}")
        live = supplied.resolve(strict=True)
        destination_parent = staging_root / key
        destination_parent.mkdir()
        staged = destination_parent / live.name
        if key in tree_keys:
            tree, captured = _capture_tree(live, staged)
            trees.append(tree)
            files.extend(captured)
        else:
            files.append(_capture_regular_file(live, staged))
        paths[key] = staged
    packet = CapturedPacket(
        root=staging_root,
        paths=paths,
        files=tuple(files),
        trees=tuple(trees),
    )
    _verify_captured_packet(packet)
    return packet


def _load_json(path: str | Path, *, field: str) -> tuple[Path, dict[str, object]]:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is not UTF-8 JSON.") from error
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object.")
    return resolved, value


def _sha256(path: Path) -> str:
    return _stream_binding(path)[1]


def _require_hex64(value: object, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be lowercase SHA-256.")
    return text


def _require_archive_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty relative POSIX path.")
    candidate = PurePosixPath(value)
    if (
        value != candidate.as_posix()
        or "\\" in value
        or candidate.is_absolute()
        or PureWindowsPath(value).is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError(f"{field} must be a portable relative POSIX path.")
    return value


def _require_row_path(
    payload: Mapping[str, object], *, field: str, expected: str
) -> None:
    try:
        observed = _require_archive_path(payload.get(field), field=field)
    except ValueError as error:
        raise RuntimeError(f"Nonportable artifact path in {field}.") from error
    if observed != expected:
        raise RuntimeError(f"Nonportable artifact path in {field}.")


def _require_prohibitions(
    payload: Mapping[str, object],
    *,
    field: str,
    require_full_replay_boundary: bool,
) -> None:
    for name in ("selection_entropy_release", "calibration_execution", "formal_execution"):
        if payload.get(name) != "PROHIBITED":
            raise RuntimeError(f"{field} does not keep {name} PROHIBITED.")
    if payload.get("formal_authorized") is not False:
        raise RuntimeError(f"{field} must set formal_authorized=false.")
    if (
        require_full_replay_boundary
        and payload.get("full_algorithm_decision_replay") != "NOT_IMPLEMENTED"
    ):
        raise RuntimeError(f"{field} overstates full algorithm decision replay.")


def _identity(payload: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(payload.get(field) for field in ("case_id", "family", "size", "seed", "arm_id"))


def _validate_external_evidence(
    *,
    same_implementation_receipt_path: Path,
    v3_invalidation_path: Path,
    v4_snapshot_path: Path,
    v4_authorization_path: Path,
) -> dict[str, object]:
    _, snapshot = _load_json(v4_snapshot_path, field="v4_source_snapshot")
    if (
        snapshot.get("schema")
        != "pareto_v21e3r1_development_source_snapshot_freeze_v1"
        or snapshot.get("status") != "PASS_ENGINEERING_SNAPSHOT_ONLY"
        or snapshot.get("formal_authorized") is not False
        or snapshot.get("submission_status") != "IJOC_HOLD"
    ):
        raise RuntimeError("V4 source snapshot is not a held engineering snapshot.")
    source_root = _require_hex64(
        snapshot.get("bound_files_root_sha256"), field="v4 source root"
    )
    snapshot_sha = _sha256(v4_snapshot_path)

    _, authorization = _load_json(v4_authorization_path, field="v4_authorization")
    if (
        authorization.get("schema")
        != "pareto_v21e3r1_development_parity_authorization_v1"
        or authorization.get("status") != "AUTHORIZED_DEVELOPMENT_PARITY_ONLY"
        or authorization.get("source_snapshot_receipt_sha256") != snapshot_sha
        or authorization.get("source_snapshot_root_sha256") != source_root
        or authorization.get("submission_status") != "IJOC_HOLD"
    ):
        raise RuntimeError("V4 authorization does not bind the supplied V4 snapshot.")
    _require_prohibitions(
        authorization,
        field="V4 authorization",
        require_full_replay_boundary=False,
    )
    authorization_sha = _sha256(v4_authorization_path)

    _, same_implementation = _load_json(
        same_implementation_receipt_path,
        field="same_implementation_post_process_receipt",
    )
    owned_files = same_implementation.get("live_verifier_owned_files")
    owned_paths = (
        [str(entry.get("path", "")) for entry in owned_files]
        if isinstance(owned_files, list)
        else []
    )
    owned_files_valid = bool(owned_files) and all(
        isinstance(entry, Mapping)
        and set(entry) == {"path", "bytes", "sha256"}
        and isinstance(entry.get("path"), str)
        and bool(entry.get("path"))
        and isinstance(entry.get("bytes"), int)
        and not isinstance(entry.get("bytes"), bool)
        and int(entry.get("bytes", 0)) > 0
        and isinstance(entry.get("sha256"), str)
        and len(str(entry.get("sha256"))) == 64
        and all(
            char in "0123456789abcdef"
            for char in str(entry.get("sha256"))
        )
        for entry in owned_files
    )
    expected_historical_producer = {
        "source_snapshot_root_sha256": source_root,
        "authorization_receipt_sha256": authorization_sha,
    }
    if (
        same_implementation.get("schema")
        != (
            "pareto_v21e3r1_same_implementation_development_matrix_"
            "post_run_audit_v1"
        )
        or same_implementation.get("status")
        != "PASS_SAME_IMPLEMENTATION_POST_PROCESS_RECOMPUTATION"
        or same_implementation.get("implementation_independence") is not False
        or same_implementation.get("scientific_independence") is not False
        or same_implementation.get("external_third_party_audit") is not False
        or same_implementation.get(
            "fixed_author_generated_cases_descriptive_only"
        ) is not True
        or same_implementation.get("population_inference_authorized") is not False
        or same_implementation.get("sign_flip_assumptions_verified") is not False
        or same_implementation.get("trimmed_mean_distinct_from_mean") is not False
        or same_implementation.get("verifier_relationship")
        != "SAME_PROJECT_VERIFIER_POST_HOC_SUCCESSOR_NOT_HISTORICAL_PRODUCER"
        or same_implementation.get("historical_matrix_producer")
        != expected_historical_producer
        or same_implementation.get("source_snapshot_root_sha256") != source_root
        or same_implementation.get("authorization_receipt_sha256")
        != authorization_sha
        or same_implementation.get("objective_archive_and_metric_replayed_rows")
        != 108
        or same_implementation.get("submission_status") != "IJOC_HOLD"
        or not owned_files_valid
        or owned_paths != sorted(set(owned_paths))
        or same_implementation.get("live_verifier_owned_file_count")
        != len(owned_files)
        or same_implementation.get("live_verifier_owned_files_root_sha256")
        != hashlib.sha256(_canonical_bytes(owned_files)).hexdigest()
    ):
        raise RuntimeError(
            "Same-implementation post-process receipt is incomplete or differently bound."
        )
    _require_prohibitions(
        same_implementation,
        field="same-implementation post-process receipt",
        require_full_replay_boundary=True,
    )

    _, invalidation = _load_json(v3_invalidation_path, field="v3_invalidation")
    supersession = invalidation.get("v4_supersession")
    invalidation_findings = invalidation.get("invalidation_findings")
    if (
        invalidation.get("schema") != "pareto_v21e3r1_v3_invalidation_receipt_v1"
        or invalidation.get("status")
        != (
            "INVALIDATED_POST_EXECUTION_UNBOUND_DYNAMIC_TEST_DEPENDENCY_"
            "AND_FAILED_CLEAN_ROOM_P0"
        )
        or invalidation.get("v3_reuse_for_v4") != "PROHIBITED"
        or invalidation.get("v3_value_status")
        != "NON_AUTHORITATIVE_DEVELOPMENT_DIAGNOSTIC"
        or invalidation.get("formal_authorized") is not False
        or invalidation.get("submission_status") != "IJOC_HOLD"
        or invalidation_findings
        != [
            {
                "code": "P0_UNBOUND_DYNAMIC_TEST_DEPENDENCY",
                "status": "CONFIRMED",
            },
            {"code": "P0_FAILED_CLEAN_ROOM", "status": "CONFIRMED"},
        ]
        or not isinstance(supersession, Mapping)
        or supersession.get("status") != "BOUND_AUTHORIZED_SUCCESSOR"
        or supersession.get("source_snapshot_root_sha256") != source_root
    ):
        raise RuntimeError("V3 invalidation does not prohibit scientific reuse exactly.")
    supersession_bindings = (
        (
            supersession.get("source_snapshot"),
            v4_snapshot_path,
            snapshot_sha,
            "V3 invalidation V4 snapshot binding",
        ),
        (
            supersession.get("development_authorization"),
            v4_authorization_path,
            authorization_sha,
            "V3 invalidation V4 authorization binding",
        ),
    )
    for binding, path, digest, label in supersession_bindings:
        if (
            not isinstance(binding, Mapping)
            or binding.get("bytes") != path.stat().st_size
            or binding.get("sha256") != digest
        ):
            raise RuntimeError(f"{label} failed.")
        _require_archive_path(binding.get("path"), field=label)
    return {
        "source_snapshot_root_sha256": source_root,
        "source_snapshot_receipt_sha256": snapshot_sha,
        "authorization_receipt_sha256": authorization_sha,
        "prospective_source_root_sha256": _require_hex64(
            snapshot.get("prospective_source_root_sha256"),
            field="V4 prospective source root",
        ),
        "snapshot": snapshot,
        "authorization": authorization,
        "same_implementation": same_implementation,
    }


def _validate_frozen_inputs(
    *,
    build_manifest_path: Path,
    config_manifest_path: Path,
    metric_manifest_path: Path,
    reference_manifest_path: Path,
    protocol_path: Path,
) -> dict[str, str]:
    expected = (
        (
            build_manifest_path,
            "pareto_v21e3_development_manifest_build_receipt_v1",
            "PASS",
        ),
        (
            config_manifest_path,
            "pareto_v21e3_development_config_manifest_v1",
            "FROZEN_DEVELOPMENT_INPUT_CALIBRATION_EXECUTION_BLOCKED",
        ),
        (
            metric_manifest_path,
            "pareto_v21e3_metric_manifest_v1",
            "FROZEN_DEVELOPMENT_AND_FUTURE_CALIBRATION_INPUT",
        ),
        (
            reference_manifest_path,
            "pareto_v21e3_analytic_reference_manifest_v1",
            "FROZEN_DEVELOPMENT_ONLY",
        ),
        (
            protocol_path,
            "pareto_v21e3_c0_parity_protocol_v2",
            "ENGINEERING_ADAPTERS_AVAILABLE_SUCCESSOR_SNAPSHOT_PENDING",
        ),
    )
    for path, schema, status in expected:
        _, payload = _load_json(path, field=path.name)
        if payload.get("schema") != schema or payload.get("status") != status:
            raise RuntimeError(f"Frozen input identity changed: {path.name}")
    return {
        "config_manifest": _sha256(config_manifest_path),
        "metric_manifest": _sha256(metric_manifest_path),
        "reference_manifest": _sha256(reference_manifest_path),
        "protocol": _sha256(protocol_path),
    }


def _load_reference_contract(
    reference_manifest_path: Path,
) -> dict[str, dict[str, object]]:
    _, manifest = _load_json(reference_manifest_path, field="reference_manifest")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 12:
        raise RuntimeError("Frozen reference manifest must contain exactly 12 cases.")
    cases_by_id: dict[str, dict[str, object]] = {}
    strata: dict[tuple[object, object], int] = {}
    for item in cases:
        if not isinstance(item, Mapping):
            raise RuntimeError("Frozen reference manifest has a non-object case.")
        case_id = item.get("case_id")
        family = item.get("family")
        size = item.get("size")
        lower = item.get("objective_lower_bounds")
        upper = item.get("objective_upper_bounds")
        reference = item.get("normalized_reference_point")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in cases_by_id
            or family not in TARGET_FAMILIES
            or size not in (100, 200, 500)
            or not isinstance(lower, list)
            or not isinstance(upper, list)
            or len(lower) != 2
            or len(upper) != 2
            or reference != [1.0, 1.0]
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in (*lower, *upper)
            )
            or any(float(lo) >= float(hi) for lo, hi in zip(lower, upper))
        ):
            raise RuntimeError("Frozen reference manifest has an invalid analytic box.")
        cases_by_id[case_id] = {
            "family": family,
            "size": size,
            "analytic_box": {
                "lower": [float(value) for value in lower],
                "upper": [float(value) for value in upper],
                "normalized_reference": [1.0, 1.0],
            },
        }
        strata[(family, size)] = strata.get((family, size), 0) + 1
    if strata != {
        (family, size): 2
        for family in TARGET_FAMILIES
        for size in (100, 200, 500)
    }:
        raise RuntimeError("Frozen reference manifest is not the exact 12-case strata.")
    return cases_by_id


def _require_exact_keys(
    payload: Mapping[str, object], *, expected: set[str], field: str
) -> None:
    observed = set(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(f"{field} fields changed (missing={missing}, extra={extra}).")


def _require_exact_int(
    value: object, *, field: str, minimum: int = 0
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"{field} must be an integer >= {minimum}.")
    return value


def _require_unit_interval(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise RuntimeError(f"{field} must be finite and in [0,1].")
    return float(value)


def _terminal_payload_sha256(terminal: Mapping[str, object]) -> str:
    core = dict(terminal)
    core.pop("receipt_payload_sha256", None)
    raw = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_execution_receipt_semantics(
    *,
    row: Mapping[str, object],
    terminal: Mapping[str, object],
    pre: Mapping[str, object],
    objective: Mapping[str, object],
    metric: Mapping[str, object],
    budget: int,
    checkpoint_period: int,
    source_root: str,
    metric_manifest_sha256: str,
    analytic_box: Mapping[str, object],
    trace_bytes: int,
    label: str,
) -> None:
    """Reject self-consistent receipt rewrites that violate producer semantics."""

    identity_fields = ("case_id", "family", "size", "seed", "arm_id")
    _require_exact_keys(
        pre,
        expected={
            "schema",
            "status",
            *identity_fields,
            "source_snapshot_root_sha256",
            "trace_database_path",
            "detached_terminal_receipt_path",
            "detached_terminal_receipt_sha256",
            "selection_entropy_release",
            "calibration_execution",
            "formal_execution",
            "formal_authorized",
        },
        field=f"{label} preverification receipt",
    )
    if (
        pre.get("schema") != "pareto_v21e3r1_row_preverification_v1"
        or pre.get("status") != "READY_FOR_OBJECTIVE_AND_ARCHIVE_REPLAY"
        or _identity(pre) != _identity(row)
        or pre.get("source_snapshot_root_sha256") != source_root
    ):
        raise RuntimeError(f"{label} preverification identity/source binding failed.")
    _require_prohibitions(
        pre,
        field=f"{label} preverification receipt",
        require_full_replay_boundary=False,
    )

    terminal_fields = {
        "schema",
        "status",
        "problem",
        "family",
        "failure_code",
        "failure_detail",
        "attempt_count",
        "physical_call_started_count",
        "charged_evaluation_count",
        "decision_count",
        "cache_hit_count",
        "unresolved_decision_count",
        "terminal_evaluation_chain_sha256",
        "terminal_decision_chain_sha256",
        "terminal_attempt_chain_sha256",
        "run_context_digest_sha256",
        "database_path",
        "durability_mode",
        "finalization_gates",
        "receipt_payload_sha256",
    }
    _require_exact_keys(
        terminal, expected=terminal_fields, field=f"{label} terminal receipt"
    )
    cache_hits = _require_exact_int(
        terminal.get("cache_hit_count"), field=f"{label} terminal cache_hit_count"
    )
    attempts = _require_exact_int(
        terminal.get("attempt_count"), field=f"{label} terminal attempt_count"
    )
    physical_starts = _require_exact_int(
        terminal.get("physical_call_started_count"),
        field=f"{label} terminal physical_call_started_count",
    )
    charged = _require_exact_int(
        terminal.get("charged_evaluation_count"),
        field=f"{label} terminal charged_evaluation_count",
    )
    decisions = _require_exact_int(
        terminal.get("decision_count"), field=f"{label} terminal decision_count"
    )
    unresolved = _require_exact_int(
        terminal.get("unresolved_decision_count"),
        field=f"{label} terminal unresolved_decision_count",
    )
    if (
        terminal.get("schema") != "v21e3_terminal_receipt_v1"
        or terminal.get("status") != "SUCCESS"
        or terminal.get("problem") != row.get("case_id")
        or terminal.get("family") != row.get("family")
        or terminal.get("failure_code") is not None
        or terminal.get("failure_detail") is not None
        or terminal.get("durability_mode") != "SQLITE_WAL_SYNCHRONOUS_FULL"
        or charged != budget
        or decisions != budget
        or physical_starts != budget
        or unresolved != 0
        or attempts != budget + cache_hits
    ):
        raise RuntimeError(f"{label} terminal SUCCESS/counter contract failed.")
    chain_fields = (
        "run_context_digest_sha256",
        "terminal_attempt_chain_sha256",
        "terminal_evaluation_chain_sha256",
        "terminal_decision_chain_sha256",
    )
    for field in chain_fields:
        _require_hex64(terminal.get(field), field=f"{label} terminal {field}")
    terminal_payload_sha = _require_hex64(
        terminal.get("receipt_payload_sha256"),
        field=f"{label} terminal receipt_payload_sha256",
    )
    if terminal_payload_sha != _terminal_payload_sha256(terminal):
        raise RuntimeError(f"{label} terminal canonical payload hash failed.")
    gates = terminal.get("finalization_gates")
    expected_gates = {
        "expected_charged_evaluations": budget,
        "expected_decisions": budget,
        "run_context_charged_evaluation_budget": budget,
        "persisted_attempts": attempts,
        "persisted_evaluations": budget,
        "persisted_decisions": budget,
        "physical_call_starts": budget,
        "cache_hits": cache_hits,
        "nonterminal_attempts": 0,
        "evaluation_index_bounds": [1, budget],
        "expected_evaluation_index_bounds": [1, budget],
        "sqlite_integrity": "ok",
    }
    if gates != expected_gates:
        raise RuntimeError(f"{label} terminal finalization gates are not exact.")

    _require_exact_keys(
        objective,
        expected={
            "schema",
            "status",
            "verification_scope",
            "full_algorithm_decision_replay",
            "selection_authorization",
            "database_path",
            "database_bytes",
            "database_sha256",
            "detached_terminal_receipt_path",
            "detached_terminal_receipt_sha256",
            "attempt_records",
            "evaluation_records",
            "decision_records",
            "cache_hit_records",
            "unique_solution_replays",
            "archive_reconstruction",
            "archive_size",
            "terminal_status",
            "run_context_digest_sha256",
            "terminal_attempt_chain_sha256",
            "terminal_evaluation_chain_sha256",
            "terminal_decision_chain_sha256",
            "terminal_receipt_sha256",
        },
        field=f"{label} objective replay receipt",
    )
    unique_replays = _require_exact_int(
        objective.get("unique_solution_replays"),
        field=f"{label} objective unique_solution_replays",
        minimum=1,
    )
    objective_archive_size = _require_exact_int(
        objective.get("archive_size"),
        field=f"{label} objective archive_size",
        minimum=1,
    )
    objective_counts = {
        "attempt_records": attempts,
        "evaluation_records": budget,
        "decision_records": budget,
        "cache_hit_records": cache_hits,
    }
    if (
        objective.get("schema")
        != "v21e3r1_objective_archive_replay_receipt_v2"
        or objective.get("status") != "OBJECTIVE_AND_ARCHIVE_REPLAY_PASS"
        or objective.get("verification_scope")
        != "objective_solution_chain_archive_and_terminal_replay_v1"
        or objective.get("database_bytes") != trace_bytes
        or any(objective.get(field) != value for field, value in objective_counts.items())
        or unique_replays != budget
        or objective.get("archive_reconstruction") != "PASS"
        or objective.get("terminal_status") != "SUCCESS"
        or objective.get("terminal_receipt_sha256") != terminal_payload_sha
        or objective.get("full_algorithm_decision_replay") != "NOT_IMPLEMENTED"
        or objective.get("selection_authorization") != "PROHIBITED"
        or any(objective.get(field) != terminal.get(field) for field in chain_fields)
    ):
        raise RuntimeError(f"{label} objective/terminal semantic binding failed.")

    _require_exact_keys(
        metric,
        expected={
            "schema",
            "status",
            "verification_scope",
            "database_path",
            "database_sha256",
            "metric_manifest_sha256",
            "charged_evaluation_budget",
            "checkpoint_period",
            "analytic_box",
            "normalized_left_continuous_hv_auc",
            "normalized_terminal_hv",
            "checkpoints",
            "selection_authorization",
            "formal_authorized",
        },
        field=f"{label} metric replay receipt",
    )
    if (
        metric.get("schema") != "pareto_v21e3r1_metric_replay_receipt_v1"
        or metric.get("status") != "NORMALIZED_HV_AUC_REPLAY_PASS"
        or metric.get("verification_scope")
        != "frozen_metric_from_objective_ledger_checkpoints_v1"
        or metric.get("charged_evaluation_budget") != budget
        or metric.get("checkpoint_period") != checkpoint_period
        or metric.get("metric_manifest_sha256") != metric_manifest_sha256
        or metric.get("analytic_box") != dict(analytic_box)
        or metric.get("selection_authorization") != "PROHIBITED"
        or metric.get("formal_authorized") is not False
    ):
        raise RuntimeError(f"{label} metric frozen-contract binding failed.")
    observed_auc = _require_unit_interval(
        metric.get("normalized_left_continuous_hv_auc"),
        field=f"{label} normalized_left_continuous_hv_auc",
    )
    observed_terminal_hv = _require_unit_interval(
        metric.get("normalized_terminal_hv"),
        field=f"{label} normalized_terminal_hv",
    )
    checkpoints = metric.get("checkpoints")
    expected_grid = list(range(checkpoint_period, budget + 1, checkpoint_period))
    if not isinstance(checkpoints, list) or len(checkpoints) != len(expected_grid):
        raise RuntimeError(f"{label} metric checkpoint grid length changed.")
    previous_evaluation = 0
    previous_hv = 0.0
    area = 0.0
    for expected_evaluation, checkpoint in zip(expected_grid, checkpoints):
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError(f"{label} metric checkpoint is not an object.")
        _require_exact_keys(
            checkpoint,
            expected={"evaluation", "normalized_hv", "archive_size"},
            field=f"{label} metric checkpoint",
        )
        if checkpoint.get("evaluation") != expected_evaluation:
            raise RuntimeError(f"{label} metric checkpoint grid changed.")
        current_hv = _require_unit_interval(
            checkpoint.get("normalized_hv"),
            field=f"{label} checkpoint normalized_hv",
        )
        checkpoint_archive_size = _require_exact_int(
            checkpoint.get("archive_size"),
            field=f"{label} checkpoint archive_size",
            minimum=1,
        )
        area += previous_hv * (expected_evaluation - previous_evaluation)
        previous_evaluation = expected_evaluation
        previous_hv = current_hv
    recomputed_auc = area / float(budget)
    if (
        checkpoints[-1].get("normalized_hv") != metric.get("normalized_terminal_hv")
        or checkpoints[-1].get("archive_size") != objective_archive_size
        or not math.isclose(
            observed_auc, recomputed_auc, rel_tol=0.0, abs_tol=1e-15
        )
    ):
        raise RuntimeError(f"{label} metric terminal/AUC recomputation failed.")

    if (
        row.get("metric_manifest_sha256") != metric_manifest_sha256
        or row.get("trace_database_bytes") != trace_bytes
        or row.get("run_context_digest_sha256")
        != terminal.get("run_context_digest_sha256")
        or row.get("runtime_efficiency_claim_authorized") is not False
        or row.get("normalized_left_continuous_hv_auc")
        != metric.get("normalized_left_continuous_hv_auc")
        or row.get("normalized_terminal_hv") != metric.get("normalized_terminal_hv")
        or row.get("checkpoints") != checkpoints
    ):
        raise RuntimeError(f"{label} row/metric semantic binding failed.")


def _validate_code_release(
    *,
    code_archive_path: Path,
    code_manifest_path: Path,
    code_checksum_path: Path,
    code_clean_room_receipt_path: Path,
) -> dict[str, object]:
    archive = code_archive_path.resolve()
    if not archive.is_file() or archive.is_symlink():
        raise FileNotFoundError(archive)
    archive_bytes, archive_sha = _stream_binding(archive)
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip() is not None:
            raise RuntimeError("The bound code archive fails ZIP integrity.")
    _, manifest = _load_json(code_manifest_path, field="code_manifest")
    manifest_bytes, manifest_sha = _stream_binding(code_manifest_path)
    binding = manifest.get("archive")
    if (
        manifest.get("schema") != "ijoc_v21e3r1_standalone_release_manifest_v1"
        or manifest.get("formal_authorized") is not False
        or not isinstance(binding, Mapping)
        or binding.get("sha256") != archive_sha
        or binding.get("bytes") != archive_bytes
    ):
        raise RuntimeError("Code manifest does not bind the supplied code ZIP.")
    checksum_tokens = code_checksum_path.read_text(encoding="ascii").split()
    if not checksum_tokens or checksum_tokens[0] != archive_sha:
        raise RuntimeError("Code checksum does not bind the supplied code ZIP.")
    _, clean_room = _load_json(
        code_clean_room_receipt_path, field="code_clean_room_receipt"
    )
    pinned = clean_room.get("pinned_inputs")
    verified = clean_room.get("archive_verification")
    if (
        not str(clean_room.get("schema", "")).startswith(
            "ijoc_v21e3r1_clean_room_gate_receipt_"
        )
        or clean_room.get("status") != "PASS"
        or clean_room.get("formal_authorized") is not False
        or not isinstance(pinned, Mapping)
        or pinned.get("archive_sha256") != archive_sha
        or pinned.get("archive_bytes") != archive_bytes
        or pinned.get("manifest_sha256") != manifest_sha
        or pinned.get("manifest_bytes") != manifest_bytes
        or not isinstance(verified, Mapping)
        or verified.get("status") != "PASS"
        or verified.get("archive_sha256") != archive_sha
        or verified.get("manifest_sha256") != manifest_sha
        or clean_room.get("deterministic_rebuild_comparison_gate") != "PASS"
    ):
        raise RuntimeError("Code clean-room receipt does not close the supplied code ZIP.")
    return {"bytes": archive_bytes, "sha256": archive_sha}


def _validate_matrix(
    matrix_directory: Path,
    *,
    archive_prefix: str,
    source_root: str,
    authorization_sha: str,
    frozen_input_sha256: Mapping[str, str],
    reference_contract: Mapping[str, Mapping[str, object]],
    same_implementation: Mapping[str, object],
) -> tuple[list[ReleaseEntry], dict[str, object]]:
    matrix = matrix_directory.resolve()
    if not matrix.is_dir() or matrix.is_symlink():
        raise FileNotFoundError(matrix)
    if {path.name for path in matrix.iterdir()} != {
        "matrix.plan.json",
        "matrix.aggregate.json",
        "post_run_audit.json",
        "rows",
    }:
        raise RuntimeError("Completed matrix root contains missing or unexpected entries.")
    plan_path = matrix / "matrix.plan.json"
    aggregate_path = matrix / "matrix.aggregate.json"
    post_run_path = matrix / "post_run_audit.json"
    _, plan = _load_json(plan_path, field="matrix_plan")
    _, aggregate = _load_json(aggregate_path, field="matrix_aggregate")
    _, post_run = _load_json(post_run_path, field="matrix_post_run_audit")
    if (
        plan.get("schema")
        != "pareto_v21e3r1_development_matched_matrix_plan_v1"
        or plan.get("status") != "AUTHORIZED_DEVELOPMENT_MATRIX_PLAN"
        or plan.get("scientific_scope") != MATRIX_SCIENTIFIC_SCOPE
        or plan.get("expected_rows") != 108
        or plan.get("budget") != 2000
        or plan.get("checkpoint_period") != 200
        or plan.get("source_snapshot_root_sha256") != source_root
        or plan.get("authorization_receipt_sha256") != authorization_sha
    ):
        raise RuntimeError("Matrix plan identity or authorization binding changed.")
    _require_prohibitions(
        plan, field="matrix plan", require_full_replay_boundary=False
    )
    plan_inputs = plan.get("input_sha256")
    required_inputs = {
        "case_manifest",
        "config_manifest",
        "metric_manifest",
        "protocol",
        "reference_manifest",
    }
    if not isinstance(plan_inputs, Mapping) or set(plan_inputs) != required_inputs:
        raise RuntimeError("Matrix plan input bindings are incomplete.")
    normalized_inputs = {
        name: _require_hex64(value, field=f"matrix {name}")
        for name, value in plan_inputs.items()
    }
    if any(
        normalized_inputs[name] != expected
        for name, expected in frozen_input_sha256.items()
    ):
        raise RuntimeError("Matrix plan does not bind the supplied frozen inputs.")
    if (
        aggregate.get("schema")
        != "pareto_v21e3r1_development_matched_matrix_aggregate_v1"
        or aggregate.get("status")
        != "COMPLETE_DEVELOPMENT_MATRIX_ENGINEERING_EVIDENCE"
        or aggregate.get("scientific_scope") != MATRIX_SCIENTIFIC_SCOPE
        or aggregate.get("artifact_path_semantics") != MATRIX_PATH_SEMANTICS
        or aggregate.get("expected_rows") != 108
        or aggregate.get("observed_rows") != 108
        or aggregate.get("source_snapshot_root_sha256") != source_root
        or aggregate.get("authorization_receipt_sha256") != authorization_sha
        or aggregate.get("matrix_plan_sha256") != _sha256(plan_path)
        or aggregate.get("input_sha256") != dict(plan_inputs)
        or aggregate.get("runtime_efficiency_claim_authorized") is not False
    ):
        raise RuntimeError("Matrix aggregate is incomplete or nonportable.")
    _require_prohibitions(
        aggregate,
        field="matrix aggregate",
        require_full_replay_boundary=True,
    )
    if (
        post_run.get("schema")
        != "pareto_v21e3r1_development_matrix_post_run_audit_v1"
        or post_run.get("status") != "PASS_COMPLETE_DEVELOPMENT_MATRIX_AUDITED"
        or post_run.get("matrix_aggregate_path") != "matrix.aggregate.json"
        or post_run.get("matrix_aggregate_sha256") != _sha256(aggregate_path)
        or post_run.get("source_snapshot_root_sha256") != source_root
        or post_run.get("authorization_receipt_sha256") != authorization_sha
        or post_run.get("expected_rows") != 108
        or post_run.get("observed_rows") != 108
        or post_run.get("objective_and_archive_replay_pass_rows") != 108
        or post_run.get("metric_replay_pass_rows") != 108
        or post_run.get("runtime_efficiency_claim_authorized") is not False
    ):
        raise RuntimeError("Runner post-run audit is incomplete or nonportable.")
    _require_prohibitions(
        post_run,
        field="matrix post-run audit",
        require_full_replay_boundary=True,
    )
    if (
        same_implementation.get("matrix_plan_sha256") != _sha256(plan_path)
        or same_implementation.get("matrix_aggregate_sha256")
        != _sha256(aggregate_path)
        or same_implementation.get("runner_post_run_audit_sha256")
        != _sha256(post_run_path)
    ):
        raise RuntimeError(
            "Same-implementation receipt hashes another matrix result packet."
        )

    planned = plan.get("rows")
    aggregate_rows = aggregate.get("rows")
    if not isinstance(planned, list) or not isinstance(aggregate_rows, list):
        raise RuntimeError("Matrix plan or aggregate omits its row list.")
    if len(planned) != 108 or len(aggregate_rows) != 108:
        raise RuntimeError("Matrix row lists are not the exact 108-row product.")
    planned_by_slug: dict[str, Mapping[str, object]] = {}
    planned_identities: set[tuple[object, ...]] = set()
    case_cells: dict[tuple[object, object, object], set[tuple[object, object]]] = {}
    for item in planned:
        if not isinstance(item, Mapping):
            raise RuntimeError("Matrix plan has a non-object row.")
        _require_exact_keys(
            item,
            expected={"case_id", "family", "size", "seed", "arm_id", "row_slug"},
            field="matrix plan row",
        )
        slug = _require_archive_path(item.get("row_slug"), field="row_slug")
        identity = _identity(item)
        family = item.get("family")
        size = item.get("size")
        seed = item.get("seed")
        arm = item.get("arm_id")
        case_id = item.get("case_id")
        expected_slug = f"{case_id}__seed-{seed}__arm-{str(arm).lower()}"
        if (
            "/" in slug
            or slug != expected_slug
            or slug in planned_by_slug
            or identity in planned_identities
            or not isinstance(case_id, str)
            or not case_id
            or family not in TARGET_FAMILIES
            or size not in (100, 200, 500)
            or seed not in MATRIX_SEEDS
            or arm not in MATRIX_ARMS
        ):
            raise RuntimeError("Matrix plan row product or slug is not exact.")
        planned_by_slug[slug] = item
        planned_identities.add(identity)
        case_key = (case_id, family, size)
        case_cells.setdefault(case_key, set()).add((seed, arm))
    expected_cells = {
        (seed, arm) for seed in MATRIX_SEEDS for arm in MATRIX_ARMS
    }
    case_strata: dict[tuple[object, object], int] = {}
    for (_, family, size), cells in case_cells.items():
        if cells != expected_cells:
            raise RuntimeError("Matrix plan has an incomplete case/seed/arm product.")
        case_strata[(family, size)] = case_strata.get((family, size), 0) + 1
    if len(case_cells) != 12 or case_strata != {
        (family, size): 2
        for family in TARGET_FAMILIES
        for size in (100, 200, 500)
    }:
        raise RuntimeError("Matrix plan is not the exact 12-case stratified product.")
    aggregate_by_identity: dict[tuple[object, ...], Mapping[str, object]] = {}
    for item in aggregate_rows:
        if not isinstance(item, Mapping) or _identity(item) in aggregate_by_identity:
            raise RuntimeError("Matrix aggregate row identities are not unique.")
        _require_exact_keys(
            item,
            expected={
                "case_id",
                "family",
                "size",
                "seed",
                "arm_id",
                "normalized_left_continuous_hv_auc",
                "normalized_terminal_hv",
                "source_snapshot_root_sha256",
                "trace_database_sha256",
                "detached_terminal_receipt_sha256",
                "objective_archive_replay_receipt_sha256",
                "metric_replay_receipt_sha256",
                "row_receipt_path",
                "row_receipt_sha256",
            },
            field="matrix aggregate row",
        )
        aggregate_by_identity[_identity(item)] = item
    if set(aggregate_by_identity) != planned_identities:
        raise RuntimeError("Matrix aggregate identities differ from the plan product.")

    rows_root = matrix / "rows"
    if not rows_root.is_dir() or rows_root.is_symlink():
        raise RuntimeError("Matrix rows directory is missing or linked.")
    observed_slugs = {
        path.name
        for path in rows_root.iterdir()
        if path.is_dir() and not path.is_symlink()
    }
    if observed_slugs != set(planned_by_slug) or len(list(rows_root.iterdir())) != 108:
        raise RuntimeError("Matrix does not contain exactly the planned 108 row directories.")

    entries = [
        _capture_entry(
            plan_path,
            archive_path=f"{archive_prefix}/matrix/matrix.plan.json",
            role="matrix_plan",
        ),
        _capture_entry(
            aggregate_path,
            archive_path=f"{archive_prefix}/matrix/matrix.aggregate.json",
            role="matrix_aggregate",
        ),
        _capture_entry(
            post_run_path,
            archive_path=f"{archive_prefix}/matrix/post_run_audit.json",
            role="matrix_post_run_audit",
        ),
    ]
    file_roles = {
        "trace.sqlite3": "matrix_row_trace",
        "terminal.receipt.json": "matrix_row_terminal_receipt",
        "row.preverification.json": "matrix_row_preverification_receipt",
        "objective_archive_replay.receipt.json": "matrix_row_objective_replay_receipt",
        "metric_replay.receipt.json": "matrix_row_metric_replay_receipt",
        "row.json": "matrix_row_receipt",
    }
    for slug in sorted(planned_by_slug):
        row_directory = rows_root / slug
        children = list(row_directory.iterdir())
        if (
            {path.name for path in children} != ROW_FILES
            or any(not path.is_file() or path.is_symlink() for path in children)
        ):
            raise RuntimeError(
                f"Row {slug} must contain exactly six regular artifacts and no WAL/SHM/tmp."
            )
        paths = {name: row_directory / name for name in ROW_FILES}
        _, terminal = _load_json(paths["terminal.receipt.json"], field=f"{slug} terminal")
        _, pre = _load_json(
            paths["row.preverification.json"], field=f"{slug} preverification"
        )
        _, objective = _load_json(
            paths["objective_archive_replay.receipt.json"],
            field=f"{slug} objective replay",
        )
        _, metric = _load_json(
            paths["metric_replay.receipt.json"], field=f"{slug} metric replay"
        )
        _, row = _load_json(paths["row.json"], field=f"{slug} row")
        planned_row = planned_by_slug[slug]
        if (
            row.get("schema") != "pareto_v21e3r1_matched_matrix_row_v1"
            or row.get("status")
            != "PASS_ENGINEERING_ROW_OBJECTIVE_ARCHIVE_REPLAYED"
            or row.get("scientific_scope") != MATRIX_SCIENTIFIC_SCOPE
            or row.get("artifact_path_semantics") != ROW_PATH_SEMANTICS
            or row.get("source_snapshot_root_sha256") != source_root
            or _identity(row) != _identity(planned_row)
            or row.get("charged_evaluation_budget") != 2000
            or row.get("checkpoint_period") != 200
        ):
            raise RuntimeError(f"Row receipt identity changed: {slug}")
        _require_prohibitions(
            row, field=f"row {slug}", require_full_replay_boundary=True
        )
        path_contract = (
            (row, "trace_database_path", "trace.sqlite3"),
            (row, "detached_terminal_receipt_path", "terminal.receipt.json"),
            (row, "preverification_receipt_path", "row.preverification.json"),
            (
                row,
                "objective_archive_replay_receipt_path",
                "objective_archive_replay.receipt.json",
            ),
            (row, "metric_replay_receipt_path", "metric_replay.receipt.json"),
            (terminal, "database_path", "trace.sqlite3"),
            (pre, "trace_database_path", "trace.sqlite3"),
            (pre, "detached_terminal_receipt_path", "terminal.receipt.json"),
            (objective, "database_path", "trace.sqlite3"),
            (
                objective,
                "detached_terminal_receipt_path",
                "terminal.receipt.json",
            ),
            (metric, "database_path", "trace.sqlite3"),
        )
        for payload, field, expected in path_contract:
            _require_row_path(payload, field=field, expected=expected)
        hashes = {
            "trace_database_sha256": _sha256(paths["trace.sqlite3"]),
            "detached_terminal_receipt_sha256": _sha256(
                paths["terminal.receipt.json"]
            ),
            "preverification_receipt_sha256": _sha256(
                paths["row.preverification.json"]
            ),
            "objective_archive_replay_receipt_sha256": _sha256(
                paths["objective_archive_replay.receipt.json"]
            ),
            "metric_replay_receipt_sha256": _sha256(
                paths["metric_replay.receipt.json"]
            ),
        }
        if any(row.get(field) != digest for field, digest in hashes.items()):
            raise RuntimeError(f"Row artifact hash binding failed: {slug}")
        reference_case = reference_contract.get(str(row.get("case_id")))
        if (
            reference_case is None
            or reference_case.get("family") != row.get("family")
            or reference_case.get("size") != row.get("size")
            or not isinstance(reference_case.get("analytic_box"), Mapping)
        ):
            raise RuntimeError(f"Row case is absent from the frozen reference manifest: {slug}")
        _validate_execution_receipt_semantics(
            row=row,
            terminal=terminal,
            pre=pre,
            objective=objective,
            metric=metric,
            budget=2000,
            checkpoint_period=200,
            source_root=source_root,
            metric_manifest_sha256=normalized_inputs["metric_manifest"],
            analytic_box=reference_case["analytic_box"],
            trace_bytes=paths["trace.sqlite3"].stat().st_size,
            label=f"matrix row {slug}",
        )
        if (
            terminal.get("schema") != "v21e3_terminal_receipt_v1"
            or terminal.get("status") != "SUCCESS"
            or pre.get("schema") != "pareto_v21e3r1_row_preverification_v1"
            or pre.get("status") != "READY_FOR_OBJECTIVE_AND_ARCHIVE_REPLAY"
            or objective.get("schema")
            != "v21e3r1_objective_archive_replay_receipt_v2"
            or objective.get("status") != "OBJECTIVE_AND_ARCHIVE_REPLAY_PASS"
            or metric.get("schema") != "pareto_v21e3r1_metric_replay_receipt_v1"
            or metric.get("status") != "NORMALIZED_HV_AUC_REPLAY_PASS"
            or pre.get("detached_terminal_receipt_sha256")
            != hashes["detached_terminal_receipt_sha256"]
            or objective.get("database_sha256") != hashes["trace_database_sha256"]
            or objective.get("detached_terminal_receipt_sha256")
            != hashes["detached_terminal_receipt_sha256"]
            or objective.get("full_algorithm_decision_replay")
            != "NOT_IMPLEMENTED"
            or objective.get("selection_authorization") != "PROHIBITED"
            or metric.get("database_sha256") != hashes["trace_database_sha256"]
            or metric.get("metric_manifest_sha256")
            != normalized_inputs["metric_manifest"]
            or not isinstance(
                metric.get("normalized_left_continuous_hv_auc"), (int, float)
            )
            or isinstance(metric.get("normalized_left_continuous_hv_auc"), bool)
            or not 0.0
            <= float(metric.get("normalized_left_continuous_hv_auc"))
            <= 1.0
            or not isinstance(metric.get("normalized_terminal_hv"), (int, float))
            or isinstance(metric.get("normalized_terminal_hv"), bool)
            or not 0.0 <= float(metric.get("normalized_terminal_hv")) <= 1.0
            or metric.get("normalized_left_continuous_hv_auc")
            != row.get("normalized_left_continuous_hv_auc")
            or metric.get("normalized_terminal_hv")
            != row.get("normalized_terminal_hv")
            or not isinstance(metric.get("checkpoints"), list)
            or not metric.get("checkpoints")
            or metric.get("checkpoints") != row.get("checkpoints")
            or metric.get("selection_authorization") != "PROHIBITED"
            or metric.get("formal_authorized") is not False
        ):
            raise RuntimeError(f"Row replay receipt binding failed: {slug}")
        aggregate_row = aggregate_by_identity.get(_identity(row))
        expected_row_path = f"rows/{slug}/row.json"
        if (
            aggregate_row is None
            or aggregate_row.get("row_receipt_path") != expected_row_path
            or aggregate_row.get("row_receipt_sha256") != _sha256(paths["row.json"])
            or aggregate_row.get("source_snapshot_root_sha256") != source_root
            or aggregate_row.get("normalized_left_continuous_hv_auc")
            != row.get("normalized_left_continuous_hv_auc")
            or aggregate_row.get("normalized_terminal_hv")
            != row.get("normalized_terminal_hv")
            or any(aggregate_row.get(field) != digest for field, digest in hashes.items() if field != "preverification_receipt_sha256")
        ):
            raise RuntimeError(f"Aggregate-to-row binding failed: {slug}")
        for name in sorted(ROW_FILES):
            entries.append(
                _capture_entry(
                    paths[name],
                    archive_path=(
                        f"{archive_prefix}/matrix/rows/{slug}/{name}"
                    ),
                    role=file_roles[name],
                )
            )
    if len(entries) != 651:
        raise RuntimeError("Matrix release inventory is not exactly 651 files.")
    return entries, {
        "matrix_plan_sha256": _sha256(plan_path),
        "matrix_aggregate_sha256": _sha256(aggregate_path),
        "matrix_post_run_audit_sha256": _sha256(post_run_path),
        "development_gate": aggregate.get("analysis", {}).get("overall_gate")
        if isinstance(aggregate.get("analysis"), Mapping)
        else None,
    }


def _validate_target_execution(
    target_execution_directory: Path,
    *,
    archive_prefix: str,
    snapshot: Mapping[str, object],
    authorization: Mapping[str, object],
    prospective_source_root: str,
    frozen_input_sha256: Mapping[str, str],
    reference_contract: Mapping[str, Mapping[str, object]],
) -> tuple[list[ReleaseEntry], dict[str, object]]:
    target = target_execution_directory.resolve()
    if not target.is_dir() or target.is_symlink():
        raise FileNotFoundError(target)
    root_children = list(target.iterdir())
    if (
        {path.name for path in root_children}
        != {"target_structural.plan.json", TARGET_RECEIPT_NAME, "rows"}
        or len(root_children) != 3
    ):
        raise RuntimeError(
            "Target-size execution root contains missing or unexpected entries."
        )

    plan_path = target / "target_structural.plan.json"
    receipt_path = target / TARGET_RECEIPT_NAME
    rows_root = target / "rows"
    if (
        not plan_path.is_file()
        or plan_path.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.is_symlink()
        or not rows_root.is_dir()
        or rows_root.is_symlink()
    ):
        raise RuntimeError("Target-size execution packet contains a linked or absent root artifact.")
    _, plan = _load_json(plan_path, field="target_size_plan")
    _, receipt = _load_json(receipt_path, field="target_size_execution_receipt")

    plan_rows = plan.get("rows")
    if (
        plan.get("schema") != "pareto_v21e3r1_target_size_three_arm_plan_v1"
        or plan.get("status") != "READY_TARGET_SIZE_SMALL_BUDGET_ENGINEERING_ONLY"
        or plan.get("scientific_scope") != TARGET_SCIENTIFIC_SCOPE
        or plan.get("source_snapshot_root_sha256") != prospective_source_root
        or plan.get("budget") != 200
        or plan.get("checkpoint_period") != 200
        or plan.get("baseline_population_sizes") != {"MOTSP": 48, "MOKP": 40}
        or plan.get("budget_initializes_every_baseline_population") is not True
        or plan.get("development_parity_execution")
        != "NOT_AUTHORIZED_BY_THIS_PLAN"
        or not isinstance(plan_rows, list)
        or len(plan_rows) != 6
    ):
        raise RuntimeError("Target-size execution plan is not the frozen six-row plan.")
    _require_prohibitions(
        plan, field="target-size plan", require_full_replay_boundary=False
    )

    plan_inputs = plan.get("input_sha256")
    receipt_inputs = receipt.get("input_sha256")
    required_inputs = {
        "case_manifest",
        "config_manifest",
        "metric_manifest",
        "protocol",
        "reference_manifest",
    }
    if (
        not isinstance(plan_inputs, Mapping)
        or not isinstance(receipt_inputs, Mapping)
        or set(plan_inputs) != required_inputs
        or dict(receipt_inputs) != dict(plan_inputs)
    ):
        raise RuntimeError("Target-size input bindings are incomplete or inconsistent.")
    normalized_inputs = {
        name: _require_hex64(value, field=f"target-size {name}")
        for name, value in plan_inputs.items()
    }
    if any(
        normalized_inputs[name] != expected
        for name, expected in frozen_input_sha256.items()
    ):
        raise RuntimeError("Target-size packet does not bind the supplied frozen inputs.")

    receipt_rows = receipt.get("rows")
    observed_pairs = {
        (item.get("family"), item.get("arm_id"))
        for item in receipt_rows
        if isinstance(item, Mapping)
    } if isinstance(receipt_rows, list) else set()
    if (
        receipt.get("schema")
        != "pareto_v21e3r1_target_size_execution_receipt_v1"
        or receipt.get("status")
        != "PASS_TARGET_SIZE_THREE_ARM_EXECUTION_ENGINEERING_ONLY"
        or receipt.get("scientific_scope") != TARGET_SCIENTIFIC_SCOPE
        or receipt.get("artifact_path_semantics") != TARGET_PATH_SEMANTICS
        or receipt.get("source_snapshot_root_sha256") != prospective_source_root
        or receipt.get("target_size") != 500
        or receipt.get("seed") != 31051
        or receipt.get("charged_evaluation_budget") != 200
        or receipt.get("checkpoint_period") != 200
        or receipt.get("baseline_population_sizes") != {"MOTSP": 48, "MOKP": 40}
        or receipt.get("budget_initializes_every_baseline_population") is not True
        or receipt.get("row_count") != 6
        or receipt.get("families") != list(TARGET_FAMILIES)
        or receipt.get("arms") != list(TARGET_ARMS)
        or not isinstance(receipt_rows, list)
        or len(receipt_rows) != 6
        or observed_pairs
        != {
            (family, arm)
            for family in TARGET_FAMILIES
            for arm in TARGET_ARMS
        }
        or receipt.get("objective_and_archive_replay_pass_rows") != 6
        or receipt.get("metric_replay_pass_rows") != 6
        or receipt.get("target_budget_evidence") != "NOT_ESTABLISHED"
        or receipt.get("performance_evidence") != "NOT_ESTABLISHED"
        or receipt.get("runtime_efficiency_claim_authorized") is not False
        or receipt.get("development_parity_execution")
        != "NOT_AUTHORIZED_BY_THIS_RECEIPT"
    ):
        raise RuntimeError("Target-size execution receipt is not the held six-row receipt.")
    _require_prohibitions(
        receipt,
        field="target-size execution receipt",
        require_full_replay_boundary=True,
    )

    plan_sha = _sha256(plan_path)
    receipt_sha = _sha256(receipt_path)
    stored_plan_path = _require_archive_path(
        receipt.get("plan_path"), field="target-size plan_path"
    )
    stored_base = PurePosixPath(stored_plan_path).parent
    expected_receipt_storage = (stored_base / TARGET_RECEIPT_NAME).as_posix()
    stored_snapshot_receipt = _require_archive_path(
        snapshot.get("target_size_execution_receipt_path"),
        field="V4 target-size execution receipt path",
    )
    if (
        PurePosixPath(stored_plan_path).name != plan_path.name
        or stored_base.name != target.name
        or receipt.get("plan_sha256") != plan_sha
        or stored_snapshot_receipt != expected_receipt_storage
        or authorization.get("target_size_execution_receipt_sha256") != receipt_sha
    ):
        raise RuntimeError("V4 snapshot/authorization does not bind this target-size packet.")

    planned_by_slug: dict[str, Mapping[str, object]] = {}
    expected_pairs = {
        (family, arm) for family in TARGET_FAMILIES for arm in TARGET_ARMS
    }
    planned_pairs: set[tuple[object, object]] = set()
    assert isinstance(plan_rows, list)
    for item in plan_rows:
        if not isinstance(item, Mapping):
            raise RuntimeError("Target-size plan contains a non-object row.")
        _require_exact_keys(
            item,
            expected={"case_id", "family", "size", "seed", "arm_id", "row_slug"},
            field="target-size plan row",
        )
        slug = _require_archive_path(item.get("row_slug"), field="target row_slug")
        pair = (item.get("family"), item.get("arm_id"))
        expected_slug = (
            f"{item.get('case_id')}__seed-{item.get('seed')}__"
            f"arm-{str(item.get('arm_id')).lower()}"
        )
        if (
            "/" in slug
            or slug != expected_slug
            or slug in planned_by_slug
            or pair in planned_pairs
            or pair not in expected_pairs
            or item.get("size") != 500
            or item.get("seed") != 31051
        ):
            raise RuntimeError("Target-size plan row product is not exact.")
        planned_by_slug[slug] = item
        planned_pairs.add(pair)
    if planned_pairs != expected_pairs:
        raise RuntimeError("Target-size plan omits a family/arm cell.")

    summary_by_identity: dict[tuple[object, ...], Mapping[str, object]] = {}
    assert isinstance(receipt_rows, list)
    for item in receipt_rows:
        if not isinstance(item, Mapping) or _identity(item) in summary_by_identity:
            raise RuntimeError("Target-size receipt row identities are not unique.")
        _require_exact_keys(
            item,
            expected={
                "case_id",
                "family",
                "size",
                "seed",
                "arm_id",
                "charged_evaluation_budget",
                "checkpoint_period",
                "normalized_terminal_hv_diagnostic_only",
                "trace_database_sha256",
                "detached_terminal_receipt_sha256",
                "objective_archive_replay_receipt_sha256",
                "metric_replay_receipt_sha256",
                "row_receipt_path",
                "row_receipt_sha256",
                "objective_and_archive_replay",
                "metric_replay",
            },
            field="target-size receipt row",
        )
        summary_by_identity[_identity(item)] = item
    observed_row_children = list(rows_root.iterdir())
    if (
        len(observed_row_children) != 6
        or {path.name for path in observed_row_children} != set(planned_by_slug)
        or any(not path.is_dir() or path.is_symlink() for path in observed_row_children)
    ):
        raise RuntimeError("Target-size packet does not contain exactly six row directories.")

    file_roles = {
        "trace.sqlite3": "target_size_row_trace",
        "terminal.receipt.json": "target_size_row_terminal_receipt",
        "row.preverification.json": "target_size_row_preverification_receipt",
        "objective_archive_replay.receipt.json": "target_size_row_objective_replay_receipt",
        "metric_replay.receipt.json": "target_size_row_metric_replay_receipt",
        "row.json": "target_size_row_receipt",
    }
    entries = [
        _capture_entry(
            plan_path,
            archive_path=f"{archive_prefix}/target_size_execution/{plan_path.name}",
            role="target_size_plan",
        ),
        _capture_entry(
            receipt_path,
            archive_path=f"{archive_prefix}/target_size_execution/{receipt_path.name}",
            role="target_size_execution_receipt",
        ),
    ]
    if entries[0].sha256 != plan_sha or entries[1].sha256 != receipt_sha:
        raise RuntimeError("Target-size root artifact changed after validation.")

    child_bindings: list[dict[str, object]] = []
    for slug in sorted(planned_by_slug):
        row_directory = rows_root / slug
        row_children = list(row_directory.iterdir())
        if (
            {path.name for path in row_children} != ROW_FILES
            or len(row_children) != 6
            or any(not path.is_file() or path.is_symlink() for path in row_children)
        ):
            raise RuntimeError(
                f"Target row {slug} must contain exactly six regular artifacts and no WAL/SHM/tmp."
            )
        paths = {name: row_directory / name for name in ROW_FILES}
        _, terminal = _load_json(paths["terminal.receipt.json"], field=f"{slug} terminal")
        _, pre = _load_json(paths["row.preverification.json"], field=f"{slug} preverification")
        _, objective = _load_json(
            paths["objective_archive_replay.receipt.json"],
            field=f"{slug} objective replay",
        )
        _, metric = _load_json(
            paths["metric_replay.receipt.json"], field=f"{slug} metric replay"
        )
        _, row = _load_json(paths["row.json"], field=f"{slug} row")
        planned_row = planned_by_slug[slug]
        if (
            row.get("schema") != "pareto_v21e3r1_matched_matrix_row_v1"
            or row.get("status")
            != "PASS_ENGINEERING_ROW_OBJECTIVE_ARCHIVE_REPLAYED"
            or row.get("scientific_scope") != TARGET_SCIENTIFIC_SCOPE
            or row.get("artifact_path_semantics") != ROW_PATH_SEMANTICS
            or row.get("source_snapshot_root_sha256") != prospective_source_root
            or _identity(row) != _identity(planned_row)
            or row.get("charged_evaluation_budget") != 200
            or row.get("checkpoint_period") != 200
        ):
            raise RuntimeError(f"Target-size row identity changed: {slug}")
        _require_prohibitions(
            row, field=f"target-size row {slug}", require_full_replay_boundary=True
        )
        path_contract = (
            (row, "trace_database_path", "trace.sqlite3"),
            (row, "detached_terminal_receipt_path", "terminal.receipt.json"),
            (row, "preverification_receipt_path", "row.preverification.json"),
            (
                row,
                "objective_archive_replay_receipt_path",
                "objective_archive_replay.receipt.json",
            ),
            (row, "metric_replay_receipt_path", "metric_replay.receipt.json"),
            (terminal, "database_path", "trace.sqlite3"),
            (pre, "trace_database_path", "trace.sqlite3"),
            (pre, "detached_terminal_receipt_path", "terminal.receipt.json"),
            (objective, "database_path", "trace.sqlite3"),
            (
                objective,
                "detached_terminal_receipt_path",
                "terminal.receipt.json",
            ),
            (metric, "database_path", "trace.sqlite3"),
        )
        for payload, field, expected in path_contract:
            _require_row_path(payload, field=field, expected=expected)
        hashes = {
            "trace_database_sha256": _sha256(paths["trace.sqlite3"]),
            "detached_terminal_receipt_sha256": _sha256(
                paths["terminal.receipt.json"]
            ),
            "preverification_receipt_sha256": _sha256(
                paths["row.preverification.json"]
            ),
            "objective_archive_replay_receipt_sha256": _sha256(
                paths["objective_archive_replay.receipt.json"]
            ),
            "metric_replay_receipt_sha256": _sha256(
                paths["metric_replay.receipt.json"]
            ),
        }
        if any(row.get(field) != digest for field, digest in hashes.items()):
            raise RuntimeError(f"Target-size row artifact hash binding failed: {slug}")
        reference_case = reference_contract.get(str(row.get("case_id")))
        if (
            reference_case is None
            or reference_case.get("family") != row.get("family")
            or reference_case.get("size") != row.get("size")
            or not isinstance(reference_case.get("analytic_box"), Mapping)
        ):
            raise RuntimeError(
                f"Target-size case is absent from the frozen reference manifest: {slug}"
            )
        _validate_execution_receipt_semantics(
            row=row,
            terminal=terminal,
            pre=pre,
            objective=objective,
            metric=metric,
            budget=200,
            checkpoint_period=200,
            source_root=prospective_source_root,
            metric_manifest_sha256=normalized_inputs["metric_manifest"],
            analytic_box=reference_case["analytic_box"],
            trace_bytes=paths["trace.sqlite3"].stat().st_size,
            label=f"target-size row {slug}",
        )
        if (
            terminal.get("schema") != "v21e3_terminal_receipt_v1"
            or terminal.get("status") != "SUCCESS"
            or pre.get("schema") != "pareto_v21e3r1_row_preverification_v1"
            or pre.get("status") != "READY_FOR_OBJECTIVE_AND_ARCHIVE_REPLAY"
            or pre.get("detached_terminal_receipt_sha256")
            != hashes["detached_terminal_receipt_sha256"]
            or objective.get("schema")
            != "v21e3r1_objective_archive_replay_receipt_v2"
            or objective.get("status") != "OBJECTIVE_AND_ARCHIVE_REPLAY_PASS"
            or objective.get("database_sha256") != hashes["trace_database_sha256"]
            or objective.get("detached_terminal_receipt_sha256")
            != hashes["detached_terminal_receipt_sha256"]
            or objective.get("full_algorithm_decision_replay") != "NOT_IMPLEMENTED"
            or objective.get("selection_authorization") != "PROHIBITED"
            or metric.get("schema") != "pareto_v21e3r1_metric_replay_receipt_v1"
            or metric.get("status") != "NORMALIZED_HV_AUC_REPLAY_PASS"
            or metric.get("database_sha256") != hashes["trace_database_sha256"]
            or metric.get("metric_manifest_sha256")
            != normalized_inputs["metric_manifest"]
            or metric.get("normalized_left_continuous_hv_auc")
            != row.get("normalized_left_continuous_hv_auc")
            or metric.get("normalized_terminal_hv")
            != row.get("normalized_terminal_hv")
            or metric.get("checkpoints") != row.get("checkpoints")
            or metric.get("selection_authorization") != "PROHIBITED"
            or metric.get("formal_authorized") is not False
        ):
            raise RuntimeError(f"Target-size replay receipt binding failed: {slug}")

        summary = summary_by_identity.get(_identity(row))
        expected_row_storage = (stored_base / "rows" / slug / "row.json").as_posix()
        summary_fields = (
            "case_id",
            "family",
            "size",
            "seed",
            "arm_id",
            "charged_evaluation_budget",
            "checkpoint_period",
            "trace_database_sha256",
            "detached_terminal_receipt_sha256",
            "objective_archive_replay_receipt_sha256",
            "metric_replay_receipt_sha256",
        )
        if (
            summary is None
            or any(summary.get(field) != row.get(field) for field in summary_fields)
            or summary.get("objective_and_archive_replay") != "PASS"
            or summary.get("metric_replay") != "PASS"
            or summary.get("normalized_terminal_hv_diagnostic_only")
            != row.get("normalized_terminal_hv")
            or _require_archive_path(
                summary.get("row_receipt_path"), field="target row_receipt_path"
            )
            != expected_row_storage
            or summary.get("row_receipt_sha256") != _sha256(paths["row.json"])
        ):
            raise RuntimeError(f"Target-size receipt-to-row binding failed: {slug}")

        for name in sorted(ROW_FILES):
            entry = _capture_entry(
                paths[name],
                archive_path=(
                    f"{archive_prefix}/target_size_execution/rows/{slug}/{name}"
                ),
                role=file_roles[name],
            )
            if entry.sha256 != _sha256(paths[name]):
                raise RuntimeError(f"Target-size artifact changed after validation: {slug}")
            entries.append(entry)
            child_bindings.append(
                {
                    "path": (stored_base / "rows" / slug / name).as_posix(),
                    "bytes": entry.bytes,
                    "sha256": entry.sha256,
                }
            )
    child_bindings.sort(key=lambda item: str(item["path"]))
    child_root = hashlib.sha256(_canonical_bytes(child_bindings)).hexdigest()
    if (
        authorization.get("target_size_execution_child_artifact_root_sha256")
        != child_root
    ):
        raise RuntimeError("V4 authorization target-size child-artifact root mismatch.")
    if len(entries) != 38:
        raise RuntimeError("Target-size release inventory is not exactly 38 files.")
    return entries, {
        "target_size_plan_sha256": plan_sha,
        "target_size_execution_receipt_sha256": receipt_sha,
        "target_size_execution_child_artifact_root_sha256": child_root,
        "row_count": 6,
        "scientific_scope": TARGET_SCIENTIFIC_SCOPE,
        "performance_evidence": "NOT_ESTABLISHED",
    }


def _stream_file_into_zip(
    archive: zipfile.ZipFile,
    entry: ReleaseEntry,
    *,
    chunk_size: int,
) -> None:
    info = zipfile.ZipInfo(entry.archive_path, (1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_STORED
    info.file_size = entry.bytes
    digest = hashlib.sha256()
    total = 0
    with entry.source_path.open("rb") as source, archive.open(
        info, mode="w", force_zip64=True
    ) as destination:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            destination.write(chunk)
            digest.update(chunk)
            total += len(chunk)
    if total != entry.bytes or digest.hexdigest() != entry.sha256:
        raise RuntimeError(f"TOCTOU: release input changed while streaming: {entry.role}")


def _verify_zip(path: Path, entries: Sequence[ReleaseEntry]) -> None:
    expected = {entry.archive_path: entry for entry in entries}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if [info.filename for info in infos] != [entry.archive_path for entry in entries]:
            raise RuntimeError("Finished results ZIP has another ordered file inventory.")
        if len(infos) != len(expected):
            raise RuntimeError("Finished results ZIP has duplicate entries.")
        for info in infos:
            entry = expected[info.filename]
            if (
                info.date_time != (1980, 1, 1, 0, 0, 0)
                or info.compress_type != zipfile.ZIP_STORED
                or info.create_system != 3
                or info.external_attr >> 16 != 0o100644
                or info.file_size != entry.bytes
            ):
                raise RuntimeError(f"Finished ZIP metadata drifted: {info.filename}")
            digest = hashlib.sha256()
            total = 0
            with archive.open(info, "r") as handle:
                while True:
                    chunk = handle.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    digest.update(chunk)
                    total += len(chunk)
            if total != entry.bytes or digest.hexdigest() != entry.sha256:
                raise RuntimeError(f"Finished ZIP content verification failed: {info.filename}")


def _verify_live_inputs(entries: Sequence[ReleaseEntry]) -> None:
    for entry in entries:
        observed_bytes, observed_sha = _stream_binding(entry.source_path)
        if observed_bytes != entry.bytes or observed_sha != entry.sha256:
            raise RuntimeError(f"TOCTOU: release input changed after capture: {entry.role}")


def _verify_external_binding(
    path: Path, binding: Mapping[str, object], *, label: str
) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"TOCTOU: {label} is no longer a regular file.")
    before = path.stat()
    observed_bytes, observed_sha = _stream_binding(path)
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or observed_bytes != binding.get("bytes")
        or observed_sha != binding.get("sha256")
    ):
        raise RuntimeError(f"TOCTOU: {label} changed after its release binding.")


def _write_temporary(path: Path, raw: bytes) -> Path:
    temporary = path.with_name("." + path.name + ".building")
    if temporary.exists():
        raise FileExistsError(temporary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _publish_temporary(temporary: Path, destination: Path) -> None:
    os.link(temporary, destination)
    temporary.unlink()


def _cleanup(paths: Sequence[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _build_results_release_from_staged(
    *,
    matrix_directory: str | Path,
    target_execution_directory: str | Path,
    same_implementation_receipt_path: str | Path,
    v3_invalidation_path: str | Path,
    v4_snapshot_path: str | Path,
    v4_authorization_path: str | Path,
    build_manifest_path: str | Path,
    config_manifest_path: str | Path,
    metric_manifest_path: str | Path,
    reference_manifest_path: str | Path,
    protocol_path: str | Path,
    code_archive_path: str | Path,
    code_manifest_path: str | Path,
    code_checksum_path: str | Path,
    code_clean_room_receipt_path: str | Path,
    archive_path: str | Path,
    file_manifest_path: str | Path,
    release_index_path: str | Path,
    archive_prefix: str = DEFAULT_PREFIX,
    chunk_size: int = CHUNK_SIZE,
    captured_packet: CapturedPacket,
) -> dict[str, object]:
    """Validate and exclusively publish a deterministic data-only ZIP64 packet."""

    prefix = _require_archive_path(archive_prefix, field="archive_prefix")
    if "/" in prefix:
        raise ValueError("archive_prefix must be one portable path component.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    outputs = tuple(
        Path(path).resolve()
        for path in (archive_path, file_manifest_path, release_index_path)
    )
    if len(set(outputs)) != 3:
        raise ValueError("Results release outputs must be three distinct paths.")
    if any(path.exists() for path in outputs):
        raise FileExistsError("Results release outputs are exclusive and versioned.")

    same_implementation_path = Path(same_implementation_receipt_path).resolve()
    invalidation_path = Path(v3_invalidation_path).resolve()
    snapshot_path = Path(v4_snapshot_path).resolve()
    authorization_path = Path(v4_authorization_path).resolve()
    build_path = Path(build_manifest_path).resolve()
    config_path = Path(config_manifest_path).resolve()
    metric_path = Path(metric_manifest_path).resolve()
    reference_path = Path(reference_manifest_path).resolve()
    protocol = Path(protocol_path).resolve()
    code_archive = Path(code_archive_path).resolve()
    code_manifest = Path(code_manifest_path).resolve()
    code_checksum = Path(code_checksum_path).resolve()
    clean_room = Path(code_clean_room_receipt_path).resolve()

    evidence = _validate_external_evidence(
        same_implementation_receipt_path=same_implementation_path,
        v3_invalidation_path=invalidation_path,
        v4_snapshot_path=snapshot_path,
        v4_authorization_path=authorization_path,
    )
    frozen_input_sha256 = _validate_frozen_inputs(
        build_manifest_path=build_path,
        config_manifest_path=config_path,
        metric_manifest_path=metric_path,
        reference_manifest_path=reference_path,
        protocol_path=protocol,
    )
    reference_contract = _load_reference_contract(reference_path)
    code_binding = _validate_code_release(
        code_archive_path=code_archive,
        code_manifest_path=code_manifest,
        code_checksum_path=code_checksum,
        code_clean_room_receipt_path=clean_room,
    )
    matrix_entries, matrix_binding = _validate_matrix(
        Path(matrix_directory),
        archive_prefix=prefix,
        source_root=str(evidence["source_snapshot_root_sha256"]),
        authorization_sha=str(evidence["authorization_receipt_sha256"]),
        frozen_input_sha256=frozen_input_sha256,
        reference_contract=reference_contract,
        same_implementation=evidence["same_implementation"],
    )
    target_entries, target_binding = _validate_target_execution(
        Path(target_execution_directory),
        archive_prefix=prefix,
        snapshot=evidence["snapshot"],
        authorization=evidence["authorization"],
        prospective_source_root=str(evidence["prospective_source_root_sha256"]),
        frozen_input_sha256=frozen_input_sha256,
        reference_contract=reference_contract,
    )
    ancillary_specs = (
        (
            same_implementation_path,
            "evidence/same_implementation_development_matrix_"
            "post_run_audit_v6.json",
            "same_implementation_post_process_receipt",
        ),
        (invalidation_path, "evidence/v3_invalidation_receipt.json", "v3_invalidation_receipt"),
        (snapshot_path, "evidence/v4_source_snapshot.json", "v4_source_snapshot"),
        (authorization_path, "evidence/v4_authorization.json", "v4_authorization"),
        (build_path, "frozen_inputs/build_receipt.json", "frozen_build_manifest"),
        (config_path, "frozen_inputs/config_manifest_development.json", "frozen_config_manifest"),
        (metric_path, "frozen_inputs/metric_manifest.json", "frozen_metric_manifest"),
        (reference_path, "frozen_inputs/reference_manifest_development.json", "frozen_reference_manifest"),
        (protocol, "frozen_inputs/V21E3_C0_PARITY_PROTOCOL_V2.json", "frozen_protocol"),
        (code_manifest, "code_release/experiment_code.manifest.json", "code_release_manifest"),
        (code_checksum, "code_release/experiment_code.zip.sha256", "code_release_checksum"),
        (clean_room, "code_release/clean_room.receipt.json", "code_clean_room_receipt"),
    )
    entries = list(matrix_entries) + list(target_entries)
    for source, relative, role in ancillary_specs:
        entries.append(
            _capture_entry(
                source,
                archive_path=f"{prefix}/{relative}",
                role=role,
            )
        )
    entries.sort(key=lambda entry: entry.archive_path)
    if len(entries) != 701 or len({entry.archive_path for entry in entries}) != 701:
        raise RuntimeError("Results release inventory must contain exactly 701 unique files.")
    if any(entry.source_path == code_archive for entry in entries):
        raise RuntimeError("Code ZIP must not be nested in the results data ZIP.")

    archive_output, manifest_output, index_output = outputs
    archive_temporary = archive_output.with_name("." + archive_output.name + ".building")
    if archive_temporary.exists():
        raise FileExistsError(archive_temporary)
    archive_output.parent.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    temporaries: list[Path] = []
    try:
        with zipfile.ZipFile(
            archive_temporary,
            mode="x",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
            strict_timestamps=True,
        ) as archive:
            for entry in entries:
                _stream_file_into_zip(
                    archive, entry, chunk_size=chunk_size
                )
        temporaries.append(archive_temporary)
        _verify_zip(archive_temporary, entries)
        _verify_live_inputs(entries)
        _verify_external_binding(code_archive, code_binding, label="code archive")
        _verify_captured_packet(captured_packet)
        archive_bytes, archive_sha = _stream_binding(archive_temporary)

        same_receipt = evidence["same_implementation"]
        manifest_payload = {
            "schema": "ijoc_v21e3r1_results_release_file_manifest_v2",
            "status": "PASS_CAPTURED_AND_ZIP_VERIFIED",
            "archive_prefix": prefix,
            "archive_format": "DETERMINISTIC_ZIP64_STORED_V1",
            "fixed_timestamp": "1980-01-01T00:00:00",
            "fixed_file_mode_octal": "100644",
            "file_count": len(entries),
            "files": [
                {
                    "archive_path": entry.archive_path,
                    "role": entry.role,
                    "bytes": entry.bytes,
                    "sha256": entry.sha256,
                }
                for entry in entries
            ],
            "scientific_scope": "development_only_engineering_evidence_not_formal_evidence",
            "post_process_relationship": same_receipt["verifier_relationship"],
            "implementation_independence": False,
            "scientific_independence": False,
            "external_third_party_audit": False,
            "fixed_author_generated_cases_descriptive_only": True,
            "population_inference_authorized": False,
            "sign_flip_assumptions_verified": False,
            "trimmed_mean_distinct_from_mean": False,
            "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
            "formal_authorized": False,
            "submission_status": "IJOC_HOLD",
        }
        manifest_raw = _canonical_bytes(manifest_payload)
        manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
        index_payload = {
            "schema": "ijoc_v21e3r1_results_release_index_v2",
            "status": "PASS_VERIFIED_DEVELOPMENT_RESULTS_RELEASE",
            "scientific_scope": "development_only_engineering_evidence_not_formal_evidence",
            "data_archive": {
                "bytes": archive_bytes,
                "sha256": archive_sha,
                "file_count": len(entries),
                "format": "DETERMINISTIC_ZIP64_STORED_V1",
                "archive_prefix": prefix,
            },
            "external_file_manifest": {
                "bytes": len(manifest_raw),
                "sha256": manifest_sha,
                "schema": manifest_payload["schema"],
            },
            "code_archive_binding": {
                **code_binding,
                "included_in_data_archive": False,
                "artifact_relationship": (
                    "HISTORICAL_MATRIX_PRODUCER_CODE_ARCHIVE_V4"
                ),
                "historical_matrix_source_snapshot_root_sha256": evidence[
                    "source_snapshot_root_sha256"
                ],
                "live_verifier_relationship": (
                    "NOT_THE_LIVE_POST_HOC_VERIFIER_CODE_IDENTITY"
                ),
            },
            "v4_source_snapshot": {
                "root_sha256": evidence["source_snapshot_root_sha256"],
                "receipt_sha256": evidence["source_snapshot_receipt_sha256"],
            },
            "v4_authorization_receipt_sha256": evidence[
                "authorization_receipt_sha256"
            ],
            "matrix_bindings": matrix_binding,
            "target_size_execution_bindings": target_binding,
            "fixed_author_generated_cases_descriptive_only": True,
            "population_inference_authorized": False,
            "sign_flip_assumptions_verified": False,
            "trimmed_mean_distinct_from_mean": False,
            "same_implementation_post_process": {
                "receipt_sha256": _sha256(same_implementation_path),
                "schema": same_receipt["schema"],
                "status": same_receipt["status"],
                "verifier_relationship": same_receipt[
                    "verifier_relationship"
                ],
                "implementation_independence": False,
                "scientific_independence": False,
                "external_third_party_audit": False,
                "fixed_author_generated_cases_descriptive_only": True,
                "population_inference_authorized": False,
                "sign_flip_assumptions_verified": False,
                "trimmed_mean_distinct_from_mean": False,
                "live_verifier_owned_file_count": same_receipt[
                    "live_verifier_owned_file_count"
                ],
                "live_verifier_owned_files_root_sha256": same_receipt[
                    "live_verifier_owned_files_root_sha256"
                ],
            },
            "v3_invalidation_receipt_sha256": _sha256(invalidation_path),
            "development_gate": matrix_binding["development_gate"],
            "selection_entropy_release": "PROHIBITED",
            "calibration_execution": "PROHIBITED",
            "formal_execution": "PROHIBITED",
            "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
            "formal_authorized": False,
            "submission_status": "IJOC_HOLD",
        }
        index_raw = _canonical_bytes(index_payload)
        manifest_temporary = _write_temporary(manifest_output, manifest_raw)
        index_temporary = _write_temporary(index_output, index_raw)
        temporaries.extend((manifest_temporary, index_temporary))

        _publish_temporary(archive_temporary, archive_output)
        temporaries.remove(archive_temporary)
        published.append(archive_output)
        _publish_temporary(manifest_temporary, manifest_output)
        temporaries.remove(manifest_temporary)
        published.append(manifest_output)
        _publish_temporary(index_temporary, index_output)
        temporaries.remove(index_temporary)
        published.append(index_output)
        return index_payload
    except Exception:
        _cleanup(tuple(temporaries) + tuple(published) + (archive_temporary,))
        raise


def build_results_release(
    *,
    matrix_directory: str | Path,
    target_execution_directory: str | Path,
    same_implementation_receipt_path: str | Path,
    v3_invalidation_path: str | Path,
    v4_snapshot_path: str | Path,
    v4_authorization_path: str | Path,
    build_manifest_path: str | Path,
    config_manifest_path: str | Path,
    metric_manifest_path: str | Path,
    reference_manifest_path: str | Path,
    protocol_path: str | Path,
    code_archive_path: str | Path,
    code_manifest_path: str | Path,
    code_checksum_path: str | Path,
    code_clean_room_receipt_path: str | Path,
    archive_path: str | Path,
    file_manifest_path: str | Path,
    release_index_path: str | Path,
    archive_prefix: str = DEFAULT_PREFIX,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, object]:
    """Capture once, then validate and publish only from that immutable packet."""

    outputs = tuple(
        Path(path).resolve()
        for path in (archive_path, file_manifest_path, release_index_path)
    )
    if len(set(outputs)) != 3:
        raise ValueError("Results release outputs must be three distinct paths.")
    if any(path.exists() for path in outputs):
        raise FileExistsError("Results release outputs are exclusive and versioned.")
    tree_sources = (
        Path(matrix_directory).resolve(strict=True),
        Path(target_execution_directory).resolve(strict=True),
    )
    for output in outputs:
        for tree_source in tree_sources:
            try:
                output.relative_to(tree_source)
            except ValueError:
                continue
            raise ValueError("Results release outputs must be outside captured input trees.")
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    input_paths = {
        "matrix_directory": matrix_directory,
        "target_execution_directory": target_execution_directory,
        "same_implementation_receipt_path": same_implementation_receipt_path,
        "v3_invalidation_path": v3_invalidation_path,
        "v4_snapshot_path": v4_snapshot_path,
        "v4_authorization_path": v4_authorization_path,
        "build_manifest_path": build_manifest_path,
        "config_manifest_path": config_manifest_path,
        "metric_manifest_path": metric_manifest_path,
        "reference_manifest_path": reference_manifest_path,
        "protocol_path": protocol_path,
        "code_archive_path": code_archive_path,
        "code_manifest_path": code_manifest_path,
        "code_checksum_path": code_checksum_path,
        "code_clean_room_receipt_path": code_clean_room_receipt_path,
    }
    default_temporary_parent = Path(tempfile.gettempdir()).resolve()
    staging_candidates = [outputs[0].parent]
    if default_temporary_parent.drive.lower() == outputs[0].drive.lower():
        staging_candidates.append(default_temporary_parent)
    staging_parent = min(staging_candidates, key=lambda path: len(str(path)))
    temporary_prefix = ".p-"
    with tempfile.TemporaryDirectory(
        prefix=temporary_prefix, dir=staging_parent
    ) as temporary_name:
        packet = _stage_release_packet(Path(temporary_name).resolve(), input_paths)
        staged = packet.paths
        return _build_results_release_from_staged(
            matrix_directory=staged["matrix_directory"],
            target_execution_directory=staged["target_execution_directory"],
            same_implementation_receipt_path=staged[
                "same_implementation_receipt_path"
            ],
            v3_invalidation_path=staged["v3_invalidation_path"],
            v4_snapshot_path=staged["v4_snapshot_path"],
            v4_authorization_path=staged["v4_authorization_path"],
            build_manifest_path=staged["build_manifest_path"],
            config_manifest_path=staged["config_manifest_path"],
            metric_manifest_path=staged["metric_manifest_path"],
            reference_manifest_path=staged["reference_manifest_path"],
            protocol_path=staged["protocol_path"],
            code_archive_path=staged["code_archive_path"],
            code_manifest_path=staged["code_manifest_path"],
            code_checksum_path=staged["code_checksum_path"],
            code_clean_room_receipt_path=staged[
                "code_clean_room_receipt_path"
            ],
            archive_path=outputs[0],
            file_manifest_path=outputs[1],
            release_index_path=outputs[2],
            archive_prefix=archive_prefix,
            chunk_size=chunk_size,
            captured_packet=packet,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-directory", type=Path, required=True)
    parser.add_argument("--target-execution-directory", type=Path, required=True)
    parser.add_argument(
        "--same-implementation-receipt", type=Path, required=True
    )
    parser.add_argument("--v3-invalidation", type=Path, required=True)
    parser.add_argument("--v4-snapshot", type=Path, required=True)
    parser.add_argument("--v4-authorization", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--config-manifest", type=Path, required=True)
    parser.add_argument("--metric-manifest", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--code-archive", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--code-checksum", type=Path, required=True)
    parser.add_argument("--code-clean-room-receipt", type=Path, required=True)
    parser.add_argument("--archive-path", type=Path, required=True)
    parser.add_argument("--file-manifest-path", type=Path, required=True)
    parser.add_argument("--release-index-path", type=Path, required=True)
    parser.add_argument("--archive-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()
    result = build_results_release(
        matrix_directory=args.matrix_directory,
        target_execution_directory=args.target_execution_directory,
        same_implementation_receipt_path=args.same_implementation_receipt,
        v3_invalidation_path=args.v3_invalidation,
        v4_snapshot_path=args.v4_snapshot,
        v4_authorization_path=args.v4_authorization,
        build_manifest_path=args.build_manifest,
        config_manifest_path=args.config_manifest,
        metric_manifest_path=args.metric_manifest,
        reference_manifest_path=args.reference_manifest,
        protocol_path=args.protocol,
        code_archive_path=args.code_archive,
        code_manifest_path=args.code_manifest,
        code_checksum_path=args.code_checksum,
        code_clean_room_receipt_path=args.code_clean_room_receipt,
        archive_path=args.archive_path,
        file_manifest_path=args.file_manifest_path,
        release_index_path=args.release_index_path,
        archive_prefix=args.archive_prefix,
        chunk_size=args.chunk_size,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "data_archive": result["data_archive"],
                "submission_status": result["submission_status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
