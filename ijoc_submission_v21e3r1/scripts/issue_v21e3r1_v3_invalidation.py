from __future__ import annotations

"""Issue the deterministic invalidation receipt for the executed V21e3r1 V3 chain.

The receipt is deliberately retrospective.  It binds the immutable V3 execution
artifacts while proving that a test file admitted by the V3 snapshot dynamically
loaded two files that neither the snapshot nor the V3 release archive admitted.
It also binds the resulting clean-room P0 failure.  Nothing in this module
authorizes reuse of V3 values or execution of a successor phase.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Mapping, Sequence
import zipfile


SCHEMA = "pareto_v21e3r1_v3_invalidation_receipt_v1"
INVALIDATION_STATUS = (
    "INVALIDATED_POST_EXECUTION_UNBOUND_DYNAMIC_TEST_DEPENDENCY_"
    "AND_FAILED_CLEAN_ROOM_P0"
)
V3_VALUE_STATUS = "NON_AUTHORITATIVE_DEVELOPMENT_DIAGNOSTIC"
V3_SOURCE_ROOT_SHA256 = (
    "f8aa5cd1f57b51654d303fa4e9a2996c717249799869087508b92cd1aae10114"
)
V3_AUTHORIZATION_SHA256 = (
    "37a86a0d398a7583f924fd70933f7d2b81970b9fceb57ace4bb701ebad7e7ea6"
)
V3_TRIGGERING_TEST_PATH = "tests/test_pareto_v21e3_release.py"
V3_TRIGGERING_TEST_BYTES = 14_567
V3_TRIGGERING_TEST_SHA256 = (
    "04e6ae5fa321d011c5bb429339544b51e6b9dd13c1ef66246c7a0a911b83bd4b"
)
V21E3_PARENT_PREFIX = "ijoc_v21e3_experiment_code"
V21E3R1_V3_PREFIX = "ijoc_v21e3r1_experiment_code"
DYNAMIC_PATHS = (
    "ijoc_submission_v21e3/scripts/build_v21e3_code_release.py",
    "ijoc_submission_v21e3/scripts/verify_v21e3_clean_room.py",
)


ARTIFACT_PATHS = {
    "v3_source_snapshot": (
        "ijoc_submission_v21e3r1/provenance/"
        "V21E3R1_DEVELOPMENT_SOURCE_SNAPSHOT_FREEZE_V3.json"
    ),
    "v3_development_authorization": (
        "ijoc_submission_v21e3r1/provenance/"
        "V21E3R1_DEVELOPMENT_PARITY_AUTHORIZATION_V3.json"
    ),
    "v3_matrix_aggregate": (
        "ijoc_submission_v21e3r1/provenance/"
        "V21E3R1_DEVELOPMENT_MATRIX_AGGREGATE_V1.json"
    ),
    "v3_matrix_raw_aggregate": (
        "artifacts/v21e3r1_development_parity_v3/matrix.aggregate.json"
    ),
    "v3_runner_post_run_audit": (
        "ijoc_submission_v21e3r1/provenance/"
        "V21E3R1_DEVELOPMENT_MATRIX_RUNNER_POST_RUN_AUDIT_V1.json"
    ),
    "v3_matrix_raw_runner_post_run_audit": (
        "artifacts/v21e3r1_development_parity_v3/post_run_audit.json"
    ),
    "v3_independent_post_run_audit": (
        "ijoc_submission_v21e3r1/provenance/"
        "V21E3R1_INDEPENDENT_DEVELOPMENT_MATRIX_POST_RUN_AUDIT_V1.json"
    ),
    "v3_matrix_stdout_log": "artifacts/v21e3r1_matrix_logs_v3/stdout.log",
    "v3_matrix_stderr_log": "artifacts/v21e3r1_matrix_logs_v3/stderr.log",
    "v3_independent_audit_stdout_log": (
        "artifacts/v21e3r1_independent_audit_logs_v1/stdout.log"
    ),
    "v3_independent_audit_stderr_log": (
        "artifacts/v21e3r1_independent_audit_logs_v1/stderr.log"
    ),
    "v3_clean_room_stdout_log": (
        "artifacts/v21e3r1_clean_room_logs_v1/stdout.log"
    ),
    "v3_clean_room_stderr_log": (
        "artifacts/v21e3r1_clean_room_logs_v1/stderr.log"
    ),
    "v3_release_archive": (
        "ijoc_submission_v21e3r1/release/ijoc_v21e3r1_experiment_code.zip"
    ),
    "v3_release_manifest": (
        "ijoc_submission_v21e3r1/release/"
        "ijoc_v21e3r1_experiment_code.manifest.json"
    ),
    "v3_release_checksum": (
        "ijoc_submission_v21e3r1/release/"
        "ijoc_v21e3r1_experiment_code.zip.sha256"
    ),
    "v3_clean_room_fail_receipt": (
        "ijoc_submission_v21e3r1/release/"
        "ijoc_v21e3r1_clean_room.receipt.json"
    ),
    "v3_clean_room_step05_log": (
        "artifacts/v21e3r1_clean_room_v1/logs/"
        "05_run_extracted_tree_v21_tests.log"
    ),
    "v21e3_immutable_parent_archive": (
        "ijoc_submission_v21e3/release/ijoc_v21e3_experiment_code.zip"
    ),
}


EXPECTED_ARTIFACT_SHA256 = {
    "v3_source_snapshot": (
        "f8e5506188edfc818ec2a95fb024f4de4d7f1f43c732ad4909198a022b1b9294"
    ),
    "v3_development_authorization": V3_AUTHORIZATION_SHA256,
    "v3_matrix_aggregate": (
        "de7e5d9507a4a69bb79cd8ab5af163ca74209e8281a265ca2154f2d4dd1ce526"
    ),
    "v3_matrix_raw_aggregate": (
        "de7e5d9507a4a69bb79cd8ab5af163ca74209e8281a265ca2154f2d4dd1ce526"
    ),
    "v3_runner_post_run_audit": (
        "fbe15200e16276a083c98f4aa656ec76d0b873d69d7329bd504286a4fe576ce1"
    ),
    "v3_matrix_raw_runner_post_run_audit": (
        "fbe15200e16276a083c98f4aa656ec76d0b873d69d7329bd504286a4fe576ce1"
    ),
    "v3_independent_post_run_audit": (
        "93d95698ca600e7b2515ac03b9cdc91511925769b5d1af65d52053802405d19f"
    ),
    "v3_matrix_stdout_log": (
        "5036129fce2fd418b38005eac5e9f824f06d391e25d7b72c11357cc264f70047"
    ),
    "v3_matrix_stderr_log": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "v3_independent_audit_stdout_log": (
        "27b66c2b11a15716ca5324937b0e93b79e6c24b276958ab0810f54a74350615b"
    ),
    "v3_independent_audit_stderr_log": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "v3_clean_room_stdout_log": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "v3_clean_room_stderr_log": (
        "6b2f8561ba1465fc36306d4a64ea79db398f4affe15e553684b8a3f9cd5e73c4"
    ),
    "v3_release_archive": (
        "ba56e35ab4d043a4709eb157f272f214089f97d6210ca88baeb7d3f07fa0b099"
    ),
    "v3_release_manifest": (
        "6c1db072fcce60260ce42a27f17337231d21a503a8e52f3d36f385e7062f2f28"
    ),
    "v3_release_checksum": (
        "51c9321d7b229b310feeeb8f2fa5cf00091aa0efed448d5227b938dc50f52653"
    ),
    "v3_clean_room_fail_receipt": (
        "71f4fb5d5a933836b8f31a4a16e135f1c8927f708d6fdbc7ab23a7307b3fab6e"
    ),
    "v3_clean_room_step05_log": (
        "8db60d120860345deab782f3ff9190005f974174509f2c64f2a3568a9e567773"
    ),
    "v21e3_immutable_parent_archive": (
        "7881b30e6f6059e36e0ed8279f8932ab5f48f2f8e0bc38885e59a74fb45fb3b0"
    ),
}


EXPECTED_DYNAMIC_SHA256 = {
    DYNAMIC_PATHS[0]: {
        "live": (
            "29f117db34068c1a566a886d5fcb80318818351ee37cf1f46dee73e5ccaaec16"
        ),
        "immutable_parent": (
            "29f117db34068c1a566a886d5fcb80318818351ee37cf1f46dee73e5ccaaec16"
        ),
    },
    DYNAMIC_PATHS[1]: {
        "live": (
            "d95c4c3ee12cf9ad723f370f4d9df75dc7ebceaf3172279881da601c91686ad9"
        ),
        "immutable_parent": (
            "b7f811ec1ce129b219bb9ed0ea8897302d8701c646d2092ce98cdae5999ba2b0"
        ),
    },
}


def default_verification_contract() -> dict[str, object]:
    """Return a fresh contract pinned to the historical on-disk V3 artifacts."""

    return {
        "artifact_paths": dict(ARTIFACT_PATHS),
        "artifact_sha256": dict(EXPECTED_ARTIFACT_SHA256),
        "authorization_sha256": V3_AUTHORIZATION_SHA256,
        "dynamic_paths": list(DYNAMIC_PATHS),
        "dynamic_sha256": {
            path: dict(digests) for path, digests in EXPECTED_DYNAMIC_SHA256.items()
        },
        "matrix_row_count": 108,
        "parent_archive_prefix": V21E3_PARENT_PREFIX,
        "release_file_count": 147,
        "snapshot_bound_file_count": 326,
        "source_root_sha256": V3_SOURCE_ROOT_SHA256,
        "triggering_test_bytes": V3_TRIGGERING_TEST_BYTES,
        "triggering_test_path": V3_TRIGGERING_TEST_PATH,
        "triggering_test_sha256": V3_TRIGGERING_TEST_SHA256,
        "v3_archive_prefix": V21E3R1_V3_PREFIX,
    }


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


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_regular_file(path: Path, *, label: str) -> tuple[Path, bytes]:
    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symbolic link: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing {label}: {path}") from error
    if not resolved.is_file():
        raise RuntimeError(f"{label} is not a regular file: {resolved}")
    return resolved, resolved.read_bytes()


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _binding(path: Path, raw: bytes, repo_root: Path) -> dict[str, object]:
    return {
        "bytes": len(raw),
        "path": _display_path(path, repo_root),
        "sha256": _sha256(raw),
    }


def _read_canonical_json(
    path: Path, *, label: str, repo_root: Path
) -> tuple[Mapping[str, object], bytes, dict[str, object]]:
    resolved, raw = _read_regular_file(path, label=label)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON: {resolved}") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{label} must be a JSON object: {resolved}")
    if _canonical_bytes(payload) != raw:
        raise RuntimeError(f"{label} is not canonical newline-terminated JSON.")
    return payload, raw, _binding(resolved, raw, repo_root)


def _require_hex_digest(value: object, *, label: str) -> str:
    digest = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RuntimeError(f"{label} is not a lowercase SHA-256 digest.")
    return digest


def _validated_contract(
    supplied: Mapping[str, object] | None,
) -> dict[str, object]:
    contract = default_verification_contract() if supplied is None else dict(supplied)
    required = {
        "artifact_paths",
        "artifact_sha256",
        "authorization_sha256",
        "dynamic_paths",
        "dynamic_sha256",
        "matrix_row_count",
        "parent_archive_prefix",
        "release_file_count",
        "snapshot_bound_file_count",
        "source_root_sha256",
        "triggering_test_bytes",
        "triggering_test_path",
        "triggering_test_sha256",
        "v3_archive_prefix",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise ValueError(f"Verification contract is missing keys: {missing}")
    artifact_paths = contract["artifact_paths"]
    artifact_sha256 = contract["artifact_sha256"]
    if not isinstance(artifact_paths, Mapping) or not isinstance(
        artifact_sha256, Mapping
    ):
        raise ValueError("Verification contract artifact maps must be mappings.")
    expected_names = set(ARTIFACT_PATHS)
    if set(artifact_paths) != expected_names or set(artifact_sha256) != expected_names:
        raise ValueError("Verification contract must bind every required artifact exactly.")
    for name in expected_names:
        _require_hex_digest(
            artifact_sha256[name], label=f"contract artifact hash for {name}"
        )
    source_root = _require_hex_digest(
        contract["source_root_sha256"], label="contract source root"
    )
    authorization_sha256 = _require_hex_digest(
        contract["authorization_sha256"], label="contract authorization hash"
    )
    triggering_test_sha256 = _require_hex_digest(
        contract["triggering_test_sha256"], label="contract triggering-test hash"
    )
    dynamic_paths_value = contract["dynamic_paths"]
    if not isinstance(dynamic_paths_value, Sequence) or isinstance(
        dynamic_paths_value, (str, bytes)
    ):
        raise ValueError("Verification contract dynamic_paths must be a sequence.")
    dynamic_paths = tuple(str(path) for path in dynamic_paths_value)
    if len(dynamic_paths) != 2 or len(set(dynamic_paths)) != 2:
        raise ValueError("Verification contract must identify exactly two dynamic paths.")
    for path in dynamic_paths:
        pure = PurePosixPath(path)
        if (
            path != pure.as_posix()
            or pure.is_absolute()
            or ".." in pure.parts
            or "\\" in path
        ):
            raise ValueError(f"Dynamic dependency path is not canonical: {path!r}")
    dynamic_sha256 = contract["dynamic_sha256"]
    if not isinstance(dynamic_sha256, Mapping) or set(dynamic_sha256) != set(
        dynamic_paths
    ):
        raise ValueError("Verification contract dynamic hashes do not match its paths.")
    normalized_dynamic_hashes: dict[str, dict[str, str]] = {}
    for path in dynamic_paths:
        relation = dynamic_sha256[path]
        if not isinstance(relation, Mapping):
            raise ValueError(f"Dynamic hash record is not an object: {path}")
        normalized_dynamic_hashes[path] = {
            "immutable_parent": _require_hex_digest(
                relation.get("immutable_parent"),
                label=f"immutable-parent hash for {path}",
            ),
            "live": _require_hex_digest(
                relation.get("live"), label=f"live hash for {path}"
            ),
        }
    integer_fields = (
        "matrix_row_count",
        "release_file_count",
        "snapshot_bound_file_count",
        "triggering_test_bytes",
    )
    for field in integer_fields:
        if not isinstance(contract[field], int) or int(contract[field]) < 0:
            raise ValueError(f"Verification contract {field} must be non-negative.")
    for field in ("parent_archive_prefix", "triggering_test_path", "v3_archive_prefix"):
        value = str(contract[field])
        pure = PurePosixPath(value)
        if (
            not value
            or value != pure.as_posix()
            or pure.is_absolute()
            or ".." in pure.parts
            or "\\" in value
        ):
            raise ValueError(f"Verification contract {field} is not canonical.")
    contract["artifact_paths"] = dict(artifact_paths)
    contract["artifact_sha256"] = {
        str(name): str(digest) for name, digest in artifact_sha256.items()
    }
    contract["authorization_sha256"] = authorization_sha256
    contract["dynamic_paths"] = dynamic_paths
    contract["dynamic_sha256"] = normalized_dynamic_hashes
    contract["source_root_sha256"] = source_root
    contract["triggering_test_sha256"] = triggering_test_sha256
    return contract


def _validate_snapshot(
    snapshot: Mapping[str, object],
    *,
    source_root_sha256: str,
    bound_file_count: int,
) -> dict[str, Mapping[str, object]]:
    _require(
        snapshot.get("schema")
        == "pareto_v21e3r1_development_source_snapshot_freeze_v1",
        "Unexpected V3 source snapshot schema.",
    )
    _require(
        snapshot.get("status") == "PASS_ENGINEERING_SNAPSHOT_ONLY",
        "The V3 source snapshot is not the engineering snapshot.",
    )
    _require(snapshot.get("formal_authorized") is False, "V3 must not be formal.")
    _require(
        snapshot.get("submission_status") == "IJOC_HOLD",
        "The V3 source snapshot must retain IJOC_HOLD.",
    )
    entries = snapshot.get("bound_files")
    if not isinstance(entries, list):
        raise RuntimeError("V3 snapshot bound_files must be a list.")
    _require(
        snapshot.get("bound_file_count") == len(entries) == bound_file_count,
        "Unexpected V3 bound-file count.",
    )
    indexed: dict[str, Mapping[str, object]] = {}
    ordered_paths: list[str] = []
    for index, item in enumerate(entries):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"V3 bound_files[{index}] is not an object.")
        path = str(item.get("path"))
        pure = PurePosixPath(path)
        if (
            path != pure.as_posix()
            or pure.is_absolute()
            or ".." in pure.parts
            or "\\" in path
        ):
            raise RuntimeError(f"Non-canonical V3 snapshot path: {path!r}")
        if path in indexed:
            raise RuntimeError(f"Duplicate V3 snapshot path: {path}")
        size = item.get("bytes")
        if not isinstance(size, int) or size < 0:
            raise RuntimeError(f"Invalid byte count for V3 snapshot path: {path}")
        _require_hex_digest(item.get("sha256"), label=f"snapshot hash for {path}")
        indexed[path] = item
        ordered_paths.append(path)
    _require(
        ordered_paths == sorted(ordered_paths),
        "V3 snapshot paths are not canonically ordered.",
    )
    computed_root = _sha256(_canonical_bytes(entries))
    _require(
        computed_root == snapshot.get("bound_files_root_sha256") == source_root_sha256,
        "V3 bound-files root does not match its immutable root.",
    )
    return indexed


def _validate_authorization(
    authorization: Mapping[str, object],
    *,
    snapshot_sha256: str,
    source_root_sha256: str,
) -> None:
    _require(
        authorization.get("schema")
        == "pareto_v21e3r1_development_parity_authorization_v1",
        "Unexpected V3 development authorization schema.",
    )
    _require(
        authorization.get("status") == "AUTHORIZED_DEVELOPMENT_PARITY_ONLY",
        "Unexpected V3 development authorization status.",
    )
    _require(
        authorization.get("source_snapshot_receipt_sha256") == snapshot_sha256,
        "V3 authorization does not bind the V3 snapshot receipt.",
    )
    _require(
        authorization.get("source_snapshot_root_sha256") == source_root_sha256,
        "V3 authorization does not bind the immutable snapshot root.",
    )
    _require(
        authorization.get("formal_authorized") is False,
        "V3 authorization must remain non-formal.",
    )
    _require(
        authorization.get("submission_status") == "IJOC_HOLD",
        "V3 authorization must retain IJOC_HOLD.",
    )


def _validate_matrix_chain(
    *,
    aggregate: Mapping[str, object],
    aggregate_sha256: str,
    runner: Mapping[str, object],
    runner_sha256: str,
    independent: Mapping[str, object],
    authorization_sha256: str,
    source_root_sha256: str,
    row_count: int,
) -> dict[str, object]:
    _require(
        aggregate.get("schema")
        == "pareto_v21e3r1_development_matched_matrix_aggregate_v1",
        "Unexpected V3 matrix aggregate schema.",
    )
    _require(
        aggregate.get("status")
        == "COMPLETE_DEVELOPMENT_MATRIX_ENGINEERING_EVIDENCE",
        "Unexpected V3 matrix aggregate status.",
    )
    _require(
        aggregate.get("authorization_receipt_sha256") == authorization_sha256,
        "V3 matrix aggregate does not bind the authorization.",
    )
    _require(
        aggregate.get("source_snapshot_root_sha256") == source_root_sha256,
        "V3 matrix aggregate does not bind the source root.",
    )
    _require(
        aggregate.get("expected_rows")
        == aggregate.get("observed_rows")
        == row_count,
        "V3 matrix row count does not match its verification contract.",
    )
    rows = aggregate.get("rows")
    _require(
        isinstance(rows, list) and len(rows) == row_count,
        "V3 matrix rows are incomplete.",
    )
    _require(
        aggregate.get("formal_authorized") is False
        and aggregate.get("formal_execution") == "PROHIBITED",
        "V3 aggregate contains an invalid formal boundary.",
    )

    _require(
        runner.get("schema")
        == "pareto_v21e3r1_development_matrix_post_run_audit_v1",
        "Unexpected runner audit schema.",
    )
    _require(
        runner.get("status") == "PASS_COMPLETE_DEVELOPMENT_MATRIX_AUDITED",
        "Unexpected runner audit status.",
    )
    _require(
        runner.get("matrix_aggregate_sha256") == aggregate_sha256,
        "Runner audit does not bind the matrix aggregate.",
    )
    _require(
        runner.get("authorization_receipt_sha256") == authorization_sha256
        and runner.get("source_snapshot_root_sha256") == source_root_sha256,
        "Runner audit does not bind the V3 authorization chain.",
    )
    _require(
        runner.get("expected_rows") == runner.get("observed_rows") == row_count,
        "Runner audit row count mismatch.",
    )

    _require(
        independent.get("schema")
        == "pareto_v21e3r1_independent_development_matrix_post_run_audit_v1",
        "Unexpected independent audit schema.",
    )
    _require(
        independent.get("status") == "PASS_INDEPENDENT_POST_PROCESS_RECOMPUTATION",
        "Unexpected independent audit status.",
    )
    _require(
        independent.get("matrix_aggregate_sha256") == aggregate_sha256
        and independent.get("runner_post_run_audit_sha256") == runner_sha256,
        "Independent audit does not bind the V3 matrix and runner audit.",
    )
    _require(
        independent.get("authorization_receipt_sha256") == authorization_sha256
        and independent.get("source_snapshot_root_sha256") == source_root_sha256,
        "Independent audit does not bind the V3 authorization chain.",
    )
    _require(
        independent.get("objective_archive_and_metric_replayed_rows") == row_count,
        "Independent audit did not replay all V3 rows.",
    )
    _require(
        independent.get("submission_status") == "IJOC_HOLD",
        "Independent V3 audit must retain IJOC_HOLD.",
    )
    return {
        "authorization_receipt_sha256": authorization_sha256,
        "matrix_aggregate_sha256": aggregate_sha256,
        "observed_rows": row_count,
        "runner_post_run_audit_sha256": runner_sha256,
        "status": "BOUND_BUT_INVALIDATED_WITH_V3",
    }


def _safe_zip_names(archive: zipfile.ZipFile, *, label: str) -> list[str]:
    names = [info.filename for info in archive.infolist()]
    if len(names) != len(set(names)):
        raise RuntimeError(f"{label} contains duplicate ZIP entry names.")
    for name in names:
        pure = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or pure.is_absolute()
            or ".." in pure.parts
            or name != pure.as_posix()
        ):
            raise RuntimeError(f"{label} contains unsafe ZIP entry: {name!r}")
    return names


def _verify_v3_release(
    *,
    archive_raw: bytes,
    archive_binding: Mapping[str, object],
    manifest: Mapping[str, object],
    checksum_raw: bytes,
    snapshot_binding: Mapping[str, object],
    authorization_binding: Mapping[str, object],
    source_root_sha256: str,
    release_file_count: int,
    archive_prefix: str,
    triggering_test_path: str,
    triggering_test_bytes: int,
    triggering_test_sha256: str,
) -> tuple[set[str], bytes]:
    _require(
        manifest.get("schema") == "ijoc_v21e3r1_standalone_release_manifest_v1",
        "Unexpected V3 release manifest schema.",
    )
    _require(
        manifest.get("archive_prefix") == archive_prefix,
        "Unexpected V3 release archive prefix.",
    )
    archive_record = manifest.get("archive")
    if not isinstance(archive_record, Mapping):
        raise RuntimeError("V3 release manifest archive record is missing.")
    _require(
        archive_record.get("filename") == "ijoc_v21e3r1_experiment_code.zip"
        and archive_record.get("bytes") == archive_binding["bytes"]
        and archive_record.get("sha256") == archive_binding["sha256"],
        "V3 release manifest does not bind the release archive.",
    )
    frozen = manifest.get("frozen_source_provenance")
    if not isinstance(frozen, Mapping):
        raise RuntimeError("V3 release manifest lacks frozen source provenance.")
    _require(
        frozen.get("source_snapshot_receipt_sha256") == snapshot_binding["sha256"]
        and frozen.get("source_snapshot_root_sha256") == source_root_sha256
        and frozen.get("authorization_receipt_sha256")
        == authorization_binding["sha256"],
        "V3 release manifest does not bind the V3 snapshot and authorization.",
    )
    expected_checksum = (
        f"{archive_binding['sha256']}  ijoc_v21e3r1_experiment_code.zip\n"
    ).encode("ascii")
    _require(
        checksum_raw == expected_checksum,
        "V3 release checksum file does not exactly bind the release archive.",
    )
    files = manifest.get("files")
    if not isinstance(files, list):
        raise RuntimeError("V3 release manifest files must be a list.")
    _require(
        manifest.get("file_count") == len(files) == release_file_count,
        "Unexpected V3 release file count.",
    )
    expected_outer: dict[str, Mapping[str, object]] = {}
    archive_paths: list[str] = []
    for index, item in enumerate(files):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"V3 release files[{index}] is not an object.")
        relative = str(item.get("archive_path"))
        pure = PurePosixPath(relative)
        if (
            relative != pure.as_posix()
            or pure.is_absolute()
            or ".." in pure.parts
            or "\\" in relative
        ):
            raise RuntimeError(f"Unsafe V3 release manifest path: {relative!r}")
        outer = f"{archive_prefix}/{relative}"
        if outer in expected_outer:
            raise RuntimeError(f"Duplicate V3 release manifest path: {relative}")
        _require_hex_digest(item.get("sha256"), label=f"release hash for {relative}")
        expected_outer[outer] = item
        archive_paths.append(relative)
    _require(
        archive_paths == sorted(archive_paths),
        "V3 release manifest paths are not canonically ordered.",
    )

    with zipfile.ZipFile(io.BytesIO(archive_raw), "r") as archive:
        names = _safe_zip_names(archive, label="V3 release archive")
        _require(
            set(names) == set(expected_outer),
            "V3 archive entries do not exactly match its manifest.",
        )
        triggering_test_raw = b""
        for name in names:
            raw = archive.read(name)
            record = expected_outer[name]
            _require(
                record.get("bytes") == len(raw)
                and record.get("sha256") == _sha256(raw),
                f"V3 archive member does not match its manifest: {name}",
            )
            if name == f"{archive_prefix}/{triggering_test_path}":
                triggering_test_raw = raw
    _require(
        len(triggering_test_raw) == triggering_test_bytes
        and _sha256(triggering_test_raw) == triggering_test_sha256,
        "The immutable V3 triggering test member is missing or changed.",
    )
    _require(
        b"spec_from_file_location" in triggering_test_raw
        and b"build_v21e3_code_release.py" in triggering_test_raw
        and b"verify_v21e3_clean_room.py" in triggering_test_raw,
        "The frozen V3 test does not establish both dynamic loads.",
    )
    return set(expected_outer), triggering_test_raw


def _verify_clean_room_failure(
    *,
    receipt: Mapping[str, object],
    step05_raw: bytes,
    clean_stdout_raw: bytes,
    clean_stderr_raw: bytes,
    release_binding: Mapping[str, object],
    manifest_binding: Mapping[str, object],
    dynamic_paths: Sequence[str],
) -> dict[str, object]:
    _require(
        receipt.get("schema") == "ijoc_v21e3r1_clean_room_gate_receipt_v2",
        "Unexpected V3 clean-room receipt schema.",
    )
    _require(receipt.get("status") == "FAIL", "V3 clean-room receipt is not FAIL.")
    _require(
        receipt.get("formal_authorized") is False,
        "V3 clean-room receipt must remain non-formal.",
    )
    archive_verification = receipt.get("archive_verification")
    pinned_inputs = receipt.get("pinned_inputs")
    if not isinstance(archive_verification, Mapping) or not isinstance(
        pinned_inputs, Mapping
    ):
        raise RuntimeError("V3 clean-room receipt lacks pinned archive bindings.")
    _require(
        archive_verification.get("archive_sha256") == release_binding["sha256"]
        and archive_verification.get("manifest_sha256") == manifest_binding["sha256"]
        and pinned_inputs.get("archive_sha256") == release_binding["sha256"]
        and pinned_inputs.get("manifest_sha256") == manifest_binding["sha256"],
        "V3 clean-room FAIL receipt does not bind the verified release inputs.",
    )
    error = str(receipt.get("error", ""))
    _require(
        "05_run_extracted_tree_v21_tests" in error
        and "FileNotFoundError" in error
        and "verify_v21e3_clean_room.py" in error,
        "V3 clean-room receipt does not identify the failed test step.",
    )
    try:
        step05 = step05_raw.decode("utf-8")
        clean_stderr = clean_stderr_raw.decode("utf-8")
    except UnicodeDecodeError as decode_error:
        raise RuntimeError("V3 clean-room failure logs are not UTF-8.") from decode_error
    _require(clean_stdout_raw == b"", "Unexpected V3 clean-room stdout bytes.")
    _require(
        "05_run_extracted_tree_v21_tests" in clean_stderr
        and "FileNotFoundError" in clean_stderr,
        "V3 clean-room stderr does not bind the failed step.",
    )
    normalized_step05 = step05.replace("\\\\", "\\")
    for relative in dynamic_paths:
        windows_relative = relative.replace("/", "\\")
        _require(
            windows_relative in normalized_step05
            and "FileNotFoundError" in normalized_step05,
            f"Step05 log does not expose the missing dynamic path: {relative}",
        )
    summary_match = re.search(r"7 failed, 197 passed(?: in [^\r\n]+)?", step05)
    _require(summary_match is not None, "Unexpected V3 clean-room pytest summary.")
    return {
        "failed_step": "05_run_extracted_tree_v21_tests",
        "failure_class": "UNBOUND_DYNAMIC_TEST_DEPENDENCY",
        "missing_dynamic_paths_observed": list(dynamic_paths),
        "pytest_summary": "7 failed, 197 passed",
        "receipt_status": "FAIL",
    }


def _v4_supersession(
    *,
    repo_root: Path,
    snapshot_path: Path | None,
    authorization_path: Path | None,
) -> dict[str, object]:
    if (snapshot_path is None) != (authorization_path is None):
        raise ValueError(
            "V4 snapshot and V4 authorization must be provided together."
        )
    if snapshot_path is None or authorization_path is None:
        return {"status": "PENDING_NOT_BOUND"}
    snapshot, snapshot_raw, snapshot_binding = _read_canonical_json(
        snapshot_path, label="V4 source snapshot", repo_root=repo_root
    )
    authorization, _, authorization_binding = _read_canonical_json(
        authorization_path,
        label="V4 development authorization",
        repo_root=repo_root,
    )
    _require(
        snapshot.get("schema")
        == "pareto_v21e3r1_development_source_snapshot_freeze_v1",
        "Unexpected V4 source snapshot schema.",
    )
    _require(
        snapshot.get("status") == "PASS_ENGINEERING_SNAPSHOT_ONLY"
        and snapshot.get("formal_authorized") is False
        and snapshot.get("submission_status") == "IJOC_HOLD",
        "V4 source snapshot does not retain the engineering-only boundary.",
    )
    root_digest = _require_hex_digest(
        snapshot.get("bound_files_root_sha256"), label="V4 source snapshot root"
    )
    _require(
        authorization.get("schema")
        == "pareto_v21e3r1_development_parity_authorization_v1",
        "Unexpected V4 development authorization schema.",
    )
    _require(
        authorization.get("status") == "AUTHORIZED_DEVELOPMENT_PARITY_ONLY"
        and authorization.get("formal_authorized") is False
        and authorization.get("submission_status") == "IJOC_HOLD",
        "V4 authorization does not retain the development-only boundary.",
    )
    _require(
        authorization.get("source_snapshot_receipt_sha256")
        == _sha256(snapshot_raw)
        and authorization.get("source_snapshot_root_sha256") == root_digest,
        "V4 authorization does not bind the supplied V4 snapshot.",
    )
    return {
        "development_authorization": authorization_binding,
        "source_snapshot": snapshot_binding,
        "source_snapshot_root_sha256": root_digest,
        "status": "BOUND_AUTHORIZED_SUCCESSOR",
    }


def _write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def issue_v3_invalidation(
    *,
    repo_root: Path,
    output: Path,
    v4_snapshot_path: Path | None = None,
    v4_authorization_path: Path | None = None,
    verification_contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Verify the immutable V3 failure chain and exclusively issue its receipt."""

    root = repo_root.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"Repository root is not a directory: {root}")
    destination = output.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to replace invalidation receipt: {destination}")
    supersession = _v4_supersession(
        repo_root=root,
        snapshot_path=v4_snapshot_path,
        authorization_path=v4_authorization_path,
    )
    contract = _validated_contract(verification_contract)
    artifact_paths = contract["artifact_paths"]
    artifact_hashes = contract["artifact_sha256"]
    assert isinstance(artifact_paths, Mapping)
    assert isinstance(artifact_hashes, Mapping)
    source_root_sha256 = str(contract["source_root_sha256"])
    authorization_sha256 = str(contract["authorization_sha256"])
    dynamic_paths = tuple(str(path) for path in contract["dynamic_paths"])
    dynamic_hashes = contract["dynamic_sha256"]
    assert isinstance(dynamic_hashes, Mapping)
    v3_archive_prefix = str(contract["v3_archive_prefix"])
    parent_archive_prefix = str(contract["parent_archive_prefix"])
    triggering_test_path = str(contract["triggering_test_path"])

    raw_artifacts: dict[str, bytes] = {}
    bindings: dict[str, dict[str, object]] = {}
    for name in ARTIFACT_PATHS:
        supplied_path = Path(str(artifact_paths[name]))
        candidate = supplied_path if supplied_path.is_absolute() else root / supplied_path
        resolved, raw = _read_regular_file(candidate, label=name)
        digest = _sha256(raw)
        _require(
            digest == artifact_hashes[name],
            f"Immutable V3 artifact digest mismatch for {name}: {digest}",
        )
        raw_artifacts[name] = raw
        bindings[name] = _binding(resolved, raw, root)

    parsed: dict[str, Mapping[str, object]] = {}
    for name in (
        "v3_source_snapshot",
        "v3_development_authorization",
        "v3_matrix_aggregate",
        "v3_matrix_raw_aggregate",
        "v3_runner_post_run_audit",
        "v3_matrix_raw_runner_post_run_audit",
        "v3_independent_post_run_audit",
        "v3_release_manifest",
        "v3_clean_room_fail_receipt",
    ):
        try:
            payload = json.loads(raw_artifacts[name])
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"{name} is not valid UTF-8 JSON.") from error
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"{name} must be a JSON object.")
        _require(
            _canonical_bytes(payload) == raw_artifacts[name],
            f"{name} is not canonical newline-terminated JSON.",
        )
        parsed[name] = payload

    snapshot_index = _validate_snapshot(
        parsed["v3_source_snapshot"],
        source_root_sha256=source_root_sha256,
        bound_file_count=int(contract["snapshot_bound_file_count"]),
    )
    _validate_authorization(
        parsed["v3_development_authorization"],
        snapshot_sha256=str(bindings["v3_source_snapshot"]["sha256"]),
        source_root_sha256=source_root_sha256,
    )
    _require(
        raw_artifacts["v3_matrix_aggregate"]
        == raw_artifacts["v3_matrix_raw_aggregate"],
        "Published and raw V3 matrix aggregates differ.",
    )
    _require(
        raw_artifacts["v3_runner_post_run_audit"]
        == raw_artifacts["v3_matrix_raw_runner_post_run_audit"],
        "Published and raw V3 runner audits differ.",
    )
    matrix_chain = _validate_matrix_chain(
        aggregate=parsed["v3_matrix_aggregate"],
        aggregate_sha256=str(bindings["v3_matrix_aggregate"]["sha256"]),
        runner=parsed["v3_runner_post_run_audit"],
        runner_sha256=str(bindings["v3_runner_post_run_audit"]["sha256"]),
        independent=parsed["v3_independent_post_run_audit"],
        authorization_sha256=authorization_sha256,
        source_root_sha256=source_root_sha256,
        row_count=int(contract["matrix_row_count"]),
    )
    _require(
        json.loads(raw_artifacts["v3_matrix_stdout_log"])
        == parsed["v3_runner_post_run_audit"],
        "V3 matrix stdout does not reproduce the runner audit.",
    )
    _require(
        raw_artifacts["v3_matrix_stderr_log"] == b"",
        "V3 matrix stderr is not empty.",
    )
    _require(
        json.loads(raw_artifacts["v3_independent_audit_stdout_log"])
        == parsed["v3_independent_post_run_audit"],
        "V3 independent-audit stdout does not reproduce its receipt.",
    )
    _require(
        raw_artifacts["v3_independent_audit_stderr_log"] == b"",
        "V3 independent-audit stderr is not empty.",
    )

    v3_outer_entries, triggering_test_raw = _verify_v3_release(
        archive_raw=raw_artifacts["v3_release_archive"],
        archive_binding=bindings["v3_release_archive"],
        manifest=parsed["v3_release_manifest"],
        checksum_raw=raw_artifacts["v3_release_checksum"],
        snapshot_binding=bindings["v3_source_snapshot"],
        authorization_binding=bindings["v3_development_authorization"],
        source_root_sha256=source_root_sha256,
        release_file_count=int(contract["release_file_count"]),
        archive_prefix=v3_archive_prefix,
        triggering_test_path=triggering_test_path,
        triggering_test_bytes=int(contract["triggering_test_bytes"]),
        triggering_test_sha256=str(contract["triggering_test_sha256"]),
    )
    triggering_entry = snapshot_index.get(triggering_test_path)
    if not isinstance(triggering_entry, Mapping):
        raise RuntimeError("The V3 snapshot does not bind the triggering test.")
    _require(
        triggering_entry.get("bytes") == int(contract["triggering_test_bytes"])
        and triggering_entry.get("sha256")
        == str(contract["triggering_test_sha256"]),
        "The V3 snapshot triggering-test binding is unexpected.",
    )

    clean_room_failure = _verify_clean_room_failure(
        receipt=parsed["v3_clean_room_fail_receipt"],
        step05_raw=raw_artifacts["v3_clean_room_step05_log"],
        clean_stdout_raw=raw_artifacts["v3_clean_room_stdout_log"],
        clean_stderr_raw=raw_artifacts["v3_clean_room_stderr_log"],
        release_binding=bindings["v3_release_archive"],
        manifest_binding=bindings["v3_release_manifest"],
        dynamic_paths=dynamic_paths,
    )

    parent_raw = raw_artifacts["v21e3_immutable_parent_archive"]
    dynamic_items: list[dict[str, object]] = []
    with zipfile.ZipFile(io.BytesIO(parent_raw), "r") as parent_archive:
        parent_names = set(
            _safe_zip_names(parent_archive, label="immutable V21e3 parent archive")
        )
        for relative in dynamic_paths:
            _require(
                relative not in snapshot_index,
                f"Dynamic path unexpectedly appears in the V3 snapshot: {relative}",
            )
            v3_outer = f"{v3_archive_prefix}/{relative}"
            _require(
                v3_outer not in v3_outer_entries,
                f"Dynamic path unexpectedly appears in the V3 archive: {relative}",
            )
            live_path, live_raw = _read_regular_file(
                root / relative, label=f"live dynamic dependency {relative}"
            )
            parent_member_path = f"{parent_archive_prefix}/{relative}"
            _require(
                parent_member_path in parent_names,
                f"Immutable parent lacks dynamic member: {relative}",
            )
            parent_member_raw = parent_archive.read(parent_member_path)
            live_sha256 = _sha256(live_raw)
            parent_member_sha256 = _sha256(parent_member_raw)
            expected = dynamic_hashes[relative]
            if not isinstance(expected, Mapping):
                raise RuntimeError(f"Invalid dynamic hash contract for {relative}.")
            _require(
                live_sha256 == expected["live"],
                f"Unexpected live dynamic-dependency digest: {relative}",
            )
            _require(
                parent_member_sha256 == expected["immutable_parent"],
                f"Unexpected immutable-parent member digest: {relative}",
            )
            dynamic_items.append(
                {
                    "absent_from_v3_release_archive": True,
                    "absent_from_v3_snapshot": True,
                    "immutable_parent_member": {
                        "archive_path": parent_member_path,
                        "bytes": len(parent_member_raw),
                        "sha256": parent_member_sha256,
                    },
                    "live": _binding(live_path, live_raw, root),
                    "live_vs_immutable_parent": (
                        "IDENTICAL" if live_raw == parent_member_raw else "DIFFERENT"
                    ),
                    "path": relative,
                }
            )
    _require(
        dynamic_items[0]["live_vs_immutable_parent"] == "IDENTICAL",
        "The V21e3 builder live/parent relation is not the audited relation.",
    )
    _require(
        dynamic_items[1]["live_vs_immutable_parent"] == "DIFFERENT",
        "The live verifier must differ from the immutable-parent verifier.",
    )

    receipt: dict[str, object] = {
        "clean_room_failure_audit": clean_room_failure,
        "dynamic_test_dependency_audit": {
            "parent_archive": bindings["v21e3_immutable_parent_archive"],
            "parent_verifier_differs_from_live": True,
            "required_dynamic_paths": dynamic_items,
            "status": "P0_CONFIRMED",
            "triggering_test": {
                "bytes": len(triggering_test_raw),
                "path": triggering_test_path,
                "sha256": _sha256(triggering_test_raw),
                "v3_release_archive_path": (
                    f"{v3_archive_prefix}/{triggering_test_path}"
                ),
                "v3_snapshot_bound": True,
            },
        },
        "evidence_artifacts": bindings,
        "formal_authorized": False,
        "invalidation_findings": [
            {
                "code": "P0_UNBOUND_DYNAMIC_TEST_DEPENDENCY",
                "status": "CONFIRMED",
            },
            {
                "code": "P0_FAILED_CLEAN_ROOM",
                "status": "CONFIRMED",
            },
        ],
        "matrix_chain_audit": matrix_chain,
        "schema": SCHEMA,
        "status": INVALIDATION_STATUS,
        "submission_status": "IJOC_HOLD",
        "v3_reuse_for_v4": "PROHIBITED",
        "v3_value_status": V3_VALUE_STATUS,
        "v4_supersession": supersession,
    }
    _write_exclusive(destination, _canonical_bytes(receipt))
    return receipt


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Issue the canonical V21e3r1 V3 invalidation receipt."
    )
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v4-source-snapshot", type=Path)
    parser.add_argument("--v4-development-authorization", type=Path)
    args = parser.parse_args(argv)
    receipt = issue_v3_invalidation(
        repo_root=args.repo_root,
        output=args.output,
        v4_snapshot_path=args.v4_source_snapshot,
        v4_authorization_path=args.v4_development_authorization,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
