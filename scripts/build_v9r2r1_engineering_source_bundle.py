"""Build and verify the deterministic V21E3R1 V9R2R1 source bundle.

V9R2R1 is an engineering-only packaging revision of V9R2.  The archive closes
the live V9 Python package plus the integration writer that was repaired after
the 0.21.3.13 wheel had been frozen.  It is a source-freeze *candidate* only:
building or verifying it does not satisfy the current protocol, authorize the
full development matrix, or establish implementation/scientific independence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Mapping, Sequence
import zipfile


VERSION = "0.21.3.14"
REVISION = "V21E3R1_V9R2R1"
MANIFEST_NAME = f"{REVISION}_SOURCE_MANIFEST.json"
ARCHIVE_NAME = f"mo_nco-{VERSION}-v9r2r1-source.zip"
MANIFEST_SCHEMA = "v21e3r1_v9r2r1_engineering_source_manifest_v1"
_FIXED_ZIP_TIME = (2023, 11, 14, 22, 13, 20)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

_TARGETED_TESTS = (
    "tests/test_pareto_v21e3r1_v9_information_search.py",
    "tests/test_pareto_v21e3r1_v9_strict_regressions.py",
    "tests/test_pareto_v21e3r1_v9r1_theory_strict.py",
    "tests/test_pareto_v21e3r1_v9_diagnostics_strict.py",
    "tests/test_pareto_v21e3r1_branch_replay.py",
    "tests/test_pareto_v21e3r1_v9_runner.py",
    "tests/test_pareto_v21e3_trace.py",
    "tests/test_pareto_v21e3_trace_verify.py",
    "tests/test_pareto_v21e3_trace_chunks.py",
    "tests/test_pareto_v21e3r1_v9_protocol.py",
    "tests/test_pareto_v21e3r1_v9_gate.py",
    "tests/test_pareto_v21e3r1_v9_packaging.py",
    "tests/test_build_v9r2r1_engineering_source_bundle.py",
    "tests/test_check_v9r2r1_full_suite_environment.py",
    "tests/test_pareto_v21e3r1_development_diagnostic_runner.py",
    "tests/test_v21e3r1_same_implementation_branch_replay_coverage.py",
)
_INTEGRATION_SOURCES = (
    "ijoc_submission_v21e3r1/scripts/run_v21e3r1_development_diagnostics.py",
    (
        "ijoc_submission_v21e3r1/scripts/"
        "run_v21e3r1_same_implementation_branch_replay_coverage.py"
    ),
)
_SUPPORT_FILES = (
    "pyproject.toml",
    "docs/V21E3R1_V9R1_THEORY.md",
    "docs/V21E3R1_V9R2_ALGORITHM_SPEC.md",
    "docs/V21E3R1_V9R2_ENGINEERING_CLOSURE.md",
    "docs/V21E3R1_V9R2_RUNBOOK.md",
    "docs/V21E3R1_V9R2_TRACE_REPLAY_SPEC.md",
    "docs/V21E3R1_V9R2R1_STRICT_EVALUATION.md",
    "docs/V21E3R1_V9R2R1_RUNBOOK.md",
    "scripts/build_v9r2_engineering_source_bundle.py",
    "scripts/build_v9r2r1_engineering_source_bundle.py",
    "scripts/check_v9r2r1_full_suite_environment.py",
)


class SourceBundleError(ValueError):
    """Raised when the source-freeze candidate is incomplete or drifts."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SourceBundleError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise SourceBundleError(f"non-finite JSON value prohibited: {value}")


def _strict_manifest(raw: bytes) -> dict[str, object]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceBundleError("manifest must be strict UTF-8 JSON") from error
    if type(payload) is not dict:
        raise SourceBundleError("manifest must be an exact JSON object")
    if raw != _canonical_json(payload) + b"\n":
        raise SourceBundleError(
            "manifest must be canonical JSON followed by exactly one newline"
        )
    return payload


def _safe_archive_name(value: object) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise SourceBundleError("archive paths must be nonempty POSIX strings")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or ".." in path.parts:
        raise SourceBundleError(f"unsafe archive path: {value!r}")
    return value


def _source_paths(root: Path) -> list[Path]:
    package_root = root / "mo_nco"
    if not package_root.is_dir():
        raise SourceBundleError(f"missing package directory: {package_root}")
    relative = {
        path.relative_to(root)
        for path in package_root.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    relative.update(
        path.relative_to(root)
        for path in (package_root / "specs").glob("*.json")
    )
    relative.update(Path(item) for item in _TARGETED_TESTS)
    relative.update(Path(item) for item in _INTEGRATION_SOURCES)
    relative.update(Path(item) for item in _SUPPORT_FILES)
    paths = sorted(relative, key=lambda path: path.as_posix())
    folded: set[str] = set()
    for path in paths:
        name = _safe_archive_name(path.as_posix())
        folded_name = name.casefold()
        if folded_name in folded:
            raise SourceBundleError(f"case-colliding source path: {name}")
        folded.add(folded_name)
        absolute = root / path
        if not absolute.is_file() or absolute.is_symlink():
            raise SourceBundleError(f"required regular source file missing: {name}")
    return paths


def _write_exclusive(path: Path, raw: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise SourceBundleError(f"refusing to overwrite: {path}") from error


def _manifest_payload(entries: list[dict[str, object]]) -> dict[str, object]:
    core: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "identity": {
            "distribution": "mo-nco",
            "version": VERSION,
            "revision": REVISION,
            "algorithm_predecessor": "V21E3R1_V9R2",
            "algorithm_semantics_changed": False,
        },
        "scope": (
            "all_mo_nco_python_sources_package_specs_v9_tests_docs_and_"
            "same_implementation_coverage_integration_sources"
        ),
        "status": "ENGINEERING_SOURCE_FREEZE_CANDIDATE_PRE_DEVELOPMENT_HOLD",
        "file_count": len(entries),
        "files": entries,
        "source_tree_sha256": _sha256(_canonical_json(entries)),
        "includes_post_v9r2_branch_coverage_writer_fix": True,
        "bound_into_authorizing_protocol": False,
        "full_source_freeze_requirement_satisfied": False,
        "scientific_independence": False,
        "full_development_matrix_authorized": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_authorized": False,
        "ijoc_submission_authorized": False,
    }
    return {**core, "manifest_payload_sha256": _sha256(_canonical_json(core))}


def _validate_manifest_payload(payload: Mapping[str, object]) -> list[dict[str, object]]:
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise SourceBundleError("unexpected source manifest schema")
    identity = payload.get("identity")
    if type(identity) is not dict or identity != {
        "distribution": "mo-nco",
        "version": VERSION,
        "revision": REVISION,
        "algorithm_predecessor": "V21E3R1_V9R2",
        "algorithm_semantics_changed": False,
    }:
        raise SourceBundleError("source manifest identity drifted")
    declared_payload_hash = payload.get("manifest_payload_sha256")
    if type(declared_payload_hash) is not str or not _SHA256_RE.fullmatch(
        declared_payload_hash
    ):
        raise SourceBundleError("invalid manifest_payload_sha256")
    core = dict(payload)
    del core["manifest_payload_sha256"]
    if declared_payload_hash != _sha256(_canonical_json(core)):
        raise SourceBundleError("manifest payload self-hash mismatch")
    for field in (
        "includes_post_v9r2_branch_coverage_writer_fix",
        "bound_into_authorizing_protocol",
        "full_source_freeze_requirement_satisfied",
        "scientific_independence",
        "full_development_matrix_authorized",
        "selection_authorized",
        "confirmation_authorized",
        "formal_authorized",
        "ijoc_submission_authorized",
    ):
        expected = field == "includes_post_v9r2_branch_coverage_writer_fix"
        if payload.get(field) is not expected:
            raise SourceBundleError(f"authorization boundary drifted: {field}")
    entries = payload.get("files")
    if type(entries) is not list or payload.get("file_count") != len(entries):
        raise SourceBundleError("manifest file_count does not match files")
    validated: list[dict[str, object]] = []
    names: list[str] = []
    folded: set[str] = set()
    for index, entry in enumerate(entries):
        if type(entry) is not dict or set(entry) != {"path", "bytes", "sha256"}:
            raise SourceBundleError(f"invalid manifest entry {index}")
        name = _safe_archive_name(entry["path"])
        if name.casefold() in folded:
            raise SourceBundleError(f"duplicate/case-colliding manifest path: {name}")
        folded.add(name.casefold())
        if type(entry["bytes"]) is not int or entry["bytes"] < 0:
            raise SourceBundleError(f"invalid byte count for {name}")
        if type(entry["sha256"]) is not str or not _SHA256_RE.fullmatch(
            entry["sha256"]
        ):
            raise SourceBundleError(f"invalid SHA-256 for {name}")
        names.append(name)
        validated.append(dict(entry))
    if names != sorted(names) or len(names) != len(set(names)):
        raise SourceBundleError("manifest paths must be unique and sorted")
    if payload.get("source_tree_sha256") != _sha256(_canonical_json(validated)):
        raise SourceBundleError("source_tree_sha256 mismatch")
    for required in _INTEGRATION_SOURCES:
        if required not in names:
            raise SourceBundleError(f"integration source absent from closure: {required}")
    for required in (
        "tests/test_v21e3r1_same_implementation_branch_replay_coverage.py",
        "tests/test_pareto_v21e3r1_v9_packaging.py",
    ):
        if required not in names:
            raise SourceBundleError(f"required regression absent from closure: {required}")
    return validated


def verify_bundle(
    manifest_path: Path,
    archive_path: Path,
    *,
    root: Path | None = None,
) -> dict[str, object]:
    """Verify canonical manifest, ZIP closure, bytes, metadata, and live tree."""

    manifest_path = manifest_path.resolve(strict=True)
    archive_path = archive_path.resolve(strict=True)
    manifest_raw = manifest_path.read_bytes()
    payload = _strict_manifest(manifest_raw)
    entries = _validate_manifest_payload(payload)
    expected = {entry["path"]: entry for entry in entries}
    expected_names = [*sorted(expected), MANIFEST_NAME]

    with zipfile.ZipFile(archive_path, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if names != expected_names or len(names) != len(set(names)):
            raise SourceBundleError("ZIP member set/order differs from manifest closure")
        for info in infos:
            name = _safe_archive_name(info.filename)
            if info.is_dir():
                raise SourceBundleError(f"directory entry prohibited: {name}")
            if info.date_time != _FIXED_ZIP_TIME:
                raise SourceBundleError(f"non-deterministic ZIP timestamp: {name}")
            if info.compress_type != zipfile.ZIP_DEFLATED:
                raise SourceBundleError(f"unexpected ZIP compression: {name}")
            if info.create_system != 3 or (info.external_attr >> 16) != 0o100644:
                raise SourceBundleError(f"unexpected ZIP mode metadata: {name}")
            raw = archive.read(info)
            if name == MANIFEST_NAME:
                if raw != manifest_raw:
                    raise SourceBundleError("embedded manifest differs from detached manifest")
                continue
            entry = expected[name]
            if len(raw) != entry["bytes"] or _sha256(raw) != entry["sha256"]:
                raise SourceBundleError(f"ZIP member bytes/hash mismatch: {name}")

    live_tree_verified = root is not None
    if root is not None:
        resolved_root = root.resolve(strict=True)
        live_paths = [path.as_posix() for path in _source_paths(resolved_root)]
        if live_paths != sorted(expected):
            raise SourceBundleError("live source set differs from manifest source set")
        for name, entry in expected.items():
            path = resolved_root.joinpath(*PurePosixPath(name).parts)
            raw = path.read_bytes()
            if len(raw) != entry["bytes"] or _sha256(raw) != entry["sha256"]:
                raise SourceBundleError(f"live source bytes/hash mismatch: {name}")

    return {
        "schema": "v21e3r1_v9r2r1_engineering_source_verification_v1",
        "status": "PASS_ENGINEERING_SOURCE_FREEZE_CANDIDATE_ONLY",
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256(archive_path.read_bytes()),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_raw),
        "manifest_payload_sha256": payload["manifest_payload_sha256"],
        "source_tree_sha256": payload["source_tree_sha256"],
        "file_count": len(entries),
        "zip_member_count": len(expected_names),
        "live_tree_verified": live_tree_verified,
        "bound_into_authorizing_protocol": False,
        "full_source_freeze_requirement_satisfied": False,
        "full_development_matrix_authorized": False,
    }


def build_bundle(root: Path, output_directory: Path) -> dict[str, object]:
    root = root.resolve(strict=True)
    output_directory = output_directory.resolve(strict=True)
    if not (root / "pyproject.toml").is_file():
        raise SourceBundleError(f"not a project root: {root}")
    if not output_directory.is_dir():
        raise SourceBundleError(
            f"output directory must already exist: {output_directory}"
        )
    if any(output_directory.iterdir()):
        raise SourceBundleError(
            f"output directory must be empty: {output_directory}"
        )

    manifest_path = output_directory / MANIFEST_NAME
    archive_path = output_directory / ARCHIVE_NAME
    paths = _source_paths(root)
    entries: list[dict[str, object]] = []
    content: dict[str, bytes] = {}
    for relative in paths:
        name = relative.as_posix()
        raw = (root / relative).read_bytes()
        content[name] = raw
        entries.append({"path": name, "bytes": len(raw), "sha256": _sha256(raw)})

    payload = _manifest_payload(entries)
    manifest_raw = _canonical_json(payload) + b"\n"
    _write_exclusive(manifest_path, manifest_raw)

    try:
        with archive_path.open("xb") as raw_archive:
            with zipfile.ZipFile(
                raw_archive,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            ) as archive:
                for name in sorted(content):
                    info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_TIME)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.create_system = 3
                    info.external_attr = 0o100644 << 16
                    archive.writestr(info, content[name], compresslevel=9)
                info = zipfile.ZipInfo(MANIFEST_NAME, date_time=_FIXED_ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, manifest_raw, compresslevel=9)
            raw_archive.flush()
            os.fsync(raw_archive.fileno())
    except FileExistsError as error:
        raise SourceBundleError(f"refusing to overwrite: {archive_path}") from error

    verification = verify_bundle(manifest_path, archive_path, root=root)
    return {
        "schema": "v21e3r1_v9r2r1_engineering_source_build_receipt_v1",
        "status": "PASS_ENGINEERING_SOURCE_FREEZE_CANDIDATE_ONLY",
        "verification": verification,
        "full_source_freeze_requirement_satisfied": False,
        "full_development_matrix_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--output-directory", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        receipt = build_bundle(args.root, args.output_directory)
    else:
        receipt = verify_bundle(args.manifest, args.archive, root=args.root)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHIVE_NAME",
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "REVISION",
    "SourceBundleError",
    "VERSION",
    "build_bundle",
    "main",
    "verify_bundle",
]
