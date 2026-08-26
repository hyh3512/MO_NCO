"""Build and verify the public GitHub export manifest from the Git index.

Only stage-0 regular-file entries in the index are included.  Working-tree
changes and untracked files are intentionally ignored.  The manifest excludes
itself so rebuilding it cannot create a recursive content identity.

This is an engineering export inventory, not a scientific authorization or a
claim that the repository-wide test suite is green.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Mapping, Sequence
import uuid


MANIFEST_NAME = "GITHUB_EXPORT_CONTENTS.json"
MANIFEST_SCHEMA = "mo_nco_github_public_export_contents_v2"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_REGULAR_MODES = {"100644", "100755"}

_IDENTITY = {
    "distribution": "mo-nco",
    "revision": "V21E3R1_V9R2R1",
    "version": "0.21.3.14",
}
_REPOSITORY = {
    "name": "MO_NCO",
    "owner": "hyh3512",
    "public_repository": True,
    "target_branch": "main",
    "visibility": "public",
}
_SCOPE = {
    "confirmation_authorized": False,
    "engineering_only": True,
    "environment_lock": False,
    "formal_authorized": False,
    "full_development_matrix_authorized": False,
    "ijoc_submission_authorized": False,
    "repository_wide_green": False,
    "scientific_independence": False,
    "selection_authorized": False,
    "status": "PUBLIC_ENGINEERING_EXPORT_ONLY",
}
_EXCLUSIONS = {
    "large_artifacts_excluded": True,
    "large_artifact_policy": (
        "large traces, databases, wheels, source archives, caches, and "
        "experiment result warehouses are outside this Git export"
    ),
    "v8_tag_redistribution_authorized": False,
    "v8_tag_redistribution_status": "BLOCKED",
}


class ExportManifestError(ValueError):
    """Raised when the index or export manifest violates its strict contract."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ExportManifestError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ExportManifestError(f"non-finite JSON value prohibited: {value}")


def _safe_index_path(value: object) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ExportManifestError("index paths must be nonempty POSIX strings")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ExportManifestError(f"control character prohibited in index path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ExportManifestError(f"unsafe or noncanonical index path: {value!r}")
    return value


def _git(root: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace").strip()
        raise ExportManifestError(
            f"git {' '.join(arguments)} failed: {detail or error.returncode}"
        ) from error
    return completed.stdout


def _repository_root(root: Path) -> Path:
    requested = root.resolve(strict=True)
    try:
        raw = _git(requested, "rev-parse", "--show-toplevel")
        discovered = Path(raw.decode("utf-8").strip()).resolve(strict=True)
    except UnicodeDecodeError as error:
        raise ExportManifestError("Git repository root is not UTF-8") from error
    if os.path.normcase(str(requested)) != os.path.normcase(str(discovered)):
        raise ExportManifestError(
            f"root must be the exact Git top level: {requested} != {discovered}"
        )
    return requested


def _parse_index_entries(raw: bytes) -> list[dict[str, str]]:
    parsed: list[dict[str, str]] = []
    names: set[str] = set()
    folded: set[str] = set()
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, encoded_path = record.split(b"\t", 1)
            mode, oid, stage = metadata.decode("ascii").split(" ")
            name = _safe_index_path(encoded_path.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise ExportManifestError("malformed or non-UTF-8 Git index entry") from error
        if stage != "0":
            raise ExportManifestError(f"unmerged Git index entry prohibited: {name}")
        if mode not in _REGULAR_MODES:
            kind = "symlink" if mode == "120000" else "non-regular"
            raise ExportManifestError(
                f"{kind} Git index entry prohibited: {name} (mode {mode})"
            )
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", oid):
            raise ExportManifestError(f"invalid Git object id for {name}")
        if name.casefold() == MANIFEST_NAME.casefold():
            if name != MANIFEST_NAME:
                raise ExportManifestError(
                    f"manifest path has noncanonical casing: {name!r}"
                )
            continue
        if name in names or name.casefold() in folded:
            raise ExportManifestError(f"duplicate or case-colliding index path: {name}")
        names.add(name)
        folded.add(name.casefold())
        parsed.append({"mode": mode, "oid": oid, "path": name})
    parsed.sort(key=lambda item: item["path"])
    return parsed


def _index_snapshot(root: Path) -> tuple[bytes, list[dict[str, str]]]:
    raw = _git(root, "ls-files", "--stage", "-z")
    return raw, _parse_index_entries(raw)


def build_payload(root: Path) -> dict[str, object]:
    """Return a deterministic manifest payload for the current Git index."""

    root = _repository_root(root)
    before, index_entries = _index_snapshot(root)
    files: list[dict[str, object]] = []
    for item in index_entries:
        raw = _git(root, "cat-file", "blob", item["oid"])
        files.append(
            {
                "bytes": len(raw),
                "path": item["path"],
                "sha256": _sha256(raw),
            }
        )
    after = _git(root, "ls-files", "--stage", "-z")
    if after != before:
        raise ExportManifestError("Git index changed while manifest was being built")

    core: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "identity": dict(_IDENTITY),
        "repository": dict(_REPOSITORY),
        "scope": dict(_SCOPE),
        "exclusions": dict(_EXCLUSIONS),
        "index_source": {
            "included_entries": "stage-0 regular-file Git index blobs only",
            "manifest_excluded": MANIFEST_NAME,
            "untracked_files_included": False,
            "working_tree_bytes_used": False,
        },
        "file_count": len(files),
        "files": files,
        "files_root_sha256": _sha256(_canonical_json(files)),
    }
    return {**core, "manifest_payload_sha256": _sha256(_canonical_json(core))}


def render_manifest(root: Path) -> bytes:
    return _canonical_json(build_payload(root)) + b"\n"


def _strict_manifest(raw: bytes) -> dict[str, object]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExportManifestError("manifest must be strict UTF-8 JSON") from error
    if type(payload) is not dict:
        raise ExportManifestError("manifest must be an exact JSON object")
    if raw != _canonical_json(payload) + b"\n":
        raise ExportManifestError(
            "manifest must be canonical JSON followed by exactly one newline"
        )
    return payload


def _validate_payload(payload: Mapping[str, object]) -> None:
    required_keys = {
        "schema",
        "identity",
        "repository",
        "scope",
        "exclusions",
        "index_source",
        "file_count",
        "files",
        "files_root_sha256",
        "manifest_payload_sha256",
    }
    if set(payload) != required_keys:
        raise ExportManifestError("manifest top-level fields drifted")
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ExportManifestError("manifest schema drifted")
    if _canonical_json(payload.get("identity")) != _canonical_json(_IDENTITY):
        raise ExportManifestError("distribution identity drifted")
    if _canonical_json(payload.get("repository")) != _canonical_json(_REPOSITORY):
        raise ExportManifestError("public repository metadata drifted")
    if _canonical_json(payload.get("scope")) != _canonical_json(_SCOPE):
        raise ExportManifestError("engineering or authorization boundary drifted")
    if _canonical_json(payload.get("exclusions")) != _canonical_json(_EXCLUSIONS):
        raise ExportManifestError("export exclusion boundary drifted")
    expected_index_source = {
        "included_entries": "stage-0 regular-file Git index blobs only",
        "manifest_excluded": MANIFEST_NAME,
        "untracked_files_included": False,
        "working_tree_bytes_used": False,
    }
    if _canonical_json(payload.get("index_source")) != _canonical_json(
        expected_index_source
    ):
        raise ExportManifestError("Git index source contract drifted")

    files = payload.get("files")
    if (
        type(files) is not list
        or type(payload.get("file_count")) is not int
        or payload.get("file_count") != len(files)
    ):
        raise ExportManifestError("file_count does not match files")
    names: list[str] = []
    folded: set[str] = set()
    for index, entry in enumerate(files):
        if type(entry) is not dict or set(entry) != {"path", "bytes", "sha256"}:
            raise ExportManifestError(f"invalid file entry {index}")
        name = _safe_index_path(entry["path"])
        if name.casefold() == MANIFEST_NAME.casefold():
            raise ExportManifestError("manifest must exclude itself")
        if name.casefold() in folded:
            raise ExportManifestError(f"duplicate or case-colliding path: {name}")
        folded.add(name.casefold())
        if type(entry["bytes"]) is not int or entry["bytes"] < 0:
            raise ExportManifestError(f"invalid byte count for {name}")
        if type(entry["sha256"]) is not str or not _SHA256_RE.fullmatch(
            entry["sha256"]
        ):
            raise ExportManifestError(f"invalid SHA-256 for {name}")
        names.append(name)
    if names != sorted(names) or len(names) != len(set(names)):
        raise ExportManifestError("manifest paths must be sorted and unique")
    if payload.get("files_root_sha256") != _sha256(_canonical_json(files)):
        raise ExportManifestError("files_root_sha256 mismatch")

    declared = payload.get("manifest_payload_sha256")
    if type(declared) is not str or not _SHA256_RE.fullmatch(declared):
        raise ExportManifestError("invalid manifest_payload_sha256")
    core = dict(payload)
    del core["manifest_payload_sha256"]
    if declared != _sha256(_canonical_json(core)):
        raise ExportManifestError("manifest_payload_sha256 mismatch")


def verify_manifest(root: Path, manifest_path: Path) -> dict[str, object]:
    raw = manifest_path.resolve(strict=True).read_bytes()
    payload = _strict_manifest(raw)
    _validate_payload(payload)
    expected = build_payload(root)
    if payload != expected:
        raise ExportManifestError("manifest differs from the current Git index")
    return {
        "schema": "mo_nco_github_public_export_verification_v1",
        "status": "PASS_PUBLIC_ENGINEERING_EXPORT_INDEX_IDENTITY",
        "file_count": payload["file_count"],
        "files_root_sha256": payload["files_root_sha256"],
        "manifest_sha256": _sha256(raw),
        "repository_wide_green": False,
        "environment_lock": False,
        "scientific_independence": False,
    }


def write_manifest(
    root: Path,
    output_path: Path,
    *,
    replace: bool = False,
) -> dict[str, object]:
    raw = render_manifest(root)
    output = output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not replace:
        raise ExportManifestError(f"refusing to overwrite without --replace: {output}")
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if not replace and output.exists():
            raise ExportManifestError(f"refusing to overwrite without --replace: {output}")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "schema": "mo_nco_github_public_export_build_receipt_v1",
        "status": "PASS_PUBLIC_ENGINEERING_EXPORT_MANIFEST_BUILT",
        "manifest": str(output),
        "manifest_bytes": len(raw),
        "manifest_sha256": _sha256(raw),
        "file_count": json.loads(raw)["file_count"],
        "repository_wide_green": False,
        "environment_lock": False,
        "scientific_independence": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--replace", action="store_true")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)

    render = subparsers.add_parser("render")
    render.add_argument("--root", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            receipt = write_manifest(args.root, args.output, replace=args.replace)
            print(_canonical_json(receipt).decode("utf-8"))
        elif args.command == "verify":
            receipt = verify_manifest(args.root, args.manifest)
            print(_canonical_json(receipt).decode("utf-8"))
        else:
            sys.stdout.buffer.write(render_manifest(args.root))
        return 0
    except (ExportManifestError, OSError) as error:
        failure = {
            "schema": "mo_nco_github_public_export_error_v1",
            "status": "HOLD_EXPORT_MANIFEST",
            "error": str(error),
            "repository_wide_green": False,
            "environment_lock": False,
            "scientific_independence": False,
        }
        print(_canonical_json(failure).decode("utf-8"), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ExportManifestError",
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "build_payload",
    "main",
    "render_manifest",
    "verify_manifest",
    "write_manifest",
]
