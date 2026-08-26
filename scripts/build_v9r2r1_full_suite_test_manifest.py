"""Build or verify the current public-checkout full-suite test source closure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Sequence
import xml.etree.ElementTree as ET


SCHEMA = "v21e3r1_v9r2r1_full_suite_test_source_manifest_v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def safe_path(value: object) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError("paths must be nonempty POSIX strings")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        raise ValueError(f"unsafe path: {value!r}")
    return value


def modules_from_tree(root: Path) -> list[str]:
    tests = root / "tests"
    if not tests.is_dir():
        raise ValueError("tests directory is missing")
    modules = []
    for path in tests.glob("test_*.py"):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"test module must be a regular file: {path}")
        modules.append(safe_path(path.relative_to(root).as_posix()))
    if not modules:
        raise ValueError("no current test modules found")
    return sorted(modules)


def build(root: Path, junit: Path) -> dict[str, object]:
    root = root.resolve()
    junit = junit.resolve()
    entries = []
    # Parse the historical reference as a strict well-formedness check, but do
    # not use it to hide public-checkout validator tests added after that run.
    ET.fromstring(junit.read_bytes())
    for relative in modules_from_tree(root):
        raw = (root / relative).read_bytes()
        entries.append({"bytes": len(raw), "path": relative, "sha256": sha256(raw)})
    core: dict[str, object] = {
        "schema": SCHEMA,
        "status": "ENGINEERING_TEST_SOURCE_CLOSURE_CANDIDATE",
        "test_module_selection": "ALL_CURRENT_TESTS_TEST_STAR_PY",
        "recorded_junit_path": (
            "evidence/v9r2r1_environment_recovery_20260825_002/"
            "full_repository.junit.xml"
        ),
        "recorded_junit_sha256": sha256(junit.read_bytes()),
        "test_module_count": len(entries),
        "files": entries,
        "files_root_sha256": sha256(canonical_json(entries)),
        "recorded_result_reexecution_performed": False,
        "repository_wide_green": False,
        "scientific_stage_authorized": False,
    }
    return {**core, "manifest_payload_sha256": sha256(canonical_json(core))}


def strict_json(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    pairs_seen: list[str] = []

    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
            pairs_seen.append(key)
        return result

    payload = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    if type(payload) is not dict or raw != canonical_json(payload) + b"\n":
        raise ValueError("manifest must be canonical UTF-8 JSON plus one newline")
    return payload


def verify(root: Path, junit: Path, manifest: Path) -> dict[str, object]:
    declared = strict_json(manifest)
    if declared.get("schema") != SCHEMA:
        raise ValueError("manifest schema drifted")
    expected = build(root, junit)
    if declared != expected:
        raise ValueError("test source manifest or recorded JUnit binding drifted")
    return {
        "schema": "v21e3r1_v9r2r1_full_suite_test_source_verification_v1",
        "status": "PASS_FULL_SUITE_TEST_SOURCE_CLOSURE",
        "passed": True,
        "test_module_count": expected["test_module_count"],
        "files_root_sha256": expected["files_root_sha256"],
        "repository_wide_green": False,
        "scientific_stage_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    verify_parser = subparsers.add_parser("verify")
    for item in (build_parser, verify_parser):
        item.add_argument("--root", type=Path, default=Path("."))
        item.add_argument(
            "--junit",
            type=Path,
            default=Path(
                "evidence/v9r2r1_environment_recovery_20260825_002/"
                "full_repository.junit.xml"
            ),
        )
    verify_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("provenance/V9R2R1_FULL_SUITE_TEST_MANIFEST.json"),
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            payload = build(args.root, args.junit)
        else:
            payload = verify(args.root, args.junit, args.manifest)
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "HOLD_FULL_SUITE_TEST_SOURCE_UNVERIFIABLE",
                    "passed": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "repository_wide_green": False,
                    "scientific_stage_authorized": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(canonical_json(payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
