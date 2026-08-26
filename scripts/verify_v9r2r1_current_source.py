"""Verify the checked-out V9R2R1 source closure against its canonical manifest.

This verifier checks the manifest self-hash, every declared path, byte count,
SHA-256 digest, and (optionally) Git tracking.  It intentionally does not
authorize a scientific stage or reinterpret the historical V8 identity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


def _load_bundle_module(root: Path):
    path = root / "scripts" / "build_v9r2r1_engineering_source_bundle.py"
    spec = importlib.util.spec_from_file_location("v9r2r1_source_bundle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source-manifest implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_tracked(root: Path) -> set[str]:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Git path inventory unavailable: {detail}")
    return {
        item.decode("utf-8", errors="strict")
        for item in completed.stdout.split(b"\0")
        if item
    }


def verify(root: Path, manifest_path: Path, *, require_git_tracked: bool) -> dict[str, object]:
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    module = _load_bundle_module(root)
    payload = module._strict_manifest(manifest_path.read_bytes())
    entries = module._validate_manifest_payload(payload)
    tracked = _git_tracked(root) if require_git_tracked else set()

    missing: list[str] = []
    nonregular_or_symlink: list[str] = []
    byte_mismatches: list[dict[str, object]] = []
    sha256_mismatches: list[dict[str, str]] = []
    untracked: list[str] = []

    for entry in entries:
        relative = str(entry["path"])
        path = root / relative
        if not path.exists():
            missing.append(relative)
            continue
        if not path.is_file() or path.is_symlink():
            nonregular_or_symlink.append(relative)
            continue
        raw = path.read_bytes()
        if len(raw) != entry["bytes"]:
            byte_mismatches.append(
                {"path": relative, "expected": entry["bytes"], "actual": len(raw)}
            )
        actual = hashlib.sha256(raw).hexdigest()
        if actual != entry["sha256"]:
            sha256_mismatches.append(
                {"path": relative, "expected": str(entry["sha256"]), "actual": actual}
            )
        if require_git_tracked and relative not in tracked:
            untracked.append(relative)

    passed = not any(
        (missing, nonregular_or_symlink, byte_mismatches, sha256_mismatches, untracked)
    )
    return {
        "schema": "v21e3r1_v9r2r1_checked_out_source_verification_v1",
        "status": "PASS_CURRENT_SOURCE_MANIFEST" if passed else "FAIL_CURRENT_SOURCE_MANIFEST",
        "passed": passed,
        "declared_file_count": len(entries),
        "declared_source_tree_sha256": payload["source_tree_sha256"],
        "manifest_payload_sha256": payload["manifest_payload_sha256"],
        "require_git_tracked": require_git_tracked,
        "missing": missing,
        "nonregular_or_symlink": nonregular_or_symlink,
        "byte_mismatches": byte_mismatches,
        "sha256_mismatches": sha256_mismatches,
        "untracked": untracked,
        "repository_wide_green": False,
        "scientific_stage_authorized": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_authorized": False,
        "ijoc_submission_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("V21E3R1_V9R2R1_SOURCE_MANIFEST.json"),
    )
    parser.add_argument("--require-git-tracked", action="store_true")
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Accepted for explicit fail-closed automation; drift always returns nonzero.",
    )
    args = parser.parse_args(argv)
    try:
        result = verify(
            args.root,
            args.manifest,
            require_git_tracked=args.require_git_tracked,
        )
    except Exception as error:
        result = {
            "schema": "v21e3r1_v9r2r1_checked_out_source_verification_v1",
            "status": "HOLD_CURRENT_SOURCE_UNVERIFIABLE",
            "passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "repository_wide_green": False,
            "scientific_stage_authorized": False,
        }
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
