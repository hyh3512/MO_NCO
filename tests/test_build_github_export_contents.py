from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_github_export_contents.py"
SPEC = importlib.util.spec_from_file_location("build_github_export_contents", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def _git(root: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    return root


def _write(root: Path, relative: str, raw: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def test_render_is_canonical_deterministic_and_uses_only_staged_blobs(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    staged_alpha = b"staged alpha\n"
    staged_binary = bytes(range(32))
    _write(root, "alpha.txt", staged_alpha)
    _write(root, "nested/data.bin", staged_binary)
    _write(root, BUILDER.MANIFEST_NAME, b"old recursive manifest")
    _git(root, "add", "alpha.txt", "nested/data.bin", BUILDER.MANIFEST_NAME)

    _write(root, "alpha.txt", b"unstaged replacement\n")
    _write(root, "untracked.txt", b"must not be exported\n")

    raw_a = BUILDER.render_manifest(root)
    raw_b = BUILDER.render_manifest(root)
    assert raw_a == raw_b
    payload = json.loads(raw_a)
    assert raw_a == BUILDER._canonical_json(payload) + b"\n"
    assert payload["file_count"] == 2
    assert [entry["path"] for entry in payload["files"]] == [
        "alpha.txt",
        "nested/data.bin",
    ]
    assert payload["files"][0] == {
        "bytes": len(staged_alpha),
        "path": "alpha.txt",
        "sha256": hashlib.sha256(staged_alpha).hexdigest(),
    }
    assert payload["files"][1]["sha256"] == hashlib.sha256(
        staged_binary
    ).hexdigest()
    assert payload["files_root_sha256"] == hashlib.sha256(
        BUILDER._canonical_json(payload["files"])
    ).hexdigest()
    core = dict(payload)
    declared = core.pop("manifest_payload_sha256")
    assert declared == hashlib.sha256(BUILDER._canonical_json(core)).hexdigest()

    assert payload["repository"]["visibility"] == "public"
    assert payload["repository"]["public_repository"] is True
    assert payload["scope"]["engineering_only"] is True
    for field in (
        "repository_wide_green",
        "environment_lock",
        "scientific_independence",
        "full_development_matrix_authorized",
        "selection_authorized",
        "confirmation_authorized",
        "formal_authorized",
        "ijoc_submission_authorized",
    ):
        assert payload["scope"][field] is False
    assert payload["exclusions"]["v8_tag_redistribution_status"] == "BLOCKED"
    assert payload["exclusions"]["v8_tag_redistribution_authorized"] is False
    assert payload["exclusions"]["large_artifacts_excluded"] is True


def test_build_verify_and_index_drift_detection(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _write(root, "tracked.txt", b"one\n")
    _git(root, "add", "tracked.txt")
    manifest = root / BUILDER.MANIFEST_NAME

    receipt = BUILDER.write_manifest(root, manifest)
    assert receipt["file_count"] == 1
    verification = BUILDER.verify_manifest(root, manifest)
    assert verification["status"] == "PASS_PUBLIC_ENGINEERING_EXPORT_INDEX_IDENTITY"

    with pytest.raises(BUILDER.ExportManifestError, match="refusing to overwrite"):
        BUILDER.write_manifest(root, manifest)

    _write(root, "second.txt", b"two\n")
    _git(root, "add", "second.txt")
    with pytest.raises(BUILDER.ExportManifestError, match="differs from"):
        BUILDER.verify_manifest(root, manifest)

    replaced = BUILDER.write_manifest(root, manifest, replace=True)
    assert replaced["file_count"] == 2
    assert BUILDER.verify_manifest(root, manifest)["file_count"] == 2


@pytest.mark.parametrize(
    "value",
    [
        "",
        ".",
        "/absolute",
        "../escape",
        "a/../b",
        "a//b",
        "a\\b",
        "line\nbreak",
    ],
)
def test_unsafe_paths_are_rejected(value: str) -> None:
    with pytest.raises(BUILDER.ExportManifestError):
        BUILDER._safe_index_path(value)


def test_symlink_and_unmerged_index_entries_are_rejected() -> None:
    symlink = b"120000 " + b"a" * 40 + b" 0\tlink\0"
    with pytest.raises(BUILDER.ExportManifestError, match="symlink"):
        BUILDER._parse_index_entries(symlink)

    unmerged = b"100644 " + b"b" * 40 + b" 2\tconflicted.txt\0"
    with pytest.raises(BUILDER.ExportManifestError, match="unmerged"):
        BUILDER._parse_index_entries(unmerged)


def test_duplicate_and_case_colliding_index_paths_are_rejected() -> None:
    raw = (
        b"100644 " + b"a" * 40 + b" 0\tCase.txt\0"
        b"100644 " + b"b" * 40 + b" 0\tcase.txt\0"
    )
    with pytest.raises(BUILDER.ExportManifestError, match="case-colliding"):
        BUILDER._parse_index_entries(raw)


def test_strict_manifest_rejects_noncanonical_duplicate_and_nonfinite_json() -> None:
    for raw in (
        b'{"a":1, "b":2}\n',
        b'{"a":1,"a":2}\n',
        b'{"a":NaN}\n',
        b'{"a":1}',
    ):
        with pytest.raises(BUILDER.ExportManifestError):
            BUILDER._strict_manifest(raw)


def test_rehashed_metadata_tamper_is_still_rejected(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _write(root, "tracked.txt", b"content\n")
    _git(root, "add", "tracked.txt")
    payload = BUILDER.build_payload(root)
    payload["scope"]["environment_lock"] = True
    core = dict(payload)
    del core["manifest_payload_sha256"]
    payload["manifest_payload_sha256"] = hashlib.sha256(
        BUILDER._canonical_json(core)
    ).hexdigest()
    with pytest.raises(BUILDER.ExportManifestError, match="authorization boundary"):
        BUILDER._validate_payload(payload)


def test_boolean_metadata_cannot_be_replaced_by_equal_integer(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _write(root, "tracked.txt", b"content\n")
    _git(root, "add", "tracked.txt")
    payload = BUILDER.build_payload(root)
    payload["scope"]["repository_wide_green"] = 0
    core = dict(payload)
    del core["manifest_payload_sha256"]
    payload["manifest_payload_sha256"] = hashlib.sha256(
        BUILDER._canonical_json(core)
    ).hexdigest()
    with pytest.raises(BUILDER.ExportManifestError, match="authorization boundary"):
        BUILDER._validate_payload(payload)
