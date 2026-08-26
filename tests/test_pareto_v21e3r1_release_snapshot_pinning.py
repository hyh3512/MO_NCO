from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "build_v21e3r1_code_release.py"
)


def _builder():
    spec = importlib.util.spec_from_file_location(
        "v21e3r1_release_snapshot_pinning_builder", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical(payload: object) -> bytes:
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


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _authorized_fixture(tmp_path: Path) -> tuple[Path, Path, Path, bytes]:
    repo = tmp_path / "repo"
    original_source = b"VALUE = 'AUTHORIZED'\n"
    files = {
        "mo_nco/__init__.py": b"",
        "mo_nco/pareto_v21e3_entry.py": original_source,
        "tests/test_pareto_v21e3_entry.py": (
            b"from mo_nco.pareto_v21e3_entry import VALUE\n"
            b"def test_entry(): assert VALUE == 'AUTHORIZED'\n"
        ),
        "ijoc_submission_v21e3r1/release/pyproject.toml": (
            b"[build-system]\nrequires=['setuptools==80.10.2']\n"
            b"build-backend='setuptools.build_meta'\n[project]\n"
            b"name='fixture'\nversion='0.21.3.1'\n"
        ),
        "ijoc_submission_v21e3r1/release/README.md": b"fixture\n",
        "ijoc_submission_v21e3r1/release/requirements-test.lock": (
            b"pytest==9.1.1\n"
        ),
        "ijoc_submission_v21e3r1/release/mo_nco_init.py": (
            b"__version__ = '0.21.3.1'\n"
        ),
        "ijoc_submission_v21e3r1/scripts/build_v21e3r1_code_release.py": (
            b"# fixture builder\n"
        ),
    }
    for relative, raw in files.items():
        _write(repo / relative, raw)

    entries = [
        {
            "path": relative,
            "bytes": len(raw),
            "sha256": _sha256(raw),
        }
        for relative, raw in sorted(files.items())
    ]
    source_root = _sha256(_canonical(entries))
    snapshot_relative = (
        "ijoc_submission_v21e3r1/provenance/TEST_SOURCE_SNAPSHOT.json"
    )
    snapshot = {
        "bound_file_count": len(entries),
        "bound_files_root_sha256": source_root,
        "bound_files": entries,
    }
    snapshot_raw = _canonical(snapshot)
    _write(repo / snapshot_relative, snapshot_raw)
    authorization = {
        "source_snapshot_path": snapshot_relative,
        "source_snapshot_receipt_sha256": _sha256(snapshot_raw),
        "source_snapshot_root_sha256": source_root,
        "selection_entropy_release": "PROHIBITED",
        "formal_authorized": False,
    }
    authorization_path = (
        repo
        / "ijoc_submission_v21e3r1"
        / "provenance"
        / "TEST_AUTHORIZATION.json"
    )
    _write(authorization_path, _canonical(authorization))
    return (
        repo,
        authorization_path,
        repo / "mo_nco" / "pareto_v21e3_entry.py",
        original_source,
    )


def test_verified_release_pins_authorized_bytes_before_live_tree_can_change(
    tmp_path: Path, monkeypatch,
) -> None:
    """A post-preflight path replacement cannot enter the verified ZIP."""

    builder = _builder()
    repo, authorization_path, source_path, original_source = _authorized_fixture(
        tmp_path
    )
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    preflight = importlib.import_module(
        "ijoc_submission_v21e3r1.scripts.preflight_v21e3r1_development_parity"
    )

    def verified_authorization(**kwargs):
        assert kwargs["repo_root"] == repo.resolve()
        assert kwargs["authorization_path"] == authorization_path.resolve()
        return authorization

    monkeypatch.setattr(
        preflight,
        "verify_existing_development_parity_authorization",
        verified_authorization,
    )
    real_build_release = builder.build_release
    attacker_source = b"VALUE = 'POST_PREFLIGHT_REPLACEMENT'\n"

    def replace_live_path_then_package(build_root: Path, **kwargs):
        source_path.write_bytes(attacker_source)
        staged_source = build_root / "mo_nco" / "pareto_v21e3_entry.py"
        if staged_source != source_path:
            staged_source.write_bytes(attacker_source)
        return real_build_release(build_root, **kwargs)

    monkeypatch.setattr(builder, "build_release", replace_live_path_then_package)
    archive_path = tmp_path / "verified.zip"
    manifest_path = tmp_path / "verified.manifest.json"
    checksum_path = tmp_path / "verified.sha256"

    builder.build_verified_release(
        repo,
        authorization_path=authorization_path,
        archive_path=archive_path,
        manifest_path=manifest_path,
        checksum_path=checksum_path,
    )

    assert source_path.read_bytes() == attacker_source
    with zipfile.ZipFile(archive_path) as archive:
        archived = archive.read(
            "ijoc_v21e3r1_experiment_code/mo_nco/pareto_v21e3_entry.py"
        )
    assert archived == original_source
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in manifest["files"]
        if item["archive_path"] == "mo_nco/pareto_v21e3_entry.py"
    )
    assert entry["sha256"] == _sha256(original_source)
    assert entry["bytes"] == len(original_source)


def test_verified_release_excludes_a_file_created_after_authorization(
    tmp_path: Path, monkeypatch,
) -> None:
    builder = _builder()
    repo, authorization_path, _, _ = _authorized_fixture(tmp_path)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    preflight = importlib.import_module(
        "ijoc_submission_v21e3r1.scripts.preflight_v21e3r1_development_parity"
    )
    monkeypatch.setattr(
        preflight,
        "verify_existing_development_parity_authorization",
        lambda **kwargs: authorization,
    )
    real_build_release = builder.build_release
    unbound = (
        repo
        / "ijoc_submission_v21e3r1"
        / "provenance"
        / "POST_PREFLIGHT_UNBOUND.json"
    )

    def add_unbound_file_then_package(build_root: Path, **kwargs):
        _write(unbound, b'{"unauthorized":true}\n')
        return real_build_release(build_root, **kwargs)

    monkeypatch.setattr(builder, "build_release", add_unbound_file_then_package)
    archive_path = tmp_path / "verified.zip"
    builder.build_verified_release(
        repo,
        authorization_path=authorization_path,
        archive_path=archive_path,
        manifest_path=tmp_path / "verified.manifest.json",
        checksum_path=tmp_path / "verified.sha256",
    )

    assert unbound.is_file()
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert not any(name.endswith("POST_PREFLIGHT_UNBOUND.json") for name in names)


def test_verified_release_fails_closed_if_a_bound_file_changes_after_preflight(
    tmp_path: Path, monkeypatch,
) -> None:
    builder = _builder()
    repo, authorization_path, source_path, _ = _authorized_fixture(tmp_path)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    preflight = importlib.import_module(
        "ijoc_submission_v21e3r1.scripts.preflight_v21e3r1_development_parity"
    )

    def mutate_after_preflight(**kwargs):
        source_path.write_bytes(b"VALUE = 'DRIFTED_BEFORE_CAPTURE'\n")
        return authorization

    monkeypatch.setattr(
        preflight,
        "verify_existing_development_parity_authorization",
        mutate_after_preflight,
    )
    archive_path = tmp_path / "must-not-exist.zip"

    with pytest.raises(RuntimeError, match="source/evidence file drifted"):
        builder.build_verified_release(
            repo,
            authorization_path=authorization_path,
            archive_path=archive_path,
            manifest_path=tmp_path / "must-not-exist.manifest.json",
            checksum_path=tmp_path / "must-not-exist.sha256",
        )
    assert not archive_path.exists()

