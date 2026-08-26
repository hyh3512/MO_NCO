from __future__ import annotations

import hashlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "build_v21e3r1_code_release.py"
)
PARENT_ARCHIVE = (
    REPO_ROOT
    / "ijoc_submission_v21e3"
    / "release"
    / "ijoc_v21e3_experiment_code.zip"
)
DYNAMIC_PATHS = (
    "ijoc_submission_v21e3/scripts/build_v21e3_code_release.py",
    "ijoc_submission_v21e3/scripts/verify_v21e3_clean_room.py",
)


def _builder():
    spec = importlib.util.spec_from_file_location(
        "v21e3r1_release_dynamic_dependencies_builder", SCRIPT_PATH
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


def _authorized_fixture(
    tmp_path: Path, *, included_dynamic_paths: tuple[str, ...]
) -> tuple[object, Path, Path, dict[str, object], bytes, dict[str, bytes]]:
    builder = _builder()
    repo = tmp_path / "repo"
    parent_raw = PARENT_ARCHIVE.read_bytes()
    assert _sha256(parent_raw) == builder.V21E3_RELEASE_ZIP_SHA256
    files = {
        "mo_nco/__init__.py": b"",
        "mo_nco/pareto_v21e3_entry.py": b"VALUE = 21\n",
        "tests/test_pareto_v21e3_release.py": (
            b"from pathlib import Path\n"
            b"ROOT = Path(__file__).resolve().parents[1]\n"
            b"BUILDER = ROOT / 'ijoc_submission_v21e3/scripts/"
            b"build_v21e3_code_release.py'\n"
            b"VERIFIER = ROOT / 'ijoc_submission_v21e3/scripts/"
            b"verify_v21e3_clean_room.py'\n"
            b"def test_dynamic_files_exist():\n"
            b"    assert BUILDER.is_file() and VERIFIER.is_file()\n"
        ),
        "ijoc_submission_v21e3/release/ijoc_v21e3_experiment_code.zip": (
            parent_raw
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
    selected_dynamic = {
        relative: (REPO_ROOT / relative).read_bytes()
        for relative in DYNAMIC_PATHS
    }
    files.update(
        {
            relative: selected_dynamic[relative]
            for relative in included_dynamic_paths
        }
    )
    for relative, raw in files.items():
        _write(repo / relative, raw)
    entries = [
        {"path": relative, "bytes": len(raw), "sha256": _sha256(raw)}
        for relative, raw in sorted(files.items())
    ]
    source_root = _sha256(_canonical(entries))
    snapshot_relative = (
        "ijoc_submission_v21e3r1/provenance/TEST_SOURCE_SNAPSHOT.json"
    )
    snapshot_raw = _canonical(
        {
            "bound_file_count": len(entries),
            "bound_files_root_sha256": source_root,
            "bound_files": entries,
        }
    )
    _write(repo / snapshot_relative, snapshot_raw)
    authorization: dict[str, object] = {
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
        builder,
        repo,
        authorization_path,
        authorization,
        parent_raw,
        selected_dynamic,
    )


def test_verified_release_selects_snapshot_dynamic_dependencies_and_inventories_parent(
    tmp_path: Path, monkeypatch,
) -> None:
    """A hardened successor override wins over the immutable parent member."""

    (
        builder,
        repo,
        authorization_path,
        authorization,
        parent_raw,
        selected_dynamic,
    ) = _authorized_fixture(tmp_path, included_dynamic_paths=DYNAMIC_PATHS)
    preflight = importlib.import_module(
        "ijoc_submission_v21e3r1.scripts.preflight_v21e3r1_development_parity"
    )
    monkeypatch.setattr(
        preflight,
        "verify_existing_development_parity_authorization",
        lambda **kwargs: authorization,
    )

    archive_path = tmp_path / "dynamic-dependencies.zip"
    manifest_path = tmp_path / "dynamic-dependencies.manifest.json"
    builder.build_verified_release(
        repo,
        authorization_path=authorization_path,
        archive_path=archive_path,
        manifest_path=manifest_path,
        checksum_path=tmp_path / "dynamic-dependencies.sha256",
    )

    with zipfile.ZipFile(io.BytesIO(parent_raw)) as parent:
        parent_members = {
            relative: parent.read(
                "ijoc_v21e3_experiment_code/" + relative
            )
            for relative in DYNAMIC_PATHS
        }
    with zipfile.ZipFile(archive_path) as release:
        for relative, raw in selected_dynamic.items():
            assert release.read(
                "ijoc_v21e3r1_experiment_code/" + relative
            ) == raw
        packaged_verifier = release.read(
            "ijoc_v21e3r1_experiment_code/" + DYNAMIC_PATHS[1]
        )
    assert b"streamed_to_fsynced" in packaged_verifier
    assert b"PIP_NO_INDEX" in packaged_verifier
    assert b"streamed_to_fsynced" not in parent_members[DYNAMIC_PATHS[1]]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archived_paths = {item["archive_path"] for item in manifest["files"]}
    assert set(DYNAMIC_PATHS) <= archived_paths
    inventory = {
        item["path"]: item
        for item in manifest["frozen_source_provenance"][
            "dynamic_path_dependency_inventory"
        ]
    }
    builder_item = inventory[DYNAMIC_PATHS[0]]
    assert builder_item["relation"] == "IDENTICAL_TO_IMMUTABLE_PARENT"
    assert builder_item["parent_member_sha256"] == _sha256(
        parent_members[DYNAMIC_PATHS[0]]
    )
    assert builder_item["selected_authorized_member_sha256"] == _sha256(
        selected_dynamic[DYNAMIC_PATHS[0]]
    )
    verifier_item = inventory[DYNAMIC_PATHS[1]]
    assert verifier_item["relation"] == "SUCCESSOR_SNAPSHOT_OVERRIDE"
    assert verifier_item["parent_member_sha256"] == (
        "b7f811ec1ce129b219bb9ed0ea8897302d8701c646d2092ce98cdae5999ba2b0"
    )
    assert verifier_item["selected_authorized_member_sha256"] == (
        "d95c4c3ee12cf9ad723f370f4d9df75dc7ebceaf3172279881da601c91686ad9"
    )


def test_verified_release_rejects_parent_substitution_when_snapshot_omits_paths(
    tmp_path: Path, monkeypatch,
) -> None:
    builder, repo, authorization_path, authorization, _, _ = _authorized_fixture(
        tmp_path, included_dynamic_paths=(DYNAMIC_PATHS[0],)
    )
    preflight = importlib.import_module(
        "ijoc_submission_v21e3r1.scripts.preflight_v21e3r1_development_parity"
    )
    monkeypatch.setattr(
        preflight,
        "verify_existing_development_parity_authorization",
        lambda **kwargs: authorization,
    )
    archive_path = tmp_path / "must-not-exist.zip"
    manifest_path = tmp_path / "must-not-exist.manifest.json"
    checksum_path = tmp_path / "must-not-exist.sha256"

    with pytest.raises(RuntimeError, match="freeze a new successor snapshot"):
        builder.build_verified_release(
            repo,
            authorization_path=authorization_path,
            archive_path=archive_path,
            manifest_path=manifest_path,
            checksum_path=checksum_path,
        )

    assert not archive_path.exists()
    assert not manifest_path.exists()
    assert not checksum_path.exists()

