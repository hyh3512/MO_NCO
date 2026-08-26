from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import pytest


VERIFY_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "verify_v21e3r1_clean_room.py"
)


def _verifier():
    spec = importlib.util.spec_from_file_location(
        "v21e3r1_clean_room_input_pinning", VERIFY_SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_verified_release_fixture(tmp_path: Path) -> tuple[Path, Path, bytes]:
    prefix = "ijoc_v21e3r1_experiment_code"
    wheel_name = "fixture-1.0-py3-none-any.whl"
    wheel_raw = b"fixture wheel bytes"
    wheel_manifest_raw = (
        json.dumps(
            {
                "schema": "pareto_v21e3_offline_wheelhouse_manifest_v1",
                "files": [
                    {
                        "filename": wheel_name,
                        "bytes": len(wheel_raw),
                        "sha256": hashlib.sha256(wheel_raw).hexdigest(),
                    }
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    lock_raw = (
        "--no-index\n"
        "--find-links wheelhouse\n"
        "--require-hashes\n\n"
        "fixture==1.0 \\\n"
        f"    --hash=sha256:{hashlib.sha256(wheel_raw).hexdigest()}\n"
    ).encode("utf-8")
    files = {
        "ijoc_submission_v21e3r1/scripts/build_v21e3r1_code_release.py": (
            b"# fixture builder\n"
        ),
        "mo_nco/pareto_v21e3_fixture.py": b"VALUE = 1\n",
        "requirements-test.lock": lock_raw,
        "tests/test_pareto_v21e3_fixture.py": b"def test_fixture(): assert True\n",
        f"wheelhouse/{wheel_name}": wheel_raw,
        "wheelhouse_manifest.json": wheel_manifest_raw,
    }
    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(
        archive,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as handle:
        for relative, raw in sorted(files.items()):
            info = zipfile.ZipInfo(f"{prefix}/{relative}", (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            handle.writestr(info, raw)
    archive_raw = archive.read_bytes()
    entries = [
        {
            "source_path": relative,
            "archive_path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        for relative, raw in sorted(files.items())
    ]
    manifest = {
        "schema": "ijoc_v21e3r1_standalone_release_manifest_v1",
        "project_version": "0.21.3.1",
        "formal_authorized": False,
        "formal_status": "NOT_MATERIALIZED",
        "dependency_closure": {
            "gate": "PASS",
            "unresolved_internal_imports": [],
        },
        "offline_wheelhouse_gate": "PASS",
        "archive_prefix": prefix,
        "archive": {
            "sha256": hashlib.sha256(archive_raw).hexdigest(),
            "bytes": len(archive_raw),
        },
        "file_count": len(entries),
        "files": entries,
    }
    manifest_path = tmp_path / "release.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return archive, manifest_path, archive_raw


def test_clean_room_rejects_release_bytes_changed_after_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _verifier()
    archive = tmp_path / "release.zip"
    archive.write_bytes(b"original archive")
    manifest = tmp_path / "release.manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")

    def verify_then_mutate(*_):
        result = {
            "offline_wheelhouse_gate": "PASS",
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }
        archive.write_bytes(b"changed archive")
        return result

    monkeypatch.setattr(verifier, "verify_release_archive", verify_then_mutate)

    with pytest.raises(RuntimeError, match="changed after verification"):
        verifier.run_clean_room_gate(
            archive_path=archive,
            manifest_path=manifest,
            work_directory=tmp_path / "work",
            receipt_path=tmp_path / "receipt.json",
        )

    assert not (tmp_path / "work").exists()


def test_clean_room_rebuild_compares_against_pinned_verified_archive_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _verifier()
    archive, manifest, pinned_archive_raw = _write_verified_release_fixture(tmp_path)
    pinned_manifest_raw = manifest.read_bytes()

    def fake_run(name, command, *, cwd, logs, environment):
        command = list(command)
        if name == "07_deterministic_rebuild":
            rebuilt = Path(command[command.index("--archive-path") + 1])
            provenance_manifest = Path(
                command[command.index("--rebuild-provenance-manifest") + 1]
            )
            assert provenance_manifest.resolve() != manifest.resolve()
            assert provenance_manifest.read_bytes() == pinned_manifest_raw
            rebuilt.parent.mkdir(parents=True, exist_ok=True)
            rebuilt.write_bytes(pinned_archive_raw)
            archive.write_bytes(b"replacement after verified-byte extraction")
            manifest.write_text("{}\n", encoding="utf-8")
        return {"name": name, "command": command, "exit_code": 0}

    monkeypatch.setattr(verifier, "_run", fake_run)
    monkeypatch.setattr(verifier, "_venv_python", lambda *_: Path(sys.executable))
    monkeypatch.setattr(verifier, "_console_script", lambda *_: Path(sys.executable))

    result = verifier.run_clean_room_gate(
        archive_path=archive,
        manifest_path=manifest,
        work_directory=tmp_path / "work",
        receipt_path=tmp_path / "receipt.json",
        base_python=sys.executable,
    )

    assert result["status"] == "PASS"
    assert result["pinned_input_bytes_gate"] == "PASS"
    assert result["extraction_source_gate"] == (
        "PASS_PINNED_VERIFIED_ARCHIVE_BYTES"
    )
    assert result["deterministic_rebuild_comparison_source"] == (
        "PINNED_VERIFIED_ARCHIVE_BYTES"
    )
    assert result["requirements_lock_gate"]["status"] == "PASS"
    assert result["requirements_lock_gate"]["package_count"] == 1
    assert result["pip_network_policy"][
        "requirements_lock_semantic_validation"
    ] == "PASS"


def test_offline_requirements_lock_rejects_a_direct_network_source(
    tmp_path: Path,
) -> None:
    verifier = _verifier()
    lock = tmp_path / "requirements-test.lock"
    lock.write_text(
        "--no-index\n"
        "--find-links wheelhouse\n"
        "--require-hashes\n\n"
        "fixture @ https://example.invalid/fixture.whl \\\n"
        "    --hash=sha256:" + "0" * 64 + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="network-capable"):
        verifier.validate_offline_requirements_lock(lock)

