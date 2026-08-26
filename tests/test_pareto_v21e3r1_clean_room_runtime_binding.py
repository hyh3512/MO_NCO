from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sqlite3
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = (
    REPO_ROOT
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "verify_v21e3r1_clean_room.py"
)
BUILDER_PATH = (
    REPO_ROOT
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "build_v21e3r1_code_release.py"
)


def _verifier():
    spec = importlib.util.spec_from_file_location(
        "v21e3r1_clean_room_runtime_binding_verifier", VERIFIER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _builder():
    spec = importlib.util.spec_from_file_location(
        "v21e3r1_runtime_binding_release_builder", BUILDER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _runtime_fixture(tmp_path: Path, verifier, monkeypatch):
    archive = tmp_path / "release.zip"
    archive.write_bytes(b"verified release bytes")
    manifest = tmp_path / "release.manifest.json"
    manifest_payload = {
        "archive_prefix": "ijoc_v21e3r1_experiment_code",
        "files": [
            {"archive_path": "mo_nco/runtime_probe.py"},
            {"archive_path": "tests/test_pareto_v21e3_runtime_probe.py"},
        ],
    }
    manifest.write_text(
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    package_root = tmp_path / "package"
    package_root.mkdir()
    (package_root / "wheelhouse").mkdir()
    (package_root / "requirements-test.lock").write_text(
        "--no-index\n"
        "--find-links wheelhouse\n"
        "--require-hashes\n"
        "fixture==1.0 \\\n"
        "--hash=sha256:" + "0" * 64 + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        verifier,
        "verify_release_archive",
        lambda *_: {
            "status": "PASS",
            "offline_wheelhouse_gate": "PASS",
            "archive_sha256": _sha256(archive.read_bytes()),
            "manifest_sha256": _sha256(manifest.read_bytes()),
        },
    )
    monkeypatch.setattr(verifier, "_extract_verified", lambda *_: package_root)
    return archive, manifest, package_root


def test_clean_room_persists_exact_base_python_binding_before_first_step(
    tmp_path: Path, monkeypatch,
) -> None:
    verifier = _verifier()
    archive, manifest, _ = _runtime_fixture(tmp_path, verifier, monkeypatch)
    receipt_path = tmp_path / "clean-room.receipt.json"
    observed_before_step: list[dict[str, object]] = []

    def fake_run(name, command, *, cwd, logs, environment):
        if not observed_before_step:
            assert receipt_path.is_file()
            observed_before_step.append(
                json.loads(receipt_path.read_text(encoding="utf-8"))
            )
        command_list = list(command)
        if name == "07_deterministic_rebuild":
            rebuilt = Path(command_list[command_list.index("--archive-path") + 1])
            rebuilt.parent.mkdir(parents=True, exist_ok=True)
            rebuilt.write_bytes(archive.read_bytes())
        return {"name": name, "command": command_list, "exit_code": 0}

    monkeypatch.setattr(verifier, "_run", fake_run)
    result = verifier.run_clean_room_gate(
        archive_path=archive,
        manifest_path=manifest,
        work_directory=tmp_path / "work",
        receipt_path=receipt_path,
        base_python=sys.executable,
    )

    assert result["status"] == "PASS"
    assert len(observed_before_step) == 1
    initial = observed_before_step[0]
    assert initial["status"] == "RUNNING"
    binding = initial["base_python_runtime_binding"]
    executable = Path(sys.executable).resolve(strict=True)
    executable_raw = executable.read_bytes()
    expected_os_pin = (
        "PASS_WINDOWS_READ_HANDLE_DENIES_WRITE_DELETE_AND_RENAME"
        if os.name == "nt"
        else "NOT_PERFORMED_UNSUPPORTED_PLATFORM"
    )
    assert binding == {
        "schema": "ijoc_v21e3r1_base_python_runtime_binding_v1",
        "status": "PASS",
        "canonical_resolved_path": str(executable),
        "bytes": len(executable_raw),
        "sha256": _sha256(executable_raw),
        "sys_version": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "sqlite_version": sqlite3.sqlite_version,
        "runtime_probe_executable_revalidation": "PASS_BEFORE_AND_AFTER_PROBE",
        "venv_creation_executable_revalidation": "NOT_RUN",
        "os_level_executable_handle_pin": expected_os_pin,
    }
    final_binding = result["base_python_runtime_binding"]
    assert {
        key: final_binding[key]
        for key in binding
        if key
        not in {
            "venv_creation_executable_revalidation",
            "os_level_executable_handle_pin",
        }
    } == {
        key: binding[key]
        for key in binding
        if key
        not in {
            "venv_creation_executable_revalidation",
            "os_level_executable_handle_pin",
        }
    }
    assert final_binding["venv_creation_executable_revalidation"] == (
        "PASS_BEFORE_AND_AFTER_VENV_CREATION"
    )
    assert final_binding["os_level_executable_handle_pin"] == (
        expected_os_pin + "_FOR_RUNTIME_PROBE_AND_VENV_CREATION"
        if expected_os_pin.startswith("PASS_")
        else expected_os_pin
    )
    assert result["pip_network_policy"]["os_network_namespace_isolation"] == (
        "NOT_PERFORMED"
    )
    assert result["pip_network_policy"]["arbitrary_test_network_isolation"] == (
        "NOT_PERFORMED"
    )
    assert result["base_python"] == str(executable)
    assert "PYTHONPATH" not in observed_before_step[0].get(
        "inherited_environment", {}
    )


def test_clean_room_failure_receipt_retains_base_python_runtime_binding(
    tmp_path: Path, monkeypatch,
) -> None:
    verifier = _verifier()
    archive, manifest, _ = _runtime_fixture(tmp_path, verifier, monkeypatch)
    receipt_path = tmp_path / "failed-clean-room.receipt.json"

    def fail_first_step(*args, **kwargs):
        raise RuntimeError("synthetic first-step failure")

    monkeypatch.setattr(verifier, "_run", fail_first_step)
    with pytest.raises(RuntimeError, match="synthetic first-step failure"):
        verifier.run_clean_room_gate(
            archive_path=archive,
            manifest_path=manifest,
            work_directory=tmp_path / "failed-work",
            receipt_path=receipt_path,
            base_python=sys.executable,
        )

    failed = json.loads(receipt_path.read_text(encoding="utf-8"))
    executable = Path(sys.executable).resolve(strict=True)
    assert failed["status"] == "FAIL"
    assert failed["error_type"] == "RuntimeError"
    assert failed["base_python_runtime_binding"]["canonical_resolved_path"] == str(
        executable
    )
    assert failed["base_python_runtime_binding"]["sha256"] == _sha256(
        executable.read_bytes()
    )
    assert failed["base_python_runtime_binding"]["sys_version"] == sys.version
    assert failed["pip_network_policy"]["os_network_namespace_isolation"] == (
        "NOT_PERFORMED"
    )
    assert failed["pip_network_policy"]["arbitrary_test_network_isolation"] == (
        "NOT_PERFORMED"
    )


def test_archive_preflight_failure_also_seals_runtime_bound_failure_receipt(
    tmp_path: Path, monkeypatch,
) -> None:
    verifier = _verifier()
    archive = tmp_path / "bad-release.zip"
    archive.write_bytes(b"bad")
    manifest = tmp_path / "bad-release.manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    receipt_path = tmp_path / "archive-failure.receipt.json"
    monkeypatch.setattr(
        verifier,
        "verify_release_archive",
        lambda *_: (_ for _ in ()).throw(
            ValueError("synthetic archive verification failure")
        ),
    )

    with pytest.raises(ValueError, match="synthetic archive verification failure"):
        verifier.run_clean_room_gate(
            archive_path=archive,
            manifest_path=manifest,
            work_directory=tmp_path / "archive-failure-work",
            receipt_path=receipt_path,
            base_python=sys.executable,
        )

    failed = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert failed["status"] == "FAIL"
    assert failed["error_type"] == "ValueError"
    assert failed["base_python_runtime_binding"]["status"] == "PASS"
    assert failed["base_python_runtime_binding"]["sha256"] == _sha256(
        Path(sys.executable).resolve(strict=True).read_bytes()
    )


def test_release_builder_default_authorization_targets_planned_v4_path() -> None:
    builder = _builder()
    expected = (
        REPO_ROOT
        / "ijoc_submission_v21e3r1"
        / "provenance"
        / "V21E3R1_DEVELOPMENT_PARITY_AUTHORIZATION_V4.json"
    )

    assert builder.DEFAULT_AUTHORIZATION_PATH == expected
    parsed = builder.parse_args([])
    assert parsed.authorization_receipt == expected
    assert parsed.archive_path.name == "ijoc_v21e3r1_experiment_code_v4.zip"
    assert parsed.manifest_path.name == (
        "ijoc_v21e3r1_experiment_code_v4.manifest.json"
    )
    assert parsed.checksum_path.name == (
        "ijoc_v21e3r1_experiment_code_v4.zip.sha256"
    )


def test_runtime_probe_failure_still_leaves_a_pre_execution_hash_receipt(
    tmp_path: Path, monkeypatch,
) -> None:
    verifier = _verifier()
    archive = tmp_path / "release.zip"
    archive.write_bytes(b"unused")
    manifest = tmp_path / "release.manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    receipt_path = tmp_path / "runtime-probe-failure.receipt.json"

    def fail_probe(*args, **kwargs):
        raise RuntimeError("synthetic runtime probe failure")

    monkeypatch.setattr(verifier.subprocess, "run", fail_probe)
    with pytest.raises(RuntimeError, match="synthetic runtime probe failure"):
        verifier.run_clean_room_gate(
            archive_path=archive,
            manifest_path=manifest,
            work_directory=tmp_path / "must-not-exist",
            receipt_path=receipt_path,
            base_python=sys.executable,
        )

    failed = json.loads(receipt_path.read_text(encoding="utf-8"))
    binding = failed["base_python_runtime_binding"]
    executable = Path(sys.executable).resolve(strict=True)
    assert failed["status"] == "FAIL"
    assert failed["error_type"] == "RuntimeError"
    assert binding["status"] == "RUNTIME_PROBE_PENDING"
    assert binding["sha256"] == _sha256(executable.read_bytes())
    assert binding["runtime_probe_executable_revalidation"] == "NOT_RUN"
    assert not (tmp_path / "must-not-exist").exists()


def test_missing_base_python_still_leaves_a_failure_receipt(tmp_path: Path) -> None:
    verifier = _verifier()
    archive = tmp_path / "unused.zip"
    archive.write_bytes(b"unused")
    manifest = tmp_path / "unused.manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    receipt_path = tmp_path / "missing-python.receipt.json"
    missing_python = tmp_path / "absent-python.exe"

    with pytest.raises(FileNotFoundError):
        verifier.run_clean_room_gate(
            archive_path=archive,
            manifest_path=manifest,
            work_directory=tmp_path / "must-not-exist",
            receipt_path=receipt_path,
            base_python=missing_python,
        )

    failed = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert failed["status"] == "FAIL"
    assert failed["error_type"] == "FileNotFoundError"
    assert failed["base_python"] == str(missing_python.resolve(strict=False))
    assert failed["base_python_runtime_binding"]["status"] == (
        "BASE_RUNTIME_BINDING_UNAVAILABLE"
    )
    assert failed["base_python_runtime_binding"]["sha256"] == "NOT_AVAILABLE"
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows executable-share contract")
def test_windows_executable_handle_pin_blocks_aba_write_and_replace(
    tmp_path: Path,
) -> None:
    verifier = _verifier()
    executable = tmp_path / "python-copy.exe"
    executable.write_bytes(Path(sys.executable).read_bytes())
    replacement = tmp_path / "replacement.exe"
    replacement.write_bytes(b"replacement")

    with verifier._hold_executable_bytes(executable) as status:
        assert status == "PASS_WINDOWS_READ_HANDLE_DENIES_WRITE_DELETE_AND_RENAME"
        with pytest.raises(PermissionError):
            executable.write_bytes(b"attacked")
        with pytest.raises(PermissionError):
            replacement.replace(executable)

    replacement.replace(executable)
    assert executable.read_bytes() == b"replacement"

