from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "ijoc_submission_v21e3"
    / "scripts"
    / "build_v21e3_code_release.py"
)
VERIFY_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "ijoc_submission_v21e3"
    / "scripts"
    / "verify_v21e3_clean_room.py"
)


def _builder():
    spec = importlib.util.spec_from_file_location("v21e3_release_builder", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verifier():
    spec = importlib.util.spec_from_file_location("v21e3_clean_room", VERIFY_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def test_v21e3_release_closes_internal_dependencies_and_is_deterministic(
    tmp_path: Path,
) -> None:
    builder = _builder()
    repo = tmp_path / "repo"
    _write(repo / "mo_nco" / "__init__.py", "")
    _write(
        repo / "mo_nco" / "pareto_v21e3_entry.py",
        "from .archive import VALUE\nENTRY = VALUE\n",
    )
    _write(repo / "mo_nco" / "archive.py", "from .types import VALUE\n")
    _write(repo / "mo_nco" / "types.py", "VALUE = 21\n")
    _write(repo / "mo_nco" / "unrelated.py", "NO = True\n")
    _write(
        repo / "tests" / "test_pareto_v21e3_entry.py",
        "from mo_nco.pareto_v21e3_entry import ENTRY\ndef test_entry(): assert ENTRY == 21\n",
    )
    _write(
        repo / "ijoc_submission_v21e3" / "scripts" / "build_v21e3_code_release.py",
        "# fixture builder copy\n",
    )
    _write(
        repo / "ijoc_submission_v21e3" / "release" / "pyproject.toml",
        "[build-system]\nrequires=['setuptools>=68']\nbuild-backend='setuptools.build_meta'\n"
        "[project]\nname='mo-nco-v21e3-artifact'\nversion='0.21.3'\n",
    )
    _write(
        repo / "ijoc_submission_v21e3" / "release" / "README.md",
        "# V21e3 standalone package\n",
    )
    _write(
        repo / "ijoc_submission_v21e3" / "release" / "requirements-test.lock",
        "pytest==9.1.1\n",
    )

    first = builder.build_release(
        repo,
        archive_path=tmp_path / "first.zip",
        manifest_path=tmp_path / "first.manifest.json",
        checksum_path=tmp_path / "first.sha256",
    )
    second = builder.build_release(
        repo,
        archive_path=tmp_path / "second.zip",
        manifest_path=tmp_path / "second.manifest.json",
        checksum_path=tmp_path / "second.sha256",
    )

    assert first["archive"]["sha256"] == second["archive"]["sha256"]
    assert (tmp_path / "first.zip").read_bytes() == (tmp_path / "second.zip").read_bytes()
    manifest = json.loads((tmp_path / "first.manifest.json").read_text("utf-8"))
    archive_paths = [entry["archive_path"] for entry in manifest["files"]]
    assert archive_paths == sorted(archive_paths)
    assert "mo_nco/pareto_v21e3_entry.py" in archive_paths
    assert "mo_nco/archive.py" in archive_paths
    assert "mo_nco/types.py" in archive_paths
    assert "mo_nco/__init__.py" in archive_paths
    assert "mo_nco/unrelated.py" not in archive_paths
    assert manifest["dependency_closure"]["unresolved_internal_imports"] == []
    assert manifest["project_version"] == "0.21.3"
    assert manifest["v21e2_immutable_baseline"]["release_zip_sha256"] == (
        "ecc13e7b174dd53ceb9e644ee5a97e2dd0883c4b27433be86e2d5ee056a0a102"
    )
    with zipfile.ZipFile(tmp_path / "first.zip") as archive:
        names = [info.filename for info in archive.infolist()]
        assert names == sorted(names)
        prefix = "ijoc_v21e3_experiment_code/"
        assert prefix + "pyproject.toml" in names
        pyproject = archive.read(prefix + "pyproject.toml").decode("utf-8")
        assert "version='0.21.3'" in pyproject
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
    assert first["archive"]["sha256"] == hashlib.sha256(
        (tmp_path / "first.zip").read_bytes()
    ).hexdigest()


def test_clean_room_archive_verifier_rejects_unmanifested_entry(tmp_path: Path) -> None:
    builder = _builder()
    verifier = _verifier()
    repo = tmp_path / "repo"
    _write(repo / "mo_nco" / "__init__.py", "")
    _write(repo / "mo_nco" / "pareto_v21e3_entry.py", "VALUE = 21\n")
    _write(repo / "tests" / "test_pareto_v21e3_entry.py", "def test_ok(): pass\n")
    _write(
        repo / "ijoc_submission_v21e3" / "scripts" / "build_v21e3_code_release.py",
        "# fixture\n",
    )
    _write(
        repo / "ijoc_submission_v21e3" / "release" / "pyproject.toml",
        "[build-system]\nrequires=['setuptools>=68']\nbuild-backend='setuptools.build_meta'\n"
        "[project]\nname='fixture'\nversion='0.21.3'\n",
    )
    _write(repo / "ijoc_submission_v21e3" / "release" / "README.md", "fixture\n")
    _write(
        repo / "ijoc_submission_v21e3" / "release" / "requirements-test.lock",
        "pytest==9.1.1\n",
    )
    archive = tmp_path / "release.zip"
    manifest = tmp_path / "release.manifest.json"
    builder.build_release(
        repo,
        archive_path=archive,
        manifest_path=manifest,
        checksum_path=tmp_path / "release.sha256",
    )

    valid = verifier.verify_release_archive(archive, manifest)
    assert valid["status"] == "PASS"
    with zipfile.ZipFile(archive, "a") as bundle:
        bundle.writestr("ijoc_v21e3_experiment_code/unmanifested.txt", "bad")
    try:
        verifier.verify_release_archive(archive, manifest)
    except ValueError as exc:
        assert "archive" in str(exc).lower() or "entry" in str(exc).lower()
    else:
        raise AssertionError("Verifier accepted an unmanifested ZIP entry.")


def test_release_binds_offline_wheelhouse_into_standalone_root(tmp_path: Path) -> None:
    builder = _builder()
    repo = tmp_path / "repo"
    _write(repo / "mo_nco" / "__init__.py", "")
    _write(repo / "mo_nco" / "pareto_v21e3_entry.py", "VALUE = 21\n")
    _write(repo / "tests" / "test_pareto_v21e3_entry.py", "def test_ok(): pass\n")
    _write(
        repo / "ijoc_submission_v21e3" / "scripts" / "build_v21e3_code_release.py",
        "# fixture\n",
    )
    _write(
        repo / "ijoc_submission_v21e3" / "release" / "pyproject.toml",
        "[build-system]\nrequires=['setuptools>=68']\nbuild-backend='setuptools.build_meta'\n"
        "[project]\nname='fixture'\nversion='0.21.3'\n",
    )
    _write(repo / "ijoc_submission_v21e3" / "release" / "README.md", "fixture\n")
    _write(
        repo / "ijoc_submission_v21e3" / "release" / "requirements-test.lock",
        "--no-index\n--find-links wheelhouse\n",
    )
    wheel_raw = b"synthetic pure-python wheel fixture"
    wheel_name = "fixture-1.0-py3-none-any.whl"
    wheel_path = (
        repo / "ijoc_submission_v21e3" / "release" / "wheelhouse" / wheel_name
    )
    wheel_path.parent.mkdir(parents=True)
    wheel_path.write_bytes(wheel_raw)
    _write(
        repo / "ijoc_submission_v21e3" / "release" / "wheelhouse_manifest.json",
        json.dumps(
            {
                "schema": "pareto_v21e3_offline_wheelhouse_manifest_v1",
                "install_contract": "pip_no_index_require_hashes_v1",
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
        + "\n",
    )
    archive = tmp_path / "release.zip"
    manifest_path = tmp_path / "release.manifest.json"
    builder.build_release(
        repo,
        archive_path=archive,
        manifest_path=manifest_path,
        checksum_path=tmp_path / "release.sha256",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["offline_wheelhouse_gate"] == "PASS"
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.read(
            "ijoc_v21e3_experiment_code/wheelhouse/" + wheel_name
        ) == wheel_raw


def test_clean_room_gate_enforces_explicit_no_index_for_pip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _verifier()
    archive = tmp_path / "release.zip"
    archive.write_bytes(b"frozen archive bytes")
    manifest_path = tmp_path / "release.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "archive_prefix": "ijoc_v21e3_experiment_code",
                "files": [
                    {"archive_path": "mo_nco/pareto_v21e3_entry.py"},
                    {"archive_path": "tests/test_pareto_v21e3_entry.py"},
                ],
            }
        ),
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
        "    --hash=sha256:" + "0" * 64 + "\n",
        encoding="utf-8",
    )
    calls: list[tuple[str, list[str], dict[str, str]]] = []
    call_cwds: dict[str, Path] = {}
    monkeypatch.setenv("PYTHONPATH", "C:/contaminating/source")
    monkeypatch.setenv("PYTHONHOME", "C:/contaminating/runtime")

    monkeypatch.setattr(
        verifier,
        "verify_release_archive",
        lambda *_: {
            "status": "PASS",
            "offline_wheelhouse_gate": "PASS",
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
        },
    )
    monkeypatch.setattr(
        verifier,
        "_extract_verified",
        lambda *_: package_root,
    )

    def fake_run(name, command, *, cwd, logs, environment):
        command_list = list(command)
        calls.append((name, command_list, dict(environment)))
        call_cwds[name] = Path(cwd)
        if name == "07_deterministic_rebuild":
            rebuilt = Path(command_list[command_list.index("--archive-path") + 1])
            rebuilt.parent.mkdir(parents=True, exist_ok=True)
            rebuilt.write_bytes(archive.read_bytes())
        return {"name": name, "command": command_list, "exit_code": 0}

    monkeypatch.setattr(verifier, "_run", fake_run)
    result = verifier.run_clean_room_gate(
        archive_path=archive,
        manifest_path=manifest_path,
        work_directory=tmp_path / "work",
        receipt_path=tmp_path / "receipt.json",
        base_python=Path("C:/fixture/python.exe"),
    )

    pip_calls = [
        call for call in calls if "pip" in call[1] and "install" in call[1]
    ]
    assert result["status"] == "PASS"
    assert result["offline_no_index_gate"] == "PASS"
    assert result["pip_network_policy"]["explicit_no_index_each_install"] is True
    assert result["pip_network_policy"]["os_network_namespace_isolation"] == (
        "NOT_PERFORMED"
    )
    assert len(pip_calls) == 2
    for _, command, environment in pip_calls:
        assert "--no-index" in command
        assert environment["PIP_NO_INDEX"] == "1"
        assert environment["PIP_CONFIG_FILE"] == os.devnull
        assert "PYTHONPATH" not in environment
        assert "PYTHONHOME" not in environment
    import_call = next(call for call in calls if call[0] == "04_import_all_packaged_modules")
    assert import_call[1][1] == "-c"
    assert "uninstalled package origin" in import_call[1][2]
    assert call_cwds["04_import_all_packaged_modules"] != package_root
    pytest_call = next(call for call in calls if call[0] == "05_run_all_packaged_v21_tests")
    assert "--import-mode=importlib" in pytest_call[1]
    assert "-c" in pytest_call[1]
    assert str(package_root / "tests/test_pareto_v21e3_entry.py") in pytest_call[1]
    assert call_cwds["05_run_all_packaged_v21_tests"] != package_root


def test_clean_room_rejects_direct_url_even_under_no_index(tmp_path: Path) -> None:
    verifier = _verifier()
    lock = tmp_path / "requirements-test.lock"
    lock.write_text(
        "--no-index\n"
        "--find-links wheelhouse\n"
        "--require-hashes\n"
        "fixture @ https://example.invalid/fixture.whl \\\n"
        "    --hash=sha256:" + "0" * 64 + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="network-capable"):
        verifier.validate_offline_requirements_lock(lock)


def test_clean_room_rejects_release_input_change_after_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _verifier()
    archive = tmp_path / "release.zip"
    archive.write_bytes(b"original archive")
    manifest = tmp_path / "release.manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")

    def verify_then_mutate(*_):
        receipt = {
            "offline_wheelhouse_gate": "PASS",
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }
        archive.write_bytes(b"changed archive")
        return receipt

    monkeypatch.setattr(verifier, "verify_release_archive", verify_then_mutate)

    with pytest.raises(RuntimeError, match="changed after verification"):
        verifier.run_clean_room_gate(
            archive_path=archive,
            manifest_path=manifest,
            work_directory=tmp_path / "work",
            receipt_path=tmp_path / "receipt.json",
        )


def test_clean_room_subprocess_output_is_streamed_to_bounded_tail(
    tmp_path: Path,
) -> None:
    verifier = _verifier()
    logs = tmp_path / "logs"
    logs.mkdir()

    result = verifier._run(
        "noisy_probe",
        (sys.executable, "-c", "print('x' * 200000)"),
        cwd=tmp_path,
        logs=logs,
        environment=dict(os.environ),
    )

    assert result["exit_code"] == 0
    assert result["log_bytes"] > 200_000
    assert sum(len(line) for line in result["log_tail"]) <= 64 * 1024
    assert result["subprocess_output_capture"].startswith("streamed_to_fsynced")

