from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


VERIFY_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "verify_v21e3r1_clean_room.py"
)


def _verifier():
    spec = importlib.util.spec_from_file_location(
        "v21e3r1_clean_room", VERIFY_SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gate_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    verifier = _verifier()
    archive = tmp_path / "release.zip"
    archive.write_bytes(b"pinned V21e3r1 release bytes")
    manifest_path = tmp_path / "release.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "archive_prefix": "ijoc_v21e3r1_experiment_code",
                "files": [
                    {"archive_path": "mo_nco/pareto_v21e3_entry.py"},
                    {"archive_path": "tests/test_pareto_v21e3_entry.py"},
                ],
            }
        ),
        encoding="utf-8",
    )
    package_root = tmp_path / "verified-package"
    (package_root / "wheelhouse").mkdir(parents=True)
    (package_root / "requirements-test.lock").write_text(
        "--no-index\n--find-links wheelhouse\n--require-hashes\n"
        "pytest==9.1.1 \\\n"
        "    --hash=sha256:" + "0" * 64 + "\n",
        encoding="utf-8",
    )
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
    monkeypatch.setattr(verifier, "_extract_verified", lambda *_: package_root)
    monkeypatch.setattr(verifier, "_venv_python", lambda *_: Path(sys.executable))
    monkeypatch.setattr(verifier, "_console_script", lambda *_: Path(sys.executable))
    return verifier, archive, manifest_path, package_root


def test_clean_room_does_not_inherit_host_python_environment_or_malicious_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier, archive, manifest_path, _ = _gate_fixture(tmp_path, monkeypatch)
    malicious_root = tmp_path / "host-malicious"
    malicious_package = malicious_root / "mo_nco"
    malicious_package.mkdir(parents=True)
    (malicious_package / "__init__.py").write_text(
        "raise RuntimeError('HOST_MALICIOUS_MO_NCO_IMPORTED')\n",
        encoding="utf-8",
    )
    startup_marker = tmp_path / "host-startup-ran.txt"
    startup = tmp_path / "host-startup.py"
    startup.write_text(
        f"from pathlib import Path\nPath({str(startup_marker)!r}).write_text('ran')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(malicious_root))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "host-fake-python-home"))
    monkeypatch.setenv("PYTHONSTARTUP", str(startup))

    observed_environments: list[dict[str, str]] = []

    def run_with_attack_probe(name, command, *, cwd, logs, environment):
        environment = dict(environment)
        observed_environments.append(environment)
        if name == "04_import_installed_distribution_modules":
            probe = (
                "import importlib.util, json, os, pathlib; "
                "blocked=('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP'); "
                "assert all(key not in os.environ for key in blocked), "
                "{key:os.environ.get(key) for key in blocked}; "
                "spec=importlib.util.find_spec('mo_nco'); "
                f"evil=pathlib.Path({str(malicious_root)!r}).resolve(); "
                "origin=None if spec is None or spec.origin is None else "
                "pathlib.Path(spec.origin).resolve(); "
                "assert origin is None or not origin.is_relative_to(evil), origin; "
                "print(json.dumps({'status':'PASS','origin':str(origin)}))"
            )
            completed = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=cwd,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            assert completed.returncode == 0, completed.stdout
        if name == "07_deterministic_rebuild":
            command = list(command)
            rebuilt = Path(command[command.index("--archive-path") + 1])
            rebuilt.parent.mkdir(parents=True, exist_ok=True)
            rebuilt.write_bytes(archive.read_bytes())
        return {"name": name, "command": list(command), "exit_code": 0}

    monkeypatch.setattr(verifier, "_run", run_with_attack_probe)
    result = verifier.run_clean_room_gate(
        archive_path=archive,
        manifest_path=manifest_path,
        work_directory=tmp_path / "work",
        receipt_path=tmp_path / "receipt.json",
        base_python=sys.executable,
    )

    assert result["status"] == "PASS"
    assert observed_environments
    assert all(
        key not in environment
        for environment in observed_environments
        for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")
    )
    assert not startup_marker.exists()


def test_clean_room_receipt_separates_installed_and_extracted_tree_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier, archive, manifest_path, package_root = _gate_fixture(
        tmp_path, monkeypatch
    )
    calls: list[tuple[str, list[str], Path, dict[str, str]]] = []

    def fake_run(name, command, *, cwd, logs, environment):
        command = list(command)
        calls.append((name, command, Path(cwd), dict(environment)))
        if name == "07_deterministic_rebuild":
            rebuilt = Path(command[command.index("--archive-path") + 1])
            rebuilt.parent.mkdir(parents=True, exist_ok=True)
            rebuilt.write_bytes(archive.read_bytes())
        return {"name": name, "command": command, "exit_code": 0}

    monkeypatch.setattr(verifier, "_run", fake_run)
    receipt_path = tmp_path / "receipt.json"
    result = verifier.run_clean_room_gate(
        archive_path=archive,
        manifest_path=manifest_path,
        work_directory=tmp_path / "work",
        receipt_path=receipt_path,
        base_python=sys.executable,
    )

    assert result["schema"] == "ijoc_v21e3r1_clean_room_gate_receipt_v2"
    assert result["python_environment_isolation"] == {
        "policy": "scrub_inherited_python_bootstrap_variables_v1",
        "scrubbed_variables": ["PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP"],
        "host_pythonpath_inheritance": "PROHIBITED",
    }
    assert result["execution_evidence"] == {
        "installed_distribution_import": {
            "gate": "PASS",
            "step": "04_import_installed_distribution_modules",
            "artifact_under_test": "fresh_venv_installed_distribution",
            "origin_constraint": "venv_purelib_or_platlib",
            "module_count": 1,
        },
        "extracted_tree_pytest": {
            "gate": "PASS",
            "step": "05_run_extracted_tree_v21_tests",
            "artifact_under_test": "verified_extracted_release_tree",
            "test_file_count": 1,
            "installed_distribution_test_claim": "NOT_CLAIMED",
        },
        "installed_distribution_console_cli": {
            "gate": "PASS",
            "step": "06_run_installed_distribution_console_cli",
            "artifact_under_test": "fresh_venv_installed_console_script",
        },
    }
    assert result["installed_distribution_import_gate"] == "PASS"
    assert result["extracted_tree_pytest_gate"] == "PASS"
    assert result["installed_distribution_console_cli_gate"] == "PASS"
    assert "all_module_import_gate" not in result
    assert "all_packaged_v21_tests_gate" not in result
    assert "console_cli_smoke_gate" not in result

    pip_calls = [
        call
        for call in calls
        if len(call[1]) >= 4 and call[1][1:4] == ["-m", "pip", "install"]
    ]
    assert len(pip_calls) == 2
    for _, command, _, environment in pip_calls:
        assert "--no-index" in command
        assert environment["PIP_NO_INDEX"] == "1"
        assert environment["PIP_CONFIG_FILE"] == os.devnull
        assert environment["PIP_INDEX_URL"] == ""
        assert environment["PIP_EXTRA_INDEX_URL"] == ""

    import_call = next(
        call for call in calls if call[0] == "04_import_installed_distribution_modules"
    )
    assert "uninstalled package origin" in import_call[1][2]
    assert import_call[2] != package_root
    pytest_call = next(
        call for call in calls if call[0] == "05_run_extracted_tree_v21_tests"
    )
    assert pytest_call[2] == package_root
    assert "--import-mode=prepend" in pytest_call[1]
    cli_call = next(
        call
        for call in calls
        if call[0] == "06_run_installed_distribution_console_cli"
    )
    assert cli_call[2] != package_root
    assert json.loads(receipt_path.read_text(encoding="utf-8"))[
        "execution_evidence"
    ] == result["execution_evidence"]

