from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_v9r2r1_full_suite_environment.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("v9r2r1_environment_preflight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREFLIGHT = _load_module()
EXPECTED_EXE = str(Path("C:/miniconda3/python.exe").resolve())


def _versions(name: str) -> str:
    return {"pymoo": "0.6.1.6", "moocore": "0.3.1"}[name]


def _native(_name: str) -> list[dict[str, object]]:
    return [
        {
            "path": "C:\\frozen\\native.pyd",
            "bytes": 3,
            "sha256": hashlib.sha256(b"abc").hexdigest(),
        }
    ]


def _evaluate(**overrides):
    arguments = {
        "expected_python_executable": EXPECTED_EXE,
        "expected_python_version_prefix": "3.13.12",
        "expected_pymoo_version": "0.6.1.6",
        "expected_moocore_version": "0.3.1",
        "observed_python_executable": EXPECTED_EXE,
        "observed_python_version": "3.13.12 frozen",
        "distribution_version": _versions,
        "module_importer": lambda _name: object(),
        "native_artifact_collector": _native,
    }
    arguments.update(overrides)
    return PREFLIGHT.evaluate_full_suite_environment(**arguments)


def _assert_self_hash(receipt: dict[str, object]) -> None:
    declared = receipt.pop("receipt_payload_sha256")
    assert declared == hashlib.sha256(PREFLIGHT._canonical_bytes(receipt)).hexdigest()


def test_pass_only_authorizes_starting_the_full_test_command() -> None:
    receipt, exit_code = _evaluate()
    assert exit_code == 0
    assert receipt["status"] == "PASS_FULL_SUITE_ENVIRONMENT_PREFLIGHT"
    assert receipt["hold_reasons"] == []
    assert receipt["full_suite_execution_preflight_passed"] is True
    assert receipt["full_suite_execution_recommended"] is True
    assert receipt["environment_lock_requirement_satisfied"] is False
    assert receipt["full_development_matrix_authorized"] is False
    assert receipt["selection_authorized"] is False
    _assert_self_hash(receipt)


def test_native_backend_load_failure_is_a_fail_closed_environment_hold() -> None:
    def importer(name: str) -> object:
        if name == "moocore":
            raise ImportError(
                "DLL load failed: application control policy blocked this file"
            )
        return object()

    receipt, exit_code = _evaluate(module_importer=importer)
    assert exit_code == 2
    assert receipt["status"] == "HOLD_FULL_SUITE_ENVIRONMENT"
    assert receipt["full_suite_execution_preflight_passed"] is False
    assert receipt["full_suite_execution_recommended"] is False
    assert receipt["backend_imports"]["moocore"] == {
        "status": "FAIL",
        "exception_type": "ImportError",
        "exception_message": (
            "DLL load failed: application control policy blocked this file"
        ),
    }
    assert "backend_import_failed:moocore" in receipt["hold_reasons"]
    assert receipt["scoped_v9_tests_affected"] is False
    _assert_self_hash(receipt)


def test_version_or_interpreter_drift_is_not_accepted() -> None:
    receipt, exit_code = _evaluate(
        observed_python_executable="D:/different/python.exe",
        observed_python_version="3.13.13 drifted",
        distribution_version=lambda name: (
            "0.6.2" if name == "pymoo" else "0.3.1"
        ),
    )
    assert exit_code == 2
    assert set(receipt["hold_reasons"]) == {
        "distribution_version_mismatch:pymoo",
        "interpreter_executable_exact_match",
        "interpreter_version_prefix_match",
    }
    assert receipt["distributions"]["pymoo"]["exact_match"] is False
    _assert_self_hash(receipt)


def test_native_inventory_error_fails_closed_without_hiding_import_pass() -> None:
    def broken_inventory(name: str) -> list[dict[str, object]]:
        if name == "moocore":
            raise OSError("stable read failed")
        return []

    receipt, exit_code = _evaluate(native_artifact_collector=broken_inventory)
    assert exit_code == 2
    assert receipt["backend_imports"]["moocore"]["status"] == "PASS"
    assert receipt["native_artifacts"]["moocore"]["status"] == "FAIL"
    assert "native_artifact_inventory_failed:moocore" in receipt["hold_reasons"]
    _assert_self_hash(receipt)


def test_exclusive_receipt_is_canonical_and_refuses_overwrite(tmp_path: Path) -> None:
    receipt, _exit_code = _evaluate()
    output = tmp_path / "environment.json"
    PREFLIGHT._write_exclusive(output, receipt)
    assert output.read_bytes() == PREFLIGHT._canonical_bytes(receipt) + b"\n"
    try:
        PREFLIGHT._write_exclusive(output, receipt)
    except PREFLIGHT.FullSuiteEnvironmentError as error:
        assert "refusing to overwrite" in str(error)
    else:
        raise AssertionError("exclusive output was overwritten")

