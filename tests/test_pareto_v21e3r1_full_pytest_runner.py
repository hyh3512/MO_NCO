from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import platform
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_SCRIPT = (
    ROOT
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "run_v21e3r1_full_pytest.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "v21e3r1_full_pytest_runner", RUNNER_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_pytest_summary_uses_the_terminal_summary_line() -> None:
    runner = _load_runner()
    output = (
        "a test printed 99 passed in 0.01s\n"
        "================ 2 failed, 17 passed, 3 errors in 4.25s ================\n"
    )

    assert runner.parse_pytest_summary(output) == {
        "summary_parsed": True,
        "passed": 17,
        "failed": 2,
        "errors": 3,
        "pytest_reported_duration_seconds": 4.25,
    }


def test_parse_pytest_summary_accepts_pytests_long_duration_suffix() -> None:
    runner = _load_runner()

    assert runner.parse_pytest_summary(
        "======= 824 passed, 3 skipped in 125.25s (0:02:05) =======\n"
    ) == {
        "summary_parsed": True,
        "passed": 824,
        "failed": 0,
        "errors": 0,
        "pytest_reported_duration_seconds": 125.25,
    }


def test_run_full_pytest_streams_and_commits_canonical_pass_receipt(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "test_smoke.py").write_text(
        "def test_smoke():\n    assert 6 * 7 == 42\n",
        encoding="utf-8",
    )
    log_path = repo / "evidence" / "full-pytest.log"
    receipt_path = repo / "evidence" / "full-pytest.json"
    display = io.StringIO()

    receipt = runner.run_full_pytest(
        repo_root=repo,
        log_path=log_path,
        receipt_path=receipt_path,
        prospective_source_root_sha256="a" * 64,
        display_stream=display,
    )

    executable = Path(os.path.abspath(sys.executable))
    log_raw = log_path.read_bytes()
    assert receipt["schema"] == "pareto_v21e3r1_full_pytest_receipt_v1"
    assert receipt["status"] == "PASS"
    assert receipt["suite_scope"] == "repository_full_pytest_q_v1"
    assert receipt["prospective_source_root_sha256"] == "a" * 64
    assert receipt["command"] == [str(executable), "-m", "pytest", "-q"]
    assert receipt["cwd"] == "."
    assert receipt["cwd_path_semantics"] == "repo_root_self_v1"
    assert receipt["runtime"] == {
        "implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    assert receipt["executable"] == str(executable)
    assert receipt["executable_sha256"] == hashlib.sha256(
        executable.read_bytes()
    ).hexdigest()
    assert receipt["artifact_path_semantics"] == "repo_root_relative_posix_v1"
    assert receipt["log_path"] == "evidence/full-pytest.log"
    assert receipt["log_sha256"] == hashlib.sha256(log_raw).hexdigest()
    assert receipt["log_bytes"] == len(log_raw)
    assert display.getvalue().encode("utf-8") == log_raw
    assert receipt["exit_code"] == 0
    assert receipt["summary_parsed"] is True
    assert receipt["passed"] == 1
    assert receipt["failed"] == 0
    assert receipt["errors"] == 0
    assert receipt["duration_seconds"] >= 0.0
    assert receipt["pytest_reported_duration_seconds"] >= 0.0
    assert receipt["selection_authorization"] == "PROHIBITED"
    assert receipt["formal_authorized"] is False
    expected_raw = (
        json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert receipt_path.read_bytes() == expected_raw


def test_result_status_requires_exit_zero_and_a_parsed_positive_pass_count() -> None:
    runner = _load_runner()

    assert runner.classify_pytest_result(
        exit_code=0, summary_parsed=True, passed=1
    ) == "PASS"
    for exit_code, summary_parsed, passed in (
        (1, True, 1),
        (0, False, 1),
        (0, True, 0),
    ):
        assert runner.classify_pytest_result(
            exit_code=exit_code,
            summary_parsed=summary_parsed,
            passed=passed,
        ) == "FAIL"


def test_existing_receipt_is_never_replaced_or_followed_by_execution(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    repo = tmp_path / "repo"
    repo.mkdir()
    receipt_path = repo / "evidence" / "full-pytest.json"
    receipt_path.parent.mkdir()
    receipt_path.write_bytes(b"pre-existing receipt\n")
    log_path = repo / "evidence" / "full-pytest.log"

    with pytest.raises(FileExistsError, match="receipt"):
        runner.run_full_pytest(
            repo_root=repo,
            log_path=log_path,
            receipt_path=receipt_path,
            prospective_source_root_sha256="a" * 64,
            display_stream=io.StringIO(),
        )

    assert receipt_path.read_bytes() == b"pre-existing receipt\n"
    assert not log_path.exists()

