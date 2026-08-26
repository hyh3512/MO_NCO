from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ijoc_submission_v21e3r1.scripts import (
    run_v21e3r1_development_diagnostics as diagnostic_runner,
)


ROOT = Path(__file__).resolve().parents[1]


def _run_single_c0_smoke(
    output: Path,
    *,
    resume: bool = False,
) -> dict[str, object]:
    return diagnostic_runner.run_matrix(
        ROOT,
        output,
        ("C0_STANDARD",),
        case_ids=(diagnostic_runner.EXPECTED_CASE_IDS[0],),
        seeds=(diagnostic_runner.SEEDS[0],),
        budget=48,
        checkpoint_period=12,
        smoke=True,
        resume=resume,
        row_timeout_seconds=120,
    )


@pytest.mark.parametrize(
    ("arms", "error_match"),
    (
        (
            diagnostic_runner.DIAGNOSTIC_ARMS[:-1],
            "Full mode requires the exact 12 cases, 3 seeds, 14 arms",
        ),
        (
            diagnostic_runner.DIAGNOSTIC_ARMS
            + (diagnostic_runner.DIAGNOSTIC_ARMS[-1],),
            "arms must not contain duplicates",
        ),
        (
            (
                diagnostic_runner.DIAGNOSTIC_ARMS[1],
                diagnostic_runner.DIAGNOSTIC_ARMS[0],
                *diagnostic_runner.DIAGNOSTIC_ARMS[2:],
            ),
            "Full mode requires the exact 12 cases, 3 seeds, 14 arms",
        ),
    ),
    ids=("subset", "duplicate", "out-of-order"),
)
def test_full_mode_rejects_any_nonexact_arm_design(
    tmp_path: Path,
    arms: tuple[str, ...],
    error_match: str,
) -> None:
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match=error_match):
        diagnostic_runner.run_matrix(ROOT, output, arms)

    assert not output.exists()


def test_smoke_rejects_a_case_outside_the_frozen_exposed_packet(
    tmp_path: Path,
) -> None:
    output = tmp_path / "must-not-exist"

    with pytest.raises(
        ValueError,
        match="Only frozen exposed-development case IDs are allowed",
    ):
        diagnostic_runner.run_matrix(
            ROOT,
            output,
            ("C0_STANDARD",),
            case_ids=("v21e3-mokp-formal-n100-s00",),
            seeds=(diagnostic_runner.SEEDS[0],),
            budget=48,
            checkpoint_period=12,
            smoke=True,
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("overrides", "error_match"),
    (
        ({"budget": True}, "budget must be an exact integer >= 1"),
        (
            {"seeds": (True,)},
            "seeds must contain exact nonnegative integers",
        ),
    ),
    ids=("boolean-budget", "boolean-seed"),
)
def test_smoke_rejects_booleans_in_exact_integer_fields(
    tmp_path: Path,
    overrides: dict[str, object],
    error_match: str,
) -> None:
    output = tmp_path / "must-not-exist"
    arguments: dict[str, object] = {
        "case_ids": (diagnostic_runner.EXPECTED_CASE_IDS[0],),
        "seeds": (diagnostic_runner.SEEDS[0],),
        "budget": 48,
        "checkpoint_period": 12,
        "smoke": True,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=error_match):
        diagnostic_runner.run_matrix(
            ROOT,
            output,
            ("C0_STANDARD",),
            **arguments,
        )

    assert not output.exists()


def test_single_c0_smoke_passes_strict_48_evaluation_replay_at_reserved_path(
    tmp_path: Path,
) -> None:
    output = tmp_path / "#%[]"

    receipt = _run_single_c0_smoke(output)

    assert receipt["status"] == "PASS_DIAGNOSTIC_SMOKE_ONLY"
    assert receipt["matrix_mode"] == "SMOKE_ONLY"
    assert receipt["completed_rows"] == receipt["expected_rows"] == 1
    assert receipt["selection_entropy_release"] == "PROHIBITED"
    assert receipt["confirmation_materialization"] == "PROHIBITED"
    assert receipt["formal_materialization"] == "PROHIBITED"

    completed_files = list((output / "completed").glob("*.json"))
    assert len(completed_files) == 1
    completed = json.loads(completed_files[0].read_text(encoding="utf-8"))
    attempt = output / str(completed["attempt_directory"])
    row = json.loads((attempt / "row.json").read_text(encoding="utf-8"))
    verification = row["strict_trace_verification"]
    independent_path = attempt / "independent.metric.json"
    independent = json.loads(independent_path.read_text(encoding="utf-8"))
    trace_sha256 = hashlib.sha256((attempt / "trace.sqlite3").read_bytes()).hexdigest()

    assert row["status"] == "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY"
    assert row["charged_evaluation_budget"] == 48
    assert verification["status"] == "OBJECTIVE_AND_ARCHIVE_REPLAY_PASS"
    assert verification["evaluation_records"] == 48
    assert verification["terminal_status"] == "SUCCESS"
    assert verification["unresolved_decision_records"] == 0
    assert independent["status"] == "PASS_INDEPENDENT_METRIC_IMPLEMENTATION"
    assert independent["evaluation_count"] == 48
    assert independent["trace_sha256"] == trace_sha256
    assert row["trace_database_sha256"] == trace_sha256
    assert verification["database_sha256"] == trace_sha256
    assert row["independent_metric_replay"] == independent
    assert row["independent_metric_receipt_path"] == "independent.metric.json"
    assert row["independent_metric_receipt_sha256"] == hashlib.sha256(
        independent_path.read_bytes()
    ).hexdigest()
    assert Path(str(verification["database_path"])).resolve() == (
        attempt / "trace.sqlite3"
    ).resolve()


def test_completed_smoke_resume_returns_the_existing_sealed_receipt(
    tmp_path: Path,
) -> None:
    output = tmp_path / "completed-resume"
    original = _run_single_c0_smoke(output)

    resumed = _run_single_c0_smoke(output, resume=True)

    assert resumed == original
    assert len(list((output / "attempts").glob("**/attempt-*"))) == 1


def test_completed_smoke_resume_rejects_a_tampered_row_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "tampered-completed-resume"
    _run_single_c0_smoke(output)
    completed_path = next((output / "completed").glob("*.json"))
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    row_path = output / str(completed["attempt_directory"]) / "row.json"
    row_path.write_bytes(row_path.read_bytes() + b" ")

    with pytest.raises(RuntimeError, match="artifact drifted"):
        _run_single_c0_smoke(output, resume=True)
