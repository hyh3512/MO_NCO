from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_RUNNER_PATH = (
    ROOT
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "run_v21e3r1_development_diagnostics.py"
)
BATCH_RUNNER_PATH = (
    ROOT
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "run_v21e3r1_same_implementation_branch_replay_coverage.py"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DIAGNOSTIC = _load_module(DIAGNOSTIC_RUNNER_PATH, "v21e3r1_diagnostic_runner_fixture")
BATCH = _load_module(BATCH_RUNNER_PATH, "v21e3r1_branch_replay_coverage")


@pytest.fixture(scope="module")
def one_row_diagnostic(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("v21e3r1_diag") / "diagnostic"
    receipt = DIAGNOSTIC.run_matrix(
        ROOT,
        output,
        ("C0_STANDARD",),
        case_ids=("v21e3-mokp-development-n100-s00",),
        seeds=(31051,),
        budget=24,
        checkpoint_period=6,
        smoke=True,
        row_timeout_seconds=60,
    )
    assert receipt["status"] == "PASS_DIAGNOSTIC_SMOKE_ONLY"
    return output


@pytest.fixture(scope="module")
def two_row_diagnostic(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("v21e3r1_diag_parallel") / "diagnostic"
    receipt = DIAGNOSTIC.run_matrix(
        ROOT,
        output,
        ("C0_STANDARD", "C0_RANDOM"),
        case_ids=("v21e3-mokp-development-n100-s00",),
        seeds=(31051,),
        budget=24,
        checkpoint_period=6,
        smoke=True,
        row_timeout_seconds=60,
    )
    assert receipt["status"] == "PASS_DIAGNOSTIC_SMOKE_ONLY"
    return output


@pytest.fixture(scope="module")
def baseline_family_diagnostic(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("v21e3r1_diag_baselines") / "diagnostic"
    receipt = DIAGNOSTIC.run_matrix(
        ROOT,
        output,
        ("NSGAII_STANDARD", "MOEAD_STANDARD"),
        case_ids=("v21e3-mokp-development-n100-s00",),
        seeds=(31051,),
        budget=48,
        checkpoint_period=12,
        smoke=True,
        row_timeout_seconds=60,
    )
    assert receipt["status"] == "PASS_DIAGNOSTIC_SMOKE_ONLY"
    return output


def test_one_row_smoke_replays_end_to_end_without_authorizing_later_phases(
    one_row_diagnostic: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "coverage"
    receipt = BATCH.run_coverage(
        project_root=ROOT,
        diagnostic_output_root=one_row_diagnostic,
        coverage_output_root=output,
        allow_smoke=True,
        jobs=1,
        row_timeout_seconds=60,
    )

    assert receipt["status"] == "HOLD_COMPLETE_SMOKE_COVERAGE_ONLY_NOT_EXACT_504"
    assert receipt["completed_rows"] == receipt["expected_rows"] == 1
    assert receipt["verification_jobs"] == 1
    assert receipt["implementation_independence"] is False
    assert receipt["scientific_independence"] is False
    assert receipt["third_party_replication"] is False
    assert receipt["selection_authorized"] is False
    assert receipt["confirmation_authorized"] is False
    assert receipt["formal_authorized"] is False
    assert receipt["runtime_efficiency_claims"] is False
    assert receipt["scientific_performance_claims"] is False
    assert (output / "branch_replay_coverage.receipt.json").is_file()


def test_resume_rejects_a_tampered_exclusive_final_receipt(
    one_row_diagnostic: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "coverage"
    BATCH.run_coverage(
        project_root=ROOT,
        diagnostic_output_root=one_row_diagnostic,
        coverage_output_root=output,
        allow_smoke=True,
        jobs=1,
        row_timeout_seconds=60,
    )
    final_path = output / "branch_replay_coverage.receipt.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["diagnostic_receipt_sha256"] = "0" * 64
    final_path.write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(BATCH.CoverageError, match="final coverage receipt"):
        BATCH.run_coverage(
            project_root=ROOT,
            diagnostic_output_root=one_row_diagnostic,
            coverage_output_root=output,
            allow_smoke=True,
            resume=True,
            jobs=1,
            row_timeout_seconds=60,
        )


def test_timeout_terminates_the_entire_worker_process_tree(tmp_path: Path) -> None:
    sentinel = tmp_path / "descendant-survived.txt"
    child = tmp_path / "child.py"
    child.write_text(
        "import pathlib, sys, time\n"
        "time.sleep(3)\n"
        "pathlib.Path(sys.argv[1]).write_text('escaped', encoding='utf-8')\n",
        encoding="utf-8",
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )

    with pytest.raises(BATCH.IsolatedProcessTimeout):
        BATCH.run_isolated_process(
            [sys.executable, str(parent), str(child), str(sentinel)],
            cwd=tmp_path,
            timeout_seconds=1,
        )
    time.sleep(3.25)

    assert not sentinel.exists()


def test_a_terminal_diagnostic_receipt_cannot_coexist_with_missing_row_markers(
    one_row_diagnostic: Path,
    tmp_path: Path,
) -> None:
    diagnostic = tmp_path / "diagnostic"
    shutil.copytree(one_row_diagnostic, diagnostic)
    marker = next((diagnostic / "completed").glob("*.json"))
    marker.unlink()

    with pytest.raises(BATCH.CoverageError, match="claims completion"):
        BATCH.run_coverage(
            project_root=ROOT,
            diagnostic_output_root=diagnostic,
            coverage_output_root=tmp_path / "coverage",
            allow_smoke=True,
            jobs=1,
            row_timeout_seconds=60,
        )


def test_inflight_partial_diagnostic_returns_resumable_hold_without_final_receipt(
    one_row_diagnostic: Path,
    tmp_path: Path,
) -> None:
    diagnostic = tmp_path / "diagnostic"
    shutil.copytree(one_row_diagnostic, diagnostic)
    next((diagnostic / "completed").glob("*.json")).unlink()
    (diagnostic / "diagnostic.receipt.json").unlink()
    (diagnostic / "diagnostic.aggregate.json").unlink()
    output = tmp_path / "coverage"

    progress = BATCH.run_coverage(
        project_root=ROOT,
        diagnostic_output_root=diagnostic,
        coverage_output_root=output,
        allow_smoke=True,
        jobs=1,
        row_timeout_seconds=60,
    )

    assert progress["status"] == BATCH.INCOMPLETE_STATUS
    assert progress["diagnostic_completed_rows"] == 0
    assert progress["completed_rows"] == 0
    assert not (output / "branch_replay_coverage.receipt.json").exists()


def test_diagnostic_artifact_hashes_are_rechecked_before_replay(
    one_row_diagnostic: Path,
    tmp_path: Path,
) -> None:
    diagnostic = tmp_path / "diagnostic"
    shutil.copytree(one_row_diagnostic, diagnostic)
    marker = json.loads(
        next((diagnostic / "completed").glob("*.json")).read_text(encoding="utf-8")
    )
    row_path = diagnostic / marker["attempt_directory"] / "row.json"
    row_path.write_bytes(row_path.read_bytes() + b" ")
    output = tmp_path / "coverage"

    with pytest.raises(BATCH.CoverageError, match="Diagnostic artifact drifted"):
        BATCH.run_coverage(
            project_root=ROOT,
            diagnostic_output_root=diagnostic,
            coverage_output_root=output,
            allow_smoke=True,
            jobs=1,
            row_timeout_seconds=60,
        )

    assert not (output / "attempts").exists()


def test_failed_attempt_is_append_only_and_resume_uses_the_next_attempt(
    one_row_diagnostic: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "coverage"

    def forced_timeout(*args, **kwargs):
        raise BATCH.IsolatedProcessTimeout(
            "forced owned-launcher timeout",
            timeout_seconds=1,
            stdout="",
            stderr="",
            isolation="TEST_OWNED_LAUNCHER_TIMEOUT",
        )

    with monkeypatch.context() as patcher:
        patcher.setattr(BATCH, "run_isolated_process", forced_timeout)
        with pytest.raises(BATCH.CoverageError, match=r"row\(s\) failed"):
            BATCH.run_coverage(
                project_root=ROOT,
                diagnostic_output_root=one_row_diagnostic,
                coverage_output_root=output,
                allow_smoke=True,
                jobs=1,
                row_timeout_seconds=60,
            )

    row_attempt_root = next((output / "attempts").iterdir())
    first = row_attempt_root / "attempt-0001"
    failure_path = first / "failure.receipt.json"
    failure_bytes = failure_path.read_bytes()
    assert json.loads(failure_bytes)["status"] == (
        "FAIL_ROW_TIMEOUT_PROCESS_TREE_TERMINATED"
    )

    receipt = BATCH.run_coverage(
        project_root=ROOT,
        diagnostic_output_root=one_row_diagnostic,
        coverage_output_root=output,
        allow_smoke=True,
        resume=True,
        jobs=1,
        row_timeout_seconds=60,
    )

    assert receipt["status"] == BATCH.SMOKE_COVERAGE_STATUS
    assert failure_path.read_bytes() == failure_bytes
    assert (row_attempt_root / "attempt-0002" / "branch.replay.json").is_file()
    completed = json.loads(next((output / "completed").glob("*.json")).read_text())
    assert completed["attempt_directory"].endswith("/attempt-0002")


def test_unconfirmed_tree_termination_is_never_labeled_terminated(
    one_row_diagnostic: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "coverage"

    def unconfirmed(*args, **kwargs):
        raise BATCH.ProcessTreeTerminationUnconfirmed(
            "forced termination API failure",
            timeout_seconds=1,
            stdout="",
            stderr="TerminateJobObject and CloseHandle failed",
            isolation="TEST_TERMINATION_UNCONFIRMED",
        )

    monkeypatch.setattr(BATCH, "run_isolated_process", unconfirmed)
    with pytest.raises(BATCH.CoverageError, match=r"row\(s\) failed"):
        BATCH.run_coverage(
            project_root=ROOT,
            diagnostic_output_root=one_row_diagnostic,
            coverage_output_root=output,
            allow_smoke=True,
            jobs=1,
            row_timeout_seconds=60,
        )

    failure = json.loads(
        next((output / "attempts").glob("*/*/failure.receipt.json")).read_text(
            encoding="utf-8"
        )
    )
    assert failure["status"] == (
        "FAIL_ROW_TIMEOUT_PROCESS_TREE_TERMINATION_UNCONFIRMED"
    )
    assert not failure["status"].endswith("_TERMINATED")


def test_jobs_two_is_verification_only_and_final_rows_follow_plan_order(
    two_row_diagnostic: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "coverage"

    receipt = BATCH.run_coverage(
        project_root=ROOT,
        diagnostic_output_root=two_row_diagnostic,
        coverage_output_root=output,
        allow_smoke=True,
        jobs=2,
        row_timeout_seconds=60,
    )

    plan = json.loads(
        (two_row_diagnostic / "diagnostic.plan.json").read_text(encoding="utf-8")
    )
    expected = [
        f"{plan['case_ids'][0]}__seed-{plan['seeds'][0]}__arm-{arm.lower()}"
        for arm in plan["arms"]
    ]
    assert receipt["verification_jobs"] == 2
    assert receipt["verification_jobs_observed"] == [2]
    assert [row["row_id"] for row in receipt["row_seals"]] == expected
    assert receipt["runtime_efficiency_claims"] is False
    assert receipt["scientific_performance_claims"] is False
    assert receipt["parallel_execution_semantics"].startswith("VERIFICATION_ONLY")


def test_nsga2_and_moead_completed_rows_replay_through_existing_dispatch(
    baseline_family_diagnostic: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "coverage"

    receipt = BATCH.run_coverage(
        project_root=ROOT,
        diagnostic_output_root=baseline_family_diagnostic,
        coverage_output_root=output,
        allow_smoke=True,
        jobs=2,
        row_timeout_seconds=60,
    )

    assert receipt["status"] == BATCH.SMOKE_COVERAGE_STATUS
    assert [row["row_id"].rsplit("__arm-", 1)[1] for row in receipt["row_seals"]] == [
        "nsgaii_standard",
        "moead_standard",
    ]
    for replay_path in (output / "attempts").glob("*/*/branch.replay.json"):
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        assert replay["status"] == BATCH.BRANCH_PASS_STATUS
        assert replay["implementation_independence"] is False
        assert replay["scientific_independence"] is False


def test_declared_full_plan_must_be_the_exact_frozen_504_row_design(
    one_row_diagnostic: Path,
    tmp_path: Path,
) -> None:
    forged_plan = json.loads(
        (one_row_diagnostic / "diagnostic.plan.json").read_text(encoding="utf-8")
    )
    forged_plan["status"] = BATCH.FULL_PLAN_STATUS
    forged_plan_path = tmp_path / "forged-full-plan.json"
    forged_plan_path.write_text(
        json.dumps(forged_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output = tmp_path / "coverage"

    with pytest.raises(BATCH.CoverageError, match="not the exact frozen 504-row plan"):
        BATCH.run_coverage(
            project_root=ROOT,
            diagnostic_output_root=one_row_diagnostic,
            diagnostic_plan_path=forged_plan_path,
            coverage_output_root=output,
            allow_smoke=True,
            jobs=1,
            row_timeout_seconds=60,
        )

    assert not output.exists()


def test_smoke_plan_requires_explicit_engineering_opt_in(
    one_row_diagnostic: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "coverage"

    with pytest.raises(BATCH.CoverageError, match="allow_smoke=True"):
        BATCH.run_coverage(
            project_root=ROOT,
            diagnostic_output_root=one_row_diagnostic,
            coverage_output_root=output,
        )

    assert not output.exists()


def test_completed_coverage_is_exclusive_and_resume_is_validation_only(
    one_row_diagnostic: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "coverage"
    original = BATCH.run_coverage(
        project_root=ROOT,
        diagnostic_output_root=one_row_diagnostic,
        coverage_output_root=output,
        allow_smoke=True,
        jobs=1,
        row_timeout_seconds=60,
    )
    final_path = output / "branch_replay_coverage.receipt.json"
    final_bytes = final_path.read_bytes()
    attempt_dirs_before = sorted((output / "attempts").glob("*/*"))

    with pytest.raises(FileExistsError):
        BATCH.run_coverage(
            project_root=ROOT,
            diagnostic_output_root=one_row_diagnostic,
            coverage_output_root=output,
            allow_smoke=True,
        )
    resumed = BATCH.run_coverage(
        project_root=ROOT,
        diagnostic_output_root=one_row_diagnostic,
        coverage_output_root=output,
        allow_smoke=True,
        resume=True,
        jobs=7,
        row_timeout_seconds=1,
    )

    assert resumed == original
    assert final_path.read_bytes() == final_bytes
    assert sorted((output / "attempts").glob("*/*")) == attempt_dirs_before


def test_resume_rechecks_sealed_branch_replay_artifact_hashes(
    one_row_diagnostic: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "coverage"
    BATCH.run_coverage(
        project_root=ROOT,
        diagnostic_output_root=one_row_diagnostic,
        coverage_output_root=output,
        allow_smoke=True,
        jobs=1,
        row_timeout_seconds=60,
    )
    marker = json.loads(
        next((output / "completed").glob("*.json")).read_text(encoding="utf-8")
    )
    replay = output / marker["attempt_directory"] / "branch.replay.json"
    replay.write_bytes(replay.read_bytes() + b" ")

    with pytest.raises(BATCH.CoverageError, match="attempt artifact drifted"):
        BATCH.run_coverage(
            project_root=ROOT,
            diagnostic_output_root=one_row_diagnostic,
            coverage_output_root=output,
            allow_smoke=True,
            resume=True,
            jobs=1,
            row_timeout_seconds=60,
        )


def test_exact_full_plan_has_504_unique_keys_but_cannot_pass_without_all_markers(
    one_row_diagnostic: Path,
    tmp_path: Path,
) -> None:
    plan = json.loads(
        (one_row_diagnostic / "diagnostic.plan.json").read_text(encoding="utf-8")
    )
    plan.update(
        {
            "status": BATCH.FULL_PLAN_STATUS,
            "case_ids": list(BATCH.FULL_CASE_IDS),
            "seeds": list(BATCH.FULL_SEEDS),
            "arms": list(BATCH.FULL_ARMS),
            "charged_evaluation_budget": BATCH.FULL_BUDGET,
            "checkpoint_period": BATCH.FULL_CHECKPOINT_PERIOD,
            "expected_rows": BATCH.FULL_ROW_COUNT,
        }
    )
    diagnostic = tmp_path / "diagnostic"
    diagnostic.mkdir()
    (diagnostic / "diagnostic.plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output = tmp_path / "coverage"

    progress = BATCH.run_coverage(
        project_root=ROOT,
        diagnostic_output_root=diagnostic,
        coverage_output_root=output,
        jobs=1,
        row_timeout_seconds=60,
    )
    coverage_plan = json.loads(
        (output / "branch_replay_coverage.plan.json").read_text(encoding="utf-8")
    )
    keys = coverage_plan["row_keys_in_plan_order"]

    assert progress["status"] == BATCH.INCOMPLETE_STATUS
    assert progress["expected_rows"] == len(keys) == 504
    assert len(set(keys)) == 504
    assert keys[0] == (
        "v21e3-mokp-development-n100-s00__seed-31051__arm-c0_standard"
    )
    assert keys[-1] == (
        "v21e3-motsp-development-n500-s01__seed-31059__arm-moead_seeded_pop21"
    )
    assert not (output / "branch_replay_coverage.receipt.json").exists()
