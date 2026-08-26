from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

import mo_nco.pareto_v21e3_trace as trace_module
from mo_nco.pareto_ijoc_problem import MultiObjectiveKnapsackInstance
from mo_nco.pareto_v21e3_trace import (
    DecisionInput,
    EvaluationContext,
    EvaluationContextError,
    ObjectiveContractError,
    ObjectiveEvaluationError,
    SolutionValidationError,
    V21E3SQLiteLedger,
    build_v21e3_run_context,
    recover_v21e3_terminal_receipt,
    V21E3RunContext,
)


REPOSITORY_ROOT = Path(__file__).parents[1]


def _run_context(problem, *, budget: int = 1):
    return build_v21e3_run_context(
        problem,
        case_artifact_sha256=hashlib.sha256(
            f"test-case:{problem.name}".encode("utf-8")
        ).hexdigest(),
        candidate_id="TEST",
        algorithm_config={"schema": "v21e3_trace_test_config_v1", "budget": budget},
        algorithm_source_sha256=hashlib.sha256(b"v21e3-trace-tests").hexdigest(),
        reference_directions=((0.5, 0.5),),
        seed=1,
        charged_evaluation_budget=budget,
        evidence_partition="calibration",
    )


class _ObjectiveContractProblem:
    name = "v21e3-objective-contract"
    num_objectives = 2
    solution_size = 1
    objective_lower_bounds = (0.0, 0.0)
    objective_upper_bounds = (1.0, 1.0)
    symmetric_proposal_contract = "test-only"

    def __init__(self, objective: object) -> None:
        self.objective = objective

    def random_solution(self, rng):
        return (0,)

    def propose(self, solution, rng):
        return (0,)

    def proposal_probability(self, source, target):
        return 1.0

    def validate_solution(self, solution) -> None:
        if solution != (0,):
            raise ValueError("invalid test solution")

    def evaluate(self, solution):
        return self.objective

    def canonical_payload(self):
        return {"name": self.name}


class _RaisingObjectiveProblem(_ObjectiveContractProblem):
    def evaluate(self, solution):
        raise RuntimeError("synthetic evaluator failure")


class _CountingObjectiveProblem(_ObjectiveContractProblem):
    def __init__(self, objective: object) -> None:
        super().__init__(objective)
        self.call_count = 0

    def evaluate(self, solution):
        self.call_count += 1
        return self.objective


def test_v21e3_evaluation_is_durable_before_return(tmp_path: Path) -> None:
    database_path = tmp_path / "evaluation_crash.sqlite3"
    child = """
import hashlib, os
from mo_nco.pareto_ijoc_problem import MultiObjectiveKnapsackInstance
from mo_nco.pareto_v21e3_trace import (
    EvaluationContext, V21E3SQLiteLedger, build_v21e3_run_context
)

problem = MultiObjectiveKnapsackInstance(
    (1, 1), ((2, 1), (1, 2)), 1, name="v21e3-durable-evaluation"
)
ledger = V21E3SQLiteLedger.from_problem(
    problem,
    run_context=build_v21e3_run_context(
        problem,
        case_artifact_sha256=hashlib.sha256(b"case").hexdigest(),
        candidate_id="TEST",
        algorithm_config={"budget": 1},
        algorithm_source_sha256=hashlib.sha256(b"source").hexdigest(),
        reference_directions=((0.5, 0.5),), seed=1,
        charged_evaluation_budget=1, evidence_partition="calibration",
    ),
    database_path=os.environ["V21E3_TEST_DATABASE"],
)
outcome = ledger.attempt(
    (1, 0),
    EvaluationContext("calibration", "test", "test", 0, "op", 1),
)
assert outcome.status == "EVALUATED"
assert outcome.charged_evaluation_index == 1
os._exit(17)
"""
    environment = {
        **os.environ,
        "PYTHONPATH": str(REPOSITORY_ROOT),
        "V21E3_TEST_DATABASE": str(database_path),
    }

    completed = subprocess.run(
        [sys.executable, "-c", child],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
    )

    assert completed.returncode == 17
    with sqlite3.connect(database_path) as connection:
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("solutions", "attempts", "evaluations", "decisions")
        }
    assert counts == {
        "solutions": 1,
        "attempts": 1,
        "evaluations": 1,
        "decisions": 0,
    }


def test_v21e3_decision_is_a_separate_durable_transaction(tmp_path: Path) -> None:
    database_path = tmp_path / "decision_crash.sqlite3"
    child = """
import hashlib, os
from mo_nco.pareto_ijoc_problem import MultiObjectiveKnapsackInstance
from mo_nco.pareto_v21e3_trace import (
    DecisionInput,
    EvaluationContext,
    V21E3SQLiteLedger,
    build_v21e3_run_context,
)

problem = MultiObjectiveKnapsackInstance(
    (1, 1), ((2, 1), (1, 2)), 1, name="v21e3-durable-decision"
)
ledger = V21E3SQLiteLedger.from_problem(
    problem,
    run_context=build_v21e3_run_context(
        problem,
        case_artifact_sha256=hashlib.sha256(b"case").hexdigest(),
        candidate_id="TEST", algorithm_config={"budget": 1},
        algorithm_source_sha256=hashlib.sha256(b"source").hexdigest(),
        reference_directions=((0.5, 0.5),), seed=1,
        charged_evaluation_budget=1, evidence_partition="calibration",
    ),
    database_path=os.environ["V21E3_TEST_DATABASE"],
)
outcome = ledger.attempt(
    (1, 0),
    EvaluationContext("calibration", "test", "test", 0, "op", 1),
)
ledger.commit_decision(
    outcome.evaluation_index,
    DecisionInput(True, 1, (0,), "test", True, True, 1),
)
os._exit(19)
"""
    environment = {
        **os.environ,
        "PYTHONPATH": str(REPOSITORY_ROOT),
        "V21E3_TEST_DATABASE": str(database_path),
    }

    completed = subprocess.run(
        [sys.executable, "-c", child],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
    )

    assert completed.returncode == 19
    with sqlite3.connect(database_path) as connection:
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("evaluations", "decisions")
        }
    assert counts == {"evaluations": 1, "decisions": 1}


def test_v21e3_dimension_failure_is_durable_and_terminal(tmp_path: Path) -> None:
    database_path = tmp_path / "dimension_failure.sqlite3"
    receipt_path = tmp_path / "dimension_failure.receipt.json"
    ledger = V21E3SQLiteLedger.from_problem(
        _ObjectiveContractProblem((0.5,)),
        run_context=_run_context(_ObjectiveContractProblem((0.5,))),
        database_path=database_path,
        receipt_path=receipt_path,
    )

    with pytest.raises(ObjectiveContractError, match="dimension") as captured:
        ledger.attempt(
            (0,),
            EvaluationContext("calibration", "test", "test", 0, "op", 1),
        )

    assert captured.value.code == "OBJECTIVE_DIMENSION_MISMATCH"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "FAILURE"
    assert receipt["failure_code"] == "OBJECTIVE_DIMENSION_MISMATCH"
    assert receipt["attempt_count"] == 1
    assert receipt["charged_evaluation_count"] == 0
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT status,failure_code FROM attempts"
        ).fetchone() == ("FAILED", "OBJECTIVE_DIMENSION_MISMATCH")
        assert connection.execute("SELECT COUNT(*) FROM evaluations").fetchone() == (
            0,
        )
        assert connection.execute(
            "SELECT status,failure_code FROM terminal_receipts"
        ).fetchone() == ("FAILURE", "OBJECTIVE_DIMENSION_MISMATCH")


@pytest.mark.parametrize(
    ("objective", "expected_code"),
    [
        ((True, 0.5), "OBJECTIVE_NONNUMERIC"),
        (("0.5", 0.5), "OBJECTIVE_NONNUMERIC"),
        ((float("nan"), 0.5), "OBJECTIVE_NONFINITE"),
        ((float("inf"), 0.5), "OBJECTIVE_NONFINITE"),
        ((float("-inf"), 0.5), "OBJECTIVE_NONFINITE"),
        ((-0.01, 0.5), "OBJECTIVE_OUT_OF_BOUNDS"),
        ((0.5, 1.01), "OBJECTIVE_OUT_OF_BOUNDS"),
    ],
)
def test_v21e3_nonfinite_and_box_fail_before_evaluation_record(
    tmp_path: Path,
    objective: object,
    expected_code: str,
) -> None:
    database_path = tmp_path / "invalid_objective.sqlite3"
    receipt_path = tmp_path / "invalid_objective.receipt.json"
    problem = _ObjectiveContractProblem(objective)
    ledger = V21E3SQLiteLedger.from_problem(
        problem,
        run_context=_run_context(problem),
        database_path=database_path,
        receipt_path=receipt_path,
    )

    with pytest.raises(ObjectiveContractError) as captured:
        ledger.attempt(
            (0,),
            EvaluationContext("calibration", "test", "test", 0, "op", 1),
        )

    assert captured.value.code == expected_code
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["failure_code"] == expected_code
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM evaluations").fetchone() == (
            0,
        )
        assert connection.execute(
            "SELECT status,failure_code FROM attempts"
        ).fetchone() == ("FAILED", expected_code)


def test_v21e3_evaluator_exception_is_a_durable_failed_attempt(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "evaluator_exception.sqlite3"
    receipt_path = tmp_path / "evaluator_exception.receipt.json"
    problem = _RaisingObjectiveProblem((0.5, 0.5))
    ledger = V21E3SQLiteLedger.from_problem(
        problem,
        run_context=_run_context(problem),
        database_path=database_path,
        receipt_path=receipt_path,
    )

    with pytest.raises(ObjectiveEvaluationError, match="synthetic evaluator failure"):
        ledger.attempt(
            (0,),
            EvaluationContext("calibration", "test", "test", 0, "op", 1),
        )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["failure_code"] == "OBJECTIVE_EVALUATOR_EXCEPTION"
    assert receipt["physical_call_started_count"] == 1
    assert receipt["charged_evaluation_count"] == 0
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT status,failure_code FROM attempts"
        ).fetchone() == ("FAILED", "OBJECTIVE_EVALUATOR_EXCEPTION")
        assert connection.execute("SELECT COUNT(*) FROM evaluations").fetchone() == (
            0,
        )


def test_v21e3_cache_hit_is_a_durable_zero_charge_attempt(tmp_path: Path) -> None:
    problem = _CountingObjectiveProblem((0.25, 0.75))
    database_path = tmp_path / "cache_hit.sqlite3"
    ledger = V21E3SQLiteLedger.from_problem(
        problem,
        run_context=_run_context(problem),
        database_path=database_path,
    )

    first = ledger.attempt(
        (0,),
        EvaluationContext("calibration", "test", "test", 0, "op", 1),
    )
    second = ledger.attempt(
        (0,),
        EvaluationContext("calibration", "test", "test", 0, "op", 2),
    )

    assert first.status == "EVALUATED"
    assert first.charged_evaluation_index == 1
    assert second.status == "CACHE_HIT"
    assert second.cache_hit is True
    assert second.charged_evaluation_index is None
    assert second.duplicate_of_evaluation_index == 1
    assert second.objectives == first.objectives
    assert problem.call_count == 1
    assert ledger.attempt_count == 2
    assert ledger.physical_call_count == 1
    assert ledger.evaluation_count == 1
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT status,physical_call_started FROM attempts "
            "ORDER BY attempt_index"
        ).fetchall() == [("EVALUATED", 1), ("CACHE_HIT", 0)]
        assert connection.execute("SELECT COUNT(*) FROM evaluations").fetchone() == (
            1,
        )


def test_v21e3_post_commit_fault_cannot_recharge_the_same_proposal(
    tmp_path: Path,
) -> None:
    problem = _CountingObjectiveProblem((0.25, 0.75))
    database_path = tmp_path / "post_commit_cache.sqlite3"
    armed = True

    def fault(boundary: str) -> None:
        nonlocal armed
        if armed and boundary == "after_evaluation_commit":
            armed = False
            raise RuntimeError("synthetic post-commit interruption")

    ledger = V21E3SQLiteLedger.from_problem(
        problem,
        run_context=_run_context(problem),
        database_path=database_path,
        fault_injector=fault,
    )

    with pytest.raises(RuntimeError, match="post-commit interruption"):
        ledger.attempt(
            (0,),
            EvaluationContext("calibration", "test", "test", 0, "op", 1),
        )

    duplicate = ledger.attempt(
        (0,),
        EvaluationContext("calibration", "test", "test", 0, "op", 2),
    )
    assert duplicate.status == "CACHE_HIT"
    assert duplicate.charged_evaluation_index is None
    assert duplicate.duplicate_of_evaluation_index == 1
    assert problem.call_count == 1
    ledger.commit_decision(
        1,
        DecisionInput(True, 1, (0,), "test", True, True, 1),
    )
    receipt = ledger.finalize(expected_charged_evaluations=1)
    assert receipt["charged_evaluation_count"] == 1
    assert receipt["cache_hit_count"] == 1


def test_v21e3_solution_digest_collision_cannot_alias_exact_solutions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = MultiObjectiveKnapsackInstance(
        (1, 1), ((2, 1), (1, 2)), 1, name="v21e3-collision-guard"
    )
    monkeypatch.setattr(
        trace_module,
        "_solution_sha256",
        lambda _solution: "f" * 64,
    )
    ledger = V21E3SQLiteLedger.from_problem(
        problem,
        run_context=_run_context(problem),
        database_path=tmp_path / "collision.sqlite3",
    )
    first = ledger.attempt(
        (1, 0),
        EvaluationContext("calibration", "test", "test", 0, "op", 1),
    )
    assert first.status == "EVALUATED"

    with pytest.raises(RuntimeError, match="SHA-256 collision"):
        ledger.attempt(
            (0, 1),
            EvaluationContext("calibration", "test", "test", 0, "op", 2),
        )


def test_v21e3_schema_rejects_duplicate_charged_proposal_sha(
    tmp_path: Path,
) -> None:
    problem = MultiObjectiveKnapsackInstance(
        (1, 1), ((2, 1), (1, 2)), 1, name="v21e3-charge-unique-index"
    )
    database_path = tmp_path / "charge_unique.sqlite3"
    ledger = V21E3SQLiteLedger.from_problem(
        problem,
        run_context=_run_context(problem, budget=2),
        database_path=database_path,
    )
    for operator_call_id, proposal in enumerate(((1, 0), (0, 1)), start=1):
        ledger.attempt(
            proposal,
            EvaluationContext(
                "calibration", "test", "test", 0, "op", operator_call_id
            ),
        )

    with sqlite3.connect(database_path) as connection:
        first_sha = connection.execute(
            "SELECT proposal_sha256 FROM evaluations WHERE evaluation_index=1"
        ).fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
            connection.execute(
                "UPDATE evaluations SET proposal_sha256=? WHERE evaluation_index=2",
                (first_sha,),
            )


def test_v21e3_success_finalize_writes_a_terminal_receipt(tmp_path: Path) -> None:
    database_path = tmp_path / "success.sqlite3"
    receipt_path = tmp_path / "success.receipt.json"
    problem = _ObjectiveContractProblem((0.25, 0.75))
    ledger = V21E3SQLiteLedger.from_problem(
        problem,
        run_context=_run_context(problem),
        database_path=database_path,
        receipt_path=receipt_path,
    )
    outcome = ledger.attempt(
        (0,),
        EvaluationContext("calibration", "test", "test", 0, "op", 1),
    )
    ledger.commit_decision(
        outcome.evaluation_index,
        DecisionInput(True, 1, (0,), "test", True, True, 1),
    )

    receipt = ledger.finalize(expected_charged_evaluations=1)

    assert receipt["status"] == "SUCCESS"
    assert receipt["failure_code"] is None
    assert receipt["attempt_count"] == 1
    assert receipt["charged_evaluation_count"] == 1
    assert receipt["decision_count"] == 1
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT status,failure_code FROM terminal_receipts"
        ).fetchone() == ("SUCCESS", None)
        assert connection.execute(
            "SELECT status FROM run_attempt WHERE run_id=1"
        ).fetchone() == ("SUCCESS",)


def test_v21e3_hard_crash_recovers_failure_receipt_with_unresolved_decision(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "recover_after_evaluation.sqlite3"
    receipt_path = tmp_path / "recover_after_evaluation.receipt.json"
    child = """
import hashlib, os
from mo_nco.pareto_ijoc_problem import MultiObjectiveKnapsackInstance
from mo_nco.pareto_v21e3_trace import (
    EvaluationContext, V21E3SQLiteLedger, build_v21e3_run_context
)

problem = MultiObjectiveKnapsackInstance(
    (1, 1), ((2, 1), (1, 2)), 1, name="v21e3-recovery"
)
ledger = V21E3SQLiteLedger.from_problem(
    problem,
    run_context=build_v21e3_run_context(
        problem,
        case_artifact_sha256=hashlib.sha256(b"case").hexdigest(),
        candidate_id="TEST", algorithm_config={"budget": 1},
        algorithm_source_sha256=hashlib.sha256(b"source").hexdigest(),
        reference_directions=((0.5, 0.5),), seed=1,
        charged_evaluation_budget=1, evidence_partition="calibration",
    ),
    database_path=os.environ["V21E3_TEST_DATABASE"],
)
ledger.attempt(
    (1, 0),
    EvaluationContext("calibration", "test", "test", 0, "op", 1),
)
os._exit(23)
"""
    completed = subprocess.run(
        [sys.executable, "-c", child],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(REPOSITORY_ROOT),
            "V21E3_TEST_DATABASE": str(database_path),
        },
        check=False,
    )
    assert completed.returncode == 23

    receipt = recover_v21e3_terminal_receipt(
        database_path,
        receipt_path,
        failure_code="PROCESS_INTERRUPTION_AFTER_EVALUATION",
        receipt_database_path="trace.sqlite3",
    )

    assert receipt["status"] == "FAILURE"
    assert receipt["failure_code"] == "PROCESS_INTERRUPTION_AFTER_EVALUATION"
    assert receipt["charged_evaluation_count"] == 1
    assert receipt["decision_count"] == 0
    assert receipt["unresolved_decision_count"] == 1
    assert receipt["recovered_after_interruption"] is True
    assert receipt["database_path"] == "trace.sqlite3"
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM evaluations").fetchone() == (
            1,
        )
        assert connection.execute("SELECT COUNT(*) FROM decisions").fetchone() == (
            0,
        )
        assert connection.execute(
            "SELECT status FROM run_attempt WHERE run_id=1"
        ).fetchone() == ("FAILURE",)

    assert (
        recover_v21e3_terminal_receipt(
            database_path,
            receipt_path,
            receipt_database_path="trace.sqlite3",
        )
        == receipt
    )


def test_v21e3_solution_validation_failure_retains_proposal_attempt(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "invalid_solution.sqlite3"
    receipt_path = tmp_path / "invalid_solution.receipt.json"
    ledger = V21E3SQLiteLedger.from_problem(
        _ObjectiveContractProblem((0.5, 0.5)),
        run_context=_run_context(_ObjectiveContractProblem((0.5, 0.5))),
        database_path=database_path,
        receipt_path=receipt_path,
    )

    with pytest.raises(SolutionValidationError, match="invalid test solution"):
        ledger.attempt(
            (1,),
            EvaluationContext("calibration", "test", "test", 0, "op", 1),
        )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["failure_code"] == "SOLUTION_VALIDATION_FAILED"
    assert receipt["attempt_count"] == 1
    assert receipt["physical_call_started_count"] == 0
    assert receipt["charged_evaluation_count"] == 0
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            """
            SELECT status,physical_call_started,proposal_solution_ref,failure_code
            FROM attempts
            """
        ).fetchone() == ("FAILED", 0, None, "SOLUTION_VALIDATION_FAILED")
        raw_json, raw_sha, solution_sha = connection.execute(
            """
            SELECT proposal_json,proposal_raw_sha256,proposal_sha256 FROM attempts
            """
        ).fetchone()
        assert json.loads(raw_json)["items"] == [{"kind": "int", "decimal": "1"}]
        assert hashlib.sha256(raw_json.encode("utf-8")).hexdigest() == raw_sha
        assert solution_sha is None
        assert connection.execute("SELECT COUNT(*) FROM evaluations").fetchone() == (
            0,
        )


@pytest.mark.parametrize("proposal", [(1.2,), object()], ids=["float", "object"])
def test_v21e3_raw_invalid_proposal_is_recorded_before_any_int_coercion(
    tmp_path: Path,
    proposal: object,
) -> None:
    problem = _ObjectiveContractProblem((0.5, 0.5))
    database_path = tmp_path / "raw_invalid.sqlite3"
    ledger = V21E3SQLiteLedger.from_problem(
        problem,
        run_context=_run_context(problem),
        database_path=database_path,
        receipt_path=tmp_path / "raw_invalid.receipt.json",
    )

    with pytest.raises(SolutionValidationError):
        ledger.attempt(
            proposal,  # type: ignore[arg-type]
            EvaluationContext("calibration", "test", "test", 0, "op", 1),
        )

    with sqlite3.connect(database_path) as connection:
        raw_json, raw_sha = connection.execute(
            "SELECT proposal_json,proposal_raw_sha256 FROM attempts"
        ).fetchone()
    raw = json.loads(raw_json)
    assert hashlib.sha256(raw_json.encode("utf-8")).hexdigest() == raw_sha
    if isinstance(proposal, tuple):
        assert raw["items"] == [
            {"kind": "float64", "ieee754_be_hex": "3ff3333333333333"}
        ]
    else:
        assert raw["iterable"] is False
        assert raw["container_type"] == "builtins.object"


def test_v21e3_partition_mismatch_is_durably_failed_before_evaluation(
    tmp_path: Path,
) -> None:
    problem = _ObjectiveContractProblem((0.5, 0.5))
    database_path = tmp_path / "partition_mismatch.sqlite3"
    receipt_path = tmp_path / "partition_mismatch.receipt.json"
    ledger = V21E3SQLiteLedger.from_problem(
        problem,
        run_context=_run_context(problem),
        database_path=database_path,
        receipt_path=receipt_path,
    )

    with pytest.raises(EvaluationContextError, match="evidence partition"):
        ledger.attempt(
            (0,),
            EvaluationContext("development", "test", "test", 0, "op", 1),
        )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["failure_code"] == "EVALUATION_CONTEXT_MISMATCH"
    assert receipt["attempt_count"] == 1
    assert receipt["physical_call_started_count"] == 0
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT status,failure_code FROM attempts"
        ).fetchone() == ("FAILED", "EVALUATION_CONTEXT_MISMATCH")
        assert connection.execute("SELECT COUNT(*) FROM evaluations").fetchone() == (
            0,
        )


@pytest.mark.parametrize(
    (
        "boundary",
        "expected_attempts",
        "expected_evaluations",
        "expected_decisions",
        "expected_terminal_rows",
        "expected_run_status",
        "expected_receipt_status",
    ),
    [
        ("after_run_start_commit", 0, 0, 0, 0, "STARTED", "FAILURE"),
        ("after_proposal_commit", 1, 0, 0, 0, "STARTED", "FAILURE"),
        ("after_objective_start_commit", 1, 0, 0, 0, "STARTED", "FAILURE"),
        ("after_raw_objective_return", 1, 0, 0, 0, "STARTED", "FAILURE"),
        ("before_evaluation_commit", 1, 0, 0, 0, "STARTED", "FAILURE"),
        ("after_evaluation_commit", 1, 1, 0, 0, "STARTED", "FAILURE"),
        ("before_decision_commit", 1, 1, 0, 0, "STARTED", "FAILURE"),
        ("after_decision_commit", 1, 1, 1, 0, "STARTED", "FAILURE"),
        ("before_terminal_commit", 1, 1, 1, 0, "STARTED", "FAILURE"),
        ("after_terminal_commit", 1, 1, 1, 1, "SUCCESS", "SUCCESS"),
    ],
)
def test_v21e3_crash_at_every_durable_boundary_is_recoverable(
    tmp_path: Path,
    boundary: str,
    expected_attempts: int,
    expected_evaluations: int,
    expected_decisions: int,
    expected_terminal_rows: int,
    expected_run_status: str,
    expected_receipt_status: str,
) -> None:
    database_path = tmp_path / f"{boundary}.sqlite3"
    receipt_path = tmp_path / f"{boundary}.receipt.json"
    child = """
import hashlib, os
from mo_nco.pareto_ijoc_problem import MultiObjectiveKnapsackInstance
from mo_nco.pareto_v21e3_trace import (
    DecisionInput,
    EvaluationContext,
    V21E3SQLiteLedger,
    build_v21e3_run_context,
)

target = os.environ["V21E3_FAULT_BOUNDARY"]
def crash(boundary):
    if boundary == target:
        os._exit(71)

problem = MultiObjectiveKnapsackInstance(
    (1, 1), ((2, 1), (1, 2)), 1, name="v21e3-boundary-crash"
)
ledger = V21E3SQLiteLedger.from_problem(
    problem,
    run_context=build_v21e3_run_context(
        problem,
        case_artifact_sha256=hashlib.sha256(b"case").hexdigest(),
        candidate_id="TEST", algorithm_config={"budget": 1},
        algorithm_source_sha256=hashlib.sha256(b"source").hexdigest(),
        reference_directions=((0.5, 0.5),), seed=1,
        charged_evaluation_budget=1, evidence_partition="calibration",
    ),
    database_path=os.environ["V21E3_TEST_DATABASE"],
    fault_injector=crash,
)
outcome = ledger.attempt(
    (1, 0),
    EvaluationContext("calibration", "test", "test", 0, "op", 1),
)
ledger.commit_decision(
    outcome.evaluation_index,
    DecisionInput(True, 1, (0,), "test", True, True, 1),
)
ledger.finalize(expected_charged_evaluations=1)
raise AssertionError("the requested fault boundary was not reached")
"""
    completed = subprocess.run(
        [sys.executable, "-c", child],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(REPOSITORY_ROOT),
            "V21E3_TEST_DATABASE": str(database_path),
            "V21E3_FAULT_BOUNDARY": boundary,
        },
        check=False,
    )
    assert completed.returncode == 71
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM attempts").fetchone() == (
            expected_attempts,
        )
        assert connection.execute("SELECT COUNT(*) FROM evaluations").fetchone() == (
            expected_evaluations,
        )
        assert connection.execute("SELECT COUNT(*) FROM decisions").fetchone() == (
            expected_decisions,
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM terminal_receipts"
        ).fetchone() == (expected_terminal_rows,)
        assert connection.execute(
            "SELECT status FROM run_attempt WHERE run_id=1"
        ).fetchone() == (expected_run_status,)

    receipt = recover_v21e3_terminal_receipt(
        database_path,
        receipt_path,
        failure_code=f"FAULT_INJECTED_{boundary.upper()}",
    )
    assert receipt["status"] == expected_receipt_status
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt


def _v21e3r1_context_payload() -> dict[str, object]:
    algorithm_config = {
        "candidate_id": "C3",
        "reference_directions": ((0.2, 0.8), (0.8, 0.2)),
        "charged_evaluations": 9,
        "seed": 99,
        "phase": "development",
    }
    raw = json.dumps(
        algorithm_config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema": "v21e3r1_run_context_v2",
        "case_artifact_sha256": "a" * 64,
        "problem_semantic_sha256": "b" * 64,
        "candidate_id": "C3",
        "algorithm_config": algorithm_config,
        "candidate_config_sha256": hashlib.sha256(raw).hexdigest(),
        "algorithm_source_sha256": "c" * 64,
        "reference_directions": ((0.2, 0.8), (0.8, 0.2)),
        "seed": 99,
        "charged_evaluation_budget": 9,
        "evidence_partition": "development",
    }


@pytest.mark.parametrize(
    ("outer_key", "contradictory_value"),
    [
        ("candidate_id", "C0"),
        ("seed", 100),
        ("charged_evaluation_budget", 10),
        ("evidence_partition", "calibration"),
        ("reference_directions", ((0.5, 0.5),)),
    ],
)
def test_v21e3r1_context_rejects_every_contradictory_mirror(
    outer_key: str,
    contradictory_value: object,
) -> None:
    payload = _v21e3r1_context_payload()
    payload[outer_key] = contradictory_value

    with pytest.raises(ValueError, match="inconsistent mirrored field"):
        V21E3RunContext(payload)


@pytest.mark.parametrize(
    "missing_config_key",
    [
        "candidate_id",
        "seed",
        "charged_evaluations",
        "phase",
        "reference_directions",
    ],
)
def test_v21e3r1_context_v2_requires_every_authoritative_mirror(
    missing_config_key: str,
) -> None:
    payload = _v21e3r1_context_payload()
    algorithm_config = dict(payload["algorithm_config"])
    del algorithm_config[missing_config_key]
    payload["algorithm_config"] = algorithm_config
    payload["candidate_config_sha256"] = hashlib.sha256(
        json.dumps(
            algorithm_config,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="missing mirrored field"):
        V21E3RunContext(payload)


def test_v21e3_legacy_v1_context_rejects_a_present_contradictory_mirror() -> None:
    payload = _v21e3r1_context_payload()
    payload["schema"] = "v21e3_run_context_v1"
    algorithm_config = dict(payload["algorithm_config"])
    algorithm_config["candidate_id"] = "C0"
    payload["algorithm_config"] = algorithm_config
    payload["candidate_config_sha256"] = hashlib.sha256(
        json.dumps(
            algorithm_config,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="inconsistent mirrored field"):
        V21E3RunContext(payload)
