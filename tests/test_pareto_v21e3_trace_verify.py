from __future__ import annotations

import json
import hashlib
import sqlite3
from pathlib import Path

import pytest

import mo_nco.pareto_v21e3_trace_verify as trace_verifier
from mo_nco.pareto_ijoc_problem import MultiObjectiveKnapsackInstance
from mo_nco.pareto_v21e3_hybrid import (
    V21E3HybridConfig,
    V21E3TypedHybridParetoSearch,
)
from mo_nco.pareto_v21e3_trace import (
    DecisionInput,
    EvaluationContext,
    ObjectiveContractError,
    V21E3SQLiteLedger,
    build_v21e3_run_context,
)


def _run_context(problem, *, budget: int):
    return build_v21e3_run_context(
        problem,
        case_artifact_sha256=hashlib.sha256(
            f"test-case:{problem.name}".encode("utf-8")
        ).hexdigest(),
        candidate_id="TEST",
        algorithm_config={"schema": "v21e3_verifier_test_v1", "budget": budget},
        algorithm_source_sha256=hashlib.sha256(b"v21e3-verifier-tests").hexdigest(),
        reference_directions=((0.5, 0.5),),
        seed=1,
        charged_evaluation_budget=budget,
        evidence_partition="calibration",
    )


def _detached_receipt_binding(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "detached_terminal_receipt_path": path,
        "expected_detached_terminal_receipt_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


from mo_nco.pareto_v21e3_trace_verify import (
    iter_v21e3_canonical_records,
    verify_v21e3_trace_database,
)


class _OutOfBoxFailureProblem:
    name = "v21e3-verifier-failure"
    num_objectives = 2
    solution_size = 1
    objective_lower_bounds = (0.0, 0.0)
    objective_upper_bounds = (1.0, 1.0)
    symmetric_proposal_contract = "test-only"

    def validate_solution(self, solution) -> None:
        if solution != (0,):
            raise ValueError("invalid")

    def evaluate(self, solution):
        return (2.0, 2.0)

    def random_solution(self, rng):
        return (0,)

    def propose(self, solution, rng):
        return (0,)

    def proposal_probability(self, source, target):
        return 1.0

    def canonical_payload(self):
        return {"name": self.name}


def _build_two_evaluation_trace(
    root: Path,
) -> tuple[Path, MultiObjectiveKnapsackInstance]:
    problem = MultiObjectiveKnapsackInstance(
        (1, 1),
        ((2, 1), (1, 2)),
        1,
        name="v21e3-chain-replay",
    )
    database_path = root / "chain.sqlite3"
    ledger = V21E3SQLiteLedger.from_problem(
        problem,
        run_context=_run_context(problem, budget=2),
        database_path=database_path,
        receipt_path=root / "chain.receipt.json",
    )
    for index, proposal in enumerate(((1, 0), (0, 1)), start=1):
        outcome = ledger.attempt(
            proposal,
            EvaluationContext(
                "calibration",
                "test",
                "test",
                index - 1,
                "op",
                index,
            ),
        )
        ledger.commit_decision(
            outcome.evaluation_index,
            DecisionInput(True, 1, (index - 1,), "test", True, True, index),
        )
    ledger.finalize(expected_charged_evaluations=2)
    return database_path, problem


def _build_one_evaluation_trace(
    root: Path,
) -> tuple[Path, Path, MultiObjectiveKnapsackInstance]:
    problem = MultiObjectiveKnapsackInstance(
        (1, 1),
        ((2, 1), (1, 2)),
        1,
        name="v21e3-decision-types",
    )
    database_path = root / "decision-types.sqlite3"
    receipt_path = root / "decision-types.receipt.json"
    ledger = V21E3SQLiteLedger.from_problem(
        problem,
        run_context=_run_context(problem, budget=1),
        database_path=database_path,
        receipt_path=receipt_path,
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
    return database_path, receipt_path, problem


def _coherently_rewrite_only_decision_field(
    database_path: Path,
    receipt_path: Path,
    *,
    field: str,
    value: object,
) -> None:
    with sqlite3.connect(database_path) as connection:
        payload = json.loads(
            connection.execute(
                "SELECT decision_json FROM decisions WHERE evaluation_index=1"
            ).fetchone()[0]
        )
        payload[field] = value
        decision_raw = _canonical_bytes(payload)
        decision_sha = hashlib.sha256(decision_raw).hexdigest()
        connection.execute(
            "UPDATE decisions SET decision_json=?,decision_sha256=? "
            "WHERE evaluation_index=1",
            (decision_raw.decode("utf-8"), decision_sha),
        )
        terminal = json.loads(receipt_path.read_text(encoding="utf-8"))
        terminal["terminal_decision_chain_sha256"] = decision_sha
        terminal.pop("receipt_payload_sha256")
        receipt_sha = hashlib.sha256(_canonical_bytes(terminal)).hexdigest()
        terminal["receipt_payload_sha256"] = receipt_sha
        receipt_raw = _canonical_bytes(terminal)
        connection.execute(
            "UPDATE terminal_receipts SET receipt_json=?,receipt_sha256=? "
            "WHERE run_id=1",
            (receipt_raw.decode("utf-8"), receipt_sha),
        )
        connection.execute(
            "UPDATE run_attempt SET terminal_receipt_sha256=? WHERE run_id=1",
            (receipt_sha,),
        )
        connection.commit()
    receipt_path.write_bytes(receipt_raw)


def _coherently_rewrite_terminal_count(
    database_path: Path,
    receipt_path: Path,
    *,
    field: str,
    value: object,
) -> None:
    terminal = json.loads(receipt_path.read_text(encoding="utf-8"))
    terminal[field] = value
    terminal.pop("receipt_payload_sha256")
    receipt_sha = hashlib.sha256(_canonical_bytes(terminal)).hexdigest()
    terminal["receipt_payload_sha256"] = receipt_sha
    receipt_raw = _canonical_bytes(terminal)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE terminal_receipts SET receipt_json=?,receipt_sha256=? "
            "WHERE run_id=1",
            (receipt_raw.decode("utf-8"), receipt_sha),
        )
        connection.execute(
            "UPDATE run_attempt SET terminal_receipt_sha256=? WHERE run_id=1",
            (receipt_sha,),
        )
        connection.commit()
    receipt_path.write_bytes(receipt_raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("archive_changed", 1),
        ("archive_size_after", True),
    ],
)
def test_v21e3_verifier_requires_exact_decision_boolean_and_integer_types(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    database_path, receipt_path, problem = _build_one_evaluation_trace(tmp_path)
    _coherently_rewrite_only_decision_field(
        database_path,
        receipt_path,
        field=field,
        value=value,
    )

    with pytest.raises(ValueError, match="decision.*exact type"):
        verify_v21e3_trace_database(
            database_path,
            problem,
            expected_run_context=_run_context(problem, budget=1).payload,
            **_detached_receipt_binding(receipt_path),
            expected_charged_evaluations=1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attempt_count", "1"),
        ("physical_call_started_count", True),
        ("charged_evaluation_count", True),
        ("decision_count", 1.0),
        ("cache_hit_count", "0"),
        ("unresolved_decision_count", 0.0),
    ],
)
def test_v21e3_verifier_requires_exact_terminal_count_types(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    database_path, receipt_path, problem = _build_one_evaluation_trace(tmp_path)
    _coherently_rewrite_terminal_count(
        database_path,
        receipt_path,
        field=field,
        value=value,
    )

    with pytest.raises(ValueError, match="terminal.*exact type"):
        verify_v21e3_trace_database(
            database_path,
            problem,
            expected_run_context=_run_context(problem, budget=1).payload,
            **_detached_receipt_binding(receipt_path),
            expected_charged_evaluations=1,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "physical_call_started_count",
            999,
            "terminal physical-start count binding failed",
        ),
        (
            "cache_hit_count",
            999,
            "terminal cache-hit count binding failed",
        ),
        (
            "unresolved_decision_count",
            999,
            "terminal unresolved-decision count binding failed",
        ),
    ],
)
def test_v21e3_verifier_binds_all_terminal_accounting_counts(
    tmp_path: Path,
    field: str,
    value: int,
    message: str,
) -> None:
    database_path, receipt_path, problem = _build_one_evaluation_trace(tmp_path)
    _coherently_rewrite_terminal_count(
        database_path,
        receipt_path,
        field=field,
        value=value,
    )

    with pytest.raises(ValueError, match=message):
        verify_v21e3_trace_database(
            database_path,
            problem,
            expected_run_context=_run_context(problem, budget=1).payload,
            **_detached_receipt_binding(receipt_path),
            expected_charged_evaluations=1,
        )


def _coherently_rewrite_second_charge_as_duplicate(
    database_path: Path, receipt_path: Path
) -> None:
    """Forge a fully rehashed ledger with one solution charged twice."""

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        # Strip the producer's defense-in-depth uniqueness index so this fixture
        # continues to exercise the independent verifier against a coherently
        # forged database rather than stopping at SQLite's write-time guard.
        connection.execute("DROP INDEX evaluations_unique_proposal_sha256")
        context_digest = str(
            connection.execute(
                "SELECT run_context_digest_sha256 FROM run_attempt WHERE run_id=1"
            ).fetchone()[0]
        )
        first_attempt = connection.execute(
            "SELECT * FROM attempts WHERE attempt_index=1"
        ).fetchone()
        connection.execute(
            """
            UPDATE attempts
            SET proposal_solution_ref=?,proposal_sha256=?,proposal_json=?,
                proposal_raw_sha256=?
            WHERE attempt_index=2
            """,
            (
                first_attempt["proposal_solution_ref"],
                first_attempt["proposal_sha256"],
                first_attempt["proposal_json"],
                first_attempt["proposal_raw_sha256"],
            ),
        )
        second_attempt = connection.execute(
            "SELECT * FROM attempts WHERE attempt_index=2"
        ).fetchone()
        attempt_semantic = {
            "attempt_index": 2,
            "proposal_solution_ref": int(second_attempt["proposal_solution_ref"]),
            "proposal_sha256": str(second_attempt["proposal_sha256"]),
            "proposal_raw": json.loads(str(second_attempt["proposal_json"])),
            "proposal_raw_sha256": str(second_attempt["proposal_raw_sha256"]),
            "evaluation_context": json.loads(str(second_attempt["context_json"])),
            "status": "EVALUATED",
            "physical_call_started": 1,
            "charged_evaluation_index": 2,
            "cache_source_evaluation_index": None,
            "failure_code": None,
            "failure_detail": None,
            "run_context_digest_sha256": context_digest,
            "prev_attempt_sha256": str(second_attempt["prev_attempt_sha256"]),
        }
        second_attempt_sha = hashlib.sha256(
            _canonical_bytes(attempt_semantic)
        ).hexdigest()
        connection.execute(
            "UPDATE attempts SET attempt_sha256=? WHERE attempt_index=2",
            (second_attempt_sha,),
        )

        first_evaluation = connection.execute(
            "SELECT * FROM evaluations WHERE evaluation_index=1"
        ).fetchone()
        connection.execute(
            """
            UPDATE evaluations
            SET proposal_solution_ref=?,proposal_sha256=?,objectives_json=?
            WHERE evaluation_index=2
            """,
            (
                first_evaluation["proposal_solution_ref"],
                first_evaluation["proposal_sha256"],
                first_evaluation["objectives_json"],
            ),
        )
        second_evaluation = connection.execute(
            "SELECT * FROM evaluations WHERE evaluation_index=2"
        ).fetchone()
        evaluation_semantic = {
            "evaluation_index": 2,
            "attempt_index": 2,
            "context": json.loads(str(second_attempt["context_json"])),
            "proposal_solution_ref": int(
                second_evaluation["proposal_solution_ref"]
            ),
            "proposal_sha256": str(second_evaluation["proposal_sha256"]),
            "objectives": tuple(
                float(value)
                for value in json.loads(str(second_evaluation["objectives_json"]))
            ),
            "run_context_digest_sha256": context_digest,
            "prev_record_sha256": str(second_evaluation["prev_record_sha256"]),
        }
        second_evaluation_sha = hashlib.sha256(
            _canonical_bytes(evaluation_semantic)
        ).hexdigest()
        connection.execute(
            "UPDATE evaluations SET record_sha256=? WHERE evaluation_index=2",
            (second_evaluation_sha,),
        )

        second_decision = connection.execute(
            "SELECT * FROM decisions WHERE evaluation_index=2"
        ).fetchone()
        decision_payload = json.loads(str(second_decision["decision_json"]))
        decision_payload["archive_changed"] = False
        decision_payload["retained_after_update"] = True
        decision_payload["archive_size_after"] = 1
        second_decision_sha = hashlib.sha256(
            _canonical_bytes(decision_payload)
        ).hexdigest()
        connection.execute(
            """
            UPDATE decisions
            SET decision_json=?,decision_sha256=?
            WHERE evaluation_index=2
            """,
            (
                _canonical_bytes(decision_payload).decode("utf-8"),
                second_decision_sha,
            ),
        )

        terminal = json.loads(
            str(
                connection.execute(
                    "SELECT receipt_json FROM terminal_receipts WHERE run_id=1"
                ).fetchone()[0]
            )
        )
        terminal["terminal_attempt_chain_sha256"] = second_attempt_sha
        terminal["terminal_evaluation_chain_sha256"] = second_evaluation_sha
        terminal["terminal_decision_chain_sha256"] = second_decision_sha
        terminal.pop("receipt_payload_sha256")
        terminal_sha = hashlib.sha256(_canonical_bytes(terminal)).hexdigest()
        terminal["receipt_payload_sha256"] = terminal_sha
        terminal_raw = _canonical_bytes(terminal)
        connection.execute(
            """
            UPDATE terminal_receipts
            SET receipt_json=?,receipt_sha256=? WHERE run_id=1
            """,
            (terminal_raw.decode("utf-8"), terminal_sha),
        )
        connection.execute(
            "UPDATE run_attempt SET terminal_receipt_sha256=? WHERE run_id=1",
            (terminal_sha,),
        )
        connection.commit()
    receipt_path.write_bytes(terminal_raw)


def test_v21e3_terminal_receipt_binds_replayable_semantic_chains(
    tmp_path: Path,
) -> None:
    database_path, problem = _build_two_evaluation_trace(tmp_path)
    with sqlite3.connect(database_path) as connection:
        terminal = json.loads(
            connection.execute(
                "SELECT receipt_json FROM terminal_receipts WHERE run_id=1"
            ).fetchone()[0]
        )

    assert terminal["terminal_evaluation_chain_sha256"] != "0" * 64
    assert terminal["terminal_decision_chain_sha256"] != "0" * 64
    assert terminal["terminal_attempt_chain_sha256"] != "0" * 64
    with sqlite3.connect(database_path) as connection:
        attempts = connection.execute(
            """
            SELECT prev_attempt_sha256,attempt_sha256
            FROM attempts ORDER BY attempt_index
            """
        ).fetchall()
        evaluations = connection.execute(
            """
            SELECT prev_record_sha256,record_sha256
            FROM evaluations ORDER BY evaluation_index
            """
        ).fetchall()
        decisions = connection.execute(
            """
            SELECT prev_decision_sha256,decision_sha256
            FROM decisions ORDER BY evaluation_index
            """
        ).fetchall()
    assert attempts[0][0] == "0" * 64
    assert attempts[1][0] == attempts[0][1]
    assert evaluations[0][0] == "0" * 64
    assert evaluations[1][0] == evaluations[0][1]
    assert decisions[0][0] == "0" * 64
    assert decisions[1][0] == decisions[0][1]
    assert terminal["terminal_evaluation_chain_sha256"] == evaluations[-1][1]
    assert terminal["terminal_decision_chain_sha256"] == decisions[-1][1]
    assert terminal["terminal_attempt_chain_sha256"] == attempts[-1][1]

    verification = verify_v21e3_trace_database(
        database_path,
        problem,
        expected_run_context=_run_context(problem, budget=2).payload,
        **_detached_receipt_binding(tmp_path / "chain.receipt.json"),
        expected_charged_evaluations=2,
    )
    assert verification["status"] == "OBJECTIVE_AND_ARCHIVE_REPLAY_PASS"
    assert verification["verification_scope"] == (
        "objective_solution_chain_archive_and_terminal_replay_v1"
    )
    assert verification["full_algorithm_decision_replay"] == "NOT_IMPLEMENTED"
    assert verification["selection_authorization"] == "PROHIBITED"
    assert verification["unique_solution_replays"] == 2
    assert verification["evaluation_records"] == 2
    assert verification["decision_records"] == 2
    assert verification["terminal_attempt_chain_sha256"] == attempts[-1][1]


def test_v21e3_verifier_rejects_duplicate_solution_as_a_second_charge(
    tmp_path: Path,
) -> None:
    database_path, problem = _build_two_evaluation_trace(tmp_path)
    receipt_path = tmp_path / "chain.receipt.json"
    _coherently_rewrite_second_charge_as_duplicate(database_path, receipt_path)

    with pytest.raises(ValueError, match="charged evaluation.*repeats|cache hit"):
        verify_v21e3_trace_database(
            database_path,
            problem,
            expected_run_context=_run_context(problem, budget=2).payload,
            **_detached_receipt_binding(receipt_path),
            expected_charged_evaluations=2,
        )


@pytest.mark.parametrize(
    ("attempt_index", "column", "value", "expected_error"),
    [
        (1, "cache_source_evaluation_index", 1, "evaluated.*invalid charge"),
        (2, "charged_evaluation_index", 1, "cached.*invalid charge"),
    ],
)
def test_v21e3_verifier_requires_mutually_exclusive_charge_fields(
    tmp_path: Path,
    attempt_index: int,
    column: str,
    value: int,
    expected_error: str,
) -> None:
    problem = MultiObjectiveKnapsackInstance(
        (1, 1), ((2, 1), (1, 2)), 1, name="v21e3-charge-field-exclusion"
    )
    database_path = tmp_path / "charge_fields.sqlite3"
    receipt_path = tmp_path / "charge_fields.receipt.json"
    ledger = V21E3SQLiteLedger.from_problem(
        problem,
        run_context=_run_context(problem, budget=1),
        database_path=database_path,
        receipt_path=receipt_path,
    )
    first = ledger.attempt(
        (1, 0), EvaluationContext("calibration", "test", "test", 0, "op", 1)
    )
    ledger.commit_decision(
        first.evaluation_index,
        DecisionInput(True, 1, (0,), "test", True, True, 1),
    )
    duplicate = ledger.attempt(
        (1, 0), EvaluationContext("calibration", "test", "test", 0, "op", 2)
    )
    assert duplicate.cache_hit
    ledger.finalize(expected_charged_evaluations=1)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            f"UPDATE attempts SET {column}=? WHERE attempt_index=?",
            (value, attempt_index),
        )
        connection.commit()

    with pytest.raises(ValueError, match=expected_error):
        verify_v21e3_trace_database(
            database_path,
            problem,
            expected_run_context=_run_context(problem, budget=1).payload,
            **_detached_receipt_binding(receipt_path),
            expected_charged_evaluations=1,
        )


def test_v21e3_verifier_requires_cache_source_exact_solution(
    tmp_path: Path,
) -> None:
    problem = MultiObjectiveKnapsackInstance(
        (1, 1), ((2, 1), (1, 2)), 1, name="v21e3-cache-source-exact"
    )
    database_path = tmp_path / "cache_source_exact.sqlite3"
    receipt_path = tmp_path / "cache_source_exact.receipt.json"
    ledger = V21E3SQLiteLedger.from_problem(
        problem,
        run_context=_run_context(problem, budget=2),
        database_path=database_path,
        receipt_path=receipt_path,
    )
    for operator_call_id, proposal in enumerate(((1, 0), (0, 1)), start=1):
        evaluated = ledger.attempt(
            proposal,
            EvaluationContext(
                "calibration", "test", "test", 0, "op", operator_call_id
            ),
        )
        ledger.commit_decision(
            evaluated.evaluation_index,
            DecisionInput(
                True, 1, (0,), "test", True, True, operator_call_id
            ),
        )
    cached = ledger.attempt(
        (1, 0), EvaluationContext("calibration", "test", "test", 0, "op", 3)
    )
    assert cached.cache_hit and cached.duplicate_of_evaluation_index == 1
    ledger.finalize(expected_charged_evaluations=2)

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM attempts WHERE attempt_index=3"
        ).fetchone()
        context_digest = str(
            connection.execute(
                "SELECT run_context_digest_sha256 FROM run_attempt WHERE run_id=1"
            ).fetchone()[0]
        )
        attempt_semantic = {
            "attempt_index": 3,
            "proposal_solution_ref": int(row["proposal_solution_ref"]),
            "proposal_sha256": str(row["proposal_sha256"]),
            "proposal_raw": json.loads(str(row["proposal_json"])),
            "proposal_raw_sha256": str(row["proposal_raw_sha256"]),
            "evaluation_context": json.loads(str(row["context_json"])),
            "status": "CACHE_HIT",
            "physical_call_started": 0,
            "charged_evaluation_index": None,
            "cache_source_evaluation_index": 2,
            "failure_code": None,
            "failure_detail": None,
            "run_context_digest_sha256": context_digest,
            "prev_attempt_sha256": str(row["prev_attempt_sha256"]),
        }
        attempt_sha = hashlib.sha256(
            _canonical_bytes(attempt_semantic)
        ).hexdigest()
        connection.execute(
            "UPDATE attempts SET cache_source_evaluation_index=2,attempt_sha256=? "
            "WHERE attempt_index=3",
            (attempt_sha,),
        )
        terminal = json.loads(
            str(
                connection.execute(
                    "SELECT receipt_json FROM terminal_receipts WHERE run_id=1"
                ).fetchone()[0]
            )
        )
        terminal["terminal_attempt_chain_sha256"] = attempt_sha
        terminal.pop("receipt_payload_sha256")
        terminal_sha = hashlib.sha256(_canonical_bytes(terminal)).hexdigest()
        terminal["receipt_payload_sha256"] = terminal_sha
        terminal_raw = _canonical_bytes(terminal)
        connection.execute(
            "UPDATE terminal_receipts SET receipt_json=?,receipt_sha256=? "
            "WHERE run_id=1",
            (terminal_raw.decode("utf-8"), terminal_sha),
        )
        connection.execute(
            "UPDATE run_attempt SET terminal_receipt_sha256=? WHERE run_id=1",
            (terminal_sha,),
        )
        connection.commit()
    receipt_path.write_bytes(terminal_raw)

    with pytest.raises(ValueError, match="cache hit references a different solution"):
        verify_v21e3_trace_database(
            database_path,
            problem,
            expected_run_context=_run_context(problem, budget=2).payload,
            **_detached_receipt_binding(receipt_path),
            expected_charged_evaluations=2,
        )


def test_v21e3r1_verifier_accepts_the_strict_v2_run_context(
    tmp_path: Path,
) -> None:
    problem = MultiObjectiveKnapsackInstance(
        (1, 1, 1, 1),
        ((9, 7, 6, 5), (5, 6, 8, 9)),
        2,
        name="v21e3r1-v2-replay",
    )
    database_path = tmp_path / "v21e3r1.sqlite3"
    receipt_path = tmp_path / "v21e3r1.receipt.json"
    optimizer = V21E3TypedHybridParetoSearch(
        problem,
        V21E3HybridConfig(
            candidate_id="C0",
            reference_directions=((0.5, 0.5),),
            charged_evaluations=3,
            checkpoint_period=1,
            seed=31,
            phase="development",
            trace_database=str(database_path),
            terminal_receipt=str(receipt_path),
        ),
    )
    optimizer.run()
    with sqlite3.connect(database_path) as connection:
        expected_context = json.loads(
            connection.execute(
                "SELECT run_context_json FROM run_attempt WHERE run_id=1"
            ).fetchone()[0]
        )

    verification = verify_v21e3_trace_database(
        database_path,
        problem,
        expected_run_context=expected_context,
        **_detached_receipt_binding(receipt_path),
        expected_charged_evaluations=3,
    )

    assert expected_context["schema"] == "v21e3r1_run_context_v2"
    assert verification["status"] == "OBJECTIVE_AND_ARCHIVE_REPLAY_PASS"


@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    [
        ("solution_payload", "solution fails its SHA-256 binding"),
        ("objective", "Objective replay failed"),
        ("decision", "Decision semantic hash chain failed"),
        ("terminal", "SQLite and detached V21e3 terminal receipts disagree"),
    ],
)
def test_v21e3_independent_verifier_rejects_semantic_tampering(
    tmp_path: Path,
    tamper: str,
    expected_error: str,
) -> None:
    database_path, problem = _build_two_evaluation_trace(tmp_path)
    with sqlite3.connect(database_path) as connection:
        if tamper == "solution_payload":
            connection.execute(
                "UPDATE solutions SET payload=? WHERE solution_ref=1",
                (sqlite3.Binary(b"\x00"),),
            )
        elif tamper == "objective":
            connection.execute(
                "UPDATE evaluations SET objectives_json='[0.0,0.0]' "
                "WHERE evaluation_index=1"
            )
        elif tamper == "decision":
            payload = json.loads(
                connection.execute(
                    "SELECT decision_json FROM decisions WHERE evaluation_index=1"
                ).fetchone()[0]
            )
            payload["decision_reason"] = "tampered"
            connection.execute(
                "UPDATE decisions SET decision_json=? WHERE evaluation_index=1",
                (json.dumps(payload, sort_keys=True, separators=(",", ":")),),
            )
        else:
            payload = json.loads(
                connection.execute(
                    "SELECT receipt_json FROM terminal_receipts WHERE run_id=1"
                ).fetchone()[0]
            )
            payload["attempt_count"] = 999
            connection.execute(
                "UPDATE terminal_receipts SET receipt_json=? WHERE run_id=1",
                (json.dumps(payload, sort_keys=True, separators=(",", ":")),),
            )
        connection.commit()

    with pytest.raises(ValueError, match=expected_error):
        verify_v21e3_trace_database(
            database_path,
            problem,
            expected_run_context=_run_context(problem, budget=2).payload,
            **_detached_receipt_binding(tmp_path / "chain.receipt.json"),
            expected_charged_evaluations=2,
        )


def test_v21e3_verifier_rejects_cache_hit_context_tampering(
    tmp_path: Path,
) -> None:
    problem = MultiObjectiveKnapsackInstance(
        (1, 1), ((2, 1), (1, 2)), 1, name="v21e3-cache-context-chain"
    )
    database_path = tmp_path / "cache_context.sqlite3"
    receipt_path = tmp_path / "cache_context.receipt.json"
    ledger = V21E3SQLiteLedger.from_problem(
        problem,
        run_context=_run_context(problem, budget=1),
        database_path=database_path,
        receipt_path=receipt_path,
    )
    first = ledger.attempt(
        (1, 0), EvaluationContext("calibration", "test", "test", 0, "op", 1)
    )
    ledger.commit_decision(
        first.evaluation_index,
        DecisionInput(True, 1, (0,), "test", True, True, 1),
    )
    duplicate = ledger.attempt(
        (1, 0), EvaluationContext("calibration", "test", "test", 0, "op", 2)
    )
    assert duplicate.cache_hit
    ledger.finalize(expected_charged_evaluations=1)
    trusted_detached_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()

    legal_verification = verify_v21e3_trace_database(
        database_path,
        problem,
        expected_run_context=_run_context(problem, budget=1).payload,
        detached_terminal_receipt_path=receipt_path,
        expected_detached_terminal_receipt_sha256=trusted_detached_hash,
        expected_charged_evaluations=1,
    )
    assert legal_verification["cache_hit_records"] == 1
    assert legal_verification["evaluation_records"] == 1

    with sqlite3.connect(database_path) as connection:
        context = json.loads(
            connection.execute(
                "SELECT context_json FROM attempts WHERE attempt_index=2"
            ).fetchone()[0]
        )
        context["operator_id"] = "tampered-cache-context"
        connection.execute(
            "UPDATE attempts SET context_json=? WHERE attempt_index=2",
            (json.dumps(context, sort_keys=True, separators=(",", ":")),),
        )
        connection.commit()

    with pytest.raises(ValueError, match="Attempt semantic hash chain failed"):
        verify_v21e3_trace_database(
            database_path,
            problem,
            expected_run_context=_run_context(problem, budget=1).payload,
            detached_terminal_receipt_path=receipt_path,
            expected_detached_terminal_receipt_sha256=trusted_detached_hash,
            expected_charged_evaluations=1,
        )

    # Model a coherent SQLite rewrite: update the attempt hash and every mutable
    # in-database terminal binding.  The previously frozen detached-receipt hash
    # must still reject the rewritten artifact.
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM attempts WHERE attempt_index=2"
        ).fetchone()
        run_context_digest = connection.execute(
            "SELECT run_context_digest_sha256 FROM run_attempt WHERE run_id=1"
        ).fetchone()[0]
        semantic = {
            "attempt_index": int(row["attempt_index"]),
            "proposal_solution_ref": int(row["proposal_solution_ref"]),
            "proposal_sha256": str(row["proposal_sha256"]),
            "proposal_raw": json.loads(str(row["proposal_json"])),
            "proposal_raw_sha256": str(row["proposal_raw_sha256"]),
            "evaluation_context": json.loads(str(row["context_json"])),
            "status": str(row["status"]),
            "physical_call_started": int(row["physical_call_started"]),
            "charged_evaluation_index": None,
            "cache_source_evaluation_index": int(
                row["cache_source_evaluation_index"]
            ),
            "failure_code": None,
            "failure_detail": None,
            "run_context_digest_sha256": str(run_context_digest),
            "prev_attempt_sha256": str(row["prev_attempt_sha256"]),
        }
        new_attempt_hash = hashlib.sha256(
            json.dumps(
                semantic,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        connection.execute(
            "UPDATE attempts SET attempt_sha256=? WHERE attempt_index=2",
            (new_attempt_hash,),
        )
        terminal = json.loads(receipt_path.read_text(encoding="utf-8"))
        terminal["terminal_attempt_chain_sha256"] = new_attempt_hash
        terminal.pop("receipt_payload_sha256")
        terminal_core_raw = json.dumps(
            terminal,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        new_payload_hash = hashlib.sha256(terminal_core_raw).hexdigest()
        terminal["receipt_payload_sha256"] = new_payload_hash
        rewritten_receipt_raw = json.dumps(
            terminal,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        connection.execute(
            "UPDATE terminal_receipts SET receipt_json=?,receipt_sha256=? "
            "WHERE run_id=1",
            (rewritten_receipt_raw.decode("utf-8"), new_payload_hash),
        )
        connection.execute(
            "UPDATE run_attempt SET terminal_receipt_sha256=? WHERE run_id=1",
            (new_payload_hash,),
        )
        connection.commit()
    receipt_path.write_bytes(rewritten_receipt_raw)

    with pytest.raises(ValueError, match="external SHA-256 binding"):
        verify_v21e3_trace_database(
            database_path,
            problem,
            expected_run_context=_run_context(problem, budget=1).payload,
            detached_terminal_receipt_path=receipt_path,
            expected_detached_terminal_receipt_sha256=trusted_detached_hash,
            expected_charged_evaluations=1,
        )


def test_v21e3_verifier_requires_the_exact_external_run_context(
    tmp_path: Path,
) -> None:
    database_path, problem = _build_two_evaluation_trace(tmp_path)
    wrong_context = _run_context(problem, budget=2).payload
    wrong_context["seed"] = int(wrong_context["seed"]) + 1

    with pytest.raises(ValueError, match="run-context binding"):
        verify_v21e3_trace_database(
            database_path,
            problem,
            expected_run_context=wrong_context,
            **_detached_receipt_binding(tmp_path / "chain.receipt.json"),
            expected_charged_evaluations=2,
        )


def test_v21e3_canonical_record_iterator_is_complete_and_json_safe(
    tmp_path: Path,
) -> None:
    database_path, _ = _build_two_evaluation_trace(tmp_path)

    records = list(iter_v21e3_canonical_records(database_path))

    assert [record["record_index"] for record in records] == list(
        range(1, len(records) + 1)
    )
    kinds = [record["record_kind"] for record in records]
    assert kinds.count("run_attempt") == 1
    assert kinds.count("solution") == 2
    assert kinds.count("attempt") == 2
    assert kinds.count("evaluation") == 2
    assert kinds.count("decision") == 2
    assert kinds.count("terminal_receipt") == 1
    solution_records = [
        record for record in records if record["record_kind"] == "solution"
    ]
    assert all("payload_hex" in record["row"] for record in solution_records)
    assert all("payload" not in record["row"] for record in solution_records)
    json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)


@pytest.mark.parametrize(
    "raw",
    ("[true,0.0]", '["1",0.0]', "[1.0]", "[1.0,0.0,2.0]"),
)
def test_v21e3_recorded_objectives_require_exact_json_numbers(raw: str) -> None:
    decoder = getattr(trace_verifier, "decode_v21e3_objectives_json")

    with pytest.raises(ValueError, match="exact JSON numbers"):
        decoder(raw, expected_dimension=2)


def test_v21e3_independent_verifier_accepts_bound_failure_receipt(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "failed.sqlite3"
    problem = _OutOfBoxFailureProblem()
    ledger = V21E3SQLiteLedger.from_problem(
        problem,
        run_context=_run_context(problem, budget=0),
        database_path=database_path,
        receipt_path=tmp_path / "failed.receipt.json",
    )
    with pytest.raises(ObjectiveContractError):
        ledger.attempt(
            (0,),
            EvaluationContext("calibration", "test", "test", 0, "op", 1),
        )

    verification = verify_v21e3_trace_database(
        database_path,
        problem,
        expected_run_context=_run_context(problem, budget=0).payload,
        **_detached_receipt_binding(tmp_path / "failed.receipt.json"),
        expected_charged_evaluations=0,
    )

    assert verification["status"] == "OBJECTIVE_AND_ARCHIVE_REPLAY_PASS"
    assert verification["full_algorithm_decision_replay"] == "NOT_IMPLEMENTED"
    assert verification["terminal_status"] == "FAILURE"
    assert verification["attempt_records"] == 1
    assert verification["evaluation_records"] == 0
    assert verification["decision_records"] == 0
    assert verification["terminal_evaluation_chain_sha256"] == "0" * 64
    assert verification["terminal_decision_chain_sha256"] == "0" * 64
