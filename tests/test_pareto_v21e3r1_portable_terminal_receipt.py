from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from mo_nco.pareto_ijoc_problem import MultiObjectiveKnapsackInstance
from mo_nco.pareto_v21e3_trace import (
    build_v21e3_run_context,
    DecisionInput,
    EvaluationContext,
    V21E3SQLiteLedger,
)


def _problem() -> MultiObjectiveKnapsackInstance:
    return MultiObjectiveKnapsackInstance(
        item_weights=(1, 2),
        profits_by_objective=((2, 1), (1, 3)),
        capacity=2,
        name="portable-terminal-fixture",
    )


def _context(problem: MultiObjectiveKnapsackInstance):
    import hashlib

    return build_v21e3_run_context(
        problem,
        case_artifact_sha256=hashlib.sha256(b"portable-case").hexdigest(),
        candidate_id="PORTABLE_TEST",
        algorithm_config={"fixture": True},
        algorithm_source_sha256=hashlib.sha256(b"portable-source").hexdigest(),
        reference_directions=((0.5, 0.5),),
        seed=31051,
        charged_evaluation_budget=1,
        evidence_partition="development",
    )


def test_terminal_receipt_can_bind_a_portable_database_display_path(
    tmp_path: Path,
) -> None:
    problem = _problem()
    database = tmp_path / "trace.sqlite3"
    terminal = tmp_path / "terminal.receipt.json"
    ledger = V21E3SQLiteLedger.from_problem(
        problem,
        run_context=_context(problem),
        database_path=database,
        receipt_path=terminal,
        receipt_database_path="trace.sqlite3",
    )
    outcome = ledger.attempt(
        (1, 0),
        EvaluationContext("development", "test", "test", 0, "fixture", 1),
    )
    assert outcome.evaluation_index == 1
    ledger.commit_decision(
        1,
        DecisionInput(True, 1, (0,), "test", True, True, 1),
    )

    receipt = ledger.finalize(expected_charged_evaluations=1)

    assert receipt["database_path"] == "trace.sqlite3"
    assert str(tmp_path) not in terminal.read_text(encoding="utf-8")
    with sqlite3.connect(database) as connection:
        embedded = json.loads(
            connection.execute(
                "SELECT receipt_json FROM terminal_receipts WHERE run_id=1"
            ).fetchone()[0]
        )
    assert embedded == receipt


def test_terminal_receipt_rejects_noncanonical_or_escaping_display_paths(
    tmp_path: Path,
) -> None:
    problem = _problem()
    for bad in (
        "../trace.sqlite3",
        "rows\\trace.sqlite3",
        "/trace.sqlite3",
        "C:/trace.sqlite3",
    ):
        try:
            V21E3SQLiteLedger.from_problem(
                problem,
                run_context=_context(problem),
                database_path=tmp_path / (bad.replace("/", "_").replace("\\", "_")),
                receipt_database_path=bad,
            )
        except ValueError:
            continue
        raise AssertionError(f"accepted unsafe receipt_database_path: {bad}")

