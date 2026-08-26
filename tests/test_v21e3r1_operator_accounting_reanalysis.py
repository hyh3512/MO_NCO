from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from independent_reproduction import reanalyze_v21e3r1_operator_accounting as reanalysis


HEX64 = "a" * 64


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object, *, canonical: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if canonical:
        path.write_bytes(_canonical_bytes(value))
    else:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def _context(
    operator: str,
    *,
    retry_ordinal: int,
    fallback_used: bool = False,
) -> str:
    return _canonical_bytes(
        {
            "evidence_partition": "development",
            "operator_call_id": retry_ordinal + 1,
            "operator_id": operator,
            "operator_witness": {
                "fallback_used": fallback_used,
                "retry_ordinal": retry_ordinal,
            },
            "search_phase_id": "native_backbone",
            "stage_id": "search_v21e3",
            "type_id": 0,
        }
    ).decode("utf-8")


def _decision(index: int, *, changed: bool, accepted: bool) -> str:
    return _canonical_bytes(
        {
            "accepted_into_population": accepted,
            "archive_changed": changed,
            "evaluation_index": index,
            "new_evaluated_cell": changed,
            "new_nondominated_cell": False,
            "population_replacement_count": int(accepted),
            "retained_after_update": changed,
            "scalar_advantage": 0.25 if accepted else None,
        }
    ).decode("utf-8")


def _create_trace(
    path: Path,
    *,
    source_sha256: str,
    invalid_retry_type: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run_context = {
        "algorithm_source_sha256": source_sha256,
        "charged_evaluation_budget": 2,
        "evidence_partition": "development",
        "family": "MOKP",
        "problem": "v21e3-mokp-development-n100-s00",
        "schema": "v21e3r1_run_context_v2",
    }
    run_context_raw = _canonical_bytes(run_context).decode("utf-8")
    run_context_sha = hashlib.sha256(run_context_raw.encode("utf-8")).hexdigest()
    terminal_core = {
        "attempt_count": 3,
        "cache_hit_count": 1,
        "charged_evaluation_count": 2,
        "decision_count": 2,
        "family": "MOKP",
        "physical_call_started_count": 2,
        "problem": "v21e3-mokp-development-n100-s00",
        "run_context_digest_sha256": run_context_sha,
        "status": "SUCCESS",
    }
    terminal_sha = hashlib.sha256(_canonical_bytes(terminal_core)).hexdigest()
    terminal = {**terminal_core, "receipt_payload_sha256": terminal_sha}
    terminal_raw = _canonical_bytes(terminal).decode("utf-8")
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE solutions(
              solution_ref INTEGER PRIMARY KEY,
              solution_sha256 TEXT NOT NULL UNIQUE,
              family TEXT NOT NULL,
              codec TEXT NOT NULL,
              solution_size INTEGER NOT NULL,
              payload BLOB NOT NULL
            );
            CREATE TABLE attempts(
              attempt_index INTEGER PRIMARY KEY,
              proposal_solution_ref INTEGER,
              proposal_sha256 TEXT,
              proposal_json TEXT NOT NULL,
              proposal_raw_sha256 TEXT NOT NULL,
              context_json TEXT NOT NULL,
              status TEXT NOT NULL,
              physical_call_started INTEGER NOT NULL,
              charged_evaluation_index INTEGER,
              cache_source_evaluation_index INTEGER,
              failure_code TEXT,
              failure_detail_json TEXT,
              elapsed_monotonic_ns INTEGER NOT NULL,
              prev_attempt_sha256 TEXT,
              attempt_sha256 TEXT UNIQUE
            );
            CREATE TABLE evaluations(
              evaluation_index INTEGER PRIMARY KEY,
              attempt_index INTEGER NOT NULL UNIQUE,
              evidence_partition TEXT NOT NULL,
              search_phase_id TEXT NOT NULL,
              stage_id TEXT NOT NULL,
              type_id INTEGER,
              operator_id TEXT NOT NULL,
              operator_call_id INTEGER NOT NULL,
              proposal_solution_ref INTEGER NOT NULL,
              proposal_sha256 TEXT NOT NULL,
              objectives_json TEXT NOT NULL,
              prev_record_sha256 TEXT NOT NULL,
              record_sha256 TEXT NOT NULL UNIQUE
            );
            CREATE TABLE decisions(
              evaluation_index INTEGER PRIMARY KEY,
              decision_json TEXT NOT NULL,
              prev_decision_sha256 TEXT NOT NULL,
              decision_sha256 TEXT NOT NULL UNIQUE
            );
            CREATE TABLE run_attempt(
              run_id INTEGER PRIMARY KEY,
              problem TEXT NOT NULL,
              family TEXT NOT NULL,
              run_context_json TEXT NOT NULL,
              run_context_digest_sha256 TEXT NOT NULL,
              status TEXT NOT NULL,
              terminal_receipt_sha256 TEXT
            );
            CREATE TABLE terminal_receipts(
              run_id INTEGER PRIMARY KEY,
              status TEXT NOT NULL,
              failure_code TEXT,
              receipt_json TEXT NOT NULL,
              receipt_sha256 TEXT NOT NULL UNIQUE
            );
            """
        )
        first = _context("construct", retry_ordinal=0)
        cached = _context("neighbor", retry_ordinal=0)
        retry = json.loads(_context("retry", retry_ordinal=1))
        if invalid_retry_type:
            retry["operator_witness"]["retry_ordinal"] = True
        retry_raw = _canonical_bytes(retry).decode("utf-8")
        connection.executemany(
            "INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                (1, 1, "1" * 64, "[1]", "1" * 64, first, "EVALUATED", 1, 1, None, None, None, 1, "0" * 64, "a" * 64),
                (2, 1, "1" * 64, "[1]", "1" * 64, cached, "CACHE_HIT", 0, None, 1, None, None, 2, "a" * 64, "b" * 64),
                (3, 2, "2" * 64, "[2]", "2" * 64, retry_raw, "EVALUATED", 1, 2, None, None, None, 3, "b" * 64, "c" * 64),
            ),
        )
        connection.executemany(
            "INSERT INTO evaluations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                (1, 1, "development", "native_backbone", "search_v21e3", 0, "construct", 1, 1, "1" * 64, "[0.1,0.9]", "0" * 64, "d" * 64),
                (2, 3, "development", "native_backbone", "search_v21e3", 0, "retry", 2, 2, "2" * 64, "[0.2,0.8]", "d" * 64, "e" * 64),
            ),
        )
        connection.executemany(
            "INSERT INTO decisions VALUES (?,?,?,?)",
            (
                (1, _decision(1, changed=True, accepted=True), "0" * 64, "f" * 64),
                (2, _decision(2, changed=False, accepted=False), "f" * 64, "9" * 64),
            ),
        )
        connection.execute(
            "INSERT INTO run_attempt VALUES (1,?,?,?,?,?,?)",
            (
                "v21e3-mokp-development-n100-s00",
                "MOKP",
                run_context_raw,
                run_context_sha,
                "SUCCESS",
                terminal_sha,
            ),
        )
        connection.execute(
            "INSERT INTO terminal_receipts VALUES (1,'SUCCESS',NULL,?,?)",
            (terminal_raw, terminal_sha),
        )
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.commit()


def _make_sealed_smoke(
    root: Path,
    *,
    invalid_retry_type: bool = False,
) -> Path:
    diagnostic = root / "diagnostic"
    attempt_rel = (
        "attempts/v21e3-mokp-development-n100-s00__seed-31051__arm-c0_standard/"
        "attempt-0001"
    )
    attempt = diagnostic / attempt_rel
    trace = attempt / "trace.sqlite3"
    source_entries = [{"bytes": 1, "path": "mo_nco/frozen.py", "sha256": HEX64}]
    source_sha = hashlib.sha256(_canonical_bytes(source_entries)).hexdigest()
    _create_trace(
        trace,
        source_sha256=source_sha,
        invalid_retry_type=invalid_retry_type,
    )

    with sqlite3.connect(trace) as connection:
        terminal = json.loads(
            connection.execute(
                "SELECT receipt_json FROM terminal_receipts WHERE run_id=1"
            ).fetchone()[0]
        )
    _write_json(attempt / "terminal.receipt.json", terminal, canonical=True)
    diagnostic_row = {
        "arm_id": "C0_STANDARD",
        "attempt_count": 3,
        "budget": 2,
        "cache_hit_count": 1,
        "case_id": "v21e3-mokp-development-n100-s00",
        "charged_evaluation_count": 2,
        "family": "MOKP",
        "operators": {
            "construct": {"charged_evaluations": 2.0},
            "neighbor": {"charged_evaluations": 0.0},
            "retry": {"charged_evaluations": 2.0},
        },
        "physical_start_count": 2,
        "schema": "v21e3r1_existing_trace_diagnostic_v1",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "seed": 31051,
        "size": 100,
        "status": "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY",
    }
    _write_json(attempt / "diagnostic.json", diagnostic_row)
    _write_json(attempt / "independent.metric.json", {"status": "PASS"})

    plan = {
        "arms": ["C0_STANDARD"],
        "case_ids": ["v21e3-mokp-development-n100-s00"],
        "charged_evaluation_budget": 2,
        "checkpoint_period": 1,
        "confirmation_materialization": "PROHIBITED",
        "expected_rows": 1,
        "formal_materialization": "PROHIBITED",
        "input_binding": {"schema": "fixture"},
        "row_timeout_seconds": 60,
        "schema": "v21e3r1_exposed_development_diagnostic_plan_v2",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "seeds": [31051],
        "selection_entropy_release": "PROHIBITED",
        "source_manifest": {
            "entries": source_entries,
            "entry_count": 1,
            "hash_rule": "sha256(canonical_json(sorted_entries))",
            "schema": "v21e3r1_diagnostic_source_manifest_v1",
            "source_snapshot_sha256": source_sha,
        },
        "status": "FROZEN_DIAGNOSTIC_SMOKE_ONLY",
    }
    _write_json(diagnostic / "diagnostic.plan.json", plan)
    plan_sha = _sha256(diagnostic / "diagnostic.plan.json")
    row = {
        "arm_id": "C0_STANDARD",
        "case_id": "v21e3-mokp-development-n100-s00",
        "charged_evaluation_budget": 2,
        "family": "MOKP",
        "independent_metric_receipt_path": "independent.metric.json",
        "independent_metric_receipt_sha256": _sha256(attempt / "independent.metric.json"),
        "plan_sha256": plan_sha,
        "schema": "v21e3r1_exposed_development_diagnostic_row_v2",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "seed": 31051,
        "selection_entropy_release": "PROHIBITED",
        "confirmation_materialization": "PROHIBITED",
        "formal_materialization": "PROHIBITED",
        "source_snapshot_sha256": source_sha,
        "status": "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "terminal_receipt_path": "terminal.receipt.json",
        "terminal_receipt_sha256": _sha256(attempt / "terminal.receipt.json"),
        "trace_database_path": "trace.sqlite3",
        "trace_database_sha256": _sha256(trace),
    }
    _write_json(attempt / "row.json", row)
    row_id = "v21e3-mokp-development-n100-s00__seed-31051__arm-c0_standard"
    completed = {
        "attempt_directory": attempt_rel,
        "diagnostic_sha256": _sha256(attempt / "diagnostic.json"),
        "independent_metric_receipt_sha256": _sha256(attempt / "independent.metric.json"),
        "plan_sha256": plan_sha,
        "row_id": row_id,
        "row_sha256": _sha256(attempt / "row.json"),
        "status": "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "terminal_receipt_sha256": _sha256(attempt / "terminal.receipt.json"),
        "trace_sha256": _sha256(trace),
    }
    _write_json(diagnostic / "completed" / f"{row_id}.json", completed)
    aggregate = {"matrix_mode": "SMOKE_ONLY", "rows": 1}
    _write_json(diagnostic / "diagnostic.aggregate.json", aggregate)
    receipt = {
        "aggregate_sha256": _sha256(diagnostic / "diagnostic.aggregate.json"),
        "completed_rows": 1,
        "confirmation_materialization": "PROHIBITED",
        "expected_rows": 1,
        "formal_materialization": "PROHIBITED",
        "matrix_mode": "SMOKE_ONLY",
        "plan_sha256": plan_sha,
        "schema": "v21e3r1_exposed_development_diagnostic_receipt_v2",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "selection_entropy_release": "PROHIBITED",
        "source_snapshot_sha256": source_sha,
        "status": "PASS_DIAGNOSTIC_SMOKE_ONLY",
    }
    _write_json(diagnostic / "diagnostic.receipt.json", receipt)
    return diagnostic


def test_corrects_double_count_and_exposes_primary_ordinal_zero(tmp_path: Path) -> None:
    diagnostic = _make_sealed_smoke(tmp_path)
    before = {
        path.relative_to(diagnostic).as_posix(): _sha256(path)
        for path in diagnostic.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "reanalysis"

    receipt = reanalysis.reanalyze(diagnostic, output, allow_smoke=True)

    assert receipt["status"] == "PASS_CORRECTED_REANALYSIS_SMOKE_ONLY"
    assert receipt["evaluation_charged_evaluations_sum"] == 2
    assert receipt["attempt_charged_evaluations_sum"] == 2
    assert receipt["legacy_operator_charged_evaluations_sum"] == 4
    assert receipt["operator_charge_double_count_corrected"] is True
    assert receipt["original_artifacts_modified"] is False
    assert receipt["implementation_independence"] is False
    assert receipt["scientific_independence"] is False
    assert receipt["publication_status"] == "IJOC_HOLD"
    rows_path = output / str(receipt["rows_path"])
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    operators = rows[0]["operators"]
    assert operators["construct"]["evaluation_charged_evaluations"] == 1
    assert operators["construct"]["attempt_charged_evaluations"] == 1
    assert operators["construct"]["primary_ordinal_zero_attempts"] == 1
    assert operators["neighbor"]["evaluation_charged_evaluations"] == 0
    assert operators["neighbor"]["primary_ordinal_zero_cache_hits"] == 1
    assert operators["retry"]["primary_ordinal_zero_attempts"] == 0
    assert receipt["rows_sha256"] == _sha256(rows_path)
    assert receipt["aggregate_sha256"] == _sha256(
        output / str(receipt["aggregate_path"])
    )
    after = {
        path.relative_to(diagnostic).as_posix(): _sha256(path)
        for path in diagnostic.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_full_mode_rejects_smoke_before_creating_output(tmp_path: Path) -> None:
    diagnostic = _make_sealed_smoke(tmp_path)
    output = tmp_path / "reanalysis"

    with pytest.raises(reanalysis.ContractError, match="exact frozen 504"):
        reanalysis.reanalyze(diagnostic, output)

    assert not output.exists()


def test_output_directory_is_exclusive_and_existing_bytes_survive(tmp_path: Path) -> None:
    diagnostic = _make_sealed_smoke(tmp_path)
    output = tmp_path / "reanalysis"
    receipt = reanalysis.reanalyze(diagnostic, output, allow_smoke=True)
    receipt_path = output / reanalysis.RECEIPT_NAME
    original = receipt_path.read_bytes()

    with pytest.raises(FileExistsError):
        reanalysis.reanalyze(diagnostic, output, allow_smoke=True)

    assert receipt_path.read_bytes() == original
    assert json.loads(original) == receipt


def test_rejects_tampered_trace_hash_binding(tmp_path: Path) -> None:
    diagnostic = _make_sealed_smoke(tmp_path)
    completed_path = next((diagnostic / "completed").glob("*.json"))
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    completed["trace_sha256"] = "0" * 64
    _write_json(completed_path, completed)
    output = tmp_path / "reanalysis"

    with pytest.raises(reanalysis.ContractError, match="trace SHA-256"):
        reanalysis.reanalyze(diagnostic, output, allow_smoke=True)

    assert not output.exists()


def test_rejects_non_exact_retry_ordinal_type(tmp_path: Path) -> None:
    diagnostic = _make_sealed_smoke(tmp_path, invalid_retry_type=True)
    output = tmp_path / "reanalysis"

    with pytest.raises(reanalysis.ContractError, match="retry_ordinal"):
        reanalysis.reanalyze(diagnostic, output, allow_smoke=True)

    assert not output.exists()


def test_rejects_missing_final_receipt(tmp_path: Path) -> None:
    diagnostic = _make_sealed_smoke(tmp_path)
    (diagnostic / "diagnostic.receipt.json").unlink()
    output = tmp_path / "reanalysis"

    with pytest.raises(reanalysis.ContractError, match="diagnostic.receipt.json"):
        reanalysis.reanalyze(diagnostic, output, allow_smoke=True)

    assert not output.exists()


def test_rejects_extra_completed_row(tmp_path: Path) -> None:
    diagnostic = _make_sealed_smoke(tmp_path)
    extra = diagnostic / "completed" / "extra.json"
    _write_json(extra, {"status": "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY"})
    output = tmp_path / "reanalysis"

    with pytest.raises(reanalysis.ContractError, match="completed row set"):
        reanalysis.reanalyze(diagnostic, output, allow_smoke=True)

    assert not output.exists()


def test_receipt_and_aggregate_are_canonical_json_without_newline(tmp_path: Path) -> None:
    diagnostic = _make_sealed_smoke(tmp_path)
    output = tmp_path / "reanalysis"
    receipt = reanalysis.reanalyze(diagnostic, output, allow_smoke=True)

    receipt_path = output / reanalysis.RECEIPT_NAME
    aggregate_path = output / reanalysis.AGGREGATE_NAME
    assert receipt_path.read_bytes() == _canonical_bytes(receipt)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert aggregate_path.read_bytes() == _canonical_bytes(aggregate)


def test_receipt_exact_key_contract(tmp_path: Path) -> None:
    diagnostic = _make_sealed_smoke(tmp_path)
    receipt = reanalysis.reanalyze(
        diagnostic, tmp_path / "reanalysis", allow_smoke=True
    )

    assert set(receipt) == reanalysis.RECEIPT_KEYS


def test_cli_writes_the_same_exclusive_receipt(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    diagnostic = _make_sealed_smoke(tmp_path)
    output = tmp_path / "cli-output"

    assert reanalysis.main(
        [
            "--diagnostic-output-root",
            str(diagnostic),
            "--output-directory",
            str(output),
            "--allow-smoke",
        ]
    ) == 0

    printed = json.loads(capsys.readouterr().out)
    sealed = json.loads((output / reanalysis.RECEIPT_NAME).read_text(encoding="utf-8"))
    assert printed == sealed

