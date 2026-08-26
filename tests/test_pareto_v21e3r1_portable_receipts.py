from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from ijoc_submission_v21e3r1.scripts import run_v21e3r1_development_parity as runner


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "ijoc_submission_v21e3"
SOURCE_ROOT = "d" * 64
ROW_PATH_SEMANTICS = "row_directory_relative_posix_v1"
MATRIX_PATH_SEMANTICS = "matrix_directory_relative_posix_v1"


@pytest.fixture(scope="module")
def portable_packet(tmp_path_factory: pytest.TempPathFactory):
    base = tmp_path_factory.mktemp("portable-v21e3r1")
    contract = runner.load_frozen_contract(
        case_manifest_path=OLD
        / "development_partitions_v1"
        / "case_manifest.json",
        reference_manifest_path=OLD
        / "development_manifests_v1"
        / "reference_manifest_development.json",
        config_manifest_path=OLD
        / "development_manifests_v1"
        / "config_manifest_development.json",
        metric_manifest_path=OLD
        / "development_manifests_v1"
        / "metric_manifest.json",
        protocol_path=OLD / "protocol" / "V21E3_C0_PARITY_PROTOCOL_V2.json",
        authorization_path=None,
        source_snapshot_root_sha256=SOURCE_ROOT,
        require_matrix_authorization=False,
    )
    contract = replace(contract, budget=21, checkpoint_period=21)
    case = next(
        item for item in contract.cases if item.family == "MOKP" and item.size == 100
    )
    planned = {
        "case_id": case.case_id,
        "family": case.family,
        "size": case.size,
        "seed": 31051,
        "arm_id": "V21E3_C0",
        "row_slug": f"{case.case_id}__seed-31051__arm-v21e3_c0",
    }
    packet = base / "original-matrix"
    row_directory = packet / "rows" / planned["row_slug"]
    row_directory.parent.mkdir(parents=True)
    runner.execute_row(
        case=case,
        arm_id="V21E3_C0",
        seed=31051,
        budget=contract.budget,
        checkpoint_period=contract.checkpoint_period,
        reference_directions=contract.reference_directions,
        source_snapshot_root_sha256=SOURCE_ROOT,
        row_directory=row_directory,
        metric_manifest_sha256=contract.input_sha256["metric_manifest"],
        scientific_scope="authors_generated_development_only_not_formal_evidence",
    )
    (packet / "matrix.plan.json").write_text(
        json.dumps({"rows": [planned]}, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    return packet, contract, planned


def _receipt(row_directory: Path, name: str) -> dict[str, object]:
    payload = json.loads((row_directory / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_row_and_matrix_receipts_remain_verifiable_after_absolute_relocation(
    tmp_path: Path,
    portable_packet,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, contract, planned = portable_packet
    relocated = tmp_path / "another-absolute-parent" / "matrix"
    shutil.copytree(original, relocated)
    row_directory = relocated / "rows" / planned["row_slug"]

    terminal = _receipt(row_directory, "terminal.receipt.json")
    preverification = _receipt(row_directory, "row.preverification.json")
    objective_replay = _receipt(
        row_directory, "objective_archive_replay.receipt.json"
    )
    metric_replay = _receipt(row_directory, "metric_replay.receipt.json")
    row = _receipt(row_directory, "row.json")

    assert terminal["database_path"] == "trace.sqlite3"
    assert preverification["trace_database_path"] == "trace.sqlite3"
    assert (
        preverification["detached_terminal_receipt_path"]
        == "terminal.receipt.json"
    )
    assert objective_replay["database_path"] == "trace.sqlite3"
    assert (
        objective_replay["detached_terminal_receipt_path"]
        == "terminal.receipt.json"
    )
    assert metric_replay["database_path"] == "trace.sqlite3"
    assert row["artifact_path_semantics"] == ROW_PATH_SEMANTICS
    assert row["trace_database_path"] == "trace.sqlite3"
    assert row["detached_terminal_receipt_path"] == "terminal.receipt.json"
    assert row["preverification_receipt_path"] == "row.preverification.json"
    assert (
        row["objective_archive_replay_receipt_path"]
        == "objective_archive_replay.receipt.json"
    )
    assert row["metric_replay_receipt_path"] == "metric_replay.receipt.json"

    loaded = runner._load_completed_row(
        row_directory,
        planned=planned,
        contract=contract,
    )
    assert loaded == row

    monkeypatch.setattr(
        runner,
        "analyze_development_parity",
        lambda *_args, **_kwargs: {"overall_gate": "FAIL_PORTABILITY_UNIT_ONLY"},
    )
    monkeypatch.setattr(
        runner,
        "verify_finalized_matrix_output",
        lambda *_args, **_kwargs: {"status": "UNIT_ONLY"},
    )
    runner.finalize_matrix_output(
        relocated,
        contract=contract,
        plan={"rows": [planned]},
    )
    aggregate = _receipt(relocated, "matrix.aggregate.json")
    post_run = _receipt(relocated, "post_run_audit.json")
    assert aggregate["artifact_path_semantics"] == MATRIX_PATH_SEMANTICS
    assert aggregate["rows"][0]["row_receipt_path"] == (
        f"rows/{planned['row_slug']}/row.json"
    )
    assert post_run["matrix_aggregate_path"] == "matrix.aggregate.json"


def test_completed_row_remains_verifiable_below_uri_reserved_hash_parent(
    tmp_path: Path,
    portable_packet,
) -> None:
    original, contract, planned = portable_packet
    relocated = tmp_path / "uri#reserved-parent" / "matrix"
    shutil.copytree(original, relocated)
    row_directory = relocated / "rows" / planned["row_slug"]

    loaded = runner._load_completed_row(
        row_directory,
        planned=planned,
        contract=contract,
    )

    assert loaded["trace_database_path"] == "trace.sqlite3"


def test_metric_replay_rejects_coercible_objective_types(
    tmp_path: Path,
    portable_packet,
) -> None:
    original, contract, planned = portable_packet
    relocated = tmp_path / "strict-objective-types" / "matrix"
    shutil.copytree(original, relocated)
    row_directory = relocated / "rows" / planned["row_slug"]
    database_path = row_directory / "trace.sqlite3"
    case = next(item for item in contract.cases if item.case_id == planned["case_id"])

    import sqlite3

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE evaluations SET objectives_json='[true,0.0]' "
            "WHERE evaluation_index=1"
        )

    with pytest.raises(ValueError, match="exact JSON numbers"):
        runner.replay_normalized_metric_from_trace(
            database_path,
            budget=contract.budget,
            checkpoint_period=contract.checkpoint_period,
            lower=case.lower,
            upper=case.upper,
            metric_manifest_sha256=contract.input_sha256["metric_manifest"],
        )


def test_second_pass_failure_removes_unverified_final_receipts(
    tmp_path: Path,
    portable_packet,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, contract, planned = portable_packet
    relocated = tmp_path / "second-pass-failure" / "matrix"
    shutil.copytree(original, relocated)
    monkeypatch.setattr(
        runner,
        "analyze_development_parity",
        lambda *_args, **_kwargs: {"overall_gate": "PASS_UNIT_ONLY"},
    )

    def fail_second_pass(*_args, **_kwargs):
        raise RuntimeError("synthetic second-pass failure")

    monkeypatch.setattr(runner, "verify_finalized_matrix_output", fail_second_pass)

    with pytest.raises(RuntimeError, match="second-pass failure"):
        runner.finalize_matrix_output(
            relocated,
            contract=contract,
            plan={"rows": [planned]},
        )

    assert not (relocated / "matrix.aggregate.json").exists()
    assert not (relocated / "post_run_audit.json").exists()


def test_completed_row_rejects_absolute_and_escaping_artifact_paths(
    tmp_path: Path,
    portable_packet,
) -> None:
    original, contract, planned = portable_packet
    malicious_paths = (
        (tmp_path / "outside-trace.sqlite3").resolve().as_posix(),
        "../trace.sqlite3",
    )
    for ordinal, malicious_path in enumerate(malicious_paths):
        relocated = tmp_path / f"tampered-{ordinal}"
        shutil.copytree(original, relocated)
        row_directory = relocated / "rows" / planned["row_slug"]
        row_path = row_directory / "row.json"
        row = json.loads(row_path.read_text(encoding="utf-8"))
        row["trace_database_path"] = malicious_path
        row_path.write_text(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError, match="artifact path"):
            runner._load_completed_row(
                row_directory,
                planned=planned,
                contract=contract,
            )


def test_baseline_terminal_receipt_is_portable_without_post_hoc_sqlite_rewrite(
    tmp_path: Path,
    portable_packet,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, contract, _ = portable_packet
    contract = replace(contract, budget=40, checkpoint_period=40)
    case = next(
        item for item in contract.cases if item.family == "MOKP" and item.size == 100
    )
    row_directory = tmp_path / "baseline-original"
    statements: list[str] = []
    real_connect = runner.sqlite3.connect

    def traced_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(runner.sqlite3, "connect", traced_connect)
    runner.execute_row(
        case=case,
        arm_id="NSGAII",
        seed=31051,
        budget=contract.budget,
        checkpoint_period=contract.checkpoint_period,
        reference_directions=contract.reference_directions,
        source_snapshot_root_sha256=SOURCE_ROOT,
        row_directory=row_directory,
        metric_manifest_sha256=contract.input_sha256["metric_manifest"],
        scientific_scope="authors_generated_development_only_not_formal_evidence",
    )

    terminal = _receipt(row_directory, "terminal.receipt.json")
    assert terminal["database_path"] == "trace.sqlite3"
    normalized_statements = {" ".join(statement.upper().split()) for statement in statements}
    assert not any(
        statement.startswith("UPDATE TERMINAL_RECEIPTS SET RECEIPT_JSON")
        for statement in normalized_statements
    )

    relocated = tmp_path / "baseline-relocated"
    shutil.copytree(row_directory, relocated)
    planned = {
        "case_id": case.case_id,
        "family": case.family,
        "size": case.size,
        "seed": 31051,
        "arm_id": "NSGAII",
    }
    loaded = runner._load_completed_row(
        relocated,
        planned=planned,
        contract=contract,
    )
    assert loaded["artifact_path_semantics"] == ROW_PATH_SEMANTICS

