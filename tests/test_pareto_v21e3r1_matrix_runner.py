from __future__ import annotations

import hashlib
import json
from pathlib import Path
from dataclasses import replace

import pytest

from ijoc_submission_v21e3r1.scripts import run_v21e3r1_development_parity
from ijoc_submission_v21e3r1.scripts import run_v21e3r1_target_structural


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "ijoc_submission_v21e3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authorization(tmp_path: Path, source_root: str) -> Path:
    paths = {
        "case_manifest": OLD / "development_partitions_v1" / "case_manifest.json",
        "reference_manifest": OLD
        / "development_manifests_v1"
        / "reference_manifest_development.json",
        "config_manifest": OLD
        / "development_manifests_v1"
        / "config_manifest_development.json",
        "metric_manifest": OLD
        / "development_manifests_v1"
        / "metric_manifest.json",
        "protocol": OLD / "protocol" / "V21E3_C0_PARITY_PROTOCOL_V2.json",
    }
    bindings = {
        role: {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for role, path in paths.items()
    }
    payload = {
        "schema": "pareto_v21e3r1_development_parity_authorization_v1",
        "status": "AUTHORIZED_DEVELOPMENT_PARITY_ONLY",
        "source_snapshot_root_sha256": source_root,
        "bindings": bindings,
        "development_parity_execution": "AUTHORIZED_DEVELOPMENT_ONLY",
        "selection_entropy_release": "PROHIBITED",
        "calibration_execution": "PROHIBITED",
        "formal_execution": "PROHIBITED",
        "formal_authorized": False,
    }
    path = tmp_path / "authorization.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def test_contract_loader_accepts_only_the_frozen_complete_design(tmp_path: Path) -> None:
    source_root = "a" * 64
    contract = run_v21e3r1_development_parity.load_frozen_contract(
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
        source_snapshot_root_sha256=source_root,
        require_matrix_authorization=False,
    )

    assert len(contract.cases) == 12
    assert contract.seeds == (31051, 31057, 31059)
    assert contract.arms == ("V21E3_C0", "NSGAII", "MOEAD")
    assert contract.budget == 2_000
    assert contract.checkpoint_period == 200


def test_one_row_is_durable_and_objective_archive_replayed_before_row_last(
    tmp_path: Path,
) -> None:
    source_root = "b" * 64
    contract = run_v21e3r1_development_parity.load_frozen_contract(
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
        source_snapshot_root_sha256=source_root,
        require_matrix_authorization=False,
    )
    case = next(
        item for item in contract.cases if item.family == "MOKP" and item.size == 100
    )
    row_directory = tmp_path / "one-row"

    row = run_v21e3r1_development_parity.execute_row(
        case=case,
        arm_id="V21E3_C0",
        seed=31051,
        budget=21,
        checkpoint_period=21,
        reference_directions=contract.reference_directions,
        source_snapshot_root_sha256=source_root,
        row_directory=row_directory,
        metric_manifest_sha256=contract.input_sha256["metric_manifest"],
        scientific_scope="authors_generated_development_only_not_formal_evidence",
    )

    terminal_path = row_directory / "terminal.receipt.json"
    preverification = json.loads(
        (row_directory / "row.preverification.json").read_text(encoding="utf-8")
    )
    replay = json.loads(
        (row_directory / "objective_archive_replay.receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert preverification["detached_terminal_receipt_sha256"] == _sha256(
        terminal_path
    )
    assert replay["status"] == "OBJECTIVE_AND_ARCHIVE_REPLAY_PASS"
    assert row["source_snapshot_root_sha256"] == source_root
    assert row["trace_database_sha256"] == replay["database_sha256"]
    assert row["selection_entropy_release"] == "PROHIBITED"
    metric_replay = json.loads(
        (row_directory / "metric_replay.receipt.json").read_text(encoding="utf-8")
    )
    assert metric_replay["status"] == "NORMALIZED_HV_AUC_REPLAY_PASS"
    assert metric_replay["normalized_left_continuous_hv_auc"] == row[
        "normalized_left_continuous_hv_auc"
    ]
    assert (row_directory / "row.json").stat().st_mtime_ns >= (
        row_directory / "objective_archive_replay.receipt.json"
    ).stat().st_mtime_ns

    # The scientific estimand must be replayed from the objective ledger.  A
    # locally edited row.json cannot become an admissible aggregate input.
    row_path = row_directory / "row.json"
    tampered = json.loads(row_path.read_text(encoding="utf-8"))
    tampered["normalized_left_continuous_hv_auc"] = min(
        1.0, float(tampered["normalized_left_continuous_hv_auc"]) + 0.125
    )
    row_path.write_text(
        json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    planned = {
        "case_id": case.case_id,
        "family": case.family,
        "size": case.size,
        "seed": 31051,
        "arm_id": "V21E3_C0",
    }
    with pytest.raises(RuntimeError, match="metric replay"):
        run_v21e3r1_development_parity._load_completed_row(
            row_directory,
            planned=planned,
            contract=replace(contract, budget=21, checkpoint_period=21),
        )


def test_matrix_plan_is_exact_and_resume_rejects_a_partial_row(tmp_path: Path) -> None:
    source_root = "c" * 64
    contract = run_v21e3r1_development_parity.load_frozen_contract(
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
        source_snapshot_root_sha256=source_root,
        require_matrix_authorization=False,
    )
    contract = replace(
        contract,
        authorization_sha256="e" * 64,
        authorization_path=tmp_path / "unit-only-authorization.json",
        live_authorization_verified=True,
    )
    output = tmp_path / "matrix"
    plan = run_v21e3r1_development_parity.initialize_matrix_output(
        output, contract=contract, resume=False
    )

    assert len(plan["rows"]) == 108
    assert len(
        {
            (row["case_id"], row["seed"], row["arm_id"])
            for row in plan["rows"]
        }
    ) == 108
    first_slug = plan["rows"][0]["row_slug"]
    (output / "rows" / first_slug).mkdir()
    (output / "rows" / first_slug / "trace.sqlite3").write_bytes(b"partial")

    with pytest.raises(RuntimeError, match="partial row"):
        run_v21e3r1_development_parity.initialize_matrix_output(
            output, contract=contract, resume=True
        )


def test_target_structural_plan_is_two_n500_cases_by_three_arms(tmp_path: Path) -> None:
    source_root = "d" * 64
    contract = run_v21e3r1_development_parity.load_frozen_contract(
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
        source_snapshot_root_sha256=source_root,
        require_matrix_authorization=False,
    )

    plan = run_v21e3r1_target_structural.build_target_structural_plan(contract)

    assert plan["budget"] == 200
    assert plan["checkpoint_period"] == 200
    assert len(plan["rows"]) == 6
    assert {row["family"] for row in plan["rows"]} == {"MOTSP", "MOKP"}
    assert {row["size"] for row in plan["rows"]} == {500}
    assert {row["arm_id"] for row in plan["rows"]} == {
        "V21E3_C0",
        "NSGAII",
        "MOEAD",
    }
    assert {row["seed"] for row in plan["rows"]} == {31051}

