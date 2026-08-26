from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FREEZE_SCRIPT = (
    ROOT
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "freeze_v21e3r1_development_snapshot.py"
)
PREFLIGHT_SCRIPT = (
    ROOT
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "preflight_v21e3r1_development_parity.py"
)
STRUCTURAL_SCRIPT = (
    ROOT
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "audit_v21e3r1_target_size_structure.py"
)
OLD_V21E3_ZIP_SHA256 = (
    "7881b30e6f6059e36e0ed8279f8932ab5f48f2f8e0bc38885e59a74fb45fb3b0"
)


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _write_json(path: Path, value: object) -> None:
    _write(
        path,
        (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _binding(repo: Path, path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(repo).as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _fixture_sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fixture_reference(case_id: str, family: str) -> dict[str, object]:
    lower = [0.0, 0.0] if family == "MOTSP" else [-10.0, -10.0]
    upper = [10.0, 10.0] if family == "MOTSP" else [0.0, 0.0]
    return {
        "case_id": case_id,
        "family": family,
        "artifact_sha256": _fixture_sha256(f"artifact:{case_id}"),
        "problem_sha256": _fixture_sha256(f"problem:{case_id}"),
        "objective_lower_bounds": lower,
        "objective_upper_bounds": upper,
        "normalized_reference_point": [1.0, 1.0],
    }


def _fixture_target_reference_cases() -> dict[str, dict[str, object]]:
    return {
        f"{family.lower()}-500-0": {
            **_fixture_reference(f"{family.lower()}-500-0", family),
            "size": 500,
        }
        for family in ("MOTSP", "MOKP")
    }


def _terminal_payload_sha256(payload: dict[str, object]) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_immutable_parent_chain(repo: Path, parent_zip_sha256: str) -> None:
    provenance = repo / "ijoc_submission_v21e3" / "provenance"
    release = repo / "ijoc_submission_v21e3" / "release"
    _write_json(
        provenance / "V21E2_IMMUTABLE_BASELINE.json",
        {"status": "IMMUTABLE_CALIBRATION_EVIDENCE_NOT_MODIFIED"},
    )
    _write_json(
        provenance / "V21E2_IMMUTABLE_CALIBRATION_EVIDENCE.json",
        {"status": "IMMUTABLE_CALIBRATION_EVIDENCE"},
    )
    _write_json(
        provenance / "V21E3_DEVELOPMENT_SNAPSHOT_FREEZE_V1.json",
        {"status": "PASS_ENGINEERING_SNAPSHOT"},
    )
    _write_json(
        release / "ijoc_v21e3_experiment_code.manifest.json",
        {"schema": "ijoc_v21e3_standalone_release_manifest_v1"},
    )
    _write_json(
        release / "ijoc_v21e3_clean_room.receipt.json",
        {"status": "PASS"},
    )
    _write(
        release / "ijoc_v21e3_experiment_code.zip.sha256",
        (parent_zip_sha256 + "  ijoc_v21e3_experiment_code.zip\n").encode("ascii"),
    )


def _write_contract_files(repo: Path) -> dict[str, Path]:
    manifests = repo / "ijoc_submission_v21e3" / "development_manifests_v1"
    partition = repo / "ijoc_submission_v21e3" / "development_partitions_v1"
    cases = [
        {
            "case_id": f"{family.lower()}-{size}-{ordinal}",
            "family": family,
            "size": size,
            "split": "development",
            "artifact": {
                "sha256": _fixture_sha256(
                    f"artifact:{family.lower()}-{size}-{ordinal}"
                )
            },
            "fingerprints": {
                "problem_sha256": _fixture_sha256(
                    f"problem:{family.lower()}-{size}-{ordinal}"
                )
            },
        }
        for family in ("MOTSP", "MOKP")
        for size in (100, 200, 500)
        for ordinal in range(2)
    ]
    case_manifest = partition / "case_manifest.json"
    _write_json(
        case_manifest,
        {
            "schema": "pareto_v21_partition_manifest_v1",
            "split": "development",
            "formal_confirmatory_eligibility": False,
            "cases": cases,
        },
    )
    reference = manifests / "reference_manifest_development.json"
    _write_json(
        reference,
        {
            "schema": "pareto_v21e3_analytic_reference_manifest_v1",
            "status": "FROZEN_DEVELOPMENT_ONLY",
            "split": "development",
            "formal_use": "NOT_AUTHORIZED",
            "case_count": 12,
            "partition_manifest": _binding(repo, case_manifest),
            "cases": [
                {
                    **_fixture_reference(
                        str(case["case_id"]), str(case["family"])
                    ),
                    "size": case["size"],
                }
                for case in cases
            ],
        },
    )
    metric = manifests / "metric_manifest.json"
    _write_json(
        metric,
        {
            "schema": "pareto_v21e3_metric_manifest_v1",
            "status": "FROZEN_DEVELOPMENT_AND_FUTURE_CALIBRATION_INPUT",
            "formal_use": "NOT_AUTHORIZED",
            "selection_grid": {
                "charged_budget": 2000,
                "checkpoint_period": 200,
            },
        },
    )
    config = manifests / "config_manifest_development.json"
    _write_json(
        config,
        {
            "schema": "pareto_v21e3_development_config_manifest_v1",
            "status": "FROZEN_DEVELOPMENT_INPUT_CALIBRATION_EXECUTION_BLOCKED",
            "selection_partition": "NOT_GENERATED",
            "calibration_confirmation_partition": "NOT_GENERATED",
            "calibration_execution_authorized": False,
            "formal_cases": "NOT_MATERIALIZED",
            "formal_execution_authorized": False,
            "reference_manifest": reference.name,
            "metric_manifest": metric.name,
            "reference_directions": [[0.0, 1.0]] * 21,
        },
    )
    protocol = repo / "ijoc_submission_v21e3" / "protocol" / "parity.json"
    _write_json(
        protocol,
        {
            "schema": "pareto_v21e3_c0_parity_protocol_v2",
            "status": "ENGINEERING_ADAPTERS_AVAILABLE_SUCCESSOR_SNAPSHOT_PENDING",
            "successor_version": "V21e3r1",
            "families": ["MOTSP", "MOKP"],
            "common_execution_contract": {
                "charged_evaluation_budget": 2000,
                "checkpoint_period": 200,
            },
            "candidate_reference_directions": {
                "count": 21,
                "source_binding": _binding(repo, config),
                "source_field": "reference_directions",
            },
            "arms": {
                "V21E3_C0": {"execution_adapter_status": "DEVELOPMENT_ONLY_AVAILABLE"},
                "NSGAII": {"execution_adapter_status": "DEVELOPMENT_ONLY_AVAILABLE"},
                "MOEAD": {"execution_adapter_status": "DEVELOPMENT_ONLY_AVAILABLE"},
            },
            "case_design": {
                "manifest": case_manifest.relative_to(repo).as_posix(),
                "case_count_per_family": 6,
                "sizes": [100, 200, 500],
                "cases_per_size_per_family": 2,
                "seeds": [31051, 31057, 31059],
            },
            "preflight_gates": {
                "successor_source_snapshot": "PENDING",
                "independent_protocol_preflight": "NOT_RUN",
                "matched_matrix": "NOT_RUN",
                "selection_entropy_release": "PROHIBITED",
                "calibration_execution": "PROHIBITED",
                "formal_execution": "PROHIBITED",
                "formal_authorized": False,
            },
        },
    )
    return {
        "protocol": protocol,
        "case": case_manifest,
        "reference": reference,
        "config": config,
        "metric": metric,
    }


def _target_execution_payload(
    repo: Path,
    *,
    source_root_sha256: str,
    input_sha256: dict[str, str],
) -> dict[str, object]:
    target = repo / "evidence" / "target-run"
    plan_path = target / "target_structural.plan.json"
    planned_rows = []
    for family in ("MOTSP", "MOKP"):
        for arm in ("V21E3_C0", "NSGAII", "MOEAD"):
            case_id = f"{family.lower()}-500-0"
            planned_rows.append(
                {
                    "case_id": case_id,
                    "family": family,
                    "size": 500,
                    "seed": 31051,
                    "arm_id": arm,
                    "row_slug": (
                        f"{case_id}__seed-31051__arm-{arm.lower()}"
                    ),
                }
            )
    _write_json(
        plan_path,
        {
            "schema": "pareto_v21e3r1_target_size_three_arm_plan_v1",
            "status": "READY_TARGET_SIZE_SMALL_BUDGET_ENGINEERING_ONLY",
            "scientific_scope": (
                "target_size_small_budget_structure_and_objective_archive_"
                "replay_not_performance_evidence"
            ),
            "budget": 200,
            "checkpoint_period": 200,
            "baseline_population_sizes": {"MOTSP": 48, "MOKP": 40},
            "budget_initializes_every_baseline_population": True,
            "source_snapshot_root_sha256": source_root_sha256,
            "input_sha256": input_sha256,
            "rows": planned_rows,
            "development_parity_execution": "NOT_AUTHORIZED_BY_THIS_PLAN",
            "selection_entropy_release": "PROHIBITED",
            "calibration_execution": "PROHIBITED",
            "formal_execution": "PROHIBITED",
            "formal_authorized": False,
        },
    )
    summaries: list[dict[str, object]] = []
    for planned in planned_rows:
            family = str(planned["family"])
            arm = str(planned["arm_id"])
            case_id = str(planned["case_id"])
            row_root = target / "rows" / str(planned["row_slug"])
            row_root.mkdir(parents=True)
            database = row_root / "trace.sqlite3"
            terminal = row_root / "terminal.receipt.json"
            pre = row_root / "row.preverification.json"
            replay = row_root / "objective_archive_replay.receipt.json"
            metric_replay = row_root / "metric_replay.receipt.json"
            row_path = row_root / "row.json"
            reference = _fixture_reference(case_id, family)
            point = [5.0, 5.0] if family == "MOTSP" else [-5.0, -5.0]
            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    "CREATE TABLE evaluations("
                    "evaluation_index INTEGER PRIMARY KEY,objectives_json TEXT NOT NULL)"
                )
                connection.executemany(
                    "INSERT INTO evaluations(evaluation_index,objectives_json) VALUES (?,?)",
                    [
                        (
                            index,
                            json.dumps(point, separators=(",", ":")),
                        )
                        for index in range(1, 201)
                    ],
                )
                connection.commit()
            finally:
                connection.close()
            database_sha256 = hashlib.sha256(database.read_bytes()).hexdigest()
            context_sha256 = _fixture_sha256(f"context:{case_id}:{arm}")
            attempt_chain_sha256 = _fixture_sha256(f"attempt:{case_id}:{arm}")
            evaluation_chain_sha256 = _fixture_sha256(
                f"evaluation:{case_id}:{arm}"
            )
            decision_chain_sha256 = _fixture_sha256(f"decision:{case_id}:{arm}")
            terminal_core: dict[str, object] = {
                "schema": "v21e3_terminal_receipt_v1",
                "status": "SUCCESS",
                "problem": case_id,
                "family": family,
                "failure_code": None,
                "failure_detail": None,
                "attempt_count": 200,
                "physical_call_started_count": 200,
                "charged_evaluation_count": 200,
                "decision_count": 200,
                "cache_hit_count": 0,
                "unresolved_decision_count": 0,
                "terminal_evaluation_chain_sha256": evaluation_chain_sha256,
                "terminal_decision_chain_sha256": decision_chain_sha256,
                "terminal_attempt_chain_sha256": attempt_chain_sha256,
                "run_context_digest_sha256": context_sha256,
                "database_path": "trace.sqlite3",
                "durability_mode": "SQLITE_WAL_SYNCHRONOUS_FULL",
                "finalization_gates": {
                    "expected_charged_evaluations": 200,
                    "expected_decisions": 200,
                    "run_context_charged_evaluation_budget": 200,
                    "persisted_attempts": 200,
                    "persisted_evaluations": 200,
                    "persisted_decisions": 200,
                    "physical_call_starts": 200,
                    "cache_hits": 0,
                    "nonterminal_attempts": 0,
                    "evaluation_index_bounds": [1, 200],
                    "expected_evaluation_index_bounds": [1, 200],
                    "sqlite_integrity": "ok",
                },
            }
            terminal_payload_sha256 = _terminal_payload_sha256(terminal_core)
            _write_json(
                terminal,
                {
                    **terminal_core,
                    "receipt_payload_sha256": terminal_payload_sha256,
                },
            )
            terminal_sha256 = hashlib.sha256(terminal.read_bytes()).hexdigest()
            _write_json(
                pre,
                {
                    "schema": "pareto_v21e3r1_row_preverification_v1",
                    "status": "READY_FOR_OBJECTIVE_AND_ARCHIVE_REPLAY",
                    "case_id": case_id,
                    "family": family,
                    "size": 500,
                    "seed": 31051,
                    "arm_id": arm,
                    "source_snapshot_root_sha256": source_root_sha256,
                    "trace_database_path": "trace.sqlite3",
                    "detached_terminal_receipt_path": "terminal.receipt.json",
                    "detached_terminal_receipt_sha256": terminal_sha256,
                    "selection_entropy_release": "PROHIBITED",
                    "calibration_execution": "PROHIBITED",
                    "formal_execution": "PROHIBITED",
                    "formal_authorized": False,
                },
            )
            _write_json(
                replay,
                {
                    "schema": "v21e3r1_objective_archive_replay_receipt_v2",
                    "status": "OBJECTIVE_AND_ARCHIVE_REPLAY_PASS",
                    "verification_scope": (
                        "objective_solution_chain_archive_and_terminal_replay_v1"
                    ),
                    "database_path": "trace.sqlite3",
                    "database_bytes": database.stat().st_size,
                    "database_sha256": database_sha256,
                    "detached_terminal_receipt_path": "terminal.receipt.json",
                    "detached_terminal_receipt_sha256": terminal_sha256,
                    "attempt_records": 200,
                    "evaluation_records": 200,
                    "decision_records": 200,
                    "cache_hit_records": 0,
                    "unique_solution_replays": 200,
                    "archive_reconstruction": "PASS",
                    "archive_size": 1,
                    "terminal_status": "SUCCESS",
                    "run_context_digest_sha256": context_sha256,
                    "terminal_attempt_chain_sha256": attempt_chain_sha256,
                    "terminal_evaluation_chain_sha256": evaluation_chain_sha256,
                    "terminal_decision_chain_sha256": decision_chain_sha256,
                    "terminal_receipt_sha256": terminal_payload_sha256,
                    "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
                    "selection_authorization": "PROHIBITED",
                },
            )
            checkpoint_rows = [
                {"evaluation": 200, "normalized_hv": 0.25, "archive_size": 1}
            ]
            _write_json(
                metric_replay,
                {
                    "schema": "pareto_v21e3r1_metric_replay_receipt_v1",
                    "status": "NORMALIZED_HV_AUC_REPLAY_PASS",
                    "verification_scope": (
                        "frozen_metric_from_objective_ledger_checkpoints_v1"
                    ),
                    "database_path": "trace.sqlite3",
                    "database_sha256": database_sha256,
                    "metric_manifest_sha256": input_sha256["metric_manifest"],
                    "charged_evaluation_budget": 200,
                    "checkpoint_period": 200,
                    "analytic_box": {
                        "lower": reference["objective_lower_bounds"],
                        "upper": reference["objective_upper_bounds"],
                        "normalized_reference": [1.0, 1.0],
                    },
                    "normalized_left_continuous_hv_auc": 0.0,
                    "normalized_terminal_hv": 0.25,
                    "checkpoints": checkpoint_rows,
                    "selection_authorization": "PROHIBITED",
                    "formal_authorized": False,
                },
            )
            pre_sha256 = hashlib.sha256(pre.read_bytes()).hexdigest()
            replay_sha256 = hashlib.sha256(replay.read_bytes()).hexdigest()
            metric_replay_sha256 = hashlib.sha256(
                metric_replay.read_bytes()
            ).hexdigest()
            row = {
                "schema": "pareto_v21e3r1_matched_matrix_row_v1",
                "status": "PASS_ENGINEERING_ROW_OBJECTIVE_ARCHIVE_REPLAYED",
                "scientific_scope": (
                    "target_size_small_budget_structure_and_objective_archive_"
                    "replay_not_performance_evidence"
                ),
                "artifact_path_semantics": "row_directory_relative_posix_v1",
                "case_id": case_id,
                "family": family,
                "size": 500,
                "case_artifact_sha256": reference["artifact_sha256"],
                "generator_problem_fingerprint_sha256": reference[
                    "problem_sha256"
                ],
                "runtime_problem_semantic_sha256": _fixture_sha256(
                    f"runtime:{case_id}"
                ),
                "seed": 31051,
                "arm_id": arm,
                "charged_evaluation_budget": 200,
                "checkpoint_period": 200,
                "normalized_left_continuous_hv_auc": 0.0,
                "normalized_terminal_hv": 0.25,
                "checkpoints": checkpoint_rows,
                "elapsed_process_wall_ns_diagnostic_only": 1,
                "source_snapshot_root_sha256": source_root_sha256,
                "run_context_digest_sha256": context_sha256,
                "trace_database_path": "trace.sqlite3",
                "trace_database_bytes": database.stat().st_size,
                "trace_database_sha256": database_sha256,
                "detached_terminal_receipt_path": "terminal.receipt.json",
                "detached_terminal_receipt_sha256": terminal_sha256,
                "preverification_receipt_path": "row.preverification.json",
                "preverification_receipt_sha256": pre_sha256,
                "objective_archive_replay_receipt_path": (
                    "objective_archive_replay.receipt.json"
                ),
                "objective_archive_replay_receipt_sha256": replay_sha256,
                "metric_replay_receipt_path": "metric_replay.receipt.json",
                "metric_replay_receipt_sha256": metric_replay_sha256,
                "metric_manifest_sha256": input_sha256["metric_manifest"],
                "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
                "runtime_efficiency_claim_authorized": False,
                "selection_entropy_release": "PROHIBITED",
                "calibration_execution": "PROHIBITED",
                "formal_execution": "PROHIBITED",
                "formal_authorized": False,
            }
            _write_json(row_path, row)
            summaries.append(
                {
                    "case_id": case_id,
                    "family": family,
                    "size": 500,
                    "seed": 31051,
                    "arm_id": arm,
                    "charged_evaluation_budget": 200,
                    "checkpoint_period": 200,
                    "trace_database_sha256": database_sha256,
                    "detached_terminal_receipt_sha256": terminal_sha256,
                    "objective_archive_replay_receipt_sha256": replay_sha256,
                    "metric_replay_receipt_sha256": metric_replay_sha256,
                    "row_receipt_path": row_path.relative_to(repo).as_posix(),
                    "row_receipt_sha256": hashlib.sha256(
                        row_path.read_bytes()
                    ).hexdigest(),
                    "objective_and_archive_replay": "PASS",
                    "metric_replay": "PASS",
                    "normalized_terminal_hv_diagnostic_only": 0.25,
                }
            )
    return {
        "schema": "pareto_v21e3r1_target_size_execution_receipt_v1",
        "status": "PASS_TARGET_SIZE_THREE_ARM_EXECUTION_ENGINEERING_ONLY",
        "scientific_scope": (
            "target_size_small_budget_structure_and_objective_archive_replay_"
            "not_performance_evidence"
        ),
        "artifact_path_semantics": "repo_root_relative_posix_v1",
        "source_snapshot_root_sha256": source_root_sha256,
        "input_sha256": input_sha256,
        "plan_path": plan_path.relative_to(repo).as_posix(),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "target_size": 500,
        "seed": 31051,
        "charged_evaluation_budget": 200,
        "checkpoint_period": 200,
        "row_count": 6,
        "families": ["MOTSP", "MOKP"],
        "arms": ["V21E3_C0", "NSGAII", "MOEAD"],
        "baseline_population_sizes": {"MOTSP": 48, "MOKP": 40},
        "budget_initializes_every_baseline_population": True,
        "rows": summaries,
        "objective_and_archive_replay_pass_rows": 6,
        "metric_replay_pass_rows": 6,
        "target_budget_evidence": "NOT_ESTABLISHED",
        "performance_evidence": "NOT_ESTABLISHED",
        "runtime_efficiency_claim_authorized": False,
        "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
        "development_parity_execution": "NOT_AUTHORIZED_BY_THIS_RECEIPT",
        "selection_entropy_release": "PROHIBITED",
        "calibration_execution": "PROHIBITED",
        "formal_execution": "PROHIBITED",
        "formal_authorized": False,
    }


def test_target_preflight_rejects_coherently_rehashed_metric_forgery(
    tmp_path: Path,
) -> None:
    preflight = _load_script(PREFLIGHT_SCRIPT, "v21e3r1_metric_forgery_preflight")
    repo = tmp_path / "repo"
    repo.mkdir()
    source_root = "a" * 64
    roles = (
        "protocol",
        "case_manifest",
        "reference_manifest",
        "config_manifest",
        "metric_manifest",
    )
    input_sha256 = {role: _fixture_sha256(role) for role in roles}
    receipt = _target_execution_payload(
        repo,
        source_root_sha256=source_root,
        input_sha256=input_sha256,
    )
    summary = receipt["rows"][0]
    assert isinstance(summary, dict)
    row_path = repo / str(summary["row_receipt_path"])
    row = json.loads(row_path.read_text(encoding="utf-8"))
    metric_path = row_path.parent / str(row["metric_replay_receipt_path"])
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    metric["normalized_left_continuous_hv_auc"] = 0.875
    metric["normalized_terminal_hv"] = 0.9375
    metric["checkpoints"] = [
        {"evaluation": 17, "normalized_hv": 0.9375, "archive_size": 999}
    ]
    _write_json(metric_path, metric)
    row["normalized_left_continuous_hv_auc"] = metric[
        "normalized_left_continuous_hv_auc"
    ]
    row["normalized_terminal_hv"] = metric["normalized_terminal_hv"]
    row["checkpoints"] = metric["checkpoints"]
    row["metric_replay_receipt_sha256"] = hashlib.sha256(
        metric_path.read_bytes()
    ).hexdigest()
    _write_json(row_path, row)
    summary["metric_replay_receipt_sha256"] = row[
        "metric_replay_receipt_sha256"
    ]
    summary["row_receipt_sha256"] = hashlib.sha256(
        row_path.read_bytes()
    ).hexdigest()
    summary["normalized_terminal_hv_diagnostic_only"] = metric[
        "normalized_terminal_hv"
    ]

    with pytest.raises(ValueError, match="metric"):
        preflight._validate_target_execution_receipt(
            receipt,
            root=repo,
            prospective_source_root_sha256=source_root,
            expected_input_sha256=input_sha256,
            expected_reference_cases=_fixture_target_reference_cases(),
        )


def test_target_preflight_recomputes_terminal_hv_from_objective_ledger(
    tmp_path: Path,
) -> None:
    preflight = _load_script(PREFLIGHT_SCRIPT, "v21e3r1_metric_ledger_preflight")
    repo = tmp_path / "repo"
    repo.mkdir()
    source_root = "a" * 64
    roles = (
        "protocol",
        "case_manifest",
        "reference_manifest",
        "config_manifest",
        "metric_manifest",
    )
    input_sha256 = {role: _fixture_sha256(role) for role in roles}
    receipt = _target_execution_payload(
        repo,
        source_root_sha256=source_root,
        input_sha256=input_sha256,
    )
    summary = receipt["rows"][0]
    assert isinstance(summary, dict)
    row_path = repo / str(summary["row_receipt_path"])
    row = json.loads(row_path.read_text(encoding="utf-8"))
    metric_path = row_path.parent / str(row["metric_replay_receipt_path"])
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    metric["normalized_terminal_hv"] = 0.9375
    metric["checkpoints"] = [
        {"evaluation": 200, "normalized_hv": 0.9375, "archive_size": 1}
    ]
    _write_json(metric_path, metric)
    row["normalized_terminal_hv"] = metric["normalized_terminal_hv"]
    row["checkpoints"] = metric["checkpoints"]
    row["metric_replay_receipt_sha256"] = hashlib.sha256(
        metric_path.read_bytes()
    ).hexdigest()
    _write_json(row_path, row)
    summary["metric_replay_receipt_sha256"] = row[
        "metric_replay_receipt_sha256"
    ]
    summary["row_receipt_sha256"] = hashlib.sha256(
        row_path.read_bytes()
    ).hexdigest()
    summary["normalized_terminal_hv_diagnostic_only"] = metric[
        "normalized_terminal_hv"
    ]

    with pytest.raises(ValueError, match="metric"):
        preflight._validate_target_execution_receipt(
            receipt,
            root=repo,
            prospective_source_root_sha256=source_root,
            expected_input_sha256=input_sha256,
            expected_reference_cases=_fixture_target_reference_cases(),
        )


def test_target_preflight_binds_metric_box_to_frozen_reference(
    tmp_path: Path,
) -> None:
    preflight = _load_script(PREFLIGHT_SCRIPT, "v21e3r1_metric_box_preflight")
    repo = tmp_path / "repo"
    repo.mkdir()
    source_root = "a" * 64
    roles = (
        "protocol",
        "case_manifest",
        "reference_manifest",
        "config_manifest",
        "metric_manifest",
    )
    input_sha256 = {role: _fixture_sha256(role) for role in roles}
    receipt = _target_execution_payload(
        repo,
        source_root_sha256=source_root,
        input_sha256=input_sha256,
    )
    summary = receipt["rows"][0]
    assert isinstance(summary, dict)
    row_path = repo / str(summary["row_receipt_path"])
    row = json.loads(row_path.read_text(encoding="utf-8"))
    metric_path = row_path.parent / str(row["metric_replay_receipt_path"])
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    metric["analytic_box"] = {
        "lower": [-5.0, -5.0],
        "upper": [15.0, 15.0],
        "normalized_reference": [1.0, 1.0],
    }
    _write_json(metric_path, metric)
    row["metric_replay_receipt_sha256"] = hashlib.sha256(
        metric_path.read_bytes()
    ).hexdigest()
    _write_json(row_path, row)
    summary["metric_replay_receipt_sha256"] = row[
        "metric_replay_receipt_sha256"
    ]
    summary["row_receipt_sha256"] = hashlib.sha256(
        row_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="analytic box"):
        preflight._validate_target_execution_receipt(
            receipt,
            root=repo,
            prospective_source_root_sha256=source_root,
            expected_input_sha256=input_sha256,
            expected_reference_cases=_fixture_target_reference_cases(),
        )


def test_target_preflight_rejects_coherently_rehashed_failure_terminal(
    tmp_path: Path,
) -> None:
    preflight = _load_script(PREFLIGHT_SCRIPT, "v21e3r1_terminal_forgery_preflight")
    repo = tmp_path / "repo"
    repo.mkdir()
    source_root = "a" * 64
    roles = (
        "protocol",
        "case_manifest",
        "reference_manifest",
        "config_manifest",
        "metric_manifest",
    )
    input_sha256 = {role: _fixture_sha256(role) for role in roles}
    receipt = _target_execution_payload(
        repo,
        source_root_sha256=source_root,
        input_sha256=input_sha256,
    )
    summary = receipt["rows"][0]
    assert isinstance(summary, dict)
    row_path = repo / str(summary["row_receipt_path"])
    row = json.loads(row_path.read_text(encoding="utf-8"))
    terminal_path = row_path.parent / str(row["detached_terminal_receipt_path"])
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal.pop("receipt_payload_sha256")
    terminal["status"] = "FAILURE"
    terminal["failure_code"] = "FORGED_FAILURE"
    terminal["failure_detail"] = {"forged": True}
    terminal_payload_sha256 = _terminal_payload_sha256(terminal)
    terminal["receipt_payload_sha256"] = terminal_payload_sha256
    _write_json(terminal_path, terminal)
    terminal_file_sha256 = hashlib.sha256(terminal_path.read_bytes()).hexdigest()

    pre_path = row_path.parent / str(row["preverification_receipt_path"])
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    pre["detached_terminal_receipt_sha256"] = terminal_file_sha256
    _write_json(pre_path, pre)
    pre_sha256 = hashlib.sha256(pre_path.read_bytes()).hexdigest()

    replay_path = row_path.parent / str(
        row["objective_archive_replay_receipt_path"]
    )
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["detached_terminal_receipt_sha256"] = terminal_file_sha256
    replay["terminal_status"] = "FAILURE"
    replay["terminal_receipt_sha256"] = terminal_payload_sha256
    _write_json(replay_path, replay)
    replay_sha256 = hashlib.sha256(replay_path.read_bytes()).hexdigest()

    row["detached_terminal_receipt_sha256"] = terminal_file_sha256
    row["preverification_receipt_sha256"] = pre_sha256
    row["objective_archive_replay_receipt_sha256"] = replay_sha256
    _write_json(row_path, row)
    summary["detached_terminal_receipt_sha256"] = terminal_file_sha256
    summary["objective_archive_replay_receipt_sha256"] = replay_sha256
    summary["row_receipt_sha256"] = hashlib.sha256(
        row_path.read_bytes()
    ).hexdigest()

    with pytest.raises(ValueError, match="terminal"):
        preflight._validate_target_execution_receipt(
            receipt,
            root=repo,
            prospective_source_root_sha256=source_root,
            expected_input_sha256=input_sha256,
            expected_reference_cases=_fixture_target_reference_cases(),
        )


def test_target_preflight_rejects_unbound_target_row_artifacts(
    tmp_path: Path,
) -> None:
    preflight = _load_script(PREFLIGHT_SCRIPT, "v21e3r1_row_inventory_preflight")
    repo = tmp_path / "repo"
    repo.mkdir()
    source_root = "a" * 64
    roles = (
        "protocol",
        "case_manifest",
        "reference_manifest",
        "config_manifest",
        "metric_manifest",
    )
    input_sha256 = {role: _fixture_sha256(role) for role in roles}
    receipt = _target_execution_payload(
        repo,
        source_root_sha256=source_root,
        input_sha256=input_sha256,
    )
    summary = receipt["rows"][0]
    assert isinstance(summary, dict)
    row_path = repo / str(summary["row_receipt_path"])
    _write(row_path.parent / "trace.sqlite3-wal", b"unbound WAL bytes")

    with pytest.raises(ValueError, match="six regular artifacts"):
        preflight._validate_target_execution_receipt(
            receipt,
            root=repo,
            prospective_source_root_sha256=source_root,
            expected_input_sha256=input_sha256,
            expected_reference_cases=_fixture_target_reference_cases(),
        )


def test_target_preflight_rejects_coherently_rehashed_plan_identity_drift(
    tmp_path: Path,
) -> None:
    preflight = _load_script(PREFLIGHT_SCRIPT, "v21e3r1_plan_identity_preflight")
    repo = tmp_path / "repo"
    repo.mkdir()
    source_root = "a" * 64
    roles = (
        "protocol",
        "case_manifest",
        "reference_manifest",
        "config_manifest",
        "metric_manifest",
    )
    input_sha256 = {role: _fixture_sha256(role) for role in roles}
    receipt = _target_execution_payload(
        repo,
        source_root_sha256=source_root,
        input_sha256=input_sha256,
    )
    plan_path = repo / str(receipt["plan_path"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["rows"][0]["case_id"] = "forged-target-case"
    _write_json(plan_path, plan)
    receipt["plan_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="plan"):
        preflight._validate_target_execution_receipt(
            receipt,
            root=repo,
            prospective_source_root_sha256=source_root,
            expected_input_sha256=input_sha256,
            expected_reference_cases=_fixture_target_reference_cases(),
        )


def test_target_preflight_rejects_row_level_runtime_claim_authorization(
    tmp_path: Path,
) -> None:
    preflight = _load_script(PREFLIGHT_SCRIPT, "v21e3r1_row_claim_preflight")
    repo = tmp_path / "repo"
    repo.mkdir()
    source_root = "a" * 64
    roles = (
        "protocol",
        "case_manifest",
        "reference_manifest",
        "config_manifest",
        "metric_manifest",
    )
    input_sha256 = {role: _fixture_sha256(role) for role in roles}
    receipt = _target_execution_payload(
        repo,
        source_root_sha256=source_root,
        input_sha256=input_sha256,
    )
    summary = receipt["rows"][0]
    assert isinstance(summary, dict)
    row_path = repo / str(summary["row_receipt_path"])
    row = json.loads(row_path.read_text(encoding="utf-8"))
    row["runtime_efficiency_claim_authorized"] = True
    _write_json(row_path, row)
    summary["row_receipt_sha256"] = hashlib.sha256(
        row_path.read_bytes()
    ).hexdigest()

    with pytest.raises(ValueError, match="row receipt"):
        preflight._validate_target_execution_receipt(
            receipt,
            root=repo,
            prospective_source_root_sha256=source_root,
            expected_input_sha256=input_sha256,
            expected_reference_cases=_fixture_target_reference_cases(),
        )


def test_target_size_structural_gate_is_pre_freeze_and_fail_closed(
    tmp_path: Path,
) -> None:
    structural = _load_script(STRUCTURAL_SCRIPT, "v21e3r1_structural")
    repo = tmp_path / "repo"
    paths = _write_contract_files(repo)
    output = repo / "structural.json"

    receipt = structural.audit_target_size_structure(
        repo_root=repo,
        protocol_path=paths["protocol"],
        case_manifest_path=paths["case"],
        reference_manifest_path=paths["reference"],
        config_manifest_path=paths["config"],
        metric_manifest_path=paths["metric"],
        output=output,
    )

    assert receipt["schema"] == (
        "pareto_v21e3r1_target_size_input_structure_receipt_v1"
    )
    assert receipt["status"] == (
        "PASS_TARGET_SIZE_INPUT_STRUCTURE_ENGINEERING_ONLY"
    )
    assert receipt["scientific_scope"] == (
        "pre_freeze_input_structure_not_execution_or_performance_evidence"
    )
    assert receipt["target_sizes"] == {"MOTSP": [100, 200, 500], "MOKP": [100, 200, 500]}
    assert receipt["case_count"] == 12
    assert receipt["arms"] == ["V21E3_C0", "NSGAII", "MOEAD"]
    assert receipt["development_parity_execution"] == "NOT_AUTHORIZED_BY_THIS_RECEIPT"
    assert receipt["selection_entropy_release"] == "PROHIBITED"
    assert receipt["formal_authorized"] is False
    assert set(receipt["bindings"]) == {
        "protocol",
        "case_manifest",
        "reference_manifest",
        "config_manifest",
        "metric_manifest",
    }


def test_input_structure_gate_rejects_protocol_budget_drift(tmp_path: Path) -> None:
    structural = _load_script(STRUCTURAL_SCRIPT, "v21e3r1_structural_drift")
    repo = tmp_path / "repo"
    paths = _write_contract_files(repo)
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    protocol["common_execution_contract"]["charged_evaluation_budget"] = 1_999
    _write_json(paths["protocol"], protocol)

    with pytest.raises(ValueError, match="budget/checkpoint"):
        structural.audit_target_size_structure(
            repo_root=repo,
            protocol_path=paths["protocol"],
            case_manifest_path=paths["case"],
            reference_manifest_path=paths["reference"],
            config_manifest_path=paths["config"],
            metric_manifest_path=paths["metric"],
            output=repo / "must-not-exist.json",
        )

    assert not (repo / "must-not-exist.json").exists()



def test_snapshot_is_engineering_only_and_binds_immutable_parent(
    tmp_path: Path,
) -> None:
    freeze = _load_script(FREEZE_SCRIPT, "v21e3r1_freeze")
    repo = tmp_path / "repo"
    old_zip = (
        repo
        / "ijoc_submission_v21e3"
        / "release"
        / "ijoc_v21e3_experiment_code.zip"
    )
    # The production constant is deliberately injected into this fixture so
    # the test exercises the immutable-parent gate without embedding a 9 MB ZIP.
    _write(old_zip, b"fixture parent")
    parent_sha256 = hashlib.sha256(old_zip.read_bytes()).hexdigest()
    _write_immutable_parent_chain(repo, parent_sha256)
    _write(
        repo / "mo_nco" / "pareto_v21e3_hybrid.py",
        b"from .runtime_dependency import VALUE\n",
    )
    _write(repo / "mo_nco" / "runtime_dependency.py", b"VALUE = 1\n")
    _write(repo / "tests" / "test_pareto_v21e3_hybrid.py", b"# test\n")
    _write(
        repo
        / "ijoc_submission_v21e3r1"
        / "provenance"
        / "audit_inputs"
        / "audit.txt",
        b"audit\n",
    )
    _write_json(
        repo
        / "ijoc_submission_v21e3"
        / "provenance"
        / "V21E3R1_TRACE_STREAMING_SMALL_SCALE_V6.json",
        {"status": "PASS_SMALL_SCALE_STREAMING_ENGINEERING_ONLY"},
    )
    protocol = repo / "ijoc_submission_v21e3" / "protocol" / "parity.json"
    _write_json(protocol, {"schema": "fixture"})
    structural = repo / "evidence" / "structural.json"
    target_execution = repo / "evidence" / "target-execution.json"
    pytest_receipt = repo / "evidence" / "pytest.json"
    pytest_log = repo / "evidence" / "pytest.log"
    _write_json(
        structural,
        {"status": "PASS_TARGET_SIZE_INPUT_STRUCTURE_ENGINEERING_ONLY"},
    )
    _write_json(
        target_execution,
        {"status": "PASS_TARGET_SIZE_THREE_ARM_EXECUTION_ENGINEERING_ONLY"},
    )
    _write(pytest_log, b"1 passed in 0.01s\n")
    _write_json(
        pytest_receipt,
        {
            "status": "PASS",
            "artifact_path_semantics": "repo_root_relative_posix_v1",
            "log_path": pytest_log.relative_to(repo).as_posix(),
        },
    )
    output = repo / "snapshot.json"

    receipt = freeze.freeze_development_snapshot(
        repo_root=repo,
        output=output,
        protocol_path=protocol,
        input_structural_receipt_path=structural,
        target_execution_receipt_path=target_execution,
        pytest_receipt_path=pytest_receipt,
        expected_v21e3_zip_sha256=parent_sha256,
    )

    assert output.is_file()
    assert receipt["status"] == "PASS_ENGINEERING_SNAPSHOT_ONLY"
    assert receipt["v21e3_immutable_parent_zip_sha256"] == parent_sha256
    assert set(receipt["immutable_parent_chain_bindings"]) == {
        "v21e2_immutable_baseline",
        "v21e2_immutable_calibration_evidence",
        "v21e3_development_snapshot",
        "v21e3_release_manifest",
        "v21e3_clean_room_receipt",
        "v21e3_zip_checksum_file",
    }
    assert receipt["authorization"] == {
        "development_parity_preflight": "NOT_YET_RUN",
        "development_parity_execution": "NOT_AUTHORIZED_BY_SNAPSHOT_ALONE",
        "selection_entropy_release": "PROHIBITED",
        "calibration_execution": "PROHIBITED",
        "formal_execution": "PROHIBITED",
    }
    assert receipt["formal_authorized"] is False
    bound = {entry["path"]: entry for entry in receipt["bound_files"]}
    old_relative = old_zip.relative_to(repo).as_posix()
    assert bound[old_relative]["sha256"] == parent_sha256
    assert receipt["bound_files_root_sha256"] == freeze.bound_files_root(
        receipt["bound_files"]
    )
    with pytest.raises(FileExistsError):
        freeze.freeze_development_snapshot(
            repo_root=repo,
            output=output,
            protocol_path=protocol,
            input_structural_receipt_path=structural,
            target_execution_receipt_path=target_execution,
            pytest_receipt_path=pytest_receipt,
            expected_v21e3_zip_sha256=parent_sha256,
        )


def test_independent_preflight_authorizes_only_development_parity(
    tmp_path: Path,
) -> None:
    structural = _load_script(STRUCTURAL_SCRIPT, "v21e3r1_structural_for_auth")
    freeze = _load_script(FREEZE_SCRIPT, "v21e3r1_freeze_for_auth")
    preflight = _load_script(PREFLIGHT_SCRIPT, "v21e3r1_preflight")
    repo = tmp_path / "repo"
    paths = _write_contract_files(repo)
    old_zip = (
        repo
        / "ijoc_submission_v21e3"
        / "release"
        / "ijoc_v21e3_experiment_code.zip"
    )
    _write(old_zip, b"fixture immutable V21e3 ZIP")
    parent_sha256 = hashlib.sha256(old_zip.read_bytes()).hexdigest()
    _write_immutable_parent_chain(repo, parent_sha256)
    _write(
        repo / "mo_nco" / "pareto_v21e3_hybrid.py",
        b"from .runtime_dependency import VALUE\n",
    )
    _write(repo / "mo_nco" / "runtime_dependency.py", b"VALUE = 1\n")
    _write(repo / "mo_nco" / "pareto_v21e3_baselines.py", b"# baselines\n")
    _write(repo / "tests" / "test_pareto_v21e3_hybrid.py", b"# tests\n")
    _write(
        repo
        / "ijoc_submission_v21e3r1"
        / "provenance"
        / "audit_inputs"
        / "audit.txt",
        b"independent audit input\n",
    )
    _write_json(
        repo
        / "ijoc_submission_v21e3"
        / "provenance"
        / "V21E3R1_TRACE_STREAMING_SMALL_SCALE_V6.json",
        {
            "schema": "pareto_v21e3r1_trace_streaming_small_scale_receipt_v2",
            "status": "PASS_SMALL_SCALE_STREAMING_ENGINEERING_ONLY",
            "target_scale_capacity_status": "NOT_RUN",
            "formal_authorized": False,
        },
    )
    prospective = freeze.compute_prospective_source_root(
        repo_root=repo,
        protocol_path=paths["protocol"],
        expected_v21e3_zip_sha256=parent_sha256,
    )
    structural_path = repo / "evidence" / "structural.json"
    input_structure = structural.audit_target_size_structure(
        repo_root=repo,
        protocol_path=paths["protocol"],
        case_manifest_path=paths["case"],
        reference_manifest_path=paths["reference"],
        config_manifest_path=paths["config"],
        metric_manifest_path=paths["metric"],
        output=structural_path,
    )
    target_execution_path = repo / "evidence" / "target-execution.json"
    _write_json(
        target_execution_path,
        _target_execution_payload(
            repo,
            source_root_sha256=str(prospective["prospective_source_root_sha256"]),
            input_sha256={
                role: binding["sha256"]
                for role, binding in input_structure["bindings"].items()
            },
        ),
    )
    pytest_path = repo / "evidence" / "pytest.json"
    pytest_log = repo / "evidence" / "pytest.log"
    _write(pytest_log, b"700 passed in 1.00s\n")
    executable = Path(sys.executable).resolve()
    _write_json(
        pytest_path,
        {
            "schema": "pareto_v21e3r1_full_pytest_receipt_v1",
            "status": "PASS",
            "suite_scope": "repository_full_pytest_q_v1",
            "prospective_source_root_sha256": prospective[
                "prospective_source_root_sha256"
            ],
            "command": [str(executable), "-m", "pytest", "-q"],
            "cwd": ".",
            "cwd_path_semantics": "repo_root_self_v1",
            "executable": str(executable),
            "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            "artifact_path_semantics": "repo_root_relative_posix_v1",
            "log_path": pytest_log.relative_to(repo).as_posix(),
            "log_sha256": hashlib.sha256(pytest_log.read_bytes()).hexdigest(),
            "log_bytes": pytest_log.stat().st_size,
            "exit_code": 0,
            "passed": 700,
            "failed": 0,
            "errors": 0,
            "selection_authorization": "PROHIBITED",
            "formal_authorized": False,
        },
    )
    snapshot_path = repo / "snapshot.json"
    snapshot = freeze.freeze_development_snapshot(
        repo_root=repo,
        output=snapshot_path,
        protocol_path=paths["protocol"],
        input_structural_receipt_path=structural_path,
        target_execution_receipt_path=target_execution_path,
        pytest_receipt_path=pytest_path,
        expected_v21e3_zip_sha256=parent_sha256,
    )
    output = repo / "authorization.json"

    authorization = preflight.authorize_development_parity(
        repo_root=repo,
        snapshot_path=snapshot_path,
        output=output,
        expected_v21e3_zip_sha256=parent_sha256,
    )

    assert authorization["schema"] == (
        "pareto_v21e3r1_development_parity_authorization_v1"
    )
    assert authorization["status"] == "AUTHORIZED_DEVELOPMENT_PARITY_ONLY"
    assert authorization["source_snapshot_root_sha256"] == snapshot[
        "bound_files_root_sha256"
    ]
    assert authorization["development_parity_execution"] == (
        "AUTHORIZED_DEVELOPMENT_ONLY"
    )
    assert authorization["matched_matrix"] == "AUTHORIZED_DEVELOPMENT_ONLY"
    assert authorization["charged_evaluation_budget"] == 2000
    assert authorization["case_count"] == 12
    assert authorization["seeds"] == [31051, 31057, 31059]
    assert authorization["arms"] == ["V21E3_C0", "NSGAII", "MOEAD"]
    assert set(authorization["bindings"]) == {
        "protocol",
        "case_manifest",
        "reference_manifest",
        "config_manifest",
        "metric_manifest",
    }
    assert authorization["selection_entropy_release"] == "PROHIBITED"
    assert authorization["calibration_execution"] == "PROHIBITED"
    assert authorization["formal_execution"] == "PROHIBITED"
    assert authorization["formal_authorized"] is False
    assert authorization["performance_claim"] == "NOT_ESTABLISHED"
    assert preflight.verify_existing_development_parity_authorization(
        repo_root=repo,
        authorization_path=output,
        expected_source_snapshot_root_sha256=snapshot[
            "bound_files_root_sha256"
        ],
        expected_v21e3_zip_sha256=parent_sha256,
    ) == authorization

    prospective_paths = {
        entry["path"] for entry in snapshot["bound_files"]
    }
    assert "mo_nco/runtime_dependency.py" in prospective_paths

    # A transitive runtime dependency is source, even though its filename does
    # not carry the V21e3 prefix.  Drift must invalidate live authorization.
    _write(repo / "mo_nco" / "runtime_dependency.py", b"VALUE = 2\n")
    with pytest.raises(ValueError, match="snapshot file drifted"):
        preflight.authorize_development_parity(
            repo_root=repo,
            snapshot_path=snapshot_path,
            output=repo / "must-not-authorize.json",
            expected_v21e3_zip_sha256=parent_sha256,
        )
    with pytest.raises(ValueError, match="snapshot file drifted"):
        preflight.verify_existing_development_parity_authorization(
            repo_root=repo,
            authorization_path=output,
            expected_source_snapshot_root_sha256=snapshot[
                "bound_files_root_sha256"
            ],
            expected_v21e3_zip_sha256=parent_sha256,
        )
    assert not (repo / "must-not-authorize.json").exists()

    # Restore the bound bytes, then add a new executable module.  Independent
    # preflight must compare the live expected path set, not merely rehash the
    # paths self-declared by the old snapshot.
    _write(repo / "mo_nco" / "runtime_dependency.py", b"VALUE = 1\n")
    _write(repo / "mo_nco" / "late_injected_module.py", b"VALUE = 3\n")
    with pytest.raises(ValueError, match="source path set drifted"):
        preflight.authorize_development_parity(
            repo_root=repo,
            snapshot_path=snapshot_path,
            output=repo / "must-not-authorize-late-file.json",
            expected_v21e3_zip_sha256=parent_sha256,
        )
    assert not (repo / "must-not-authorize-late-file.json").exists()

