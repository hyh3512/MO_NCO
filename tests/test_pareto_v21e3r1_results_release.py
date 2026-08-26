from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

import pytest

from ijoc_submission_v21e3r1.scripts import build_v21e3r1_results_release as builder


ROOT_SHA = "a" * 64
ARMS = ("V21E3_C0", "NSGAII", "MOEAD")
SEEDS = (31051, 31057, 31059)
ROW_FILES = frozenset(
    {
        "trace.sqlite3",
        "terminal.receipt.json",
        "row.preverification.json",
        "objective_archive_replay.receipt.json",
        "metric_replay.receipt.json",
        "row.json",
    }
)


def _canonical(value: object, *, newline: bool = True) -> bytes:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _write_json(path: Path, value: object, *, newline: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value, newline=newline))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _fake_sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _checkpoints(
    *, budget: int, period: int, terminal_hv: float
) -> list[dict[str, object]]:
    return [
        {
            "evaluation": evaluation,
            "normalized_hv": terminal_hv * (evaluation / budget),
            "archive_size": max(1, evaluation // period),
        }
        for evaluation in range(period, budget + 1, period)
    ]


def _left_continuous_auc(
    checkpoints: list[dict[str, object]], *, budget: int
) -> float:
    previous_evaluation = 0
    previous_hv = 0.0
    area = 0.0
    for checkpoint in checkpoints:
        evaluation = int(checkpoint["evaluation"])
        area += previous_hv * (evaluation - previous_evaluation)
        previous_evaluation = evaluation
        previous_hv = float(checkpoint["normalized_hv"])
    return area / float(budget)


def _terminal_receipt(
    *,
    problem: str,
    family: str,
    budget: int,
    run_context_sha: str,
    attempt_chain: str,
    decision_chain: str,
    evaluation_chain: str,
) -> dict[str, object]:
    core: dict[str, object] = {
        "schema": "v21e3_terminal_receipt_v1",
        "status": "SUCCESS",
        "problem": problem,
        "family": family,
        "failure_code": None,
        "failure_detail": None,
        "attempt_count": budget,
        "physical_call_started_count": budget,
        "charged_evaluation_count": budget,
        "decision_count": budget,
        "cache_hit_count": 0,
        "unresolved_decision_count": 0,
        "terminal_evaluation_chain_sha256": evaluation_chain,
        "terminal_decision_chain_sha256": decision_chain,
        "terminal_attempt_chain_sha256": attempt_chain,
        "run_context_digest_sha256": run_context_sha,
        "database_path": "trace.sqlite3",
        "durability_mode": "SQLITE_WAL_SYNCHRONOUS_FULL",
        "finalization_gates": {
            "expected_charged_evaluations": budget,
            "expected_decisions": budget,
            "run_context_charged_evaluation_budget": budget,
            "persisted_attempts": budget,
            "persisted_evaluations": budget,
            "persisted_decisions": budget,
            "physical_call_starts": budget,
            "cache_hits": 0,
            "nonterminal_attempts": 0,
            "evaluation_index_bounds": [1, budget],
            "expected_evaluation_index_bounds": [1, budget],
            "sqlite_integrity": "ok",
        },
    }
    return {
        **core,
        "receipt_payload_sha256": hashlib.sha256(
            _canonical(core, newline=False)
        ).hexdigest(),
    }


def _make_code_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_STORED) as archive:
        info = zipfile.ZipInfo("code/README.txt", (1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        info.compress_type = zipfile.ZIP_STORED
        archive.writestr(info, b"synthetic code archive\n")


def _make_synthetic_inputs(root: Path) -> dict[str, Path]:
    evidence = root / "evidence"
    frozen = root / "frozen"
    code = root / "code"
    matrix = root / "matrix"
    case_specs: list[dict[str, object]] = []
    reference_cases: list[dict[str, object]] = []
    reference_boxes: dict[str, dict[str, object]] = {}
    for case_index in range(12):
        family = "MOTSP" if case_index < 6 else "MOKP"
        size = (100, 200, 500)[case_index % 3]
        case_id = f"synthetic-{family.lower()}-{case_index:02d}"
        lower = [float(case_index + 1), float(case_index + 2)]
        upper = [float(case_index + 101), float(case_index + 202)]
        case_specs.append({"case_id": case_id, "family": family, "size": size})
        reference_cases.append(
            {
                "case_id": case_id,
                "family": family,
                "size": size,
                "objective_lower_bounds": lower,
                "objective_upper_bounds": upper,
                "normalized_reference_point": [1.0, 1.0],
            }
        )
        reference_boxes[case_id] = {
            "lower": lower,
            "upper": upper,
            "normalized_reference": [1.0, 1.0],
        }

    manifests = {
        "build_manifest_path": (
            frozen / "build_receipt.json",
            "pareto_v21e3_development_manifest_build_receipt_v1",
            "PASS",
        ),
        "config_manifest_path": (
            frozen / "config_manifest_development.json",
            "pareto_v21e3_development_config_manifest_v1",
            "FROZEN_DEVELOPMENT_INPUT_CALIBRATION_EXECUTION_BLOCKED",
        ),
        "metric_manifest_path": (
            frozen / "metric_manifest.json",
            "pareto_v21e3_metric_manifest_v1",
            "FROZEN_DEVELOPMENT_AND_FUTURE_CALIBRATION_INPUT",
        ),
        "reference_manifest_path": (
            frozen / "reference_manifest_development.json",
            "pareto_v21e3_analytic_reference_manifest_v1",
            "FROZEN_DEVELOPMENT_ONLY",
        ),
    }
    for key, (path, schema, status) in manifests.items():
        payload: dict[str, object] = {"schema": schema, "status": status}
        if key == "reference_manifest_path":
            payload["cases"] = reference_cases
        _write_json(path, payload)
    protocol = frozen / "V21E3_C0_PARITY_PROTOCOL_V2.json"
    _write_json(
        protocol,
        {
            "schema": "pareto_v21e3_c0_parity_protocol_v2",
            "status": "ENGINEERING_ADAPTERS_AVAILABLE_SUCCESSOR_SNAPSHOT_PENDING",
        },
    )

    target = root / "target_size_execution_v4"
    target_rows: list[dict[str, object]] = []
    for family in ("MOTSP", "MOKP"):
        target_case = next(
            case
            for case in case_specs
            if case["family"] == family and case["size"] == 500
        )
        case_id = str(target_case["case_id"])
        for arm_id in ARMS:
            target_rows.append(
                {
                    "case_id": case_id,
                    "family": family,
                    "size": 500,
                    "seed": 31051,
                    "arm_id": arm_id,
                    "row_slug": (
                        f"{case_id}__seed-31051__arm-{arm_id.lower()}"
                    ),
                }
            )
    target_source_root = "b" * 64
    input_sha256 = {
        "case_manifest": "c" * 64,
        "config_manifest": _sha256(manifests["config_manifest_path"][0]),
        "metric_manifest": _sha256(manifests["metric_manifest_path"][0]),
        "protocol": _sha256(protocol),
        "reference_manifest": _sha256(manifests["reference_manifest_path"][0]),
    }
    target_plan = target / "target_structural.plan.json"
    _write_json(
        target_plan,
        {
            "schema": "pareto_v21e3r1_target_size_three_arm_plan_v1",
            "status": "READY_TARGET_SIZE_SMALL_BUDGET_ENGINEERING_ONLY",
            "scientific_scope": (
                "target_size_small_budget_structure_and_objective_archive_replay_"
                "not_performance_evidence"
            ),
            "source_snapshot_root_sha256": target_source_root,
            "input_sha256": input_sha256,
            "budget": 200,
            "checkpoint_period": 200,
            "baseline_population_sizes": {"MOTSP": 48, "MOKP": 40},
            "budget_initializes_every_baseline_population": True,
            "rows": target_rows,
            "development_parity_execution": "NOT_AUTHORIZED_BY_THIS_PLAN",
            "selection_entropy_release": "PROHIBITED",
            "calibration_execution": "PROHIBITED",
            "formal_execution": "PROHIBITED",
            "formal_authorized": False,
        },
    )
    target_summaries: list[dict[str, object]] = []
    child_bindings: list[dict[str, object]] = []
    target_repo_prefix = (
        "ijoc_submission_v21e3r1/provenance/target_size_execution_v4"
    )
    for ordinal, planned in enumerate(target_rows):
        row_dir = target / "rows" / str(planned["row_slug"])
        row_dir.mkdir(parents=True)
        trace = row_dir / "trace.sqlite3"
        trace.write_bytes(f"target-sqlite-{ordinal}\n".encode("ascii"))
        terminal_hv = ordinal / 6.0
        checkpoints = _checkpoints(
            budget=200, period=200, terminal_hv=terminal_hv
        )
        auc = _left_continuous_auc(checkpoints, budget=200)
        run_context_sha = _fake_sha(f"target-context-{ordinal}")
        attempt_chain = _fake_sha(f"target-attempt-{ordinal}")
        decision_chain = _fake_sha(f"target-decision-{ordinal}")
        evaluation_chain = _fake_sha(f"target-evaluation-{ordinal}")
        terminal_payload = _terminal_receipt(
            problem=str(planned["case_id"]),
            family=str(planned["family"]),
            budget=200,
            run_context_sha=run_context_sha,
            attempt_chain=attempt_chain,
            decision_chain=decision_chain,
            evaluation_chain=evaluation_chain,
        )
        terminal_payload_sha = str(terminal_payload["receipt_payload_sha256"])
        terminal = row_dir / "terminal.receipt.json"
        _write_json(terminal, terminal_payload, newline=False)
        pre = row_dir / "row.preverification.json"
        _write_json(
            pre,
            {
                "schema": "pareto_v21e3r1_row_preverification_v1",
                "status": "READY_FOR_OBJECTIVE_AND_ARCHIVE_REPLAY",
                **{
                    key: planned[key]
                    for key in ("case_id", "family", "size", "seed", "arm_id")
                },
                "source_snapshot_root_sha256": target_source_root,
                "trace_database_path": "trace.sqlite3",
                "detached_terminal_receipt_path": "terminal.receipt.json",
                "detached_terminal_receipt_sha256": _sha256(terminal),
                "selection_entropy_release": "PROHIBITED",
                "calibration_execution": "PROHIBITED",
                "formal_execution": "PROHIBITED",
                "formal_authorized": False,
            },
        )
        objective = row_dir / "objective_archive_replay.receipt.json"
        _write_json(
            objective,
            {
                "schema": "v21e3r1_objective_archive_replay_receipt_v2",
                "status": "OBJECTIVE_AND_ARCHIVE_REPLAY_PASS",
                "database_path": "trace.sqlite3",
                "database_bytes": trace.stat().st_size,
                "database_sha256": _sha256(trace),
                "detached_terminal_receipt_path": "terminal.receipt.json",
                "detached_terminal_receipt_sha256": _sha256(terminal),
                "verification_scope": (
                    "objective_solution_chain_archive_and_terminal_replay_v1"
                ),
                "attempt_records": 200,
                "evaluation_records": 200,
                "decision_records": 200,
                "cache_hit_records": 0,
                "unique_solution_replays": 200,
                "archive_reconstruction": "PASS",
                "archive_size": int(checkpoints[-1]["archive_size"]),
                "terminal_status": "SUCCESS",
                "run_context_digest_sha256": run_context_sha,
                "terminal_attempt_chain_sha256": attempt_chain,
                "terminal_decision_chain_sha256": decision_chain,
                "terminal_evaluation_chain_sha256": evaluation_chain,
                "terminal_receipt_sha256": terminal_payload_sha,
                "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
                "selection_authorization": "PROHIBITED",
            },
        )
        metric = row_dir / "metric_replay.receipt.json"
        _write_json(
            metric,
            {
                "schema": "pareto_v21e3r1_metric_replay_receipt_v1",
                "status": "NORMALIZED_HV_AUC_REPLAY_PASS",
                "database_path": "trace.sqlite3",
                "database_sha256": _sha256(trace),
                "metric_manifest_sha256": input_sha256["metric_manifest"],
                "verification_scope": (
                    "frozen_metric_from_objective_ledger_checkpoints_v1"
                ),
                "charged_evaluation_budget": 200,
                "checkpoint_period": 200,
                "analytic_box": reference_boxes[str(planned["case_id"])],
                "normalized_left_continuous_hv_auc": auc,
                "normalized_terminal_hv": terminal_hv,
                "checkpoints": checkpoints,
                "selection_authorization": "PROHIBITED",
                "formal_authorized": False,
            },
        )
        row_path = row_dir / "row.json"
        row_payload = {
            "schema": "pareto_v21e3r1_matched_matrix_row_v1",
            "status": "PASS_ENGINEERING_ROW_OBJECTIVE_ARCHIVE_REPLAYED",
            "scientific_scope": (
                "target_size_small_budget_structure_and_objective_archive_replay_"
                "not_performance_evidence"
            ),
            "artifact_path_semantics": "row_directory_relative_posix_v1",
            **{key: planned[key] for key in ("case_id", "family", "size", "seed", "arm_id")},
            "charged_evaluation_budget": 200,
            "checkpoint_period": 200,
            "source_snapshot_root_sha256": target_source_root,
            "run_context_digest_sha256": run_context_sha,
            "trace_database_path": "trace.sqlite3",
            "trace_database_bytes": trace.stat().st_size,
            "trace_database_sha256": _sha256(trace),
            "detached_terminal_receipt_path": "terminal.receipt.json",
            "detached_terminal_receipt_sha256": _sha256(terminal),
            "preverification_receipt_path": "row.preverification.json",
            "preverification_receipt_sha256": _sha256(pre),
            "objective_archive_replay_receipt_path": (
                "objective_archive_replay.receipt.json"
            ),
            "objective_archive_replay_receipt_sha256": _sha256(objective),
            "metric_replay_receipt_path": "metric_replay.receipt.json",
            "metric_replay_receipt_sha256": _sha256(metric),
            "metric_manifest_sha256": input_sha256["metric_manifest"],
            "normalized_left_continuous_hv_auc": auc,
            "normalized_terminal_hv": terminal_hv,
            "checkpoints": checkpoints,
            "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
            "runtime_efficiency_claim_authorized": False,
            "selection_entropy_release": "PROHIBITED",
            "calibration_execution": "PROHIBITED",
            "formal_execution": "PROHIBITED",
            "formal_authorized": False,
        }
        _write_json(row_path, row_payload)
        row_repo_path = f"{target_repo_prefix}/rows/{planned['row_slug']}/row.json"
        summary = {
            **{
                key: row_payload[key]
                for key in (
                    "case_id",
                    "family",
                    "size",
                    "seed",
                    "arm_id",
                    "charged_evaluation_budget",
                    "checkpoint_period",
                    "trace_database_sha256",
                    "detached_terminal_receipt_sha256",
                    "objective_archive_replay_receipt_sha256",
                    "metric_replay_receipt_sha256",
                )
            },
            "objective_and_archive_replay": "PASS",
            "metric_replay": "PASS",
            "normalized_terminal_hv_diagnostic_only": row_payload[
                "normalized_terminal_hv"
            ],
            "row_receipt_path": row_repo_path,
            "row_receipt_sha256": _sha256(row_path),
        }
        target_summaries.append(summary)
        for child_name in sorted(ROW_FILES):
            child = row_dir / child_name
            child_bindings.append(
                {
                    "path": f"{target_repo_prefix}/rows/{planned['row_slug']}/{child_name}",
                    "bytes": child.stat().st_size,
                    "sha256": _sha256(child),
                }
            )
    child_bindings.sort(key=lambda item: item["path"])
    target_child_root = hashlib.sha256(_canonical(child_bindings)).hexdigest()
    target_receipt = target / "V21E3R1_TARGET_SIZE_EXECUTION_RECEIPT_V1.json"
    _write_json(
        target_receipt,
        {
            "schema": "pareto_v21e3r1_target_size_execution_receipt_v1",
            "status": "PASS_TARGET_SIZE_THREE_ARM_EXECUTION_ENGINEERING_ONLY",
            "scientific_scope": (
                "target_size_small_budget_structure_and_objective_archive_replay_"
                "not_performance_evidence"
            ),
            "artifact_path_semantics": "repo_root_relative_posix_v1",
            "source_snapshot_root_sha256": target_source_root,
            "input_sha256": input_sha256,
            "plan_path": f"{target_repo_prefix}/target_structural.plan.json",
            "plan_sha256": _sha256(target_plan),
            "target_size": 500,
            "seed": 31051,
            "charged_evaluation_budget": 200,
            "checkpoint_period": 200,
            "baseline_population_sizes": {"MOTSP": 48, "MOKP": 40},
            "budget_initializes_every_baseline_population": True,
            "row_count": 6,
            "families": ["MOTSP", "MOKP"],
            "arms": list(ARMS),
            "rows": target_summaries,
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
        },
    )

    snapshot = evidence / "V4_SOURCE_SNAPSHOT.json"
    _write_json(
        snapshot,
        {
            "schema": "pareto_v21e3r1_development_source_snapshot_freeze_v1",
            "status": "PASS_ENGINEERING_SNAPSHOT_ONLY",
            "bound_files_root_sha256": ROOT_SHA,
            "prospective_source_root_sha256": target_source_root,
            "target_size_execution_receipt_path": (
                f"{target_repo_prefix}/V21E3R1_TARGET_SIZE_EXECUTION_RECEIPT_V1.json"
            ),
            "formal_authorized": False,
            "submission_status": "IJOC_HOLD",
        },
    )
    authorization = evidence / "V4_AUTHORIZATION.json"
    _write_json(
        authorization,
        {
            "schema": "pareto_v21e3r1_development_parity_authorization_v1",
            "status": "AUTHORIZED_DEVELOPMENT_PARITY_ONLY",
            "source_snapshot_receipt_sha256": _sha256(snapshot),
            "source_snapshot_root_sha256": ROOT_SHA,
            "target_size_execution_receipt_sha256": _sha256(target_receipt),
            "target_size_execution_child_artifact_root_sha256": target_child_root,
            "selection_entropy_release": "PROHIBITED",
            "calibration_execution": "PROHIBITED",
            "formal_execution": "PROHIBITED",
            "formal_authorized": False,
            "submission_status": "IJOC_HOLD",
        },
    )
    authorization_sha = _sha256(authorization)

    planned_rows: list[dict[str, object]] = []
    for case_index in range(12):
        family = "MOTSP" if case_index < 6 else "MOKP"
        size = (100, 200, 500)[case_index % 3]
        case_id = f"synthetic-{family.lower()}-{case_index:02d}"
        for seed in SEEDS:
            for arm_id in ARMS:
                planned_rows.append(
                    {
                        "case_id": case_id,
                        "family": family,
                        "size": size,
                        "seed": seed,
                        "arm_id": arm_id,
                        "row_slug": (
                            f"{case_id}__seed-{seed}__arm-{arm_id.lower()}"
                        ),
                    }
                )
    assert len(planned_rows) == 108
    plan = matrix / "matrix.plan.json"
    _write_json(
        plan,
        {
            "schema": "pareto_v21e3r1_development_matched_matrix_plan_v1",
            "status": "AUTHORIZED_DEVELOPMENT_MATRIX_PLAN",
            "scientific_scope": (
                "authors_generated_development_only_not_formal_evidence"
            ),
            "expected_rows": 108,
            "budget": 2000,
            "checkpoint_period": 200,
            "source_snapshot_root_sha256": ROOT_SHA,
            "authorization_receipt_sha256": authorization_sha,
            "input_sha256": input_sha256,
            "rows": planned_rows,
            "selection_entropy_release": "PROHIBITED",
            "calibration_execution": "PROHIBITED",
            "formal_execution": "PROHIBITED",
            "formal_authorized": False,
        },
    )

    aggregate_rows: list[dict[str, object]] = []
    for ordinal, planned in enumerate(planned_rows):
        row_dir = matrix / "rows" / str(planned["row_slug"])
        row_dir.mkdir(parents=True)
        trace = row_dir / "trace.sqlite3"
        trace.write_bytes(f"synthetic-sqlite-{ordinal:03d}\n".encode("ascii"))
        terminal_hv = ordinal / 108.0
        checkpoints = _checkpoints(
            budget=2000, period=200, terminal_hv=terminal_hv
        )
        auc = _left_continuous_auc(checkpoints, budget=2000)
        run_context_sha = _fake_sha(f"matrix-context-{ordinal}")
        attempt_chain = _fake_sha(f"matrix-attempt-{ordinal}")
        decision_chain = _fake_sha(f"matrix-decision-{ordinal}")
        evaluation_chain = _fake_sha(f"matrix-evaluation-{ordinal}")
        terminal_payload = _terminal_receipt(
            problem=str(planned["case_id"]),
            family=str(planned["family"]),
            budget=2000,
            run_context_sha=run_context_sha,
            attempt_chain=attempt_chain,
            decision_chain=decision_chain,
            evaluation_chain=evaluation_chain,
        )
        terminal_payload_sha = str(terminal_payload["receipt_payload_sha256"])
        terminal = row_dir / "terminal.receipt.json"
        _write_json(terminal, terminal_payload, newline=False)
        preverification = row_dir / "row.preverification.json"
        _write_json(
            preverification,
            {
                "schema": "pareto_v21e3r1_row_preverification_v1",
                "status": "READY_FOR_OBJECTIVE_AND_ARCHIVE_REPLAY",
                **{
                    key: planned[key]
                    for key in ("case_id", "family", "size", "seed", "arm_id")
                },
                "source_snapshot_root_sha256": ROOT_SHA,
                "trace_database_path": "trace.sqlite3",
                "detached_terminal_receipt_path": "terminal.receipt.json",
                "detached_terminal_receipt_sha256": _sha256(terminal),
                "selection_entropy_release": "PROHIBITED",
                "calibration_execution": "PROHIBITED",
                "formal_execution": "PROHIBITED",
                "formal_authorized": False,
            },
        )
        objective = row_dir / "objective_archive_replay.receipt.json"
        _write_json(
            objective,
            {
                "schema": "v21e3r1_objective_archive_replay_receipt_v2",
                "status": "OBJECTIVE_AND_ARCHIVE_REPLAY_PASS",
                "database_path": "trace.sqlite3",
                "database_bytes": trace.stat().st_size,
                "database_sha256": _sha256(trace),
                "detached_terminal_receipt_path": "terminal.receipt.json",
                "detached_terminal_receipt_sha256": _sha256(terminal),
                "verification_scope": (
                    "objective_solution_chain_archive_and_terminal_replay_v1"
                ),
                "attempt_records": 2000,
                "evaluation_records": 2000,
                "decision_records": 2000,
                "cache_hit_records": 0,
                "unique_solution_replays": 2000,
                "archive_reconstruction": "PASS",
                "archive_size": int(checkpoints[-1]["archive_size"]),
                "terminal_status": "SUCCESS",
                "run_context_digest_sha256": run_context_sha,
                "terminal_attempt_chain_sha256": attempt_chain,
                "terminal_decision_chain_sha256": decision_chain,
                "terminal_evaluation_chain_sha256": evaluation_chain,
                "terminal_receipt_sha256": terminal_payload_sha,
                "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
                "selection_authorization": "PROHIBITED",
            },
        )
        metric = row_dir / "metric_replay.receipt.json"
        _write_json(
            metric,
            {
                "schema": "pareto_v21e3r1_metric_replay_receipt_v1",
                "status": "NORMALIZED_HV_AUC_REPLAY_PASS",
                "database_path": "trace.sqlite3",
                "database_sha256": _sha256(trace),
                "metric_manifest_sha256": _sha256(
                    manifests["metric_manifest_path"][0]
                ),
                "verification_scope": (
                    "frozen_metric_from_objective_ledger_checkpoints_v1"
                ),
                "charged_evaluation_budget": 2000,
                "checkpoint_period": 200,
                "analytic_box": reference_boxes[str(planned["case_id"])],
                "normalized_left_continuous_hv_auc": auc,
                "normalized_terminal_hv": terminal_hv,
                "checkpoints": checkpoints,
                "selection_authorization": "PROHIBITED",
                "formal_authorized": False,
            },
        )
        row = row_dir / "row.json"
        row_payload = {
            "schema": "pareto_v21e3r1_matched_matrix_row_v1",
            "status": "PASS_ENGINEERING_ROW_OBJECTIVE_ARCHIVE_REPLAYED",
            "scientific_scope": (
                "authors_generated_development_only_not_formal_evidence"
            ),
            "artifact_path_semantics": "row_directory_relative_posix_v1",
            **{key: planned[key] for key in ("case_id", "family", "size", "seed", "arm_id")},
            "charged_evaluation_budget": 2000,
            "checkpoint_period": 200,
            "source_snapshot_root_sha256": ROOT_SHA,
            "run_context_digest_sha256": run_context_sha,
            "trace_database_path": "trace.sqlite3",
            "trace_database_bytes": trace.stat().st_size,
            "trace_database_sha256": _sha256(trace),
            "detached_terminal_receipt_path": "terminal.receipt.json",
            "detached_terminal_receipt_sha256": _sha256(terminal),
            "preverification_receipt_path": "row.preverification.json",
            "preverification_receipt_sha256": _sha256(preverification),
            "objective_archive_replay_receipt_path": (
                "objective_archive_replay.receipt.json"
            ),
            "objective_archive_replay_receipt_sha256": _sha256(objective),
            "metric_replay_receipt_path": "metric_replay.receipt.json",
            "metric_replay_receipt_sha256": _sha256(metric),
            "metric_manifest_sha256": input_sha256["metric_manifest"],
            "normalized_left_continuous_hv_auc": auc,
            "normalized_terminal_hv": terminal_hv,
            "checkpoints": checkpoints,
            "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
            "runtime_efficiency_claim_authorized": False,
            "selection_entropy_release": "PROHIBITED",
            "calibration_execution": "PROHIBITED",
            "formal_execution": "PROHIBITED",
            "formal_authorized": False,
        }
        _write_json(row, row_payload)
        aggregate_rows.append(
            {
                **{
                    key: planned[key]
                    for key in ("case_id", "family", "size", "seed", "arm_id")
                },
                "source_snapshot_root_sha256": ROOT_SHA,
                "trace_database_sha256": row_payload["trace_database_sha256"],
                "detached_terminal_receipt_sha256": row_payload[
                    "detached_terminal_receipt_sha256"
                ],
                "objective_archive_replay_receipt_sha256": row_payload[
                    "objective_archive_replay_receipt_sha256"
                ],
                "metric_replay_receipt_sha256": row_payload[
                    "metric_replay_receipt_sha256"
                ],
                "row_receipt_path": f"rows/{planned['row_slug']}/row.json",
                "row_receipt_sha256": _sha256(row),
                "normalized_left_continuous_hv_auc": row_payload[
                    "normalized_left_continuous_hv_auc"
                ],
                "normalized_terminal_hv": row_payload["normalized_terminal_hv"],
            }
        )

    aggregate = matrix / "matrix.aggregate.json"
    _write_json(
        aggregate,
        {
            "schema": "pareto_v21e3r1_development_matched_matrix_aggregate_v1",
            "status": "COMPLETE_DEVELOPMENT_MATRIX_ENGINEERING_EVIDENCE",
            "scientific_scope": (
                "authors_generated_development_only_not_formal_evidence"
            ),
            "artifact_path_semantics": "matrix_directory_relative_posix_v1",
            "source_snapshot_root_sha256": ROOT_SHA,
            "authorization_receipt_sha256": authorization_sha,
            "matrix_plan_sha256": _sha256(plan),
            "input_sha256": input_sha256,
            "expected_rows": 108,
            "observed_rows": 108,
            "rows": aggregate_rows,
            "analysis": {"overall_gate": "PASS_DEVELOPMENT_NONINFERIORITY"},
            "runtime_efficiency_claim_authorized": False,
            "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
            "selection_entropy_release": "PROHIBITED",
            "calibration_execution": "PROHIBITED",
            "formal_execution": "PROHIBITED",
            "formal_authorized": False,
        },
    )
    post_run = matrix / "post_run_audit.json"
    _write_json(
        post_run,
        {
            "schema": "pareto_v21e3r1_development_matrix_post_run_audit_v1",
            "status": "PASS_COMPLETE_DEVELOPMENT_MATRIX_AUDITED",
            "matrix_aggregate_path": "matrix.aggregate.json",
            "matrix_aggregate_sha256": _sha256(aggregate),
            "source_snapshot_root_sha256": ROOT_SHA,
            "authorization_receipt_sha256": authorization_sha,
            "expected_rows": 108,
            "observed_rows": 108,
            "objective_and_archive_replay_pass_rows": 108,
            "metric_replay_pass_rows": 108,
            "runtime_efficiency_claim_authorized": False,
            "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
            "selection_entropy_release": "PROHIBITED",
            "calibration_execution": "PROHIBITED",
            "formal_execution": "PROHIBITED",
            "formal_authorized": False,
        },
    )
    same_implementation = evidence / "SAME_IMPLEMENTATION_RECEIPT_V6.json"
    owned_files = [
        {
            "path": (
                "ijoc_submission_v21e3r1/scripts/"
                "audit_v21e3r1_development_matrix.py"
            ),
            "bytes": 123,
            "sha256": _fake_sha("live-verifier-owned-file"),
        }
    ]
    _write_json(
        same_implementation,
        {
            "schema": (
                "pareto_v21e3r1_same_implementation_development_matrix_"
                "post_run_audit_v1"
            ),
            "status": "PASS_SAME_IMPLEMENTATION_POST_PROCESS_RECOMPUTATION",
            "implementation_independence": False,
            "scientific_independence": False,
            "external_third_party_audit": False,
            "fixed_author_generated_cases_descriptive_only": True,
            "population_inference_authorized": False,
            "sign_flip_assumptions_verified": False,
            "trimmed_mean_distinct_from_mean": False,
            "verifier_relationship": (
                "SAME_PROJECT_VERIFIER_POST_HOC_SUCCESSOR_"
                "NOT_HISTORICAL_PRODUCER"
            ),
            "historical_matrix_producer": {
                "source_snapshot_root_sha256": ROOT_SHA,
                "authorization_receipt_sha256": authorization_sha,
            },
            "live_verifier_owned_file_count": len(owned_files),
            "live_verifier_owned_files": owned_files,
            "live_verifier_owned_files_root_sha256": hashlib.sha256(
                _canonical(owned_files)
            ).hexdigest(),
            "source_snapshot_root_sha256": ROOT_SHA,
            "authorization_receipt_sha256": authorization_sha,
            "matrix_plan_sha256": _sha256(plan),
            "matrix_aggregate_sha256": _sha256(aggregate),
            "runner_post_run_audit_sha256": _sha256(post_run),
            "objective_archive_and_metric_replayed_rows": 108,
            "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
            "selection_entropy_release": "PROHIBITED",
            "calibration_execution": "PROHIBITED",
            "formal_execution": "PROHIBITED",
            "formal_authorized": False,
            "submission_status": "IJOC_HOLD",
        },
    )
    invalidation = evidence / "V3_INVALIDATION.json"
    _write_json(
        invalidation,
        {
            "schema": "pareto_v21e3r1_v3_invalidation_receipt_v1",
            "status": (
                "INVALIDATED_POST_EXECUTION_UNBOUND_DYNAMIC_TEST_DEPENDENCY_"
                "AND_FAILED_CLEAN_ROOM_P0"
            ),
            "scientific_disposition": {"scientific_result": "NONE"},
            "v3_reuse_for_v4": "PROHIBITED",
            "v3_value_status": "NON_AUTHORITATIVE_DEVELOPMENT_DIAGNOSTIC",
            "formal_authorized": False,
            "invalidation_findings": [
                {
                    "code": "P0_UNBOUND_DYNAMIC_TEST_DEPENDENCY",
                    "status": "CONFIRMED",
                },
                {"code": "P0_FAILED_CLEAN_ROOM", "status": "CONFIRMED"},
            ],
            "v4_supersession": {
                "status": "BOUND_AUTHORIZED_SUCCESSOR",
                "source_snapshot_root_sha256": ROOT_SHA,
                "source_snapshot": {
                    "path": "evidence/V4_SOURCE_SNAPSHOT.json",
                    **_binding(snapshot),
                },
                "development_authorization": {
                    "path": "evidence/V4_AUTHORIZATION.json",
                    **_binding(authorization),
                },
            },
            "submission_status": "IJOC_HOLD",
        },
    )

    code_archive = code / "ijoc_v21e3r1_experiment_code_v4.zip"
    _make_code_archive(code_archive)
    code_binding = _binding(code_archive)
    code_manifest = code / "ijoc_v21e3r1_experiment_code_v4.manifest.json"
    _write_json(
        code_manifest,
        {
            "schema": "ijoc_v21e3r1_standalone_release_manifest_v1",
            "archive": code_binding,
            "formal_authorized": False,
            "formal_status": "NOT_MATERIALIZED",
        },
    )
    code_checksum = code / "ijoc_v21e3r1_experiment_code_v4.zip.sha256"
    code_checksum.write_text(
        f"{code_binding['sha256']}  {code_archive.name}\n", encoding="ascii"
    )
    clean_room = code / "ijoc_v21e3r1_clean_room_v4.receipt.json"
    code_manifest_binding = _binding(code_manifest)
    _write_json(
        clean_room,
        {
            "schema": "ijoc_v21e3r1_clean_room_gate_receipt_v2",
            "status": "PASS",
            "archive_verification": {
                "status": "PASS",
                "archive_sha256": code_binding["sha256"],
                "manifest_sha256": code_manifest_binding["sha256"],
            },
            "pinned_inputs": {
                "archive_sha256": code_binding["sha256"],
                "archive_bytes": code_binding["bytes"],
                "manifest_sha256": code_manifest_binding["sha256"],
                "manifest_bytes": code_manifest_binding["bytes"],
            },
            "deterministic_rebuild_comparison_gate": "PASS",
            "formal_authorized": False,
            "formal_status": "NOT_MATERIALIZED",
        },
    )

    return {
        "matrix_directory": matrix,
        "target_execution_directory": target,
        "same_implementation_receipt_path": same_implementation,
        "v3_invalidation_path": invalidation,
        "v4_snapshot_path": snapshot,
        "v4_authorization_path": authorization,
        **{key: value[0] for key, value in manifests.items()},
        "protocol_path": protocol,
        "code_archive_path": code_archive,
        "code_manifest_path": code_manifest,
        "code_checksum_path": code_checksum,
        "code_clean_room_receipt_path": clean_room,
    }


def _outputs(root: Path) -> dict[str, Path]:
    return {
        "archive_path": root / "ijoc_v21e3r1_results_release_v1.zip",
        "file_manifest_path": root
        / "ijoc_v21e3r1_results_release_v1.manifest.json",
        "release_index_path": root
        / "ijoc_v21e3r1_results_release_v1.index.json",
    }


def _coherently_rehash_matrix_row_packet(
    inputs: dict[str, Path], row_directory: Path
) -> None:
    terminal_path = row_directory / "terminal.receipt.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal_core = dict(terminal)
    terminal_core.pop("receipt_payload_sha256", None)
    terminal["receipt_payload_sha256"] = hashlib.sha256(
        _canonical(terminal_core, newline=False)
    ).hexdigest()
    _write_json(terminal_path, terminal, newline=False)

    pre_path = row_directory / "row.preverification.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    pre["detached_terminal_receipt_sha256"] = _sha256(terminal_path)
    _write_json(pre_path, pre)

    objective_path = row_directory / "objective_archive_replay.receipt.json"
    objective = json.loads(objective_path.read_text(encoding="utf-8"))
    objective["detached_terminal_receipt_sha256"] = _sha256(terminal_path)
    objective["terminal_receipt_sha256"] = terminal["receipt_payload_sha256"]
    _write_json(objective_path, objective)

    metric_path = row_directory / "metric_replay.receipt.json"
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    row_path = row_directory / "row.json"
    row = json.loads(row_path.read_text(encoding="utf-8"))
    row["detached_terminal_receipt_sha256"] = _sha256(terminal_path)
    row["preverification_receipt_sha256"] = _sha256(pre_path)
    row["objective_archive_replay_receipt_sha256"] = _sha256(objective_path)
    row["metric_replay_receipt_sha256"] = _sha256(metric_path)
    for field in (
        "normalized_left_continuous_hv_auc",
        "normalized_terminal_hv",
        "checkpoints",
    ):
        row[field] = metric[field]
    _write_json(row_path, row)

    aggregate_path = inputs["matrix_directory"] / "matrix.aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate_row = next(
        item
        for item in aggregate["rows"]
        if all(
            item[field] == row[field]
            for field in ("case_id", "family", "size", "seed", "arm_id")
        )
    )
    for field in (
        "trace_database_sha256",
        "detached_terminal_receipt_sha256",
        "objective_archive_replay_receipt_sha256",
        "metric_replay_receipt_sha256",
        "normalized_left_continuous_hv_auc",
        "normalized_terminal_hv",
    ):
        aggregate_row[field] = row[field]
    aggregate_row["row_receipt_sha256"] = _sha256(row_path)
    _write_json(aggregate_path, aggregate)

    post_run_path = inputs["matrix_directory"] / "post_run_audit.json"
    post_run = json.loads(post_run_path.read_text(encoding="utf-8"))
    post_run["matrix_aggregate_sha256"] = _sha256(aggregate_path)
    _write_json(post_run_path, post_run)
    receipt_path = inputs["same_implementation_receipt_path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["matrix_aggregate_sha256"] = _sha256(aggregate_path)
    receipt["runner_post_run_audit_sha256"] = _sha256(post_run_path)
    _write_json(receipt_path, receipt)


def test_results_release_is_byte_deterministic_zip64_and_code_data_separated(
    tmp_path: Path,
) -> None:
    inputs = _make_synthetic_inputs(tmp_path / "inputs")
    first = _outputs(tmp_path / "release-a")
    second = _outputs(tmp_path / "release-b")
    builder.build_results_release(**inputs, **first)
    builder.build_results_release(**inputs, **second)

    for key in first:
        assert first[key].read_bytes() == second[key].read_bytes()
    manifest = json.loads(first["file_manifest_path"].read_text(encoding="utf-8"))
    index = json.loads(first["release_index_path"].read_text(encoding="utf-8"))
    assert manifest["file_count"] == 701
    assert manifest["schema"] == "ijoc_v21e3r1_results_release_file_manifest_v2"
    assert manifest["implementation_independence"] is False
    assert manifest["scientific_independence"] is False
    assert manifest["external_third_party_audit"] is False
    assert manifest["fixed_author_generated_cases_descriptive_only"] is True
    assert manifest["population_inference_authorized"] is False
    assert manifest["sign_flip_assumptions_verified"] is False
    assert manifest["trimmed_mean_distinct_from_mean"] is False
    assert index["status"] == "PASS_VERIFIED_DEVELOPMENT_RESULTS_RELEASE"
    assert index["schema"] == "ijoc_v21e3r1_results_release_index_v2"
    assert index["submission_status"] == "IJOC_HOLD"
    assert index["full_algorithm_decision_replay"] == "NOT_IMPLEMENTED"
    assert index["fixed_author_generated_cases_descriptive_only"] is True
    assert index["population_inference_authorized"] is False
    assert index["sign_flip_assumptions_verified"] is False
    assert index["trimmed_mean_distinct_from_mean"] is False
    assert index["code_archive_binding"]["included_in_data_archive"] is False
    assert index["code_archive_binding"]["artifact_relationship"] == (
        "HISTORICAL_MATRIX_PRODUCER_CODE_ARCHIVE_V4"
    )
    assert index["code_archive_binding"]["live_verifier_relationship"] == (
        "NOT_THE_LIVE_POST_HOC_VERIFIER_CODE_IDENTITY"
    )
    assert index["code_archive_binding"]["sha256"] == _sha256(
        inputs["code_archive_path"]
    )
    same = index["same_implementation_post_process"]
    assert same["receipt_sha256"] == _sha256(
        inputs["same_implementation_receipt_path"]
    )
    assert same["implementation_independence"] is False
    assert same["scientific_independence"] is False
    assert same["external_third_party_audit"] is False
    assert same["fixed_author_generated_cases_descriptive_only"] is True
    assert same["population_inference_authorized"] is False
    assert same["sign_flip_assumptions_verified"] is False
    assert same["trimmed_mean_distinct_from_mean"] is False
    assert same["verifier_relationship"] == (
        "SAME_PROJECT_VERIFIER_POST_HOC_SUCCESSOR_NOT_HISTORICAL_PRODUCER"
    )
    assert "independent_audit_sha256" not in index
    roles = {entry["role"] for entry in manifest["files"]}
    assert "same_implementation_post_process_receipt" in roles
    assert "independent_matrix_audit" not in roles

    with zipfile.ZipFile(first["archive_path"]) as archive:
        names = archive.namelist()
        assert len(names) == 701
        target_names = [
            name
            for name in names
            if "/target_size_execution/" in name
        ]
        assert len(target_names) == 38
        assert all(not PurePosixPath(name).is_absolute() for name in names)
        assert all(".." not in PurePosixPath(name).parts for name in names)
        assert all("\\" not in name for name in names)
        assert all(inputs["code_archive_path"].name not in name for name in names)
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.compress_type == zipfile.ZIP_STORED
            assert info.create_system == 3
            assert info.external_attr >> 16 == 0o100644
    assert all(
        not PurePosixPath(entry["archive_path"]).is_absolute()
        and ".." not in PurePosixPath(entry["archive_path"]).parts
        and "\\" not in entry["archive_path"]
        for entry in manifest["files"]
    )
    assert "D:\\" not in first["file_manifest_path"].read_text(encoding="utf-8")
    assert "D:\\" not in first["release_index_path"].read_text(encoding="utf-8")


def test_results_release_rejects_tampered_row_and_absolute_receipt_path(
    tmp_path: Path,
) -> None:
    tampered_inputs = _make_synthetic_inputs(tmp_path / "tampered-inputs")
    trace = next(
        (tampered_inputs["matrix_directory"] / "rows").glob("*/trace.sqlite3")
    )
    trace.write_bytes(trace.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="hash|binding"):
        builder.build_results_release(
            **tampered_inputs, **_outputs(tmp_path / "tampered-output")
        )

    absolute_inputs = _make_synthetic_inputs(tmp_path / "absolute-inputs")
    row_path = next(
        (absolute_inputs["matrix_directory"] / "rows").glob("*/row.json")
    )
    row = json.loads(row_path.read_text(encoding="utf-8"))
    row["trace_database_path"] = (tmp_path / "outside.sqlite3").as_posix()
    _write_json(row_path, row)
    with pytest.raises(RuntimeError, match="portable|artifact path"):
        builder.build_results_release(
            **absolute_inputs, **_outputs(tmp_path / "absolute-output")
        )

    invalidation_inputs = _make_synthetic_inputs(tmp_path / "invalidation-inputs")
    invalidation_path = invalidation_inputs["v3_invalidation_path"]
    invalidation = json.loads(invalidation_path.read_text(encoding="utf-8"))
    invalidation["status"] = "NOT_INVALIDATED"
    _write_json(invalidation_path, invalidation)
    with pytest.raises(RuntimeError, match="invalidation|scientific reuse"):
        builder.build_results_release(
            **invalidation_inputs, **_outputs(tmp_path / "invalidation-output")
        )

    semantic_inputs = _make_synthetic_inputs(tmp_path / "semantic-inputs")
    semantic_row_dir = next(
        (semantic_inputs["matrix_directory"] / "rows").iterdir()
    )
    objective_path = semantic_row_dir / "objective_archive_replay.receipt.json"
    objective = json.loads(objective_path.read_text(encoding="utf-8"))
    objective["status"] = "OBJECTIVE_AND_ARCHIVE_REPLAY_FAIL"
    _write_json(objective_path, objective)
    semantic_row_path = semantic_row_dir / "row.json"
    semantic_row = json.loads(semantic_row_path.read_text(encoding="utf-8"))
    semantic_row["objective_archive_replay_receipt_sha256"] = _sha256(
        objective_path
    )
    _write_json(semantic_row_path, semantic_row)
    aggregate_path = semantic_inputs["matrix_directory"] / "matrix.aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate_row = next(
        item
        for item in aggregate["rows"]
        if all(
            item[field] == semantic_row[field]
            for field in ("case_id", "family", "size", "seed", "arm_id")
        )
    )
    aggregate_row["objective_archive_replay_receipt_sha256"] = _sha256(
        objective_path
    )
    aggregate_row["row_receipt_sha256"] = _sha256(semantic_row_path)
    _write_json(aggregate_path, aggregate)
    post_run_path = semantic_inputs["matrix_directory"] / "post_run_audit.json"
    post_run = json.loads(post_run_path.read_text(encoding="utf-8"))
    post_run["matrix_aggregate_sha256"] = _sha256(aggregate_path)
    _write_json(post_run_path, post_run)
    receipt_path = semantic_inputs["same_implementation_receipt_path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["matrix_aggregate_sha256"] = _sha256(aggregate_path)
    receipt["runner_post_run_audit_sha256"] = _sha256(post_run_path)
    _write_json(receipt_path, receipt)
    with pytest.raises(RuntimeError, match="status|replay|semantic"):
        builder.build_results_release(
            **semantic_inputs, **_outputs(tmp_path / "semantic-output")
        )


def test_results_release_detects_stream_time_toctou_and_leaves_no_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _make_synthetic_inputs(tmp_path / "toctou-inputs")
    outputs = _outputs(tmp_path / "toctou-output")
    original = builder._stream_file_into_zip
    attacked = False

    def attack(archive, entry, *, chunk_size):
        nonlocal attacked
        if not attacked and entry.role == "matrix_row_trace":
            entry.source_path.write_bytes(entry.source_path.read_bytes() + b"race")
            attacked = True
        return original(archive, entry, chunk_size=chunk_size)

    monkeypatch.setattr(builder, "_stream_file_into_zip", attack)
    with pytest.raises(RuntimeError, match="TOCTOU|changed"):
        builder.build_results_release(**inputs, **outputs)
    assert attacked
    assert all(not path.exists() for path in outputs.values())


def test_results_release_detects_external_code_binding_toctou(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _make_synthetic_inputs(tmp_path / "code-toctou-inputs")
    outputs = _outputs(tmp_path / "code-toctou-output")
    original = builder._stream_file_into_zip
    attacked = False

    def attack(archive, entry, *, chunk_size):
        nonlocal attacked
        if not attacked:
            code_archive = inputs["code_archive_path"]
            code_archive.write_bytes(code_archive.read_bytes() + b"code-race")
            attacked = True
        return original(archive, entry, chunk_size=chunk_size)

    monkeypatch.setattr(builder, "_stream_file_into_zip", attack)
    with pytest.raises(RuntimeError, match="TOCTOU|changed|code archive"):
        builder.build_results_release(**inputs, **outputs)
    assert attacked
    assert all(not path.exists() for path in outputs.values())

    manifest_inputs = _make_synthetic_inputs(tmp_path / "manifest-binding-inputs")
    clean_room_path = manifest_inputs["code_clean_room_receipt_path"]
    clean_room = json.loads(clean_room_path.read_text(encoding="utf-8"))
    clean_room["pinned_inputs"]["manifest_sha256"] = "0" * 64
    _write_json(clean_room_path, clean_room)
    with pytest.raises(RuntimeError, match="clean-room|manifest"):
        builder.build_results_release(
            **manifest_inputs, **_outputs(tmp_path / "manifest-binding-output")
        )


def test_results_release_rejects_validation_to_capture_live_plan_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _make_synthetic_inputs(tmp_path / "plan-swap-inputs")
    outputs = _outputs(tmp_path / "plan-swap-output")
    live_plan = inputs["matrix_directory"] / "matrix.plan.json"
    original = builder._capture_entry
    attacked = False

    def attack(path, *, archive_path, role):
        nonlocal attacked
        if not attacked and role == "matrix_plan":
            plan = json.loads(live_plan.read_text(encoding="utf-8"))
            plan["status"] = "ATTACKED_AFTER_VALIDATION"
            _write_json(live_plan, plan)
            attacked = True
        return original(path, archive_path=archive_path, role=role)

    monkeypatch.setattr(builder, "_capture_entry", attack)
    with pytest.raises(RuntimeError, match="TOCTOU|changed"):
        builder.build_results_release(**inputs, **outputs)
    assert attacked
    assert all(not path.exists() for path in outputs.values())


def test_results_release_rejects_coherently_rehashed_terminal_failure_semantics(
    tmp_path: Path,
) -> None:
    inputs = _make_synthetic_inputs(tmp_path / "terminal-semantic-inputs")
    row_directory = next((inputs["matrix_directory"] / "rows").iterdir())
    terminal_path = row_directory / "terminal.receipt.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["failure_code"] = "FORGED_FAILURE_WITH_SUCCESS_STATUS"
    _write_json(terminal_path, terminal, newline=False)
    _coherently_rehash_matrix_row_packet(inputs, row_directory)

    with pytest.raises(RuntimeError, match="terminal|failure|semantic"):
        builder.build_results_release(
            **inputs, **_outputs(tmp_path / "terminal-semantic-output")
        )


@pytest.mark.parametrize(
    "attack",
    (
        "pre_identity",
        "objective_count",
        "objective_unique_replays",
        "metric_grid",
        "metric_box",
    ),
)
def test_results_release_rejects_coherently_rehashed_cross_receipt_semantics(
    tmp_path: Path, attack: str
) -> None:
    inputs = _make_synthetic_inputs(tmp_path / f"{attack}-inputs")
    row_directory = next((inputs["matrix_directory"] / "rows").iterdir())
    if attack == "pre_identity":
        path = row_directory / "row.preverification.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["case_id"] = "forged-case"
        _write_json(path, payload)
    elif attack in {"objective_count", "objective_unique_replays"}:
        path = row_directory / "objective_archive_replay.receipt.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if attack == "objective_count":
            payload["evaluation_records"] = 1999
        else:
            payload["unique_solution_replays"] = 1999
        _write_json(path, payload)
    else:
        path = row_directory / "metric_replay.receipt.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if attack == "metric_grid":
            payload["checkpoints"][0]["evaluation"] = 199
            payload["normalized_left_continuous_hv_auc"] = _left_continuous_auc(
                payload["checkpoints"], budget=2000
            )
        else:
            payload["analytic_box"]["lower"][0] += 1.0
        _write_json(path, payload)
    _coherently_rehash_matrix_row_packet(inputs, row_directory)

    with pytest.raises(RuntimeError, match="preverification|objective|metric|semantic"):
        builder.build_results_release(
            **inputs, **_outputs(tmp_path / f"{attack}-output")
        )

