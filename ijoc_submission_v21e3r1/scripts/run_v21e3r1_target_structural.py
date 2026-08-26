from __future__ import annotations

"""Run the V21e3r1 n=500 three-arm small-budget structural preflight."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

try:
    from ijoc_submission_v21e3r1.scripts.run_v21e3r1_development_parity import (
        ARMS,
        FrozenContract,
        _default_paths,
        execute_row,
        load_frozen_contract,
    )
except ModuleNotFoundError:  # Direct ``python path/to/script.py`` execution.
    from run_v21e3r1_development_parity import (  # type: ignore[no-redef]
        ARMS,
        FrozenContract,
        _default_paths,
        execute_row,
        load_frozen_contract,
    )


TARGET_BUDGET = 200
TARGET_CHECKPOINT = 200
TARGET_SEED = 31051
RECEIPT_NAME = "V21E3R1_TARGET_SIZE_EXECUTION_RECEIPT_V1.json"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    root = repo_root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(
            f"Target-size evidence path escapes the repository root: {resolved}"
        ) from error


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exclusive_write(path: Path, payload: object) -> str:
    raw = _canonical_bytes(payload)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(raw).hexdigest()


def build_target_structural_plan(contract: FrozenContract) -> dict[str, object]:
    """Select the first canonical n=500 case in each family and all three arms."""

    if contract.authorization_sha256 is not None:
        raise ValueError("Target structural execution precedes matrix authorization.")
    baseline_populations = {"MOTSP": 48, "MOKP": 40}
    if TARGET_BUDGET < max(baseline_populations.values()):
        raise RuntimeError("Target structural budget cannot initialize each baseline.")
    selected = []
    for family in ("MOTSP", "MOKP"):
        cases = sorted(
            (case for case in contract.cases if case.family == family and case.size == 500),
            key=lambda case: case.case_id,
        )
        if len(cases) != 2:
            raise ValueError("Target structural plan requires two frozen n=500 cases per family.")
        selected.append(cases[0])
    rows = []
    for case in selected:
        for arm_id in ARMS:
            rows.append(
                {
                    "case_id": case.case_id,
                    "family": case.family,
                    "size": case.size,
                    "seed": TARGET_SEED,
                    "arm_id": arm_id,
                    "row_slug": f"{case.case_id}__seed-{TARGET_SEED}__arm-{arm_id.lower()}",
                }
            )
    return {
        "schema": "pareto_v21e3r1_target_size_three_arm_plan_v1",
        "status": "READY_TARGET_SIZE_SMALL_BUDGET_ENGINEERING_ONLY",
        "scientific_scope": (
            "target_size_small_budget_structure_and_objective_archive_replay_"
            "not_performance_evidence"
        ),
        "budget": TARGET_BUDGET,
        "checkpoint_period": TARGET_CHECKPOINT,
        "baseline_population_sizes": baseline_populations,
        "budget_initializes_every_baseline_population": True,
        "source_snapshot_root_sha256": contract.source_snapshot_root_sha256,
        "input_sha256": dict(contract.input_sha256),
        "rows": rows,
        "development_parity_execution": "NOT_AUTHORIZED_BY_THIS_PLAN",
        "selection_entropy_release": "PROHIBITED",
        "calibration_execution": "PROHIBITED",
        "formal_execution": "PROHIBITED",
        "formal_authorized": False,
    }


def run_target_structural(
    *,
    contract: FrozenContract,
    output_directory: str | Path,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, object]:
    """Execute six real n=500 rows and commit a fail-closed final receipt."""

    output = Path(output_directory).resolve()
    root = Path(repo_root).resolve()
    _repo_relative(output, repo_root=root)
    output.mkdir(parents=True, exist_ok=False)
    rows_root = output / "rows"
    rows_root.mkdir()
    plan = build_target_structural_plan(contract)
    plan_path = output / "target_structural.plan.json"
    plan_sha256 = _exclusive_write(plan_path, plan)
    case_by_id = {case.case_id: case for case in contract.cases}
    completed: list[dict[str, object]] = []
    for planned in plan["rows"]:
        row_directory = rows_root / str(planned["row_slug"])
        row = execute_row(
            case=case_by_id[str(planned["case_id"])],
            arm_id=str(planned["arm_id"]),
            seed=int(planned["seed"]),
            budget=TARGET_BUDGET,
            checkpoint_period=TARGET_CHECKPOINT,
            reference_directions=contract.reference_directions,
            source_snapshot_root_sha256=contract.source_snapshot_root_sha256,
            row_directory=row_directory,
            metric_manifest_sha256=contract.input_sha256["metric_manifest"],
            scientific_scope=(
                "target_size_small_budget_structure_and_objective_archive_replay_"
                "not_performance_evidence"
            ),
        )
        if row.get("status") != "PASS_ENGINEERING_ROW_OBJECTIVE_ARCHIVE_REPLAYED":
            raise RuntimeError("A target structural row did not pass replay.")
        completed.append(
            {
                "case_id": row["case_id"],
                "family": row["family"],
                "size": row["size"],
                "seed": row["seed"],
                "arm_id": row["arm_id"],
                "charged_evaluation_budget": row["charged_evaluation_budget"],
                "checkpoint_period": row["checkpoint_period"],
                "normalized_terminal_hv_diagnostic_only": row[
                    "normalized_terminal_hv"
                ],
                "trace_database_sha256": row["trace_database_sha256"],
                "detached_terminal_receipt_sha256": row[
                    "detached_terminal_receipt_sha256"
                ],
                "objective_archive_replay_receipt_sha256": row[
                    "objective_archive_replay_receipt_sha256"
                ],
                "metric_replay_receipt_sha256": row[
                    "metric_replay_receipt_sha256"
                ],
                "row_receipt_path": _repo_relative(
                    row_directory / "row.json", repo_root=root
                ),
                "row_receipt_sha256": _sha256(row_directory / "row.json"),
                "objective_and_archive_replay": "PASS",
                "metric_replay": "PASS",
            }
        )
    receipt = {
        "schema": "pareto_v21e3r1_target_size_execution_receipt_v1",
        "status": "PASS_TARGET_SIZE_THREE_ARM_EXECUTION_ENGINEERING_ONLY",
        "scientific_scope": (
            "target_size_small_budget_structure_and_objective_archive_replay_"
            "not_performance_evidence"
        ),
        "artifact_path_semantics": "repo_root_relative_posix_v1",
        "source_snapshot_root_sha256": contract.source_snapshot_root_sha256,
        "input_sha256": dict(contract.input_sha256),
        "plan_path": _repo_relative(plan_path, repo_root=root),
        "plan_sha256": plan_sha256,
        "families": ["MOTSP", "MOKP"],
        "target_size": 500,
        "arms": list(ARMS),
        "seed": TARGET_SEED,
        "charged_evaluation_budget": TARGET_BUDGET,
        "checkpoint_period": TARGET_CHECKPOINT,
        "baseline_population_sizes": {"MOTSP": 48, "MOKP": 40},
        "budget_initializes_every_baseline_population": True,
        "row_count": len(completed),
        "objective_and_archive_replay_pass_rows": sum(
            row["objective_and_archive_replay"] == "PASS" for row in completed
        ),
        "metric_replay_pass_rows": sum(
            row["metric_replay"] == "PASS" for row in completed
        ),
        "rows": completed,
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
    _exclusive_write(output / RECEIPT_NAME, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(
        description="Run the V21e3r1 n=500 three-arm engineering preflight."
    )
    parser.add_argument("--case-manifest", type=Path, default=defaults["case_manifest"])
    parser.add_argument(
        "--reference-manifest", type=Path, default=defaults["reference_manifest"]
    )
    parser.add_argument(
        "--config-manifest", type=Path, default=defaults["config_manifest"]
    )
    parser.add_argument(
        "--metric-manifest", type=Path, default=defaults["metric_manifest"]
    )
    parser.add_argument("--protocol", type=Path, default=defaults["protocol"])
    parser.add_argument("--source-snapshot-root-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    contract = load_frozen_contract(
        case_manifest_path=args.case_manifest,
        reference_manifest_path=args.reference_manifest,
        config_manifest_path=args.config_manifest,
        metric_manifest_path=args.metric_manifest,
        protocol_path=args.protocol,
        authorization_path=None,
        source_snapshot_root_sha256=args.source_snapshot_root_sha256,
        require_matrix_authorization=False,
    )
    receipt = run_target_structural(
        contract=contract,
        output_directory=args.output_directory,
        repo_root=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
