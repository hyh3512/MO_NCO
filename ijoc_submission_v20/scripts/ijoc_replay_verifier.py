from __future__ import annotations

"""Independent objective replay for one frozen IJOC matrix row."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mo_nco.archive import dominates
from mo_nco.instance import MultiObjectiveTSPInstance
from mo_nco.pareto_ijoc_problem import MultiObjectiveKnapsackInstance

from ijoc_submission_v20.scripts.ijoc_algorithm_adapter import (
    canonical_bytes,
    file_sha256,
    load_problem,
    strict_json,
)


def resolve_result_artifact(
    result_path: Path,
    binding: dict[str, Any],
) -> Path:
    raw = Path(str(binding["path"]))
    if raw.is_absolute():
        raise ValueError("Algorithm-result artifact paths must be relative.")
    path = (result_path.parent / raw).resolve()
    path.relative_to(result_path.parent.resolve())
    if file_sha256(path) != str(binding["sha256"]):
        raise ValueError("Algorithm-result artifact SHA-256 mismatch.")
    return path


def nondominated_gate(entries: list[dict[str, Any]]) -> None:
    objectives = [
        tuple(float(value) for value in entry["objectives"])
        for entry in entries
    ]
    if any(
        not all(math.isfinite(value) for value in objective)
        for objective in objectives
    ):
        raise ValueError("Archive contains a non-finite objective.")
    for left in range(len(objectives)):
        for right in range(len(objectives)):
            if left != right and dominates(
                objectives[left],
                objectives[right],
                tol=0.0,
            ):
                raise ValueError("Final archive contains a dominated entry.")


def replay_entries(problem: Any, entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        solution = tuple(int(value) for value in entry["solution"])
        expected = tuple(float(value) for value in entry["objectives"])
        actual = tuple(float(value) for value in problem.evaluate(solution))
        if actual != expected:
            raise ValueError("A solution witness failed exact objective replay.")


def analytic_metric_box(problem: Any) -> tuple[list[float], list[float], list[float]]:
    if isinstance(problem, MultiObjectiveTSPInstance):
        ideal = [0.0 for _ in range(problem.num_objectives)]
        nadir = [
            max(
                1.0,
                float(problem.num_cities)
                * max(float(value) for row in matrix for value in row),
            )
            for matrix in problem.distance_matrices
        ]
    elif isinstance(problem, MultiObjectiveKnapsackInstance):
        ideal = [
            -float(max(1, sum(profits)))
            for profits in problem.profits_by_objective
        ]
        nadir = [0.0 for _ in range(problem.num_objectives)]
    else:
        raise TypeError(f"Unsupported reference problem type: {type(problem)!r}.")
    hv_reference = [
        upper + 0.05 * (upper - lower) + 1.0
        for lower, upper in zip(ideal, nadir)
    ]
    return ideal, nadir, hv_reference


def verify_reference_witness(
    *,
    witness_path: Path,
    instance_path: Path,
    output_path: Path,
) -> None:
    witness = strict_json(witness_path)
    if witness.get("schema") == "ijoc_reference_calibration_case_union_witness_v1":
        verify_reference_union_witness(
            witness=witness,
            witness_path=witness_path,
            instance_path=instance_path,
            output_path=output_path,
        )
        return
    expected_keys = {
        "schema",
        "run_key",
        "family",
        "problem_sha256",
        "instance_packet_sha256",
        "raw_instance_artifact_sha256",
        "reference_calibration_precommit_sha256",
        "evaluations_used",
        "entries",
        "analytic_metric_box",
        "metric_contract",
        "builder_configuration_sha256",
        "claim_boundary",
        "formal_matrix_status",
    }
    if set(witness) != expected_keys:
        raise ValueError("Reference witness has an unexpected shape.")
    if witness.get("schema") != "ijoc_reference_calibration_run_witness_v1":
        raise ValueError("Reference witness schema mismatch.")
    run_key = witness.get("run_key")
    if not isinstance(run_key, dict) or set(run_key) != {
        "case_id",
        "algorithm",
        "seed",
        "budget",
    }:
        raise ValueError("Reference witness run_key is malformed.")
    case_id = str(run_key["case_id"])
    budget = run_key["budget"]
    seed = run_key["seed"]
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or isinstance(budget, bool)
        or not isinstance(budget, int)
        or budget <= 0
    ):
        raise ValueError("Reference witness seed or budget is invalid.")
    if (
        witness.get("evaluations_used") != budget
        or witness.get("formal_matrix_status") != "NOT_RUN"
        or witness.get("claim_boundary")
        != "supplied_reference_relative_only_not_true_front"
    ):
        raise ValueError("Reference witness budget or evidence boundary mismatch.")
    packet_sha = file_sha256(instance_path)
    if witness.get("instance_packet_sha256") != packet_sha:
        raise ValueError("Reference witness binds the wrong instance packet.")
    family, problem, packet = load_problem(instance_path, case_id)
    if (
        witness.get("family") != family
        or witness.get("problem_sha256") != packet.get("problem_sha256")
    ):
        raise ValueError("Reference witness problem identity mismatch.")
    raw_hashes = [
        str(binding["sha256"])
        for binding in packet["artifacts"]
    ]
    if witness.get("raw_instance_artifact_sha256") != raw_hashes:
        raise ValueError("Reference witness raw instance hashes mismatch.")
    metric_contract = witness.get("metric_contract")
    verifier_sha = file_sha256(Path(__file__).resolve())
    if (
        not isinstance(metric_contract, dict)
        or metric_contract.get("evaluation_code_sha256") != verifier_sha
        or metric_contract.get("objective_sense") != ["minimize", "minimize"]
        or metric_contract.get("dominance_tolerance") != 0.0
        or metric_contract.get("normalization")
        != "frozen_ideal_nadir_affine"
        or metric_contract.get("archive_semantics")
        != "calibration_all_evaluated_nondominated"
    ):
        raise ValueError("Reference witness metric contract mismatch.")
    entries = witness.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Reference witness archive is empty.")
    replay_entries(problem, entries)
    nondominated_gate(entries)
    ideal, nadir, hv_reference = analytic_metric_box(problem)
    box = witness.get("analytic_metric_box")
    if (
        not isinstance(box, dict)
        or box.get("ideal") != ideal
        or box.get("nadir") != nadir
        or box.get("hv_reference") != hv_reference
        or not isinstance(box.get("derivation"), str)
        or not box["derivation"]
    ):
        raise ValueError("Reference witness analytic metric box mismatch.")
    for entry in entries:
        objective = [float(value) for value in entry["objectives"]]
        if any(
            value < lower or value > upper
            for value, lower, upper in zip(objective, ideal, nadir)
        ):
            raise ValueError("Reference objective lies outside the analytic box.")
        if any(
            reference <= value
            for reference, value in zip(hv_reference, objective)
        ):
            raise ValueError("Reference objective is not strictly inside the HV box.")
    receipt = {
        "schema": "ijoc_reference_replay_receipt_v1",
        "status": "PASS",
        "run_key": run_key,
        "witness_sha256": file_sha256(witness_path),
        "instance_packet_sha256": packet_sha,
        "raw_instance_artifact_sha256": raw_hashes,
        "evaluation_code_sha256": verifier_sha,
        "evaluations_used": budget,
        "replayed_entry_count": len(entries),
        "zero_tolerance_nondominance": "PASS",
        "analytic_metric_box": "PASS",
        "formal_matrix_status": "NOT_RUN",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_bytes(receipt))


def verify_reference_union_witness(
    *,
    witness: dict[str, Any],
    witness_path: Path,
    instance_path: Path,
    output_path: Path,
) -> None:
    expected_keys = {
        "schema",
        "case_id",
        "family",
        "problem_sha256",
        "instance_packet_sha256",
        "raw_instance_artifact_sha256",
        "reference_calibration_precommit_sha256",
        "constituent_run_keys",
        "total_evaluations",
        "entries",
        "analytic_metric_box",
        "metric_contract",
        "builder_configuration_sha256",
        "claim_boundary",
        "formal_matrix_status",
    }
    if set(witness) != expected_keys:
        raise ValueError("Reference case-union witness has an unexpected shape.")
    case_id = witness.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("Reference case-union witness has no case_id.")
    run_keys = witness.get("constituent_run_keys")
    if not isinstance(run_keys, list) or not run_keys:
        raise ValueError("Reference case-union witness has no constituent runs.")
    normalized_run_keys = []
    for run_key in run_keys:
        if not isinstance(run_key, dict) or set(run_key) != {
            "case_id",
            "algorithm",
            "seed",
            "budget",
        }:
            raise ValueError("Reference case-union constituent run_key is malformed.")
        if run_key.get("case_id") != case_id:
            raise ValueError("Reference case-union contains another case.")
        normalized_run_keys.append(
            (
                str(run_key["case_id"]),
                str(run_key["algorithm"]),
                int(run_key["seed"]),
                int(run_key["budget"]),
            )
        )
    if len(normalized_run_keys) != len(set(normalized_run_keys)):
        raise ValueError("Reference case-union repeats a constituent run.")
    total_evaluations = witness.get("total_evaluations")
    expected_evaluations = sum(key[3] for key in normalized_run_keys)
    if (
        isinstance(total_evaluations, bool)
        or not isinstance(total_evaluations, int)
        or total_evaluations != expected_evaluations
        or witness.get("formal_matrix_status") != "NOT_RUN"
        or witness.get("claim_boundary")
        != "supplied_reference_relative_only_not_true_front"
    ):
        raise ValueError("Reference case-union budget or evidence boundary mismatch.")
    packet_sha = file_sha256(instance_path)
    if witness.get("instance_packet_sha256") != packet_sha:
        raise ValueError("Reference case-union binds the wrong instance packet.")
    family, problem, packet = load_problem(instance_path, case_id)
    if (
        witness.get("family") != family
        or witness.get("problem_sha256") != packet.get("problem_sha256")
    ):
        raise ValueError("Reference case-union problem identity mismatch.")
    raw_hashes = [str(binding["sha256"]) for binding in packet["artifacts"]]
    if witness.get("raw_instance_artifact_sha256") != raw_hashes:
        raise ValueError("Reference case-union raw instance hashes mismatch.")
    metric_contract = witness.get("metric_contract")
    verifier_sha = file_sha256(Path(__file__).resolve())
    if (
        not isinstance(metric_contract, dict)
        or metric_contract.get("evaluation_code_sha256") != verifier_sha
        or metric_contract.get("objective_sense") != ["minimize", "minimize"]
        or metric_contract.get("dominance_tolerance") != 0.0
        or metric_contract.get("normalization")
        != "frozen_ideal_nadir_affine"
        or metric_contract.get("archive_semantics")
        != "calibration_all_evaluated_nondominated"
    ):
        raise ValueError("Reference case-union metric contract mismatch.")
    entries = witness.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Reference case-union archive is empty.")
    replay_entries(problem, entries)
    nondominated_gate(entries)
    ideal, nadir, hv_reference = analytic_metric_box(problem)
    box = witness.get("analytic_metric_box")
    if (
        not isinstance(box, dict)
        or box.get("ideal") != ideal
        or box.get("nadir") != nadir
        or box.get("hv_reference") != hv_reference
        or not isinstance(box.get("derivation"), str)
        or not box["derivation"]
    ):
        raise ValueError("Reference case-union analytic metric box mismatch.")
    for entry in entries:
        objective = [float(value) for value in entry["objectives"]]
        if any(
            value < lower or value > upper
            for value, lower, upper in zip(objective, ideal, nadir)
        ):
            raise ValueError("Case-union objective lies outside the analytic box.")
        if any(
            reference <= value
            for reference, value in zip(hv_reference, objective)
        ):
            raise ValueError("Case-union objective is not strictly inside the HV box.")
    receipt = {
        "schema": "ijoc_reference_union_replay_receipt_v1",
        "status": "PASS",
        "case_id": case_id,
        "witness_sha256": file_sha256(witness_path),
        "instance_packet_sha256": packet_sha,
        "raw_instance_artifact_sha256": raw_hashes,
        "evaluation_code_sha256": verifier_sha,
        "total_evaluations": total_evaluations,
        "constituent_run_count": len(run_keys),
        "replayed_entry_count": len(entries),
        "zero_tolerance_nondominance": "PASS",
        "analytic_metric_box": "PASS",
        "formal_matrix_status": "NOT_RUN",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_bytes(receipt))


def verify_formal_result(
    *,
    input_path: Path,
    result_path: Path,
    output_path: Path,
) -> None:
    cold_input = strict_json(input_path)
    result = strict_json(result_path)
    if result.get("schema") != "ijoc_algorithm_result_v1":
        raise ValueError("Algorithm result schema mismatch.")
    run_key = cold_input["run_key"]
    if result.get("run_key") != run_key:
        raise ValueError("Algorithm result run_key mismatch.")
    instance_path = Path(cold_input["instance_artifact"]["path"]).resolve()
    instance_sha = file_sha256(instance_path)
    if instance_sha != cold_input["instance_artifact"]["sha256"]:
        raise ValueError("Instance artifact SHA-256 mismatch.")
    _, problem, _ = load_problem(instance_path, str(run_key["case_id"]))
    archive_path = resolve_result_artifact(
        result_path,
        result["archive_artifact"],
    )
    checkpoint_path = resolve_result_artifact(
        result_path,
        result["checkpoint_artifact"],
    )
    archive = strict_json(archive_path)
    checkpoints = strict_json(checkpoint_path)
    if archive.get("schema") != "ijoc_all_evaluated_archive_v1":
        raise ValueError("Archive artifact schema mismatch.")
    if checkpoints.get("schema") != "ijoc_checkpoint_solution_witnesses_v1":
        raise ValueError("Checkpoint artifact schema mismatch.")
    if archive.get("run_key") != run_key or checkpoints.get("run_key") != run_key:
        raise ValueError("Witness artifact run_key mismatch.")
    archive_entries = list(archive["entries"])
    if not archive_entries:
        raise ValueError("Final archive is empty.")
    replay_entries(problem, archive_entries)
    nondominated_gate(archive_entries)

    budget = int(run_key["budget"])
    period = int(cold_input["anytime_checkpoint_period"])
    expected = list(range(period, budget + 1, period))
    observed = [
        int(checkpoint["evaluation"])
        for checkpoint in checkpoints["checkpoints"]
    ]
    if observed != expected or result["observed_checkpoints"] != expected:
        raise ValueError("Checkpoint witness grid is incomplete.")
    for checkpoint in checkpoints["checkpoints"]:
        entries = list(checkpoint["entries"])
        if not entries:
            raise ValueError("Checkpoint archive witness is empty.")
        replay_entries(problem, entries)
        nondominated_gate(entries)
    final_checkpoint = list(checkpoints["checkpoints"][-1]["entries"])
    final_set = {
        (
            tuple(int(value) for value in entry["solution"]),
            tuple(float(value) for value in entry["objectives"]),
        )
        for entry in archive_entries
    }
    checkpoint_set = {
        (
            tuple(int(value) for value in entry["solution"]),
            tuple(float(value) for value in entry["objectives"]),
        )
        for entry in final_checkpoint
    }
    if final_set != checkpoint_set:
        raise ValueError("Final checkpoint and final archive disagree.")
    receipt = {
        "schema": "ijoc_replay_receipt_v1",
        "run_key": run_key,
        "status": "PASS",
        "instance_sha256": instance_sha,
        "algorithm_result_sha256": file_sha256(result_path),
        "archive_sha256": file_sha256(archive_path),
        "checkpoint_artifact_sha256": file_sha256(checkpoint_path),
        "evaluations_used": int(result["evaluations_used"]),
        "observed_checkpoints": observed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_bytes(receipt))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--reference-witness", type=Path)
    parser.add_argument("--instance", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = args.output.resolve()
    if args.reference_witness is not None:
        if args.instance is None or args.input is not None or args.result is not None:
            parser.error(
                "--reference-witness requires --instance and forbids --input/--result"
            )
        verify_reference_witness(
            witness_path=args.reference_witness.resolve(),
            instance_path=args.instance.resolve(),
            output_path=output_path,
        )
        return
    if args.input is None or args.result is None or args.instance is not None:
        parser.error(
            "formal-row mode requires --input and --result and forbids --instance"
        )
    verify_formal_result(
        input_path=args.input.resolve(),
        result_path=args.result.resolve(),
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
