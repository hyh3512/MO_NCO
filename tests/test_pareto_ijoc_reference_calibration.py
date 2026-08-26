from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from mo_nco.instance import MultiObjectiveTSPInstance, instance_sha256
from mo_nco.pareto_ijoc_problem import (
    MultiObjectiveKnapsackInstance,
    problem_sha256,
)
from mo_nco.pareto_ijoc_reference import (
    CLAIM_BOUNDARY,
    FormalCase,
    analytic_metric_box,
    build_reference_suite,
    file_sha256,
    search_reference_case,
    strict_json,
    verify_reference_suite,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_ROOT = REPO_ROOT / "ijoc_submission_v20"


def _formal_case(
    case_id: str,
    family: str,
    problem: MultiObjectiveTSPInstance | MultiObjectiveKnapsackInstance,
) -> FormalCase:
    digest = (
        instance_sha256(problem)
        if isinstance(problem, MultiObjectiveTSPInstance)
        else problem_sha256(problem)
    )
    return FormalCase(
        case_id=case_id,
        family=family,
        size=(
            problem.num_cities
            if isinstance(problem, MultiObjectiveTSPInstance)
            else problem.solution_size
        ),
        problem_sha256=digest,
        artifact_paths=(),
        artifact_bindings=(),
        problem=problem,
    )


def test_analytic_boxes_cover_feasible_toy_spaces() -> None:
    tsp = MultiObjectiveTSPInstance.from_distance_matrices(
        (
            (
                (0, 1, 3, 2),
                (1, 0, 2, 4),
                (3, 2, 0, 1),
                (2, 4, 1, 0),
            ),
            (
                (0, 4, 1, 2),
                (4, 0, 3, 1),
                (1, 3, 0, 2),
                (2, 1, 2, 0),
            ),
        ),
        name="toy-tsp",
    )
    mokp = MultiObjectiveKnapsackInstance(
        item_weights=(1, 2, 3, 2),
        profits_by_objective=((5, 1, 7, 2), (1, 8, 2, 4)),
        capacity=4,
        name="toy-mokp",
    )

    tsp_solutions = [
        (0, *permutation)
        for permutation in itertools.permutations(range(1, tsp.num_cities))
    ]
    mokp_solutions = [
        bits
        for bits in itertools.product((0, 1), repeat=mokp.solution_size)
        if mokp.total_weight(bits) <= mokp.capacity
    ]
    for problem, solutions in ((tsp, tsp_solutions), (mokp, mokp_solutions)):
        ideal, nadir, hv_reference, _ = analytic_metric_box(problem)
        for solution in solutions:
            objective = problem.evaluate(solution)
            assert all(
                lower <= value <= upper
                for value, lower, upper in zip(objective, ideal, nadir)
            )
            assert all(
                reference > value
                for reference, value in zip(hv_reference, objective)
            )


@pytest.mark.parametrize("family", ["MOTSP", "MOKP"])
def test_reference_search_is_budget_exact_and_deterministic(family: str) -> None:
    if family == "MOTSP":
        problem = MultiObjectiveTSPInstance.random_biobjective(9, seed=91)
    else:
        problem = MultiObjectiveKnapsackInstance(
            item_weights=(2, 5, 3, 4, 1, 6, 2, 5, 4),
            profits_by_objective=(
                (9, 3, 8, 4, 2, 7, 5, 6, 1),
                (1, 8, 2, 7, 6, 3, 9, 4, 5),
            ),
            capacity=13,
            name="toy-mokp-search",
        )
    case = _formal_case(f"toy-{family.lower()}", family, problem)
    left = search_reference_case(
        case,
        seeds=(91000, 91001),
        evaluation_budget=101,
        weight_grid_size=7,
        restart_period=11,
    )
    right = search_reference_case(
        case,
        seeds=(91000, 91001),
        evaluation_budget=101,
        weight_grid_size=7,
        restart_period=11,
    )
    assert left.evaluations_used == 101
    assert left.seed_evaluation_counts == (51, 50)
    assert left.entries == right.entries
    assert left.ideal == right.ideal
    assert left.hv_reference == right.hv_reference


def test_real_two_family_gate_is_hash_bound_and_idempotent(tmp_path: Path) -> None:
    output = tmp_path / "reference-gate"
    keyword = {
        "formal_case_manifest_path": (
            SUBMISSION_ROOT / "formal_study" / "case_manifest.json"
        ),
        "instance_packet_manifest_path": (
            SUBMISSION_ROOT
            / "formal_study"
            / "instance_packet_manifest.json"
        ),
        "tail_calibration_receipt_path": (
            SUBMISSION_ROOT
            / "calibration"
            / "frozen"
            / "calibration_suite_receipt.json"
        ),
        "tail_policy_path": (
            SUBMISSION_ROOT
            / "calibration"
            / "frozen"
            / "tail_policy_freeze.json"
        ),
        "replay_verifier_path": (
            SUBMISSION_ROOT / "scripts" / "ijoc_replay_verifier.py"
        ),
        "output_directory": output,
        "reference_seeds": (91000, 91001),
        "formal_seeds": (8100, 8101),
        "evaluation_budgets": (10,),
        "weight_grid_size": 5,
        "restart_period": 5,
        "case_limit_per_family": 1,
        "time_limit_seconds": 30.0,
    }
    first = build_reference_suite(**keyword)
    second = build_reference_suite(**keyword)
    assert first.status == "COMPLETE"
    assert first.case_count == 2
    assert first.family_counts == {"MOKP": 1, "MOTSP": 1}
    assert not first.reused_existing
    assert second.reused_existing

    evidence = verify_reference_suite(output)
    assert evidence["status"] == "PASS"
    assert evidence["case_count"] == 2
    assert evidence["claim_boundary"] == CLAIM_BOUNDARY
    receipt = strict_json(output / "reference_calibration_completion_receipt.json")
    manifest = strict_json(output / "reference_calibration_completion_evidence.json")
    assert receipt["artifact_manifest"]["sha256"] == file_sha256(
        output / "reference_calibration_completion_evidence.json"
    )
    assert len(manifest["reference_runs"]) == 4


def test_verifier_rejects_tampered_case_artifact(tmp_path: Path) -> None:
    output = tmp_path / "reference-gate"
    build_reference_suite(
        formal_case_manifest_path=(
            SUBMISSION_ROOT / "formal_study" / "case_manifest.json"
        ),
        instance_packet_manifest_path=(
            SUBMISSION_ROOT
            / "formal_study"
            / "instance_packet_manifest.json"
        ),
        tail_calibration_receipt_path=(
            SUBMISSION_ROOT
            / "calibration"
            / "frozen"
            / "calibration_suite_receipt.json"
        ),
        tail_policy_path=(
            SUBMISSION_ROOT
            / "calibration"
            / "frozen"
            / "tail_policy_freeze.json"
        ),
        replay_verifier_path=(
            SUBMISSION_ROOT / "scripts" / "ijoc_replay_verifier.py"
        ),
        output_directory=output,
        reference_seeds=(91000,),
        formal_seeds=(8100,),
        evaluation_budgets=(5,),
        weight_grid_size=3,
        restart_period=2,
        case_limit_per_family=1,
    )
    manifest = strict_json(
        output / "reference_calibration_completion_evidence.json"
    )
    reference = output.parent / manifest["case_outputs"][0]["path"]
    reference.write_bytes(reference.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_reference_suite(output)


def test_seed_overlap_is_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seeds overlap"):
        build_reference_suite(
            formal_case_manifest_path=(
                SUBMISSION_ROOT / "formal_study" / "case_manifest.json"
            ),
            instance_packet_manifest_path=(
                SUBMISSION_ROOT
                / "formal_study"
                / "instance_packet_manifest.json"
            ),
            tail_calibration_receipt_path=(
                SUBMISSION_ROOT
                / "calibration"
                / "frozen"
                / "calibration_suite_receipt.json"
            ),
            tail_policy_path=(
                SUBMISSION_ROOT
                / "calibration"
                / "frozen"
                / "tail_policy_freeze.json"
            ),
            replay_verifier_path=(
                SUBMISSION_ROOT / "scripts" / "ijoc_replay_verifier.py"
            ),
            output_directory=tmp_path / "must-not-exist",
            reference_seeds=(8000,),
            formal_seeds=(8000,),
            evaluation_budgets=(5,),
            case_limit_per_family=1,
        )


def test_content_addressed_verifier_imports_only_from_frozen_pythonpath(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime" / "source"
    shutil.copytree(REPO_ROOT / "mo_nco", runtime_root / "mo_nco")
    frozen_scripts = runtime_root / "ijoc_submission_v20" / "scripts"
    frozen_scripts.mkdir(parents=True)
    shutil.copyfile(
        SUBMISSION_ROOT / "scripts" / "ijoc_algorithm_adapter.py",
        frozen_scripts / "ijoc_algorithm_adapter.py",
    )
    verifier_source = SUBMISSION_ROOT / "scripts" / "ijoc_replay_verifier.py"
    verifier_sha = file_sha256(verifier_source)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    verifier = artifacts / f"{verifier_sha}.py"
    shutil.copyfile(verifier_source, verifier)

    packet = (
        SUBMISSION_ROOT
        / "formal_study"
        / "instances"
        / "mokp-formal_test-n100-s00.packet.json"
    )
    packet_payload = strict_json(packet)
    raw_hashes = [
        str(binding["sha256"]) for binding in packet_payload["artifacts"]
    ]
    profits = strict_json(
        SUBMISSION_ROOT
        / "formal_study"
        / "instances"
        / "mokp"
        / "mokp-formal_test-n100-s00.json"
    )["profits_by_objective"]
    ideal = [-float(max(1, sum(row))) for row in profits]
    nadir = [0.0, 0.0]
    hv_reference = [
        upper + 0.05 * (upper - lower) + 1.0
        for lower, upper in zip(ideal, nadir)
    ]
    run_key = {
        "case_id": "mokp-formal_test-n100-s00",
        "algorithm": "reference-test",
        "seed": 91000,
        "budget": 1,
    }
    witness_payload = {
        "schema": "ijoc_reference_calibration_run_witness_v1",
        "run_key": run_key,
        "family": "MOKP",
        "problem_sha256": packet_payload["problem_sha256"],
        "instance_packet_sha256": file_sha256(packet),
        "raw_instance_artifact_sha256": raw_hashes,
        "reference_calibration_precommit_sha256": "1" * 64,
        "evaluations_used": 1,
        "entries": [
            {
                "solution": [0] * 100,
                "objectives": [0.0, 0.0],
            }
        ],
        "analytic_metric_box": {
            "ideal": ideal,
            "nadir": nadir,
            "hv_reference": hv_reference,
            "derivation": "Exhaustive analytic feasibility box used by test.",
        },
        "metric_contract": {
            "objective_sense": ["minimize", "minimize"],
            "dominance_tolerance": 0.0,
            "normalization": "frozen_ideal_nadir_affine",
            "archive_semantics": "calibration_all_evaluated_nondominated",
            "evaluation_code_sha256": verifier_sha,
        },
        "builder_configuration_sha256": "2" * 64,
        "claim_boundary": CLAIM_BOUNDARY,
        "formal_matrix_status": "NOT_RUN",
    }
    witness = tmp_path / "witness.json"
    witness.write_text(
        json.dumps(witness_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    receipt = tmp_path / "receipt.json"
    isolated_cwd = tmp_path / "isolated"
    isolated_cwd.mkdir()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(runtime_root)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            str(verifier),
            "--reference-witness",
            str(witness),
            "--instance",
            str(packet),
            "--output",
            str(receipt),
        ],
        cwd=isolated_cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        shell=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    replay = strict_json(receipt)
    assert replay["status"] == "PASS"
    assert replay["evaluation_code_sha256"] == verifier_sha

