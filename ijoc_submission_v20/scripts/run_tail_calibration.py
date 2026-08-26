from __future__ import annotations

"""Run and freeze the disjoint V20 tail-policy calibration.

This runner is intentionally separate from the formal matched matrix.  It uses
the selection split to choose one member of a finite menu and the confirmation
split to decide whether that candidate may replace a tail-matched uniform
fallback.  A failed or inconclusive gate freezes uniform allocation.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
import traceback
from typing import Any, Iterable

from mo_nco.instance import MultiObjectiveTSPInstance, instance_sha256
from mo_nco.pareto_ijoc_allocation import SearchRewardWeights
from mo_nco.pareto_ijoc_generic_smc import GenericAnnealedParetoSMCOptimizer
from mo_nco.pareto_ijoc_problem import (
    MultiObjectiveKnapsackInstance,
    problem_sha256,
)
from mo_nco.pareto_smc import AnnealedParetoSMCOptimizer


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = SUBMISSION_ROOT / "protocol" / "tail_calibration_plan.json"
DEFAULT_OUTPUT = SUBMISSION_ROOT / "calibration" / "runs"
FORMAL_BUDGETS = (10_000, 50_000, 100_000)


def canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def payload_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def canonical_value_sha256(payload: object) -> str:
    """Hash a canonical JSON value without the file-terminating newline."""

    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload))
    return file_sha256(path)


def strict_json(path: Path) -> dict[str, Any]:
    def reject_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"Duplicate JSON key {key!r}: {path}")
            output[key] = value
        return output

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant {value!r}: {path}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def resolve_bound(
    parent: Path,
    binding: dict[str, Any],
    *,
    allowed_root: Path | None = None,
) -> Path:
    raw = Path(str(binding["path"]))
    if raw.is_absolute():
        raise ValueError("Bound paths must be relative.")
    resolved = (parent / raw).resolve()
    resolved.relative_to(
        parent.resolve() if allowed_root is None else allowed_root.resolve()
    )
    if file_sha256(resolved) != str(binding["sha256"]):
        raise ValueError(f"SHA-256 mismatch: {resolved}")
    return resolved


def load_problem(case: dict[str, Any], manifest_dir: Path):
    artifact_paths = []
    for artifact in case["artifacts"]:
        path = resolve_bound(manifest_dir, artifact)
        artifact_paths.append(path)
    if case["family"] == "MOTSP":
        problem = MultiObjectiveTSPInstance.from_tsplib_files(artifact_paths)
        actual_problem_sha = instance_sha256(problem)
    elif case["family"] == "MOKP":
        payload = strict_json(artifact_paths[0])
        problem = MultiObjectiveKnapsackInstance(
            item_weights=tuple(int(value) for value in payload["item_weights"]),
            profits_by_objective=tuple(
                tuple(int(value) for value in row)
                for row in payload["profits_by_objective"]
            ),
            capacity=int(payload["capacity"]),
            name=str(payload["case_id"]),
        )
        actual_problem_sha = problem_sha256(problem)
    else:
        raise ValueError(f"Unsupported family: {case['family']!r}")
    if actual_problem_sha != case["problem_sha256"]:
        raise ValueError(f"Problem SHA-256 mismatch for {case['case_id']}.")
    return problem


def candidate_parameters(
    candidate: dict[str, Any],
    *,
    budget: int,
    num_types: int,
) -> tuple[int, int]:
    tail = int(round(budget * float(candidate["tail_fraction"])))
    if tail <= 0 or tail >= budget:
        raise ValueError("Every candidate must leave positive core and tail budgets.")
    quota = int(
        math.floor(
            tail * float(candidate["quota_fraction"]) / num_types
        )
    )
    if quota * num_types > tail:
        raise ValueError("Candidate quota exceeds its tail budget.")
    return tail, quota


def normalized_auc(
    diagnostics: Iterable[Any],
    *,
    checkpoints: tuple[int, ...],
    budget: int,
    box_volume: float,
) -> tuple[float, tuple[dict[str, float | int], ...]]:
    by_iteration = {
        int(item.iteration): float(item.hypervolume_2d)
        for item in diagnostics
    }
    if set(checkpoints) - set(by_iteration):
        raise RuntimeError(
            f"Missing calibration checkpoints: {sorted(set(checkpoints) - set(by_iteration))}"
        )
    previous_evaluation = 0
    previous_hv = 0.0
    area = 0.0
    trace = []
    for evaluation in checkpoints:
        hv = by_iteration[evaluation]
        area += previous_hv * (evaluation - previous_evaluation)
        trace.append(
            {
                "evaluation": evaluation,
                "hypervolume": hv,
                "normalized_hypervolume": hv / box_volume,
            }
        )
        previous_evaluation = evaluation
        previous_hv = hv
    return area / (budget * box_volume), tuple(trace)


def replay_witnesses(problem: Any, witnesses: Iterable[dict[str, Any]]) -> tuple[str, int]:
    checked = 0
    for checkpoint in witnesses:
        for entry in checkpoint["entries"]:
            raw_solution = entry.get("tour", entry.get("solution"))
            solution = tuple(int(value) for value in raw_solution)
            expected = tuple(float(value) for value in entry["objectives"])
            actual = tuple(float(value) for value in problem.evaluate(solution))
            if actual != expected:
                raise RuntimeError(
                    "A checkpoint solution witness failed local objective replay."
                )
            checked += 1
    return "PASS", checked


def execute_task(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    task_sha = payload_sha256(task)
    try:
        manifest_dir = Path(task["case_manifest_dir"])
        case = task["case"]
        candidate = task["candidate"]
        fixed = task["fixed_core"]
        budget = int(task["budget"])
        checkpoint_period = int(task["checkpoint_period"])
        checkpoints = tuple(range(checkpoint_period, budget + 1, checkpoint_period))
        directions = tuple(
            tuple(float(value) for value in row)
            for row in fixed["reference_directions"]
        )
        tail, quota = candidate_parameters(
            candidate,
            budget=budget,
            num_types=len(directions),
        )
        reward_weights = SearchRewardWeights(
            hypervolume=float(candidate["reward_weights"]["hypervolume"]),
            new_cell=float(candidate["reward_weights"]["new_cell"]),
            scalar_improvement=float(
                candidate["reward_weights"]["scalar_improvement"]
            ),
        )
        problem = load_problem(case, manifest_dir)
        common = {
            "reference_directions": directions,
            "particles_per_reference": int(fixed["particles_per_reference"]),
            "evaluations": budget,
            "beta_schedule": tuple(float(value) for value in fixed["beta_schedule"]),
            "ess_threshold": float(fixed["ess_threshold"]),
            "chebyshev_rho": float(fixed["chebyshev_rho"]),
            "adaptive_search_evaluations": tail,
            "adaptive_allocation_policy": str(
                candidate["allocation_policy"]
            ),
            "minimum_pulls_per_type": quota,
            "exp3_exploration": candidate["exp3_exploration"],
            "reward_weights": reward_weights,
            "deployment_archive_max_size": int(
                fixed["deployment_archive_max_size"]
            ),
            "anytime_checkpoint_period": checkpoint_period,
            "seed": int(task["seed"]),
        }
        if case["family"] == "MOTSP":
            result = AnnealedParetoSMCOptimizer(
                problem,
                particles_per_reference=common["particles_per_reference"],
                evaluations=common["evaluations"],
                seed=common["seed"],
                beta_schedule=common["beta_schedule"],
                reference_directions=common["reference_directions"],
                ess_threshold=common["ess_threshold"],
                chebyshev_rho=common["chebyshev_rho"],
                adaptive_search_evaluations=common[
                    "adaptive_search_evaluations"
                ],
                adaptive_allocation_policy=str(
                    candidate["allocation_policy"]
                ),
                adaptive_minimum_pulls_per_type=common[
                    "minimum_pulls_per_type"
                ],
                exp3_exploration=common["exp3_exploration"],
                search_reward_weights=common["reward_weights"],
                archive_tolerance=float(fixed["archive_tolerance"]),
                archive_max_size=common["deployment_archive_max_size"],
                audit_trace_level="summary",
                anytime_checkpoint_period=checkpoint_period,
            ).run()
            lower = tuple(float(value) for value in result.metadata["objective_lower_bounds"])
            upper = tuple(float(value) for value in result.metadata["objective_upper_bounds"])
            pre_tail_sha = str(result.metadata["certificate_snapshot_hash"])
        else:
            result = GenericAnnealedParetoSMCOptimizer(
                problem,
                **common,
            ).run()
            lower = tuple(float(value) for value in problem.objective_lower_bounds)
            upper = tuple(float(value) for value in problem.objective_upper_bounds)
            pre_tail_sha = str(result.metadata["pre_tail_state_sha256"])
        if int(result.metadata.get("evaluations_used", budget)) != budget:
            raise RuntimeError("Optimizer did not consume the exact budget.")
        if tuple(result.metadata["observed_anytime_checkpoints"]) != checkpoints:
            raise RuntimeError("Observed checkpoint grid does not match the plan.")
        box_volume = math.prod(
            right - left for left, right in zip(lower, upper)
        )
        if not math.isfinite(box_volume) or box_volume <= 0.0:
            raise RuntimeError("Invalid objective normalization box.")
        auc, anytime = normalized_auc(
            result.diagnostics,
            checkpoints=checkpoints,
            budget=budget,
            box_volume=box_volume,
        )
        replay_gate, replay_count = replay_witnesses(
            problem,
            result.metadata["checkpoint_solution_witnesses"],
        )
        final_entries = tuple(
            {
                "solution": entry.tour,
                "objectives": entry.objectives,
            }
            for entry in result.archive.entries
        )
        final_gate, final_count = replay_witnesses(
            problem,
            ({"evaluation": budget, "entries": final_entries},),
        )
        final_hv = float(
            result.archive.hypervolume_2d(reference=upper)
        )
        return {
            "schema": "ijoc_tail_calibration_row_v1",
            "task_sha256": task_sha,
            "status": "SUCCESS",
            "phase": task["phase"],
            "case_id": case["case_id"],
            "family": case["family"],
            "split": case["split"],
            "candidate_id": candidate["candidate_id"],
            "seed": int(task["seed"]),
            "budget": budget,
            "tail_evaluations": tail,
            "minimum_pulls_per_type": quota,
            "pre_tail_state_sha256": pre_tail_sha,
            "evaluations_used": budget,
            "checkpoint_period": checkpoint_period,
            "checkpoints": checkpoints,
            "normalized_left_continuous_evaluation_auc": auc,
            "normalized_final_hypervolume": final_hv / box_volume,
            "anytime_trace": anytime,
            "checkpoint_solution_replay_gate": replay_gate,
            "checkpoint_solution_replay_count": replay_count,
            "final_archive_replay_gate": final_gate,
            "final_archive_replay_count": final_count,
            "final_archive_sha256": payload_sha256(final_entries),
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as error:
        return {
            "schema": "ijoc_tail_calibration_row_v1",
            "task_sha256": task_sha,
            "status": "FAILURE",
            "phase": task["phase"],
            "case_id": task["case"]["case_id"],
            "family": task["case"]["family"],
            "split": task["case"]["split"],
            "candidate_id": task["candidate"]["candidate_id"],
            "seed": int(task["seed"]),
            "budget": int(task["budget"]),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }


def task_path(output_root: Path, task: dict[str, Any]) -> Path:
    identity = {
        key: task[key]
        for key in ("phase", "budget", "checkpoint_period", "seed")
    }
    identity["case_id"] = task["case"]["case_id"]
    identity["candidate_id"] = task["candidate"]["candidate_id"]
    digest = payload_sha256(identity)[:16]
    return (
        output_root
        / str(task["phase"])
        / str(task["case"]["family"]).lower()
        / f"{task['case']['case_id']}__{task['candidate']['candidate_id']}__s{task['seed']}__{digest}.json"
    )


def run_tasks(
    tasks: list[dict[str, Any]],
    *,
    output_root: Path,
    workers: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], Path]] = []
    for task in tasks:
        path = task_path(output_root, task)
        if path.is_file():
            row = strict_json(path)
            if row.get("task_sha256") != payload_sha256(task):
                raise RuntimeError(f"Resume row task hash mismatch: {path}")
            rows.append(row)
        else:
            pending.append((task, path))
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(execute_task, task): (task, path)
                for task, path in pending
            }
            completed = 0
            for future in as_completed(future_map):
                _, path = future_map[future]
                row = future.result()
                write_json(path, row)
                rows.append(row)
                completed += 1
                if completed % 10 == 0 or completed == len(pending):
                    print(
                        f"completed {completed}/{len(pending)} new rows "
                        f"({len(rows)}/{len(tasks)} including resume)",
                        flush=True,
                    )
    rows.sort(
        key=lambda item: (
            str(item["phase"]),
            str(item["family"]),
            str(item["case_id"]),
            str(item["candidate_id"]),
            int(item["seed"]),
        )
    )
    return rows


def make_tasks(
    *,
    phase: str,
    cases: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    seeds: list[int],
    budget: int,
    checkpoint_period: int,
    fixed_core: dict[str, Any],
    case_manifest_dir: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "schema": "ijoc_tail_calibration_task_v1",
            "phase": phase,
            "case_manifest_dir": str(case_manifest_dir.resolve()),
            "case": case,
            "candidate": candidate,
            "seed": seed,
            "budget": budget,
            "checkpoint_period": checkpoint_period,
            "fixed_core": fixed_core,
        }
        for case in cases
        for candidate in candidates
        for seed in seeds
    ]


def require_complete(rows: list[dict[str, Any]], expected: int, label: str) -> None:
    if len(rows) != expected:
        raise RuntimeError(f"{label} row count mismatch: {len(rows)} != {expected}.")
    failures = [row for row in rows if row.get("status") != "SUCCESS"]
    if failures:
        raise RuntimeError(
            f"{label} contains {len(failures)} failed rows; expansion refused."
        )
    keys = {
        (
            row["case_id"],
            row["candidate_id"],
            int(row["seed"]),
            int(row["budget"]),
        )
        for row in rows
    }
    if len(keys) != expected:
        raise RuntimeError(f"{label} contains duplicate row identities.")


def validate_pre_tail_pairing(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, int, int], set[str]] = {}
    for row in rows:
        key = (
            str(row["case_id"]),
            int(row["seed"]),
            int(row["tail_evaluations"]),
        )
        groups.setdefault(key, set()).add(str(row["pre_tail_state_sha256"]))
    bad = {key: values for key, values in groups.items() if len(values) != 1}
    if bad:
        raise RuntimeError(
            f"Pre-tail paired-state identity failed for {len(bad)} groups."
        )


def matched_deltas(
    rows: list[dict[str, Any]],
    *,
    candidate_id: str,
    uniform_id: str,
    metric: str,
) -> list[dict[str, Any]]:
    by_key = {
        (str(row["case_id"]), int(row["seed"]), str(row["candidate_id"])): row
        for row in rows
    }
    pairs = []
    case_seed_keys = sorted(
        {
            (str(row["case_id"]), int(row["seed"]))
            for row in rows
            if row["candidate_id"] == candidate_id
        }
    )
    for case_id, seed in case_seed_keys:
        treatment = by_key[(case_id, seed, candidate_id)]
        control = by_key[(case_id, seed, uniform_id)]
        pairs.append(
            {
                "case_id": case_id,
                "family": treatment["family"],
                "seed": seed,
                "delta": float(treatment[metric]) - float(control[metric]),
                "runtime_ratio": float(treatment["elapsed_seconds"])
                / max(float(control["elapsed_seconds"]), 1e-12),
            }
        )
    return pairs


def cluster_means(
    pairs: list[dict[str, Any]], field: str
) -> list[tuple[str, float]]:
    grouped: dict[str, list[float]] = {}
    for pair in pairs:
        grouped.setdefault(str(pair["case_id"]), []).append(float(pair[field]))
    return sorted(
        (case_id, statistics.fmean(values))
        for case_id, values in grouped.items()
    )


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot take a quantile of an empty sample.")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    fraction = position - left
    return ordered[left] * (1.0 - fraction) + ordered[right] * fraction


def bootstrap_mean_ci(
    values: list[float],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    samples = [
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(replicates)
    ]
    return quantile(samples, 0.025), quantile(samples, 0.975)


def sign_flip_p_value(
    values: list[float],
    *,
    replicates: int,
    seed: int,
) -> float:
    observed = abs(statistics.fmean(values))
    rng = random.Random(seed)
    exceed = 0
    for _ in range(replicates):
        statistic = abs(
            statistics.fmean(
                value if rng.getrandbits(1) else -value for value in values
            )
        )
        exceed += int(statistic >= observed - 1e-18)
    return (exceed + 1.0) / (replicates + 1.0)


def analyze_selection(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    scores = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if candidate["allocation_policy"] == "uniform":
            continue
        uniform_id = (
            "uniform_t20"
            if math.isclose(float(candidate["tail_fraction"]), 0.2)
            else "uniform_t30"
        )
        pairs = matched_deltas(
            rows,
            candidate_id=candidate_id,
            uniform_id=uniform_id,
            metric="normalized_left_continuous_evaluation_auc",
        )
        case_values = [
            value for _, value in cluster_means(pairs, "delta")
        ]
        scores.append(
            {
                "candidate_id": candidate_id,
                "uniform_id": uniform_id,
                "case_count": len(case_values),
                "case_cluster_mean_delta": statistics.fmean(case_values),
                "case_cluster_median_delta": statistics.median(case_values),
            }
        )
    scores.sort(
        key=lambda item: (
            -float(item["case_cluster_mean_delta"]),
            str(item["candidate_id"]),
        )
    )
    return {
        "schema": "ijoc_tail_calibration_selection_result_v1",
        "estimand": (
            "case_cluster_mean_paired_normalized_left_continuous_"
            "evaluation_auc_delta_vs_tail_matched_uniform"
        ),
        "scores": scores,
        "selected_candidate_id": scores[0]["candidate_id"],
        "matched_uniform_id": scores[0]["uniform_id"],
    }


def analyze_confirmation(
    rows: list[dict[str, Any]],
    *,
    selected_id: str,
    uniform_id: str,
    gate: dict[str, Any],
) -> dict[str, Any]:
    pairs = matched_deltas(
        rows,
        candidate_id=selected_id,
        uniform_id=uniform_id,
        metric="normalized_left_continuous_evaluation_auc",
    )
    case_auc = cluster_means(pairs, "delta")
    case_runtime = cluster_means(pairs, "runtime_ratio")
    auc_values = [value for _, value in case_auc]
    runtime_values = [value for _, value in case_runtime]
    ci = bootstrap_mean_ci(
        auc_values,
        replicates=int(gate["bootstrap_replicates"]),
        seed=2026073101,
    )
    runtime_ci = bootstrap_mean_ci(
        runtime_values,
        replicates=int(gate["bootstrap_replicates"]),
        seed=2026073102,
    )
    p_value = sign_flip_p_value(
        auc_values,
        replicates=int(gate["sign_flip_replicates"]),
        seed=2026073103,
    )
    tolerance = 1e-15
    wins = sum(float(pair["delta"]) > tolerance for pair in pairs)
    ties = sum(abs(float(pair["delta"])) <= tolerance for pair in pairs)
    losses = sum(float(pair["delta"]) < -tolerance for pair in pairs)
    checks = {
        "auc_delta_ci_lower_positive": (
            ci[0] > float(gate["minimum_auc_delta_ci_lower"])
        ),
        "wins_greater_than_losses": wins > losses,
        "randomization_p_value_within_limit": (
            p_value <= float(gate["maximum_randomization_p_value"])
        ),
        "runtime_ratio_ci_upper_within_limit": (
            runtime_ci[1] <= float(gate["maximum_runtime_ratio_ci_upper"])
        ),
    }
    passed = all(checks.values())
    return {
        "schema": "ijoc_tail_calibration_confirmation_result_v1",
        "selected_candidate_id": selected_id,
        "matched_uniform_id": uniform_id,
        "paired_case_seed_count": len(pairs),
        "case_cluster_count": len(case_auc),
        "auc_delta_case_cluster_mean": statistics.fmean(auc_values),
        "auc_delta_case_cluster_ci95": ci,
        "runtime_ratio_case_cluster_mean": statistics.fmean(runtime_values),
        "runtime_ratio_case_cluster_ci95": runtime_ci,
        "paired_wins_ties_losses": {
            "wins": wins,
            "ties": ties,
            "losses": losses,
        },
        "case_cluster_sign_flip_randomization_p_value": p_value,
        "gate_checks": checks,
        "confirmation_gate": "PASS" if passed else "FAIL",
        "frozen_candidate_id": selected_id if passed else uniform_id,
        "fallback_applied": not passed,
    }


def build_pipeline_gate_result(
    *,
    row_count: int,
    resumed_row_count: int,
    launcher_elapsed_seconds: float,
    maximum_elapsed_seconds: float,
) -> dict[str, Any]:
    """Build a deterministic gate receipt when all rows are resumed.

    A resume validates already materialized row evidence; its launcher duration
    is not a fresh capacity measurement. Recording the incidental validation
    duration would mutate the frozen evidence on every no-op resume.
    """

    if row_count <= 0 or resumed_row_count < 0 or resumed_row_count > row_count:
        raise ValueError("Invalid pipeline-gate row counts.")
    if launcher_elapsed_seconds < 0 or maximum_elapsed_seconds <= 0:
        raise ValueError("Invalid pipeline-gate elapsed time.")
    fresh_execution = resumed_row_count == 0
    recorded_elapsed = float(launcher_elapsed_seconds) if fresh_execution else 0.0
    return {
        "schema": "ijoc_tail_calibration_pipeline_gate_v1",
        "row_count": row_count,
        "new_row_count": row_count - resumed_row_count,
        "resumed_row_count": resumed_row_count,
        "launcher_elapsed_seconds": recorded_elapsed,
        "elapsed_interpretation": (
            "fresh_execution"
            if fresh_execution
            else "resume_validation_elapsed_not_remeasured"
        ),
        "maximum_elapsed_seconds": maximum_elapsed_seconds,
        "exact_budget_gate": "PASS",
        "checkpoint_grid_gate": "PASS",
        "solution_replay_gate": "PASS",
        "pre_tail_pairing_gate": "PASS",
        "pipeline_gate": (
            "PASS"
            if resumed_row_count > 0
            or launcher_elapsed_seconds <= maximum_elapsed_seconds
            else "FAIL"
        ),
    }


def freeze_specs(
    *,
    plan: dict[str, Any],
    plan_sha: str,
    case_manifest_sha: str,
    calibration_case_ids: list[str],
    calibration_instance_artifacts: list[dict[str, Any]],
    output_root: Path,
    gate_result: dict[str, Any],
    selection_result: dict[str, Any],
    confirmation_result: dict[str, Any],
    all_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_by_id = {
        str(item["candidate_id"]): item for item in plan["candidate_menu"]
    }
    frozen_id = str(confirmation_result["frozen_candidate_id"])
    frozen_candidate = candidate_by_id[frozen_id]
    frozen_dir = output_root.parent / "frozen"
    base_spec = {
        "schema": "annealed_pareto_smc_spec_v1",
        "target": {
            "family": "typed_augmented_tchebycheff_gibbs",
            "beta_schedule": plan["fixed_core"]["beta_schedule"],
            "chebyshev_rho": plan["fixed_core"]["chebyshev_rho"],
            "stage_frozen": True,
        },
        "reference_directions": plan["fixed_core"]["reference_directions"],
        "particle_allocation": {
            "policy": "split_cli_population_equally_across_reference_types"
        },
        "resampling": {
            "method": "multinomial",
            "scope": "within_reference_type",
            "ess_threshold_fraction": plan["fixed_core"]["ess_threshold"],
            "ess_is_not_a_coverage_certificate": True,
        },
        "mutation": {
            "proposal": "family_specific_symmetric_local_proposal_v1",
            "acceptance": "exact_log_domain_mh",
            "objective_evaluation": (
                "motsp_exact_incremental_two_opt_on_verified_integer_domain_"
                "else_full; mokp_full_integer_profit_evaluation"
            ),
        },
        "objective_box": {
            "source": "family_specific_frozen_analytic_box",
            "archive_independent": True,
        },
        "epsilon_cells": {
            "coordinate_system": "normalized_frozen_objective_box",
            "widths": [0.05, 0.05],
            "role": "reporting_and_coverage_only",
            "archive_independent": True,
        },
        "reporting": {
            "archive_role": "reporting_only",
            "archive_max_size": None,
            "cell_ledger": (
                "untruncated_first_evaluated_representative_per_cell"
            ),
        },
    }
    base_path = frozen_dir / "pareto_smc_v20_base_smc_spec.json"
    base_sha = write_json(base_path, base_spec)
    spec_bindings = []
    for budget in FORMAL_BUDGETS:
        tail, quota = candidate_parameters(
            frozen_candidate,
            budget=budget,
            num_types=len(plan["fixed_core"]["reference_directions"]),
        )
        spec = {
            "schema": "ijoc_typed_pareto_smc_spec_v2",
            "base_smc": {
                "path": base_path.name,
                "sha256": base_sha,
            },
            "adaptive_search": {
                "evaluations": tail,
                "allocation_policy": frozen_candidate["allocation_policy"],
                "minimum_pulls_per_type": (
                    quota
                    if frozen_candidate["allocation_policy"] == "exp3"
                    else 0
                ),
                "exp3_exploration": (
                    frozen_candidate["exp3_exploration"]
                    if frozen_candidate["allocation_policy"] == "exp3"
                    else None
                ),
                "reward_weights": frozen_candidate["reward_weights"],
            },
            "output": {
                "competitive_archive": (
                    "unbounded_all_evaluated_nondominated"
                ),
                "deployment_archive_max_size": plan["fixed_core"][
                    "deployment_archive_max_size"
                ],
            },
        }
        spec_path = frozen_dir / f"pareto_smc_v20_ijoc_spec_budget_{budget}.json"
        spec_sha = write_json(spec_path, spec)
        spec_bindings.append(
            {
                "budget": budget,
                "path": spec_path.relative_to(SUBMISSION_ROOT).as_posix(),
                "sha256": spec_sha,
                "adaptive_search_evaluations": tail,
            }
        )
    row_files: list[Path] = []
    for path in sorted(output_root.rglob("*.json")):
        if strict_json(path).get("schema") == "ijoc_tail_calibration_row_v1":
            row_files.append(path)
    row_ledger = [
        {
            "path": path.relative_to(SUBMISSION_ROOT).as_posix(),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in row_files
    ]
    fresh_gate_path = output_root.parent / "gate_fresh_recheck" / "gate_result.json"
    fresh_gate_binding = None
    if fresh_gate_path.is_file():
        fresh_gate_binding = {
            "path": fresh_gate_path.relative_to(SUBMISSION_ROOT).as_posix(),
            "sha256": file_sha256(fresh_gate_path),
        }
    evidence = {
        "schema": "ijoc_tail_calibration_evidence_v1",
        "status": "COMPLETE",
        "claim_scope": "algorithm_selection_only_not_competitive_evidence",
        "plan": {
            "path": DEFAULT_PLAN.relative_to(SUBMISSION_ROOT).as_posix(),
            "sha256": plan_sha,
        },
        "case_manifest": {
            "path": plan["case_manifest"]["path"],
            "sha256": case_manifest_sha,
        },
        "gate_result": gate_result,
        "fresh_gate_recheck": fresh_gate_binding,
        "selection_result": selection_result,
        "confirmation_result": confirmation_result,
        "frozen_candidate": frozen_candidate,
        "fallback_rule": plan["confirmation_gate"]["failure_fallback"],
        "formal_budget_specs": spec_bindings,
        "calibration_success_row_count": sum(
            row.get("status") == "SUCCESS" for row in all_rows
        ),
        "calibration_failure_row_count": sum(
            row.get("status") != "SUCCESS" for row in all_rows
        ),
        "calibration_instance_artifacts": calibration_instance_artifacts,
        "row_ledger_sha256": payload_sha256(row_ledger),
        "row_ledger": row_ledger,
        "formal_test_data_used_for_selection": False,
        "competitive_evidence": "NOT_RUN",
    }
    evidence_path = frozen_dir / "tail_calibration_evidence.json"
    evidence_sha = write_json(evidence_path, evidence)
    receipt = {
        "schema": "ijoc_calibration_suite_receipt_v1",
        "suite_id": str(plan["plan_id"]),
        "status": "COMPLETE",
        "evidence_scope": "tail_policy_selection_only",
        "calibration_case_ids": sorted(calibration_case_ids),
        "candidate_policy_ids": sorted(candidate_by_id),
        "seeds": sorted(
            {
                *(int(value) for value in plan["seeds"]),
                *(int(value) for value in plan["gate"]["seeds"]),
            }
        ),
        "artifact_manifest": {
            "path": evidence_path.name,
            "sha256": evidence_sha,
        },
        "instance_artifacts": calibration_instance_artifacts,
    }
    receipt_path = frozen_dir / "calibration_suite_receipt.json"
    receipt_sha = write_json(receipt_path, receipt)
    decision_rule = {
        "selection_rule": plan["selection_rule"],
        "confirmation_gate": plan["confirmation_gate"],
    }
    freeze = {
        "schema": "ijoc_tail_policy_freeze_v1",
        "status": "FROZEN",
        "policy_id": frozen_id,
        "calibration_suite_sha256": receipt_sha,
        "selection_gate": (
            "FALLBACK"
            if confirmation_result["fallback_applied"]
            else "PASS"
        ),
        "decision_rule": decision_rule,
        "decision_rule_sha256": canonical_value_sha256(decision_rule),
        "configuration": {
            "candidate": frozen_candidate,
            "formal_budget_specs": spec_bindings,
        },
        "fallback_applied": confirmation_result["fallback_applied"],
    }
    freeze_path = frozen_dir / "tail_policy_freeze.json"
    freeze_sha = write_json(freeze_path, freeze)
    return {
        "path": freeze_path.relative_to(SUBMISSION_ROOT).as_posix(),
        "sha256": freeze_sha,
        "calibration_suite_receipt": {
            "path": receipt_path.relative_to(SUBMISSION_ROOT).as_posix(),
            "sha256": receipt_sha,
        },
        "calibration_evidence": {
            "path": evidence_path.relative_to(SUBMISSION_ROOT).as_posix(),
            "sha256": evidence_sha,
        },
        "frozen_candidate_id": frozen_id,
        "fallback_applied": confirmation_result["fallback_applied"],
        "formal_budget_specs": spec_bindings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--phase",
        choices=("gate", "all"),
        default="all",
        help="Run only the 24-row plumbing gate or the complete freeze workflow.",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    plan_path = args.plan.expanduser().resolve()
    plan = strict_json(plan_path)
    plan_sha = file_sha256(plan_path)
    case_manifest_path = resolve_bound(
        plan_path.parent,
        plan["case_manifest"],
        allowed_root=SUBMISSION_ROOT,
    )
    case_manifest = strict_json(case_manifest_path)
    cases = list(case_manifest["cases"])
    calibration_instance_artifacts = []
    for case in cases:
        for artifact in case["artifacts"]:
            source = (
                case_manifest_path.parent / str(artifact["path"])
            ).resolve()
            source.relative_to(SUBMISSION_ROOT.resolve())
            actual_sha = file_sha256(source)
            if actual_sha != str(artifact["sha256"]):
                raise ValueError(
                    f"Calibration instance SHA-256 mismatch: {source}"
                )
            if source.stat().st_size != int(artifact["bytes"]):
                raise ValueError(
                    f"Calibration instance byte-count mismatch: {source}"
                )
            calibration_instance_artifacts.append(
                {
                    "case_id": str(case["case_id"]),
                    "family": str(case["family"]),
                    "path": source.relative_to(
                        SUBMISSION_ROOT.resolve()
                    ).as_posix(),
                    "sha256": actual_sha,
                    "bytes": source.stat().st_size,
                }
            )
    calibration_instance_artifacts.sort(
        key=lambda item: (item["case_id"], item["path"])
    )
    candidates = list(plan["candidate_menu"])
    candidate_by_id = {
        str(item["candidate_id"]): item for item in candidates
    }
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    gate_cases = []
    for family in ("MOTSP", "MOKP"):
        family_cases = sorted(
            (
                case
                for case in cases
                if case["family"] == family and case["split"] == "selection"
            ),
            key=lambda item: str(item["case_id"]),
        )
        gate_cases.extend(
            family_cases[: int(plan["gate"]["cases_per_family"])]
        )
    gate_candidates = [
        candidate_by_id[item] for item in plan["gate"]["candidate_ids"]
    ]
    gate_tasks = make_tasks(
        phase="gate",
        cases=gate_cases,
        candidates=gate_candidates,
        seeds=[int(value) for value in plan["gate"]["seeds"]],
        budget=int(plan["gate"]["budget"]),
        checkpoint_period=int(plan["gate"]["checkpoint_period"]),
        fixed_core=plan["fixed_core"],
        case_manifest_dir=case_manifest_path.parent,
    )
    gate_resumed_row_count = sum(
        task_path(output_root, task).is_file() for task in gate_tasks
    )
    gate_started = time.perf_counter()
    gate_rows = run_tasks(
        gate_tasks,
        output_root=output_root,
        workers=args.workers,
    )
    gate_elapsed = time.perf_counter() - gate_started
    require_complete(gate_rows, len(gate_tasks), "pipeline gate")
    validate_pre_tail_pairing(gate_rows)
    gate_result = build_pipeline_gate_result(
        row_count=len(gate_rows),
        resumed_row_count=gate_resumed_row_count,
        launcher_elapsed_seconds=gate_elapsed,
        maximum_elapsed_seconds=float(plan["gate"]["maximum_elapsed_seconds"]),
    )
    write_json(output_root / "gate_result.json", gate_result)
    if gate_result["pipeline_gate"] != "PASS":
        raise RuntimeError("The time-boxed pipeline gate failed; expansion refused.")
    if args.phase == "gate":
        print(json.dumps(gate_result, indent=2, sort_keys=True))
        return

    selection_cases = [
        case for case in cases if case["split"] == "selection"
    ]
    selection_tasks = make_tasks(
        phase="selection",
        cases=selection_cases,
        candidates=candidates,
        seeds=[int(value) for value in plan["seeds"]],
        budget=int(plan["calibration_budget"]),
        checkpoint_period=int(plan["checkpoint_period"]),
        fixed_core=plan["fixed_core"],
        case_manifest_dir=case_manifest_path.parent,
    )
    selection_rows = run_tasks(
        selection_tasks,
        output_root=output_root,
        workers=args.workers,
    )
    require_complete(selection_rows, len(selection_tasks), "selection")
    validate_pre_tail_pairing(selection_rows)
    selection_result = analyze_selection(selection_rows, candidates)
    write_json(output_root / "selection_result.json", selection_result)

    selected_id = str(selection_result["selected_candidate_id"])
    uniform_id = str(selection_result["matched_uniform_id"])
    confirmation_cases = [
        case for case in cases if case["split"] == "confirmation"
    ]
    confirmation_candidates = [
        candidate_by_id[uniform_id],
        candidate_by_id[selected_id],
    ]
    confirmation_tasks = make_tasks(
        phase="confirmation",
        cases=confirmation_cases,
        candidates=confirmation_candidates,
        seeds=[int(value) for value in plan["seeds"]],
        budget=int(plan["calibration_budget"]),
        checkpoint_period=int(plan["checkpoint_period"]),
        fixed_core=plan["fixed_core"],
        case_manifest_dir=case_manifest_path.parent,
    )
    confirmation_rows = run_tasks(
        confirmation_tasks,
        output_root=output_root,
        workers=args.workers,
    )
    require_complete(
        confirmation_rows,
        len(confirmation_tasks),
        "confirmation",
    )
    validate_pre_tail_pairing(confirmation_rows)
    confirmation_result = analyze_confirmation(
        confirmation_rows,
        selected_id=selected_id,
        uniform_id=uniform_id,
        gate=plan["confirmation_gate"],
    )
    write_json(output_root / "confirmation_result.json", confirmation_result)
    freeze = freeze_specs(
        plan=plan,
        plan_sha=plan_sha,
        case_manifest_sha=file_sha256(case_manifest_path),
        calibration_case_ids=[str(case["case_id"]) for case in cases],
        calibration_instance_artifacts=calibration_instance_artifacts,
        output_root=output_root,
        gate_result=gate_result,
        selection_result=selection_result,
        confirmation_result=confirmation_result,
        all_rows=gate_rows + selection_rows + confirmation_rows,
    )
    summary = {
        "schema": "ijoc_tail_calibration_run_summary_v1",
        "gate": gate_result,
        "selection": selection_result,
        "confirmation": confirmation_result,
        "freeze": freeze,
    }
    write_json(output_root / "calibration_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
