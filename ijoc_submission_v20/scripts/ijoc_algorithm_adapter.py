from __future__ import annotations

"""Single-row adapter for the frozen IJOC cold-process matrix."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mo_nco.archive import ArchiveEntry, ParetoArchive
from mo_nco.baselines import MOTSPParetoLocalSearchOptimizer
from mo_nco.external_pymoo_baseline import run_pymoo
from mo_nco.ijoc_mokp_baselines import run_mokp_baseline
from mo_nco.instance import MultiObjectiveTSPInstance, instance_sha256
from mo_nco.pareto_ijoc_allocation import SearchRewardWeights
from mo_nco.pareto_ijoc_generic_smc import GenericAnnealedParetoSMCOptimizer
from mo_nco.pareto_ijoc_problem import (
    MultiObjectiveKnapsackInstance,
    problem_sha256,
)
from mo_nco.pareto_smc import AnnealedParetoSMCOptimizer
from mo_nco.pareto_smc_spec import analytic_objective_box
from mo_nco.sampler import Diagnostic, OptimizationResult


def canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
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
                raise ValueError(f"Duplicate JSON field {key!r}: {path}")
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
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def resolve_packet_artifact(
    packet_path: Path,
    binding: dict[str, Any],
) -> Path:
    raw = Path(str(binding["path"]))
    if raw.is_absolute():
        raise ValueError("Case-packet artifact paths must be relative.")
    path = (packet_path.parent / raw).resolve()
    path.relative_to(packet_path.parent.resolve())
    if file_sha256(path) != str(binding["sha256"]):
        raise ValueError(f"Case-packet artifact SHA-256 mismatch: {path}")
    return path


def load_problem(packet_path: Path, expected_case_id: str):
    packet = strict_json(packet_path)
    if packet.get("schema") != "ijoc_case_instance_packet_v1":
        raise ValueError("Instance artifact is not an IJOC case packet.")
    if packet.get("case_id") != expected_case_id:
        raise ValueError("Instance packet case_id does not match the run key.")
    artifacts = [
        resolve_packet_artifact(packet_path, binding)
        for binding in packet["artifacts"]
    ]
    family = str(packet["family"])
    if family == "MOTSP":
        if len(artifacts) != 2:
            raise ValueError("MOTSP packet must bind exactly two objectives.")
        problem = MultiObjectiveTSPInstance.from_tsplib_files(artifacts)
        digest = instance_sha256(problem)
    elif family == "MOKP":
        if len(artifacts) != 1:
            raise ValueError("MOKP packet must bind exactly one instance.")
        payload = strict_json(artifacts[0])
        problem = MultiObjectiveKnapsackInstance(
            item_weights=tuple(int(value) for value in payload["item_weights"]),
            profits_by_objective=tuple(
                tuple(int(value) for value in row)
                for row in payload["profits_by_objective"]
            ),
            capacity=int(payload["capacity"]),
            name=str(payload["case_id"]),
        )
        digest = problem_sha256(problem)
    else:
        raise ValueError(f"Unsupported problem family: {family!r}.")
    if digest != packet["problem_sha256"]:
        raise ValueError("Case packet problem SHA-256 does not match.")
    return family, problem, packet


def treatment_parameters(
    configuration: dict[str, Any],
    *,
    budget: int,
) -> tuple[dict[str, Any], int, int]:
    treatment = configuration.get("treatment")
    if not isinstance(treatment, dict):
        raise ValueError("Treatment configuration is missing.")
    fixed = treatment.get("fixed_core")
    tail = treatment.get("frozen_tail_policy")
    if not isinstance(fixed, dict) or not isinstance(tail, dict):
        raise ValueError("Treatment core or tail policy is missing.")
    tail_evaluations = int(round(budget * float(tail["tail_fraction"])))
    directions = tail.get(
        "reference_directions",
        fixed["reference_directions"],
    )
    num_types = len(directions)
    quota = int(
        math.floor(
            tail_evaluations * float(tail["quota_fraction"]) / num_types
        )
    )
    return {
        "fixed": fixed,
        "tail": tail,
        "directions": tuple(
            tuple(float(value) for value in row) for row in directions
        ),
    }, tail_evaluations, quota


def run_treatment(
    family: str,
    problem: Any,
    configuration: dict[str, Any],
    *,
    seed: int,
    budget: int,
    checkpoint_period: int,
) -> OptimizationResult:
    parameters, tail_evaluations, quota = treatment_parameters(
        configuration,
        budget=budget,
    )
    fixed = parameters["fixed"]
    tail = parameters["tail"]
    directions = parameters["directions"]
    reward = SearchRewardWeights(
        hypervolume=float(tail["reward_weights"]["hypervolume"]),
        new_cell=float(tail["reward_weights"]["new_cell"]),
        scalar_improvement=float(
            tail["reward_weights"]["scalar_improvement"]
        ),
    )
    particles_per_reference = int(fixed["particles_per_reference"])
    if family == "MOTSP":
        lower, upper = analytic_objective_box(problem)
        normalized_cell_width = float(fixed["normalized_cell_width"])
        epsilon = tuple(
            (right - left) * normalized_cell_width
            for left, right in zip(lower, upper)
        )
        return AnnealedParetoSMCOptimizer(
            problem,
            particles_per_reference=particles_per_reference,
            evaluations=budget,
            seed=seed,
            beta_schedule=tuple(
                float(value) for value in fixed["beta_schedule"]
            ),
            reference_directions=directions,
            epsilon=epsilon,
            ess_threshold=float(fixed["ess_threshold"]),
            chebyshev_rho=float(fixed["chebyshev_rho"]),
            global_refresh_probability=float(
                fixed["global_refresh_probability"]
            ),
            adaptive_search_evaluations=tail_evaluations,
            adaptive_allocation_policy=str(tail["allocation_policy"]),
            adaptive_minimum_pulls_per_type=quota,
            exp3_exploration=tail["exp3_exploration"],
            search_reward_weights=reward,
            archive_tolerance=0.0,
            archive_max_size=int(fixed["deployment_archive_max_size"]),
            audit_trace_level="summary",
            anytime_checkpoint_period=checkpoint_period,
        ).run()
    cell_width = float(fixed["normalized_cell_width"])
    cell_widths = tuple(
        (right - left) * cell_width
        for left, right in zip(
            problem.objective_lower_bounds,
            problem.objective_upper_bounds,
        )
    )
    return GenericAnnealedParetoSMCOptimizer(
        problem,
        reference_directions=directions,
        particles_per_reference=particles_per_reference,
        evaluations=budget,
        beta_schedule=tuple(
            float(value) for value in fixed["beta_schedule"]
        ),
        ess_threshold=float(fixed["ess_threshold"]),
        chebyshev_rho=float(fixed["chebyshev_rho"]),
        adaptive_search_evaluations=tail_evaluations,
        adaptive_allocation_policy=str(tail["allocation_policy"]),
        minimum_pulls_per_type=quota,
        exp3_exploration=tail["exp3_exploration"],
        reward_weights=reward,
        cell_widths=cell_widths,
        deployment_archive_max_size=int(
            fixed["deployment_archive_max_size"]
        ),
        anytime_checkpoint_period=checkpoint_period,
        seed=seed,
    ).run()


def parse_pymoo_result(
    problem: MultiObjectiveTSPInstance,
    *,
    algorithm: str,
    seed: int,
    budget: int,
    checkpoint_period: int,
    population_size: int,
) -> OptimizationResult:
    with tempfile.TemporaryDirectory(prefix="ijoc-pymoo-") as temporary:
        root = Path(temporary)
        input_path = root / "input.json"
        output_path = root / "archive.csv"
        input_path.write_text(
            json.dumps(
                {
                    "distance_matrices": problem.distance_matrices,
                    "num_cities": problem.num_cities,
                    "num_objectives": problem.num_objectives,
                    "population_size": population_size,
                    "evaluations": budget,
                    "anytime_checkpoint_period": checkpoint_period,
                    "seed": seed,
                },
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        run_pymoo(
            "nsga2" if algorithm == "pymoo-nsga2" else "moead",
            input_path,
            output_path,
        )
        archive = ParetoArchive(max_size=None, tol=0.0)
        with output_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                tour = tuple(int(value) for value in row["tour"].split())
                objectives = tuple(
                    float(row[f"objective_{index}"])
                    for index in range(problem.num_objectives)
                )
                archive.update((ArchiveEntry(tour, objectives),))
                if int(row["evaluations"]) != budget:
                    raise RuntimeError("pymoo final archive budget mismatch.")
        grouped: dict[int, list[ArchiveEntry]] = {}
        elapsed: dict[int, float] = {}
        diagnostics_path = output_path.with_suffix(".diagnostics.csv")
        with diagnostics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                evaluation = int(row["evaluations"])
                if evaluation % checkpoint_period != 0:
                    continue
                tour = tuple(int(value) for value in row["tour"].split())
                objectives = tuple(
                    float(row[f"objective_{index}"])
                    for index in range(problem.num_objectives)
                )
                grouped.setdefault(evaluation, []).append(
                    ArchiveEntry(tour, objectives)
                )
                elapsed[evaluation] = float(row["elapsed_seconds"])
        diagnostics = []
        witnesses = []
        expected = tuple(range(checkpoint_period, budget + 1, checkpoint_period))
        for evaluation in expected:
            snapshot = ParetoArchive(max_size=None, tol=0.0)
            snapshot.update(grouped.get(evaluation, ()))
            if not snapshot.entries:
                raise RuntimeError(
                    f"pymoo checkpoint {evaluation} has no solution witness."
                )
            diagnostics.append(
                Diagnostic(
                    iteration=evaluation,
                    temperature=math.inf,
                    acceptance_rate=0.0,
                    archive_size=len(snapshot),
                    hypervolume_2d=snapshot.hypervolume_2d(
                        reference=analytic_objective_box(problem)[1]
                    ),
                    empirical_energy=0.0,
                    positive_archive_jump=0.0,
                    front=tuple(
                        entry.objectives for entry in snapshot.entries
                    ),
                    elapsed_seconds=elapsed[evaluation],
                )
            )
            witnesses.append(
                {
                    "evaluation": evaluation,
                    "entries": tuple(
                        {
                            "tour": entry.tour,
                            "objectives": entry.objectives,
                        }
                        for entry in snapshot.entries
                    ),
                }
            )
        if {
            (entry.tour, entry.objectives) for entry in archive.entries
        } != {
            (tuple(entry["tour"]), tuple(entry["objectives"]))
            for entry in witnesses[-1]["entries"]
        }:
            raise RuntimeError("pymoo final checkpoint differs from final archive.")
        return OptimizationResult(
            particles=tuple(entry.tour for entry in archive.entries),
            objectives=tuple(entry.objectives for entry in archive.entries),
            archive=archive,
            diagnostics=tuple(diagnostics),
            metadata={
                "algorithm": algorithm,
                "evaluations_used": budget,
                "observed_anytime_checkpoints": expected,
                "checkpoint_solution_witnesses": tuple(witnesses),
                "native_archive_completeness_gate": "PASS",
                "native_archive_completeness_contract": (
                    "unbounded_exact_nondominated_all_evaluated_candidates_v2"
                ),
            },
        )


def run_row(
    family: str,
    problem: Any,
    configuration: dict[str, Any],
    *,
    seed: int,
    budget: int,
    checkpoint_period: int,
) -> OptimizationResult:
    algorithm = str(configuration["algorithm"])
    if algorithm == "ijoc-pareto-smc":
        return run_treatment(
            family,
            problem,
            configuration,
            seed=seed,
            budget=budget,
            checkpoint_period=checkpoint_period,
        )
    if family == "MOKP":
        return run_mokp_baseline(
            algorithm,
            problem,
            evaluations=budget,
            seed=seed,
            anytime_checkpoint_period=checkpoint_period,
        )
    if algorithm in {
        "motsp-pls-native-v1",
        "motsp-pls-restart-native-v2",
    }:
        restart_policy = "none"
        restart_random_attempts = 64
        archive_tolerance = 0.0
        if algorithm == "motsp-pls-restart-native-v2":
            restart_policy = str(configuration["stalled_expansion_policy"])
            restart_random_attempts = int(
                configuration["restart_random_attempts"]
            )
            archive_tolerance = float(configuration["archive_tolerance"])
            if (
                restart_policy != "uniform-random-unvisited-v1"
                or restart_random_attempts != 64
                or archive_tolerance != 0.0
                or configuration.get("liveness_contract")
                != "each_nonterminal_step_adds_evaluation_or_fails_v1"
            ):
                raise ValueError(
                    "MOTSP PLS restart-v2 configuration contract mismatch."
                )
        return MOTSPParetoLocalSearchOptimizer(
            problem,
            population_size=int(configuration["population_size"]),
            evaluations=budget,
            seed=seed,
            log_period=checkpoint_period,
            archive_max_size=None,
            archive_tolerance=archive_tolerance,
            neighborhood_sample=int(configuration["neighborhood_sample"]),
            scalar_guided=True,
            anytime_checkpoint_period=checkpoint_period,
            stalled_expansion_policy=restart_policy,
            restart_random_attempts=restart_random_attempts,
        ).run()
    if algorithm in {"pymoo-nsga2", "pymoo-moead"}:
        return parse_pymoo_result(
            problem,
            algorithm=algorithm,
            seed=seed,
            budget=budget,
            checkpoint_period=checkpoint_period,
            population_size=int(configuration["population_size"]),
        )
    raise ValueError(
        f"Algorithm {algorithm!r} is not valid for family {family!r}."
    )


def objective_bounds(family: str, problem: Any):
    if family == "MOTSP":
        return analytic_objective_box(problem)
    return (
        tuple(float(value) for value in problem.objective_lower_bounds),
        tuple(float(value) for value in problem.objective_upper_bounds),
    )


def normalized_auc(
    diagnostics: Iterable[Diagnostic],
    *,
    budget: int,
    checkpoint_period: int,
    box_volume: float,
) -> tuple[float, tuple[int, ...]]:
    expected = tuple(range(checkpoint_period, budget + 1, checkpoint_period))
    by_iteration = {
        int(item.iteration): float(item.hypervolume_2d)
        for item in diagnostics
    }
    if set(expected) - set(by_iteration):
        raise RuntimeError("Algorithm result is missing a common checkpoint.")
    area = 0.0
    previous_evaluation = 0
    previous_hv = 0.0
    for evaluation in expected:
        area += previous_hv * (evaluation - previous_evaluation)
        previous_evaluation = evaluation
        previous_hv = by_iteration[evaluation]
    return area / (budget * box_volume), expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    payload = strict_json(input_path)
    if payload.get("schema") != "ijoc_cold_process_input_v1":
        raise ValueError("Cold-process input schema mismatch.")
    configuration = payload["configuration"]
    if canonical_digest(configuration) != payload["configuration_sha256"]:
        raise ValueError("Readable configuration SHA-256 mismatch.")
    run_key = payload["run_key"]
    for key in ("case_id", "algorithm", "seed", "budget"):
        if configuration.get(key) != run_key.get(key):
            raise ValueError(f"Configuration field {key!r} does not match run_key.")
    instance_path = Path(payload["instance_artifact"]["path"]).resolve()
    if file_sha256(instance_path) != payload["instance_artifact"]["sha256"]:
        raise ValueError("Frozen instance artifact SHA-256 mismatch.")
    family, problem, packet = load_problem(
        instance_path,
        str(run_key["case_id"]),
    )
    if family not in configuration["families"]:
        raise ValueError("Algorithm configuration does not support this family.")
    budget = int(run_key["budget"])
    seed = int(run_key["seed"])
    checkpoint_period = int(payload["anytime_checkpoint_period"])
    result = run_row(
        family,
        problem,
        configuration,
        seed=seed,
        budget=budget,
        checkpoint_period=checkpoint_period,
    )
    if str(run_key["algorithm"]) == "motsp-pls-restart-native-v2" and (
        result.metadata.get("algorithm") != "motsp-pls-restart-native-v2"
        or result.metadata.get("liveness_gate") != "PASS"
        or result.metadata.get("stalled_expansion_policy")
        != "uniform-random-unvisited-v1"
        or result.archive.tol != 0.0
    ):
        raise RuntimeError("MOTSP PLS restart-v2 liveness contract failed.")
    evaluations_used = int(result.metadata.get("evaluations_used", budget))
    if evaluations_used != budget:
        raise RuntimeError("Algorithm did not consume the exact frozen budget.")
    lower, upper = objective_bounds(family, problem)
    box_volume = math.prod(
        right - left for left, right in zip(lower, upper)
    )
    auc, observed = normalized_auc(
        result.diagnostics,
        budget=budget,
        checkpoint_period=checkpoint_period,
        box_volume=box_volume,
    )
    witnesses = result.metadata.get("checkpoint_solution_witnesses")
    if not isinstance(witnesses, (tuple, list)):
        raise RuntimeError("Algorithm did not expose checkpoint solution witnesses.")
    witness_checkpoints = tuple(int(item["evaluation"]) for item in witnesses)
    if witness_checkpoints != observed:
        raise RuntimeError("Checkpoint witness grid is incomplete.")
    archive_entries = tuple(
        {
            "solution": entry.tour,
            "objectives": entry.objectives,
        }
        for entry in result.archive.entries
    )
    checkpoint_entries = tuple(
        {
            "evaluation": int(item["evaluation"]),
            "entries": tuple(
                {
                    "solution": entry.get("solution", entry.get("tour")),
                    "objectives": entry["objectives"],
                }
                for entry in item["entries"]
            ),
        }
        for item in witnesses
    )
    archive_payload = {
        "schema": "ijoc_all_evaluated_archive_v1",
        "run_key": run_key,
        "instance_packet_sha256": file_sha256(instance_path),
        "problem_sha256": packet["problem_sha256"],
        "dominance_tolerance": 0.0,
        "archive_contract": (
            "unbounded_exact_nondominated_union_of_all_evaluated_candidates_v2"
        ),
        "entries": archive_entries,
    }
    checkpoint_payload = {
        "schema": "ijoc_checkpoint_solution_witnesses_v1",
        "run_key": run_key,
        "checkpoint_period": checkpoint_period,
        "checkpoints": checkpoint_entries,
    }
    archive_path = output_path.with_name("all_evaluated_archive.json")
    checkpoint_path = output_path.with_name("checkpoint_witnesses.json")
    archive_sha = write_json(archive_path, archive_payload)
    checkpoint_sha = write_json(checkpoint_path, checkpoint_payload)
    final_hv = result.archive.hypervolume_2d(reference=upper)
    result_payload = {
        "schema": "ijoc_algorithm_result_v1",
        "run_key": run_key,
        "status": "SUCCESS",
        "evaluations_used": evaluations_used,
        "observed_checkpoints": observed,
        "archive_artifact": {
            "path": archive_path.name,
            "sha256": archive_sha,
        },
        "checkpoint_artifact": {
            "path": checkpoint_path.name,
            "sha256": checkpoint_sha,
        },
        "metrics": {
            "normalized_left_continuous_evaluation_auc": auc,
            "normalized_final_hypervolume": final_hv / box_volume,
            "final_hypervolume": final_hv,
            "archive_size": len(result.archive),
            "objective_box_lower": lower,
            "objective_box_upper": upper,
            "problem_family": family,
        },
    }
    write_json(output_path, result_payload)


if __name__ == "__main__":
    main()
