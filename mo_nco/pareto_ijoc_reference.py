from __future__ import annotations

"""Independent metric-reference construction for the IJOC matched study.

This module deliberately does not call any formal-study treatment or baseline.
It builds a supplied reference set with a separate, fixed weighted-scalarization
multistart search.  Normalization boxes come from analytic feasibility bounds,
not from a formal-arm outcome.  Consequently, the resulting IGD-style metrics
are supplied-reference-relative and are not evidence of true-front coverage.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

from .archive import ArchiveEntry, ParetoArchive, dominates
from .instance import MultiObjectiveTSPInstance, instance_sha256
from .pareto_ijoc_problem import (
    MultiObjectiveKnapsackInstance,
    problem_sha256,
)


REFERENCE_CASE_SCHEMA = "ijoc_calibration_reference_case_v1"
REFERENCE_CONTEXT_SCHEMA = "ijoc_reference_calibration_precommit_v1"
REFERENCE_MANIFEST_SCHEMA = "ijoc_reference_calibration_completion_evidence_v1"
REFERENCE_RECEIPT_SCHEMA = "ijoc_reference_calibration_completion_receipt_v1"
REFERENCE_WITNESS_SCHEMA = "ijoc_reference_calibration_run_witness_v1"
REFERENCE_AUDIT_SCHEMA = "ijoc_reference_calibration_audit_v1"
FORMAL_CASE_MANIFEST_SCHEMA = "ijoc_case_suite_manifest_v1"
INSTANCE_PACKET_MANIFEST_SCHEMA = "ijoc_case_instance_packet_manifest_v1"
SEARCH_ALGORITHM_ID = "calibration-weighted-multistart-local-search-v1"
SOURCE_ROLE = "reference_calibration_precommitted_disjoint_arms_and_seeds"
CLAIM_BOUNDARY = "supplied_reference_relative_only_not_true_front"

FORMAL_ARM_IDS = (
    "ijoc-pareto-smc",
    "motsp-pls-native-v1",
    "pymoo-moead",
    "pymoo-nsga2",
    "mokp-binary-moead-native-v1",
    "mokp-binary-nsga2-native-v1",
    "mokp-pls-native-v1",
)


@dataclass(frozen=True)
class FormalCase:
    case_id: str
    family: str
    size: int
    problem_sha256: str
    artifact_paths: tuple[Path, ...]
    artifact_bindings: tuple[dict[str, object], ...]
    problem: MultiObjectiveTSPInstance | MultiObjectiveKnapsackInstance
    packet_path: Path | None = None
    packet_sha256: str | None = None


@dataclass(frozen=True)
class ReferenceSearchResult:
    entries: tuple[ArchiveEntry, ...]
    evaluations_used: int
    seed_evaluation_counts: tuple[int, ...]
    ideal: tuple[float, ...]
    nadir: tuple[float, ...]
    hv_reference: tuple[float, ...]
    bound_derivation: str


@dataclass(frozen=True)
class ReferenceSuiteResult:
    output_directory: Path
    context_path: Path
    manifest_path: Path
    receipt_path: Path
    audit_path: Path
    case_count: int
    family_counts: Mapping[str, int]
    status: str
    elapsed_seconds: float
    reused_existing: bool


def _reject_duplicate_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field is forbidden: {key!r}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}.")


def strict_json(path: str | Path) -> Mapping[str, Any]:
    source = Path(path).resolve()
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid strict JSON at {source}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return value


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes(value)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _safe_relative_path(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a nonempty relative path.")
    raw = Path(relative)
    if raw.is_absolute():
        raise ValueError(f"{label} must be relative.")
    path = (root / raw).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes its manifest directory.") from error
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    return path


def _sha256_string(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal.") from error
    return value.lower()


def _load_mokp(path: Path, case_id: str) -> MultiObjectiveKnapsackInstance:
    payload = strict_json(path)
    required = {
        "schema",
        "case_id",
        "family",
        "num_items",
        "num_objectives",
        "item_weights",
        "profits_by_objective",
        "capacity",
        "generator",
    }
    if set(payload) != required:
        raise ValueError(
            f"MOKP instance {case_id!r} has an unexpected shape: "
            f"missing={sorted(required - set(payload))}, "
            f"extra={sorted(set(payload) - required)}."
        )
    if (
        payload.get("schema") != "ijoc_mokp_integer_instance_v1"
        or payload.get("case_id") != case_id
        or payload.get("family") != "MOKP"
    ):
        raise ValueError(f"MOKP instance identity mismatch for {case_id!r}.")
    weights_raw = payload.get("item_weights")
    profits_raw = payload.get("profits_by_objective")
    if not isinstance(weights_raw, list) or not isinstance(profits_raw, list):
        raise ValueError(f"MOKP arrays are malformed for {case_id!r}.")
    problem = MultiObjectiveKnapsackInstance(
        item_weights=tuple(int(value) for value in weights_raw),
        profits_by_objective=tuple(
            tuple(int(value) for value in row)
            for row in profits_raw
        ),
        capacity=int(payload["capacity"]),
        name=case_id,
    )
    if (
        int(payload["num_items"]) != problem.solution_size
        or int(payload["num_objectives"]) != problem.num_objectives
    ):
        raise ValueError(f"MOKP declared dimensions disagree for {case_id!r}.")
    return problem


def load_formal_cases(
    manifest_path: str | Path,
) -> tuple[Mapping[str, Any], tuple[FormalCase, ...]]:
    path = Path(manifest_path).resolve()
    manifest = strict_json(path)
    if manifest.get("schema") != FORMAL_CASE_MANIFEST_SCHEMA:
        raise ValueError("Formal case-manifest schema mismatch.")
    if manifest.get("role") != "formal_test_frozen_before_optimizer_runs":
        raise ValueError("Case manifest is not the frozen formal-test suite.")
    if manifest.get("calibration_overlap_count") != 0:
        raise ValueError("Formal manifest declares calibration overlap.")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Formal manifest must contain cases.")

    seen: set[str] = set()
    cases: list[FormalCase] = []
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Formal case {index} must be an object.")
        case_id = raw_case.get("case_id")
        family = raw_case.get("family")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"Formal case {index} has no case_id.")
        if case_id in seen:
            raise ValueError(f"Duplicate formal case_id: {case_id!r}.")
        seen.add(case_id)
        if family not in {"MOTSP", "MOKP"}:
            raise ValueError(f"Unsupported family for {case_id!r}: {family!r}.")
        if raw_case.get("split") != "formal_test":
            raise ValueError(f"Case {case_id!r} is not in formal_test.")
        declared_problem_sha = _sha256_string(
            raw_case.get("problem_sha256"),
            f"{case_id}.problem_sha256",
        )
        raw_artifacts = raw_case.get("artifacts")
        expected_count = 2 if family == "MOTSP" else 1
        if (
            not isinstance(raw_artifacts, list)
            or len(raw_artifacts) != expected_count
        ):
            raise ValueError(
                f"Case {case_id!r} must bind exactly {expected_count} artifacts."
            )
        artifact_paths: list[Path] = []
        artifact_bindings: list[dict[str, object]] = []
        for artifact_index, raw_artifact in enumerate(raw_artifacts):
            if not isinstance(raw_artifact, dict):
                raise ValueError(f"Artifact {artifact_index} for {case_id!r} is invalid.")
            if set(raw_artifact) != {"path", "sha256", "bytes"}:
                raise ValueError(
                    f"Artifact binding {artifact_index} for {case_id!r} "
                    "has an unexpected shape."
                )
            artifact_path = _safe_relative_path(
                path.parent,
                raw_artifact.get("path"),
                f"{case_id}.artifact[{artifact_index}]",
            )
            declared_sha = _sha256_string(
                raw_artifact.get("sha256"),
                f"{case_id}.artifact[{artifact_index}].sha256",
            )
            actual_sha = file_sha256(artifact_path)
            if actual_sha != declared_sha:
                raise ValueError(f"Artifact hash mismatch for {case_id!r}.")
            if raw_artifact.get("bytes") != artifact_path.stat().st_size:
                raise ValueError(f"Artifact byte count mismatch for {case_id!r}.")
            artifact_paths.append(artifact_path)
            artifact_bindings.append(
                {
                    "path": artifact_path.relative_to(path.parent).as_posix(),
                    "sha256": actual_sha,
                    "bytes": artifact_path.stat().st_size,
                }
            )

        if family == "MOTSP":
            problem: MultiObjectiveTSPInstance | MultiObjectiveKnapsackInstance
            problem = MultiObjectiveTSPInstance.from_tsplib_files(artifact_paths)
            actual_problem_sha = instance_sha256(problem)
        else:
            problem = _load_mokp(artifact_paths[0], case_id)
            actual_problem_sha = problem_sha256(problem)
        if actual_problem_sha != declared_problem_sha:
            raise ValueError(f"Problem-state hash mismatch for {case_id!r}.")
        problem_size = (
            problem.num_cities
            if isinstance(problem, MultiObjectiveTSPInstance)
            else problem.solution_size
        )
        if int(raw_case.get("size", -1)) != problem_size:
            raise ValueError(f"Problem size mismatch for {case_id!r}.")
        if int(raw_case.get("num_objectives", -1)) != problem.num_objectives:
            raise ValueError(f"Objective dimension mismatch for {case_id!r}.")
        if problem.num_objectives != 2:
            raise ValueError("The current IJOC metric contract requires two objectives.")
        cases.append(
            FormalCase(
                case_id=case_id,
                family=family,
                size=problem_size,
                problem_sha256=actual_problem_sha,
                artifact_paths=tuple(artifact_paths),
                artifact_bindings=tuple(artifact_bindings),
                problem=problem,
            )
        )
    return manifest, tuple(cases)


def bind_instance_packets(
    cases: Sequence[FormalCase],
    packet_manifest_path: str | Path,
) -> tuple[FormalCase, ...]:
    """Validate the adapter packets and bind each one to its raw instance bytes."""

    path = Path(packet_manifest_path).resolve()
    manifest = strict_json(path)
    if manifest.get("schema") != INSTANCE_PACKET_MANIFEST_SCHEMA:
        raise ValueError("Instance-packet manifest schema mismatch.")
    raw_packets = manifest.get("packets")
    if not isinstance(raw_packets, list) or len(raw_packets) != len(cases):
        raise ValueError("Instance-packet manifest does not cover the formal cases.")
    by_case = {case.case_id: case for case in cases}
    bound: dict[str, FormalCase] = {}
    for index, raw_packet in enumerate(raw_packets):
        if not isinstance(raw_packet, dict) or set(raw_packet) != {
            "case_id",
            "family",
            "path",
            "sha256",
            "child_artifact_sha256",
        }:
            raise ValueError(f"Instance packet binding {index} has an invalid shape.")
        case_id = raw_packet.get("case_id")
        if not isinstance(case_id, str) or case_id not in by_case or case_id in bound:
            raise ValueError(f"Unknown or duplicate packet case_id: {case_id!r}.")
        case = by_case[case_id]
        if raw_packet.get("family") != case.family:
            raise ValueError(f"Instance packet family mismatch for {case_id!r}.")
        packet_path = _safe_relative_path(
            path.parent,
            raw_packet.get("path"),
            f"{case_id}.packet",
        )
        packet_sha = _sha256_string(
            raw_packet.get("sha256"),
            f"{case_id}.packet.sha256",
        )
        if file_sha256(packet_path) != packet_sha:
            raise ValueError(f"Instance packet hash mismatch for {case_id!r}.")
        raw_hashes = tuple(
            str(binding["sha256"]) for binding in case.artifact_bindings
        )
        if raw_packet.get("child_artifact_sha256") != list(raw_hashes):
            raise ValueError(f"Instance packet child hashes mismatch for {case_id!r}.")
        packet = strict_json(packet_path)
        if (
            packet.get("schema") != "ijoc_case_instance_packet_v1"
            or packet.get("case_id") != case_id
            or packet.get("family") != case.family
            or packet.get("problem_sha256") != case.problem_sha256
        ):
            raise ValueError(f"Instance packet identity mismatch for {case_id!r}.")
        packet_artifacts = packet.get("artifacts")
        if not isinstance(packet_artifacts, list) or [
            item.get("sha256") if isinstance(item, dict) else None
            for item in packet_artifacts
        ] != list(raw_hashes):
            raise ValueError(f"Instance packet raw-artifact binding mismatch for {case_id!r}.")
        bound[case_id] = FormalCase(
            case_id=case.case_id,
            family=case.family,
            size=case.size,
            problem_sha256=case.problem_sha256,
            artifact_paths=case.artifact_paths,
            artifact_bindings=case.artifact_bindings,
            problem=case.problem,
            packet_path=packet_path,
            packet_sha256=packet_sha,
        )
    if set(bound) != set(by_case):
        raise ValueError("Instance packets do not exactly cover the formal cases.")
    return tuple(bound[case.case_id] for case in cases)


def analytic_metric_box(
    problem: MultiObjectiveTSPInstance | MultiObjectiveKnapsackInstance,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    str,
]:
    """Return a feasibility-valid normalization box and strict HV reference."""

    if isinstance(problem, MultiObjectiveTSPInstance):
        ideal = tuple(0.0 for _ in range(problem.num_objectives))
        nadir = tuple(
            max(
                1.0,
                float(problem.num_cities)
                * max(float(value) for row in matrix for value in row),
            )
            for matrix in problem.distance_matrices
        )
        derivation = (
            "A Hamiltonian cycle uses exactly n nonnegative edges. For each "
            "objective, every edge is bounded by the maximum entry of its "
            "frozen distance matrix, so 0 <= f_j <= n*max_edge_j. A unit floor "
            "only widens a degenerate zero-cost box."
        )
    elif isinstance(problem, MultiObjectiveKnapsackInstance):
        ideal = tuple(
            -float(max(1, sum(profits)))
            for profits in problem.profits_by_objective
        )
        nadir = tuple(0.0 for _ in range(problem.num_objectives))
        derivation = (
            "Every feasible selected-profit sum lies between zero and the sum "
            "of all nonnegative item profits. After canonical negation, "
            "-sum(profit_j) <= f_j <= 0. A unit floor only widens a "
            "zero-profit objective."
        )
    else:  # pragma: no cover - guarded by the public loader
        raise TypeError(f"Unsupported problem type: {type(problem)!r}.")
    hv_reference = tuple(
        upper + 0.05 * (upper - lower) + 1.0
        for lower, upper in zip(ideal, nadir)
    )
    return ideal, nadir, hv_reference, derivation


def _normalized_scalar_score(
    objectives: Sequence[float],
    ideal: Sequence[float],
    nadir: Sequence[float],
    weight: float,
) -> float:
    normalized = [
        (float(value) - lower) / (upper - lower)
        for value, lower, upper in zip(objectives, ideal, nadir)
    ]
    return weight * normalized[0] + (1.0 - weight) * normalized[1]


def _weight_grid(count: int) -> tuple[float, ...]:
    if count < 2:
        raise ValueError("weight_grid_size must be at least two.")
    return tuple(index / (count - 1) for index in range(count))


def _stable_case_seed(case_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{case_id}\0{seed}".encode("utf-8")).digest()
    return seed ^ int.from_bytes(digest[:8], "big")


def _nearest_neighbor_tour(
    problem: MultiObjectiveTSPInstance,
    weight: float,
) -> tuple[int, ...]:
    matrices = problem.distance_matrices
    remaining = set(range(1, problem.num_cities))
    tour = [0]
    while remaining:
        source = tour[-1]
        target = min(
            remaining,
            key=lambda city: (
                weight * float(matrices[0][source][city])
                + (1.0 - weight) * float(matrices[1][source][city]),
                city,
            ),
        )
        remaining.remove(target)
        tour.append(target)
    return tuple(tour)


def _random_tour(problem: MultiObjectiveTSPInstance, rng: random.Random) -> tuple[int, ...]:
    remainder = list(range(1, problem.num_cities))
    rng.shuffle(remainder)
    return (0, *remainder)


def _greedy_knapsack_solution(
    problem: MultiObjectiveKnapsackInstance,
    weight: float,
    rng: random.Random,
) -> tuple[int, ...]:
    scales = [
        max(1.0, float(sum(profits)))
        for profits in problem.profits_by_objective
    ]
    ranked = []
    for index, item_weight in enumerate(problem.item_weights):
        utility = (
            weight * problem.profits_by_objective[0][index] / scales[0]
            + (1.0 - weight)
            * problem.profits_by_objective[1][index]
            / scales[1]
        ) / item_weight
        ranked.append((-(utility * (1.0 + rng.uniform(-0.025, 0.025))), index))
    solution = [0] * problem.solution_size
    remaining = problem.capacity
    for _, index in sorted(ranked):
        item_weight = problem.item_weights[index]
        if item_weight <= remaining:
            solution[index] = 1
            remaining -= item_weight
    return tuple(solution)


def _mutate_knapsack(
    problem: MultiObjectiveKnapsackInstance,
    current: tuple[int, ...],
    rng: random.Random,
) -> tuple[int, ...]:
    candidate = list(current)
    index = rng.randrange(problem.solution_size)
    if candidate[index]:
        candidate[index] = 0
        return tuple(candidate)
    current_weight = problem.total_weight(current)
    required = current_weight + problem.item_weights[index] - problem.capacity
    if required > 0:
        selected = [item for item, value in enumerate(candidate) if value]
        rng.shuffle(selected)
        removed = 0
        for item in selected:
            candidate[item] = 0
            removed += problem.item_weights[item]
            if removed >= required:
                break
    candidate[index] = 1
    proposed = tuple(candidate)
    problem.validate_solution(proposed)
    return proposed


def _partition_budget(total: int, count: int) -> tuple[int, ...]:
    if total < count:
        raise ValueError("Evaluation budget must allocate at least one row per seed.")
    quotient, remainder = divmod(total, count)
    return tuple(
        quotient + (1 if index < remainder else 0)
        for index in range(count)
    )


def search_reference_case(
    case: FormalCase,
    *,
    seeds: Sequence[int],
    evaluation_budget: int,
    weight_grid_size: int = 21,
    restart_period: int = 64,
) -> ReferenceSearchResult:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Reference seeds must be nonempty and duplicate-free.")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("Reference seeds must be nonnegative integers.")
    if restart_period < 2:
        raise ValueError("restart_period must be at least two.")
    ideal, nadir, hv_reference, bound_derivation = analytic_metric_box(case.problem)
    weights = _weight_grid(weight_grid_size)
    allocations = _partition_budget(evaluation_budget, len(seeds))
    archive = ParetoArchive(max_size=None, tol=0.0)

    for seed_index, (seed, allocation) in enumerate(zip(seeds, allocations)):
        rng = random.Random(_stable_case_seed(case.case_id, seed))
        used = 0
        restart = 0
        current: tuple[int, ...] | None = None
        current_objectives: tuple[float, ...] | None = None
        current_weight = 0.5
        while used < allocation:
            if current is None or used % restart_period == 0:
                current_weight = weights[
                    (restart + seed_index * 7) % len(weights)
                ]
                if isinstance(case.problem, MultiObjectiveTSPInstance):
                    if restart % 3 == 0:
                        current = _nearest_neighbor_tour(
                            case.problem,
                            current_weight,
                        )
                    else:
                        current = _random_tour(case.problem, rng)
                else:
                    current = _greedy_knapsack_solution(
                        case.problem,
                        current_weight,
                        rng,
                    )
                current_objectives = tuple(
                    float(value) for value in case.problem.evaluate(current)
                )
                archive.update([ArchiveEntry(current, current_objectives)])
                used += 1
                restart += 1
                continue

            if isinstance(case.problem, MultiObjectiveTSPInstance):
                left, right = sorted(
                    rng.sample(range(1, case.problem.num_cities), 2)
                )
                candidate_list = list(current)
                candidate_list[left : right + 1] = reversed(
                    candidate_list[left : right + 1]
                )
                candidate = tuple(candidate_list)
                candidate_objectives = tuple(
                    float(value)
                    for value in case.problem.evaluate_two_opt(
                        current,
                        current_objectives,
                        left,
                        right,
                    )
                )
            else:
                candidate = _mutate_knapsack(case.problem, current, rng)
                candidate_objectives = tuple(
                    float(value) for value in case.problem.evaluate(candidate)
                )
            archive.update([ArchiveEntry(candidate, candidate_objectives)])
            used += 1
            if _normalized_scalar_score(
                candidate_objectives,
                ideal,
                nadir,
                current_weight,
            ) < _normalized_scalar_score(
                current_objectives,
                ideal,
                nadir,
                current_weight,
            ):
                current = candidate
                current_objectives = candidate_objectives

    entries = tuple(archive.entries)
    if not entries:
        raise RuntimeError(f"Reference search returned an empty archive for {case.case_id}.")
    _audit_search_entries(case, entries, ideal, nadir, hv_reference)
    return ReferenceSearchResult(
        entries=entries,
        evaluations_used=sum(allocations),
        seed_evaluation_counts=allocations,
        ideal=ideal,
        nadir=nadir,
        hv_reference=hv_reference,
        bound_derivation=bound_derivation,
    )


def _audit_search_entries(
    case: FormalCase,
    entries: Sequence[ArchiveEntry],
    ideal: Sequence[float],
    nadir: Sequence[float],
    hv_reference: Sequence[float],
) -> None:
    objective_seen: set[tuple[float, ...]] = set()
    for index, entry in enumerate(entries):
        if isinstance(case.problem, MultiObjectiveTSPInstance):
            case.problem.validate_tour(entry.tour)
        else:
            case.problem.validate_solution(entry.tour)
        replayed = tuple(float(value) for value in case.problem.evaluate(entry.tour))
        if replayed != tuple(entry.objectives):
            raise RuntimeError(
                f"Reference witness replay failed for {case.case_id} entry {index}."
            )
        if replayed in objective_seen:
            raise RuntimeError(f"Duplicate reference objective for {case.case_id}.")
        objective_seen.add(replayed)
        if any(
            value < lower or value > upper
            for value, lower, upper in zip(replayed, ideal, nadir)
        ):
            raise RuntimeError(
                f"Analytic metric box does not contain {case.case_id} entry {index}."
            )
        if any(
            reference <= value
            for reference, value in zip(hv_reference, replayed)
        ):
            raise RuntimeError(
                f"HV reference does not strictly dominate {case.case_id} entry {index}."
            )
    for left, left_entry in enumerate(entries):
        for right, right_entry in enumerate(entries):
            if left != right and dominates(
                left_entry.objectives,
                right_entry.objectives,
                tol=0.0,
            ):
                raise RuntimeError(
                    f"Reference archive is dominated for {case.case_id}."
                )


def _binding(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _final_output_binding(
    path: Path,
    *,
    staging_root: Path,
    output_root: Path,
    source_root: Path,
) -> dict[str, object]:
    relative = path.resolve().relative_to(staging_root.resolve())
    final_path = (output_root / relative).resolve()
    try:
        declared_path = final_path.relative_to(source_root.resolve())
    except ValueError as error:
        raise ValueError(
            "Reference output must be a descendant of artifact_source_root."
        ) from error
    return {
        "path": declared_path.as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _select_cases(
    cases: Sequence[FormalCase],
    case_limit_per_family: int | None,
) -> tuple[FormalCase, ...]:
    ordered = sorted(cases, key=lambda case: (case.family, case.case_id))
    if case_limit_per_family is None:
        return tuple(ordered)
    if case_limit_per_family <= 0:
        raise ValueError("case_limit_per_family must be positive.")
    selected: list[FormalCase] = []
    for family in ("MOKP", "MOTSP"):
        family_cases = [case for case in ordered if case.family == family]
        if len(family_cases) < case_limit_per_family:
            raise ValueError(f"Not enough {family} cases for the requested gate.")
        selected.extend(family_cases[:case_limit_per_family])
    return tuple(selected)


def _validate_external_receipts(
    *,
    formal_cases: Sequence[FormalCase],
    tail_calibration_receipt_path: Path,
    tail_policy_path: Path,
) -> tuple[str, str]:
    tail_receipt = strict_json(tail_calibration_receipt_path)
    if (
        tail_receipt.get("schema") != "ijoc_calibration_suite_receipt_v1"
        or tail_receipt.get("status") != "COMPLETE"
        or tail_receipt.get("evidence_scope") != "tail_policy_selection_only"
    ):
        raise ValueError("Tail-policy calibration receipt is not a completed selection receipt.")
    calibration_case_ids = tail_receipt.get("calibration_case_ids")
    if not isinstance(calibration_case_ids, list):
        raise ValueError("Tail-policy receipt lacks calibration_case_ids.")
    overlap = set(str(item) for item in calibration_case_ids) & {
        case.case_id for case in formal_cases
    }
    if overlap:
        raise ValueError(
            "Tail-policy calibration and formal case IDs overlap: "
            f"{sorted(overlap)}."
        )
    tail_receipt_sha = file_sha256(tail_calibration_receipt_path)
    tail_policy = strict_json(tail_policy_path)
    if (
        tail_policy.get("schema") != "ijoc_tail_policy_freeze_v1"
        or tail_policy.get("status") != "FROZEN"
        or tail_policy.get("calibration_suite_sha256") != tail_receipt_sha
    ):
        raise ValueError("Tail-policy artifact is not frozen against the supplied receipt.")
    return tail_receipt_sha, file_sha256(tail_policy_path)


def _existing_matches(output: Path, context_digest: str) -> bool:
    context_path = output / "reference_calibration_precommit.json"
    receipt_path = output / "reference_calibration_completion_receipt.json"
    audit_path = output / "reference_calibration_audit.json"
    manifest_path = output / "reference_calibration_completion_evidence.json"
    if not all(path.is_file() for path in (context_path, receipt_path, audit_path, manifest_path)):
        return False
    try:
        receipt = strict_json(receipt_path)
    except ValueError:
        return False
    return (
        receipt.get("schema") == REFERENCE_RECEIPT_SCHEMA
        and receipt.get("status") == "COMPLETE"
        and receipt.get("reference_calibration_precommit_sha256") == context_digest
        and receipt.get("artifact_manifest", {}).get("sha256")
        == file_sha256(manifest_path)
    )


def _archive_previous_output(output: Path) -> Path:
    manifest = output / "reference_calibration_completion_evidence.json"
    suffix = file_sha256(manifest)[:12] if manifest.is_file() else canonical_digest(
        sorted(path.relative_to(output).as_posix() for path in output.rglob("*"))
    )[:12]
    archived = output.with_name(f"{output.name}.superseded-{suffix}")
    if archived.exists():
        raise FileExistsError(
            f"Preserved prior output already exists at {archived}; "
            "refusing to overwrite either copy."
        )
    os.replace(output, archived)
    return archived


def _execute_reference_replay(
    *,
    python_executable: Path,
    verifier_path: Path,
    witness_path: Path,
    packet_path: Path,
    receipt_path: Path,
    timeout_seconds: float | None,
) -> Mapping[str, Any]:
    command = [
        str(python_executable),
        str(verifier_path),
        "--reference-witness",
        str(witness_path),
        "--instance",
        str(packet_path),
        "--output",
        str(receipt_path),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(verifier_path.parents[2]),
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f"Independent reference replay timed out for {witness_path.name}."
        ) from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            "Independent reference replay failed: "
            f"exit={completed.returncode}: {detail[-2000:]}"
        )
    receipt = strict_json(receipt_path)
    if (
        receipt.get("schema") != "ijoc_reference_replay_receipt_v1"
        or receipt.get("status") != "PASS"
        or receipt.get("witness_sha256") != file_sha256(witness_path)
        or receipt.get("instance_packet_sha256") != file_sha256(packet_path)
        or receipt.get("evaluation_code_sha256") != file_sha256(verifier_path)
    ):
        raise RuntimeError("Independent reference replay receipt binding mismatch.")
    return receipt


def _execute_reference_union_replay(
    *,
    python_executable: Path,
    verifier_path: Path,
    witness_path: Path,
    packet_path: Path,
    receipt_path: Path,
    timeout_seconds: float | None,
) -> Mapping[str, Any]:
    command = [
        str(python_executable),
        str(verifier_path),
        "--reference-witness",
        str(witness_path),
        "--instance",
        str(packet_path),
        "--output",
        str(receipt_path),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(verifier_path.parents[2]),
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f"Independent reference union replay timed out for {witness_path.name}."
        ) from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            "Independent reference union replay failed: "
            f"exit={completed.returncode}: {detail[-2000:]}"
        )
    receipt = strict_json(receipt_path)
    if (
        receipt.get("schema") != "ijoc_reference_union_replay_receipt_v1"
        or receipt.get("status") != "PASS"
        or receipt.get("witness_sha256") != file_sha256(witness_path)
        or receipt.get("instance_packet_sha256") != file_sha256(packet_path)
        or receipt.get("evaluation_code_sha256") != file_sha256(verifier_path)
    ):
        raise RuntimeError(
            "Independent reference union replay receipt binding mismatch."
        )
    return receipt


def build_reference_suite(
    *,
    formal_case_manifest_path: str | Path,
    instance_packet_manifest_path: str | Path,
    tail_calibration_receipt_path: str | Path,
    tail_policy_path: str | Path,
    replay_verifier_path: str | Path,
    output_directory: str | Path,
    reference_seeds: Sequence[int],
    formal_seeds: Sequence[int],
    evaluation_budgets: Sequence[int],
    weight_grid_size: int = 21,
    restart_period: int = 64,
    case_limit_per_family: int | None = None,
    time_limit_seconds: float | None = None,
    replace_existing: bool = False,
    python_executable: str | Path = sys.executable,
    artifact_source_root: str | Path | None = None,
) -> ReferenceSuiteResult:
    """Build a transactionally committed independent reference-calibration suite."""

    started = time.perf_counter()
    manifest_path = Path(formal_case_manifest_path).resolve()
    packet_manifest_path = Path(instance_packet_manifest_path).resolve()
    tail_receipt_path = Path(tail_calibration_receipt_path).resolve()
    frozen_tail_path = Path(tail_policy_path).resolve()
    verifier_path = Path(replay_verifier_path).resolve()
    interpreter_path = Path(python_executable).resolve()
    output = Path(output_directory).resolve()
    source_root = (
        Path(artifact_source_root).resolve()
        if artifact_source_root is not None
        else output.parent.resolve()
    )
    if not verifier_path.is_file():
        raise ValueError(f"Replay verifier is missing: {verifier_path}")
    if not interpreter_path.is_file():
        raise ValueError(f"Python executable is missing: {interpreter_path}")
    if (
        not reference_seeds
        or len(set(reference_seeds)) != len(reference_seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in reference_seeds)
        or tuple(sorted(reference_seeds)) != tuple(reference_seeds)
    ):
        raise ValueError(
            "reference_seeds must be strictly increasing nonnegative integers."
        )
    if (
        not formal_seeds
        or len(set(formal_seeds)) != len(formal_seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in formal_seeds)
        or tuple(sorted(formal_seeds)) != tuple(formal_seeds)
    ):
        raise ValueError(
            "formal_seeds must be strictly increasing nonnegative integers."
        )
    if (
        not evaluation_budgets
        or len(set(evaluation_budgets)) != len(evaluation_budgets)
        or any(
            isinstance(budget, bool)
            or not isinstance(budget, int)
            or budget <= 0
            for budget in evaluation_budgets
        )
        or tuple(sorted(evaluation_budgets)) != tuple(evaluation_budgets)
    ):
        raise ValueError("evaluation_budgets must be strictly increasing positive integers.")
    if set(reference_seeds) & set(formal_seeds):
        raise ValueError("Reference-calibration and formal seeds overlap.")
    if time_limit_seconds is not None and time_limit_seconds <= 0.0:
        raise ValueError("time_limit_seconds must be positive.")

    formal_manifest, all_cases = load_formal_cases(manifest_path)
    all_cases = bind_instance_packets(all_cases, packet_manifest_path)
    selected_cases = _select_cases(all_cases, case_limit_per_family)
    family_counts = {
        family: sum(case.family == family for case in selected_cases)
        for family in ("MOKP", "MOTSP")
    }
    if any(count == 0 for count in family_counts.values()):
        raise ValueError("Reference suite must cover both MOTSP and MOKP.")
    tail_receipt_sha, tail_policy_sha = _validate_external_receipts(
        formal_cases=all_cases,
        tail_calibration_receipt_path=tail_receipt_path,
        tail_policy_path=frozen_tail_path,
    )
    verifier_sha = file_sha256(verifier_path)
    builder_source_sha = file_sha256(Path(__file__).resolve())
    builder_configuration = {
        "base_algorithm_id": SEARCH_ALGORITHM_ID,
        "builder_source_sha256": builder_source_sha,
        "weight_grid_size": weight_grid_size,
        "restart_period": restart_period,
        "proposal_family": {
            "MOTSP": "weighted_nearest_or_random_multistart_two_opt_v1",
            "MOKP": "weighted_profit_density_multistart_capacity_repaired_toggle_v1",
        },
        "acceptance": "strict_weighted_normalized_scalar_improvement",
        "archive": "all_evaluated_zero_tolerance_nondominated_union",
    }
    builder_configuration_sha = canonical_digest(builder_configuration)
    reference_algorithm_id = (
        f"{SEARCH_ALGORITHM_ID}__cfg_{builder_configuration_sha[:16]}"
    )
    metric_contract = {
        "objective_sense": ["minimize", "minimize"],
        "dominance_tolerance": 0.0,
        "normalization": "frozen_ideal_nadir_affine",
        "archive_semantics": "calibration_all_evaluated_nondominated",
        "evaluation_code_sha256": verifier_sha,
    }

    context = {
        "schema": REFERENCE_CONTEXT_SCHEMA,
        "suite_id": "pareto_smc_v20_ijoc_metric_reference_calibration_v1",
        "status": "PRECOMMITTED",
        "evidence_scope": "metric_reference_construction_only",
        "cases": [
            {
                "case_id": case.case_id,
                "family": case.family,
                "instance_artifact_sha256": [
                    str(binding["sha256"])
                    for binding in case.artifact_bindings
                ],
            }
            for case in selected_cases
        ],
        "algorithms": [reference_algorithm_id],
        "seeds": list(reference_seeds),
        "budgets": list(evaluation_budgets),
        "metric_contract": metric_contract,
    }
    context_digest = canonical_digest(context)
    if output.exists() and _existing_matches(output, context_digest):
        elapsed = time.perf_counter() - started
        return ReferenceSuiteResult(
            output_directory=output,
            context_path=output / "reference_calibration_precommit.json",
            manifest_path=output / "reference_calibration_completion_evidence.json",
            receipt_path=output / "reference_calibration_completion_receipt.json",
            audit_path=output / "reference_calibration_audit.json",
            case_count=len(selected_cases),
            family_counts=family_counts,
            status="COMPLETE",
            elapsed_seconds=elapsed,
            reused_existing=True,
        )
    if output.exists() and not replace_existing:
        raise FileExistsError(
            f"Reference output exists with different inputs: {output}. "
            "Use replace_existing=True to preserve and supersede it."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    committed = False
    archived_previous: Path | None = None
    try:
        precommit_path = staging / "reference_calibration_precommit.json"
        precommit_sha = write_json(precommit_path, context)
        if precommit_sha != context_digest:
            raise RuntimeError("Canonical precommit digest mismatch.")
        references_dir = staging / "cases"
        runs_dir = staging / "runs"
        case_outputs: list[dict[str, object]] = []
        reference_runs: list[dict[str, object]] = []
        total_evaluations = 0
        total_points = 0
        for case in selected_cases:
            if (
                time_limit_seconds is not None
                and time.perf_counter() - started > time_limit_seconds
            ):
                raise TimeoutError(
                    "Reference calibration exceeded its predeclared time box "
                    f"before {case.case_id!r}."
                )
            if case.packet_path is None or case.packet_sha256 is None:
                raise RuntimeError(f"Case {case.case_id!r} has no bound instance packet.")
            case_archive = ParetoArchive(max_size=None, tol=0.0)
            analytic_ideal, analytic_nadir, analytic_hv_reference, bound_derivation = (
                analytic_metric_box(case.problem)
            )
            case_run_start = len(reference_runs)
            for seed in reference_seeds:
                for budget in evaluation_budgets:
                    result = search_reference_case(
                        case,
                        seeds=(seed,),
                        evaluation_budget=budget,
                        weight_grid_size=weight_grid_size,
                        restart_period=restart_period,
                    )
                    total_evaluations += result.evaluations_used
                    case_archive.update(result.entries)
                    run_key = {
                        "case_id": case.case_id,
                        "algorithm": reference_algorithm_id,
                        "seed": seed,
                        "budget": budget,
                    }
                    safe_algorithm = hashlib.sha256(
                        reference_algorithm_id.encode("utf-8")
                    ).hexdigest()[:12]
                    run_directory = (
                        runs_dir
                        / case.case_id
                        / f"{safe_algorithm}__s{seed}__b{budget}"
                    )
                    witness_payload = {
                        "schema": REFERENCE_WITNESS_SCHEMA,
                        "run_key": run_key,
                        "family": case.family,
                        "problem_sha256": case.problem_sha256,
                        "instance_packet_sha256": case.packet_sha256,
                        "raw_instance_artifact_sha256": [
                            str(binding["sha256"])
                            for binding in case.artifact_bindings
                        ],
                        "reference_calibration_precommit_sha256": precommit_sha,
                        "evaluations_used": result.evaluations_used,
                        "entries": [
                            {
                                "solution": [int(value) for value in entry.tour],
                                "objectives": [
                                    float(value) for value in entry.objectives
                                ],
                            }
                            for entry in result.entries
                        ],
                        "analytic_metric_box": {
                            "ideal": list(result.ideal),
                            "nadir": list(result.nadir),
                            "hv_reference": list(result.hv_reference),
                            "derivation": result.bound_derivation,
                        },
                        "metric_contract": metric_contract,
                        "builder_configuration_sha256": builder_configuration_sha,
                        "claim_boundary": CLAIM_BOUNDARY,
                        "formal_matrix_status": "NOT_RUN",
                    }
                    witness_path = run_directory / "run_witness.json"
                    write_json(witness_path, witness_payload)
                    replay_path = run_directory / "replay_receipt.json"
                    remaining = (
                        None
                        if time_limit_seconds is None
                        else max(
                            0.001,
                            time_limit_seconds - (time.perf_counter() - started),
                        )
                    )
                    replay_receipt = _execute_reference_replay(
                        python_executable=interpreter_path,
                        verifier_path=verifier_path,
                        witness_path=witness_path,
                        packet_path=case.packet_path,
                        receipt_path=replay_path,
                        timeout_seconds=remaining,
                    )
                    if (
                        replay_receipt.get("run_key") != run_key
                        or replay_receipt.get("replayed_entry_count")
                        != len(result.entries)
                        or replay_receipt.get("evaluations_used")
                        != budget
                    ):
                        raise RuntimeError(
                            f"Reference replay result mismatch for {run_key}."
                        )
                    reference_runs.append(
                        {
                            **run_key,
                            "source_artifacts": [
                                {
                                    "role": "all_evaluated_nondominated_witness",
                                    **_final_output_binding(
                                        witness_path,
                                        staging_root=staging,
                                        output_root=output,
                                        source_root=source_root,
                                    ),
                                },
                                {
                                    "role": "independent_replay_receipt",
                                    **_final_output_binding(
                                        replay_path,
                                        staging_root=staging,
                                        output_root=output,
                                        source_root=source_root,
                                    ),
                                },
                            ],
                        }
                    )
            union_entries = tuple(case_archive.entries)
            if not union_entries:
                raise RuntimeError(f"Empty reference union for {case.case_id!r}.")
            _audit_search_entries(
                case,
                union_entries,
                analytic_ideal,
                analytic_nadir,
                analytic_hv_reference,
            )
            constituent_run_keys = [
                {
                    "case_id": str(run["case_id"]),
                    "algorithm": str(run["algorithm"]),
                    "seed": int(run["seed"]),
                    "budget": int(run["budget"]),
                }
                for run in reference_runs[case_run_start:]
            ]
            union_witness_payload = {
                "schema": "ijoc_reference_calibration_case_union_witness_v1",
                "case_id": case.case_id,
                "family": case.family,
                "problem_sha256": case.problem_sha256,
                "instance_packet_sha256": case.packet_sha256,
                "raw_instance_artifact_sha256": [
                    str(binding["sha256"])
                    for binding in case.artifact_bindings
                ],
                "reference_calibration_precommit_sha256": precommit_sha,
                "constituent_run_keys": constituent_run_keys,
                "total_evaluations": sum(
                    int(run_key["budget"]) for run_key in constituent_run_keys
                ),
                "entries": [
                    {
                        "solution": [int(value) for value in entry.tour],
                        "objectives": [
                            float(value) for value in entry.objectives
                        ],
                    }
                    for entry in union_entries
                ],
                "analytic_metric_box": {
                    "ideal": list(analytic_ideal),
                    "nadir": list(analytic_nadir),
                    "hv_reference": list(analytic_hv_reference),
                    "derivation": bound_derivation,
                },
                "metric_contract": metric_contract,
                "builder_configuration_sha256": builder_configuration_sha,
                "claim_boundary": CLAIM_BOUNDARY,
                "formal_matrix_status": "NOT_RUN",
            }
            union_directory = runs_dir / case.case_id / "case_union"
            union_witness_path = union_directory / "union_witness.json"
            write_json(union_witness_path, union_witness_payload)
            union_replay_path = union_directory / "union_replay_receipt.json"
            remaining = (
                None
                if time_limit_seconds is None
                else max(
                    0.001,
                    time_limit_seconds - (time.perf_counter() - started),
                )
            )
            union_receipt = _execute_reference_union_replay(
                python_executable=interpreter_path,
                verifier_path=verifier_path,
                witness_path=union_witness_path,
                packet_path=case.packet_path,
                receipt_path=union_replay_path,
                timeout_seconds=remaining,
            )
            if (
                union_receipt.get("case_id") != case.case_id
                or union_receipt.get("replayed_entry_count") != len(union_entries)
                or union_receipt.get("constituent_run_count")
                != len(constituent_run_keys)
            ):
                raise RuntimeError(
                    f"Reference case-union replay mismatch for {case.case_id!r}."
                )
            reference_runs[case_run_start]["source_artifacts"].extend(
                [
                    {
                        "role": "case_union_nondominated_witness",
                        **_final_output_binding(
                            union_witness_path,
                            staging_root=staging,
                            output_root=output,
                            source_root=source_root,
                        ),
                    },
                    {
                        "role": "case_union_replay_receipt",
                        **_final_output_binding(
                            union_replay_path,
                            staging_root=staging,
                            output_root=output,
                            source_root=source_root,
                        ),
                    },
                ]
            )
            total_points += len(union_entries)
            reference_points = [
                [float(value) for value in entry.objectives]
                for entry in union_entries
            ]
            reference_payload = {
                "schema": REFERENCE_CASE_SCHEMA,
                "case_id": case.case_id,
                "source_role": SOURCE_ROLE,
                "reference_calibration_precommit_sha256": precommit_sha,
                "metric_contract": metric_contract,
                "reference_points": reference_points,
                "ideal": list(analytic_ideal),
                "nadir": list(analytic_nadir),
                "hv_reference": list(analytic_hv_reference),
            }
            reference_path = references_dir / f"{case.case_id}.json"
            write_json(reference_path, reference_payload)
            case_outputs.append(
                {
                    "case_id": case.case_id,
                    **_final_output_binding(
                        reference_path,
                        staging_root=staging,
                        output_root=output,
                        source_root=source_root,
                    ),
                }
            )

        audit = {
            "schema": REFERENCE_AUDIT_SCHEMA,
            "status": "PASS",
            "scope": "reference_calibration_only",
            "checks": {
                "both_problem_families_present": all(
                    count > 0 for count in family_counts.values()
                ),
                "formal_arm_algorithm_overlap_count": 0,
                "formal_seed_overlap_count": 0,
                "tail_selection_case_overlap_count": 0,
                "instance_hashes_recomputed": "PASS",
                "problem_state_hashes_recomputed": "PASS",
                "evaluation_budgets_exact": "PASS",
                "all_reference_witnesses_cold_process_replayed": "PASS",
                "all_case_unions_cold_process_replayed": "PASS",
                "reference_archives_zero_tolerance_nondominated": "PASS",
                "analytic_boxes_cover_reference_witnesses": "PASS",
                "hv_references_strictly_worse_than_reference_witnesses": "PASS",
                "formal_matrix_consumed": False,
            },
            "bindings": {
                "precommit_sha256": precommit_sha,
                "tail_policy_selection_receipt_sha256": tail_receipt_sha,
                "tail_policy_sha256": tail_policy_sha,
                "evaluation_code_sha256": verifier_sha,
                "builder_source_sha256": builder_source_sha,
                "builder_configuration_sha256": builder_configuration_sha,
                "formal_case_manifest_sha256": file_sha256(manifest_path),
                "instance_packet_manifest_sha256": file_sha256(packet_manifest_path),
            },
            "builder_configuration": builder_configuration,
            "reference_algorithm_id": reference_algorithm_id,
            "reference_seeds": list(reference_seeds),
            "excluded_formal_seeds": list(formal_seeds),
            "budgets": list(evaluation_budgets),
            "counts": {
                "cases": len(selected_cases),
                "families": family_counts,
                "runs": len(reference_runs),
                "evaluations": total_evaluations,
                "reference_points": total_points,
            },
            "strict_interpretation": (
                "PASS establishes a byte-bound, independently seeded supplied "
                "reference set and analytic normalization box. It does not "
                "establish a true or complete Pareto front, formal-arm "
                "performance, superiority, scalability, or submission readiness."
            ),
            "claim_boundary": CLAIM_BOUNDARY,
            "formal_matrix_status": "NOT_RUN",
        }
        audit_path = staging / "reference_calibration_audit.json"
        write_json(audit_path, audit)
        reference_manifest = {
            "schema": REFERENCE_MANIFEST_SCHEMA,
            "status": "COMPLETE",
            "reference_calibration_precommit_sha256": precommit_sha,
            "reference_runs": reference_runs,
            "case_outputs": case_outputs,
            "audit_artifact": _binding(audit_path, staging),
        }
        reference_manifest_path = (
            staging / "reference_calibration_completion_evidence.json"
        )
        reference_manifest_sha = write_json(
            reference_manifest_path,
            reference_manifest,
        )
        receipt = {
            "schema": REFERENCE_RECEIPT_SCHEMA,
            "suite_id": context["suite_id"],
            "status": "COMPLETE",
            "evidence_scope": "metric_reference_construction_only",
            "reference_calibration_precommit_sha256": precommit_sha,
            "reference_runs": reference_runs,
            "case_outputs": case_outputs,
            "artifact_manifest": {
                "path": "reference_calibration_completion_evidence.json",
                "sha256": reference_manifest_sha,
            },
        }
        receipt_path = staging / "reference_calibration_completion_receipt.json"
        write_json(receipt_path, receipt)
        if (
            time_limit_seconds is not None
            and time.perf_counter() - started > time_limit_seconds
        ):
            raise TimeoutError(
                "Reference calibration exceeded its time box before commit."
            )
        if output.exists():
            archived_previous = _archive_previous_output(output)
        os.replace(staging, output)
        committed = True
        elapsed = time.perf_counter() - started
        return ReferenceSuiteResult(
            output_directory=output,
            context_path=output / precommit_path.name,
            manifest_path=output / reference_manifest_path.name,
            receipt_path=output / receipt_path.name,
            audit_path=output / audit_path.name,
            case_count=len(selected_cases),
            family_counts=family_counts,
            status="COMPLETE",
            elapsed_seconds=elapsed,
            reused_existing=False,
        )
    except Exception:
        if (
            archived_previous is not None
            and archived_previous.exists()
            and not output.exists()
        ):
            os.replace(archived_previous, output)
        raise
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)


def verify_reference_suite(
    output_directory: str | Path,
    *,
    expected_case_ids: Iterable[str] | None = None,
    instance_packet_manifest_path: str | Path | None = None,
    replay_verifier_path: str | Path | None = None,
    python_executable: str | Path = sys.executable,
    rerun_cold_replay: bool = False,
    artifact_source_root: str | Path | None = None,
) -> Mapping[str, Any]:
    """Independently replay hashes, witnesses, bounds, and receipt linkage."""

    root = Path(output_directory).resolve()
    source_root = (
        Path(artifact_source_root).resolve()
        if artifact_source_root is not None
        else root.parent.resolve()
    )
    context_path = root / "reference_calibration_precommit.json"
    manifest_path = root / "reference_calibration_completion_evidence.json"
    receipt_path = root / "reference_calibration_completion_receipt.json"
    audit_path = root / "reference_calibration_audit.json"
    context = strict_json(context_path)
    manifest = strict_json(manifest_path)
    receipt = strict_json(receipt_path)
    audit = strict_json(audit_path)
    if context.get("schema") != REFERENCE_CONTEXT_SCHEMA:
        raise ValueError("Reference precommit schema mismatch.")
    if manifest.get("schema") != REFERENCE_MANIFEST_SCHEMA:
        raise ValueError("Reference manifest schema mismatch.")
    if (
        receipt.get("schema") != REFERENCE_RECEIPT_SCHEMA
        or receipt.get("status") != "COMPLETE"
    ):
        raise ValueError("Reference suite receipt is incomplete.")
    if audit.get("schema") != REFERENCE_AUDIT_SCHEMA or audit.get("status") != "PASS":
        raise ValueError("Reference suite audit is not PASS.")
    context_sha = file_sha256(context_path)
    manifest_sha = file_sha256(manifest_path)
    if receipt.get("reference_calibration_precommit_sha256") != context_sha:
        raise ValueError("Reference receipt does not bind the precommit.")
    if receipt.get("artifact_manifest") != {
        "path": "reference_calibration_completion_evidence.json",
        "sha256": manifest_sha,
    }:
        raise ValueError("Reference receipt does not bind the manifest.")
    if (
        manifest.get("reference_calibration_precommit_sha256") != context_sha
        or manifest.get("reference_runs") != receipt.get("reference_runs")
        or manifest.get("case_outputs") != receipt.get("case_outputs")
    ):
        raise ValueError("Reference completion evidence disagrees with its receipt.")
    audit_binding = manifest.get("audit_artifact")
    if not isinstance(audit_binding, dict):
        raise ValueError("Reference completion evidence lacks its audit binding.")
    if (
        audit_binding.get("path") != "reference_calibration_audit.json"
        or audit_binding.get("sha256") != file_sha256(audit_path)
        or audit_binding.get("bytes") != audit_path.stat().st_size
    ):
        raise ValueError("Reference audit binding mismatch.")

    raw_context_cases = context.get("cases")
    algorithms = context.get("algorithms")
    seeds = context.get("seeds")
    budgets = context.get("budgets")
    if (
        not isinstance(raw_context_cases, list)
        or not raw_context_cases
        or not isinstance(algorithms, list)
        or not algorithms
        or not isinstance(seeds, list)
        or not seeds
        or not isinstance(budgets, list)
        or not budgets
    ):
        raise ValueError("Reference precommit Cartesian sets are malformed.")
    context_cases = {
        str(item["case_id"]): item
        for item in raw_context_cases
        if isinstance(item, dict)
    }
    if len(context_cases) != len(raw_context_cases):
        raise ValueError("Reference precommit case descriptors are invalid.")
    expected_run_keys = {
        (case_id, algorithm, seed, budget)
        for case_id in context_cases
        for algorithm in algorithms
        for seed in seeds
        for budget in budgets
    }
    raw_runs = receipt.get("reference_runs")
    if not isinstance(raw_runs, list):
        raise ValueError("Reference receipt lacks reference_runs.")
    observed_run_keys: set[tuple[str, str, int, int]] = set()
    seen_paths: set[str] = set()
    entries_by_case: dict[str, list[ArchiveEntry]] = {
        case_id: [] for case_id in context_cases
    }
    union_artifacts_by_case: dict[str, tuple[Path, Path]] = {}
    verifier = (
        Path(replay_verifier_path).resolve()
        if replay_verifier_path is not None
        else None
    )
    if verifier is not None and not verifier.is_file():
        raise ValueError(f"Replay verifier is missing: {verifier}")
    packet_by_case: dict[str, Path] = {}
    if instance_packet_manifest_path is not None:
        packet_manifest_file = Path(instance_packet_manifest_path).resolve()
        packet_manifest = strict_json(packet_manifest_file)
        if packet_manifest.get("schema") != INSTANCE_PACKET_MANIFEST_SCHEMA:
            raise ValueError("Instance packet manifest schema mismatch.")
        for binding in packet_manifest.get("packets", []):
            if not isinstance(binding, dict):
                raise ValueError("Instance packet binding is malformed.")
            packet_path = _safe_relative_path(
                packet_manifest_file.parent,
                binding.get("path"),
                f"{binding.get('case_id')}.packet",
            )
            if file_sha256(packet_path) != binding.get("sha256"):
                raise ValueError("Instance packet hash mismatch during verification.")
            packet_by_case[str(binding["case_id"])] = packet_path
    if rerun_cold_replay and (
        verifier is None or not set(context_cases).issubset(packet_by_case)
    ):
        raise ValueError(
            "Cold replay requires the verifier and an exact instance-packet manifest."
        )

    for index, raw_run in enumerate(raw_runs):
        if not isinstance(raw_run, dict) or set(raw_run) != {
            "case_id",
            "algorithm",
            "seed",
            "budget",
            "source_artifacts",
        }:
            raise ValueError(f"Reference run {index} has an invalid shape.")
        run_key_tuple = (
            str(raw_run["case_id"]),
            str(raw_run["algorithm"]),
            int(raw_run["seed"]),
            int(raw_run["budget"]),
        )
        if run_key_tuple in observed_run_keys:
            raise ValueError("Duplicate reference run key.")
        observed_run_keys.add(run_key_tuple)
        sources = raw_run.get("source_artifacts")
        source_roles = {
            source.get("role") for source in sources if isinstance(source, dict)
        } if isinstance(sources, list) else set()
        required_roles = {
            "all_evaluated_nondominated_witness",
            "independent_replay_receipt",
        }
        union_roles = {
            "case_union_nondominated_witness",
            "case_union_replay_receipt",
        }
        if (
            not isinstance(sources, list)
            or not required_roles.issubset(source_roles)
            or source_roles - required_roles not in (set(), union_roles)
        ):
            raise ValueError("Reference run source roles are incomplete.")
        paths_by_role: dict[str, Path] = {}
        for source in sources:
            if not isinstance(source, dict) or set(source) != {
                "role",
                "path",
                "sha256",
                "bytes",
            }:
                raise ValueError("Reference run source binding is malformed.")
            raw_path = str(source["path"])
            if raw_path in seen_paths:
                raise ValueError("Reference source paths are not globally unique.")
            seen_paths.add(raw_path)
            source_path = _safe_relative_path(
                source_root,
                raw_path,
                f"reference run {index} source",
            )
            if (
                file_sha256(source_path) != source["sha256"]
                or source_path.stat().st_size != source["bytes"]
            ):
                raise ValueError("Reference run source artifact binding mismatch.")
            paths_by_role[str(source["role"])] = source_path
        witness_path = paths_by_role["all_evaluated_nondominated_witness"]
        replay_path = paths_by_role["independent_replay_receipt"]
        witness = strict_json(witness_path)
        replay = strict_json(replay_path)
        run_key = {
            "case_id": run_key_tuple[0],
            "algorithm": run_key_tuple[1],
            "seed": run_key_tuple[2],
            "budget": run_key_tuple[3],
        }
        if (
            witness.get("schema") != REFERENCE_WITNESS_SCHEMA
            or witness.get("run_key") != run_key
            or witness.get("reference_calibration_precommit_sha256")
            != context_sha
            or witness.get("evaluations_used") != run_key_tuple[3]
            or witness.get("metric_contract") != context.get("metric_contract")
        ):
            raise ValueError("Reference run witness binding mismatch.")
        entries = witness.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Reference run witness archive is empty.")
        if (
            replay.get("schema") != "ijoc_reference_replay_receipt_v1"
            or replay.get("status") != "PASS"
            or replay.get("run_key") != run_key
            or replay.get("witness_sha256") != file_sha256(witness_path)
            or replay.get("replayed_entry_count") != len(entries)
            or replay.get("evaluations_used") != run_key_tuple[3]
            or replay.get("evaluation_code_sha256")
            != context["metric_contract"]["evaluation_code_sha256"]
        ):
            raise ValueError("Reference replay receipt binding mismatch.")
        for entry in entries:
            entries_by_case[run_key_tuple[0]].append(
                ArchiveEntry(
                    tuple(int(value) for value in entry["solution"]),
                    tuple(float(value) for value in entry["objectives"]),
                )
            )
        if union_roles.issubset(paths_by_role):
            case_id = run_key_tuple[0]
            if case_id in union_artifacts_by_case:
                raise ValueError("A case has multiple union artifact pairs.")
            union_witness_path = paths_by_role[
                "case_union_nondominated_witness"
            ]
            union_replay_path = paths_by_role["case_union_replay_receipt"]
            union_witness = strict_json(union_witness_path)
            union_replay = strict_json(union_replay_path)
            if (
                union_witness.get("schema")
                != "ijoc_reference_calibration_case_union_witness_v1"
                or union_witness.get("case_id") != case_id
                or union_witness.get(
                    "reference_calibration_precommit_sha256"
                )
                != context_sha
                or union_witness.get("metric_contract")
                != context.get("metric_contract")
                or union_replay.get("schema")
                != "ijoc_reference_union_replay_receipt_v1"
                or union_replay.get("status") != "PASS"
                or union_replay.get("case_id") != case_id
                or union_replay.get("witness_sha256")
                != file_sha256(union_witness_path)
                or union_replay.get("replayed_entry_count")
                != len(union_witness.get("entries", []))
                or union_replay.get("evaluation_code_sha256")
                != context["metric_contract"]["evaluation_code_sha256"]
            ):
                raise ValueError("Reference case-union replay binding mismatch.")
            union_artifacts_by_case[case_id] = (
                union_witness_path,
                union_replay_path,
            )
            if rerun_cold_replay:
                packet_path = packet_by_case[case_id]
                with tempfile.TemporaryDirectory(
                    prefix="ijoc-reference-union-replay-verify-"
                ) as temporary:
                    fresh_path = Path(temporary) / "union_replay_receipt.json"
                    fresh = _execute_reference_union_replay(
                        python_executable=Path(python_executable).resolve(),
                        verifier_path=verifier,
                        witness_path=union_witness_path,
                        packet_path=packet_path,
                        receipt_path=fresh_path,
                        timeout_seconds=None,
                    )
                    if fresh != union_replay:
                        raise ValueError(
                            "Fresh cold case-union replay is not reproducible."
                        )
        if rerun_cold_replay:
            packet_path = packet_by_case[run_key_tuple[0]]
            with tempfile.TemporaryDirectory(
                prefix="ijoc-reference-replay-verify-"
            ) as temporary:
                fresh_path = Path(temporary) / "replay_receipt.json"
                fresh = _execute_reference_replay(
                    python_executable=Path(python_executable).resolve(),
                    verifier_path=verifier,
                    witness_path=witness_path,
                    packet_path=packet_path,
                    receipt_path=fresh_path,
                    timeout_seconds=None,
                )
                if fresh != replay:
                    raise ValueError("Fresh cold replay receipt is not reproducible.")
    if observed_run_keys != expected_run_keys:
        raise ValueError("Reference run matrix is not the exact precommit Cartesian product.")
    if set(union_artifacts_by_case) != set(context_cases):
        raise ValueError("Reference case-union replay artifacts are incomplete.")

    raw_outputs = receipt.get("case_outputs")
    if not isinstance(raw_outputs, list):
        raise ValueError("Reference receipt lacks case_outputs.")
    observed_ids: list[str] = []
    for raw_output in raw_outputs:
        if not isinstance(raw_output, dict) or set(raw_output) != {
            "case_id",
            "path",
            "sha256",
            "bytes",
        }:
            raise ValueError("Reference case output binding is malformed.")
        case_id = str(raw_output["case_id"])
        observed_ids.append(case_id)
        raw_path = str(raw_output["path"])
        if raw_path in seen_paths:
            raise ValueError("Reference case output path is not globally unique.")
        seen_paths.add(raw_path)
        reference_path = _safe_relative_path(
            source_root,
            raw_path,
            f"{case_id}.reference_output",
        )
        if (
            file_sha256(reference_path) != raw_output["sha256"]
            or reference_path.stat().st_size != raw_output["bytes"]
        ):
            raise ValueError(f"{case_id} reference output hash mismatch.")
        reference = strict_json(reference_path)
        if (
            reference.get("schema") != REFERENCE_CASE_SCHEMA
            or reference.get("case_id") != case_id
            or reference.get("source_role") != SOURCE_ROLE
            or reference.get("reference_calibration_precommit_sha256")
            != context_sha
            or reference.get("metric_contract") != context.get("metric_contract")
        ):
            raise ValueError(f"{case_id} reference output binding mismatch.")
        union = ParetoArchive(max_size=None, tol=0.0)
        union.update(entries_by_case[case_id])
        expected_points = [
            [float(value) for value in entry.objectives]
            for entry in union.entries
        ]
        if reference.get("reference_points") != expected_points:
            raise ValueError(f"{case_id} reference points are not the exact run union.")
        union_witness = strict_json(union_artifacts_by_case[case_id][0])
        union_points = [
            entry["objectives"] for entry in union_witness.get("entries", [])
        ]
        if union_points != expected_points:
            raise ValueError(
                f"{case_id} case-union witness is not the exact run union."
            )
        expected_case_run_keys = [
            {
                "case_id": key[0],
                "algorithm": key[1],
                "seed": key[2],
                "budget": key[3],
            }
            for key in sorted(
                key for key in expected_run_keys if key[0] == case_id
            )
        ]
        if union_witness.get("constituent_run_keys") != expected_case_run_keys:
            raise ValueError(
                f"{case_id} case-union constituent runs are incomplete."
            )
        box = union_witness.get("analytic_metric_box")
        if (
            not isinstance(box, dict)
            or reference.get("ideal") != box.get("ideal")
            or reference.get("nadir") != box.get("nadir")
            or reference.get("hv_reference") != box.get("hv_reference")
        ):
            raise ValueError(
                f"{case_id} reference metric box differs from its replayed union."
            )
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("Reference manifest contains duplicate case IDs.")
    if set(observed_ids) != set(context_cases):
        raise ValueError("Reference case outputs do not exactly cover the precommit.")
    if expected_case_ids is not None and set(observed_ids) != set(expected_case_ids):
        raise ValueError("Reference suite case IDs do not match the expected set.")
    return {
        "status": "PASS",
        "case_count": len(observed_ids),
        "run_count": len(observed_run_keys),
        "manifest_sha256": manifest_sha,
        "receipt_sha256": file_sha256(receipt_path),
        "precommit_sha256": context_sha,
        "fresh_cold_replay": "PASS" if rerun_cold_replay else "NOT_RERUN",
        "case_union_replay": "PASS",
        "claim_boundary": CLAIM_BOUNDARY,
        "formal_matrix_status": "NOT_RUN",
    }
