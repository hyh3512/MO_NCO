from __future__ import annotations

"""Prospective common-budget NSGA-II and MOEA/D development adapters.

The adapters in this module are new V21e3 stochastic objects.  They do not
wrap the historical generation-budget implementations: every algorithm step
is driven by the same durable attempt ledger, and only the first objective
call for an exact solution consumes the quality budget.  Duplicate attempts
are recorded at zero charge and use one frozen retry/fallback policy.

This module intentionally authorizes development execution only.  It cannot
materialize selection, calibration, confirmation, or formal evidence.
"""

import argparse
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Literal, Mapping, Sequence, Tuple

from .archive import ArchiveEntry, ParetoArchive, dominates
from .instance import MultiObjectiveTSPInstance
from .moves import order_crossover, sample_two_opt_indices, two_opt, two_opt_at
from .pareto_v21e3r1_construction import family_aware_initial_solution
from .pareto_ijoc_problem import (
    MultiObjectiveCombinatorialProblem,
    MultiObjectiveKnapsackInstance,
    MultiObjectiveTSPProblemAdapter,
    Solution,
    problem_sha256,
)
from .pareto_v21e3_trace import (
    DecisionInput,
    EvaluationContext,
    V21E3RunContext,
    V21E3SQLiteLedger,
)
from .sampler import Diagnostic, OptimizationResult
from .types import ObjectiveVector


V21E3BaselineArm = Literal["NSGAII", "MOEAD"]
V21E3BaselineFamily = Literal["MOTSP", "MOKP"]

_OBJECTIVE_CALL_SEMANTICS = "first_true_objective_evaluation_v1"
_ATTEMPT_HISTORY_SEMANTICS = "all_attempts_terminal_receipt_v1"
_DUPLICATE_POLICY = "exact_solution_cache_zero_charge_retry_then_fallback_v1"
_INITIALIZATION_POLICY = "problem_native_exact_random_solution_v1"
_RETRY_POLICY = "same_family_single_perturbation_v1"
_FALLBACK_POLICY = "problem_native_exact_random_solution_v1"
_ARCHIVE_POLICY = "unbounded_exact_nondominated_all_unique_evaluations_v1"
_CHECKPOINT_POLICY = "genuine_archive_snapshot_on_charged_evaluation_grid_v1"
_ALGORITHM_RNG_POLICY = "repository_single_python_random_seed_stream_v1"
_DUPLICATE_RNG_POLICY = "domain_separated_retry_and_fallback_streams_v1"
_NSGA_MOTSP_SELECTION = (
    "repository_distinct_binary_tournament_rank_crowding_sampled_left_tie_v1"
)
_NSGA_MOKP_SELECTION = (
    "repository_with_replacement_binary_tournament_rank_crowding_stable_index_v1"
)
_NSGA_SURVIVAL = "nondominated_rank_then_crowding_stable_index_v1"
_NSGA_GENERATIONS = "generation_batched_full_then_frozen_partial_survival_v1"
_MOEAD_MOTSP_PARENT_SELECTION = (
    "repository_uniform_one_neighborhood_parent_v1"
)
_MOEAD_MOKP_PARENT_SELECTION = (
    "repository_uniform_two_distinct_neighborhood_parents_v1"
)
_MOEAD_SURVIVAL = "decomposition_replacement_is_survival_v1"
_MOEAD_SCHEDULE = "cyclic_one_subproblem_per_unique_evaluation_v1"
_MOEAD_MOTSP_NEIGHBORHOOD = (
    "repository_squared_euclidean_reference_distance_stable_index_v1"
)
_MOEAD_MOKP_NEIGHBORHOOD = (
    "repository_absolute_first_weight_distance_stable_index_v1"
)
_MOEAD_REPLACEMENT = "bounded_neighborhood_nonworse_tchebycheff_v1"
_MOEAD_REPLACEMENT_ORDER = "neighborhood_order_stop_at_maximum_replacements_v1"


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@lru_cache(maxsize=1)
def _runtime_identity_cached() -> tuple[tuple[str, object], ...]:
    executable = Path(sys.executable).resolve()
    executable_raw = executable.read_bytes()
    payload: dict[str, object] = {
        "schema": "pareto_v21e3_python_runtime_identity_v1",
        "python_implementation": platform.python_implementation(),
        "python_version": sys.version,
        "python_executable": str(executable),
        "python_executable_bytes": len(executable_raw),
        "python_executable_sha256": hashlib.sha256(executable_raw).hexdigest(),
        "platform": platform.platform(),
        "external_algorithm_dependency": (
            "NONE_STDLIB_AND_BOUND_MO_NCO_SOURCE_ONLY"
        ),
    }
    return tuple(payload.items())


def _runtime_identity() -> dict[str, object]:
    # The interpreter binary is immutable during one process.  Hash it once
    # even when a matched matrix executes many case-seed-arm rows.
    return dict(_runtime_identity_cached())


def _development_directions(
    count: int,
    family: V21E3BaselineFamily,
) -> Tuple[Tuple[float, float], ...]:
    if count < 2:
        raise ValueError("At least two reference directions are required.")
    output = []
    for index in range(count):
        first = index / (count - 1)
        second = 1.0 - first
        if family == "MOTSP":
            first = max(1e-3, first)
            second = max(1e-3, second)
            total = first + second
            first, second = first / total, second / total
        output.append((first, second))
    return tuple(output)


@dataclass(frozen=True)
class V21E3BaselineConfig:
    """Fully exposed configuration of one prospective baseline adapter."""

    arm_id: V21E3BaselineArm
    family: V21E3BaselineFamily
    reference_directions: Tuple[Tuple[float, ...], ...]
    charged_evaluations: int
    checkpoint_period: int
    seed: int
    evidence_partition: str = "development"
    population_size: int = 40
    neighborhood_size: int = 8
    maximum_replacements: int = 8
    duplicate_retry_cap: int = 4
    fallback_attempt_cap: int = 16
    initialization_policy: str = _INITIALIZATION_POLICY
    crossover_policy: str = "repository_uniform_bit_crossover_v1"
    mutation_policy: str = "repository_one_over_n_bit_mutation_force_one_v1"
    repair_policy: str = "repository_random_drop_capacity_repair_v1"
    motsp_mutation_probability: float = 0.35
    mokp_mutation_rate_policy: str = "one_over_solution_size_v1"
    selection_policy: str = _NSGA_MOKP_SELECTION
    survival_policy: str = _NSGA_SURVIVAL
    survival_schedule: str = _NSGA_GENERATIONS
    reference_direction_policy: str = (
        "repository_mokp_evenly_spaced_endpoint_including_v1"
    )
    neighborhood_policy: str = _MOEAD_MOKP_NEIGHBORHOOD
    scalarization_policy: str = (
        "repository_raw_dynamic_ideal_tchebycheff_v1"
    )
    scalar_weight_floor: float = 1e-6
    replacement_policy: str = _MOEAD_REPLACEMENT
    replacement_order_policy: str = _MOEAD_REPLACEMENT_ORDER
    duplicate_policy: str = _DUPLICATE_POLICY
    retry_policy: str = _RETRY_POLICY
    fallback_policy: str = _FALLBACK_POLICY
    archive_policy: str = _ARCHIVE_POLICY
    checkpoint_policy: str = _CHECKPOINT_POLICY
    algorithm_rng_policy: str = _ALGORITHM_RNG_POLICY
    duplicate_rng_policy: str = _DUPLICATE_RNG_POLICY
    objective_call_semantics: str = _OBJECTIVE_CALL_SEMANTICS
    attempt_history_semantics: str = _ATTEMPT_HISTORY_SEMANTICS
    trace_database: str | None = None
    terminal_receipt: str | None = None
    receipt_database_path: str | None = None
    capture_trace: bool = True
    case_artifact_sha256: str | None = None
    source_snapshot_sha256: str | None = None
    selection_authorized: bool = False
    formal_authorized: bool = False
    development_diagnostic_id: str | None = None

    def __post_init__(self) -> None:
        if self.arm_id not in {"NSGAII", "MOEAD"}:
            raise ValueError("arm_id must be NSGAII or MOEAD.")
        if self.family not in {"MOTSP", "MOKP"}:
            raise ValueError("family must be MOTSP or MOKP.")
        if self.population_size < 2:
            raise ValueError("population_size must be at least two.")
        if self.charged_evaluations < self.population_size:
            raise ValueError("The charged budget must initialize the population.")
        if (
            self.checkpoint_period <= 0
            or self.charged_evaluations % self.checkpoint_period != 0
        ):
            raise ValueError("checkpoint_period must divide the charged budget.")
        if self.neighborhood_size < 2 or self.neighborhood_size > self.population_size:
            raise ValueError("neighborhood_size must lie in [2, population_size].")
        if self.maximum_replacements < 1 or self.maximum_replacements > self.neighborhood_size:
            raise ValueError("maximum_replacements exceeds the frozen neighborhood.")
        if self.duplicate_retry_cap < 0 or self.fallback_attempt_cap <= 0:
            raise ValueError("Duplicate retry/fallback caps must be finite and valid.")
        if len(self.reference_directions) != self.population_size:
            raise ValueError("One reference direction per population slot is required.")
        for direction in self.reference_directions:
            if len(direction) != 2:
                raise ValueError("V21e3 parity adapters currently require two objectives.")
            if any(not math.isfinite(value) or value < 0.0 for value in direction):
                raise ValueError("Reference weights must be finite and nonnegative.")
            if not any(value > 0.0 for value in direction):
                raise ValueError("Every reference direction must be nonzero.")
            if not math.isclose(sum(direction), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("Reference directions must sum to one.")
        expected_variation = {
            ("NSGAII", "MOTSP"): (
                "repository_fixed_origin_order_crossover_v1",
                "repository_two_opt_with_probability_0_35_v1",
                "motsp_permutation_identity_repair_v1",
            ),
            ("MOEAD", "MOTSP"): (
                "disabled_single_neighborhood_parent_v1",
                "repository_single_two_opt_v1",
                "motsp_permutation_identity_repair_v1",
            ),
            ("NSGAII", "MOKP"): (
                "repository_uniform_bit_crossover_v1",
                "repository_one_over_n_bit_mutation_force_one_v1",
                "repository_random_drop_capacity_repair_v1",
            ),
            ("MOEAD", "MOKP"): (
                "repository_uniform_bit_crossover_v1",
                "repository_one_over_n_bit_mutation_force_one_v1",
                "repository_random_drop_capacity_repair_v1",
            ),
        }[(self.arm_id, self.family)]
        if (
            self.crossover_policy,
            self.mutation_policy,
            self.repair_policy,
        ) != expected_variation:
            raise ValueError("The family-specific repository variation changed.")
        expected_direction_policy = (
            "repository_motsp_positive_floor_1e3_evenly_spaced_v1"
            if self.family == "MOTSP"
            else "repository_mokp_evenly_spaced_endpoint_including_v1"
        )
        if self.reference_direction_policy != expected_direction_policy:
            raise ValueError("The family-specific direction rule changed.")
        expected_floor = 1e-3 if self.family == "MOTSP" else 1e-6
        if self.scalar_weight_floor != expected_floor:
            raise ValueError("The family-specific scalar weight floor changed.")
        expected_neighborhood = (
            _MOEAD_MOTSP_NEIGHBORHOOD
            if self.family == "MOTSP"
            else _MOEAD_MOKP_NEIGHBORHOOD
        )
        if self.neighborhood_policy != expected_neighborhood:
            raise ValueError("The family-specific neighborhood rule changed.")
        if self.motsp_mutation_probability != 0.35:
            raise ValueError("The repository MOTSP mutation probability changed.")
        expected_selection = (
            (
                _NSGA_MOTSP_SELECTION
                if self.family == "MOTSP"
                else _NSGA_MOKP_SELECTION
            )
            if self.arm_id == "NSGAII"
            else (
                _MOEAD_MOTSP_PARENT_SELECTION
                if self.family == "MOTSP"
                else _MOEAD_MOKP_PARENT_SELECTION
            )
        )
        if self.selection_policy != expected_selection:
            raise ValueError("The family-specific parent-selection policy changed.")
        expected_survival = (
            _NSGA_SURVIVAL if self.arm_id == "NSGAII" else _MOEAD_SURVIVAL
        )
        expected_schedule = (
            _NSGA_GENERATIONS if self.arm_id == "NSGAII" else _MOEAD_SCHEDULE
        )
        if (
            self.survival_policy != expected_survival
            or self.survival_schedule != expected_schedule
        ):
            raise ValueError("The arm-specific survival schedule changed.")
        allowed_initialization = {
            _INITIALIZATION_POLICY,
            "family_aware_per_slot_construction_development_diagnostic_v1",
        }
        if self.initialization_policy not in allowed_initialization:
            raise ValueError("Unsupported baseline initialization policy.")
        if self.initialization_policy != _INITIALIZATION_POLICY:
            if not (
                self.evidence_partition == "development"
                and self.development_diagnostic_id is not None
            ):
                raise ValueError(
                    "Matched seeded baselines are development-only diagnostics."
                )
        if self.development_diagnostic_id is not None and self.evidence_partition != "development":
            raise ValueError("Development diagnostics cannot enter later evidence phases.")
        frozen_strings = {
            "scalarization_policy": (
                "repository_raw_dynamic_ideal_tchebycheff_v1"
            ),
            "replacement_policy": _MOEAD_REPLACEMENT,
            "replacement_order_policy": _MOEAD_REPLACEMENT_ORDER,
            "duplicate_policy": _DUPLICATE_POLICY,
            "retry_policy": _RETRY_POLICY,
            "fallback_policy": _FALLBACK_POLICY,
            "archive_policy": _ARCHIVE_POLICY,
            "checkpoint_policy": _CHECKPOINT_POLICY,
            "algorithm_rng_policy": _ALGORITHM_RNG_POLICY,
            "duplicate_rng_policy": _DUPLICATE_RNG_POLICY,
            "objective_call_semantics": _OBJECTIVE_CALL_SEMANTICS,
            "attempt_history_semantics": _ATTEMPT_HISTORY_SEMANTICS,
        }
        for field, expected in frozen_strings.items():
            if getattr(self, field) != expected:
                raise ValueError(f"Unsupported baseline policy for {field}.")
        if self.selection_authorized or self.formal_authorized:
            raise ValueError("V21e3 baseline adapters cannot authorize later evidence.")
        for field in ("case_artifact_sha256", "source_snapshot_sha256"):
            value = getattr(self, field)
            if value is not None and (
                len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{field} must be lowercase SHA-256.")

    def semantic_payload(self) -> dict[str, object]:
        excluded = {
            "trace_database",
            "terminal_receipt",
            "receipt_database_path",
            "capture_trace",
            "case_artifact_sha256",
            "source_snapshot_sha256",
        }
        payload = {
            key: value
            for key, value in asdict(self).items()
            if key not in excluded
        }
        payload["candidate_id"] = self.arm_id
        payload["phase"] = self.evidence_partition
        payload["adaptation_identity"] = self.adaptation_identity
        return payload

    @property
    def adaptation_identity(self) -> str:
        if self.arm_id == "NSGAII":
            return (
                "prospective_generation_batched_first_true_adaptation_of_"
                "repository_baseline_v1"
            )
        return (
            "prospective_steady_state_first_true_adaptation_of_repository_"
            "baseline_v1"
        )


def frozen_development_baseline_configs(
    *,
    family: V21E3BaselineFamily,
    charged_evaluations: int = 2_000,
    checkpoint_period: int = 200,
    seed: int,
    trace_directory: str | Path | None = None,
    initialization_policy: str = _INITIALIZATION_POLICY,
    development_diagnostic_id: str | None = None,
    population_size_override: int | None = None,
) -> dict[V21E3BaselineArm, V21E3BaselineConfig]:
    """Materialize the two frozen development-only parity configurations."""

    if family not in {"MOTSP", "MOKP"}:
        raise ValueError("family must be MOTSP or MOKP.")
    population_size = (
        int(population_size_override)
        if population_size_override is not None
        else (48 if family == "MOTSP" else 40)
    )
    if population_size < 2:
        raise ValueError("population_size_override must be at least two.")
    directions = _development_directions(population_size, family)
    trace_root = None if trace_directory is None else Path(trace_directory)
    output: dict[V21E3BaselineArm, V21E3BaselineConfig] = {}
    for arm_id in ("NSGAII", "MOEAD"):
        slug = "nsga2" if arm_id == "NSGAII" else "moead"
        output[arm_id] = V21E3BaselineConfig(
            arm_id=arm_id,
            family=family,
            reference_directions=directions,
            charged_evaluations=charged_evaluations,
            checkpoint_period=checkpoint_period,
            seed=seed,
            population_size=population_size,
            initialization_policy=initialization_policy,
            development_diagnostic_id=development_diagnostic_id,
            neighborhood_size=8,
            maximum_replacements=8,
            duplicate_retry_cap=4,
            fallback_attempt_cap=16,
            crossover_policy=(
                "repository_fixed_origin_order_crossover_v1"
                if arm_id == "NSGAII" and family == "MOTSP"
                else (
                    "disabled_single_neighborhood_parent_v1"
                    if family == "MOTSP"
                    else "repository_uniform_bit_crossover_v1"
                )
            ),
            mutation_policy=(
                "repository_two_opt_with_probability_0_35_v1"
                if arm_id == "NSGAII" and family == "MOTSP"
                else (
                    "repository_single_two_opt_v1"
                    if family == "MOTSP"
                    else "repository_one_over_n_bit_mutation_force_one_v1"
                )
            ),
            repair_policy=(
                "motsp_permutation_identity_repair_v1"
                if family == "MOTSP"
                else "repository_random_drop_capacity_repair_v1"
            ),
            reference_direction_policy=(
                "repository_motsp_positive_floor_1e3_evenly_spaced_v1"
                if family == "MOTSP"
                else "repository_mokp_evenly_spaced_endpoint_including_v1"
            ),
            scalar_weight_floor=(1e-3 if family == "MOTSP" else 1e-6),
            neighborhood_policy=(
                _MOEAD_MOTSP_NEIGHBORHOOD
                if family == "MOTSP"
                else _MOEAD_MOKP_NEIGHBORHOOD
            ),
            selection_policy=(
                (
                    _NSGA_MOTSP_SELECTION
                    if family == "MOTSP"
                    else _NSGA_MOKP_SELECTION
                )
                if arm_id == "NSGAII"
                else (
                    _MOEAD_MOTSP_PARENT_SELECTION
                    if family == "MOTSP"
                    else _MOEAD_MOKP_PARENT_SELECTION
                )
            ),
            survival_policy=(
                _NSGA_SURVIVAL if arm_id == "NSGAII" else _MOEAD_SURVIVAL
            ),
            survival_schedule=(
                _NSGA_GENERATIONS if arm_id == "NSGAII" else _MOEAD_SCHEDULE
            ),
            trace_database=(
                None
                if trace_root is None
                else str(trace_root / f"{slug}.trace.sqlite3")
            ),
            terminal_receipt=(
                None
                if trace_root is None
                else str(trace_root / f"{slug}.terminal.receipt.json")
            ),
        )
    return output


@dataclass(frozen=True)
class V21E3BaselineAttemptEvent:
    attempt_index: int
    charged_evaluation_index: int | None
    status: str
    cache_hit: bool
    proposal: Solution
    proposal_sha256: str
    operator: str
    retry_ordinal: int
    fallback_used: bool
    population_slot: int


@dataclass(frozen=True)
class V21E3BaselineEvaluationEvent:
    charged_evaluation_index: int
    attempt_index: int
    proposal: Solution
    objectives: ObjectiveVector
    operator: str
    population_slot: int
    accepted_into_population: bool
    population_target_slots: Tuple[int, ...]
    archive_changed: bool
    retained_after_update: bool


@dataclass(frozen=True)
class V21E3BaselineRunResult:
    optimization_result: OptimizationResult
    attempts: Tuple[V21E3BaselineAttemptEvent, ...]
    evaluations: Tuple[V21E3BaselineEvaluationEvent, ...]


@dataclass(frozen=True)
class _PendingNSGAEvaluation:
    outcome: object
    population_slot: int
    operator: str
    archive_changed: bool
    retained_after_update: bool
    archive_size_after: int
    cell_id: str
    new_evaluated_cell: bool
    new_nondominated_cell: bool


def _nondominated_sort(objectives: Sequence[ObjectiveVector]) -> list[list[int]]:
    dominates_set = [[] for _ in objectives]
    domination_count = [0 for _ in objectives]
    first: list[int] = []
    for left in range(len(objectives)):
        for right in range(left + 1, len(objectives)):
            if dominates(objectives[left], objectives[right], tol=0.0):
                dominates_set[left].append(right)
                domination_count[right] += 1
            elif dominates(objectives[right], objectives[left], tol=0.0):
                dominates_set[right].append(left)
                domination_count[left] += 1
        if domination_count[left] == 0:
            first.append(left)
    fronts = [first]
    while fronts[-1]:
        following: list[int] = []
        for left in fronts[-1]:
            for right in dominates_set[left]:
                domination_count[right] -= 1
                if domination_count[right] == 0:
                    following.append(right)
        if not following:
            break
        fronts.append(following)
    return fronts


def _crowding(
    front: Sequence[int],
    objectives: Sequence[ObjectiveVector],
) -> dict[int, float]:
    if not front:
        return {}
    distance = {index: 0.0 for index in front}
    for objective_index in range(len(objectives[0])):
        ordered = sorted(
            front,
            key=lambda index: (objectives[index][objective_index], index),
        )
        distance[ordered[0]] = math.inf
        distance[ordered[-1]] = math.inf
        span = (
            objectives[ordered[-1]][objective_index]
            - objectives[ordered[0]][objective_index]
        )
        if span <= 0.0:
            continue
        for position in range(1, len(ordered) - 1):
            distance[ordered[position]] += (
                objectives[ordered[position + 1]][objective_index]
                - objectives[ordered[position - 1]][objective_index]
            ) / span
    return distance


class _V21E3BaselineEngine:
    def __init__(
        self,
        problem: MultiObjectiveCombinatorialProblem,
        config: V21E3BaselineConfig,
    ) -> None:
        if problem.num_objectives != 2:
            raise ValueError("V21e3 parity adapters currently require two objectives.")
        if not isinstance(
            problem,
            (MultiObjectiveKnapsackInstance, MultiObjectiveTSPProblemAdapter),
        ):
            raise TypeError("Unsupported V21e3 parity problem family.")
        observed_family = (
            "MOKP"
            if isinstance(problem, MultiObjectiveKnapsackInstance)
            else "MOTSP"
        )
        if config.family != observed_family:
            raise ValueError("Baseline configuration names another problem family.")
        self.problem = problem
        self.config = config
        self._rng_algorithm = random.Random(config.seed)
        self._rng_initialization = self._rng_algorithm
        self._rng_variation = self._rng_algorithm
        self._rng_retry = random.Random(self._domain_seed("duplicate_retry"))
        self._rng_fallback = random.Random(self._domain_seed("unique_fallback"))
        semantic_config = config.semantic_payload()
        module_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        context = V21E3RunContext(
            {
                "schema": "v21e3r1_run_context_v2",
                "case_artifact_sha256": (
                    config.case_artifact_sha256 or problem_sha256(problem)
                ),
                "case_artifact_binding_kind": (
                    "explicit_case_artifact_sha256_v1"
                    if config.case_artifact_sha256 is not None
                    else "problem_semantic_sha256_fallback_development_only_v1"
                ),
                "problem_semantic_sha256": problem_sha256(problem),
                "candidate_id": config.arm_id,
                "algorithm_config": semantic_config,
                "candidate_config_sha256": hashlib.sha256(
                    _canonical_bytes(semantic_config)
                ).hexdigest(),
                "algorithm_source_sha256": (
                    config.source_snapshot_sha256 or module_sha
                ),
                "algorithm_source_binding_kind": (
                    "explicit_successor_source_snapshot_sha256_v1"
                    if config.source_snapshot_sha256 is not None
                    else "development_adapter_module_sha256_fallback_pre_snapshot_v1"
                ),
                "reference_directions": config.reference_directions,
                "seed": config.seed,
                "charged_evaluation_budget": config.charged_evaluations,
                "evidence_partition": config.evidence_partition,
            }
        )
        self._run_context = context
        self._ledger = V21E3SQLiteLedger.from_problem(
            problem,
            run_context=context,
            database_path=config.trace_database,
            receipt_path=config.terminal_receipt,
            receipt_database_path=config.receipt_database_path,
        )
        self.archive = ParetoArchive(max_size=None, tol=0.0)
        self._population: list[Solution] = []
        self._objectives: list[ObjectiveVector] = []
        self._attempts: list[V21E3BaselineAttemptEvent] = []
        self._evaluations: list[V21E3BaselineEvaluationEvent] = []
        self._diagnostics: list[Diagnostic] = []
        self._checkpoint_witnesses: list[dict[str, object]] = []
        self._cache_hits = 0
        self._retry_count = 0
        self._fallback_count = 0
        self._accepted_count = 0
        self._operator_call_id = 0
        self._completed_full_generations = 0
        self._partial_generation_offspring = 0
        self._generation_transitions: list[dict[str, object]] = []
        self._start = time.perf_counter()
        self._lower = tuple(float(value) for value in problem.objective_lower_bounds)
        self._upper = tuple(float(value) for value in problem.objective_upper_bounds)
        self._evaluated_cells: set[Tuple[int, ...]] = set()
        self._nondominated_cells: set[Tuple[int, ...]] = set()
        self._neighborhoods = self._build_neighborhoods()
        self._ideal: ObjectiveVector | None = None

    def run(self) -> V21E3BaselineRunResult:
        for slot in range(self.config.population_size):
            proposal, initialization_operator = self._initial_solution(slot)
            outcome, effective_operator = self._attempt_unique(
                proposal,
                population_slot=slot,
                operator=initialization_operator,
                stage="initialization_v21e3_baseline",
                parents=(),
                parent_slots=(),
            )
            self._commit_candidate(
                outcome,
                population_slot=slot,
                operator=effective_operator,
            )
        cursor = 0
        while self._ledger.evaluation_count < self.config.charged_evaluations:
            if self.config.arm_id == "NSGAII":
                self._run_nsga_generation()
            else:
                slot = cursor % self.config.population_size
                cursor += 1
                proposal, parents, parent_slots = self._moead_proposal(slot)
                operator = "moead_neighborhood_family_native_variation_v1"
                outcome, effective_operator = self._attempt_unique(
                    proposal,
                    population_slot=slot,
                    operator=operator,
                    stage="search_v21e3_baseline",
                    parents=parents,
                    parent_slots=parent_slots,
                    operator_witness={
                        "cyclic_update_ordinal": cursor,
                        "subproblem_index": slot,
                        "update_schedule": self.config.survival_schedule,
                    },
                )
                self._commit_candidate(
                    outcome,
                    population_slot=slot,
                    operator=effective_operator,
                )
        receipt = self._ledger.finalize(
            expected_charged_evaluations=self.config.charged_evaluations,
            expected_decisions=self.config.charged_evaluations,
        )
        expected_checkpoints = tuple(
            range(
                self.config.checkpoint_period,
                self.config.charged_evaluations + 1,
                self.config.checkpoint_period,
            )
        )
        observed = tuple(item.iteration for item in self._diagnostics)
        if observed != expected_checkpoints:
            raise RuntimeError("The baseline omitted a genuine common checkpoint.")
        result = OptimizationResult(
            particles=tuple(self._population),
            objectives=tuple(self._objectives),
            archive=self.archive,
            diagnostics=tuple(self._diagnostics),
            metadata={
                "algorithm": f"v21e3-common-budget-{self.config.arm_id.lower()}",
                "arm_id": self.config.arm_id,
                "family": self.config.family,
                "adaptation_identity": self.config.adaptation_identity,
                "repository_baseline_deviation_scope": (
                    "first_true_budget_duplicate_trace_and_cumulative_measurement_archive_seam_only_v1"
                ),
                "algorithm_config": self.config.semantic_payload(),
                "common_budget_adapter_status": "DEVELOPMENT_ONLY_AVAILABLE",
                "objective_call_semantics": self.config.objective_call_semantics,
                "attempt_history_semantics": self.config.attempt_history_semantics,
                "charged_evaluation_budget": self.config.charged_evaluations,
                "charged_evaluation_count": self._ledger.evaluation_count,
                "unique_true_evaluation_count": self._ledger.evaluation_count,
                "physical_objective_call_count": self._ledger.physical_call_count,
                "attempt_count": self._ledger.attempt_count,
                "cache_hit_count": self._cache_hits,
                "retry_count": self._retry_count,
                "fallback_count": self._fallback_count,
                "completed_full_generations": self._completed_full_generations,
                "partial_generation_offspring": self._partial_generation_offspring,
                "generation_survival_transitions": tuple(
                    self._generation_transitions
                ),
                "exact_charged_budget_gate": "PASS",
                "expected_anytime_checkpoints": expected_checkpoints,
                "observed_anytime_checkpoints": observed,
                "checkpoint_solution_witnesses": tuple(self._checkpoint_witnesses),
                "trace_receipt": receipt,
                "run_context": self._run_context.payload,
                "runtime_identity": _runtime_identity(),
                "selection_authorized": False,
                "formal_authorized": False,
                "scientific_scope": "engineering_preflight_not_performance_evidence",
            },
        )
        return V21E3BaselineRunResult(
            optimization_result=result,
            attempts=(tuple(self._attempts) if self.config.capture_trace else ()),
            evaluations=(
                tuple(self._evaluations) if self.config.capture_trace else ()
            ),
        )

    def _initial_solution(self, slot: int) -> tuple[Solution, str]:
        if self.config.initialization_policy == _INITIALIZATION_POLICY:
            return (
                self.problem.random_solution(self._rng_initialization),
                "problem_native_exact_random_initialization_v1",
            )
        direction = self.config.reference_directions[slot]
        return family_aware_initial_solution(self.problem, direction, slot)

    def _attempt_unique(
        self,
        proposal: Solution,
        *,
        population_slot: int,
        operator: str,
        stage: str,
        parents: Tuple[Solution, ...],
        parent_slots: Tuple[int, ...],
        operator_witness: Mapping[str, object] | None = None,
    ) -> tuple[object, str]:
        current = tuple(proposal)
        current_operator = operator
        current_parents = parents
        current_parent_slots = parent_slots
        total_limit = self.config.duplicate_retry_cap + self.config.fallback_attempt_cap
        for ordinal in range(total_limit + 1):
            fallback = ordinal > self.config.duplicate_retry_cap
            if ordinal > 0:
                if fallback:
                    self._fallback_count += 1
                    current = self.problem.random_solution(self._rng_fallback)
                    current_operator = "frozen_unique_fallback_v21e3"
                    current_parents = ()
                    current_parent_slots = ()
                else:
                    self._retry_count += 1
                    retry_parent = current
                    current = self._retry_candidate(current)
                    current_operator = "duplicate_retry_perturbation_v21e3"
                    current_parents = (retry_parent,)
                    current_parent_slots = (population_slot,)
            self._operator_call_id += 1
            outcome = self._ledger.attempt(
                current,
                EvaluationContext(
                    evidence_partition=self.config.evidence_partition,
                    search_phase_id="common_budget_baseline_v21e3",
                    stage_id=stage,
                    type_id=population_slot,
                    operator_id=current_operator,
                    operator_call_id=self._operator_call_id,
                    parent_solutions=current_parents,
                    parent_type_ids=current_parent_slots,
                    repair_applied=(
                        isinstance(self.problem, MultiObjectiveKnapsackInstance)
                        and current_operator
                        not in {
                            "problem_native_exact_random_initialization_v1",
                            "frozen_unique_fallback_v21e3",
                        }
                    ),
                    repair_operator_id=(
                        "repository_random_drop_capacity_repair_v1"
                        if isinstance(self.problem, MultiObjectiveKnapsackInstance)
                        else None
                    ),
                    operator_witness={
                        "arm_id": self.config.arm_id,
                        "family": self.config.family,
                        "population_slot": population_slot,
                        "retry_ordinal": ordinal,
                        "fallback_used": fallback,
                        **({} if operator_witness is None else operator_witness),
                    },
                ),
            )
            self._attempts.append(
                V21E3BaselineAttemptEvent(
                    attempt_index=outcome.attempt_index,
                    charged_evaluation_index=outcome.charged_evaluation_index,
                    status=outcome.status,
                    cache_hit=outcome.cache_hit,
                    proposal=outcome.proposal,
                    proposal_sha256=outcome.proposal_sha256,
                    operator=current_operator,
                    retry_ordinal=ordinal,
                    fallback_used=fallback,
                    population_slot=population_slot,
                )
            )
            if outcome.cache_hit:
                self._cache_hits += 1
                continue
            return outcome, current_operator
        try:
            self._ledger.finalize(
                expected_charged_evaluations=self.config.charged_evaluations,
                expected_decisions=self.config.charged_evaluations,
            )
        except RuntimeError as error:
            raise RuntimeError(
                "V21e3 baseline retry/fallback cap exhausted; the ledger wrote "
                "a terminal FAILURE receipt."
            ) from error
        raise RuntimeError("Baseline retry/fallback exhaustion did not fail closed.")

    def _run_nsga_generation(self) -> None:
        remaining = (
            self.config.charged_evaluations - self._ledger.evaluation_count
        )
        batch_size = min(self.config.population_size, remaining)
        ranks, crowding = self._rank_and_crowding(self._objectives)
        pending: list[_PendingNSGAEvaluation] = []
        first_evaluation = self._ledger.evaluation_count + 1
        for offspring_position in range(batch_size):
            slot, proposal, parents, parent_slots = self._nsga_proposal(
                ranks,
                crowding,
            )
            outcome, effective_operator = self._attempt_unique(
                proposal,
                population_slot=slot,
                operator="generation_batched_nsga2_family_native_variation_v1",
                stage="generation_offspring_v21e3_baseline",
                parents=parents,
                parent_slots=parent_slots,
                operator_witness={
                    "generation_index": len(self._generation_transitions) + 1,
                    "offspring_position": offspring_position,
                    "planned_batch_size": batch_size,
                    "survival_schedule": self.config.survival_schedule,
                },
            )
            pending.append(
                self._record_pending_nsga(
                    outcome,
                    population_slot=slot,
                    operator=effective_operator,
                )
            )
        selected_offspring_count = self._finish_nsga_generation(pending)
        full_generation = batch_size == self.config.population_size
        self._generation_transitions.append(
            {
                "generation_index": len(self._generation_transitions) + 1,
                "first_charged_evaluation": first_evaluation,
                "last_charged_evaluation": self._ledger.evaluation_count,
                "offspring_count": batch_size,
                "selected_offspring_count": selected_offspring_count,
                "survival_kind": (
                    "FULL_GENERATION"
                    if full_generation
                    else "FROZEN_PARTIAL_GENERATION"
                ),
                "parent_population_frozen_during_batch": True,
            }
        )
        if full_generation:
            self._completed_full_generations += 1
        else:
            self._partial_generation_offspring = batch_size

    def _record_pending_nsga(
        self,
        outcome: object,
        *,
        population_slot: int,
        operator: str,
    ) -> _PendingNSGAEvaluation:
        proposal = tuple(outcome.proposal)
        objective = tuple(float(value) for value in outcome.objectives)
        entry = ArchiveEntry(proposal, objective)
        archive_changed = self.archive.update((entry,))
        retained = self.archive.contains(entry)
        cell_id, new_evaluated_cell, new_nondominated_cell = self._observe_cell(
            objective
        )
        self._checkpoint_if_due(outcome.charged_evaluation_index)
        return _PendingNSGAEvaluation(
            outcome=outcome,
            population_slot=population_slot,
            operator=operator,
            archive_changed=archive_changed,
            retained_after_update=retained,
            archive_size_after=len(self.archive),
            cell_id=cell_id,
            new_evaluated_cell=new_evaluated_cell,
            new_nondominated_cell=new_nondominated_cell,
        )

    def _finish_nsga_generation(
        self,
        pending: Sequence[_PendingNSGAEvaluation],
    ) -> int:
        if not pending:
            raise RuntimeError("NSGA-II cannot finish an empty generation.")
        offspring_objectives = [
            tuple(float(value) for value in item.outcome.objectives)
            for item in pending
        ]
        combined_objectives = self._objectives + offspring_objectives
        chosen: list[int] = []
        for front in _nondominated_sort(combined_objectives):
            if len(chosen) + len(front) <= self.config.population_size:
                chosen.extend(front)
                continue
            crowding = _crowding(front, combined_objectives)
            chosen.extend(
                sorted(front, key=lambda index: (-crowding[index], index))[
                    : self.config.population_size - len(chosen)
                ]
            )
            break
        selected_children = [
            index - self.config.population_size
            for index in chosen
            if index >= self.config.population_size
        ]
        rejected_parent_slots = sorted(
            set(range(self.config.population_size))
            - {index for index in chosen if index < self.config.population_size}
        )
        if len(selected_children) != len(rejected_parent_slots):
            raise RuntimeError("Generation survival has inconsistent replacement sets.")
        target_by_child = {
            combined_index - self.config.population_size: target_slot
            for target_slot, combined_index in enumerate(chosen)
            if combined_index >= self.config.population_size
        }
        for child_index, item in enumerate(pending):
            outcome = item.outcome
            target = target_by_child.get(child_index)
            targets = () if target is None else (target,)
            accepted = target is not None
            if accepted:
                self._accepted_count += 1
            self._ledger.commit_decision(
                outcome.charged_evaluation_index,
                DecisionInput(
                    accepted_into_population=accepted,
                    population_replacement_count=len(targets),
                    population_target_type_ids=targets,
                    decision_reason=(
                        "generation_rank_crowding_replacement"
                        if accepted
                        else "generation_rank_crowding_rejection"
                    ),
                    archive_changed=item.archive_changed,
                    retained_after_update=item.retained_after_update,
                    archive_size_after=item.archive_size_after,
                    scalarization_id=None,
                    scalar_parent=None,
                    scalar_candidate=None,
                    scalar_advantage=None,
                    cell_id=item.cell_id,
                    new_evaluated_cell=item.new_evaluated_cell,
                    new_nondominated_cell=item.new_nondominated_cell,
                ),
            )
            self._evaluations.append(
                V21E3BaselineEvaluationEvent(
                    charged_evaluation_index=outcome.charged_evaluation_index,
                    attempt_index=outcome.attempt_index,
                    proposal=tuple(outcome.proposal),
                    objectives=tuple(float(value) for value in outcome.objectives),
                    operator=item.operator,
                    population_slot=item.population_slot,
                    accepted_into_population=accepted,
                    population_target_slots=targets,
                    archive_changed=item.archive_changed,
                    retained_after_update=item.retained_after_update,
                )
            )
        combined_solutions = self._population + [
            tuple(item.outcome.proposal) for item in pending
        ]
        self._population = [combined_solutions[index] for index in chosen]
        self._objectives = [combined_objectives[index] for index in chosen]
        return len(selected_children)

    def _commit_candidate(
        self,
        outcome: object,
        *,
        population_slot: int,
        operator: str,
    ) -> V21E3BaselineEvaluationEvent:
        proposal = tuple(outcome.proposal)
        objective = tuple(float(value) for value in outcome.objectives)
        entry = ArchiveEntry(proposal, objective)
        archive_changed = self.archive.update((entry,))
        retained = self.archive.contains(entry)
        if len(self._population) < self.config.population_size:
            target_slots = (len(self._population),)
            self._population.append(proposal)
            self._objectives.append(objective)
            accepted = True
            decision_reason = "initial_population_fill"
            scalar_parent = None
            scalar_candidate = (
                self._scalar(
                    objective,
                    self.config.reference_directions[population_slot],
                )
                if self.config.arm_id == "MOEAD"
                else None
            )
        elif self.config.arm_id == "NSGAII":
            raise RuntimeError(
                "NSGA-II search decisions must be committed at generation survival."
            )
        else:
            target_slots, scalar_parent, scalar_candidate = self._moead_targets(
                population_slot,
                objective,
            )
            accepted = bool(target_slots)
            for target in target_slots:
                self._population[target] = proposal
                self._objectives[target] = objective
            decision_reason = (
                "bounded_neighborhood_nonworse_replacement"
                if accepted
                else "worse_neighborhood_rejection"
            )
        if accepted:
            self._accepted_count += 1
        cell_id, new_evaluated_cell, new_nondominated_cell = self._observe_cell(objective)
        self._ledger.commit_decision(
            outcome.charged_evaluation_index,
            DecisionInput(
                accepted_into_population=accepted,
                population_replacement_count=len(target_slots),
                population_target_type_ids=target_slots,
                decision_reason=decision_reason,
                archive_changed=archive_changed,
                retained_after_update=retained,
                archive_size_after=len(self.archive),
                scalarization_id=(
                    self.config.scalarization_policy
                    if self.config.arm_id == "MOEAD"
                    else None
                ),
                scalar_parent=scalar_parent,
                scalar_candidate=scalar_candidate,
                scalar_advantage=(
                    None
                    if scalar_parent is None or scalar_candidate is None
                    else scalar_parent - scalar_candidate
                ),
                cell_id=cell_id,
                new_evaluated_cell=new_evaluated_cell,
                new_nondominated_cell=new_nondominated_cell,
            ),
        )
        event = V21E3BaselineEvaluationEvent(
            charged_evaluation_index=outcome.charged_evaluation_index,
            attempt_index=outcome.attempt_index,
            proposal=proposal,
            objectives=objective,
            operator=operator,
            population_slot=population_slot,
            accepted_into_population=accepted,
            population_target_slots=target_slots,
            archive_changed=archive_changed,
            retained_after_update=retained,
        )
        self._evaluations.append(event)
        self._checkpoint_if_due(outcome.charged_evaluation_index)
        return event

    def _nsga_proposal(
        self,
        ranks: Mapping[int, int],
        crowding: Mapping[int, float],
    ) -> tuple[int, Solution, Tuple[Solution, ...], Tuple[int, ...]]:
        left_slot = self._tournament(ranks, crowding)
        right_slot = self._tournament(ranks, crowding)
        left = self._population[left_slot]
        right = self._population[right_slot]
        child = self._variation(left, right)
        return left_slot, child, (left, right), (left_slot, right_slot)

    def _moead_proposal(
        self,
        slot: int,
    ) -> tuple[Solution, Tuple[Solution, ...], Tuple[int, ...]]:
        neighborhood = self._neighborhoods[slot]
        if isinstance(self.problem, MultiObjectiveTSPProblemAdapter):
            parent_slot = self._rng_variation.choice(neighborhood)
            parent = self._population[parent_slot]
            return (
                two_opt(parent, self._rng_variation),
                (parent,),
                (parent_slot,),
            )
        left_slot, right_slot = self._rng_variation.sample(list(neighborhood), 2)
        left = self._population[left_slot]
        right = self._population[right_slot]
        return self._variation(left, right), (left, right), (left_slot, right_slot)

    def _variation(self, left: Solution, right: Solution) -> Solution:
        if isinstance(self.problem, MultiObjectiveKnapsackInstance):
            child = [
                left[index] if self._rng_variation.random() < 0.5 else right[index]
                for index in range(self.problem.solution_size)
            ]
            crossed = self._repair_mokp(child)
            child = list(crossed)
            mutated = False
            rate = 1.0 / self.problem.solution_size
            for index in range(self.problem.solution_size):
                if self._rng_variation.random() < rate:
                    child[index] ^= 1
                    mutated = True
            if not mutated:
                child[self._rng_variation.randrange(self.problem.solution_size)] ^= 1
            return self._repair_mokp(child)
        child = order_crossover(left, right, self._rng_variation)
        if self._rng_variation.random() < self.config.motsp_mutation_probability:
            child = two_opt(child, self._rng_variation)
        return child

    def _retry_candidate(self, proposal: Solution) -> Solution:
        if isinstance(self.problem, MultiObjectiveKnapsackInstance):
            child = list(proposal)
            child[self._rng_retry.randrange(self.problem.solution_size)] ^= 1
            return self._repair_mokp(child, rng=self._rng_retry)
        i, j = sample_two_opt_indices(self.problem.solution_size, self._rng_retry)
        return two_opt_at(proposal, i, j)

    def _repair_mokp(
        self,
        values: Sequence[int],
        *,
        rng: random.Random | None = None,
    ) -> Solution:
        problem = self.problem
        if not isinstance(problem, MultiObjectiveKnapsackInstance):
            raise TypeError("MOKP repair used for another family.")
        selected = [index for index, value in enumerate(values) if value]
        active_rng = self._rng_variation if rng is None else rng
        active_rng.shuffle(selected)
        child = [int(value) for value in values]
        weight = sum(
            problem.item_weights[index]
            for index, value in enumerate(child)
            if value
        )
        for index in selected:
            if weight <= problem.capacity:
                break
            child[index] = 0
            weight -= problem.item_weights[index]
        result = tuple(child)
        problem.validate_solution(result)
        return result

    def _rank_and_crowding(
        self,
        objectives: Sequence[ObjectiveVector],
    ) -> tuple[dict[int, int], dict[int, float]]:
        fronts = _nondominated_sort(objectives)
        ranks = {
            index: rank
            for rank, front in enumerate(fronts)
            for index in front
        }
        crowding: dict[int, float] = {}
        for front in fronts:
            crowding.update(_crowding(front, objectives))
        return ranks, crowding

    def _tournament(
        self,
        ranks: Mapping[int, int],
        crowding: Mapping[int, float],
    ) -> int:
        if self.config.family == "MOTSP":
            left, right = self._rng_variation.sample(
                range(self.config.population_size),
                2,
            )
            if ranks[left] < ranks[right]:
                return left
            if ranks[right] < ranks[left]:
                return right
            return left if crowding[left] >= crowding[right] else right
        left = self._rng_variation.randrange(self.config.population_size)
        right = self._rng_variation.randrange(self.config.population_size)
        return min(
            (left, right),
            key=lambda index: (ranks[index], -crowding[index], index),
        )

    def _moead_targets(
        self,
        slot: int,
        objective: ObjectiveVector,
    ) -> tuple[Tuple[int, ...], float, float]:
        if self._ideal is None:
            self._ideal = tuple(
                min(row[index] for row in self._objectives)
                for index in range(2)
            )
        old_ideal = self._ideal
        self._ideal = tuple(min(old_ideal[index], objective[index]) for index in range(2))
        direction = self.config.reference_directions[slot]
        scalar_candidate = self._scalar(objective, direction, ideal=self._ideal)
        scalar_parent = self._scalar(self._objectives[slot], direction, ideal=self._ideal)
        targets = []
        for target in self._neighborhoods[slot]:
            weight = self.config.reference_directions[target]
            if self._scalar(objective, weight, ideal=self._ideal) <= self._scalar(
                self._objectives[target],
                weight,
                ideal=self._ideal,
            ):
                targets.append(target)
                if len(targets) == self.config.maximum_replacements:
                    break
        return tuple(targets), scalar_parent, scalar_candidate

    def _scalar(
        self,
        objective: Sequence[float],
        direction: Sequence[float],
        *,
        ideal: Sequence[float] | None = None,
    ) -> float:
        anchor = self._lower if ideal is None else ideal
        return max(
            max(weight, self.config.scalar_weight_floor)
            * abs(float(value) - float(anchor_value))
            for value, anchor_value, weight in zip(
                objective,
                anchor,
                direction,
            )
        )

    def _build_neighborhoods(self) -> Tuple[Tuple[int, ...], ...]:
        directions = self.config.reference_directions
        def distance(index: int, other: int) -> float:
            if self.config.family == "MOKP":
                return abs(directions[index][0] - directions[other][0])
            return sum(
                (left - right) ** 2
                for left, right in zip(directions[index], directions[other])
            )
        return tuple(
            tuple(
                sorted(
                    range(len(directions)),
                    key=lambda other: (
                        distance(index, other),
                        other,
                    ),
                )[: self.config.neighborhood_size]
            )
            for index in range(len(directions))
        )

    def _observe_cell(self, objective: ObjectiveVector) -> tuple[str, bool, bool]:
        bins = 20
        region = tuple(
            min(
                bins - 1,
                int(bins * (value - lower) / (upper - lower)),
            )
            for value, lower, upper in zip(objective, self._lower, self._upper)
        )
        new_evaluated = region not in self._evaluated_cells
        nondominated = {
            tuple(
                min(
                    bins - 1,
                    int(bins * (value - lower) / (upper - lower)),
                )
                for value, lower, upper in zip(entry.objectives, self._lower, self._upper)
            )
            for entry in self.archive.entries
        }
        new_nondominated = region in nondominated and region not in self._nondominated_cells
        self._evaluated_cells.add(region)
        self._nondominated_cells = nondominated
        return ":".join(str(value) for value in region), new_evaluated, new_nondominated

    def _checkpoint_if_due(self, evaluation_index: int) -> None:
        if evaluation_index % self.config.checkpoint_period != 0:
            return
        self._diagnostics.append(
            Diagnostic(
                iteration=evaluation_index,
                temperature=0.0,
                acceptance_rate=self._accepted_count / evaluation_index,
                archive_size=len(self.archive),
                hypervolume_2d=self.archive.hypervolume_2d(reference=self._upper),
                empirical_energy=0.0,
                positive_archive_jump=0.0,
                front=tuple(entry.objectives for entry in self.archive.entries),
                elapsed_seconds=time.perf_counter() - self._start,
            )
        )
        self._checkpoint_witnesses.append(
            {
                "evaluation": evaluation_index,
                "entries": tuple(
                    {
                        "solution": entry.tour,
                        "objectives": entry.objectives,
                    }
                    for entry in self.archive.entries
                ),
            }
        )

    def _domain_seed(self, domain: str) -> int:
        payload = (
            f"v21e3-baseline|{self.config.arm_id}|{self.config.seed}|{domain}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def run_v21e3_development_baseline(
    problem: MultiObjectiveCombinatorialProblem,
    config: V21E3BaselineConfig,
) -> V21E3BaselineRunResult:
    """Run one adapter while preserving all later-phase fail-closed gates."""

    if config.evidence_partition != "development":
        raise ValueError(
            "V21e3 common-budget baselines are development-only; selection, "
            "calibration, confirmation, and formal execution remain prohibited."
        )
    return _V21E3BaselineEngine(problem, config).run()


def load_v21e3_development_problem(
    path: str | Path,
) -> MultiObjectiveCombinatorialProblem:
    """Load one canonical V21e3 development instance without hidden sources."""

    instance_path = Path(path).resolve()
    payload = json.loads(instance_path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    case_id = str(payload.get("case_id", instance_path.stem))
    if schema == "pareto_v21_mokp_integer_instance_v1":
        return MultiObjectiveKnapsackInstance(
            item_weights=tuple(int(value) for value in payload["item_weights"]),
            profits_by_objective=tuple(
                tuple(int(value) for value in row)
                for row in payload["profits_by_objective"]
            ),
            capacity=int(payload["capacity"]),
            name=case_id,
        )
    if schema == "pareto_v21_motsp_integer_coordinates_v1":
        coordinates = tuple(
            tuple((float(point[0]), float(point[1])) for point in rows)
            for rows in payload["coordinates_by_objective"]
        )
        return MultiObjectiveTSPProblemAdapter(
            MultiObjectiveTSPInstance(
                coords_by_objective=coordinates,
                name=case_id,
            )
        )
    raise ValueError(f"Unsupported V21e3 instance schema: {schema!r}.")


def _exclusive_write_json(path: Path, payload: object) -> None:
    raw = _canonical_bytes(payload) + b"\n"
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one V21e3 development-only common-budget baseline."
    )
    parser.add_argument("--arm", choices=("NSGAII", "MOEAD"), required=True)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=2_000)
    parser.add_argument("--checkpoint-period", type=int, default=200)
    parser.add_argument("--source-snapshot-sha256")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    instance_path = args.instance.resolve()
    instance_raw = instance_path.read_bytes()
    problem = load_v21e3_development_problem(instance_path)
    configs = frozen_development_baseline_configs(
        family=(
            "MOKP"
            if isinstance(problem, MultiObjectiveKnapsackInstance)
            else "MOTSP"
        ),
        charged_evaluations=args.budget,
        checkpoint_period=args.checkpoint_period,
        seed=args.seed,
        trace_directory=output_directory,
    )
    config = replace(
        configs[args.arm],
        case_artifact_sha256=hashlib.sha256(instance_raw).hexdigest(),
        source_snapshot_sha256=args.source_snapshot_sha256,
    )
    run = run_v21e3_development_baseline(problem, config)
    result = run.optimization_result
    payload = {
        "schema": "pareto_v21e3_development_baseline_result_v1",
        "status": "SUCCESS_ENGINEERING_ONLY",
        "scientific_scope": "engineering_preflight_not_performance_evidence",
        "arm_id": args.arm,
        "family": config.family,
        "adaptation_identity": config.adaptation_identity,
        "repository_baseline_deviation_scope": result.metadata[
            "repository_baseline_deviation_scope"
        ],
        "case_id": problem.name,
        "case_artifact": {
            "path": str(instance_path),
            "bytes": len(instance_raw),
            "sha256": hashlib.sha256(instance_raw).hexdigest(),
        },
        "problem_semantic_sha256": problem_sha256(problem),
        "algorithm_config": config.semantic_payload(),
        "charged_evaluation_count": result.metadata["charged_evaluation_count"],
        "physical_objective_call_count": result.metadata[
            "physical_objective_call_count"
        ],
        "attempt_count": result.metadata["attempt_count"],
        "cache_hit_count": result.metadata["cache_hit_count"],
        "retry_count": result.metadata["retry_count"],
        "fallback_count": result.metadata["fallback_count"],
        "completed_full_generations": result.metadata[
            "completed_full_generations"
        ],
        "partial_generation_offspring": result.metadata[
            "partial_generation_offspring"
        ],
        "generation_survival_transitions": result.metadata[
            "generation_survival_transitions"
        ],
        "observed_anytime_checkpoints": result.metadata[
            "observed_anytime_checkpoints"
        ],
        "archive": [
            {
                "solution": entry.tour,
                "objectives": entry.objectives,
            }
            for entry in result.archive.entries
        ],
        "trace_receipt": result.metadata["trace_receipt"],
        "run_context": result.metadata["run_context"],
        "runtime_identity": result.metadata["runtime_identity"],
        "selection_authorized": False,
        "formal_authorized": False,
    }
    result_path = output_directory / "result.json"
    _exclusive_write_json(result_path, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "arm_id": args.arm,
                "result": str(result_path),
                "charged_evaluation_count": payload["charged_evaluation_count"],
                "selection_authorized": False,
                "formal_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "V21E3BaselineArm",
    "V21E3BaselineAttemptEvent",
    "V21E3BaselineConfig",
    "V21E3BaselineEvaluationEvent",
    "V21E3BaselineRunResult",
    "frozen_development_baseline_configs",
    "load_v21e3_development_problem",
    "run_v21e3_development_baseline",
]


if __name__ == "__main__":
    raise SystemExit(main())
