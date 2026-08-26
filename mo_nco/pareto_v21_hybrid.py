from __future__ import annotations

"""Trace-complete development engine for V21 Typed Hybrid Pareto Search.

V21 is a new heuristic object.  It does not inherit a convergence or
competitiveness claim from the V20 typed Pareto-SMC implementation.  The first
public contract provided here is deliberately narrower: every true objective
evaluation has exactly one trace event, and the reported search archive is the
exact nondominated reduction of those events.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import random
import time
from typing import Literal, Sequence, Tuple

from .archive import ArchiveEntry, ParetoArchive
from .pareto_ijoc_problem import (
    MultiObjectiveCombinatorialProblem,
    MultiObjectiveKnapsackInstance,
    MultiObjectiveTSPProblemAdapter,
    Solution,
)
from .moves import two_opt_at
from .sampler import Diagnostic, OptimizationResult
from .pareto_v21_trace import (
    DecisionInput,
    EvaluationContext,
    SQLiteEvaluationLedger,
)
from .types import ObjectiveVector


CandidateId = Literal["C0", "C1", "C2", "C3", "C4"]
EvidencePhase = Literal["development", "calibration", "formal_confirmation"]


class _Exp3OperatorAllocator:
    def __init__(self, operators: Sequence[str], exploration: float) -> None:
        self.operators = tuple(str(value) for value in operators)
        self.exploration = float(exploration)
        self.weights = [1.0] * len(self.operators)

    def probabilities(self) -> tuple[float, ...]:
        total = sum(self.weights)
        count = len(self.weights)
        return tuple(
            (1.0 - self.exploration) * weight / total
            + self.exploration / count
            for weight in self.weights
        )

    def choose(self, rng: random.Random) -> tuple[int, tuple[float, ...]]:
        probabilities = self.probabilities()
        draw = rng.random()
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if draw <= cumulative:
                return index, probabilities
        return len(probabilities) - 1, probabilities

    def update(self, index: int, reward: float, probability: float) -> None:
        count = len(self.weights)
        estimated = float(reward) / max(float(probability), 1e-15)
        self.weights[index] *= math.exp(
            self.exploration * estimated / count
        )


@dataclass(frozen=True)
class V21HybridConfig:
    candidate_id: CandidateId
    reference_directions: Tuple[Tuple[float, ...], ...]
    evaluations: int
    checkpoint_period: int
    seed: int
    phase: EvidencePhase
    trace_database: str | None = None
    capture_trace: bool = True
    diversification_period: int = 16
    exchange_period: int = 11
    operator_exploration: float = 0.15
    neighborhood_size: int = 4

    def __post_init__(self) -> None:
        if self.candidate_id not in {"C0", "C1", "C2", "C3", "C4"}:
            raise ValueError("candidate_id must be one of C0, C1, C2, C3, C4.")
        if not self.reference_directions:
            raise ValueError("At least one reference direction is required.")
        dimension = len(self.reference_directions[0])
        if dimension <= 0:
            raise ValueError("Reference directions must be nonempty.")
        for direction in self.reference_directions:
            if len(direction) != dimension:
                raise ValueError("Reference directions must share a dimension.")
            if any(not math.isfinite(value) or value <= 0.0 for value in direction):
                raise ValueError("Reference weights must be finite and strictly positive.")
            if not math.isclose(sum(direction), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("Reference directions must sum to one.")
        if self.evaluations < len(self.reference_directions):
            raise ValueError("The budget must initialize every reference type.")
        if self.checkpoint_period <= 0 or self.evaluations % self.checkpoint_period:
            raise ValueError("checkpoint_period must be a positive budget divisor.")
        if self.phase not in {"development", "calibration", "formal_confirmation"}:
            raise ValueError("Unsupported evidence phase.")
        if self.diversification_period <= 0:
            raise ValueError("diversification_period must be positive.")
        if self.exchange_period <= 0:
            raise ValueError("exchange_period must be positive.")
        if not 0.0 < self.operator_exploration <= 1.0:
            raise ValueError("operator_exploration must lie in (0, 1].")
        if self.candidate_id in {"C3", "C4"} and len(self.reference_directions) < 2:
            raise ValueError("C3 and C4 require at least two reference types.")
        if self.neighborhood_size < 2:
            raise ValueError("neighborhood_size must be at least two.")


@dataclass(frozen=True)
class V21EvaluationEvent:
    evaluation_index: int
    problem: str
    case: str
    seed: int
    phase: EvidencePhase
    search_phase: str
    type_index: int
    parent: Solution
    parent_sha256: str | None
    operator: str
    proposal: Solution
    proposal_sha256: str
    objectives: ObjectiveVector
    accepted_into_population: bool
    archive_changed: bool
    retained_after_update: bool
    duplicate: bool
    scalar_delta: float | None
    hv_delta: float | None
    new_region: bool
    local_search_depth: int
    elapsed_seconds: float


@dataclass(frozen=True)
class V21RunResult:
    optimization_result: OptimizationResult
    trace: Tuple[V21EvaluationEvent, ...]


class V21TypedHybridParetoSearch:
    """Budget-exact V21 engine with an all-evaluated trace ledger.

    This initial C0 implementation supplies the trace-complete execution seam.
    Family-native backbone operators and the additive C1--C4 mechanisms are
    layered onto this seam by subsequent prospective development slices.
    """

    def __init__(
        self,
        problem: MultiObjectiveCombinatorialProblem,
        config: V21HybridConfig,
    ) -> None:
        if problem.num_objectives != len(config.reference_directions[0]):
            raise ValueError("Problem and reference-direction dimensions disagree.")
        self.problem = problem
        self.config = config
        self._objective_lower_bounds = tuple(
            float(value) for value in problem.objective_lower_bounds
        )
        self._objective_upper_bounds = tuple(
            float(value) for value in problem.objective_upper_bounds
        )
        self._initialization_rng = random.Random(
            self._domain_seed(config.seed, "initialization")
        )
        self._backbone_rng = random.Random(
            self._domain_seed(config.seed, "native_backbone")
        )
        self._diversification_rng = random.Random(
            self._domain_seed(config.seed, "typed_diversification")
        )
        self._exchange_rng = random.Random(
            self._domain_seed(config.seed, "neighbor_path_relinking")
        )
        self._allocation_rng = random.Random(
            self._domain_seed(config.seed, "adaptive_operator_allocation")
        )
        self._operator_allocator = (
            _Exp3OperatorAllocator(
                (
                    "native_backbone",
                    "typed_diversification",
                    "neighbor_path_relinking",
                ),
                config.operator_exploration,
            )
            if config.candidate_id == "C4"
            else None
        )
        self._type_neighbors = tuple(
            tuple(
                sorted(
                    range(len(config.reference_directions)),
                    key=lambda other: (
                        sum(
                            (left - right) ** 2
                            for left, right in zip(
                                config.reference_directions[index],
                                config.reference_directions[other],
                            )
                        ),
                        other,
                    ),
                )[: min(config.neighborhood_size, len(config.reference_directions))]
            )
            for index in range(len(config.reference_directions))
        )
        self.archive = ParetoArchive(max_size=None, tol=0.0)
        self._best_archive_entry_by_type: list[ArchiveEntry | None] = [
            None for _ in config.reference_directions
        ]
        self._motsp_candidate_cities: tuple[tuple[int, ...], ...] = ()
        if isinstance(problem, MultiObjectiveTSPProblemAdapter):
            matrices = problem.instance.distance_matrices
            count = problem.instance.num_cities
            self._motsp_candidate_cities = tuple(
                tuple(
                    sorted(
                        (other for other in range(count) if other != city),
                        key=lambda other: (
                            sum(matrix[city][other] for matrix in matrices),
                            other,
                        ),
                    )[: min(8, count - 1)]
                )
                for city in range(count)
            )
        self._ledger = SQLiteEvaluationLedger.from_problem(
            problem,
            database_path=config.trace_database,
        )
        self._captured_trace: list[V21EvaluationEvent] = []
        self._diagnostics: list[Diagnostic] = []
        self._solutions: list[Solution] = []
        self._objectives: list[ObjectiveVector] = []
        self._regions: set[tuple[int, ...]] = set()
        self._accepted_count = 0
        self._start = time.perf_counter()
        self._has_run = False
        labels_by_boundary: dict[int, list[str]] = {}
        for label, boundary in (
            ("init_end", len(config.reference_directions)),
            (
                "early_10pct",
                max(
                    len(config.reference_directions),
                    int(math.ceil(0.10 * config.evaluations)),
                ),
            ),
            ("mid_70pct", int(math.ceil(0.70 * config.evaluations))),
            ("budget_end", config.evaluations),
        ):
            labels_by_boundary.setdefault(boundary, []).append(label)
        self._snapshot_labels_by_boundary = {
            boundary: tuple(labels)
            for boundary, labels in sorted(labels_by_boundary.items())
        }

    def run(self) -> V21RunResult:
        if self._has_run:
            raise RuntimeError("A V21 optimizer instance is single-use.")
        self._has_run = True
        for type_index in range(len(self.config.reference_directions)):
            if (
                self.config.candidate_id != "C0"
                and isinstance(self.problem, MultiObjectiveKnapsackInstance)
            ):
                proposal = self._mokp_greedy_repair(
                    (0,) * self.problem.solution_size,
                    self.config.reference_directions[type_index],
                )
                operator = "mokp_typed_profit_density_initialization_v1"
            elif (
                self.config.candidate_id != "C0"
                and isinstance(self.problem, MultiObjectiveTSPProblemAdapter)
            ):
                proposal = self._motsp_typed_initial_solution(type_index)
                operator = (
                    "motsp_typed_weighted_nearest_neighbor_initialization_v1"
                )
            else:
                proposal = self.problem.random_solution(self._initialization_rng)
                operator = "native_random_initialization_v1"
            evaluation_index, _ = self._record_evaluation(
                type_index=type_index,
                parent=None,
                proposal=proposal,
                operator=operator,
                search_phase="initialization",
                local_search_depth=0,
            )
            self._record_population_snapshot_if_due(evaluation_index)

        cursor = 0
        while self._ledger.evaluation_count < self.config.evaluations:
            type_index = cursor % len(self.config.reference_directions)
            cursor += 1
            parent = self._solutions[type_index]
            generation_parents: tuple[Solution, ...] | None = None
            generation_parent_types: tuple[int, ...] | None = None
            allocation_index: int | None = None
            allocation_probabilities: tuple[float, ...] | None = None
            if self._operator_allocator is not None:
                (
                    allocation_index,
                    allocation_probabilities,
                ) = self._operator_allocator.choose(self._allocation_rng)
                allocated = self._operator_allocator.operators[allocation_index]
                if allocated == "neighbor_path_relinking":
                    (
                        proposal,
                        operator,
                        guide,
                        guide_type,
                    ) = self._neighbor_path_relink_candidate(type_index)
                    search_phase = "neighbor_path_relinking"
                    generation_parents = (parent, guide)
                    generation_parent_types = (type_index, guide_type)
                elif allocated == "typed_diversification":
                    (
                        proposal,
                        operator,
                        source,
                    ) = self._typed_diversification_candidate(type_index)
                    search_phase = "typed_diversification"
                    generation_parents = (source,)
                    generation_parent_types = (type_index,)
                else:
                    (
                        proposal,
                        operator,
                        generation_parents,
                        generation_parent_types,
                    ) = self._native_candidate(
                        type_index,
                        cursor - 1,
                    )
                    search_phase = "native_backbone"
            elif (
                self._candidate_rank >= 3
                and cursor % self.config.exchange_period == 0
            ):
                (
                    proposal,
                    operator,
                    guide,
                    guide_type,
                ) = self._neighbor_path_relink_candidate(type_index)
                search_phase = "neighbor_path_relinking"
                generation_parents = (parent, guide)
                generation_parent_types = (type_index, guide_type)
            elif (
                self._candidate_rank >= 2
                and cursor % self.config.diversification_period == 0
            ):
                proposal, operator, source = self._typed_diversification_candidate(
                    type_index
                )
                search_phase = "typed_diversification"
                generation_parents = (source,)
                generation_parent_types = (type_index,)
            else:
                (
                    proposal,
                    operator,
                    generation_parents,
                    generation_parent_types,
                ) = self._native_candidate(
                    type_index,
                    cursor - 1,
                )
                search_phase = "native_backbone"
            evaluation_index, reward = self._record_evaluation(
                type_index=type_index,
                parent=parent,
                proposal=proposal,
                operator=operator,
                search_phase=search_phase,
                local_search_depth=1,
                generation_parents=generation_parents,
                generation_parent_types=generation_parent_types,
            )
            if (
                self._operator_allocator is not None
                and allocation_index is not None
                and allocation_probabilities is not None
            ):
                chosen = self._operator_allocator.operators[allocation_index]
                chosen_probability = allocation_probabilities[allocation_index]
                self._operator_allocator.update(
                    allocation_index,
                    reward,
                    chosen_probability,
                )
                self._ledger.record_mechanism(
                    after_evaluation_index=evaluation_index,
                    event_kind="operator_selection",
                    payload={
                        "available_operators": self._operator_allocator.operators,
                        "probabilities": allocation_probabilities,
                        "chosen_operator": chosen,
                        "reward": reward,
                        "post_update_weights": tuple(
                            self._operator_allocator.weights
                        ),
                    },
                )
            elif search_phase in {
                "typed_diversification",
                "neighbor_path_relinking",
            }:
                self._ledger.record_mechanism(
                    after_evaluation_index=evaluation_index,
                    event_kind=search_phase,
                    payload={
                        "type_index": type_index,
                        "operator": operator,
                        "reward": reward,
                    },
                )
            self._record_population_snapshot_if_due(evaluation_index)

        rebuilt = ParetoArchive(max_size=None, tol=0.0)
        for _, proposal, objectives in self._ledger.iter_evaluated_solutions():
            rebuilt.update((ArchiveEntry(proposal, objectives),))
        trace_complete = rebuilt.entries == self.archive.entries
        if not trace_complete:
            raise RuntimeError("The all-evaluated trace does not reproduce the archive.")
        trace_receipt = self._ledger.finalize(
            expected_budget=self.config.evaluations
        )

        result = OptimizationResult(
            particles=tuple(self._solutions),
            objectives=tuple(self._objectives),
            archive=self.archive,
            diagnostics=tuple(self._diagnostics),
            metadata={
                "algorithm": (
                    "v21-typed-hybrid-pareto-search-"
                    f"{self.config.candidate_id.lower()}"
                ),
                "candidate_id": self.config.candidate_id,
                "enabled_components": self._enabled_components,
                "evidence_phase": self.config.phase,
                "evaluation_budget": self.config.evaluations,
                "evaluations_used": self.config.evaluations,
                "exact_budget_gate": (
                    "PASS"
                    if trace_receipt["evaluation_record_count"]
                    == self.config.evaluations
                    else "FAIL"
                ),
                "all_evaluated_trace_complete": trace_complete,
                "trace_store_status": trace_receipt["status"],
                "trace_database_path": trace_receipt["database_path"],
                "trace_database_sha256": trace_receipt.get("database_sha256"),
                "trace_database_bytes": trace_receipt.get("database_bytes"),
                "physical_objective_calls": trace_receipt[
                    "physical_call_count"
                ],
                "evaluation_record_count": trace_receipt[
                    "evaluation_record_count"
                ],
                "decision_record_count": trace_receipt[
                    "decision_record_count"
                ],
                "terminal_evaluation_chain_sha256": trace_receipt[
                    "terminal_evaluation_chain_sha256"
                ],
                "terminal_decision_chain_sha256": trace_receipt[
                    "terminal_decision_chain_sha256"
                ],
                "terminal_mechanism_chain_sha256": trace_receipt[
                    "terminal_mechanism_chain_sha256"
                ],
                "mechanism_event_count": trace_receipt[
                    "mechanism_event_count"
                ],
                "competitive_search_archive_contract": (
                    "unbounded_exact_nondominated_all_evaluated_trace_v1"
                ),
                "competitive_search_archive_dominance_tolerance": 0.0,
                "v20_certificate_transfer": "PROHIBITED",
            },
        )
        return V21RunResult(result, tuple(self._captured_trace))

    def _record_population_snapshot_if_due(self, evaluation_index: int) -> None:
        labels = self._snapshot_labels_by_boundary.get(evaluation_index)
        if labels is None:
            return
        solution_hashes = tuple(
            hashlib.sha256(
                json.dumps(
                    [int(value) for value in solution],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for solution in self._solutions
        )
        archive_entries = set(self.archive.entries)
        current_archive_contribution = tuple(
            int(ArchiveEntry(solution, objectives) in archive_entries)
            for solution, objectives in zip(self._solutions, self._objectives)
        )
        self._ledger.record_mechanism(
            after_evaluation_index=evaluation_index,
            event_kind="population_snapshot",
            payload={
                "boundary_labels": labels,
                "type_ids": tuple(range(len(self._solutions))),
                "population_solution_sha256": solution_hashes,
                "population_unique_count": len(set(solution_hashes)),
                "population_unique_fraction": (
                    len(set(solution_hashes)) / len(solution_hashes)
                    if solution_hashes
                    else 0.0
                ),
                "reference_region_coverage_count": len(self._regions),
                "archive_size": len(self.archive),
                "current_archive_contribution_by_type": (
                    current_archive_contribution
                ),
                "resampling_ess_over_population": {
                    "status": "NOT_APPLICABLE",
                    "reason": (
                        "V21 diversification does not use particle resampling."
                    ),
                },
                "ancestor_multiplicity": {
                    "status": "NOT_APPLICABLE",
                    "reason": (
                        "No ancestor-resampling operation is present in this candidate."
                    ),
                },
            },
        )

    @property
    def _candidate_rank(self) -> int:
        return int(self.config.candidate_id[1:])

    @property
    def _enabled_components(self) -> tuple[str, ...]:
        components = ["native_backbone"]
        if self._candidate_rank >= 1:
            components.append("typed_initialization")
        if self._candidate_rank >= 2:
            components.append("typed_diversification")
        if self._candidate_rank >= 3:
            components.append("neighbor_path_relinking")
        if self._candidate_rank >= 4:
            components.append("adaptive_operator_allocation")
        return tuple(components)

    @staticmethod
    def _domain_seed(seed: int, domain: str) -> int:
        payload = f"v21|{int(seed)}|{domain}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    def _typed_diversification_candidate(
        self,
        type_index: int,
    ) -> tuple[Solution, str, Solution]:
        direction = self.config.reference_directions[type_index]
        cached = self._best_archive_entry_by_type[type_index]
        if cached is None:
            raise RuntimeError("The typed archive cache was not initialized.")
        source = cached.tour
        rng = self._diversification_rng
        if isinstance(self.problem, MultiObjectiveKnapsackInstance):
            child = list(source)
            count = min(
                self.problem.solution_size,
                max(2, self.problem.solution_size // 20),
            )
            for index in rng.sample(range(self.problem.solution_size), count):
                child[index] = 1 - child[index]
            return (
                self._mokp_greedy_repair(child, direction),
                "mokp_typed_archive_restart_multibit_repair_v1",
                source,
            )
        if isinstance(self.problem, MultiObjectiveTSPProblemAdapter):
            n = len(source)
            left, right = sorted(rng.sample(range(1, n), 2))
            child = list(two_opt_at(source, left, right))
            move_from, move_to = rng.sample(range(1, n), 2)
            city = child.pop(move_from)
            child.insert(move_to, city)
            return (
                tuple(child),
                "motsp_typed_archive_restart_two_opt_relocate_v1",
                source,
            )
        return (
            self.problem.propose(source, rng),
            "generic_typed_archive_restart_v1",
            source,
        )

    def _neighbor_path_relink_candidate(
        self,
        type_index: int,
    ) -> tuple[Solution, str, Solution, int]:
        directions = self.config.reference_directions
        ranked_neighbors = sorted(
            (index for index in range(len(directions)) if index != type_index),
            key=lambda index: (
                sum(
                    (left - right) ** 2
                    for left, right in zip(
                        directions[type_index],
                        directions[index],
                    )
                ),
                index,
            ),
        )
        guide_type = self._exchange_rng.choice(
            ranked_neighbors[: min(2, len(ranked_neighbors))]
        )
        parent = self._solutions[type_index]
        guide = self._solutions[guide_type]
        if isinstance(self.problem, MultiObjectiveKnapsackInstance):
            differences = [
                index
                for index, (left, right) in enumerate(zip(parent, guide))
                if left != right
            ]
            if not differences:
                differences = [self._exchange_rng.randrange(len(parent))]
            index = self._exchange_rng.choice(differences)
            child = list(parent)
            child[index] = guide[index] if parent[index] != guide[index] else 1 - child[index]
            return (
                self._mokp_greedy_repair(
                    child,
                    directions[type_index],
                ),
                "mokp_neighbor_binary_path_relink_repair_v1",
                guide,
                guide_type,
            )
        if isinstance(self.problem, MultiObjectiveTSPProblemAdapter):
            first = next(
                (
                    index
                    for index in range(1, len(parent))
                    if parent[index] != guide[index]
                ),
                None,
            )
            if first is None:
                first, second = sorted(
                    self._exchange_rng.sample(range(1, len(parent)), 2)
                )
                child = two_opt_at(parent, first, second)
            else:
                target_city = guide[first]
                source = parent.index(target_city)
                child_list = list(parent)
                city = child_list.pop(source)
                child_list.insert(first, city)
                child = tuple(child_list)
            return (
                child,
                "motsp_neighbor_path_relink_relocate_v1",
                guide,
                guide_type,
            )
        return (
            self.problem.propose(parent, self._exchange_rng),
            "generic_neighbor_path_relink_v1",
            guide,
            guide_type,
        )

    def _motsp_typed_initial_solution(self, type_index: int) -> Solution:
        problem = self.problem
        if not isinstance(problem, MultiObjectiveTSPProblemAdapter):
            raise TypeError("The MOTSP initializer requires a TSP adapter.")
        direction = self.config.reference_directions[type_index]
        matrices = problem.instance.distance_matrices
        remaining = set(range(1, problem.instance.num_cities))
        tour = [0]
        while remaining:
            current = tour[-1]
            following = min(
                remaining,
                key=lambda city: (
                    sum(
                        weight * matrix[current][city]
                        for weight, matrix in zip(direction, matrices)
                    ),
                    city,
                ),
            )
            tour.append(following)
            remaining.remove(following)
        output = tuple(tour)
        problem.validate_solution(output)
        return output

    def _native_candidate(
        self,
        type_index: int,
        search_step: int,
    ) -> tuple[
        Solution,
        str,
        tuple[Solution, ...],
        tuple[int, ...],
    ]:
        if isinstance(self.problem, MultiObjectiveKnapsackInstance):
            proposal, operator, parents, parent_types = self._mokp_native_candidate(
                type_index,
                search_step,
            )
            return (
                proposal,
                operator,
                parents,
                parent_types,
            )
        if isinstance(self.problem, MultiObjectiveTSPProblemAdapter):
            proposal, operator, source = self._motsp_native_candidate(
                type_index,
                search_step,
            )
            return (
                proposal,
                operator,
                (source,),
                (type_index,),
            )
        return (
            self.problem.propose(
                self._solutions[type_index],
                self._backbone_rng,
            ),
            "native_problem_proposal_v1",
            (self._solutions[type_index],),
            (type_index,),
        )

    def _motsp_native_candidate(
        self,
        type_index: int,
        search_step: int,
    ) -> tuple[Solution, str, Solution]:
        problem = self.problem
        if not isinstance(problem, MultiObjectiveTSPProblemAdapter):
            raise TypeError("The MOTSP operator requires a TSP adapter.")
        direction = self.config.reference_directions[type_index]
        cached = self._best_archive_entry_by_type[type_index]
        if cached is None:
            raise RuntimeError("The typed archive cache was not initialized.")
        parent = cached.tour
        n = len(parent)
        operator_index = search_step % 3
        if operator_index == 0:
            first = self._backbone_rng.randrange(1, n)
            city = parent[first]
            position_by_city = {
                candidate_city: position
                for position, candidate_city in enumerate(parent)
            }
            shortlist = [
                position_by_city[candidate_city]
                for candidate_city in self._motsp_candidate_cities[city]
                if position_by_city[candidate_city] != first
                and position_by_city[candidate_city] > 0
            ]
            if not shortlist:
                shortlist = [
                    position for position in range(1, n) if position != first
                ]
            second = self._backbone_rng.choice(shortlist)
            left, right = sorted((first, second))
            return (
                two_opt_at(parent, left, right),
                "motsp_candidate_list_two_opt_v1",
                parent,
            )
        if operator_index == 1:
            source = self._backbone_rng.randrange(1, n)
            target = self._backbone_rng.choice(
                [position for position in range(1, n) if position != source]
            )
            child = list(parent)
            city = child.pop(source)
            child.insert(target, city)
            return tuple(child), "motsp_relocate_v1", parent
        if n < 7:
            left, right = sorted(self._backbone_rng.sample(range(1, n), 2))
            return (
                two_opt_at(parent, left, right),
                "motsp_restricted_three_opt_v1",
                parent,
            )
        first, second, third = sorted(
            self._backbone_rng.sample(range(1, n), 3)
        )
        child = (
            parent[:first]
            + tuple(reversed(parent[first:second]))
            + tuple(reversed(parent[second:third]))
            + parent[third:]
        )
        return child, "motsp_restricted_three_opt_v1", parent

    def _mokp_native_candidate(
        self,
        type_index: int,
        search_step: int,
    ) -> tuple[
        Solution,
        str,
        tuple[Solution, ...],
        tuple[int, ...],
    ]:
        problem = self.problem
        if not isinstance(problem, MultiObjectiveKnapsackInstance):
            raise TypeError("The MOKP operator requires a knapsack problem.")
        operator_index = search_step % 4
        source = self._solutions[type_index]
        direction = self.config.reference_directions[type_index]
        densities = self._mokp_weighted_densities(direction)
        if operator_index == 1:
            child = list(source)
            current_weight = problem.total_weight(source)
            feasible_additions = [
                index
                for index, selected in enumerate(child)
                if not selected
                and current_weight + problem.item_weights[index]
                <= problem.capacity
            ]
            if feasible_additions:
                index = max(
                    feasible_additions,
                    key=lambda item: (densities[item], -item),
                )
                child[index] = 1
            else:
                selected = [
                    index for index, value in enumerate(child) if value
                ]
                index = min(
                    selected,
                    key=lambda item: (densities[item], item),
                )
                child[index] = 0
            return (
                self._mokp_greedy_repair(child, direction, refill=False),
                "mokp_add_drop_greedy_repair_v1",
                (source,),
                (type_index,),
            )
        if operator_index == 2:
            child = list(source)
            selected = [index for index, value in enumerate(child) if value]
            unselected = [index for index, value in enumerate(child) if not value]
            if selected and unselected:
                removed = min(
                    selected,
                    key=lambda item: (densities[item], item),
                )
                added = max(
                    unselected,
                    key=lambda item: (densities[item], -item),
                )
                child[removed] = 0
                child[added] = 1
            return (
                self._mokp_greedy_repair(child, direction, refill=False),
                "mokp_one_out_one_in_greedy_repair_v1",
                (source,),
                (type_index,),
            )
        if operator_index == 3:
            child = list(source)
            current_weight = problem.total_weight(source)
            feasible_additions = [
                index
                for index, selected in enumerate(child)
                if not selected
                and current_weight + problem.item_weights[index]
                <= problem.capacity
            ]
            if feasible_additions:
                added = max(
                    feasible_additions,
                    key=lambda item: (densities[item], -item),
                )
                child[added] = 1
            else:
                selected = sorted(
                    (index for index, value in enumerate(child) if value),
                    key=lambda item: (densities[item], item),
                )[:8]
                unselected = sorted(
                    (index for index, value in enumerate(child) if not value),
                    key=lambda item: (-densities[item], item),
                )[:8]
                if selected and unselected:
                    removed, added = max(
                        (
                            (removed, added)
                            for removed in selected
                            for added in unselected
                            if current_weight
                            - problem.item_weights[removed]
                            + problem.item_weights[added]
                            <= problem.capacity
                        ),
                        key=lambda pair: (
                            densities[pair[1]] - densities[pair[0]],
                            -pair[0],
                            -pair[1],
                        ),
                        default=(selected[0], unselected[0]),
                    )
                    child[removed] = 0
                    child[added] = 1
            return (
                self._mokp_greedy_repair(child, direction, refill=False),
                "mokp_bounded_density_local_improvement_greedy_repair_v1",
                (source,),
                (type_index,),
            )
        neighborhood = self._type_neighbors[type_index]
        left_type, right_type = self._backbone_rng.sample(
            list(neighborhood),
            2,
        )
        parent = self._solutions[left_type]
        guide = self._solutions[right_type]
        child = [
            left if self._backbone_rng.random() < 0.5 else right
            for left, right in zip(parent, guide)
        ]
        mutation_rate = 1.0 / problem.solution_size
        changed = False
        for index in range(problem.solution_size):
            if self._backbone_rng.random() < mutation_rate:
                child[index] = 1 - child[index]
                changed = True
        if not changed:
            index = self._backbone_rng.randrange(problem.solution_size)
            child[index] = 1 - child[index]
        return (
            self._mokp_greedy_repair(
                child,
                direction,
                refill=False,
            ),
            "mokp_uniform_crossover_bit_mutation_greedy_repair_v1",
            (parent, guide),
            (left_type, right_type),
        )

    def _mokp_weighted_densities(
        self,
        direction: Sequence[float],
    ) -> tuple[float, ...]:
        problem = self.problem
        if not isinstance(problem, MultiObjectiveKnapsackInstance):
            raise TypeError("The MOKP density requires a knapsack problem.")
        return tuple(
            sum(
                float(weight) * float(profits[index])
                for weight, profits in zip(
                    direction,
                    problem.profits_by_objective,
                )
            )
            / float(problem.item_weights[index])
            for index in range(problem.solution_size)
        )

    def _mokp_greedy_repair(
        self,
        solution: Sequence[int],
        direction: Sequence[float],
        *,
        refill: bool = True,
    ) -> Solution:
        problem = self.problem
        if not isinstance(problem, MultiObjectiveKnapsackInstance):
            raise TypeError("The MOKP repair requires a knapsack problem.")
        repaired = [int(value) for value in solution]
        densities = self._mokp_weighted_densities(direction)
        total_weight = sum(
            item_weight
            for selected, item_weight in zip(repaired, problem.item_weights)
            if selected
        )
        for index in sorted(
            (index for index, selected in enumerate(repaired) if selected),
            key=lambda item: (densities[item], item),
        ):
            if total_weight <= problem.capacity:
                break
            repaired[index] = 0
            total_weight -= problem.item_weights[index]
        if refill:
            for index in sorted(
                (index for index, selected in enumerate(repaired) if not selected),
                key=lambda item: (-densities[item], item),
            ):
                weight = problem.item_weights[index]
                if total_weight + weight <= problem.capacity:
                    repaired[index] = 1
                    total_weight += weight
        output = tuple(repaired)
        problem.validate_solution(output)
        return output

    def _record_evaluation(
        self,
        *,
        type_index: int,
        parent: Solution | None,
        proposal: Solution,
        operator: str,
        search_phase: str,
        local_search_depth: int,
        generation_parents: tuple[Solution, ...] | None = None,
        generation_parent_types: tuple[int, ...] | None = None,
    ) -> tuple[int, float]:
        self.problem.validate_solution(proposal)
        ledger_evaluation = self._ledger.evaluate(
            proposal,
            EvaluationContext(
                evidence_partition=self.config.phase,
                search_phase_id=search_phase,
                stage_id=(
                    "initialization_v1"
                    if parent is None
                    else "search_v1"
                ),
                type_id=type_index,
                operator_id=operator,
                operator_call_id=self._ledger.evaluation_count + 1,
                parent_solutions=(
                    ()
                    if parent is None
                    else (
                        generation_parents
                        if generation_parents is not None
                        else (parent,)
                    )
                ),
                parent_type_ids=(
                    ()
                    if parent is None
                    else (
                        generation_parent_types
                        if generation_parent_types is not None
                        else (type_index,)
                    )
                ),
                repair_applied=("repair" in operator),
                repair_operator_id=(
                    "mokp_weighted_greedy_repair_v1"
                    if "repair" in operator
                    else None
                ),
                local_search_depth=local_search_depth,
            ),
        )
        objectives = ledger_evaluation.objectives
        evaluation_index = ledger_evaluation.evaluation_index
        entry = ArchiveEntry(proposal, objectives)
        archive_changed = self.archive.update((entry,))
        retained_after_update = self.archive.contains(entry)
        if retained_after_update:
            for cached_type, cached_direction in enumerate(
                self.config.reference_directions
            ):
                current = self._best_archive_entry_by_type[cached_type]
                if current is None or self._archive_score(
                    entry,
                    cached_direction,
                ) < self._archive_score(current, cached_direction):
                    self._best_archive_entry_by_type[cached_type] = entry

        direction = self.config.reference_directions[type_index]
        parent_objective = (
            self._objectives[type_index]
            if parent is not None and type_index < len(self._objectives)
            else None
        )
        proposal_scalar = self._scalar(objectives, direction)
        scalar_delta = (
            proposal_scalar - self._scalar(parent_objective, direction)
            if parent_objective is not None
            else None
        )
        replacement_targets: tuple[int, ...]
        if parent is None:
            replacement_targets = (type_index,)
        elif isinstance(self.problem, MultiObjectiveKnapsackInstance):
            replacement_targets = tuple(
                target
                for target in self._type_neighbors[type_index]
                if self._scalar(
                    objectives,
                    self.config.reference_directions[target],
                )
                <= self._scalar(
                    self._objectives[target],
                    self.config.reference_directions[target],
                )
            )
        else:
            replacement_targets = (
                (type_index,)
                if scalar_delta is not None and scalar_delta <= 0.0
                else ()
            )
        accepted = bool(replacement_targets)
        if parent is None:
            self._solutions.append(proposal)
            self._objectives.append(objectives)
        else:
            for target in replacement_targets:
                self._solutions[target] = proposal
                self._objectives[target] = objectives
        if accepted:
            self._accepted_count += 1

        region = self._region(objectives)
        new_region = region not in self._regions
        self._regions.add(region)
        self._ledger.commit_decision(
            evaluation_index,
            DecisionInput(
                accepted_into_population=accepted,
                population_replacement_count=len(replacement_targets),
                population_target_type_ids=replacement_targets,
                decision_reason=(
                    "initial_population_fill"
                    if parent is None
                    else (
                        "nonworse_tchebycheff_replacement"
                        if accepted
                        else "worse_tchebycheff_rejection"
                    )
                ),
                archive_changed=archive_changed,
                retained_after_update=retained_after_update,
                archive_size_after=len(self.archive),
                scalarization_id="normalized_tchebycheff_v1",
                scalar_parent=(
                    None
                    if parent_objective is None
                    else self._scalar(parent_objective, direction)
                ),
                scalar_candidate=proposal_scalar,
                scalar_advantage=(
                    None if scalar_delta is None else -scalar_delta
                ),
                cell_id=":".join(str(value) for value in region),
                new_evaluated_cell=new_region,
                new_nondominated_cell=(new_region and archive_changed),
            ),
        )
        if self.config.capture_trace:
            self._captured_trace.append(
                V21EvaluationEvent(
                    evaluation_index=evaluation_index,
                    problem=self.problem.name,
                    case=self.problem.name,
                    seed=self.config.seed,
                    phase=self.config.phase,
                    search_phase=search_phase,
                    type_index=type_index,
                    parent=parent or (),
                    parent_sha256=(
                        None
                        if not ledger_evaluation.parent_solution_refs
                        else "stored_by_solution_ref"
                    ),
                    operator=operator,
                    proposal=proposal,
                    proposal_sha256=ledger_evaluation.proposal_sha256,
                    objectives=objectives,
                    accepted_into_population=accepted,
                    archive_changed=archive_changed,
                    retained_after_update=retained_after_update,
                    duplicate=(
                        ledger_evaluation.duplicate_of_evaluation_index is not None
                    ),
                    scalar_delta=scalar_delta,
                    hv_delta=None,
                    new_region=new_region,
                    local_search_depth=local_search_depth,
                    elapsed_seconds=(
                        ledger_evaluation.elapsed_monotonic_ns / 1_000_000_000.0
                    ),
                )
            )
        if evaluation_index % self.config.checkpoint_period == 0:
            checkpoint_hv = self.archive.hypervolume_2d(
                reference=self._objective_upper_bounds
            )
            self._diagnostics.append(
                Diagnostic(
                    iteration=evaluation_index,
                    temperature=0.0,
                    acceptance_rate=(
                        self._accepted_count / evaluation_index
                    ),
                    archive_size=len(self.archive),
                    hypervolume_2d=checkpoint_hv,
                    empirical_energy=0.0,
                    positive_archive_jump=0.0,
                    front=tuple(entry.objectives for entry in self.archive.entries),
                    elapsed_seconds=time.perf_counter() - self._start,
                )
            )
        reward = 1.0 if archive_changed else (0.25 if accepted else 0.0)
        return evaluation_index, reward

    def _scalar(
        self,
        objective: Sequence[float],
        direction: Sequence[float],
    ) -> float:
        normalized = tuple(
            (float(value) - float(lower)) / (float(upper) - float(lower))
            for value, lower, upper in zip(
                objective,
                self._objective_lower_bounds,
                self._objective_upper_bounds,
            )
        )
        return max(weight * value for weight, value in zip(direction, normalized))

    def _archive_score(
        self,
        entry: ArchiveEntry,
        direction: Sequence[float],
    ) -> tuple[float, ObjectiveVector, Solution]:
        return (
            self._scalar(entry.objectives, direction),
            entry.objectives,
            entry.tour,
        )

    def _region(self, objective: Sequence[float]) -> tuple[int, ...]:
        return tuple(
            min(
                19,
                max(
                    0,
                    int(
                        20.0
                        * (float(value) - float(lower))
                        / (float(upper) - float(lower))
                    ),
                ),
            )
            for value, lower, upper in zip(
                objective,
                self._objective_lower_bounds,
                self._objective_upper_bounds,
            )
        )


__all__ = [
    "V21EvaluationEvent",
    "V21HybridConfig",
    "V21RunResult",
    "V21TypedHybridParetoSearch",
]
