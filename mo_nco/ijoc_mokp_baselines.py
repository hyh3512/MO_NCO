from __future__ import annotations

"""Budget-exact native baselines for the released biobjective MOKP suite.

These implementations are intentionally transparent and dependency-free.  They
provide matched objective-evaluation accounting, cumulative all-evaluated
archives, and checkpoint solution witnesses for the IJOC study.  They are not
presented as reproductions of any external package.
"""

import math
import random
import time
from typing import Callable, Iterable, List, Sequence, Tuple

from .archive import ArchiveEntry, ParetoArchive, dominates
from .pareto_ijoc_problem import MultiObjectiveKnapsackInstance, Solution
from .sampler import Diagnostic, OptimizationResult
from .types import ObjectiveVector


def _nondominated_sort(objectives: Sequence[ObjectiveVector]) -> List[List[int]]:
    dominates_set = [[] for _ in objectives]
    domination_count = [0] * len(objectives)
    first = []
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
        following = []
        for left in fronts[-1]:
            for right in dominates_set[left]:
                domination_count[right] -= 1
                if domination_count[right] == 0:
                    following.append(right)
        if following:
            fronts.append(following)
        else:
            break
    return fronts


def _crowding_distances(
    front: Sequence[int],
    objectives: Sequence[ObjectiveVector],
) -> dict[int, float]:
    if not front:
        return {}
    distances = {index: 0.0 for index in front}
    for objective_index in range(len(objectives[0])):
        ordered = sorted(front, key=lambda index: objectives[index][objective_index])
        distances[ordered[0]] = math.inf
        distances[ordered[-1]] = math.inf
        span = (
            objectives[ordered[-1]][objective_index]
            - objectives[ordered[0]][objective_index]
        )
        if span <= 0.0:
            continue
        for position in range(1, len(ordered) - 1):
            distances[ordered[position]] += (
                objectives[ordered[position + 1]][objective_index]
                - objectives[ordered[position - 1]][objective_index]
            ) / span
    return distances


class _BudgetExactMOKPBaseline:
    algorithm_id = "abstract-mokp-baseline"

    def __init__(
        self,
        problem: MultiObjectiveKnapsackInstance,
        *,
        evaluations: int,
        seed: int,
        anytime_checkpoint_period: int,
    ) -> None:
        if evaluations <= 0:
            raise ValueError("evaluations must be positive.")
        if (
            anytime_checkpoint_period <= 0
            or evaluations % anytime_checkpoint_period != 0
        ):
            raise ValueError(
                "anytime_checkpoint_period must be a positive divisor of evaluations."
            )
        self.problem = problem
        self.evaluations = int(evaluations)
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.anytime_checkpoint_period = int(anytime_checkpoint_period)
        self.archive = ParetoArchive(max_size=None, tol=0.0)
        self._evaluations = 0
        self._diagnostics: List[Diagnostic] = []
        self._checkpoint_witnesses: List[dict[str, object]] = []
        self._start = time.perf_counter()

    @property
    def objective_box_volume(self) -> float:
        return math.prod(
            upper - lower
            for lower, upper in zip(
                self.problem.objective_lower_bounds,
                self.problem.objective_upper_bounds,
            )
        )

    def _evaluate(self, solution: Solution) -> ObjectiveVector:
        objective = tuple(float(value) for value in self.problem.evaluate(solution))
        self._evaluations += 1
        self.archive.update((ArchiveEntry(solution, objective),))
        if self._evaluations % self.anytime_checkpoint_period == 0:
            self._log_checkpoint()
        return objective

    def _log_checkpoint(self) -> None:
        hypervolume = self.archive.hypervolume_2d(
            reference=self.problem.objective_upper_bounds
        )
        self._diagnostics.append(
            Diagnostic(
                iteration=self._evaluations,
                temperature=math.inf,
                acceptance_rate=0.0,
                archive_size=len(self.archive),
                hypervolume_2d=hypervolume,
                empirical_energy=0.0,
                positive_archive_jump=0.0,
                front=tuple(entry.objectives for entry in self.archive.entries),
                elapsed_seconds=time.perf_counter() - self._start,
            )
        )
        self._checkpoint_witnesses.append(
            {
                "evaluation": self._evaluations,
                "entries": tuple(
                    {
                        "solution": entry.tour,
                        "objectives": entry.objectives,
                    }
                    for entry in self.archive.entries
                ),
            }
        )

    def _repair(self, solution: Sequence[int]) -> Solution:
        repaired = [int(value) for value in solution]
        selected = [index for index, value in enumerate(repaired) if value]
        self.rng.shuffle(selected)
        total = sum(
            self.problem.item_weights[index]
            for index, value in enumerate(repaired)
            if value
        )
        for index in selected:
            if total <= self.problem.capacity:
                break
            repaired[index] = 0
            total -= self.problem.item_weights[index]
        output = tuple(repaired)
        self.problem.validate_solution(output)
        return output

    def _mutate(self, solution: Sequence[int], rate: float) -> Solution:
        mutated = [
            1 - value if self.rng.random() < rate else value
            for value in solution
        ]
        if tuple(mutated) == tuple(solution):
            index = self.rng.randrange(self.problem.solution_size)
            mutated[index] = 1 - mutated[index]
        return self._repair(mutated)

    def _crossover(self, left: Solution, right: Solution) -> Solution:
        child = [
            left[index] if self.rng.random() < 0.5 else right[index]
            for index in range(self.problem.solution_size)
        ]
        return self._repair(child)

    def _result(
        self,
        population: Sequence[Solution],
        objectives: Sequence[ObjectiveVector],
        *,
        algorithm_specific: dict[str, object],
    ) -> OptimizationResult:
        if self._evaluations != self.evaluations:
            raise RuntimeError(
                f"{self.algorithm_id} failed exact budget accounting."
            )
        expected = tuple(
            range(
                self.anytime_checkpoint_period,
                self.evaluations + 1,
                self.anytime_checkpoint_period,
            )
        )
        observed = tuple(item.iteration for item in self._diagnostics)
        if observed != expected:
            raise RuntimeError(
                f"{self.algorithm_id} emitted an incomplete checkpoint grid."
            )
        return OptimizationResult(
            particles=tuple(population),
            objectives=tuple(objectives),
            archive=self.archive,
            diagnostics=tuple(self._diagnostics),
            metadata={
                "algorithm": self.algorithm_id,
                "algorithm_scope": (
                    "transparent_in_repo_native_baseline_not_external_package_replication"
                ),
                "problem_family": "MOKP",
                "evaluation_budget": self.evaluations,
                "evaluations_used": self._evaluations,
                "exact_budget_gate": "PASS",
                "anytime_checkpoint_period": self.anytime_checkpoint_period,
                "expected_anytime_checkpoints": expected,
                "observed_anytime_checkpoints": observed,
                "anytime_checkpoint_grid_complete": True,
                "checkpoint_solution_witnesses": tuple(
                    self._checkpoint_witnesses
                ),
                "competitive_search_archive_contract": (
                    "unbounded_exact_nondominated_all_evaluated_candidates_v2"
                ),
                "competitive_search_archive_dominance_tolerance": 0.0,
                **algorithm_specific,
            },
        )


class BinaryNSGA2MOKPBaseline(_BudgetExactMOKPBaseline):
    algorithm_id = "mokp-binary-nsga2-native-v1"

    def __init__(
        self,
        problem: MultiObjectiveKnapsackInstance,
        *,
        evaluations: int,
        seed: int,
        anytime_checkpoint_period: int,
        population_size: int = 40,
        mutation_rate: float | None = None,
    ) -> None:
        super().__init__(
            problem,
            evaluations=evaluations,
            seed=seed,
            anytime_checkpoint_period=anytime_checkpoint_period,
        )
        if population_size < 2 or population_size > evaluations:
            raise ValueError("population_size must lie in [2, evaluations].")
        self.population_size = int(population_size)
        self.mutation_rate = (
            1.0 / problem.solution_size
            if mutation_rate is None
            else float(mutation_rate)
        )
        if not 0.0 < self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must lie in (0, 1].")

    @staticmethod
    def _rank_and_crowding(
        objectives: Sequence[ObjectiveVector],
    ) -> tuple[dict[int, int], dict[int, float], List[List[int]]]:
        fronts = _nondominated_sort(objectives)
        ranks = {
            index: rank
            for rank, front in enumerate(fronts)
            for index in front
        }
        crowding = {}
        for front in fronts:
            crowding.update(_crowding_distances(front, objectives))
        return ranks, crowding, fronts

    def _tournament(
        self,
        ranks: dict[int, int],
        crowding: dict[int, float],
    ) -> int:
        left = self.rng.randrange(len(ranks))
        right = self.rng.randrange(len(ranks))
        left_key = (ranks[left], -crowding[left], left)
        right_key = (ranks[right], -crowding[right], right)
        return left if left_key <= right_key else right

    def run(self) -> OptimizationResult:
        population = [
            self.problem.random_solution(self.rng)
            for _ in range(self.population_size)
        ]
        objectives = [self._evaluate(solution) for solution in population]
        while self._evaluations < self.evaluations:
            ranks, crowding, _ = self._rank_and_crowding(objectives)
            offspring = []
            offspring_objectives = []
            remaining = min(
                self.population_size,
                self.evaluations - self._evaluations,
            )
            for _ in range(remaining):
                left = population[self._tournament(ranks, crowding)]
                right = population[self._tournament(ranks, crowding)]
                child = self._mutate(
                    self._crossover(left, right),
                    self.mutation_rate,
                )
                offspring.append(child)
                offspring_objectives.append(self._evaluate(child))
            combined = population + offspring
            combined_objectives = objectives + offspring_objectives
            _, _, fronts = self._rank_and_crowding(combined_objectives)
            chosen = []
            for front in fronts:
                if len(chosen) + len(front) <= self.population_size:
                    chosen.extend(front)
                    continue
                crowding_front = _crowding_distances(
                    front,
                    combined_objectives,
                )
                chosen.extend(
                    sorted(
                        front,
                        key=lambda index: (
                            -crowding_front[index],
                            index,
                        ),
                    )[: self.population_size - len(chosen)]
                )
                break
            population = [combined[index] for index in chosen]
            objectives = [combined_objectives[index] for index in chosen]
        return self._result(
            population,
            objectives,
            algorithm_specific={
                "population_size": self.population_size,
                "mutation_rate": self.mutation_rate,
                "selection": "binary_tournament_rank_then_crowding_v1",
                "variation": "uniform_crossover_bit_mutation_random_drop_repair_v1",
            },
        )


class BinaryMOEADMOKPBaseline(_BudgetExactMOKPBaseline):
    algorithm_id = "mokp-binary-moead-native-v1"

    def __init__(
        self,
        problem: MultiObjectiveKnapsackInstance,
        *,
        evaluations: int,
        seed: int,
        anytime_checkpoint_period: int,
        population_size: int = 40,
        neighborhood_size: int = 8,
        mutation_rate: float | None = None,
    ) -> None:
        super().__init__(
            problem,
            evaluations=evaluations,
            seed=seed,
            anytime_checkpoint_period=anytime_checkpoint_period,
        )
        if population_size < 2 or population_size > evaluations:
            raise ValueError("population_size must lie in [2, evaluations].")
        if neighborhood_size < 2 or neighborhood_size > population_size:
            raise ValueError("Invalid MOEA/D neighborhood_size.")
        self.population_size = int(population_size)
        self.neighborhood_size = int(neighborhood_size)
        self.mutation_rate = (
            1.0 / problem.solution_size
            if mutation_rate is None
            else float(mutation_rate)
        )
        self.weights = tuple(
            (
                index / (self.population_size - 1),
                1.0 - index / (self.population_size - 1),
            )
            for index in range(self.population_size)
        )
        self.neighborhoods = tuple(
            tuple(
                sorted(
                    range(self.population_size),
                    key=lambda other: (
                        abs(self.weights[index][0] - self.weights[other][0]),
                        other,
                    ),
                )[: self.neighborhood_size]
            )
            for index in range(self.population_size)
        )

    @staticmethod
    def _scalar(
        objective: ObjectiveVector,
        ideal: ObjectiveVector,
        weight: tuple[float, float],
    ) -> float:
        safeguarded = (
            max(weight[0], 1e-6),
            max(weight[1], 1e-6),
        )
        return max(
            safeguarded[index] * abs(objective[index] - ideal[index])
            for index in range(2)
        )

    def run(self) -> OptimizationResult:
        population = [
            self.problem.random_solution(self.rng)
            for _ in range(self.population_size)
        ]
        objectives = [self._evaluate(solution) for solution in population]
        ideal = tuple(
            min(objective[index] for objective in objectives)
            for index in range(2)
        )
        cursor = 0
        while self._evaluations < self.evaluations:
            subproblem = cursor % self.population_size
            cursor += 1
            neighborhood = self.neighborhoods[subproblem]
            left_index, right_index = self.rng.sample(neighborhood, 2)
            child = self._mutate(
                self._crossover(
                    population[left_index],
                    population[right_index],
                ),
                self.mutation_rate,
            )
            objective = self._evaluate(child)
            ideal = tuple(
                min(ideal[index], objective[index]) for index in range(2)
            )
            for index in neighborhood:
                if self._scalar(
                    objective,
                    ideal,
                    self.weights[index],
                ) <= self._scalar(
                    objectives[index],
                    ideal,
                    self.weights[index],
                ):
                    population[index] = child
                    objectives[index] = objective
        return self._result(
            population,
            objectives,
            algorithm_specific={
                "population_size": self.population_size,
                "neighborhood_size": self.neighborhood_size,
                "mutation_rate": self.mutation_rate,
                "decomposition": "augmented_endpoint_safe_tchebycheff_v1",
                "variation": "uniform_crossover_bit_mutation_random_drop_repair_v1",
            },
        )


class ParetoLocalSearchMOKPBaseline(_BudgetExactMOKPBaseline):
    algorithm_id = "mokp-pls-native-v1"

    def run(self) -> OptimizationResult:
        initial = self.problem.random_solution(self.rng)
        initial_objective = self._evaluate(initial)
        queue = [initial]
        queued = {initial}
        cursor = 0
        last_solution = initial
        last_objective = initial_objective
        while self._evaluations < self.evaluations:
            if cursor >= len(queue):
                restart = self.problem.random_solution(self.rng)
                restart_objective = self._evaluate(restart)
                last_solution = restart
                last_objective = restart_objective
                if self.archive.contains(
                    ArchiveEntry(restart, restart_objective)
                ) and restart not in queued:
                    queue.append(restart)
                    queued.add(restart)
                continue
            current = queue[cursor]
            cursor += 1
            indices = list(range(self.problem.solution_size))
            self.rng.shuffle(indices)
            for index in indices:
                if self._evaluations >= self.evaluations:
                    break
                candidate = list(current)
                candidate[index] = 1 - candidate[index]
                candidate_tuple = tuple(candidate)
                try:
                    self.problem.validate_solution(candidate_tuple)
                except ValueError:
                    continue
                objective = self._evaluate(candidate_tuple)
                last_solution = candidate_tuple
                last_objective = objective
                if (
                    self.archive.contains(
                        ArchiveEntry(candidate_tuple, objective)
                    )
                    and candidate_tuple not in queued
                ):
                    queue.append(candidate_tuple)
                    queued.add(candidate_tuple)
        return self._result(
            (last_solution,),
            (last_objective,),
            algorithm_specific={
                "neighborhood": "all_feasible_one_bit_toggles_random_order_v1",
                "restart": "exact_uniform_feasible_when_queue_exhausted_v1",
                "visited_solution_count": len(queued),
            },
        )


MOKP_BASELINES: dict[
    str,
    Callable[..., _BudgetExactMOKPBaseline],
] = {
    BinaryNSGA2MOKPBaseline.algorithm_id: BinaryNSGA2MOKPBaseline,
    BinaryMOEADMOKPBaseline.algorithm_id: BinaryMOEADMOKPBaseline,
    ParetoLocalSearchMOKPBaseline.algorithm_id: ParetoLocalSearchMOKPBaseline,
}


def run_mokp_baseline(
    algorithm: str,
    problem: MultiObjectiveKnapsackInstance,
    *,
    evaluations: int,
    seed: int,
    anytime_checkpoint_period: int,
) -> OptimizationResult:
    try:
        constructor = MOKP_BASELINES[algorithm]
    except KeyError as error:
        raise ValueError(f"Unknown MOKP baseline: {algorithm!r}.") from error
    return constructor(
        problem,
        evaluations=evaluations,
        seed=seed,
        anytime_checkpoint_period=anytime_checkpoint_period,
    ).run()
