from __future__ import annotations

import itertools
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .archive import ArchiveEntry, ParetoArchive, dominates
from .evaluation import can_evaluate, evaluation_count
from .instance import MultiObjectiveTSPInstance
from .moves import order_crossover, random_tour, sample_two_opt_indices, two_opt, two_opt_at
from .potential import ScalarArchivePotential
from .sampler import Diagnostic, OptimizationResult
from .types import ObjectiveVector, Tour


def nondominated_sort(objectives: Sequence[ObjectiveVector]) -> List[List[int]]:
    n = len(objectives)
    dominated_by_count = [0] * n
    dominates_list: List[List[int]] = [[] for _ in range(n)]
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                dominates_list[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                dominated_by_count[p] += 1
        if dominated_by_count[p] == 0:
            fronts[0].append(p)

    idx = 0
    while idx < len(fronts) and fronts[idx]:
        next_front: List[int] = []
        for p in fronts[idx]:
            for q in dominates_list[p]:
                dominated_by_count[q] -= 1
                if dominated_by_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        idx += 1
    return fronts


def crowding_distance(front: Sequence[int], objectives: Sequence[ObjectiveVector]) -> Dict[int, float]:
    if not front:
        return {}
    dim = len(objectives[front[0]])
    distance = {idx: 0.0 for idx in front}
    if len(front) <= 2:
        for idx in front:
            distance[idx] = float("inf")
        return distance

    for m in range(dim):
        ordered = sorted(front, key=lambda idx: objectives[idx][m])
        distance[ordered[0]] = float("inf")
        distance[ordered[-1]] = float("inf")
        lo = objectives[ordered[0]][m]
        hi = objectives[ordered[-1]][m]
        scale = max(1e-12, hi - lo)
        for pos in range(1, len(ordered) - 1):
            prev_value = objectives[ordered[pos - 1]][m]
            next_value = objectives[ordered[pos + 1]][m]
            if math.isfinite(distance[ordered[pos]]):
                distance[ordered[pos]] += (next_value - prev_value) / scale
    return distance


def rank_and_crowding(objectives: Sequence[ObjectiveVector]) -> Tuple[Dict[int, int], Dict[int, float]]:
    fronts = nondominated_sort(objectives)
    rank: Dict[int, int] = {}
    crowding: Dict[int, float] = {}
    for rank_idx, front in enumerate(fronts):
        for idx in front:
            rank[idx] = rank_idx
        crowding.update(crowding_distance(front, objectives))
    return rank, crowding


def evaluate_two_opt_candidate(
    instance: MultiObjectiveTSPInstance,
    tour: Tour,
    objective: ObjectiveVector,
    rng: random.Random,
) -> Tuple[Tour, ObjectiveVector]:
    i, j = sample_two_opt_indices(len(tour), rng)
    child = two_opt_at(tour, i, j)
    method = getattr(instance, "evaluate_two_opt", None)
    child_obj = method(tour, objective, i, j) if callable(method) else instance.evaluate(child)
    return child, child_obj


class RandomTwoOptOptimizer:
    """Random-walk 2-opt baseline with a Pareto archive."""

    def __init__(
        self,
        instance: MultiObjectiveTSPInstance,
        num_particles: int = 48,
        iterations: int = 2000,
        seed: int = 0,
        archive_update_period: int = 25,
        log_period: int = 50,
        archive_max_size: Optional[int] = 200,
    ) -> None:
        self._start_time = time.perf_counter()
        self.instance = instance
        self.num_particles = num_particles
        self.iterations = iterations
        self.rng = random.Random(seed)
        self.archive_update_period = archive_update_period
        self.log_period = log_period
        self.archive = ParetoArchive(max_size=archive_max_size)
        self.particles = [random_tour(instance.num_cities, self.rng) for _ in range(num_particles)]
        self.objectives = [instance.evaluate(tour) for tour in self.particles]
        self.archive.update(self._entries())
        self.diagnostics: List[Diagnostic] = []

    def run(self) -> OptimizationResult:
        iteration = 0
        self._log_diagnostic(self._used_evaluations())
        while iteration < self.iterations and can_evaluate(self.instance):
            iteration += 1
            idx = self.rng.randrange(self.num_particles)
            self.particles[idx], self.objectives[idx] = evaluate_two_opt_candidate(
                self.instance,
                self.particles[idx],
                self.objectives[idx],
                self.rng,
            )
            if iteration % self.archive_update_period == 0:
                self.archive.update(self._entries())
            if iteration % self.log_period == 0 or iteration == self.iterations or not can_evaluate(self.instance):
                self._log_diagnostic(self._used_evaluations())
        return OptimizationResult(tuple(self.particles), tuple(self.objectives), self.archive, tuple(self.diagnostics))

    def _used_evaluations(self) -> int:
        counted = evaluation_count(self.instance)
        return counted if counted > 0 else min(self.iterations, len(self.objectives) + len(self.diagnostics) * self.log_period)

    def _log_diagnostic(self, evaluations: int) -> None:
        hv = self.archive.hypervolume_2d() if self.instance.num_objectives == 2 else 0.0
        self.diagnostics.append(
            Diagnostic(
                evaluations,
                0.0,
                1.0,
                len(self.archive),
                hv,
                0.0,
                0.0,
                tuple(entry.objectives for entry in self.archive.entries),
                time.perf_counter() - self._start_time,
            )
        )

    def _entries(self) -> Tuple[ArchiveEntry, ...]:
        return tuple(ArchiveEntry(tour, obj) for tour, obj in zip(self.particles, self.objectives))


class NSGAIIOptimizer:
    """Compact NSGA-II baseline for permutation-coded TSP."""

    def __init__(
        self,
        instance: MultiObjectiveTSPInstance,
        population_size: int = 48,
        evaluations: int = 2000,
        seed: int = 0,
        mutation_probability: float = 0.35,
        log_period: int = 50,
        archive_max_size: Optional[int] = 200,
    ) -> None:
        self._start_time = time.perf_counter()
        self.instance = instance
        self.population_size = population_size
        self.generations = max(1, evaluations // max(1, population_size))
        self.rng = random.Random(seed)
        self.mutation_probability = mutation_probability
        self.log_period = max(1, log_period // max(1, population_size))
        self.archive = ParetoArchive(max_size=archive_max_size)
        self.population = [random_tour(instance.num_cities, self.rng) for _ in range(population_size)]
        self.objectives = [instance.evaluate(tour) for tour in self.population]
        self.archive.update(self._entries())
        self.diagnostics: List[Diagnostic] = []

    def run(self) -> OptimizationResult:
        generation = 0
        self._log_diagnostic(self._used_evaluations())
        while generation < self.generations and can_evaluate(self.instance):
            generation += 1
            rank, crowding = rank_and_crowding(self.objectives)
            offspring: List[Tour] = []
            while len(offspring) < self.population_size and can_evaluate(self.instance):
                p1 = self.population[self._tournament(rank, crowding)]
                p2 = self.population[self._tournament(rank, crowding)]
                child = order_crossover(p1, p2, self.rng)
                if self.rng.random() < self.mutation_probability:
                    child = two_opt(child, self.rng)
                offspring.append(child)
            offspring_objectives = [self.instance.evaluate(tour) for tour in offspring if can_evaluate(self.instance)]
            offspring = offspring[: len(offspring_objectives)]
            if not offspring:
                break

            combined = self.population + offspring
            combined_objectives = self.objectives + offspring_objectives
            selected = self._environmental_selection(combined, combined_objectives)
            self.population = [combined[idx] for idx in selected]
            self.objectives = [combined_objectives[idx] for idx in selected]
            self.archive.update(self._entries())

            if generation % self.log_period == 0 or generation == self.generations or not can_evaluate(self.instance):
                self._log_diagnostic(self._used_evaluations(generation))

        return OptimizationResult(tuple(self.population), tuple(self.objectives), self.archive, tuple(self.diagnostics))

    def _used_evaluations(self, generation: int = 0) -> int:
        counted = evaluation_count(self.instance)
        return counted if counted > 0 else min(self.generations * self.population_size, self.population_size * (generation + 1))

    def _log_diagnostic(self, evaluations: int) -> None:
        hv = self.archive.hypervolume_2d() if self.instance.num_objectives == 2 else 0.0
        self.diagnostics.append(
            Diagnostic(
                evaluations,
                0.0,
                0.0,
                len(self.archive),
                hv,
                0.0,
                0.0,
                tuple(entry.objectives for entry in self.archive.entries),
                time.perf_counter() - self._start_time,
            )
        )

    def _tournament(self, rank: Dict[int, int], crowding: Dict[int, float]) -> int:
        a, b = self.rng.sample(range(self.population_size), 2)
        if rank[a] < rank[b]:
            return a
        if rank[b] < rank[a]:
            return b
        return a if crowding.get(a, 0.0) >= crowding.get(b, 0.0) else b

    def _environmental_selection(
        self,
        population: Sequence[Tour],
        objectives: Sequence[ObjectiveVector],
    ) -> List[int]:
        selected: List[int] = []
        for front in nondominated_sort(objectives):
            if len(selected) + len(front) <= self.population_size:
                selected.extend(front)
                continue
            crowd = crowding_distance(front, objectives)
            remaining = self.population_size - len(selected)
            selected.extend(sorted(front, key=lambda idx: crowd[idx], reverse=True)[:remaining])
            break
        return selected

    def _entries(self) -> Tuple[ArchiveEntry, ...]:
        return tuple(ArchiveEntry(tour, obj) for tour, obj in zip(self.population, self.objectives))


class MOEADOptimizer:
    """Small MOEA/D-style decomposition baseline with 2-opt variation."""

    def __init__(
        self,
        instance: MultiObjectiveTSPInstance,
        population_size: int = 48,
        evaluations: int = 2000,
        seed: int = 0,
        neighbor_size: int = 8,
        log_period: int = 50,
        archive_max_size: Optional[int] = 200,
    ) -> None:
        self._start_time = time.perf_counter()
        self.instance = instance
        self.population_size = population_size
        self.generations = max(1, evaluations // max(1, population_size))
        self.rng = random.Random(seed)
        self.log_period = max(1, log_period // max(1, population_size))
        self.archive = ParetoArchive(max_size=archive_max_size)
        self.weights = ScalarArchivePotential.reference_directions(instance.num_objectives, population_size)
        self.neighbors = self._build_neighbors(max(2, min(neighbor_size, population_size)))
        self.population = [random_tour(instance.num_cities, self.rng) for _ in range(population_size)]
        self.objectives = [instance.evaluate(tour) for tour in self.population]
        self.ideal = self._ideal(self.objectives)
        self.archive.update(self._entries())
        self.diagnostics: List[Diagnostic] = []

    def run(self) -> OptimizationResult:
        generation = 0
        self._log_diagnostic(self._used_evaluations())
        while generation < self.generations and can_evaluate(self.instance):
            generation += 1
            for i in range(self.population_size):
                if not can_evaluate(self.instance):
                    break
                parent_idx = self.rng.choice(self.neighbors[i])
                child, child_obj = evaluate_two_opt_candidate(
                    self.instance,
                    self.population[parent_idx],
                    self.objectives[parent_idx],
                    self.rng,
                )
                self.ideal = tuple(min(a, b) for a, b in zip(self.ideal, child_obj))
                for j in self.neighbors[i]:
                    if self._scalar(child_obj, self.weights[j]) <= self._scalar(self.objectives[j], self.weights[j]):
                        self.population[j] = child
                        self.objectives[j] = child_obj
            self.archive.update(self._entries())
            if generation % self.log_period == 0 or generation == self.generations or not can_evaluate(self.instance):
                self._log_diagnostic(self._used_evaluations(generation))
        return OptimizationResult(tuple(self.population), tuple(self.objectives), self.archive, tuple(self.diagnostics))

    def _used_evaluations(self, generation: int = 0) -> int:
        counted = evaluation_count(self.instance)
        return counted if counted > 0 else min(self.generations * self.population_size, self.population_size * (generation + 1))

    def _log_diagnostic(self, evaluations: int) -> None:
        hv = self.archive.hypervolume_2d() if self.instance.num_objectives == 2 else 0.0
        self.diagnostics.append(
            Diagnostic(
                evaluations,
                0.0,
                0.0,
                len(self.archive),
                hv,
                0.0,
                0.0,
                tuple(entry.objectives for entry in self.archive.entries),
                time.perf_counter() - self._start_time,
            )
        )

    def _scalar(self, objective: ObjectiveVector, weight: ObjectiveVector) -> float:
        return max(max(1e-3, w) * abs(value - ideal) for value, ideal, w in zip(objective, self.ideal, weight))

    def _build_neighbors(self, neighbor_size: int) -> List[List[int]]:
        neighbors: List[List[int]] = []
        for i, wi in enumerate(self.weights):
            ranked = sorted(
                range(len(self.weights)),
                key=lambda j: sum((a - b) ** 2 for a, b in zip(wi, self.weights[j])),
            )
            neighbors.append(ranked[:neighbor_size])
        return neighbors

    @staticmethod
    def _ideal(objectives: Sequence[ObjectiveVector]) -> ObjectiveVector:
        dim = len(objectives[0])
        return tuple(min(obj[i] for obj in objectives) for i in range(dim))

    def _entries(self) -> Tuple[ArchiveEntry, ...]:
        return tuple(ArchiveEntry(tour, obj) for tour, obj in zip(self.population, self.objectives))


class MOTSPParetoLocalSearchOptimizer:
    """Pareto local search baseline specialized for bi-objective TSP.

    The baseline repeatedly expands nondominated archive tours with exact
    counted 2-opt moves. In scalar-guided mode it selects archive parents by
    reference-direction Tchebycheff scores, which is a compact TPLS/MOGLS-style
    pressure test for the IPS archive machinery.
    """

    def __init__(
        self,
        instance: MultiObjectiveTSPInstance,
        population_size: int = 48,
        evaluations: int = 2000,
        seed: int = 0,
        log_period: int = 50,
        archive_max_size: Optional[int] = 500,
        archive_tolerance: float = 1e-12,
        neighborhood_sample: int = 12,
        scalar_guided: bool = True,
        anytime_checkpoint_period: Optional[int] = None,
        stalled_expansion_policy: str = "none",
        restart_random_attempts: int = 64,
    ) -> None:
        if (
            anytime_checkpoint_period is not None
            and (
                isinstance(anytime_checkpoint_period, bool)
                or anytime_checkpoint_period <= 0
                or anytime_checkpoint_period > evaluations
                or evaluations % anytime_checkpoint_period != 0
            )
        ):
            raise ValueError(
                "anytime_checkpoint_period must be a positive divisor of "
                "the evaluation budget."
            )
        if stalled_expansion_policy not in {
            "none",
            "uniform-random-unvisited-v1",
        }:
            raise ValueError("Unsupported stalled_expansion_policy.")
        if (
            isinstance(restart_random_attempts, bool)
            or restart_random_attempts < 0
        ):
            raise ValueError("restart_random_attempts must be nonnegative.")
        self.instance = instance
        self.population_size = population_size
        self.evaluations = evaluations
        self.rng = random.Random(seed)
        self.log_period = log_period
        self.archive = ParetoArchive(
            max_size=archive_max_size,
            tol=archive_tolerance,
        )
        self.neighborhood_sample = max(1, neighborhood_sample)
        self.scalar_guided = scalar_guided
        self.anytime_checkpoint_period = anytime_checkpoint_period
        self.stalled_expansion_policy = stalled_expansion_policy
        self.restart_random_attempts = restart_random_attempts
        self._start_time = time.perf_counter()
        self.diagnostics: List[Diagnostic] = []
        self._checkpoint_solution_witnesses: List[Dict[str, object]] = []
        self.weights = ScalarArchivePotential.reference_directions(instance.num_objectives, max(2, population_size))
        self.population = []
        self.objectives = []
        self._local_evaluations = 0
        self._stalled_expansions = 0
        self._restart_evaluations = 0
        self._restart_random_draws = 0
        self._restart_fallbacks = 0
        self._current_stalled_expansion_streak = 0
        self._max_stalled_expansion_streak = 0
        self._restart_fallback_permutations = itertools.permutations(
            range(1, instance.num_cities)
        )
        for _ in range(population_size):
            tour = random_tour(instance.num_cities, self.rng)
            objective = instance.evaluate(tour)
            self._local_evaluations += 1
            self.population.append(tour)
            self.objectives.append(objective)
            self.archive.update((ArchiveEntry(tour, objective),))
            self._maybe_log_checkpoint()
        self._visited_tours = {tour for tour in self.population}

    def run(self) -> OptimizationResult:
        last_log = self._used_evaluations()
        step = 0
        while can_evaluate(self.instance) and self._used_evaluations() < self.evaluations:
            parent = self._select_archive_parent(step)
            if parent is None:
                break
            evaluations_before = self._used_evaluations()
            self._expand_parent(parent, step)
            if self._used_evaluations() == evaluations_before:
                self._stalled_expansions += 1
                self._current_stalled_expansion_streak += 1
                self._max_stalled_expansion_streak = max(
                    self._max_stalled_expansion_streak,
                    self._current_stalled_expansion_streak,
                )
                if self.stalled_expansion_policy == "uniform-random-unvisited-v1":
                    self._restart_from_unvisited_tour(step)
                    if self._used_evaluations() <= evaluations_before:
                        raise RuntimeError(
                            "MOTSP PLS restart failed to advance the evaluation ledger."
                        )
            else:
                self._current_stalled_expansion_streak = 0
            step += 1
            evals = self._used_evaluations()
            if evals - last_log >= self.log_period or not can_evaluate(self.instance):
                self._log_diagnostic(evals)
                last_log = evals
        if (
            not self.diagnostics
            or self.diagnostics[-1].iteration
            != self._used_evaluations()
        ):
            self._log_diagnostic(self._used_evaluations())
        if self._used_evaluations() != self.evaluations:
            raise RuntimeError(
                "MOTSP PLS failed to consume the exact evaluation budget."
            )
        expected_checkpoints = (
            tuple(
                range(
                    self.anytime_checkpoint_period,
                    self.evaluations + 1,
                    self.anytime_checkpoint_period,
                )
            )
            if self.anytime_checkpoint_period is not None
            else ()
        )
        observed_checkpoints = tuple(
            diagnostic.iteration
            for diagnostic in self.diagnostics
            if diagnostic.iteration in expected_checkpoints
        )
        if observed_checkpoints != expected_checkpoints:
            raise RuntimeError("MOTSP PLS emitted an incomplete checkpoint grid.")
        tours = tuple(entry.tour for entry in self.archive.entries)
        objectives = tuple(entry.objectives for entry in self.archive.entries)
        return OptimizationResult(
            tours,
            objectives,
            self.archive,
            tuple(self.diagnostics),
            metadata={
                "algorithm": (
                    "motsp-pls-restart-native-v2"
                    if self.stalled_expansion_policy
                    == "uniform-random-unvisited-v1"
                    else "motsp-pls-native-v1"
                ),
                "evaluation_budget": self.evaluations,
                "evaluations_used": self._used_evaluations(),
                "exact_budget_gate": "PASS",
                "archive_tolerance": self.archive.tol,
                "stalled_expansion_policy": self.stalled_expansion_policy,
                "restart_random_attempts": self.restart_random_attempts,
                "stalled_expansions": self._stalled_expansions,
                "restart_evaluations": self._restart_evaluations,
                "restart_random_draws": self._restart_random_draws,
                "restart_fallbacks": self._restart_fallbacks,
                "max_stalled_expansion_streak": (
                    self._max_stalled_expansion_streak
                ),
                "liveness_gate": (
                    "PASS"
                    if self.stalled_expansion_policy
                    == "uniform-random-unvisited-v1"
                    else "NOT_APPLICABLE"
                ),
                "anytime_checkpoint_period": self.anytime_checkpoint_period,
                "expected_anytime_checkpoints": expected_checkpoints,
                "observed_anytime_checkpoints": observed_checkpoints,
                "anytime_checkpoint_grid_complete": (
                    observed_checkpoints == expected_checkpoints
                ),
                "checkpoint_solution_witnesses": tuple(
                    self._checkpoint_solution_witnesses
                ),
            },
        )

    def _select_archive_parent(self, step: int) -> Optional[ArchiveEntry]:
        entries = self.archive.entries
        if not entries:
            return None
        if not self.scalar_guided or self.instance.num_objectives != 2:
            return entries[step % len(entries)]
        weight = self.weights[step % len(self.weights)]
        ideal, nadir = self._ideal_nadir(entries)
        archive_parent = min(entries, key=lambda entry: self._scalar(entry.objectives, weight, ideal, nadir))
        current_idx = step % len(self.population)
        current = ArchiveEntry(self.population[current_idx], self.objectives[current_idx])
        if self._scalar(current.objectives, weight, ideal, nadir) < self._scalar(archive_parent.objectives, weight, ideal, nadir):
            return current
        return archive_parent

    def _expand_parent(self, parent: ArchiveEntry, step: int) -> None:
        n = len(parent.tour)
        if n < 4:
            return
        if n <= 80 and self.neighborhood_sample >= n:
            pairs = [(i, j) for i in range(1, n) for j in range(i + 1, n)]
            self.rng.shuffle(pairs)
            pairs = pairs[: self.neighborhood_sample]
        else:
            pairs = [sample_two_opt_indices(n, self.rng) for _ in range(self.neighborhood_sample)]
        weight = self.weights[step % len(self.weights)]
        ideal, nadir = self._ideal_nadir(self.archive.entries)
        parent_scalar = self._scalar(parent.objectives, weight, ideal, nadir)
        best_scalar_entry = parent
        best_scalar = parent_scalar
        new_entries: List[ArchiveEntry] = []
        for i, j in pairs:
            if not can_evaluate(self.instance) or self._used_evaluations() >= self.evaluations:
                break
            child = two_opt_at(parent.tour, i, j)
            if child in self._visited_tours:
                continue
            self._visited_tours.add(child)
            method = getattr(self.instance, "evaluate_two_opt", None)
            child_obj = method(parent.tour, parent.objectives, i, j) if callable(method) else self.instance.evaluate(child)
            if evaluation_count(self.instance) == 0:
                self._local_evaluations += 1
            child_entry = ArchiveEntry(child, child_obj)
            new_entries.append(child_entry)
            self.archive.update((child_entry,))
            self._maybe_log_checkpoint()
            value = self._scalar(child_obj, weight, ideal, nadir)
            if value < best_scalar:
                best_scalar = value
                best_scalar_entry = child_entry
        if best_scalar_entry is not parent:
            idx = step % len(self.population)
            self.population[idx] = best_scalar_entry.tour
            self.objectives[idx] = best_scalar_entry.objectives
            self.archive.update([best_scalar_entry])

    def _restart_from_unvisited_tour(self, step: int) -> None:
        """Evaluate one new tour after a zero-progress neighborhood expansion.

        The primary proposal is a bounded sequence of uniform random tours.
        If every draw has already been evaluated, a persistent lexicographic
        iterator supplies a deterministic unvisited fallback without rescanning
        earlier permutations on later restarts.
        """

        if not can_evaluate(self.instance) or self._used_evaluations() >= self.evaluations:
            return
        selected: Optional[Tour] = None
        for _ in range(self.restart_random_attempts):
            candidate = random_tour(self.instance.num_cities, self.rng)
            self._restart_random_draws += 1
            if candidate not in self._visited_tours:
                selected = candidate
                break
        if selected is None:
            self._restart_fallbacks += 1
            for tail in self._restart_fallback_permutations:
                candidate = (0, *tail)
                if candidate not in self._visited_tours:
                    selected = candidate
                    break
        if selected is None:
            raise RuntimeError(
                "MOTSP PLS exhausted every fixed-zero tour before the "
                "requested evaluation budget."
            )

        self._visited_tours.add(selected)
        objective = self.instance.evaluate(selected)
        if evaluation_count(self.instance) == 0:
            self._local_evaluations += 1
        entry = ArchiveEntry(selected, objective)
        population_index = step % len(self.population)
        self.population[population_index] = selected
        self.objectives[population_index] = objective
        self.archive.update((entry,))
        self._restart_evaluations += 1
        self._maybe_log_checkpoint()

    def _log_diagnostic(self, evaluations: int) -> None:
        hv = self.archive.hypervolume_2d() if self.instance.num_objectives == 2 else 0.0
        diagnostic = Diagnostic(
                evaluations,
                0.0,
                0.0,
                len(self.archive),
                hv,
                0.0,
                0.0,
                tuple(entry.objectives for entry in self.archive.entries),
                time.perf_counter() - self._start_time,
            )
        if (
            self.diagnostics
            and self.diagnostics[-1].iteration == evaluations
        ):
            self.diagnostics[-1] = diagnostic
        else:
            self.diagnostics.append(diagnostic)
        period = self.anytime_checkpoint_period
        if (
            period is not None
            and evaluations > 0
            and evaluations % period == 0
        ):
            witness: Dict[str, object] = {
                "evaluation": evaluations,
                "entries": tuple(
                    {
                        "tour": entry.tour,
                        "objectives": entry.objectives,
                    }
                    for entry in self.archive.entries
                ),
            }
            if (
                self._checkpoint_solution_witnesses
                and self._checkpoint_solution_witnesses[-1]["evaluation"]
                == evaluations
            ):
                self._checkpoint_solution_witnesses[-1] = witness
            else:
                self._checkpoint_solution_witnesses.append(witness)

    def _maybe_log_checkpoint(self) -> None:
        period = self.anytime_checkpoint_period
        evaluations = self._used_evaluations()
        if (
            period is not None
            and evaluations > 0
            and evaluations % period == 0
        ):
            self._log_diagnostic(evaluations)

    def _used_evaluations(self) -> int:
        counted = evaluation_count(self.instance)
        return counted if counted > 0 else self._local_evaluations

    def _scalar(
        self,
        objective: ObjectiveVector,
        weight: ObjectiveVector,
        ideal: ObjectiveVector,
        nadir: ObjectiveVector,
    ) -> float:
        normalized = [
            (value - lo) / max(1e-9, hi - lo)
            for value, lo, hi in zip(objective, ideal, nadir)
        ]
        terms = [max(1e-3, w) * value for value, w in zip(normalized, weight)]
        return max(terms) + 0.03 * sum(terms)

    def _ideal_nadir(self, entries: Sequence[ArchiveEntry]) -> Tuple[ObjectiveVector, ObjectiveVector]:
        vectors = [entry.objectives for entry in entries] or list(self.objectives)
        dim = len(vectors[0])
        ideal = tuple(min(vector[idx] for vector in vectors) for idx in range(dim))
        nadir = tuple(max(vector[idx] for vector in vectors) for idx in range(dim))
        return ideal, nadir

    def _entries(self) -> Tuple[ArchiveEntry, ...]:
        return tuple(ArchiveEntry(tour, obj) for tour, obj in zip(self.population, self.objectives))
