from __future__ import annotations

import csv
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .archive import ArchiveEntry, ParetoArchive
from .evaluation import can_evaluate, evaluation_count
from .instance import MultiObjectiveTSPInstance
from .moves import mixed_move, order_crossover, random_tour
from .potential import PotentialContext, ScalarArchivePotential
from .types import ObjectiveVector, Tour


@dataclass(frozen=True)
class AnnealingSchedule:
    initial_temperature: float = 1.0
    final_temperature: float = 0.05
    total_iterations: int = 1000

    def temperature(self, iteration: int) -> float:
        if self.total_iterations <= 1:
            return self.final_temperature
        t = min(1.0, max(0.0, iteration / (self.total_iterations - 1)))
        ratio = self.final_temperature / self.initial_temperature
        return self.initial_temperature * (ratio**t)


@dataclass(frozen=True)
class Diagnostic:
    iteration: int
    temperature: float
    acceptance_rate: float
    archive_size: int
    hypervolume_2d: float
    empirical_energy: float
    positive_archive_jump: float
    front: Tuple[ObjectiveVector, ...] = ()
    elapsed_seconds: float = 0.0
    replacement_attempts: int = 0
    accepted_replacements: int = 0
    rejected_replacements: int = 0
    rejection_rate: float = 0.0
    current_rejection_streak: int = 0
    max_rejection_streak: int = 0


@dataclass(frozen=True)
class OptimizationResult:
    particles: Tuple[Tour, ...]
    objectives: Tuple[ObjectiveVector, ...]
    archive: ParetoArchive
    diagnostics: Tuple[Diagnostic, ...]
    metadata: Dict[str, object] = field(default_factory=dict)

    def write_archive_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = self.archive.entries
        if not entries:
            return
        dim = len(entries[0].objectives)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["tour"] + [f"objective_{idx + 1}" for idx in range(dim)])
            for entry in entries:
                writer.writerow([" ".join(map(str, entry.tour)), *entry.objectives])


class IPSMetropolisOptimizer:
    """Interacting-particle Metropolis optimizer for multi-objective TSP."""

    def __init__(
        self,
        instance: MultiObjectiveTSPInstance,
        num_particles: int = 48,
        iterations: int = 2000,
        seed: int = 0,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.05,
        archive_update_period: int = 25,
        log_period: int = 50,
        archive_max_size: Optional[int] = 200,
        potential: Optional[ScalarArchivePotential] = None,
        candidate_trials: int = 1,
        selection_tournament: int = 1,
        resample_fraction: float = 0.0,
        crossover_probability: float = 0.0,
        local_search_steps: int = 0,
        local_search_directions: int = 0,
        directional_probability: float = 0.0,
    ) -> None:
        if num_particles <= 0:
            raise ValueError("num_particles must be positive.")
        if iterations <= 0:
            raise ValueError("iterations must be positive.")
        if archive_update_period <= 0:
            raise ValueError("archive_update_period must be positive.")
        if log_period <= 0:
            raise ValueError("log_period must be positive.")

        self.instance = instance
        self.num_particles = num_particles
        self.iterations = iterations
        self.rng = random.Random(seed)
        self.schedule = AnnealingSchedule(initial_temperature, final_temperature, iterations)
        self.archive_update_period = archive_update_period
        self.log_period = log_period
        self.archive = ParetoArchive(max_size=archive_max_size)
        self.potential = potential or ScalarArchivePotential()
        self.candidate_trials = max(1, candidate_trials)
        self.selection_tournament = max(1, selection_tournament)
        self.resample_fraction = max(0.0, min(0.8, resample_fraction))
        self.crossover_probability = max(0.0, min(0.95, crossover_probability))
        self.local_search_steps = max(0, local_search_steps)
        self.local_search_directions = max(0, local_search_directions)
        self.directional_probability = max(0.0, min(0.95, directional_probability))

        self.particles: List[Tour] = [random_tour(instance.num_cities, self.rng) for _ in range(num_particles)]
        self.objectives: List[ObjectiveVector] = [instance.evaluate(tour) for tour in self.particles]
        self.archive.update(self._current_entries())
        self.context: PotentialContext = self.potential.build_context(self.archive, self.objectives)
        self._fit_potential(self.context)

        self.accepted_moves = 0
        self.total_moves = 0
        self.diagnostics: List[Diagnostic] = []

    def run(self) -> OptimizationResult:
        positive_archive_jump = 0.0
        iteration = 0
        start_time = time.perf_counter()
        while iteration < self.iterations and can_evaluate(self.instance):
            iteration += 1
            temperature = self.schedule.temperature(iteration)
            self._metropolis_step(temperature)

            if iteration % self.archive_update_period == 0:
                positive_archive_jump = self._archive_update_jump()

            if iteration % self.log_period == 0 or iteration == self.iterations or not can_evaluate(self.instance):
                self.diagnostics.append(
                    self._diagnostic(iteration, temperature, positive_archive_jump, time.perf_counter() - start_time)
                )
                positive_archive_jump = 0.0

        return OptimizationResult(
            particles=tuple(self.particles),
            objectives=tuple(self.objectives),
            archive=self.archive,
            diagnostics=tuple(self.diagnostics),
        )

    def _metropolis_step(self, temperature: float) -> None:
        if self.directional_probability > 0.0 and self.rng.random() < self.directional_probability:
            if self._directional_step(temperature):
                return

        idx = self._select_particle_index()
        current_tour = self.particles[idx]
        best_tour = current_tour
        best_objective = self.objectives[idx]
        best_delta_e = float("inf")
        for _ in range(self.candidate_trials):
            if not can_evaluate(self.instance):
                break
            proposed_tour = self._proposal(current_tour)
            proposed_objective = self.instance.evaluate(proposed_tour)
            delta_e = self.potential.delta_replace(self.objectives, idx, proposed_objective, self.context)
            if delta_e < best_delta_e:
                best_delta_e = delta_e
                best_tour = proposed_tour
                best_objective = proposed_objective

        delta_e = best_delta_e
        delta_h = self.num_particles * delta_e

        accept = delta_h <= 0.0
        if not accept:
            exponent = -delta_h / max(temperature, 1e-12)
            accept = self.rng.random() < math.exp(min(0.0, exponent))

        self.total_moves += 1
        if accept:
            self.accepted_moves += 1
            self.particles[idx] = best_tour
            self.objectives[idx] = best_objective

    def _directional_step(self, temperature: float) -> bool:
        if not can_evaluate(self.instance):
            return False
        weight = self.rng.choice(self.context.reference_directions)
        entries = list(self.archive.entries) + list(self._current_entries())
        if not entries:
            return False
        seed = min(entries, key=lambda entry: self._scalarized(entry.objectives, weight))
        idx = max(
            range(self.num_particles),
            key=lambda item: self._scalarized(self.objectives[item], weight),
        )
        proposed_tour = self._proposal(seed.tour)
        proposed_objective = self.instance.evaluate(proposed_tour)
        scalar_improved = self._scalarized(proposed_objective, weight) <= self._scalarized(self.objectives[idx], weight)
        delta_e = self.potential.delta_replace(self.objectives, idx, proposed_objective, self.context)
        delta_h = self.num_particles * delta_e
        accept = scalar_improved or delta_h <= 0.0
        if not accept:
            exponent = -delta_h / max(temperature, 1e-12)
            accept = self.rng.random() < math.exp(min(0.0, exponent))

        self.total_moves += 1
        if accept:
            self.accepted_moves += 1
            self.particles[idx] = proposed_tour
            self.objectives[idx] = proposed_objective
        return True

    def _archive_update_jump(self) -> float:
        old_energy = self.potential.empirical_energy(self.objectives, self.context)
        self.archive.update(self._current_entries())
        new_context = self.potential.build_context(self.archive, self.objectives)
        self.context = new_context
        self._fit_potential(self.context)
        self._archive_local_search()
        self.context = self.potential.build_context(self.archive, self.objectives)
        self._fit_potential(self.context)
        self._resample_from_archive()
        new_energy = self.potential.empirical_energy(self.objectives, self.context)
        return max(0.0, new_energy - old_energy)

    def _diagnostic(
        self,
        iteration: int,
        temperature: float,
        positive_archive_jump: float,
        elapsed_seconds: float,
    ) -> Diagnostic:
        acceptance_rate = self.accepted_moves / max(1, self.total_moves)
        hv = 0.0
        if self.instance.num_objectives == 2:
            hv = self.archive.hypervolume_2d()
        energy = self.potential.empirical_energy(self.objectives, self.context)
        return Diagnostic(
            iteration=iteration,
            temperature=temperature,
            acceptance_rate=acceptance_rate,
            archive_size=len(self.archive),
            hypervolume_2d=hv,
            empirical_energy=energy,
            positive_archive_jump=positive_archive_jump,
            front=tuple(entry.objectives for entry in self.archive.entries),
            elapsed_seconds=elapsed_seconds,
        )

    def _current_entries(self) -> Tuple[ArchiveEntry, ...]:
        return tuple(
            ArchiveEntry(tour=tour, objectives=objectives)
            for tour, objectives in zip(self.particles, self.objectives)
        )

    def _fit_potential(self, context: PotentialContext) -> None:
        fit = getattr(self.potential, "fit", None)
        if callable(fit):
            fit(self.objectives, self.archive.entries, context)

    def _select_particle_index(self) -> int:
        if self.selection_tournament <= 1:
            return self.rng.randrange(self.num_particles)
        sample_size = min(self.selection_tournament, self.num_particles)
        candidates = self.rng.sample(range(self.num_particles), sample_size)
        return max(candidates, key=lambda idx: self.potential.single_energy(self.objectives[idx], self.context))

    def _resample_from_archive(self) -> None:
        if self.resample_fraction <= 0.0 or not self.archive.entries:
            return
        count = max(1, int(self.num_particles * self.resample_fraction))
        worst = sorted(
            range(self.num_particles),
            key=lambda idx: self.potential.single_energy(self.objectives[idx], self.context),
            reverse=True,
        )[:count]
        archive_entries = list(self.archive.entries)
        for idx in worst:
            if not can_evaluate(self.instance):
                return
            source = self.rng.choice(archive_entries)
            candidate = source.tour
            for _ in range(2):
                candidate = self._proposal(candidate)
            candidate_obj = self.instance.evaluate(candidate)
            current_score = self.potential.single_energy(self.objectives[idx], self.context)
            candidate_score = self.potential.single_energy(candidate_obj, self.context)
            if candidate_score <= current_score or self.rng.random() < 0.25:
                self.particles[idx] = candidate
                self.objectives[idx] = candidate_obj

    def _proposal(self, current_tour: Tour) -> Tour:
        if self.crossover_probability > 0.0 and self.rng.random() < self.crossover_probability:
            parents = [entry.tour for entry in self.archive.entries] + self.particles
            parent = self.rng.choice(parents)
            child = order_crossover(current_tour, parent, self.rng)
            return mixed_move(child, self.rng)
        return mixed_move(current_tour, self.rng)

    def _archive_local_search(self) -> None:
        if self.local_search_steps <= 0 or self.local_search_directions <= 0:
            return
        seeds = list(self.archive.entries) + list(self._current_entries())
        if not seeds:
            return
        directions = self._selected_reference_directions()
        new_entries = []
        for weight in directions:
            seed = min(seeds, key=lambda entry: self._scalarized(entry.objectives, weight))
            best_tour = seed.tour
            best_obj = seed.objectives
            best_score = self._scalarized(best_obj, weight)
            for _ in range(self.local_search_steps):
                if not can_evaluate(self.instance):
                    break
                candidate_tour = self._proposal(best_tour)
                candidate_obj = self.instance.evaluate(candidate_tour)
                candidate_score = self._scalarized(candidate_obj, weight)
                if candidate_score <= best_score:
                    best_tour = candidate_tour
                    best_obj = candidate_obj
                    best_score = candidate_score
            new_entries.append(ArchiveEntry(best_tour, best_obj))
        self.archive.update(new_entries)

    def _selected_reference_directions(self) -> Tuple[ObjectiveVector, ...]:
        directions = self.context.reference_directions
        if self.local_search_directions >= len(directions):
            return directions
        if self.local_search_directions <= 1:
            return (directions[len(directions) // 2],)
        step = (len(directions) - 1) / (self.local_search_directions - 1)
        indices = sorted({round(i * step) for i in range(self.local_search_directions)})
        return tuple(directions[idx] for idx in indices)

    def _scalarized(self, objective: ObjectiveVector, weight: ObjectiveVector) -> float:
        z = self.potential.normalize(objective, self.context)
        cheb = max(max(1e-3, w) * value for w, value in zip(weight, z))
        aug = 0.03 * sum(max(1e-3, w) * value for w, value in zip(weight, z))
        return cheb + aug
        if best_delta_e == float("inf"):
            return
