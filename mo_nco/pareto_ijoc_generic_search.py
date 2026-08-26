from __future__ import annotations

"""Problem-generic typed archive search used as an IJOC generality bridge."""

import hashlib
import math
import random
import time
from typing import Optional, Sequence, Tuple

from .archive import ArchiveEntry, ParetoArchive
from .pareto_ijoc_allocation import (
    Exp3TypeAllocator,
    SearchRewardWeights,
    derive_domain_separated_seed,
    normalized_hypervolume_gain,
)
from .pareto_ijoc_problem import MultiObjectiveCombinatorialProblem, problem_sha256
from .sampler import Diagnostic, OptimizationResult
from .types import ObjectiveVector


class GenericTypedArchiveSearch:
    """Typed Metropolis local search with EXP3 type allocation.

    This reusable engine is intentionally smaller than the TSP-specific SMC
    implementation.  It establishes that the adaptive allocation and archive
    separation layers do not depend on 2-opt or on a tour representation.
    """

    def __init__(
        self,
        problem: MultiObjectiveCombinatorialProblem,
        *,
        reference_directions: Sequence[Sequence[float]],
        population_per_type: int,
        evaluations: int,
        seed: int = 0,
        beta: float = 1.0,
        exp3_exploration: Optional[float] = None,
        minimum_pulls_per_type: int = 0,
        reward_weights: Optional[SearchRewardWeights] = None,
        cell_widths: Optional[Sequence[float]] = None,
        deployment_archive_max_size: Optional[int] = 100,
    ) -> None:
        if population_per_type <= 0:
            raise ValueError("population_per_type must be positive.")
        if evaluations <= 0:
            raise ValueError("evaluations must be positive.")
        if (
            isinstance(minimum_pulls_per_type, bool)
            or not isinstance(minimum_pulls_per_type, int)
            or minimum_pulls_per_type < 0
        ):
            raise ValueError("minimum_pulls_per_type must be nonnegative.")
        if not math.isfinite(beta) or beta < 0.0:
            raise ValueError("beta must be finite and nonnegative.")
        directions = tuple(tuple(float(value) for value in row) for row in reference_directions)
        if not directions:
            raise ValueError("At least one reference direction is required.")
        for direction in directions:
            if len(direction) != problem.num_objectives:
                raise ValueError("A reference direction has the wrong dimension.")
            if any(not math.isfinite(value) or value <= 0.0 for value in direction):
                raise ValueError("Reference weights must be finite and positive.")
            if not math.isclose(sum(direction), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("Reference directions must sum to one.")
        self.problem = problem
        self.objective_lower_bounds = tuple(
            float(value) for value in problem.objective_lower_bounds
        )
        self.objective_upper_bounds = tuple(
            float(value) for value in problem.objective_upper_bounds
        )
        if (
            len(self.objective_lower_bounds) != problem.num_objectives
            or len(self.objective_upper_bounds) != problem.num_objectives
            or any(
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or not lower < upper
                for lower, upper in zip(
                    self.objective_lower_bounds,
                    self.objective_upper_bounds,
                )
            )
        ):
            raise ValueError("The problem must expose a finite strict objective box.")
        self.reference_directions = directions
        self.population_per_type = population_per_type
        self.num_types = len(directions)
        self.num_particles = self.num_types * population_per_type
        if evaluations < self.num_particles:
            raise ValueError("The evaluation budget must initialize every typed particle.")
        self.evaluations = evaluations
        self.minimum_pulls_per_type = int(minimum_pulls_per_type)
        adaptive_rounds = evaluations - self.num_particles
        self.uniform_prefix_evaluations = (
            self.num_types * self.minimum_pulls_per_type
        )
        if self.uniform_prefix_evaluations > adaptive_rounds:
            raise ValueError("The uniform-prefix quota exceeds the search-tail budget.")
        self.beta = beta
        self.rng = random.Random(seed)
        self.seed = int(seed)
        stochastic_context = problem_sha256(problem)
        self._adaptive_selection_seed = derive_domain_separated_seed(
            self.seed,
            context=stochastic_context,
            domain="generic_search_adaptive_type_selection",
        )
        self._adaptive_environment_seed = derive_domain_separated_seed(
            self.seed,
            context=stochastic_context,
            domain="generic_search_counterfactual_environment",
        )
        self._adaptive_selection_rng = random.Random(
            self._adaptive_selection_seed
        )
        self._adaptive_environment_rng = random.Random(
            self._adaptive_environment_seed
        )
        self.reward_weights = reward_weights or SearchRewardWeights()
        if cell_widths is None:
            self.cell_widths = tuple(
                (upper - lower) / 20.0
                for lower, upper in zip(
                    self.objective_lower_bounds,
                    self.objective_upper_bounds,
                )
            )
        else:
            self.cell_widths = tuple(float(value) for value in cell_widths)
        if len(self.cell_widths) != self.problem.num_objectives or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.cell_widths
        ):
            raise ValueError(
                "cell_widths must contain one finite positive value per objective."
            )
        exploration = (
            exp3_exploration
            if exp3_exploration is not None
            else Exp3TypeAllocator.recommended_exploration(
                self.num_types,
                max(1, adaptive_rounds - self.uniform_prefix_evaluations),
            )
        )
        self.allocator = (
            None
            if adaptive_rounds == self.uniform_prefix_evaluations
            else Exp3TypeAllocator(self.num_types, exploration=exploration)
        )
        self.search_archive = ParetoArchive(max_size=None, tol=0.0)
        self.deployment_archive = ParetoArchive(
            max_size=deployment_archive_max_size,
            tol=0.0,
        )
        self._solutions = [[] for _ in range(self.num_types)]
        self._objectives = [[] for _ in range(self.num_types)]
        self._energies = [[] for _ in range(self.num_types)]
        self._cursors = [0] * self.num_types
        self._evaluations = 0
        self._accepted = 0
        self._observed_cells: set[Tuple[int, ...]] = set()
        self._nondominated_cells_ever: set[Tuple[int, ...]] = set()
        self._start = time.perf_counter()
        self._has_run = False

    def _energy(self, objective: ObjectiveVector, type_index: int) -> float:
        normalized = tuple(
            (value - lower) / (upper - lower)
            for value, lower, upper in zip(
                objective,
                self.objective_lower_bounds,
                self.objective_upper_bounds,
            )
        )
        if any(value < 0.0 or value > 1.0 for value in normalized):
            raise ValueError("Objective vector leaves the frozen problem box.")
        weighted = tuple(
            weight * value
            for weight, value in zip(self.reference_directions[type_index], normalized)
        )
        return max(weighted) + 0.03 * sum(weighted)

    def _cell_index(self, objective: ObjectiveVector) -> Tuple[int, ...]:
        cells = []
        for value, lower, upper, width in zip(
            objective,
            self.objective_lower_bounds,
            self.objective_upper_bounds,
            self.cell_widths,
        ):
            if not math.isfinite(value) or value < lower or value > upper:
                raise ValueError("Objective vector leaves the frozen problem box.")
            count = max(1, int(math.ceil((upper - lower) / width)))
            index = count - 1 if value == upper else int((value - lower) / width)
            if index < 0 or index >= count:
                raise RuntimeError("Cell classification left the frozen partition.")
            cells.append(index)
        return tuple(cells)

    def _box_volume(self) -> float:
        volume = 1.0
        for lower, upper in zip(
            self.objective_lower_bounds,
            self.objective_upper_bounds,
        ):
            volume *= upper - lower
        return volume

    def _hv(self) -> float:
        if self.problem.num_objectives != 2:
            return 0.0
        return self.search_archive.hypervolume_2d(
            reference=self.objective_upper_bounds
        )

    def _observe(self, solution: Tuple[int, ...]) -> Tuple[ObjectiveVector, bool]:
        objective = self.problem.evaluate(solution)
        self._evaluations += 1
        cell = self._cell_index(objective)
        self._observed_cells.add(cell)
        entry = ArchiveEntry(solution, objective)
        self.search_archive.update((entry,))
        self.deployment_archive.update((entry,))
        candidate_nondominated = self.search_archive.contains(entry)
        new_nondominated_cell = (
            candidate_nondominated
            and cell not in self._nondominated_cells_ever
        )
        if candidate_nondominated:
            self._nondominated_cells_ever.add(cell)
        return objective, new_nondominated_cell

    def run(self) -> OptimizationResult:
        if self._has_run:
            raise RuntimeError(
                "GenericTypedArchiveSearch instances are single-use."
            )
        self._has_run = True
        for type_index in range(self.num_types):
            for _ in range(self.population_per_type):
                solution = self.problem.random_solution(self.rng)
                objective, _ = self._observe(solution)
                self._solutions[type_index].append(solution)
                self._objectives[type_index].append(objective)
                self._energies[type_index].append(self._energy(objective, type_index))

        while self._evaluations < self.evaluations:
            tail_index = self._evaluations - self.num_particles
            arm_rngs = tuple(
                random.Random(
                    self._adaptive_environment_rng.getrandbits(256)
                )
                for _ in range(self.num_types)
            )
            in_uniform_prefix = tail_index < self.uniform_prefix_evaluations
            if in_uniform_prefix:
                type_index = tail_index % self.num_types
                probability = 1.0
            else:
                assert self.allocator is not None
                type_index, probability = self.allocator.select(
                    self._adaptive_selection_rng
                )
            proposal_rng = arm_rngs[type_index]
            particle_index = self._cursors[type_index]
            self._cursors[type_index] = (particle_index + 1) % self.population_per_type
            current = self._solutions[type_index][particle_index]
            current_energy = self._energies[type_index][particle_index]
            hv_before = self._hv()
            proposed = self.problem.propose(current, proposal_rng)
            forward = self.problem.proposal_probability(current, proposed)
            reverse = self.problem.proposal_probability(proposed, current)
            if (
                forward <= 0.0
                or not math.isclose(
                    forward, reverse, rel_tol=0.0, abs_tol=1e-15
                )
            ):
                raise RuntimeError(
                    "The generic IJOC search observed a non-symmetric "
                    "proposal transition."
                )
            objective, new_cell = self._observe(proposed)
            proposed_energy = self._energy(objective, type_index)
            delta = proposed_energy - current_energy
            accepted = delta <= 0.0 or proposal_rng.random() < math.exp(min(0.0, -self.beta * delta))
            if accepted:
                self._solutions[type_index][particle_index] = proposed
                self._objectives[type_index][particle_index] = objective
                self._energies[type_index][particle_index] = proposed_energy
                self._accepted += 1
            reward = self.reward_weights.combine(
                normalized_hypervolume_gain=normalized_hypervolume_gain(
                    hv_before,
                    self._hv(),
                    objective_box_volume=self._box_volume(),
                ),
                new_cell=new_cell,
                normalized_scalar_improvement=(
                    max(0.0, current_energy - proposed_energy) / 1.03
                ),
            )
            if not in_uniform_prefix:
                assert self.allocator is not None
                self.allocator.observe(type_index, reward, probability)

        diagnostic = Diagnostic(
            iteration=self._evaluations,
            temperature=(math.inf if self.beta == 0.0 else 1.0 / self.beta),
            acceptance_rate=self._accepted / max(1, self.evaluations - self.num_particles),
            archive_size=len(self.search_archive),
            hypervolume_2d=self._hv(),
            empirical_energy=sum(sum(row) for row in self._energies) / self.num_particles,
            positive_archive_jump=0.0,
            front=tuple(entry.objectives for entry in self.search_archive.entries),
            elapsed_seconds=time.perf_counter() - self._start,
        )
        return OptimizationResult(
            particles=tuple(solution for row in self._solutions for solution in row),
            objectives=tuple(objective for row in self._objectives for objective in row),
            archive=self.search_archive,
            diagnostics=(diagnostic,),
            metadata={
                "algorithm_contract": "generic_typed_archive_search_exp3_v1",
                "problem_sha256": problem_sha256(self.problem),
                "problem_name": self.problem.name,
                "problem_family_interface": "multiobjective_combinatorial_problem_v1",
                "symmetric_proposal_contract": self.problem.symmetric_proposal_contract,
                "evaluation_budget": self.evaluations,
                "evaluations_used": self._evaluations,
                "reference_directions": self.reference_directions,
                "cell_widths": self.cell_widths,
                "observed_cell_count": len(self._observed_cells),
                "nondominated_cells_ever_count": len(
                    self._nondominated_cells_ever
                ),
                "adaptive_new_cell_reward_semantics": (
                    "first_nondominated_candidate_in_frozen_cell_v1"
                ),
                "allocator": (
                    None if self.allocator is None else self.allocator.metadata()
                ),
                "minimum_pulls_per_type": self.minimum_pulls_per_type,
                "uniform_prefix_evaluations": self.uniform_prefix_evaluations,
                "adaptive_search_counterfactual_reward_contract": (
                    "private_per_type_random_tapes_sampled_before_arm_selection_v1"
                ),
                "adaptive_rng_domain_separation_contract": (
                    "independent_mutable_rng_states_for_selection_and_environment_v1"
                ),
                "adaptive_selection_seed_sha256": hashlib.sha256(
                    str(self._adaptive_selection_seed).encode("ascii")
                ).hexdigest(),
                "adaptive_environment_seed_sha256": hashlib.sha256(
                    str(self._adaptive_environment_seed).encode("ascii")
                ).hexdigest(),
                "competitive_search_archive_contract": (
                    "unbounded_exact_nondominated_all_evaluated_candidates_v2"
                ),
                "competitive_search_archive_size": len(self.search_archive),
                "competitive_search_archive_dominance_tolerance": 0.0,
                "deployment_archive_size": len(self.deployment_archive),
                "deployment_archive_dominance_tolerance": 0.0,
                "deployment_archive_entries": tuple(
                    {"solution": entry.tour, "objectives": entry.objectives}
                    for entry in self.deployment_archive.entries
                ),
                "generality_claim_scope": (
                    "software_interface_and_smoke_only_until_multi_family_matched_evidence"
                ),
            },
        )
