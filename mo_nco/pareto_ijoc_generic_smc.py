from __future__ import annotations

"""Problem-generic typed annealed SMC for the IJOC software interface.

The production MOTSP implementation remains optimized for exact 2-opt deltas.
This module keeps the same typed weighting/resampling/MH skeleton while using
only the public combinatorial-problem protocol.  It is therefore suitable for
cross-family validation, not as a claim that every family has equal runtime.
"""

import hashlib
import json
import math
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .archive import ArchiveEntry, ParetoArchive
from .pareto_ijoc_allocation import (
    Exp3TypeAllocator,
    SearchRewardWeights,
    derive_domain_separated_seed,
    normalized_hypervolume_gain,
)
from .pareto_ijoc_problem import MultiObjectiveCombinatorialProblem, Solution, problem_sha256
from .sampler import Diagnostic, OptimizationResult
from .types import ObjectiveVector


class GenericAnnealedParetoSMCOptimizer:
    contract_name = "generic_typed_annealed_pareto_smc_ijoc_v1"

    def __init__(
        self,
        problem: MultiObjectiveCombinatorialProblem,
        *,
        reference_directions: Sequence[Sequence[float]],
        particles_per_reference: int,
        evaluations: int,
        beta_schedule: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
        ess_threshold: float = 0.5,
        chebyshev_rho: float = 0.03,
        adaptive_search_evaluations: int = 0,
        adaptive_allocation_policy: str = "exp3",
        minimum_pulls_per_type: int = 0,
        exp3_exploration: Optional[float] = None,
        reward_weights: Optional[SearchRewardWeights] = None,
        cell_widths: Optional[Sequence[float]] = None,
        deployment_archive_max_size: Optional[int] = 100,
        anytime_checkpoint_period: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        if not isinstance(problem, MultiObjectiveCombinatorialProblem):
            raise ValueError("problem does not implement the IJOC combinatorial protocol.")
        if particles_per_reference <= 0:
            raise ValueError("particles_per_reference must be positive.")
        if evaluations <= 0:
            raise ValueError("evaluations must be positive.")
        if not 0.0 < ess_threshold <= 1.0:
            raise ValueError("ess_threshold must lie in (0, 1].")
        if not math.isfinite(chebyshev_rho) or chebyshev_rho <= 0.0:
            raise ValueError("chebyshev_rho must be finite and positive.")
        if (
            isinstance(adaptive_search_evaluations, bool)
            or not isinstance(adaptive_search_evaluations, int)
            or adaptive_search_evaluations < 0
        ):
            raise ValueError("adaptive_search_evaluations must be nonnegative.")
        if adaptive_allocation_policy not in {"exp3", "uniform"}:
            raise ValueError(
                "adaptive_allocation_policy must be 'exp3' or 'uniform'."
            )
        if (
            isinstance(minimum_pulls_per_type, bool)
            or not isinstance(minimum_pulls_per_type, int)
            or minimum_pulls_per_type < 0
        ):
            raise ValueError("minimum_pulls_per_type must be nonnegative.")
        if (
            anytime_checkpoint_period is not None
            and (
                isinstance(anytime_checkpoint_period, bool)
                or not isinstance(anytime_checkpoint_period, int)
                or anytime_checkpoint_period <= 0
                or anytime_checkpoint_period > evaluations
                or evaluations % anytime_checkpoint_period != 0
            )
        ):
            raise ValueError(
                "anytime_checkpoint_period must be a positive divisor of "
                "the exact evaluation budget."
            )
        beta = tuple(float(value) for value in beta_schedule)
        if len(beta) < 2 or beta[0] != 0.0 or any(
            not math.isfinite(value) or value < 0.0 for value in beta
        ) or any(right <= left for left, right in zip(beta, beta[1:])):
            raise ValueError("beta_schedule must start at zero and increase strictly.")
        directions = tuple(tuple(float(value) for value in row) for row in reference_directions)
        if not directions:
            raise ValueError("At least one reference direction is required.")
        for direction in directions:
            if len(direction) != problem.num_objectives:
                raise ValueError("A reference direction has the wrong dimension.")
            if any(not math.isfinite(value) or value <= 0.0 for value in direction):
                raise ValueError("Reference weights must be finite and positive.")
            if not math.isclose(math.fsum(direction), 1.0, rel_tol=0.0, abs_tol=1e-12):
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
        self.num_types = len(directions)
        self.particles_per_reference = int(particles_per_reference)
        self.num_particles = self.num_types * self.particles_per_reference
        self.evaluations = int(evaluations)
        self.adaptive_search_evaluations = adaptive_search_evaluations
        self.adaptive_allocation_policy = adaptive_allocation_policy
        self.minimum_pulls_per_type = int(minimum_pulls_per_type)
        if (
            self.adaptive_allocation_policy == "uniform"
            and self.minimum_pulls_per_type != 0
        ):
            raise ValueError(
                "Uniform allocation must set minimum_pulls_per_type to zero."
            )
        if (
            self.adaptive_allocation_policy == "uniform"
            and exp3_exploration is not None
        ):
            raise ValueError(
                "Uniform allocation must set exp3_exploration to None."
            )
        self.anytime_checkpoint_period = anytime_checkpoint_period
        self.adaptive_uniform_prefix_evaluations = (
            self.num_types * self.minimum_pulls_per_type
        )
        if self.adaptive_uniform_prefix_evaluations > self.adaptive_search_evaluations:
            raise ValueError("The uniform-prefix quota exceeds the adaptive tail budget.")
        self.core_evaluations = self.evaluations - self.adaptive_search_evaluations
        self.beta_schedule = beta
        self.ess_threshold = float(ess_threshold)
        self.chebyshev_rho = float(chebyshev_rho)
        self.reward_weights = reward_weights or SearchRewardWeights()
        self.rng = random.Random(seed)
        self.seed = int(seed)
        stochastic_context = problem_sha256(problem)
        self._adaptive_selection_seed = derive_domain_separated_seed(
            self.seed,
            context=stochastic_context,
            domain="generic_smc_adaptive_type_selection",
        )
        self._adaptive_environment_seed = derive_domain_separated_seed(
            self.seed,
            context=stochastic_context,
            domain="generic_smc_counterfactual_environment",
        )
        self._adaptive_selection_rng = random.Random(
            self._adaptive_selection_seed
        )
        self._adaptive_environment_rng = random.Random(
            self._adaptive_environment_seed
        )
        minimum_core = self.num_particles * len(self.beta_schedule)
        if self.core_evaluations < minimum_core:
            raise ValueError(
                f"The core needs at least {minimum_core} evaluations for initialization "
                "and one mutation per particle per positive stage."
            )
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
            not math.isfinite(value) or value <= 0.0 for value in self.cell_widths
        ):
            raise ValueError("cell_widths must contain one finite positive width per objective.")

        adaptive_suffix = (
            self.adaptive_search_evaluations
            - self.adaptive_uniform_prefix_evaluations
        )
        exploration = (
            exp3_exploration
            if exp3_exploration is not None
            else Exp3TypeAllocator.recommended_exploration(
                self.num_types,
                max(1, adaptive_suffix),
            )
        )
        self.allocator = (
            None
            if self.adaptive_allocation_policy == "uniform"
            or adaptive_suffix == 0
            else Exp3TypeAllocator(self.num_types, exploration=exploration)
        )
        self.search_archive = ParetoArchive(max_size=None, tol=0.0)
        self.deployment_archive = ParetoArchive(
            max_size=deployment_archive_max_size,
            tol=0.0,
        )
        self._solutions: List[List[Solution]] = [[] for _ in range(self.num_types)]
        self._objectives: List[List[ObjectiveVector]] = [[] for _ in range(self.num_types)]
        self._energies: List[List[float]] = [[] for _ in range(self.num_types)]
        uniform_log_weight = -math.log(self.particles_per_reference)
        self._log_weights = [
            [uniform_log_weight] * self.particles_per_reference
            for _ in range(self.num_types)
        ]
        self._observed_cells: set[Tuple[int, ...]] = set()
        self._nondominated_cells_ever: set[Tuple[int, ...]] = set()
        self._tail_cursors = [0] * self.num_types
        self._evaluations = 0
        self._accepted = 0
        self._resampling_events = 0
        self._start = time.perf_counter()
        self._stage_budgets = self._allocate_stage_budgets()
        self._stage_ledger: List[Dict[str, object]] = []
        self._diagnostics: List[Diagnostic] = []
        self._checkpoint_witnesses: List[Dict[str, object]] = []
        self._current_beta = self.beta_schedule[0]
        self._has_run = False

    def _allocate_stage_budgets(self) -> Tuple[int, ...]:
        remaining = self.core_evaluations - self.num_particles
        stages = len(self.beta_schedule) - 1
        base, extra = divmod(remaining, stages)
        return tuple(base + int(index < extra) for index in range(stages))

    @staticmethod
    def _logsumexp(values: Sequence[float]) -> float:
        maximum = max(values)
        return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))

    @staticmethod
    def _ess(log_weights: Sequence[float]) -> float:
        return 1.0 / math.fsum(math.exp(2.0 * value) for value in log_weights)

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
        return max(weighted) + self.chebyshev_rho * math.fsum(weighted)

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

    def _observe(self, solution: Solution) -> tuple[ObjectiveVector, bool]:
        objective = tuple(float(value) for value in self.problem.evaluate(solution))
        if len(objective) != self.problem.num_objectives:
            raise ValueError("Objective evaluation has the wrong dimension.")
        self._evaluations += 1
        cell = self._cell_index(objective)
        self._observed_cells.add(cell)
        entry = ArchiveEntry(solution, objective)
        self.search_archive.update((entry,))
        self.deployment_archive.update((entry,))
        survives = self.search_archive.contains(entry)
        novel = survives and cell not in self._nondominated_cells_ever
        if survives:
            self._nondominated_cells_ever.add(cell)
        self._maybe_log_checkpoint()
        return objective, novel

    def _box_volume(self) -> float:
        volume = 1.0
        for lower, upper in zip(
            self.objective_lower_bounds,
            self.objective_upper_bounds,
        ):
            volume *= upper - lower
        if not math.isfinite(volume) or volume <= 0.0:
            raise RuntimeError("Problem objective box has nonpositive volume.")
        return volume

    def _hv(self) -> float:
        if self.problem.num_objectives != 2:
            return 0.0
        return self.search_archive.hypervolume_2d(
            reference=self.objective_upper_bounds
        )

    def _build_diagnostic(self) -> Diagnostic:
        energy_count = sum(len(row) for row in self._energies)
        empirical_energy = (
            sum(sum(row) for row in self._energies) / energy_count
            if energy_count
            else 0.0
        )
        return Diagnostic(
            iteration=self._evaluations,
            temperature=(
                math.inf if self._current_beta == 0.0 else 1.0 / self._current_beta
            ),
            acceptance_rate=self._accepted
            / max(1, self._evaluations - self.num_particles),
            archive_size=len(self.search_archive),
            hypervolume_2d=self._hv(),
            empirical_energy=empirical_energy,
            positive_archive_jump=0.0,
            front=tuple(entry.objectives for entry in self.search_archive.entries),
            elapsed_seconds=time.perf_counter() - self._start,
        )

    def _maybe_log_checkpoint(self) -> None:
        period = self.anytime_checkpoint_period
        if (
            period is None
            or self._evaluations <= 0
            or self._evaluations % period != 0
        ):
            return
        self._diagnostics.append(self._build_diagnostic())
        self._checkpoint_witnesses.append(
            {
                "evaluation": self._evaluations,
                "entries": tuple(
                    {
                        "solution": entry.tour,
                        "objectives": entry.objectives,
                    }
                    for entry in self.search_archive.entries
                ),
            }
        )

    def _update_one(
        self,
        type_index: int,
        particle_index: int,
        beta: float,
        rng: random.Random,
    ) -> float:
        current = self._solutions[type_index][particle_index]
        current_energy = self._energies[type_index][particle_index]
        proposed = self.problem.propose(current, rng)
        forward = self.problem.proposal_probability(current, proposed)
        reverse = self.problem.proposal_probability(proposed, current)
        if forward <= 0.0 or not math.isclose(forward, reverse, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError("The generic MH path observed a non-symmetric proposal transition.")
        hv_before = self._hv()
        objective, new_cell = self._observe(proposed)
        proposed_energy = self._energy(objective, type_index)
        delta = proposed_energy - current_energy
        accepted = delta <= 0.0 or rng.random() < math.exp(min(0.0, -beta * delta))
        if accepted:
            self._solutions[type_index][particle_index] = proposed
            self._objectives[type_index][particle_index] = objective
            self._energies[type_index][particle_index] = proposed_energy
            self._accepted += 1
        return self.reward_weights.combine(
            normalized_hypervolume_gain=normalized_hypervolume_gain(
                hv_before,
                self._hv(),
                objective_box_volume=self._box_volume(),
            ),
            new_cell=new_cell,
            normalized_scalar_improvement=(
                max(0.0, current_energy - proposed_energy)
                / (1.0 + self.chebyshev_rho)
            ),
        )

    def _initialize(self) -> None:
        for type_index in range(self.num_types):
            for _ in range(self.particles_per_reference):
                solution = self.problem.random_solution(self.rng)
                objective, _ = self._observe(solution)
                self._solutions[type_index].append(solution)
                self._objectives[type_index].append(objective)
                self._energies[type_index].append(self._energy(objective, type_index))

    def run(self) -> OptimizationResult:
        if self._has_run:
            raise RuntimeError(
                "GenericAnnealedParetoSMCOptimizer instances are single-use."
            )
        self._has_run = True
        self._initialize()
        for stage_index, stage_budget in enumerate(self._stage_budgets, start=1):
            beta_previous = self.beta_schedule[stage_index - 1]
            beta = self.beta_schedule[stage_index]
            self._current_beta = beta
            delta_beta = beta - beta_previous
            base, extra = divmod(stage_budget, self.num_types)
            reference_records = []
            for type_index in range(self.num_types):
                unnormalized = [
                    old - delta_beta * energy
                    for old, energy in zip(
                        self._log_weights[type_index],
                        self._energies[type_index],
                    )
                ]
                normalizer = self._logsumexp(unnormalized)
                normalized = [value - normalizer for value in unnormalized]
                ess = self._ess(normalized)
                resampled = ess < self.ess_threshold * self.particles_per_reference
                if resampled:
                    ancestors = self.rng.choices(
                        range(self.particles_per_reference),
                        weights=[math.exp(value) for value in normalized],
                        k=self.particles_per_reference,
                    )
                    self._solutions[type_index] = [
                        self._solutions[type_index][index] for index in ancestors
                    ]
                    self._objectives[type_index] = [
                        self._objectives[type_index][index] for index in ancestors
                    ]
                    self._energies[type_index] = [
                        self._energies[type_index][index] for index in ancestors
                    ]
                    normalized = [
                        -math.log(self.particles_per_reference)
                    ] * self.particles_per_reference
                    self._resampling_events += 1
                self._log_weights[type_index] = normalized
                attempts = base + int(type_index < extra)
                accepted_before = self._accepted
                for attempt in range(attempts):
                    self._update_one(
                        type_index,
                        attempt % self.particles_per_reference,
                        beta,
                        self.rng,
                    )
                reference_records.append(
                    {
                        "type_index": type_index,
                        "ess": ess,
                        "resampled": resampled,
                        "mutation_attempts": attempts,
                        "accepted_mutations": self._accepted - accepted_before,
                    }
                )
            self._stage_ledger.append(
                {
                    "stage_index": stage_index,
                    "beta": beta,
                    "evaluations": stage_budget,
                    "references": tuple(reference_records),
                }
            )

        pre_tail_hash = hashlib.sha256(
            json.dumps(
                {
                    "objectives": self._objectives,
                    "log_weights": self._log_weights,
                    "evaluations": self._evaluations,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        self._current_beta = self.beta_schedule[-1]
        for tail_index in range(self.adaptive_search_evaluations):
            arm_rngs = tuple(
                random.Random(
                    self._adaptive_environment_rng.getrandbits(256)
                )
                for _ in range(self.num_types)
            )
            in_uniform_prefix = (
                tail_index < self.adaptive_uniform_prefix_evaluations
            )
            use_uniform = (
                self.adaptive_allocation_policy == "uniform"
                or in_uniform_prefix
            )
            if use_uniform:
                type_index = tail_index % self.num_types
                probability = 1.0
            else:
                assert self.allocator is not None
                type_index, probability = self.allocator.select(
                    self._adaptive_selection_rng
                )
            particle_index = self._tail_cursors[type_index]
            self._tail_cursors[type_index] = (
                particle_index + 1
            ) % self.particles_per_reference
            reward = self._update_one(
                type_index,
                particle_index,
                self.beta_schedule[-1],
                arm_rngs[type_index],
            )
            if not use_uniform:
                assert self.allocator is not None
                self.allocator.observe(type_index, reward, probability)

        if self._evaluations != self.evaluations:
            raise RuntimeError("Generic IJOC SMC failed to consume its exact budget.")
        if self.anytime_checkpoint_period is None:
            diagnostics = (self._build_diagnostic(),)
            expected_checkpoints: Tuple[int, ...] = ()
            observed_checkpoints: Tuple[int, ...] = ()
        else:
            expected_checkpoints = tuple(
                range(
                    self.anytime_checkpoint_period,
                    self.evaluations + 1,
                    self.anytime_checkpoint_period,
                )
            )
            observed_checkpoints = tuple(
                diagnostic.iteration for diagnostic in self._diagnostics
            )
            if observed_checkpoints != expected_checkpoints:
                raise RuntimeError(
                    "The generic IJOC anytime archive grid is incomplete; "
                    f"expected={expected_checkpoints}, observed={observed_checkpoints}."
                )
            diagnostics = tuple(self._diagnostics)
        return OptimizationResult(
            particles=tuple(solution for row in self._solutions for solution in row),
            objectives=tuple(objective for row in self._objectives for objective in row),
            archive=self.search_archive,
            diagnostics=diagnostics,
            metadata={
                "algorithm_contract": self.contract_name,
                "problem_sha256": problem_sha256(self.problem),
                "problem_name": self.problem.name,
                "problem_family_interface": "multiobjective_combinatorial_problem_v1",
                "symmetric_proposal_contract": self.problem.symmetric_proposal_contract,
                "evaluation_budget": self.evaluations,
                "core_evaluation_budget": self.core_evaluations,
                "adaptive_search_evaluation_budget": self.adaptive_search_evaluations,
                "adaptive_allocation_policy": self.adaptive_allocation_policy,
                "adaptive_minimum_pulls_per_type": self.minimum_pulls_per_type,
                "adaptive_uniform_prefix_evaluations": (
                    self.adaptive_uniform_prefix_evaluations
                ),
                "evaluations_used": self._evaluations,
                "beta_schedule": self.beta_schedule,
                "stage_ledger": tuple(self._stage_ledger),
                "anytime_checkpoint_period": self.anytime_checkpoint_period,
                "anytime_checkpoint_emission_contract": (
                    "per_evaluation_passive_archive_snapshot_with_solution_witness_v1"
                    if self.anytime_checkpoint_period is not None
                    else "disabled"
                ),
                "expected_anytime_checkpoints": expected_checkpoints,
                "observed_anytime_checkpoints": observed_checkpoints,
                "anytime_checkpoint_grid_complete": (
                    expected_checkpoints == observed_checkpoints
                ),
                "checkpoint_solution_witnesses": tuple(
                    self._checkpoint_witnesses
                ),
                "resampling_events": self._resampling_events,
                "pre_tail_state_sha256": pre_tail_hash,
                "allocator": None if self.allocator is None else self.allocator.metadata(),
                "counterfactual_reward_contract": (
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
                "observed_cell_count": len(self._observed_cells),
                "nondominated_cells_ever_count": len(self._nondominated_cells_ever),
                "generality_claim_scope": (
                    "shared_algorithmic_skeleton_until_multi_family_matched_evidence"
                ),
            },
        )
