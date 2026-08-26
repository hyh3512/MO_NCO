from __future__ import annotations

import hashlib
import json
import math
import random
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .archive import ArchiveEntry, ParetoArchive
from .contracts import ClaimLevel
from .evaluation import (
    CountingTSPInstance,
    evaluation_count,
    remaining_evaluations,
)
from .instance import MultiObjectiveTSPInstance, instance_sha256
from .moves import random_tour, sample_two_opt_indices, two_opt_at
from .pareto_fk_certificate import (
    make_bootstrap_fk_plan,
    make_contraction_aware_fk_plan,
)
from .pareto_smc_spec import analytic_objective_box
from .pareto_execution_contract import (
    DomainSeparatedSeed,
    verify_domain_separated_seed,
    verify_full_type_sweep_checkpoints,
)
from .pareto_ijoc_allocation import (
    Exp3TypeAllocator,
    SearchRewardWeights,
    derive_domain_separated_seed,
    normalized_hypervolume_gain,
)
from .potential import ScalarArchivePotential
from .sampler import Diagnostic, OptimizationResult
from .types import ObjectiveVector, Tour

TwoOptEvaluator = Callable[
    [Tour, ObjectiveVector, int, int],
    ObjectiveVector,
]


class ObjectiveBoundsViolation(RuntimeError):
    """Raised when an evaluated objective leaves the predeclared objective box."""


class AnnealedParetoSMCOptimizer:
    """Strict typed annealed SMC with an external Pareto epsilon observer.

    Reference types never exchange particles.  Type ``r`` follows the
    Feynman--Kac path

        pi_{s,r}(x) proportional to exp(-beta_s U_r(f(x))).

    The frozen objective box defines the continuous normalized augmented
    Tchebycheff energy.  The separately predeclared epsilon grid is only a
    coverage observer.  Neither that observer nor the nondominated reporting
    archive is read by weighting, resampling or mutation.
    """

    contract_name = "annealed_pareto_smc_feynman_kac_v4"
    implementation_version = "0.6.0"

    def __init__(
        self,
        instance: MultiObjectiveTSPInstance,
        *,
        particles_per_reference: int = 8,
        evaluations: int = 512,
        seed: int = 0,
        beta_schedule: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 4.0),
        reference_directions: Optional[Sequence[Sequence[float]]] = None,
        num_reference_types: int = 5,
        objective_lower_bounds: Optional[Sequence[float]] = None,
        objective_upper_bounds: Optional[Sequence[float]] = None,
        epsilon: Optional[float | Sequence[float]] = None,
        ess_threshold: float = 0.5,
        resampling_policy: str = "ess",
        mutations_per_particle_per_stage: Optional[int] = None,
        mutation_steps_by_stage: Optional[Sequence[int]] = None,
        finite_particle_delta: float = 0.05,
        chebyshev_rho: float = 0.03,
        global_refresh_probability: float = 0.0,
        adaptive_search_evaluations: int = 0,
        adaptive_allocation_policy: str = "exp3",
        adaptive_minimum_pulls_per_type: int = 0,
        exp3_exploration: Optional[float] = None,
        search_reward_weights: Optional[SearchRewardWeights] = None,
        enable_exact_incremental_two_opt: bool = True,
        archive_tolerance: float = 1e-12,
        archive_max_size: Optional[int] = 200,
        audit_trace_level: str = "full",
        anytime_checkpoint_period: Optional[int] = None,
        domain_separated_seed: Optional[DomainSeparatedSeed] = None,
    ) -> None:
        if instance.num_cities < 4:
            raise ValueError("Annealed Pareto-SMC requires at least four cities for 2-opt mutation.")
        if particles_per_reference <= 0:
            raise ValueError("particles_per_reference must be positive.")
        if num_reference_types <= 0:
            raise ValueError("num_reference_types must be positive.")
        if not (0.0 < ess_threshold <= 1.0):
            raise ValueError("ess_threshold must lie in (0, 1].")
        if resampling_policy not in {"ess", "always"}:
            raise ValueError("resampling_policy must be 'ess' or 'always'.")
        if mutations_per_particle_per_stage is not None and (
            isinstance(mutations_per_particle_per_stage, bool)
            or not isinstance(mutations_per_particle_per_stage, int)
            or mutations_per_particle_per_stage < 0
        ):
            raise ValueError(
                "mutations_per_particle_per_stage must be a nonnegative integer."
            )
        if (
            mutations_per_particle_per_stage is not None
            and mutation_steps_by_stage is not None
        ):
            raise ValueError(
                "Use either mutations_per_particle_per_stage or "
                "mutation_steps_by_stage, not both."
            )
        if (
            not math.isfinite(float(finite_particle_delta))
            or not 0.0 < float(finite_particle_delta) < 1.0
        ):
            raise ValueError("finite_particle_delta must lie in (0, 1).")
        if not math.isfinite(float(chebyshev_rho)) or chebyshev_rho <= 0.0:
            raise ValueError("chebyshev_rho must be finite and strictly positive.")
        if (
            not math.isfinite(float(global_refresh_probability))
            or global_refresh_probability < 0.0
            or global_refresh_probability > 1.0
        ):
            raise ValueError("global_refresh_probability must lie in [0, 1].")
        if (
            isinstance(adaptive_search_evaluations, bool)
            or not isinstance(adaptive_search_evaluations, int)
            or adaptive_search_evaluations < 0
        ):
            raise ValueError("adaptive_search_evaluations must be a nonnegative integer.")
        if adaptive_allocation_policy not in {"exp3", "uniform"}:
            raise ValueError(
                "adaptive_allocation_policy must be 'exp3' or 'uniform'."
            )
        if (
            isinstance(adaptive_minimum_pulls_per_type, bool)
            or not isinstance(adaptive_minimum_pulls_per_type, int)
            or adaptive_minimum_pulls_per_type < 0
        ):
            raise ValueError(
                "adaptive_minimum_pulls_per_type must be a nonnegative integer."
            )
        if (
            adaptive_allocation_policy == "uniform"
            and adaptive_minimum_pulls_per_type != 0
        ):
            raise ValueError(
                "The uniform tail must set adaptive_minimum_pulls_per_type to zero."
            )
        if exp3_exploration is not None and (
            not math.isfinite(float(exp3_exploration))
            or not 0.0 < float(exp3_exploration) <= 1.0
        ):
            raise ValueError("exp3_exploration must lie in (0, 1].")
        if audit_trace_level not in {"full", "summary"}:
            raise ValueError(
                "audit_trace_level must be 'full' or 'summary'."
            )
        if not isinstance(enable_exact_incremental_two_opt, bool):
            raise ValueError(
                "enable_exact_incremental_two_opt must be boolean."
            )
        if (
            not math.isfinite(float(archive_tolerance))
            or float(archive_tolerance) < 0.0
        ):
            raise ValueError("archive_tolerance must be finite and nonnegative.")
        if (
            anytime_checkpoint_period is not None
            and (
                isinstance(anytime_checkpoint_period, bool)
                or not isinstance(anytime_checkpoint_period, int)
                or anytime_checkpoint_period <= 0
                or anytime_checkpoint_period > evaluations
            )
        ):
            raise ValueError(
                "anytime_checkpoint_period must be a positive integer no "
                "larger than the evaluation budget."
            )

        self.instance = instance
        self.instance_sha256 = instance_sha256(instance)
        if domain_separated_seed is not None:
            verify_domain_separated_seed(domain_separated_seed)
            if domain_separated_seed.instance_sha256 != self.instance_sha256:
                raise ValueError(
                    "The domain-separated seed is bound to a different "
                    "instance SHA-256."
                )
            if seed != domain_separated_seed.seed:
                raise ValueError(
                    "seed must exactly equal the bound domain-separated seed."
                )
        self.domain_separated_seed = domain_separated_seed
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.particles_per_reference = int(particles_per_reference)
        self.evaluations = int(evaluations)
        self.ess_threshold = float(ess_threshold)
        self.resampling_policy = resampling_policy
        self.mutations_per_particle_per_stage = (
            mutations_per_particle_per_stage
        )
        self.mutation_steps_by_stage = (
            None
            if mutation_steps_by_stage is None
            else tuple(mutation_steps_by_stage)
        )
        self.bootstrap_mutations_by_stage: Optional[Tuple[int, ...]] = None
        self.finite_particle_delta = float(finite_particle_delta)
        self.chebyshev_rho = float(chebyshev_rho)
        self.global_refresh_probability = float(global_refresh_probability)
        self.adaptive_search_evaluations = int(adaptive_search_evaluations)
        self.adaptive_allocation_policy = adaptive_allocation_policy
        self.adaptive_minimum_pulls_per_type = int(
            adaptive_minimum_pulls_per_type
        )
        self.exp3_exploration = (
            None if exp3_exploration is None else float(exp3_exploration)
        )
        self.search_reward_weights = search_reward_weights or SearchRewardWeights()
        self.enable_exact_incremental_two_opt = (
            enable_exact_incremental_two_opt
        )
        self.archive_tolerance = float(archive_tolerance)
        self.audit_trace_level = audit_trace_level
        self.anytime_checkpoint_period = anytime_checkpoint_period
        self.mutation_proposal = (
            "uniform_symmetric_two_opt"
            if self.global_refresh_probability == 0.0
            else "local_two_opt_plus_uniform_global_refresh"
        )
        # The computational paper uses three distinct output objects:
        #   * search_archive: unbounded nondominated set of every evaluated point;
        #   * deployment_archive: optional crowding-capped output for deployment;
        #   * _cell_representatives: one first witness per frozen certificate cell.
        # Only the first object is used for competitive HV/anytime reporting.
        self.search_archive = ParetoArchive(
            max_size=None,
            tol=self.archive_tolerance,
        )
        self.deployment_archive = ParetoArchive(
            max_size=archive_max_size,
            tol=self.archive_tolerance,
        )
        self.archive = self.search_archive  # Backward-compatible public result.
        self._archive_max_size = archive_max_size
        self._start_time = time.perf_counter()
        self._evaluation_counter_start = evaluation_count(instance)
        self._counted_instance = hasattr(instance, "evaluations")
        self._logical_evaluations = 0
        self._incremental_two_opt_evaluator = (
            self._resolve_exact_incremental_two_opt_evaluator()
            if self.enable_exact_incremental_two_opt
            else None
        )
        self._has_run = False

        self.beta_schedule = self._validate_beta_schedule(beta_schedule)
        if self.mutation_steps_by_stage is not None:
            if len(self.mutation_steps_by_stage) != len(
                self.beta_schedule
            ) - 1 or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in self.mutation_steps_by_stage
            ):
                raise ValueError(
                    "mutation_steps_by_stage must contain one nonnegative "
                    "integer per positive beta stage."
                )
        (
            self.objective_lower_bounds,
            self.objective_upper_bounds,
            self.bounds_source,
        ) = self._resolve_objective_bounds(
            objective_lower_bounds,
            objective_upper_bounds,
        )
        self.epsilon = self._resolve_epsilon(epsilon)
        self.cell_counts = tuple(
            max(1, int(math.ceil((upper - lower) / width)))
            for lower, upper, width in zip(
                self.objective_lower_bounds,
                self.objective_upper_bounds,
                self.epsilon,
            )
        )
        self.reference_directions = self._resolve_reference_directions(
            reference_directions,
            num_reference_types,
        )
        self.num_reference_types = len(self.reference_directions)
        self.num_particles = self.num_reference_types * self.particles_per_reference
        self.adaptive_uniform_prefix_evaluations = (
            self.num_reference_types * self.adaptive_minimum_pulls_per_type
        )
        if self.adaptive_uniform_prefix_evaluations > self.adaptive_search_evaluations:
            raise ValueError(
                "The adaptive uniform-prefix quota exceeds the adaptive tail budget."
            )
        if self.adaptive_search_evaluations > 0 and self.resampling_policy == "always":
            raise ValueError(
                "The published deterministic-bootstrap certificate cannot share "
                "a run with the adaptive IJOC search tail. Use resampling_policy='ess' "
                "or set adaptive_search_evaluations=0."
            )
        self.core_evaluations = self.evaluations - self.adaptive_search_evaluations
        if self.core_evaluations <= 0:
            raise ValueError(
                "adaptive_search_evaluations leaves no budget for the typed SMC core."
            )
        if self.resampling_policy == "always":
            if self.core_evaluations % self.num_particles != 0:
                raise ValueError(
                    "The deterministic bootstrap branch requires an evaluation "
                    "budget divisible by the total particle count, so every "
                    "particle receives the same number of mutations within "
                    "each stage."
                )
            positive_stages = len(self.beta_schedule) - 1
            total_per_particle = self.core_evaluations // self.num_particles - 1
            if total_per_particle < 0:
                raise ValueError(
                    "The deterministic bootstrap branch must evaluate every "
                    "initial particle once."
                )
            if self.mutation_steps_by_stage is not None:
                expected_per_particle = sum(self.mutation_steps_by_stage)
                if total_per_particle != expected_per_particle:
                    required = self.num_particles * (
                        1 + expected_per_particle
                    )
                    raise ValueError(
                        "The deterministic bootstrap branch requires the exact "
                        "budget M*(1+sum_l s_l); the predeclared stage counts "
                        f"require {required} evaluations."
                    )
                self.bootstrap_mutations_by_stage = (
                    self.mutation_steps_by_stage
                )
            elif self.mutations_per_particle_per_stage is None:
                base_steps, extra_stages = divmod(
                    total_per_particle,
                    positive_stages,
                )
                self.bootstrap_mutations_by_stage = tuple(
                    base_steps + int(stage < extra_stages)
                    for stage in range(positive_stages)
                )
            else:
                expected_per_particle = (
                    positive_stages
                    * self.mutations_per_particle_per_stage
                )
                if total_per_particle != expected_per_particle:
                    required = self.num_particles * (1 + expected_per_particle)
                    raise ValueError(
                        "The deterministic bootstrap branch requires the exact "
                        "budget M*(1+sum_l s_l); the requested constant stage "
                        f"count requires {required} evaluations."
                    )
                self.bootstrap_mutations_by_stage = tuple(
                    self.mutations_per_particle_per_stage
                    for _ in range(positive_stages)
                )
            self.minimum_evaluation_budget = self.num_particles
        else:
            if self.mutation_steps_by_stage is not None:
                raise ValueError(
                    "mutation_steps_by_stage applies only to deterministic "
                    "resampling at every positive stage."
                )
            self.minimum_evaluation_budget = (
                self.num_particles * len(self.beta_schedule)
            )
            if self.core_evaluations < self.minimum_evaluation_budget:
                raise ValueError(
                    "The SMC-core budget must cover initialization and at least one "
                    "complete binary64 log-domain MH "
                    "mutation per particle at every positive-beta stage "
                    f"(minimum {self.minimum_evaluation_budget})."
                )
        available = remaining_evaluations(instance)
        if available is not None and available < self.evaluations:
            raise ValueError(
                "The counting-instance budget is smaller than the requested strict SMC run."
            )

        self.context_hash = self._make_context_hash()
        self.reporting_context_hash = self._make_reporting_context_hash()
        self._adaptive_selection_seed = derive_domain_separated_seed(
            self.seed,
            context=self.context_hash,
            domain="adaptive_type_selection",
        )
        self._adaptive_environment_seed = derive_domain_separated_seed(
            self.seed,
            context=self.context_hash,
            domain="adaptive_counterfactual_environment",
        )
        self._adaptive_selection_rng = random.Random(
            self._adaptive_selection_seed
        )
        self._adaptive_environment_rng = random.Random(
            self._adaptive_environment_seed
        )
        self._stage_mutation_budgets = self._allocate_stage_mutations()
        self.run_contract_hash = self._make_run_contract_hash()
        self._particles: List[List[Tour]] = [
            [] for _ in range(self.num_reference_types)
        ]
        self._objectives: List[List[ObjectiveVector]] = [
            [] for _ in range(self.num_reference_types)
        ]
        self._energies: List[List[float]] = [
            [] for _ in range(self.num_reference_types)
        ]
        uniform_log_weight = -math.log(self.particles_per_reference)
        self._log_weights: List[List[float]] = [
            [uniform_log_weight] * self.particles_per_reference
            for _ in range(self.num_reference_types)
        ]
        self._log_normalizer_estimates: List[float] = [
            0.0 for _ in range(self.num_reference_types)
        ]
        self._stage_ledger: List[Dict[str, object]] = []
        self._diagnostics: List[Diagnostic] = []
        self._checkpoint_solution_witnesses: List[Dict[str, object]] = []
        self._resampling_events = 0
        self._mutation_attempts = 0
        self._accepted_mutations = 0
        self._max_db_log_residual = 0.0
        self._initial_population_full_tour_evaluations = 0
        self._local_two_opt_incremental_evaluations = 0
        self._local_two_opt_full_fallback_evaluations = 0
        self._global_refresh_full_tour_evaluations = 0
        self._adaptive_tail_attempts = 0
        self._adaptive_tail_accepts = 0
        self._adaptive_tail_reward_sum = 0.0
        self._adaptive_allocator: Optional[Exp3TypeAllocator] = None
        self._adaptive_uniform_cursor = 0
        self._tail_particle_cursors = [0] * self.num_reference_types
        self._adaptive_tail_ledger: List[Dict[str, object]] = []
        self._last_evaluation_new_cell = False
        self._search_nondominated_cells_ever: set[Tuple[int, ...]] = set()
        self._cell_representatives: Dict[
            Tuple[int, ...],
            ArchiveEntry,
        ] = {}

        self._initialize_population()
        self._record_initial_stage()

    def run(self) -> OptimizationResult:
        if self._has_run:
            raise RuntimeError("AnnealedParetoSMCOptimizer instances are single-use.")
        self._assert_frozen_contract()
        self._has_run = True

        for stage_index in range(1, len(self.beta_schedule)):
            self._assert_frozen_contract()
            self._run_stage(stage_index)

        # Freeze only the state required by the pre-tail certificate.  Tours are
        # deliberately excluded: retaining O(n * particles) permutations in
        # run metadata would dominate memory on the large IJOC study.
        certificate_snapshot = {
            "evaluation_end": self._evaluations_used(),
            "objectives": tuple(
                tuple(group) for group in self._objectives
            ),
            "normalized_weights_by_reference": tuple(
                tuple(math.exp(value) for value in group)
                for group in self._log_weights
            ),
            "epsilon_cells_by_reference": tuple(
                tuple(self._cell_index(objective) for objective in group)
                for group in self._objectives
            ),
        }
        certificate_snapshot_hash = self._payload_sha256(certificate_snapshot)
        certificate_snapshot_metadata = (
            certificate_snapshot
            if self.audit_trace_level == "full"
            else {
                "evaluation_end": certificate_snapshot["evaluation_end"],
                "num_reference_types": self.num_reference_types,
                "particles_per_reference": self.particles_per_reference,
                "objectives_sha256": self._payload_sha256(
                    certificate_snapshot["objectives"]
                ),
                "normalized_weights_sha256": self._payload_sha256(
                    certificate_snapshot["normalized_weights_by_reference"]
                ),
                "epsilon_cells_sha256": self._payload_sha256(
                    certificate_snapshot["epsilon_cells_by_reference"]
                ),
                "trace_compacted": True,
            }
        )
        certificate_cell_masses = tuple(
            self._cell_masses_from_state(
                objectives=certificate_snapshot["objectives"][reference_index],
                normalized_weights=certificate_snapshot[
                    "normalized_weights_by_reference"
                ][reference_index],
            )
            for reference_index in range(self.num_reference_types)
        )
        certificate_cell_representatives = tuple(
            {
                "epsilon_cell": cell,
                "tour": entry.tour,
                "objectives": entry.objectives,
            }
            for cell, entry in sorted(self._cell_representatives.items())
        )
        certificate_cell_representatives_hash = self._payload_sha256(
            certificate_cell_representatives
        )

        if self.adaptive_search_evaluations > 0:
            self._run_adaptive_search_tail()

        if self._evaluations_used() != self.evaluations:
            raise RuntimeError(
                "Strict SMC ended without consuming its exact run-local evaluation budget."
            )
        expected_anytime_checkpoints: Tuple[int, ...] = ()
        observed_anytime_checkpoints: Tuple[int, ...] = ()
        if self.anytime_checkpoint_period is not None:
            if self.evaluations % self.anytime_checkpoint_period != 0:
                raise RuntimeError(
                    "The strict IJOC anytime grid must divide the total evaluation budget."
                )
            expected_anytime_checkpoints = tuple(
                range(
                    self.anytime_checkpoint_period,
                    self.evaluations + 1,
                    self.anytime_checkpoint_period,
                )
            )
            observed_iterations = {
                diagnostic.iteration for diagnostic in self._diagnostics
            }
            missing = tuple(
                checkpoint
                for checkpoint in expected_anytime_checkpoints
                if checkpoint not in observed_iterations
            )
            if missing:
                raise RuntimeError(
                    "The strict IJOC anytime archive grid is incomplete; "
                    f"missing checkpoints={missing}."
                )
            observed_anytime_checkpoints = tuple(
                checkpoint
                for checkpoint in expected_anytime_checkpoints
                if checkpoint in observed_iterations
            )

        particles = tuple(tour for group in self._particles for tour in group)
        objectives = tuple(
            objective for group in self._objectives for objective in group
        )
        bootstrap_plan = None
        contraction_plan = None
        if self.resampling_policy == "always":
            bootstrap_plan = make_bootstrap_fk_plan(
                self.beta_schedule,
                potential_upper_bound=1.0 + self.chebyshev_rho,
                particle_count=self.particles_per_reference,
                failure_budget=(
                    self.finite_particle_delta / self.num_reference_types
                ),
            )
            assert self.bootstrap_mutations_by_stage is not None
            contraction_plan = make_contraction_aware_fk_plan(
                self.beta_schedule,
                potential_upper_bound=1.0 + self.chebyshev_rho,
                global_refresh_probability=self.global_refresh_probability,
                mutation_steps_by_stage=self.bootstrap_mutations_by_stage,
                particle_count=self.particles_per_reference,
                observable_count=self.num_reference_types,
                failure_budget=self.finite_particle_delta,
            )
        claim_level = (
            ClaimLevel.PARETO_SMC_BOOTSTRAP_BOUND.value
            if bootstrap_plan is not None
            else ClaimLevel.PARETO_SMC_MECHANICAL.value
        )
        core_diagnostics = tuple(
            diagnostic
            for diagnostic in self._diagnostics
            if diagnostic.iteration <= self.core_evaluations
        )
        full_type_sweep_verification = verify_full_type_sweep_checkpoints(
            stage_ledger=self._stage_ledger,
            num_reference_types=self.num_reference_types,
            particles_per_reference=self.particles_per_reference,
            total_evaluations=self.core_evaluations,
            checkpoint_period=(
                None
                if self.adaptive_search_evaluations > 0
                else self.anytime_checkpoint_period
            ),
            diagnostic_iterations=tuple(
                diagnostic.iteration for diagnostic in core_diagnostics
            ),
        )
        metadata: Dict[str, object] = {
            "algorithm_contract": self.contract_name,
            "implementation_version": self.implementation_version,
            "algorithm_identity": (
                "typed_fixed_schedule_interacting_pareto_smc"
                if self.particles_per_reference > 1
                else "typed_annealed_independent_mh_chain_per_type"
            ),
            "population_interaction_present": self.particles_per_reference > 1,
            "single_particle_resampling_is_identity": (
                self.particles_per_reference == 1
            ),
            "claim_level": claim_level,
            "target_family": (
                "typed_frozen_box_continuous_augmented_tchebycheff_gibbs"
            ),
            "pareto_monotonicity_scope": (
                "strict_on_objective_vectors_under_componentwise_dominance"
            ),
            "feynman_kac_increment": "exp(-(beta_s-beta_prev)*fixed_type_energy)",
            "context_hash": self.context_hash,
            "reporting_context_hash": self.reporting_context_hash,
            "run_contract_hash": self.run_contract_hash,
            "context_frozen": True,
            "context_refresh_count": 0,
            "stage_targets_frozen": True,
            "frozen_contract_checked_at_stage_boundaries": True,
            "instance_sha256": self.instance_sha256,
            "bounds_source": self.bounds_source,
            "bounds_violations_fail_closed": True,
            "normalized_objective_clipping_contract": "disabled_fail_closed_v2",
            "analytic_box_formula": (
                "outward_nextafter_of_n_times_min_and_max_offdiagonal_edge_"
                "per_objective_v2"
            ),
            "objective_lower_bounds": self.objective_lower_bounds,
            "objective_upper_bounds": self.objective_upper_bounds,
            "epsilon": self.epsilon,
            "epsilon_cell_counts": self.cell_counts,
            "epsilon_cells_predeclared": True,
            "epsilon_cells_role": (
                "external_reporting_coverage_observer_no_target_feedback"
            ),
            "target_independent_of_epsilon_cells": True,
            "reference_directions": self.reference_directions,
            "reference_types_predeclared": True,
            "particles_per_reference": self.particles_per_reference,
            "num_reference_types": self.num_reference_types,
            "num_particles": self.num_particles,
            "beta_schedule": self.beta_schedule,
            "chebyshev_rho": self.chebyshev_rho,
            "ess_threshold_fraction": self.ess_threshold,
            "resampling_policy": self.resampling_policy,
            "ess_resampling_rule": (
                "resample_every_positive_stage"
                if self.resampling_policy == "always"
                else "resample_iff_ess_strictly_below_threshold_times_type_size"
            ),
            "ess_is_not_coverage_certificate": True,
            "resampling_method": "multinomial",
            "resampling_scope": "within_fixed_reference_type_only",
            "resampling_events": self._resampling_events,
            "mutations_per_particle_per_stage": (
                self.mutations_per_particle_per_stage
            ),
            "mutation_steps_by_stage_predeclared": (
                self.mutation_steps_by_stage
            ),
            "bootstrap_mutations_by_stage": self.bootstrap_mutations_by_stage,
            "bootstrap_feynman_kac_finite_particle_certificate": (
                None
                if bootstrap_plan is None
                else {
                    "scope": (
                        "per_reference_type; deterministic multinomial "
                        "resampling at every positive stage"
                    ),
                    "beta_schedule": bootstrap_plan.beta_schedule,
                    "potential_upper_bound": (
                        bootstrap_plan.potential_upper_bound
                    ),
                    "backward_oscillation_ratios": (
                        bootstrap_plan.backward_oscillation_ratios
                    ),
                    "stability_sum": bootstrap_plan.stability_sum,
                    "per_type_mse_constant_B_L_2": (
                        bootstrap_plan.finite_particle_mse_constant
                    ),
                    "per_type_mse_statement": (
                        "E[(eta_L^m(f)-pi_L(f))^2] <= B_L^(2)/m "
                        "for every f in [0,1]"
                    ),
                    "simultaneous_per_type_error_radius": (
                        bootstrap_plan.cellwise_error_radius
                    ),
                    "simultaneous_failure_probability": (
                        self.finite_particle_delta
                    ),
                    "uniform_type_mixture_mse_constant_over_total_M": (
                        self.num_reference_types
                        * bootstrap_plan.finite_particle_mse_constant
                    ),
                    "coverage_gate_requires_external_p_min": True,
                    "adaptive_ess_branch_covered": False,
                }
            ),
            "contraction_aware_fixed_schedule_certificate": (
                None
                if contraction_plan is None
                else {
                    **contraction_plan.__dict__,
                    "scope": (
                        "simultaneous_one_fixed_observable_per_reference_type; "
                        "deterministic_schedule_and_mutation_counts"
                    ),
                    "observable_selection": (
                        "must_be_frozen_independently_of_the_certified_stream"
                    ),
                    "adaptive_ess_branch_covered": False,
                    "pilot_confirm_reference_front_supported": True,
                }
            ),
            "proposal": self.mutation_proposal,
            "proposal_symmetric": True,
            "global_refresh_probability": self.global_refresh_probability,
            "global_refresh_base_measure": (
                "uniform_fixed_zero_tours"
                if self.global_refresh_probability > 0.0
                else None
            ),
            "stage_doeblin_minorization": tuple(
                self.global_refresh_probability
                * math.exp(-beta * (1.0 + self.chebyshev_rho))
                for beta in self.beta_schedule
            ),
            "stage_dobrushin_contraction": tuple(
                1.0
                - self.global_refresh_probability
                * math.exp(-beta * (1.0 + self.chebyshev_rho))
                for beta in self.beta_schedule
            ),
            "proposal_log_ratio": 0.0,
            "acceptance_computation": "log_uniform_comparison",
            "objective_evaluation_contract": (
                (
                    "initial_full_tour_local_exact_incremental_with_fail_safe_"
                    "full_fallback_global_refresh_full_tour"
                )
                if self.enable_exact_incremental_two_opt
                else "full_tour_all_proposals_v1"
            ),
            "initial_population_evaluation_contract": "full_tour_state_function",
            "local_two_opt_evaluation_contract": (
                (
                    "symmetric_nonnegative_integer_binary64_safe_delta_else_"
                    "full_tour"
                )
                if self.enable_exact_incremental_two_opt
                else "full_tour"
            ),
            "local_two_opt_incremental_exactness_scope": (
                (
                    "bitwise_full_sum_equivalence_when_each_objective_has "
                    "nonnegative_integer_edges_and_n_times_max_edge_le_2_pow_53"
                )
                if self.enable_exact_incremental_two_opt
                else None
            ),
            "exact_incremental_two_opt_requested": (
                self.enable_exact_incremental_two_opt
            ),
            "local_two_opt_incremental_enabled": (
                self._incremental_two_opt_evaluator is not None
            ),
            "global_refresh_evaluation_contract": "full_tour_state_function",
            "mutation_attempts": self._mutation_attempts,
            "accepted_mutations": self._accepted_mutations,
            "mutation_acceptance_rate": self._accepted_mutations
            / max(1, self._mutation_attempts),
            "db_max_abs_log_residual_real_arithmetic_identity": self._max_db_log_residual,
            "machine_exact_detailed_balance_claimed": False,
            "detailed_balance_scope": "ideal_real_arithmetic_kernel_only",
            "rng_contract": "python_random_mt19937_finite_precision",
            "archive_role": "reporting_only_no_smc_feedback",
            "archive_feedback": False,
            "archive_kernel_reads": 0,
            "competitive_search_archive_contract": (
                "unbounded_exact_nondominated_union_of_all_evaluated_candidates_v2"
            ),
            "competitive_search_archive_size": len(self.search_archive),
            "competitive_search_archive_dominance_tolerance": (
                self.archive_tolerance
            ),
            "search_nondominated_cells_ever_count": len(
                self._search_nondominated_cells_ever
            ),
            "adaptive_new_cell_reward_semantics": (
                "first_nondominated_candidate_in_frozen_cell_v1"
            ),
            "deployment_archive_contract": (
                "crowding_capped_exact_nondominated_view_of_all_evaluated_candidates_v2"
            ),
            "deployment_archive_max_size": self._archive_max_size,
            "deployment_archive_size": len(self.deployment_archive),
            "deployment_archive_dominance_tolerance": self.archive_tolerance,
            "deployment_archive_entries": tuple(
                {
                    "tour": entry.tour,
                    "objectives": entry.objectives,
                }
                for entry in self.deployment_archive.entries
            ),
            "archive_max_size": None,
            "anytime_checkpoint_period": (
                self.anytime_checkpoint_period
            ),
            "anytime_checkpoint_emission_contract": (
                "per_evaluation_passive_archive_snapshot_v2"
                if self.anytime_checkpoint_period is not None
                else "disabled"
            ),
            "anytime_checkpoint_grid_gate": (
                "PASS" if self.anytime_checkpoint_period is not None else "NOT_RUN"
            ),
            "expected_anytime_checkpoints": expected_anytime_checkpoints,
            "observed_anytime_checkpoints": observed_anytime_checkpoints,
            "anytime_checkpoint_grid_complete": (
                expected_anytime_checkpoints == observed_anytime_checkpoints
            ),
            "checkpoint_solution_witnesses": tuple(
                self._checkpoint_solution_witnesses
            ),
            **full_type_sweep_verification.metadata(),
            "audit_trace_level": self.audit_trace_level,
            "full_per_mutation_trace_recorded": (
                self.audit_trace_level == "full"
            ),
            "cell_observer_role": "reporting_only_one_first_query_per_epsilon_cell",
            "cell_observer_feedback": False,
            "cell_observer_kernel_reads": 0,
            "all_evaluated_epsilon_cell_count": len(self._cell_representatives),
            "all_evaluated_cell_representatives": tuple(
                {
                    "epsilon_cell": cell,
                    "tour": entry.tour,
                    "objectives": entry.objectives,
                }
                for cell, entry in sorted(self._cell_representatives.items())
            ),
            "certificate_cell_representatives_before_adaptive_tail": (
                certificate_cell_representatives
            ),
            "certificate_cell_representatives_hash": (
                certificate_cell_representatives_hash
            ),
            "certificate_cell_representatives_exclude_adaptive_tail": True,
            "queried_epsilon_cell_count": len(
                certificate_cell_representatives
            ),
            "cell_representatives": certificate_cell_representatives,
            "cell_representative_tours": tuple(
                row["tour"] for row in certificate_cell_representatives
            ),
            "cell_representative_objectives": tuple(
                row["objectives"] for row in certificate_cell_representatives
            ),
            "ordinary_igd_same_cell_support": (
                "terminal_weighted_support_or_reporting_cell_observer"
            ),
            "ordinary_igd_same_cell_support_scope": (
                "pre_adaptive_tail_only"
            ),
            "pareto_archive_metric_scope": (
                "additive_igd_plus_and_hypervolume_not_unconditional_ordinary_igd"
            ),
            "stage_ledger": tuple(self._stage_ledger),
            "stage_ledger_hash": self._payload_sha256(self._stage_ledger),
            "certificate_snapshot_before_adaptive_tail": certificate_snapshot_metadata,
            "certificate_snapshot_hash": certificate_snapshot_hash,
            "certificate_snapshot_excludes_adaptive_search_tail": True,
            "final_normalized_weights_by_reference": tuple(
                tuple(math.exp(value) for value in group)
                for group in self._log_weights
            ),
            "final_log_normalizer_estimates_by_reference": tuple(
                self._log_normalizer_estimates
            ),
            "final_epsilon_cells_by_reference": tuple(
                tuple(cells)
                for cells in certificate_snapshot["epsilon_cells_by_reference"]
            ),
            "final_epsilon_cell_masses_by_reference": tuple(
                certificate_cell_masses
            ),
            "final_weighted_support_cell_counts": tuple(
                len(group) for group in certificate_cell_masses
            ),
            "search_final_epsilon_cells_by_reference": tuple(
                tuple(self._cell_index(objective) for objective in group)
                for group in self._objectives
            ),
            "finite_particle_coverage_observable": (
                "positive_terminal_weight_mass_per_predeclared_epsilon_cell"
            ),
            "seed": self.seed,
            "domain_separated_seed_gate": (
                "PASS"
                if self.domain_separated_seed is not None
                else "NOT_RUN"
            ),
            "domain_separated_seed_contract": (
                None
                if self.domain_separated_seed is None
                else self.domain_separated_seed.metadata()
            ),
            "evaluation_counter_start": self._evaluation_counter_start,
            "evaluation_budget": self.evaluations,
            "smc_core_evaluation_budget": self.core_evaluations,
            "adaptive_search_evaluation_budget": self.adaptive_search_evaluations,
            "adaptive_search_allocation_policy": self.adaptive_allocation_policy,
            "adaptive_search_minimum_pulls_per_type": (
                self.adaptive_minimum_pulls_per_type
            ),
            "adaptive_search_uniform_prefix_evaluations": (
                self.adaptive_uniform_prefix_evaluations
            ),
            "adaptive_search_exp3_suffix_evaluations": (
                self.adaptive_search_evaluations
                - self.adaptive_uniform_prefix_evaluations
            ),
            "adaptive_search_tail_attempts": self._adaptive_tail_attempts,
            "adaptive_search_tail_accepts": self._adaptive_tail_accepts,
            "adaptive_search_tail_reward_sum": self._adaptive_tail_reward_sum,
            "adaptive_search_reward_weights": {
                "hypervolume": self.search_reward_weights.hypervolume,
                "new_cell": self.search_reward_weights.new_cell,
                "scalar_improvement": self.search_reward_weights.scalar_improvement,
            },
            "adaptive_search_allocator": (
                None
                if self._adaptive_allocator is None
                else self._adaptive_allocator.metadata()
            ),
            "adaptive_search_total_external_regret_upper_bound": (
                None
                if self.adaptive_allocation_policy != "exp3"
                else min(
                    float(self.adaptive_search_evaluations),
                    float(self.adaptive_uniform_prefix_evaluations)
                    + (
                        0.0
                        if self._adaptive_allocator is None
                        else self._adaptive_allocator.regret_upper_bound()
                    ),
                )
            ),
            "adaptive_search_regret_decomposition": (
                "uniform_prefix_trivial_regret_plus_exp3_suffix_external_regret_v1"
                if self.adaptive_allocation_policy == "exp3"
                else None
            ),
            "adaptive_search_tail_ledger": tuple(self._adaptive_tail_ledger),
            "adaptive_search_tail_ledger_hash": self._payload_sha256(
                self._adaptive_tail_ledger
            ),
            "adaptive_search_regret_scope": (
                "observable_bounded_tail_reward_not_final_hypervolume_or_igd_regret"
            ),
            "adaptive_search_counterfactual_reward_contract": (
                "private_per_type_random_tapes_sampled_before_arm_selection_v1"
            ),
            "adaptive_search_rng_domain_separation_contract": (
                "independent_mutable_rng_states_for_selection_and_environment_v1"
            ),
            "adaptive_search_selection_seed_sha256": hashlib.sha256(
                str(self._adaptive_selection_seed).encode("ascii")
            ).hexdigest(),
            "adaptive_search_environment_seed_sha256": hashlib.sha256(
                str(self._adaptive_environment_seed).encode("ascii")
            ).hexdigest(),
            "minimum_evaluation_budget": self.minimum_evaluation_budget,
            "stage_mutation_budgets": self._stage_mutation_budgets,
            "initial_population_evaluations": self.num_particles,
            "mutation_evaluations": self._mutation_attempts,
            "initial_population_full_tour_evaluations": (
                self._initial_population_full_tour_evaluations
            ),
            "local_two_opt_incremental_evaluations": (
                self._local_two_opt_incremental_evaluations
            ),
            "local_two_opt_full_fallback_evaluations": (
                self._local_two_opt_full_fallback_evaluations
            ),
            "local_two_opt_proposal_evaluations": (
                self._local_two_opt_incremental_evaluations
                + self._local_two_opt_full_fallback_evaluations
            ),
            "global_refresh_full_tour_evaluations": (
                self._global_refresh_full_tour_evaluations
            ),
            "global_refresh_proposal_evaluations": (
                self._global_refresh_full_tour_evaluations
            ),
            "mutation_evaluation_path_accounting_complete": (
                self._local_two_opt_incremental_evaluations
                + self._local_two_opt_full_fallback_evaluations
                + self._global_refresh_full_tour_evaluations
                == self._mutation_attempts
            ),
            "full_tour_evaluations": (
                self._initial_population_full_tour_evaluations
                + self._local_two_opt_full_fallback_evaluations
                + self._global_refresh_full_tour_evaluations
            ),
            "evaluation_path_accounting_complete": (
                self._initial_population_full_tour_evaluations
                + self._local_two_opt_incremental_evaluations
                + self._local_two_opt_full_fallback_evaluations
                + self._global_refresh_full_tour_evaluations
                == self._evaluations_used()
            ),
            "evaluations_used": self._evaluations_used(),
        }
        return OptimizationResult(
            particles=particles,
            objectives=objectives,
            archive=self.archive,
            diagnostics=tuple(self._diagnostics),
            metadata=metadata,
        )

    def _initialize_population(self) -> None:
        for reference_index in range(self.num_reference_types):
            for _ in range(self.particles_per_reference):
                tour = random_tour(self.instance.num_cities, self.rng)
                objective = self._evaluate(
                    tour,
                    evaluation_kind="initial_population_full_tour",
                )
                self._particles[reference_index].append(tour)
                self._objectives[reference_index].append(objective)
                self._energies[reference_index].append(
                    self._energy(objective, reference_index)
                )
                entry = ArchiveEntry(tour, objective)
                self._update_output_archives(entry)
                self._maybe_log_diagnostic(self.beta_schedule[0])

    def _record_initial_stage(self) -> None:
        references = []
        for reference_index, direction in enumerate(self.reference_directions):
            if self.audit_trace_level == "full":
                reference_record: Dict[str, object] = {
                    "reference_index": reference_index,
                    "reference_direction": direction,
                    "particle_tours": tuple(
                        self._particles[reference_index]
                    ),
                    "particle_objectives": tuple(
                        self._objectives[reference_index]
                    ),
                    "epsilon_cells": tuple(
                        self._cell_index(objective)
                        for objective in self._objectives[reference_index]
                    ),
                    "occupied_epsilon_cell_count_before_weighting": len(
                        {
                            self._cell_index(objective)
                            for objective in self._objectives[reference_index]
                        }
                    ),
                    "pre_weight_energies": tuple(
                        self._energies[reference_index]
                    ),
                    "incremental_log_weights": tuple(
                        0.0 for _ in range(self.particles_per_reference)
                    ),
                    "incremental_log_weight_min": 0.0,
                    "incremental_log_weight_max": 0.0,
                    "log_normalizer_increment": 0.0,
                    "cumulative_log_normalizer_estimate": 0.0,
                    "normalized_weights_before_increment": tuple(
                        1.0 / self.particles_per_reference
                        for _ in range(self.particles_per_reference)
                    ),
                    "normalized_weights_before_resampling": tuple(
                        1.0 / self.particles_per_reference
                        for _ in range(self.particles_per_reference)
                    ),
                    "ess_before_resampling": float(self.particles_per_reference),
                    "ess_after_resampling": float(self.particles_per_reference),
                    "resampled": False,
                    "resampling_method": "none",
                    "ancestor_indices": tuple(
                        range(self.particles_per_reference)
                    ),
                    "unique_ancestors": self.particles_per_reference,
                    "normalized_weights_after_resampling": tuple(
                        1.0 / self.particles_per_reference
                        for _ in range(self.particles_per_reference)
                    ),
                    "mutation_attempts": 0,
                    "accepted_mutations": 0,
                    "mutations": (),
                }
            else:
                occupied_cells = {
                    self._cell_index(objective)
                    for objective in self._objectives[reference_index]
                }
                reference_record = {
                    "reference_index": reference_index,
                    "reference_direction": direction,
                    "particle_count": self.particles_per_reference,
                    "occupied_epsilon_cell_count_before_weighting": len(
                        occupied_cells
                    ),
                    "incremental_log_weight_min": 0.0,
                    "incremental_log_weight_max": 0.0,
                    "log_normalizer_increment": 0.0,
                    "cumulative_log_normalizer_estimate": 0.0,
                    "ess_before_resampling": float(
                        self.particles_per_reference
                    ),
                    "ess_after_resampling": float(
                        self.particles_per_reference
                    ),
                    "resampled": False,
                    "resampling_method": "none",
                    "unique_ancestors": self.particles_per_reference,
                    "mutation_attempts": 0,
                    "accepted_mutations": 0,
                    "mutations": (),
                    "trace_compacted": True,
                }
            references.append(reference_record)
        self._stage_ledger.append(
            {
                "stage_index": 0,
                "beta_previous": 0.0,
                "beta": self.beta_schedule[0],
                "delta_beta": 0.0,
                "target_frozen_during_stage": True,
                "stage_target_hash": self._stage_target_hash(0),
                "evaluation_start": 0,
                "evaluation_end": self._evaluations_used(),
                "evaluations": self._evaluations_used(),
                "references": tuple(references),
            }
        )
        self._log_diagnostic(self.beta_schedule[0])

    def _run_stage(self, stage_index: int) -> None:
        beta_previous = self.beta_schedule[stage_index - 1]
        beta = self.beta_schedule[stage_index]
        delta_beta = beta - beta_previous
        evaluation_start = self._evaluations_used()
        stage_budget = self._stage_mutation_budgets[stage_index - 1]
        base_attempts, extra_groups = divmod(
            stage_budget,
            self.num_reference_types,
        )
        reference_ledgers: List[Dict[str, object]] = []

        for reference_index, direction in enumerate(self.reference_directions):
            pre_weight_energies = tuple(self._energies[reference_index])
            full_trace = self.audit_trace_level == "full"
            pre_weight_objectives = (
                tuple(self._objectives[reference_index])
                if full_trace
                else ()
            )
            pre_weight_occupied_cell_count = len(
                {
                    self._cell_index(objective)
                    for objective in self._objectives[reference_index]
                }
            )
            probabilities_before_increment = (
                tuple(
                    math.exp(value)
                    for value in self._log_weights[reference_index]
                )
                if full_trace
                else ()
            )
            incremental = [
                -delta_beta * energy
                for energy in pre_weight_energies
            ]
            unnormalized = [
                old + increment
                for old, increment in zip(
                    self._log_weights[reference_index],
                    incremental,
                )
            ]
            log_normalizer = self._logsumexp(unnormalized)
            self._log_normalizer_estimates[reference_index] += log_normalizer
            normalized = [value - log_normalizer for value in unnormalized]
            probabilities_before_resampling = (
                tuple(math.exp(value) for value in normalized)
                if full_trace
                else ()
            )
            ess = self._effective_sample_size(normalized)
            resampled = (
                True
                if self.resampling_policy == "always"
                else ess < self.ess_threshold * self.particles_per_reference
            )
            unique_ancestors = self.particles_per_reference
            ancestors = list(range(self.particles_per_reference))
            if resampled:
                ancestors = self._multinomial_indices(normalized)
                unique_ancestors = len(set(ancestors))
                self._particles[reference_index] = [
                    self._particles[reference_index][index]
                    for index in ancestors
                ]
                self._objectives[reference_index] = [
                    self._objectives[reference_index][index]
                    for index in ancestors
                ]
                self._energies[reference_index] = [
                    self._energies[reference_index][index]
                    for index in ancestors
                ]
                normalized = [
                    -math.log(self.particles_per_reference)
                ] * self.particles_per_reference
                self._resampling_events += 1
            self._log_weights[reference_index] = normalized

            attempts = base_attempts + int(reference_index < extra_groups)
            accepted, mutations, _ = self._mutate_reference(
                reference_index,
                beta,
                attempts,
            )
            if full_trace:
                reference_record = {
                    "reference_index": reference_index,
                    "reference_direction": direction,
                    "particle_objectives_before_weighting": pre_weight_objectives,
                    "epsilon_cells_before_weighting": tuple(
                        self._cell_index(objective)
                        for objective in pre_weight_objectives
                    ),
                    "occupied_epsilon_cell_count_before_weighting": (
                        pre_weight_occupied_cell_count
                    ),
                    "pre_weight_energies": pre_weight_energies,
                    "incremental_log_weights": tuple(incremental),
                    "incremental_log_weight_min": min(incremental),
                    "incremental_log_weight_max": max(incremental),
                    "log_normalizer_increment": log_normalizer,
                    "cumulative_log_normalizer_estimate": (
                        self._log_normalizer_estimates[reference_index]
                    ),
                    "normalized_weights_before_increment": probabilities_before_increment,
                    "normalized_weights_before_resampling": probabilities_before_resampling,
                    "ess_before_resampling": ess,
                    "ess_after_resampling": self._effective_sample_size(normalized),
                    "resampled": resampled,
                    "resampling_method": (
                        "multinomial" if resampled else "none"
                    ),
                    "ancestor_indices": tuple(ancestors),
                    "unique_ancestors": unique_ancestors,
                    "normalized_weights_after_resampling": tuple(
                        math.exp(value) for value in normalized
                    ),
                    "mutation_attempts": attempts,
                    "accepted_mutations": accepted,
                    "mutations": mutations,
                }
            else:
                reference_record = {
                    "reference_index": reference_index,
                    "reference_direction": direction,
                    "particle_count": self.particles_per_reference,
                    "occupied_epsilon_cell_count_before_weighting": (
                        pre_weight_occupied_cell_count
                    ),
                    "incremental_log_weight_min": min(incremental),
                    "incremental_log_weight_max": max(incremental),
                    "log_normalizer_increment": log_normalizer,
                    "cumulative_log_normalizer_estimate": (
                        self._log_normalizer_estimates[reference_index]
                    ),
                    "ess_before_resampling": ess,
                    "ess_after_resampling": self._effective_sample_size(
                        normalized
                    ),
                    "resampled": resampled,
                    "resampling_method": (
                        "multinomial" if resampled else "none"
                    ),
                    "unique_ancestors": unique_ancestors,
                    "mutation_attempts": attempts,
                    "accepted_mutations": accepted,
                    "mutations": (),
                    "trace_compacted": True,
                }
            reference_ledgers.append(reference_record)

        evaluation_end = self._evaluations_used()
        self._stage_ledger.append(
            {
                "stage_index": stage_index,
                "beta_previous": beta_previous,
                "beta": beta,
                "delta_beta": delta_beta,
                "target_frozen_during_stage": True,
                "stage_target_hash": self._stage_target_hash(stage_index),
                "evaluation_start": evaluation_start,
                "evaluation_end": evaluation_end,
                "evaluations": evaluation_end - evaluation_start,
                "references": tuple(reference_ledgers),
            }
        )
        self._log_diagnostic(beta)

    def _run_adaptive_search_tail(self) -> None:
        """Spend the post-certificate budget on adaptive typed search.

        The SMC state at the end of the final Feynman--Kac stage is recorded
        before this method is called.  The tail may improve the competitive
        search archive, but it is never read by the finite-particle certificate.
        """

        beta = self.beta_schedule[-1]
        adaptive_suffix_evaluations = (
            self.adaptive_search_evaluations
            - self.adaptive_uniform_prefix_evaluations
        )
        if (
            self.adaptive_allocation_policy == "exp3"
            and adaptive_suffix_evaluations > 0
        ):
            exploration = (
                self.exp3_exploration
                if self.exp3_exploration is not None
                else Exp3TypeAllocator.recommended_exploration(
                    self.num_reference_types,
                    adaptive_suffix_evaluations,
                )
            )
            self._adaptive_allocator = Exp3TypeAllocator(
                self.num_reference_types,
                exploration=exploration,
            )

        for tail_index in range(self.adaptive_search_evaluations):
            # Define the full counterfactual reward vector before drawing the
            # arm: every type receives an independent private proposal tape,
            # but only the selected tape is evaluated.  This is the
            # nonanticipation interface required by the EXP3 regret theorem.
            arm_rngs = tuple(
                random.Random(
                    self._adaptive_environment_rng.getrandbits(256)
                )
                for _ in range(self.num_reference_types)
            )
            in_uniform_prefix = (
                self.adaptive_allocation_policy == "exp3"
                and tail_index < self.adaptive_uniform_prefix_evaluations
            )
            if self._adaptive_allocator is None or in_uniform_prefix:
                reference_index = (
                    self._adaptive_uniform_cursor % self.num_reference_types
                )
                selection_probability = 1.0
                self._adaptive_uniform_cursor += 1
            else:
                reference_index, selection_probability = (
                    self._adaptive_allocator.select(
                        self._adaptive_selection_rng
                    )
                )

            particle_offset = self._tail_particle_cursors[reference_index]
            accepted, mutations, rewards = self._mutate_reference(
                reference_index,
                beta,
                1,
                particle_offset=particle_offset,
                rng=arm_rngs[reference_index],
            )
            self._tail_particle_cursors[reference_index] = (
                particle_offset + 1
            ) % self.particles_per_reference
            reward = rewards[0]
            if self._adaptive_allocator is not None and not in_uniform_prefix:
                self._adaptive_allocator.observe(
                    reference_index,
                    reward,
                    selection_probability,
                )
            self._adaptive_tail_attempts += 1
            self._adaptive_tail_accepts += accepted
            self._adaptive_tail_reward_sum += reward
            if self.audit_trace_level == "full":
                self._adaptive_tail_ledger.append(
                    {
                        "tail_index": tail_index,
                        "evaluation": self._evaluations_used(),
                        "reference_index": reference_index,
                        "selection_probability": selection_probability,
                        "allocation_phase": (
                            "uniform_prefix" if in_uniform_prefix else self.adaptive_allocation_policy
                        ),
                        "reward": reward,
                        "accepted": bool(accepted),
                        "mutation": mutations[0] if mutations else None,
                    }
                )

        self._log_diagnostic(beta)

    def _mutate_reference(
        self,
        reference_index: int,
        beta: float,
        attempts: int,
        *,
        particle_offset: int = 0,
        rng: Optional[random.Random] = None,
    ) -> Tuple[int, Tuple[Dict[str, object], ...], Tuple[float, ...]]:
        accepted_count = 0
        mutations: List[Dict[str, object]] = []
        rewards: List[float] = []
        proposal_rng = self.rng if rng is None else rng
        for attempt in range(attempts):
            particle_index = (
                particle_offset + attempt
            ) % self.particles_per_reference
            current_tour = self._particles[reference_index][particle_index]
            current_objective = self._objectives[reference_index][
                particle_index
            ]
            current_energy = self._energies[reference_index][particle_index]
            # Gamma zero is the legacy pure-two-opt kernel. Avoid a mixture
            # selection draw so its seeded random stream remains unchanged.
            use_global_refresh = (
                self.global_refresh_probability > 0.0
                and proposal_rng.random() < self.global_refresh_probability
            )
            hv_before = self._search_hypervolume_2d()
            if use_global_refresh:
                i = None
                j = None
                proposed_tour = random_tour(len(current_tour), proposal_rng)
                proposal_kind = "uniform_global_refresh"
                proposed_objective = self._evaluate(
                    proposed_tour,
                    evaluation_kind="global_refresh_full_tour",
                )
                objective_evaluation_kind = "full_tour_global_refresh"
            else:
                i, j = sample_two_opt_indices(len(current_tour), proposal_rng)
                proposed_tour = two_opt_at(current_tour, i, j)
                proposal_kind = "uniform_symmetric_two_opt"
                (
                    proposed_objective,
                    objective_evaluation_kind,
                ) = self._evaluate_local_two_opt(
                    current_tour=current_tour,
                    current_objective=current_objective,
                    proposed_tour=proposed_tour,
                    i=i,
                    j=j,
                )
            proposed_energy = self._energy(proposed_objective, reference_index)
            entry = ArchiveEntry(proposed_tour, proposed_objective)
            candidate_nondominated, new_nondominated_cell = (
                self._update_output_archives(entry)
            )
            hv_after = self._search_hypervolume_2d()

            delta_energy = proposed_energy - current_energy
            log_alpha_forward = min(0.0, -beta * delta_energy)
            log_alpha_reverse = min(0.0, beta * delta_energy)
            residual = abs(
                beta * delta_energy
                + log_alpha_forward
                - log_alpha_reverse
            )
            self._max_db_log_residual = max(
                self._max_db_log_residual,
                residual,
            )
            uniform_draw = proposal_rng.random()
            log_uniform = (
                -math.inf
                if uniform_draw == 0.0
                else math.log(uniform_draw)
            )
            accepted = log_uniform < log_alpha_forward
            self._mutation_attempts += 1
            if accepted:
                self._particles[reference_index][particle_index] = proposed_tour
                self._objectives[reference_index][particle_index] = proposed_objective
                self._energies[reference_index][particle_index] = proposed_energy
                self._accepted_mutations += 1
                accepted_count += 1
            reward = self.search_reward_weights.combine(
                normalized_hypervolume_gain=normalized_hypervolume_gain(
                    hv_before,
                    hv_after,
                    objective_box_volume=self._objective_box_volume(),
                ),
                new_cell=new_nondominated_cell,
                normalized_scalar_improvement=(
                    max(0.0, current_energy - proposed_energy)
                    / (1.0 + self.chebyshev_rho)
                ),
            )
            rewards.append(reward)
            if self.audit_trace_level == "full":
                mutations.append(
                    {
                    "mutation_index": attempt,
                    "particle_index": particle_index,
                    "proposal_kind": proposal_kind,
                    "objective_evaluation_kind": objective_evaluation_kind,
                    "two_opt_i": i,
                    "two_opt_j": j,
                    "current_tour": current_tour,
                    "proposed_tour": proposed_tour,
                    "current_objective": current_objective,
                    "proposed_objective": proposed_objective,
                    "current_epsilon_cell": self._cell_index(current_objective),
                    "proposed_epsilon_cell": self._cell_index(proposed_objective),
                    "current_energy": current_energy,
                    "proposed_energy": proposed_energy,
                    "delta_energy": delta_energy,
                    "beta": beta,
                    "log_alpha": log_alpha_forward,
                    "log_uniform": (
                        log_uniform if math.isfinite(log_uniform) else "-inf"
                    ),
                    "accepted": accepted,
                    "candidate_nondominated_after_insertion": candidate_nondominated,
                    "new_nondominated_cell": new_nondominated_cell,
                    "search_reward": reward,
                    }
                )
            self._maybe_log_diagnostic(beta)
        return accepted_count, tuple(mutations), tuple(rewards)

    def _evaluate(
        self,
        tour: Tour,
        *,
        evaluation_kind: str,
    ) -> ObjectiveVector:
        objective = self._record_evaluated_objective(
            tour,
            self.instance.evaluate(tour),
        )
        if evaluation_kind == "initial_population_full_tour":
            self._initial_population_full_tour_evaluations += 1
        elif evaluation_kind == "local_two_opt_full_fallback":
            self._local_two_opt_full_fallback_evaluations += 1
        elif evaluation_kind == "global_refresh_full_tour":
            self._global_refresh_full_tour_evaluations += 1
        else:
            raise ValueError(f"Unknown objective evaluation kind: {evaluation_kind!r}")
        return objective

    def _evaluate_local_two_opt(
        self,
        *,
        current_tour: Tour,
        current_objective: ObjectiveVector,
        proposed_tour: Tour,
        i: int,
        j: int,
    ) -> Tuple[ObjectiveVector, str]:
        method = self._incremental_two_opt_evaluator
        if method is not None:
            try:
                raw_objective = method(
                    current_tour,
                    current_objective,
                    i,
                    j,
                )
            except NotImplementedError:
                return self._evaluate_unsupported_local_two_opt(
                    proposed_tour,
                )
            if raw_objective is NotImplemented:
                return self._evaluate_unsupported_local_two_opt(
                    proposed_tour,
                )
            objective = self._record_evaluated_objective(
                proposed_tour,
                raw_objective,
            )
            self._local_two_opt_incremental_evaluations += 1
            return objective, "exact_incremental_two_opt"

        objective = self._evaluate(
            proposed_tour,
            evaluation_kind="local_two_opt_full_fallback",
        )
        return objective, "full_tour_local_two_opt_fallback"

    def _evaluate_unsupported_local_two_opt(
        self,
        proposed_tour: Tour,
    ) -> Tuple[ObjectiveVector, str]:
        self._incremental_two_opt_evaluator = None
        if isinstance(self.instance, CountingTSPInstance):
            objective = self._record_evaluated_objective(
                proposed_tour,
                self.instance.base.evaluate(proposed_tour),
            )
            self._local_two_opt_full_fallback_evaluations += 1
        else:
            objective = self._evaluate(
                proposed_tour,
                evaluation_kind="local_two_opt_full_fallback",
            )
        return objective, "full_tour_local_two_opt_fallback"

    def _resolve_exact_incremental_two_opt_evaluator(
        self,
    ) -> Optional[TwoOptEvaluator]:
        evaluation_source = (
            self.instance.base
            if isinstance(self.instance, CountingTSPInstance)
            else self.instance
        )
        source_method = getattr(evaluation_source, "evaluate_two_opt", None)
        interface_method = getattr(self.instance, "evaluate_two_opt", None)
        symmetric = getattr(evaluation_source, "symmetric_objectives", None)
        exact_binary64 = getattr(
            evaluation_source,
            "exact_two_opt_delta_in_binary64",
            False,
        )
        try:
            symmetric_flags = tuple(symmetric)
        except TypeError:
            return None
        if (
            callable(source_method)
            and callable(interface_method)
            and exact_binary64 is True
            and len(symmetric_flags) == self.instance.num_objectives
            and all(flag is True for flag in symmetric_flags)
        ):
            return interface_method
        return None

    def _update_output_archives(
        self,
        entry: ArchiveEntry,
    ) -> Tuple[bool, bool]:
        """Update search/deployment archives and classify search novelty."""
        self.search_archive.update((entry,))
        self.deployment_archive.update((entry,))
        candidate_nondominated = self.search_archive.contains(entry)
        cell = self._cell_index(entry.objectives)
        new_nondominated_cell = (
            candidate_nondominated
            and cell not in self._search_nondominated_cells_ever
        )
        if candidate_nondominated:
            self._search_nondominated_cells_ever.add(cell)
        return candidate_nondominated, new_nondominated_cell

    def _record_evaluated_objective(
        self,
        tour: Tour,
        raw_objective: Sequence[float],
    ) -> ObjectiveVector:
        objective = self._validate_objective_vector(raw_objective)
        if not self._counted_instance:
            self._logical_evaluations += 1
        cell = self._cell_index(objective)
        self._last_evaluation_new_cell = cell not in self._cell_representatives
        self._cell_representatives.setdefault(
            cell,
            ArchiveEntry(tour=tour, objectives=objective),
        )
        return objective

    def _objective_box_volume(self) -> float:
        volume = 1.0
        for lower, upper in zip(
            self.objective_lower_bounds,
            self.objective_upper_bounds,
        ):
            volume *= upper - lower
        if not math.isfinite(volume) or volume <= 0.0:
            raise RuntimeError("The frozen objective box has nonpositive volume.")
        return volume

    def _search_hypervolume_2d(self) -> float:
        if self.instance.num_objectives != 2:
            return 0.0
        return self.search_archive.hypervolume_2d(
            reference=self.objective_upper_bounds
        )

    def target_energy(
        self,
        objective: Sequence[float],
        *,
        reference_index: int,
    ) -> float:
        """Evaluate the frozen continuous target energy without changing state."""
        if reference_index < 0 or reference_index >= self.num_reference_types:
            raise IndexError("reference_index is out of range.")
        validated = self._validate_objective_vector(objective)
        return self._energy(validated, reference_index)

    def _validate_objective_vector(
        self,
        objective: Sequence[float],
    ) -> ObjectiveVector:
        values = tuple(float(value) for value in objective)
        if len(values) != self.instance.num_objectives:
            raise ValueError("Objective evaluation has the wrong dimension.")
        for index, (value, lower, upper) in enumerate(
            zip(
                values,
                self.objective_lower_bounds,
                self.objective_upper_bounds,
            )
        ):
            if (
                not math.isfinite(value)
                or value < lower
                or value > upper
            ):
                raise ObjectiveBoundsViolation(
                    f"objective {index} value {value!r} leaves the predeclared "
                    f"box [{lower!r}, {upper!r}]"
                )
        return values

    def _cell_index(self, objective: ObjectiveVector) -> Tuple[int, ...]:
        cells = []
        for coordinate, (value, lower, upper, width, count) in enumerate(
            zip(
            objective,
            self.objective_lower_bounds,
            self.objective_upper_bounds,
            self.epsilon,
            self.cell_counts,
            )
        ):
            if not math.isfinite(value) or value < lower or value > upper:
                raise ObjectiveBoundsViolation(
                    f"objective {coordinate} value {value!r} leaves the "
                    f"predeclared box [{lower!r}, {upper!r}]"
                )
            raw = (
                count - 1
                if value == upper
                else int(math.floor((value - lower) / width))
            )
            if raw < 0 or raw >= count:
                raise ObjectiveBoundsViolation(
                    f"objective {coordinate} produced invalid frozen cell "
                    f"index {raw}; count={count}"
                )
            cells.append(raw)
        return tuple(cells)

    def _final_cell_masses(
        self,
        reference_index: int,
    ) -> Tuple[Dict[str, object], ...]:
        return self._cell_masses_from_state(
            objectives=self._objectives[reference_index],
            normalized_weights=tuple(
                math.exp(value)
                for value in self._log_weights[reference_index]
            ),
        )

    def _cell_masses_from_state(
        self,
        *,
        objectives: Sequence[ObjectiveVector],
        normalized_weights: Sequence[float],
    ) -> Tuple[Dict[str, object], ...]:
        if len(objectives) != len(normalized_weights):
            raise ValueError("objectives and normalized_weights must have equal length.")
        masses: Dict[Tuple[int, ...], float] = {}
        for objective, weight in zip(objectives, normalized_weights):
            cell = self._cell_index(objective)
            masses[cell] = masses.get(cell, 0.0) + float(weight)
        return tuple(
            {
                "epsilon_cell": cell,
                "mass": mass,
            }
            for cell, mass in sorted(masses.items())
        )

    def _energy(self, objective: ObjectiveVector, reference_index: int) -> float:
        normalized_objective = tuple(
            (value - lower) / (upper - lower)
            for value, lower, upper in zip(
                objective,
                self.objective_lower_bounds,
                self.objective_upper_bounds,
            )
        )
        if any(value < 0.0 or value > 1.0 for value in normalized_objective):
            raise ObjectiveBoundsViolation(
                "Normalized objective left [0, 1]; certified paths do not clamp."
            )
        weighted = tuple(
            weight * value
            for weight, value in zip(
                self.reference_directions[reference_index],
                normalized_objective,
            )
        )
        return max(weighted) + self.chebyshev_rho * sum(weighted)

    def _log_diagnostic(self, beta: float) -> None:
        weighted_energy = 0.0
        for energies, log_weights in zip(self._energies, self._log_weights):
            weighted_energy += sum(
                math.exp(log_weight) * energy
                for log_weight, energy in zip(log_weights, energies)
            )
        weighted_energy /= self.num_reference_types
        hypervolume = (
            self.archive.hypervolume_2d(
                reference=self.objective_upper_bounds
            )
            if self.instance.num_objectives == 2
            else 0.0
        )
        diagnostic = Diagnostic(
                iteration=self._evaluations_used(),
                temperature=(math.inf if beta == 0.0 else 1.0 / beta),
                acceptance_rate=self._accepted_mutations
                / max(1, self._mutation_attempts),
                archive_size=len(self.archive),
                hypervolume_2d=hypervolume,
                empirical_energy=weighted_energy,
                positive_archive_jump=0.0,
                front=tuple(
                    entry.objectives for entry in self.archive.entries
                ),
                elapsed_seconds=time.perf_counter() - self._start_time,
                replacement_attempts=self._mutation_attempts,
                accepted_replacements=self._accepted_mutations,
                rejected_replacements=self._mutation_attempts
                - self._accepted_mutations,
                rejection_rate=(
                    self._mutation_attempts - self._accepted_mutations
                )
                / max(1, self._mutation_attempts),
            )
        if (
            self._diagnostics
            and self._diagnostics[-1].iteration
            == diagnostic.iteration
        ):
            self._diagnostics[-1] = diagnostic
        else:
            self._diagnostics.append(diagnostic)
        period = self.anytime_checkpoint_period
        if (
            period is not None
            and diagnostic.iteration > 0
            and diagnostic.iteration % period == 0
        ):
            witness = {
                "evaluation": diagnostic.iteration,
                "entries": tuple(
                    {
                        "tour": entry.tour,
                        "objectives": entry.objectives,
                    }
                    for entry in self.search_archive.entries
                ),
            }
            if (
                self._checkpoint_solution_witnesses
                and self._checkpoint_solution_witnesses[-1]["evaluation"]
                == diagnostic.iteration
            ):
                self._checkpoint_solution_witnesses[-1] = witness
            else:
                self._checkpoint_solution_witnesses.append(witness)

    def _maybe_log_diagnostic(self, beta: float) -> None:
        period = self.anytime_checkpoint_period
        evaluations = self._evaluations_used()
        if (
            period is not None
            and evaluations > 0
            and evaluations % period == 0
        ):
            self._log_diagnostic(beta)

    def _resolve_objective_bounds(
        self,
        lower: Optional[Sequence[float]],
        upper: Optional[Sequence[float]],
    ) -> Tuple[ObjectiveVector, ObjectiveVector, str]:
        if (lower is None) != (upper is None):
            raise ValueError(
                "objective_lower_bounds and objective_upper_bounds must be supplied together."
            )
        if lower is None:
            lower_values, upper_values = analytic_objective_box(self.instance)
            return (
                lower_values,
                upper_values,
                "analytic_distance_matrix_box",
            )

        lower_values = tuple(float(value) for value in lower)
        upper_values = tuple(float(value) for value in upper or ())
        if (
            len(lower_values) != self.instance.num_objectives
            or len(upper_values) != self.instance.num_objectives
        ):
            raise ValueError("Explicit objective bounds have the wrong dimension.")
        for low, high in zip(lower_values, upper_values):
            if (
                not math.isfinite(low)
                or not math.isfinite(high)
                or high <= low
            ):
                raise ValueError(
                    "Explicit objective bounds must be finite with upper > lower."
                )
        return lower_values, upper_values, "explicit_predeclared_box"

    def _resolve_epsilon(
        self,
        epsilon: Optional[float | Sequence[float]],
    ) -> ObjectiveVector:
        spans = tuple(
            upper - lower
            for lower, upper in zip(
                self.objective_lower_bounds,
                self.objective_upper_bounds,
            )
        )
        if epsilon is None:
            values = tuple(0.05 * span for span in spans)
        elif isinstance(epsilon, (int, float)):
            values = tuple(float(epsilon) for _ in spans)
        else:
            values = tuple(float(value) for value in epsilon)
        if len(values) != self.instance.num_objectives:
            raise ValueError("epsilon has the wrong dimension.")
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("epsilon widths must be finite and positive.")
        return values

    def _resolve_reference_directions(
        self,
        directions: Optional[Sequence[Sequence[float]]],
        count: int,
    ) -> Tuple[ObjectiveVector, ...]:
        if directions is None:
            return ScalarArchivePotential.reference_directions(
                self.instance.num_objectives,
                count,
            )
        resolved = tuple(
            tuple(float(value) for value in direction)
            for direction in directions
        )
        if not resolved:
            raise ValueError("At least one reference direction is required.")
        for direction in resolved:
            if len(direction) != self.instance.num_objectives:
                raise ValueError("A reference direction has the wrong dimension.")
            if any(not math.isfinite(value) or value <= 0.0 for value in direction):
                raise ValueError(
                    "Reference-direction components must be finite and strictly positive."
                )
            if not math.isclose(sum(direction), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("Each reference direction must sum to one.")
        return resolved

    @staticmethod
    def _validate_beta_schedule(schedule: Sequence[float]) -> Tuple[float, ...]:
        values = tuple(float(value) for value in schedule)
        if len(values) < 2:
            raise ValueError("beta_schedule must contain beta_0 and a positive stage.")
        if values[0] != 0.0:
            raise ValueError("beta_schedule must start at beta_0 = 0.")
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("beta_schedule must contain finite nonnegative values.")
        if any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError("beta_schedule must be strictly increasing.")
        return values

    def _allocate_stage_mutations(self) -> Tuple[int, ...]:
        if self.resampling_policy == "always":
            assert self.bootstrap_mutations_by_stage is not None
            return tuple(
                self.num_particles * steps
                for steps in self.bootstrap_mutations_by_stage
            )
        remaining = self.core_evaluations - self.num_particles
        transitions = len(self.beta_schedule) - 1
        base, extra = divmod(remaining, transitions)
        return tuple(
            base + int(index < extra)
            for index in range(transitions)
        )

    def _multinomial_indices(self, normalized_log_weights: Sequence[float]) -> List[int]:
        weights = [math.exp(value) for value in normalized_log_weights]
        return self.rng.choices(
            range(self.particles_per_reference),
            weights=weights,
            k=self.particles_per_reference,
        )

    @staticmethod
    def _effective_sample_size(normalized_log_weights: Sequence[float]) -> float:
        return 1.0 / sum(
            math.exp(2.0 * value)
            for value in normalized_log_weights
        )

    @staticmethod
    def _logsumexp(values: Sequence[float]) -> float:
        maximum = max(values)
        return maximum + math.log(
            sum(math.exp(value - maximum) for value in values)
        )

    def _evaluations_used(self) -> int:
        if self._counted_instance:
            return evaluation_count(self.instance) - self._evaluation_counter_start
        return self._logical_evaluations

    def _make_context_hash(self) -> str:
        payload = {
            "algorithm_contract": self.contract_name,
            "instance_sha256": instance_sha256(self.instance),
            "objective_lower_bounds": self.objective_lower_bounds,
            "objective_upper_bounds": self.objective_upper_bounds,
            "reference_directions": self.reference_directions,
            "beta_schedule": self.beta_schedule,
            "chebyshev_rho": self.chebyshev_rho,
            "global_refresh_probability": self.global_refresh_probability,
            "energy": (
                "continuous_augmented_tchebycheff_of_strict_fail_closed_"
                "unclipped_frozen_box_normalized_objective_v2"
            ),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _make_reporting_context_hash(self) -> str:
        return self._payload_sha256(
            {
                "target_context_hash": self.context_hash,
                "epsilon": self.epsilon,
                "epsilon_cell_counts": self.cell_counts,
                "epsilon_cells_role": (
                    "external_reporting_coverage_observer_no_target_feedback"
                ),
            }
        )

    def _stage_target_hash(self, stage_index: int) -> str:
        payload = {
            "context_hash": self.context_hash,
            "stage_index": stage_index,
            "beta": self.beta_schedule[stage_index],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _make_run_contract_hash(self) -> str:
        return self._payload_sha256(
            {
                "context_hash": self.context_hash,
                "particles_per_reference": self.particles_per_reference,
                "num_reference_types": self.num_reference_types,
                "evaluation_budget": self.evaluations,
                "minimum_evaluation_budget": self.minimum_evaluation_budget,
                "stage_mutation_budgets": self._stage_mutation_budgets,
                "ess_threshold_fraction": self.ess_threshold,
                "resampling_policy": self.resampling_policy,
                "mutations_per_particle_per_stage": (
                    self.mutations_per_particle_per_stage
                ),
                "mutation_steps_by_stage": self.mutation_steps_by_stage,
                "bootstrap_mutations_by_stage": (
                    self.bootstrap_mutations_by_stage
                ),
                "finite_particle_delta": self.finite_particle_delta,
                "resampling_method": "multinomial",
                "resampling_scope": "within_fixed_reference_type_only",
                "proposal": self.mutation_proposal,
                "global_refresh_probability": self.global_refresh_probability,
                "adaptive_search_evaluations": self.adaptive_search_evaluations,
                "adaptive_allocation_policy": self.adaptive_allocation_policy,
                "adaptive_minimum_pulls_per_type": (
                    self.adaptive_minimum_pulls_per_type
                ),
                "exp3_exploration": self.exp3_exploration,
                "search_reward_weights": {
                    "hypervolume": self.search_reward_weights.hypervolume,
                    "new_cell": self.search_reward_weights.new_cell,
                    "scalar_improvement": self.search_reward_weights.scalar_improvement,
                },
                "enable_exact_incremental_two_opt": (
                    self.enable_exact_incremental_two_opt
                ),
                "archive_tolerance": self.archive_tolerance,
                "audit_trace_level": self.audit_trace_level,
                "seed": self.seed,
                "domain_separated_seed_contract": (
                    None
                    if self.domain_separated_seed is None
                    else self.domain_separated_seed.metadata()
                ),
                "rng_contract": "python_random_mt19937_finite_precision",
            }
        )

    def _assert_frozen_contract(self) -> None:
        if self._make_context_hash() != self.context_hash:
            raise RuntimeError(
                "The predeclared target context changed after initialization."
            )
        if self._make_reporting_context_hash() != self.reporting_context_hash:
            raise RuntimeError(
                "The predeclared reporting context changed after initialization."
            )
        if self._make_run_contract_hash() != self.run_contract_hash:
            raise RuntimeError(
                "The predeclared SMC run contract changed after initialization."
            )

    @staticmethod
    def _payload_sha256(payload: object) -> str:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
