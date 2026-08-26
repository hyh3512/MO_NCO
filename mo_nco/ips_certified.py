from __future__ import annotations

"""Mechanically auditable single-site positive-temperature IPS control.

This module implements the strict branch stated in ``theory_corrected.tex``.
It is intentionally separate from the fast heuristic optimizer: one coordinate
is selected uniformly, one uniform symmetric 2-opt move is proposed, the
implemented typed Hamiltonian is evaluated exactly, and every context value is
frozen for the whole run.  The Pareto archive is reporting-only and never feeds
back into the transition kernel.
"""

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import IO, List, Optional, Tuple

from .archive import ArchiveEntry, ParetoArchive
from .contracts import ClaimLevel
from .evaluation import can_evaluate, evaluation_count, remaining_evaluations
from .instance import MultiObjectiveTSPInstance, instance_sha256
from .moves import random_tour, sample_two_opt_indices, two_opt_at
from .potential import ScalarArchivePotential
from .sampler import Diagnostic, OptimizationResult
from .types import ObjectiveVector, Tour


class CertifiedSingleSiteIPSOptimizer:
    """Strict typed-Hamiltonian MH control for theory/implementation audits."""

    contract_name = "theory_certified_single_site_v4"
    implementation_version = "0.8.0"
    claim_level = ClaimLevel.CERTIFIED_MH

    def __init__(
        self,
        instance: MultiObjectiveTSPInstance,
        num_particles: int = 32,
        evaluations: int = 512,
        seed: int = 0,
        temperature: float = 0.05,
        chebyshev_rho: float = 0.03,
        log_period: int = 128,
        archive_max_size: Optional[int] = 300,
        uniformization_rate: float = 1.0,
        lazy_probability: float = 0.05,
        minimum_scale_fraction: float = 1e-3,
        absolute_scale_floor: float = 1e-12,
        trace_path: Optional[str | Path] = None,
    ) -> None:
        if num_particles <= 0:
            raise ValueError("num_particles must be positive.")
        if instance.num_cities < 4:
            raise ValueError("The certified 2-opt kernel requires at least four cities.")
        if evaluations < num_particles:
            raise ValueError("evaluations must cover all initial particle evaluations.")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and strictly positive.")
        if not math.isfinite(chebyshev_rho) or chebyshev_rho < 0.0:
            raise ValueError("chebyshev_rho must be finite and nonnegative.")
        if log_period <= 0:
            raise ValueError("log_period must be positive.")
        if not math.isfinite(uniformization_rate) or uniformization_rate <= 0.0:
            raise ValueError("uniformization_rate must be finite and strictly positive.")
        if not math.isfinite(lazy_probability) or not 0.0 < lazy_probability < 1.0:
            raise ValueError("lazy_probability must be finite and strictly between zero and one.")
        if not math.isfinite(minimum_scale_fraction) or minimum_scale_fraction < 0.0:
            raise ValueError("minimum_scale_fraction must be finite and nonnegative.")
        if not math.isfinite(absolute_scale_floor) or absolute_scale_floor <= 0.0:
            raise ValueError("absolute_scale_floor must be finite and strictly positive.")
        remaining = remaining_evaluations(instance)
        if remaining is not None and remaining < evaluations:
            raise ValueError(
                "The counting-instance budget is smaller than the requested certified run."
            )

        self._start_time = time.perf_counter()
        self.instance = instance
        self._counted_instance = hasattr(instance, "evaluations")
        self._evaluation_counter_start = evaluation_count(instance)
        self.num_particles = num_particles
        self.evaluations = evaluations
        self.temperature = float(temperature)
        self.chebyshev_rho = float(chebyshev_rho)
        self.log_period = log_period
        self.uniformization_rate = float(uniformization_rate)
        self.lazy_probability = float(lazy_probability)
        self.minimum_scale_fraction = float(minimum_scale_fraction)
        self.absolute_scale_floor = float(absolute_scale_floor)
        self.trace_path = Path(trace_path) if trace_path else None
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.archive = ParetoArchive(max_size=archive_max_size)
        self.weights = ScalarArchivePotential.reference_directions(
            instance.num_objectives,
            num_particles,
        )
        if len(self.weights) != num_particles:
            raise RuntimeError("reference-direction generator did not return one weight per particle.")

        self.population: List[Tour] = [random_tour(instance.num_cities, self.rng) for _ in range(num_particles)]
        self.objectives: List[ObjectiveVector] = [instance.evaluate(tour) for tour in self.population]
        self.archive.update(self._entries())
        self._hv_reference = self.archive.fixed_reference_2d() if instance.num_objectives == 2 else None
        self.ideal, self.nadir = self._fixed_ideal_nadir(self.objectives)
        scale_estimates = tuple(
            float(value)
            for value in getattr(instance, "objective_scale_estimates", ())
        )
        if len(scale_estimates) != instance.num_objectives:
            scale_estimates = tuple(max(1.0, abs(lo), abs(hi)) for lo, hi in zip(self.ideal, self.nadir))
        self._scales = tuple(
            max(
                self.absolute_scale_floor,
                hi - lo,
                self.minimum_scale_fraction * estimate,
            )
            for lo, hi, estimate in zip(self.ideal, self.nadir, scale_estimates)
        )
        self.context_hash = self._make_context_hash()
        self.instance_sha256 = instance_sha256(instance)

        self.diagnostics: List[Diagnostic] = []
        self._logical_evaluations = num_particles
        self._proposal_evaluations = 0
        self._accepted = 0
        self._rejected = 0
        self._current_rejection_streak = 0
        self._max_rejection_streak = 0
        self._max_db_log_residual = 0.0
        self._min_log_alpha = 0.0
        self._max_abs_delta_over_temperature = 0.0
        self._trace_records = 0
        self._trace_chain_hash = "0" * 64
        self._transition_attempts = 0
        self._lazy_self_loops = 0
        self._identity_evaluations = 0
        self._log_diagnostic()

    def run(self) -> OptimizationResult:
        trace_handle: Optional[IO[str]] = None
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_handle = self.trace_path.open("w", encoding="utf-8")
            self._write_trace_record(trace_handle, self._trace_header())
        try:
            while self._total_evaluations() < self.evaluations and can_evaluate(self.instance):
                self._transition_attempts += 1
                lazy_uniform = self.rng.random()
                if lazy_uniform < self.lazy_probability:
                    # The benchmark stops on an objective-evaluation clock.
                    # Re-evaluate a fixed current state so that the identity
                    # mixture is present on that same clock, rather than only
                    # on an unreported attempt clock.
                    identity_tour = self.population[0]
                    identity_objective = self.instance.evaluate(identity_tour)
                    self._logical_evaluations += 1
                    self._identity_evaluations += 1
                    if identity_objective != self.objectives[0]:
                        raise RuntimeError(
                            "Instance objective evaluation is not a deterministic tour state function."
                        )
                    self._lazy_self_loops += 1
                    if trace_handle is not None:
                        population_hash = self._population_hash()
                        self._write_trace_record(
                            trace_handle,
                            {
                                "record_type": "lazy_transition",
                                "transition_attempt": self._transition_attempts,
                                "lazy_uniform": lazy_uniform,
                                "identity_tour": identity_tour,
                                "identity_objective": identity_objective,
                                "population_hash_before": population_hash,
                                "population_hash_after": population_hash,
                            },
                        )
                    if self._total_evaluations() % self.log_period == 0:
                        self._log_diagnostic()
                    continue
                index = self.rng.randrange(self.num_particles)
                current_tour = self.population[index]
                current_objective = self.objectives[index]
                i, j = sample_two_opt_indices(len(current_tour), self.rng)
                proposed_tour = two_opt_at(current_tour, i, j)
                # The certified state is the tour itself. Recompute the full
                # objective deterministically so that an inverse move returns
                # exactly to the same machine state. Cached floating 2-opt
                # deltas are safe for the fast heuristic, but their round-off
                # drift would make the objective cache an unmodelled state
                # variable and invalidate DB on the declared tour space.
                proposed_objective = self.instance.evaluate(proposed_tour)
                self._logical_evaluations += 1
                self._proposal_evaluations += 1

                # Reporting archive: queried solutions are retained for fair
                # black-box evaluation, but archive contents never affect H or Q.
                self.archive.update((ArchiveEntry(proposed_tour, proposed_objective),))

                delta_h = self._typed_single_energy(proposed_objective, index) - self._typed_single_energy(
                    current_objective,
                    index,
                )
                delta_over_temperature = delta_h / self.temperature
                log_alpha_forward = min(0.0, -delta_over_temperature)
                log_alpha_reverse = min(0.0, delta_over_temperature)
                residual = abs(delta_over_temperature + log_alpha_forward - log_alpha_reverse)
                self._max_db_log_residual = max(self._max_db_log_residual, residual)
                self._min_log_alpha = min(self._min_log_alpha, log_alpha_forward)
                self._max_abs_delta_over_temperature = max(
                    self._max_abs_delta_over_temperature,
                    abs(delta_over_temperature),
                )

                # Compare in log space.  This removes the artificial exp(-745)
                # support truncation while keeping the finite-precision nature
                # of Python's pseudorandom generator explicit in metadata.
                uniform_draw = self.rng.random()
                log_uniform = -math.inf if uniform_draw == 0.0 else math.log(uniform_draw)
                accepted = log_uniform < log_alpha_forward
                before_hash = self._population_hash() if trace_handle is not None else ""

                if accepted:
                    self.population[index] = proposed_tour
                    self.objectives[index] = proposed_objective
                    self._accepted += 1
                    self._current_rejection_streak = 0
                else:
                    self._rejected += 1
                    self._current_rejection_streak += 1
                    self._max_rejection_streak = max(
                        self._max_rejection_streak,
                        self._current_rejection_streak,
                    )

                if trace_handle is not None:
                    self._write_trace_record(
                        trace_handle,
                        {
                            "record_type": "transition",
                            "transition_attempt": self._transition_attempts,
                            "lazy_uniform": lazy_uniform,
                            "proposal_index": self._proposal_evaluations,
                            "coordinate": index,
                            "two_opt_i": i,
                            "two_opt_j": j,
                            "current_tour": current_tour,
                            "proposed_tour": proposed_tour,
                            "current_objective": current_objective,
                            "proposed_objective": proposed_objective,
                            "delta_h": delta_h,
                            "delta_over_temperature": delta_over_temperature,
                            "log_alpha": log_alpha_forward,
                            "log_uniform": log_uniform if math.isfinite(log_uniform) else "-inf",
                            "accepted": accepted,
                            "population_hash_before": before_hash,
                            "population_hash_after": self._population_hash(),
                        },
                    )

                if self._total_evaluations() % self.log_period == 0 or not can_evaluate(self.instance):
                    self._log_diagnostic()
        finally:
            if trace_handle is not None:
                trace_handle.close()

        if self._total_evaluations() != self.evaluations:
            raise RuntimeError(
                "Certified run ended without consuming its exact run-local evaluation budget."
            )
        self._log_diagnostic()
        metadata = {
            "algorithm_contract": self.contract_name,
            "implementation_version": self.implementation_version,
            "claim_level": self.claim_level.value,
            "implemented_hamiltonian": "sum_typed_augmented_tchebycheff",
            "objective_evaluation_contract": "full_tour_state_function",
            "instance_sha256": self.instance_sha256,
            "context_hash": self.context_hash,
            "context_frozen": True,
            "context_refresh_count": 0,
            "bounds_frozen": True,
            "single_coordinate_transition": True,
            "proposal": "uniform_symmetric_two_opt",
            "proposal_symmetric": True,
            "proposal_log_ratio": 0.0,
            "temperature_schedule": "constant",
            "temperature": self.temperature,
            "temperature_min": self.temperature,
            "positive_temperature": True,
            "archive_role": "reporting_only_no_kernel_feedback",
            "archive_feedback": False,
            "mean_field_enabled": False,
            "neural_enabled": False,
            "compiled_polish_enabled": False,
            "crossover_enabled": False,
            "local_refinement_enabled": False,
            "uniformization_rate": self.uniformization_rate,
            "uniformization_role": "declaration_only_not_executed",
            "lazy_probability": self.lazy_probability,
            "explicit_laziness": True,
            "aperiodicity_mechanism": "explicit_identity_mixture",
            "transition_attempts": self._transition_attempts,
            "lazy_self_loops": self._lazy_self_loops,
            "identity_evaluations": self._identity_evaluations,
            "transition_evaluations": self._transition_attempts,
            "evaluation_clock_kernel": "explicit_lazy_identity_mixture",
            "rng_contract": "python_random_mt19937_seed_replay_v1",
            "seed": self.seed,
            "initial_population_evaluations": self.num_particles,
            "proposal_evaluations": self._proposal_evaluations,
            "accepted_single_site_moves": self._accepted,
            "rejected_single_site_moves": self._rejected,
            "db_max_abs_log_residual": self._max_db_log_residual,
            "acceptance_computation": "log_uniform_comparison",
            "minimum_log_alpha": self._min_log_alpha,
            "max_abs_delta_over_temperature": self._max_abs_delta_over_temperature,
            "energy_scales": self._scales,
            "diagnostic_hypervolume_reference": self._hv_reference,
            "scale_policy": "max(initial_range, minimum_scale_fraction*objective_scale_estimate, absolute_floor)",
            "minimum_scale_fraction": self.minimum_scale_fraction,
            "absolute_scale_floor": self.absolute_scale_floor,
            "trace_path": str(self.trace_path) if self.trace_path is not None else "",
            "trace_records": self._trace_records,
            "trace_chain_hash": self._trace_chain_hash if self.trace_path is not None else "",
            "evaluation_counter_start": self._evaluation_counter_start,
            "evaluation_budget": self.evaluations,
            "evaluations_used": self._total_evaluations(),
        }
        return OptimizationResult(
            particles=tuple(self.population),
            objectives=tuple(self.objectives),
            archive=self.archive,
            diagnostics=tuple(self.diagnostics),
            metadata=metadata,
        )

    def _typed_single_energy(self, objective: ObjectiveVector, index: int) -> float:
        weight = self.weights[index]
        normalized = tuple(
            (value - lo) / scale
            for value, lo, scale in zip(objective, self.ideal, self._scales)
        )
        weighted = tuple(max(1e-3, w) * value for w, value in zip(weight, normalized))
        return max(weighted) + self.chebyshev_rho * sum(weighted)

    def _typed_phi(self) -> float:
        return sum(
            self._typed_single_energy(objective, index)
            for index, objective in enumerate(self.objectives)
        ) / self.num_particles

    def _total_evaluations(self) -> int:
        if self._counted_instance:
            return evaluation_count(self.instance) - self._evaluation_counter_start
        return self._logical_evaluations

    def _log_diagnostic(self) -> None:
        evaluations = self._total_evaluations()
        diagnostic = Diagnostic(
            iteration=evaluations,
            temperature=self.temperature,
            acceptance_rate=self._accepted / max(1, self._proposal_evaluations),
            archive_size=len(self.archive),
            hypervolume_2d=(
                self.archive.hypervolume_2d(reference=self._hv_reference)
                if self.instance.num_objectives == 2
                else 0.0
            ),
            empirical_energy=self._typed_phi(),
            positive_archive_jump=0.0,
            front=tuple(entry.objectives for entry in self.archive.entries),
            elapsed_seconds=time.perf_counter() - self._start_time,
            replacement_attempts=self._proposal_evaluations,
            accepted_replacements=self._accepted,
            rejected_replacements=self._rejected,
            rejection_rate=self._rejected / max(1, self._proposal_evaluations),
            current_rejection_streak=self._current_rejection_streak,
            max_rejection_streak=self._max_rejection_streak,
        )
        if self.diagnostics and self.diagnostics[-1].iteration == evaluations:
            self.diagnostics[-1] = diagnostic
        else:
            self.diagnostics.append(diagnostic)

    def _trace_header(self) -> dict:
        return {
            "record_type": "header",
            "algorithm_contract": self.contract_name,
            "implementation_version": self.implementation_version,
            "context_hash": self.context_hash,
            "instance_sha256": self.instance_sha256,
            "rng_contract": "python_random_mt19937_seed_replay_v1",
            "seed": self.seed,
            "num_cities": self.instance.num_cities,
            "num_objectives": self.instance.num_objectives,
            "num_particles": self.num_particles,
            "temperature": self.temperature,
            "lazy_probability": self.lazy_probability,
            "chebyshev_rho": self.chebyshev_rho,
            "minimum_scale_fraction": self.minimum_scale_fraction,
            "absolute_scale_floor": self.absolute_scale_floor,
            "ideal": self.ideal,
            "nadir": self.nadir,
            "scales": self._scales,
            "weights": self.weights,
            "initial_population": self.population,
            "initial_objectives": self.objectives,
            "initial_population_hash": self._population_hash(),
        }

    def _write_trace_record(self, handle: IO[str], payload: dict) -> None:
        chained = dict(payload)
        chained["previous_record_hash"] = self._trace_chain_hash
        encoded = json.dumps(chained, sort_keys=True, separators=(",", ":"), allow_nan=False)
        record_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        chained["record_hash"] = record_hash
        handle.write(json.dumps(chained, sort_keys=True, allow_nan=False) + "\n")
        self._trace_chain_hash = record_hash
        self._trace_records += 1

    def _population_hash(self) -> str:
        encoded = json.dumps(self.population, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _make_context_hash(self) -> str:
        payload = {
            "ideal": self.ideal,
            "nadir": self.nadir,
            "scales": self._scales,
            "weights": self.weights,
            "chebyshev_rho": self.chebyshev_rho,
            "temperature": self.temperature,
            "lazy_probability": self.lazy_probability,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _entries(self) -> Tuple[ArchiveEntry, ...]:
        return tuple(ArchiveEntry(tour, objective) for tour, objective in zip(self.population, self.objectives))

    @staticmethod
    def _fixed_ideal_nadir(objectives: List[ObjectiveVector]) -> Tuple[ObjectiveVector, ObjectiveVector]:
        dim = len(objectives[0])
        ideal = tuple(min(objective[d] for objective in objectives) for d in range(dim))
        nadir = tuple(max(objective[d] for objective in objectives) for d in range(dim))
        return ideal, nadir
