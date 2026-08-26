from __future__ import annotations

import bisect
import hashlib
import heapq
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from .archive import ArchiveEntry, ParetoArchive
from .contracts import ClaimLevel
from .evaluation import can_evaluate, evaluation_count, remaining_evaluations
from .instance import MultiObjectiveTSPInstance
from .learned_move_generator import SparseMoveGenerator
from .moves import mixed_move, order_crossover, random_tour, sample_two_opt_indices, two_opt, two_opt_at
from .neural_potential import TinyMLP
from .paretoflow_net import ParetoFlowScalarNet
from .pcd_net import PCDResidualScalarNet
from .potential import ScalarArchivePotential
from .sampler import Diagnostic, OptimizationResult
from .types import ObjectiveVector, Tour


class EfficientIPSOptimizer:
    """Theory-aware archive-conditioned IPS heuristic.

    This fast path deliberately includes batched neighbor replacement, changing
    normalization/archive context, and optional deterministic refinement.  It is
    therefore an experimental zero/low-temperature optimizer, not the certified
    homogeneous single-site MH kernel implemented in ``ips_certified.py``.
    """

    contract_name = "fast_nonautonomous_batch_descent_v2"
    claim_level = ClaimLevel.HEURISTIC_DESCENT

    def __init__(
        self,
        instance: MultiObjectiveTSPInstance,
        num_particles: int = 48,
        evaluations: int = 2000,
        seed: int = 0,
        neighbor_size: int = 6,
        archive_max_size: Optional[int] = 300,
        log_period: int = 50,
        crossover_probability: float = 0.25,
        archive_parent_probability: float = 0.35,
        archive_update_period: int = 8,
        archive_parent_sample: int = 6,
        proposal: str = "mixed",
        extra_two_opt_probability: float = 0.25,
        initial_temperature: float = 0.0,
        final_temperature: float = 0.0,
        archive_conditioning: bool = True,
        archive_conditioning_weight: float = 0.0,
        neural_scalar_weight: float = 0.0,
        neural_backend: str = "tiny",
        neural_hidden_units: int = 8,
        neural_training_epochs: int = 15,
        neural_learning_rate: float = 0.03,
        neural_archive_repeats: int = 3,
        neural_proposal_probability: float = 0.0,
        neural_proposal_weight: float = 0.0,
        neural_candidate_pool: int = 1,
        neural_relocate_candidate_probability: float = 0.0,
        neural_proposal_min_samples: int = 64,
        neural_prior_path: str = "",
        enable_neural_scalar: bool = True,
        require_endpoint_only_prior: bool = False,
        neural_online_training: bool = True,
        neural_fit_period: int = 1,
        neural_directional_coverage_weight: float = 0.0,
        neural_extreme_progress_weight: float = 0.0,
        neural_gap_fill_weight: float = 0.0,
        neural_hv_center_bias: float = 0.0,
        neural_extreme_repeats: int = 0,
        neural_action_sample_pool: int = 1,
        neural_mean_field_features: bool = False,
        neural_prefilter_pool: int = 1,
        neural_refine_top_k: int = 1,
        neural_exact_two_opt_prefilter: bool = False,
        neural_flow_pair_samples: int = 0,
        neural_flow_residual_weight: float = 0.0,
        neural_ranking_weight: float = 0.0,
        neural_hypercone_loss_weight: float = 0.0,
        neural_coverage_pair_weight: float = 0.0,
        neural_expert_pair_weight: float = 0.0,
        neural_expert_pair_samples: int = 0,
        neural_weight_norm_bound: float = 0.0,
        neural_mean_field_update_period: int = 0,
        neural_mean_field_target_weight: float = 0.0,
        neural_active_fraction: float = 1.0,
        neural_stagnation_patience: int = 0,
        neural_stagnation_epsilon: float = 0.0,
        neural_stagnation_wake_steps: int = 0,
        neural_late_repair_fraction: float = 0.0,
        neural_rank_fusion_weight: float = 0.0,
        neural_mean_field_guidance_weight: float = 0.0,
        neural_gap_direction_probability: float = 0.0,
        neural_learned_move_probability: float = 0.0,
        neural_learned_move_sparse_nodes: int = 16,
        neural_learned_move_sparse_partners: int = 16,
        neural_learned_move_samples: int = 1,
        neural_learned_move_learning_rate: float = 0.04,
        neural_learned_move_prior_path: str = "",
        require_target_only_move_prior: bool = False,
        allow_move_without_scalar: bool = False,
        ablation_contract: str = "",
        neural_condition_guidance_scale: float = 1.0,
        neural_front_reweighting_strength: float = 0.0,
        extreme_anchor_fraction: float = 0.0,
        extreme_anchor_period: int = 0,
        initialization: str = "random",
        greedy_candidate_pool: int = 3,
        greedy_start_pool: int = 1,
        initial_2opt_passes: int = 0,
        proposal_2opt_passes: int = 0,
        initial_relocate_passes: int = 0,
        proposal_relocate_passes: int = 0,
        initial_swap_passes: int = 0,
        proposal_swap_passes: int = 0,
        jit_polish_fraction: float = 1.1,
        jit_polish_chunk_size: int = 0,
        isolate_prior_loading_rng: bool = False,
        enable_mechanism_diagnostics: bool = True,
    ) -> None:
        if num_particles <= 0:
            raise ValueError("num_particles must be positive.")
        if evaluations < num_particles:
            raise ValueError("evaluations must cover the initial population.")
        if log_period <= 0:
            raise ValueError("log_period must be positive.")
        if neighbor_size <= 0:
            raise ValueError("neighbor_size must be positive.")
        if instance.num_cities < 4:
            raise ValueError("Efficient 2-opt search requires at least four cities.")

        self._start_time = time.perf_counter()
        self.instance = instance
        self.num_particles = num_particles
        self.evaluations = evaluations
        self._logical_evaluations = 0
        self.rng = random.Random(seed)
        self.enable_mechanism_diagnostics = bool(enable_mechanism_diagnostics)
        self.isolate_prior_loading_rng = bool(isolate_prior_loading_rng)
        self._scalar_model_rng = (
            random.Random(self._derived_stream_seed(seed, "scalar-model-v1"))
            if self.isolate_prior_loading_rng
            else self.rng
        )
        self._move_policy_rng = (
            random.Random(self._derived_stream_seed(seed, "move-policy-v1"))
            if self.isolate_prior_loading_rng
            else self.rng
        )
        self.archive = ParetoArchive(max_size=archive_max_size)
        self.weights = ScalarArchivePotential.reference_directions(instance.num_objectives, num_particles)
        if len(self.weights) != num_particles:
            raise RuntimeError("reference-direction generator did not return one weight per particle.")
        self._weight0 = tuple(max(1e-3, weight[0]) for weight in self.weights) if instance.num_objectives == 2 else ()
        self._weight1 = tuple(max(1e-3, weight[1]) for weight in self.weights) if instance.num_objectives == 2 else ()
        self._weight_extremeness_cache = tuple(
            abs(w0 - w1) / max(1e-9, w0 + w1) for w0, w1 in zip(self._weight0, self._weight1)
        )
        self._weight_norm_cache = tuple(
            math.hypot(w0, w1) for w0, w1 in zip(self._weight0, self._weight1)
        )
        self._weight_prefers_first = tuple(w0 >= w1 for w0, w1 in zip(self._weight0, self._weight1))
        self._edge_scale0, self._edge_scale1 = self._edge_scales2()
        self.neighbors = self._build_neighbors(max(2, min(neighbor_size, num_particles)))
        self.log_period = log_period
        self.crossover_probability = crossover_probability
        self.archive_parent_probability = archive_parent_probability
        self.archive_update_period = max(1, archive_update_period)
        self.archive_parent_sample = max(1, archive_parent_sample)
        if proposal not in {"mixed", "two_opt"}:
            raise ValueError("proposal must be 'mixed' or 'two_opt'.")
        self.proposal = proposal
        self.extra_two_opt_probability = max(0.0, min(1.0, extra_two_opt_probability))
        self.initial_temperature = max(0.0, initial_temperature)
        self.final_temperature = max(0.0, final_temperature)
        self.archive_conditioning = archive_conditioning
        self.archive_conditioning_weight = max(0.0, archive_conditioning_weight)
        self.neural_scalar_weight = max(0.0, neural_scalar_weight)
        self.neural_backend = neural_backend.lower().strip() if neural_backend else "tiny"
        if self.neural_backend not in {"tiny", "paretoflow", "pcd", "auto"}:
            raise ValueError("neural_backend must be 'tiny', 'paretoflow', 'pcd', or 'auto'.")
        self.neural_training_epochs = max(0, neural_training_epochs)
        self.neural_learning_rate = max(0.0, neural_learning_rate)
        self.neural_archive_repeats = max(1, neural_archive_repeats)
        self.neural_proposal_probability = max(0.0, min(1.0, neural_proposal_probability))
        self.neural_proposal_weight = max(0.0, neural_proposal_weight)
        self.neural_candidate_pool = max(1, neural_candidate_pool)
        self.neural_relocate_candidate_probability = max(0.0, min(1.0, neural_relocate_candidate_probability))
        self.neural_proposal_min_samples = max(1, neural_proposal_min_samples)
        self.neural_prior_path = str(neural_prior_path)
        self._neural_prior_sha256 = self._file_sha256(self.neural_prior_path)
        self.enable_neural_scalar = bool(enable_neural_scalar)
        self.require_endpoint_only_prior = bool(require_endpoint_only_prior)
        self._neural_prior_feature_contract = ""
        self.neural_online_training = neural_online_training
        self.neural_fit_period = max(1, neural_fit_period)
        self.neural_directional_coverage_weight = max(0.0, neural_directional_coverage_weight)
        self.neural_extreme_progress_weight = max(0.0, neural_extreme_progress_weight)
        self.neural_gap_fill_weight = max(0.0, neural_gap_fill_weight)
        self.neural_hv_center_bias = max(0.0, min(1.0, neural_hv_center_bias))
        self.neural_extreme_repeats = max(0, neural_extreme_repeats)
        self.neural_action_sample_pool = max(1, neural_action_sample_pool)
        self.neural_mean_field_features = neural_mean_field_features
        self.neural_prefilter_pool = max(1, neural_prefilter_pool)
        self.neural_refine_top_k = max(1, neural_refine_top_k)
        self.neural_exact_two_opt_prefilter = neural_exact_two_opt_prefilter
        self.neural_flow_pair_samples = max(0, neural_flow_pair_samples)
        self.neural_flow_residual_weight = max(0.0, neural_flow_residual_weight)
        self.neural_ranking_weight = max(0.0, neural_ranking_weight)
        self.neural_hypercone_loss_weight = max(0.0, neural_hypercone_loss_weight)
        self.neural_coverage_pair_weight = max(0.0, neural_coverage_pair_weight)
        self.neural_expert_pair_weight = max(0.0, neural_expert_pair_weight)
        self.neural_expert_pair_samples = max(0, neural_expert_pair_samples)
        self.neural_weight_norm_bound = max(0.0, neural_weight_norm_bound)
        self.neural_mean_field_update_period = max(
            1, neural_mean_field_update_period if neural_mean_field_update_period > 0 else archive_update_period
        )
        self.neural_mean_field_target_weight = max(0.0, neural_mean_field_target_weight)
        self.neural_active_fraction = max(0.0, min(1.0, neural_active_fraction))
        self.neural_stagnation_patience = max(0, neural_stagnation_patience)
        self.neural_stagnation_epsilon = max(0.0, neural_stagnation_epsilon)
        self.neural_stagnation_wake_steps = max(0, neural_stagnation_wake_steps)
        self.neural_late_repair_fraction = max(0.0, min(1.0, neural_late_repair_fraction))
        self.neural_rank_fusion_weight = max(0.0, min(1.0, neural_rank_fusion_weight))
        self.neural_mean_field_guidance_weight = max(0.0, min(1.0, neural_mean_field_guidance_weight))
        self.neural_gap_direction_probability = max(0.0, min(1.0, neural_gap_direction_probability))
        self.neural_learned_move_probability = max(0.0, min(1.0, neural_learned_move_probability))
        self.neural_learned_move_sparse_nodes = max(4, neural_learned_move_sparse_nodes)
        self.neural_learned_move_sparse_partners = max(4, neural_learned_move_sparse_partners)
        self.neural_learned_move_samples = max(1, neural_learned_move_samples)
        self.neural_learned_move_learning_rate = max(0.0, neural_learned_move_learning_rate)
        self.require_target_only_move_prior = bool(require_target_only_move_prior)
        self.allow_move_without_scalar = bool(allow_move_without_scalar)
        self.ablation_contract = str(ablation_contract)
        self.neural_learned_move_prior_path = neural_learned_move_prior_path or os.environ.get(
            "MO_NCO_LEARNED_MOVE_PRIOR_PATH",
            "",
        )
        self._learned_move_prior_sha256 = (
            self._file_sha256(self.neural_learned_move_prior_path)
            if self.neural_learned_move_probability > 0.0
            else ""
        )
        self.neural_condition_guidance_scale = max(0.0, neural_condition_guidance_scale)
        self.neural_front_reweighting_strength = max(0.0, neural_front_reweighting_strength)
        self.extreme_anchor_fraction = max(0.0, min(1.0, extreme_anchor_fraction))
        self.extreme_anchor_period = max(0, extreme_anchor_period)
        if initialization not in {"random", "scalar_greedy", "mixed_scalar_greedy"}:
            raise ValueError("initialization must be 'random', 'scalar_greedy', or 'mixed_scalar_greedy'.")
        self.initialization = initialization
        self.greedy_candidate_pool = max(1, greedy_candidate_pool)
        self.greedy_start_pool = max(1, greedy_start_pool)
        self.initial_2opt_passes = max(0, initial_2opt_passes)
        self.proposal_2opt_passes = max(0, proposal_2opt_passes)
        self.initial_relocate_passes = max(0, initial_relocate_passes)
        self.proposal_relocate_passes = max(0, proposal_relocate_passes)
        self.initial_swap_passes = max(0, initial_swap_passes)
        self.proposal_swap_passes = max(0, proposal_swap_passes)
        self.jit_polish_fraction = max(0.0, jit_polish_fraction)
        self.jit_polish_chunk_size = max(0, jit_polish_chunk_size)
        self._neural_training_samples = 0
        self._neural_prior_loaded = False
        self._neural_policy_calls = 0
        self._neural_raw_candidates = 0
        self._neural_refined_candidates = 0
        self._neural_score_batches = 0
        self._neural_rank_changed_decisions = 0
        self._neural_mv_changed_decisions = 0
        self._neural_gap_direction_steps = 0
        self._neural_learned_move_calls = 0
        self._neural_learned_move_children = 0
        self._neural_learned_move_updates = 0
        self._neural_learned_move_angle_filtered = 0
        self._neural_learned_move_candidate_count = 0
        self._neural_learned_move_reward_sum = 0.0
        self._neural_learned_move_positive_rewards = 0
        self._neural_learned_move_angle_penalty_sum = 0.0
        self._neural_learned_move_cone_pass_count = 0
        self._neural_learned_move_mf_reward_sum = 0.0
        self._neural_learned_move_mass_observations = 0
        self._neural_learned_move_good_mass_sum = 0.0
        self._neural_baseline_good_mass_sum = 0.0
        self._neural_learned_move_crossing_mass_sum = 0.0
        self._neural_baseline_crossing_mass_sum = 0.0
        self._neural_sampled_pool_policy_reward_sum = 0.0
        self._neural_sampled_pool_uniform_reward_sum = 0.0
        self._neural_sampled_pool_reward_observations = 0
        self._neural_gate_eligible_steps = 0
        self._neural_gate_fired_steps = 0
        self._scalar_proposal_suppressed_by_learned_move = 0
        self._scalar_candidate_decision_observations = 0
        self._scalar_candidate_changed_decisions = 0
        self._scalar_candidate_target_margin_vs_analytic_sum = 0.0
        self._scalar_candidate_target_margin_vs_pool_mean_sum = 0.0
        self._scalar_candidate_target_regret_to_pool_oracle_sum = 0.0
        self._scalar_candidate_analytic_score_regret_sum = 0.0
        self._scalar_parent_decision_observations = 0
        self._scalar_parent_changed_decisions = 0
        self._scalar_parent_analytic_score_regret_sum = 0.0
        self._scalar_archive_parent_decision_observations = 0
        self._scalar_archive_parent_changed_decisions = 0
        self._scalar_archive_parent_analytic_score_regret_sum = 0.0
        self._scalar_replacement_preference_observations = 0
        self._scalar_replacement_flip_to_accept = 0
        self._scalar_replacement_flip_to_reject = 0
        self._learned_move_selection_observations = 0
        self._learned_move_selected_reward_sum = 0.0
        self._learned_move_selected_pool_uniform_reward_sum = 0.0
        self._learned_move_selected_pool_oracle_reward_sum = 0.0
        self._compiled_polish_children = 0
        self._neural_generated_children = 0
        self._neural_scalar_generated_children = 0
        self._neural_move_generated_children = 0
        self._neural_accepted_children = 0
        self._neural_accepted_replacements = 0
        self._neural_scalar_accepted_children = 0
        self._neural_scalar_accepted_replacements = 0
        self._neural_move_accepted_children = 0
        self._neural_move_accepted_replacements = 0
        self._neural_coverage_pairs = 0
        self._neural_expert_pairs = 0
        self._neural_scalar_forward_calls = 0
        self._neural_scalar_scored_states = 0
        self._neural_scalar_cache_hits = 0
        self._neural_scalar_cache_misses = 0
        self._neural_scalar_inference_seconds = 0.0
        self._local_two_opt_check_upper_bound = 0
        self._local_relocate_check_upper_bound = 0
        self._local_swap_check_upper_bound = 0
        self._archive_flush_count = 0
        self._context_refresh_count = 0
        self._normalization_refresh_count = 0
        self._positive_context_jump_since_log = 0.0
        self._cumulative_positive_context_jump = 0.0
        self._cumulative_signed_context_jump = 0.0
        self._positive_context_jump_by_kind: dict[str, float] = {}
        self._signed_context_jump_by_kind: dict[str, float] = {}
        self._context_jump_event_counts: dict[str, int] = {}
        self._context_jump_accounting_errors = 0
        self._unattributed_compiled_energy_delta = 0.0
        self._unattributed_compiled_positive_delta = 0.0
        self._unattributed_compiled_event_count = 0
        self._child_steps = 0
        self._accelerator_fallbacks: List[str] = []
        self._replacement_attempts = 0
        self._accepted_replacements = 0
        self._rejected_replacements = 0
        self._current_rejection_streak = 0
        self._max_rejection_streak = 0
        self._best_logged_hv = -1.0
        self._last_hv_improvement_eval = 0
        self._neural_wake_until_eval = 0
        self._neural_stagnation_wake_count = 0
        self._last_neural_spectral_diagnostics: dict = {}
        self._neural_bias_cache: dict[Tuple[int, float, float], float] = {}
        self._archive_endpoint_bias_cache: dict[Tuple[float, float], Tuple[float, float, float, float, float]] = {}
        self._archive_direction_bias_cache: dict[Tuple[float, float, int], float] = {}
        self._neural_scalar: Optional[Any] = None
        if (
            self.enable_neural_scalar
            and (self.neural_scalar_weight > 0.0 or self.neural_proposal_probability > 0.0)
            and instance.num_objectives == 2
        ):
            self._neural_scalar = self._load_neural_prior(neural_prior_path) if neural_prior_path else None
            if self._neural_scalar is None:
                input_dim = (
                    24
                    if (self.neural_mean_field_features and self.neural_backend in {"pcd", "paretoflow"})
                    else (20 if self.neural_mean_field_features else 6)
                )
                self._neural_scalar = self._new_neural_backend(input_dim, neural_hidden_units)
        self._learned_move_generator: Optional[SparseMoveGenerator] = None
        if self.neural_learned_move_probability > 0.0 and instance.num_objectives == 2:
            self._learned_move_generator = self._load_learned_move_prior(self.neural_learned_move_prior_path)
            if self._learned_move_generator is None:
                self._learned_move_generator = SparseMoveGenerator(
                    learning_rate=self.neural_learned_move_learning_rate,
                    rng=self._move_policy_rng,
                    flow_head_weight=0.35 if self.neural_flow_residual_weight > 0.0 else 0.0,
                    mean_field_head_weight=0.20 if self.neural_mean_field_features else 0.0,
                )
            # A loaded multi-head prior retains the head weights stored in its
            # JSON payload.  Apply runtime ablations after loading as well;
            # otherwise "no-mf" and "no-flow-consistency" are only partial
            # ablations despite their names.
            if self.neural_flow_residual_weight <= 0.0:
                self._learned_move_generator.flow_head_weight = 0.0
            if not self.neural_mean_field_features:
                self._learned_move_generator.mean_field_head_weight = 0.0
            if self.require_target_only_move_prior:
                non_target_weights = (
                    self._learned_move_generator.flow_head_weight,
                    self._learned_move_generator.mean_field_head_weight,
                    self._learned_move_generator.conductance_head_weight,
                )
                if any(abs(float(value)) > 1e-12 for value in non_target_weights):
                    raise ValueError(
                        "Theory-optimized move prior must be target-only: flow, mean-field, and conductance head weights must be zero."
                    )

        self._weighted_matrix_cache: dict[int, object] = {}
        self.population: List[Tour] = []
        self.objectives: List[ObjectiveVector] = []
        self.diagnostics: List[Diagnostic] = []
        self._initialize_population()
        self._initial_population_sha256 = self._population_sha256(self.population)
        self._base_rng_state_after_initialization_sha256 = self._rng_state_sha256(self.rng)
        self._diagnostic_hv_reference = (
            self.archive.fixed_reference_2d() if instance.num_objectives == 2 and self.archive.entries else None
        )
        self.ideal, self.nadir = self._compute_ideal_nadir(self.objectives)
        self._refresh_scale_cache()
        self._archive_norm_points: Tuple[Tuple[float, float], ...] = ()
        self._archive_norm_x: Tuple[float, ...] = ()
        self._archive_norm_y: Tuple[float, ...] = ()
        self._archive_hv_cache = 0.0
        self._archive_best_scalar_by_weight: Tuple[float, ...] = ()
        self._particle_direction_summary: Tuple[Tuple[float, float, float, float, float, float], ...] = ()
        self._archive_cache_ideal0 = self._ideal0
        self._archive_cache_ideal1 = self._ideal1
        self._archive_cache_inv0 = self._inv0
        self._archive_cache_inv1 = self._inv1
        self._refresh_archive_conditioning_cache()
        self._refresh_particle_summary_cache()
        self._pending_archive: List[ArchiveEntry] = []
        self._fit_neural_scalar()

    def run(self) -> OptimizationResult:
        step = 0
        self._log_diagnostic(self._used_evaluations())
        while self._used_evaluations() < self.evaluations and can_evaluate(self.instance):
            if self._should_run_compiled_scalar_polish(step):
                polished = self._run_compiled_scalar_polish(step, self._compiled_polish_chunk_limit(step))
                if polished > 0:
                    if self.enable_mechanism_diagnostics:
                        self._compiled_polish_children += polished
                    step += polished
                    self._child_steps = step
                    if step % self.log_period != 0:
                        self._log_diagnostic(self._used_evaluations())
                    if self.jit_polish_chunk_size <= 0 or self._used_evaluations() >= self.evaluations or not can_evaluate(self.instance):
                        break
                    continue
            direction_idx = self._direction_index_for_step(step)
            child, child_obj = self._make_child(direction_idx)
            self._update_bounds(child_obj)
            child_entry = ArchiveEntry(child, child_obj)
            self._pending_archive.append(child_entry)
            if (step + 1) % self.archive_update_period == 0:
                self._flush_archive()
            self._replace_neighbors(direction_idx, child, child_obj, step)
            step += 1
            self._child_steps = step

            if step % self.log_period == 0 or not can_evaluate(self.instance):
                self._log_diagnostic(self._used_evaluations())

        self._flush_archive()
        final_evaluations = self._used_evaluations()
        self._log_diagnostic(final_evaluations)
        return OptimizationResult(
            tuple(self.population),
            tuple(self.objectives),
            self.archive,
            tuple(self.diagnostics),
            self.neural_inference_stats(),
        )

    def _should_run_compiled_scalar_polish(self, step: int) -> bool:
        del step  # activation is defined on the true evaluation budget
        if self.jit_polish_fraction >= 1.0:
            return False
        if self._used_evaluations() < int(self.jit_polish_fraction * self.evaluations):
            return False
        return not self._neural_stagnation_wake_active() and not self._neural_late_repair_active()

    def _compiled_polish_chunk_limit(self, step: int) -> Optional[int]:
        del step
        if self.jit_polish_chunk_size <= 0:
            return None
        return max(1, min(self.jit_polish_chunk_size, self.evaluations - self._used_evaluations()))

    def _direction_index_for_step(self, step: int) -> int:
        if (
            self.neural_gap_direction_probability > 0.0
            and self._neural_is_active()
            and self.archive.entries
            and self.rng.random() < self.neural_gap_direction_probability
        ):
            gap_idx = self._archive_gap_direction_index()
            if gap_idx is not None:
                self._neural_gap_direction_steps += 1
                return gap_idx
        if (
            self.extreme_anchor_fraction > 0.0
            and self.num_particles > 1
            and self._used_evaluations() <= int(self.extreme_anchor_fraction * self.evaluations)
        ):
            period = max(2, self.extreme_anchor_period or 4)
            slot = step % period
            if slot == 0:
                return 0
            if slot == period // 2:
                return self.num_particles - 1
        return step % self.num_particles

    def _archive_gap_direction_index(self) -> Optional[int]:
        points = self._archive_norm_points
        if len(points) < 2 or not self.weights:
            if self.num_particles > 1:
                return 0 if self._used_evaluations() % 2 == 0 else self.num_particles - 1
            return 0 if self.weights else None
        best_gap = -1.0
        target = points[0]
        for left, right in zip(points, points[1:]):
            gap = (right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2
            if gap > best_gap:
                best_gap = gap
                target = ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)
        if best_gap <= 1e-12:
            edge_left = points[0][0] + points[0][1]
            edge_right = points[-1][0] + points[-1][1]
            return 0 if edge_left > edge_right else self.num_particles - 1
        desired0 = 1.0 / max(1e-6, target[0])
        desired1 = 1.0 / max(1e-6, target[1])
        denom = desired0 + desired1
        desired0 /= denom
        desired1 /= denom
        return min(
            range(len(self.weights)),
            key=lambda idx: (self._weight0[idx] - desired0) ** 2 + (self._weight1[idx] - desired1) ** 2,
        )

    def _used_evaluations(self) -> int:
        counted = getattr(self.instance, "evaluations", None)
        return int(counted) if counted is not None else self._logical_evaluations

    def _base_instance(self) -> object:
        return getattr(self.instance, "base", self.instance)

    def _distance_matrices(self):  # type: ignore[no-untyped-def]
        base = self._base_instance()
        return getattr(base, "distance_matrices", getattr(base, "_distance_matrices", None))

    def _symmetric_objectives(self):  # type: ignore[no-untyped-def]
        base = self._base_instance()
        return getattr(base, "symmetric_objectives", getattr(base, "_symmetric_matrices", None))

    def _evaluate(self, tour: Tour) -> ObjectiveVector:
        objective = self.instance.evaluate(tour)
        self._logical_evaluations += 1
        return objective

    def _evaluate_two_opt(
        self,
        tour: Tour,
        current_objective: ObjectiveVector,
        i: int,
        j: int,
    ) -> ObjectiveVector:
        method = getattr(self.instance, "evaluate_two_opt", None)
        objective = method(tour, current_objective, i, j) if callable(method) else self.instance.evaluate(two_opt_at(tour, i, j))
        self._logical_evaluations += 1
        return objective

    def _charge_compiled_evaluations(self, count: int) -> None:
        charge = getattr(self.instance, "charge_evaluations", None)
        if callable(charge):
            charge(count)
        self._logical_evaluations += count

    def _log_diagnostic(self, evaluations: int, elapsed_seconds: Optional[float] = None) -> None:
        hv = (
            self.archive.hypervolume_2d(reference=getattr(self, "_diagnostic_hv_reference", None))
            if self.instance.num_objectives == 2
            else 0.0
        )
        self._observe_hv_stagnation(evaluations, hv)
        acceptance_rate = self._accepted_replacements / max(1, self._replacement_attempts)
        rejection_rate = self._rejected_replacements / max(1, self._replacement_attempts)
        elapsed = time.perf_counter() - self._start_time if elapsed_seconds is None else elapsed_seconds
        initialized = hasattr(self, "ideal") and bool(self.objectives)
        temperature = self._temperature(self._child_steps) if initialized else self.initial_temperature
        empirical_energy = self._typed_population_energy() if initialized else 0.0
        positive_context_jump = self._positive_context_jump_since_log
        self._positive_context_jump_since_log = 0.0
        diagnostic = Diagnostic(
            evaluations,
            temperature,
            acceptance_rate,
            len(self.archive),
            hv,
            empirical_energy,
            positive_context_jump,
            tuple(entry.objectives for entry in self.archive.entries),
            elapsed,
            self._replacement_attempts,
            self._accepted_replacements,
            self._rejected_replacements,
            rejection_rate,
            self._current_rejection_streak,
            self._max_rejection_streak,
        )
        if self.diagnostics and self.diagnostics[-1].iteration == evaluations:
            self.diagnostics[-1] = diagnostic
        else:
            self.diagnostics.append(diagnostic)

    def _typed_population_energy(self) -> float:
        if not self.objectives:
            return 0.0
        if self.instance.num_objectives == 2:
            values = self._scalar2_pairs(
                [(objective, index) for index, objective in enumerate(self.objectives)]
            )
        else:
            values = [
                self._scalar(objective, self.weights[index])
                for index, objective in enumerate(self.objectives)
            ]
        return sum(values) / len(values)

    def _record_context_transition(self, kind: str, before: float, after: float) -> None:
        """Record an energy jump caused only by a live-context refresh.

        Callers must snapshot ``before`` after any population replacement and
        immediately before mutating normalization, archive, mean-field, or
        neural context. ``after`` is evaluated on that same population.
        """
        if not math.isfinite(before) or not math.isfinite(after):
            self._context_jump_accounting_errors += 1
            return
        delta = after - before
        positive = max(0.0, delta)
        self._positive_context_jump_since_log += positive
        self._cumulative_positive_context_jump += positive
        self._cumulative_signed_context_jump += delta
        self._positive_context_jump_by_kind[kind] = (
            self._positive_context_jump_by_kind.get(kind, 0.0) + positive
        )
        self._signed_context_jump_by_kind[kind] = (
            self._signed_context_jump_by_kind.get(kind, 0.0) + delta
        )
        self._context_jump_event_counts[kind] = self._context_jump_event_counts.get(kind, 0) + 1

    def _observe_hv_stagnation(self, evaluations: int, hv: float) -> None:
        if hv > self._best_logged_hv + self.neural_stagnation_epsilon:
            self._best_logged_hv = hv
            self._last_hv_improvement_eval = evaluations
            return
        if self.neural_stagnation_patience <= 0 or self.neural_stagnation_wake_steps <= 0:
            return
        if evaluations <= 0 or evaluations >= self.evaluations:
            return
        if evaluations - self._last_hv_improvement_eval < self.neural_stagnation_patience:
            return
        self._neural_wake_until_eval = max(
            self._neural_wake_until_eval,
            min(self.evaluations, evaluations + self.neural_stagnation_wake_steps),
        )
        self._last_hv_improvement_eval = evaluations
        self._neural_stagnation_wake_count += 1
        if self.neural_mean_field_features and self._neural_scalar is not None:
            before = self._typed_population_energy()
            self._refresh_particle_summary_cache()
            self._record_context_transition(
                "stagnation_mean_field_refresh",
                before,
                self._typed_population_energy(),
            )

    def _make_child(self, direction_idx: int) -> Tuple[Tour, ObjectiveVector]:
        parent_idx = self._select_parent_index(direction_idx)
        parent = self.population[parent_idx]
        parent_obj: Optional[ObjectiveVector] = self.objectives[parent_idx]

        if (
            self.archive_parent_probability > 0.0
            and self.archive.entries
            and self.rng.random() < self.archive_parent_probability
        ):
            weight = self.weights[direction_idx]
            entries = self.archive.entries
            if len(entries) > self.archive_parent_sample:
                entries = tuple(self.rng.sample(list(entries), self.archive_parent_sample))
            if self.instance.num_objectives == 2:
                archive_values = self._scalar2_pairs([(entry.objectives, direction_idx) for entry in entries])
                if self._scalar_decision_diagnostics_active():
                    analytic_values = [
                        self._analytic_proposal_score2(entry.objectives, direction_idx)
                        for entry in entries
                    ]
                    self._record_scalar_argmin_decision(
                        analytic_values,
                        archive_values,
                        scope="archive_parent",
                    )
                archive_entry = entries[min(range(len(entries)), key=lambda pos: archive_values[pos])]
            else:
                archive_entry = min(entries, key=lambda entry: self._scalar(entry.objectives, weight))
            archive_parent = archive_entry.tour
            if self.rng.random() < self.crossover_probability:
                parent = order_crossover(parent, archive_parent, self.rng)
                parent_obj = None
            else:
                parent = archive_parent
                parent_obj = archive_entry.objectives

        neural_child = self._make_neural_proposal_child(parent, parent_obj, direction_idx)
        if neural_child is not None:
            neural_source = getattr(self, "_last_neural_proposal_source", "scalar")
            self._last_child_source = f"neural_{neural_source}"
            self._neural_generated_children += 1
            if neural_source == "move":
                self._neural_move_generated_children += 1
            else:
                self._neural_scalar_generated_children += 1
            return neural_child

        self._last_child_source = "analytic"
        can_use_delta = self.proposal == "two_opt" and self.extra_two_opt_probability == 0.0 and parent_obj is not None
        if can_use_delta:
            i, j = sample_two_opt_indices(len(parent), self.rng)
            child = two_opt_at(parent, i, j)
            child = self._scalar_local_descent(
                child,
                direction_idx,
                self.proposal_2opt_passes,
                self.proposal_relocate_passes,
                self.proposal_swap_passes,
            )
            method = getattr(self.instance, "evaluate_two_opt", None)
            child_obj = (
                self._evaluate_two_opt(parent, parent_obj, i, j)
                if (
                    callable(method)
                    and self.proposal_2opt_passes == 0
                    and self.proposal_relocate_passes == 0
                    and self.proposal_swap_passes == 0
                )
                else self._evaluate(child)
            )
            return child, child_obj

        child = two_opt(parent, self.rng) if self.proposal == "two_opt" else mixed_move(parent, self.rng)
        if self.extra_two_opt_probability > 0.0 and self.rng.random() < self.extra_two_opt_probability:
            child = two_opt(child, self.rng)
        return child, self._evaluate(child)

    def _select_parent_index(self, direction_idx: int) -> int:
        candidates = self.neighbors[direction_idx]
        if self.instance.num_objectives == 2:
            values = self._scalar2_pairs([(self.objectives[idx], direction_idx) for idx in candidates])
            if self._scalar_decision_diagnostics_active():
                analytic_values = [
                    self._analytic_proposal_score2(self.objectives[idx], direction_idx)
                    for idx in candidates
                ]
                self._record_scalar_argmin_decision(
                    analytic_values,
                    values,
                    scope="parent",
                )
            return candidates[min(range(len(candidates)), key=lambda pos: values[pos])]
        weight = self.weights[direction_idx]
        return min(candidates, key=lambda idx: self._scalar(self.objectives[idx], weight))

    def _replace_neighbors(
        self,
        direction_idx: int,
        child: Tour,
        child_obj: ObjectiveVector,
        step: int,
    ) -> None:
        # Update the closest directions first. This is a particle-level analogue
        # of a zero-temperature transition under local scalar potentials.
        accepted_this_child = 0
        if self.instance.num_objectives == 2:
            neighbor_indices = self.neighbors[direction_idx]
            neighbor_count = len(neighbor_indices)
            paired_values = self._scalar2_pairs(
                [(child_obj, idx) for idx in neighbor_indices]
                + [(self.objectives[idx], idx) for idx in neighbor_indices]
            )
            child_values = paired_values[:neighbor_count]
            incumbent_values = paired_values[neighbor_count:]
            if self._scalar_decision_diagnostics_active():
                analytic_pairs = [
                    (child_obj, idx) for idx in neighbor_indices
                ] + [
                    (self.objectives[idx], idx) for idx in neighbor_indices
                ]
                analytic_values = [
                    self._analytic_proposal_score2(objective, idx)
                    for objective, idx in analytic_pairs
                ]
                analytic_child_values = analytic_values[:neighbor_count]
                analytic_incumbent_values = analytic_values[neighbor_count:]
                for treated_child, treated_incumbent, analytic_child, analytic_incumbent in zip(
                    child_values,
                    incumbent_values,
                    analytic_child_values,
                    analytic_incumbent_values,
                ):
                    self._record_scalar_replacement_preference(
                        analytic_child - analytic_incumbent,
                        treated_child - treated_incumbent,
                    )
            for pos, idx in enumerate(neighbor_indices):
                delta = child_values[pos] - incumbent_values[pos]
                self._replacement_attempts += 1
                if self._accept_delta(delta, step):
                    self.population[idx] = child
                    self.objectives[idx] = child_obj
                    self._accepted_replacements += 1
                    accepted_this_child += 1
                else:
                    self._rejected_replacements += 1
        else:
            for idx in self.neighbors[direction_idx]:
                weight = self.weights[idx]
                delta = self._scalar(child_obj, weight) - self._scalar(self.objectives[idx], weight)
                self._replacement_attempts += 1
                if self._accept_delta(delta, step):
                    self.population[idx] = child
                    self.objectives[idx] = child_obj
                    self._accepted_replacements += 1
                    accepted_this_child += 1
                else:
                    self._rejected_replacements += 1
        if accepted_this_child == 0:
            self._current_rejection_streak += 1
            self._max_rejection_streak = max(self._max_rejection_streak, self._current_rejection_streak)
        else:
            self._current_rejection_streak = 0
            child_source = getattr(self, "_last_child_source", "")
            if child_source == "neural" or child_source.startswith("neural_"):
                self._neural_accepted_children += 1
                self._neural_accepted_replacements += accepted_this_child
                if child_source == "neural_move":
                    self._neural_move_accepted_children += 1
                    self._neural_move_accepted_replacements += accepted_this_child
                else:
                    self._neural_scalar_accepted_children += 1
                    self._neural_scalar_accepted_replacements += accepted_this_child
        if (
            self.neural_mean_field_features
            and self._neural_scalar is not None
            and (step + 1) % self.neural_mean_field_update_period == 0
        ):
            before = self._typed_population_energy()
            self._refresh_particle_summary_cache()
            self._record_context_transition(
                "periodic_mean_field_refresh",
                before,
                self._typed_population_energy(),
            )

    def _flush_archive(self) -> None:
        if self._pending_archive:
            before = self._typed_population_energy() if hasattr(self, "ideal") else 0.0
            self.archive.update(self._pending_archive)
            self._pending_archive = []
            self._refresh_archive_conditioning_cache()
            self._refresh_particle_summary_cache()
            self._archive_flush_count += 1
            self._context_refresh_count += 1
            if self._neural_is_active() and self._archive_flush_count % self.neural_fit_period == 0:
                self._fit_neural_scalar()
            after = self._typed_population_energy() if hasattr(self, "ideal") else before
            self._record_context_transition("archive_neural_refresh", before, after)

    def _run_compiled_scalar_polish(self, start_step: int, max_children_limit: Optional[int] = None) -> int:
        if not self._can_run_compiled_scalar_polish():
            return 0
        remaining = remaining_evaluations(self.instance)
        max_children = self.evaluations - self._used_evaluations()
        if max_children_limit is not None:
            max_children = min(max_children, max_children_limit)
        if remaining is not None:
            max_children = min(max_children, remaining)
        if max_children <= 0:
            return 0
        try:
            import numpy as np

            from .numba_kernels import NUMBA_AVAILABLE, ips_scalar_polish_epoch_numba

            if not NUMBA_AVAILABLE:
                return 0
            before_context_energy = self._typed_population_energy()
            old_ideal = self.ideal
            old_nadir = self.nadir
            matrices_source = self._distance_matrices()
            if matrices_source is None:
                return 0
            matrices = np.asarray(matrices_source, dtype=np.float64)
            weighted = self._weighted_matrices2()
            if weighted is None:
                return 0
            population = np.asarray(self.population, dtype=np.int64)
            objectives = np.asarray(self.objectives, dtype=np.float64)
            neighbors = np.asarray(self.neighbors, dtype=np.int64)
            weight0 = np.asarray(self._weight0, dtype=np.float64)
            weight1 = np.asarray(self._weight1, dtype=np.float64)
            seed = self.rng.randrange(1, 2_147_483_647)
            (
                population_out,
                objectives_out,
                child_tours,
                child_objs,
                attempts,
                accepted,
                rejected,
                current_streak,
                max_streak,
                ideal0,
                ideal1,
                nadir0,
                nadir1,
            ) = ips_scalar_polish_epoch_numba(
                matrices,
                weighted,
                population,
                objectives,
                weight0,
                weight1,
                neighbors,
                int(max_children),
                int(start_step),
                int(seed),
                int(self.proposal_2opt_passes),
                int(self.proposal_relocate_passes),
                float(self._ideal0),
                float(self._ideal1),
                float(self.nadir[0]),
                float(self.nadir[1]),
                int(self._current_rejection_streak),
                int(self._max_rejection_streak),
            )
            self._record_local_search_check_upper_bound(
                self.instance.num_cities,
                self.proposal_2opt_passes * max_children,
                self.proposal_relocate_passes * max_children,
                self.proposal_swap_passes * max_children,
            )
            self.population = [tuple(int(city) for city in row.tolist()) for row in population_out]
            self.objectives = [tuple(float(value) for value in row.tolist()) for row in objectives_out]
            self.ideal = (float(ideal0), float(ideal1))
            self.nadir = (float(nadir0), float(nadir1))
            if self.ideal != old_ideal or self.nadir != old_nadir:
                self._context_refresh_count += 1
                self._normalization_refresh_count += 1
            self._refresh_scale_cache()
            self._replacement_attempts += int(attempts)
            self._accepted_replacements += int(accepted)
            self._rejected_replacements += int(rejected)
            self._current_rejection_streak = int(current_streak)
            self._max_rejection_streak = int(max_streak)

            # Apply all reporting-archive observations in one batch.  The old
            # code replayed max_children prefixes and logged the final
            # population as though it were every intermediate state.  Batched
            # insertion is faster and avoids synthetic diagnostics.
            entries = [
                ArchiveEntry(
                    tuple(int(city) for city in child_tours[idx].tolist()),
                    (float(child_objs[idx, 0]), float(child_objs[idx, 1])),
                )
                for idx in range(int(max_children))
            ]
            if entries:
                self.archive.update(entries)
                self._archive_flush_count += 1
                self._context_refresh_count += 1
            self._charge_compiled_evaluations(int(max_children))
            self._pending_archive = []
            self._refresh_archive_conditioning_cache()
            self._refresh_particle_summary_cache()
            after_context_energy = self._typed_population_energy()
            aggregate_delta = after_context_energy - before_context_energy
            self._unattributed_compiled_energy_delta += aggregate_delta
            self._unattributed_compiled_positive_delta += max(0.0, aggregate_delta)
            self._unattributed_compiled_event_count += 1
            return int(max_children)
        except Exception as exc:
            self._record_accelerator_fallback("compiled_scalar_polish", exc)
            return 0

    def _record_accelerator_fallback(self, stage: str, exc: BaseException) -> None:
        message = f"{stage}:{type(exc).__name__}:{exc}"
        if len(self._accelerator_fallbacks) < 32:
            self._accelerator_fallbacks.append(message)
        if os.environ.get("MO_NCO_STRICT_ACCELERATOR", "0") == "1":
            raise RuntimeError(message) from exc

    def _can_run_compiled_scalar_polish(self) -> bool:
        if self.instance.num_objectives != 2:
            return False
        if self.proposal != "two_opt" or self.extra_two_opt_probability > 0.0:
            return False
        if self.initial_temperature > 0.0 or self.final_temperature > 0.0:
            return False
        if self.proposal_swap_passes > 0:
            return False
        matrices = self._distance_matrices()
        symmetric = self._symmetric_objectives()
        return matrices is not None and len(matrices) == 2 and symmetric is not None and all(symmetric)

    def _scalar(self, objective: ObjectiveVector, weight: ObjectiveVector) -> float:
        if len(objective) == 2:
            z0 = (objective[0] - self._ideal0) * self._inv0
            z1 = (objective[1] - self._ideal1) * self._inv1
            a = max(1e-3, weight[0]) * z0
            b = max(1e-3, weight[1]) * z1
            return (a if a >= b else b) + 0.03 * (a + b)
        values = []
        for value, lo, hi in zip(objective, self.ideal, self.nadir):
            values.append((value - lo) / max(1e-9, hi - lo))
        cheb = max(max(1e-3, w) * value for value, w in zip(values, weight))
        aug = 0.03 * sum(max(1e-3, w) * value for value, w in zip(values, weight))
        return cheb + aug

    def _scalar2_by_weight(self, objective: ObjectiveVector, weight_idx: int) -> float:
        return self._scalar2_pairs([(objective, weight_idx)])[0]

    def _scalar2_pairs(self, pairs: Sequence[Tuple[ObjectiveVector, int]]) -> List[float]:
        """Score endpoint/weight pairs while sharing endpoint-only archive work.

        A child is commonly scored against several neighboring directions.  HV
        gain, novelty, and gap geometry depend only on the endpoint, not on the
        direction, so compute them once per distinct objective in the batch.
        """
        values = [self._base_scalar2_by_weight(objective, weight_idx) for objective, weight_idx in pairs]
        if self.archive_conditioning and self.archive_conditioning_weight > 0.0 and pairs:
            values = [
                value + self._archive_improvement_bias_cached2(objective, weight_idx)
                for value, (objective, weight_idx) in zip(values, pairs)
            ]
        if self._neural_is_active() and self._neural_scalar is not None and self.neural_scalar_weight > 0.0:
            biases = self._predict_neural_scalar_biases(pairs)
            values = [value + self.neural_scalar_weight * bias for value, bias in zip(values, biases)]
        return values

    def _neural_proposal_score2(
        self,
        objective: ObjectiveVector,
        weight_idx: int,
        parent_objective: Optional[ObjectiveVector] = None,
    ) -> float:
        value = self._base_scalar2_by_weight(objective, weight_idx)
        if self.archive_conditioning and self.archive_conditioning_weight > 0.0:
            value += self._archive_improvement_bias2(objective, value, weight_idx)
        if self._neural_is_active() and self._neural_scalar is not None and self.neural_proposal_weight > 0.0:
            # The scalar network is an endpoint-only evaluator.  Source/move
            # features belong to SparseMoveGenerator, which is proposal-only.
            # Keeping this call endpoint-only also removes a train/inference
            # distribution shift: replacement and proposal ranking now query
            # the same scalar function on the same feature contract.
            value += self.neural_proposal_weight * self._predict_neural_scalar_bias(objective, weight_idx)
        return value

    def _neural_is_active(self) -> bool:
        if self._neural_stagnation_wake_active():
            return True
        if self._neural_late_repair_active():
            return True
        if self.neural_active_fraction >= 1.0:
            return True
        if self.neural_active_fraction <= 0.0:
            return False
        return self._used_evaluations() <= max(1, int(self.neural_active_fraction * self.evaluations))

    def _neural_stagnation_wake_active(self) -> bool:
        return self.neural_stagnation_wake_steps > 0 and self._used_evaluations() <= self._neural_wake_until_eval

    def _neural_late_repair_active(self) -> bool:
        if self.neural_late_repair_fraction <= 0.0:
            return False
        threshold = int((1.0 - self.neural_late_repair_fraction) * self.evaluations)
        return self._used_evaluations() >= max(1, threshold)

    def _analytic_proposal_score2(self, objective: ObjectiveVector, weight_idx: int) -> float:
        value = self._base_scalar2_by_weight(objective, weight_idx)
        if self.archive_conditioning and self.archive_conditioning_weight > 0.0:
            value += self._archive_improvement_bias2(objective, value, weight_idx)
        return value

    def _base_scalar2_by_weight(self, objective: ObjectiveVector, weight_idx: int) -> float:
        z0 = (objective[0] - self._ideal0) * self._inv0
        z1 = (objective[1] - self._ideal1) * self._inv1
        a = self._weight0[weight_idx] * z0
        b = self._weight1[weight_idx] * z1
        return (a if a >= b else b) + 0.03 * (a + b)

    def _archive_improvement_bias2(self, objective: ObjectiveVector, _scalar_value: float, weight_idx: int) -> float:
        return self._archive_improvement_bias_cached2(objective, weight_idx)

    def _archive_improvement_bias_cached2(self, objective: ObjectiveVector, weight_idx: int) -> float:
        key = (float(objective[0]), float(objective[1]), int(weight_idx))
        if key in self._archive_direction_bias_cache:
            return self._archive_direction_bias_cache[key]
        cached_terms = self._archive_bias_terms_cached2(objective)
        if cached_terms is None:
            value = 0.0
        else:
            z0, z1, hv_gain, novelty, gap_value = cached_terms
            value = self._archive_improvement_bias_from_terms2(
                z0, z1, hv_gain, novelty, weight_idx, gap_value=gap_value
            )
        if len(self._archive_direction_bias_cache) >= 65536:
            self._archive_direction_bias_cache.clear()
        self._archive_direction_bias_cache[key] = value
        return value

    def _archive_bias_terms_cached2(
        self, objective: ObjectiveVector
    ) -> Optional[Tuple[float, float, float, float, float]]:
        key = (float(objective[0]), float(objective[1]))
        cached = self._archive_endpoint_bias_cache.get(key)
        if cached is not None:
            return cached
        terms = self._archive_bias_terms2(objective)
        if terms is None:
            return None
        z0, z1, hv_gain, novelty = terms
        packed = (z0, z1, hv_gain, novelty, self._archive_gap_value_from_norm(z0, z1))
        if len(self._archive_endpoint_bias_cache) >= 4096:
            self._archive_endpoint_bias_cache.clear()
        self._archive_endpoint_bias_cache[key] = packed
        return packed

    def _archive_bias_terms2(self, objective: ObjectiveVector) -> Optional[Tuple[float, float, float, float]]:
        if not self._archive_norm_points:
            return None
        z0 = (objective[0] - self._archive_cache_ideal0) * self._archive_cache_inv0
        z1 = (objective[1] - self._archive_cache_ideal1) * self._archive_cache_inv1
        best_dist2 = self._nearest_archive_dist2(z0, z1)
        hv_gain = self._normalized_archive_hv_gain_from_norm2((z0, z1))
        return z0, z1, hv_gain, min(0.5, best_dist2)

    def _archive_improvement_bias_from_terms2(
        self,
        z0: float,
        z1: float,
        hv_gain: float,
        novelty: float,
        weight_idx: int,
        gap_value: Optional[float] = None,
    ) -> float:
        candidate_scalar = self._base_scalar2_from_norm(z0, z1, weight_idx)
        best_archive = self._archive_best_scalar_by_weight[weight_idx]
        improvement = max(0.0, best_archive - candidate_scalar)
        extreme = self._weight_extremeness(weight_idx)
        drift = self._directional_drift_from_norm(z0, z1, weight_idx)
        extreme_progress = self._extreme_progress_from_norm(z0, z1, weight_idx)
        if gap_value is None:
            gap_value = self._archive_gap_value_from_norm(z0, z1)
        hv_term = hv_gain * (1.0 - self.neural_hv_center_bias * extreme)
        reward = (
            hv_term
            + 0.2 * improvement / (1.0 + improvement)
            + 0.05 * novelty
            + self.neural_directional_coverage_weight * (1.0 - drift)
            + self.neural_extreme_progress_weight * extreme * extreme_progress
            + self.neural_gap_fill_weight * gap_value
        )
        return -self.archive_conditioning_weight * reward

    def _weight_extremeness(self, weight_idx: int) -> float:
        return self._weight_extremeness_cache[weight_idx]

    def _directional_drift_from_norm(self, z0: float, z1: float, weight_idx: int) -> float:
        improve0 = max(0.0, 1.0 - z0)
        improve1 = max(0.0, 1.0 - z1)
        improve_norm = (improve0 * improve0 + improve1 * improve1) ** 0.5
        weight_norm = self._weight_norm_cache[weight_idx]
        if improve_norm <= 1e-12 or weight_norm <= 1e-12:
            return 1.0
        cosine = (self._weight0[weight_idx] * improve0 + self._weight1[weight_idx] * improve1) / (
            weight_norm * improve_norm
        )
        return max(0.0, min(1.0, 1.0 - cosine))

    def _extreme_progress_from_norm(self, z0: float, z1: float, weight_idx: int) -> float:
        if self._weight_prefers_first[weight_idx]:
            return max(0.0, min(1.0, 1.0 - z0))
        return max(0.0, min(1.0, 1.0 - z1))

    def _archive_gap_value_from_norm(self, z0: float, z1: float) -> float:
        points = self._archive_norm_points
        if len(points) < 2:
            return 0.0
        k = bisect.bisect_left(self._archive_norm_x, z0)
        if 0 < k < len(points):
            left = points[k - 1]
            right = points[k]
            chord = ((right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2) ** 0.5
            return min(0.5, chord)
        if k == 0:
            edge = ((points[0][0] - z0) ** 2 + (points[0][1] - z1) ** 2) ** 0.5
        else:
            edge = ((z0 - points[-1][0]) ** 2 + (z1 - points[-1][1]) ** 2) ** 0.5
        return min(0.5, edge)

    def _normalized_archive_hv_gain2(self, objective: ObjectiveVector) -> float:
        return self._normalized_archive_hv_gain_from_norm2(self._normalize2(objective))

    def _normalized_archive_hv_gain_from_norm2(self, candidate: Tuple[float, float]) -> float:
        points = self._archive_norm_points
        if not points:
            return 0.0
        x, y = candidate
        if x > 1.35 or y > 1.35:
            return 0.0
        ref_x, ref_y = 1.12, 1.12
        k = bisect.bisect_left(self._archive_norm_x, x)
        if k > 0 and self._archive_norm_y[k - 1] <= y:
            return 0.0
        if k < len(points) and self._archive_norm_x[k] <= x + 1e-12 and self._archive_norm_y[k] <= y:
            return 0.0

        r = k
        n = len(points)
        while r < n and self._archive_norm_y[r] >= y:
            r += 1

        prev_y = ref_y if k == 0 else self._archive_norm_y[k - 1]
        before = 0.0
        local_end = r + 1 if r < n else r
        local_prev = prev_y
        for idx in range(k, local_end):
            px, py = points[idx]
            before += max(0.0, ref_x - px) * max(0.0, local_prev - py)
            local_prev = min(local_prev, py)

        after = max(0.0, ref_x - x) * max(0.0, prev_y - y)
        if r < n:
            px, py = points[r]
            after += max(0.0, ref_x - px) * max(0.0, y - py)
        return max(0.0, after - before)

    def _nearest_archive_dist2(self, z0: float, z1: float) -> float:
        if not self._archive_norm_points:
            return 0.0
        anchor = bisect.bisect_left(self._archive_norm_x, z0)
        best = float("inf")
        start = max(0, anchor - 4)
        stop = min(len(self._archive_norm_points), anchor + 5)
        for idx in range(start, stop):
            a0, a1 = self._archive_norm_points[idx]
            dist2 = (z0 - a0) * (z0 - a0) + (z1 - a1) * (z1 - a1)
            if dist2 < best:
                best = dist2
        return best

    def _normalize2(self, objective: ObjectiveVector) -> Tuple[float, float]:
        return (
            (objective[0] - self._ideal0) * self._inv0,
            (objective[1] - self._ideal1) * self._inv1,
        )

    def _refresh_archive_conditioning_cache(self) -> None:
        self._archive_endpoint_bias_cache.clear()
        self._archive_direction_bias_cache.clear()
        if self.instance.num_objectives != 2 or not self.archive.entries:
            self._archive_norm_points = ()
            self._archive_norm_x = ()
            self._archive_norm_y = ()
            self._archive_hv_cache = 0.0
            self._archive_best_scalar_by_weight = ()
            return
        self._archive_cache_ideal0 = self._ideal0
        self._archive_cache_ideal1 = self._ideal1
        self._archive_cache_inv0 = self._inv0
        self._archive_cache_inv1 = self._inv1
        points = tuple(
            sorted(
                (
                    (
                        (entry.objectives[0] - self._archive_cache_ideal0) * self._archive_cache_inv0,
                        (entry.objectives[1] - self._archive_cache_ideal1) * self._archive_cache_inv1,
                    )
                    for entry in self.archive.entries
                ),
                key=lambda point: point[0],
            )
        )
        nondominated = []
        best_y = float("inf")
        for x, y in points:
            if x > 1.35 or y > 1.35:
                continue
            if y < best_y:
                nondominated.append((x, y))
                best_y = y
        self._archive_norm_points = tuple(nondominated)
        self._archive_norm_x = tuple(point[0] for point in self._archive_norm_points)
        self._archive_norm_y = tuple(point[1] for point in self._archive_norm_points)
        self._archive_hv_cache = self._normalized_hv2(self._archive_norm_points)
        self._archive_best_scalar_by_weight = tuple(
            min(self._base_scalar2_from_norm(z0, z1, idx) for z0, z1 in self._archive_norm_points)
            for idx in range(len(self.weights))
        )
        self._clear_neural_bias_cache()

    def _refresh_particle_summary_cache(self) -> None:
        summary_required = (
            self.neural_mean_field_features
            or self.neural_mean_field_guidance_weight > 0.0
            or self.neural_mean_field_target_weight > 0.0
        )
        if not summary_required or self.instance.num_objectives != 2 or not self.objectives:
            self._particle_direction_summary = ()
            return
        summaries = []
        for weight_idx in range(len(self.weights)):
            values = [self._base_scalar2_by_weight(objective, weight_idx) for objective in self.objectives]
            best = min(values)
            mean = sum(values) / len(values)
            variance = sum((value - mean) * (value - mean) for value in values) / len(values)
            neighbor_values = [
                values[idx]
                for idx in self.neighbors[weight_idx]
                if idx < len(values)
            ]
            if neighbor_values:
                neighbor_best = min(neighbor_values)
                neighbor_mean = sum(neighbor_values) / len(neighbor_values)
            else:
                neighbor_best = best
                neighbor_mean = mean
            archive_best = (
                self._archive_best_scalar_by_weight[weight_idx]
                if weight_idx < len(self._archive_best_scalar_by_weight)
                else best
            )
            summaries.append((best, mean, variance ** 0.5, neighbor_best, neighbor_mean, archive_best))
        self._particle_direction_summary = tuple(summaries)
        self._clear_neural_bias_cache()

    def _base_scalar2_from_norm(self, z0: float, z1: float, weight_idx: int) -> float:
        a = self._weight0[weight_idx] * z0
        b = self._weight1[weight_idx] * z1
        return (a if a >= b else b) + 0.03 * (a + b)

    @staticmethod
    def _normalized_hv2(points: Sequence[Tuple[float, float]]) -> float:
        nondominated = []
        best_y = float("inf")
        for x, y in sorted(points):
            if x > 1.35 or y > 1.35:
                continue
            if y < best_y:
                nondominated.append((x, y))
                best_y = y
        hv = 0.0
        ref_x, ref_y = 1.12, 1.12
        prev_y = ref_y
        for x, y in nondominated:
            hv += max(0.0, ref_x - x) * max(0.0, prev_y - y)
            prev_y = min(prev_y, y)
        return hv

    def _fit_neural_scalar(self) -> None:
        if (
            self._neural_scalar is None
            or not self.archive.entries
            or self.neural_training_epochs <= 0
            or not self.neural_online_training
        ):
            return
        objectives = list(self.objectives)
        archive_objectives = [entry.objectives for entry in self.archive.entries]
        for _ in range(self.neural_archive_repeats):
            objectives.extend(archive_objectives)
        inputs = []
        targets = []
        for objective in objectives:
            for weight_idx in range(len(self.weights)):
                repeat = self._neural_weight_repeat(weight_idx)
                features = self._neural_features(objective, weight_idx)
                target = self._neural_state_target(objective, weight_idx)
                for _ in range(repeat):
                    inputs.append(features)
                    targets.append(target)
        self._neural_training_samples = len(inputs)
        spectral_before = self._neural_scalar.spectral_diagnostics(self.neural_weight_norm_bound)
        residual_pairs: List[Tuple[Tuple[float, ...], Tuple[float, ...], float]] = []
        ranking_pairs: List[Tuple[Tuple[float, ...], Tuple[float, ...], float]] = []
        hypercone_pairs: List[Tuple[Tuple[float, ...], Tuple[float, ...], float]] = []
        coverage_pairs: List[Tuple[Tuple[float, ...], Tuple[float, ...], float]] = []
        expert_pairs: List[Tuple[Tuple[float, ...], Tuple[float, ...], float]] = []
        if self.neural_mean_field_features:
            self._collect_neural_action_training_pairs(
                inputs,
                targets,
                residual_pairs,
                ranking_pairs,
                hypercone_pairs,
                coverage_pairs,
                expert_pairs,
            )
            self._neural_scalar.fit_mixed(
                inputs,
                targets,
                residual_pairs,
                ranking_pairs,
                hypercone_pairs,
                self.neural_training_epochs,
                self.neural_learning_rate,
                self.neural_flow_residual_weight,
                self.neural_ranking_weight,
                self.neural_hypercone_loss_weight,
                self.neural_weight_norm_bound,
                coverage_pairs,
                expert_pairs,
                self.neural_coverage_pair_weight,
                self.neural_expert_pair_weight,
            )
        else:
            self._neural_scalar.fit(inputs, targets, self.neural_training_epochs, self.neural_learning_rate)
            self._neural_scalar.clip_weight_norms(self.neural_weight_norm_bound)
        spectral_after = self._neural_scalar.spectral_diagnostics(self.neural_weight_norm_bound)
        self._last_neural_spectral_diagnostics = {
            "eval": self._used_evaluations(),
            "archive_flush": self._archive_flush_count,
            "archive_size": len(self.archive),
            "neural_backend": self._neural_backend_name(),
            "training_samples": len(inputs),
            "residual_pairs": len(residual_pairs),
            "ranking_pairs": len(ranking_pairs),
            "hypercone_pairs": len(hypercone_pairs),
            "coverage_pairs": len(coverage_pairs),
            "expert_pairs": len(expert_pairs),
            "mean_field_features": self.neural_mean_field_features,
            "training_epochs": self.neural_training_epochs,
            "flow_residual_weight": self.neural_flow_residual_weight,
            "ranking_weight": self.neural_ranking_weight,
            "hypercone_weight": self.neural_hypercone_loss_weight,
            "coverage_weight": self.neural_coverage_pair_weight,
            "expert_weight": self.neural_expert_pair_weight,
            "before": spectral_before,
            "after": spectral_after,
        }
        self._neural_coverage_pairs += len(coverage_pairs)
        self._neural_expert_pairs += len(expert_pairs)
        self._write_neural_spectral_log(self._last_neural_spectral_diagnostics)
        self._clear_neural_bias_cache()

    def _write_neural_spectral_log(self, payload: dict) -> None:
        path = os.environ.get("MO_NCO_NEURAL_SPECTRAL_LOG", "")
        if not path:
            return
        try:
            log_path = Path(path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception:
            return

    def neural_training_examples(self) -> Tuple[List[Tuple[float, ...]], List[float]]:
        inputs: List[Tuple[float, ...]] = []
        targets: List[float] = []
        if self.instance.num_objectives != 2 or not self.archive.entries:
            return inputs, targets
        objectives = list(self.objectives) + [entry.objectives for entry in self.archive.entries]
        for objective in objectives:
            for weight_idx in range(len(self.weights)):
                repeat = self._neural_weight_repeat(weight_idx)
                features = self._neural_features(objective, weight_idx)
                target = self._neural_state_target(objective, weight_idx)
                for _ in range(repeat):
                    inputs.append(features)
                    targets.append(target)
        return inputs, targets

    def _collect_neural_action_training_pairs(
        self,
        inputs: List[Tuple[float, ...]],
        targets: List[float],
        residual_pairs: List[Tuple[Tuple[float, ...], Tuple[float, ...], float]],
        ranking_pairs: List[Tuple[Tuple[float, ...], Tuple[float, ...], float]],
        hypercone_pairs: List[Tuple[Tuple[float, ...], Tuple[float, ...], float]],
        coverage_pairs: List[Tuple[Tuple[float, ...], Tuple[float, ...], float]],
        expert_pairs: List[Tuple[Tuple[float, ...], Tuple[float, ...], float]],
    ) -> None:
        total_pair_samples = max(self.neural_flow_pair_samples, self.neural_expert_pair_samples)
        if total_pair_samples <= 0 or not self.population:
            return
        for _ in range(total_pair_samples):
            parent_idx = self.rng.randrange(len(self.population))
            direction_idx = self.rng.randrange(len(self.weights))
            parent = self.population[parent_idx]
            parent_obj = self.objectives[parent_idx]
            parent_features = self._neural_features(parent_obj, direction_idx)
            parent_target = self._neural_state_target(parent_obj, direction_idx)
            first = self._sample_raw_neural_candidate(parent, parent_obj)
            second = self._sample_raw_neural_candidate(parent, parent_obj)
            if first is not None:
                _, _, _, first_obj = first
                first_features = self._neural_features(first_obj, direction_idx)
                first_target = self._neural_state_target(first_obj, direction_idx)
                residual_pairs.append((parent_features, first_features, first_target - parent_target))
                inputs.append(first_features)
                targets.append(first_target)
            if first is None or second is None:
                continue
            _, _, _, first_obj = first
            _, _, _, second_obj = second
            first_features = self._neural_features(first_obj, direction_idx)
            second_features = self._neural_features(second_obj, direction_idx)
            first_target = self._neural_state_target(first_obj, direction_idx)
            second_target = self._neural_state_target(second_obj, direction_idx)
            diff = abs(first_target - second_target)
            if diff > 1e-9:
                margin = min(0.08, 0.01 + 0.25 * diff)
                if first_target < second_target:
                    ranking_pairs.append((first_features, second_features, margin))
                else:
                    ranking_pairs.append((second_features, first_features, margin))
            drift_first = self._directional_drift_from_norm(*self._normalize2(first_obj), direction_idx)
            drift_second = self._directional_drift_from_norm(*self._normalize2(second_obj), direction_idx)
            drift_diff = abs(drift_first - drift_second)
            if drift_diff > 0.03:
                extreme = self._weight_extremeness(direction_idx)
                margin = min(0.08, (0.01 + 0.2 * drift_diff) * (1.0 + extreme))
                if drift_first < drift_second:
                    hypercone_pairs.append((first_features, second_features, margin))
                else:
                    hypercone_pairs.append((second_features, first_features, margin))
            self._append_pareto_conditioned_action_pairs(
                parent,
                parent_obj,
                direction_idx,
                coverage_pairs,
                expert_pairs,
            )

    def _append_pareto_conditioned_action_pairs(
        self,
        parent: Tour,
        parent_obj: ObjectiveVector,
        direction_idx: int,
        coverage_pairs: List[Tuple[Tuple[float, ...], Tuple[float, ...], float]],
        expert_pairs: List[Tuple[Tuple[float, ...], Tuple[float, ...], float]],
    ) -> None:
        if (
            self.neural_expert_pair_samples <= 0
            or (self.neural_coverage_pair_weight <= 0.0 and self.neural_expert_pair_weight <= 0.0)
        ):
            return
        pool_size = max(2, self.neural_expert_pair_samples)
        if pool_size <= 1:
            return
        candidates = []
        for _ in range(pool_size):
            candidate = self._sample_raw_neural_candidate(parent, parent_obj)
            if candidate is not None:
                candidates.append(candidate)
        if len(candidates) < 2:
            return
        scored = [
            (self._pareto_conditioned_action_target(candidate[3], direction_idx, parent_obj), candidate)
            for candidate in candidates
        ]
        scored.sort(key=lambda item: item[0])
        best_score, best_candidate = scored[0]
        best_features = self._neural_features(best_candidate[3], direction_idx)
        for score, candidate in scored[1:]:
            diff = score - best_score
            if diff <= 1e-9:
                continue
            other_features = self._neural_features(candidate[3], direction_idx)
            margin = min(0.12, 0.01 + 0.20 * diff)
            expert_pairs.append((best_features, other_features, margin))
            best_gap = self._archive_gap_value_from_norm(*self._normalize2(best_candidate[3]))
            other_gap = self._archive_gap_value_from_norm(*self._normalize2(candidate[3]))
            best_extreme = self._extreme_progress_from_norm(*self._normalize2(best_candidate[3]), direction_idx)
            other_extreme = self._extreme_progress_from_norm(*self._normalize2(candidate[3]), direction_idx)
            if best_gap + best_extreme > other_gap + other_extreme + 1e-3:
                coverage_pairs.append((best_features, other_features, min(0.10, 0.01 + 0.10 * diff)))

    def _pareto_conditioned_action_target(
        self,
        objective: ObjectiveVector,
        weight_idx: int,
        parent_objective: ObjectiveVector,
    ) -> float:
        z0, z1 = self._normalize2(objective)
        state_target = self._neural_state_target(objective, weight_idx)
        scalar_delta = self._base_scalar2_by_weight(objective, weight_idx) - self._base_scalar2_by_weight(
            parent_objective, weight_idx
        )
        gap_reward = self._archive_gap_value_from_norm(z0, z1)
        extreme_reward = self._weight_extremeness(weight_idx) * self._extreme_progress_from_norm(z0, z1, weight_idx)
        drift_penalty = self._directional_drift_from_norm(z0, z1, weight_idx)
        return state_target + 0.35 * scalar_delta + 0.20 * drift_penalty - 0.35 * gap_reward - 0.25 * extreme_reward

    def _neural_state_target(self, objective: ObjectiveVector, weight_idx: int) -> float:
        terms = self._archive_bias_terms2(objective)
        if terms is None:
            return 0.0
        z0, z1, hv_gain, novelty = terms
        target = self._archive_improvement_bias_from_terms2(z0, z1, hv_gain, novelty, weight_idx)
        if self.neural_mean_field_features and self.neural_mean_field_target_weight > 0.0:
            target -= self.neural_mean_field_target_weight * self._mean_field_target_reward_from_norm(z0, z1, weight_idx)
        return target

    def _mean_field_target_reward_from_norm(self, z0: float, z1: float, weight_idx: int) -> float:
        if weight_idx >= len(self._particle_direction_summary):
            return 0.0
        candidate_scalar = self._base_scalar2_from_norm(z0, z1, weight_idx)
        best, mean_value, std_value, neighbor_best, neighbor_mean, archive_best = self._particle_direction_summary[
            weight_idx
        ]
        local_progress = max(0.0, neighbor_best - candidate_scalar)
        if local_progress <= 0.0:
            return 0.0
        archive_deficit = max(0.0, neighbor_best - archive_best)
        dispersion = max(0.0, min(1.0, std_value / (1.0 + std_value)))
        neighborhood_deficit = max(0.0, neighbor_mean - best)
        progress_term = local_progress / (1.0 + local_progress)
        archive_term = archive_deficit / (1.0 + archive_deficit)
        neighborhood_term = neighborhood_deficit / (1.0 + neighborhood_deficit)
        return progress_term * (0.5 + 0.35 * archive_term + 0.15 * neighborhood_term) * (0.75 + 0.25 * dispersion)

    def _neural_weight_repeat(self, weight_idx: int) -> int:
        if self.neural_extreme_repeats <= 0:
            return 1
        return 1 + int(round(self.neural_extreme_repeats * self._weight_extremeness(weight_idx)))

    def _predict_neural_scalar_bias(
        self,
        objective: ObjectiveVector,
        weight_idx: int,
        parent_objective: Optional[ObjectiveVector] = None,
    ) -> float:
        if self._neural_scalar is None:
            return 0.0
        if parent_objective is not None:
            features = self._neural_features(objective, weight_idx, parent_objective)
            self._neural_scalar_cache_misses += 1
            return max(-1.0, min(0.0, self._predict_neural_batch([features])[0]))
        return self._predict_neural_scalar_biases([(objective, weight_idx)])[0]

    def _predict_neural_scalar_biases(
        self,
        pairs: Sequence[Tuple[ObjectiveVector, int]],
    ) -> List[float]:
        """Score endpoint/weight pairs with one batched backend call per cache miss set."""

        if self._neural_scalar is None or not pairs:
            return [0.0 for _ in pairs]
        results = [0.0 for _ in pairs]
        missing_by_key: dict[Tuple[int, float, float], List[int]] = {}
        request_by_key: dict[Tuple[int, float, float], Tuple[ObjectiveVector, int]] = {}
        for pos, (objective, weight_idx) in enumerate(pairs):
            key = (weight_idx, float(objective[0]), float(objective[1]))
            cached = self._neural_bias_cache.get(key)
            if cached is not None:
                self._neural_scalar_cache_hits += 1
                results[pos] = cached
                continue
            self._neural_scalar_cache_misses += 1
            missing_by_key.setdefault(key, []).append(pos)
            request_by_key[key] = (objective, weight_idx)
        if not missing_by_key:
            return results
        keys = list(missing_by_key)
        features = [
            self._neural_features(request_by_key[key][0], request_by_key[key][1])
            for key in keys
        ]
        predicted = self._predict_neural_batch(features)
        if len(self._neural_bias_cache) + len(keys) > 100_000:
            self._neural_bias_cache.clear()
        for key, raw_value in zip(keys, predicted):
            value = max(-1.0, min(0.0, raw_value))
            self._neural_bias_cache[key] = value
            for pos in missing_by_key[key]:
                results[pos] = value
        return results

    def _predict_neural_batch(self, features: Sequence[Tuple[float, ...]]) -> List[float]:
        if self._neural_scalar is None or not features:
            return [0.0 for _ in features]
        started = time.perf_counter()
        self._neural_scalar_forward_calls += 1
        self._neural_scalar_scored_states += len(features)
        conditioned = list(self._neural_scalar.predict_batch(features))
        if self.neural_condition_guidance_scale == 1.0:
            self._neural_scalar_inference_seconds += time.perf_counter() - started
            return conditioned
        unconditioned_features = [self._unconditioned_neural_features(item) for item in features]
        unconditioned = list(self._neural_scalar.predict_batch(unconditioned_features))
        scale = self.neural_condition_guidance_scale
        values = [uncond + scale * (cond - uncond) for cond, uncond in zip(conditioned, unconditioned)]
        self._neural_scalar_inference_seconds += time.perf_counter() - started
        return values

    @staticmethod
    def _unconditioned_neural_features(features: Tuple[float, ...]) -> Tuple[float, ...]:
        # Keep state/action displacement coordinates and remove reference, target,
        # archive, and mean-field context.  This is classifier-free-style
        # guidance over finite state features, not a new gradient-flow claim.
        keep = {0, 1}
        if len(features) >= 20:
            keep.update({2, 3, 12, 13})
        return tuple(value if idx in keep else 0.0 for idx, value in enumerate(features))

    def _neural_features(
        self,
        objective: ObjectiveVector,
        weight_idx: int,
        parent_objective: Optional[ObjectiveVector] = None,
    ) -> Tuple[float, ...]:
        z0 = (objective[0] - self._ideal0) * self._inv0
        z1 = (objective[1] - self._ideal1) * self._inv1
        base_features = (z0, z1, self._weight0[weight_idx], self._weight1[weight_idx])
        input_dim = self._neural_scalar.input_dim if self._neural_scalar is not None else 6
        if input_dim <= 4:
            return base_features
        terms = self._archive_bias_terms2(objective)
        if terms is None:
            hv_gain = 0.0
            novelty = 0.0
        else:
            _, _, hv_gain, novelty = terms
        if input_dim <= 6:
            return (*base_features, hv_gain, novelty)

        if parent_objective is None:
            dz0 = 0.0
            dz1 = 0.0
            scalar_delta = 0.0
        else:
            dz0 = (objective[0] - parent_objective[0]) * self._inv0
            dz1 = (objective[1] - parent_objective[1]) * self._inv1
            scalar_delta = self._base_scalar2_by_weight(objective, weight_idx) - self._base_scalar2_by_weight(
                parent_objective, weight_idx
            )
        gap_value = self._archive_gap_value_from_norm(z0, z1)
        drift = self._directional_drift_from_norm(z0, z1, weight_idx)
        extreme = self._weight_extremeness(weight_idx)
        extreme_progress = self._extreme_progress_from_norm(z0, z1, weight_idx)
        state_scalar = self._base_scalar2_by_weight(objective, weight_idx)
        if weight_idx < len(self._particle_direction_summary):
            best, mean_value, std_value, neighbor_best, neighbor_mean, archive_best = self._particle_direction_summary[
                weight_idx
            ]
        else:
            best = mean_value = neighbor_best = neighbor_mean = archive_best = state_scalar
            std_value = 0.0
        particle_summary = (
            state_scalar - best,
            state_scalar - mean_value,
            std_value / (1.0 + abs(std_value)),
            state_scalar - neighbor_best,
            state_scalar - neighbor_mean,
            state_scalar - archive_best,
        )
        features20 = (
            z0,
            z1,
            dz0,
            dz1,
            self._weight0[weight_idx],
            self._weight1[weight_idx],
            hv_gain,
            novelty,
            gap_value,
            drift,
            extreme,
            extreme_progress,
            state_scalar,
            scalar_delta,
            *particle_summary,
        )
        features20 = tuple(self._bounded_neural_feature(value) for value in features20)
        if input_dim > len(features20):
            target0, target1 = self._reference_target_point(weight_idx)
            target_dist = abs(z0 - target0) + abs(z1 - target1)
            front_weight = self._pcd_front_reweight_from_terms(
                hv_gain,
                novelty,
                gap_value,
                extreme,
                extreme_progress,
            )
            features24 = (
                *features20,
                self._bounded_neural_feature(target0),
                self._bounded_neural_feature(target1),
                self._bounded_neural_feature(target_dist),
                self._bounded_neural_feature(front_weight),
            )
            if input_dim <= len(features24):
                return features24[:input_dim]
            return (*features24, *((0.0,) * (input_dim - len(features24))))
        if input_dim <= len(features20):
            return features20[:input_dim]
        return (*features20, *((0.0,) * (input_dim - len(features20))))

    def _reference_target_point(self, weight_idx: int) -> Tuple[float, float]:
        if not self._weight0 or not self._weight1:
            return 0.5, 0.5
        w0 = self._weight0[weight_idx]
        w1 = self._weight1[weight_idx]
        total = max(1e-9, w0 + w1)
        # In normalized minimization space, a large weight on objective 0 means
        # the target should lean toward low z0 and tolerate larger z1.
        return w1 / total, w0 / total

    def _pcd_front_reweight_from_terms(
        self,
        hv_gain: float,
        novelty: float,
        gap_value: float,
        extreme: float,
        extreme_progress: float,
    ) -> float:
        if self.neural_front_reweighting_strength <= 0.0:
            return 0.0
        value = (
            0.20 * max(0.0, hv_gain)
            + 0.25 * max(0.0, novelty)
            + 0.30 * max(0.0, gap_value)
            + 0.15 * max(0.0, extreme * extreme_progress)
            + 0.10 * max(0.0, extreme)
        )
        return max(0.0, min(1.0, self.neural_front_reweighting_strength * value))

    @staticmethod
    def _bounded_neural_feature(value: float) -> float:
        if value != value:
            return 0.0
        return max(-2.0, min(2.0, float(value)))

    def _accept_delta(self, delta: float, step: int) -> bool:
        if delta <= 0.0:
            return True
        temperature = self._temperature(step)
        if temperature <= 0.0:
            return False
        log_alpha = -delta / temperature
        draw = self.rng.random()
        log_draw = -math.inf if draw == 0.0 else math.log(draw)
        return log_draw < log_alpha

    def _temperature(self, step: int) -> float:
        if self.initial_temperature <= 0.0 and self.final_temperature <= 0.0:
            return 0.0
        horizon = max(1, self.evaluations - len(self.population) - 1)
        t = min(1.0, max(0.0, step / horizon))
        return (1.0 - t) * self.initial_temperature + t * self.final_temperature

    def _make_neural_proposal_child(
        self,
        parent: Tour,
        parent_obj: Optional[ObjectiveVector],
        direction_idx: int,
    ) -> Optional[Tuple[Tour, ObjectiveVector]]:
        scalar_ready = (
            self._neural_scalar is not None
            and self.neural_candidate_pool > 1
            and (self._neural_prior_loaded or self._neural_training_samples >= self.neural_proposal_min_samples)
        )
        move_ready = self._learned_move_generator is not None and self.neural_learned_move_probability > 0.0
        gate_eligible = not (
            not self._neural_is_active()
            or parent_obj is None
            or self.instance.num_objectives != 2
            or self.proposal != "two_opt"
            or self.extra_two_opt_probability > 0.0
            or self.neural_proposal_probability <= 0.0
            or not (scalar_ready or (move_ready and self.allow_move_without_scalar))
        )
        if not gate_eligible:
            return None
        if self.enable_mechanism_diagnostics:
            self._neural_gate_eligible_steps += 1
        if self.rng.random() >= self.neural_proposal_probability:
            return None
        if self.enable_mechanism_diagnostics:
            self._neural_gate_fired_steps += 1
        best: Optional[Tuple[float, str, int, int, ObjectiveVector]] = None
        learned_update: Optional[
            Tuple[Tuple[float, ...], Tuple[float, ...], float, float, float, float]
        ] = None
        if (
            move_ready
            and self.rng.random() < self.neural_learned_move_probability
        ):
            learned_candidate = self._sample_learned_move_candidate(parent, parent_obj, direction_idx)
            if learned_candidate is not None:
                (
                    kind,
                    i,
                    j,
                    candidate_obj,
                    first_features,
                    second_features,
                    target_advantage,
                    flow_advantage,
                    mean_field_advantage,
                    conductance_advantage,
                    reward,
                ) = learned_candidate
                score = -reward
                best = (score, kind, i, j, candidate_obj)
                learned_update = (
                    first_features,
                    second_features,
                    target_advantage,
                    flow_advantage,
                    mean_field_advantage,
                    conductance_advantage,
                )
        if scalar_ready and learned_update is not None and self.neural_backend == "pcd":
            if self.enable_mechanism_diagnostics:
                self._scalar_proposal_suppressed_by_learned_move += 1
        elif scalar_ready:
            if self.neural_prefilter_pool > 1:
                candidate = self._sample_neural_policy_candidate(parent, parent_obj, direction_idx)
                if candidate is None and best is None:
                    return None
                if candidate is not None:
                    kind, i, j, candidate_obj = candidate
                    score = self._neural_proposal_score2(candidate_obj, direction_idx, parent_obj)
                    if best is None or score < best[0]:
                        best = (score, kind, i, j, candidate_obj)
                        learned_update = None
            else:
                scalar_objectives: List[ObjectiveVector] = []
                scalar_analytic_scores: List[float] = []
                scalar_treatment_scores: List[float] = []
                for _ in range(self.neural_candidate_pool):
                    candidate = self._sample_neural_candidate(parent, parent_obj, direction_idx)
                    if candidate is None:
                        continue
                    kind, i, j, candidate_obj = candidate
                    analytic_score = self._analytic_proposal_score2(candidate_obj, direction_idx)
                    score = self._neural_proposal_score2(candidate_obj, direction_idx, parent_obj)
                    scalar_objectives.append(candidate_obj)
                    scalar_analytic_scores.append(analytic_score)
                    scalar_treatment_scores.append(score)
                    if best is None or score < best[0]:
                        best = (score, kind, i, j, candidate_obj)
                        learned_update = None
                self._record_scalar_candidate_decision(
                    scalar_objectives,
                    direction_idx,
                    scalar_analytic_scores,
                    scalar_treatment_scores,
                )
        if best is None:
            return None
        _, kind, i, j, _ = best
        if kind == "two_opt":
            child = two_opt_at(parent, i, j)
        else:
            child = self._relocate_at(parent, i, j)
        child = self._scalar_local_descent(
            child,
            direction_idx,
            self.proposal_2opt_passes,
            self.proposal_relocate_passes,
            self.proposal_swap_passes,
        )
        method = getattr(self.instance, "evaluate_two_opt", None)
        child_obj = (
            self._evaluate_two_opt(parent, parent_obj, i, j)
            if (
                kind == "two_opt"
                and callable(method)
                and self.proposal_2opt_passes == 0
                and self.proposal_relocate_passes == 0
                and self.proposal_swap_passes == 0
            )
            else self._evaluate(child)
        )
        if (
            learned_update is not None
            and self._learned_move_generator is not None
            and self.neural_online_training
        ):
            self._learned_move_generator.update_joint(*learned_update)
            self._neural_learned_move_updates += 1
        self._last_neural_proposal_source = "move" if learned_update is not None else "scalar"
        return child, child_obj

    def _sample_learned_move_candidate(
        self,
        parent: Tour,
        parent_obj: ObjectiveVector,
        direction_idx: int,
    ) -> Optional[
        Tuple[
            str,
            int,
            int,
            ObjectiveVector,
            Tuple[float, ...],
            Tuple[float, ...],
            float,
            float,
            float,
            float,
            float,
        ]
    ]:
        generator = self._learned_move_generator
        if generator is None or self.instance.num_objectives != 2:
            return None
        matrices = self._distance_matrices()
        symmetric = self._symmetric_objectives()
        if matrices is None or symmetric is None or len(matrices) != 2 or not all(symmetric):
            return None
        self._neural_learned_move_calls += 1
        reference = (self._weight0[direction_idx], self._weight1[direction_idx])
        target = self._reference_target_point(direction_idx)
        context = self._learned_move_context(parent_obj, direction_idx)
        samples = generator.sample_two_opt_candidates(
            parent,
            parent_obj,
            matrices,
            reference,
            target,
            self._edge_scale0,
            self._edge_scale1,
            self.neural_learned_move_sparse_nodes,
            self.neural_learned_move_sparse_partners,
            context,
            self.neural_learned_move_samples,
            self.neural_condition_guidance_scale,
        )
        if not samples:
            return None
        pairs = [(i, j) for i, j, _, _ in samples]
        candidate_objs = self._uncounted_symmetric_two_opt_batch_objectives(parent, parent_obj, pairs)
        viable: List[dict] = []
        for (i, j, first_features, second_features), candidate_obj in zip(samples, candidate_objs):
            if candidate_obj is None:
                continue
            (
                target_advantage,
                flow_advantage,
                mean_field_advantage,
                conductance_advantage,
                reward,
                basin_crossing,
            ) = self._learned_move_action_components(candidate_obj, direction_idx, parent_obj)
            angle_penalty = self._learned_move_angle_penalty(candidate_obj, direction_idx, parent_obj)
            self._neural_learned_move_candidate_count += 1
            self._neural_learned_move_reward_sum += reward
            self._neural_learned_move_angle_penalty_sum += angle_penalty
            self._neural_learned_move_mf_reward_sum += mean_field_advantage
            if reward > 0.0:
                self._neural_learned_move_positive_rewards += 1
            if angle_penalty <= 0.35:
                self._neural_learned_move_cone_pass_count += 1
            viable.append(
                {
                    "policy_score": generator.action_score(first_features, second_features),
                    "angle_penalty": angle_penalty,
                    "i": i,
                    "j": j,
                    "objective": candidate_obj,
                    "first": first_features,
                    "second": second_features,
                    "target_advantage": target_advantage,
                    "flow_advantage": flow_advantage,
                    "mean_field_advantage": mean_field_advantage,
                    "conductance_advantage": conductance_advantage,
                    "reward": reward,
                    "basin_crossing": basin_crossing,
                }
            )
        if not viable:
            return None
        probabilities = generator.action_probabilities([(item["first"], item["second"]) for item in viable])
        self._neural_sampled_pool_policy_reward_sum += sum(
            probability * float(item["reward"]) for probability, item in zip(probabilities, viable)
        )
        self._neural_sampled_pool_uniform_reward_sum += sum(float(item["reward"]) for item in viable) / len(viable)
        self._neural_sampled_pool_reward_observations += 1
        best_reward = max(float(item["reward"]) for item in viable)
        good_threshold = max(0.0, best_reward - 0.05)
        good_indices = [
            idx
            for idx, item in enumerate(viable)
            if float(item["reward"]) >= good_threshold and float(item["reward"]) > 0.0
        ]
        crossing_indices = [
            idx
            for idx, item in enumerate(viable)
            if bool(item["basin_crossing"]) and float(item["target_advantage"]) > 0.0
        ]
        self._neural_learned_move_mass_observations += 1
        self._neural_learned_move_good_mass_sum += sum(probabilities[idx] for idx in good_indices)
        self._neural_baseline_good_mass_sum += len(good_indices) / len(viable)
        self._neural_learned_move_crossing_mass_sum += sum(probabilities[idx] for idx in crossing_indices)
        self._neural_baseline_crossing_mass_sum += len(crossing_indices) / len(viable)
        angle_filtered = [item for item in viable if float(item["angle_penalty"]) <= 0.35]
        if angle_filtered:
            self._neural_learned_move_angle_filtered += len(viable) - len(angle_filtered)
            viable = angle_filtered
        best = max(
            viable,
            key=lambda item: float(item["policy_score"]) - 0.05 * float(item["angle_penalty"]),
        )
        self._record_learned_move_selection(viable, best)
        self._neural_learned_move_children += 1
        return (
            "two_opt",
            int(best["i"]),
            int(best["j"]),
            best["objective"],
            best["first"],
            best["second"],
            float(best["target_advantage"]),
            float(best["flow_advantage"]),
            float(best["mean_field_advantage"]),
            float(best["conductance_advantage"]),
            float(best["reward"]),
        )

    def _learned_move_context(self, objective: ObjectiveVector, direction_idx: int) -> Tuple[float, ...]:
        z0, z1 = self._normalize2(objective)
        target0, target1 = self._reference_target_point(direction_idx)
        return (
            self._archive_gap_value_from_norm(z0, z1),
            self._learned_move_mean_field_reward(objective, direction_idx),
            self._weight_extremeness(direction_idx),
            abs(z0 - target0) + abs(z1 - target1),
        )

    def _learned_move_action_reward(
        self,
        objective: ObjectiveVector,
        direction_idx: int,
        parent_objective: ObjectiveVector,
    ) -> float:
        return self._learned_move_action_components(objective, direction_idx, parent_objective)[4]

    def _learned_move_action_components(
        self,
        objective: ObjectiveVector,
        direction_idx: int,
        parent_objective: ObjectiveVector,
    ) -> Tuple[float, float, float, float, float, bool]:
        z0, z1 = self._normalize2(objective)
        p0, p1 = self._normalize2(parent_objective)
        target0, target1 = self._reference_target_point(direction_idx)
        distance = abs(z0 - target0) + abs(z1 - target1)
        parent_distance = abs(p0 - target0) + abs(p1 - target1)
        target_progress = parent_distance - distance
        dominance_progress = max(0.0, p0 - z0) + max(0.0, p1 - z1)
        archive_hv_increment = self._normalized_archive_hv_gain2(objective)
        hv_signal = archive_hv_increment / (0.01 + archive_hv_increment)
        target_advantage = max(
            -1.0,
            min(1.0, 0.65 * hv_signal + 0.20 * target_progress + 0.15 * dominance_progress),
        )
        if self.neural_flow_residual_weight > 0.0:
            parent_flow = self._neural_state_target(parent_objective, direction_idx)
            child_flow = self._neural_state_target(objective, direction_idx)
            flow_advantage = max(-1.0, min(1.0, parent_flow - child_flow))
        else:
            flow_advantage = 0.0
        parent_mf = self._learned_move_mean_field_reward(parent_objective, direction_idx)
        child_mf = self._learned_move_mean_field_reward(objective, direction_idx)
        mean_field_advantage = max(-1.0, min(1.0, child_mf - parent_mf))
        basin_crossing = self._learned_move_basin_id(objective) != self._learned_move_basin_id(parent_objective)
        conductance_advantage = (
            max(0.0, target_advantage + 0.25 * flow_advantage) if basin_crossing else 0.0
        )
        reward = (
            0.70 * target_advantage
            + 0.15 * flow_advantage
            + 0.10 * mean_field_advantage
            + 0.05 * conductance_advantage
        )
        return (
            target_advantage,
            flow_advantage,
            mean_field_advantage,
            max(-1.0, min(1.0, conductance_advantage)),
            max(-1.0, min(1.0, reward)),
            basin_crossing,
        )

    def _learned_move_basin_id(self, objective: ObjectiveVector, bins: int = 6) -> Tuple[int, int]:
        z0, z1 = self._normalize2(objective)
        return (
            min(bins - 1, max(0, int(max(0.0, min(0.999999, z0)) * bins))),
            min(bins - 1, max(0, int(max(0.0, min(0.999999, z1)) * bins))),
        )

    def _learned_move_mean_field_reward(self, objective: ObjectiveVector, direction_idx: int) -> float:
        if not self.neural_mean_field_features:
            return 0.0
        z0, z1 = self._normalize2(objective)
        return self._mean_field_target_reward_from_norm(z0, z1, direction_idx)

    def _learned_move_angle_penalty(
        self,
        objective: ObjectiveVector,
        direction_idx: int,
        parent_objective: ObjectiveVector,
    ) -> float:
        z0, z1 = self._normalize2(objective)
        p0, p1 = self._normalize2(parent_objective)
        target0, target1 = self._reference_target_point(direction_idx)
        gain0 = p0 - z0
        gain1 = p1 - z1
        desired0 = p0 - target0
        desired1 = p1 - target1
        gain_norm = math.sqrt(gain0 * gain0 + gain1 * gain1)
        desired_norm = math.sqrt(desired0 * desired0 + desired1 * desired1)
        if gain_norm <= 1e-12 or desired_norm <= 1e-12:
            return 0.5
        cos_value = (gain0 * desired0 + gain1 * desired1) / max(1e-12, gain_norm * desired_norm)
        cos_value = max(-1.0, min(1.0, cos_value))
        return 0.5 * (1.0 - cos_value)

    def _sample_neural_candidate(
        self,
        parent: Tour,
        parent_obj: ObjectiveVector,
        direction_idx: int,
    ) -> Optional[Tuple[str, int, int, ObjectiveVector]]:
        if self.neural_action_sample_pool > 1:
            return self._sample_neural_policy_candidate(parent, parent_obj, direction_idx)
        if (
            self.proposal_relocate_passes > 0
            and self.neural_relocate_candidate_probability > 0.0
            and self.rng.random() < self.neural_relocate_candidate_probability
        ):
            relocate = self._sample_relocate_indices(len(parent))
            if relocate is not None:
                i, j = relocate
                candidate_obj = self._uncounted_relocate_objectives(parent, parent_obj, i, j)
                if candidate_obj is not None:
                    return "relocate", i, j, candidate_obj
        i, j = sample_two_opt_indices(len(parent), self.rng)
        candidate_obj = self._uncounted_symmetric_two_opt_objectives(parent, parent_obj, i, j)
        if candidate_obj is not None:
            return "two_opt", i, j, candidate_obj
        relocate = self._sample_relocate_indices(len(parent))
        if relocate is None:
            return None
        i, j = relocate
        candidate_obj = self._uncounted_relocate_objectives(parent, parent_obj, i, j)
        if candidate_obj is None:
            return None
        return "relocate", i, j, candidate_obj

    def _sample_neural_policy_candidate(
        self,
        parent: Tour,
        parent_obj: ObjectiveVector,
        direction_idx: int,
    ) -> Optional[Tuple[str, int, int, ObjectiveVector]]:
        self._neural_policy_calls += 1
        if self.neural_exact_two_opt_prefilter:
            exact_candidate = self._sample_exact_two_opt_prefilter_candidate(parent, parent_obj, direction_idx)
            if exact_candidate is not None:
                return exact_candidate
        pool: List[Tuple[float, Tuple[str, int, int, ObjectiveVector]]] = []
        pool_size = max(self.neural_action_sample_pool, self.neural_prefilter_pool)
        for _ in range(pool_size):
            candidate = self._sample_raw_neural_candidate(parent, parent_obj)
            if candidate is None:
                continue
            score = self._analytic_proposal_score2(candidate[3], direction_idx)
            pool.append((score, candidate))
        if not pool:
            return None
        self._neural_raw_candidates += len(pool)
        ranked = heapq.nsmallest(
            min(self.neural_refine_top_k, len(pool)),
            pool,
            key=lambda item: item[0],
        )
        top_k = [candidate for _, candidate in ranked]
        base_scores = [score for score, _ in ranked]
        self._neural_refined_candidates += len(top_k)
        scores = self._neural_batched_action_scores(parent_obj, top_k, direction_idx, base_scores)
        self._neural_score_batches += 1
        self._record_scalar_candidate_decision(
            [candidate[3] for candidate in top_k],
            direction_idx,
            base_scores,
            scores,
        )
        best_idx = min(range(len(top_k)), key=scores.__getitem__)
        kind, i, j, candidate_obj = top_k[best_idx]
        return kind, i, j, candidate_obj

    def _sample_exact_two_opt_prefilter_candidate(
        self,
        parent: Tour,
        parent_obj: ObjectiveVector,
        direction_idx: int,
    ) -> Optional[Tuple[str, int, int, ObjectiveVector]]:
        matrix = self._weighted_matrix2(direction_idx)
        if matrix is None:
            return None
        symmetric = self._symmetric_objectives()
        if symmetric is None or not all(symmetric):
            return None
        try:
            import numpy as np

            current = np.asarray(parent, dtype=np.int64)
            n = len(current)
            if n < 4:
                return None
            a_nodes = current[: n - 2]
            b_nodes = current[1 : n - 1]
            c_nodes = current[2:n]
            d_nodes = np.empty_like(c_nodes)
            if len(c_nodes) > 1:
                d_nodes[:-1] = current[3:n]
            d_nodes[-1] = current[0]
            deltas = (
                matrix[a_nodes[:, None], c_nodes[None, :]]
                + matrix[b_nodes[:, None], d_nodes[None, :]]
                - matrix[a_nodes, b_nodes][:, None]
                - matrix[c_nodes, d_nodes][None, :]
            )
            deltas[np.tri(deltas.shape[0], deltas.shape[1], k=-1, dtype=bool)] = float("inf")
            flat_count = deltas.size
            if flat_count <= 0:
                return None
            candidate_count = min(max(self.neural_prefilter_pool, self.neural_action_sample_pool), flat_count)
            flat = deltas.ravel()
            if candidate_count < flat_count:
                flat_indices = np.argpartition(flat, candidate_count - 1)[:candidate_count]
                flat_indices = flat_indices[np.argsort(flat[flat_indices])]
            else:
                flat_indices = np.argsort(flat)
            candidates: List[Tuple[str, int, int, ObjectiveVector]] = []
            base_scores: List[float] = []
            for flat_idx in flat_indices.tolist():
                delta = float(flat[flat_idx])
                if not np.isfinite(delta):
                    continue
                row, col = divmod(int(flat_idx), deltas.shape[1])
                i, j = row + 1, col + 2
                candidate_obj = self._uncounted_symmetric_two_opt_objectives(parent, parent_obj, i, j)
                if candidate_obj is None:
                    continue
                candidate = ("two_opt", i, j, candidate_obj)
                candidates.append(candidate)
                base_scores.append(self._analytic_proposal_score2(candidate_obj, direction_idx))
                if len(candidates) >= max(1, self.neural_refine_top_k):
                    break
            if not candidates:
                return None
            scores = self._neural_batched_action_scores(parent_obj, candidates, direction_idx, base_scores)
            self._neural_raw_candidates += candidate_count
            self._neural_refined_candidates += len(candidates)
            self._neural_score_batches += 1
            self._record_scalar_candidate_decision(
                [candidate[3] for candidate in candidates],
                direction_idx,
                base_scores,
                scores,
            )
            best_idx = min(range(len(candidates)), key=scores.__getitem__)
            return candidates[best_idx]
        except Exception:
            return None

    def _neural_batched_action_scores(
        self,
        parent_obj: ObjectiveVector,
        candidates: Sequence[Tuple[str, int, int, ObjectiveVector]],
        direction_idx: int,
        base_scores: Optional[Sequence[float]] = None,
    ) -> List[float]:
        if base_scores is None:
            base_scores = [self._analytic_proposal_score2(candidate[3], direction_idx) for candidate in candidates]
        if not self._neural_is_active() or self._neural_scalar is None or self.neural_proposal_weight <= 0.0:
            return list(base_scores)
        features = [self._neural_features(candidate[3], direction_idx, parent_obj) for candidate in candidates]
        predictions = self._predict_neural_batch(features)
        if self.neural_rank_fusion_weight > 0.0 and len(predictions) > 1:
            neural_values = [max(-1.0, min(0.0, pred)) for pred in predictions]
            if max(neural_values) - min(neural_values) > 1e-9:
                base_rank = self._normalized_rank_scores(base_scores)
                neural_rank = self._normalized_rank_scores(neural_values)
                mix = self.neural_rank_fusion_weight
                scores = [
                    (1.0 - mix) * b + mix * n
                    for b, n in zip(base_rank, neural_rank)
                ]
                if min(range(len(base_scores)), key=list(base_scores).__getitem__) != min(range(len(scores)), key=scores.__getitem__):
                    self._neural_rank_changed_decisions += 1
                return self._apply_mean_field_guidance_scores(candidates, direction_idx, scores)
        scores = [
            base + self.neural_proposal_weight * max(-1.0, min(0.0, pred))
            for base, pred in zip(base_scores, predictions)
        ]
        return self._apply_mean_field_guidance_scores(candidates, direction_idx, scores)

    def _apply_mean_field_guidance_scores(
        self,
        candidates: Sequence[Tuple[str, int, int, ObjectiveVector]],
        direction_idx: int,
        scores: Sequence[float],
    ) -> List[float]:
        if (
            not self.neural_mean_field_features
            or self.neural_mean_field_guidance_weight <= 0.0
            or len(scores) <= 1
        ):
            return list(scores)
        mv_values = []
        for _, _, _, objective in candidates:
            z0, z1 = self._normalize2(objective)
            reward = self._mean_field_target_reward_from_norm(z0, z1, direction_idx)
            reward += self.neural_gap_fill_weight * self._archive_gap_value_from_norm(z0, z1)
            reward += self.neural_extreme_progress_weight * self._weight_extremeness(direction_idx) * self._extreme_progress_from_norm(
                z0, z1, direction_idx
            )
            mv_values.append(-reward)
        if max(mv_values) - min(mv_values) <= 1e-12:
            return list(scores)
        score_rank = self._normalized_rank_scores(scores)
        mv_rank = self._normalized_rank_scores(mv_values)
        mix = self.neural_mean_field_guidance_weight
        fused = [(1.0 - mix) * base + mix * mv for base, mv in zip(score_rank, mv_rank)]
        if min(range(len(scores)), key=list(scores).__getitem__) != min(range(len(fused)), key=fused.__getitem__):
            self._neural_mv_changed_decisions += 1
        return fused

    def _scalar_decision_diagnostics_active(self) -> bool:
        return (
            self.enable_mechanism_diagnostics
            and self._neural_is_active()
            and self._neural_scalar is not None
            and self.neural_scalar_weight > 0.0
        )

    def _record_scalar_argmin_decision(
        self,
        analytic_scores: Sequence[float],
        treatment_scores: Sequence[float],
        *,
        scope: str,
    ) -> None:
        if (
            not self.enable_mechanism_diagnostics
            or len(analytic_scores) <= 1
            or len(analytic_scores) != len(treatment_scores)
        ):
            return
        analytic_idx = min(range(len(analytic_scores)), key=analytic_scores.__getitem__)
        treatment_idx = min(range(len(treatment_scores)), key=treatment_scores.__getitem__)
        changed = int(analytic_idx != treatment_idx)
        regret = max(0.0, float(analytic_scores[treatment_idx]) - float(analytic_scores[analytic_idx]))
        if scope == "parent":
            self._scalar_parent_decision_observations += 1
            self._scalar_parent_changed_decisions += changed
            self._scalar_parent_analytic_score_regret_sum += regret
        elif scope == "archive_parent":
            self._scalar_archive_parent_decision_observations += 1
            self._scalar_archive_parent_changed_decisions += changed
            self._scalar_archive_parent_analytic_score_regret_sum += regret
        else:
            raise ValueError(f"Unknown scalar decision diagnostic scope: {scope}")

    def _record_scalar_replacement_preference(
        self,
        analytic_delta: float,
        treatment_delta: float,
    ) -> None:
        if not self.enable_mechanism_diagnostics:
            return
        self._scalar_replacement_preference_observations += 1
        analytic_accept = analytic_delta <= 0.0
        treatment_accept = treatment_delta <= 0.0
        if treatment_accept and not analytic_accept:
            self._scalar_replacement_flip_to_accept += 1
        elif analytic_accept and not treatment_accept:
            self._scalar_replacement_flip_to_reject += 1

    def _record_scalar_candidate_decision(
        self,
        candidate_objectives: Sequence[ObjectiveVector],
        direction_idx: int,
        analytic_scores: Sequence[float],
        treatment_scores: Sequence[float],
    ) -> None:
        count = len(candidate_objectives)
        if (
            not self.enable_mechanism_diagnostics
            or count <= 1
            or len(analytic_scores) != count
            or len(treatment_scores) != count
        ):
            return
        analytic_idx = min(range(count), key=analytic_scores.__getitem__)
        treatment_idx = min(range(count), key=treatment_scores.__getitem__)
        targets = [
            float(self._neural_state_target(objective, direction_idx))
            for objective in candidate_objectives
        ]
        selected_target = targets[treatment_idx]
        self._scalar_candidate_decision_observations += 1
        self._scalar_candidate_changed_decisions += int(analytic_idx != treatment_idx)
        self._scalar_candidate_target_margin_vs_analytic_sum += (
            targets[analytic_idx] - selected_target
        )
        self._scalar_candidate_target_margin_vs_pool_mean_sum += (
            sum(targets) / count - selected_target
        )
        self._scalar_candidate_target_regret_to_pool_oracle_sum += (
            selected_target - min(targets)
        )
        self._scalar_candidate_analytic_score_regret_sum += max(
            0.0,
            float(analytic_scores[treatment_idx]) - float(analytic_scores[analytic_idx]),
        )

    def _record_learned_move_selection(
        self,
        viable: Sequence[dict],
        selected: dict,
    ) -> None:
        if not self.enable_mechanism_diagnostics or not viable:
            return
        rewards = [float(item["reward"]) for item in viable]
        selected_reward = float(selected["reward"])
        self._learned_move_selection_observations += 1
        self._learned_move_selected_reward_sum += selected_reward
        self._learned_move_selected_pool_uniform_reward_sum += sum(rewards) / len(rewards)
        self._learned_move_selected_pool_oracle_reward_sum += max(rewards)

    @staticmethod
    def _normalized_rank_scores(values: Sequence[float]) -> List[float]:
        if len(values) <= 1:
            return [0.0 for _ in values]
        order = sorted(range(len(values)), key=lambda idx: (values[idx], idx))
        ranks = [0.0 for _ in values]
        denom = max(1, len(values) - 1)
        for rank, idx in enumerate(order):
            ranks[idx] = rank / denom
        return ranks

    def neural_inference_stats(self) -> dict:
        learned_candidates = max(1, self._neural_learned_move_candidate_count)
        mass_observations = max(1, self._neural_learned_move_mass_observations)
        learned_good_mass = self._neural_learned_move_good_mass_sum / mass_observations
        baseline_good_mass = self._neural_baseline_good_mass_sum / mass_observations
        learned_crossing_mass = self._neural_learned_move_crossing_mass_sum / mass_observations
        baseline_crossing_mass = self._neural_baseline_crossing_mass_sum / mass_observations
        reward_observations = max(1, self._neural_sampled_pool_reward_observations)
        policy_pool_reward = self._neural_sampled_pool_policy_reward_sum / reward_observations
        uniform_pool_reward = self._neural_sampled_pool_uniform_reward_sum / reward_observations
        scalar_candidate_observations = max(1, self._scalar_candidate_decision_observations)
        scalar_parent_observations = max(1, self._scalar_parent_decision_observations)
        scalar_archive_parent_observations = max(1, self._scalar_archive_parent_decision_observations)
        scalar_replacement_observations = max(1, self._scalar_replacement_preference_observations)
        move_selection_observations = max(1, self._learned_move_selection_observations)
        move_selected_reward = self._learned_move_selected_reward_sum / move_selection_observations
        move_pool_uniform_reward = (
            self._learned_move_selected_pool_uniform_reward_sum / move_selection_observations
        )
        move_pool_oracle_reward = (
            self._learned_move_selected_pool_oracle_reward_sum / move_selection_observations
        )
        stats = {
            "algorithm_contract": self.contract_name,
            "claim_level": self.claim_level.value,
            "algorithm_family": "nonautonomous_batch_descent",
            "random_stream_contract": (
                "base-search-v1|scalar-model-v1|move-policy-v1"
                if self.isolate_prior_loading_rng
                else "legacy-shared-rng"
            ),
            "prior_loading_rng_isolated": self.isolate_prior_loading_rng,
            "mechanism_diagnostics_enabled": self.enable_mechanism_diagnostics,
            "initial_population_sha256": self._initial_population_sha256,
            "base_rng_state_after_initialization_sha256": self._base_rng_state_after_initialization_sha256,
            "evaluations_used": self._used_evaluations(),
            "evaluation_budget": self.evaluations,
            "initial_population_size": len(self.population),
            "child_steps": self._child_steps,
            "positive_temperature": self.initial_temperature > 0.0 or self.final_temperature > 0.0,
            "context_refresh_count": self._context_refresh_count,
            "normalization_refresh_count": self._normalization_refresh_count,
            "archive_flush_count": self._archive_flush_count,
            "cumulative_positive_context_jump": self._cumulative_positive_context_jump,
            "cumulative_signed_context_jump": self._cumulative_signed_context_jump,
            "cumulative_positive_archive_context_jump": self._positive_context_jump_by_kind.get(
                "archive_neural_refresh",
                0.0,
            ),
            "positive_context_jump_by_kind": dict(self._positive_context_jump_by_kind),
            "signed_context_jump_by_kind": dict(self._signed_context_jump_by_kind),
            "context_jump_event_counts": dict(self._context_jump_event_counts),
            "context_jump_accounting_errors": self._context_jump_accounting_errors,
            "unattributed_compiled_energy_delta": self._unattributed_compiled_energy_delta,
            "unattributed_compiled_positive_delta": self._unattributed_compiled_positive_delta,
            "unattributed_compiled_event_count": self._unattributed_compiled_event_count,
            "context_jump_accounting_complete": (
                self._context_jump_accounting_errors == 0
                and self._unattributed_compiled_event_count == 0
            ),
            "context_jump_accounting_scope": (
                "complete for Python-path normalization, archive/neural, and mean-field refreshes; "
                "compiled-polish state/context changes are reported as unattributed aggregate deltas"
            ),
            "diagnostic_hypervolume_reference": getattr(self, "_diagnostic_hv_reference", None),
            "acceptance_computation": "log_uniform_comparison",
            "accelerator_fallbacks": tuple(self._accelerator_fallbacks),
            "ablation_contract": self.ablation_contract,
            "neural_backend": self._neural_backend_name(),
            "policy_calls": self._neural_policy_calls,
            "raw_candidates": self._neural_raw_candidates,
            "refined_candidates": self._neural_refined_candidates,
            "score_batches": self._neural_score_batches,
            "mean_raw_pool": self._neural_raw_candidates / max(1, self._neural_policy_calls),
            "mean_refined_top_k": self._neural_refined_candidates / max(1, self._neural_score_batches),
            "stagnation_wake_count": self._neural_stagnation_wake_count,
            "rank_changed_decisions": self._neural_rank_changed_decisions,
            "mv_changed_decisions": self._neural_mv_changed_decisions,
            "gap_direction_steps": self._neural_gap_direction_steps,
            "neural_gate_eligible_steps": self._neural_gate_eligible_steps,
            "neural_gate_fired_steps": self._neural_gate_fired_steps,
            "neural_gate_fire_rate": self._neural_gate_fired_steps / max(1, self._neural_gate_eligible_steps),
            "scalar_proposal_suppressed_by_learned_move": self._scalar_proposal_suppressed_by_learned_move,
            "scalar_candidate_decision_observations": self._scalar_candidate_decision_observations,
            "scalar_candidate_changed_decisions": self._scalar_candidate_changed_decisions,
            "scalar_candidate_changed_decision_rate": (
                self._scalar_candidate_changed_decisions / scalar_candidate_observations
            ),
            "scalar_candidate_target_margin_vs_analytic": (
                self._scalar_candidate_target_margin_vs_analytic_sum / scalar_candidate_observations
            ),
            "scalar_candidate_target_margin_vs_pool_mean": (
                self._scalar_candidate_target_margin_vs_pool_mean_sum / scalar_candidate_observations
            ),
            "scalar_candidate_target_regret_to_pool_oracle": (
                self._scalar_candidate_target_regret_to_pool_oracle_sum / scalar_candidate_observations
            ),
            "scalar_candidate_analytic_score_regret": (
                self._scalar_candidate_analytic_score_regret_sum / scalar_candidate_observations
            ),
            "scalar_parent_decision_observations": self._scalar_parent_decision_observations,
            "scalar_parent_changed_decisions": self._scalar_parent_changed_decisions,
            "scalar_parent_changed_decision_rate": (
                self._scalar_parent_changed_decisions / scalar_parent_observations
            ),
            "scalar_parent_analytic_score_regret": (
                self._scalar_parent_analytic_score_regret_sum / scalar_parent_observations
            ),
            "scalar_archive_parent_decision_observations": self._scalar_archive_parent_decision_observations,
            "scalar_archive_parent_changed_decisions": self._scalar_archive_parent_changed_decisions,
            "scalar_archive_parent_changed_decision_rate": (
                self._scalar_archive_parent_changed_decisions / scalar_archive_parent_observations
            ),
            "scalar_archive_parent_analytic_score_regret": (
                self._scalar_archive_parent_analytic_score_regret_sum
                / scalar_archive_parent_observations
            ),
            "scalar_replacement_preference_observations": self._scalar_replacement_preference_observations,
            "scalar_replacement_flip_to_accept": self._scalar_replacement_flip_to_accept,
            "scalar_replacement_flip_to_reject": self._scalar_replacement_flip_to_reject,
            "scalar_replacement_preference_flip_rate": (
                (self._scalar_replacement_flip_to_accept + self._scalar_replacement_flip_to_reject)
                / scalar_replacement_observations
            ),
            "compiled_polish_children": self._compiled_polish_children,
            "compiled_polish_child_fraction": self._compiled_polish_children / max(1, self._child_steps),
            "learned_move_calls": self._neural_learned_move_calls,
            "learned_move_children": self._neural_learned_move_children,
            "learned_move_updates": self._neural_learned_move_updates,
            "learned_move_angle_filtered": self._neural_learned_move_angle_filtered,
            "learned_move_candidate_count": self._neural_learned_move_candidate_count,
            "learned_move_mean_reward": self._neural_learned_move_reward_sum / learned_candidates,
            "learned_move_positive_reward_rate": self._neural_learned_move_positive_rewards / learned_candidates,
            "learned_move_mean_angle_penalty": self._neural_learned_move_angle_penalty_sum / learned_candidates,
            "learned_move_cone_pass_rate": self._neural_learned_move_cone_pass_count / learned_candidates,
            "learned_move_mean_mf_reward": self._neural_learned_move_mf_reward_sum / learned_candidates,
            "learned_move_good_action_mass": learned_good_mass,
            "baseline_good_action_mass": baseline_good_mass,
            "learned_move_good_action_mass_margin": learned_good_mass - baseline_good_mass,
            "learned_move_basin_crossing_mass": learned_crossing_mass,
            "baseline_basin_crossing_mass": baseline_crossing_mass,
            "learned_move_conductance_margin": learned_crossing_mass - baseline_crossing_mass,
            "learned_move_sampled_pool_policy_reward": policy_pool_reward,
            "learned_move_sampled_pool_uniform_reward": uniform_pool_reward,
            "learned_move_sampled_pool_reward_margin": policy_pool_reward - uniform_pool_reward,
            "learned_move_sampled_pool_reward_observations": self._neural_sampled_pool_reward_observations,
            "learned_move_sampled_pool_baseline_scope": (
                "uniform_reweighting_within_policy_sampled_pool_not_uniform_action_sampling"
            ),
            "learned_move_selection_observations": self._learned_move_selection_observations,
            "learned_move_selected_reward": move_selected_reward,
            "learned_move_selected_pool_uniform_reward": move_pool_uniform_reward,
            "learned_move_selected_pool_oracle_reward": move_pool_oracle_reward,
            "learned_move_selected_reward_margin_vs_pool_uniform": (
                move_selected_reward - move_pool_uniform_reward
            ),
            "learned_move_selected_reward_regret_to_pool_oracle": (
                move_pool_oracle_reward - move_selected_reward
            ),
            "learned_move_uniform_comparator_scope": (
                "uniform_reweighting_within_policy_sampled_pool_not_uniform_action_sampling"
            ),
            "learned_move_mass_observations": self._neural_learned_move_mass_observations,
            "learned_move_backend": getattr(self._learned_move_generator, "backend_name", "none")
            if self._learned_move_generator is not None
            else "none",
            "learned_move_prior_loaded": bool(self._learned_move_generator is not None and self.neural_learned_move_prior_path),
            "learned_move_runtime_target_head_weight": (
                self._learned_move_generator.target_head_weight if self._learned_move_generator is not None else 0.0
            ),
            "learned_move_runtime_flow_head_weight": (
                self._learned_move_generator.flow_head_weight if self._learned_move_generator is not None else 0.0
            ),
            "learned_move_runtime_mean_field_head_weight": (
                self._learned_move_generator.mean_field_head_weight if self._learned_move_generator is not None else 0.0
            ),
            "learned_move_runtime_conductance_head_weight": (
                self._learned_move_generator.conductance_head_weight if self._learned_move_generator is not None else 0.0
            ),
            "neural_online_training": self.neural_online_training,
            "neural_prior_feature_contract": self._neural_prior_feature_contract,
            "neural_prior_sha256": self._neural_prior_sha256,
            "neural_endpoint_contract_required": self.require_endpoint_only_prior,
            "neural_endpoint_contract_satisfied": (
                self._neural_prior_feature_contract == "endpoint_state_v1"
                if self.require_endpoint_only_prior
                else True
            ),
            "move_target_only_contract_required": self.require_target_only_move_prior,
            "learned_move_prior_sha256": self._learned_move_prior_sha256,
            "move_target_only_contract_satisfied": (
                self._learned_move_generator is not None
                and abs(float(self._learned_move_generator.flow_head_weight)) <= 1e-12
                and abs(float(self._learned_move_generator.mean_field_head_weight)) <= 1e-12
                and abs(float(self._learned_move_generator.conductance_head_weight)) <= 1e-12
                if self.require_target_only_move_prior
                else True
            ),
            "neural_generated_children": self._neural_generated_children,
            "neural_scalar_generated_children": self._neural_scalar_generated_children,
            "neural_move_generated_children": self._neural_move_generated_children,
            "neural_accepted_children": self._neural_accepted_children,
            "neural_accepted_replacements": self._neural_accepted_replacements,
            "neural_scalar_accepted_children": self._neural_scalar_accepted_children,
            "neural_scalar_accepted_replacements": self._neural_scalar_accepted_replacements,
            "neural_move_accepted_children": self._neural_move_accepted_children,
            "neural_move_accepted_replacements": self._neural_move_accepted_replacements,
            "coverage_pairs": self._neural_coverage_pairs,
            "expert_pairs": self._neural_expert_pairs,
            "neural_scalar_forward_calls": self._neural_scalar_forward_calls,
            "neural_scalar_scored_states": self._neural_scalar_scored_states,
            "neural_scalar_cache_hits": self._neural_scalar_cache_hits,
            "neural_scalar_cache_misses": self._neural_scalar_cache_misses,
            "neural_scalar_inference_seconds": self._neural_scalar_inference_seconds,
            "local_two_opt_check_upper_bound": self._local_two_opt_check_upper_bound,
            "local_relocate_check_upper_bound": self._local_relocate_check_upper_bound,
            "local_swap_check_upper_bound": self._local_swap_check_upper_bound,
            "local_move_check_upper_bound": (
                self._local_two_opt_check_upper_bound
                + self._local_relocate_check_upper_bound
                + self._local_swap_check_upper_bound
            ),
        }
        spectral_after = self._last_neural_spectral_diagnostics.get("after", {})
        if spectral_after:
            stats["last_w1_spectral_norm"] = spectral_after.get("w1_spectral_norm", 0.0)
            stats["last_w2_euclidean_norm"] = spectral_after.get("w2_euclidean_norm", 0.0)
            stats["last_lipschitz_proxy"] = spectral_after.get("lipschitz_proxy", 0.0)
            stats["last_clip_active"] = spectral_after.get("clip_active", False)
        return stats

    @staticmethod
    def _file_sha256(path: str) -> str:
        if not path:
            return ""
        source = Path(path)
        if not source.is_file():
            return ""
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _derived_stream_seed(seed: int, stream: str) -> int:
        payload = f"{int(seed)}:{stream}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)

    @staticmethod
    def _population_sha256(population: Sequence[Tour]) -> str:
        payload = json.dumps(
            [list(tour) for tour in population],
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _rng_state_sha256(rng: random.Random) -> str:
        return hashlib.sha256(repr(rng.getstate()).encode("utf-8")).hexdigest()

    def _new_neural_backend(self, input_dim: int, hidden_units: int) -> Any:
        if self.neural_backend == "pcd":
            return PCDResidualScalarNet(input_dim, hidden_units, self._scalar_model_rng)
        if self.neural_backend in {"paretoflow", "auto"}:
            return ParetoFlowScalarNet(input_dim, hidden_units, self._scalar_model_rng)
        return TinyMLP(input_dim, hidden_units, self._scalar_model_rng)

    def _load_learned_move_prior(self, path: str) -> Optional[SparseMoveGenerator]:
        if not path:
            return None
        try:
            generator = SparseMoveGenerator.load(Path(path), self._move_policy_rng)
            generator.learning_rate = self.neural_learned_move_learning_rate
            if not self.neural_mean_field_features:
                generator.mean_field_head_weight = 0.0
            if self.neural_flow_residual_weight <= 0.0:
                generator.flow_head_weight = 0.0
            return generator
        except Exception:
            return None

    def _neural_backend_name(self) -> str:
        if self._neural_scalar is None:
            return "none"
        return str(getattr(self._neural_scalar, "backend_name", "tiny"))

    def _sample_raw_neural_candidate(
        self,
        parent: Tour,
        parent_obj: ObjectiveVector,
    ) -> Optional[Tuple[str, int, int, ObjectiveVector]]:
        if (
            self.proposal_relocate_passes > 0
            and self.neural_relocate_candidate_probability > 0.0
            and self.rng.random() < self.neural_relocate_candidate_probability
        ):
            relocate = self._sample_relocate_indices(len(parent))
            if relocate is not None:
                i, j = relocate
                candidate_obj = self._uncounted_relocate_objectives(parent, parent_obj, i, j)
                if candidate_obj is not None:
                    return "relocate", i, j, candidate_obj
        i, j = sample_two_opt_indices(len(parent), self.rng)
        candidate_obj = self._uncounted_symmetric_two_opt_objectives(parent, parent_obj, i, j)
        if candidate_obj is not None:
            return "two_opt", i, j, candidate_obj
        relocate = self._sample_relocate_indices(len(parent))
        if relocate is None:
            return None
        i, j = relocate
        candidate_obj = self._uncounted_relocate_objectives(parent, parent_obj, i, j)
        if candidate_obj is None:
            return None
        return "relocate", i, j, candidate_obj

    def _uncounted_symmetric_two_opt_objectives(
        self,
        tour: Tour,
        current_objectives: ObjectiveVector,
        i: int,
        j: int,
    ) -> Optional[ObjectiveVector]:
        matrices = self._distance_matrices()
        symmetric = self._symmetric_objectives()
        if matrices is None or symmetric is None or not all(symmetric):
            return None
        if i > j:
            i, j = j, i
        a = tour[i - 1]
        b = tour[i]
        c = tour[j]
        d = tour[(j + 1) % len(tour)]
        values = []
        for current, matrix in zip(current_objectives, matrices):
            values.append(current - matrix[a][b] - matrix[c][d] + matrix[a][c] + matrix[b][d])
        return tuple(values)

    def _uncounted_symmetric_two_opt_batch_objectives(
        self,
        tour: Tour,
        current_objectives: ObjectiveVector,
        pairs: Sequence[Tuple[int, int]],
    ) -> List[Optional[ObjectiveVector]]:
        if not pairs:
            return []
        matrices = self._distance_matrices()
        symmetric = self._symmetric_objectives()
        if matrices is None or symmetric is None or not all(symmetric):
            return [None for _ in pairs]
        try:
            import numpy as np

            from .numba_kernels import NUMBA_AVAILABLE, two_opt_objectives_batch_numba

            if NUMBA_AVAILABLE and len(current_objectives) == 2 and len(matrices) == 2:
                out = two_opt_objectives_batch_numba(
                    np.asarray(matrices, dtype=np.float64),
                    np.asarray(tour, dtype=np.int64),
                    float(current_objectives[0]),
                    float(current_objectives[1]),
                    np.asarray(pairs, dtype=np.int64),
                )
                return [(float(row[0]), float(row[1])) for row in out]
        except Exception:
            pass
        return [
            self._uncounted_symmetric_two_opt_objectives(tour, current_objectives, i, j)
            for i, j in pairs
        ]

    def _sample_relocate_indices(self, num_cities: int) -> Optional[Tuple[int, int]]:
        if num_cities < 4:
            return None
        for _ in range(16):
            i = self.rng.randrange(1, num_cities)
            j = self.rng.randrange(num_cities)
            if j != i and j != i - 1:
                return i, j
        choices = [(i, j) for i in range(1, num_cities) for j in range(num_cities) if j != i and j != i - 1]
        return self.rng.choice(choices) if choices else None

    def _relocate_at(self, tour: Tour, i: int, j: int) -> Tour:
        if i <= 0 or i >= len(tour) or j < 0 or j >= len(tour) or j in {i, i - 1}:
            raise ValueError("relocate indices must satisfy 1 <= i < n and j not in {i, i-1}.")
        current = list(tour)
        city = current.pop(i)
        insert_at = j + 1 if j < i else j
        current.insert(insert_at, city)
        return tuple(current)

    def _uncounted_relocate_objectives(
        self,
        tour: Tour,
        current_objectives: ObjectiveVector,
        i: int,
        j: int,
    ) -> Optional[ObjectiveVector]:
        matrices = self._distance_matrices()
        if matrices is None or i <= 0 or i >= len(tour) or j < 0 or j >= len(tour) or j in {i, i - 1}:
            return None
        a = tour[i - 1]
        v = tour[i]
        b = tour[(i + 1) % len(tour)]
        c = tour[j]
        d = tour[(j + 1) % len(tour)]
        values = []
        for current, matrix in zip(current_objectives, matrices):
            remove_delta = matrix[a][b] - matrix[a][v] - matrix[v][b]
            insert_delta = matrix[c][v] + matrix[v][d] - matrix[c][d]
            values.append(current + remove_delta + insert_delta)
        return tuple(values)

    def _update_bounds(self, objective: ObjectiveVector) -> bool:
        ideal = tuple(min(a, b) for a, b in zip(self.ideal, objective))
        nadir = tuple(max(a, b) for a, b in zip(self.nadir, objective))
        if ideal == self.ideal and nadir == self.nadir:
            return False
        before = self._typed_population_energy()
        self.ideal = ideal
        self.nadir = nadir
        self._refresh_scale_cache()
        self._context_refresh_count += 1
        self._normalization_refresh_count += 1
        self._record_context_transition(
            "normalization_refresh",
            before,
            self._typed_population_energy(),
        )
        return True

    def _refresh_scale_cache(self) -> None:
        if len(self.ideal) == 2:
            self._ideal0 = self.ideal[0]
            self._ideal1 = self.ideal[1]
            self._inv0 = 1.0 / max(1e-9, self.nadir[0] - self.ideal[0])
            self._inv1 = 1.0 / max(1e-9, self.nadir[1] - self.ideal[1])
            self._clear_neural_bias_cache()

    def _clear_neural_bias_cache(self) -> None:
        cache = getattr(self, "_neural_bias_cache", None)
        if cache is not None:
            cache.clear()

    @staticmethod
    def _compute_ideal_nadir(vectors: Sequence[ObjectiveVector]) -> Tuple[ObjectiveVector, ObjectiveVector]:
        dim = len(vectors[0])
        ideal = tuple(min(v[i] for v in vectors) for i in range(dim))
        nadir = tuple(max(v[i] for v in vectors) for i in range(dim))
        return ideal, nadir

    def _build_neighbors(self, neighbor_size: int) -> List[List[int]]:
        neighbors: List[List[int]] = []
        for i, wi in enumerate(self.weights):
            ranked = sorted(
                range(len(self.weights)),
                key=lambda j: sum((a - b) ** 2 for a, b in zip(wi, self.weights[j])),
            )
            neighbors.append(ranked[:neighbor_size])
        return neighbors

    def _initialize_population(self) -> None:
        if self._initialize_population_numba():
            return
        for idx in range(self.num_particles):
            if not can_evaluate(self.instance):
                break
            tour = self._initial_tour(idx)
            objective = self._evaluate(tour)
            self.population.append(tour)
            self.objectives.append(objective)
            self.archive.update([ArchiveEntry(tour, objective)])
            self._log_diagnostic(self._used_evaluations())
        if not self.population:
            raise RuntimeError("Initial evaluation budget is too small to create one IPS particle.")

    def _initialize_population_numba(self) -> bool:
        if self.initialization != "scalar_greedy":
            return False
        if self.instance.num_objectives != 2 or self.initial_swap_passes > 0:
            return False
        matrices = self._distance_matrices()
        symmetric = self._symmetric_objectives()
        if matrices is None or len(matrices) != 2 or symmetric is None or not all(symmetric):
            return False
        remaining = remaining_evaluations(self.instance)
        population_size = self.num_particles if remaining is None else min(self.num_particles, remaining)
        if population_size <= 0:
            return False
        weighted = self._weighted_matrices2()
        if weighted is None:
            return False
        try:
            import numpy as np

            from .numba_kernels import NUMBA_AVAILABLE, scalar_greedy_population_numba

            if not NUMBA_AVAILABLE:
                return False
            seed = self.rng.randrange(1, 2_147_483_647)
            start_elapsed = time.perf_counter() - self._start_time
            start_evaluations = self._used_evaluations()
            core_start = time.perf_counter()
            population, objectives = scalar_greedy_population_numba(
                np.asarray(matrices, dtype=np.float64),
                weighted[:population_size],
                np.asarray(self._weight0[:population_size], dtype=np.float64),
                np.asarray(self._weight1[:population_size], dtype=np.float64),
                float(self._edge_scale0),
                float(self._edge_scale1),
                int(population_size),
                int(self.greedy_start_pool),
                int(self.greedy_candidate_pool),
                int(seed),
                int(self.initial_2opt_passes),
                int(self.initial_relocate_passes),
            )
            self._record_local_search_check_upper_bound(
                self.instance.num_cities,
                self.initial_2opt_passes * population_size,
                self.initial_relocate_passes * population_size,
                self.initial_swap_passes * population_size,
            )
            core_elapsed = max(1e-12, time.perf_counter() - core_start)
            for idx in range(int(population_size)):
                tour = tuple(int(city) for city in population[idx].tolist())
                objective = (float(objectives[idx, 0]), float(objectives[idx, 1]))
                self.population.append(tour)
                self.objectives.append(objective)
                self.archive.update([ArchiveEntry(tour, objective)])
                elapsed = start_elapsed + core_elapsed * ((idx + 1) / max(1, int(population_size)))
                self._log_diagnostic(start_evaluations + idx + 1, elapsed)
            if not self.population:
                raise RuntimeError("Initial evaluation budget is too small to create one IPS particle.")
            self._charge_compiled_evaluations(int(population_size))
            return True
        except Exception as exc:
            self.population = []
            self.objectives = []
            self.archive = ParetoArchive(max_size=self.archive.max_size)
            self._record_accelerator_fallback("compiled_initialization", exc)
            return False

    def _initial_tour(self, idx: int) -> Tour:
        if self.initialization == "random" or self.instance.num_objectives != 2:
            return random_tour(self.instance.num_cities, self.rng)
        if self.initialization == "mixed_scalar_greedy" and idx % 5 == 0:
            return random_tour(self.instance.num_cities, self.rng)
        return self._scalar_greedy_tour(idx)

    def _scalar_greedy_tour(self, weight_idx: int) -> Tour:
        matrices = self._distance_matrices()
        if matrices is None or len(matrices) != 2:
            return random_tour(self.instance.num_cities, self.rng)
        n = self.instance.num_cities
        scale0 = self._edge_scale0
        scale1 = self._edge_scale1
        w0 = self._weight0[weight_idx]
        w1 = self._weight1[weight_idx]
        starts = [0]
        if self.greedy_start_pool > 1 and n > 1:
            pool_size = min(self.greedy_start_pool - 1, n - 1)
            starts.extend(self.rng.sample(range(1, n), pool_size))
        candidates = [self._scalar_greedy_from_start(start, matrices, w0, w1, scale0, scale1) for start in starts]
        result = min(candidates, key=lambda candidate: self._scalar_cycle_cost(candidate, weight_idx))
        return self._scalar_local_descent(
            result,
            weight_idx,
            self.initial_2opt_passes,
            self.initial_relocate_passes,
            self.initial_swap_passes,
        )

    def _scalar_greedy_from_start(
        self,
        start: int,
        matrices: Sequence[Sequence[Sequence[float]]],
        w0: float,
        w1: float,
        scale0: float,
        scale1: float,
    ) -> Tour:
        n = self.instance.num_cities
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start
        while unvisited:
            ranked = sorted(
                unvisited,
                key=lambda city: w0 * matrices[0][current][city] / scale0
                + w1 * matrices[1][current][city] / scale1,
            )
            pool = ranked[: min(self.greedy_candidate_pool, len(ranked))]
            city = pool[self.rng.randrange(len(pool))]
            tour.append(city)
            unvisited.remove(city)
            current = city
        return self._rotate_to_zero(tuple(tour))

    def _scalar_cycle_cost(self, tour: Tour, weight_idx: int) -> float:
        matrices = self._distance_matrices()
        if matrices is None or len(matrices) != 2:
            return 0.0
        w0 = self._weight0[weight_idx]
        w1 = self._weight1[weight_idx]
        total = 0.0
        for idx, city in enumerate(tour):
            nxt = tour[(idx + 1) % len(tour)]
            total += w0 * matrices[0][city][nxt] / self._edge_scale0 + w1 * matrices[1][city][nxt] / self._edge_scale1
        return total

    @staticmethod
    def _rotate_to_zero(tour: Tour) -> Tour:
        idx = tour.index(0)
        return tuple(tour[idx:] + tour[:idx])

    def _scalar_local_descent(
        self,
        tour: Tour,
        weight_idx: int,
        two_opt_passes: int,
        relocate_passes: int,
        swap_passes: int,
    ) -> Tour:
        self._record_local_search_check_upper_bound(
            len(tour),
            two_opt_passes,
            relocate_passes,
            swap_passes,
        )
        result = tour
        if two_opt_passes > 0:
            result = self._scalar_two_opt_descent(result, weight_idx, two_opt_passes)
        if relocate_passes > 0:
            result = self._scalar_relocate_descent(result, weight_idx, relocate_passes)
        if swap_passes > 0:
            result = self._scalar_swap_descent(result, weight_idx, swap_passes)
        return result

    def _record_local_search_check_upper_bound(
        self,
        num_cities: int,
        two_opt_passes: int,
        relocate_passes: int,
        swap_passes: int,
    ) -> None:
        """Record a deterministic upper bound on local-neighborhood delta checks.

        Accelerated kernels stop early when a pass finds no improving move, so
        exact checks would require changing their hot-loop ABI.  The upper bound
        remains comparable across algorithms and is deliberately labelled as
        such in runtime metadata.
        """

        n = max(0, int(num_cities))
        pair_checks = max(0, (n - 2) * (n - 1) // 2)
        relocate_checks = max(0, (n - 1) * (n - 2))
        self._local_two_opt_check_upper_bound += pair_checks * max(0, int(two_opt_passes))
        self._local_relocate_check_upper_bound += relocate_checks * max(0, int(relocate_passes))
        self._local_swap_check_upper_bound += pair_checks * max(0, int(swap_passes))

    def _scalar_two_opt_descent(self, tour: Tour, weight_idx: int, max_passes: int) -> Tour:
        matrices = self._distance_matrices()
        symmetric = self._symmetric_objectives()
        if matrices is None or len(matrices) != 2 or symmetric is None or not all(symmetric):
            return tour
        n = len(tour)
        if n < 4:
            return tour
        if n >= 64:
            accelerated = self._scalar_two_opt_descent_numpy(tour, weight_idx, max_passes)
            if accelerated is not None:
                return accelerated
        scale0 = self._edge_scale0
        scale1 = self._edge_scale1
        w0 = self._weight0[weight_idx]
        w1 = self._weight1[weight_idx]
        current = list(tour)
        for _ in range(max_passes):
            best_delta = -1e-12
            best_pair: Optional[Tuple[int, int]] = None
            for i in range(1, n - 1):
                a = current[i - 1]
                b = current[i]
                row0_a = matrices[0][a]
                row1_a = matrices[1][a]
                for j in range(i + 1, n):
                    c = current[j]
                    d = current[(j + 1) % n]
                    delta0 = row0_a[c] + matrices[0][b][d] - row0_a[b] - matrices[0][c][d]
                    delta1 = row1_a[c] + matrices[1][b][d] - row1_a[b] - matrices[1][c][d]
                    delta = w0 * delta0 / scale0 + w1 * delta1 / scale1
                    if delta < best_delta:
                        best_delta = delta
                        best_pair = (i, j)
            if best_pair is None:
                break
            i, j = best_pair
            current[i : j + 1] = reversed(current[i : j + 1])
        return tuple(current)

    def _scalar_two_opt_descent_numpy(self, tour: Tour, weight_idx: int, max_passes: int) -> Optional[Tour]:
        matrix = self._weighted_matrix2(weight_idx)
        if matrix is None:
            return None
        try:
            import numpy as np

            from .numba_kernels import NUMBA_AVAILABLE, scalar_two_opt_descent_numba

            if NUMBA_AVAILABLE:
                current = np.asarray(tour, dtype=np.int64)
                accelerated = scalar_two_opt_descent_numba(matrix, current, max_passes)
                return tuple(int(city) for city in accelerated.tolist())

            current = np.asarray(tour, dtype=np.int64)
            n = len(current)
            for _ in range(max_passes):
                a_nodes = current[: n - 2]
                b_nodes = current[1 : n - 1]
                c_nodes = current[2:n]
                d_nodes = np.empty_like(c_nodes)
                if len(c_nodes) > 1:
                    d_nodes[:-1] = current[3:n]
                d_nodes[-1] = current[0]
                deltas = (
                    matrix[a_nodes[:, None], c_nodes[None, :]]
                    + matrix[b_nodes[:, None], d_nodes[None, :]]
                    - matrix[a_nodes, b_nodes][:, None]
                    - matrix[c_nodes, d_nodes][None, :]
                )
                deltas[np.tri(deltas.shape[0], deltas.shape[1], k=-1, dtype=bool)] = float("inf")
                flat_idx = int(np.argmin(deltas))
                best_delta = float(deltas.flat[flat_idx])
                if best_delta >= -1e-12:
                    break
                row, col = divmod(flat_idx, deltas.shape[1])
                i, j = row + 1, col + 2
                current[i : j + 1] = current[i : j + 1][::-1].copy()
            return tuple(int(city) for city in current.tolist())
        except Exception:
            return None

    def _scalar_swap_descent(self, tour: Tour, weight_idx: int, max_passes: int) -> Tour:
        matrices = self._distance_matrices()
        symmetric = self._symmetric_objectives()
        if matrices is None or len(matrices) != 2 or symmetric is None or not all(symmetric):
            return tour
        n = len(tour)
        if n < 4:
            return tour
        current = list(tour)
        for _ in range(max_passes):
            best_delta = -1e-12
            best_pair: Optional[Tuple[int, int]] = None
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    delta = self._scalar_swap_delta(current, i, j, weight_idx)
                    if delta < best_delta:
                        best_delta = delta
                        best_pair = (i, j)
            if best_pair is None:
                break
            i, j = best_pair
            current[i], current[j] = current[j], current[i]
        return tuple(current)

    def _scalar_swap_delta(self, tour: Sequence[int], i: int, j: int, weight_idx: int) -> float:
        matrices = self._distance_matrices()
        if matrices is None:
            raise RuntimeError("distance matrices are unavailable for this accelerator")

        def edge(city_a: int, city_b: int) -> float:
            return (
                self._weight0[weight_idx] * matrices[0][city_a][city_b] / self._edge_scale0
                + self._weight1[weight_idx] * matrices[1][city_a][city_b] / self._edge_scale1
            )

        a = tour[i - 1]
        u = tour[i]
        v = tour[j]
        d = tour[(j + 1) % len(tour)]
        if j == i + 1:
            return edge(a, v) + edge(v, u) + edge(u, d) - edge(a, u) - edge(u, v) - edge(v, d)
        b = tour[i + 1]
        c = tour[j - 1]
        return (
            edge(a, v)
            + edge(v, b)
            + edge(c, u)
            + edge(u, d)
            - edge(a, u)
            - edge(u, b)
            - edge(c, v)
            - edge(v, d)
        )

    def _scalar_relocate_descent(self, tour: Tour, weight_idx: int, max_passes: int) -> Tour:
        matrices = self._distance_matrices()
        symmetric = self._symmetric_objectives()
        if matrices is None or len(matrices) != 2 or symmetric is None or not all(symmetric):
            return tour
        n = len(tour)
        if n < 4:
            return tour
        if n >= 64:
            accelerated = self._scalar_relocate_descent_numpy(tour, weight_idx, max_passes)
            if accelerated is not None:
                return accelerated
        scale0 = self._edge_scale0
        scale1 = self._edge_scale1
        w0 = self._weight0[weight_idx]
        w1 = self._weight1[weight_idx]
        current = list(tour)
        for _ in range(max_passes):
            best_delta = -1e-12
            best_move: Optional[Tuple[int, int]] = None
            for i in range(1, n):
                a = current[i - 1]
                v = current[i]
                b = current[(i + 1) % n]
                remove0 = matrices[0][a][b] - matrices[0][a][v] - matrices[0][v][b]
                remove1 = matrices[1][a][b] - matrices[1][a][v] - matrices[1][v][b]
                for j in range(n):
                    if j == i or j == i - 1:
                        continue
                    c = current[j]
                    d = current[(j + 1) % n]
                    insert0 = matrices[0][c][v] + matrices[0][v][d] - matrices[0][c][d]
                    insert1 = matrices[1][c][v] + matrices[1][v][d] - matrices[1][c][d]
                    delta = w0 * (remove0 + insert0) / scale0 + w1 * (remove1 + insert1) / scale1
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (i, j)
            if best_move is None:
                break
            i, j = best_move
            city = current.pop(i)
            insert_at = j + 1 if j < i else j
            current.insert(insert_at, city)
        return tuple(current)

    def _scalar_relocate_descent_numpy(self, tour: Tour, weight_idx: int, max_passes: int) -> Optional[Tour]:
        matrix = self._weighted_matrix2(weight_idx)
        if matrix is None:
            return None
        try:
            import numpy as np

            from .numba_kernels import NUMBA_AVAILABLE, scalar_relocate_descent_numba

            if NUMBA_AVAILABLE:
                current_array = np.asarray(tour, dtype=np.int64)
                accelerated = scalar_relocate_descent_numba(matrix, current_array, max_passes)
                return tuple(int(city) for city in accelerated.tolist())

            current = list(tour)
            n = len(current)
            for _ in range(max_passes):
                arr = np.asarray(current, dtype=np.int64)
                next_nodes = np.empty_like(arr)
                next_nodes[:-1] = arr[1:]
                next_nodes[-1] = arr[0]
                best_delta = -1e-12
                best_move: Optional[Tuple[int, int]] = None
                for i in range(1, n):
                    a = int(arr[i - 1])
                    v = int(arr[i])
                    b = int(arr[(i + 1) % n])
                    remove_delta = matrix[a, b] - matrix[a, v] - matrix[v, b]
                    deltas = remove_delta + matrix[arr, v] + matrix[v, next_nodes] - matrix[arr, next_nodes]
                    deltas[i] = float("inf")
                    deltas[i - 1] = float("inf")
                    j = int(np.argmin(deltas))
                    delta = float(deltas[j])
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (i, j)
                if best_move is None:
                    break
                i, j = best_move
                city = current.pop(i)
                insert_at = j + 1 if j < i else j
                current.insert(insert_at, city)
            return tuple(current)
        except Exception:
            return None

    def _weighted_matrix2(self, weight_idx: int) -> Optional[object]:
        cached = self._weighted_matrix_cache.get(weight_idx)
        if cached is not None:
            return cached
        matrices = self._distance_matrices()
        if matrices is None or len(matrices) != 2:
            return None
        try:
            import numpy as np

            matrix = (
                self._weight0[weight_idx] * np.asarray(matrices[0], dtype=float) / self._edge_scale0
                + self._weight1[weight_idx] * np.asarray(matrices[1], dtype=float) / self._edge_scale1
            )
            self._weighted_matrix_cache[weight_idx] = matrix
            return matrix
        except Exception:
            return None

    def _weighted_matrices2(self) -> Optional[object]:
        try:
            import numpy as np

            matrices = [self._weighted_matrix2(idx) for idx in range(len(self.weights))]
            if any(matrix is None for matrix in matrices):
                return None
            return np.stack(matrices).astype(np.float64, copy=False)
        except Exception:
            return None

    def _edge_scales2(self) -> Tuple[float, float]:
        matrices = self._distance_matrices()
        if matrices is None or len(matrices) != 2:
            return 1.0, 1.0
        return self._mean_positive(matrices[0]), self._mean_positive(matrices[1])

    @staticmethod
    def _mean_positive(matrix: Sequence[Sequence[float]]) -> float:
        total = 0.0
        count = 0
        for row in matrix:
            for value in row:
                if value > 0.0:
                    total += float(value)
                    count += 1
        return total / count if count else 1.0

    def _load_neural_prior(self, path: str) -> Optional[Any]:
        payload_path = Path(path)
        if not payload_path.exists():
            raise FileNotFoundError(f"Neural prior not found: {path}")
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        feature_contract = str(payload.get("feature_contract", ""))
        if self.require_endpoint_only_prior and feature_contract != "endpoint_state_v1":
            raise ValueError(
                "Theory-optimized scalar prior must declare feature_contract='endpoint_state_v1'."
            )
        self._neural_prior_feature_contract = feature_contract
        net_payload = payload.get("network", payload)
        backend = str(net_payload.get("backend", "")).lower()
        if backend == PCDResidualScalarNet.backend_name:
            net = PCDResidualScalarNet.from_dict(net_payload, self._scalar_model_rng)
        elif backend == ParetoFlowScalarNet.backend_name:
            net = ParetoFlowScalarNet.from_dict(net_payload, self._scalar_model_rng)
        else:
            net = TinyMLP.from_dict(net_payload, self._scalar_model_rng)
        if net.input_dim not in {4, 6, 18, 20, 24}:
            raise ValueError("IPS neural prior must have input_dim 4, 6, 18, 20, or 24.")
        self._neural_prior_loaded = True
        self._neural_training_samples = int(payload.get("training_samples", self.neural_proposal_min_samples))
        return net

    def _entries(self) -> Tuple[ArchiveEntry, ...]:
        return tuple(ArchiveEntry(tour, obj) for tour, obj in zip(self.population, self.objectives))


class TheoryAlignedIPSOptimizer(EfficientIPSOptimizer):
    """Deprecated compatibility name for the non-certified fast optimizer.

    New code should select :class:`EfficientIPSOptimizer` for the heuristic
    batch path or :class:`mo_nco.ips_certified.CertifiedSingleSiteIPSOptimizer`
    for the frozen-context single-site MH control.
    """
