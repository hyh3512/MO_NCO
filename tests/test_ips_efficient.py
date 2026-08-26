from __future__ import annotations

import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mo_nco.evaluation import CountingTSPInstance, evaluation_count
from mo_nco.ips_efficient import EfficientIPSOptimizer
from mo_nco.instance import MultiObjectiveTSPInstance
from mo_nco.learned_move_generator import SparseMoveGenerator
from mo_nco.neural_potential import TinyMLP


def _torch_available() -> bool:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


class EfficientIPSTests(unittest.TestCase):
    def test_frozen_prior_loading_does_not_perturb_base_search_rng(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(12, seed=60)
        with tempfile.TemporaryDirectory() as tmp:
            scalar_path = Path(tmp) / "endpoint_scalar.json"
            move_path = Path(tmp) / "target_move.json"
            scalar_path.write_text(
                json.dumps(
                    {
                        "feature_contract": "endpoint_state_v1",
                        "training_samples": 128,
                        "network": TinyMLP(6, 8, random.Random(600)).to_dict(),
                    }
                ),
                encoding="utf-8",
            )
            move_path.write_text(
                json.dumps(
                    {
                        "move_generator": SparseMoveGenerator(
                            input_dim=16,
                            hidden_units=8,
                            rng=random.Random(601),
                            flow_head_weight=0.0,
                            mean_field_head_weight=0.0,
                            conductance_head_weight=0.0,
                        ).to_dict()
                    }
                ),
                encoding="utf-8",
            )

            common = {
                "instance": instance,
                "num_particles": 8,
                "evaluations": 24,
                "seed": 60,
                "initialization": "random",
                "proposal": "two_opt",
                "extra_two_opt_probability": 0.0,
                "neural_online_training": False,
                "neural_active_fraction": 0.0,
                "jit_polish_fraction": 1.1,
                "isolate_prior_loading_rng": True,
            }
            control = EfficientIPSOptimizer(
                **common,
                enable_neural_scalar=False,
                neural_proposal_probability=0.0,
                neural_learned_move_probability=0.0,
            )
            scalar = EfficientIPSOptimizer(
                **common,
                neural_scalar_weight=0.0,
                neural_proposal_probability=1.0,
                neural_prior_path=str(scalar_path),
                require_endpoint_only_prior=True,
            )
            move = EfficientIPSOptimizer(
                **common,
                enable_neural_scalar=False,
                neural_proposal_probability=1.0,
                neural_learned_move_probability=1.0,
                neural_learned_move_prior_path=str(move_path),
                allow_move_without_scalar=True,
                require_target_only_move_prior=True,
            )
            full = EfficientIPSOptimizer(
                **common,
                neural_scalar_weight=0.0,
                neural_proposal_probability=1.0,
                neural_prior_path=str(scalar_path),
                require_endpoint_only_prior=True,
                neural_learned_move_probability=1.0,
                neural_learned_move_prior_path=str(move_path),
                require_target_only_move_prior=True,
            )

        for treatment in (scalar, move, full):
            self.assertEqual(treatment.population, control.population)
            self.assertEqual(treatment.objectives, control.objectives)
            self.assertEqual(treatment.rng.getstate(), control.rng.getstate())
            self.assertEqual(
                treatment.neural_inference_stats()["initial_population_sha256"],
                control.neural_inference_stats()["initial_population_sha256"],
            )
            self.assertTrue(treatment.neural_inference_stats()["prior_loading_rng_isolated"])

    def test_neighbor_scalar_scoring_batches_endpoint_prior_calls(self) -> None:
        class FakeScalar:
            input_dim = 6

            def __init__(self) -> None:
                self.batch_sizes = []

            def predict_batch(self, inputs):  # type: ignore[no-untyped-def]
                self.batch_sizes.append(len(inputs))
                return [0.0 for _ in inputs]

        instance = MultiObjectiveTSPInstance.random_biobjective(10, seed=61)
        opt = EfficientIPSOptimizer(instance, num_particles=8, evaluations=24, seed=61, initialization="random")
        fake = FakeScalar()
        opt._neural_scalar = fake
        opt.neural_scalar_weight = 0.1
        opt.neural_active_fraction = 1.0
        opt._neural_bias_cache.clear()
        opt._replace_neighbors(0, opt.population[0], opt.objectives[0], 0)
        stats = opt.neural_inference_stats()
        self.assertEqual(stats["neural_scalar_forward_calls"], 1)
        self.assertGreater(stats["neural_scalar_scored_states"], stats["neural_scalar_forward_calls"])
        self.assertTrue(any(size > 1 for size in fake.batch_sizes))

    def test_move_only_proposal_does_not_require_scalar_network(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(12, seed=62)
        opt = EfficientIPSOptimizer(
            instance,
            num_particles=8,
            evaluations=32,
            seed=62,
            initialization="random",
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            enable_neural_scalar=False,
            neural_proposal_probability=1.0,
            neural_learned_move_probability=1.0,
            neural_learned_move_samples=4,
            allow_move_without_scalar=True,
            jit_polish_fraction=1.1,
        )
        self.assertIsNone(opt._neural_scalar)
        opt.run()
        stats = opt.neural_inference_stats()
        self.assertGreater(stats["learned_move_calls"], 0)
        self.assertGreater(stats["neural_generated_children"], 0)

    def test_scalar_counterfactual_diagnostic_records_change_margin_and_regret(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(10, seed=620)
        opt = EfficientIPSOptimizer(
            instance,
            num_particles=6,
            evaluations=24,
            seed=620,
            initialization="random",
        )
        candidates = ((10.0, 30.0), (20.0, 20.0), (30.0, 10.0))
        target_by_objective = {
            candidates[0]: 0.40,
            candidates[1]: 0.10,
            candidates[2]: 0.25,
        }
        with patch.object(
            opt,
            "_neural_state_target",
            side_effect=lambda objective, _direction: target_by_objective[objective],
        ):
            opt._record_scalar_candidate_decision(
                candidates,
                direction_idx=0,
                analytic_scores=(0.0, 1.0, 2.0),
                treatment_scores=(2.0, 0.0, 1.0),
            )
        stats = opt.neural_inference_stats()
        self.assertEqual(stats["scalar_candidate_decision_observations"], 1)
        self.assertEqual(stats["scalar_candidate_changed_decisions"], 1)
        self.assertAlmostEqual(stats["scalar_candidate_target_margin_vs_analytic"], 0.30)
        self.assertAlmostEqual(stats["scalar_candidate_target_margin_vs_pool_mean"], 0.15)
        self.assertAlmostEqual(stats["scalar_candidate_target_regret_to_pool_oracle"], 0.0)
        self.assertAlmostEqual(stats["scalar_candidate_analytic_score_regret"], 1.0)

    def test_move_selection_diagnostic_is_scoped_to_policy_sampled_pool(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(10, seed=621)
        opt = EfficientIPSOptimizer(
            instance,
            num_particles=6,
            evaluations=24,
            seed=621,
            initialization="random",
        )
        viable = [{"reward": 0.10}, {"reward": 0.40}, {"reward": -0.10}]
        opt._record_learned_move_selection(viable, viable[0])
        stats = opt.neural_inference_stats()
        self.assertEqual(stats["learned_move_selection_observations"], 1)
        self.assertAlmostEqual(stats["learned_move_selected_reward"], 0.10)
        self.assertAlmostEqual(
            stats["learned_move_selected_reward_margin_vs_pool_uniform"],
            0.10 - (0.10 + 0.40 - 0.10) / 3.0,
        )
        self.assertAlmostEqual(stats["learned_move_selected_reward_regret_to_pool_oracle"], 0.30)
        self.assertEqual(
            stats["learned_move_uniform_comparator_scope"],
            "uniform_reweighting_within_policy_sampled_pool_not_uniform_action_sampling",
        )

    def test_scalar_parent_and_replacement_diagnostics_do_not_consume_rng_or_evaluations(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(10, seed=622)
        counted = CountingTSPInstance(base, max_evaluations=24)
        opt = EfficientIPSOptimizer(
            counted,  # type: ignore[arg-type]
            num_particles=6,
            evaluations=24,
            seed=622,
            initialization="random",
        )
        rng_before = opt.rng.getstate()
        evaluations_before = evaluation_count(counted)
        opt._record_scalar_argmin_decision((0.0, 1.0), (1.0, 0.0), scope="parent")
        opt._record_scalar_argmin_decision((0.0, 1.0), (1.0, 0.0), scope="archive_parent")
        opt._record_scalar_replacement_preference(1.0, -1.0)
        opt._record_scalar_replacement_preference(-1.0, 1.0)
        stats = opt.neural_inference_stats()
        self.assertEqual(stats["scalar_parent_changed_decisions"], 1)
        self.assertEqual(stats["scalar_archive_parent_changed_decisions"], 1)
        self.assertEqual(stats["scalar_replacement_flip_to_accept"], 1)
        self.assertEqual(stats["scalar_replacement_flip_to_reject"], 1)
        self.assertEqual(opt.rng.getstate(), rng_before)
        self.assertEqual(evaluation_count(counted), evaluations_before)

    def test_mechanism_diagnostics_do_not_change_search_result_or_rng(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(12, seed=623)
        common = {
            "instance": instance,
            "num_particles": 8,
            "evaluations": 48,
            "seed": 623,
            "log_period": 8,
            "archive_update_period": 4,
            "archive_conditioning_weight": 2.0,
            "neural_scalar_weight": 0.05,
            "neural_backend": "tiny",
            "neural_hidden_units": 8,
            "neural_training_epochs": 1,
            "neural_proposal_probability": 1.0,
            "neural_proposal_weight": 0.25,
            "neural_candidate_pool": 2,
            "neural_proposal_min_samples": 1,
            "neural_active_fraction": 1.0,
            "proposal": "two_opt",
            "extra_two_opt_probability": 0.0,
            "initialization": "random",
            "initial_2opt_passes": 0,
            "proposal_2opt_passes": 0,
            "jit_polish_fraction": 1.1,
            "isolate_prior_loading_rng": True,
        }
        enabled = EfficientIPSOptimizer(**common, enable_mechanism_diagnostics=True)
        disabled = EfficientIPSOptimizer(**common, enable_mechanism_diagnostics=False)
        enabled_result = enabled.run()
        disabled_result = disabled.run()
        self.assertEqual(enabled_result.particles, disabled_result.particles)
        self.assertEqual(enabled_result.objectives, disabled_result.objectives)
        self.assertEqual(enabled_result.archive.entries, disabled_result.archive.entries)
        self.assertEqual(enabled.rng.getstate(), disabled.rng.getstate())
        self.assertEqual(
            enabled.neural_inference_stats()["evaluations_used"],
            disabled.neural_inference_stats()["evaluations_used"],
        )

    def test_local_move_check_upper_bound_is_reported(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(10, seed=63)
        opt = EfficientIPSOptimizer(instance, num_particles=6, evaluations=24, seed=63, initialization="random")
        opt._scalar_local_descent(opt.population[0], 0, two_opt_passes=2, relocate_passes=3, swap_passes=1)
        stats = opt.neural_inference_stats()
        self.assertEqual(stats["local_two_opt_check_upper_bound"], 36 * 2)
        self.assertEqual(stats["local_relocate_check_upper_bound"], 72 * 3)
        self.assertEqual(stats["local_swap_check_upper_bound"], 36)
        self.assertEqual(stats["local_move_check_upper_bound"], 324)

    def test_sparse_move_generator_samples_valid_two_opt_without_delta_features(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(12, seed=11)
        generator = SparseMoveGenerator(rng=random.Random(11))
        tour = tuple(range(12))
        objective = base.evaluate(tour)
        sample = generator.sample_two_opt(
            tour,
            objective,
            base._distance_matrices,  # type: ignore[attr-defined]
            (0.8, 0.2),
            (0.2, 0.8),
            1.0,
            1.0,
            sparse_nodes=8,
            sparse_partners=8,
            context=(0.1, 0.2, 0.3, 0.4),
        )
        self.assertIsNotNone(sample)
        assert sample is not None
        i, j, first_features, second_features = sample
        self.assertGreater(i, 0)
        self.assertGreater(j, i + 1)
        self.assertLess(j, len(tour))
        self.assertEqual(len(first_features), generator.input_dim)
        self.assertEqual(len(second_features), generator.input_dim)
        before = tuple(generator.node_weights)
        generator.update(first_features, second_features, 0.5)
        self.assertNotEqual(before, tuple(generator.node_weights))

    def test_sparse_move_generator_cfg_drops_only_conditioning(self) -> None:
        generator = SparseMoveGenerator(input_dim=16, rng=random.Random(12))
        features = tuple(float(idx) / 10.0 for idx in range(16))
        unconditioned = generator._unconditioned_features(features)
        self.assertEqual(unconditioned[:8], features[:8])
        self.assertTrue(all(value == 0.0 for value in unconditioned[8:]))
        self.assertEqual(generator._guided_score(2.0, 1.0, 1.0), 2.0)
        self.assertEqual(generator._guided_score(2.0, 1.0, 2.5), 3.5)

    def test_learned_move_angle_penalty_prefers_target_direction(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(12, seed=13)
        opt = EfficientIPSOptimizer(
            instance,
            num_particles=8,
            evaluations=24,
            seed=13,
            log_period=8,
            neural_scalar_weight=0.05,
            neural_proposal_probability=1.0,
            proposal="two_opt",
        )
        direction_idx = 0
        target0, target1 = opt._reference_target_point(direction_idx)
        parent_z0 = min(0.95, max(0.25, target0 + 0.35))
        parent_z1 = min(0.95, max(0.25, target1 + 0.35))

        def raw(z0: float, z1: float):
            return (opt._ideal0 + z0 / opt._inv0, opt._ideal1 + z1 / opt._inv1)

        parent = raw(parent_z0, parent_z1)
        toward = raw((parent_z0 + target0) * 0.5, (parent_z1 + target1) * 0.5)
        away = raw(min(1.2, parent_z0 + 0.10), min(1.2, parent_z1 + 0.10))
        self.assertLess(
            opt._learned_move_angle_penalty(toward, direction_idx, parent),
            opt._learned_move_angle_penalty(away, direction_idx, parent),
        )

    def test_no_mf_learned_move_context_removes_mean_field_reward(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(10, seed=14)
        opt = EfficientIPSOptimizer(
            instance,
            num_particles=6,
            evaluations=24,
            seed=14,
            log_period=8,
            archive_update_period=4,
            neural_mean_field_features=False,
            proposal="two_opt",
            initialization="random",
            jit_polish_fraction=0.0,
        )
        context = opt._learned_move_context(opt.objectives[0], 0)
        self.assertEqual(context[1], 0.0)
        self.assertEqual(opt._learned_move_mean_field_reward(opt.objectives[0], 0), 0.0)

    def test_loaded_prior_obeys_runtime_no_mf_and_no_flow_ablation(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(10, seed=15)
        prior = SparseMoveGenerator(
            flow_head_weight=0.35,
            mean_field_head_weight=0.20,
            rng=random.Random(15),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "move_prior.json"
            path.write_text(json.dumps({"move_generator": prior.to_dict()}), encoding="utf-8")
            opt = EfficientIPSOptimizer(
                instance,
                num_particles=6,
                evaluations=24,
                seed=15,
                log_period=8,
                archive_update_period=4,
                neural_mean_field_features=False,
                neural_flow_residual_weight=0.0,
                neural_online_training=False,
                neural_scalar_weight=0.01,
                neural_proposal_probability=1.0,
                neural_candidate_pool=2,
                neural_learned_move_probability=1.0,
                neural_learned_move_prior_path=str(path),
                proposal="two_opt",
                initialization="random",
                jit_polish_fraction=0.0,
            )
        assert opt._learned_move_generator is not None
        self.assertEqual(opt._learned_move_generator.flow_head_weight, 0.0)
        self.assertEqual(opt._learned_move_generator.mean_field_head_weight, 0.0)
        components = opt._learned_move_action_components(opt.objectives[1], 0, opt.objectives[0])
        self.assertEqual(components[1], 0.0)
        stats = opt.neural_inference_stats()
        self.assertEqual(stats["learned_move_runtime_flow_head_weight"], 0.0)
        self.assertEqual(stats["learned_move_runtime_mean_field_head_weight"], 0.0)
        opt._neural_prior_loaded = True
        opt.run()
        self.assertEqual(opt.neural_inference_stats()["learned_move_updates"], 0)

    def test_required_endpoint_prior_rejects_legacy_payload(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(10, seed=16)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy_scalar.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "endpoint_state_v1"):
                EfficientIPSOptimizer(
                    instance,
                    num_particles=6,
                    evaluations=24,
                    seed=16,
                    neural_scalar_weight=0.1,
                    neural_prior_path=str(path),
                    require_endpoint_only_prior=True,
                )

    def test_required_target_only_move_prior_rejects_conductance_head(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(10, seed=19)
        prior = SparseMoveGenerator(
            flow_head_weight=0.0,
            mean_field_head_weight=0.0,
            conductance_head_weight=0.2,
            rng=random.Random(19),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "move_prior.json"
            path.write_text(json.dumps({"move_generator": prior.to_dict()}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "target-only"):
                EfficientIPSOptimizer(
                    instance,
                    num_particles=6,
                    evaluations=24,
                    seed=19,
                    neural_learned_move_probability=1.0,
                    neural_learned_move_prior_path=str(path),
                    require_target_only_move_prior=True,
                )

    def test_small_run(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(10, seed=17)
        opt = EfficientIPSOptimizer(instance, num_particles=8, evaluations=32, seed=17, log_period=8)
        result = opt.run()
        self.assertGreater(len(result.archive), 0)
        self.assertGreater(len(result.diagnostics), 0)

    def test_neural_child_acceptance_is_separate_from_neighbor_replacement_count(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(10, seed=18)
        opt = EfficientIPSOptimizer(
            instance,
            num_particles=8,
            evaluations=24,
            seed=18,
            log_period=8,
            proposal="two_opt",
        )
        opt._last_child_source = "neural"
        opt._neural_generated_children = 1
        opt._replace_neighbors(0, opt.population[0], opt.objectives[0], 0)
        stats = opt.neural_inference_stats()
        self.assertEqual(stats["neural_accepted_children"], 1)
        self.assertGreaterEqual(stats["neural_accepted_replacements"], 1)
        self.assertLessEqual(stats["neural_accepted_children"], stats["neural_generated_children"])

    def test_neural_proposal_and_greedy_initialization_respect_budget(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(14, seed=23)
        counted = CountingTSPInstance(base, max_evaluations=48)
        opt = EfficientIPSOptimizer(
            counted,  # type: ignore[arg-type]
            num_particles=8,
            evaluations=48,
            seed=23,
            log_period=8,
            archive_update_period=8,
            archive_conditioning_weight=2.0,
            neural_scalar_weight=0.05,
            neural_training_epochs=2,
            neural_proposal_probability=1.0,
            neural_proposal_weight=0.25,
            neural_candidate_pool=3,
            neural_proposal_min_samples=1,
            initialization="mixed_scalar_greedy",
        )
        result = opt.run()
        self.assertGreater(len(result.archive), 0)
        self.assertLessEqual(evaluation_count(counted), 48)
        for tour in result.particles:
            base.validate_tour(tour)

    def test_diagnostics_use_true_evaluation_counter(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(14, seed=29)
        counted = CountingTSPInstance(base, max_evaluations=40)
        opt = EfficientIPSOptimizer(
            counted,  # type: ignore[arg-type]
            num_particles=8,
            evaluations=40,
            seed=29,
            log_period=8,
            archive_update_period=8,
            initialization="scalar_greedy",
            initial_2opt_passes=2,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
        )
        result = opt.run()
        self.assertEqual(result.diagnostics[0].iteration, 1)
        self.assertEqual(result.diagnostics[-1].iteration, evaluation_count(counted))
        self.assertLessEqual(result.diagnostics[-1].iteration, 40)

    def test_relocate_descent_preserves_feasible_tours(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(16, seed=31)
        counted = CountingTSPInstance(base, max_evaluations=48)
        opt = EfficientIPSOptimizer(
            counted,  # type: ignore[arg-type]
            num_particles=8,
            evaluations=48,
            seed=31,
            log_period=8,
            archive_update_period=8,
            initialization="scalar_greedy",
            initial_2opt_passes=2,
            proposal_2opt_passes=1,
            initial_relocate_passes=2,
            proposal_relocate_passes=1,
            initial_swap_passes=1,
            proposal_swap_passes=1,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
        )
        result = opt.run()
        self.assertLessEqual(evaluation_count(counted), 48)
        for tour in result.particles:
            base.validate_tour(tour)

    def test_neural_training_examples_use_archive_conditioned_features(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(12, seed=37)
        opt = EfficientIPSOptimizer(
            instance,
            num_particles=8,
            evaluations=40,
            seed=37,
            log_period=8,
            archive_update_period=8,
            archive_conditioning_weight=2.0,
            initialization="scalar_greedy",
        )
        opt.run()
        inputs, targets = opt.neural_training_examples()
        self.assertGreater(len(inputs), 0)
        self.assertEqual(len(inputs), len(targets))
        self.assertEqual(len(inputs[0]), 6)
        self.assertGreaterEqual(inputs[0][-2], 0.0)
        self.assertGreaterEqual(inputs[0][-1], 0.0)

    def test_extreme_neural_targets_weight_edge_directions(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(12, seed=39)
        opt = EfficientIPSOptimizer(
            instance,
            num_particles=8,
            evaluations=40,
            seed=39,
            log_period=8,
            archive_update_period=8,
            archive_conditioning_weight=2.0,
            neural_directional_coverage_weight=0.2,
            neural_extreme_progress_weight=0.3,
            neural_gap_fill_weight=0.1,
            neural_hv_center_bias=0.7,
            neural_extreme_repeats=3,
            neural_scalar_weight=0.05,
            neural_training_epochs=1,
            initialization="scalar_greedy",
        )
        opt.run()
        edge_idx = 0
        middle_idx = len(opt.weights) // 2
        self.assertGreater(opt._neural_weight_repeat(edge_idx), opt._neural_weight_repeat(middle_idx))
        edge_objective = opt.archive.entries[0].objectives
        terms = opt._archive_bias_terms2(edge_objective)
        self.assertIsNotNone(terms)
        z0, z1, hv_gain, novelty = terms or (0.0, 0.0, 0.0, 0.0)
        drift = opt._directional_drift_from_norm(z0, z1, edge_idx)
        self.assertGreaterEqual(drift, 0.0)
        self.assertLessEqual(drift, 1.0)
        self.assertLessEqual(opt._archive_improvement_bias_from_terms2(z0, z1, hv_gain, novelty, edge_idx), 0.0)

    def test_mckean_vlasov_lite_features_and_training(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(14, seed=43)
        counted = CountingTSPInstance(base, max_evaluations=56)
        opt = EfficientIPSOptimizer(
            counted,  # type: ignore[arg-type]
            num_particles=8,
            evaluations=56,
            seed=43,
            log_period=8,
            archive_update_period=8,
            archive_conditioning_weight=2.0,
            neural_scalar_weight=0.05,
            neural_hidden_units=10,
            neural_training_epochs=2,
            neural_proposal_probability=1.0,
            neural_proposal_weight=0.25,
            neural_candidate_pool=3,
            neural_proposal_min_samples=1,
            neural_relocate_candidate_probability=0.5,
            neural_mean_field_features=True,
            neural_prefilter_pool=6,
            neural_refine_top_k=2,
            neural_flow_pair_samples=8,
            neural_flow_residual_weight=0.5,
            neural_ranking_weight=0.1,
            neural_hypercone_loss_weight=0.1,
            neural_weight_norm_bound=2.5,
            neural_action_sample_pool=3,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            initialization="scalar_greedy",
            initial_2opt_passes=1,
            proposal_2opt_passes=0,
            proposal_relocate_passes=1,
        )
        result = opt.run()
        inputs, targets = opt.neural_training_examples()
        self.assertGreater(len(inputs), 0)
        self.assertEqual(len(inputs), len(targets))
        self.assertEqual(len(inputs[0]), 20)
        self.assertEqual(len(opt._particle_direction_summary), len(opt.weights))
        stats = opt.neural_inference_stats()
        self.assertGreater(stats["policy_calls"], 0)
        self.assertGreater(stats["raw_candidates"], 0)
        self.assertLessEqual(stats["mean_refined_top_k"], 2.0)
        self.assertLessEqual(evaluation_count(counted), 56)
        for tour in result.particles:
            base.validate_tour(tour)

    def test_paretoflow_backend_runs_as_scalar_oracle(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(12, seed=44)
        counted = CountingTSPInstance(base, max_evaluations=64)
        opt = EfficientIPSOptimizer(
            counted,  # type: ignore[arg-type]
            num_particles=8,
            evaluations=64,
            seed=44,
            log_period=8,
            archive_update_period=4,
            archive_conditioning_weight=2.0,
            neural_scalar_weight=0.05,
            neural_backend="paretoflow",
            neural_hidden_units=8,
            neural_training_epochs=1,
            neural_proposal_probability=1.0,
            neural_proposal_weight=0.25,
            neural_candidate_pool=3,
            neural_proposal_min_samples=1,
            neural_mean_field_features=True,
            neural_prefilter_pool=6,
            neural_refine_top_k=2,
            neural_flow_pair_samples=2,
            neural_weight_norm_bound=1.5,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            initialization="random",
            initial_2opt_passes=0,
            proposal_2opt_passes=0,
            jit_polish_fraction=1.1,
        )
        self.assertEqual(opt._neural_scalar.input_dim, 24)  # type: ignore[union-attr]
        features = opt._neural_features(opt.objectives[0], 0)
        self.assertEqual(len(features), 24)
        result = opt.run()
        stats = opt.neural_inference_stats()
        self.assertEqual(stats["neural_backend"], "paretoflow_scalar")
        self.assertGreater(stats["policy_calls"], 0)
        self.assertLessEqual(evaluation_count(counted), 64)
        for tour in result.particles:
            base.validate_tour(tour)

    @unittest.skipUnless(_torch_available(), "PCD residual backend requires torch")
    def test_pcd_backend_runs_as_scalar_oracle(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(12, seed=45)
        counted = CountingTSPInstance(base, max_evaluations=64)
        opt = EfficientIPSOptimizer(
            counted,  # type: ignore[arg-type]
            num_particles=8,
            evaluations=64,
            seed=45,
            log_period=8,
            archive_update_period=4,
            archive_conditioning_weight=2.0,
            neural_scalar_weight=0.05,
            neural_backend="pcd",
            neural_hidden_units=16,
            neural_training_epochs=1,
            neural_proposal_probability=1.0,
            neural_proposal_weight=0.25,
            neural_candidate_pool=3,
            neural_proposal_min_samples=1,
            neural_mean_field_features=True,
            neural_prefilter_pool=6,
            neural_refine_top_k=2,
            neural_flow_pair_samples=2,
            neural_weight_norm_bound=2.0,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            initialization="random",
            initial_2opt_passes=0,
            proposal_2opt_passes=0,
            jit_polish_fraction=1.1,
        )
        result = opt.run()
        stats = opt.neural_inference_stats()
        self.assertEqual(stats["neural_backend"], "pcd_residual_scalar")
        self.assertGreater(stats["policy_calls"], 0)
        self.assertLessEqual(evaluation_count(counted), 64)
        for tour in result.particles:
            base.validate_tour(tour)

    @unittest.skipUnless(_torch_available(), "PCD residual backend requires torch")
    def test_pcd_backend_uses_24d_target_conditioned_learned_moves(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(12, seed=46)
        counted = CountingTSPInstance(base, max_evaluations=72)
        opt = EfficientIPSOptimizer(
            counted,  # type: ignore[arg-type]
            num_particles=8,
            evaluations=72,
            seed=46,
            log_period=8,
            archive_update_period=4,
            archive_conditioning_weight=2.0,
            neural_scalar_weight=0.05,
            neural_backend="pcd",
            neural_hidden_units=16,
            neural_training_epochs=1,
            neural_proposal_probability=1.0,
            neural_proposal_weight=0.25,
            neural_candidate_pool=2,
            neural_proposal_min_samples=1,
            neural_mean_field_features=True,
            neural_prefilter_pool=1,
            neural_refine_top_k=1,
            neural_exact_two_opt_prefilter=False,
            neural_flow_pair_samples=2,
            neural_weight_norm_bound=2.0,
            neural_learned_move_probability=1.0,
            neural_learned_move_sparse_nodes=6,
            neural_learned_move_sparse_partners=6,
            neural_learned_move_samples=2,
            neural_condition_guidance_scale=1.4,
            neural_front_reweighting_strength=0.3,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            initialization="random",
            initial_2opt_passes=0,
            proposal_2opt_passes=0,
            jit_polish_fraction=1.1,
        )
        self.assertEqual(opt._neural_scalar.input_dim, 24)  # type: ignore[union-attr]
        features = opt._neural_features(opt.objectives[0], 0)
        self.assertEqual(len(features), 24)
        result = opt.run()
        stats = opt.neural_inference_stats()
        self.assertEqual(stats["neural_backend"], "pcd_residual_scalar")
        self.assertGreater(stats["learned_move_calls"], 0)
        self.assertGreater(stats["learned_move_children"], 0)
        self.assertGreater(stats["learned_move_updates"], 0)
        self.assertGreater(stats["learned_move_mass_observations"], 0)
        self.assertIn("learned_move_good_action_mass_margin", stats)
        self.assertIn("learned_move_conductance_margin", stats)
        self.assertGreater(stats["scalar_proposal_suppressed_by_learned_move"], 0)
        self.assertGreater(stats["neural_move_generated_children"], 0)
        self.assertLessEqual(evaluation_count(counted), 72)
        for tour in result.particles:
            base.validate_tour(tour)

    def test_neural_spectral_log_is_written_when_enabled(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(12, seed=47)
        old_path = os.environ.get("MO_NCO_NEURAL_SPECTRAL_LOG")
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "spectral.jsonl")
            os.environ["MO_NCO_NEURAL_SPECTRAL_LOG"] = log_path
            try:
                opt = EfficientIPSOptimizer(
                    base,
                    num_particles=8,
                    evaluations=32,
                    seed=47,
                    log_period=8,
                    archive_update_period=8,
                    archive_conditioning_weight=2.0,
                    neural_scalar_weight=0.05,
                    neural_hidden_units=8,
                    neural_training_epochs=1,
                    neural_proposal_probability=1.0,
                    neural_proposal_weight=0.25,
                    neural_candidate_pool=3,
                    neural_proposal_min_samples=1,
                    neural_mean_field_features=True,
                    neural_flow_pair_samples=4,
                    neural_weight_norm_bound=1.25,
                    proposal="two_opt",
                    initialization="scalar_greedy",
                    initial_2opt_passes=1,
                    proposal_2opt_passes=0,
                )
                opt.run()
            finally:
                if old_path is None:
                    os.environ.pop("MO_NCO_NEURAL_SPECTRAL_LOG", None)
                else:
                    os.environ["MO_NCO_NEURAL_SPECTRAL_LOG"] = old_path

            self.assertTrue(os.path.exists(log_path))
            with open(log_path, "r", encoding="utf-8") as handle:
                payload = json.loads(handle.readline())
            self.assertIn("before", payload)
            self.assertIn("after", payload)
            self.assertIn("lipschitz_proxy", payload["after"])
            stats = opt.neural_inference_stats()
            self.assertIn("last_lipschitz_proxy", stats)

    def test_late_repair_window_reactivates_neural_and_pauses_compiled_polish(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(12, seed=49)
        counted = CountingTSPInstance(base, max_evaluations=40)
        opt = EfficientIPSOptimizer(
            counted,  # type: ignore[arg-type]
            num_particles=8,
            evaluations=40,
            seed=49,
            log_period=8,
            archive_update_period=8,
            neural_scalar_weight=0.05,
            neural_proposal_probability=1.0,
            neural_proposal_weight=0.25,
            neural_candidate_pool=3,
            neural_proposal_min_samples=1,
            neural_late_repair_fraction=0.25,
            jit_polish_fraction=0.1,
            initialization="scalar_greedy",
            proposal="two_opt",
        )
        counted.evaluations = 31
        self.assertTrue(opt._neural_late_repair_active())
        self.assertTrue(opt._neural_is_active())
        self.assertFalse(opt._should_run_compiled_scalar_polish(31))

    def test_rank_fusion_can_make_neural_topk_decisive(self) -> None:
        class FakeScalar:
            input_dim = 20

            def predict_batch(self, _features):
                return [-0.1, -0.9]

        instance = MultiObjectiveTSPInstance.random_biobjective(12, seed=51)
        opt = EfficientIPSOptimizer(
            instance,
            num_particles=6,
            evaluations=24,
            seed=51,
            log_period=8,
            neural_scalar_weight=0.05,
            neural_proposal_probability=1.0,
            neural_proposal_weight=0.25,
            neural_rank_fusion_weight=1.0,
            neural_proposal_min_samples=1,
            initialization="scalar_greedy",
            proposal="two_opt",
        )
        opt._neural_scalar = FakeScalar()  # type: ignore[assignment]
        opt._neural_training_samples = 10
        parent_obj = opt.objectives[0]
        candidates = [
            ("two_opt", 1, 4, opt.objectives[0]),
            ("two_opt", 2, 5, opt.objectives[1]),
        ]
        scores = opt._neural_batched_action_scores(parent_obj, candidates, 0, base_scores=[0.0, 10.0])
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(opt.neural_inference_stats()["rank_changed_decisions"], 0)

    def test_mean_field_guidance_can_override_candidate_rank(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(12, seed=53)
        opt = EfficientIPSOptimizer(
            instance,
            num_particles=6,
            evaluations=24,
            seed=53,
            log_period=8,
            neural_mean_field_features=True,
            neural_mean_field_guidance_weight=1.0,
            neural_scalar_weight=0.05,
            neural_proposal_probability=1.0,
            proposal="two_opt",
        )
        opt._mean_field_target_reward_from_norm = lambda z0, _z1, _idx: 1.0 if z0 < 0.9 else 0.0  # type: ignore[method-assign]
        candidates = [
            ("two_opt", 1, 4, (opt.ideal[0] + 1.0, opt.ideal[1] + 8.0)),
            ("two_opt", 2, 5, (opt.ideal[0], opt.ideal[1] + 7.0)),
        ]
        scores = opt._apply_mean_field_guidance_scores(candidates, 0, [0.0, 10.0])
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(opt.neural_inference_stats()["mv_changed_decisions"], 0)

    def test_gap_direction_scheduler_records_archive_guided_steps(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(14, seed=55)
        opt = EfficientIPSOptimizer(
            instance,
            num_particles=8,
            evaluations=32,
            seed=55,
            log_period=8,
            neural_scalar_weight=0.05,
            neural_gap_direction_probability=1.0,
            proposal="two_opt",
            initialization="scalar_greedy",
        )
        direction = opt._direction_index_for_step(0)
        self.assertGreaterEqual(direction, 0)
        self.assertLess(direction, len(opt.weights))
        self.assertGreater(opt.neural_inference_stats()["gap_direction_steps"], 0)

    def test_mean_field_features_are_bounded_relative_statistics(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(12, seed=57)
        opt = EfficientIPSOptimizer(
            instance,
            num_particles=8,
            evaluations=32,
            seed=57,
            log_period=8,
            neural_scalar_weight=0.05,
            neural_hidden_units=10,
            neural_mean_field_features=True,
            proposal="two_opt",
        )
        features = opt._neural_features((1e9, -1e9), 0, opt.objectives[0])
        self.assertEqual(len(features), 20)
        self.assertTrue(all(-2.0 <= value <= 2.0 for value in features))

    def test_pareto_conditioned_expert_pairs_are_audited(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(14, seed=59)
        opt = EfficientIPSOptimizer(
            instance,
            num_particles=8,
            evaluations=48,
            seed=59,
            log_period=8,
            archive_update_period=8,
            neural_scalar_weight=0.05,
            neural_hidden_units=10,
            neural_training_epochs=1,
            neural_proposal_probability=1.0,
            neural_proposal_weight=0.25,
            neural_proposal_min_samples=1,
            neural_mean_field_features=True,
            neural_flow_pair_samples=4,
            neural_expert_pair_samples=4,
            neural_coverage_pair_weight=0.2,
            neural_expert_pair_weight=0.2,
            proposal="two_opt",
            initialization="scalar_greedy",
            initial_2opt_passes=1,
        )
        opt.run()
        stats = opt.neural_inference_stats()
        self.assertGreater(stats["expert_pairs"], 0)

    def test_uncounted_relocate_objectives_match_full_evaluation(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(12, seed=41)
        opt = EfficientIPSOptimizer(instance, num_particles=6, evaluations=24, seed=41, log_period=8)
        tour = opt.population[0]
        objective = instance.evaluate(tour)
        i, j = 2, 7
        relocated = opt._relocate_at(tour, i, j)
        delta_objective = opt._uncounted_relocate_objectives(tour, objective, i, j)
        full_objective = instance.evaluate(relocated)
        self.assertIsNotNone(delta_objective)
        for fast, full in zip(delta_objective or (), full_objective):
            self.assertAlmostEqual(fast, full, places=9)
        instance.validate_tour(relocated)


if __name__ == "__main__":
    unittest.main()

