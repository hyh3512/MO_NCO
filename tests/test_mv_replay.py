from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mo_nco.benchmark_suite import BenchmarkCase, BenchmarkSuite
from mo_nco.mv_replay import (
    MVReplayGenerationConfig,
    ReplayMovePolicyTrainingConfig,
    ReplayPriorTrainingConfig,
    _endpoint_state_features,
    generate_mv_replay_dataset,
    train_move_policy_from_replay,
    train_prior_from_replay,
)


class MVReplayTests(unittest.TestCase):
    def test_generate_replay_and_train_tiny_prior(self) -> None:
        suite = BenchmarkSuite(
            name="tiny_replay_suite",
            cases=(
                BenchmarkCase(name="train_case", kind="synthetic", cities=10, instance_seed=101),
                BenchmarkCase(name="test_case", kind="synthetic", cities=10, instance_seed=102),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "replay"
            summary = generate_mv_replay_dataset(
                suite,
                output_dir,
                MVReplayGenerationConfig(
                    seed=7,
                    train_fraction=0.5,
                    population=6,
                    warmup_evaluations=24,
                    log_period=12,
                    archive_update_period=12,
                    max_state_examples_per_case=48,
                    action_pairs_per_case=8,
                    action_candidate_pool=4,
                ),
            )
            replay_path = Path(summary["replay_path"])
            self.assertTrue(replay_path.exists())
            records = [json.loads(line) for line in replay_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(record.get("kind") == "state" for record in records))
            self.assertTrue(any(record.get("kind") == "residual" for record in records))
            move_record = next(record for record in records if record.get("kind") == "move_action")
            self.assertIn("archive_hv_increment", move_record)
            self.assertIn("long_horizon_advantage", move_record)
            self.assertIn("target_advantage", move_record)
            self.assertIn("flow_advantage", move_record)
            self.assertIn("mean_field_advantage", move_record)
            self.assertIn("conductance_advantage", move_record)
            state = next(record for record in records if record.get("kind") == "state")
            self.assertEqual(len(state["features"]), 24)

            prior_path = Path(tmp) / "prior.json"
            payload = train_prior_from_replay(
                [replay_path],
                prior_path,
                ReplayPriorTrainingConfig(
                    seed=7,
                    backend="tiny",
                    hidden_units=8,
                    training_epochs=2,
                    learning_rate=0.01,
                    max_state_examples=64,
                    max_pairs_per_kind=64,
                ),
            )
            self.assertTrue(prior_path.exists())
            self.assertEqual(payload["network"]["input_dim"], 24)
            self.assertGreater(payload["training_samples"], 0)
            self.assertEqual(payload["feature_contract"], "endpoint_state_v1")
            self.assertEqual(payload["zeroed_action_feature_indices"], [2, 3, 13])

            state_only_path = Path(tmp) / "state_only_prior.json"
            state_only = train_prior_from_replay(
                [replay_path],
                state_only_path,
                ReplayPriorTrainingConfig(
                    seed=8,
                    backend="tiny",
                    hidden_units=8,
                    training_epochs=1,
                    learning_rate=0.01,
                    max_state_examples=32,
                    max_pairs_per_kind=32,
                    flow_residual_weight=0.0,
                    ranking_weight=0.0,
                    hypercone_weight=0.0,
                    coverage_weight=0.0,
                    expert_weight=0.0,
                ),
            )
            self.assertEqual(state_only["active_pair_kinds"], [])
            self.assertTrue(all(count == 0 for count in state_only["pair_counts"].values()))

            move_path = Path(tmp) / "move_policy.json"
            move_payload = train_move_policy_from_replay(
                [replay_path],
                move_path,
                ReplayMovePolicyTrainingConfig(
                    seed=7,
                    hidden_units=8,
                    training_epochs=2,
                    max_action_examples=64,
                ),
            )
            self.assertTrue(move_path.exists())
            self.assertEqual(move_payload["move_generator"]["input_dim"], 16)
            self.assertEqual(
                move_payload["move_generator"]["backend"],
                "shared_edge_set_joint_action_policy",
            )
            self.assertGreater(move_payload["training_samples"], 0)
            self.assertIn("signal_means", move_payload)

    def test_endpoint_state_projection_removes_source_action_coordinates(self) -> None:
        features = tuple(float(index + 1) for index in range(24))
        projected = _endpoint_state_features(features)
        self.assertEqual(projected[2], 0.0)
        self.assertEqual(projected[3], 0.0)
        self.assertEqual(projected[13], 0.0)
        self.assertEqual(projected[0], features[0])
        self.assertEqual(projected[12], features[12])


if __name__ == "__main__":
    unittest.main()

