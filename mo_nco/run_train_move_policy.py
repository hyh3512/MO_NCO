from __future__ import annotations

import argparse
from pathlib import Path

from .mv_replay import ReplayMovePolicyTrainingConfig, train_move_policy_from_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a learned sparse move policy from MV/MMD-GFEF replay JSONL.")
    parser.add_argument("--replay", type=str, required=True, help="Comma-separated replay JSONL paths.")
    parser.add_argument("--output", type=Path, default=Path("outputs/mv_mmd_replay_prior/move_policy.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--training-epochs", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--max-action-examples", type=int, default=250000)
    parser.add_argument("--positive-reweight", type=float, default=1.25)
    parser.add_argument("--hard-negative-reweight", type=float, default=1.50)
    parser.add_argument("--weight-norm-bound", type=float, default=3.0)
    parser.add_argument("--target-head-weight", type=float, default=1.0)
    parser.add_argument("--flow-head-weight", type=float, default=0.35)
    parser.add_argument("--mean-field-head-weight", type=float, default=0.20)
    parser.add_argument("--conductance-head-weight", type=float, default=0.20)
    args = parser.parse_args()

    replay_paths = [Path(item.strip()) for item in args.replay.split(",") if item.strip()]
    payload = train_move_policy_from_replay(
        replay_paths=replay_paths,
        output_path=args.output,
        config=ReplayMovePolicyTrainingConfig(
            seed=args.seed,
            input_dim=args.input_dim,
            hidden_units=args.hidden_units,
            training_epochs=args.training_epochs,
            learning_rate=args.learning_rate,
            max_action_examples=args.max_action_examples,
            positive_reweight=args.positive_reweight,
            hard_negative_reweight=args.hard_negative_reweight,
            weight_norm_bound=args.weight_norm_bound,
            target_head_weight=args.target_head_weight,
            flow_head_weight=args.flow_head_weight,
            mean_field_head_weight=args.mean_field_head_weight,
            conductance_head_weight=args.conductance_head_weight,
        ),
    )
    print(f"Wrote move policy prior: {args.output}")
    print(f"Training samples: {payload['training_samples']}")
    print(f"Kind counts: {payload['kind_counts']}")
    print(f"Backend: {payload['move_generator']['backend']}")


if __name__ == "__main__":
    main()
