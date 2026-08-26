from __future__ import annotations

import argparse
from pathlib import Path

from .mv_replay import ReplayPriorTrainingConfig, train_prior_from_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a neural scalar prior from MV/MMD-GFEF replay JSONL.")
    parser.add_argument("--replay", type=str, required=True, help="Comma-separated replay JSONL paths.")
    parser.add_argument("--output", type=Path, default=Path("outputs/mv_mmd_replay_prior/prior.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=["tiny", "paretoflow", "pcd"], default="pcd")
    parser.add_argument("--hidden-units", type=int, default=48)
    parser.add_argument("--training-epochs", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-state-examples", type=int, default=100000)
    parser.add_argument("--max-pairs-per-kind", type=int, default=100000)
    parser.add_argument("--flow-residual-weight", type=float, default=0.35)
    parser.add_argument("--ranking-weight", type=float, default=0.12)
    parser.add_argument("--hypercone-weight", type=float, default=0.10)
    parser.add_argument("--coverage-weight", type=float, default=0.10)
    parser.add_argument("--expert-weight", type=float, default=0.10)
    parser.add_argument("--weight-norm-bound", type=float, default=3.0)
    parser.add_argument(
        "--endpoint-only-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zero source/move feature coordinates for the scalar state potential (default: enabled).",
    )
    args = parser.parse_args()

    replay_paths = [Path(item.strip()) for item in args.replay.split(",") if item.strip()]
    payload = train_prior_from_replay(
        replay_paths=replay_paths,
        output_path=args.output,
        config=ReplayPriorTrainingConfig(
            seed=args.seed,
            backend=args.backend,
            hidden_units=args.hidden_units,
            training_epochs=args.training_epochs,
            learning_rate=args.learning_rate,
            max_state_examples=args.max_state_examples,
            max_pairs_per_kind=args.max_pairs_per_kind,
            flow_residual_weight=args.flow_residual_weight,
            ranking_weight=args.ranking_weight,
            hypercone_weight=args.hypercone_weight,
            coverage_weight=args.coverage_weight,
            expert_weight=args.expert_weight,
            weight_norm_bound=args.weight_norm_bound,
            endpoint_only_features=args.endpoint_only_features,
        ),
    )
    print(f"Wrote replay prior: {args.output}")
    print(f"Training samples: {payload['training_samples']}")
    print(f"Pair counts: {payload['pair_counts']}")
    print(f"Backend: {payload['network'].get('backend', args.backend)}")
    print(f"Feature contract: {payload['feature_contract']}")


if __name__ == "__main__":
    main()
