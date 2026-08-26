from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark_suite import BenchmarkSuite
from .mv_replay import MVReplayGenerationConfig, generate_mv_replay_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate independent MV/MMD-GFEF replay data for neural priors.")
    parser.add_argument("--suite", type=Path, default=Path("benchmarks/suite_public_motsp_35.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mv_mmd_replay"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1, help="Generate multiple replay shards with consecutive seeds.")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--warmup-evaluations", type=int, default=256)
    parser.add_argument("--log-period", type=int, default=128)
    parser.add_argument("--archive-update-period", type=int, default=64)
    parser.add_argument("--max-state-examples-per-case", type=int, default=4096)
    parser.add_argument("--action-pairs-per-case", type=int, default=2048)
    parser.add_argument("--action-candidate-pool", type=int, default=8)
    parser.add_argument("--long-horizon-candidates", type=int, default=2)
    parser.add_argument("--long-horizon-discount", type=float, default=0.60)
    parser.add_argument("--mmd-bandwidth", type=float, default=0.18)
    args = parser.parse_args()

    suite = BenchmarkSuite.from_json(args.suite)
    summaries = []
    for shard in range(max(1, args.shards)):
        shard_seed = args.seed + shard
        shard_dir = args.output_dir if args.shards <= 1 else args.output_dir / f"shard_seed{shard_seed}"
        summary = generate_mv_replay_dataset(
            suite=suite,
            output_dir=shard_dir,
            config=MVReplayGenerationConfig(
                seed=shard_seed,
                train_fraction=args.train_fraction,
                population=args.population,
                warmup_evaluations=args.warmup_evaluations,
                log_period=args.log_period,
                archive_update_period=args.archive_update_period,
                max_state_examples_per_case=args.max_state_examples_per_case,
                action_pairs_per_case=args.action_pairs_per_case,
                action_candidate_pool=args.action_candidate_pool,
                long_horizon_candidates=args.long_horizon_candidates,
                long_horizon_discount=args.long_horizon_discount,
                mmd_bandwidth=args.mmd_bandwidth,
            ),
        )
        summaries.append(summary)
        print(f"Wrote replay dataset: {summary['replay_path']}")
        print(f"Records: {summary['records']}")
        print(f"Train suite: {summary['train_suite_path']}")
        print(f"Test suite: {summary['test_suite_path']}")
    if len(summaries) > 1:
        replay_arg = ",".join(str(summary["replay_path"]) for summary in summaries)
        print(f"Replay shards: {replay_arg}")


if __name__ == "__main__":
    main()
