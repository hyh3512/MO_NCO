from __future__ import annotations

import argparse
from pathlib import Path

from .ablation import run_ips_ablation
from .run_benchmark import load_instance, parse_csv_ints


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IPS theory-module ablations.")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--evaluations", type=int, default=3000)
    parser.add_argument("--cities", type=int, default=30)
    parser.add_argument("--instance-seed", type=int, default=123)
    parser.add_argument("--tsplib-files", default="")
    parser.add_argument("--bitsp-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablation"))
    parser.add_argument("--log-period", type=int, default=250)
    args = parser.parse_args()

    instance = load_instance(args)
    if instance is None:
        from .instance import MultiObjectiveTSPInstance

        instance = MultiObjectiveTSPInstance.random_biobjective(args.cities, seed=args.instance_seed)

    run_ips_ablation(
        instance=instance,
        seeds=parse_csv_ints(args.seeds),
        output_dir=args.output_dir,
        population=args.population,
        evaluations=args.evaluations,
        log_period=args.log_period,
    )
    print(f"Wrote ablation results to {args.output_dir}")


if __name__ == "__main__":
    main()
