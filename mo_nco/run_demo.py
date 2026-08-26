from __future__ import annotations

import argparse
import json
from pathlib import Path

from .instance import MultiObjectiveTSPInstance
from .neural_potential import NeuralScalarPotential
from .potential import ScalarArchivePotential
from .sampler import IPSMetropolisOptimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a theory-guided IPS demo on bi-objective TSP.")
    parser.add_argument("--cities", type=int, default=30)
    parser.add_argument("--particles", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--initial-temperature", type=float, default=1.0)
    parser.add_argument("--final-temperature", type=float, default=0.05)
    parser.add_argument("--archive-update-period", type=int, default=25)
    parser.add_argument("--log-period", type=int, default=100)
    parser.add_argument("--archive-csv", type=Path, default=Path("outputs/archive.csv"))
    parser.add_argument("--potential", choices=["analytic", "neural"], default="analytic")
    args = parser.parse_args()

    instance = MultiObjectiveTSPInstance.random_biobjective(args.cities, seed=args.seed)
    potential = (
        NeuralScalarPotential(seed=args.seed)
        if args.potential == "neural"
        else ScalarArchivePotential()
    )
    optimizer = IPSMetropolisOptimizer(
        instance=instance,
        num_particles=args.particles,
        iterations=args.iterations,
        seed=args.seed,
        initial_temperature=args.initial_temperature,
        final_temperature=args.final_temperature,
        archive_update_period=args.archive_update_period,
        log_period=args.log_period,
        potential=potential,
    )
    result = optimizer.run()
    result.write_archive_csv(args.archive_csv)

    final = result.diagnostics[-1]
    summary = {
        "cities": args.cities,
        "particles": args.particles,
        "iterations": args.iterations,
        "potential": args.potential,
        "archive_size": len(result.archive),
        "acceptance_rate": final.acceptance_rate,
        "hypervolume_2d": final.hypervolume_2d,
        "empirical_energy": final.empirical_energy,
        "archive_csv": str(args.archive_csv),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
