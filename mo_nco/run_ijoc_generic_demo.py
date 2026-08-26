from __future__ import annotations

import argparse
import json
from pathlib import Path

from .instance import MultiObjectiveTSPInstance
from .pareto_ijoc_generic_smc import GenericAnnealedParetoSMCOptimizer
from .pareto_ijoc_problem import (
    MultiObjectiveKnapsackInstance,
    MultiObjectiveTSPProblemAdapter,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the generic IJOC typed annealed SMC skeleton."
    )
    parser.add_argument("--problem", choices=("tsp", "knapsack"), required=True)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--evaluations", type=int, default=500)
    parser.add_argument("--particles-per-type", type=int, default=8)
    parser.add_argument("--adaptive-tail", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.problem == "tsp":
        instance = MultiObjectiveTSPInstance.random_biobjective(
            args.size,
            seed=args.seed,
        )
        problem = MultiObjectiveTSPProblemAdapter(instance)
    else:
        problem = MultiObjectiveKnapsackInstance.random_instance(
            args.size,
            num_objectives=2,
            seed=args.seed,
        )

    result = GenericAnnealedParetoSMCOptimizer(
        problem,
        reference_directions=((0.75, 0.25), (0.25, 0.75)),
        particles_per_reference=args.particles_per_type,
        evaluations=args.evaluations,
        adaptive_search_evaluations=args.adaptive_tail,
        seed=args.seed,
    ).run()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "metadata": result.metadata,
                "archive": [
                    {"solution": entry.tour, "objectives": entry.objectives}
                    for entry in result.archive.entries
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), **result.metadata}, sort_keys=True))


if __name__ == "__main__":
    main()
