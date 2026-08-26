from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark_suite import BenchmarkSuite
from .neural_prior import NeuralPriorTrainingConfig, train_neural_prior


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a frozen cross-instance neural scalar prior.")
    parser.add_argument("--suite", type=Path, default=Path("benchmarks/suite_public_motsp_35.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/neural_prior/prior.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--warmup-evaluations", type=int, default=256)
    parser.add_argument("--hidden-units", type=int, default=16)
    parser.add_argument("--training-epochs", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-examples-per-case", type=int, default=4096)
    args = parser.parse_args()

    suite = BenchmarkSuite.from_json(args.suite)
    payload = train_neural_prior(
        suite=suite,
        output_path=args.output,
        config=NeuralPriorTrainingConfig(
            seed=args.seed,
            train_fraction=args.train_fraction,
            population=args.population,
            warmup_evaluations=args.warmup_evaluations,
            hidden_units=args.hidden_units,
            training_epochs=args.training_epochs,
            learning_rate=args.learning_rate,
            max_examples_per_case=args.max_examples_per_case,
        ),
    )
    print(f"Wrote neural prior to {args.output}")
    print(f"Training cases: {len(payload['train_cases'])}; test cases: {len(payload['test_cases'])}")
    print(f"Training samples: {payload['training_samples']}")
    print(f"Train suite: {payload['train_suite_path']}")
    print(f"Test suite: {payload['test_suite_path']}")


if __name__ == "__main__":
    main()
