from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from .benchmark_suite import BenchmarkCase, BenchmarkSuite
from .instance import MultiObjectiveTSPInstance
from .ips_efficient import TheoryAlignedIPSOptimizer
from .neural_potential import TinyMLP
from .paretoflow_net import ParetoFlowScalarNet
from .pcd_net import PCDResidualScalarNet


@dataclass(frozen=True)
class NeuralPriorTrainingConfig:
    seed: int = 0
    train_fraction: float = 0.7
    population: int = 32
    warmup_evaluations: int = 256
    log_period: int = 128
    archive_update_period: int = 64
    hidden_units: int = 16
    neural_backend: str = "tiny"
    training_epochs: int = 24
    learning_rate: float = 0.03
    max_examples_per_case: int = 4096


def split_suite_cases(
    suite: BenchmarkSuite,
    train_fraction: float,
) -> Tuple[Tuple[BenchmarkCase, ...], Tuple[BenchmarkCase, ...]]:
    fraction = min(0.95, max(0.05, train_fraction))
    cutoff = max(1, min(len(suite.cases) - 1, int(round(len(suite.cases) * fraction))))
    return tuple(suite.cases[:cutoff]), tuple(suite.cases[cutoff:])


def write_suite_split(
    suite: BenchmarkSuite,
    train_cases: Sequence[BenchmarkCase],
    test_cases: Sequence[BenchmarkCase],
    output_dir: Path,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"{suite.name}_train.json"
    test_path = output_dir / f"{suite.name}_test.json"
    _write_suite_json(train_path, f"{suite.name}_train", train_cases)
    _write_suite_json(test_path, f"{suite.name}_test", test_cases)
    return train_path, test_path


def train_neural_prior(
    suite: BenchmarkSuite,
    output_path: Path,
    config: NeuralPriorTrainingConfig,
) -> dict:
    rng = random.Random(config.seed)
    train_cases, test_cases = split_suite_cases(suite, config.train_fraction)
    inputs: List[Tuple[float, ...]] = []
    targets: List[float] = []
    case_example_counts = {}
    for case_idx, case in enumerate(train_cases):
        instance = case.load_instance()
        optimizer = TheoryAlignedIPSOptimizer(
            instance=instance or MultiObjectiveTSPInstance.random_biobjective(
                case.cities,
                seed=case.instance_seed,
            ),
            num_particles=int(case.population or config.population),
            evaluations=int(case.evaluations or config.warmup_evaluations),
            seed=config.seed + case_idx,
            log_period=config.log_period,
            neighbor_size=8,
            crossover_probability=0.0,
            archive_parent_probability=0.10,
            archive_parent_sample=4,
            archive_update_period=config.archive_update_period,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            archive_conditioning=True,
            archive_conditioning_weight=3.0,
            neural_scalar_weight=0.0,
            neural_proposal_probability=0.0,
            initialization="mixed_scalar_greedy",
            greedy_candidate_pool=3,
        )
        optimizer.run()
        case_inputs, case_targets = optimizer.neural_training_examples()
        if len(case_inputs) > config.max_examples_per_case:
            selected = rng.sample(range(len(case_inputs)), config.max_examples_per_case)
            case_inputs = [case_inputs[idx] for idx in selected]
            case_targets = [case_targets[idx] for idx in selected]
        inputs.extend(case_inputs)
        targets.extend(case_targets)
        case_example_counts[case.name] = len(case_inputs)

    input_dim = len(inputs[0]) if inputs else 6
    backend = config.neural_backend.lower().strip()
    if backend == "paretoflow":
        net = ParetoFlowScalarNet(input_dim, config.hidden_units, rng)
    elif backend == "pcd":
        net = PCDResidualScalarNet(input_dim, config.hidden_units, rng)
    else:
        net = TinyMLP(input_dim, config.hidden_units, rng)
    net.fit(inputs, targets, config.training_epochs, config.learning_rate)
    payload = {
        "kind": "mo_nco_neural_scalar_prior",
        "config": asdict(config),
        "train_cases": [case.name for case in train_cases],
        "test_cases": [case.name for case in test_cases],
        "case_example_counts": case_example_counts,
        "training_samples": len(inputs),
        "network": net.to_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    split_dir = output_path.parent / "splits"
    train_path, test_path = write_suite_split(suite, train_cases, test_cases, split_dir)
    payload["train_suite_path"] = str(train_path)
    payload["test_suite_path"] = str(test_path)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _write_suite_json(path: Path, name: str, cases: Sequence[BenchmarkCase]) -> None:
    payload = {"name": name, "cases": [_case_to_json(case) for case in cases]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _case_to_json(case: BenchmarkCase) -> dict:
    payload = {
        "name": case.name,
        "kind": case.kind,
        "cities": case.cities,
        "instance_seed": case.instance_seed,
    }
    if case.tsplib_files:
        payload["tsplib_files"] = list(case.tsplib_files)
    if case.bitsp_file:
        payload["bitsp_file"] = case.bitsp_file
    if case.population is not None:
        payload["population"] = case.population
    if case.evaluations is not None:
        payload["evaluations"] = case.evaluations
    return payload
