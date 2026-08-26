from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import run_benchmark
from .instance import MultiObjectiveTSPInstance


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_instance(args: argparse.Namespace) -> MultiObjectiveTSPInstance | None:
    if args.tsplib_files:
        paths = [Path(item) for item in parse_csv_strings(args.tsplib_files)]
        return MultiObjectiveTSPInstance.from_tsplib_files(paths)
    if args.bitsp_file:
        return MultiObjectiveTSPInstance.from_bitsp_file(args.bitsp_file)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed MO-NCO benchmarks.")
    parser.add_argument("--algorithms", default="ips,ips-neural,nsga2,moead,random2opt")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--cities", type=int, default=30)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=2000, help="Approximate candidate-evaluation budget.")
    parser.add_argument("--instance-seed", type=int, default=123)
    parser.add_argument(
        "--tsplib-files",
        default="",
        help="Comma-separated TSPLIB objective files, e.g. obj1.tsp,obj2.tsp.",
    )
    parser.add_argument(
        "--bitsp-file",
        type=Path,
        default=None,
        help="Simple bi-objective CSV with rows id,x1,y1,x2,y2 or x1,y1,x2,y2.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--log-period", type=int, default=100)
    parser.add_argument("--archive-update-period", type=int, default=25)
    parser.add_argument(
        "--measure-python-memory",
        action="store_true",
        help=(
            "Measure Python allocator peak increments with tracemalloc; "
            "the measurement uses a separate exact-state replay, and native "
            "and accelerator memory are excluded."
        ),
    )
    args = parser.parse_args()
    instance = load_instance(args)

    records, summary = run_benchmark(
        algorithms=parse_csv_strings(args.algorithms),
        seeds=parse_csv_ints(args.seeds),
        cities=args.cities,
        population=args.population,
        iterations=args.iterations,
        instance_seed=args.instance_seed,
        output_dir=args.output_dir,
        log_period=args.log_period,
        archive_update_period=args.archive_update_period,
        instance=instance,
        measure_python_memory=args.measure_python_memory,
    )

    payload = {
        "runs": len(records),
        "output_dir": str(args.output_dir),
        "instance": instance.name if instance is not None else f"synthetic:{args.cities}:{args.instance_seed}",
        "algorithms": parse_csv_strings(args.algorithms),
        "seeds": parse_csv_ints(args.seeds),
        "population": args.population,
        "iterations": args.iterations,
        "measure_python_memory": args.measure_python_memory,
        "memory_measurement_contract": (
            "python_tracemalloc_separate_replay_peak_increment_v1"
            if args.measure_python_memory
            else "disabled"
        ),
        "tsplib_files": args.tsplib_files,
        "bitsp_file": str(args.bitsp_file) if args.bitsp_file else "",
        "summary": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "benchmark_config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
