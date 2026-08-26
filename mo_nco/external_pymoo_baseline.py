from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from .archive import ArchiveEntry, ParetoArchive


def evaluate_tour(tour: Sequence[int], matrices: Sequence[Sequence[Sequence[float]]]) -> tuple[float, ...]:
    values = []
    n = len(tour)
    for matrix in matrices:
        total = 0.0
        for idx, city in enumerate(tour):
            total += float(matrix[city][tour[(idx + 1) % n]])
        values.append(total)
    return tuple(values)


def run_pymoo(algorithm_name: str, input_path: Path, output_path: Path) -> None:
    try:
        import numpy as np
        from pymoo.algorithms.moo.moead import MOEAD
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.callback import Callback
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.operators.crossover.ox import OrderCrossover
        from pymoo.operators.mutation.inversion import InversionMutation
        from pymoo.operators.sampling.rnd import PermutationRandomSampling
        from pymoo.optimize import minimize
        from pymoo.util.ref_dirs import get_reference_directions
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("pymoo is required for this external baseline.") from exc

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    matrices = payload["distance_matrices"]
    num_cities = int(payload["num_cities"])
    num_objectives = int(payload["num_objectives"])
    population_size = int(payload["population_size"])
    evaluations = int(payload["evaluations"])
    if population_size <= 0 or evaluations < population_size:
        raise ValueError(
            "pymoo requires a positive population and at least one full "
            "population of evaluations."
        )
    if evaluations % population_size != 0:
        raise ValueError(
            "The exact evaluation contract requires the requested budget "
            "to be divisible by the pymoo population size."
        )
    if algorithm_name == "moead" and population_size < 2:
        raise ValueError(
            "The pymoo MOEA/D adapter requires population_size >= 2."
        )
    target_evaluations = evaluations
    checkpoint_period_raw = payload.get("anytime_checkpoint_period")
    checkpoint_period = (
        None
        if checkpoint_period_raw is None
        else int(checkpoint_period_raw)
    )
    if checkpoint_period is not None and (
        checkpoint_period <= 0
        or checkpoint_period > target_evaluations
    ):
        raise ValueError(
            "anytime_checkpoint_period must be positive and no larger "
            "than the evaluation budget."
        )
    seed = int(payload["seed"])
    online_archive = ParetoArchive()
    pending_entries: list[ArchiveEntry] = []
    written_steps: set[int] = set()

    class TailPermutationTSP(ElementwiseProblem):
        def __init__(self) -> None:
            super().__init__(n_var=num_cities - 1, n_obj=num_objectives, vtype=int)
            self.evaluations_seen = 0

        def _evaluate(self, x, out, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            tail = [int(value) + 1 for value in x]
            tour = (0, *tail)
            objectives = evaluate_tour(tour, matrices)
            pending_entries.append(ArchiveEntry(tour, objectives))
            self.evaluations_seen += 1
            if (
                checkpoint_period is not None
                and (
                    self.evaluations_seen % checkpoint_period == 0
                    or self.evaluations_seen == target_evaluations
                )
            ):
                flush_pending()
                write_snapshot(self.evaluations_seen)
            out["F"] = np.array(objectives, dtype=float)

    diagnostics_path = output_path.with_suffix(".diagnostics.csv")
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_handle = diagnostics_path.open(
        "w",
        newline="",
        encoding="utf-8",
    )
    diagnostic_fields = [
        "evaluations",
        "elapsed_seconds",
        "tour",
        *[f"objective_{idx}" for idx in range(num_objectives)],
    ]
    diagnostics_writer = csv.DictWriter(
        diagnostics_handle,
        fieldnames=diagnostic_fields,
    )
    diagnostics_writer.writeheader()

    def flush_pending() -> None:
        if pending_entries:
            online_archive.update(pending_entries)
            pending_entries.clear()

    def write_snapshot(evaluations_used: int) -> None:
        if evaluations_used in written_steps:
            return
        elapsed_seconds = max(
            time.perf_counter() - start_time,
            1e-12,
        )
        for entry in online_archive.entries:
            row = {
                "evaluations": evaluations_used,
                "elapsed_seconds": elapsed_seconds,
                "tour": " ".join(str(city) for city in entry.tour),
            }
            for index, value in enumerate(entry.objectives):
                row[f"objective_{index}"] = value
            diagnostics_writer.writerow(row)
        written_steps.add(evaluations_used)

    class AllEvaluatedArchiveCallback(Callback):
        def __init__(self) -> None:
            super().__init__()

        def notify(self, algorithm: object) -> None:
            flush_pending()
            evaluator = getattr(algorithm, "evaluator", None)
            evaluations_used = int(
                getattr(evaluator, "n_eval", 0) or 0
            )
            if (
                evaluations_used > 0
            ):
                write_snapshot(evaluations_used)

    sampling = PermutationRandomSampling()
    crossover = OrderCrossover()
    mutation = InversionMutation()
    if algorithm_name == "nsga2":
        algorithm = NSGA2(
            pop_size=population_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            # Duplicate elimination can trigger replacement sampling and
            # overshoot pymoo's nominal n_eval termination.  Disabling it is
            # required by the exact matched-budget contract.
            eliminate_duplicates=False,
        )
    elif algorithm_name == "moead":
        ref_dirs = get_reference_directions("das-dennis", num_objectives, n_partitions=max(1, population_size - 1))
        if len(ref_dirs) > population_size:
            ref_dirs = ref_dirs[:population_size]
        algorithm = MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=min(15, max(2, len(ref_dirs))),
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
        )
    else:
        raise ValueError(f"Unsupported pymoo algorithm: {algorithm_name}")

    callback = AllEvaluatedArchiveCallback()
    start_time = time.perf_counter()
    try:
        result = minimize(
            TailPermutationTSP(),
            algorithm,
            termination=("n_eval", target_evaluations),
            seed=seed,
            verbose=False,
            save_history=False,
            callback=callback,
        )
        flush_pending()
        if target_evaluations not in written_steps:
            write_snapshot(target_evaluations)
    finally:
        diagnostics_handle.close()
    n_eval = int(getattr(result.algorithm.evaluator, "n_eval", evaluations))
    if n_eval != evaluations:
        raise RuntimeError(
            "pymoo did not consume exactly the requested objective-"
            f"evaluation budget: requested={evaluations}, observed={n_eval}."
        )
    archive = online_archive

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["tour", *[f"objective_{idx}" for idx in range(num_objectives)], "evaluations"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in archive.entries:
            row = {"tour": " ".join(str(city) for city in entry.tour), "evaluations": n_eval}
            for idx, value in enumerate(entry.objectives):
                row[f"objective_{idx}"] = value
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> None:
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) != 3:
        raise SystemExit("Usage: python -m mo_nco.external_pymoo_baseline nsga2|moead input.json output.csv")
    algorithm, input_path, output_path = args
    run_pymoo(algorithm.lower(), Path(input_path), Path(output_path))


if __name__ == "__main__":
    main()
