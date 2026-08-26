from __future__ import annotations

"""Development-only diagnostics for the already exposed V21e3r1 V4 matrix.

This module never authorizes selection, confirmation, formal execution, or a
paper-level performance claim.  It can (a) stream and analyze the immutable
108-row results package, and (b) define a fourteen-arm confounding-diagnostic menu
for the same already exposed development cases.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import shutil
import sqlite3
import statistics
import tempfile
from typing import Iterable, Mapping, Sequence
from zipfile import ZipFile

from .archive import ArchiveEntry, ParetoArchive
from .pareto_v21e3_baselines import (
    V21E3BaselineConfig,
    frozen_development_baseline_configs,
)
from .pareto_v21e3_hybrid import V21E3HybridConfig
from .pareto_v21e3_trace_verify import decode_v21e3_objectives_json


DIAGNOSTIC_SCOPE = "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE"
DIAGNOSTIC_ARMS = (
    "C0_STANDARD",
    "C0_RANDOM",
    "C0_NO_LS",
    "C0_RANDOM_NO_LS",
    "C0_SELF_REPLACE",
    "C0_POP_MATCH",
    "NSGAII_STANDARD",
    "NSGAII_SEEDED",
    "NSGAII_POP21",
    "NSGAII_SEEDED_POP21",
    "MOEAD_STANDARD",
    "MOEAD_SEEDED",
    "MOEAD_POP21",
    "MOEAD_SEEDED_POP21",
)


@dataclass(frozen=True)
class BudgetSlice:
    label: str
    start_fraction: float
    end_fraction: float


SLICES = (
    BudgetSlice("0_10", 0.0, 0.10),
    BudgetSlice("10_25", 0.10, 0.25),
    BudgetSlice("25_50", 0.25, 0.50),
    BudgetSlice("50_100", 0.50, 1.00),
)


def _normalized_point(
    objective: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> tuple[float, float]:
    if len(objective) != 2 or len(lower) != 2 or len(upper) != 2:
        raise ValueError("The current development diagnostic is biobjective.")
    result = []
    for value, lo, hi in zip(objective, lower, upper):
        value = float(value)
        lo = float(lo)
        hi = float(hi)
        if not math.isfinite(value) or not lo < hi or value < lo or value > hi:
            raise ValueError("Objective or analytic box is invalid.")
        result.append((value - lo) / (hi - lo))
    return (result[0], result[1])


def normalized_hv_2d(
    objectives: Sequence[Sequence[float]],
    *,
    lower: Sequence[float],
    upper: Sequence[float],
) -> float:
    if not objectives:
        return 0.0
    archive = ParetoArchive(max_size=None, tol=0.0)
    archive.update(
        tuple(
            ArchiveEntry((index,), _normalized_point(objective, lower, upper))
            for index, objective in enumerate(objectives)
        )
    )
    return archive.hypervolume_2d(reference=(1.0, 1.0))


def _load_json_text(raw: str) -> object:
    value = json.loads(raw)
    return value


def _run_context(connection: sqlite3.Connection) -> dict[str, object]:
    row = connection.execute("SELECT run_context_json FROM run_attempt WHERE run_id=1").fetchone()
    if row is None:
        raise RuntimeError("Trace omits run context.")
    payload = json.loads(str(row[0]))
    if not isinstance(payload, dict):
        raise RuntimeError("Run context is not an object.")
    return payload


def _exact_bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise RuntimeError(f"{field} must be an exact JSON boolean.")
    return value


def _exact_nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeError(f"{field} must be an exact nonnegative JSON integer.")
    return value


def analyze_trace_database(
    database_path: str | Path,
    *,
    row: Mapping[str, object],
    lower: Sequence[float],
    upper: Sequence[float],
) -> dict[str, object]:
    """Reconstruct exact per-evaluation quality and mechanism diagnostics."""

    path = Path(database_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        run_status = connection.execute(
            "SELECT status FROM run_attempt WHERE run_id=1"
        ).fetchone()
        if run_status is None or str(run_status[0]) != "SUCCESS":
            raise RuntimeError("Trace is not a terminal successful run.")
        context = _run_context(connection)
        algorithm_config = context.get("algorithm_config")
        if not isinstance(algorithm_config, dict):
            raise RuntimeError("Trace run context omits algorithm_config.")
        evaluation_rows = list(
            connection.execute(
                "SELECT evaluation_index,attempt_index,search_phase_id,stage_id,type_id,operator_id,objectives_json "
                "FROM evaluations ORDER BY evaluation_index"
            )
        )
        decision_by_eval = {
            int(index): json.loads(str(raw))
            for index, raw in connection.execute(
                "SELECT evaluation_index,decision_json FROM decisions ORDER BY evaluation_index"
            )
        }
        attempt_rows = list(
            connection.execute(
                "SELECT attempt_index,context_json,status,physical_call_started,charged_evaluation_index,cache_source_evaluation_index "
                "FROM attempts ORDER BY attempt_index"
            )
        )
        terminal_row = connection.execute(
            "SELECT receipt_json FROM terminal_receipts WHERE run_id=1"
        ).fetchone()
        if terminal_row is None:
            raise RuntimeError("Trace omits terminal receipt.")
        terminal = json.loads(str(terminal_row[0]))
    finally:
        connection.close()

    budget = int(row["charged_evaluation_budget"])
    if len(evaluation_rows) != budget:
        raise RuntimeError("Trace evaluation count disagrees with row budget.")
    archive = ParetoArchive(max_size=None, tol=0.0)
    hv_before: list[float] = []
    hv_after: list[float] = []
    operator = defaultdict(lambda: defaultdict(float))
    phase = defaultdict(lambda: defaultdict(float))
    type_stats = defaultdict(lambda: defaultdict(float))
    archive_change_evaluations: list[int] = []

    current_hv = 0.0
    for expected, raw in enumerate(evaluation_rows, start=1):
        evaluation_index = int(raw[0])
        if evaluation_index != expected:
            raise RuntimeError("Evaluation indices are not contiguous.")
        hv_before.append(current_hv)
        attempt_index = int(raw[1])
        search_phase = str(raw[2])
        stage_id = str(raw[3])
        type_id = None if raw[4] is None else int(raw[4])
        operator_id = str(raw[5])
        objective = decode_v21e3_objectives_json(
            str(raw[6]), expected_dimension=len(lower)
        )
        normalized = _normalized_point(objective, lower, upper)
        archive.update((ArchiveEntry((evaluation_index,), normalized),))
        current_hv = archive.hypervolume_2d(reference=(1.0, 1.0))
        hv_after.append(current_hv)
        decision = decision_by_eval.get(evaluation_index)
        if not isinstance(decision, dict):
            raise RuntimeError("Trace omits a decision for an evaluation.")
        op = operator[operator_id]
        ph = phase[search_phase]
        ts = type_stats[str(type_id)]
        for bucket in (op, ph, ts):
            bucket["charged_evaluations"] += 1
            archive_changed = _exact_bool(
                decision.get("archive_changed"), "archive_changed"
            )
            retained = _exact_bool(
                decision.get("retained_after_update"), "retained_after_update"
            )
            accepted = _exact_bool(
                decision.get("accepted_into_population"),
                "accepted_into_population",
            )
            replacement_count = _exact_nonnegative_int(
                decision.get("population_replacement_count"),
                "population_replacement_count",
            )
            bucket["archive_changed"] += float(archive_changed)
            bucket["retained_after_update"] += float(retained)
            bucket["accepted_into_population"] += float(accepted)
            bucket["replacement_count"] += float(replacement_count)
            if _exact_bool(decision.get("new_evaluated_cell"), "new_evaluated_cell"):
                bucket["new_evaluated_cell"] += 1
            if _exact_bool(
                decision.get("new_nondominated_cell"), "new_nondominated_cell"
            ):
                bucket["new_nondominated_cell"] += 1
            advantage = decision.get("scalar_advantage")
            if isinstance(advantage, (int, float)) and not isinstance(advantage, bool):
                bucket["scalar_advantage_sum"] += float(advantage)
                bucket["scalar_advantage_count"] += 1
                if float(advantage) > 0:
                    bucket["positive_scalar_advantage"] += 1
        if archive_changed:
            archive_change_evaluations.append(evaluation_index)

    attempts_by_operator = defaultdict(lambda: defaultdict(float))
    for raw in attempt_rows:
        context_payload = json.loads(str(raw[1]))
        operator_id = str(context_payload.get("operator_id", "UNKNOWN"))
        witness = context_payload.get("operator_witness")
        if not isinstance(witness, dict):
            witness = {}
        bucket = attempts_by_operator[operator_id]
        bucket["attempts"] += 1
        bucket["physical_starts"] += float(bool(raw[3]))
        bucket["charged_evaluations"] += float(raw[4] is not None)
        bucket["cache_hits"] += float(raw[5] is not None)
        bucket["retries"] += float(int(witness.get("retry_ordinal", 0)) > 0 and not bool(witness.get("fallback_used")))
        bucket["fallbacks"] += float(bool(witness.get("fallback_used")))

    for operator_id, attempt_bucket in attempts_by_operator.items():
        for key, value in attempt_bucket.items():
            if key == "charged_evaluations":
                evaluation_charges = operator[operator_id][
                    "charged_evaluations"
                ]
                if evaluation_charges != value:
                    raise RuntimeError(
                        "Operator evaluation and attempt charge accounting disagree."
                    )
                continue
            operator[operator_id][key] += value
    if sum(
        values["charged_evaluations"] for values in operator.values()
    ) != len(evaluation_rows):
        raise RuntimeError("Operator charges do not sum to the evaluation ledger.")

    def finalize_buckets(raw_buckets: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
        output: dict[str, dict[str, float]] = {}
        for key, values in sorted(raw_buckets.items()):
            item = {name: float(value) for name, value in values.items()}
            attempts = item.get("attempts", item.get("charged_evaluations", 0.0))
            charges = item.get("charged_evaluations", 0.0)
            item["cache_hit_rate_per_attempt"] = item.get("cache_hits", 0.0) / attempts if attempts else 0.0
            item["archive_change_rate_per_charge"] = item.get("archive_changed", 0.0) / charges if charges else 0.0
            item["accepted_rate_per_charge"] = item.get("accepted_into_population", 0.0) / charges if charges else 0.0
            output[key] = item
        return output

    slices = {}
    for item in SLICES:
        start = int(round(item.start_fraction * budget))
        end = int(round(item.end_fraction * budget))
        if not 0 <= start < end <= budget:
            raise RuntimeError("Invalid development budget slice.")
        values = hv_before[start:end]
        slices[item.label] = {
            "start_evaluation_exclusive": start,
            "end_evaluation_inclusive": end,
            "mean_left_continuous_hv": sum(values) / len(values),
            "hv_at_start": 0.0 if start == 0 else hv_after[start - 1],
            "hv_at_end": hv_after[end - 1],
            "positive_hv_increment": hv_after[end - 1] - (0.0 if start == 0 else hv_after[start - 1]),
            "archive_change_count": sum(start < index <= end for index in archive_change_evaluations),
        }

    population_size = int(
        algorithm_config.get(
            "population_size",
            len(algorithm_config.get("reference_directions", [])),
        )
    )
    init_end = min(population_size, budget)
    exact_auc = sum(hv_before) / budget
    row_auc = float(row["normalized_left_continuous_hv_auc"])
    return {
        "schema": "v21e3r1_existing_trace_diagnostic_v1",
        "status": "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "scientific_scope": DIAGNOSTIC_SCOPE,
        "case_id": row["case_id"],
        "family": row["family"],
        "size": int(row["size"]),
        "seed": int(row["seed"]),
        "arm_id": row["arm_id"],
        "budget": budget,
        "population_size": population_size,
        "attempt_count": _exact_nonnegative_int(terminal.get("attempt_count"), "attempt_count"),
        "physical_start_count": _exact_nonnegative_int(terminal.get("physical_call_started_count"), "physical_call_started_count"),
        "charged_evaluation_count": _exact_nonnegative_int(terminal.get("charged_evaluation_count"), "charged_evaluation_count"),
        "cache_hit_count": _exact_nonnegative_int(terminal.get("cache_hit_count"), "cache_hit_count"),
        "attempts_per_charge": _exact_nonnegative_int(terminal.get("attempt_count"), "attempt_count") / budget,
        "cache_hit_rate_per_attempt": _exact_nonnegative_int(terminal.get("cache_hit_count"), "cache_hit_count") / _exact_nonnegative_int(terminal.get("attempt_count"), "attempt_count"),
        "initialization_end_evaluation": init_end,
        "initialization_terminal_hv": hv_after[init_end - 1],
        "exact_per_evaluation_left_continuous_hv_auc": exact_auc,
        "frozen_checkpoint_left_continuous_hv_auc": row_auc,
        "checkpoint_discretization_difference": exact_auc - row_auc,
        "terminal_hv_replayed": hv_after[-1],
        "terminal_hv_row": float(row["normalized_terminal_hv"]),
        "first_archive_change_evaluation": min(archive_change_evaluations) if archive_change_evaluations else None,
        "last_archive_change_evaluation": max(archive_change_evaluations) if archive_change_evaluations else None,
        "archive_change_count": len(archive_change_evaluations),
        "budget_slices": slices,
        "operators": finalize_buckets(operator),
        "phases": finalize_buckets(phase),
        "types": finalize_buckets(type_stats),
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def aggregate_existing_diagnostics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    groups = defaultdict(list)
    for row in rows:
        groups[(str(row["family"]), str(row["arm_id"]))].append(row)
    aggregate: dict[str, object] = {}
    for (family, arm), items in sorted(groups.items()):
        aggregate[f"{family}/{arm}"] = {
            "row_count": len(items),
            "mean_exact_auc": _mean(float(x["exact_per_evaluation_left_continuous_hv_auc"]) for x in items),
            "mean_checkpoint_auc": _mean(float(x["frozen_checkpoint_left_continuous_hv_auc"]) for x in items),
            "mean_checkpoint_discretization_difference": _mean(float(x["checkpoint_discretization_difference"]) for x in items),
            "mean_initialization_terminal_hv": _mean(float(x["initialization_terminal_hv"]) for x in items),
            "mean_terminal_hv": _mean(float(x["terminal_hv_replayed"]) for x in items),
            "mean_attempts_per_charge": _mean(float(x["attempts_per_charge"]) for x in items),
            "mean_cache_hit_rate_per_attempt": _mean(float(x["cache_hit_rate_per_attempt"]) for x in items),
            "mean_archive_change_rate": _mean(float(x["archive_change_count"]) / float(x["budget"]) for x in items),
            "slice_mean_hv": {
                label: _mean(float(x["budget_slices"][label]["mean_left_continuous_hv"]) for x in items)
                for label in (slice_.label for slice_ in SLICES)
            },
            "slice_mean_positive_increment": {
                label: _mean(float(x["budget_slices"][label]["positive_hv_increment"]) for x in items)
                for label in (slice_.label for slice_ in SLICES)
            },
        }

    row_index = {
        (
            str(row["family"]),
            str(row["case_id"]),
            int(row["seed"]),
            str(row["arm_id"]),
        ): row
        for row in rows
    }
    comparisons: dict[str, object] = {}
    for family in sorted({str(row["family"]) for row in rows}):
        c0 = aggregate.get(f"{family}/V21E3_C0")
        if not isinstance(c0, dict):
            continue
        for comparator in ("NSGAII", "MOEAD"):
            comp = aggregate.get(f"{family}/{comparator}")
            if not isinstance(comp, dict):
                continue
            paired_rows: list[dict[str, object]] = []
            for key, left_row in sorted(row_index.items()):
                row_family, case_id, seed, arm = key
                if row_family != family or arm != "V21E3_C0":
                    continue
                right_row = row_index.get((family, case_id, seed, comparator))
                if right_row is None:
                    raise RuntimeError(
                        f"Missing paired comparator row: {family}/{case_id}/{seed}/{comparator}"
                    )
                paired_rows.append(
                    {
                        "case_id": case_id,
                        "seed": seed,
                        "exact_auc_difference": (
                            float(left_row["exact_per_evaluation_left_continuous_hv_auc"])
                            - float(right_row["exact_per_evaluation_left_continuous_hv_auc"])
                        ),
                        "initialization_terminal_hv_difference": (
                            float(left_row["initialization_terminal_hv"])
                            - float(right_row["initialization_terminal_hv"])
                        ),
                        "terminal_hv_difference": (
                            float(left_row["terminal_hv_replayed"])
                            - float(right_row["terminal_hv_replayed"])
                        ),
                        "late_search_gain_difference": (
                            (
                                float(left_row["terminal_hv_replayed"])
                                - float(left_row["initialization_terminal_hv"])
                            )
                            - (
                                float(right_row["terminal_hv_replayed"])
                                - float(right_row["initialization_terminal_hv"])
                            )
                        ),
                    }
                )
            case_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
            for item in paired_rows:
                case_groups[str(item["case_id"])].append(item)
            case_means = {
                case_id: {
                    metric: _mean(float(item[metric]) for item in items)
                    for metric in (
                        "exact_auc_difference",
                        "initialization_terminal_hv_difference",
                        "terminal_hv_difference",
                        "late_search_gain_difference",
                    )
                }
                for case_id, items in sorted(case_groups.items())
            }
            comparisons[f"{family}/C0-minus-{comparator}"] = {
                "initialization_terminal_hv_difference": c0["mean_initialization_terminal_hv"] - comp["mean_initialization_terminal_hv"],
                "exact_auc_difference": c0["mean_exact_auc"] - comp["mean_exact_auc"],
                "terminal_hv_difference": c0["mean_terminal_hv"] - comp["mean_terminal_hv"],
                "slice_mean_hv_difference": {
                    label: c0["slice_mean_hv"][label] - comp["slice_mean_hv"][label]
                    for label in c0["slice_mean_hv"]
                },
                "slice_positive_increment_difference": {
                    label: c0["slice_mean_positive_increment"][label] - comp["slice_mean_positive_increment"][label]
                    for label in c0["slice_mean_positive_increment"]
                },
                "attempts_per_charge_difference": c0["mean_attempts_per_charge"] - comp["mean_attempts_per_charge"],
                "paired_case_seed_count": len(paired_rows),
                "case_cluster_count": len(case_means),
                "case_cluster_means": case_means,
                "case_cluster_signs": {
                    metric: {
                        "positive": sum(
                            1 for values in case_means.values() if values[metric] > 0
                        ),
                        "zero": sum(
                            1 for values in case_means.values() if values[metric] == 0
                        ),
                        "negative": sum(
                            1 for values in case_means.values() if values[metric] < 0
                        ),
                    }
                    for metric in (
                        "exact_auc_difference",
                        "initialization_terminal_hv_difference",
                        "terminal_hv_difference",
                        "late_search_gain_difference",
                    )
                },
            }
    return {
        "schema": "v21e3r1_existing_108_diagnostic_aggregate_v1",
        "status": "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "scientific_scope": DIAGNOSTIC_SCOPE,
        "row_count": len(rows),
        "groups": aggregate,
        "comparisons": comparisons,
        "interpretation_rule": {
            "early_advantage": "A large initialization or 0-10% difference is evidence of construction/search-strength confounding, not a typed-mechanism theorem.",
            "late_advantage": "A positive later-slice increment suggests ongoing search-dynamic value but remains a development-only total-effect observation.",
            "theory_ceiling": "No objective upper bound or oracle approximation ratio is instantiated; these traces cannot identify a theoretical performance ceiling."
        },
    }


def analyze_results_zip(zip_path: str | Path, output_directory: str | Path) -> dict[str, object]:
    source = Path(zip_path)
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with ZipFile(source) as archive:
        reference_name = "ijoc_v21e3r1_results_release/frozen_inputs/reference_manifest_development.json"
        reference = json.loads(archive.read(reference_name))
        bounds = {
            item["case_id"]: (item["objective_lower_bounds"], item["objective_upper_bounds"])
            for item in reference["cases"]
        }
        matrix_prefix = "ijoc_v21e3r1_results_release/matrix/rows/"
        row_names = sorted(
            name for name in archive.namelist()
            if name.startswith(matrix_prefix) and name.endswith("/row.json")
        )
        if len(row_names) != 108:
            raise RuntimeError("The immutable development package must contain 108 rows.")
        with tempfile.TemporaryDirectory(prefix="v21e3r1_diag_") as temp:
            temp_path = Path(temp) / "trace.sqlite3"
            for ordinal, row_name in enumerate(row_names, start=1):
                row = json.loads(archive.read(row_name))
                trace_name = row_name.rsplit("/", 1)[0] + "/trace.sqlite3"
                with archive.open(trace_name) as source_file, temp_path.open("wb") as target:
                    shutil.copyfileobj(source_file, target, length=8 * 1024 * 1024)
                lower, upper = bounds[row["case_id"]]
                diagnosis = analyze_trace_database(temp_path, row=row, lower=lower, upper=upper)
                rows.append(diagnosis)
                temp_path.unlink()
                if ordinal % 12 == 0:
                    print(f"analyzed {ordinal}/108", flush=True)
    aggregate = aggregate_existing_diagnostics(rows)
    row_path = out / "existing_108_row_diagnostics.json"
    aggregate_path = out / "existing_108_aggregate_diagnostics.json"
    row_path.write_text(json.dumps(rows, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return {
        "rows": str(row_path),
        "aggregate": str(aggregate_path),
        "row_count": len(rows),
        "status": aggregate["status"],
    }



def aggregate_diagnostic_matrix(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate the exposed-development confounding and factorial matrix."""

    groups = defaultdict(list)
    for row in rows:
        groups[(str(row["family"]), str(row["arm_id"]))].append(row)
    summaries: dict[str, object] = {}
    for (family, arm), items in sorted(groups.items()):
        summaries[f"{family}/{arm}"] = {
            "row_count": len(items),
            "mean_exact_auc": _mean(float(x["exact_per_evaluation_left_continuous_hv_auc"]) for x in items),
            "mean_initialization_terminal_hv": _mean(float(x["initialization_terminal_hv"]) for x in items),
            "mean_terminal_hv": _mean(float(x["terminal_hv_replayed"]) for x in items),
            "mean_attempts_per_charge": _mean(float(x["attempts_per_charge"]) for x in items),
            "slice_mean_hv": {
                label: _mean(float(x["budget_slices"][label]["mean_left_continuous_hv"]) for x in items)
                for label in (slice_.label for slice_ in SLICES)
            },
        }
    contrasts = {
        "construction": ("C0_STANDARD", "C0_RANDOM"),
        "bounded_local_search": ("C0_STANDARD", "C0_NO_LS"),
        "construction_without_local_search": ("C0_NO_LS", "C0_RANDOM_NO_LS"),
        "local_search_under_random_initialization": ("C0_RANDOM", "C0_RANDOM_NO_LS"),
        "neighborhood_replacement": ("C0_STANDARD", "C0_SELF_REPLACE"),
        "c0_population_expansion": ("C0_POP_MATCH", "C0_STANDARD"),
        "nsga2_family_aware_construction_policy": ("NSGAII_SEEDED", "NSGAII_STANDARD"),
        "moead_family_aware_construction_policy": ("MOEAD_SEEDED", "MOEAD_STANDARD"),
        "nsga2_population_21_random": ("NSGAII_STANDARD", "NSGAII_POP21"),
        "nsga2_population_21_seeded": ("NSGAII_SEEDED", "NSGAII_SEEDED_POP21"),
        "moead_population_21_random": ("MOEAD_STANDARD", "MOEAD_POP21"),
        "moead_population_21_seeded": ("MOEAD_SEEDED", "MOEAD_SEEDED_POP21"),
        "c0_vs_seeded_nsga2": ("C0_STANDARD", "NSGAII_SEEDED"),
        "c0_vs_seeded_moead": ("C0_STANDARD", "MOEAD_SEEDED"),
        "c0_vs_seeded_nsga2_pop21": ("C0_STANDARD", "NSGAII_SEEDED_POP21"),
        "c0_vs_seeded_moead_pop21": ("C0_STANDARD", "MOEAD_SEEDED_POP21"),
    }
    contrast_payload: dict[str, object] = {}
    for family in sorted({str(row["family"]) for row in rows}):
        for name, (left, right) in contrasts.items():
            l = summaries.get(f"{family}/{left}")
            r = summaries.get(f"{family}/{right}")
            if not isinstance(l, dict) or not isinstance(r, dict):
                continue
            contrast_payload[f"{family}/{name}"] = {
                "left_arm": left,
                "right_arm": right,
                "mean_exact_auc_difference": l["mean_exact_auc"] - r["mean_exact_auc"],
                "mean_initialization_terminal_hv_difference": l["mean_initialization_terminal_hv"] - r["mean_initialization_terminal_hv"],
                "mean_terminal_hv_difference": l["mean_terminal_hv"] - r["mean_terminal_hv"],
                "slice_mean_hv_difference": {
                    label: l["slice_mean_hv"][label] - r["slice_mean_hv"][label]
                    for label in l["slice_mean_hv"]
                },
            }
    factorial: dict[str, object] = {}
    for family in sorted({str(row["family"]) for row in rows}):
        cells = {
            key: summaries.get(f"{family}/{arm}")
            for key, arm in {
                "strong_ls": "C0_STANDARD",
                "random_ls": "C0_RANDOM",
                "strong_no_ls": "C0_NO_LS",
                "random_no_ls": "C0_RANDOM_NO_LS",
            }.items()
        }
        if not all(isinstance(value, dict) for value in cells.values()):
            continue
        q11 = float(cells["strong_ls"]["mean_exact_auc"])
        q01 = float(cells["random_ls"]["mean_exact_auc"])
        q10 = float(cells["strong_no_ls"]["mean_exact_auc"])
        q00 = float(cells["random_no_ls"]["mean_exact_auc"])
        factorial[family] = {
            "estimand": "development_only_two_by_two_descriptive_factorial_on_exact_auc",
            "construction_main_effect": 0.5 * ((q11 - q01) + (q10 - q00)),
            "local_search_main_effect": 0.5 * ((q11 - q10) + (q01 - q00)),
            "construction_by_local_search_interaction": q11 - q01 - q10 + q00,
            "cell_means": {
                "strong_initialization_with_local_search": q11,
                "random_initialization_with_local_search": q01,
                "strong_initialization_without_local_search": q10,
                "random_initialization_without_local_search": q00,
            },
            "claim_boundary": (
                "Descriptive exposed-development factorial only; no population "
                "causal estimand or later-phase authorization."
            ),
        }
    return {
        "schema": "v21e3r1_fourteen_arm_development_diagnostic_aggregate_v2",
        "status": "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "scientific_scope": DIAGNOSTIC_SCOPE,
        "row_count": len(rows),
        "summaries": summaries,
        "contrasts": contrast_payload,
        "factorial_initialization_local_search": factorial,
        "seeded_arm_limitation": (
            "SEEDED arms share the family-aware constructor policy but do not "
            "claim byte-identical realized seed pools across algorithms."
        ),
        "later_phase_authorization": "PROHIBITED",
    }


def _resize_open_biobjective_directions(
    reference_directions: Sequence[Sequence[float]],
    count: int,
) -> tuple[tuple[float, float], ...]:
    if count < 2:
        raise ValueError("A population diagnostic requires at least two slots.")
    points = sorted(float(row[0]) for row in reference_directions)
    if not points or any(len(row) != 2 for row in reference_directions):
        raise ValueError("Biobjective reference directions are required.")
    low = max(min(points), 1e-9)
    high = min(max(points), 1.0 - 1e-9)
    if not low < high:
        low = 1.0 / (count + 1)
        high = count / (count + 1)
    return tuple(
        (
            low + (high - low) * index / (count - 1),
            1.0 - (low + (high - low) * index / (count - 1)),
        )
        for index in range(count)
    )

def hybrid_diagnostic_config(
    *,
    arm_id: str,
    reference_directions: Sequence[Sequence[float]],
    charged_evaluations: int,
    checkpoint_period: int,
    seed: int,
    family: str | None = None,
    trace_database: str | None = None,
    terminal_receipt: str | None = None,
) -> V21E3HybridConfig:
    directions = tuple(tuple(float(x) for x in row) for row in reference_directions)
    if arm_id == "C0_STANDARD":
        return V21E3HybridConfig(
            candidate_id="C0", reference_directions=directions,
            charged_evaluations=charged_evaluations, checkpoint_period=checkpoint_period,
            seed=seed, phase="development", trace_database=trace_database,
            terminal_receipt=terminal_receipt, development_diagnostic_id="C0_STANDARD",
        )
    if arm_id == "C0_RANDOM":
        return V21E3HybridConfig(
            candidate_id="C0", reference_directions=directions,
            charged_evaluations=charged_evaluations, checkpoint_period=checkpoint_period,
            seed=seed, phase="development", trace_database=trace_database,
            terminal_receipt=terminal_receipt,
            initialization_policy="problem_native_exact_random_solution_development_diagnostic_v1",
            development_diagnostic_id="C0_RANDOM",
        )
    if arm_id == "C0_NO_LS":
        return V21E3HybridConfig(
            candidate_id="C0", reference_directions=directions,
            charged_evaluations=charged_evaluations, checkpoint_period=checkpoint_period,
            seed=seed, phase="development", trace_database=trace_database,
            terminal_receipt=terminal_receipt, local_improvement_steps=0,
            development_diagnostic_id="C0_NO_LS",
        )
    if arm_id == "C0_RANDOM_NO_LS":
        return V21E3HybridConfig(
            candidate_id="C0", reference_directions=directions,
            charged_evaluations=charged_evaluations, checkpoint_period=checkpoint_period,
            seed=seed, phase="development", trace_database=trace_database,
            terminal_receipt=terminal_receipt, local_improvement_steps=0,
            initialization_policy="problem_native_exact_random_solution_development_diagnostic_v1",
            development_diagnostic_id="C0_RANDOM_NO_LS",
        )
    if arm_id == "C0_SELF_REPLACE":
        return V21E3HybridConfig(
            candidate_id="C0", reference_directions=directions,
            charged_evaluations=charged_evaluations, checkpoint_period=checkpoint_period,
            seed=seed, phase="development", trace_database=trace_database,
            terminal_receipt=terminal_receipt,
            replacement_policy="self_type_nonworse_replacement_development_diagnostic_v1",
            development_diagnostic_id="C0_SELF_REPLACE",
        )
    if arm_id == "C0_POP_MATCH":
        if family not in {"MOTSP", "MOKP"}:
            raise ValueError("C0_POP_MATCH requires the problem family.")
        count = 48 if family == "MOTSP" else 40
        resized = _resize_open_biobjective_directions(directions, count)
        return V21E3HybridConfig(
            candidate_id="C0", reference_directions=resized,
            charged_evaluations=charged_evaluations, checkpoint_period=checkpoint_period,
            seed=seed, phase="development", trace_database=trace_database,
            terminal_receipt=terminal_receipt,
            development_diagnostic_id="C0_POP_MATCH",
        )
    raise ValueError("Not a hybrid development-diagnostic arm.")


def baseline_diagnostic_configs(
    *,
    family: str,
    arm_id: str,
    charged_evaluations: int,
    checkpoint_period: int,
    seed: int,
    trace_directory: str | Path | None = None,
) -> V21E3BaselineConfig:
    if arm_id not in {
        "NSGAII_STANDARD", "NSGAII_SEEDED", "NSGAII_POP21",
        "NSGAII_SEEDED_POP21", "MOEAD_STANDARD", "MOEAD_SEEDED",
        "MOEAD_POP21", "MOEAD_SEEDED_POP21",
    }:
        raise ValueError("Not a baseline development-diagnostic arm.")
    base = "NSGAII" if arm_id.startswith("NSGAII") else "MOEAD"
    seeded = "SEEDED" in arm_id
    population_size = 21 if arm_id.endswith("POP21") else None
    configs = frozen_development_baseline_configs(
        family=family,
        charged_evaluations=charged_evaluations,
        checkpoint_period=checkpoint_period,
        seed=seed,
        trace_directory=trace_directory,
        initialization_policy=(
            "family_aware_per_slot_construction_development_diagnostic_v1"
            if seeded
            else "problem_native_exact_random_solution_v1"
        ),
        development_diagnostic_id=arm_id,
        population_size_override=population_size,
    )
    return configs[base]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-zip", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(analyze_results_zip(args.results_zip, args.output_directory), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
