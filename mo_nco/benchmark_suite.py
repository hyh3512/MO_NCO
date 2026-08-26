from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .benchmark import (
    RunRecord,
    _validated_metric_reference,
    paired_sign_summary,
    resolve_predeclared_algorithm_configuration,
    run_benchmark,
)
from .instance import MultiObjectiveTSPInstance, instance_sha256


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    kind: str = "synthetic"
    cities: int = 30
    instance_seed: int = 0
    tsplib_files: Sequence[str] = ()
    tsplib_sha256: Sequence[str] = ()
    bitsp_file: str = ""
    bitsp_sha256: str = ""
    instance_sha256: str = ""
    population: Optional[int] = None
    evaluations: Optional[int] = None
    manifest_directory: Path = Path(".")

    def load_instance(self) -> Optional[MultiObjectiveTSPInstance]:
        if self.kind == "synthetic":
            return None
        if self.kind == "tsplib":
            paths = tuple(self._resolve_path(path) for path in self.tsplib_files)
            if self.tsplib_sha256:
                if len(self.tsplib_sha256) != len(paths):
                    raise ValueError(
                        f"Case {self.name} must bind one SHA-256 per TSPLIB file."
                    )
                for path, expected in zip(paths, self.tsplib_sha256):
                    self._verify_file_sha256(path, expected)
            instance = MultiObjectiveTSPInstance.from_tsplib_files(paths)
        elif self.kind == "bitsp":
            path = self._resolve_path(self.bitsp_file)
            if self.bitsp_sha256:
                self._verify_file_sha256(path, self.bitsp_sha256)
            instance = MultiObjectiveTSPInstance.from_bitsp_file(path)
        else:
            raise ValueError(f"Unknown benchmark case kind: {self.kind}")
        if self.instance_sha256:
            observed = instance_sha256(instance)
            if observed != self.instance_sha256:
                raise ValueError(
                    f"Case {self.name} instance SHA-256 mismatch: "
                    f"expected={self.instance_sha256}, observed={observed}."
                )
        return instance

    def _resolve_path(self, value: str) -> Path:
        candidate = Path(value).expanduser()
        if candidate.is_absolute() or candidate.is_file():
            return candidate
        return self.manifest_directory / candidate

    def _verify_file_sha256(self, path: Path, expected: str) -> None:
        if (
            len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError(
                f"Case {self.name} contains an invalid artifact SHA-256."
            )
        if not path.is_file():
            raise ValueError(f"Case {self.name} artifact is missing: {path}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(
                f"Case {self.name} artifact SHA-256 mismatch for {path}: "
                f"expected={expected}, observed={observed}."
            )


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    cases: Sequence[BenchmarkCase]

    @staticmethod
    def from_json(path: Path) -> "BenchmarkSuite":
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases = [
            BenchmarkCase(
                name=item["name"],
                kind=item.get("kind", "synthetic"),
                cities=int(item.get("cities", 30)),
                instance_seed=int(item.get("instance_seed", 0)),
                tsplib_files=tuple(item.get("tsplib_files", ())),
                tsplib_sha256=tuple(item.get("tsplib_sha256", ())),
                bitsp_file=item.get("bitsp_file", ""),
                bitsp_sha256=item.get("bitsp_sha256", ""),
                instance_sha256=item.get("instance_sha256", ""),
                population=item.get("population"),
                evaluations=item.get("evaluations"),
                manifest_directory=path.expanduser().resolve().parent,
            )
            for item in payload.get("cases", [])
        ]
        return BenchmarkSuite(name=payload.get("name", path.stem), cases=cases)


def load_metric_reference_manifest(path: Path) -> tuple[Dict[str, Dict[str, object]], str]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid metric reference manifest: {path}") from exc
    if payload.get("schema_version") != 1:
        raise ValueError("Metric reference manifest schema_version must be 1.")
    expected_keys = {
        "schema_version",
        "contract",
        "created_utc",
        "calibration_only",
        "evaluated_theory_arms_forbidden",
        "calibration_algorithms",
        "reference_margin",
        "source_files",
        "cases",
    }
    if set(payload) != expected_keys:
        raise ValueError(
            "Metric reference manifest has an unexpected top-level shape."
        )
    if payload.get("contract") != "frozen_external_v1":
        raise ValueError(
            "Metric reference manifest contract must be "
            "'frozen_external_v1'."
        )
    if payload.get("calibration_only") is not True:
        raise ValueError(
            "Metric reference manifest must declare calibration_only=true."
        )
    created_utc = payload.get("created_utc")
    if not isinstance(created_utc, str) or not created_utc:
        raise ValueError(
            "Metric reference manifest created_utc must be nonempty."
        )
    forbidden = payload.get("evaluated_theory_arms_forbidden")
    calibration_algorithms = payload.get("calibration_algorithms")
    if (
        not isinstance(forbidden, list)
        or any(not isinstance(item, str) or not item for item in forbidden)
        or not isinstance(calibration_algorithms, list)
        or not calibration_algorithms
        or any(
            not isinstance(item, str) or not item
            for item in calibration_algorithms
        )
        or set(forbidden).intersection(calibration_algorithms)
    ):
        raise ValueError(
            "Metric reference manifest has invalid or overlapping "
            "calibration algorithm declarations."
        )
    try:
        reference_margin = float(payload.get("reference_margin"))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Metric reference manifest reference_margin is invalid."
        ) from error
    if not math.isfinite(reference_margin) or reference_margin <= 0.0:
        raise ValueError(
            "Metric reference manifest reference_margin must be positive "
            "and finite."
        )
    source_files = payload.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError(
            "Metric reference manifest source_files must be nonempty."
        )
    for index, source in enumerate(source_files):
        if (
            not isinstance(source, dict)
            or set(source) != {"path", "sha256"}
            or not isinstance(source.get("path"), str)
            or not source["path"]
            or not isinstance(source.get("sha256"), str)
            or len(source["sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in source["sha256"]
            )
        ):
            raise ValueError(
                "Metric reference manifest source_files"
                f"[{index}] is invalid."
            )
    cases = payload.get("cases")
    if not isinstance(cases, dict) or not cases:
        raise ValueError("Metric reference manifest must contain a non-empty cases mapping.")
    normalized: Dict[str, Dict[str, object]] = {}
    for case, reference in cases.items():
        if not isinstance(case, str) or not isinstance(reference, dict):
            raise ValueError("Metric reference manifest cases must map names to objects.")
        reference_payload = dict(reference)
        declared_digest = reference_payload.pop(
            "reference_sha256",
            None,
        )
        if (
            not isinstance(declared_digest, str)
            or len(declared_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in declared_digest
            )
        ):
            raise ValueError(
                f"Metric reference case {case} lacks a valid reference_sha256."
            )
        canonical = json.dumps(
            reference_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != declared_digest:
            raise ValueError(
                f"Metric reference case {case} payload hash mismatch."
            )
        try:
            _validated_metric_reference(reference_payload)
        except ValueError as error:
            raise ValueError(
                f"Metric reference case {case} fails semantic "
                f"validation: {error}"
            ) from error
        reference_payload["reference_sha256"] = declared_digest
        normalized[case] = reference_payload
    return normalized, hashlib.sha256(raw).hexdigest()


def run_benchmark_suite(
    suite: BenchmarkSuite,
    algorithms: Sequence[str],
    seeds: Sequence[int],
    output_dir: Path,
    default_population: int,
    default_evaluations: int,
    log_period: int,
    archive_update_period: int,
    override_case_evaluations: bool = False,
    execution_order: str = "algorithm-major",
    metric_references: Optional[Dict[str, Dict[str, object]]] = None,
    metric_reference_manifest_sha256: str = "",
    certified_traces: bool = False,
    measure_python_memory: bool = False,
    output_archive_limit: Optional[int] = None,
    suite_manifest_sha256: str = "",
    information_contract_sha256: str = "",
    budget_scope: str = "single_run_objective_evaluations",
    expected_algorithm_configurations: Optional[
        Dict[Tuple[str, str, int], str]
    ] = None,
    anytime_checkpoint_period: Optional[int] = None,
) -> List[RunRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records: List[RunRecord] = []
    aggregate_rows: List[Dict[str, object]] = []

    for case_position, case in enumerate(suite.cases):
        case_output = output_dir / case.name
        instance = case.load_instance()
        if instance is None:
            instance = MultiObjectiveTSPInstance.random_biobjective(
                case.cities,
                seed=case.instance_seed,
            )
        metric_reference = None
        if metric_references is not None:
            if case.name not in metric_references:
                raise ValueError(f"Metric reference manifest is missing case: {case.name}")
            metric_reference = metric_references[case.name]
        records, _ = run_benchmark(
            algorithms=algorithms,
            seeds=seeds,
            cities=case.cities,
            population=int(case.population or default_population),
            iterations=(
                int(default_evaluations)
                if override_case_evaluations
                else int(case.evaluations or default_evaluations)
            ),
            instance_seed=case.instance_seed,
            output_dir=case_output,
            log_period=log_period,
            archive_update_period=archive_update_period,
            instance=instance,
            execution_order=execution_order,
            execution_order_offset=case_position * len(seeds),
            metric_reference=metric_reference,
            metric_reference_manifest_sha256=metric_reference_manifest_sha256,
            certified_traces=certified_traces,
            measure_python_memory=measure_python_memory,
            output_archive_limit=output_archive_limit,
            case_name=case.name,
            expected_algorithm_configurations=(
                None
                if expected_algorithm_configurations is None
                else {
                    (algorithm, seed): digest
                    for (
                        expected_case,
                        algorithm,
                        seed,
                    ), digest in expected_algorithm_configurations.items()
                    if expected_case == case.name
                }
            ),
            anytime_checkpoint_period=anytime_checkpoint_period,
        )
        all_records.extend(records)
        for record in records:
            row = record.__dict__.copy()
            row["case"] = case.name
            row["num_cities"] = instance.num_cities
            row["instance_sha256"] = instance_sha256(instance)
            row["suite_sha256"] = suite_manifest_sha256
            row["reference_manifest_sha256"] = (
                metric_reference_manifest_sha256
            )
            row["reference_sha256"] = (
                str(metric_reference.get("reference_sha256", ""))
                if metric_reference is not None
                else ""
            )
            row["information_signature_sha256"] = (
                information_contract_sha256
            )
            row["budget_scope"] = budget_scope
            row["archive_limit"] = (
                output_archive_limit
                if output_archive_limit is not None
                else ""
            )
            row["runtime_measurement_contract"] = (
                "uninstrumented_wall_clock_inprocess_v1"
            )
            row["execution_order_contract"] = execution_order
            row["memory_measurement_contract"] = (
                "python_tracemalloc_separate_replay_peak_increment_v1"
                if measure_python_memory
                else "disabled"
            )
            row["memory_replay_order_contract"] = (
                "all_case_timed_runs_before_case_memory_replays_v1"
                if measure_python_memory
                else "not_applicable"
            )
            row["memory_replay_state_equivalence_gate"] = (
                "PASS" if measure_python_memory else "NOT_RUN"
            )
            aggregate_rows.append(row)

    add_case_relative_metrics(aggregate_rows)
    write_aggregate_runs(output_dir / "aggregate_runs.csv", aggregate_rows)
    write_suite_summary(output_dir / "suite_summary.md", aggregate_rows)
    write_suite_pairwise(output_dir / "suite_pairwise.md", aggregate_rows)
    return all_records


def build_algorithm_configuration_manifest(
    *,
    suite: BenchmarkSuite,
    suite_sha256: str,
    algorithms: Sequence[str],
    seeds: Sequence[int],
    default_population: int,
    default_evaluations: int,
    log_period: int,
    archive_update_period: int,
    override_case_evaluations: bool,
    output_archive_limit: Optional[int],
    certified_traces: bool,
    anytime_checkpoint_period: Optional[int] = None,
) -> Dict[str, object]:
    """Build the complete prelaunch matrix without evaluating a tour."""

    rows = []
    for case in suite.cases:
        instance = case.load_instance()
        if instance is None:
            instance = MultiObjectiveTSPInstance.random_biobjective(
                case.cities,
                seed=case.instance_seed,
            )
        population = int(case.population or default_population)
        evaluations = (
            int(default_evaluations)
            if override_case_evaluations
            else int(case.evaluations or default_evaluations)
        )
        for algorithm in algorithms:
            for seed in seeds:
                configuration = (
                    resolve_predeclared_algorithm_configuration(
                        case_name=case.name,
                        instance=instance,
                        algorithm=algorithm,
                        seed=seed,
                        population=population,
                        iterations=evaluations,
                        log_period=log_period,
                        archive_update_period=archive_update_period,
                        output_archive_limit=output_archive_limit,
                        certified_traces=certified_traces,
                        anytime_checkpoint_period=(
                            anytime_checkpoint_period
                        ),
                    )
                )
                readable_configuration = json.loads(
                    json.dumps(
                        configuration.payload,
                        sort_keys=True,
                        allow_nan=False,
                    )
                )
                rows.append(
                    {
                        "case": case.name,
                        "algorithm": algorithm,
                        "seed": seed,
                        "population": population,
                        "algorithm_configuration_sha256": (
                            configuration.sha256
                        ),
                        "search_evaluations": (
                            configuration.search_evaluations
                        ),
                        "pilot_evaluations": (
                            configuration.pilot_evaluations
                        ),
                        "confirm_evaluations": (
                            configuration.confirm_evaluations
                        ),
                        "algorithm_configuration": (
                            readable_configuration
                        ),
                    }
                )
    return {
        "schema": "pareto_smc_algorithm_configuration_manifest_v2",
        "suite_sha256": suite_sha256,
        "runs": rows,
    }


def load_and_verify_algorithm_configuration_manifest(
    path: Path,
    *,
    expected: Dict[str, object],
) -> Tuple[Dict[Tuple[str, str, int], str], str]:
    """Verify exact prelaunch bytes against the locally resolved matrix."""

    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Algorithm-configuration manifest is not valid UTF-8 JSON."
        ) from error
    if payload != expected:
        raise ValueError(
            "Frozen algorithm-configuration manifest does not exactly match "
            "the current prelaunch configuration matrix."
        )
    rows = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(
            "Algorithm-configuration manifest runs must be an array."
        )
    index: Dict[Tuple[str, str, int], str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(
                "Algorithm-configuration manifest row must be an object."
            )
        key = (
            str(row.get("case", "")),
            str(row.get("algorithm", "")),
            int(row.get("seed")),
        )
        if key in index:
            raise ValueError(
                f"Duplicate algorithm-configuration row: {key}."
            )
        configuration_payload = row.get("algorithm_configuration")
        if not isinstance(configuration_payload, dict):
            raise ValueError(
                f"Algorithm-configuration row {key} has no readable "
                "configuration payload."
            )
        declared_digest = str(
            row.get("algorithm_configuration_sha256", "")
        )
        observed_digest = hashlib.sha256(
            json.dumps(
                configuration_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if declared_digest != observed_digest:
            raise ValueError(
                f"Algorithm-configuration payload hash mismatch: {key}."
            )
        index[key] = declared_digest
    return index, hashlib.sha256(raw).hexdigest()


def write_aggregate_runs(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = ["case", *[field for field in rows[0] if field != "case"]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_case_relative_metrics(rows: Sequence[Dict[str, object]]) -> None:
    by_case: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case"]), []).append(row)
    metric_pairs = [
        ("hypervolume_2d", "case_relative_hypervolume_2d"),
        ("anytime_hv_eval_auc", "case_relative_anytime_hv_eval_auc"),
        ("anytime_hv_time_auc", "case_relative_anytime_hv_time_auc"),
        ("hypervolume_per_second", "case_relative_hypervolume_per_second"),
    ]
    for case_rows in by_case.values():
        for metric, relative_metric in metric_pairs:
            best = max(float(row.get(metric, 0.0)) for row in case_rows)
            for row in case_rows:
                row[relative_metric] = float(row.get(metric, 0.0)) / best if best > 0.0 else 0.0


def write_suite_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["algorithm"]), []).append(row)

    lines = [
        "# Benchmark Suite Summary",
        "",
        "| algorithm | cases | HV rank mean | rel HV mean | eval-AUC rank mean | rel eval-AUC mean | time-AUC rank mean | rel time-AUC mean | HV/sec rank mean | rel HV/sec mean | evals mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    case_names = sorted({str(row["case"]) for row in rows})
    for algorithm, group in sorted(grouped.items()):
        hv_ranks = []
        auc_ranks = []
        time_auc_ranks = []
        speed_ranks = []
        rel_hv_values = [float(row.get("case_relative_hypervolume_2d", 0.0)) for row in group]
        rel_auc_values = [float(row.get("case_relative_anytime_hv_eval_auc", 0.0)) for row in group]
        rel_time_auc_values = [float(row.get("case_relative_anytime_hv_time_auc", 0.0)) for row in group]
        rel_speed_values = [float(row.get("case_relative_hypervolume_per_second", 0.0)) for row in group]
        for case in case_names:
            case_rows = [row for row in rows if row["case"] == case]
            algorithms = sorted({str(row["algorithm"]) for row in case_rows})
            means = []
            for item in algorithms:
                algo_rows = [row for row in case_rows if row["algorithm"] == item]
                means.append(
                    {
                        "algorithm": item,
                        "hypervolume_2d": sum(float(row["hypervolume_2d"]) for row in algo_rows) / len(algo_rows),
                        "anytime_hv_eval_auc": sum(float(row["anytime_hv_eval_auc"]) for row in algo_rows)
                        / len(algo_rows),
                        "anytime_hv_time_auc": sum(float(row["anytime_hv_time_auc"]) for row in algo_rows)
                        / len(algo_rows),
                        "hypervolume_per_second": sum(float(row["hypervolume_per_second"]) for row in algo_rows)
                        / len(algo_rows),
                    }
                )
            hv_order = sorted(means, key=lambda row: float(row["hypervolume_2d"]), reverse=True)
            auc_order = sorted(means, key=lambda row: float(row["anytime_hv_eval_auc"]), reverse=True)
            time_auc_order = sorted(means, key=lambda row: float(row["anytime_hv_time_auc"]), reverse=True)
            speed_order = sorted(means, key=lambda row: float(row["hypervolume_per_second"]), reverse=True)
            hv_ranks.extend(idx + 1 for idx, row in enumerate(hv_order) if row["algorithm"] == algorithm)
            auc_ranks.extend(idx + 1 for idx, row in enumerate(auc_order) if row["algorithm"] == algorithm)
            time_auc_ranks.extend(idx + 1 for idx, row in enumerate(time_auc_order) if row["algorithm"] == algorithm)
            speed_ranks.extend(idx + 1 for idx, row in enumerate(speed_order) if row["algorithm"] == algorithm)
        evals = [float(row["evaluations"]) for row in group]
        lines.append(
            f"| {algorithm} | {len(case_names)} | {sum(hv_ranks) / max(1, len(hv_ranks)):.3g} | "
            f"{_mean(rel_hv_values):.4g} | "
            f"{sum(auc_ranks) / max(1, len(auc_ranks)):.3g} | {_mean(rel_auc_values):.4g} | "
            f"{sum(time_auc_ranks) / max(1, len(time_auc_ranks)):.3g} | {_mean(rel_time_auc_values):.4g} | "
            f"{sum(speed_ranks) / max(1, len(speed_ranks)):.3g} | {_mean(rel_speed_values):.4g} | "
            f"{sum(evals) / max(1, len(evals)):.4g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_suite_pairwise(path: Path, rows: Sequence[Dict[str, object]], anchor: str = "ips-theory") -> None:
    if not rows:
        path.write_text("# Benchmark Suite Pairwise Tests\n\nNo rows.\n", encoding="utf-8")
        return
    algorithms = sorted({str(row["algorithm"]) for row in rows})
    if anchor not in algorithms:
        anchor = algorithms[0]
    keyed = {
        (str(row["case"]), int(row["seed"]), str(row["algorithm"])): row
        for row in rows
    }
    lines = [
        "# Benchmark Suite Pairwise Tests",
        "",
        f"Anchor: `{anchor}`. Deltas are anchor minus comparator over matched `(case, seed)` pairs.",
        "The p-values use an exact two-sided sign test after dropping ties.",
        "",
        "| comparator | pairs | Δrel HV | rel HV wins-losses | rel HV p | Δrel eval-AUC | rel eval-AUC wins-losses | rel eval-AUC p | Δrel time-AUC | rel time-AUC wins-losses | rel time-AUC p | Δrel HV/sec | rel HV/sec wins-losses | rel HV/sec p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    anchor_rows = [row for row in rows if str(row["algorithm"]) == anchor]
    for comparator in algorithms:
        if comparator == anchor:
            continue
        hv_deltas = []
        auc_deltas = []
        time_auc_deltas = []
        speed_deltas = []
        for row in anchor_rows:
            key = (str(row["case"]), int(row["seed"]), comparator)
            other = keyed.get(key)
            if other is None:
                continue
            hv_deltas.append(
                float(row.get("case_relative_hypervolume_2d", 0.0))
                - float(other.get("case_relative_hypervolume_2d", 0.0))
            )
            auc_deltas.append(
                float(row.get("case_relative_anytime_hv_eval_auc", 0.0))
                - float(other.get("case_relative_anytime_hv_eval_auc", 0.0))
            )
            time_auc_deltas.append(
                float(row.get("case_relative_anytime_hv_time_auc", 0.0))
                - float(other.get("case_relative_anytime_hv_time_auc", 0.0))
            )
            speed_deltas.append(
                float(row.get("case_relative_hypervolume_per_second", 0.0))
                - float(other.get("case_relative_hypervolume_per_second", 0.0))
            )
        hv_wins, hv_losses, hv_p = paired_sign_summary(hv_deltas)
        auc_wins, auc_losses, auc_p = paired_sign_summary(auc_deltas)
        time_auc_wins, time_auc_losses, time_auc_p = paired_sign_summary(time_auc_deltas)
        speed_wins, speed_losses, speed_p = paired_sign_summary(speed_deltas)
        lines.append(
            "| "
            f"{comparator} | {len(hv_deltas)} | "
            f"{_mean(hv_deltas):.6g} | {hv_wins}-{hv_losses} | {hv_p:.4g} | "
            f"{_mean(auc_deltas):.6g} | {auc_wins}-{auc_losses} | {auc_p:.4g} | "
            f"{_mean(time_auc_deltas):.6g} | {time_auc_wins}-{time_auc_losses} | {time_auc_p:.4g} | "
            f"{_mean(speed_deltas):.6g} | {speed_wins}-{speed_losses} | {speed_p:.4g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
