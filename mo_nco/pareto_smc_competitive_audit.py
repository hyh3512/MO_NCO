from __future__ import annotations

"""Fail-closed empirical gate for matched Pareto-SMC comparisons.

This module deliberately separates competitive evidence from the
finite-particle reference certificate.  It accepts only a complete,
predeclared case x algorithm x seed matrix whose budget, reference and
information signatures are matched within every pair.
"""

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence, Tuple

from .run_pareto_smc_strict_analysis import (
    _cluster_bootstrap_ci,
    _cluster_randomization_p,
    _trimmed_mean,
    _winsorized_mean,
    analyze,
)


PROTOCOL_SCHEMA = "pareto_smc_competitive_protocol_v2"
V12_COMPETITIVE_ANCHOR = "pareto-smc-pilot-confirm-v12"
PUBLICATION_CERTIFICATE_PACKET_GATE_COLUMN = (
    "publication_certificate_packet_gate"
)
INFORMATION_CONTRACT_SCHEMA = (
    "pareto_smc_competitive_information_contract_v2"
)
INFORMATION_CONTRACT_LIST_FIELDS = {
    "search_time_information",
    "forbidden_search_time_information",
    "postprocessing_information_shared_by_all_algorithms",
}
INFORMATION_CONTRACT_STRING_FIELDS = {
    "metric_reference_scope",
    "budget_scope",
    "timing_scope",
    "runtime_scope",
    "memory_scope",
    "objective_output_scope",
    "archive_output_scope",
    "anytime_front_scope",
    "evaluation_evidence_scope",
    "configuration_scope",
    "claim_limit",
}
PILOT_CONFIRM_ANCHOR_CONTRACTS = {
    "pareto-smc-pilot-confirm-v11": (
        "published",
        "v11_published",
    ),
    V12_COMPETITIVE_ANCHOR: (
        "regeneration",
        "v12_regeneration",
    ),
}
REQUIRED_COLUMNS = {
    "case",
    "algorithm",
    "seed",
    "population",
    "algorithm_configuration_sha256",
    "search_evaluations",
    "pilot_evaluations",
    "confirm_evaluations",
    "suite_sha256",
    "reference_manifest_sha256",
    "num_cities",
    "instance_sha256",
    "evaluations",
    "budget_scope",
    "archive_size",
    "archive_limit",
    "reference_sha256",
    "information_signature_sha256",
    "case_relative_hypervolume_2d",
    "case_relative_anytime_hv_eval_auc",
    "igd_plus",
    "additive_epsilon",
    "runtime_seconds",
    "python_peak_traced_memory_bytes",
    "runtime_measurement_contract",
    "execution_order_contract",
    "memory_measurement_contract",
    "memory_replay_order_contract",
    "memory_replay_state_equivalence_gate",
    "output_objective_equivalence_gate",
    "output_objective_max_abs_error",
    "output_objective_equivalence_contract",
    "anytime_objective_equivalence_gate",
    "anytime_objective_equivalence_contract",
    "evaluation_evidence_gate",
    "evaluation_evidence_contract",
    "native_archive_completeness_gate",
    "native_archive_completeness_contract",
    "anytime_front_semantics",
    "anytime_checkpoint_gate",
    "anytime_checkpoint_contract",
    "anytime_checkpoint_period",
    "anytime_checkpoint_count",
    "anytime_auc_integration_contract",
    "anytime_time_auc_status",
    "max_diagnostic_archive_size",
    "diagnostic_archive_limit_gate",
    "diagnostic_archive_limit_contract",
}


@dataclass(frozen=True)
class CompetitiveProtocol:
    path: Path
    sha256: str
    anchor: str
    algorithms: Tuple[str, ...]
    suite_path: Path
    suite_sha256: str
    reference_manifest_path: Path | None
    reference_manifest_sha256: str | None
    information_contract_path: Path
    information_contract_sha256: str
    algorithm_configuration_manifest_path: Path | None
    algorithm_configuration_manifest_sha256: str | None
    expected_run_configurations: Tuple[
        Tuple[str, str, int, int, str, int, int, int],
        ...,
    ] | None
    expected_cases: int
    expected_case_ids: Tuple[str, ...]
    expected_case_cities: Tuple[Tuple[str, int], ...]
    expected_instance_sha256: Tuple[Tuple[str, str], ...]
    expected_reference_sha256: Tuple[Tuple[str, str], ...] | None
    seed_ids: Tuple[int, ...]
    minimum_cities: int
    required_city_sizes: Tuple[int, ...]
    evaluations_per_run: int
    budget_scope: str
    archive_limit: int
    runtime_measurement_contract: str
    execution_order_contract: str
    memory_measurement_contract: str
    memory_replay_order_contract: str
    output_objective_equivalence_contract: str
    native_archive_completeness_contract: str
    anytime_front_semantics: str
    anytime_checkpoint_contract: str
    anytime_checkpoint_period: int
    anytime_auc_integration_contract: str
    anytime_time_auc_status: str
    diagnostic_archive_limit_contract: str
    igd_plus_noninferiority_margin: float
    maximum_runtime_ratio: float
    maximum_python_memory_ratio: float
    bootstrap_repetitions: int
    randomization_repetitions: int
    analysis_random_seed: int


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} has an unexpected shape; "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}."
        )


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer.")
    return value


def _positive_ratio(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 1.0:
        raise ValueError(f"{label} must be finite and at least one.")
    return result


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _artifact_path(
    value: object,
    *,
    protocol_path: Path,
    label: str,
    allow_none: bool = False,
) -> Path | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty path string.")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = protocol_path.parent / candidate
    return candidate.resolve()


def _verified_json_artifact(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected_sha256:
        raise ValueError(
            f"{label} hash mismatch: expected={expected_sha256}, "
            f"observed={observed}."
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON.") from error
    return _mapping(payload, label)


def _validate_information_contract(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected_keys = {
        "schema",
        "objective_sense",
        *INFORMATION_CONTRACT_LIST_FIELDS,
        *INFORMATION_CONTRACT_STRING_FIELDS,
    }
    _exact_keys(payload, expected_keys, "information contract")
    if payload.get("schema") != INFORMATION_CONTRACT_SCHEMA:
        raise ValueError(
            "information contract has the wrong schema; expected "
            f"{INFORMATION_CONTRACT_SCHEMA!r}."
        )
    if payload.get("objective_sense") != "minimize_all":
        raise ValueError(
            "information contract objective_sense must be 'minimize_all'."
        )
    for field in sorted(INFORMATION_CONTRACT_LIST_FIELDS):
        values = payload.get(field)
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(value, str) or not value.strip()
                for value in values
            )
            or len(set(values)) != len(values)
        ):
            raise ValueError(
                f"information contract {field} must be a nonempty array "
                "of distinct nonempty strings."
            )
    for field in sorted(INFORMATION_CONTRACT_STRING_FIELDS):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"information contract {field} must be a nonempty string."
            )
    return payload


def _validate_pilot_confirm_anchor_configuration(
    configuration: Mapping[str, Any],
    *,
    anchor: str,
    key: tuple[str, str, int],
) -> None:
    contract = PILOT_CONFIRM_ANCHOR_CONTRACTS.get(anchor)
    if contract is None:
        return
    if (
        configuration.get("schema")
        != "mo_nco_predeclared_algorithm_configuration_v2"
    ):
        raise ValueError(
            f"Anchor algorithm configuration {key} has the wrong schema."
        )
    algorithm_specific = _mapping(
        configuration.get("algorithm_specific"),
        f"anchor algorithm configuration {key} algorithm_specific",
    )
    expected_mode, expected_protocol_version = contract
    if algorithm_specific.get("certificate_mode") != expected_mode:
        raise ValueError(
            f"Anchor algorithm configuration {key} certificate_mode must "
            f"be {expected_mode!r}."
        )
    if (
        algorithm_specific.get("pilot_confirm_protocol_version")
        != expected_protocol_version
    ):
        raise ValueError(
            f"Anchor algorithm configuration {key} "
            "pilot_confirm_protocol_version must be "
            f"{expected_protocol_version!r}."
        )
    _sha256(
        algorithm_specific.get("certificate_specification_sha256"),
        (
            f"anchor algorithm configuration {key} "
            "certificate_specification_sha256"
        ),
    )
    _sha256(
        algorithm_specific.get("certificate_manifest_sha256"),
        (
            f"anchor algorithm configuration {key} "
            "certificate_manifest_sha256"
        ),
    )


def load_competitive_protocol(path: str | Path) -> CompetitiveProtocol:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Competitive protocol is missing: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Competitive protocol is not valid UTF-8 JSON.") from error
    root = _mapping(payload, "competitive protocol")
    _exact_keys(
        root,
        {
            "schema",
            "anchor",
            "algorithms",
            "suite_path",
            "suite_sha256",
            "reference_manifest_path",
            "reference_manifest_sha256",
            "information_contract_path",
            "information_contract_sha256",
            "algorithm_configuration_manifest_path",
            "algorithm_configuration_manifest_sha256",
            "cases",
            "seeds",
            "fairness",
            "gates",
            "analysis",
        },
        "competitive protocol",
    )
    if root.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"Competitive protocol schema must be {PROTOCOL_SCHEMA!r}.")

    anchor = root.get("anchor")
    algorithms_raw = root.get("algorithms")
    if (
        not isinstance(anchor, str)
        or not anchor
        or not isinstance(algorithms_raw, list)
        or len(algorithms_raw) < 2
        or any(not isinstance(item, str) or not item for item in algorithms_raw)
    ):
        raise ValueError("anchor and algorithms must name at least two algorithms.")
    algorithms = tuple(algorithms_raw)
    if len(set(algorithms)) != len(algorithms) or anchor not in algorithms:
        raise ValueError("algorithms must be unique and include anchor.")
    suite_sha256 = _sha256(root.get("suite_sha256"), "suite_sha256")
    reference_manifest_raw = root.get("reference_manifest_sha256")
    reference_manifest_sha256 = (
        None
        if reference_manifest_raw is None
        else _sha256(
            reference_manifest_raw,
            "reference_manifest_sha256",
        )
    )
    information_contract_sha256 = _sha256(
        root.get("information_contract_sha256"),
        "information_contract_sha256",
    )
    suite_path = _artifact_path(
        root.get("suite_path"),
        protocol_path=resolved,
        label="suite_path",
    )
    assert suite_path is not None
    reference_manifest_path = _artifact_path(
        root.get("reference_manifest_path"),
        protocol_path=resolved,
        label="reference_manifest_path",
        allow_none=True,
    )
    if (reference_manifest_path is None) != (
        reference_manifest_sha256 is None
    ):
        raise ValueError(
            "reference_manifest_path and reference_manifest_sha256 must "
            "either both be null or both be frozen."
        )
    information_contract_path = _artifact_path(
        root.get("information_contract_path"),
        protocol_path=resolved,
        label="information_contract_path",
    )
    assert information_contract_path is not None
    configuration_manifest_raw = root.get(
        "algorithm_configuration_manifest_sha256"
    )
    configuration_manifest_sha256 = (
        None
        if configuration_manifest_raw is None
        else _sha256(
            configuration_manifest_raw,
            "algorithm_configuration_manifest_sha256",
        )
    )
    configuration_manifest_path = _artifact_path(
        root.get("algorithm_configuration_manifest_path"),
        protocol_path=resolved,
        label="algorithm_configuration_manifest_path",
        allow_none=True,
    )
    if (configuration_manifest_path is None) != (
        configuration_manifest_sha256 is None
    ):
        raise ValueError(
            "algorithm_configuration_manifest_path and SHA-256 must either "
            "both be null or both be frozen."
        )

    cases = _mapping(root.get("cases"), "cases")
    _exact_keys(
        cases,
        {
            "expected_count",
            "expected_ids",
            "minimum_cities",
            "required_city_sizes",
        },
        "cases",
    )
    expected_cases = _positive_int(cases.get("expected_count"), "cases.expected_count")
    expected_ids_raw = cases.get("expected_ids")
    if (
        not isinstance(expected_ids_raw, list)
        or any(
            not isinstance(value, str) or not value
            for value in expected_ids_raw
        )
        or len(set(expected_ids_raw)) != len(expected_ids_raw)
        or len(expected_ids_raw) != expected_cases
    ):
        raise ValueError(
            "cases.expected_ids must contain exactly expected_count "
            "distinct nonempty case IDs."
        )
    expected_case_ids = tuple(sorted(expected_ids_raw))
    minimum_cities = _positive_int(cases.get("minimum_cities"), "cases.minimum_cities")
    sizes_raw = cases.get("required_city_sizes")
    if (
        not isinstance(sizes_raw, list)
        or not sizes_raw
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < minimum_cities
            for value in sizes_raw
        )
    ):
        raise ValueError(
            "cases.required_city_sizes must contain integer sizes no smaller "
            "than cases.minimum_cities."
        )
    required_sizes = tuple(sorted(set(sizes_raw)))

    suite_payload = _verified_json_artifact(
        suite_path,
        expected_sha256=suite_sha256,
        label="benchmark suite",
    )
    suite_cases_raw = suite_payload.get("cases")
    if not isinstance(suite_cases_raw, list):
        raise ValueError("benchmark suite must contain a cases array.")
    suite_case_cities: dict[str, int] = {}
    suite_instance_hashes: dict[str, str] = {}
    suite_case_evaluations: dict[str, int] = {}
    for index, raw_case in enumerate(suite_cases_raw):
        case = _mapping(raw_case, f"benchmark suite case {index}")
        name = case.get("name")
        cities = case.get("cities")
        instance_digest = case.get("instance_sha256")
        case_evaluations = case.get("evaluations")
        if not isinstance(name, str) or not name or name in suite_case_cities:
            raise ValueError(
                "benchmark suite case names must be distinct and nonempty."
            )
        if isinstance(cities, bool) or not isinstance(cities, int) or cities <= 0:
            raise ValueError(
                f"benchmark suite case {name} has invalid cities."
            )
        if (
            isinstance(case_evaluations, bool)
            or not isinstance(case_evaluations, int)
            or case_evaluations <= 0
        ):
            raise ValueError(
                f"benchmark suite case {name} has invalid evaluations."
            )
        suite_case_cities[name] = cities
        suite_instance_hashes[name] = _sha256(
            instance_digest,
            f"benchmark suite case {name} instance_sha256",
        )
        suite_case_evaluations[name] = case_evaluations
    if tuple(sorted(suite_case_cities)) != expected_case_ids:
        raise ValueError(
            "benchmark suite case IDs do not match cases.expected_ids."
        )
    if any(
        suite_case_cities[name] < minimum_cities
        for name in expected_case_ids
    ):
        raise ValueError(
            "benchmark suite contains a case below cases.minimum_cities."
        )
    if not set(required_sizes).issubset(set(suite_case_cities.values())):
        raise ValueError(
            "benchmark suite does not contain every required city-size stratum."
        )

    information_contract = _validate_information_contract(
        _verified_json_artifact(
            information_contract_path,
            expected_sha256=information_contract_sha256,
            label="information contract",
        )
    )
    if reference_manifest_path is None:
        expected_reference_hashes = None
    else:
        assert reference_manifest_sha256 is not None
        from .benchmark_suite import load_metric_reference_manifest

        try:
            normalized_references, observed_reference_manifest_sha = (
                load_metric_reference_manifest(
                    reference_manifest_path
                )
            )
        except (OSError, ValueError) as error:
            raise ValueError(
                "metric-reference manifest fails structural, provenance, "
                f"or semantic validation: {error}"
            ) from error
        if (
            observed_reference_manifest_sha
            != reference_manifest_sha256
        ):
            raise ValueError(
                "metric-reference manifest hash mismatch."
            )
        if tuple(sorted(normalized_references)) != expected_case_ids:
            raise ValueError(
                "metric-reference manifest case IDs do not match the suite."
            )
        expected_reference_hashes = {
            name: str(
                normalized_references[name]["reference_sha256"]
            )
            for name in expected_case_ids
        }

    seeds_raw = root.get("seeds")
    if (
        not isinstance(seeds_raw, list)
        or len(seeds_raw) < 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in seeds_raw)
        or len(set(seeds_raw)) != len(seeds_raw)
    ):
        raise ValueError("seeds must contain at least two distinct integer IDs.")
    seed_ids = tuple(sorted(seeds_raw))

    fairness = _mapping(root.get("fairness"), "fairness")
    _exact_keys(
        fairness,
        {
            "evaluations_per_run",
            "budget_scope",
            "archive_limit",
            "runtime_measurement_contract",
            "execution_order_contract",
            "memory_measurement_contract",
            "memory_replay_order_contract",
            "output_objective_equivalence_contract",
            "native_archive_completeness_contract",
            "anytime_front_semantics",
            "anytime_checkpoint_contract",
            "anytime_checkpoint_period",
            "anytime_auc_integration_contract",
            "anytime_time_auc_status",
            "diagnostic_archive_limit_contract",
        },
        "fairness",
    )
    evaluations = _positive_int(
        fairness.get("evaluations_per_run"),
        "fairness.evaluations_per_run",
    )
    if any(
        suite_case_evaluations[name] != evaluations
        for name in expected_case_ids
    ):
        raise ValueError(
            "Every frozen suite case must use fairness.evaluations_per_run."
        )
    archive_limit = _positive_int(
        fairness.get("archive_limit"),
        "fairness.archive_limit",
    )
    budget_scope = fairness.get("budget_scope")
    runtime_contract = fairness.get("runtime_measurement_contract")
    execution_order_contract = fairness.get(
        "execution_order_contract"
    )
    memory_contract = fairness.get("memory_measurement_contract")
    memory_replay_order_contract = fairness.get(
        "memory_replay_order_contract"
    )
    output_equivalence_contract = fairness.get(
        "output_objective_equivalence_contract"
    )
    native_archive_completeness_contract = fairness.get(
        "native_archive_completeness_contract"
    )
    anytime_front_semantics = fairness.get(
        "anytime_front_semantics"
    )
    anytime_checkpoint_contract = fairness.get(
        "anytime_checkpoint_contract"
    )
    anytime_checkpoint_period = _positive_int(
        fairness.get("anytime_checkpoint_period"),
        "fairness.anytime_checkpoint_period",
    )
    anytime_auc_integration_contract = fairness.get(
        "anytime_auc_integration_contract"
    )
    anytime_time_auc_status = fairness.get(
        "anytime_time_auc_status"
    )
    diagnostic_archive_limit_contract = fairness.get(
        "diagnostic_archive_limit_contract"
    )
    if budget_scope != "matched_total_objective_evaluations_including_pilot_confirm":
        raise ValueError(
            "fairness.budget_scope must charge the complete pilot-confirm budget."
        )
    if information_contract.get("budget_scope") != budget_scope:
        raise ValueError(
            "information contract budget_scope does not match "
            "fairness.budget_scope."
        )
    if (
        memory_contract
        != "python_tracemalloc_separate_replay_peak_increment_v1"
    ):
        raise ValueError(
            "fairness.memory_measurement_contract must use the verified "
            "separate-replay peak-increment contract."
        )
    if runtime_contract != "uninstrumented_wall_clock_inprocess_v1":
        raise ValueError(
            "fairness.runtime_measurement_contract must require an "
            "uninstrumented main run."
        )
    if execution_order_contract != "seed-major-balanced-v1":
        raise ValueError(
            "fairness.execution_order_contract must use the balanced "
            "seed-major schedule."
        )
    if (
        memory_replay_order_contract
        != "all_case_timed_runs_before_case_memory_replays_v1"
    ):
        raise ValueError(
            "fairness.memory_replay_order_contract must keep every memory "
            "replay after every timed arm for that matched case."
        )
    if (
        output_equivalence_contract
        != "local_full_tour_exact_on_integer_domain_else_"
        "rel1e-12_abs1e-12_v1"
    ):
        raise ValueError(
            "fairness.output_objective_equivalence_contract must require "
            "local full-tour replay."
        )
    if (
        anytime_front_semantics
        != "cumulative_nondominated_best_so_far_v1"
    ):
        raise ValueError(
            "fairness.anytime_front_semantics must require cumulative "
            "nondominated best-so-far fronts."
        )
    if (
        native_archive_completeness_contract
        != "unbounded_nondominated_all_evaluated_candidates_v1"
    ):
        raise ValueError(
            "fairness.native_archive_completeness_contract must require "
            "an unbounded native nondominated archive before the common cap."
        )
    if (
        diagnostic_archive_limit_contract
        != "deterministic_nondominated_crowding_"
        "truncation_per_snapshot_v1"
    ):
        raise ValueError(
            "fairness.diagnostic_archive_limit_contract must bind the "
            "common deterministic cap on every anytime snapshot."
        )
    if (
        anytime_checkpoint_contract
        != "exact_common_evaluation_checkpoint_archive_snapshot_v1"
    ):
        raise ValueError(
            "fairness.anytime_checkpoint_contract must require genuine "
            "archive snapshots on the frozen common evaluation grid."
        )
    if anytime_checkpoint_period > evaluations:
        raise ValueError(
            "fairness.anytime_checkpoint_period cannot exceed the run "
            "evaluation budget."
        )
    if (
        anytime_auc_integration_contract
        != "left_continuous_step_on_evaluation_snapshots_v1"
    ):
        raise ValueError(
            "fairness.anytime_auc_integration_contract must use "
            "left-continuous step integration."
        )
    if (
        anytime_time_auc_status
        != "descriptive_only_not_formal_quality_gate_v1"
    ):
        raise ValueError(
            "fairness.anytime_time_auc_status must keep time-AUC out of "
            "the formal quality gate."
        )

    gates = _mapping(root.get("gates"), "gates")
    _exact_keys(
        gates,
        {
            "igd_plus_noninferiority_margin",
            "maximum_runtime_ratio",
            "maximum_python_memory_ratio",
        },
        "gates",
    )
    igd_margin = float(gates.get("igd_plus_noninferiority_margin"))
    if not math.isfinite(igd_margin) or igd_margin < 0.0:
        raise ValueError(
            "gates.igd_plus_noninferiority_margin must be finite and nonnegative."
        )
    analysis = _mapping(root.get("analysis"), "analysis")
    _exact_keys(
        analysis,
        {
            "bootstrap_repetitions",
            "randomization_repetitions",
            "random_seed",
        },
        "analysis",
    )
    bootstrap_repetitions = _positive_int(
        analysis.get("bootstrap_repetitions"),
        "analysis.bootstrap_repetitions",
    )
    randomization_repetitions = _positive_int(
        analysis.get("randomization_repetitions"),
        "analysis.randomization_repetitions",
    )
    analysis_random_seed_raw = analysis.get("random_seed")
    if isinstance(analysis_random_seed_raw, bool) or not isinstance(
        analysis_random_seed_raw,
        int,
    ):
        raise ValueError("analysis.random_seed must be an integer.")

    if configuration_manifest_path is None:
        expected_run_configurations = None
    else:
        assert configuration_manifest_sha256 is not None
        configuration_payload = _verified_json_artifact(
            configuration_manifest_path,
            expected_sha256=configuration_manifest_sha256,
            label="algorithm-configuration manifest",
        )
        _exact_keys(
            configuration_payload,
            {"schema", "suite_sha256", "runs"},
            "algorithm-configuration manifest",
        )
        if (
            configuration_payload.get("schema")
            != "pareto_smc_algorithm_configuration_manifest_v2"
        ):
            raise ValueError(
                "algorithm-configuration manifest has the wrong schema."
            )
        if configuration_payload.get("suite_sha256") != suite_sha256:
            raise ValueError(
                "algorithm-configuration manifest suite hash mismatch."
            )
        configuration_rows = configuration_payload.get("runs")
        if not isinstance(configuration_rows, list):
            raise ValueError(
                "algorithm-configuration manifest runs must be an array."
            )
        expected_configuration_keys = {
            (case, algorithm, seed)
            for case in expected_case_ids
            for algorithm in algorithms
            for seed in seed_ids
        }
        observed_configuration_keys = set()
        parsed_configurations = []
        for row_index, raw_configuration in enumerate(
            configuration_rows
        ):
            configuration = _mapping(
                raw_configuration,
                f"algorithm configuration row {row_index}",
            )
            _exact_keys(
                configuration,
                {
                    "case",
                    "algorithm",
                    "seed",
                    "population",
                    "algorithm_configuration_sha256",
                    "search_evaluations",
                    "pilot_evaluations",
                    "confirm_evaluations",
                    "algorithm_configuration",
                },
                f"algorithm configuration row {row_index}",
            )
            case = configuration.get("case")
            algorithm = configuration.get("algorithm")
            seed = configuration.get("seed")
            if (
                not isinstance(case, str)
                or not isinstance(algorithm, str)
                or isinstance(seed, bool)
                or not isinstance(seed, int)
            ):
                raise ValueError(
                    "Algorithm configuration keys have invalid types."
                )
            key = (case, algorithm, seed)
            if key in observed_configuration_keys:
                raise ValueError(
                    f"Duplicate algorithm configuration row: {key}."
                )
            observed_configuration_keys.add(key)
            population_value = _positive_int(
                configuration.get("population"),
                f"algorithm configuration {key} population",
            )
            configuration_sha = _sha256(
                configuration.get(
                    "algorithm_configuration_sha256"
                ),
                f"algorithm configuration {key} SHA-256",
            )
            readable_configuration = _mapping(
                configuration.get("algorithm_configuration"),
                f"algorithm configuration {key} payload",
            )
            observed_configuration_sha = hashlib.sha256(
                json.dumps(
                    readable_configuration,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            if observed_configuration_sha != configuration_sha:
                raise ValueError(
                    f"Algorithm configuration {key} readable payload "
                    "does not match its SHA-256."
                )
            if (
                readable_configuration.get("case") != case
                or readable_configuration.get("algorithm") != algorithm
                or readable_configuration.get("seed") != seed
                or readable_configuration.get("population")
                != population_value
            ):
                raise ValueError(
                    f"Algorithm configuration {key} readable payload "
                    "identity does not match its manifest row."
                )
            if algorithm == anchor:
                _validate_pilot_confirm_anchor_configuration(
                    readable_configuration,
                    anchor=anchor,
                    key=key,
                )
            search_evaluations = _nonnegative_int(
                configuration.get("search_evaluations"),
                f"algorithm configuration {key} search_evaluations",
            )
            pilot_evaluations = _nonnegative_int(
                configuration.get("pilot_evaluations"),
                f"algorithm configuration {key} pilot_evaluations",
            )
            confirm_evaluations = _nonnegative_int(
                configuration.get("confirm_evaluations"),
                f"algorithm configuration {key} confirm_evaluations",
            )
            if (
                search_evaluations
                + pilot_evaluations
                + confirm_evaluations
                != evaluations
            ):
                raise ValueError(
                    f"Algorithm configuration {key} does not charge the "
                    "complete evaluation budget."
                )
            if algorithm == anchor:
                if (
                    search_evaluations != 0
                    or pilot_evaluations <= 0
                    or confirm_evaluations <= 0
                ):
                    raise ValueError(
                        "The anchor configuration must predeclare two charged "
                        "pilot-confirm streams."
                    )
            elif (
                search_evaluations != evaluations
                or pilot_evaluations != 0
                or confirm_evaluations != 0
            ):
                raise ValueError(
                    "Non-anchor configurations must charge the budget to "
                    "their search stream."
                )
            parsed_configurations.append(
                (
                    case,
                    algorithm,
                    seed,
                    population_value,
                    configuration_sha,
                    search_evaluations,
                    pilot_evaluations,
                    confirm_evaluations,
                )
            )
        if observed_configuration_keys != expected_configuration_keys:
            raise ValueError(
                "Algorithm-configuration manifest does not contain the exact "
                "case x algorithm x seed matrix."
            )
        expected_run_configurations = tuple(
            sorted(parsed_configurations)
        )

    return CompetitiveProtocol(
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        anchor=anchor,
        algorithms=algorithms,
        suite_path=suite_path,
        suite_sha256=suite_sha256,
        reference_manifest_path=reference_manifest_path,
        reference_manifest_sha256=reference_manifest_sha256,
        information_contract_path=information_contract_path,
        information_contract_sha256=information_contract_sha256,
        algorithm_configuration_manifest_path=(
            configuration_manifest_path
        ),
        algorithm_configuration_manifest_sha256=(
            configuration_manifest_sha256
        ),
        expected_run_configurations=expected_run_configurations,
        expected_cases=expected_cases,
        expected_case_ids=expected_case_ids,
        expected_case_cities=tuple(sorted(suite_case_cities.items())),
        expected_instance_sha256=tuple(
            sorted(suite_instance_hashes.items())
        ),
        expected_reference_sha256=(
            None
            if expected_reference_hashes is None
            else tuple(sorted(expected_reference_hashes.items()))
        ),
        seed_ids=seed_ids,
        minimum_cities=minimum_cities,
        required_city_sizes=required_sizes,
        evaluations_per_run=evaluations,
        budget_scope=budget_scope,
        archive_limit=archive_limit,
        runtime_measurement_contract=runtime_contract,
        execution_order_contract=execution_order_contract,
        memory_measurement_contract=memory_contract,
        memory_replay_order_contract=memory_replay_order_contract,
        output_objective_equivalence_contract=(
            output_equivalence_contract
        ),
        native_archive_completeness_contract=(
            native_archive_completeness_contract
        ),
        anytime_front_semantics=anytime_front_semantics,
        anytime_checkpoint_contract=anytime_checkpoint_contract,
        anytime_checkpoint_period=anytime_checkpoint_period,
        anytime_auc_integration_contract=(
            anytime_auc_integration_contract
        ),
        anytime_time_auc_status=anytime_time_auc_status,
        diagnostic_archive_limit_contract=(
            diagnostic_archive_limit_contract
        ),
        igd_plus_noninferiority_margin=igd_margin,
        maximum_runtime_ratio=_positive_ratio(
            gates.get("maximum_runtime_ratio"),
            "gates.maximum_runtime_ratio",
        ),
        maximum_python_memory_ratio=_positive_ratio(
            gates.get("maximum_python_memory_ratio"),
            "gates.maximum_python_memory_ratio",
        ),
        bootstrap_repetitions=bootstrap_repetitions,
        randomization_repetitions=randomization_repetitions,
        analysis_random_seed=analysis_random_seed_raw,
    )


def _not_run(
    aggregate_csv: Path,
    protocol: CompetitiveProtocol,
    reasons: Sequence[str],
) -> dict[str, object]:
    return {
        "schema": "pareto_smc_competitive_audit_v2",
        "aggregate_csv": str(aggregate_csv.resolve()),
        "protocol_path": str(protocol.path),
        "protocol_sha256": protocol.sha256,
        "suite_sha256": protocol.suite_sha256,
        "reference_manifest_sha256": (
            protocol.reference_manifest_sha256
        ),
        "information_contract_sha256": (
            protocol.information_contract_sha256
        ),
        "algorithm_configuration_manifest_sha256": (
            protocol.algorithm_configuration_manifest_sha256
        ),
        "evidence_status": "NOT_RUN",
        "contract_gate": "FAIL",
        "overall_adoption_verdict": "NOT_RUN",
        "competitive_claim_verdict": "NOT_RUN",
        "submission_verdict": "HOLD",
        "empirical_claim_scope": (
            f"Only the frozen {protocol.expected_cases}-case synthetic "
            "biobjective suite at city sizes "
            f"{list(protocol.required_city_sizes)}; no many-objective or "
            "general state-of-the-art inference."
        ),
        "reasons": tuple(reasons),
        "claim_limit": (
            "Missing or unmatched empirical evidence cannot support scalable "
            "or state-of-the-art claims."
        ),
    }


def _finite_float(row: Mapping[str, str], column: str) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Column {column!r} must contain finite numbers.") from error
    if not math.isfinite(value):
        raise ValueError(f"Column {column!r} must contain finite numbers.")
    return value


def _cost_ratio_summary(
    index: Mapping[tuple[str, str, int], Mapping[str, str]],
    *,
    cases: Sequence[str],
    seed_ids: Sequence[int],
    anchor: str,
    comparator: str,
    column: str,
    maximum_ratio: float,
    bootstrap_repetitions: int,
    randomization_repetitions: int,
    rng: random.Random,
) -> dict[str, object]:
    log_advantages = []
    by_case: dict[str, list[float]] = defaultdict(list)
    raw_ratios = []
    for case in cases:
        for seed in seed_ids:
            anchor_value = _finite_float(index[(case, anchor, seed)], column)
            comparator_value = _finite_float(index[(case, comparator, seed)], column)
            if anchor_value <= 0.0 or comparator_value <= 0.0:
                raise ValueError(f"{column} must be strictly positive for ratio analysis.")
            ratio = anchor_value / comparator_value
            advantage = math.log(comparator_value / anchor_value)
            raw_ratios.append(ratio)
            log_advantages.append(advantage)
            by_case[case].append(advantage)
    ci_low, ci_high = _cluster_bootstrap_ci(
        by_case,
        repetitions=bootstrap_repetitions,
        rng=random.Random(rng.randrange(1 << 63)),
    )
    p_value, p_method = _cluster_randomization_p(
        by_case,
        repetitions=randomization_repetitions,
        rng=random.Random(rng.randrange(1 << 63)),
    )
    threshold = -math.log(maximum_ratio)
    tolerance = 1e-12
    wins = sum(value > tolerance for value in log_advantages)
    losses = sum(value < -tolerance for value in log_advantages)
    ties = len(log_advantages) - wins - losses
    trimmed = _trimmed_mean(log_advantages)
    winsorized = _winsorized_mean(log_advantages)
    gate = bool(
        ci_low >= threshold - 1e-15
        and trimmed >= threshold - 1e-15
        and winsorized >= threshold - 1e-15
    )
    return {
        "comparator": comparator,
        "metric": column,
        "ratio_definition": "anchor_divided_by_comparator",
        "maximum_allowed_ratio": maximum_ratio,
        "geometric_mean_ratio": math.exp(
            sum(math.log(value) for value in raw_ratios) / len(raw_ratios)
        ),
        "median_ratio": median(raw_ratios),
        "log_ratio_advantage_mean": sum(log_advantages) / len(log_advantages),
        "log_ratio_cluster_bootstrap_ci95": [ci_low, ci_high],
        "log_ratio_noninferiority_threshold": threshold,
        "log_ratio_trimmed_mean_10pct": trimmed,
        "log_ratio_winsorized_mean_10pct": winsorized,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "cluster_randomization_p_two_sided": p_value,
        "cluster_randomization_method": p_method,
        "noninferiority_gate": "PASS" if gate else "FAIL",
    }


def audit_competitive_results(
    aggregate_csv: Path,
    protocol: CompetitiveProtocol,
    *,
    bootstrap_repetitions: int | None = None,
    randomization_repetitions: int | None = None,
    random_seed: int | None = None,
) -> dict[str, object]:
    """Validate contracts and compute paired quality/cost evidence."""

    requested_analysis = (
        protocol.bootstrap_repetitions
        if bootstrap_repetitions is None
        else bootstrap_repetitions,
        protocol.randomization_repetitions
        if randomization_repetitions is None
        else randomization_repetitions,
        protocol.analysis_random_seed
        if random_seed is None
        else random_seed,
    )
    frozen_analysis = (
        protocol.bootstrap_repetitions,
        protocol.randomization_repetitions,
        protocol.analysis_random_seed,
    )
    if requested_analysis != frozen_analysis:
        return _not_run(
            aggregate_csv,
            protocol,
            (
                "analysis repetitions or random seed differ from the "
                "predeclared protocol",
            ),
        )
    (
        resolved_bootstrap_repetitions,
        resolved_randomization_repetitions,
        resolved_random_seed,
    ) = frozen_analysis
    pending_contract_reasons = []
    if protocol.reference_manifest_sha256 is None:
        pending_contract_reasons.append(
            "frozen metric-reference manifest hash is not yet predeclared"
        )
    if protocol.algorithm_configuration_manifest_sha256 is None:
        pending_contract_reasons.append(
            "frozen algorithm-configuration manifest hash is not yet "
            "predeclared"
        )
    if pending_contract_reasons:
        return _not_run(
            aggregate_csv,
            protocol,
            tuple(pending_contract_reasons),
        )
    if not aggregate_csv.is_file():
        return _not_run(aggregate_csv, protocol, ("aggregate CSV is missing",))
    with aggregate_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        rows = list(reader)
    reasons = []
    required_columns = set(REQUIRED_COLUMNS)
    if protocol.anchor == V12_COMPETITIVE_ANCHOR:
        required_columns.add(
            PUBLICATION_CERTIFICATE_PACKET_GATE_COLUMN
        )
    missing_columns = sorted(required_columns - fieldnames)
    if missing_columns:
        reasons.append("missing columns: " + ", ".join(missing_columns))
    if not rows:
        reasons.append("aggregate CSV is empty")
    if reasons:
        return _not_run(aggregate_csv, protocol, reasons)

    integer_columns = {
        "seed": None,
        "population": 1,
        "num_cities": 1,
        "evaluations": 1,
        "search_evaluations": 0,
        "pilot_evaluations": 0,
        "confirm_evaluations": 0,
        "archive_size": 0,
        "archive_limit": 1,
        "max_diagnostic_archive_size": 0,
        "anytime_checkpoint_period": 1,
        "anytime_checkpoint_count": 1,
    }
    nonnegative_float_columns = (
        "case_relative_hypervolume_2d",
        "case_relative_anytime_hv_eval_auc",
        "igd_plus",
        "runtime_seconds",
        "python_peak_traced_memory_bytes",
        "output_objective_max_abs_error",
    )
    signed_float_columns = ("additive_epsilon",)
    for row_number, row in enumerate(rows, start=2):
        if not row["case"] or not row["algorithm"]:
            reasons.append(
                f"row {row_number} has an empty case or algorithm"
            )
        for column, minimum in integer_columns.items():
            try:
                parsed = int(row[column])
            except (TypeError, ValueError):
                reasons.append(
                    f"row {row_number} has invalid integer {column}"
                )
                continue
            if minimum is not None and parsed < minimum:
                reasons.append(
                    f"row {row_number} has {column} below {minimum}"
                )
        for column in nonnegative_float_columns:
            try:
                parsed_float = float(row[column])
            except (TypeError, ValueError):
                reasons.append(
                    f"row {row_number} has invalid numeric {column}"
                )
                continue
            if not math.isfinite(parsed_float) or parsed_float < 0.0:
                reasons.append(
                    f"row {row_number} has non-finite or negative {column}"
                )
            if (
                column
                in {
                    "runtime_seconds",
                    "python_peak_traced_memory_bytes",
                }
                and parsed_float <= 0.0
            ):
                reasons.append(
                    f"row {row_number} has non-positive {column}"
                )
            if (
                column
                in {
                    "case_relative_hypervolume_2d",
                    "case_relative_anytime_hv_eval_auc",
                }
                and parsed_float > 1.0 + 1e-12
            ):
                reasons.append(
                    f"row {row_number} has {column} above one"
                )
        for column in signed_float_columns:
            try:
                parsed_float = float(row[column])
            except (TypeError, ValueError):
                reasons.append(
                    f"row {row_number} has invalid numeric {column}"
                )
                continue
            if not math.isfinite(parsed_float):
                reasons.append(
                    f"row {row_number} has non-finite {column}"
                )
        for column in (
            "suite_sha256",
            "reference_manifest_sha256",
            "instance_sha256",
            "reference_sha256",
            "information_signature_sha256",
            "algorithm_configuration_sha256",
        ):
            try:
                _sha256(row[column], f"row {row_number} {column}")
            except ValueError as error:
                reasons.append(str(error))
        if row["suite_sha256"] != protocol.suite_sha256:
            reasons.append(f"row {row_number} suite hash mismatch")
        if (
            row["reference_manifest_sha256"]
            != protocol.reference_manifest_sha256
        ):
            reasons.append(
                f"row {row_number} reference-manifest hash mismatch"
            )
        if (
            row["information_signature_sha256"]
            != protocol.information_contract_sha256
        ):
            reasons.append(
                f"row {row_number} information-contract hash mismatch"
            )
        if protocol.anchor == V12_COMPETITIVE_ANCHOR:
            publication_gate = row[
                PUBLICATION_CERTIFICATE_PACKET_GATE_COLUMN
            ]
            if (
                row["algorithm"] == protocol.anchor
                and publication_gate != "PASS"
            ):
                reasons.append(
                    f"row {row_number} v12 anchor "
                    "publication_certificate_packet_gate must be PASS"
                )
            elif (
                row["algorithm"] != protocol.anchor
                and publication_gate != "NOT_APPLICABLE"
            ):
                reasons.append(
                    f"row {row_number} non-anchor "
                    "publication_certificate_packet_gate must be "
                    "NOT_APPLICABLE"
                )
    if reasons:
        return _not_run(
            aggregate_csv,
            protocol,
            tuple(dict.fromkeys(reasons)),
        )

    algorithms = tuple(sorted({row["algorithm"] for row in rows}))
    if algorithms != tuple(sorted(protocol.algorithms)):
        reasons.append(
            "algorithm set mismatch: "
            f"expected={sorted(protocol.algorithms)}, observed={list(algorithms)}"
        )
    cases = sorted({row["case"] for row in rows})
    if tuple(cases) != protocol.expected_case_ids:
        reasons.append(
            "case IDs mismatch: "
            f"expected={list(protocol.expected_case_ids)}, observed={cases}"
        )
    observed_seeds = tuple(sorted({int(row["seed"]) for row in rows}))
    if observed_seeds != protocol.seed_ids:
        reasons.append(
            f"seed IDs mismatch: expected={protocol.seed_ids}, observed={observed_seeds}"
        )

    index: dict[tuple[str, str, int], Mapping[str, str]] = {}
    for row in rows:
        key = (row["case"], row["algorithm"], int(row["seed"]))
        if key in index:
            reasons.append(f"duplicate matched row: {key}")
        index[key] = row
    for case in cases:
        for algorithm in protocol.algorithms:
            for seed in protocol.seed_ids:
                if (case, algorithm, seed) not in index:
                    reasons.append(
                        f"missing matched row: case={case}, algorithm={algorithm}, seed={seed}"
                    )

    observed_sizes = set()
    expected_cities = dict(protocol.expected_case_cities)
    expected_instances = dict(protocol.expected_instance_sha256)
    expected_references = (
        {}
        if protocol.expected_reference_sha256 is None
        else dict(protocol.expected_reference_sha256)
    )
    expected_configurations = {
        (case, algorithm, seed): (
            population,
            configuration_sha,
            search_evaluations,
            pilot_evaluations,
            confirm_evaluations,
        )
        for (
            case,
            algorithm,
            seed,
            population,
            configuration_sha,
            search_evaluations,
            pilot_evaluations,
            confirm_evaluations,
        ) in (protocol.expected_run_configurations or ())
    }
    for case in cases:
        case_rows = [row for row in rows if row["case"] == case]
        sizes = {int(row["num_cities"]) for row in case_rows}
        instances = {row["instance_sha256"] for row in case_rows}
        references = {row["reference_sha256"] for row in case_rows}
        reference_manifests = {
            row["reference_manifest_sha256"] for row in case_rows
        }
        information_signatures = {
            row["information_signature_sha256"] for row in case_rows
        }
        if len(sizes) != 1:
            reasons.append(f"case {case} has inconsistent num_cities")
            continue
        size = next(iter(sizes))
        observed_sizes.add(size)
        if expected_cities.get(case) != size:
            reasons.append(
                f"case {case} num_cities does not match the frozen suite"
            )
        if size < protocol.minimum_cities:
            reasons.append(
                f"case {case} has {size} cities below minimum {protocol.minimum_cities}"
            )
        if len(instances) != 1:
            reasons.append(f"case {case} has inconsistent instance_sha256")
        elif expected_instances.get(case) != next(iter(instances)):
            reasons.append(
                f"case {case} instance_sha256 does not match the frozen suite"
            )
        if len(references) != 1:
            reasons.append(f"case {case} has inconsistent reference_sha256")
        elif expected_references.get(case) != next(iter(references)):
            reasons.append(
                f"case {case} reference_sha256 does not match the frozen "
                "metric-reference manifest"
            )
        if len(reference_manifests) != 1:
            reasons.append(
                f"case {case} has inconsistent reference_manifest_sha256"
            )
        if len(information_signatures) != 1:
            reasons.append(
                f"case {case} has inconsistent information_signature_sha256"
            )
    missing_sizes = sorted(set(protocol.required_city_sizes) - observed_sizes)
    if missing_sizes:
        reasons.append(f"required city-size strata absent: {missing_sizes}")
    for case in cases:
        case_rows = [row for row in rows if row["case"] == case]
        for column in (
            "case_relative_hypervolume_2d",
            "case_relative_anytime_hv_eval_auc",
        ):
            maximum = max(float(row[column]) for row in case_rows)
            if maximum > 0.0 and not math.isclose(
                maximum,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                reasons.append(
                    f"case {case} {column} is not normalized to a maximum "
                    "of one"
                )

    for case in cases:
        for seed in protocol.seed_ids:
            pair = [
                index.get((case, algorithm, seed))
                for algorithm in protocol.algorithms
            ]
            if any(row is None for row in pair):
                continue
            matched_rows = [row for row in pair if row is not None]
            for column in ("reference_sha256", "information_signature_sha256"):
                if len({row[column] for row in matched_rows}) != 1:
                    reasons.append(
                        f"{column} mismatch for case={case}, seed={seed}"
                    )
            for row in matched_rows:
                expected_configuration = expected_configurations.get(
                    (case, row["algorithm"], seed)
                )
                observed_configuration = (
                    int(row["population"]),
                    row["algorithm_configuration_sha256"],
                    int(row["search_evaluations"]),
                    int(row["pilot_evaluations"]),
                    int(row["confirm_evaluations"]),
                )
                if expected_configuration != observed_configuration:
                    reasons.append(
                        f"algorithm configuration mismatch for case={case}, "
                        f"algorithm={row['algorithm']}, seed={seed}"
                    )
                if int(row["evaluations"]) != protocol.evaluations_per_run:
                    reasons.append(
                        f"evaluation budget mismatch for case={case}, "
                        f"algorithm={row['algorithm']}, seed={seed}"
                    )
                if row["budget_scope"] != protocol.budget_scope:
                    reasons.append(
                        f"budget scope mismatch for case={case}, "
                        f"algorithm={row['algorithm']}, seed={seed}"
                    )
                if int(row["archive_limit"]) != protocol.archive_limit:
                    reasons.append(
                        f"archive limit mismatch for case={case}, "
                        f"algorithm={row['algorithm']}, seed={seed}"
                    )
                if int(row["archive_size"]) > protocol.archive_limit:
                    reasons.append(
                        f"archive overflow for case={case}, "
                        f"algorithm={row['algorithm']}, seed={seed}"
                    )
                if (
                    int(row["max_diagnostic_archive_size"])
                    > protocol.archive_limit
                ):
                    reasons.append(
                        f"anytime archive overflow for case={case}, "
                        f"algorithm={row['algorithm']}, seed={seed}"
                    )
                if row["diagnostic_archive_limit_gate"] != "PASS":
                    reasons.append(
                        f"anytime archive cap failed for case={case}, "
                        f"algorithm={row['algorithm']}, seed={seed}"
                    )
                if (
                    row["diagnostic_archive_limit_contract"]
                    != protocol.diagnostic_archive_limit_contract
                ):
                    reasons.append(
                        f"anytime archive-cap contract mismatch for "
                        f"case={case}, algorithm={row['algorithm']}, "
                        f"seed={seed}"
                    )
                if (
                    row["anytime_front_semantics"]
                    != protocol.anytime_front_semantics
                ):
                    reasons.append(
                        f"anytime front semantics mismatch for case={case}, "
                        f"algorithm={row['algorithm']}, seed={seed}"
                    )
                if row["anytime_checkpoint_gate"] != "PASS":
                    reasons.append(
                        f"common anytime checkpoint gate failed for "
                        f"case={case}, algorithm={row['algorithm']}, "
                        f"seed={seed}"
                    )
                if (
                    row["anytime_checkpoint_contract"]
                    != protocol.anytime_checkpoint_contract
                ):
                    reasons.append(
                        f"anytime checkpoint contract mismatch for "
                        f"case={case}, algorithm={row['algorithm']}, "
                        f"seed={seed}"
                    )
                if (
                    int(row["anytime_checkpoint_period"])
                    != protocol.anytime_checkpoint_period
                ):
                    reasons.append(
                        f"anytime checkpoint period mismatch for "
                        f"case={case}, algorithm={row['algorithm']}, "
                        f"seed={seed}"
                    )
                expected_checkpoint_count = (
                    protocol.evaluations_per_run
                    + protocol.anytime_checkpoint_period
                    - 1
                ) // protocol.anytime_checkpoint_period
                if (
                    int(row["anytime_checkpoint_count"])
                    != expected_checkpoint_count
                ):
                    reasons.append(
                        f"anytime checkpoint count mismatch for "
                        f"case={case}, algorithm={row['algorithm']}, "
                        f"seed={seed}"
                    )
                if (
                    row["anytime_auc_integration_contract"]
                    != protocol.anytime_auc_integration_contract
                ):
                    reasons.append(
                        f"anytime AUC integration mismatch for case={case}, "
                        f"algorithm={row['algorithm']}, seed={seed}"
                    )
                if (
                    row["anytime_time_auc_status"]
                    != protocol.anytime_time_auc_status
                ):
                    reasons.append(
                        f"time-AUC status mismatch for case={case}, "
                        f"algorithm={row['algorithm']}, seed={seed}"
                    )
                if (
                    row["runtime_measurement_contract"]
                    != protocol.runtime_measurement_contract
                ):
                    reasons.append(
                        f"runtime contract mismatch for case={case}, "
                        f"algorithm={row['algorithm']}, seed={seed}"
                    )
                if (
                    row["execution_order_contract"]
                    != protocol.execution_order_contract
                ):
                    reasons.append(
                        f"execution-order contract mismatch for case={case}, "
                        f"algorithm={row['algorithm']}, seed={seed}"
                    )
                if (
                    row["memory_measurement_contract"]
                    != protocol.memory_measurement_contract
                ):
                    reasons.append(
                        f"memory contract mismatch for case={case}, "
                        f"algorithm={row['algorithm']}, seed={seed}"
                    )
                if (
                    row["memory_replay_order_contract"]
                    != protocol.memory_replay_order_contract
                ):
                    reasons.append(
                        f"memory-replay order mismatch for case={case}, "
                        f"algorithm={row['algorithm']}, seed={seed}"
                    )
                if (
                    row["memory_replay_state_equivalence_gate"]
                    != "PASS"
                ):
                    reasons.append(
                        f"memory replay did not reproduce the timed state "
                        f"for case={case}, algorithm={row['algorithm']}, "
                        f"seed={seed}"
                    )
                if row["output_objective_equivalence_gate"] != "PASS":
                    reasons.append(
                        f"final objective replay failed for case={case}, "
                        f"algorithm={row['algorithm']}, seed={seed}"
                    )
                if (
                    row["output_objective_equivalence_contract"]
                    != protocol.output_objective_equivalence_contract
                ):
                    reasons.append(
                        f"final objective replay contract mismatch for "
                        f"case={case}, algorithm={row['algorithm']}, "
                        f"seed={seed}"
                    )
                if _finite_float(
                    row,
                    "output_objective_max_abs_error",
                ) != 0.0:
                    reasons.append(
                        f"integer-suite objective replay was not exact for "
                        f"case={case}, algorithm={row['algorithm']}, "
                        f"seed={seed}"
                    )
                if row["anytime_objective_equivalence_gate"] != "PASS":
                    reasons.append(
                        f"anytime objective replay failed for case={case}, "
                        f"algorithm={row['algorithm']}, seed={seed}"
                    )
                if row["anytime_objective_equivalence_contract"] not in {
                    "internal_diagnostic_front_from_local_evaluations_v1",
                    "diagnostic_tour_local_full_recompute_"
                    "rel1e-12_abs1e-12_v1",
                    "diagnostic_tour_local_full_recompute_exact_"
                    "binary64_integer_v1",
                }:
                    reasons.append(
                        f"unknown anytime objective contract for case={case}, "
                        f"algorithm={row['algorithm']}, seed={seed}"
                    )
                if row["evaluation_evidence_gate"] != "PASS":
                    reasons.append(
                        f"evaluation evidence failed for case={case}, "
                        f"algorithm={row['algorithm']}, seed={seed}"
                    )
                if row["native_archive_completeness_gate"] != "PASS":
                    reasons.append(
                        f"native archive completeness failed for case={case}, "
                        f"algorithm={row['algorithm']}, seed={seed}"
                    )
                if (
                    row["native_archive_completeness_contract"]
                    != protocol.native_archive_completeness_contract
                ):
                    reasons.append(
                        f"native archive completeness contract mismatch for "
                        f"case={case}, algorithm={row['algorithm']}, "
                        f"seed={seed}"
                    )
                if row["evaluation_evidence_contract"] not in {
                    "inprocess_counting_instance_exact_budget_v1",
                    "every_final_row_exact_requested_budget_and_"
                    "ordered_anytime_steps_ending_at_final_v1",
                }:
                    reasons.append(
                        f"unknown evaluation evidence contract for "
                        f"case={case}, algorithm={row['algorithm']}, "
                        f"seed={seed}"
                    )
                if _finite_float(row, "python_peak_traced_memory_bytes") <= 0.0:
                    reasons.append(
                        f"memory measurement missing for case={case}, "
                        f"algorithm={row['algorithm']}, seed={seed}"
                    )
    if reasons:
        return _not_run(aggregate_csv, protocol, tuple(dict.fromkeys(reasons)))

    quality = analyze(
        aggregate_csv,
        anchor=protocol.anchor,
        expected_cases=protocol.expected_cases,
        expected_seeds=len(protocol.seed_ids),
        bootstrap_repetitions=resolved_bootstrap_repetitions,
        randomization_repetitions=resolved_randomization_repetitions,
        random_seed=resolved_random_seed,
        igd_noninferiority_margin=protocol.igd_plus_noninferiority_margin,
    )
    rng = random.Random(resolved_random_seed ^ 0x5EEDC0DE)
    cost_results = []
    for comparator in protocol.algorithms:
        if comparator == protocol.anchor:
            continue
        cost_results.append(
            _cost_ratio_summary(
                index,
                cases=cases,
                seed_ids=protocol.seed_ids,
                anchor=protocol.anchor,
                comparator=comparator,
                column="runtime_seconds",
                maximum_ratio=protocol.maximum_runtime_ratio,
                bootstrap_repetitions=resolved_bootstrap_repetitions,
                randomization_repetitions=resolved_randomization_repetitions,
                rng=rng,
            )
        )
        cost_results.append(
            _cost_ratio_summary(
                index,
                cases=cases,
                seed_ids=protocol.seed_ids,
                anchor=protocol.anchor,
                comparator=comparator,
                column="python_peak_traced_memory_bytes",
                maximum_ratio=protocol.maximum_python_memory_ratio,
                bootstrap_repetitions=resolved_bootstrap_repetitions,
                randomization_repetitions=resolved_randomization_repetitions,
                rng=rng,
            )
        )
    quality_gate = quality["overall_adoption_verdict"] == "ADOPT"
    cost_gate = all(
        item["noninferiority_gate"] == "PASS" for item in cost_results
    )
    adopted = bool(quality_gate and cost_gate)
    return {
        "schema": "pareto_smc_competitive_audit_v2",
        "aggregate_csv": str(aggregate_csv.resolve()),
        "protocol_path": str(protocol.path),
        "protocol_sha256": protocol.sha256,
        "suite_sha256": protocol.suite_sha256,
        "reference_manifest_sha256": (
            protocol.reference_manifest_sha256
        ),
        "information_contract_sha256": (
            protocol.information_contract_sha256
        ),
        "algorithm_configuration_manifest_sha256": (
            protocol.algorithm_configuration_manifest_sha256
        ),
        "evidence_status": "COMPLETE",
        "contract_gate": "PASS",
        "quality_analysis": quality,
        "cost_ratio_results": cost_results,
        "quality_gate": "PASS" if quality_gate else "FAIL",
        "cost_noninferiority_gate": "PASS" if cost_gate else "FAIL",
        "overall_adoption_verdict": "ADOPT" if adopted else "REJECT",
        "within_contract_empirical_verdict": (
            "ADOPT" if adopted else "REJECT"
        ),
        "competitive_claim_verdict": "HOLD",
        "submission_verdict": "HOLD",
        "empirical_claim_scope": (
            f"Only the frozen {protocol.expected_cases}-case synthetic "
            "biobjective suite at city sizes "
            f"{list(protocol.required_city_sizes)}; no many-objective or "
            "general state-of-the-art inference."
        ),
        "submission_hold_reason": (
            "The competitive gate cannot discharge the independent theorem, "
            "joint-design, many-objective, literature-review, and external-"
            "review gates."
        ),
        "memory_scope_limit": (
            "The configured metric covers Python-traced allocations only "
            "and external child-process/native allocations are outside that "
            "scope."
        ),
        "runtime_scope_limit": (
            "In-process and external-process arms do not share an equivalent "
            "serialization, startup, or cache boundary; the cost gate is only "
            "descriptive within this frozen adapter contract."
        ),
        "claim_limit": (
            "This audit evaluates matched empirical competition only; it does "
            "not strengthen the finite-particle theorem or certify the unknown "
            "Pareto front."
        ),
    }


def _markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Pareto-SMC competitive audit",
        "",
        f"Evidence: **{report['evidence_status']}**",
        f"Contract: **{report['contract_gate']}**",
        f"Adoption: **{report['overall_adoption_verdict']}**",
        f"Submission: **{report['submission_verdict']}**",
        "",
    ]
    if report["evidence_status"] != "COMPLETE":
        lines.extend(["## Missing evidence", ""])
        for reason in report.get("reasons", ()):
            lines.append(f"- {reason}")
        lines.extend(["", str(report["claim_limit"]), ""])
        return "\n".join(lines)
    lines.extend(
        [
            "## Cost non-inferiority",
            "",
            "| comparator | metric | geometric ratio | CI95 log advantage | "
            "trim10 | winsor10 | W/T/L | gate |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["cost_ratio_results"]:  # type: ignore[index]
        low, high = item["log_ratio_cluster_bootstrap_ci95"]
        lines.append(
            "| {comparator} | {metric} | {geometric_mean_ratio:.6g} | "
            "[{low:.6g}, {high:.6g}] | {log_ratio_trimmed_mean_10pct:.6g} | "
            "{log_ratio_winsorized_mean_10pct:.6g} | "
            "{wins}/{ties}/{losses} | {noninferiority_gate} |".format(
                low=low,
                high=high,
                **item,
            )
        )
    lines.extend(
        [
            "",
            str(report["memory_scope_limit"]),
            "",
            str(report["claim_limit"]),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=None,
        help="Optional assertion; must equal the frozen protocol value.",
    )
    parser.add_argument(
        "--randomization-repetitions",
        type=int,
        default=None,
        help="Optional assertion; must equal the frozen protocol value.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional assertion; must equal the frozen protocol value.",
    )
    args = parser.parse_args()
    protocol = load_competitive_protocol(args.protocol)
    report = audit_competitive_results(
        args.aggregate,
        protocol,
        bootstrap_repetitions=args.bootstrap_repetitions,
        randomization_repetitions=args.randomization_repetitions,
        random_seed=args.random_seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(f"EVIDENCE {report['evidence_status']}")
    print(f"ADOPTION {report['overall_adoption_verdict']}")
    print(f"SUBMISSION {report['submission_verdict']}")
    return 0 if report["overall_adoption_verdict"] == "ADOPT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
