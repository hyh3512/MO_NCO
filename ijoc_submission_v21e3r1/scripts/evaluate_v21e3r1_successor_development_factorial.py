from __future__ import annotations

"""Evaluate the frozen V21e3r1 successor development factorial.

The generic selection/confirmation evaluator is intentionally not reused: its
balanced C0--C3 design cannot encode this workflow's four-arm MOKP factorial,
two-arm MOTSP contrast, or cache-hit-rate hypothesis.  This standard-library
implementation uses the same frozen one-sided observed-SE max-t paired-case
cluster bootstrap, but it grants development-promotion status only and claims
no implementation, producer, custody, scientific, or publication independence.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Mapping, Sequence

from ijoc_submission_v21e3r1.scripts import (
    run_v21e3r1_successor_development_factorial as runner,
)


EVALUATION_SCHEMA = "v21e3r1_successor_development_factorial_evaluation_receipt_v2"
PROMOTION_SCOPE = (
    "SUCCESSOR_DEVELOPMENT_PROMOTION_ONLY_HASH_BOUND_PRODUCER_RECEIPT_"
    "NO_PROSPECTIVE_108_ROW_RECOMPUTATION_NO_SCIENTIFIC_CLAIM"
)
METHOD = runner.METHOD
FAMILIES = ("MOKP", "MOTSP")
HYPOTHESES = (
    "MOKP:BOTH_MINUS_LEGACY:EAUC",
    "MOKP:ANCHOR_MAIN_EFFECT:EAUC",
    "MOKP:NOVELTY_MAIN_EFFECT:EAUC",
    "MOKP:NOVELTY_MAIN_EFFECT:CACHE_HIT_RATE_REDUCTION",
    "MOTSP:ANCHOR_MINUS_LEGACY:EAUC",
)
CELL_CONTRACTS = (
    ("MOKP", "exact_per_evaluation_left_continuous_hv_auc", "SUPERIORITY", 0.005),
    ("MOKP", "exact_per_evaluation_left_continuous_hv_auc", "SUPERIORITY", 0.0),
    ("MOKP", "exact_per_evaluation_left_continuous_hv_auc", "NONINFERIORITY", -0.005),
    ("MOKP", "cache_hit_rate_per_attempt", "SUPERIORITY", 0.1),
    ("MOTSP", "exact_per_evaluation_left_continuous_hv_auc", "SUPERIORITY", 0.0),
)
EVALUATION_KEYS = frozenset(
    {
        "schema", "status", "phase", "promotion_scope", "study_id",
        "candidate_id", "successor_source_sha256", "successor_config_sha256",
        "source_freeze_receipt_sha256", "source_manifest_sha256",
        "study_metric_spec_sha256", "simultaneous_inference_spec_sha256",
        "matrix_directory", "matrix_plan_sha256", "matrix_receipt_sha256",
        "row_evidence_replay_sha256", "inference_spec_sha256", "method",
        "familywise_alpha", "bootstrap_samples", "bootstrap_seed", "rng_protocol",
        "rng_domain", "quantile_convention", "cluster_unit", "seed_aggregation",
        "familywise_scope", "critical_value", "bootstrap_maxima_sha256",
        "matrix_row_count", "expected_matrix_row_count", "hypothesis_order", "cells",
        "development_promotion_gate_passed", "gate_reasons",
        "zero_standard_error_hypotheses", "selection_confirmation_evaluator_reused",
        "selection_confirmation_evaluator_reuse_reason",
        "selection_cases_materialized", "confirmation_cases_materialized",
        "formal_cases_materialized", "algorithm_execution_independence",
        "statistics_implementation_independence", "producer_independence",
        "custody_independence", "scientific_independence", "selection_authorized",
        "confirmation_authorized", "formal_study_authorized",
        "scientific_claim_authorized", "ijoc_submission_status",
        "receipt_payload_sha256",
    }
)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise runner.ContractError("cannot average an empty sequence")
    try:
        result = math.fsum(values) / len(values)
    except OverflowError as error:
        raise runner.ContractError("numeric aggregation overflowed") from error
    if not math.isfinite(result):
        raise runner.ContractError("numeric aggregation produced a non-finite value")
    return result


def _sample_standard_error(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = _mean(values)
    variance = math.fsum((value - center) ** 2 for value in values) / (len(values) - 1)
    result = math.sqrt(variance / len(values))
    if not math.isfinite(result):
        raise runner.ContractError("standard error is non-finite")
    return result


class _Sha256CounterRng:
    def __init__(self, seed: int, domain: str) -> None:
        self._key = hashlib.sha256(
            runner._canonical_bytes({"domain": domain, "seed": seed})
        ).digest()
        self._counter = 0

    def _next_u64(self) -> int:
        block = hashlib.sha256(
            self._key + self._counter.to_bytes(16, "big")
        ).digest()
        self._counter += 1
        return int.from_bytes(block[:8], "big")

    def randbelow(self, bound: int) -> int:
        if type(bound) is not int or bound <= 0:
            raise ValueError("bound must be an exact positive integer")
        modulus = 1 << 64
        limit = modulus - modulus % bound
        while True:
            value = self._next_u64()
            if value < limit:
                return value % bound


MATRIX_RECEIPT_KEYS = runner.FACTORIAL_RECEIPT_KEYS


def _derive_replayed_row_metrics(
    row: Mapping[str, object],
    terminal: Mapping[str, object],
    independent: Mapping[str, object],
    *,
    row_id: str,
) -> dict[str, float | int]:
    attempts = runner._exact_int(
        terminal.get("attempt_count"), f"{row_id}.terminal.attempt_count", minimum=1
    )
    cache_hits = runner._exact_int(
        terminal.get("cache_hit_count"), f"{row_id}.terminal.cache_hit_count"
    )
    charges = runner._exact_int(
        terminal.get("charged_evaluation_count"),
        f"{row_id}.terminal.charged_evaluation_count",
        minimum=1,
    )
    if charges != runner.FULL_BUDGET or cache_hits > attempts:
        raise runner.ContractError(f"terminal accounting drifted: {row_id}")
    cache_rate = cache_hits / attempts
    if (
        row.get("attempt_count") != attempts
        or type(row.get("attempt_count")) is not int
        or row.get("cache_hit_count") != cache_hits
        or type(row.get("cache_hit_count")) is not int
        or row.get("charged_evaluation_count") != charges
        or type(row.get("charged_evaluation_count")) is not int
        or row.get("cache_hit_rate_per_attempt") != cache_rate
        or type(row.get("cache_hit_rate_per_attempt")) is not float
    ):
        raise runner.ContractError(f"row cache metric disagrees with terminal replay: {row_id}")
    eauc = runner._exact_number(
        independent.get("exact_left_continuous_hv_auc"),
        f"{row_id}.independent exact EAUC",
    )
    terminal_hv = runner._exact_number(
        independent.get("terminal_hv"), f"{row_id}.independent terminal HV"
    )
    if (
        row.get("exact_per_evaluation_left_continuous_hv_auc") != eauc
        or type(row.get("exact_per_evaluation_left_continuous_hv_auc")) is not float
    ):
        raise runner.ContractError(f"row EAUC disagrees with independent replay: {row_id}")
    row_terminal = runner._exact_number(
        row.get("normalized_terminal_hv"), f"{row_id}.row terminal HV"
    )
    if not math.isclose(row_terminal, terminal_hv, rel_tol=0.0, abs_tol=1e-12):
        raise runner.ContractError(f"row terminal HV disagrees with independent replay: {row_id}")
    if not 0.0 <= eauc <= 1.0 or not 0.0 <= terminal_hv <= 1.0:
        raise runner.ContractError(f"replayed metric is outside [0,1]: {row_id}")
    return {
        "attempt_count": attempts,
        "cache_hit_count": cache_hits,
        "charged_evaluation_count": charges,
        "cache_hit_rate_per_attempt": cache_rate,
        "exact_per_evaluation_left_continuous_hv_auc": eauc,
        "normalized_terminal_hv": terminal_hv,
    }


TERMINAL_RECEIPT_KEYS = frozenset(
    {
        "attempt_count", "cache_hit_count", "charged_evaluation_count",
        "database_path", "decision_count", "durability_mode", "failure_code",
        "failure_detail", "family", "finalization_gates",
        "physical_call_started_count", "problem", "receipt_payload_sha256",
        "run_context_digest_sha256", "schema", "status",
        "terminal_attempt_chain_sha256", "terminal_decision_chain_sha256",
        "terminal_evaluation_chain_sha256", "unresolved_decision_count",
    }
)
TERMINAL_GATE_KEYS = frozenset(
    {
        "cache_hits", "evaluation_index_bounds", "expected_charged_evaluations",
        "expected_decisions", "expected_evaluation_index_bounds",
        "nonterminal_attempts", "persisted_attempts", "persisted_decisions",
        "persisted_evaluations", "physical_call_starts",
        "run_context_charged_evaluation_budget", "sqlite_integrity",
    }
)


def _validate_terminal_receipt(
    path: Path, row_spec: Mapping[str, object]
) -> dict[str, object]:
    row_id = str(row_spec["row_id"])
    terminal = runner._require_keys(
        runner._load_json(path), TERMINAL_RECEIPT_KEYS, f"terminal receipt {row_id}"
    )
    core = dict(terminal)
    payload_sha = runner._sha_text(
        core.pop("receipt_payload_sha256"), f"{row_id}.terminal payload hash"
    )
    if runner._payload_sha256(core) != payload_sha:
        raise runner.ContractError(f"terminal receipt payload hash drifted: {row_id}")
    expected = {
        "schema": "v21e3_terminal_receipt_v1",
        "status": "SUCCESS",
        "database_path": "trace.sqlite3",
        "family": row_spec["family"],
        "problem": row_spec["case_id"],
        "charged_evaluation_count": runner.FULL_BUDGET,
        "decision_count": runner.FULL_BUDGET,
        "failure_code": None,
        "failure_detail": None,
        "unresolved_decision_count": 0,
    }
    for key, expected_value in expected.items():
        if terminal[key] != expected_value or type(terminal[key]) is not type(expected_value):
            raise runner.ContractError(f"terminal receipt field drifted: {row_id}/{key}")
    attempts = runner._exact_int(
        terminal["attempt_count"], f"{row_id}.terminal attempts", minimum=1
    )
    cache_hits = runner._exact_int(
        terminal["cache_hit_count"], f"{row_id}.terminal cache hits"
    )
    physical = runner._exact_int(
        terminal["physical_call_started_count"],
        f"{row_id}.terminal physical starts",
        minimum=1,
    )
    if physical != runner.FULL_BUDGET or attempts != physical + cache_hits:
        raise runner.ContractError(f"terminal receipt accounting identity drifted: {row_id}")
    for field in (
        "run_context_digest_sha256", "terminal_attempt_chain_sha256",
        "terminal_decision_chain_sha256", "terminal_evaluation_chain_sha256",
    ):
        runner._sha_text(terminal[field], f"{row_id}.terminal.{field}")
    gates = runner._require_keys(
        terminal["finalization_gates"], TERMINAL_GATE_KEYS, f"terminal gates {row_id}"
    )
    expected_gates = {
        "cache_hits": cache_hits,
        "evaluation_index_bounds": [1, runner.FULL_BUDGET],
        "expected_charged_evaluations": runner.FULL_BUDGET,
        "expected_decisions": runner.FULL_BUDGET,
        "expected_evaluation_index_bounds": [1, runner.FULL_BUDGET],
        "nonterminal_attempts": 0,
        "persisted_attempts": attempts,
        "persisted_decisions": runner.FULL_BUDGET,
        "persisted_evaluations": runner.FULL_BUDGET,
        "physical_call_starts": runner.FULL_BUDGET,
        "run_context_charged_evaluation_budget": runner.FULL_BUDGET,
        "sqlite_integrity": "ok",
    }
    for key, expected_value in expected_gates.items():
        if gates[key] != expected_value or type(gates[key]) is not type(expected_value):
            raise runner.ContractError(f"terminal finalization gate drifted: {row_id}/{key}")
    return terminal


def _replay_row_evidence(
    project: Path,
    attempt: Path,
    row_spec: Mapping[str, object],
    row: Mapping[str, object],
    source_binding: Mapping[str, object],
    lower: Sequence[float],
    upper: Sequence[float],
    expected_algorithm_config: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    row_id = str(row_spec["row_id"])
    trace = attempt / "trace.sqlite3"
    terminal_path = attempt / "terminal.receipt.json"
    independent_path = attempt / "independent.metric.json"
    case_path = runner._contained(
        project, str(row_spec["case_artifact_path"]), f"factorial case {row_id}"
    )
    if runner._sha256(case_path) != row_spec["case_artifact_sha256"]:
        raise runner.ContractError(f"factorial case artifact drifted: {row_id}")
    problem = runner.v7_runner.load_v21e3_development_problem(case_path)
    terminal = _validate_terminal_receipt(terminal_path, row_spec)
    context = runner.v7_runner._load_trace_context(trace)
    runner.v7_runner._assert_context(
        context,
        algorithm_config=expected_algorithm_config,
        case_sha256=str(row_spec["case_artifact_sha256"]),
        source_sha256=str(source_binding["source_snapshot_sha256"]),
        seed=runner._exact_int(row_spec["seed"], f"{row_id}.seed"),
        budget=runner.FULL_BUDGET,
    )
    terminal_sha = runner._sha256(terminal_path)
    verification = runner.verify_v21e3_trace_database(
        trace,
        problem,
        expected_run_context=context,
        detached_terminal_receipt_path=terminal_path,
        expected_detached_terminal_receipt_sha256=terminal_sha,
        expected_charged_evaluations=runner.FULL_BUDGET,
    )
    if verification != row["strict_trace_verification"]:
        raise runner.ContractError(f"stored/fresh strict trace verification drifted: {row_id}")
    stored_independent = runner._load_json(independent_path)
    with tempfile.TemporaryDirectory(
        prefix="v21e3r1-successor-factorial-metric-replay-"
    ) as temporary:
        fresh_independent = runner._successor_independent_metric_replay(
            project_root=project,
            trace=trace,
            lower=tuple(float(item) for item in lower),
            upper=tuple(float(item) for item in upper),
            budget=runner.FULL_BUDGET,
            output=Path(temporary) / "independent.metric.json",
        )
    if stored_independent != fresh_independent:
        raise runner.ContractError(f"stored/fresh independent metric replay drifted: {row_id}")
    metrics = _derive_replayed_row_metrics(
        row, terminal, fresh_independent, row_id=row_id
    )
    sanitized = dict(row)
    sanitized.update(metrics)
    witness = {
        "row_id": row_id,
        "trace_sha256": runner._sha256(trace),
        "terminal_receipt_sha256": terminal_sha,
        "strict_trace_verification_sha256": runner._payload_sha256(verification),
        "independent_metric_replay_sha256": runner._payload_sha256(fresh_independent),
        "exact_per_evaluation_left_continuous_hv_auc": metrics[
            "exact_per_evaluation_left_continuous_hv_auc"
        ],
        "cache_hit_rate_per_attempt": metrics["cache_hit_rate_per_attempt"],
    }
    return sanitized, witness


def _validate_matrix_receipt(
    matrix: Path, plan: Mapping[str, object], plan_sha: str
) -> dict[str, object]:
    receipt_path = matrix / "factorial.receipt.json"
    aggregate_path = matrix / "factorial.aggregate.json"
    if not aggregate_path.is_file():
        raise runner.ContractError("factorial aggregate is absent")
    aggregate_sha = runner._sha256(aggregate_path)
    receipt = runner._validate_factorial_receipt_payload(
        runner._load_json(receipt_path), plan, plan_sha, aggregate_sha
    )
    if receipt_path.read_bytes() != runner._canonical_bytes(receipt, newline=True):
        raise runner.ContractError("factorial receipt is not canonical JSON plus LF")
    if receipt["aggregate_sha256"] != aggregate_sha:
        raise runner.ContractError("factorial aggregate hash drifted")
    aggregate = runner._require_keys(
        runner._load_json(aggregate_path),
        runner.FACTORIAL_AGGREGATE_KEYS,
        "factorial aggregate",
    )
    if (
        aggregate["schema"] != runner.AGGREGATE_SCHEMA
        or aggregate["status"] != receipt["status"]
        or aggregate["phase"] != "development"
        or aggregate["scientific_scope"]
        != "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE"
        or aggregate["plan_sha256"] != plan_sha
        or aggregate["row_count"] != runner.FULL_ROW_COUNT
        or type(aggregate["row_count"]) is not int
        or type(aggregate["rows"]) is not list
        or len(aggregate["rows"]) != runner.FULL_ROW_COUNT
        or aggregate["development_promotion_evaluated"] is not False
        or aggregate["selection_authorized"] is not False
        or aggregate["confirmation_authorized"] is not False
        or aggregate["formal_study_authorized"] is not False
        or aggregate["scientific_claim_authorized"] is not False
        or aggregate["ijoc_submission_status"] != "IJOC_HOLD"
    ):
        raise runner.ContractError("factorial aggregate identity/cardinality drifted")
    if aggregate_path.read_bytes() != runner._canonical_bytes(aggregate, newline=True):
        raise runner.ContractError("factorial aggregate is not canonical JSON plus LF")
    for index, summary in enumerate(aggregate["rows"]):
        runner._require_keys(
            summary, runner.FACTORIAL_SUMMARY_KEYS, f"factorial aggregate row[{index}]"
        )
    return receipt


def load_matrix_rows(
    project_root: str | Path, matrix_directory: str | Path
) -> tuple[dict[str, object], list[dict[str, object]], str, str, str]:
    root = Path(project_root).resolve()
    matrix = Path(matrix_directory).resolve()
    try:
        matrix.relative_to(root)
    except ValueError as error:
        raise runner.ContractError("factorial matrix escapes the project root") from error
    plan_path = matrix / "factorial.plan.json"
    plan = runner.validate_plan_payload(runner._load_json(plan_path))
    if plan_path.read_bytes() != runner._canonical_bytes(plan, newline=True):
        raise runner.ContractError("factorial plan is not canonical JSON plus LF")
    parent_path = runner._contained(
        root,
        str(plan["parent_v7_diagnostic_plan_path"]),
        "factorial parent V7 plan",
    )
    runner.validate_sealed_parent_diagnostic(root, parent_path)
    source_receipt_path = runner._contained(
        root,
        str(plan["source_binding"]["receipt_path"]),
        "factorial successor source receipt",
    )
    current_source = runner.validate_successor_source_freeze(
        root, source_receipt_path
    )
    if current_source != plan["source_binding"]:
        raise runner.ContractError("factorial plan/live successor source binding drifted")
    expected_plan = runner.build_plan_payload(
        root,
        parent_path,
        current_source,
        row_timeout_seconds=int(plan["row_timeout_seconds"]),
    )
    if plan != expected_plan:
        raise runner.ContractError("factorial plan disagrees with re-derived frozen design")
    plan_sha = runner._sha256(plan_path)
    inference, inference_sha = runner.load_inference_spec(root)
    if (
        plan["inference_spec_binding"]["sha256"] != inference_sha
        or plan["inference_spec_binding"]["method"] != inference["method"]
    ):
        raise runner.ContractError("factorial plan/inference spec binding drifted")
    receipt = _validate_matrix_receipt(matrix, plan, plan_sha)
    strict_receipt = runner._verify_completed_receipt(root, matrix, plan, plan_sha)
    if strict_receipt != receipt:
        raise runner.ContractError("factorial receipt validators disagree")
    aggregate = runner._load_json(matrix / "factorial.aggregate.json")
    aggregate_by_id = {str(item.get("row_id")): item for item in aggregate["rows"]}
    if len(aggregate_by_id) != runner.FULL_ROW_COUNT:
        raise runner.ContractError("factorial aggregate row IDs are not unique")
    cases, bounds, directions, input_binding = runner.v7_runner._load_inputs(root)
    if input_binding != plan["input_binding"]:
        raise runner.ContractError("factorial evaluator input binding drifted")
    if tuple(str(case["case_id"]) for case in cases) != tuple(plan["case_ids"]):
        raise runner.ContractError("factorial evaluator case order drifted")
    rows: list[dict[str, object]] = []
    replay_witnesses: list[dict[str, object]] = []
    for row_spec in plan["rows"]:
        row_id = str(row_spec["row_id"])
        expected_config = runner._expected_semantic_config(row_spec, directions)
        completed = runner._completed_payload(
            matrix, row_spec, plan_sha, current_source, expected_config
        )
        if completed is None:
            raise runner.ContractError(f"factorial row is absent/unsealed: {row_id}")
        attempt = runner._contained(
            matrix, str(completed["attempt_directory"]), "factorial completed attempt"
        )
        row = runner._validate_row_payload(
            runner._load_json(attempt / "row.json"),
            row_spec,
            plan_sha,
            current_source,
            expected_config,
        )
        lower, upper = bounds[row_id.split("__seed-", 1)[0]]
        row, witness = _replay_row_evidence(
            root,
            attempt,
            row_spec,
            row,
            current_source,
            lower,
            upper,
            expected_config,
        )
        summary = aggregate_by_id.get(row_id)
        if type(summary) is not dict:
            raise runner.ContractError(f"factorial aggregate row binding drifted: {row_id}")
        for field in (
            "ordinal", "row_id", "case_id", "family", "seed", "arm_id",
            "exact_per_evaluation_left_continuous_hv_auc", "cache_hit_rate_per_attempt",
        ):
            if summary.get(field) != row[field] or type(summary.get(field)) is not type(row[field]):
                raise runner.ContractError(f"factorial aggregate value drifted: {row_id}/{field}")
        for field in (
            "row_sha256", "trace_sha256", "terminal_receipt_sha256",
            "independent_metric_receipt_sha256",
        ):
            if summary.get(field) != completed[field]:
                raise runner.ContractError(f"factorial aggregate artifact drifted: {row_id}/{field}")
        rows.append(row)
        replay_witnesses.append(witness)
    if runner.validate_successor_source_freeze(root, source_receipt_path) != current_source:
        raise runner.ContractError("successor source changed during factorial evaluation")
    runner.validate_sealed_parent_diagnostic(root, parent_path)
    return (
        plan,
        rows,
        plan_sha,
        runner._sha256(matrix / "factorial.receipt.json"),
        runner._payload_sha256(replay_witnesses),
    )


def _case_arm_means(
    plan: Mapping[str, object], rows: Sequence[Mapping[str, object]], metric: str
) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, int, str], float] = {}
    for row in rows:
        key = (str(row["case_id"]), int(row["seed"]), str(row["arm_id"]))
        if key in values:
            raise runner.ContractError(f"duplicate factorial row: {key}")
        values[key] = runner._exact_number(row[metric], f"row metric {metric}")
    result: dict[tuple[str, str], float] = {}
    for case_id in plan["case_ids"]:
        family = "MOKP" if "-mokp-" in str(case_id) else "MOTSP"
        arms = runner.MOKP_ARMS if family == "MOKP" else runner.MOTSP_ARMS
        for arm_id, _search, _novelty in arms:
            observations = []
            for seed in runner.SEEDS:
                key = (str(case_id), seed, arm_id)
                if key not in values:
                    raise runner.ContractError(f"missing factorial row: {key}")
                observations.append(values[key])
            result[(str(case_id), arm_id)] = _mean(observations)
    expected = runner.FULL_ROW_COUNT
    if len(values) != expected:
        raise runner.ContractError("factorial row coverage is not exactly 108")
    return result


def build_hypothesis_cells(
    plan: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    inference: Mapping[str, object],
) -> list[dict[str, object]]:
    eauc = _case_arm_means(
        plan, rows, "exact_per_evaluation_left_continuous_hv_auc"
    )
    cache = _case_arm_means(plan, rows, "cache_hit_rate_per_attempt")
    cases = {
        family: [
            str(case_id)
            for case_id in plan["case_ids"]
            if ("-mokp-" in str(case_id)) == (family == "MOKP")
        ]
        for family in FAMILIES
    }
    if any(len(cases[family]) != 6 for family in FAMILIES):
        raise runner.ContractError("each factorial family must contain exactly six cases")

    formulas: dict[str, tuple[str, str, list[float]]] = {}
    mokp = cases["MOKP"]
    formulas[HYPOTHESES[0]] = (
        "MOKP", "exact_per_evaluation_left_continuous_hv_auc",
        [eauc[(case, "MOKP_BOTH")] - eauc[(case, "MOKP_LEGACY")] for case in mokp],
    )
    formulas[HYPOTHESES[1]] = (
        "MOKP", "exact_per_evaluation_left_continuous_hv_auc",
        [
            0.5 * (
                (eauc[(case, "MOKP_ANCHOR_ONLY")] - eauc[(case, "MOKP_LEGACY")])
                + (eauc[(case, "MOKP_BOTH")] - eauc[(case, "MOKP_NOVELTY_ONLY")])
            )
            for case in mokp
        ],
    )
    formulas[HYPOTHESES[2]] = (
        "MOKP", "exact_per_evaluation_left_continuous_hv_auc",
        [
            0.5 * (
                (eauc[(case, "MOKP_NOVELTY_ONLY")] - eauc[(case, "MOKP_LEGACY")])
                + (eauc[(case, "MOKP_BOTH")] - eauc[(case, "MOKP_ANCHOR_ONLY")])
            )
            for case in mokp
        ],
    )
    formulas[HYPOTHESES[3]] = (
        "MOKP", "cache_hit_rate_per_attempt",
        [
            0.5 * (
                (cache[(case, "MOKP_LEGACY")] - cache[(case, "MOKP_NOVELTY_ONLY")])
                + (cache[(case, "MOKP_ANCHOR_ONLY")] - cache[(case, "MOKP_BOTH")])
            )
            for case in mokp
        ],
    )
    formulas[HYPOTHESES[4]] = (
        "MOTSP", "exact_per_evaluation_left_continuous_hv_auc",
        [
            eauc[(case, "MOTSP_ANCHOR")] - eauc[(case, "MOTSP_LEGACY")]
            for case in cases["MOTSP"]
        ],
    )
    thresholds = {
        str(item["hypothesis_id"]): float(item["threshold"])
        for item in inference["hypotheses"]
    }
    roles = {
        str(item["hypothesis_id"]): str(item["role"])
        for item in inference["hypotheses"]
    }
    cells: list[dict[str, object]] = []
    for hypothesis_id in HYPOTHESES:
        family, metric, effects = formulas[hypothesis_id]
        threshold = thresholds[hypothesis_id]
        cells.append(
            {
                "hypothesis_id": hypothesis_id,
                "family": family,
                "metric": metric,
                "role": roles[hypothesis_id],
                "threshold": threshold,
                "case_count": len(effects),
                "seed_count_per_case_arm": len(runner.SEEDS),
                "case_effects": effects,
                "observed_mean": _mean(effects),
                "standard_error": _sample_standard_error(effects),
                "median": statistics.median(effects),
                "wins_above_threshold": sum(value > threshold for value in effects),
                "ties_at_threshold": sum(value == threshold for value in effects),
                "losses_below_threshold": sum(value < threshold for value in effects),
            }
        )
    return cells


def apply_simultaneous_bounds(
    cells: list[dict[str, object]], inference: Mapping[str, object]
) -> tuple[float | None, str | None, list[str]]:
    zero_se = [
        str(cell["hypothesis_id"])
        for cell in cells
        if float(cell["standard_error"]) <= 0.0
        or not math.isfinite(float(cell["standard_error"]))
    ]
    if zero_se:
        for cell in cells:
            cell["simultaneous_lower_bound"] = None
            cell["gate_passed"] = False
        return None, None, zero_se
    samples = runner._exact_int(inference["bootstrap_samples"], "bootstrap samples", minimum=1)
    alpha = runner._exact_number(inference["familywise_alpha"], "familywise alpha")
    seed = runner._exact_int(inference["bootstrap_seed"], "bootstrap seed")
    rng = _Sha256CounterRng(seed, str(inference["rng_domain"]))
    by_family = {
        family: [cell for cell in cells if cell["family"] == family]
        for family in FAMILIES
    }
    maxima: list[float] = []
    for _ in range(samples):
        statistics_values: list[float] = []
        for family in FAMILIES:
            family_cells = by_family[family]
            case_count = int(family_cells[0]["case_count"])
            sampled = [rng.randbelow(case_count) for _ in range(case_count)]
            for cell in family_cells:
                effects = cell["case_effects"]
                bootstrap_mean = _mean([float(effects[index]) for index in sampled])
                centered_t = (
                    bootstrap_mean - float(cell["observed_mean"])
                ) / float(cell["standard_error"])
                if not math.isfinite(centered_t):
                    raise runner.ContractError("bootstrap t statistic is non-finite")
                statistics_values.append(centered_t)
        maxima.append(max(statistics_values))
    rank = math.ceil((1.0 - alpha) * (samples + 1))
    if not 1 <= rank <= samples:
        raise runner.ContractError("frozen bootstrap quantile rank is invalid")
    raw = sorted(maxima)[rank - 1]
    critical = max(0.0, raw)
    for cell in cells:
        lower = float(cell["observed_mean"]) - critical * float(cell["standard_error"])
        if not math.isfinite(lower):
            raise runner.ContractError("simultaneous lower bound is non-finite")
        cell["simultaneous_lower_bound"] = lower
        cell["gate_passed"] = lower > float(cell["threshold"])
    maxima_hash = hashlib.sha256(runner._canonical_bytes(maxima)).hexdigest()
    return critical, maxima_hash, []


def evaluate_rows(
    plan: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    inference: Mapping[str, object],
) -> dict[str, object]:
    runner.validate_plan_payload(plan)
    inference = runner.validate_inference_payload(inference)
    cells = build_hypothesis_cells(plan, rows, inference)
    critical, maxima_hash, zero_se = apply_simultaneous_bounds(cells, inference)
    if zero_se:
        status = "HOLD_SUCCESSOR_DEVELOPMENT_PROMOTION_ZERO_STANDARD_ERROR"
        passed = False
        reasons = [f"zero_standard_error:{item}" for item in zero_se]
    else:
        failed = [
            str(cell["hypothesis_id"]) for cell in cells if cell["gate_passed"] is not True
        ]
        passed = not failed
        status = (
            "PASS_SUCCESSOR_DEVELOPMENT_PROMOTION_ONLY"
            if passed
            else "HOLD_SUCCESSOR_DEVELOPMENT_PROMOTION_THRESHOLDS_NOT_MET"
        )
        reasons = [f"simultaneous_lower_bound_not_above_threshold:{item}" for item in failed]
    public_cells = [
        {key: value for key, value in cell.items() if key != "case_effects"}
        for cell in cells
    ]
    return {
        "status": status,
        "development_promotion_gate_passed": passed,
        "gate_reasons": reasons,
        "zero_standard_error_hypotheses": zero_se,
        "cells": public_cells,
        "critical_value": critical,
        "bootstrap_maxima_sha256": maxima_hash,
    }


EVALUATION_CELL_KEYS = frozenset(
    {
        "hypothesis_id", "family", "metric", "role", "threshold", "case_count",
        "seed_count_per_case_arm", "observed_mean", "standard_error", "median",
        "wins_above_threshold", "ties_at_threshold", "losses_below_threshold",
        "simultaneous_lower_bound", "gate_passed",
    }
)


def validate_evaluation_receipt_payload(value: object) -> dict[str, object]:
    receipt = runner._require_keys(value, EVALUATION_KEYS, "factorial evaluation receipt")
    core = dict(receipt)
    payload_sha = runner._sha_text(
        core.pop("receipt_payload_sha256"), "factorial evaluation payload SHA-256"
    )
    if runner._payload_sha256(core) != payload_sha:
        raise runner.ContractError("factorial evaluation receipt payload hash drifted")
    expected_scalars = {
        "schema": EVALUATION_SCHEMA,
        "phase": "development",
        "promotion_scope": PROMOTION_SCOPE,
        "method": METHOD,
        "familywise_alpha": 0.05,
        "bootstrap_samples": 9999,
        "bootstrap_seed": 2026082301,
        "rng_protocol": "SHA256_COUNTER_U64_REJECTION_V1",
        "rng_domain": "v21e3r1-successor-development-factorial-bootstrap-v1",
        "quantile_convention": "CEIL((1-ALPHA)*(B+1))_ORDER_STATISTIC",
        "cluster_unit": "PAIRED_CASE",
        "seed_aggregation": "MEAN_WITHIN_CASE_ARM",
        "familywise_scope": "JOINT_ACROSS_ALL_FIVE_DEVELOPMENT_PROMOTION_HYPOTHESES",
        "matrix_row_count": runner.FULL_ROW_COUNT,
        "expected_matrix_row_count": runner.FULL_ROW_COUNT,
        "hypothesis_order": list(HYPOTHESES),
        "selection_confirmation_evaluator_reused": False,
        "selection_confirmation_evaluator_reuse_reason": (
            "INCOMPATIBLE_ASYMMETRIC_4_ARM_MOKP_2_ARM_MOTSP_AND_MIXED_METRICS"
        ),
        "selection_cases_materialized": False,
        "confirmation_cases_materialized": False,
        "formal_cases_materialized": False,
        "algorithm_execution_independence": False,
        "statistics_implementation_independence": False,
        "producer_independence": False,
        "custody_independence": False,
        "scientific_independence": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_study_authorized": False,
        "scientific_claim_authorized": False,
        "ijoc_submission_status": "IJOC_HOLD",
        "inference_spec_sha256": runner.INFERENCE_SPEC_SHA256,
    }
    for key, expected_value in expected_scalars.items():
        if receipt[key] != expected_value or type(receipt[key]) is not type(expected_value):
            raise runner.ContractError(f"factorial evaluation field drifted: {key}")
    for field in ("study_id", "candidate_id"):
        if type(receipt[field]) is not str or not receipt[field]:
            raise runner.ContractError(f"factorial evaluation {field} is invalid")
    runner._relative_path(receipt["matrix_directory"], "factorial evaluation matrix directory")
    for field in (
        "successor_source_sha256", "successor_config_sha256",
        "source_freeze_receipt_sha256", "source_manifest_sha256",
        "study_metric_spec_sha256", "simultaneous_inference_spec_sha256",
        "matrix_plan_sha256", "matrix_receipt_sha256", "row_evidence_replay_sha256",
        "inference_spec_sha256",
    ):
        runner._sha_text(receipt[field], f"factorial evaluation.{field}")
    statuses = {
        "PASS_SUCCESSOR_DEVELOPMENT_PROMOTION_ONLY": True,
        "HOLD_SUCCESSOR_DEVELOPMENT_PROMOTION_THRESHOLDS_NOT_MET": False,
        "HOLD_SUCCESSOR_DEVELOPMENT_PROMOTION_ZERO_STANDARD_ERROR": False,
    }
    if receipt["status"] not in statuses:
        raise runner.ContractError("factorial evaluation status drifted")
    if (
        type(receipt["development_promotion_gate_passed"]) is not bool
        or receipt["development_promotion_gate_passed"]
        is not statuses[str(receipt["status"])]
    ):
        raise runner.ContractError("factorial evaluation status/gate relationship drifted")
    cells = receipt["cells"]
    if type(cells) is not list or len(cells) != len(HYPOTHESES):
        raise runner.ContractError("factorial evaluation cell cardinality drifted")
    for index, cell in enumerate(cells):
        raw = runner._require_keys(
            cell, EVALUATION_CELL_KEYS, f"factorial evaluation cells[{index}]"
        )
        family, metric, role, threshold = CELL_CONTRACTS[index]
        expected_cell_identity = {
            "hypothesis_id": HYPOTHESES[index],
            "family": family,
            "metric": metric,
            "role": role,
            "threshold": threshold,
            "case_count": 6,
            "seed_count_per_case_arm": len(runner.SEEDS),
        }
        for field, expected_value in expected_cell_identity.items():
            if raw[field] != expected_value or type(raw[field]) is not type(expected_value):
                raise runner.ContractError(
                    f"factorial evaluation cell identity drifted: {HYPOTHESES[index]}/{field}"
                )
        if type(raw["gate_passed"]) is not bool:
            raise runner.ContractError("factorial evaluation cell gate must be exact boolean")
        for field in ("observed_mean", "standard_error", "median"):
            if type(raw[field]) is not float or not math.isfinite(raw[field]):
                raise runner.ContractError(
                    f"factorial evaluation cell {field} must be an exact finite float"
                )
        if raw["standard_error"] < 0.0:
            raise runner.ContractError("factorial evaluation standard error is negative")
        for field in (
            "wins_above_threshold", "ties_at_threshold", "losses_below_threshold"
        ):
            runner._exact_int(raw[field], f"factorial evaluation cell {field}")
        if (
            raw["wins_above_threshold"]
            + raw["ties_at_threshold"]
            + raw["losses_below_threshold"]
            != raw["case_count"]
        ):
            raise runner.ContractError("factorial evaluation W/T/L cardinality drifted")
    zero = receipt["zero_standard_error_hypotheses"]
    reasons = receipt["gate_reasons"]
    if (
        type(zero) is not list
        or any(item not in HYPOTHESES for item in zero)
        or len(set(zero)) != len(zero)
        or type(reasons) is not list
        or any(type(item) is not str or not item for item in reasons)
    ):
        raise runner.ContractError("factorial evaluation HOLD reason payload drifted")
    expected_zero = [
        str(cell["hypothesis_id"])
        for cell in cells
        if cell["standard_error"] == 0.0
    ]
    if zero != expected_zero:
        raise runner.ContractError("factorial evaluation zero-SE witness drifted")
    if zero:
        expected_reasons = [f"zero_standard_error:{item}" for item in zero]
        if (
            receipt["status"] != "HOLD_SUCCESSOR_DEVELOPMENT_PROMOTION_ZERO_STANDARD_ERROR"
            or receipt["critical_value"] is not None
            or receipt["bootstrap_maxima_sha256"] is not None
            or any(cell["simultaneous_lower_bound"] is not None for cell in cells)
            or any(cell["gate_passed"] is not False for cell in cells)
            or reasons != expected_reasons
        ):
            raise runner.ContractError("factorial zero-SE HOLD payload drifted")
    else:
        if (
            type(receipt["critical_value"]) is not float
            or not math.isfinite(receipt["critical_value"])
            or receipt["critical_value"] < 0.0
        ):
            raise runner.ContractError("factorial critical value must be a nonnegative float")
        runner._sha_text(
            receipt["bootstrap_maxima_sha256"], "factorial bootstrap maxima SHA-256"
        )
        failed: list[str] = []
        for cell in cells:
            lower = cell["simultaneous_lower_bound"]
            if type(lower) is not float or not math.isfinite(lower):
                raise runner.ContractError(
                    "factorial simultaneous lower bound must be an exact finite float"
                )
            expected_lower = (
                cell["observed_mean"]
                - receipt["critical_value"] * cell["standard_error"]
            )
            if lower != expected_lower:
                raise runner.ContractError("factorial simultaneous lower bound drifted")
            expected_gate = lower > cell["threshold"]
            if cell["gate_passed"] is not expected_gate:
                raise runner.ContractError("factorial cell threshold decision drifted")
            if not expected_gate:
                failed.append(str(cell["hypothesis_id"]))
        expected_status = (
            "PASS_SUCCESSOR_DEVELOPMENT_PROMOTION_ONLY"
            if not failed
            else "HOLD_SUCCESSOR_DEVELOPMENT_PROMOTION_THRESHOLDS_NOT_MET"
        )
        expected_reasons = [
            f"simultaneous_lower_bound_not_above_threshold:{item}" for item in failed
        ]
        if receipt["status"] != expected_status or reasons != expected_reasons:
            raise runner.ContractError("factorial statistical status/reasons drifted")
    return receipt


def evaluate_matrix(
    project_root: str | Path,
    matrix_directory: str | Path,
    output_path: str | Path,
) -> tuple[dict[str, object], int]:
    root = Path(project_root).resolve()
    matrix = Path(matrix_directory).resolve()
    output = Path(output_path).resolve()
    try:
        output.relative_to(root)
    except ValueError as error:
        raise runner.ContractError("evaluation output escapes the project root") from error
    if output.exists():
        raise runner.ContractError("evaluation output already exists; exclusive create required")
    if not output.parent.is_dir():
        raise runner.ContractError("evaluation output parent directory must already exist")
    plan, rows, plan_sha, matrix_receipt_sha, row_evidence_replay_sha = load_matrix_rows(
        root, matrix
    )
    inference, inference_sha = runner.load_inference_spec(root)
    result = evaluate_rows(plan, rows, inference)
    source = runner._validate_source_binding(plan["source_binding"])
    if inference_sha != source["factorial_inference_spec_sha256"]:
        raise runner.ContractError("evaluation/source-freeze factorial inference binding drifted")
    core = {
        "schema": EVALUATION_SCHEMA,
        "status": result["status"],
        "phase": "development",
        "promotion_scope": PROMOTION_SCOPE,
        "study_id": source["study_id"],
        "candidate_id": source["candidate_id"],
        "successor_source_sha256": source["source_snapshot_sha256"],
        "successor_config_sha256": source["semantic_config_sha256"],
        "source_freeze_receipt_sha256": source["receipt_sha256"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "study_metric_spec_sha256": source["study_metric_spec_sha256"],
        "simultaneous_inference_spec_sha256": source[
            "simultaneous_inference_spec_sha256"
        ],
        "matrix_directory": matrix.relative_to(root).as_posix(),
        "matrix_plan_sha256": plan_sha,
        "matrix_receipt_sha256": matrix_receipt_sha,
        "row_evidence_replay_sha256": row_evidence_replay_sha,
        "inference_spec_sha256": inference_sha,
        "method": METHOD,
        "familywise_alpha": inference["familywise_alpha"],
        "bootstrap_samples": inference["bootstrap_samples"],
        "bootstrap_seed": inference["bootstrap_seed"],
        "rng_protocol": inference["rng_protocol"],
        "rng_domain": inference["rng_domain"],
        "quantile_convention": inference["quantile_convention"],
        "cluster_unit": inference["cluster_unit"],
        "seed_aggregation": inference["seed_aggregation"],
        "familywise_scope": inference["familywise_scope"],
        "critical_value": result["critical_value"],
        "bootstrap_maxima_sha256": result["bootstrap_maxima_sha256"],
        "matrix_row_count": len(rows),
        "expected_matrix_row_count": runner.FULL_ROW_COUNT,
        "hypothesis_order": list(HYPOTHESES),
        "cells": result["cells"],
        "development_promotion_gate_passed": result["development_promotion_gate_passed"],
        "gate_reasons": result["gate_reasons"],
        "zero_standard_error_hypotheses": result["zero_standard_error_hypotheses"],
        "selection_confirmation_evaluator_reused": False,
        "selection_confirmation_evaluator_reuse_reason": inference["selection_confirmation_evaluator_reuse"],
        "selection_cases_materialized": False,
        "confirmation_cases_materialized": False,
        "formal_cases_materialized": False,
        "algorithm_execution_independence": False,
        "statistics_implementation_independence": False,
        "producer_independence": False,
        "custody_independence": False,
        "scientific_independence": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_study_authorized": False,
        "scientific_claim_authorized": False,
        "ijoc_submission_status": "IJOC_HOLD",
    }
    receipt = {**core, "receipt_payload_sha256": runner._payload_sha256(core)}
    validate_evaluation_receipt_payload(receipt)
    runner._exclusive_json(output, receipt)
    return receipt, 0 if result["development_promotion_gate_passed"] else 2


def _materialize_integrity_hold_receipt(
    project_root: str | Path,
    matrix_directory: str | Path,
    output_path: str | Path,
    error: BaseException,
) -> dict[str, object] | None:
    root = Path(project_root).resolve()
    output = Path(output_path).resolve()
    matrix = Path(matrix_directory).resolve()
    try:
        output.relative_to(root)
        matrix_relative = matrix.relative_to(root).as_posix()
    except ValueError:
        return None
    if output.exists() or not output.parent.is_dir():
        return None
    core = {
        "schema": (
            "v21e3r1_successor_development_factorial_evaluation_integrity_hold_v1"
        ),
        "status": "HOLD_INTEGRITY_ERROR",
        "phase": "development",
        "promotion_scope": PROMOTION_SCOPE,
        "matrix_directory": matrix_relative,
        "error": str(error),
        "integrity_bindings_validated": False,
        "development_promotion_gate_passed": False,
        "selection_cases_materialized": False,
        "confirmation_cases_materialized": False,
        "formal_cases_materialized": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_study_authorized": False,
        "scientific_claim_authorized": False,
        "ijoc_submission_status": "IJOC_HOLD",
    }
    receipt = {**core, "receipt_payload_sha256": runner._payload_sha256(core)}
    runner._exclusive_json(output, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--matrix-directory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        receipt, exit_code = evaluate_matrix(
            args.project_root, args.matrix_directory, args.output
        )
        print(json.dumps(receipt, sort_keys=True))
        return exit_code
    except (
        runner.ContractError,
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    ) as error:
        hold_error: Exception | None = None
        try:
            hold = _materialize_integrity_hold_receipt(
                args.project_root, args.matrix_directory, args.output, error
            )
        except Exception as materialization_error:
            hold = None
            hold_error = materialization_error
        print(
            json.dumps(
                hold
                if hold is not None
                else {
                    "schema": "v21e3r1_successor_factorial_evaluator_error_v1",
                    "status": "HOLD_INTEGRITY_ERROR",
                    "error": str(error),
                    "receipt_materialization_error": (
                        None if hold_error is None else str(hold_error)
                    ),
                    "selection_authorized": False,
                    "formal_study_authorized": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
