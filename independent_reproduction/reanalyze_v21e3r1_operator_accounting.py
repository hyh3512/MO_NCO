#!/usr/bin/env python3
from __future__ import annotations

"""Read-only correction of V21e3r1 development operator accounting.

The original diagnostic analyzer added each charged evaluation once from the
evaluation ledger and once again while merging attempt-ledger counts.  This
standalone standard-library implementation preserves both quantities under
different names and never imports or executes project algorithm code.

This is an engineering/accounting reanalysis of exposed development traces.
It is not an independent algorithm reproduction and grants no scientific,
selection, confirmation, formal-study, or publication authority.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from typing import Iterable, Mapping, Sequence


PLAN_NAME = "diagnostic.plan.json"
DIAGNOSTIC_RECEIPT_NAME = "diagnostic.receipt.json"
DIAGNOSTIC_AGGREGATE_NAME = "diagnostic.aggregate.json"
ROWS_NAME = "operator_accounting.rows.jsonl"
AGGREGATE_NAME = "operator_accounting.aggregate.json"
RECEIPT_NAME = "operator_accounting.reanalysis.receipt.json"
METRIC_SPEC_SOURCE_NAME = "v21e3r1_operator_accounting_reanalysis_spec_v1.json"
METRIC_SPEC_OUTPUT_PATH = f"metric/{METRIC_SPEC_SOURCE_NAME}"
METRIC_SOURCE_OUTPUT_PATH = "source/reanalyze_v21e3r1_operator_accounting.py"

PLAN_SCHEMA = "v21e3r1_exposed_development_diagnostic_plan_v2"
DIAGNOSTIC_RECEIPT_SCHEMA = "v21e3r1_exposed_development_diagnostic_receipt_v2"
ROW_SCHEMA = "v21e3r1_exposed_development_diagnostic_row_v2"
LEGACY_DIAGNOSTIC_SCHEMA = "v21e3r1_existing_trace_diagnostic_v1"
OUTPUT_ROW_SCHEMA = "v21e3r1_corrected_operator_accounting_row_v1"
OUTPUT_AGGREGATE_SCHEMA = "v21e3r1_corrected_operator_accounting_aggregate_v1"
OUTPUT_RECEIPT_SCHEMA = (
    "v21e3r1_corrected_operator_accounting_reanalysis_receipt_v1"
)
METRIC_SPEC_SCHEMA = "v21e3r1_operator_accounting_reanalysis_metric_spec_v1"
SCOPE = "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE"
PRIMARY_DEFINITION = (
    "operator_witness.retry_ordinal == 0 and "
    "operator_witness.fallback_used == false"
)

FULL_CASE_IDS = (
    "v21e3-mokp-development-n100-s00",
    "v21e3-mokp-development-n100-s01",
    "v21e3-mokp-development-n200-s00",
    "v21e3-mokp-development-n200-s01",
    "v21e3-mokp-development-n500-s00",
    "v21e3-mokp-development-n500-s01",
    "v21e3-motsp-development-n100-s00",
    "v21e3-motsp-development-n100-s01",
    "v21e3-motsp-development-n200-s00",
    "v21e3-motsp-development-n200-s01",
    "v21e3-motsp-development-n500-s00",
    "v21e3-motsp-development-n500-s01",
)
FULL_SEEDS = (31051, 31057, 31059)
FULL_ARMS = (
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
FULL_ROWS = 504
FULL_BUDGET = 2000
FULL_CHECKPOINT_PERIOD = 200

RECEIPT_KEYS = frozenset(
    {
        "schema",
        "status",
        "scientific_scope",
        "matrix_mode",
        "completed_rows",
        "expected_rows",
        "charged_evaluations_per_row",
        "evaluation_charged_evaluations_sum",
        "attempt_charged_evaluations_sum",
        "legacy_operator_charged_evaluations_sum",
        "operator_charge_double_count_corrected",
        "all_rows_reanalyzed",
        "original_artifacts_modified",
        "plan_sha256",
        "diagnostic_receipt_sha256",
        "diagnostic_aggregate_sha256",
        "development_source_sha256",
        "rows_path",
        "rows_sha256",
        "aggregate_path",
        "aggregate_sha256",
        "metric_spec_path",
        "metric_spec_sha256",
        "metric_source_path",
        "metric_source_sha256",
        "implementation_independence",
        "algorithm_execution_independence",
        "scientific_independence",
        "selection_authorized",
        "confirmation_authorized",
        "formal_study_authorized",
        "publication_status",
    }
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_TABLES = {
    "attempts",
    "decisions",
    "evaluations",
    "run_attempt",
    "solutions",
    "terminal_receipts",
}


class ContractError(RuntimeError):
    """Raised when any sealed-input or accounting contract fails."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ContractError("payload is not finite canonical JSON") from error


def _reject_constant(value: str) -> object:
    raise ContractError(f"non-finite JSON constant is forbidden: {value}")


def _object_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _loads(raw: bytes | str, label: str) -> object:
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return json.loads(
            text,
            object_pairs_hook=_object_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} is not strict UTF-8 JSON") from error


def _load_object(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    if not path.is_file():
        raise ContractError(f"missing {label}: {path.name}")
    raw = path.read_bytes()
    value = _loads(raw, label)
    if type(value) is not dict:
        raise ContractError(f"{label} must be a JSON object")
    return value, raw


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ContractError(f"{label} must be an exact JSON boolean")
    return value


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{label} must be an exact integer >= {minimum}")
    return value


def _exact_number(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ContractError(f"{label} must be a finite exact JSON number")
    return float(value)


def _string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ContractError(f"{label} must be a nonempty exact string")
    return value


def _sha256_value(value: object, label: str) -> str:
    text = _string(value, label)
    if _HEX64.fullmatch(text) is None:
        raise ContractError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise ContractError(f"{label} must be a nonempty JSON array")
    result = tuple(_string(item, f"{label}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise ContractError(f"{label} contains duplicates")
    return result


def _int_list(value: object, label: str) -> tuple[int, ...]:
    if type(value) is not list or not value:
        raise ContractError(f"{label} must be a nonempty JSON array")
    result = tuple(_exact_int(item, f"{label}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise ContractError(f"{label} contains duplicates")
    return result


def _relative_file(root: Path, value: object, label: str) -> Path:
    text = _string(value, label)
    relative = Path(text)
    if relative.is_absolute():
        raise ContractError(f"{label} must be relative")
    resolved = (root / relative).resolve()
    if root != resolved and root not in resolved.parents:
        raise ContractError(f"{label} escapes its sealed root")
    if not resolved.is_file():
        raise ContractError(f"{label} does not name a file")
    return resolved


def _family_for_case(case_id: str) -> str:
    if case_id.startswith("v21e3-mokp-"):
        return "MOKP"
    if case_id.startswith("v21e3-motsp-"):
        return "MOTSP"
    raise ContractError(f"unsupported frozen case family: {case_id}")


def _validate_metric_spec(path: Path) -> tuple[bytes, str]:
    spec, raw = _load_object(path, "metric specification")
    canonical = _canonical_bytes(spec)
    if raw not in (canonical, canonical + b"\n"):
        raise ContractError(
            "metric specification must be compact canonical JSON with at most "
            "one trailing newline"
        )
    if (
        spec.get("schema") != METRIC_SPEC_SCHEMA
        or spec.get("status") != "FROZEN_BEFORE_REANALYSIS"
        or spec.get("scope") != "EXPOSED_DEVELOPMENT_DIAGNOSTIC_ACCOUNTING_ONLY"
        or type(spec.get("definitions")) is not dict
        or type(spec.get("invariants")) is not list
    ):
        raise ContractError("metric specification contract drifted")
    authority = spec.get("authority")
    if type(authority) is not dict or authority != {
        "algorithm_execution_independence": False,
        "confirmation_authorized": False,
        "formal_study_authorized": False,
        "implementation_independence": False,
        "publication_status": "IJOC_HOLD",
        "scientific_independence": False,
        "selection_authorized": False,
    }:
        raise ContractError("metric specification authority boundary drifted")
    return raw, _sha256_bytes(raw)


def _validate_source_manifest(plan: Mapping[str, object]) -> str:
    manifest = plan.get("source_manifest")
    if type(manifest) is not dict:
        raise ContractError("plan.source_manifest must be an object")
    if (
        manifest.get("schema") != "v21e3r1_diagnostic_source_manifest_v1"
        or manifest.get("hash_rule")
        != "sha256(canonical_json(sorted_entries))"
    ):
        raise ContractError("diagnostic source-manifest contract drifted")
    entries = manifest.get("entries")
    if type(entries) is not list or not entries:
        raise ContractError("diagnostic source manifest has no entries")
    if _exact_int(manifest.get("entry_count"), "source_manifest.entry_count", minimum=1) != len(entries):
        raise ContractError("diagnostic source manifest entry count disagrees")
    paths: list[str] = []
    for index, entry in enumerate(entries):
        if type(entry) is not dict:
            raise ContractError(f"source entry {index} must be an object")
        paths.append(_string(entry.get("path"), f"source entry {index}.path"))
        _exact_int(entry.get("bytes"), f"source entry {index}.bytes")
        _sha256_value(entry.get("sha256"), f"source entry {index}.sha256")
    if paths != sorted(paths, key=str.lower) or len(set(paths)) != len(paths):
        raise ContractError("diagnostic source entries are not uniquely sorted")
    observed = _sha256_bytes(_canonical_bytes(entries))
    claimed = _sha256_value(
        manifest.get("source_snapshot_sha256"),
        "source_manifest.source_snapshot_sha256",
    )
    if observed != claimed:
        raise ContractError("diagnostic source-root SHA-256 binding failed")
    return claimed


def _validate_plan(
    root: Path, *, allow_smoke: bool
) -> tuple[dict[str, object], bytes, tuple[str, ...], tuple[int, ...], tuple[str, ...], int, str, str]:
    plan, raw = _load_object(root / PLAN_NAME, PLAN_NAME)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("scientific_scope") != SCOPE:
        raise ContractError("diagnostic plan schema or scientific scope drifted")
    for field in (
        "selection_entropy_release",
        "confirmation_materialization",
        "formal_materialization",
    ):
        if plan.get(field) != "PROHIBITED":
            raise ContractError(f"diagnostic plan {field} is not PROHIBITED")
    case_ids = _string_list(plan.get("case_ids"), "plan.case_ids")
    seeds = _int_list(plan.get("seeds"), "plan.seeds")
    arms = _string_list(plan.get("arms"), "plan.arms")
    budget = _exact_int(
        plan.get("charged_evaluation_budget"),
        "plan.charged_evaluation_budget",
        minimum=1,
    )
    checkpoint = _exact_int(
        plan.get("checkpoint_period"), "plan.checkpoint_period", minimum=1
    )
    expected = _exact_int(plan.get("expected_rows"), "plan.expected_rows", minimum=1)
    if expected != len(case_ids) * len(seeds) * len(arms):
        raise ContractError("diagnostic plan row cardinality disagrees with its design")
    source_sha = _validate_source_manifest(plan)
    full = (
        case_ids == FULL_CASE_IDS
        and seeds == FULL_SEEDS
        and arms == FULL_ARMS
        and budget == FULL_BUDGET
        and checkpoint == FULL_CHECKPOINT_PERIOD
        and expected == FULL_ROWS
        and plan.get("status") == "FROZEN_FULL_504_DEVELOPMENT_DIAGNOSTIC"
    )
    if not full:
        if not allow_smoke:
            raise ContractError("input is not the exact frozen 504 diagnostic design")
        if plan.get("status") != "FROZEN_DIAGNOSTIC_SMOKE_ONLY":
            raise ContractError("non-full input is not a declared diagnostic smoke plan")
    return plan, raw, case_ids, seeds, arms, budget, source_sha, ("FULL_504" if full else "SMOKE_ONLY")


def _validate_final_receipt(
    root: Path,
    *,
    plan_sha: str,
    source_sha: str,
    expected_rows: int,
    matrix_mode: str,
) -> tuple[dict[str, object], bytes, str]:
    receipt, raw = _load_object(root / DIAGNOSTIC_RECEIPT_NAME, DIAGNOSTIC_RECEIPT_NAME)
    aggregate_path = root / DIAGNOSTIC_AGGREGATE_NAME
    if not aggregate_path.is_file():
        raise ContractError(f"missing {DIAGNOSTIC_AGGREGATE_NAME}")
    aggregate_sha = _sha256_file(aggregate_path)
    expected_status = (
        "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY"
        if matrix_mode == "FULL_504"
        else "PASS_DIAGNOSTIC_SMOKE_ONLY"
    )
    if (
        receipt.get("schema") != DIAGNOSTIC_RECEIPT_SCHEMA
        or receipt.get("status") != expected_status
        or receipt.get("scientific_scope") != SCOPE
        or receipt.get("matrix_mode") != matrix_mode
        or _exact_int(receipt.get("completed_rows"), "receipt.completed_rows", minimum=1)
        != expected_rows
        or _exact_int(receipt.get("expected_rows"), "receipt.expected_rows", minimum=1)
        != expected_rows
        or receipt.get("plan_sha256") != plan_sha
        or receipt.get("source_snapshot_sha256") != source_sha
        or receipt.get("aggregate_sha256") != aggregate_sha
    ):
        raise ContractError("sealed diagnostic final receipt binding failed")
    for field in (
        "selection_entropy_release",
        "confirmation_materialization",
        "formal_materialization",
    ):
        if receipt.get(field) != "PROHIBITED":
            raise ContractError(f"diagnostic receipt {field} is not PROHIBITED")
    return receipt, raw, aggregate_sha


_RAW_BUCKET_FIELDS = (
    "attempts",
    "attempt_charged_evaluations",
    "cache_hits",
    "physical_starts",
    "retry_attempts",
    "fallback_attempts",
    "primary_ordinal_zero_attempts",
    "primary_ordinal_zero_attempt_charged_evaluations",
    "primary_ordinal_zero_cache_hits",
    "primary_ordinal_zero_physical_starts",
    "evaluation_charged_evaluations",
    "archive_changed",
    "retained_after_update",
    "accepted_into_population",
    "population_replacement_count",
    "new_evaluated_cell",
    "new_nondominated_cell",
    "scalar_advantage_count",
    "positive_scalar_advantage",
)


def _bucket() -> dict[str, int | float]:
    return {**{field: 0 for field in _RAW_BUCKET_FIELDS}, "scalar_advantage_sum": 0.0}


def _merge(target: dict[str, int | float], source: Mapping[str, object]) -> None:
    for field in _RAW_BUCKET_FIELDS:
        target[field] = int(target[field]) + _exact_int(source.get(field), field)
    target["scalar_advantage_sum"] = float(target["scalar_advantage_sum"]) + _exact_number(
        source.get("scalar_advantage_sum"), "scalar_advantage_sum"
    )


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _finalize_bucket(raw: Mapping[str, int | float]) -> dict[str, int | float]:
    result = {field: int(raw[field]) for field in _RAW_BUCKET_FIELDS}
    result["scalar_advantage_sum"] = float(raw["scalar_advantage_sum"])
    result["cache_hit_rate_per_attempt"] = _ratio(
        result["cache_hits"], result["attempts"]
    )
    result["primary_ordinal_zero_cache_hit_rate_per_attempt"] = _ratio(
        result["primary_ordinal_zero_cache_hits"],
        result["primary_ordinal_zero_attempts"],
    )
    result["archive_change_rate_per_evaluation_charge"] = _ratio(
        result["archive_changed"], result["evaluation_charged_evaluations"]
    )
    result["accepted_rate_per_evaluation_charge"] = _ratio(
        result["accepted_into_population"],
        result["evaluation_charged_evaluations"],
    )
    return result


def _sqlite_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{label} must be an exact SQLite integer >= {minimum}")
    return value


def _canonical_sqlite_object(raw: object, label: str) -> dict[str, object]:
    if type(raw) is not str:
        raise ContractError(f"{label} must be exact JSON text")
    value = _loads(raw, label)
    if type(value) is not dict or raw.encode("utf-8") != _canonical_bytes(value):
        raise ContractError(f"{label} must be a canonical JSON object")
    return value


def _analyze_trace(
    trace: Path,
    *,
    expected_sha: str,
    expected_budget: int,
    expected_case: str,
    expected_family: str,
    expected_source_sha: str,
    detached_terminal: Mapping[str, object],
) -> tuple[dict[str, dict[str, int | float]], dict[str, int]]:
    if trace.with_name(trace.name + "-wal").exists() or trace.with_name(trace.name + "-shm").exists():
        raise ContractError("sealed trace has a live WAL/SHM sidecar")
    before_sha = _sha256_file(trace)
    if before_sha != expected_sha:
        raise ContractError("sealed trace SHA-256 binding failed")
    connection = sqlite3.connect(f"{trace.as_uri()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ContractError("sealed trace SQLite integrity_check failed")
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if tables != _EXPECTED_TABLES:
            raise ContractError("sealed trace table schema drifted")
        run_rows = list(connection.execute("SELECT * FROM run_attempt"))
        if len(run_rows) != 1 or _sqlite_int(run_rows[0]["run_id"], "run_id", minimum=1) != 1:
            raise ContractError("sealed trace must contain exactly run_id=1")
        run = run_rows[0]
        context = _canonical_sqlite_object(run["run_context_json"], "run context")
        context_raw = str(run["run_context_json"])
        context_sha = _sha256_bytes(context_raw.encode("utf-8"))
        if (
            str(run["status"]) != "SUCCESS"
            or str(run["problem"]) != expected_case
            or str(run["family"]) != expected_family
            or str(run["run_context_digest_sha256"]) != context_sha
            or context.get("schema") != "v21e3r1_run_context_v2"
            or context.get("charged_evaluation_budget") != expected_budget
            or context.get("evidence_partition") != "development"
            or context.get("algorithm_source_sha256") != expected_source_sha
        ):
            raise ContractError("sealed trace run-context binding failed")

        terminal_rows = list(connection.execute("SELECT * FROM terminal_receipts"))
        if len(terminal_rows) != 1 or _sqlite_int(terminal_rows[0]["run_id"], "terminal.run_id", minimum=1) != 1:
            raise ContractError("sealed trace must contain exactly one terminal receipt")
        terminal_row = terminal_rows[0]
        terminal = _canonical_sqlite_object(
            terminal_row["receipt_json"], "terminal receipt"
        )
        if terminal != detached_terminal:
            raise ContractError("detached and SQLite terminal receipts disagree")
        if (
            str(terminal_row["status"]) != "SUCCESS"
            or terminal_row["failure_code"] is not None
            or terminal.get("status") != "SUCCESS"
            or terminal.get("problem") != expected_case
            or terminal.get("family") != expected_family
            or terminal.get("run_context_digest_sha256") != context_sha
        ):
            raise ContractError("terminal success/context binding failed")
        payload_claim = terminal.get("receipt_payload_sha256")
        if payload_claim is not None:
            payload_sha = _sha256_value(payload_claim, "terminal.receipt_payload_sha256")
            terminal_core = dict(terminal)
            terminal_core.pop("receipt_payload_sha256")
            if (
                payload_sha != _sha256_bytes(_canonical_bytes(terminal_core))
                or str(terminal_row["receipt_sha256"]) != payload_sha
                or str(run["terminal_receipt_sha256"]) != payload_sha
            ):
                raise ContractError("terminal payload SHA-256 binding failed")

        operators: defaultdict[str, dict[str, int | float]] = defaultdict(_bucket)
        attempt_by_index: dict[int, dict[str, object]] = {}
        charged_indices: list[int] = []
        attempt_rows = list(
            connection.execute("SELECT * FROM attempts ORDER BY attempt_index")
        )
        for expected_attempt, attempt in enumerate(attempt_rows, start=1):
            attempt_index = _sqlite_int(
                attempt["attempt_index"], "attempt_index", minimum=1
            )
            if attempt_index != expected_attempt:
                raise ContractError("attempt indices are not contiguous")
            context_item = _canonical_sqlite_object(
                attempt["context_json"], f"attempt {attempt_index} context"
            )
            operator_id = _string(
                context_item.get("operator_id"),
                f"attempt {attempt_index}.operator_id",
            )
            _string(
                context_item.get("search_phase_id"),
                f"attempt {attempt_index}.search_phase_id",
            )
            _string(
                context_item.get("stage_id"),
                f"attempt {attempt_index}.stage_id",
            )
            _exact_int(
                context_item.get("operator_call_id"),
                f"attempt {attempt_index}.operator_call_id",
                minimum=1,
            )
            type_id = context_item.get("type_id")
            if type_id is not None:
                _exact_int(
                    type_id,
                    f"attempt {attempt_index}.type_id",
                )
            witness = context_item.get("operator_witness")
            if type(witness) is not dict:
                raise ContractError(
                    f"attempt {attempt_index}.operator_witness must be an object"
                )
            retry_ordinal = _exact_int(
                witness.get("retry_ordinal"),
                f"attempt {attempt_index}.operator_witness.retry_ordinal",
            )
            fallback = _exact_bool(
                witness.get("fallback_used"),
                f"attempt {attempt_index}.operator_witness.fallback_used",
            )
            physical = _sqlite_int(
                attempt["physical_call_started"],
                f"attempt {attempt_index}.physical_call_started",
            )
            if physical not in (0, 1):
                raise ContractError("physical_call_started must be exactly 0 or 1")
            status = str(attempt["status"])
            charged_raw = attempt["charged_evaluation_index"]
            cache_raw = attempt["cache_source_evaluation_index"]
            if status == "EVALUATED":
                charged = _sqlite_int(
                    charged_raw,
                    f"attempt {attempt_index}.charged_evaluation_index",
                    minimum=1,
                )
                if physical != 1 or cache_raw is not None:
                    raise ContractError("evaluated attempt charge fields are inconsistent")
                charged_indices.append(charged)
            elif status == "CACHE_HIT":
                if physical != 0 or charged_raw is not None:
                    raise ContractError("cache-hit attempt charge fields are inconsistent")
                cache_source = _sqlite_int(
                    cache_raw,
                    f"attempt {attempt_index}.cache_source_evaluation_index",
                    minimum=1,
                )
                if cache_source > len(charged_indices):
                    raise ContractError(
                        "cache-hit attempt does not reference an earlier evaluation"
                    )
            else:
                raise ContractError("successful trace contains a nonterminal attempt status")
            item = operators[operator_id]
            item["attempts"] = int(item["attempts"]) + 1
            item["physical_starts"] = int(item["physical_starts"]) + physical
            item["attempt_charged_evaluations"] = int(
                item["attempt_charged_evaluations"]
            ) + int(charged_raw is not None)
            item["cache_hits"] = int(item["cache_hits"]) + int(cache_raw is not None)
            item["retry_attempts"] = int(item["retry_attempts"]) + int(
                retry_ordinal > 0 and not fallback
            )
            item["fallback_attempts"] = int(item["fallback_attempts"]) + int(fallback)
            primary = retry_ordinal == 0 and not fallback
            if primary:
                item["primary_ordinal_zero_attempts"] = int(
                    item["primary_ordinal_zero_attempts"]
                ) + 1
                item["primary_ordinal_zero_attempt_charged_evaluations"] = int(
                    item["primary_ordinal_zero_attempt_charged_evaluations"]
                ) + int(charged_raw is not None)
                item["primary_ordinal_zero_cache_hits"] = int(
                    item["primary_ordinal_zero_cache_hits"]
                ) + int(cache_raw is not None)
                item["primary_ordinal_zero_physical_starts"] = int(
                    item["primary_ordinal_zero_physical_starts"]
                ) + physical
            attempt_by_index[attempt_index] = {
                "operator_id": operator_id,
                "status": status,
                "charged": charged_raw,
                "context": context_item,
            }
        if charged_indices != list(range(1, expected_budget + 1)):
            raise ContractError("attempt-ledger charged indices are not the exact budget")

        evaluations = list(
            connection.execute(
                "SELECT e.*,d.decision_json FROM evaluations e "
                "LEFT JOIN decisions d USING(evaluation_index) "
                "ORDER BY e.evaluation_index"
            )
        )
        if len(evaluations) != expected_budget:
            raise ContractError("evaluation row count disagrees with top-level budget")
        for expected_evaluation, evaluation in enumerate(evaluations, start=1):
            index = _sqlite_int(
                evaluation["evaluation_index"], "evaluation_index", minimum=1
            )
            if index != expected_evaluation:
                raise ContractError("evaluation indices are not contiguous")
            attempt_index = _sqlite_int(
                evaluation["attempt_index"],
                f"evaluation {index}.attempt_index",
                minimum=1,
            )
            attempt = attempt_by_index.get(attempt_index)
            if (
                attempt is None
                or attempt["status"] != "EVALUATED"
                or attempt["charged"] != index
            ):
                raise ContractError("evaluation does not bind its charged attempt")
            operator_id = _string(
                evaluation["operator_id"], f"evaluation {index}.operator_id"
            )
            _sqlite_int(
                evaluation["operator_call_id"],
                f"evaluation {index}.operator_call_id",
                minimum=1,
            )
            evaluation_type_id = evaluation["type_id"]
            if evaluation_type_id is not None:
                _sqlite_int(
                    evaluation_type_id,
                    f"evaluation {index}.type_id",
                )
            context_item = attempt["context"]
            if (
                operator_id != attempt["operator_id"]
                or evaluation["search_phase_id"] != context_item.get("search_phase_id")
                or evaluation["stage_id"] != context_item.get("stage_id")
                or evaluation["type_id"] != context_item.get("type_id")
                or evaluation["operator_call_id"]
                != context_item.get("operator_call_id")
            ):
                raise ContractError("evaluation columns disagree with attempt context")
            decision = _canonical_sqlite_object(
                evaluation["decision_json"], f"decision {index}"
            )
            if _exact_int(
                decision.get("evaluation_index"),
                f"decision {index}.evaluation_index",
                minimum=1,
            ) != index:
                raise ContractError("decision binds the wrong evaluation")
            item = operators[operator_id]
            item["evaluation_charged_evaluations"] = int(
                item["evaluation_charged_evaluations"]
            ) + 1
            for field in (
                "archive_changed",
                "retained_after_update",
                "accepted_into_population",
                "new_evaluated_cell",
                "new_nondominated_cell",
            ):
                item[field] = int(item[field]) + int(
                    _exact_bool(decision.get(field), f"decision {index}.{field}")
                )
            replacements = _exact_int(
                decision.get("population_replacement_count"),
                f"decision {index}.population_replacement_count",
            )
            item["population_replacement_count"] = int(
                item["population_replacement_count"]
            ) + replacements
            advantage = decision.get("scalar_advantage")
            if advantage is not None:
                numeric = _exact_number(
                    advantage, f"decision {index}.scalar_advantage"
                )
                item["scalar_advantage_count"] = int(
                    item["scalar_advantage_count"]
                ) + 1
                item["scalar_advantage_sum"] = float(
                    item["scalar_advantage_sum"]
                ) + numeric
                item["positive_scalar_advantage"] = int(
                    item["positive_scalar_advantage"]
                ) + int(numeric > 0.0)

        decision_count = int(
            connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        )
        attempt_charges = sum(
            int(item["attempt_charged_evaluations"])
            for item in operators.values()
        )
        evaluation_charges = sum(
            int(item["evaluation_charged_evaluations"])
            for item in operators.values()
        )
        attempt_count = len(attempt_rows)
        cache_hits = sum(int(item["cache_hits"]) for item in operators.values())
        physical_starts = sum(
            int(item["physical_starts"]) for item in operators.values()
        )
        if (
            evaluation_charges != expected_budget
            or attempt_charges != expected_budget
            or decision_count != expected_budget
            or any(
                int(item["evaluation_charged_evaluations"])
                != int(item["attempt_charged_evaluations"])
                for item in operators.values()
            )
            or terminal.get("charged_evaluation_count") != expected_budget
            or terminal.get("decision_count") != expected_budget
            or terminal.get("attempt_count") != attempt_count
            or terminal.get("cache_hit_count") != cache_hits
            or terminal.get("physical_call_started_count") != physical_starts
        ):
            raise ContractError("trace terminal/operator accounting invariant failed")
    finally:
        connection.close()
    if _sha256_file(trace) != before_sha:
        raise ContractError("sealed trace changed during read-only reanalysis")
    return dict(operators), {
        "attempt_count": attempt_count,
        "attempt_charged_evaluations": attempt_charges,
        "cache_hit_count": cache_hits,
        "evaluation_charged_evaluations": evaluation_charges,
        "physical_start_count": physical_starts,
    }


def _legacy_sum(diagnostic: Mapping[str, object], budget: int) -> int:
    if (
        diagnostic.get("schema") != LEGACY_DIAGNOSTIC_SCHEMA
        or diagnostic.get("status") != "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY"
        or _exact_int(
            diagnostic.get("charged_evaluation_count"),
            "legacy diagnostic charged_evaluation_count",
            minimum=1,
        )
        != budget
    ):
        raise ContractError("legacy diagnostic row contract drifted")
    operators = diagnostic.get("operators")
    if type(operators) is not dict or not operators:
        raise ContractError("legacy diagnostic has no operator accounting")
    total = 0
    for operator_id, raw in operators.items():
        _string(operator_id, "legacy operator id")
        if type(raw) is not dict:
            raise ContractError("legacy operator accounting must be an object")
        value = _exact_number(
            raw.get("charged_evaluations"),
            f"legacy operator {operator_id}.charged_evaluations",
        )
        if value < 0.0 or not value.is_integer():
            raise ContractError("legacy operator charge must be a nonnegative integer value")
        total += int(value)
    if total != 2 * budget:
        raise ContractError("legacy operator double-count signature is absent or drifted")
    return total


def _expected_row_ids(
    cases: Iterable[str], seeds: Iterable[int], arms: Iterable[str]
) -> list[str]:
    return [
        f"{case_id}__seed-{seed}__arm-{arm.lower()}"
        for case_id in cases
        for seed in seeds
        for arm in arms
    ]


def _reanalyze_row(
    root: Path,
    *,
    row_id: str,
    case_id: str,
    seed: int,
    arm: str,
    plan_sha: str,
    source_sha: str,
    budget: int,
) -> tuple[dict[str, object], dict[str, dict[str, int | float]]]:
    completed_path = root / "completed" / f"{row_id}.json"
    completed, completed_raw = _load_object(completed_path, f"completed row {row_id}")
    if set(completed) != {
        "attempt_directory",
        "diagnostic_sha256",
        "independent_metric_receipt_sha256",
        "plan_sha256",
        "row_id",
        "row_sha256",
        "status",
        "terminal_receipt_sha256",
        "trace_sha256",
    }:
        raise ContractError(f"completed row key contract drifted: {row_id}")
    if (
        completed.get("row_id") != row_id
        or completed.get("status") != "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY"
        or completed.get("plan_sha256") != plan_sha
    ):
        raise ContractError(f"completed row seal binding failed: {row_id}")
    attempt = _relative_file(
        root,
        str(completed.get("attempt_directory")) + "/row.json",
        f"{row_id}.attempt row",
    ).parent
    artifact_contract = (
        ("row.json", "row_sha256"),
        ("diagnostic.json", "diagnostic_sha256"),
        ("independent.metric.json", "independent_metric_receipt_sha256"),
        ("terminal.receipt.json", "terminal_receipt_sha256"),
        ("trace.sqlite3", "trace_sha256"),
    )
    for filename, key in artifact_contract:
        path = attempt / filename
        if not path.is_file():
            raise ContractError(f"sealed row artifact is missing: {row_id}/{filename}")
        claimed = _sha256_value(completed.get(key), f"{row_id}.{key}")
        if _sha256_file(path) != claimed:
            label = "trace SHA-256" if filename == "trace.sqlite3" else f"{filename} SHA-256"
            raise ContractError(f"{label} binding failed for {row_id}")

    row, row_raw = _load_object(attempt / "row.json", f"row artifact {row_id}")
    family = _family_for_case(case_id)
    if (
        row.get("schema") != ROW_SCHEMA
        or row.get("status") != "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY"
        or row.get("scientific_scope") != SCOPE
        or row.get("case_id") != case_id
        or row.get("family") != family
        or row.get("seed") != seed
        or row.get("arm_id") != arm
        or row.get("charged_evaluation_budget") != budget
        or row.get("plan_sha256") != plan_sha
        or row.get("source_snapshot_sha256") != source_sha
    ):
        raise ContractError(f"diagnostic row semantic binding failed: {row_id}")
    for field in (
        "selection_entropy_release",
        "confirmation_materialization",
        "formal_materialization",
    ):
        if row.get(field) != "PROHIBITED":
            raise ContractError(f"diagnostic row {field} is not PROHIBITED: {row_id}")
    for path_field, sha_field, expected_name in (
        ("trace_database_path", "trace_database_sha256", "trace.sqlite3"),
        ("terminal_receipt_path", "terminal_receipt_sha256", "terminal.receipt.json"),
        (
            "independent_metric_receipt_path",
            "independent_metric_receipt_sha256",
            "independent.metric.json",
        ),
    ):
        if row.get(path_field) != expected_name or row.get(sha_field) != completed.get(
            "trace_sha256" if expected_name == "trace.sqlite3" else (
                "terminal_receipt_sha256"
                if expected_name == "terminal.receipt.json"
                else "independent_metric_receipt_sha256"
            )
        ):
            raise ContractError(f"diagnostic row artifact binding drifted: {row_id}")

    terminal, terminal_raw = _load_object(
        attempt / "terminal.receipt.json", f"detached terminal {row_id}"
    )
    if terminal_raw != _canonical_bytes(terminal):
        raise ContractError(f"detached terminal receipt is not canonical: {row_id}")
    diagnostic, diagnostic_raw = _load_object(
        attempt / "diagnostic.json", f"legacy diagnostic {row_id}"
    )
    if (
        diagnostic.get("case_id") != case_id
        or diagnostic.get("family") != family
        or diagnostic.get("seed") != seed
        or diagnostic.get("arm_id") != arm
        or diagnostic.get("budget") != budget
    ):
        raise ContractError(f"legacy diagnostic identity binding failed: {row_id}")
    legacy_charges = _legacy_sum(diagnostic, budget)
    operators, totals = _analyze_trace(
        attempt / "trace.sqlite3",
        expected_sha=str(completed["trace_sha256"]),
        expected_budget=budget,
        expected_case=case_id,
        expected_family=family,
        expected_source_sha=source_sha,
        detached_terminal=terminal,
    )
    if (
        diagnostic.get("attempt_count") != totals["attempt_count"]
        or diagnostic.get("cache_hit_count") != totals["cache_hit_count"]
        or diagnostic.get("physical_start_count") != totals["physical_start_count"]
    ):
        raise ContractError(f"legacy top-level accounting binding failed: {row_id}")
    finalized = {
        operator_id: _finalize_bucket(raw)
        for operator_id, raw in sorted(operators.items())
    }
    primary_totals = {
        "attempts": sum(
            int(item["primary_ordinal_zero_attempts"])
            for item in finalized.values()
        ),
        "attempt_charged_evaluations": sum(
            int(item["primary_ordinal_zero_attempt_charged_evaluations"])
            for item in finalized.values()
        ),
        "cache_hits": sum(
            int(item["primary_ordinal_zero_cache_hits"])
            for item in finalized.values()
        ),
        "physical_starts": sum(
            int(item["primary_ordinal_zero_physical_starts"])
            for item in finalized.values()
        ),
    }
    primary_totals["cache_hit_rate_per_attempt"] = _ratio(
        primary_totals["cache_hits"], primary_totals["attempts"]
    )
    output = {
        "schema": OUTPUT_ROW_SCHEMA,
        "status": "PASS_CORRECTED_OPERATOR_ACCOUNTING_DEVELOPMENT_ONLY",
        "scientific_scope": SCOPE,
        "row_id": row_id,
        "case_id": case_id,
        "family": family,
        "seed": seed,
        "arm_id": arm,
        "charged_evaluation_budget": budget,
        "attempt_count": totals["attempt_count"],
        "cache_hit_count": totals["cache_hit_count"],
        "physical_start_count": totals["physical_start_count"],
        "evaluation_charged_evaluations": totals[
            "evaluation_charged_evaluations"
        ],
        "attempt_charged_evaluations": totals["attempt_charged_evaluations"],
        "legacy_operator_charged_evaluations": legacy_charges,
        "primary_ordinal_zero_definition": PRIMARY_DEFINITION,
        "primary_ordinal_zero": primary_totals,
        "operators": finalized,
        "invariants": {
            "evaluation_charge_sum_equals_budget": True,
            "attempt_charge_sum_equals_ledger_charged_flags": True,
            "per_operator_evaluation_and_attempt_charges_match": True,
            "legacy_double_count_signature_observed": True,
            "trace_sha256_unchanged_during_read": True,
        },
        "bindings": {
            "plan_sha256": plan_sha,
            "completed_receipt_path": f"completed/{row_id}.json",
            "completed_receipt_sha256": _sha256_bytes(completed_raw),
            "row_path": f"{completed['attempt_directory']}/row.json",
            "row_sha256": _sha256_bytes(row_raw),
            "diagnostic_path": f"{completed['attempt_directory']}/diagnostic.json",
            "diagnostic_sha256": _sha256_bytes(diagnostic_raw),
            "terminal_receipt_path": f"{completed['attempt_directory']}/terminal.receipt.json",
            "terminal_receipt_sha256": _sha256_bytes(terminal_raw),
            "trace_path": f"{completed['attempt_directory']}/trace.sqlite3",
            "trace_sha256": str(completed["trace_sha256"]),
            "development_source_sha256": source_sha,
        },
        "authority": {
            "implementation_independence": False,
            "algorithm_execution_independence": False,
            "scientific_independence": False,
            "selection_authorized": False,
            "confirmation_authorized": False,
            "formal_study_authorized": False,
            "publication_status": "IJOC_HOLD",
        },
    }
    return output, operators


def _aggregate_raw_buckets(
    rows: Sequence[tuple[dict[str, object], dict[str, dict[str, int | float]]]]
) -> tuple[
    dict[str, dict[str, int | float]],
    dict[str, dict[str, int | float]],
    dict[str, dict[str, int | float]],
]:
    by_family: defaultdict[str, dict[str, int | float]] = defaultdict(_bucket)
    by_arm: defaultdict[str, dict[str, int | float]] = defaultdict(_bucket)
    by_operator: defaultdict[str, dict[str, int | float]] = defaultdict(_bucket)
    for row, operators in rows:
        row_total = _bucket()
        for operator_id, raw in operators.items():
            _merge(row_total, raw)
            _merge(by_operator[operator_id], raw)
        _merge(by_family[str(row["family"])], row_total)
        _merge(by_arm[str(row["arm_id"])], row_total)
    return dict(by_family), dict(by_arm), dict(by_operator)


def _exclusive_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def reanalyze(
    diagnostic_output_root: str | Path,
    output_directory: str | Path,
    *,
    allow_smoke: bool = False,
) -> dict[str, object]:
    if type(allow_smoke) is not bool:
        raise ContractError("allow_smoke must be an exact boolean")
    root = Path(diagnostic_output_root).resolve()
    output = Path(output_directory).resolve()
    if not root.is_dir():
        raise ContractError("diagnostic output root is not a directory")
    if output.exists():
        raise FileExistsError(output)
    if output == root or root in output.parents:
        raise ContractError("reanalysis output must be outside the sealed diagnostic root")

    source_path = Path(__file__).resolve()
    source_raw = source_path.read_bytes()
    source_tool_sha = _sha256_bytes(source_raw)
    metric_spec_path = source_path.with_name(METRIC_SPEC_SOURCE_NAME)
    metric_spec_raw, metric_spec_sha = _validate_metric_spec(metric_spec_path)

    (
        plan,
        plan_raw,
        cases,
        seeds,
        arms,
        budget,
        source_sha,
        matrix_mode,
    ) = _validate_plan(root, allow_smoke=allow_smoke)
    plan_sha = _sha256_bytes(plan_raw)
    expected_rows = _exact_int(plan.get("expected_rows"), "plan.expected_rows", minimum=1)
    _, diagnostic_receipt_raw, diagnostic_aggregate_sha = _validate_final_receipt(
        root,
        plan_sha=plan_sha,
        source_sha=source_sha,
        expected_rows=expected_rows,
        matrix_mode=matrix_mode,
    )
    expected_ids = _expected_row_ids(cases, seeds, arms)
    completed_directory = root / "completed"
    if not completed_directory.is_dir():
        raise ContractError("sealed diagnostic completed directory is missing")
    observed_files = sorted(
        path.name for path in completed_directory.iterdir() if path.is_file()
    )
    expected_files = sorted(f"{row_id}.json" for row_id in expected_ids)
    if observed_files != expected_files:
        raise ContractError("sealed diagnostic completed row set is not exact")

    row_results: list[tuple[dict[str, object], dict[str, dict[str, int | float]]]] = []
    for case_id in cases:
        for seed in seeds:
            for arm in arms:
                row_id = f"{case_id}__seed-{seed}__arm-{arm.lower()}"
                row_results.append(
                    _reanalyze_row(
                        root,
                        row_id=row_id,
                        case_id=case_id,
                        seed=seed,
                        arm=arm,
                        plan_sha=plan_sha,
                        source_sha=source_sha,
                        budget=budget,
                    )
                )
    rows_only = [item[0] for item in row_results]
    if len(rows_only) != expected_rows:
        raise ContractError("not all sealed diagnostic rows were reanalyzed")
    rows_raw = b"".join(_canonical_bytes(row) + b"\n" for row in rows_only)
    rows_sha = _sha256_bytes(rows_raw)

    by_family, by_arm, by_operator = _aggregate_raw_buckets(row_results)
    evaluation_sum = sum(
        int(row["evaluation_charged_evaluations"]) for row in rows_only
    )
    attempt_sum = sum(int(row["attempt_charged_evaluations"]) for row in rows_only)
    legacy_sum = sum(
        int(row["legacy_operator_charged_evaluations"]) for row in rows_only
    )
    expected_sum = expected_rows * budget
    if (
        evaluation_sum != expected_sum
        or attempt_sum != expected_sum
        or legacy_sum != 2 * expected_sum
    ):
        raise ContractError("matrix-level corrected accounting invariant failed")
    aggregate_status = (
        "PASS_CORRECTED_REANALYSIS_EXACT_504_DEVELOPMENT_ONLY"
        if matrix_mode == "FULL_504"
        else "PASS_CORRECTED_REANALYSIS_SMOKE_ONLY"
    )
    aggregate = {
        "schema": OUTPUT_AGGREGATE_SCHEMA,
        "status": aggregate_status,
        "scientific_scope": SCOPE,
        "matrix_mode": matrix_mode,
        "completed_rows": len(rows_only),
        "expected_rows": expected_rows,
        "charged_evaluations_per_row": budget,
        "evaluation_charged_evaluations_sum": evaluation_sum,
        "attempt_charged_evaluations_sum": attempt_sum,
        "legacy_operator_charged_evaluations_sum": legacy_sum,
        "primary_ordinal_zero_definition": PRIMARY_DEFINITION,
        "by_family": {
            key: _finalize_bucket(raw) for key, raw in sorted(by_family.items())
        },
        "by_arm": {
            key: _finalize_bucket(raw) for key, raw in sorted(by_arm.items())
        },
        "by_operator": {
            key: _finalize_bucket(raw) for key, raw in sorted(by_operator.items())
        },
        "invariants": {
            "all_rows_reanalyzed": True,
            "evaluation_charge_sum_equals_top_level_budget": True,
            "attempt_charge_sum_equals_attempt_ledger_charged_flags": True,
            "legacy_double_count_signature_observed_for_every_row": True,
            "operator_charge_double_count_corrected": True,
            "original_artifacts_modified": False,
        },
        "bindings": {
            "plan_sha256": plan_sha,
            "diagnostic_receipt_sha256": _sha256_bytes(diagnostic_receipt_raw),
            "diagnostic_aggregate_sha256": diagnostic_aggregate_sha,
            "development_source_sha256": source_sha,
            "rows_path": ROWS_NAME,
            "rows_sha256": rows_sha,
            "metric_spec_path": METRIC_SPEC_OUTPUT_PATH,
            "metric_spec_sha256": metric_spec_sha,
            "metric_source_path": METRIC_SOURCE_OUTPUT_PATH,
            "metric_source_sha256": source_tool_sha,
        },
        "authority": {
            "implementation_independence": False,
            "algorithm_execution_independence": False,
            "scientific_independence": False,
            "selection_authorized": False,
            "confirmation_authorized": False,
            "formal_study_authorized": False,
            "publication_status": "IJOC_HOLD",
        },
    }
    aggregate_raw = _canonical_bytes(aggregate)
    aggregate_sha = _sha256_bytes(aggregate_raw)
    receipt = {
        "schema": OUTPUT_RECEIPT_SCHEMA,
        "status": aggregate_status,
        "scientific_scope": SCOPE,
        "matrix_mode": matrix_mode,
        "completed_rows": len(rows_only),
        "expected_rows": expected_rows,
        "charged_evaluations_per_row": budget,
        "evaluation_charged_evaluations_sum": evaluation_sum,
        "attempt_charged_evaluations_sum": attempt_sum,
        "legacy_operator_charged_evaluations_sum": legacy_sum,
        "operator_charge_double_count_corrected": True,
        "all_rows_reanalyzed": True,
        "original_artifacts_modified": False,
        "plan_sha256": plan_sha,
        "diagnostic_receipt_sha256": _sha256_bytes(diagnostic_receipt_raw),
        "diagnostic_aggregate_sha256": diagnostic_aggregate_sha,
        "development_source_sha256": source_sha,
        "rows_path": ROWS_NAME,
        "rows_sha256": rows_sha,
        "aggregate_path": AGGREGATE_NAME,
        "aggregate_sha256": aggregate_sha,
        "metric_spec_path": METRIC_SPEC_OUTPUT_PATH,
        "metric_spec_sha256": metric_spec_sha,
        "metric_source_path": METRIC_SOURCE_OUTPUT_PATH,
        "metric_source_sha256": source_tool_sha,
        "implementation_independence": False,
        "algorithm_execution_independence": False,
        "scientific_independence": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_study_authorized": False,
        "publication_status": "IJOC_HOLD",
    }
    if set(receipt) != RECEIPT_KEYS:
        raise AssertionError("internal receipt key contract drifted")
    receipt_raw = _canonical_bytes(receipt)

    output.mkdir(parents=True, exist_ok=False)
    _exclusive_write(output / ROWS_NAME, rows_raw)
    _exclusive_write(output / AGGREGATE_NAME, aggregate_raw)
    _exclusive_write(output / METRIC_SPEC_OUTPUT_PATH, metric_spec_raw)
    _exclusive_write(output / METRIC_SOURCE_OUTPUT_PATH, source_raw)
    _exclusive_write(output / RECEIPT_NAME, receipt_raw)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only, fail-closed correction of V21e3r1 development "
            "operator accounting."
        )
    )
    parser.add_argument("--diagnostic-output-root", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="permit a sealed non-504 smoke fixture; never grants study authority",
    )
    args = parser.parse_args(argv)
    receipt = reanalyze(
        args.diagnostic_output_root,
        args.output_directory,
        allow_smoke=args.allow_smoke,
    )
    print(_canonical_bytes(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
