#!/usr/bin/env python3
from __future__ import annotations

"""Independent simultaneous-inference evaluator for V21e3r1.

The implementation is intentionally Python-standard-library-only and does not
import ``mo_nco``.  It recomputes a frozen paired-case analysis from row-level
scores.  It does not authenticate producer, custodian, institutional, or
scientific independence; higher-level signed verifiers retain that authority.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Mapping, Sequence


INPUT_SCHEMA = "v21e3r1_simultaneous_evaluation_input_v1"
RECEIPT_SCHEMA = "v21e3r1_independent_simultaneous_inference_receipt_v1"
METHOD = "one_sided_observed_se_max_t_paired_case_cluster_bootstrap_v1"
FAMILIES = ("MOKP", "MOTSP")
CANDIDATES = ("C0", "C1", "C2", "C3")
SELECTION_CONTRASTS = (
    ("C1", "C0"),
    ("C2", "C0"),
    ("C2", "C1"),
    ("C3", "C0"),
    ("C3", "C2"),
)
TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "phase",
        "study_id",
        "study_freeze_sha256",
        "phase_manifest_sha256",
        "matrix_receipt_sha256",
        "source_root_sha256",
        "metric_spec_sha256",
        "decision_spec_sha256",
        "effect_direction",
        "case_ids_by_family",
        "seeds",
        "inference",
        "thresholds",
        "selection_binding",
        "confirmation_controls",
        "rows",
    }
)
INFERENCE_KEYS = frozenset(
    {"method", "alpha", "bootstrap_samples", "bootstrap_seed"}
)
THRESHOLD_KEYS = frozenset({"primary", "adjacent"})
ROW_KEYS = frozenset({"family", "case_id", "seed", "candidate", "score"})
SELECTION_BINDING_KEYS = frozenset(
    {"selection_receipt_sha256", "selection_status", "selected_candidate"}
)
CONFIRMATION_CONTROL_KEYS = frozenset(
    {
        "external_producer",
        "external_producer_receipt_sha256",
        "independent_custody",
        "custody_receipt_sha256",
        "independent_statistics",
        "statistics_source_sha256",
    }
)
HEX_DIGITS = frozenset("0123456789abcdef")


class ContractError(ValueError):
    """An input violates the frozen artifact contract."""


class ZeroStandardErrorHold(RuntimeError):
    """A valid matrix cannot support the frozen studentized procedure."""

    def __init__(self, hypotheses: Sequence[str]) -> None:
        super().__init__("zero observed case-cluster standard error")
        self.hypotheses = tuple(hypotheses)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> object:
    raise ContractError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _check_json_scalars(value: object, location: str = "input") -> None:
    if type(value) is float and not math.isfinite(value):
        raise ContractError(f"{location} contains a non-finite JSON number")
    if type(value) is str and any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ContractError(f"{location} contains a non-Unicode scalar surrogate")
    if isinstance(value, dict):
        for key, child in value.items():
            _check_json_scalars(key, f"{location}.<key>")
            _check_json_scalars(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_json_scalars(child, f"{location}[{index}]")


def _load_strict_json(path: Path) -> tuple[dict[str, object], str]:
    if not path.is_file():
        raise ContractError(f"input is not a regular file: {path}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError("input must be strict UTF-8 JSON") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ContractError(f"invalid JSON: {error.msg}") from error
    if type(value) is not dict:
        raise ContractError("input root must be an exact JSON object")
    _check_json_scalars(value)
    return value, hashlib.sha256(raw).hexdigest()


def _require_keys(
    value: object, expected: frozenset[str], location: str
) -> dict[str, object]:
    if type(value) is not dict:
        raise ContractError(f"{location} must be an exact JSON object")
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ContractError(
            f"{location} must have the exact frozen key set; "
            f"missing={missing}, extra={extra}"
        )
    return value


def _require_string(value: object, location: str) -> str:
    if type(value) is not str or not value.strip():
        raise ContractError(f"{location} must be a nonempty exact string")
    return value


def _require_sha256(value: object, location: str) -> str:
    text = _require_string(value, location)
    if len(text) != 64 or any(character not in HEX_DIGITS for character in text):
        raise ContractError(f"{location} must be a lowercase SHA-256 digest")
    return text


def _require_number(value: object, location: str) -> float:
    if type(value) not in (int, float):
        raise ContractError(f"{location} must be an exact finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{location} must be an exact finite JSON number")
    return result


def _require_integer(
    value: object,
    location: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ContractError(f"{location} must be an exact integer")
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f">= {minimum}" if maximum is None else f"in [{minimum}, {maximum}]"
        raise ContractError(f"{location} must be {bounds}")
    return value


def _mean(values: Sequence[float]) -> float:
    try:
        result = math.fsum(values) / len(values)
    except OverflowError as error:
        raise ContractError("numeric aggregation overflowed") from error
    if not math.isfinite(result):
        raise ContractError("numeric aggregation produced a non-finite result")
    return result


def _sample_standard_error(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = _mean(values)
    try:
        variance = math.fsum((value - center) ** 2 for value in values) / (
            len(values) - 1
        )
        result = math.sqrt(variance / len(values))
    except OverflowError as error:
        raise ContractError("numeric aggregation overflowed") from error
    if not math.isfinite(result):
        raise ContractError("numeric aggregation produced a non-finite result")
    return result


def _trimmed_mean(values: Sequence[float]) -> float:
    ordered = sorted(values)
    trim = len(ordered) // 10
    retained = ordered[trim : len(ordered) - trim] if trim else ordered
    return _mean(retained)


class _Sha256CounterRng:
    """Version-independent deterministic stream for bootstrap index draws."""

    def __init__(self, seed: int) -> None:
        self._key = hashlib.sha256(
            _canonical_bytes(
                {
                    "domain": "v21e3r1-simultaneous-case-bootstrap-v1",
                    "seed": seed,
                }
            )
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


def _validate_case_design(payload: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    raw = payload["case_ids_by_family"]
    if type(raw) is not dict or frozenset(raw) != frozenset(FAMILIES):
        raise ContractError(
            "case_ids_by_family must be an exact MOKP/MOTSP object"
        )
    result: dict[str, tuple[str, ...]] = {}
    all_cases: set[str] = set()
    for family in FAMILIES:
        case_ids = raw[family]
        if type(case_ids) is not list or len(case_ids) < 2:
            raise ContractError(
                f"case_ids_by_family.{family} must contain at least two cases"
            )
        normalized = tuple(
            _require_string(case_id, f"case_ids_by_family.{family}[]")
            for case_id in case_ids
        )
        if len(set(normalized)) != len(normalized):
            raise ContractError(f"case_ids_by_family.{family} contains duplicates")
        overlap = all_cases.intersection(normalized)
        if overlap:
            raise ContractError(
                f"case identifiers must be disjoint across families: {sorted(overlap)}"
            )
        all_cases.update(normalized)
        result[family] = normalized
    return result


def _validate_seeds(payload: Mapping[str, object]) -> tuple[int, ...]:
    raw = payload["seeds"]
    if type(raw) is not list or not raw:
        raise ContractError("seeds must be a nonempty exact JSON array")
    seeds = tuple(
        _require_integer(seed, "seeds[]", minimum=0, maximum=(1 << 63) - 1)
        for seed in raw
    )
    if len(set(seeds)) != len(seeds):
        raise ContractError("seeds must be unique")
    return seeds


def _validate_inference(payload: Mapping[str, object]) -> dict[str, object]:
    raw = _require_keys(payload["inference"], INFERENCE_KEYS, "inference")
    if raw["method"] != METHOD:
        raise ContractError(f"inference.method must equal {METHOD}")
    alpha = _require_number(raw["alpha"], "inference.alpha")
    if alpha != 0.05:
        raise ContractError("inference.alpha must equal the frozen one-sided 0.05")
    samples = _require_integer(
        raw["bootstrap_samples"],
        "inference.bootstrap_samples",
        minimum=99,
        maximum=1_000_000,
    )
    seed = _require_integer(
        raw["bootstrap_seed"],
        "inference.bootstrap_seed",
        minimum=0,
        maximum=(1 << 63) - 1,
    )
    rank = math.ceil((1.0 - alpha) * (samples + 1))
    if rank > samples:
        raise ContractError(
            "bootstrap_samples is insufficient for the frozen quantile convention"
        )
    return {
        "method": METHOD,
        "alpha": alpha,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "quantile_rank_one_based": rank,
    }


def _validate_thresholds(payload: Mapping[str, object]) -> dict[str, float]:
    raw = _require_keys(payload["thresholds"], THRESHOLD_KEYS, "thresholds")
    primary = _require_number(raw["primary"], "thresholds.primary")
    adjacent = _require_number(raw["adjacent"], "thresholds.adjacent")
    if primary != 0.0:
        raise ContractError("thresholds.primary must equal the frozen value 0.0")
    if adjacent != 0.005:
        raise ContractError("thresholds.adjacent must equal the frozen value 0.005")
    return {"primary": primary, "adjacent": adjacent}


def _validate_selection_phase(payload: Mapping[str, object]) -> None:
    if payload["selection_binding"] is not None:
        raise ContractError("selection phase requires selection_binding=null")
    if payload["confirmation_controls"] is not None:
        raise ContractError("selection phase requires confirmation_controls=null")


def _validate_confirmation_phase(
    payload: Mapping[str, object],
) -> tuple[str, dict[str, object], dict[str, object]]:
    binding = _require_keys(
        payload["selection_binding"],
        SELECTION_BINDING_KEYS,
        "selection_binding",
    )
    if binding["selection_status"] != "PASS_SELECTION":
        raise ContractError(
            "confirmation requires selection_binding.selection_status=PASS_SELECTION"
        )
    selected_candidate = _require_string(
        binding["selected_candidate"], "selection_binding.selected_candidate"
    )
    if selected_candidate not in ("C1", "C2", "C3"):
        raise ContractError(
            "selection_binding.selected_candidate must be one of C1, C2, C3"
        )
    normalized_binding: dict[str, object] = {
        "selection_receipt_sha256": _require_sha256(
            binding["selection_receipt_sha256"],
            "selection_binding.selection_receipt_sha256",
        ),
        "selection_status": "PASS_SELECTION",
        "selected_candidate": selected_candidate,
    }
    controls = _require_keys(
        payload["confirmation_controls"],
        CONFIRMATION_CONTROL_KEYS,
        "confirmation_controls",
    )
    for field in (
        "external_producer",
        "independent_custody",
        "independent_statistics",
    ):
        if type(controls[field]) is not bool or controls[field] is not True:
            raise ContractError(
                f"confirmation_controls.{field} must be exact true"
            )
    normalized_controls: dict[str, object] = {
        "external_producer": True,
        "external_producer_receipt_sha256": _require_sha256(
            controls["external_producer_receipt_sha256"],
            "confirmation_controls.external_producer_receipt_sha256",
        ),
        "independent_custody": True,
        "custody_receipt_sha256": _require_sha256(
            controls["custody_receipt_sha256"],
            "confirmation_controls.custody_receipt_sha256",
        ),
        "independent_statistics": True,
        "statistics_source_sha256": _require_sha256(
            controls["statistics_source_sha256"],
            "confirmation_controls.statistics_source_sha256",
        ),
    }
    evidence_hashes = {
        str(normalized_controls["external_producer_receipt_sha256"]),
        str(normalized_controls["custody_receipt_sha256"]),
        str(normalized_controls["statistics_source_sha256"]),
    }
    if len(evidence_hashes) != 3:
        raise ContractError(
            "confirmation control evidence hashes must be pairwise distinct"
        )
    source_sha256 = _file_sha256(Path(__file__).resolve())
    if normalized_controls["statistics_source_sha256"] != source_sha256:
        raise ContractError(
            "confirmation_controls.statistics_source_sha256 does not bind this "
            "independent evaluator source"
        )
    return selected_candidate, normalized_binding, normalized_controls


def _validate_rows(
    payload: Mapping[str, object],
    cases: Mapping[str, tuple[str, ...]],
    seeds: tuple[int, ...],
    candidates: tuple[str, ...],
) -> dict[tuple[str, str, int, str], float]:
    raw_rows = payload["rows"]
    if type(raw_rows) is not list:
        raise ContractError("rows must be an exact JSON array")
    rows: dict[tuple[str, str, int, str], float] = {}
    for index, raw_row in enumerate(raw_rows):
        row = _require_keys(raw_row, ROW_KEYS, f"rows[{index}]")
        family = _require_string(row["family"], f"rows[{index}].family")
        if family not in FAMILIES:
            raise ContractError(f"rows[{index}].family is outside the frozen design")
        case_id = _require_string(row["case_id"], f"rows[{index}].case_id")
        if case_id not in cases[family]:
            raise ContractError(f"rows[{index}].case_id is outside the frozen design")
        seed = _require_integer(
            row["seed"],
            f"rows[{index}].seed",
            minimum=0,
            maximum=(1 << 63) - 1,
        )
        if seed not in seeds:
            raise ContractError(f"rows[{index}].seed is outside the frozen design")
        candidate = _require_string(
            row["candidate"], f"rows[{index}].candidate"
        )
        if candidate not in candidates:
            raise ContractError(
                f"rows[{index}].candidate is outside the phase design"
            )
        score = _require_number(row["score"], f"rows[{index}].score")
        key = (family, case_id, seed, candidate)
        if key in rows:
            raise ContractError(f"duplicate case x seed x candidate row: {key}")
        rows[key] = score
    expected = {
        (family, case_id, seed, candidate)
        for family in FAMILIES
        for case_id in cases[family]
        for seed in seeds
        for candidate in candidates
    }
    actual = set(rows)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ContractError(
            "rows must provide exact case x seed x candidate full coverage; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return rows


def _build_cells(
    cases: Mapping[str, tuple[str, ...]],
    seeds: tuple[int, ...],
    rows: Mapping[tuple[str, str, int, str], float],
    contrasts: Sequence[tuple[str, str]],
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for family in FAMILIES:
        for candidate, reference in contrasts:
            differences: list[float] = []
            for case_id in cases[family]:
                candidate_mean = _mean(
                    [rows[(family, case_id, seed, candidate)] for seed in seeds]
                )
                reference_mean = _mean(
                    [rows[(family, case_id, seed, reference)] for seed in seeds]
                )
                difference = candidate_mean - reference_mean
                if not math.isfinite(difference):
                    raise ContractError(
                        "numeric aggregation produced a non-finite paired difference"
                    )
                differences.append(difference)
            observed_mean = _mean(differences)
            standard_error = _sample_standard_error(differences)
            cells.append(
                {
                    "hypothesis_id": f"{family}:{candidate}-{reference}",
                    "family": family,
                    "candidate": candidate,
                    "reference": reference,
                    "case_count": len(differences),
                    "seed_count_per_case_arm": len(seeds),
                    "case_differences": differences,
                    "observed_mean": observed_mean,
                    "standard_error": standard_error,
                    "median": statistics.median(differences),
                    "trimmed_mean_10_percent": _trimmed_mean(differences),
                    "wins": sum(value > 0.0 for value in differences),
                    "ties": sum(value == 0.0 for value in differences),
                    "losses": sum(value < 0.0 for value in differences),
                }
            )
    return cells


def _simultaneous_bounds(
    cells: list[dict[str, object]],
    inference: Mapping[str, object],
) -> tuple[float, float, str]:
    zero_se = [
        str(cell["hypothesis_id"])
        for cell in cells
        if not math.isfinite(float(cell["standard_error"]))
        or float(cell["standard_error"]) <= 0.0
    ]
    if zero_se:
        raise ZeroStandardErrorHold(zero_se)
    by_family = {
        family: [cell for cell in cells if cell["family"] == family]
        for family in FAMILIES
    }
    rng = _Sha256CounterRng(int(inference["bootstrap_seed"]))
    maxima: list[float] = []
    for _ in range(int(inference["bootstrap_samples"])):
        replicate_statistics: list[float] = []
        for family in FAMILIES:
            family_cells = by_family[family]
            case_count = int(family_cells[0]["case_count"])
            sampled_indices = [rng.randbelow(case_count) for _ in range(case_count)]
            for cell in family_cells:
                differences = cell["case_differences"]
                bootstrap_mean = _mean(
                    [float(differences[index]) for index in sampled_indices]
                )
                centered_t = (
                    bootstrap_mean - float(cell["observed_mean"])
                ) / float(cell["standard_error"])
                if not math.isfinite(centered_t):
                    raise ContractError("bootstrap generated a non-finite t statistic")
                replicate_statistics.append(centered_t)
        maxima.append(max(replicate_statistics))
    ordered = sorted(maxima)
    raw_critical = ordered[int(inference["quantile_rank_one_based"]) - 1]
    critical = max(0.0, raw_critical)
    for cell in cells:
        lower = float(cell["observed_mean"]) - critical * float(
            cell["standard_error"]
        )
        if not math.isfinite(lower):
            raise ContractError("simultaneous lower bound is non-finite")
        cell["simultaneous_lower_bound"] = lower
    return critical, raw_critical, _payload_sha256(maxima)


def _selection_gate(
    cells: Sequence[Mapping[str, object]], thresholds: Mapping[str, float]
) -> dict[str, object]:
    lookup = {str(cell["hypothesis_id"]): cell for cell in cells}
    reached: list[str] = []
    selected = "C0"
    blocked_candidate: str | None = None
    reasons: list[str] = []
    for candidate, predecessor in (("C1", "C0"), ("C2", "C1"), ("C3", "C2")):
        reached.append(candidate)
        candidate_reasons: list[str] = []
        for family in FAMILIES:
            primary = lookup[f"{family}:{candidate}-C0"]
            adjacent = lookup[f"{family}:{candidate}-{predecessor}"]
            if not float(primary["simultaneous_lower_bound"]) > thresholds["primary"]:
                candidate_reasons.append(f"{family}:primary_lower_bound")
            if not float(adjacent["simultaneous_lower_bound"]) > thresholds["adjacent"]:
                candidate_reasons.append(f"{family}:adjacent_lower_bound")
            if not float(primary["median"]) > 0.0:
                candidate_reasons.append(f"{family}:median")
            if not float(primary["trimmed_mean_10_percent"]) > 0.0:
                candidate_reasons.append(f"{family}:trimmed_mean")
            if not int(primary["wins"]) > int(primary["losses"]):
                candidate_reasons.append(f"{family}:wins_not_greater_than_losses")
        if candidate_reasons:
            blocked_candidate = candidate
            reasons.extend(candidate_reasons)
            break
        selected = candidate
    not_reached = [
        candidate for candidate in ("C1", "C2", "C3") if candidate not in reached
    ]
    return {
        "selected_candidate": selected,
        "reached_candidates": reached,
        "not_reached_candidates": not_reached,
        "blocked_candidate": blocked_candidate,
        "reasons": reasons,
    }


def _confirmation_gate(
    cells: Sequence[Mapping[str, object]],
    selected_candidate: str,
    thresholds: Mapping[str, float],
) -> list[str]:
    predecessor = {"C1": "C0", "C2": "C1", "C3": "C2"}[
        selected_candidate
    ]
    lookup = {str(cell["hypothesis_id"]): cell for cell in cells}
    reasons: list[str] = []
    for family in FAMILIES:
        primary = lookup[f"{family}:{selected_candidate}-C0"]
        adjacent = lookup[f"{family}:{selected_candidate}-{predecessor}"]
        if not float(primary["simultaneous_lower_bound"]) > thresholds["primary"]:
            reasons.append(f"{family}:primary_lower_bound")
        if not float(adjacent["simultaneous_lower_bound"]) > thresholds["adjacent"]:
            reasons.append(f"{family}:adjacent_lower_bound")
        if not float(primary["median"]) > 0.0:
            reasons.append(f"{family}:median")
        if not float(primary["trimmed_mean_10_percent"]) > 0.0:
            reasons.append(f"{family}:trimmed_mean")
        if not int(primary["wins"]) > int(primary["losses"]):
            reasons.append(f"{family}:wins_not_greater_than_losses")
    return reasons


def _evaluate(payload: dict[str, object], input_sha256: str) -> tuple[dict[str, object], int]:
    _require_keys(payload, TOP_LEVEL_KEYS, "input")
    if payload["schema"] != INPUT_SCHEMA:
        raise ContractError(f"schema must equal {INPUT_SCHEMA}")
    phase = _require_string(payload["phase"], "phase")
    if phase not in ("selection", "confirmation"):
        raise ContractError("phase must equal selection or confirmation")
    study_id = _require_string(payload["study_id"], "study_id")
    binding_fields = (
        "study_freeze_sha256",
        "phase_manifest_sha256",
        "matrix_receipt_sha256",
        "source_root_sha256",
        "metric_spec_sha256",
        "decision_spec_sha256",
    )
    bindings = {
        field: _require_sha256(payload[field], field) for field in binding_fields
    }
    if payload["effect_direction"] != "larger_is_better":
        raise ContractError("effect_direction must equal larger_is_better")
    cases = _validate_case_design(payload)
    seeds = _validate_seeds(payload)
    inference = _validate_inference(payload)
    thresholds = _validate_thresholds(payload)
    selection_binding: dict[str, object] | None
    confirmation_controls: dict[str, object] | None
    if phase == "selection":
        _validate_selection_phase(payload)
        candidates = CANDIDATES
        contrasts = SELECTION_CONTRASTS
        selection_binding = None
        confirmation_controls = None
        selected_candidate_from_binding = None
    else:
        (
            selected_candidate_from_binding,
            selection_binding,
            confirmation_controls,
        ) = _validate_confirmation_phase(payload)
        predecessor = {
            "C1": "C0",
            "C2": "C1",
            "C3": "C2",
        }[selected_candidate_from_binding]
        candidates = tuple(
            candidate
            for candidate in CANDIDATES
            if candidate in {"C0", predecessor, selected_candidate_from_binding}
        )
        contrasts = ((selected_candidate_from_binding, "C0"),)
        if predecessor != "C0":
            contrasts += ((selected_candidate_from_binding, predecessor),)
    rows = _validate_rows(payload, cases, seeds, candidates)
    cells = _build_cells(cases, seeds, rows, contrasts)
    try:
        critical, raw_critical, maxima_sha256 = _simultaneous_bounds(
            cells, inference
        )
    except ZeroStandardErrorHold as hold:
        public_cells = [
            {key: value for key, value in cell.items() if key != "case_differences"}
            for cell in cells
        ]
        hold_receipt: dict[str, object] = {
            "schema": RECEIPT_SCHEMA,
            "status": "HOLD_ZERO_STANDARD_ERROR",
            "phase": phase,
            "study_id": study_id,
            "input_sha256": input_sha256,
            **bindings,
            "source_sha256": _file_sha256(Path(__file__).resolve()),
            "effect_direction": "larger_is_better",
            "families": list(FAMILIES),
            "candidate_order": list(candidates),
            "case_count_by_family": {
                family: len(cases[family]) for family in FAMILIES
            },
            "matrix_row_count": len(rows),
            "expected_matrix_row_count": sum(
                len(cases[family]) for family in FAMILIES
            )
            * len(seeds)
            * len(candidates),
            "seeds": list(seeds),
            "thresholds": thresholds,
            "hypothesis_order": [str(cell["hypothesis_id"]) for cell in cells],
            "cells": public_cells,
            "inference": {
                **inference,
                "familywise_scope": "JOINT_ACROSS_BOTH_FAMILIES",
                "cluster_unit": "PAIRED_CASE",
                "seed_aggregation": "MEAN_WITHIN_CASE_ARM",
                "case_resampling": (
                    "WITH_REPLACEMENT_INDEPENDENT_BY_FAMILY_"
                    "SHARED_ACROSS_CELLS_WITHIN_FAMILY"
                ),
                "quantile_convention": (
                    "CEIL((1-ALPHA)*(B+1))_ORDER_STATISTIC"
                ),
                "critical_value_floor": 0.0,
                "zero_standard_error_action": "HOLD",
            },
            "zero_standard_error_hypotheses": list(hold.hypotheses),
            "simultaneous_coverage_certified": False,
            "selected_candidate": None,
            "reached_candidates": [],
            "not_reached_candidates": list(("C1", "C2", "C3"))
            if phase == "selection"
            else [],
            "blocked_candidate": None,
            "gate_reasons": ["zero_standard_error"],
            "selection_binding": selection_binding,
            "confirmation_control_bindings": confirmation_controls,
            "confirmation_control_bindings_validated": phase == "confirmation",
            "confirmation_control_bindings_scope": (
                "INPUT_DECLARATIONS_AND_HASH_BINDINGS_ONLY_NOT_AUTHENTICATION"
                if phase == "confirmation"
                else None
            ),
            "statistics_implementation_independent_from_mo_nco": True,
            "external_independence_claim_authorized": False,
            "scientific_independence": False,
            "formal_authority": False,
        }
        return hold_receipt, 3
    if phase == "selection":
        gate = _selection_gate(cells, thresholds)
        selected_candidate = str(gate["selected_candidate"])
        status = (
            "PASS_SELECTION"
            if selected_candidate != "C0"
            else "STOP_SELECTION_NO_CANDIDATE"
        )
        reached_candidates = gate["reached_candidates"]
        not_reached_candidates = gate["not_reached_candidates"]
        blocked_candidate = gate["blocked_candidate"]
        gate_reasons = gate["reasons"]
    else:
        assert selected_candidate_from_binding is not None
        selected_candidate = selected_candidate_from_binding
        gate_reasons = _confirmation_gate(cells, selected_candidate, thresholds)
        status = "PASS_CONFIRMATION" if not gate_reasons else "FAIL_CONFIRMATION"
        reached_candidates = [selected_candidate]
        not_reached_candidates = []
        blocked_candidate = None if not gate_reasons else selected_candidate
    exit_code = 0 if status in ("PASS_SELECTION", "PASS_CONFIRMATION") else 2
    public_cells = [
        {key: value for key, value in cell.items() if key != "case_differences"}
        for cell in cells
    ]
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "status": status,
        "phase": phase,
        "study_id": study_id,
        "input_sha256": input_sha256,
        **bindings,
        "source_sha256": _file_sha256(Path(__file__).resolve()),
        "effect_direction": "larger_is_better",
        "families": list(FAMILIES),
        "candidate_order": list(candidates),
        "case_count_by_family": {
            family: len(cases[family]) for family in FAMILIES
        },
        "matrix_row_count": len(rows),
        "expected_matrix_row_count": sum(
            len(cases[family]) for family in FAMILIES
        )
        * len(seeds)
        * len(candidates),
        "seeds": list(seeds),
        "thresholds": thresholds,
        "hypothesis_order": [str(cell["hypothesis_id"]) for cell in cells],
        "cells": public_cells,
        "inference": {
            **inference,
            "familywise_scope": "JOINT_ACROSS_BOTH_FAMILIES",
            "cluster_unit": "PAIRED_CASE",
            "seed_aggregation": "MEAN_WITHIN_CASE_ARM",
            "case_resampling": (
                "WITH_REPLACEMENT_INDEPENDENT_BY_FAMILY_"
                "SHARED_ACROSS_CELLS_WITHIN_FAMILY"
            ),
            "quantile_convention": (
                "CEIL((1-ALPHA)*(B+1))_ORDER_STATISTIC"
            ),
            "critical_value_floor": 0.0,
            "rng": "SHA256_COUNTER_U64_REJECTION_V1",
            "centering": "BOOTSTRAP_MEAN_MINUS_OBSERVED_MEAN",
            "studentization_denominator": "OBSERVED_CASE_CLUSTER_STANDARD_ERROR",
            "critical_value": critical,
            "raw_quantile_value": raw_critical,
            "bootstrap_maxima_sha256": maxima_sha256,
        },
        "simultaneous_coverage_certified": True,
        "selected_candidate": selected_candidate,
        "reached_candidates": reached_candidates,
        "not_reached_candidates": not_reached_candidates,
        "blocked_candidate": blocked_candidate,
        "gate_reasons": gate_reasons,
        "selection_binding": selection_binding,
        "confirmation_control_bindings": confirmation_controls,
        "confirmation_control_bindings_validated": phase == "confirmation",
        "confirmation_control_bindings_scope": (
            "INPUT_DECLARATIONS_AND_HASH_BINDINGS_ONLY_NOT_AUTHENTICATION"
            if phase == "confirmation"
            else None
        ),
        "statistics_implementation_independent_from_mo_nco": True,
        "external_independence_claim_authorized": False,
        "scientific_independence": False,
        "formal_authority": False,
    }
    return receipt, exit_code


def _write_exclusive_receipt(path: Path, receipt: dict[str, object]) -> None:
    if not path.parent.is_dir():
        raise ContractError("output parent directory must already exist")
    core = dict(receipt)
    core["receipt_payload_sha256"] = _payload_sha256(receipt)
    text = json.dumps(
        core,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")
    except FileExistsError as error:
        raise ContractError("output receipt already exists; exclusive create required") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    try:
        if output.exists():
            raise ContractError(
                "output receipt already exists; exclusive create required"
            )
        payload, input_sha256 = _load_strict_json(Path(args.input))
        receipt, exit_code = _evaluate(payload, input_sha256)
        _write_exclusive_receipt(output, receipt)
        print(
            json.dumps(
                {**receipt, "receipt_payload_sha256": _payload_sha256(receipt)},
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return exit_code
    except (ContractError, OSError) as error:
        print(
            json.dumps(
                {
                    "schema": "v21e3r1_simultaneous_evaluator_error_v1",
                    "status": "HOLD_INTEGRITY_ERROR",
                    "error": str(error),
                    "external_independence_claim_authorized": False,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
