from __future__ import annotations

"""Fail-closed prospective authorization evaluator for V21e3r1.

The evaluator verifies hash-bound receipts and derives whether selection,
confirmation, or formal-input materialization may be authorized.  It never
creates cases, executes a study, or upgrades a scientific claim.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, NoReturn, Sequence


SPEC_SCHEMA = "v21e3r1_prospective_authorization_gate_spec_v3"
RECEIPT_SCHEMA = "v21e3r1_prospective_authorization_receipt_v3"
HOLD_STATUS = "HOLD_PROSPECTIVE_AUTHORIZATION_PREREQUISITES"
REQUESTS = {
    "selection",
    "confirmation",
    "formal_input_materialization",
}
COMMON_BINDINGS = {
    "historical_preservation",
    "exact_504_diagnostic",
    "corrected_reanalysis",
    "successor_source_freeze",
    "successor_development_promotion",
    "same_implementation_coverage",
    "baseline_registry",
    "external_algorithm_replay",
    "simultaneous_inference_spec",
}
IDENTITY_KEYS = {
    "study_id",
    "candidate_id",
    "development_source_sha256",
    "successor_source_sha256",
    "successor_config_sha256",
    "operator_reanalysis_spec_sha256",
    "study_metric_spec_sha256",
    "simultaneous_inference_spec_sha256",
}
BINDING_KEYS = {"path", "sha256"}
FAMILIES = ("MOKP", "MOTSP")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
PROMOTION_SCHEMA = "v21e3r1_successor_development_factorial_evaluation_receipt_v2"
PROMOTION_PASS_STATUS = "PASS_SUCCESSOR_DEVELOPMENT_PROMOTION_ONLY"
PROMOTION_HOLD_STATUSES = {
    "HOLD_SUCCESSOR_DEVELOPMENT_PROMOTION_THRESHOLDS_NOT_MET",
    "HOLD_SUCCESSOR_DEVELOPMENT_PROMOTION_ZERO_STANDARD_ERROR",
}
FACTORIAL_INFERENCE_RELATIVE = PurePosixPath(
    "ijoc_submission_v21e3r1/development/"
    "V21E3R1_SUCCESSOR_DEVELOPMENT_FACTORIAL_INFERENCE_V1.json"
)
FACTORIAL_INFERENCE_SHA256 = (
    "5aa767bcc00c5ee8d220defa86b358d3e72a5849a99712a0f486159f1f032f3d"
)
FACTORIAL_METHOD = "one_sided_observed_se_max_t_paired_case_cluster_bootstrap_v1"
FACTORIAL_PARENT_PLAN_SHA256 = (
    "4408d10944cb6511e99ff0bd95ded256b9c230b91d8806a7bd5b962f10622886"
)
FACTORIAL_CASE_IDS = (
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
FACTORIAL_SEEDS = (31051, 31057, 31059)
FACTORIAL_LEGACY_SEARCH = "proposal_chain_v21e3r1_v1"
FACTORIAL_NEW_SEARCH = "post_commit_type_incumbent_anchor_development_v1"
FACTORIAL_LEGACY_NOVELTY = "legacy_retry_and_local_v21e3r1_v1"
FACTORIAL_NEW_NOVELTY = (
    "single_attempt_rotating_feasible_exchange_no_refill_development_v1"
)
FACTORIAL_ARMS = {
    "MOKP": (
        ("MOKP_LEGACY", FACTORIAL_LEGACY_SEARCH, FACTORIAL_LEGACY_NOVELTY),
        ("MOKP_ANCHOR_ONLY", FACTORIAL_NEW_SEARCH, FACTORIAL_LEGACY_NOVELTY),
        ("MOKP_NOVELTY_ONLY", FACTORIAL_LEGACY_SEARCH, FACTORIAL_NEW_NOVELTY),
        ("MOKP_BOTH", FACTORIAL_NEW_SEARCH, FACTORIAL_NEW_NOVELTY),
    ),
    "MOTSP": (
        ("MOTSP_LEGACY", FACTORIAL_LEGACY_SEARCH, FACTORIAL_LEGACY_NOVELTY),
        ("MOTSP_ANCHOR", FACTORIAL_NEW_SEARCH, FACTORIAL_LEGACY_NOVELTY),
    ),
}
FACTORIAL_INPUT_BINDING = {
    "schema": "v21e3r1_exposed_development_input_binding_v1",
    "case_ids": list(FACTORIAL_CASE_IDS),
    "manifest_sha256": {
        "ijoc_submission_v21e3/development_manifests_v1/config_manifest_development.json": (
            "d33ba2d83909af4fecff85f4663791b7c63b5ed56738a67f0eec6ccfd6336d4e"
        ),
        "ijoc_submission_v21e3/development_manifests_v1/reference_manifest_development.json": (
            "86336403c3e098f0e5022c796db1778552c0d92ca40d85953dc341eb534a4402"
        ),
        "ijoc_submission_v21e3/development_partitions_v1/case_manifest.json": (
            "1970361ba557aadd26de38aed008de11d11d158c797c00db1036cc4616cbdc8c"
        ),
    },
}
FACTORIAL_HYPOTHESES = (
    "MOKP:BOTH_MINUS_LEGACY:EAUC",
    "MOKP:ANCHOR_MAIN_EFFECT:EAUC",
    "MOKP:NOVELTY_MAIN_EFFECT:EAUC",
    "MOKP:NOVELTY_MAIN_EFFECT:CACHE_HIT_RATE_REDUCTION",
    "MOTSP:ANCHOR_MINUS_LEGACY:EAUC",
)
EXPECTED_SUCCESSOR_SEMANTIC_PARAMETERS = {
    "legacy_post_initialization_search_policy": "proposal_chain_v21e3r1_v1",
    "successor_post_initialization_search_policy": (
        "post_commit_type_incumbent_anchor_development_v1"
    ),
    "legacy_mokp_novelty_generation_policy": (
        "legacy_retry_and_local_v21e3r1_v1"
    ),
    "successor_mokp_novelty_generation_policy": (
        "single_attempt_rotating_feasible_exchange_no_refill_development_v1"
    ),
    "promoted_arm_by_family": {"MOKP": "MOKP_BOTH", "MOTSP": "MOTSP_ANCHOR"},
    "factorial_arm_ids_by_family": {
        "MOKP": [
            "MOKP_LEGACY",
            "MOKP_ANCHOR_ONLY",
            "MOKP_NOVELTY_ONLY",
            "MOKP_BOTH",
        ],
        "MOTSP": ["MOTSP_LEGACY", "MOTSP_ANCHOR"],
    },
}
PROMOTION_KEYS = {
    "schema", "status", "phase", "promotion_scope", "study_id", "candidate_id",
    "successor_source_sha256", "successor_config_sha256",
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
    "selection_confirmation_evaluator_reuse_reason", "selection_cases_materialized",
    "confirmation_cases_materialized", "formal_cases_materialized",
    "algorithm_execution_independence", "statistics_implementation_independence",
    "producer_independence", "custody_independence", "scientific_independence",
    "selection_authorized", "confirmation_authorized", "formal_study_authorized",
    "scientific_claim_authorized", "ijoc_submission_status", "receipt_payload_sha256",
}
PROMOTION_CELL_KEYS = {
    "hypothesis_id", "family", "metric", "role", "threshold", "case_count",
    "seed_count_per_case_arm", "observed_mean", "standard_error", "median",
    "wins_above_threshold", "ties_at_threshold", "losses_below_threshold",
    "simultaneous_lower_bound", "gate_passed",
}
FACTORIAL_PLAN_KEYS = {
    "schema", "status", "phase", "scientific_scope",
    "parent_v7_diagnostic_plan_path", "parent_v7_diagnostic_plan_sha256",
    "parent_v7_source_snapshot_sha256", "case_ids", "seeds", "arms_by_family",
    "charged_evaluation_budget", "checkpoint_period", "expected_rows",
    "row_timeout_seconds", "input_binding", "source_binding",
    "inference_spec_binding", "rows", "selection_entropy_release",
    "selection_cases_materialized", "confirmation_materialization",
    "formal_materialization", "implementation_independence",
    "scientific_independence", "selection_authorized", "confirmation_authorized",
    "formal_study_authorized", "scientific_claim_authorized", "ijoc_submission_status",
}
FACTORIAL_SOURCE_BINDING_KEYS = {
    "schema", "study_id", "candidate_id", "parent_development_source_sha256",
    "receipt_path", "receipt_sha256", "source_manifest_path",
    "source_manifest_sha256", "source_snapshot_sha256", "semantic_config_sha256",
    "study_metric_spec_sha256", "simultaneous_inference_spec_sha256",
    "factorial_inference_spec_path", "factorial_inference_spec_sha256",
}
FACTORIAL_INFERENCE_BINDING_KEYS = {
    "path", "sha256", "schema", "method", "bootstrap_samples",
    "bootstrap_seed", "familywise_alpha",
}
FACTORIAL_ROW_SPEC_KEYS = {
    "ordinal", "row_id", "case_id", "family", "size", "seed", "arm_id",
    "post_initialization_search_policy", "mokp_novelty_generation_policy",
    "case_artifact_path", "case_artifact_sha256",
}
FACTORIAL_AGGREGATE_KEYS = {
    "schema", "status", "phase", "scientific_scope", "plan_sha256",
    "row_count", "rows", "development_promotion_evaluated",
    "selection_authorized", "confirmation_authorized", "formal_study_authorized",
    "scientific_claim_authorized", "ijoc_submission_status",
}
FACTORIAL_AGGREGATE_ROW_KEYS = {
    "ordinal", "row_id", "case_id", "family", "seed", "arm_id",
    "exact_per_evaluation_left_continuous_hv_auc", "cache_hit_rate_per_attempt",
    "row_sha256", "trace_sha256", "terminal_receipt_sha256",
    "independent_metric_receipt_sha256",
}
FACTORIAL_MATRIX_RECEIPT_KEYS = {
    "schema", "status", "phase", "scientific_scope", "completed_rows",
    "expected_rows", "plan_sha256", "aggregate_sha256",
    "parent_v7_diagnostic_plan_sha256", "parent_v7_source_snapshot_sha256",
    "study_id", "candidate_id", "successor_source_sha256",
    "successor_config_sha256", "source_freeze_receipt_sha256",
    "source_manifest_sha256", "study_metric_spec_sha256",
    "simultaneous_inference_spec_sha256", "inference_spec_sha256",
    "development_promotion_evaluated", "selection_cases_materialized",
    "confirmation_cases_materialized", "formal_cases_materialized",
    "implementation_independence", "scientific_independence",
    "selection_authorized", "confirmation_authorized", "formal_study_authorized",
    "scientific_claim_authorized", "ijoc_submission_status", "receipt_payload_sha256",
}


class AuthorizationError(ValueError):
    """A gate specification or bound receipt failed integrity validation."""


def _fail(message: str) -> NoReturn:
    raise AuthorizationError(message)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON number is prohibited: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            _fail(f"duplicate JSON key is prohibited: {key!r}")
        output[key] = value
    return output


def _validate_json_tree(value: object, *, label: str) -> None:
    value_type = type(value)
    if value is None or value_type in {str, bool, int}:
        return
    if value_type is float:
        if not math.isfinite(value):
            _fail(f"{label} contains a non-finite float")
        return
    if value_type is list:
        for index, item in enumerate(value):
            _validate_json_tree(item, label=f"{label}[{index}]")
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str or not key:
                _fail(f"{label} contains an invalid object key")
            _validate_json_tree(item, label=f"{label}.{key}")
        return
    _fail(f"{label} contains prohibited type {value_type.__name__}")


def _parse_json(
    raw: bytes,
    *,
    label: str,
    canonical: bool,
    allow_canonical_pretty: bool = False,
    allow_canonical_newline: bool = False,
) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        _fail(f"{label} is not strict UTF-8 JSON: {error}")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except AuthorizationError:
        raise
    except (json.JSONDecodeError, RecursionError) as error:
        _fail(f"{label} is malformed JSON: {error}")
    if type(value) is not dict:
        _fail(f"{label} must be a JSON object")
    _validate_json_tree(value, label=label)
    canonical_encodings = {_canonical_bytes(value)}
    if allow_canonical_pretty:
        canonical_encodings.add(_canonical_pretty_bytes(value))
    if allow_canonical_newline:
        canonical_encodings.add(_canonical_bytes(value) + b"\n")
    if canonical and raw not in canonical_encodings:
        _fail(f"{label} is not canonical JSON")
    return value


def _exact_keys(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{label} must be an exact JSON object")
    observed = set(value)
    if observed != expected:
        _fail(
            f"{label} key set drifted; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    return value


def _string(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{label} must be an exact nonempty string")
    return value


def _identifier(value: object, *, label: str) -> str:
    result = _string(value, label=label)
    if IDENTIFIER_RE.fullmatch(result) is None:
        _fail(f"{label} is not a canonical identifier")
    return result


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an exact integer >= {minimum}")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be an exact JSON boolean")
    return value


def _float(value: object, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        _fail(f"{label} must be an exact finite JSON float")
    return value


def _number(value: object, *, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        _fail(f"{label} must be an exact finite JSON number")
    return float(value)


def _string_list(
    value: object,
    *,
    label: str,
    nonempty: bool = True,
    sorted_unique: bool = False,
) -> list[str]:
    if type(value) is not list or (nonempty and not value):
        _fail(f"{label} must be an exact {'nonempty ' if nonempty else ''}array")
    if any(type(item) is not str or not item for item in value):
        _fail(f"{label} must contain exact nonempty strings")
    if len(set(value)) != len(value):
        _fail(f"{label} contains duplicates")
    if sorted_unique and value != sorted(value):
        _fail(f"{label} must be sorted")
    return value


def _canonical_relative_path(value: object, *, label: str) -> PurePosixPath:
    raw = _string(value, label=label)
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if (
        "\\" in raw
        or any(ord(character) < 32 for character in raw)
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or posix.as_posix() != raw
        or any(part in {"", ".", ".."} for part in posix.parts)
        or ":" in posix.parts[0]
    ):
        _fail(f"{label} is not a canonical contained POSIX path")
    return posix


def _contained_file(root: Path, value: object, *, label: str) -> Path:
    relative = _canonical_relative_path(value, label=label)
    try:
        candidate = root.joinpath(*relative.parts).resolve(strict=True)
    except OSError as error:
        _fail(f"{label} does not resolve to an existing file: {error}")
    try:
        candidate.relative_to(root)
    except ValueError:
        _fail(f"{label} escapes evidence root")
    if not candidate.is_file():
        _fail(f"{label} is not a regular file")
    return candidate


def _contained_directory(root: Path, value: object, *, label: str) -> Path:
    relative = _canonical_relative_path(value, label=label)
    try:
        candidate = root.joinpath(*relative.parts).resolve(strict=True)
    except OSError as error:
        _fail(f"{label} does not resolve to an existing directory: {error}")
    try:
        candidate.relative_to(root)
    except ValueError:
        _fail(f"{label} escapes evidence root")
    if not candidate.is_dir():
        _fail(f"{label} is not a directory")
    return candidate


def _parse_canonical_newline_file(path: Path, *, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    value = _parse_json(
        raw,
        label=label,
        canonical=False,
    )
    if raw != _canonical_bytes(value) + b"\n":
        _fail(f"{label} is not canonical JSON plus one LF")
    return value


def _verified_artifact(
    root: Path,
    *,
    path_value: object,
    sha256_value: object,
    label: str,
) -> Path:
    path = _contained_file(root, path_value, label=f"{label}.path")
    expected_sha256 = _sha256(sha256_value, label=f"{label}.sha256")
    if _sha256_file(path) != expected_sha256:
        _fail(f"{label} SHA-256 disagrees with its file")
    return path


def _contained_output(root: Path, output: Path) -> Path:
    if output.exists():
        _fail("output receipt already exists; exclusive create required")
    try:
        parent = output.parent.resolve(strict=True)
    except OSError as error:
        _fail(f"output parent does not exist: {error}")
    try:
        parent.relative_to(root)
    except ValueError:
        _fail("output receipt escapes evidence root")
    return parent / output.name


def _identity(value: object) -> dict[str, str]:
    raw = _exact_keys(value, IDENTITY_KEYS, label="gate spec identity")
    result = {
        "study_id": _identifier(raw["study_id"], label="identity.study_id"),
        "candidate_id": _identifier(raw["candidate_id"], label="identity.candidate_id"),
    }
    for field in sorted(IDENTITY_KEYS - {"study_id", "candidate_id"}):
        result[field] = _sha256(raw[field], label=f"identity.{field}")
    return result


def _validate_source_manifest_binding(
    path: Path, *, expected_root_sha256: str, label: str
) -> tuple[int, int, dict[str, tuple[int, str]]]:
    raw = path.read_bytes()
    manifest = _parse_json(raw, label=label, canonical=True)
    manifest = _exact_keys(
        manifest,
        {"schema", "source_root_sha256", "entries"},
        label=label,
    )
    if manifest["schema"] != "v21e3r1_branch_replay_source_manifest_binding_v1":
        _fail(f"{label} schema drifted")
    declared_root = _sha256(
        manifest["source_root_sha256"], label=f"{label}.source_root_sha256"
    )
    entries = manifest["entries"]
    if type(entries) is not list or not entries:
        _fail(f"{label}.entries must be an exact nonempty array")
    observed_paths: list[str] = []
    source_bindings: dict[str, tuple[int, str]] = {}
    total_bytes = 0
    for index, value in enumerate(entries):
        entry = _exact_keys(
            value,
            {"path", "bytes", "sha256"},
            label=f"{label}.entries[{index}]",
        )
        relative = _canonical_relative_path(
            entry["path"], label=f"{label}.entries[{index}].path"
        ).as_posix()
        observed_paths.append(relative)
        byte_count = _integer(
            entry["bytes"], label=f"{label}.entries[{index}].bytes", minimum=1
        )
        digest = _sha256(
            entry["sha256"], label=f"{label}.entries[{index}].sha256"
        )
        total_bytes += byte_count
        source_bindings[relative] = (byte_count, digest)
    ordered = sorted(observed_paths, key=lambda item: (item.casefold(), item))
    if observed_paths != ordered:
        _fail(f"{label}.entries are not in canonical casefold path order")
    if len({item.casefold() for item in observed_paths}) != len(observed_paths):
        _fail(f"{label}.entries contain case-insensitive duplicate paths")
    computed_root = _sha256_bytes(_canonical_bytes(entries))
    if declared_root != computed_root:
        _fail(f"{label} canonical source-root digest drifted")
    if declared_root != expected_root_sha256:
        _fail(f"{label} source root disagrees with the gate identity")
    return len(entries), total_bytes, source_bindings


def _validate_study_metric_spec(
    value: Mapping[str, object], *, source_bindings: Mapping[str, tuple[int, str]]
) -> None:
    raw = _exact_keys(
        value,
        {
            "schema",
            "status",
            "metric_id",
            "effect_direction",
            "evaluation_axis",
            "objective_dimension",
            "normalization_contract",
            "reference_point",
            "archive_contract",
            "integration_contract",
            "primary_metric",
            "secondary_reporting_metrics",
            "seed_within_case_aggregation",
            "case_cluster_estimand",
            "row_crosscheck",
            "production_metric_source",
            "independent_metric_source",
            "practical_thresholds_bound_in_simultaneous_spec",
            "selection_authorized",
            "confirmation_authorized",
            "formal_study_authorized",
            "scientific_claim_authorized",
            "ijoc_submission_status",
            "receipt_payload_sha256",
        },
        label="study metric spec",
    )
    fixed = {
        "schema": "v21e3r1_study_metric_spec_v1",
        "status": "FROZEN_BEFORE_SELECTION",
        "metric_id": "normalized_left_continuous_hypervolume_auc",
        "effect_direction": "LARGER_IS_BETTER",
        "evaluation_axis": "CHARGED_EVALUATIONS",
        "normalization_contract": "CASE_FROZEN_LOWER_UPPER_AFFINE_TO_UNIT_SQUARE",
        "archive_contract": "ALL_CHARGED_EVALUATED_NONDOMINATED_ARCHIVE",
        "integration_contract": "EAUC=(1/B)*SUM_{b=1..B}HV(A_{b-1})",
        "primary_metric": "normalized_left_continuous_hypervolume_auc",
        "seed_within_case_aggregation": "ARITHMETIC_MEAN_WITHIN_CASE_ARM",
        "case_cluster_estimand": "MEAN_OF_PAIRED_CASE_DIFFERENCES",
        "ijoc_submission_status": "IJOC_HOLD",
    }
    if any(raw[field] != expected for field, expected in fixed.items()):
        _fail("study metric identity or estimand contract drifted")
    if _integer(raw["objective_dimension"], label="study metric objective_dimension") != 2:
        _fail("study metric objective dimension must equal two")
    reference = raw["reference_point"]
    if (
        type(reference) is not list
        or reference != [1.0, 1.0]
        or any(type(item) is not float for item in reference)
    ):
        _fail("study metric reference point must be exact float vector [1.0,1.0]")
    expected_secondary = [
        "terminal_hypervolume",
        "attempt_count",
        "physical_start_count",
        "charged_evaluation_count",
        "wall_time_seconds",
        "peak_rss_bytes",
    ]
    if raw["secondary_reporting_metrics"] != expected_secondary:
        _fail("study metric secondary-reporting list drifted")
    row_crosscheck = _exact_keys(
        raw["row_crosscheck"],
        {"required", "scope", "tolerance", "failure_policy"},
        label="study metric row_crosscheck",
    )
    if (
        not _boolean(row_crosscheck["required"], label="row_crosscheck.required")
        or row_crosscheck["scope"] != "EVERY_FORMAL_STUDY_ROW"
        or _float(row_crosscheck["tolerance"], label="row_crosscheck.tolerance") != 0.0
        or row_crosscheck["failure_policy"] != "HOLD_ON_ANY_MISMATCH"
    ):
        _fail("study metric row-crosscheck contract drifted")
    for field, expected_path in (
        ("production_metric_source", "mo_nco/pareto_ijoc_analysis.py"),
        (
            "independent_metric_source",
            "independent_reproduction/recompute_v21e3r1_successor_metrics.py",
        ),
    ):
        binding = _exact_keys(
            raw[field], {"path", "bytes", "sha256"}, label=f"study metric {field}"
        )
        path = _canonical_relative_path(
            binding["path"], label=f"study metric {field}.path"
        ).as_posix()
        if path != expected_path:
            _fail(f"study metric {field} path drifted")
        byte_count = _integer(
            binding["bytes"], label=f"study metric {field}.bytes", minimum=1
        )
        digest = _sha256(binding["sha256"], label=f"study metric {field}.sha256")
        if source_bindings.get(path) != (byte_count, digest):
            _fail(f"study metric {field} disagrees with successor source manifest")
    if not _boolean(
        raw["practical_thresholds_bound_in_simultaneous_spec"],
        label="study metric practical-threshold delegation",
    ):
        _fail("study metric practical thresholds are not delegated to simultaneous spec")
    for field in (
        "selection_authorized",
        "confirmation_authorized",
        "formal_study_authorized",
        "scientific_claim_authorized",
    ):
        if _boolean(raw[field], label=f"study metric {field}"):
            _fail(f"study metric {field} must remain false")
    payload_sha = _sha256(
        raw["receipt_payload_sha256"], label="study metric receipt_payload_sha256"
    )
    core = dict(raw)
    core.pop("receipt_payload_sha256")
    if payload_sha != _sha256_bytes(_canonical_bytes(core)):
        _fail("study metric receipt payload digest drifted")


def _validate_historical(receipt: Mapping[str, object]) -> bool:
    raw = _exact_keys(
        receipt,
        {
            "schema",
            "status",
            "historical_row_count_each",
            "archive_member_count_each",
            "unchanged_member_count",
            "v4_removed_member",
            "v6_replacement_member",
            "identities",
            "historical_outputs_modified",
            "implementation_independence",
            "scientific_independence",
            "selection_authorized",
            "formal_authorized",
            "submission_status",
        },
        label="historical preservation receipt",
    )
    if raw["schema"] != "v21e3r1_v4_v6_historical_preservation_receipt_v1":
        _fail("historical preservation receipt schema drifted")
    identities = raw["identities"]
    if type(identities) is not dict or len(identities) != 6:
        _fail("historical receipt must bind exactly six V4/V6 release artifacts")
    for name, item in identities.items():
        _string(name, label="historical identity name")
        entry = _exact_keys(item, {"bytes", "sha256"}, label=f"identity {name}")
        _integer(entry["bytes"], label=f"identity {name}.bytes", minimum=1)
        _sha256(entry["sha256"], label=f"identity {name}.sha256")
    protected_false = (
        not _boolean(raw["historical_outputs_modified"], label="historical_outputs_modified")
        and not _boolean(raw["implementation_independence"], label="implementation_independence")
        and not _boolean(raw["scientific_independence"], label="scientific_independence")
        and not _boolean(raw["selection_authorized"], label="selection_authorized")
        and not _boolean(raw["formal_authorized"], label="formal_authorized")
    )
    return bool(
        raw["status"] == "PASS_HISTORICAL_V4_V6_IDENTITY_AND_RELATIONSHIP"
        and _integer(raw["historical_row_count_each"], label="historical_row_count_each") == 108
        and _integer(raw["archive_member_count_each"], label="archive_member_count_each") == 701
        and _integer(raw["unchanged_member_count"], label="unchanged_member_count") == 700
        and _string(raw["v4_removed_member"], label="v4_removed_member")
        and _string(raw["v6_replacement_member"], label="v6_replacement_member")
        and protected_false
        and raw["submission_status"] == "IJOC_HOLD"
    )


def _validate_diagnostic(
    receipt: Mapping[str, object], identity: Mapping[str, str]
) -> bool:
    raw = _exact_keys(
        receipt,
        {
            "schema",
            "status",
            "scientific_scope",
            "matrix_mode",
            "completed_rows",
            "expected_rows",
            "plan_sha256",
            "source_snapshot_sha256",
            "aggregate_sha256",
            "selection_entropy_release",
            "confirmation_materialization",
            "formal_materialization",
        },
        label="exact-504 diagnostic receipt",
    )
    for field in ("plan_sha256", "source_snapshot_sha256", "aggregate_sha256"):
        _sha256(raw[field], label=f"diagnostic.{field}")
    return bool(
        raw["schema"] == "v21e3r1_exposed_development_diagnostic_receipt_v2"
        and raw["status"] == "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY"
        and raw["scientific_scope"]
        == "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE"
        and raw["matrix_mode"] == "FULL_504"
        and _integer(raw["completed_rows"], label="diagnostic.completed_rows") == 504
        and _integer(raw["expected_rows"], label="diagnostic.expected_rows") == 504
        and raw["source_snapshot_sha256"] == identity["development_source_sha256"]
        and raw["selection_entropy_release"] == "PROHIBITED"
        and raw["confirmation_materialization"] == "PROHIBITED"
        and raw["formal_materialization"] == "PROHIBITED"
    )


def _validate_reanalysis(
    receipt: Mapping[str, object],
    identity: Mapping[str, str],
    diagnostic_receipt: Mapping[str, object],
    diagnostic_sha256: str,
    receipt_path: Path,
) -> bool:
    raw = _exact_keys(
        receipt,
        {
            "schema",
            "status",
            "scientific_scope",
            "matrix_mode",
            "development_source_sha256",
            "plan_sha256",
            "diagnostic_receipt_sha256",
            "diagnostic_aggregate_sha256",
            "completed_rows",
            "expected_rows",
            "charged_evaluations_per_row",
            "evaluation_charged_evaluations_sum",
            "attempt_charged_evaluations_sum",
            "legacy_operator_charged_evaluations_sum",
            "operator_charge_double_count_corrected",
            "all_rows_reanalyzed",
            "original_artifacts_modified",
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
        },
        label="corrected reanalysis receipt",
    )
    for field in (
        "development_source_sha256",
        "plan_sha256",
        "diagnostic_receipt_sha256",
        "diagnostic_aggregate_sha256",
        "rows_sha256",
        "aggregate_sha256",
        "metric_spec_sha256",
        "metric_source_sha256",
    ):
        _sha256(raw[field], label=f"reanalysis.{field}")
    artifacts = {
        _verified_artifact(
            receipt_path.parent,
            path_value=raw[path_field],
            sha256_value=raw[sha_field],
            label=f"reanalysis.{artifact}",
        )
        for artifact, path_field, sha_field in (
            ("rows", "rows_path", "rows_sha256"),
            ("aggregate", "aggregate_path", "aggregate_sha256"),
            ("metric_spec", "metric_spec_path", "metric_spec_sha256"),
            ("metric_source", "metric_source_path", "metric_source_sha256"),
        )
    }
    if len(artifacts) != 4:
        _fail("corrected reanalysis artifacts must resolve to four distinct files")
    if receipt_path in artifacts:
        _fail("corrected reanalysis receipt cannot be one of its own bound artifacts")
    completed = _integer(raw["completed_rows"], label="reanalysis.completed_rows")
    expected = _integer(raw["expected_rows"], label="reanalysis.expected_rows")
    per_row = _integer(
        raw["charged_evaluations_per_row"],
        label="reanalysis.charged_evaluations_per_row",
        minimum=1,
    )
    expected_sum = expected * per_row
    return bool(
        raw["schema"]
        == "v21e3r1_corrected_operator_accounting_reanalysis_receipt_v1"
        and raw["status"]
        == "PASS_CORRECTED_REANALYSIS_EXACT_504_DEVELOPMENT_ONLY"
        and raw["scientific_scope"]
        == "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE"
        and raw["matrix_mode"] == "FULL_504"
        and completed == expected == 504
        and per_row == 2000
        and _integer(
            raw["evaluation_charged_evaluations_sum"],
            label="reanalysis.evaluation_charged_evaluations_sum",
        )
        == expected_sum
        and _integer(
            raw["attempt_charged_evaluations_sum"],
            label="reanalysis.attempt_charged_evaluations_sum",
        )
        == expected_sum
        and _integer(
            raw["legacy_operator_charged_evaluations_sum"],
            label="reanalysis.legacy_operator_charged_evaluations_sum",
        )
        == 2 * expected_sum
        and _boolean(
            raw["operator_charge_double_count_corrected"],
            label="reanalysis.operator_charge_double_count_corrected",
        )
        and _boolean(raw["all_rows_reanalyzed"], label="reanalysis.all_rows_reanalyzed")
        and not _boolean(
            raw["original_artifacts_modified"],
            label="reanalysis.original_artifacts_modified",
        )
        and raw["development_source_sha256"] == identity["development_source_sha256"]
        and raw["plan_sha256"] == diagnostic_receipt["plan_sha256"]
        and raw["diagnostic_receipt_sha256"] == diagnostic_sha256
        and raw["diagnostic_aggregate_sha256"]
        == diagnostic_receipt["aggregate_sha256"]
        and raw["rows_path"] == "operator_accounting.rows.jsonl"
        and raw["aggregate_path"] == "operator_accounting.aggregate.json"
        and raw["metric_spec_path"]
        == "metric/v21e3r1_operator_accounting_reanalysis_spec_v1.json"
        and raw["metric_source_path"]
        == "source/reanalyze_v21e3r1_operator_accounting.py"
        and raw["metric_spec_sha256"]
        == identity["operator_reanalysis_spec_sha256"]
        and not _boolean(
            raw["implementation_independence"],
            label="reanalysis.implementation_independence",
        )
        and not _boolean(
            raw["algorithm_execution_independence"],
            label="reanalysis.algorithm_execution_independence",
        )
        and not _boolean(
            raw["scientific_independence"],
            label="reanalysis.scientific_independence",
        )
        and not _boolean(
            raw["selection_authorized"], label="reanalysis.selection_authorized"
        )
        and not _boolean(
            raw["confirmation_authorized"],
            label="reanalysis.confirmation_authorized",
        )
        and not _boolean(
            raw["formal_study_authorized"],
            label="reanalysis.formal_study_authorized",
        )
        and raw["publication_status"] == "IJOC_HOLD"
    )


def _validate_source_freeze(
    receipt: Mapping[str, object], identity: Mapping[str, str], receipt_path: Path
) -> bool:
    raw = _exact_keys(
        receipt,
        {
            "schema",
            "status",
            "study_id",
            "candidate_id",
            "parent_development_source_sha256",
            "source_snapshot_sha256",
            "source_manifest_sha256",
            "semantic_config_sha256",
            "study_metric_spec_sha256",
            "simultaneous_inference_spec_sha256",
            "source_entry_count",
            "source_total_bytes",
            "all_source_files_verified",
            "source_frozen",
            "selection_cases_materialized",
            "confirmation_cases_materialized",
            "formal_cases_materialized",
            "source_archive_materialized",
            "source_archive_path",
            "source_archive_sha256",
            "source_archive_scope",
            "implementation_independence",
            "scientific_independence",
            "selection_authorized",
            "confirmation_authorized",
            "formal_study_authorized",
            "scientific_claim_authorized",
            "public_redistribution_authorized",
            "ijoc_submission_status",
            "receipt_payload_sha256",
        },
        label="successor source-freeze receipt",
    )
    if raw["schema"] != "v21e3r1_successor_source_freeze_receipt_v2":
        _fail("successor source-freeze receipt schema drifted")
    payload_sha = _sha256(
        raw["receipt_payload_sha256"], label="source.receipt_payload_sha256"
    )
    core = dict(raw)
    core.pop("receipt_payload_sha256")
    if payload_sha != _sha256_bytes(_canonical_bytes(core)):
        _fail("successor source-freeze receipt payload digest drifted")
    _identifier(raw["study_id"], label="source.study_id")
    _identifier(raw["candidate_id"], label="source.candidate_id")
    for field in (
        "parent_development_source_sha256",
        "source_snapshot_sha256",
        "source_manifest_sha256",
        "semantic_config_sha256",
        "study_metric_spec_sha256",
        "simultaneous_inference_spec_sha256",
    ):
        _sha256(raw[field], label=f"source.{field}")

    siblings = {
        "source_manifest": _verified_artifact(
            receipt_path.parent,
            path_value="source.manifest.json",
            sha256_value=raw["source_manifest_sha256"],
            label="source.source_manifest",
        ),
        "semantic_config": _verified_artifact(
            receipt_path.parent,
            path_value="semantic.config.json",
            sha256_value=raw["semantic_config_sha256"],
            label="source.semantic_config",
        ),
        "study_metric_spec": _verified_artifact(
            receipt_path.parent,
            path_value="study.metric-spec.json",
            sha256_value=raw["study_metric_spec_sha256"],
            label="source.study_metric_spec",
        ),
        "simultaneous_spec": _verified_artifact(
            receipt_path.parent,
            path_value="simultaneous-inference.spec.json",
            sha256_value=raw["simultaneous_inference_spec_sha256"],
            label="source.simultaneous_spec",
        ),
    }
    if len(set(siblings.values())) != 4 or receipt_path in siblings.values():
        _fail("successor source-freeze siblings must be four distinct non-receipt files")
    entry_count, total_bytes, source_bindings = _validate_source_manifest_binding(
        siblings["source_manifest"],
        expected_root_sha256=str(raw["source_snapshot_sha256"]),
        label="successor source manifest",
    )
    semantic = _parse_json(
        siblings["semantic_config"].read_bytes(),
        label="successor semantic config",
        canonical=True,
    )
    semantic = _exact_keys(
        semantic,
        {"schema", "study_id", "candidate_id", "parameters"},
        label="successor semantic config",
    )
    if type(semantic["parameters"]) is not dict or not semantic["parameters"]:
        _fail("successor semantic config parameters must be a nonempty object")
    if (
        semantic["schema"] != "v21e3r1_successor_semantic_config_v1"
        or semantic["study_id"] != identity["study_id"]
        or semantic["candidate_id"] != identity["candidate_id"]
    ):
        _fail("successor semantic config identity drifted")
    if semantic["parameters"] != EXPECTED_SUCCESSOR_SEMANTIC_PARAMETERS:
        _fail("successor semantic config policy contract drifted")
    study_metric = _parse_json(
        siblings["study_metric_spec"].read_bytes(),
        label="study metric spec",
        canonical=True,
    )
    _validate_study_metric_spec(study_metric, source_bindings=source_bindings)
    _parse_json(
        siblings["simultaneous_spec"].read_bytes(),
        label="source-freeze simultaneous spec",
        canonical=True,
    )

    archive_materialized = _boolean(
        raw["source_archive_materialized"], label="source.source_archive_materialized"
    )
    if archive_materialized:
        if raw["source_archive_path"] != "successor-source.zip":
            _fail("source archive path must use the fixed sibling name")
        archive_sha = _sha256(
            raw["source_archive_sha256"], label="source.source_archive_sha256"
        )
        _verified_artifact(
            receipt_path.parent,
            path_value=raw["source_archive_path"],
            sha256_value=archive_sha,
            label="source.source_archive",
        )
        if (
            raw["source_archive_scope"]
            != "SOURCE_INVENTORY_ONLY_INTERNAL_CUSTODY_NO_REDISTRIBUTION_AUTHORITY"
        ):
            _fail("source archive scope drifted")
    elif not (
        raw["source_archive_path"] is None
        and raw["source_archive_sha256"] is None
        and raw["source_archive_scope"] == "NOT_MATERIALIZED"
    ):
        _fail("non-materialized source archive fields must be exact null/NOT_MATERIALIZED")

    return bool(
        raw["status"] == "PASS_SUCCESSOR_SOURCE_AND_CONFIG_FREEZE_ENGINEERING_ONLY"
        and raw["study_id"] == identity["study_id"]
        and raw["candidate_id"] == identity["candidate_id"]
        and raw["parent_development_source_sha256"]
        == identity["development_source_sha256"]
        and raw["source_snapshot_sha256"] == identity["successor_source_sha256"]
        and raw["semantic_config_sha256"] == identity["successor_config_sha256"]
        and raw["study_metric_spec_sha256"] == identity["study_metric_spec_sha256"]
        and raw["simultaneous_inference_spec_sha256"]
        == identity["simultaneous_inference_spec_sha256"]
        and _integer(raw["source_entry_count"], label="source.source_entry_count", minimum=1)
        == entry_count
        and _integer(raw["source_total_bytes"], label="source.source_total_bytes", minimum=1)
        == total_bytes
        and _boolean(raw["all_source_files_verified"], label="source.all_source_files_verified")
        and _boolean(raw["source_frozen"], label="source.source_frozen")
        and not _boolean(raw["selection_cases_materialized"], label="source.selection_cases_materialized")
        and not _boolean(raw["confirmation_cases_materialized"], label="source.confirmation_cases_materialized")
        and not _boolean(raw["formal_cases_materialized"], label="source.formal_cases_materialized")
        and not _boolean(raw["implementation_independence"], label="source.implementation_independence")
        and not _boolean(raw["scientific_independence"], label="source.scientific_independence")
        and not _boolean(raw["selection_authorized"], label="source.selection_authorized")
        and not _boolean(raw["confirmation_authorized"], label="source.confirmation_authorized")
        and not _boolean(raw["formal_study_authorized"], label="source.formal_study_authorized")
        and not _boolean(raw["scientific_claim_authorized"], label="source.scientific_claim_authorized")
        and not _boolean(raw["public_redistribution_authorized"], label="source.public_redistribution_authorized")
        and raw["ijoc_submission_status"] == "IJOC_HOLD"
    )


def _validate_factorial_plan_design(
    plan: Mapping[str, object],
) -> list[dict[str, Any]]:
    _canonical_relative_path(
        plan["parent_v7_diagnostic_plan_path"],
        label="promotion factorial parent V7 plan path",
    )
    if plan["parent_v7_diagnostic_plan_sha256"] != FACTORIAL_PARENT_PLAN_SHA256:
        _fail("promotion factorial parent V7 plan SHA-256 drifted")
    if plan["case_ids"] != list(FACTORIAL_CASE_IDS):
        _fail("promotion factorial exposed-development case boundary drifted")
    if plan["seeds"] != list(FACTORIAL_SEEDS):
        _fail("promotion factorial seed boundary drifted")
    expected_arms = {
        family: [arm_id for arm_id, _search, _novelty in arms]
        for family, arms in FACTORIAL_ARMS.items()
    }
    if plan["arms_by_family"] != expected_arms:
        _fail("promotion factorial arm boundary drifted")
    if plan["input_binding"] != FACTORIAL_INPUT_BINDING:
        _fail("promotion factorial frozen input binding drifted")
    _integer(
        plan["row_timeout_seconds"],
        label="promotion factorial row_timeout_seconds",
        minimum=1,
    )

    rows = plan["rows"]
    if type(rows) is not list or len(rows) != 108:
        _fail("promotion factorial plan row cardinality drifted")
    expected_rows: list[dict[str, object]] = []
    for case_id in FACTORIAL_CASE_IDS:
        family = "MOKP" if "-mokp-" in case_id else "MOTSP"
        size = next(size for size in (100, 200, 500) if f"-n{size}-" in case_id)
        for seed in FACTORIAL_SEEDS:
            for arm_id, search_policy, novelty_policy in FACTORIAL_ARMS[family]:
                expected_rows.append(
                    {
                        "ordinal": len(expected_rows) + 1,
                        "row_id": f"{case_id}__seed-{seed}__arm-{arm_id.lower()}",
                        "case_id": case_id,
                        "family": family,
                        "size": size,
                        "seed": seed,
                        "arm_id": arm_id,
                        "post_initialization_search_policy": search_policy,
                        "mokp_novelty_generation_policy": novelty_policy,
                        "case_artifact_path": (
                            "ijoc_submission_v21e3/development_partitions_v1/"
                            f"instances/{case_id}.json"
                        ),
                    }
                )
    validated: list[dict[str, Any]] = []
    for index, (value, expected) in enumerate(zip(rows, expected_rows, strict=True)):
        row = _exact_keys(
            value,
            FACTORIAL_ROW_SPEC_KEYS,
            label=f"promotion factorial plan rows[{index}]",
        )
        for field, expected_value in expected.items():
            if row[field] != expected_value or type(row[field]) is not type(
                expected_value
            ):
                _fail(f"promotion factorial row design drifted: {index + 1}/{field}")
        _sha256(
            row["case_artifact_sha256"],
            label=f"promotion factorial rows[{index}].case_artifact_sha256",
        )
        validated.append(row)
    return validated


def _validate_factorial_aggregate(
    aggregate_path: Path,
    *,
    plan_sha256: str,
    plan_rows: Sequence[Mapping[str, object]],
) -> None:
    aggregate = _exact_keys(
        _parse_canonical_newline_file(
            aggregate_path, label="promotion factorial aggregate"
        ),
        FACTORIAL_AGGREGATE_KEYS,
        label="promotion factorial aggregate",
    )
    expected_scalars = {
        "schema": "v21e3r1_successor_development_factorial_aggregate_v2",
        "status": "PASS_EXACT_108_SUCCESSOR_DEVELOPMENT_FACTORIAL_ENGINEERING_ONLY",
        "phase": "development",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "plan_sha256": plan_sha256,
        "row_count": 108,
        "development_promotion_evaluated": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_study_authorized": False,
        "scientific_claim_authorized": False,
        "ijoc_submission_status": "IJOC_HOLD",
    }
    for field, expected in expected_scalars.items():
        if aggregate[field] != expected or type(aggregate[field]) is not type(expected):
            _fail(f"promotion factorial aggregate field drifted: {field}")
    rows = aggregate["rows"]
    if type(rows) is not list or len(rows) != 108:
        _fail("promotion factorial aggregate row cardinality drifted")
    identity_fields = ("ordinal", "row_id", "case_id", "family", "seed", "arm_id")
    for index, (value, plan_row) in enumerate(zip(rows, plan_rows, strict=True)):
        row = _exact_keys(
            value,
            FACTORIAL_AGGREGATE_ROW_KEYS,
            label=f"promotion factorial aggregate rows[{index}]",
        )
        for field in identity_fields:
            if row[field] != plan_row[field] or type(row[field]) is not type(
                plan_row[field]
            ):
                _fail(
                    f"promotion factorial aggregate/plan row identity drifted: "
                    f"{index + 1}/{field}"
                )
        for field in (
            "exact_per_evaluation_left_continuous_hv_auc",
            "cache_hit_rate_per_attempt",
        ):
            metric = _number(
                row[field], label=f"promotion factorial aggregate rows[{index}].{field}"
            )
            if not 0.0 <= metric <= 1.0:
                _fail(f"promotion factorial aggregate metric outside [0,1]: {index + 1}/{field}")
        for field in (
            "row_sha256",
            "trace_sha256",
            "terminal_receipt_sha256",
            "independent_metric_receipt_sha256",
        ):
            _sha256(
                row[field], label=f"promotion factorial aggregate rows[{index}].{field}"
            )


def _validate_development_promotion(
    receipt: Mapping[str, object],
    identity: Mapping[str, str],
    source_receipt: Mapping[str, object],
    source_receipt_sha256: str,
    source_receipt_path: Path,
    promotion_receipt_path: Path,
    root: Path,
) -> bool:
    raw = _exact_keys(receipt, PROMOTION_KEYS, label="successor development promotion")
    if promotion_receipt_path.read_bytes() != _canonical_bytes(raw) + b"\n":
        _fail("successor development promotion is not canonical JSON plus one LF")
    payload_sha = _sha256(
        raw["receipt_payload_sha256"], label="promotion.receipt_payload_sha256"
    )
    core = dict(raw)
    core.pop("receipt_payload_sha256")
    if payload_sha != _sha256_bytes(_canonical_bytes(core)):
        _fail("successor development promotion payload digest drifted")

    expected_scalars: dict[str, object] = {
        "schema": PROMOTION_SCHEMA,
        "phase": "development",
        "promotion_scope": "SUCCESSOR_DEVELOPMENT_PROMOTION_ONLY_HASH_BOUND_PRODUCER_RECEIPT_NO_PROSPECTIVE_108_ROW_RECOMPUTATION_NO_SCIENTIFIC_CLAIM",
        "method": FACTORIAL_METHOD,
        "familywise_alpha": 0.05,
        "bootstrap_samples": 9999,
        "bootstrap_seed": 2026082301,
        "rng_protocol": "SHA256_COUNTER_U64_REJECTION_V1",
        "rng_domain": "v21e3r1-successor-development-factorial-bootstrap-v1",
        "quantile_convention": "CEIL((1-ALPHA)*(B+1))_ORDER_STATISTIC",
        "cluster_unit": "PAIRED_CASE",
        "seed_aggregation": "MEAN_WITHIN_CASE_ARM",
        "familywise_scope": "JOINT_ACROSS_ALL_FIVE_DEVELOPMENT_PROMOTION_HYPOTHESES",
        "matrix_row_count": 108,
        "expected_matrix_row_count": 108,
        "hypothesis_order": list(FACTORIAL_HYPOTHESES),
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
        "inference_spec_sha256": FACTORIAL_INFERENCE_SHA256,
    }
    for field, expected in expected_scalars.items():
        if raw[field] != expected or type(raw[field]) is not type(expected):
            _fail(f"successor development promotion field drifted: {field}")
    _identifier(raw["study_id"], label="promotion.study_id")
    _identifier(raw["candidate_id"], label="promotion.candidate_id")
    digest_fields = (
        "successor_source_sha256",
        "successor_config_sha256",
        "source_freeze_receipt_sha256",
        "source_manifest_sha256",
        "study_metric_spec_sha256",
        "simultaneous_inference_spec_sha256",
        "matrix_plan_sha256",
        "matrix_receipt_sha256",
        "row_evidence_replay_sha256",
        "inference_spec_sha256",
    )
    for field in digest_fields:
        _sha256(raw[field], label=f"promotion.{field}")
    expected_identity = {
        "study_id": identity["study_id"],
        "candidate_id": identity["candidate_id"],
        "successor_source_sha256": identity["successor_source_sha256"],
        "successor_config_sha256": identity["successor_config_sha256"],
        "source_freeze_receipt_sha256": source_receipt_sha256,
        "source_manifest_sha256": source_receipt["source_manifest_sha256"],
        "study_metric_spec_sha256": identity["study_metric_spec_sha256"],
        "simultaneous_inference_spec_sha256": identity[
            "simultaneous_inference_spec_sha256"
        ],
    }
    for field, expected in expected_identity.items():
        if raw[field] != expected:
            _fail(f"successor development promotion {field} disagrees with gate identity")

    inference_path = _contained_file(
        root,
        FACTORIAL_INFERENCE_RELATIVE.as_posix(),
        label="successor factorial inference specification",
    )
    if _sha256_file(inference_path) != FACTORIAL_INFERENCE_SHA256:
        _fail("successor factorial inference specification SHA-256 drifted")
    inference = _parse_canonical_newline_file(
        inference_path,
        label="successor factorial inference specification",
    )
    if (
        inference.get("schema")
        != "v21e3r1_successor_development_factorial_inference_spec_v1"
        or inference.get("status")
        != "FROZEN_PROSPECTIVELY_BEFORE_SUCCESSOR_FACTORIAL_EXECUTION"
        or inference.get("phase") != "development"
        or inference.get("promotion_scope")
        != "SUCCESSOR_DEVELOPMENT_PROMOTION_ONLY_NO_SCIENTIFIC_CLAIM"
        or inference.get("method") != FACTORIAL_METHOD
        or inference.get("selection_cases_materialized") is not False
    ):
        _fail("successor factorial inference specification boundary drifted")
    for field in (
        "familywise_alpha",
        "bootstrap_samples",
        "bootstrap_seed",
        "rng_protocol",
        "rng_domain",
        "quantile_convention",
        "cluster_unit",
        "seed_aggregation",
        "familywise_scope",
    ):
        if raw[field] != inference.get(field) or type(raw[field]) is not type(
            inference.get(field)
        ):
            _fail(f"promotion/inference specification binding drifted: {field}")

    matrix = _contained_directory(
        root, raw["matrix_directory"], label="promotion.matrix_directory"
    )
    matrix_relative = _canonical_relative_path(
        raw["matrix_directory"], label="promotion.matrix_directory"
    )
    plan_path = _contained_file(
        root,
        (matrix_relative / "factorial.plan.json").as_posix(),
        label="promotion factorial plan",
    )
    matrix_receipt_path = _contained_file(
        root,
        (matrix_relative / "factorial.receipt.json").as_posix(),
        label="promotion factorial matrix receipt",
    )
    aggregate_path = _contained_file(
        root,
        (matrix_relative / "factorial.aggregate.json").as_posix(),
        label="promotion factorial aggregate",
    )
    if plan_path.parent != matrix or matrix_receipt_path.parent != matrix:
        _fail("promotion matrix fixed siblings escaped the matrix directory")
    if _sha256_file(plan_path) != raw["matrix_plan_sha256"]:
        _fail("promotion matrix plan SHA-256 drifted")
    if _sha256_file(matrix_receipt_path) != raw["matrix_receipt_sha256"]:
        _fail("promotion matrix receipt SHA-256 drifted")

    plan = _exact_keys(
        _parse_canonical_newline_file(plan_path, label="promotion factorial plan"),
        FACTORIAL_PLAN_KEYS,
        label="promotion factorial plan",
    )
    plan_scalars = {
        "schema": "v21e3r1_successor_development_factorial_plan_v2",
        "status": "FROZEN_EXACT_108_SUCCESSOR_DEVELOPMENT_FACTORIAL",
        "phase": "development",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "parent_v7_source_snapshot_sha256": identity["development_source_sha256"],
        "charged_evaluation_budget": 2000,
        "checkpoint_period": 200,
        "expected_rows": 108,
        "selection_entropy_release": "PROHIBITED",
        "selection_cases_materialized": False,
        "confirmation_materialization": "PROHIBITED",
        "formal_materialization": "PROHIBITED",
        "implementation_independence": False,
        "scientific_independence": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_study_authorized": False,
        "scientific_claim_authorized": False,
        "ijoc_submission_status": "IJOC_HOLD",
    }
    for field, expected in plan_scalars.items():
        if plan[field] != expected or type(plan[field]) is not type(expected):
            _fail(f"promotion factorial plan field drifted: {field}")
    plan_rows = _validate_factorial_plan_design(plan)
    source_binding = _exact_keys(
        plan["source_binding"],
        FACTORIAL_SOURCE_BINDING_KEYS,
        label="promotion factorial source binding",
    )
    source_binding_expected = {
        "schema": "v21e3r1_successor_factorial_source_binding_v2",
        "study_id": identity["study_id"],
        "candidate_id": identity["candidate_id"],
        "parent_development_source_sha256": identity["development_source_sha256"],
        "receipt_sha256": source_receipt_sha256,
        "source_manifest_sha256": source_receipt["source_manifest_sha256"],
        "source_snapshot_sha256": identity["successor_source_sha256"],
        "semantic_config_sha256": identity["successor_config_sha256"],
        "study_metric_spec_sha256": identity["study_metric_spec_sha256"],
        "simultaneous_inference_spec_sha256": identity[
            "simultaneous_inference_spec_sha256"
        ],
        "factorial_inference_spec_path": FACTORIAL_INFERENCE_RELATIVE.as_posix(),
        "factorial_inference_spec_sha256": FACTORIAL_INFERENCE_SHA256,
    }
    for field, expected in source_binding_expected.items():
        if source_binding[field] != expected:
            _fail(f"promotion factorial source binding drifted: {field}")
    bound_source_receipt = _verified_artifact(
        root,
        path_value=source_binding["receipt_path"],
        sha256_value=source_binding["receipt_sha256"],
        label="promotion factorial source receipt binding",
    )
    if bound_source_receipt != source_receipt_path:
        _fail("promotion source-freeze path binding drifted")
    manifest_path = _verified_artifact(
        root,
        path_value=source_binding["source_manifest_path"],
        sha256_value=source_binding["source_manifest_sha256"],
        label="promotion source manifest binding",
    )
    if manifest_path.parent != source_receipt_path.parent:
        _fail("promotion source manifest is not a source-freeze sibling")
    inference_binding = _exact_keys(
        plan["inference_spec_binding"],
        FACTORIAL_INFERENCE_BINDING_KEYS,
        label="promotion factorial inference binding",
    )
    expected_inference_binding = {
        "path": FACTORIAL_INFERENCE_RELATIVE.as_posix(),
        "sha256": FACTORIAL_INFERENCE_SHA256,
        "schema": "v21e3r1_successor_development_factorial_inference_spec_v1",
        "method": FACTORIAL_METHOD,
        "bootstrap_samples": 9999,
        "bootstrap_seed": 2026082301,
        "familywise_alpha": 0.05,
    }
    if inference_binding != expected_inference_binding:
        _fail("promotion factorial plan inference binding drifted")

    matrix_receipt = _exact_keys(
        _parse_canonical_newline_file(
            matrix_receipt_path, label="promotion factorial matrix receipt"
        ),
        FACTORIAL_MATRIX_RECEIPT_KEYS,
        label="promotion factorial matrix receipt",
    )
    matrix_payload = _sha256(
        matrix_receipt["receipt_payload_sha256"],
        label="promotion matrix receipt payload SHA-256",
    )
    matrix_core = dict(matrix_receipt)
    matrix_core.pop("receipt_payload_sha256")
    if matrix_payload != _sha256_bytes(_canonical_bytes(matrix_core)):
        _fail("promotion factorial matrix receipt payload digest drifted")
    matrix_expected = {
        "schema": "v21e3r1_successor_development_factorial_receipt_v2",
        "status": "PASS_EXACT_108_SUCCESSOR_DEVELOPMENT_FACTORIAL_ENGINEERING_ONLY",
        "phase": "development",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "completed_rows": 108,
        "expected_rows": 108,
        "plan_sha256": raw["matrix_plan_sha256"],
        "parent_v7_diagnostic_plan_sha256": FACTORIAL_PARENT_PLAN_SHA256,
        "parent_v7_source_snapshot_sha256": identity["development_source_sha256"],
        "study_id": identity["study_id"],
        "candidate_id": identity["candidate_id"],
        "successor_source_sha256": identity["successor_source_sha256"],
        "successor_config_sha256": identity["successor_config_sha256"],
        "source_freeze_receipt_sha256": source_receipt_sha256,
        "source_manifest_sha256": source_receipt["source_manifest_sha256"],
        "study_metric_spec_sha256": identity["study_metric_spec_sha256"],
        "simultaneous_inference_spec_sha256": identity[
            "simultaneous_inference_spec_sha256"
        ],
        "inference_spec_sha256": FACTORIAL_INFERENCE_SHA256,
        "development_promotion_evaluated": False,
        "selection_cases_materialized": False,
        "confirmation_cases_materialized": False,
        "formal_cases_materialized": False,
        "implementation_independence": False,
        "scientific_independence": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_study_authorized": False,
        "scientific_claim_authorized": False,
        "ijoc_submission_status": "IJOC_HOLD",
    }
    for field, expected in matrix_expected.items():
        if matrix_receipt[field] != expected or type(matrix_receipt[field]) is not type(
            expected
        ):
            _fail(f"promotion factorial matrix receipt field drifted: {field}")
    if matrix_receipt["aggregate_sha256"] != _sha256_file(aggregate_path):
        _fail("promotion factorial aggregate SHA-256 drifted")
    _validate_factorial_aggregate(
        aggregate_path,
        plan_sha256=raw["matrix_plan_sha256"],
        plan_rows=plan_rows,
    )

    statuses = {PROMOTION_PASS_STATUS: True, **{status: False for status in PROMOTION_HOLD_STATUSES}}
    status = _string(raw["status"], label="promotion.status")
    if status not in statuses:
        _fail("successor development promotion status drifted")
    declared_gate = _boolean(
        raw["development_promotion_gate_passed"],
        label="promotion.development_promotion_gate_passed",
    )
    if declared_gate is not statuses[status]:
        _fail("successor development promotion status/gate relationship drifted")
    cells = raw["cells"]
    hypotheses = inference.get("hypotheses")
    if (
        type(cells) is not list
        or len(cells) != len(FACTORIAL_HYPOTHESES)
        or type(hypotheses) is not list
        or len(hypotheses) != len(FACTORIAL_HYPOTHESES)
    ):
        _fail("successor development promotion hypothesis cardinality drifted")
    cell_gates: list[bool] = []
    for index, value in enumerate(cells):
        cell = _exact_keys(
            value, PROMOTION_CELL_KEYS, label=f"promotion.cells[{index}]"
        )
        hypothesis = hypotheses[index]
        if type(hypothesis) is not dict:
            _fail(f"promotion inference hypotheses[{index}] must be an object")
        expected_cell = {
            "hypothesis_id": FACTORIAL_HYPOTHESES[index],
            "family": hypothesis.get("family"),
            "metric": hypothesis.get("metric"),
            "role": hypothesis.get("role"),
            "threshold": hypothesis.get("threshold"),
            "case_count": 6,
            "seed_count_per_case_arm": 3,
        }
        for field, expected in expected_cell.items():
            if cell[field] != expected or type(cell[field]) is not type(expected):
                _fail(f"promotion cell {index} binding drifted: {field}")
        for field in ("observed_mean", "standard_error", "median"):
            _float(cell[field], label=f"promotion.cells[{index}].{field}")
        if _float(
            cell["standard_error"], label=f"promotion.cells[{index}].standard_error"
        ) < 0.0:
            _fail("promotion cell standard error must be nonnegative")
        wtl = [
            _integer(cell[field], label=f"promotion.cells[{index}].{field}")
            for field in (
                "wins_above_threshold",
                "ties_at_threshold",
                "losses_below_threshold",
            )
        ]
        if sum(wtl) != 6:
            _fail("promotion cell W/T/L counts disagree with case_count")
        gate = _boolean(
            cell["gate_passed"], label=f"promotion.cells[{index}].gate_passed"
        )
        cell_gates.append(gate)

    zero = _string_list(
        raw["zero_standard_error_hypotheses"],
        label="promotion.zero_standard_error_hypotheses",
        nonempty=False,
    )
    if any(item not in FACTORIAL_HYPOTHESES for item in zero):
        _fail("promotion zero-standard-error hypothesis drifted")
    expected_zero = [
        str(cell["hypothesis_id"])
        for cell in cells
        if cell["standard_error"] == 0.0
    ]
    if zero != expected_zero:
        _fail("promotion zero-standard-error witness drifted")
    reasons = _string_list(
        raw["gate_reasons"], label="promotion.gate_reasons", nonempty=False
    )
    if zero:
        if (
            status != "HOLD_SUCCESSOR_DEVELOPMENT_PROMOTION_ZERO_STANDARD_ERROR"
            or raw["critical_value"] is not None
            or raw["bootstrap_maxima_sha256"] is not None
            or any(cell["simultaneous_lower_bound"] is not None for cell in cells)
            or any(cell_gates)
            or reasons != [f"zero_standard_error:{item}" for item in zero]
        ):
            _fail("promotion zero-standard-error HOLD payload drifted")
    else:
        critical = _float(raw["critical_value"], label="promotion.critical_value")
        if critical < 0.0:
            _fail("promotion critical value must be nonnegative")
        _sha256(raw["bootstrap_maxima_sha256"], label="promotion.bootstrap_maxima_sha256")
        for index, cell in enumerate(cells):
            lower = _float(
                cell["simultaneous_lower_bound"],
                label=f"promotion.cells[{index}].simultaneous_lower_bound",
            )
            expected_lower = float(cell["observed_mean"]) - critical * float(
                cell["standard_error"]
            )
            if lower != expected_lower:
                _fail("promotion cell simultaneous lower bound drifted")
            expected_gate = lower > float(cell["threshold"])
            if cell_gates[index] is not expected_gate:
                _fail("promotion cell lower-bound/gate relationship drifted")
        failed = [
            FACTORIAL_HYPOTHESES[index]
            for index, gate in enumerate(cell_gates)
            if not gate
        ]
        expected_status = (
            PROMOTION_PASS_STATUS
            if not failed
            else "HOLD_SUCCESSOR_DEVELOPMENT_PROMOTION_THRESHOLDS_NOT_MET"
        )
        expected_reasons = [
            f"simultaneous_lower_bound_not_above_threshold:{item}" for item in failed
        ]
        if status != expected_status or reasons != expected_reasons:
            _fail("promotion threshold decision payload drifted")
    if declared_gate is not all(cell_gates):
        _fail("promotion aggregate gate disagrees with its hypothesis cells")
    return declared_gate


def _validate_inner_same_implementation(
    receipt: Mapping[str, object],
    identity: Mapping[str, str],
    diagnostic_receipt: Mapping[str, object],
    diagnostic_receipt_sha256: str,
    receipt_path: Path,
) -> bool:
    raw = _exact_keys(
        receipt,
        {
            "schema",
            "status",
            "scientific_scope",
            "matrix_mode",
            "completed_rows",
            "expected_rows",
            "exact_full_504_coverage",
            "diagnostic_plan_sha256",
            "diagnostic_receipt_sha256",
            "diagnostic_aggregate_sha256",
            "source_snapshot_sha256",
            "source_manifest_path",
            "source_manifest_sha256",
            "row_order_rule",
            "row_seals",
            "row_seals_sha256",
            "verification_jobs",
            "verification_jobs_observed",
            "row_timeout_seconds",
            "parallel_execution_semantics",
            "implementation_independence",
            "scientific_independence",
            "third_party_replication",
            "selection_authorized",
            "confirmation_authorized",
            "formal_authorized",
            "selection_entropy_release",
            "confirmation_materialization",
            "formal_materialization",
            "runtime_efficiency_claims",
            "scientific_performance_claims",
            "ijoc_submission_status",
        },
        label="same-implementation coverage receipt",
    )
    for field in (
        "diagnostic_plan_sha256",
        "diagnostic_receipt_sha256",
        "diagnostic_aggregate_sha256",
        "source_snapshot_sha256",
        "source_manifest_sha256",
        "row_seals_sha256",
    ):
        _sha256(raw[field], label=f"same implementation.{field}")
    source_manifest_path = _verified_artifact(
        receipt_path.parent,
        path_value=raw["source_manifest_path"],
        sha256_value=raw["source_manifest_sha256"],
        label="same implementation.source_manifest",
    )
    if source_manifest_path == receipt_path:
        _fail("same-implementation receipt cannot be its own source manifest")
    _validate_source_manifest_binding(
        source_manifest_path,
        expected_root_sha256=identity["development_source_sha256"],
        label="same-implementation development source manifest",
    )
    row_seals = raw["row_seals"]
    if type(row_seals) is not list or len(row_seals) != 504:
        _fail("same-implementation coverage must bind exactly 504 row seals")
    row_ids: set[str] = set()
    for ordinal, seal_value in enumerate(row_seals):
        seal = _exact_keys(
            seal_value,
            {
                "row_id",
                "plan_ordinal",
                "coverage_completed_marker_sha256",
                "diagnostic_completed_marker_sha256",
                "diagnostic_trace_sha256",
                "branch_replay_receipt_sha256",
            },
            label=f"same implementation row_seals[{ordinal}]",
        )
        row_id = _string(seal["row_id"], label=f"same implementation row {ordinal}.id")
        if row_id in row_ids:
            _fail("same-implementation row seals contain duplicate row IDs")
        row_ids.add(row_id)
        if _integer(
            seal["plan_ordinal"],
            label=f"same implementation row {ordinal}.plan_ordinal",
        ) != ordinal + 1:
            _fail("same-implementation row seals are not in frozen plan order")
        for field in (
            "coverage_completed_marker_sha256",
            "diagnostic_completed_marker_sha256",
            "diagnostic_trace_sha256",
            "branch_replay_receipt_sha256",
        ):
            _sha256(seal[field], label=f"same implementation row {ordinal}.{field}")
    if raw["row_seals_sha256"] != _sha256_bytes(_canonical_bytes(row_seals)):
        _fail("same-implementation row-seal digest drifted")
    observed_jobs = raw["verification_jobs_observed"]
    if type(observed_jobs) is not list or not observed_jobs:
        _fail("same-implementation verification_jobs_observed must be nonempty")
    if (
        any(type(value) is not int or value < 1 for value in observed_jobs)
        or len(set(observed_jobs)) != len(observed_jobs)
        or observed_jobs != sorted(observed_jobs)
    ):
        _fail("same-implementation observed jobs must be sorted unique positive integers")
    return bool(
        raw["schema"]
        == "v21e3r1_branch_replay_coverage_receipt_v1"
        and raw["status"]
        == "PASS_SAME_IMPLEMENTATION_BRANCH_REPLAY_EXACT_504_DEVELOPMENT_ONLY"
        and raw["scientific_scope"]
        == "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE"
        and raw["matrix_mode"] == "FULL_504"
        and _integer(raw["completed_rows"], label="same implementation completed_rows")
        == 504
        and _integer(raw["expected_rows"], label="same implementation expected_rows")
        == 504
        and _boolean(
            raw["exact_full_504_coverage"], label="same implementation exact_full_504"
        )
        and raw["diagnostic_plan_sha256"] == diagnostic_receipt["plan_sha256"]
        and raw["diagnostic_receipt_sha256"] == diagnostic_receipt_sha256
        and raw["diagnostic_aggregate_sha256"]
        == diagnostic_receipt["aggregate_sha256"]
        and raw["source_snapshot_sha256"] == identity["development_source_sha256"]
        and raw["source_manifest_path"] == "source.manifest.json"
        and raw["row_order_rule"] == "FROZEN_DIAGNOSTIC_PLAN_CASE_SEED_ARM_ORDER"
        and _integer(
            raw["verification_jobs"], label="same implementation verification_jobs", minimum=1
        )
        >= 1
        and _integer(
            raw["row_timeout_seconds"],
            label="same implementation row_timeout_seconds",
            minimum=1,
        )
        >= 1
        and raw["parallel_execution_semantics"]
        == "VERIFICATION_ONLY_PLAN_ORDERED_FINALIZATION_NO_RUNTIME_OR_PERFORMANCE_CLAIM"
        and not _boolean(
            raw["implementation_independence"],
            label="same implementation independence",
        )
        and not _boolean(
            raw["scientific_independence"],
            label="same implementation scientific independence",
        )
        and not _boolean(
            raw["third_party_replication"],
            label="same implementation third party replication",
        )
        and not _boolean(raw["selection_authorized"], label="same implementation selection")
        and not _boolean(
            raw["confirmation_authorized"], label="same implementation confirmation"
        )
        and not _boolean(raw["formal_authorized"], label="same implementation formal")
        and raw["selection_entropy_release"] == "PROHIBITED"
        and raw["confirmation_materialization"] == "PROHIBITED"
        and raw["formal_materialization"] == "PROHIBITED"
        and not _boolean(
            raw["runtime_efficiency_claims"], label="same implementation runtime claims"
        )
        and not _boolean(
            raw["scientific_performance_claims"],
            label="same implementation performance claims",
        )
        and raw["ijoc_submission_status"] == "IJOC_HOLD"
    )


_RECOVERY_A6_KEYS = {
    "runtime_authority",
    "scientific_authority",
    "selection_authority",
    "confirmation_authority",
    "formal_study_authority",
    "publication_status",
}
_RECOVERY_FALSE_AUTHORITY_FIELDS = {
    "implementation_independence",
    "algorithm_execution_independence",
    "scientific_independence",
    "runtime_authority",
    "scientific_authority",
    "selection_authority",
    "confirmation_authority",
    "formal_study_authority",
    "selection_authorized",
    "confirmation_authorized",
    "formal_authorized",
    "third_party_replication",
    "runtime_efficiency_claims",
    "scientific_performance_claims",
}
_RECOVERY_OUTER_RECEIPT_KEYS = {
    "schema",
    "status",
    "expected_rows",
    "jobs",
    "row_timeout_seconds",
    "wall_time_seconds",
    "process_isolation",
    "plan_sha256",
    "plan_payload_sha256",
    "provenance_payload_sha256",
    "diagnostic_binding_sha256",
    "preflight_receipt_sha256",
    "preflight_seal_sha256",
    "execution_claim_path",
    "execution_claim_sha256",
    "execution_claim_payload_sha256",
    "inner_receipt_path",
    "inner_receipt_sha256",
    "inner_receipt_payload_sha256",
    "runtime_identities_before",
    "runtime_identities_after",
    "same_implementation_only",
    "implementation_independence",
    "algorithm_execution_independence",
    "scientific_independence",
    "receipt_payload_sha256",
} | _RECOVERY_A6_KEYS
_RECOVERY_OUTER_SEAL_KEYS = {
    "schema",
    "status",
    "receipt_path",
    "receipt_sha256",
    "receipt_payload_sha256",
    "plan_sha256",
    "provenance_payload_sha256",
    "preflight_receipt_sha256",
    "inner_receipt_sha256",
    "seal_payload_sha256",
} | _RECOVERY_A6_KEYS
_RECOVERY_PLAN_KEYS = {
    "schema",
    "status",
    "project_root",
    "diagnostic_root",
    "output_root",
    "expected_rows",
    "jobs",
    "row_timeout_seconds",
    "provenance_payload_sha256",
    "diagnostic_binding_sha256",
    "preflight_binding",
    "runtime_identities",
    "implementation_independence",
    "scientific_independence",
    "plan_payload_sha256",
} | _RECOVERY_A6_KEYS
_RECOVERY_PREFLIGHT_RECEIPT_KEYS = {
    "schema",
    "status",
    "plan_path",
    "plan_sha256",
    "plan_payload_sha256",
    "selected_row",
    "charged_evaluation_budget",
    "verification_jobs",
    "row_timeout_seconds",
    "wall_time_seconds",
    "process_isolation",
    "branch_replay_receipt_path",
    "branch_replay_receipt_sha256",
    "branch_replay_payload_sha256",
    "diagnostic_provenance_payload_sha256",
    "diagnostic_binding_sha256",
    "runtime_identities_before",
    "runtime_identities_after",
    "preflight_required_for_full_coverage",
    "operational_only",
    "implementation_independence",
    "scientific_independence",
    "receipt_payload_sha256",
} | _RECOVERY_A6_KEYS
_RECOVERY_PREFLIGHT_SEAL_KEYS = {
    "schema",
    "status",
    "receipt_path",
    "receipt_sha256",
    "receipt_payload_sha256",
    "plan_sha256",
    "seal_payload_sha256",
} | _RECOVERY_A6_KEYS
_RECOVERY_PREFLIGHT_PLAN_KEYS = {
    "schema",
    "status",
    "project_root",
    "diagnostic_root",
    "output_root",
    "selected_row",
    "charged_evaluation_budget",
    "verification_jobs",
    "row_timeout_seconds",
    "preflight_required_for_full_coverage",
    "diagnostic_provenance_payload_sha256",
    "diagnostic_binding_sha256",
    "runtime_identities",
    "implementation_independence",
    "scientific_independence",
    "plan_payload_sha256",
} | _RECOVERY_A6_KEYS
_RECOVERY_CLAIM_KEYS = {
    "schema",
    "status",
    "execution_number",
    "plan_sha256",
    "provenance_payload_sha256",
    "preflight_receipt_sha256",
    "preflight_seal_sha256",
    "runtime_identities",
    "jobs",
    "row_timeout_seconds",
    "inner_resume",
    "implementation_independence",
    "scientific_independence",
    "claim_payload_sha256",
} | _RECOVERY_A6_KEYS
_RECOVERY_RUNTIME_KEYS = {
    "outer_runner_path",
    "outer_runner_sha256",
    "inner_runner_path",
    "inner_runner_sha256",
    "python_path",
    "python_sha256",
    "python_version",
}


def _validate_recovery_hold_authority(
    value: Mapping[str, object], *, label: str, require_a6: bool = False
) -> None:
    if require_a6 and not _RECOVERY_A6_KEYS.issubset(value):
        _fail(f"{label} omits the frozen HOLD authority fields")
    for field in _RECOVERY_FALSE_AUTHORITY_FIELDS & set(value):
        if _boolean(value[field], label=f"{label}.{field}"):
            _fail(f"{label} expands authority at {field}")
    for field in ("publication_status", "ijoc_submission_status"):
        if field in value and value[field] != "IJOC_HOLD":
            _fail(f"{label} publication status drifted")


def _validate_recovery_bound_payload(
    value: Mapping[str, object],
    *,
    label: str,
    digest_field: str,
    schema: str,
    status: str,
    expected_keys: set[str] | None = None,
    require_a6: bool = False,
) -> str:
    raw = (
        _exact_keys(value, expected_keys, label=label)
        if expected_keys is not None
        else dict(value)
    )
    if raw.get("schema") != schema or raw.get("status") != status:
        _fail(f"{label} schema/status drifted")
    declared = _sha256(raw.get(digest_field), label=f"{label}.{digest_field}")
    core = dict(raw)
    del core[digest_field]
    if declared != _sha256_bytes(_canonical_bytes(core)):
        _fail(f"{label} payload digest drifted")
    _validate_recovery_hold_authority(raw, label=label, require_a6=require_a6)
    return declared


def _load_stable_recovery_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    try:
        first = path.read_bytes()
    except OSError as error:
        _fail(f"cannot read {label}: {error}")
    value = _parse_json(
        first,
        label=label,
        canonical=True,
        allow_canonical_pretty=True,
        allow_canonical_newline=True,
    )
    try:
        second = path.read_bytes()
    except OSError as error:
        _fail(f"cannot re-read {label}: {error}")
    if first != second:
        _fail(f"{label} changed while being validated")
    return value, _sha256_bytes(first)


def _contained_declared_recovery_file(
    root: Path, value: object, *, label: str
) -> Path:
    raw = _string(value, label=label)
    candidate = Path(raw)
    if not candidate.is_absolute():
        return _contained_file(root, raw, label=label)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        _fail(f"{label} does not resolve to an existing file: {error}")
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(f"{label} escapes evidence root")
    if not resolved.is_file():
        _fail(f"{label} is not a regular file")
    return resolved


def _validate_recovery_runtime_identities(value: object) -> dict[str, str]:
    raw = _exact_keys(value, _RECOVERY_RUNTIME_KEYS, label="recovery runtime identities")
    outer_path = Path(
        _string(raw["outer_runner_path"], label="runtime.outer_runner_path")
    ).resolve(strict=True)
    inner_path = Path(
        _string(raw["inner_runner_path"], label="runtime.inner_runner_path")
    ).resolve(strict=True)
    python_path = Path(
        _string(raw["python_path"], label="runtime.python_path")
    ).resolve(strict=True)
    expected_outer = Path(__file__).with_name(
        "run_v21e3r1_recovery_bound_same_implementation_branch_replay_coverage.py"
    ).resolve(strict=True)
    expected_inner = Path(__file__).with_name(
        "run_v21e3r1_same_implementation_branch_replay_coverage.py"
    ).resolve(strict=True)
    if outer_path != expected_outer or inner_path != expected_inner:
        _fail("recovery runtime runner path drifted")
    if python_path != Path(sys.executable).resolve(strict=True):
        _fail("recovery runtime Python path drifted")
    for path, field in (
        (outer_path, "outer_runner_sha256"),
        (inner_path, "inner_runner_sha256"),
        (python_path, "python_sha256"),
    ):
        declared = _sha256(raw[field], label=f"runtime.{field}")
        if _sha256_file(path) != declared:
            _fail(f"recovery runtime {field} drifted")
    if raw["python_version"] != sys.version:
        _fail("recovery runtime Python version drifted")
    return {field: str(raw[field]) for field in _RECOVERY_RUNTIME_KEYS}


def _validate_recovery_preflight(
    *,
    evidence_root: Path,
    binding_value: object,
    provenance_payload_sha256: str,
    diagnostic_binding: Mapping[str, object],
    diagnostic_binding_sha256: str,
    runtime_identities: Mapping[str, str],
) -> dict[str, object]:
    binding = _exact_keys(
        binding_value,
        {
            "schema",
            "status",
            "selected_row_id",
            "charged_evaluation_budget",
            "verification_jobs",
            "row_timeout_seconds",
            "receipt_path",
            "receipt_sha256",
            "receipt_payload_sha256",
            "seal_path",
            "seal_sha256",
            "seal_payload_sha256",
            "plan_sha256",
            "diagnostic_provenance_payload_sha256",
            "diagnostic_binding_sha256",
        }
        | _RECOVERY_A6_KEYS,
        label="recovery preflight binding",
    )
    if (
        binding["schema"] != "v21e3r1_recovery_bound_n500_preflight_binding_v1"
        or binding["status"]
        != "PASS_SEALED_N500_PREFLIGHT_REQUIRED_FOR_FULL_COVERAGE"
    ):
        _fail("recovery preflight binding schema/status drifted")
    _validate_recovery_hold_authority(
        binding, label="recovery preflight binding", require_a6=True
    )
    receipt_path = _contained_declared_recovery_file(
        evidence_root, binding["receipt_path"], label="recovery preflight receipt path"
    )
    seal_path = _contained_declared_recovery_file(
        evidence_root, binding["seal_path"], label="recovery preflight seal path"
    )
    if receipt_path.parent != seal_path.parent:
        _fail("recovery preflight receipt/seal roots disagree")
    receipt, receipt_raw_sha = _load_stable_recovery_json(
        receipt_path, label="recovery preflight receipt"
    )
    receipt_payload_sha = _validate_recovery_bound_payload(
        receipt,
        label="recovery preflight receipt",
        digest_field="receipt_payload_sha256",
        schema="v21e3r1_recovery_bound_n500_operational_preflight_receipt_v1",
        status="PASS_N500_FULL_BUDGET_OPERATIONAL_PREFLIGHT_ONLY",
        expected_keys=_RECOVERY_PREFLIGHT_RECEIPT_KEYS,
        require_a6=True,
    )
    seal, seal_raw_sha = _load_stable_recovery_json(
        seal_path, label="recovery preflight success seal"
    )
    seal_payload_sha = _validate_recovery_bound_payload(
        seal,
        label="recovery preflight success seal",
        digest_field="seal_payload_sha256",
        schema="v21e3r1_recovery_bound_n500_preflight_success_seal_v1",
        status="SEALED_N500_OPERATIONAL_PREFLIGHT_SUCCESS_RECEIPT",
        expected_keys=_RECOVERY_PREFLIGHT_SEAL_KEYS,
        require_a6=True,
    )
    rows = diagnostic_binding.get("rows")
    selected = receipt["selected_row"]
    if type(rows) is not list or type(selected) is not dict:
        _fail("recovery preflight selected diagnostic row is absent")
    matches = [row for row in rows if type(row) is dict and row.get("row_id") == selected.get("row_id")]
    if (
        len(matches) != 1
        or matches[0] != selected
        or selected.get("size") != 500
        or selected.get("budget") != 2000
        or receipt["charged_evaluation_budget"] != 2000
        or receipt["verification_jobs"] != 1
        or receipt["row_timeout_seconds"] != 2400
        or receipt["preflight_required_for_full_coverage"] is not True
        or receipt["operational_only"] is not True
        or receipt["diagnostic_provenance_payload_sha256"]
        != provenance_payload_sha256
        or receipt["diagnostic_binding_sha256"] != diagnostic_binding_sha256
        or receipt["runtime_identities_before"] != runtime_identities
        or receipt["runtime_identities_after"] != runtime_identities
    ):
        _fail("recovery preflight receipt invocation binding drifted")
    preflight_root = receipt_path.parent
    plan_path = _contained_file(
        preflight_root, receipt["plan_path"], label="recovery preflight plan path"
    )
    plan, plan_raw_sha = _load_stable_recovery_json(
        plan_path, label="recovery preflight plan"
    )
    plan_payload_sha = _validate_recovery_bound_payload(
        plan,
        label="recovery preflight plan",
        digest_field="plan_payload_sha256",
        schema="v21e3r1_recovery_bound_n500_operational_preflight_plan_v1",
        status="FROZEN_N500_FULL_BUDGET_OPERATIONAL_PREFLIGHT_ONLY",
        expected_keys=_RECOVERY_PREFLIGHT_PLAN_KEYS,
        require_a6=True,
    )
    if (
        receipt["plan_sha256"] != plan_raw_sha
        or receipt["plan_payload_sha256"] != plan_payload_sha
        or plan["selected_row"] != selected
        or plan["charged_evaluation_budget"] != 2000
        or plan["verification_jobs"] != 1
        or plan["row_timeout_seconds"] != 2400
        or plan["preflight_required_for_full_coverage"] is not True
        or plan["diagnostic_provenance_payload_sha256"]
        != provenance_payload_sha256
        or plan["diagnostic_binding_sha256"] != diagnostic_binding_sha256
        or plan["runtime_identities"] != runtime_identities
    ):
        _fail("recovery preflight plan binding drifted")
    branch_path = _contained_file(
        preflight_root,
        receipt["branch_replay_receipt_path"],
        label="recovery preflight branch receipt path",
    )
    branch, branch_raw_sha = _load_stable_recovery_json(
        branch_path, label="recovery preflight branch receipt"
    )
    branch_payload_sha = _validate_recovery_bound_payload(
        branch,
        label="recovery preflight branch receipt",
        digest_field="receipt_payload_sha256",
        schema="v21e3r1_same_implementation_branch_replay_v1",
        status="PASS_SAME_IMPLEMENTATION_FULL_BRANCH_REEXECUTION",
    )
    if (
        receipt["branch_replay_receipt_sha256"] != branch_raw_sha
        or receipt["branch_replay_payload_sha256"] != branch_payload_sha
        or seal["receipt_path"] != receipt_path.name
        or seal["receipt_sha256"] != receipt_raw_sha
        or seal["receipt_payload_sha256"] != receipt_payload_sha
        or seal["plan_sha256"] != plan_raw_sha
    ):
        _fail("recovery preflight receipt/seal chain drifted")
    derived: dict[str, object] = {
        "schema": "v21e3r1_recovery_bound_n500_preflight_binding_v1",
        "status": "PASS_SEALED_N500_PREFLIGHT_REQUIRED_FOR_FULL_COVERAGE",
        "selected_row_id": selected["row_id"],
        "charged_evaluation_budget": 2000,
        "verification_jobs": 1,
        "row_timeout_seconds": 2400,
        "receipt_path": receipt_path.as_posix(),
        "receipt_sha256": receipt_raw_sha,
        "receipt_payload_sha256": receipt_payload_sha,
        "seal_path": seal_path.as_posix(),
        "seal_sha256": seal_raw_sha,
        "seal_payload_sha256": seal_payload_sha,
        "plan_sha256": plan_raw_sha,
        "diagnostic_provenance_payload_sha256": provenance_payload_sha256,
        "diagnostic_binding_sha256": diagnostic_binding_sha256,
        "runtime_authority": False,
        "scientific_authority": False,
        "selection_authority": False,
        "confirmation_authority": False,
        "formal_study_authority": False,
        "publication_status": "IJOC_HOLD",
    }
    if binding != derived:
        _fail("recovery preflight derived binding drifted")
    return derived


def _validate_same_implementation(
    receipt: Mapping[str, object],
    identity: Mapping[str, str],
    diagnostic_receipt: Mapping[str, object],
    diagnostic_receipt_sha256: str,
    receipt_path: Path,
    evidence_root: Path,
) -> bool:
    if receipt.get("schema") != "v21e3r1_recovery_bound_coverage_receipt_v1":
        _fail(
            "same implementation gate requires the recovery-bound outer receipt "
            "and success-seal chain; inner-v1 coverage is not gate evidence"
        )
    outer_root = receipt_path.parent
    if receipt_path.name != "recovery_bound_coverage.receipt.json":
        _fail("recovery-bound outer receipt path drifted")
    outer_payload_sha = _validate_recovery_bound_payload(
        receipt,
        label="recovery-bound outer receipt",
        digest_field="receipt_payload_sha256",
        schema="v21e3r1_recovery_bound_coverage_receipt_v1",
        status="PASS_RECOVERY_BOUND_SAME_IMPLEMENTATION_EXACT504_ONLY",
        expected_keys=_RECOVERY_OUTER_RECEIPT_KEYS,
        require_a6=True,
    )
    outer_raw_sha = _sha256_file(receipt_path)
    plan_path = outer_root / "recovery_bound_coverage.plan.json"
    provenance_path = outer_root / "diagnostic_provenance.binding.json"
    seal_path = outer_root / "recovery_bound_coverage.receipt.seal.json"
    for sibling, label in (
        (plan_path, "recovery-bound plan"),
        (provenance_path, "recovery-bound provenance"),
        (seal_path, "recovery-bound success seal"),
    ):
        if not sibling.is_file():
            _fail(f"{label} fixed sibling is absent")
    plan, plan_raw_sha = _load_stable_recovery_json(plan_path, label="recovery-bound plan")
    plan_payload_sha = _validate_recovery_bound_payload(
        plan,
        label="recovery-bound plan",
        digest_field="plan_payload_sha256",
        schema="v21e3r1_recovery_bound_coverage_plan_v1",
        status="FROZEN_RECOVERY_BOUND_SAME_IMPLEMENTATION_EXACT504",
        expected_keys=_RECOVERY_PLAN_KEYS,
        require_a6=True,
    )
    provenance, _ = _load_stable_recovery_json(
        provenance_path, label="recovery-bound diagnostic provenance"
    )
    provenance_payload_sha = _validate_recovery_bound_payload(
        provenance,
        label="recovery-bound diagnostic provenance",
        digest_field="provenance_payload_sha256",
        schema="v21e3r1_recovered_diagnostic_provenance_binding_v1",
        status="PASS_CHAIN_VALIDATED_FOR_REPLAY_INPUT_ONLY",
        require_a6=True,
    )
    diagnostic_binding = provenance.get("exact504_diagnostic_binding")
    if type(diagnostic_binding) is not dict:
        _fail("recovery-bound provenance omits exact504 diagnostic binding")
    diagnostic_binding_sha = _sha256_bytes(_canonical_bytes(diagnostic_binding))
    if (
        provenance.get("expected_rows") != 504
        or provenance.get("exact504_diagnostic_binding_sha256")
        != diagnostic_binding_sha
        or diagnostic_binding.get("schema")
        != "v21e3r1_recovery_bound_exact504_diagnostic_binding_v1"
        or diagnostic_binding.get("status")
        != "PASS_EXACT504_DIAGNOSTIC_TREE_REVALIDATED_INPUT_ONLY"
        or diagnostic_binding.get("expected_rows") != 504
        or diagnostic_binding.get("plan_sha256") != diagnostic_receipt["plan_sha256"]
        or diagnostic_binding.get("source_snapshot_sha256")
        != identity["development_source_sha256"]
        or diagnostic_binding.get("diagnostic_receipt_sha256")
        != diagnostic_receipt_sha256
        or diagnostic_binding.get("diagnostic_aggregate_sha256")
        != diagnostic_receipt["aggregate_sha256"]
    ):
        _fail("recovery-bound exact504 provenance disagrees with gate evidence")
    _validate_recovery_hold_authority(
        diagnostic_binding, label="recovery-bound exact504 diagnostic binding", require_a6=True
    )
    rows = diagnostic_binding.get("rows")
    if (
        type(rows) is not list
        or len(rows) != 504
        or any(type(row) is not dict or row.get("ordinal") != index for index, row in enumerate(rows, start=1))
    ):
        _fail("recovery-bound exact504 diagnostic rows drifted")
    runtime_identities = _validate_recovery_runtime_identities(plan["runtime_identities"])
    preflight = _validate_recovery_preflight(
        evidence_root=evidence_root,
        binding_value=plan["preflight_binding"],
        provenance_payload_sha256=provenance_payload_sha,
        diagnostic_binding=diagnostic_binding,
        diagnostic_binding_sha256=diagnostic_binding_sha,
        runtime_identities=runtime_identities,
    )
    jobs = _integer(plan["jobs"], label="recovery-bound plan.jobs", minimum=1)
    timeout = _integer(
        plan["row_timeout_seconds"],
        label="recovery-bound plan.row_timeout_seconds",
        minimum=2400,
    )
    if (
        plan["expected_rows"] != 504
        or Path(_string(plan["output_root"], label="recovery-bound plan.output_root")).resolve()
        != outer_root.resolve()
        or plan["provenance_payload_sha256"] != provenance_payload_sha
        or plan["diagnostic_binding_sha256"] != diagnostic_binding_sha
        or plan["runtime_identities"] != runtime_identities
        or receipt["expected_rows"] != 504
        or receipt["jobs"] != jobs
        or receipt["row_timeout_seconds"] != timeout
        or receipt["plan_sha256"] != plan_raw_sha
        or receipt["plan_payload_sha256"] != plan_payload_sha
        or receipt["provenance_payload_sha256"] != provenance_payload_sha
        or receipt["diagnostic_binding_sha256"] != diagnostic_binding_sha
        or receipt["preflight_receipt_sha256"] != preflight["receipt_sha256"]
        or receipt["preflight_seal_sha256"] != preflight["seal_sha256"]
        or receipt["runtime_identities_before"] != runtime_identities
        or receipt["runtime_identities_after"] != runtime_identities
        or receipt["same_implementation_only"] is not True
        or receipt["process_isolation"]
        != "INNER_V1_PER_ROW_ISOLATED_PROCESS_BOUNDARY"
        or _number(receipt["wall_time_seconds"], label="recovery-bound wall_time_seconds") < 0
    ):
        _fail("recovery-bound outer receipt binding drifted")
    claim_path = _contained_file(
        outer_root,
        receipt["execution_claim_path"],
        label="recovery-bound execution claim path",
    )
    claim, claim_raw_sha = _load_stable_recovery_json(
        claim_path, label="recovery-bound execution claim"
    )
    claim_payload_sha = _validate_recovery_bound_payload(
        claim,
        label="recovery-bound execution claim",
        digest_field="claim_payload_sha256",
        schema="v21e3r1_recovery_bound_coverage_execution_claim_v1",
        status="SEALED_EXCLUSIVE_SAME_IMPLEMENTATION_EXECUTION_CLAIM",
        expected_keys=_RECOVERY_CLAIM_KEYS,
        require_a6=True,
    )
    execution_number = _integer(
        claim["execution_number"], label="recovery-bound execution number", minimum=1
    )
    if (
        claim_path.name != f"execution-{execution_number:04d}.claim.json"
        or claim["plan_sha256"] != plan_raw_sha
        or claim["provenance_payload_sha256"] != provenance_payload_sha
        or claim["preflight_receipt_sha256"] != preflight["receipt_sha256"]
        or claim["preflight_seal_sha256"] != preflight["seal_sha256"]
        or claim["runtime_identities"] != runtime_identities
        or claim["jobs"] != jobs
        or claim["row_timeout_seconds"] != timeout
        or type(claim["inner_resume"]) is not bool
        or receipt["execution_claim_sha256"] != claim_raw_sha
        or receipt["execution_claim_payload_sha256"] != claim_payload_sha
    ):
        _fail("recovery-bound execution claim binding drifted")
    inner_path = _contained_file(
        outer_root,
        receipt["inner_receipt_path"],
        label="recovery-bound inner-v1 receipt path",
    )
    inner, inner_raw_sha = _load_stable_recovery_json(
        inner_path, label="recovery-bound inner-v1 receipt"
    )
    inner_payload_sha = _sha256_bytes(_canonical_bytes(inner))
    if (
        inner_path != (outer_root / "inner_v1" / "branch_replay_coverage.receipt.json").resolve()
        or receipt["inner_receipt_sha256"] != inner_raw_sha
        or receipt["inner_receipt_payload_sha256"] != inner_payload_sha
    ):
        _fail("recovery-bound inner-v1 receipt hash binding drifted")
    inner_valid = _validate_inner_same_implementation(
        inner,
        identity,
        diagnostic_receipt,
        diagnostic_receipt_sha256,
        inner_path,
    )
    inner_seals = inner["row_seals"]
    if any(
        seal.get("row_id") != row.get("row_id")
        or seal.get("plan_ordinal") != row.get("ordinal")
        or seal.get("diagnostic_completed_marker_sha256")
        != row.get("completed_marker_sha256")
        or seal.get("diagnostic_trace_sha256") != row.get("trace_sha256")
        for seal, row in zip(inner_seals, rows)
    ):
        _fail("recovery-bound inner row seals disagree with provenance rows")
    seal, _ = _load_stable_recovery_json(
        seal_path, label="recovery-bound success seal"
    )
    _validate_recovery_bound_payload(
        seal,
        label="recovery-bound success seal",
        digest_field="seal_payload_sha256",
        schema="v21e3r1_recovery_bound_coverage_success_seal_v1",
        status="SEALED_RECOVERY_BOUND_COVERAGE_SUCCESS_RECEIPT",
        expected_keys=_RECOVERY_OUTER_SEAL_KEYS,
        require_a6=True,
    )
    if (
        seal["receipt_path"] != receipt_path.name
        or seal["receipt_sha256"] != outer_raw_sha
        or seal["receipt_payload_sha256"] != outer_payload_sha
        or seal["plan_sha256"] != plan_raw_sha
        or seal["provenance_payload_sha256"] != provenance_payload_sha
        or seal["preflight_receipt_sha256"] != preflight["receipt_sha256"]
        or seal["inner_receipt_sha256"] != inner_raw_sha
    ):
        _fail("recovery-bound outer success seal chain drifted")
    return inner_valid


def _validate_baselines(
    receipt: Mapping[str, object], identity: Mapping[str, str], receipt_path: Path
) -> tuple[bool, dict[str, int]]:
    if receipt.get("schema") == "v21e3r1_external_family_native_strong_baseline_registry_receipt_v2":
        raw = _exact_keys(
            receipt,
            {
                "schema",
                "status",
                "scope",
                "study_id",
                "candidate_id",
                "study_metric_spec_sha256",
                "primary_source_cutoff_date",
                "registry_path",
                "registry_sha256",
                "registry_payload_sha256",
                "verifier_source_path",
                "verifier_source_sha256",
                "verification_receipt_path",
                "verification_receipt_sha256",
                "artifact_count",
                "comparator_count",
                "all_registry_artifacts_verified",
                "families",
                "selection_authorized",
                "confirmation_authorized",
                "formal_study_authorized",
                "scientific_claim_authorized",
                "ijoc_submission_status",
                "receipt_payload_sha256",
            },
            label="baseline registry receipt v2",
        )
        payload_sha = _sha256(
            raw["receipt_payload_sha256"], label="baseline.receipt_payload_sha256"
        )
        core = dict(raw)
        core.pop("receipt_payload_sha256")
        if payload_sha != _sha256_bytes(_canonical_bytes(core)):
            _fail("baseline registry receipt payload digest drifted")
        for field in (
            "study_metric_spec_sha256",
            "registry_sha256",
            "registry_payload_sha256",
            "verifier_source_sha256",
            "verification_receipt_sha256",
        ):
            _sha256(raw[field], label=f"baseline.{field}")
        registry_path = _verified_artifact(
            receipt_path.parent,
            path_value=raw["registry_path"],
            sha256_value=raw["registry_sha256"],
            label="baseline.registry",
        )
        verifier_path = _verified_artifact(
            receipt_path.parent,
            path_value=raw["verifier_source_path"],
            sha256_value=raw["verifier_source_sha256"],
            label="baseline.verifier_source",
        )
        verification_path = _verified_artifact(
            receipt_path.parent,
            path_value=raw["verification_receipt_path"],
            sha256_value=raw["verification_receipt_sha256"],
            label="baseline.verification_receipt",
        )
        if len({receipt_path, registry_path, verifier_path, verification_path}) != 4:
            _fail("baseline registry receipt and top-level siblings must be distinct")
        registry = _parse_json(
            registry_path.read_bytes(), label="bound reference comparator registry", canonical=False
        )
        if registry.get("schema") != "ijoc_v21e3r1_v7_reference_comparator_registry_v1":
            _fail("bound reference comparator registry schema drifted")
        supplied_registry_payload = _sha256(
            registry.get("registry_payload_sha256"),
            label="bound registry.registry_payload_sha256",
        )
        registry_core = dict(registry)
        registry_core.pop("registry_payload_sha256")
        if supplied_registry_payload != _sha256_bytes(
            _canonical_bytes(registry_core) + b"\n"
        ):
            _fail("bound reference comparator registry payload digest drifted")
        if supplied_registry_payload != raw["registry_payload_sha256"]:
            _fail("baseline receipt registry payload binding drifted")
        verification = _parse_json(
            verification_path.read_bytes(),
            label="bound registry verification receipt",
            canonical=True,
        )
        if (
            verification.get("schema")
            != "ijoc_v21e3r1_v7_reference_comparator_registry_verification_v1"
            or verification.get("status")
            != "PASS_STRICT_OFFLINE_DEVELOPMENT_REFERENCE_FREEZE_ONLY"
            or verification.get("registry_file_sha256") != raw["registry_sha256"]
            or verification.get("registry_payload_sha256")
            != raw["registry_payload_sha256"]
            or verification.get("network_calls") != 0
            or verification.get("external_family_native_strong_baseline_count") != 0
            or verification.get("registry_artifacts_verified") is not True
        ):
            _fail("bound registry verification receipt disagrees with the baseline freeze")
        verification_payload = _sha256(
            verification.get("receipt_payload_sha256"),
            label="baseline verification receipt payload",
        )
        verification_core = dict(verification)
        verification_core.pop("receipt_payload_sha256")
        if verification_payload != _sha256_bytes(_canonical_bytes(verification_core)):
            _fail("bound registry verification receipt payload digest drifted")

        families = raw["families"]
        if type(families) is not list or len(families) != len(FAMILIES):
            _fail("baseline registry v2 must contain exactly two families")
        counts: dict[str, int] = {}
        observed_families: list[str] = []
        baseline_ids: set[str] = set()
        observed_paths: set[Path] = {receipt_path, registry_path, verifier_path, verification_path}
        for ordinal, family_value in enumerate(families):
            family = _exact_keys(
                family_value,
                {"family", "external_family_native_strong_baseline_count", "baselines"},
                label=f"baseline v2 families[{ordinal}]",
            )
            family_id = _string(family["family"], label=f"baseline v2 family {ordinal}")
            observed_families.append(family_id)
            baselines = family["baselines"]
            if type(baselines) is not list or not baselines:
                _fail(f"baseline v2 family {family_id} has no development references")
            eligible = 0
            observed_order: list[str] = []
            for index, baseline_value in enumerate(baselines):
                baseline = _exact_keys(
                    baseline_value,
                    {
                        "baseline_id",
                        "classification",
                        "external",
                        "family_native",
                        "strong",
                        "development_reference_eligible",
                        "external_family_native_strong_baseline_eligible",
                        "source_manifest_path",
                        "source_manifest_sha256",
                        "evaluation_receipt_path",
                        "evaluation_receipt_sha256",
                        "study_metric_spec_sha256",
                    },
                    label=f"baseline v2 {family_id}[{index}]",
                )
                baseline_id = _identifier(
                    baseline["baseline_id"], label=f"baseline v2 {family_id}[{index}].id"
                )
                observed_order.append(baseline_id)
                if baseline_id in baseline_ids:
                    _fail("baseline v2 IDs must be globally unique")
                baseline_ids.add(baseline_id)
                classification = _string(
                    baseline["classification"], label=f"baseline {baseline_id}.classification"
                )
                for field in (
                    "source_manifest_sha256",
                    "evaluation_receipt_sha256",
                    "study_metric_spec_sha256",
                ):
                    _sha256(baseline[field], label=f"baseline {baseline_id}.{field}")
                manifest_path = _verified_artifact(
                    receipt_path.parent,
                    path_value=baseline["source_manifest_path"],
                    sha256_value=baseline["source_manifest_sha256"],
                    label=f"baseline {baseline_id}.source_manifest",
                )
                evaluation_path = _verified_artifact(
                    receipt_path.parent,
                    path_value=baseline["evaluation_receipt_path"],
                    sha256_value=baseline["evaluation_receipt_sha256"],
                    label=f"baseline {baseline_id}.evaluation_receipt",
                )
                if manifest_path in observed_paths or evaluation_path in observed_paths:
                    _fail("baseline v2 sibling artifact paths must be globally distinct")
                observed_paths.update({manifest_path, evaluation_path})
                source_manifest = _parse_json(
                    manifest_path.read_bytes(),
                    label=f"baseline {baseline_id} source manifest",
                    canonical=True,
                )
                source_manifest = _exact_keys(
                    source_manifest,
                    {
                        "schema",
                        "baseline_id",
                        "problem_family",
                        "source_identity",
                        "artifact_bindings",
                        "source_root_sha256",
                    },
                    label=f"baseline {baseline_id} source manifest",
                )
                if (
                    source_manifest["schema"]
                    != "v21e3r1_reference_comparator_source_manifest_v1"
                    or source_manifest["baseline_id"] != baseline_id
                    or source_manifest["problem_family"] != family_id
                    or type(source_manifest["source_identity"]) is not dict
                    or not source_manifest["source_identity"]
                ):
                    _fail(f"baseline {baseline_id} source manifest identity drifted")
                artifact_bindings = source_manifest["artifact_bindings"]
                if type(artifact_bindings) is not list or not artifact_bindings:
                    _fail(f"baseline {baseline_id} source manifest has no artifacts")
                artifact_ids: set[str] = set()
                for artifact_index, artifact_value in enumerate(artifact_bindings):
                    artifact = _exact_keys(
                        artifact_value,
                        {"artifact_id", "role", "path", "bytes", "sha256"},
                        label=f"baseline {baseline_id} artifact[{artifact_index}]",
                    )
                    artifact_id = _identifier(
                        artifact["artifact_id"],
                        label=f"baseline {baseline_id} artifact id",
                    )
                    if artifact_id in artifact_ids:
                        _fail(f"baseline {baseline_id} repeats artifact IDs")
                    artifact_ids.add(artifact_id)
                    _string(artifact["role"], label=f"baseline {baseline_id} artifact role")
                    _canonical_relative_path(
                        artifact["path"], label=f"baseline {baseline_id} artifact path"
                    )
                    _integer(
                        artifact["bytes"], label=f"baseline {baseline_id} artifact bytes", minimum=1
                    )
                    _sha256(
                        artifact["sha256"], label=f"baseline {baseline_id} artifact sha256"
                    )
                source_root = _sha256(
                    source_manifest["source_root_sha256"],
                    label=f"baseline {baseline_id} source root",
                )
                if source_root != _sha256_bytes(_canonical_bytes(artifact_bindings)):
                    _fail(f"baseline {baseline_id} source-root digest drifted")
                availability = _parse_json(
                    evaluation_path.read_bytes(),
                    label=f"baseline {baseline_id} availability receipt",
                    canonical=True,
                )
                if (
                    availability.get("schema")
                    != "v21e3r1_reference_comparator_availability_receipt_v1"
                    or availability.get("baseline_id") != baseline_id
                    or availability.get("problem_family") != family_id
                    or availability.get("source_manifest_sha256")
                    != baseline["source_manifest_sha256"]
                    or availability.get("study_metric_spec_sha256")
                    != baseline["study_metric_spec_sha256"]
                    or availability.get("evaluation_executed") is not False
                    or availability.get(
                        "external_family_native_strong_baseline_eligible"
                    )
                    is not False
                ):
                    _fail(f"baseline {baseline_id} availability receipt drifted")
                availability_payload = _sha256(
                    availability.get("receipt_payload_sha256"),
                    label=f"baseline {baseline_id} availability payload",
                )
                availability_core = dict(availability)
                availability_core.pop("receipt_payload_sha256")
                if availability_payload != _sha256_bytes(_canonical_bytes(availability_core)):
                    _fail(f"baseline {baseline_id} availability payload drifted")
                external = _boolean(
                    baseline["external"], label=f"baseline {baseline_id}.external"
                )
                native = _boolean(
                    baseline["family_native"], label=f"baseline {baseline_id}.family_native"
                )
                strong = _boolean(
                    baseline["strong"], label=f"baseline {baseline_id}.strong"
                )
                _boolean(
                    baseline["development_reference_eligible"],
                    label=f"baseline {baseline_id}.development_reference_eligible",
                )
                declared_eligible = _boolean(
                    baseline["external_family_native_strong_baseline_eligible"],
                    label=f"baseline {baseline_id}.external strong eligibility",
                )
                exact_class = classification == "EXTERNAL_FAMILY_NATIVE_STRONG_BASELINE"
                derived_eligible = bool(
                    exact_class
                    and external
                    and native
                    and strong
                    and baseline["study_metric_spec_sha256"]
                    == identity["study_metric_spec_sha256"]
                    and baseline["source_manifest_sha256"]
                    != identity["successor_source_sha256"]
                )
                if declared_eligible != derived_eligible:
                    _fail(f"baseline {baseline_id} eligibility declaration drifted")
                eligible += int(derived_eligible)
            if observed_order != sorted(observed_order):
                _fail(f"baseline v2 family {family_id} baselines are not ID-sorted")
            declared_count = _integer(
                family["external_family_native_strong_baseline_count"],
                label=f"baseline v2 family {family_id} eligible count",
            )
            if declared_count != eligible:
                _fail(f"baseline v2 family {family_id} eligible count drifted")
            counts[family_id] = eligible
        if observed_families != list(FAMILIES):
            _fail("baseline v2 families must be exact frozen MOKP, MOTSP order")
        if _integer(raw["artifact_count"], label="baseline.artifact_count", minimum=1) != verification.get("artifact_count"):
            _fail("baseline artifact count disagrees with verification receipt")
        if _integer(raw["comparator_count"], label="baseline.comparator_count", minimum=1) != len(baseline_ids):
            _fail("baseline comparator count disagrees with family entries")
        if (
            raw["study_id"] != identity["study_id"]
            or raw["candidate_id"] != identity["candidate_id"]
            or raw["study_metric_spec_sha256"]
            != identity["study_metric_spec_sha256"]
        ):
            _fail("baseline v2 identity disagrees with the gate specification")
        _boolean(
            raw["all_registry_artifacts_verified"],
            label="baseline.all_registry_artifacts_verified",
        )
        for field in (
            "selection_authorized",
            "confirmation_authorized",
            "formal_study_authorized",
            "scientific_claim_authorized",
        ):
            if _boolean(raw[field], label=f"baseline.{field}"):
                _fail(f"baseline v2 design-only receipt expands authority at {field}")
        if raw["ijoc_submission_status"] != "IJOC_HOLD":
            _fail("baseline v2 publication status drifted")
        # Schema v2 is the frozen offline development-reference registry.  It
        # has no evaluated strong-baseline evidence and is permanently
        # design-only: resealing its declarations cannot promote this gate.
        return False, {family: 0 for family in FAMILIES}

    raw = _exact_keys(
        receipt,
        {"schema", "status", "study_id", "candidate_id", "metric_spec_sha256", "families"},
        label="baseline registry receipt",
    )
    _sha256(raw["metric_spec_sha256"], label="baseline metric_spec_sha256")
    if raw["schema"] != "v21e3r1_external_family_native_strong_baseline_registry_receipt_v1":
        _fail("baseline registry receipt schema drifted")
    if (
        raw["study_id"] != identity["study_id"]
        or raw["candidate_id"] != identity["candidate_id"]
        or raw["metric_spec_sha256"] != identity["study_metric_spec_sha256"]
    ):
        _fail("claim-only baseline v1 identity disagrees with the gate specification")
    families = raw["families"]
    if type(families) is not list or len(families) != len(FAMILIES):
        _fail("baseline registry must contain exactly the two required families")
    counts: dict[str, int] = {}
    observed: list[str] = []
    baseline_ids: set[str] = set()
    for ordinal, family_value in enumerate(families):
        family = _exact_keys(
            family_value, {"family", "baselines"}, label=f"baseline families[{ordinal}]"
        )
        family_id = _string(family["family"], label=f"baseline family {ordinal}")
        observed.append(family_id)
        baselines = family["baselines"]
        if type(baselines) is not list or not baselines:
            _fail(f"baseline family {family_id} has no baselines")
        eligible = 0
        for index, baseline_value in enumerate(baselines):
            baseline = _exact_keys(
                baseline_value,
                {
                    "baseline_id",
                    "classification",
                    "external",
                    "family_native",
                    "strong",
                    "source_manifest_sha256",
                    "evaluation_receipt_sha256",
                    "metric_spec_sha256",
                },
                label=f"baseline {family_id}[{index}]",
            )
            baseline_id = _identifier(
                baseline["baseline_id"], label=f"baseline {family_id}[{index}].id"
            )
            if baseline_id in baseline_ids:
                _fail("baseline IDs must be globally unique")
            baseline_ids.add(baseline_id)
            for field in (
                "source_manifest_sha256",
                "evaluation_receipt_sha256",
                "metric_spec_sha256",
            ):
                _sha256(baseline[field], label=f"baseline {baseline_id}.{field}")
            exact_class = (
                baseline["classification"]
                == "EXTERNAL_FAMILY_NATIVE_STRONG_BASELINE"
            )
            external = _boolean(baseline["external"], label=f"baseline {baseline_id}.external")
            native = _boolean(
                baseline["family_native"], label=f"baseline {baseline_id}.family_native"
            )
            strong = _boolean(baseline["strong"], label=f"baseline {baseline_id}.strong")
            if (
                exact_class
                and external
                and native
                and strong
                and baseline["metric_spec_sha256"]
                == identity["study_metric_spec_sha256"]
                and baseline["source_manifest_sha256"]
                != identity["successor_source_sha256"]
            ):
                eligible += 1
        counts[family_id] = eligible
    if observed != list(FAMILIES):
        _fail("baseline families must be in exact frozen MOKP, MOTSP order")
    # Schema v1 binds claim strings and digests, but no source/evaluation files
    # that this consumer can locate and re-verify.  It is therefore evidence for
    # a HOLD report only, never for baseline authority.
    return False, counts


def _verify_contract_payload(receipt: Mapping[str, object], *, label: str) -> None:
    declared = _sha256(
        receipt.get("receipt_payload_sha256"),
        label=f"{label}.receipt_payload_sha256",
    )
    core = dict(receipt)
    core.pop("receipt_payload_sha256")
    if declared != _sha256_bytes(_canonical_bytes(core)):
        _fail(f"{label} payload digest drifted")


def _validate_external_replay_v3_contract(
    receipt: Mapping[str, object], identity: Mapping[str, str]
) -> bool:
    raw = _exact_keys(
        receipt,
        {
            "schema",
            "status",
            "scope",
            "study_id",
            "candidate_id",
            "successor_source_sha256",
            "successor_config_sha256",
            "study_metric_spec_sha256",
            "target_receipt_schema",
            "required_path_bound_roles",
            "all_artifacts_require_path_sha256_and_payload_sha256",
            "producer_authorship_requires_external_authority",
            "custody_requires_external_authority",
            "external_producer_present",
            "independent_custody_authority_present",
            "implementation_code_disjoint_verified",
            "algorithm_execution_independence_verified",
            "gate_clearable_by_this_contract",
            "selection_authorized",
            "confirmation_authorized",
            "formal_study_authorized",
            "scientific_claim_authorized",
            "ijoc_submission_status",
            "receipt_payload_sha256",
        },
        label="external replay v3 evidence contract",
    )
    _verify_contract_payload(raw, label="external replay v3 evidence contract")
    if (
        raw["schema"] != "v21e3r1_external_algorithm_replay_evidence_contract_v3"
        or raw["status"]
        != "FROZEN_FUTURE_EXTERNAL_PRODUCER_CONTRACT_NO_LOCAL_EVIDENCE"
        or raw["scope"]
        != "MACHINE_CHECKABLE_PATH_BOUND_REQUIREMENTS_ONLY_NO_INDEPENDENCE_CLAIM"
        or raw["target_receipt_schema"]
        != "v21e3r1_external_algorithm_replay_receipt_v3"
    ):
        _fail("external replay v3 evidence contract schema/status drifted")
    for field in (
        "study_id",
        "candidate_id",
        "successor_source_sha256",
        "successor_config_sha256",
        "study_metric_spec_sha256",
    ):
        if raw[field] != identity[field]:
            _fail(f"external replay v3 contract identity drifted at {field}")
    expected_roles = [
        "reference_source_manifest",
        "external_source_manifest",
        "reference_event_stream",
        "external_event_stream",
        "neutral_comparison_receipt",
        "producer_authorship_authority_receipt",
        "independent_custody_authority_receipt",
        "external_execution_environment_receipt",
    ]
    if raw["required_path_bound_roles"] != expected_roles:
        _fail("external replay v3 required artifact roles drifted")
    for field in (
        "all_artifacts_require_path_sha256_and_payload_sha256",
        "producer_authorship_requires_external_authority",
        "custody_requires_external_authority",
    ):
        if not _boolean(raw[field], label=f"external v3 contract.{field}"):
            _fail(f"external replay v3 requirement is not enabled: {field}")
    for field in (
        "external_producer_present",
        "independent_custody_authority_present",
        "implementation_code_disjoint_verified",
        "algorithm_execution_independence_verified",
    ):
        if _boolean(raw[field], label=f"external v3 contract.{field}"):
            _fail("local v3 contract cannot assert external producer or custody authority")
    for field in (
        "gate_clearable_by_this_contract",
        "selection_authorized",
        "confirmation_authorized",
        "formal_study_authorized",
        "scientific_claim_authorized",
    ):
        if _boolean(raw[field], label=f"external v3 contract.{field}"):
            _fail(f"external replay v3 design contract expands authority at {field}")
    if raw["ijoc_submission_status"] != "IJOC_HOLD":
        _fail("external replay v3 design contract publication status drifted")
    return False


def _validate_path_bound_phase_v3_contract(
    receipt: Mapping[str, object],
    identity: Mapping[str, str],
    *,
    phase: str,
) -> bool:
    raw = _exact_keys(
        receipt,
        {
            "schema",
            "status",
            "scope",
            "study_id",
            "candidate_id",
            "successor_source_sha256",
            "successor_config_sha256",
            "study_metric_spec_sha256",
            "target_receipt_schema",
            "supported_phases",
            "required_common_path_bound_roles",
            "confirmation_additional_path_bound_roles",
            "selection_confirmation_case_disjointness_required",
            "prospective_chronology_required",
            "external_phase_producer_present",
            "independent_custody_authority_present",
            "gate_clearable_by_this_contract",
            "selection_result_materialized",
            "confirmation_result_materialized",
            "formal_study_materialized",
            "selection_authorized",
            "confirmation_authorized",
            "formal_study_authorized",
            "scientific_claim_authorized",
            "ijoc_submission_status",
            "receipt_payload_sha256",
        },
        label="path-bound phase v3 evidence contract",
    )
    _verify_contract_payload(raw, label="path-bound phase v3 evidence contract")
    if (
        raw["schema"] != "v21e3r1_path_bound_phase_evidence_contract_v3"
        or raw["status"]
        != "FROZEN_FUTURE_PATH_BOUND_PHASE_CONTRACT_NO_LOCAL_EVIDENCE"
        or raw["target_receipt_schema"]
        != "v21e3r1_path_bound_independent_phase_receipt_v3"
        or raw["supported_phases"] != ["selection", "confirmation"]
        or phase not in raw["supported_phases"]
    ):
        _fail("path-bound phase v3 evidence contract schema/phase drifted")
    for field in (
        "study_id",
        "candidate_id",
        "successor_source_sha256",
        "successor_config_sha256",
        "study_metric_spec_sha256",
    ):
        if raw[field] != identity[field]:
            _fail(f"path-bound phase v3 contract identity drifted at {field}")
    for field in (
        "selection_confirmation_case_disjointness_required",
        "prospective_chronology_required",
    ):
        if not _boolean(raw[field], label=f"phase v3 contract.{field}"):
            _fail(f"path-bound phase v3 requirement is not enabled: {field}")
    for field in (
        "external_phase_producer_present",
        "independent_custody_authority_present",
        "gate_clearable_by_this_contract",
        "selection_result_materialized",
        "confirmation_result_materialized",
        "formal_study_materialized",
        "selection_authorized",
        "confirmation_authorized",
        "formal_study_authorized",
        "scientific_claim_authorized",
    ):
        if _boolean(raw[field], label=f"phase v3 contract.{field}"):
            _fail(f"local path-bound phase v3 contract expands authority at {field}")
    if raw["ijoc_submission_status"] != "IJOC_HOLD":
        _fail("path-bound phase v3 contract publication status drifted")
    return False


def _validate_external_replay(
    receipt: Mapping[str, object], identity: Mapping[str, str], receipt_path: Path
) -> tuple[bool, bool, bool, bool, bool, str]:
    if receipt.get("schema") == "v21e3r1_external_algorithm_replay_receipt_v2":
        raw = _exact_keys(
            receipt,
            {
                "schema",
                "status",
                "scope",
                "study_id",
                "candidate_id",
                "successor_source_sha256",
                "successor_config_sha256",
                "study_metric_spec_sha256",
                "design_spec_path",
                "design_spec_sha256",
                "comparator_source_path",
                "comparator_source_sha256",
                "comparator_test_path",
                "comparator_test_sha256",
                "golden_streams",
                "positive_comparison_receipt_path",
                "positive_comparison_receipt_sha256",
                "negative_comparison_receipt_path",
                "negative_comparison_receipt_sha256",
                "reference_producer_id",
                "external_producer_id",
                "reference_source_manifest_sha256",
                "external_source_manifest_sha256",
                "event_streams_match",
                "reference_algorithm_producer_present",
                "external_producer_present",
                "producer_authorship_authenticated",
                "independent_producer",
                "independent_custody",
                "implementation_code_disjoint",
                "algorithm_execution_independence",
                "custody_receipt_path",
                "custody_receipt_sha256",
                "selection_authorized",
                "confirmation_authorized",
                "formal_study_authorized",
                "scientific_claim_authorized",
                "ijoc_submission_status",
                "receipt_payload_sha256",
            },
            label="external algorithm replay receipt v2",
        )
        payload_sha = _sha256(
            raw["receipt_payload_sha256"], label="external replay.receipt_payload_sha256"
        )
        core = dict(raw)
        core.pop("receipt_payload_sha256")
        if payload_sha != _sha256_bytes(_canonical_bytes(core)):
            _fail("external replay v2 receipt payload digest drifted")
        for field in (
            "successor_source_sha256",
            "successor_config_sha256",
            "study_metric_spec_sha256",
            "design_spec_sha256",
            "comparator_source_sha256",
            "comparator_test_sha256",
            "positive_comparison_receipt_sha256",
            "negative_comparison_receipt_sha256",
            "reference_source_manifest_sha256",
            "external_source_manifest_sha256",
            "custody_receipt_sha256",
        ):
            _sha256(raw[field], label=f"external replay.{field}")
        identity_match = bool(
            raw["study_id"] == identity["study_id"]
            and raw["candidate_id"] == identity["candidate_id"]
            and raw["successor_source_sha256"] == identity["successor_source_sha256"]
            and raw["successor_config_sha256"] == identity["successor_config_sha256"]
            and raw["study_metric_spec_sha256"]
            == identity["study_metric_spec_sha256"]
        )
        if not identity_match:
            _fail("external replay v2 identity disagrees with the gate specification")
        artifact_specs = (
            ("design_spec", "design_spec_path", "design_spec_sha256"),
            ("comparator_source", "comparator_source_path", "comparator_source_sha256"),
            ("comparator_test", "comparator_test_path", "comparator_test_sha256"),
            (
                "positive_comparison",
                "positive_comparison_receipt_path",
                "positive_comparison_receipt_sha256",
            ),
            (
                "negative_comparison",
                "negative_comparison_receipt_path",
                "negative_comparison_receipt_sha256",
            ),
            ("custody", "custody_receipt_path", "custody_receipt_sha256"),
        )
        observed_paths: set[Path] = {receipt_path}
        artifacts: dict[str, Path] = {}
        for artifact, path_field, sha_field in artifact_specs:
            path = _verified_artifact(
                receipt_path.parent,
                path_value=raw[path_field],
                sha256_value=raw[sha_field],
                label=f"external replay.{artifact}",
            )
            if path in observed_paths:
                _fail("external replay v2 sibling artifacts must be distinct")
            observed_paths.add(path)
            artifacts[artifact] = path
        golden = _exact_keys(
            raw["golden_streams"],
            {"reference_valid", "external_placeholder", "negative_decision_mismatch"},
            label="external replay golden streams",
        )
        for role in ("reference_valid", "external_placeholder", "negative_decision_mismatch"):
            binding = _exact_keys(
                golden[role], {"path", "bytes", "sha256"}, label=f"golden stream {role}"
            )
            path = _verified_artifact(
                receipt_path.parent,
                path_value=binding["path"],
                sha256_value=binding["sha256"],
                label=f"golden stream {role}",
            )
            if path in observed_paths:
                _fail("external replay golden stream paths must be distinct")
            observed_paths.add(path)
            if path.stat().st_size != _integer(
                binding["bytes"], label=f"golden stream {role}.bytes", minimum=1
            ):
                _fail(f"golden stream {role} byte count drifted")
        positive = _parse_json(
            artifacts["positive_comparison"].read_bytes(),
            label="positive golden comparison receipt",
            canonical=True,
            allow_canonical_pretty=True,
        )
        negative = _parse_json(
            artifacts["negative_comparison"].read_bytes(),
            label="negative golden comparison receipt",
            canonical=True,
            allow_canonical_pretty=True,
        )
        if positive.get("status") != "PASS_NEUTRAL_EVENT_STREAM_COMPARISON":
            _fail("positive golden comparison status drifted")
        if negative.get("status") != "FAIL_ALGORITHM_EVENT_STREAM_MISMATCH":
            _fail("negative golden comparison status drifted")
        positive_streams = positive.get("streams")
        if type(positive_streams) is not dict:
            _fail("positive golden comparison stream bindings are absent")
        reference_stream = positive_streams.get("reference")
        candidate_stream = positive_streams.get("candidate")
        if type(reference_stream) is not dict or type(candidate_stream) is not dict:
            _fail("positive golden comparison stream bindings are malformed")
        if (
            reference_stream.get("producer_id") != raw["reference_producer_id"]
            or candidate_stream.get("producer_id") != raw["external_producer_id"]
            or reference_stream.get("producer_source_manifest_sha256")
            != raw["reference_source_manifest_sha256"]
            or candidate_stream.get("producer_source_manifest_sha256")
            != raw["external_source_manifest_sha256"]
            or positive.get("reference_algorithm_producer_present") is not False
            or positive.get("external_producer_present") is not False
        ):
            _fail("positive golden comparison identity/boundary drifted")
        custody_receipt = _parse_json(
            artifacts["custody"].read_bytes(),
            label="external replay custody receipt",
            canonical=True,
        )
        custody_payload = _sha256(
            custody_receipt.get("receipt_payload_sha256"),
            label="external replay custody payload",
        )
        custody_core = dict(custody_receipt)
        custody_core.pop("receipt_payload_sha256")
        if custody_payload != _sha256_bytes(_canonical_bytes(custody_core)):
            _fail("external replay custody receipt payload drifted")
        if (
            custody_receipt.get("schema")
            != "v21e3r1_external_algorithm_replay_custody_receipt_v1"
            or custody_receipt.get("status")
            != "HOLD_DESIGN_ONLY_NO_EXTERNAL_CUSTODY"
        ):
            _fail("external replay custody boundary drifted")
        event_match = _boolean(raw["event_streams_match"], label="event_streams_match")
        producer = _boolean(raw["independent_producer"], label="independent_producer")
        custody = _boolean(raw["independent_custody"], label="independent_custody")
        disjoint = _boolean(
            raw["implementation_code_disjoint"], label="implementation_code_disjoint"
        )
        algorithm = _boolean(
            raw["algorithm_execution_independence"],
            label="algorithm_execution_independence",
        )
        for field, value in (
            ("independent_producer", producer),
            ("independent_custody", custody),
            ("implementation_code_disjoint", disjoint),
            ("algorithm_execution_independence", algorithm),
        ):
            if value:
                _fail(f"design-only external replay v2 cannot set {field}=true")
        for field in (
            "reference_algorithm_producer_present",
            "external_producer_present",
            "producer_authorship_authenticated",
            "selection_authorized",
            "confirmation_authorized",
            "formal_study_authorized",
            "scientific_claim_authorized",
        ):
            if _boolean(raw[field], label=f"external replay.{field}"):
                _fail(f"design-only external replay cannot set {field}=true")
        return False, False, False, False, False, str(
            raw["custody_receipt_sha256"]
        )

    raw = _exact_keys(
        receipt,
        {
            "schema",
            "status",
            "study_id",
            "candidate_id",
            "successor_source_sha256",
            "successor_config_sha256",
            "metric_spec_sha256",
            "reference_producer_id",
            "external_producer_id",
            "reference_source_manifest_sha256",
            "external_source_manifest_sha256",
            "neutral_comparison_receipt_sha256",
            "event_streams_match",
            "independent_producer",
            "independent_custody",
            "implementation_code_disjoint",
            "algorithm_execution_independence",
            "custody_receipt_sha256",
        },
        label="external algorithm replay receipt",
    )
    if raw["schema"] != "v21e3r1_external_algorithm_replay_receipt_v1":
        _fail("external algorithm replay receipt schema drifted")
    for field in (
        "successor_source_sha256",
        "successor_config_sha256",
        "metric_spec_sha256",
        "reference_source_manifest_sha256",
        "external_source_manifest_sha256",
        "neutral_comparison_receipt_sha256",
        "custody_receipt_sha256",
    ):
        _sha256(raw[field], label=f"external replay.{field}")
    reference_id = _identifier(raw["reference_producer_id"], label="reference producer")
    external_id = _identifier(raw["external_producer_id"], label="external producer")
    if reference_id == external_id:
        _fail("algorithm replay producer IDs must be distinct")
    if raw["reference_source_manifest_sha256"] == raw["external_source_manifest_sha256"]:
        _fail("algorithm replay producer source manifests must be distinct")
    identity_match = bool(
        raw["study_id"] == identity["study_id"]
        and raw["candidate_id"] == identity["candidate_id"]
        and raw["successor_source_sha256"] == identity["successor_source_sha256"]
        and raw["successor_config_sha256"] == identity["successor_config_sha256"]
        and raw["metric_spec_sha256"] == identity["study_metric_spec_sha256"]
    )
    if not identity_match:
        _fail("external replay identity disagrees with the gate specification")
    event_match = _boolean(raw["event_streams_match"], label="event_streams_match")
    producer = _boolean(raw["independent_producer"], label="independent_producer")
    custody = _boolean(raw["independent_custody"], label="independent_custody")
    disjoint = _boolean(
        raw["implementation_code_disjoint"], label="implementation_code_disjoint"
    )
    algorithm = _boolean(
        raw["algorithm_execution_independence"],
        label="algorithm_execution_independence",
    )
    # Schema v1 has no path-bound producer streams, manifests, comparator
    # receipt, or custody artifact.  Claimed booleans remain non-authoritative.
    return False, False, False, False, False, str(
        raw["custody_receipt_sha256"]
    )


def _validate_simultaneous_spec(
    receipt: Mapping[str, object],
    identity: Mapping[str, str],
    raw_sha256: str,
    receipt_path: Path,
) -> bool:
    raw = _exact_keys(
        receipt,
        {
            "schema",
            "status",
            "scope",
            "study_id",
            "candidate_id",
            "successor_source_sha256",
            "successor_config_sha256",
            "study_metric_spec_sha256",
            "evaluator_source_path",
            "evaluator_source_sha256",
            "evaluator_test_path",
            "evaluator_test_sha256",
            "method",
            "families",
            "candidates",
            "familywise_alpha",
            "bootstrap_samples",
            "bootstrap_seed",
            "quantile_convention",
            "critical_value_floor",
            "rng_protocol",
            "rng_domain",
            "cluster_unit",
            "seed_aggregation",
            "resampling_rule",
            "centering",
            "studentization_denominator",
            "familywise_scope",
            "practical_thresholds",
            "selection_cells",
            "selection_cell_count",
            "confirmation_cells",
            "confirmation_cell_count",
            "selection_and_confirmation_disjoint_by_construction",
            "frozen_before_selection",
            "selection_cases_materialized",
            "confirmation_cases_materialized",
            "formal_cases_materialized",
            "selection_authorized",
            "confirmation_authorized",
            "formal_study_authorized",
            "scientific_claim_authorized",
            "ijoc_submission_status",
            "receipt_payload_sha256",
        },
        label="simultaneous-inference specification",
    )
    if raw["schema"] != "v21e3r1_simultaneous_inference_spec_v2":
        _fail("simultaneous-inference specification schema drifted")
    payload_sha = _sha256(
        raw["receipt_payload_sha256"], label="simultaneous.receipt_payload_sha256"
    )
    core = dict(raw)
    core.pop("receipt_payload_sha256")
    if payload_sha != _sha256_bytes(_canonical_bytes(core)):
        _fail("simultaneous-inference specification payload digest drifted")
    for field in (
        "successor_source_sha256",
        "successor_config_sha256",
        "study_metric_spec_sha256",
        "evaluator_source_sha256",
        "evaluator_test_sha256",
    ):
        _sha256(raw[field], label=f"simultaneous.{field}")
    evaluator_source = _verified_artifact(
        receipt_path.parent,
        path_value=raw["evaluator_source_path"],
        sha256_value=raw["evaluator_source_sha256"],
        label="simultaneous.evaluator_source",
    )
    evaluator_test = _verified_artifact(
        receipt_path.parent,
        path_value=raw["evaluator_test_path"],
        sha256_value=raw["evaluator_test_sha256"],
        label="simultaneous.evaluator_test",
    )
    if len({receipt_path, evaluator_source, evaluator_test}) != 3:
        _fail("simultaneous spec and bound evaluator artifacts must be distinct")
    try:
        evaluator_text = evaluator_source.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        _fail(f"simultaneous evaluator source is not UTF-8: {error}")
    method = "one_sided_observed_se_max_t_paired_case_cluster_bootstrap_v1"
    required_source_tokens = (
        f'METHOD = "{method}"',
        'FAMILIES = ("MOKP", "MOTSP")',
        'CANDIDATES = ("C0", "C1", "C2", "C3")',
        '"domain": "v21e3r1-simultaneous-case-bootstrap-v1"',
        '"rng": "SHA256_COUNTER_U64_REJECTION_V1"',
        '"centering": "BOOTSTRAP_MEAN_MINUS_OBSERVED_MEAN"',
        '"studentization_denominator": "OBSERVED_CASE_CLUSTER_STANDARD_ERROR"',
    )
    missing_tokens = [token for token in required_source_tokens if token not in evaluator_text]
    if missing_tokens:
        _fail(f"simultaneous evaluator source constants drifted: {missing_tokens}")
    identity_bindings = {
        "receipt_sha256": (raw_sha256, identity["simultaneous_inference_spec_sha256"]),
        "study_id": (raw["study_id"], identity["study_id"]),
        "candidate_id": (raw["candidate_id"], identity["candidate_id"]),
        "successor_source_sha256": (
            raw["successor_source_sha256"],
            identity["successor_source_sha256"],
        ),
        "successor_config_sha256": (
            raw["successor_config_sha256"],
            identity["successor_config_sha256"],
        ),
        "study_metric_spec_sha256": (
            raw["study_metric_spec_sha256"],
            identity["study_metric_spec_sha256"],
        ),
    }
    drifted_identity = [
        field for field, (actual, expected) in identity_bindings.items() if actual != expected
    ]
    if drifted_identity:
        _fail(f"simultaneous identity provenance drifted: {drifted_identity}")
    if raw["method"] != method:
        _fail("simultaneous method provenance drifted")
    for field, expected in (("bootstrap_samples", 10000), ("bootstrap_seed", 20260823)):
        if _integer(raw[field], label=field) != expected:
            _fail(f"simultaneous {field} provenance drifted")
    families = _string_list(raw["families"], label="simultaneous families")
    candidates = _string_list(raw["candidates"], label="simultaneous candidates")
    thresholds = _exact_keys(
        raw["practical_thresholds"],
        {"primary_effect", "adjacent_mechanism_effect"},
        label="practical thresholds",
    )
    for field in ("primary_effect", "adjacent_mechanism_effect"):
        _float(thresholds[field], label=f"practical thresholds.{field}")
    alpha = _float(raw["familywise_alpha"], label="familywise_alpha")
    _float(raw["critical_value_floor"], label="critical_value_floor")
    selection_cells = raw["selection_cells"]
    confirmation_cells = raw["confirmation_cells"]
    if type(selection_cells) is not list or type(confirmation_cells) is not list:
        _fail("simultaneous selection/confirmation cells must be exact arrays")
    expected_selection = [
        {
            "hypothesis_id": f"{family}:{candidate}-{reference}",
            "family": family,
            "candidate": candidate,
            "reference": reference,
            "threshold_roles": roles,
        }
        for family in FAMILIES
        for candidate, reference, roles in (
            ("C1", "C0", ["primary", "adjacent"]),
            ("C2", "C0", ["primary"]),
            ("C2", "C1", ["adjacent"]),
            ("C3", "C0", ["primary"]),
            ("C3", "C2", ["adjacent"]),
        )
    ]
    expected_confirmation = [
        {
            "hypothesis_template": f"{family}:SELECTED-{reference}",
            "family": family,
            "candidate": "SELECTED",
            "reference": reference,
            "threshold_role": role,
        }
        for family in FAMILIES
        for reference, role in (("C0", "primary"), ("PREDECESSOR", "adjacent"))
    ]
    for index, cell in enumerate(selection_cells):
        _exact_keys(
            cell,
            {"hypothesis_id", "family", "candidate", "reference", "threshold_roles"},
            label=f"simultaneous selection_cells[{index}]",
        )
    for index, cell in enumerate(confirmation_cells):
        _exact_keys(
            cell,
            {"hypothesis_template", "family", "candidate", "reference", "threshold_role"},
            label=f"simultaneous confirmation_cells[{index}]",
        )
    if families != list(FAMILIES) or candidates != ["C0", "C1", "C2", "C3"]:
        _fail("simultaneous family/candidate provenance drifted")
    selection_cell_count = _integer(
        raw["selection_cell_count"], label="selection_cell_count"
    )
    confirmation_cell_count = _integer(
        raw["confirmation_cell_count"], label="confirmation_cell_count"
    )
    if (
        selection_cells != expected_selection
        or confirmation_cells != expected_confirmation
        or selection_cell_count != 10
        or confirmation_cell_count != 4
    ):
        _fail("simultaneous cell cardinality/design provenance drifted")
    legacy_c1_cells = [
        cell for cell in selection_cells if cell["candidate"] == "C1"
    ]
    legacy_c1_roles = [
        f"{cell['hypothesis_id']}:{role}"
        for cell in legacy_c1_cells
        for role in cell["threshold_roles"]
    ]
    if (
        len(legacy_c1_cells) != 2
        or len(legacy_c1_roles) != 4
        or len({cell["hypothesis_id"] for cell in legacy_c1_cells}) != 2
    ):
        _fail("legacy C1 must bind two statistical contrasts and four decision roles")
    return bool(
        raw["status"] == "PASS_FROZEN_BEFORE_SELECTION_ENGINEERING_ONLY"
        and raw["scope"] == "FROZEN_PROSPECTIVE_DESIGN_ONLY_NO_CASE_MATERIALIZATION"
        and alpha == 0.05
        and raw["quantile_convention"]
        == "CEIL((1-ALPHA)*(B+1))_ORDER_STATISTIC"
        and raw["critical_value_floor"] == 0.0
        and raw["rng_protocol"] == "SHA256_COUNTER_U64_REJECTION_V1"
        and raw["rng_domain"] == "v21e3r1-simultaneous-case-bootstrap-v1"
        and raw["cluster_unit"] == "PAIRED_CASE"
        and raw["seed_aggregation"] == "MEAN_WITHIN_CASE_ARM"
        and raw["resampling_rule"]
        == "WITH_REPLACEMENT_INDEPENDENT_BY_FAMILY_SHARED_ACROSS_CELLS_WITHIN_FAMILY"
        and raw["centering"] == "BOOTSTRAP_MEAN_MINUS_OBSERVED_MEAN"
        and raw["studentization_denominator"]
        == "OBSERVED_CASE_CLUSTER_STANDARD_ERROR"
        and raw["familywise_scope"] == "JOINT_ACROSS_BOTH_FAMILIES"
        and thresholds["primary_effect"] == 0.0
        and thresholds["adjacent_mechanism_effect"] == 0.005
        and identity["candidate_id"] in candidates
        and _boolean(
            raw["selection_and_confirmation_disjoint_by_construction"],
            label="selection_and_confirmation_disjoint_by_construction",
        )
        and _boolean(raw["frozen_before_selection"], label="frozen_before_selection")
        and not _boolean(
            raw["selection_cases_materialized"],
            label="selection_cases_materialized",
        )
        and not _boolean(
            raw["confirmation_cases_materialized"],
            label="confirmation_cases_materialized",
        )
        and not _boolean(raw["formal_cases_materialized"], label="formal_cases_materialized")
        and not _boolean(raw["selection_authorized"], label="simultaneous.selection_authorized")
        and not _boolean(raw["confirmation_authorized"], label="simultaneous.confirmation_authorized")
        and not _boolean(raw["formal_study_authorized"], label="simultaneous.formal_study_authorized")
        and not _boolean(raw["scientific_claim_authorized"], label="simultaneous.scientific_claim_authorized")
        and raw["ijoc_submission_status"] == "IJOC_HOLD"
    )


_PHASE_RESULT_KEYS = {
    "schema",
    "status",
    "phase",
    "study_id",
    "input_sha256",
    "study_freeze_sha256",
    "phase_manifest_sha256",
    "matrix_receipt_sha256",
    "source_root_sha256",
    "metric_spec_sha256",
    "decision_spec_sha256",
    "source_sha256",
    "effect_direction",
    "families",
    "candidate_order",
    "case_count_by_family",
    "matrix_row_count",
    "expected_matrix_row_count",
    "seeds",
    "thresholds",
    "hypothesis_order",
    "cells",
    "inference",
    "simultaneous_coverage_certified",
    "selected_candidate",
    "reached_candidates",
    "not_reached_candidates",
    "blocked_candidate",
    "gate_reasons",
    "selection_binding",
    "confirmation_control_bindings",
    "confirmation_control_bindings_validated",
    "confirmation_control_bindings_scope",
    "statistics_implementation_independent_from_mo_nco",
    "external_independence_claim_authorized",
    "scientific_independence",
    "formal_authority",
    "receipt_payload_sha256",
}


def _validate_phase_result(
    receipt: Mapping[str, object],
    *,
    phase: str,
    identity: Mapping[str, str],
    source_freeze_sha256: str,
    expected_statistics_source_sha256: str,
    selection_result_sha256: str | None = None,
    external_replay_sha256: str | None = None,
    custody_receipt_sha256: str | None = None,
) -> bool:
    if receipt.get("schema") == "v21e3r1_path_bound_phase_evidence_contract_v3":
        return _validate_path_bound_phase_v3_contract(
            receipt, identity, phase=phase
        )
    raw = _exact_keys(receipt, _PHASE_RESULT_KEYS, label=f"{phase} result receipt")
    if raw["schema"] != "v21e3r1_independent_simultaneous_inference_receipt_v1":
        _fail(f"{phase} result receipt schema drifted")
    payload_sha = _sha256(
        raw["receipt_payload_sha256"], label=f"{phase}.receipt_payload_sha256"
    )
    core = dict(raw)
    core.pop("receipt_payload_sha256")
    if payload_sha != _sha256_bytes(_canonical_bytes(core)):
        _fail(f"{phase} result receipt payload digest drifted")
    for field in (
        "input_sha256",
        "study_freeze_sha256",
        "phase_manifest_sha256",
        "matrix_receipt_sha256",
        "source_root_sha256",
        "metric_spec_sha256",
        "decision_spec_sha256",
        "source_sha256",
    ):
        _sha256(raw[field], label=f"{phase}.{field}")
    expected_statistics_source_sha256 = _sha256(
        expected_statistics_source_sha256,
        label=f"{phase}.expected_statistics_source_sha256",
    )
    if raw["source_sha256"] != expected_statistics_source_sha256:
        _fail(f"{phase} statistics source does not match the frozen evaluator")
    families = _string_list(raw["families"], label=f"{phase}.families")
    candidates = _string_list(
        raw["candidate_order"], label=f"{phase}.candidate_order"
    )
    selected = _identifier(raw["selected_candidate"], label=f"{phase}.selected_candidate")
    if selected not in candidates:
        _fail(f"{phase} selected candidate is absent from candidate_order")
    case_counts = _exact_keys(
        raw["case_count_by_family"], set(FAMILIES), label=f"{phase}.case_count_by_family"
    )
    total_cases = sum(
        _integer(case_counts[family], label=f"{phase}.{family} case count", minimum=1)
        for family in FAMILIES
    )
    seeds = raw["seeds"]
    if type(seeds) is not list or not seeds:
        _fail(f"{phase}.seeds must be a nonempty array")
    if any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        _fail(f"{phase}.seeds must be unique exact nonnegative integers")
    matrix_rows = _integer(raw["matrix_row_count"], label=f"{phase}.matrix_row_count")
    expected_rows = _integer(
        raw["expected_matrix_row_count"], label=f"{phase}.expected_matrix_row_count"
    )
    if matrix_rows != expected_rows or expected_rows != total_cases * len(seeds) * len(candidates):
        _fail(f"{phase} matrix cardinality is not exact")
    thresholds = raw["thresholds"]
    if type(thresholds) is not dict or not thresholds:
        _fail(f"{phase}.thresholds must be a nonempty object")
    for name, value in thresholds.items():
        _string(name, label=f"{phase} threshold name")
        _float(value, label=f"{phase}.thresholds.{name}")
    hypotheses = _string_list(
        raw["hypothesis_order"], label=f"{phase}.hypothesis_order"
    )
    cells = raw["cells"]
    if type(cells) is not list or not cells or any(type(cell) is not dict for cell in cells):
        _fail(f"{phase}.cells must contain nonempty objects")
    if len(hypotheses) < len(cells):
        _fail(f"{phase} hypothesis order cannot be shorter than cells")
    if type(raw["inference"]) is not dict or not raw["inference"]:
        _fail(f"{phase}.inference must be a nonempty object")
    reached = _string_list(raw["reached_candidates"], label=f"{phase}.reached_candidates")
    _string_list(
        raw["not_reached_candidates"],
        label=f"{phase}.not_reached_candidates",
        nonempty=False,
    )
    reasons = raw["gate_reasons"]
    if type(reasons) is not list or any(type(reason) is not str or not reason for reason in reasons):
        _fail(f"{phase}.gate_reasons must contain exact strings")
    common = bool(
        raw["phase"] == phase
        and raw["study_id"] == identity["study_id"]
        and raw["study_freeze_sha256"] == source_freeze_sha256
        and raw["source_root_sha256"] == identity["successor_source_sha256"]
        and raw["metric_spec_sha256"] == identity["study_metric_spec_sha256"]
        and raw["decision_spec_sha256"]
        == identity["simultaneous_inference_spec_sha256"]
        and raw["effect_direction"] == "larger_is_better"
        and families == list(FAMILIES)
        and selected == identity["candidate_id"]
        and selected in reached
        and raw["blocked_candidate"] is None
        and not reasons
        and _boolean(
            raw["simultaneous_coverage_certified"],
            label=f"{phase}.simultaneous_coverage_certified",
        )
        and _boolean(
            raw["statistics_implementation_independent_from_mo_nco"],
            label=f"{phase}.statistics_implementation_independent_from_mo_nco",
        )
        and not _boolean(
            raw["external_independence_claim_authorized"],
            label=f"{phase}.external_independence_claim_authorized",
        )
        and not _boolean(
            raw["scientific_independence"], label=f"{phase}.scientific_independence"
        )
        and not _boolean(raw["formal_authority"], label=f"{phase}.formal_authority")
    )
    if phase == "selection":
        structurally_valid = bool(
            common
            and raw["status"] == "PASS_SELECTION"
            and raw["selection_binding"] is None
            and raw["confirmation_control_bindings"] is None
            and not _boolean(
                raw["confirmation_control_bindings_validated"],
                label="selection.confirmation_control_bindings_validated",
            )
            and raw["confirmation_control_bindings_scope"] is None
        )
        if not structurally_valid:
            _fail("legacy v1 selection result boundary drifted")
        # Schema v1 has no path-bound input, phase-manifest, or matrix artifacts
        # and therefore cannot clear a later-phase authority gate.
        return False
    if phase != "confirmation":
        _fail("phase result validator received an unsupported phase")
    if selection_result_sha256 is None or external_replay_sha256 is None or custody_receipt_sha256 is None:
        _fail("confirmation validation omitted required receipt hashes")
    binding = _exact_keys(
        raw["selection_binding"],
        {"selection_receipt_sha256", "selection_status", "selected_candidate"},
        label="confirmation.selection_binding",
    )
    _sha256(
        binding["selection_receipt_sha256"],
        label="confirmation.selection_receipt_sha256",
    )
    controls = _exact_keys(
        raw["confirmation_control_bindings"],
        {
            "external_producer",
            "external_producer_receipt_sha256",
            "independent_custody",
            "custody_receipt_sha256",
            "statistics_source_sha256",
        },
        label="confirmation.control_bindings",
    )
    for field in (
        "external_producer_receipt_sha256",
        "custody_receipt_sha256",
        "statistics_source_sha256",
    ):
        _sha256(controls[field], label=f"confirmation.controls.{field}")
    structurally_valid = bool(
        common
        and raw["status"] == "PASS_CONFIRMATION"
        and binding["selection_receipt_sha256"] == selection_result_sha256
        and binding["selection_status"] == "PASS_SELECTION"
        and binding["selected_candidate"] == identity["candidate_id"]
        and _boolean(
            controls["external_producer"], label="confirmation.controls.external_producer"
        )
        and controls["external_producer_receipt_sha256"] == external_replay_sha256
        and _boolean(
            controls["independent_custody"], label="confirmation.controls.independent_custody"
        )
        and controls["custody_receipt_sha256"] == custody_receipt_sha256
        and controls["statistics_source_sha256"] == raw["source_sha256"]
        and _boolean(
            raw["confirmation_control_bindings_validated"],
            label="confirmation.confirmation_control_bindings_validated",
        )
        and raw["confirmation_control_bindings_scope"]
        == "INPUT_DECLARATIONS_AND_HASH_BINDINGS_ONLY_NOT_AUTHENTICATION"
    )
    if not structurally_valid:
        _fail("legacy v1 confirmation result boundary drifted")
    # Schema v1 validates as hash-bound metadata only.  A future audited,
    # path-bound schema is required before confirmation can clear authority.
    return False


def _load_bound_receipts(
    *, root: Path, bindings_value: object, expected: set[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    bindings = _exact_keys(bindings_value, expected, label="gate spec bindings")
    receipts: dict[str, dict[str, Any]] = {}
    evidence: dict[str, dict[str, str]] = {}
    observed_paths: set[Path] = set()
    for binding_id in sorted(expected):
        binding = _exact_keys(
            bindings[binding_id], BINDING_KEYS, label=f"binding {binding_id}"
        )
        declared_sha = _sha256(binding["sha256"], label=f"binding {binding_id}.sha256")
        path = _contained_file(root, binding["path"], label=f"binding {binding_id}.path")
        if path in observed_paths:
            _fail("two evidence bindings resolve to the same file")
        observed_paths.add(path)
        raw = path.read_bytes()
        observed_sha = _sha256_bytes(raw)
        if observed_sha != declared_sha:
            _fail(f"binding {binding_id} SHA-256 disagrees with its file")
        receipt = _parse_json(
            raw,
            label=f"binding {binding_id}",
            canonical=True,
            allow_canonical_pretty=True,
            allow_canonical_newline=binding_id == "successor_development_promotion",
        )
        receipts[binding_id] = receipt
        evidence[binding_id] = {
            "path": str(binding["path"]),
            "sha256": observed_sha,
            "schema": _string(receipt.get("schema"), label=f"{binding_id}.schema"),
            "status": _string(receipt.get("status"), label=f"{binding_id}.status"),
        }
    return receipts, evidence


def _evaluate(
    *, spec: Mapping[str, object], spec_sha256: str, root: Path
) -> tuple[dict[str, object], int]:
    raw_spec = _exact_keys(
        spec,
        {"schema", "requested_authorization", "identity", "bindings"},
        label="gate specification",
    )
    if raw_spec["schema"] != SPEC_SCHEMA:
        _fail("gate specification schema drifted")
    requested = _string(
        raw_spec["requested_authorization"], label="requested_authorization"
    )
    if requested not in REQUESTS:
        _fail("requested_authorization is unsupported")
    identity = _identity(raw_spec["identity"])
    expected_bindings = set(COMMON_BINDINGS)
    if requested in {"confirmation", "formal_input_materialization"}:
        expected_bindings.add("selection_result")
    if requested == "formal_input_materialization":
        expected_bindings.add("confirmation_result")
    receipts, evidence = _load_bound_receipts(
        root=root, bindings_value=raw_spec["bindings"], expected=expected_bindings
    )

    historical = _validate_historical(receipts["historical_preservation"])
    diagnostic = _validate_diagnostic(receipts["exact_504_diagnostic"], identity)
    reanalysis = _validate_reanalysis(
        receipts["corrected_reanalysis"],
        identity,
        receipts["exact_504_diagnostic"],
        evidence["exact_504_diagnostic"]["sha256"],
        _contained_file(
            root,
            evidence["corrected_reanalysis"]["path"],
            label="corrected reanalysis receipt path",
        ),
    )
    source_receipt_path = _contained_file(
        root,
        evidence["successor_source_freeze"]["path"],
        label="successor source-freeze receipt path",
    )
    source = _validate_source_freeze(
        receipts["successor_source_freeze"],
        identity,
        source_receipt_path,
    )
    promotion_receipt_path = _contained_file(
        root,
        evidence["successor_development_promotion"]["path"],
        label="successor development promotion receipt path",
    )
    promotion = _validate_development_promotion(
        receipts["successor_development_promotion"],
        identity,
        receipts["successor_source_freeze"],
        evidence["successor_source_freeze"]["sha256"],
        source_receipt_path,
        promotion_receipt_path,
        root,
    )
    same_impl = _validate_same_implementation(
        receipts["same_implementation_coverage"],
        identity,
        receipts["exact_504_diagnostic"],
        evidence["exact_504_diagnostic"]["sha256"],
        _contained_file(
            root,
            evidence["same_implementation_coverage"]["path"],
            label="same-implementation coverage receipt path",
        ),
        root,
    )
    baselines, baseline_counts = _validate_baselines(
        receipts["baseline_registry"],
        identity,
        _contained_file(
            root,
            evidence["baseline_registry"]["path"],
            label="baseline registry receipt path",
        ),
    )
    replay, producer, custody, disjoint, algorithm, custody_sha = _validate_external_replay(
        receipts["external_algorithm_replay"],
        identity,
        _contained_file(
            root,
            evidence["external_algorithm_replay"]["path"],
            label="external algorithm replay receipt path",
        ),
    )
    simultaneous = _validate_simultaneous_spec(
        receipts["simultaneous_inference_spec"],
        identity,
        evidence["simultaneous_inference_spec"]["sha256"],
        _contained_file(
            root,
            evidence["simultaneous_inference_spec"]["path"],
            label="simultaneous-inference specification path",
        ),
    )
    frozen_statistics_source_sha256 = _sha256(
        receipts["simultaneous_inference_spec"]["evaluator_source_sha256"],
        label="simultaneous.evaluator_source_sha256",
    )

    selection_result = False
    if "selection_result" in receipts:
        selection_result = _validate_phase_result(
            receipts["selection_result"],
            phase="selection",
            identity=identity,
            source_freeze_sha256=evidence["successor_source_freeze"]["sha256"],
            expected_statistics_source_sha256=frozen_statistics_source_sha256,
        )
    confirmation_result = False
    if "confirmation_result" in receipts:
        confirmation_result = _validate_phase_result(
            receipts["confirmation_result"],
            phase="confirmation",
            identity=identity,
            source_freeze_sha256=evidence["successor_source_freeze"]["sha256"],
            expected_statistics_source_sha256=frozen_statistics_source_sha256,
            selection_result_sha256=evidence["selection_result"]["sha256"],
            external_replay_sha256=evidence["external_algorithm_replay"]["sha256"],
            custody_receipt_sha256=custody_sha,
        )

    gates: dict[str, bool] = {
        "historical_v4_v6_preserved": historical,
        "exact_504_development_diagnostic": diagnostic,
        "corrected_reanalysis": reanalysis,
        "successor_source_and_config_frozen": source,
        "successor_development_promotion_gate_passed": promotion,
        "same_implementation_branch_coverage": same_impl,
        "external_family_native_strong_baseline_each_family": baselines,
        "external_algorithm_replay": replay,
        "simultaneous_inference_spec_frozen": simultaneous,
        "selection_result": selection_result,
        "independent_producer": producer,
        "independent_custody": custody,
        "implementation_code_disjoint": disjoint,
        "algorithm_execution_independence": algorithm,
        "confirmation_result": confirmation_result,
    }
    common_names = (
        "historical_v4_v6_preserved",
        "exact_504_development_diagnostic",
        "corrected_reanalysis",
        "successor_source_and_config_frozen",
        "successor_development_promotion_gate_passed",
        "same_implementation_branch_coverage",
        "external_family_native_strong_baseline_each_family",
        "external_algorithm_replay",
        "simultaneous_inference_spec_frozen",
        "independent_producer",
        "independent_custody",
        "implementation_code_disjoint",
        "algorithm_execution_independence",
    )
    common_pass = all(gates[name] for name in common_names)
    selection_authorized = common_pass
    confirmation_authorized = bool(common_pass and selection_result)
    formal_authorized = bool(confirmation_authorized and confirmation_result)

    requested_pass = (
        selection_authorized
        if requested == "selection"
        else confirmation_authorized
        if requested == "confirmation"
        else formal_authorized
    )
    status = (
        "PASS_SELECTION_AUTHORIZATION"
        if requested == "selection" and requested_pass
        else "PASS_CONFIRMATION_AUTHORIZATION"
        if requested == "confirmation" and requested_pass
        else "PASS_FORMAL_INPUT_MATERIALIZATION_AUTHORIZATION"
        if requested == "formal_input_materialization" and requested_pass
        else HOLD_STATUS
    )
    reasons = [name for name in common_names if not gates[name]]
    if requested in {"confirmation", "formal_input_materialization"}:
        for name in ("selection_result",):
            if not gates[name]:
                reasons.append(name)
    if requested == "formal_input_materialization" and not gates["confirmation_result"]:
        reasons.append("confirmation_result")
    core: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "status": status,
        "requested_authorization": requested,
        "gate_spec_sha256": spec_sha256,
        "evaluator_source_sha256": _sha256_file(Path(__file__).resolve()),
        "identity": dict(identity),
        "evidence": evidence,
        "gates": gates,
        "baseline_eligible_count_by_family": baseline_counts,
        "selection_authorized": selection_authorized,
        "confirmation_authorized": confirmation_authorized,
        "formal_input_materialization_authorized": formal_authorized,
        "hold_reasons": reasons,
        "bound_custody_receipt_sha256": custody_sha,
        "authorization_scope": "AUTHORIZATION_DECISION_ONLY_NO_EXECUTION_OR_CASE_GENERATION",
        "case_generation_performed": False,
        "generated_case_count": 0,
        "formal_study_executed": False,
        "scientific_claim_authorized": False,
        "ijoc_submission_authorized": False,
    }
    receipt = dict(core)
    receipt["receipt_payload_sha256"] = _sha256_bytes(_canonical_bytes(core))
    return receipt, 0 if requested_pass else 2


def _write_exclusive(path: Path, receipt: Mapping[str, object]) -> None:
    raw = json.dumps(
        receipt,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        _fail("output receipt already exists; exclusive create required")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-spec", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    try:
        root = Path(arguments.evidence_root).resolve(strict=True)
        if not root.is_dir():
            _fail("evidence root is not a directory")
        spec_path = Path(arguments.gate_spec).resolve(strict=True)
        try:
            spec_path.relative_to(root)
        except ValueError:
            _fail("gate specification escapes evidence root")
        output = _contained_output(root, Path(arguments.output))
        spec_raw = spec_path.read_bytes()
        spec = _parse_json(spec_raw, label="gate specification", canonical=True)
        receipt, exit_code = _evaluate(
            spec=spec,
            spec_sha256=_sha256_bytes(spec_raw),
            root=root,
        )
        _write_exclusive(output, receipt)
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        return exit_code
    except (AuthorizationError, OSError) as error:
        print(
            json.dumps(
                {
                    "schema": "v21e3r1_prospective_authorization_error_v1",
                    "status": "HOLD_INTEGRITY_ERROR",
                    "error": str(error),
                    "case_generation_performed": False,
                    "generated_case_count": 0,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
