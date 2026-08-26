#!/usr/bin/env python3
from __future__ import annotations

"""Build one hash-bound V21e3r1 prospective authorization gate specification.

The builder derives identity only from the supplied evidence.  It does not
accept authority declarations, execute an authorization decision, or create
selection, confirmation, or formal-study cases.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, NoReturn, Sequence


SPEC_SCHEMA = "v21e3r1_prospective_authorization_gate_spec_v3"
REQUESTS = {
    "selection",
    "confirmation",
    "formal_input_materialization",
}
COMMON_BINDINGS = (
    "historical_preservation",
    "exact_504_diagnostic",
    "corrected_reanalysis",
    "successor_source_freeze",
    "successor_development_promotion",
    "same_implementation_coverage",
    "baseline_registry",
    "external_algorithm_replay",
    "simultaneous_inference_spec",
)
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


class GateSpecBuildError(ValueError):
    """An input path or identity binding violated the builder contract."""


@dataclass(frozen=True)
class FileRecord:
    path: Path
    relative_path: str
    raw: bytes
    sha256: str
    identity: tuple[int, int, int, int, int, int]
    label: str


def _fail(message: str) -> NoReturn:
    raise GateSpecBuildError(message)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (UnicodeEncodeError, ValueError) as error:
        _fail(f"canonical JSON encoding failed: {error}")


def _canonical_pretty_bytes(value: object) -> bytes:
    try:
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
    except (UnicodeEncodeError, ValueError) as error:
        _fail(f"canonical JSON encoding failed: {error}")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _reject_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON number is prohibited: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key is prohibited: {key!r}")
        result[key] = value
    return result


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


def _parse_canonical_json(
    record: FileRecord, *, allow_compact_newline: bool = False
) -> dict[str, Any]:
    try:
        text = record.raw.decode("utf-8")
    except UnicodeDecodeError as error:
        _fail(f"{record.label} is not strict UTF-8 JSON: {error}")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except GateSpecBuildError:
        raise
    except (json.JSONDecodeError, RecursionError) as error:
        _fail(f"{record.label} is malformed JSON: {error}")
    if type(value) is not dict:
        _fail(f"{record.label} must be a JSON object")
    _validate_json_tree(value, label=record.label)
    encodings = {_canonical_bytes(value), _canonical_pretty_bytes(value)}
    if allow_compact_newline:
        encodings.add(_canonical_bytes(value) + b"\n")
    if record.raw not in encodings:
        _fail(f"{record.label} is not canonical JSON")
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


def _number(value: object, *, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        _fail(f"{label} must be an exact finite JSON number")
    return float(value)


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{label} must be an exact JSON object")
    return value


def _exact_keys(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    result = _mapping(value, label=label)
    observed = set(result)
    if observed != expected:
        _fail(
            f"{label} key set drifted; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    return result


def _schema(receipt: Mapping[str, object], expected: str, *, label: str) -> None:
    if receipt.get("schema") != expected:
        _fail(f"{label} schema drifted")


def _verify_payload_digest(receipt: Mapping[str, object], *, label: str) -> None:
    declared = _sha256(
        receipt.get("receipt_payload_sha256"),
        label=f"{label}.receipt_payload_sha256",
    )
    core = dict(receipt)
    core.pop("receipt_payload_sha256", None)
    if declared != _sha256_bytes(_canonical_bytes(core)):
        _fail(f"{label} receipt payload digest drifted")


def _stat_identity(result: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(result.st_dev),
        int(result.st_ino),
        int(result.st_mode),
        int(result.st_size),
        int(result.st_mtime_ns),
        int(result.st_ctime_ns),
    )


def _is_link_or_reparse(result: os.stat_result) -> bool:
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    attributes = int(getattr(result, "st_file_attributes", 0))
    return stat.S_ISLNK(result.st_mode) or bool(
        reparse_flag and attributes & reparse_flag
    )


def _lstat_is_link_or_reparse(path: Path) -> bool:
    try:
        return _is_link_or_reparse(path.lstat())
    except FileNotFoundError:
        return False
    except OSError as error:
        _fail(f"path component cannot be inspected: {path}: {error}")


def _reject_lexical_traversal(path: Path, *, label: str) -> None:
    if any(part in {".", ".."} for part in path.parts):
        _fail(f"{label} contains a traversal segment")


def _evidence_root(value: str | Path) -> Path:
    raw = Path(value)
    _reject_lexical_traversal(raw, label="evidence root")
    absolute = Path(os.path.abspath(raw))
    try:
        before = absolute.lstat()
    except OSError as error:
        _fail(f"evidence root cannot be inspected: {error}")
    if _is_link_or_reparse(before) or not stat.S_ISDIR(before.st_mode):
        _fail("evidence root must be an existing non-link directory")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        _fail(f"evidence root cannot be resolved: {error}")
    if resolved != absolute:
        _fail("evidence root must not resolve through an alias or link")
    return resolved


def _canonical_relative(path: Path, *, root: Path, label: str) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        _fail(f"{label} escapes evidence root")
    raw = PurePosixPath(*relative.parts).as_posix()
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if (
        not raw
        or "\\" in raw
        or any(ord(character) < 32 for character in raw)
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or posix.as_posix() != raw
        or any(part in {"", ".", ".."} for part in posix.parts)
        or ":" in posix.parts[0]
    ):
        _fail(f"{label} is not a canonical contained POSIX path")
    return raw


def _contained_file(root: Path, value: str | Path, *, label: str) -> Path:
    raw_path = Path(value)
    _reject_lexical_traversal(raw_path, label=label)
    absolute = Path(
        os.path.abspath(raw_path if raw_path.is_absolute() else root / raw_path)
    )
    try:
        relative = absolute.relative_to(root)
    except ValueError:
        _fail(f"{label} escapes evidence root")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if _lstat_is_link_or_reparse(cursor):
            _fail(f"{label} traverses a symbolic link, junction, or reparse point")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        _fail(f"{label} is absent: {error}")
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(f"{label} escapes evidence root")
    if not resolved.is_file():
        _fail(f"{label} is not a regular file")
    return resolved


def _canonical_embedded_path(value: object, *, label: str) -> PurePosixPath:
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


def _stable_read(root: Path, path: Path, *, label: str) -> FileRecord:
    contained = _contained_file(root, path, label=label)
    try:
        before = contained.lstat()
        raw = contained.read_bytes()
        after = contained.lstat()
    except OSError as error:
        _fail(f"{label} cannot be read: {error}")
    if _is_link_or_reparse(before) or _is_link_or_reparse(after):
        _fail(f"{label} must not be a symbolic link, junction, or reparse point")
    if not stat.S_ISREG(before.st_mode) or not stat.S_ISREG(after.st_mode):
        _fail(f"{label} is not a regular file")
    before_identity = _stat_identity(before)
    after_identity = _stat_identity(after)
    if before_identity != after_identity or len(raw) != after.st_size:
        _fail(f"{label} changed while it was being read")
    if not raw:
        _fail(f"{label} must not be empty")
    return FileRecord(
        path=contained,
        relative_path=_canonical_relative(contained, root=root, label=label),
        raw=raw,
        sha256=_sha256_bytes(raw),
        identity=after_identity,
        label=label,
    )


def _revalidate_records(root: Path, records: Sequence[FileRecord]) -> None:
    for frozen in records:
        observed = _stable_read(root, frozen.path, label=frozen.label)
        if observed.identity != frozen.identity or observed.sha256 != frozen.sha256:
            _fail(f"{frozen.label} changed after identity derivation")


def _read_embedded_root_path(
    *, root: Path, value: object, label: str
) -> FileRecord:
    relative = _canonical_embedded_path(value, label=label)
    return _stable_read(root, root.joinpath(*relative.parts), label=label)


def _read_fixed_sibling(
    *, root: Path, parent: Path, name: str, label: str
) -> FileRecord:
    if Path(name).name != name:
        _fail(f"{label} fixed sibling name is invalid")
    return _stable_read(root, parent / name, label=label)


def _validate_recovery_bound_same_implementation(
    *,
    receipt: Mapping[str, object],
    identity: Mapping[str, str],
    diagnostic_receipt: Mapping[str, object],
    diagnostic_receipt_sha256: str,
    receipt_path: Path,
    evidence_root: Path,
) -> None:
    validator_path = Path(__file__).with_name(
        "evaluate_v21e3r1_prospective_authorization.py"
    ).resolve(strict=True)
    spec = importlib.util.spec_from_file_location(
        "_v21e3r1_prospective_authorization_builder_validator", validator_path
    )
    if spec is None or spec.loader is None:
        _fail("cannot load the prospective authorization validator")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        valid = module._validate_same_implementation(
            receipt,
            identity,
            diagnostic_receipt,
            diagnostic_receipt_sha256,
            receipt_path,
            evidence_root,
        )
    except Exception as error:
        if error.__class__.__name__ == "AuthorizationError":
            _fail(f"recovery-bound same-implementation chain failed: {error}")
        raise
    if valid is not True:
        _fail("recovery-bound same-implementation chain did not pass")


def _agree(observed: object, expected: str, *, label: str, digest: bool = True) -> str:
    result = _sha256(observed, label=label) if digest else _identifier(observed, label=label)
    if result != expected:
        _fail(f"{label} disagrees with the derived gate identity")
    return result


def _validate_source_manifest_value(
    value: Mapping[str, object], *, expected_root: str, label: str
) -> None:
    manifest = _exact_keys(
        value,
        {"schema", "source_root_sha256", "entries"},
        label=label,
    )
    _schema(
        manifest,
        "v21e3r1_branch_replay_source_manifest_binding_v1",
        label=label,
    )
    declared_root = _sha256(
        manifest.get("source_root_sha256"), label=f"{label}.source_root_sha256"
    )
    entries = manifest.get("entries")
    if type(entries) is not list or not entries:
        _fail(f"{label}.entries must be an exact nonempty array")
    paths: list[str] = []
    for index, value_entry in enumerate(entries):
        entry = _exact_keys(
            value_entry,
            {"path", "bytes", "sha256"},
            label=f"{label}.entries[{index}]",
        )
        relative = _canonical_embedded_path(
            entry.get("path"), label=f"{label}.entries[{index}].path"
        ).as_posix()
        paths.append(relative)
        _integer(
            entry.get("bytes"),
            label=f"{label}.entries[{index}].bytes",
            minimum=1,
        )
        _sha256(
            entry.get("sha256"), label=f"{label}.entries[{index}].sha256"
        )
    ordered = sorted(paths, key=lambda item: (item.casefold(), item))
    if paths != ordered:
        _fail(f"{label}.entries are not in canonical casefold path order")
    if len({path.casefold() for path in paths}) != len(paths):
        _fail(f"{label}.entries contain case-insensitive duplicate paths")
    computed_root = _sha256_bytes(_canonical_bytes(entries))
    if declared_root != computed_root:
        _fail(f"{label} canonical source-root digest drifted")
    if declared_root != expected_root:
        _fail(f"{label} source root disagrees with the derived gate identity")


def _validate_factorial_plan_design(
    plan: Mapping[str, object],
) -> list[dict[str, Any]]:
    _canonical_embedded_path(
        plan.get("parent_v7_diagnostic_plan_path"),
        label="promotion factorial parent V7 plan path",
    )
    if plan.get("parent_v7_diagnostic_plan_sha256") != FACTORIAL_PARENT_PLAN_SHA256:
        _fail("promotion factorial parent V7 plan SHA-256 drifted")
    if plan.get("case_ids") != list(FACTORIAL_CASE_IDS):
        _fail("promotion factorial exposed-development case boundary drifted")
    if plan.get("seeds") != list(FACTORIAL_SEEDS):
        _fail("promotion factorial seed boundary drifted")
    expected_arms = {
        family: [arm_id for arm_id, _search, _novelty in arms]
        for family, arms in FACTORIAL_ARMS.items()
    }
    if plan.get("arms_by_family") != expected_arms:
        _fail("promotion factorial arm boundary drifted")
    if plan.get("input_binding") != FACTORIAL_INPUT_BINDING:
        _fail("promotion factorial frozen input binding drifted")
    _integer(
        plan.get("row_timeout_seconds"),
        label="promotion factorial row_timeout_seconds",
        minimum=1,
    )

    rows = plan.get("rows")
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
            if row.get(field) != expected_value or type(row.get(field)) is not type(
                expected_value
            ):
                _fail(f"promotion factorial row design drifted: {index + 1}/{field}")
        _sha256(
            row.get("case_artifact_sha256"),
            label=f"promotion factorial rows[{index}].case_artifact_sha256",
        )
        validated.append(row)
    return validated


def _validate_factorial_aggregate(
    record: FileRecord,
    *,
    plan_sha256: str,
    plan_rows: Sequence[Mapping[str, object]],
) -> None:
    aggregate = _exact_keys(
        _parse_canonical_json(record, allow_compact_newline=True),
        FACTORIAL_AGGREGATE_KEYS,
        label="promotion factorial aggregate",
    )
    if record.raw != _canonical_bytes(aggregate) + b"\n":
        _fail("promotion factorial aggregate must be canonical JSON plus one LF")
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
        if aggregate.get(field) != expected or type(aggregate.get(field)) is not type(
            expected
        ):
            _fail(f"promotion factorial aggregate field drifted: {field}")
    rows = aggregate.get("rows")
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
            if row.get(field) != plan_row[field] or type(row.get(field)) is not type(
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
                row.get(field),
                label=f"promotion factorial aggregate rows[{index}].{field}",
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
                row.get(field),
                label=f"promotion factorial aggregate rows[{index}].{field}",
            )


def _validate_development_promotion_identity(
    *,
    root: Path,
    record: FileRecord,
    receipt: Mapping[str, object],
    identity: Mapping[str, str],
    source_record: FileRecord,
    source_manifest_record: FileRecord,
) -> list[FileRecord]:
    label = "successor development promotion receipt"
    raw = _exact_keys(receipt, PROMOTION_KEYS, label=label)
    if record.raw != _canonical_bytes(raw) + b"\n":
        _fail(f"{label} must be canonical compact JSON plus one LF")
    _schema(raw, PROMOTION_SCHEMA, label=label)
    _verify_payload_digest(raw, label=label)
    statuses = {PROMOTION_PASS_STATUS, *PROMOTION_HOLD_STATUSES}
    if raw.get("status") not in statuses:
        _fail("successor development promotion status drifted")
    promotion_gate = raw.get("development_promotion_gate_passed")
    if (
        type(promotion_gate) is not bool
        or promotion_gate is not (raw.get("status") == PROMOTION_PASS_STATUS)
    ):
        _fail("successor development promotion status/gate relationship drifted")
    expected_scalars: dict[str, object] = {
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
        if raw.get(field) != expected or type(raw.get(field)) is not type(expected):
            _fail(f"successor development promotion field drifted: {field}")
    expected_identity = {
        "study_id": identity["study_id"],
        "candidate_id": identity["candidate_id"],
        "successor_source_sha256": identity["successor_source_sha256"],
        "successor_config_sha256": identity["successor_config_sha256"],
        "source_freeze_receipt_sha256": source_record.sha256,
        "source_manifest_sha256": source_manifest_record.sha256,
        "study_metric_spec_sha256": identity["study_metric_spec_sha256"],
        "simultaneous_inference_spec_sha256": identity[
            "simultaneous_inference_spec_sha256"
        ],
    }
    for field, expected in expected_identity.items():
        _agree(
            raw.get(field),
            expected,
            label=f"promotion.{field}",
            digest=field not in {"study_id", "candidate_id"},
        )
    for field in (
        "matrix_plan_sha256",
        "matrix_receipt_sha256",
        "row_evidence_replay_sha256",
        "inference_spec_sha256",
    ):
        _sha256(raw.get(field), label=f"promotion.{field}")

    inference_record = _read_embedded_root_path(
        root=root,
        value=FACTORIAL_INFERENCE_RELATIVE.as_posix(),
        label="successor factorial inference specification",
    )
    if inference_record.sha256 != FACTORIAL_INFERENCE_SHA256:
        _fail("successor factorial inference specification SHA-256 drifted")
    inference = _parse_canonical_json(inference_record, allow_compact_newline=True)
    if inference_record.raw != _canonical_bytes(inference) + b"\n":
        _fail("successor factorial inference specification must end in one LF")
    if (
        inference.get("schema")
        != "v21e3r1_successor_development_factorial_inference_spec_v1"
        or inference.get("status")
        != "FROZEN_PROSPECTIVELY_BEFORE_SUCCESSOR_FACTORIAL_EXECUTION"
        or inference.get("method") != FACTORIAL_METHOD
        or inference.get("selection_cases_materialized") is not False
    ):
        _fail("successor factorial inference specification boundary drifted")

    matrix_relative = _canonical_embedded_path(
        raw.get("matrix_directory"), label="promotion.matrix_directory"
    )
    matrix_parent = root.joinpath(*matrix_relative.parts)
    plan_record = _stable_read(
        root, matrix_parent / "factorial.plan.json", label="promotion factorial plan"
    )
    matrix_receipt_record = _stable_read(
        root,
        matrix_parent / "factorial.receipt.json",
        label="promotion factorial matrix receipt",
    )
    aggregate_record = _stable_read(
        root,
        matrix_parent / "factorial.aggregate.json",
        label="promotion factorial aggregate",
    )
    _agree(
        raw.get("matrix_plan_sha256"),
        plan_record.sha256,
        label="promotion.matrix_plan_sha256",
    )
    _agree(
        raw.get("matrix_receipt_sha256"),
        matrix_receipt_record.sha256,
        label="promotion.matrix_receipt_sha256",
    )
    plan = _exact_keys(
        _parse_canonical_json(plan_record, allow_compact_newline=True),
        FACTORIAL_PLAN_KEYS,
        label="promotion factorial plan",
    )
    if plan_record.raw != _canonical_bytes(plan) + b"\n":
        _fail("promotion factorial plan must be canonical JSON plus one LF")
    plan_expected = {
        "schema": "v21e3r1_successor_development_factorial_plan_v2",
        "status": "FROZEN_EXACT_108_SUCCESSOR_DEVELOPMENT_FACTORIAL",
        "phase": "development",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "parent_v7_source_snapshot_sha256": identity["development_source_sha256"],
        "charged_evaluation_budget": 2000,
        "checkpoint_period": 200,
        "expected_rows": 108,
        "selection_cases_materialized": False,
        "implementation_independence": False,
        "scientific_independence": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_study_authorized": False,
        "scientific_claim_authorized": False,
        "ijoc_submission_status": "IJOC_HOLD",
    }
    for field, expected in plan_expected.items():
        if plan.get(field) != expected or type(plan.get(field)) is not type(expected):
            _fail(f"promotion factorial plan field drifted: {field}")
    plan_rows = _validate_factorial_plan_design(plan)
    source_binding = _exact_keys(
        plan.get("source_binding"),
        FACTORIAL_SOURCE_BINDING_KEYS,
        label="promotion factorial source binding",
    )
    source_expected = {
        "schema": "v21e3r1_successor_factorial_source_binding_v2",
        "study_id": identity["study_id"],
        "candidate_id": identity["candidate_id"],
        "parent_development_source_sha256": identity["development_source_sha256"],
        "receipt_sha256": source_record.sha256,
        "source_manifest_sha256": source_manifest_record.sha256,
        "source_snapshot_sha256": identity["successor_source_sha256"],
        "semantic_config_sha256": identity["successor_config_sha256"],
        "study_metric_spec_sha256": identity["study_metric_spec_sha256"],
        "simultaneous_inference_spec_sha256": identity[
            "simultaneous_inference_spec_sha256"
        ],
        "factorial_inference_spec_path": FACTORIAL_INFERENCE_RELATIVE.as_posix(),
        "factorial_inference_spec_sha256": FACTORIAL_INFERENCE_SHA256,
    }
    for field, expected in source_expected.items():
        if source_binding.get(field) != expected:
            _fail(f"promotion factorial source binding drifted: {field}")
    source_binding_record = _read_embedded_root_path(
        root=root,
        value=source_binding.get("receipt_path"),
        label="promotion factorial source-freeze binding",
    )
    if (
        source_binding_record.path != source_record.path
        or source_binding_record.sha256 != source_record.sha256
    ):
        _fail("promotion factorial source-freeze raw binding drifted")
    manifest_binding_record = _read_embedded_root_path(
        root=root,
        value=source_binding.get("source_manifest_path"),
        label="promotion factorial source-manifest binding",
    )
    if (
        manifest_binding_record.path != source_manifest_record.path
        or manifest_binding_record.sha256 != source_manifest_record.sha256
    ):
        _fail("promotion factorial source-manifest raw binding drifted")
    inference_binding = _exact_keys(
        plan.get("inference_spec_binding"),
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
        _fail("promotion factorial inference binding drifted")

    matrix_receipt = _exact_keys(
        _parse_canonical_json(matrix_receipt_record, allow_compact_newline=True),
        FACTORIAL_MATRIX_RECEIPT_KEYS,
        label="promotion factorial matrix receipt",
    )
    if matrix_receipt_record.raw != _canonical_bytes(matrix_receipt) + b"\n":
        _fail("promotion factorial matrix receipt must be canonical JSON plus one LF")
    _verify_payload_digest(matrix_receipt, label="promotion factorial matrix receipt")
    matrix_expected = {
        "schema": "v21e3r1_successor_development_factorial_receipt_v2",
        "status": "PASS_EXACT_108_SUCCESSOR_DEVELOPMENT_FACTORIAL_ENGINEERING_ONLY",
        "phase": "development",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "completed_rows": 108,
        "expected_rows": 108,
        "plan_sha256": plan_record.sha256,
        "aggregate_sha256": aggregate_record.sha256,
        "parent_v7_diagnostic_plan_sha256": FACTORIAL_PARENT_PLAN_SHA256,
        "parent_v7_source_snapshot_sha256": identity["development_source_sha256"],
        "study_id": identity["study_id"],
        "candidate_id": identity["candidate_id"],
        "successor_source_sha256": identity["successor_source_sha256"],
        "successor_config_sha256": identity["successor_config_sha256"],
        "source_freeze_receipt_sha256": source_record.sha256,
        "source_manifest_sha256": source_manifest_record.sha256,
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
        if (
            matrix_receipt.get(field) != expected
            or type(matrix_receipt.get(field)) is not type(expected)
        ):
            _fail(f"promotion factorial matrix receipt field drifted: {field}")
    _validate_factorial_aggregate(
        aggregate_record,
        plan_sha256=plan_record.sha256,
        plan_rows=plan_rows,
    )

    cells = raw.get("cells")
    hypotheses = inference.get("hypotheses")
    if (
        type(cells) is not list
        or len(cells) != len(FACTORIAL_HYPOTHESES)
        or type(hypotheses) is not list
        or len(hypotheses) != len(FACTORIAL_HYPOTHESES)
    ):
        _fail("successor development promotion cell cardinality drifted")
    observed_gates: list[bool] = []
    for index, cell_value in enumerate(cells):
        cell = _exact_keys(
            cell_value, PROMOTION_CELL_KEYS, label=f"promotion.cells[{index}]"
        )
        hypothesis = hypotheses[index]
        if type(hypothesis) is not dict:
            _fail("successor factorial inference hypothesis is not an object")
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
            if cell.get(field) != expected or type(cell.get(field)) is not type(expected):
                _fail(f"successor development promotion cell drifted: {field}")
        cell_gate = cell.get("gate_passed")
        if type(cell_gate) is not bool:
            _fail("successor development promotion cell gate must be exact boolean")
        for field in ("observed_mean", "standard_error", "median"):
            value = cell.get(field)
            if type(value) is not float or not math.isfinite(value):
                _fail(f"successor development promotion {field} must be a finite float")
        if cell["standard_error"] < 0.0:
            _fail("successor development promotion standard error is negative")
        counts = []
        for field in (
            "wins_above_threshold",
            "ties_at_threshold",
            "losses_below_threshold",
        ):
            counts.append(_integer(cell.get(field), label=f"promotion.{field}"))
        if sum(counts) != 6:
            _fail("successor development promotion W/T/L cardinality drifted")
        observed_gates.append(cell_gate)
    zero = raw.get("zero_standard_error_hypotheses")
    reasons = raw.get("gate_reasons")
    if (
        type(zero) is not list
        or any(type(item) is not str or item not in FACTORIAL_HYPOTHESES for item in zero)
        or len(set(zero)) != len(zero)
        or type(reasons) is not list
        or any(type(item) is not str or not item for item in reasons)
    ):
        _fail("successor development promotion HOLD reason payload drifted")
    expected_zero = [
        str(cell["hypothesis_id"])
        for cell in cells
        if cell["standard_error"] == 0.0
    ]
    if zero != expected_zero:
        _fail("successor development promotion zero-SE witness drifted")
    if zero:
        if (
            raw.get("status")
            != "HOLD_SUCCESSOR_DEVELOPMENT_PROMOTION_ZERO_STANDARD_ERROR"
            or raw.get("critical_value") is not None
            or raw.get("bootstrap_maxima_sha256") is not None
            or any(cell["simultaneous_lower_bound"] is not None for cell in cells)
            or any(observed_gates)
            or reasons != [f"zero_standard_error:{item}" for item in zero]
        ):
            _fail("successor development promotion zero-SE HOLD payload drifted")
    else:
        critical = raw.get("critical_value")
        if type(critical) is not float or not math.isfinite(critical) or critical < 0.0:
            _fail("successor development promotion critical value is invalid")
        _sha256(
            raw.get("bootstrap_maxima_sha256"),
            label="promotion.bootstrap_maxima_sha256",
        )
        failed: list[str] = []
        for index, cell in enumerate(cells):
            lower = cell["simultaneous_lower_bound"]
            if type(lower) is not float or not math.isfinite(lower):
                _fail("successor development promotion lower bound is invalid")
            expected_lower = cell["observed_mean"] - critical * cell["standard_error"]
            if lower != expected_lower:
                _fail("successor development promotion lower bound drifted")
            expected_gate = lower > cell["threshold"]
            if observed_gates[index] is not expected_gate:
                _fail("successor development promotion threshold decision drifted")
            if not expected_gate:
                failed.append(str(cell["hypothesis_id"]))
        expected_status = (
            PROMOTION_PASS_STATUS
            if not failed
            else "HOLD_SUCCESSOR_DEVELOPMENT_PROMOTION_THRESHOLDS_NOT_MET"
        )
        expected_reasons = [
            f"simultaneous_lower_bound_not_above_threshold:{item}" for item in failed
        ]
        if raw.get("status") != expected_status or reasons != expected_reasons:
            _fail("successor development promotion decision payload drifted")
    if promotion_gate is not all(observed_gates):
        _fail("successor development promotion aggregate gate drifted")
    return [
        inference_record,
        plan_record,
        matrix_receipt_record,
        aggregate_record,
        source_binding_record,
        manifest_binding_record,
    ]


def _derive_identity(
    *,
    root: Path,
    records: Mapping[str, FileRecord],
    receipts: Mapping[str, Mapping[str, object]],
    study_id: str,
    candidate_id: str,
) -> tuple[dict[str, str], list[FileRecord]]:
    nested: list[FileRecord] = []
    historical = receipts["historical_preservation"]
    _schema(
        historical,
        "v21e3r1_v4_v6_historical_preservation_receipt_v1",
        label="historical preservation receipt",
    )

    diagnostic = receipts["exact_504_diagnostic"]
    _schema(
        diagnostic,
        "v21e3r1_exposed_development_diagnostic_receipt_v2",
        label="exact-504 diagnostic receipt",
    )
    if (
        diagnostic.get("matrix_mode") != "FULL_504"
        or _integer(diagnostic.get("completed_rows"), label="diagnostic.completed_rows")
        != 504
        or _integer(diagnostic.get("expected_rows"), label="diagnostic.expected_rows")
        != 504
    ):
        _fail("exact-504 diagnostic cardinality drifted")
    development_source = _sha256(
        diagnostic.get("source_snapshot_sha256"),
        label="diagnostic.source_snapshot_sha256",
    )

    corrected = receipts["corrected_reanalysis"]
    _schema(
        corrected,
        "v21e3r1_corrected_operator_accounting_reanalysis_receipt_v1",
        label="corrected reanalysis receipt",
    )
    _agree(
        corrected.get("development_source_sha256"),
        development_source,
        label="corrected_reanalysis.development_source_sha256",
    )
    _agree(
        corrected.get("diagnostic_receipt_sha256"),
        records["exact_504_diagnostic"].sha256,
        label="corrected_reanalysis.diagnostic_receipt_sha256",
    )
    operator_spec = _read_embedded_root_path(
        root=root,
        value=corrected.get("metric_spec_path"),
        label="corrected reanalysis operator spec",
    )
    nested.append(operator_spec)
    operator_reanalysis_spec = _agree(
        corrected.get("metric_spec_sha256"),
        operator_spec.sha256,
        label="corrected_reanalysis.metric_spec_sha256",
    )
    _parse_canonical_json(operator_spec)

    source = receipts["successor_source_freeze"]
    _schema(
        source,
        "v21e3r1_successor_source_freeze_receipt_v2",
        label="successor source-freeze receipt",
    )
    _verify_payload_digest(source, label="successor source-freeze receipt")
    _agree(source.get("study_id"), study_id, label="source_freeze.study_id", digest=False)
    _agree(
        source.get("candidate_id"),
        candidate_id,
        label="source_freeze.candidate_id",
        digest=False,
    )
    _agree(
        source.get("parent_development_source_sha256"),
        development_source,
        label="source_freeze.parent_development_source_sha256",
    )
    successor_source = _sha256(
        source.get("source_snapshot_sha256"),
        label="source_freeze.source_snapshot_sha256",
    )
    successor_config = _sha256(
        source.get("semantic_config_sha256"),
        label="source_freeze.semantic_config_sha256",
    )
    study_metric = _sha256(
        source.get("study_metric_spec_sha256"),
        label="source_freeze.study_metric_spec_sha256",
    )
    simultaneous_sha = _sha256(
        source.get("simultaneous_inference_spec_sha256"),
        label="source_freeze.simultaneous_inference_spec_sha256",
    )

    source_parent = records["successor_source_freeze"].path.parent
    fixed_records = {
        "manifest": _read_fixed_sibling(
            root=root,
            parent=source_parent,
            name="source.manifest.json",
            label="successor source manifest",
        ),
        "semantic": _read_fixed_sibling(
            root=root,
            parent=source_parent,
            name="semantic.config.json",
            label="successor semantic config",
        ),
        "metric": _read_fixed_sibling(
            root=root,
            parent=source_parent,
            name="study.metric-spec.json",
            label="successor study metric spec",
        ),
        "simultaneous": _read_fixed_sibling(
            root=root,
            parent=source_parent,
            name="simultaneous-inference.spec.json",
            label="successor simultaneous-inference spec",
        ),
    }
    nested.extend(fixed_records.values())
    _agree(
        source.get("source_manifest_sha256"),
        fixed_records["manifest"].sha256,
        label="source_freeze.source_manifest_sha256",
    )
    _agree(
        source.get("semantic_config_sha256"),
        fixed_records["semantic"].sha256,
        label="source_freeze.semantic_config_sha256",
    )
    _agree(
        source.get("study_metric_spec_sha256"),
        fixed_records["metric"].sha256,
        label="source_freeze.study_metric_spec_sha256",
    )
    _agree(
        source.get("simultaneous_inference_spec_sha256"),
        fixed_records["simultaneous"].sha256,
        label="source_freeze.simultaneous_inference_spec_sha256",
    )
    manifest = _parse_canonical_json(fixed_records["manifest"])
    _validate_source_manifest_value(
        manifest,
        expected_root=successor_source,
        label="successor source manifest",
    )
    semantic = _parse_canonical_json(fixed_records["semantic"])
    _schema(
        semantic,
        "v21e3r1_successor_semantic_config_v1",
        label="successor semantic config",
    )
    _agree(semantic.get("study_id"), study_id, label="semantic.study_id", digest=False)
    _agree(
        semantic.get("candidate_id"),
        candidate_id,
        label="semantic.candidate_id",
        digest=False,
    )
    metric = _parse_canonical_json(fixed_records["metric"])
    _schema(metric, "v21e3r1_study_metric_spec_v1", label="study metric spec")
    _verify_payload_digest(metric, label="study metric spec")

    simultaneous = receipts["simultaneous_inference_spec"]
    _schema(
        simultaneous,
        "v21e3r1_simultaneous_inference_spec_v2",
        label="simultaneous-inference spec",
    )
    _verify_payload_digest(simultaneous, label="simultaneous-inference spec")
    _agree(
        records["simultaneous_inference_spec"].sha256,
        simultaneous_sha,
        label="simultaneous-inference binding SHA-256",
    )
    _agree(
        simultaneous.get("study_id"),
        study_id,
        label="simultaneous.study_id",
        digest=False,
    )
    _agree(
        simultaneous.get("candidate_id"),
        candidate_id,
        label="simultaneous.candidate_id",
        digest=False,
    )
    _agree(
        simultaneous.get("successor_source_sha256"),
        successor_source,
        label="simultaneous.successor_source_sha256",
    )
    _agree(
        simultaneous.get("successor_config_sha256"),
        successor_config,
        label="simultaneous.successor_config_sha256",
    )
    _agree(
        simultaneous.get("study_metric_spec_sha256"),
        study_metric,
        label="simultaneous.study_metric_spec_sha256",
    )
    if _integer(
        simultaneous.get("bootstrap_samples"), label="simultaneous.bootstrap_samples"
    ) != 10000:
        _fail("simultaneous.bootstrap_samples must equal the frozen value 10000")
    if _integer(
        simultaneous.get("bootstrap_seed"), label="simultaneous.bootstrap_seed"
    ) != 20260823:
        _fail("simultaneous.bootstrap_seed must equal the frozen value 20260823")

    promotion_identity = {
        "study_id": study_id,
        "candidate_id": candidate_id,
        "development_source_sha256": development_source,
        "successor_source_sha256": successor_source,
        "successor_config_sha256": successor_config,
        "operator_reanalysis_spec_sha256": operator_reanalysis_spec,
        "study_metric_spec_sha256": study_metric,
        "simultaneous_inference_spec_sha256": simultaneous_sha,
    }
    nested.extend(
        _validate_development_promotion_identity(
            root=root,
            record=records["successor_development_promotion"],
            receipt=receipts["successor_development_promotion"],
            identity=promotion_identity,
            source_record=records["successor_source_freeze"],
            source_manifest_record=fixed_records["manifest"],
        )
    )

    same_impl = receipts["same_implementation_coverage"]
    _validate_recovery_bound_same_implementation(
        receipt=same_impl,
        identity=promotion_identity,
        diagnostic_receipt=diagnostic,
        diagnostic_receipt_sha256=records["exact_504_diagnostic"].sha256,
        receipt_path=records["same_implementation_coverage"].path,
        evidence_root=root,
    )

    baseline = receipts["baseline_registry"]
    baseline_schema = baseline.get("schema")
    if baseline_schema == "v21e3r1_external_family_native_strong_baseline_registry_receipt_v1":
        if "study_metric_spec_sha256" in baseline:
            _fail("baseline registry contains hybrid metric identity aliases")
        baseline_metric_field = "metric_spec_sha256"
    elif baseline_schema == "v21e3r1_external_family_native_strong_baseline_registry_receipt_v2":
        if "metric_spec_sha256" in baseline:
            _fail("baseline registry contains hybrid metric identity aliases")
        baseline_metric_field = "study_metric_spec_sha256"
        _verify_payload_digest(baseline, label="baseline registry receipt")
        if baseline.get("scope") != (
            "DEVELOPMENT_REFERENCE_FREEZE_ONLY_NOT_STRONG_EXTERNAL_BASELINE_EVIDENCE"
        ):
            _fail("baseline v2 must remain the frozen design-only registry")
        for field in (
            "selection_authorized",
            "confirmation_authorized",
            "formal_study_authorized",
            "scientific_claim_authorized",
        ):
            value = baseline.get(field)
            if type(value) is not bool:
                _fail(f"baseline.{field} must be an exact boolean")
            if value:
                _fail(f"baseline v2 expands authority at {field}")
        if baseline.get("ijoc_submission_status") != "IJOC_HOLD":
            _fail("baseline v2 publication status drifted")
    else:
        _fail("baseline registry receipt schema drifted")
    _agree(baseline.get("study_id"), study_id, label="baseline.study_id", digest=False)
    _agree(
        baseline.get("candidate_id"),
        candidate_id,
        label="baseline.candidate_id",
        digest=False,
    )
    _agree(
        baseline.get(baseline_metric_field),
        study_metric,
        label=f"baseline.{baseline_metric_field}",
    )

    external = receipts["external_algorithm_replay"]
    external_schema = external.get("schema")
    if external_schema == "v21e3r1_external_algorithm_replay_receipt_v1":
        if "study_metric_spec_sha256" in external:
            _fail("external replay contains hybrid metric identity aliases")
        external_metric_field = "metric_spec_sha256"
    elif external_schema == "v21e3r1_external_algorithm_replay_receipt_v2":
        if "metric_spec_sha256" in external:
            _fail("external replay contains hybrid metric identity aliases")
        external_metric_field = "study_metric_spec_sha256"
        _verify_payload_digest(external, label="external replay receipt")
    else:
        _fail("external replay receipt schema drifted")
    _agree(external.get("study_id"), study_id, label="external.study_id", digest=False)
    _agree(
        external.get("candidate_id"),
        candidate_id,
        label="external.candidate_id",
        digest=False,
    )
    _agree(
        external.get("successor_source_sha256"),
        successor_source,
        label="external.successor_source_sha256",
    )
    _agree(
        external.get("successor_config_sha256"),
        successor_config,
        label="external.successor_config_sha256",
    )
    _agree(
        external.get(external_metric_field),
        study_metric,
        label=f"external.{external_metric_field}",
    )

    identity = {
        "study_id": study_id,
        "candidate_id": candidate_id,
        "development_source_sha256": development_source,
        "successor_source_sha256": successor_source,
        "successor_config_sha256": successor_config,
        "operator_reanalysis_spec_sha256": operator_reanalysis_spec,
        "study_metric_spec_sha256": study_metric,
        "simultaneous_inference_spec_sha256": simultaneous_sha,
    }
    if set(identity) != IDENTITY_KEYS:
        _fail("internal identity construction drifted")
    return identity, nested


def _validate_phase_identity(
    *,
    receipt: Mapping[str, object],
    phase: str,
    identity: Mapping[str, str],
    source_freeze_sha256: str,
    selection_result_sha256: str | None = None,
    external_replay_sha256: str | None = None,
    custody_receipt_sha256: str | None = None,
) -> None:
    label = f"{phase}_result"
    _schema(
        receipt,
        "v21e3r1_independent_simultaneous_inference_receipt_v1",
        label=f"{phase} result receipt",
    )
    _verify_payload_digest(receipt, label=f"{phase} result receipt")
    if receipt.get("phase") != phase:
        _fail(f"{label}.phase disagrees with its binding role")
    _agree(
        receipt.get("study_id"),
        identity["study_id"],
        label=f"{label}.study_id",
        digest=False,
    )
    _agree(
        receipt.get("selected_candidate"),
        identity["candidate_id"],
        label=f"{label}.selected_candidate",
        digest=False,
    )
    _agree(
        receipt.get("study_freeze_sha256"),
        source_freeze_sha256,
        label=f"{label}.study_freeze_sha256",
    )
    _agree(
        receipt.get("source_root_sha256"),
        identity["successor_source_sha256"],
        label=f"{label}.source_root_sha256",
    )
    _agree(
        receipt.get("metric_spec_sha256"),
        identity["study_metric_spec_sha256"],
        label=f"{label}.metric_spec_sha256",
    )
    _agree(
        receipt.get("decision_spec_sha256"),
        identity["simultaneous_inference_spec_sha256"],
        label=f"{label}.decision_spec_sha256",
    )
    if phase == "selection":
        if (
            receipt.get("selection_binding") is not None
            or receipt.get("confirmation_control_bindings") is not None
        ):
            _fail("selection_result contains later-phase bindings")
        return
    if phase != "confirmation":
        _fail("unsupported phase result role")
    if (
        selection_result_sha256 is None
        or external_replay_sha256 is None
        or custody_receipt_sha256 is None
    ):
        _fail("confirmation result validation omitted prerequisite hashes")
    selection_binding = _mapping(
        receipt.get("selection_binding"), label="confirmation_result.selection_binding"
    )
    _agree(
        selection_binding.get("selection_receipt_sha256"),
        selection_result_sha256,
        label="confirmation_result.selection_receipt_sha256",
    )
    _agree(
        selection_binding.get("selected_candidate"),
        identity["candidate_id"],
        label="confirmation_result.selection_binding.selected_candidate",
        digest=False,
    )
    controls = _mapping(
        receipt.get("confirmation_control_bindings"),
        label="confirmation_result.confirmation_control_bindings",
    )
    _agree(
        controls.get("external_producer_receipt_sha256"),
        external_replay_sha256,
        label="confirmation_result.external_producer_receipt_sha256",
    )
    _agree(
        controls.get("custody_receipt_sha256"),
        custody_receipt_sha256,
        label="confirmation_result.custody_receipt_sha256",
    )
    _agree(
        controls.get("statistics_source_sha256"),
        _sha256(receipt.get("source_sha256"), label="confirmation_result.source_sha256"),
        label="confirmation_result.statistics_source_sha256",
    )


def _contained_output(root: Path, value: str | Path) -> Path:
    raw = Path(value)
    _reject_lexical_traversal(raw, label="output path")
    absolute = Path(os.path.abspath(raw if raw.is_absolute() else root / raw))
    try:
        relative = absolute.relative_to(root)
    except ValueError:
        _fail("output path escapes evidence root")
    if not relative.parts:
        _fail("output path must name a file")
    cursor = root
    for part in relative.parts[:-1]:
        cursor = cursor / part
        if _lstat_is_link_or_reparse(cursor):
            _fail("output path traverses a symbolic link, junction, or reparse point")
    try:
        parent = absolute.parent.resolve(strict=True)
    except OSError as error:
        _fail(f"output parent does not exist: {error}")
    try:
        parent.relative_to(root)
    except ValueError:
        _fail("output path escapes evidence root")
    if not parent.is_dir() or _lstat_is_link_or_reparse(parent):
        _fail("output parent must be a non-link directory")
    if absolute.exists() or _lstat_is_link_or_reparse(absolute):
        _fail("output already exists; exclusive create required")
    return parent / absolute.name


def _write_exclusive(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def build_gate_spec(
    *,
    evidence_root: str | Path,
    output: str | Path,
    requested_authorization: str,
    study_id: str,
    candidate_id: str,
    common_binding_paths: Mapping[str, str | Path],
    selection_result_path: str | Path | None = None,
    confirmation_result_path: str | Path | None = None,
) -> dict[str, object]:
    root = _evidence_root(evidence_root)
    requested = _string(
        requested_authorization, label="requested_authorization"
    )
    if requested not in REQUESTS:
        _fail("requested_authorization is unsupported")
    study = _identifier(study_id, label="study_id")
    candidate = _identifier(candidate_id, label="candidate_id")
    if set(common_binding_paths) != set(COMMON_BINDINGS):
        _fail("common binding path key set drifted")
    if requested == "selection" and (
        selection_result_path is not None or confirmation_result_path is not None
    ):
        _fail("selection request must not bind later-phase results")
    if requested == "confirmation" and (
        selection_result_path is None or confirmation_result_path is not None
    ):
        _fail("confirmation request requires only --selection-result")
    if requested == "formal_input_materialization" and (
        selection_result_path is None or confirmation_result_path is None
    ):
        _fail("formal request requires --selection-result and --confirmation-result")

    records: dict[str, FileRecord] = {}
    receipts: dict[str, dict[str, Any]] = {}
    observed_paths: set[Path] = set()
    for binding_id in sorted(COMMON_BINDINGS):
        record = _stable_read(
            root,
            Path(common_binding_paths[binding_id]),
            label=f"binding {binding_id}",
        )
        if record.path in observed_paths:
            _fail("two evidence bindings resolve to the same file")
        observed_paths.add(record.path)
        records[binding_id] = record
        receipts[binding_id] = _parse_canonical_json(
            record,
            allow_compact_newline=binding_id
            == "successor_development_promotion",
        )

    identity, nested = _derive_identity(
        root=root,
        records=records,
        receipts=receipts,
        study_id=study,
        candidate_id=candidate,
    )

    bindings: dict[str, dict[str, str]] = {
        binding_id: {
            "path": records[binding_id].relative_path,
            "sha256": records[binding_id].sha256,
        }
        for binding_id in COMMON_BINDINGS
    }
    # Phase-specific bindings are conditional, so selection remains exactly
    # the nine common evidence bindings accepted by the evaluator.
    if selection_result_path is not None:
        selection_record = _stable_read(
            root, Path(selection_result_path), label="binding selection_result"
        )
        if selection_record.path in observed_paths:
            _fail("two evidence bindings resolve to the same file")
        observed_paths.add(selection_record.path)
        records["selection_result"] = selection_record
        receipts["selection_result"] = _parse_canonical_json(selection_record)
        _validate_phase_identity(
            receipt=receipts["selection_result"],
            phase="selection",
            identity=identity,
            source_freeze_sha256=records["successor_source_freeze"].sha256,
        )
        bindings["selection_result"] = {
            "path": selection_record.relative_path,
            "sha256": selection_record.sha256,
        }
    if confirmation_result_path is not None:
        confirmation_record = _stable_read(
            root, Path(confirmation_result_path), label="binding confirmation_result"
        )
        if confirmation_record.path in observed_paths:
            _fail("two evidence bindings resolve to the same file")
        observed_paths.add(confirmation_record.path)
        records["confirmation_result"] = confirmation_record
        receipts["confirmation_result"] = _parse_canonical_json(confirmation_record)
        external_receipt = receipts["external_algorithm_replay"]
        custody_sha256 = _sha256(
            external_receipt.get("custody_receipt_sha256"),
            label="external replay custody_receipt_sha256",
        )
        _validate_phase_identity(
            receipt=receipts["confirmation_result"],
            phase="confirmation",
            identity=identity,
            source_freeze_sha256=records["successor_source_freeze"].sha256,
            selection_result_sha256=records["selection_result"].sha256,
            external_replay_sha256=records["external_algorithm_replay"].sha256,
            custody_receipt_sha256=custody_sha256,
        )
        bindings["confirmation_result"] = {
            "path": confirmation_record.relative_path,
            "sha256": confirmation_record.sha256,
        }

    spec: dict[str, object] = {
        "schema": SPEC_SCHEMA,
        "requested_authorization": requested,
        "identity": identity,
        "bindings": bindings,
    }
    raw = _canonical_bytes(spec)
    _revalidate_records(root, [*records.values(), *nested])
    output_path = _contained_output(root, output)
    _write_exclusive(output_path, raw)
    return {
        "schema": "v21e3r1_prospective_gate_spec_build_receipt_v1",
        "status": "PASS_GATE_SPEC_BUILT_ENGINEERING_ONLY",
        "gate_spec_path": _canonical_relative(
            output_path, root=root, label="gate specification output"
        ),
        "gate_spec_sha256": _sha256_bytes(raw),
        "requested_authorization": requested,
        "binding_count": len(bindings),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--requested-authorization", required=True, choices=sorted(REQUESTS))
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    for binding_id in COMMON_BINDINGS:
        parser.add_argument("--" + binding_id.replace("_", "-"), required=True)
    parser.add_argument("--selection-result")
    parser.add_argument("--confirmation-result")
    arguments = parser.parse_args(argv)
    binding_paths = {
        binding_id: getattr(arguments, binding_id) for binding_id in COMMON_BINDINGS
    }
    try:
        receipt = build_gate_spec(
            evidence_root=arguments.evidence_root,
            output=arguments.output,
            requested_authorization=arguments.requested_authorization,
            study_id=arguments.study_id,
            candidate_id=arguments.candidate_id,
            common_binding_paths=binding_paths,
            selection_result_path=arguments.selection_result,
            confirmation_result_path=arguments.confirmation_result,
        )
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        return 0
    except (GateSpecBuildError, OSError) as error:
        print(
            json.dumps(
                {
                    "schema": "v21e3r1_prospective_gate_spec_build_error_v1",
                    "status": "HOLD_INTEGRITY_ERROR",
                    "error": str(error),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
