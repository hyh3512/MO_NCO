from __future__ import annotations

"""Freeze the V21e3r1 successor source/config identity without study execution.

The command inventories executable project source, copies the already-frozen
semantic/configuration specifications into a new exclusive custody directory,
and writes an engineering-only receipt.  It never materializes cases, runs an
algorithm, or authorizes scientific/publication claims.
"""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import sys
from typing import Any, Mapping, NamedTuple, NoReturn, Sequence
import zipfile


MANIFEST_SCHEMA = "v21e3r1_branch_replay_source_manifest_binding_v1"
SEMANTIC_CONFIG_SCHEMA = "v21e3r1_successor_semantic_config_v1"
EXPECTED_SEMANTIC_PARAMETERS = {
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
    "promoted_arm_by_family": {
        "MOKP": "MOKP_BOTH",
        "MOTSP": "MOTSP_ANCHOR",
    },
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
SIMULTANEOUS_SCHEMA = "v21e3r1_simultaneous_inference_spec_v2"
RECEIPT_SCHEMA = "v21e3r1_successor_source_freeze_receipt_v2"
PASS_STATUS = "PASS_SUCCESSOR_SOURCE_AND_CONFIG_FREEZE_ENGINEERING_ONLY"
DEVELOPMENT_PLAN_SCHEMA = "v21e3r1_exposed_development_diagnostic_plan_v2"
DEVELOPMENT_MANIFEST_SCHEMA = "v21e3r1_diagnostic_source_manifest_v1"
DEVELOPMENT_RECEIPT_SCHEMA = "v21e3r1_development_source_snapshot_receipt_v1"
DEVELOPMENT_PASS_STATUS = (
    "PASS_DEVELOPMENT_EXECUTION_SOURCE_SNAPSHOT_ENGINEERING_ONLY"
)
FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
SOURCE_ROOTS = (
    PurePosixPath("mo_nco"),
    PurePosixPath("ijoc_submission_v21e3r1/scripts"),
    PurePosixPath("independent_reproduction"),
)
SUCCESSOR_INDEPENDENT_METRIC_RELATIVE = PurePosixPath(
    "independent_reproduction/recompute_v21e3r1_successor_metrics.py"
)
RELATED_SOURCE_FILES = (
    PurePosixPath(
        "ijoc_submission_v21e3r1/development/"
        "V21E3R1_SUCCESSOR_DEVELOPMENT_FACTORIAL_INFERENCE_V1.json"
    ),
    PurePosixPath(
        "ijoc_submission_v21e3r1/development/"
        "V21E3R1_SUCCESSOR_DEVELOPMENT_FACTORIAL_RUNBOOK_V1.md"
    ),
    PurePosixPath(
        "ijoc_submission_v21e3r1/development/"
        "V21E3R1_V8_STRICT_EXECUTION_RUNBOOK_2026-08-23.md"
    ),
    PurePosixPath("tests/test_v21e3r1_independent_simultaneous_inference.py"),
    PurePosixPath("tests/test_v21e3r1_successor_metric.py"),
)
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".cache",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "cache",
        "dist",
        "output",
        "outputs",
        "temp",
        "tmp",
        "venv",
    }
)
FIXED_OUTPUT_NAMES = {
    "source_manifest": "source.manifest.json",
    "semantic_config": "semantic.config.json",
    "study_metric_spec": "study.metric-spec.json",
    "simultaneous_spec": "simultaneous-inference.spec.json",
    "receipt": "successor-source.freeze.receipt.json",
    "archive": "successor-source.zip",
}
DEVELOPMENT_OUTPUT_NAMES = {
    "plan": "development-diagnostic.plan.json",
    "source_manifest": "development-source.manifest.json",
    "archive": "development-source.zip",
    "receipt": "development-source.snapshot.receipt.json",
}
DEVELOPMENT_RECEIPT_KEYS = {
    "schema",
    "status",
    "scientific_scope",
    "diagnostic_plan_schema",
    "diagnostic_plan_copy_path",
    "diagnostic_plan_sha256",
    "source_manifest_schema",
    "source_manifest_path",
    "source_manifest_sha256",
    "source_snapshot_sha256",
    "source_entry_count",
    "source_total_bytes",
    "all_source_files_verified",
    "source_archive_materialized",
    "source_archive_path",
    "source_archive_sha256",
    "source_archive_scope",
    "development_execution_replayed",
    "original_development_diagnostic_artifacts_modified",
    "selection_cases_materialized",
    "confirmation_cases_materialized",
    "formal_cases_materialized",
    "implementation_independence",
    "scientific_independence",
    "selection_authorized",
    "confirmation_authorized",
    "formal_study_authorized",
    "scientific_claim_authorized",
    "public_redistribution_authorized",
    "ijoc_submission_status",
    "receipt_payload_sha256",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SIMULTANEOUS_KEYS = {
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
}
STUDY_METRIC_KEYS = {
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
}
DEVELOPMENT_PLAN_KEYS = {
    "schema",
    "status",
    "scientific_scope",
    "arms",
    "case_ids",
    "seeds",
    "charged_evaluation_budget",
    "checkpoint_period",
    "row_timeout_seconds",
    "expected_rows",
    "input_binding",
    "source_manifest",
    "selection_entropy_release",
    "confirmation_materialization",
    "formal_materialization",
}
DEVELOPMENT_ARMS = [
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
]
DEVELOPMENT_CASE_IDS = [
    f"v21e3-{family}-development-n{size}-s{index:02d}"
    for family in ("mokp", "motsp")
    for size in (100, 200, 500)
    for index in (0, 1)
]
DEVELOPMENT_SEEDS = [31051, 31057, 31059]
DEVELOPMENT_INPUT_MANIFEST_PATHS = {
    "ijoc_submission_v21e3/development_manifests_v1/"
    "config_manifest_development.json",
    "ijoc_submission_v21e3/development_manifests_v1/"
    "reference_manifest_development.json",
    "ijoc_submission_v21e3/development_partitions_v1/case_manifest.json",
}


class FreezeError(ValueError):
    """A source/config custody or integrity contract failed."""


class FileRecord(NamedTuple):
    path: Path
    label: str
    relative_path: str | None
    raw: bytes
    sha256: str
    identity: tuple[int, int, int, int, int, int]


def _fail(message: str) -> NoReturn:
    raise FreezeError(message)


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


def _parse_json(
    record: FileRecord, *, label: str, require_canonical: bool
) -> dict[str, Any]:
    try:
        text = record.raw.decode("utf-8")
    except UnicodeDecodeError as error:
        _fail(f"{label} is not strict UTF-8 JSON: {error}")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except FreezeError:
        raise
    except (json.JSONDecodeError, RecursionError) as error:
        _fail(f"{label} is malformed JSON: {error}")
    if type(value) is not dict:
        _fail(f"{label} must be a JSON object")
    _validate_json_tree(value, label=label)
    if require_canonical and record.raw != _canonical_bytes(value):
        _fail(f"{label} is not canonical JSON")
    return value


def _parse_canonical_json(record: FileRecord, *, label: str) -> dict[str, Any]:
    return _parse_json(record, label=label, require_canonical=True)


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
    file_attributes = int(getattr(result, "st_file_attributes", 0))
    return stat.S_ISLNK(result.st_mode) or bool(reparse_flag and file_attributes & reparse_flag)


def _lstat_is_link_or_reparse(path: Path) -> bool:
    try:
        return _is_link_or_reparse(path.lstat())
    except FileNotFoundError:
        return False
    except OSError as error:
        _fail(f"path component cannot be inspected: {path}: {error}")


def _stable_read(path: Path, *, label: str, relative_path: str | None = None) -> FileRecord:
    try:
        before = path.lstat()
    except OSError as error:
        _fail(f"{label} cannot be inspected: {error}")
    if _is_link_or_reparse(before):
        _fail(f"{label} must not be a symbolic link, junction, or reparse point")
    if not stat.S_ISREG(before.st_mode):
        _fail(f"{label} is not a regular file")
    try:
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        _fail(f"{label} cannot be read: {error}")
    before_identity = _stat_identity(before)
    after_identity = _stat_identity(after)
    if before_identity != after_identity or len(raw) != after.st_size:
        _fail(f"{label} changed while it was being read")
    if not raw:
        _fail(f"{label} must not be empty")
    return FileRecord(
        path=path,
        label=label,
        relative_path=relative_path,
        raw=raw,
        sha256=_sha256_bytes(raw),
        identity=after_identity,
    )


def _revalidate_records(records: Sequence[FileRecord]) -> None:
    for frozen in records:
        observed = _stable_read(
            frozen.path,
            label=frozen.label,
            relative_path=frozen.relative_path,
        )
        if observed.identity != frozen.identity or observed.sha256 != frozen.sha256:
            _fail(f"{frozen.label} changed after collection")


def _canonical_relative_path(path: Path, *, root: Path, label: str) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        _fail(f"{label} escapes project root")
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
        or any(part in {"", ".", ".."} for part in posix.parts)
        or ":" in posix.parts[0]
    ):
        _fail(f"{label} is not a canonical relative POSIX path")
    return raw


def _manifest_relative_path(value: object, *, label: str) -> str:
    raw = _string(value, label=label)
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if (
        "\\" in raw
        or any(ord(character) < 32 for character in raw)
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in posix.parts)
        or ":" in posix.parts[0]
        or posix.as_posix() != raw
    ):
        _fail(f"{label} is not a canonical relative POSIX path")
    return raw


def _contained_input_file(root: Path, value: str | Path, *, label: str) -> Path:
    raw_path = Path(value)
    absolute = Path(os.path.abspath(raw_path if raw_path.is_absolute() else root / raw_path))
    try:
        relative = absolute.relative_to(root)
    except ValueError:
        _fail(f"{label} escapes project root")
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
        _fail(f"{label} escapes project root")
    if not resolved.is_file():
        _fail(f"{label} is not a regular file")
    return resolved


def _walk_python_sources(root: Path, relative_root: PurePosixPath) -> list[Path]:
    directory = root.joinpath(*relative_root.parts)
    if not directory.is_dir() or _lstat_is_link_or_reparse(directory):
        _fail(f"required source directory is absent or linked: {relative_root.as_posix()}")
    result: list[Path] = []

    def walk_error(error: OSError) -> NoReturn:
        _fail(f"source inventory traversal failed: {error}")

    for current_raw, directory_names, file_names in os.walk(
        directory, followlinks=False, onerror=walk_error
    ):
        current = Path(current_raw)
        kept_directories: list[str] = []
        for name in sorted(directory_names, key=lambda item: (item.casefold(), item)):
            child = current / name
            if name.casefold() in EXCLUDED_DIRECTORY_NAMES:
                continue
            if _lstat_is_link_or_reparse(child):
                _fail(f"source directory must not be linked: {child}")
            kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names, key=lambda item: (item.casefold(), item)):
            if Path(name).suffix.casefold() != ".py":
                continue
            candidate = current / name
            if _lstat_is_link_or_reparse(candidate):
                _fail(f"source file must not be linked: {candidate}")
            result.append(candidate)
    if not result:
        _fail(f"source scope contains no Python files: {relative_root.as_posix()}")
    return result


def _collect_source_records(root: Path) -> list[FileRecord]:
    candidates: list[Path] = []
    for relative_root in SOURCE_ROOTS:
        candidates.extend(_walk_python_sources(root, relative_root))
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file() or _lstat_is_link_or_reparse(pyproject):
        _fail("required source file is absent or linked: pyproject.toml")
    candidates.append(pyproject)
    for relative_file in RELATED_SOURCE_FILES:
        candidate = root.joinpath(*relative_file.parts)
        if not candidate.is_file() or _lstat_is_link_or_reparse(candidate):
            _fail(f"required related source file is absent or linked: {relative_file}")
        candidates.append(candidate)
    keyed: list[tuple[str, Path]] = []
    for path in candidates:
        relative = _canonical_relative_path(path, root=root, label="source path")
        keyed.append((relative, path))
    keyed.sort(key=lambda item: (item[0].casefold(), item[0]))
    paths = [relative for relative, _ in keyed]
    if len({path.casefold() for path in paths}) != len(paths):
        _fail("source inventory contains case-insensitive duplicate paths")
    return [
        _stable_read(path, label=f"source file {relative}", relative_path=relative)
        for relative, path in keyed
    ]


def _revalidate_source_inventory(root: Path, frozen: Sequence[FileRecord]) -> None:
    observed = _collect_source_records(root)
    frozen_identity = [
        (record.relative_path, record.identity, record.sha256) for record in frozen
    ]
    observed_identity = [
        (record.relative_path, record.identity, record.sha256) for record in observed
    ]
    if observed_identity != frozen_identity:
        _fail("executable source inventory changed after collection")


def _validate_semantic_config(
    value: Mapping[str, object], *, study_id: str, candidate_id: str
) -> None:
    raw = _exact_keys(
        value,
        {"schema", "study_id", "candidate_id", "parameters"},
        label="semantic config",
    )
    parameters = raw["parameters"]
    if type(parameters) is not dict or not parameters:
        _fail("semantic config parameters must be a nonempty exact object")
    if (
        raw["schema"] != SEMANTIC_CONFIG_SCHEMA
        or raw["study_id"] != study_id
        or raw["candidate_id"] != candidate_id
    ):
        _fail("semantic config identity disagrees with the freeze request")
    if parameters != EXPECTED_SEMANTIC_PARAMETERS:
        _fail("semantic config policy contract drifted")


def _validate_study_metric_spec(
    value: Mapping[str, object], *, source_records: Sequence[FileRecord]
) -> None:
    schema = value.get("schema")
    normalized_schema = (
        re.sub(r"[-. ]+", "_", schema.casefold()) if type(schema) is str else ""
    )
    if "operator_accounting" in normalized_schema:
        _fail("study metric spec is an operator-accounting spec")
    raw = _exact_keys(value, STUDY_METRIC_KEYS, label="study metric spec")
    fixed_strings = {
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
    if any(raw[field] != expected for field, expected in fixed_strings.items()):
        _fail("study metric identity or estimand contract drifted")
    if _integer(raw["objective_dimension"], label="study metric objective_dimension") != 2:
        _fail("study metric objective dimension must be exactly two")
    reference = raw["reference_point"]
    if (
        type(reference) is not list
        or len(reference) != 2
        or any(type(item) is not float for item in reference)
        or reference != [1.0, 1.0]
    ):
        _fail("study metric reference point must be exact float vector [1.0,1.0]")
    secondary = raw["secondary_reporting_metrics"]
    if type(secondary) is not list or secondary != [
        "terminal_hypervolume",
        "attempt_count",
        "physical_start_count",
        "charged_evaluation_count",
        "wall_time_seconds",
        "peak_rss_bytes",
    ] or any(type(item) is not str for item in secondary):
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
    source_by_path = {
        record.relative_path: (len(record.raw), record.sha256) for record in source_records
    }
    for field, expected_path in (
        ("production_metric_source", "mo_nco/pareto_ijoc_analysis.py"),
        (
            "independent_metric_source",
            SUCCESSOR_INDEPENDENT_METRIC_RELATIVE.as_posix(),
        ),
    ):
        binding = _exact_keys(
            raw[field], {"path", "bytes", "sha256"}, label=f"study metric {field}"
        )
        if _manifest_relative_path(binding["path"], label=f"study metric {field}.path") != expected_path:
            _fail(f"study metric {field} path drifted")
        byte_count = _integer(binding["bytes"], label=f"study metric {field}.bytes", minimum=1)
        digest = _sha256(binding["sha256"], label=f"study metric {field}.sha256")
        if source_by_path.get(expected_path) != (byte_count, digest):
            _fail(f"study metric {field} disagrees with successor source manifest")
    if not _boolean(
        raw["practical_thresholds_bound_in_simultaneous_spec"],
        label="practical_thresholds_bound_in_simultaneous_spec",
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
    core = dict(raw)
    payload_sha = _sha256(
        core.pop("receipt_payload_sha256"), label="study metric receipt_payload_sha256"
    )
    if payload_sha != _sha256_bytes(_canonical_bytes(core)):
        _fail("study metric receipt payload hash disagrees with its canonical core")


def _validate_simultaneous_spec(
    value: Mapping[str, object],
    *,
    study_id: str,
    candidate_id: str,
    source_root_sha256: str,
    semantic_config_sha256: str,
    study_metric_spec_sha256: str,
    source_records: Sequence[FileRecord],
) -> None:
    raw = _exact_keys(value, SIMULTANEOUS_KEYS, label="simultaneous-inference spec")
    if (
        raw["schema"] != SIMULTANEOUS_SCHEMA
        or raw["status"] != "PASS_FROZEN_BEFORE_SELECTION_ENGINEERING_ONLY"
        or raw["scope"] != "FROZEN_PROSPECTIVE_DESIGN_ONLY_NO_CASE_MATERIALIZATION"
        or raw["study_id"] != study_id
        or raw["candidate_id"] != candidate_id
        or raw["successor_source_sha256"] != source_root_sha256
        or raw["successor_config_sha256"] != semantic_config_sha256
        or raw["study_metric_spec_sha256"] != study_metric_spec_sha256
    ):
        _fail("simultaneous-inference identity disagrees with the source/config freeze")
    families = raw["families"]
    if type(families) is not list or families != ["MOKP", "MOTSP"]:
        _fail("simultaneous-inference families must be exact frozen MOKP, MOTSP order")
    if any(type(item) is not str for item in families):
        _fail("simultaneous-inference family identifiers must be exact strings")
    candidates = raw["candidates"]
    if (
        type(candidates) is not list
        or candidates != ["C0", "C1", "C2", "C3"]
        or any(type(item) is not str for item in candidates)
    ):
        _fail("simultaneous-inference candidate order drifted")
    alpha = _float(raw["familywise_alpha"], label="simultaneous familywise_alpha")
    fixed_strings = {
        "evaluator_source_path": (
            "simultaneous-inference-design/recompute-simultaneous-bounds.py"
        ),
        "evaluator_test_path": (
            "simultaneous-inference-design/recompute-simultaneous-bounds.tests.py"
        ),
        "method": "one_sided_observed_se_max_t_paired_case_cluster_bootstrap_v1",
        "quantile_convention": "CEIL((1-ALPHA)*(B+1))_ORDER_STATISTIC",
        "rng_protocol": "SHA256_COUNTER_U64_REJECTION_V1",
        "rng_domain": "v21e3r1-simultaneous-case-bootstrap-v1",
        "cluster_unit": "PAIRED_CASE",
        "seed_aggregation": "MEAN_WITHIN_CASE_ARM",
        "resampling_rule": (
            "WITH_REPLACEMENT_INDEPENDENT_BY_FAMILY_"
            "SHARED_ACROSS_CELLS_WITHIN_FAMILY"
        ),
        "centering": "BOOTSTRAP_MEAN_MINUS_OBSERVED_MEAN",
        "studentization_denominator": "OBSERVED_CASE_CLUSTER_STANDARD_ERROR",
        "familywise_scope": "JOINT_ACROSS_BOTH_FAMILIES",
        "ijoc_submission_status": "IJOC_HOLD",
    }
    if any(raw[field] != expected for field, expected in fixed_strings.items()):
        _fail("simultaneous-inference evaluator or method contract drifted")
    _manifest_relative_path(
        raw["evaluator_source_path"], label="simultaneous evaluator_source_path"
    )
    _manifest_relative_path(
        raw["evaluator_test_path"], label="simultaneous evaluator_test_path"
    )
    evaluator_source_sha = _sha256(
        raw["evaluator_source_sha256"], label="simultaneous evaluator_source_sha256"
    )
    evaluator_test_sha = _sha256(
        raw["evaluator_test_sha256"], label="simultaneous evaluator_test_sha256"
    )
    source_by_path = {record.relative_path: record.sha256 for record in source_records}
    if (
        source_by_path.get(
            "independent_reproduction/recompute_v21e3r1_simultaneous_bounds.py"
        )
        != evaluator_source_sha
        or source_by_path.get(
            "tests/test_v21e3r1_independent_simultaneous_inference.py"
        )
        != evaluator_test_sha
    ):
        _fail("simultaneous evaluator source/test hashes disagree with source inventory")
    if (
        alpha != 0.05
        or _integer(raw["bootstrap_samples"], label="bootstrap_samples") != 10_000
        or _integer(raw["bootstrap_seed"], label="bootstrap_seed") != 20_260_823
        or _float(raw["critical_value_floor"], label="critical_value_floor") != 0.0
    ):
        _fail("simultaneous-inference alpha/bootstrap/RNG boundary drifted")
    thresholds = _exact_keys(
        raw["practical_thresholds"],
        {"primary_effect", "adjacent_mechanism_effect"},
        label="simultaneous practical thresholds",
    )
    if (
        _float(thresholds["primary_effect"], label="primary practical threshold") != 0.0
        or _float(
            thresholds["adjacent_mechanism_effect"],
            label="adjacent-mechanism practical threshold",
        )
        != 0.005
    ):
        _fail("simultaneous practical thresholds drifted")
    expected_selection = [
        {
            "hypothesis_id": f"{family}:{candidate}-{reference}",
            "family": family,
            "candidate": candidate,
            "reference": reference,
            "threshold_roles": roles,
        }
        for family in ("MOKP", "MOTSP")
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
        for family in ("MOKP", "MOTSP")
        for reference, role in (("C0", "primary"), ("PREDECESSOR", "adjacent"))
    ]
    if (
        type(raw["selection_cells"]) is not list
        or raw["selection_cells"] != expected_selection
        or _integer(raw["selection_cell_count"], label="selection_cell_count") != 10
        or type(raw["confirmation_cells"]) is not list
        or raw["confirmation_cells"] != expected_confirmation
        or _integer(raw["confirmation_cell_count"], label="confirmation_cell_count") != 4
    ):
        _fail("simultaneous selection/confirmation cells drifted")
    if not _boolean(
        raw["selection_and_confirmation_disjoint_by_construction"],
        label="selection_and_confirmation_disjoint_by_construction",
    ):
        _fail("selection and confirmation must be frozen disjoint")
    if not _boolean(raw["frozen_before_selection"], label="frozen_before_selection"):
        _fail("simultaneous spec was not frozen before selection")
    for field in (
        "selection_cases_materialized",
        "confirmation_cases_materialized",
        "formal_cases_materialized",
        "selection_authorized",
        "confirmation_authorized",
        "formal_study_authorized",
        "scientific_claim_authorized",
    ):
        if _boolean(raw[field], label=f"simultaneous {field}"):
            _fail(f"simultaneous boundary field {field} must remain false")
    core = dict(raw)
    payload_sha = _sha256(
        core.pop("receipt_payload_sha256"),
        label="simultaneous receipt_payload_sha256",
    )
    if payload_sha != _sha256_bytes(_canonical_bytes(core)):
        _fail("simultaneous receipt payload hash disagrees with its canonical core")


def _exact_string_list(value: object, *, label: str) -> list[str]:
    if type(value) is not list or any(type(item) is not str or not item for item in value):
        _fail(f"{label} must be an exact list of nonempty strings")
    return value


def _exact_integer_list(value: object, *, label: str) -> list[int]:
    if type(value) is not list or any(type(item) is not int for item in value):
        _fail(f"{label} must be an exact list of integers")
    return value


def _development_manifest_records(
    root: Path, manifest_value: object
) -> tuple[dict[str, Any], list[FileRecord], bytes]:
    manifest = _exact_keys(
        manifest_value,
        {"schema", "entry_count", "hash_rule", "source_snapshot_sha256", "entries"},
        label="development diagnostic source manifest",
    )
    if (
        manifest["schema"] != DEVELOPMENT_MANIFEST_SCHEMA
        or manifest["hash_rule"] != "sha256(canonical_json(sorted_entries))"
    ):
        _fail("development diagnostic source-manifest contract drifted")
    entries_raw = manifest["entries"]
    if type(entries_raw) is not list:
        _fail("development diagnostic source-manifest entries must be a list")
    entry_count = _integer(
        manifest["entry_count"],
        label="development diagnostic source-manifest entry_count",
        minimum=1,
    )
    if entry_count != 170 or len(entries_raw) != entry_count:
        _fail("development diagnostic source manifest must contain exactly 170 entries")
    entries: list[dict[str, object]] = []
    records: list[FileRecord] = []
    for index, entry_value in enumerate(entries_raw):
        entry = _exact_keys(
            entry_value,
            {"path", "bytes", "sha256"},
            label=f"development source entry {index}",
        )
        relative = _manifest_relative_path(
            entry["path"], label=f"development source entry {index}.path"
        )
        posix = PurePosixPath(relative)
        allowed_scope = any(
            tuple(part.casefold() for part in posix.parts[: len(scope.parts)])
            == tuple(part.casefold() for part in scope.parts)
            for scope in SOURCE_ROOTS
        )
        if not allowed_scope or posix.suffix != ".py":
            _fail(f"development source entry is outside executable Python scopes: {relative}")
        expected_bytes = _integer(
            entry["bytes"], label=f"development source entry {index}.bytes", minimum=1
        )
        expected_sha = _sha256(
            entry["sha256"], label=f"development source entry {index}.sha256"
        )
        path = _contained_input_file(root, root.joinpath(*posix.parts), label=relative)
        if _canonical_relative_path(path, root=root, label=relative) != relative:
            _fail(f"development source path spelling or case drifted: {relative}")
        record = _stable_read(
            path, label=f"development source file {relative}", relative_path=relative
        )
        if len(record.raw) != expected_bytes or record.sha256 != expected_sha:
            _fail(f"development source file disagrees with frozen plan: {relative}")
        entries.append({"path": relative, "bytes": expected_bytes, "sha256": expected_sha})
        records.append(record)
    paths = [str(entry["path"]) for entry in entries]
    if paths != sorted(paths, key=lambda item: (item.casefold(), item)):
        _fail("development diagnostic source-manifest entries are not canonically sorted")
    if len({path.casefold() for path in paths}) != len(paths):
        _fail("development diagnostic source manifest repeats a path")
    source_snapshot = _sha256_bytes(_canonical_bytes(entries))
    if _sha256(
        manifest["source_snapshot_sha256"],
        label="development diagnostic source_snapshot_sha256",
    ) != source_snapshot:
        _fail("development diagnostic source-manifest root hash disagrees with entries")
    canonical_manifest = {
        "schema": DEVELOPMENT_MANIFEST_SCHEMA,
        "entry_count": entry_count,
        "hash_rule": "sha256(canonical_json(sorted_entries))",
        "source_snapshot_sha256": source_snapshot,
        "entries": entries,
    }
    return canonical_manifest, records, _canonical_bytes(canonical_manifest)


def _validate_development_plan(
    value: object, *, root: Path
) -> tuple[dict[str, Any], list[FileRecord], bytes]:
    raw = _exact_keys(value, DEVELOPMENT_PLAN_KEYS, label="development diagnostic plan")
    if (
        raw["schema"] != DEVELOPMENT_PLAN_SCHEMA
        or raw["status"] != "FROZEN_FULL_504_DEVELOPMENT_DIAGNOSTIC"
        or raw["scientific_scope"]
        != "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE"
        or raw["selection_entropy_release"] != "PROHIBITED"
        or raw["confirmation_materialization"] != "PROHIBITED"
        or raw["formal_materialization"] != "PROHIBITED"
    ):
        _fail("development diagnostic plan boundary contract drifted")
    arms = _exact_string_list(raw["arms"], label="development diagnostic arms")
    cases = _exact_string_list(raw["case_ids"], label="development diagnostic case_ids")
    seeds = _exact_integer_list(raw["seeds"], label="development diagnostic seeds")
    if arms != DEVELOPMENT_ARMS or cases != DEVELOPMENT_CASE_IDS or seeds != DEVELOPMENT_SEEDS:
        _fail("development diagnostic 14-arm/case/seed matrix drifted")
    if (
        _integer(raw["charged_evaluation_budget"], label="charged_evaluation_budget")
        != 2000
        or _integer(raw["checkpoint_period"], label="checkpoint_period") != 200
        or _integer(raw["row_timeout_seconds"], label="row_timeout_seconds") != 1800
        or _integer(raw["expected_rows"], label="expected_rows") != 504
        or len(arms) * len(cases) * len(seeds) != 504
    ):
        _fail("development diagnostic budget or exact-504 matrix drifted")
    input_binding = _exact_keys(
        raw["input_binding"],
        {"schema", "case_ids", "manifest_sha256"},
        label="development diagnostic input binding",
    )
    if (
        input_binding["schema"] != "v21e3r1_exposed_development_input_binding_v1"
        or _exact_string_list(
            input_binding["case_ids"], label="development input-binding case_ids"
        )
        != cases
    ):
        _fail("development diagnostic input binding drifted")
    manifest_hashes = _exact_keys(
        input_binding["manifest_sha256"],
        DEVELOPMENT_INPUT_MANIFEST_PATHS,
        label="development input manifest hashes",
    )
    for path, digest in manifest_hashes.items():
        _sha256(digest, label=f"development input manifest hash {path}")
    return _development_manifest_records(root, raw["source_manifest"])


def _validate_parent_development_snapshot(
    *, root: Path, receipt_path_value: str | Path, expected_source_sha256: str
) -> list[FileRecord]:
    receipt_path = _contained_input_file(
        root, receipt_path_value, label="parent development snapshot receipt"
    )
    receipt_record = _stable_read(
        receipt_path, label="parent development snapshot receipt"
    )
    receipt = _parse_canonical_json(
        receipt_record, label="parent development snapshot receipt"
    )
    raw = _exact_keys(
        receipt, DEVELOPMENT_RECEIPT_KEYS, label="parent development snapshot receipt"
    )
    fixed_values = {
        "schema": DEVELOPMENT_RECEIPT_SCHEMA,
        "status": DEVELOPMENT_PASS_STATUS,
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "diagnostic_plan_schema": DEVELOPMENT_PLAN_SCHEMA,
        "diagnostic_plan_copy_path": DEVELOPMENT_OUTPUT_NAMES["plan"],
        "source_manifest_schema": DEVELOPMENT_MANIFEST_SCHEMA,
        "source_manifest_path": DEVELOPMENT_OUTPUT_NAMES["source_manifest"],
        "source_archive_path": DEVELOPMENT_OUTPUT_NAMES["archive"],
        "source_archive_scope": (
            "PLAN_BOUND_DEVELOPMENT_EXECUTION_SOURCE_ONLY_INTERNAL_CUSTODY_"
            "NO_REDISTRIBUTION_AUTHORITY"
        ),
        "ijoc_submission_status": "IJOC_HOLD",
    }
    if any(raw[field] != expected for field, expected in fixed_values.items()):
        _fail("parent development snapshot receipt contract drifted")
    if (
        _sha256(raw["source_snapshot_sha256"], label="parent source_snapshot_sha256")
        != expected_source_sha256
        or _integer(raw["source_entry_count"], label="parent source_entry_count") != 170
        or _integer(raw["source_total_bytes"], label="parent source_total_bytes", minimum=1)
        < 1
        or not _boolean(raw["all_source_files_verified"], label="parent all_source_files_verified")
        or not _boolean(raw["source_archive_materialized"], label="parent source_archive_materialized")
    ):
        _fail("parent development snapshot identity or completeness drifted")
    for field in (
        "development_execution_replayed",
        "original_development_diagnostic_artifacts_modified",
        "selection_cases_materialized",
        "confirmation_cases_materialized",
        "formal_cases_materialized",
        "implementation_independence",
        "scientific_independence",
        "selection_authorized",
        "confirmation_authorized",
        "formal_study_authorized",
        "scientific_claim_authorized",
        "public_redistribution_authorized",
    ):
        observed = _boolean(raw[field], label=f"parent snapshot {field}")
        if observed is not False:
            _fail(f"parent development snapshot field {field} must remain false")
    core = dict(raw)
    payload_sha = _sha256(
        core.pop("receipt_payload_sha256"),
        label="parent snapshot receipt_payload_sha256",
    )
    if payload_sha != _sha256_bytes(_canonical_bytes(core)):
        _fail("parent development snapshot receipt payload hash drifted")

    parent = receipt_path.parent
    plan_record = _stable_read(
        _contained_input_file(
            root, parent / DEVELOPMENT_OUTPUT_NAMES["plan"], label="parent plan copy"
        ),
        label="parent development plan copy",
    )
    manifest_record = _stable_read(
        _contained_input_file(
            root,
            parent / DEVELOPMENT_OUTPUT_NAMES["source_manifest"],
            label="parent source manifest",
        ),
        label="parent development source manifest",
    )
    archive_record = _stable_read(
        _contained_input_file(
            root,
            parent / DEVELOPMENT_OUTPUT_NAMES["archive"],
            label="parent source archive",
        ),
        label="parent development source archive",
    )
    if (
        plan_record.sha256
        != _sha256(raw["diagnostic_plan_sha256"], label="parent diagnostic_plan_sha256")
        or manifest_record.sha256
        != _sha256(raw["source_manifest_sha256"], label="parent source_manifest_sha256")
        or archive_record.sha256
        != _sha256(raw["source_archive_sha256"], label="parent source_archive_sha256")
    ):
        _fail("parent development snapshot sibling artifact hash drifted")
    manifest = _parse_canonical_json(
        manifest_record, label="parent development source manifest"
    )
    manifest = _exact_keys(
        manifest,
        {"schema", "entry_count", "hash_rule", "source_snapshot_sha256", "entries"},
        label="parent development source manifest",
    )
    entries_raw = manifest["entries"]
    if (
        manifest["schema"] != DEVELOPMENT_MANIFEST_SCHEMA
        or manifest["hash_rule"] != "sha256(canonical_json(sorted_entries))"
        or _integer(manifest["entry_count"], label="parent manifest entry_count") != 170
        or type(entries_raw) is not list
        or len(entries_raw) != 170
    ):
        _fail("parent development source manifest structure drifted")
    entries: list[dict[str, object]] = []
    for index, entry_value in enumerate(entries_raw):
        entry = _exact_keys(
            entry_value,
            {"path", "bytes", "sha256"},
            label=f"parent development source entry {index}",
        )
        path = _manifest_relative_path(
            entry["path"], label=f"parent development source entry {index}.path"
        )
        posix = PurePosixPath(path)
        if posix.suffix != ".py" or not any(
            tuple(part.casefold() for part in posix.parts[: len(scope.parts)])
            == tuple(part.casefold() for part in scope.parts)
            for scope in SOURCE_ROOTS
        ):
            _fail(f"parent development source entry escapes executable scope: {path}")
        byte_count = _integer(
            entry["bytes"],
            label=f"parent development source entry {index}.bytes",
            minimum=1,
        )
        digest = _sha256(
            entry["sha256"], label=f"parent development source entry {index}.sha256"
        )
        entries.append({"path": path, "bytes": byte_count, "sha256": digest})
    paths = [str(entry["path"]) for entry in entries]
    if (
        paths != sorted(paths, key=lambda item: (item.casefold(), item))
        or len({path.casefold() for path in paths}) != len(paths)
    ):
        _fail("parent development source manifest paths are not canonical and unique")
    source_root = _sha256_bytes(_canonical_bytes(entries))
    if (
        source_root != expected_source_sha256
        or manifest["source_snapshot_sha256"] != source_root
        or sum(int(entry["bytes"]) for entry in entries) != raw["source_total_bytes"]
    ):
        _fail("parent development source manifest root or byte total drifted")
    plan = _parse_json(
        plan_record, label="parent development plan copy", require_canonical=False
    )
    if (
        plan.get("schema") != DEVELOPMENT_PLAN_SCHEMA
        or plan.get("source_manifest") != manifest
    ):
        _fail("parent development plan copy disagrees with source manifest")
    try:
        with zipfile.ZipFile(io.BytesIO(archive_record.raw), mode="r") as archive:
            infos = archive.infolist()
            if [info.filename for info in infos] != paths:
                _fail("parent development source archive member closure drifted")
            for info, entry in zip(infos, entries, strict=True):
                archived = archive.read(info)
                if (
                    info.compress_type != zipfile.ZIP_STORED
                    or info.date_time != FIXED_TIMESTAMP
                    or info.create_system != 3
                    or info.external_attr >> 16 != 0o100644
                    or len(archived) != entry["bytes"]
                    or _sha256_bytes(archived) != entry["sha256"]
                ):
                    _fail(f"parent development archive member drifted: {info.filename}")
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        _fail(f"parent development source archive is invalid: {error}")
    return [receipt_record, plan_record, manifest_record, archive_record]


def _validate_output_path(root: Path, output: Path) -> Path:
    absolute = Path(os.path.abspath(output if output.is_absolute() else root / output))
    try:
        relative = absolute.relative_to(root)
    except ValueError:
        _fail("output directory escapes project root")
    if not relative.parts:
        _fail("output directory cannot be the project root")
    prefixes = tuple(tuple(part.casefold() for part in item.parts) for item in SOURCE_ROOTS)
    relative_parts = tuple(part.casefold() for part in relative.parts)
    if any(relative_parts[: len(prefix)] == prefix for prefix in prefixes):
        _fail("output directory cannot be inside the frozen source scope")
    cursor = root
    for part in relative.parts[:-1]:
        cursor = cursor / part
        if _lstat_is_link_or_reparse(cursor):
            _fail("output directory traverses a symbolic link, junction, or reparse point")
    try:
        absolute.resolve(strict=False).relative_to(root)
    except (OSError, ValueError):
        _fail("resolved output directory escapes project root")
    if absolute.exists() or _lstat_is_link_or_reparse(absolute):
        _fail("output directory already exists; exclusive create required")
    return absolute


def _revalidate_output_directory(root: Path, output: Path) -> None:
    try:
        resolved = output.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        _fail("created output directory escapes project root")
    cursor = root
    relative = output.relative_to(root)
    for part in relative.parts:
        cursor = cursor / part
        if _lstat_is_link_or_reparse(cursor):
            _fail("created output directory traverses a link or reparse point")
    if not output.is_dir():
        _fail("created output directory is not a directory")


def _write_exclusive(path: Path, raw: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        _fail(f"output artifact already exists; exclusive create required: {path.name}")


def _source_archive(records: Sequence[FileRecord]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for record in records:
            if record.relative_path is None:
                raise AssertionError("source archive record lacks a relative path")
            info = zipfile.ZipInfo(record.relative_path, FIXED_TIMESTAMP)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED
            info.extra = b""
            info.comment = b""
            archive.writestr(
                info,
                record.raw,
                compress_type=zipfile.ZIP_STORED,
            )
    return buffer.getvalue()


def freeze_development_diagnostic_source(
    *,
    project_root: str | Path,
    output_directory: str | Path,
    development_diagnostic_plan_path: str | Path,
) -> dict[str, object]:
    """Materialize the exact source bytes bound by the frozen 504-row plan."""

    root = Path(project_root).resolve(strict=True)
    if not root.is_dir():
        _fail("project root is not a directory")
    plan_path = _contained_input_file(
        root,
        development_diagnostic_plan_path,
        label="development diagnostic plan",
    )
    output = _validate_output_path(root, Path(output_directory))
    try:
        output.relative_to(plan_path.parent)
    except ValueError:
        pass
    else:
        _fail("snapshot output cannot be inside the live diagnostic output directory")

    plan_record = _stable_read(plan_path, label="development diagnostic plan")
    plan = _parse_json(
        plan_record,
        label="development diagnostic plan",
        require_canonical=False,
    )
    source_manifest, source_records, manifest_raw = _validate_development_plan(
        plan, root=root
    )
    bound_inputs = [plan_record, *source_records]
    _revalidate_records(bound_inputs)
    archive_raw = _source_archive(source_records)
    archive_sha = _sha256_bytes(archive_raw)

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir()
    except FileExistsError:
        _fail("output directory already exists; exclusive create required")
    _revalidate_output_directory(root, output)
    _write_exclusive(output / DEVELOPMENT_OUTPUT_NAMES["plan"], plan_record.raw)
    _write_exclusive(output / DEVELOPMENT_OUTPUT_NAMES["source_manifest"], manifest_raw)
    _write_exclusive(output / DEVELOPMENT_OUTPUT_NAMES["archive"], archive_raw)

    # Re-read every plan-bound source immediately before publishing PASS.  The
    # live diagnostic directory is never opened for writing.
    _revalidate_records(bound_inputs)
    _revalidate_output_directory(root, output)
    entries = source_manifest["entries"]
    if type(entries) is not list:
        raise AssertionError("validated development manifest lost its entries list")
    core: dict[str, object] = {
        "schema": DEVELOPMENT_RECEIPT_SCHEMA,
        "status": DEVELOPMENT_PASS_STATUS,
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "diagnostic_plan_schema": DEVELOPMENT_PLAN_SCHEMA,
        "diagnostic_plan_copy_path": DEVELOPMENT_OUTPUT_NAMES["plan"],
        "diagnostic_plan_sha256": plan_record.sha256,
        "source_manifest_schema": DEVELOPMENT_MANIFEST_SCHEMA,
        "source_manifest_path": DEVELOPMENT_OUTPUT_NAMES["source_manifest"],
        "source_manifest_sha256": _sha256_bytes(manifest_raw),
        "source_snapshot_sha256": source_manifest["source_snapshot_sha256"],
        "source_entry_count": len(entries),
        "source_total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "all_source_files_verified": True,
        "source_archive_materialized": True,
        "source_archive_path": DEVELOPMENT_OUTPUT_NAMES["archive"],
        "source_archive_sha256": archive_sha,
        "source_archive_scope": (
            "PLAN_BOUND_DEVELOPMENT_EXECUTION_SOURCE_ONLY_INTERNAL_CUSTODY_"
            "NO_REDISTRIBUTION_AUTHORITY"
        ),
        "development_execution_replayed": False,
        "original_development_diagnostic_artifacts_modified": False,
        "selection_cases_materialized": False,
        "confirmation_cases_materialized": False,
        "formal_cases_materialized": False,
        "implementation_independence": False,
        "scientific_independence": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_study_authorized": False,
        "scientific_claim_authorized": False,
        "public_redistribution_authorized": False,
        "ijoc_submission_status": "IJOC_HOLD",
    }
    receipt = dict(core)
    receipt["receipt_payload_sha256"] = _sha256_bytes(_canonical_bytes(core))
    _write_exclusive(
        output / DEVELOPMENT_OUTPUT_NAMES["receipt"], _canonical_bytes(receipt)
    )
    return {
        "mode": "development_diagnostic_source_snapshot",
        "receipt": receipt,
        "output_directory": str(output),
        "source_manifest_sha256": _sha256_bytes(manifest_raw),
        "source_archive_sha256": archive_sha,
    }


def freeze_source(
    *,
    project_root: str | Path,
    output_directory: str | Path,
    study_id: str,
    candidate_id: str,
    parent_development_source_sha256: str,
    parent_development_snapshot_receipt_path: str | Path,
    semantic_config_path: str | Path,
    study_metric_spec_path: str | Path,
    simultaneous_inference_spec_path: str | Path,
    source_archive: bool,
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=True)
    if not root.is_dir():
        _fail("project root is not a directory")
    study = _identifier(study_id, label="study_id")
    candidate = _identifier(candidate_id, label="candidate_id")
    parent_sha = _sha256(
        parent_development_source_sha256,
        label="parent_development_source_sha256",
    )
    parent_snapshot_records = _validate_parent_development_snapshot(
        root=root,
        receipt_path_value=parent_development_snapshot_receipt_path,
        expected_source_sha256=parent_sha,
    )
    if type(source_archive) is not bool:
        _fail("source_archive must be an exact boolean")
    output = _validate_output_path(root, Path(output_directory))

    source_records = _collect_source_records(root)
    entries = [
        {
            "path": record.relative_path,
            "bytes": len(record.raw),
            "sha256": record.sha256,
        }
        for record in source_records
    ]
    source_root_sha = _sha256_bytes(_canonical_bytes(entries))
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "source_root_sha256": source_root_sha,
        "entries": entries,
    }
    manifest_raw = _canonical_bytes(manifest)

    semantic_path = _contained_input_file(
        root, semantic_config_path, label="semantic config"
    )
    study_metric_path = _contained_input_file(
        root, study_metric_spec_path, label="study metric spec"
    )
    simultaneous_path = _contained_input_file(
        root,
        simultaneous_inference_spec_path,
        label="simultaneous-inference spec",
    )
    if len({semantic_path, study_metric_path, simultaneous_path}) != 3:
        _fail("semantic, study-metric, and simultaneous specs must be distinct files")
    semantic_record = _stable_read(semantic_path, label="semantic config")
    metric_record = _stable_read(study_metric_path, label="study metric spec")
    simultaneous_record = _stable_read(
        simultaneous_path, label="simultaneous-inference spec"
    )
    semantic = _parse_canonical_json(semantic_record, label="semantic config")
    study_metric = _parse_canonical_json(metric_record, label="study metric spec")
    simultaneous = _parse_canonical_json(
        simultaneous_record, label="simultaneous-inference spec"
    )
    _validate_semantic_config(semantic, study_id=study, candidate_id=candidate)
    _validate_study_metric_spec(study_metric, source_records=source_records)
    _validate_simultaneous_spec(
        simultaneous,
        study_id=study,
        candidate_id=candidate,
        source_root_sha256=source_root_sha,
        semantic_config_sha256=semantic_record.sha256,
        study_metric_spec_sha256=metric_record.sha256,
        source_records=source_records,
    )
    bound_specs = [
        *parent_snapshot_records,
        semantic_record,
        metric_record,
        simultaneous_record,
    ]
    _revalidate_records(bound_specs)
    _revalidate_source_inventory(root, source_records)

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir()
    except FileExistsError:
        _fail("output directory already exists; exclusive create required")
    _revalidate_output_directory(root, output)
    _write_exclusive(output / FIXED_OUTPUT_NAMES["source_manifest"], manifest_raw)
    _write_exclusive(
        output / FIXED_OUTPUT_NAMES["semantic_config"], semantic_record.raw
    )
    _write_exclusive(
        output / FIXED_OUTPUT_NAMES["study_metric_spec"], metric_record.raw
    )
    _write_exclusive(
        output / FIXED_OUTPUT_NAMES["simultaneous_spec"], simultaneous_record.raw
    )
    archive_raw: bytes | None = None
    archive_sha: str | None = None
    if source_archive:
        archive_raw = _source_archive(source_records)
        archive_sha = _sha256_bytes(archive_raw)
        _write_exclusive(output / FIXED_OUTPUT_NAMES["archive"], archive_raw)

    # Receipt publication is last.  A source/config mutation during materialization
    # leaves no PASS receipt, and all later consumers re-check the bound hashes.
    _revalidate_records(bound_specs)
    _revalidate_source_inventory(root, source_records)
    _revalidate_output_directory(root, output)
    core: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "status": PASS_STATUS,
        "study_id": study,
        "candidate_id": candidate,
        "parent_development_source_sha256": parent_sha,
        "source_snapshot_sha256": source_root_sha,
        "source_manifest_sha256": _sha256_bytes(manifest_raw),
        "semantic_config_sha256": semantic_record.sha256,
        "study_metric_spec_sha256": metric_record.sha256,
        "simultaneous_inference_spec_sha256": simultaneous_record.sha256,
        "source_entry_count": len(entries),
        "source_total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "all_source_files_verified": True,
        "source_frozen": True,
        "selection_cases_materialized": False,
        "confirmation_cases_materialized": False,
        "formal_cases_materialized": False,
        "source_archive_materialized": source_archive,
        "source_archive_path": FIXED_OUTPUT_NAMES["archive"] if source_archive else None,
        "source_archive_sha256": archive_sha,
        "source_archive_scope": (
            "SOURCE_INVENTORY_ONLY_INTERNAL_CUSTODY_NO_REDISTRIBUTION_AUTHORITY"
            if source_archive
            else "NOT_MATERIALIZED"
        ),
        "implementation_independence": False,
        "scientific_independence": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_study_authorized": False,
        "scientific_claim_authorized": False,
        "public_redistribution_authorized": False,
        "ijoc_submission_status": "IJOC_HOLD",
    }
    receipt = dict(core)
    receipt["receipt_payload_sha256"] = _sha256_bytes(_canonical_bytes(core))
    receipt_raw = _canonical_bytes(receipt)
    _write_exclusive(output / FIXED_OUTPUT_NAMES["receipt"], receipt_raw)
    return {
        "receipt": receipt,
        "output_directory": str(output),
        "source_manifest_sha256": _sha256_bytes(manifest_raw),
        "source_archive_sha256": archive_sha,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--study-id")
    parser.add_argument("--candidate-id")
    parser.add_argument("--parent-development-source-sha256")
    parser.add_argument("--parent-development-snapshot-receipt")
    parser.add_argument("--semantic-config")
    parser.add_argument("--study-metric-spec")
    parser.add_argument("--simultaneous-inference-spec")
    parser.add_argument(
        "--development-diagnostic-plan",
        help=(
            "Use the independent development snapshot mode: verify the exact "
            "170-entry source manifest in diagnostic.plan.json and create a "
            "mandatory deterministic restorable source archive."
        ),
    )
    parser.add_argument(
        "--source-archive",
        action="store_true",
        help=(
            "In successor mode, create a deterministic source-only internal custody "
            "ZIP. Development snapshot mode always creates its plan-bound archive. "
            "Neither mode authorizes public redistribution."
        ),
    )
    arguments = parser.parse_args(argv)
    try:
        successor_arguments = {
            "study_id": arguments.study_id,
            "candidate_id": arguments.candidate_id,
            "parent_development_source_sha256": (
                arguments.parent_development_source_sha256
            ),
            "parent_development_snapshot_receipt": (
                arguments.parent_development_snapshot_receipt
            ),
            "semantic_config": arguments.semantic_config,
            "study_metric_spec": arguments.study_metric_spec,
            "simultaneous_inference_spec": arguments.simultaneous_inference_spec,
        }
        if arguments.development_diagnostic_plan is not None:
            supplied = sorted(
                name for name, value in successor_arguments.items() if value is not None
            )
            if supplied:
                _fail(
                    "development snapshot mode cannot mix successor arguments: "
                    f"{supplied}"
                )
            result = freeze_development_diagnostic_source(
                project_root=arguments.project_root,
                output_directory=arguments.output_directory,
                development_diagnostic_plan_path=arguments.development_diagnostic_plan,
            )
        else:
            missing = sorted(
                name for name, value in successor_arguments.items() if value is None
            )
            if missing:
                _fail(f"successor freeze mode is missing required arguments: {missing}")
            result = freeze_source(
                project_root=arguments.project_root,
                output_directory=arguments.output_directory,
                study_id=arguments.study_id,
                candidate_id=arguments.candidate_id,
                parent_development_source_sha256=(
                    arguments.parent_development_source_sha256
                ),
                parent_development_snapshot_receipt_path=(
                    arguments.parent_development_snapshot_receipt
                ),
                semantic_config_path=arguments.semantic_config,
                study_metric_spec_path=arguments.study_metric_spec,
                simultaneous_inference_spec_path=arguments.simultaneous_inference_spec,
                source_archive=bool(arguments.source_archive),
            )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except (FreezeError, OSError, zipfile.BadZipFile) as error:
        print(
            json.dumps(
                {
                    "schema": "v21e3r1_successor_source_freeze_error_v1",
                    "status": "HOLD_INTEGRITY_ERROR",
                    "error": str(error),
                    "source_frozen": False,
                    "selection_authorized": False,
                    "confirmation_authorized": False,
                    "formal_study_authorized": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
