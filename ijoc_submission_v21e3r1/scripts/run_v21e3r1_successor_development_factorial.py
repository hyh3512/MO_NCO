from __future__ import annotations

"""Run the exact V21e3r1 successor factorial on exposed development cases.

This program has no interface for selection, confirmation, or formal cases.  It
requires the sealed V7 504-row diagnostic, a frozen successor-source receipt,
and the packaged prospective inference specification before it will create a
plan.  Rows and attempts are append-only; ``--resume`` verifies every sealed
artifact before continuing.
"""

import argparse
from dataclasses import fields, replace
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess
import sys
from typing import Mapping, Sequence

from ijoc_submission_v21e3r1.scripts import (
    run_v21e3r1_development_diagnostics as v7_runner,
)
from ijoc_submission_v21e3r1.scripts.freeze_v21e3r1_successor_source import (
    FIXED_OUTPUT_NAMES as SOURCE_FREEZE_OUTPUT_NAMES,
    FreezeError as SourceFreezeError,
    _collect_source_records,
)
from mo_nco.pareto_v21e3_hybrid import (
    V21E3HybridConfig,
    V21E3TypedHybridParetoSearch,
)
from mo_nco.pareto_v21e3_parity import normalized_left_continuous_auc
from mo_nco.pareto_v21e3_trace_verify import verify_v21e3_trace_database
from mo_nco.pareto_v21e3r1_development_diagnostics import hybrid_diagnostic_config


PLAN_SCHEMA = "v21e3r1_successor_development_factorial_plan_v2"
ROW_SCHEMA = "v21e3r1_successor_development_factorial_row_v2"
COMPLETED_SCHEMA = "v21e3r1_successor_development_factorial_completed_row_v2"
RECEIPT_SCHEMA = "v21e3r1_successor_development_factorial_receipt_v2"
AGGREGATE_SCHEMA = "v21e3r1_successor_development_factorial_aggregate_v2"
INFERENCE_SCHEMA = "v21e3r1_successor_development_factorial_inference_spec_v1"
WORKER_SPEC_SCHEMA = "v21e3r1_successor_factorial_worker_spec_v2"
SOURCE_RECEIPT_SCHEMA = "v21e3r1_successor_source_freeze_receipt_v2"
SOURCE_RECEIPT_STATUS = "PASS_SUCCESSOR_SOURCE_AND_CONFIG_FREEZE_ENGINEERING_ONLY"
PARENT_PLAN_SCHEMA = "v21e3r1_exposed_development_diagnostic_plan_v2"
PARENT_RECEIPT_SCHEMA = "v21e3r1_exposed_development_diagnostic_receipt_v2"
PARENT_PLAN_SHA256 = "4408d10944cb6511e99ff0bd95ded256b9c230b91d8806a7bd5b962f10622886"
PARENT_SOURCE_SHA256 = "218bc398f04722d1da305928a9c206641f9b43d74b2afbc46c29ba1f08d6639b"
INFERENCE_SPEC_SHA256 = "5aa767bcc00c5ee8d220defa86b358d3e72a5849a99712a0f486159f1f032f3d"
METHOD = "one_sided_observed_se_max_t_paired_case_cluster_bootstrap_v1"
SEEDS = (31051, 31057, 31059)
FULL_BUDGET = 2000
FULL_CHECKPOINT_PERIOD = 200
FULL_ROW_COUNT = 108
LEGACY_SEARCH = "proposal_chain_v21e3r1_v1"
NEW_SEARCH = "post_commit_type_incumbent_anchor_development_v1"
LEGACY_NOVELTY = "legacy_retry_and_local_v21e3r1_v1"
NEW_NOVELTY = (
    "single_attempt_rotating_feasible_exchange_no_refill_development_v1"
)
MOKP_ARMS: tuple[tuple[str, str, str], ...] = (
    ("MOKP_LEGACY", LEGACY_SEARCH, LEGACY_NOVELTY),
    ("MOKP_ANCHOR_ONLY", NEW_SEARCH, LEGACY_NOVELTY),
    ("MOKP_NOVELTY_ONLY", LEGACY_SEARCH, NEW_NOVELTY),
    ("MOKP_BOTH", NEW_SEARCH, NEW_NOVELTY),
)
MOTSP_ARMS: tuple[tuple[str, str, str], ...] = (
    ("MOTSP_LEGACY", LEGACY_SEARCH, LEGACY_NOVELTY),
    ("MOTSP_ANCHOR", NEW_SEARCH, LEGACY_NOVELTY),
)
EXPECTED_SEMANTIC_PARAMETERS = {
    "legacy_post_initialization_search_policy": LEGACY_SEARCH,
    "successor_post_initialization_search_policy": NEW_SEARCH,
    "legacy_mokp_novelty_generation_policy": LEGACY_NOVELTY,
    "successor_mokp_novelty_generation_policy": NEW_NOVELTY,
    "promoted_arm_by_family": {
        "MOKP": "MOKP_BOTH",
        "MOTSP": "MOTSP_ANCHOR",
    },
    "factorial_arm_ids_by_family": {
        "MOKP": [arm_id for arm_id, _search, _novelty in MOKP_ARMS],
        "MOTSP": [arm_id for arm_id, _search, _novelty in MOTSP_ARMS],
    },
}
INFERENCE_SPEC_RELATIVE = PurePosixPath(
    "ijoc_submission_v21e3r1/development/"
    "V21E3R1_SUCCESSOR_DEVELOPMENT_FACTORIAL_INFERENCE_V1.json"
)
SUCCESSOR_METRIC_RELATIVE = PurePosixPath(
    "independent_reproduction/recompute_v21e3r1_successor_metrics.py"
)
FROZEN_V2_METRIC_RELATIVE = PurePosixPath(
    "independent_reproduction/recompute_v21e3r1_metrics.py"
)
SUCCESSOR_METRIC_SCHEMA = (
    "v21e3r1_successor_independent_metric_reimplementation_v3"
)
SUCCESSOR_METRIC_STATUS = "PASS_SUCCESSOR_INDEPENDENT_METRIC_IMPLEMENTATION"
SUCCESSOR_METRIC_SEMANTICS_ID = (
    "normalized_left_continuous_hv_auc_binary64_v21e3r1_v2"
)
SUCCESSOR_METRIC_KERNEL_ID = (
    "incremental_sorted_nondominated_front_order_preserving_v1"
)
FROZEN_V2_METRIC_SHA256 = (
    "587d4ed4d647d8293b36449c835109ee3afa6e9899fe155f917a492fdf303ea2"
)
HISTORICAL_METRIC_INTERPRETER = Path(r"C:\miniconda3\python.exe")
HISTORICAL_METRIC_INTERPRETER_SHA256 = (
    "f77193cf0405ab440c39324bdb2f8864596321c1df888adbbe357f3d760f4716"
)
HEX = frozenset("0123456789abcdef")


class ContractError(ValueError):
    """A frozen workflow contract failed closed."""


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key is prohibited: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ContractError(f"non-finite JSON constant is prohibited: {value}")


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ContractError(f"required JSON file is absent: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as error:
        raise ContractError(f"JSON is not strict UTF-8: {path}") from error
    except json.JSONDecodeError as error:
        raise ContractError(f"malformed JSON at {path}: {error.msg}") from error
    if type(value) is not dict:
        raise ContractError(f"JSON root must be an exact object: {path}")
    return value


def _require_keys(
    value: object, expected: frozenset[str], location: str
) -> dict[str, object]:
    if type(value) is not dict:
        raise ContractError(f"{location} must be an exact JSON object")
    actual = frozenset(value)
    if actual != expected:
        raise ContractError(
            f"{location} key set drifted; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _exact_int(value: object, location: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{location} must be an exact integer >= {minimum}")
    return value


def _exact_number(value: object, location: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ContractError(f"{location} must be an exact finite JSON number")
    return float(value)


def _sha_text(value: object, location: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in HEX for char in value):
        raise ContractError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _relative_path(value: object, location: str) -> str:
    if type(value) is not str or not value:
        raise ContractError(f"{location} must be a nonempty relative POSIX path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        "\\" in value
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or posix.as_posix() != value
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ContractError(f"{location} must be a canonical relative POSIX path")
    return value


def _contained(root: Path, relative: str, location: str) -> Path:
    path = (root / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ContractError(f"{location} escapes the project root") from error
    return path


def _successor_independent_metric_replay(
    *,
    project_root: Path,
    trace: Path,
    lower: Sequence[float],
    upper: Sequence[float],
    budget: int,
    output: Path,
) -> dict[str, object]:
    project = project_root.resolve()
    script = _contained(
        project, SUCCESSOR_METRIC_RELATIVE.as_posix(), "successor metric source"
    )
    frozen_v2_metric = _contained(
        project, FROZEN_V2_METRIC_RELATIVE.as_posix(), "historical frozen V2 metric"
    )
    if _sha256(frozen_v2_metric) != FROZEN_V2_METRIC_SHA256:
        raise ContractError("historical frozen V2 metric source drifted")
    interpreter = HISTORICAL_METRIC_INTERPRETER.resolve()
    if (
        not interpreter.is_file()
        or _sha256(interpreter) != HISTORICAL_METRIC_INTERPRETER_SHA256
    ):
        raise ContractError("historical successor metric interpreter identity drifted")
    command = [
        str(interpreter),
        str(script),
        "--trace",
        str(trace.resolve()),
        "--lower=" + ",".join(repr(float(value)) for value in lower),
        "--upper=" + ",".join(repr(float(value)) for value in upper),
        "--expected-evaluations",
        str(budget),
        "--output",
        str(output.resolve()),
    ]
    completed = subprocess.run(
        command,
        cwd=project,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise ContractError(
            "successor independent metric replay failed: "
            + completed.stderr[-2000:]
        )
    receipt = _load_json(output)
    interpreter_identity = receipt.get("historical_metric_interpreter")
    operation_counts = receipt.get("metric_kernel_operation_counts")
    if (
        receipt.get("schema") != SUCCESSOR_METRIC_SCHEMA
        or receipt.get("status") != SUCCESSOR_METRIC_STATUS
        or receipt.get("metric_semantics_id") != SUCCESSOR_METRIC_SEMANTICS_ID
        or receipt.get("metric_kernel_id") != SUCCESSOR_METRIC_KERNEL_ID
        or receipt.get("evaluation_count") != budget
        or receipt.get("decision_count") != budget
        or receipt.get("terminal_accounting_gate") != "PASS"
        or receipt.get("trace_sha256") != _sha256(trace)
        or receipt.get("legacy_reference_metric_source_sha256")
        != FROZEN_V2_METRIC_SHA256
        or receipt.get("successor_metric_source_sha256") != _sha256(script)
        or receipt.get("reimplementation_source_sha256") != _sha256(script)
        or receipt.get("implementation_independence_from_project_metrics") is not True
        or receipt.get("algorithm_execution_independence") is not False
        or receipt.get("scientific_independence") is not False
        or receipt.get("selection_authority") is not False
        or receipt.get("confirmation_authority") is not False
        or receipt.get("formal_study_authority") is not False
        or receipt.get("scientific_claim_authority") is not False
        or receipt.get("publication_status") != "IJOC_HOLD"
        or type(interpreter_identity) is not dict
        or Path(str(interpreter_identity.get("path"))).resolve() != interpreter
        or interpreter_identity.get("sha256")
        != HISTORICAL_METRIC_INTERPRETER_SHA256
        or interpreter_identity.get("implementation") != "cpython"
        or type(interpreter_identity.get("version")) is not str
        or type(operation_counts) is not dict
        or operation_counts.get("point_count") != budget
        or any(
            type(operation_counts.get(field)) is not int
            or int(operation_counts[field]) < 0
            for field in (
                "insertion_front_probe_count",
                "hypervolume_front_scan_count",
                "max_front_size",
                "final_front_size",
            )
        )
    ):
        raise ContractError("successor independent metric receipt fails strict gates")
    return receipt


def _exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(value, newline=True))


def _inference_spec_path(root: Path) -> Path:
    return _contained(root, INFERENCE_SPEC_RELATIVE.as_posix(), "inference spec")


WORKER_SPEC_KEYS = frozenset({"schema", "row_id", "ordinal", "plan_sha256"})


def _load_worker_spec(path: Path) -> dict[str, object]:
    spec = _require_keys(_load_json(path), WORKER_SPEC_KEYS, "worker spec")
    if path.read_bytes() != _canonical_bytes(spec, newline=True):
        raise ContractError("worker spec is not canonical JSON plus LF")
    if spec["schema"] != WORKER_SPEC_SCHEMA:
        raise ContractError("worker spec schema drifted")
    if type(spec["row_id"]) is not str or not spec["row_id"]:
        raise ContractError("worker spec row_id must be an exact nonempty string")
    _exact_int(spec["ordinal"], "worker spec ordinal", minimum=1)
    _sha_text(spec["plan_sha256"], "worker spec plan_sha256")
    return spec


def validate_inference_payload(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ContractError("frozen inference payload must be an exact JSON object")
    try:
        observed = hashlib.sha256(_canonical_bytes(value, newline=True)).hexdigest()
    except (TypeError, ValueError) as error:
        raise ContractError("frozen inference payload is not canonical finite JSON") from error
    if observed != INFERENCE_SPEC_SHA256:
        raise ContractError("frozen inference payload SHA-256 drifted")
    return value


def load_inference_spec(root: str | Path) -> tuple[dict[str, object], str]:
    project = Path(root).resolve()
    path = _inference_spec_path(project)
    value = _load_json(path)
    if path.read_bytes() != _canonical_bytes(value, newline=True):
        raise ContractError("factorial inference spec is not canonical JSON plus LF")
    expected_keys = frozenset(
        {
            "bootstrap_samples",
            "bootstrap_seed",
            "case_resampling",
            "centering",
            "cluster_unit",
            "critical_value_floor",
            "effect_direction",
            "evaluator_source_path",
            "familywise_alpha",
            "familywise_scope",
            "hypotheses",
            "method",
            "parent_v7_diagnostic_plan_sha256",
            "phase",
            "promotion_scope",
            "quantile_convention",
            "rng_domain",
            "rng_protocol",
            "schema",
            "seed_aggregation",
            "selection_cases_materialized",
            "selection_confirmation_evaluator_reuse",
            "status",
            "studentization_denominator",
        }
    )
    _require_keys(value, expected_keys, "inference spec")
    scalar_expectations = {
        "schema": INFERENCE_SCHEMA,
        "status": "FROZEN_PROSPECTIVELY_BEFORE_SUCCESSOR_FACTORIAL_EXECUTION",
        "phase": "development",
        "promotion_scope": "SUCCESSOR_DEVELOPMENT_PROMOTION_ONLY_NO_SCIENTIFIC_CLAIM",
        "method": METHOD,
        "familywise_alpha": 0.05,
        "bootstrap_samples": 9999,
        "bootstrap_seed": 2026082301,
        "parent_v7_diagnostic_plan_sha256": PARENT_PLAN_SHA256,
        "effect_direction": "larger_is_better",
        "cluster_unit": "PAIRED_CASE",
        "seed_aggregation": "MEAN_WITHIN_CASE_ARM",
        "critical_value_floor": 0.0,
        "selection_cases_materialized": False,
        "selection_confirmation_evaluator_reuse": (
            "INCOMPATIBLE_ASYMMETRIC_4_ARM_MOKP_2_ARM_MOTSP_AND_MIXED_METRICS"
        ),
        "case_resampling": "WITH_REPLACEMENT_INDEPENDENT_BY_FAMILY_SHARED_ACROSS_CELLS_WITHIN_FAMILY",
        "centering": "BOOTSTRAP_MEAN_MINUS_OBSERVED_MEAN",
        "evaluator_source_path": "ijoc_submission_v21e3r1/scripts/evaluate_v21e3r1_successor_development_factorial.py",
        "familywise_scope": "JOINT_ACROSS_ALL_FIVE_DEVELOPMENT_PROMOTION_HYPOTHESES",
        "quantile_convention": "CEIL((1-ALPHA)*(B+1))_ORDER_STATISTIC",
        "rng_domain": "v21e3r1-successor-development-factorial-bootstrap-v1",
        "rng_protocol": "SHA256_COUNTER_U64_REJECTION_V1",
        "studentization_denominator": "OBSERVED_CASE_CLUSTER_STANDARD_ERROR",
    }
    for key, expected in scalar_expectations.items():
        if value.get(key) != expected or type(value.get(key)) is not type(expected):
            raise ContractError(f"frozen inference field drifted: {key}")
    hypotheses = value.get("hypotheses")
    if type(hypotheses) is not list or len(hypotheses) != 5:
        raise ContractError("inference spec must contain exactly five hypotheses")
    expected_hypotheses = (
        ("MOKP:BOTH_MINUS_LEGACY:EAUC", "MOKP", "exact_per_evaluation_left_continuous_hv_auc", "SUPERIORITY", 0.005, "MOKP_BOTH-MOKP_LEGACY"),
        ("MOKP:ANCHOR_MAIN_EFFECT:EAUC", "MOKP", "exact_per_evaluation_left_continuous_hv_auc", "SUPERIORITY", 0.0, "0.5*((MOKP_ANCHOR_ONLY-MOKP_LEGACY)+(MOKP_BOTH-MOKP_NOVELTY_ONLY))"),
        ("MOKP:NOVELTY_MAIN_EFFECT:EAUC", "MOKP", "exact_per_evaluation_left_continuous_hv_auc", "NONINFERIORITY", -0.005, "0.5*((MOKP_NOVELTY_ONLY-MOKP_LEGACY)+(MOKP_BOTH-MOKP_ANCHOR_ONLY))"),
        ("MOKP:NOVELTY_MAIN_EFFECT:CACHE_HIT_RATE_REDUCTION", "MOKP", "cache_hit_rate_per_attempt", "SUPERIORITY", 0.1, "0.5*((MOKP_LEGACY-MOKP_NOVELTY_ONLY)+(MOKP_ANCHOR_ONLY-MOKP_BOTH))"),
        ("MOTSP:ANCHOR_MINUS_LEGACY:EAUC", "MOTSP", "exact_per_evaluation_left_continuous_hv_auc", "SUPERIORITY", 0.0, "MOTSP_ANCHOR-MOTSP_LEGACY"),
    )
    for index, expected in enumerate(expected_hypotheses):
        raw = _require_keys(
            hypotheses[index],
            frozenset({"arm_contrast", "family", "hypothesis_id", "metric", "role", "threshold"}),
            f"hypotheses[{index}]",
        )
        observed = (
            raw["hypothesis_id"], raw["family"], raw["metric"], raw["role"], raw["threshold"], raw["arm_contrast"]
        )
        if observed != expected:
            raise ContractError(f"frozen hypothesis drifted at ordinal {index + 1}")
    observed_sha256 = _sha256(path)
    if observed_sha256 != INFERENCE_SPEC_SHA256:
        raise ContractError("factorial inference spec SHA-256 drifted")
    return validate_inference_payload(value), observed_sha256


SOURCE_RECEIPT_KEYS = frozenset(
    {
        "schema", "status", "study_id", "candidate_id",
        "parent_development_source_sha256", "source_snapshot_sha256",
        "source_manifest_sha256", "semantic_config_sha256",
        "study_metric_spec_sha256", "simultaneous_inference_spec_sha256",
        "source_entry_count", "source_total_bytes", "all_source_files_verified",
        "source_frozen", "selection_cases_materialized",
        "confirmation_cases_materialized", "formal_cases_materialized",
        "source_archive_materialized", "source_archive_path", "source_archive_sha256",
        "source_archive_scope", "implementation_independence",
        "scientific_independence", "selection_authorized", "confirmation_authorized",
        "formal_study_authorized", "scientific_claim_authorized",
        "public_redistribution_authorized", "ijoc_submission_status",
        "receipt_payload_sha256",
    }
)


def validate_successor_source_freeze(
    root: str | Path, receipt_path: str | Path
) -> dict[str, object]:
    project = Path(root).resolve()
    receipt_file = Path(receipt_path).resolve()
    try:
        receipt_file.relative_to(project)
    except ValueError as error:
        raise ContractError("successor source receipt escapes the project root") from error
    receipt = _require_keys(
        _load_json(receipt_file), SOURCE_RECEIPT_KEYS, "successor source receipt"
    )
    if receipt_file.read_bytes() != _canonical_bytes(receipt):
        raise ContractError("successor source receipt is not canonical JSON")
    core = dict(receipt)
    payload_hash = _sha_text(core.pop("receipt_payload_sha256"), "receipt payload hash")
    if _payload_sha256(core) != payload_hash:
        raise ContractError("successor source receipt payload hash drifted")
    if receipt["schema"] != SOURCE_RECEIPT_SCHEMA or receipt["status"] != SOURCE_RECEIPT_STATUS:
        raise ContractError("successor source receipt identity/status drifted")
    if (
        type(receipt["study_id"]) is not str
        or not receipt["study_id"]
        or type(receipt["candidate_id"]) is not str
        or not receipt["candidate_id"]
    ):
        raise ContractError("successor source study/candidate identity is invalid")
    if receipt["parent_development_source_sha256"] != PARENT_SOURCE_SHA256:
        raise ContractError("successor source parent development source drifted")
    source_entry_count = _exact_int(
        receipt["source_entry_count"], "successor source entry count", minimum=1
    )
    source_total_bytes = _exact_int(
        receipt["source_total_bytes"], "successor source total bytes", minimum=1
    )
    for field in (
        "parent_development_source_sha256", "source_snapshot_sha256",
        "source_manifest_sha256", "semantic_config_sha256",
        "study_metric_spec_sha256", "simultaneous_inference_spec_sha256",
    ):
        _sha_text(receipt[field], f"successor source receipt.{field}")
    for key in (
        "all_source_files_verified", "source_frozen",
    ):
        if receipt[key] is not True:
            raise ContractError(f"successor source receipt requires {key}=true")
    for key in (
        "selection_cases_materialized", "confirmation_cases_materialized",
        "formal_cases_materialized", "implementation_independence",
        "scientific_independence", "selection_authorized", "confirmation_authorized",
        "formal_study_authorized", "scientific_claim_authorized",
        "public_redistribution_authorized",
    ):
        if receipt[key] is not False:
            raise ContractError(f"successor source receipt requires {key}=false")
    if receipt["ijoc_submission_status"] != "IJOC_HOLD":
        raise ContractError("successor source freeze must retain IJOC_HOLD")

    directory = receipt_file.parent
    archive_materialized = receipt["source_archive_materialized"]
    if type(archive_materialized) is not bool:
        raise ContractError("successor source archive flag must be an exact boolean")
    if archive_materialized:
        if (
            receipt["source_archive_path"] != SOURCE_FREEZE_OUTPUT_NAMES["archive"]
            or receipt["source_archive_scope"]
            != "SOURCE_INVENTORY_ONLY_INTERNAL_CUSTODY_NO_REDISTRIBUTION_AUTHORITY"
        ):
            raise ContractError("successor source archive identity/scope drifted")
        archive_sha = _sha_text(
            receipt["source_archive_sha256"], "successor source archive SHA-256"
        )
        archive_path = directory / SOURCE_FREEZE_OUTPUT_NAMES["archive"]
        if not archive_path.is_file() or _sha256(archive_path) != archive_sha:
            raise ContractError("successor source archive artifact drifted")
    elif (
        receipt["source_archive_path"] is not None
        or receipt["source_archive_sha256"] is not None
        or receipt["source_archive_scope"] != "NOT_MATERIALIZED"
    ):
        raise ContractError("unmaterialized successor source archive has bindings")
    manifest_path = directory / "source.manifest.json"
    semantic_path = directory / "semantic.config.json"
    metric_path = directory / "study.metric-spec.json"
    simultaneous_path = directory / "simultaneous-inference.spec.json"
    bindings = (
        (manifest_path, "source_manifest_sha256"),
        (semantic_path, "semantic_config_sha256"),
        (metric_path, "study_metric_spec_sha256"),
        (simultaneous_path, "simultaneous_inference_spec_sha256"),
    )
    for path, field in bindings:
        if not path.is_file() or _sha256(path) != receipt[field]:
            raise ContractError(f"successor freeze sibling drifted: {path.name}")
    semantic = _require_keys(
        _load_json(semantic_path),
        frozenset({"schema", "study_id", "candidate_id", "parameters"}),
        "successor semantic config",
    )
    if semantic_path.read_bytes() != _canonical_bytes(semantic):
        raise ContractError("successor semantic config is not canonical JSON")
    if (
        semantic["schema"] != "v21e3r1_successor_semantic_config_v1"
        or semantic["study_id"] != receipt["study_id"]
        or semantic["candidate_id"] != receipt["candidate_id"]
        or type(semantic["parameters"]) is not dict
        or not semantic["parameters"]
    ):
        raise ContractError("successor semantic config identity drifted")
    if semantic["parameters"] != EXPECTED_SEMANTIC_PARAMETERS:
        raise ContractError("successor semantic config policy contract drifted")
    manifest = _require_keys(
        _load_json(manifest_path),
        frozenset({"schema", "source_root_sha256", "entries"}),
        "successor source manifest",
    )
    if manifest_path.read_bytes() != _canonical_bytes(manifest):
        raise ContractError("successor source manifest is not canonical JSON")
    if manifest["schema"] != "v21e3r1_branch_replay_source_manifest_binding_v1":
        raise ContractError("successor source manifest schema drifted")
    entries = manifest["entries"]
    if type(entries) is not list or len(entries) != source_entry_count:
        raise ContractError("successor source manifest entry count drifted")
    if _payload_sha256(entries) != manifest["source_root_sha256"]:
        raise ContractError("successor source manifest root hash drifted")
    if manifest["source_root_sha256"] != receipt["source_snapshot_sha256"]:
        raise ContractError("successor receipt/manifest source roots disagree")
    total_bytes = 0
    observed_paths: list[str] = []
    entry_by_path: dict[str, dict[str, object]] = {}
    for index, entry_value in enumerate(entries):
        entry = _require_keys(
            entry_value, frozenset({"path", "bytes", "sha256"}), f"source entries[{index}]"
        )
        relative = _relative_path(entry["path"], f"source entries[{index}].path")
        if relative.casefold() in {item.casefold() for item in observed_paths}:
            raise ContractError("successor source manifest contains duplicate paths")
        observed_paths.append(relative)
        entry_by_path[relative] = entry
        expected_bytes = _exact_int(entry["bytes"], f"source entries[{index}].bytes")
        expected_hash = _sha_text(entry["sha256"], f"source entries[{index}].sha256")
        path = _contained(project, relative, f"source entries[{index}]")
        if not path.is_file() or path.stat().st_size != expected_bytes or _sha256(path) != expected_hash:
            raise ContractError(f"successor source tree drifted: {relative}")
        total_bytes += expected_bytes
    if total_bytes != source_total_bytes:
        raise ContractError("successor source total bytes drifted")
    try:
        live_records = _collect_source_records(project)
    except SourceFreezeError as error:
        raise ContractError(
            f"successor source inventory closure cannot be re-enumerated: {error}"
        ) from error
    live_entries = [
        {
            "path": record.relative_path,
            "bytes": len(record.raw),
            "sha256": record.sha256,
        }
        for record in live_records
    ]
    if live_entries != entries:
        raise ContractError("successor source inventory closure drifted")
    required_source_paths = (
        "ijoc_submission_v21e3r1/scripts/run_v21e3r1_successor_development_factorial.py",
        "ijoc_submission_v21e3r1/scripts/evaluate_v21e3r1_successor_development_factorial.py",
        SUCCESSOR_METRIC_RELATIVE.as_posix(),
        "mo_nco/pareto_v21e3_hybrid.py",
        "mo_nco/pareto_v21e3r1_development_diagnostics.py",
        INFERENCE_SPEC_RELATIVE.as_posix(),
    )
    missing_required = [path for path in required_source_paths if path not in entry_by_path]
    if missing_required:
        raise ContractError(
            "successor source manifest omits factorial execution sources: "
            + ",".join(missing_required)
        )
    inference_entry = entry_by_path[INFERENCE_SPEC_RELATIVE.as_posix()]
    if inference_entry["sha256"] != INFERENCE_SPEC_SHA256:
        raise ContractError("successor source manifest factorial inference binding drifted")
    return {
        "schema": "v21e3r1_successor_factorial_source_binding_v2",
        "study_id": receipt["study_id"],
        "candidate_id": receipt["candidate_id"],
        "parent_development_source_sha256": receipt[
            "parent_development_source_sha256"
        ],
        "receipt_path": receipt_file.relative_to(project).as_posix(),
        "receipt_sha256": _sha256(receipt_file),
        "source_manifest_path": manifest_path.relative_to(project).as_posix(),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_snapshot_sha256": receipt["source_snapshot_sha256"],
        "semantic_config_sha256": receipt["semantic_config_sha256"],
        "study_metric_spec_sha256": receipt["study_metric_spec_sha256"],
        "simultaneous_inference_spec_sha256": receipt["simultaneous_inference_spec_sha256"],
        "factorial_inference_spec_path": INFERENCE_SPEC_RELATIVE.as_posix(),
        "factorial_inference_spec_sha256": INFERENCE_SPEC_SHA256,
    }


def _validate_parent_plan(root: Path, path: Path) -> dict[str, object]:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ContractError("V7 diagnostic plan escapes the project root") from error
    if _sha256(path) != PARENT_PLAN_SHA256:
        raise ContractError("V7 diagnostic plan SHA-256 drifted")
    plan = _load_json(path)
    if (
        plan.get("schema") != PARENT_PLAN_SCHEMA
        or plan.get("status") != "FROZEN_FULL_504_DEVELOPMENT_DIAGNOSTIC"
        or plan.get("scientific_scope") != "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE"
        or plan.get("case_ids") != list(v7_runner.EXPECTED_CASE_IDS)
        or plan.get("seeds") != list(SEEDS)
        or plan.get("arms") != list(v7_runner.DIAGNOSTIC_ARMS)
        or plan.get("charged_evaluation_budget") != FULL_BUDGET
        or plan.get("checkpoint_period") != FULL_CHECKPOINT_PERIOD
        or plan.get("expected_rows") != 504
        or plan.get("source_manifest", {}).get("source_snapshot_sha256") != PARENT_SOURCE_SHA256
        or plan.get("selection_entropy_release") != "PROHIBITED"
        or plan.get("confirmation_materialization") != "PROHIBITED"
        or plan.get("formal_materialization") != "PROHIBITED"
    ):
        raise ContractError("V7 diagnostic plan is not the exact frozen 504 design")
    return plan


PARENT_RECEIPT_KEYS = frozenset(
    {
        "schema", "status", "scientific_scope", "matrix_mode",
        "completed_rows", "expected_rows", "plan_sha256",
        "source_snapshot_sha256", "aggregate_sha256",
        "selection_entropy_release", "confirmation_materialization",
        "formal_materialization",
    }
)


def _validate_parent_receipt_payload(value: object) -> dict[str, object]:
    receipt = _require_keys(value, PARENT_RECEIPT_KEYS, "V7 diagnostic receipt")
    expected = {
        "schema": PARENT_RECEIPT_SCHEMA,
        "status": "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "matrix_mode": "FULL_504",
        "completed_rows": 504,
        "expected_rows": 504,
        "plan_sha256": PARENT_PLAN_SHA256,
        "source_snapshot_sha256": PARENT_SOURCE_SHA256,
        "selection_entropy_release": "PROHIBITED",
        "confirmation_materialization": "PROHIBITED",
        "formal_materialization": "PROHIBITED",
    }
    for key, expected_value in expected.items():
        if receipt[key] != expected_value or type(receipt[key]) is not type(expected_value):
            raise ContractError(f"V7 diagnostic receipt field drifted: {key}")
    _sha_text(receipt["aggregate_sha256"], "V7 diagnostic aggregate SHA-256")
    return receipt


PARENT_COMPLETED_KEYS = frozenset(
    {
        "status", "row_id", "attempt_directory", "plan_sha256", "row_sha256",
        "diagnostic_sha256", "trace_sha256", "terminal_receipt_sha256",
        "independent_metric_receipt_sha256",
    }
)


def _validate_parent_completed_identity(
    value: object, row_id: str
) -> dict[str, object]:
    completed = _require_keys(value, PARENT_COMPLETED_KEYS, f"V7 completed row {row_id}")
    if (
        completed["status"] != "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY"
        or completed["row_id"] != row_id
        or completed["plan_sha256"] != PARENT_PLAN_SHA256
    ):
        raise ContractError(f"V7 completed row identity drifted: {row_id}")
    relative = _relative_path(
        completed["attempt_directory"], f"V7 completed row {row_id} attempt"
    )
    parts = PurePosixPath(relative).parts
    if (
        len(parts) != 3
        or parts[0] != "attempts"
        or parts[1] != row_id
        or not parts[2].startswith("attempt-")
        or len(parts[2].removeprefix("attempt-")) != 4
        or not parts[2].removeprefix("attempt-").isdigit()
        or int(parts[2].removeprefix("attempt-")) <= 0
    ):
        raise ContractError(f"V7 completed row does not bind its exact row attempt directory: {row_id}")
    for field in PARENT_COMPLETED_KEYS - {
        "status", "row_id", "attempt_directory", "plan_sha256"
    }:
        _sha_text(completed[field], f"V7 completed row {row_id}.{field}")
    return completed


def validate_sealed_parent_diagnostic(root: Path, plan_path: Path) -> dict[str, object]:
    plan = _validate_parent_plan(root, plan_path)
    directory = plan_path.parent
    receipt_path = directory / "diagnostic.receipt.json"
    aggregate_path = directory / "diagnostic.aggregate.json"
    receipt = _validate_parent_receipt_payload(_load_json(receipt_path))
    if (
        not aggregate_path.is_file()
        or receipt["aggregate_sha256"] != _sha256(aggregate_path)
    ):
        raise ContractError("V7 diagnostic is not a sealed exact-504 PASS")
    aggregate = _require_keys(
        _load_json(aggregate_path),
        frozenset(
            {
                "schema", "status", "scientific_scope", "row_count", "summaries",
                "contrasts", "factorial_initialization_local_search",
                "seeded_arm_limitation", "later_phase_authorization", "matrix_mode",
                "plan_sha256",
            }
        ),
        "V7 diagnostic aggregate",
    )
    if (
        aggregate["schema"] != "v21e3r1_fourteen_arm_development_diagnostic_aggregate_v2"
        or aggregate["status"] != "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY"
        or aggregate["scientific_scope"]
        != "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE"
        or aggregate["row_count"] != 504
        or type(aggregate["row_count"]) is not int
        or aggregate["matrix_mode"] != "FULL_504"
        or aggregate["plan_sha256"] != PARENT_PLAN_SHA256
        or aggregate["later_phase_authorization"] != "PROHIBITED"
    ):
        raise ContractError("V7 diagnostic aggregate identity/cardinality drifted")
    expected_ids = {
        f"{case_id}__seed-{seed}__arm-{arm.lower()}"
        for case_id in v7_runner.EXPECTED_CASE_IDS
        for seed in SEEDS
        for arm in v7_runner.DIAGNOSTIC_ARMS
    }
    completed_dir = directory / "completed"
    expected_names = {f"{row_id}.json" for row_id in expected_ids}
    observed_names = {path.name for path in completed_dir.iterdir()}
    if observed_names != expected_names:
        raise ContractError("V7 completed-row seal coverage is not exactly 504")
    cases, _bounds, _directions, input_binding = v7_runner._load_inputs(root)
    if input_binding != plan["input_binding"]:
        raise ContractError("V7 sealed input binding drifted")
    case_by_id = {str(case["case_id"]): case for case in cases}
    for row_id in sorted(expected_ids):
        completed = _validate_parent_completed_identity(
            _load_json(completed_dir / f"{row_id}.json"), row_id
        )
        attempt = _contained(
            directory,
            str(completed["attempt_directory"]),
            f"V7 completed row {row_id} attempt",
        )
        for name, field in (
            ("row.json", "row_sha256"),
            ("diagnostic.json", "diagnostic_sha256"),
            ("trace.sqlite3", "trace_sha256"),
            ("terminal.receipt.json", "terminal_receipt_sha256"),
            ("independent.metric.json", "independent_metric_receipt_sha256"),
        ):
            artifact = attempt / name
            if not artifact.is_file() or _sha256(artifact) != completed[field]:
                raise ContractError(f"V7 completed-row artifact drifted: {row_id}/{name}")
        case_id, seed_arm = row_id.split("__seed-", 1)
        seed_text, arm_text = seed_arm.split("__arm-", 1)
        seed = _exact_int(int(seed_text), f"V7 row {row_id} seed")
        arm = next(
            (candidate for candidate in v7_runner.DIAGNOSTIC_ARMS if candidate.lower() == arm_text),
            None,
        )
        case = case_by_id.get(case_id)
        if arm is None or case is None:
            raise ContractError(f"V7 completed-row design identity drifted: {row_id}")
        case_path = v7_runner._case_path(root, case)
        row = _load_json(attempt / "row.json")
        expected_row = {
            "schema": "v21e3r1_exposed_development_diagnostic_row_v2",
            "status": "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY",
            "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
            "case_id": case_id,
            "family": case["family"],
            "size": case["size"],
            "seed": seed,
            "arm_id": arm,
            "charged_evaluation_budget": FULL_BUDGET,
            "checkpoint_period": FULL_CHECKPOINT_PERIOD,
            "case_artifact_sha256": _sha256(case_path),
            "source_snapshot_sha256": PARENT_SOURCE_SHA256,
            "plan_sha256": PARENT_PLAN_SHA256,
            "trace_database_path": "trace.sqlite3",
            "terminal_receipt_path": "terminal.receipt.json",
            "selection_entropy_release": "PROHIBITED",
            "confirmation_materialization": "PROHIBITED",
            "formal_materialization": "PROHIBITED",
        }
        for field, expected_value in expected_row.items():
            if row.get(field) != expected_value or type(row.get(field)) is not type(expected_value):
                raise ContractError(f"V7 completed-row payload drifted: {row_id}/{field}")
    return plan


SOURCE_BINDING_KEYS = frozenset(
    {
        "schema", "study_id", "candidate_id", "parent_development_source_sha256",
        "receipt_path", "receipt_sha256", "source_manifest_path",
        "source_manifest_sha256", "source_snapshot_sha256",
        "semantic_config_sha256", "study_metric_spec_sha256",
        "simultaneous_inference_spec_sha256", "factorial_inference_spec_path",
        "factorial_inference_spec_sha256",
    }
)


def _validate_source_binding(value: object) -> dict[str, object]:
    raw = _require_keys(value, SOURCE_BINDING_KEYS, "source binding")
    if raw["schema"] != "v21e3r1_successor_factorial_source_binding_v2":
        raise ContractError("source binding schema drifted")
    for field in ("receipt_path", "source_manifest_path", "factorial_inference_spec_path"):
        _relative_path(raw[field], f"source binding.{field}")
    for field in ("study_id", "candidate_id"):
        if type(raw[field]) is not str or not raw[field]:
            raise ContractError(f"source binding.{field} must be an exact nonempty string")
    for field in SOURCE_BINDING_KEYS - {
        "schema", "study_id", "candidate_id", "receipt_path",
        "source_manifest_path", "factorial_inference_spec_path",
    }:
        _sha_text(raw[field], f"source binding.{field}")
    if (
        raw["parent_development_source_sha256"] != PARENT_SOURCE_SHA256
        or raw["factorial_inference_spec_path"] != INFERENCE_SPEC_RELATIVE.as_posix()
        or raw["factorial_inference_spec_sha256"] != INFERENCE_SPEC_SHA256
    ):
        raise ContractError("source binding parent/inference lineage drifted")
    return raw


ROW_SPEC_KEYS = frozenset(
    {
        "ordinal", "row_id", "case_id", "family", "size", "seed", "arm_id",
        "post_initialization_search_policy", "mokp_novelty_generation_policy",
        "case_artifact_path", "case_artifact_sha256",
    }
)

ROW_KEYS = frozenset(
    {
        "schema", "status", "phase", "scientific_scope", *ROW_SPEC_KEYS,
        "charged_evaluation_budget", "checkpoint_period",
        "checkpoint_left_continuous_hv_auc",
        "exact_per_evaluation_left_continuous_hv_auc", "normalized_terminal_hv",
        "checkpoints", "attempt_count", "charged_evaluation_count", "cache_hit_count",
        "cache_hit_rate_per_attempt", "algorithm_config", "plan_sha256",
        "source_snapshot_sha256", "trace_database_path", "trace_database_sha256",
        "terminal_receipt_path", "terminal_receipt_sha256",
        "independent_metric_receipt_path", "independent_metric_receipt_sha256",
        "strict_trace_verification", "selection_cases_materialized",
        "confirmation_cases_materialized", "formal_cases_materialized",
        "implementation_independence", "scientific_independence",
        "selection_authorized", "confirmation_authorized", "formal_study_authorized",
        "scientific_claim_authorized", "ijoc_submission_status",
    }
)


def build_plan_payload(
    project_root: str | Path,
    parent_plan_path: str | Path,
    source_binding: Mapping[str, object],
    *,
    row_timeout_seconds: int = 1800,
) -> dict[str, object]:
    root = Path(project_root).resolve()
    parent_path = Path(parent_plan_path).resolve()
    parent = _validate_parent_plan(root, parent_path)
    inference, inference_sha = load_inference_spec(root)
    source = _validate_source_binding(dict(source_binding))
    timeout = _exact_int(row_timeout_seconds, "row_timeout_seconds", minimum=1)
    cases, _bounds, _directions, input_binding = v7_runner._load_inputs(root)
    if parent.get("input_binding") != input_binding:
        raise ContractError("V7 plan/current frozen input bindings disagree")
    if tuple(str(case.get("case_id")) for case in cases) != v7_runner.EXPECTED_CASE_IDS:
        raise ContractError("current input cases do not match the V7 plan")
    rows: list[dict[str, object]] = []
    ordinal = 0
    for case in cases:
        case_id = str(case["case_id"])
        family = str(case["family"])
        size = _exact_int(case.get("size"), "case.size", minimum=1)
        if family not in {"MOKP", "MOTSP"}:
            raise ContractError("factorial encountered a non-MOKP/MOTSP case")
        case_path = v7_runner._case_path(root, case)
        arms = MOKP_ARMS if family == "MOKP" else MOTSP_ARMS
        for seed in SEEDS:
            for arm_id, search_policy, novelty_policy in arms:
                ordinal += 1
                rows.append(
                    {
                        "ordinal": ordinal,
                        "row_id": f"{case_id}__seed-{seed}__arm-{arm_id.lower()}",
                        "case_id": case_id,
                        "family": family,
                        "size": size,
                        "seed": seed,
                        "arm_id": arm_id,
                        "post_initialization_search_policy": search_policy,
                        "mokp_novelty_generation_policy": novelty_policy,
                        "case_artifact_path": case_path.relative_to(root).as_posix(),
                        "case_artifact_sha256": _sha256(case_path),
                    }
                )
    if ordinal != FULL_ROW_COUNT:
        raise ContractError("derived successor factorial row count is not 108")
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "FROZEN_EXACT_108_SUCCESSOR_DEVELOPMENT_FACTORIAL",
        "phase": "development",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "parent_v7_diagnostic_plan_path": parent_path.relative_to(root).as_posix(),
        "parent_v7_diagnostic_plan_sha256": PARENT_PLAN_SHA256,
        "parent_v7_source_snapshot_sha256": PARENT_SOURCE_SHA256,
        "case_ids": list(parent["case_ids"]),
        "seeds": list(parent["seeds"]),
        "arms_by_family": {
            "MOKP": [arm[0] for arm in MOKP_ARMS],
            "MOTSP": [arm[0] for arm in MOTSP_ARMS],
        },
        "charged_evaluation_budget": parent["charged_evaluation_budget"],
        "checkpoint_period": parent["checkpoint_period"],
        "expected_rows": FULL_ROW_COUNT,
        "row_timeout_seconds": timeout,
        "input_binding": input_binding,
        "source_binding": source,
        "inference_spec_binding": {
            "path": INFERENCE_SPEC_RELATIVE.as_posix(),
            "sha256": inference_sha,
            "schema": inference["schema"],
            "method": inference["method"],
            "bootstrap_samples": inference["bootstrap_samples"],
            "bootstrap_seed": inference["bootstrap_seed"],
            "familywise_alpha": inference["familywise_alpha"],
        },
        "rows": rows,
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
    validate_plan_payload(plan)
    return plan


PLAN_KEYS = frozenset(
    {
        "schema", "status", "phase", "scientific_scope",
        "parent_v7_diagnostic_plan_path", "parent_v7_diagnostic_plan_sha256",
        "parent_v7_source_snapshot_sha256", "case_ids", "seeds",
        "arms_by_family", "charged_evaluation_budget", "checkpoint_period",
        "expected_rows", "row_timeout_seconds", "input_binding", "source_binding",
        "inference_spec_binding", "rows", "selection_entropy_release",
        "selection_cases_materialized", "confirmation_materialization",
        "formal_materialization", "implementation_independence",
        "scientific_independence", "selection_authorized", "confirmation_authorized",
        "formal_study_authorized", "scientific_claim_authorized",
        "ijoc_submission_status",
    }
)


def validate_plan_payload(value: object) -> dict[str, object]:
    plan = _require_keys(value, PLAN_KEYS, "factorial plan")
    expected_scalars = {
        "schema": PLAN_SCHEMA,
        "status": "FROZEN_EXACT_108_SUCCESSOR_DEVELOPMENT_FACTORIAL",
        "phase": "development",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "parent_v7_diagnostic_plan_sha256": PARENT_PLAN_SHA256,
        "parent_v7_source_snapshot_sha256": PARENT_SOURCE_SHA256,
        "charged_evaluation_budget": FULL_BUDGET,
        "checkpoint_period": FULL_CHECKPOINT_PERIOD,
        "expected_rows": FULL_ROW_COUNT,
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
    for key, expected in expected_scalars.items():
        if plan.get(key) != expected or type(plan.get(key)) is not type(expected):
            raise ContractError(f"factorial plan field drifted: {key}")
    if plan["case_ids"] != list(v7_runner.EXPECTED_CASE_IDS) or plan["seeds"] != list(SEEDS):
        raise ContractError("factorial plan case/seed boundary drifted")
    expected_input_binding = {
        "schema": "v21e3r1_exposed_development_input_binding_v1",
        "case_ids": list(v7_runner.EXPECTED_CASE_IDS),
        "manifest_sha256": dict(v7_runner.INPUT_MANIFEST_SHA256),
    }
    if plan["input_binding"] != expected_input_binding:
        raise ContractError("factorial plan frozen input binding drifted")
    if plan["arms_by_family"] != {
        "MOKP": [arm[0] for arm in MOKP_ARMS],
        "MOTSP": [arm[0] for arm in MOTSP_ARMS],
    }:
        raise ContractError("factorial plan arm boundary drifted")
    _validate_source_binding(plan["source_binding"])
    inference = _require_keys(
        plan["inference_spec_binding"],
        frozenset({"path", "sha256", "schema", "method", "bootstrap_samples", "bootstrap_seed", "familywise_alpha"}),
        "factorial plan inference binding",
    )
    if (
        inference["path"] != INFERENCE_SPEC_RELATIVE.as_posix()
        or inference["schema"] != INFERENCE_SCHEMA
        or inference["method"] != METHOD
        or inference["bootstrap_samples"] != 9999
        or inference["bootstrap_seed"] != 2026082301
        or inference["familywise_alpha"] != 0.05
        or inference["sha256"] != INFERENCE_SPEC_SHA256
    ):
        raise ContractError("factorial plan inference binding drifted")
    _sha_text(inference["sha256"], "factorial plan inference hash")
    rows = plan["rows"]
    if type(rows) is not list or len(rows) != FULL_ROW_COUNT:
        raise ContractError("factorial plan must contain exactly 108 rows")
    cases = list(v7_runner.EXPECTED_CASE_IDS)
    expected_keys: list[tuple[str, int, str, str, str]] = []
    for case_id in cases:
        family = "MOKP" if "-mokp-" in case_id else "MOTSP"
        arms = MOKP_ARMS if family == "MOKP" else MOTSP_ARMS
        for seed in SEEDS:
            for arm_id, search, novelty in arms:
                expected_keys.append((case_id, seed, arm_id, search, novelty))
    observed_keys: list[tuple[str, int, str, str, str]] = []
    row_ids: set[str] = set()
    for index, row_value in enumerate(rows):
        row = _require_keys(row_value, ROW_SPEC_KEYS, f"factorial plan rows[{index}]")
        if row["ordinal"] != index + 1 or type(row["ordinal"]) is not int:
            raise ContractError("factorial row ordinals must be exact contiguous integers")
        if type(row["row_id"]) is not str or row["row_id"] in row_ids:
            raise ContractError("factorial row IDs must be unique exact strings")
        row_ids.add(row["row_id"])
        expected_row_id = (
            f"{row['case_id']}__seed-{row['seed']}__arm-{str(row['arm_id']).lower()}"
        )
        if row["row_id"] != expected_row_id:
            raise ContractError("factorial row ID disagrees with case/seed/arm")
        if row["family"] not in {"MOKP", "MOTSP"}:
            raise ContractError("factorial row family drifted")
        expected_family = "MOKP" if "-mokp-" in str(row["case_id"]) else "MOTSP"
        matching_sizes = [
            size for size in (100, 200, 500) if f"-n{size}-" in str(row["case_id"])
        ]
        if (
            row["family"] != expected_family
            or len(matching_sizes) != 1
            or row["size"] != matching_sizes[0]
            or type(row["size"]) is not int
        ):
            raise ContractError("factorial row case/family/size boundary drifted")
        if row["family"] == "MOTSP" and row["mokp_novelty_generation_policy"] != LEGACY_NOVELTY:
            raise ContractError("MOTSP rows cannot activate the MOKP novelty policy")
        _relative_path(row["case_artifact_path"], f"rows[{index}].case_artifact_path")
        _sha_text(row["case_artifact_sha256"], f"rows[{index}].case_artifact_sha256")
        observed_keys.append(
            (row["case_id"], row["seed"], row["arm_id"], row["post_initialization_search_policy"], row["mokp_novelty_generation_policy"])
        )
    if observed_keys != expected_keys:
        raise ContractError("factorial row case/seed/policy design drifted")
    return plan


def _preflight_policy_fields() -> None:
    available = {field.name for field in fields(V21E3HybridConfig)}
    missing = {
        "post_initialization_search_policy",
        "mokp_novelty_generation_policy",
    } - available
    if missing:
        raise ContractError(
            "successor policy implementation is not present in V21E3HybridConfig: "
            + ",".join(sorted(missing))
        )


def _expected_semantic_config(
    row_spec: Mapping[str, object],
    directions: Sequence[Sequence[float]],
) -> dict[str, object]:
    _preflight_policy_fields()
    normalized_directions = tuple(
        tuple(float(item) for item in direction) for direction in directions
    )
    config = hybrid_diagnostic_config(
        arm_id="C0_STANDARD",
        reference_directions=normalized_directions,
        charged_evaluations=FULL_BUDGET,
        checkpoint_period=FULL_CHECKPOINT_PERIOD,
        seed=_exact_int(row_spec["seed"], "factorial row seed"),
        family=str(row_spec["family"]),
        trace_database=None,
        terminal_receipt=None,
    )
    config = replace(
        config,
        receipt_database_path="trace.sqlite3",
        capture_trace=False,
        case_artifact_sha256=str(row_spec["case_artifact_sha256"]),
        source_snapshot_sha256=None,
        development_diagnostic_id=(
            "V21E3R1_SUCCESSOR_FACTORIAL_" + str(row_spec["arm_id"])
        ),
        post_initialization_search_policy=str(
            row_spec["post_initialization_search_policy"]
        ),
        mokp_novelty_generation_policy=str(
            row_spec["mokp_novelty_generation_policy"]
        ),
    )
    semantic = config.semantic_payload()
    if (
        semantic.get("phase") != "development"
        or semantic.get("post_initialization_search_policy")
        != row_spec["post_initialization_search_policy"]
        or semantic.get("mokp_novelty_generation_policy")
        != row_spec["mokp_novelty_generation_policy"]
        or semantic.get("development_diagnostic_id")
        != "V21E3R1_SUCCESSOR_FACTORIAL_" + str(row_spec["arm_id"])
    ):
        raise ContractError("expected successor factorial semantic config drifted")
    return semantic


def _validate_row_payload(
    value: object,
    row_spec: Mapping[str, object],
    plan_sha: str,
    source_binding: Mapping[str, object],
    expected_algorithm_config: Mapping[str, object],
) -> dict[str, object]:
    row_id = str(row_spec["row_id"])
    row = _require_keys(value, ROW_KEYS, f"factorial row {row_id}")
    expected_scalars = {
        "schema": ROW_SCHEMA,
        "status": "PASS_SUCCESSOR_DEVELOPMENT_FACTORIAL_ROW_ENGINEERING_ONLY",
        "phase": "development",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "charged_evaluation_budget": FULL_BUDGET,
        "checkpoint_period": FULL_CHECKPOINT_PERIOD,
        "plan_sha256": plan_sha,
        "source_snapshot_sha256": source_binding["source_snapshot_sha256"],
        "trace_database_path": "trace.sqlite3",
        "terminal_receipt_path": "terminal.receipt.json",
        "independent_metric_receipt_path": "independent.metric.json",
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
    for key, expected_value in expected_scalars.items():
        if row[key] != expected_value or type(row[key]) is not type(expected_value):
            raise ContractError(f"factorial row field drifted: {row_id}/{key}")
    for field in ROW_SPEC_KEYS:
        if row[field] != row_spec[field] or type(row[field]) is not type(row_spec[field]):
            raise ContractError(f"factorial row/spec drifted: {row_id}/{field}")
    if type(row["algorithm_config"]) is not dict:
        raise ContractError(f"factorial row algorithm_config is not an exact object: {row_id}")
    try:
        algorithm_config_matches = _canonical_bytes(row["algorithm_config"]) == _canonical_bytes(
            dict(expected_algorithm_config)
        )
    except (TypeError, ValueError) as error:
        raise ContractError(
            f"factorial row algorithm_config is not canonical finite JSON: {row_id}"
        ) from error
    if not algorithm_config_matches:
        raise ContractError(f"factorial row algorithm_config drifted: {row_id}")
    attempts = _exact_int(row["attempt_count"], f"{row_id}.attempt_count", minimum=1)
    charges = _exact_int(
        row["charged_evaluation_count"], f"{row_id}.charged_evaluation_count", minimum=1
    )
    cache_hits = _exact_int(row["cache_hit_count"], f"{row_id}.cache_hit_count")
    if charges != FULL_BUDGET or cache_hits > attempts:
        raise ContractError(f"factorial row accounting drifted: {row_id}")
    cache_rate = _exact_number(row["cache_hit_rate_per_attempt"], f"{row_id}.cache rate")
    if not math.isclose(cache_rate, cache_hits / attempts, rel_tol=0.0, abs_tol=1e-15):
        raise ContractError(f"factorial row cache rate drifted: {row_id}")
    for field in (
        "checkpoint_left_continuous_hv_auc",
        "exact_per_evaluation_left_continuous_hv_auc",
        "normalized_terminal_hv",
        "cache_hit_rate_per_attempt",
    ):
        number = _exact_number(row[field], f"{row_id}.{field}")
        if not 0.0 <= number <= 1.0:
            raise ContractError(f"factorial row metric is outside [0,1]: {row_id}/{field}")
    checkpoints = row["checkpoints"]
    if type(checkpoints) is not list or len(checkpoints) != FULL_BUDGET // FULL_CHECKPOINT_PERIOD:
        raise ContractError(f"factorial row checkpoint coverage drifted: {row_id}")
    checkpoint_area = 0.0
    previous_evaluation = 0
    previous_hv = 0.0
    for index, checkpoint in enumerate(checkpoints, start=1):
        witness = _require_keys(
            checkpoint,
            frozenset({"evaluation", "normalized_hv", "archive_size"}),
            f"{row_id}.checkpoints[{index - 1}] witness",
        )
        evaluation = _exact_int(
            witness["evaluation"],
            f"{row_id}.checkpoints[{index - 1}].evaluation",
            minimum=1,
        )
        expected_evaluation = index * FULL_CHECKPOINT_PERIOD
        if evaluation != expected_evaluation:
            raise ContractError(
                f"factorial row checkpoint evaluation drifted: {row_id}"
            )
        normalized_hv = _exact_number(
            witness["normalized_hv"],
            f"{row_id}.checkpoints[{index - 1}].normalized_hv",
        )
        if not 0.0 <= normalized_hv <= 1.0:
            raise ContractError(
                f"factorial row checkpoint normalized_hv is outside [0,1]: {row_id}"
            )
        _exact_int(
            witness["archive_size"],
            f"{row_id}.checkpoints[{index - 1}].archive_size",
            minimum=1,
        )
        checkpoint_area += previous_hv * (evaluation - previous_evaluation)
        previous_evaluation = evaluation
        previous_hv = normalized_hv
    checkpoint_auc = _exact_number(
        row["checkpoint_left_continuous_hv_auc"],
        f"{row_id}.checkpoint_left_continuous_hv_auc",
    )
    if checkpoint_area / FULL_BUDGET != checkpoint_auc:
        raise ContractError(f"factorial row checkpoint AUC witness drifted: {row_id}")
    terminal_hv = _exact_number(
        row["normalized_terminal_hv"], f"{row_id}.normalized_terminal_hv"
    )
    if previous_hv != terminal_hv:
        raise ContractError(f"factorial row terminal checkpoint witness drifted: {row_id}")
    for field in (
        "trace_database_sha256", "terminal_receipt_sha256",
        "independent_metric_receipt_sha256",
    ):
        _sha_text(row[field], f"{row_id}.{field}")
    if type(row["strict_trace_verification"]) is not dict:
        raise ContractError(f"factorial row trace verification is not an exact object: {row_id}")
    return row


def _seal_sqlite(path: Path) -> None:
    v7_runner._seal_sqlite(path)


def _derive_worker_contract(spec_path: str | Path) -> dict[str, object]:
    spec_file = Path(spec_path).resolve()
    project = Path.cwd().resolve()
    spec = _load_worker_spec(spec_file)
    try:
        spec_file.relative_to(project)
    except ValueError as error:
        raise ContractError("worker spec escapes the project root") from error
    if spec_file.name != "worker.spec.json":
        raise ContractError("worker spec must use the frozen worker.spec.json name")
    attempt = spec_file.parent
    attempt_suffix = attempt.name.removeprefix("attempt-")
    if (
        attempt.parent.name != spec["row_id"]
        or attempt.parent.parent.name != "attempts"
        or not attempt.name.startswith("attempt-")
        or len(attempt_suffix) != 4
        or not attempt_suffix.isdigit()
        or int(attempt_suffix) <= 0
    ):
        raise ContractError("worker spec is outside its exact row attempt directory")
    output = attempt.parents[2]
    plan_path = output / "factorial.plan.json"
    plan = validate_plan_payload(_load_json(plan_path))
    if plan_path.read_bytes() != _canonical_bytes(plan, newline=True):
        raise ContractError("worker factorial plan is not canonical JSON plus LF")
    plan_sha = _sha256(plan_path)
    if plan_sha != spec["plan_sha256"]:
        raise ContractError("worker factorial plan hash drifted")
    source_receipt = _contained(
        project,
        str(plan["source_binding"]["receipt_path"]),
        "worker source receipt",
    )
    current_source = validate_successor_source_freeze(project, source_receipt)
    if current_source != plan["source_binding"]:
        raise ContractError("worker successor source binding drifted")
    parent_path = _contained(
        project,
        str(plan["parent_v7_diagnostic_plan_path"]),
        "worker parent V7 plan",
    )
    expected_plan = build_plan_payload(
        project,
        parent_path,
        current_source,
        row_timeout_seconds=_exact_int(
            plan["row_timeout_seconds"], "worker row timeout", minimum=1
        ),
    )
    if plan != expected_plan:
        raise ContractError("worker factorial plan disagrees with re-derived frozen design")
    ordinal = _exact_int(spec["ordinal"], "worker ordinal", minimum=1)
    if ordinal > len(plan["rows"]):
        raise ContractError("worker ordinal is outside the frozen factorial plan")
    row_spec = plan["rows"][ordinal - 1]
    if row_spec["row_id"] != spec["row_id"] or row_spec["ordinal"] != ordinal:
        raise ContractError("worker row identity disagrees with its frozen ordinal")
    cases, bounds, directions, input_binding = v7_runner._load_inputs(project)
    if input_binding != plan["input_binding"]:
        raise ContractError("worker frozen input binding drifted")
    case = next(
        (item for item in cases if item.get("case_id") == row_spec["case_id"]),
        None,
    )
    if case is None:
        raise ContractError("worker case is absent from frozen inputs")
    case_path = v7_runner._case_path(project, case)
    if (
        case_path.relative_to(project).as_posix() != row_spec["case_artifact_path"]
        or _sha256(case_path) != row_spec["case_artifact_sha256"]
    ):
        raise ContractError("worker case artifact binding drifted")
    lower, upper = bounds[str(row_spec["case_id"])]
    return {
        "spec": spec,
        "spec_file": spec_file,
        "project": project,
        "output": output,
        "attempt": attempt,
        "plan": plan,
        "plan_path": plan_path,
        "plan_sha256": plan_sha,
        "source_receipt": source_receipt,
        "source_binding": current_source,
        "row_spec": row_spec,
        "case_path": case_path,
        "lower": tuple(float(item) for item in lower),
        "upper": tuple(float(item) for item in upper),
        "directions": tuple(tuple(float(item) for item in row) for row in directions),
    }


def _worker_run(spec_path: str | Path) -> dict[str, object]:
    contract = _derive_worker_contract(spec_path)
    spec = contract["spec"]
    attempt = contract["attempt"]
    project = contract["project"]
    plan = contract["plan"]
    source_receipt = contract["source_receipt"]
    row_spec = contract["row_spec"]
    case_path = contract["case_path"]
    trace = attempt / "trace.sqlite3"
    terminal = attempt / "terminal.receipt.json"
    problem = v7_runner.load_v21e3_development_problem(case_path)
    lower = contract["lower"]
    upper = contract["upper"]
    directions = contract["directions"]
    seed = _exact_int(row_spec["seed"], "worker seed")
    family = str(row_spec["family"])
    config = hybrid_diagnostic_config(
        arm_id="C0_STANDARD",
        reference_directions=directions,
        charged_evaluations=FULL_BUDGET,
        checkpoint_period=FULL_CHECKPOINT_PERIOD,
        seed=seed,
        family=family,
        trace_database=str(trace),
        terminal_receipt=str(terminal),
    )
    config = replace(
        config,
        receipt_database_path="trace.sqlite3",
        capture_trace=False,
        case_artifact_sha256=str(row_spec["case_artifact_sha256"]),
        source_snapshot_sha256=str(plan["source_binding"]["source_snapshot_sha256"]),
        development_diagnostic_id="V21E3R1_SUCCESSOR_FACTORIAL_" + str(row_spec["arm_id"]),
        post_initialization_search_policy=str(row_spec["post_initialization_search_policy"]),
        mokp_novelty_generation_policy=str(row_spec["mokp_novelty_generation_policy"]),
    )
    semantic = config.semantic_payload()
    if semantic != _expected_semantic_config(row_spec, directions):
        raise ContractError("executed semantic policy payload drifted")
    optimizer = V21E3TypedHybridParetoSearch(problem, config)
    run = optimizer.run()
    del optimizer
    checkpoint_auc, terminal_hv, checkpoints = normalized_left_continuous_auc(
        run.optimization_result.diagnostics,
        budget=FULL_BUDGET,
        checkpoint_period=FULL_CHECKPOINT_PERIOD,
        lower=lower,
        upper=upper,
    )
    _seal_sqlite(trace)
    context = v7_runner._load_trace_context(trace)
    v7_runner._assert_context(
        context,
        algorithm_config=semantic,
        case_sha256=str(row_spec["case_artifact_sha256"]),
        source_sha256=str(plan["source_binding"]["source_snapshot_sha256"]),
        seed=seed,
        budget=FULL_BUDGET,
    )
    terminal_sha = _sha256(terminal)
    verification = verify_v21e3_trace_database(
        trace,
        problem,
        expected_run_context=context,
        detached_terminal_receipt_path=terminal,
        expected_detached_terminal_receipt_sha256=terminal_sha,
        expected_charged_evaluations=FULL_BUDGET,
    )
    independent_path = attempt / "independent.metric.json"
    independent = _successor_independent_metric_replay(
        project_root=project,
        trace=trace,
        lower=lower,
        upper=upper,
        budget=FULL_BUDGET,
        output=independent_path,
    )
    exact_auc = _exact_number(independent.get("exact_left_continuous_hv_auc"), "independent exact EAUC")
    if not math.isclose(float(independent["terminal_hv"]), terminal_hv, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError("independent/project terminal hypervolumes disagree")
    terminal_payload = _load_json(terminal)
    attempts = _exact_int(terminal_payload.get("attempt_count"), "terminal attempt_count", minimum=1)
    charges = _exact_int(terminal_payload.get("charged_evaluation_count"), "terminal charged_evaluation_count", minimum=1)
    cache_hits = _exact_int(terminal_payload.get("cache_hit_count"), "terminal cache_hit_count")
    if charges != FULL_BUDGET or cache_hits > attempts:
        raise ContractError("terminal accounting fails the frozen factorial budget")
    # Re-check the live frozen source after execution before publishing the row.
    if validate_successor_source_freeze(project, source_receipt) != plan["source_binding"]:
        raise ContractError("successor source changed during factorial row execution")
    row = {
        "schema": ROW_SCHEMA,
        "status": "PASS_SUCCESSOR_DEVELOPMENT_FACTORIAL_ROW_ENGINEERING_ONLY",
        "phase": "development",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        **dict(row_spec),
        "charged_evaluation_budget": FULL_BUDGET,
        "checkpoint_period": FULL_CHECKPOINT_PERIOD,
        "checkpoint_left_continuous_hv_auc": checkpoint_auc,
        "exact_per_evaluation_left_continuous_hv_auc": exact_auc,
        "normalized_terminal_hv": terminal_hv,
        "checkpoints": list(checkpoints),
        "attempt_count": attempts,
        "charged_evaluation_count": charges,
        "cache_hit_count": cache_hits,
        "cache_hit_rate_per_attempt": cache_hits / attempts,
        "algorithm_config": semantic,
        "plan_sha256": contract["plan_sha256"],
        "source_snapshot_sha256": plan["source_binding"]["source_snapshot_sha256"],
        "trace_database_path": "trace.sqlite3",
        "trace_database_sha256": _sha256(trace),
        "terminal_receipt_path": "terminal.receipt.json",
        "terminal_receipt_sha256": terminal_sha,
        "independent_metric_receipt_path": "independent.metric.json",
        "independent_metric_receipt_sha256": _sha256(independent_path),
        "strict_trace_verification": verification,
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
    _validate_row_payload(
        row,
        row_spec,
        str(contract["plan_sha256"]),
        plan["source_binding"],
        semantic,
    )
    _exclusive_json(attempt / "row.json", row)
    return {
        "status": "PASS_SUCCESSOR_DEVELOPMENT_FACTORIAL_ROW_ENGINEERING_ONLY",
        "ordinal": row_spec["ordinal"],
        "row_id": row_spec["row_id"],
        "plan_sha256": contract["plan_sha256"],
        "row_spec_sha256": _payload_sha256(dict(row_spec)),
        "worker_spec_sha256": _sha256(contract["spec_file"]),
        "row_sha256": _sha256(attempt / "row.json"),
        "trace_sha256": _sha256(trace),
        "terminal_receipt_sha256": terminal_sha,
        "independent_metric_receipt_sha256": _sha256(independent_path),
        "source_snapshot_sha256": plan["source_binding"]["source_snapshot_sha256"],
        "source_receipt_sha256": plan["source_binding"]["receipt_sha256"],
    }


def _validate_failure_receipt(path: Path) -> dict[str, object]:
    payload = _load_json(path)
    status = payload.get("status")
    if status == "FAIL_ROW_TIMEOUT":
        expected_keys = frozenset(
            {
                "schema", "status", "timeout_seconds", "stdout_tail", "stderr_tail",
                "selection_authorized", "formal_study_authorized",
            }
        )
        _exact_int(payload.get("timeout_seconds"), "row failure receipt timeout", minimum=1)
    elif status == "FAIL_ROW_PROCESS":
        expected_keys = frozenset(
            {
                "schema", "status", "returncode", "stdout_tail", "stderr_tail",
                "selection_authorized", "formal_study_authorized",
            }
        )
        returncode = payload.get("returncode")
        if type(returncode) is not int or returncode == 0:
            raise ContractError(
                "row failure receipt returncode must be an exact nonzero integer"
            )
    else:
        raise ContractError("row failure receipt status drifted")
    payload = _require_keys(payload, expected_keys, "row failure receipt")
    if (
        payload["schema"] != "v21e3r1_successor_factorial_row_failure_v1"
        or type(payload["stdout_tail"]) is not str
        or type(payload["stderr_tail"]) is not str
        or payload["selection_authorized"] is not False
        or payload["formal_study_authorized"] is not False
        or path.read_bytes() != _canonical_bytes(payload, newline=True)
    ):
        raise ContractError("row failure receipt boundary drifted")
    return payload


def _next_attempt(output: Path, row_id: str) -> Path:
    root = output / "attempts" / row_id
    root.mkdir(parents=True, exist_ok=True)
    observed: list[int] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        suffix = path.name.removeprefix("attempt-")
        if (
            not path.is_dir()
            or not path.name.startswith("attempt-")
            or len(suffix) != 4
            or not suffix.isdigit()
            or int(suffix) <= 0
        ):
            raise ContractError(f"attempt directory inventory drifted: {row_id}/{path.name}")
        failure = path / "failure.receipt.json"
        worker_result = path / "worker.result.json"
        if not failure.is_file() or worker_result.exists():
            raise ContractError(
                f"active or unterminated prior attempt blocks retry: {row_id}/{path.name}"
            )
        _validate_failure_receipt(failure)
        observed.append(int(suffix))
    if observed != list(range(1, len(observed) + 1)):
        raise ContractError(f"attempt directory sequence drifted: {row_id}")
    attempt = root / f"attempt-{max(observed, default=0) + 1:04d}"
    try:
        attempt.mkdir()
    except FileExistsError as error:
        raise ContractError(
            f"concurrent attempt creation detected: {row_id}/{attempt.name}"
        ) from error
    return attempt


def _run_child(spec_path: Path, *, project: Path, timeout: int) -> None:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-spec", str(spec_path)]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    try:
        completed = subprocess.run(
            command, cwd=project, env=environment, text=True, capture_output=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as error:
        _exclusive_json(
            spec_path.parent / "failure.receipt.json",
            {
                "schema": "v21e3r1_successor_factorial_row_failure_v1",
                "status": "FAIL_ROW_TIMEOUT",
                "timeout_seconds": timeout,
                "stdout_tail": (error.stdout or "")[-4000:],
                "stderr_tail": (error.stderr or "")[-4000:],
                "selection_authorized": False,
                "formal_study_authorized": False,
            },
        )
        raise ContractError(f"factorial row timed out: {spec_path}") from error
    if completed.returncode != 0:
        _exclusive_json(
            spec_path.parent / "failure.receipt.json",
            {
                "schema": "v21e3r1_successor_factorial_row_failure_v1",
                "status": "FAIL_ROW_PROCESS",
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
                "selection_authorized": False,
                "formal_study_authorized": False,
            },
        )
        raise ContractError(f"factorial row process failed: {spec_path}")


COMPLETED_KEYS = frozenset(
    {
        "schema", "status", "ordinal", "row_id", "attempt_directory",
        "plan_sha256", "row_spec_sha256", "worker_spec_sha256",
        "row_sha256", "trace_sha256", "terminal_receipt_sha256",
        "independent_metric_receipt_sha256", "source_snapshot_sha256",
        "source_receipt_sha256",
    }
)


def _validate_completed_identity(
    value: object,
    row_spec: Mapping[str, object],
    plan_sha: str,
    source_binding: Mapping[str, object],
) -> dict[str, object]:
    row_id = str(row_spec["row_id"])
    payload = _require_keys(value, COMPLETED_KEYS, f"completed row {row_id}")
    expected = {
        "schema": COMPLETED_SCHEMA,
        "status": "PASS_SUCCESSOR_DEVELOPMENT_FACTORIAL_ROW_ENGINEERING_ONLY",
        "ordinal": row_spec["ordinal"],
        "row_id": row_id,
        "plan_sha256": plan_sha,
        "row_spec_sha256": _payload_sha256(dict(row_spec)),
        "source_snapshot_sha256": source_binding["source_snapshot_sha256"],
        "source_receipt_sha256": source_binding["receipt_sha256"],
    }
    for key, expected_value in expected.items():
        if payload[key] != expected_value or type(payload[key]) is not type(expected_value):
            raise ContractError(f"completed row identity drifted: {row_id}/{key}")
    relative = _relative_path(payload["attempt_directory"], "attempt directory")
    parts = PurePosixPath(relative).parts
    if (
        len(parts) != 3
        or parts[0] != "attempts"
        or parts[1] != row_id
        or not parts[2].startswith("attempt-")
        or len(parts[2].removeprefix("attempt-")) != 4
        or not parts[2].removeprefix("attempt-").isdigit()
        or int(parts[2].removeprefix("attempt-")) <= 0
    ):
        raise ContractError(f"completed row does not bind its exact row attempt directory: {row_id}")
    for field in COMPLETED_KEYS - {
        "schema", "status", "ordinal", "row_id", "attempt_directory"
    }:
        _sha_text(payload[field], f"completed row {row_id}.{field}")
    return payload


WORKER_RESULT_KEYS = COMPLETED_KEYS - {"schema", "attempt_directory"}
SUCCESSFUL_ATTEMPT_FILES = frozenset(
    {
        "worker.spec.json", "worker.result.json", "row.json", "trace.sqlite3",
        "terminal.receipt.json", "independent.metric.json",
    }
)


def _completed_payload(
    output: Path,
    row_spec: Mapping[str, object],
    plan_sha: str,
    source_binding: Mapping[str, object],
    expected_algorithm_config: Mapping[str, object],
) -> dict[str, object] | None:
    row_id = str(row_spec["row_id"])
    path = output / "completed" / f"{row_id}.json"
    if not path.is_file():
        return None
    payload = _validate_completed_identity(
        _load_json(path), row_spec, plan_sha, source_binding
    )
    if path.read_bytes() != _canonical_bytes(payload, newline=True):
        raise ContractError(f"completed row seal is not canonical JSON plus LF: {row_id}")
    relative = str(payload["attempt_directory"])
    attempt = _contained(output, relative, "attempt directory")
    observed_attempt_files = {item.name for item in attempt.iterdir()}
    if observed_attempt_files != SUCCESSFUL_ATTEMPT_FILES:
        raise ContractError(f"completed row attempt directory set drifted: {row_id}")
    worker_spec = _load_worker_spec(attempt / "worker.spec.json")
    if (
        worker_spec["row_id"] != row_id
        or worker_spec["ordinal"] != row_spec["ordinal"]
        or worker_spec["plan_sha256"] != plan_sha
        or _sha256(attempt / "worker.spec.json") != payload["worker_spec_sha256"]
    ):
        raise ContractError(f"completed row worker spec drifted: {row_id}")
    for name, field in (
        ("row.json", "row_sha256"), ("trace.sqlite3", "trace_sha256"),
        ("terminal.receipt.json", "terminal_receipt_sha256"),
        ("independent.metric.json", "independent_metric_receipt_sha256"),
    ):
        artifact = attempt / name
        if not artifact.is_file() or _sha256(artifact) != payload[field]:
            raise ContractError(f"completed row artifact drifted: {row_id}/{name}")
    row = _validate_row_payload(
        _load_json(attempt / "row.json"),
        row_spec,
        plan_sha,
        source_binding,
        expected_algorithm_config,
    )
    if (attempt / "row.json").read_bytes() != _canonical_bytes(row, newline=True):
        raise ContractError(f"factorial row is not canonical JSON plus LF: {row_id}")
    worker_result = _require_keys(
        _load_json(attempt / "worker.result.json"),
        WORKER_RESULT_KEYS,
        f"worker result {row_id}",
    )
    if (attempt / "worker.result.json").read_bytes() != _canonical_bytes(
        worker_result, newline=True
    ):
        raise ContractError(f"worker result is not canonical JSON plus LF: {row_id}")
    for field in WORKER_RESULT_KEYS:
        if worker_result[field] != payload[field] or type(worker_result[field]) is not type(payload[field]):
            raise ContractError(f"completed row/worker result drifted: {row_id}/{field}")
    return payload


FACTORIAL_RECEIPT_KEYS = frozenset(
    {
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
        "scientific_claim_authorized", "ijoc_submission_status",
        "receipt_payload_sha256",
    }
)


def _factorial_receipt_core(
    plan: Mapping[str, object], plan_sha: str, aggregate_sha: str
) -> dict[str, object]:
    source = _validate_source_binding(plan["source_binding"])
    _sha_text(plan_sha, "factorial plan SHA-256")
    _sha_text(aggregate_sha, "factorial aggregate SHA-256")
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS_EXACT_108_SUCCESSOR_DEVELOPMENT_FACTORIAL_ENGINEERING_ONLY",
        "phase": "development",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "completed_rows": FULL_ROW_COUNT,
        "expected_rows": FULL_ROW_COUNT,
        "plan_sha256": plan_sha,
        "aggregate_sha256": aggregate_sha,
        "parent_v7_diagnostic_plan_sha256": PARENT_PLAN_SHA256,
        "parent_v7_source_snapshot_sha256": PARENT_SOURCE_SHA256,
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
        "inference_spec_sha256": INFERENCE_SPEC_SHA256,
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


def _validate_factorial_receipt_payload(
    value: object,
    plan: Mapping[str, object],
    plan_sha: str,
    aggregate_sha: str,
) -> dict[str, object]:
    receipt = _require_keys(value, FACTORIAL_RECEIPT_KEYS, "factorial receipt")
    core = dict(receipt)
    payload_sha = _sha_text(
        core.pop("receipt_payload_sha256"), "factorial receipt payload SHA-256"
    )
    if _payload_sha256(core) != payload_sha:
        raise ContractError("factorial receipt payload hash drifted")
    expected = _factorial_receipt_core(plan, plan_sha, aggregate_sha)
    for key, expected_value in expected.items():
        if receipt[key] != expected_value or type(receipt[key]) is not type(expected_value):
            raise ContractError(f"factorial receipt field drifted: {key}")
    return receipt


FACTORIAL_SUMMARY_KEYS = frozenset(
    {
        "ordinal", "row_id", "case_id", "family", "seed", "arm_id",
        "exact_per_evaluation_left_continuous_hv_auc",
        "cache_hit_rate_per_attempt", "row_sha256", "trace_sha256",
        "terminal_receipt_sha256", "independent_metric_receipt_sha256",
    }
)
FACTORIAL_AGGREGATE_KEYS = frozenset(
    {
        "schema", "status", "phase", "scientific_scope", "plan_sha256",
        "row_count", "rows", "development_promotion_evaluated",
        "selection_authorized", "confirmation_authorized", "formal_study_authorized",
        "scientific_claim_authorized", "ijoc_submission_status",
    }
)


def _factorial_summary(
    row: Mapping[str, object], completed: Mapping[str, object]
) -> dict[str, object]:
    return {
        "ordinal": row["ordinal"],
        "row_id": row["row_id"],
        "case_id": row["case_id"],
        "family": row["family"],
        "seed": row["seed"],
        "arm_id": row["arm_id"],
        "exact_per_evaluation_left_continuous_hv_auc": row[
            "exact_per_evaluation_left_continuous_hv_auc"
        ],
        "cache_hit_rate_per_attempt": row["cache_hit_rate_per_attempt"],
        "row_sha256": completed["row_sha256"],
        "trace_sha256": completed["trace_sha256"],
        "terminal_receipt_sha256": completed["terminal_receipt_sha256"],
        "independent_metric_receipt_sha256": completed[
            "independent_metric_receipt_sha256"
        ],
    }


def _factorial_aggregate_payload(
    plan_sha: str, summaries: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    rows = [dict(summary) for summary in summaries]
    for index, summary in enumerate(rows):
        _require_keys(summary, FACTORIAL_SUMMARY_KEYS, f"factorial summary[{index}]")
    return {
        "schema": AGGREGATE_SCHEMA,
        "status": "PASS_EXACT_108_SUCCESSOR_DEVELOPMENT_FACTORIAL_ENGINEERING_ONLY",
        "phase": "development",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "plan_sha256": plan_sha,
        "row_count": len(rows),
        "rows": rows,
        "development_promotion_evaluated": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_study_authorized": False,
        "scientific_claim_authorized": False,
        "ijoc_submission_status": "IJOC_HOLD",
    }


def _validate_factorial_aggregate_payload(
    value: object,
    plan_sha: str,
    expected_summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    aggregate = _require_keys(value, FACTORIAL_AGGREGATE_KEYS, "factorial aggregate")
    expected = _factorial_aggregate_payload(plan_sha, expected_summaries)
    if aggregate != expected:
        raise ContractError("factorial aggregate disagrees with exact completed rows")
    return aggregate


def _verify_completed_receipt(
    root: Path, output: Path, plan: Mapping[str, object], plan_sha: str
) -> dict[str, object]:
    plan_path = output / "factorial.plan.json"
    receipt_path = output / "factorial.receipt.json"
    aggregate_path = output / "factorial.aggregate.json"
    if (
        not plan_path.is_file()
        or _sha256(plan_path) != plan_sha
        or plan_path.read_bytes() != _canonical_bytes(plan, newline=True)
    ):
        raise ContractError("completed factorial plan fails strict resume validation")
    parent_path = _contained(
        root,
        str(plan["parent_v7_diagnostic_plan_path"]),
        "completed factorial parent V7 plan",
    )
    validate_sealed_parent_diagnostic(root, parent_path)
    source_receipt = _contained(
        root,
        str(plan["source_binding"]["receipt_path"]),
        "completed factorial successor source receipt",
    )
    current_source = validate_successor_source_freeze(root, source_receipt)
    if current_source != plan["source_binding"]:
        raise ContractError("completed factorial source binding drifted")
    expected_plan = build_plan_payload(
        root,
        parent_path,
        current_source,
        row_timeout_seconds=_exact_int(
            plan["row_timeout_seconds"], "completed factorial row timeout", minimum=1
        ),
    )
    if plan != expected_plan:
        raise ContractError("completed factorial plan disagrees with frozen design")
    _inference, inference_sha = load_inference_spec(root)
    if inference_sha != plan["inference_spec_binding"]["sha256"]:
        raise ContractError("completed factorial inference binding drifted")
    if not aggregate_path.is_file() or not receipt_path.is_file():
        raise ContractError("completed factorial final artifacts are incomplete")
    _cases, _bounds, directions, input_binding = v7_runner._load_inputs(root)
    if input_binding != plan["input_binding"]:
        raise ContractError("completed factorial input binding drifted")
    expected_completed_names = {
        f"{row['row_id']}.json" for row in plan["rows"]
    }
    completed_dir = output / "completed"
    if (
        not completed_dir.is_dir()
        or {path.name for path in completed_dir.iterdir()} != expected_completed_names
    ):
        raise ContractError("completed factorial directory set drifted")
    summaries: list[dict[str, object]] = []
    for row in plan["rows"]:
        expected_config = _expected_semantic_config(row, directions)
        completed = _completed_payload(
            output, row, plan_sha, current_source, expected_config
        )
        if completed is None:
            raise ContractError("completed factorial coverage drifted")
        attempt = _contained(
            output, str(completed["attempt_directory"]), "completed factorial attempt"
        )
        row_payload = _validate_row_payload(
            _load_json(attempt / "row.json"),
            row,
            plan_sha,
            current_source,
            expected_config,
        )
        summaries.append(_factorial_summary(row_payload, completed))
    aggregate = _validate_factorial_aggregate_payload(
        _load_json(aggregate_path), plan_sha, summaries
    )
    if aggregate_path.read_bytes() != _canonical_bytes(aggregate, newline=True):
        raise ContractError("completed factorial aggregate is not canonical JSON plus LF")
    aggregate_sha = _sha256(aggregate_path)
    receipt = _validate_factorial_receipt_payload(
        _load_json(receipt_path), plan, plan_sha, aggregate_sha
    )
    if receipt_path.read_bytes() != _canonical_bytes(receipt, newline=True):
        raise ContractError("completed factorial receipt is not canonical JSON plus LF")
    if validate_successor_source_freeze(root, source_receipt) != current_source:
        raise ContractError("successor source changed during finalized resume validation")
    validate_sealed_parent_diagnostic(root, parent_path)
    return receipt


def run_matrix(
    project_root: str | Path,
    output_directory: str | Path,
    *,
    v7_diagnostic_plan: str | Path,
    source_freeze_receipt: str | Path,
    resume: bool = False,
    plan_only: bool = False,
    row_timeout_seconds: int = 1800,
) -> dict[str, object]:
    root = Path(project_root).resolve()
    output = Path(output_directory).resolve()
    parent_path = Path(v7_diagnostic_plan).resolve()
    try:
        output_relative = output.relative_to(root)
    except ValueError as error:
        raise ContractError("factorial output directory escapes the project root") from error
    if not output_relative.parts:
        raise ContractError("factorial output directory cannot be the project root")
    validate_sealed_parent_diagnostic(root, parent_path)
    source_binding = validate_successor_source_freeze(root, source_freeze_receipt)
    plan = build_plan_payload(
        root, parent_path, source_binding, row_timeout_seconds=row_timeout_seconds
    )
    plan_path = output / "factorial.plan.json"
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        if (
            not plan_path.is_file()
            or _load_json(plan_path) != plan
            or plan_path.read_bytes() != _canonical_bytes(plan, newline=True)
        ):
            raise ContractError("resume plan disagrees with current frozen factorial plan")
    else:
        output.mkdir(parents=True)
        _exclusive_json(plan_path, plan)
    plan_sha = _sha256(plan_path)
    aggregate_path = output / "factorial.aggregate.json"
    receipt_path = output / "factorial.receipt.json"
    if receipt_path.exists():
        if not receipt_path.is_file():
            raise ContractError("factorial receipt path is not a regular file")
        if not resume:
            raise FileExistsError(receipt_path)
        _preflight_policy_fields()
        return _verify_completed_receipt(root, output, plan, plan_sha)
    aggregate_only_recovery = aggregate_path.exists()
    if aggregate_only_recovery:
        if not aggregate_path.is_file():
            raise ContractError("factorial aggregate path is not a regular file")
        expected_completed_names = {
            f"{row['row_id']}.json" for row in plan["rows"]
        }
        completed_dir = output / "completed"
        if (
            not completed_dir.is_dir()
            or {path.name for path in completed_dir.iterdir()} != expected_completed_names
        ):
            raise ContractError(
                "aggregate-only finalization has incomplete completed-row coverage"
            )
    if plan_only and not aggregate_only_recovery:
        return {
            "status": "FROZEN_PLAN_ONLY_NO_FACTORIAL_ROWS_EXECUTED",
            "plan_sha256": plan_sha,
            "expected_rows": FULL_ROW_COUNT,
            "selection_authorized": False,
            "formal_study_authorized": False,
        }
    _preflight_policy_fields()

    _cases, _bounds, directions, input_binding = v7_runner._load_inputs(root)
    if input_binding != plan["input_binding"]:
        raise ContractError("factorial execution input binding drifted")
    summaries: list[dict[str, object]] = []
    for row_spec in plan["rows"]:
        row_id = str(row_spec["row_id"])
        expected_config = _expected_semantic_config(row_spec, directions)
        completed = _completed_payload(
            output, row_spec, plan_sha, source_binding, expected_config
        )
        if completed is None:
            attempt = _next_attempt(output, row_id)
            worker_spec = {
                "schema": WORKER_SPEC_SCHEMA,
                "row_id": row_id,
                "ordinal": row_spec["ordinal"],
                "plan_sha256": plan_sha,
            }
            _exclusive_json(attempt / "worker.spec.json", worker_spec)
            _run_child(
                attempt / "worker.spec.json",
                project=root,
                timeout=_exact_int(plan["row_timeout_seconds"], "row timeout", minimum=1),
            )
            worker_result = _load_json(attempt / "worker.result.json")
            _require_keys(worker_result, WORKER_RESULT_KEYS, "new worker result")
            completed = {
                "schema": COMPLETED_SCHEMA,
                **worker_result,
                "attempt_directory": attempt.relative_to(output).as_posix(),
            }
            _require_keys(completed, COMPLETED_KEYS, "new completed row")
            _exclusive_json(output / "completed" / f"{row_id}.json", completed)
            completed = _completed_payload(
                output, row_spec, plan_sha, source_binding, expected_config
            )
            if completed is None:
                raise ContractError(f"new completed row seal is absent: {row_id}")
        attempt = _contained(output, str(completed["attempt_directory"]), "completed attempt")
        row = _load_json(attempt / "row.json")
        summaries.append(_factorial_summary(row, completed))
        print(f"completed {len(summaries)}/{FULL_ROW_COUNT} {row_id}", flush=True)
    if len(summaries) != FULL_ROW_COUNT:
        raise ContractError("factorial matrix is incomplete")
    expected_completed_names = {
        f"{row['row_id']}.json" for row in plan["rows"]
    }
    observed_completed_names = {
        path.name for path in (output / "completed").iterdir()
    }
    if observed_completed_names != expected_completed_names:
        raise ContractError("factorial completed directory set is not exactly 108 rows")
    if validate_successor_source_freeze(root, source_freeze_receipt) != source_binding:
        raise ContractError("successor source changed before final factorial receipt")
    aggregate = _factorial_aggregate_payload(plan_sha, summaries)
    expected_aggregate_raw = _canonical_bytes(aggregate, newline=True)
    if aggregate_only_recovery:
        if aggregate_path.read_bytes() != expected_aggregate_raw:
            raise ContractError(
                "aggregate-only finalization aggregate drifted from recomputed completed rows"
            )
    else:
        if aggregate_path.exists():
            raise ContractError("factorial aggregate appeared during finalization")
        _exclusive_json(aggregate_path, aggregate)
    aggregate_sha = _sha256(aggregate_path)
    if aggregate_path.read_bytes() != expected_aggregate_raw:
        raise ContractError("factorial aggregate changed during finalization")
    if receipt_path.exists():
        raise ContractError("factorial receipt appeared during finalization")
    receipt_core = _factorial_receipt_core(plan, plan_sha, aggregate_sha)
    receipt = {**receipt_core, "receipt_payload_sha256": _payload_sha256(receipt_core)}
    _exclusive_json(receipt_path, receipt)
    return _verify_completed_receipt(root, output, plan, plan_sha)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-directory")
    parser.add_argument("--v7-diagnostic-plan")
    parser.add_argument("--source-freeze-receipt")
    parser.add_argument("--row-timeout-seconds", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--worker-spec", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker_spec:
            result = _worker_run(args.worker_spec)
            _exclusive_json(Path(args.worker_spec).resolve().parent / "worker.result.json", result)
            print(json.dumps(result, sort_keys=True))
            return 0
        for field in ("output_directory", "v7_diagnostic_plan", "source_freeze_receipt"):
            if not getattr(args, field):
                parser.error("--" + field.replace("_", "-") + " is required")
        result = run_matrix(
            args.project_root,
            args.output_directory,
            v7_diagnostic_plan=args.v7_diagnostic_plan,
            source_freeze_receipt=args.source_freeze_receipt,
            resume=args.resume,
            plan_only=args.plan_only,
            row_timeout_seconds=args.row_timeout_seconds,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (
        ContractError,
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    ) as error:
        print(
            json.dumps(
                {
                    "schema": "v21e3r1_successor_factorial_runner_error_v1",
                    "status": "HOLD_INTEGRITY_OR_EXECUTION_ERROR",
                    "error": str(error),
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
