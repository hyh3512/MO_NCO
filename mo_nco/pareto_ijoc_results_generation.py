from __future__ import annotations

"""Fail-closed IJOC statistics-to-LaTeX/status materialization.

This module does not recompute experimental metrics and does not inspect run
directories.  It verifies the completed matrix, post-run, statistical-audit,
paired-inference, and consumed-artifact contracts before deterministically
materializing manuscript-facing macros.  A failed scientific superiority gate
is a reportable result, not an integrity error; integrity or hash failures are
rejected.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence


MATRIX_SUMMARY_SCHEMA = "ijoc_cold_process_matrix_summary_v1"
POST_RUN_AUDIT_SCHEMA = "ijoc_post_run_audit_v2"
STATISTICAL_AUDIT_SCHEMA = "ijoc_formal_metric_statistical_audit_v2"
PAIRED_INFERENCE_SCHEMA = "ijoc_formal_paired_inference_v2"
CONSUMED_MANIFEST_SCHEMA = "ijoc_formal_analysis_consumed_artifacts_v1"
STATUS_SCHEMA = "ijoc_formal_results_generated_status_v1"

QUALITY_POSTRUN_GATES = (
    "frozen_preflight_gate",
    "full_invocation_gate",
    "complete_row_set_gate",
    "terminal_success_gate",
    "budget_checkpoint_gate",
    "hash_binding_gate",
    "reported_archive_witness_self_consistency_gate",
    "attempt_history_enumeration_gate",
    "retry_quality_selection_gate",
    "frozen_command_gate",
)
STATISTICAL_GATES = (
    "post_run_formal_gate",
    "frozen_input_hash_binding_gate",
    "complete_row_recomputation_gate",
    "paired_matrix_gate",
    "case_cluster_bootstrap_gate",
    "exact_sign_flip_gate",
    "six_comparison_holm_gate",
    "formal_metric_statistical_gate",
)
TOP_LEVEL_INPUTS = (
    "study",
    "execution_plan",
    "formal_analysis_plan",
    "metric_reference_manifest",
    "algorithm_configuration_matrix",
    "freeze_receipt",
    "matrix_invocation",
    "post_run_audit",
)
ROW_ARTIFACTS = (
    "terminal_receipt",
    "algorithm_result",
    "all_evaluated_archive",
    "checkpoint_witnesses",
    "replay_receipt",
)
PRIMARY_METRIC = "normalized_left_continuous_hypervolume_auc"
EXPECTED_FAMILIES = ("MOKP", "MOTSP")
REPORTED_ARCHIVE_EVIDENCE = (
    "REPORTED_ARCHIVE_MATRIX_INTEGRITY_ESTABLISHED"
)
FORMAL_EVIDENCE_SCOPE = (
    "reported_archive_relative_matched_matrix_metric_and_"
    "precommitted_paired_inference"
)
REFERENCE_SCOPE = (
    "supplied-reference-relative_only_not_true_pareto_front_completeness"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class IJOCGeneratedResults:
    output_directory: Path
    tex_path: Path
    status_path: Path
    primary_superiority_gate: str
    scientific_result_action: str


def _reject_duplicate_pairs(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field is forbidden: {key!r}.")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}.")


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], str, int]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Required {label} is missing: {resolved}.")
    raw = resolved.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON: {error}.") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object.")
    return value, hashlib.sha256(raw).hexdigest(), len(raw)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _safe_relative_path(value: object, *, label: str) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError(f"{label} path is not a canonical POSIX path.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{label} path escapes its declared root.")
    return path


def _binding_shape(
    value: object,
    *,
    label: str,
    extra_keys: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    expected = {"path", "sha256", "bytes", *extra_keys}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} must be an exact path/SHA/bytes binding.")
    _safe_relative_path(value.get("path"), label=label)
    if not _is_sha256(value.get("sha256")):
        raise ValueError(f"{label} SHA-256 is invalid.")
    size = value.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"{label} byte count is invalid.")
    return value


def _verify_bound_file(
    value: object,
    *,
    path: Path,
    digest: str,
    size: int,
    label: str,
) -> None:
    binding = _binding_shape(value, label=label)
    bound_path = _safe_relative_path(binding["path"], label=label)
    if bound_path.name != path.name:
        raise ValueError(f"{label} path does not identify the supplied file.")
    if binding["sha256"] != digest or binding["bytes"] != size:
        raise ValueError(f"{label} hash/size drift detected.")


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _same_float(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)


def _validate_matrix_and_postrun(
    matrix: Mapping[str, Any], postrun: Mapping[str, Any]
) -> int:
    if matrix.get("schema") != MATRIX_SUMMARY_SCHEMA:
        raise ValueError("Matrix-summary schema mismatch.")
    expected = _positive_int(
        matrix.get("expected_run_count"), label="matrix expected row count"
    )
    if (
        matrix.get("execution_scope") != "formal_candidate"
        or matrix.get("formal_evidence_status") != "PENDING_POST_RUN_AUDIT"
        or matrix.get("selected_run_count") != expected
        or matrix.get("terminal_run_count") != expected
        or matrix.get("success_count") != expected
        or matrix.get("failure_count") != 0
    ):
        raise ValueError("Matrix summary is not a complete successful formal matrix.")
    if postrun.get("schema") != POST_RUN_AUDIT_SCHEMA:
        raise ValueError("Post-run audit schema mismatch.")
    implementation = postrun.get("audit_implementation")
    if (
        not isinstance(implementation, dict)
        or implementation.get("scope")
        != "posthoc_fail_closed_amendment_not_frozen_algorithm_runtime"
        or not _is_sha256(implementation.get("postrun_source_sha256"))
        or implementation.get("frozen_algorithm_modified") is not False
        or implementation.get("formal_results_modified") is not False
    ):
        raise ValueError("Post-run audit implementation binding is invalid.")
    identity = {
        "study_sha256": "study_sha256",
        "configuration_matrix_sha256": "configuration_matrix_sha256",
        "execution_plan_sha256": "execution_plan_sha256",
        "freeze_receipt_sha256": "freeze_receipt_sha256",
    }
    for matrix_key, postrun_key in identity.items():
        value = matrix.get(matrix_key)
        if not _is_sha256(value) or postrun.get(postrun_key) != value:
            raise ValueError(f"Matrix/post-run identity drift: {matrix_key}.")
    required_zero = (
        "missing_run_count",
        "duplicate_run_count",
        "unexpected_run_count",
        "invalid_run_count",
    )
    if (
        postrun.get("expected_run_count") != expected
        or postrun.get("terminal_receipt_count") != expected
        or postrun.get("observed_unique_run_count") != expected
        or postrun.get("valid_run_count") != expected
        or any(postrun.get(key) != 0 for key in required_zero)
        or postrun.get("formal_matched_matrix_gate") != "PASS"
        or postrun.get("evidence_status") != REPORTED_ARCHIVE_EVIDENCE
    ):
        raise ValueError("Post-run audit does not establish the complete matrix.")
    gates = postrun.get("gates")
    if not isinstance(gates, dict) or any(
        gates.get(name) != "PASS" for name in QUALITY_POSTRUN_GATES
    ):
        raise ValueError("A required post-run quality gate is not PASS.")
    if (
        gates.get("all_evaluated_trace_completeness_gate")
        != "NOT_ESTABLISHED"
        or gates.get("resource_design_balance_gate") != "NOT_ESTABLISHED"
        or gates.get("resource_efficiency_gate") != "NOT_ESTABLISHED"
        or postrun.get("quality_estimand_scope") != "reported_archive_relative"
        or postrun.get("all_evaluated_archive_claim_status")
        != "NOT_ESTABLISHED"
        or postrun.get("resource_estimand_scope")
        != "descriptive_terminal_attempt_only"
        or postrun.get("resource_efficiency_claim_status")
        != "NOT_ESTABLISHED"
    ):
        raise ValueError("Post-run claim-boundary contract drifted.")
    return expected


def _validate_consumed_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_rows: int,
    matrix: Mapping[str, Any],
    postrun: Mapping[str, Any],
    postrun_path: Path,
    postrun_sha: str,
    postrun_bytes: int,
) -> Mapping[str, Mapping[str, Any]]:
    if manifest.get("schema") != CONSUMED_MANIFEST_SCHEMA:
        raise ValueError("Consumed-artifact manifest schema mismatch.")
    top = manifest.get("top_level_inputs")
    if not isinstance(top, dict) or set(top) != set(TOP_LEVEL_INPUTS):
        raise ValueError("Consumed manifest top-level input set drifted.")
    for name in TOP_LEVEL_INPUTS:
        _binding_shape(top[name], label=f"consumed top-level {name}")
    _verify_bound_file(
        top["post_run_audit"],
        path=postrun_path,
        digest=postrun_sha,
        size=postrun_bytes,
        label="consumed post-run audit",
    )
    identity = {
        "study": "study_sha256",
        "algorithm_configuration_matrix": "configuration_matrix_sha256",
        "execution_plan": "execution_plan_sha256",
        "freeze_receipt": "freeze_receipt_sha256",
        "matrix_invocation": "matrix_invocation_sha256",
    }
    for top_key, audit_key in identity.items():
        expected_sha = postrun.get(audit_key)
        if not _is_sha256(expected_sha) or top[top_key]["sha256"] != expected_sha:
            raise ValueError(f"Consumed manifest identity drift: {top_key}.")
    for key in (
        "study_sha256",
        "configuration_matrix_sha256",
        "execution_plan_sha256",
        "freeze_receipt_sha256",
    ):
        if matrix.get(key) != postrun.get(key):
            raise ValueError(f"Matrix/consumed identity drift: {key}.")

    reference_sources = manifest.get("metric_reference_sources")
    if not isinstance(reference_sources, list) or not reference_sources:
        raise ValueError("Consumed metric-reference source list is empty.")
    seen_cases: set[str] = set()
    for index, source in enumerate(reference_sources):
        binding = _binding_shape(
            source,
            label=f"metric reference source {index}",
            extra_keys=frozenset({"case_id"}),
        )
        case_id = binding.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen_cases:
            raise ValueError("Consumed metric-reference case identity is invalid.")
        seen_cases.add(case_id)

    rows = manifest.get("row_artifacts")
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise ValueError("Consumed row-artifact inventory is incomplete.")
    seen_run_keys: set[str] = set()
    ordered_run_keys: list[str] = []
    artifact_count = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
            "run_key",
            "run_key_sha256",
            "artifacts",
        }:
            raise ValueError(f"Consumed row {index} has an invalid shape.")
        run_key = row["run_key"]
        if not isinstance(run_key, dict) or set(run_key) != {
            "case_id",
            "algorithm",
            "seed",
            "budget",
        }:
            raise ValueError(f"Consumed row {index} run key is invalid.")
        if (
            not isinstance(run_key["case_id"], str)
            or not run_key["case_id"]
            or not isinstance(run_key["algorithm"], str)
            or not run_key["algorithm"]
            or isinstance(run_key["seed"], bool)
            or not isinstance(run_key["seed"], int)
            or isinstance(run_key["budget"], bool)
            or not isinstance(run_key["budget"], int)
            or run_key["budget"] <= 0
        ):
            raise ValueError(f"Consumed row {index} run-key values are invalid.")
        run_sha = row["run_key_sha256"]
        if run_sha != _canonical_sha256(run_key) or run_sha in seen_run_keys:
            raise ValueError("Consumed run-key hash or uniqueness check failed.")
        seen_run_keys.add(run_sha)
        ordered_run_keys.append(run_sha)
        artifacts = row["artifacts"]
        if not isinstance(artifacts, dict) or set(artifacts) != set(ROW_ARTIFACTS):
            raise ValueError("Consumed row artifact set drifted.")
        for name in ROW_ARTIFACTS:
            _binding_shape(
                artifacts[name], label=f"consumed row {index} {name}"
            )
        artifact_count += len(artifacts)
    if ordered_run_keys != sorted(ordered_run_keys):
        raise ValueError("Consumed rows are not in canonical run-key order.")
    if manifest.get("row_artifact_count") != artifact_count:
        raise ValueError("Consumed row-artifact count drifted.")
    terminal_bindings = [
        row["artifacts"]["terminal_receipt"] for row in rows
    ]
    if (
        manifest.get("terminal_receipt_set_sha256")
        != _canonical_sha256(terminal_bindings)
        or manifest.get("consumed_row_artifact_set_sha256")
        != _canonical_sha256(rows)
    ):
        raise ValueError("Consumed-artifact internal digest drift detected.")
    return top


def _validate_statistical_audit(
    audit: Mapping[str, Any],
    *,
    expected_rows: int,
    top_inputs: Mapping[str, Mapping[str, Any]],
    postrun_path: Path,
    postrun_sha: str,
    postrun_bytes: int,
    inference_path: Path,
    inference_sha: str,
    inference_bytes: int,
    consumed_path: Path,
    consumed_sha: str,
    consumed_bytes: int,
) -> None:
    if (
        audit.get("schema") != STATISTICAL_AUDIT_SCHEMA
        or audit.get("status") != "COMPLETE"
        or audit.get("formal_evidence_scope") != FORMAL_EVIDENCE_SCOPE
    ):
        raise ValueError("Formal metric/statistical audit contract mismatch.")
    implementation = audit.get("audit_implementation")
    if (
        not isinstance(implementation, dict)
        or implementation.get("scope")
        != "posthoc_fail_closed_amendment_not_frozen_algorithm_runtime"
        or not _is_sha256(implementation.get("analysis_source_sha256"))
        or implementation.get("frozen_algorithm_modified") is not False
        or implementation.get("formal_results_modified") is not False
    ):
        raise ValueError("Formal statistical implementation binding is invalid.")
    gates = audit.get("gates")
    if not isinstance(gates, dict) or set(gates) != set(STATISTICAL_GATES):
        raise ValueError("Formal metric/statistical gate set drifted.")
    if any(gates[name] != "PASS" for name in STATISTICAL_GATES):
        raise ValueError("Formal metric/statistical integrity gate is not PASS.")
    if (
        audit.get("expected_row_count") != expected_rows
        or audit.get("recomputed_row_count") != expected_rows
        or audit.get("primary_comparison_count") != 6
        or _positive_int(
            audit.get("paired_comparison_count"),
            label="paired comparison count",
        )
        <= 0
    ):
        raise ValueError("Formal metric/statistical row counts drifted.")
    outputs = audit.get("outputs")
    expected_output_names = {
        "consumed_artifacts_manifest",
        "row_metrics",
        "row_metrics_csv",
        "paired_inference",
        "paired_inference_csv",
        "formal_analysis_report",
    }
    if not isinstance(outputs, dict) or set(outputs) != expected_output_names:
        raise ValueError("Formal metric/statistical output set drifted.")
    for name, binding in outputs.items():
        _binding_shape(binding, label=f"statistical output {name}")
    _verify_bound_file(
        outputs["consumed_artifacts_manifest"],
        path=consumed_path,
        digest=consumed_sha,
        size=consumed_bytes,
        label="statistical consumed manifest",
    )
    _verify_bound_file(
        outputs["paired_inference"],
        path=inference_path,
        digest=inference_sha,
        size=inference_bytes,
        label="statistical paired inference",
    )
    inputs = audit.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        *TOP_LEVEL_INPUTS,
        "consumed_artifact_manifest_sha256",
    }:
        raise ValueError("Formal metric/statistical input set drifted.")
    if inputs.get("consumed_artifact_manifest_sha256") != consumed_sha:
        raise ValueError("Consumed-manifest audit digest drifted.")
    for name in TOP_LEVEL_INPUTS:
        if inputs.get(name) != top_inputs[name]:
            raise ValueError(f"Audit/consumed top-level binding drift: {name}.")
    _verify_bound_file(
        inputs["post_run_audit"],
        path=postrun_path,
        digest=postrun_sha,
        size=postrun_bytes,
        label="statistical post-run audit",
    )
    boundaries = audit.get("claim_boundaries")
    if (
        not isinstance(boundaries, dict)
        or boundaries.get("reference") != REFERENCE_SCOPE
        or boundaries.get("archive") != "reported_archive_relative"
        or boundaries.get("reported_archive_witness_self_consistency") != "PASS"
        or boundaries.get("all_evaluated_trace_completeness")
        != "NOT_ESTABLISHED"
        or boundaries.get("resource") != "descriptive_terminal_attempt_only"
        or boundaries.get("resource_efficiency") != "NOT_ESTABLISHED"
    ):
        raise ValueError("Formal audit claim boundaries drifted.")
    if (
        audit.get("efficiency_claim_gate") != "NOT_ESTABLISHED"
        or audit.get("memory_claim_gate") != "NOT_ESTABLISHED"
        or audit.get("submission_verdict")
        != "HOLD_PENDING_MANUSCRIPT_CONSISTENCY_AND_RELEASE_AUDIT"
    ):
        raise ValueError("Resource results are no longer descriptive-only.")


def _validate_primary_comparisons(
    inference: Mapping[str, Any], audit: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if inference.get("schema") != PAIRED_INFERENCE_SCHEMA:
        raise ValueError("Paired-inference schema mismatch.")
    families = inference.get("families")
    if not isinstance(families, list) or tuple(families) != EXPECTED_FAMILIES:
        raise ValueError("Paired-inference family contract drifted.")
    primary_budget = _positive_int(
        inference.get("primary_budget"), label="primary budget"
    )
    multiplicity = inference.get("multiplicity")
    if (
        not isinstance(multiplicity, dict)
        or multiplicity.get("method") != "Holm"
        or multiplicity.get("family")
        != "six_primary_family_by_required_baseline_comparisons"
    ):
        raise ValueError("Paired-inference multiplicity contract drifted.")
    alpha = _finite_number(
        multiplicity.get("familywise_alpha"), label="familywise alpha"
    )
    if not 0.0 < alpha < 1.0:
        raise ValueError("Familywise alpha is outside (0, 1).")
    randomization = inference.get("randomization")
    bootstrap = inference.get("bootstrap")
    if (
        not isinstance(randomization, dict)
        or randomization.get("method")
        != "exact_two_sided_case_cluster_sign_flip"
        or randomization.get("monte_carlo_used") is not False
        or not isinstance(bootstrap, dict)
        or bootstrap.get("method") != "case_cluster_percentile"
        or _finite_number(
            bootstrap.get("confidence_level"),
            label="bootstrap confidence level",
        )
        != 0.95
        or _positive_int(
            bootstrap.get("replicates"), label="bootstrap replicates"
        )
        <= 0
        or isinstance(bootstrap.get("base_seed"), bool)
        or not isinstance(bootstrap.get("base_seed"), int)
    ):
        raise ValueError("Paired-inference randomization/bootstrap contract drifted.")
    if (
        inference.get("comparison_unit") != "same_family_case_seed_budget"
        or inference.get("cluster_unit") != "case_id"
        or inference.get("family_pooling") != "forbidden"
        or inference.get("budget_pooling") != "forbidden"
        or inference.get("reference_scope") != REFERENCE_SCOPE
        or inference.get("quality_estimand_scope")
        != "reported_archive_relative"
        or inference.get("reported_archive_witness_self_consistency") != "PASS"
        or inference.get("all_evaluated_trace_completeness")
        != "NOT_ESTABLISHED"
        or inference.get("resource_estimand_scope")
        != "descriptive_terminal_attempt_only"
        or inference.get("resource_efficiency_evidence_gate")
        != "NOT_ESTABLISHED"
        or inference.get("efficiency_claim_gate") != "NOT_ESTABLISHED"
        or inference.get("memory_claim_gate") != "NOT_ESTABLISHED"
    ):
        raise ValueError("Paired-inference estimand boundary drifted.")

    comparisons = inference.get("comparisons")
    if (
        not isinstance(comparisons, list)
        or len(comparisons) != audit.get("paired_comparison_count")
    ):
        raise ValueError("Paired comparison inventory drifted.")
    comparison_index: dict[tuple[str, int, str, str], Mapping[str, Any]] = {}
    for raw in comparisons:
        if not isinstance(raw, dict):
            raise ValueError("Paired comparison must be an object.")
        raw_family = raw.get("family")
        raw_budget = raw.get("budget")
        raw_baseline = raw.get("baseline")
        raw_metric = raw.get("metric")
        if (
            not isinstance(raw_family, str)
            or isinstance(raw_budget, bool)
            or not isinstance(raw_budget, int)
            or not isinstance(raw_baseline, str)
            or not raw_baseline
            or not isinstance(raw_metric, str)
            or not raw_metric
        ):
            raise ValueError("Paired comparison identity is invalid.")
        key = (
            raw_family,
            raw_budget,
            raw_baseline,
            raw_metric,
        )
        if key in comparison_index:
            raise ValueError("Paired comparison key is duplicated.")
        comparison_index[key] = raw

    primary_raw = inference.get("primary_comparisons")
    if not isinstance(primary_raw, list) or len(primary_raw) != 6:
        raise ValueError("Exactly six primary comparisons are required.")
    primary: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    family_counts = {family: 0 for family in EXPECTED_FAMILIES}
    for index, raw in enumerate(primary_raw):
        if not isinstance(raw, dict):
            raise ValueError("Primary comparison must be an object.")
        family = raw.get("family")
        baseline = raw.get("baseline")
        if (
            family not in EXPECTED_FAMILIES
            or not isinstance(baseline, str)
            or not baseline
            or (family, baseline) in seen
            or raw.get("budget") != primary_budget
            or raw.get("metric") != PRIMARY_METRIC
        ):
            raise ValueError(f"Primary comparison {index} identity is invalid.")
        seen.add((family, baseline))
        family_counts[family] += 1
        comparison = comparison_index.get(
            (family, primary_budget, baseline, PRIMARY_METRIC)
        )
        if comparison is None:
            raise ValueError("Primary comparison lacks its paired source record.")
        if (
            comparison.get("orientation") != "higher"
            or comparison.get("paired_contrast")
            != "positive_values_always_favor_treatment"
            or comparison.get("randomization_method")
            != "exact_two_sided_case_cluster_sign_flip"
        ):
            raise ValueError("Primary paired-source semantics drifted.")
        mean = _finite_number(
            raw.get("case_cluster_mean_advantage"),
            label="primary mean advantage",
        )
        ci = raw.get("case_cluster_bootstrap_ci95")
        if not isinstance(ci, list) or len(ci) != 2:
            raise ValueError("Primary CI95 must contain two endpoints.")
        lower = _finite_number(ci[0], label="primary CI95 lower")
        upper = _finite_number(ci[1], label="primary CI95 upper")
        if lower > upper:
            raise ValueError("Primary CI95 endpoints are reversed.")
        wtl = raw.get("paired_wins_ties_losses")
        if not isinstance(wtl, dict) or set(wtl) != {"wins", "ties", "losses"}:
            raise ValueError("Primary W/T/L shape is invalid.")
        for name in ("wins", "ties", "losses"):
            value = wtl[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Primary W/T/L contains an invalid count.")
        if sum(wtl.values()) <= 0:
            raise ValueError("Primary W/T/L is empty.")
        if comparison.get("paired_case_seed_count") != sum(wtl.values()):
            raise ValueError("Primary W/T/L does not match the paired row count.")
        _positive_int(
            comparison.get("case_cluster_count"),
            label="primary case-cluster count",
        )
        exact_p = _finite_number(
            raw.get("exact_case_cluster_sign_flip_p_value"),
            label="primary exact p",
        )
        holm_p = _finite_number(
            raw.get("holm_adjusted_p_value"), label="primary Holm p"
        )
        if not 0.0 <= exact_p <= 1.0 or not 0.0 <= holm_p <= 1.0:
            raise ValueError("A primary p-value lies outside [0, 1].")
        for name, value in (
            ("case_cluster_mean_advantage", mean),
            ("exact_case_cluster_sign_flip_p_value", exact_p),
        ):
            source_value = _finite_number(
                comparison.get(name), label=f"paired source {name}"
            )
            if not _same_float(value, source_value):
                raise ValueError("Primary/source comparison statistic drifted.")
        source_ci = comparison.get("case_cluster_bootstrap_ci95")
        source_wtl = comparison.get("paired_wins_ties_losses")
        if source_ci != ci or source_wtl != wtl:
            raise ValueError("Primary/source CI or W/T/L drifted.")
        primary.append(
            {
                "source": raw,
                "family": family,
                "baseline": baseline,
                "mean": mean,
                "lower": lower,
                "upper": upper,
                "wins": wtl["wins"],
                "ties": wtl["ties"],
                "losses": wtl["losses"],
                "exact_p": exact_p,
                "holm_p": holm_p,
            }
        )
    if any(count != 3 for count in family_counts.values()):
        raise ValueError("Each formal family must have three primary baselines.")

    holm_order = sorted(
        primary,
        key=lambda item: (item["exact_p"], item["family"], item["baseline"]),
    )
    running = 0.0
    for rank, item in enumerate(holm_order, start=1):
        running = max(running, (7 - rank) * item["exact_p"])
        adjusted = min(1.0, running)
        raw = item["source"]
        if (
            raw.get("holm_rank") != rank
            or not _same_float(item["holm_p"], adjusted)
            or raw.get("holm_reject_at_familywise_alpha")
            != (adjusted <= alpha)
        ):
            raise ValueError("Stored Holm result does not recompute exactly.")

    for item in primary:
        raw = item["source"]
        checks = {
            "auc_delta_ci95_lower_strictly_positive": item["lower"] > 0.0,
            "holm_adjusted_exact_p_at_most_familywise_alpha": (
                item["holm_p"] <= alpha
            ),
            "paired_wins_exceed_losses": item["wins"] > item["losses"],
        }
        expected_gate = "PASS" if all(checks.values()) else "FAIL"
        if raw.get("quality_gate_checks") != checks:
            raise ValueError("Stored primary quality checks drifted.")
        if raw.get("quality_comparison_gate") != expected_gate:
            raise ValueError("Stored primary quality gate drifted.")
        item["quality_gate"] = expected_gate

    family_gates = inference.get("family_gates")
    if not isinstance(family_gates, list) or len(family_gates) != 2:
        raise ValueError("Paired-inference family gate inventory drifted.")
    by_family: dict[str, Mapping[str, Any]] = {}
    for gate in family_gates:
        if not isinstance(gate, dict) or gate.get("family") in by_family:
            raise ValueError("Paired-inference family gate is invalid.")
        by_family[str(gate.get("family"))] = gate
    for family in EXPECTED_FAMILIES:
        expected_gate = (
            "PASS"
            if all(
                item["quality_gate"] == "PASS"
                for item in primary
                if item["family"] == family
            )
            else "FAIL"
        )
        gate = by_family.get(family)
        if (
            gate is None
            or gate.get("primary_comparison_count") != 3
            or gate.get("primary_superiority_gate") != expected_gate
            or gate.get("efficiency_claim_gate") != "NOT_ESTABLISHED"
        ):
            raise ValueError("Stored family gate drifted.")
    overall = (
        "PASS"
        if all(item["quality_gate"] == "PASS" for item in primary)
        else "FAIL"
    )
    action = (
        "REPORTED_ARCHIVE_RELATIVE_SUPERIORITY_CLAIM_PERMITTED_"
        "WITHIN_FROZEN_SCOPE"
        if overall == "PASS"
        else "REPORT_NON_SUPERIORITY_OR_INCONCLUSIVE_RESULT"
    )
    if (
        inference.get("primary_superiority_gate") != overall
        or audit.get("primary_superiority_gate") != overall
        or audit.get("scientific_result_action") != action
        or audit.get("efficiency_claim_gate")
        != inference.get("efficiency_claim_gate")
        or audit.get("memory_claim_gate") != inference.get("memory_claim_gate")
    ):
        raise ValueError("Overall scientific gate/action drifted.")
    return sorted(primary, key=lambda item: (item["family"], item["baseline"]))


_LATEX_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "_": r"\_",
    "%": r"\%",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(value: object) -> str:
    """Escape an arbitrary scalar for a LaTeX text-mode macro argument."""

    text = str(value)
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise ValueError("LaTeX text values may not contain control characters.")
    return "".join(_LATEX_REPLACEMENTS.get(char, char) for char in text)


def _format_number(value: float) -> str:
    if value == 0.0:
        value = 0.0
    return format(value, ".4g")


_LATEX_ROW_WORDS = ("One", "Two", "Three", "Four", "Five", "Six")


def _macro(name: str, value: object) -> str:
    return rf"\newcommand{{\{name}}}{{{latex_escape(value)}}}"


def _render_tex(
    primary: Sequence[Mapping[str, Any]],
    *,
    expected_rows: int,
    gate: str,
    action: str,
) -> bytes:
    lines = [
        "% Deterministically generated; do not edit by hand.",
        _macro("IJOCFormalMatrixRows", expected_rows),
        _macro("IJOCFormalMetricStatisticalGate", "PASS"),
        _macro("IJOCPrimarySuperiorityGate", gate),
        _macro("IJOCScientificResultAction", action),
        _macro("IJOCQualityEstimandScope", "reported_archive_relative"),
        _macro("IJOCAllEvaluatedTraceCompleteness", "NOT_ESTABLISHED"),
        _macro("IJOCResourceEstimandScope", "descriptive_terminal_attempt_only"),
        _macro("IJOCResourceEfficiencyClaimGate", "NOT_ESTABLISHED"),
        _macro("IJOCReferenceScope", REFERENCE_SCOPE),
        _macro("IJOCPolicyReselectionStatus", "NOT_PERFORMED"),
        _macro(
            "IJOCScientificInterpretation",
            (
                "reported-archive-relative superiority within the frozen scope"
                if gate == "PASS"
                else (
                    "non-superiority or inconclusive result; no policy, case, "
                    "seed, budget, or metric reselection"
                )
            ),
        ),
    ]
    row_macro_names: list[str] = []
    if len(primary) > len(_LATEX_ROW_WORDS):
        raise ValueError("LaTeX output supports at most six primary rows.")
    for row_word, item in zip(
        _LATEX_ROW_WORDS[: len(primary)], primary, strict=True
    ):
        # TeX control-sequence names may contain letters only.  Spell the
        # ordinal instead of embedding a decimal digit in the macro name.
        prefix = f"IJOCPrimaryRow{row_word}"
        values = {
            "Family": item["family"],
            "Baseline": item["baseline"],
            "Mean": _format_number(float(item["mean"])),
            "CILower": _format_number(float(item["lower"])),
            "CIUpper": _format_number(float(item["upper"])),
            "CI": (
                f"[{_format_number(float(item['lower']))}, "
                f"{_format_number(float(item['upper']))}]"
            ),
            "Wins": item["wins"],
            "Ties": item["ties"],
            "Losses": item["losses"],
            "WTL": f"{item['wins']}/{item['ties']}/{item['losses']}",
            "ExactP": _format_number(float(item["exact_p"])),
            "HolmP": _format_number(float(item["holm_p"])),
            "QualityGate": item["quality_gate"],
        }
        for suffix, value in values.items():
            lines.append(_macro(prefix + suffix, value))
        row_name = prefix + "TableRow"
        row_macro_names.append(row_name)
        lines.extend(
            [
                rf"\newcommand{{\{row_name}}}{{%",
                (
                    rf"\{prefix}Family & \{prefix}Baseline & "
                    rf"\{prefix}Mean & \{prefix}CI & \{prefix}WTL & "
                    rf"\{prefix}ExactP & \{prefix}HolmP & "
                    rf"\{prefix}QualityGate \\%"
                ),
                "}",
            ]
        )
    lines.append(r"\newcommand{\IJOCPrimaryComparisonRows}{%")
    lines.extend(rf"\{name}%" for name in row_macro_names)
    lines.append("}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _input_binding(path: Path, digest: str, size: int) -> dict[str, object]:
    return {"path": path.name, "sha256": digest, "bytes": size}


def _write_atomic(path: Path, raw: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def generate_ijoc_formal_result_artifacts(
    matrix_summary_path: str | Path,
    post_run_audit_path: str | Path,
    statistical_audit_path: str | Path,
    paired_inference_path: str | Path,
    consumed_artifacts_manifest_path: str | Path,
    output_directory: str | Path,
) -> IJOCGeneratedResults:
    """Validate the v2 evidence chain and generate canonical status/LaTeX."""

    matrix_path = Path(matrix_summary_path).expanduser().resolve()
    postrun_path = Path(post_run_audit_path).expanduser().resolve()
    audit_path = Path(statistical_audit_path).expanduser().resolve()
    inference_path = Path(paired_inference_path).expanduser().resolve()
    consumed_path = Path(consumed_artifacts_manifest_path).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()

    matrix, matrix_sha, matrix_bytes = _read_json(
        matrix_path, label="matrix summary"
    )
    postrun, postrun_sha, postrun_bytes = _read_json(
        postrun_path, label="post-run audit"
    )
    audit, audit_sha, audit_bytes = _read_json(
        audit_path, label="formal metric/statistical audit"
    )
    inference, inference_sha, inference_bytes = _read_json(
        inference_path, label="paired inference"
    )
    consumed, consumed_sha, consumed_bytes = _read_json(
        consumed_path, label="consumed-artifact manifest"
    )

    expected_rows = _validate_matrix_and_postrun(matrix, postrun)
    top_inputs = _validate_consumed_manifest(
        consumed,
        expected_rows=expected_rows,
        matrix=matrix,
        postrun=postrun,
        postrun_path=postrun_path,
        postrun_sha=postrun_sha,
        postrun_bytes=postrun_bytes,
    )
    _validate_statistical_audit(
        audit,
        expected_rows=expected_rows,
        top_inputs=top_inputs,
        postrun_path=postrun_path,
        postrun_sha=postrun_sha,
        postrun_bytes=postrun_bytes,
        inference_path=inference_path,
        inference_sha=inference_sha,
        inference_bytes=inference_bytes,
        consumed_path=consumed_path,
        consumed_sha=consumed_sha,
        consumed_bytes=consumed_bytes,
    )
    if (
        inference.get("study_sha256") != top_inputs["study"]["sha256"]
        or inference.get("formal_analysis_plan_sha256")
        != top_inputs["formal_analysis_plan"]["sha256"]
        or inference.get("row_metrics_sha256")
        != audit["outputs"]["row_metrics"]["sha256"]
    ):
        raise ValueError("Paired-inference upstream hash binding drifted.")
    primary = _validate_primary_comparisons(inference, audit)
    gate = str(inference["primary_superiority_gate"])
    action = str(audit["scientific_result_action"])

    tex_raw = _render_tex(
        primary, expected_rows=expected_rows, gate=gate, action=action
    )
    tex_binding = {
        "path": "formal_results_generated.tex",
        "sha256": hashlib.sha256(tex_raw).hexdigest(),
        "bytes": len(tex_raw),
    }
    status = {
        "schema": STATUS_SCHEMA,
        "status": "COMPLETE",
        "generator_scope": (
            "deterministic_materialization_from_verified_v2_analysis_outputs"
        ),
        "inputs": {
            "matrix_summary": _input_binding(
                matrix_path, matrix_sha, matrix_bytes
            ),
            "post_run_audit": _input_binding(
                postrun_path, postrun_sha, postrun_bytes
            ),
            "formal_metric_statistical_audit": _input_binding(
                audit_path, audit_sha, audit_bytes
            ),
            "paired_inference": _input_binding(
                inference_path, inference_sha, inference_bytes
            ),
            "consumed_artifacts_manifest": _input_binding(
                consumed_path, consumed_sha, consumed_bytes
            ),
        },
        "matrix": {
            "expected_row_count": expected_rows,
            "successful_row_count": expected_rows,
            "failed_row_count": 0,
            "integrity_gate": "PASS",
        },
        "formal_metric_statistical_gate": "PASS",
        "primary_comparison_count": 6,
        "primary_superiority_gate": gate,
        "scientific_result_action": action,
        "scientific_interpretation": (
            "REPORTED_ARCHIVE_RELATIVE_SUPERIORITY_WITHIN_FROZEN_SCOPE"
            if gate == "PASS"
            else "NON_SUPERIORITY_OR_INCONCLUSIVE_NO_RESELECTION"
        ),
        "reselection": {
            "policy_case_seed_budget_metric_reselection": "NOT_PERFORMED",
            "failure_action_preserved": True,
        },
        "claim_boundaries": {
            "quality_estimand_scope": "reported_archive_relative",
            "all_evaluated_trace_completeness": "NOT_ESTABLISHED",
            "resource_estimand_scope": "descriptive_terminal_attempt_only",
            "resource_efficiency_claim_gate": "NOT_ESTABLISHED",
            "reference_scope": REFERENCE_SCOPE,
        },
        "primary_comparisons": [
            {
                "family": item["family"],
                "baseline": item["baseline"],
                "case_cluster_mean_advantage": item["mean"],
                "case_cluster_bootstrap_ci95": [
                    item["lower"],
                    item["upper"],
                ],
                "paired_wins_ties_losses": {
                    "wins": item["wins"],
                    "ties": item["ties"],
                    "losses": item["losses"],
                },
                "exact_case_cluster_sign_flip_p_value": item["exact_p"],
                "holm_adjusted_p_value": item["holm_p"],
                "quality_comparison_gate": item["quality_gate"],
            }
            for item in primary
        ],
        "outputs": {"formal_results_generated_tex": tex_binding},
        "submission_verdict": audit.get("submission_verdict"),
    }
    status_raw = _canonical_bytes(status)
    output.mkdir(parents=True, exist_ok=True)
    tex_path = output / "formal_results_generated.tex"
    status_path = output / "STATUS.generated.json"
    _write_atomic(tex_path, tex_raw)
    try:
        _write_atomic(status_path, status_raw)
    except BaseException:
        tex_path.unlink(missing_ok=True)
        raise
    return IJOCGeneratedResults(
        output_directory=output,
        tex_path=tex_path,
        status_path=status_path,
        primary_superiority_gate=gate,
        scientific_result_action=action,
    )
