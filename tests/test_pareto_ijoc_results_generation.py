from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from mo_nco.pareto_ijoc_results_generation import (
    generate_ijoc_formal_result_artifacts,
)


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


def _write_json(path: Path, value: object) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_bytes(value)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _dummy_binding(path: str, label: str) -> dict[str, object]:
    return {"path": path, "sha256": _hash(label), "bytes": len(label)}


def _file_binding(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _holm(primary: list[dict[str, object]], alpha: float = 0.05) -> None:
    ordered = sorted(
        primary,
        key=lambda item: (
            float(item["exact_case_cluster_sign_flip_p_value"]),
            str(item["family"]),
            str(item["baseline"]),
        ),
    )
    running = 0.0
    for rank, item in enumerate(ordered, start=1):
        raw = float(item["exact_case_cluster_sign_flip_p_value"])
        running = max(running, (len(ordered) - rank + 1) * raw)
        adjusted = min(1.0, running)
        item["holm_rank"] = rank
        item["holm_adjusted_p_value"] = adjusted
        item["holm_reject_at_familywise_alpha"] = adjusted <= alpha


def _materialize_fixture(
    root: Path,
    *,
    pass_gate: bool = True,
    escaped_baseline: bool = False,
    precision_edge_case: bool = False,
) -> dict[str, Path]:
    results = root / "formal_results"
    analysis = root / "formal_analysis"
    results.mkdir(parents=True)
    analysis.mkdir(parents=True)
    study_sha = _hash("study")
    config_sha = _hash("configuration")
    execution_sha = _hash("execution")
    freeze_sha = _hash("freeze")
    invocation_sha = _hash("invocation")
    analysis_plan_sha = _hash("analysis-plan")
    row_metrics_sha = _hash("row-metrics")
    expected_rows = 6

    matrix = {
        "schema": "ijoc_cold_process_matrix_summary_v1",
        "study_sha256": study_sha,
        "configuration_matrix_sha256": config_sha,
        "execution_plan_sha256": execution_sha,
        "freeze_receipt_sha256": freeze_sha,
        "execution_scope": "formal_candidate",
        "formal_evidence_status": "PENDING_POST_RUN_AUDIT",
        "expected_run_count": expected_rows,
        "selected_run_count": expected_rows,
        "terminal_run_count": expected_rows,
        "success_count": expected_rows,
        "failure_count": 0,
        "workers": 4,
        "submission_verdict": "HOLD_PENDING_POST_RUN_AUDIT",
        "platform": "fixture-os",
        "python_executable": "fixture-python",
    }
    matrix_path = results / "matrix_summary.json"
    _write_json(matrix_path, matrix)

    postrun_gates = {
        "frozen_preflight_gate": "PASS",
        "full_invocation_gate": "PASS",
        "complete_row_set_gate": "PASS",
        "terminal_success_gate": "PASS",
        "budget_checkpoint_gate": "PASS",
        "hash_binding_gate": "PASS",
        "reported_archive_witness_self_consistency_gate": "PASS",
        "all_evaluated_trace_completeness_gate": "NOT_ESTABLISHED",
        "terminal_process_resource_measurement_gate": "PASS",
        "attempt_history_enumeration_gate": "PASS",
        "retry_quality_selection_gate": "PASS",
        "single_attempt_resource_cleanliness_gate": "PASS",
        "resource_design_balance_gate": "NOT_ESTABLISHED",
        "resource_efficiency_gate": "NOT_ESTABLISHED",
        "frozen_command_gate": "PASS",
    }
    postrun = {
        "schema": "ijoc_post_run_audit_v2",
        "audit_implementation": {
            "scope": (
                "posthoc_fail_closed_amendment_not_frozen_algorithm_runtime"
            ),
            "postrun_source_sha256": _hash("postrun-source"),
            "frozen_algorithm_modified": False,
            "formal_results_modified": False,
        },
        "study_sha256": study_sha,
        "configuration_matrix_sha256": config_sha,
        "execution_plan_sha256": execution_sha,
        "freeze_receipt_sha256": freeze_sha,
        "matrix_invocation_sha256": invocation_sha,
        "expected_run_count": expected_rows,
        "terminal_receipt_count": expected_rows,
        "observed_unique_run_count": expected_rows,
        "valid_run_count": expected_rows,
        "missing_run_count": 0,
        "duplicate_run_count": 0,
        "unexpected_run_count": 0,
        "invalid_run_count": 0,
        "gates": postrun_gates,
        "quality_estimand_scope": "reported_archive_relative",
        "all_evaluated_archive_claim_status": "NOT_ESTABLISHED",
        "resource_estimand_scope": "descriptive_terminal_attempt_only",
        "resource_efficiency_claim_status": "NOT_ESTABLISHED",
        "formal_matched_matrix_gate": "PASS",
        "evidence_status": (
            "REPORTED_ARCHIVE_MATRIX_INTEGRITY_ESTABLISHED"
        ),
        "competitive_superiority_status": "NOT_EVALUATED_BY_THIS_AUDIT",
        "submission_verdict": "HOLD_PENDING_METRIC_AND_STATISTICAL_AUDIT",
    }
    postrun_path = results / "post_run_audit.json"
    _write_json(postrun_path, postrun)

    top_inputs = {
        "study": {
            "path": "study.json",
            "sha256": study_sha,
            "bytes": 101,
        },
        "execution_plan": {
            "path": "execution_plan.json",
            "sha256": execution_sha,
            "bytes": 102,
        },
        "formal_analysis_plan": {
            "path": "formal_analysis_plan.json",
            "sha256": analysis_plan_sha,
            "bytes": 103,
        },
        "metric_reference_manifest": _dummy_binding(
            "metric_reference_manifest.json", "metric-reference"
        ),
        "algorithm_configuration_matrix": {
            "path": "algorithm_configuration_matrix.json",
            "sha256": config_sha,
            "bytes": 104,
        },
        "freeze_receipt": {
            "path": "freeze_receipt.json",
            "sha256": freeze_sha,
            "bytes": 105,
        },
        "matrix_invocation": {
            "path": "matrix_invocation.json",
            "sha256": invocation_sha,
            "bytes": 106,
        },
        "post_run_audit": _file_binding(postrun_path),
    }
    rows = []
    for index in range(expected_rows):
        run_key = {
            "case_id": f"case-{index}",
            "algorithm": f"algorithm-{index}",
            "seed": 8100 + index,
            "budget": 100000,
        }
        run_sha = _canonical_sha256(run_key)
        artifacts = {
            name: _dummy_binding(
                f"runs/{run_sha}/{name}.json", f"{run_sha}-{name}"
            )
            for name in (
                "terminal_receipt",
                "algorithm_result",
                "all_evaluated_archive",
                "checkpoint_witnesses",
                "replay_receipt",
            )
        }
        rows.append(
            {
                "run_key": run_key,
                "run_key_sha256": run_sha,
                "artifacts": artifacts,
            }
        )
    rows.sort(key=lambda row: str(row["run_key_sha256"]))
    consumed = {
        "schema": "ijoc_formal_analysis_consumed_artifacts_v1",
        "top_level_inputs": top_inputs,
        "metric_reference_sources": [
            {
                **_dummy_binding("artifacts/reference.json", "reference"),
                "case_id": "case-0",
            }
        ],
        "row_artifacts": rows,
        "row_artifact_count": expected_rows * 5,
        "terminal_receipt_set_sha256": _canonical_sha256(
            [row["artifacts"]["terminal_receipt"] for row in rows]
        ),
        "consumed_row_artifact_set_sha256": _canonical_sha256(rows),
    }
    consumed_path = analysis / "consumed_artifacts_manifest.json"
    consumed_sha, _ = _write_json(consumed_path, consumed)

    baseline_names = {
        "MOKP": ["mokp-a", "mokp-b", "mokp-c"],
        "MOTSP": ["motsp-a", "motsp-b", "motsp-c"],
    }
    if escaped_baseline:
        baseline_names["MOKP"][0] = "base_%&#${}~^\\one"
    primary: list[dict[str, object]] = []
    exact_values = iter((0.001, 0.002, 0.003, 0.004, 0.005, 0.006))
    for family in ("MOKP", "MOTSP"):
        for baseline in baseline_names[family]:
            p_value = next(exact_values)
            failing = not pass_gate and family == "MOKP" and baseline == baseline_names[family][0]
            precision_row = (
                precision_edge_case
                and family == "MOKP"
                and baseline == baseline_names[family][0]
            )
            lower = (
                -0.0234567890123456
                if precision_row
                else -0.02
                if failing
                else 0.01
            )
            upper = (
                0.000987654321098765
                if precision_row
                else 0.01
                if failing
                else 0.03
            )
            wins = 4 if failing or precision_row else 10
            ties = 1 if precision_row else 0
            losses = 5 if precision_row else 6 if failing else 0
            primary.append(
                {
                    "family": family,
                    "baseline": baseline,
                    "budget": 100000,
                    "metric": (
                        "normalized_left_continuous_hypervolume_auc"
                    ),
                    "case_cluster_mean_advantage": (
                        -0.0123456789012345
                        if precision_row
                        else -0.005
                        if failing
                        else 0.02
                    ),
                    "case_cluster_bootstrap_ci95": [lower, upper],
                    "paired_wins_ties_losses": {
                        "wins": wins,
                        "ties": ties,
                        "losses": losses,
                    },
                    "exact_case_cluster_sign_flip_p_value": (
                        0.0000123456789012345
                        if precision_row
                        else 0.5
                        if failing
                        else p_value
                    ),
                }
            )
    _holm(primary)
    for item in primary:
        wtl = item["paired_wins_ties_losses"]
        checks = {
            "auc_delta_ci95_lower_strictly_positive": (
                float(item["case_cluster_bootstrap_ci95"][0]) > 0.0
            ),
            "holm_adjusted_exact_p_at_most_familywise_alpha": (
                float(item["holm_adjusted_p_value"]) <= 0.05
            ),
            "paired_wins_exceed_losses": wtl["wins"] > wtl["losses"],
        }
        item["quality_gate_checks"] = checks
        item["quality_comparison_gate"] = (
            "PASS" if all(checks.values()) else "FAIL"
        )
    comparisons = [
        {
            "family": item["family"],
            "baseline": item["baseline"],
            "budget": item["budget"],
            "metric": item["metric"],
            "case_cluster_mean_advantage": item[
                "case_cluster_mean_advantage"
            ],
            "case_cluster_bootstrap_ci95": item[
                "case_cluster_bootstrap_ci95"
            ],
            "paired_wins_ties_losses": item["paired_wins_ties_losses"],
            "exact_case_cluster_sign_flip_p_value": item[
                "exact_case_cluster_sign_flip_p_value"
            ],
            "orientation": "higher",
            "paired_contrast": "positive_values_always_favor_treatment",
            "paired_case_seed_count": sum(
                item["paired_wins_ties_losses"].values()
            ),
            "case_cluster_count": 10,
            "randomization_method": (
                "exact_two_sided_case_cluster_sign_flip"
            ),
        }
        for item in primary
    ]
    overall_gate = (
        "PASS"
        if all(item["quality_comparison_gate"] == "PASS" for item in primary)
        else "FAIL"
    )
    family_gates = []
    for family in ("MOKP", "MOTSP"):
        family_gate = (
            "PASS"
            if all(
                item["quality_comparison_gate"] == "PASS"
                for item in primary
                if item["family"] == family
            )
            else "FAIL"
        )
        family_gates.append(
            {
                "family": family,
                "primary_comparison_count": 3,
                "primary_superiority_gate": family_gate,
                "efficiency_claim_gate": "NOT_ESTABLISHED",
            }
        )
    inference = {
        "schema": "ijoc_formal_paired_inference_v2",
        "treatment": "ijoc-pareto-smc",
        "families": ["MOKP", "MOTSP"],
        "budgets": [100000],
        "primary_budget": 100000,
        "comparison_unit": "same_family_case_seed_budget",
        "cluster_unit": "case_id",
        "family_pooling": "forbidden",
        "budget_pooling": "forbidden",
        "multiplicity": {
            "method": "Holm",
            "family": (
                "six_primary_family_by_required_baseline_comparisons"
            ),
            "familywise_alpha": 0.05,
        },
        "bootstrap": {
            "method": "case_cluster_percentile",
            "confidence_level": 0.95,
            "replicates": 1000,
            "base_seed": 20260831,
        },
        "randomization": {
            "method": "exact_two_sided_case_cluster_sign_flip",
            "monte_carlo_used": False,
        },
        "comparisons": comparisons,
        "primary_comparisons": primary,
        "family_gates": family_gates,
        "primary_superiority_gate": overall_gate,
        "efficiency_claim_gate": "NOT_ESTABLISHED",
        "memory_claim_gate": "NOT_ESTABLISHED",
        "reference_scope": (
            "supplied-reference-relative_only_not_true_pareto_front_"
            "completeness"
        ),
        "quality_estimand_scope": "reported_archive_relative",
        "reported_archive_witness_self_consistency": "PASS",
        "all_evaluated_trace_completeness": "NOT_ESTABLISHED",
        "resource_estimand_scope": "descriptive_terminal_attempt_only",
        "resource_efficiency_evidence_gate": "NOT_ESTABLISHED",
        "study_sha256": study_sha,
        "formal_analysis_plan_sha256": analysis_plan_sha,
        "row_metrics_sha256": row_metrics_sha,
    }
    inference_path = analysis / "paired_inference.json"
    _write_json(inference_path, inference)

    action = (
        "REPORTED_ARCHIVE_RELATIVE_SUPERIORITY_CLAIM_PERMITTED_"
        "WITHIN_FROZEN_SCOPE"
        if overall_gate == "PASS"
        else "REPORT_NON_SUPERIORITY_OR_INCONCLUSIVE_RESULT"
    )
    statistical_gates = {
        name: "PASS"
        for name in (
            "post_run_formal_gate",
            "frozen_input_hash_binding_gate",
            "complete_row_recomputation_gate",
            "paired_matrix_gate",
            "case_cluster_bootstrap_gate",
            "exact_sign_flip_gate",
            "six_comparison_holm_gate",
            "formal_metric_statistical_gate",
        )
    }
    audit = {
        "schema": "ijoc_formal_metric_statistical_audit_v2",
        "audit_implementation": {
            "scope": (
                "posthoc_fail_closed_amendment_not_frozen_algorithm_runtime"
            ),
            "analysis_source_sha256": _hash("analysis-source"),
            "frozen_algorithm_modified": False,
            "formal_results_modified": False,
        },
        "status": "COMPLETE",
        "formal_evidence_scope": (
            "reported_archive_relative_matched_matrix_metric_and_"
            "precommitted_paired_inference"
        ),
        "inputs": {
            **top_inputs,
            "consumed_artifact_manifest_sha256": consumed_sha,
        },
        "expected_row_count": expected_rows,
        "recomputed_row_count": expected_rows,
        "paired_comparison_count": len(comparisons),
        "primary_comparison_count": 6,
        "outputs": {
            "consumed_artifacts_manifest": _file_binding(consumed_path),
            "row_metrics": {
                "path": "row_metrics.json",
                "sha256": row_metrics_sha,
                "bytes": 201,
            },
            "row_metrics_csv": _dummy_binding(
                "row_metrics.csv", "row-metrics-csv"
            ),
            "paired_inference": _file_binding(inference_path),
            "paired_inference_csv": _dummy_binding(
                "paired_inference.csv", "inference-csv"
            ),
            "formal_analysis_report": _dummy_binding(
                "FORMAL_ANALYSIS_REPORT.md", "analysis-report"
            ),
        },
        "gates": statistical_gates,
        "primary_superiority_gate": overall_gate,
        "efficiency_claim_gate": "NOT_ESTABLISHED",
        "memory_claim_gate": "NOT_ESTABLISHED",
        "scientific_result_action": action,
        "submission_verdict": (
            "HOLD_PENDING_MANUSCRIPT_CONSISTENCY_AND_RELEASE_AUDIT"
        ),
        "claim_boundaries": {
            "reference": (
                "supplied-reference-relative_only_not_true_pareto_front_"
                "completeness"
            ),
            "archive": "reported_archive_relative",
            "reported_archive_witness_self_consistency": "PASS",
            "all_evaluated_trace_completeness": "NOT_ESTABLISHED",
            "resource": "descriptive_terminal_attempt_only",
            "resource_efficiency": "NOT_ESTABLISHED",
        },
    }
    audit_path = analysis / "formal_metric_statistical_audit.json"
    _write_json(audit_path, audit)
    return {
        "matrix": matrix_path,
        "postrun": postrun_path,
        "audit": audit_path,
        "inference": inference_path,
        "consumed": consumed_path,
    }


def _generate(paths: dict[str, Path], output: Path):
    return generate_ijoc_formal_result_artifacts(
        paths["matrix"],
        paths["postrun"],
        paths["audit"],
        paths["inference"],
        paths["consumed"],
        output,
    )


def test_pass_fixture_generates_canonical_tex_and_status(tmp_path: Path) -> None:
    paths = _materialize_fixture(tmp_path / "fixture")

    first = _generate(paths, tmp_path / "first")
    second = _generate(paths, tmp_path / "second")

    assert first.primary_superiority_gate == "PASS"
    assert first.tex_path.read_bytes() == second.tex_path.read_bytes()
    assert first.status_path.read_bytes() == second.status_path.read_bytes()
    tex = first.tex_path.read_text(encoding="utf-8")
    assert r"\newcommand{\IJOCPrimaryRowOneFamily}" in tex
    assert r"\newcommand{\IJOCPrimaryRowSixQualityGate}{PASS}" in tex
    assert r"\newcommand{\IJOCPrimaryComparisonRows}" in tex
    assert not re.search(r"\\IJOC[A-Za-z]*\d", tex)
    status = json.loads(first.status_path.read_text(encoding="utf-8"))
    assert status["formal_metric_statistical_gate"] == "PASS"
    assert status["primary_superiority_gate"] == "PASS"
    assert status["claim_boundaries"]["all_evaluated_trace_completeness"] == (
        "NOT_ESTABLISHED"
    )
    assert status["claim_boundaries"]["resource_estimand_scope"] == (
        "descriptive_terminal_attempt_only"
    )
    assert status["outputs"]["formal_results_generated_tex"]["sha256"] == (
        hashlib.sha256(first.tex_path.read_bytes()).hexdigest()
    )


def test_failed_quality_gate_is_generated_as_non_superiority(
    tmp_path: Path,
) -> None:
    paths = _materialize_fixture(tmp_path / "fixture", pass_gate=False)

    generated = _generate(paths, tmp_path / "output")

    assert generated.primary_superiority_gate == "FAIL"
    status = json.loads(generated.status_path.read_text(encoding="utf-8"))
    assert status["scientific_result_action"] == (
        "REPORT_NON_SUPERIORITY_OR_INCONCLUSIVE_RESULT"
    )
    assert status["scientific_interpretation"] == (
        "NON_SUPERIORITY_OR_INCONCLUSIVE_NO_RESELECTION"
    )
    assert status["reselection"] == {
        "failure_action_preserved": True,
        "policy_case_seed_budget_metric_reselection": "NOT_PERFORMED",
    }
    assert any(
        item["quality_comparison_gate"] == "FAIL"
        for item in status["primary_comparisons"]
    )
    assert r"\newcommand{\IJOCPrimarySuperiorityGate}{FAIL}" in (
        generated.tex_path.read_text(encoding="utf-8")
    )


def test_tex_uses_four_significant_digits_without_rounding_status_json(
    tmp_path: Path,
) -> None:
    paths = _materialize_fixture(
        tmp_path / "fixture",
        precision_edge_case=True,
    )
    inference_before = paths["inference"].read_bytes()

    generated = _generate(paths, tmp_path / "output")

    assert paths["inference"].read_bytes() == inference_before
    tex = generated.tex_path.read_text(encoding="utf-8")
    assert r"\newcommand{\IJOCPrimarySuperiorityGate}{FAIL}" in tex
    assert (
        r"\newcommand{\IJOCAllEvaluatedTraceCompleteness}"
        r"{NOT\_ESTABLISHED}"
        in tex
    )
    assert r"\newcommand{\IJOCPrimaryRowOneMean}{-0.01235}" in tex
    assert r"\newcommand{\IJOCPrimaryRowOneCILower}{-0.02346}" in tex
    assert r"\newcommand{\IJOCPrimaryRowOneCIUpper}{0.0009877}" in tex
    assert (
        r"\newcommand{\IJOCPrimaryRowOneCI}{[-0.02346, 0.0009877]}"
        in tex
    )
    assert r"\newcommand{\IJOCPrimaryRowOneWTL}{4/1/5}" in tex
    assert r"\newcommand{\IJOCPrimaryRowOneExactP}{1.235e-05}" in tex
    assert r"\newcommand{\IJOCPrimaryRowOneHolmP}{7.407e-05}" in tex
    assert r"\newcommand{\IJOCPrimaryRowOneQualityGate}{FAIL}" in tex

    status = json.loads(generated.status_path.read_text(encoding="utf-8"))
    row = status["primary_comparisons"][0]
    assert row["case_cluster_mean_advantage"] == -0.0123456789012345
    assert row["case_cluster_bootstrap_ci95"] == [
        -0.0234567890123456,
        0.000987654321098765,
    ]
    assert row["paired_wins_ties_losses"] == {
        "wins": 4,
        "ties": 1,
        "losses": 5,
    }
    assert row["exact_case_cluster_sign_flip_p_value"] == (
        0.0000123456789012345
    )
    assert row["holm_adjusted_p_value"] == 0.00007407407340740701
    assert row["quality_comparison_gate"] == "FAIL"


def test_missing_required_input_is_rejected_without_outputs(
    tmp_path: Path,
) -> None:
    paths = _materialize_fixture(tmp_path / "fixture")
    paths["consumed"].unlink()
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="consumed-artifact manifest is missing"):
        _generate(paths, output)

    assert not output.exists()


def test_bound_input_hash_drift_is_rejected_without_outputs(
    tmp_path: Path,
) -> None:
    paths = _materialize_fixture(tmp_path / "fixture")
    inference = json.loads(paths["inference"].read_text(encoding="utf-8"))
    inference["treatment"] = "tampered-after-audit"
    _write_json(paths["inference"], inference)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="hash/size drift"):
        _generate(paths, output)

    assert not output.exists()


def test_latex_special_characters_are_escaped_in_fixture(
    tmp_path: Path,
) -> None:
    paths = _materialize_fixture(
        tmp_path / "fixture", escaped_baseline=True
    )

    generated = _generate(paths, tmp_path / "output")

    tex = generated.tex_path.read_text(encoding="utf-8")
    assert (
        r"base\_\%\&\#\$\{\}\textasciitilde{}"
        r"\textasciicircum{}\textbackslash{}one"
    ) in tex
    assert "base_%&#" not in tex


def test_cli_uses_the_same_verified_generation_contract(tmp_path: Path) -> None:
    paths = _materialize_fixture(tmp_path / "fixture")
    output = tmp_path / "cli-output"

    completed = subprocess.run(
        [
            sys.executable,
            "ijoc_submission_v20/scripts/generate_ijoc_results_tex.py",
            "--matrix-summary",
            str(paths["matrix"]),
            "--post-run-audit",
            str(paths["postrun"]),
            "--statistical-audit",
            str(paths["audit"]),
            "--paired-inference",
            str(paths["inference"]),
            "--consumed-artifacts-manifest",
            str(paths["consumed"]),
            "--output-directory",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["primary_superiority_gate"] == "PASS"
    assert (output / "formal_results_generated.tex").is_file()
    assert (output / "STATUS.generated.json").is_file()

