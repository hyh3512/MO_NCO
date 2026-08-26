from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mo_nco.pareto_ijoc_analysis import (
    analyze_ijoc_formal_results,
    build_paired_inference,
    recompute_quality_metrics,
)


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _make_completed_synthetic_matrix(
    root: Path,
) -> tuple[Path, Path, Path, Path]:
    frozen = root / "frozen"
    results = root / "results"
    frozen.mkdir()
    results.mkdir()
    treatment = "ijoc-pareto-smc"
    family_baselines = {
        "MOKP": ["mokp-a", "mokp-b", "mokp-c"],
        "MOTSP": ["motsp-a", "motsp-b", "motsp-c"],
    }
    cases = {
        family: [f"{family.lower()}-{index:02d}" for index in range(8)]
        for family in family_baselines
    }
    plan = {
        "schema": "ijoc_formal_analysis_plan_v1",
        "plan_id": "synthetic-seam-test",
        "status": "PRECOMMITTED_BEFORE_FORMAL_EXECUTION",
        "formal_evidence_status": "NOT_RUN",
        "families": ["MOKP", "MOTSP"],
        "treatment": treatment,
        "required_baselines": family_baselines,
        "formal_seeds": [8100],
        "evaluation_budgets": [20],
        "anytime_checkpoint_period": 10,
        "primary_budget": 20,
        "primary_metric": {
            "name": "normalized_left_continuous_hypervolume_auc",
            "orientation": "larger_is_better",
            "normalization": "per_case_frozen_ideal_nadir_affine",
            "hypervolume_reference": (
                "per_case_frozen_reference_calibration_artifact"
            ),
            "checkpoint_semantics": (
                "all_evaluated_nondominated_archive_at_each_common_checkpoint"
            ),
            "initial_value": 0.0,
        },
        "secondary_metrics": [
            {
                "name": "normalized_final_hypervolume",
                "orientation": "larger_is_better",
            },
            {
                "name": "igd_plus_to_frozen_supplied_reference",
                "orientation": "smaller_is_better",
            },
            {
                "name": "additive_epsilon_to_frozen_supplied_reference",
                "orientation": "smaller_is_better",
            },
            {"name": "wall_time_seconds", "orientation": "smaller_is_better"},
            {
                "name": "sampled_peak_process_tree_rss_bytes",
                "orientation": "smaller_is_better",
            },
        ],
        "comparison_unit": "same_family_case_seed_budget",
        "cluster_unit": "case_id",
        "family_pooling": "forbidden",
        "budget_pooling": "forbidden",
        "paired_contrast_orientation": (
            "positive_always_favors_treatment"
        ),
        "uncertainty": {
            "confidence_level": 0.95,
            "case_cluster_bootstrap_replicates": 200,
            "bootstrap_seed": 12345,
            "randomization_test": (
                "exact_two_sided_case_cluster_sign_flip"
            ),
            "multiplicity": (
                "holm_across_six_primary_family_by_baseline_comparisons"
            ),
            "familywise_alpha": 0.05,
        },
        "wins_ties_losses": {
            "unit": "paired_case_seed",
            "normalized_metric_absolute_tie_tolerance": 1e-12,
        },
        "primary_gate": {
            "scope": "each_family_separately_at_primary_budget",
            "requirements_for_superiority_claim": [
                (
                    "all_required_baseline_case_cluster_mean_auc_delta_"
                    "ci95_lower_bounds_are_strictly_positive"
                ),
                (
                    "all_required_baseline_holm_adjusted_exact_sign_flip_"
                    "p_values_are_at_most_0.05"
                ),
                "wins_exceed_losses_against_every_required_baseline",
            ],
            "failure_action": (
                "report_non_superiority_or_inconclusive_result_without_"
                "reselecting_policy_cases_seeds_budgets_or_metrics"
            ),
        },
        "efficiency_claim_gate": {
            "quality_gate_must_pass": True,
            "maximum_case_cluster_mean_runtime_ratio_ci95_upper": 1.25,
            "memory_claim": "reported_without_a_superiority_threshold",
        },
        "missing_or_failed_rows": {
            "imputation": "forbidden",
            "formal_matrix_completeness": "FAIL",
            "submission_status": "HOLD",
        },
        "reference_scope": (
            "supplied-reference-relative_only_not_true_pareto_front_"
            "completeness"
        ),
        "randomness_scope": (
            "pseudo_random_seeded_computational_experiment_not_a_true_"
            "randomness_certificate"
        ),
    }
    plan_path = frozen / "formal_analysis_plan.json"
    plan_sha = _write_json(plan_path, plan)

    metric_cases = {}
    for family, family_cases in cases.items():
        for case_id in family_cases:
            reference = {
                "schema": "ijoc_calibration_reference_case_v1",
                "case_id": case_id,
                "source_role": (
                    "reference_calibration_precommitted_disjoint_arms_and_seeds"
                ),
                "reference_calibration_precommit_sha256": "1" * 64,
                "metric_contract": {
                    "objective_sense": ["minimize", "minimize"],
                    "dominance_tolerance": 0.0,
                    "normalization": "frozen_ideal_nadir_affine",
                    "archive_semantics": (
                        "calibration_all_evaluated_nondominated"
                    ),
                    "evaluation_code_sha256": "2" * 64,
                },
                "reference_points": [[0.0, 10.0], [10.0, 0.0]],
                "ideal": [0.0, 0.0],
                "nadir": [10.0, 10.0],
                "hv_reference": [11.0, 11.0],
            }
            source_path = frozen / "artifacts" / f"{case_id}.reference.json"
            source_sha = _write_json(source_path, reference)
            metric_cases[case_id] = {
                "source_artifact": {
                    "path": source_path.relative_to(frozen).as_posix(),
                    "sha256": source_sha,
                },
                "source_role": reference["source_role"],
                "reference_sha256": _canonical_sha256(
                    reference["reference_points"]
                ),
                "reference_points": reference["reference_points"],
                "ideal": reference["ideal"],
                "nadir": reference["nadir"],
                "hv_reference": reference["hv_reference"],
            }
    metric_path = frozen / "metric_reference_manifest.json"
    metric_sha = _write_json(
        metric_path,
        {
            "schema": "ijoc_metric_reference_manifest_v2",
            "cases": metric_cases,
        },
    )

    matrix_rows = []
    problem_families = []
    for family, family_cases in cases.items():
        algorithms = [treatment, *family_baselines[family]]
        problem_families.append(
            {
                "id": family.lower(),
                "cases": family_cases,
                "algorithms": algorithms,
                "required_baselines": family_baselines[family],
            }
        )
        for case_id in family_cases:
            for algorithm in algorithms:
                readable = {
                    "case_id": case_id,
                    "algorithm": algorithm,
                    "seed": 8100,
                    "budget": 20,
                }
                matrix_rows.append(
                    {
                        **readable,
                        "configuration": readable,
                        "configuration_sha256": _canonical_sha256(readable),
                    }
                )
    matrix_path = frozen / "algorithm_configuration_matrix.json"
    matrix_sha = _write_json(
        matrix_path,
        {
            "schema": "ijoc_algorithm_configuration_matrix_v1",
            "rows": matrix_rows,
        },
    )
    release_path = frozen / "reproducibility_manifest.json"
    release_sha = _write_json(
        release_path, {"schema": "synthetic_test_release_v1"}
    )
    study = {
        "schema": "ijoc_competitive_study_v3",
        "study_id": "synthetic-complete-matrix",
        "problem_families": problem_families,
        "seeds": [8100],
        "budgets": [20],
        "anytime_checkpoint_period": 10,
        "formal_analysis_plan": {
            "path": plan_path.name,
            "sha256": plan_sha,
        },
        "metric_reference_manifest": {
            "path": metric_path.name,
            "sha256": metric_sha,
        },
        "algorithm_configuration_matrix": {
            "path": matrix_path.name,
            "sha256": matrix_sha,
        },
        "artifact_release": {
            "path": release_path.name,
            "sha256": release_sha,
        },
    }
    study_path = frozen / "study.json"
    study_sha = _write_json(study_path, study)
    execution_plan = {
        "schema": "ijoc_cold_process_execution_plan_v1",
        "study_sha256": study_sha,
        "configuration_matrix_sha256": matrix_sha,
        "execution_scope": "formal_candidate",
        "formal_evidence_status": "NOT_RUN",
        "formal_analysis_plan": {
            "path": plan_path.name,
            "sha256": plan_sha,
        },
    }
    execution_plan_path = frozen / "execution_plan.json"
    execution_plan_sha = _write_json(execution_plan_path, execution_plan)
    freeze_receipt_path = frozen / "freeze_receipt.json"
    freeze_receipt_sha = _write_json(
        freeze_receipt_path,
        {"schema": "synthetic_freeze_receipt_v1"},
    )
    invocation_path = results / "matrix_invocation.json"
    invocation_sha = _write_json(
        invocation_path,
        {
            "schema": "ijoc_cold_process_matrix_invocation_v1",
            "execution_scope": "formal_candidate",
            "formal_evidence_status": "NOT_RUN",
            "selection": {"kind": "all"},
            "selected_run_count": len(matrix_rows),
            "expected_run_count": len(matrix_rows),
        },
    )

    for row in matrix_rows:
        run_key = {
            "case_id": row["case_id"],
            "algorithm": row["algorithm"],
            "seed": row["seed"],
            "budget": row["budget"],
        }
        run_sha = _canonical_sha256(run_key)
        run_directory = results / "runs" / run_sha
        attempt = run_directory / "attempts" / "000001"
        better = row["algorithm"] == treatment
        first_entries = (
            [
                {"solution": [0], "objectives": [2.0, 8.0]},
                {"solution": [1], "objectives": [8.0, 2.0]},
            ]
            if better
            else [
                {"solution": [0], "objectives": [4.0, 9.0]},
                {"solution": [1], "objectives": [9.0, 4.0]},
            ]
        )
        final_entries = (
            [
                {"solution": [0], "objectives": [1.0, 7.0]},
                {"solution": [1], "objectives": [7.0, 1.0]},
            ]
            if better
            else [
                {"solution": [0], "objectives": [3.0, 8.0]},
                {"solution": [1], "objectives": [8.0, 3.0]},
            ]
        )
        archive_path = attempt / "all_evaluated_archive.json"
        archive_sha = _write_json(
            archive_path,
            {
                "schema": "ijoc_all_evaluated_archive_v1",
                "run_key": run_key,
                "instance_packet_sha256": "3" * 64,
                "problem_sha256": "4" * 64,
                "dominance_tolerance": 0.0,
                "archive_contract": (
                    "unbounded_exact_nondominated_union_of_all_evaluated_"
                    "candidates_v2"
                ),
                "entries": final_entries,
            },
        )
        checkpoint_path = attempt / "checkpoint_witnesses.json"
        checkpoint_sha = _write_json(
            checkpoint_path,
            {
                "schema": "ijoc_checkpoint_solution_witnesses_v1",
                "run_key": run_key,
                "checkpoint_period": 10,
                "checkpoints": [
                    {"evaluation": 10, "entries": first_entries},
                    {"evaluation": 20, "entries": final_entries},
                ],
            },
        )
        result_path = attempt / "algorithm_result.json"
        result_sha = _write_json(
            result_path,
            {
                "schema": "ijoc_algorithm_result_v1",
                "run_key": run_key,
                "status": "SUCCESS",
                "evaluations_used": 20,
                "observed_checkpoints": [10, 20],
                "archive_artifact": {
                    "path": archive_path.name,
                    "sha256": archive_sha,
                },
                "checkpoint_artifact": {
                    "path": checkpoint_path.name,
                    "sha256": checkpoint_sha,
                },
                # Deliberately false; the audit must not trust adapter metrics.
                "metrics": {"normalized_final_hypervolume": -999.0},
            },
        )
        replay_path = attempt / "replay_result.json"
        replay_sha = _write_json(
            replay_path,
            {
                "schema": "ijoc_replay_receipt_v1",
                "run_key": run_key,
                "status": "PASS",
                "instance_sha256": "3" * 64,
                "algorithm_result_sha256": result_sha,
                "archive_sha256": archive_sha,
                "checkpoint_artifact_sha256": checkpoint_sha,
                "evaluations_used": 20,
                "observed_checkpoints": [10, 20],
            },
        )
        relative = lambda path: path.relative_to(run_directory).as_posix()
        terminal = {
            "schema": "ijoc_cold_process_run_receipt_v1",
            "run_key": run_key,
            "run_key_sha256": run_sha,
            "study_sha256": study_sha,
            "configuration_matrix_sha256": matrix_sha,
            "execution_plan_sha256": execution_plan_sha,
            "freeze_receipt_sha256": freeze_receipt_sha,
            "execution_scope": "formal_candidate",
            "formal_evidence_status": "PENDING_POST_RUN_AUDIT",
            "attempt_number": 1,
            "status": "SUCCESS",
            "algorithm_process": {
                "wall_time_seconds": 1.0 if better else 2.0,
                "sampled_peak_process_tree_rss_bytes": (
                    100 if better else 200
                ),
                "resource_measurement_status": "PASS",
            },
            "algorithm_result": {
                "path": relative(result_path),
                "sha256": result_sha,
                "archive_path": relative(archive_path),
                "archive_sha256": archive_sha,
                "checkpoint_path": relative(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
            },
            "replay_result": {
                "path": relative(replay_path),
                "sha256": replay_sha,
            },
        }
        _write_json(run_directory / "terminal_receipt.json", terminal)

    postrun_path = results / "post_run_audit.json"
    _write_json(
        postrun_path,
        {
            "schema": "ijoc_post_run_audit_v2",
            "audit_implementation": {
                "scope": (
                    "posthoc_fail_closed_amendment_not_frozen_algorithm_runtime"
                ),
                "postrun_source_sha256": _sha256(
                    Path(__file__).parents[1]
                    / "mo_nco"
                    / "pareto_ijoc_postrun.py"
                ),
                "frozen_algorithm_modified": False,
                "formal_results_modified": False,
            },
            "study_sha256": study_sha,
            "configuration_matrix_sha256": matrix_sha,
            "execution_plan_sha256": execution_plan_sha,
            "freeze_receipt_sha256": freeze_receipt_sha,
            "matrix_invocation_sha256": invocation_sha,
            "expected_run_count": len(matrix_rows),
            "observed_unique_run_count": len(matrix_rows),
            "valid_run_count": len(matrix_rows),
            "missing_run_count": 0,
            "duplicate_run_count": 0,
            "unexpected_run_count": 0,
            "invalid_run_count": 0,
            "attempt_audit": {
                "retry_run_count": 0,
                "prior_attempt_count": 0,
                "retry_run_key_sha256": [],
                "quality_retry_failures": [],
                "histories": [],
            },
            "gates": {
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
            },
            "quality_estimand_scope": "reported_archive_relative",
            "all_evaluated_archive_claim_status": "NOT_ESTABLISHED",
            "resource_estimand_scope": (
                "descriptive_terminal_attempt_only"
            ),
            "resource_efficiency_claim_status": "NOT_ESTABLISHED",
            "formal_matched_matrix_gate": "PASS",
            "evidence_status": (
                "REPORTED_ARCHIVE_MATRIX_INTEGRITY_ESTABLISHED"
            ),
            "submission_verdict": (
                "HOLD_PENDING_METRIC_AND_STATISTICAL_AUDIT"
            ),
        },
    )
    return study_path, execution_plan_path, results, postrun_path


class IJOCFormalAnalysisTests(unittest.TestCase):
    def test_recomputes_frozen_normalized_metrics_and_left_continuous_auc(
        self,
    ) -> None:
        reference = {
            "reference_points": [[0.0, 10.0], [10.0, 0.0]],
            "ideal": [0.0, 0.0],
            "nadir": [10.0, 10.0],
            "hv_reference": [11.0, 11.0],
        }
        checkpoints = [
            {
                "evaluation": 10,
                "entries": [
                    {"solution": [0], "objectives": [2.0, 8.0]},
                    {"solution": [1], "objectives": [8.0, 2.0]},
                ],
            },
            {
                "evaluation": 20,
                "entries": [
                    {"solution": [0], "objectives": [1.0, 7.0]},
                    {"solution": [1], "objectives": [7.0, 1.0]},
                ],
            },
        ]
        final_entries = checkpoints[-1]["entries"]

        metrics = recompute_quality_metrics(
            final_entries=final_entries,
            checkpoints=checkpoints,
            budget=20,
            checkpoint_period=10,
            reference=reference,
        )

        # Frozen affine coordinates at checkpoint 10 are
        # (0.2, 0.8), (0.8, 0.2), with reference (1.1, 1.1):
        # HV = .9*.3 + .3*.6 = .45.  The first half of the
        # left-continuous trace is fixed to zero, hence AUC=.225.
        self.assertAlmostEqual(
            metrics["normalized_left_continuous_hypervolume_auc"],
            0.225,
        )
        self.assertAlmostEqual(
            metrics["normalized_final_hypervolume"],
            0.64,
        )
        self.assertAlmostEqual(
            metrics["igd_plus_to_frozen_supplied_reference"],
            0.1,
        )
        self.assertAlmostEqual(
            metrics["additive_epsilon_to_frozen_supplied_reference"],
            0.1,
        )

    def test_exact_case_sign_flip_holm_and_primary_gate_are_precommitted(
        self,
    ) -> None:
        treatment = "ijoc-pareto-smc"
        baselines = ["baseline-a", "baseline-b", "baseline-c"]
        rows = []
        for family in ("MOKP", "MOTSP"):
            for case_index in range(8):
                common = {
                    "family": family,
                    "case_id": f"{family.lower()}-{case_index:02d}",
                    "seed": 8100,
                    "budget": 100,
                }
                rows.append(
                    {
                        **common,
                        "algorithm": treatment,
                        "metrics": {
                            "normalized_left_continuous_hypervolume_auc": 0.8,
                            "normalized_final_hypervolume": 0.9,
                            "igd_plus_to_frozen_supplied_reference": 0.1,
                            "additive_epsilon_to_frozen_supplied_reference": 0.1,
                            "wall_time_seconds": 1.0,
                            "sampled_peak_process_tree_rss_bytes": 100,
                        },
                    }
                )
                for baseline in baselines:
                    rows.append(
                        {
                            **common,
                            "algorithm": baseline,
                            "metrics": {
                                "normalized_left_continuous_hypervolume_auc": 0.7,
                                "normalized_final_hypervolume": 0.8,
                                "igd_plus_to_frozen_supplied_reference": 0.2,
                                "additive_epsilon_to_frozen_supplied_reference": 0.2,
                                "wall_time_seconds": 2.0,
                                "sampled_peak_process_tree_rss_bytes": 200,
                            },
                        }
                    )
        plan = {
            "families": ["MOKP", "MOTSP"],
            "treatment": treatment,
            "required_baselines": {
                "MOKP": baselines,
                "MOTSP": baselines,
            },
            "formal_seeds": [8100],
            "evaluation_budgets": [100],
            "primary_budget": 100,
            "uncertainty": {
                "confidence_level": 0.95,
                "case_cluster_bootstrap_replicates": 200,
                "bootstrap_seed": 12345,
                "randomization_test": (
                    "exact_two_sided_case_cluster_sign_flip"
                ),
                "multiplicity": (
                    "holm_across_six_primary_family_by_baseline_comparisons"
                ),
                "familywise_alpha": 0.05,
            },
            "wins_ties_losses": {
                "unit": "paired_case_seed",
                "normalized_metric_absolute_tie_tolerance": 1e-12,
            },
            "efficiency_claim_gate": {
                "quality_gate_must_pass": True,
                "maximum_case_cluster_mean_runtime_ratio_ci95_upper": 1.25,
                "memory_claim": "reported_without_a_superiority_threshold",
            },
        }

        report = build_paired_inference(rows, plan=plan)

        self.assertEqual(len(report["primary_comparisons"]), 6)
        first = report["primary_comparisons"][0]
        self.assertEqual(first["paired_wins_ties_losses"], {
            "wins": 8,
            "ties": 0,
            "losses": 0,
        })
        self.assertAlmostEqual(
            first["exact_case_cluster_sign_flip_p_value"],
            2.0 / 256.0,
        )
        self.assertAlmostEqual(
            first["holm_adjusted_p_value"],
            6.0 * 2.0 / 256.0,
        )
        self.assertEqual(report["primary_superiority_gate"], "PASS")
        self.assertEqual(report["efficiency_claim_gate"], "PASS")

    def test_analysis_refuses_nonpassing_postrun_gate_before_reading_rows(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            postrun = root / "post_run_audit.json"
            postrun.write_text(
                json.dumps(
                    {
                        "schema": "ijoc_post_run_audit_v2",
                        "formal_matched_matrix_gate": "FAIL",
                        "evidence_status": "NOT_ESTABLISHED",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "analysis"

            with self.assertRaisesRegex(
                ValueError, "post-run.*PASS"
            ):
                analyze_ijoc_formal_results(
                    root / "missing-study.json",
                    root / "missing-execution-plan.json",
                    root / "missing-results",
                    postrun,
                    output,
                )

            self.assertFalse(output.exists())

    def test_completed_matrix_is_recomputed_bound_and_gated_end_to_end(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study, execution_plan, results, postrun = (
                _make_completed_synthetic_matrix(root)
            )
            output = root / "analysis"

            summary = analyze_ijoc_formal_results(
                study,
                execution_plan,
                results,
                postrun,
                output,
            )

            self.assertEqual(summary.row_count, 64)
            self.assertEqual(summary.formal_metric_statistical_gate, "PASS")
            self.assertEqual(summary.primary_superiority_gate, "PASS")
            self.assertEqual(
                summary.efficiency_claim_gate, "NOT_ESTABLISHED"
            )
            row_payload = json.loads(
                (output / "row_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(row_payload["row_count"], 64)
            self.assertEqual(
                row_payload["quality_estimand_scope"],
                "reported_archive_relative",
            )
            self.assertNotEqual(
                row_payload["rows"][0]["metrics"][
                    "normalized_final_hypervolume"
                ],
                -999.0,
            )
            inference = json.loads(
                (output / "paired_inference.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(inference["primary_superiority_gate"], "PASS")
            self.assertEqual(
                inference["efficiency_claim_gate"], "NOT_ESTABLISHED"
            )
            self.assertEqual(
                inference["quality_estimand_scope"],
                "reported_archive_relative",
            )
            self.assertEqual(
                len(
                    (output / "row_metrics.csv")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                65,
            )
            self.assertIn(
                "Primary superiority gate: **PASS**",
                (output / "FORMAL_ANALYSIS_REPORT.md").read_text(
                    encoding="utf-8"
                ),
            )
            audit = json.loads(
                (output / "formal_metric_statistical_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            audit_schema = json.loads(
                (
                    Path(__file__).parents[1]
                    / "ijoc_submission_v20"
                    / "protocol"
                    / "schemas"
                    / "ijoc_formal_metric_statistical_audit_v2.schema.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(set(audit), set(audit_schema["required"]))
            self.assertEqual(
                audit["gates"]["post_run_formal_gate"], "PASS"
            )
            self.assertEqual(
                audit["claim_boundaries"]["archive"],
                "reported_archive_relative",
            )
            self.assertEqual(
                audit["claim_boundaries"][
                    "all_evaluated_trace_completeness"
                ],
                "NOT_ESTABLISHED",
            )
            for binding in audit["outputs"].values():
                path = output / binding["path"]
                self.assertEqual(_sha256(path), binding["sha256"])
                self.assertEqual(path.stat().st_size, binding["bytes"])

    def test_stale_passing_postrun_cannot_hide_a_missing_terminal_row(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study, execution_plan, results, postrun = (
                _make_completed_synthetic_matrix(root)
            )
            missing = next(
                results.glob("runs/*/terminal_receipt.json")
            )
            missing.unlink()
            output = root / "must-not-exist"

            with self.assertRaisesRegex(
                ValueError, "terminal receipt|missing"
            ):
                analyze_ijoc_formal_results(
                    study,
                    execution_plan,
                    results,
                    postrun,
                    output,
                )

            self.assertFalse(output.exists())

    def test_stale_passing_postrun_cannot_hide_a_new_attempt_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study, execution_plan, results, postrun = (
                _make_completed_synthetic_matrix(root)
            )
            attempts = next(results.glob("runs/*/attempts"))
            (attempts / "000002").mkdir()
            output = root / "must-not-exist"

            with self.assertRaisesRegex(
                ValueError, "[Aa]ttempt audit|attempt"
            ):
                analyze_ijoc_formal_results(
                    study,
                    execution_plan,
                    results,
                    postrun,
                    output,
                )

            self.assertFalse(output.exists())

    def test_paired_inference_rejects_nonfinite_recomputed_values(self) -> None:
        plan = {
            "families": ["MOKP", "MOTSP"],
            "treatment": "ijoc-pareto-smc",
            "required_baselines": {
                "MOKP": ["m-a", "m-b", "m-c"],
                "MOTSP": ["t-a", "t-b", "t-c"],
            },
            "formal_seeds": [1],
            "evaluation_budgets": [10],
            "primary_budget": 10,
            "uncertainty": {
                "confidence_level": 0.95,
                "case_cluster_bootstrap_replicates": 10,
                "bootstrap_seed": 1,
                "randomization_test": (
                    "exact_two_sided_case_cluster_sign_flip"
                ),
                "multiplicity": (
                    "holm_across_six_primary_family_by_baseline_comparisons"
                ),
                "familywise_alpha": 0.05,
            },
            "wins_ties_losses": {
                "normalized_metric_absolute_tie_tolerance": 1e-12,
            },
            "efficiency_claim_gate": {
                "quality_gate_must_pass": True,
                "maximum_case_cluster_mean_runtime_ratio_ci95_upper": 1.25,
                "memory_claim": "reported_without_a_superiority_threshold",
            },
        }
        invalid_row = {
            "family": "MOKP",
            "case_id": "m-00",
            "algorithm": "ijoc-pareto-smc",
            "seed": 1,
            "budget": 10,
            "metrics": {
                metric: (
                    float("nan")
                    if metric
                    == "normalized_left_continuous_hypervolume_auc"
                    else 1.0
                )
                for metric in (
                    "normalized_left_continuous_hypervolume_auc",
                    "normalized_final_hypervolume",
                    "igd_plus_to_frozen_supplied_reference",
                    "additive_epsilon_to_frozen_supplied_reference",
                    "wall_time_seconds",
                    "sampled_peak_process_tree_rss_bytes",
                )
            },
        }

        with self.assertRaisesRegex(ValueError, "finite"):
            build_paired_inference([invalid_row], plan=plan)

    def test_cli_materializes_the_same_fail_closed_analysis_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study, execution_plan, results, postrun = (
                _make_completed_synthetic_matrix(root)
            )
            output = root / "cli-analysis"
            completed = subprocess.run(
                [
                    sys.executable,
                    "ijoc_submission_v20/scripts/"
                    "analyze_ijoc_formal_results.py",
                    "--study",
                    str(study),
                    "--execution-plan",
                    str(execution_plan),
                    "--results-directory",
                    str(results),
                    "--post-run-audit",
                    str(postrun),
                    "--output-directory",
                    str(output),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["formal_metric_statistical_gate"], "PASS")
            self.assertTrue(
                (output / "formal_metric_statistical_audit.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()

