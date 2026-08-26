from __future__ import annotations

import hashlib
import json

import pytest

from mo_nco.pareto_ijoc_problem import MultiObjectiveKnapsackInstance
from mo_nco.pareto_v21_diagnostics import (
    analyze_v21_trace,
    compare_paired_cluster_metric,
    write_v21_diagnostics_receipt,
)
from mo_nco.pareto_v21_hybrid import (
    V21HybridConfig,
    V21TypedHybridParetoSearch,
)
from mo_nco.pareto_v21_trace import (
    DecisionInput,
    EvaluationContext,
    SQLiteEvaluationLedger,
)


def test_analyze_v21_trace_reports_fixed_prefixes_and_operator_evidence(
    tmp_path,
) -> None:
    trace_path = tmp_path / "run.sqlite3"
    problem = MultiObjectiveKnapsackInstance.random_instance(18, seed=31001)
    config = V21HybridConfig(
        candidate_id="C3",
        reference_directions=((0.25, 0.75), (0.75, 0.25)),
        evaluations=20,
        checkpoint_period=5,
        seed=91,
        phase="development",
        trace_database=str(trace_path),
    )
    V21TypedHybridParetoSearch(problem, config).run()

    receipt = analyze_v21_trace(trace_path)

    assert receipt["schema"] == "v21_trace_diagnostics_v1"
    assert receipt["status"] == "PASS"
    assert [row["label"] for row in receipt["phase_checkpoints"]] == [
        "init_end",
        "early_10pct",
        "mid_70pct",
        "budget_end",
    ]
    assert [row["evaluation_index"] for row in receipt["phase_checkpoints"]] == [
        2,
        2,
        14,
        20,
    ]
    assert receipt["d2_evaluation_efficiency"]["evaluations"] == 20
    assert (
        receipt["d2_evaluation_efficiency"]["unique_evaluation_rate"]["status"]
        == "ESTABLISHED"
    )
    assert receipt["d3_operator_quality"]
    assert sum(row["attempts"] for row in receipt["d3_operator_quality"]) == 20
    assert sum(
        row["accepted_into_population"]
        for row in receipt["d3_operator_quality"]
    ) <= 20
    assert sum(
        row["archive_changes"] for row in receipt["d3_operator_quality"]
    ) == receipt["d2_evaluation_efficiency"]["archive_changes"]
    assert sum(
        row["retained_after_update"]
        for row in receipt["d3_operator_quality"]
    ) <= 20
    assert all(
        row["mean_positive_hv_gain"]["status"] == "NOT_ESTABLISHED"
        for row in receipt["d3_operator_quality"]
    )
    assert receipt["d4_typed_population_collapse"]["status"] == "ESTABLISHED"
    assert len(receipt["d4_typed_population_collapse"]["snapshots"]) == 3
    assert all(
        "population_solution_sha256" in snapshot["source_population_snapshot"]
        for snapshot in receipt["d4_typed_population_collapse"]["snapshots"]
    )


def test_diagnostics_receipt_writer_is_canonical_and_idempotent(tmp_path) -> None:
    trace_path = tmp_path / "canonical.sqlite3"
    output_path = tmp_path / "diagnostics.json"
    problem = MultiObjectiveKnapsackInstance.random_instance(12, seed=31002)
    config = V21HybridConfig(
        candidate_id="C0",
        reference_directions=((0.3, 0.7), (0.7, 0.3)),
        evaluations=10,
        checkpoint_period=5,
        seed=92,
        phase="calibration",
        trace_database=str(trace_path),
    )
    V21TypedHybridParetoSearch(problem, config).run()

    first = write_v21_diagnostics_receipt(
        trace_path,
        output_path,
        run_identity={"case_id": "mokp-dev-001", "seed": 92},
    )
    first_bytes = output_path.read_bytes()
    second = write_v21_diagnostics_receipt(
        trace_path,
        output_path,
        run_identity={"case_id": "mokp-dev-001", "seed": 92},
    )

    assert first == second
    assert first_bytes == output_path.read_bytes()
    assert first_bytes.endswith(b"\n")
    assert first["receipt_sha256"] == hashlib.sha256(first_bytes).hexdigest()
    payload = json.loads(first_bytes)
    assert payload["run_identity"] == {"case_id": "mokp-dev-001", "seed": 92}


def test_paired_comparator_aggregates_seeds_before_cluster_inference() -> None:
    rows = []
    values = {
        "a": {"V21": (3.0, 5.0), "C0": (1.0, 1.0)},
        "b": {"V21": (2.0, 2.0), "C0": (4.0, 4.0)},
        "c": {"V21": (5.0, 5.0), "C0": (4.0, 4.0)},
    }
    for case_id, arms in values.items():
        for arm, replicates in arms.items():
            for seed, value in enumerate(replicates):
                rows.append(
                    {
                        "family": "MOKP",
                        "case_id": case_id,
                        "seed": seed,
                        "algorithm": arm,
                        "auc": value,
                    }
                )

    result = compare_paired_cluster_metric(
        rows,
        cluster_keys=("family", "case_id"),
        arm_key="algorithm",
        treatment_arm="V21",
        control_arm="C0",
        value_key="auc",
        replicate_keys=("seed",),
        bootstrap_samples=2000,
        randomization_seed=21021,
    )

    assert result["schema"] == "v21_paired_cluster_comparison_v1"
    assert result["inference_unit"] == "case_cluster"
    assert result["cluster_count"] == 3
    assert result["input_observation_count"] == 12
    assert result["mean_difference"] == 2.0 / 3.0
    assert result["wins_ties_losses"] == {"wins": 2, "ties": 0, "losses": 1}
    assert result["sign_flip_test"]["method"] == "exact_cluster_sign_flip"
    assert result["sign_flip_test"]["two_sided_p"] == 0.75
    assert result["cluster_bootstrap_ci95"]["status"] == "ESTABLISHED"
    assert result == compare_paired_cluster_metric(
        rows,
        cluster_keys=("family", "case_id"),
        arm_key="algorithm",
        treatment_arm="V21",
        control_arm="C0",
        value_key="auc",
        replicate_keys=("seed",),
        bootstrap_samples=2000,
        randomization_seed=21021,
    )


def test_diagnostics_separate_observed_zero_from_missing_mechanism_evidence(
    tmp_path,
) -> None:
    trace_path = tmp_path / "zero-versus-missing.sqlite3"
    problem = MultiObjectiveKnapsackInstance.random_instance(8, seed=31003)
    ledger = SQLiteEvaluationLedger.from_problem(problem, database_path=trace_path)
    proposal = (0,) * problem.solution_size

    first = ledger.evaluate(
        proposal,
        EvaluationContext(
            evidence_partition="development",
            search_phase_id="initialization",
            stage_id="init",
            type_id=0,
            operator_id="seed",
            operator_call_id=1,
        ),
    )
    ledger.commit_decision(
        first.evaluation_index,
        DecisionInput(
            accepted_into_population=True,
            population_replacement_count=0,
            population_target_type_ids=(),
            decision_reason="initial_population_fill",
            archive_changed=True,
            retained_after_update=True,
            archive_size_after=1,
        ),
    )
    second = ledger.evaluate(
        proposal,
        EvaluationContext(
            evidence_partition="development",
            search_phase_id="initialization",
            stage_id="init",
            type_id=1,
            operator_id="never_archive",
            operator_call_id=2,
        ),
    )
    ledger.commit_decision(
        second.evaluation_index,
        DecisionInput(
            accepted_into_population=False,
            population_replacement_count=0,
            population_target_type_ids=(),
            decision_reason="duplicate",
            archive_changed=False,
            retained_after_update=True,
            archive_size_after=1,
        ),
    )
    ledger.finalize(expected_budget=2)

    receipt = analyze_v21_trace(trace_path)
    operator = next(
        row
        for row in receipt["d3_operator_quality"]
        if row["operator_id"] == "never_archive"
    )

    assert operator["archive_change_rate"] == {
        "status": "ESTABLISHED",
        "value": 0.0,
        "numerator": 0,
        "denominator": 1,
    }
    assert operator["archive_entry_rate"]["value"] == 0.0
    assert operator["archive_retention_rate"]["value"] == 1.0
    assert operator["feasible_before_repair_rate"]["status"] == "NOT_ESTABLISHED"
    assert receipt["d4_typed_population_collapse"]["status"] == "NOT_ESTABLISHED"
    assert receipt["d1_phase_localization"]["segments"][1]["status"] == (
        "NOT_APPLICABLE"
    )


def test_paired_comparator_fails_closed_on_unmatched_case() -> None:
    with pytest.raises(ValueError, match="Unmatched"):
        compare_paired_cluster_metric(
            [
                {"case": "a", "arm": "V21", "metric": 1.0},
                {"case": "a", "arm": "C0", "metric": 0.0},
                {"case": "b", "arm": "V21", "metric": 2.0},
            ],
            cluster_keys=("case",),
            arm_key="arm",
            treatment_arm="V21",
            control_arm="C0",
            value_key="metric",
        )


def test_paired_comparator_fails_closed_on_unmatched_seed_identity() -> None:
    with pytest.raises(ValueError, match="replicate identities"):
        compare_paired_cluster_metric(
            [
                {"case": "a", "seed": 1, "arm": "V21", "metric": 1.0},
                {"case": "a", "seed": 2, "arm": "C0", "metric": 0.0},
            ],
            cluster_keys=("case",),
            replicate_keys=("seed",),
            arm_key="arm",
            treatment_arm="V21",
            control_arm="C0",
            value_key="metric",
        )

