from __future__ import annotations

from dataclasses import dataclass

import pytest

from mo_nco.pareto_v21e3_parity import (
    analyze_development_parity,
    normalized_left_continuous_auc,
)


@dataclass(frozen=True)
class _Checkpoint:
    iteration: int
    archive_size: int
    front: tuple[tuple[float, float], ...]


def test_normalized_auc_is_left_continuous_and_zero_before_first_checkpoint() -> None:
    diagnostics = (
        _Checkpoint(2, 1, ((0.5, 0.5),)),
        _Checkpoint(4, 1, ((0.25, 0.5),)),
        _Checkpoint(6, 1, ((0.25, 0.25),)),
    )

    auc, final_hv, checkpoints = normalized_left_continuous_auc(
        diagnostics,
        budget=6,
        checkpoint_period=2,
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
    )

    # [0,2) contributes zero; [2,4) contributes .25; [4,6) contributes .375.
    assert auc == pytest.approx((2.0 * 0.25 + 2.0 * 0.375) / 6.0)
    assert final_hv == pytest.approx(0.5625)
    assert [item["evaluation"] for item in checkpoints] == [2, 4, 6]


def _complete_rows(delta: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cases: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    seeds = (31051, 31057, 31059)
    for family in ("MOTSP", "MOKP"):
        for size in (100, 200, 500):
            for ordinal in range(2):
                case_id = f"{family.lower()}-{size}-{ordinal}"
                cases.append({"case_id": case_id, "family": family, "size": size})
                for seed in seeds:
                    values = {
                        "V21E3_C0": 0.60 + delta,
                        "NSGAII": 0.60,
                        "MOEAD": 0.60,
                    }
                    for arm_id, value in values.items():
                        rows.append(
                            {
                                "case_id": case_id,
                                "family": family,
                                "size": size,
                                "seed": seed,
                                "arm_id": arm_id,
                                "normalized_left_continuous_hv_auc": value,
                            }
                        )
    return rows, cases


def test_complete_case_cluster_analysis_reports_all_frozen_statistics() -> None:
    rows, cases = _complete_rows(0.01)

    receipt = analyze_development_parity(
        rows,
        case_records=cases,
        seeds=(31051, 31057, 31059),
        bootstrap_samples=500,
        bootstrap_seed=31061,
    )

    assert receipt["completeness_gate"] == "PASS"
    assert receipt["observed_rows"] == 108
    assert receipt["overall_gate"] == "PASS_DEVELOPMENT_NONINFERIORITY"
    assert receipt["selection_entropy_release"] == "PROHIBITED"
    assert receipt["formal_authorized"] is False
    for family in ("MOTSP", "MOKP"):
        for comparator in ("NSGAII", "MOEAD"):
            comparison = receipt["comparisons"][family][comparator]
            assert comparison["cluster_count"] == 6
            assert comparison["mean_difference"] == pytest.approx(0.01)
            assert comparison["wins_ties_losses"] == {
                "wins": 6,
                "ties": 0,
                "losses": 0,
            }
            assert comparison["cluster_bootstrap_ci95"]["lower"] == pytest.approx(0.01)
            assert comparison["sign_flip_test"]["method"] == "exact_cluster_sign_flip"
            assert comparison["gate"] == "PASS"


def test_parity_failure_issues_the_frozen_stop_action() -> None:
    rows, cases = _complete_rows(-0.02)

    receipt = analyze_development_parity(
        rows,
        case_records=cases,
        seeds=(31051, 31057, 31059),
        bootstrap_samples=200,
        bootstrap_seed=31061,
    )

    assert receipt["overall_gate"] == "FAIL_STOP_BEFORE_SELECTION_PARTITION_MATERIALIZATION"
    assert receipt["comparisons"]["MOTSP"]["NSGAII"]["gate"] == "FAIL"


def test_parity_analysis_rejects_a_duplicate_or_missing_matrix_row() -> None:
    rows, cases = _complete_rows(0.01)
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        analyze_development_parity(
            rows,
            case_records=cases,
            seeds=(31051, 31057, 31059),
            bootstrap_samples=20,
            bootstrap_seed=31061,
        )

    rows, cases = _complete_rows(0.01)
    rows.pop()
    with pytest.raises(ValueError, match="complete matched"):
        analyze_development_parity(
            rows,
            case_records=cases,
            seeds=(31051, 31057, 31059),
            bootstrap_samples=20,
            bootstrap_seed=31061,
        )


@pytest.mark.parametrize("boolean_metric", [False, True])
def test_parity_analysis_rejects_boolean_matrix_metrics(boolean_metric: bool) -> None:
    rows, cases = _complete_rows(0.01)
    rows[0]["normalized_left_continuous_hv_auc"] = boolean_metric

    with pytest.raises(ValueError, match="metric.*number"):
        analyze_development_parity(
            rows,
            case_records=cases,
            seeds=(31051, 31057, 31059),
            bootstrap_samples=20,
            bootstrap_seed=31061,
        )

