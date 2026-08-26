from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from mo_nco.pareto_ijoc_problem import MultiObjectiveKnapsackInstance
from mo_nco.pareto_v21e3_hybrid import (
    V21E3HybridConfig,
    V21E3TypedHybridParetoSearch,
)
from mo_nco.pareto_v21e3r1_v9_theory import (
    DualResourceBudget,
    archive_compensated_replacement,
    information_time_equivalent,
    information_time_path,
    operator_productivity,
    select_first_unseen,
)


def test_information_time_discards_only_repeated_attempts() -> None:
    path = information_time_path(["a", "a", "b", "a", "c", "c"])
    assert path.first_visit_attempt_indices == (1, 3, 5)
    assert path.first_visit_states == ("a", "b", "c")
    assert path.total_attempts == 6
    assert information_time_equivalent(
        ["a", "a", "b", "c"], ["a", "b", "b", "c"]
    )


def test_operator_productivity_factorization() -> None:
    value = operator_productivity(
        attempts=20,
        new_states=5,
        total_quality_gain=0.4,
        elapsed_seconds=2.0,
    )
    assert value.unseen_rate == pytest.approx(0.25)
    assert value.conditional_gain_per_new_state == pytest.approx(0.08)
    assert value.gain_per_attempt == pytest.approx(0.02)
    assert value.factorization_residual == pytest.approx(0.0, abs=1e-15)
    assert value.gain_per_second == pytest.approx(0.2)


def test_archive_compensated_replacement_certifies_nonincrease() -> None:
    decision = archive_compensated_replacement(
        {0: -0.1, 1: 0.02, 2: 0.09},
        normalized_hv_gain=0.1,
        tradeoff_lambda=0.5,
    )
    assert decision.selected_targets == (0, 1)
    assert decision.scalar_delta_sum == pytest.approx(-0.08)
    assert decision.positive_scalar_worsening_sum == pytest.approx(0.02)
    assert decision.archive_credit == pytest.approx(0.05)
    assert decision.composite_potential_change <= 0.0
    assert decision.certified_nonincrease


def test_archive_compensated_replacement_rejects_unpaid_worsening() -> None:
    decision = archive_compensated_replacement(
        {0: 0.03, 1: 0.08},
        normalized_hv_gain=0.02,
        tradeoff_lambda=1.0,
    )
    assert decision.selected_targets == ()
    assert decision.composite_potential_change == pytest.approx(-0.02)


def test_first_unseen_screen_is_bounded() -> None:
    decision = select_first_unseen(
        ["a", "b", "c", "d"],
        is_seen=lambda value: value in {"a", "b"},
        cap=3,
    )
    assert decision.selected == "c"
    assert decision.candidates_examined == 3
    assert decision.cached_candidates_skipped == 2
    assert not decision.exhausted


def test_dual_resource_budget_requires_all_caps() -> None:
    budget = DualResourceBudget(10, 30, 100, 5.0)
    assert budget.permits(
        first_evaluations=10, attempts=30, screenings=100, elapsed_seconds=5.0
    )
    assert not budget.permits(
        first_evaluations=10, attempts=31, screenings=100, elapsed_seconds=5.0
    )


def _config(tmp_path: Path) -> V21E3HybridConfig:
    return V21E3HybridConfig(
        candidate_id="C0",
        reference_directions=((0.25, 0.75), (0.75, 0.25)),
        charged_evaluations=24,
        checkpoint_period=6,
        seed=17,
        phase="development",
        trace_database=str(tmp_path / "trace.sqlite3"),
        terminal_receipt=str(tmp_path / "terminal.json"),
        receipt_database_path="trace.sqlite3",
        capture_trace=True,
        local_improvement_steps=1,
        development_diagnostic_id="V21E3R1_V9_INFORMATION_LYAPUNOV_MOKP",
        candidate_screening_policy=(
            "bounded_cache_aware_structural_screen_development_v1"
        ),
        candidate_screening_cap=8,
        archive_tradeoff_lambda=0.5,
        replacement_policy=(
            "archive_compensated_information_lyapunov_development_v1"
        ),
        attempt_cap=256,
        structural_screening_cap=4096,
        wall_time_cap_seconds=30.0,
    )


def test_v9_policy_is_development_only() -> None:
    with pytest.raises(ValueError):
        V21E3HybridConfig(
            candidate_id="C0",
            reference_directions=((0.5, 0.5),),
            charged_evaluations=10,
            checkpoint_period=5,
            seed=0,
            phase="calibration",
            development_diagnostic_id="V21E3R1_V9_INFORMATION_LYAPUNOV_MOKP",
            candidate_screening_policy=(
                "bounded_cache_aware_structural_screen_development_v1"
            ),
            candidate_screening_cap=4,
            archive_tradeoff_lambda=0.5,
            replacement_policy=(
                "archive_compensated_information_lyapunov_development_v1"
            ),
            attempt_cap=64,
            structural_screening_cap=4096,
            wall_time_cap_seconds=30.0,
        )


def test_v9_mokp_integration_emits_screen_and_lyapunov_witnesses(
    tmp_path: Path,
) -> None:
    problem = MultiObjectiveKnapsackInstance.random_instance(
        12, num_objectives=2, seed=91
    )
    result = V21E3TypedHybridParetoSearch(problem, _config(tmp_path)).run()
    metadata = result.optimization_result.metadata
    assert metadata["charged_evaluation_count"] == 24
    assert metadata["candidate_screening_policy"] == (
        "bounded_cache_aware_structural_screen_development_v1"
    )
    assert metadata["replacement_contract"] == (
        "archive_compensated_information_lyapunov_development_v1"
    )
    assert metadata["candidate_screen_count"] >= 1
    connection = sqlite3.connect(tmp_path / "trace.sqlite3")
    try:
        decisions = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT decision_json FROM decisions ORDER BY evaluation_index"
            )
        ]
        attempts = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT context_json FROM attempts ORDER BY attempt_index"
            )
        ]
    finally:
        connection.close()
    witnesses = [item.get("policy_witness") for item in decisions]
    assert any(
        isinstance(item, dict)
        and item.get("schema") == "v21e3r1_archive_compensated_replacement_v2"
        and isinstance(item.get("finite_scalar_delta_by_target"), list)
        and isinstance(item.get("decision_selected_target_type_ids"), list)
        for item in witnesses
    )
    assert all(
        item is None
        or float(item["composite_potential_change"]) <= 1e-10
        for item in witnesses
    )
    assert any(
        isinstance(context.get("operator_witness"), dict)
        and "information_time_candidate_screen" in context["operator_witness"]
        for context in attempts
    )


def test_v9_semantic_payload_is_replay_complete(tmp_path: Path) -> None:
    from mo_nco.pareto_v21e3r1_branch_replay import _hybrid_dataclass_kwargs

    payload = _config(tmp_path).semantic_payload()
    kwargs = _hybrid_dataclass_kwargs(payload)
    assert kwargs["candidate_screening_policy"] == (
        "bounded_cache_aware_structural_screen_development_v1"
    )
    assert kwargs["archive_tradeoff_lambda"] == pytest.approx(0.5)


def test_v9_operator_diagnostic_reads_completed_trace(tmp_path: Path) -> None:
    from mo_nco.pareto_v21e3r1_v9_diagnostics import analyze_v9_trace_database

    problem = MultiObjectiveKnapsackInstance.random_instance(
        10, num_objectives=2, seed=7
    )
    V21E3TypedHybridParetoSearch(problem, _config(tmp_path)).run()
    report = analyze_v9_trace_database(tmp_path / "trace.sqlite3")
    assert report["status"] == "DEVELOPMENT_ONLY_NO_LATER_PHASE_AUTHORIZATION"
    assert report["operator_count"] >= 1
    assert sum(int(row["attempts"]) for row in report["operators"]) >= 24


@pytest.mark.parametrize(
    "diagnostic_id,screening,replacement,lam",
    [
        (
            "V21E3R1_V9_INFORMATION_SCREEN_MOKP",
            "bounded_cache_aware_structural_screen_development_v1",
            "bounded_reference_neighborhood_nonworse_replacement_v1",
            0.0,
        ),
        (
            "V21E3R1_V9_LYAPUNOV_MOKP",
            "disabled_v1",
            "archive_compensated_information_lyapunov_development_v1",
            0.5,
        ),
        (
            "V21E3R1_V9_INFORMATION_LYAPUNOV_MOKP",
            "bounded_cache_aware_structural_screen_development_v1",
            "archive_compensated_information_lyapunov_development_v1",
            0.5,
        ),
    ],
)
def test_v9_factorial_diagnostic_contracts(
    diagnostic_id: str,
    screening: str,
    replacement: str,
    lam: float,
) -> None:
    config = V21E3HybridConfig(
        candidate_id="C0",
        reference_directions=((0.5, 0.5),),
        charged_evaluations=8,
        checkpoint_period=4,
        seed=0,
        phase="development",
        capture_trace=False,
        local_improvement_steps=0,
        development_diagnostic_id=diagnostic_id,
        candidate_screening_policy=screening,
        candidate_screening_cap=4,
        archive_tradeoff_lambda=lam,
        replacement_policy=replacement,
        attempt_cap=64,
        structural_screening_cap=(4096 if screening != "disabled_v1" else 0),
        wall_time_cap_seconds=30.0,
    )
    assert config.development_diagnostic_id == diagnostic_id
