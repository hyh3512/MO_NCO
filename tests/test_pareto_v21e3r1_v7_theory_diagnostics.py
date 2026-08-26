from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
import importlib.util
from pathlib import Path

import pytest

from mo_nco.instance import MultiObjectiveTSPInstance
from mo_nco.pareto_ijoc_problem import (
    MultiObjectiveKnapsackInstance,
    MultiObjectiveTSPProblemAdapter,
)
from mo_nco.pareto_v21e3_hybrid import (
    V21E3HybridConfig,
    V21E3TypedHybridParetoSearch,
)
from mo_nco.pareto_v21e3r1_branch_replay import reexecute_and_compare
from mo_nco.pareto_v21e3r1_construction import family_aware_initial_solution
from mo_nco.pareto_v21e3r1_development_diagnostics import (
    DIAGNOSTIC_SCOPE,
    aggregate_diagnostic_matrix,
    analyze_trace_database,
    baseline_diagnostic_configs,
    hybrid_diagnostic_config,
)
from mo_nco.pareto_v21e3r1_precedent import load_precedent_matrix, render_markdown
from mo_nco.pareto_v21e3r1_theory import (
    InterventionSummary,
    complexity_first_selection,
    duplicate_liveness_certificate,
    information_time_equivalent,
    resource_rank_reversal_example,
    successful_attempt_bound,
    validate_prefix_accounting,
)


def _mokp() -> MultiObjectiveKnapsackInstance:
    return MultiObjectiveKnapsackInstance(
        item_weights=(1, 2, 2, 3, 3, 4, 4, 5),
        profits_by_objective=(
            (9, 8, 2, 6, 3, 7, 1, 5),
            (1, 3, 9, 2, 8, 4, 7, 6),
        ),
        capacity=12,
        name="v7-diagnostic-mokp",
    )


def _motsp() -> MultiObjectiveTSPProblemAdapter:
    return MultiObjectiveTSPProblemAdapter(
        MultiObjectiveTSPInstance.random_biobjective(10, seed=8701)
    )


@pytest.mark.parametrize("problem", [_mokp(), _motsp()])
def test_shared_family_aware_constructor_is_deterministic_and_valid(problem) -> None:
    first, first_label = family_aware_initial_solution(problem, (0.25, 0.75), 3)
    second, second_label = family_aware_initial_solution(problem, (0.25, 0.75), 3)
    problem.validate_solution(first)
    assert first == second
    assert first_label == second_label


def test_hybrid_diagnostic_policies_are_development_only() -> None:
    random_config = hybrid_diagnostic_config(
        arm_id="C0_RANDOM",
        reference_directions=((0.2, 0.8), (0.8, 0.2)),
        charged_evaluations=8,
        checkpoint_period=2,
        seed=1,
    )
    assert random_config.phase == "development"
    assert random_config.development_diagnostic_id == "C0_RANDOM"

    with pytest.raises(ValueError, match="Alternative initialization"):
        V21E3HybridConfig(
            candidate_id="C0",
            reference_directions=((0.5, 0.5),),
            charged_evaluations=1,
            checkpoint_period=1,
            seed=1,
            phase="calibration",
            case_artifact_sha256="a" * 64,
            source_snapshot_sha256="b" * 64,
            initialization_policy=(
                "problem_native_exact_random_solution_development_diagnostic_v1"
            ),
        )


def test_zero_local_search_is_only_a_named_c0_development_diagnostic() -> None:
    config = hybrid_diagnostic_config(
        arm_id="C0_NO_LS",
        reference_directions=((0.2, 0.8), (0.8, 0.2)),
        charged_evaluations=8,
        checkpoint_period=2,
        seed=2,
    )
    assert config.local_improvement_steps == 0
    with pytest.raises(ValueError, match="Zero local-improvement"):
        V21E3HybridConfig(
            candidate_id="C1",
            reference_directions=((0.5, 0.5),),
            charged_evaluations=1,
            checkpoint_period=1,
            seed=1,
            phase="development",
            local_improvement_steps=0,
            development_diagnostic_id="ILLEGAL",
        )


def test_full_c0_factorial_and_replacement_diagnostics_are_development_only() -> None:
    directions = ((0.2, 0.8), (0.5, 0.5), (0.8, 0.2))
    random_no_ls = hybrid_diagnostic_config(
        arm_id="C0_RANDOM_NO_LS",
        reference_directions=directions,
        charged_evaluations=8,
        checkpoint_period=2,
        seed=12,
        family="MOKP",
    )
    assert random_no_ls.local_improvement_steps == 0
    assert "random_solution" in random_no_ls.initialization_policy
    self_replace = hybrid_diagnostic_config(
        arm_id="C0_SELF_REPLACE",
        reference_directions=directions,
        charged_evaluations=8,
        checkpoint_period=2,
        seed=13,
        family="MOKP",
    )
    assert self_replace.replacement_policy.startswith("self_type_nonworse")
    pop_match = hybrid_diagnostic_config(
        arm_id="C0_POP_MATCH",
        reference_directions=directions,
        charged_evaluations=48,
        checkpoint_period=2,
        seed=14,
        family="MOTSP",
    )
    assert len(pop_match.reference_directions) == 48


@pytest.mark.parametrize("family", ["MOTSP", "MOKP"])
@pytest.mark.parametrize(
    "arm",
    [
        "NSGAII_SEEDED", "MOEAD_SEEDED",
        "NSGAII_SEEDED_POP21", "MOEAD_SEEDED_POP21",
    ],
)
def test_seeded_baselines_are_explicit_development_diagnostics(family: str, arm: str) -> None:
    config = baseline_diagnostic_configs(
        family=family,
        arm_id=arm,
        charged_evaluations=96 if family == "MOTSP" else 80,
        checkpoint_period=48 if family == "MOTSP" else 40,
        seed=3,
    )
    assert config.initialization_policy == (
        "family_aware_per_slot_construction_development_diagnostic_v1"
    )
    assert config.development_diagnostic_id == arm
    assert config.evidence_partition == "development"
    if arm.endswith("POP21"):
        assert config.population_size == 21


def test_trace_analyzer_reconstructs_exact_per_evaluation_auc(tmp_path: Path) -> None:
    problem = _mokp()
    trace = tmp_path / "trace.sqlite3"
    terminal = tmp_path / "terminal.json"
    config = V21E3HybridConfig(
        candidate_id="C0",
        reference_directions=((0.2, 0.8), (0.5, 0.5), (0.8, 0.2)),
        charged_evaluations=24,
        checkpoint_period=6,
        seed=4,
        phase="development",
        trace_database=str(trace),
        terminal_receipt=str(terminal),
        local_improvement_steps=1,
    )
    V21E3TypedHybridParetoSearch(problem, config).run()
    row = {
        "case_id": problem.name,
        "family": "MOKP",
        "size": problem.solution_size,
        "seed": 4,
        "arm_id": "V21E3_C0",
        "charged_evaluation_budget": 24,
        "normalized_left_continuous_hv_auc": 0.0,
        "normalized_terminal_hv": 0.0,
    }
    result = analyze_trace_database(
        trace,
        row=row,
        lower=problem.objective_lower_bounds,
        upper=problem.objective_upper_bounds,
    )
    assert result["scientific_scope"] == DIAGNOSTIC_SCOPE
    assert result["attempt_count"] >= 24
    assert 0.0 <= result["exact_per_evaluation_left_continuous_hv_auc"] <= 1.0
    assert result["budget_slices"]["0_10"]["end_evaluation_inclusive"] == 2
    assert result["operators"]


def test_diagnostic_aggregate_never_authorizes_later_phases() -> None:
    base = {
        "family": "MOKP",
        "budget": 10,
        "exact_per_evaluation_left_continuous_hv_auc": 0.5,
        "initialization_terminal_hv": 0.4,
        "terminal_hv_replayed": 0.6,
        "attempts_per_charge": 1.0,
        "budget_slices": {
            label: {"mean_left_continuous_hv": 0.5}
            for label in ("0_10", "10_25", "25_50", "50_100")
        },
    }
    rows = []
    from mo_nco.pareto_v21e3r1_development_diagnostics import DIAGNOSTIC_ARMS
    for arm in DIAGNOSTIC_ARMS:
        rows.append({**base, "arm_id": arm})
    aggregate = aggregate_diagnostic_matrix(rows)
    assert aggregate["later_phase_authorization"] == "PROHIBITED"
    assert aggregate["scientific_scope"] == DIAGNOSTIC_SCOPE
    assert "MOKP" in aggregate["factorial_initialization_local_search"]


def test_precedent_matrix_is_complete_and_conservative() -> None:
    path = Path("ijoc_submission_v21e3r1/novelty/precedent_mechanism_matrix.json")
    payload = load_precedent_matrix(path)
    text = render_markdown(payload)
    assert "MOMAD" in text
    assert "V21e3r1" in text
    assert "NR=not established" in text
    current = next(x for x in payload["methods"] if x["method_id"] == "V21E3R1_CURRENT")
    assert current["mechanisms"]["full_algorithm_decision_replay"] == "NO"
    assert current["mechanisms"]["independent_confirmation"] == "NO"



def test_same_implementation_branch_replay_matches_hybrid_trace(tmp_path: Path) -> None:
    case = Path(
        "ijoc_submission_v21e3/development_partitions_v1/instances/"
        "v21e3-mokp-development-n100-s00.json"
    )
    from mo_nco.pareto_v21e3_baselines import load_v21e3_development_problem

    problem = load_v21e3_development_problem(case)
    trace = tmp_path / "trace.sqlite3"
    config = V21E3HybridConfig(
        candidate_id="C0",
        reference_directions=((0.2, 0.8), (0.5, 0.5), (0.8, 0.2)),
        charged_evaluations=24,
        checkpoint_period=6,
        seed=811,
        phase="development",
        trace_database=str(trace),
        terminal_receipt=str(tmp_path / "terminal.json"),
        local_improvement_steps=1,
    )
    V21E3TypedHybridParetoSearch(problem, config).run()
    result = reexecute_and_compare(
        original_database=trace,
        problem_artifact=case,
    )
    assert result["status"] == "PASS_SAME_IMPLEMENTATION_FULL_BRANCH_REEXECUTION"
    assert result["implementation_independence"] is False
    assert all(result["checks"].values())


def test_standard_library_independent_metric_reimplementation(tmp_path: Path) -> None:
    script = Path("independent_reproduction/recompute_v21e3r1_metrics.py")
    spec = importlib.util.spec_from_file_location("independent_metric_v7", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    problem = _mokp()
    trace = tmp_path / "trace.sqlite3"
    config = V21E3HybridConfig(
        candidate_id="C0",
        reference_directions=((0.2, 0.8), (0.5, 0.5), (0.8, 0.2)),
        charged_evaluations=24,
        checkpoint_period=6,
        seed=29,
        phase="development",
        trace_database=str(trace),
        terminal_receipt=str(tmp_path / "terminal.json"),
        local_improvement_steps=1,
    )
    V21E3TypedHybridParetoSearch(problem, config).run()
    result = module.recompute(
        trace,
        problem.objective_lower_bounds,
        problem.objective_upper_bounds,
    )
    assert result["status"] == "PASS_INDEPENDENT_METRIC_IMPLEMENTATION"
    assert result["evaluation_count"] == 24
    assert result["implementation_independence_from_project_metrics"] is True

def test_attempt_bounds_and_resource_rank_reversal() -> None:
    cert = successful_attempt_bound(
        100, retry_cap=4, fallback_cap=16, observed_attempts=350
    )
    assert cert.lower_bound == 100
    assert cert.upper_bound == 2100
    assert cert.passed is True
    accounting = validate_prefix_accounting(
        attempts=350,
        physical_starts=100,
        charges=100,
        cache_hits=250,
        unresolved_decisions=0,
        terminal_success=True,
    )
    assert accounting.terminal_success_pass
    example = resource_rank_reversal_example()
    assert example["two_first_true_evaluations"]["winner"] == "A"
    assert example["two_attempts"]["winner"] == "C"


def test_duplicate_liveness_and_information_time_invariance() -> None:
    certificate = duplicate_liveness_certificate(
        conditional_new_state_probability_lower_bound=Fraction(1, 2),
        retry_cap=1,
        fallback_cap=1,
        requested_charges=4,
    )
    assert certificate.attempts_per_service == 3
    assert certificate.per_service_failure_upper_bound == Fraction(1, 8)
    assert certificate.run_failure_upper_bound == Fraction(1, 2)
    assert certificate.expected_attempts_per_service_upper_bound == Fraction(7, 4)
    assert information_time_equivalent(
        ("a", "a", "b", "b", "c"),
        ("a", "b", "b", "c", "c"),
    )
    assert not information_time_equivalent(("a", "b"), ("b", "a"))


def _passing_intervention_summaries(
    families: tuple[str, ...],
) -> list[InterventionSummary]:
    return [
        InterventionSummary(
            candidate=candidate,
            primary_lower_bound_by_family={family: 0.04 for family in families},
            adjacent_lower_bound_by_family=(
                None
                if candidate == "C1"
                else {family: 0.04 for family in families}
            ),
            median_by_family={family: 0.04 for family in families},
            trimmed_mean_by_family={family: 0.04 for family in families},
            wins_by_family={family: 10 for family in families},
            losses_by_family={family: 2 for family in families},
        )
        for candidate in ("C1", "C2", "C3")
    ]


def test_complexity_first_gate_rejects_empty_family_set() -> None:
    with pytest.raises(ValueError, match="family_ids must be nonempty"):
        complexity_first_selection(
            _passing_intervention_summaries(()),
            family_ids=(),
            primary_threshold=0.01,
            adjacent_threshold=0.01,
            simultaneous_coverage_certified=True,
        )


@pytest.mark.parametrize("certificate", ["NOT_CERTIFIED", 1])
def test_complexity_first_gate_requires_exact_boolean_certificate(
    certificate: object,
) -> None:
    families = ("MOKP", "MOTSP")
    with pytest.raises(TypeError, match="exact Boolean"):
        complexity_first_selection(
            _passing_intervention_summaries(families),
            family_ids=families,
            primary_threshold=0.01,
            adjacent_threshold=0.01,
            simultaneous_coverage_certified=certificate,  # type: ignore[arg-type]
        )


def test_complexity_first_gate_rejects_duplicate_candidate_summary() -> None:
    families = ("MOKP", "MOTSP")
    summaries = _passing_intervention_summaries(families)
    summaries.append(summaries[0])
    with pytest.raises(ValueError, match="Exactly one summary"):
        complexity_first_selection(
            summaries,
            family_ids=families,
            primary_threshold=0.01,
            adjacent_threshold=0.01,
            simultaneous_coverage_certified=True,
        )


def test_complexity_first_gate_rejects_duplicate_family_id() -> None:
    summary_families = ("MOKP",)
    with pytest.raises(ValueError, match="family_ids must be unique"):
        complexity_first_selection(
            _passing_intervention_summaries(summary_families),
            family_ids=("MOKP", "MOKP"),
            primary_threshold=0.01,
            adjacent_threshold=0.01,
            simultaneous_coverage_certified=True,
        )


@pytest.mark.parametrize("families", [("",), ("   ",), (1,), (True,)])
def test_complexity_first_gate_requires_nonempty_exact_string_family_ids(
    families: tuple[object, ...],
) -> None:
    with pytest.raises(TypeError, match="nonempty exact strings"):
        complexity_first_selection(
            _passing_intervention_summaries(families),  # type: ignore[arg-type]
            family_ids=families,  # type: ignore[arg-type]
            primary_threshold=0.01,
            adjacent_threshold=0.01,
            simultaneous_coverage_certified=True,
        )


@pytest.mark.parametrize(
    ("primary_threshold", "adjacent_threshold"),
    [(True, 0.01), (0.01, False)],
)
def test_complexity_first_gate_rejects_boolean_thresholds(
    primary_threshold: object,
    adjacent_threshold: object,
) -> None:
    families = ("MOKP", "MOTSP")
    with pytest.raises(TypeError, match="thresholds must be finite real numbers"):
        complexity_first_selection(
            _passing_intervention_summaries(families),
            family_ids=families,
            primary_threshold=primary_threshold,  # type: ignore[arg-type]
            adjacent_threshold=adjacent_threshold,  # type: ignore[arg-type]
            simultaneous_coverage_certified=True,
        )


@pytest.mark.parametrize(
    ("primary_threshold", "adjacent_threshold"),
    [(-0.01, 0.01), (0.01, -0.01)],
)
def test_complexity_first_gate_rejects_negative_thresholds(
    primary_threshold: float,
    adjacent_threshold: float,
) -> None:
    families = ("MOKP", "MOTSP")
    with pytest.raises(ValueError, match="nonnegative"):
        complexity_first_selection(
            _passing_intervention_summaries(families),
            family_ids=families,
            primary_threshold=primary_threshold,
            adjacent_threshold=adjacent_threshold,
            simultaneous_coverage_certified=True,
        )


@pytest.mark.parametrize(
    ("primary_threshold", "adjacent_threshold"),
    [
        (float("nan"), 0.01),
        (0.01, float("nan")),
        (float("inf"), 0.01),
        (0.01, float("inf")),
    ],
)
def test_complexity_first_gate_rejects_nonfinite_thresholds(
    primary_threshold: float,
    adjacent_threshold: float,
) -> None:
    families = ("MOKP", "MOTSP")
    with pytest.raises(ValueError, match="finite"):
        complexity_first_selection(
            _passing_intervention_summaries(families),
            family_ids=families,
            primary_threshold=primary_threshold,
            adjacent_threshold=adjacent_threshold,
            simultaneous_coverage_certified=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("wins_by_family", True),
        ("wins_by_family", 3.5),
        ("wins_by_family", "3"),
        ("losses_by_family", False),
        ("losses_by_family", 2.0),
        ("losses_by_family", "2"),
    ],
)
def test_complexity_first_gate_requires_exact_integer_win_loss_counts(
    field: str,
    value: object,
) -> None:
    families = ("MOKP", "MOTSP")
    summaries = _passing_intervention_summaries(families)
    bad_counts = dict(getattr(summaries[0], field))
    bad_counts["MOKP"] = value
    summaries[0] = replace(summaries[0], **{field: bad_counts})
    with pytest.raises(TypeError, match="exact nonnegative integers"):
        complexity_first_selection(
            summaries,
            family_ids=families,
            primary_threshold=0.01,
            adjacent_threshold=0.01,
            simultaneous_coverage_certified=True,
        )


@pytest.mark.parametrize("field", ["wins_by_family", "losses_by_family"])
def test_complexity_first_gate_rejects_negative_win_loss_counts(field: str) -> None:
    families = ("MOKP", "MOTSP")
    summaries = _passing_intervention_summaries(families)
    bad_counts = dict(getattr(summaries[0], field))
    bad_counts["MOKP"] = -1
    summaries[0] = replace(summaries[0], **{field: bad_counts})
    with pytest.raises(ValueError, match="nonnegative"):
        complexity_first_selection(
            summaries,
            family_ids=families,
            primary_threshold=0.01,
            adjacent_threshold=0.01,
            simultaneous_coverage_certified=True,
        )


@pytest.mark.parametrize(
    ("candidate_index", "field"),
    [
        (0, "primary_lower_bound_by_family"),
        (0, "median_by_family"),
        (0, "trimmed_mean_by_family"),
        (1, "adjacent_lower_bound_by_family"),
    ],
)
def test_complexity_first_gate_rejects_boolean_effect_statistics(
    candidate_index: int,
    field: str,
) -> None:
    families = ("MOKP", "MOTSP")
    summaries = _passing_intervention_summaries(families)
    bad_values = dict(getattr(summaries[candidate_index], field))
    bad_values["MOKP"] = True
    summaries[candidate_index] = replace(
        summaries[candidate_index],
        **{field: bad_values},
    )
    with pytest.raises(TypeError, match="finite real numbers"):
        complexity_first_selection(
            summaries,
            family_ids=families,
            primary_threshold=0.01,
            adjacent_threshold=0.01,
            simultaneous_coverage_certified=True,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
@pytest.mark.parametrize(
    ("candidate_index", "field"),
    [
        (0, "primary_lower_bound_by_family"),
        (0, "median_by_family"),
        (0, "trimmed_mean_by_family"),
        (1, "adjacent_lower_bound_by_family"),
    ],
)
def test_complexity_first_gate_rejects_nonfinite_effect_statistics(
    candidate_index: int,
    field: str,
    value: float,
) -> None:
    families = ("MOKP", "MOTSP")
    summaries = _passing_intervention_summaries(families)
    bad_values = dict(getattr(summaries[candidate_index], field))
    bad_values["MOKP"] = value
    summaries[candidate_index] = replace(
        summaries[candidate_index],
        **{field: bad_values},
    )
    with pytest.raises(ValueError, match="finite"):
        complexity_first_selection(
            summaries,
            family_ids=families,
            primary_threshold=0.01,
            adjacent_threshold=0.01,
            simultaneous_coverage_certified=True,
        )


def test_complexity_first_gate_validates_not_reached_candidate_payloads() -> None:
    families = ("MOKP", "MOTSP")
    summaries = _passing_intervention_summaries(families)
    summaries[0] = replace(
        summaries[0],
        primary_lower_bound_by_family={family: 0.0 for family in families},
    )
    summaries[2] = replace(
        summaries[2],
        primary_lower_bound_by_family={
            "MOKP": float("inf"),
            "MOTSP": 0.04,
        },
    )
    with pytest.raises(ValueError, match="finite"):
        complexity_first_selection(
            summaries,
            family_ids=families,
            primary_threshold=0.01,
            adjacent_threshold=0.01,
            simultaneous_coverage_certified=True,
        )


def test_complexity_first_gate_preserves_all_pass_success_semantics() -> None:
    families = ("MOKP", "MOTSP")
    decision = complexity_first_selection(
        _passing_intervention_summaries(families),
        family_ids=families,
        primary_threshold=0.01,
        adjacent_threshold=0.01,
        simultaneous_coverage_certified=True,
    )
    assert decision.selected_candidate == "C3"
    assert decision.reached_candidates == ("C1", "C2", "C3")
    assert decision.not_reached_candidates == ()
    assert decision.reasons == ()


def test_complexity_first_gate_cannot_skip_failed_adjacent_mechanism() -> None:
    families = ("MOKP", "MOTSP")
    summaries = [
        InterventionSummary(
            candidate="C1",
            primary_lower_bound_by_family={f: 0.02 for f in families},
            adjacent_lower_bound_by_family=None,
            median_by_family={f: 0.02 for f in families},
            trimmed_mean_by_family={f: 0.02 for f in families},
            wins_by_family={f: 10 for f in families},
            losses_by_family={f: 2 for f in families},
        ),
        InterventionSummary(
            candidate="C2",
            primary_lower_bound_by_family={f: 0.03 for f in families},
            adjacent_lower_bound_by_family={f: 0.0 for f in families},
            median_by_family={f: 0.03 for f in families},
            trimmed_mean_by_family={f: 0.03 for f in families},
            wins_by_family={f: 10 for f in families},
            losses_by_family={f: 2 for f in families},
        ),
        InterventionSummary(
            candidate="C3",
            primary_lower_bound_by_family={f: 0.04 for f in families},
            adjacent_lower_bound_by_family={f: 0.04 for f in families},
            median_by_family={f: 0.04 for f in families},
            trimmed_mean_by_family={f: 0.04 for f in families},
            wins_by_family={f: 10 for f in families},
            losses_by_family={f: 2 for f in families},
        ),
    ]
    decision = complexity_first_selection(
        summaries,
        family_ids=families,
        primary_threshold=0.01,
        adjacent_threshold=0.01,
        simultaneous_coverage_certified=True,
    )
    assert decision.selected_candidate == "C1"
    assert decision.not_reached_candidates == ("C3",)

