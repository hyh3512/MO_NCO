from __future__ import annotations

import json
import pytest

from mo_nco.pareto_v21e3_hybrid import (
    V21E3HybridConfig,
    V21E3RegionOccupancy,
    V21E3TypedHybridParetoSearch,
    v21e3_candidate_spec,
    v21e3_schedule_slot,
)
from mo_nco.instance import MultiObjectiveTSPInstance
from mo_nco.pareto_ijoc_problem import MultiObjectiveTSPProblemAdapter
from mo_nco.pareto_ijoc_problem import MultiObjectiveKnapsackInstance


def test_v21e3_rejects_the_v21_formal_confirmation_phase_label() -> None:
    with pytest.raises(ValueError, match="Unsupported evidence phase"):
        V21E3HybridConfig(
            candidate_id="C0",
            reference_directions=((0.5, 0.5),),
            charged_evaluations=1,
            checkpoint_period=1,
            seed=1,
            phase="formal_confirmation",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("phase", ["calibration", "calibration_confirmation"])
def test_v21e3_calibration_requires_case_and_source_snapshot_bindings(
    phase: str,
) -> None:
    common = dict(
        candidate_id="C0",
        reference_directions=((0.5, 0.5),),
        charged_evaluations=1,
        checkpoint_period=1,
        seed=1,
        phase=phase,
    )
    with pytest.raises(ValueError, match="explicit case-artifact and source-snapshot"):
        V21E3HybridConfig(**common)  # type: ignore[arg-type]

    config = V21E3HybridConfig(
        **common,  # type: ignore[arg-type]
        case_artifact_sha256="a" * 64,
        source_snapshot_sha256="b" * 64,
    )
    assert config.case_artifact_sha256 == "a" * 64
    assert config.source_snapshot_sha256 == "b" * 64


def test_v21e3_c0_c1_are_matched_except_for_direction_conditioning() -> None:
    c0 = v21e3_candidate_spec("C0")
    c1 = v21e3_candidate_spec("C1")

    assert c0.construction_portfolio == c1.construction_portfolio
    assert c0.native_portfolio == c1.native_portfolio
    assert c0.local_improvement_contract == c1.local_improvement_contract
    assert c0.replacement_contract == c1.replacement_contract
    assert c0.direction_policy == "central_untyped_direction_v1"
    assert c1.direction_policy == "reference_type_direction_v1"
    assert c0.enabled_components == ("strong_native_backbone",)
    assert c1.enabled_components == (
        "strong_native_backbone",
        "direction_conditioning",
    )


def test_v21e3_c2_c3_share_schedule_and_use_matched_exchange_contrast() -> None:
    c2 = v21e3_candidate_spec("C2")
    c3 = v21e3_candidate_spec("C3")

    assert c2.diversification_schedule == c3.diversification_schedule
    assert c2.exchange_schedule == c3.exchange_schedule
    assert c2.exchange_effort_units == c3.exchange_effort_units
    assert c2.exchange_operator == "matched_exchange_control_v1"
    assert c3.exchange_operator == "neighbor_path_relinking_v1"
    assert c2.enabled_components == (
        "strong_native_backbone",
        "direction_conditioning",
        "typed_diversification",
        "matched_exchange_control",
    )
    assert c3.enabled_components[:-1] == c2.enabled_components[:-1]
    assert c3.enabled_components[-1] == "neighbor_path_relinking"


def test_v21e3_c2_c3_have_identical_diversification_and_exchange_slots() -> None:
    c2 = [
        v21e3_schedule_slot("C2", slot, diversification_period=16, exchange_period=11)
        for slot in range(1, 353)
    ]
    c3 = [
        v21e3_schedule_slot("C3", slot, diversification_period=16, exchange_period=11)
        for slot in range(1, 353)
    ]

    assert [item.slot for item in c2 if item.kind == "diversification"] == [
        item.slot for item in c3 if item.kind == "diversification"
    ]
    assert [item.slot for item in c2 if item.kind == "exchange"] == [
        item.slot for item in c3 if item.kind == "exchange"
    ]
    assert all(
        left.kind == right.kind and left.effort_units == right.effort_units
        for left, right in zip(c2, c3)
    )
    assert v21e3_schedule_slot(
        "C2", 176, diversification_period=16, exchange_period=11
    ).kind == "exchange"


def test_v21e3_motsp_native_portfolio_is_balanced_within_every_type() -> None:
    problem = MultiObjectiveTSPProblemAdapter(
        MultiObjectiveTSPInstance.random_biobjective(14, seed=31001)
    )
    directions = (
        (0.05, 0.95),
        (0.20, 0.80),
        (0.35, 0.65),
        (0.65, 0.35),
        (0.80, 0.20),
        (0.95, 0.05),
    )
    config = V21E3HybridConfig(
        candidate_id="C0",
        reference_directions=directions,
        charged_evaluations=54,
        checkpoint_period=9,
        seed=31003,
        phase="development",
        local_improvement_steps=1,
    )

    run = V21E3TypedHybridParetoSearch(problem, config).run()

    expected = {
        "motsp_candidate_list_two_opt_v21e3",
        "motsp_relocate_v21e3",
        "motsp_restricted_three_opt_v21e3",
    }
    for type_index in range(len(directions)):
        operators = [
            event.operator
            for event in run.attempts
            if event.type_index == type_index
            and event.search_phase == "native_backbone"
            and event.retry_ordinal == 0
        ]
        assert set(operators) == expected
        counts = [operators.count(operator) for operator in sorted(expected)]
        assert max(counts) - min(counts) <= 1


def test_v21e3_duplicate_attempts_do_not_consume_true_evaluation_budget() -> None:
    problem = MultiObjectiveKnapsackInstance.random_instance(14, seed=31005)
    config = V21E3HybridConfig(
        candidate_id="C0",
        reference_directions=((0.1, 0.9), (0.5, 0.5), (0.9, 0.1)),
        charged_evaluations=24,
        checkpoint_period=6,
        seed=31007,
        phase="development",
        local_improvement_steps=1,
        duplicate_retry_cap=4,
        fallback_attempt_cap=16,
    )

    run = V21E3TypedHybridParetoSearch(problem, config).run()
    metadata = run.optimization_result.metadata

    assert metadata["exact_charged_budget_gate"] == "PASS"
    assert metadata["charged_evaluation_count"] == 24
    assert metadata["unique_true_evaluation_count"] == 24
    assert metadata["physical_objective_call_count"] == 24
    assert metadata["attempt_count"] > 24
    assert metadata["cache_hit_count"] == metadata["attempt_count"] - 24
    assert metadata["retry_count"] > 0
    assert len({event.proposal_sha256 for event in run.trace}) == 24
    cache_hits = [event for event in run.attempts if event.cache_hit]
    assert cache_hits
    assert all(event.charged_evaluation_index is None for event in cache_hits)


def test_v21e3_duplicate_retry_cap_uses_frozen_fallback_without_overcharge() -> None:
    problem = MultiObjectiveKnapsackInstance.random_instance(8, seed=1)
    config = V21E3HybridConfig(
        candidate_id="C0",
        reference_directions=((0.2, 0.8), (0.5, 0.5), (0.8, 0.2)),
        charged_evaluations=20,
        checkpoint_period=5,
        seed=1,
        phase="development",
        local_improvement_steps=1,
        duplicate_retry_cap=0,
        fallback_attempt_cap=30,
    )

    run = V21E3TypedHybridParetoSearch(problem, config).run()
    metadata = run.optimization_result.metadata

    assert metadata["fallback_count"] > 0
    assert metadata["retry_count"] == 0
    assert metadata["charged_evaluation_count"] == 20
    assert metadata["physical_objective_call_count"] == 20
    assert metadata["attempt_count"] > 20
    assert any(event.fallback_used for event in run.attempts)


def test_v21e3_unique_state_exhaustion_writes_terminal_failure_receipt(
    tmp_path,
) -> None:
    problem = MultiObjectiveKnapsackInstance(
        item_weights=(1, 1),
        profits_by_objective=((1, 2), (2, 1)),
        capacity=1,
        name="three-state-mokp",
    )
    receipt_path = tmp_path / "terminal.json"
    config = V21E3HybridConfig(
        candidate_id="C0",
        reference_directions=((0.2, 0.8), (0.8, 0.2)),
        charged_evaluations=4,
        checkpoint_period=2,
        seed=31008,
        phase="development",
        trace_database=str(tmp_path / "trace.sqlite3"),
        terminal_receipt=str(receipt_path),
        local_improvement_steps=1,
        duplicate_retry_cap=1,
        fallback_attempt_cap=2,
    )

    with pytest.raises(RuntimeError, match="retry/fallback cap exhausted"):
        V21E3TypedHybridParetoSearch(problem, config).run()

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "FAILURE"
    assert receipt["failure_code"] == "FINALIZATION_GATE_FAILED"
    assert receipt["charged_evaluation_count"] < 4
    assert receipt["decision_count"] == receipt["charged_evaluation_count"]


@pytest.mark.parametrize("family", ["MOKP", "MOTSP"])
def test_v21e3_c0_c1_use_same_unique_construction_variants(family: str) -> None:
    if family == "MOKP":
        problem = MultiObjectiveKnapsackInstance.random_instance(40, seed=31009)
    else:
        problem = MultiObjectiveTSPProblemAdapter(
            MultiObjectiveTSPInstance.random_biobjective(18, seed=31009)
        )
    directions = (
        (0.1, 0.9),
        (0.3, 0.7),
        (0.5, 0.5),
        (0.7, 0.3),
        (0.9, 0.1),
    )
    common = dict(
        reference_directions=directions,
        charged_evaluations=len(directions),
        checkpoint_period=len(directions),
        seed=31011,
        phase="development",
    )

    c0 = V21E3TypedHybridParetoSearch(
        problem,
        V21E3HybridConfig(candidate_id="C0", **common),
    ).run()
    c1 = V21E3TypedHybridParetoSearch(
        problem,
        V21E3HybridConfig(candidate_id="C1", **common),
    ).run()

    assert [event.operator for event in c0.trace] == [
        event.operator for event in c1.trace
    ]
    assert [event.construction_variant for event in c0.trace] == list(
        range(len(directions))
    )
    assert [event.construction_variant for event in c1.trace] == list(
        range(len(directions))
    )
    assert len({event.proposal_sha256 for event in c0.trace}) == len(directions)
    assert c0.optimization_result.metadata["cache_hit_count"] == 0
    assert all(event.effective_direction == (0.5, 0.5) for event in c0.trace)
    assert [event.effective_direction for event in c1.trace] == list(directions)


@pytest.mark.parametrize("family", ["MOKP", "MOTSP"])
def test_v21e3_executes_bounded_multistep_local_search_and_neighbor_replacement(
    family: str,
) -> None:
    if family == "MOKP":
        problem = MultiObjectiveKnapsackInstance.random_instance(40, seed=31013)
    else:
        problem = MultiObjectiveTSPProblemAdapter(
            MultiObjectiveTSPInstance.random_biobjective(18, seed=31013)
        )
    config = V21E3HybridConfig(
        candidate_id="C1",
        reference_directions=(
            (0.1, 0.9),
            (0.3, 0.7),
            (0.5, 0.5),
            (0.7, 0.3),
            (0.9, 0.1),
        ),
        charged_evaluations=50,
        checkpoint_period=10,
        seed=31015,
        phase="development",
        local_improvement_steps=2,
        neighborhood_size=4,
    )

    run = V21E3TypedHybridParetoSearch(problem, config).run()
    local = [
        event
        for event in run.trace
        if event.search_phase == "bounded_local_improvement"
    ]

    assert local
    complete_blocks = {}
    for event in local:
        complete_blocks.setdefault(event.local_search_block_id, set()).add(
            event.local_search_depth
        )
    assert any(depths == {1, 2} for depths in complete_blocks.values())
    assert all(event.charged_evaluation_index is not None for event in local)
    search = [event for event in run.trace if event.search_slot > 0]
    assert search
    assert all(len(event.population_considered_type_ids) == 4 for event in search)
    assert all(len(event.population_target_type_ids) <= 4 for event in run.trace)


def test_v21e3_c2_c3_execute_the_same_slots_with_one_exchange_draw_each() -> None:
    problem = MultiObjectiveKnapsackInstance.random_instance(44, seed=31017)
    common = dict(
        reference_directions=(
            (0.1, 0.9),
            (0.3, 0.7),
            (0.5, 0.5),
            (0.7, 0.3),
            (0.9, 0.1),
        ),
        charged_evaluations=90,
        checkpoint_period=15,
        seed=31019,
        phase="development",
        local_improvement_steps=1,
        diversification_period=4,
        exchange_period=3,
    )

    c2 = V21E3TypedHybridParetoSearch(
        problem,
        V21E3HybridConfig(candidate_id="C2", **common),
    ).run()
    c3 = V21E3TypedHybridParetoSearch(
        problem,
        V21E3HybridConfig(candidate_id="C3", **common),
    ).run()

    def base_attempts(run):
        return [
            event
            for event in run.attempts
            if event.search_slot > 0
            and event.search_phase != "bounded_local_improvement"
            and event.retry_ordinal == 0
        ]

    c2_base = base_attempts(c2)
    c3_base = base_attempts(c3)
    assert [event.search_slot for event in c2_base] == [
        event.search_slot for event in c3_base
    ]
    assert [
        event.search_slot
        for event in c2_base
        if event.search_phase == "typed_diversification"
    ] == [
        event.search_slot
        for event in c3_base
        if event.search_phase == "typed_diversification"
    ]
    assert [
        event.search_slot
        for event in c2_base
        if event.search_phase == "matched_exchange_control"
    ] == [
        event.search_slot
        for event in c3_base
        if event.search_phase == "neighbor_path_relinking"
    ]
    c2_meta = c2.optimization_result.metadata
    c3_meta = c3.optimization_result.metadata
    assert c2_meta["exchange_operator_call_count"] == c3_meta[
        "exchange_operator_call_count"
    ]
    assert c2_meta["exchange_random_draw_count"] == c2_meta[
        "exchange_operator_call_count"
    ]
    assert c3_meta["exchange_random_draw_count"] == c3_meta[
        "exchange_operator_call_count"
    ]
    assert all(
        len(event.generation_parent_type_ids) == 2
        for event in c2_base
        if event.search_phase == "matched_exchange_control"
    )
    assert all(
        len(event.generation_parent_type_ids) == 2
        for event in c3_base
        if event.search_phase == "neighbor_path_relinking"
    )


def test_v21e3_nondominated_region_is_independent_of_evaluated_region() -> None:
    tracker = V21E3RegionOccupancy(
        lower_bounds=(0.0, 0.0),
        upper_bounds=(1.0, 1.0),
        bins=20,
    )
    dominating = (0.49, 0.54)
    dominated = (0.54, 0.54)
    later_nondominated = (0.50, 0.50)

    first = tracker.observe(dominating, nondominated=(dominating,))
    dominated_event = tracker.observe(dominated, nondominated=(dominating,))
    nondominated_event = tracker.observe(
        later_nondominated,
        nondominated=(dominating, later_nondominated),
    )

    assert first.new_evaluated_cell is True
    assert first.new_nondominated_cell is True
    assert dominated_event.region == nondominated_event.region
    assert dominated_event.new_evaluated_cell is True
    assert dominated_event.new_nondominated_cell is False
    assert nondominated_event.new_evaluated_cell is False
    assert nondominated_event.new_nondominated_cell is True
    assert tracker.evaluated_region_count == 2
    assert tracker.nondominated_region_count == 2


def _v21e3r1_mokp_operator_fixture(
    *,
    source=(1, 1, 0, 0),
    capacity: int = 3,
) -> tuple[
    MultiObjectiveKnapsackInstance,
    V21E3TypedHybridParetoSearch,
]:
    problem = MultiObjectiveKnapsackInstance(
        item_weights=(1, 1, 1, 1),
        profits_by_objective=((9, 7, 6, 5), (5, 6, 8, 9)),
        capacity=capacity,
        name="mokp-operator-separation",
    )
    config = V21E3HybridConfig(
        candidate_id="C1",
        reference_directions=((0.5, 0.5),),
        charged_evaluations=1,
        checkpoint_period=1,
        seed=991,
        phase="development",
    )
    optimizer = V21E3TypedHybridParetoSearch(problem, config)
    optimizer._solutions[0] = source
    optimizer._objectives[0] = problem.evaluate(source)
    return problem, optimizer


def _bit_transitions(source, target) -> tuple[int, int]:
    removed = sum(left == 1 and right == 0 for left, right in zip(source, target))
    added = sum(left == 0 and right == 1 for left, right in zip(source, target))
    return removed, added


def test_v21e3r1_mokp_add_drop_and_swap_have_distinct_support_and_labels() -> None:
    source = (1, 1, 0, 0)
    _, optimizer = _v21e3r1_mokp_operator_fixture(source=source)

    added, add_label, _, _ = optimizer._mokp_native(0, 1, operator_call=1)
    dropped, drop_label, _, _ = optimizer._mokp_native(0, 1, operator_call=5)
    swapped, swap_label, _, _ = optimizer._mokp_native(0, 2, operator_call=2)

    assert _bit_transitions(source, added) == (0, 1)
    assert _bit_transitions(source, dropped) == (1, 0)
    assert _bit_transitions(source, swapped) == (1, 1)
    assert len({added, dropped, swapped}) == 3
    assert add_label == "mokp_add_repair_v21e3r1"
    assert drop_label == "mokp_drop_repair_v21e3r1"
    assert swap_label == "mokp_swap_repair_v21e3"


def test_v21e3r1_mokp_drop_repair_does_not_refill_the_removed_bit() -> None:
    source = (1, 1, 0, 0)
    _, optimizer = _v21e3r1_mokp_operator_fixture(source=source)

    optimizer._mokp_native(0, 1, operator_call=1)
    dropped, label, _, _ = optimizer._mokp_native(0, 1, operator_call=5)

    assert sum(dropped) == sum(source) - 1
    assert _bit_transitions(source, dropped) == (1, 0)
    assert label == "mokp_drop_repair_v21e3r1"


def test_v21e3r1_mokp_each_type_cycles_over_four_independent_native_arms() -> None:
    problem = MultiObjectiveKnapsackInstance(
        item_weights=(1, 1, 1, 1, 1, 1),
        profits_by_objective=((9, 8, 7, 6, 5, 4), (4, 5, 6, 7, 8, 9)),
        capacity=4,
        name="mokp-per-type-native-cycle",
    )
    directions = ((0.2, 0.8), (0.5, 0.5), (0.8, 0.2))
    optimizer = V21E3TypedHybridParetoSearch(
        problem,
        V21E3HybridConfig(
            candidate_id="C1",
            reference_directions=directions,
            charged_evaluations=3,
            checkpoint_period=3,
            seed=993,
            phase="development",
        ),
    )
    source = (1, 1, 0, 0, 0, 0)
    for type_index in range(len(directions)):
        optimizer._solutions[type_index] = source
        optimizer._objectives[type_index] = problem.evaluate(source)

    for type_index in range(len(directions)):
        generated = [optimizer._native_candidate(type_index) for _ in range(4)]
        labels = [candidate.operator for candidate in generated]
        assert labels[0] == "mokp_uniform_crossover_mutation_repair_v21e3"
        assert labels[1].startswith("mokp_add_")
        assert labels[2] == "mokp_swap_repair_v21e3"
        assert labels[3] == "mokp_multibit_repair_v21e3"
        assert _bit_transitions(source, generated[1].solution) == (0, 1)
        assert _bit_transitions(source, generated[2].solution) == (1, 1)


def test_v21e3r1_mokp_unavailable_move_uses_truthful_noop_label() -> None:
    full = (1, 1, 1, 1)
    _, optimizer = _v21e3r1_mokp_operator_fixture(source=full, capacity=4)

    add_result, add_label, _, _ = optimizer._mokp_native(
        0, 1, operator_call=1
    )
    swap_result, swap_label, _, _ = optimizer._mokp_native(
        0, 2, operator_call=2
    )

    assert add_result == full
    assert add_label == "mokp_add_noop_no_feasible_item_v21e3r1"
    assert swap_result == full
    assert swap_label == "mokp_swap_noop_no_feasible_exchange_v21e3r1"

    empty = (0, 0, 0, 0)
    _, empty_optimizer = _v21e3r1_mokp_operator_fixture(source=empty, capacity=4)
    empty_optimizer._mokp_native(0, 1, operator_call=1)
    drop_result, drop_label, _, _ = empty_optimizer._mokp_native(
        0, 1, operator_call=5
    )
    assert drop_result == empty
    assert drop_label == "mokp_drop_noop_empty_solution_v21e3r1"


def test_v21e3r1_hybrid_materializes_authoritative_v2_run_context() -> None:
    _, optimizer = _v21e3r1_mokp_operator_fixture()
    payload = optimizer._ledger._run_context.payload
    algorithm_config = payload["algorithm_config"]

    assert payload["schema"] == "v21e3r1_run_context_v2"
    assert payload["candidate_id"] == algorithm_config["candidate_id"]
    assert payload["seed"] == algorithm_config["seed"]
    assert payload["charged_evaluation_budget"] == algorithm_config[
        "charged_evaluations"
    ]
    assert payload["evidence_partition"] == algorithm_config["phase"]
    assert payload["reference_directions"] == algorithm_config[
        "reference_directions"
    ]

