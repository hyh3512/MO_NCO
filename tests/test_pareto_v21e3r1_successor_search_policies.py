from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sqlite3

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
from mo_nco.pareto_v21e3r1_development_diagnostics import analyze_trace_database
import mo_nco.pareto_v21e3_hybrid as hybrid_module


LEGACY_SEARCH = "proposal_chain_v21e3r1_v1"
NEW_SEARCH = "post_commit_type_incumbent_anchor_development_v1"
LEGACY_NOVELTY = "legacy_retry_and_local_v21e3r1_v1"
NEW_NOVELTY = (
    "single_attempt_rotating_feasible_exchange_no_refill_development_v1"
)

POLICY_FIELDS = {
    "post_initialization_search_policy",
    "mokp_novelty_generation_policy",
}

EXACT_ARMS = {
    "MOKP_LEGACY": ("MOKP", LEGACY_SEARCH, LEGACY_NOVELTY),
    "MOKP_ANCHOR_ONLY": ("MOKP", NEW_SEARCH, LEGACY_NOVELTY),
    "MOKP_NOVELTY_ONLY": ("MOKP", LEGACY_SEARCH, NEW_NOVELTY),
    "MOKP_BOTH": ("MOKP", NEW_SEARCH, NEW_NOVELTY),
    "MOTSP_LEGACY": ("MOTSP", LEGACY_SEARCH, LEGACY_NOVELTY),
    "MOTSP_ANCHOR": ("MOTSP", NEW_SEARCH, LEGACY_NOVELTY),
}


def _mokp() -> MultiObjectiveKnapsackInstance:
    return MultiObjectiveKnapsackInstance(
        item_weights=(1, 2, 2, 3, 4, 5),
        profits_by_objective=(
            (9, 8, 7, 6, 5, 4),
            (4, 5, 6, 7, 8, 9),
        ),
        capacity=8,
        name="successor-policy-mokp",
    )


def _motsp() -> MultiObjectiveTSPProblemAdapter:
    return MultiObjectiveTSPProblemAdapter(
        MultiObjectiveTSPInstance.random_biobjective(8, seed=911)
    )


def _bit_transitions(
    source: tuple[int, ...], target: tuple[int, ...]
) -> tuple[int, int]:
    removed = sum(left == 1 and right == 0 for left, right in zip(source, target))
    added = sum(left == 0 and right == 1 for left, right in zip(source, target))
    return removed, added


def _config(
    *,
    arm: str | None = None,
    search_policy: str = LEGACY_SEARCH,
    novelty_policy: str = LEGACY_NOVELTY,
    trace_database: str | None = None,
    charged_evaluations: int = 2,
    local_improvement_steps: int = 2,
) -> V21E3HybridConfig:
    return V21E3HybridConfig(
        candidate_id="C0",
        reference_directions=((0.25, 0.75), (0.75, 0.25)),
        charged_evaluations=charged_evaluations,
        checkpoint_period=1,
        seed=919,
        phase="development",
        trace_database=trace_database,
        local_improvement_steps=local_improvement_steps,
        development_diagnostic_id=(
            None if arm is None else f"V21E3R1_SUCCESSOR_FACTORIAL_{arm}"
        ),
        post_initialization_search_policy=search_policy,
        mokp_novelty_generation_policy=novelty_policy,
    )


def test_successor_policy_fields_exist_with_legacy_defaults_without_legacy_payload_drift() -> None:
    available = {field.name for field in fields(V21E3HybridConfig)}
    assert POLICY_FIELDS <= available

    config = _config()
    assert config.post_initialization_search_policy == LEGACY_SEARCH
    assert config.mokp_novelty_generation_policy == LEGACY_NOVELTY
    assert POLICY_FIELDS.isdisjoint(config.semantic_payload())
    canonical = json.dumps(
        config.semantic_payload(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == (
        "b03c96f773c2be48216bd2dab7dee1f89d012eb3c0b56bc323f70f256cde5b9e"
    )


@pytest.mark.parametrize("arm", tuple(EXACT_ARMS))
def test_exact_successor_factorial_arm_binds_both_policy_fields(arm: str) -> None:
    family, search_policy, novelty_policy = EXACT_ARMS[arm]
    config = _config(
        arm=arm,
        search_policy=search_policy,
        novelty_policy=novelty_policy,
    )
    semantic = config.semantic_payload()

    assert semantic["post_initialization_search_policy"] == search_policy
    assert semantic["mokp_novelty_generation_policy"] == novelty_policy
    problem = _mokp() if family == "MOKP" else _motsp()
    V21E3TypedHybridParetoSearch(problem, config)


@pytest.mark.parametrize(
    ("arm", "search_policy", "novelty_policy"),
    (
        ("MOKP_LEGACY", NEW_SEARCH, LEGACY_NOVELTY),
        ("MOKP_BOTH", LEGACY_SEARCH, NEW_NOVELTY),
        ("MOTSP_ANCHOR", NEW_SEARCH, NEW_NOVELTY),
        ("UNKNOWN", LEGACY_SEARCH, LEGACY_NOVELTY),
    ),
)
def test_successor_factorial_arm_rejects_policy_or_identity_drift(
    arm: str,
    search_policy: str,
    novelty_policy: str,
) -> None:
    with pytest.raises(ValueError, match="successor factorial"):
        _config(
            arm=arm,
            search_policy=search_policy,
            novelty_policy=novelty_policy,
        )


def test_new_policy_is_rejected_outside_exact_named_successor_diagnostic() -> None:
    with pytest.raises(ValueError, match="successor factorial"):
        _config(search_policy=NEW_SEARCH)


def test_successor_family_mismatch_fails_before_creating_a_ledger(tmp_path: Path) -> None:
    trace = tmp_path / "must-not-exist.sqlite3"
    config = _config(
        arm="MOKP_NOVELTY_ONLY",
        search_policy=LEGACY_SEARCH,
        novelty_policy=NEW_NOVELTY,
        trace_database=str(trace),
    )

    with pytest.raises(ValueError, match="problem family"):
        V21E3TypedHybridParetoSearch(_motsp(), config)

    assert not trace.exists()


def test_post_commit_anchor_uses_type_incumbent_after_primary_rejection() -> None:
    problem = _mokp()
    config = _config(
        arm="MOKP_ANCHOR_ONLY",
        search_policy=NEW_SEARCH,
        charged_evaluations=4,
        local_improvement_steps=1,
    )
    optimizer = V21E3TypedHybridParetoSearch(problem, config)
    incumbent = (1, 1, 0, 0, 0, 0)
    other_type = (0, 0, 1, 1, 0, 0)
    rejected_primary = (0, 0, 0, 0, 0, 0)
    initial = (incumbent, other_type)
    optimizer._initial_solution = lambda type_index: (  # type: ignore[method-assign]
        initial[type_index],
        f"controlled_initial_{type_index}",
    )
    optimizer._scheduled_candidate = (  # type: ignore[method-assign]
        lambda type_index, search_slot: hybrid_module._GeneratedCandidate(
            rejected_primary,
            "controlled_rejected_primary",
            "native_backbone",
            (incumbent,),
            (type_index,),
        )
    )
    observed_anchors: list[tuple[int, ...]] = []
    original_local = optimizer._local_neighbor

    def observe_local(anchor: tuple[int, ...], type_index: int, depth: int):
        observed_anchors.append(anchor)
        return original_local(anchor, type_index, depth)

    optimizer._local_neighbor = observe_local  # type: ignore[method-assign]

    run = optimizer.run()

    primary = next(
        event for event in run.trace if event.operator == "controlled_rejected_primary"
    )
    assert primary.accepted_into_population is False
    assert observed_anchors[0] == incumbent
    assert observed_anchors[0] != rejected_primary


def test_post_commit_anchor_reloads_equal_replacement_before_next_depth() -> None:
    problem = MultiObjectiveKnapsackInstance(
        item_weights=(1, 1, 1, 2, 3, 4),
        profits_by_objective=(
            (1, 9, 9, 3, 2, 1),
            (1, 9, 9, 3, 2, 1),
        ),
        capacity=4,
        name="successor-equal-replacement-mokp",
    )
    config = _config(
        arm="MOKP_ANCHOR_ONLY",
        search_policy=NEW_SEARCH,
        charged_evaluations=5,
        local_improvement_steps=2,
    )
    optimizer = V21E3TypedHybridParetoSearch(problem, config)
    initial_type_zero = (1, 0, 0, 0, 0, 0)
    initial_type_one = (0, 0, 0, 1, 0, 0)
    accepted_primary = (0, 1, 0, 0, 0, 0)
    equal_replacement = (0, 0, 1, 0, 0, 0)
    final_local = (0, 1, 0, 1, 0, 0)
    initial = (initial_type_zero, initial_type_one)
    optimizer._initial_solution = lambda type_index: (  # type: ignore[method-assign]
        initial[type_index],
        f"controlled_initial_{type_index}",
    )
    optimizer._scheduled_candidate = (  # type: ignore[method-assign]
        lambda type_index, search_slot: hybrid_module._GeneratedCandidate(
            accepted_primary,
            "controlled_accepted_primary",
            "native_backbone",
            (initial_type_zero,),
            (type_index,),
        )
    )
    observed_anchors: list[tuple[int, ...]] = []

    def controlled_local(anchor: tuple[int, ...], type_index: int, depth: int):
        observed_anchors.append(anchor)
        return (
            equal_replacement if depth == 1 else final_local,
            f"controlled_local_depth_{depth}",
        )

    optimizer._local_neighbor = controlled_local  # type: ignore[method-assign]

    run = optimizer.run()

    equal_event = next(
        event for event in run.trace if event.operator == "controlled_local_depth_1"
    )
    assert equal_event.accepted_into_population is True
    assert equal_event.objectives == problem.evaluate(accepted_primary)
    assert observed_anchors == [accepted_primary, equal_replacement]


def test_mokp_successor_novelty_rotates_per_type_and_origin_without_refill() -> None:
    problem = _mokp()
    config = _config(
        arm="MOKP_NOVELTY_ONLY",
        novelty_policy=NEW_NOVELTY,
    )
    optimizer = V21E3TypedHybridParetoSearch(problem, config)
    source = (1, 1, 0, 0, 0, 0)

    local = [
        optimizer._mokp_successor_novelty_candidate(  # type: ignore[attr-defined]
            source,
            0,
            origin="bounded_local_improvement",
        )
        for _ in range(3)
    ]
    modes = [witness["rotation_mode"] for _solution, _operator, witness in local]
    ordinals = [
        witness["origin_call_ordinal_by_type"]
        for _solution, _operator, witness in local
    ]
    assert modes == ["add", "drop", "swap"]
    assert ordinals == [1, 2, 3]
    assert [_bit_transitions(source, item[0]) for item in local] == [
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    for solution, operator, witness in local:
        problem.validate_solution(solution)
        assert operator.startswith("mokp_successor_local_")
        assert "no_refill_repair_development_v1" in operator
        assert witness == {
            "schema": "v21e3r1_mokp_successor_novelty_witness_v1",
            "policy": NEW_NOVELTY,
            "origin": "bounded_local_improvement",
            "origin_call_ordinal_by_type": witness[
                "origin_call_ordinal_by_type"
            ],
            "rotation_mode": witness["rotation_mode"],
            "move_applied": True,
            "removed_item_indices": witness["removed_item_indices"],
            "added_item_indices": witness["added_item_indices"],
            "repair_refill": False,
            "rng_draws_consumed": 0,
        }

    retry_first = optimizer._mokp_successor_novelty_candidate(  # type: ignore[attr-defined]
        source,
        0,
        origin="post_initialization_duplicate_retry",
    )
    other_type_first = optimizer._mokp_successor_novelty_candidate(  # type: ignore[attr-defined]
        source,
        1,
        origin="bounded_local_improvement",
    )
    assert retry_first[2]["rotation_mode"] == "add"
    assert retry_first[2]["origin_call_ordinal_by_type"] == 1
    assert other_type_first[2]["rotation_mode"] == "add"
    assert other_type_first[2]["origin_call_ordinal_by_type"] == 1


def test_mokp_successor_local_dispatch_is_conditional_and_legacy_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (1, 1, 0, 0, 0, 0)
    successor = V21E3TypedHybridParetoSearch(
        _mokp(),
        _config(
            arm="MOKP_NOVELTY_ONLY",
            novelty_policy=NEW_NOVELTY,
        ),
    )
    monkeypatch.setattr(
        successor,
        "_local_neighbor",
        lambda *_args: pytest.fail("successor MOKP local dispatch reached legacy code"),
    )

    proposal, operator, witness = successor._local_candidate(  # type: ignore[attr-defined]
        source, 0, 1
    )

    assert _bit_transitions(source, proposal) == (0, 1)
    assert operator.startswith("mokp_successor_local_add_")
    assert witness["origin"] == "bounded_local_improvement"

    legacy = V21E3TypedHybridParetoSearch(_mokp(), _config())
    expected = legacy._local_neighbor(source, 0, 1)
    assert legacy._local_candidate(source, 0, 1) == (  # type: ignore[attr-defined]
        expected[0],
        expected[1],
        None,
    )


def test_initialization_retry_stays_legacy_and_successor_noop_reaches_ledger() -> None:
    problem = MultiObjectiveKnapsackInstance(
        item_weights=(1, 1, 1, 1),
        profits_by_objective=((8, 7, 6, 5), (5, 6, 7, 8)),
        capacity=4,
        name="successor-noop-ledger-mokp",
    )
    config = _config(
        arm="MOKP_NOVELTY_ONLY",
        novelty_policy=NEW_NOVELTY,
        charged_evaluations=4,
        local_improvement_steps=1,
    )
    optimizer = V21E3TypedHybridParetoSearch(problem, config)
    full = (1, 1, 1, 1)
    common = {
        "search_slot": 0,
        "search_phase": "matched_construction",
        "local_search_block_id": None,
        "local_search_depth": 0,
        "construction_variant": 0,
        "generation_parents": (),
        "generation_parent_type_ids": (),
    }
    optimizer._charge_unique(
        type_index=0,
        operator="controlled_initial_0",
        proposal=full,
        parent=None,
        **common,
    )
    optimizer._charge_unique(
        type_index=1,
        operator="controlled_initial_1",
        proposal=full,
        parent=None,
        **{**common, "construction_variant": 1},
    )
    optimizer._charge_unique(
        type_index=0,
        search_slot=1,
        search_phase="native_backbone",
        operator="controlled_duplicate_primary",
        proposal=full,
        parent=full,
        local_search_block_id=1,
        local_search_depth=0,
        construction_variant=None,
        generation_parents=(full,),
        generation_parent_type_ids=(0,),
    )

    rows = list(
        optimizer._ledger._connection.execute(  # type: ignore[attr-defined]
            "SELECT status,context_json FROM attempts ORDER BY attempt_index"
        )
    )
    decoded = [(str(status), json.loads(str(raw))) for status, raw in rows]
    initialization_retries = [
        (status, context)
        for status, context in decoded
        if context["stage_id"] == "initialization_v21e3"
        and context["operator_witness"]["retry_ordinal"] > 0
    ]
    assert initialization_retries
    assert all(
        context["operator_id"] == "duplicate_retry_perturbation_v21e3"
        and "successor_mokp_novelty" not in context["operator_witness"]
        for _status, context in initialization_retries
    )

    successor_attempts = [
        (status, context)
        for status, context in decoded
        if "successor_mokp_novelty" in context["operator_witness"]
    ]
    assert successor_attempts
    first_status, first_context = successor_attempts[0]
    first_witness = first_context["operator_witness"]["successor_mokp_novelty"]
    assert first_status == "CACHE_HIT"
    assert first_witness["origin"] == "post_initialization_duplicate_retry"
    assert first_witness["origin_call_ordinal_by_type"] == 1
    assert first_witness["rotation_mode"] == "add"
    assert first_witness["move_applied"] is False
    assert first_context["operator_witness"]["retry_ordinal"] > 0

    successor_witnesses = [
        context["operator_witness"]["successor_mokp_novelty"]
        for _status, context in successor_attempts
    ]
    successor_ordinals = [
        witness["origin_call_ordinal_by_type"] for witness in successor_witnesses
    ]
    assert successor_ordinals == list(range(1, len(successor_attempts) + 1))
    assert all(witness["repair_refill"] is False for witness in successor_witnesses)
    assert all(witness["rng_draws_consumed"] == 0 for witness in successor_witnesses)


def test_development_analyzer_counts_each_operator_charge_once(tmp_path: Path) -> None:
    problem = _mokp()
    trace = tmp_path / "trace.sqlite3"
    budget = 12
    config = V21E3HybridConfig(
        candidate_id="C0",
        reference_directions=((0.25, 0.75), (0.75, 0.25)),
        charged_evaluations=budget,
        checkpoint_period=4,
        seed=929,
        phase="development",
        trace_database=str(trace),
        receipt_database_path="trace.sqlite3",
        local_improvement_steps=1,
    )
    V21E3TypedHybridParetoSearch(problem, config).run()

    analysis = analyze_trace_database(
        trace,
        row={
            "case_id": problem.name,
            "family": "MOKP",
            "size": problem.solution_size,
            "seed": config.seed,
            "arm_id": "C0_STANDARD",
            "charged_evaluation_budget": budget,
            "normalized_left_continuous_hv_auc": 0.0,
            "normalized_terminal_hv": 0.0,
        },
        lower=problem.objective_lower_bounds,
        upper=problem.objective_upper_bounds,
    )
    operators = analysis["operators"]

    assert sum(item["charged_evaluations"] for item in operators.values()) == budget
    assert sum(item["physical_starts"] for item in operators.values()) == budget
    assert all(
        item["charged_evaluations"] == item["physical_starts"]
        for item in operators.values()
    )
    construction = next(
        item for name, item in operators.items() if "construction" in name
    )
    assert construction["accepted_rate_per_charge"] == 1.0


def test_development_analyzer_rejects_evaluation_attempt_charge_disagreement(
    tmp_path: Path,
) -> None:
    problem = _mokp()
    trace = tmp_path / "trace.sqlite3"
    budget = 8
    config = V21E3HybridConfig(
        candidate_id="C0",
        reference_directions=((0.25, 0.75), (0.75, 0.25)),
        charged_evaluations=budget,
        checkpoint_period=4,
        seed=937,
        phase="development",
        trace_database=str(trace),
        receipt_database_path="trace.sqlite3",
        local_improvement_steps=1,
    )
    V21E3TypedHybridParetoSearch(problem, config).run()
    with sqlite3.connect(trace) as connection:
        connection.execute(
            "UPDATE attempts SET charged_evaluation_index=NULL "
            "WHERE attempt_index=(SELECT MIN(attempt_index) FROM attempts "
            "WHERE charged_evaluation_index IS NOT NULL)"
        )
        connection.commit()

    with pytest.raises(
        RuntimeError,
        match="Operator evaluation and attempt charge accounting disagree",
    ):
        analyze_trace_database(
            trace,
            row={
                "case_id": problem.name,
                "family": "MOKP",
                "size": problem.solution_size,
                "seed": config.seed,
                "arm_id": "C0_STANDARD",
                "charged_evaluation_budget": budget,
                "normalized_left_continuous_hv_auc": 0.0,
                "normalized_terminal_hv": 0.0,
            },
            lower=problem.objective_lower_bounds,
            upper=problem.objective_upper_bounds,
        )

