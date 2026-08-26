from __future__ import annotations

import ast
import json
import sqlite3
import struct
from pathlib import Path

import pytest

from mo_nco.archive import ArchiveEntry, ParetoArchive
from mo_nco.instance import MultiObjectiveTSPInstance
from mo_nco.pareto_ijoc_problem import (
    MultiObjectiveKnapsackInstance,
    MultiObjectiveTSPProblemAdapter,
)
from mo_nco.pareto_v21_hybrid import (
    V21HybridConfig,
    V21TypedHybridParetoSearch,
)
from mo_nco.pareto_v21_trace_verify import verify_v21_trace_database


def test_v21_trace_is_a_complete_exact_budget_ledger() -> None:
    problem = MultiObjectiveKnapsackInstance.random_instance(
        12,
        num_objectives=2,
        seed=21001,
    )
    config = V21HybridConfig(
        candidate_id="C0",
        reference_directions=((0.1, 0.9), (0.5, 0.5), (0.9, 0.1)),
        evaluations=24,
        checkpoint_period=6,
        seed=17,
        phase="development",
    )

    run = V21TypedHybridParetoSearch(problem, config).run()

    assert run.optimization_result.metadata["algorithm"] == (
        "v21-typed-hybrid-pareto-search-c0"
    )
    assert run.optimization_result.metadata["exact_budget_gate"] == "PASS"
    assert run.optimization_result.metadata["all_evaluated_trace_complete"] is True
    assert len(run.trace) == config.evaluations
    assert tuple(event.evaluation_index for event in run.trace) == tuple(
        range(1, config.evaluations + 1)
    )
    assert tuple(item.iteration for item in run.optimization_result.diagnostics) == (
        6,
        12,
        18,
        24,
    )

    for event in run.trace:
        assert event.problem == problem.name
        assert event.seed == config.seed
        assert event.phase == config.phase
        assert event.operator
        assert event.proposal
        assert len(event.objectives) == problem.num_objectives
        assert event.elapsed_seconds >= 0.0

    rebuilt = ParetoArchive(max_size=None, tol=0.0)
    for event in run.trace:
        rebuilt.update((ArchiveEntry(event.proposal, event.objectives),))
    assert rebuilt.entries == run.optimization_result.archive.entries


def test_v21_c0_mokp_uses_full_native_operator_portfolio_and_repair() -> None:
    problem = MultiObjectiveKnapsackInstance.random_instance(18, seed=21002)
    config = V21HybridConfig(
        candidate_id="C0",
        reference_directions=((0.1, 0.9), (0.5, 0.5), (0.9, 0.1)),
        evaluations=30,
        checkpoint_period=10,
        seed=23,
        phase="development",
    )

    run = V21TypedHybridParetoSearch(problem, config).run()

    search_events = run.trace[len(config.reference_directions) :]
    assert {event.operator for event in search_events} == {
        "mokp_uniform_crossover_bit_mutation_greedy_repair_v1",
        "mokp_add_drop_greedy_repair_v1",
        "mokp_one_out_one_in_greedy_repair_v1",
        "mokp_bounded_density_local_improvement_greedy_repair_v1",
    }
    for event in search_events:
        problem.validate_solution(event.proposal)


def test_v21_c1_adds_typed_profit_density_initialization_only() -> None:
    problem = MultiObjectiveKnapsackInstance.random_instance(20, seed=21003)
    common = dict(
        reference_directions=((0.1, 0.9), (0.5, 0.5), (0.9, 0.1)),
        evaluations=30,
        checkpoint_period=10,
        seed=29,
        phase="development",
    )

    c0 = V21TypedHybridParetoSearch(
        problem,
        V21HybridConfig(candidate_id="C0", **common),
    ).run()
    c1 = V21TypedHybridParetoSearch(
        problem,
        V21HybridConfig(candidate_id="C1", **common),
    ).run()

    initial_count = len(common["reference_directions"])
    assert {event.operator for event in c0.trace[:initial_count]} == {
        "native_random_initialization_v1"
    }
    assert {event.operator for event in c1.trace[:initial_count]} == {
        "mokp_typed_profit_density_initialization_v1"
    }
    assert len({event.proposal for event in c1.trace[:initial_count]}) >= 2
    assert {
        event.operator for event in c1.trace[initial_count:]
    } == {
        "mokp_uniform_crossover_bit_mutation_greedy_repair_v1",
        "mokp_add_drop_greedy_repair_v1",
        "mokp_one_out_one_in_greedy_repair_v1",
        "mokp_bounded_density_local_improvement_greedy_repair_v1",
    }


def test_v21_c0_motsp_cycles_problem_native_local_search_portfolio() -> None:
    problem = MultiObjectiveTSPProblemAdapter(
        MultiObjectiveTSPInstance.random_biobjective(14, seed=21004)
    )
    config = V21HybridConfig(
        candidate_id="C0",
        reference_directions=((0.1, 0.9), (0.5, 0.5), (0.9, 0.1)),
        evaluations=30,
        checkpoint_period=10,
        seed=31,
        phase="development",
    )

    run = V21TypedHybridParetoSearch(problem, config).run()

    search_events = run.trace[len(config.reference_directions) :]
    assert {event.operator for event in search_events} == {
        "motsp_candidate_list_two_opt_v1",
        "motsp_relocate_v1",
        "motsp_restricted_three_opt_v1",
    }
    for event in search_events:
        problem.validate_solution(event.proposal)


def test_v21_c1_adds_weighted_nearest_neighbor_motsp_initialization() -> None:
    problem = MultiObjectiveTSPProblemAdapter(
        MultiObjectiveTSPInstance.random_biobjective(16, seed=21005)
    )
    config = V21HybridConfig(
        candidate_id="C1",
        reference_directions=((0.1, 0.9), (0.5, 0.5), (0.9, 0.1)),
        evaluations=30,
        checkpoint_period=10,
        seed=37,
        phase="development",
    )

    run = V21TypedHybridParetoSearch(problem, config).run()

    initial = run.trace[: len(config.reference_directions)]
    assert {event.operator for event in initial} == {
        "motsp_typed_weighted_nearest_neighbor_initialization_v1"
    }
    assert len({event.proposal for event in initial}) >= 2
    for event in initial:
        problem.validate_solution(event.proposal)


def test_v21_sqlite_trace_is_streamed_and_decisions_are_one_to_one(
    tmp_path,
) -> None:
    problem = MultiObjectiveKnapsackInstance.random_instance(24, seed=21006)
    trace_path = tmp_path / "trace.sqlite3"
    config = V21HybridConfig(
        candidate_id="C0",
        reference_directions=((0.1, 0.9), (0.5, 0.5), (0.9, 0.1)),
        evaluations=36,
        checkpoint_period=12,
        seed=41,
        phase="development",
        trace_database=str(trace_path),
        capture_trace=False,
    )

    run = V21TypedHybridParetoSearch(problem, config).run()

    assert run.trace == ()
    assert run.optimization_result.metadata["trace_store_status"] == "FINALIZED"
    assert run.optimization_result.metadata["physical_objective_calls"] == 36
    assert trace_path.is_file()
    with sqlite3.connect(trace_path) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute(
            "SELECT COUNT(*) FROM evaluations"
        ).fetchone() == (36,)
        assert connection.execute(
            "SELECT COUNT(*) FROM decisions"
        ).fetchone() == (36,)
        assert connection.execute(
            "SELECT MIN(evaluation_index), MAX(evaluation_index) FROM evaluations"
        ).fetchone() == (1, 36)
        assert connection.execute(
            "SELECT COUNT(DISTINCT evaluation_index) FROM evaluations"
        ).fetchone() == (36,)


def test_v21_c2_adds_periodic_typed_archive_diversification() -> None:
    problem = MultiObjectiveKnapsackInstance.random_instance(26, seed=21007)
    common = dict(
        reference_directions=((0.1, 0.9), (0.5, 0.5), (0.9, 0.1)),
        evaluations=36,
        checkpoint_period=12,
        seed=43,
        phase="development",
        diversification_period=4,
    )

    c1 = V21TypedHybridParetoSearch(
        problem,
        V21HybridConfig(candidate_id="C1", **common),
    ).run()
    c2 = V21TypedHybridParetoSearch(
        problem,
        V21HybridConfig(candidate_id="C2", **common),
    ).run()

    assert "typed_diversification" not in {
        event.search_phase for event in c1.trace
    }
    assert "typed_diversification" in {
        event.search_phase for event in c2.trace
    }
    assert set(c1.optimization_result.metadata["enabled_components"]) == {
        "native_backbone",
        "typed_initialization",
    }
    assert set(c2.optimization_result.metadata["enabled_components"]) == {
        "native_backbone",
        "typed_initialization",
        "typed_diversification",
    }


def test_v21_c3_adds_neighbor_path_relinking_with_two_parents(tmp_path) -> None:
    problem = MultiObjectiveKnapsackInstance.random_instance(28, seed=21008)
    trace_path = tmp_path / "c3.sqlite3"
    config = V21HybridConfig(
        candidate_id="C3",
        reference_directions=((0.1, 0.9), (0.5, 0.5), (0.9, 0.1)),
        evaluations=45,
        checkpoint_period=15,
        seed=47,
        phase="development",
        diversification_period=5,
        exchange_period=4,
        trace_database=str(trace_path),
    )

    run = V21TypedHybridParetoSearch(problem, config).run()

    assert "neighbor_path_relinking" in {
        event.search_phase for event in run.trace
    }
    assert "neighbor_path_relinking" in set(
        run.optimization_result.metadata["enabled_components"]
    )
    with sqlite3.connect(trace_path) as connection:
        parent_rows = connection.execute(
            """
            SELECT parent_solution_refs_json
            FROM evaluations
            WHERE search_phase_id='neighbor_path_relinking'
            """
        ).fetchall()
    assert parent_rows
    assert all(len(__import__("json").loads(row[0])) == 2 for row in parent_rows)


def test_v21_c4_allocates_operators_and_records_probabilities(tmp_path) -> None:
    problem = MultiObjectiveKnapsackInstance.random_instance(30, seed=21009)
    trace_path = tmp_path / "c4.sqlite3"
    config = V21HybridConfig(
        candidate_id="C4",
        reference_directions=((0.1, 0.9), (0.5, 0.5), (0.9, 0.1)),
        evaluations=48,
        checkpoint_period=16,
        seed=53,
        phase="development",
        trace_database=str(trace_path),
        operator_exploration=0.2,
    )

    run = V21TypedHybridParetoSearch(problem, config).run()

    assert "adaptive_operator_allocation" in set(
        run.optimization_result.metadata["enabled_components"]
    )
    with sqlite3.connect(trace_path) as connection:
        rows = connection.execute(
            """
            SELECT payload_json
            FROM mechanisms
            WHERE event_kind='operator_selection'
            ORDER BY event_index
            """
        ).fetchall()
    assert len(rows) == config.evaluations - len(config.reference_directions)
    for (raw,) in rows:
        payload = __import__("json").loads(raw)
        assert payload["chosen_operator"] in payload["available_operators"]
        assert abs(sum(payload["probabilities"]) - 1.0) <= 1e-12
        assert 0.0 <= payload["reward"] <= 1.0


def test_v21_c0_mokp_uses_moead_neighbor_parents_and_replacement(tmp_path) -> None:
    problem = MultiObjectiveKnapsackInstance.random_instance(36, seed=21010)
    trace_path = tmp_path / "c0_moead.sqlite3"
    directions = tuple(
        (index / 6.0 + 1e-6, 1.0 - index / 6.0 + 1e-6)
        for index in range(7)
    )
    directions = tuple(
        (left / (left + right), right / (left + right))
        for left, right in directions
    )
    config = V21HybridConfig(
        candidate_id="C0",
        reference_directions=directions,
        evaluations=84,
        checkpoint_period=21,
        seed=59,
        phase="development",
        trace_database=str(trace_path),
        neighborhood_size=4,
    )

    V21TypedHybridParetoSearch(problem, config).run()

    with sqlite3.connect(trace_path) as connection:
        parent_rows = connection.execute(
            """
            SELECT parent_solution_refs_json
            FROM evaluations
            WHERE operator_id='mokp_uniform_crossover_bit_mutation_greedy_repair_v1'
            """
        ).fetchall()
        maximum_replacements = connection.execute(
            "SELECT MAX(population_replacement_count) FROM decisions"
        ).fetchone()[0]
    assert parent_rows
    assert all(len(__import__("json").loads(row[0])) == 2 for row in parent_rows)
    assert maximum_replacements >= 2


def test_v21_trace_verifier_replays_objectives_and_hash_chains(tmp_path) -> None:
    problem = MultiObjectiveKnapsackInstance.random_instance(22, seed=21012)
    trace_path = tmp_path / "verified.sqlite3"
    config = V21HybridConfig(
        candidate_id="C2",
        reference_directions=((0.1, 0.9), (0.5, 0.5), (0.9, 0.1)),
        evaluations=42,
        checkpoint_period=14,
        seed=61,
        phase="development",
        trace_database=str(trace_path),
        capture_trace=False,
        diversification_period=5,
    )
    run = V21TypedHybridParetoSearch(problem, config).run()

    receipt = verify_v21_trace_database(
        trace_path,
        problem,
        expected_budget=config.evaluations,
        expected_archive=run.optimization_result.archive,
    )

    assert receipt["status"] == "PASS"
    assert receipt["unique_solution_replays"] <= config.evaluations
    assert receipt["evaluation_records"] == config.evaluations
    assert receipt["decision_records"] == config.evaluations
    assert receipt["archive_reconstruction"] == "PASS"


def test_v21_records_fixed_population_snapshots_for_d4(tmp_path) -> None:
    problem = MultiObjectiveKnapsackInstance.random_instance(24, seed=21013)
    trace_path = tmp_path / "snapshots.sqlite3"
    config = V21HybridConfig(
        candidate_id="C3",
        reference_directions=((0.1, 0.9), (0.5, 0.5), (0.9, 0.1)),
        evaluations=50,
        checkpoint_period=10,
        seed=67,
        phase="development",
        trace_database=str(trace_path),
    )

    V21TypedHybridParetoSearch(problem, config).run()

    with sqlite3.connect(trace_path) as connection:
        rows = connection.execute(
            """
            SELECT after_evaluation_index,payload_json
            FROM mechanisms
            WHERE event_kind='population_snapshot'
            ORDER BY after_evaluation_index
            """
        ).fetchall()
    assert tuple(row[0] for row in rows) == (3, 5, 35, 50)
    labels = [json.loads(row[1])["boundary_labels"] for row in rows]
    assert labels == [["init_end"], ["early_10pct"], ["mid_70pct"], ["budget_end"]]
    for _, raw in rows:
        payload = json.loads(raw)
        assert len(payload["population_solution_sha256"]) == 3
        assert 0.0 < payload["population_unique_fraction"] <= 1.0
        assert payload["resampling_ess_over_population"]["status"] == "NOT_APPLICABLE"
        assert payload["ancestor_multiplicity"]["status"] == "NOT_APPLICABLE"


def test_v21_algorithm_has_no_raw_problem_evaluation_bypass() -> None:
    source_path = Path(__file__).parents[1] / "mo_nco" / "pareto_v21_hybrid.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "evaluate"
    ]
    assert calls
    assert all(
        isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "_ledger"
        for node in calls
    )


def test_v21_optimizer_is_single_use() -> None:
    problem = MultiObjectiveKnapsackInstance.random_instance(12, seed=21014)
    optimizer = V21TypedHybridParetoSearch(
        problem,
        V21HybridConfig(
            candidate_id="C0",
            reference_directions=((0.2, 0.8), (0.8, 0.2)),
            evaluations=12,
            checkpoint_period=4,
            seed=71,
            phase="development",
        ),
    )
    optimizer.run()

    with pytest.raises(RuntimeError, match="single-use"):
        optimizer.run()


def test_v21_verifier_rejects_persisted_decision_tampering(tmp_path) -> None:
    problem = MultiObjectiveKnapsackInstance.random_instance(14, seed=21015)
    trace_path = tmp_path / "tampered.sqlite3"
    config = V21HybridConfig(
        candidate_id="C0",
        reference_directions=((0.2, 0.8), (0.8, 0.2)),
        evaluations=16,
        checkpoint_period=4,
        seed=73,
        phase="development",
        trace_database=str(trace_path),
        capture_trace=False,
    )
    V21TypedHybridParetoSearch(problem, config).run()
    with sqlite3.connect(trace_path) as connection:
        connection.execute(
            "UPDATE decisions SET decision_reason='tampered' WHERE evaluation_index=8"
        )
        connection.commit()

    with pytest.raises(ValueError, match="hash chain failed at evaluation 8"):
        verify_v21_trace_database(
            trace_path,
            problem,
            expected_budget=config.evaluations,
        )


def test_v21_motsp_backbone_expands_type_best_archive_parent(tmp_path) -> None:
    problem = MultiObjectiveTSPProblemAdapter(
        MultiObjectiveTSPInstance.random_biobjective(18, seed=21016)
    )
    trace_path = tmp_path / "motsp_archive_parent.sqlite3"
    directions = ((0.1, 0.9), (0.3, 0.7), (0.5, 0.5), (0.7, 0.3), (0.9, 0.1))
    config = V21HybridConfig(
        candidate_id="C0",
        reference_directions=directions,
        evaluations=45,
        checkpoint_period=9,
        seed=79,
        phase="development",
        trace_database=str(trace_path),
        capture_trace=False,
    )
    V21TypedHybridParetoSearch(problem, config).run()

    with sqlite3.connect(trace_path) as connection:
        connection.row_factory = sqlite3.Row
        solutions = {
            int(row["solution_ref"]): tuple(
                struct.unpack(
                    f"<{int(row['solution_size'])}H",
                    bytes(row["payload"]),
                )
            )
            for row in connection.execute(
                "SELECT solution_ref,solution_size,payload FROM solutions"
            )
        }
        rows = list(
            connection.execute(
                "SELECT * FROM evaluations ORDER BY evaluation_index"
            )
        )

    archive = ParetoArchive(max_size=None, tol=0.0)
    for row in rows:
        objective = tuple(json.loads(row["objectives_json"]))
        proposal = solutions[int(row["proposal_solution_ref"])]
        if row["search_phase_id"] == "native_backbone":
            type_index = int(row["type_id"])
            direction = directions[type_index]

            def score(entry):
                normalized = tuple(
                    (value - lower) / (upper - lower)
                    for value, lower, upper in zip(
                        entry.objectives,
                        problem.objective_lower_bounds,
                        problem.objective_upper_bounds,
                    )
                )
                return (
                    max(weight * value for weight, value in zip(direction, normalized)),
                    entry.objectives,
                    entry.tour,
                )

            expected_parent = min(archive.entries, key=score).tour
            parent_refs = json.loads(row["parent_solution_refs_json"])
            assert len(parent_refs) == 1
            assert solutions[int(parent_refs[0])] == expected_parent
        archive.update((ArchiveEntry(proposal, objective),))

