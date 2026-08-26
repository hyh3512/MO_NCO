from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from mo_nco.baselines import MOEADOptimizer, NSGAIIOptimizer
from mo_nco.evaluation import CountingTSPInstance
from mo_nco.ijoc_mokp_baselines import (
    BinaryMOEADMOKPBaseline,
    BinaryNSGA2MOKPBaseline,
)
from mo_nco.instance import MultiObjectiveTSPInstance
from mo_nco.moves import two_opt_at
from mo_nco.pareto_v21e3_artifacts import ArtifactRoot
from mo_nco.pareto_v21e3_common_runner import (
    preflight_development_parity_protocol_v2,
)
from mo_nco.pareto_v21e3_trace_verify import verify_v21e3_trace_database
from mo_nco.pareto_ijoc_problem import (
    MultiObjectiveKnapsackInstance,
    MultiObjectiveTSPProblemAdapter,
)
from mo_nco.pareto_v21e3_baselines import (
    V21E3BaselineConfig,
    frozen_development_baseline_configs,
    run_v21e3_development_baseline,
)


def _mokp() -> MultiObjectiveKnapsackInstance:
    return MultiObjectiveKnapsackInstance(
        item_weights=(1, 2, 2, 3, 3, 4, 4, 5),
        profits_by_objective=(
            (9, 8, 2, 6, 3, 7, 1, 5),
            (1, 3, 9, 2, 8, 4, 7, 6),
        ),
        capacity=12,
        name="v21e3-baseline-test-mokp",
    )


def _motsp(count: int = 8) -> MultiObjectiveTSPProblemAdapter:
    first = tuple(
        tuple(0.0 if left == right else float(1 + abs(left - right)) for right in range(count))
        for left in range(count)
    )
    second = tuple(
        tuple(
            0.0
            if left == right
            else float(1 + ((3 * left + 5 * right) % 11))
            for right in range(count)
        )
        for left in range(count)
    )
    return MultiObjectiveTSPProblemAdapter(
        MultiObjectiveTSPInstance.from_distance_matrices((first, second))
    )


@pytest.mark.parametrize(
    ("family", "population_size", "mutation_policy", "direction_policy"),
    [
        (
            "MOTSP",
            48,
            "repository_two_opt_with_probability_0_35_v1",
            "repository_motsp_positive_floor_1e3_evenly_spaced_v1",
        ),
        (
            "MOKP",
            40,
            "repository_one_over_n_bit_mutation_force_one_v1",
            "repository_mokp_evenly_spaced_endpoint_including_v1",
        ),
    ],
)
def test_frozen_development_configs_expose_family_repository_parameters(
    family: str,
    population_size: int,
    mutation_policy: str,
    direction_policy: str,
) -> None:
    configs = frozen_development_baseline_configs(
        family=family,
        charged_evaluations=2 * population_size,
        checkpoint_period=population_size,
        seed=31051,
    )

    assert tuple(configs) == ("NSGAII", "MOEAD")
    assert configs["NSGAII"].population_size == population_size
    assert configs["NSGAII"].mutation_policy == mutation_policy
    assert configs["NSGAII"].survival_policy == (
        "nondominated_rank_then_crowding_stable_index_v1"
    )
    assert configs["NSGAII"].survival_schedule == (
        "generation_batched_full_then_frozen_partial_survival_v1"
    )
    assert configs["MOEAD"].reference_direction_policy == direction_policy
    assert configs["MOEAD"].neighborhood_size == 8
    assert configs["MOEAD"].maximum_replacements == 8
    assert configs["MOEAD"].survival_schedule == (
        "cyclic_one_subproblem_per_unique_evaluation_v1"
    )
    for config in configs.values():
        assert config.objective_call_semantics == (
            "first_true_objective_evaluation_v1"
        )
        assert config.duplicate_policy == (
            "exact_solution_cache_zero_charge_retry_then_fallback_v1"
        )
        assert config.selection_authorized is False
        assert config.formal_authorized is False


@pytest.mark.parametrize("arm_id", ["NSGAII", "MOEAD"])
def test_mokp_adapters_are_driven_by_unique_true_evaluation_budget(
    arm_id: str,
    tmp_path: Path,
) -> None:
    config = frozen_development_baseline_configs(
        family="MOKP",
        charged_evaluations=100,
        checkpoint_period=20,
        seed=31051,
        trace_directory=tmp_path,
    )[arm_id]

    run = run_v21e3_development_baseline(_mokp(), config)

    metadata = run.optimization_result.metadata
    assert metadata["arm_id"] == arm_id
    assert metadata["charged_evaluation_count"] == 100
    assert metadata["physical_objective_call_count"] == 100
    assert metadata["attempt_count"] >= 100
    assert metadata["exact_charged_budget_gate"] == "PASS"
    assert metadata["common_budget_adapter_status"] == (
        "DEVELOPMENT_ONLY_AVAILABLE"
    )
    assert metadata["selection_authorized"] is False
    assert metadata["formal_authorized"] is False
    assert [item.iteration for item in run.optimization_result.diagnostics] == [
        20,
        40,
        60,
        80,
        100,
    ]
    receipt = metadata["trace_receipt"]
    assert receipt["status"] == "SUCCESS"
    assert receipt["charged_evaluation_count"] == 100
    assert receipt["attempt_count"] == metadata["attempt_count"]
    assert len(run.attempts) == metadata["attempt_count"]
    assert len(run.evaluations) == 100
    assert metadata["cache_hit_count"] > 0
    assert metadata["retry_count"] > 0
    if arm_id == "NSGAII":
        assert metadata["completed_full_generations"] == 1
        assert metadata["partial_generation_offspring"] == 20
        assert metadata["adaptation_identity"] == (
            "prospective_generation_batched_first_true_adaptation_of_"
            "repository_baseline_v1"
        )
        transitions = metadata["generation_survival_transitions"]
        assert [item["first_charged_evaluation"] for item in transitions] == [
            41,
            81,
        ]
        assert [item["last_charged_evaluation"] for item in transitions] == [
            80,
            100,
        ]
        assert [item["survival_kind"] for item in transitions] == [
            "FULL_GENERATION",
            "FROZEN_PARTIAL_GENERATION",
        ]
        assert all(
            item["parent_population_frozen_during_batch"]
            for item in transitions
        )


@pytest.mark.parametrize("arm_id", ["NSGAII", "MOEAD"])
def test_motsp_adapters_use_the_same_exact_budget_and_checkpoint_contract(
    arm_id: str,
    tmp_path: Path,
) -> None:
    config = frozen_development_baseline_configs(
        family="MOTSP",
        charged_evaluations=120,
        checkpoint_period=24,
        seed=31057,
        trace_directory=tmp_path,
    )[arm_id]

    run = run_v21e3_development_baseline(_motsp(), config)

    metadata = run.optimization_result.metadata
    assert metadata["charged_evaluation_count"] == 120
    assert metadata["physical_objective_call_count"] == 120
    assert metadata["trace_receipt"]["status"] == "SUCCESS"
    assert [item.iteration for item in run.optimization_result.diagnostics] == [
        24,
        48,
        72,
        96,
        120,
    ]
    if arm_id == "NSGAII":
        assert metadata["completed_full_generations"] == 1
        assert metadata["partial_generation_offspring"] == 24
        transitions = metadata["generation_survival_transitions"]
        assert [item["offspring_count"] for item in transitions] == [48, 24]
        assert [item["survival_kind"] for item in transitions] == [
            "FULL_GENERATION",
            "FROZEN_PARTIAL_GENERATION",
        ]


def test_non_development_baseline_execution_is_fail_closed() -> None:
    config = V21E3BaselineConfig(
        arm_id="NSGAII",
        family="MOKP",
        reference_directions=((0.25, 0.75), (0.5, 0.5), (0.75, 0.25)),
        charged_evaluations=12,
        checkpoint_period=3,
        seed=7,
        evidence_partition="selection",
        population_size=3,
        neighborhood_size=3,
        maximum_replacements=3,
    )

    with pytest.raises(ValueError, match="development"):
        run_v21e3_development_baseline(_mokp(), config)


def test_development_adapter_cli_writes_a_fail_closed_auditable_result(
    tmp_path: Path,
) -> None:
    problem = _mokp()
    instance_path = tmp_path / "instance.json"
    instance_path.write_text(
        json.dumps(
            {
                "schema": "pareto_v21_mokp_integer_instance_v1",
                "case_id": problem.name,
                "family": "MOKP",
                "num_items": problem.solution_size,
                "num_objectives": problem.num_objectives,
                "item_weights": problem.item_weights,
                "profits_by_objective": problem.profits_by_objective,
                "capacity": problem.capacity,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "nsga2-output"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "mo_nco.pareto_v21e3_baselines",
            "--arm",
            "NSGAII",
            "--instance",
            str(instance_path),
            "--seed",
            "31051",
            "--budget",
            "80",
            "--checkpoint-period",
            "20",
            "--source-snapshot-sha256",
            "ab" * 32,
            "--output-directory",
            str(output),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "SUCCESS_ENGINEERING_ONLY"
    assert result["arm_id"] == "NSGAII"
    assert result["charged_evaluation_count"] == 80
    assert result["adaptation_identity"] == (
        "prospective_generation_batched_first_true_adaptation_of_"
        "repository_baseline_v1"
    )
    assert result["generation_survival_transitions"][0]["survival_kind"] == (
        "FULL_GENERATION"
    )
    assert result["runtime_identity"]["python_implementation"] == "CPython"
    assert len(result["runtime_identity"]["python_executable_sha256"]) == 64
    assert result["run_context"]["algorithm_source_sha256"] == "ab" * 32
    assert result["run_context"]["algorithm_source_binding_kind"] == (
        "explicit_successor_source_snapshot_sha256_v1"
    )
    assert result["selection_authorized"] is False
    assert result["formal_authorized"] is False
    assert (output / "nsga2.trace.sqlite3").is_file()
    assert (output / "nsga2.terminal.receipt.json").is_file()


def test_parity_protocol_v2_preflight_requires_post_adapter_snapshot() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol_path = (
        root
        / "ijoc_submission_v21e3"
        / "protocol"
        / "V21E3_C0_PARITY_PROTOCOL_V2.json"
    )
    raw = protocol_path.read_bytes()

    receipt = preflight_development_parity_protocol_v2(
        ArtifactRoot(root),
        {
            "path": protocol_path.relative_to(root).as_posix(),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    )

    assert receipt["status"] == (
        "ADAPTER_PREFLIGHT_PASS_SUCCESSOR_SNAPSHOT_REQUIRED"
    )
    assert receipt["execution_adapter_status_by_arm"] == {
        "V21E3_C0": "DEVELOPMENT_ONLY_AVAILABLE",
        "NSGAII": "DEVELOPMENT_ONLY_AVAILABLE",
        "MOEAD": "DEVELOPMENT_ONLY_AVAILABLE",
    }
    assert receipt["successor_source_snapshot"] == "PENDING"
    assert receipt["matched_matrix"] == "NOT_RUN"
    assert receipt["parity_execution_authorized"] is False
    assert receipt["selection_entropy_release"] == "PROHIBITED"
    assert receipt["formal_authorized"] is False


def test_unique_state_exhaustion_writes_terminal_failure_receipt(
    tmp_path: Path,
) -> None:
    problem = MultiObjectiveKnapsackInstance(
        item_weights=(2, 3),
        profits_by_objective=((2, 1), (1, 2)),
        capacity=1,
        name="only-one-feasible-state",
    )
    receipt_path = tmp_path / "failure.receipt.json"
    config = V21E3BaselineConfig(
        arm_id="NSGAII",
        family="MOKP",
        reference_directions=((0.25, 0.75), (0.75, 0.25)),
        charged_evaluations=2,
        checkpoint_period=1,
        seed=11,
        population_size=2,
        neighborhood_size=2,
        maximum_replacements=2,
        duplicate_retry_cap=1,
        fallback_attempt_cap=2,
        trace_database=str(tmp_path / "failure.trace.sqlite3"),
        terminal_receipt=str(receipt_path),
    )

    with pytest.raises(RuntimeError, match="terminal FAILURE"):
        run_v21e3_development_baseline(problem, config)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "FAILURE"
    assert receipt["failure_code"] == "FINALIZATION_GATE_FAILED"
    assert receipt["charged_evaluation_count"] == 1
    assert receipt["attempt_count"] == 5


@pytest.mark.parametrize("arm_id", ["NSGAII", "MOEAD"])
@pytest.mark.parametrize("family", ["MOKP", "MOTSP"])
def test_successor_baseline_receipts_use_v2_mirrors_and_replay(
    arm_id: str,
    family: str,
    tmp_path: Path,
) -> None:
    problem = _mokp() if family == "MOKP" else _motsp()
    config = frozen_development_baseline_configs(
        family=family,
        charged_evaluations=(100 if family == "MOKP" else 120),
        checkpoint_period=(20 if family == "MOKP" else 24),
        seed=31059,
        trace_directory=tmp_path,
    )[arm_id]

    run = run_v21e3_development_baseline(problem, config)
    metadata = run.optimization_result.metadata
    context = metadata["run_context"]

    assert context["schema"] == "v21e3r1_run_context_v2"
    assert context["candidate_id"] == arm_id
    assert context["algorithm_source_binding_kind"] == (
        "development_adapter_module_sha256_fallback_pre_snapshot_v1"
    )
    assert context["algorithm_config"]["candidate_id"] == arm_id
    assert context["algorithm_config"]["seed"] == context["seed"]
    expected_budget = 100 if family == "MOKP" else 120
    assert context["algorithm_config"]["charged_evaluations"] == expected_budget
    assert context["algorithm_config"]["phase"] == "development"
    assert (
        context["algorithm_config"]["reference_directions"]
        == context["reference_directions"]
    )
    receipt_path = Path(config.terminal_receipt)
    replay = verify_v21e3_trace_database(
        Path(config.trace_database),
        problem,
        expected_run_context=context,
        detached_terminal_receipt_path=receipt_path,
        expected_detached_terminal_receipt_sha256=hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest(),
        expected_charged_evaluations=expected_budget,
    )
    assert replay["status"] == "OBJECTIVE_AND_ARCHIVE_REPLAY_PASS"
    assert replay["selection_authorization"] == "PROHIBITED"
    assert replay["evaluation_records"] == expected_budget


def test_motsp_generation_batch_preserves_repository_primary_rng_proposals() -> None:
    problem = _motsp(count=12)
    config = frozen_development_baseline_configs(
        family="MOTSP",
        charged_evaluations=96,
        checkpoint_period=24,
        seed=31057,
    )["NSGAII"]
    adapted = run_v21e3_development_baseline(problem, config)

    counted = CountingTSPInstance(problem.instance, max_evaluations=96)
    native_calls: list[tuple[int, ...]] = []
    native_evaluate = counted.evaluate
    counted.evaluate = lambda tour: (  # type: ignore[method-assign]
        native_calls.append(tuple(tour)),
        native_evaluate(tour),
    )[1]
    NSGAIIOptimizer(
        counted,
        population_size=48,
        evaluations=96,
        seed=31057,
        mutation_probability=0.35,
        log_period=24,
        archive_max_size=200,
    ).run()

    adapted_primary_proposals = [
        event.proposal
        for event in adapted.attempts
        if event.operator
        in {
            "problem_native_exact_random_initialization_v1",
            "generation_batched_nsga2_family_native_variation_v1",
        }
    ]
    assert adapted_primary_proposals == native_calls
    assert adapted.optimization_result.metadata["cache_hit_count"] > 0
    assert adapted.optimization_result.metadata[
        "generation_survival_transitions"
    ][0]["survival_kind"] == "FULL_GENERATION"


def test_mokp_generation_batch_preserves_repository_primary_rng_proposals() -> None:
    problem = MultiObjectiveKnapsackInstance.random_instance(20, seed=123)
    config = frozen_development_baseline_configs(
        family="MOKP",
        charged_evaluations=80,
        checkpoint_period=20,
        seed=31051,
    )["NSGAII"]
    adapted = run_v21e3_development_baseline(problem, config)

    class LoggingProblem:
        def __init__(self, base: MultiObjectiveKnapsackInstance) -> None:
            self.base = base
            self.calls: list[tuple[int, ...]] = []

        def __getattr__(self, name: str) -> object:
            return getattr(self.base, name)

        def evaluate(self, solution: tuple[int, ...]) -> tuple[float, ...]:
            self.calls.append(tuple(solution))
            return self.base.evaluate(solution)

    logging_problem = LoggingProblem(problem)
    BinaryNSGA2MOKPBaseline(
        logging_problem,  # type: ignore[arg-type]
        evaluations=80,
        seed=31051,
        anytime_checkpoint_period=20,
        population_size=40,
    ).run()

    adapted_primary_proposals = [
        event.proposal
        for event in adapted.attempts
        if event.operator
        in {
            "problem_native_exact_random_initialization_v1",
            "generation_batched_nsga2_family_native_variation_v1",
        }
    ]
    assert adapted_primary_proposals == logging_problem.calls
    assert adapted.optimization_result.metadata[
        "generation_survival_transitions"
    ][0]["survival_kind"] == "FULL_GENERATION"


def test_motsp_moead_preserves_repository_steady_state_primary_proposals() -> None:
    problem = _motsp(count=12)
    config = frozen_development_baseline_configs(
        family="MOTSP",
        charged_evaluations=96,
        checkpoint_period=24,
        seed=31051,
    )["MOEAD"]
    adapted = run_v21e3_development_baseline(problem, config)

    counted = CountingTSPInstance(problem.instance, max_evaluations=96)
    native_calls: list[tuple[int, ...]] = []
    native_evaluate = counted.evaluate
    native_two_opt = counted.evaluate_two_opt
    counted.evaluate = lambda tour: (  # type: ignore[method-assign]
        native_calls.append(tuple(tour)),
        native_evaluate(tour),
    )[1]
    counted.evaluate_two_opt = (  # type: ignore[method-assign]
        lambda tour, objective, left, right: (
            native_calls.append(two_opt_at(tour, left, right)),
            native_two_opt(tour, objective, left, right),
        )[1]
    )
    native = MOEADOptimizer(
        counted,
        population_size=48,
        evaluations=96,
        seed=31051,
        neighbor_size=8,
        log_period=24,
        archive_max_size=200,
    ).run()

    adapted_primary_proposals = [
        event.proposal
        for event in adapted.attempts
        if event.operator
        in {
            "problem_native_exact_random_initialization_v1",
            "moead_neighborhood_family_native_variation_v1",
        }
    ]
    assert adapted.optimization_result.metadata["cache_hit_count"] == 0
    assert adapted_primary_proposals == native_calls
    assert adapted.optimization_result.particles == native.particles
    assert adapted.optimization_result.objectives == native.objectives


def test_mokp_moead_preserves_repository_steady_state_primary_proposals() -> None:
    problem = MultiObjectiveKnapsackInstance.random_instance(40, seed=123)
    config = frozen_development_baseline_configs(
        family="MOKP",
        charged_evaluations=80,
        checkpoint_period=20,
        seed=31051,
    )["MOEAD"]
    adapted = run_v21e3_development_baseline(problem, config)

    class LoggingProblem:
        def __init__(self, base: MultiObjectiveKnapsackInstance) -> None:
            self.base = base
            self.calls: list[tuple[int, ...]] = []

        def __getattr__(self, name: str) -> object:
            return getattr(self.base, name)

        def evaluate(self, solution: tuple[int, ...]) -> tuple[float, ...]:
            self.calls.append(tuple(solution))
            return self.base.evaluate(solution)

    logging_problem = LoggingProblem(problem)
    BinaryMOEADMOKPBaseline(
        logging_problem,  # type: ignore[arg-type]
        evaluations=80,
        seed=31051,
        anytime_checkpoint_period=20,
        population_size=40,
        neighborhood_size=8,
    ).run()

    adapted_primary_proposals = [
        event.proposal
        for event in adapted.attempts
        if event.operator
        in {
            "problem_native_exact_random_initialization_v1",
            "moead_neighborhood_family_native_variation_v1",
        }
    ]
    assert adapted_primary_proposals == logging_problem.calls


def test_moead_trace_binds_the_native_cyclic_update_schedule(
    tmp_path: Path,
) -> None:
    config = frozen_development_baseline_configs(
        family="MOKP",
        charged_evaluations=80,
        checkpoint_period=20,
        seed=31051,
        trace_directory=tmp_path,
    )["MOEAD"]

    run_v21e3_development_baseline(_mokp(), config)

    connection = sqlite3.connect(config.trace_database)
    try:
        contexts = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT context_json FROM attempts ORDER BY attempt_index"
            )
            if json.loads(row[0])["operator_id"]
            == "moead_neighborhood_family_native_variation_v1"
        ]
    finally:
        connection.close()
    assert [
        item["operator_witness"]["cyclic_update_ordinal"]
        for item in contexts
    ] == list(range(1, 41))
    assert [
        item["operator_witness"]["subproblem_index"]
        for item in contexts
    ] == list(range(40))
    assert all(
        item["operator_witness"]["update_schedule"]
        == "cyclic_one_subproblem_per_unique_evaluation_v1"
        for item in contexts
    )


def test_nsga_trace_binds_full_and_frozen_partial_generation_positions(
    tmp_path: Path,
) -> None:
    config = frozen_development_baseline_configs(
        family="MOKP",
        charged_evaluations=100,
        checkpoint_period=20,
        seed=31051,
        trace_directory=tmp_path,
    )["NSGAII"]

    run_v21e3_development_baseline(_mokp(), config)

    connection = sqlite3.connect(config.trace_database)
    try:
        contexts = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT context_json FROM attempts ORDER BY attempt_index"
            )
            if json.loads(row[0])["operator_id"]
            == "generation_batched_nsga2_family_native_variation_v1"
        ]
    finally:
        connection.close()
    assert [
        item["operator_witness"]["generation_index"] for item in contexts
    ] == [1] * 40 + [2] * 20
    assert [
        item["operator_witness"]["offspring_position"] for item in contexts
    ] == list(range(40)) + list(range(20))
    assert [
        item["operator_witness"]["planned_batch_size"] for item in contexts
    ] == [40] * 40 + [20] * 20
    assert all(
        item["operator_witness"]["survival_schedule"]
        == "generation_batched_full_then_frozen_partial_survival_v1"
        for item in contexts
    )

