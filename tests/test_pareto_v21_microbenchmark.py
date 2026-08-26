from __future__ import annotations

from mo_nco.pareto_ijoc_problem import MultiObjectiveKnapsackInstance
from mo_nco.pareto_v21_microbenchmark import run_v21_trace_microbenchmark


def test_v21_trace_microbenchmark_compares_memory_and_persistent_ledgers(tmp_path) -> None:
    problem = MultiObjectiveKnapsackInstance.random_instance(30, seed=21101)

    receipt = run_v21_trace_microbenchmark(
        problem=problem,
        trace_path=tmp_path / "benchmark.sqlite3",
        receipt_path=tmp_path / "benchmark_receipt.json",
        evaluation_budget=40,
        checkpoint_period=10,
        seed=83,
        candidate_id="C0",
        reference_directions=((0.2, 0.8), (0.5, 0.5), (0.8, 0.2)),
    )

    assert receipt["status"] == "PASS"
    assert receipt["evaluation_budget"] == 40
    assert receipt["trace_verification_status"] == "PASS"
    assert receipt["semantic_reproducibility_gate"] == "PASS"
    assert receipt["trace_bytes_per_evaluation"] > 0.0
    assert receipt["persistent_wall_seconds"] > 0.0
    assert receipt["memory_wall_seconds"] > 0.0
    assert (tmp_path / "benchmark_receipt.json").is_file()

