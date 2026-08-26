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
    V21E3ResourceLimitExceeded,
    V21E3TypedHybridParetoSearch,
)


SCREEN = "bounded_cache_aware_structural_screen_development_v1"
NONWORSE = "bounded_reference_neighborhood_nonworse_replacement_v1"
LYAPUNOV = "archive_compensated_information_lyapunov_development_v1"


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _reseal_v9_decision_chain_after_policy_witness_tamper(
    database_path: Path,
    terminal_path: Path,
    *,
    tamper_kind: str = "archive_credit",
) -> None:
    """Coherently reseal hashes after changing one Lyapunov witness field.

    This models an artifact producer that can recompute the self-authenticating
    hashes.  The independent verifier must reject the semantic inconsistency,
    rather than treating the hash chain itself as proof of the policy decision.
    """

    with sqlite3.connect(database_path) as connection:
        rows = list(
            connection.execute(
                "SELECT evaluation_index,decision_json FROM decisions "
                "ORDER BY evaluation_index"
            )
        )
        previous = "0" * 64
        changed = False
        for evaluation_index, raw in rows:
            payload = json.loads(raw)
            witness = payload.get("policy_witness")
            if not changed and isinstance(witness, dict):
                if witness.get("schema") in {
                    "v21e3r1_archive_compensated_replacement_v1",
                    "v21e3r1_archive_compensated_replacement_v2",
                }:
                    if tamper_kind == "archive_credit":
                        witness["archive_credit"] = (
                            float(witness["archive_credit"]) + 0.125
                        )
                    elif tamper_kind == "paid_worsening_target_count":
                        witness["paid_worsening_target_count"] = (
                            int(witness["paid_worsening_target_count"]) + 1
                        )
                    elif tamper_kind == "unknown_field":
                        witness["attacker_extension"] = "resealed"
                    else:  # pragma: no cover - test helper contract
                        raise AssertionError(f"unknown tamper_kind {tamper_kind!r}")
                    changed = True
            payload["prev_decision_sha256"] = previous
            decision_raw = _canonical_bytes(payload)
            decision_sha256 = hashlib.sha256(decision_raw).hexdigest()
            connection.execute(
                "UPDATE decisions SET decision_json=?,prev_decision_sha256=?,"
                "decision_sha256=? WHERE evaluation_index=?",
                (
                    decision_raw.decode("utf-8"),
                    previous,
                    decision_sha256,
                    evaluation_index,
                ),
            )
            previous = decision_sha256
        assert changed, "The test trace did not emit a Lyapunov policy witness."

        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        terminal["terminal_decision_chain_sha256"] = previous
        terminal.pop("receipt_payload_sha256")
        receipt_sha256 = hashlib.sha256(_canonical_bytes(terminal)).hexdigest()
        terminal["receipt_payload_sha256"] = receipt_sha256
        terminal_raw = _canonical_bytes(terminal)
        connection.execute(
            "UPDATE terminal_receipts SET receipt_json=?,receipt_sha256=? "
            "WHERE run_id=1",
            (terminal_raw.decode("utf-8"), receipt_sha256),
        )
        connection.execute(
            "UPDATE run_attempt SET terminal_receipt_sha256=? WHERE run_id=1",
            (receipt_sha256,),
        )
        connection.commit()
    terminal_path.write_bytes(terminal_raw)


def _reseal_v9_decision_chain_after_false_empty_target_forgery(
    database_path: Path,
    terminal_path: Path,
) -> None:
    """Forge a populated target as empty and coherently reseal every hash.

    The final search decision is used so the forged population update cannot
    invalidate a later decision.  All decision, terminal, SQLite, and detached
    receipt bindings are recomputed; only an independent durable-population
    replay can detect the false empty-target claim.
    """

    with sqlite3.connect(database_path) as connection:
        rows = list(
            connection.execute(
                "SELECT evaluation_index,decision_json FROM decisions "
                "ORDER BY evaluation_index"
            )
        )
        lyapunov_indices = [
            evaluation_index
            for evaluation_index, raw in rows
            if (json.loads(raw).get("policy_witness") or {}).get("schema")
            == "v21e3r1_archive_compensated_replacement_v2"
        ]
        assert lyapunov_indices, "The test trace did not emit a Lyapunov witness."
        forged_index = max(lyapunov_indices)

        previous = "0" * 64
        changed = False
        for evaluation_index, raw in rows:
            payload = json.loads(raw)
            if evaluation_index == forged_index:
                witness = payload["policy_witness"]
                assert witness["preselected_empty_target_type_ids"] == []
                false_empty_target = witness["considered_target_type_ids"][0]
                forged_targets = [
                    false_empty_target,
                    *witness["decision_selected_target_type_ids"],
                ]
                witness["preselected_empty_target_type_ids"] = [
                    false_empty_target
                ]
                witness["selected_target_type_ids"] = forged_targets
                payload["population_target_type_ids"] = forged_targets
                payload["accepted_into_population"] = True
                payload["population_replacement_count"] = len(forged_targets)
                payload["decision_reason"] = (
                    "archive_compensated_lyapunov_replacement"
                )
                changed = True
            payload["prev_decision_sha256"] = previous
            decision_raw = _canonical_bytes(payload)
            decision_sha256 = hashlib.sha256(decision_raw).hexdigest()
            connection.execute(
                "UPDATE decisions SET decision_json=?,prev_decision_sha256=?,"
                "decision_sha256=? WHERE evaluation_index=?",
                (
                    decision_raw.decode("utf-8"),
                    previous,
                    decision_sha256,
                    evaluation_index,
                ),
            )
            previous = decision_sha256
        assert changed

        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        terminal["terminal_decision_chain_sha256"] = previous
        terminal.pop("receipt_payload_sha256")
        receipt_sha256 = hashlib.sha256(_canonical_bytes(terminal)).hexdigest()
        terminal["receipt_payload_sha256"] = receipt_sha256
        terminal_raw = _canonical_bytes(terminal)
        connection.execute(
            "UPDATE terminal_receipts SET receipt_json=?,receipt_sha256=? "
            "WHERE run_id=1",
            (terminal_raw.decode("utf-8"), receipt_sha256),
        )
        connection.execute(
            "UPDATE run_attempt SET terminal_receipt_sha256=? WHERE run_id=1",
            (receipt_sha256,),
        )
        connection.commit()
    terminal_path.write_bytes(terminal_raw)


def _v9_config(
    tmp_path: Path,
    *,
    diagnostic_id: str,
    screening: str,
    replacement: str,
    tradeoff_lambda: float,
    budget: int = 24,
) -> V21E3HybridConfig:
    kwargs: dict[str, object] = {
        "candidate_id": "C0",
        "reference_directions": ((0.25, 0.75), (0.75, 0.25)),
        "charged_evaluations": budget,
        "checkpoint_period": 6,
        "seed": 1701,
        "phase": "development",
        "trace_database": str(tmp_path / "trace.sqlite3"),
        "terminal_receipt": str(tmp_path / "terminal.json"),
        "receipt_database_path": "trace.sqlite3",
        "capture_trace": True,
        "local_improvement_steps": 1,
        "development_diagnostic_id": diagnostic_id,
        "candidate_screening_policy": screening,
        "candidate_screening_cap": 8,
        "archive_tradeoff_lambda": tradeoff_lambda,
        "replacement_policy": replacement,
    }
    available = {field.name for field in fields(V21E3HybridConfig)}
    if "attempt_cap" in available:
        kwargs["attempt_cap"] = 256
        kwargs["structural_screening_cap"] = 4096 if screening == SCREEN else 0
        kwargs["wall_time_cap_seconds"] = 30.0
    return V21E3HybridConfig(**kwargs)  # type: ignore[arg-type]


def test_v9_execution_config_exposes_global_dual_resource_caps() -> None:
    available = {field.name for field in fields(V21E3HybridConfig)}
    assert {
        "attempt_cap",
        "structural_screening_cap",
        "wall_time_cap_seconds",
    } <= available


@pytest.mark.parametrize(
    "diagnostic_id,screening,replacement,tradeoff_lambda",
    [
        (
            "V21E3R1_V9_INFORMATION_SCREEN_MOKP",
            SCREEN,
            NONWORSE,
            0.0,
        ),
        (
            "V21E3R1_V9_LYAPUNOV_MOKP",
            "disabled_v1",
            LYAPUNOV,
            0.5,
        ),
        (
            "V21E3R1_V9_INFORMATION_SCREEN_MOTSP",
            SCREEN,
            NONWORSE,
            0.0,
        ),
        (
            "V21E3R1_V9_LYAPUNOV_MOTSP",
            "disabled_v1",
            LYAPUNOV,
            0.5,
        ),
    ],
)
def test_all_single_mechanism_v9_payloads_are_branch_replay_complete(
    tmp_path: Path,
    diagnostic_id: str,
    screening: str,
    replacement: str,
    tradeoff_lambda: float,
) -> None:
    from mo_nco.pareto_v21e3r1_branch_replay import _hybrid_dataclass_kwargs

    config = _v9_config(
        tmp_path,
        diagnostic_id=diagnostic_id,
        screening=screening,
        replacement=replacement,
        tradeoff_lambda=tradeoff_lambda,
    )
    kwargs = _hybrid_dataclass_kwargs(config.semantic_payload())
    assert kwargs["development_diagnostic_id"] == diagnostic_id


def test_screen_only_operator_diagnostic_reconstructs_all_archive_gain(
    tmp_path: Path,
) -> None:
    from mo_nco.pareto_v21e3r1_v9_diagnostics import analyze_v9_trace_database

    problem = MultiObjectiveKnapsackInstance.random_instance(
        12, num_objectives=2, seed=91
    )
    config = _v9_config(
        tmp_path,
        diagnostic_id="V21E3R1_V9_INFORMATION_SCREEN_MOKP",
        screening=SCREEN,
        replacement=NONWORSE,
        tradeoff_lambda=0.0,
    )
    run = V21E3TypedHybridParetoSearch(problem, config).run()
    lower = tuple(float(value) for value in problem.objective_lower_bounds)
    upper = tuple(float(value) for value in problem.objective_upper_bounds)
    scale = (upper[0] - lower[0]) * (upper[1] - lower[1])
    expected_gain = run.optimization_result.archive.hypervolume_2d(upper) / scale
    assert expected_gain > 0.0

    report = analyze_v9_trace_database(tmp_path / "trace.sqlite3")
    observed_gain = sum(float(row["hv_gain"]) for row in report["operators"])
    assert observed_gain == pytest.approx(expected_gain, abs=1e-12)
    assert sum(int(row["screenings"]) for row in report["operators"]) > 0


def test_non_v9_three_objective_run_does_not_require_2d_hypervolume() -> None:
    problem = MultiObjectiveKnapsackInstance.random_instance(
        8, num_objectives=3, seed=113
    )
    config = V21E3HybridConfig(
        candidate_id="C0",
        reference_directions=((0.2, 0.3, 0.5), (0.5, 0.3, 0.2)),
        charged_evaluations=8,
        checkpoint_period=4,
        seed=7,
        phase="development",
        capture_trace=False,
        local_improvement_steps=1,
    )
    run = V21E3TypedHybridParetoSearch(problem, config).run()
    assert run.optimization_result.metadata["charged_evaluation_count"] == 8


def test_v9_motsp_both_arm_emits_screen_and_lyapunov_evidence(
    tmp_path: Path,
) -> None:
    problem = MultiObjectiveTSPProblemAdapter(
        MultiObjectiveTSPInstance.random_biobjective(8, seed=911)
    )
    config = _v9_config(
        tmp_path,
        diagnostic_id="V21E3R1_V9_INFORMATION_LYAPUNOV_MOTSP",
        screening=SCREEN,
        replacement=LYAPUNOV,
        tradeoff_lambda=0.5,
    )
    run = V21E3TypedHybridParetoSearch(problem, config).run()
    assert run.optimization_result.metadata["candidate_screen_count"] > 0
    assert run.optimization_result.metadata[
        "archive_lyapunov_replacement_count"
    ] >= 0


@pytest.mark.parametrize(
    "resource,screening,attempt_cap,structural_cap,wall_cap",
    [
        ("attempts", "disabled_v1", 24, 0, 30.0),
        ("structural_screening_work", SCREEN, 256, 8, 30.0),
        ("wall_time_seconds", "disabled_v1", 256, 0, 1e-12),
    ],
)
def test_v9_resource_caps_fail_closed_with_terminal_receipt(
    tmp_path: Path,
    resource: str,
    screening: str,
    attempt_cap: int,
    structural_cap: int,
    wall_cap: float,
) -> None:
    problem = MultiObjectiveKnapsackInstance.random_instance(
        12, num_objectives=2, seed=91
    )
    replacement = NONWORSE
    diagnostic_id = (
        "V21E3R1_V9_INFORMATION_SCREEN_MOKP"
        if screening == SCREEN
        else "V21E3R1_V9_LEGACY_MOKP"
    )
    config = V21E3HybridConfig(
        candidate_id="C0",
        reference_directions=((0.25, 0.75), (0.75, 0.25)),
        charged_evaluations=24,
        checkpoint_period=6,
        seed=1701,
        phase="development",
        trace_database=str(tmp_path / f"{resource}.sqlite3"),
        terminal_receipt=str(tmp_path / f"{resource}.json"),
        receipt_database_path=f"{resource}.sqlite3",
        local_improvement_steps=1,
        development_diagnostic_id=diagnostic_id,
        candidate_screening_policy=screening,
        candidate_screening_cap=8,
        archive_tradeoff_lambda=0.0,
        replacement_policy=replacement,
        attempt_cap=attempt_cap,
        structural_screening_cap=structural_cap,
        wall_time_cap_seconds=wall_cap,
    )
    with pytest.raises(V21E3ResourceLimitExceeded, match=resource):
        V21E3TypedHybridParetoSearch(problem, config).run()
    receipt = json.loads((tmp_path / f"{resource}.json").read_text("utf-8"))
    assert receipt["status"] == "FAILURE"
    assert receipt["failure_code"] == "V9_RESOURCE_CAP_EXHAUSTED"
    assert receipt["failure_detail"]["resource"] == resource


def test_v9_success_receipt_resource_accounting_independently_replays(
    tmp_path: Path,
) -> None:
    from mo_nco.pareto_v21e3_trace_verify import verify_v21e3_trace_database

    problem = MultiObjectiveKnapsackInstance.random_instance(
        12, num_objectives=2, seed=91
    )
    config = _v9_config(
        tmp_path,
        diagnostic_id="V21E3R1_V9_INFORMATION_SCREEN_MOKP",
        screening=SCREEN,
        replacement=NONWORSE,
        tradeoff_lambda=0.0,
    )
    V21E3TypedHybridParetoSearch(problem, config).run()
    with sqlite3.connect(tmp_path / "trace.sqlite3") as connection:
        raw_context = connection.execute(
            "SELECT run_context_json FROM run_attempt WHERE run_id=1"
        ).fetchone()[0]
    terminal_path = tmp_path / "terminal.json"
    verification = verify_v21e3_trace_database(
        tmp_path / "trace.sqlite3",
        problem,
        expected_run_context=json.loads(raw_context),
        detached_terminal_receipt_path=terminal_path,
        expected_detached_terminal_receipt_sha256=hashlib.sha256(
            terminal_path.read_bytes()
        ).hexdigest(),
        expected_charged_evaluations=24,
    )
    assert verification["v9_resource_contract_replay"] == "PASS"
    assert verification["v9_candidate_screen_witness_replay"] == "PASS"
    assert verification["v9_candidate_screen_witnesses_verified"] > 0
    assert verification["v9_population_policy_replay"] == "PASS"
    assert verification["v9_population_policy_decisions_verified"] == 24


@pytest.mark.parametrize(
    "tamper_kind",
    ["archive_credit", "paid_worsening_target_count", "unknown_field"],
)
def test_v9_verifier_rejects_resealed_lyapunov_policy_witness_tamper(
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    from mo_nco.pareto_v21e3_trace_verify import verify_v21e3_trace_database

    problem = MultiObjectiveKnapsackInstance.random_instance(
        12, num_objectives=2, seed=91
    )
    config = _v9_config(
        tmp_path,
        diagnostic_id="V21E3R1_V9_LYAPUNOV_MOKP",
        screening="disabled_v1",
        replacement=LYAPUNOV,
        tradeoff_lambda=0.5,
    )
    V21E3TypedHybridParetoSearch(problem, config).run()
    database_path = tmp_path / "trace.sqlite3"
    terminal_path = tmp_path / "terminal.json"
    with sqlite3.connect(database_path) as connection:
        raw_context = connection.execute(
            "SELECT run_context_json FROM run_attempt WHERE run_id=1"
        ).fetchone()[0]

    before = verify_v21e3_trace_database(
        database_path,
        problem,
        expected_run_context=json.loads(raw_context),
        detached_terminal_receipt_path=terminal_path,
        expected_detached_terminal_receipt_sha256=hashlib.sha256(
            terminal_path.read_bytes()
        ).hexdigest(),
        expected_charged_evaluations=24,
    )
    assert before["v9_lyapunov_policy_witness_replay"] == "PASS"
    assert before["v9_lyapunov_policy_witnesses_verified"] > 0

    _reseal_v9_decision_chain_after_policy_witness_tamper(
        database_path,
        terminal_path,
        tamper_kind=tamper_kind,
    )

    with pytest.raises(ValueError, match="V9 Lyapunov policy witness"):
        verify_v21e3_trace_database(
            database_path,
            problem,
            expected_run_context=json.loads(raw_context),
            detached_terminal_receipt_path=terminal_path,
            expected_detached_terminal_receipt_sha256=hashlib.sha256(
                terminal_path.read_bytes()
            ).hexdigest(),
            expected_charged_evaluations=24,
        )


def test_v9_verifier_rejects_resealed_false_empty_population_target(
    tmp_path: Path,
) -> None:
    from mo_nco.pareto_v21e3_trace_verify import verify_v21e3_trace_database

    problem = MultiObjectiveKnapsackInstance.random_instance(
        12, num_objectives=2, seed=91
    )
    config = _v9_config(
        tmp_path,
        diagnostic_id="V21E3R1_V9_LYAPUNOV_MOKP",
        screening="disabled_v1",
        replacement=LYAPUNOV,
        tradeoff_lambda=0.5,
    )
    V21E3TypedHybridParetoSearch(problem, config).run()
    database_path = tmp_path / "trace.sqlite3"
    terminal_path = tmp_path / "terminal.json"
    with sqlite3.connect(database_path) as connection:
        raw_context = connection.execute(
            "SELECT run_context_json FROM run_attempt WHERE run_id=1"
        ).fetchone()[0]

    _reseal_v9_decision_chain_after_false_empty_target_forgery(
        database_path,
        terminal_path,
    )

    with pytest.raises(ValueError, match="empty targets disagree with durable"):
        verify_v21e3_trace_database(
            database_path,
            problem,
            expected_run_context=json.loads(raw_context),
            detached_terminal_receipt_path=terminal_path,
            expected_detached_terminal_receipt_sha256=hashlib.sha256(
                terminal_path.read_bytes()
            ).hexdigest(),
            expected_charged_evaluations=24,
        )
