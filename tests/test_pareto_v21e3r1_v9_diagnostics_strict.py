from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from mo_nco.pareto_ijoc_problem import MultiObjectiveKnapsackInstance
from mo_nco.pareto_v21e3_hybrid import (
    V21E3HybridConfig,
    V21E3TypedHybridParetoSearch,
)
from mo_nco.pareto_v21e3r1_v9_diagnostics import (
    V9TraceDiagnosticError,
    analyze_v9_trace_database,
    main,
)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _arm_contract(diagnostic_id: str) -> tuple[str, str, str, float]:
    family = diagnostic_id.rsplit("_", 1)[-1]
    screening = (
        "bounded_cache_aware_structural_screen_development_v1"
        if "INFORMATION" in diagnostic_id
        else "disabled_v1"
    )
    lyapunov = "LYAPUNOV" in diagnostic_id
    replacement = (
        "archive_compensated_information_lyapunov_development_v1"
        if lyapunov
        else "bounded_reference_neighborhood_nonworse_replacement_v1"
    )
    return family, screening, replacement, (0.5 if lyapunov else 0.0)


def _screen_witness(
    *,
    operator: str,
    solutions: tuple[tuple[int, ...], ...],
    seen: tuple[bool, ...],
    exhausted: bool,
    generated: int,
) -> dict[str, object]:
    checks = [
        {
            "rank": rank,
            "solution": list(solution),
            "solution_sha256": hashlib.sha256(
                _canonical(list(solution)).encode("utf-8")
            ).hexdigest(),
            "operator": operator,
            "seen_before_attempt": was_seen,
        }
        for rank, (solution, was_seen) in enumerate(zip(solutions, seen))
    ]
    selected_rank = len(checks) - 1
    probes = len(checks)
    return {
        "schema": "v21e3r1_information_time_candidate_screen_v2",
        "policy": "bounded_cache_aware_structural_screen_development_v1",
        "screen_cap": 4,
        "candidates_examined": probes,
        "cached_candidates_skipped": sum(seen),
        "selected_rank": selected_rank,
        "screen_exhausted": exhausted,
        "selected_operator": operator,
        "selected_solution_sha256": checks[selected_rank]["solution_sha256"],
        "candidate_membership_checks": checks,
        "objective_calls_during_screen": 0,
        "structural_candidates_generated": generated,
        "cache_membership_probes": probes,
        "total_structural_screening_work": generated + probes,
    }


def _write_trace(
    path: Path,
    *,
    diagnostic_id: str = "V21E3R1_V9_INFORMATION_LYAPUNOV_MOKP",
    objectives: tuple[tuple[float, float], ...] = (
        (0.8, 0.8),
        (0.4, 0.9),
        (0.5, 0.5),
        (0.9, 0.4),
    ),
    include_cache_hit: bool = True,
    omit_lower_bound: bool = False,
    population_size: int | None = None,
) -> None:
    family, screening, replacement, tradeoff_lambda = _arm_contract(diagnostic_id)
    budget = len(objectives)
    frozen_population_size = budget if population_size is None else population_size
    reference_directions = [[0.5, 0.5] for _ in range(frozen_population_size)]
    config = {
        "candidate_id": "C0",
        "development_diagnostic_id": diagnostic_id,
        "candidate_screening_policy": screening,
        "replacement_policy": replacement,
        "archive_tradeoff_lambda": tradeoff_lambda,
        "charged_evaluations": budget,
        "reference_directions": reference_directions,
        "neighborhood_size": max(1, min(2, frozen_population_size)),
        "phase": "development",
    }
    run_context: dict[str, object] = {
        "schema": "v21e3r1_run_context_v2",
        "candidate_id": "C0",
        "algorithm_config": config,
        "charged_evaluation_budget": budget,
        "evidence_partition": "development",
        "objective_upper_bounds": [1.0, 1.0],
        "reference_directions": reference_directions,
        "v9_resource_contract_schema": "v21e3r1_v9_ast_resource_contract_v1",
    }
    if not omit_lower_bound:
        run_context["objective_lower_bounds"] = [0.0, 0.0]
    run_context_json = _canonical(run_context)
    run_context_digest = hashlib.sha256(run_context_json.encode("utf-8")).hexdigest()

    attempt_count = budget + int(include_cache_hit)
    cache_hit_count = int(include_cache_hit)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE attempts(
                attempt_index INTEGER PRIMARY KEY,
                proposal_solution_ref INTEGER,
                proposal_sha256 TEXT,
                proposal_json TEXT NOT NULL,
                proposal_raw_sha256 TEXT NOT NULL,
                context_json TEXT NOT NULL,
                status TEXT NOT NULL,
                physical_call_started INTEGER NOT NULL,
                charged_evaluation_index INTEGER,
                cache_source_evaluation_index INTEGER,
                failure_code TEXT,
                failure_detail_json TEXT,
                elapsed_monotonic_ns INTEGER NOT NULL,
                prev_attempt_sha256 TEXT,
                attempt_sha256 TEXT
            );
            CREATE TABLE evaluations(
                evaluation_index INTEGER PRIMARY KEY,
                attempt_index INTEGER NOT NULL UNIQUE,
                evidence_partition TEXT NOT NULL,
                search_phase_id TEXT NOT NULL,
                stage_id TEXT NOT NULL,
                type_id INTEGER,
                operator_id TEXT NOT NULL,
                operator_call_id INTEGER NOT NULL,
                proposal_solution_ref INTEGER NOT NULL,
                proposal_sha256 TEXT NOT NULL,
                objectives_json TEXT NOT NULL,
                prev_record_sha256 TEXT NOT NULL,
                record_sha256 TEXT NOT NULL
            );
            CREATE TABLE decisions(
                evaluation_index INTEGER PRIMARY KEY,
                decision_json TEXT NOT NULL,
                prev_decision_sha256 TEXT NOT NULL,
                decision_sha256 TEXT NOT NULL
            );
            CREATE TABLE run_attempt(
                run_id INTEGER PRIMARY KEY,
                family TEXT NOT NULL,
                run_context_json TEXT NOT NULL,
                run_context_digest_sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                terminal_receipt_sha256 TEXT
            );
            CREATE TABLE terminal_receipts(
                run_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                receipt_json TEXT NOT NULL,
                receipt_sha256 TEXT NOT NULL
            );
            """
        )
        operators = ("OP_A", "OP_B", "OP_A", "OP_B")
        previous_attempt_hash = "0" * 64
        previous_evaluation_hash = "0" * 64
        previous_decision_hash = "0" * 64
        for evaluation_index, objective in enumerate(objectives, 1):
            operator = operators[(evaluation_index - 1) % len(operators)]
            screen = None
            if evaluation_index == 2 and screening != "disabled_v1":
                screen = _screen_witness(
                    operator=operator,
                    solutions=((1,), (2,)),
                    seen=(True, False),
                    exhausted=False,
                    generated=1,
                )
            context = {
                "evidence_partition": "development",
                "search_phase_id": "TEST_SEARCH",
                "stage_id": "TEST_STAGE",
                "type_id": (evaluation_index - 1) % frozen_population_size,
                "operator_id": operator,
                "operator_call_id": evaluation_index,
                "operator_witness": (
                    {}
                    if screen is None
                    else {"information_time_candidate_screen": screen}
                ),
            }
            context_json = _canonical(context)
            proposal = {
                "schema": "v21e3_raw_proposal_v1",
                "container_type": "builtins.tuple",
                "iterable": True,
                "items": [{"kind": "int", "value": evaluation_index}],
            }
            proposal_json = _canonical(proposal)
            proposal_raw_sha256 = hashlib.sha256(
                proposal_json.encode("utf-8")
            ).hexdigest()
            proposal_sha256 = hashlib.sha256(
                _canonical([evaluation_index]).encode("utf-8")
            ).hexdigest()
            attempt_semantic = {
                "attempt_index": evaluation_index,
                "proposal_solution_ref": evaluation_index,
                "proposal_sha256": proposal_sha256,
                "proposal_raw": proposal,
                "proposal_raw_sha256": proposal_raw_sha256,
                "evaluation_context": context,
                "status": "EVALUATED",
                "physical_call_started": 1,
                "charged_evaluation_index": evaluation_index,
                "cache_source_evaluation_index": None,
                "failure_code": None,
                "failure_detail": None,
                "run_context_digest_sha256": run_context_digest,
                "prev_attempt_sha256": previous_attempt_hash,
            }
            attempt_sha256 = hashlib.sha256(
                _canonical(attempt_semantic).encode("utf-8")
            ).hexdigest()
            connection.execute(
                "INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    evaluation_index,
                    evaluation_index,
                    proposal_sha256,
                    proposal_json,
                    proposal_raw_sha256,
                    context_json,
                    "EVALUATED",
                    1,
                    evaluation_index,
                    None,
                    None,
                    None,
                    evaluation_index,
                    previous_attempt_hash,
                    attempt_sha256,
                ),
            )
            objective_payload = list(objective)
            evaluation_semantic = {
                "evaluation_index": evaluation_index,
                "attempt_index": evaluation_index,
                "context": context,
                "proposal_solution_ref": evaluation_index,
                "proposal_sha256": proposal_sha256,
                "objectives": objective_payload,
                "run_context_digest_sha256": run_context_digest,
                "prev_record_sha256": previous_evaluation_hash,
            }
            record_sha256 = hashlib.sha256(
                _canonical(evaluation_semantic).encode("utf-8")
            ).hexdigest()
            connection.execute(
                "INSERT INTO evaluations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    evaluation_index,
                    evaluation_index,
                    "development",
                    "TEST_SEARCH",
                    "TEST_STAGE",
                    (evaluation_index - 1) % frozen_population_size,
                    operator,
                    evaluation_index,
                    evaluation_index,
                    proposal_sha256,
                    _canonical(objective_payload),
                    previous_evaluation_hash,
                    record_sha256,
                ),
            )
            decision = {
                "evaluation_index": evaluation_index,
                "accepted_into_population": True,
                "population_replacement_count": 1,
                "population_target_type_ids": [
                    (evaluation_index - 1) % frozen_population_size
                ],
                "decision_reason": "test_initialization",
                "archive_changed": True,
                "retained_after_update": True,
                "archive_size_after": evaluation_index,
                "scalarization_id": None,
                "scalar_parent": None,
                "scalar_candidate": None,
                "scalar_advantage": None,
                "cell_id": None,
                "new_evaluated_cell": None,
                "new_nondominated_cell": None,
                "policy_witness": None,
                "run_context_digest_sha256": run_context_digest,
                "prev_decision_sha256": previous_decision_hash,
            }
            decision_json = _canonical(decision)
            decision_sha256 = hashlib.sha256(
                decision_json.encode("utf-8")
            ).hexdigest()
            connection.execute(
                "INSERT INTO decisions VALUES (?,?,?,?)",
                (
                    evaluation_index,
                    decision_json,
                    previous_decision_hash,
                    decision_sha256,
                ),
            )
            previous_attempt_hash = attempt_sha256
            previous_evaluation_hash = record_sha256
            previous_decision_hash = decision_sha256
        if include_cache_hit:
            cache_screen = _screen_witness(
                operator="OP_A",
                solutions=((2,), (1,)),
                seen=(True, True),
                exhausted=True,
                generated=1,
            )
            cache_context = {
                "evidence_partition": "development",
                "search_phase_id": "TEST_SEARCH",
                "stage_id": "TEST_STAGE",
                "type_id": 0,
                "operator_id": "OP_A",
                "operator_call_id": attempt_count,
                "operator_witness": (
                    {"information_time_candidate_screen": cache_screen}
                    if screening != "disabled_v1"
                    else {}
                ),
            }
            cache_proposal = {
                "schema": "v21e3_raw_proposal_v1",
                "container_type": "builtins.tuple",
                "iterable": True,
                "items": [{"kind": "int", "value": 1}],
            }
            cache_proposal_json = _canonical(cache_proposal)
            cache_raw_sha256 = hashlib.sha256(
                cache_proposal_json.encode("utf-8")
            ).hexdigest()
            source_sha256 = hashlib.sha256(
                _canonical([1]).encode("utf-8")
            ).hexdigest()
            cache_semantic = {
                "attempt_index": attempt_count,
                "proposal_solution_ref": 1,
                "proposal_sha256": source_sha256,
                "proposal_raw": cache_proposal,
                "proposal_raw_sha256": cache_raw_sha256,
                "evaluation_context": cache_context,
                "status": "CACHE_HIT",
                "physical_call_started": 0,
                "charged_evaluation_index": None,
                "cache_source_evaluation_index": 1,
                "failure_code": None,
                "failure_detail": None,
                "run_context_digest_sha256": run_context_digest,
                "prev_attempt_sha256": previous_attempt_hash,
            }
            cache_attempt_sha256 = hashlib.sha256(
                _canonical(cache_semantic).encode("utf-8")
            ).hexdigest()
            connection.execute(
                "INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    attempt_count,
                    1,
                    source_sha256,
                    cache_proposal_json,
                    cache_raw_sha256,
                    _canonical(cache_context),
                    "CACHE_HIT",
                    0,
                    None,
                    1,
                    None,
                    None,
                    attempt_count,
                    previous_attempt_hash,
                    cache_attempt_sha256,
                ),
            )
            previous_attempt_hash = cache_attempt_sha256
        receipt_core = {
            "schema": "v21e3_terminal_receipt_v1",
            "status": "SUCCESS",
            "attempt_count": attempt_count,
            "physical_call_started_count": budget,
            "charged_evaluation_count": budget,
            "decision_count": budget,
            "cache_hit_count": cache_hit_count,
            "unresolved_decision_count": 0,
            "terminal_evaluation_chain_sha256": previous_evaluation_hash,
            "terminal_decision_chain_sha256": previous_decision_hash,
            "terminal_attempt_chain_sha256": previous_attempt_hash,
            "run_context_digest_sha256": run_context_digest,
            "database_path": path.name,
            "durability_mode": "SQLITE_WAL_SYNCHRONOUS_FULL",
        }
        receipt_sha256 = hashlib.sha256(
            _canonical(receipt_core).encode("utf-8")
        ).hexdigest()
        receipt = {**receipt_core, "receipt_payload_sha256": receipt_sha256}
        receipt_json = _canonical(receipt)
        connection.execute(
            "INSERT INTO run_attempt VALUES (1,?,?,?,?,?)",
            (family, run_context_json, run_context_digest, "SUCCESS", receipt_sha256),
        )
        connection.execute(
            "INSERT INTO terminal_receipts VALUES (1,'SUCCESS',?,?)",
            (receipt_json, receipt_sha256),
        )
        connection.commit()
        path.with_name("terminal.json").write_bytes(receipt_json.encode("utf-8"))
    finally:
        connection.close()


def _write_real_v9_trace(directory: Path) -> Path:
    database = directory / "trace.sqlite3"
    problem = MultiObjectiveKnapsackInstance.random_instance(
        8, num_objectives=2, seed=907
    )
    config = V21E3HybridConfig(
        candidate_id="C0",
        reference_directions=((0.25, 0.75), (0.75, 0.25)),
        charged_evaluations=8,
        checkpoint_period=2,
        seed=17,
        phase="development",
        trace_database=str(database),
        terminal_receipt=str(directory / "terminal.json"),
        receipt_database_path="trace.sqlite3",
        capture_trace=True,
        development_diagnostic_id="V21E3R1_V9_INFORMATION_LYAPUNOV_MOKP",
        candidate_screening_policy=(
            "bounded_cache_aware_structural_screen_development_v1"
        ),
        candidate_screening_cap=4,
        replacement_policy=(
            "archive_compensated_information_lyapunov_development_v1"
        ),
        archive_tradeoff_lambda=0.5,
        attempt_cap=128,
        structural_screening_cap=2048,
        wall_time_cap_seconds=30.0,
    )
    V21E3TypedHybridParetoSearch(problem, config).run()
    return database


def test_strict_diagnostic_rejects_objective_tamper_via_evaluation_chain(
    tmp_path: Path,
) -> None:
    database = _write_real_v9_trace(tmp_path)
    connection = sqlite3.connect(database)
    try:
        run_context = json.loads(
            connection.execute(
                "SELECT run_context_json FROM run_attempt WHERE run_id=1"
            ).fetchone()[0]
        )
        lower = run_context["objective_lower_bounds"]
        upper = run_context["objective_upper_bounds"]
        tampered = [
            float(lo) + 0.123456789 * (float(hi) - float(lo))
            for lo, hi in zip(lower, upper)
        ]
        connection.execute(
            "UPDATE evaluations SET objectives_json=? WHERE evaluation_index=1",
            (_canonical(tampered),),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(V9TraceDiagnosticError, match="Evaluation semantic hash chain"):
        analyze_v9_trace_database(database)


def test_strict_diagnostic_rejects_cache_attempt_context_tamper_via_attempt_chain(
    tmp_path: Path,
) -> None:
    database = tmp_path / "trace.sqlite3"
    _write_trace(database)
    connection = sqlite3.connect(database)
    try:
        attempt_count = connection.execute(
            "SELECT MAX(attempt_index) FROM attempts"
        ).fetchone()[0]
        context = json.loads(
            connection.execute(
                "SELECT context_json FROM attempts WHERE attempt_index=?",
                (attempt_count,),
            ).fetchone()[0]
        )
        context["unbound_tamper"] = True
        connection.execute(
            "UPDATE attempts SET context_json=? WHERE attempt_index=?",
            (_canonical(context), attempt_count),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(V9TraceDiagnosticError, match="Attempt semantic hash chain"):
        analyze_v9_trace_database(database)


def test_strict_diagnostic_rejects_decision_payload_tamper_via_decision_chain(
    tmp_path: Path,
) -> None:
    database = tmp_path / "trace.sqlite3"
    _write_trace(database)
    connection = sqlite3.connect(database)
    try:
        decision = json.loads(
            connection.execute(
                "SELECT decision_json FROM decisions WHERE evaluation_index=2"
            ).fetchone()[0]
        )
        decision["unbound_tamper"] = True
        connection.execute(
            "UPDATE decisions SET decision_json=? WHERE evaluation_index=2",
            (_canonical(decision),),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(V9TraceDiagnosticError, match="Decision semantic hash chain"):
        analyze_v9_trace_database(database)


def test_strict_diagnostic_rejects_rehashed_wrong_terminal_chain_binding(
    tmp_path: Path,
) -> None:
    database = tmp_path / "trace.sqlite3"
    _write_trace(database)
    connection = sqlite3.connect(database)
    try:
        receipt = json.loads(
            connection.execute(
                "SELECT receipt_json FROM terminal_receipts WHERE run_id=1"
            ).fetchone()[0]
        )
        receipt["terminal_evaluation_chain_sha256"] = "f" * 64
        receipt_core = dict(receipt)
        receipt_core.pop("receipt_payload_sha256")
        receipt_sha256 = hashlib.sha256(
            _canonical(receipt_core).encode("utf-8")
        ).hexdigest()
        receipt["receipt_payload_sha256"] = receipt_sha256
        receipt_json = _canonical(receipt)
        connection.execute(
            "UPDATE terminal_receipts SET receipt_json=?,receipt_sha256=? WHERE run_id=1",
            (receipt_json, receipt_sha256),
        )
        connection.execute(
            "UPDATE run_attempt SET terminal_receipt_sha256=? WHERE run_id=1",
            (receipt_sha256,),
        )
        connection.commit()
        database.with_name("terminal.json").write_bytes(receipt_json.encode("utf-8"))
    finally:
        connection.close()

    with pytest.raises(V9TraceDiagnosticError, match="terminal evaluation chain binding"):
        analyze_v9_trace_database(database)


def test_strict_diagnostic_rejects_detached_terminal_receipt_mismatch(
    tmp_path: Path,
) -> None:
    database = tmp_path / "trace.sqlite3"
    _write_trace(database)
    database.with_name("terminal.json").write_bytes(b'{"tampered":true}')

    with pytest.raises(V9TraceDiagnosticError, match="detached terminal receipt"):
        analyze_v9_trace_database(database)


def test_strict_diagnostic_rejects_missing_detached_terminal_receipt(
    tmp_path: Path,
) -> None:
    database = tmp_path / "trace.sqlite3"
    _write_trace(database)
    database.with_name("terminal.json").unlink()

    with pytest.raises(V9TraceDiagnosticError, match="detached terminal receipt is missing"):
        analyze_v9_trace_database(database)


def test_strict_diagnostic_rebuilds_hv_and_screening_by_operator(
    tmp_path: Path,
) -> None:
    database = tmp_path / "trace # percent%.sqlite3"
    _write_trace(database)
    before = hashlib.sha256(database.read_bytes()).hexdigest()

    report = analyze_v9_trace_database(database)

    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    assert report["schema"] == "v21e3r1_v9_operator_productivity_diagnostic_v3"
    assert report["status"] == "DEVELOPMENT_ONLY_NO_LATER_PHASE_AUTHORIZATION"
    assert report["implementation_independence"] is False
    assert report["policy_witness_independent_hv_reconstruction"] is True
    assert report["validation"] == {
        "sqlite_read_only_uri": True,
        "sqlite_query_only": True,
        "sqlite_integrity": "ok",
        "terminal_success": True,
        "contiguous_attempts": True,
        "contiguous_evaluations": True,
        "complete_decisions": True,
        "accounting_consistent": True,
        "attempt_semantic_hash_chain": True,
        "evaluation_semantic_hash_chain": True,
        "decision_semantic_hash_chain": True,
        "terminal_chain_bindings": True,
        "detached_terminal_receipt_exact_match": True,
        "detached_terminal_receipt_external_sha256_bound": False,
        "lyapunov_witness_durable_state_arithmetic": (
            "DURABLE_STATE_ARITHMETIC_REPLAY_PASS"
        ),
    }
    assert report["final_normalized_hv"] == pytest.approx(0.27)
    assert report["total_reconstructed_hv_gain"] == pytest.approx(0.27)
    by_operator = {row["operator"]: row for row in report["operators"]}
    assert by_operator["OP_A"]["attempts"] == 3
    assert by_operator["OP_A"]["cache_hits"] == 1
    assert by_operator["OP_A"]["first_evaluations"] == 2
    assert by_operator["OP_A"]["screenings"] == 2
    assert by_operator["OP_A"]["screen_cache_skips"] == 2
    assert by_operator["OP_A"]["hv_gain"] == pytest.approx(0.22)
    assert by_operator["OP_B"]["attempts"] == 2
    assert by_operator["OP_B"]["first_evaluations"] == 2
    assert by_operator["OP_B"]["screenings"] == 2
    assert by_operator["OP_B"]["screen_cache_skips"] == 1
    assert by_operator["OP_B"]["hv_gain"] == pytest.approx(0.05)


def test_strict_diagnostic_reconstructs_left_continuous_auc_and_post_init_gain(
    tmp_path: Path,
) -> None:
    database = tmp_path / "trace.sqlite3"
    _write_trace(
        database,
        diagnostic_id="V21E3R1_V9_INFORMATION_SCREEN_MOKP",
        population_size=2,
    )

    report = analyze_v9_trace_database(database)

    assert report["population_size"] == 2
    assert report["initialization_end_evaluation"] == 2
    assert report["initialization_terminal_hv"] == pytest.approx(0.08)
    assert report["exact_per_evaluation_left_continuous_hv_auc"] == pytest.approx(
        (0.0 + 0.04 + 0.08 + 0.26) / 4.0
    )
    assert report["post_initialization_incremental_hv_gain"] == pytest.approx(0.19)


def test_strict_diagnostic_replays_lyapunov_witnesses_against_durable_state(
    tmp_path: Path,
) -> None:
    database = _write_real_v9_trace(tmp_path)

    report = analyze_v9_trace_database(database)

    assert report["lyapunov_witness_count"] == 6
    assert report["lyapunov_witness_violation_count"] == 0
    assert report["lyapunov_witness_replay"] == (
        "DURABLE_STATE_ARITHMETIC_REPLAY_PASS"
    )
    assert report["full_algorithm_decision_replay"] == "NOT_IMPLEMENTED"


def test_strict_diagnostic_rejects_coherently_rehashed_false_lyapunov_witness(
    tmp_path: Path,
) -> None:
    database = _write_real_v9_trace(tmp_path)
    connection = sqlite3.connect(database)
    try:
        rows = list(
            connection.execute(
                "SELECT evaluation_index,decision_json FROM decisions "
                "ORDER BY evaluation_index"
            )
        )
        previous = "0" * 64
        tampered = False
        for evaluation_index, raw in rows:
            decision = json.loads(raw)
            witness = decision.get("policy_witness")
            if not tampered and isinstance(witness, dict):
                witness["normalized_hv_after"] = (
                    float(witness["normalized_hv_after"]) + 0.125
                )
                tampered = True
            decision["prev_decision_sha256"] = previous
            decision_json = _canonical(decision)
            decision_sha256 = hashlib.sha256(
                decision_json.encode("utf-8")
            ).hexdigest()
            connection.execute(
                "UPDATE decisions SET decision_json=?,prev_decision_sha256=?,"
                "decision_sha256=? WHERE evaluation_index=?",
                (decision_json, previous, decision_sha256, evaluation_index),
            )
            previous = decision_sha256
        assert tampered
        receipt = json.loads(
            connection.execute(
                "SELECT receipt_json FROM terminal_receipts WHERE run_id=1"
            ).fetchone()[0]
        )
        receipt["terminal_decision_chain_sha256"] = previous
        receipt_core = dict(receipt)
        receipt_core.pop("receipt_payload_sha256")
        receipt_sha256 = hashlib.sha256(
            _canonical(receipt_core).encode("utf-8")
        ).hexdigest()
        receipt["receipt_payload_sha256"] = receipt_sha256
        receipt_json = _canonical(receipt)
        connection.execute(
            "UPDATE terminal_receipts SET receipt_json=?,receipt_sha256=? WHERE run_id=1",
            (receipt_json, receipt_sha256),
        )
        connection.execute(
            "UPDATE run_attempt SET terminal_receipt_sha256=? WHERE run_id=1",
            (receipt_sha256,),
        )
        connection.commit()
        database.with_name("terminal.json").write_bytes(receipt_json.encode("utf-8"))
    finally:
        connection.close()

    with pytest.raises(
        V9TraceDiagnosticError, match="durable-state arithmetic replay"
    ):
        analyze_v9_trace_database(database)


@pytest.mark.parametrize(
    "diagnostic_id",
    [
        f"V21E3R1_V9_{arm}_{family}"
        for family in ("MOKP", "MOTSP")
        for arm in (
            "LEGACY",
            "INFORMATION_SCREEN",
            "LYAPUNOV",
            "INFORMATION_LYAPUNOV",
        )
    ],
)
def test_strict_diagnostic_accepts_every_exact_v9_arm(
    tmp_path: Path,
    diagnostic_id: str,
) -> None:
    database = tmp_path / f"{diagnostic_id}.sqlite3"
    _write_trace(
        database,
        diagnostic_id=diagnostic_id,
        objectives=((0.5, 0.5),),
        include_cache_hit=False,
    )

    report = analyze_v9_trace_database(database)

    assert report["development_diagnostic_id"] == diagnostic_id
    assert report["family"] == diagnostic_id.rsplit("_", 1)[-1]
    assert report["final_normalized_hv"] == pytest.approx(0.25)


def test_strict_diagnostic_rejects_missing_objective_bounds(tmp_path: Path) -> None:
    database = tmp_path / "missing-bounds.sqlite3"
    _write_trace(database, omit_lower_bound=True)

    with pytest.raises(V9TraceDiagnosticError, match="objective_lower_bounds"):
        analyze_v9_trace_database(database)


@pytest.mark.parametrize("tamper", ["terminal", "attempt_gap", "decision"])
def test_strict_diagnostic_fails_closed_on_incomplete_trace(
    tmp_path: Path,
    tamper: str,
) -> None:
    database = tmp_path / f"{tamper}.sqlite3"
    _write_trace(database)
    connection = sqlite3.connect(database)
    try:
        if tamper == "terminal":
            connection.execute("UPDATE run_attempt SET status='STARTED'")
        elif tamper == "attempt_gap":
            connection.execute("UPDATE attempts SET attempt_index=9 WHERE attempt_index=5")
        else:
            connection.execute("DELETE FROM decisions WHERE evaluation_index=4")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(V9TraceDiagnosticError):
        analyze_v9_trace_database(database)


def test_diagnostic_cli_writes_canonical_report_without_overwrite(
    tmp_path: Path,
) -> None:
    database = tmp_path / "trace.sqlite3"
    output = tmp_path / "diagnostic.json"
    _write_trace(database)

    assert main(["--trace", str(database), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert output.read_bytes() == _canonical(report).encode("utf-8") + b"\n"
    assert report["status"] == "DEVELOPMENT_ONLY_NO_LATER_PHASE_AUTHORIZATION"
    with pytest.raises(FileExistsError):
        main(["--trace", str(database), "--output", str(output)])


def test_diagnostic_cli_accepts_explicit_detached_receipt_external_hash(
    tmp_path: Path,
) -> None:
    database = tmp_path / "trace.sqlite3"
    detached = tmp_path / "terminal.json"
    output = tmp_path / "diagnostic.json"
    _write_trace(database)
    detached_sha256 = hashlib.sha256(detached.read_bytes()).hexdigest()

    assert (
        main(
            [
                "--trace",
                str(database),
                "--terminal-receipt",
                str(detached),
                "--expected-terminal-receipt-sha256",
                detached_sha256,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["detached_terminal_receipt_externally_bound"] is True
    assert report["detached_terminal_receipt_sha256"] == detached_sha256
