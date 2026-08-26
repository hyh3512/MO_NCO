from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mo_nco.pareto_v21e3_artifacts import ArtifactRoot
from mo_nco.pareto_v21e3_common_runner import (
    FormalMaterializationProhibited,
    preflight_formal_common_runner,
    run_formal_matrix,
)


def _write(root: Path, relative: str, payload: object) -> dict[str, object]:
    raw = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {
        "path": relative,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def test_common_runner_preflight_is_four_arm_bound_but_formal_is_prohibited(
    tmp_path: Path,
) -> None:
    timing = _write(
        tmp_path,
        "protocol/timing.json",
        {
            "schema": "pareto_v21e3_timing_policy_v1",
                "clock": "time.monotonic_ns_process_local_elapsed_v1",
                "elapsed_fields": ["elapsed_monotonic_ns"],
            "semantic_role": "DIAGNOSTIC_ONLY_NOT_IN_ALGORITHM_HASH_CHAIN",
            "formal_quality_decision_uses_timing": False,
            "resource_efficiency_claim_authorized": False,
            "future_efficiency_protocol_required": True,
        },
    )
    trace = _write(
        tmp_path,
        "protocol/trace.json",
        {
            "schema": "pareto_v21e3_trace_storage_policy_v1",
            "implementation_status": "PROTOTYPE_ONLY",
            "formal_authorized": False,
            "codec": "canonical_jsonl_zlib_chunk_v1",
            "chunk_hash_chain": "sha256_header_and_compressed_payload_v1",
            "restore_replay_gate": "REQUIRED_BEFORE_FORMAL",
        },
    )
    arms = []
    for arm_id in ("V21E3_CANDIDATE", "V21E3_C0", "NSGAII", "MOEAD"):
        source = _write(tmp_path, f"sources/{arm_id}.json", {"arm_id": arm_id})
        arms.append(
            {
                "arm_id": arm_id,
                "families": ["MOTSP", "MOKP"],
                "objective_call_semantics": "first_true_objective_evaluation_v1",
                "attempt_history_semantics": "all_attempts_terminal_receipt_v1",
                "source_bindings": [source],
                "execution_adapter_status": (
                    "DEVELOPMENT_ONLY_AVAILABLE"
                    if arm_id == "V21E3_C0"
                    else "NOT_IMPLEMENTED"
                ),
            }
        )
    context = _write(
        tmp_path,
        "protocol/common_runner.json",
        {
            "schema": "pareto_v21e3_formal_common_runner_context_v1",
            "status": "PROTOTYPE_ONLY_FORMAL_MATERIALIZATION_PROHIBITED",
            "artifact_root_id": "fixture-root-v1",
            "arms": arms,
            "evaluation_budgets": [10000, 50000, 100000],
            "checkpoint_fractions": [0.1, 0.3, 0.5, 0.7, 1.0],
            "timing_policy": timing,
            "trace_storage_policy": trace,
            "formal_case_manifest": None,
            "future_external_entropy_status": "NOT_ESTABLISHED",
            "formal_authorized": False,
            "formal_cases_status": "NOT_MATERIALIZED",
        },
    )

    receipt = preflight_formal_common_runner(ArtifactRoot(tmp_path), context)

    assert receipt["status"] == "PROTOTYPE_PASS_FORMAL_PROHIBITED"
    assert receipt["arm_ids"] == [
        "V21E3_CANDIDATE",
        "V21E3_C0",
        "NSGAII",
        "MOEAD",
    ]
    assert receipt["formal_cases_status"] == "NOT_MATERIALIZED"
    assert receipt["formal_authorized"] is False
    assert receipt["common_budget_parity_status"] == "NOT_ESTABLISHED"
    assert receipt["execution_adapter_status_by_arm"]["V21E3_C0"] == (
        "DEVELOPMENT_ONLY_AVAILABLE"
    )
    assert receipt["execution_adapter_status_by_arm"]["NSGAII"] == (
        "NOT_IMPLEMENTED"
    )
    with pytest.raises(FormalMaterializationProhibited):
        run_formal_matrix(receipt)

