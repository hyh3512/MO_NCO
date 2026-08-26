from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mo_nco.pareto_v21e3_gate import select_complexity_first_candidate


def _write_matrix(
    root: Path,
    effects: dict[str, dict[str, float]],
) -> tuple[Path, Path]:
    rows: list[dict[str, object]] = []
    arms = ("C0", "C1", "C2", "C3")
    seeds = (101, 103)
    for family in ("MOTSP", "MOKP"):
        for case_index in range(5):
            base = 0.30 + case_index * 0.001
            for seed in seeds:
                for arm in arms:
                    rows.append(
                        {
                            "schema": "pareto_v21e3_calibration_run_row_v1",
                            "family": family,
                            "case_id": f"{family}-{case_index}",
                            "candidate_id": arm,
                            "seed": seed,
                            "normalized_hv_auc": base
                            + effects[family].get(arm, 0.0),
                            "trace_verification_status": "PASS",
                            "attempt_history_gate": "PASS",
                            "objective_contract_gate": "PASS",
                            "charged_budget_gate": "PASS",
                        }
                    )
    rows_path = root / "rows.jsonl"
    rows_raw = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
        for row in rows
    )
    rows_path.write_bytes(rows_raw)
    receipt = {
        "schema": "pareto_v21e3_calibration_matrix_receipt_v1",
        "status": "PASS",
        "candidate_ids": list(arms),
        "seeds": list(seeds),
        "expected_rows": len(rows),
        "completed_rows": len(rows),
        "all_trace_verifications_pass": True,
        "attempt_history_gate": "PASS",
        "objective_contract_gate": "PASS",
        "charged_budget_gate": "PASS",
        "artifact_root_gate": "PASS",
        "rows_sha256": hashlib.sha256(rows_raw).hexdigest(),
    }
    receipt_path = root / "matrix_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return rows_path, receipt_path


def test_complexity_first_gate_keeps_simplest_candidate_without_practical_adjacent_gain(
    tmp_path: Path,
) -> None:
    rows, receipt = _write_matrix(
        tmp_path,
        {
            "MOTSP": {"C1": 0.010, "C2": 0.0104, "C3": 0.014},
            "MOKP": {"C1": 0.011, "C2": 0.0113, "C3": 0.015},
        },
    )

    result = select_complexity_first_candidate(
        rows_path=rows,
        matrix_receipt_path=receipt,
        output_path=tmp_path / "selection.json",
        candidate_chain=("C0", "C1", "C2", "C3"),
        expected_seeds=(101, 103),
        expected_cases_per_family=5,
        primary_delta_min=0.002,
        adjacent_delta_min=0.002,
        bootstrap_samples=200,
        randomization_seed=22041,
    )

    assert result["status"] == "PASS"
    assert result["candidate_selected"] == "C1"
    assert result["candidate_results"]["C2"]["adjacent_gate"] == "FAIL"
    assert result["candidate_results"]["C3"]["gate"] == "NOT_REACHED"
    assert result["selection_rule"] == "STRICT_COMPLEXITY_FIRST_CHAIN_V1"


def test_complexity_first_gate_promotes_only_when_each_adjacent_ci_is_positive_and_practical(
    tmp_path: Path,
) -> None:
    rows, receipt = _write_matrix(
        tmp_path,
        {
            "MOTSP": {"C1": 0.006, "C2": 0.009, "C3": 0.012},
            "MOKP": {"C1": 0.007, "C2": 0.010, "C3": 0.013},
        },
    )

    result = select_complexity_first_candidate(
        rows_path=rows,
        matrix_receipt_path=receipt,
        output_path=tmp_path / "selection.json",
        candidate_chain=("C0", "C1", "C2", "C3"),
        expected_seeds=(101, 103),
        expected_cases_per_family=5,
        primary_delta_min=0.002,
        adjacent_delta_min=0.002,
        bootstrap_samples=200,
        randomization_seed=22043,
    )

    assert result["status"] == "PASS"
    assert result["candidate_selected"] == "C3"
    assert result["candidate_results"]["C2"]["adjacent_gate"] == "PASS"
    assert result["candidate_results"]["C3"]["adjacent_gate"] == "PASS"


def test_complexity_first_gate_stops_if_the_minimal_treatment_fails_primary_gate(
    tmp_path: Path,
) -> None:
    rows, receipt = _write_matrix(
        tmp_path,
        {
            "MOTSP": {"C1": 0.001, "C2": 0.010, "C3": 0.015},
            "MOKP": {"C1": 0.001, "C2": 0.011, "C3": 0.016},
        },
    )

    result = select_complexity_first_candidate(
        rows_path=rows,
        matrix_receipt_path=receipt,
        output_path=tmp_path / "selection.json",
        candidate_chain=("C0", "C1", "C2", "C3"),
        expected_seeds=(101, 103),
        expected_cases_per_family=5,
        primary_delta_min=0.002,
        adjacent_delta_min=0.002,
        bootstrap_samples=200,
        randomization_seed=22047,
    )

    assert result["status"] == "STOP"
    assert result["candidate_selected"] is None
    assert result["formal_materialization_authorized"] is False
    assert result["candidate_results"]["C2"]["gate"] == "NOT_REACHED"

