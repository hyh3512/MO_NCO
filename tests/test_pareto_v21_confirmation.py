from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from mo_nco.pareto_v21_confirmation import evaluate_v21_confirmation_gate


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, payload: object) -> str:
    raw = _canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _build_confirmation_fixture(
    root: Path,
    *,
    predecessor: str = "C1",
    candidate_effects: dict[str, float] | None = None,
) -> tuple[Path, Path]:
    candidate_effects = candidate_effects or {"MOKP": 0.02, "MOTSP": 0.02}
    run_context_sha256 = "a" * 64
    cases = [
        {"case_id": f"{family}-{index}", "family": family}
        for family in ("MOKP", "MOTSP")
        for index in range(3)
    ]
    manifest_path = root / "case_manifest.json"
    manifest_sha = _write_json(
        manifest_path,
        {
            "schema": "pareto_v21_partition_manifest_v1",
            "split": "confirmation",
            "cases": cases,
        },
    )
    arms = tuple(dict.fromkeys(("C2", "C0", predecessor)))
    rows = []
    for case in cases:
        for arm in arms:
            increment = (
                candidate_effects[case["family"]]
                if arm == "C2"
                else {"C0": 0.0, "C1": 0.01}[arm]
            )
            for seed in (31031, 31037):
                trace_relative = (
                    Path("traces") / arm / case["case_id"] / f"seed-{seed}.sqlite3"
                )
                trace_path = root / trace_relative
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                trace_bytes = f"{case['case_id']}|{arm}|{seed}".encode("ascii")
                trace_path.write_bytes(trace_bytes)
                trace_sha = hashlib.sha256(trace_bytes).hexdigest()
                rows.append(
                    {
                        "schema": "pareto_v21_calibration_run_row_v1",
                        "split": "confirmation",
                        "case_id": case["case_id"],
                        "family": case["family"],
                        "candidate_id": arm,
                        "seed": seed,
                        "evaluation_budget": 3000,
                        "checkpoint_period": 300,
                        "normalized_hv_auc": 0.30 + increment,
                        "run_context_sha256": run_context_sha256,
                        "trace_verification_status": "PASS",
                        "trace_database_sha256": trace_sha,
                        "terminal_evaluation_chain_sha256": "1" * 64,
                        "terminal_decision_chain_sha256": "2" * 64,
                        "terminal_mechanism_chain_sha256": "3" * 64,
                        "trace_relative_path": trace_relative.as_posix(),
                    }
                )
    rows_path = root / "run_rows.jsonl"
    rows_path.write_bytes(
        b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    )
    _write_json(
        root / "matrix_receipt.json",
        {
            "schema": "pareto_v21_calibration_matrix_receipt_v1",
            "status": "PASS",
            "split": "confirmation",
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": manifest_sha,
            "candidate_ids": arms,
            "seeds": (31031, 31037),
            "evaluation_budget": 3000,
            "checkpoint_period": 300,
            "expected_rows": len(rows),
            "completed_rows": len(rows),
            "all_trace_verifications_pass": True,
            "full_partition_binding_gate": "PASS",
            "run_context_binding_gate": "PASS",
            "run_context_sha256": run_context_sha256,
            "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        },
    )
    thresholds_path = root / "confirmation_gate_precommit.json"
    _write_json(
        thresholds_path,
        {
            "schema": "pareto_v21_confirmation_gate_precommit_v1",
            "status": "FROZEN_BEFORE_CONFIRMATION_RUNS",
            "selected_candidate": "C2",
            "control_id": "C0",
            "predecessor_id": predecessor,
            "confirmation_manifest_sha256": manifest_sha,
            "expected_families": ["MOKP", "MOTSP"],
            "expected_cases_per_family": 3,
            "expected_seeds": [31031, 31037],
            "expected_evaluation_budget": 3000,
            "expected_checkpoint_period": 300,
            "primary_metric": "normalized_hv_auc",
            "delta_min": 0.005,
            "noninferiority_margin": 0.005,
            "mechanism_noninferiority_margin": 0.005,
            "bootstrap_samples": 500,
            "bootstrap_randomization_seed": 21071,
            "tie_tolerance": 0.0,
            "trim_fraction_each_tail": 0.10,
        },
    )
    return rows_path, thresholds_path


def test_confirmation_gate_passes_only_complete_bound_confirmation_matrix(
    tmp_path,
) -> None:
    rows_path, thresholds_path = _build_confirmation_fixture(tmp_path)
    output_path = tmp_path / "confirmation_gate_receipt.json"

    receipt = evaluate_v21_confirmation_gate(
        rows_path=rows_path,
        thresholds_path=thresholds_path,
        output_path=output_path,
        selected_candidate="C2",
        control_id="C0",
        predecessor_id="C1",
    )

    assert receipt["schema"] == "pareto_v21_confirmation_gate_receipt_v1"
    assert receipt["status"] == "PASS"
    assert receipt["gate_status"] == "PASS"
    assert receipt["integrity_gate"] == "PASS"
    assert receipt["primary_gate"] == "PASS"
    assert receipt["adjacent_mechanism_gate"] == "PASS"
    assert receipt["expected_rows"] == receipt["observed_rows"] == 36
    assert receipt["action"] == (
        "AUTHORIZE_CANDIDATE_FREEZE_AND_FORMAL_ENTROPY_REVEAL"
    )
    assert set(receipt["primary_family_results"]) == {"MOKP", "MOTSP"}
    assert set(receipt["mechanism_family_results"]) == {"MOKP", "MOTSP"}
    assert output_path.read_bytes() == _canonical_bytes(receipt)
    with pytest.raises(FileExistsError):
        evaluate_v21_confirmation_gate(
            rows_path=rows_path,
            thresholds_path=thresholds_path,
            output_path=output_path,
            selected_candidate="C2",
            control_id="C0",
            predecessor_id="C1",
        )


def test_confirmation_gate_fails_on_one_family_primary_effect(tmp_path) -> None:
    rows_path, thresholds_path = _build_confirmation_fixture(
        tmp_path,
        candidate_effects={"MOKP": 0.02, "MOTSP": -0.01},
    )

    receipt = evaluate_v21_confirmation_gate(
        rows_path=rows_path,
        thresholds_path=thresholds_path,
        output_path=tmp_path / "failed_gate.json",
        selected_candidate="C2",
        control_id="C0",
        predecessor_id="C1",
    )

    assert receipt["integrity_gate"] == "PASS"
    assert receipt["gate_status"] == "FAIL"
    assert receipt["primary_gate"] == "FAIL"
    assert receipt["primary_family_results"]["MOTSP"]["gate"] == "FAIL"
    assert receipt["action"] == "STOP_BEFORE_FORMAL_MATERIALIZATION"


def test_confirmation_gate_holds_before_statistics_on_trace_hash_failure(
    tmp_path,
) -> None:
    rows_path, thresholds_path = _build_confirmation_fixture(tmp_path)
    first_row = json.loads(rows_path.read_text(encoding="utf-8").splitlines()[0])
    (tmp_path / first_row["trace_relative_path"]).write_bytes(b"tampered")

    receipt = evaluate_v21_confirmation_gate(
        rows_path=rows_path,
        thresholds_path=thresholds_path,
        output_path=tmp_path / "trace_hold.json",
        selected_candidate="C2",
        control_id="C0",
        predecessor_id="C1",
    )

    assert receipt["gate_status"] == "HOLD"
    assert receipt["completeness_gate"] == "PASS"
    assert receipt["trace_verified_gate"] == "FAIL"
    assert receipt["partition_binding_gate"] == "PASS"
    assert receipt["primary_gate"] == "NOT_RUN_DUE_TO_INTEGRITY_HOLD"
    assert receipt["primary_family_results"] == {}
    assert receipt["action"] == "STOP_BEFORE_FORMAL_MATERIALIZATION"


def test_confirmation_gate_reuses_primary_when_predecessor_is_control(
    tmp_path,
) -> None:
    rows_path, thresholds_path = _build_confirmation_fixture(
        tmp_path,
        predecessor="C0",
    )

    receipt = evaluate_v21_confirmation_gate(
        rows_path=rows_path,
        thresholds_path=thresholds_path,
        output_path=tmp_path / "reused_primary.json",
        selected_candidate="C2",
        control_id="C0",
        predecessor_id="C0",
    )

    assert receipt["gate_status"] == "PASS"
    assert receipt["expected_rows"] == receipt["observed_rows"] == 24
    assert receipt["adjacent_comparison_reused_primary"] is True
    assert all(
        result["comparison_source"] == "PRIMARY_CANDIDATE_VS_CONTROL_REUSED"
        and "comparison" not in result
        for result in receipt["mechanism_family_results"].values()
    )


def test_confirmation_gate_cli_writes_authoritative_receipt(tmp_path) -> None:
    rows_path, thresholds_path = _build_confirmation_fixture(tmp_path)
    output_path = tmp_path / "cli_receipt.json"
    script = (
        Path(__file__).parents[1]
        / "ijoc_submission_v21"
        / "scripts"
        / "evaluate_confirmation_gate.py"
    )

    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--rows",
            str(rows_path),
            "--thresholds",
            str(thresholds_path),
            "--output",
            str(output_path),
            "--selected-candidate",
            "C2",
            "--control",
            "C0",
            "--predecessor",
            "C1",
        ],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, process.stderr
    printed = json.loads(process.stdout)
    stored = json.loads(output_path.read_text(encoding="utf-8"))
    assert printed == stored
    assert stored["gate_status"] == "PASS"


def test_confirmation_gate_holds_on_partition_manifest_drift(tmp_path) -> None:
    rows_path, thresholds_path = _build_confirmation_fixture(tmp_path)
    manifest_path = tmp_path / "case_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["post_freeze_mutation"] = True
    manifest_path.write_bytes(_canonical_bytes(manifest))

    receipt = evaluate_v21_confirmation_gate(
        rows_path=rows_path,
        thresholds_path=thresholds_path,
        output_path=tmp_path / "partition_hold.json",
        selected_candidate="C2",
        control_id="C0",
        predecessor_id="C1",
    )

    assert receipt["gate_status"] == "HOLD"
    assert receipt["completeness_gate"] == "PASS"
    assert receipt["trace_verified_gate"] == "PASS"
    assert receipt["partition_binding_gate"] == "FAIL"
    assert receipt["primary_gate"] == "NOT_RUN_DUE_TO_INTEGRITY_HOLD"


def test_confirmation_gate_holds_on_incomplete_run_key_matrix(tmp_path) -> None:
    rows_path, thresholds_path = _build_confirmation_fixture(tmp_path)
    lines = rows_path.read_bytes().splitlines(keepends=True)
    rows_path.write_bytes(b"".join(lines[:-1]))
    matrix_path = tmp_path / "matrix_receipt.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix["rows_sha256"] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    matrix["completed_rows"] = len(lines) - 1
    matrix_path.write_bytes(_canonical_bytes(matrix))

    receipt = evaluate_v21_confirmation_gate(
        rows_path=rows_path,
        thresholds_path=thresholds_path,
        output_path=tmp_path / "incomplete_hold.json",
        selected_candidate="C2",
        control_id="C0",
        predecessor_id="C1",
    )

    assert receipt["gate_status"] == "HOLD"
    assert receipt["completeness_gate"] == "FAIL"
    assert receipt["trace_verified_gate"] == "PASS"
    assert receipt["partition_binding_gate"] == "PASS"
    assert receipt["primary_family_results"] == {}


def test_confirmation_gate_holds_on_row_run_context_detachment(tmp_path) -> None:
    rows_path, thresholds_path = _build_confirmation_fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["run_context_sha256"] = "b" * 64
    rows_path.write_bytes(
        b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    )
    matrix_path = tmp_path / "matrix_receipt.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix["rows_sha256"] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    matrix_path.write_bytes(_canonical_bytes(matrix))

    receipt = evaluate_v21_confirmation_gate(
        rows_path=rows_path,
        thresholds_path=thresholds_path,
        output_path=tmp_path / "context_hold.json",
        selected_candidate="C2",
        control_id="C0",
        predecessor_id="C1",
    )

    assert receipt["gate_status"] == "HOLD"
    assert receipt["completeness_gate"] == "PASS"
    assert receipt["run_context_binding_gate"] == "FAIL"
    assert receipt["primary_gate"] == "NOT_RUN_DUE_TO_INTEGRITY_HOLD"


def test_confirmation_gate_accepts_v21e2_candidate_menu_precommit(tmp_path) -> None:
    rows_path, thresholds_path = _build_confirmation_fixture(tmp_path)
    frozen = json.loads(thresholds_path.read_text(encoding="utf-8"))
    thresholds_path.write_bytes(
        _canonical_bytes(
            {
                "schema": "pareto_v21_candidate_menu_precommit_v2",
                "status": "FROZEN_BEFORE_SELECTION_RUNS",
                "candidate_menu": [
                    {"candidate_id": value}
                    for value in ("C0", "C1", "C2", "C3", "C4")
                ],
                "selection_gate": {
                    "control_id": "C0",
                    "delta_min": frozen["delta_min"],
                    "noninferiority_margin": frozen["noninferiority_margin"],
                    "mechanism_noninferiority_margin": frozen[
                        "mechanism_noninferiority_margin"
                    ],
                    "tie_tolerance": frozen["tie_tolerance"],
                    "trim_fraction_each_tail": frozen[
                        "trim_fraction_each_tail"
                    ],
                },
                "confirmation_design_if_authorized": {
                    "partition_manifest": {
                        "sha256": frozen["confirmation_manifest_sha256"]
                    },
                    "seeds": frozen["expected_seeds"],
                    "evaluation_budget": frozen["expected_evaluation_budget"],
                    "checkpoint_period": frozen["expected_checkpoint_period"],
                    "expected_cases_per_family": frozen[
                        "expected_cases_per_family"
                    ],
                    "gate_thresholds_identical_to_selection": True,
                    "bootstrap_samples": frozen["bootstrap_samples"],
                    "bootstrap_randomization_seed": frozen[
                        "bootstrap_randomization_seed"
                    ],
                },
            }
        )
    )

    receipt = evaluate_v21_confirmation_gate(
        rows_path=rows_path,
        thresholds_path=thresholds_path,
        output_path=tmp_path / "v2_receipt.json",
        selected_candidate="C2",
        control_id="C0",
        predecessor_id="C1",
    )

    assert receipt["gate_status"] == "PASS"
    assert receipt["thresholds_schema"] == (
        "pareto_v21_candidate_menu_precommit_v2"
    )

