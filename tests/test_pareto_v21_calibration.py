from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mo_nco.pareto_ijoc_problem import MultiObjectiveKnapsackInstance
from mo_nco.pareto_v21_calibration import (
    load_v21_problem_packet,
    normalized_hypervolume_2d,
    run_v21_calibration_matrix,
    select_v21_calibration_candidate,
)


def _write_json(path: Path, payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _write_selection_evidence(
    root: Path,
    rows: list[dict[str, object]],
    *,
    candidates: tuple[str, ...],
    seeds: tuple[int, ...],
) -> tuple[Path, Path]:
    context_sha256 = "a" * 64
    for row in rows:
        row.setdefault("run_context_sha256", context_sha256)
    rows_path = root / "rows.jsonl"
    rows_bytes = b"".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )
    rows_path.write_bytes(rows_bytes)
    receipt_path = root / "matrix_receipt.json"
    _write_json(
        receipt_path,
        {
            "schema": "pareto_v21_calibration_matrix_receipt_v1",
            "status": "PASS",
            "split": "selection",
            "candidate_ids": list(candidates),
            "seeds": list(seeds),
            "expected_rows": len(rows),
            "completed_rows": len(rows),
            "all_trace_verifications_pass": True,
            "full_partition_binding_gate": "PASS",
            "run_context_binding_gate": "PASS",
            "run_context_sha256": context_sha256,
            "rows_sha256": hashlib.sha256(rows_bytes).hexdigest(),
        },
    )
    return rows_path, receipt_path


def test_v21_problem_packet_loader_supports_both_families(tmp_path) -> None:
    mokp_path = tmp_path / "mokp.json"
    _write_json(
        mokp_path,
        {
            "schema": "pareto_v21_mokp_integer_instance_v1",
            "case_id": "mokp-a",
            "item_weights": [2, 3, 4],
            "profits_by_objective": [[5, 4, 3], [2, 6, 5]],
            "capacity": 5,
        },
    )
    motsp_path = tmp_path / "motsp.json"
    _write_json(
        motsp_path,
        {
            "schema": "pareto_v21_motsp_integer_coordinates_v1",
            "case_id": "motsp-a",
            "coordinates_by_objective": [
                [[0, 0], [1, 0], [0, 1], [1, 1]],
                [[0, 0], [2, 0], [0, 2], [2, 2]],
            ],
        },
    )

    mokp = load_v21_problem_packet(mokp_path)
    motsp = load_v21_problem_packet(motsp_path)

    assert isinstance(mokp, MultiObjectiveKnapsackInstance)
    assert mokp.name == "mokp-a"
    assert motsp.name == "motsp-a"
    assert motsp.solution_size == 4


def test_normalized_hypervolume_uses_frozen_analytic_box() -> None:
    value = normalized_hypervolume_2d(
        ((-8.0, -2.0), (-4.0, -6.0)),
        lower=(-10.0, -10.0),
        upper=(0.0, 0.0),
    )

    assert value == pytest.approx(0.32)


def test_v21_calibration_matrix_is_exact_complete_and_exclusive(tmp_path) -> None:
    packet = {
        "schema": "pareto_v21_mokp_integer_instance_v1",
        "case_id": "mokp-cal-a",
        "item_weights": [2, 3, 4, 5, 6, 7],
        "profits_by_objective": [
            [5, 4, 3, 7, 8, 9],
            [2, 6, 5, 3, 9, 7],
        ],
        "capacity": 12,
    }
    packet_path = tmp_path / "instances" / "mokp-cal-a.json"
    packet_sha = _write_json(packet_path, packet)
    manifest_path = tmp_path / "case_manifest.json"
    manifest_sha = _write_json(
        manifest_path,
        {
            "schema": "pareto_v21_partition_manifest_v1",
            "split": "selection",
            "cases": [
                {
                    "case_id": "mokp-cal-a",
                    "family": "MOKP",
                    "split": "selection",
                    "size": 6,
                    "num_objectives": 2,
                    "artifact": {
                        "path": "instances/mokp-cal-a.json",
                        "sha256": packet_sha,
                        "bytes": packet_path.stat().st_size,
                    },
                }
            ],
        },
    )
    metric_path = tmp_path / "metric_manifest.json"
    metric_sha = _write_json(
        metric_path,
        {
            "schema": "pareto_v21_metric_manifest_v1",
            "metric_id": "normalized_left_continuous_hypervolume_auc_analytic_box_reference_1_1_v1",
        },
    )
    reference_path = tmp_path / "reference_manifest.json"
    reference_sha = _write_json(
        reference_path,
        {
            "schema": "pareto_v21_analytic_reference_manifest_v1",
            "split": "selection",
            "case_count": 1,
            "cases": [
                {
                    "case_id": "mokp-cal-a",
                    "family": "MOKP",
                    "packet_sha256": packet_sha,
                    "objective_lower_bounds": [-36.0, -32.0],
                    "objective_upper_bounds": [0.0, 0.0],
                    "normalized_reference_point": [1.0, 1.0],
                }
            ],
        },
    )
    source_path = tmp_path / "frozen_source.py"
    source_path.write_text("# frozen test source\n", encoding="utf-8")
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    precommit_path = tmp_path / "precommit.json"
    precommit_sha = _write_json(precommit_path, {"schema": "test_precommit_v1"})
    directions = ((0.2, 0.8), (0.5, 0.5), (0.8, 0.2))
    context_path = tmp_path / "run_context.json"
    _write_json(
        context_path,
        {
            "schema": "pareto_v21_calibration_run_context_v1",
            "status": "FROZEN_BEFORE_RUNS",
            "split": "selection",
            "partition_manifest": {
                "path": str(manifest_path),
                "sha256": manifest_sha,
            },
            "metric_manifest": {"path": str(metric_path), "sha256": metric_sha},
            "reference_manifest": {
                "path": str(reference_path),
                "sha256": reference_sha,
            },
            "precommit": {"path": str(precommit_path), "sha256": precommit_sha},
            "candidate_ids": ["C0", "C1"],
            "seeds": [11, 13],
            "evaluation_budget": 20,
            "checkpoint_period": 5,
            "reference_directions": [list(value) for value in directions],
            "source_bindings": [
                {"path": str(source_path), "sha256": source_sha}
            ],
        },
    )
    output = tmp_path / "runs"

    receipt = run_v21_calibration_matrix(
        manifest_path=manifest_path,
        run_context_path=context_path,
        output_directory=output,
        candidate_ids=("C0", "C1"),
        seeds=(11, 13),
        evaluation_budget=20,
        checkpoint_period=5,
        reference_directions=directions,
        require_full_partition_binding=False,
    )

    assert receipt["status"] == "PASS"
    assert receipt["expected_rows"] == 4
    assert receipt["completed_rows"] == 4
    assert receipt["run_context_binding_gate"] == "PASS"
    rows = [
        json.loads(line)
        for line in (output / "run_rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 4
    assert all(row["trace_verification_status"] == "PASS" for row in rows)
    assert all(0.0 <= row["normalized_hv_auc"] <= 1.0 for row in rows)
    assert all(0.0 <= row["normalized_final_hv"] <= 1.0 for row in rows)

    with pytest.raises(FileExistsError):
        run_v21_calibration_matrix(
            manifest_path=manifest_path,
            run_context_path=context_path,
            output_directory=output,
            candidate_ids=("C0", "C1"),
            seeds=(11, 13),
            evaluation_budget=20,
            checkpoint_period=5,
            reference_directions=directions,
            require_full_partition_binding=False,
        )


def test_v21_selection_gate_uses_case_clusters_and_both_families(tmp_path) -> None:
    rows = []
    for family in ("MOTSP", "MOKP"):
        for case_index in range(3):
            for seed in (101, 103):
                base = 0.30 + 0.01 * case_index
                rows.extend(
                    (
                        {
                            "schema": "pareto_v21_calibration_run_row_v1",
                            "split": "selection",
                            "family": family,
                            "case_id": f"{family}-{case_index}",
                            "candidate_id": "C0",
                            "seed": seed,
                            "normalized_hv_auc": base,
                            "trace_verification_status": "PASS",
                        },
                        {
                            "schema": "pareto_v21_calibration_run_row_v1",
                            "split": "selection",
                            "family": family,
                            "case_id": f"{family}-{case_index}",
                            "candidate_id": "C1",
                            "seed": seed,
                            "normalized_hv_auc": base + 0.01,
                            "trace_verification_status": "PASS",
                        },
                        {
                            "schema": "pareto_v21_calibration_run_row_v1",
                            "split": "selection",
                            "family": family,
                            "case_id": f"{family}-{case_index}",
                            "candidate_id": "C2",
                            "seed": seed,
                            "normalized_hv_auc": (
                                base + 0.02 if family == "MOKP" else base + 0.009
                            ),
                            "trace_verification_status": "PASS",
                        },
                    )
                )
    rows_path, matrix_receipt_path = _write_selection_evidence(
        tmp_path,
        rows,
        candidates=("C0", "C1", "C2"),
        seeds=(101, 103),
    )

    receipt = select_v21_calibration_candidate(
        rows_path=rows_path,
        matrix_receipt_path=matrix_receipt_path,
        output_path=tmp_path / "selection_receipt.json",
        candidate_ids=("C1", "C2"),
        control_id="C0",
        delta_min=0.005,
        noninferiority_margin=0.02,
        bootstrap_samples=500,
        randomization_seed=21103,
        expected_seeds=(101, 103),
    )

    assert receipt["status"] == "PASS"
    assert receipt["candidate_selected"] == "C1"
    assert receipt["confirmation_authorized"] is True
    assert receipt["candidate_results"]["C1"]["gate"] == "PASS"
    assert receipt["candidate_results"]["C2"]["gate"] == "FAIL"
    assert set(receipt["candidate_results"]["C2"]["family_gates"].values()) == {
        "PASS"
    }
    assert receipt["candidate_results"]["C2"]["adjacent_mechanism_gate"] == "FAIL"


def test_v21_selection_gate_stops_when_no_candidate_improves_both_families(
    tmp_path,
) -> None:
    rows = []
    for family, difference in (("MOTSP", 0.01), ("MOKP", -0.01)):
        for case_index in range(3):
            rows.extend(
                (
                    {
                        "schema": "pareto_v21_calibration_run_row_v1",
                        "split": "selection",
                        "family": family,
                        "case_id": f"{family}-{case_index}",
                        "candidate_id": "C0",
                        "seed": 107,
                        "normalized_hv_auc": 0.3,
                        "trace_verification_status": "PASS",
                    },
                    {
                        "schema": "pareto_v21_calibration_run_row_v1",
                        "split": "selection",
                        "family": family,
                        "case_id": f"{family}-{case_index}",
                        "candidate_id": "C1",
                        "seed": 107,
                        "normalized_hv_auc": 0.3 + difference,
                        "trace_verification_status": "PASS",
                    },
                )
            )
    rows_path, matrix_receipt_path = _write_selection_evidence(
        tmp_path,
        rows,
        candidates=("C0", "C1"),
        seeds=(107,),
    )

    receipt = select_v21_calibration_candidate(
        rows_path=rows_path,
        matrix_receipt_path=matrix_receipt_path,
        output_path=tmp_path / "stop_receipt.json",
        candidate_ids=("C1",),
        control_id="C0",
        delta_min=0.005,
        noninferiority_margin=0.02,
        bootstrap_samples=500,
        randomization_seed=21105,
        expected_seeds=(107,),
    )

    assert receipt["status"] == "STOP"
    assert receipt["candidate_selected"] is None
    assert receipt["confirmation_authorized"] is False


def test_v21_selection_rejects_unbound_or_incomplete_matrix(tmp_path) -> None:
    rows: list[dict[str, object]] = []
    for family in ("MOKP", "MOTSP"):
        for candidate, value in (("C0", 0.3), ("C1", 0.31)):
            rows.append(
                {
                    "schema": "pareto_v21_calibration_run_row_v1",
                    "split": "selection",
                    "family": family,
                    "case_id": f"{family}-0",
                    "candidate_id": candidate,
                    "seed": 109,
                    "normalized_hv_auc": value,
                    "trace_verification_status": "PASS",
                }
            )
    rows_path, matrix_receipt_path = _write_selection_evidence(
        tmp_path,
        rows,
        candidates=("C0", "C1"),
        seeds=(109,),
    )
    rows.append(dict(rows[-1]))
    rows_path.write_bytes(
        b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\n"
            for row in rows
        )
    )

    with pytest.raises(ValueError, match="receipt does not bind"):
        select_v21_calibration_candidate(
            rows_path=rows_path,
            matrix_receipt_path=matrix_receipt_path,
            output_path=tmp_path / "invalid.json",
            candidate_ids=("C1",),
            control_id="C0",
            delta_min=0.005,
            noninferiority_margin=0.02,
            mechanism_noninferiority_margin=0.001,
            bootstrap_samples=100,
            randomization_seed=21107,
            expected_cases_per_family=1,
            expected_seeds=(109,),
        )

