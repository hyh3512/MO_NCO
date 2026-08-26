from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mo_nco.pareto_v21e3r1_v9_gate import (
    evaluate_v9_predevelopment_readiness,
    main,
    write_v9_predevelopment_readiness_receipt,
)


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def test_gate_materializes_exact_predevelopment_hold() -> None:
    receipt = evaluate_v9_predevelopment_readiness()

    assert receipt["status"] == "PRE_DEVELOPMENT_HOLD"
    assert receipt["development_matrix_execution_started"] is False
    assert receipt["development_rows_materialized"] == 0
    assert receipt["development_promotion_evaluated"] is False
    assert receipt["development_promotion_gate_passed"] is False
    assert receipt["simultaneous_inference_performed"] is False
    assert receipt["gates"]["single_case_engineering_smoke_authorized"] is True
    assert receipt["gates"]["full_development_matrix_authorized"] is False
    assert receipt["gates"]["all_required_artifacts_bound"] is False
    assert receipt["known_required_artifact_count"] == 10
    assert (
        receipt[
            "required_artifact_inventory_exhaustive_for_later_scientific_phases"
        ]
        is False
    )
    assert (
        receipt["additional_external_authority_and_submission_requirements_may_apply"]
        is True
    )
    assert all(
        receipt[field] is False
        for field in (
            "selection_authorized",
            "confirmation_authorized",
            "formal_authorized",
            "ijoc_submission_authorized",
        )
    )
    core = dict(receipt)
    digest = core.pop("receipt_payload_sha256")
    assert digest == hashlib.sha256(_canonical_bytes(core)).hexdigest()


def test_gate_writes_canonical_receipt_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "readiness.json"
    expected = write_v9_predevelopment_readiness_receipt(output)

    assert output.read_bytes() == _canonical_bytes(expected) + b"\n"
    with pytest.raises(FileExistsError):
        write_v9_predevelopment_readiness_receipt(output)


def test_gate_cli_returns_hold_exit_code_and_writes_receipt(tmp_path: Path) -> None:
    output = tmp_path / "cli.json"

    assert main(["--output", str(output)]) == 2
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == (
        "PRE_DEVELOPMENT_HOLD"
    )
