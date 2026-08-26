from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "audit_v21e3r1_development_matrix.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "v21e3r1_independent_audit_portability", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def test_same_implementation_receipt_is_portable_and_disclaims_independence(
    tmp_path: Path, monkeypatch,
) -> None:
    audit = _module()
    matrix = tmp_path / "copied" / "matrix"
    matrix.mkdir(parents=True)
    plan = {"schema": "portable-plan-fixture"}
    (matrix / "matrix.plan.json").write_text(
        json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    contract = SimpleNamespace(
        source_snapshot_root_sha256="1" * 64,
        authorization_sha256="2" * 64,
    )
    analysis = {"overall_gate": "PASS_DEVELOPMENT_NONINFERIORITY"}
    monkeypatch.setattr(audit, "load_frozen_contract", lambda **_: contract)
    monkeypatch.setattr(audit, "_matrix_plan", lambda _: plan)
    monkeypatch.setattr(
        audit,
        "verify_finalized_matrix_output",
        lambda *_, **__: {
            "analysis": analysis,
            "aggregate_sha256": "3" * 64,
            "runner_post_run_audit_sha256": "4" * 64,
            "row_count": 108,
        },
    )
    output = tmp_path / "same-implementation.receipt.json"

    receipt = audit.audit_development_matrix(
        matrix_directory=matrix,
        output=output,
        case_manifest_path=tmp_path / "case.json",
        reference_manifest_path=tmp_path / "reference.json",
        config_manifest_path=tmp_path / "config.json",
        metric_manifest_path=tmp_path / "metric.json",
        protocol_path=tmp_path / "protocol.json",
        authorization_path=tmp_path / "authorization.json",
        source_snapshot_root_sha256="1" * 64,
    )

    assert receipt["matrix_directory"] == "."
    assert (
        receipt["matrix_directory_path_semantics"]
        == "self_describing_matrix_root_v1"
    )
    assert str(matrix) not in output.read_text(encoding="utf-8")
    assert receipt["schema"] == (
        "pareto_v21e3r1_same_implementation_development_matrix_"
        "post_run_audit_v1"
    )
    assert receipt["status"] == (
        "PASS_SAME_IMPLEMENTATION_POST_PROCESS_RECOMPUTATION"
    )
    assert receipt["implementation_independence"] is False
    assert receipt["scientific_independence"] is False
    assert receipt["external_third_party_audit"] is False
    assert receipt["fixed_author_generated_cases_descriptive_only"] is True
    assert receipt["population_inference_authorized"] is False
    assert receipt["sign_flip_assumptions_verified"] is False
    assert receipt["trimmed_mean_distinct_from_mean"] is False
    assert receipt["verifier_relationship"] == (
        "SAME_PROJECT_VERIFIER_POST_HOC_SUCCESSOR_NOT_HISTORICAL_PRODUCER"
    )
    assert receipt["historical_matrix_producer"] == {
        "authorization_receipt_sha256": "2" * 64,
        "source_snapshot_root_sha256": "1" * 64,
    }
    owned = receipt["live_verifier_owned_files"]
    assert [entry["path"] for entry in owned] == sorted(
        {
            "ijoc_submission_v21e3r1/scripts/"
            "audit_v21e3r1_development_matrix.py",
            "ijoc_submission_v21e3r1/scripts/"
            "run_v21e3r1_development_parity.py",
            "mo_nco/pareto_v21e3_parity.py",
            "mo_nco/pareto_v21e3_trace_verify.py",
        }
    )
    assert all(
        isinstance(entry["bytes"], int)
        and not isinstance(entry["bytes"], bool)
        and entry["bytes"] > 0
        and len(entry["sha256"]) == 64
        for entry in owned
    )
    assert receipt["live_verifier_owned_files_root_sha256"] == hashlib.sha256(
        _canonical(owned)
    ).hexdigest()

