from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile

import pytest

from ijoc_submission_v21e3r1.scripts import (
    evaluate_v21e3r1_successor_development_factorial as evaluator,
)
from ijoc_submission_v21e3r1.scripts import (
    run_v21e3r1_successor_development_factorial as runner,
)


ROOT = Path(__file__).resolve().parents[1]
PARENT_PLAN = (
    ROOT
    / "outputs/v21e3r1_v7_exposed_development_diagnostics_20260823/diagnostic.plan.json"
)
EXPECTED_SEMANTIC_PARAMETERS = {
    "legacy_post_initialization_search_policy": "proposal_chain_v21e3r1_v1",
    "successor_post_initialization_search_policy": (
        "post_commit_type_incumbent_anchor_development_v1"
    ),
    "legacy_mokp_novelty_generation_policy": (
        "legacy_retry_and_local_v21e3r1_v1"
    ),
    "successor_mokp_novelty_generation_policy": (
        "single_attempt_rotating_feasible_exchange_no_refill_development_v1"
    ),
    "promoted_arm_by_family": {"MOKP": "MOKP_BOTH", "MOTSP": "MOTSP_ANCHOR"},
    "factorial_arm_ids_by_family": {
        "MOKP": [
            "MOKP_LEGACY",
            "MOKP_ANCHOR_ONLY",
            "MOKP_NOVELTY_ONLY",
            "MOKP_BOTH",
        ],
        "MOTSP": ["MOTSP_LEGACY", "MOTSP_ANCHOR"],
    },
}


def _source_binding() -> dict[str, object]:
    return {
        "schema": "v21e3r1_successor_factorial_source_binding_v2",
        "study_id": "v21e3r1-successor-factorial-test",
        "candidate_id": "C1",
        "parent_development_source_sha256": runner.PARENT_SOURCE_SHA256,
        "receipt_path": "outputs/frozen-successor/successor-source.freeze.receipt.json",
        "receipt_sha256": "0" * 64,
        "source_manifest_path": "outputs/frozen-successor/source.manifest.json",
        "source_manifest_sha256": "1" * 64,
        "source_snapshot_sha256": "2" * 64,
        "semantic_config_sha256": "3" * 64,
        "study_metric_spec_sha256": "4" * 64,
        "simultaneous_inference_spec_sha256": "5" * 64,
        "factorial_inference_spec_path": runner.INFERENCE_SPEC_RELATIVE.as_posix(),
        "factorial_inference_spec_sha256": runner.INFERENCE_SPEC_SHA256,
    }


def _write_source_freeze_fixture(
    project: Path, *, parent_source_sha256: str = runner.PARENT_SOURCE_SHA256
) -> Path:
    freeze = project / "outputs/frozen-successor"
    freeze.mkdir(parents=True)
    source_relatives = (
        runner.INFERENCE_SPEC_RELATIVE.as_posix(),
        "ijoc_submission_v21e3r1/development/V21E3R1_SUCCESSOR_DEVELOPMENT_FACTORIAL_RUNBOOK_V1.md",
        "ijoc_submission_v21e3r1/development/V21E3R1_V8_STRICT_EXECUTION_RUNBOOK_2026-08-23.md",
        "ijoc_submission_v21e3r1/scripts/freeze_v21e3r1_successor_source.py",
        "ijoc_submission_v21e3r1/scripts/run_v21e3r1_successor_development_factorial.py",
        "ijoc_submission_v21e3r1/scripts/evaluate_v21e3r1_successor_development_factorial.py",
        "independent_reproduction/recompute_v21e3r1_successor_metrics.py",
        "mo_nco/pareto_v21e3_hybrid.py",
        "mo_nco/pareto_v21e3r1_development_diagnostics.py",
        "pyproject.toml",
        "tests/test_v21e3r1_independent_simultaneous_inference.py",
        "tests/test_v21e3r1_successor_metric.py",
    )
    entries: list[dict[str, object]] = []
    for relative in sorted(source_relatives):
        source = ROOT / Path(*runner.PurePosixPath(relative).parts)
        target = project / Path(*runner.PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        entries.append(
            {
                "path": relative,
                "bytes": target.stat().st_size,
                "sha256": runner._sha256(target),
            }
        )
    source_root = runner._payload_sha256(entries)
    manifest = {
        "schema": "v21e3r1_branch_replay_source_manifest_binding_v1",
        "source_root_sha256": source_root,
        "entries": entries,
    }
    semantic = {
        "schema": "v21e3r1_successor_semantic_config_v1",
        "study_id": "v21e3r1-successor-factorial-test",
        "candidate_id": "C1",
        "parameters": EXPECTED_SEMANTIC_PARAMETERS,
    }
    metric = {"schema": "test-study-metric"}
    simultaneous = {"schema": "test-simultaneous-inference"}
    siblings = {
        "source.manifest.json": manifest,
        "semantic.config.json": semantic,
        "study.metric-spec.json": metric,
        "simultaneous-inference.spec.json": simultaneous,
    }
    sibling_sha: dict[str, str] = {}
    for name, value in siblings.items():
        path = freeze / name
        path.write_bytes(runner._canonical_bytes(value))
        sibling_sha[name] = runner._sha256(path)
    core = {
        "schema": runner.SOURCE_RECEIPT_SCHEMA,
        "status": runner.SOURCE_RECEIPT_STATUS,
        "study_id": semantic["study_id"],
        "candidate_id": semantic["candidate_id"],
        "parent_development_source_sha256": parent_source_sha256,
        "source_snapshot_sha256": source_root,
        "source_manifest_sha256": sibling_sha["source.manifest.json"],
        "semantic_config_sha256": sibling_sha["semantic.config.json"],
        "study_metric_spec_sha256": sibling_sha["study.metric-spec.json"],
        "simultaneous_inference_spec_sha256": sibling_sha[
            "simultaneous-inference.spec.json"
        ],
        "source_entry_count": len(entries),
        "source_total_bytes": sum(int(item["bytes"]) for item in entries),
        "all_source_files_verified": True,
        "source_frozen": True,
        "selection_cases_materialized": False,
        "confirmation_cases_materialized": False,
        "formal_cases_materialized": False,
        "source_archive_materialized": False,
        "source_archive_path": None,
        "source_archive_sha256": None,
        "source_archive_scope": "NOT_MATERIALIZED",
        "implementation_independence": False,
        "scientific_independence": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_study_authorized": False,
        "scientific_claim_authorized": False,
        "public_redistribution_authorized": False,
        "ijoc_submission_status": "IJOC_HOLD",
    }
    receipt = {**core, "receipt_payload_sha256": runner._payload_sha256(core)}
    receipt_path = freeze / "successor-source.freeze.receipt.json"
    receipt_path.write_bytes(runner._canonical_bytes(receipt))
    return receipt_path


def _valid_factorial_row(
    row_spec: dict[str, object],
    *,
    plan_sha256: str,
    source_binding: dict[str, object],
    algorithm_config: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": runner.ROW_SCHEMA,
        "status": "PASS_SUCCESSOR_DEVELOPMENT_FACTORIAL_ROW_ENGINEERING_ONLY",
        "phase": "development",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        **row_spec,
        "charged_evaluation_budget": runner.FULL_BUDGET,
        "checkpoint_period": runner.FULL_CHECKPOINT_PERIOD,
        "checkpoint_left_continuous_hv_auc": 0.45,
        "exact_per_evaluation_left_continuous_hv_auc": 0.4,
        "normalized_terminal_hv": 0.5,
        "checkpoints": [
            {
                "evaluation": evaluation,
                "normalized_hv": 0.5,
                "archive_size": 2,
            }
            for evaluation in range(
                runner.FULL_CHECKPOINT_PERIOD,
                runner.FULL_BUDGET + 1,
                runner.FULL_CHECKPOINT_PERIOD,
            )
        ],
        "attempt_count": 2100,
        "charged_evaluation_count": runner.FULL_BUDGET,
        "cache_hit_count": 100,
        "cache_hit_rate_per_attempt": 100 / 2100,
        "algorithm_config": algorithm_config,
        "plan_sha256": plan_sha256,
        "source_snapshot_sha256": source_binding["source_snapshot_sha256"],
        "trace_database_path": "trace.sqlite3",
        "trace_database_sha256": "6" * 64,
        "terminal_receipt_path": "terminal.receipt.json",
        "terminal_receipt_sha256": "7" * 64,
        "independent_metric_receipt_path": "independent.metric.json",
        "independent_metric_receipt_sha256": "8" * 64,
        "strict_trace_verification": {"status": "OBJECTIVE_AND_ARCHIVE_REPLAY_PASS"},
        "selection_cases_materialized": False,
        "confirmation_cases_materialized": False,
        "formal_cases_materialized": False,
        "implementation_independence": False,
        "scientific_independence": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_study_authorized": False,
        "scientific_claim_authorized": False,
        "ijoc_submission_status": "IJOC_HOLD",
    }


@pytest.fixture(scope="module")
def exact_plan() -> dict[str, object]:
    return runner.build_plan_payload(ROOT, PARENT_PLAN, _source_binding())


def test_packaged_inference_spec_is_canonical_and_prospectively_exact() -> None:
    spec, digest = runner.load_inference_spec(ROOT)
    path = ROOT / Path(*runner.INFERENCE_SPEC_RELATIVE.parts)

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.read_bytes() == runner._canonical_bytes(spec, newline=True)
    assert spec["method"] == runner.METHOD
    assert spec["familywise_alpha"] == 0.05
    assert spec["bootstrap_samples"] == 9999
    assert spec["bootstrap_seed"] == 2026082301
    assert spec["parent_v7_diagnostic_plan_sha256"] == runner.PARENT_PLAN_SHA256
    assert [item["threshold"] for item in spec["hypotheses"]] == [
        0.005,
        0.0,
        -0.005,
        0.1,
        0.0,
    ]
    assert spec["selection_cases_materialized"] is False


def test_plan_derives_exact_72_plus_36_rows_from_frozen_v7(
    exact_plan: dict[str, object],
) -> None:
    rows = exact_plan["rows"]
    mokp = [row for row in rows if row["family"] == "MOKP"]
    motsp = [row for row in rows if row["family"] == "MOTSP"]

    assert len(rows) == exact_plan["expected_rows"] == 108
    assert len(mokp) == 6 * 3 * 4 == 72
    assert len(motsp) == 6 * 3 * 2 == 36
    assert exact_plan["case_ids"] == list(runner.v7_runner.EXPECTED_CASE_IDS)
    assert exact_plan["seeds"] == [31051, 31057, 31059]
    assert exact_plan["input_binding"] == json.loads(
        PARENT_PLAN.read_text(encoding="utf-8")
    )["input_binding"]
    assert [row["ordinal"] for row in rows] == list(range(1, 109))
    assert len({row["row_id"] for row in rows}) == 108
    assert all(
        row["mokp_novelty_generation_policy"] == runner.LEGACY_NOVELTY
        for row in motsp
    )
    assert not any("selection" in str(row["case_id"]) for row in rows)
    assert not any("confirmation" in str(row["case_id"]) for row in rows)
    assert not any("formal" in str(row["case_id"]) for row in rows)


def test_tampered_parent_plan_hash_fails_before_design_derivation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    copied = project / "diagnostic.plan.json"
    copied.write_bytes(PARENT_PLAN.read_bytes() + b" ")

    with pytest.raises(runner.ContractError, match="SHA-256 drifted"):
        runner._validate_parent_plan(project, copied)


def test_parent_exact504_receipt_requires_exact_source_and_key_set() -> None:
    payload = {
        "schema": runner.PARENT_RECEIPT_SCHEMA,
        "status": "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "matrix_mode": "FULL_504",
        "completed_rows": 504,
        "expected_rows": 504,
        "plan_sha256": runner.PARENT_PLAN_SHA256,
        "source_snapshot_sha256": runner.PARENT_SOURCE_SHA256,
        "aggregate_sha256": "a" * 64,
        "selection_entropy_release": "PROHIBITED",
        "confirmation_materialization": "PROHIBITED",
        "formal_materialization": "PROHIBITED",
    }
    assert runner._validate_parent_receipt_payload(payload)["completed_rows"] == 504

    wrong_source = deepcopy(payload)
    wrong_source["source_snapshot_sha256"] = "b" * 64
    with pytest.raises(runner.ContractError, match="source_snapshot_sha256"):
        runner._validate_parent_receipt_payload(wrong_source)

    extra = deepcopy(payload)
    extra["selection_authorized"] = True
    with pytest.raises(runner.ContractError, match="key set drifted"):
        runner._validate_parent_receipt_payload(extra)


def test_parent_completed_seal_rejects_cross_row_attempt_substitution() -> None:
    row_a = "v21e3-mokp-development-n100-s00__seed-31051__arm-c0_standard"
    row_b = "v21e3-mokp-development-n100-s00__seed-31051__arm-c0_random"
    payload = {
        "status": "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "row_id": row_a,
        "attempt_directory": f"attempts/{row_b}/attempt-0001",
        "plan_sha256": runner.PARENT_PLAN_SHA256,
        "row_sha256": "1" * 64,
        "diagnostic_sha256": "2" * 64,
        "trace_sha256": "3" * 64,
        "terminal_receipt_sha256": "4" * 64,
        "independent_metric_receipt_sha256": "5" * 64,
    }

    with pytest.raises(runner.ContractError, match="exact row attempt directory"):
        runner._validate_parent_completed_identity(payload, row_a)


def test_successor_completed_seal_rejects_cross_row_attempt_substitution(
    exact_plan: dict[str, object],
) -> None:
    row_a = exact_plan["rows"][0]
    row_b = exact_plan["rows"][1]
    source = exact_plan["source_binding"]
    payload = {
        "schema": "v21e3r1_successor_development_factorial_completed_row_v2",
        "status": "PASS_SUCCESSOR_DEVELOPMENT_FACTORIAL_ROW_ENGINEERING_ONLY",
        "ordinal": row_a["ordinal"],
        "row_id": row_a["row_id"],
        "attempt_directory": f"attempts/{row_b['row_id']}/attempt-0001",
        "plan_sha256": "6" * 64,
        "row_spec_sha256": runner._payload_sha256(row_a),
        "worker_spec_sha256": "7" * 64,
        "row_sha256": "8" * 64,
        "trace_sha256": "9" * 64,
        "terminal_receipt_sha256": "a" * 64,
        "independent_metric_receipt_sha256": "b" * 64,
        "source_snapshot_sha256": source["source_snapshot_sha256"],
        "source_receipt_sha256": source["receipt_sha256"],
    }

    with pytest.raises(runner.ContractError, match="exact row attempt directory"):
        runner._validate_completed_identity(
            payload, row_a, "6" * 64, source
        )


def test_finalized_receipt_requires_exact_payload_authority_and_lineage(
    exact_plan: dict[str, object],
) -> None:
    core = runner._factorial_receipt_core(
        exact_plan, "6" * 64, "7" * 64
    )
    receipt = {**core, "receipt_payload_sha256": runner._payload_sha256(core)}
    assert runner._validate_factorial_receipt_payload(
        receipt, exact_plan, "6" * 64, "7" * 64
    )["study_id"] == exact_plan["source_binding"]["study_id"]

    authority = deepcopy(receipt)
    authority["selection_authorized"] = True
    authority_core = dict(authority)
    authority_core.pop("receipt_payload_sha256")
    authority["receipt_payload_sha256"] = runner._payload_sha256(authority_core)
    with pytest.raises(runner.ContractError, match="selection_authorized"):
        runner._validate_factorial_receipt_payload(
            authority, exact_plan, "6" * 64, "7" * 64
        )

    extra = deepcopy(receipt)
    extra["unbound"] = False
    with pytest.raises(runner.ContractError, match="key set drifted"):
        runner._validate_factorial_receipt_payload(
            extra, exact_plan, "6" * 64, "7" * 64
        )


def test_source_freeze_rejects_wrong_parent_development_source(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    receipt = _write_source_freeze_fixture(
        project, parent_source_sha256="f" * 64
    )

    with pytest.raises(runner.ContractError, match="parent development source"):
        runner.validate_successor_source_freeze(project, receipt)


def test_source_freeze_binding_v2_carries_full_successor_lineage(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    receipt = _write_source_freeze_fixture(project)

    binding = runner.validate_successor_source_freeze(project, receipt)

    assert binding["schema"] == "v21e3r1_successor_factorial_source_binding_v2"
    assert binding["study_id"] == "v21e3r1-successor-factorial-test"
    assert binding["candidate_id"] == "C1"
    assert binding["parent_development_source_sha256"] == runner.PARENT_SOURCE_SHA256
    assert binding["factorial_inference_spec_path"] == (
        runner.INFERENCE_SPEC_RELATIVE.as_posix()
    )
    assert binding["factorial_inference_spec_sha256"] == runner.INFERENCE_SPEC_SHA256


def test_source_freeze_accepts_canonical_materialized_archive_contract(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    receipt_path = _write_source_freeze_fixture(project)
    archive_path = receipt_path.parent / "successor-source.zip"
    archive_path.write_bytes(b"deterministic-successor-source-archive")
    receipt = json.loads(receipt_path.read_bytes())
    receipt["source_archive_materialized"] = True
    receipt["source_archive_path"] = "successor-source.zip"
    receipt["source_archive_sha256"] = runner._sha256(archive_path)
    receipt["source_archive_scope"] = (
        "SOURCE_INVENTORY_ONLY_INTERNAL_CUSTODY_NO_REDISTRIBUTION_AUTHORITY"
    )
    core = dict(receipt)
    core.pop("receipt_payload_sha256")
    receipt["receipt_payload_sha256"] = runner._payload_sha256(core)
    receipt_path.write_bytes(runner._canonical_bytes(receipt))

    binding = runner.validate_successor_source_freeze(project, receipt_path)

    assert binding["source_snapshot_sha256"] == receipt["source_snapshot_sha256"]


def test_source_freeze_rejects_resealed_unrelated_semantic_parameters(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    receipt_path = _write_source_freeze_fixture(project)
    semantic_path = receipt_path.parent / "semantic.config.json"
    semantic = json.loads(semantic_path.read_bytes())
    semantic["parameters"] = {"x": "y"}
    semantic_path.write_bytes(runner._canonical_bytes(semantic))
    receipt = json.loads(receipt_path.read_bytes())
    receipt["semantic_config_sha256"] = runner._sha256(semantic_path)
    core = dict(receipt)
    core.pop("receipt_payload_sha256")
    receipt["receipt_payload_sha256"] = runner._payload_sha256(core)
    receipt_path.write_bytes(runner._canonical_bytes(receipt))

    with pytest.raises(runner.ContractError, match="semantic config policy contract"):
        runner.validate_successor_source_freeze(project, receipt_path)


def test_source_freeze_consumer_rejects_unmanifested_executable_source_addition(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    receipt_path = _write_source_freeze_fixture(project)
    injected = project / "mo_nco/unmanifested_runtime_module.py"
    injected.write_bytes(b"VALUE = 'unfrozen'\n")

    with pytest.raises(runner.ContractError, match="source inventory closure drifted"):
        runner.validate_successor_source_freeze(project, receipt_path)


@pytest.mark.parametrize("mutation", ("float_entry_count", "unmaterialized_archive_binding"))
def test_source_freeze_revalidation_rejects_exact_type_and_archive_drift(
    tmp_path: Path, mutation: str
) -> None:
    project = tmp_path / mutation
    project.mkdir()
    receipt_path = _write_source_freeze_fixture(project)
    receipt = json.loads(receipt_path.read_bytes())
    if mutation == "float_entry_count":
        receipt["source_entry_count"] = float(receipt["source_entry_count"])
    else:
        receipt["source_archive_path"] = "source.snapshot.zip"
        receipt["source_archive_sha256"] = "a" * 64
    core = dict(receipt)
    core.pop("receipt_payload_sha256")
    receipt["receipt_payload_sha256"] = runner._payload_sha256(core)
    receipt_path.write_bytes(runner._canonical_bytes(receipt))

    with pytest.raises(runner.ContractError):
        runner.validate_successor_source_freeze(project, receipt_path)


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        (lambda plan: plan["rows"].pop(), "exactly 108 rows"),
        (lambda plan: plan.__setitem__("phase", "selection"), "field drifted: phase"),
        (
            lambda plan: plan["rows"][0].__setitem__(
                "post_initialization_search_policy", "unfrozen_policy"
            ),
            "case/seed/policy design drifted",
        ),
        (
            lambda plan: plan["rows"][72].__setitem__(
                "mokp_novelty_generation_policy", runner.NEW_NOVELTY
            ),
            "MOTSP rows cannot activate",
        ),
        (
            lambda plan: plan["case_ids"].__setitem__(0, "v21e3-mokp-selection-n100-s00"),
            "case/seed boundary drifted",
        ),
    ),
    ids=("row-count", "phase", "search-policy", "family-policy", "case-boundary"),
)
def test_plan_contract_rejects_cardinality_phase_policy_and_case_drift(
    exact_plan: dict[str, object], mutation: object, error: str
) -> None:
    tampered = deepcopy(exact_plan)
    mutation(tampered)

    with pytest.raises(runner.ContractError, match=error):
        runner.validate_plan_payload(tampered)


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        (
            lambda row: row.__setitem__("scientific_scope", "FORMAL"),
            "scientific_scope",
        ),
        (
            lambda row: row.__setitem__("selection_authorized", True),
            "selection_authorized",
        ),
        (
            lambda row: row["algorithm_config"].__setitem__("seed", 999),
            "algorithm_config",
        ),
        (
            lambda row: row.__setitem__("charged_evaluation_budget", 1999),
            "charged_evaluation_budget",
        ),
    ),
    ids=("scope", "authority", "config", "budget"),
)
def test_row_contract_is_exact_for_scope_authority_config_and_budget(
    exact_plan: dict[str, object], mutation: object, error: str
) -> None:
    row_spec = exact_plan["rows"][0]
    source = exact_plan["source_binding"]
    expected_config = {"phase": "development", "seed": row_spec["seed"]}
    row = _valid_factorial_row(
        row_spec,
        plan_sha256="6" * 64,
        source_binding=source,
        algorithm_config=deepcopy(expected_config),
    )
    mutation(row)

    with pytest.raises(runner.ContractError, match=error):
        runner._validate_row_payload(
            row, row_spec, "6" * 64, source, expected_config
        )


def test_row_config_binding_survives_canonical_json_roundtrip(
    exact_plan: dict[str, object],
) -> None:
    row_spec = exact_plan["rows"][0]
    source = exact_plan["source_binding"]
    expected_config = {
        "phase": "development",
        "reference_directions": ((1.0, 0.0), (0.0, 1.0)),
    }
    json_config = json.loads(runner._canonical_bytes(expected_config))
    row = _valid_factorial_row(
        row_spec,
        plan_sha256="6" * 64,
        source_binding=source,
        algorithm_config=json_config,
    )

    assert runner._validate_row_payload(
        row, row_spec, "6" * 64, source, expected_config
    )["algorithm_config"] == json_config


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        (
            lambda row: row["checkpoints"].__setitem__(0, 0.5),
            "checkpoint.*witness",
        ),
        (
            lambda row: row["checkpoints"][0].__setitem__("evaluation", 199),
            "checkpoint evaluation drifted",
        ),
        (
            lambda row: row["checkpoints"][0].__setitem__("normalized_hv", 1.1),
            "normalized_hv is outside",
        ),
        (
            lambda row: row["checkpoints"][0].__setitem__("archive_size", 0),
            "archive_size",
        ),
        (
            lambda row: row.__setitem__("checkpoint_left_continuous_hv_auc", 0.4),
            "checkpoint AUC witness drifted",
        ),
        (
            lambda row: row["checkpoints"][-1].__setitem__("normalized_hv", 0.4),
            "terminal checkpoint witness drifted",
        ),
    ),
    ids=("scalar", "grid", "hv-range", "archive-size", "auc", "terminal"),
)
def test_row_checkpoint_witness_contract_matches_production_shape(
    exact_plan: dict[str, object], mutation: object, error: str
) -> None:
    row_spec = exact_plan["rows"][0]
    source = exact_plan["source_binding"]
    expected_config = {"phase": "development", "seed": row_spec["seed"]}
    row = _valid_factorial_row(
        row_spec,
        plan_sha256="6" * 64,
        source_binding=source,
        algorithm_config=deepcopy(expected_config),
    )
    mutation(row)

    with pytest.raises(runner.ContractError, match=error):
        runner._validate_row_payload(
            row, row_spec, "6" * 64, source, expected_config
        )


def test_inference_threshold_drift_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    relative = Path("development/inference.json")
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    spec, _digest = runner.load_inference_spec(ROOT)
    tampered = deepcopy(spec)
    tampered["hypotheses"][0]["threshold"] = 0.004
    target.write_bytes(runner._canonical_bytes(tampered, newline=True))
    monkeypatch.setattr(
        runner, "INFERENCE_SPEC_RELATIVE", runner.PurePosixPath(relative.as_posix())
    )

    with pytest.raises(runner.ContractError, match="frozen hypothesis drifted"):
        runner.load_inference_spec(tmp_path)


def test_evaluate_rows_revalidates_the_frozen_inference_payload(
    exact_plan: dict[str, object],
) -> None:
    inference, _digest = runner.load_inference_spec(ROOT)
    tampered = deepcopy(inference)
    tampered["bootstrap_samples"] = 99

    with pytest.raises(runner.ContractError, match="frozen inference"):
        evaluator.evaluate_rows(exact_plan, _synthetic_rows(exact_plan), tampered)


def test_attempt_directories_fail_closed_while_prior_attempt_is_unterminated(
    tmp_path: Path,
) -> None:
    first = runner._next_attempt(tmp_path, "row-a")

    assert first.name == "attempt-0001"
    with pytest.raises(runner.ContractError, match="active or unterminated"):
        runner._next_attempt(tmp_path, "row-a")

    (first / "failure.receipt.json").write_bytes(
        runner._canonical_bytes(
            {
                "schema": "v21e3r1_successor_factorial_row_failure_v1",
                "status": "FAIL_ROW_PROCESS",
                "returncode": 3,
                "stdout_tail": "",
                "stderr_tail": "fixture",
                "selection_authorized": False,
                "formal_study_authorized": False,
            },
            newline=True,
        )
    )
    second = runner._next_attempt(tmp_path, "row-a")
    assert second.name == "attempt-0002"
    assert first.is_dir() and second.is_dir()


def test_forged_failure_receipt_cannot_unlock_a_concurrent_retry(tmp_path: Path) -> None:
    first = runner._next_attempt(tmp_path, "row-a")
    (first / "failure.receipt.json").write_bytes(b"{}\n")

    with pytest.raises(runner.ContractError, match="failure receipt"):
        runner._next_attempt(tmp_path, "row-a")


def test_runner_rejects_output_directory_outside_project_before_reading_inputs(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"

    with pytest.raises(runner.ContractError, match="output directory escapes"):
        runner.run_matrix(
            project,
            outside,
            v7_diagnostic_plan=project / "missing.plan.json",
            source_freeze_receipt=project / "missing.freeze.json",
            plan_only=True,
        )


def test_plan_only_resume_cannot_bypass_finalized_receipt_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    output = project / "outputs" / "factorial"
    output.mkdir(parents=True)
    plan = {"frozen": True}
    (output / "factorial.plan.json").write_bytes(
        runner._canonical_bytes(plan, newline=True)
    )
    (output / "factorial.receipt.json").write_text("sealed", encoding="utf-8")
    monkeypatch.setattr(
        runner, "validate_sealed_parent_diagnostic", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        runner, "validate_successor_source_freeze", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(runner, "build_plan_payload", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(runner, "_preflight_policy_fields", lambda: None)

    def reject_finalized(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise runner.ContractError("sentinel finalized resume verification")

    monkeypatch.setattr(runner, "_verify_completed_receipt", reject_finalized)

    with pytest.raises(runner.ContractError, match="sentinel finalized resume"):
        runner.run_matrix(
            project,
            output,
            v7_diagnostic_plan=project / "parent.plan.json",
            source_freeze_receipt=project / "source.freeze.json",
            resume=True,
            plan_only=True,
        )


def _factorial_summary_fixture(row: dict[str, object]) -> dict[str, object]:
    row_id = str(row["row_id"])
    return {
        "ordinal": row["ordinal"],
        "row_id": row_id,
        "case_id": row["case_id"],
        "family": row["family"],
        "seed": row["seed"],
        "arm_id": row["arm_id"],
        "exact_per_evaluation_left_continuous_hv_auc": 0.4,
        "cache_hit_rate_per_attempt": 0.1,
        "row_sha256": hashlib.sha256(f"row:{row_id}".encode()).hexdigest(),
        "trace_sha256": hashlib.sha256(f"trace:{row_id}".encode()).hexdigest(),
        "terminal_receipt_sha256": hashlib.sha256(
            f"terminal:{row_id}".encode()
        ).hexdigest(),
        "independent_metric_receipt_sha256": hashlib.sha256(
            f"metric:{row_id}".encode()
        ).hexdigest(),
    }


def _prepare_aggregate_only_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exact_plan: dict[str, object],
    *,
    drift_aggregate: bool,
) -> tuple[Path, Path, dict[str, object]]:
    project = tmp_path / "project"
    output = project / "outputs" / "factorial"
    completed_dir = output / "completed"
    completed_dir.mkdir(parents=True)
    plan = deepcopy(exact_plan)
    plan_path = output / "factorial.plan.json"
    plan_path.write_bytes(runner._canonical_bytes(plan, newline=True))
    plan_sha = runner._sha256(plan_path)
    for row in plan["rows"]:
        (completed_dir / f"{row['row_id']}.json").write_bytes(b"{}\n")
    (output / "row.json").write_bytes(b"{}\n")
    summaries = [_factorial_summary_fixture(row) for row in plan["rows"]]
    aggregate = runner._factorial_aggregate_payload(plan_sha, summaries)
    if drift_aggregate:
        aggregate["rows"][0]["cache_hit_rate_per_attempt"] = 0.2
    (output / "factorial.aggregate.json").write_bytes(
        runner._canonical_bytes(aggregate, newline=True)
    )

    source = deepcopy(plan["source_binding"])
    monkeypatch.setattr(
        runner, "validate_sealed_parent_diagnostic", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        runner,
        "validate_successor_source_freeze",
        lambda *_args, **_kwargs: deepcopy(source),
    )
    monkeypatch.setattr(
        runner, "build_plan_payload", lambda *_args, **_kwargs: deepcopy(plan)
    )
    monkeypatch.setattr(runner, "_preflight_policy_fields", lambda: None)
    monkeypatch.setattr(
        runner.v7_runner,
        "_load_inputs",
        lambda *_args, **_kwargs: ([], {}, {}, deepcopy(plan["input_binding"])),
    )
    monkeypatch.setattr(runner, "_expected_semantic_config", lambda *_args: {})
    monkeypatch.setattr(
        runner,
        "_completed_payload",
        lambda _output, row, *_args: {
            "attempt_directory": "attempts/fixture/attempt-0001",
            "fixture_row": row,
        },
    )
    monkeypatch.setattr(runner, "_contained", lambda *_args, **_kwargs: output)
    monkeypatch.setattr(
        runner,
        "_factorial_summary",
        lambda _row, completed: _factorial_summary_fixture(completed["fixture_row"]),
    )

    def verify_written_receipt(
        _root: Path,
        _output: Path,
        verified_plan: dict[str, object],
        verified_plan_sha: str,
    ) -> dict[str, object]:
        receipt_path = output / "factorial.receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        return runner._validate_factorial_receipt_payload(
            receipt,
            verified_plan,
            verified_plan_sha,
            runner._sha256(output / "factorial.aggregate.json"),
        )

    monkeypatch.setattr(runner, "_verify_completed_receipt", verify_written_receipt)
    return project, output, plan


def test_resume_recovers_exact_aggregate_only_finalization_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exact_plan: dict[str, object],
) -> None:
    project, output, _plan = _prepare_aggregate_only_resume(
        tmp_path, monkeypatch, exact_plan, drift_aggregate=False
    )

    receipt = runner.run_matrix(
        project,
        output,
        v7_diagnostic_plan=project / "parent.plan.json",
        source_freeze_receipt=project / "source.freeze.json",
        resume=True,
    )

    assert receipt["status"] == (
        "PASS_EXACT_108_SUCCESSOR_DEVELOPMENT_FACTORIAL_ENGINEERING_ONLY"
    )
    assert (output / "factorial.receipt.json").is_file()


def test_resume_rejects_drifted_aggregate_only_finalization_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exact_plan: dict[str, object],
) -> None:
    project, output, _plan = _prepare_aggregate_only_resume(
        tmp_path, monkeypatch, exact_plan, drift_aggregate=True
    )

    with pytest.raises(runner.ContractError, match="aggregate.*drifted"):
        runner.run_matrix(
            project,
            output,
            v7_diagnostic_plan=project / "parent.plan.json",
            source_freeze_receipt=project / "source.freeze.json",
            resume=True,
        )

    assert not (output / "factorial.receipt.json").exists()


def test_finalized_resume_rejects_receipt_without_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exact_plan: dict[str, object],
) -> None:
    project = tmp_path / "project"
    output = project / "outputs" / "factorial"
    output.mkdir(parents=True)
    plan = deepcopy(exact_plan)
    plan_path = output / "factorial.plan.json"
    plan_path.write_bytes(runner._canonical_bytes(plan, newline=True))
    (output / "factorial.receipt.json").write_bytes(b"{}\n")
    parent = project / str(plan["parent_v7_diagnostic_plan_path"])
    parent.parent.mkdir(parents=True, exist_ok=True)
    parent.write_bytes(b"{}\n")
    source_path = project / str(plan["source_binding"]["receipt_path"])
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"{}\n")
    source = deepcopy(plan["source_binding"])
    monkeypatch.setattr(
        runner, "validate_sealed_parent_diagnostic", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        runner,
        "validate_successor_source_freeze",
        lambda *_args, **_kwargs: deepcopy(source),
    )
    monkeypatch.setattr(
        runner, "build_plan_payload", lambda *_args, **_kwargs: deepcopy(plan)
    )
    monkeypatch.setattr(
        runner,
        "load_inference_spec",
        lambda *_args, **_kwargs: ({}, plan["inference_spec_binding"]["sha256"]),
    )

    with pytest.raises(runner.ContractError, match="final artifacts are incomplete"):
        runner._verify_completed_receipt(
            project, output, plan, runner._sha256(plan_path)
        )


def test_worker_spec_rejects_injected_execution_inputs_before_plan_access(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "worker.spec.json"
    spec_path.write_bytes(
        runner._canonical_bytes(
            {
                "schema": "v21e3r1_successor_factorial_worker_spec_v2",
                "row_id": "v21e3-mokp-development-n100-s00__seed-31051__arm-mokp_legacy",
                "ordinal": 1,
                "plan_sha256": "0" * 64,
                "objective_bounds": {"lower": [0.0, 0.0], "upper": [1.0, 1.0]},
                "reference_directions": [[0.5, 0.5]],
            },
            newline=True,
        )
    )

    with pytest.raises(runner.ContractError, match="worker spec key set drifted"):
        runner._worker_run(spec_path)


def test_worker_rederives_row_bounds_and_directions_from_the_frozen_plan(
    exact_plan: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(
        runner,
        "validate_successor_source_freeze",
        lambda _root, _path: deepcopy(exact_plan["source_binding"]),
    )
    with tempfile.TemporaryDirectory(prefix=".factorial-worker-test-", dir=ROOT) as raw:
        output = Path(raw) / "matrix"
        row_spec = exact_plan["rows"][0]
        attempt = output / "attempts" / row_spec["row_id"] / "attempt-0001"
        attempt.mkdir(parents=True)
        plan_path = output / "factorial.plan.json"
        plan_path.write_bytes(runner._canonical_bytes(exact_plan, newline=True))
        spec_path = attempt / "worker.spec.json"
        spec_path.write_bytes(
            runner._canonical_bytes(
                {
                    "schema": "v21e3r1_successor_factorial_worker_spec_v2",
                    "row_id": row_spec["row_id"],
                    "ordinal": row_spec["ordinal"],
                    "plan_sha256": runner._sha256(plan_path),
                },
                newline=True,
            )
        )

        contract = runner._derive_worker_contract(spec_path)
        _cases, bounds, directions, _binding = runner.v7_runner._load_inputs(ROOT)

        assert contract["row_spec"] == row_spec
        assert contract["lower"] == tuple(bounds[row_spec["case_id"]][0])
        assert contract["upper"] == tuple(bounds[row_spec["case_id"]][1])
        assert contract["directions"] == directions
        assert contract["project"] == ROOT


def _synthetic_rows(
    plan: dict[str, object], *, variable_effects: bool = True
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    case_index = {
        case_id: index % 6 for index, case_id in enumerate(plan["case_ids"])
    }
    for row_spec in plan["rows"]:
        index = case_index[row_spec["case_id"]] if variable_effects else 0
        arm = row_spec["arm_id"]
        if row_spec["family"] == "MOKP":
            eauc = {
                "MOKP_LEGACY": 0.4000,
                "MOKP_ANCHOR_ONLY": 0.4200 + 0.0005 * index,
                "MOKP_NOVELTY_ONLY": 0.4020 - 0.0002 * index,
                "MOKP_BOTH": 0.4350 + 0.0003 * index,
            }[arm]
            cache = {
                "MOKP_LEGACY": 0.450,
                "MOKP_ANCHOR_ONLY": 0.430 - 0.001 * index,
                "MOKP_NOVELTY_ONLY": 0.100 + 0.002 * index,
                "MOKP_BOTH": 0.090 + 0.001 * index,
            }[arm]
        else:
            eauc = (
                0.4000
                if arm == "MOTSP_LEGACY"
                else 0.4200 + 0.0004 * index
            )
            cache = 0.05
        rows.append(
            {
                "case_id": row_spec["case_id"],
                "family": row_spec["family"],
                "seed": row_spec["seed"],
                "arm_id": arm,
                "exact_per_evaluation_left_continuous_hv_auc": eauc,
                "cache_hit_rate_per_attempt": cache,
            }
        )
    return rows


def _statistical_golden_rows(plan: dict[str, object]) -> list[dict[str, object]]:
    case_index = {
        case_id: index % 6 for index, case_id in enumerate(plan["case_ids"])
    }
    seed_index = {seed: index for index, seed in enumerate(runner.SEEDS)}
    anchor_shift = (-3, 1, 2)
    novelty_shift = (2, -3, 1)
    cache_shift = (-4, 1, 3)
    motsp_shift = (-2, 3, -1)
    rows: list[dict[str, object]] = []
    for row_spec in plan["rows"]:
        case = case_index[row_spec["case_id"]]
        seed = seed_index[row_spec["seed"]]
        arm = row_spec["arm_id"]
        if row_spec["family"] == "MOKP":
            anchor = 4 + case
            novelty = 2 + case
            cache_reduction = 128 + 16 * case
            eauc_numerator = {
                "MOKP_LEGACY": 256,
                "MOKP_ANCHOR_ONLY": 256 + anchor + anchor_shift[seed],
                "MOKP_NOVELTY_ONLY": 256 + novelty + novelty_shift[seed],
                "MOKP_BOTH": (
                    256
                    + anchor
                    + anchor_shift[seed]
                    + novelty
                    + novelty_shift[seed]
                ),
            }[arm]
            cache_numerator = {
                "MOKP_LEGACY": 512,
                "MOKP_ANCHOR_ONLY": 496,
                "MOKP_NOVELTY_ONLY": (
                    512 - cache_reduction + cache_shift[seed]
                ),
                "MOKP_BOTH": 496 - cache_reduction + cache_shift[seed],
            }[arm]
        else:
            motsp = 8 + 2 * case
            eauc_numerator = (
                320
                if arm == "MOTSP_LEGACY"
                else 320 + motsp + motsp_shift[seed]
            )
            cache_numerator = 64
        rows.append(
            {
                "case_id": row_spec["case_id"],
                "family": row_spec["family"],
                "seed": row_spec["seed"],
                "arm_id": arm,
                "exact_per_evaluation_left_continuous_hv_auc": (
                    eauc_numerator / 1024.0
                ),
                "cache_hit_rate_per_attempt": cache_numerator / 1024.0,
            }
        )
    return rows


def test_joint_five_hypothesis_evaluator_can_pass_only_development_promotion(
    exact_plan: dict[str, object],
) -> None:
    inference, _digest = runner.load_inference_spec(ROOT)
    result = evaluator.evaluate_rows(
        exact_plan, _synthetic_rows(exact_plan), inference
    )

    assert result["status"] == "PASS_SUCCESSOR_DEVELOPMENT_PROMOTION_ONLY"
    assert result["development_promotion_gate_passed"] is True
    assert result["gate_reasons"] == []
    assert len(result["cells"]) == 5
    assert all(cell["gate_passed"] is True for cell in result["cells"])
    assert result["critical_value"] is not None
    assert len(result["bootstrap_maxima_sha256"]) == 64


def test_evaluation_receipt_v2_binds_cross_stage_identity_for_common_gate(
    exact_plan: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    with tempfile.TemporaryDirectory(prefix=".factorial-evaluation-test-", dir=ROOT) as raw:
        directory = Path(raw)
        matrix = directory / "matrix"
        matrix.mkdir()
        output = directory / "development-promotion.evaluation.json"
        monkeypatch.setattr(
            evaluator,
            "load_matrix_rows",
            lambda _root, _matrix: (
                exact_plan,
                _synthetic_rows(exact_plan),
                "6" * 64,
                "7" * 64,
                "8" * 64,
            ),
        )

        receipt, exit_code = evaluator.evaluate_matrix(ROOT, matrix, output)

        assert exit_code == 0
        assert receipt["schema"] == evaluator.EVALUATION_SCHEMA
        assert receipt["status"] == "PASS_SUCCESSOR_DEVELOPMENT_PROMOTION_ONLY"
        assert receipt["promotion_scope"] == (
            "SUCCESSOR_DEVELOPMENT_PROMOTION_ONLY_HASH_BOUND_PRODUCER_RECEIPT_"
            "NO_PROSPECTIVE_108_ROW_RECOMPUTATION_NO_SCIENTIFIC_CLAIM"
        )
        assert receipt["study_id"] == exact_plan["source_binding"]["study_id"]
        assert receipt["candidate_id"] == exact_plan["source_binding"]["candidate_id"]
        assert receipt["successor_source_sha256"] == exact_plan["source_binding"][
            "source_snapshot_sha256"
        ]
        assert receipt["successor_config_sha256"] == exact_plan["source_binding"][
            "semantic_config_sha256"
        ]
        assert receipt["source_freeze_receipt_sha256"] == exact_plan[
            "source_binding"
        ]["receipt_sha256"]
        assert receipt["matrix_receipt_sha256"] == "7" * 64
        assert receipt["inference_spec_sha256"] == runner.INFERENCE_SPEC_SHA256
        assert output.read_bytes() == runner._canonical_bytes(receipt, newline=True)
        assert evaluator.validate_evaluation_receipt_payload(receipt) == receipt


def test_evaluation_receipt_rejects_resealed_statistical_and_protocol_tampering(
    exact_plan: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    with tempfile.TemporaryDirectory(prefix=".factorial-evaluation-tamper-", dir=ROOT) as raw:
        directory = Path(raw)
        matrix = directory / "matrix"
        matrix.mkdir()
        monkeypatch.setattr(
            evaluator,
            "load_matrix_rows",
            lambda _root, _matrix: (
                exact_plan,
                _synthetic_rows(exact_plan),
                "6" * 64,
                "7" * 64,
                "8" * 64,
            ),
        )
        receipt, _exit_code = evaluator.evaluate_matrix(
            ROOT, matrix, directory / "development-promotion.evaluation.json"
        )

        mutations = []
        threshold = deepcopy(receipt)
        threshold["cells"][0]["threshold"] = 0.0
        mutations.append(threshold)
        protocol = deepcopy(receipt)
        protocol["rng_domain"] = "attacker-selected-domain"
        mutations.append(protocol)
        inconsistent_gate = deepcopy(receipt)
        inconsistent_gate["cells"][0]["gate_passed"] = False
        mutations.append(inconsistent_gate)
        invalid_standard_error = deepcopy(receipt)
        invalid_standard_error["cells"][0]["standard_error"] = -0.001
        mutations.append(invalid_standard_error)

        for tampered in mutations:
            core = dict(tampered)
            core.pop("receipt_payload_sha256")
            tampered["receipt_payload_sha256"] = runner._payload_sha256(core)
            with pytest.raises(runner.ContractError):
                evaluator.validate_evaluation_receipt_payload(tampered)


def test_canonical_statistical_hold_receipts_remain_valid_common_evidence(
    exact_plan: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    threshold_rows = _statistical_golden_rows(exact_plan)
    for row in threshold_rows:
        if row["family"] == "MOKP" and row["arm_id"] == "MOKP_BOTH":
            row["exact_per_evaluation_left_continuous_hv_auc"] -= 0.05
    scenarios = (
        (
            threshold_rows,
            "HOLD_SUCCESSOR_DEVELOPMENT_PROMOTION_THRESHOLDS_NOT_MET",
        ),
        (
            _synthetic_rows(exact_plan, variable_effects=False),
            "HOLD_SUCCESSOR_DEVELOPMENT_PROMOTION_ZERO_STANDARD_ERROR",
        ),
    )
    with tempfile.TemporaryDirectory(prefix=".factorial-valid-holds-", dir=ROOT) as raw:
        directory = Path(raw)
        matrix = directory / "matrix"
        matrix.mkdir()
        for index, (rows, expected_status) in enumerate(scenarios, start=1):
            monkeypatch.setattr(
                evaluator,
                "load_matrix_rows",
                lambda _root, _matrix, bound_rows=rows: (
                    exact_plan,
                    bound_rows,
                    "6" * 64,
                    "7" * 64,
                    "8" * 64,
                ),
            )
            receipt, exit_code = evaluator.evaluate_matrix(
                ROOT, matrix, directory / f"development-promotion-{index}.json"
            )
            assert exit_code == 2
            assert receipt["status"] == expected_status
            assert receipt["development_promotion_gate_passed"] is False
            assert evaluator.validate_evaluation_receipt_payload(receipt) == receipt


def test_matrix_evaluator_rechecks_exact504_before_reading_successor_rows(
    exact_plan: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    with tempfile.TemporaryDirectory(prefix=".factorial-parent-check-", dir=ROOT) as raw:
        matrix = Path(raw)
        (matrix / "factorial.plan.json").write_bytes(
            runner._canonical_bytes(exact_plan, newline=True)
        )

        def hold_parent(_root: Path, _path: Path) -> dict[str, object]:
            raise runner.ContractError("sentinel exact504 seal rejection")

        monkeypatch.setattr(runner, "validate_sealed_parent_diagnostic", hold_parent)

        with pytest.raises(runner.ContractError, match="sentinel exact504"):
            evaluator.load_matrix_rows(ROOT, matrix)


def test_matrix_evaluator_rechecks_live_source_freeze_before_rows(
    exact_plan: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    with tempfile.TemporaryDirectory(prefix=".factorial-source-check-", dir=ROOT) as raw:
        matrix = Path(raw)
        (matrix / "factorial.plan.json").write_bytes(
            runner._canonical_bytes(exact_plan, newline=True)
        )
        monkeypatch.setattr(
            runner,
            "validate_sealed_parent_diagnostic",
            lambda _root, _path: {},
        )

        def hold_source(_root: Path, _path: Path) -> dict[str, object]:
            raise runner.ContractError("sentinel source freeze rejection")

        monkeypatch.setattr(runner, "validate_successor_source_freeze", hold_source)

        with pytest.raises(runner.ContractError, match="sentinel source freeze"):
            evaluator.load_matrix_rows(ROOT, matrix)


def test_integrity_failure_materializes_an_exclusive_hold_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(prefix=".factorial-integrity-hold-", dir=ROOT) as raw:
        directory = Path(raw)
        output = directory / "development-promotion.evaluation.json"
        monkeypatch.setattr(
            evaluator,
            "evaluate_matrix",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                runner.ContractError("sentinel integrity failure")
            ),
        )

        exit_code = evaluator.main(
            [
                "--project-root",
                str(ROOT),
                "--matrix-directory",
                str(directory / "matrix"),
                "--output",
                str(output),
            ]
        )

        assert exit_code == 3
        hold = json.loads(output.read_bytes())
        assert hold["schema"] == (
            "v21e3r1_successor_development_factorial_evaluation_integrity_hold_v1"
        )
        assert hold["status"] == "HOLD_INTEGRITY_ERROR"
        assert hold["development_promotion_gate_passed"] is False
        assert hold["selection_authorized"] is False
        core = dict(hold)
        payload_sha = core.pop("receipt_payload_sha256")
        assert payload_sha == runner._payload_sha256(core)


def test_integrity_hold_materialization_failure_still_exits_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "evaluate_matrix",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runner.ContractError("sentinel integrity failure")
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "_materialize_integrity_hold_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("sentinel receipt materialization failure")
        ),
    )

    assert evaluator.main(
        [
            "--project-root",
            str(ROOT),
            "--matrix-directory",
            str(ROOT / "not-used-matrix"),
            "--output",
            str(ROOT / "not-used-evaluation.json"),
        ]
    ) == 3


def test_statistical_golden_locks_seed_aggregation_factorial_and_joint_max_t(
    exact_plan: dict[str, object],
) -> None:
    inference, _digest = runner.load_inference_spec(ROOT)
    result = evaluator.evaluate_rows(
        exact_plan, _statistical_golden_rows(exact_plan), inference
    )

    assert [cell["observed_mean"] for cell in result["cells"]] == pytest.approx(
        [0.0107421875, 0.00634765625, 0.00439453125, 0.1640625, 0.0126953125],
        rel=0.0,
        abs=1e-15,
    )
    assert [cell["standard_error"] for cell in result["cells"]] == pytest.approx(
        [
            0.0014917238590351043,
            0.0007458619295175521,
            0.0007458619295175521,
            0.011933790872280834,
            0.0014917238590351043,
        ],
        rel=0.0,
        abs=1e-15,
    )
    assert result["critical_value"] == pytest.approx(
        1.7457431218879398, rel=0.0, abs=1e-15
    )
    assert result["bootstrap_maxima_sha256"] == (
        "f706945cbb26ac7dd1743faccf17ad6c2d141348c43170a70d7e16818ac3afb3"
    )


def test_evaluator_rejects_missing_row_coverage(
    exact_plan: dict[str, object],
) -> None:
    inference, _digest = runner.load_inference_spec(ROOT)
    rows = _synthetic_rows(exact_plan)
    rows.pop()

    with pytest.raises(runner.ContractError, match="missing factorial row"):
        evaluator.evaluate_rows(exact_plan, rows, inference)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("cache_hit_rate_per_attempt", 0.9, "cache metric disagrees"),
        ("exact_per_evaluation_left_continuous_hv_auc", 0.7, "EAUC disagrees"),
    ),
)
def test_replayed_metrics_override_untrusted_row_claims(
    field: str, value: float, error: str
) -> None:
    row = {
        "attempt_count": 10,
        "cache_hit_count": 2,
        "charged_evaluation_count": runner.FULL_BUDGET,
        "cache_hit_rate_per_attempt": 0.2,
        "exact_per_evaluation_left_continuous_hv_auc": 0.4,
        "normalized_terminal_hv": 0.5,
    }
    row[field] = value
    terminal = {
        "attempt_count": 10,
        "cache_hit_count": 2,
        "charged_evaluation_count": runner.FULL_BUDGET,
    }
    independent = {
        "exact_left_continuous_hv_auc": 0.4,
        "terminal_hv": 0.5,
    }

    with pytest.raises(runner.ContractError, match=error):
        evaluator._derive_replayed_row_metrics(
            row, terminal, independent, row_id="synthetic-row"
        )


def test_row_evidence_is_freshly_trace_verified_and_metric_replayed(
    exact_plan: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    row_spec = exact_plan["rows"][0]
    source = exact_plan["source_binding"]
    expected_config = {"phase": "development", "seed": row_spec["seed"]}
    verification = {"status": "OBJECTIVE_AND_ARCHIVE_REPLAY_PASS"}
    row = _valid_factorial_row(
        row_spec,
        plan_sha256="6" * 64,
        source_binding=source,
        algorithm_config=expected_config,
    )
    row["strict_trace_verification"] = verification
    independent = {
        "schema": "v21e3r1_successor_independent_metric_reimplementation_v3",
        "status": "PASS_SUCCESSOR_INDEPENDENT_METRIC_IMPLEMENTATION",
        "trace": "bound-by-test",
        "evaluation_count": runner.FULL_BUDGET,
        "attempt_count": 2100,
        "decision_count": runner.FULL_BUDGET,
        "exact_left_continuous_hv_auc": 0.4,
        "terminal_hv": 0.5,
        "trace_sha256": "9" * 64,
        "run_context_digest_sha256": "a" * 64,
        "reimplementation_source_sha256": "b" * 64,
        "terminal_accounting_gate": "PASS",
        "implementation_independence_from_project_metrics": True,
        "algorithm_execution_independence": False,
        "scientific_independence": False,
    }
    terminal_core = {
        "attempt_count": 2100,
        "cache_hit_count": 100,
        "charged_evaluation_count": runner.FULL_BUDGET,
        "database_path": "trace.sqlite3",
        "decision_count": runner.FULL_BUDGET,
        "durability_mode": "SQLITE_WAL_SYNCHRONOUS_FULL",
        "failure_code": None,
        "failure_detail": None,
        "family": row_spec["family"],
        "finalization_gates": {
            "cache_hits": 100,
            "evaluation_index_bounds": [1, runner.FULL_BUDGET],
            "expected_charged_evaluations": runner.FULL_BUDGET,
            "expected_decisions": runner.FULL_BUDGET,
            "expected_evaluation_index_bounds": [1, runner.FULL_BUDGET],
            "nonterminal_attempts": 0,
            "persisted_attempts": 2100,
            "persisted_decisions": runner.FULL_BUDGET,
            "persisted_evaluations": runner.FULL_BUDGET,
            "physical_call_starts": runner.FULL_BUDGET,
            "run_context_charged_evaluation_budget": runner.FULL_BUDGET,
            "sqlite_integrity": "ok",
        },
        "physical_call_started_count": runner.FULL_BUDGET,
        "problem": row_spec["case_id"],
        "run_context_digest_sha256": "a" * 64,
        "schema": "v21e3_terminal_receipt_v1",
        "status": "SUCCESS",
        "terminal_attempt_chain_sha256": "c" * 64,
        "terminal_decision_chain_sha256": "d" * 64,
        "terminal_evaluation_chain_sha256": "e" * 64,
        "unresolved_decision_count": 0,
    }
    terminal = {
        **terminal_core,
        "receipt_payload_sha256": runner._payload_sha256(terminal_core),
    }
    calls = {"verify": 0, "independent": 0}
    with tempfile.TemporaryDirectory(prefix=".factorial-row-replay-", dir=ROOT) as raw:
        attempt = Path(raw)
        (attempt / "trace.sqlite3").write_bytes(b"sealed-test-trace")
        (attempt / "terminal.receipt.json").write_bytes(
            runner._canonical_bytes(terminal)
        )
        (attempt / "independent.metric.json").write_bytes(
            runner._canonical_bytes(independent)
        )
        monkeypatch.setattr(
            runner.v7_runner,
            "load_v21e3_development_problem",
            lambda _path: object(),
        )
        monkeypatch.setattr(runner.v7_runner, "_load_trace_context", lambda _path: {})
        monkeypatch.setattr(runner.v7_runner, "_assert_context", lambda *_a, **_k: None)

        def fresh_verify(*_args: object, **_kwargs: object) -> dict[str, object]:
            calls["verify"] += 1
            return verification

        def fresh_independent(**kwargs: object) -> dict[str, object]:
            calls["independent"] += 1
            assert Path(kwargs["output"]).resolve().parent != attempt.resolve()
            return independent

        monkeypatch.setattr(runner, "verify_v21e3_trace_database", fresh_verify)
        monkeypatch.setattr(
            runner, "_successor_independent_metric_replay", fresh_independent
        )

        replayed, witness = evaluator._replay_row_evidence(
            ROOT,
            attempt,
            row_spec,
            row,
            source,
            (0.0, 0.0),
            (1.0, 1.0),
            expected_config,
        )

    assert calls == {"verify": 1, "independent": 1}
    assert replayed["exact_per_evaluation_left_continuous_hv_auc"] == 0.4
    assert replayed["cache_hit_rate_per_attempt"] == 100 / 2100
    assert witness["row_id"] == row_spec["row_id"]


def test_zero_case_standard_error_is_a_hold_not_a_pass(
    exact_plan: dict[str, object],
) -> None:
    inference, _digest = runner.load_inference_spec(ROOT)
    result = evaluator.evaluate_rows(
        exact_plan,
        _synthetic_rows(exact_plan, variable_effects=False),
        inference,
    )

    assert result["status"] == "HOLD_SUCCESSOR_DEVELOPMENT_PROMOTION_ZERO_STANDARD_ERROR"
    assert result["development_promotion_gate_passed"] is False
    assert result["zero_standard_error_hypotheses"]


def test_one_zero_standard_error_cell_holds_all_five_bounds_explicitly(
    exact_plan: dict[str, object],
) -> None:
    inference, _digest = runner.load_inference_spec(ROOT)
    rows = _statistical_golden_rows(exact_plan)
    seed_index = {seed: index for index, seed in enumerate(runner.SEEDS)}
    cache_shift = (-4, 1, 3)
    for row in rows:
        if row["family"] != "MOKP":
            continue
        seed = seed_index[row["seed"]]
        row["cache_hit_rate_per_attempt"] = {
            "MOKP_LEGACY": 512,
            "MOKP_ANCHOR_ONLY": 496,
            "MOKP_NOVELTY_ONLY": 512 - 128 + cache_shift[seed],
            "MOKP_BOTH": 496 - 128 + cache_shift[seed],
        }[row["arm_id"]] / 1024.0

    result = evaluator.evaluate_rows(exact_plan, rows, inference)

    assert result["zero_standard_error_hypotheses"] == [evaluator.HYPOTHESES[3]]
    assert result["development_promotion_gate_passed"] is False
    assert result["critical_value"] is None
    assert result["bootstrap_maxima_sha256"] is None
    assert all(cell["simultaneous_lower_bound"] is None for cell in result["cells"])
    assert all(cell["gate_passed"] is False for cell in result["cells"])


def test_runner_preflight_is_fail_closed_until_both_policy_fields_exist() -> None:
    available = {field.name for field in runner.fields(runner.V21E3HybridConfig)}
    required = {
        "post_initialization_search_policy",
        "mokp_novelty_generation_policy",
    }
    if required <= available:
        runner._preflight_policy_fields()
    else:
        with pytest.raises(runner.ContractError, match="policy implementation is not present"):
            runner._preflight_policy_fields()

