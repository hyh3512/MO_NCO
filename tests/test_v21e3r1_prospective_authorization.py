from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ijoc_submission_v21e3r1.scripts import (
    evaluate_v21e3r1_prospective_authorization as prospective_evaluator,
)


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "evaluate_v21e3r1_prospective_authorization.py"
)
BOUNDARY_FREEZER = (
    Path(__file__).resolve().parents[1]
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "freeze_v21e3r1_prospective_boundaries.py"
)
HEX = {
    "comparison": "6" * 64,
    "custody": "7" * 64,
}
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
PARENT_V7_DIAGNOSTIC_PLAN_SHA256 = (
    "4408d10944cb6511e99ff0bd95ded256b9c230b91d8806a7bd5b962f10622886"
)
EXPOSED_DEVELOPMENT_CASE_IDS = (
    "v21e3-mokp-development-n100-s00",
    "v21e3-mokp-development-n100-s01",
    "v21e3-mokp-development-n200-s00",
    "v21e3-mokp-development-n200-s01",
    "v21e3-mokp-development-n500-s00",
    "v21e3-mokp-development-n500-s01",
    "v21e3-motsp-development-n100-s00",
    "v21e3-motsp-development-n100-s01",
    "v21e3-motsp-development-n200-s00",
    "v21e3-motsp-development-n200-s01",
    "v21e3-motsp-development-n500-s00",
    "v21e3-motsp-development-n500-s01",
)
FACTORIAL_SEEDS = (31051, 31057, 31059)
LEGACY_SEARCH = "proposal_chain_v21e3r1_v1"
NEW_SEARCH = "post_commit_type_incumbent_anchor_development_v1"
LEGACY_NOVELTY = "legacy_retry_and_local_v21e3r1_v1"
NEW_NOVELTY = "single_attempt_rotating_feasible_exchange_no_refill_development_v1"
MOKP_FACTORIAL_ARMS = (
    ("MOKP_LEGACY", LEGACY_SEARCH, LEGACY_NOVELTY),
    ("MOKP_ANCHOR_ONLY", NEW_SEARCH, LEGACY_NOVELTY),
    ("MOKP_NOVELTY_ONLY", LEGACY_SEARCH, NEW_NOVELTY),
    ("MOKP_BOTH", NEW_SEARCH, NEW_NOVELTY),
)
MOTSP_FACTORIAL_ARMS = (
    ("MOTSP_LEGACY", LEGACY_SEARCH, LEGACY_NOVELTY),
    ("MOTSP_ANCHOR", NEW_SEARCH, LEGACY_NOVELTY),
)
EXPOSED_INPUT_BINDING = {
    "schema": "v21e3r1_exposed_development_input_binding_v1",
    "case_ids": list(EXPOSED_DEVELOPMENT_CASE_IDS),
    "manifest_sha256": {
        "ijoc_submission_v21e3/development_manifests_v1/config_manifest_development.json": (
            "d33ba2d83909af4fecff85f4663791b7c63b5ed56738a67f0eec6ccfd6336d4e"
        ),
        "ijoc_submission_v21e3/development_manifests_v1/reference_manifest_development.json": (
            "86336403c3e098f0e5022c796db1778552c0d92ca40d85953dc341eb534a4402"
        ),
        "ijoc_submission_v21e3/development_partitions_v1/case_manifest.json": (
            "1970361ba557aadd26de38aed008de11d11d158c797c00db1036cc4616cbdc8c"
        ),
    },
}


def _factorial_plan_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case_id in EXPOSED_DEVELOPMENT_CASE_IDS:
        family = "MOKP" if "-mokp-" in case_id else "MOTSP"
        size = next(size for size in (100, 200, 500) if f"-n{size}-" in case_id)
        arms = MOKP_FACTORIAL_ARMS if family == "MOKP" else MOTSP_FACTORIAL_ARMS
        for seed in FACTORIAL_SEEDS:
            for arm_id, search_policy, novelty_policy in arms:
                rows.append(
                    {
                        "ordinal": len(rows) + 1,
                        "row_id": (
                            f"{case_id}__seed-{seed}__arm-{arm_id.lower()}"
                        ),
                        "case_id": case_id,
                        "family": family,
                        "size": size,
                        "seed": seed,
                        "arm_id": arm_id,
                        "post_initialization_search_policy": search_policy,
                        "mokp_novelty_generation_policy": novelty_policy,
                        "case_artifact_path": (
                            "ijoc_submission_v21e3/development_partitions_v1/"
                            f"instances/{case_id}.json"
                        ),
                        "case_artifact_sha256": hashlib.sha256(
                            f"fixture:{case_id}".encode("utf-8")
                        ).hexdigest(),
                    }
                )
    assert len(rows) == 108
    return rows


def _factorial_aggregate_rows(
    plan_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "ordinal": row["ordinal"],
            "row_id": row["row_id"],
            "case_id": row["case_id"],
            "family": row["family"],
            "seed": row["seed"],
            "arm_id": row["arm_id"],
            "exact_per_evaluation_left_continuous_hv_auc": 0.5,
            "cache_hit_rate_per_attempt": 0.1,
            "row_sha256": hashlib.sha256(
                f"row:{row['row_id']}".encode("utf-8")
            ).hexdigest(),
            "trace_sha256": hashlib.sha256(
                f"trace:{row['row_id']}".encode("utf-8")
            ).hexdigest(),
            "terminal_receipt_sha256": hashlib.sha256(
                f"terminal:{row['row_id']}".encode("utf-8")
            ).hexdigest(),
            "independent_metric_receipt_sha256": hashlib.sha256(
                f"metric:{row['row_id']}".encode("utf-8")
            ).hexdigest(),
        }
        for row in plan_rows
    ]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_json(root: Path, name: str, value: object) -> dict[str, str]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical(value)
    path.write_bytes(raw)
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}


def _write_bytes(root: Path, name: str, raw: bytes) -> dict[str, str]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}


def _rewrite_bound_json(
    root: Path, binding: dict[str, str], mutate: object
) -> dict[str, str]:
    path = root / binding["path"]
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    raw = _canonical(value)
    path.write_bytes(raw)
    binding["sha256"] = hashlib.sha256(raw).hexdigest()
    return binding


def _payload_bound(value: dict[str, object], digest_field: str) -> dict[str, object]:
    assert digest_field not in value
    bound = dict(value)
    bound[digest_field] = hashlib.sha256(_canonical(bound)).hexdigest()
    return bound


def _authority_hold_fields() -> dict[str, object]:
    return {
        "runtime_authority": False,
        "scientific_authority": False,
        "selection_authority": False,
        "confirmation_authority": False,
        "formal_study_authority": False,
        "publication_status": "IJOC_HOLD",
    }


def _runtime_identities() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[1]
    outer = (
        repository
        / "ijoc_submission_v21e3r1"
        / "scripts"
        / "run_v21e3r1_recovery_bound_same_implementation_branch_replay_coverage.py"
    ).resolve()
    inner = (
        repository
        / "ijoc_submission_v21e3r1"
        / "scripts"
        / "run_v21e3r1_same_implementation_branch_replay_coverage.py"
    ).resolve()
    python = Path(sys.executable).resolve()
    return {
        "outer_runner_path": outer.as_posix(),
        "outer_runner_sha256": hashlib.sha256(outer.read_bytes()).hexdigest(),
        "inner_runner_path": inner.as_posix(),
        "inner_runner_sha256": hashlib.sha256(inner.read_bytes()).hexdigest(),
        "python_path": python.as_posix(),
        "python_sha256": hashlib.sha256(python.read_bytes()).hexdigest(),
        "python_version": sys.version,
    }


def _wrap_recovery_bound_same_implementation(
    root: Path,
    *,
    inner_binding: dict[str, str],
    diagnostic_binding: dict[str, str],
    diagnostic_value: dict[str, object],
    development_source_sha256: str,
) -> dict[str, str]:
    """Create a finite synthetic outer receipt/seal chain for gate tests."""

    outer_root = root / "same-implementation"
    inner_path = root / inner_binding["path"]
    inner_value = json.loads(inner_path.read_text(encoding="utf-8"))
    runtime = _runtime_identities()
    hold = _authority_hold_fields()
    diagnostic_rows = [
        {
            "ordinal": ordinal,
            "row_id": seal["row_id"],
            "family": "MOTSP" if ordinal == 1 else "MOKP",
            "size": 500 if ordinal == 1 else 100,
            "budget": 2000,
            "case_path": "synthetic/case.json",
            "case_sha256": "1" * 64,
            "attempt_directory": f"attempts/{seal['row_id']}/attempt-0001",
            "completed_marker_path": f"completed/{seal['row_id']}.json",
            "completed_marker_sha256": seal["diagnostic_completed_marker_sha256"],
            "trace_path": f"attempts/{seal['row_id']}/attempt-0001/trace.sqlite3",
            "trace_sha256": seal["diagnostic_trace_sha256"],
        }
        for ordinal, seal in enumerate(inner_value["row_seals"], start=1)
    ]
    exact_diagnostic = {
        "schema": "v21e3r1_recovery_bound_exact504_diagnostic_binding_v1",
        "status": "PASS_EXACT504_DIAGNOSTIC_TREE_REVALIDATED_INPUT_ONLY",
        "expected_rows": 504,
        "plan_sha256": diagnostic_value["plan_sha256"],
        "source_snapshot_sha256": development_source_sha256,
        "diagnostic_receipt_sha256": diagnostic_binding["sha256"],
        "diagnostic_aggregate_sha256": diagnostic_value["aggregate_sha256"],
        "rows": diagnostic_rows,
        "implementation_independence": False,
        "scientific_independence": False,
        **hold,
    }
    exact_diagnostic_sha256 = hashlib.sha256(
        _canonical(exact_diagnostic)
    ).hexdigest()
    provenance = _payload_bound(
        {
            "schema": "v21e3r1_recovered_diagnostic_provenance_binding_v1",
            "status": "PASS_CHAIN_VALIDATED_FOR_REPLAY_INPUT_ONLY",
            "expected_rows": 504,
            "plan_sha256": diagnostic_value["plan_sha256"],
            "source_snapshot_sha256": development_source_sha256,
            "diagnostic_receipt_sha256": diagnostic_binding["sha256"],
            "diagnostic_aggregate_sha256": diagnostic_value["aggregate_sha256"],
            "exact504_diagnostic_binding": exact_diagnostic,
            "exact504_diagnostic_binding_sha256": exact_diagnostic_sha256,
            "implementation_independence": False,
            "scientific_independence": False,
            **hold,
        },
        "provenance_payload_sha256",
    )
    _write_json(
        root,
        "same-implementation/diagnostic_provenance.binding.json",
        provenance,
    )
    provenance_sha256 = str(provenance["provenance_payload_sha256"])

    preflight_root = outer_root / "preflight"
    selected = diagnostic_rows[0]
    preflight_plan = _payload_bound(
        {
            "schema": "v21e3r1_recovery_bound_n500_operational_preflight_plan_v1",
            "status": "FROZEN_N500_FULL_BUDGET_OPERATIONAL_PREFLIGHT_ONLY",
            "project_root": root.resolve().as_posix(),
            "diagnostic_root": "synthetic-diagnostic-root",
            "output_root": preflight_root.resolve().as_posix(),
            "selected_row": selected,
            "charged_evaluation_budget": 2000,
            "verification_jobs": 1,
            "row_timeout_seconds": 2400,
            "preflight_required_for_full_coverage": True,
            "diagnostic_provenance_payload_sha256": provenance_sha256,
            "diagnostic_binding_sha256": exact_diagnostic_sha256,
            "runtime_identities": runtime,
            "implementation_independence": False,
            "scientific_independence": False,
            **hold,
        },
        "plan_payload_sha256",
    )
    preflight_plan_binding = _write_json(
        root,
        "same-implementation/preflight/n500_preflight.plan.json",
        preflight_plan,
    )
    branch_receipt = _payload_bound(
        {
            "schema": "v21e3r1_same_implementation_branch_replay_v1",
            "status": "PASS_SAME_IMPLEMENTATION_FULL_BRANCH_REEXECUTION",
            "row_id": selected["row_id"],
            "implementation_independence": False,
            "scientific_independence": False,
            **hold,
        },
        "receipt_payload_sha256",
    )
    branch_binding = _write_json(
        root,
        "same-implementation/preflight/inner/branch.replay.json",
        branch_receipt,
    )
    preflight_receipt = _payload_bound(
        {
            "schema": "v21e3r1_recovery_bound_n500_operational_preflight_receipt_v1",
            "status": "PASS_N500_FULL_BUDGET_OPERATIONAL_PREFLIGHT_ONLY",
            "plan_path": "n500_preflight.plan.json",
            "plan_sha256": preflight_plan_binding["sha256"],
            "plan_payload_sha256": preflight_plan["plan_payload_sha256"],
            "selected_row": selected,
            "charged_evaluation_budget": 2000,
            "verification_jobs": 1,
            "row_timeout_seconds": 2400,
            "wall_time_seconds": 1.0,
            "process_isolation": "INNER_V1_ISOLATED_PROCESS_BOUNDARY",
            "branch_replay_receipt_path": "inner/branch.replay.json",
            "branch_replay_receipt_sha256": branch_binding["sha256"],
            "branch_replay_payload_sha256": branch_receipt[
                "receipt_payload_sha256"
            ],
            "diagnostic_provenance_payload_sha256": provenance_sha256,
            "diagnostic_binding_sha256": exact_diagnostic_sha256,
            "runtime_identities_before": runtime,
            "runtime_identities_after": runtime,
            "preflight_required_for_full_coverage": True,
            "operational_only": True,
            "implementation_independence": False,
            "scientific_independence": False,
            **hold,
        },
        "receipt_payload_sha256",
    )
    preflight_receipt_binding = _write_json(
        root,
        "same-implementation/preflight/n500_preflight.receipt.json",
        preflight_receipt,
    )
    preflight_seal = _payload_bound(
        {
            "schema": "v21e3r1_recovery_bound_n500_preflight_success_seal_v1",
            "status": "SEALED_N500_OPERATIONAL_PREFLIGHT_SUCCESS_RECEIPT",
            "receipt_path": "n500_preflight.receipt.json",
            "receipt_sha256": preflight_receipt_binding["sha256"],
            "receipt_payload_sha256": preflight_receipt["receipt_payload_sha256"],
            "plan_sha256": preflight_plan_binding["sha256"],
            **hold,
        },
        "seal_payload_sha256",
    )
    preflight_seal_binding = _write_json(
        root,
        "same-implementation/preflight/n500_preflight.receipt.seal.json",
        preflight_seal,
    )
    preflight_binding = {
        "schema": "v21e3r1_recovery_bound_n500_preflight_binding_v1",
        "status": "PASS_SEALED_N500_PREFLIGHT_REQUIRED_FOR_FULL_COVERAGE",
        "selected_row_id": selected["row_id"],
        "charged_evaluation_budget": 2000,
        "verification_jobs": 1,
        "row_timeout_seconds": 2400,
        "receipt_path": (
            root / preflight_receipt_binding["path"]
        ).resolve().as_posix(),
        "receipt_sha256": preflight_receipt_binding["sha256"],
        "receipt_payload_sha256": preflight_receipt["receipt_payload_sha256"],
        "seal_path": (root / preflight_seal_binding["path"]).resolve().as_posix(),
        "seal_sha256": preflight_seal_binding["sha256"],
        "seal_payload_sha256": preflight_seal["seal_payload_sha256"],
        "plan_sha256": preflight_plan_binding["sha256"],
        "diagnostic_provenance_payload_sha256": provenance_sha256,
        "diagnostic_binding_sha256": exact_diagnostic_sha256,
        **hold,
    }
    outer_plan = _payload_bound(
        {
            "schema": "v21e3r1_recovery_bound_coverage_plan_v1",
            "status": "FROZEN_RECOVERY_BOUND_SAME_IMPLEMENTATION_EXACT504",
            "project_root": root.resolve().as_posix(),
            "diagnostic_root": "synthetic-diagnostic-root",
            "output_root": outer_root.resolve().as_posix(),
            "expected_rows": 504,
            "jobs": 1,
            "row_timeout_seconds": 2400,
            "provenance_payload_sha256": provenance_sha256,
            "diagnostic_binding_sha256": exact_diagnostic_sha256,
            "preflight_binding": preflight_binding,
            "runtime_identities": runtime,
            "implementation_independence": False,
            "scientific_independence": False,
            **hold,
        },
        "plan_payload_sha256",
    )
    outer_plan_binding = _write_json(
        root,
        "same-implementation/recovery_bound_coverage.plan.json",
        outer_plan,
    )
    claim = _payload_bound(
        {
            "schema": "v21e3r1_recovery_bound_coverage_execution_claim_v1",
            "status": "SEALED_EXCLUSIVE_SAME_IMPLEMENTATION_EXECUTION_CLAIM",
            "execution_number": 1,
            "plan_sha256": outer_plan_binding["sha256"],
            "provenance_payload_sha256": provenance_sha256,
            "preflight_receipt_sha256": preflight_receipt_binding["sha256"],
            "preflight_seal_sha256": preflight_seal_binding["sha256"],
            "jobs": 1,
            "row_timeout_seconds": 2400,
            "inner_resume": False,
            "runtime_identities": runtime,
            "implementation_independence": False,
            "scientific_independence": False,
            **hold,
        },
        "claim_payload_sha256",
    )
    claim_binding = _write_json(
        root,
        "same-implementation/executions/execution-0001.claim.json",
        claim,
    )
    inner_raw = inner_path.read_bytes()
    outer_receipt = _payload_bound(
        {
            "schema": "v21e3r1_recovery_bound_coverage_receipt_v1",
            "status": "PASS_RECOVERY_BOUND_SAME_IMPLEMENTATION_EXACT504_ONLY",
            "expected_rows": 504,
            "jobs": 1,
            "row_timeout_seconds": 2400,
            "wall_time_seconds": 1.0,
            "process_isolation": "INNER_V1_PER_ROW_ISOLATED_PROCESS_BOUNDARY",
            "plan_sha256": outer_plan_binding["sha256"],
            "plan_payload_sha256": outer_plan["plan_payload_sha256"],
            "provenance_payload_sha256": provenance_sha256,
            "diagnostic_binding_sha256": exact_diagnostic_sha256,
            "preflight_receipt_sha256": preflight_receipt_binding["sha256"],
            "preflight_seal_sha256": preflight_seal_binding["sha256"],
            "execution_claim_path": "executions/execution-0001.claim.json",
            "execution_claim_sha256": claim_binding["sha256"],
            "execution_claim_payload_sha256": claim["claim_payload_sha256"],
            "inner_receipt_path": "inner_v1/branch_replay_coverage.receipt.json",
            "inner_receipt_sha256": hashlib.sha256(inner_raw).hexdigest(),
            "inner_receipt_payload_sha256": hashlib.sha256(
                _canonical(inner_value)
            ).hexdigest(),
            "runtime_identities_before": runtime,
            "runtime_identities_after": runtime,
            "same_implementation_only": True,
            "implementation_independence": False,
            "algorithm_execution_independence": False,
            "scientific_independence": False,
            **hold,
        },
        "receipt_payload_sha256",
    )
    outer_binding = _write_json(
        root,
        "same-implementation/recovery_bound_coverage.receipt.json",
        outer_receipt,
    )
    outer_seal = _payload_bound(
        {
            "schema": "v21e3r1_recovery_bound_coverage_success_seal_v1",
            "status": "SEALED_RECOVERY_BOUND_COVERAGE_SUCCESS_RECEIPT",
            "receipt_path": "recovery_bound_coverage.receipt.json",
            "receipt_sha256": outer_binding["sha256"],
            "receipt_payload_sha256": outer_receipt["receipt_payload_sha256"],
            "plan_sha256": outer_plan_binding["sha256"],
            "provenance_payload_sha256": provenance_sha256,
            "preflight_receipt_sha256": preflight_receipt_binding["sha256"],
            "inner_receipt_sha256": hashlib.sha256(inner_raw).hexdigest(),
            **hold,
        },
        "seal_payload_sha256",
    )
    _write_json(
        root,
        "same-implementation/recovery_bound_coverage.receipt.seal.json",
        outer_seal,
    )
    return outer_binding


def _reseal_recovery_bound_inner(
    root: Path, outer_binding: dict[str, str]
) -> None:
    inner_path = (
        root
        / "same-implementation"
        / "inner_v1"
        / "branch_replay_coverage.receipt.json"
    )
    inner_value = json.loads(inner_path.read_text(encoding="utf-8"))
    inner_raw = inner_path.read_bytes()
    outer_path = root / outer_binding["path"]
    outer = json.loads(outer_path.read_text(encoding="utf-8"))
    outer["inner_receipt_sha256"] = hashlib.sha256(inner_raw).hexdigest()
    outer["inner_receipt_payload_sha256"] = hashlib.sha256(
        _canonical(inner_value)
    ).hexdigest()
    outer_core = dict(outer)
    outer_core.pop("receipt_payload_sha256")
    outer["receipt_payload_sha256"] = hashlib.sha256(
        _canonical(outer_core)
    ).hexdigest()
    outer_raw = _canonical(outer)
    outer_path.write_bytes(outer_raw)
    outer_binding["sha256"] = hashlib.sha256(outer_raw).hexdigest()

    seal_path = outer_path.with_name("recovery_bound_coverage.receipt.seal.json")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["receipt_sha256"] = outer_binding["sha256"]
    seal["receipt_payload_sha256"] = outer["receipt_payload_sha256"]
    seal["inner_receipt_sha256"] = outer["inner_receipt_sha256"]
    seal_core = dict(seal)
    seal_core.pop("seal_payload_sha256")
    seal["seal_payload_sha256"] = hashlib.sha256(_canonical(seal_core)).hexdigest()
    seal_path.write_bytes(_canonical(seal))


def _identity(
    *,
    candidate_id: str = "C1",
    simultaneous_sha256: str,
    operator_reanalysis_spec_sha256: str,
    study_metric_spec_sha256: str,
    development_source_sha256: str,
    successor_source_sha256: str,
    successor_config_sha256: str,
) -> dict[str, str]:
    return {
        "study_id": "v21e3r1-prospective-test",
        "candidate_id": candidate_id,
        "development_source_sha256": development_source_sha256,
        "successor_source_sha256": successor_source_sha256,
        "successor_config_sha256": successor_config_sha256,
        "operator_reanalysis_spec_sha256": operator_reanalysis_spec_sha256,
        "study_metric_spec_sha256": study_metric_spec_sha256,
        "simultaneous_inference_spec_sha256": simultaneous_sha256,
    }


def _make_common_evidence(
    root: Path,
    *,
    external_status: str = "HOLD_DESIGN_ONLY_NO_EXTERNAL_PRODUCER",
    independent_producer: bool = False,
    independent_custody: bool = False,
    implementation_code_disjoint: bool = False,
    algorithm_execution_independence: bool = False,
    promotion_passed: bool = True,
    promotion_zero_standard_error: bool = False,
    use_recovery_bound_same_implementation: bool = True,
    candidate_id: str = "C1",
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    corrected_rows = _write_bytes(
        root,
        "operator_accounting.rows.jsonl",
        b'{"row_id":"synthetic-test-row"}\n',
    )
    corrected_aggregate = _write_json(
        root,
        "operator_accounting.aggregate.json",
        {"completed_rows": 504, "operator_charge_double_count_corrected": True},
    )
    operator_reanalysis_spec = _write_json(
        root,
        "metric/v21e3r1_operator_accounting_reanalysis_spec_v1.json",
        {"schema": "synthetic_v21e3r1_metric_spec_v1"},
    )
    metric_source = _write_bytes(
        root,
        "source/reanalyze_v21e3r1_operator_accounting.py",
        b"# synthetic metric source for gate tests\n",
    )
    development_entries = [
        {"path": "mo_nco/development.py", "bytes": 17, "sha256": "a" * 64}
    ]
    development_source_root = hashlib.sha256(_canonical(development_entries)).hexdigest()
    development_source_manifest = _write_json(
        root,
        "same-implementation/inner_v1/source.manifest.json",
        {
            "schema": "v21e3r1_branch_replay_source_manifest_binding_v1",
            "source_root_sha256": development_source_root,
            "entries": development_entries,
        },
    )
    repository = Path(__file__).resolve().parents[1]

    def real_source_entry(relative: str) -> dict[str, object]:
        raw = (repository / relative).read_bytes()
        return {
            "path": relative,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    successor_entries = sorted(
        [
            real_source_entry(
                "independent_reproduction/recompute_v21e3r1_successor_metrics.py"
            ),
            real_source_entry("mo_nco/pareto_ijoc_analysis.py"),
            {"path": "mo_nco/successor.py", "bytes": 19, "sha256": "b" * 64},
        ],
        key=lambda entry: (str(entry["path"]).casefold(), str(entry["path"])),
    )
    successor_source_root = hashlib.sha256(_canonical(successor_entries)).hexdigest()
    successor_source_manifest = _write_json(
        root,
        "successor-freeze/source.manifest.json",
        {
            "schema": "v21e3r1_branch_replay_source_manifest_binding_v1",
            "source_root_sha256": successor_source_root,
            "entries": successor_entries,
        },
    )
    semantic_config = _write_json(
        root,
        "successor-freeze/semantic.config.json",
        {
            "schema": "v21e3r1_successor_semantic_config_v1",
            "study_id": "v21e3r1-prospective-test",
            "candidate_id": candidate_id,
            "parameters": EXPECTED_SEMANTIC_PARAMETERS,
        },
    )
    by_source_path = {str(entry["path"]): entry for entry in successor_entries}
    study_metric_core = {
        "schema": "v21e3r1_study_metric_spec_v1",
        "status": "FROZEN_BEFORE_SELECTION",
        "metric_id": "normalized_left_continuous_hypervolume_auc",
        "effect_direction": "LARGER_IS_BETTER",
        "evaluation_axis": "CHARGED_EVALUATIONS",
        "objective_dimension": 2,
        "normalization_contract": "CASE_FROZEN_LOWER_UPPER_AFFINE_TO_UNIT_SQUARE",
        "reference_point": [1.0, 1.0],
        "archive_contract": "ALL_CHARGED_EVALUATED_NONDOMINATED_ARCHIVE",
        "integration_contract": "EAUC=(1/B)*SUM_{b=1..B}HV(A_{b-1})",
        "primary_metric": "normalized_left_continuous_hypervolume_auc",
        "secondary_reporting_metrics": [
            "terminal_hypervolume",
            "attempt_count",
            "physical_start_count",
            "charged_evaluation_count",
            "wall_time_seconds",
            "peak_rss_bytes",
        ],
        "seed_within_case_aggregation": "ARITHMETIC_MEAN_WITHIN_CASE_ARM",
        "case_cluster_estimand": "MEAN_OF_PAIRED_CASE_DIFFERENCES",
        "row_crosscheck": {
            "required": True,
            "scope": "EVERY_FORMAL_STUDY_ROW",
            "tolerance": 0.0,
            "failure_policy": "HOLD_ON_ANY_MISMATCH",
        },
        "production_metric_source": by_source_path["mo_nco/pareto_ijoc_analysis.py"],
        "independent_metric_source": by_source_path[
            "independent_reproduction/recompute_v21e3r1_successor_metrics.py"
        ],
        "practical_thresholds_bound_in_simultaneous_spec": True,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_study_authorized": False,
        "scientific_claim_authorized": False,
        "ijoc_submission_status": "IJOC_HOLD",
    }
    study_metric_value = dict(study_metric_core)
    study_metric_value["receipt_payload_sha256"] = hashlib.sha256(
        _canonical(study_metric_core)
    ).hexdigest()
    study_metric_spec = _write_json(
        root,
        "successor-freeze/study.metric-spec.json",
        study_metric_value,
    )
    simultaneous_source = _write_bytes(
        root,
        "successor-freeze/recompute-simultaneous-bounds.py",
        (
            Path(__file__).resolve().parents[1]
            / "independent_reproduction"
            / "recompute_v21e3r1_simultaneous_bounds.py"
        ).read_bytes(),
    )
    simultaneous_test = _write_bytes(
        root,
        "successor-freeze/recompute-simultaneous-bounds.tests.py",
        b"# bound simultaneous evaluator tests\n",
    )
    historical = _write_json(
        root,
        "historical.json",
        {
            "schema": "v21e3r1_v4_v6_historical_preservation_receipt_v1",
            "status": "PASS_HISTORICAL_V4_V6_IDENTITY_AND_RELATIONSHIP",
            "historical_row_count_each": 108,
            "archive_member_count_each": 701,
            "unchanged_member_count": 700,
            "v4_removed_member": "old/member.json",
            "v6_replacement_member": "new/member.json",
            "identities": {
                f"artifact-{index}": {"bytes": index + 1, "sha256": f"{index + 8:x}" * 64}
                for index in range(6)
            },
            "historical_outputs_modified": False,
            "implementation_independence": False,
            "scientific_independence": False,
            "selection_authorized": False,
            "formal_authorized": False,
            "submission_status": "IJOC_HOLD",
        },
    )
    diagnostic_value = {
        "schema": "v21e3r1_exposed_development_diagnostic_receipt_v2",
        "status": "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "matrix_mode": "FULL_504",
        "completed_rows": 504,
        "expected_rows": 504,
        "plan_sha256": "8" * 64,
        "source_snapshot_sha256": development_source_root,
        "aggregate_sha256": "9" * 64,
        "selection_entropy_release": "PROHIBITED",
        "confirmation_materialization": "PROHIBITED",
        "formal_materialization": "PROHIBITED",
    }
    diagnostic = _write_json(root, "diagnostic.json", diagnostic_value)
    selection_cells = [
        {
            "hypothesis_id": f"{family}:{candidate}-{reference}",
            "family": family,
            "candidate": candidate,
            "reference": reference,
            "threshold_roles": roles,
        }
        for family in ("MOKP", "MOTSP")
        for candidate, reference, roles in (
            ("C1", "C0", ["primary", "adjacent"]),
            ("C2", "C0", ["primary"]),
            ("C2", "C1", ["adjacent"]),
            ("C3", "C0", ["primary"]),
            ("C3", "C2", ["adjacent"]),
        )
    ]
    confirmation_cells = [
        {
            "hypothesis_template": f"{family}:SELECTED-{reference}",
            "family": family,
            "candidate": "SELECTED",
            "reference": reference,
            "threshold_role": role,
        }
        for family in ("MOKP", "MOTSP")
        for reference, role in (("C0", "primary"), ("PREDECESSOR", "adjacent"))
    ]
    simultaneous_core = {
        "schema": "v21e3r1_simultaneous_inference_spec_v2",
        "status": "PASS_FROZEN_BEFORE_SELECTION_ENGINEERING_ONLY",
        "scope": "FROZEN_PROSPECTIVE_DESIGN_ONLY_NO_CASE_MATERIALIZATION",
        "study_id": "v21e3r1-prospective-test",
        "candidate_id": candidate_id,
        "successor_source_sha256": successor_source_root,
        "successor_config_sha256": semantic_config["sha256"],
        "study_metric_spec_sha256": study_metric_spec["sha256"],
        "evaluator_source_path": Path(simultaneous_source["path"]).name,
        "evaluator_source_sha256": simultaneous_source["sha256"],
        "evaluator_test_path": Path(simultaneous_test["path"]).name,
        "evaluator_test_sha256": simultaneous_test["sha256"],
        "method": "one_sided_observed_se_max_t_paired_case_cluster_bootstrap_v1",
        "families": ["MOKP", "MOTSP"],
        "candidates": ["C0", "C1", "C2", "C3"],
        "familywise_alpha": 0.05,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 20_260_823,
        "quantile_convention": "CEIL((1-ALPHA)*(B+1))_ORDER_STATISTIC",
        "critical_value_floor": 0.0,
        "rng_protocol": "SHA256_COUNTER_U64_REJECTION_V1",
        "rng_domain": "v21e3r1-simultaneous-case-bootstrap-v1",
        "cluster_unit": "PAIRED_CASE",
        "seed_aggregation": "MEAN_WITHIN_CASE_ARM",
        "resampling_rule": "WITH_REPLACEMENT_INDEPENDENT_BY_FAMILY_SHARED_ACROSS_CELLS_WITHIN_FAMILY",
        "centering": "BOOTSTRAP_MEAN_MINUS_OBSERVED_MEAN",
        "studentization_denominator": "OBSERVED_CASE_CLUSTER_STANDARD_ERROR",
        "familywise_scope": "JOINT_ACROSS_BOTH_FAMILIES",
        "practical_thresholds": {
            "adjacent_mechanism_effect": 0.005,
            "primary_effect": 0.0,
        },
        "selection_cells": selection_cells,
        "selection_cell_count": 10,
        "confirmation_cells": confirmation_cells,
        "confirmation_cell_count": 4,
        "selection_and_confirmation_disjoint_by_construction": True,
        "frozen_before_selection": True,
        "selection_cases_materialized": False,
        "confirmation_cases_materialized": False,
        "formal_cases_materialized": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_study_authorized": False,
        "scientific_claim_authorized": False,
        "ijoc_submission_status": "IJOC_HOLD",
    }
    simultaneous_value = dict(simultaneous_core)
    simultaneous_value["receipt_payload_sha256"] = hashlib.sha256(
        _canonical(simultaneous_core)
    ).hexdigest()
    simultaneous = _write_json(
        root,
        "successor-freeze/simultaneous-inference.spec.json",
        simultaneous_value,
    )
    identity = _identity(
        candidate_id=candidate_id,
        simultaneous_sha256=simultaneous["sha256"],
        operator_reanalysis_spec_sha256=operator_reanalysis_spec["sha256"],
        study_metric_spec_sha256=study_metric_spec["sha256"],
        development_source_sha256=development_source_root,
        successor_source_sha256=successor_source_root,
        successor_config_sha256=semantic_config["sha256"],
    )
    reanalysis = _write_json(
        root,
        "reanalysis.json",
        {
            "schema": "v21e3r1_corrected_operator_accounting_reanalysis_receipt_v1",
            "status": "PASS_CORRECTED_REANALYSIS_EXACT_504_DEVELOPMENT_ONLY",
            "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
            "matrix_mode": "FULL_504",
            "completed_rows": 504,
            "expected_rows": 504,
            "charged_evaluations_per_row": 2000,
            "evaluation_charged_evaluations_sum": 1_008_000,
            "attempt_charged_evaluations_sum": 1_008_000,
            "legacy_operator_charged_evaluations_sum": 2_016_000,
            "operator_charge_double_count_corrected": True,
            "all_rows_reanalyzed": True,
            "original_artifacts_modified": False,
            "plan_sha256": "8" * 64,
            "diagnostic_receipt_sha256": diagnostic["sha256"],
            "diagnostic_aggregate_sha256": "9" * 64,
            "development_source_sha256": identity["development_source_sha256"],
            "rows_path": corrected_rows["path"],
            "rows_sha256": corrected_rows["sha256"],
            "aggregate_path": corrected_aggregate["path"],
            "aggregate_sha256": corrected_aggregate["sha256"],
            "metric_spec_path": operator_reanalysis_spec["path"],
            "metric_spec_sha256": operator_reanalysis_spec["sha256"],
            "metric_source_path": metric_source["path"],
            "metric_source_sha256": metric_source["sha256"],
            "implementation_independence": False,
            "algorithm_execution_independence": False,
            "scientific_independence": False,
            "selection_authorized": False,
            "confirmation_authorized": False,
            "formal_study_authorized": False,
            "publication_status": "IJOC_HOLD",
        },
    )
    source_core = {
        "schema": "v21e3r1_successor_source_freeze_receipt_v2",
        "status": "PASS_SUCCESSOR_SOURCE_AND_CONFIG_FREEZE_ENGINEERING_ONLY",
        "study_id": identity["study_id"],
        "candidate_id": identity["candidate_id"],
        "parent_development_source_sha256": identity["development_source_sha256"],
        "source_snapshot_sha256": identity["successor_source_sha256"],
        "source_manifest_sha256": successor_source_manifest["sha256"],
        "semantic_config_sha256": identity["successor_config_sha256"],
        "study_metric_spec_sha256": identity["study_metric_spec_sha256"],
        "simultaneous_inference_spec_sha256": identity[
            "simultaneous_inference_spec_sha256"
        ],
        "source_entry_count": len(successor_entries),
        "source_total_bytes": sum(entry["bytes"] for entry in successor_entries),
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
    source_value = dict(source_core)
    source_value["receipt_payload_sha256"] = hashlib.sha256(
        _canonical(source_core)
    ).hexdigest()
    source = _write_json(
        root,
        "successor-freeze/successor-source.freeze.receipt.json",
        source_value,
    )
    row_seals = [
        {
            "row_id": f"row-{ordinal:04d}",
            "plan_ordinal": ordinal + 1,
            "coverage_completed_marker_sha256": "a" * 64,
            "diagnostic_completed_marker_sha256": "b" * 64,
            "diagnostic_trace_sha256": "c" * 64,
            "branch_replay_receipt_sha256": "d" * 64,
        }
        for ordinal in range(504)
    ]
    same_impl = _write_json(
        root,
        "same-implementation/inner_v1/branch_replay_coverage.receipt.json",
        {
            "schema": "v21e3r1_branch_replay_coverage_receipt_v1",
            "status": "PASS_SAME_IMPLEMENTATION_BRANCH_REPLAY_EXACT_504_DEVELOPMENT_ONLY",
            "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
            "matrix_mode": "FULL_504",
            "completed_rows": 504,
            "expected_rows": 504,
            "exact_full_504_coverage": True,
            "diagnostic_plan_sha256": "8" * 64,
            "diagnostic_receipt_sha256": diagnostic["sha256"],
            "diagnostic_aggregate_sha256": "9" * 64,
            "source_snapshot_sha256": identity["development_source_sha256"],
            "source_manifest_path": Path(development_source_manifest["path"]).name,
            "source_manifest_sha256": development_source_manifest["sha256"],
            "row_order_rule": "FROZEN_DIAGNOSTIC_PLAN_CASE_SEED_ARM_ORDER",
            "row_seals": row_seals,
            "row_seals_sha256": hashlib.sha256(_canonical(row_seals)).hexdigest(),
            "verification_jobs": 1,
            "verification_jobs_observed": [1],
            "row_timeout_seconds": 2400,
            "parallel_execution_semantics": "VERIFICATION_ONLY_PLAN_ORDERED_FINALIZATION_NO_RUNTIME_OR_PERFORMANCE_CLAIM",
            "implementation_independence": False,
            "scientific_independence": False,
            "third_party_replication": False,
            "selection_authorized": False,
            "confirmation_authorized": False,
            "formal_authorized": False,
            "selection_entropy_release": "PROHIBITED",
            "confirmation_materialization": "PROHIBITED",
            "formal_materialization": "PROHIBITED",
            "runtime_efficiency_claims": False,
            "scientific_performance_claims": False,
            "ijoc_submission_status": "IJOC_HOLD",
        },
    )
    baseline = _write_json(
        root,
        "baselines.json",
        {
            "schema": "v21e3r1_external_family_native_strong_baseline_registry_receipt_v1",
            "status": "PASS_EXTERNAL_FAMILY_NATIVE_STRONG_BASELINES",
            "study_id": identity["study_id"],
            "candidate_id": identity["candidate_id"],
            "metric_spec_sha256": identity["study_metric_spec_sha256"],
            "families": [
                {
                    "family": family,
                    "baselines": [
                        {
                            "baseline_id": f"external-{family.lower()}",
                            "classification": "EXTERNAL_FAMILY_NATIVE_STRONG_BASELINE",
                            "external": True,
                            "family_native": True,
                            "strong": True,
                            "source_manifest_sha256": char * 64,
                            "evaluation_receipt_sha256": char.upper().lower() * 64,
                            "metric_spec_sha256": identity["study_metric_spec_sha256"],
                        }
                    ],
                }
                for family, char in (("MOKP", "b"), ("MOTSP", "c"))
            ],
        },
    )
    external = _write_json(
        root,
        "external-replay.json",
        {
            "schema": "v21e3r1_external_algorithm_replay_receipt_v1",
            "status": external_status,
            "study_id": identity["study_id"],
            "candidate_id": identity["candidate_id"],
            "successor_source_sha256": identity["successor_source_sha256"],
            "successor_config_sha256": identity["successor_config_sha256"],
            "metric_spec_sha256": identity["study_metric_spec_sha256"],
            "reference_producer_id": "reference-producer",
            "external_producer_id": "external-producer",
            "reference_source_manifest_sha256": "d" * 64,
            "external_source_manifest_sha256": "e" * 64,
            "neutral_comparison_receipt_sha256": HEX["comparison"],
            "event_streams_match": external_status == "PASS_EXTERNAL_ALGORITHM_REPLAY",
            "independent_producer": independent_producer,
            "independent_custody": independent_custody,
            "implementation_code_disjoint": implementation_code_disjoint,
            "algorithm_execution_independence": algorithm_execution_independence,
            "custody_receipt_sha256": HEX["custody"],
        },
    )
    same_impl_gate_binding = (
        _wrap_recovery_bound_same_implementation(
            root,
            inner_binding=same_impl,
            diagnostic_binding=diagnostic,
            diagnostic_value=diagnostic_value,
            development_source_sha256=identity["development_source_sha256"],
        )
        if use_recovery_bound_same_implementation
        else same_impl
    )
    bindings = {
            "historical_preservation": historical,
            "exact_504_diagnostic": diagnostic,
            "corrected_reanalysis": reanalysis,
            "successor_source_freeze": source,
            "same_implementation_coverage": same_impl_gate_binding,
            "baseline_registry": baseline,
            "external_algorithm_replay": external,
            "simultaneous_inference_spec": simultaneous,
    }
    bindings["successor_development_promotion"] = _write_development_promotion(
        root,
        identity=identity,
        source_freeze=source,
        passed=promotion_passed,
        zero_standard_error=promotion_zero_standard_error,
    )
    return bindings, identity


def _write_development_promotion(
    root: Path,
    *,
    identity: dict[str, str],
    source_freeze: dict[str, str],
    passed: bool,
    zero_standard_error: bool = False,
) -> dict[str, str]:
    if passed and zero_standard_error:
        raise AssertionError("zero-standard-error promotion cannot pass")
    inference_relative = (
        "ijoc_submission_v21e3r1/development/"
        "V21E3R1_SUCCESSOR_DEVELOPMENT_FACTORIAL_INFERENCE_V1.json"
    )
    inference_raw = (
        Path(__file__).resolve().parents[1] / inference_relative
    ).read_bytes()
    inference = _write_bytes(root, inference_relative, inference_raw)
    assert inference["sha256"] == (
        "5aa767bcc00c5ee8d220defa86b358d3e72a5849a99712a0f486159f1f032f3d"
    )
    source_value = json.loads((root / source_freeze["path"]).read_text(encoding="utf-8"))
    source_binding = {
        "schema": "v21e3r1_successor_factorial_source_binding_v2",
        "study_id": identity["study_id"],
        "candidate_id": identity["candidate_id"],
        "parent_development_source_sha256": identity["development_source_sha256"],
        "receipt_path": source_freeze["path"],
        "receipt_sha256": source_freeze["sha256"],
        "source_manifest_path": "successor-freeze/source.manifest.json",
        "source_manifest_sha256": source_value["source_manifest_sha256"],
        "source_snapshot_sha256": identity["successor_source_sha256"],
        "semantic_config_sha256": identity["successor_config_sha256"],
        "study_metric_spec_sha256": identity["study_metric_spec_sha256"],
        "simultaneous_inference_spec_sha256": identity[
            "simultaneous_inference_spec_sha256"
        ],
        "factorial_inference_spec_path": inference_relative,
        "factorial_inference_spec_sha256": inference["sha256"],
    }
    matrix_directory = root / "successor-factorial"
    matrix_directory.mkdir(parents=True, exist_ok=True)
    plan_rows = _factorial_plan_rows()
    plan = {
        "schema": "v21e3r1_successor_development_factorial_plan_v2",
        "status": "FROZEN_EXACT_108_SUCCESSOR_DEVELOPMENT_FACTORIAL",
        "phase": "development",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "parent_v7_diagnostic_plan_path": "development/frozen-v7-plan.json",
        "parent_v7_diagnostic_plan_sha256": PARENT_V7_DIAGNOSTIC_PLAN_SHA256,
        "parent_v7_source_snapshot_sha256": identity[
            "development_source_sha256"
        ],
        "case_ids": list(EXPOSED_DEVELOPMENT_CASE_IDS),
        "seeds": list(FACTORIAL_SEEDS),
        "arms_by_family": {
            "MOKP": [
                "MOKP_LEGACY",
                "MOKP_ANCHOR_ONLY",
                "MOKP_NOVELTY_ONLY",
                "MOKP_BOTH",
            ],
            "MOTSP": ["MOTSP_LEGACY", "MOTSP_ANCHOR"],
        },
        "charged_evaluation_budget": 2000,
        "checkpoint_period": 200,
        "expected_rows": 108,
        "row_timeout_seconds": 1800,
        "input_binding": EXPOSED_INPUT_BINDING,
        "source_binding": source_binding,
        "inference_spec_binding": {
            "path": inference_relative,
            "sha256": inference["sha256"],
            "schema": "v21e3r1_successor_development_factorial_inference_spec_v1",
            "method": "one_sided_observed_se_max_t_paired_case_cluster_bootstrap_v1",
            "bootstrap_samples": 9999,
            "bootstrap_seed": 2026082301,
            "familywise_alpha": 0.05,
        },
        "rows": plan_rows,
        "selection_entropy_release": "PROHIBITED",
        "selection_cases_materialized": False,
        "confirmation_materialization": "PROHIBITED",
        "formal_materialization": "PROHIBITED",
        "implementation_independence": False,
        "scientific_independence": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_study_authorized": False,
        "scientific_claim_authorized": False,
        "ijoc_submission_status": "IJOC_HOLD",
    }
    plan_raw = _canonical(plan) + b"\n"
    plan_binding = _write_bytes(
        root, "successor-factorial/factorial.plan.json", plan_raw
    )
    aggregate_value = {
        "schema": "v21e3r1_successor_development_factorial_aggregate_v2",
        "status": "PASS_EXACT_108_SUCCESSOR_DEVELOPMENT_FACTORIAL_ENGINEERING_ONLY",
        "phase": "development",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "plan_sha256": plan_binding["sha256"],
        "row_count": 108,
        "rows": _factorial_aggregate_rows(plan_rows),
        "development_promotion_evaluated": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_study_authorized": False,
        "scientific_claim_authorized": False,
        "ijoc_submission_status": "IJOC_HOLD",
    }
    aggregate = _write_bytes(
        root,
        "successor-factorial/factorial.aggregate.json",
        _canonical(aggregate_value) + b"\n",
    )
    matrix_receipt_core = {
        "schema": "v21e3r1_successor_development_factorial_receipt_v2",
        "status": "PASS_EXACT_108_SUCCESSOR_DEVELOPMENT_FACTORIAL_ENGINEERING_ONLY",
        "phase": "development",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "completed_rows": 108,
        "expected_rows": 108,
        "plan_sha256": plan_binding["sha256"],
        "aggregate_sha256": aggregate["sha256"],
        "parent_v7_diagnostic_plan_sha256": plan[
            "parent_v7_diagnostic_plan_sha256"
        ],
        "parent_v7_source_snapshot_sha256": identity[
            "development_source_sha256"
        ],
        "study_id": identity["study_id"],
        "candidate_id": identity["candidate_id"],
        "successor_source_sha256": identity["successor_source_sha256"],
        "successor_config_sha256": identity["successor_config_sha256"],
        "source_freeze_receipt_sha256": source_freeze["sha256"],
        "source_manifest_sha256": source_value["source_manifest_sha256"],
        "study_metric_spec_sha256": identity["study_metric_spec_sha256"],
        "simultaneous_inference_spec_sha256": identity[
            "simultaneous_inference_spec_sha256"
        ],
        "inference_spec_sha256": inference["sha256"],
        "development_promotion_evaluated": False,
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
    matrix_receipt = dict(matrix_receipt_core)
    matrix_receipt["receipt_payload_sha256"] = hashlib.sha256(
        _canonical(matrix_receipt_core)
    ).hexdigest()
    matrix_receipt_binding = _write_bytes(
        root,
        "successor-factorial/factorial.receipt.json",
        _canonical(matrix_receipt) + b"\n",
    )
    hypotheses = (
        ("MOKP:BOTH_MINUS_LEGACY:EAUC", "MOKP", "exact_per_evaluation_left_continuous_hv_auc", "SUPERIORITY", 0.005),
        ("MOKP:ANCHOR_MAIN_EFFECT:EAUC", "MOKP", "exact_per_evaluation_left_continuous_hv_auc", "SUPERIORITY", 0.0),
        ("MOKP:NOVELTY_MAIN_EFFECT:EAUC", "MOKP", "exact_per_evaluation_left_continuous_hv_auc", "NONINFERIORITY", -0.005),
        ("MOKP:NOVELTY_MAIN_EFFECT:CACHE_HIT_RATE_REDUCTION", "MOKP", "cache_hit_rate_per_attempt", "SUPERIORITY", 0.1),
        ("MOTSP:ANCHOR_MINUS_LEGACY:EAUC", "MOTSP", "exact_per_evaluation_left_continuous_hv_auc", "SUPERIORITY", 0.0),
    )
    cells = []
    for index, (hypothesis_id, family, metric, role, threshold) in enumerate(hypotheses):
        cell_passed = False if zero_standard_error else passed or index != 0
        lower_bound = threshold + (0.01 if cell_passed else 0.0)
        cells.append(
            {
                "hypothesis_id": hypothesis_id,
                "family": family,
                "metric": metric,
                "role": role,
                "threshold": threshold,
                "case_count": 6,
                "seed_count_per_case_arm": 3,
                "observed_mean": lower_bound,
                "standard_error": 0.0 if zero_standard_error and index == 0 else 0.01,
                "median": lower_bound,
                "wins_above_threshold": 6 if cell_passed else 0,
                "ties_at_threshold": 0 if cell_passed else 6,
                "losses_below_threshold": 0,
                "simultaneous_lower_bound": (
                    None if zero_standard_error else lower_bound
                ),
                "gate_passed": cell_passed,
            }
        )
    status = (
        "HOLD_SUCCESSOR_DEVELOPMENT_PROMOTION_ZERO_STANDARD_ERROR"
        if zero_standard_error
        else "PASS_SUCCESSOR_DEVELOPMENT_PROMOTION_ONLY"
        if passed
        else "HOLD_SUCCESSOR_DEVELOPMENT_PROMOTION_THRESHOLDS_NOT_MET"
    )
    promotion_core = {
        "schema": "v21e3r1_successor_development_factorial_evaluation_receipt_v2",
        "status": status,
        "phase": "development",
        "promotion_scope": "SUCCESSOR_DEVELOPMENT_PROMOTION_ONLY_HASH_BOUND_PRODUCER_RECEIPT_NO_PROSPECTIVE_108_ROW_RECOMPUTATION_NO_SCIENTIFIC_CLAIM",
        "study_id": identity["study_id"],
        "candidate_id": identity["candidate_id"],
        "successor_source_sha256": identity["successor_source_sha256"],
        "successor_config_sha256": identity["successor_config_sha256"],
        "source_freeze_receipt_sha256": source_freeze["sha256"],
        "source_manifest_sha256": source_value["source_manifest_sha256"],
        "study_metric_spec_sha256": identity["study_metric_spec_sha256"],
        "simultaneous_inference_spec_sha256": identity[
            "simultaneous_inference_spec_sha256"
        ],
        "matrix_directory": "successor-factorial",
        "matrix_plan_sha256": plan_binding["sha256"],
        "matrix_receipt_sha256": matrix_receipt_binding["sha256"],
        "row_evidence_replay_sha256": "9" * 64,
        "inference_spec_sha256": inference["sha256"],
        "method": "one_sided_observed_se_max_t_paired_case_cluster_bootstrap_v1",
        "familywise_alpha": 0.05,
        "bootstrap_samples": 9999,
        "bootstrap_seed": 2026082301,
        "rng_protocol": "SHA256_COUNTER_U64_REJECTION_V1",
        "rng_domain": "v21e3r1-successor-development-factorial-bootstrap-v1",
        "quantile_convention": "CEIL((1-ALPHA)*(B+1))_ORDER_STATISTIC",
        "cluster_unit": "PAIRED_CASE",
        "seed_aggregation": "MEAN_WITHIN_CASE_ARM",
        "familywise_scope": "JOINT_ACROSS_ALL_FIVE_DEVELOPMENT_PROMOTION_HYPOTHESES",
        "critical_value": None if zero_standard_error else 0.0,
        "bootstrap_maxima_sha256": None if zero_standard_error else "a" * 64,
        "matrix_row_count": 108,
        "expected_matrix_row_count": 108,
        "hypothesis_order": [item[0] for item in hypotheses],
        "cells": cells,
        "development_promotion_gate_passed": passed,
        "gate_reasons": (
            ["zero_standard_error:MOKP:BOTH_MINUS_LEGACY:EAUC"]
            if zero_standard_error
            else []
            if passed
            else [
                "simultaneous_lower_bound_not_above_threshold:"
                "MOKP:BOTH_MINUS_LEGACY:EAUC"
            ]
        ),
        "zero_standard_error_hypotheses": (
            ["MOKP:BOTH_MINUS_LEGACY:EAUC"] if zero_standard_error else []
        ),
        "selection_confirmation_evaluator_reused": False,
        "selection_confirmation_evaluator_reuse_reason": "INCOMPATIBLE_ASYMMETRIC_4_ARM_MOKP_2_ARM_MOTSP_AND_MIXED_METRICS",
        "selection_cases_materialized": False,
        "confirmation_cases_materialized": False,
        "formal_cases_materialized": False,
        "algorithm_execution_independence": False,
        "statistics_implementation_independence": False,
        "producer_independence": False,
        "custody_independence": False,
        "scientific_independence": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_study_authorized": False,
        "scientific_claim_authorized": False,
        "ijoc_submission_status": "IJOC_HOLD",
    }
    promotion = dict(promotion_core)
    promotion["receipt_payload_sha256"] = hashlib.sha256(
        _canonical(promotion_core)
    ).hexdigest()
    return _write_bytes(
        root,
        "successor-development-promotion.receipt.json",
        _canonical(promotion) + b"\n",
    )


def _mutate_reseal_factorial_design(
    root: Path,
    bindings: dict[str, dict[str, str]],
    *,
    mutation: str,
) -> None:
    matrix = root / "successor-factorial"
    plan_path = matrix / "factorial.plan.json"
    aggregate_path = matrix / "factorial.aggregate.json"
    receipt_path = matrix / "factorial.receipt.json"

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if mutation == "parent_plan_sha256":
        plan["parent_v7_diagnostic_plan_sha256"] = "0" * 64
    elif mutation == "row_policy":
        plan["rows"][0]["post_initialization_search_policy"] = "attacker_policy_v1"
    elif mutation == "input_binding":
        plan["input_binding"]["manifest_sha256"][
            "ijoc_submission_v21e3/development_partitions_v1/case_manifest.json"
        ] = "0" * 64
    elif mutation == "aggregate_schema":
        aggregate["schema"] = "attacker_self_consistent_factorial_aggregate_v1"
    else:
        raise AssertionError(f"unsupported factorial fixture mutation: {mutation}")

    plan_raw = _canonical(plan) + b"\n"
    plan_path.write_bytes(plan_raw)
    plan_sha256 = hashlib.sha256(plan_raw).hexdigest()
    aggregate["plan_sha256"] = plan_sha256
    aggregate_raw = _canonical(aggregate) + b"\n"
    aggregate_path.write_bytes(aggregate_raw)
    aggregate_sha256 = hashlib.sha256(aggregate_raw).hexdigest()

    matrix_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    matrix_receipt["plan_sha256"] = plan_sha256
    matrix_receipt["aggregate_sha256"] = aggregate_sha256
    matrix_receipt["parent_v7_diagnostic_plan_sha256"] = plan[
        "parent_v7_diagnostic_plan_sha256"
    ]
    matrix_core = dict(matrix_receipt)
    matrix_core.pop("receipt_payload_sha256")
    matrix_receipt["receipt_payload_sha256"] = hashlib.sha256(
        _canonical(matrix_core)
    ).hexdigest()
    matrix_raw = _canonical(matrix_receipt) + b"\n"
    receipt_path.write_bytes(matrix_raw)
    matrix_sha256 = hashlib.sha256(matrix_raw).hexdigest()

    promotion_binding = bindings["successor_development_promotion"]
    promotion_path = root / promotion_binding["path"]
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    promotion["matrix_plan_sha256"] = plan_sha256
    promotion["matrix_receipt_sha256"] = matrix_sha256
    promotion_core = dict(promotion)
    promotion_core.pop("receipt_payload_sha256")
    promotion["receipt_payload_sha256"] = hashlib.sha256(
        _canonical(promotion_core)
    ).hexdigest()
    promotion_raw = _canonical(promotion) + b"\n"
    promotion_path.write_bytes(promotion_raw)
    promotion_binding["sha256"] = hashlib.sha256(promotion_raw).hexdigest()


def _write_gate_spec(
    root: Path,
    *,
    requested: str,
    bindings: dict[str, dict[str, str]],
    identity: dict[str, str],
) -> Path:
    path = root / "gate-spec.json"
    path.write_bytes(
        _canonical(
            {
                "schema": "v21e3r1_prospective_authorization_gate_spec_v3",
                "requested_authorization": requested,
                "identity": identity,
                "bindings": bindings,
            }
        )
    )
    return path


def _write_phase_result(
    root: Path,
    *,
    phase: str,
    identity: dict[str, str],
    source_freeze_sha256: str,
    selection_receipt_sha256: str | None = None,
    external_replay_sha256: str | None = None,
    custody_receipt_sha256: str | None = None,
) -> dict[str, str]:
    assert phase in {"selection", "confirmation"}
    status = "PASS_SELECTION" if phase == "selection" else "PASS_CONFIRMATION"
    selection_binding = None
    controls = None
    if phase == "confirmation":
        assert selection_receipt_sha256 is not None
        assert external_replay_sha256 is not None
        assert custody_receipt_sha256 is not None
        selection_binding = {
            "selection_receipt_sha256": selection_receipt_sha256,
            "selection_status": "PASS_SELECTION",
            "selected_candidate": identity["candidate_id"],
        }
        controls = {
            "external_producer": True,
            "external_producer_receipt_sha256": external_replay_sha256,
            "independent_custody": True,
            "custody_receipt_sha256": custody_receipt_sha256,
            "statistics_source_sha256": "f" * 64,
        }
    statistics_source = root / "successor-freeze/recompute-simultaneous-bounds.py"
    statistics_source_sha256 = hashlib.sha256(statistics_source.read_bytes()).hexdigest()
    if controls is not None:
        controls["statistics_source_sha256"] = statistics_source_sha256
    core = {
        "schema": "v21e3r1_independent_simultaneous_inference_receipt_v1",
        "status": status,
        "phase": phase,
        "study_id": identity["study_id"],
        "input_sha256": "0" * 64,
        "study_freeze_sha256": source_freeze_sha256,
        "phase_manifest_sha256": "1" * 64,
        "matrix_receipt_sha256": "2" * 64,
        "source_root_sha256": identity["successor_source_sha256"],
        "metric_spec_sha256": identity["study_metric_spec_sha256"],
        "decision_spec_sha256": identity["simultaneous_inference_spec_sha256"],
        "source_sha256": statistics_source_sha256,
        "effect_direction": "larger_is_better",
        "families": ["MOKP", "MOTSP"],
        "candidate_order": ["C1", "C2", "C3"],
        "case_count_by_family": {"MOKP": 2, "MOTSP": 2},
        "matrix_row_count": 36,
        "expected_matrix_row_count": 36,
        "seeds": [31051, 31057, 31059],
        "thresholds": {"adjacent": 0.0, "primary": 0.0},
        "hypothesis_order": ["MOKP/C1/primary", "MOTSP/C1/primary"],
        "cells": [{"hypothesis_id": "MOKP/C1/primary"}],
        "inference": {"familywise_scope": "JOINT_ACROSS_BOTH_FAMILIES"},
        "simultaneous_coverage_certified": True,
        "selected_candidate": identity["candidate_id"],
        "reached_candidates": [identity["candidate_id"]],
        "not_reached_candidates": [],
        "blocked_candidate": None,
        "gate_reasons": [],
        "selection_binding": selection_binding,
        "confirmation_control_bindings": controls,
        "confirmation_control_bindings_validated": phase == "confirmation",
        "confirmation_control_bindings_scope": (
            "INPUT_DECLARATIONS_AND_HASH_BINDINGS_ONLY_NOT_AUTHENTICATION"
            if phase == "confirmation"
            else None
        ),
        "statistics_implementation_independent_from_mo_nco": True,
        "external_independence_claim_authorized": False,
        "scientific_independence": False,
        "formal_authority": False,
    }
    receipt = dict(core)
    receipt["receipt_payload_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    return _write_json(root, f"{phase}-result.json", receipt)


def _run(root: Path, spec: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPT),
            "--gate-spec",
            str(spec),
            "--evidence-root",
            str(root),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def _binding_for(root: Path, path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def test_v3_claim_only_v1_common_receipts_cannot_authorize_selection(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(
        tmp_path,
        external_status="PASS_EXTERNAL_ALGORITHM_REPLAY",
        independent_producer=True,
        independent_custody=True,
        implementation_code_disjoint=True,
        algorithm_execution_independence=True,
    )
    bindings["successor_development_promotion"] = _write_development_promotion(
        tmp_path,
        identity=identity,
        source_freeze=bindings["successor_source_freeze"],
        passed=True,
    )
    spec = tmp_path / "gate-spec-v3.json"
    spec.write_bytes(
        _canonical(
            {
                "schema": "v21e3r1_prospective_authorization_gate_spec_v3",
                "requested_authorization": "selection",
                "identity": identity,
                "bindings": bindings,
            }
        )
    )
    output = tmp_path / "selection-v3.authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 2, completed.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["schema"] == "v21e3r1_prospective_authorization_receipt_v3"
    assert receipt["gates"]["successor_development_promotion_gate_passed"] is True
    assert receipt["gates"]["external_family_native_strong_baseline_each_family"] is False
    assert receipt["gates"]["external_algorithm_replay"] is False
    assert receipt["selection_authorized"] is False
    assert receipt["case_generation_performed"] is False
    assert receipt["generated_case_count"] == 0


@pytest.mark.parametrize(
    ("binding_id", "field"),
    (("baseline_registry", "study_id"), ("external_algorithm_replay", "candidate_id")),
)
def test_claim_only_v1_common_receipt_identity_drift_is_integrity_error(
    tmp_path: Path, binding_id: str, field: str
) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    _rewrite_bound_json(
        tmp_path,
        bindings[binding_id],
        lambda receipt: receipt.__setitem__(field, "attacker-identity"),
    )
    spec = _write_gate_spec(
        tmp_path, requested="selection", bindings=bindings, identity=identity
    )
    output = tmp_path / f"{binding_id}-identity-drift.authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "identity" in completed.stderr
    assert "HOLD_INTEGRITY_ERROR" in completed.stderr


def test_v3_valid_hold_promotion_yields_authorization_hold_not_integrity_error(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(
        tmp_path,
        external_status="PASS_EXTERNAL_ALGORITHM_REPLAY",
        independent_producer=True,
        independent_custody=True,
        implementation_code_disjoint=True,
        algorithm_execution_independence=True,
        promotion_passed=False,
    )
    spec = _write_gate_spec(
        tmp_path, requested="selection", bindings=bindings, identity=identity
    )
    output = tmp_path / "selection-promotion-hold.authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 2, completed.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "HOLD_PROSPECTIVE_AUTHORIZATION_PREREQUISITES"
    assert receipt["gates"]["successor_development_promotion_gate_passed"] is False
    assert "successor_development_promotion_gate_passed" in receipt["hold_reasons"]
    assert receipt["selection_authorized"] is False
    assert receipt["case_generation_performed"] is False


@pytest.mark.parametrize(
    "field",
    [
        "successor_source_sha256",
        "successor_config_sha256",
        "source_freeze_receipt_sha256",
    ],
)
def test_v3_promotion_cross_identity_drift_is_an_integrity_error(
    tmp_path: Path, field: str
) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    promotion_binding = bindings["successor_development_promotion"]
    promotion_path = tmp_path / promotion_binding["path"]
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    promotion[field] = "e" * 64
    promotion_core = dict(promotion)
    promotion_core.pop("receipt_payload_sha256")
    promotion["receipt_payload_sha256"] = hashlib.sha256(
        _canonical(promotion_core)
    ).hexdigest()
    raw = _canonical(promotion) + b"\n"
    promotion_path.write_bytes(raw)
    promotion_binding["sha256"] = hashlib.sha256(raw).hexdigest()
    spec = _write_gate_spec(
        tmp_path, requested="selection", bindings=bindings, identity=identity
    )
    output = tmp_path / "promotion-identity-drift.authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 3
    assert not output.exists()
    assert field in completed.stderr
    assert "HOLD_INTEGRITY_ERROR" in completed.stderr


def test_v3_promotion_payload_tampering_is_an_integrity_error(tmp_path: Path) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    promotion_binding = bindings["successor_development_promotion"]
    promotion_path = tmp_path / promotion_binding["path"]
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    promotion["matrix_row_count"] = 107
    raw = _canonical(promotion) + b"\n"
    promotion_path.write_bytes(raw)
    promotion_binding["sha256"] = hashlib.sha256(raw).hexdigest()
    spec = _write_gate_spec(
        tmp_path, requested="selection", bindings=bindings, identity=identity
    )
    output = tmp_path / "promotion-payload-tamper.authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "payload digest drifted" in completed.stderr


@pytest.mark.parametrize(
    "mutation",
    ["parent_plan_sha256", "row_policy", "input_binding", "aggregate_schema"],
)
def test_v3_rejects_hash_consistent_wrong_successor_factorial_design(
    tmp_path: Path, mutation: str
) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    _mutate_reseal_factorial_design(tmp_path, bindings, mutation=mutation)
    spec = _write_gate_spec(
        tmp_path, requested="selection", bindings=bindings, identity=identity
    )
    output = tmp_path / f"wrong-factorial-{mutation}.authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 3, completed.stderr
    assert not output.exists()
    assert "HOLD_INTEGRITY_ERROR" in completed.stderr


def test_real_boundary_freezer_receipts_are_consumed_and_hold_selection(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    completed_freeze = subprocess.run(
        [
            sys.executable,
            "-I",
            str(BOUNDARY_FREEZER),
            "--repository-root",
            str(Path(__file__).resolve().parents[1]),
            "--output-directory",
            str(tmp_path / "prospective-boundaries"),
            "--study-id",
            identity["study_id"],
            "--candidate-id",
            identity["candidate_id"],
            "--successor-source-sha256",
            identity["successor_source_sha256"],
            "--successor-config-sha256",
            identity["successor_config_sha256"],
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed_freeze.returncode == 0, completed_freeze.stderr
    frozen = tmp_path / "prospective-boundaries"
    bindings["baseline_registry"] = _binding_for(
        tmp_path, frozen / "baseline-registry.receipt.json"
    )
    bindings["external_algorithm_replay"] = _binding_for(
        tmp_path, frozen / "external-algorithm-replay.receipt.json"
    )
    frozen_sim = frozen / "simultaneous-inference.spec.json"
    frozen_metric = frozen / "study.metric-spec.json"
    metric_raw = frozen_metric.read_bytes()
    metric_sha = hashlib.sha256(metric_raw).hexdigest()
    (tmp_path / "successor-freeze/study.metric-spec.json").write_bytes(metric_raw)
    identity["study_metric_spec_sha256"] = metric_sha
    bindings["simultaneous_inference_spec"] = _binding_for(tmp_path, frozen_sim)
    sim_raw = frozen_sim.read_bytes()
    sim_sha = hashlib.sha256(sim_raw).hexdigest()
    (tmp_path / "successor-freeze/simultaneous-inference.spec.json").write_bytes(sim_raw)
    identity["simultaneous_inference_spec_sha256"] = sim_sha

    def bind_sim(value: dict[str, object]) -> None:
        value["simultaneous_inference_spec_sha256"] = sim_sha
        value["study_metric_spec_sha256"] = metric_sha
        core = dict(value)
        core.pop("receipt_payload_sha256")
        value["receipt_payload_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()

    _rewrite_bound_json(
        tmp_path,
        bindings["successor_source_freeze"],
        bind_sim,
    )
    bindings["successor_development_promotion"] = _write_development_promotion(
        tmp_path,
        identity=identity,
        source_freeze=bindings["successor_source_freeze"],
        passed=True,
    )
    spec = _write_gate_spec(
        tmp_path,
        requested="selection",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 2, completed.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["gates"]["successor_source_and_config_frozen"] is True
    assert receipt["gates"]["simultaneous_inference_spec_frozen"] is True
    assert receipt["gates"]["external_family_native_strong_baseline_each_family"] is False
    assert receipt["baseline_eligible_count_by_family"] == {"MOKP": 0, "MOTSP": 0}
    assert receipt["gates"]["external_algorithm_replay"] is False
    assert receipt["gates"]["algorithm_execution_independence"] is False
    assert receipt["selection_authorized"] is False


def test_baseline_v2_resealed_pass_remains_permanently_design_only(
    tmp_path: Path,
) -> None:
    _, identity = _make_common_evidence(tmp_path)
    frozen = tmp_path / "baseline-v2-design-only"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(BOUNDARY_FREEZER),
            "--repository-root",
            str(Path(__file__).resolve().parents[1]),
            "--output-directory",
            str(frozen),
            "--study-id",
            identity["study_id"],
            "--candidate-id",
            identity["candidate_id"],
            "--successor-source-sha256",
            identity["successor_source_sha256"],
            "--successor-config-sha256",
            identity["successor_config_sha256"],
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    receipt_path = frozen / "baseline-registry.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "PASS_EXTERNAL_FAMILY_NATIVE_STRONG_BASELINES"
    for family in receipt["families"]:
        family["external_family_native_strong_baseline_count"] = len(
            family["baselines"]
        )
        for baseline in family["baselines"]:
            baseline["classification"] = "EXTERNAL_FAMILY_NATIVE_STRONG_BASELINE"
            baseline["external"] = True
            baseline["family_native"] = True
            baseline["strong"] = True
            baseline["external_family_native_strong_baseline_eligible"] = True
    core = dict(receipt)
    core.pop("receipt_payload_sha256")
    receipt["receipt_payload_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()

    gate, counts = prospective_evaluator._validate_baselines(
        receipt, identity, receipt_path
    )

    assert gate is False
    assert counts == {"MOKP": 0, "MOTSP": 0}


def test_source_candidate_absent_from_simultaneous_menu_cannot_clear_gate(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    path = tmp_path / bindings["simultaneous_inference_spec"]["path"]
    value = json.loads(path.read_text(encoding="utf-8"))
    successor_id = "V21E3R1_SUCCESSOR_SEARCH_NOVELTY_V1"
    value["candidate_id"] = successor_id
    core = dict(value)
    core.pop("receipt_payload_sha256")
    value["receipt_payload_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    raw = _canonical(value)
    path.write_bytes(raw)
    candidate_identity = dict(identity)
    candidate_identity["candidate_id"] = successor_id
    candidate_identity["simultaneous_inference_spec_sha256"] = hashlib.sha256(
        raw
    ).hexdigest()

    gate = prospective_evaluator._validate_simultaneous_spec(
        value,
        candidate_identity,
        candidate_identity["simultaneous_inference_spec_sha256"],
        path,
    )

    assert gate is False


def test_distinct_successor_candidate_produces_authorization_hold_reason(
    tmp_path: Path,
) -> None:
    successor_id = "V21E3R1_SUCCESSOR_SEARCH_NOVELTY_V1"
    bindings, identity = _make_common_evidence(
        tmp_path, candidate_id=successor_id
    )
    spec = _write_gate_spec(
        tmp_path,
        requested="selection",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "distinct-successor.authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 2, completed.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["gates"]["simultaneous_inference_spec_frozen"] is False
    assert "simultaneous_inference_spec_frozen" in receipt["hold_reasons"]
    assert receipt["selection_authorized"] is False


def test_future_v3_contracts_are_machine_checked_and_always_hold_locally(
    tmp_path: Path,
) -> None:
    _, identity = _make_common_evidence(tmp_path)
    frozen = tmp_path / "future-v3-contracts"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(BOUNDARY_FREEZER),
            "--repository-root",
            str(Path(__file__).resolve().parents[1]),
            "--output-directory",
            str(frozen),
            "--study-id",
            identity["study_id"],
            "--candidate-id",
            identity["candidate_id"],
            "--successor-source-sha256",
            identity["successor_source_sha256"],
            "--successor-config-sha256",
            identity["successor_config_sha256"],
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    external = json.loads(
        (frozen / "external-replay-v3.evidence-contract.json").read_text(
            encoding="utf-8"
        )
    )
    phase = json.loads(
        (frozen / "path-bound-phase-v3.evidence-contract.json").read_text(
            encoding="utf-8"
        )
    )
    contract_identity = dict(identity)
    contract_identity["study_metric_spec_sha256"] = external[
        "study_metric_spec_sha256"
    ]

    assert (
        prospective_evaluator._validate_external_replay_v3_contract(
            external, contract_identity
        )
        is False
    )
    assert (
        prospective_evaluator._validate_path_bound_phase_v3_contract(
            phase, contract_identity, phase="selection"
        )
        is False
    )
    assert (
        prospective_evaluator._validate_path_bound_phase_v3_contract(
            phase, contract_identity, phase="confirmation"
        )
        is False
    )
    external["external_producer_present"] = True
    external_core = dict(external)
    external_core.pop("receipt_payload_sha256")
    external["receipt_payload_sha256"] = hashlib.sha256(
        _canonical(external_core)
    ).hexdigest()
    with pytest.raises(
        prospective_evaluator.AuthorizationError,
        match="cannot assert external producer or custody authority",
    ):
        prospective_evaluator._validate_external_replay_v3_contract(
            external, contract_identity
        )


def test_design_only_external_v2_resealed_independence_true_is_integrity_error(
    tmp_path: Path,
) -> None:
    _, identity = _make_common_evidence(tmp_path)
    completed_freeze = subprocess.run(
        [
            sys.executable,
            "-I",
            str(BOUNDARY_FREEZER),
            "--repository-root",
            str(Path(__file__).resolve().parents[1]),
            "--output-directory",
            str(tmp_path / "prospective-boundaries"),
            "--study-id",
            identity["study_id"],
            "--candidate-id",
            identity["candidate_id"],
            "--successor-source-sha256",
            identity["successor_source_sha256"],
            "--successor-config-sha256",
            identity["successor_config_sha256"],
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed_freeze.returncode == 0, completed_freeze.stderr
    receipt_path = (
        tmp_path / "prospective-boundaries" / "external-algorithm-replay.receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    identity["study_metric_spec_sha256"] = receipt["study_metric_spec_sha256"]
    receipt["independent_producer"] = True
    core = dict(receipt)
    core.pop("receipt_payload_sha256")
    receipt["receipt_payload_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    receipt_path.write_bytes(_canonical(receipt))

    with pytest.raises(
        prospective_evaluator.AuthorizationError,
        match="design-only external replay v2 cannot set independent_producer=true",
    ):
        prospective_evaluator._validate_external_replay(receipt, identity, receipt_path)


def test_selection_holds_when_bound_external_algorithm_replay_is_not_pass(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    spec = _write_gate_spec(
        tmp_path,
        requested="selection",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 2, completed.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "HOLD_PROSPECTIVE_AUTHORIZATION_PREREQUISITES"
    assert receipt["requested_authorization"] == "selection"
    assert receipt["gates"]["external_algorithm_replay"] is False
    assert receipt["selection_authorized"] is False
    assert receipt["confirmation_authorized"] is False
    assert receipt["formal_input_materialization_authorized"] is False
    assert receipt["case_generation_performed"] is False
    assert receipt["generated_case_count"] == 0


def test_selection_holds_when_replay_independence_flags_are_false(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(
        tmp_path,
        external_status="PASS_EXTERNAL_ALGORITHM_REPLAY",
    )
    spec = _write_gate_spec(
        tmp_path,
        requested="selection",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 2, completed.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "HOLD_PROSPECTIVE_AUTHORIZATION_PREREQUISITES"
    assert receipt["selection_authorized"] is False
    assert receipt["confirmation_authorized"] is False
    assert receipt["formal_input_materialization_authorized"] is False
    assert receipt["gates"]["independent_producer"] is False
    assert receipt["gates"]["independent_custody"] is False
    assert receipt["case_generation_performed"] is False
    assert receipt["generated_case_count"] == 0


def test_confirmation_holds_for_legacy_v1_selection_result_even_when_well_formed(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(
        tmp_path,
        external_status="PASS_EXTERNAL_ALGORITHM_REPLAY",
        independent_producer=True,
        independent_custody=True,
        implementation_code_disjoint=True,
        algorithm_execution_independence=True,
    )
    selection = _write_phase_result(
        tmp_path,
        phase="selection",
        identity=identity,
        source_freeze_sha256=bindings["successor_source_freeze"]["sha256"],
    )
    bindings["selection_result"] = selection
    spec = _write_gate_spec(
        tmp_path,
        requested="confirmation",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 2, completed.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "HOLD_PROSPECTIVE_AUTHORIZATION_PREREQUISITES"
    assert receipt["selection_authorized"] is False
    assert receipt["confirmation_authorized"] is False
    assert receipt["formal_input_materialization_authorized"] is False
    assert receipt["gates"]["selection_result"] is False
    assert "selection_result" in receipt["hold_reasons"]
    assert receipt["gates"]["independent_producer"] is False
    assert receipt["gates"]["independent_custody"] is False
    assert receipt["gates"]["implementation_code_disjoint"] is False
    assert receipt["generated_case_count"] == 0


def test_legacy_v1_selection_result_payload_damage_remains_integrity_error(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    selection = _write_phase_result(
        tmp_path,
        phase="selection",
        identity=identity,
        source_freeze_sha256=bindings["successor_source_freeze"]["sha256"],
    )
    selection_path = tmp_path / selection["path"]
    value = json.loads(selection_path.read_text(encoding="utf-8"))
    value["matrix_row_count"] = 35
    raw = _canonical(value)
    selection_path.write_bytes(raw)
    selection["sha256"] = hashlib.sha256(raw).hexdigest()
    bindings["selection_result"] = selection
    spec = _write_gate_spec(
        tmp_path, requested="confirmation", bindings=bindings, identity=identity
    )
    output = tmp_path / "damaged-v1.authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "payload digest drifted" in completed.stderr
    assert "HOLD_INTEGRITY_ERROR" in completed.stderr


def test_confirmation_rejects_resealed_selection_from_unfrozen_statistics_source(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(
        tmp_path,
        external_status="PASS_EXTERNAL_ALGORITHM_REPLAY",
        independent_producer=True,
        independent_custody=True,
        implementation_code_disjoint=True,
        algorithm_execution_independence=True,
    )
    selection = _write_phase_result(
        tmp_path,
        phase="selection",
        identity=identity,
        source_freeze_sha256=bindings["successor_source_freeze"]["sha256"],
    )
    selection_path = tmp_path / selection["path"]
    selection_value = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_value["source_sha256"] = "f" * 64
    core = dict(selection_value)
    core.pop("receipt_payload_sha256")
    selection_value["receipt_payload_sha256"] = hashlib.sha256(
        _canonical(core)
    ).hexdigest()
    selection_raw = _canonical(selection_value)
    selection_path.write_bytes(selection_raw)
    selection["sha256"] = hashlib.sha256(selection_raw).hexdigest()
    bindings["selection_result"] = selection
    spec = _write_gate_spec(
        tmp_path,
        requested="confirmation",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "statistics source does not match the frozen evaluator" in completed.stderr


def test_formal_materialization_holds_for_legacy_v1_phase_results(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(
        tmp_path,
        external_status="PASS_EXTERNAL_ALGORITHM_REPLAY",
        independent_producer=True,
        independent_custody=True,
        implementation_code_disjoint=True,
        algorithm_execution_independence=True,
    )
    selection = _write_phase_result(
        tmp_path,
        phase="selection",
        identity=identity,
        source_freeze_sha256=bindings["successor_source_freeze"]["sha256"],
    )
    confirmation = _write_phase_result(
        tmp_path,
        phase="confirmation",
        identity=identity,
        source_freeze_sha256=bindings["successor_source_freeze"]["sha256"],
        selection_receipt_sha256=selection["sha256"],
        external_replay_sha256=bindings["external_algorithm_replay"]["sha256"],
        custody_receipt_sha256=HEX["custody"],
    )
    bindings["selection_result"] = selection
    bindings["confirmation_result"] = confirmation
    spec = _write_gate_spec(
        tmp_path,
        requested="formal_input_materialization",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 2, completed.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "HOLD_PROSPECTIVE_AUTHORIZATION_PREREQUISITES"
    assert receipt["selection_authorized"] is False
    assert receipt["confirmation_authorized"] is False
    assert receipt["formal_input_materialization_authorized"] is False
    assert receipt["gates"]["selection_result"] is False
    assert receipt["gates"]["confirmation_result"] is False
    assert "selection_result" in receipt["hold_reasons"]
    assert "confirmation_result" in receipt["hold_reasons"]
    assert receipt["case_generation_performed"] is False
    assert receipt["generated_case_count"] == 0
    assert receipt["formal_study_executed"] is False


def test_selection_holds_when_one_family_lacks_an_external_family_native_strong_baseline(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(
        tmp_path,
        external_status="PASS_EXTERNAL_ALGORITHM_REPLAY",
        independent_producer=True,
        independent_custody=True,
        implementation_code_disjoint=True,
        algorithm_execution_independence=True,
    )
    _rewrite_bound_json(
        tmp_path,
        bindings["baseline_registry"],
        lambda value: value["families"][1]["baselines"][0].__setitem__("strong", False),
    )
    spec = _write_gate_spec(
        tmp_path,
        requested="selection",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 2, completed.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "HOLD_PROSPECTIVE_AUTHORIZATION_PREREQUISITES"
    assert receipt["gates"]["external_family_native_strong_baseline_each_family"] is False
    assert receipt["baseline_eligible_count_by_family"] == {"MOKP": 1, "MOTSP": 0}
    assert receipt["selection_authorized"] is False


def test_gate_spec_rejects_self_declared_authorization_boolean(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    path = tmp_path / "gate-spec.json"
    path.write_bytes(
        _canonical(
            {
                "schema": "v21e3r1_prospective_authorization_gate_spec_v3",
                "requested_authorization": "selection",
                "identity": identity,
                "bindings": bindings,
                "selection_authorized": True,
            }
        )
    )
    output = tmp_path / "authorization.receipt.json"

    completed = _run(tmp_path, path, output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "gate specification key set drifted" in completed.stderr


def test_noncanonical_bound_receipt_and_bool_integer_are_integrity_errors(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    diagnostic_path = tmp_path / bindings["exact_504_diagnostic"]["path"]
    value = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    value["completed_rows"] = True
    raw = json.dumps(value, indent=2, sort_keys=True).encode("utf-8")
    diagnostic_path.write_bytes(raw)
    bindings["exact_504_diagnostic"]["sha256"] = hashlib.sha256(raw).hexdigest()
    spec = _write_gate_spec(
        tmp_path,
        requested="selection",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "binding exact_504_diagnostic is not canonical JSON" in completed.stderr


def test_exact_type_validation_rejects_bool_as_completed_row_count(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    _rewrite_bound_json(
        tmp_path,
        bindings["exact_504_diagnostic"],
        lambda value: value.__setitem__("completed_rows", True),
    )
    spec = _write_gate_spec(
        tmp_path,
        requested="selection",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "diagnostic.completed_rows must be an exact integer" in completed.stderr


def test_noncanonical_gate_spec_is_rejected_before_evidence_evaluation(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    path = tmp_path / "gate-spec.json"
    path.write_text(
        json.dumps(
            {
                "schema": "v21e3r1_prospective_authorization_gate_spec_v3",
                "requested_authorization": "selection",
                "identity": identity,
                "bindings": bindings,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "authorization.receipt.json"

    completed = _run(tmp_path, path, output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "gate specification is not canonical JSON" in completed.stderr


def test_binding_path_escape_and_digest_tampering_fail_before_receipt_creation(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    bindings["historical_preservation"]["path"] = "../historical.json"
    spec = _write_gate_spec(
        tmp_path,
        requested="selection",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "path-escape.receipt.json"

    escaped = _run(tmp_path, spec, output)

    assert escaped.returncode == 3
    assert not output.exists()
    assert "canonical contained POSIX path" in escaped.stderr

    bindings, identity = _make_common_evidence(tmp_path)
    spec = _write_gate_spec(
        tmp_path,
        requested="selection",
        bindings=bindings,
        identity=identity,
    )
    (tmp_path / bindings["historical_preservation"]["path"]).write_bytes(b"{}")
    tampered_output = tmp_path / "tampered.receipt.json"

    tampered = _run(tmp_path, spec, tampered_output)

    assert tampered.returncode == 3
    assert not tampered_output.exists()
    assert "SHA-256 disagrees" in tampered.stderr


def test_corrected_reanalysis_internal_artifact_hash_is_verified(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    spec = _write_gate_spec(
        tmp_path,
        requested="selection",
        bindings=bindings,
        identity=identity,
    )
    (tmp_path / "operator_accounting.rows.jsonl").write_bytes(b"tampered\n")
    output = tmp_path / "authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "reanalysis.rows SHA-256 disagrees with its file" in completed.stderr


def test_formal_authorization_rejects_confirmation_not_bound_to_exact_selection(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(
        tmp_path,
        external_status="PASS_EXTERNAL_ALGORITHM_REPLAY",
        independent_producer=True,
        independent_custody=True,
        implementation_code_disjoint=True,
        algorithm_execution_independence=True,
    )
    selection = _write_phase_result(
        tmp_path,
        phase="selection",
        identity=identity,
        source_freeze_sha256=bindings["successor_source_freeze"]["sha256"],
    )
    confirmation = _write_phase_result(
        tmp_path,
        phase="confirmation",
        identity=identity,
        source_freeze_sha256=bindings["successor_source_freeze"]["sha256"],
        selection_receipt_sha256="a" * 64,
        external_replay_sha256=bindings["external_algorithm_replay"]["sha256"],
        custody_receipt_sha256=HEX["custody"],
    )
    bindings["selection_result"] = selection
    bindings["confirmation_result"] = confirmation
    spec = _write_gate_spec(
        tmp_path,
        requested="formal_input_materialization",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "legacy v1 confirmation result boundary drifted" in completed.stderr
    assert "HOLD_INTEGRITY_ERROR" in completed.stderr


def test_output_is_exclusive_and_receipt_payload_hash_is_self_verifying(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(
        tmp_path,
        external_status="PASS_EXTERNAL_ALGORITHM_REPLAY",
        independent_producer=True,
        independent_custody=True,
        implementation_code_disjoint=True,
        algorithm_execution_independence=True,
    )
    spec = _write_gate_spec(
        tmp_path,
        requested="selection",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "authorization.receipt.json"

    first = _run(tmp_path, spec, output)

    assert first.returncode == 2, first.stderr
    original = output.read_bytes()
    receipt = json.loads(original.decode("utf-8"))
    payload_sha256 = receipt.pop("receipt_payload_sha256")
    assert hashlib.sha256(_canonical(receipt)).hexdigest() == payload_sha256

    second = _run(tmp_path, spec, output)

    assert second.returncode == 3
    assert output.read_bytes() == original
    assert "exclusive create required" in second.stderr


def test_same_implementation_rejects_zero_based_plan_ordinals(tmp_path: Path) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    inner_binding = _binding_for(
        tmp_path,
        tmp_path
        / "same-implementation"
        / "inner_v1"
        / "branch_replay_coverage.receipt.json",
    )

    def make_zero_based(value: dict[str, object]) -> None:
        row_seals = value["row_seals"]
        assert type(row_seals) is list
        row_seals[0]["plan_ordinal"] = 0
        value["row_seals_sha256"] = hashlib.sha256(_canonical(row_seals)).hexdigest()

    _rewrite_bound_json(
        tmp_path,
        inner_binding,
        make_zero_based,
    )
    _reseal_recovery_bound_inner(
        tmp_path, bindings["same_implementation_coverage"]
    )
    spec = _write_gate_spec(
        tmp_path,
        requested="selection",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "not in frozen plan order" in completed.stderr


def test_same_implementation_inner_v1_receipt_is_not_a_gate_evidence_root(
    tmp_path: Path,
) -> None:
    """The prospective gate must consume the recovery-bound outer seal chain."""

    bindings, identity = _make_common_evidence(
        tmp_path, use_recovery_bound_same_implementation=False
    )
    spec = _write_gate_spec(
        tmp_path,
        requested="selection",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "inner-only.authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "recovery-bound" in completed.stderr
    assert "HOLD_INTEGRITY_ERROR" in completed.stderr


def test_recovery_bound_outer_chain_is_consumed_as_same_implementation_gate(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    spec = _write_gate_spec(
        tmp_path,
        requested="selection",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "outer-chain.authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 2, completed.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["gates"]["same_implementation_branch_coverage"] is True
    assert receipt["selection_authorized"] is False


def test_same_implementation_manifest_must_bind_development_not_successor_source(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    manifest_path = (
        tmp_path / "same-implementation/inner_v1/source.manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    successor_manifest = json.loads(
        (tmp_path / "successor-freeze/source.manifest.json").read_text(encoding="utf-8")
    )
    manifest["entries"] = successor_manifest["entries"]
    manifest["source_root_sha256"] = identity["successor_source_sha256"]
    manifest_raw = _canonical(manifest)
    manifest_path.write_bytes(manifest_raw)

    inner_binding = _binding_for(
        tmp_path,
        tmp_path
        / "same-implementation"
        / "inner_v1"
        / "branch_replay_coverage.receipt.json",
    )
    _rewrite_bound_json(
        tmp_path,
        inner_binding,
        lambda value: value.__setitem__(
            "source_manifest_sha256", hashlib.sha256(manifest_raw).hexdigest()
        ),
    )
    _reseal_recovery_bound_inner(
        tmp_path, bindings["same_implementation_coverage"]
    )
    spec = _write_gate_spec(
        tmp_path,
        requested="selection",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "source root disagrees with the gate identity" in completed.stderr


def test_source_freeze_rehashes_fixed_semantic_config_sibling(tmp_path: Path) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    (tmp_path / "successor-freeze/semantic.config.json").write_bytes(b"{}")
    spec = _write_gate_spec(
        tmp_path,
        requested="selection",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "source.semantic_config SHA-256 disagrees with its file" in completed.stderr


def test_source_freeze_rejects_resealed_unrelated_semantic_policy(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    semantic_path = tmp_path / "successor-freeze/semantic.config.json"
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    semantic["parameters"] = {"x": "y"}
    semantic_raw = _canonical(semantic)
    semantic_path.write_bytes(semantic_raw)
    semantic_sha = hashlib.sha256(semantic_raw).hexdigest()
    identity["successor_config_sha256"] = semantic_sha

    simultaneous_path = tmp_path / bindings["simultaneous_inference_spec"]["path"]
    simultaneous = json.loads(simultaneous_path.read_text(encoding="utf-8"))
    simultaneous_core = dict(simultaneous)
    simultaneous_core.pop("receipt_payload_sha256")
    simultaneous_core["successor_config_sha256"] = semantic_sha
    simultaneous = dict(simultaneous_core)
    simultaneous["receipt_payload_sha256"] = hashlib.sha256(
        _canonical(simultaneous_core)
    ).hexdigest()
    simultaneous_raw = _canonical(simultaneous)
    simultaneous_path.write_bytes(simultaneous_raw)
    simultaneous_sha = hashlib.sha256(simultaneous_raw).hexdigest()
    bindings["simultaneous_inference_spec"]["sha256"] = simultaneous_sha
    identity["simultaneous_inference_spec_sha256"] = simultaneous_sha

    def bind_semantic(value: dict[str, object]) -> None:
        value["semantic_config_sha256"] = semantic_sha
        value["simultaneous_inference_spec_sha256"] = simultaneous_sha
        core = dict(value)
        core.pop("receipt_payload_sha256")
        value["receipt_payload_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()

    _rewrite_bound_json(tmp_path, bindings["successor_source_freeze"], bind_semantic)
    _rewrite_bound_json(
        tmp_path,
        bindings["external_algorithm_replay"],
        lambda value: value.__setitem__("successor_config_sha256", semantic_sha),
    )
    bindings["successor_development_promotion"] = _write_development_promotion(
        tmp_path,
        identity=identity,
        source_freeze=bindings["successor_source_freeze"],
        passed=True,
    )
    spec = _write_gate_spec(
        tmp_path,
        requested="selection",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "semantic config policy contract drifted" in completed.stderr


def test_operator_reanalysis_spec_cannot_be_replaced_by_study_metric_identity(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    identity["operator_reanalysis_spec_sha256"] = identity["study_metric_spec_sha256"]
    spec = _write_gate_spec(
        tmp_path,
        requested="selection",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 2, completed.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["gates"]["corrected_reanalysis"] is False
    assert receipt["gates"]["successor_source_and_config_frozen"] is True
    assert receipt["selection_authorized"] is False


def _mutate_bound_simultaneous_spec(
    root: Path,
    bindings: dict[str, dict[str, str]],
    identity: dict[str, str],
    mutate: object,
) -> None:
    path = root / bindings["simultaneous_inference_spec"]["path"]
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    core = dict(value)
    core.pop("receipt_payload_sha256")
    value["receipt_payload_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    raw = _canonical(value)
    path.write_bytes(raw)
    sim_sha = hashlib.sha256(raw).hexdigest()
    bindings["simultaneous_inference_spec"]["sha256"] = sim_sha
    identity["simultaneous_inference_spec_sha256"] = sim_sha

    def bind_source(value: dict[str, object]) -> None:
        value["simultaneous_inference_spec_sha256"] = sim_sha
        receipt_core = dict(value)
        receipt_core.pop("receipt_payload_sha256")
        value["receipt_payload_sha256"] = hashlib.sha256(
            _canonical(receipt_core)
        ).hexdigest()

    _rewrite_bound_json(root, bindings["successor_source_freeze"], bind_source)
    bindings["successor_development_promotion"] = _write_development_promotion(
        root,
        identity=identity,
        source_freeze=bindings["successor_source_freeze"],
        passed=True,
    )


def test_simultaneous_spec_rejects_legacy_method_alias(tmp_path: Path) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    _mutate_bound_simultaneous_spec(
        tmp_path,
        bindings,
        identity,
        lambda value: value.__setitem__("method", "MAX_T_CASE_CLUSTER_BOOTSTRAP_V1"),
    )
    spec = _write_gate_spec(
        tmp_path,
        requested="selection",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "simultaneous method provenance drifted" in completed.stderr


def test_simultaneous_spec_rejects_zero_adjacent_practical_threshold(
    tmp_path: Path,
) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    _mutate_bound_simultaneous_spec(
        tmp_path,
        bindings,
        identity,
        lambda value: value["practical_thresholds"].__setitem__(
            "adjacent_mechanism_effect", 0.0
        ),
    )
    spec = _write_gate_spec(
        tmp_path,
        requested="selection",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / "authorization.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 2, completed.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["gates"]["simultaneous_inference_spec_frozen"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (("bootstrap_samples", 9999), ("bootstrap_seed", 20260824)),
)
def test_simultaneous_spec_rejects_resealed_nonexact_bootstrap_contract(
    tmp_path: Path, field: str, value: int
) -> None:
    bindings, identity = _make_common_evidence(tmp_path)
    _mutate_bound_simultaneous_spec(
        tmp_path,
        bindings,
        identity,
        lambda spec: spec.__setitem__(field, value),
    )
    spec = _write_gate_spec(
        tmp_path,
        requested="selection",
        bindings=bindings,
        identity=identity,
    )
    output = tmp_path / f"authorization-{field}.receipt.json"

    completed = _run(tmp_path, spec, output)

    assert completed.returncode == 3
    assert not output.exists()
    assert field in completed.stderr
    assert "provenance drifted" in completed.stderr

