from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "freeze_v21e3r1_successor_source.py"
)
STUDY_ID = "v21e3r1-prospective-test"
CANDIDATE_ID = "C1"
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
    "promoted_arm_by_family": {
        "MOKP": "MOKP_BOTH",
        "MOTSP": "MOTSP_ANCHOR",
    },
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
EXPECTED_SOURCE_PATHS = [
    "ijoc_submission_v21e3r1/development/V21E3R1_SUCCESSOR_DEVELOPMENT_FACTORIAL_INFERENCE_V1.json",
    "ijoc_submission_v21e3r1/development/V21E3R1_SUCCESSOR_DEVELOPMENT_FACTORIAL_RUNBOOK_V1.md",
    "ijoc_submission_v21e3r1/development/V21E3R1_V8_STRICT_EXECUTION_RUNBOOK_2026-08-23.md",
    "ijoc_submission_v21e3r1/scripts/runner.py",
    "independent_reproduction/recompute_v21e3r1_metrics.py",
    "independent_reproduction/recompute_v21e3r1_simultaneous_bounds.py",
    "independent_reproduction/recompute_v21e3r1_successor_metrics.py",
    "independent_reproduction/replay.py",
    "mo_nco/core.py",
    "mo_nco/nested/solver.py",
    "mo_nco/pareto_ijoc_analysis.py",
    "pyproject.toml",
    "tests/test_v21e3r1_independent_simultaneous_inference.py",
    "tests/test_v21e3r1_successor_metric.py",
]
RECEIPT_KEYS = {
    "schema",
    "status",
    "study_id",
    "candidate_id",
    "parent_development_source_sha256",
    "source_snapshot_sha256",
    "source_manifest_sha256",
    "semantic_config_sha256",
    "study_metric_spec_sha256",
    "simultaneous_inference_spec_sha256",
    "source_entry_count",
    "source_total_bytes",
    "all_source_files_verified",
    "source_frozen",
    "selection_cases_materialized",
    "confirmation_cases_materialized",
    "formal_cases_materialized",
    "source_archive_materialized",
    "source_archive_path",
    "source_archive_sha256",
    "source_archive_scope",
    "implementation_independence",
    "scientific_independence",
    "selection_authorized",
    "confirmation_authorized",
    "formal_study_authorized",
    "scientific_claim_authorized",
    "public_redistribution_authorized",
    "ijoc_submission_status",
    "receipt_payload_sha256",
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _payload_receipt(core: dict[str, object]) -> dict[str, object]:
    value = dict(core)
    value["receipt_payload_sha256"] = _sha256(_canonical(core))
    return value


def _write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _write_canonical(path: Path, value: object) -> str:
    raw = _canonical(value)
    _write_bytes(path, raw)
    return _sha256(raw)


def _write_parent_snapshot(project: Path) -> dict[str, object]:
    parent = project / "custody" / "parent-development-snapshot"
    entries = [
        {
            "path": f"mo_nco/historical/module_{index:03d}.py",
            "bytes": len(raw := f"HISTORICAL = {index}\n".encode("ascii")),
            "sha256": _sha256(raw),
        }
        for index in range(170)
    ]
    source_root = _sha256(_canonical(entries))
    manifest = {
        "schema": "v21e3r1_diagnostic_source_manifest_v1",
        "entry_count": 170,
        "hash_rule": "sha256(canonical_json(sorted_entries))",
        "source_snapshot_sha256": source_root,
        "entries": entries,
    }
    manifest_raw = _canonical(manifest)
    plan_raw = (
        json.dumps(
            {
                "schema": "v21e3r1_exposed_development_diagnostic_plan_v2",
                "source_manifest": manifest,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(
        archive_buffer, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for index, entry in enumerate(entries):
            info = zipfile.ZipInfo(entry["path"], (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(
                info,
                f"HISTORICAL = {index}\n".encode("ascii"),
                compress_type=zipfile.ZIP_STORED,
            )
    archive_raw = archive_buffer.getvalue()
    _write_bytes(parent / "development-diagnostic.plan.json", plan_raw)
    _write_bytes(parent / "development-source.manifest.json", manifest_raw)
    _write_bytes(parent / "development-source.zip", archive_raw)
    receipt_core = {
        "schema": "v21e3r1_development_source_snapshot_receipt_v1",
        "status": "PASS_DEVELOPMENT_EXECUTION_SOURCE_SNAPSHOT_ENGINEERING_ONLY",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "diagnostic_plan_schema": "v21e3r1_exposed_development_diagnostic_plan_v2",
        "diagnostic_plan_copy_path": "development-diagnostic.plan.json",
        "diagnostic_plan_sha256": _sha256(plan_raw),
        "source_manifest_schema": "v21e3r1_diagnostic_source_manifest_v1",
        "source_manifest_path": "development-source.manifest.json",
        "source_manifest_sha256": _sha256(manifest_raw),
        "source_snapshot_sha256": source_root,
        "source_entry_count": 170,
        "source_total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "all_source_files_verified": True,
        "source_archive_materialized": True,
        "source_archive_path": "development-source.zip",
        "source_archive_sha256": _sha256(archive_raw),
        "source_archive_scope": (
            "PLAN_BOUND_DEVELOPMENT_EXECUTION_SOURCE_ONLY_INTERNAL_CUSTODY_"
            "NO_REDISTRIBUTION_AUTHORITY"
        ),
        "development_execution_replayed": False,
        "original_development_diagnostic_artifacts_modified": False,
        "selection_cases_materialized": False,
        "confirmation_cases_materialized": False,
        "formal_cases_materialized": False,
        "implementation_independence": False,
        "scientific_independence": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_study_authorized": False,
        "scientific_claim_authorized": False,
        "public_redistribution_authorized": False,
        "ijoc_submission_status": "IJOC_HOLD",
    }
    receipt = _payload_receipt(receipt_core)
    receipt_path = parent / "development-source.snapshot.receipt.json"
    _write_bytes(receipt_path, _canonical(receipt))
    return {"receipt": receipt_path, "source_root": source_root}


def _source_entries(project: Path) -> list[dict[str, object]]:
    entries = []
    for relative in EXPECTED_SOURCE_PATHS:
        raw = (project / Path(*relative.split("/"))).read_bytes()
        entries.append({"path": relative, "bytes": len(raw), "sha256": _sha256(raw)})
    return entries


def _make_project(project: Path) -> dict[str, Path | str]:
    sources = {
        "ijoc_submission_v21e3r1/development/V21E3R1_SUCCESSOR_DEVELOPMENT_FACTORIAL_INFERENCE_V1.json": (
            b'{"schema":"v21e3r1_successor_development_factorial_inference_spec_v1"}\n'
        ),
        "ijoc_submission_v21e3r1/development/V21E3R1_SUCCESSOR_DEVELOPMENT_FACTORIAL_RUNBOOK_V1.md": (
            b"# Successor factorial runbook\n"
        ),
        "ijoc_submission_v21e3r1/development/V21E3R1_V8_STRICT_EXECUTION_RUNBOOK_2026-08-23.md": (
            b"# Strict V8 execution runbook\n"
        ),
        "mo_nco/core.py": b"VALUE = 1\n",
        "mo_nco/nested/solver.py": b"def solve():\n    return 1\n",
        "ijoc_submission_v21e3r1/scripts/runner.py": b"def main():\n    return 0\n",
        "independent_reproduction/recompute_v21e3r1_successor_metrics.py": (
            b"def recompute():\n    return 0.0\n"
        ),
        "independent_reproduction/recompute_v21e3r1_metrics.py": (
            b"def recompute():\n    return 0.0\n"
        ),
        "independent_reproduction/recompute_v21e3r1_simultaneous_bounds.py": (
            b"def max_t():\n    return 0.0\n"
        ),
        "independent_reproduction/replay.py": b"def replay():\n    return True\n",
        "mo_nco/pareto_ijoc_analysis.py": b"def eauc():\n    return 0.0\n",
        "pyproject.toml": b"[project]\nname = \"synthetic-freeze\"\nversion = \"0\"\n",
        "mo_nco/outputs/ignored.py": b"raise RuntimeError('excluded')\n",
        "mo_nco/__pycache__/ignored.py": b"raise RuntimeError('excluded')\n",
        "ijoc_submission_v21e3r1/scripts/tmp/ignored.py": b"raise RuntimeError('excluded')\n",
        "independent_reproduction/.cache/ignored.py": b"raise RuntimeError('excluded')\n",
        "tests/test_v21e3r1_independent_simultaneous_inference.py": (
            b"def test_contract():\n    assert True\n"
        ),
        "tests/test_v21e3r1_successor_metric.py": (
            b"def test_successor_metric_contract():\n    assert True\n"
        ),
    }
    for relative, raw in sources.items():
        _write_bytes(project / Path(*relative.split("/")), raw)
    entries = _source_entries(project)
    source_root = _sha256(_canonical(entries))
    parent_snapshot = _write_parent_snapshot(project)

    inputs = project / "freeze-inputs"
    semantic = inputs / "semantic-config.json"
    study_metric = inputs / "study-metric-spec.json"
    simultaneous = inputs / "simultaneous-inference-spec.json"
    semantic_sha = _write_canonical(
        semantic,
        {
            "schema": "v21e3r1_successor_semantic_config_v1",
            "study_id": STUDY_ID,
            "candidate_id": CANDIDATE_ID,
            "parameters": EXPECTED_SEMANTIC_PARAMETERS,
        },
    )
    source_by_path = {entry["path"]: entry for entry in entries}
    metric_value = _payload_receipt(
        {
            "schema": "v21e3r1_study_metric_spec_v1",
            "status": "FROZEN_BEFORE_SELECTION",
            "metric_id": "normalized_left_continuous_hypervolume_auc",
            "effect_direction": "LARGER_IS_BETTER",
            "evaluation_axis": "CHARGED_EVALUATIONS",
            "objective_dimension": 2,
            "normalization_contract": (
                "CASE_FROZEN_LOWER_UPPER_AFFINE_TO_UNIT_SQUARE"
            ),
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
            "production_metric_source": source_by_path[
                "mo_nco/pareto_ijoc_analysis.py"
            ],
            "independent_metric_source": source_by_path[
                "independent_reproduction/recompute_v21e3r1_successor_metrics.py"
            ],
            "practical_thresholds_bound_in_simultaneous_spec": True,
            "selection_authorized": False,
            "confirmation_authorized": False,
            "formal_study_authorized": False,
            "scientific_claim_authorized": False,
            "ijoc_submission_status": "IJOC_HOLD",
        }
    )
    metric_sha = _write_canonical(study_metric, metric_value)
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
    simultaneous_sha = _write_canonical(
        simultaneous,
        _payload_receipt(
        {
            "schema": "v21e3r1_simultaneous_inference_spec_v2",
            "status": "PASS_FROZEN_BEFORE_SELECTION_ENGINEERING_ONLY",
            "scope": "FROZEN_PROSPECTIVE_DESIGN_ONLY_NO_CASE_MATERIALIZATION",
            "study_id": STUDY_ID,
            "candidate_id": CANDIDATE_ID,
            "successor_source_sha256": source_root,
            "successor_config_sha256": semantic_sha,
            "study_metric_spec_sha256": metric_sha,
            "evaluator_source_path": (
                "simultaneous-inference-design/recompute-simultaneous-bounds.py"
            ),
            "evaluator_source_sha256": source_by_path[
                "independent_reproduction/recompute_v21e3r1_simultaneous_bounds.py"
            ]["sha256"],
            "evaluator_test_path": (
                "simultaneous-inference-design/recompute-simultaneous-bounds.tests.py"
            ),
            "evaluator_test_sha256": source_by_path[
                "tests/test_v21e3r1_independent_simultaneous_inference.py"
            ]["sha256"],
            "method": (
                "one_sided_observed_se_max_t_paired_case_cluster_bootstrap_v1"
            ),
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
            "resampling_rule": (
                "WITH_REPLACEMENT_INDEPENDENT_BY_FAMILY_"
                "SHARED_ACROSS_CELLS_WITHIN_FAMILY"
            ),
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
        ),
    )
    return {
        "semantic": semantic,
        "semantic_sha": semantic_sha,
        "study_metric": study_metric,
        "metric_sha": metric_sha,
        "simultaneous": simultaneous,
        "simultaneous_sha": simultaneous_sha,
        "source_root": source_root,
        "parent_snapshot_receipt": parent_snapshot["receipt"],
        "parent_source_root": parent_snapshot["source_root"],
    }


def _run(
    project: Path,
    inputs: dict[str, Path | str],
    output: Path,
    *,
    archive: bool = False,
    semantic: Path | None = None,
    parent_sha256: str | None = None,
    parent_receipt: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-I",
        str(SCRIPT),
        "--project-root",
        str(project),
        "--output-directory",
        str(output),
        "--study-id",
        STUDY_ID,
        "--candidate-id",
        CANDIDATE_ID,
        "--parent-development-source-sha256",
        parent_sha256 or str(inputs["parent_source_root"]),
        "--parent-development-snapshot-receipt",
        str(parent_receipt or inputs["parent_snapshot_receipt"]),
        "--semantic-config",
        str(semantic or inputs["semantic"]),
        "--study-metric-spec",
        str(inputs["study_metric"]),
        "--simultaneous-inference-spec",
        str(inputs["simultaneous"]),
    ]
    if archive:
        command.append("--source-archive")
    return subprocess.run(command, text=True, capture_output=True, check=False)


def _make_development_plan(project: Path) -> dict[str, object]:
    source_bytes: dict[str, bytes] = {
        "ijoc_submission_v21e3r1/scripts/run_v21e3r1_development_diagnostics.py": (
            b"def main():\n    return 0\n"
        ),
        "independent_reproduction/recompute_v21e3r1_metrics.py": (
            b"def recompute():\n    return 0.0\n"
        ),
    }
    source_bytes.update(
        {
            f"mo_nco/module_{index:03d}.py": f"VALUE = {index}\n".encode("ascii")
            for index in range(168)
        }
    )
    for relative, raw in source_bytes.items():
        _write_bytes(project / Path(*relative.split("/")), raw)
    entries = [
        {"path": relative, "bytes": len(raw), "sha256": _sha256(raw)}
        for relative, raw in sorted(
            source_bytes.items(), key=lambda item: (item[0].casefold(), item[0])
        )
    ]
    source_root = _sha256(_canonical(entries))
    case_ids = [
        f"v21e3-{family}-development-n{size}-s{index:02d}"
        for family in ("mokp", "motsp")
        for size in (100, 200, 500)
        for index in (0, 1)
    ]
    plan_value = {
        "schema": "v21e3r1_exposed_development_diagnostic_plan_v2",
        "status": "FROZEN_FULL_504_DEVELOPMENT_DIAGNOSTIC",
        "scientific_scope": "EXPOSED_DEVELOPMENT_CASES_ONLY_NO_LATER_PHASE_RELEASE",
        "arms": [
            "C0_STANDARD",
            "C0_RANDOM",
            "C0_NO_LS",
            "C0_RANDOM_NO_LS",
            "C0_SELF_REPLACE",
            "C0_POP_MATCH",
            "NSGAII_STANDARD",
            "NSGAII_SEEDED",
            "NSGAII_POP21",
            "NSGAII_SEEDED_POP21",
            "MOEAD_STANDARD",
            "MOEAD_SEEDED",
            "MOEAD_POP21",
            "MOEAD_SEEDED_POP21",
        ],
        "case_ids": case_ids,
        "seeds": [31051, 31057, 31059],
        "charged_evaluation_budget": 2000,
        "checkpoint_period": 200,
        "row_timeout_seconds": 1800,
        "expected_rows": 504,
        "input_binding": {
            "schema": "v21e3r1_exposed_development_input_binding_v1",
            "case_ids": case_ids,
            "manifest_sha256": {
                "ijoc_submission_v21e3/development_manifests_v1/"
                "config_manifest_development.json": "a" * 64,
                "ijoc_submission_v21e3/development_manifests_v1/"
                "reference_manifest_development.json": "b" * 64,
                "ijoc_submission_v21e3/development_partitions_v1/"
                "case_manifest.json": "c" * 64,
            },
        },
        "source_manifest": {
            "schema": "v21e3r1_diagnostic_source_manifest_v1",
            "entry_count": 170,
            "hash_rule": "sha256(canonical_json(sorted_entries))",
            "source_snapshot_sha256": source_root,
            "entries": entries,
        },
        "selection_entropy_release": "PROHIBITED",
        "confirmation_materialization": "PROHIBITED",
        "formal_materialization": "PROHIBITED",
    }
    plan = project / "outputs" / "live-diagnostic" / "diagnostic.plan.json"
    _write_bytes(
        plan,
        (json.dumps(plan_value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    return {"plan": plan, "entries": entries, "source_root": source_root}


def _run_development(project: Path, plan: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPT),
            "--project-root",
            str(project),
            "--output-directory",
            str(output),
            "--development-diagnostic-plan",
            str(plan),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_freeze_writes_exact_hash_bound_authority_false_artifacts(tmp_path: Path) -> None:
    project = tmp_path / "project"
    inputs = _make_project(project)
    output = project / "outputs" / "freeze-a"

    completed = _run(project, inputs, output)

    assert completed.returncode == 0, completed.stderr
    manifest_raw = (output / "source.manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    assert manifest_raw == _canonical(manifest)
    assert set(manifest) == {"schema", "source_root_sha256", "entries"}
    assert manifest["schema"] == "v21e3r1_branch_replay_source_manifest_binding_v1"
    assert [entry["path"] for entry in manifest["entries"]] == EXPECTED_SOURCE_PATHS
    assert manifest["source_root_sha256"] == _sha256(_canonical(manifest["entries"]))
    entries_by_path = {entry["path"]: entry for entry in manifest["entries"]}
    for relative in (
        "independent_reproduction/recompute_v21e3r1_successor_metrics.py",
        "tests/test_v21e3r1_successor_metric.py",
    ):
        assert entries_by_path[relative]["sha256"] == _sha256(
            (project / Path(*relative.split("/"))).read_bytes()
        )

    assert (output / "semantic.config.json").read_bytes() == Path(inputs["semantic"]).read_bytes()
    assert (output / "study.metric-spec.json").read_bytes() == Path(
        inputs["study_metric"]
    ).read_bytes()
    assert (output / "simultaneous-inference.spec.json").read_bytes() == Path(
        inputs["simultaneous"]
    ).read_bytes()
    receipt_raw = (output / "successor-source.freeze.receipt.json").read_bytes()
    receipt = json.loads(receipt_raw)
    assert receipt_raw == _canonical(receipt)
    assert set(receipt) == RECEIPT_KEYS
    core = dict(receipt)
    payload_sha = core.pop("receipt_payload_sha256")
    assert payload_sha == _sha256(_canonical(core))
    assert receipt["schema"] == "v21e3r1_successor_source_freeze_receipt_v2"
    assert receipt["status"] == "PASS_SUCCESSOR_SOURCE_AND_CONFIG_FREEZE_ENGINEERING_ONLY"
    assert receipt["source_snapshot_sha256"] == inputs["source_root"]
    assert receipt["source_manifest_sha256"] == _sha256(manifest_raw)
    assert receipt["semantic_config_sha256"] == inputs["semantic_sha"]
    assert receipt["study_metric_spec_sha256"] == inputs["metric_sha"]
    assert receipt["simultaneous_inference_spec_sha256"] == inputs["simultaneous_sha"]
    assert receipt["source_entry_count"] == len(EXPECTED_SOURCE_PATHS)
    assert receipt["source_total_bytes"] == sum(entry["bytes"] for entry in manifest["entries"])
    assert receipt["all_source_files_verified"] is True
    assert receipt["source_frozen"] is True
    for field in (
        "selection_cases_materialized",
        "confirmation_cases_materialized",
        "formal_cases_materialized",
        "implementation_independence",
        "scientific_independence",
        "selection_authorized",
        "confirmation_authorized",
        "formal_study_authorized",
        "scientific_claim_authorized",
        "public_redistribution_authorized",
    ):
        assert receipt[field] is False
    assert receipt["ijoc_submission_status"] == "IJOC_HOLD"
    assert receipt["source_archive_materialized"] is False
    assert receipt["source_archive_path"] is None
    assert receipt["source_archive_sha256"] is None
    assert not (output / "successor-source.zip").exists()


def test_source_archive_is_byte_deterministic_and_restores_only_manifest_sources(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    inputs = _make_project(project)
    first = project / "outputs" / "freeze-a"
    second = project / "outputs" / "freeze-b"

    first_run = _run(project, inputs, first, archive=True)
    second_run = _run(project, inputs, second, archive=True)

    assert first_run.returncode == second_run.returncode == 0
    for name in (
        "source.manifest.json",
        "semantic.config.json",
        "study.metric-spec.json",
        "simultaneous-inference.spec.json",
        "successor-source.freeze.receipt.json",
        "successor-source.zip",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    with zipfile.ZipFile(first / "successor-source.zip", "r") as archive:
        assert archive.namelist() == EXPECTED_SOURCE_PATHS
        for info in archive.infolist():
            assert info.compress_type == zipfile.ZIP_STORED
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.create_system == 3
            assert info.external_attr >> 16 == 0o100644
            assert archive.read(info.filename) == (
                project / Path(*info.filename.split("/"))
            ).read_bytes()
    receipt = json.loads((first / "successor-source.freeze.receipt.json").read_bytes())
    assert receipt["source_archive_materialized"] is True
    assert receipt["source_archive_path"] == "successor-source.zip"
    assert receipt["source_archive_sha256"] == _sha256(
        (first / "successor-source.zip").read_bytes()
    )
    assert receipt["source_archive_scope"] == (
        "SOURCE_INVENTORY_ONLY_INTERNAL_CUSTODY_NO_REDISTRIBUTION_AUTHORITY"
    )
    assert receipt["public_redistribution_authorized"] is False


def test_noncanonical_config_and_simultaneous_identity_drift_fail_closed(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    inputs = _make_project(project)
    semantic = Path(inputs["semantic"])
    value = json.loads(semantic.read_bytes())
    semantic.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    output = project / "outputs" / "noncanonical"

    noncanonical = _run(project, inputs, output)

    assert noncanonical.returncode == 3
    assert not output.exists()
    assert "semantic config is not canonical JSON" in noncanonical.stderr

    inputs = _make_project(project)
    simultaneous = Path(inputs["simultaneous"])
    value = json.loads(simultaneous.read_bytes())
    value["candidate_id"] = "C2"
    _write_canonical(simultaneous, value)
    drift_output = project / "outputs" / "identity-drift"

    drift = _run(project, inputs, drift_output)

    assert drift.returncode == 3
    assert not drift_output.exists()
    assert "simultaneous-inference identity disagrees" in drift.stderr

    inputs = _make_project(project)
    study_metric = Path(inputs["study_metric"])
    _write_canonical(
        study_metric,
        {"schema": "v21e3r1_operator_accounting_reanalysis_spec_v1"},
    )
    wrong_metric_output = project / "outputs" / "wrong-metric-identity"

    wrong_metric = _run(project, inputs, wrong_metric_output)

    assert wrong_metric.returncode == 3
    assert not wrong_metric_output.exists()
    assert "operator-accounting spec" in wrong_metric.stderr

    inputs = _make_project(project)
    study_metric = Path(inputs["study_metric"])
    metric_value = json.loads(study_metric.read_bytes())
    metric_core = dict(metric_value)
    metric_core.pop("receipt_payload_sha256")
    metric_core["production_metric_source"]["sha256"] = "d" * 64
    _write_canonical(study_metric, _payload_receipt(metric_core))
    binding_output = project / "outputs" / "metric-source-binding-drift"

    binding_drift = _run(project, inputs, binding_output)

    assert binding_drift.returncode == 3
    assert not binding_output.exists()
    assert "disagrees with successor source manifest" in binding_drift.stderr

    inputs = _make_project(project)
    simultaneous = Path(inputs["simultaneous"])
    simultaneous_value = json.loads(simultaneous.read_bytes())
    simultaneous_core = dict(simultaneous_value)
    simultaneous_core.pop("receipt_payload_sha256")
    simultaneous_core["practical_thresholds"]["adjacent_mechanism_effect"] = -0.005
    _write_canonical(simultaneous, _payload_receipt(simultaneous_core))
    threshold_output = project / "outputs" / "negative-practical-threshold"

    threshold_drift = _run(project, inputs, threshold_output)

    assert threshold_drift.returncode == 3
    assert not threshold_output.exists()
    assert "practical thresholds drifted" in threshold_drift.stderr


def test_resealed_unrelated_semantic_parameters_fail_closed(tmp_path: Path) -> None:
    project = tmp_path / "project"
    inputs = _make_project(project)
    semantic = Path(inputs["semantic"])
    value = json.loads(semantic.read_bytes())
    value["parameters"] = {"x": "y"}
    semantic_sha = _write_canonical(semantic, value)
    simultaneous = Path(inputs["simultaneous"])
    simultaneous_value = json.loads(simultaneous.read_bytes())
    simultaneous_core = dict(simultaneous_value)
    simultaneous_core.pop("receipt_payload_sha256")
    simultaneous_core["successor_config_sha256"] = semantic_sha
    _write_canonical(simultaneous, _payload_receipt(simultaneous_core))

    completed = _run(project, inputs, project / "outputs" / "bad-semantic-policy")

    assert completed.returncode == 3
    assert "semantic config policy contract drifted" in completed.stderr
    assert not (project / "outputs" / "bad-semantic-policy").exists()


def test_parent_snapshot_and_simultaneous_evaluator_bindings_fail_closed(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    inputs = _make_project(project)
    parent_output = project / "outputs" / "wrong-parent-root"

    wrong_parent = _run(project, inputs, parent_output, parent_sha256="d" * 64)

    assert wrong_parent.returncode == 3
    assert not parent_output.exists()
    assert "parent development snapshot identity" in wrong_parent.stderr

    parent_receipt = Path(inputs["parent_snapshot_receipt"])
    archive = parent_receipt.parent / "development-source.zip"
    archive.write_bytes(archive.read_bytes() + b"corrupt")
    archive_output = project / "outputs" / "corrupt-parent-archive"

    corrupt_archive = _run(project, inputs, archive_output)

    assert corrupt_archive.returncode == 3
    assert not archive_output.exists()
    assert "sibling artifact hash drifted" in corrupt_archive.stderr

    inputs = _make_project(project)
    simultaneous = Path(inputs["simultaneous"])
    value = json.loads(simultaneous.read_bytes())
    core = dict(value)
    core.pop("receipt_payload_sha256")
    core["evaluator_source_sha256"] = "e" * 64
    _write_canonical(simultaneous, _payload_receipt(core))
    evaluator_output = project / "outputs" / "fake-sim-evaluator"

    fake_evaluator = _run(project, inputs, evaluator_output)

    assert fake_evaluator.returncode == 3
    assert not evaluator_output.exists()
    assert "source/test hashes disagree" in fake_evaluator.stderr


def test_unpaired_surrogate_and_walk_error_return_fail_closed_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    inputs = _make_project(project)
    semantic = Path(inputs["semantic"])
    semantic.write_bytes(
        (
            '{"candidate_id":"C1","parameters":{"bad":"\\ud800"},'
            '"schema":"v21e3r1_successor_semantic_config_v1",'
            '"study_id":"v21e3r1-prospective-test"}'
        ).encode("ascii")
    )
    surrogate_output = project / "outputs" / "surrogate"

    surrogate = _run(project, inputs, surrogate_output)

    assert surrogate.returncode == 3
    assert not surrogate_output.exists()
    assert "canonical JSON encoding failed" in surrogate.stderr
    assert "Traceback" not in surrogate.stderr

    spec = importlib.util.spec_from_file_location("walk_error_freeze", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def denied_walk(*args: object, **kwargs: object) -> list[object]:
        onerror = kwargs["onerror"]
        assert callable(onerror)
        onerror(PermissionError("synthetic traversal denial"))
        return []

    monkeypatch.setattr(module.os, "walk", denied_walk)
    with pytest.raises(module.FreezeError, match="source inventory traversal failed"):
        module._collect_source_records(project.resolve())


def test_paths_are_contained_and_existing_output_is_never_replaced(tmp_path: Path) -> None:
    project = tmp_path / "project"
    inputs = _make_project(project)
    outside = tmp_path / "outside.json"
    outside.write_bytes(Path(inputs["semantic"]).read_bytes())
    escaped_output = project / "outputs" / "escaped"

    escaped = _run(project, inputs, escaped_output, semantic=outside)

    assert escaped.returncode == 3
    assert not escaped_output.exists()
    assert "escapes project root" in escaped.stderr

    existing = project / "outputs" / "existing"
    existing.mkdir(parents=True)
    sentinel = existing / "sentinel.bin"
    sentinel.write_bytes(b"immutable")

    collision = _run(project, inputs, existing)

    assert collision.returncode == 3
    assert sentinel.read_bytes() == b"immutable"
    assert "output directory already exists" in collision.stderr

    if sys.platform == "win32":
        case_alias = project / "MO_NCO" / "forbidden-freeze"
        case_collision = _run(project, inputs, case_alias)
        assert case_collision.returncode == 3
        assert not case_alias.exists()
        assert "inside the frozen source scope" in case_collision.stderr


def test_second_pass_reenumeration_detects_new_source_before_pass_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    inputs = _make_project(project)
    spec = importlib.util.spec_from_file_location("successor_freeze_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module._revalidate_records
    target = project / "mo_nco" / "late_injected.PY"
    revalidation_calls = 0

    def mutate_before_revalidation(records: object) -> None:
        nonlocal revalidation_calls
        revalidation_calls += 1
        if revalidation_calls == 2:
            target.write_bytes(b"VALUE = 2\n")
        original(records)

    monkeypatch.setattr(module, "_revalidate_records", mutate_before_revalidation)
    output = project / "outputs" / "toctou"

    with pytest.raises(module.FreezeError, match="source inventory changed"):
        module.freeze_source(
            project_root=project,
            output_directory=output,
            study_id=STUDY_ID,
            candidate_id=CANDIDATE_ID,
            parent_development_source_sha256=str(inputs["parent_source_root"]),
            parent_development_snapshot_receipt_path=Path(
                inputs["parent_snapshot_receipt"]
            ),
            semantic_config_path=Path(inputs["semantic"]),
            study_metric_spec_path=Path(inputs["study_metric"]),
            simultaneous_inference_spec_path=Path(inputs["simultaneous"]),
            source_archive=False,
        )
    assert revalidation_calls == 2
    assert output.is_dir()
    assert not (output / "successor-source.freeze.receipt.json").exists()


def test_development_plan_snapshot_is_exact_restorable_and_deterministic(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    frozen = _make_development_plan(project)
    plan = Path(frozen["plan"])
    first = project / "outputs" / "development-source-snapshot-a"
    second = project / "outputs" / "development-source-snapshot-b"

    first_run = _run_development(project, plan, first)
    second_run = _run_development(project, plan, second)

    assert first_run.returncode == second_run.returncode == 0, first_run.stderr
    output_names = (
        "development-diagnostic.plan.json",
        "development-source.manifest.json",
        "development-source.zip",
        "development-source.snapshot.receipt.json",
    )
    for name in output_names:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert (first / "development-diagnostic.plan.json").read_bytes() == plan.read_bytes()
    manifest_raw = (first / "development-source.manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    assert manifest_raw == _canonical(manifest)
    assert set(manifest) == {
        "schema",
        "entry_count",
        "hash_rule",
        "source_snapshot_sha256",
        "entries",
    }
    assert manifest["entries"] == frozen["entries"]
    assert manifest["entry_count"] == 170
    assert manifest["source_snapshot_sha256"] == frozen["source_root"]
    with zipfile.ZipFile(first / "development-source.zip", "r") as archive:
        assert archive.namelist() == [entry["path"] for entry in frozen["entries"]]
        for info, entry in zip(archive.infolist(), frozen["entries"], strict=True):
            assert info.compress_type == zipfile.ZIP_STORED
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert archive.read(info.filename) == (
                project / Path(*entry["path"].split("/"))
            ).read_bytes()
    receipt_raw = (first / "development-source.snapshot.receipt.json").read_bytes()
    receipt = json.loads(receipt_raw)
    assert receipt_raw == _canonical(receipt)
    core = dict(receipt)
    payload_sha = core.pop("receipt_payload_sha256")
    assert payload_sha == _sha256(_canonical(core))
    assert receipt["schema"] == "v21e3r1_development_source_snapshot_receipt_v1"
    assert receipt["status"] == (
        "PASS_DEVELOPMENT_EXECUTION_SOURCE_SNAPSHOT_ENGINEERING_ONLY"
    )
    assert receipt["diagnostic_plan_sha256"] == _sha256(plan.read_bytes())
    assert receipt["source_manifest_sha256"] == _sha256(manifest_raw)
    assert receipt["source_archive_sha256"] == _sha256(
        (first / "development-source.zip").read_bytes()
    )
    assert receipt["source_entry_count"] == 170
    assert receipt["all_source_files_verified"] is True
    assert receipt["source_archive_materialized"] is True
    assert receipt["original_development_diagnostic_artifacts_modified"] is False
    for field in (
        "development_execution_replayed",
        "selection_cases_materialized",
        "confirmation_cases_materialized",
        "formal_cases_materialized",
        "implementation_independence",
        "scientific_independence",
        "selection_authorized",
        "confirmation_authorized",
        "formal_study_authorized",
        "scientific_claim_authorized",
        "public_redistribution_authorized",
    ):
        assert receipt[field] is False
    assert receipt["ijoc_submission_status"] == "IJOC_HOLD"


def test_development_snapshot_rejects_source_drift_and_live_output_write(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    frozen = _make_development_plan(project)
    plan = Path(frozen["plan"])
    first_entry = frozen["entries"][0]
    assert isinstance(first_entry, dict)
    source = project / Path(*str(first_entry["path"]).split("/"))
    source.write_bytes(source.read_bytes() + b"# drift\n")
    drift_output = project / "outputs" / "development-source-drift"

    drift = _run_development(project, plan, drift_output)

    assert drift.returncode == 3
    assert not drift_output.exists()
    assert "disagrees with frozen plan" in drift.stderr

    frozen = _make_development_plan(project)
    plan = Path(frozen["plan"])
    live_child = plan.parent / "forbidden-snapshot"
    live_write = _run_development(project, plan, live_child)

    assert live_write.returncode == 3
    assert not live_child.exists()
    assert "live diagnostic output directory" in live_write.stderr

