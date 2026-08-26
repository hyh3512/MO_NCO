from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "freeze_v21e3r1_prospective_boundaries.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("prospective_boundary_freezer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def test_freezes_real_development_boundaries_without_granting_authority(
    tmp_path: Path,
) -> None:
    module = _load_module()
    output = tmp_path / "prospective-boundaries"
    legacy_metric = ROOT / "independent_reproduction/recompute_v21e3r1_metrics.py"
    legacy_metric_sha256 = hashlib.sha256(legacy_metric.read_bytes()).hexdigest()

    receipt = module.freeze_prospective_boundaries(
        repository_root=ROOT,
        output_directory=output,
        study_id="v21e3r1-prospective-test",
        candidate_id="C3",
        successor_source_sha256="1" * 64,
        successor_config_sha256="2" * 64,
    )

    assert receipt["schema"] == "v21e3r1_prospective_boundary_freeze_receipt_v1"
    assert receipt["status"] == "PASS_PROSPECTIVE_BOUNDARIES_FROZEN_ENGINEERING_ONLY"
    assert receipt["external_family_native_strong_baseline_count_by_family"] == {
        "MOKP": 0,
        "MOTSP": 0,
    }
    assert receipt["selection_authorized"] is False
    assert receipt["confirmation_authorized"] is False
    assert receipt["formal_study_authorized"] is False
    assert receipt["scientific_claim_authorized"] is False
    assert receipt["ijoc_submission_status"] == "IJOC_HOLD"
    study_metric = _read(output / "study.metric-spec.json")
    assert study_metric["schema"] == "v21e3r1_study_metric_spec_v1"
    assert study_metric["integration_contract"] == (
        "EAUC=(1/B)*SUM_{b=1..B}HV(A_{b-1})"
    )
    assert study_metric["row_crosscheck"] == {
        "failure_policy": "HOLD_ON_ANY_MISMATCH",
        "required": True,
        "scope": "EVERY_FORMAL_STUDY_ROW",
        "tolerance": 0.0,
    }
    for field in ("production_metric_source", "independent_metric_source"):
        binding = study_metric[field]
        source = ROOT / binding["path"]
        raw = source.read_bytes()
        assert binding["bytes"] == len(raw)
        assert binding["sha256"] == hashlib.sha256(raw).hexdigest()
    assert study_metric["independent_metric_source"]["path"] == (
        "independent_reproduction/recompute_v21e3r1_successor_metrics.py"
    )
    assert hashlib.sha256(legacy_metric.read_bytes()).hexdigest() == legacy_metric_sha256

    baseline = _read(output / "baseline-registry.receipt.json")
    assert baseline["status"] == "HOLD_NO_EXTERNAL_FAMILY_NATIVE_STRONG_BASELINES"
    assert baseline["all_registry_artifacts_verified"] is True
    assert [family["external_family_native_strong_baseline_count"] for family in baseline["families"]] == [0, 0]

    external = _read(output / "external-algorithm-replay.receipt.json")
    assert external["status"] == "HOLD_DESIGN_ONLY_NO_EXTERNAL_PRODUCER"
    assert external["event_streams_match"] is True
    for field in (
        "independent_producer",
        "independent_custody",
        "implementation_code_disjoint",
        "algorithm_execution_independence",
        "selection_authorized",
        "confirmation_authorized",
        "formal_study_authorized",
    ):
        assert external[field] is False

    simultaneous = _read(output / "simultaneous-inference.spec.json")
    assert simultaneous["method"] == (
        "one_sided_observed_se_max_t_paired_case_cluster_bootstrap_v1"
    )
    assert simultaneous["familywise_alpha"] == 0.05
    assert simultaneous["bootstrap_samples"] == 10_000
    assert simultaneous["bootstrap_seed"] == 20_260_823
    assert simultaneous["rng_protocol"] == "SHA256_COUNTER_U64_REJECTION_V1"
    assert simultaneous["practical_thresholds"] == {
        "adjacent_mechanism_effect": 0.005,
        "primary_effect": 0.0,
    }
    assert len(simultaneous["selection_cells"]) == 10
    assert len(simultaneous["confirmation_cells"]) == 4

    candidate_menu = _read(output / "successor-candidate-menu.contract.json")
    assert candidate_menu["schema"] == (
        "v21e3r1_successor_candidate_menu_contract_v1"
    )
    assert candidate_menu["status"] == (
        "HOLD_SUCCESSOR_CANDIDATE_ABSENT_FROM_SIMULTANEOUS_SPEC_V2"
    )
    assert candidate_menu["historical_candidate_order"] == ["C0", "C1", "C2", "C3"]
    assert candidate_menu["successor_candidate_id"] == (
        "V21E3R1_SUCCESSOR_SEARCH_NOVELTY_V1"
    )
    assert candidate_menu["successor_candidate_contract"] == {
        "backbone_candidate_id": "C0",
        "legacy_mokp_novelty_generation_policy": "legacy_retry_and_local_v21e3r1_v1",
        "legacy_post_initialization_search_policy": "proposal_chain_v21e3r1_v1",
        "successor_candidate_id": "V21E3R1_SUCCESSOR_SEARCH_NOVELTY_V1",
        "successor_mokp_novelty_generation_policy": "single_attempt_rotating_feasible_exchange_no_refill_development_v1",
        "successor_post_initialization_search_policy": "post_commit_type_incumbent_anchor_development_v1",
    }
    assert candidate_menu["successor_present_in_simultaneous_spec_v2"] is False
    assert candidate_menu["selection_prohibited_until_revised_simultaneous_spec"] is True
    assert candidate_menu["selection_result_materialized"] is False
    assert candidate_menu["confirmation_result_materialized"] is False
    assert candidate_menu["formal_study_materialized"] is False
    assert candidate_menu["legacy_c1_decision_shape"] == {
        "decision_role_count": 4,
        "decision_roles": [
            "MOKP:C1-C0:primary",
            "MOKP:C1-C0:adjacent",
            "MOTSP:C1-C0:primary",
            "MOTSP:C1-C0:adjacent",
        ],
        "duplicate_statistic_per_role": False,
        "statistical_contrast_count": 2,
        "statistical_contrasts": ["MOKP:C1-C0", "MOTSP:C1-C0"],
    }

    external_v3 = _read(output / "external-replay-v3.evidence-contract.json")
    assert external_v3["schema"] == (
        "v21e3r1_external_algorithm_replay_evidence_contract_v3"
    )
    assert external_v3["external_producer_present"] is False
    assert external_v3["independent_custody_authority_present"] is False
    assert external_v3["gate_clearable_by_this_contract"] is False
    assert external_v3["required_path_bound_roles"] == [
        "reference_source_manifest",
        "external_source_manifest",
        "reference_event_stream",
        "external_event_stream",
        "neutral_comparison_receipt",
        "producer_authorship_authority_receipt",
        "independent_custody_authority_receipt",
        "external_execution_environment_receipt",
    ]

    phase_v3 = _read(output / "path-bound-phase-v3.evidence-contract.json")
    assert phase_v3["schema"] == (
        "v21e3r1_path_bound_phase_evidence_contract_v3"
    )
    assert phase_v3["supported_phases"] == ["selection", "confirmation"]
    assert phase_v3["selection_confirmation_case_disjointness_required"] is True
    assert phase_v3["prospective_chronology_required"] is True
    assert phase_v3["external_phase_producer_present"] is False
    assert phase_v3["gate_clearable_by_this_contract"] is False

    precedent = _read(output / "precedent-mechanism.receipt.json")
    assert precedent["review_scope"] == "TARGETED_NOT_SYSTEMATIC"
    assert precedent["all_renderings_verified"] is True
    assert precedent["cross_format_byte_parity_verified"] is False
    assert precedent[
        "rendering_content_parity_after_trailing_newline_normalization"
    ] is True
    assert precedent["selection_authorized"] is False

    for name in (
        "study.metric-spec.json",
        "baseline-registry.receipt.json",
        "external-algorithm-replay.receipt.json",
        "simultaneous-inference.spec.json",
        "successor-candidate-menu.contract.json",
        "external-replay-v3.evidence-contract.json",
        "path-bound-phase-v3.evidence-contract.json",
        "precedent-mechanism.receipt.json",
        "prospective-boundary.freeze.receipt.json",
    ):
        bound = _read(output / name)
        payload_sha256 = bound.pop("receipt_payload_sha256")
        canonical = json.dumps(
            bound,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        assert payload_sha256 == hashlib.sha256(canonical).hexdigest()

    with pytest.raises(module.FreezeError, match="exclusive"):
        module.freeze_prospective_boundaries(
            repository_root=ROOT,
            output_directory=output,
            study_id="v21e3r1-prospective-test",
            candidate_id="C3",
            successor_source_sha256="1" * 64,
            successor_config_sha256="2" * 64,
        )


def test_exact_sha_types_and_output_parent_are_fail_closed(tmp_path: Path) -> None:
    module = _load_module()

    with pytest.raises(module.FreezeError, match="lowercase SHA-256"):
        module.freeze_prospective_boundaries(
            repository_root=ROOT,
            output_directory=tmp_path / "bad-sha",
            study_id="v21e3r1-prospective-test",
            candidate_id="C3",
            successor_source_sha256=True,
            successor_config_sha256="2" * 64,
        )
    assert not (tmp_path / "bad-sha").exists()

    with pytest.raises(module.FreezeError, match="parent does not exist"):
        module.freeze_prospective_boundaries(
            repository_root=ROOT,
            output_directory=tmp_path / "missing" / "nested" / "freeze",
            study_id="v21e3r1-prospective-test",
            candidate_id="C3",
            successor_source_sha256="1" * 64,
            successor_config_sha256="2" * 64,
        )

