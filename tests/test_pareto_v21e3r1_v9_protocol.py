from __future__ import annotations

import copy
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from mo_nco.pareto_v21e3r1_v9_protocol import (
    V9PredevelopmentProtocolError,
    load_v9_predevelopment_protocol,
    validate_v9_predevelopment_protocol,
    validate_v9_resource_caps,
)
from mo_nco.pareto_ijoc_problem import MultiObjectiveKnapsackInstance
from mo_nco.pareto_v21e3_hybrid import (
    V21E3HybridConfig,
    V21E3TypedHybridParetoSearch,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_THEORY = _PROJECT_ROOT / "docs" / "V21E3R1_V9R1_THEORY.md"
_PROTOCOL = (
    _PROJECT_ROOT
    / "mo_nco"
    / "specs"
    / "V21E3R1_V9R2_PREDEVELOPMENT_PROTOCOL.json"
)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def test_resource_caps_accept_exact_b_a_s_t_types() -> None:
    validated = validate_v9_resource_caps(B=8, A=128, S=4096, T=30.0)

    assert validated["resource_caps"] == {
        "B": 8,
        "A": 128,
        "S": 4096,
        "T": 30.0,
    }
    assert validated["status"] == "RESOURCE_CAPS_VALIDATED_NOT_AUTHORIZED"
    assert len(validated["canonical_sha256"]) == 64


@pytest.mark.parametrize(
    ("caps", "message"),
    [
        ({"B": True, "A": 128, "S": 4096, "T": 30.0}, "B"),
        ({"B": 8, "A": 128.0, "S": 4096, "T": 30.0}, "A"),
        ({"B": 8, "A": 128, "S": False, "T": 30.0}, "S"),
        ({"B": 8, "A": 128, "S": 4096, "T": "30"}, "T"),
        ({"B": 8, "A": 128, "S": 4096, "T": Decimal("30")}, "T"),
        ({"B": 8, "A": 128, "S": 4096, "T": 10**400}, "T"),
        ({"B": 8, "A": 7, "S": 4096, "T": 30.0}, "at least B"),
    ],
)
def test_resource_caps_reject_coercible_or_invalid_values(
    caps: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(V9PredevelopmentProtocolError, match=message):
        validate_v9_resource_caps(**caps)


def test_packaged_protocol_is_a_machine_readable_predevelopment_hold() -> None:
    certificate = load_v9_predevelopment_protocol()

    assert certificate["status"] == "PRE_DEVELOPMENT_HOLD"
    assert certificate["execution_authorization"] == {
        "full_development_matrix": False,
        "scientific_development_claims": False,
        "single_case_smoke": True,
        "single_case_smoke_scope": (
            "ENGINEERING_ONLY_EXPOSED_CASE_NON_SCIENTIFIC"
        ),
    }
    assert certificate["later_phase_authorization"] == {
        "any_later_phase": False,
        "candidate_adoption": False,
        "confirmation": False,
        "formal_study": False,
        "ijoc_submission": False,
        "selection": False,
    }
    assert certificate["unmet_required_artifacts"] == [
        "current_source_test_receipt",
        "environment_lock",
        "full_development_algorithm_spec",
        "full_source_freeze",
        "independent_algorithm_decision_replay",
        "metric_reference_manifest",
        "rights_ledger",
        "strong_baseline_registry",
        "target_scale_resource_capacity_receipt",
        "trace_replay_spec",
    ]
    assert len(certificate["canonical_sha256"]) == 64
    assert len(certificate["resource_contract_sha256"]) == 64
    assert len(certificate["source_file_sha256"]) == 64


def test_protocol_loader_accepts_explicit_path_and_file_sha_binding() -> None:
    expected_file_sha256 = hashlib.sha256(_PROTOCOL.read_bytes()).hexdigest()

    certificate = load_v9_predevelopment_protocol(
        _PROTOCOL,
        expected_file_sha256=expected_file_sha256,
    )

    assert certificate["source_file_sha256"] == expected_file_sha256


def test_protocol_canonically_binds_resource_menu_and_exposed_scope() -> None:
    raw = _PROTOCOL.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    core = dict(payload)
    declared_payload_sha256 = core.pop("protocol_payload_sha256")

    assert raw == _canonical_json_bytes(payload) + b"\n"
    assert hashlib.sha256(_canonical_json_bytes(core)).hexdigest() == (
        declared_payload_sha256
    )

    certificate = validate_v9_predevelopment_protocol(payload)
    assert certificate["canonical_sha256"] == hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    resource = payload["resource_contract"]
    assert resource["caps_frozen_for_full_development_matrix"] is False
    assert resource["quantities"]["B"]["meaning"] == (
        "first_true_objective_evaluations"
    )
    assert resource["quantities"]["A"]["meaning"] == (
        "attempts_submitted_to_durable_ledger"
    )
    assert resource["quantities"]["S"]["components"] == [
        "structural_candidate_generations",
        "cache_membership_probes",
    ]
    assert resource["quantities"]["T"] == {
        "cap_field": "wall_time_cap_seconds",
        "clock": "time.perf_counter",
        "counter_field": "elapsed_seconds",
        "meaning": "monotonic_elapsed_wall_time",
        "scope": "current_python_process",
        "value_type": "finite_positive_exact_builtin_int_or_float",
    }
    assert payload["candidate_menu"]["arm_order"] == [
        "LEGACY",
        "SCREEN",
        "LYAP",
        "BOTH",
    ]
    assert payload["execution_scope"] == {
        "acknowledgement_required": True,
        "case_manifest_path": (
            "ijoc_submission_v21e3/development_partitions_v1/case_manifest.json"
        ),
        "case_manifest_sha256": (
            "1970361ba557aadd26de38aed008de11d11d158c797c00db1036cc4616cbdc8c"
        ),
        "new_case_materialization_allowed": False,
        "partition": "EXPOSED_DEVELOPMENT_ONLY",
    }


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("execution_authorization", "single_case_smoke"), 1),
        (("execution_authorization", "full_development_matrix"), True),
        (("later_phase_authorization", "selection"), True),
        (("execution_scope", "partition"), "SELECTION"),
        (
            ("required_artifacts", "environment_lock", "satisfied"),
            True,
        ),
        (("candidate_menu", "arm_order"), ["LEGACY", "SCREEN"]),
    ],
)
def test_protocol_rejects_type_authorization_scope_and_menu_drift(
    path: tuple[str, ...],
    replacement: object,
) -> None:
    payload = json.loads(_PROTOCOL.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(payload)
    target = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement

    with pytest.raises(V9PredevelopmentProtocolError):
        validate_v9_predevelopment_protocol(mutated)


def test_protocol_rejects_unknown_non_string_root_key() -> None:
    payload = json.loads(_PROTOCOL.read_text(encoding="utf-8"))
    payload[1] = "not-a-json-object-key"

    with pytest.raises(V9PredevelopmentProtocolError, match="keys"):
        validate_v9_predevelopment_protocol(payload)


def test_protocol_rejects_forged_self_hash() -> None:
    payload = json.loads(_PROTOCOL.read_text(encoding="utf-8"))
    payload["protocol_payload_sha256"] = "0" * 64

    with pytest.raises(V9PredevelopmentProtocolError, match="does not bind"):
        validate_v9_predevelopment_protocol(payload)


def test_protocol_loader_rejects_duplicate_and_noncanonical_json(
    tmp_path: Path,
) -> None:
    raw = _PROTOCOL.read_bytes()
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"schema":"forged",' + raw[1:])
    with pytest.raises(V9PredevelopmentProtocolError, match="duplicate JSON key"):
        load_v9_predevelopment_protocol(duplicate)

    payload = json.loads(raw.decode("utf-8"))
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(V9PredevelopmentProtocolError, match="canonical JSON"):
        load_v9_predevelopment_protocol(noncanonical)


def test_protocol_loader_rejects_wrong_file_sha_binding() -> None:
    with pytest.raises(V9PredevelopmentProtocolError, match="differs"):
        load_v9_predevelopment_protocol(
            _PROTOCOL,
            expected_file_sha256="0" * 64,
        )


def test_protocol_case_manifest_and_arm_menu_match_current_v9r1_code() -> None:
    from mo_nco import pareto_v21e3r1_v9_runner as runner

    payload = load_v9_predevelopment_protocol()["payload"]
    scope = payload["execution_scope"]
    manifest = _PROJECT_ROOT / scope["case_manifest_path"]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == (
        scope["case_manifest_sha256"]
    )

    menu = payload["candidate_menu"]
    assert tuple(menu["arm_order"]) == runner._ARM_ORDER
    for arm_contract in menu["arms"]:
        arm = arm_contract["arm"]
        diagnostic_stem, screening_policy, uses_lyapunov, uses_screening = (
            runner._ARM_POLICIES[arm]
        )
        assert arm_contract["diagnostic_stem"] == diagnostic_stem
        assert arm_contract["candidate_screening_policy"] == screening_policy
        assert arm_contract["lyapunov_enabled"] is uses_lyapunov
        assert arm_contract["screening_enabled"] is uses_screening


def test_theory_resource_contract_matches_executable_s_definition() -> None:
    """The published S definition must match observable runtime accounting."""

    problem = MultiObjectiveKnapsackInstance.random_instance(
        10,
        num_objectives=2,
        seed=1201,
    )
    config = V21E3HybridConfig(
        candidate_id="C0",
        reference_directions=((0.25, 0.75), (0.75, 0.25)),
        charged_evaluations=8,
        checkpoint_period=4,
        seed=1701,
        phase="development",
        capture_trace=False,
        local_improvement_steps=1,
        development_diagnostic_id="V21E3R1_V9_INFORMATION_SCREEN_MOKP",
        candidate_screening_policy=(
            "bounded_cache_aware_structural_screen_development_v1"
        ),
        candidate_screening_cap=4,
        archive_tradeoff_lambda=0.0,
        replacement_policy=(
            "bounded_reference_neighborhood_nonworse_replacement_v1"
        ),
        attempt_cap=128,
        structural_screening_cap=4096,
        wall_time_cap_seconds=30.0,
    )

    metadata = (
        V21E3TypedHybridParetoSearch(problem, config)
        .run()
        .optimization_result.metadata
    )
    accounting = metadata["v9_resource_accounting"]
    assert accounting["structural_candidate_generations"] > 0
    assert accounting["cache_membership_probes"] > 0
    assert accounting["structural_screening_work"] == (
        accounting["structural_candidate_generations"]
        + accounting["cache_membership_probes"]
    )

    theory = _THEORY.read_text(encoding="utf-8")
    assert (
        "S = structural_candidate_generations + cache_membership_probes"
        in theory
    )
    assert "S\\) 仅是 exact cache-membership queries" not in theory
    assert "`DualResourceBudget` 已在 V9R1 主运行器中 fail-closed 接入" in theory
