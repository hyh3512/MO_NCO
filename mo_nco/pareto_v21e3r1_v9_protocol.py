from __future__ import annotations

"""Fail-closed V9R2 pre-development protocol validation.

This module validates contracts and authorization boundaries only.  It does
not run experiments, manufacture missing evidence, or promote a protocol out
of ``PRE_DEVELOPMENT_HOLD``.
"""

import hashlib
from importlib import resources
import json
import math
from pathlib import Path
import re


V9_PREDEVELOPMENT_PROTOCOL_SCHEMA = (
    "pareto_v21e3r1_v9r2_predevelopment_protocol_v1"
)
V9_PREDEVELOPMENT_PROTOCOL_RESOURCE = (
    "specs/V21E3R1_V9R2_PREDEVELOPMENT_PROTOCOL.json"
)

_SCREENING_POLICY = "bounded_cache_aware_structural_screen_development_v1"
_NONWORSE_REPLACEMENT = (
    "bounded_reference_neighborhood_nonworse_replacement_v1"
)
_LYAPUNOV_REPLACEMENT = (
    "archive_compensated_information_lyapunov_development_v1"
)
_DEVELOPMENT_CASE_MANIFEST_SHA256 = (
    "1970361ba557aadd26de38aed008de11d11d158c797c00db1036cc4616cbdc8c"
)

_EXPECTED_PROTOCOL_CORE: dict[str, object] = {
    "candidate_menu": {
        "arm_order": ["LEGACY", "SCREEN", "LYAP", "BOTH"],
        "arms": [
            {
                "arm": "LEGACY",
                "candidate_screening_policy": "disabled_v1",
                "diagnostic_stem": "LEGACY",
                "lyapunov_enabled": False,
                "replacement_policy": _NONWORSE_REPLACEMENT,
                "screening_enabled": False,
            },
            {
                "arm": "SCREEN",
                "candidate_screening_policy": _SCREENING_POLICY,
                "diagnostic_stem": "INFORMATION_SCREEN",
                "lyapunov_enabled": False,
                "replacement_policy": _NONWORSE_REPLACEMENT,
                "screening_enabled": True,
            },
            {
                "arm": "LYAP",
                "candidate_screening_policy": "disabled_v1",
                "diagnostic_stem": "LYAPUNOV",
                "lyapunov_enabled": True,
                "replacement_policy": _LYAPUNOV_REPLACEMENT,
                "screening_enabled": False,
            },
            {
                "arm": "BOTH",
                "candidate_screening_policy": _SCREENING_POLICY,
                "diagnostic_stem": "INFORMATION_LYAPUNOV",
                "lyapunov_enabled": True,
                "replacement_policy": _LYAPUNOV_REPLACEMENT,
                "screening_enabled": True,
            },
        ],
        "candidate_id": "C0",
        "diagnostic_ids_by_family": {
            "MOKP": {
                "BOTH": "V21E3R1_V9_INFORMATION_LYAPUNOV_MOKP",
                "LEGACY": "V21E3R1_V9_LEGACY_MOKP",
                "LYAP": "V21E3R1_V9_LYAPUNOV_MOKP",
                "SCREEN": "V21E3R1_V9_INFORMATION_SCREEN_MOKP",
            },
            "MOTSP": {
                "BOTH": "V21E3R1_V9_INFORMATION_LYAPUNOV_MOTSP",
                "LEGACY": "V21E3R1_V9_LEGACY_MOTSP",
                "LYAP": "V21E3R1_V9_LYAPUNOV_MOTSP",
                "SCREEN": "V21E3R1_V9_INFORMATION_SCREEN_MOTSP",
            },
        },
        "families": ["MOKP", "MOTSP"],
        "menu_frozen": True,
    },
    "execution_authorization": {
        "full_development_matrix": False,
        "scientific_development_claims": False,
        "single_case_smoke": True,
        "single_case_smoke_scope": "ENGINEERING_ONLY_EXPOSED_CASE_NON_SCIENTIFIC",
    },
    "execution_scope": {
        "acknowledgement_required": True,
        "case_manifest_path": (
            "ijoc_submission_v21e3/development_partitions_v1/case_manifest.json"
        ),
        "case_manifest_sha256": _DEVELOPMENT_CASE_MANIFEST_SHA256,
        "new_case_materialization_allowed": False,
        "partition": "EXPOSED_DEVELOPMENT_ONLY",
    },
    "later_phase_authorization": {
        "any_later_phase": False,
        "candidate_adoption": False,
        "confirmation": False,
        "formal_study": False,
        "ijoc_submission": False,
        "selection": False,
    },
    "protocol_id": "V21E3R1_V9R2_PREDEVELOPMENT_PROTOCOL",
    "required_artifacts": {
        "current_source_test_receipt": {
            "artifact_sha256": None,
            "blocking_condition": (
                "current_source_test_receipt_not_bound_into_a_new_protocol"
            ),
            "required_for": "FULL_DEVELOPMENT_MATRIX",
            "satisfied": False,
            "status": "MISSING",
        },
        "environment_lock": {
            "artifact_sha256": None,
            "blocking_condition": "frozen_environment_and_dependency_lock_not_supplied",
            "required_for": "FULL_DEVELOPMENT_MATRIX",
            "satisfied": False,
            "status": "MISSING",
        },
        "full_development_algorithm_spec": {
            "artifact_sha256": None,
            "blocking_condition": (
                "full_development_algorithm_parameters_caps_and_inference_not_frozen"
            ),
            "required_for": "FULL_DEVELOPMENT_MATRIX",
            "satisfied": False,
            "status": "MISSING",
        },
        "full_source_freeze": {
            "artifact_sha256": None,
            "blocking_condition": (
                "full_source_manifest_and_archive_not_bound_into_a_new_protocol"
            ),
            "required_for": "FULL_DEVELOPMENT_MATRIX",
            "satisfied": False,
            "status": "MISSING",
        },
        "independent_algorithm_decision_replay": {
            "artifact_sha256": None,
            "blocking_condition": (
                "implementation_independent_full_algorithm_decision_replay_not_supplied"
            ),
            "required_for": "FULL_DEVELOPMENT_MATRIX",
            "satisfied": False,
            "status": "MISSING",
        },
        "metric_reference_manifest": {
            "artifact_sha256": None,
            "blocking_condition": (
                "frozen_metric_and_reference_manifest_not_supplied"
            ),
            "required_for": "FULL_DEVELOPMENT_MATRIX",
            "satisfied": False,
            "status": "MISSING",
        },
        "rights_ledger": {
            "artifact_sha256": None,
            "blocking_condition": (
                "verified_data_and_solver_rights_ledger_not_supplied"
            ),
            "required_for": "FULL_DEVELOPMENT_MATRIX",
            "satisfied": False,
            "status": "MISSING",
        },
        "strong_baseline_registry": {
            "artifact_sha256": None,
            "blocking_condition": "frozen_strong_baseline_registry_not_supplied",
            "required_for": "FULL_DEVELOPMENT_MATRIX",
            "satisfied": False,
            "status": "MISSING",
        },
        "target_scale_resource_capacity_receipt": {
            "artifact_sha256": None,
            "blocking_condition": (
                "target_scale_process_tree_rss_trace_bytes_and_replay_time_not_supplied"
            ),
            "required_for": "FULL_DEVELOPMENT_MATRIX",
            "satisfied": False,
            "status": "MISSING",
        },
        "trace_replay_spec": {
            "artifact_sha256": None,
            "blocking_condition": (
                "trace_and_replay_spec_not_bound_into_a_new_protocol"
            ),
            "required_for": "FULL_DEVELOPMENT_MATRIX",
            "satisfied": False,
            "status": "MISSING",
        },
    },
    "resource_contract": {
        "caps_frozen_for_full_development_matrix": False,
        "invariants": ["A_greater_than_or_equal_to_B"],
        "quantities": {
            "A": {
                "cap_field": "attempt_cap",
                "counter_field": "attempts",
                "meaning": "attempts_submitted_to_durable_ledger",
                "value_type": "positive_exact_builtin_int",
            },
            "B": {
                "cap_field": "charged_evaluations",
                "counter_field": "first_evaluations",
                "meaning": "first_true_objective_evaluations",
                "value_type": "positive_exact_builtin_int",
            },
            "S": {
                "aggregation": (
                    "structural_candidate_generations_plus_cache_membership_probes"
                ),
                "cap_field": "structural_screening_cap",
                "components": [
                    "structural_candidate_generations",
                    "cache_membership_probes",
                ],
                "counter_field": "structural_screening_work",
                "meaning": "exact_sum_of_declared_structural_work_events",
                "value_type": "nonnegative_exact_builtin_int",
            },
            "T": {
                "cap_field": "wall_time_cap_seconds",
                "clock": "time.perf_counter",
                "counter_field": "elapsed_seconds",
                "meaning": "monotonic_elapsed_wall_time",
                "scope": "current_python_process",
                "value_type": "finite_positive_exact_builtin_int_or_float",
            },
        },
        "schema": "v21e3r1_v9_ast_resource_contract_v1",
    },
    "schema": V9_PREDEVELOPMENT_PROTOCOL_SCHEMA,
    "status": "PRE_DEVELOPMENT_HOLD",
}


class V9PredevelopmentProtocolError(ValueError):
    """Raised when a V9 pre-development contract is malformed or drifts."""


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise V9PredevelopmentProtocolError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> object:
    raise V9PredevelopmentProtocolError(
        f"non-finite JSON constant {value!r} is prohibited"
    )


def _require_exact(value: object, expected: object, *, label: str) -> None:
    if type(value) is not type(expected):
        raise V9PredevelopmentProtocolError(
            f"{label} has type {type(value).__name__}; expected "
            f"exact {type(expected).__name__}."
        )
    if isinstance(expected, dict):
        observed_mapping = value
        if any(type(key) is not str for key in observed_mapping):
            raise V9PredevelopmentProtocolError(f"{label} keys must be exact strings.")
        observed_keys = set(observed_mapping)
        expected_keys = set(expected)
        if observed_keys != expected_keys:
            raise V9PredevelopmentProtocolError(
                f"{label} keys differ: missing={sorted(expected_keys-observed_keys)}, "
                f"extra={sorted(observed_keys-expected_keys)}."
            )
        for key, expected_child in expected.items():
            _require_exact(
                observed_mapping[key],
                expected_child,
                label=f"{label}.{key}",
            )
        return
    if isinstance(expected, list):
        observed_sequence = value
        if len(observed_sequence) != len(expected):
            raise V9PredevelopmentProtocolError(
                f"{label} must contain exactly {len(expected)} items."
            )
        for index, (observed_child, expected_child) in enumerate(
            zip(observed_sequence, expected)
        ):
            _require_exact(
                observed_child,
                expected_child,
                label=f"{label}[{index}]",
            )
        return
    if value != expected:
        raise V9PredevelopmentProtocolError(
            f"{label} differs from the frozen pre-development contract."
        )


def _validated_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise V9PredevelopmentProtocolError(
            f"{label} must be a lowercase canonical SHA-256 digest."
        )
    return value


def _validated_object(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise V9PredevelopmentProtocolError(f"{label} must be an exact JSON object.")
    return value


def validate_v9_predevelopment_protocol(payload: object) -> dict[str, object]:
    """Validate the exact V9R2 protocol and return a non-authorizing certificate."""

    if type(payload) is not dict:
        raise V9PredevelopmentProtocolError(
            "protocol payload must be an exact JSON object."
        )
    expected_keys = set(_EXPECTED_PROTOCOL_CORE) | {"protocol_payload_sha256"}
    observed_keys = set(payload)
    if observed_keys != expected_keys:
        raise V9PredevelopmentProtocolError(
            "protocol payload keys differ: "
            f"missing={sorted(expected_keys-observed_keys)}, "
            f"extra={sorted(observed_keys-expected_keys)}."
        )
    declared_payload_sha256 = _validated_sha256(
        payload["protocol_payload_sha256"],
        label="protocol_payload_sha256",
    )
    core = {key: payload[key] for key in _EXPECTED_PROTOCOL_CORE}
    _require_exact(core, _EXPECTED_PROTOCOL_CORE, label="protocol")
    recomputed_payload_sha256 = _canonical_sha256(core)
    if declared_payload_sha256 != recomputed_payload_sha256:
        raise V9PredevelopmentProtocolError(
            "protocol_payload_sha256 does not bind the canonical protocol core."
        )

    required_artifacts = _validated_object(
        payload["required_artifacts"],
        label="required_artifacts",
    )
    unmet_required_artifacts = sorted(
        key
        for key, gate in required_artifacts.items()
        if isinstance(gate, dict) and gate["satisfied"] is False
    )
    execution_authorization = _validated_object(
        payload["execution_authorization"],
        label="execution_authorization",
    )
    later_phase_authorization = _validated_object(
        payload["later_phase_authorization"],
        label="later_phase_authorization",
    )
    resource_contract = _validated_object(
        payload["resource_contract"],
        label="resource_contract",
    )
    return {
        "schema": "pareto_v21e3r1_v9r2_predevelopment_validation_v1",
        "status": "PRE_DEVELOPMENT_HOLD",
        "canonical_sha256": _canonical_sha256(payload),
        "protocol_payload_sha256": declared_payload_sha256,
        "resource_contract_sha256": _canonical_sha256(resource_contract),
        "unmet_required_artifacts": unmet_required_artifacts,
        "execution_authorization": dict(execution_authorization),
        "later_phase_authorization": dict(later_phase_authorization),
        "payload": payload,
    }


def load_v9_predevelopment_protocol(
    path: str | Path | None = None,
    *,
    expected_file_sha256: str | None = None,
) -> dict[str, object]:
    """Load canonical UTF-8 JSON from package data or an explicit path."""

    if path is None:
        resource = resources.files("mo_nco")
        for component in V9_PREDEVELOPMENT_PROTOCOL_RESOURCE.split("/"):
            resource = resource.joinpath(component)
        raw = resource.read_bytes()
    else:
        if type(path) is not str and not isinstance(path, Path):
            raise V9PredevelopmentProtocolError(
                "path must be an exact str, pathlib.Path, or None."
            )
        raw = Path(path).read_bytes()
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_file_sha256 is not None:
        expected = _validated_sha256(
            expected_file_sha256,
            label="expected_file_sha256",
        )
        if file_sha256 != expected:
            raise V9PredevelopmentProtocolError(
                "protocol file SHA-256 differs from expected_file_sha256."
            )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V9PredevelopmentProtocolError(
            "protocol file must be strict UTF-8 JSON."
        ) from error
    if raw != _canonical_json_bytes(payload) + b"\n":
        raise V9PredevelopmentProtocolError(
            "protocol file must be canonical JSON followed by one newline."
        )
    certificate = validate_v9_predevelopment_protocol(payload)
    return {**certificate, "source_file_sha256": file_sha256}


def validate_v9_resource_caps(
    *,
    B: object,
    A: object,
    S: object,
    T: object,
) -> dict[str, object]:
    """Validate one prospective B/A/S/T cap tuple without authorizing a run."""

    for name, value in {"B": B, "A": A}.items():
        if type(value) is not int or value <= 0:
            raise V9PredevelopmentProtocolError(
                f"{name} must be a positive exact built-in int."
            )
    if type(S) is not int or S < 0:
        raise V9PredevelopmentProtocolError(
            "S must be a nonnegative exact built-in int."
        )
    valid_wall_time = type(T) in {int, float} and T > 0
    if valid_wall_time:
        try:
            valid_wall_time = math.isfinite(float(T))
        except OverflowError:
            valid_wall_time = False
    if not valid_wall_time:
        raise V9PredevelopmentProtocolError(
            "T must be a finite positive exact built-in int or float."
        )
    if A < B:
        raise V9PredevelopmentProtocolError("A must be at least B.")

    core: dict[str, object] = {
        "status": "RESOURCE_CAPS_VALIDATED_NOT_AUTHORIZED",
        "resource_caps": {"B": B, "A": A, "S": S, "T": T},
    }
    return {**core, "canonical_sha256": _canonical_sha256(core)}


__all__ = [
    "V9_PREDEVELOPMENT_PROTOCOL_RESOURCE",
    "V9_PREDEVELOPMENT_PROTOCOL_SCHEMA",
    "V9PredevelopmentProtocolError",
    "load_v9_predevelopment_protocol",
    "validate_v9_predevelopment_protocol",
    "validate_v9_resource_caps",
]
