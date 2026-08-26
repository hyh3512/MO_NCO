from __future__ import annotations

"""Strict, hash-addressed publication protocol specification for v13.

The v13 file does not replace either the Pareto-SMC algorithm specification or
the fixed-anchor certificate specification.  It binds those frozen artifacts
to the additional pilot-freeze authorization, sparse-reference bridge,
domain-separated seeds, assignment feasibility ledger, and full-type-sweep
checkpoint contract.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple


V13_PROTOCOL_SPECIFICATION_SCHEMA = (
    "pareto_smc_v13_publication_protocol_specification_v1"
)
V13_ALGORITHM_ID = "pareto-smc-v13"
V13_RECEIPT_POLICY = "external_ed25519_preconfirm_authorization_v1"
_RECEIPT_IDENTITY = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}$"
)
_SEED_CASE_IDENTITY = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,127}$"
)


@dataclass(frozen=True)
class V13ProtocolSpecification:
    path: Path
    sha256: str
    instance_sha256: str
    pareto_smc_specification_sha256: str
    anchor_certificate_specification_sha256: str
    full_reference_certificate_specification_sha256: str
    sparse_compression_certificate_sha256: str
    run_id: str
    case_id: str
    algorithm_id: str
    signer_key_id: str
    signer_public_key_raw: bytes
    receipt_authorization_policy: str
    seed_derivation_schema: str
    requested_full_sweep_checkpoints: Tuple[int, ...]
    desired_target_mass_lower_bounds_by_anchor_cell: Tuple[float, ...]
    pilot_failure_budgets_by_anchor_cell: Tuple[float, ...]
    confirm_failure_budgets_by_anchor_cell: Tuple[float, ...]
    mutually_exclusive_anchor_cells: bool


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _reject_duplicate_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"Duplicate JSON field is forbidden: {key!r}."
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}.")


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} has an unexpected shape; "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}."
        )


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string.")
    if value != value.strip():
        raise ValueError(f"{label} must not contain boundary whitespace.")
    return value


def _canonical_identity(
    value: object,
    label: str,
    *,
    pattern: re.Pattern[str],
) -> str:
    result = _nonempty_string(value, label)
    if pattern.fullmatch(result) is None:
        raise ValueError(f"{label} is not a canonical protocol identity.")
    return result


def _strict_probability(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a JSON number.")
    result = float(value)
    if not math.isfinite(result) or not (0.0 < result < 1.0):
        raise ValueError(f"{label} must lie strictly between zero and one.")
    return result


def load_v13_protocol_specification(
    path: str | Path,
) -> V13ProtocolSpecification:
    """Load one exact-shape v13 protocol file and bind its raw-byte hash."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"v13 protocol specification is missing: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            "v13 protocol specification is not valid UTF-8 JSON."
        ) from error
    root = _mapping(payload, "v13 protocol specification")
    _exact_keys(
        root,
        {
            "schema",
            "bindings",
            "identity",
            "receipt_authority",
            "seed_derivation",
            "full_sweep_checkpoints",
            "assignment_preflight",
        },
        "v13 protocol specification",
    )
    if root["schema"] != V13_PROTOCOL_SPECIFICATION_SCHEMA:
        raise ValueError("Unexpected v13 protocol specification schema.")

    bindings = _mapping(root["bindings"], "bindings")
    _exact_keys(
        bindings,
        {
            "instance_sha256",
            "pareto_smc_specification_sha256",
            "anchor_certificate_specification_sha256",
            "full_reference_certificate_specification_sha256",
            "sparse_compression_certificate_sha256",
        },
        "bindings",
    )
    identity = _mapping(root["identity"], "identity")
    _exact_keys(
        identity,
        {"run_id", "case_id", "algorithm_id"},
        "identity",
    )
    algorithm_id = _nonempty_string(
        identity["algorithm_id"],
        "identity.algorithm_id",
    )
    if algorithm_id != V13_ALGORITHM_ID:
        raise ValueError(
            f"identity.algorithm_id must equal {V13_ALGORITHM_ID!r}."
        )

    authority = _mapping(
        root["receipt_authority"],
        "receipt_authority",
    )
    _exact_keys(
        authority,
        {
            "signer_key_id",
            "ed25519_public_key_hex",
            "authorization_policy",
        },
        "receipt_authority",
    )
    public_key_hex = _nonempty_string(
        authority["ed25519_public_key_hex"],
        "receipt_authority.ed25519_public_key_hex",
    )
    try:
        public_key_raw = bytes.fromhex(public_key_hex)
    except ValueError as error:
        raise ValueError(
            "receipt_authority.ed25519_public_key_hex must be hexadecimal."
        ) from error
    if len(public_key_raw) != 32 or public_key_hex != public_key_raw.hex():
        raise ValueError(
            "receipt_authority.ed25519_public_key_hex must be a canonical "
            "32-byte Ed25519 public key."
        )
    receipt_policy = _nonempty_string(
        authority["authorization_policy"],
        "receipt_authority.authorization_policy",
    )
    if receipt_policy != V13_RECEIPT_POLICY:
        raise ValueError(
            "Unsupported receipt_authority.authorization_policy."
        )

    seed_derivation = _mapping(
        root["seed_derivation"],
        "seed_derivation",
    )
    _exact_keys(seed_derivation, {"schema"}, "seed_derivation")
    seed_schema = _nonempty_string(
        seed_derivation["schema"],
        "seed_derivation.schema",
    )

    checkpoint_block = _mapping(
        root["full_sweep_checkpoints"],
        "full_sweep_checkpoints",
    )
    _exact_keys(
        checkpoint_block,
        {"evaluation_counts"},
        "full_sweep_checkpoints",
    )
    raw_checkpoints = checkpoint_block["evaluation_counts"]
    if not isinstance(raw_checkpoints, list) or not raw_checkpoints:
        raise ValueError(
            "full_sweep_checkpoints.evaluation_counts must be a nonempty "
            "array."
        )
    checkpoints = tuple(raw_checkpoints)
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        for value in checkpoints
    ):
        raise ValueError(
            "Full-sweep checkpoint counts must be positive integers."
        )
    if checkpoints != tuple(sorted(set(checkpoints))):
        raise ValueError(
            "Full-sweep checkpoint counts must be strictly increasing."
        )

    assignment = _mapping(
        root["assignment_preflight"],
        "assignment_preflight",
    )
    _exact_keys(
        assignment,
        {
            "desired_target_mass_lower_bounds_by_anchor_cell",
            "pilot_failure_budgets_by_anchor_cell",
            "confirm_failure_budgets_by_anchor_cell",
            "mutually_exclusive_anchor_cells",
        },
        "assignment_preflight",
    )
    raw_desired = assignment[
        "desired_target_mass_lower_bounds_by_anchor_cell"
    ]
    raw_budgets = assignment[
        "pilot_failure_budgets_by_anchor_cell"
    ]
    raw_confirm_budgets = assignment[
        "confirm_failure_budgets_by_anchor_cell"
    ]
    if (
        not isinstance(raw_desired, list)
        or not raw_desired
        or not isinstance(raw_budgets, list)
        or len(raw_budgets) != len(raw_desired)
        or not isinstance(raw_confirm_budgets, list)
        or len(raw_confirm_budgets) != len(raw_desired)
    ):
        raise ValueError(
            "Assignment desired masses, pilot budgets, and confirm budgets "
            "must have equal positive length."
        )
    desired = tuple(
        _strict_probability(value, f"desired mass[{index}]")
        for index, value in enumerate(raw_desired)
    )
    budgets = tuple(
        _strict_probability(value, f"pilot failure budget[{index}]")
        for index, value in enumerate(raw_budgets)
    )
    confirm_budgets = tuple(
        _strict_probability(value, f"confirm failure budget[{index}]")
        for index, value in enumerate(raw_confirm_budgets)
    )
    if assignment["mutually_exclusive_anchor_cells"] is not True:
        raise ValueError(
            "mutually_exclusive_anchor_cells must be explicitly true."
        )

    return V13ProtocolSpecification(
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        instance_sha256=_sha256(
            bindings["instance_sha256"],
            "bindings.instance_sha256",
        ),
        pareto_smc_specification_sha256=_sha256(
            bindings["pareto_smc_specification_sha256"],
            "bindings.pareto_smc_specification_sha256",
        ),
        anchor_certificate_specification_sha256=_sha256(
            bindings["anchor_certificate_specification_sha256"],
            "bindings.anchor_certificate_specification_sha256",
        ),
        full_reference_certificate_specification_sha256=_sha256(
            bindings["full_reference_certificate_specification_sha256"],
            "bindings.full_reference_certificate_specification_sha256",
        ),
        sparse_compression_certificate_sha256=_sha256(
            bindings["sparse_compression_certificate_sha256"],
            "bindings.sparse_compression_certificate_sha256",
        ),
        run_id=_canonical_identity(
            identity["run_id"],
            "identity.run_id",
            pattern=_RECEIPT_IDENTITY,
        ),
        case_id=_canonical_identity(
            identity["case_id"],
            "identity.case_id",
            pattern=_SEED_CASE_IDENTITY,
        ),
        algorithm_id=algorithm_id,
        signer_key_id=_canonical_identity(
            authority["signer_key_id"],
            "receipt_authority.signer_key_id",
            pattern=_RECEIPT_IDENTITY,
        ),
        signer_public_key_raw=public_key_raw,
        receipt_authorization_policy=receipt_policy,
        seed_derivation_schema=seed_schema,
        requested_full_sweep_checkpoints=checkpoints,
        desired_target_mass_lower_bounds_by_anchor_cell=desired,
        pilot_failure_budgets_by_anchor_cell=budgets,
        confirm_failure_budgets_by_anchor_cell=confirm_budgets,
        mutually_exclusive_anchor_cells=True,
    )
