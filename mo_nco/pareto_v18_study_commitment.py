"""Study-level future-beacon commitment and completeness audit for v18.

This module closes the *local* part of the selective-restart protocol.  It binds
an entire case x algorithm x replicate matrix before a future beacon, derives
one seed per row, and requires exactly one signed success/failure completion for
every committed row.  The final verdict remains conditional on two operational
facts that bytes alone cannot prove: future-beacon unpredictability and global
transparency-log non-equivocation.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Mapping, Sequence

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
except ImportError as exc:  # pragma: no cover
    InvalidSignature = None  # type: ignore[assignment]
    Ed25519PrivateKey = None  # type: ignore[assignment,misc]
    Ed25519PublicKey = None  # type: ignore[assignment,misc]
    serialization = None  # type: ignore[assignment]
    _IMPORT_ERROR: ImportError | None = exc
else:
    _IMPORT_ERROR = None


class StudyCommitmentError(ValueError):
    pass


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _require_crypto() -> None:
    if _IMPORT_ERROR is not None:
        raise StudyCommitmentError("cryptography is required") from _IMPORT_ERROR


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StudyCommitmentError("payload is not canonical-JSON serializable") from exc


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def public_key_raw(private_key: "Ed25519PrivateKey") -> bytes:
    _require_crypto()
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def public_key_sha256(public_key: bytes) -> str:
    if not isinstance(public_key, bytes) or len(public_key) != 32:
        raise StudyCommitmentError("Ed25519 public key must contain 32 raw bytes")
    return sha256_hex(public_key)


def sign_mapping(payload: Mapping[str, object], private_key: "Ed25519PrivateKey") -> dict[str, object]:
    _require_crypto()
    raw = canonical_bytes(payload)
    signature = private_key.sign(raw)
    return {
        "payload": dict(payload),
        "payload_sha256": sha256_hex(raw),
        "signature_algorithm": "Ed25519",
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }


def verify_signed_mapping(
    envelope: Mapping[str, object],
    public_key: bytes,
) -> Mapping[str, object]:
    _require_crypto()
    if set(envelope) != {
        "payload",
        "payload_sha256",
        "signature_algorithm",
        "signature_base64",
    }:
        raise StudyCommitmentError("signed envelope has an invalid field set")
    payload = envelope["payload"]
    if not isinstance(payload, Mapping):
        raise StudyCommitmentError("signed payload must be a mapping")
    raw = canonical_bytes(payload)
    if envelope["payload_sha256"] != sha256_hex(raw):
        raise StudyCommitmentError("signed payload SHA-256 mismatch")
    if envelope["signature_algorithm"] != "Ed25519":
        raise StudyCommitmentError("unsupported signature algorithm")
    try:
        signature = base64.b64decode(str(envelope["signature_base64"]), validate=True)
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, raw)
    except (ValueError, InvalidSignature, binascii.Error) as exc:  # type: ignore[arg-type]
        raise StudyCommitmentError("invalid Ed25519 signature") from exc
    return payload


@dataclass(frozen=True, order=True)
class StudyRow:
    case_id: str
    algorithm_id: str
    replicate_id: str
    budget: int
    configuration_sha256: str

    def __post_init__(self) -> None:
        if not self.case_id or not self.algorithm_id or not self.replicate_id:
            raise StudyCommitmentError("study row identities must be nonempty")
        if not isinstance(self.budget, int) or self.budget <= 0:
            raise StudyCommitmentError("study row budget must be positive")
        if not isinstance(self.configuration_sha256, str) or _HEX64.fullmatch(self.configuration_sha256) is None:
            raise StudyCommitmentError("configuration_sha256 is malformed")

    @property
    def row_id(self) -> str:
        return sha256_hex(canonical_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "algorithm_id": self.algorithm_id,
            "replicate_id": self.replicate_id,
            "budget": self.budget,
            "configuration_sha256": self.configuration_sha256,
        }


def leaf_hash(row: StudyRow) -> bytes:
    return hashlib.sha256(b"pareto-v18-study-leaf\x00" + canonical_bytes(row.to_dict())).digest()


def merkle_root(rows: Sequence[StudyRow]) -> str:
    ordered = tuple(sorted(rows))
    if not ordered or len({row.row_id for row in ordered}) != len(ordered):
        raise StudyCommitmentError("study rows must be nonempty and unique")
    level = [leaf_hash(row) for row in ordered]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(b"pareto-v18-study-node\x00" + level[i] + level[i + 1]).digest()
            for i in range(0, len(level), 2)
        ]
    return level[0].hex()


def study_commitment_payload(
    rows: Sequence[StudyRow],
    *,
    study_id: str,
    beacon_source: str,
    future_round: int,
    coordinator_public_key: bytes,
    beacon_public_key: bytes,
    completion_public_key: bytes,
    log_public_key: bytes,
) -> dict[str, object]:
    ordered = tuple(sorted(rows))
    key_hashes = [
        public_key_sha256(key)
        for key in (
            coordinator_public_key,
            beacon_public_key,
            completion_public_key,
            log_public_key,
        )
    ]
    if len(set(key_hashes)) != 4:
        raise StudyCommitmentError("coordinator, beacon, completion, and log keys must be distinct")
    if not study_id or not beacon_source or not isinstance(future_round, int) or future_round < 0:
        raise StudyCommitmentError("invalid study or beacon identity")
    return {
        "schema": "pareto_smc_v18_study_commitment_v1",
        "study_id": study_id,
        "row_count": len(ordered),
        "rows": [row.to_dict() | {"row_id": row.row_id} for row in ordered],
        "merkle_root": merkle_root(ordered),
        "beacon_source": beacon_source,
        "future_round": future_round,
        "coordinator_public_key_sha256": key_hashes[0],
        "beacon_public_key_sha256": key_hashes[1],
        "completion_public_key_sha256": key_hashes[2],
        "log_public_key_sha256": key_hashes[3],
    }


def beacon_payload(
    *,
    study_commitment_sha256: str,
    beacon_source: str,
    round_id: int,
    value_hex: str,
) -> dict[str, object]:
    if _HEX64.fullmatch(study_commitment_sha256) is None:
        raise StudyCommitmentError("study commitment digest is malformed")
    if _HEX64.fullmatch(value_hex) is None:
        raise StudyCommitmentError("beacon value must be a 32-byte lowercase hex string")
    return {
        "schema": "pareto_smc_v18_external_beacon_v1",
        "study_commitment_sha256": study_commitment_sha256,
        "beacon_source": beacon_source,
        "round_id": round_id,
        "value_hex": value_hex,
    }


def derive_row_seed(
    commitment_payload: Mapping[str, object],
    beacon: Mapping[str, object],
    row_id: str,
) -> int:
    if _HEX64.fullmatch(row_id) is None:
        raise StudyCommitmentError("row_id is malformed")
    material = (
        b"pareto-smc-v18-row-seed\x00"
        + bytes.fromhex(sha256_hex(canonical_bytes(commitment_payload)))
        + bytes.fromhex(str(beacon["value_hex"]))
        + bytes.fromhex(row_id)
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def completion_payload(
    *,
    study_commitment_sha256: str,
    beacon_payload_sha256: str,
    row_id: str,
    derived_seed: int,
    status: str,
    result_sha256: str,
) -> dict[str, object]:
    if status not in {"SUCCESS", "FAILURE"}:
        raise StudyCommitmentError("completion status must be SUCCESS or FAILURE")
    for name, value in (
        ("study_commitment_sha256", study_commitment_sha256),
        ("beacon_payload_sha256", beacon_payload_sha256),
        ("row_id", row_id),
        ("result_sha256", result_sha256),
    ):
        if _HEX64.fullmatch(value) is None:
            raise StudyCommitmentError(f"{name} is malformed")
    if not isinstance(derived_seed, int) or derived_seed < 0 or derived_seed >= 2**64:
        raise StudyCommitmentError("derived seed must be an unsigned 64-bit integer")
    return {
        "schema": "pareto_smc_v18_row_completion_v1",
        "study_commitment_sha256": study_commitment_sha256,
        "beacon_payload_sha256": beacon_payload_sha256,
        "row_id": row_id,
        "derived_seed": derived_seed,
        "status": status,
        "result_sha256": result_sha256,
    }


def build_log_chain(
    signed_records: Sequence[Mapping[str, object]],
    log_private_key: "Ed25519PrivateKey",
) -> list[dict[str, object]]:
    chain: list[dict[str, object]] = []
    previous = "0" * 64
    for index, signed_record in enumerate(signed_records):
        record_payload = {
            "schema": "pareto_smc_v18_transparency_log_record_v1",
            "index": index,
            "previous_record_sha256": previous,
            "signed_record_sha256": sha256_hex(canonical_bytes(signed_record)),
        }
        envelope = sign_mapping(record_payload, log_private_key)
        chain.append(envelope)
        previous = sha256_hex(canonical_bytes(envelope))
    return chain


@dataclass(frozen=True)
class StudyExecutionAudit:
    study_id: str
    row_count: int
    completed_row_count: int
    merkle_root: str
    completion_status_counts: tuple[tuple[str, int], ...]
    all_rows_completed_once: bool
    signatures_valid: bool
    log_chain_valid: bool
    local_verdict: str
    external_assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "pareto_smc_v18_study_execution_audit_v1",
            "study_id": self.study_id,
            "row_count": self.row_count,
            "completed_row_count": self.completed_row_count,
            "merkle_root": self.merkle_root,
            "completion_status_counts": dict(self.completion_status_counts),
            "all_rows_completed_once": self.all_rows_completed_once,
            "signatures_valid": self.signatures_valid,
            "log_chain_valid": self.log_chain_valid,
            "local_verdict": self.local_verdict,
            "external_assumptions": list(self.external_assumptions),
        }


def verify_study_execution(
    *,
    signed_commitment: Mapping[str, object],
    signed_beacon: Mapping[str, object],
    signed_completions: Sequence[Mapping[str, object]],
    signed_log_chain: Sequence[Mapping[str, object]],
    coordinator_public_key: bytes,
    beacon_public_key: bytes,
    completion_public_key: bytes,
    log_public_key: bytes,
) -> StudyExecutionAudit:
    commitment = verify_signed_mapping(signed_commitment, coordinator_public_key)
    beacon = verify_signed_mapping(signed_beacon, beacon_public_key)
    expected_key_hashes = {
        "coordinator_public_key_sha256": public_key_sha256(coordinator_public_key),
        "beacon_public_key_sha256": public_key_sha256(beacon_public_key),
        "completion_public_key_sha256": public_key_sha256(completion_public_key),
        "log_public_key_sha256": public_key_sha256(log_public_key),
    }
    for field, expected in expected_key_hashes.items():
        if commitment.get(field) != expected:
            raise StudyCommitmentError(f"study commitment {field} mismatch")
    if len(set(expected_key_hashes.values())) != 4:
        raise StudyCommitmentError("study execution role keys are not distinct")
    commitment_sha = sha256_hex(canonical_bytes(commitment))
    beacon_sha = sha256_hex(canonical_bytes(beacon))
    if beacon.get("study_commitment_sha256") != commitment_sha:
        raise StudyCommitmentError("beacon does not bind the study commitment")
    if beacon.get("beacon_source") != commitment.get("beacon_source"):
        raise StudyCommitmentError("beacon source mismatch")
    if beacon.get("round_id") != commitment.get("future_round"):
        raise StudyCommitmentError("beacon round mismatch")

    rows_raw = commitment.get("rows")
    if not isinstance(rows_raw, Sequence) or isinstance(rows_raw, (str, bytes)):
        raise StudyCommitmentError("committed rows are malformed")
    rows = []
    row_by_id: dict[str, StudyRow] = {}
    for item in rows_raw:
        if not isinstance(item, Mapping):
            raise StudyCommitmentError("committed row is malformed")
        row = StudyRow(
            case_id=str(item["case_id"]),
            algorithm_id=str(item["algorithm_id"]),
            replicate_id=str(item["replicate_id"]),
            budget=int(item["budget"]),
            configuration_sha256=str(item["configuration_sha256"]),
        )
        if item.get("row_id") != row.row_id or row.row_id in row_by_id:
            raise StudyCommitmentError("row ID mismatch or duplicate")
        rows.append(row)
        row_by_id[row.row_id] = row
    if commitment.get("row_count") != len(rows) or commitment.get("merkle_root") != merkle_root(rows):
        raise StudyCommitmentError("study matrix root or count mismatch")

    completions: dict[str, Mapping[str, object]] = {}
    status_counts = {"SUCCESS": 0, "FAILURE": 0}
    for envelope in signed_completions:
        payload = verify_signed_mapping(envelope, completion_public_key)
        row_id = str(payload.get("row_id"))
        if row_id not in row_by_id or row_id in completions:
            raise StudyCommitmentError("completion row is missing from the matrix or duplicated")
        if payload.get("study_commitment_sha256") != commitment_sha or payload.get("beacon_payload_sha256") != beacon_sha:
            raise StudyCommitmentError("completion binding mismatch")
        expected_seed = derive_row_seed(commitment, beacon, row_id)
        if payload.get("derived_seed") != expected_seed:
            raise StudyCommitmentError("completion seed derivation mismatch")
        status = str(payload.get("status"))
        if status not in status_counts:
            raise StudyCommitmentError("completion status is invalid")
        status_counts[status] += 1
        completions[row_id] = payload

    if len(signed_log_chain) != 2 + len(signed_completions):
        raise StudyCommitmentError("transparency log must contain commitment, beacon, and every completion")
    expected_signed_records = [signed_commitment, signed_beacon, *signed_completions]
    previous = "0" * 64
    for index, (log_envelope, signed_record) in enumerate(zip(signed_log_chain, expected_signed_records, strict=True)):
        payload = verify_signed_mapping(log_envelope, log_public_key)
        if payload.get("index") != index or payload.get("previous_record_sha256") != previous:
            raise StudyCommitmentError("transparency log order or hash chain is invalid")
        if payload.get("signed_record_sha256") != sha256_hex(canonical_bytes(signed_record)):
            raise StudyCommitmentError("transparency log record binding mismatch")
        previous = sha256_hex(canonical_bytes(log_envelope))

    complete = set(completions) == set(row_by_id)
    verdict = (
        "PASS_CONDITIONAL_EXTERNAL_FUTURE_UNPREDICTABILITY_AND_LOG_NON_EQUIVOCATION"
        if complete
        else "FAIL_INCOMPLETE_STUDY_MATRIX"
    )
    return StudyExecutionAudit(
        study_id=str(commitment["study_id"]),
        row_count=len(rows),
        completed_row_count=len(completions),
        merkle_root=str(commitment["merkle_root"]),
        completion_status_counts=tuple(sorted(status_counts.items())),
        all_rows_completed_once=complete,
        signatures_valid=True,
        log_chain_valid=True,
        local_verdict=verdict,
        external_assumptions=(
            "beacon_value_was_unpredictable_before_commitment",
            "transparency_log_is_globally_non_equivocating",
            "role_private_keys_were_operationally_separated",
        ),
    )


__all__ = [
    "StudyCommitmentError",
    "StudyExecutionAudit",
    "StudyRow",
    "beacon_payload",
    "build_log_chain",
    "canonical_bytes",
    "completion_payload",
    "derive_row_seed",
    "merkle_root",
    "public_key_raw",
    "sha256_hex",
    "sign_mapping",
    "study_commitment_payload",
    "verify_study_execution",
]
