"""Rowwise future-beacon authorization for Pareto-SMC v19.1.

The original multi-beacon v1 protocol XORs one shared beacon word into every
row seed.  That gives uniform *marginals* when one source is honest, but it does
not give independent rows: pairwise XORs are deterministic after commitment.
This module replaces the shared word by a complete row-indexed random vector
from every beacon source.

For row i, the 256-bit seed is

    mask_i XOR value_1[i] XOR ... XOR value_B[i].

If at least one source supplies a product-uniform row vector conditional on the
commitment and all other source vectors, the resulting seed vector is itself
product-uniform.  Local verification proves signatures, row completeness,
derivation, completion uniqueness, log-chain consistency, and witness quorum.
Future unpredictability, product-uniformity of an honest source, key isolation,
and global log non-equivocation remain external operational assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Mapping, Sequence

from .pareto_v18_study_commitment import (
    StudyCommitmentError,
    StudyRow,
    canonical_bytes,
    merkle_root,
    public_key_sha256,
    sha256_hex,
    verify_signed_mapping,
)


class RowwiseBeaconError(StudyCommitmentError):
    pass


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _unique_key_hashes(keys: Sequence[bytes]) -> tuple[str, ...]:
    hashes = tuple(public_key_sha256(key) for key in keys)
    if len(set(hashes)) != len(hashes):
        raise RowwiseBeaconError("all operational role public keys must be distinct")
    return hashes


def rowwise_beacon_commitment_payload(
    rows: Sequence[StudyRow],
    *,
    study_id: str,
    beacon_sources: Sequence[str],
    future_rounds: Sequence[int],
    coordinator_public_key: bytes,
    completion_public_key: bytes,
    log_public_key: bytes,
    beacon_public_keys: Sequence[bytes],
    witness_public_keys: Sequence[bytes],
    witness_quorum: int,
) -> dict[str, object]:
    ordered = tuple(sorted(rows))
    sources = tuple(str(x) for x in beacon_sources)
    rounds = tuple(int(x) for x in future_rounds)
    beacon_keys = tuple(beacon_public_keys)
    witness_keys = tuple(witness_public_keys)
    if not study_id or not ordered or len({row.row_id for row in ordered}) != len(ordered):
        raise RowwiseBeaconError("study ID and unique row matrix must be nonempty")
    if not sources or len(sources) != len(rounds) or len(sources) != len(beacon_keys):
        raise RowwiseBeaconError("one source, round and public key is required per beacon")
    if len(set(sources)) != len(sources) or any(x < 0 for x in rounds):
        raise RowwiseBeaconError("invalid beacon source or future round")
    if not witness_keys or not (1 <= witness_quorum <= len(witness_keys)):
        raise RowwiseBeaconError("invalid witness quorum")
    role_keys = (coordinator_public_key, completion_public_key, log_public_key) + beacon_keys + witness_keys
    hashes = _unique_key_hashes(role_keys)
    beacon_hashes = hashes[3 : 3 + len(beacon_keys)]
    witness_hashes = hashes[3 + len(beacon_keys) :]
    return {
        "schema": "pareto_smc_v19_rowwise_beacon_study_commitment_v2",
        "study_id": study_id,
        "row_count": len(ordered),
        "rows": [row.to_dict() | {"row_id": row.row_id} for row in ordered],
        "merkle_root": merkle_root(ordered),
        "seed_width_bits": 256,
        "row_randomness_contract": "rowwise_product_vector_xor_v2",
        "coordinator_public_key_sha256": hashes[0],
        "completion_public_key_sha256": hashes[1],
        "log_public_key_sha256": hashes[2],
        "beacons": [
            {
                "source": source,
                "future_round": round_id,
                "public_key_sha256": key_hash,
            }
            for source, round_id, key_hash in zip(sources, rounds, beacon_hashes, strict=True)
        ],
        "witness_public_key_sha256": list(witness_hashes),
        "witness_quorum": witness_quorum,
    }


def rowwise_beacon_payload(
    *,
    study_commitment_sha256: str,
    source: str,
    round_id: int,
    row_values: Mapping[str, str],
) -> dict[str, object]:
    if _HEX64.fullmatch(study_commitment_sha256) is None:
        raise RowwiseBeaconError("study commitment digest is malformed")
    if not source or not isinstance(round_id, int) or round_id < 0:
        raise RowwiseBeaconError("invalid beacon identity")
    ordered = []
    for row_id, value_hex in sorted(row_values.items()):
        if _HEX64.fullmatch(str(row_id)) is None or _HEX64.fullmatch(str(value_hex)) is None:
            raise RowwiseBeaconError("rowwise beacon IDs and values must be 32-byte lowercase hex")
        ordered.append({"row_id": str(row_id), "value_hex": str(value_hex)})
    if not ordered:
        raise RowwiseBeaconError("rowwise beacon vector must be nonempty")
    return {
        "schema": "pareto_smc_v19_rowwise_external_beacon_v2",
        "study_commitment_sha256": study_commitment_sha256,
        "source": source,
        "round_id": round_id,
        "row_values": ordered,
    }


def _row_map(beacon: Mapping[str, object]) -> dict[str, str]:
    raw = beacon.get("row_values")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RowwiseBeaconError("rowwise beacon vector is malformed")
    out: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise RowwiseBeaconError("rowwise beacon entry is malformed")
        row_id = str(item.get("row_id"))
        value_hex = str(item.get("value_hex"))
        if _HEX64.fullmatch(row_id) is None or _HEX64.fullmatch(value_hex) is None:
            raise RowwiseBeaconError("rowwise beacon entry is malformed")
        if row_id in out:
            raise RowwiseBeaconError("duplicate row in a beacon vector")
        out[row_id] = value_hex
    return out


def rowwise_beacon_set_sha256(beacons: Sequence[Mapping[str, object]]) -> str:
    ordered = sorted(
        (dict(beacon) for beacon in beacons),
        key=lambda item: (str(item["source"]), int(item["round_id"])),
    )
    return sha256_hex(canonical_bytes(ordered))


def derive_rowwise_seed_hex(
    commitment_payload: Mapping[str, object],
    beacons: Sequence[Mapping[str, object]],
    row_id: str,
) -> str:
    if _HEX64.fullmatch(row_id) is None:
        raise RowwiseBeaconError("row_id is malformed")
    if not beacons:
        raise RowwiseBeaconError("at least one rowwise beacon is required")
    value = int.from_bytes(
        hashlib.sha256(
            b"pareto-smc-v19-rowwise-mask-v2\x00"
            + bytes.fromhex(sha256_hex(canonical_bytes(commitment_payload)))
            + bytes.fromhex(row_id)
        ).digest(),
        "big",
    )
    for beacon in beacons:
        mapping = _row_map(beacon)
        if row_id not in mapping:
            raise RowwiseBeaconError("beacon vector omits the requested row")
        value ^= int(mapping[row_id], 16)
    return f"{value:064x}"


def rowwise_completion_payload(
    *,
    study_commitment_sha256: str,
    beacon_set_sha256_value: str,
    row_id: str,
    derived_seed_hex: str,
    status: str,
    result_sha256: str,
) -> dict[str, object]:
    if status not in {"SUCCESS", "FAILURE"}:
        raise RowwiseBeaconError("completion status must be SUCCESS or FAILURE")
    for name, value in (
        ("study_commitment_sha256", study_commitment_sha256),
        ("beacon_set_sha256", beacon_set_sha256_value),
        ("row_id", row_id),
        ("derived_seed_hex", derived_seed_hex),
        ("result_sha256", result_sha256),
    ):
        if _HEX64.fullmatch(value) is None:
            raise RowwiseBeaconError(f"{name} is malformed")
    return {
        "schema": "pareto_smc_v19_rowwise_completion_v2",
        "study_commitment_sha256": study_commitment_sha256,
        "beacon_set_sha256": beacon_set_sha256_value,
        "row_id": row_id,
        "derived_seed_hex": derived_seed_hex,
        "status": status,
        "result_sha256": result_sha256,
    }


def rowwise_witness_payload(
    *,
    study_commitment_sha256: str,
    final_log_root_sha256: str,
    record_count: int,
    witness_id: str,
) -> dict[str, object]:
    if _HEX64.fullmatch(study_commitment_sha256) is None or _HEX64.fullmatch(final_log_root_sha256) is None:
        raise RowwiseBeaconError("witness digest is malformed")
    if not witness_id or not isinstance(record_count, int) or record_count <= 0:
        raise RowwiseBeaconError("invalid witness statement")
    return {
        "schema": "pareto_smc_v19_rowwise_log_witness_v2",
        "study_commitment_sha256": study_commitment_sha256,
        "final_log_root_sha256": final_log_root_sha256,
        "record_count": record_count,
        "witness_id": witness_id,
    }


def _verify_log_chain(
    signed_records: Sequence[Mapping[str, object]],
    signed_log_chain: Sequence[Mapping[str, object]],
    log_public_key: bytes,
) -> str:
    if len(signed_records) != len(signed_log_chain) or not signed_log_chain:
        raise RowwiseBeaconError("transparency log length mismatch")
    previous = "0" * 64
    for index, (record, envelope) in enumerate(zip(signed_records, signed_log_chain, strict=True)):
        payload = verify_signed_mapping(envelope, log_public_key)
        if payload.get("index") != index or payload.get("previous_record_sha256") != previous:
            raise RowwiseBeaconError("transparency log chain is discontinuous")
        if payload.get("signed_record_sha256") != sha256_hex(canonical_bytes(record)):
            raise RowwiseBeaconError("transparency log record binding mismatch")
        previous = sha256_hex(canonical_bytes(envelope))
    return previous


@dataclass(frozen=True)
class RowwiseBeaconStudyAudit:
    study_id: str
    row_count: int
    beacon_count: int
    witness_count: int
    witness_quorum: int
    success_count: int
    failure_count: int
    all_beacon_vectors_complete: bool
    all_rows_completed_once: bool
    witness_quorum_pass: bool
    local_verdict: str
    external_assumptions: tuple[str, ...]
    final_log_root_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "pareto_smc_v19_rowwise_beacon_study_audit_v2",
            "study_id": self.study_id,
            "row_count": self.row_count,
            "beacon_count": self.beacon_count,
            "witness_count": self.witness_count,
            "witness_quorum": self.witness_quorum,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "all_beacon_vectors_complete": self.all_beacon_vectors_complete,
            "all_rows_completed_once": self.all_rows_completed_once,
            "witness_quorum_pass": self.witness_quorum_pass,
            "local_verdict": self.local_verdict,
            "external_assumptions": list(self.external_assumptions),
            "final_log_root_sha256": self.final_log_root_sha256,
        }


def verify_rowwise_beacon_study(
    *,
    signed_commitment: Mapping[str, object],
    signed_beacons: Sequence[Mapping[str, object]],
    signed_completions: Sequence[Mapping[str, object]],
    signed_log_chain: Sequence[Mapping[str, object]],
    signed_witnesses: Sequence[Mapping[str, object]],
    coordinator_public_key: bytes,
    completion_public_key: bytes,
    log_public_key: bytes,
    beacon_public_keys: Sequence[bytes],
    witness_public_keys: Sequence[bytes],
) -> RowwiseBeaconStudyAudit:
    commitment = verify_signed_mapping(signed_commitment, coordinator_public_key)
    if commitment.get("schema") != "pareto_smc_v19_rowwise_beacon_study_commitment_v2":
        raise RowwiseBeaconError("unsupported rowwise commitment schema")
    if commitment.get("seed_width_bits") != 256 or commitment.get("row_randomness_contract") != "rowwise_product_vector_xor_v2":
        raise RowwiseBeaconError("rowwise randomness contract mismatch")
    beacon_keys = tuple(beacon_public_keys)
    witness_keys = tuple(witness_public_keys)
    expected_hashes = _unique_key_hashes(
        (coordinator_public_key, completion_public_key, log_public_key) + beacon_keys + witness_keys
    )
    expected_beacon_records = commitment.get("beacons")
    if not isinstance(expected_beacon_records, Sequence) or isinstance(expected_beacon_records, (str, bytes)):
        raise RowwiseBeaconError("committed beacon list is malformed")
    if len(expected_beacon_records) != len(beacon_keys) or len(signed_beacons) != len(beacon_keys):
        raise RowwiseBeaconError("beacon count mismatch")
    if commitment.get("coordinator_public_key_sha256") != expected_hashes[0] or commitment.get("completion_public_key_sha256") != expected_hashes[1] or commitment.get("log_public_key_sha256") != expected_hashes[2]:
        raise RowwiseBeaconError("operational role key mismatch")
    if list(commitment.get("witness_public_key_sha256", [])) != list(expected_hashes[3 + len(beacon_keys) :]):
        raise RowwiseBeaconError("witness key list mismatch")
    quorum = int(commitment.get("witness_quorum", 0))
    if not (1 <= quorum <= len(witness_keys)):
        raise RowwiseBeaconError("invalid committed witness quorum")

    rows_raw = commitment.get("rows")
    if not isinstance(rows_raw, Sequence) or isinstance(rows_raw, (str, bytes)):
        raise RowwiseBeaconError("committed row matrix is malformed")
    reconstructed_rows = []
    row_ids_list = []
    for item in rows_raw:
        if not isinstance(item, Mapping):
            raise RowwiseBeaconError("committed row entry is malformed")
        row = StudyRow(
            case_id=str(item.get("case_id")),
            algorithm_id=str(item.get("algorithm_id")),
            replicate_id=str(item.get("replicate_id")),
            budget=int(item.get("budget", -1)),
            configuration_sha256=str(item.get("configuration_sha256")),
        )
        declared_row_id = str(item.get("row_id"))
        if declared_row_id != row.row_id:
            raise RowwiseBeaconError("committed row payload does not match its row_id")
        reconstructed_rows.append(row)
        row_ids_list.append(row.row_id)
    row_ids = tuple(row_ids_list)
    if len(row_ids) != int(commitment.get("row_count", -1)) or len(set(row_ids)) != len(row_ids):
        raise RowwiseBeaconError("committed row IDs are malformed or duplicated")
    if commitment.get("merkle_root") != merkle_root(tuple(reconstructed_rows)):
        raise RowwiseBeaconError("committed Merkle root does not match the row matrix")
    row_set = set(row_ids)
    commitment_sha = sha256_hex(canonical_bytes(commitment))

    beacon_payloads = []
    sources_seen = set()
    for envelope, key, expected_record in zip(signed_beacons, beacon_keys, expected_beacon_records, strict=True):
        if not isinstance(expected_record, Mapping):
            raise RowwiseBeaconError("committed beacon record is malformed")
        payload = verify_signed_mapping(envelope, key)
        if payload.get("schema") != "pareto_smc_v19_rowwise_external_beacon_v2":
            raise RowwiseBeaconError("unsupported rowwise beacon schema")
        if payload.get("study_commitment_sha256") != commitment_sha:
            raise RowwiseBeaconError("beacon does not bind the study commitment")
        if payload.get("source") != expected_record.get("source") or payload.get("round_id") != expected_record.get("future_round"):
            raise RowwiseBeaconError("beacon source or round mismatch")
        if expected_record.get("public_key_sha256") != public_key_sha256(key):
            raise RowwiseBeaconError("beacon key mismatch")
        if payload.get("source") in sources_seen:
            raise RowwiseBeaconError("duplicate beacon source")
        sources_seen.add(payload.get("source"))
        mapping = _row_map(payload)
        if set(mapping) != row_set:
            raise RowwiseBeaconError("every beacon vector must contain every committed row exactly once")
        beacon_payloads.append(payload)
    beacon_set_hash = rowwise_beacon_set_sha256(beacon_payloads)

    completion_by_row: dict[str, Mapping[str, object]] = {}
    success = failure = 0
    for envelope in signed_completions:
        payload = verify_signed_mapping(envelope, completion_public_key)
        if payload.get("schema") != "pareto_smc_v19_rowwise_completion_v2":
            raise RowwiseBeaconError("unsupported rowwise completion schema")
        row_id = str(payload.get("row_id"))
        if row_id not in row_set or row_id in completion_by_row:
            raise RowwiseBeaconError("unknown or duplicate completion row")
        if payload.get("study_commitment_sha256") != commitment_sha or payload.get("beacon_set_sha256") != beacon_set_hash:
            raise RowwiseBeaconError("completion binding mismatch")
        expected_seed = derive_rowwise_seed_hex(commitment, beacon_payloads, row_id)
        if payload.get("derived_seed_hex") != expected_seed:
            raise RowwiseBeaconError("completion seed derivation mismatch")
        if _HEX64.fullmatch(str(payload.get("result_sha256"))) is None:
            raise RowwiseBeaconError("completion result SHA-256 is malformed")
        status = payload.get("status")
        if status == "SUCCESS":
            success += 1
        elif status == "FAILURE":
            failure += 1
        else:
            raise RowwiseBeaconError("invalid completion status")
        completion_by_row[row_id] = payload
    all_completed = set(completion_by_row) == row_set
    if not all_completed:
        raise RowwiseBeaconError("every committed row needs exactly one completion record")

    signed_records = [signed_commitment, *signed_beacons, *signed_completions]
    final_root = _verify_log_chain(signed_records, signed_log_chain, log_public_key)
    if len(signed_witnesses) > len(witness_keys):
        raise RowwiseBeaconError("more witness statements were supplied than committed witness keys")
    accepted_witnesses = set()
    for envelope, key in zip(signed_witnesses, witness_keys, strict=False):
        payload = verify_signed_mapping(envelope, key)
        if payload.get("schema") != "pareto_smc_v19_rowwise_log_witness_v2":
            raise RowwiseBeaconError("unsupported witness schema")
        if payload.get("study_commitment_sha256") != commitment_sha or payload.get("final_log_root_sha256") != final_root or payload.get("record_count") != len(signed_records):
            raise RowwiseBeaconError("witness statement binding mismatch")
        witness_id = str(payload.get("witness_id"))
        if witness_id in accepted_witnesses:
            raise RowwiseBeaconError("duplicate witness identity")
        accepted_witnesses.add(witness_id)
    witness_pass = len(accepted_witnesses) >= quorum
    if not witness_pass:
        raise RowwiseBeaconError("witness quorum not met")
    return RowwiseBeaconStudyAudit(
        study_id=str(commitment.get("study_id")),
        row_count=len(row_ids),
        beacon_count=len(beacon_payloads),
        witness_count=len(accepted_witnesses),
        witness_quorum=quorum,
        success_count=success,
        failure_count=failure,
        all_beacon_vectors_complete=True,
        all_rows_completed_once=all_completed,
        witness_quorum_pass=witness_pass,
        local_verdict="PASS_CONDITIONAL_ROW_PRODUCT_UNIFORMITY_FUTURE_UNPREDICTABILITY_KEY_ISOLATION_AND_LOG_NON_EQUIVOCATION",
        external_assumptions=(
            "at least one beacon source supplies a rowwise product-uniform vector conditional on the commitment and all other beacon vectors",
            "the honest rowwise vector is unpredictable before the commitment",
            "operational private keys are isolated according to their declared roles",
            "the transparency log and witness ecosystem do not equivocate globally",
            "all actual executions are represented by the committed completion records",
        ),
        final_log_root_sha256=final_root,
    )


__all__ = [
    "RowwiseBeaconError",
    "RowwiseBeaconStudyAudit",
    "derive_rowwise_seed_hex",
    "rowwise_beacon_commitment_payload",
    "rowwise_beacon_payload",
    "rowwise_beacon_set_sha256",
    "rowwise_completion_payload",
    "rowwise_witness_payload",
    "verify_rowwise_beacon_study",
]
