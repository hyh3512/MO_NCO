"""Multi-beacon and witness-quorum study authorization for Pareto-SMC v19.

The v18 future-beacon protocol still depended on one beacon and one log.  V19
freezes multiple future beacon sources and a quorum of independent log
witnesses.  A row seed is the XOR of the first 64 bits of all *shared* beacon
values and a deterministic row mask.  Conditional on the commitment and all
other beacon values, an honest independent uniform source makes every row seed
marginally uniform.  It does **not** make different row seeds independent:
for two rows their XOR is a deterministic function of the committed masks.
Use ``pareto_v19_rowwise_beacon`` when a product-random row vector is required.
A beacon that can choose its value after observing the honest value can also
cancel the honest XOR contribution.

Local verification proves signatures, matrix completeness, deterministic seed
derivation and witness quorum.  It cannot prove future unpredictability, key
separation in the physical world, or global log non-equivocation; the verdict
therefore remains explicitly conditional on those operational assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Mapping, Sequence

from .pareto_v18_study_commitment import (
    StudyCommitmentError,
    StudyRow,
    build_log_chain,
    canonical_bytes,
    merkle_root,
    public_key_sha256,
    sha256_hex,
    sign_mapping,
    verify_signed_mapping,
)


class MultiBeaconError(StudyCommitmentError):
    pass


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _key_hashes_unique(keys: Sequence[bytes]) -> tuple[str, ...]:
    hashes = tuple(public_key_sha256(key) for key in keys)
    if len(set(hashes)) != len(hashes):
        raise MultiBeaconError("all operational role public keys must be distinct")
    return hashes


def multi_beacon_commitment_payload(
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
    if not study_id or not ordered:
        raise MultiBeaconError("study ID and matrix must be nonempty")
    if not sources or len(sources) != len(rounds) or len(sources) != len(beacon_keys):
        raise MultiBeaconError("one source, future round and public key is required per beacon")
    if len(set(sources)) != len(sources) or any(x < 0 for x in rounds):
        raise MultiBeaconError("beacon sources must be unique and rounds nonnegative")
    if not witness_keys or not (1 <= witness_quorum <= len(witness_keys)):
        raise MultiBeaconError("invalid witness quorum")
    role_keys = (coordinator_public_key, completion_public_key, log_public_key) + beacon_keys + witness_keys
    role_hashes = _key_hashes_unique(role_keys)
    beacon_hashes = role_hashes[3 : 3 + len(beacon_keys)]
    witness_hashes = role_hashes[3 + len(beacon_keys) :]
    return {
        "schema": "pareto_smc_v19_multi_beacon_study_commitment_v1",
        "study_id": study_id,
        "row_count": len(ordered),
        "rows": [row.to_dict() | {"row_id": row.row_id} for row in ordered],
        "merkle_root": merkle_root(ordered),
        "coordinator_public_key_sha256": role_hashes[0],
        "completion_public_key_sha256": role_hashes[1],
        "log_public_key_sha256": role_hashes[2],
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


def multi_beacon_payload(
    *,
    study_commitment_sha256: str,
    source: str,
    round_id: int,
    value_hex: str,
) -> dict[str, object]:
    if _HEX64.fullmatch(study_commitment_sha256) is None or _HEX64.fullmatch(value_hex) is None:
        raise MultiBeaconError("commitment and beacon value must be 32-byte lowercase hex")
    if not source or not isinstance(round_id, int) or round_id < 0:
        raise MultiBeaconError("invalid beacon identity")
    return {
        "schema": "pareto_smc_v19_external_beacon_v1",
        "study_commitment_sha256": study_commitment_sha256,
        "source": source,
        "round_id": round_id,
        "value_hex": value_hex,
    }


def beacon_set_sha256(beacons: Sequence[Mapping[str, object]]) -> str:
    ordered = sorted(
        (dict(beacon) for beacon in beacons),
        key=lambda item: (str(item["source"]), int(item["round_id"])),
    )
    return sha256_hex(canonical_bytes(ordered))


def derive_multi_beacon_row_seed(
    commitment_payload: Mapping[str, object],
    beacons: Sequence[Mapping[str, object]],
    row_id: str,
) -> int:
    if _HEX64.fullmatch(row_id) is None:
        raise MultiBeaconError("row_id is malformed")
    if not beacons:
        raise MultiBeaconError("at least one beacon is required")
    xor_value = 0
    for beacon in beacons:
        value = str(beacon.get("value_hex"))
        if _HEX64.fullmatch(value) is None:
            raise MultiBeaconError("beacon value is malformed")
        xor_value ^= int.from_bytes(bytes.fromhex(value)[:8], "big")
    mask = int.from_bytes(
        hashlib.sha256(
            b"pareto-smc-v19-row-mask\x00"
            + bytes.fromhex(sha256_hex(canonical_bytes(commitment_payload)))
            + bytes.fromhex(row_id)
        ).digest()[:8],
        "big",
    )
    return xor_value ^ mask


def multi_beacon_completion_payload(
    *,
    study_commitment_sha256: str,
    beacon_set_sha256_value: str,
    row_id: str,
    derived_seed: int,
    status: str,
    result_sha256: str,
) -> dict[str, object]:
    if status not in {"SUCCESS", "FAILURE"}:
        raise MultiBeaconError("completion status must be SUCCESS or FAILURE")
    for name, value in (
        ("study_commitment_sha256", study_commitment_sha256),
        ("beacon_set_sha256", beacon_set_sha256_value),
        ("row_id", row_id),
        ("result_sha256", result_sha256),
    ):
        if _HEX64.fullmatch(value) is None:
            raise MultiBeaconError(f"{name} is malformed")
    if not isinstance(derived_seed, int) or not (0 <= derived_seed < 2**64):
        raise MultiBeaconError("derived_seed must be an unsigned 64-bit integer")
    return {
        "schema": "pareto_smc_v19_row_completion_v1",
        "study_commitment_sha256": study_commitment_sha256,
        "beacon_set_sha256": beacon_set_sha256_value,
        "row_id": row_id,
        "derived_seed": derived_seed,
        "status": status,
        "result_sha256": result_sha256,
    }


def witness_payload(
    *,
    study_commitment_sha256: str,
    final_log_root_sha256: str,
    record_count: int,
    witness_id: str,
) -> dict[str, object]:
    if _HEX64.fullmatch(study_commitment_sha256) is None or _HEX64.fullmatch(final_log_root_sha256) is None:
        raise MultiBeaconError("witness digest is malformed")
    if not witness_id or not isinstance(record_count, int) or record_count <= 0:
        raise MultiBeaconError("invalid witness statement")
    return {
        "schema": "pareto_smc_v19_log_witness_v1",
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
        raise MultiBeaconError("transparency log length mismatch")
    previous = "0" * 64
    for index, (record, envelope) in enumerate(zip(signed_records, signed_log_chain, strict=True)):
        payload = verify_signed_mapping(envelope, log_public_key)
        if payload.get("index") != index or payload.get("previous_record_sha256") != previous:
            raise MultiBeaconError("transparency log chain is discontinuous")
        if payload.get("signed_record_sha256") != sha256_hex(canonical_bytes(record)):
            raise MultiBeaconError("transparency log record binding mismatch")
        previous = sha256_hex(canonical_bytes(envelope))
    return previous


@dataclass(frozen=True)
class MultiBeaconStudyAudit:
    study_id: str
    row_count: int
    beacon_count: int
    witness_count: int
    witness_quorum: int
    success_count: int
    failure_count: int
    all_rows_completed_once: bool
    witness_quorum_pass: bool
    local_verdict: str
    external_assumptions: tuple[str, ...]
    final_log_root_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "pareto_smc_v19_multi_beacon_study_audit_v1",
            "study_id": self.study_id,
            "row_count": self.row_count,
            "beacon_count": self.beacon_count,
            "witness_count": self.witness_count,
            "witness_quorum": self.witness_quorum,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "all_rows_completed_once": self.all_rows_completed_once,
            "witness_quorum_pass": self.witness_quorum_pass,
            "local_verdict": self.local_verdict,
            "external_assumptions": list(self.external_assumptions),
            "final_log_root_sha256": self.final_log_root_sha256,
        }


def verify_multi_beacon_study(
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
) -> MultiBeaconStudyAudit:
    commitment = verify_signed_mapping(signed_commitment, coordinator_public_key)
    if commitment.get("schema") != "pareto_smc_v19_multi_beacon_study_commitment_v1":
        raise MultiBeaconError("unsupported commitment schema")
    commitment_sha = sha256_hex(canonical_bytes(commitment))
    expected_role_hashes = _key_hashes_unique(
        (coordinator_public_key, completion_public_key, log_public_key)
        + tuple(beacon_public_keys)
        + tuple(witness_public_keys)
    )
    if commitment.get("coordinator_public_key_sha256") != expected_role_hashes[0]:
        raise MultiBeaconError("coordinator key mismatch")
    if commitment.get("completion_public_key_sha256") != expected_role_hashes[1]:
        raise MultiBeaconError("completion key mismatch")
    if commitment.get("log_public_key_sha256") != expected_role_hashes[2]:
        raise MultiBeaconError("log key mismatch")

    beacon_specs = commitment.get("beacons")
    if not isinstance(beacon_specs, Sequence) or isinstance(beacon_specs, (str, bytes)):
        raise MultiBeaconError("beacon commitment list is malformed")
    if len(beacon_specs) != len(beacon_public_keys) or len(signed_beacons) != len(beacon_specs):
        raise MultiBeaconError("beacon count mismatch")
    beacon_payloads: list[Mapping[str, object]] = []
    for spec, envelope, public_key in zip(beacon_specs, signed_beacons, beacon_public_keys, strict=True):
        if not isinstance(spec, Mapping):
            raise MultiBeaconError("beacon spec is malformed")
        if spec.get("public_key_sha256") != public_key_sha256(public_key):
            raise MultiBeaconError("beacon public key mismatch")
        payload = verify_signed_mapping(envelope, public_key)
        if payload.get("schema") != "pareto_smc_v19_external_beacon_v1":
            raise MultiBeaconError("unsupported beacon payload schema")
        if payload.get("study_commitment_sha256") != commitment_sha:
            raise MultiBeaconError("beacon does not bind the study commitment")
        if payload.get("source") != spec.get("source") or payload.get("round_id") != spec.get("future_round"):
            raise MultiBeaconError("beacon identity or round mismatch")
        beacon_payloads.append(payload)
    beacon_set_hash = beacon_set_sha256(beacon_payloads)

    rows_raw = commitment.get("rows")
    if not isinstance(rows_raw, Sequence) or isinstance(rows_raw, (str, bytes)):
        raise MultiBeaconError("committed rows are malformed")
    rows: list[StudyRow] = []
    row_by_id: dict[str, StudyRow] = {}
    for item in rows_raw:
        if not isinstance(item, Mapping):
            raise MultiBeaconError("committed row is malformed")
        row = StudyRow(
            case_id=str(item["case_id"]),
            algorithm_id=str(item["algorithm_id"]),
            replicate_id=str(item["replicate_id"]),
            budget=int(item["budget"]),
            configuration_sha256=str(item["configuration_sha256"]),
        )
        if item.get("row_id") != row.row_id or row.row_id in row_by_id:
            raise MultiBeaconError("row ID mismatch or duplicate")
        rows.append(row)
        row_by_id[row.row_id] = row
    if commitment.get("row_count") != len(rows) or commitment.get("merkle_root") != merkle_root(rows):
        raise MultiBeaconError("study matrix root or count mismatch")

    completions: dict[str, Mapping[str, object]] = {}
    success_count = 0
    failure_count = 0
    completion_payloads: list[Mapping[str, object]] = []
    for envelope in signed_completions:
        payload = verify_signed_mapping(envelope, completion_public_key)
        if payload.get("schema") != "pareto_smc_v19_row_completion_v1":
            raise MultiBeaconError("unsupported completion payload schema")
        row_id = str(payload.get("row_id"))
        if row_id not in row_by_id or row_id in completions:
            raise MultiBeaconError("unknown or duplicate completion row")
        if payload.get("study_commitment_sha256") != commitment_sha or payload.get("beacon_set_sha256") != beacon_set_hash:
            raise MultiBeaconError("completion commitment or beacon-set mismatch")
        expected_seed = derive_multi_beacon_row_seed(commitment, beacon_payloads, row_id)
        if payload.get("derived_seed") != expected_seed:
            raise MultiBeaconError("completion seed mismatch")
        status = payload.get("status")
        if status == "SUCCESS":
            success_count += 1
        elif status == "FAILURE":
            failure_count += 1
        else:
            raise MultiBeaconError("invalid completion status")
        if _HEX64.fullmatch(str(payload.get("result_sha256"))) is None:
            raise MultiBeaconError("completion result SHA-256 is malformed")
        completions[row_id] = payload
        completion_payloads.append(payload)
    all_completed = set(completions) == set(row_by_id)
    if not all_completed:
        raise MultiBeaconError("every committed row must have exactly one completion")

    signed_records = [signed_commitment, *signed_beacons, *signed_completions]
    final_log_root = _verify_log_chain(signed_records, signed_log_chain, log_public_key)

    witness_hashes = commitment.get("witness_public_key_sha256")
    quorum = int(commitment.get("witness_quorum"))
    if not isinstance(witness_hashes, Sequence) or isinstance(witness_hashes, (str, bytes)):
        raise MultiBeaconError("witness key list is malformed")
    if len(witness_hashes) != len(witness_public_keys):
        raise MultiBeaconError("witness public-key count mismatch")
    if len(signed_witnesses) > len(witness_public_keys):
        raise MultiBeaconError("more witness statements than committed witness keys")
    verified_witness_ids: set[str] = set()
    verified_witness_key_hashes: set[str] = set()
    committed_witness_keys = tuple(witness_public_keys)
    committed_witness_hashes = {public_key_sha256(key) for key in committed_witness_keys}
    for envelope in signed_witnesses:
        matches: list[tuple[bytes, Mapping[str, object]]] = []
        for public_key in committed_witness_keys:
            key_hash = public_key_sha256(public_key)
            if key_hash in verified_witness_key_hashes:
                continue
            try:
                payload = verify_signed_mapping(envelope, public_key)
            except Exception:
                continue
            matches.append((public_key, payload))
        if len(matches) != 1:
            raise MultiBeaconError(
                "each witness statement must verify under exactly one unused committed witness key"
            )
        public_key, payload = matches[0]
        key_hash = public_key_sha256(public_key)
        if key_hash not in committed_witness_hashes:
            raise MultiBeaconError("uncommitted witness public key")
        if payload.get("schema") != "pareto_smc_v19_log_witness_v1":
            raise MultiBeaconError("unsupported witness payload schema")
        if payload.get("study_commitment_sha256") != commitment_sha:
            raise MultiBeaconError("witness commitment mismatch")
        if payload.get("final_log_root_sha256") != final_log_root:
            raise MultiBeaconError("witness log-root mismatch")
        if payload.get("record_count") != len(signed_records):
            raise MultiBeaconError("witness record-count mismatch")
        witness_id = str(payload.get("witness_id"))
        if not witness_id or witness_id in verified_witness_ids:
            raise MultiBeaconError("empty or duplicate witness ID")
        verified_witness_ids.add(witness_id)
        verified_witness_key_hashes.add(key_hash)
    quorum_pass = len(verified_witness_ids) >= quorum
    if not quorum_pass:
        raise MultiBeaconError("witness quorum was not met")
    return MultiBeaconStudyAudit(
        study_id=str(commitment.get("study_id")),
        row_count=len(rows),
        beacon_count=len(beacon_payloads),
        witness_count=len(verified_witness_ids),
        witness_quorum=quorum,
        success_count=success_count,
        failure_count=failure_count,
        all_rows_completed_once=all_completed,
        witness_quorum_pass=quorum_pass,
        local_verdict=(
            "PASS_CONDITIONAL_AT_LEAST_ONE_FUTURE_BEACON_UNIFORM_AND_"
            "AT_LEAST_ONE_HONEST_NON_EQUIVOCATING_WITNESS"
        ),
        external_assumptions=(
            "conditional on the commitment and all other beacon values, at least one committed beacon first-64-bit value is future-unpredictable and independent uniform",
            "at least one quorum witness is honest and observes a globally non-equivocating log",
            "operational private-key roles are genuinely separated",
            "all formal executions use only the committed matrix and completion channel",
        ),
        final_log_root_sha256=final_log_root,
    )


__all__ = [
    "MultiBeaconError",
    "MultiBeaconStudyAudit",
    "beacon_set_sha256",
    "build_log_chain",
    "derive_multi_beacon_row_seed",
    "multi_beacon_commitment_payload",
    "multi_beacon_completion_payload",
    "multi_beacon_payload",
    "sign_mapping",
    "verify_multi_beacon_study",
    "witness_payload",
]
