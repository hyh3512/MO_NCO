"""Externally signed, fail-closed pre-confirm receipts for Pareto-SMC.

The receipt binds an already-created pilot commitment to the exact confirm
contract before confirm is launched.  Ed25519 authenticates the canonical
payload, but neither the signed ``issued_at_utc`` metadata nor this module can
prove wall-clock ordering by itself.  A receipt authorizes pre-confirm use only
when the caller also establishes both external controls required by
``verify_preconfirm_receipt``:

* the frozen signer private key is not held by or accessible to the runner;
* verification completes before confirm starts.

The signature is detached from the payload: Ed25519 signs the canonical request
bytes only.  The canonical receipt envelope merely transports that payload and
the detached signature.  Private-key material is never serialized.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
except ImportError as exc:  # pragma: no cover - exercised only without extra
    InvalidSignature = None  # type: ignore[assignment]
    Ed25519PrivateKey = None  # type: ignore[assignment,misc]
    Ed25519PublicKey = None  # type: ignore[assignment,misc]
    serialization = None  # type: ignore[assignment]
    _CRYPTOGRAPHY_IMPORT_ERROR: ImportError | None = exc
else:
    _CRYPTOGRAPHY_IMPORT_ERROR = None


UNSIGNED_REQUEST_SCHEMA = "pareto_smc_v13_preconfirm_receipt_request_v2"
RECEIPT_SCHEMA = "pareto_smc_v13_preconfirm_receipt_v2"
RECEIPT_VERSION = 2
SIGNATURE_ALGORITHM = "Ed25519"
SIGNATURE_ENCODING = "base64"

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}$")
_UTC_SECONDS = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
_PAYLOAD_FIELDS = frozenset(
    {
        "schema",
        "version",
        "pilot_plan_commitment_sha256",
        "pilot_result_payload_sha256",
        "pilot_terminal_support_sha256",
        "certificate_specification_sha256",
        "run_id",
        "case_id",
        "algorithm_id",
        "pilot_stream_id",
        "confirm_stream_id",
        "confirm_contract_sha256",
        "confirm_seed_commitment_sha256",
        "signer_key_id",
        "signer_public_key_sha256",
        "issued_at_utc",
    }
)
_ENVELOPE_FIELDS = frozenset({"schema", "version", "payload", "signature"})
_SIGNATURE_FIELDS = frozenset(
    {
        "algorithm",
        "encoding",
        "signed_payload_sha256",
        "value",
    }
)


class PreconfirmReceiptError(ValueError):
    """Base error for malformed or unauthorized pre-confirm receipts."""


class PreconfirmReceiptVerificationError(PreconfirmReceiptError):
    """Raised when a receipt cannot authorize the confirm launch."""


@dataclass(frozen=True)
class PreconfirmReceiptBindings:
    """Content that must match the runner's frozen pilot/confirm contract."""

    pilot_plan_commitment_sha256: str
    pilot_result_payload_sha256: str
    pilot_terminal_support_sha256: str
    certificate_specification_sha256: str
    run_id: str
    case_id: str
    algorithm_id: str
    pilot_stream_id: str
    confirm_stream_id: str
    confirm_contract_sha256: str
    confirm_seed_commitment_sha256: str

    def as_dict(self) -> dict[str, str]:
        """Return a fresh mapping in the public receipt vocabulary."""

        return {
            "pilot_plan_commitment_sha256": (
                self.pilot_plan_commitment_sha256
            ),
            "pilot_result_payload_sha256": (
                self.pilot_result_payload_sha256
            ),
            "pilot_terminal_support_sha256": (
                self.pilot_terminal_support_sha256
            ),
            "certificate_specification_sha256": (
                self.certificate_specification_sha256
            ),
            "run_id": self.run_id,
            "case_id": self.case_id,
            "algorithm_id": self.algorithm_id,
            "pilot_stream_id": self.pilot_stream_id,
            "confirm_stream_id": self.confirm_stream_id,
            "confirm_contract_sha256": self.confirm_contract_sha256,
            "confirm_seed_commitment_sha256": (
                self.confirm_seed_commitment_sha256
            ),
        }


@dataclass(frozen=True)
class VerifiedPreconfirmReceipt:
    """Successful verification result.

    ``independent_timing_proof_established`` is deliberately always false:
    verification-before-launch is an external sequencing fact, not a fact
    derivable from the signed timestamp.
    """

    authorization_gate: str
    receipt_sha256: str
    signed_payload_sha256: str
    signer_public_key_sha256: str
    payload: Mapping[str, object]
    issued_at_is_metadata_only: bool = True
    independent_timing_proof_established: bool = False


def _require_cryptography() -> None:
    if _CRYPTOGRAPHY_IMPORT_ERROR is not None:
        raise PreconfirmReceiptError(
            "Ed25519 receipts require the optional 'cryptography' package."
        ) from _CRYPTOGRAPHY_IMPORT_ERROR


def _canonical_json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PreconfirmReceiptError(
            "Receipt data is not canonical-JSON serializable."
        ) from exc
    return text.encode("utf-8")


def _reject_duplicate_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PreconfirmReceiptError(
                f"Duplicate JSON field is forbidden: {key!r}."
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise PreconfirmReceiptError(
        f"Non-finite JSON constant is forbidden: {value}."
    )


def _parse_canonical_json_object(raw: bytes, *, label: str) -> dict[str, object]:
    if not isinstance(raw, bytes):
        raise PreconfirmReceiptError(f"{label} must be bytes.")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PreconfirmReceiptError(f"{label} must be valid UTF-8.") from exc
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except PreconfirmReceiptError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise PreconfirmReceiptError(f"{label} is not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise PreconfirmReceiptError(f"{label} must be a JSON object.")
    if _canonical_json_bytes(parsed) != raw:
        raise PreconfirmReceiptError(
            f"{label} must be exact canonical UTF-8 JSON with no newline."
        )
    return parsed


def _require_exact_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    actual = frozenset(value)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown or missing:
        fragments = []
        if unknown:
            fragments.append(f"unknown={unknown}")
        if missing:
            fragments.append(f"missing={missing}")
        raise PreconfirmReceiptError(
            f"{label} has an invalid field set ({', '.join(fragments)})."
        )


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise PreconfirmReceiptError(
            f"{field} must be a lowercase 64-character SHA-256 hex digest."
        )
    return value


def _require_identity(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTITY.fullmatch(value) is None:
        raise PreconfirmReceiptError(
            f"{field} must be a nonempty canonical ASCII identity."
        )
    return value


def _validate_bindings(bindings: PreconfirmReceiptBindings) -> None:
    if not isinstance(bindings, PreconfirmReceiptBindings):
        raise PreconfirmReceiptError(
            "bindings must be a PreconfirmReceiptBindings instance."
        )
    for field in (
        "pilot_plan_commitment_sha256",
        "pilot_result_payload_sha256",
        "pilot_terminal_support_sha256",
        "certificate_specification_sha256",
        "confirm_contract_sha256",
        "confirm_seed_commitment_sha256",
    ):
        _require_sha256(getattr(bindings, field), field=field)
    for field in (
        "run_id",
        "case_id",
        "algorithm_id",
        "pilot_stream_id",
        "confirm_stream_id",
    ):
        _require_identity(getattr(bindings, field), field=field)
    if bindings.pilot_stream_id == bindings.confirm_stream_id:
        raise PreconfirmReceiptError(
            "pilot_stream_id and confirm_stream_id must be distinct. "
            "Distinct labels do not by themselves prove stream independence."
        )


def _require_public_key_raw(value: object, *, field: str) -> bytes:
    _require_cryptography()
    if not isinstance(value, bytes) or len(value) != 32:
        raise PreconfirmReceiptError(
            f"{field} must be exactly 32 raw Ed25519 public-key bytes."
        )
    try:
        Ed25519PublicKey.from_public_bytes(value)
    except ValueError as exc:
        raise PreconfirmReceiptError(
            f"{field} is not a valid raw Ed25519 public key."
        ) from exc
    return value


def _validate_issued_at(value: object) -> str:
    if not isinstance(value, str) or _UTC_SECONDS.fullmatch(value) is None:
        raise PreconfirmReceiptError(
            "issued_at_utc must use YYYY-MM-DDTHH:MM:SSZ."
        )
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise PreconfirmReceiptError(
            "issued_at_utc is not a valid UTC calendar timestamp."
        ) from exc
    return value


def _validate_payload(payload: Mapping[str, object]) -> None:
    _require_exact_fields(payload, _PAYLOAD_FIELDS, label="request payload")
    if payload["schema"] != UNSIGNED_REQUEST_SCHEMA:
        raise PreconfirmReceiptError("Unsupported unsigned-request schema.")
    if (
        type(payload["version"]) is not int
        or payload["version"] != RECEIPT_VERSION
    ):
        raise PreconfirmReceiptError("Unsupported unsigned-request version.")
    bindings = PreconfirmReceiptBindings(
        pilot_plan_commitment_sha256=_require_sha256(
            payload["pilot_plan_commitment_sha256"],
            field="pilot_plan_commitment_sha256",
        ),
        pilot_result_payload_sha256=_require_sha256(
            payload["pilot_result_payload_sha256"],
            field="pilot_result_payload_sha256",
        ),
        pilot_terminal_support_sha256=_require_sha256(
            payload["pilot_terminal_support_sha256"],
            field="pilot_terminal_support_sha256",
        ),
        certificate_specification_sha256=_require_sha256(
            payload["certificate_specification_sha256"],
            field="certificate_specification_sha256",
        ),
        run_id=_require_identity(payload["run_id"], field="run_id"),
        case_id=_require_identity(payload["case_id"], field="case_id"),
        algorithm_id=_require_identity(
            payload["algorithm_id"],
            field="algorithm_id",
        ),
        pilot_stream_id=_require_identity(
            payload["pilot_stream_id"],
            field="pilot_stream_id",
        ),
        confirm_stream_id=_require_identity(
            payload["confirm_stream_id"],
            field="confirm_stream_id",
        ),
        confirm_contract_sha256=_require_sha256(
            payload["confirm_contract_sha256"],
            field="confirm_contract_sha256",
        ),
        confirm_seed_commitment_sha256=_require_sha256(
            payload["confirm_seed_commitment_sha256"],
            field="confirm_seed_commitment_sha256",
        ),
    )
    _validate_bindings(bindings)
    _require_identity(payload["signer_key_id"], field="signer_key_id")
    _require_sha256(
        payload["signer_public_key_sha256"],
        field="signer_public_key_sha256",
    )
    _validate_issued_at(payload["issued_at_utc"])


def create_unsigned_preconfirm_receipt_request(
    *,
    bindings: PreconfirmReceiptBindings,
    signer_key_id: str,
    frozen_signer_public_key_raw: bytes,
    issued_at_utc: str,
) -> bytes:
    """Create the exact canonical bytes that an external signer must sign.

    ``issued_at_utc`` is signed metadata for audit readability only.  It is not
    accepted as evidence that signing or verification preceded confirm.
    """

    _validate_bindings(bindings)
    signer_key_id = _require_identity(signer_key_id, field="signer_key_id")
    public_key_raw = _require_public_key_raw(
        frozen_signer_public_key_raw,
        field="frozen_signer_public_key_raw",
    )
    issued_at_utc = _validate_issued_at(issued_at_utc)
    payload: dict[str, object] = {
        "schema": UNSIGNED_REQUEST_SCHEMA,
        "version": RECEIPT_VERSION,
        **bindings.as_dict(),
        "signer_key_id": signer_key_id,
        "signer_public_key_sha256": hashlib.sha256(
            public_key_raw
        ).hexdigest(),
        "issued_at_utc": issued_at_utc,
    }
    _validate_payload(payload)
    return _canonical_json_bytes(payload)


def sign_preconfirm_receipt_request(
    unsigned_request: bytes,
    *,
    private_key_raw: bytes,
) -> bytes:
    """Sign a canonical request and return a canonical receipt envelope.

    This helper is intended for an external signing process and deterministic
    tests.  The raw private key is accepted only as an argument and is never
    copied into either the signed payload or the returned receipt.
    """

    _require_cryptography()
    payload = _parse_canonical_json_object(
        unsigned_request,
        label="unsigned request",
    )
    _validate_payload(payload)
    if not isinstance(private_key_raw, bytes) or len(private_key_raw) != 32:
        raise PreconfirmReceiptError(
            "private_key_raw must be exactly 32 raw Ed25519 private-key bytes."
        )
    try:
        private_key = Ed25519PrivateKey.from_private_bytes(private_key_raw)
    except ValueError as exc:
        raise PreconfirmReceiptError(
            "private_key_raw is not a valid Ed25519 private key."
        ) from exc
    derived_public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    derived_fingerprint = hashlib.sha256(derived_public_key).hexdigest()
    if payload["signer_public_key_sha256"] != derived_fingerprint:
        raise PreconfirmReceiptError(
            "Signing key does not match the public key frozen in the request."
        )
    signature = private_key.sign(unsigned_request)
    envelope = {
        "schema": RECEIPT_SCHEMA,
        "version": RECEIPT_VERSION,
        "payload": payload,
        "signature": {
            "algorithm": SIGNATURE_ALGORITHM,
            "encoding": SIGNATURE_ENCODING,
            "signed_payload_sha256": hashlib.sha256(
                unsigned_request
            ).hexdigest(),
            "value": base64.b64encode(signature).decode("ascii"),
        },
    }
    return _canonical_json_bytes(envelope)


def _decode_signature(signature: Mapping[str, object]) -> bytes:
    _require_exact_fields(
        signature,
        _SIGNATURE_FIELDS,
        label="signature metadata",
    )
    if signature["algorithm"] != SIGNATURE_ALGORITHM:
        raise PreconfirmReceiptVerificationError(
            "Only Ed25519 signatures are accepted."
        )
    if signature["encoding"] != SIGNATURE_ENCODING:
        raise PreconfirmReceiptVerificationError(
            "Only canonical base64 signature encoding is accepted."
        )
    value = signature["value"]
    if not isinstance(value, str) or not value:
        raise PreconfirmReceiptVerificationError(
            "Signature value must be a nonempty base64 string."
        )
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PreconfirmReceiptVerificationError(
            "Signature value is not valid base64."
        ) from exc
    if base64.b64encode(decoded).decode("ascii") != value:
        raise PreconfirmReceiptVerificationError(
            "Signature value is not canonical base64."
        )
    if len(decoded) != 64:
        raise PreconfirmReceiptVerificationError(
            "Ed25519 detached signature must contain exactly 64 bytes."
        )
    return decoded


def _verify_preconfirm_receipt(
    receipt: bytes,
    *,
    frozen_signer_public_key_raw: bytes,
    expected_bindings: PreconfirmReceiptBindings,
    expected_signer_key_id: str,
    external_signer_key_not_held_by_runner: bool,
    receipt_verified_before_confirm_start: bool,
) -> VerifiedPreconfirmReceipt:
    """Verify and authorize a receipt, failing closed on every mismatch.

    The two boolean controls are required facts supplied by external
    orchestration.  A successful signature or the signed ``issued_at_utc`` does
    not substitute for either control.
    """

    _require_cryptography()
    envelope = _parse_canonical_json_object(receipt, label="receipt")
    _require_exact_fields(envelope, _ENVELOPE_FIELDS, label="receipt")
    if envelope["schema"] != RECEIPT_SCHEMA:
        raise PreconfirmReceiptVerificationError(
            "Unsupported pre-confirm receipt schema."
        )
    if (
        type(envelope["version"]) is not int
        or envelope["version"] != RECEIPT_VERSION
    ):
        raise PreconfirmReceiptVerificationError(
            "Unsupported pre-confirm receipt version."
        )
    payload = envelope["payload"]
    signature_metadata = envelope["signature"]
    if not isinstance(payload, dict):
        raise PreconfirmReceiptVerificationError(
            "Receipt payload must be an object."
        )
    if not isinstance(signature_metadata, dict):
        raise PreconfirmReceiptVerificationError(
            "Receipt signature metadata must be an object."
        )
    _validate_payload(payload)
    payload_bytes = _canonical_json_bytes(payload)
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    declared_payload_sha256 = _require_sha256(
        signature_metadata.get("signed_payload_sha256"),
        field="signed_payload_sha256",
    )
    if declared_payload_sha256 != payload_sha256:
        raise PreconfirmReceiptVerificationError(
            "signed_payload_sha256 does not match the canonical payload."
        )
    detached_signature = _decode_signature(signature_metadata)

    public_key_raw = _require_public_key_raw(
        frozen_signer_public_key_raw,
        field="frozen_signer_public_key_raw",
    )
    public_key_sha256 = hashlib.sha256(public_key_raw).hexdigest()
    if payload["signer_public_key_sha256"] != public_key_sha256:
        raise PreconfirmReceiptVerificationError(
            "Receipt signer does not match the pre-frozen public key."
        )
    try:
        Ed25519PublicKey.from_public_bytes(public_key_raw).verify(
            detached_signature,
            payload_bytes,
        )
    except InvalidSignature as exc:
        raise PreconfirmReceiptVerificationError(
            "Ed25519 detached signature verification failed."
        ) from exc

    _validate_bindings(expected_bindings)
    expected_payload_bindings = expected_bindings.as_dict()
    for field, expected_value in expected_payload_bindings.items():
        if payload[field] != expected_value:
            raise PreconfirmReceiptVerificationError(
                f"Receipt binding mismatch for {field}."
            )
    expected_signer_key_id = _require_identity(
        expected_signer_key_id,
        field="expected_signer_key_id",
    )
    if payload["signer_key_id"] != expected_signer_key_id:
        raise PreconfirmReceiptVerificationError(
            "Receipt signer_key_id does not match the frozen signer identity."
        )

    if type(external_signer_key_not_held_by_runner) is not bool:
        raise PreconfirmReceiptVerificationError(
            "external_signer_key_not_held_by_runner must be boolean."
        )
    if not external_signer_key_not_held_by_runner:
        raise PreconfirmReceiptVerificationError(
            "Authorization denied: the external signer private key must not "
            "be held by or accessible to the runner."
        )
    if type(receipt_verified_before_confirm_start) is not bool:
        raise PreconfirmReceiptVerificationError(
            "receipt_verified_before_confirm_start must be boolean."
        )
    if not receipt_verified_before_confirm_start:
        raise PreconfirmReceiptVerificationError(
            "Authorization denied: receipt verification must complete before "
            "confirm starts."
        )

    return VerifiedPreconfirmReceipt(
        authorization_gate="PASS",
        receipt_sha256=hashlib.sha256(receipt).hexdigest(),
        signed_payload_sha256=payload_sha256,
        signer_public_key_sha256=public_key_sha256,
        payload=dict(payload),
    )


def verify_preconfirm_receipt(
    receipt: bytes,
    *,
    frozen_signer_public_key_raw: bytes,
    expected_bindings: PreconfirmReceiptBindings,
    expected_signer_key_id: str,
    external_signer_key_not_held_by_runner: bool,
    receipt_verified_before_confirm_start: bool,
) -> VerifiedPreconfirmReceipt:
    """Verify and authorize a receipt, failing closed on every mismatch.

    The two boolean controls are required facts supplied by external
    orchestration.  A successful signature or the signed ``issued_at_utc`` does
    not substitute for either control.  Every malformed-receipt path is
    normalized to :class:`PreconfirmReceiptVerificationError`.
    """

    try:
        return _verify_preconfirm_receipt(
            receipt,
            frozen_signer_public_key_raw=frozen_signer_public_key_raw,
            expected_bindings=expected_bindings,
            expected_signer_key_id=expected_signer_key_id,
            external_signer_key_not_held_by_runner=(
                external_signer_key_not_held_by_runner
            ),
            receipt_verified_before_confirm_start=(
                receipt_verified_before_confirm_start
            ),
        )
    except PreconfirmReceiptVerificationError:
        raise
    except PreconfirmReceiptError as exc:
        raise PreconfirmReceiptVerificationError(str(exc)) from exc


__all__ = [
    "PreconfirmReceiptBindings",
    "PreconfirmReceiptError",
    "PreconfirmReceiptVerificationError",
    "VerifiedPreconfirmReceipt",
    "create_unsigned_preconfirm_receipt_request",
    "sign_preconfirm_receipt_request",
    "verify_preconfirm_receipt",
]
