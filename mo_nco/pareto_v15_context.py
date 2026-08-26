from __future__ import annotations

"""Canonical fail-closed identity context for v15 certificates.

Every certificate-producing or certificate-consuming stage can bind the same
case, instance, configuration, frozen cells, reference artifact, type/cell
allocation, pilot plan, and confirm plan through one deterministic digest.
There are no defaults: omitting any binding is a construction error.
"""

from dataclasses import dataclass
import hashlib
import hmac
import json
import re


V15_CERTIFICATE_CONTEXT_SCHEMA = "pareto_smc_certificate_context_v15"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_CASE_ID_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z"
)
_HASH_FIELDS = (
    "instance_sha256",
    "configuration_sha256",
    "cell_manifest_sha256",
    "reference_sha256",
    "type_cell_plan_sha256",
    "pilot_plan_sha256",
    "confirm_plan_sha256",
)


class V15CertificateContextError(ValueError):
    """Raised when a v15 certificate context is missing or noncanonical."""


def _require_case_id(value: object) -> str:
    if not isinstance(value, str) or _CASE_ID_PATTERN.fullmatch(value) is None:
        raise V15CertificateContextError(
            "case_id must be a nonempty canonical ASCII token of at most "
            "128 characters using only letters, digits, '.', '_', ':', "
            "and '-'."
        )
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise V15CertificateContextError(
            f"{field} must be a lowercase 64-character SHA-256 hex digest."
        )
    return value


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise V15CertificateContextError(
            "Certificate context is not canonical-JSON serializable."
        ) from error
    return encoded


@dataclass(frozen=True, slots=True)
class V15CertificateContext:
    """Complete hash binding shared by every v15 certificate stage."""

    case_id: str
    instance_sha256: str
    configuration_sha256: str
    cell_manifest_sha256: str
    reference_sha256: str
    type_cell_plan_sha256: str
    pilot_plan_sha256: str
    confirm_plan_sha256: str

    def __post_init__(self) -> None:
        _require_case_id(self.case_id)
        for field in _HASH_FIELDS:
            _require_sha256(getattr(self, field), field=field)

    def canonical_payload(self) -> dict[str, str]:
        """Return the complete digest preimage, excluding its own digest."""

        self.validate()
        return {
            "schema": V15_CERTIFICATE_CONTEXT_SCHEMA,
            "case_id": self.case_id,
            "instance_sha256": self.instance_sha256,
            "configuration_sha256": self.configuration_sha256,
            "cell_manifest_sha256": self.cell_manifest_sha256,
            "reference_sha256": self.reference_sha256,
            "type_cell_plan_sha256": self.type_cell_plan_sha256,
            "pilot_plan_sha256": self.pilot_plan_sha256,
            "confirm_plan_sha256": self.confirm_plan_sha256,
        }

    def canonical_json_bytes(self) -> bytes:
        """Return the exact UTF-8 JSON preimage used for ``context_sha256``."""

        return _canonical_json_bytes(self.canonical_payload())

    @property
    def context_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()

    def validate(self) -> None:
        """Revalidate all fields, including after hostile low-level mutation."""

        _require_case_id(self.case_id)
        for field in _HASH_FIELDS:
            _require_sha256(getattr(self, field), field=field)

    def to_jsonable(self) -> dict[str, str]:
        """Return the canonical payload plus its deterministic digest."""

        payload = self.canonical_payload()
        payload["context_sha256"] = self.context_sha256
        return payload


def verify_v15_context_sha256(
    context: V15CertificateContext,
    expected_context_sha256: object,
) -> V15CertificateContext:
    """Fail closed unless ``expected_context_sha256`` binds ``context``.

    The expected digest is itself required to be canonical lowercase SHA-256
    text.  Returning the validated context makes successful verification easy
    to compose without replacing it by an unbound Boolean flag.
    """

    if not isinstance(context, V15CertificateContext):
        raise V15CertificateContextError(
            "context must be a V15CertificateContext instance."
        )
    context.validate()
    expected = _require_sha256(
        expected_context_sha256,
        field="expected_context_sha256",
    )
    observed = context.context_sha256
    if not hmac.compare_digest(observed, expected):
        raise V15CertificateContextError(
            "context_sha256 does not match the canonical v15 context."
        )
    return context


validate_v15_context_sha256 = verify_v15_context_sha256


__all__ = [
    "V15_CERTIFICATE_CONTEXT_SCHEMA",
    "V15CertificateContext",
    "V15CertificateContextError",
    "validate_v15_context_sha256",
    "verify_v15_context_sha256",
]
