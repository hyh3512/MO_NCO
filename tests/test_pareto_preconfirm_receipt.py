import base64
import hashlib
import json
import unittest
from dataclasses import replace

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from mo_nco.pareto_preconfirm_receipt import (
    PreconfirmReceiptBindings,
    PreconfirmReceiptVerificationError,
    create_unsigned_preconfirm_receipt_request,
    sign_preconfirm_receipt_request,
    verify_preconfirm_receipt,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class ParetoPreconfirmReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        private_key = Ed25519PrivateKey.from_private_bytes(
            hashlib.sha256(b"pareto-smc-v13-test-signer").digest()
        )
        self.private_key_raw = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        self.public_key_raw = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.bindings = PreconfirmReceiptBindings(
            pilot_plan_commitment_sha256=_digest("pilot-plan"),
            pilot_result_payload_sha256=_digest("pilot-result"),
            pilot_terminal_support_sha256=_digest("pilot-support"),
            certificate_specification_sha256=_digest("certificate-spec"),
            run_id="run-0001",
            case_id="kroAB100",
            algorithm_id="pareto-smc-v13",
            pilot_stream_id="case:pilot:v1",
            confirm_stream_id="case:confirm:v1",
            confirm_contract_sha256=_digest("confirm-contract"),
            confirm_seed_commitment_sha256=_digest("confirm-seed"),
        )

    @staticmethod
    def _canonical(value: object) -> bytes:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    def _make_receipt(self) -> bytes:
        unsigned_request = create_unsigned_preconfirm_receipt_request(
            bindings=self.bindings,
            signer_key_id="offline-signer-2026q3",
            frozen_signer_public_key_raw=self.public_key_raw,
            issued_at_utc="2026-07-28T12:34:56Z",
        )
        return sign_preconfirm_receipt_request(
            unsigned_request,
            private_key_raw=self.private_key_raw,
        )

    def test_valid_receipt_authorizes_exact_expected_bindings(self) -> None:
        receipt = self._make_receipt()

        verified = verify_preconfirm_receipt(
            receipt,
            frozen_signer_public_key_raw=self.public_key_raw,
            expected_bindings=self.bindings,
            expected_signer_key_id="offline-signer-2026q3",
            external_signer_key_not_held_by_runner=True,
            receipt_verified_before_confirm_start=True,
        )

        self.assertEqual(verified.authorization_gate, "PASS")
        self.assertTrue(verified.issued_at_is_metadata_only)
        self.assertFalse(verified.independent_timing_proof_established)
        self.assertEqual(
            verified.payload["pilot_plan_commitment_sha256"],
            self.bindings.pilot_plan_commitment_sha256,
        )
        self.assertEqual(
            verified.payload["pilot_result_payload_sha256"],
            self.bindings.pilot_result_payload_sha256,
        )

    def test_tampered_payload_fails_detached_signature(self) -> None:
        parsed = json.loads(self._make_receipt().decode("utf-8"))
        parsed["payload"]["case_id"] = "tampered-case"
        payload_bytes = self._canonical(parsed["payload"])
        parsed["signature"]["signed_payload_sha256"] = hashlib.sha256(
            payload_bytes
        ).hexdigest()

        with self.assertRaisesRegex(
            PreconfirmReceiptVerificationError,
            "signature verification failed",
        ):
            verify_preconfirm_receipt(
                self._canonical(parsed),
                frozen_signer_public_key_raw=self.public_key_raw,
                expected_bindings=self.bindings,
                expected_signer_key_id="offline-signer-2026q3",
                external_signer_key_not_held_by_runner=True,
                receipt_verified_before_confirm_start=True,
            )

    def test_wrong_frozen_public_key_is_rejected(self) -> None:
        wrong_public_key = Ed25519PrivateKey.generate().public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        with self.assertRaisesRegex(
            PreconfirmReceiptVerificationError,
            "pre-frozen public key",
        ):
            verify_preconfirm_receipt(
                self._make_receipt(),
                frozen_signer_public_key_raw=wrong_public_key,
                expected_bindings=self.bindings,
                expected_signer_key_id="offline-signer-2026q3",
                external_signer_key_not_held_by_runner=True,
                receipt_verified_before_confirm_start=True,
            )

    def test_unknown_payload_field_is_rejected_fail_closed(self) -> None:
        parsed = json.loads(self._make_receipt().decode("utf-8"))
        parsed["payload"]["runner_claimed_before_confirm"] = True

        with self.assertRaisesRegex(
            PreconfirmReceiptVerificationError,
            "unknown=.*runner_claimed_before_confirm",
        ):
            verify_preconfirm_receipt(
                self._canonical(parsed),
                frozen_signer_public_key_raw=self.public_key_raw,
                expected_bindings=self.bindings,
                expected_signer_key_id="offline-signer-2026q3",
                external_signer_key_not_held_by_runner=True,
                receipt_verified_before_confirm_start=True,
            )

    def test_noncanonical_receipt_serialization_is_rejected(self) -> None:
        parsed = json.loads(self._make_receipt().decode("utf-8"))
        noncanonical = json.dumps(
            parsed,
            ensure_ascii=False,
            sort_keys=False,
            indent=2,
        ).encode("utf-8")

        with self.assertRaisesRegex(
            PreconfirmReceiptVerificationError,
            "exact canonical UTF-8 JSON",
        ):
            verify_preconfirm_receipt(
                noncanonical,
                frozen_signer_public_key_raw=self.public_key_raw,
                expected_bindings=self.bindings,
                expected_signer_key_id="offline-signer-2026q3",
                external_signer_key_not_held_by_runner=True,
                receipt_verified_before_confirm_start=True,
            )

    def test_confirm_seed_commitment_mismatch_is_rejected(self) -> None:
        mismatched = replace(
            self.bindings,
            confirm_seed_commitment_sha256=_digest("different-confirm-seed"),
        )

        with self.assertRaisesRegex(
            PreconfirmReceiptVerificationError,
            "confirm_seed_commitment_sha256",
        ):
            verify_preconfirm_receipt(
                self._make_receipt(),
                frozen_signer_public_key_raw=self.public_key_raw,
                expected_bindings=mismatched,
                expected_signer_key_id="offline-signer-2026q3",
                external_signer_key_not_held_by_runner=True,
                receipt_verified_before_confirm_start=True,
            )

    def test_pilot_result_payload_mismatch_is_rejected(self) -> None:
        mismatched = replace(
            self.bindings,
            pilot_result_payload_sha256=_digest(
                "different-pilot-result"
            ),
        )

        with self.assertRaisesRegex(
            PreconfirmReceiptVerificationError,
            "pilot_result_payload_sha256",
        ):
            verify_preconfirm_receipt(
                self._make_receipt(),
                frozen_signer_public_key_raw=self.public_key_raw,
                expected_bindings=mismatched,
                expected_signer_key_id="offline-signer-2026q3",
                external_signer_key_not_held_by_runner=True,
                receipt_verified_before_confirm_start=True,
            )

    def test_external_signer_separation_is_required_for_authorization(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            PreconfirmReceiptVerificationError,
            "must not be held by or accessible to the runner",
        ):
            verify_preconfirm_receipt(
                self._make_receipt(),
                frozen_signer_public_key_raw=self.public_key_raw,
                expected_bindings=self.bindings,
                expected_signer_key_id="offline-signer-2026q3",
                external_signer_key_not_held_by_runner=False,
                receipt_verified_before_confirm_start=True,
            )

    def test_verification_must_precede_confirm_for_authorization(self) -> None:
        with self.assertRaisesRegex(
            PreconfirmReceiptVerificationError,
            "must complete before confirm starts",
        ):
            verify_preconfirm_receipt(
                self._make_receipt(),
                frozen_signer_public_key_raw=self.public_key_raw,
                expected_bindings=self.bindings,
                expected_signer_key_id="offline-signer-2026q3",
                external_signer_key_not_held_by_runner=True,
                receipt_verified_before_confirm_start=False,
            )

    def test_receipt_does_not_serialize_private_key_material(self) -> None:
        receipt = self._make_receipt()
        parsed = json.loads(receipt.decode("utf-8"))

        self.assertNotIn("private_key", parsed)
        self.assertNotIn("private_key", parsed["payload"])
        self.assertNotIn(
            base64.b64encode(self.private_key_raw),
            receipt,
        )


if __name__ == "__main__":
    unittest.main()

