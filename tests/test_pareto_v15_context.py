from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import unittest

from mo_nco.pareto_v15_context import (
    V15_CERTIFICATE_CONTEXT_SCHEMA,
    V15CertificateContext,
    V15CertificateContextError,
    validate_v15_context_sha256,
    verify_v15_context_sha256,
)


class V15CertificateContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fields = {
            "case_id": "kroAB100-case_01",
            "instance_sha256": "1" * 64,
            "configuration_sha256": "2" * 64,
            "cell_manifest_sha256": "3" * 64,
            "reference_sha256": "4" * 64,
            "type_cell_plan_sha256": "5" * 64,
            "pilot_plan_sha256": "6" * 64,
            "confirm_plan_sha256": "7" * 64,
        }

    def _context(self) -> V15CertificateContext:
        return V15CertificateContext(**self.fields)

    def test_all_fields_are_mandatory_and_serialized(self) -> None:
        with self.assertRaises(TypeError):
            V15CertificateContext()  # type: ignore[call-arg]
        for missing in self.fields:
            incomplete = dict(self.fields)
            incomplete.pop(missing)
            with self.subTest(missing=missing):
                with self.assertRaises(TypeError):
                    V15CertificateContext(
                        **incomplete  # type: ignore[arg-type]
                    )

        context = self._context()
        payload = context.to_jsonable()
        self.assertEqual(
            payload["schema"],
            V15_CERTIFICATE_CONTEXT_SCHEMA,
        )
        self.assertEqual(
            payload["context_sha256"],
            context.context_sha256,
        )
        self.assertEqual(
            set(payload),
            {
                "schema",
                *self.fields,
                "context_sha256",
            },
        )
        json.dumps(payload, allow_nan=False)

    def test_digest_matches_independent_canonical_json_computation(
        self,
    ) -> None:
        context = self._context()
        preimage = {
            "schema": V15_CERTIFICATE_CONTEXT_SCHEMA,
            **self.fields,
        }
        expected_bytes = json.dumps(
            preimage,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        expected_digest = hashlib.sha256(expected_bytes).hexdigest()
        self.assertEqual(context.canonical_json_bytes(), expected_bytes)
        self.assertEqual(context.context_sha256, expected_digest)

    def test_determinism_is_independent_of_argument_mapping_order(
        self,
    ) -> None:
        forward = V15CertificateContext(**self.fields)
        reverse = V15CertificateContext(
            **dict(reversed(tuple(self.fields.items())))
        )
        self.assertEqual(
            forward.canonical_json_bytes(),
            reverse.canonical_json_bytes(),
        )
        self.assertEqual(forward.context_sha256, reverse.context_sha256)
        self.assertEqual(forward.to_jsonable(), reverse.to_jsonable())

    def test_each_field_tamper_changes_digest_and_old_hash_fails(
        self,
    ) -> None:
        original = self._context()
        original_digest = original.context_sha256
        replacements = {
            "case_id": "kroAB100-case_02",
            "instance_sha256": "8" * 64,
            "configuration_sha256": "9" * 64,
            "cell_manifest_sha256": "a" * 64,
            "reference_sha256": "b" * 64,
            "type_cell_plan_sha256": "c" * 64,
            "pilot_plan_sha256": "d" * 64,
            "confirm_plan_sha256": "e" * 64,
        }
        for field, value in replacements.items():
            with self.subTest(field=field):
                tampered = replace(original, **{field: value})
                self.assertNotEqual(
                    tampered.context_sha256,
                    original_digest,
                )
                with self.assertRaisesRegex(
                    V15CertificateContextError,
                    "does not match",
                ):
                    verify_v15_context_sha256(
                        tampered,
                        original_digest,
                    )
                self.assertIs(
                    verify_v15_context_sha256(
                        tampered,
                        tampered.context_sha256,
                    ),
                    tampered,
                )

    def test_hashes_must_be_lowercase_canonical_sha256(self) -> None:
        invalid_hashes: tuple[object, ...] = (
            "",
            "a" * 63,
            "a" * 65,
            "A" * 64,
            "g" * 64,
            "0x" + "a" * 62,
            "a" * 63 + "\n",
            None,
            False,
        )
        hash_fields = tuple(
            field
            for field in self.fields
            if field.endswith("_sha256")
        )
        for field in hash_fields:
            for invalid in invalid_hashes:
                values = dict(self.fields)
                values[field] = invalid
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaises(
                        V15CertificateContextError
                    ):
                        V15CertificateContext(
                            **values  # type: ignore[arg-type]
                        )

    def test_case_id_is_nonempty_ascii_and_path_safe(self) -> None:
        invalid_case_ids: tuple[object, ...] = (
            "",
            " case",
            "case ",
            "case id",
            "../case",
            r"case\child",
            "case/child",
            "case@domain",
            "案例",
            "a" * 129,
            None,
            True,
        )
        for invalid in invalid_case_ids:
            values = dict(self.fields)
            values["case_id"] = invalid
            with self.subTest(invalid=invalid):
                with self.assertRaises(V15CertificateContextError):
                    V15CertificateContext(
                        **values  # type: ignore[arg-type]
                    )

    def test_hash_verifier_rejects_noncanonical_and_wrong_values(
        self,
    ) -> None:
        context = self._context()
        self.assertIs(
            validate_v15_context_sha256(
                context,
                context.context_sha256,
            ),
            context,
        )
        for invalid in (
            context.context_sha256.upper(),
            "f" * 64,
            "",
            None,
            True,
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(V15CertificateContextError):
                    verify_v15_context_sha256(context, invalid)
        with self.assertRaises(V15CertificateContextError):
            verify_v15_context_sha256(  # type: ignore[arg-type]
                self.fields,
                context.context_sha256,
            )


if __name__ == "__main__":
    unittest.main()

