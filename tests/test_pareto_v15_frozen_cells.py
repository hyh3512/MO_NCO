from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from mo_nco.pareto_frozen_cells import (
    FrozenCellManifestError,
    canonical_manifest_payload,
    load_frozen_cell_manifest,
)


class FrozenCellManifestV15Tests(unittest.TestCase):
    def _write_payload(
        self,
        directory: str,
        payload: dict[str, object],
    ) -> tuple[Path, str]:
        raw = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        path = Path(directory) / "cells.json"
        path.write_bytes(raw)
        return path, hashlib.sha256(raw).hexdigest()

    def test_exact_manifest_binds_complete_partition_and_observables(self) -> None:
        payload = canonical_manifest_payload(
            lower=(0, 0),
            upper=(1, 1),
            widths=(Fraction(1, 2), Fraction(1, 2)),
            observable_cells=((0, 0), (1, 1)),
        )
        with tempfile.TemporaryDirectory() as directory:
            path, digest = self._write_payload(directory, payload)
            manifest = load_frozen_cell_manifest(
                path,
                expected_sha256=digest,
            )

        self.assertEqual(
            manifest.classify((Fraction(1, 2), Fraction(1, 2))),
            (1, 1),
        )
        self.assertEqual(manifest.classify((1, 1)), (1, 1))
        self.assertTrue(manifest.is_observable((1, 1)))
        self.assertFalse(manifest.is_observable((1, 0)))

    def test_one_unit_exact_box_escape_fails_without_clipping(self) -> None:
        payload = canonical_manifest_payload(
            lower=(0,),
            upper=(1,),
            widths=(Fraction(1, 2),),
            observable_cells=((0,),),
        )
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self._write_payload(directory, payload)
            manifest = load_frozen_cell_manifest(path)

        with self.assertRaises(FrozenCellManifestError):
            manifest.classify((Fraction(-1, 10**30),))
        with self.assertRaises(FrozenCellManifestError):
            manifest.classify((Fraction(10**30 + 1, 10**30),))

    def test_omitted_partition_cell_and_hash_mismatch_fail_closed(self) -> None:
        payload = canonical_manifest_payload(
            lower=(0,),
            upper=(1,),
            widths=(Fraction(1, 2),),
            observable_cells=((0,),),
        )
        payload["partition_cells"] = [[0]]
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self._write_payload(directory, payload)
            with self.assertRaisesRegex(
                FrozenCellManifestError,
                "complete Cartesian grid",
            ):
                load_frozen_cell_manifest(path)
            with self.assertRaisesRegex(
                FrozenCellManifestError,
                "SHA-256",
            ):
                load_frozen_cell_manifest(path, expected_sha256="0" * 64)

    def test_numeric_json_box_and_legacy_metric_semantics_are_rejected(self) -> None:
        payload = canonical_manifest_payload(
            lower=(0,),
            upper=(1,),
            widths=(Fraction(1, 2),),
            observable_cells=((0,),),
        )
        box = payload["box"]
        self.assertIsInstance(box, dict)
        box["lower"] = [0.0]
        metric = payload["metric"]
        self.assertIsInstance(metric, dict)
        metric["reference_aggregation"] = "lp_power_mean_over_references_v1"
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self._write_payload(directory, payload)
            with self.assertRaises(FrozenCellManifestError):
                load_frozen_cell_manifest(path)

    def test_programmatic_builder_and_classifier_reject_binary_float_inputs(self) -> None:
        with self.assertRaises(FrozenCellManifestError):
            canonical_manifest_payload(
                lower=(0.0,),
                upper=(1,),
                widths=(Fraction(1, 2),),
                observable_cells=((0,),),
            )

        payload = canonical_manifest_payload(
            lower=(0,),
            upper=(1,),
            widths=(Fraction(1, 2),),
            observable_cells=((0,),),
        )
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self._write_payload(directory, payload)
            manifest = load_frozen_cell_manifest(path)

        with self.assertRaises(FrozenCellManifestError):
            manifest.classify((0.5,))


if __name__ == "__main__":
    unittest.main()

