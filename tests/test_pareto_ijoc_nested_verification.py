from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from mo_nco.pareto_ijoc_nested_verification import (
    NestedEvidenceError,
    NestedEvidenceExpectations,
    canonical_json_bytes,
    verify_nested_evidence,
)


TOP_LEVEL_FROZEN = (
    "algorithm_configuration_matrix",
    "execution_plan",
    "formal_analysis_plan",
    "freeze_receipt",
    "metric_reference_manifest",
    "study",
)
TOP_LEVEL_RESULTS = ("matrix_invocation", "post_run_audit")
ROW_ARTIFACTS = (
    "algorithm_result",
    "all_evaluated_archive",
    "checkpoint_witnesses",
    "replay_receipt",
    "terminal_receipt",
)
_ROOT = Path(__file__).resolve().parents[1]


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_digest(value: object) -> str:
    return _digest(canonical_json_bytes(value, trailing_newline=False))


def _write(path: Path, raw: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {
        "bytes": len(raw),
        "path": path.as_posix(),
        "sha256": _digest(raw),
    }


class _Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.frozen = root / "frozen"
        self.results = root / "results"
        self.analysis = root / "analysis"
        self.release = root / "release"
        self.consumed = self.analysis / "consumed_artifacts_manifest.json"
        self.source_manifest = self.release / "source_file_manifest.json"
        self.source_archive = self.release / "source.tar.gz"
        self.receipt = root / "provenance" / "nested_receipt.json"
        self.receipt.parent.mkdir(parents=True)
        self.expectations = NestedEvidenceExpectations(
            row_entries=2,
            row_artifacts=10,
            top_level_inputs=8,
            metric_reference_sources=1,
            source_files=2,
        )
        self._build_consumed_manifest()
        self._build_source_archive()

    def _relative_binding(self, root: Path, relative: str, raw: bytes):
        binding = _write(root / relative, raw)
        binding["path"] = relative
        return binding

    def _build_consumed_manifest(self) -> None:
        top: dict[str, dict[str, object]] = {}
        for name in TOP_LEVEL_FROZEN:
            relative = f"inputs/{name}.json"
            top[name] = self._relative_binding(
                self.frozen, relative, f"{name}\n".encode()
            )
        for name in TOP_LEVEL_RESULTS:
            relative = f"inputs/{name}.json"
            top[name] = self._relative_binding(
                self.results, relative, f"{name}\n".encode()
            )

        metric = self._relative_binding(
            self.frozen, "artifacts/reference.json", b"metric-reference\n"
        )
        metric["case_id"] = "case-0"

        rows = []
        for seed in range(2):
            run_key = {
                "algorithm": "algorithm-a",
                "budget": 100,
                "case_id": "case-0",
                "seed": seed,
            }
            run_sha = _canonical_digest(run_key)
            artifacts = {}
            for artifact_name in ROW_ARTIFACTS:
                filename = (
                    "replay_result.json"
                    if artifact_name == "replay_receipt"
                    else f"{artifact_name}.json"
                )
                relative = f"runs/{run_sha}/{filename}"
                artifacts[artifact_name] = self._relative_binding(
                    self.results,
                    relative,
                    f"{run_sha}:{artifact_name}\n".encode(),
                )
            rows.append(
                {
                    "artifacts": artifacts,
                    "run_key": run_key,
                    "run_key_sha256": run_sha,
                }
            )
        rows.sort(key=lambda row: str(row["run_key_sha256"]))
        manifest = {
            "consumed_row_artifact_set_sha256": _canonical_digest(rows),
            "metric_reference_sources": [metric],
            "row_artifact_count": 10,
            "row_artifacts": rows,
            "schema": "ijoc_formal_analysis_consumed_artifacts_v1",
            "terminal_receipt_set_sha256": _canonical_digest(
                [row["artifacts"]["terminal_receipt"] for row in rows]
            ),
            "top_level_inputs": top,
        }
        self.consumed.parent.mkdir(parents=True, exist_ok=True)
        self.consumed.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    def _build_source_archive(self) -> None:
        prefix = "source-packet"
        source_files = {
            "mo_nco/algorithm.py": b"ALGORITHM = 'fixture'\n",
            "tests/test_algorithm.py": b"def test_fixture():\n    assert True\n",
        }
        entries = [
            {"bytes": len(raw), "path": path, "sha256": _digest(raw)}
            for path, raw in sorted(source_files.items())
        ]
        manifest = {
            "archive_prefix": prefix,
            "file_count": len(entries),
            "files": entries,
            "schema": "ijoc_source_file_manifest_v1",
        }
        manifest_raw = canonical_json_bytes(manifest, trailing_newline=True)
        self.source_manifest.parent.mkdir(parents=True, exist_ok=True)
        self.source_manifest.write_bytes(manifest_raw)
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for relative, raw in sorted(source_files.items()):
                    info = tarfile.TarInfo(f"{prefix}/{relative}")
                    info.size = len(raw)
                    archive.addfile(info, io.BytesIO(raw))
                info = tarfile.TarInfo(f"{prefix}/source_file_manifest.json")
                info.size = len(manifest_raw)
                archive.addfile(info, io.BytesIO(manifest_raw))
        self.source_archive.write_bytes(buffer.getvalue())

    def evidence_hashes(self) -> dict[str, str]:
        paths = [
            self.consumed,
            self.source_manifest,
            self.source_archive,
            *self.frozen.rglob("*"),
            *self.results.rglob("*"),
        ]
        return {
            path.relative_to(self.root).as_posix(): _digest(path.read_bytes())
            for path in paths
            if path.is_file()
        }


class NestedEvidenceVerificationTests(unittest.TestCase):
    def test_complete_nested_evidence_is_rehashed_without_mutating_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = _Fixture(Path(temporary_directory))
            before = fixture.evidence_hashes()

            result = verify_nested_evidence(
                packet_root=fixture.root,
                consumed_manifest=fixture.consumed,
                frozen_root=fixture.frozen,
                results_root=fixture.results,
                source_manifest=fixture.source_manifest,
                source_archive=fixture.source_archive,
                receipt_output=fixture.receipt,
                expectations=fixture.expectations,
                workers=2,
            )

            self.assertEqual(result.status, "PASS")
            self.assertEqual(result.issue_count, 0)
            self.assertTrue(fixture.receipt.is_file())
            receipt = json.loads(fixture.receipt.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["issues"], [])
            self.assertEqual(
                receipt["consumed_artifacts"]["verified_counts"],
                {
                    "metric_reference_sources": 1,
                    "row_artifacts": 10,
                    "row_entries": 2,
                    "top_level_inputs": 8,
                },
            )
            self.assertEqual(
                receipt["source_archive"]["verified_counts"]["source_files"],
                2,
            )
            self.assertEqual(fixture.evidence_hashes(), before)

    def test_tampered_row_artifact_produces_fail_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = _Fixture(Path(temporary_directory))
            manifest = json.loads(fixture.consumed.read_text(encoding="utf-8"))
            relative = manifest["row_artifacts"][0]["artifacts"][
                "algorithm_result"
            ]["path"]
            (fixture.results / relative).write_bytes(b"tampered\n")

            result = verify_nested_evidence(
                packet_root=fixture.root,
                consumed_manifest=fixture.consumed,
                frozen_root=fixture.frozen,
                results_root=fixture.results,
                source_manifest=fixture.source_manifest,
                source_archive=fixture.source_archive,
                receipt_output=fixture.receipt,
                expectations=fixture.expectations,
                workers=2,
            )

            self.assertEqual(result.status, "FAIL")
            receipt = json.loads(fixture.receipt.read_text(encoding="utf-8"))
            self.assertEqual(
                receipt["consumed_artifacts"]["verified_counts"]["row_entries"],
                1,
            )
            self.assertEqual(
                receipt["consumed_artifacts"]["verified_counts"]["row_artifacts"],
                9,
            )
            self.assertIn(
                "FILE_SHA256_MISMATCH",
                {issue["code"] for issue in receipt["issues"]},
            )

    def test_external_and_internal_source_manifests_must_match_byte_for_byte(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = _Fixture(Path(temporary_directory))
            external = json.loads(
                fixture.source_manifest.read_text(encoding="utf-8")
            )
            external["files"][0]["sha256"] = "0" * 64
            fixture.source_manifest.write_bytes(
                canonical_json_bytes(external, trailing_newline=True)
            )

            result = verify_nested_evidence(
                packet_root=fixture.root,
                consumed_manifest=fixture.consumed,
                frozen_root=fixture.frozen,
                results_root=fixture.results,
                source_manifest=fixture.source_manifest,
                source_archive=fixture.source_archive,
                receipt_output=fixture.receipt,
                expectations=fixture.expectations,
                workers=2,
            )

            self.assertEqual(result.status, "FAIL")
            receipt = json.loads(fixture.receipt.read_text(encoding="utf-8"))
            codes = {issue["code"] for issue in receipt["issues"]}
            self.assertIn("INTERNAL_EXTERNAL_MANIFEST_MISMATCH", codes)
            self.assertIn("SOURCE_FILE_SHA256_MISMATCH", codes)
            self.assertEqual(
                receipt["source_archive"]["verified_counts"]["source_files"],
                1,
            )

    def test_existing_receipt_is_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = _Fixture(Path(temporary_directory))
            fixture.receipt.write_bytes(b"immutable-prior-receipt")

            with self.assertRaises(NestedEvidenceError):
                verify_nested_evidence(
                    packet_root=fixture.root,
                    consumed_manifest=fixture.consumed,
                    frozen_root=fixture.frozen,
                    results_root=fixture.results,
                    source_manifest=fixture.source_manifest,
                    source_archive=fixture.source_archive,
                    receipt_output=fixture.receipt,
                    expectations=fixture.expectations,
                    workers=2,
                )

            self.assertEqual(
                fixture.receipt.read_bytes(), b"immutable-prior-receipt"
            )

    def test_declared_path_escape_is_rejected_without_reading_outside_root(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = _Fixture(Path(temporary_directory))
            outside = fixture.root.parent / f"{fixture.root.name}-outside.txt"
            outside.write_bytes(b"must-not-be-consumed\n")
            try:
                manifest = json.loads(
                    fixture.consumed.read_text(encoding="utf-8")
                )
                binding = manifest["top_level_inputs"]["study"]
                binding["path"] = f"../{outside.name}"
                binding["bytes"] = outside.stat().st_size
                binding["sha256"] = _digest(outside.read_bytes())
                fixture.consumed.write_text(
                    json.dumps(
                        manifest,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                    newline="\n",
                )

                result = verify_nested_evidence(
                    packet_root=fixture.root,
                    consumed_manifest=fixture.consumed,
                    frozen_root=fixture.frozen,
                    results_root=fixture.results,
                    source_manifest=fixture.source_manifest,
                    source_archive=fixture.source_archive,
                    receipt_output=fixture.receipt,
                    expectations=fixture.expectations,
                    workers=2,
                )

                self.assertEqual(result.status, "FAIL")
                receipt = json.loads(
                    fixture.receipt.read_text(encoding="utf-8")
                )
                self.assertIn(
                    "INVALID_BINDING",
                    {issue["code"] for issue in receipt["issues"]},
                )
                self.assertEqual(
                    receipt["consumed_artifacts"]["verified_counts"][
                        "top_level_inputs"
                    ],
                    7,
                )
            finally:
                outside.unlink(missing_ok=True)

    def test_cli_emits_machine_readable_pass_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = _Fixture(Path(temporary_directory))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(_ROOT / "scripts" / "verify_ijoc_nested_manifests.py"),
                    "--packet-root",
                    str(fixture.root),
                    "--consumed-manifest",
                    str(fixture.consumed),
                    "--frozen-root",
                    str(fixture.frozen),
                    "--results-root",
                    str(fixture.results),
                    "--source-manifest",
                    str(fixture.source_manifest),
                    "--source-archive",
                    str(fixture.source_archive),
                    "--receipt-output",
                    str(fixture.receipt),
                    "--expected-row-entries",
                    "2",
                    "--expected-row-artifacts",
                    "10",
                    "--expected-top-level-inputs",
                    "8",
                    "--expected-metric-reference-sources",
                    "1",
                    "--expected-source-files",
                    "2",
                    "--workers",
                    "2",
                ],
                cwd=_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["issue_count"], 0)
            self.assertEqual(
                summary["receipt_sha256"], _digest(fixture.receipt.read_bytes())
            )


if __name__ == "__main__":
    unittest.main()

