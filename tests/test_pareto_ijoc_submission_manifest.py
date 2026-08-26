from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

import mo_nco.pareto_ijoc_submission_manifest as submission_manifest_module
from mo_nco.pareto_ijoc_submission_manifest import (
    SubmissionArtifactError,
    audit_submission_receipt,
    build_submission_manifest,
    verify_submission_manifest,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value))


def _base_spec(paths: list[tuple[str, str]]) -> dict[str, object]:
    return {
        "schema": "ijoc_final_submission_required_files_v1",
        "journal_target": "INFORMS Journal on Computing",
        "declared_submission_status": "HOLD",
        "release_metadata": {
            "immutable_revision": None,
            "public_repository_url": None,
            "release_tag": None,
        },
        "required_files": [
            {"artifact_role": role, "path": path} for path, role in paths
        ],
        "readiness_scan": {
            "author_placeholder_paths": [],
            "hold_marker_paths": [],
        },
    }


def test_builds_canonical_manifest_from_only_explicit_required_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "packet"
    (root / "manuscript").mkdir(parents=True)
    source = root / "manuscript" / "paper.tex"
    pdf = root / "manuscript" / "paper.pdf"
    source.write_text("final source\n", encoding="utf-8")
    pdf.write_bytes(b"%PDF-final")
    ignored = root / "unlisted-large-run.bin"
    ignored.write_bytes(b"not part of the final manifest")
    spec_path = tmp_path / "required.json"
    _write_canonical(
        spec_path,
        _base_spec(
            [
                ("manuscript/paper.tex", "main_manuscript_source"),
                ("manuscript/paper.pdf", "main_manuscript_pdf"),
            ]
        ),
    )
    manifest_path = root / "final_artifact_manifest.json"

    result = build_submission_manifest(root, spec_path, manifest_path)

    raw = manifest_path.read_bytes()
    parsed = json.loads(raw.decode("utf-8"))
    assert raw == _canonical_bytes(parsed)
    assert result.manifest_sha256 == hashlib.sha256(raw).hexdigest()
    assert parsed["required_file_count"] == 2
    assert [item["path"] for item in parsed["artifacts"]] == [
        "manuscript/paper.pdf",
        "manuscript/paper.tex",
    ]
    assert parsed["artifacts"][0] == {
        "artifact_role": "main_manuscript_pdf",
        "bytes": len(b"%PDF-final"),
        "path": "manuscript/paper.pdf",
        "sha256": hashlib.sha256(b"%PDF-final").hexdigest(),
    }
    assert "unlisted-large-run.bin" not in raw.decode("utf-8")


def test_preserves_author_placeholders_and_hold_markers_as_hold_status(
    tmp_path: Path,
) -> None:
    root = tmp_path / "packet"
    (root / "manuscript").mkdir(parents=True)
    paper = root / "manuscript" / "paper.tex"
    status = root / "STATUS.json"
    paper.write_text(
        "\\author{Author names and affiliations to be inserted before submission}\n",
        encoding="utf-8",
    )
    status.write_text(
        '{"post_run_audit":"NOT_RUN","submission_verdict":"HOLD"}\n',
        encoding="utf-8",
    )
    spec = _base_spec(
        [
            ("manuscript/paper.tex", "main_manuscript_source"),
            ("STATUS.json", "formal_study_status"),
        ]
    )
    spec["declared_submission_status"] = "READY"
    spec["readiness_scan"] = {
        # Role-driven mandatory scans cannot be disabled by empty lists.
        "author_placeholder_paths": [],
        "hold_marker_paths": [],
    }
    spec_path = tmp_path / "required.json"
    _write_canonical(spec_path, spec)
    manifest_path = root / "final_artifact_manifest.json"

    result = build_submission_manifest(root, spec_path, manifest_path)

    parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    readiness = parsed["readiness_status"]
    assert result.effective_submission_status == "HOLD"
    assert readiness["declared_submission_status"] == "READY"
    assert readiness["effective_submission_status"] == "HOLD"
    assert readiness["author_placeholders"]["status"] == "PRESENT"
    assert readiness["author_placeholders"]["finding_count"] == 1
    assert readiness["author_placeholders"]["findings"] == [
        {
            "line": 1,
            "marker": "AUTHOR_NAMES_AND_AFFILIATIONS_TO_BE_INSERTED",
            "path": "manuscript/paper.tex",
        }
    ]
    assert readiness["hold_markers"]["status"] == "PRESENT"
    assert readiness["hold_markers"]["finding_count"] == 2


def test_verify_recomputes_manifest_and_writes_canonical_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "packet"
    (root / "manuscript").mkdir(parents=True)
    (root / "release").mkdir()
    pdf = root / "manuscript" / "paper.pdf"
    source_archive = root / "release" / "source.tar.gz"
    formal_source_archive = root / "release" / "formal-source.tar.gz"
    pdf.write_bytes(b"%PDF-final")
    source_archive.write_bytes(b"source archive")
    formal_source_archive.write_bytes(b"formal source archive")
    spec = _base_spec(
        [
            ("manuscript/paper.pdf", "main_manuscript_pdf"),
            ("release/source.tar.gz", "release_source_archive"),
            ("release/formal-source.tar.gz", "formal_source_archive"),
        ]
    )
    spec["release_metadata"] = {
        "immutable_revision": "0123456789abcdef",
        "public_repository_url": "https://example.invalid/repository",
        "release_tag": "ijoc-v20-final",
    }
    spec_path = tmp_path / "required.json"
    _write_canonical(spec_path, spec)
    manifest_path = root / "final_artifact_manifest.json"
    receipt_path = root / "final_artifact_receipt.json"
    build_submission_manifest(root, spec_path, manifest_path)

    result = verify_submission_manifest(
        root,
        spec_path,
        manifest_path,
        receipt_path,
        verifier_role="independent-release-auditor",
        verified_at="2026-07-31T12:34:56+00:00",
    )

    raw = receipt_path.read_bytes()
    receipt = json.loads(raw.decode("utf-8"))
    assert raw == _canonical_bytes(receipt)
    assert result.receipt_sha256 == hashlib.sha256(raw).hexdigest()
    assert receipt["integrity_status"] == "PASS"
    assert receipt["submission_status"] == "HOLD"
    assert receipt["manifest"] == {
        "bytes": manifest_path.stat().st_size,
        "path": "final_artifact_manifest.json",
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    assert receipt["final_pdf_artifacts"][0]["path"] == "manuscript/paper.pdf"
    assert [item["path"] for item in receipt["source_archive_artifacts"]] == [
        "release/formal-source.tar.gz",
        "release/source.tar.gz",
    ]
    implementation = receipt["verification_implementation"]
    module_bytes = Path(submission_manifest_module.__file__).read_bytes()
    assert implementation == {
        "cryptographic_signature_status": "NOT_PERFORMED",
        "independence_attestation_scope": "VERIFIER_ROLE_DECLARATION_ONLY",
        "module": "mo_nco.pareto_ijoc_submission_manifest",
        "module_bytes": len(module_bytes),
        "module_sha256": hashlib.sha256(module_bytes).hexdigest(),
    }

    pdf.write_bytes(b"%PDF-drifted")
    with pytest.raises(SubmissionArtifactError, match="hash drift"):
        verify_submission_manifest(
            root,
            spec_path,
            manifest_path,
            receipt_path,
            verifier_role="independent-release-auditor",
            verified_at="2026-07-31T12:35:00+00:00",
        )


def test_audit_receipt_recomputes_manifest_and_rejects_noncanonical_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "packet"
    root.mkdir()
    artifact = root / "status.json"
    artifact.write_bytes(b'{"submission":"HOLD"}\n')
    spec_path = tmp_path / "required.json"
    _write_canonical(
        spec_path,
        _base_spec([("status.json", "submission_status")]),
    )
    manifest_path = root / "manifest.json"
    receipt_path = root / "receipt.json"
    build_submission_manifest(root, spec_path, manifest_path)
    verify_submission_manifest(
        root,
        spec_path,
        manifest_path,
        receipt_path,
        verifier_role="second-person-check",
        verified_at="2026-07-31T13:00:00+00:00",
    )

    result = audit_submission_receipt(
        root, spec_path, manifest_path, receipt_path
    )
    assert result.receipt_sha256 == hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(SubmissionArtifactError, match="not canonical"):
        audit_submission_receipt(root, spec_path, manifest_path, receipt_path)


def test_rejects_missing_required_file_without_writing_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "packet"
    root.mkdir()
    spec_path = tmp_path / "required.json"
    _write_canonical(
        spec_path,
        _base_spec([("missing.pdf", "main_manuscript_pdf")]),
    )
    output = root / "manifest.json"

    with pytest.raises(SubmissionArtifactError, match="missing.pdf"):
        build_submission_manifest(root, spec_path, output)

    assert not output.exists()


def test_rejects_duplicate_required_paths_case_insensitively(
    tmp_path: Path,
) -> None:
    root = tmp_path / "packet"
    root.mkdir()
    (root / "paper.pdf").write_bytes(b"pdf")
    spec = _base_spec([("paper.pdf", "main_manuscript_pdf")])
    spec["required_files"].append(
        {"artifact_role": "supplement_pdf", "path": "PAPER.PDF"}
    )
    spec_path = tmp_path / "required.json"
    _write_canonical(spec_path, spec)

    with pytest.raises(SubmissionArtifactError, match="Duplicate required path"):
        build_submission_manifest(root, spec_path, root / "manifest.json")


@pytest.mark.parametrize("unsafe_path", ["../outside.pdf", "/absolute.pdf", "C:/escape.pdf"])
def test_rejects_path_escape_and_absolute_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    root = tmp_path / "packet"
    root.mkdir()
    spec_path = tmp_path / "required.json"
    _write_canonical(
        spec_path,
        _base_spec([(unsafe_path, "main_manuscript_pdf")]),
    )

    with pytest.raises(SubmissionArtifactError, match="relative path"):
        build_submission_manifest(root, spec_path, root / "manifest.json")


def test_rejects_noncanonical_required_file_spec(tmp_path: Path) -> None:
    root = tmp_path / "packet"
    root.mkdir()
    (root / "paper.pdf").write_bytes(b"pdf")
    spec_path = tmp_path / "required.json"
    spec_path.write_text(
        json.dumps(
            _base_spec([("paper.pdf", "main_manuscript_pdf")]),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SubmissionArtifactError, match="not canonical"):
        build_submission_manifest(root, spec_path, root / "manifest.json")


def test_rejects_manifest_output_overwriting_required_file_spec(
    tmp_path: Path,
) -> None:
    root = tmp_path / "packet"
    root.mkdir()
    (root / "paper.pdf").write_bytes(b"pdf")
    spec_path = root / "required.json"
    _write_canonical(
        spec_path,
        _base_spec([("paper.pdf", "main_manuscript_pdf")]),
    )
    original = spec_path.read_bytes()

    with pytest.raises(SubmissionArtifactError, match="required-file spec"):
        build_submission_manifest(root, spec_path, spec_path)

    assert spec_path.read_bytes() == original


def test_rejects_receipt_output_overwriting_required_file_spec(
    tmp_path: Path,
) -> None:
    root = tmp_path / "packet"
    root.mkdir()
    (root / "paper.pdf").write_bytes(b"pdf")
    spec_path = root / "required.json"
    _write_canonical(
        spec_path,
        _base_spec([("paper.pdf", "main_manuscript_pdf")]),
    )
    original = spec_path.read_bytes()
    manifest_path = root / "manifest.json"
    build_submission_manifest(root, spec_path, manifest_path)

    with pytest.raises(SubmissionArtifactError, match="required-file spec"):
        verify_submission_manifest(
            root,
            spec_path,
            manifest_path,
            spec_path,
            verifier_role="independent-release-auditor",
            verified_at="2026-07-31T12:34:56+00:00",
        )

    assert spec_path.read_bytes() == original


def test_cli_build_verify_and_audit_round_trip(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    cli = repository / "scripts" / "manage_ijoc_submission_artifacts.py"
    root = tmp_path / "packet"
    root.mkdir()
    (root / "paper.pdf").write_bytes(b"%PDF-cli")
    spec_path = tmp_path / "required.json"
    _write_canonical(
        spec_path,
        _base_spec([("paper.pdf", "main_manuscript_pdf")]),
    )
    manifest = root / "manifest.json"
    receipt = root / "receipt.json"

    build = subprocess.run(
        [
            sys.executable,
            str(cli),
            "build",
            "--packet-root",
            str(root),
            "--required-file-spec",
            str(spec_path),
            "--manifest-output",
            str(manifest),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(build.stdout)["required_file_count"] == 1

    verify = subprocess.run(
        [
            sys.executable,
            str(cli),
            "verify",
            "--packet-root",
            str(root),
            "--required-file-spec",
            str(spec_path),
            "--manifest",
            str(manifest),
            "--receipt-output",
            str(receipt),
            "--verifier-role",
            "cli-independent-auditor",
            "--verified-at",
            "2026-07-31T14:00:00+00:00",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(verify.stdout)["integrity_status"] == "PASS"

    audit = subprocess.run(
        [
            sys.executable,
            str(cli),
            "audit",
            "--packet-root",
            str(root),
            "--required-file-spec",
            str(spec_path),
            "--manifest",
            str(manifest),
            "--receipt",
            str(receipt),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(audit.stdout)["integrity_status"] == "PASS"

