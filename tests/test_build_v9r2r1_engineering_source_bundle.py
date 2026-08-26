from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts" / "build_v9r2r1_engineering_source_bundle.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("v9r2r1_source_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load_builder()


@pytest.fixture(scope="module")
def built_pair(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("v9r2r1_source_bundles")
    first = root / "first"
    second = root / "second"
    first.mkdir()
    second.mkdir()
    receipt_a = BUILDER.build_bundle(ROOT, first)
    receipt_b = BUILDER.build_bundle(ROOT, second)
    return first, second, receipt_a, receipt_b


def test_source_bundle_is_reproducible_and_self_verified(built_pair) -> None:
    first, second, receipt_a, receipt_b = built_pair
    archive_a = first / BUILDER.ARCHIVE_NAME
    archive_b = second / BUILDER.ARCHIVE_NAME
    assert archive_a.read_bytes() == archive_b.read_bytes()
    assert receipt_a["status"] == "PASS_ENGINEERING_SOURCE_FREEZE_CANDIDATE_ONLY"
    assert receipt_b["status"] == receipt_a["status"]
    assert receipt_a["verification"]["archive_sha256"] == hashlib.sha256(
        archive_a.read_bytes()
    ).hexdigest()
    assert receipt_a["full_source_freeze_requirement_satisfied"] is False
    assert receipt_a["full_development_matrix_authorized"] is False


def test_manifest_is_canonical_self_hashed_and_includes_integration_fix(
    built_pair,
) -> None:
    first, _second, _receipt_a, _receipt_b = built_pair
    manifest_raw = (first / BUILDER.MANIFEST_NAME).read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8"))
    assert manifest_raw == BUILDER._canonical_json(manifest) + b"\n"
    declared = manifest.pop("manifest_payload_sha256")
    assert declared == hashlib.sha256(BUILDER._canonical_json(manifest)).hexdigest()
    paths = {entry["path"]: entry for entry in manifest["files"]}
    coverage = (
        "ijoc_submission_v21e3r1/scripts/"
        "run_v21e3r1_same_implementation_branch_replay_coverage.py"
    )
    coverage_test = (
        "tests/test_v21e3r1_same_implementation_branch_replay_coverage.py"
    )
    diagnostic_runner = (
        "ijoc_submission_v21e3r1/scripts/"
        "run_v21e3r1_development_diagnostics.py"
    )
    for required in (coverage, coverage_test, diagnostic_runner):
        assert required in paths
        raw = (ROOT / required).read_bytes()
        assert paths[required] == {
            "path": required,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    assert manifest["identity"] == {
        "algorithm_predecessor": "V21E3R1_V9R2",
        "algorithm_semantics_changed": False,
        "distribution": "mo-nco",
        "revision": "V21E3R1_V9R2R1",
        "version": "0.21.3.14",
    }
    assert manifest["bound_into_authorizing_protocol"] is False
    assert manifest["full_source_freeze_requirement_satisfied"] is False


def test_zip_exact_closure_metadata_and_detached_manifest_match(built_pair) -> None:
    first, _second, receipt_a, _receipt_b = built_pair
    archive_path = first / BUILDER.ARCHIVE_NAME
    manifest_path = first / BUILDER.MANIFEST_NAME
    verification = BUILDER.verify_bundle(manifest_path, archive_path, root=ROOT)
    assert verification == receipt_a["verification"]
    with zipfile.ZipFile(archive_path, "r") as archive:
        infos = archive.infolist()
        assert infos[-1].filename == BUILDER.MANIFEST_NAME
        assert archive.read(BUILDER.MANIFEST_NAME) == manifest_path.read_bytes()
        assert all(info.date_time == BUILDER._FIXED_ZIP_TIME for info in infos)
        assert all(info.create_system == 3 for info in infos)
        assert all((info.external_attr >> 16) == 0o100644 for info in infos)


def test_verifier_fails_closed_on_detached_manifest_drift(
    built_pair, tmp_path: Path
) -> None:
    first, _second, _receipt_a, _receipt_b = built_pair
    manifest = tmp_path / BUILDER.MANIFEST_NAME
    shutil.copyfile(first / BUILDER.MANIFEST_NAME, manifest)
    raw = manifest.read_bytes()
    manifest.write_bytes(raw.replace(b'"file_count":', b'"file_count" :', 1))
    with pytest.raises(BUILDER.SourceBundleError, match="canonical JSON"):
        BUILDER.verify_bundle(manifest, first / BUILDER.ARCHIVE_NAME)


def test_verifier_fails_closed_on_live_source_drift(
    built_pair, tmp_path: Path
) -> None:
    first, _second, _receipt_a, _receipt_b = built_pair
    copied_root = tmp_path / "source"
    copied_root.mkdir()
    with zipfile.ZipFile(first / BUILDER.ARCHIVE_NAME, "r") as archive:
        for info in archive.infolist():
            if info.filename == BUILDER.MANIFEST_NAME:
                continue
            destination = copied_root.joinpath(*Path(info.filename).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(info))
    init_path = copied_root / "mo_nco" / "__init__.py"
    init_path.write_bytes(init_path.read_bytes() + b"# drift\n")
    with pytest.raises(BUILDER.SourceBundleError, match="live source bytes/hash"):
        BUILDER.verify_bundle(
            first / BUILDER.MANIFEST_NAME,
            first / BUILDER.ARCHIVE_NAME,
            root=copied_root,
        )


def test_builder_requires_an_existing_empty_output_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        BUILDER.build_bundle(ROOT, missing)
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "sentinel.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(BUILDER.SourceBundleError, match="must be empty"):
        BUILDER.build_bundle(ROOT, nonempty)
    assert (nonempty / "sentinel.txt").read_text(encoding="utf-8") == "preserve"

