from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "build_v21e3r1_code_release.py"
)


def _builder():
    spec = importlib.util.spec_from_file_location("v21e3r1_release_builder", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def test_v21e3r1_release_is_distinct_deterministic_and_binds_its_parents(
    tmp_path: Path,
) -> None:
    builder = _builder()
    repo = tmp_path / "repo"
    _write(repo / "mo_nco" / "__init__.py", "")
    _write(
        repo / "mo_nco" / "pareto_v21e3_entry.py",
        "from .archive import VALUE\nENTRY = VALUE\n",
    )
    _write(repo / "mo_nco" / "archive.py", "VALUE = 21\n")
    _write(
        repo / "tests" / "test_pareto_v21e3_entry.py",
        "from mo_nco.pareto_v21e3_entry import ENTRY\n"
        "def test_entry(): assert ENTRY == 21\n",
    )
    release = repo / "ijoc_submission_v21e3r1" / "release"
    _write(
        release / "pyproject.toml",
        "[build-system]\nrequires=['setuptools==80.10.2']\n"
        "build-backend='setuptools.build_meta'\n[project]\n"
        "name='fixture'\nversion='0.21.3.1'\n",
    )
    _write(release / "README.md", "V21e3r1 fixture\n")
    _write(release / "requirements-test.lock", "pytest==9.1.1\n")
    _write(release / "mo_nco_init.py", "__version__='0.21.3.1'\n")
    _write(
        repo
        / "ijoc_submission_v21e3r1"
        / "scripts"
        / "build_v21e3r1_code_release.py",
        "# fixture builder\n",
    )
    frozen_provenance = {
        "status": "PASS_LIVE_SNAPSHOT_REVALIDATED_BEFORE_BUILD",
        "selection_entropy_release": "PROHIBITED",
        "formal_authorized": False,
    }

    first = builder.build_release(
        repo,
        archive_path=tmp_path / "first.zip",
        manifest_path=tmp_path / "first.manifest.json",
        checksum_path=tmp_path / "first.sha256",
        frozen_source_provenance=frozen_provenance,
    )
    second = builder.build_release(
        repo,
        archive_path=tmp_path / "second.zip",
        manifest_path=tmp_path / "second.manifest.json",
        checksum_path=tmp_path / "second.sha256",
        frozen_source_provenance=frozen_provenance,
    )

    assert first["archive"]["sha256"] == second["archive"]["sha256"]
    assert (tmp_path / "first.zip").read_bytes() == (tmp_path / "second.zip").read_bytes()
    manifest = json.loads((tmp_path / "first.manifest.json").read_text("utf-8"))
    assert manifest["schema"] == "ijoc_v21e3r1_standalone_release_manifest_v1"
    assert manifest["project_version"] == "0.21.3.1"
    assert manifest["v21e3_immutable_parent"]["release_zip_sha256"] == (
        "7881b30e6f6059e36e0ed8279f8932ab5f48f2f8e0bc38885e59a74fb45fb3b0"
    )
    assert manifest["v21e2_immutable_baseline"]["release_zip_sha256"] == (
        "ecc13e7b174dd53ceb9e644ee5a97e2dd0883c4b27433be86e2d5ee056a0a102"
    )
    with zipfile.ZipFile(tmp_path / "first.zip") as archive:
        names = [entry.filename for entry in archive.infolist()]
        assert names == sorted(names)
        assert "ijoc_v21e3r1_experiment_code/pyproject.toml" in names
        assert all(entry.date_time == (1980, 1, 1, 0, 0, 0) for entry in archive.infolist())
    assert first["archive"]["sha256"] == hashlib.sha256(
        (tmp_path / "first.zip").read_bytes()
    ).hexdigest()

    rebuilt = builder.rebuild_from_verified_release_manifest(
        repo,
        provenance_manifest_path=tmp_path / "first.manifest.json",
        archive_path=tmp_path / "rebuilt.zip",
        manifest_path=tmp_path / "rebuilt.manifest.json",
        checksum_path=tmp_path / "rebuilt.sha256",
    )
    assert rebuilt["archive"]["sha256"] == first["archive"]["sha256"]
    assert (tmp_path / "rebuilt.zip").read_bytes() == (tmp_path / "first.zip").read_bytes()

