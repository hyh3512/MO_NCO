"""Build the deterministic V21E3R1 V9R2 engineering source bundle.

The bundle is an engineering handoff artifact.  It does not authorize the full
development matrix or any later scientific phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


_FIXED_ZIP_TIME = (2023, 11, 14, 22, 13, 20)
_TARGETED_TESTS = (
    "tests/test_pareto_v21e3r1_v9_information_search.py",
    "tests/test_pareto_v21e3r1_v9_strict_regressions.py",
    "tests/test_pareto_v21e3r1_v9r1_theory_strict.py",
    "tests/test_pareto_v21e3r1_v9_diagnostics_strict.py",
    "tests/test_pareto_v21e3r1_branch_replay.py",
    "tests/test_pareto_v21e3r1_v9_runner.py",
    "tests/test_pareto_v21e3_trace.py",
    "tests/test_pareto_v21e3_trace_verify.py",
    "tests/test_pareto_v21e3_trace_chunks.py",
    "tests/test_pareto_v21e3r1_v9_protocol.py",
    "tests/test_pareto_v21e3r1_v9_gate.py",
)
_SUPPORT_FILES = (
    "pyproject.toml",
    "docs/V21E3R1_V9R1_THEORY.md",
    "docs/V21E3R1_V9R2_ALGORITHM_SPEC.md",
    "docs/V21E3R1_V9R2_ENGINEERING_CLOSURE.md",
    "docs/V21E3R1_V9R2_RUNBOOK.md",
    "docs/V21E3R1_V9R2_TRACE_REPLAY_SPEC.md",
    "scripts/build_v9r2_engineering_source_bundle.py",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_paths(root: Path) -> list[Path]:
    relative = {
        path.relative_to(root)
        for path in (root / "mo_nco").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    relative.update(
        path.relative_to(root)
        for path in (root / "mo_nco" / "specs").glob("*.json")
    )
    relative.update(Path(item) for item in _TARGETED_TESTS)
    relative.update(Path(item) for item in _SUPPORT_FILES)
    paths = sorted(relative, key=lambda path: path.as_posix())
    for path in paths:
        absolute = root / path
        if not absolute.is_file() or absolute.is_symlink():
            raise RuntimeError(f"required regular source file missing: {path.as_posix()}")
    return paths


def build_bundle(root: Path, output_directory: Path) -> dict[str, object]:
    root = root.resolve()
    output_directory = output_directory.resolve()
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(f"not a project root: {root}")
    if not output_directory.is_dir():
        raise RuntimeError(f"output directory must already exist: {output_directory}")

    manifest_path = output_directory / "V21E3R1_V9R2_SOURCE_MANIFEST.json"
    archive_path = output_directory / "mo_nco-0.21.3.13-v9r2-source.zip"
    for path in (manifest_path, archive_path):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite: {path}")

    paths = _source_paths(root)
    entries: list[dict[str, object]] = []
    content: dict[str, bytes] = {}
    for relative in paths:
        name = relative.as_posix()
        data = (root / relative).read_bytes()
        content[name] = data
        entries.append({"path": name, "bytes": len(data), "sha256": _sha256(data)})

    payload: dict[str, object] = {
        "schema": "v21e3r1_v9r2_engineering_source_manifest_v1",
        "identity": {"distribution": "mo-nco", "version": "0.21.3.13"},
        "scope": "all_mo_nco_python_sources_package_specs_targeted_tests_and_v9r2_docs",
        "status": "ENGINEERING_SOURCE_CLOSURE_ONLY_PRE_DEVELOPMENT_HOLD",
        "file_count": len(entries),
        "files": entries,
        "source_tree_sha256": _sha256(_canonical_json(entries)),
        "scientific_independence": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_authorized": False,
        "ijoc_submission_authorized": False,
    }
    payload["manifest_payload_sha256"] = _sha256(_canonical_json(payload))
    manifest_bytes = _canonical_json(payload) + b"\n"
    manifest_path.write_bytes(manifest_bytes)

    with zipfile.ZipFile(
        archive_path,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(content):
            info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content[name], compresslevel=9)
        info = zipfile.ZipInfo("V21E3R1_V9R2_SOURCE_MANIFEST.json", date_time=_FIXED_ZIP_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, manifest_bytes, compresslevel=9)

    result = {
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256(archive_path.read_bytes()),
        "file_count": len(entries),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_bytes),
        "source_tree_sha256": payload["source_tree_sha256"],
        "status": payload["status"],
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build_bundle(args.root, args.output_directory),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
