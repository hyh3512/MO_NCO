from __future__ import annotations

"""Build a deterministic, standalone V21e3 experiment-code package."""

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
from typing import Iterable
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_ROOT = REPO_ROOT / "ijoc_submission_v21e3"
RELEASE_ROOT = SUBMISSION_ROOT / "release"
DEFAULT_ARCHIVE_PATH = RELEASE_ROOT / "ijoc_v21e3_experiment_code.zip"
DEFAULT_MANIFEST_PATH = RELEASE_ROOT / "ijoc_v21e3_experiment_code.manifest.json"
DEFAULT_CHECKSUM_PATH = RELEASE_ROOT / "ijoc_v21e3_experiment_code.zip.sha256"
DEFAULT_ARCHIVE_PREFIX = "ijoc_v21e3_experiment_code"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
PROJECT_VERSION = "0.21.3"
V21E2_RELEASE_ZIP_SHA256 = (
    "ecc13e7b174dd53ceb9e644ee5a97e2dd0883c4b27433be86e2d5ee056a0a102"
)
_PREFIX = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_TEMPLATE_MAP = {
    "ijoc_submission_v21e3/release/pyproject.toml": "pyproject.toml",
    "ijoc_submission_v21e3/release/README.md": "README.md",
    "ijoc_submission_v21e3/release/requirements-test.lock": (
        "requirements-test.lock"
    ),
    "ijoc_submission_v21e3/release/wheelhouse_manifest.json": (
        "wheelhouse_manifest.json"
    ),
    "ijoc_submission_v21e3/release/mo_nco_init.py": "mo_nco/__init__.py",
}
_MANDATORY_TEMPLATES = (
    "ijoc_submission_v21e3/release/pyproject.toml",
    "ijoc_submission_v21e3/release/README.md",
    "ijoc_submission_v21e3/release/requirements-test.lock",
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _safe_file(path: Path, root: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.is_symlink():
        raise RuntimeError(f"Release dependency closure forbids symlinks: {path}")
    resolved = path.resolve()
    resolved.relative_to(root)
    return resolved


def _seed_files(root: Path) -> set[Path]:
    seeds: set[Path] = set()
    for relative in _MANDATORY_TEMPLATES:
        seeds.add(_safe_file(root / relative, root))
    patterns = (
        "mo_nco/pareto_v21_*.py",
        "mo_nco/pareto_v21e3_*.py",
        "mo_nco/baselines.py",
        "mo_nco/ijoc_mokp_baselines.py",
        "tests/test_pareto_v21_*.py",
        "tests/test_pareto_v21e3_*.py",
        "ijoc_submission_v21/scripts/**/*.py",
        "ijoc_submission_v21/release/README.md",
        "ijoc_submission_v21e3/scripts/**/*.py",
        "ijoc_submission_v21e3/protocol/**/*.md",
        "ijoc_submission_v21e3/protocol/**/*.json",
        "ijoc_submission_v21e3/provenance/**/*.md",
        "ijoc_submission_v21e3/provenance/**/*.json",
        "ijoc_submission_v21e3/*.md",
        "ijoc_submission_v21e3/development_manifests_v1/**/*.json",
        "ijoc_submission_v21e3/development_partitions_v1/**/*.json",
        "ijoc_submission_v21e3/release/wheelhouse_manifest.json",
        "ijoc_submission_v21e3/release/mo_nco_init.py",
        "ijoc_submission_v21e3/release/wheelhouse/*.whl",
    )
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                seeds.add(_safe_file(path, root))
    if not any(
        path.name.startswith("pareto_v21e3_")
        for path in seeds
        if path.parent.name == "mo_nco"
    ):
        raise RuntimeError("No V21e3 implementation module entered the release.")
    return seeds


def _module_file(root: Path, module: str) -> Path | None:
    if module == "mo_nco":
        release_init = root / "ijoc_submission_v21e3" / "release" / "mo_nco_init.py"
        if release_init.is_file():
            return _safe_file(release_init, root)
    parts = module.split(".")
    module_path = root.joinpath(*parts).with_suffix(".py")
    if module_path.is_file():
        return _safe_file(module_path, root)
    package_path = root.joinpath(*parts, "__init__.py")
    if package_path.is_file():
        return _safe_file(package_path, root)
    return None


def _source_module(path: Path, root: Path) -> tuple[str, ...] | None:
    relative = path.relative_to(root)
    if not relative.parts or relative.parts[0] != "mo_nco":
        return None
    if relative.name == "__init__.py":
        return relative.parent.parts
    return relative.with_suffix("").parts


def _internal_imports(path: Path, root: Path) -> list[tuple[str, bool]]:
    """Return ``(module, required)`` imports rooted at ``mo_nco``."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    current = _source_module(path, root)
    imports: list[tuple[str, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "mo_nco" or alias.name.startswith("mo_nco."):
                    imports.append((alias.name, True))
        elif isinstance(node, ast.ImportFrom):
            module: str | None = None
            if node.level and current is not None:
                package = current if path.name == "__init__.py" else current[:-1]
                up = node.level - 1
                if up > len(package):
                    imports.append(("<relative-import-escape>", True))
                    continue
                base = package[: len(package) - up]
                module = ".".join((*base, *(node.module or "").split(".")))
                module = module.rstrip(".")
            elif node.module == "mo_nco" or (
                node.module and node.module.startswith("mo_nco.")
            ):
                module = node.module
            if not module or not (
                module == "mo_nco" or module.startswith("mo_nco.")
            ):
                continue
            # The imported module is required.  For ``from mo_nco import x``,
            # ``x`` may instead be an attribute; admit a matching submodule
            # when it exists but do not invent an unresolved module.
            imports.append((module, module != "mo_nco"))
            for alias in node.names:
                if alias.name == "*":
                    continue
                candidate = f"{module}.{alias.name}"
                if _module_file(root, candidate) is not None:
                    imports.append((candidate, True))
    return imports


def dependency_closed_files(
    repo_root: Path,
) -> tuple[list[Path], list[str], list[str]]:
    """Compute the transitive internal-Python dependency closure."""

    root = repo_root.resolve()
    selected = _seed_files(root)
    package_init = _module_file(root, "mo_nco")
    if package_init is not None:
        selected.add(package_init)
    queue = sorted(path for path in selected if path.suffix == ".py")
    scanned: set[Path] = set()
    unresolved: set[str] = set()
    while queue:
        source = queue.pop(0)
        if source in scanned:
            continue
        scanned.add(source)
        for module, required in _internal_imports(source, root):
            dependency = _module_file(root, module)
            if dependency is None:
                if required:
                    unresolved.add(
                        f"{source.relative_to(root).as_posix()}::{module}"
                    )
                continue
            if dependency not in selected:
                selected.add(dependency)
                queue.append(dependency)
    return (
        sorted(selected, key=lambda path: path.relative_to(root).as_posix()),
        sorted(unresolved),
        sorted(path.relative_to(root).as_posix() for path in scanned),
    )


def _archive_entries(
    repo_root: Path,
) -> tuple[list[dict[str, object]], list[str], list[str]]:
    root = repo_root.resolve()
    files, unresolved, scanned = dependency_closed_files(root)
    if unresolved:
        raise RuntimeError(
            "Internal dependency closure is incomplete: " + "; ".join(unresolved)
        )
    entries: list[dict[str, object]] = []
    archive_paths: set[str] = set()
    for path in files:
        source_relative = path.relative_to(root).as_posix()
        destinations = [source_relative]
        mapped = _TEMPLATE_MAP.get(source_relative)
        if mapped is not None:
            destinations.append(mapped)
        wheelhouse_prefix = "ijoc_submission_v21e3/release/wheelhouse/"
        if source_relative.startswith(wheelhouse_prefix):
            destinations.append(
                "wheelhouse/" + source_relative.removeprefix(wheelhouse_prefix)
            )
        raw = path.read_bytes()
        for destination in destinations:
            pure = PurePosixPath(destination)
            if pure.is_absolute() or ".." in pure.parts or destination in archive_paths:
                raise RuntimeError(f"Unsafe or duplicate archive path: {destination}")
            archive_paths.add(destination)
            entries.append(
                {
                    "source_path": source_relative,
                    "archive_path": destination,
                    "sha256": _sha256(raw),
                    "bytes": len(raw),
                    "raw": raw,
                }
            )
    entries.sort(key=lambda entry: str(entry["archive_path"]))
    return entries, unresolved, scanned


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)


def build_release(
    repo_root: Path,
    *,
    archive_path: Path,
    manifest_path: Path,
    checksum_path: Path,
    archive_prefix: str = DEFAULT_ARCHIVE_PREFIX,
) -> dict[str, object]:
    """Build a deterministic archive plus external manifest and checksum."""

    root = repo_root.resolve()
    outputs = tuple(
        Path(path).resolve() for path in (archive_path, manifest_path, checksum_path)
    )
    if len(set(outputs)) != 3:
        raise ValueError("Release output paths must be distinct.")
    if _PREFIX.fullmatch(archive_prefix) is None:
        raise ValueError("archive_prefix must be a safe single component.")
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to replace V21e3 release artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    entries, unresolved, scanned = _archive_entries(root)
    pyproject_entry = next(
        entry for entry in entries if entry["archive_path"] == "pyproject.toml"
    )
    pyproject_text = bytes(pyproject_entry["raw"]).decode("utf-8")
    if not re.search(r'(?m)^version\s*=\s*["\']0\.21\.3["\']\s*$', pyproject_text):
        raise RuntimeError("The standalone package metadata is not version 0.21.3.")
    wheel_entries = {
        str(entry["archive_path"]).removeprefix("wheelhouse/"): entry
        for entry in entries
        if str(entry["archive_path"]).startswith("wheelhouse/")
    }
    wheel_manifest_entries = [
        entry
        for entry in entries
        if entry["archive_path"] == "wheelhouse_manifest.json"
    ]
    if wheel_entries or wheel_manifest_entries:
        if not wheel_entries or len(wheel_manifest_entries) != 1:
            raise RuntimeError("Offline wheelhouse and its manifest must coexist.")
        wheel_manifest = json.loads(bytes(wheel_manifest_entries[0]["raw"]))
        expected_wheels = {
            str(item["filename"]): item for item in wheel_manifest.get("files", ())
        }
        if set(expected_wheels) != set(wheel_entries):
            raise RuntimeError("Offline wheelhouse file set differs from its manifest.")
        for name, item in expected_wheels.items():
            entry = wheel_entries[name]
            if not (
                int(item["bytes"]) == int(entry["bytes"])
                and item["sha256"] == entry["sha256"]
            ):
                raise RuntimeError(f"Offline wheelhouse binding failed: {name}")
        offline_wheelhouse_gate = "PASS"
    else:
        offline_wheelhouse_gate = "NOT_INCLUDED_TEST_FIXTURE"

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(
        archive_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for entry in entries:
            name = f"{archive_prefix}/{entry['archive_path']}"
            archive.writestr(_zip_info(name), entry["raw"])
    archive_raw = archive_buffer.getvalue()
    archive_sha = _sha256(archive_raw)
    public_entries = [
        {key: entry[key] for key in ("source_path", "archive_path", "sha256", "bytes")}
        for entry in entries
    ]
    manifest = {
        "schema": "ijoc_v21e3_standalone_release_manifest_v1",
        "artifact_scope": "V21E3_STANDALONE_EXPERIMENT_CODE_PROSPECTIVE_ONLY",
        "project_version": PROJECT_VERSION,
        "rights_status": "HOLD_UNLESS_SEPARATELY_CLOSED",
        "formal_status": "NOT_MATERIALIZED",
        "formal_authorized": False,
        "v21e2_immutable_baseline": {
            "status": "IMMUTABLE_CALIBRATION_EVIDENCE_NOT_MODIFIED",
            "release_zip_sha256": V21E2_RELEASE_ZIP_SHA256,
        },
        "dependency_closure": {
            "method": "recursive_python_ast_internal_import_closure_v1",
            "scanned_python_files": scanned,
            "unresolved_internal_imports": unresolved,
            "gate": "PASS",
        },
        "clean_room_gate": "REQUIRED_SEPARATE_RECEIPT",
        "offline_wheelhouse_gate": offline_wheelhouse_gate,
        "deterministic_zip_contract": {
            "entry_order": "UTF8_POSIX_PATH_ASCENDING",
            "entry_timestamp": "1980-01-01T00:00:00",
            "entry_unix_mode_octal": "100644",
            "compression": "DEFLATE_LEVEL_9",
        },
        "archive_prefix": archive_prefix,
        "archive": {
            "filename": outputs[0].name,
            "sha256": archive_sha,
            "bytes": len(archive_raw),
        },
        "file_count": len(public_entries),
        "files": public_entries,
    }
    manifest_raw = _canonical(manifest)
    checksum_raw = f"{archive_sha}  {outputs[0].name}\n".encode("ascii")
    artifacts = (
        (outputs[0], archive_raw),
        (outputs[1], manifest_raw),
        (outputs[2], checksum_raw),
    )
    created: list[tuple[Path, bytes]] = []
    try:
        for path, raw in artifacts:
            _write_exclusive(path, raw)
            created.append((path, raw))
    except BaseException:
        for path, expected in reversed(created):
            if path.is_file() and path.read_bytes() == expected:
                path.unlink()
        raise
    return {
        "schema": "ijoc_v21e3_standalone_release_build_result_v1",
        "archive": {
            "path": str(outputs[0]),
            "sha256": archive_sha,
            "bytes": len(archive_raw),
        },
        "manifest": {
            "path": str(outputs[1]),
            "sha256": _sha256(manifest_raw),
            "file_count": len(public_entries),
        },
        "checksum": {
            "path": str(outputs[2]),
            "sha256": _sha256(checksum_raw),
        },
        "formal_authorized": False,
        "formal_status": "NOT_MATERIALIZED",
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the deterministic standalone V21e3 code package."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--archive-path", type=Path, default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--checksum-path", type=Path, default=DEFAULT_CHECKSUM_PATH)
    parser.add_argument("--archive-prefix", default=DEFAULT_ARCHIVE_PREFIX)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_release(
        args.repo_root,
        archive_path=args.archive_path,
        manifest_path=args.manifest_path,
        checksum_path=args.checksum_path,
        archive_prefix=args.archive_prefix,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
