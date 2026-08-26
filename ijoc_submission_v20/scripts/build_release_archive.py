from __future__ import annotations

"""Build a deterministic, manifest-bearing IJOC source release archive."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = SUBMISSION_ROOT / "release"
ARCHIVE_PATH = RELEASE_ROOT / "mo_nco_pareto_smc_v20_ijoc_source.tar.gz"
MANIFEST_PATH = RELEASE_ROOT / "source_file_manifest.json"
ARCHIVE_PREFIX = "mo_nco_pareto_smc_v20_ijoc"
TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class ReleaseBuildPaths:
    __slots__ = ("archive_path", "manifest_path", "archive_prefix")

    def __init__(
        self,
        *,
        archive_path: Path,
        manifest_path: Path,
        archive_prefix: str,
    ) -> None:
        self.archive_path = archive_path
        self.manifest_path = manifest_path
        self.archive_prefix = archive_prefix


def resolve_build_paths(
    *,
    tag: str | None,
    archive_path: Path | None,
    manifest_path: Path | None,
    archive_prefix: str | None,
) -> ReleaseBuildPaths:
    """Resolve canonical or explicitly versioned release outputs."""

    if tag is not None and TAG_PATTERN.fullmatch(tag) is None:
        raise ValueError(
            "tag must start with an alphanumeric character and contain only "
            "letters, digits, dot, underscore, or hyphen"
        )
    default_archive = (
        ARCHIVE_PATH
        if tag is None
        else RELEASE_ROOT / f"mo_nco_pareto_smc_{tag}_source.tar.gz"
    )
    default_manifest = (
        MANIFEST_PATH
        if tag is None
        else RELEASE_ROOT / f"source_file_manifest_{tag}.json"
    )
    default_prefix = (
        ARCHIVE_PREFIX if tag is None else f"mo_nco_pareto_smc_{tag}"
    )
    return ReleaseBuildPaths(
        archive_path=(archive_path or default_archive).resolve(),
        manifest_path=(manifest_path or default_manifest).resolve(),
        archive_prefix=archive_prefix or default_prefix,
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new_file(path: Path, raw: bytes) -> None:
    """Write a new evidence artifact without ever replacing existing bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)


def canonical_bytes(payload: object) -> bytes:
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


def selected_files() -> list[Path]:
    candidates: set[Path] = set()
    for pattern in (
        "mo_nco/**/*.py",
        "tests/test_ijoc*.py",
        "tests/test_pareto_ijoc*.py",
        "tests/test_pareto_v20*.py",
        "tests_v17/**/*.py",
        "tests_v18/**/*.py",
        "tests_v19/**/*.py",
    ):
        candidates.update(
            path for path in REPO_ROOT.glob(pattern) if path.is_file()
        )
    for relative in (
        "pyproject.toml",
        "README.md",
        "requirements-optional.txt",
        "scripts/freeze_ijoc_manifests.py",
        "scripts/run_ijoc_cold_matrix.py",
        "scripts/audit_ijoc_postrun.py",
        "scripts/pareto_smc_v20_adversarial_checks.py",
        (
            "ijoc_submission_v20/provenance/"
            "PLS_RESTART_V2_LIVENESS_HARDENING_20260731.md"
        ),
    ):
        path = REPO_ROOT / relative
        if path.is_file():
            candidates.add(path)
    for pattern in (
        "scripts/*.py",
        "protocol/*.json",
        "protocol/*.md",
        "protocol/schemas/*.json",
        "calibration/frozen/*.json",
        "formal_study/case_manifest.json",
        "formal_study/instance_packet_manifest.json",
        "formal_study/instances/**/*.json",
        "formal_study/instances/**/*.tsp",
        "formal_study/instances/*.packet.json",
        "formal_study/metric_references/**/*.json",
        "release/requirements-formal-lock.txt",
        "release/LICENSE_STATUS.txt",
    ):
        candidates.update(
            path for path in SUBMISSION_ROOT.glob(pattern) if path.is_file()
        )
    candidates.discard(ARCHIVE_PATH)
    candidates.discard(MANIFEST_PATH)
    return sorted(
        candidates,
        key=lambda path: path.relative_to(REPO_ROOT).as_posix(),
    )


def tar_info(archive_name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(archive_name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def add_bytes(archive: tarfile.TarFile, name: str, raw: bytes) -> None:
    archive.addfile(tar_info(name, len(raw)), io.BytesIO(raw))


def submission_relative(path: Path) -> str:
    resolved = path.resolve()
    resolved.relative_to(SUBMISSION_ROOT.resolve())
    return resolved.relative_to(SUBMISSION_ROOT.resolve()).as_posix()


def build_release(
    paths: ReleaseBuildPaths,
    *,
    tag: str | None,
) -> dict[str, object]:
    """Build one immutable source archive and its canonical file manifest."""

    archive_path = paths.archive_path.resolve()
    manifest_path = paths.manifest_path.resolve()
    submission_relative(archive_path)
    submission_relative(manifest_path)
    if archive_path == manifest_path:
        raise ValueError("archive and manifest outputs must be different paths")
    collisions = [
        path for path in (archive_path, manifest_path) if path.exists()
    ]
    if collisions:
        raise FileExistsError(
            "Refusing to replace existing release evidence: "
            + ", ".join(str(path) for path in collisions)
        )
    files = selected_files()
    if not files:
        raise RuntimeError("The source release selection is empty.")
    entries = []
    for path in files:
        relative = path.relative_to(REPO_ROOT).as_posix()
        entries.append(
            {
                "path": relative,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema": "ijoc_source_file_manifest_v1",
        "archive_prefix": paths.archive_prefix,
        "file_count": len(entries),
        "files": entries,
    }
    manifest_raw = canonical_bytes(manifest)
    archive_buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=archive_buffer,
        mtime=0,
    ) as gzip_handle:
        with tarfile.open(
            mode="w",
            fileobj=gzip_handle,
            format=tarfile.PAX_FORMAT,
        ) as archive:
            for path in files:
                relative = path.relative_to(REPO_ROOT).as_posix()
                add_bytes(
                    archive,
                    f"{paths.archive_prefix}/{relative}",
                    path.read_bytes(),
                )
            add_bytes(
                archive,
                f"{paths.archive_prefix}/source_file_manifest.json",
                manifest_raw,
            )
    archive_raw = archive_buffer.getvalue()
    write_new_file(manifest_path, manifest_raw)
    try:
        write_new_file(archive_path, archive_raw)
    except BaseException:
        if (
            manifest_path.is_file()
            and manifest_path.read_bytes() == manifest_raw
        ):
            manifest_path.unlink()
        raise
    return {
        "schema": "ijoc_source_archive_build_result_v1",
        "version_tag": tag,
        "archive_prefix": paths.archive_prefix,
        "archive": {
            "path": submission_relative(archive_path),
            "sha256": file_sha256(archive_path),
            "bytes": archive_path.stat().st_size,
        },
        "manifest": {
            "path": submission_relative(manifest_path),
            "sha256": file_sha256(manifest_path),
            "file_count": len(entries),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic source archive without replacing any "
            "existing evidence artifact."
        )
    )
    parser.add_argument(
        "--tag",
        "--version-tag",
        dest="tag",
        help="Version tag used to derive new output names.",
    )
    parser.add_argument("--archive-path", type=Path)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--archive-prefix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_build_paths(
        tag=args.tag,
        archive_path=args.archive_path,
        manifest_path=args.manifest_path,
        archive_prefix=args.archive_prefix,
    )
    result = build_release(paths, tag=args.tag)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
