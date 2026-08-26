from __future__ import annotations

"""Fail-closed identity check for the immutable V4/V6 108-row releases."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence
from zipfile import ZIP_STORED, ZipFile


FROZEN_FILES: Mapping[str, tuple[int, str]] = {
    "ijoc_v21e3r1_development_results_v4.zip": (
        3307093113,
        "10483e5b65e0bf779681e699a77955ab64ac1dbf5346777f605d92f56dc98bb8",
    ),
    "ijoc_v21e3r1_development_results_v4.manifest.json": (
        186562,
        "31c62279846328703759aa01a26a12a239cbb7efcb3c626f02e80f87bdf93a08",
    ),
    "ijoc_v21e3r1_development_results_v4.index.json": (
        2351,
        "015cf22e5861e879c00fd1b63f9a132ee91e225c4f1f9cbb03252472846de889",
    ),
    "ijoc_v21e3r1_development_results_v6.zip": (
        3307094518,
        "9ba80b736fd4cf87246626384ed3644d0428a3be74f100d8b281d5648875ee42",
    ),
    "ijoc_v21e3r1_development_results_v6.manifest.json": (
        186959,
        "e48499aebc6819324b4e9fe135c7fc6212c8716c0169e188a6c4b933c676ebac",
    ),
    "ijoc_v21e3r1_development_results_v6.index.json": (
        3464,
        "8ea7cd913db05840ff0fb2beddb3f62a3089383e07dc9882555888a86cfdcf7c",
    ),
}
V4_REPLACED_MEMBER = (
    "ijoc_v21e3r1_results_release/evidence/"
    "independent_development_matrix_post_run_audit.json"
)
V6_REPLACEMENT_MEMBER = (
    "ijoc_v21e3r1_results_release/evidence/"
    "same_implementation_development_matrix_post_run_audit_v6.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _manifest_map(manifest: Mapping[str, object]) -> dict[str, tuple[int, str, str]]:
    raw = manifest.get("files")
    if not isinstance(raw, list) or len(raw) != 701:
        raise RuntimeError("Historical release manifest must contain exactly 701 files.")
    result: dict[str, tuple[int, str, str]] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise RuntimeError("Historical manifest member is not an object.")
        path = item.get("archive_path")
        size = item.get("bytes")
        digest = item.get("sha256")
        role = item.get("role")
        if (
            type(path) is not str
            or type(size) is not int
            or size < 0
            or type(digest) is not str
            or len(digest) != 64
            or type(role) is not str
            or not role
            or path in result
        ):
            raise RuntimeError("Historical manifest contains an invalid member binding.")
        result[path] = (size, digest, role)
    return result


def _verify_zip_structure(path: Path, manifest: Mapping[str, object]) -> None:
    members = _manifest_map(manifest)
    with ZipFile(path, "r") as archive:
        infos = archive.infolist()
        if len(infos) != 701 or len({item.filename for item in infos}) != 701:
            raise RuntimeError("Historical ZIP member cardinality is not exactly 701.")
        if tuple(item.filename for item in infos) != tuple(members):
            raise RuntimeError("Historical ZIP member order/path set disagrees with manifest.")
        for info in infos:
            expected_size = members[info.filename][0]
            if (
                info.is_dir()
                or info.file_size != expected_size
                or info.compress_type != ZIP_STORED
                or info.date_time != (1980, 1, 1, 0, 0, 0)
                or (info.external_attr >> 16) != 0o100644
            ):
                raise RuntimeError(f"Historical ZIP structure drifted: {info.filename}")


def verify_historical_releases(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve()
    release = (root / "ijoc_submission_v21e3r1/release").resolve()
    release.relative_to(root)
    identities: dict[str, dict[str, object]] = {}
    for name, (expected_size, expected_hash) in FROZEN_FILES.items():
        path = release / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_size = path.stat().st_size
        observed_hash = _sha256(path)
        if observed_size != expected_size or observed_hash != expected_hash:
            raise RuntimeError(f"Immutable historical artifact drifted: {name}")
        identities[name] = {"bytes": observed_size, "sha256": observed_hash}

    v4_manifest = _json_object(
        release / "ijoc_v21e3r1_development_results_v4.manifest.json"
    )
    v6_manifest = _json_object(
        release / "ijoc_v21e3r1_development_results_v6.manifest.json"
    )
    v4_index = _json_object(
        release / "ijoc_v21e3r1_development_results_v4.index.json"
    )
    v6_index = _json_object(
        release / "ijoc_v21e3r1_development_results_v6.index.json"
    )
    for version, manifest, index in (
        ("V4", v4_manifest, v4_index),
        ("V6", v6_manifest, v6_index),
    ):
        if (
            manifest.get("file_count") != 701
            or manifest.get("formal_authorized") is not False
            or index.get("formal_authorized") is not False
            or index.get("submission_status") != "IJOC_HOLD"
            or index.get("full_algorithm_decision_replay") != "NOT_IMPLEMENTED"
        ):
            raise RuntimeError(f"{version} historical claim boundary drifted.")
        data_archive = index.get("data_archive")
        if not isinstance(data_archive, dict):
            raise RuntimeError(f"{version} index omits data_archive.")
        zip_name = f"ijoc_v21e3r1_development_results_{version.lower()}.zip"
        if (
            data_archive.get("bytes") != FROZEN_FILES[zip_name][0]
            or data_archive.get("sha256") != FROZEN_FILES[zip_name][1]
            or data_archive.get("file_count") != 701
        ):
            raise RuntimeError(f"{version} index-to-ZIP binding drifted.")

    v6_post = v6_index.get("same_implementation_post_process")
    if not isinstance(v6_post, dict) or any(
        v6_post.get(field) is not False
        for field in (
            "implementation_independence",
            "scientific_independence",
            "external_third_party_audit",
        )
    ) or v6_post.get("receipt_sha256") != (
        "09abd1d4aaf5453a9f9913183ba7523f36c1d245e543a53b50c99eb01aab0ef8"
    ):
        raise RuntimeError("V6 same-implementation claim boundary drifted.")

    v4_files = _manifest_map(v4_manifest)
    v6_files = _manifest_map(v6_manifest)
    if (
        set(v4_files) - set(v6_files) != {V4_REPLACED_MEMBER}
        or set(v6_files) - set(v4_files) != {V6_REPLACEMENT_MEMBER}
        or any(v4_files[name] != v6_files[name] for name in set(v4_files) & set(v6_files))
        or len(set(v4_files) & set(v6_files)) != 700
    ):
        raise RuntimeError("V4-to-V6 700-unchanged plus one-replacement relation failed.")

    _verify_zip_structure(
        release / "ijoc_v21e3r1_development_results_v4.zip", v4_manifest
    )
    _verify_zip_structure(
        release / "ijoc_v21e3r1_development_results_v6.zip", v6_manifest
    )
    return {
        "schema": "v21e3r1_v4_v6_historical_preservation_receipt_v1",
        "status": "PASS_HISTORICAL_V4_V6_IDENTITY_AND_RELATIONSHIP",
        "historical_row_count_each": 108,
        "archive_member_count_each": 701,
        "unchanged_member_count": 700,
        "v4_removed_member": V4_REPLACED_MEMBER,
        "v6_replacement_member": V6_REPLACEMENT_MEMBER,
        "identities": identities,
        "historical_outputs_modified": False,
        "implementation_independence": False,
        "scientific_independence": False,
        "selection_authorized": False,
        "formal_authorized": False,
        "submission_status": "IJOC_HOLD",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = verify_historical_releases(args.project_root)
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            handle.write("\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
