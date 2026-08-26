from __future__ import annotations

"""Freeze the prospective V21e3r1 development source/evidence boundary.

The snapshot is an engineering provenance object only.  It is deliberately
insufficient to authorize the development parity matrix: a separate
independent preflight must re-hash every entry and validate the structural,
full-test, and protocol receipts.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import sys
from typing import Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_ROOT = REPO_ROOT / "ijoc_submission_v21e3r1"
DEFAULT_OUTPUT = (
    SUBMISSION_ROOT
    / "provenance"
    / "V21E3R1_DEVELOPMENT_SOURCE_SNAPSHOT_FREEZE_V4.json"
)
DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "ijoc_submission_v21e3"
    / "protocol"
    / "V21E3_C0_PARITY_PROTOCOL_V2.json"
)
DEFAULT_INPUT_STRUCTURAL_RECEIPT = (
    SUBMISSION_ROOT
    / "provenance"
    / "V21E3R1_TARGET_SIZE_INPUT_STRUCTURE_RECEIPT_V1.json"
)
DEFAULT_TARGET_EXECUTION_RECEIPT = (
    SUBMISSION_ROOT
    / "provenance"
    / "target_size_execution_v4"
    / "V21E3R1_TARGET_SIZE_EXECUTION_RECEIPT_V1.json"
)
DEFAULT_PYTEST_RECEIPT = (
    SUBMISSION_ROOT
    / "provenance"
    / "V21E3R1_FULL_PYTEST_RECEIPT_V4.json"
)
OLD_V21E3_ZIP = (
    REPO_ROOT
    / "ijoc_submission_v21e3"
    / "release"
    / "ijoc_v21e3_experiment_code.zip"
)
OLD_V21E3_ZIP_SHA256 = (
    "7881b30e6f6059e36e0ed8279f8932ab5f48f2f8e0bc38885e59a74fb45fb3b0"
)
_IMMUTABLE_PARENT_CHAIN_PATHS = {
    "v21e2_immutable_baseline": (
        "ijoc_submission_v21e3/provenance/V21E2_IMMUTABLE_BASELINE.json"
    ),
    "v21e2_immutable_calibration_evidence": (
        "ijoc_submission_v21e3/provenance/"
        "V21E2_IMMUTABLE_CALIBRATION_EVIDENCE.json"
    ),
    "v21e3_development_snapshot": (
        "ijoc_submission_v21e3/provenance/"
        "V21E3_DEVELOPMENT_SNAPSHOT_FREEZE_V1.json"
    ),
    "v21e3_release_manifest": (
        "ijoc_submission_v21e3/release/"
        "ijoc_v21e3_experiment_code.manifest.json"
    ),
    "v21e3_clean_room_receipt": (
        "ijoc_submission_v21e3/release/ijoc_v21e3_clean_room.receipt.json"
    ),
    "v21e3_zip_checksum_file": (
        "ijoc_submission_v21e3/release/ijoc_v21e3_experiment_code.zip.sha256"
    ),
}


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def bound_files_root(entries: object) -> str:
    """Return the canonical digest of the ordered bound-file entries."""

    return _sha256(_canonical_bytes(entries))


def _inside(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"Bound path escapes repository root: {path}") from error
    return resolved


def _repo_relative_artifact(root: Path, value: object, *, label: str) -> Path:
    text = str(value)
    pure = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or pure.is_absolute()
        or PureWindowsPath(text).is_absolute()
        or pure.as_posix() != text
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"{label} must be a canonical repository-relative POSIX path.")
    return _inside(root, root.joinpath(*pure.parts))


def _entry(root: Path, path: Path) -> dict[str, object]:
    resolved = _inside(root, path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Required bound file is absent: {resolved}")
    raw = resolved.read_bytes()
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _iter_source_files(root: Path) -> Iterable[Path]:
    patterns = (
        # Bind the complete executable package and complete test source set.
        # Prefix-only globs previously omitted transitive runtime dependencies
        # (for example archive.py and pareto_ijoc_problem.py), which allowed the
        # live algorithm to drift without changing the advertised source root.
        "mo_nco/**/*.py",
        "tests/**/*.py",
        "pyproject.toml",
        "ijoc_submission_v21e3r1/README.md",
        "ijoc_submission_v21e3r1/manuscript/*.tex",
        "ijoc_submission_v21e3r1/manuscript/*.md",
        "ijoc_submission_v21e3r1/protocol/*.json",
        "ijoc_submission_v21e3r1/protocol/*.md",
        "ijoc_submission_v21e3r1/scripts/*.py",
        "ijoc_submission_v21e3r1/provenance/audit_inputs/*",
        "ijoc_submission_v21e3r1/release/README.md",
        "ijoc_submission_v21e3r1/release/pyproject.toml",
        "ijoc_submission_v21e3r1/release/requirements-test.lock",
        "ijoc_submission_v21e3r1/release/wheelhouse_manifest.json",
        "ijoc_submission_v21e3r1/release/mo_nco_init.py",
        "ijoc_submission_v21e3r1/release/wheelhouse/*.whl",
        "ijoc_submission_v21/scripts/**/*.py",
        "ijoc_submission_v21/release/README.md",
        "ijoc_submission_v21e3/scripts/audit_v21e3_trace_streaming.py",
        "ijoc_submission_v21e3/scripts/build_v21e3_code_release.py",
        "ijoc_submission_v21e3/scripts/verify_v21e3_clean_room.py",
        "ijoc_submission_v21e3/development_manifests_v1/*.json",
        "ijoc_submission_v21e3/development_partitions_v1/case_manifest.json",
        "ijoc_submission_v21e3/development_partitions_v1/instances/*.json",
        "ijoc_submission_v21e3/protocol/DEVELOPMENT_COMMON_BUDGET_PARITY_ASSESSMENT_V2.md",
        "ijoc_submission_v21e3/provenance/development_partition_audit_v1.json",
        "ijoc_submission_v21e3/provenance/V21E3R1_TRACE_STREAMING_SMALL_SCALE_V6.json",
    )
    selected: set[Path] = set()
    for pattern in patterns:
        selected.update(path.resolve() for path in root.glob(pattern) if path.is_file())
    return sorted(
        selected,
        key=lambda path: path.relative_to(root.resolve()).as_posix(),
    )


def _assert_no_prospective_entropy(root: Path) -> None:
    prohibited_names = (
        "selection_partitions_v1",
        "calibration_partitions_v1",
        "calibration_confirmation_partitions_v1",
        "confirmation_partitions_v1",
        "formal_partitions_v1",
        "formal_cases",
        "formal_runs",
    )
    materialized: list[str] = []
    for submission_name in ("ijoc_submission_v21e3", "ijoc_submission_v21e3r1"):
        submission = root / submission_name
        for name in prohibited_names:
            candidate = submission / name
            if candidate.exists():
                materialized.append(candidate.relative_to(root).as_posix())
    if materialized:
        raise RuntimeError(
            "Prospective selection/calibration/formal entropy already exists: "
            + ",".join(sorted(materialized))
        )


def _immutable_parent(
    root: Path, expected_v21e3_zip_sha256: str
) -> tuple[Path, str]:
    old_zip = (
        root
        / "ijoc_submission_v21e3"
        / "release"
        / "ijoc_v21e3_experiment_code.zip"
    )
    old_digest = _sha256(old_zip.read_bytes())
    if old_digest != expected_v21e3_zip_sha256:
        raise RuntimeError(
            "Immutable V21e3 parent ZIP drifted: "
            f"expected {expected_v21e3_zip_sha256}, observed {old_digest}"
        )
    return old_zip.resolve(), old_digest


def _assert_required_categories(entries: list[dict[str, object]]) -> None:
    paths = {str(entry["path"]) for entry in entries}
    required_categories = {
        "source": any(path.startswith("mo_nco/") for path in paths),
        "tests": any(path.startswith("tests/") for path in paths),
        "audit_inputs": any(
            path.startswith("ijoc_submission_v21e3r1/provenance/audit_inputs/")
            for path in paths
        ),
        "streaming_v6": (
            "ijoc_submission_v21e3/provenance/"
            "V21E3R1_TRACE_STREAMING_SMALL_SCALE_V6.json"
        ) in paths,
    }
    missing_categories = sorted(
        name for name, present in required_categories.items() if not present
    )
    if missing_categories:
        raise RuntimeError(
            "Snapshot omits required binding categories: "
            + ",".join(missing_categories)
        )


def compute_prospective_source_root(
    *,
    repo_root: Path,
    protocol_path: Path,
    expected_v21e3_zip_sha256: str = OLD_V21E3_ZIP_SHA256,
) -> dict[str, object]:
    """Compute the stable pre-evidence root used by target-size execution.

    The input-structure, target-execution, and full-pytest receipts are
    intentionally absent from this root because they do not exist until after
    their respective probes.  The final snapshot binds those three receipts in
    a second, complete root and records this prospective root for equality
    checking by the independent preflight.
    """

    root = repo_root.resolve()
    old_zip, old_digest = _immutable_parent(root, expected_v21e3_zip_sha256)
    files = set(_iter_source_files(root)) | {
        _inside(root, protocol_path),
        old_zip,
    }
    files.update(
        _inside(root, root / relative)
        for relative in _IMMUTABLE_PARENT_CHAIN_PATHS.values()
    )
    entries = [
        _entry(root, path)
        for path in sorted(files, key=lambda item: item.relative_to(root).as_posix())
    ]
    _assert_required_categories(entries)
    entries_by_path = {str(entry["path"]): entry for entry in entries}
    parent_chain = {
        role: dict(entries_by_path[relative])
        for role, relative in _IMMUTABLE_PARENT_CHAIN_PATHS.items()
    }
    return {
        "prospective_source_file_count": len(entries),
        "prospective_source_root_sha256": bound_files_root(entries),
        "prospective_source_files": entries,
        "v21e3_immutable_parent_zip_sha256": old_digest,
        "immutable_parent_chain_bindings": parent_chain,
    }


def freeze_development_snapshot(
    *,
    repo_root: Path,
    output: Path,
    protocol_path: Path,
    input_structural_receipt_path: Path,
    target_execution_receipt_path: Path,
    pytest_receipt_path: Path,
    expected_v21e3_zip_sha256: str = OLD_V21E3_ZIP_SHA256,
) -> dict[str, object]:
    """Write an exclusive engineering snapshot without authorizing execution."""

    root = repo_root.resolve()
    destination = _inside(root, output)
    if destination.exists():
        raise FileExistsError(f"Refusing to replace snapshot: {destination}")
    _assert_no_prospective_entropy(root)

    prospective = compute_prospective_source_root(
        repo_root=root,
        protocol_path=protocol_path,
        expected_v21e3_zip_sha256=expected_v21e3_zip_sha256,
    )
    prospective_entries = prospective["prospective_source_files"]
    assert isinstance(prospective_entries, list)
    source_files = {
        _inside(root, root / str(entry["path"]))
        for entry in prospective_entries
        if isinstance(entry, Mapping)
    }
    pytest_receipt_resolved = _inside(root, pytest_receipt_path)
    pytest_payload = json.loads(pytest_receipt_resolved.read_text(encoding="utf-8"))
    if not isinstance(pytest_payload, Mapping):
        raise ValueError("The full pytest receipt must be a JSON object.")
    if pytest_payload.get("artifact_path_semantics") != (
        "repo_root_relative_posix_v1"
    ):
        raise ValueError("The full pytest receipt lacks portable path semantics.")
    pytest_log_path = _repo_relative_artifact(
        root,
        pytest_payload.get("log_path", ""),
        label="full pytest log_path",
    )
    evidence_files = {
        _inside(root, input_structural_receipt_path),
        _inside(root, target_execution_receipt_path),
        pytest_receipt_resolved,
        pytest_log_path,
    }
    files = source_files | evidence_files
    files.discard(destination)
    entries = [_entry(root, path) for path in sorted(
        files, key=lambda item: item.relative_to(root).as_posix()
    )]
    if not entries:
        raise RuntimeError("The V21e3r1 snapshot cannot be empty.")
    _assert_required_categories(entries)

    def relative(path: Path) -> str:
        return _inside(root, path).relative_to(root).as_posix()

    receipt: dict[str, object] = {
        "schema": "pareto_v21e3r1_development_source_snapshot_freeze_v1",
        "status": "PASS_ENGINEERING_SNAPSHOT_ONLY",
        "scientific_scope": "source_and_engineering_evidence_provenance_only",
        "bound_file_count": len(entries),
        "bound_files_root_sha256": bound_files_root(entries),
        "bound_files": entries,
        "prospective_source_file_count": prospective[
            "prospective_source_file_count"
        ],
        "prospective_source_root_sha256": prospective[
            "prospective_source_root_sha256"
        ],
        "protocol_path": relative(protocol_path),
        "target_size_input_structure_receipt_path": relative(
            input_structural_receipt_path
        ),
        "target_size_execution_receipt_path": relative(
            target_execution_receipt_path
        ),
        "full_pytest_receipt_path": relative(pytest_receipt_path),
        "full_pytest_log_path": relative(pytest_log_path),
        "trace_streaming_receipt_path": (
            "ijoc_submission_v21e3/provenance/"
            "V21E3R1_TRACE_STREAMING_SMALL_SCALE_V6.json"
        ),
        "v21e3_immutable_parent_zip_path": (
            "ijoc_submission_v21e3/release/ijoc_v21e3_experiment_code.zip"
        ),
        "v21e3_immutable_parent_zip_sha256": prospective[
            "v21e3_immutable_parent_zip_sha256"
        ],
        "immutable_parent_chain_bindings": prospective[
            "immutable_parent_chain_bindings"
        ],
        "v21e2_immutable_baseline_sha256": prospective[
            "immutable_parent_chain_bindings"
        ]["v21e2_immutable_baseline"]["sha256"],
        "v21e2_immutable_calibration_evidence_sha256": prospective[
            "immutable_parent_chain_bindings"
        ]["v21e2_immutable_calibration_evidence"]["sha256"],
        "v21e3_parent_development_snapshot_sha256": prospective[
            "immutable_parent_chain_bindings"
        ]["v21e3_development_snapshot"]["sha256"],
        "v21e3_parent_release_manifest_sha256": prospective[
            "immutable_parent_chain_bindings"
        ]["v21e3_release_manifest"]["sha256"],
        "v21e3_parent_clean_room_receipt_sha256": prospective[
            "immutable_parent_chain_bindings"
        ]["v21e3_clean_room_receipt"]["sha256"],
        "v21e3_parent_zip_checksum_file_sha256": prospective[
            "immutable_parent_chain_bindings"
        ]["v21e3_zip_checksum_file"]["sha256"],
        "authorization": {
            "development_parity_preflight": "NOT_YET_RUN",
            "development_parity_execution": "NOT_AUTHORIZED_BY_SNAPSHOT_ALONE",
            "selection_entropy_release": "PROHIBITED",
            "calibration_execution": "PROHIBITED",
            "formal_execution": "PROHIBITED",
        },
        "selection_partition": "NOT_GENERATED",
        "calibration_partition": "NOT_GENERATED",
        "formal_cases": "NOT_MATERIALIZED",
        "formal_authorized": False,
        "submission_status": "IJOC_HOLD",
        "runtime": {"python": sys.version, "platform": platform.platform()},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(_canonical_bytes(receipt))
    return receipt


def _summary(receipt: Mapping[str, object], output: Path) -> str:
    return json.dumps(
        {
            "status": receipt["status"],
            "bound_file_count": receipt["bound_file_count"],
            "bound_files_root_sha256": receipt["bound_files_root_sha256"],
            "output": str(output.resolve()),
        },
        sort_keys=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--input-structural-receipt",
        type=Path,
        default=DEFAULT_INPUT_STRUCTURAL_RECEIPT,
    )
    parser.add_argument(
        "--target-execution-receipt",
        type=Path,
        default=DEFAULT_TARGET_EXECUTION_RECEIPT,
    )
    parser.add_argument("--pytest-receipt", type=Path, default=DEFAULT_PYTEST_RECEIPT)
    args = parser.parse_args()
    receipt = freeze_development_snapshot(
        repo_root=args.repo_root,
        output=args.output,
        protocol_path=args.protocol,
        input_structural_receipt_path=args.input_structural_receipt,
        target_execution_receipt_path=args.target_execution_receipt,
        pytest_receipt_path=args.pytest_receipt,
    )
    print(_summary(receipt, args.output))


if __name__ == "__main__":
    main()
