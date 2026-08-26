from __future__ import annotations

"""Deterministic, fail-closed freezer for the IJOC study manifests.

The freezer turns a human-readable request plus real input files into the four
byte-bound artifacts consumed by :mod:`mo_nco.pareto_ijoc_preflight` and a
separate cold-process execution plan.  Freezing is deliberately not an
experiment: every emitted receipt keeps the formal evidence state ``NOT_RUN``.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import string
import tarfile
import tempfile
from typing import Any, Mapping, Sequence

from .pareto_ijoc_preflight import audit_ijoc_competitive_study


FREEZE_REQUEST_SCHEMA = "ijoc_manifest_freeze_request_v1"
EXECUTION_PLAN_SCHEMA = "ijoc_cold_process_execution_plan_v1"
FREEZE_RECEIPT_SCHEMA = "ijoc_manifest_freeze_receipt_v1"
CALIBRATION_SUITE_SCHEMA = "ijoc_calibration_suite_receipt_v1"
REFERENCE_CALIBRATION_PRECOMMIT_SCHEMA = (
    "ijoc_reference_calibration_precommit_v1"
)
REFERENCE_CALIBRATION_COMPLETION_SCHEMA = (
    "ijoc_reference_calibration_completion_receipt_v1"
)
TAIL_POLICY_SCHEMA = "ijoc_tail_policy_freeze_v1"
TREATMENT_ID = "ijoc-pareto-smc"

_ROOT_KEYS = {
    "schema",
    "study_id",
    "evidence_status",
    "problem_families",
    "algorithms",
    "seeds",
    "budgets",
    "anytime_checkpoint_period",
    "source_archive_path",
    "dependency_lock_path",
    "tail_calibration_suite_receipt_path",
    "reference_calibration_precommit_path",
    "reference_calibration_completion_receipt_path",
    "tail_policy_artifact_path",
    "formal_analysis_plan_path",
    "python_version",
    "license",
    "reproduction_commands",
}
_FAMILY_KEYS = {"id", "cases", "algorithms", "required_baselines"}
_CASE_KEYS = {"id", "instance_path", "metric_reference"}
_REFERENCE_KEYS = {"source_artifact_path"}
_CALIBRATION_REFERENCE_KEYS = {
    "schema",
    "case_id",
    "source_role",
    "reference_calibration_precommit_sha256",
    "metric_contract",
    "reference_points",
    "ideal",
    "nadir",
    "hv_reference",
}
_METRIC_CONTRACT_KEYS = {
    "objective_sense",
    "dominance_tolerance",
    "normalization",
    "archive_semantics",
    "evaluation_code_sha256",
}
_ALGORITHM_KEYS = {
    "role",
    "families",
    "kind",
    "version",
    "adapter_artifact_path",
    "command_argv",
    "replay_verifier_artifact_path",
    "replay_verifier_argv",
    "configuration",
}
_FORMAL_ANALYSIS_PLAN_KEYS = {
    "schema",
    "plan_id",
    "status",
    "formal_evidence_status",
    "families",
    "treatment",
    "required_baselines",
    "formal_seeds",
    "evaluation_budgets",
    "anytime_checkpoint_period",
    "primary_budget",
    "primary_metric",
    "secondary_metrics",
    "comparison_unit",
    "cluster_unit",
    "family_pooling",
    "budget_pooling",
    "paired_contrast_orientation",
    "uncertainty",
    "wins_ties_losses",
    "primary_gate",
    "efficiency_claim_gate",
    "missing_or_failed_rows",
    "reference_scope",
    "randomness_scope",
}
_ALLOWED_PLACEHOLDERS = {
    "python_executable",
    "adapter_path",
    "replay_verifier_path",
    "input_path",
    "configuration_path",
    "result_path",
    "replay_result_path",
    "instance_path",
    "case_id",
    "algorithm",
    "seed",
    "budget",
    "checkpoint_period",
    "tail_policy_path",
}
_SAFE_STEM = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class FrozenIJOCStudy:
    output_directory: Path
    study_path: Path
    execution_plan_path: Path
    receipt_path: Path
    study_sha256: str
    execution_plan_sha256: str
    expected_run_count: int
    evidence_status: str


@dataclass(frozen=True)
class _FrozenReferenceCalibration:
    precommit_binding: dict[str, str]
    completion_receipt_binding: dict[str, str]
    evidence_binding: dict[str, str]
    frozen_bindings_binding: dict[str, str]
    precommit_sha256: str
    case_descriptors: dict[str, tuple[str, tuple[str, ...]]]
    algorithms: set[str]
    seeds: tuple[int, ...]
    budgets: tuple[int, ...]
    metric_contract: Mapping[str, Any]
    case_output_paths: dict[str, tuple[Path, str, int]]


def _reject_duplicate_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field is forbidden: {key!r}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}.")


def _load_json(path: Path) -> tuple[Mapping[str, Any], bytes]:
    if not path.is_file():
        raise ValueError(f"Freeze request is missing: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Freeze request is not strict UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("Freeze request root must be a JSON object.")
    return value, raw


def _load_calibration_reference(path: Path) -> Mapping[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"Calibration reference is not strict UTF-8 JSON: {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"Calibration reference root must be an object: {path}")
    _exact_keys(value, _CALIBRATION_REFERENCE_KEYS, "calibration reference")
    if value.get("schema") != "ijoc_calibration_reference_case_v1":
        raise ValueError("Calibration reference schema mismatch.")
    return value


def _load_strict_object(path: Path, *, label: str) -> Mapping[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object: {path}")
    return value


def _sha256_string(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal.") from error
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} has an unexpected shape; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}."
        )


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string.")
    return value


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty array.")
    result = tuple(_nonempty_string(item, label) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must be duplicate-free.")
    return result


def _integer_list(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty array.")
    result = tuple(value)
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in result
    ):
        raise ValueError(f"{label} must contain nonnegative integers.")
    if tuple(sorted(result)) != result or len(set(result)) != len(result):
        raise ValueError(f"{label} must be strictly increasing and duplicate-free.")
    return result


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> str:
    raw = _canonical_bytes(value)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _resolve_input(root: Path, raw_path: object, label: str) -> Path:
    text = _nonempty_string(raw_path, label)
    candidate = Path(text)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be relative to the freeze request.")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the freeze-request directory.") from error
    if not resolved.is_file():
        raise ValueError(f"{label} does not name a file: {resolved}")
    return resolved


def _bound_external_file(
    root: Path,
    binding: object,
    *,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} must be an object.")
    _exact_keys(binding, {"path", "sha256"}, label)
    path = _resolve_input(root, binding.get("path"), f"{label}.path")
    declared = _sha256_string(binding.get("sha256"), f"{label}.sha256")
    actual = _file_digest(path)
    if actual != declared:
        raise ValueError(f"{label} SHA-256 mismatch.")
    return path, actual


def _copy_bound_file(source: Path, destination_root: Path) -> dict[str, str]:
    digest = _file_digest(source)
    suffixes = "".join(source.suffixes[-2:])
    suffix = _SAFE_STEM.sub("_", suffixes)[:24]
    destination = destination_root / f"{digest}{suffix}"
    if destination.exists():
        if _file_digest(destination) != digest:
            raise RuntimeError(f"Content-addressed collision at {destination}.")
    else:
        shutil.copyfile(source, destination)
    return {
        "path": destination.relative_to(destination_root.parent).as_posix(),
        "sha256": digest,
    }


def _materialize_runtime_source_archive(
    source_archive: Path,
    *,
    staging_root: Path,
) -> dict[str, str]:
    """Safely extract and byte-bind the one Python source tree used at run time."""

    runtime_root = staging_root / "runtime" / "source"
    runtime_root.mkdir(parents=True)
    maximum_members = 100_000
    maximum_unpacked_bytes = 4 * 1024 * 1024 * 1024
    seen_paths: set[str] = set()
    total_bytes = 0
    try:
        archive = tarfile.open(source_archive, mode="r:*")
    except (tarfile.TarError, OSError) as error:
        raise ValueError(
            "source_archive_path must be a readable tar archive."
        ) from error
    with archive:
        members = archive.getmembers()
        if not members or len(members) > maximum_members:
            raise ValueError("Source archive has an invalid member count.")
        for member in members:
            normalized_text = member.name.replace("\\", "/")
            pieces = tuple(
                piece for piece in normalized_text.split("/") if piece not in ("", ".")
            )
            if (
                not pieces
                or normalized_text.startswith("/")
                or any(piece == ".." for piece in pieces)
                or ":" in pieces[0]
            ):
                raise ValueError(
                    f"Source archive member path is unsafe: {member.name!r}."
                )
            relative = Path(*pieces)
            casefolded = relative.as_posix().casefold()
            if casefolded in seen_paths:
                raise ValueError("Source archive member paths must be unique.")
            seen_paths.add(casefolded)
            if not (member.isfile() or member.isdir()):
                raise ValueError(
                    "Source archive may contain only regular files and directories."
                )
            if member.size < 0:
                raise ValueError("Source archive member has a negative size.")
            total_bytes += int(member.size)
            if total_bytes > maximum_unpacked_bytes:
                raise ValueError("Source archive exceeds the unpacked-size limit.")
            destination = (runtime_root / relative).resolve()
            try:
                destination.relative_to(runtime_root.resolve())
            except ValueError as error:
                raise ValueError("Source archive member escapes runtime root.") from error
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Source archive regular file has no readable body.")
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            if destination.stat().st_size != member.size:
                raise ValueError("Extracted source member byte count changed.")

    files: list[dict[str, object]] = []
    python_roots: set[Path] = set()
    for path in sorted(
        (item for item in runtime_root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(runtime_root).as_posix(),
    ):
        relative = path.relative_to(runtime_root)
        if (
            relative.name == "__init__.py"
            and relative.parent.name == "mo_nco"
        ):
            python_roots.add(relative.parent.parent)
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": _file_digest(path),
                "bytes": path.stat().st_size,
            }
        )
    if not files:
        raise ValueError("Source archive contains no regular files.")
    if len(python_roots) != 1:
        raise ValueError(
            "Source archive must contain exactly one importable mo_nco package."
        )
    python_root = next(iter(python_roots))
    frozen_python_root = runtime_root / python_root
    manifest = {
        "schema": "ijoc_frozen_runtime_source_manifest_v1",
        "source_archive_sha256": _file_digest(source_archive),
        "python_path_root": frozen_python_root.relative_to(
            staging_root
        ).as_posix(),
        "files": files,
    }
    digest = _write_json(staging_root / "runtime_source_manifest.json", manifest)
    return {"path": "runtime_source_manifest.json", "sha256": digest}


def _materialize_case_packet(
    source_packet: Path,
    *,
    case_id: str,
    family_id: str,
    staging_root: Path,
) -> tuple[dict[str, str], tuple[str, ...]]:
    packet = _load_strict_object(source_packet, label=f"case {case_id} packet")
    _exact_keys(
        packet,
        {"schema", "case_id", "family", "problem_sha256", "artifacts"},
        f"case {case_id} packet",
    )
    if packet.get("schema") != "ijoc_case_instance_packet_v1":
        raise ValueError(f"Case {case_id!r} must use an IJOC instance packet.")
    if packet.get("case_id") != case_id:
        raise ValueError(f"Case {case_id!r} packet has the wrong case_id.")
    family = _nonempty_string(packet.get("family"), f"case {case_id} family")
    if family.upper() != family_id.upper():
        raise ValueError(f"Case {case_id!r} packet has the wrong family.")
    _sha256_string(
        packet.get("problem_sha256"), f"case {case_id} problem_sha256"
    )
    raw_artifacts = packet.get("artifacts")
    expected_count = 2 if family.upper() == "MOTSP" else 1 if family.upper() == "MOKP" else None
    if (
        expected_count is None
        or not isinstance(raw_artifacts, list)
        or len(raw_artifacts) != expected_count
    ):
        raise ValueError(
            f"Case {case_id!r} must bind "
            f"{expected_count if expected_count is not None else 'a supported number of'} "
            "raw instance artifacts."
        )
    safe_case = _SAFE_STEM.sub("_", case_id).strip("._") or "case"
    suffix = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:12]
    case_directory = staging_root / "instances" / f"{safe_case}-{suffix}"
    case_directory.mkdir(parents=True)
    frozen_artifacts: list[dict[str, str]] = []
    raw_hashes: list[str] = []
    seen_paths: set[str] = set()
    source_parent = source_packet.parent.resolve()
    for index, raw_artifact in enumerate(raw_artifacts):
        if not isinstance(raw_artifact, dict):
            raise ValueError(f"Case {case_id!r} artifact {index} must be an object.")
        _exact_keys(
            raw_artifact,
            {"path", "sha256"},
            f"case {case_id} artifact {index}",
        )
        raw_path = raw_artifact.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"Case {case_id!r} artifact path must be nonempty.")
        relative = Path(raw_path)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() in seen_paths
        ):
            raise ValueError(f"Case {case_id!r} artifact path is unsafe or duplicate.")
        seen_paths.add(relative.as_posix())
        source = (source_parent / relative).resolve()
        try:
            source.relative_to(source_parent)
        except ValueError as error:
            raise ValueError(
                f"Case {case_id!r} artifact escapes its packet directory."
            ) from error
        if not source.is_file():
            raise ValueError(f"Case {case_id!r} artifact is missing: {source}")
        declared = _sha256_string(
            raw_artifact.get("sha256"),
            f"case {case_id} artifact {index} sha256",
        )
        actual = _file_digest(source)
        if actual != declared:
            raise ValueError(f"Case {case_id!r} artifact SHA-256 mismatch.")
        destination = case_directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if _file_digest(destination) != actual:
            raise RuntimeError(f"Copied case artifact changed for {case_id!r}.")
        frozen_artifacts.append({"path": relative.as_posix(), "sha256": actual})
        raw_hashes.append(actual)
    if len(set(raw_hashes)) != expected_count:
        raise ValueError(f"Case {case_id!r} repeats raw artifact bytes.")
    frozen_packet = {
        "schema": "ijoc_case_instance_packet_v1",
        "case_id": case_id,
        "family": family,
        "problem_sha256": packet["problem_sha256"],
        "artifacts": frozen_artifacts,
    }
    frozen_packet_path = case_directory / "case_packet.json"
    packet_sha = _write_json(frozen_packet_path, frozen_packet)
    return (
        {
            "path": frozen_packet_path.relative_to(staging_root).as_posix(),
            "sha256": packet_sha,
        },
        tuple(raw_hashes),
    )


def _validate_command(
    value: object,
    label: str,
    *,
    required_placeholders: set[str],
) -> list[str]:
    argv = list(_string_list(value, label))
    fields: set[str] = set()
    formatter = string.Formatter()
    for token in argv:
        if any(marker in token.upper() for marker in ("TBD", "TODO", "PLACEHOLDER")):
            raise ValueError(f"{label} contains an unresolved placeholder marker.")
        try:
            pieces = tuple(formatter.parse(token))
        except ValueError as error:
            raise ValueError(f"{label} contains malformed braces: {token!r}.") from error
        for _, field, format_spec, conversion in pieces:
            if field is None:
                continue
            if field not in _ALLOWED_PLACEHOLDERS:
                raise ValueError(f"{label} contains unsupported field {field!r}.")
            if format_spec or conversion:
                raise ValueError(f"{label} may not use conversions or format specs.")
            fields.add(field)
    missing = required_placeholders - fields
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}.")
    return argv


def _numeric_vector(value: object, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"{label} must contain at least two numbers.")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label} must contain only finite numbers.")
        number = float(item)
        if not (-float("inf") < number < float("inf")):
            raise ValueError(f"{label} must contain only finite numbers.")
        result.append(number)
    return result


def _validated_metric_contract(
    value: object,
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    _exact_keys(value, _METRIC_CONTRACT_KEYS, label)
    senses = value.get("objective_sense")
    if (
        not isinstance(senses, list)
        or len(senses) < 2
        or any(sense != "minimize" for sense in senses)
    ):
        raise ValueError(
            f"{label} objective_sense must use canonical minimization."
        )
    tolerance = value.get("dominance_tolerance")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or float(tolerance) != 0.0
    ):
        raise ValueError(f"{label} must use zero-tolerance dominance.")
    if value.get("normalization") != "frozen_ideal_nadir_affine":
        raise ValueError(f"{label} has an unsupported normalization.")
    if (
        value.get("archive_semantics")
        != "calibration_all_evaluated_nondominated"
    ):
        raise ValueError(f"{label} has unsupported reference semantics.")
    _sha256_string(
        value.get("evaluation_code_sha256"),
        f"{label} evaluation_code_sha256",
    )
    return value


def _verified_sized_input(
    source_root: Path,
    value: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Path, str, int]:
    path = _resolve_input(source_root, value.get("path"), f"{label}.path")
    declared = _sha256_string(value.get("sha256"), f"{label}.sha256")
    actual = _file_digest(path)
    size = value.get("bytes")
    if actual != declared:
        raise ValueError(f"{label} SHA-256 mismatch.")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or path.stat().st_size != size
    ):
        raise ValueError(f"{label} byte count mismatch.")
    return path, actual, size


def _freeze_reference_calibration(
    *,
    source_root: Path,
    staging: Path,
    artifacts: Path,
    precommit_path: Path,
    completion_path: Path,
) -> _FrozenReferenceCalibration:
    """Validate and freeze the acyclic reference-calibration chain."""

    precommit = _load_strict_object(
        precommit_path, label="reference calibration precommit"
    )
    _exact_keys(
        precommit,
        {
            "schema",
            "suite_id",
            "status",
            "evidence_scope",
            "cases",
            "algorithms",
            "seeds",
            "budgets",
            "metric_contract",
        },
        "reference calibration precommit",
    )
    if precommit.get("schema") != REFERENCE_CALIBRATION_PRECOMMIT_SCHEMA:
        raise ValueError("Reference calibration precommit schema mismatch.")
    if precommit.get("status") != "PRECOMMITTED":
        raise ValueError("Reference calibration precommit is not PRECOMMITTED.")
    if (
        precommit.get("evidence_scope")
        != "metric_reference_construction_only"
    ):
        raise ValueError("Reference calibration precommit has the wrong scope.")
    suite_id = _nonempty_string(
        precommit.get("suite_id"), "reference calibration suite_id"
    )
    algorithms = set(
        _string_list(
            precommit.get("algorithms"), "reference calibration algorithms"
        )
    )
    seeds = _integer_list(
        precommit.get("seeds"), "reference calibration seeds"
    )
    budgets = _integer_list(
        precommit.get("budgets"), "reference calibration budgets"
    )
    if any(budget <= 0 for budget in budgets):
        raise ValueError("Reference calibration budgets must be positive.")
    metric_contract = _validated_metric_contract(
        precommit.get("metric_contract"),
        label="reference calibration metric_contract",
    )
    raw_cases = precommit.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Reference calibration cases must be nonempty.")
    case_descriptors: dict[str, tuple[str, tuple[str, ...]]] = {}
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(
                f"Reference calibration case {index} must be an object."
            )
        _exact_keys(
            raw_case,
            {"case_id", "family", "instance_artifact_sha256"},
            f"reference calibration case {index}",
        )
        case_id = _nonempty_string(
            raw_case.get("case_id"),
            f"reference calibration case {index} case_id",
        )
        if case_id in case_descriptors:
            raise ValueError("Reference calibration case IDs must be unique.")
        family = _nonempty_string(
            raw_case.get("family"),
            f"reference calibration case {index} family",
        ).upper()
        if family not in {"MOTSP", "MOKP"}:
            raise ValueError(
                "Reference calibration case family must be MOTSP or MOKP."
            )
        raw_hashes = raw_case.get("instance_artifact_sha256")
        expected_count = 2 if family == "MOTSP" else 1
        if (
            not isinstance(raw_hashes, list)
            or len(raw_hashes) != expected_count
        ):
            raise ValueError(
                f"Reference calibration case {case_id!r} must precommit "
                f"{expected_count} raw instance hashes."
            )
        hashes = tuple(
            _sha256_string(
                item,
                f"reference calibration case {case_id} instance hash",
            )
            for item in raw_hashes
        )
        if len(set(hashes)) != len(hashes):
            raise ValueError(
                f"Reference calibration case {case_id!r} repeats raw bytes."
            )
        case_descriptors[case_id] = (family, hashes)

    precommit_sha = _file_digest(precommit_path)
    completion = _load_strict_object(
        completion_path, label="reference calibration completion receipt"
    )
    _exact_keys(
        completion,
        {
            "schema",
            "suite_id",
            "status",
            "evidence_scope",
            "reference_calibration_precommit_sha256",
            "reference_runs",
            "case_outputs",
            "artifact_manifest",
        },
        "reference calibration completion receipt",
    )
    if completion.get("schema") != REFERENCE_CALIBRATION_COMPLETION_SCHEMA:
        raise ValueError("Reference calibration completion schema mismatch.")
    if completion.get("suite_id") != suite_id:
        raise ValueError("Reference calibration suite IDs disagree.")
    if completion.get("status") != "COMPLETE":
        raise ValueError("Reference calibration completion is not COMPLETE.")
    if (
        completion.get("evidence_scope")
        != "metric_reference_construction_only"
    ):
        raise ValueError("Reference calibration completion has the wrong scope.")
    if (
        completion.get("reference_calibration_precommit_sha256")
        != precommit_sha
    ):
        raise ValueError(
            "Reference calibration completion binds the wrong precommit."
        )
    evidence_path, evidence_sha = _bound_external_file(
        completion_path.parent,
        completion.get("artifact_manifest"),
        label="reference calibration artifact_manifest",
    )
    evidence = _load_strict_object(
        evidence_path, label="reference calibration artifact manifest"
    )
    if (
        evidence.get("schema")
        != "ijoc_reference_calibration_completion_evidence_v1"
        or evidence.get("status") != "COMPLETE"
        or evidence.get("reference_calibration_precommit_sha256")
        != precommit_sha
        or evidence.get("reference_runs") != completion.get("reference_runs")
        or evidence.get("case_outputs") != completion.get("case_outputs")
    ):
        raise ValueError(
            "Reference calibration completion evidence does not reproduce "
            "the receipt."
        )

    raw_runs = completion.get("reference_runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("Reference calibration runs must be nonempty.")
    seen_run_keys: set[tuple[str, str, int, int]] = set()
    seen_source_paths: set[str] = set()
    frozen_runs: list[dict[str, object]] = []
    for index, raw_run in enumerate(raw_runs):
        if not isinstance(raw_run, dict):
            raise ValueError(
                f"Reference calibration run {index} must be an object."
            )
        _exact_keys(
            raw_run,
            {"case_id", "algorithm", "seed", "budget", "source_artifacts"},
            f"reference calibration run {index}",
        )
        case_id = _nonempty_string(
            raw_run.get("case_id"),
            f"reference calibration run {index} case_id",
        )
        algorithm = _nonempty_string(
            raw_run.get("algorithm"),
            f"reference calibration run {index} algorithm",
        )
        seed = raw_run.get("seed")
        budget = raw_run.get("budget")
        if (
            case_id not in case_descriptors
            or algorithm not in algorithms
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in seeds
            or isinstance(budget, bool)
            or not isinstance(budget, int)
            or budget not in budgets
        ):
            raise ValueError(
                f"Reference calibration run {index} is outside its precommit."
            )
        run_key = (case_id, algorithm, seed, budget)
        if run_key in seen_run_keys:
            raise ValueError("Reference calibration run keys must be unique.")
        seen_run_keys.add(run_key)
        raw_sources = raw_run.get("source_artifacts")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ValueError(
                f"Reference calibration run {index} needs source artifacts."
            )
        seen_roles: set[str] = set()
        frozen_sources: list[dict[str, object]] = []
        for artifact_index, raw_source in enumerate(raw_sources):
            if not isinstance(raw_source, dict):
                raise ValueError(
                    "Reference calibration source artifact must be an object."
                )
            _exact_keys(
                raw_source,
                {"role", "path", "sha256", "bytes"},
                (
                    f"reference calibration run {index} source "
                    f"{artifact_index}"
                ),
            )
            role = _nonempty_string(
                raw_source.get("role"),
                f"reference calibration run {index} source role",
            )
            raw_path = _nonempty_string(
                raw_source.get("path"),
                f"reference calibration run {index} source path",
            )
            if role in seen_roles or raw_path in seen_source_paths:
                raise ValueError(
                    "Reference calibration source roles per run and paths "
                    "globally must be unique."
                )
            seen_roles.add(role)
            seen_source_paths.add(raw_path)
            source, actual, size = _verified_sized_input(
                source_root,
                raw_source,
                label=f"reference calibration run {index} source",
            )
            frozen_sources.append(
                {
                    "role": role,
                    "source_sha256": actual,
                    "source_bytes": size,
                    "artifact": _copy_bound_file(source, artifacts),
                }
            )
        frozen_runs.append(
            {
                "case_id": case_id,
                "algorithm": algorithm,
                "seed": seed,
                "budget": budget,
                "source_artifacts": frozen_sources,
            }
        )
    expected_run_keys = {
        (case_id, algorithm, seed, budget)
        for case_id in case_descriptors
        for algorithm in algorithms
        for seed in seeds
        for budget in budgets
    }
    if seen_run_keys != expected_run_keys:
        raise ValueError(
            "Reference calibration completion is not the exact precommitted "
            "case-by-algorithm-by-seed-by-budget matrix."
        )

    raw_outputs = completion.get("case_outputs")
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ValueError("Reference calibration case_outputs must be nonempty.")
    case_output_paths: dict[str, tuple[Path, str, int]] = {}
    frozen_outputs: list[dict[str, object]] = []
    for index, raw_output in enumerate(raw_outputs):
        if not isinstance(raw_output, dict):
            raise ValueError(
                f"Reference calibration case output {index} must be an object."
            )
        _exact_keys(
            raw_output,
            {"case_id", "path", "sha256", "bytes"},
            f"reference calibration case output {index}",
        )
        case_id = _nonempty_string(
            raw_output.get("case_id"),
            f"reference calibration case output {index} case_id",
        )
        raw_path = _nonempty_string(
            raw_output.get("path"),
            f"reference calibration case output {index} path",
        )
        if (
            case_id not in case_descriptors
            or case_id in case_output_paths
            or raw_path in seen_source_paths
        ):
            raise ValueError(
                "Reference calibration case outputs must cover unique "
                "precommitted cases with distinct paths."
            )
        seen_source_paths.add(raw_path)
        source, actual, size = _verified_sized_input(
            source_root,
            raw_output,
            label=f"reference calibration case output {index}",
        )
        case_output_paths[case_id] = (source, actual, size)
        frozen_outputs.append(
            {
                "case_id": case_id,
                "source_sha256": actual,
                "source_bytes": size,
                "artifact": _copy_bound_file(source, artifacts),
            }
        )
    if set(case_output_paths) != set(case_descriptors):
        raise ValueError(
            "Reference calibration case outputs do not exactly cover "
            "the precommitted cases."
        )

    precommit_binding = _copy_bound_file(precommit_path, artifacts)
    completion_binding = _copy_bound_file(completion_path, artifacts)
    evidence_binding = _copy_bound_file(evidence_path, artifacts)
    if evidence_binding["sha256"] != evidence_sha:
        raise RuntimeError(
            "Copied reference calibration evidence hash changed."
        )
    frozen_bindings = {
        "schema": "ijoc_frozen_reference_calibration_bindings_v1",
        "reference_calibration_precommit_sha256": precommit_sha,
        "reference_calibration_completion_receipt_sha256": (
            completion_binding["sha256"]
        ),
        "reference_calibration_artifact_manifest_sha256": evidence_sha,
        "reference_runs": frozen_runs,
        "case_outputs": frozen_outputs,
    }
    frozen_bindings_sha = _write_json(
        staging / "reference_calibration_bindings.json",
        frozen_bindings,
    )
    return _FrozenReferenceCalibration(
        precommit_binding=precommit_binding,
        completion_receipt_binding=completion_binding,
        evidence_binding=evidence_binding,
        frozen_bindings_binding={
            "path": "reference_calibration_bindings.json",
            "sha256": frozen_bindings_sha,
        },
        precommit_sha256=precommit_sha,
        case_descriptors=case_descriptors,
        algorithms=algorithms,
        seeds=seeds,
        budgets=budgets,
        metric_contract=metric_contract,
        case_output_paths=case_output_paths,
    )


def freeze_ijoc_study(
    request_path: str | Path,
    output_directory: str | Path,
) -> FrozenIJOCStudy:
    """Freeze a formal-shaped study packet without executing any study row.

    The output directory must not already exist.  All input paths must be
    relative descendants of the request file's directory.  Files are copied
    into the frozen packet under content-addressed names, so later preflight
    does not depend on mutable external paths.
    """

    request_file = Path(request_path).expanduser().resolve()
    request, request_raw = _load_json(request_file)
    _exact_keys(request, _ROOT_KEYS, "freeze request")
    if request.get("schema") != FREEZE_REQUEST_SCHEMA:
        raise ValueError(f"freeze request schema must be {FREEZE_REQUEST_SCHEMA!r}.")
    if request.get("evidence_status") != "NOT_RUN":
        raise ValueError("A freeze request must declare evidence_status='NOT_RUN'.")
    study_id = _nonempty_string(request.get("study_id"), "study_id")
    source_root = request_file.parent.resolve()

    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    committed = False
    try:
        artifacts = staging / "artifacts"
        artifacts.mkdir()

        source_archive_path = _resolve_input(
            source_root,
            request.get("source_archive_path"),
            "source_archive_path",
        )
        source_archive = _copy_bound_file(
            source_archive_path,
            artifacts,
        )
        runtime_source_manifest_binding = _materialize_runtime_source_archive(
            source_archive_path,
            staging_root=staging,
        )
        dependency_lock = _copy_bound_file(
            _resolve_input(
                source_root,
                request.get("dependency_lock_path"),
                "dependency_lock_path",
            ),
            artifacts,
        )
        formal_analysis_plan_path = _resolve_input(
            source_root,
            request.get("formal_analysis_plan_path"),
            "formal_analysis_plan_path",
        )
        formal_analysis_plan = _load_strict_object(
            formal_analysis_plan_path,
            label="formal analysis plan",
        )
        _exact_keys(
            formal_analysis_plan,
            _FORMAL_ANALYSIS_PLAN_KEYS,
            "formal analysis plan",
        )
        if (
            formal_analysis_plan.get("schema")
            != "ijoc_formal_analysis_plan_v1"
            or formal_analysis_plan.get("status")
            != "PRECOMMITTED_BEFORE_FORMAL_EXECUTION"
            or formal_analysis_plan.get("formal_evidence_status") != "NOT_RUN"
        ):
            raise ValueError(
                "Formal analysis plan is not a valid pre-execution precommit."
            )
        formal_analysis_plan_binding = _copy_bound_file(
            formal_analysis_plan_path,
            artifacts,
        )
        calibration_receipt_path = _resolve_input(
            source_root,
            request.get("tail_calibration_suite_receipt_path"),
            "tail_calibration_suite_receipt_path",
        )
        calibration_receipt = _load_strict_object(
            calibration_receipt_path, label="calibration suite receipt"
        )
        _exact_keys(
            calibration_receipt,
            {
                "schema",
                "suite_id",
                "status",
                "evidence_scope",
                "calibration_case_ids",
                "candidate_policy_ids",
                "seeds",
                "artifact_manifest",
                "instance_artifacts",
            },
            "calibration suite receipt",
        )
        if calibration_receipt.get("schema") != CALIBRATION_SUITE_SCHEMA:
            raise ValueError("Calibration suite receipt schema mismatch.")
        if calibration_receipt.get("status") != "COMPLETE":
            raise ValueError("Calibration suite receipt is not COMPLETE.")
        if calibration_receipt.get("evidence_scope") != "tail_policy_selection_only":
            raise ValueError("Calibration suite has an invalid evidence scope.")
        _nonempty_string(calibration_receipt.get("suite_id"), "calibration suite_id")
        calibration_case_ids = set(
            _string_list(
                calibration_receipt.get("calibration_case_ids"),
                "calibration_case_ids",
            )
        )
        candidate_policy_ids = set(
            _string_list(
                calibration_receipt.get("candidate_policy_ids"),
                "candidate_policy_ids",
            )
        )
        calibration_seeds = _integer_list(
            calibration_receipt.get("seeds"), "calibration seeds"
        )
        if not calibration_seeds:
            raise ValueError("Calibration seeds must be nonempty.")
        calibration_manifest_path, calibration_manifest_sha = _bound_external_file(
            calibration_receipt_path.parent,
            calibration_receipt.get("artifact_manifest"),
            label="calibration artifact_manifest",
        )
        calibration_evidence = _load_strict_object(
            calibration_manifest_path, label="calibration artifact manifest"
        )
        if (
            calibration_evidence.get("schema")
            != "ijoc_tail_calibration_evidence_v1"
            or calibration_evidence.get("status") != "COMPLETE"
        ):
            raise ValueError("Calibration artifact manifest is not complete evidence.")
        receipt_instance_artifacts = calibration_receipt.get("instance_artifacts")
        if (
            not isinstance(receipt_instance_artifacts, list)
            or not receipt_instance_artifacts
        ):
            raise ValueError(
                "Calibration receipt instance_artifacts must be nonempty."
            )
        if (
            calibration_evidence.get("calibration_instance_artifacts")
            != receipt_instance_artifacts
        ):
            raise ValueError(
                "Calibration receipt and evidence instance-artifact lists differ."
            )
        calibration_instance_hashes: set[str] = set()
        calibration_instance_counts: dict[str, int] = {}
        calibration_instance_binding_rows: list[dict[str, object]] = []
        seen_calibration_paths: set[str] = set()
        for index, raw_artifact in enumerate(receipt_instance_artifacts):
            if not isinstance(raw_artifact, dict):
                raise ValueError(
                    f"Calibration instance artifact {index} must be an object."
                )
            _exact_keys(
                raw_artifact,
                {"case_id", "family", "path", "sha256", "bytes"},
                f"calibration instance artifact {index}",
            )
            case_id = _nonempty_string(
                raw_artifact.get("case_id"),
                f"calibration instance artifact {index} case_id",
            )
            family = _nonempty_string(
                raw_artifact.get("family"),
                f"calibration instance artifact {index} family",
            ).upper()
            if case_id not in calibration_case_ids:
                raise ValueError(
                    f"Calibration instance artifact names undeclared case {case_id!r}."
                )
            if family not in {"MOTSP", "MOKP"}:
                raise ValueError("Calibration instance family must be MOTSP or MOKP.")
            raw_path = raw_artifact.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError("Calibration instance path must be nonempty.")
            if raw_path in seen_calibration_paths:
                raise ValueError("Calibration instance paths must be unique.")
            seen_calibration_paths.add(raw_path)
            source = _resolve_input(
                source_root,
                raw_path,
                f"calibration instance artifact {index} path",
            )
            declared = _sha256_string(
                raw_artifact.get("sha256"),
                f"calibration instance artifact {index} sha256",
            )
            actual = _file_digest(source)
            if actual != declared:
                raise ValueError("Calibration instance artifact SHA-256 mismatch.")
            size = raw_artifact.get("bytes")
            if (
                isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or source.stat().st_size != size
            ):
                raise ValueError("Calibration instance artifact byte count mismatch.")
            if actual in calibration_instance_hashes:
                raise ValueError("Calibration suite repeats raw instance bytes.")
            calibration_instance_hashes.add(actual)
            calibration_instance_counts[case_id] = (
                calibration_instance_counts.get(case_id, 0) + 1
            )
            calibration_instance_binding_rows.append(
                {
                    "case_id": case_id,
                    "family": family,
                    "source_sha256": actual,
                    "artifact": _copy_bound_file(source, artifacts),
                }
            )
        if set(calibration_instance_counts) != calibration_case_ids:
            raise ValueError(
                "Calibration instance artifacts do not cover every declared case."
            )
        for case_id, count in calibration_instance_counts.items():
            family = next(
                str(row["family"])
                for row in calibration_instance_binding_rows
                if row["case_id"] == case_id
            )
            expected = 2 if family == "MOTSP" else 1
            if count != expected:
                raise ValueError(
                    f"Calibration case {case_id!r} must bind {expected} artifacts."
                )
        calibration_instance_bindings = {
            "schema": "ijoc_frozen_tail_calibration_instance_bindings_v1",
            "tail_calibration_artifact_manifest_sha256": (
                calibration_manifest_sha
            ),
            "artifacts": calibration_instance_binding_rows,
        }
        calibration_instance_bindings_sha = _write_json(
            staging / "tail_calibration_instance_bindings.json",
            calibration_instance_bindings,
        )
        calibration_instance_bindings_binding = {
            "path": "tail_calibration_instance_bindings.json",
            "sha256": calibration_instance_bindings_sha,
        }
        calibration_receipt_binding = _copy_bound_file(
            calibration_receipt_path, artifacts
        )
        calibration_manifest_binding = _copy_bound_file(
            calibration_manifest_path, artifacts
        )
        if calibration_manifest_binding["sha256"] != calibration_manifest_sha:
            raise RuntimeError("Copied calibration artifact manifest hash changed.")
        calibration_receipt_sha = calibration_receipt_binding["sha256"]

        tail_policy_path = _resolve_input(
            source_root,
            request.get("tail_policy_artifact_path"),
            "tail_policy_artifact_path",
        )
        tail_policy = _load_strict_object(
            tail_policy_path, label="tail policy artifact"
        )
        _exact_keys(
            tail_policy,
            {
                "schema",
                "status",
                "policy_id",
                "calibration_suite_sha256",
                "selection_gate",
                "decision_rule",
                "decision_rule_sha256",
                "configuration",
                "fallback_applied",
            },
            "tail policy artifact",
        )
        if tail_policy.get("schema") != TAIL_POLICY_SCHEMA:
            raise ValueError("Tail policy artifact schema mismatch.")
        if tail_policy.get("status") != "FROZEN":
            raise ValueError("Tail policy artifact is not FROZEN.")
        policy_id = _nonempty_string(tail_policy.get("policy_id"), "tail policy_id")
        if policy_id not in candidate_policy_ids:
            raise ValueError("Frozen tail policy was not a calibration candidate.")
        if tail_policy.get("calibration_suite_sha256") != calibration_receipt_sha:
            raise ValueError("Tail policy is bound to the wrong calibration receipt.")
        selection_gate = tail_policy.get("selection_gate")
        fallback_applied = tail_policy.get("fallback_applied")
        if selection_gate not in {"PASS", "FALLBACK"}:
            raise ValueError("Tail policy selection_gate must be PASS or FALLBACK.")
        if not isinstance(fallback_applied, bool) or (
            (selection_gate == "FALLBACK") != fallback_applied
        ):
            raise ValueError("Tail policy fallback flag disagrees with its gate.")
        decision_rule = tail_policy.get("decision_rule")
        if not isinstance(decision_rule, dict) or not decision_rule:
            raise ValueError("Tail policy decision_rule must be a nonempty object.")
        if _canonical_digest(decision_rule) != tail_policy.get(
            "decision_rule_sha256"
        ):
            raise ValueError("Tail policy decision-rule hash mismatch.")
        configuration = tail_policy.get("configuration")
        if not isinstance(configuration, dict) or not configuration:
            raise ValueError("Tail policy configuration must be a nonempty object.")
        tail_policy_binding = _copy_bound_file(tail_policy_path, artifacts)
        tail_policy_sha = tail_policy_binding["sha256"]

        reference_calibration = _freeze_reference_calibration(
            source_root=source_root,
            staging=staging,
            artifacts=artifacts,
            precommit_path=_resolve_input(
                source_root,
                request.get("reference_calibration_precommit_path"),
                "reference_calibration_precommit_path",
            ),
            completion_path=_resolve_input(
                source_root,
                request.get("reference_calibration_completion_receipt_path"),
                "reference_calibration_completion_receipt_path",
            ),
        )

        algorithm_payload = request.get("algorithms")
        if not isinstance(algorithm_payload, dict) or not algorithm_payload:
            raise ValueError("algorithms must be a nonempty JSON object.")
        frozen_algorithms: dict[str, dict[str, object]] = {}
        baseline_bindings: list[dict[str, object]] = []
        for algorithm_id in sorted(algorithm_payload):
            _nonempty_string(algorithm_id, "algorithm id")
            raw = algorithm_payload[algorithm_id]
            if not isinstance(raw, dict):
                raise ValueError(f"Algorithm {algorithm_id!r} must be an object.")
            _exact_keys(raw, _ALGORITHM_KEYS, f"algorithm {algorithm_id}")
            role = raw.get("role")
            if role not in {"treatment", "baseline"}:
                raise ValueError(f"Algorithm {algorithm_id!r} has invalid role.")
            if (algorithm_id == TREATMENT_ID) != (role == "treatment"):
                raise ValueError(
                    f"Exactly {TREATMENT_ID!r} must carry the treatment role."
                )
            families = _string_list(
                raw.get("families"), f"algorithm {algorithm_id} families"
            )
            kind = raw.get("kind")
            if kind not in {"executable", "python_module", "wrapper_script"}:
                raise ValueError(f"Algorithm {algorithm_id!r} has invalid kind.")
            version = _nonempty_string(
                raw.get("version"), f"algorithm {algorithm_id} version"
            )
            configuration = raw.get("configuration")
            if not isinstance(configuration, dict):
                raise ValueError(
                    f"Algorithm {algorithm_id!r} configuration must be an object."
                )
            forbidden_configuration = {
                "case_id",
                "algorithm",
                "seed",
                "budget",
            } & set(configuration)
            if forbidden_configuration:
                raise ValueError(
                    f"Algorithm {algorithm_id!r} configuration shadows row fields: "
                    f"{sorted(forbidden_configuration)}."
                )
            adapter = _copy_bound_file(
                _resolve_input(
                    source_root,
                    raw.get("adapter_artifact_path"),
                    f"algorithm {algorithm_id} adapter_artifact_path",
                ),
                artifacts,
            )
            replay_verifier = _copy_bound_file(
                _resolve_input(
                    source_root,
                    raw.get("replay_verifier_artifact_path"),
                    f"algorithm {algorithm_id} replay_verifier_artifact_path",
                ),
                artifacts,
            )
            command_argv = _validate_command(
                raw.get("command_argv"),
                f"algorithm {algorithm_id} command_argv",
                required_placeholders={"adapter_path", "input_path", "result_path"},
            )
            replay_argv = _validate_command(
                raw.get("replay_verifier_argv"),
                f"algorithm {algorithm_id} replay_verifier_argv",
                required_placeholders={
                    "replay_verifier_path",
                    "input_path",
                    "result_path",
                    "replay_result_path",
                },
            )
            frozen_algorithms[algorithm_id] = {
                "role": role,
                "families": list(families),
                "kind": kind,
                "version": version,
                "adapter_artifact": adapter,
                "command_argv": command_argv,
                "replay_verifier_artifact": replay_verifier,
                "replay_verifier_argv": replay_argv,
                "configuration": configuration,
            }
            if role == "baseline":
                baseline_bindings.append(
                    {
                        "algorithm": algorithm_id,
                        "kind": kind,
                        "version": version,
                        "command": "argv-json:"
                        + json.dumps(command_argv, separators=(",", ":")),
                        "artifact": adapter,
                    }
                )

        families_raw = request.get("problem_families")
        if not isinstance(families_raw, list) or not families_raw:
            raise ValueError("problem_families must be a nonempty array.")
        family_ids: set[str] = set()
        formal_baselines_by_family: dict[str, set[str]] = {}
        case_ids: set[str] = set()
        frozen_families: list[dict[str, object]] = []
        metric_cases: dict[str, dict[str, object]] = {}
        instance_files: list[dict[str, str]] = []
        case_instance_bindings: dict[str, dict[str, str]] = {}
        formal_case_descriptors: dict[
            str, tuple[str, tuple[str, ...]]
        ] = {}
        for family_index, raw_family in enumerate(families_raw):
            if not isinstance(raw_family, dict):
                raise ValueError(f"problem_families[{family_index}] must be an object.")
            _exact_keys(raw_family, _FAMILY_KEYS, f"family {family_index}")
            family_id = _nonempty_string(raw_family.get("id"), "family id")
            if family_id in family_ids:
                raise ValueError(f"Duplicate family id: {family_id!r}.")
            family_ids.add(family_id)
            family_algorithms = _string_list(
                raw_family.get("algorithms"), f"family {family_id} algorithms"
            )
            family_baselines = _string_list(
                raw_family.get("required_baselines"),
                f"family {family_id} required_baselines",
            )
            formal_baselines_by_family[family_id.upper()] = set(
                family_baselines
            )
            if set(family_algorithms) - set(frozen_algorithms):
                raise ValueError(f"Family {family_id!r} names an undefined algorithm.")
            if not set(family_baselines).issubset(family_algorithms):
                raise ValueError(f"Family {family_id!r} baselines leave its algorithms.")
            if TREATMENT_ID not in family_algorithms:
                raise ValueError(f"Family {family_id!r} omits {TREATMENT_ID!r}.")
            for algorithm_id in family_algorithms:
                if family_id not in frozen_algorithms[algorithm_id]["families"]:
                    raise ValueError(
                        f"Algorithm {algorithm_id!r} is not bound to family "
                        f"{family_id!r}."
                    )
            if any(
                frozen_algorithms[algorithm_id]["role"] != "baseline"
                for algorithm_id in family_baselines
            ):
                raise ValueError(f"Family {family_id!r} has a non-baseline requirement.")
            cases_raw = raw_family.get("cases")
            if not isinstance(cases_raw, list) or not cases_raw:
                raise ValueError(f"Family {family_id!r} cases must be nonempty.")
            frozen_case_ids: list[str] = []
            for case_index, raw_case in enumerate(cases_raw):
                if not isinstance(raw_case, dict):
                    raise ValueError(
                        f"Family {family_id!r} case {case_index} must be an object."
                    )
                _exact_keys(
                    raw_case,
                    _CASE_KEYS,
                    f"family {family_id} case {case_index}",
                )
                case_id = _nonempty_string(raw_case.get("id"), "case id")
                if case_id in case_ids:
                    raise ValueError(f"Duplicate case id: {case_id!r}.")
                case_ids.add(case_id)
                frozen_case_ids.append(case_id)
                source_packet = _resolve_input(
                    source_root,
                    raw_case.get("instance_path"),
                    f"case {case_id} instance_path",
                )
                instance_binding, formal_raw_hashes = _materialize_case_packet(
                    source_packet,
                    case_id=case_id,
                    family_id=family_id,
                    staging_root=staging,
                )
                byte_overlap = calibration_instance_hashes & set(
                    formal_raw_hashes
                )
                if byte_overlap:
                    raise ValueError(
                        f"Case {case_id!r} reuses calibration instance bytes: "
                        + ", ".join(sorted(byte_overlap))
                    )
                instance_files.append(
                    {"case_id": case_id, **instance_binding}
                )
                case_instance_bindings[case_id] = instance_binding
                formal_case_descriptors[case_id] = (
                    family_id.upper(),
                    formal_raw_hashes,
                )
                reference = raw_case.get("metric_reference")
                if not isinstance(reference, dict):
                    raise ValueError(
                        f"Case {case_id!r} metric_reference must be an object."
                    )
                _exact_keys(reference, _REFERENCE_KEYS, f"case {case_id} reference")
                reference_source_path = _resolve_input(
                    source_root,
                    reference.get("source_artifact_path"),
                    f"case {case_id} reference source",
                )
                reference_payload = _load_calibration_reference(
                    reference_source_path
                )
                if reference_payload.get("case_id") != case_id:
                    raise ValueError(
                        f"Case {case_id!r} calibration reference has the wrong case_id."
                    )
                if (
                    reference_payload.get("source_role")
                    != (
                        "reference_calibration_precommitted_"
                        "disjoint_arms_and_seeds"
                    )
                ):
                    raise ValueError(
                        f"Case {case_id!r} lacks disjoint calibration provenance."
                    )
                if (
                    _sha256_string(
                        reference_payload.get(
                            "reference_calibration_precommit_sha256"
                        ),
                        (
                            f"case {case_id} "
                            "reference_calibration_precommit_sha256"
                        ),
                    )
                    != reference_calibration.precommit_sha256
                ):
                    raise ValueError(
                        f"Case {case_id!r} references the wrong reference "
                        "calibration precommit."
                    )
                completed_output = (
                    reference_calibration.case_output_paths.get(case_id)
                )
                if completed_output is None:
                    raise ValueError(
                        f"Case {case_id!r} has no completed reference output."
                    )
                completed_path, completed_sha, completed_size = (
                    completed_output
                )
                if (
                    completed_path != reference_source_path
                    or completed_sha != _file_digest(reference_source_path)
                    or completed_size != reference_source_path.stat().st_size
                ):
                    raise ValueError(
                        f"Case {case_id!r} reference output disagrees with "
                        "the completion receipt."
                    )
                metric_contract = _validated_metric_contract(
                    reference_payload.get("metric_contract"),
                    label=f"case {case_id} metric_contract",
                )
                if metric_contract != reference_calibration.metric_contract:
                    raise ValueError(
                        f"Case {case_id!r} metric contract differs from "
                        "the reference-calibration precommit."
                    )
                senses = metric_contract["objective_sense"]
                evaluation_code_sha = _sha256_string(
                    metric_contract.get("evaluation_code_sha256"),
                    f"case {case_id} evaluation_code_sha256",
                )
                for algorithm_id in family_algorithms:
                    replay_binding = frozen_algorithms[algorithm_id][
                        "replay_verifier_artifact"
                    ]
                    if replay_binding["sha256"] != evaluation_code_sha:
                        raise ValueError(
                            f"Case {case_id!r} metric evaluator is not the "
                            f"frozen replay verifier for {algorithm_id!r}."
                        )
                reference_source = _copy_bound_file(
                    reference_source_path,
                    artifacts,
                )
                points_raw = reference_payload.get("reference_points")
                if not isinstance(points_raw, list) or not points_raw:
                    raise ValueError(
                        f"Case {case_id!r} reference_points must be nonempty."
                    )
                points = [
                    _numeric_vector(point, f"case {case_id} reference point")
                    for point in points_raw
                ]
                ideal = _numeric_vector(
                    reference_payload.get("ideal"), f"case {case_id} ideal"
                )
                nadir = _numeric_vector(
                    reference_payload.get("nadir"), f"case {case_id} nadir"
                )
                hv_reference = _numeric_vector(
                    reference_payload.get("hv_reference"),
                    f"case {case_id} hv_reference",
                )
                dimension = len(ideal)
                if (
                    any(len(point) != dimension for point in points)
                    or len(nadir) != dimension
                    or len(hv_reference) != dimension
                    or len(senses) != dimension
                ):
                    raise ValueError(f"Case {case_id!r} metric dimensions disagree.")
                metric_cases[case_id] = {
                    "source_artifact": reference_source,
                    "source_role": reference_payload["source_role"],
                    "reference_sha256": _canonical_digest(
                        [tuple(point) for point in points]
                    ),
                    "reference_points": points,
                    "ideal": ideal,
                    "nadir": nadir,
                    "hv_reference": hv_reference,
                }
            frozen_families.append(
                {
                    "id": family_id,
                    "cases": frozen_case_ids,
                    "algorithms": list(family_algorithms),
                    "required_baselines": list(family_baselines),
                }
            )

        declared_families = {
            family
            for algorithm in frozen_algorithms.values()
            for family in algorithm["families"]
        }
        if declared_families != family_ids:
            raise ValueError(
                "Algorithm family bindings do not exactly match the study families."
            )
        overlap = calibration_case_ids & case_ids
        if overlap:
            raise ValueError(
                "Tail-calibration and formal cases overlap: "
                + ", ".join(sorted(overlap))
            )
        if formal_case_descriptors != reference_calibration.case_descriptors:
            raise ValueError(
                "Reference-calibration precommit cases, families, or raw "
                "instance hashes do not exactly match the formal study."
            )
        reference_arm_overlap = (
            set(frozen_algorithms) & reference_calibration.algorithms
        )
        if reference_arm_overlap:
            raise ValueError(
                "Reference-calibration and formal algorithm arms overlap: "
                + ", ".join(sorted(reference_arm_overlap))
            )

        seeds = _integer_list(request.get("seeds"), "seeds")
        budgets = _integer_list(request.get("budgets"), "budgets")
        reference_seed_overlap = set(seeds) & set(
            reference_calibration.seeds
        )
        if reference_seed_overlap:
            raise ValueError(
                "Reference-calibration and formal seeds overlap: "
                + ", ".join(str(seed) for seed in sorted(reference_seed_overlap))
            )
        if any(budget <= 0 for budget in budgets):
            raise ValueError("budgets must be positive.")
        checkpoint = request.get("anytime_checkpoint_period")
        if (
            isinstance(checkpoint, bool)
            or not isinstance(checkpoint, int)
            or checkpoint <= 0
        ):
            raise ValueError("anytime_checkpoint_period must be positive.")
        if any(budget % checkpoint for budget in budgets):
            raise ValueError("The checkpoint period must divide every budget.")
        analysis_families = set(
            _string_list(
                formal_analysis_plan.get("families"),
                "formal analysis plan families",
            )
        )
        expected_analysis_families = {
            family_id.upper() for family_id in family_ids
        }
        if analysis_families != expected_analysis_families:
            raise ValueError(
                "Formal analysis plan families do not match the matrix."
            )
        if formal_analysis_plan.get("treatment") != TREATMENT_ID:
            raise ValueError(
                "Formal analysis plan treatment does not match the matrix."
            )
        analysis_baselines = formal_analysis_plan.get(
            "required_baselines"
        )
        if (
            not isinstance(analysis_baselines, dict)
            or set(analysis_baselines) != expected_analysis_families
        ):
            raise ValueError(
                "Formal analysis plan baseline families do not match."
            )
        for family, expected_baselines in formal_baselines_by_family.items():
            observed_baselines = set(
                _string_list(
                    analysis_baselines.get(family),
                    f"formal analysis plan baselines for {family}",
                )
            )
            if observed_baselines != expected_baselines:
                raise ValueError(
                    f"Formal analysis plan baselines differ for {family}."
                )
        if (
            _integer_list(
                formal_analysis_plan.get("formal_seeds"),
                "formal analysis plan seeds",
            )
            != seeds
            or _integer_list(
                formal_analysis_plan.get("evaluation_budgets"),
                "formal analysis plan budgets",
            )
            != budgets
            or formal_analysis_plan.get("anytime_checkpoint_period")
            != checkpoint
            or formal_analysis_plan.get("primary_budget") != max(budgets)
        ):
            raise ValueError(
                "Formal analysis plan seed, budget, checkpoint, or primary "
                "budget contract differs from the matrix."
            )

        metric = {
            "schema": "ijoc_metric_reference_manifest_v2",
            "cases": metric_cases,
        }
        metric_sha = _write_json(staging / "metric_reference_manifest.json", metric)

        rows: list[dict[str, object]] = []
        for family in frozen_families:
            for case_id in family["cases"]:
                for algorithm_id in family["algorithms"]:
                    algorithm_configuration = frozen_algorithms[algorithm_id][
                        "configuration"
                    ]
                    for seed in seeds:
                        for budget in budgets:
                            readable = {
                                "case_id": case_id,
                                "algorithm": algorithm_id,
                                "seed": seed,
                                "budget": budget,
                                **algorithm_configuration,
                            }
                            rows.append(
                                {
                                    "case_id": case_id,
                                    "algorithm": algorithm_id,
                                    "seed": seed,
                                    "budget": budget,
                                    "configuration": readable,
                                    "configuration_sha256": _canonical_digest(readable),
                                }
                            )
        configuration_matrix = {
            "schema": "ijoc_algorithm_configuration_matrix_v1",
            "rows": rows,
        }
        config_sha = _write_json(
            staging / "algorithm_configuration_matrix.json",
            configuration_matrix,
        )

        reproduction_commands = _string_list(
            request.get("reproduction_commands"), "reproduction_commands"
        )
        if any(
            marker in command.upper()
            for command in reproduction_commands
            for marker in ("TBD", "TODO", "<PATH>", "PLACEHOLDER")
        ):
            raise ValueError("reproduction_commands contain an unresolved placeholder.")
        reproducibility = {
            "schema": "ijoc_reproducibility_manifest_v2",
            "source_archive": source_archive,
            "formal_analysis_plan": formal_analysis_plan_binding,
            "instance_files": sorted(
                instance_files, key=lambda item: str(item["case_id"])
            ),
            "reproduction_commands": list(reproduction_commands),
            "baseline_bindings": sorted(
                baseline_bindings, key=lambda item: str(item["algorithm"])
            ),
            "license": _nonempty_string(request.get("license"), "license"),
            "environment": {
                "python_version": _nonempty_string(
                    request.get("python_version"), "python_version"
                ),
                "dependency_lock": dependency_lock,
            },
        }
        reproducibility_sha = _write_json(
            staging / "reproducibility_manifest.json", reproducibility
        )

        study = {
            "schema": "ijoc_competitive_study_v3",
            "study_id": study_id,
            "problem_families": frozen_families,
            "seeds": list(seeds),
            "budgets": list(budgets),
            "anytime_checkpoint_period": checkpoint,
            "formal_analysis_plan": formal_analysis_plan_binding,
            "metric_reference_manifest": {
                "path": "metric_reference_manifest.json",
                "sha256": metric_sha,
            },
            "algorithm_configuration_matrix": {
                "path": "algorithm_configuration_matrix.json",
                "sha256": config_sha,
            },
            "artifact_release": {
                "path": "reproducibility_manifest.json",
                "sha256": reproducibility_sha,
            },
        }
        study_sha = _write_json(staging / "study.json", study)
        preflight = audit_ijoc_competitive_study(staging / "study.json")
        if preflight.submission_preflight_gate != "PASS":
            raise ValueError(
                "Frozen study failed its own preflight: "
                + "; ".join(preflight.reasons)
            )

        execution_plan = {
            "schema": EXECUTION_PLAN_SCHEMA,
            "study_sha256": study_sha,
            "configuration_matrix_sha256": config_sha,
            "request_sha256": hashlib.sha256(request_raw).hexdigest(),
            "execution_scope": "formal_candidate",
            "formal_evidence_status": "NOT_RUN",
            "runtime_source_manifest": runtime_source_manifest_binding,
            "formal_analysis_plan": formal_analysis_plan_binding,
            "tail_calibration_suite_receipt": calibration_receipt_binding,
            "tail_calibration_artifact_manifest": calibration_manifest_binding,
            "tail_calibration_instance_bindings": (
                calibration_instance_bindings_binding
            ),
            "reference_calibration_precommit": (
                reference_calibration.precommit_binding
            ),
            "reference_calibration_completion_receipt": (
                reference_calibration.completion_receipt_binding
            ),
            "reference_calibration_artifact_manifest": (
                reference_calibration.evidence_binding
            ),
            "reference_calibration_bindings": (
                reference_calibration.frozen_bindings_binding
            ),
            "tail_policy_artifact": tail_policy_binding,
            "case_instances": case_instance_bindings,
            "algorithms": frozen_algorithms,
        }
        plan_sha = _write_json(staging / "execution_plan.json", execution_plan)
        receipt = {
            "schema": FREEZE_RECEIPT_SCHEMA,
            "status": "FROZEN",
            "formal_evidence_status": "NOT_RUN",
            "expected_run_count": preflight.expected_run_count,
            "artifacts": {
                "study": {"path": "study.json", "sha256": study_sha},
                "metric_reference_manifest": {
                    "path": "metric_reference_manifest.json",
                    "sha256": metric_sha,
                },
                "algorithm_configuration_matrix": {
                    "path": "algorithm_configuration_matrix.json",
                    "sha256": config_sha,
                },
                "reproducibility_manifest": {
                    "path": "reproducibility_manifest.json",
                    "sha256": reproducibility_sha,
                },
                "execution_plan": {
                    "path": "execution_plan.json",
                    "sha256": plan_sha,
                },
                "runtime_source_manifest": runtime_source_manifest_binding,
                "formal_analysis_plan": formal_analysis_plan_binding,
                "tail_calibration_suite_receipt": calibration_receipt_binding,
                "tail_calibration_artifact_manifest": (
                    calibration_manifest_binding
                ),
                "tail_calibration_instance_bindings": (
                    calibration_instance_bindings_binding
                ),
                "reference_calibration_precommit": (
                    reference_calibration.precommit_binding
                ),
                "reference_calibration_completion_receipt": (
                    reference_calibration.completion_receipt_binding
                ),
                "reference_calibration_artifact_manifest": (
                    reference_calibration.evidence_binding
                ),
                "reference_calibration_bindings": (
                    reference_calibration.frozen_bindings_binding
                ),
                "tail_policy_artifact": tail_policy_binding,
            },
        }
        _write_json(staging / "freeze_receipt.json", receipt)
        os.replace(staging, output)
        committed = True
        return FrozenIJOCStudy(
            output_directory=output,
            study_path=output / "study.json",
            execution_plan_path=output / "execution_plan.json",
            receipt_path=output / "freeze_receipt.json",
            study_sha256=study_sha,
            execution_plan_sha256=plan_sha,
            expected_run_count=preflight.expected_run_count,
            evidence_status="NOT_RUN",
        )
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)
