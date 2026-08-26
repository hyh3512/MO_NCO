from __future__ import annotations

"""Independently authorize only the frozen V21e3r1 development parity matrix."""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import sqlite3
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = (
    REPO_ROOT
    / "ijoc_submission_v21e3r1"
    / "provenance"
    / "V21E3R1_DEVELOPMENT_SOURCE_SNAPSHOT_FREEZE_V4.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "ijoc_submission_v21e3r1"
    / "provenance"
    / "V21E3R1_DEVELOPMENT_PARITY_AUTHORIZATION_V4.json"
)
OLD_V21E3_ZIP_SHA256 = (
    "7881b30e6f6059e36e0ed8279f8932ab5f48f2f8e0bc38885e59a74fb45fb3b0"
)
_PARENT_CHAIN_SCALAR_FIELDS = {
    "v21e2_immutable_baseline": "v21e2_immutable_baseline_sha256",
    "v21e2_immutable_calibration_evidence": (
        "v21e2_immutable_calibration_evidence_sha256"
    ),
    "v21e3_development_snapshot": (
        "v21e3_parent_development_snapshot_sha256"
    ),
    "v21e3_release_manifest": "v21e3_parent_release_manifest_sha256",
    "v21e3_clean_room_receipt": (
        "v21e3_parent_clean_room_receipt_sha256"
    ),
    "v21e3_zip_checksum_file": (
        "v21e3_parent_zip_checksum_file_sha256"
    ),
}
_PROSPECTIVE_SOURCE_PATTERNS = (
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
    # These are loaded dynamically by tests/test_pareto_v21e3_release.py and
    # therefore are executable test dependencies even though normal Python
    # import discovery cannot see them.
    "ijoc_submission_v21e3/scripts/build_v21e3_code_release.py",
    "ijoc_submission_v21e3/scripts/verify_v21e3_clean_room.py",
    "ijoc_submission_v21e3/development_manifests_v1/*.json",
    "ijoc_submission_v21e3/development_partitions_v1/case_manifest.json",
    "ijoc_submission_v21e3/development_partitions_v1/instances/*.json",
    "ijoc_submission_v21e3/protocol/DEVELOPMENT_COMMON_BUDGET_PARITY_ASSESSMENT_V2.md",
    "ijoc_submission_v21e3/provenance/development_partition_audit_v1.json",
    "ijoc_submission_v21e3/provenance/V21E3R1_TRACE_STREAMING_SMALL_SCALE_V6.json",
)
_IMMUTABLE_PARENT_PATHS = (
    "ijoc_submission_v21e3/provenance/V21E2_IMMUTABLE_BASELINE.json",
    "ijoc_submission_v21e3/provenance/V21E2_IMMUTABLE_CALIBRATION_EVIDENCE.json",
    "ijoc_submission_v21e3/provenance/V21E3_DEVELOPMENT_SNAPSHOT_FREEZE_V1.json",
    "ijoc_submission_v21e3/release/ijoc_v21e3_experiment_code.manifest.json",
    "ijoc_submission_v21e3/release/ijoc_v21e3_clean_room.receipt.json",
    "ijoc_submission_v21e3/release/ijoc_v21e3_experiment_code.zip.sha256",
)


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


def _sqlite_read_only_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=ro"


def _inside(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"Preflight path escapes repository root: {path}") from error
    return resolved


def _canonical_relative_posix(value: object, *, label: str) -> PurePosixPath:
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
        raise ValueError(f"{label} must be a canonical relative POSIX path.")
    return pure


def _repo_artifact_path(root: Path, value: object, *, label: str) -> Path:
    pure = _canonical_relative_posix(value, label=label)
    return _inside(root, root.joinpath(*pure.parts))


def _row_artifact_path(row_root: Path, value: object, *, label: str) -> Path:
    pure = _canonical_relative_posix(value, label=label)
    resolved_root = row_root.resolve()
    resolved = resolved_root.joinpath(*pure.parts).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its target-size row directory.") from error
    return resolved


def _load_json(path: Path) -> tuple[bytes, dict[str, object]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return raw, value


def _expected_prospective_paths(
    root: Path, snapshot: Mapping[str, object]
) -> list[str]:
    """Independently reconstruct the live source path set.

    Rehashing only paths declared by a snapshot cannot detect a late-added
    executable module.  This reconstruction intentionally duplicates the
    discovery contract instead of trusting the producer's path list.
    """

    selected: set[Path] = set()
    for pattern in _PROSPECTIVE_SOURCE_PATTERNS:
        selected.update(path.resolve() for path in root.glob(pattern) if path.is_file())
    explicit = (
        str(snapshot.get("protocol_path", "")),
        "ijoc_submission_v21e3/release/ijoc_v21e3_experiment_code.zip",
        *_IMMUTABLE_PARENT_PATHS,
    )
    if not explicit[0]:
        raise ValueError("The snapshot omits its protocol path.")
    for relative in explicit:
        path = _inside(root, root / relative)
        if not path.is_file():
            raise ValueError(f"A required prospective source file is absent: {relative}")
        selected.add(path)
    return sorted(path.relative_to(root).as_posix() for path in selected)


def _verify_snapshot(
    root: Path,
    snapshot: Mapping[str, object],
    *,
    expected_v21e3_zip_sha256: str,
) -> dict[str, dict[str, object]]:
    if snapshot.get("schema") != (
        "pareto_v21e3r1_development_source_snapshot_freeze_v1"
    ) or snapshot.get("status") != "PASS_ENGINEERING_SNAPSHOT_ONLY":
        raise ValueError("The V21e3r1 source snapshot is not an engineering PASS.")
    if snapshot.get("scientific_scope") != (
        "source_and_engineering_evidence_provenance_only"
    ):
        raise ValueError("The source snapshot overstates its scope.")
    if snapshot.get("authorization") != {
        "development_parity_preflight": "NOT_YET_RUN",
        "development_parity_execution": "NOT_AUTHORIZED_BY_SNAPSHOT_ALONE",
        "selection_entropy_release": "PROHIBITED",
        "calibration_execution": "PROHIBITED",
        "formal_execution": "PROHIBITED",
    } or snapshot.get("formal_authorized") is not False:
        raise ValueError("The source snapshot opened an execution gate.")
    entries = snapshot.get("bound_files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("The source snapshot has no bound-file entries.")
    paths = [
        str(entry.get("path", "")) if isinstance(entry, Mapping) else ""
        for entry in entries
    ]
    if "" in paths or paths != sorted(paths) or len(set(paths)) != len(paths):
        raise ValueError("Snapshot paths must be unique and canonically ordered.")
    observed: dict[str, dict[str, object]] = {}
    for entry in entries:
        assert isinstance(entry, Mapping)
        relative = str(entry["path"])
        path = _inside(root, root / relative)
        if not path.is_file():
            raise ValueError(f"A snapshot file is absent: {relative}")
        raw = path.read_bytes()
        if entry.get("bytes") != len(raw) or entry.get("sha256") != _sha256(raw):
            raise ValueError(f"A snapshot file drifted: {relative}")
        observed[relative] = {
            "path": relative,
            "bytes": len(raw),
            "sha256": _sha256(raw),
        }
    if snapshot.get("bound_file_count") != len(entries):
        raise ValueError("The snapshot bound-file count is inconsistent.")
    root_digest = _sha256(_canonical_bytes(entries))
    if snapshot.get("bound_files_root_sha256") != root_digest:
        raise ValueError("The source snapshot root digest is invalid.")
    evidence_paths = {
        str(snapshot.get("target_size_input_structure_receipt_path", "")),
        str(snapshot.get("target_size_execution_receipt_path", "")),
        str(snapshot.get("full_pytest_receipt_path", "")),
        str(snapshot.get("full_pytest_log_path", "")),
    }
    if "" in evidence_paths or not evidence_paths.issubset(observed):
        raise ValueError("The snapshot omits a required post-source evidence receipt.")
    prospective_entries = [
        entry for entry in entries if str(entry.get("path", "")) not in evidence_paths
    ]
    prospective_paths = [str(entry.get("path", "")) for entry in prospective_entries]
    expected_prospective_paths = _expected_prospective_paths(root, snapshot)
    if prospective_paths != expected_prospective_paths:
        raise ValueError("The live prospective source path set drifted.")
    prospective_root = _sha256(_canonical_bytes(prospective_entries))
    if not (
        snapshot.get("prospective_source_file_count") == len(prospective_entries)
        and snapshot.get("prospective_source_root_sha256") == prospective_root
    ):
        raise ValueError("The pre-evidence prospective source root is invalid.")
    parent_path = str(snapshot.get("v21e3_immutable_parent_zip_path", ""))
    if not (
        snapshot.get("v21e3_immutable_parent_zip_sha256")
        == expected_v21e3_zip_sha256
        and parent_path in observed
        and observed[parent_path]["sha256"] == expected_v21e3_zip_sha256
    ):
        raise ValueError("The immutable V21e3 parent ZIP binding is invalid.")
    parent_chain = snapshot.get("immutable_parent_chain_bindings")
    if not isinstance(parent_chain, Mapping) or set(parent_chain) != set(
        _PARENT_CHAIN_SCALAR_FIELDS
    ):
        raise ValueError("The immutable V21e2/V21e3 parent chain is incomplete.")
    parent_payloads: dict[str, dict[str, object]] = {}
    for role, scalar_field in _PARENT_CHAIN_SCALAR_FIELDS.items():
        binding = parent_chain[role]
        if not isinstance(binding, Mapping):
            raise ValueError(f"Malformed immutable-parent binding: {role}")
        relative = str(binding.get("path", ""))
        if relative not in observed or not (
            binding.get("bytes") == observed[relative]["bytes"]
            and binding.get("sha256") == observed[relative]["sha256"]
            and snapshot.get(scalar_field) == observed[relative]["sha256"]
        ):
            raise ValueError(f"Immutable-parent binding drifted: {role}")
        path = _inside(root, root / relative)
        if path.suffix == ".json":
            _, payload = _load_json(path)
            parent_payloads[role] = payload
    if not (
        parent_payloads["v21e2_immutable_baseline"].get("status")
        == "IMMUTABLE_CALIBRATION_EVIDENCE_NOT_MODIFIED"
        and parent_payloads["v21e2_immutable_calibration_evidence"].get(
            "status"
        )
        == "IMMUTABLE_CALIBRATION_EVIDENCE"
        and parent_payloads["v21e3_development_snapshot"].get("status")
        == "PASS_ENGINEERING_SNAPSHOT"
        and parent_payloads["v21e3_release_manifest"].get("schema")
        == "ijoc_v21e3_standalone_release_manifest_v1"
        and parent_payloads["v21e3_clean_room_receipt"].get("status") == "PASS"
    ):
        raise ValueError("An immutable-parent receipt has the wrong identity/status.")
    checksum_binding = parent_chain["v21e3_zip_checksum_file"]
    assert isinstance(checksum_binding, Mapping)
    checksum_text = _inside(
        root, root / str(checksum_binding["path"])
    ).read_text(encoding="ascii")
    if expected_v21e3_zip_sha256 not in checksum_text.lower().split():
        raise ValueError("The V21e3 checksum file does not name the immutable ZIP.")
    return observed


def _required_entry(
    entries: Mapping[str, dict[str, object]], path: object, role: str
) -> dict[str, object]:
    relative = str(path or "")
    if relative not in entries:
        raise ValueError(f"The snapshot omits its {role}: {relative}")
    return entries[relative]


def _validate_protocol(protocol: Mapping[str, object]) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if not (
        protocol.get("schema") == "pareto_v21e3_c0_parity_protocol_v2"
        and protocol.get("status")
        == "ENGINEERING_ADAPTERS_AVAILABLE_SUCCESSOR_SNAPSHOT_PENDING"
        and protocol.get("successor_version") == "V21e3r1"
        and protocol.get("families") == ["MOTSP", "MOKP"]
    ):
        raise ValueError("The frozen parity protocol identity is invalid.")
    common = protocol.get("common_execution_contract")
    design = protocol.get("case_design")
    arms = protocol.get("arms")
    if not isinstance(common, Mapping) or not (
        common.get("charged_evaluation_budget") == 2_000
        and common.get("checkpoint_period") == 200
    ):
        raise ValueError("The frozen parity budget/checkpoint grid drifted.")
    if not isinstance(design, Mapping) or not (
        design.get("case_count_per_family") == 6
        and design.get("sizes") == [100, 200, 500]
        and design.get("cases_per_size_per_family") == 2
        and design.get("seeds") == [31051, 31057, 31059]
    ):
        raise ValueError("The frozen parity case/seed design drifted.")
    if not isinstance(arms, Mapping) or set(arms) != {
        "V21E3_C0",
        "NSGAII",
        "MOEAD",
    }:
        raise ValueError("The frozen parity arm set drifted.")
    for arm in ("V21E3_C0", "NSGAII", "MOEAD"):
        contract = arms[arm]
        if not isinstance(contract, Mapping) or contract.get(
            "execution_adapter_status"
        ) != "DEVELOPMENT_ONLY_AVAILABLE":
            raise ValueError(f"The {arm} development adapter is not available.")
    gates = protocol.get("preflight_gates")
    if not isinstance(gates, Mapping) or not (
        gates.get("successor_source_snapshot") == "PENDING"
        and gates.get("independent_protocol_preflight") == "NOT_RUN"
        and gates.get("matched_matrix") == "NOT_RUN"
        and gates.get("selection_entropy_release") == "PROHIBITED"
        and gates.get("calibration_execution") == "PROHIBITED"
        and gates.get("formal_execution") == "PROHIBITED"
        and gates.get("formal_authorized") is False
    ):
        raise ValueError("The frozen parity protocol opened a later-stage gate.")
    return common, design


def _validate_structural_receipt(receipt: Mapping[str, object]) -> Mapping[str, object]:
    if not (
        receipt.get("schema")
        == "pareto_v21e3r1_target_size_input_structure_receipt_v1"
        and receipt.get("status")
        == "PASS_TARGET_SIZE_INPUT_STRUCTURE_ENGINEERING_ONLY"
        and receipt.get("scientific_scope")
        == "pre_freeze_input_structure_not_execution_or_performance_evidence"
        and receipt.get("families") == ["MOTSP", "MOKP"]
        and receipt.get("target_sizes")
        == {"MOTSP": [100, 200, 500], "MOKP": [100, 200, 500]}
        and receipt.get("case_count") == 12
        and receipt.get("seeds") == [31051, 31057, 31059]
        and receipt.get("arms") == ["V21E3_C0", "NSGAII", "MOEAD"]
        and receipt.get("charged_evaluation_budget") == 2_000
        and receipt.get("checkpoint_period") == 200
        and receipt.get("development_parity_execution")
        == "NOT_AUTHORIZED_BY_THIS_RECEIPT"
        and receipt.get("selection_entropy_release") == "PROHIBITED"
        and receipt.get("calibration_execution") == "PROHIBITED"
        and receipt.get("formal_execution") == "PROHIBITED"
        and receipt.get("formal_authorized") is False
    ):
        raise ValueError("The target-size structural receipt is not a fail-closed PASS.")
    bindings = receipt.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "protocol",
        "case_manifest",
        "reference_manifest",
        "config_manifest",
        "metric_manifest",
    }:
        raise ValueError("The target-size receipt has incomplete manifest bindings.")
    return bindings


def _replay_target_metric_from_ledger(
    database_path: Path, metric: Mapping[str, object]
) -> tuple[float, int]:
    analytic_box = metric.get("analytic_box")
    if not isinstance(analytic_box, Mapping):
        raise ValueError("The target-size metric analytic box is absent.")
    lower = analytic_box.get("lower")
    upper = analytic_box.get("upper")
    if not (
        isinstance(lower, list)
        and isinstance(upper, list)
        and len(lower) == len(upper) == 2
        and analytic_box.get("normalized_reference") == [1.0, 1.0]
    ):
        raise ValueError("The target-size metric analytic box is malformed.")
    lower_values = tuple(float(value) for value in lower)
    upper_values = tuple(float(value) for value in upper)
    spans = tuple(high - low for low, high in zip(lower_values, upper_values))
    if any(not math.isfinite(span) or span <= 0.0 for span in spans):
        raise ValueError("The target-size metric analytic box has an invalid span.")
    connection = sqlite3.connect(_sqlite_read_only_uri(database_path), uri=True)
    try:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("The target-size objective ledger failed integrity_check.")
        ledger_rows = list(
            connection.execute(
                "SELECT evaluation_index,objectives_json "
                "FROM evaluations ORDER BY evaluation_index"
            )
        )
    except sqlite3.DatabaseError as error:
        raise ValueError("The target-size objective ledger is not replayable.") from error
    finally:
        connection.close()
    if len(ledger_rows) != 200:
        raise ValueError("The target-size objective ledger is not the 200-row ledger.")
    normalized: list[tuple[float, float]] = []
    for expected_index, ledger_row in enumerate(ledger_rows, start=1):
        if int(ledger_row[0]) != expected_index:
            raise ValueError("The target-size objective ledger index grid drifted.")
        objective = json.loads(str(ledger_row[1]))
        if not isinstance(objective, list) or len(objective) != 2:
            raise ValueError("The target-size objective ledger is not biobjective.")
        point = tuple(
            (float(value) - low) / span
            for value, low, span in zip(objective, lower_values, spans)
        )
        if any(
            not math.isfinite(value) or value < -1e-12 or value > 1.0 + 1e-12
            for value in point
        ):
            raise ValueError("A target-size objective escaped the analytic box.")
        normalized.append((float(point[0]), float(point[1])))
    unique = sorted(set(normalized))
    nondominated = [
        point
        for point in unique
        if not any(
            other != point
            and other[0] <= point[0]
            and other[1] <= point[1]
            for other in unique
        )
    ]
    hypervolume = 0.0
    best_y = 1.0
    for x_value, y_value in sorted(nondominated):
        if y_value < best_y:
            hypervolume += (1.0 - x_value) * (best_y - y_value)
            best_y = y_value
    if not -1e-12 <= hypervolume <= 1.0 + 1e-12:
        raise ValueError("The target-size replayed metric escaped [0,1].")
    return min(1.0, max(0.0, hypervolume)), len(nondominated)


def _valid_target_metric_grid(
    metric: Mapping[str, object],
    row: Mapping[str, object],
    *,
    database_path: Path,
) -> bool:
    checkpoints = metric.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 1:
        return False
    checkpoint = checkpoints[0]
    if not isinstance(checkpoint, Mapping):
        return False
    normalized_hv = checkpoint.get("normalized_hv")
    archive_size = checkpoint.get("archive_size")
    terminal_hv = metric.get("normalized_terminal_hv")
    auc = metric.get("normalized_left_continuous_hv_auc")
    numeric = (normalized_hv, terminal_hv, auc)
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
        for value in numeric
    ):
        return False
    replayed_hv, replayed_archive_size = _replay_target_metric_from_ledger(
        database_path, metric
    )
    return bool(
        metric.get("verification_scope")
        == "frozen_metric_from_objective_ledger_checkpoints_v1"
        and metric.get("charged_evaluation_budget") == 200
        and metric.get("checkpoint_period") == 200
        and float(auc) == 0.0
        and checkpoint.get("evaluation") == 200
        and isinstance(archive_size, int)
        and not isinstance(archive_size, bool)
        and archive_size > 0
        and terminal_hv == normalized_hv
        and float(terminal_hv) == replayed_hv
        and archive_size == replayed_archive_size
        and metric.get("checkpoints") == row.get("checkpoints")
    )


def _is_lower_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _target_identity(value: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(
        value.get(field)
        for field in ("case_id", "family", "size", "seed", "arm_id")
    )


def _terminal_payload_sha256(terminal: Mapping[str, object]) -> str | None:
    core = dict(terminal)
    embedded = core.pop("receipt_payload_sha256", None)
    raw = json.dumps(
        core, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    computed = hashlib.sha256(raw).hexdigest()
    return computed if embedded == computed else None


def _validate_target_terminal_objective_contract(
    *,
    terminal: Mapping[str, object],
    pre: Mapping[str, object],
    replay: Mapping[str, object],
    metric: Mapping[str, object],
    row: Mapping[str, object],
    database_path: Path,
    prospective_source_root_sha256: object,
) -> None:
    terminal_payload_sha256 = _terminal_payload_sha256(terminal)
    attempts = terminal.get("attempt_count")
    cache_hits = terminal.get("cache_hit_count")
    context_sha256 = terminal.get("run_context_digest_sha256")
    evaluation_chain = terminal.get("terminal_evaluation_chain_sha256")
    decision_chain = terminal.get("terminal_decision_chain_sha256")
    attempt_chain = terminal.get("terminal_attempt_chain_sha256")
    if not (
        terminal.get("schema") == "v21e3_terminal_receipt_v1"
        and terminal.get("status") == "SUCCESS"
        and terminal.get("problem") == row.get("case_id")
        and terminal.get("family") == row.get("family")
        and terminal.get("failure_code") is None
        and terminal.get("failure_detail") is None
        and isinstance(attempts, int)
        and not isinstance(attempts, bool)
        and isinstance(cache_hits, int)
        and not isinstance(cache_hits, bool)
        and cache_hits >= 0
        and attempts == 200 + cache_hits
        and terminal.get("physical_call_started_count") == 200
        and terminal.get("charged_evaluation_count") == 200
        and terminal.get("decision_count") == 200
        and terminal.get("unresolved_decision_count") == 0
        and _is_lower_sha256(context_sha256)
        and _is_lower_sha256(evaluation_chain)
        and _is_lower_sha256(decision_chain)
        and _is_lower_sha256(attempt_chain)
        and terminal.get("database_path") == "trace.sqlite3"
        and terminal.get("durability_mode") == "SQLITE_WAL_SYNCHRONOUS_FULL"
        and terminal_payload_sha256 is not None
    ):
        raise ValueError("A target-size terminal receipt is not a successful terminal.")
    expected_gates = {
        "expected_charged_evaluations": 200,
        "expected_decisions": 200,
        "run_context_charged_evaluation_budget": 200,
        "persisted_attempts": attempts,
        "persisted_evaluations": 200,
        "persisted_decisions": 200,
        "physical_call_starts": 200,
        "cache_hits": cache_hits,
        "nonterminal_attempts": 0,
        "evaluation_index_bounds": [1, 200],
        "expected_evaluation_index_bounds": [1, 200],
        "sqlite_integrity": "ok",
    }
    if terminal.get("finalization_gates") != expected_gates:
        raise ValueError("The target-size terminal finalization gates drifted.")
    identity_fields = ("case_id", "family", "size", "seed", "arm_id")
    if not (
        pre.get("schema") == "pareto_v21e3r1_row_preverification_v1"
        and pre.get("status") == "READY_FOR_OBJECTIVE_AND_ARCHIVE_REPLAY"
        and all(pre.get(field) == row.get(field) for field in identity_fields)
        and pre.get("source_snapshot_root_sha256")
        == prospective_source_root_sha256
        and pre.get("selection_entropy_release") == "PROHIBITED"
        and pre.get("calibration_execution") == "PROHIBITED"
        and pre.get("formal_execution") == "PROHIBITED"
        and pre.get("formal_authorized") is False
    ):
        raise ValueError("A target-size preverification receipt is not fail-closed.")
    checkpoints = metric.get("checkpoints")
    checkpoint = checkpoints[0] if isinstance(checkpoints, list) and checkpoints else None
    if not (
        replay.get("schema") == "v21e3r1_objective_archive_replay_receipt_v2"
        and replay.get("status") == "OBJECTIVE_AND_ARCHIVE_REPLAY_PASS"
        and replay.get("verification_scope")
        == "objective_solution_chain_archive_and_terminal_replay_v1"
        and replay.get("database_bytes") == database_path.stat().st_size
        and replay.get("database_bytes") == row.get("trace_database_bytes")
        and replay.get("attempt_records") == attempts
        and replay.get("evaluation_records") == 200
        and replay.get("decision_records") == 200
        and replay.get("cache_hit_records") == cache_hits
        and replay.get("unique_solution_replays") == 200
        and replay.get("archive_reconstruction") == "PASS"
        and isinstance(checkpoint, Mapping)
        and replay.get("archive_size") == checkpoint.get("archive_size")
        and replay.get("terminal_status") == "SUCCESS"
        and replay.get("run_context_digest_sha256") == context_sha256
        and replay.get("run_context_digest_sha256")
        == row.get("run_context_digest_sha256")
        and replay.get("terminal_evaluation_chain_sha256") == evaluation_chain
        and replay.get("terminal_decision_chain_sha256") == decision_chain
        and replay.get("terminal_attempt_chain_sha256") == attempt_chain
        and replay.get("terminal_receipt_sha256") == terminal_payload_sha256
        and replay.get("full_algorithm_decision_replay") == "NOT_IMPLEMENTED"
        and replay.get("selection_authorization") == "PROHIBITED"
    ):
        raise ValueError(
            "A target-size terminal/objective/metric replay cross-binding failed."
        )


def _validate_target_execution_receipt(
    receipt: Mapping[str, object], *, root: Path,
    prospective_source_root_sha256: object,
    expected_input_sha256: Mapping[str, str],
    expected_reference_cases: Mapping[str, Mapping[str, object]] | None = None,
) -> str:
    if expected_reference_cases is None:
        raise ValueError("The target-size frozen reference cases are required.")
    rows = receipt.get("rows")
    observed_pairs = {
        (row.get("family"), row.get("arm_id"))
        for row in rows
        if isinstance(row, Mapping)
    } if isinstance(rows, list) else set()
    if not (
        receipt.get("schema")
        == "pareto_v21e3r1_target_size_execution_receipt_v1"
        and receipt.get("status")
        == "PASS_TARGET_SIZE_THREE_ARM_EXECUTION_ENGINEERING_ONLY"
        and receipt.get("scientific_scope")
        == (
            "target_size_small_budget_structure_and_objective_archive_replay_"
            "not_performance_evidence"
        )
        and receipt.get("artifact_path_semantics")
        == "repo_root_relative_posix_v1"
        and receipt.get("source_snapshot_root_sha256")
        == prospective_source_root_sha256
        and receipt.get("target_size") == 500
        and receipt.get("seed") == 31051
        and receipt.get("charged_evaluation_budget") == 200
        and receipt.get("checkpoint_period") == 200
        and receipt.get("row_count") == 6
        and receipt.get("families") == ["MOTSP", "MOKP"]
        and receipt.get("arms") == ["V21E3_C0", "NSGAII", "MOEAD"]
        and receipt.get("input_sha256") == dict(expected_input_sha256)
        and receipt.get("baseline_population_sizes")
        == {"MOTSP": 48, "MOKP": 40}
        and receipt.get("budget_initializes_every_baseline_population") is True
        and isinstance(rows, list)
        and len(rows) == 6
        and observed_pairs
        == {
            (family, arm)
            for family in ("MOTSP", "MOKP")
            for arm in ("V21E3_C0", "NSGAII", "MOEAD")
        }
        and all(
            isinstance(row, Mapping)
            and row.get("size") == 500
            and row.get("seed") == 31051
            and row.get("charged_evaluation_budget") == 200
            and row.get("checkpoint_period") == 200
            and row.get("objective_and_archive_replay") == "PASS"
            and row.get("metric_replay") == "PASS"
            for row in rows
        )
        and receipt.get("objective_and_archive_replay_pass_rows") == 6
        and receipt.get("metric_replay_pass_rows") == 6
        and receipt.get("target_budget_evidence") == "NOT_ESTABLISHED"
        and receipt.get("performance_evidence") == "NOT_ESTABLISHED"
        and receipt.get("runtime_efficiency_claim_authorized") is False
        and receipt.get("full_algorithm_decision_replay") == "NOT_IMPLEMENTED"
        and receipt.get("development_parity_execution")
        == "NOT_AUTHORIZED_BY_THIS_RECEIPT"
        and receipt.get("selection_entropy_release") == "PROHIBITED"
        and receipt.get("calibration_execution") == "PROHIBITED"
        and receipt.get("formal_execution") == "PROHIBITED"
        and receipt.get("formal_authorized") is False
    ):
        raise ValueError("The three-arm target-size execution receipt is not PASS.")
    plan_path = _repo_artifact_path(
        root, receipt.get("plan_path", ""), label="target-size plan_path"
    )
    if (
        not plan_path.is_file()
        or plan_path.is_symlink()
        or _sha256(plan_path.read_bytes()) != receipt.get("plan_sha256")
    ):
        raise ValueError("The target-size execution plan binding failed.")
    _, plan = _load_json(plan_path)
    plan_rows = plan.get("rows")
    if not (
        plan.get("schema") == "pareto_v21e3r1_target_size_three_arm_plan_v1"
        and plan.get("status") == "READY_TARGET_SIZE_SMALL_BUDGET_ENGINEERING_ONLY"
        and plan.get("scientific_scope")
        == (
            "target_size_small_budget_structure_and_objective_archive_replay_"
            "not_performance_evidence"
        )
        and plan.get("budget") == 200
        and plan.get("checkpoint_period") == 200
        and plan.get("baseline_population_sizes") == {"MOTSP": 48, "MOKP": 40}
        and plan.get("budget_initializes_every_baseline_population") is True
        and plan.get("source_snapshot_root_sha256")
        == prospective_source_root_sha256
        and plan.get("input_sha256") == dict(expected_input_sha256)
        and plan.get("development_parity_execution")
        == "NOT_AUTHORIZED_BY_THIS_PLAN"
        and plan.get("selection_entropy_release") == "PROHIBITED"
        and plan.get("calibration_execution") == "PROHIBITED"
        and plan.get("formal_execution") == "PROHIBITED"
        and plan.get("formal_authorized") is False
        and isinstance(plan_rows, list)
        and len(plan_rows) == 6
    ):
        raise ValueError("The target-size execution plan is not fail-closed.")
    target_case_by_family: dict[str, str] = {}
    for family in ("MOTSP", "MOKP"):
        candidates = sorted(
            case_id
            for case_id, case in expected_reference_cases.items()
            if case.get("family") == family and case.get("size") == 500
        )
        if not candidates:
            raise ValueError("The frozen reference omits a target-size case.")
        target_case_by_family[family] = candidates[0]
    expected_plan_identities = {
        (target_case_by_family[family], family, 500, 31051, arm)
        for family in ("MOTSP", "MOKP")
        for arm in ("V21E3_C0", "NSGAII", "MOEAD")
    }
    planned_by_identity: dict[tuple[object, ...], Mapping[str, object]] = {}
    planned_slugs: set[str] = set()
    assert isinstance(plan_rows, list)
    for planned in plan_rows:
        if not isinstance(planned, Mapping):
            raise ValueError("The target-size plan contains a non-object row.")
        identity = _target_identity(planned)
        slug = str(planned.get("row_slug", ""))
        expected_slug = (
            f"{planned.get('case_id')}__seed-31051__arm-"
            f"{str(planned.get('arm_id', '')).lower()}"
        )
        if (
            identity not in expected_plan_identities
            or identity in planned_by_identity
            or slug != expected_slug
            or slug in planned_slugs
            or "/" in slug
            or "\\" in slug
        ):
            raise ValueError("The target-size plan row identity or slug drifted.")
        planned_by_identity[identity] = planned
        planned_slugs.add(slug)
    if set(planned_by_identity) != expected_plan_identities:
        raise ValueError("The target-size plan is not the exact six-row product.")
    rows_root = plan_path.parent / "rows"
    if not rows_root.is_dir() or rows_root.is_symlink():
        raise ValueError("The target-size plan rows directory is absent.")
    observed_row_roots = list(rows_root.iterdir())
    if (
        {path.name for path in observed_row_roots} != planned_slugs
        or len(observed_row_roots) != 6
        or any(not path.is_dir() or path.is_symlink() for path in observed_row_roots)
    ):
        raise ValueError("The target-size plan does not bind exactly six row directories.")
    child_bindings: list[dict[str, object]] = []
    assert isinstance(rows, list)
    for summary in rows:
        assert isinstance(summary, Mapping)
        identity = _target_identity(summary)
        planned = planned_by_identity.get(identity)
        if not isinstance(planned, Mapping):
            raise ValueError("A target-size receipt row is absent from the plan.")
        expected_row_path = (
            rows_root / str(planned["row_slug"]) / "row.json"
        ).relative_to(root).as_posix()
        if summary.get("row_receipt_path") != expected_row_path:
            raise ValueError("A target-size row receipt path disagrees with the plan.")
        row_path = _repo_artifact_path(
            root,
            summary.get("row_receipt_path", ""),
            label="target-size row_receipt_path",
        )
        if not row_path.is_file() or _sha256(row_path.read_bytes()) != summary.get(
            "row_receipt_sha256"
        ):
            raise ValueError("A target-size row receipt binding failed.")
        row_raw, row = _load_json(row_path)
        for field in (
            "case_id",
            "family",
            "size",
            "seed",
            "arm_id",
            "charged_evaluation_budget",
            "checkpoint_period",
            "trace_database_sha256",
            "detached_terminal_receipt_sha256",
            "objective_archive_replay_receipt_sha256",
            "metric_replay_receipt_sha256",
        ):
            if row.get(field) != summary.get(field):
                raise ValueError(f"Target-size row summary mismatch: {field}")
        if summary.get("normalized_terminal_hv_diagnostic_only") != row.get(
            "normalized_terminal_hv"
        ):
            raise ValueError("Target-size row summary mismatch: normalized terminal HV")
        elapsed_ns = row.get("elapsed_process_wall_ns_diagnostic_only")
        trace_bytes = row.get("trace_database_bytes")
        if not (
            row.get("schema") == "pareto_v21e3r1_matched_matrix_row_v1"
            and row.get("status")
            == "PASS_ENGINEERING_ROW_OBJECTIVE_ARCHIVE_REPLAYED"
            and row.get("scientific_scope")
            == (
                "target_size_small_budget_structure_and_objective_archive_replay_"
                "not_performance_evidence"
            )
            and row.get("artifact_path_semantics")
            == "row_directory_relative_posix_v1"
            and row.get("source_snapshot_root_sha256")
            == prospective_source_root_sha256
            and _is_lower_sha256(row.get("runtime_problem_semantic_sha256"))
            and _is_lower_sha256(row.get("run_context_digest_sha256"))
            and isinstance(trace_bytes, int)
            and not isinstance(trace_bytes, bool)
            and trace_bytes > 0
            and isinstance(elapsed_ns, int)
            and not isinstance(elapsed_ns, bool)
            and elapsed_ns >= 0
            and row.get("metric_manifest_sha256")
            == expected_input_sha256["metric_manifest"]
            and row.get("full_algorithm_decision_replay") == "NOT_IMPLEMENTED"
            and row.get("runtime_efficiency_claim_authorized") is False
            and row.get("selection_entropy_release") == "PROHIBITED"
            and row.get("calibration_execution") == "PROHIBITED"
            and row.get("formal_execution") == "PROHIBITED"
            and row.get("formal_authorized") is False
        ):
            raise ValueError("A target-size row receipt is not fail-closed.")
        child_bindings.append({
            "path": row_path.relative_to(root).as_posix(),
            "bytes": len(row_raw),
            "sha256": _sha256(row_raw),
        })
        row_root = row_path.parent
        expected_row_names = {
            "trace.sqlite3",
            "terminal.receipt.json",
            "row.preverification.json",
            "objective_archive_replay.receipt.json",
            "metric_replay.receipt.json",
            "row.json",
        }
        row_children = list(row_root.iterdir())
        if (
            row_path.name != "row.json"
            or row_path.is_symlink()
            or {path.name for path in row_children} != expected_row_names
            or len(row_children) != 6
            or any(not path.is_file() or path.is_symlink() for path in row_children)
        ):
            raise ValueError(
                "A target-size row must contain exactly six regular artifacts."
            )
        exact_row_paths = {
            "trace_database_path": "trace.sqlite3",
            "detached_terminal_receipt_path": "terminal.receipt.json",
            "preverification_receipt_path": "row.preverification.json",
            "objective_archive_replay_receipt_path": (
                "objective_archive_replay.receipt.json"
            ),
            "metric_replay_receipt_path": "metric_replay.receipt.json",
        }
        if any(row.get(field) != expected for field, expected in exact_row_paths.items()):
            raise ValueError("A target-size row artifact path is not the fixed basename.")
        child_specs = (
            ("trace_database_path", "trace_database_sha256", None),
            (
                "detached_terminal_receipt_path",
                "detached_terminal_receipt_sha256",
                "terminal",
            ),
            (
                "preverification_receipt_path",
                "preverification_receipt_sha256",
                "preverification",
            ),
            (
                "objective_archive_replay_receipt_path",
                "objective_archive_replay_receipt_sha256",
                "replay",
            ),
            (
                "metric_replay_receipt_path",
                "metric_replay_receipt_sha256",
                "metric_replay",
            ),
        )
        loaded: dict[str, dict[str, object]] = {}
        child_paths: dict[str, Path] = {}
        for path_field, hash_field, json_role in child_specs:
            child = _row_artifact_path(
                row_root,
                row.get(path_field, ""),
                label=f"target-size {path_field}",
            )
            if not child.is_file():
                raise ValueError(f"A target-size child artifact is absent: {path_field}")
            raw = child.read_bytes()
            digest = _sha256(raw)
            if digest != row.get(hash_field):
                raise ValueError(f"A target-size child artifact drifted: {path_field}")
            child_paths[path_field] = child
            child_bindings.append({
                "path": child.relative_to(root).as_posix(),
                "bytes": len(raw),
                "sha256": digest,
            })
            if json_role is not None:
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError(f"Malformed target-size {json_role} receipt.")
                loaded[json_role] = value
        pre = loaded["preverification"]
        terminal = loaded["terminal"]
        replay = loaded["replay"]
        metric_replay = loaded["metric_replay"]
        reference_case = expected_reference_cases.get(str(row.get("case_id", "")))
        if not isinstance(reference_case, Mapping):
            raise ValueError("A target-size row is absent from the frozen reference.")
        expected_analytic_box = {
            "lower": reference_case.get("objective_lower_bounds"),
            "upper": reference_case.get("objective_upper_bounds"),
            "normalized_reference": reference_case.get(
                "normalized_reference_point"
            ),
        }
        if not (
            reference_case.get("family") == row.get("family")
            and reference_case.get("size") == row.get("size")
            and row.get("case_artifact_sha256")
            == reference_case.get("artifact_sha256")
            and row.get("generator_problem_fingerprint_sha256")
            == reference_case.get("problem_sha256")
            and metric_replay.get("analytic_box") == expected_analytic_box
        ):
            raise ValueError(
                "A target-size metric analytic box or case binding drifted."
            )
        _validate_target_terminal_objective_contract(
            terminal=terminal,
            pre=pre,
            replay=replay,
            metric=metric_replay,
            row=row,
            database_path=child_paths["trace_database_path"],
            prospective_source_root_sha256=prospective_source_root_sha256,
        )
        if not (
            terminal.get("database_path") == "trace.sqlite3"
            and pre.get("status") == "READY_FOR_OBJECTIVE_AND_ARCHIVE_REPLAY"
            and pre.get("trace_database_path") == "trace.sqlite3"
            and pre.get("detached_terminal_receipt_path")
            == "terminal.receipt.json"
            and pre.get("detached_terminal_receipt_sha256")
            == row.get("detached_terminal_receipt_sha256")
            and replay.get("schema")
            == "v21e3r1_objective_archive_replay_receipt_v2"
            and replay.get("status") == "OBJECTIVE_AND_ARCHIVE_REPLAY_PASS"
            and replay.get("database_path") == "trace.sqlite3"
            and replay.get("detached_terminal_receipt_path")
            == "terminal.receipt.json"
            and replay.get("database_sha256") == row.get("trace_database_sha256")
            and replay.get("detached_terminal_receipt_sha256")
            == row.get("detached_terminal_receipt_sha256")
            and replay.get("full_algorithm_decision_replay") == "NOT_IMPLEMENTED"
            and replay.get("selection_authorization") == "PROHIBITED"
            and metric_replay.get("schema")
            == "pareto_v21e3r1_metric_replay_receipt_v1"
            and metric_replay.get("status") == "NORMALIZED_HV_AUC_REPLAY_PASS"
            and metric_replay.get("database_path") == "trace.sqlite3"
            and metric_replay.get("database_sha256")
            == row.get("trace_database_sha256")
            and metric_replay.get("metric_manifest_sha256")
            == expected_input_sha256["metric_manifest"]
            and _valid_target_metric_grid(
                metric_replay,
                row,
                database_path=child_paths["trace_database_path"],
            )
            and metric_replay.get("normalized_left_continuous_hv_auc")
            == row.get("normalized_left_continuous_hv_auc")
            and metric_replay.get("normalized_terminal_hv")
            == row.get("normalized_terminal_hv")
            and metric_replay.get("checkpoints") == row.get("checkpoints")
            and metric_replay.get("selection_authorization") == "PROHIBITED"
            and metric_replay.get("formal_authorized") is False
        ):
            raise ValueError(
                "A target-size objective/archive or metric replay binding failed."
            )
    child_bindings.sort(key=lambda item: str(item["path"]))
    return _sha256(_canonical_bytes(child_bindings))


def _validate_full_pytest_receipt(
    receipt: Mapping[str, object],
    *,
    root: Path,
    expected_prospective_source_root_sha256: object,
    expected_log_path: object,
    entries: Mapping[str, Mapping[str, object]],
) -> None:
    if not (
        receipt.get("schema") == "pareto_v21e3r1_full_pytest_receipt_v1"
        and receipt.get("status") == "PASS"
        and receipt.get("suite_scope") == "repository_full_pytest_q_v1"
        and receipt.get("prospective_source_root_sha256")
        == expected_prospective_source_root_sha256
        and receipt.get("exit_code") == 0
        and receipt.get("artifact_path_semantics")
        == "repo_root_relative_posix_v1"
        and receipt.get("cwd") == "."
        and receipt.get("cwd_path_semantics") == "repo_root_self_v1"
        and isinstance(receipt.get("passed"), int)
        and receipt.get("passed", 0) > 0
        and receipt.get("failed") == 0
        and receipt.get("errors") == 0
        and receipt.get("selection_authorization") == "PROHIBITED"
        and receipt.get("formal_authorized") is False
    ):
        raise ValueError("The complete repository pytest receipt is not PASS.")
    log_path = _repo_artifact_path(
        root, receipt.get("log_path", ""), label="full pytest log_path"
    )
    expected_relative = str(expected_log_path)
    if log_path.relative_to(root).as_posix() != expected_relative:
        raise ValueError("The pytest receipt and snapshot name another log.")
    log_raw = log_path.read_bytes()
    log_entry = entries.get(expected_relative)
    if not isinstance(log_entry, Mapping) or not (
        receipt.get("log_bytes") == len(log_raw)
        and receipt.get("log_sha256") == _sha256(log_raw)
        and log_entry.get("bytes") == len(log_raw)
        and log_entry.get("sha256") == _sha256(log_raw)
    ):
        raise ValueError("The complete pytest log binding failed.")
    executable = Path(str(receipt.get("executable", ""))).resolve()
    command = receipt.get("command")
    if not executable.is_file() or not (
        command == [str(executable), "-m", "pytest", "-q"]
        and receipt.get("executable_sha256") == _sha256(executable.read_bytes())
    ):
        raise ValueError("The complete pytest command/runtime binding failed.")


def _build_development_parity_authorization(
    *,
    repo_root: Path,
    snapshot_path: Path,
    expected_v21e3_zip_sha256: str = OLD_V21E3_ZIP_SHA256,
) -> dict[str, object]:
    """Re-hash every live source/evidence byte and build the expected receipt."""

    root = repo_root.resolve()
    snapshot_path = _inside(root, snapshot_path)
    snapshot_raw, snapshot = _load_json(snapshot_path)
    entries = _verify_snapshot(
        root,
        snapshot,
        expected_v21e3_zip_sha256=expected_v21e3_zip_sha256,
    )

    structural_entry = _required_entry(
        entries,
        snapshot.get("target_size_input_structure_receipt_path"),
        "target-size input-structure receipt",
    )
    structural_path = _inside(root, root / str(structural_entry["path"]))
    _, structural = _load_json(structural_path)
    structural_bindings = _validate_structural_receipt(structural)

    execution_entry = _required_entry(
        entries,
        snapshot.get("target_size_execution_receipt_path"),
        "target-size three-arm execution receipt",
    )
    execution_path = _inside(root, root / str(execution_entry["path"]))
    _, execution_receipt = _load_json(execution_path)
    pytest_entry = _required_entry(
        entries, snapshot.get("full_pytest_receipt_path"), "full pytest receipt"
    )
    pytest_path = _inside(root, root / str(pytest_entry["path"]))
    _, pytest_receipt = _load_json(pytest_path)
    _validate_full_pytest_receipt(
        pytest_receipt,
        root=root,
        expected_prospective_source_root_sha256=snapshot.get(
            "prospective_source_root_sha256"
        ),
        expected_log_path=snapshot.get("full_pytest_log_path"),
        entries=entries,
    )

    output_bindings: dict[str, dict[str, object]] = {}
    for role in (
        "protocol",
        "case_manifest",
        "reference_manifest",
        "config_manifest",
        "metric_manifest",
    ):
        binding = structural_bindings[role]
        if not isinstance(binding, Mapping):
            raise ValueError(f"Malformed target-size binding: {role}")
        entry = _required_entry(entries, binding.get("path"), role)
        if not (
            binding.get("bytes") == entry["bytes"]
            and binding.get("sha256") == entry["sha256"]
        ):
            raise ValueError(f"Target-size and snapshot bindings disagree: {role}")
        output_bindings[role] = dict(entry)

    reference_path = _inside(
        root, root / str(output_bindings["reference_manifest"]["path"])
    )
    _, reference_manifest = _load_json(reference_path)
    reference_rows = reference_manifest.get("cases")
    if not isinstance(reference_rows, list):
        raise ValueError("The frozen reference manifest omits its cases.")
    expected_reference_cases = {
        str(case.get("case_id")): case
        for case in reference_rows
        if isinstance(case, Mapping)
    }
    if len(expected_reference_cases) != len(reference_rows):
        raise ValueError("The frozen reference manifest has duplicate case identifiers.")

    execution_child_root = _validate_target_execution_receipt(
        execution_receipt,
        root=root,
        prospective_source_root_sha256=snapshot.get(
            "prospective_source_root_sha256"
        ),
        expected_input_sha256={
            role: str(binding["sha256"])
            for role, binding in output_bindings.items()
        },
        expected_reference_cases=expected_reference_cases,
    )

    protocol_entry = output_bindings["protocol"]
    if protocol_entry["path"] != snapshot.get("protocol_path"):
        raise ValueError("The snapshot and structural receipt name another protocol.")
    protocol_path = _inside(root, root / str(protocol_entry["path"]))
    _, protocol = _load_json(protocol_path)
    common, design = _validate_protocol(protocol)
    if design.get("manifest") != output_bindings["case_manifest"]["path"]:
        raise ValueError("The protocol and authorization bind another case manifest.")

    trace_entry = _required_entry(
        entries, snapshot.get("trace_streaming_receipt_path"), "V6 streaming receipt"
    )
    trace_path = _inside(root, root / str(trace_entry["path"]))
    _, trace_receipt = _load_json(trace_path)
    if not (
        trace_receipt.get("schema")
        == "pareto_v21e3r1_trace_streaming_small_scale_receipt_v2"
        and trace_receipt.get("status")
        == "PASS_SMALL_SCALE_STREAMING_ENGINEERING_ONLY"
        and trace_receipt.get("target_scale_capacity_status") == "NOT_RUN"
        and trace_receipt.get("formal_authorized") is False
    ):
        raise ValueError("The bound V6 streaming receipt is not fail-closed.")

    receipt: dict[str, object] = {
        "schema": "pareto_v21e3r1_development_parity_authorization_v1",
        "status": "AUTHORIZED_DEVELOPMENT_PARITY_ONLY",
        "scientific_scope": "engineering_development_matrix_not_performance_evidence",
        "source_snapshot_path": snapshot_path.relative_to(root).as_posix(),
        "source_snapshot_receipt_sha256": _sha256(snapshot_raw),
        "source_snapshot_root_sha256": snapshot["bound_files_root_sha256"],
        "prospective_source_root_sha256": snapshot[
            "prospective_source_root_sha256"
        ],
        "target_size_input_structure_receipt_sha256": structural_entry[
            "sha256"
        ],
        "target_size_execution_receipt_sha256": execution_entry["sha256"],
        "target_size_execution_child_artifact_root_sha256": execution_child_root,
        "full_pytest_receipt_sha256": pytest_entry["sha256"],
        "bindings": output_bindings,
        "families": ["MOTSP", "MOKP"],
        "charged_evaluation_budget": common["charged_evaluation_budget"],
        "checkpoint_period": common["checkpoint_period"],
        "case_count": 12,
        "target_sizes": [100, 200, 500],
        "cases_per_size_per_family": 2,
        "seeds": list(design["seeds"]),
        "arms": ["V21E3_C0", "NSGAII", "MOEAD"],
        "development_parity_execution": "AUTHORIZED_DEVELOPMENT_ONLY",
        "matched_matrix": "AUTHORIZED_DEVELOPMENT_ONLY",
        "selection_entropy_release": "PROHIBITED",
        "calibration_execution": "PROHIBITED",
        "calibration_confirmation": "PROHIBITED",
        "formal_execution": "PROHIBITED",
        "formal_cases": "NOT_MATERIALIZED",
        "formal_authorized": False,
        "performance_claim": "NOT_ESTABLISHED",
        "submission_status": "IJOC_HOLD",
    }
    return receipt


def authorize_development_parity(
    *,
    repo_root: Path,
    snapshot_path: Path,
    output: Path,
    expected_v21e3_zip_sha256: str = OLD_V21E3_ZIP_SHA256,
) -> dict[str, object]:
    """Write one exclusive authorization after live byte verification."""

    root = repo_root.resolve()
    destination = _inside(root, output)
    if destination.exists():
        raise FileExistsError(f"Refusing to replace authorization: {destination}")
    receipt = _build_development_parity_authorization(
        repo_root=root,
        snapshot_path=snapshot_path,
        expected_v21e3_zip_sha256=expected_v21e3_zip_sha256,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(_canonical_bytes(receipt))
    return receipt


def verify_existing_development_parity_authorization(
    *,
    repo_root: Path,
    authorization_path: Path,
    expected_source_snapshot_root_sha256: str,
    expected_v21e3_zip_sha256: str = OLD_V21E3_ZIP_SHA256,
) -> dict[str, object]:
    """Fail closed unless an existing receipt still matches every live byte."""

    root = repo_root.resolve()
    resolved = _inside(root, authorization_path)
    _, observed = _load_json(resolved)
    snapshot_relative = str(observed.get("source_snapshot_path", ""))
    if not snapshot_relative:
        raise ValueError("The authorization omits its source snapshot path.")
    expected = _build_development_parity_authorization(
        repo_root=root,
        snapshot_path=_inside(root, root / snapshot_relative),
        expected_v21e3_zip_sha256=expected_v21e3_zip_sha256,
    )
    if observed != expected:
        raise ValueError("The authorization receipt is not the live verified receipt.")
    if observed.get("source_snapshot_root_sha256") != str(
        expected_source_snapshot_root_sha256
    ):
        raise ValueError("The authorization names another final source snapshot root.")
    return observed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = authorize_development_parity(
        repo_root=args.repo_root,
        snapshot_path=args.snapshot,
        output=args.output,
    )
    print(json.dumps(
        {
            "status": receipt["status"],
            "source_snapshot_root_sha256": receipt[
                "source_snapshot_root_sha256"
            ],
            "output": str(args.output.resolve()),
        },
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
