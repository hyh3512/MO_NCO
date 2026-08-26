from __future__ import annotations

"""Prospective V21e3r1 development-only matched-matrix runner.

The runner is deliberately fail closed.  It can execute only the frozen
12-case x 3-seed x 3-arm development product after an independent receipt has
authorized that exact product.  Selection, calibration, confirmation, and
formal evidence are never authorized by this program.
"""

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import shutil
import sqlite3
import sys
import time
from typing import Mapping, Sequence

from mo_nco.archive import ArchiveEntry, ParetoArchive
from mo_nco.pareto_ijoc_problem import problem_sha256
from mo_nco.pareto_v21e3_baselines import (
    frozen_development_baseline_configs,
    load_v21e3_development_problem,
    run_v21e3_development_baseline,
)
from mo_nco.pareto_v21e3_hybrid import (
    V21E3HybridConfig,
    V21E3TypedHybridParetoSearch,
)
from mo_nco.pareto_v21e3_parity import (
    analyze_development_parity,
    normalized_hypervolume_2d,
    normalized_left_continuous_auc,
)
from mo_nco.pareto_v21e3_trace_verify import (
    decode_v21e3_objectives_json,
    verify_v21e3_trace_database,
)


ARMS = ("V21E3_C0", "NSGAII", "MOEAD")
SEEDS = (31051, 31057, 31059)
FAMILIES = ("MOTSP", "MOKP")
SIZES = (100, 200, 500)
HEX = frozenset("0123456789abcdef")
ROW_ARTIFACT_PATH_SEMANTICS = "row_directory_relative_posix_v1"
MATRIX_ARTIFACT_PATH_SEMANTICS = "matrix_directory_relative_posix_v1"


@dataclass(frozen=True)
class FrozenCase:
    case_id: str
    family: str
    size: int
    instance_path: Path
    artifact_bytes: int
    artifact_sha256: str
    generator_problem_fingerprint_sha256: str
    lower: tuple[float, float]
    upper: tuple[float, float]


@dataclass(frozen=True)
class FrozenContract:
    cases: tuple[FrozenCase, ...]
    seeds: tuple[int, ...]
    arms: tuple[str, ...]
    budget: int
    checkpoint_period: int
    reference_directions: tuple[tuple[float, float], ...]
    source_snapshot_root_sha256: str
    input_sha256: Mapping[str, str]
    authorization_sha256: str | None
    authorization_path: Path | None
    live_authorization_verified: bool


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sqlite_read_only_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=ro"


def _require_sha256(value: object, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in HEX for char in text):
        raise ValueError(f"{field} must be lowercase SHA-256.")
    return text


def _load_json(path: str | Path, *, field: str) -> tuple[Path, dict[str, object]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is not UTF-8 JSON.") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must be a JSON object.")
    return resolved, payload


def _require_prohibitions(payload: Mapping[str, object], *, field: str) -> None:
    for name in (
        "selection_entropy_release",
        "calibration_execution",
        "formal_execution",
    ):
        if payload.get(name) != "PROHIBITED":
            raise ValueError(f"{field} does not keep {name} PROHIBITED.")
    if payload.get("formal_authorized") is not False:
        raise ValueError(f"{field} must set formal_authorized=false.")


def load_frozen_contract(
    *,
    case_manifest_path: str | Path,
    reference_manifest_path: str | Path,
    config_manifest_path: str | Path,
    metric_manifest_path: str | Path,
    protocol_path: str | Path,
    authorization_path: str | Path | None,
    source_snapshot_root_sha256: str,
    require_matrix_authorization: bool,
) -> FrozenContract:
    """Validate and materialize the exact prospective development contract."""

    source_root = _require_sha256(
        source_snapshot_root_sha256, "source_snapshot_root_sha256"
    )
    case_path, case_manifest = _load_json(
        case_manifest_path, field="case_manifest"
    )
    reference_path, reference = _load_json(
        reference_manifest_path, field="reference_manifest"
    )
    config_path, config = _load_json(config_manifest_path, field="config_manifest")
    metric_path, metric = _load_json(metric_manifest_path, field="metric_manifest")
    protocol_path_resolved, protocol = _load_json(protocol_path, field="protocol")
    expected_schemas = {
        "case_manifest": (
            case_manifest,
            "pareto_v21_partition_manifest_v1",
        ),
        "reference_manifest": (
            reference,
            "pareto_v21e3_analytic_reference_manifest_v1",
        ),
        "config_manifest": (
            config,
            "pareto_v21e3_development_config_manifest_v1",
        ),
        "metric_manifest": (metric, "pareto_v21e3_metric_manifest_v1"),
        "protocol": (protocol, "pareto_v21e3_c0_parity_protocol_v2"),
    }
    for field, (payload, schema) in expected_schemas.items():
        if payload.get("schema") != schema:
            raise ValueError(f"{field} has the wrong schema.")
    if (
        protocol.get("status")
        != "ENGINEERING_ADAPTERS_AVAILABLE_SUCCESSOR_SNAPSHOT_PENDING"
        or protocol.get("successor_version") != "V21e3r1"
        or protocol.get("families") != ["MOTSP", "MOKP"]
        or set(protocol.get("arms", {})) != set(ARMS)
    ):
        raise ValueError("The pending V21e3r1 protocol identity changed.")

    if (
        case_manifest.get("split") != "development"
        or case_manifest.get("role") != "prospective_algorithm_development"
        or case_manifest.get("formal_confirmatory_eligibility") is not False
    ):
        raise ValueError("The case manifest is not development-only.")
    if (
        reference.get("split") != "development"
        or reference.get("formal_use") != "NOT_AUTHORIZED"
        or reference.get("objective_sense") != "minimize"
        or reference.get("out_of_box_action") != "FAIL"
    ):
        raise ValueError("The reference manifest is not fail-closed development input.")
    if (
        config.get("selection_partition") != "NOT_GENERATED"
        or config.get("calibration_execution_authorized") is not False
        or config.get("formal_execution_authorized") is not False
    ):
        raise ValueError("The configuration manifest releases a later evidence phase.")
    if (
        metric.get("metric_id")
        != "normalized_left_continuous_hypervolume_auc_analytic_box_reference_1_1_v1"
        or metric.get("checkpoint_semantics")
        != "left_continuous_hv_zero_before_first_checkpoint"
        or metric.get("missing_checkpoint_action") != "FAIL"
        or metric.get("out_of_box_action")
        != "FAIL_BEFORE_SCALARIZATION_ARCHIVE_OR_METRIC"
        or metric.get("formal_use") != "NOT_AUTHORIZED"
    ):
        raise ValueError("The metric manifest changed the frozen estimand.")

    common = protocol.get("common_execution_contract")
    design = protocol.get("case_design")
    gates = protocol.get("preflight_gates")
    if not isinstance(common, dict) or not isinstance(design, dict) or not isinstance(gates, dict):
        raise ValueError("The parity protocol omits a required contract section.")
    budget = int(common.get("charged_evaluation_budget", -1))
    checkpoint = int(common.get("checkpoint_period", -1))
    if (
        budget != 2_000
        or checkpoint != 200
        or common.get("objective_call_semantics")
        != "first_true_objective_evaluation_v1"
        or common.get("archive_policy")
        != "unbounded_exact_nondominated_all_unique_evaluations_v1"
    ):
        raise ValueError("The protocol changed the common first-true budget.")
    selection_grid = metric.get("selection_grid")
    if not isinstance(selection_grid, dict) or selection_grid != {
        "charged_budget": budget,
        "checkpoint_period": checkpoint,
    }:
        raise ValueError("The metric and protocol checkpoint grids disagree.")
    if (
        tuple(design.get("seeds", ())) != SEEDS
        or tuple(design.get("sizes", ())) != SIZES
        or int(design.get("case_count_per_family", -1)) != 6
        or int(design.get("cases_per_size_per_family", -1)) != 2
    ):
        raise ValueError("The protocol no longer describes the frozen 12-case design.")
    _require_prohibitions(gates, field="protocol preflight gates")

    directions_raw = config.get("reference_directions")
    if not isinstance(directions_raw, list) or len(directions_raw) != 21:
        raise ValueError("The C0 reference-direction packet must contain 21 rows.")
    directions: list[tuple[float, float]] = []
    for row in directions_raw:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("A C0 reference direction is malformed.")
        direction = (float(row[0]), float(row[1]))
        if min(direction) <= 0.0 or abs(sum(direction) - 1.0) > 1e-12:
            raise ValueError("A C0 reference direction violates the frozen simplex.")
        directions.append(direction)
    source_binding = protocol.get("candidate_reference_directions")
    if not isinstance(source_binding, dict):
        raise ValueError("The protocol omits the C0 direction source binding.")
    source_packet = source_binding.get("source_binding")
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / str(design.get("manifest", ""))).resolve() != case_path:
        raise ValueError("The protocol names another development case manifest.")
    if (
        config.get("reference_manifest") != reference_path.name
        or config.get("metric_manifest") != metric_path.name
    ):
        raise ValueError("The configuration manifest names another metric/reference packet.")
    if (
        source_binding.get("count") != 21
        or not isinstance(source_packet, dict)
        or (repo_root / str(source_packet.get("path", ""))).resolve()
        != config_path
        or source_packet.get("bytes") != config_path.stat().st_size
        or source_packet.get("sha256") != _sha256(config_path)
    ):
        raise ValueError("The protocol does not bind the supplied config manifest.")

    partition_binding = reference.get("partition_manifest")
    if (
        not isinstance(partition_binding, dict)
        or (repo_root / str(partition_binding.get("path", ""))).resolve()
        != case_path
        or partition_binding.get("bytes") != case_path.stat().st_size
        or partition_binding.get("sha256") != _sha256(case_path)
    ):
        raise ValueError("The reference manifest does not bind the case manifest.")
    case_rows = case_manifest.get("cases")
    reference_rows = reference.get("cases")
    if (
        not isinstance(case_rows, list)
        or not isinstance(reference_rows, list)
        or len(case_rows) != 12
        or len(reference_rows) != 12
    ):
        raise ValueError("A development manifest omits case records.")
    manifest_by_id = {str(row.get("case_id")): row for row in case_rows}
    reference_by_id = {str(row.get("case_id")): row for row in reference_rows}
    if len(manifest_by_id) != 12 or set(manifest_by_id) != set(reference_by_id):
        raise ValueError("Case and reference manifests do not bind the same 12 cases.")

    cases: list[FrozenCase] = []
    observed_cells: dict[tuple[str, int], int] = {
        (family, size): 0 for family in FAMILIES for size in SIZES
    }
    for case_id in sorted(manifest_by_id):
        case = manifest_by_id[case_id]
        ref = reference_by_id[case_id]
        family = str(case.get("family"))
        size = int(case.get("size", -1))
        if (family, size) not in observed_cells:
            raise ValueError("A case lies outside the frozen family/size cells.")
        if family != str(ref.get("family")) or size != int(ref.get("size", -1)):
            raise ValueError("A reference row disagrees with its case row.")
        artifact = case.get("artifact")
        if not isinstance(artifact, dict):
            raise ValueError("A case omits its artifact binding.")
        instance_path = (case_path.parent / str(artifact.get("path"))).resolve()
        try:
            instance_path.relative_to(case_path.parent.resolve())
        except ValueError as error:
            raise ValueError(f"Case artifact escapes its partition: {case_id}.") from error
        artifact_hash = _require_sha256(artifact.get("sha256"), "case artifact hash")
        if (
            not instance_path.is_file()
            or instance_path.stat().st_size != int(artifact.get("bytes", -1))
            or _sha256(instance_path) != artifact_hash
            or ref.get("artifact_sha256") != artifact_hash
        ):
            raise ValueError(f"Case artifact binding failed for {case_id}.")
        problem = load_v21e3_development_problem(instance_path)
        semantic_hash = problem_sha256(problem)
        expected_problem_hash = _require_sha256(
            ref.get("problem_sha256"), "case problem hash"
        )
        fingerprints = case.get("fingerprints")
        if (
            not isinstance(fingerprints, dict)
            or fingerprints.get("problem_sha256") != expected_problem_hash
        ):
            raise ValueError(f"Generator problem binding failed for {case_id}.")
        # The generator fingerprint and runtime adapter semantic are distinct
        # typed hashes.  Loading plus hashing here ensures that the instance is
        # accepted by the deployed adapter; the v2 run context records the
        # adapter semantic independently of the generator fingerprint.
        _require_sha256(semantic_hash, "runtime problem semantic hash")
        lower = tuple(float(value) for value in ref.get("objective_lower_bounds", ()))
        upper = tuple(float(value) for value in ref.get("objective_upper_bounds", ()))
        if len(lower) != 2 or len(upper) != 2 or any(
            lo >= hi for lo, hi in zip(lower, upper)
        ):
            raise ValueError(f"Analytic objective box is invalid for {case_id}.")
        runtime_lower = tuple(float(value) for value in problem.objective_lower_bounds)
        runtime_upper = tuple(float(value) for value in problem.objective_upper_bounds)
        if runtime_lower != lower or runtime_upper != upper:
            raise ValueError(
                f"Runtime adapter and reference analytic boxes differ for {case_id}."
            )
        if (
            ref.get("normalized_reference_point") != [1.0, 1.0]
            or ref.get("regime") != case.get("regime")
            or problem.name != case_id
        ):
            raise ValueError(f"Reference identity changed for {case_id}.")
        reference_core = dict(ref)
        reference_digest = reference_core.pop("reference_packet_sha256", None)
        if reference_digest != hashlib.sha256(
            _canonical_bytes(reference_core) + b"\n"
        ).hexdigest():
            raise ValueError(f"Reference packet digest failed for {case_id}.")
        observed_cells[(family, size)] += 1
        cases.append(
            FrozenCase(
                case_id=case_id,
                family=family,
                size=size,
                instance_path=instance_path,
                artifact_bytes=instance_path.stat().st_size,
                artifact_sha256=artifact_hash,
                generator_problem_fingerprint_sha256=expected_problem_hash,
                lower=(lower[0], lower[1]),
                upper=(upper[0], upper[1]),
            )
        )
    if any(count != 2 for count in observed_cells.values()):
        raise ValueError("The case manifest is not two cases per frozen cell.")

    input_hashes = {
        "case_manifest": _sha256(case_path),
        "reference_manifest": _sha256(reference_path),
        "config_manifest": _sha256(config_path),
        "metric_manifest": _sha256(metric_path),
        "protocol": _sha256(protocol_path_resolved),
    }
    input_paths = {
        "case_manifest": case_path,
        "reference_manifest": reference_path,
        "config_manifest": config_path,
        "metric_manifest": metric_path,
        "protocol": protocol_path_resolved,
    }
    authorization_hash: str | None = None
    resolved_authorization_path: Path | None = None
    live_authorization_verified = False
    if require_matrix_authorization:
        if authorization_path is None:
            raise ValueError("The matched matrix requires an authorization receipt.")
        auth_path, authorization = _load_json(
            authorization_path, field="authorization_receipt"
        )
        try:
            from ijoc_submission_v21e3r1.scripts.preflight_v21e3r1_development_parity import (
                verify_existing_development_parity_authorization,
            )
        except ModuleNotFoundError:  # Direct ``python path/to/script.py`` execution.
            from preflight_v21e3r1_development_parity import (  # type: ignore[no-redef]
                verify_existing_development_parity_authorization,
            )

        verify_existing_development_parity_authorization(
            repo_root=Path(__file__).resolve().parents[2],
            authorization_path=auth_path,
            expected_source_snapshot_root_sha256=source_root,
        )
        live_authorization_verified = True
        resolved_authorization_path = auth_path
        if (
            authorization.get("schema")
            != "pareto_v21e3r1_development_parity_authorization_v1"
            or authorization.get("status")
            != "AUTHORIZED_DEVELOPMENT_PARITY_ONLY"
            or authorization.get("development_parity_execution")
            != "AUTHORIZED_DEVELOPMENT_ONLY"
            or authorization.get("source_snapshot_root_sha256") != source_root
        ):
            raise ValueError("The independent receipt does not authorize this matrix.")
        _require_prohibitions(authorization, field="authorization receipt")
        auth_bindings = authorization.get("bindings")
        if not isinstance(auth_bindings, dict) or set(auth_bindings) != set(input_paths):
            raise ValueError("Authorization receipt has incomplete input bindings.")
        repo_root = Path(__file__).resolve().parents[2]
        for role, path in input_paths.items():
            binding = auth_bindings.get(role)
            if not isinstance(binding, dict):
                raise ValueError(f"Authorization receipt omits {role}.")
            bound_path = (repo_root / str(binding.get("path", ""))).resolve()
            if (
                bound_path != path
                or binding.get("bytes") != path.stat().st_size
                or binding.get("sha256") != _sha256(path)
            ):
                raise ValueError(f"Authorization receipt binding failed for {role}.")
        authorization_hash = _sha256(auth_path)
    elif authorization_path is not None:
        raise ValueError("Structural preflight must not consume matrix authorization.")

    return FrozenContract(
        cases=tuple(cases),
        seeds=SEEDS,
        arms=ARMS,
        budget=budget,
        checkpoint_period=checkpoint,
        reference_directions=tuple(directions),
        source_snapshot_root_sha256=source_root,
        input_sha256=input_hashes,
        authorization_sha256=authorization_hash,
        authorization_path=resolved_authorization_path,
        live_authorization_verified=live_authorization_verified,
    )


def _exclusive_write_json(path: Path, payload: object) -> str:
    raw = _canonical_bytes(payload) + b"\n"
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(raw).hexdigest()


def _read_and_validate_v2_context(
    database_path: Path,
    *,
    case: FrozenCase,
    problem_semantic_sha256: str,
    arm_id: str,
    seed: int,
    budget: int,
    source_snapshot_root_sha256: str,
) -> dict[str, object]:
    connection = sqlite3.connect(_sqlite_read_only_uri(database_path), uri=True)
    try:
        row = connection.execute(
            "SELECT run_context_json FROM run_attempt WHERE run_id=1"
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise RuntimeError("The row trace omits its v2 run context.")
    context = json.loads(str(row[0]))
    expected = {
        "schema": "v21e3r1_run_context_v2",
        "case_artifact_sha256": case.artifact_sha256,
        "problem_semantic_sha256": problem_semantic_sha256,
        "candidate_id": ("C0" if arm_id == "V21E3_C0" else arm_id),
        "algorithm_source_sha256": source_snapshot_root_sha256,
        "seed": seed,
        "charged_evaluation_budget": budget,
        "evidence_partition": "development",
    }
    for field, value in expected.items():
        if context.get(field) != value:
            raise RuntimeError(f"The row v2 context disagrees on {field}.")
    algorithm_config = context.get("algorithm_config")
    if not isinstance(algorithm_config, dict):
        raise RuntimeError("The row v2 context omits its algorithm config.")
    config_raw = _canonical_bytes(algorithm_config)
    if context.get("candidate_config_sha256") != hashlib.sha256(config_raw).hexdigest():
        raise RuntimeError("The row v2 context has a stale config digest.")
    mirrors = {
        "candidate_id": "candidate_id",
        "seed": "seed",
        "charged_evaluation_budget": "charged_evaluations",
        "evidence_partition": "phase",
        "reference_directions": "reference_directions",
    }
    for top_level, config_field in mirrors.items():
        if context.get(top_level) != algorithm_config.get(config_field):
            raise RuntimeError(f"The row v2 context has a contradictory {top_level} mirror.")
    return context


def _seal_sqlite(database_path: Path) -> None:
    """Checkpoint WAL bytes into one portable database artifact."""

    connection = sqlite3.connect(database_path)
    try:
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is None or int(checkpoint[0]) != 0:
            raise RuntimeError("The row SQLite WAL checkpoint did not complete.")
        mode = str(connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0])
        if mode.lower() != "delete":
            raise RuntimeError("The row SQLite artifact did not enter DELETE mode.")
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("The sealed row SQLite artifact failed integrity_check.")
        connection.commit()
    finally:
        connection.close()
    for suffix in ("-wal", "-shm"):
        if Path(str(database_path) + suffix).exists():
            raise RuntimeError("A sealed row still depends on SQLite sidecar state.")


def _normalize_objective_replay_paths(
    payload: Mapping[str, object],
    *,
    database_path: Path,
    terminal_path: Path,
) -> dict[str, object]:
    normalized = dict(payload)
    expected = {
        "database_path": (database_path.resolve(), "trace.sqlite3"),
        "detached_terminal_receipt_path": (
            terminal_path.resolve(),
            "terminal.receipt.json",
        ),
    }
    for field, (actual, relative) in expected.items():
        observed = normalized.get(field)
        if observed != relative and Path(str(observed)).resolve() != actual:
            raise RuntimeError(f"Objective replay returned another {field}.")
        normalized[field] = relative
    return normalized


def _normalize_metric_replay_paths(
    payload: Mapping[str, object],
    *,
    database_path: Path,
) -> dict[str, object]:
    normalized = dict(payload)
    observed = normalized.get("database_path")
    if observed != "trace.sqlite3" and Path(str(observed)).resolve() != database_path.resolve():
        raise RuntimeError("Metric replay returned another database_path.")
    normalized["database_path"] = "trace.sqlite3"
    return normalized


def _require_portable_row_artifact_path(
    payload: Mapping[str, object],
    *,
    field: str,
    expected: str,
    row_directory: Path,
) -> Path:
    observed = payload.get(field)
    if not isinstance(observed, str):
        raise RuntimeError(f"Strict resume artifact path {field} is not text.")
    posix_path = PurePosixPath(observed)
    if (
        observed != posix_path.as_posix()
        or "\\" in observed
        or posix_path.is_absolute()
        or PureWindowsPath(observed).is_absolute()
        or any(part in {"", ".", ".."} for part in posix_path.parts)
        or observed != expected
    ):
        raise RuntimeError(f"Strict resume artifact path {field} is not portable.")
    resolved_root = row_directory.resolve()
    resolved = (resolved_root / Path(*posix_path.parts)).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise RuntimeError(
            f"Strict resume artifact path {field} escapes its row directory."
        ) from error
    return resolved


def replay_normalized_metric_from_trace(
    database_path: str | Path,
    *,
    budget: int,
    checkpoint_period: int,
    lower: Sequence[float],
    upper: Sequence[float],
    metric_manifest_sha256: str,
) -> dict[str, object]:
    """Derive the frozen HV-AUC estimand only from the sealed objective ledger."""

    path = Path(database_path).resolve()
    metric_hash = _require_sha256(metric_manifest_sha256, "metric_manifest_sha256")
    if budget <= 0 or checkpoint_period <= 0 or budget % checkpoint_period:
        raise ValueError("Metric replay requires a positive divisible budget grid.")
    connection = sqlite3.connect(_sqlite_read_only_uri(path), uri=True)
    try:
        rows = list(
            connection.execute(
                "SELECT evaluation_index,objectives_json "
                "FROM evaluations ORDER BY evaluation_index"
            )
        )
    finally:
        connection.close()
    if len(rows) != budget:
        raise RuntimeError("Metric replay found another charged evaluation count.")

    archive = ParetoArchive(max_size=None, tol=0.0)
    checkpoints: list[dict[str, object]] = []
    previous_evaluation = 0
    previous_hv = 0.0
    area = 0.0
    for expected_index, raw_row in enumerate(rows, start=1):
        evaluation_index = int(raw_row[0])
        if evaluation_index != expected_index:
            raise RuntimeError("Metric replay evaluation indices are not contiguous.")
        objectives = decode_v21e3_objectives_json(
            str(raw_row[1]), expected_dimension=2
        )
        # Validate every evaluated point against the frozen analytic box, even
        # if it becomes dominated and therefore leaves the replayed archive.
        normalized_hypervolume_2d((objectives,), lower=lower, upper=upper)
        archive.update((ArchiveEntry((evaluation_index,), objectives),))
        if evaluation_index % checkpoint_period:
            continue
        area += previous_hv * (evaluation_index - previous_evaluation)
        current_hv = normalized_hypervolume_2d(
            tuple(entry.objectives for entry in archive.entries),
            lower=lower,
            upper=upper,
        )
        checkpoints.append(
            {
                "evaluation": evaluation_index,
                "normalized_hv": current_hv,
                "archive_size": len(archive),
            }
        )
        previous_evaluation = evaluation_index
        previous_hv = current_hv
    expected_grid = list(range(checkpoint_period, budget + 1, checkpoint_period))
    if [int(item["evaluation"]) for item in checkpoints] != expected_grid:
        raise RuntimeError("Metric replay omitted a frozen checkpoint.")
    return {
        "schema": "pareto_v21e3r1_metric_replay_receipt_v1",
        "status": "NORMALIZED_HV_AUC_REPLAY_PASS",
        "verification_scope": "frozen_metric_from_objective_ledger_checkpoints_v1",
        "database_path": str(path),
        "database_sha256": _sha256(path),
        "metric_manifest_sha256": metric_hash,
        "charged_evaluation_budget": budget,
        "checkpoint_period": checkpoint_period,
        "analytic_box": {
            "lower": [float(value) for value in lower],
            "upper": [float(value) for value in upper],
            "normalized_reference": [1.0, 1.0],
        },
        "normalized_left_continuous_hv_auc": area / float(budget),
        "normalized_terminal_hv": previous_hv,
        "checkpoints": checkpoints,
        "selection_authorization": "PROHIBITED",
        "formal_authorized": False,
    }


def _load_live_bound_problem(case: FrozenCase):
    """Recheck instance bytes immediately before each execution or resume replay."""

    if not case.instance_path.is_file() or not (
        case.instance_path.stat().st_size == case.artifact_bytes
        and _sha256(case.instance_path) == case.artifact_sha256
    ):
        raise RuntimeError(f"The live case artifact drifted: {case.case_id}")
    problem = load_v21e3_development_problem(case.instance_path)
    if not (
        problem.name == case.case_id
        and tuple(float(value) for value in problem.objective_lower_bounds) == case.lower
        and tuple(float(value) for value in problem.objective_upper_bounds) == case.upper
    ):
        raise RuntimeError(f"The live runtime-adapter bridge drifted: {case.case_id}")
    return problem


def execute_row(
    *,
    case: FrozenCase,
    arm_id: str,
    seed: int,
    budget: int,
    checkpoint_period: int,
    reference_directions: Sequence[Sequence[float]],
    source_snapshot_root_sha256: str,
    row_directory: str | Path,
    metric_manifest_sha256: str,
    scientific_scope: str,
) -> dict[str, object]:
    """Execute, persist, independently replay, and commit one row last."""

    if arm_id not in ARMS:
        raise ValueError("Unknown matched-matrix arm.")
    source_root = _require_sha256(
        source_snapshot_root_sha256, "source_snapshot_root_sha256"
    )
    if seed not in SEEDS:
        raise ValueError("The row seed is outside the frozen seed packet.")
    if budget <= 0 or checkpoint_period <= 0 or budget % checkpoint_period:
        raise ValueError("The row budget must be a positive checkpoint multiple.")
    output = Path(row_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    database_path = output / "trace.sqlite3"
    terminal_path = output / "terminal.receipt.json"
    problem = _load_live_bound_problem(case)
    semantic_hash = problem_sha256(problem)
    started = time.perf_counter_ns()
    if arm_id == "V21E3_C0":
        config = V21E3HybridConfig(
            candidate_id="C0",
            reference_directions=tuple(
                tuple(float(value) for value in direction)
                for direction in reference_directions
            ),
            charged_evaluations=budget,
            checkpoint_period=checkpoint_period,
            seed=seed,
            phase="development",
            trace_database=str(database_path),
            terminal_receipt=str(terminal_path),
            receipt_database_path="trace.sqlite3",
            capture_trace=False,
            case_artifact_sha256=case.artifact_sha256,
            source_snapshot_sha256=source_root,
        )
        run = V21E3TypedHybridParetoSearch(problem, config).run()
    else:
        configs = frozen_development_baseline_configs(
            family=case.family,
            charged_evaluations=budget,
            checkpoint_period=checkpoint_period,
            seed=seed,
        )
        config = replace(
            configs[arm_id],
            trace_database=str(database_path),
            terminal_receipt=str(terminal_path),
            receipt_database_path="trace.sqlite3",
            capture_trace=False,
            case_artifact_sha256=case.artifact_sha256,
            source_snapshot_sha256=source_root,
        )
        run = run_v21e3_development_baseline(problem, config)
    elapsed_ns = time.perf_counter_ns() - started
    result = run.optimization_result
    _seal_sqlite(database_path)
    context = _read_and_validate_v2_context(
        database_path,
        case=case,
        problem_semantic_sha256=semantic_hash,
        arm_id=arm_id,
        seed=seed,
        budget=budget,
        source_snapshot_root_sha256=source_root,
    )
    detached_hash = _sha256(terminal_path)
    preverification = {
        "schema": "pareto_v21e3r1_row_preverification_v1",
        "status": "READY_FOR_OBJECTIVE_AND_ARCHIVE_REPLAY",
        "case_id": case.case_id,
        "family": case.family,
        "size": case.size,
        "seed": seed,
        "arm_id": arm_id,
        "source_snapshot_root_sha256": source_root,
        "trace_database_path": "trace.sqlite3",
        "detached_terminal_receipt_path": "terminal.receipt.json",
        "detached_terminal_receipt_sha256": detached_hash,
        "selection_entropy_release": "PROHIBITED",
        "calibration_execution": "PROHIBITED",
        "formal_execution": "PROHIBITED",
        "formal_authorized": False,
    }
    preverification_path = output / "row.preverification.json"
    preverification_sha256 = _exclusive_write_json(
        preverification_path, preverification
    )
    replay = _normalize_objective_replay_paths(
        verify_v21e3_trace_database(
            database_path,
            problem,
            expected_run_context=context,
            detached_terminal_receipt_path=terminal_path,
            expected_detached_terminal_receipt_sha256=detached_hash,
            expected_charged_evaluations=budget,
        ),
        database_path=database_path,
        terminal_path=terminal_path,
    )
    replay_path = output / "objective_archive_replay.receipt.json"
    replay_sha256 = _exclusive_write_json(replay_path, replay)
    diagnostic_auc, diagnostic_terminal_hv, diagnostic_checkpoints = (
        normalized_left_continuous_auc(
        result.diagnostics,
        budget=budget,
        checkpoint_period=checkpoint_period,
        lower=case.lower,
        upper=case.upper,
        )
    )
    metric_replay = _normalize_metric_replay_paths(
        replay_normalized_metric_from_trace(
            database_path,
            budget=budget,
            checkpoint_period=checkpoint_period,
            lower=case.lower,
            upper=case.upper,
            metric_manifest_sha256=metric_manifest_sha256,
        ),
        database_path=database_path,
    )
    diagnostic_metric = {
        "normalized_left_continuous_hv_auc": diagnostic_auc,
        "normalized_terminal_hv": diagnostic_terminal_hv,
        "checkpoints": list(diagnostic_checkpoints),
    }
    replayed_metric = {
        field: metric_replay[field]
        for field in (
            "normalized_left_continuous_hv_auc",
            "normalized_terminal_hv",
            "checkpoints",
        )
    }
    if diagnostic_metric != replayed_metric:
        raise RuntimeError("In-memory diagnostics disagree with objective-ledger metric replay.")
    metric_replay_path = output / "metric_replay.receipt.json"
    metric_replay_sha256 = _exclusive_write_json(metric_replay_path, metric_replay)
    row = {
        "schema": "pareto_v21e3r1_matched_matrix_row_v1",
        "status": "PASS_ENGINEERING_ROW_OBJECTIVE_ARCHIVE_REPLAYED",
        "scientific_scope": scientific_scope,
        "artifact_path_semantics": ROW_ARTIFACT_PATH_SEMANTICS,
        "case_id": case.case_id,
        "family": case.family,
        "size": case.size,
        "case_artifact_sha256": case.artifact_sha256,
        "generator_problem_fingerprint_sha256": (
            case.generator_problem_fingerprint_sha256
        ),
        "runtime_problem_semantic_sha256": semantic_hash,
        "seed": seed,
        "arm_id": arm_id,
        "charged_evaluation_budget": budget,
        "checkpoint_period": checkpoint_period,
        "normalized_left_continuous_hv_auc": metric_replay[
            "normalized_left_continuous_hv_auc"
        ],
        "normalized_terminal_hv": metric_replay["normalized_terminal_hv"],
        "checkpoints": metric_replay["checkpoints"],
        "elapsed_process_wall_ns_diagnostic_only": elapsed_ns,
        "source_snapshot_root_sha256": source_root,
        "run_context_digest_sha256": replay["run_context_digest_sha256"],
        "trace_database_path": "trace.sqlite3",
        "trace_database_bytes": database_path.stat().st_size,
        "trace_database_sha256": replay["database_sha256"],
        "detached_terminal_receipt_path": "terminal.receipt.json",
        "detached_terminal_receipt_sha256": detached_hash,
        "preverification_receipt_path": "row.preverification.json",
        "preverification_receipt_sha256": preverification_sha256,
        "objective_archive_replay_receipt_path": (
            "objective_archive_replay.receipt.json"
        ),
        "objective_archive_replay_receipt_sha256": replay_sha256,
        "metric_replay_receipt_path": "metric_replay.receipt.json",
        "metric_replay_receipt_sha256": metric_replay_sha256,
        "metric_manifest_sha256": _require_sha256(
            metric_manifest_sha256, "metric_manifest_sha256"
        ),
        "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
        "runtime_efficiency_claim_authorized": False,
        "selection_entropy_release": "PROHIBITED",
        "calibration_execution": "PROHIBITED",
        "formal_execution": "PROHIBITED",
        "formal_authorized": False,
    }
    _exclusive_write_json(output / "row.json", row)
    return row


def _matrix_plan(contract: FrozenContract) -> dict[str, object]:
    rows = []
    for case in contract.cases:
        for seed in contract.seeds:
            for arm_id in contract.arms:
                rows.append(
                    {
                        "case_id": case.case_id,
                        "family": case.family,
                        "size": case.size,
                        "seed": seed,
                        "arm_id": arm_id,
                        "row_slug": f"{case.case_id}__seed-{seed}__arm-{arm_id.lower()}",
                    }
                )
    return {
        "schema": "pareto_v21e3r1_development_matched_matrix_plan_v1",
        "status": "AUTHORIZED_DEVELOPMENT_MATRIX_PLAN",
        "scientific_scope": "authors_generated_development_only_not_formal_evidence",
        "expected_rows": 108,
        "budget": contract.budget,
        "checkpoint_period": contract.checkpoint_period,
        "source_snapshot_root_sha256": contract.source_snapshot_root_sha256,
        "authorization_receipt_sha256": contract.authorization_sha256,
        "input_sha256": dict(contract.input_sha256),
        "rows": rows,
        "selection_entropy_release": "PROHIBITED",
        "calibration_execution": "PROHIBITED",
        "formal_execution": "PROHIBITED",
        "formal_authorized": False,
    }


def _load_completed_row(
    row_directory: Path,
    *,
    planned: Mapping[str, object],
    contract: FrozenContract,
) -> dict[str, object]:
    row_path = row_directory / "row.json"
    if not row_path.is_file():
        raise RuntimeError(f"Strict resume found a partial row: {row_directory.name}")
    expected_names = {
        "trace.sqlite3",
        "terminal.receipt.json",
        "row.preverification.json",
        "objective_archive_replay.receipt.json",
        "metric_replay.receipt.json",
        "row.json",
    }
    observed_names = {path.name for path in row_directory.iterdir()}
    if observed_names != expected_names or any(
        not path.is_file() for path in row_directory.iterdir()
    ):
        raise RuntimeError(f"Strict resume found unbound row artifacts: {row_directory.name}")
    row = json.loads(row_path.read_text(encoding="utf-8"))
    if (
        row.get("schema") != "pareto_v21e3r1_matched_matrix_row_v1"
        or row.get("status") != "PASS_ENGINEERING_ROW_OBJECTIVE_ARCHIVE_REPLAYED"
        or row.get("scientific_scope")
        != "authors_generated_development_only_not_formal_evidence"
        or row.get("artifact_path_semantics")
        != ROW_ARTIFACT_PATH_SEMANTICS
        or row.get("source_snapshot_root_sha256")
        != contract.source_snapshot_root_sha256
        or row.get("charged_evaluation_budget") != contract.budget
        or row.get("checkpoint_period") != contract.checkpoint_period
    ):
        raise RuntimeError(f"Strict resume rejected row receipt: {row_directory.name}")
    for field in ("case_id", "family", "size", "seed", "arm_id"):
        if row.get(field) != planned.get(field):
            raise RuntimeError(f"Strict resume row key mismatch: {row_directory.name}")
    _require_prohibitions(row, field="completed row")
    database_path = row_directory / "trace.sqlite3"
    terminal_path = row_directory / "terminal.receipt.json"
    pre_path = row_directory / "row.preverification.json"
    replay_path = row_directory / "objective_archive_replay.receipt.json"
    metric_replay_path = row_directory / "metric_replay.receipt.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    stored_metric_replay = json.loads(
        metric_replay_path.read_text(encoding="utf-8")
    )
    receipts = (row, terminal, pre, replay, stored_metric_replay)
    if any(not isinstance(receipt, dict) for receipt in receipts):
        raise RuntimeError(f"Strict resume found a non-object receipt: {row_directory.name}")
    portable_paths = (
        (row, "trace_database_path", "trace.sqlite3"),
        (row, "detached_terminal_receipt_path", "terminal.receipt.json"),
        (row, "preverification_receipt_path", "row.preverification.json"),
        (
            row,
            "objective_archive_replay_receipt_path",
            "objective_archive_replay.receipt.json",
        ),
        (row, "metric_replay_receipt_path", "metric_replay.receipt.json"),
        (terminal, "database_path", "trace.sqlite3"),
        (pre, "trace_database_path", "trace.sqlite3"),
        (pre, "detached_terminal_receipt_path", "terminal.receipt.json"),
        (replay, "database_path", "trace.sqlite3"),
        (replay, "detached_terminal_receipt_path", "terminal.receipt.json"),
        (stored_metric_replay, "database_path", "trace.sqlite3"),
    )
    for payload, field, expected in portable_paths:
        _require_portable_row_artifact_path(
            payload,
            field=field,
            expected=expected,
            row_directory=row_directory,
        )
    if (
        row.get("trace_database_sha256") != _sha256(database_path)
        or row.get("detached_terminal_receipt_sha256") != _sha256(terminal_path)
        or row.get("preverification_receipt_sha256") != _sha256(pre_path)
        or row.get("objective_archive_replay_receipt_sha256") != _sha256(replay_path)
        or row.get("metric_replay_receipt_sha256") != _sha256(metric_replay_path)
        or row.get("metric_manifest_sha256")
        != contract.input_sha256["metric_manifest"]
    ):
        raise RuntimeError(f"Strict resume artifact hash mismatch: {row_directory.name}")
    if (
        pre.get("detached_terminal_receipt_sha256")
        != row.get("detached_terminal_receipt_sha256")
        or replay.get("status") != "OBJECTIVE_AND_ARCHIVE_REPLAY_PASS"
        or replay.get("database_sha256") != row.get("trace_database_sha256")
        or replay.get("detached_terminal_receipt_sha256")
        != row.get("detached_terminal_receipt_sha256")
    ):
        raise RuntimeError(f"Strict resume replay binding mismatch: {row_directory.name}")
    if (
        pre_path.stat().st_mtime_ns > replay_path.stat().st_mtime_ns
        or replay_path.stat().st_mtime_ns > metric_replay_path.stat().st_mtime_ns
        or row_path.stat().st_mtime_ns < metric_replay_path.stat().st_mtime_ns
    ):
        raise RuntimeError(f"Strict resume row-last ordering failed: {row_directory.name}")
    case = next(
        (item for item in contract.cases if item.case_id == planned["case_id"]),
        None,
    )
    if case is None:
        raise RuntimeError("Strict resume could not resolve the planned case.")
    problem = _load_live_bound_problem(case)
    context = _read_and_validate_v2_context(
        database_path,
        case=case,
        problem_semantic_sha256=problem_sha256(problem),
        arm_id=str(planned["arm_id"]),
        seed=int(planned["seed"]),
        budget=contract.budget,
        source_snapshot_root_sha256=contract.source_snapshot_root_sha256,
    )
    independently_replayed = _normalize_objective_replay_paths(
        verify_v21e3_trace_database(
            database_path,
            problem,
            expected_run_context=context,
            detached_terminal_receipt_path=terminal_path,
            expected_detached_terminal_receipt_sha256=str(
                pre["detached_terminal_receipt_sha256"]
            ),
            expected_charged_evaluations=contract.budget,
        ),
        database_path=database_path,
        terminal_path=terminal_path,
    )
    if independently_replayed != replay:
        raise RuntimeError(f"Strict resume reproduced another replay: {row_directory.name}")
    independently_replayed_metric = _normalize_metric_replay_paths(
        replay_normalized_metric_from_trace(
            database_path,
            budget=contract.budget,
            checkpoint_period=contract.checkpoint_period,
            lower=case.lower,
            upper=case.upper,
            metric_manifest_sha256=contract.input_sha256["metric_manifest"],
        ),
        database_path=database_path,
    )
    if independently_replayed_metric != stored_metric_replay:
        raise RuntimeError(
            f"Strict resume reproduced another metric replay: {row_directory.name}"
        )
    for field in (
        "normalized_left_continuous_hv_auc",
        "normalized_terminal_hv",
        "checkpoints",
    ):
        if row.get(field) != stored_metric_replay.get(field):
            raise RuntimeError(
                f"Strict resume metric replay mismatch: {row_directory.name}"
            )
    return row


def initialize_matrix_output(
    output_directory: str | Path,
    *,
    contract: FrozenContract,
    resume: bool,
) -> dict[str, object]:
    """Create an exclusive plan or strictly validate an existing checkpoint."""

    if contract.authorization_sha256 is None:
        raise ValueError("A matched matrix cannot start without authorization.")
    output = Path(output_directory).resolve()
    expected_plan = _matrix_plan(contract)
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        (output / "rows").mkdir()
        _exclusive_write_json(output / "matrix.plan.json", expected_plan)
        return expected_plan
    if not output.is_dir():
        raise FileNotFoundError(output)
    plan_path = output / "matrix.plan.json"
    if not plan_path.is_file():
        raise RuntimeError("Strict resume requires the original matrix.plan.json.")
    observed_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if observed_plan != expected_plan:
        raise RuntimeError("Strict resume rejected a changed matrix plan.")
    rows_root = output / "rows"
    if not rows_root.is_dir():
        raise RuntimeError("Strict resume requires the original rows directory.")
    expected_by_slug = {
        str(row["row_slug"]): row for row in expected_plan["rows"]
    }
    for row_directory in rows_root.iterdir():
        if not row_directory.is_dir() or row_directory.name not in expected_by_slug:
            raise RuntimeError("Strict resume found an unplanned rows entry.")
        _load_completed_row(
            row_directory,
            planned=expected_by_slug[row_directory.name],
            contract=contract,
        )
    return expected_plan


def _collect_complete_rows(
    output: Path,
    *,
    contract: FrozenContract,
    plan: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for planned in plan["rows"]:
        row_directory = output / "rows" / str(planned["row_slug"])
        row = _load_completed_row(
            row_directory, planned=planned, contract=contract
        )
        copied = dict(row)
        copied["row_receipt_sha256"] = _sha256(row_directory / "row.json")
        copied["row_receipt_path"] = (row_directory / "row.json").relative_to(
            output.resolve()
        ).as_posix()
        rows.append(copied)
    return rows


def finalize_matrix_output(
    output_directory: str | Path,
    *,
    contract: FrozenContract,
    plan: Mapping[str, object],
) -> dict[str, object]:
    """Analyze the complete matrix and write aggregate then audit last."""

    output = Path(output_directory).resolve()
    aggregate_path = output / "matrix.aggregate.json"
    audit_path = output / "post_run_audit.json"
    if aggregate_path.exists() or audit_path.exists():
        raise FileExistsError("Final matrix receipts already exist.")
    rows = _collect_complete_rows(output, contract=contract, plan=plan)
    case_records = [
        {"case_id": case.case_id, "family": case.family, "size": case.size}
        for case in contract.cases
    ]
    analysis = analyze_development_parity(
        rows,
        case_records=case_records,
        seeds=contract.seeds,
        margin=0.005,
        size_stratum_margin=0.010,
        bootstrap_samples=20_000,
        bootstrap_seed=31_061,
        tie_tolerance=1e-12,
    )
    aggregate_rows = [
        {
            key: row[key]
            for key in (
                "case_id",
                "family",
                "size",
                "seed",
                "arm_id",
                "normalized_left_continuous_hv_auc",
                "normalized_terminal_hv",
                "source_snapshot_root_sha256",
                "trace_database_sha256",
                "detached_terminal_receipt_sha256",
                "objective_archive_replay_receipt_sha256",
                "metric_replay_receipt_sha256",
                "row_receipt_path",
                "row_receipt_sha256",
            )
        }
        for row in rows
    ]
    aggregate = {
        "schema": "pareto_v21e3r1_development_matched_matrix_aggregate_v1",
        "status": "COMPLETE_DEVELOPMENT_MATRIX_ENGINEERING_EVIDENCE",
        "scientific_scope": "authors_generated_development_only_not_formal_evidence",
        "artifact_path_semantics": MATRIX_ARTIFACT_PATH_SEMANTICS,
        "source_snapshot_root_sha256": contract.source_snapshot_root_sha256,
        "authorization_receipt_sha256": contract.authorization_sha256,
        "matrix_plan_sha256": _sha256(output / "matrix.plan.json"),
        "input_sha256": dict(contract.input_sha256),
        "expected_rows": 108,
        "observed_rows": len(rows),
        "rows": aggregate_rows,
        "analysis": analysis,
        "runtime_efficiency_claim_authorized": False,
        "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
        "selection_entropy_release": "PROHIBITED",
        "calibration_execution": "PROHIBITED",
        "formal_execution": "PROHIBITED",
        "formal_authorized": False,
    }
    aggregate_sha256 = _exclusive_write_json(aggregate_path, aggregate)
    all_replayed = all(
        row["status"] == "PASS_ENGINEERING_ROW_OBJECTIVE_ARCHIVE_REPLAYED"
        for row in rows
    )
    audit = {
        "schema": "pareto_v21e3r1_development_matrix_post_run_audit_v1",
        "status": (
            "PASS_COMPLETE_DEVELOPMENT_MATRIX_AUDITED"
            if all_replayed and len(rows) == 108
            else "FAIL_INCOMPLETE_OR_UNVERIFIED_MATRIX"
        ),
        "matrix_aggregate_path": "matrix.aggregate.json",
        "matrix_aggregate_sha256": aggregate_sha256,
        "source_snapshot_root_sha256": contract.source_snapshot_root_sha256,
        "authorization_receipt_sha256": contract.authorization_sha256,
        "expected_rows": 108,
        "observed_rows": len(rows),
        "objective_and_archive_replay_pass_rows": sum(
            row["status"] == "PASS_ENGINEERING_ROW_OBJECTIVE_ARCHIVE_REPLAYED"
            for row in rows
        ),
        "metric_replay_pass_rows": sum(
            row.get("metric_replay_receipt_sha256") is not None for row in rows
        ),
        "terminal_receipt_hash_bound_before_replay_rows": len(rows),
        "row_last_receipts_verified": len(rows),
        "noninferiority_gate": analysis["overall_gate"],
        "randomization_evidence": {
            "bootstrap_method": "paired_case_cluster_percentile_bootstrap",
            "bootstrap_samples_per_comparison": 20_000,
            "base_randomization_seed": 31_061,
            "sign_flip_method": "exact_cluster_sign_flip",
        },
        "phase_release_effect": (
            "STOP_BEFORE_SELECTION_PARTITION_MATERIALIZATION"
            if str(analysis["overall_gate"]).startswith("FAIL_")
            else "DEVELOPMENT_GATE_ONLY_NO_LATER_PHASE_RELEASE"
        ),
        "runtime_efficiency_claim_authorized": False,
        "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
        "selection_entropy_release": "PROHIBITED",
        "calibration_execution": "PROHIBITED",
        "formal_execution": "PROHIBITED",
        "formal_authorized": False,
    }
    _exclusive_write_json(audit_path, audit)
    try:
        verify_finalized_matrix_output(output, contract=contract, plan=plan)
    except BaseException as error:
        cleanup_errors: list[OSError] = []
        for owned_path in (audit_path, aggregate_path):
            try:
                owned_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            raise RuntimeError(
                "Second-pass verification failed and final receipt cleanup was incomplete."
            ) from error
        raise
    return audit


def verify_finalized_matrix_output(
    output_directory: str | Path,
    *,
    contract: FrozenContract,
    plan: Mapping[str, object],
) -> dict[str, object]:
    """Replay all rows and independently recompute the finalized statistics."""

    output = Path(output_directory).resolve()
    aggregate_path = output / "matrix.aggregate.json"
    audit_path = output / "post_run_audit.json"
    if not aggregate_path.is_file() or not audit_path.is_file():
        raise RuntimeError("Final matrix verification requires both final receipts.")
    rows = _collect_complete_rows(output, contract=contract, plan=plan)
    case_records = [
        {"case_id": case.case_id, "family": case.family, "size": case.size}
        for case in contract.cases
    ]
    analysis = analyze_development_parity(
        rows,
        case_records=case_records,
        seeds=contract.seeds,
        margin=0.005,
        size_stratum_margin=0.010,
        bootstrap_samples=20_000,
        bootstrap_seed=31_061,
        tie_tolerance=1e-12,
    )
    aggregate_rows = [
        {
            key: row[key]
            for key in (
                "case_id",
                "family",
                "size",
                "seed",
                "arm_id",
                "normalized_left_continuous_hv_auc",
                "normalized_terminal_hv",
                "source_snapshot_root_sha256",
                "trace_database_sha256",
                "detached_terminal_receipt_sha256",
                "objective_archive_replay_receipt_sha256",
                "metric_replay_receipt_sha256",
                "row_receipt_path",
                "row_receipt_sha256",
            )
        }
        for row in rows
    ]
    expected_aggregate = {
        "schema": "pareto_v21e3r1_development_matched_matrix_aggregate_v1",
        "status": "COMPLETE_DEVELOPMENT_MATRIX_ENGINEERING_EVIDENCE",
        "scientific_scope": "authors_generated_development_only_not_formal_evidence",
        "artifact_path_semantics": MATRIX_ARTIFACT_PATH_SEMANTICS,
        "source_snapshot_root_sha256": contract.source_snapshot_root_sha256,
        "authorization_receipt_sha256": contract.authorization_sha256,
        "matrix_plan_sha256": _sha256(output / "matrix.plan.json"),
        "input_sha256": dict(contract.input_sha256),
        "expected_rows": 108,
        "observed_rows": len(rows),
        "rows": aggregate_rows,
        "analysis": analysis,
        "runtime_efficiency_claim_authorized": False,
        "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
        "selection_entropy_release": "PROHIBITED",
        "calibration_execution": "PROHIBITED",
        "formal_execution": "PROHIBITED",
        "formal_authorized": False,
    }
    observed_aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if observed_aggregate != expected_aggregate:
        raise RuntimeError("Independent recomputation rejected the matrix aggregate.")
    aggregate_sha256 = _sha256(aggregate_path)
    expected_audit = {
        "schema": "pareto_v21e3r1_development_matrix_post_run_audit_v1",
        "status": "PASS_COMPLETE_DEVELOPMENT_MATRIX_AUDITED",
        "matrix_aggregate_path": "matrix.aggregate.json",
        "matrix_aggregate_sha256": aggregate_sha256,
        "source_snapshot_root_sha256": contract.source_snapshot_root_sha256,
        "authorization_receipt_sha256": contract.authorization_sha256,
        "expected_rows": 108,
        "observed_rows": len(rows),
        "objective_and_archive_replay_pass_rows": len(rows),
        "metric_replay_pass_rows": len(rows),
        "terminal_receipt_hash_bound_before_replay_rows": len(rows),
        "row_last_receipts_verified": len(rows),
        "noninferiority_gate": analysis["overall_gate"],
        "randomization_evidence": {
            "bootstrap_method": "paired_case_cluster_percentile_bootstrap",
            "bootstrap_samples_per_comparison": 20_000,
            "base_randomization_seed": 31_061,
            "sign_flip_method": "exact_cluster_sign_flip",
        },
        "phase_release_effect": (
            "STOP_BEFORE_SELECTION_PARTITION_MATERIALIZATION"
            if str(analysis["overall_gate"]).startswith("FAIL_")
            else "DEVELOPMENT_GATE_ONLY_NO_LATER_PHASE_RELEASE"
        ),
        "runtime_efficiency_claim_authorized": False,
        "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
        "selection_entropy_release": "PROHIBITED",
        "calibration_execution": "PROHIBITED",
        "formal_execution": "PROHIBITED",
        "formal_authorized": False,
    }
    observed_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if observed_audit != expected_audit:
        raise RuntimeError("Independent recomputation rejected the runner post-run receipt.")
    return {
        "status": "PASS_RECOMPUTED_FROM_108_OBJECTIVE_AND_METRIC_REPLAYS",
        "row_count": len(rows),
        "aggregate_sha256": aggregate_sha256,
        "runner_post_run_audit_sha256": _sha256(audit_path),
        "analysis": analysis,
    }


def run_development_matrix(
    *,
    contract: FrozenContract,
    output_directory: str | Path,
    resume: bool,
) -> dict[str, object]:
    """Run or strictly resume the exact 108-row development matrix."""

    if (
        not contract.live_authorization_verified
        or contract.authorization_path is None
        or contract.authorization_sha256 is None
    ):
        raise ValueError("The matrix lacks a live-verified authorization receipt.")
    try:
        from ijoc_submission_v21e3r1.scripts.preflight_v21e3r1_development_parity import (
            verify_existing_development_parity_authorization,
        )
    except ModuleNotFoundError:  # Direct ``python path/to/script.py`` execution.
        from preflight_v21e3r1_development_parity import (  # type: ignore[no-redef]
            verify_existing_development_parity_authorization,
        )

    verify_existing_development_parity_authorization(
        repo_root=Path(__file__).resolve().parents[2],
        authorization_path=contract.authorization_path,
        expected_source_snapshot_root_sha256=contract.source_snapshot_root_sha256,
    )
    if _sha256(contract.authorization_path) != contract.authorization_sha256:
        raise RuntimeError("The authorization receipt bytes drifted before matrix execution.")
    output = Path(output_directory).resolve()
    plan = initialize_matrix_output(output, contract=contract, resume=resume)
    aggregate_path = output / "matrix.aggregate.json"
    audit_path = output / "post_run_audit.json"
    if aggregate_path.exists() or audit_path.exists():
        if not resume or not aggregate_path.is_file() or not audit_path.is_file():
            raise RuntimeError("Strict resume found partial final matrix receipts.")
        verify_finalized_matrix_output(output, contract=contract, plan=plan)
        return json.loads(audit_path.read_text(encoding="utf-8"))
    case_by_id = {case.case_id: case for case in contract.cases}
    for planned in plan["rows"]:
        row_directory = output / "rows" / str(planned["row_slug"])
        if row_directory.exists():
            _load_completed_row(
                row_directory, planned=planned, contract=contract
            )
            continue
        execute_row(
            case=case_by_id[str(planned["case_id"])],
            arm_id=str(planned["arm_id"]),
            seed=int(planned["seed"]),
            budget=contract.budget,
            checkpoint_period=contract.checkpoint_period,
            reference_directions=contract.reference_directions,
            source_snapshot_root_sha256=contract.source_snapshot_root_sha256,
            row_directory=row_directory,
            metric_manifest_sha256=contract.input_sha256["metric_manifest"],
            scientific_scope="authors_generated_development_only_not_formal_evidence",
        )
    return finalize_matrix_output(output, contract=contract, plan=plan)


def _default_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[2]
    old = root / "ijoc_submission_v21e3"
    return {
        "case_manifest": old
        / "development_partitions_v1"
        / "case_manifest.json",
        "reference_manifest": old
        / "development_manifests_v1"
        / "reference_manifest_development.json",
        "config_manifest": old
        / "development_manifests_v1"
        / "config_manifest_development.json",
        "metric_manifest": old
        / "development_manifests_v1"
        / "metric_manifest.json",
        "protocol": old / "protocol" / "V21E3_C0_PARITY_PROTOCOL_V2.json",
    }


def main(argv: list[str] | None = None) -> int:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(
        description=(
            "Run the authorized V21e3r1 12-case x 3-seed x 3-arm "
            "development-only matched matrix."
        )
    )
    parser.add_argument("--case-manifest", type=Path, default=defaults["case_manifest"])
    parser.add_argument(
        "--reference-manifest", type=Path, default=defaults["reference_manifest"]
    )
    parser.add_argument(
        "--config-manifest", type=Path, default=defaults["config_manifest"]
    )
    parser.add_argument(
        "--metric-manifest", type=Path, default=defaults["metric_manifest"]
    )
    parser.add_argument("--protocol", type=Path, default=defaults["protocol"])
    parser.add_argument("--authorization-receipt", type=Path, required=True)
    parser.add_argument("--source-snapshot-root-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    contract = load_frozen_contract(
        case_manifest_path=args.case_manifest,
        reference_manifest_path=args.reference_manifest,
        config_manifest_path=args.config_manifest,
        metric_manifest_path=args.metric_manifest,
        protocol_path=args.protocol,
        authorization_path=args.authorization_receipt,
        source_snapshot_root_sha256=args.source_snapshot_root_sha256,
        require_matrix_authorization=True,
    )
    audit = run_development_matrix(
        contract=contract,
        output_directory=args.output_directory,
        resume=args.resume,
    )
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
