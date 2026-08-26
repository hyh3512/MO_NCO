from __future__ import annotations

"""Same-implementation full branch reexecution for V21e3r1 and baselines.

This is stronger than objective/archive replay because it re-runs the frozen
stochastic program and compares proposal, retry, evaluation, and decision
records.  It is explicitly *not* implementation-independent scientific
replication: the production algorithm modules are reused.
"""

import argparse
from dataclasses import fields
import hashlib
import inspect
import json
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Mapping, Sequence

from .pareto_v21e3_baselines import (
    V21E3BaselineConfig,
    load_v21e3_development_problem,
    run_v21e3_development_baseline,
)
from .pareto_v21e3_hybrid import (
    V21E3HybridConfig,
    V21E3TypedHybridParetoSearch,
)
from .pareto_ijoc_problem import problem_sha256
from .pareto_v21e3_trace import V21E3RunContext


_HYBRID_FALLBACK_SOURCE_KIND = "hybrid_module_sha256_fallback_development_only_v1"
_BASELINE_FALLBACK_SOURCE_KIND = (
    "development_adapter_module_sha256_fallback_pre_snapshot_v1"
)
_EXPLICIT_SOURCE_KINDS = {
    "explicit_source_snapshot_or_release_manifest_sha256_v1",
    "explicit_successor_source_snapshot_sha256_v1",
}
_NONSEMANTIC_CONFIG_FIELDS = {
    "trace_database",
    "terminal_receipt",
    "receipt_database_path",
    "capture_trace",
    "case_artifact_sha256",
    "source_snapshot_sha256",
}
_V8_SUCCESSOR_POLICY_CONFIG_FIELDS = frozenset(
    {
        "post_initialization_search_policy",
        "mokp_novelty_generation_policy",
    }
)
_V9_INFORMATION_POLICY_CONFIG_FIELDS = frozenset(
    {
        "candidate_screening_policy",
        "candidate_screening_cap",
        "archive_tradeoff_lambda",
        "attempt_cap",
        "structural_screening_cap",
        "wall_time_cap_seconds",
    }
)
_SUCCESSOR_POLICY_CONFIG_FIELDS = (
    _V8_SUCCESSOR_POLICY_CONFIG_FIELDS | _V9_INFORMATION_POLICY_CONFIG_FIELDS
)
_V8_SUCCESSOR_FACTORIAL_DIAGNOSTICS = frozenset(
    {
        "V21E3R1_SUCCESSOR_FACTORIAL_MOKP_LEGACY",
        "V21E3R1_SUCCESSOR_FACTORIAL_MOKP_ANCHOR_ONLY",
        "V21E3R1_SUCCESSOR_FACTORIAL_MOKP_NOVELTY_ONLY",
        "V21E3R1_SUCCESSOR_FACTORIAL_MOKP_BOTH",
        "V21E3R1_SUCCESSOR_FACTORIAL_MOTSP_LEGACY",
        "V21E3R1_SUCCESSOR_FACTORIAL_MOTSP_ANCHOR",
    }
)
_V9_INFORMATION_DIAGNOSTICS = frozenset(
    {
        "V21E3R1_V9_LEGACY_MOKP",
        "V21E3R1_V9_INFORMATION_SCREEN_MOKP",
        "V21E3R1_V9_LYAPUNOV_MOKP",
        "V21E3R1_V9_INFORMATION_LYAPUNOV_MOKP",
        "V21E3R1_V9_LEGACY_MOTSP",
        "V21E3R1_V9_INFORMATION_SCREEN_MOTSP",
        "V21E3R1_V9_LYAPUNOV_MOTSP",
        "V21E3R1_V9_INFORMATION_LYAPUNOV_MOTSP",
    }
)
_SUCCESSOR_FACTORIAL_DIAGNOSTICS = (
    _V8_SUCCESSOR_FACTORIAL_DIAGNOSTICS | _V9_INFORMATION_DIAGNOSTICS
)
_LEGACY_SUCCESSOR_POLICY_DEFAULTS = {
    "post_initialization_search_policy": "proposal_chain_v21e3r1_v1",
    "mokp_novelty_generation_policy": "legacy_retry_and_local_v21e3r1_v1",
    "candidate_screening_policy": "disabled_v1",
    "candidate_screening_cap": 1,
    "archive_tradeoff_lambda": 0.0,
    "attempt_cap": None,
    "structural_screening_cap": None,
    "wall_time_cap_seconds": None,
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _reject_duplicate_json_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"The source manifest contains duplicate JSON key {key!r}."
            )
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> object:
    raise ValueError(
        f"The source manifest contains non-finite JSON constant {value!r}."
    )


def _load_strict_canonical_manifest(raw: bytes) -> dict[str, object]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("The source manifest is not strict UTF-8 JSON.") from error
    if not isinstance(payload, dict):
        raise ValueError("The source manifest is not an object.")
    if raw != _canonical_bytes(payload) + b"\n":
        raise ValueError(
            "The source manifest must be canonical JSON followed by one newline."
        )
    return payload


def _sqlite_read_only_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=ro"


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} is not lowercase SHA-256.")
    return value


def _source_module(candidate: str) -> tuple[Path, str]:
    target = (
        V21E3TypedHybridParetoSearch
        if candidate in {"C0", "C1", "C2", "C3"}
        else run_v21e3_development_baseline
    )
    raw = inspect.getsourcefile(target)
    if raw is None:
        raise RuntimeError("Cannot locate the executing algorithm source module.")
    path = Path(raw).resolve()
    return path, f"mo_nco/{path.name}"


def _manifest_entries(payload: Mapping[str, object]) -> Sequence[object]:
    for key in ("entries", "files", "bound_files", "prospective_source_files"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    raise ValueError("The source manifest omits a source-file inventory.")


def _validate_source_binding(
    context: Mapping[str, object],
    *,
    candidate: str,
    source_manifest_path: str | Path | None,
) -> dict[str, object]:
    expected = _require_sha256(
        context.get("algorithm_source_sha256"),
        label="The run-context algorithm source digest",
    )
    kind = context.get("algorithm_source_binding_kind")
    if type(kind) is not str:
        raise ValueError("The run context omits an exact algorithm source binding kind.")
    module_path, manifest_relative_path = _source_module(candidate)
    observed_module = _sha256_file(module_path)
    expected_fallback = (
        _HYBRID_FALLBACK_SOURCE_KIND
        if candidate in {"C0", "C1", "C2", "C3"}
        else _BASELINE_FALLBACK_SOURCE_KIND
    )
    if kind == expected_fallback:
        if source_manifest_path is not None:
            raise ValueError(
                "A fallback module binding cannot be replaced by a source manifest."
            )
        if observed_module != expected:
            raise ValueError("The executing algorithm source module drifted.")
        return {
            "binding_kind": kind,
            "context_source_sha256": expected,
            "executing_module": manifest_relative_path,
            "executing_module_sha256": observed_module,
            "source_manifest": None,
            "source_manifest_sha256": None,
            "replay_source_snapshot_sha256": None,
        }
    if kind not in _EXPLICIT_SOURCE_KINDS:
        raise ValueError(f"Unsupported algorithm source binding kind: {kind}")
    if source_manifest_path is None:
        raise ValueError(
            "An explicit algorithm source binding requires a source manifest."
        )
    manifest_path = Path(source_manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_raw = manifest_path.read_bytes()
    manifest = _load_strict_canonical_manifest(manifest_raw)
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    root_inventory_keys = {
        "source_root_sha256": None,
        "prospective_source_root_sha256": "prospective_source_files",
        "bound_files_root_sha256": "bound_files",
        "files_root_sha256": "files",
        "root_sha256": None,
    }
    matching_roots: list[tuple[str, Sequence[object]]] = []
    for root_key, inventory_key in root_inventory_keys.items():
        declared = manifest.get(root_key)
        if declared is None:
            continue
        declared_sha = _require_sha256(
            declared,
            label=f"The source-manifest {root_key}",
        )
        if declared_sha != expected:
            continue
        inventory = (
            manifest.get(inventory_key)
            if inventory_key is not None
            else _manifest_entries(manifest)
        )
        if not isinstance(inventory, list):
            raise ValueError(
                f"The source manifest omits inventory for {root_key}."
            )
        matching_roots.append((root_key, inventory))
    if expected == manifest_sha:
        bound_inventory = _manifest_entries(manifest)
    else:
        if not matching_roots:
            raise ValueError(
                "The source manifest does not bind the run-context source digest."
            )
        bound_inventory = matching_roots[0][1]
        canonical_bound_inventory = _canonical_bytes(bound_inventory)
        for root_key, inventory in matching_roots:
            inventory_bytes = _canonical_bytes(inventory)
            observed_root = hashlib.sha256(inventory_bytes).hexdigest()
            if observed_root != expected:
                raise ValueError(
                    f"The source-manifest {root_key} does not match its inventory."
                )
            if inventory_bytes != canonical_bound_inventory:
                raise ValueError(
                    "The source manifest has ambiguous context-bound inventories."
                )
    matching_entries = []
    for raw_entry in bound_inventory:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("The source manifest contains a malformed file entry.")
        raw_path = raw_entry.get("path", raw_entry.get("relative_path"))
        normalized = str(raw_path).replace("\\", "/")
        if normalized == manifest_relative_path or normalized.endswith(
            "/" + manifest_relative_path
        ):
            matching_entries.append(raw_entry)
    if len(matching_entries) != 1:
        raise ValueError(
            "The source manifest must bind the executing module exactly once."
        )
    manifest_module_sha = _require_sha256(
        matching_entries[0].get("sha256"),
        label="The source-manifest module digest",
    )
    if manifest_module_sha != observed_module:
        raise ValueError("The executing module disagrees with the source manifest.")
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent
    live_python_sha256 = {
        path.relative_to(project_root).as_posix(): _sha256_file(path)
        for path in package_root.rglob("*.py")
    }
    frozen_python_sha256: dict[str, str] = {}
    for raw_entry in bound_inventory:
        if not isinstance(raw_entry, Mapping):
            continue
        raw_path = raw_entry.get("path", raw_entry.get("relative_path"))
        if type(raw_path) is not str:
            continue
        normalized = raw_path.replace("\\", "/")
        marker = "mo_nco/"
        marker_index = normalized.find(marker)
        if marker_index >= 0:
            normalized = normalized[marker_index:]
        if normalized.startswith(marker) and normalized.endswith(".py"):
            if normalized in frozen_python_sha256:
                raise ValueError(
                    "The source manifest duplicates a frozen mo_nco Python source: "
                    f"{normalized}"
                )
            frozen_python_sha256[normalized] = _require_sha256(
                raw_entry.get("sha256"),
                label=f"The source-manifest digest for {normalized}",
            )
    live_python_paths = set(live_python_sha256)
    frozen_python_paths = set(frozen_python_sha256)
    missing_python_paths = sorted(live_python_paths - frozen_python_paths)
    if missing_python_paths:
        raise ValueError(
            "The source manifest is missing live mo_nco Python source files: "
            + ", ".join(missing_python_paths)
        )
    extra_python_paths = sorted(frozen_python_paths - live_python_paths)
    if extra_python_paths:
        raise ValueError(
            "The source manifest contains extra frozen mo_nco Python source files: "
            + ", ".join(extra_python_paths)
        )
    drifted_python_paths = sorted(
        path
        for path in live_python_paths
        if frozen_python_sha256[path] != live_python_sha256[path]
    )
    if drifted_python_paths:
        raise ValueError(
            "The live mo_nco Python source hash drifted from the frozen manifest: "
            + ", ".join(drifted_python_paths)
        )
    return {
        "binding_kind": kind,
        "context_source_sha256": expected,
        "executing_module": manifest_relative_path,
        "executing_module_sha256": observed_module,
        "source_closure_scope": "all_live_mo_nco_python_sources",
        "source_closure_file_count": len(live_python_sha256),
        "source_closure_verified": True,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": manifest_sha,
        "replay_source_snapshot_sha256": expected,
    }


def _load_context(database: Path) -> dict[str, object]:
    connection = sqlite3.connect(_sqlite_read_only_uri(database), uri=True)
    try:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("The branch-replay SQLite database fails integrity_check.")
        row = connection.execute(
            "SELECT run_context_json,run_context_digest_sha256 "
            "FROM run_attempt WHERE run_id=1"
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise RuntimeError("Trace omits run context.")
    raw = str(row[0])
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("Run context is not an object.")
    binding = V21E3RunContext(payload)
    if raw != binding.canonical_json:
        raise ValueError("The trace run context is not canonical JSON.")
    if str(row[1]) != binding.digest_sha256:
        raise ValueError("The trace run-context digest is invalid.")
    return payload


def _validate_problem_binding(
    context: Mapping[str, object],
    *,
    problem_artifact: Path,
    problem: object,
) -> dict[str, object]:
    artifact_sha = _sha256_file(problem_artifact)
    semantic_sha = problem_sha256(problem)  # type: ignore[arg-type]
    expected_semantic = _require_sha256(
        context.get("problem_semantic_sha256"),
        label="The run-context problem semantic digest",
    )
    if semantic_sha != expected_semantic:
        raise ValueError("The replay problem semantic identity drifted.")
    expected_case = _require_sha256(
        context.get("case_artifact_sha256"),
        label="The run-context case-artifact digest",
    )
    kind = context.get("case_artifact_binding_kind")
    if kind == "explicit_case_artifact_sha256_v1":
        if artifact_sha != expected_case:
            raise ValueError("The replay problem artifact bytes drifted.")
        replay_case_sha: str | None = expected_case
    elif kind == "problem_semantic_sha256_fallback_development_only_v1":
        if semantic_sha != expected_case:
            raise ValueError("The replay problem fallback binding drifted.")
        replay_case_sha = None
    else:
        raise ValueError("Unsupported case-artifact binding kind.")
    return {
        "binding_kind": kind,
        "context_case_artifact_sha256": expected_case,
        "problem_artifact_sha256": artifact_sha,
        "problem_semantic_sha256": semantic_sha,
        "replay_case_artifact_sha256": replay_case_sha,
    }


def _dataclass_kwargs(
    cls: type,
    payload: Mapping[str, object],
    *,
    semantic_aliases: Sequence[str] = (),
) -> dict[str, object]:
    names = {field.name for field in fields(cls)}
    semantic_names = names - _NONSEMANTIC_CONFIG_FIELDS
    expected = semantic_names | set(semantic_aliases)
    observed = set(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            "The replay context does not have exact config keys: "
            f"missing={missing}, extra={extra}."
        )
    return {name: payload[name] for name in semantic_names}


def _hybrid_dataclass_kwargs(payload: Mapping[str, object]) -> dict[str, object]:
    names = {field.name for field in fields(V21E3HybridConfig)}
    semantic_names = names - _NONSEMANTIC_CONFIG_FIELDS
    diagnostic = payload.get("development_diagnostic_id")
    if diagnostic in _V9_INFORMATION_DIAGNOSTICS:
        expected = semantic_names
    elif diagnostic in _V8_SUCCESSOR_FACTORIAL_DIAGNOSTICS:
        expected = semantic_names - _V9_INFORMATION_POLICY_CONFIG_FIELDS
    else:
        expected = semantic_names - _SUCCESSOR_POLICY_CONFIG_FIELDS
    observed = set(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            "The replay context does not have exact config keys: "
            f"missing={missing}, extra={extra}."
        )
    kwargs = {name: payload[name] for name in expected}
    if diagnostic in _V8_SUCCESSOR_FACTORIAL_DIAGNOSTICS:
        for key in _V9_INFORMATION_POLICY_CONFIG_FIELDS:
            kwargs[key] = _LEGACY_SUCCESSOR_POLICY_DEFAULTS[key]
    elif diagnostic not in _V9_INFORMATION_DIAGNOSTICS:
        kwargs.update(_LEGACY_SUCCESSOR_POLICY_DEFAULTS)
    return kwargs


def _hybrid_config(
    context: Mapping[str, object],
    *,
    trace_path: Path,
    terminal_path: Path,
    source_snapshot_sha256: str | None,
    case_artifact_sha256: str | None,
) -> V21E3HybridConfig:
    raw = context.get("algorithm_config")
    if not isinstance(raw, dict):
        raise RuntimeError("Hybrid context omits algorithm_config.")
    kwargs = _hybrid_dataclass_kwargs(raw)
    if "reference_directions" in kwargs:
        kwargs["reference_directions"] = tuple(tuple(float(x) for x in row) for row in kwargs["reference_directions"])
    kwargs.update(
        trace_database=str(trace_path),
        terminal_receipt=str(terminal_path),
        receipt_database_path="trace.sqlite3",
        capture_trace=False,
        case_artifact_sha256=case_artifact_sha256,
        source_snapshot_sha256=source_snapshot_sha256,
    )
    return V21E3HybridConfig(**kwargs)  # type: ignore[arg-type]


def _baseline_config(
    context: Mapping[str, object],
    *,
    trace_path: Path,
    terminal_path: Path,
    source_snapshot_sha256: str | None,
    case_artifact_sha256: str | None,
) -> V21E3BaselineConfig:
    raw = context.get("algorithm_config")
    if not isinstance(raw, dict):
        raise RuntimeError("Baseline context omits algorithm_config.")
    kwargs = _dataclass_kwargs(
        V21E3BaselineConfig,
        raw,
        semantic_aliases=("candidate_id", "phase", "adaptation_identity"),
    )
    if (
        raw["candidate_id"] != raw["arm_id"]
        or raw["phase"] != raw["evidence_partition"]
    ):
        raise ValueError("Baseline semantic config aliases disagree.")
    normalized = dict(kwargs)
    normalized.pop("candidate_id", None)
    normalized.pop("phase", None)
    normalized.pop("adaptation_identity", None)
    kwargs = normalized
    if "reference_directions" in kwargs:
        kwargs["reference_directions"] = tuple(tuple(float(x) for x in row) for row in kwargs["reference_directions"])
    kwargs.update(
        trace_database=str(trace_path),
        terminal_receipt=str(terminal_path),
        receipt_database_path="trace.sqlite3",
        capture_trace=False,
        case_artifact_sha256=case_artifact_sha256,
        source_snapshot_sha256=source_snapshot_sha256,
        selection_authorized=False,
        formal_authorized=False,
    )
    return V21E3BaselineConfig(**kwargs)  # type: ignore[arg-type]


def _rows(database: Path, table: str, columns: Sequence[str]) -> list[tuple[Any, ...]]:
    connection = sqlite3.connect(_sqlite_read_only_uri(database), uri=True)
    try:
        return list(connection.execute(f"SELECT {','.join(columns)} FROM {table} ORDER BY 1"))
    finally:
        connection.close()


def _semantic_decision(raw: str) -> dict[str, object]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError("Decision payload is not an object.")
    return dict(value)


def _terminal_semantics(database: Path) -> tuple[dict[str, object], dict[str, object]]:
    connection = sqlite3.connect(_sqlite_read_only_uri(database), uri=True)
    try:
        row = connection.execute(
            "SELECT status,failure_code,receipt_json,receipt_sha256 "
            "FROM terminal_receipts WHERE run_id=1"
        ).fetchone()
        run = connection.execute(
            "SELECT status,terminal_receipt_sha256 FROM run_attempt WHERE run_id=1"
        ).fetchone()
    finally:
        connection.close()
    if row is None or run is None:
        raise ValueError("A replay trace omits terminal state.")
    raw = str(row[2])
    receipt = json.loads(raw)
    if not isinstance(receipt, dict) or _canonical_bytes(receipt).decode("utf-8") != raw:
        raise ValueError("A replay terminal receipt is not canonical JSON.")
    stored_sha = _require_sha256(row[3], label="The embedded terminal receipt digest")
    payload_sha = _require_sha256(
        receipt.get("receipt_payload_sha256"),
        label="The terminal receipt payload digest",
    )
    core = dict(receipt)
    core.pop("receipt_payload_sha256")
    if hashlib.sha256(_canonical_bytes(core)).hexdigest() != payload_sha:
        raise ValueError("A replay terminal receipt has an invalid payload digest.")
    if stored_sha != payload_sha or run[1] != stored_sha:
        raise ValueError("A replay terminal receipt is detached from run_attempt.")
    semantic = dict(receipt)
    semantic.pop("database_path", None)
    semantic.pop("receipt_payload_sha256", None)
    resources = semantic.get("resource_accounting")
    if isinstance(resources, dict) and "elapsed_seconds" in resources:
        resources = dict(resources)
        resources["elapsed_seconds"] = "RUNTIME_DEPENDENT_MONOTONIC_MEASUREMENT"
        semantic["resource_accounting"] = resources
    gates = semantic.get("finalization_gates")
    if isinstance(gates, dict):
        gates = dict(gates)
        gate_resources = gates.get("resource_accounting")
        if isinstance(gate_resources, dict) and "elapsed_seconds" in gate_resources:
            gate_resources = dict(gate_resources)
            gate_resources["elapsed_seconds"] = (
                "RUNTIME_DEPENDENT_MONOTONIC_MEASUREMENT"
            )
            gates["resource_accounting"] = gate_resources
        semantic["finalization_gates"] = gates
    binding = {
        "run_status": str(run[0]),
        "terminal_status": str(row[0]),
        "failure_code": row[1],
        "receipt_sha256": stored_sha,
    }
    return semantic, binding


def _accounting_semantics(database: Path) -> dict[str, object]:
    connection = sqlite3.connect(_sqlite_read_only_uri(database), uri=True)
    try:
        attempts = list(
            connection.execute(
                "SELECT status,physical_call_started,charged_evaluation_index,"
                "cache_source_evaluation_index FROM attempts ORDER BY attempt_index"
            )
        )
        evaluation_count = int(
            connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
        )
        decision_count = int(
            connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        )
        terminal_row = connection.execute(
            "SELECT receipt_json FROM terminal_receipts WHERE run_id=1"
        ).fetchone()
        terminal_chains = connection.execute(
            "SELECT "
            "(SELECT attempt_sha256 FROM attempts ORDER BY attempt_index DESC LIMIT 1),"
            "(SELECT record_sha256 FROM evaluations ORDER BY evaluation_index DESC LIMIT 1),"
            "(SELECT decision_sha256 FROM decisions ORDER BY evaluation_index DESC LIMIT 1)"
        ).fetchone()
    finally:
        connection.close()
    if terminal_row is None:
        raise ValueError("A replay trace omits terminal accounting.")
    terminal = json.loads(str(terminal_row[0]))
    count_fields = (
        "attempt_count",
        "physical_call_started_count",
        "charged_evaluation_count",
        "decision_count",
        "cache_hit_count",
        "unresolved_decision_count",
    )
    if not isinstance(terminal, dict) or any(
        type(terminal.get(key)) is not int for key in count_fields
    ):
        raise ValueError("A replay terminal receipt has non-exact accounting types.")
    physical_starts = 0
    cache_hits = 0
    for status, physical, charged, cache_source in attempts:
        if type(physical) is not int or physical not in (0, 1):
            raise ValueError("A replay attempt has an invalid physical-start flag.")
        physical_starts += physical
        cache_hits += int(str(status) == "CACHE_HIT")
        if (str(status) == "CACHE_HIT") != (cache_source is not None):
            raise ValueError("A replay cache-hit attempt has inconsistent accounting.")
        if (str(status) == "EVALUATED") != (charged is not None):
            if str(status) != "CACHE_HIT":
                raise ValueError("A replay attempt has inconsistent charge accounting.")
    observed = {
        "attempt_count": len(attempts),
        "physical_call_started_count": physical_starts,
        "charged_evaluation_count": evaluation_count,
        "decision_count": decision_count,
        "cache_hit_count": cache_hits,
        "unresolved_decision_count": evaluation_count - decision_count,
    }
    if any(terminal[key] != value for key, value in observed.items()):
        raise ValueError("A replay terminal receipt disagrees with observed accounting.")
    if terminal.get("status") != "SUCCESS":
        raise ValueError("Branch replay requires a successful original trace.")
    if not (
        physical_starts == evaluation_count
        and decision_count == evaluation_count
        and len(attempts) == evaluation_count + cache_hits
    ):
        raise ValueError("A successful replay trace violates terminal accounting identities.")
    expected_chains = (
        terminal.get("terminal_attempt_chain_sha256"),
        terminal.get("terminal_evaluation_chain_sha256"),
        terminal.get("terminal_decision_chain_sha256"),
    )
    if tuple(terminal_chains) != expected_chains:
        raise ValueError("A replay terminal receipt disagrees with terminal chains.")
    return observed


def _archive_semantics(decisions: Sequence[tuple[Any, ...]]) -> list[tuple[object, ...]]:
    fields_to_bind = (
        "archive_changed",
        "retained_after_update",
        "archive_size_after",
        "new_evaluated_cell",
        "new_nondominated_cell",
    )
    output: list[tuple[object, ...]] = []
    for evaluation_index, raw, *_unused in decisions:
        payload = _semantic_decision(str(raw))
        output.append(
            (evaluation_index, *(payload.get(key) for key in fields_to_bind))
        )
    return output


def _first_mismatch(
    original: Sequence[object], replay: Sequence[object]
) -> dict[str, object] | None:
    def witness_value(value: object) -> object:
        if isinstance(value, bytes):
            return {
                "blob_bytes": len(value),
                "blob_sha256": hashlib.sha256(value).hexdigest(),
            }
        if isinstance(value, (list, tuple)):
            return [witness_value(item) for item in value]
        if isinstance(value, Mapping):
            return {
                str(key): witness_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        return value

    limit = max(len(original), len(replay))
    for index in range(limit):
        left = original[index] if index < len(original) else None
        right = replay[index] if index < len(replay) else None
        if left != right:
            return {
                "ordinal": index + 1,
                "original_sha256": hashlib.sha256(
                    _canonical_bytes(witness_value(left))
                ).hexdigest(),
                "replay_sha256": hashlib.sha256(
                    _canonical_bytes(witness_value(right))
                ).hexdigest(),
            }
    return None


def compare_trace_semantics(original: Path, replay: Path) -> dict[str, object]:
    _load_context(original)
    _load_context(replay)
    attempt_columns = (
        "attempt_index", "proposal_solution_ref", "proposal_sha256", "proposal_json",
        "proposal_raw_sha256", "context_json",
        "status", "physical_call_started", "charged_evaluation_index",
        "cache_source_evaluation_index", "failure_code", "failure_detail_json",
        "prev_attempt_sha256", "attempt_sha256",
    )
    evaluation_columns = (
        "evaluation_index", "attempt_index", "evidence_partition", "search_phase_id",
        "stage_id", "type_id", "operator_id", "operator_call_id", "proposal_sha256",
        "proposal_solution_ref", "objectives_json", "prev_record_sha256", "record_sha256",
    )
    solution_columns = (
        "solution_ref", "solution_sha256", "family", "codec", "solution_size", "payload",
    )
    run_columns = (
        "run_id", "problem", "family", "run_context_json",
        "run_context_digest_sha256", "status",
    )
    original_run = _rows(original, "run_attempt", run_columns)
    replay_run = _rows(replay, "run_attempt", run_columns)
    original_solutions = _rows(original, "solutions", solution_columns)
    replay_solutions = _rows(replay, "solutions", solution_columns)
    original_attempts = _rows(original, "attempts", attempt_columns)
    replay_attempts = _rows(replay, "attempts", attempt_columns)
    original_evaluations = _rows(original, "evaluations", evaluation_columns)
    replay_evaluations = _rows(replay, "evaluations", evaluation_columns)
    decision_columns = (
        "evaluation_index", "decision_json", "prev_decision_sha256", "decision_sha256",
    )
    original_decisions = _rows(original, "decisions", decision_columns)
    replay_decisions = _rows(replay, "decisions", decision_columns)
    original_archive = _archive_semantics(original_decisions)
    replay_archive = _archive_semantics(replay_decisions)
    original_terminal, original_terminal_binding = _terminal_semantics(original)
    replay_terminal, replay_terminal_binding = _terminal_semantics(replay)
    original_accounting = _accounting_semantics(original)
    replay_accounting = _accounting_semantics(replay)
    semantic_groups: dict[str, tuple[Sequence[object], Sequence[object]]] = {
        "run_context": (original_run, replay_run),
        "solutions": (original_solutions, replay_solutions),
        "attempts": (original_attempts, replay_attempts),
        "evaluations": (original_evaluations, replay_evaluations),
        "decisions": (original_decisions, replay_decisions),
        "archive": (original_archive, replay_archive),
        "terminal": ([original_terminal], [replay_terminal]),
        "accounting": ([original_accounting], [replay_accounting]),
    }
    checks = {
        key: left == right for key, (left, right) in semantic_groups.items()
    }
    first_mismatch = next(
        (
            {"semantic_group": key, **witness}
            for key, (left, right) in semantic_groups.items()
            if (witness := _first_mismatch(left, right)) is not None
        ),
        None,
    )
    return {
        "schema": "v21e3r1_same_implementation_branch_replay_v1",
        "status": "PASS_SAME_IMPLEMENTATION_FULL_BRANCH_REEXECUTION" if all(checks.values()) else "FAIL_BRANCH_REEXECUTION_MISMATCH",
        "checks": checks,
        "original_counts": {
            "solutions": len(original_solutions),
            "attempts": len(original_attempts),
            "evaluations": len(original_evaluations),
            "decisions": len(original_decisions),
        },
        "replay_counts": {
            "solutions": len(replay_solutions),
            "attempts": len(replay_attempts),
            "evaluations": len(replay_evaluations),
            "decisions": len(replay_decisions),
        },
        "terminal_bindings": {
            "original": original_terminal_binding,
            "replay": replay_terminal_binding,
        },
        "accounting": {
            "original": original_accounting,
            "replay": replay_accounting,
        },
        "first_mismatch": first_mismatch,
        "implementation_independence": False,
        "scientific_independence": False,
        "third_party_replication": False,
        "scope": "same_source_stochastic_program_reexecution_not_independent_replication",
    }


def reexecute_and_compare(
    *,
    original_database: str | Path,
    problem_artifact: str | Path,
    output_receipt: str | Path | None = None,
    source_manifest_path: str | Path | None = None,
) -> dict[str, object]:
    original = Path(original_database).resolve()
    if not original.is_file():
        raise FileNotFoundError(original)
    original_database_sha256_before = _sha256_file(original)
    problem_path = Path(problem_artifact).resolve()
    if not problem_path.is_file():
        raise FileNotFoundError(problem_path)
    output_path = None if output_receipt is None else Path(output_receipt).resolve()
    if output_path is not None and output_path.exists():
        raise FileExistsError(output_path)
    problem = load_v21e3_development_problem(problem_path)
    context = _load_context(original)
    candidate = str(context.get("candidate_id"))
    if candidate not in {"C0", "C1", "C2", "C3", "NSGAII", "MOEAD"}:
        raise ValueError(f"Unsupported candidate for branch replay: {candidate}")
    problem_binding = _validate_problem_binding(
        context,
        problem_artifact=problem_path,
        problem=problem,
    )
    source_binding = _validate_source_binding(
        context,
        candidate=candidate,
        source_manifest_path=source_manifest_path,
    )
    with tempfile.TemporaryDirectory(prefix="v21e3r1_branch_replay_") as temp:
        root = Path(temp)
        replay = root / "trace.sqlite3"
        terminal = root / "terminal.json"
        if candidate in {"C0", "C1", "C2", "C3"}:
            config = _hybrid_config(
                context,
                trace_path=replay,
                terminal_path=terminal,
                source_snapshot_sha256=source_binding[
                    "replay_source_snapshot_sha256"
                ],
                case_artifact_sha256=problem_binding[
                    "replay_case_artifact_sha256"
                ],
            )
            V21E3TypedHybridParetoSearch(problem, config).run()
        elif candidate in {"NSGAII", "MOEAD"}:
            config = _baseline_config(
                context,
                trace_path=replay,
                terminal_path=terminal,
                source_snapshot_sha256=source_binding[
                    "replay_source_snapshot_sha256"
                ],
                case_artifact_sha256=problem_binding[
                    "replay_case_artifact_sha256"
                ],
            )
            run_v21e3_development_baseline(problem, config)
        else:
            raise AssertionError("Candidate validation and dispatch disagree.")
        replay_context = _load_context(replay)
        replay_source_binding = _validate_source_binding(
            replay_context,
            candidate=candidate,
            source_manifest_path=source_manifest_path,
        )
        replay_problem_binding = _validate_problem_binding(
            replay_context,
            problem_artifact=problem_path,
            problem=problem,
        )
        if replay_context != context:
            raise ValueError("The replay run context differs from the original context.")
        if (
            replay_source_binding["context_source_sha256"]
            != source_binding["context_source_sha256"]
            or replay_source_binding["executing_module_sha256"]
            != source_binding["executing_module_sha256"]
            or replay_source_binding["source_manifest_sha256"]
            != source_binding["source_manifest_sha256"]
            or replay_problem_binding["context_case_artifact_sha256"]
            != problem_binding["context_case_artifact_sha256"]
            or replay_problem_binding["problem_artifact_sha256"]
            != problem_binding["problem_artifact_sha256"]
        ):
            raise ValueError("The replay source/problem bindings changed.")
        receipt = compare_trace_semantics(original, replay)
        original_database_sha256_after = _sha256_file(original)
        if original_database_sha256_after != original_database_sha256_before:
            raise ValueError("The original replay database changed during verification.")
        receipt["artifacts"] = {
            "original_database_bytes": original.stat().st_size,
            "original_database_sha256": original_database_sha256_after,
            "replay_database_bytes": replay.stat().st_size,
            "replay_database_sha256": _sha256_file(replay),
        }
        receipt["problem_binding"] = {
            key: value
            for key, value in problem_binding.items()
            if key != "replay_case_artifact_sha256"
        }
        receipt["source_binding"] = {
            key: value
            for key, value in source_binding.items()
            if key != "replay_source_snapshot_sha256"
        }
        receipt["source_binding"]["replay_verified"] = True
        context_binding = V21E3RunContext(context)
        receipt["run_context_binding"] = {
            "schema": context["schema"],
            "digest_sha256": context_binding.digest_sha256,
            "candidate_config_sha256": context["candidate_config_sha256"],
            "algorithm_source_sha256": context["algorithm_source_sha256"],
        }
        receipt["receipt_payload_sha256"] = hashlib.sha256(
            _canonical_bytes(receipt)
        ).hexdigest()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
            handle.write("\n")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--output")
    parser.add_argument("--source-manifest")
    args = parser.parse_args(argv)
    result = reexecute_and_compare(
        original_database=args.trace,
        problem_artifact=args.problem,
        output_receipt=args.output,
        source_manifest_path=args.source_manifest,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
