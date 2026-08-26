from __future__ import annotations

"""Run the V21e3r1 14-arm diagnostic on exposed development cases only.

Full mode is deliberately rigid: it consumes the exact frozen 12-case
development packet, three frozen seeds, all fourteen unique arms, and the
2000/200 evaluation/checkpoint schedule. A deliberately smaller invocation
must opt into ``--smoke`` and can never emit a full diagnostic PASS.

Every row runs in an isolated child process. Completed and failed attempts are
append-only, so an interrupted matrix can be resumed without overwriting a
trace. Selection, confirmation, and formal materialization are not exposed by
this program.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Mapping, Sequence

from mo_nco.pareto_v21e3_baselines import (
    load_v21e3_development_problem,
    run_v21e3_development_baseline,
)
from mo_nco.pareto_v21e3_hybrid import V21E3TypedHybridParetoSearch
from mo_nco.pareto_v21e3_parity import normalized_left_continuous_auc
from mo_nco.pareto_v21e3_trace_verify import verify_v21e3_trace_database
from mo_nco.pareto_v21e3r1_development_diagnostics import (
    DIAGNOSTIC_ARMS,
    DIAGNOSTIC_SCOPE,
    aggregate_diagnostic_matrix,
    analyze_trace_database,
    baseline_diagnostic_configs,
    hybrid_diagnostic_config,
)


SEEDS = (31051, 31057, 31059)
FULL_BUDGET = 2000
FULL_CHECKPOINT_PERIOD = 200
FULL_ROW_COUNT = 504
EXPECTED_CASE_IDS = (
    "v21e3-mokp-development-n100-s00",
    "v21e3-mokp-development-n100-s01",
    "v21e3-mokp-development-n200-s00",
    "v21e3-mokp-development-n200-s01",
    "v21e3-mokp-development-n500-s00",
    "v21e3-mokp-development-n500-s01",
    "v21e3-motsp-development-n100-s00",
    "v21e3-motsp-development-n100-s01",
    "v21e3-motsp-development-n200-s00",
    "v21e3-motsp-development-n200-s01",
    "v21e3-motsp-development-n500-s00",
    "v21e3-motsp-development-n500-s01",
)
INPUT_MANIFEST_SHA256 = {
    "ijoc_submission_v21e3/development_partitions_v1/case_manifest.json": (
        "1970361ba557aadd26de38aed008de11d11d158c797c00db1036cc4616cbdc8c"
    ),
    "ijoc_submission_v21e3/development_manifests_v1/reference_manifest_development.json": (
        "86336403c3e098f0e5022c796db1778552c0d92ca40d85953dc341eb534a4402"
    ),
    "ijoc_submission_v21e3/development_manifests_v1/config_manifest_development.json": (
        "d33ba2d83909af4fecff85f4663791b7c63b5ed56738a67f0eec6ccfd6336d4e"
    ),
}


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        handle.write("\n")


def _load_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _exact_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an exact integer >= {minimum}.")
    return value


def _exact_string_sequence(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} must be a nonempty sequence.")
    result = tuple(value)
    if any(type(item) is not str or not item for item in result):
        raise ValueError(f"{field} must contain exact nonempty strings.")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must not contain duplicates.")
    return result


def _seal_sqlite(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if row is None or int(row[0]) != 0:
            raise RuntimeError("WAL checkpoint failed.")
        mode = str(connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0])
        if mode.lower() != "delete":
            raise RuntimeError("SQLite did not enter DELETE journal mode.")
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("SQLite integrity_check failed.")
        connection.commit()
    finally:
        connection.close()


def _source_manifest(root: Path) -> dict[str, object]:
    files = sorted((root / "mo_nco").glob("*.py"))
    files.append(Path(__file__).resolve())
    files.append(
        (root / "independent_reproduction/recompute_v21e3r1_metrics.py").resolve()
    )
    entries: list[dict[str, object]] = []
    for path in sorted(set(files), key=lambda item: item.as_posix().lower()):
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError as error:
            raise RuntimeError("A diagnostic source escaped the project root.") from error
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    root_sha256 = hashlib.sha256(
        _canonical_json(entries).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "v21e3r1_diagnostic_source_manifest_v1",
        "hash_rule": "sha256(canonical_json(sorted_entries))",
        "entry_count": len(entries),
        "entries": entries,
        "source_snapshot_sha256": root_sha256,
    }


def _load_inputs(
    root: Path,
) -> tuple[
    list[dict[str, object]],
    dict[str, tuple[list[float], list[float]]],
    tuple[tuple[float, float], ...],
    dict[str, object],
]:
    manifests: dict[str, dict[str, object]] = {}
    observed_hashes: dict[str, str] = {}
    for relative, expected_hash in INPUT_MANIFEST_SHA256.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError("An input manifest escaped the project root.") from error
        observed_hash = _sha256(path)
        if observed_hash != expected_hash:
            raise RuntimeError(f"Frozen development manifest drifted: {relative}")
        manifests[relative] = _load_json_object(path)
        observed_hashes[relative] = observed_hash

    case_manifest = manifests[
        "ijoc_submission_v21e3/development_partitions_v1/case_manifest.json"
    ]
    reference_manifest = manifests[
        "ijoc_submission_v21e3/development_manifests_v1/reference_manifest_development.json"
    ]
    config_manifest = manifests[
        "ijoc_submission_v21e3/development_manifests_v1/config_manifest_development.json"
    ]
    if (
        case_manifest.get("split") != "development"
        or case_manifest.get("formal_confirmatory_eligibility") is not False
        or reference_manifest.get("formal_use") != "NOT_AUTHORIZED"
        or config_manifest.get("selection_partition") != "NOT_GENERATED"
    ):
        raise RuntimeError("Inputs are not the frozen exposed-development packet.")
    raw_cases = case_manifest.get("cases")
    if not isinstance(raw_cases, list) or not all(
        isinstance(case, dict) for case in raw_cases
    ):
        raise RuntimeError("Development case manifest has an invalid cases array.")
    cases = [dict(case) for case in raw_cases]
    case_ids = tuple(str(case.get("case_id")) for case in cases)
    if case_ids != EXPECTED_CASE_IDS:
        raise RuntimeError("The frozen exposed-development case IDs/order drifted.")

    reference_cases = reference_manifest.get("cases")
    if not isinstance(reference_cases, list):
        raise RuntimeError("Reference manifest has an invalid cases array.")
    bounds = {
        str(item["case_id"]): (
            [float(x) for x in item["objective_lower_bounds"]],
            [float(x) for x in item["objective_upper_bounds"]],
        )
        for item in reference_cases
        if isinstance(item, dict)
    }
    if tuple(bounds) != EXPECTED_CASE_IDS:
        raise RuntimeError("Reference bounds do not exactly cover the frozen cases.")
    raw_directions = config_manifest.get("reference_directions")
    if not isinstance(raw_directions, list):
        raise RuntimeError("Config manifest omits reference directions.")
    directions = tuple(tuple(float(x) for x in row) for row in raw_directions)
    if len(directions) != 21 or any(len(row) != 2 for row in directions):
        raise RuntimeError("The frozen C0 packet must contain 21 biobjective directions.")
    input_receipt = {
        "schema": "v21e3r1_exposed_development_input_binding_v1",
        "manifest_sha256": observed_hashes,
        "case_ids": list(EXPECTED_CASE_IDS),
    }
    return cases, bounds, directions, input_receipt


def _case_path(root: Path, case: Mapping[str, object]) -> Path:
    manifest_dir = (
        root / "ijoc_submission_v21e3/development_partitions_v1"
    ).resolve()
    artifact = case.get("artifact")
    if not isinstance(artifact, dict):
        raise RuntimeError("Case omits artifact binding.")
    relative = artifact.get("path")
    digest = artifact.get("sha256")
    if type(relative) is not str or type(digest) is not str:
        raise RuntimeError("Case artifact binding has invalid exact types.")
    path = (manifest_dir / relative).resolve()
    try:
        path.relative_to(manifest_dir)
    except ValueError as error:
        raise RuntimeError("A case artifact escaped the development root.") from error
    if not path.is_file() or _sha256(path) != digest:
        raise RuntimeError(f"Case artifact drifted: {case.get('case_id')}")
    return path


def _load_trace_context(database: Path) -> dict[str, object]:
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT run_context_json FROM run_attempt WHERE run_id=1"
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise RuntimeError("Trace omits run context.")
    payload = json.loads(str(row[0]))
    if not isinstance(payload, dict):
        raise RuntimeError("Trace run context is not an object.")
    return payload


def _assert_context(
    context: Mapping[str, object],
    *,
    algorithm_config: Mapping[str, object],
    case_sha256: str,
    source_sha256: str,
    seed: int,
    budget: int,
) -> None:
    source_binding = (
        "explicit_successor_source_snapshot_sha256_v1"
        if "adaptation_identity" in algorithm_config
        else "explicit_source_snapshot_or_release_manifest_sha256_v1"
    )
    required = {
        "case_artifact_sha256": case_sha256,
        "case_artifact_binding_kind": "explicit_case_artifact_sha256_v1",
        "algorithm_source_sha256": source_sha256,
        "algorithm_source_binding_kind": source_binding,
        "seed": seed,
        "charged_evaluation_budget": budget,
        "evidence_partition": "development",
    }
    for field, expected in required.items():
        if context.get(field) != expected:
            raise RuntimeError(f"Trace run context drifted at {field}.")
    if _canonical_json(context.get("algorithm_config")) != _canonical_json(
        dict(algorithm_config)
    ):
        raise RuntimeError("Trace algorithm_config disagrees with executed config.")
    expected_config_hash = hashlib.sha256(
        _canonical_json(dict(algorithm_config)).encode("utf-8")
    ).hexdigest()
    if context.get("candidate_config_sha256") != expected_config_hash:
        raise RuntimeError("Trace candidate_config_sha256 disagrees with config.")


def _independent_metric_replay(
    *,
    project_root: Path,
    trace: Path,
    lower: Sequence[float],
    upper: Sequence[float],
    budget: int,
    output: Path,
) -> dict[str, object]:
    script = (
        project_root / "independent_reproduction/recompute_v21e3r1_metrics.py"
    ).resolve()
    command = [
        sys.executable,
        str(script),
        "--trace",
        str(trace),
        "--lower=" + ",".join(repr(value) for value in lower),
        "--upper=" + ",".join(repr(value) for value in upper),
        "--expected-evaluations",
        str(budget),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=project_root,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Independent metric replay failed: " + completed.stderr[-2000:]
        )
    receipt = _load_json_object(output)
    if (
        receipt.get("status") != "PASS_INDEPENDENT_METRIC_IMPLEMENTATION"
        or receipt.get("evaluation_count") != budget
        or receipt.get("decision_count") != budget
        or receipt.get("terminal_accounting_gate") != "PASS"
        or receipt.get("trace_sha256") != _sha256(trace)
        or receipt.get("algorithm_execution_independence") is not False
        or receipt.get("scientific_independence") is not False
    ):
        raise RuntimeError("Independent metric receipt fails strict result gates.")
    return receipt


def _worker_run(spec_path: str | Path) -> dict[str, object]:
    spec_file = Path(spec_path).resolve()
    spec = _load_json_object(spec_file)
    attempt = spec_file.parent
    trace = attempt / "trace.sqlite3"
    terminal = attempt / "terminal.receipt.json"
    case_path = Path(str(spec["case_path"])).resolve()
    case_sha256 = str(spec["case_artifact_sha256"])
    source_sha256 = str(spec["source_snapshot_sha256"])
    project_root = Path(str(spec["project_root"])).resolve()
    if _source_manifest(project_root).get("source_snapshot_sha256") != source_sha256:
        raise RuntimeError("Diagnostic source tree drifted after plan freeze.")
    family = str(spec["family"])
    arm = str(spec["arm_id"])
    seed = _exact_int(spec.get("seed"), "seed")
    budget = _exact_int(spec.get("charged_evaluation_budget"), "budget", minimum=1)
    checkpoint = _exact_int(
        spec.get("checkpoint_period"), "checkpoint_period", minimum=1
    )
    if budget % checkpoint != 0:
        raise ValueError("Budget must be divisible by checkpoint_period.")
    lower = tuple(float(value) for value in spec["objective_lower_bounds"])
    upper = tuple(float(value) for value in spec["objective_upper_bounds"])
    directions = tuple(
        tuple(float(value) for value in row)
        for row in spec["reference_directions"]
    )
    if _sha256(case_path) != case_sha256:
        raise RuntimeError("Worker case artifact fails its SHA-256 binding.")
    problem = load_v21e3_development_problem(case_path)

    if arm.startswith("C0_"):
        config = hybrid_diagnostic_config(
            arm_id=arm,
            reference_directions=directions,
            charged_evaluations=budget,
            checkpoint_period=checkpoint,
            seed=seed,
            family=family,
            trace_database=str(trace),
            terminal_receipt=str(terminal),
        )
        config = replace(
            config,
            receipt_database_path="trace.sqlite3",
            capture_trace=False,
            case_artifact_sha256=case_sha256,
            source_snapshot_sha256=source_sha256,
        )
        optimizer = V21E3TypedHybridParetoSearch(problem, config)
        run = optimizer.run()
        del optimizer
    else:
        config = baseline_diagnostic_configs(
            family=family,
            arm_id=arm,
            charged_evaluations=budget,
            checkpoint_period=checkpoint,
            seed=seed,
        )
        config = replace(
            config,
            trace_database=str(trace),
            terminal_receipt=str(terminal),
            receipt_database_path="trace.sqlite3",
            capture_trace=False,
            case_artifact_sha256=case_sha256,
            source_snapshot_sha256=source_sha256,
        )
        run = run_v21e3_development_baseline(problem, config)

    algorithm_config = config.semantic_payload()
    auc, terminal_hv, checkpoints = normalized_left_continuous_auc(
        run.optimization_result.diagnostics,
        budget=budget,
        checkpoint_period=checkpoint,
        lower=lower,
        upper=upper,
    )
    _seal_sqlite(trace)
    context = _load_trace_context(trace)
    _assert_context(
        context,
        algorithm_config=algorithm_config,
        case_sha256=case_sha256,
        source_sha256=source_sha256,
        seed=seed,
        budget=budget,
    )
    terminal_sha256 = _sha256(terminal)
    verification = verify_v21e3_trace_database(
        trace,
        problem,
        expected_run_context=context,
        detached_terminal_receipt_path=terminal,
        expected_detached_terminal_receipt_sha256=terminal_sha256,
        expected_charged_evaluations=budget,
    )
    row = {
        "schema": "v21e3r1_exposed_development_diagnostic_row_v2",
        "status": "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "scientific_scope": DIAGNOSTIC_SCOPE,
        "case_id": str(spec["case_id"]),
        "family": family,
        "size": _exact_int(spec.get("size"), "size", minimum=1),
        "seed": seed,
        "arm_id": arm,
        "charged_evaluation_budget": budget,
        "checkpoint_period": checkpoint,
        "normalized_left_continuous_hv_auc": auc,
        "normalized_terminal_hv": terminal_hv,
        "checkpoints": list(checkpoints),
        "algorithm_config": algorithm_config,
        "case_artifact_sha256": case_sha256,
        "source_snapshot_sha256": source_sha256,
        "plan_sha256": str(spec["plan_sha256"]),
        "trace_database_path": "trace.sqlite3",
        "trace_database_sha256": _sha256(trace),
        "terminal_receipt_path": "terminal.receipt.json",
        "terminal_receipt_sha256": terminal_sha256,
        "strict_trace_verification": verification,
        "selection_entropy_release": "PROHIBITED",
        "confirmation_materialization": "PROHIBITED",
        "formal_materialization": "PROHIBITED",
    }
    diagnostic = analyze_trace_database(
        trace, row=row, lower=lower, upper=upper
    )
    if not math.isclose(
        float(diagnostic["terminal_hv_replayed"]),
        terminal_hv,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Checkpoint and exact replay terminal HV disagree.")
    independent_path = attempt / "independent.metric.json"
    independent = _independent_metric_replay(
        project_root=project_root,
        trace=trace,
        lower=lower,
        upper=upper,
        budget=budget,
        output=independent_path,
    )
    if not math.isclose(
        float(independent["exact_left_continuous_hv_auc"]),
        float(diagnostic["exact_per_evaluation_left_continuous_hv_auc"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        float(independent["terminal_hv"]),
        float(diagnostic["terminal_hv_replayed"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Independent and project metric implementations disagree.")
    row["independent_metric_receipt_path"] = "independent.metric.json"
    row["independent_metric_receipt_sha256"] = _sha256(independent_path)
    row["independent_metric_replay"] = independent
    _exclusive_json(attempt / "diagnostic.json", diagnostic)
    _exclusive_json(attempt / "row.json", row)
    return {
        "status": "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "row_sha256": _sha256(attempt / "row.json"),
        "diagnostic_sha256": _sha256(attempt / "diagnostic.json"),
        "trace_sha256": _sha256(trace),
        "terminal_receipt_sha256": terminal_sha256,
        "independent_metric_receipt_sha256": _sha256(independent_path),
    }


def _next_attempt_directory(output: Path, row_id: str) -> Path:
    root = output / "attempts" / row_id
    root.mkdir(parents=True, exist_ok=True)
    used: list[int] = []
    for path in root.glob("attempt-*"):
        try:
            used.append(int(path.name.removeprefix("attempt-")))
        except ValueError:
            continue
    attempt = root / f"attempt-{max(used, default=0) + 1:04d}"
    attempt.mkdir()
    return attempt


def _run_child(spec: Path, *, project_root: Path, timeout_seconds: int) -> None:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-spec", str(spec)]
    environment = dict(os.environ)
    current_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(project_root)
        if not current_path
        else str(project_root) + os.pathsep + current_path
    )
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        failure = {
            "schema": "v21e3r1_diagnostic_row_failure_v1",
            "status": "FAIL_ROW_TIMEOUT",
            "timeout_seconds": timeout_seconds,
            "stdout_tail": (error.stdout or "")[-4000:],
            "stderr_tail": (error.stderr or "")[-4000:],
        }
        _exclusive_json(spec.parent / "failure.receipt.json", failure)
        raise RuntimeError(f"Diagnostic row timed out: {spec}") from error
    if completed.returncode != 0:
        failure = {
            "schema": "v21e3r1_diagnostic_row_failure_v1",
            "status": "FAIL_ROW_PROCESS",
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
        _exclusive_json(spec.parent / "failure.receipt.json", failure)
        raise RuntimeError(f"Diagnostic row failed: {spec}")


def _completed_payload(output: Path, row_id: str) -> dict[str, object] | None:
    path = output / "completed" / f"{row_id}.json"
    if not path.is_file():
        return None
    payload = _load_json_object(path)
    if payload.get("status") != "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY":
        raise RuntimeError(f"Completed row has a non-PASS status: {row_id}")
    attempt = output / str(payload.get("attempt_directory"))
    try:
        attempt.resolve().relative_to(output)
    except ValueError as error:
        raise RuntimeError("Completed row points outside the output root.") from error
    for name, hash_field in (
        ("row.json", "row_sha256"),
        ("diagnostic.json", "diagnostic_sha256"),
        ("trace.sqlite3", "trace_sha256"),
        ("terminal.receipt.json", "terminal_receipt_sha256"),
        ("independent.metric.json", "independent_metric_receipt_sha256"),
    ):
        artifact = attempt / name
        if not artifact.is_file() or _sha256(artifact) != payload.get(hash_field):
            raise RuntimeError(f"Completed row artifact drifted: {row_id}/{name}")
    return payload


def run_matrix(
    project_root: str | Path,
    output_directory: str | Path,
    arms: Sequence[str] = DIAGNOSTIC_ARMS,
    *,
    case_ids: Sequence[str] | None = None,
    seeds: Sequence[int] = SEEDS,
    budget: int = FULL_BUDGET,
    checkpoint_period: int = FULL_CHECKPOINT_PERIOD,
    smoke: bool = False,
    resume: bool = False,
    row_timeout_seconds: int = 1800,
) -> dict[str, object]:
    root = Path(project_root).resolve()
    output = Path(output_directory).resolve()
    arms_tuple = _exact_string_sequence(arms, "arms")
    if any(arm not in DIAGNOSTIC_ARMS for arm in arms_tuple):
        raise ValueError("Unknown diagnostic arm.")
    seed_tuple = tuple(seeds)
    if not seed_tuple or any(type(seed) is not int or seed < 0 for seed in seed_tuple):
        raise ValueError("seeds must contain exact nonnegative integers.")
    if len(set(seed_tuple)) != len(seed_tuple):
        raise ValueError("seeds must not contain duplicates.")
    budget = _exact_int(budget, "budget", minimum=1)
    checkpoint_period = _exact_int(
        checkpoint_period, "checkpoint_period", minimum=1
    )
    row_timeout_seconds = _exact_int(
        row_timeout_seconds, "row_timeout_seconds", minimum=1
    )
    if budget % checkpoint_period != 0:
        raise ValueError("budget must be divisible by checkpoint_period.")

    cases, bounds, directions, input_binding = _load_inputs(root)
    requested_case_ids = (
        EXPECTED_CASE_IDS
        if case_ids is None
        else _exact_string_sequence(case_ids, "case_ids")
    )
    if any(case_id not in EXPECTED_CASE_IDS for case_id in requested_case_ids):
        raise ValueError("Only frozen exposed-development case IDs are allowed.")
    if not smoke and (
        requested_case_ids != EXPECTED_CASE_IDS
        or seed_tuple != SEEDS
        or arms_tuple != DIAGNOSTIC_ARMS
        or budget != FULL_BUDGET
        or checkpoint_period != FULL_CHECKPOINT_PERIOD
    ):
        raise ValueError(
            "Full mode requires the exact 12 cases, 3 seeds, 14 arms, and 2000/200 schedule; use --smoke for a subset."
        )
    selected = [case for case in cases if case.get("case_id") in requested_case_ids]
    if tuple(str(case["case_id"]) for case in selected) != requested_case_ids:
        raise ValueError("case_ids must preserve frozen manifest order.")

    source_manifest = _source_manifest(root)
    source_sha256 = str(source_manifest["source_snapshot_sha256"])
    expected_rows = len(selected) * len(seed_tuple) * len(arms_tuple)
    if not smoke and expected_rows != FULL_ROW_COUNT:
        raise RuntimeError("Full diagnostic row cardinality is not 504.")
    plan = {
        "schema": "v21e3r1_exposed_development_diagnostic_plan_v2",
        "status": (
            "FROZEN_FULL_504_DEVELOPMENT_DIAGNOSTIC"
            if not smoke
            else "FROZEN_DIAGNOSTIC_SMOKE_ONLY"
        ),
        "scientific_scope": DIAGNOSTIC_SCOPE,
        "case_ids": list(requested_case_ids),
        "seeds": list(seed_tuple),
        "arms": list(arms_tuple),
        "charged_evaluation_budget": budget,
        "checkpoint_period": checkpoint_period,
        "expected_rows": expected_rows,
        "input_binding": input_binding,
        "source_manifest": source_manifest,
        "row_timeout_seconds": row_timeout_seconds,
        "selection_entropy_release": "PROHIBITED",
        "confirmation_materialization": "PROHIBITED",
        "formal_materialization": "PROHIBITED",
    }
    plan_path = output / "diagnostic.plan.json"
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        if not plan_path.is_file() or _canonical_json(
            _load_json_object(plan_path)
        ) != _canonical_json(plan):
            raise RuntimeError("Resume plan disagrees with the frozen current plan.")
    else:
        output.mkdir(parents=True)
        _exclusive_json(plan_path, plan)
    plan_sha256 = _sha256(plan_path)
    if resume and (output / "diagnostic.receipt.json").is_file():
        receipt = _load_json_object(output / "diagnostic.receipt.json")
        aggregate_path = output / "diagnostic.aggregate.json"
        expected_status = (
            "PASS_DIAGNOSTIC_SMOKE_ONLY"
            if smoke
            else "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY"
        )
        if (
            not aggregate_path.is_file()
            or receipt.get("status") != expected_status
            or receipt.get("completed_rows") != expected_rows
            or receipt.get("expected_rows") != expected_rows
            or receipt.get("plan_sha256") != plan_sha256
            or receipt.get("source_snapshot_sha256") != source_sha256
            or receipt.get("aggregate_sha256") != _sha256(aggregate_path)
        ):
            raise RuntimeError("Completed diagnostic receipt fails resume validation.")
        for case in selected:
            case_id = str(case["case_id"])
            for seed in seed_tuple:
                for arm in arms_tuple:
                    row_id = f"{case_id}__seed-{seed}__arm-{arm.lower()}"
                    completed = _completed_payload(output, row_id)
                    if (
                        completed is None
                        or completed.get("row_id") != row_id
                        or completed.get("plan_sha256") != plan_sha256
                    ):
                        raise RuntimeError(
                            f"Completed diagnostic row fails resume validation: {row_id}"
                        )
        return receipt

    completed_rows: list[dict[str, object]] = []
    ordinal = 0
    for case in selected:
        case_id = str(case["case_id"])
        family = str(case["family"])
        size = _exact_int(case.get("size"), "case.size", minimum=1)
        case_path = _case_path(root, case)
        case_sha256 = _sha256(case_path)
        lower, upper = bounds[case_id]
        for seed in seed_tuple:
            for arm in arms_tuple:
                ordinal += 1
                row_id = f"{case_id}__seed-{seed}__arm-{arm.lower()}"
                completed = _completed_payload(output, row_id)
                if completed is None:
                    attempt = _next_attempt_directory(output, row_id)
                    spec = {
                        "schema": "v21e3r1_diagnostic_row_worker_spec_v1",
                        "project_root": str(root),
                        "case_id": case_id,
                        "family": family,
                        "size": size,
                        "case_path": str(case_path),
                        "case_artifact_sha256": case_sha256,
                        "objective_lower_bounds": lower,
                        "objective_upper_bounds": upper,
                        "reference_directions": directions,
                        "seed": seed,
                        "arm_id": arm,
                        "charged_evaluation_budget": budget,
                        "checkpoint_period": checkpoint_period,
                        "source_snapshot_sha256": source_sha256,
                        "plan_sha256": plan_sha256,
                    }
                    spec_path = attempt / "worker.spec.json"
                    _exclusive_json(spec_path, spec)
                    _run_child(
                        spec_path,
                        project_root=root,
                        timeout_seconds=row_timeout_seconds,
                    )
                    worker_result = _load_json_object(attempt / "worker.result.json")
                    completed = {
                        **worker_result,
                        "row_id": row_id,
                        "attempt_directory": attempt.relative_to(output).as_posix(),
                        "plan_sha256": plan_sha256,
                    }
                    _exclusive_json(output / "completed" / f"{row_id}.json", completed)
                attempt = output / str(completed["attempt_directory"])
                diagnostic = _load_json_object(attempt / "diagnostic.json")
                completed_rows.append(diagnostic)
                print(f"completed {ordinal}/{expected_rows} {row_id}", flush=True)

    if len(completed_rows) != expected_rows:
        raise RuntimeError("Diagnostic matrix is incomplete.")
    aggregate = aggregate_diagnostic_matrix(completed_rows)
    aggregate["matrix_mode"] = "SMOKE_ONLY" if smoke else "FULL_504"
    aggregate["plan_sha256"] = plan_sha256
    aggregate_path = output / "diagnostic.aggregate.json"
    receipt_path = output / "diagnostic.receipt.json"
    if aggregate_path.exists() or receipt_path.exists():
        raise FileExistsError("Final diagnostic artifacts already exist.")
    _exclusive_json(aggregate_path, aggregate)
    receipt = {
        "schema": "v21e3r1_exposed_development_diagnostic_receipt_v2",
        "status": (
            "PASS_DIAGNOSTIC_SMOKE_ONLY"
            if smoke
            else "PASS_DEVELOPMENT_DIAGNOSTIC_ONLY"
        ),
        "scientific_scope": DIAGNOSTIC_SCOPE,
        "matrix_mode": "SMOKE_ONLY" if smoke else "FULL_504",
        "completed_rows": len(completed_rows),
        "expected_rows": expected_rows,
        "plan_sha256": plan_sha256,
        "source_snapshot_sha256": source_sha256,
        "aggregate_sha256": _sha256(aggregate_path),
        "selection_entropy_release": "PROHIBITED",
        "confirmation_materialization": "PROHIBITED",
        "formal_materialization": "PROHIBITED",
    }
    _exclusive_json(receipt_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-directory")
    parser.add_argument("--arms", default=",".join(DIAGNOSTIC_ARMS))
    parser.add_argument("--case-ids")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--budget", type=int, default=FULL_BUDGET)
    parser.add_argument("--checkpoint-period", type=int, default=FULL_CHECKPOINT_PERIOD)
    parser.add_argument("--row-timeout-seconds", type=int, default=1800)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-spec", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_spec:
        result = _worker_run(args.worker_spec)
        _exclusive_json(Path(args.worker_spec).resolve().parent / "worker.result.json", result)
        print(json.dumps(result, sort_keys=True))
        return 0
    if not args.output_directory:
        parser.error("--output-directory is required")
    arms = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    cases = (
        None
        if args.case_ids is None
        else tuple(item.strip() for item in args.case_ids.split(",") if item.strip())
    )
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    result = run_matrix(
        args.project_root,
        args.output_directory,
        arms,
        case_ids=cases,
        seeds=seeds,
        budget=args.budget,
        checkpoint_period=args.checkpoint_period,
        smoke=args.smoke,
        resume=args.resume,
        row_timeout_seconds=args.row_timeout_seconds,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
