from __future__ import annotations

import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path
import sqlite3

import pytest

import mo_nco.pareto_v21e3_hybrid as hybrid_module
from mo_nco.pareto_v21e3_baselines import (
    frozen_development_baseline_configs,
    load_v21e3_development_problem,
    run_v21e3_development_baseline,
)
from mo_nco.pareto_v21e3_hybrid import (
    V21E3HybridConfig,
    V21E3TypedHybridParetoSearch,
)
from mo_nco.pareto_v21e3r1_branch_replay import (
    _hybrid_dataclass_kwargs,
    compare_trace_semantics,
    reexecute_and_compare,
)
from mo_nco.pareto_v21e3_trace import V21E3RunContext


CASE = Path(
    "ijoc_submission_v21e3/development_partitions_v1/instances/"
    "v21e3-mokp-development-n100-s00.json"
)
MOTSP_CASE = Path(
    "ijoc_submission_v21e3/development_partitions_v1/instances/"
    "v21e3-motsp-development-n100-s00.json"
)

_V9_DIAGNOSTIC_CASES = (
    (
        "V21E3R1_V9_LEGACY_MOKP",
        "MOKP",
        "disabled_v1",
        "bounded_reference_neighborhood_nonworse_replacement_v1",
        0.0,
    ),
    (
        "V21E3R1_V9_INFORMATION_SCREEN_MOKP",
        "MOKP",
        "bounded_cache_aware_structural_screen_development_v1",
        "bounded_reference_neighborhood_nonworse_replacement_v1",
        0.0,
    ),
    (
        "V21E3R1_V9_LYAPUNOV_MOKP",
        "MOKP",
        "disabled_v1",
        "archive_compensated_information_lyapunov_development_v1",
        0.5,
    ),
    (
        "V21E3R1_V9_INFORMATION_LYAPUNOV_MOKP",
        "MOKP",
        "bounded_cache_aware_structural_screen_development_v1",
        "archive_compensated_information_lyapunov_development_v1",
        0.5,
    ),
    (
        "V21E3R1_V9_LEGACY_MOTSP",
        "MOTSP",
        "disabled_v1",
        "bounded_reference_neighborhood_nonworse_replacement_v1",
        0.0,
    ),
    (
        "V21E3R1_V9_INFORMATION_SCREEN_MOTSP",
        "MOTSP",
        "bounded_cache_aware_structural_screen_development_v1",
        "bounded_reference_neighborhood_nonworse_replacement_v1",
        0.0,
    ),
    (
        "V21E3R1_V9_LYAPUNOV_MOTSP",
        "MOTSP",
        "disabled_v1",
        "archive_compensated_information_lyapunov_development_v1",
        0.5,
    ),
    (
        "V21E3R1_V9_INFORMATION_LYAPUNOV_MOTSP",
        "MOTSP",
        "bounded_cache_aware_structural_screen_development_v1",
        "archive_compensated_information_lyapunov_development_v1",
        0.5,
    ),
)


def _v9_resource_limits(screening_policy: str) -> dict[str, object]:
    """Supply the V9 exact resource fields once the config exposes them."""

    config_fields = {field.name for field in fields(V21E3HybridConfig)}
    values: dict[str, object] = {
        "attempt_cap": 256,
        "structural_screening_cap": (
            0 if screening_policy == "disabled_v1" else 4096
        ),
        "wall_time_cap_seconds": 120.0,
    }
    return {key: value for key, value in values.items() if key in config_fields}


def _write_hybrid_trace(
    root: Path,
    *,
    source_snapshot_sha256: str | None = None,
    successor_arm: str | None = None,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    problem = load_v21e3_development_problem(CASE)
    trace = root / "trace.sqlite3"
    terminal = root / "terminal.json"
    config = V21E3HybridConfig(
        candidate_id="C0",
        reference_directions=((0.2, 0.8), (0.5, 0.5), (0.8, 0.2)),
        charged_evaluations=24,
        checkpoint_period=6,
        seed=811,
        phase="development",
        trace_database=str(trace),
        terminal_receipt=str(terminal),
        receipt_database_path="trace.sqlite3",
        source_snapshot_sha256=source_snapshot_sha256,
        local_improvement_steps=1,
        development_diagnostic_id=(
            None
            if successor_arm is None
            else f"V21E3R1_SUCCESSOR_FACTORIAL_{successor_arm}"
        ),
        post_initialization_search_policy=(
            "post_commit_type_incumbent_anchor_development_v1"
            if successor_arm in {"MOKP_ANCHOR_ONLY", "MOKP_BOTH"}
            else "proposal_chain_v21e3r1_v1"
        ),
        mokp_novelty_generation_policy=(
            "single_attempt_rotating_feasible_exchange_no_refill_development_v1"
            if successor_arm in {"MOKP_NOVELTY_ONLY", "MOKP_BOTH"}
            else "legacy_retry_and_local_v21e3r1_v1"
        ),
    )
    V21E3TypedHybridParetoSearch(problem, config).run()
    return trace, terminal


def _v9_config(
    *,
    diagnostic_id: str,
    screening_policy: str,
    replacement_policy: str,
    tradeoff_lambda: float,
    trace: Path | None = None,
    terminal: Path | None = None,
) -> V21E3HybridConfig:
    return V21E3HybridConfig(
        candidate_id="C0",
        reference_directions=((0.25, 0.75), (0.75, 0.25)),
        charged_evaluations=8,
        checkpoint_period=4,
        seed=1701,
        phase="development",
        trace_database=None if trace is None else str(trace),
        terminal_receipt=None if terminal is None else str(terminal),
        receipt_database_path=None if trace is None else "trace.sqlite3",
        capture_trace=False,
        local_improvement_steps=1,
        development_diagnostic_id=diagnostic_id,
        candidate_screening_policy=screening_policy,
        candidate_screening_cap=4,
        replacement_policy=replacement_policy,
        archive_tradeoff_lambda=tradeoff_lambda,
        **_v9_resource_limits(screening_policy),
    )


def _write_v9_trace(
    root: Path,
    *,
    family: str,
    diagnostic_id: str,
    screening_policy: str,
    replacement_policy: str,
    tradeoff_lambda: float,
) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    case = CASE if family == "MOKP" else MOTSP_CASE
    problem = load_v21e3_development_problem(case)
    trace = root / "trace.sqlite3"
    terminal = root / "terminal.json"
    config = _v9_config(
        diagnostic_id=diagnostic_id,
        screening_policy=screening_policy,
        replacement_policy=replacement_policy,
        tradeoff_lambda=tradeoff_lambda,
        trace=trace,
        terminal=terminal,
    )
    V21E3TypedHybridParetoSearch(problem, config).run()
    return trace, terminal, case


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_file_bytes(payload: object) -> bytes:
    return _canonical_bytes(payload) + b"\n"


def _live_mo_nco_python_entries() -> list[dict[str, object]]:
    package_root = Path(hybrid_module.__file__).resolve().parent
    project_root = package_root.parent
    entries: list[dict[str, object]] = []
    for path in sorted(package_root.rglob("*.py"), key=lambda item: item.as_posix()):
        raw = path.read_bytes()
        entries.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return entries


@pytest.mark.parametrize(
    "diagnostic_id,family,screening,replacement,tradeoff_lambda",
    _V9_DIAGNOSTIC_CASES,
)
def test_replay_semantic_kwargs_accept_all_v9_diagnostic_ids(
    diagnostic_id: str,
    family: str,
    screening: str,
    replacement: str,
    tradeoff_lambda: float,
) -> None:
    del family
    config = _v9_config(
        diagnostic_id=diagnostic_id,
        screening_policy=screening,
        replacement_policy=replacement,
        tradeoff_lambda=tradeoff_lambda,
    )

    kwargs = _hybrid_dataclass_kwargs(config.semantic_payload())

    assert kwargs["development_diagnostic_id"] == diagnostic_id
    assert kwargs["candidate_screening_policy"] == screening
    assert kwargs["replacement_policy"] == replacement
    assert kwargs["archive_tradeoff_lambda"] == pytest.approx(tradeoff_lambda)
    for key, value in _v9_resource_limits(screening).items():
        assert kwargs[key] == value


@pytest.mark.parametrize(
    "missing_field",
    ("attempt_cap", "structural_screening_cap", "wall_time_cap_seconds"),
)
def test_replay_v9_semantic_kwargs_fail_closed_on_missing_resource_limit(
    missing_field: str,
) -> None:
    config = _v9_config(
        diagnostic_id="V21E3R1_V9_INFORMATION_LYAPUNOV_MOKP",
        screening_policy="bounded_cache_aware_structural_screen_development_v1",
        replacement_policy=(
            "archive_compensated_information_lyapunov_development_v1"
        ),
        tradeoff_lambda=0.5,
    )
    payload = config.semantic_payload()
    payload.pop(missing_field)

    with pytest.raises(ValueError, match="exact config keys"):
        _hybrid_dataclass_kwargs(payload)


@pytest.mark.parametrize(
    "diagnostic_id,family,screening,replacement,tradeoff_lambda",
    _V9_DIAGNOSTIC_CASES,
)
def test_replay_reexecutes_all_v9_diagnostic_ids(
    tmp_path: Path,
    diagnostic_id: str,
    family: str,
    screening: str,
    replacement: str,
    tradeoff_lambda: float,
) -> None:
    trace, _terminal, case = _write_v9_trace(
        tmp_path / diagnostic_id.lower(),
        family=family,
        diagnostic_id=diagnostic_id,
        screening_policy=screening,
        replacement_policy=replacement,
        tradeoff_lambda=tradeoff_lambda,
    )

    result = reexecute_and_compare(
        original_database=trace,
        problem_artifact=case,
    )

    assert result["status"] == "PASS_SAME_IMPLEMENTATION_FULL_BRANCH_REEXECUTION"
    assert all(result["checks"].values())


def _write_baseline_trace(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    problem = load_v21e3_development_problem(CASE)
    trace = root / "trace.sqlite3"
    terminal = root / "terminal.json"
    config = frozen_development_baseline_configs(
        family="MOKP",
        charged_evaluations=48,
        checkpoint_period=8,
        seed=907,
    )["NSGAII"]
    config = replace(
        config,
        trace_database=str(trace),
        terminal_receipt=str(terminal),
        receipt_database_path="trace.sqlite3",
        capture_trace=False,
    )
    run_v21e3_development_baseline(problem, config)
    return trace


def test_replay_rejects_unverifiable_explicit_source_snapshot(tmp_path: Path) -> None:
    trace, _terminal = _write_hybrid_trace(
        tmp_path / "original",
        source_snapshot_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="source manifest"):
        reexecute_and_compare(
            original_database=trace,
            problem_artifact=CASE,
        )


def test_replay_passes_reserved_path_and_binds_complete_semantics(
    tmp_path: Path,
) -> None:
    trace, _terminal = _write_hybrid_trace(
        tmp_path / "reserved # percent% original",
    )
    output = tmp_path / "receipts" / "branch replay.json"

    result = reexecute_and_compare(
        original_database=trace,
        problem_artifact=CASE,
        output_receipt=output,
    )

    assert result["status"] == "PASS_SAME_IMPLEMENTATION_FULL_BRANCH_REEXECUTION"
    assert set(result["checks"]) == {
        "run_context",
        "solutions",
        "attempts",
        "evaluations",
        "decisions",
        "archive",
        "terminal",
        "accounting",
    }
    assert all(result["checks"].values())
    assert result["artifacts"]["original_database_sha256"]
    assert result["artifacts"]["replay_database_sha256"]
    assert result["problem_binding"]["problem_artifact_sha256"]
    assert result["problem_binding"]["problem_semantic_sha256"]
    assert result["run_context_binding"]["digest_sha256"]
    assert result["source_binding"]["source_manifest"] is None
    assert result["implementation_independence"] is False
    assert output.is_file()


def test_replay_covers_exact_successor_policy_context(tmp_path: Path) -> None:
    trace, _terminal = _write_hybrid_trace(
        tmp_path / "successor original",
        successor_arm="MOKP_BOTH",
    )

    result = reexecute_and_compare(
        original_database=trace,
        problem_artifact=CASE,
    )

    assert result["status"] == "PASS_SAME_IMPLEMENTATION_FULL_BRANCH_REEXECUTION"
    assert all(result["checks"].values())
    with sqlite3.connect(trace) as connection:
        raw = connection.execute(
            "SELECT run_context_json FROM run_attempt WHERE run_id=1"
        ).fetchone()[0]
    config = json.loads(str(raw))["algorithm_config"]
    assert config["post_initialization_search_policy"] == (
        "post_commit_type_incumbent_anchor_development_v1"
    )
    assert config["mokp_novelty_generation_policy"] == (
        "single_attempt_rotating_feasible_exchange_no_refill_development_v1"
    )


def test_replay_rejects_partial_successor_policy_context(tmp_path: Path) -> None:
    trace, _terminal = _write_hybrid_trace(
        tmp_path / "successor original",
        successor_arm="MOKP_BOTH",
    )
    connection = sqlite3.connect(trace)
    try:
        raw = connection.execute(
            "SELECT run_context_json FROM run_attempt WHERE run_id=1"
        ).fetchone()[0]
        context = json.loads(str(raw))
        context["algorithm_config"].pop("mokp_novelty_generation_policy")
        context["candidate_config_sha256"] = hashlib.sha256(
            _canonical_bytes(context["algorithm_config"])
        ).hexdigest()
        rebound = V21E3RunContext(context)
        connection.execute(
            "UPDATE run_attempt SET run_context_json=?,run_context_digest_sha256=? "
            "WHERE run_id=1",
            (rebound.canonical_json, rebound.digest_sha256),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError, match="exact config keys"):
        reexecute_and_compare(
            original_database=trace,
            problem_artifact=CASE,
        )


def test_replay_rejects_coherently_rehashed_extra_config_key(tmp_path: Path) -> None:
    trace, _terminal = _write_hybrid_trace(tmp_path / "original")
    connection = sqlite3.connect(trace)
    try:
        raw = connection.execute(
            "SELECT run_context_json FROM run_attempt WHERE run_id=1"
        ).fetchone()[0]
        context = json.loads(str(raw))
        context["algorithm_config"]["unexpected_semantic_key"] = "forbidden"
        context["candidate_config_sha256"] = hashlib.sha256(
            _canonical_bytes(context["algorithm_config"])
        ).hexdigest()
        rebound = V21E3RunContext(context)
        connection.execute(
            "UPDATE run_attempt SET run_context_json=?,run_context_digest_sha256=? "
            "WHERE run_id=1",
            (rebound.canonical_json, rebound.digest_sha256),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError, match="exact config keys"):
        reexecute_and_compare(
            original_database=trace,
            problem_artifact=CASE,
        )


def test_replay_rejects_successor_policy_fields_in_legacy_context(
    tmp_path: Path,
) -> None:
    trace, _terminal = _write_hybrid_trace(tmp_path / "legacy original")
    connection = sqlite3.connect(trace)
    try:
        raw = connection.execute(
            "SELECT run_context_json FROM run_attempt WHERE run_id=1"
        ).fetchone()[0]
        context = json.loads(str(raw))
        context["algorithm_config"].update(
            {
                "post_initialization_search_policy": "proposal_chain_v21e3r1_v1",
                "mokp_novelty_generation_policy": (
                    "legacy_retry_and_local_v21e3r1_v1"
                ),
            }
        )
        context["candidate_config_sha256"] = hashlib.sha256(
            _canonical_bytes(context["algorithm_config"])
        ).hexdigest()
        rebound = V21E3RunContext(context)
        connection.execute(
            "UPDATE run_attempt SET run_context_json=?,run_context_digest_sha256=? "
            "WHERE run_id=1",
            (rebound.canonical_json, rebound.digest_sha256),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError, match="exact config keys"):
        reexecute_and_compare(
            original_database=trace,
            problem_artifact=CASE,
        )


def test_replay_covers_baseline_branch_with_exact_config(tmp_path: Path) -> None:
    trace = _write_baseline_trace(tmp_path / "baseline original")

    result = reexecute_and_compare(
        original_database=trace,
        problem_artifact=CASE,
    )

    assert result["status"] == "PASS_SAME_IMPLEMENTATION_FULL_BRANCH_REEXECUTION"
    assert all(result["checks"].values())
    assert result["source_binding"]["binding_kind"] == (
        "development_adapter_module_sha256_fallback_pre_snapshot_v1"
    )


def test_replay_rejects_manifest_missing_mo_nco_python_dependency(
    tmp_path: Path,
) -> None:
    entries = [
        entry
        for entry in _live_mo_nco_python_entries()
        if entry["path"] != "mo_nco/pareto_v21e3r1_v9_theory.py"
    ]
    source_root = hashlib.sha256(_canonical_bytes(entries)).hexdigest()
    manifest = {
        "schema": "test_source_manifest_v1",
        "files_root_sha256": source_root,
        "files": entries,
    }
    manifest_path = tmp_path / "missing dependency source manifest.json"
    manifest_path.write_bytes(_canonical_json_file_bytes(manifest))
    trace, _terminal = _write_hybrid_trace(
        tmp_path / "explicit original",
        source_snapshot_sha256=source_root,
    )

    with pytest.raises(ValueError, match="missing live mo_nco Python source"):
        reexecute_and_compare(
            original_database=trace,
            problem_artifact=CASE,
            source_manifest_path=manifest_path,
        )


def test_replay_rejects_manifest_with_extra_mo_nco_python_source(
    tmp_path: Path,
) -> None:
    entries = _live_mo_nco_python_entries()
    entries.append(
        {
            "path": "mo_nco/not_present_in_live_tree.py",
            "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        }
    )
    source_root = hashlib.sha256(_canonical_bytes(entries)).hexdigest()
    manifest = {
        "schema": "test_source_manifest_v1",
        "files_root_sha256": source_root,
        "files": entries,
    }
    manifest_path = tmp_path / "extra source manifest.json"
    manifest_path.write_bytes(_canonical_json_file_bytes(manifest))
    trace, _terminal = _write_hybrid_trace(
        tmp_path / "explicit original",
        source_snapshot_sha256=source_root,
    )

    with pytest.raises(ValueError, match="extra frozen mo_nco Python source"):
        reexecute_and_compare(
            original_database=trace,
            problem_artifact=CASE,
            source_manifest_path=manifest_path,
        )


def test_replay_rejects_manifest_with_mo_nco_dependency_hash_drift(
    tmp_path: Path,
) -> None:
    entries = _live_mo_nco_python_entries()
    drifted_path = "mo_nco/pareto_v21e3r1_v9_theory.py"
    for entry in entries:
        if entry["path"] == drifted_path:
            entry["sha256"] = "0" * 64
            break
    else:
        raise AssertionError(f"Test dependency is absent: {drifted_path}")
    source_root = hashlib.sha256(_canonical_bytes(entries)).hexdigest()
    manifest = {
        "schema": "test_source_manifest_v1",
        "files_root_sha256": source_root,
        "files": entries,
    }
    manifest_path = tmp_path / "dependency hash drift source manifest.json"
    manifest_path.write_bytes(_canonical_json_file_bytes(manifest))
    trace, _terminal = _write_hybrid_trace(
        tmp_path / "explicit original",
        source_snapshot_sha256=source_root,
    )

    with pytest.raises(ValueError, match="mo_nco Python source hash drift"):
        reexecute_and_compare(
            original_database=trace,
            problem_artifact=CASE,
            source_manifest_path=manifest_path,
        )


def test_replay_accepts_manifest_that_binds_entire_mo_nco_python_closure(
    tmp_path: Path,
) -> None:
    module_path = Path(hybrid_module.__file__).resolve()
    module_sha = hashlib.sha256(module_path.read_bytes()).hexdigest()
    entries = _live_mo_nco_python_entries()
    source_root = hashlib.sha256(_canonical_bytes(entries)).hexdigest()
    manifest = {
        "schema": "test_source_manifest_v1",
        "source_root_sha256": source_root,
        "files": entries,
    }
    manifest_path = tmp_path / "source manifest.json"
    manifest_path.write_bytes(_canonical_json_file_bytes(manifest))
    trace, _terminal = _write_hybrid_trace(
        tmp_path / "explicit original",
        source_snapshot_sha256=source_root,
    )

    result = reexecute_and_compare(
        original_database=trace,
        problem_artifact=CASE,
        source_manifest_path=manifest_path,
    )

    assert result["status"] == "PASS_SAME_IMPLEMENTATION_FULL_BRANCH_REEXECUTION"
    assert result["source_binding"]["context_source_sha256"] == source_root
    assert result["source_binding"]["executing_module_sha256"] == module_sha
    assert result["source_binding"]["source_closure_scope"] == (
        "all_live_mo_nco_python_sources"
    )
    assert result["source_binding"]["source_closure_file_count"] == len(entries)
    assert result["source_binding"]["source_closure_verified"] is True
    assert result["implementation_independence"] is False
    assert result["source_binding"]["source_manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    ("malformation", "error_match"),
    (
        ("duplicate_key", "duplicate JSON key"),
        ("nonfinite_constant", "non-finite JSON constant"),
        ("noncanonical_encoding", "canonical JSON"),
    ),
)
def test_replay_rejects_non_strict_source_manifest_json(
    tmp_path: Path,
    malformation: str,
    error_match: str,
) -> None:
    entries = _live_mo_nco_python_entries()
    source_root = hashlib.sha256(_canonical_bytes(entries)).hexdigest()
    manifest = {
        "schema": "test_source_manifest_v1",
        "files_root_sha256": source_root,
        "files": entries,
    }
    canonical = _canonical_json_file_bytes(manifest)
    if malformation == "duplicate_key":
        raw = b'{"files":[],' + canonical[1:]
    elif malformation == "nonfinite_constant":
        raw = canonical[:-2] + b',"probe":NaN}\n'
    elif malformation == "noncanonical_encoding":
        raw = (
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False).encode(
                "utf-8"
            )
            + b"\n"
        )
    else:
        raise AssertionError(f"unknown malformation: {malformation}")
    manifest_path = tmp_path / f"{malformation} source manifest.json"
    manifest_path.write_bytes(raw)
    trace, _terminal = _write_hybrid_trace(
        tmp_path / f"explicit original {malformation}",
        source_snapshot_sha256=source_root,
    )

    with pytest.raises(ValueError, match=error_match):
        reexecute_and_compare(
            original_database=trace,
            problem_artifact=CASE,
            source_manifest_path=manifest_path,
        )


def test_replay_rejects_module_only_in_unbound_manifest_inventory(
    tmp_path: Path,
) -> None:
    module_path = Path(hybrid_module.__file__).resolve()
    module_sha = hashlib.sha256(module_path.read_bytes()).hexdigest()
    bound_files = [
        {
            "path": "README.md",
            "sha256": "0" * 64,
        }
    ]
    source_root = hashlib.sha256(_canonical_bytes(bound_files)).hexdigest()
    manifest = {
        "schema": "test_source_manifest_v1",
        "bound_files_root_sha256": source_root,
        "bound_files": bound_files,
        "entries": [
            {
                "path": "mo_nco/pareto_v21e3_hybrid.py",
                "sha256": module_sha,
            }
        ],
    }
    manifest_path = tmp_path / "split source manifest.json"
    manifest_path.write_bytes(_canonical_json_file_bytes(manifest))
    trace, _terminal = _write_hybrid_trace(
        tmp_path / "explicit original",
        source_snapshot_sha256=source_root,
    )

    with pytest.raises(ValueError, match="executing module"):
        reexecute_and_compare(
            original_database=trace,
            problem_artifact=CASE,
            source_manifest_path=manifest_path,
        )


def test_compare_reports_packed_solution_tamper(tmp_path: Path) -> None:
    original, _terminal = _write_hybrid_trace(tmp_path / "original")
    replay, _terminal = _write_hybrid_trace(tmp_path / "replay")
    connection = sqlite3.connect(replay)
    try:
        solution_ref, payload = connection.execute(
            "SELECT solution_ref,payload FROM solutions ORDER BY solution_ref LIMIT 1"
        ).fetchone()
        raw = bytes(payload)
        tampered = bytes((raw[0] ^ 1,)) + raw[1:]
        connection.execute(
            "UPDATE solutions SET payload=? WHERE solution_ref=?",
            (tampered, solution_ref),
        )
        connection.commit()
    finally:
        connection.close()

    result = compare_trace_semantics(original, replay)

    assert result["status"] == "FAIL_BRANCH_REEXECUTION_MISMATCH"
    assert result["checks"]["solutions"] is False
    assert result["first_mismatch"]["semantic_group"] == "solutions"


def test_replay_receipt_is_exclusive_create(tmp_path: Path) -> None:
    trace, _terminal = _write_hybrid_trace(tmp_path / "original")
    output = tmp_path / "existing receipt.json"
    output.write_bytes(b"preexisting-custody-bytes")

    with pytest.raises(FileExistsError):
        reexecute_and_compare(
            original_database=trace,
            problem_artifact=CASE,
            output_receipt=output,
        )

    assert output.read_bytes() == b"preexisting-custody-bytes"


def test_replay_rejects_coherently_rehashed_terminal_accounting_tamper(
    tmp_path: Path,
) -> None:
    trace, _terminal = _write_hybrid_trace(tmp_path / "original")
    connection = sqlite3.connect(trace)
    try:
        raw = connection.execute(
            "SELECT receipt_json FROM terminal_receipts WHERE run_id=1"
        ).fetchone()[0]
        receipt = json.loads(str(raw))
        receipt["attempt_count"] += 1
        receipt.pop("receipt_payload_sha256")
        receipt_sha = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
        receipt["receipt_payload_sha256"] = receipt_sha
        receipt_json = _canonical_bytes(receipt).decode("utf-8")
        connection.execute(
            "UPDATE terminal_receipts SET receipt_json=?,receipt_sha256=? WHERE run_id=1",
            (receipt_json, receipt_sha),
        )
        connection.execute(
            "UPDATE run_attempt SET terminal_receipt_sha256=? WHERE run_id=1",
            (receipt_sha,),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError, match="observed accounting"):
        reexecute_and_compare(
            original_database=trace,
            problem_artifact=CASE,
        )
