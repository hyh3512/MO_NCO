from __future__ import annotations

"""Fail-closed development diagnostics for V21e3r1 V9 traces.

The analyzer never trusts a policy witness as a quality measurement. It opens
one terminal SQLite trace read-only, validates the durable accounting, and
rebuilds the all-evaluated biobjective archive in charged-evaluation order.
The resulting report remains same-project, descriptive engineering evidence;
it is not an independent reproduction and authorizes no later evidence phase.
"""

from collections.abc import Mapping, Sequence
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any

from .pareto_v21e3r1_v9_theory import operator_productivity


_SCREENING_POLICY = "bounded_cache_aware_structural_screen_development_v1"
_LEGACY_REPLACEMENT = "bounded_reference_neighborhood_nonworse_replacement_v1"
_LYAPUNOV_REPLACEMENT = (
    "archive_compensated_information_lyapunov_development_v1"
)
_SCREEN_WITNESS_SCHEMA = "v21e3r1_information_time_candidate_screen_v2"

# diagnostic id -> (family, screening policy, replacement policy, lambda positive)
_V9_ARMS: dict[str, tuple[str, str, str, bool]] = {
    f"V21E3R1_V9_{arm}_{family}": (
        family,
        (_SCREENING_POLICY if "INFORMATION" in arm else "disabled_v1"),
        (_LYAPUNOV_REPLACEMENT if "LYAPUNOV" in arm else _LEGACY_REPLACEMENT),
        "LYAPUNOV" in arm,
    )
    for family in ("MOKP", "MOTSP")
    for arm in (
        "LEGACY",
        "INFORMATION_SCREEN",
        "LYAPUNOV",
        "INFORMATION_LYAPUNOV",
    )
}

_REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "attempts": frozenset(
        {
            "attempt_index",
            "proposal_solution_ref",
            "proposal_sha256",
            "proposal_json",
            "proposal_raw_sha256",
            "context_json",
            "status",
            "physical_call_started",
            "charged_evaluation_index",
            "cache_source_evaluation_index",
            "failure_code",
            "failure_detail_json",
            "prev_attempt_sha256",
            "attempt_sha256",
        }
    ),
    "evaluations": frozenset(
        {
            "evaluation_index",
            "attempt_index",
            "evidence_partition",
            "search_phase_id",
            "stage_id",
            "type_id",
            "operator_id",
            "operator_call_id",
            "proposal_solution_ref",
            "proposal_sha256",
            "objectives_json",
            "prev_record_sha256",
            "record_sha256",
        }
    ),
    "decisions": frozenset(
        {
            "evaluation_index",
            "decision_json",
            "prev_decision_sha256",
            "decision_sha256",
        }
    ),
    "run_attempt": frozenset(
        {
            "run_id",
            "family",
            "run_context_json",
            "run_context_digest_sha256",
            "status",
            "terminal_receipt_sha256",
        }
    ),
    "terminal_receipts": frozenset(
        {"run_id", "status", "receipt_json", "receipt_sha256"}
    ),
}


class V9TraceDiagnosticError(RuntimeError):
    """The persisted V9 trace cannot support a fail-closed diagnostic."""


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise V9TraceDiagnosticError("A diagnostic payload is not canonical JSON.") from error


def _reject_json_constant(value: str) -> object:
    raise V9TraceDiagnosticError(f"JSON contains forbidden non-finite value {value}.")


def _parse_json(raw: object, *, label: str) -> object:
    if type(raw) is not str:
        raise V9TraceDiagnosticError(f"{label} is not persisted JSON text.")
    try:
        return json.loads(raw, parse_constant=_reject_json_constant)
    except V9TraceDiagnosticError:
        raise
    except (json.JSONDecodeError, TypeError) as error:
        raise V9TraceDiagnosticError(f"{label} is invalid JSON.") from error


def _parse_json_object(raw: object, *, label: str) -> dict[str, object]:
    value = _parse_json(raw, label=label)
    if not isinstance(value, dict):
        raise V9TraceDiagnosticError(f"{label} must be a JSON object.")
    return dict(value)


def _exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or int(value) < minimum:
        raise V9TraceDiagnosticError(
            f"{label} must be an exact integer not less than {minimum}."
        )
    return int(value)


def _finite_real(value: object, *, label: str) -> float:
    if type(value) not in {int, float}:
        raise V9TraceDiagnosticError(f"{label} must be an exact finite real.")
    result = float(value)
    if not math.isfinite(result):
        raise V9TraceDiagnosticError(f"{label} must be an exact finite real.")
    return result


def _optional_exact_int(
    value: object, *, label: str, minimum: int = 0
) -> int | None:
    if value is None:
        return None
    return _exact_int(value, label=label, minimum=minimum)


def _require_columns(connection: sqlite3.Connection) -> None:
    for table, required in _REQUIRED_COLUMNS.items():
        rows = list(connection.execute(f'PRAGMA table_info("{table}")'))
        observed = {str(row[1]) for row in rows}
        missing = sorted(required - observed)
        if missing:
            raise V9TraceDiagnosticError(
                f"SQLite table {table} misses required columns: {missing}."
            )


def _detached_terminal_receipt(
    database_path: Path,
    *,
    detached_terminal_receipt_path: str | Path | None,
    expected_detached_terminal_receipt_sha256: str | None,
) -> tuple[Path, str, dict[str, object], str, bool]:
    detached_path = (
        database_path.with_name("terminal.json")
        if detached_terminal_receipt_path is None
        else Path(detached_terminal_receipt_path).resolve()
    )
    if not detached_path.is_file():
        raise V9TraceDiagnosticError(
            f"V9 detached terminal receipt is missing: {detached_path}."
        )
    raw = detached_path.read_bytes()
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    externally_bound = expected_detached_terminal_receipt_sha256 is not None
    if externally_bound:
        expected_sha256 = str(expected_detached_terminal_receipt_sha256)
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise V9TraceDiagnosticError(
                "Expected detached terminal receipt SHA-256 must be lowercase hex."
            )
        if observed_sha256 != expected_sha256:
            raise V9TraceDiagnosticError(
                "V9 detached terminal receipt fails its external SHA-256 binding."
            )
    try:
        raw_text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise V9TraceDiagnosticError(
            "V9 detached terminal receipt is not UTF-8 JSON."
        ) from error
    payload = _parse_json_object(raw_text, label="detached terminal receipt")
    if _canonical_json(payload) != raw_text:
        raise V9TraceDiagnosticError(
            "V9 detached terminal receipt is not canonical JSON."
        )
    return detached_path, raw_text, payload, observed_sha256, externally_bound


def _objective_bounds(
    run_context: Mapping[str, object],
) -> tuple[tuple[float, float], tuple[float, float]]:
    for key in ("objective_lower_bounds", "objective_upper_bounds"):
        if key not in run_context:
            raise V9TraceDiagnosticError(
                f"V9 run_context is missing required {key}; refusing HV reconstruction."
            )
        value = run_context[key]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise V9TraceDiagnosticError(f"V9 run_context {key} must be a length-2 array.")
        if len(value) != 2:
            raise V9TraceDiagnosticError(f"V9 run_context {key} must be a length-2 array.")
    lower = tuple(
        _finite_real(value, label=f"objective_lower_bounds[{index}]")
        for index, value in enumerate(run_context["objective_lower_bounds"])
    )
    upper = tuple(
        _finite_real(value, label=f"objective_upper_bounds[{index}]")
        for index, value in enumerate(run_context["objective_upper_bounds"])
    )
    if any(not lo < hi for lo, hi in zip(lower, upper)):
        raise V9TraceDiagnosticError(
            "Every V9 objective lower bound must be strictly below its upper bound."
        )
    return (lower[0], lower[1]), (upper[0], upper[1])


def _normalized_point(
    raw: object,
    *,
    lower: tuple[float, float],
    upper: tuple[float, float],
    label: str,
) -> tuple[float, float]:
    value = _parse_json(raw, label=label)
    if not isinstance(value, list) or len(value) != 2:
        raise V9TraceDiagnosticError(f"{label} must be one exact 2D objective vector.")
    point: list[float] = []
    for index, (item, lo, hi) in enumerate(zip(value, lower, upper)):
        objective = _finite_real(item, label=f"{label}[{index}]")
        if objective < lo or objective > hi:
            raise V9TraceDiagnosticError(
                f"{label}[{index}] lies outside the frozen analytic bounds."
            )
        point.append((objective - lo) / (hi - lo))
    return (point[0], point[1])


def _normalized_hv_2d(points: Sequence[tuple[float, float]]) -> float:
    """Independent minimization HV against the normalized reference (1, 1)."""

    unique = sorted(set(points))
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
    if not math.isfinite(hypervolume) or not -1e-12 <= hypervolume <= 1.0 + 1e-12:
        raise V9TraceDiagnosticError("Reconstructed normalized HV escaped [0, 1].")
    return min(1.0, max(0.0, hypervolume))


def _operator_item(
    by_operator: dict[str, dict[str, float | int]], operator: str
) -> dict[str, float | int]:
    return by_operator.setdefault(
        operator,
        {
            "attempts": 0,
            "cache_hits": 0,
            "first_evaluations": 0,
            "screenings": 0,
            "screen_cache_skips": 0,
            "hv_gain": 0.0,
        },
    )


def _screen_counts(
    context: Mapping[str, object],
    *,
    screening_policy: str,
    label: str,
) -> tuple[int, int]:
    raw_witness = context.get("operator_witness")
    if raw_witness is None:
        return 0, 0
    if not isinstance(raw_witness, Mapping):
        raise V9TraceDiagnosticError(f"{label}.operator_witness must be an object.")
    raw_screen = raw_witness.get("information_time_candidate_screen")
    if raw_screen is None:
        return 0, 0
    if screening_policy != _SCREENING_POLICY:
        raise V9TraceDiagnosticError(
            f"{label} contains screening evidence for a screening-disabled V9 arm."
        )
    if not isinstance(raw_screen, Mapping):
        raise V9TraceDiagnosticError(
            f"{label}.information_time_candidate_screen must be an object."
        )
    screen = dict(raw_screen)
    if screen.get("schema") != _SCREEN_WITNESS_SCHEMA:
        raise V9TraceDiagnosticError(f"{label} has an unsupported screening schema.")
    if screen.get("policy") != _SCREENING_POLICY:
        raise V9TraceDiagnosticError(f"{label} has a drifted screening policy.")
    cap = _exact_int(screen.get("screen_cap"), label=f"{label}.screen_cap", minimum=1)
    examined = _exact_int(
        screen.get("candidates_examined"),
        label=f"{label}.candidates_examined",
        minimum=1,
    )
    skipped = _exact_int(
        screen.get("cached_candidates_skipped"),
        label=f"{label}.cached_candidates_skipped",
    )
    selected_rank = _exact_int(
        screen.get("selected_rank"), label=f"{label}.selected_rank"
    )
    objective_calls = _exact_int(
        screen.get("objective_calls_during_screen"),
        label=f"{label}.objective_calls_during_screen",
    )
    if examined > cap or skipped > examined or selected_rank >= examined:
        raise V9TraceDiagnosticError(f"{label} has inconsistent screening accounting.")
    if objective_calls != 0:
        raise V9TraceDiagnosticError(
            f"{label} reports objective calls during structural screening."
        )
    if type(screen.get("screen_exhausted")) is not bool:
        raise V9TraceDiagnosticError(f"{label}.screen_exhausted must be Boolean.")
    if type(screen.get("selected_operator")) is not str or not str(
        screen["selected_operator"]
    ):
        raise V9TraceDiagnosticError(f"{label}.selected_operator is invalid.")
    selected_solution_sha256 = screen.get("selected_solution_sha256")
    if type(selected_solution_sha256) is not str or (
        len(selected_solution_sha256) != 64
        or any(character not in "0123456789abcdef" for character in selected_solution_sha256)
    ):
        raise V9TraceDiagnosticError(f"{label}.selected_solution_sha256 is invalid.")
    raw_checks = screen.get("candidate_membership_checks")
    if not isinstance(raw_checks, list) or len(raw_checks) != examined:
        raise V9TraceDiagnosticError(
            f"{label}.candidate_membership_checks does not cover examined candidates."
        )
    seen_count = 0
    for expected_rank, raw_check in enumerate(raw_checks):
        if not isinstance(raw_check, Mapping):
            raise V9TraceDiagnosticError(
                f"{label}.candidate_membership_checks[{expected_rank}] is invalid."
            )
        check = dict(raw_check)
        if _exact_int(
            check.get("rank"),
            label=f"{label}.candidate_membership_checks[{expected_rank}].rank",
        ) != expected_rank:
            raise V9TraceDiagnosticError(
                f"{label}.candidate_membership_checks ranks are not contiguous."
            )
        solution = check.get("solution")
        if not isinstance(solution, list) or any(type(value) is not int for value in solution):
            raise V9TraceDiagnosticError(
                f"{label}.candidate_membership_checks[{expected_rank}].solution is invalid."
            )
        solution_sha256 = hashlib.sha256(
            _canonical_json(solution).encode("utf-8")
        ).hexdigest()
        if check.get("solution_sha256") != solution_sha256:
            raise V9TraceDiagnosticError(
                f"{label}.candidate_membership_checks[{expected_rank}] hash is invalid."
            )
        if type(check.get("operator")) is not str or not check.get("operator"):
            raise V9TraceDiagnosticError(
                f"{label}.candidate_membership_checks[{expected_rank}].operator is invalid."
            )
        if type(check.get("seen_before_attempt")) is not bool:
            raise V9TraceDiagnosticError(
                f"{label}.candidate_membership_checks[{expected_rank}] seen flag is invalid."
            )
        seen_count += int(bool(check["seen_before_attempt"]))
    selected_check = dict(raw_checks[selected_rank])
    if not (
        selected_check["solution_sha256"] == selected_solution_sha256
        and selected_check["operator"] == screen["selected_operator"]
    ):
        raise V9TraceDiagnosticError(
            f"{label} selected candidate does not bind its membership check."
        )
    exhausted = bool(screen["screen_exhausted"])
    if skipped != seen_count or (
        exhausted and seen_count != examined
    ) or (
        not exhausted
        and (
            selected_rank != examined - 1
            or bool(selected_check["seen_before_attempt"])
        )
    ):
        raise V9TraceDiagnosticError(f"{label} has inconsistent membership accounting.")
    generated = _exact_int(
        screen.get("structural_candidates_generated"),
        label=f"{label}.structural_candidates_generated",
    )
    probes = _exact_int(
        screen.get("cache_membership_probes"),
        label=f"{label}.cache_membership_probes",
    )
    total_work = _exact_int(
        screen.get("total_structural_screening_work"),
        label=f"{label}.total_structural_screening_work",
    )
    if probes != examined or total_work != generated + probes:
        raise V9TraceDiagnosticError(f"{label} has inconsistent structural work accounting.")
    return examined, skipped


def _terminal_run(
    connection: sqlite3.Connection,
    *,
    detached_receipt_raw: str,
    detached_receipt: Mapping[str, object],
) -> tuple[dict[str, object], str, str, dict[str, object]]:
    run_rows = list(
        connection.execute(
            "SELECT run_id,family,run_context_json,run_context_digest_sha256,"
            "status,terminal_receipt_sha256 FROM run_attempt ORDER BY run_id"
        )
    )
    terminal_rows = list(
        connection.execute(
            "SELECT run_id,status,receipt_json,receipt_sha256 "
            "FROM terminal_receipts ORDER BY run_id"
        )
    )
    if len(run_rows) != 1 or int(run_rows[0]["run_id"]) != 1:
        raise V9TraceDiagnosticError("Trace must contain exactly one run_attempt row.")
    if len(terminal_rows) != 1 or int(terminal_rows[0]["run_id"]) != 1:
        raise V9TraceDiagnosticError(
            "Trace must contain exactly one terminal_receipts row."
        )
    run = run_rows[0]
    terminal = terminal_rows[0]
    if str(run["status"]) != "SUCCESS" or str(terminal["status"]) != "SUCCESS":
        raise V9TraceDiagnosticError("V9 diagnostic requires a terminal SUCCESS trace.")

    run_context_raw = run["run_context_json"]
    run_context = _parse_json_object(run_context_raw, label="run_context_json")
    if _canonical_json(run_context) != run_context_raw:
        raise V9TraceDiagnosticError("V9 run_context_json is not canonical JSON.")
    run_context_digest = hashlib.sha256(
        str(run_context_raw).encode("utf-8")
    ).hexdigest()
    if run_context_digest != str(run["run_context_digest_sha256"]):
        raise V9TraceDiagnosticError("V9 run-context digest does not match its payload.")

    receipt_raw = terminal["receipt_json"]
    receipt = _parse_json_object(receipt_raw, label="terminal receipt_json")
    if _canonical_json(receipt) != receipt_raw:
        raise V9TraceDiagnosticError("V9 terminal receipt is not canonical JSON.")
    if receipt_raw != detached_receipt_raw or receipt != detached_receipt:
        raise V9TraceDiagnosticError(
            "SQLite and detached terminal receipt payloads disagree."
        )
    receipt_hash = str(terminal["receipt_sha256"])
    embedded_hash = receipt.get("receipt_payload_sha256")
    receipt_core = dict(receipt)
    receipt_core.pop("receipt_payload_sha256", None)
    computed_hash = hashlib.sha256(
        _canonical_json(receipt_core).encode("utf-8")
    ).hexdigest()
    if not (
        embedded_hash == receipt_hash == computed_hash
        and str(run["terminal_receipt_sha256"]) == receipt_hash
    ):
        raise V9TraceDiagnosticError("V9 terminal-receipt hash binding is invalid.")
    if receipt.get("status") != "SUCCESS":
        raise V9TraceDiagnosticError("V9 terminal receipt payload is not SUCCESS.")
    if receipt.get("run_context_digest_sha256") != run_context_digest:
        raise V9TraceDiagnosticError("V9 terminal receipt binds another run context.")
    return run_context, run_context_digest, str(run["family"]), receipt


def _v9_arm(
    run_context: Mapping[str, object], *, family: str
) -> tuple[str, str, int, int]:
    if run_context.get("schema") != "v21e3r1_run_context_v2":
        raise V9TraceDiagnosticError("V9 diagnostic requires run-context schema v2.")
    if run_context.get("evidence_partition") != "development":
        raise V9TraceDiagnosticError("V9 diagnostics cannot analyze a later evidence phase.")
    config = run_context.get("algorithm_config")
    if not isinstance(config, Mapping):
        raise V9TraceDiagnosticError("V9 run context omits algorithm_config.")
    diagnostic_id = config.get("development_diagnostic_id")
    if type(diagnostic_id) is not str or diagnostic_id not in _V9_ARMS:
        raise V9TraceDiagnosticError("Trace does not bind one exact V9 diagnostic arm.")
    expected_family, screening, replacement, lambda_positive = _V9_ARMS[diagnostic_id]
    if family != expected_family:
        raise V9TraceDiagnosticError("V9 diagnostic identity and trace family disagree.")
    if config.get("phase") != "development":
        raise V9TraceDiagnosticError("V9 algorithm config is not development-only.")
    if config.get("candidate_id") != "C0" or run_context.get("candidate_id") != "C0":
        raise V9TraceDiagnosticError("V9 diagnostic must remain bound to candidate C0.")
    if config.get("candidate_screening_policy") != screening:
        raise V9TraceDiagnosticError("V9 diagnostic screening policy drifted.")
    if config.get("replacement_policy") != replacement:
        raise V9TraceDiagnosticError("V9 diagnostic replacement policy drifted.")
    tradeoff_lambda = _finite_real(
        config.get("archive_tradeoff_lambda"), label="archive_tradeoff_lambda"
    )
    if (lambda_positive and tradeoff_lambda <= 0.0) or (
        not lambda_positive and tradeoff_lambda != 0.0
    ):
        raise V9TraceDiagnosticError("V9 diagnostic archive tradeoff lambda drifted.")
    budget = _exact_int(
        run_context.get("charged_evaluation_budget"),
        label="charged_evaluation_budget",
        minimum=1,
    )
    if _exact_int(
        config.get("charged_evaluations"),
        label="algorithm_config.charged_evaluations",
        minimum=1,
    ) != budget:
        raise V9TraceDiagnosticError("V9 charged-evaluation budget is not mirrored.")
    reference_directions = config.get("reference_directions")
    if not isinstance(reference_directions, Sequence) or isinstance(
        reference_directions, (str, bytes)
    ):
        raise V9TraceDiagnosticError(
            "V9 algorithm config omits frozen reference_directions."
        )
    population_size = len(reference_directions)
    if not 1 <= population_size <= budget:
        raise V9TraceDiagnosticError(
            "V9 frozen reference-direction population is incompatible with its budget."
        )
    for index, direction in enumerate(reference_directions):
        if not isinstance(direction, Sequence) or isinstance(direction, (str, bytes)):
            raise V9TraceDiagnosticError(
                f"V9 reference_directions[{index}] is not a vector."
            )
        if len(direction) != 2:
            raise V9TraceDiagnosticError(
                f"V9 reference_directions[{index}] is not two-dimensional."
            )
        for coordinate, value in enumerate(direction):
            _finite_real(
                value,
                label=f"reference_directions[{index}][{coordinate}]",
            )
    if run_context.get("reference_directions") != list(reference_directions):
        raise V9TraceDiagnosticError(
            "V9 run context does not mirror frozen reference_directions."
        )
    return diagnostic_id, screening, budget, population_size


def _receipt_count(receipt: Mapping[str, object], key: str) -> int:
    if key not in receipt:
        raise V9TraceDiagnosticError(f"Terminal receipt omits {key}.")
    return _exact_int(receipt[key], label=f"terminal receipt {key}")


def _exact_int_sequence(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise V9TraceDiagnosticError(f"{label} must be an exact integer array.")
    result = tuple(
        _exact_int(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise V9TraceDiagnosticError(f"{label} contains duplicate identifiers.")
    return result


def _replayed_finite(
    value: object,
    *,
    expected: float,
    label: str,
    tolerance: float = 1e-12,
) -> float:
    observed = _finite_real(value, label=label)
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance):
        raise V9TraceDiagnosticError(
            f"{label} disagrees with durable-state arithmetic replay."
        )
    return observed


def _lyapunov_witness_replay(
    *,
    algorithm_config: Mapping[str, object],
    population_size: int,
    normalized_objectives: Sequence[tuple[float, float]],
    evaluation_type_ids: Sequence[int],
    hv_before: Sequence[float],
    hv_after: Sequence[float],
    decision_by_index: Mapping[int, Mapping[str, object]],
) -> tuple[int, int, str]:
    replacement_policy = algorithm_config.get("replacement_policy")
    uses_lyapunov = replacement_policy == _LYAPUNOV_REPLACEMENT
    directions_raw = algorithm_config.get("reference_directions")
    if not isinstance(directions_raw, list):
        raise V9TraceDiagnosticError("V9 reference directions are not durable arrays.")
    directions = tuple(
        tuple(
            _finite_real(value, label=f"reference_directions[{index}]")
            for value in direction
        )
        for index, direction in enumerate(directions_raw)
        if isinstance(direction, list)
    )
    if len(directions) != population_size or any(len(item) != 2 for item in directions):
        raise V9TraceDiagnosticError("V9 reference-direction replay shape is invalid.")
    neighborhood_size = _exact_int(
        algorithm_config.get("neighborhood_size"),
        label="algorithm_config.neighborhood_size",
        minimum=1,
    )
    tradeoff_lambda = _finite_real(
        algorithm_config.get("archive_tradeoff_lambda"),
        label="algorithm_config.archive_tradeoff_lambda",
    )
    population: list[tuple[float, float] | None] = [None] * population_size
    witness_count = 0
    for offset, candidate in enumerate(normalized_objectives):
        evaluation_index = offset + 1
        decision = decision_by_index[evaluation_index]
        targets = _exact_int_sequence(
            decision.get("population_target_type_ids"),
            label=f"decision {evaluation_index} population_target_type_ids",
        )
        if any(target >= population_size for target in targets):
            raise V9TraceDiagnosticError(
                f"decision {evaluation_index} targets an unknown population type."
            )
        replacement_count = _exact_int(
            decision.get("population_replacement_count"),
            label=f"decision {evaluation_index} population_replacement_count",
        )
        accepted = decision.get("accepted_into_population")
        if type(accepted) is not bool:
            raise V9TraceDiagnosticError(
                f"decision {evaluation_index} accepted_into_population is not Boolean."
            )
        if replacement_count != len(targets) or accepted != bool(targets):
            raise V9TraceDiagnosticError(
                f"decision {evaluation_index} population replacement accounting is invalid."
            )
        witness = decision.get("policy_witness")
        if not uses_lyapunov:
            if witness is not None:
                raise V9TraceDiagnosticError(
                    f"Non-Lyapunov V9 decision {evaluation_index} contains a policy witness."
                )
        elif evaluation_index <= population_size:
            if witness is not None:
                raise V9TraceDiagnosticError(
                    f"Initialization decision {evaluation_index} contains a Lyapunov witness."
                )
        else:
            if not isinstance(witness, Mapping):
                raise V9TraceDiagnosticError(
                    f"Lyapunov decision {evaluation_index} omits its durable witness."
                )
            replay_label = f"Lyapunov witness evaluation {evaluation_index}"
            witness_count += 1
            witness_payload = dict(witness)
            if witness_payload.get("schema") != (
                "v21e3r1_archive_compensated_replacement_v2"
            ):
                raise V9TraceDiagnosticError(f"{replay_label} has an unsupported schema.")
            if witness_payload.get("replacement_policy") != _LYAPUNOV_REPLACEMENT:
                raise V9TraceDiagnosticError(f"{replay_label} has a drifted policy.")
            type_id = evaluation_type_ids[offset]
            if not 0 <= type_id < population_size:
                raise V9TraceDiagnosticError(
                    f"evaluation {evaluation_index} has an invalid population type."
                )
            considered = tuple(
                sorted(
                    range(population_size),
                    key=lambda other: (
                        sum(
                            (left - right) ** 2
                            for left, right in zip(
                                directions[type_id], directions[other]
                            )
                        ),
                        other,
                    ),
                )[: min(neighborhood_size, population_size)]
            )
            observed_considered = _exact_int_sequence(
                witness_payload.get("considered_target_type_ids"),
                label=f"{replay_label} considered_target_type_ids",
            )
            if observed_considered != considered:
                raise V9TraceDiagnosticError(
                    f"{replay_label} considered targets disagree with frozen neighborhoods."
                )
            empty_targets = tuple(
                target for target in considered if population[target] is None
            )
            observed_empty = _exact_int_sequence(
                witness_payload.get("preselected_empty_target_type_ids"),
                label=f"{replay_label} preselected_empty_target_type_ids",
            )
            if observed_empty != empty_targets:
                raise V9TraceDiagnosticError(
                    f"{replay_label} empty targets disagree with durable population state."
                )

            def scalar(point: tuple[float, float]) -> float:
                return max(0.5 * point[0], 0.5 * point[1])

            candidate_scalar = scalar(candidate)
            deltas = {
                target: candidate_scalar - scalar(population[target])
                for target in considered
                if population[target] is not None
            }
            expected_delta_rows = [
                {"target_type_id": target, "scalar_delta": delta}
                for target, delta in sorted(deltas.items())
            ]
            observed_delta_rows = witness_payload.get(
                "finite_scalar_delta_by_target"
            )
            if not isinstance(observed_delta_rows, list) or len(
                observed_delta_rows
            ) != len(expected_delta_rows):
                raise V9TraceDiagnosticError(
                    f"{replay_label} finite scalar-delta rows are incomplete."
                )
            for row_index, (observed_row, expected_row) in enumerate(
                zip(observed_delta_rows, expected_delta_rows)
            ):
                if not isinstance(observed_row, Mapping) or observed_row.get(
                    "target_type_id"
                ) != expected_row["target_type_id"]:
                    raise V9TraceDiagnosticError(
                        f"{replay_label} scalar-delta target {row_index} drifted."
                    )
                _replayed_finite(
                    observed_row.get("scalar_delta"),
                    expected=float(expected_row["scalar_delta"]),
                    label=f"{replay_label} scalar_delta[{row_index}]",
                )
            selection_capacity = len(considered) - len(empty_targets)
            if _exact_int(
                witness_payload.get("finite_selection_capacity"),
                label=f"{replay_label} finite_selection_capacity",
            ) != selection_capacity:
                raise V9TraceDiagnosticError(
                    f"{replay_label} finite selection capacity drifted."
                )
            gain = max(0.0, hv_after[offset] - hv_before[offset])
            credit = tradeoff_lambda * gain
            selected: list[int] = []
            positive_total = 0.0
            for target, delta in sorted(
                (item for item in deltas.items() if item[1] <= 0.0),
                key=lambda item: (item[1], item[0]),
            ):
                if len(selected) >= selection_capacity:
                    break
                selected.append(target)
            for target, delta in sorted(
                (item for item in deltas.items() if item[1] > 0.0),
                key=lambda item: (item[1], item[0]),
            ):
                if len(selected) >= selection_capacity:
                    break
                if positive_total + delta <= credit + 1e-12:
                    selected.append(target)
                    positive_total += delta
            decision_selected = tuple(selected)
            replacement_targets = empty_targets + decision_selected
            if _exact_int_sequence(
                witness_payload.get("decision_selected_target_type_ids"),
                label=f"{replay_label} decision_selected_target_type_ids",
            ) != decision_selected:
                raise V9TraceDiagnosticError(
                    f"{replay_label} finite decision selection did not replay."
                )
            if _exact_int_sequence(
                witness_payload.get("selected_target_type_ids"),
                label=f"{replay_label} selected_target_type_ids",
            ) != replacement_targets or targets != replacement_targets:
                raise V9TraceDiagnosticError(
                    f"{replay_label} replacement targets did not replay."
                )
            selected_delta_sum = sum(
                deltas[target]
                for target in replacement_targets
                if target in deltas
            )
            positive_worsening_sum = sum(
                max(0.0, deltas[target])
                for target in replacement_targets
                if target in deltas
            )
            paid_worsening_count = sum(
                deltas[target] > 0.0
                for target in replacement_targets
                if target in deltas
            )
            potential_change = selected_delta_sum - credit
            replay_fields = (
                ("normalized_hv_before", hv_before[offset]),
                ("normalized_hv_after", hv_after[offset]),
                ("normalized_hv_gain", gain),
                ("tradeoff_lambda", tradeoff_lambda),
                ("selected_scalar_delta_sum", selected_delta_sum),
                ("positive_scalar_worsening_sum", positive_worsening_sum),
                ("archive_credit", credit),
                ("composite_potential_change", potential_change),
            )
            for field, expected in replay_fields:
                _replayed_finite(
                    witness_payload.get(field),
                    expected=expected,
                    label=f"{replay_label} {field}",
                )
            if _exact_int(
                witness_payload.get("paid_worsening_target_count"),
                label=f"{replay_label} paid_worsening_target_count",
            ) != paid_worsening_count:
                raise V9TraceDiagnosticError(
                    f"{replay_label} paid-worsening count did not replay."
                )
            if positive_worsening_sum > credit + 1e-10 or potential_change > 1e-10:
                raise V9TraceDiagnosticError(
                    f"{replay_label} violates its archive-compensated bound."
                )
        for target in targets:
            population[target] = candidate
    return (
        witness_count,
        0,
        (
            "DURABLE_STATE_ARITHMETIC_REPLAY_PASS"
            if uses_lyapunov
            else "NOT_APPLICABLE_NON_LYAPUNOV_ARM"
        ),
    )


def _analyze(
    connection: sqlite3.Connection,
    *,
    path: Path,
    detached_receipt_path: Path,
    detached_receipt_raw: str,
    detached_receipt: Mapping[str, object],
    detached_receipt_sha256: str,
    detached_receipt_externally_bound: bool,
) -> dict[str, object]:
    _require_columns(connection)
    run_context, run_context_digest, family, receipt = _terminal_run(
        connection,
        detached_receipt_raw=detached_receipt_raw,
        detached_receipt=detached_receipt,
    )
    diagnostic_id, screening_policy, budget, population_size = _v9_arm(
        run_context, family=family
    )
    lower, upper = _objective_bounds(run_context)

    attempt_rows = list(
        connection.execute(
            "SELECT attempt_index,proposal_solution_ref,proposal_sha256,proposal_json,"
            "proposal_raw_sha256,context_json,status,physical_call_started,"
            "charged_evaluation_index,cache_source_evaluation_index,failure_code,"
            "failure_detail_json,prev_attempt_sha256,attempt_sha256 "
            "FROM attempts ORDER BY attempt_index"
        )
    )
    attempt_indices = [int(row["attempt_index"]) for row in attempt_rows]
    if attempt_indices != list(range(1, len(attempt_rows) + 1)):
        raise V9TraceDiagnosticError("V9 attempt rows are not contiguous from one.")

    by_operator: dict[str, dict[str, float | int]] = {}
    attempt_by_index: dict[int, dict[str, object]] = {}
    evaluated_attempt_indices: list[int] = []
    physical_starts = 0
    cache_hits = 0
    previous_attempt_hash = "0" * 64
    for row in attempt_rows:
        attempt_index = int(row["attempt_index"])
        context = _parse_json_object(
            row["context_json"], label=f"attempt {attempt_index} context_json"
        )
        if _canonical_json(context) != row["context_json"]:
            raise V9TraceDiagnosticError(
                f"attempt {attempt_index} context_json is not canonical JSON."
            )
        operator = context.get("operator_id")
        if type(operator) is not str or not operator:
            raise V9TraceDiagnosticError(
                f"attempt {attempt_index} does not bind one operator_id."
            )
        status = str(row["status"])
        physical = _exact_int(
            row["physical_call_started"],
            label=f"attempt {attempt_index} physical_call_started",
        )
        charged = row["charged_evaluation_index"]
        cache_source = row["cache_source_evaluation_index"]
        if row["attempt_sha256"] is None:
            raise V9TraceDiagnosticError(
                f"terminal attempt {attempt_index} omits attempt_sha256."
            )
        if status == "EVALUATED":
            charged_index = _exact_int(
                charged,
                label=f"attempt {attempt_index} charged_evaluation_index",
                minimum=1,
            )
            if physical != 1 or cache_source is not None:
                raise V9TraceDiagnosticError(
                    f"evaluated attempt {attempt_index} has invalid resource accounting."
                )
            evaluated_attempt_indices.append(attempt_index)
        elif status == "CACHE_HIT":
            if physical != 0 or charged is not None:
                raise V9TraceDiagnosticError(
                    f"cache-hit attempt {attempt_index} has invalid resource accounting."
                )
            charged_index = None
            _exact_int(
                cache_source,
                label=f"attempt {attempt_index} cache_source_evaluation_index",
                minimum=1,
            )
            cache_hits += 1
        else:
            raise V9TraceDiagnosticError(
                f"terminal SUCCESS trace contains nonterminal attempt status {status}."
            )
        proposal_raw = _parse_json(
            row["proposal_json"], label=f"attempt {attempt_index} proposal_json"
        )
        if _canonical_json(proposal_raw) != row["proposal_json"]:
            raise V9TraceDiagnosticError(
                f"attempt {attempt_index} proposal_json is not canonical JSON."
            )
        proposal_raw_sha256 = hashlib.sha256(
            str(row["proposal_json"]).encode("utf-8")
        ).hexdigest()
        if proposal_raw_sha256 != str(row["proposal_raw_sha256"]):
            raise V9TraceDiagnosticError(
                f"attempt {attempt_index} proposal raw SHA-256 binding is invalid."
            )
        failure_detail = (
            None
            if row["failure_detail_json"] is None
            else _parse_json(
                row["failure_detail_json"],
                label=f"attempt {attempt_index} failure_detail_json",
            )
        )
        if row["failure_detail_json"] is not None and (
            _canonical_json(failure_detail) != row["failure_detail_json"]
        ):
            raise V9TraceDiagnosticError(
                f"attempt {attempt_index} failure_detail_json is not canonical JSON."
            )
        if row["failure_code"] is not None or failure_detail is not None:
            raise V9TraceDiagnosticError(
                f"terminal successful attempt {attempt_index} contains failure fields."
            )
        attempt_semantic = {
            "attempt_index": attempt_index,
            "proposal_solution_ref": _optional_exact_int(
                row["proposal_solution_ref"],
                label=f"attempt {attempt_index} proposal_solution_ref",
                minimum=1,
            ),
            "proposal_sha256": (
                None
                if row["proposal_sha256"] is None
                else str(row["proposal_sha256"])
            ),
            "proposal_raw": proposal_raw,
            "proposal_raw_sha256": proposal_raw_sha256,
            "evaluation_context": context,
            "status": status,
            "physical_call_started": physical,
            "charged_evaluation_index": charged_index,
            "cache_source_evaluation_index": _optional_exact_int(
                cache_source,
                label=f"attempt {attempt_index} cache_source_evaluation_index",
                minimum=1,
            ),
            "failure_code": None,
            "failure_detail": None,
            "run_context_digest_sha256": run_context_digest,
            "prev_attempt_sha256": previous_attempt_hash,
        }
        observed_attempt_hash = hashlib.sha256(
            _canonical_json(attempt_semantic).encode("utf-8")
        ).hexdigest()
        if not (
            str(row["prev_attempt_sha256"]) == previous_attempt_hash
            and str(row["attempt_sha256"]) == observed_attempt_hash
        ):
            raise V9TraceDiagnosticError(
                f"Attempt semantic hash chain failed at attempt {attempt_index}."
            )
        previous_attempt_hash = observed_attempt_hash
        physical_starts += physical
        screenings, screen_skips = _screen_counts(
            context,
            screening_policy=screening_policy,
            label=f"attempt {attempt_index}",
        )
        item = _operator_item(by_operator, operator)
        item["attempts"] = int(item["attempts"]) + 1
        item["cache_hits"] = int(item["cache_hits"]) + int(status == "CACHE_HIT")
        item["first_evaluations"] = int(item["first_evaluations"]) + int(
            status == "EVALUATED"
        )
        item["screenings"] = int(item["screenings"]) + screenings
        item["screen_cache_skips"] = int(item["screen_cache_skips"]) + screen_skips
        attempt_by_index[attempt_index] = {
            "context": context,
            "operator": operator,
            "status": status,
            "charged_evaluation_index": charged_index,
            "cache_source_evaluation_index": cache_source,
            "proposal_solution_ref": attempt_semantic["proposal_solution_ref"],
            "proposal_sha256": attempt_semantic["proposal_sha256"],
        }

    evaluation_rows = list(
        connection.execute(
            "SELECT evaluation_index,attempt_index,evidence_partition,search_phase_id,"
            "stage_id,type_id,operator_id,operator_call_id,proposal_solution_ref,"
            "proposal_sha256,objectives_json,prev_record_sha256,record_sha256 "
            "FROM evaluations ORDER BY evaluation_index"
        )
    )
    evaluation_indices = [int(row["evaluation_index"]) for row in evaluation_rows]
    if evaluation_indices != list(range(1, budget + 1)):
        raise V9TraceDiagnosticError(
            "V9 evaluation rows do not exactly cover the frozen charged budget."
        )
    if len(set(evaluated_attempt_indices)) != budget:
        raise V9TraceDiagnosticError(
            "V9 evaluated-attempt count disagrees with the frozen charged budget."
        )

    decision_rows = list(
        connection.execute(
            "SELECT evaluation_index,decision_json,prev_decision_sha256,decision_sha256 "
            "FROM decisions ORDER BY evaluation_index"
        )
    )
    decision_indices = [int(row["evaluation_index"]) for row in decision_rows]
    if decision_indices != evaluation_indices:
        raise V9TraceDiagnosticError("V9 decisions do not cover every evaluation exactly.")
    decision_by_index: dict[int, dict[str, object]] = {}
    previous_decision_hash = "0" * 64
    for row in decision_rows:
        evaluation_index = int(row["evaluation_index"])
        decision = _parse_json_object(
            row["decision_json"], label=f"decision {evaluation_index} decision_json"
        )
        if _canonical_json(decision) != row["decision_json"]:
            raise V9TraceDiagnosticError(
                f"decision {evaluation_index} decision_json is not canonical JSON."
            )
        if _exact_int(
            decision.get("evaluation_index"),
            label=f"decision {evaluation_index} evaluation_index",
            minimum=1,
        ) != evaluation_index:
            raise V9TraceDiagnosticError("A decision payload binds another evaluation.")
        if decision.get("run_context_digest_sha256") != run_context_digest:
            raise V9TraceDiagnosticError("A decision payload binds another run context.")
        if decision.get("prev_decision_sha256") != previous_decision_hash:
            raise V9TraceDiagnosticError(
                f"Decision semantic hash chain embeds the wrong predecessor at "
                f"evaluation {evaluation_index}."
            )
        observed_decision_hash = hashlib.sha256(
            _canonical_json(decision).encode("utf-8")
        ).hexdigest()
        if not (
            str(row["prev_decision_sha256"]) == previous_decision_hash
            and str(row["decision_sha256"]) == observed_decision_hash
        ):
            raise V9TraceDiagnosticError(
                f"Decision semantic hash chain failed at evaluation {evaluation_index}."
            )
        previous_decision_hash = observed_decision_hash
        decision_by_index[evaluation_index] = decision

    points: list[tuple[float, float]] = []
    previous_hv = 0.0
    hv_before: list[float] = []
    hv_after: list[float] = []
    hv_trace: list[dict[str, object]] = []
    evaluation_attempts: set[int] = set()
    previous_evaluation_hash = "0" * 64
    normalized_objectives: list[tuple[float, float]] = []
    evaluation_type_ids: list[int] = []
    for row in evaluation_rows:
        evaluation_index = int(row["evaluation_index"])
        attempt_index = _exact_int(
            row["attempt_index"],
            label=f"evaluation {evaluation_index} attempt_index",
            minimum=1,
        )
        attempt = attempt_by_index.get(attempt_index)
        if attempt is None or attempt["status"] != "EVALUATED":
            raise V9TraceDiagnosticError(
                f"evaluation {evaluation_index} does not bind one evaluated attempt."
            )
        if attempt["charged_evaluation_index"] != evaluation_index:
            raise V9TraceDiagnosticError(
                f"evaluation {evaluation_index} and its attempt disagree on charge index."
            )
        operator = str(row["operator_id"])
        if operator != attempt["operator"]:
            raise V9TraceDiagnosticError(
                f"evaluation {evaluation_index} and its attempt disagree on operator."
            )
        context = attempt["context"]
        if not isinstance(context, Mapping):
            raise V9TraceDiagnosticError(
                f"evaluation {evaluation_index} has no durable attempt context."
            )
        for field in (
            "evidence_partition",
            "search_phase_id",
            "stage_id",
            "type_id",
            "operator_id",
            "operator_call_id",
        ):
            if row[field] != context.get(field):
                raise V9TraceDiagnosticError(
                    f"evaluation {evaluation_index} column {field} disagrees with its attempt."
                )
        if not (
            row["proposal_solution_ref"] == attempt["proposal_solution_ref"]
            and row["proposal_sha256"] == attempt["proposal_sha256"]
        ):
            raise V9TraceDiagnosticError(
                f"evaluation {evaluation_index} proposal binding disagrees with its attempt."
            )
        if attempt_index in evaluation_attempts:
            raise V9TraceDiagnosticError("Two evaluations bind the same attempt.")
        evaluation_attempts.add(attempt_index)
        objective_payload = _parse_json(
            row["objectives_json"],
            label=f"evaluation {evaluation_index} objectives_json",
        )
        if _canonical_json(objective_payload) != row["objectives_json"]:
            raise V9TraceDiagnosticError(
                f"evaluation {evaluation_index} objectives_json is not canonical JSON."
            )
        semantic = {
            "evaluation_index": evaluation_index,
            "attempt_index": attempt_index,
            "context": attempt["context"],
            "proposal_solution_ref": _exact_int(
                row["proposal_solution_ref"],
                label=f"evaluation {evaluation_index} proposal_solution_ref",
                minimum=1,
            ),
            "proposal_sha256": str(row["proposal_sha256"]),
            "objectives": objective_payload,
            "run_context_digest_sha256": run_context_digest,
            "prev_record_sha256": previous_evaluation_hash,
        }
        observed_record_hash = hashlib.sha256(
            _canonical_json(semantic).encode("utf-8")
        ).hexdigest()
        if not (
            str(row["prev_record_sha256"]) == previous_evaluation_hash
            and str(row["record_sha256"]) == observed_record_hash
        ):
            raise V9TraceDiagnosticError(
                f"Evaluation semantic hash chain failed at evaluation {evaluation_index}."
            )
        previous_evaluation_hash = observed_record_hash
        point = _normalized_point(
            row["objectives_json"],
            lower=lower,
            upper=upper,
            label=f"evaluation {evaluation_index} objectives_json",
        )
        normalized_objectives.append(point)
        evaluation_type_ids.append(
            _exact_int(
                row["type_id"],
                label=f"evaluation {evaluation_index} type_id",
            )
        )
        hv_before.append(previous_hv)
        points.append(point)
        current_hv = _normalized_hv_2d(points)
        hv_after.append(current_hv)
        if current_hv < previous_hv - 1e-12:
            raise V9TraceDiagnosticError("All-evaluated archive HV decreased unexpectedly.")
        gain = max(0.0, current_hv - previous_hv)
        item = _operator_item(by_operator, operator)
        item["hv_gain"] = float(item["hv_gain"]) + gain
        hv_trace.append(
            {
                "evaluation_index": evaluation_index,
                "operator": operator,
                "normalized_hv": current_hv,
                "normalized_hv_gain": gain,
            }
        )
        previous_hv = current_hv

    for attempt_index, attempt in attempt_by_index.items():
        if attempt["status"] != "CACHE_HIT":
            continue
        source = int(attempt["cache_source_evaluation_index"])
        if source > budget:
            raise V9TraceDiagnosticError(
                f"cache-hit attempt {attempt_index} references a missing evaluation."
            )
        source_attempt = int(evaluation_rows[source - 1]["attempt_index"])
        if source_attempt >= attempt_index:
            raise V9TraceDiagnosticError(
                f"cache-hit attempt {attempt_index} does not reference an earlier evaluation."
            )

    terminal_chains = (
        (
            "terminal_attempt_chain_sha256",
            previous_attempt_hash,
            "terminal attempt chain binding",
        ),
        (
            "terminal_evaluation_chain_sha256",
            previous_evaluation_hash,
            "terminal evaluation chain binding",
        ),
        (
            "terminal_decision_chain_sha256",
            previous_decision_hash,
            "terminal decision chain binding",
        ),
    )
    for field, expected_hash, label in terminal_chains:
        if receipt.get(field) != expected_hash:
            raise V9TraceDiagnosticError(f"V9 {label} is invalid.")

    attempts_count = len(attempt_rows)
    decisions_count = len(decision_rows)
    if not (
        physical_starts == budget
        and len(evaluation_rows) == budget
        and decisions_count == budget
        and cache_hits == attempts_count - budget
        and evaluation_attempts == set(evaluated_attempt_indices)
        and _receipt_count(receipt, "attempt_count") == attempts_count
        and _receipt_count(receipt, "physical_call_started_count") == budget
        and _receipt_count(receipt, "charged_evaluation_count") == budget
        and _receipt_count(receipt, "decision_count") == budget
        and _receipt_count(receipt, "cache_hit_count") == cache_hits
        and _receipt_count(receipt, "unresolved_decision_count") == 0
    ):
        raise V9TraceDiagnosticError("V9 terminal accounting is inconsistent.")

    total_gain = sum(float(item["hv_gain"]) for item in by_operator.values())
    if not math.isclose(total_gain, previous_hv, rel_tol=0.0, abs_tol=1e-12):
        raise V9TraceDiagnosticError(
            "Per-operator HV attribution does not sum to the reconstructed final HV."
        )
    exact_left_continuous_auc = sum(hv_before) / float(budget)
    initialization_terminal_hv = hv_after[population_size - 1]
    post_initialization_gain = previous_hv - initialization_terminal_hv
    if post_initialization_gain < -1e-12:
        raise V9TraceDiagnosticError(
            "Reconstructed post-initialization HV gain is unexpectedly negative."
        )
    post_initialization_gain = max(0.0, post_initialization_gain)
    algorithm_config = run_context.get("algorithm_config")
    if not isinstance(algorithm_config, Mapping):
        raise V9TraceDiagnosticError("V9 run context omits algorithm_config.")
    (
        lyapunov_witness_count,
        lyapunov_witness_violation_count,
        lyapunov_witness_replay,
    ) = _lyapunov_witness_replay(
        algorithm_config=algorithm_config,
        population_size=population_size,
        normalized_objectives=normalized_objectives,
        evaluation_type_ids=evaluation_type_ids,
        hv_before=hv_before,
        hv_after=hv_after,
        decision_by_index=decision_by_index,
    )
    rows: list[dict[str, Any]] = []
    for operator in sorted(by_operator):
        raw = by_operator[operator]
        productivity = operator_productivity(
            attempts=int(raw["attempts"]),
            new_states=int(raw["first_evaluations"]),
            total_quality_gain=float(raw["hv_gain"]),
        )
        rows.append({"operator": operator, **raw, **asdict(productivity)})

    hv_trace_sha256 = hashlib.sha256(
        _canonical_json(hv_trace).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "v21e3r1_v9_operator_productivity_diagnostic_v3",
        "status": "DEVELOPMENT_ONLY_NO_LATER_PHASE_AUTHORIZATION",
        "verification_scope": (
            "durable_semantic_chains_terminal_detached_and_arithmetic_reconstruction_v1"
        ),
        "objective_function_replay": "NOT_IMPLEMENTED_NO_PROBLEM_INPUT",
        "database_path": str(path),
        "detached_terminal_receipt_path": str(detached_receipt_path),
        "detached_terminal_receipt_sha256": detached_receipt_sha256,
        "detached_terminal_receipt_externally_bound": (
            detached_receipt_externally_bound
        ),
        "family": family,
        "development_diagnostic_id": diagnostic_id,
        "operator_count": len(rows),
        "attempt_count": attempts_count,
        "first_evaluation_count": budget,
        "decision_count": decisions_count,
        "population_size": population_size,
        "initialization_end_evaluation": population_size,
        "initialization_terminal_hv": initialization_terminal_hv,
        "exact_per_evaluation_left_continuous_hv_auc": (
            exact_left_continuous_auc
        ),
        "post_initialization_incremental_hv_gain": post_initialization_gain,
        "lyapunov_witness_count": lyapunov_witness_count,
        "lyapunov_witness_violation_count": lyapunov_witness_violation_count,
        "lyapunov_witness_replay": lyapunov_witness_replay,
        "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
        "total_screenings": sum(int(item["screenings"]) for item in by_operator.values()),
        "total_screen_cache_skips": sum(
            int(item["screen_cache_skips"]) for item in by_operator.values()
        ),
        "final_normalized_hv": previous_hv,
        "total_reconstructed_hv_gain": total_gain,
        "hv_reconstruction": {
            "schema": "v21e3r1_v9_all_evaluated_2d_hv_reconstruction_v1",
            "objective_lower_bounds": list(lower),
            "objective_upper_bounds": list(upper),
            "normalized_reference": [1.0, 1.0],
            "evaluation_order": "charged_evaluation_index_ascending",
            "trace_sha256": hv_trace_sha256,
        },
        "implementation_independence": False,
        "scientific_independence": False,
        "third_party_independence": False,
        "policy_witness_independent_hv_reconstruction": True,
        "authorization": {
            "selection": False,
            "confirmation": False,
            "formal": False,
            "submission": False,
        },
        "validation": {
            "sqlite_read_only_uri": True,
            "sqlite_query_only": True,
            "sqlite_integrity": "ok",
            "terminal_success": True,
            "contiguous_attempts": True,
            "contiguous_evaluations": True,
            "complete_decisions": True,
            "accounting_consistent": True,
            "attempt_semantic_hash_chain": True,
            "evaluation_semantic_hash_chain": True,
            "decision_semantic_hash_chain": True,
            "terminal_chain_bindings": True,
            "detached_terminal_receipt_exact_match": True,
            "detached_terminal_receipt_external_sha256_bound": (
                detached_receipt_externally_bound
            ),
            "lyapunov_witness_durable_state_arithmetic": (
                lyapunov_witness_replay
            ),
        },
        "operators": rows,
    }


def analyze_v9_trace_database(
    database_path: str | Path,
    *,
    detached_terminal_receipt_path: str | Path | None = None,
    expected_detached_terminal_receipt_sha256: str | None = None,
) -> dict[str, object]:
    """Validate durable V9 bindings and reanalyze without objective replay.

    With no problem object, this interface cannot re-execute the objective
    function or claim full algorithm-decision replay.  It does recompute every
    semantic hash chain, exact terminal/detached bindings, normalized HV path,
    and the arithmetic encoded by V9 Lyapunov witnesses.
    """

    path = Path(database_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    (
        detached_path,
        detached_raw,
        detached_receipt,
        detached_sha256,
        detached_externally_bound,
    ) = _detached_terminal_receipt(
        path,
        detached_terminal_receipt_path=detached_terminal_receipt_path,
        expected_detached_terminal_receipt_sha256=(
            expected_detached_terminal_receipt_sha256
        ),
    )
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        query_only_row = connection.execute("PRAGMA query_only").fetchone()
        if query_only_row is None or int(query_only_row[0]) != 1:
            raise V9TraceDiagnosticError("SQLite query_only could not be enforced.")
        connection.execute("BEGIN")
        integrity_rows = [
            str(row[0]) for row in connection.execute("PRAGMA integrity_check")
        ]
        if integrity_rows != ["ok"]:
            raise V9TraceDiagnosticError(
                f"SQLite integrity_check failed: {integrity_rows}."
            )
        return _analyze(
            connection,
            path=path,
            detached_receipt_path=detached_path,
            detached_receipt_raw=detached_raw,
            detached_receipt=detached_receipt,
            detached_receipt_sha256=detached_sha256,
            detached_receipt_externally_bound=detached_externally_bound,
        )
    except V9TraceDiagnosticError:
        raise
    except sqlite3.Error as error:
        raise V9TraceDiagnosticError(
            f"SQLite read-only V9 diagnostic failed: {error}."
        ) from error
    finally:
        if connection is not None:
            if connection.in_transaction:
                connection.rollback()
            connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only V9 operator productivity and all-evaluated HV diagnostic."
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--terminal-receipt",
        type=Path,
        help="Detached terminal receipt; defaults to terminal.json beside the trace.",
    )
    parser.add_argument(
        "--expected-terminal-receipt-sha256",
        help="Optional external lowercase SHA-256 anchor for the detached receipt.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = analyze_v9_trace_database(
        args.trace,
        detached_terminal_receipt_path=args.terminal_receipt,
        expected_detached_terminal_receipt_sha256=(
            args.expected_terminal_receipt_sha256
        ),
    )
    if args.output is not None:
        output = args.output.resolve()
        with output.open("xb") as handle:
            handle.write(_canonical_json(report).encode("utf-8") + b"\n")
    print(
        _canonical_json(
            {
                "status": report["status"],
                "trace": str(Path(args.trace).resolve()),
                "output": (
                    None if args.output is None else str(args.output.resolve())
                ),
                "selection_authorized": False,
                "confirmation_authorized": False,
                "formal_authorized": False,
                "submission_authorized": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["V9TraceDiagnosticError", "analyze_v9_trace_database", "main"]
