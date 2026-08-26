from __future__ import annotations

"""Independent objective/archive and deterministic-policy trace replay.

The verifier reconstructs V9 screening, Lyapunov, and typed-population updates
after each durable proposal.  It deliberately does not regenerate RNG,
operator choices, or proposal generation, so full algorithm decision replay
remains unimplemented and cannot authorize selection evidence.
"""

import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct
from typing import Iterator, Mapping, Sequence

from .archive import ArchiveEntry, ParetoArchive
from .pareto_ijoc_problem import MultiObjectiveCombinatorialProblem, Solution


_V9_SCREEN_WITNESS_SCHEMA = "v21e3r1_information_time_candidate_screen_v2"
_V9_LYAPUNOV_WITNESS_SCHEMA = "v21e3r1_archive_compensated_replacement_v2"
_V9_LYAPUNOV_REPLACEMENT_POLICY = (
    "archive_compensated_information_lyapunov_development_v1"
)
_V9_LYAPUNOV_WITNESS_KEYS = {
    "archive_credit",
    "composite_potential_change",
    "considered_target_type_ids",
    "decision_selected_target_type_ids",
    "finite_scalar_delta_by_target",
    "finite_selection_capacity",
    "normalized_hv_after",
    "normalized_hv_before",
    "normalized_hv_gain",
    "paid_worsening_target_count",
    "positive_scalar_worsening_sum",
    "preselected_empty_target_type_ids",
    "replacement_policy",
    "schema",
    "selected_scalar_delta_sum",
    "selected_target_type_ids",
    "tradeoff_lambda",
}


def _exact_finite_real(value: object, *, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be an exact finite JSON number.")
    return float(value)


def _same_real(left: object, right: float, *, label: str) -> None:
    observed = _exact_finite_real(left, label=label)
    if not math.isclose(observed, right, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"V9 Lyapunov policy witness {label} drifted.")


def _validate_v9_lyapunov_policy_witness(
    witness: Mapping[str, object],
) -> tuple[int, ...]:
    """Independently replay the finite deterministic Lyapunov subset rule."""

    if type(witness) is not dict or set(witness) != _V9_LYAPUNOV_WITNESS_KEYS:
        raise ValueError("V9 Lyapunov policy witness shape drifted.")
    if witness.get("schema") != _V9_LYAPUNOV_WITNESS_SCHEMA:
        raise ValueError("V9 Lyapunov policy witness schema drifted.")
    if witness.get("replacement_policy") != _V9_LYAPUNOV_REPLACEMENT_POLICY:
        raise ValueError("V9 Lyapunov replacement policy drifted.")
    considered = witness.get("considered_target_type_ids")
    if (
        not isinstance(considered, list)
        or any(type(value) is not int or value < 0 for value in considered)
        or len(set(considered)) != len(considered)
    ):
        raise ValueError("V9 Lyapunov considered targets drifted.")
    raw_deltas = witness.get("finite_scalar_delta_by_target")
    if not isinstance(raw_deltas, list):
        raise ValueError("V9 Lyapunov policy witness omits scalar deltas.")
    deltas: list[tuple[int, float]] = []
    for raw in raw_deltas:
        if not isinstance(raw, dict) or set(raw) != {
            "target_type_id",
            "scalar_delta",
        }:
            raise ValueError("V9 Lyapunov policy witness scalar-delta shape drifted.")
        target = raw["target_type_id"]
        if type(target) is not int or target < 0:
            raise ValueError("V9 Lyapunov policy witness target id is invalid.")
        delta = _exact_finite_real(
            raw["scalar_delta"], label="Lyapunov scalar delta"
        )
        deltas.append((target, delta))
    if len({target for target, _ in deltas}) != len(deltas):
        raise ValueError("V9 Lyapunov policy witness repeats a target id.")
    capacity = witness.get("finite_selection_capacity")
    if type(capacity) is not int or capacity < 0:
        raise ValueError("V9 Lyapunov policy witness capacity is invalid.")
    gain = _exact_finite_real(
        witness.get("normalized_hv_gain"), label="Lyapunov normalized HV gain"
    )
    tradeoff = _exact_finite_real(
        witness.get("tradeoff_lambda"), label="Lyapunov tradeoff lambda"
    )
    if gain < -1e-12 or gain > 1.0 + 1e-12 or tradeoff < 0.0:
        raise ValueError("V9 Lyapunov policy witness has invalid gain or lambda.")
    credit = tradeoff * max(0.0, gain)
    nonpositive = sorted(
        (item for item in deltas if item[1] <= 0.0),
        key=lambda item: (item[1], item[0]),
    )
    positive = sorted(
        (item for item in deltas if item[1] > 0.0),
        key=lambda item: (item[1], item[0]),
    )
    selected: list[tuple[int, float]] = []
    positive_total = 0.0
    for item in nonpositive:
        if len(selected) >= capacity:
            break
        selected.append(item)
    for item in positive:
        if len(selected) >= capacity:
            break
        if positive_total + item[1] <= credit + 1e-12:
            selected.append(item)
            positive_total += item[1]
    selected_targets = tuple(target for target, _ in selected)
    selected_sum = sum(delta for _, delta in selected)
    raw_selected = witness.get("decision_selected_target_type_ids")
    if not isinstance(raw_selected, list) or tuple(raw_selected) != selected_targets:
        raise ValueError("V9 Lyapunov policy witness selected subset drifted.")
    preselected = witness.get("preselected_empty_target_type_ids")
    final_selected = witness.get("selected_target_type_ids")
    if (
        not isinstance(preselected, list)
        or any(type(value) is not int or value < 0 for value in preselected)
        or len(set(preselected)) != len(preselected)
        or any(value not in considered for value in preselected)
        or any(value in selected_targets for value in preselected)
        or not isinstance(final_selected, list)
        or tuple(final_selected) != tuple(preselected) + selected_targets
    ):
        raise ValueError("V9 Lyapunov policy witness final target set drifted.")
    paid_worsening_count = witness.get("paid_worsening_target_count")
    expected_paid_worsening_count = sum(delta > 0.0 for _, delta in selected)
    if (
        type(paid_worsening_count) is not int
        or paid_worsening_count != expected_paid_worsening_count
    ):
        raise ValueError(
            "V9 Lyapunov policy witness paid-worsening count drifted."
        )
    _same_real(witness.get("archive_credit"), credit, label="archive_credit")
    _same_real(
        witness.get("selected_scalar_delta_sum"),
        selected_sum,
        label="selected_scalar_delta_sum",
    )
    _same_real(
        witness.get("positive_scalar_worsening_sum"),
        positive_total,
        label="positive_scalar_worsening_sum",
    )
    _same_real(
        witness.get("composite_potential_change"),
        selected_sum - credit,
        label="composite_potential_change",
    )
    return selected_targets


def _validate_v9_candidate_screen_witness(
    witness: Mapping[str, object],
    *,
    evaluated_solution_sha256_before_attempt: set[str],
    selected_attempt_sha256: str,
    problem: MultiObjectiveCombinatorialProblem,
) -> None:
    """Replay first-unseen selection from exact durable candidate identities."""

    if witness.get("schema") != _V9_SCREEN_WITNESS_SCHEMA:
        raise ValueError("V9 candidate-screen witness schema drifted.")
    raw_checks = witness.get("candidate_membership_checks")
    if not isinstance(raw_checks, list) or not raw_checks:
        raise ValueError("V9 candidate-screen witness omits membership checks.")
    observed: list[tuple[str, bool, str]] = []
    for expected_rank, raw in enumerate(raw_checks):
        if not isinstance(raw, dict) or set(raw) != {
            "rank",
            "solution",
            "solution_sha256",
            "operator",
            "seen_before_attempt",
        }:
            raise ValueError("V9 candidate-screen membership shape drifted.")
        if raw["rank"] != expected_rank or type(raw["rank"]) is not int:
            raise ValueError("V9 candidate-screen rank drifted.")
        solution_raw = raw["solution"]
        if (
            not isinstance(solution_raw, list)
            or any(type(value) is not int for value in solution_raw)
        ):
            raise ValueError("V9 candidate-screen solution is not exact integer JSON.")
        solution = tuple(solution_raw)
        problem.validate_solution(solution)
        digest = _solution_sha256(solution)
        if raw["solution_sha256"] != digest:
            raise ValueError("V9 candidate-screen solution SHA-256 drifted.")
        expected_seen = digest in evaluated_solution_sha256_before_attempt
        if type(raw["seen_before_attempt"]) is not bool or (
            raw["seen_before_attempt"] is not expected_seen
        ):
            raise ValueError("V9 candidate-screen cache-membership result drifted.")
        operator = raw["operator"]
        if type(operator) is not str or not operator:
            raise ValueError("V9 candidate-screen operator identity drifted.")
        observed.append((digest, expected_seen, operator))
    candidates_examined = witness.get("candidates_examined")
    cached_skipped = witness.get("cached_candidates_skipped")
    selected_rank = witness.get("selected_rank")
    screen_cap = witness.get("screen_cap")
    if (
        type(candidates_examined) is not int
        or candidates_examined != len(observed)
        or type(cached_skipped) is not int
        or cached_skipped < 0
        or type(selected_rank) is not int
        or type(screen_cap) is not int
        or screen_cap <= 0
        or len(observed) > screen_cap
    ):
        raise ValueError("V9 candidate-screen count contract drifted.")
    first_unseen = next(
        (index for index, (_digest, seen, _operator) in enumerate(observed) if not seen),
        None,
    )
    expected_exhausted = first_unseen is None
    expected_rank = len(observed) - 1 if expected_exhausted else first_unseen
    expected_skipped = (
        len(observed) if expected_exhausted else int(first_unseen)
    )
    if (
        witness.get("screen_exhausted") is not expected_exhausted
        or selected_rank != expected_rank
        or cached_skipped != expected_skipped
        or witness.get("selected_solution_sha256") != observed[expected_rank][0]
        or selected_attempt_sha256 != observed[expected_rank][0]
        or witness.get("selected_operator") != observed[expected_rank][2]
        or witness.get("objective_calls_during_screen") != 0
    ):
        raise ValueError("V9 candidate-screen first-unseen decision drifted.")


def _v9_reference_directions(
    algorithm_config: Mapping[str, object],
    *,
    objective_dimension: int,
) -> tuple[tuple[float, ...], ...]:
    raw = algorithm_config.get("reference_directions")
    if not isinstance(raw, list) or not raw:
        raise ValueError("V9 policy replay lacks reference directions.")
    directions: list[tuple[float, ...]] = []
    for row in raw:
        if (
            not isinstance(row, list)
            or len(row) != objective_dimension
            or any(type(value) not in {int, float} for value in row)
        ):
            raise ValueError("V9 policy replay reference directions drifted.")
        direction = tuple(float(value) for value in row)
        if any(not math.isfinite(value) or value <= 0.0 for value in direction):
            raise ValueError("V9 policy replay reference direction is invalid.")
        directions.append(direction)
    return tuple(directions)


def _v9_type_neighbors(
    directions: Sequence[Sequence[float]],
    *,
    neighborhood_size: int,
) -> tuple[tuple[int, ...], ...]:
    if type(neighborhood_size) is not int or neighborhood_size < 1:
        raise ValueError("V9 policy replay neighborhood size is invalid.")
    return tuple(
        tuple(
            sorted(
                range(len(directions)),
                key=lambda other: (
                    sum(
                        (float(left) - float(right)) ** 2
                        for left, right in zip(
                            directions[index], directions[other]
                        )
                    ),
                    other,
                ),
            )[: min(neighborhood_size, len(directions))]
        )
        for index in range(len(directions))
    )


def _v9_scalar(
    objectives: Sequence[float],
    direction: Sequence[float],
    *,
    lower: Sequence[float],
    upper: Sequence[float],
) -> float:
    normalized = tuple(
        (float(value) - float(lo)) / (float(hi) - float(lo))
        for value, lo, hi in zip(objectives, lower, upper)
    )
    return max(
        float(weight) * value for weight, value in zip(direction, normalized)
    )


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def decode_v21e3_objectives_json(
    raw: str,
    *,
    expected_dimension: int,
) -> tuple[float, ...]:
    """Decode one canonical ledger vector without numeric type coercion."""

    if type(raw) is not str or type(expected_dimension) is not int:
        raise ValueError("Recorded objectives require exact JSON numbers.")
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Recorded objectives require exact JSON numbers.") from error
    if (
        not isinstance(decoded, list)
        or len(decoded) != expected_dimension
        or any(type(value) not in (int, float) for value in decoded)
    ):
        raise ValueError("Recorded objectives require exact JSON numbers.")
    if _canonical_bytes(decoded) != raw.encode("utf-8"):
        raise ValueError("Recorded objectives require canonical JSON encoding.")
    objectives = tuple(float(value) for value in decoded)
    if any(not math.isfinite(value) for value in objectives):
        raise ValueError("Recorded objectives require finite JSON numbers.")
    return objectives


def _sqlite_read_only_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=ro"


def _solution_sha256(solution: Sequence[int]) -> str:
    return hashlib.sha256(
        _canonical_bytes([int(value) for value in solution])
    ).hexdigest()


def _qualified_type(value: object) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _raw_value_payload(value: object) -> object:
    if value is None:
        return {"kind": "none"}
    if type(value) is bool:
        return {"kind": "bool", "value": bool(value)}
    if type(value) is int:
        return {"kind": "int", "decimal": str(value)}
    if type(value) is float:
        return {"kind": "float64", "ieee754_be_hex": struct.pack(">d", value).hex()}
    if type(value) is str:
        return {"kind": "str", "value": value}
    if type(value) is bytes:
        return {"kind": "bytes", "hex": value.hex()}
    if type(value) in {tuple, list}:
        return {
            "kind": "sequence",
            "container_type": _qualified_type(value),
            "items": [_raw_value_payload(item) for item in value],
        }
    return {
        "kind": "unsupported",
        "qualified_type": _qualified_type(value),
        "repr": repr(value),
    }


def _canonical_raw_proposal(proposal: object) -> tuple[str, str]:
    payload = {
        "schema": "v21e3_raw_proposal_v1",
        "container_type": _qualified_type(proposal),
        "iterable": True,
        "items": [_raw_value_payload(item) for item in proposal],  # type: ignore[union-attr]
    }
    raw = _canonical_bytes(payload)
    return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()


def _attempt_semantic(
    row: sqlite3.Row,
    *,
    run_context_digest_sha256: str,
    previous_attempt_hash: str,
) -> dict[str, object]:
    return {
        "attempt_index": int(row["attempt_index"]),
        "proposal_solution_ref": (
            None
            if row["proposal_solution_ref"] is None
            else int(row["proposal_solution_ref"])
        ),
        "proposal_sha256": (
            None if row["proposal_sha256"] is None else str(row["proposal_sha256"])
        ),
        "proposal_raw": json.loads(str(row["proposal_json"])),
        "proposal_raw_sha256": str(row["proposal_raw_sha256"]),
        "evaluation_context": json.loads(str(row["context_json"])),
        "status": str(row["status"]),
        "physical_call_started": int(row["physical_call_started"]),
        "charged_evaluation_index": (
            None
            if row["charged_evaluation_index"] is None
            else int(row["charged_evaluation_index"])
        ),
        "cache_source_evaluation_index": (
            None
            if row["cache_source_evaluation_index"] is None
            else int(row["cache_source_evaluation_index"])
        ),
        "failure_code": (
            None if row["failure_code"] is None else str(row["failure_code"])
        ),
        "failure_detail": (
            None
            if row["failure_detail_json"] is None
            else json.loads(str(row["failure_detail_json"]))
        ),
        "run_context_digest_sha256": run_context_digest_sha256,
        "prev_attempt_sha256": previous_attempt_hash,
    }


def _decode_solution(codec: str, size: int, payload: bytes) -> Solution:
    if size < 0:
        raise ValueError("A V21e3 solution has negative size.")
    if codec == "mokp-bitpack-lsb-v1":
        expected_bytes = (size + 7) // 8
        if len(payload) != expected_bytes:
            raise ValueError("A V21e3 MOKP payload has the wrong byte length.")
        if size % 8 and payload and payload[-1] >> (size % 8):
            raise ValueError("A V21e3 MOKP payload has nonzero padding bits.")
        return tuple(
            (payload[index // 8] >> (index % 8)) & 1 for index in range(size)
        )
    if codec == "motsp-uint16le-v1":
        if len(payload) != 2 * size:
            raise ValueError("A V21e3 uint16 MOTSP payload has the wrong length.")
        return tuple(struct.unpack(f"<{size}H", payload))
    if codec == "motsp-uint32le-v1":
        if len(payload) != 4 * size:
            raise ValueError("A V21e3 uint32 MOTSP payload has the wrong length.")
        return tuple(struct.unpack(f"<{size}I", payload))
    if codec == "generic-json-v1":
        decoded = json.loads(payload)
        if _canonical_bytes(decoded) != payload:
            raise ValueError("A generic V21e3 solution payload is not canonical JSON.")
        return tuple(int(value) for value in decoded)
    raise ValueError(f"Unsupported V21e3 solution codec: {codec}")


def _validate_replayed_objective(
    problem: MultiObjectiveCombinatorialProblem,
    raw: object,
) -> tuple[float, ...]:
    values = tuple(raw)  # type: ignore[arg-type]
    expected_dimension = int(problem.num_objectives)
    if len(values) != expected_dimension:
        raise ValueError("Replayed objective has the wrong dimension.")
    if any(type(value) not in (int, float) for value in values):
        raise ValueError("Replayed objective requires exact numeric coordinate types.")
    objectives = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in objectives):
        raise ValueError("Replayed objective is not finite.")
    lower = tuple(float(value) for value in problem.objective_lower_bounds)
    upper = tuple(float(value) for value in problem.objective_upper_bounds)
    if not len(lower) == len(upper) == expected_dimension:
        raise ValueError("Replay problem has an invalid objective box.")
    if any(
        value < lo or value > hi
        for value, lo, hi in zip(objectives, lower, upper)
    ):
        raise ValueError("Replayed objective falls outside the frozen box.")
    return objectives


def _validate_run_context_intrinsic(payload: Mapping[str, object]) -> None:
    """Independently validate legacy-v1 or strict V21e3r1-v2 run roots."""

    schema = payload.get("schema")
    if schema not in {"v21e3_run_context_v1", "v21e3r1_run_context_v2"}:
        raise ValueError("The V21e3 run-context schema is invalid.")
    algorithm_config = payload.get("algorithm_config")
    if not isinstance(algorithm_config, Mapping):
        raise ValueError("The V21e3 run context has a non-mapping algorithm config.")
    config_hash = hashlib.sha256(_canonical_bytes(algorithm_config)).hexdigest()
    if payload.get("candidate_config_sha256") != config_hash:
        raise ValueError("The V21e3 run context has a bad config digest.")
    mirror_fields = (
        ("candidate_id", "candidate_id"),
        ("seed", "seed"),
        ("charged_evaluation_budget", "charged_evaluations"),
        ("evidence_partition", "phase"),
        ("reference_directions", "reference_directions"),
    )
    for outer_key, config_key in mirror_fields:
        if config_key not in algorithm_config:
            if schema == "v21e3r1_run_context_v2":
                raise ValueError(
                    "The V21e3r1 run context is missing an authoritative mirror: "
                    f"algorithm_config.{config_key}."
                )
            continue
        if _canonical_bytes(payload.get(outer_key)) != _canonical_bytes(
            algorithm_config[config_key]
        ):
            raise ValueError(
                "The V21e3 run context has inconsistent mirrored fields: "
                f"{outer_key} != algorithm_config.{config_key}."
            )
    diagnostic_id = algorithm_config.get("development_diagnostic_id")
    if type(diagnostic_id) is str and diagnostic_id.startswith("V21E3R1_V9_"):
        if payload.get("v9_resource_contract_schema") != (
            "v21e3r1_v9_ast_resource_contract_v1"
        ):
            raise ValueError("The V9 run context lacks its exact A/S/T schema.")
        lower = payload.get("objective_lower_bounds")
        upper = payload.get("objective_upper_bounds")
        if not isinstance(lower, list) or not isinstance(upper, list):
            raise ValueError("The V9 run context lacks its normalized-HV box.")
        if len(lower) != 2 or len(upper) != 2:
            raise ValueError("The V9 normalized-HV box must be two-dimensional.")
        if any(
            type(value) not in {int, float} or not math.isfinite(float(value))
            for value in (*lower, *upper)
        ):
            raise ValueError("The V9 normalized-HV bounds have invalid exact types.")
        if any(float(lo) >= float(hi) for lo, hi in zip(lower, upper)):
            raise ValueError("The V9 normalized-HV box is degenerate.")


def _validate_decision_exact_types(payload: Mapping[str, object]) -> None:
    integer_fields = (
        "evaluation_index",
        "population_replacement_count",
        "archive_size_after",
    )
    boolean_fields = (
        "accepted_into_population",
        "archive_changed",
        "retained_after_update",
    )
    if any(type(payload.get(field)) is not int for field in integer_fields):
        raise ValueError("A V21e3 decision integer field has the wrong exact type.")
    if any(type(payload.get(field)) is not bool for field in boolean_fields):
        raise ValueError("A V21e3 decision boolean field has the wrong exact type.")
    target_type_ids = payload.get("population_target_type_ids")
    if not isinstance(target_type_ids, list) or any(
        type(value) is not int for value in target_type_ids
    ):
        raise ValueError("A V21e3 decision target field has the wrong exact type.")
    for field in ("new_evaluated_cell", "new_nondominated_cell"):
        value = payload.get(field)
        if value is not None and type(value) is not bool:
            raise ValueError("A V21e3 decision optional field has the wrong exact type.")
    witness = payload.get("policy_witness")
    if witness is not None and not isinstance(witness, dict):
        raise ValueError("A V21e3 decision policy witness must be an object or null.")


def _validate_terminal_exact_types(payload: Mapping[str, object]) -> None:
    """Validate terminal accounting fields without permissive coercion.

    ``cache_hit_count`` was absent from some legacy failure receipts, so it is
    optional only for those receipts.  Every current success receipt must bind
    it explicitly.
    """

    count_fields = (
        "attempt_count",
        "physical_call_started_count",
        "charged_evaluation_count",
        "decision_count",
        "unresolved_decision_count",
    )
    if any(type(payload.get(field)) is not int for field in count_fields):
        raise ValueError("A V21e3 terminal count field has the wrong exact type.")
    if "cache_hit_count" in payload and type(payload["cache_hit_count"]) is not int:
        raise ValueError("A V21e3 terminal count field has the wrong exact type.")
    if payload.get("status") == "SUCCESS" and type(payload.get("cache_hit_count")) is not int:
        raise ValueError("A successful V21e3 terminal receipt lacks an exact cache-hit count.")


def iter_v21e3_canonical_records(
    database_path: str | Path,
) -> Iterator[dict[str, object]]:
    """Yield every authoritative SQLite row in one frozen, JSON-safe order."""

    path = Path(database_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(_sqlite_read_only_uri(path), uri=True)
    connection.row_factory = sqlite3.Row
    table_order = (
        ("run_attempt", "run_attempt", "run_id"),
        ("solution", "solutions", "solution_ref"),
        ("attempt", "attempts", "attempt_index"),
        ("evaluation", "evaluations", "evaluation_index"),
        ("decision", "decisions", "evaluation_index"),
        ("terminal_receipt", "terminal_receipts", "run_id"),
    )
    record_index = 0
    try:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("V21e3 SQLite integrity_check failed before export.")
        for record_kind, table_name, order_column in table_order:
            for sqlite_row in connection.execute(
                f"SELECT * FROM {table_name} ORDER BY {order_column}"
            ):
                row: dict[str, object] = {}
                for key in sqlite_row.keys():
                    value = sqlite_row[key]
                    if isinstance(value, bytes):
                        row[f"{key}_hex"] = value.hex()
                    else:
                        row[str(key)] = value
                record_index += 1
                yield {
                    "schema": "v21e3_canonical_sqlite_record_v1",
                    "record_index": record_index,
                    "record_kind": record_kind,
                    "row": row,
                }
    finally:
        connection.close()


def verify_v21e3_trace_database(
    database_path: str | Path,
    problem: MultiObjectiveCombinatorialProblem,
    *,
    expected_run_context: Mapping[str, object],
    detached_terminal_receipt_path: str | Path,
    expected_detached_terminal_receipt_sha256: str,
    expected_charged_evaluations: int | None = None,
) -> dict[str, object]:
    """Replay the objective/archive scope under an external receipt anchor.

    A successful return authenticates the durable chains, re-evaluates every
    unique solution, and reconstructs the all-evaluated archive.  It is not a
    full algorithm-decision replay receipt.
    """

    path = Path(database_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    detached_path = Path(detached_terminal_receipt_path).resolve()
    if not detached_path.is_file():
        raise FileNotFoundError(detached_path)
    expected_detached_hash = str(expected_detached_terminal_receipt_sha256)
    if len(expected_detached_hash) != 64 or any(
        char not in "0123456789abcdef" for char in expected_detached_hash
    ):
        raise ValueError(
            "The expected detached V21e3 terminal receipt hash is not lowercase SHA-256."
        )
    detached_raw = detached_path.read_bytes()
    observed_detached_hash = hashlib.sha256(detached_raw).hexdigest()
    if observed_detached_hash != expected_detached_hash:
        raise ValueError(
            "The detached V21e3 terminal receipt fails its external SHA-256 binding."
        )
    try:
        detached_terminal = json.loads(detached_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "The detached V21e3 terminal receipt is not canonical UTF-8 JSON."
        ) from error
    if _canonical_bytes(detached_terminal) != detached_raw:
        raise ValueError(
            "The detached V21e3 terminal receipt is not canonical JSON."
        )
    # ``immutable=1`` is intentionally forbidden here: it can ignore a live WAL
    # and thereby verify a stale main-database image instead of the artifact state.
    uri = _sqlite_read_only_uri(path)
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("V21e3 SQLite integrity_check failed.")
        run = connection.execute(
            "SELECT problem,family,run_context_json,run_context_digest_sha256,"
            "status,terminal_receipt_sha256 "
            "FROM run_attempt WHERE run_id=1"
        ).fetchone()
        if run is None or str(run["status"]) not in {"SUCCESS", "FAILURE"}:
            raise ValueError("The V21e3 trace does not have a terminal run receipt.")
        if str(run["problem"]) != str(problem.name):
            raise ValueError("The V21e3 trace is bound to another problem.")
        expected_context_raw = _canonical_bytes(dict(expected_run_context))
        expected_context_digest = hashlib.sha256(expected_context_raw).hexdigest()
        if (
            str(run["run_context_json"]) != expected_context_raw.decode("utf-8")
            or str(run["run_context_digest_sha256"]) != expected_context_digest
        ):
            raise ValueError("The V21e3 run-context binding does not match expectation.")
        stored_context = json.loads(str(run["run_context_json"]))
        _validate_run_context_intrinsic(stored_context)
        problem_digest = hashlib.sha256(
            _canonical_bytes(problem.canonical_payload())
        ).hexdigest()
        if stored_context.get("problem_semantic_sha256") != problem_digest:
            raise ValueError("The V21e3 run context binds another problem semantic.")

        solutions: dict[int, tuple[Solution, str]] = {}
        solution_by_sha256: dict[str, Solution] = {}
        for row in connection.execute(
            """
            SELECT solution_ref,solution_sha256,codec,solution_size,payload
            FROM solutions ORDER BY solution_ref
            """
        ):
            solution = _decode_solution(
                str(row["codec"]),
                int(row["solution_size"]),
                bytes(row["payload"]),
            )
            digest = _solution_sha256(solution)
            if digest != str(row["solution_sha256"]):
                raise ValueError("A V21e3 packed solution fails its SHA-256 binding.")
            problem.validate_solution(solution)
            prior_solution = solution_by_sha256.get(digest)
            if prior_solution is not None and prior_solution != solution:
                raise ValueError(
                    "A V21e3 solution SHA-256 collision maps distinct exact solutions."
                )
            solution_by_sha256[digest] = solution
            solutions[int(row["solution_ref"])] = (solution, digest)

        attempts = list(
            connection.execute("SELECT * FROM attempts ORDER BY attempt_index")
        )
        attempt_by_index: dict[int, sqlite3.Row] = {}
        previous_attempt_hash = "0" * 64
        for expected_attempt, row in enumerate(attempts, start=1):
            attempt_index = int(row["attempt_index"])
            if attempt_index != expected_attempt:
                raise ValueError("V21e3 attempt indices are not contiguous.")
            raw_proposal = json.loads(str(row["proposal_json"]))
            if _canonical_bytes(raw_proposal).decode("utf-8") != str(
                row["proposal_json"]
            ):
                raise ValueError("A V21e3 raw proposal is not canonical JSON.")
            raw_sha = hashlib.sha256(str(row["proposal_json"]).encode("utf-8")).hexdigest()
            if raw_sha != str(row["proposal_raw_sha256"]):
                raise ValueError("A V21e3 raw proposal fails its SHA-256 binding.")
            proposal_ref = row["proposal_solution_ref"]
            if proposal_ref is not None:
                stored = solutions.get(int(proposal_ref))
                if stored is None or stored[1] != str(row["proposal_sha256"]):
                    raise ValueError("A V21e3 attempt disagrees with its solution row.")
                expected_raw_json, expected_raw_sha = _canonical_raw_proposal(stored[0])
                if (
                    expected_raw_json != str(row["proposal_json"])
                    or expected_raw_sha != raw_sha
                ):
                    raise ValueError(
                        "A valid V21e3 proposal raw representation is not exact."
                    )
            elif row["proposal_sha256"] is not None:
                raise ValueError("An invalid V21e3 proposal has a solution hash.")
            context = json.loads(str(row["context_json"]))
            if _canonical_bytes(context).decode("utf-8") != str(row["context_json"]):
                raise ValueError("A V21e3 attempt context is not canonical JSON.")
            if any(
                int(ref) not in solutions
                for ref in context.get("parent_solution_refs", [])
            ):
                raise ValueError("A V21e3 context references an unknown parent.")
            status = str(row["status"])
            physical = int(row["physical_call_started"])
            context_partition = context.get("evidence_partition")
            bound_partition = stored_context.get("evidence_partition")
            context_mismatch_failure = (
                status == "FAILED"
                and str(row["failure_code"]) == "EVALUATION_CONTEXT_MISMATCH"
            )
            if context_partition != bound_partition and not context_mismatch_failure:
                raise ValueError(
                    "A V21e3 attempt disagrees with the frozen evidence partition."
                )
            if context_partition == bound_partition and context_mismatch_failure:
                raise ValueError(
                    "A V21e3 context-mismatch failure has no actual mismatch."
                )
            if status == "EVALUATED" and (
                physical != 1
                or row["charged_evaluation_index"] is None
                or row["cache_source_evaluation_index"] is not None
            ):
                raise ValueError("An evaluated V21e3 attempt has invalid charge fields.")
            if status == "CACHE_HIT" and (
                physical != 0
                or row["charged_evaluation_index"] is not None
                or row["cache_source_evaluation_index"] is None
            ):
                raise ValueError("A cached V21e3 attempt has invalid charge fields.")
            if str(run["status"]) == "SUCCESS" and status not in {
                "EVALUATED",
                "CACHE_HIT",
            }:
                raise ValueError("A successful V21e3 run contains a failed attempt.")
            semantic = _attempt_semantic(
                row,
                run_context_digest_sha256=expected_context_digest,
                previous_attempt_hash=previous_attempt_hash,
            )
            observed_attempt_hash = hashlib.sha256(
                _canonical_bytes(semantic)
            ).hexdigest()
            if (
                str(row["prev_attempt_sha256"]) != previous_attempt_hash
                or str(row["attempt_sha256"]) != observed_attempt_hash
            ):
                raise ValueError(
                    f"Attempt semantic hash chain failed at attempt {attempt_index}."
                )
            previous_attempt_hash = observed_attempt_hash
            attempt_by_index[attempt_index] = row

        evaluations = list(
            connection.execute("SELECT * FROM evaluations ORDER BY evaluation_index")
        )
        if (
            expected_charged_evaluations is not None
            and len(evaluations) != int(expected_charged_evaluations)
        ):
            raise ValueError("V21e3 charged evaluation count does not match expectation.")
        previous_evaluation_hash = "0" * 64
        replayed_objectives: dict[str, tuple[float, ...]] = {}
        evaluated: dict[int, tuple[Solution, tuple[float, ...]]] = {}
        evaluation_attempt_index: dict[int, int] = {}
        charged_evaluation_by_solution: dict[Solution, int] = {}
        for expected_index, row in enumerate(evaluations, start=1):
            index = int(row["evaluation_index"])
            if index != expected_index:
                raise ValueError("V21e3 evaluation indices are not contiguous.")
            attempt = attempt_by_index.get(int(row["attempt_index"]))
            if attempt is None or str(attempt["status"]) != "EVALUATED":
                raise ValueError("A V21e3 evaluation lacks its evaluated attempt.")
            if int(attempt["charged_evaluation_index"]) != index:
                raise ValueError("A V21e3 attempt binds the wrong evaluation index.")
            proposal_ref = int(row["proposal_solution_ref"])
            if proposal_ref not in solutions:
                raise ValueError("A V21e3 evaluation references an unknown solution.")
            proposal, proposal_sha = solutions[proposal_ref]
            if proposal_sha != str(row["proposal_sha256"]):
                raise ValueError("A V21e3 evaluation has a mismatched solution hash.")
            prior_charge = charged_evaluation_by_solution.get(proposal)
            if prior_charge is not None:
                raise ValueError(
                    "A V21e3 charged evaluation repeats an exact canonical proposal; "
                    "a subsequent attempt must be a cache hit "
                    f"to evaluation {prior_charge}."
                )
            charged_evaluation_by_solution[proposal] = index
            objectives = decode_v21e3_objectives_json(
                str(row["objectives_json"]),
                expected_dimension=int(problem.num_objectives),
            )
            replayed = replayed_objectives.get(proposal_sha)
            if replayed is None:
                replayed = _validate_replayed_objective(
                    problem, problem.evaluate(proposal)
                )
                replayed_objectives[proposal_sha] = replayed
            if replayed != objectives:
                raise ValueError(f"Objective replay failed at evaluation {index}.")
            context = json.loads(str(attempt["context_json"]))
            for field in (
                "evidence_partition",
                "search_phase_id",
                "stage_id",
                "type_id",
                "operator_id",
                "operator_call_id",
            ):
                if row[field] != context[field]:
                    raise ValueError("Evaluation columns disagree with attempt context.")
            semantic = {
                "evaluation_index": index,
                "attempt_index": int(row["attempt_index"]),
                "context": context,
                "proposal_solution_ref": proposal_ref,
                "proposal_sha256": proposal_sha,
                "objectives": objectives,
                "run_context_digest_sha256": expected_context_digest,
                "prev_record_sha256": previous_evaluation_hash,
            }
            observed_hash = hashlib.sha256(_canonical_bytes(semantic)).hexdigest()
            if (
                str(row["prev_record_sha256"]) != previous_evaluation_hash
                or str(row["record_sha256"]) != observed_hash
            ):
                raise ValueError(
                    f"Evaluation semantic hash chain failed at evaluation {index}."
                )
            previous_evaluation_hash = observed_hash
            evaluated[index] = (proposal, objectives)
            evaluation_attempt_index[index] = int(row["attempt_index"])

        decisions = list(
            connection.execute("SELECT * FROM decisions ORDER BY evaluation_index")
        )
        previous_decision_hash = "0" * 64
        decision_by_index: dict[int, dict[str, object]] = {}
        v9_lyapunov_policy_witnesses_verified = 0
        for expected_index, row in enumerate(decisions, start=1):
            index = int(row["evaluation_index"])
            if index != expected_index or index not in evaluated:
                raise ValueError("V21e3 decision indices are not a valid prefix.")
            payload = json.loads(str(row["decision_json"]))
            if _canonical_bytes(payload).decode("utf-8") != str(row["decision_json"]):
                raise ValueError("A V21e3 decision payload is not canonical JSON.")
            if not isinstance(payload, Mapping):
                raise ValueError("A V21e3 decision payload is not an object.")
            _validate_decision_exact_types(payload)
            if payload["evaluation_index"] != index:
                raise ValueError("A V21e3 decision binds the wrong evaluation.")
            if payload.get("prev_decision_sha256") != previous_decision_hash:
                raise ValueError("A V21e3 decision embeds the wrong previous hash.")
            if payload.get("run_context_digest_sha256") != expected_context_digest:
                raise ValueError("A V21e3 decision embeds the wrong run context.")
            policy_witness = payload.get("policy_witness")
            if isinstance(policy_witness, Mapping) and policy_witness.get(
                "schema"
            ) == _V9_LYAPUNOV_WITNESS_SCHEMA:
                _validate_v9_lyapunov_policy_witness(policy_witness)
                v9_lyapunov_policy_witnesses_verified += 1
            observed_hash = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
            if (
                str(row["prev_decision_sha256"]) != previous_decision_hash
                or str(row["decision_sha256"]) != observed_hash
            ):
                raise ValueError(
                    f"Decision semantic hash chain failed at evaluation {index}."
                )
            previous_decision_hash = observed_hash
            decision_by_index[index] = payload

        algorithm_config = stored_context.get("algorithm_config")
        diagnostic_id = (
            algorithm_config.get("development_diagnostic_id")
            if isinstance(algorithm_config, Mapping)
            else None
        )
        is_v9 = type(diagnostic_id) is str and diagnostic_id.startswith(
            "V21E3R1_V9_"
        )
        v9_population_objectives: dict[int, tuple[float, ...]] = {}
        v9_population_policy_decisions_verified = 0
        if is_v9:
            if not isinstance(algorithm_config, Mapping):  # pragma: no cover
                raise AssertionError("V9 config narrowing failed.")
            v9_lower_raw = stored_context.get("objective_lower_bounds")
            v9_upper_raw = stored_context.get("objective_upper_bounds")
            if (
                not isinstance(v9_lower_raw, list)
                or not isinstance(v9_upper_raw, list)
                or len(v9_lower_raw) != int(problem.num_objectives)
                or len(v9_upper_raw) != int(problem.num_objectives)
            ):
                raise ValueError("V9 policy replay objective bounds drifted.")
            v9_lower = tuple(
                _exact_finite_real(value, label="V9 lower bound")
                for value in v9_lower_raw
            )
            v9_upper = tuple(
                _exact_finite_real(value, label="V9 upper bound")
                for value in v9_upper_raw
            )
            if any(hi <= lo for lo, hi in zip(v9_lower, v9_upper)):
                raise ValueError("V9 policy replay objective box is invalid.")
            v9_directions = _v9_reference_directions(
                algorithm_config,
                objective_dimension=int(problem.num_objectives),
            )
            v9_neighbors = _v9_type_neighbors(
                v9_directions,
                neighborhood_size=int(algorithm_config.get("neighborhood_size", 0)),
            )
            v9_central_direction = tuple(
                1.0 / int(problem.num_objectives)
                for _ in range(int(problem.num_objectives))
            )
            v9_scale = math.prod(
                hi - lo for lo, hi in zip(v9_lower, v9_upper)
            )
        archive = ParetoArchive(max_size=None, tol=0.0)
        for index, (proposal, objectives) in evaluated.items():
            normalized_hv_before = (
                archive.hypervolume_2d(reference=v9_upper) / v9_scale
                if is_v9
                else None
            )
            changed = archive.update((ArchiveEntry(proposal, objectives),))
            decision = decision_by_index.get(index)
            if decision is None:
                continue
            retained = archive.contains(ArchiveEntry(proposal, objectives))
            if (
                decision["archive_changed"] != changed
                or decision["retained_after_update"] != retained
                or decision["archive_size_after"] != len(archive)
            ):
                raise ValueError(f"Archive decision replay failed at evaluation {index}.")
            if not is_v9:
                continue
            attempt = attempt_by_index[evaluation_attempt_index[index]]
            context = json.loads(str(attempt["context_json"]))
            type_id = context.get("type_id")
            if type(type_id) is not int or not 0 <= type_id < len(v9_directions):
                raise ValueError("V9 population policy replay type id drifted.")
            raw_targets = decision.get("population_target_type_ids")
            if not isinstance(raw_targets, list) or any(
                type(target) is not int or not 0 <= target < len(v9_directions)
                for target in raw_targets
            ):
                raise ValueError("V9 population policy replay targets drifted.")
            observed_targets = tuple(raw_targets)
            stage = context.get("stage_id")
            if stage == "initialization_v21e3":
                expected_targets = (type_id,)
            elif stage == "search_v21e3":
                considered = v9_neighbors[type_id]
                replacement_policy = algorithm_config.get("replacement_policy")
                if replacement_policy == (
                    "archive_compensated_information_lyapunov_development_v1"
                ):
                    witness = decision.get("policy_witness")
                    if not isinstance(witness, Mapping):
                        raise ValueError("V9 Lyapunov policy witness is missing.")
                    decision_selected_targets = (
                        _validate_v9_lyapunov_policy_witness(witness)
                    )
                    direction_for = lambda target: (
                        v9_central_direction
                        if algorithm_config.get("candidate_id") == "C0"
                        else v9_directions[target]
                    )
                    expected_empty_targets = tuple(
                        target
                        for target in considered
                        if target not in v9_population_objectives
                    )
                    if tuple(
                        witness.get("preselected_empty_target_type_ids", ())
                    ) != expected_empty_targets:
                        raise ValueError(
                            "V9 Lyapunov empty targets disagree with durable "
                            "population state."
                        )
                    expected_finite_capacity = len(considered) - len(
                        expected_empty_targets
                    )
                    if (
                        witness.get("finite_selection_capacity")
                        != expected_finite_capacity
                    ):
                        raise ValueError(
                            "V9 Lyapunov finite selection capacity disagrees "
                            "with durable population state."
                        )
                    expected_deltas = {
                        target: (
                            float("-inf")
                            if target not in v9_population_objectives
                            else _v9_scalar(
                                objectives,
                                direction_for(target),
                                lower=v9_lower,
                                upper=v9_upper,
                            )
                            - _v9_scalar(
                                v9_population_objectives[target],
                                direction_for(target),
                                lower=v9_lower,
                                upper=v9_upper,
                            )
                        )
                        for target in considered
                    }
                    finite_expected = {
                        target: delta
                        for target, delta in expected_deltas.items()
                        if math.isfinite(delta)
                    }
                    raw_delta_rows = witness.get("finite_scalar_delta_by_target")
                    if not isinstance(raw_delta_rows, list):
                        raise ValueError("V9 Lyapunov policy witness omits raw deltas.")
                    observed_deltas = {
                        int(row["target_type_id"]): float(row["scalar_delta"])
                        for row in raw_delta_rows
                        if isinstance(row, Mapping)
                        and type(row.get("target_type_id")) is int
                        and type(row.get("scalar_delta")) in {int, float}
                    }
                    if set(observed_deltas) != set(finite_expected) or any(
                        not math.isclose(
                            observed_deltas[target],
                            finite_expected[target],
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                        for target in finite_expected
                    ):
                        raise ValueError("V9 Lyapunov policy witness scalar deltas drifted.")
                    normalized_hv_after = (
                        archive.hypervolume_2d(reference=v9_upper) / v9_scale
                    )
                    expected_gain = max(
                        0.0, normalized_hv_after - float(normalized_hv_before)
                    )
                    _same_real(
                        witness.get("normalized_hv_before"),
                        float(normalized_hv_before),
                        label="normalized_hv_before",
                    )
                    _same_real(
                        witness.get("normalized_hv_after"),
                        normalized_hv_after,
                        label="normalized_hv_after",
                    )
                    _same_real(
                        witness.get("normalized_hv_gain"),
                        expected_gain,
                        label="normalized_hv_gain",
                    )
                    _same_real(
                        witness.get("tradeoff_lambda"),
                        float(algorithm_config.get("archive_tradeoff_lambda")),
                        label="tradeoff_lambda",
                    )
                    if tuple(witness.get("considered_target_type_ids", ())) != considered:
                        raise ValueError("V9 Lyapunov policy witness neighborhood drifted.")
                    expected_targets = (
                        expected_empty_targets + decision_selected_targets
                    )
                    if tuple(witness.get("selected_target_type_ids", ())) != (
                        expected_targets
                    ):
                        raise ValueError(
                            "V9 Lyapunov final targets disagree with durable "
                            "population replay."
                        )
                else:
                    direction_for = lambda target: (
                        v9_central_direction
                        if algorithm_config.get("candidate_id") == "C0"
                        else v9_directions[target]
                    )
                    expected_targets = tuple(
                        target
                        for target in considered
                        if target not in v9_population_objectives
                        or _v9_scalar(
                            objectives,
                            direction_for(target),
                            lower=v9_lower,
                            upper=v9_upper,
                        )
                        <= _v9_scalar(
                            v9_population_objectives[target],
                            direction_for(target),
                            lower=v9_lower,
                            upper=v9_upper,
                        )
                    )
            else:
                raise ValueError("V9 population policy replay stage drifted.")
            if (
                observed_targets != expected_targets
                or decision.get("accepted_into_population") is not bool(expected_targets)
                or decision.get("population_replacement_count") != len(expected_targets)
            ):
                raise ValueError("V9 population replacement decision replay failed.")
            for target in expected_targets:
                v9_population_objectives[target] = objectives
            v9_population_policy_decisions_verified += 1

        physical_call_starts = sum(
            int(attempt["physical_call_started"]) for attempt in attempts
        )
        cache_hits = 0
        for attempt in attempts:
            if str(attempt["status"]) != "CACHE_HIT":
                continue
            cache_hits += 1
            source = int(attempt["cache_source_evaluation_index"])
            if source not in evaluated:
                raise ValueError("A V21e3 cache hit references an unknown evaluation.")
            if evaluation_attempt_index[source] >= int(attempt["attempt_index"]):
                raise ValueError(
                    "A V21e3 cache hit does not reference an earlier attempt."
                )
            source_solution, _ = evaluated[source]
            proposal_ref = attempt["proposal_solution_ref"]
            if proposal_ref is None or int(proposal_ref) not in solutions:
                raise ValueError("A V21e3 cache hit has no exact proposal solution.")
            cached_solution, cached_sha = solutions[int(proposal_ref)]
            if (
                cached_solution != source_solution
                or cached_sha != str(attempt["proposal_sha256"])
            ):
                raise ValueError("A V21e3 cache hit references a different solution.")

        terminal_row = connection.execute(
            "SELECT * FROM terminal_receipts WHERE run_id=1"
        ).fetchone()
        if terminal_row is None:
            raise ValueError("The V21e3 terminal receipt row is missing.")
        terminal = json.loads(str(terminal_row["receipt_json"]))
        if (
            str(terminal_row["receipt_json"]).encode("utf-8") != detached_raw
            or terminal != detached_terminal
        ):
            raise ValueError(
                "The SQLite and detached V21e3 terminal receipts disagree."
            )
        terminal_core = dict(terminal)
        embedded_hash = terminal_core.pop("receipt_payload_sha256", None)
        terminal_hash = hashlib.sha256(_canonical_bytes(terminal_core)).hexdigest()
        if not (
            embedded_hash
            == terminal_hash
            == str(terminal_row["receipt_sha256"])
            == str(run["terminal_receipt_sha256"])
        ):
            raise ValueError("The V21e3 terminal receipt hash binding failed.")
        if terminal.get("status") != str(run["status"]):
            raise ValueError("The V21e3 terminal statuses disagree.")
        if terminal.get("terminal_evaluation_chain_sha256") != previous_evaluation_hash:
            raise ValueError("The terminal evaluation chain binding failed.")
        if terminal.get("terminal_decision_chain_sha256") != previous_decision_hash:
            raise ValueError("The terminal decision chain binding failed.")
        if terminal.get("terminal_attempt_chain_sha256") != previous_attempt_hash:
            raise ValueError("The terminal attempt chain binding failed.")
        if terminal.get("run_context_digest_sha256") != expected_context_digest:
            raise ValueError("The terminal run-context binding failed.")
        _validate_terminal_exact_types(terminal)
        unresolved_decisions = len(evaluations) - len(decisions)
        if unresolved_decisions < 0:
            raise ValueError("The V21e3 trace has more decisions than evaluations.")
        if terminal["attempt_count"] != len(attempts):
            raise ValueError("The terminal attempt count binding failed.")
        if terminal["physical_call_started_count"] != physical_call_starts:
            raise ValueError("The terminal physical-start count binding failed.")
        if terminal["charged_evaluation_count"] != len(evaluations):
            raise ValueError("The terminal evaluation count binding failed.")
        if terminal["decision_count"] != len(decisions):
            raise ValueError("The terminal decision count binding failed.")
        if terminal["unresolved_decision_count"] != unresolved_decisions:
            raise ValueError("The terminal unresolved-decision count binding failed.")
        if "cache_hit_count" in terminal and terminal["cache_hit_count"] != cache_hits:
            raise ValueError("The terminal cache-hit count binding failed.")
        v9_resource_contract_verified = False
        screen_witnesses_verified = 0
        if str(run["status"]) == "SUCCESS":
            if terminal.get("cache_hit_count") != cache_hits:
                raise ValueError("The successful terminal cache-hit count binding failed.")
            if physical_call_starts != len(evaluations):
                raise ValueError(
                    "A successful V21e3 trace has an unresolved physical start."
                )
            if unresolved_decisions != 0:
                raise ValueError("A successful V21e3 trace has an unresolved decision.")
            if len(attempts) != len(evaluations) + cache_hits:
                raise ValueError(
                    "A successful V21e3 trace violates attempt = evaluation + cache-hit accounting."
                )
            if type(diagnostic_id) is str and diagnostic_id.startswith(
                "V21E3R1_V9_"
            ):
                resources = terminal.get("resource_accounting")
                if not isinstance(resources, dict):
                    raise ValueError(
                        "A successful V9 trace lacks terminal A/S/T accounting."
                    )
                integer_fields = (
                    "first_evaluations",
                    "first_evaluation_cap",
                    "attempts",
                    "attempt_cap",
                    "structural_candidate_generations",
                    "cache_membership_probes",
                    "structural_screening_work",
                    "structural_screening_cap",
                )
                if any(type(resources.get(field)) is not int for field in integer_fields):
                    raise ValueError("V9 terminal resource counts lack exact integer types.")
                if resources.get("schema") != (
                    "v21e3r1_v9_ast_resource_accounting_v1"
                ):
                    raise ValueError("The V9 terminal resource schema is invalid.")
                generated_from_witnesses = 0
                probes_from_witnesses = 0
                screen_witnesses_verified = 0
                evaluated_solution_sha256: set[str] = set()
                for attempt in attempts:
                    context = json.loads(str(attempt["context_json"]))
                    operator_witness = context.get("operator_witness")
                    if isinstance(operator_witness, dict):
                        screen = operator_witness.get(
                            "information_time_candidate_screen"
                        )
                        if isinstance(screen, dict):
                            generated = screen.get("structural_candidates_generated")
                            probes = screen.get("cache_membership_probes")
                            total = screen.get("total_structural_screening_work")
                            if (
                                type(generated) is not int
                                or generated < 0
                                or type(probes) is not int
                                or probes < 0
                                or type(total) is not int
                                or total != generated + probes
                            ):
                                raise ValueError(
                                    "A V9 candidate-screen witness has invalid resource counts."
                                )
                            _validate_v9_candidate_screen_witness(
                                screen,
                                evaluated_solution_sha256_before_attempt=(
                                    evaluated_solution_sha256
                                ),
                                selected_attempt_sha256=str(
                                    attempt["proposal_sha256"]
                                ),
                                problem=problem,
                            )
                            screen_witnesses_verified += 1
                            generated_from_witnesses += generated
                            probes_from_witnesses += probes
                    if str(attempt["status"]) == "EVALUATED":
                        evaluated_solution_sha256.add(
                            str(attempt["proposal_sha256"])
                        )
                elapsed = resources.get("elapsed_seconds")
                wall_cap = resources.get("wall_time_cap_seconds")
                if (
                    type(elapsed) not in {int, float}
                    or not math.isfinite(float(elapsed))
                    or float(elapsed) < 0.0
                ):
                    raise ValueError("The V9 terminal elapsed time is invalid.")
                if wall_cap is not None and (
                    type(wall_cap) not in {int, float}
                    or not math.isfinite(float(wall_cap))
                    or float(wall_cap) <= 0.0
                ):
                    raise ValueError("The V9 terminal wall-time cap is invalid.")
                if not (
                    resources["first_evaluations"] == len(evaluations)
                    and resources["first_evaluation_cap"]
                    == algorithm_config.get("charged_evaluations")
                    and resources["attempts"] == len(attempts)
                    and resources["attempt_cap"]
                    == algorithm_config.get("attempt_cap")
                    and resources["structural_candidate_generations"]
                    == generated_from_witnesses
                    and resources["cache_membership_probes"]
                    == probes_from_witnesses
                    and resources["structural_screening_work"]
                    == generated_from_witnesses + probes_from_witnesses
                    and resources["structural_screening_cap"]
                    == algorithm_config.get("structural_screening_cap")
                    and resources["first_evaluations"]
                    <= resources["first_evaluation_cap"]
                    and resources["attempts"] <= resources["attempt_cap"]
                    and resources["structural_screening_work"]
                    <= resources["structural_screening_cap"]
                    and (wall_cap is None or float(elapsed) <= float(wall_cap))
                    and resources.get("all_configured_caps_satisfied") is True
                    and terminal.get("finalization_gates", {}).get(
                        "resource_accounting"
                    )
                    == resources
                    and terminal.get("finalization_gates", {}).get(
                        "resource_accounting_gate"
                    )
                    == "PASS"
                ):
                    raise ValueError("The V9 terminal A/S/T accounting does not replay.")
                v9_resource_contract_verified = True

        artifact_bytes = path.read_bytes()
        return {
            "schema": "v21e3r1_objective_archive_replay_receipt_v2",
            "status": "OBJECTIVE_AND_ARCHIVE_REPLAY_PASS",
            "verification_scope": (
                "objective_solution_chain_archive_and_terminal_replay_v1"
            ),
            "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
            "selection_authorization": "PROHIBITED",
            "database_path": str(path),
            "database_bytes": len(artifact_bytes),
            "database_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "detached_terminal_receipt_path": str(detached_path),
            "detached_terminal_receipt_sha256": observed_detached_hash,
            "attempt_records": len(attempts),
            "physical_call_started_records": physical_call_starts,
            "evaluation_records": len(evaluations),
            "decision_records": len(decisions),
            "unresolved_decision_records": unresolved_decisions,
            "cache_hit_records": cache_hits,
            "unique_solution_replays": len(replayed_objectives),
            "archive_reconstruction": "PASS",
            "archive_size": len(archive),
            "terminal_status": str(run["status"]),
            "run_context_digest_sha256": expected_context_digest,
            "terminal_attempt_chain_sha256": previous_attempt_hash,
            "terminal_evaluation_chain_sha256": previous_evaluation_hash,
            "terminal_decision_chain_sha256": previous_decision_hash,
            "terminal_receipt_sha256": terminal_hash,
            "v9_resource_contract_replay": (
                "PASS" if v9_resource_contract_verified else "NOT_APPLICABLE"
            ),
            "v9_lyapunov_policy_witness_replay": (
                "PASS"
                if v9_lyapunov_policy_witnesses_verified > 0
                else "NOT_APPLICABLE"
            ),
            "v9_lyapunov_policy_witnesses_verified": (
                v9_lyapunov_policy_witnesses_verified
            ),
            "v9_candidate_screen_witness_replay": (
                "PASS"
                if v9_resource_contract_verified and screen_witnesses_verified > 0
                else "NOT_APPLICABLE"
            ),
            "v9_candidate_screen_witnesses_verified": (
                screen_witnesses_verified if v9_resource_contract_verified else 0
            ),
            "v9_population_policy_replay": (
                "PASS" if is_v9 else "NOT_APPLICABLE"
            ),
            "v9_population_policy_decisions_verified": (
                v9_population_policy_decisions_verified
            ),
        }
    finally:
        connection.close()


__all__ = [
    "iter_v21e3_canonical_records",
    "verify_v21e3_trace_database",
]
