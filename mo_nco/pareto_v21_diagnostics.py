from __future__ import annotations

"""Read-only, evidence-status-aware diagnostics for V21 trace databases.

The module deliberately derives only quantities represented in the finalized
SQLite ledger.  In particular, an absent hypervolume delta is reported as
``NOT_ESTABLISHED`` rather than silently converted to a numerical zero.
"""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics
from typing import Iterable, Mapping, Sequence


_CHECKPOINT_LABELS = (
    "init_end",
    "early_10pct",
    "mid_70pct",
    "budget_end",
)


def _established(
    value: float | int,
    *,
    numerator: int | None = None,
    denominator: int | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {"status": "ESTABLISHED", "value": value}
    if numerator is not None:
        result["numerator"] = numerator
    if denominator is not None:
        result["denominator"] = denominator
    return result


def _not_established(reason: str) -> dict[str, str]:
    return {"status": "NOT_ESTABLISHED", "reason": reason}


def _not_applicable(reason: str) -> dict[str, str]:
    return {"status": "NOT_APPLICABLE", "reason": reason}


def _rate(numerator: int, denominator: int) -> dict[str, object]:
    if denominator == 0:
        return _not_applicable("The denominator is zero for this evidence slice.")
    return _established(
        numerator / denominator,
        numerator=numerator,
        denominator=denominator,
    )


def _read_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in connection.execute(
            "SELECT key,value FROM metadata ORDER BY key"
        )
    }


def _validate_finalized_trace(
    connection: sqlite3.Connection,
) -> tuple[dict[str, str], int]:
    integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    if integrity != "ok":
        raise ValueError(f"V21 trace SQLite integrity is not ok: {integrity}")
    metadata = _read_metadata(connection)
    if metadata.get("schema") != "v21_sqlite_evaluation_trace_v1":
        raise ValueError("Unsupported or missing V21 trace schema metadata.")
    if metadata.get("status") != "FINALIZED":
        raise ValueError("Diagnostics require a finalized V21 trace.")
    try:
        budget = int(metadata["expected_budget"])
    except (KeyError, ValueError) as error:
        raise ValueError("The finalized trace has no valid expected budget.") from error
    evaluations = int(
        connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
    )
    decisions = int(
        connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    )
    bounds = connection.execute(
        "SELECT MIN(evaluation_index),MAX(evaluation_index) FROM evaluations"
    ).fetchone()
    if evaluations != budget or decisions != budget or bounds != (1, budget):
        raise ValueError(
            "Finalized V21 trace rows do not match the declared exact budget."
        )
    return metadata, budget


def _slice_metrics(
    connection: sqlite3.Connection,
    *,
    start: int,
    end: int,
) -> dict[str, object]:
    if end < start:
        return {
            "status": "NOT_APPLICABLE",
            "reason": "This phase has no evaluations because its boundaries coincide.",
            "evaluation_start": start,
            "evaluation_end": end,
            "evaluations": 0,
        }
    row = connection.execute(
        """
        SELECT COUNT(*) AS evaluations,
               COUNT(DISTINCT e.proposal_sha256) AS distinct_states_in_slice,
               SUM(CASE WHEN e.duplicate_of_evaluation_index IS NULL
                        THEN 1 ELSE 0 END) AS first_time_states,
               SUM(CASE WHEN e.duplicate_of_evaluation_index IS NOT NULL
                        THEN 1 ELSE 0 END) AS duplicates,
               SUM(d.archive_changed) AS archive_changes,
               SUM(d.retained_after_update) AS retained,
               SUM(d.accepted_into_population) AS accepted
        FROM evaluations AS e
        JOIN decisions AS d USING(evaluation_index)
        WHERE e.evaluation_index BETWEEN ? AND ?
        """,
        (start, end),
    ).fetchone()
    evaluations = int(row[0])
    distinct_states_in_slice = int(row[1])
    unique_states = int(row[2])
    duplicates = int(row[3])
    archive_changes = int(row[4])
    retained = int(row[5])
    accepted = int(row[6])
    archive_size_at_end = int(
        connection.execute(
            "SELECT archive_size_after FROM decisions WHERE evaluation_index=?",
            (end,),
        ).fetchone()[0]
    )
    return {
        "status": "ESTABLISHED",
        "evaluation_start": start,
        "evaluation_end": end,
        "evaluations": evaluations,
        "unique_states": unique_states,
        "distinct_states_in_slice": distinct_states_in_slice,
        "duplicates": duplicates,
        "archive_changes": archive_changes,
        "archive_size_at_end": archive_size_at_end,
        "retained_after_update": retained,
        "accepted_into_population": accepted,
        "unique_evaluation_rate": _rate(unique_states, evaluations),
        "duplicate_rate": _rate(duplicates, evaluations),
        "archive_change_rate": _rate(archive_changes, evaluations),
        "archive_retention_rate": _rate(retained, evaluations),
        "population_acceptance_rate": _rate(accepted, evaluations),
    }


def _population_snapshots(
    connection: sqlite3.Connection,
) -> list[dict[str, object]]:
    snapshots: list[dict[str, object]] = []
    for evaluation_index, raw_payload in connection.execute(
        """
        SELECT after_evaluation_index,payload_json
        FROM mechanisms
        WHERE event_kind='population_snapshot'
        ORDER BY after_evaluation_index,event_index
        """
    ):
        payload = json.loads(str(raw_payload))
        if not isinstance(payload, dict):
            raise ValueError("A V21 population snapshot payload must be an object.")
        snapshots.append(
            {
                "after_evaluation_index": int(evaluation_index),
                "payload": payload,
            }
        )
    return snapshots


def _checkpoint_boundaries(
    connection: sqlite3.Connection,
    *,
    budget: int,
    snapshots: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    recorded: dict[str, int] = {}
    for snapshot in snapshots:
        payload = snapshot["payload"]
        assert isinstance(payload, Mapping)
        labels = payload.get("boundary_labels", ())
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise ValueError("Population snapshot boundary_labels must be a sequence.")
        for label in labels:
            if label in _CHECKPOINT_LABELS:
                if label in recorded:
                    raise ValueError(f"Duplicate V21 population boundary label: {label}")
                recorded[str(label)] = int(snapshot["after_evaluation_index"])

    init_row = connection.execute(
        """
        SELECT MAX(evaluation_index)
        FROM evaluations
        WHERE search_phase_id='initialization'
        """
    ).fetchone()
    if init_row[0] is None:
        raise ValueError("The V21 trace does not establish an initialization boundary.")
    init_end = int(init_row[0])
    inferred = {
        "init_end": init_end,
        "early_10pct": max(init_end, int(math.ceil(0.10 * budget))),
        "mid_70pct": max(init_end, int(math.ceil(0.70 * budget))),
        "budget_end": budget,
    }
    boundaries = {label: recorded.get(label, inferred[label]) for label in _CHECKPOINT_LABELS}
    if not (
        1 <= boundaries["init_end"]
        <= boundaries["early_10pct"]
        <= boundaries["mid_70pct"]
        <= boundaries["budget_end"]
        == budget
    ):
        raise ValueError(f"Invalid V21 diagnostic boundaries: {boundaries}")
    return boundaries


def _operator_quality(connection: sqlite3.Connection) -> list[dict[str, object]]:
    groups: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in connection.execute(
        """
        SELECT e.operator_id,e.duplicate_of_evaluation_index,
               e.feasible_before_repair,e.repair_applied,e.local_search_depth,
               d.accepted_into_population,d.archive_changed,
               d.retained_after_update,d.scalar_advantage
        FROM evaluations AS e
        JOIN decisions AS d USING(evaluation_index)
        ORDER BY e.evaluation_index
        """
    ):
        groups[str(row[0])].append(row)

    result: list[dict[str, object]] = []
    for operator_id in sorted(groups):
        rows = groups[operator_id]
        attempts = len(rows)
        duplicates = sum(row[1] is not None for row in rows)
        accepted = sum(int(row[5]) for row in rows)
        archive_changes = sum(int(row[6]) for row in rows)
        retained = sum(int(row[7]) for row in rows)
        feasible_values = [int(row[2]) for row in rows if row[2] is not None]
        scalar_values = [float(row[8]) for row in rows if row[8] is not None]
        local_depth_values = [int(row[4]) for row in rows if row[4] is not None]
        feasible_rate = (
            _rate(sum(feasible_values), len(feasible_values))
            if feasible_values
            else _not_established(
                "feasible_before_repair is absent for every attempt of this operator."
            )
        )
        if feasible_values:
            feasible_rate["missing_records"] = attempts - len(feasible_values)
        scalar_rate = (
            _rate(sum(value > 0.0 for value in scalar_values), len(scalar_values))
            if scalar_values
            else _not_established(
                "scalar_advantage is absent for every attempt of this operator."
            )
        )
        if scalar_values:
            scalar_rate["missing_records"] = attempts - len(scalar_values)
        result.append(
            {
                "operator_id": operator_id,
                "attempts": attempts,
                "duplicates": duplicates,
                "accepted_into_population": accepted,
                "archive_changes": archive_changes,
                "retained_after_update": retained,
                "duplicate_rate": _rate(duplicates, attempts),
                "feasible_before_repair_rate": feasible_rate,
                "repair_rate": _rate(sum(int(row[3]) for row in rows), attempts),
                "population_acceptance_rate": _rate(
                    accepted, attempts
                ),
                "archive_change_rate": _rate(
                    archive_changes, attempts
                ),
                "archive_entry_rate": _rate(
                    archive_changes, attempts
                ),
                "archive_retention_rate": _rate(
                    retained, attempts
                ),
                "scalar_improvement_rate": scalar_rate,
                "mean_local_search_depth": _established(
                    sum(local_depth_values) / len(local_depth_values)
                ),
                "mean_positive_hv_gain": _not_established(
                    "Per-evaluation hypervolume delta is not recorded in trace schema v1."
                ),
                "local_search_gain": _not_established(
                    "The ledger records local-search depth but no pre/post block objective pair."
                ),
            }
        )
    return result


def _d4_snapshot_diagnostics(
    connection: sqlite3.Connection,
    snapshots: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not snapshots:
        return {
            "status": "NOT_ESTABLISHED",
            "reason": "No population_snapshot mechanism events are present.",
            "snapshots": [],
        }
    rows: list[dict[str, object]] = []
    for snapshot in snapshots:
        boundary = int(snapshot["after_evaluation_index"])
        payload = snapshot["payload"]
        assert isinstance(payload, Mapping)
        per_type = [
            {
                "type_id": int(type_id),
                "unique_evaluated_states": int(unique_states),
                "evaluations": int(evaluations),
            }
            for type_id, unique_states, evaluations in connection.execute(
                """
                SELECT type_id,COUNT(DISTINCT proposal_sha256),COUNT(*)
                FROM evaluations
                WHERE evaluation_index <= ? AND type_id IS NOT NULL
                GROUP BY type_id
                ORDER BY type_id
                """,
                (boundary,),
            )
        ]
        rows.append(
            {
                "after_evaluation_index": boundary,
                "boundary_labels": list(payload.get("boundary_labels", ())),
                "source_population_snapshot": dict(payload),
                "population_unique_count": (
                    _established(int(payload["population_unique_count"]))
                    if "population_unique_count" in payload
                    else _not_established("The population snapshot omits unique count.")
                ),
                "population_unique_fraction": (
                    _established(float(payload["population_unique_fraction"]))
                    if "population_unique_fraction" in payload
                    else _not_established("The population snapshot omits unique fraction.")
                ),
                "per_type_unique_evaluated_states": per_type,
                "type_specific_archive_contribution": (
                    {
                        "status": "ESTABLISHED",
                        "by_type": list(payload["current_archive_contribution_by_type"]),
                    }
                    if "current_archive_contribution_by_type" in payload
                    else _not_established(
                        "The population snapshot omits type-specific archive contribution."
                    )
                ),
                "reference_region_coverage_count": (
                    _established(int(payload["reference_region_coverage_count"]))
                    if "reference_region_coverage_count" in payload
                    else _not_established(
                        "The population snapshot omits reference-region coverage."
                    )
                ),
                "resampling_ess_over_population": payload.get(
                    "resampling_ess_over_population",
                    _not_established("The population snapshot omits ESS evidence."),
                ),
                "ancestor_multiplicity": payload.get(
                    "ancestor_multiplicity",
                    _not_established(
                        "The population snapshot omits ancestor-multiplicity evidence."
                    ),
                ),
                "post_resampling_duplicate_rate": (
                    _not_applicable(
                        "No particle-resampling mechanism is used by this V21 candidate."
                    )
                    if isinstance(payload.get("resampling_ess_over_population"), Mapping)
                    and payload["resampling_ess_over_population"].get("status")
                    == "NOT_APPLICABLE"
                    else _not_established(
                        "No post-resampling duplicate statistic is recorded."
                    )
                ),
            }
        )
    return {"status": "ESTABLISHED", "snapshots": rows}


def analyze_v21_trace(database_path: str | Path) -> dict[str, object]:
    """Derive D1--D4 diagnostics from one finalized V21 SQLite ledger."""

    path = Path(database_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    trace_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    uri = path.as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        metadata, budget = _validate_finalized_trace(connection)
        snapshots = _population_snapshots(connection)
        boundaries = _checkpoint_boundaries(
            connection,
            budget=budget,
            snapshots=snapshots,
        )
        checkpoints = [
            {
                "label": label,
                "evaluation_index": boundaries[label],
                "metrics": _slice_metrics(
                    connection,
                    start=1,
                    end=boundaries[label],
                ),
                "population_snapshot_status": (
                    "ESTABLISHED"
                    if any(
                        label
                        in snapshot["payload"].get("boundary_labels", ())
                        for snapshot in snapshots
                    )
                    else "NOT_ESTABLISHED"
                ),
            }
            for label in _CHECKPOINT_LABELS
        ]
        segment_bounds = (
            ("init", 1, boundaries["init_end"]),
            ("early", boundaries["init_end"] + 1, boundaries["early_10pct"]),
            ("mid", boundaries["early_10pct"] + 1, boundaries["mid_70pct"]),
            ("tail", boundaries["mid_70pct"] + 1, boundaries["budget_end"]),
        )
        d1 = {
            "status": "ESTABLISHED",
            "segments": [
                {"segment": label, **_slice_metrics(connection, start=start, end=end)}
                for label, start, end in segment_bounds
            ],
            "comparative_loss": _not_established(
                "A single-run ledger has no matched comparator trajectory."
            ),
        }
        full = _slice_metrics(connection, start=1, end=budget)
        d2 = {
            "status": "ESTABLISHED",
            "evaluations": full["evaluations"],
            "unique_states": full["unique_states"],
            "duplicates": full["duplicates"],
            "archive_changes": full["archive_changes"],
            "unique_evaluation_rate": full["unique_evaluation_rate"],
            "duplicate_rate": full["duplicate_rate"],
            "archive_improving_evaluation_rate": full["archive_change_rate"],
            "positive_hv_gain_per_evaluation": _not_established(
                "Trace schema v1 does not persist a fixed-reference per-evaluation HV delta."
            ),
        }
        result = {
            "schema": "v21_trace_diagnostics_v1",
            "status": "PASS",
            "trace_database_sha256": trace_sha256,
            "trace_schema": metadata["schema"],
            "problem": metadata.get("problem"),
            "family": metadata.get("family"),
            "evidence_partitions": [
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT evidence_partition FROM evaluations ORDER BY 1"
                )
            ],
            "expected_budget": budget,
            "phase_checkpoints": checkpoints,
            "d1_phase_localization": d1,
            "d2_evaluation_efficiency": d2,
            "d3_operator_quality": _operator_quality(connection),
            "d4_typed_population_collapse": _d4_snapshot_diagnostics(
                connection, snapshots
            ),
            "evidence_boundary": {
                "hypervolume_delta": "NOT_ESTABLISHED",
                "cross_run_performance_difference": "NOT_ESTABLISHED",
                "evaluation_rows_are_inferential_units": False,
            },
        }
    return result


def write_v21_diagnostics_receipt(
    database_path: str | Path,
    output_path: str | Path,
    *,
    run_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write a canonical JSON D1--D4 receipt without silent replacement.

    Repeating the same request is idempotent.  If an existing path contains
    different bytes, the function fails closed instead of overwriting an
    evidence artifact.
    """

    payload = analyze_v21_trace(database_path)
    if run_identity is not None:
        payload["run_identity"] = {
            str(key): value for key, value in sorted(run_identity.items())
        }
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    destination = Path(output_path).resolve()
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise FileExistsError(
                f"Refusing to replace a different V21 diagnostics receipt: {destination}"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encoded)
    return {
        "schema": "v21_diagnostics_file_receipt_v1",
        "status": "PASS",
        "path": str(destination),
        "bytes": len(encoded),
        "receipt_sha256": hashlib.sha256(encoded).hexdigest(),
        "trace_database_sha256": payload["trace_database_sha256"],
    }


def _empirical_quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("A quantile requires at least one value.")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return float(
        sorted_values[lower]
        + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def _cluster_bootstrap_ci(
    differences: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    if len(differences) < 2:
        return _not_established(
            "At least two independent paired clusters are required for a CI."
        )
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    generator = random.Random(f"v21-cluster-bootstrap:{int(seed)}")
    count = len(differences)
    estimates = sorted(
        math.fsum(differences[generator.randrange(count)] for _ in range(count))
        / count
        for _ in range(samples)
    )
    return {
        "status": "ESTABLISHED",
        "method": "paired_cluster_percentile_bootstrap",
        "confidence": 0.95,
        "lower": _empirical_quantile(estimates, 0.025),
        "upper": _empirical_quantile(estimates, 0.975),
        "samples": samples,
        "randomization_seed": int(seed),
        "resampling_unit": "paired_case_cluster",
    }


def _sign_flip_test(
    differences: Sequence[float],
    *,
    seed: int,
    exact_limit: int,
    monte_carlo_samples: int,
) -> dict[str, object]:
    nonzero = tuple(value for value in differences if value != 0.0)
    observed = abs(math.fsum(differences) / len(differences))

    def is_extreme(signs: Iterable[int]) -> bool:
        randomized = math.fsum(
            sign * value for sign, value in zip(signs, nonzero)
        ) / len(differences)
        return abs(randomized) >= observed - 1e-15

    if len(nonzero) <= exact_limit:
        total = 1 << len(nonzero)
        extreme = sum(
            is_extreme(
                1 if (mask >> index) & 1 else -1
                for index in range(len(nonzero))
            )
            for mask in range(total)
        )
        probability = extreme / total
        return {
            "status": "ESTABLISHED",
            "method": "exact_cluster_sign_flip",
            "alternative": "two_sided",
            "two_sided_p": probability,
            "randomizations": total,
            "nonzero_cluster_count": len(nonzero),
        }
    if monte_carlo_samples <= 0:
        raise ValueError("sign_flip_samples must be positive.")
    generator = random.Random(f"v21-cluster-sign-flip:{int(seed)}")
    extreme = sum(
        is_extreme(
            1 if generator.getrandbits(1) else -1 for _ in range(len(nonzero))
        )
        for _ in range(monte_carlo_samples)
    )
    return {
        "status": "ESTABLISHED",
        "method": "monte_carlo_cluster_sign_flip",
        "alternative": "two_sided",
        "two_sided_p": (extreme + 1) / (monte_carlo_samples + 1),
        "randomizations": monte_carlo_samples,
        "randomization_seed": int(seed),
        "nonzero_cluster_count": len(nonzero),
    }


def compare_paired_cluster_metric(
    rows: Iterable[Mapping[str, object]],
    *,
    cluster_keys: Sequence[str],
    arm_key: str,
    treatment_arm: str,
    control_arm: str,
    value_key: str,
    replicate_keys: Sequence[str] = (),
    bootstrap_samples: int = 20_000,
    randomization_seed: int = 21_021,
    exact_sign_flip_limit: int = 20,
    sign_flip_samples: int = 65_536,
    tie_tolerance: float = 0.0,
) -> dict[str, object]:
    """Compare run rows after aggregating replicates within case clusters.

    The function accepts ordinary experiment rows (including several seeds per
    case).  It first averages the metric within ``cluster x arm`` and only then
    performs paired inference across clusters.  Missing treatment/control
    clusters fail closed; they are never silently dropped.
    """

    if not cluster_keys:
        raise ValueError("cluster_keys must identify the independent case unit.")
    if treatment_arm == control_arm:
        raise ValueError("Treatment and control arms must differ.")
    if tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be nonnegative.")
    selected: list[
        tuple[tuple[object, ...], str, tuple[object, ...], float]
    ] = []
    for index, row in enumerate(rows):
        try:
            arm = str(row[arm_key])
            cluster = tuple(row[key] for key in cluster_keys)
            replicate = tuple(row[key] for key in replicate_keys)
        except KeyError as error:
            raise ValueError(f"Comparison row {index} omits required key {error}.") from error
        if arm not in {treatment_arm, control_arm}:
            continue
        try:
            value = float(row[value_key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Comparison row {index} has no finite numeric {value_key}."
            ) from error
        if not math.isfinite(value):
            raise ValueError(f"Comparison row {index} has non-finite {value_key}.")
        selected.append((cluster, arm, replicate, value))
    if not selected:
        raise ValueError("No rows match the requested treatment and control arms.")

    grouped: dict[tuple[object, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    replicates: dict[
        tuple[object, ...], dict[str, list[tuple[object, ...]]]
    ] = defaultdict(lambda: defaultdict(list))
    for cluster, arm, replicate, value in selected:
        grouped[cluster][arm].append(value)
        replicates[cluster][arm].append(replicate)
    incomplete = [
        cluster
        for cluster, arms in grouped.items()
        if treatment_arm not in arms or control_arm not in arms
    ]
    if incomplete:
        rendered = [
            json.dumps(list(cluster), ensure_ascii=False, separators=(",", ":"))
            for cluster in incomplete
        ]
        raise ValueError(
            "Unmatched treatment/control case clusters: " + ", ".join(sorted(rendered))
        )
    for cluster in grouped:
        treatment_replicates = replicates[cluster][treatment_arm]
        control_replicates = replicates[cluster][control_arm]
        if replicate_keys:
            if len(set(treatment_replicates)) != len(treatment_replicates):
                raise ValueError(
                    "Duplicate treatment replicate identity within a case cluster."
                )
            if len(set(control_replicates)) != len(control_replicates):
                raise ValueError(
                    "Duplicate control replicate identity within a case cluster."
                )
            if set(treatment_replicates) != set(control_replicates):
                raise ValueError(
                    "Treatment/control replicate identities are not matched within "
                    "a case cluster."
                )
        elif len(treatment_replicates) != len(control_replicates):
            raise ValueError(
                "Treatment/control replicate counts differ; provide matched rows or "
                "replicate_keys for an identity audit."
            )

    ordered_clusters = sorted(
        grouped,
        key=lambda cluster: json.dumps(
            list(cluster), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ),
    )
    paired: list[dict[str, object]] = []
    differences: list[float] = []
    for cluster in ordered_clusters:
        treatment_values = grouped[cluster][treatment_arm]
        control_values = grouped[cluster][control_arm]
        treatment_mean = math.fsum(treatment_values) / len(treatment_values)
        control_mean = math.fsum(control_values) / len(control_values)
        difference = treatment_mean - control_mean
        differences.append(difference)
        paired.append(
            {
                "cluster": {
                    key: value for key, value in zip(cluster_keys, cluster)
                },
                "treatment_replicates": len(treatment_values),
                "control_replicates": len(control_values),
                "treatment_mean": treatment_mean,
                "control_mean": control_mean,
                "difference": difference,
            }
        )

    ordered_differences = sorted(differences)
    trim = int(math.floor(0.10 * len(differences)))
    trimmed = (
        ordered_differences[trim : len(differences) - trim]
        if trim
        else ordered_differences
    )
    wins = sum(value > tie_tolerance for value in differences)
    losses = sum(value < -tie_tolerance for value in differences)
    ties = len(differences) - wins - losses
    return {
        "schema": "v21_paired_cluster_comparison_v1",
        "status": "PASS",
        "metric": value_key,
        "treatment_arm": treatment_arm,
        "control_arm": control_arm,
        "effect_direction": "treatment_minus_control",
        "inference_unit": "case_cluster",
        "cluster_keys": list(cluster_keys),
        "replicate_keys": list(replicate_keys),
        "cluster_count": len(differences),
        "input_observation_count": len(selected),
        "replicate_aggregation": "arithmetic_mean_within_cluster_and_arm",
        "mean_difference": math.fsum(differences) / len(differences),
        "median_difference": float(statistics.median(differences)),
        "trimmed_mean_difference": math.fsum(trimmed) / len(trimmed),
        "trim_fraction_each_tail": 0.10,
        "wins_ties_losses": {"wins": wins, "ties": ties, "losses": losses},
        "tie_tolerance": tie_tolerance,
        "cluster_bootstrap_ci95": _cluster_bootstrap_ci(
            differences,
            samples=bootstrap_samples,
            seed=randomization_seed,
        ),
        "sign_flip_test": _sign_flip_test(
            differences,
            seed=randomization_seed,
            exact_limit=exact_sign_flip_limit,
            monte_carlo_samples=sign_flip_samples,
        ),
        "paired_clusters": paired,
        "evidence_boundary": {
            "seeds_are_inferential_units": False,
            "evaluation_rows_are_inferential_units": False,
            "generality_beyond_observed_case_clusters": "NOT_ESTABLISHED",
        },
    }


__all__ = [
    "analyze_v21_trace",
    "compare_paired_cluster_metric",
    "write_v21_diagnostics_receipt",
]
