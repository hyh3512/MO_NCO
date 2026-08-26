from __future__ import annotations

"""Prospective V21 calibration runner and frozen normalized HV metric.

The runner is deliberately exclusive-create.  A partially produced output
directory is evidence of an interrupted calibration attempt and is never
silently resumed or overwritten by this module.
"""

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Iterable, Sequence

from .instance import MultiObjectiveTSPInstance
from .pareto_ijoc_problem import (
    MultiObjectiveCombinatorialProblem,
    MultiObjectiveKnapsackInstance,
    MultiObjectiveTSPProblemAdapter,
)
from .pareto_v21_hybrid import V21HybridConfig, V21TypedHybridParetoSearch
from .pareto_v21_partitions import load_partition_case
from .pareto_v21_trace_verify import verify_v21_trace_database
from .pareto_v21_diagnostics import compare_paired_cluster_metric
from .types import ObjectiveVector


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _resolve_context_binding(context_file: Path, raw_path: object) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path.resolve()
    candidates = (Path.cwd(), *context_file.parents)
    for root in candidates:
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def _verify_context_binding(context_file: Path, binding: object) -> Path:
    if not isinstance(binding, dict):
        raise ValueError("A run-context artifact binding is malformed.")
    path = _resolve_context_binding(context_file, binding.get("path"))
    payload = path.read_bytes()
    if _sha256_bytes(payload) != binding.get("sha256"):
        raise ValueError(f"Run-context SHA-256 mismatch: {path}")
    if "bytes" in binding and len(payload) != int(binding["bytes"]):
        raise ValueError(f"Run-context byte-count mismatch: {path}")
    return path


def _write_canonical_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload))


def load_v21_problem_packet(
    path: str | Path,
) -> MultiObjectiveCombinatorialProblem:
    packet_path = Path(path).resolve()
    payload = json.loads(packet_path.read_text(encoding="utf-8"))
    return _problem_from_payload(payload)


def _problem_from_payload(
    payload: dict[str, object],
) -> MultiObjectiveCombinatorialProblem:
    schema = payload.get("schema")
    case_id = str(payload["case_id"])
    if schema == "pareto_v21_mokp_integer_instance_v1":
        return MultiObjectiveKnapsackInstance(
            item_weights=tuple(int(value) for value in payload["item_weights"]),
            profits_by_objective=tuple(
                tuple(int(value) for value in profits)
                for profits in payload["profits_by_objective"]
            ),
            capacity=int(payload["capacity"]),
            name=case_id,
        )
    if schema == "pareto_v21_motsp_integer_coordinates_v1":
        coordinates = tuple(
            tuple(
                (float(point[0]), float(point[1]))
                for point in objective_coordinates
            )
            for objective_coordinates in payload["coordinates_by_objective"]
        )
        return MultiObjectiveTSPProblemAdapter(
            MultiObjectiveTSPInstance(
                coords_by_objective=coordinates,
                name=case_id,
            )
        )
    raise ValueError(f"Unsupported V21 problem-packet schema: {schema!r}")


def normalized_hypervolume_2d(
    front: Iterable[ObjectiveVector],
    *,
    lower: Sequence[float],
    upper: Sequence[float],
) -> float:
    """Exact 2-D minimization HV after analytic-box normalization.

    The fixed normalized reference is ``(1, 1)``.  Values outside the bound
    contract are rejected instead of clipped because clipping would conceal a
    metric-reference failure.
    """

    if len(lower) != 2 or len(upper) != 2:
        raise ValueError("V21 normalized HV currently requires two objectives.")
    spans = tuple(float(hi) - float(lo) for lo, hi in zip(lower, upper))
    if any(span <= 0.0 for span in spans):
        raise ValueError("Every analytic objective span must be positive.")
    points: list[tuple[float, float]] = []
    for objective in front:
        if len(objective) != 2:
            raise ValueError("Every V21 HV point must be two-dimensional.")
        point = tuple(
            (float(value) - float(lo)) / span
            for value, lo, span in zip(objective, lower, spans)
        )
        if any(value < -1e-12 or value > 1.0 + 1e-12 for value in point):
            raise ValueError("An objective lies outside the frozen analytic box.")
        points.append((float(point[0]), float(point[1])))
    nondominated: list[tuple[float, float]] = []
    for point in sorted(set(points)):
        if any(
            other[0] <= point[0]
            and other[1] <= point[1]
            and other != point
            for other in points
        ):
            continue
        nondominated.append(point)
    hypervolume = 0.0
    best_y = 1.0
    for x_value, y_value in sorted(nondominated):
        if y_value < best_y:
            hypervolume += (1.0 - x_value) * (best_y - y_value)
            best_y = y_value
    if not -1e-12 <= hypervolume <= 1.0 + 1e-12:
        raise RuntimeError("Normalized HV escaped [0, 1].")
    return min(1.0, max(0.0, hypervolume))


def _normalized_anytime_metrics(
    diagnostics: Sequence[object],
    *,
    budget: int,
    lower: Sequence[float],
    upper: Sequence[float],
) -> tuple[float, float, tuple[dict[str, object], ...]]:
    previous_evaluation = 0
    previous_hv = 0.0
    area = 0.0
    checkpoints: list[dict[str, object]] = []
    for diagnostic in diagnostics:
        evaluation = int(getattr(diagnostic, "iteration"))
        if evaluation <= previous_evaluation or evaluation > budget:
            raise RuntimeError("V21 diagnostics do not form a strict budget grid.")
        area += previous_hv * (evaluation - previous_evaluation)
        current_hv = normalized_hypervolume_2d(
            getattr(diagnostic, "front"),
            lower=lower,
            upper=upper,
        )
        checkpoints.append(
            {
                "evaluation": evaluation,
                "normalized_hv": current_hv,
                "archive_size": int(getattr(diagnostic, "archive_size")),
            }
        )
        previous_evaluation = evaluation
        previous_hv = current_hv
    if previous_evaluation != budget:
        raise RuntimeError("V21 diagnostics omit the terminal budget checkpoint.")
    return area / float(budget), previous_hv, tuple(checkpoints)


def run_v21_calibration_matrix(
    *,
    manifest_path: str | Path,
    run_context_path: str | Path,
    output_directory: str | Path,
    candidate_ids: Sequence[str],
    seeds: Sequence[int],
    evaluation_budget: int,
    checkpoint_period: int,
    reference_directions: Sequence[Sequence[float]],
    require_full_partition_binding: bool = True,
) -> dict[str, object]:
    manifest_file = Path(manifest_path).resolve()
    manifest_bytes = manifest_file.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "pareto_v21_partition_manifest_v1":
        raise ValueError("The V21 calibration manifest schema is not frozen.")
    split = str(manifest.get("split"))
    if split not in {"selection", "confirmation"}:
        raise ValueError("Only V21 calibration partitions may enter this runner.")
    cases = tuple(manifest.get("cases", ()))
    candidates = tuple(str(value) for value in candidate_ids)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Candidate IDs must be nonempty and unique.")
    frozen_seeds = tuple(int(value) for value in seeds)
    if not frozen_seeds or len(set(frozen_seeds)) != len(frozen_seeds):
        raise ValueError("Calibration seeds must be nonempty and unique.")
    directions = tuple(
        tuple(float(value) for value in direction)
        for direction in reference_directions
    )
    context_file = Path(run_context_path).resolve()
    context_bytes = context_file.read_bytes()
    context = json.loads(context_bytes)
    if context.get("schema") != "pareto_v21_calibration_run_context_v1":
        raise ValueError("The V21 run-context schema is not frozen.")
    if context.get("status") != "FROZEN_BEFORE_RUNS":
        raise ValueError("The V21 run context was not frozen before execution.")
    if not (
        context.get("split") == split
        and tuple(context.get("candidate_ids", ())) == candidates
        and tuple(int(value) for value in context.get("seeds", ())) == frozen_seeds
        and int(context.get("evaluation_budget", -1)) == int(evaluation_budget)
        and int(context.get("checkpoint_period", -1)) == int(checkpoint_period)
        and tuple(
            tuple(float(value) for value in direction)
            for direction in context.get("reference_directions", ())
        )
        == directions
    ):
        raise ValueError("The invocation differs from the frozen V21 run context.")
    bound_manifest = _verify_context_binding(
        context_file, context.get("partition_manifest")
    )
    if bound_manifest != manifest_file:
        raise ValueError("The run context binds another partition manifest.")
    metric_file = _verify_context_binding(
        context_file, context.get("metric_manifest")
    )
    metric_manifest = json.loads(metric_file.read_bytes())
    if not (
        metric_manifest.get("schema") == "pareto_v21_metric_manifest_v1"
        and metric_manifest.get("metric_id")
        == "normalized_left_continuous_hypervolume_auc_analytic_box_reference_1_1_v1"
    ):
        raise ValueError("The run context binds an unexpected V21 metric.")
    reference_file = _verify_context_binding(
        context_file, context.get("reference_manifest")
    )
    reference_manifest = json.loads(reference_file.read_bytes())
    reference_cases = tuple(reference_manifest.get("cases", ()))
    if not (
        reference_manifest.get("schema")
        == "pareto_v21_analytic_reference_manifest_v1"
        and reference_manifest.get("split") == split
        and int(reference_manifest.get("case_count", -1)) == len(cases)
        and len(reference_cases) == len(cases)
    ):
        raise ValueError("The analytic reference manifest fails its structural gate.")
    reference_by_case = {
        str(case["case_id"]): case for case in reference_cases
    }
    if len(reference_by_case) != len(reference_cases) or set(reference_by_case) != {
        str(case["case_id"]) for case in cases
    }:
        raise ValueError("The analytic reference manifest binds another case set.")
    _verify_context_binding(context_file, context.get("precommit"))
    source_bindings = tuple(context.get("source_bindings", ()))
    if not source_bindings:
        raise ValueError("The V21 run context omits source bindings.")
    for binding in source_bindings:
        _verify_context_binding(context_file, binding)
    context_sha256 = _sha256_bytes(context_bytes)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    trace_root = output / "traces"
    trace_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    expected_rows = len(cases) * len(candidates) * len(frozen_seeds)
    started = time.perf_counter()
    full_binding_count = 0
    for case in cases:
        if case.get("split") != split:
            raise ValueError("A case carries a split inconsistent with its manifest.")
        artifact = case["artifact"]
        packet_path = (manifest_file.parent / artifact["path"]).resolve()
        packet_bytes = packet_path.read_bytes()
        if _sha256_bytes(packet_bytes) != artifact["sha256"]:
            raise ValueError(f"Case artifact hash mismatch: {case['case_id']}")
        if len(packet_bytes) != int(artifact["bytes"]):
            raise ValueError(f"Case artifact size mismatch: {case['case_id']}")
        if require_full_partition_binding:
            bound_payload = load_partition_case(
                manifest_file,
                str(case["case_id"]),
            )
            full_binding_count += 1
        else:
            bound_payload = json.loads(packet_bytes)
        for candidate in candidates:
            for seed in frozen_seeds:
                problem = _problem_from_payload(bound_payload)
                reference_case = reference_by_case[str(case["case_id"])]
                if not (
                    reference_case.get("family") == case.get("family")
                    and reference_case.get("packet_sha256") == artifact.get("sha256")
                    and tuple(
                        float(value)
                        for value in reference_case.get("objective_lower_bounds", ())
                    )
                    == tuple(float(value) for value in problem.objective_lower_bounds)
                    and tuple(
                        float(value)
                        for value in reference_case.get("objective_upper_bounds", ())
                    )
                    == tuple(float(value) for value in problem.objective_upper_bounds)
                    and tuple(
                        float(value)
                        for value in reference_case.get("normalized_reference_point", ())
                    )
                    == (1.0, 1.0)
                ):
                    raise ValueError(
                        f"Analytic reference mismatch: {case['case_id']}"
                    )
                trace_path = (
                    trace_root
                    / candidate.lower()
                    / str(case["case_id"])
                    / f"seed-{seed}.sqlite3"
                )
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                run_started = time.perf_counter()
                run = V21TypedHybridParetoSearch(
                    problem,
                    V21HybridConfig(
                        candidate_id=candidate,
                        reference_directions=directions,
                        evaluations=int(evaluation_budget),
                        checkpoint_period=int(checkpoint_period),
                        seed=seed,
                        phase="calibration",
                        trace_database=str(trace_path),
                        capture_trace=False,
                    ),
                ).run()
                verification = verify_v21_trace_database(
                    trace_path,
                    problem,
                    expected_budget=int(evaluation_budget),
                    expected_archive=run.optimization_result.archive,
                )
                auc, final_hv, checkpoints = _normalized_anytime_metrics(
                    run.optimization_result.diagnostics,
                    budget=int(evaluation_budget),
                    lower=problem.objective_lower_bounds,
                    upper=problem.objective_upper_bounds,
                )
                archive_payload = tuple(
                    {
                        "solution": entry.tour,
                        "objectives": entry.objectives,
                    }
                    for entry in run.optimization_result.archive.entries
                )
                rows.append(
                    {
                        "schema": "pareto_v21_calibration_run_row_v1",
                        "split": split,
                        "case_id": str(case["case_id"]),
                        "family": str(case["family"]),
                        "size": int(case["size"]),
                        "regime": case.get("regime", "NOT_BOUND"),
                        "candidate_id": candidate,
                        "run_context_sha256": context_sha256,
                        "seed": seed,
                        "evaluation_budget": int(evaluation_budget),
                        "checkpoint_period": int(checkpoint_period),
                        "normalized_hv_auc": auc,
                        "normalized_final_hv": final_hv,
                        "checkpoints": checkpoints,
                        "archive_sha256": _sha256_bytes(
                            _canonical_bytes(archive_payload)
                        ),
                        "archive_size": len(run.optimization_result.archive),
                        "trace_verification_status": verification["status"],
                        "trace_database_sha256": verification["database_sha256"],
                        "terminal_evaluation_chain_sha256": verification[
                            "terminal_evaluation_chain_sha256"
                        ],
                        "terminal_decision_chain_sha256": verification[
                            "terminal_decision_chain_sha256"
                        ],
                        "terminal_mechanism_chain_sha256": verification[
                            "terminal_mechanism_chain_sha256"
                        ],
                        "trace_relative_path": trace_path.relative_to(output).as_posix(),
                        "wall_seconds": time.perf_counter() - run_started,
                    }
                )
    if len(rows) != expected_rows:
        raise RuntimeError("The V21 calibration matrix is incomplete.")
    rows_path = output / "run_rows.jsonl"
    rows_path.write_bytes(
        b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    )
    receipt = {
        "schema": "pareto_v21_calibration_matrix_receipt_v1",
        "status": "PASS",
        "evidence_scope": "engineering_and_calibration_only_not_formal",
        "split": split,
        "manifest_path": str(manifest_file),
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "run_context_path": str(context_file),
        "run_context_sha256": context_sha256,
        "run_context_binding_gate": "PASS",
        "candidate_ids": candidates,
        "seeds": frozen_seeds,
        "evaluation_budget": int(evaluation_budget),
        "checkpoint_period": int(checkpoint_period),
        "reference_directions": directions,
        "expected_rows": expected_rows,
        "completed_rows": len(rows),
        "all_trace_verifications_pass": all(
            row["trace_verification_status"] == "PASS" for row in rows
        ),
        "full_partition_binding_gate": (
            "PASS"
            if full_binding_count == len(cases)
            else "NOT_ESTABLISHED_TEST_ONLY"
        ),
        "rows_sha256": _sha256_bytes(rows_path.read_bytes()),
        "wall_seconds": time.perf_counter() - started,
    }
    _write_canonical_json(output / "matrix_receipt.json", receipt)
    return receipt


def select_v21_calibration_candidate(
    *,
    rows_path: str | Path,
    matrix_receipt_path: str | Path,
    output_path: str | Path,
    candidate_ids: Sequence[str],
    control_id: str,
    delta_min: float,
    noninferiority_margin: float,
    bootstrap_samples: int,
    randomization_seed: int,
    expected_seeds: Sequence[int],
    expected_cases_per_family: int | None = None,
    mechanism_noninferiority_margin: float = 0.001,
) -> dict[str, object]:
    """Apply the predeclared two-family selection gate against C0.

    Seeds are averaged inside each case; every robustness and uncertainty
    calculation is performed on matched case clusters.
    """

    if (
        delta_min <= 0.0
        or noninferiority_margin <= 0.0
        or mechanism_noninferiority_margin <= 0.0
    ):
        raise ValueError("Calibration effect and noninferiority margins must be positive.")
    candidates = tuple(str(value) for value in candidate_ids)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Selection candidates must be nonempty and unique.")
    if control_id in candidates:
        raise ValueError("The control must not appear in the treatment candidate list.")
    source = Path(rows_path).resolve()
    source_bytes = source.read_bytes()
    rows = [
        json.loads(line)
        for line in source_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    matrix_receipt_file = Path(matrix_receipt_path).resolve()
    matrix_receipt_bytes = matrix_receipt_file.read_bytes()
    matrix_receipt = json.loads(matrix_receipt_bytes)
    expected_arms = (str(control_id),) + candidates
    frozen_expected_seeds = tuple(int(value) for value in expected_seeds)
    if (
        not frozen_expected_seeds
        or len(set(frozen_expected_seeds)) != len(frozen_expected_seeds)
    ):
        raise ValueError("Expected selection seeds must be nonempty and unique.")
    if matrix_receipt.get("schema") != "pareto_v21_calibration_matrix_receipt_v1":
        raise ValueError("The matrix receipt schema is not frozen.")
    if _sha256_bytes(source_bytes) != matrix_receipt.get("rows_sha256"):
        raise ValueError("The matrix receipt does not bind the supplied rows.")
    if not (
        matrix_receipt.get("status") == "PASS"
        and matrix_receipt.get("split") == "selection"
        and tuple(matrix_receipt.get("candidate_ids", ())) == expected_arms
        and tuple(int(value) for value in matrix_receipt.get("seeds", ()))
        == frozen_expected_seeds
        and int(matrix_receipt.get("expected_rows", -1)) == len(rows)
        and int(matrix_receipt.get("completed_rows", -1)) == len(rows)
        and matrix_receipt.get("all_trace_verifications_pass") is True
        and matrix_receipt.get("full_partition_binding_gate") == "PASS"
        and matrix_receipt.get("run_context_binding_gate") == "PASS"
    ):
        raise ValueError("The selection matrix receipt fails a completeness gate.")
    matrix_context_sha256 = str(matrix_receipt.get("run_context_sha256", ""))
    if len(matrix_context_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in matrix_context_sha256.lower()
    ):
        raise ValueError("The selection matrix has no valid run-context digest.")

    seen_run_keys: set[tuple[str, str, int]] = set()
    case_families: dict[str, str] = {}
    coverage: dict[tuple[str, str], dict[str, set[int]]] = {}
    for row in rows:
        if row.get("schema") != "pareto_v21_calibration_run_row_v1":
            raise ValueError("A selection row has an unfrozen schema.")
        if row.get("split") != "selection":
            raise ValueError("A non-selection row entered the selection gate.")
        if row.get("trace_verification_status") != "PASS":
            raise ValueError("A selection trace failed independent verification.")
        if str(row.get("run_context_sha256", "")).lower() != (
            matrix_context_sha256.lower()
        ):
            raise ValueError("A selection row is detached from the run context.")
        candidate_id = str(row.get("candidate_id"))
        if candidate_id not in expected_arms:
            raise ValueError("An undeclared arm entered the selection gate.")
        family = str(row.get("family"))
        case_id = str(row.get("case_id"))
        seed = int(row.get("seed"))
        value = float(row.get("normalized_hv_auc"))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("A selection metric is non-finite or outside [0,1].")
        if seed not in frozen_expected_seeds:
            raise ValueError("An undeclared seed entered the selection gate.")
        run_key = (case_id, candidate_id, seed)
        if run_key in seen_run_keys:
            raise ValueError("The selection matrix contains a duplicate run key.")
        seen_run_keys.add(run_key)
        prior_family = case_families.setdefault(case_id, family)
        if prior_family != family:
            raise ValueError("A case ID is assigned to multiple families.")
        coverage.setdefault((family, case_id), {}).setdefault(candidate_id, set()).add(
            seed
        )

    for (family, case_id), arm_seeds in coverage.items():
        if set(arm_seeds) != set(expected_arms):
            raise ValueError(f"{family}/{case_id} omits a declared selection arm.")
        if any(
            tuple(sorted(seeds_for_arm)) != tuple(sorted(frozen_expected_seeds))
            for seeds_for_arm in arm_seeds.values()
        ):
            raise ValueError(f"{family}/{case_id} has incomplete seed coverage.")
    families = tuple(sorted({str(row.get("family")) for row in rows}))
    if families != ("MOKP", "MOTSP"):
        raise ValueError("V21 selection requires exactly the MOKP and MOTSP families.")
    if expected_cases_per_family is not None:
        for family in families:
            family_case_count = len(
                {case_id for row_family, case_id in coverage if row_family == family}
            )
            if family_case_count != int(expected_cases_per_family):
                raise ValueError(
                    f"{family} has {family_case_count} cases, expected "
                    f"{expected_cases_per_family}."
                )

    results: dict[str, dict[str, object]] = {}
    eligible: list[tuple[str, float, float, float]] = []
    for candidate_index, candidate in enumerate(candidates):
        family_results: dict[str, object] = {}
        family_gates: dict[str, str] = {}
        family_means: list[float] = []
        family_trimmed: list[float] = []
        total_win_minus_loss = 0
        for family_index, family in enumerate(families):
            family_rows = [row for row in rows if row.get("family") == family]
            comparison = compare_paired_cluster_metric(
                family_rows,
                cluster_keys=("case_id",),
                arm_key="candidate_id",
                treatment_arm=candidate,
                control_arm=control_id,
                value_key="normalized_hv_auc",
                replicate_keys=("seed",),
                bootstrap_samples=int(bootstrap_samples),
                randomization_seed=(
                    int(randomization_seed)
                    + 1009 * candidate_index
                    + 9173 * family_index
                ),
            )
            if (
                expected_cases_per_family is not None
                and comparison["cluster_count"] != int(expected_cases_per_family)
            ):
                raise ValueError(
                    f"{candidate}/{family} has {comparison['cluster_count']} case "
                    f"clusters, expected {expected_cases_per_family}."
                )
            mean = float(comparison["mean_difference"])
            median = float(comparison["median_difference"])
            trimmed = float(comparison["trimmed_mean_difference"])
            counts = comparison["wins_ties_losses"]
            ci = comparison["cluster_bootstrap_ci95"]
            checks = {
                "practical_mean_at_least_delta_min": mean >= float(delta_min),
                "median_strictly_positive": median > 0.0,
                "trimmed_mean_strictly_positive": trimmed > 0.0,
                "wins_exceed_losses": int(counts["wins"]) > int(counts["losses"]),
                "cluster_ci_noninferior": (
                    ci.get("status") == "ESTABLISHED"
                    and float(ci["lower"]) > -float(noninferiority_margin)
                ),
            }
            family_gate = "PASS" if all(checks.values()) else "FAIL"
            comparison["selection_gate_checks"] = checks
            comparison["selection_gate"] = family_gate
            family_results[family] = comparison
            family_gates[family] = family_gate
            family_means.append(mean)
            family_trimmed.append(trimmed)
            total_win_minus_loss += int(counts["wins"]) - int(counts["losses"])

        try:
            candidate_rank = int(candidate.removeprefix("C"))
        except ValueError as exc:
            raise ValueError("Adjacent mechanism gates require Ck candidate IDs.") from exc
        predecessor = control_id if candidate_rank == 1 else f"C{candidate_rank - 1}"
        mechanism_results: dict[str, object] = {}
        mechanism_family_gates: dict[str, str] = {}
        for family_index, family in enumerate(families):
            family_rows = [row for row in rows if row.get("family") == family]
            mechanism = compare_paired_cluster_metric(
                family_rows,
                cluster_keys=("case_id",),
                arm_key="candidate_id",
                treatment_arm=candidate,
                control_arm=predecessor,
                value_key="normalized_hv_auc",
                replicate_keys=("seed",),
                bootstrap_samples=int(bootstrap_samples),
                randomization_seed=(
                    int(randomization_seed)
                    + 50_021
                    + 1009 * candidate_index
                    + 9173 * family_index
                ),
            )
            mechanism_counts = mechanism["wins_ties_losses"]
            mechanism_ci = mechanism["cluster_bootstrap_ci95"]
            mechanism_checks = {
                "mean_strictly_positive": float(mechanism["mean_difference"]) > 0.0,
                "median_strictly_positive": float(mechanism["median_difference"]) > 0.0,
                "trimmed_mean_strictly_positive": (
                    float(mechanism["trimmed_mean_difference"]) > 0.0
                ),
                "wins_exceed_losses": (
                    int(mechanism_counts["wins"])
                    > int(mechanism_counts["losses"])
                ),
                "cluster_ci_noninferior": (
                    mechanism_ci.get("status") == "ESTABLISHED"
                    and float(mechanism_ci["lower"])
                    > -float(mechanism_noninferiority_margin)
                ),
            }
            mechanism_gate = (
                "PASS" if all(mechanism_checks.values()) else "FAIL"
            )
            mechanism["mechanism_gate_checks"] = mechanism_checks
            mechanism["mechanism_gate"] = mechanism_gate
            mechanism_results[family] = mechanism
            mechanism_family_gates[family] = mechanism_gate
        adjacent_mechanism_gate = (
            "PASS"
            if all(value == "PASS" for value in mechanism_family_gates.values())
            else "FAIL"
        )
        candidate_gate = (
            "PASS"
            if all(value == "PASS" for value in family_gates.values())
            and adjacent_mechanism_gate == "PASS"
            else "FAIL"
        )
        minimum_family_mean = min(family_means)
        average_family_mean = sum(family_means) / len(family_means)
        results[candidate] = {
            "gate": candidate_gate,
            "family_gates": family_gates,
            "minimum_family_mean_difference": minimum_family_mean,
            "average_family_mean_difference": average_family_mean,
            "family_results": family_results,
            "adjacent_predecessor": predecessor,
            "adjacent_mechanism_gate": adjacent_mechanism_gate,
            "adjacent_mechanism_family_gates": mechanism_family_gates,
            "adjacent_mechanism_results": mechanism_results,
        }
        if candidate_gate == "PASS":
            eligible.append(
                (
                    candidate,
                    minimum_family_mean,
                    min(family_trimmed),
                    float(total_win_minus_loss),
                )
            )

    selected = (
        None
        if not eligible
        else sorted(
            eligible,
            key=lambda item: (-item[1], -item[2], -item[3], item[0]),
        )[0][0]
    )
    receipt: dict[str, object] = {
        "schema": "pareto_v21_calibration_selection_receipt_v1",
        "status": "PASS" if selected is not None else "STOP",
        "scientific_scope": "prospective_calibration_selection_not_formal_evidence",
        "rows_path": str(source),
        "rows_sha256": _sha256_bytes(source_bytes),
        "matrix_receipt_path": str(matrix_receipt_file),
        "matrix_receipt_sha256": _sha256_bytes(matrix_receipt_bytes),
        "matrix_binding_gate": "PASS",
        "control_id": str(control_id),
        "candidate_ids": candidates,
        "families": families,
        "expected_cases_per_family": expected_cases_per_family,
        "expected_seeds": frozen_expected_seeds,
        "inference_unit": "case_cluster",
        "replicate_aggregation": "seed_mean_within_case_and_arm",
        "delta_min": float(delta_min),
        "noninferiority_margin": float(noninferiority_margin),
        "mechanism_noninferiority_margin": float(
            mechanism_noninferiority_margin
        ),
        "bootstrap_samples": int(bootstrap_samples),
        "randomization_seed": int(randomization_seed),
        "selection_rule": (
            "primary_and_adjacent_mechanism_family_gates_then_maximum_minimum_"
            "family_mean_then_minimum_trimmed_mean_then_win_loss_then_simplicity"
        ),
        "candidate_results": results,
        "candidate_selected": selected,
        "confirmation_authorized": selected is not None,
        "formal_authorized": False,
        "formal_status": "NOT_MATERIALIZED",
    }
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    _write_canonical_json(destination, receipt)
    return receipt
