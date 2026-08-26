from __future__ import annotations

"""Fail-closed, complexity-first candidate selection for V21e3.

Selection is a calibration decision, not formal evidence.  The candidate chain
is traversed in order.  A more complex candidate is reachable only after every
simpler adjacent mechanism has cleared a prospectively declared practical
effect and a strictly positive case-cluster bootstrap lower bound.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .pareto_v21_diagnostics import compare_paired_cluster_metric


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_rows(path: Path) -> tuple[list[dict[str, object]], bytes]:
    raw = path.read_bytes()
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Calibration row {line_number} is not valid JSON."
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"Calibration row {line_number} is not an object.")
        rows.append(value)
    return rows, raw


def _validate_chain(candidate_chain: Sequence[str]) -> tuple[str, ...]:
    chain = tuple(str(value) for value in candidate_chain)
    if len(chain) < 2 or chain[0] != "C0" or len(set(chain)) != len(chain):
        raise ValueError("candidate_chain must be a unique C0-first chain.")
    for rank, candidate in enumerate(chain):
        if candidate != f"C{rank}":
            raise ValueError("candidate_chain must be contiguous C0,C1,... order.")
    return chain


def _strict_effect_gate(
    comparison: Mapping[str, object],
    *,
    practical_delta: float,
) -> tuple[str, dict[str, bool]]:
    ci = comparison.get("cluster_bootstrap_ci95")
    wtl = comparison.get("wins_ties_losses")
    if not isinstance(ci, Mapping) or not isinstance(wtl, Mapping):
        raise ValueError("A paired comparison omits CI or W/T/L evidence.")
    checks = {
        "mean_at_least_practical_delta": (
            float(comparison["mean_difference"]) + 1e-15
            >= float(practical_delta)
        ),
        "median_strictly_positive": float(comparison["median_difference"]) > 0.0,
        "trimmed_mean_strictly_positive": (
            float(comparison["trimmed_mean_difference"]) > 0.0
        ),
        "wins_exceed_losses": int(wtl["wins"]) > int(wtl["losses"]),
        "bootstrap_ci95_lower_strictly_positive": float(ci["lower"]) > 0.0,
    }
    return ("PASS" if all(checks.values()) else "FAIL"), checks


def _comparison_by_family(
    rows: Sequence[Mapping[str, object]],
    *,
    treatment: str,
    control: str,
    families: Sequence[str],
    bootstrap_samples: int,
    randomization_seed: int,
    tie_tolerance: float,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for family_index, family in enumerate(families):
        family_rows = [row for row in rows if str(row.get("family")) == family]
        results[family] = compare_paired_cluster_metric(
            family_rows,
            cluster_keys=("case_id",),
            arm_key="candidate_id",
            treatment_arm=treatment,
            control_arm=control,
            value_key="normalized_hv_auc",
            replicate_keys=("seed",),
            bootstrap_samples=bootstrap_samples,
            randomization_seed=randomization_seed + 1009 * family_index,
            tie_tolerance=tie_tolerance,
        )
    return results


def select_complexity_first_candidate(
    *,
    rows_path: str | Path,
    matrix_receipt_path: str | Path,
    output_path: str | Path,
    candidate_chain: Sequence[str],
    expected_seeds: Sequence[int],
    expected_cases_per_family: int,
    primary_delta_min: float,
    adjacent_delta_min: float,
    bootstrap_samples: int,
    randomization_seed: int,
    families: Sequence[str] = ("MOTSP", "MOKP"),
    tie_tolerance: float = 0.0,
) -> dict[str, object]:
    """Select the simplest supported V21e3 candidate.

    The returned receipt can authorize an independent calibration confirmation
    stage only.  It never authorizes formal-case materialization.
    """

    chain = _validate_chain(candidate_chain)
    frozen_families = tuple(str(value) for value in families)
    frozen_seeds = tuple(int(value) for value in expected_seeds)
    if (
        len(set(frozen_families)) != len(frozen_families)
        or not frozen_families
        or len(set(frozen_seeds)) != len(frozen_seeds)
        or not frozen_seeds
    ):
        raise ValueError("Families and seeds must be nonempty and unique.")
    if expected_cases_per_family <= 0:
        raise ValueError("expected_cases_per_family must be positive.")
    if (
        not math.isfinite(primary_delta_min)
        or primary_delta_min <= 0.0
        or not math.isfinite(adjacent_delta_min)
        or adjacent_delta_min <= 0.0
    ):
        raise ValueError("Practical effect thresholds must be finite and positive.")
    if bootstrap_samples <= 0 or tie_tolerance < 0.0:
        raise ValueError("Invalid bootstrap or tie policy.")

    rows_file = Path(rows_path).resolve()
    matrix_file = Path(matrix_receipt_path).resolve()
    rows, rows_raw = _load_rows(rows_file)
    matrix = json.loads(matrix_file.read_text(encoding="utf-8"))
    if not isinstance(matrix, dict):
        raise ValueError("The matrix receipt is not an object.")
    required_matrix = {
        "schema": "pareto_v21e3_calibration_matrix_receipt_v1",
        "status": "PASS",
        "candidate_ids": list(chain),
        "seeds": list(frozen_seeds),
        "expected_rows": len(rows),
        "completed_rows": len(rows),
        "all_trace_verifications_pass": True,
        "attempt_history_gate": "PASS",
        "objective_contract_gate": "PASS",
        "charged_budget_gate": "PASS",
        "artifact_root_gate": "PASS",
        "rows_sha256": _sha256(rows_raw),
    }
    mismatches = [
        key for key, expected in required_matrix.items() if matrix.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "V21e3 matrix receipt fails closed: " + ",".join(sorted(mismatches))
        )

    expected_run_keys: set[tuple[str, str, str, int]] = set()
    cases_by_family: dict[str, set[str]] = {
        family: set() for family in frozen_families
    }
    observed_run_keys: set[tuple[str, str, str, int]] = set()
    for index, row in enumerate(rows):
        family = str(row.get("family"))
        case_id = str(row.get("case_id"))
        candidate = str(row.get("candidate_id"))
        try:
            seed = int(row["seed"])
            value = float(row["normalized_hv_auc"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Run row {index} is malformed.") from error
        if (
            row.get("schema") != "pareto_v21e3_calibration_run_row_v1"
            or family not in cases_by_family
            or candidate not in chain
            or seed not in frozen_seeds
            or not math.isfinite(value)
            or row.get("trace_verification_status") != "PASS"
            or row.get("attempt_history_gate") != "PASS"
            or row.get("objective_contract_gate") != "PASS"
            or row.get("charged_budget_gate") != "PASS"
        ):
            raise ValueError(f"Run row {index} fails a V21e3 evidence gate.")
        key = (family, case_id, candidate, seed)
        if key in observed_run_keys:
            raise ValueError("Duplicate V21e3 run key.")
        observed_run_keys.add(key)
        cases_by_family[family].add(case_id)
    if any(
        len(case_ids) != expected_cases_per_family
        for case_ids in cases_by_family.values()
    ):
        raise ValueError("The matrix has the wrong case count for a family.")
    for family, case_ids in cases_by_family.items():
        for case_id in case_ids:
            for candidate in chain:
                for seed in frozen_seeds:
                    expected_run_keys.add((family, case_id, candidate, seed))
    if observed_run_keys != expected_run_keys:
        raise ValueError("The V21e3 matrix is not an exact matched cross product.")

    candidate_results: dict[str, dict[str, object]] = {}
    selected: str | None = None
    chain_open = True
    for candidate_index, candidate in enumerate(chain[1:], start=1):
        predecessor = chain[candidate_index - 1]
        if not chain_open:
            candidate_results[candidate] = {
                "gate": "NOT_REACHED",
                "reason": "A simpler adjacent mechanism failed the frozen chain.",
                "predecessor": predecessor,
            }
            continue
        primary = _comparison_by_family(
            rows,
            treatment=candidate,
            control=chain[0],
            families=frozen_families,
            bootstrap_samples=bootstrap_samples,
            randomization_seed=randomization_seed + 10_000 * candidate_index,
            tie_tolerance=tie_tolerance,
        )
        adjacent = (
            primary
            if predecessor == chain[0]
            else _comparison_by_family(
                rows,
                treatment=candidate,
                control=predecessor,
                families=frozen_families,
                bootstrap_samples=bootstrap_samples,
                randomization_seed=randomization_seed + 20_000 * candidate_index,
                tie_tolerance=tie_tolerance,
            )
        )
        primary_checks: dict[str, dict[str, bool]] = {}
        adjacent_checks: dict[str, dict[str, bool]] = {}
        primary_gates: dict[str, str] = {}
        adjacent_gates: dict[str, str] = {}
        for family in frozen_families:
            primary_gates[family], primary_checks[family] = _strict_effect_gate(
                primary[family], practical_delta=primary_delta_min
            )
            adjacent_gates[family], adjacent_checks[family] = _strict_effect_gate(
                adjacent[family], practical_delta=adjacent_delta_min
            )
        primary_gate = (
            "PASS" if all(value == "PASS" for value in primary_gates.values()) else "FAIL"
        )
        adjacent_gate = (
            "PASS" if all(value == "PASS" for value in adjacent_gates.values()) else "FAIL"
        )
        gate = "PASS" if primary_gate == adjacent_gate == "PASS" else "FAIL"
        candidate_results[candidate] = {
            "gate": gate,
            "predecessor": predecessor,
            "primary_gate": primary_gate,
            "adjacent_gate": adjacent_gate,
            "primary_by_family": primary,
            "adjacent_by_family": adjacent,
            "primary_checks_by_family": primary_checks,
            "adjacent_checks_by_family": adjacent_checks,
        }
        if gate == "PASS":
            selected = candidate
        else:
            chain_open = False

    status = "PASS" if selected is not None else "STOP"
    receipt: dict[str, object] = {
        "schema": "pareto_v21e3_complexity_first_selection_receipt_v1",
        "status": status,
        "selection_rule": "STRICT_COMPLEXITY_FIRST_CHAIN_V1",
        "candidate_chain": list(chain),
        "candidate_selected": selected,
        "confirmation_authorized": selected is not None,
        "formal_materialization_authorized": False,
        "formal_materialization_status": "NOT_MATERIALIZED",
        "primary_delta_min": primary_delta_min,
        "adjacent_delta_min": adjacent_delta_min,
        "bootstrap_samples": bootstrap_samples,
        "randomization_seed": randomization_seed,
        "tie_tolerance": tie_tolerance,
        "families": list(frozen_families),
        "expected_cases_per_family": expected_cases_per_family,
        "expected_seeds": list(frozen_seeds),
        "matrix_receipt_sha256": _sha256(matrix_file.read_bytes()),
        "rows_sha256": _sha256(rows_raw),
        "candidate_results": candidate_results,
        "action": (
            "AUTHORIZE_INDEPENDENT_CALIBRATION_CONFIRMATION_ONLY"
            if selected is not None
            else "STOP_BEFORE_CONFIRMATION_AND_FORMAL_MATERIALIZATION"
        ),
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(_canonical_bytes(receipt))
    return receipt


__all__ = ["select_complexity_first_candidate"]
