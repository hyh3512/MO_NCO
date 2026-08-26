#!/usr/bin/env python3
from __future__ import annotations

"""Standalone metric reimplementation for V21e3r1 traces.

This file intentionally imports only the Python standard library.  It is an
implementation-independent metric/checkpoint calculation, not a third-party
scientific reproduction of the algorithm.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Iterable, Sequence


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


def _exact_number(value: object, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be an exact finite JSON number")
    return float(value)


def _decode_objectives(raw: object) -> tuple[float, float]:
    if type(raw) is not str:
        raise RuntimeError("objective payload must be exact JSON text")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("objective payload is invalid JSON") from error
    if (
        not isinstance(decoded, list)
        or len(decoded) != 2
        or any(type(value) not in (int, float) for value in decoded)
        or _canonical_json(decoded) != raw
    ):
        raise RuntimeError("objective payload is not a canonical exact numeric two-vector")
    return (
        _exact_number(decoded[0], "objective[0]"),
        _exact_number(decoded[1], "objective[1]"),
    )


def _normalize(point: Sequence[float], lower: Sequence[float], upper: Sequence[float]) -> tuple[float, float]:
    if len(point) != 2 or len(lower) != 2 or len(upper) != 2:
        raise ValueError("biobjective inputs required")
    output = []
    for value, lo, hi in zip(point, lower, upper):
        value = _exact_number(value, "objective")
        lo = _exact_number(lo, "lower bound")
        hi = _exact_number(hi, "upper bound")
        if not math.isfinite(value) or not lo < hi or not lo <= value <= hi:
            raise ValueError("invalid objective or analytic box")
        output.append((value - lo) / (hi - lo))
    return (output[0], output[1])


def _dominates(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[0] <= right[0] and left[1] <= right[1] and left != right


def nondominated(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    result = []
    for point in unique:
        if any(_dominates(other, point) for other in unique):
            continue
        result.append(point)
    return result


def hypervolume_2d(points: Iterable[tuple[float, float]]) -> float:
    front = sorted(nondominated(points), key=lambda p: (p[0], p[1]))
    area = 0.0
    previous_y = 1.0
    for x, y in front:
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError("normalized point outside [0,1]^2")
        if y < previous_y:
            area += (1.0 - x) * (previous_y - y)
            previous_y = y
    return area


def recompute(
    trace: str | Path,
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    expected_evaluations: int | None = None,
) -> dict[str, object]:
    path = Path(trace).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_evaluations is not None and (
        type(expected_evaluations) is not int or expected_evaluations <= 0
    ):
        raise ValueError("expected_evaluations must be an exact positive integer")
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("SQLite integrity_check failed")
        run = connection.execute(
            "SELECT run_context_json,run_context_digest_sha256,status "
            "FROM run_attempt WHERE run_id=1"
        ).fetchone()
        if run is None or str(run[2]) != "SUCCESS":
            raise RuntimeError("trace is not a terminal successful run")
        context_raw = str(run[0])
        context = json.loads(context_raw)
        if not isinstance(context, dict) or _canonical_json(context) != context_raw:
            raise RuntimeError("run context is not canonical JSON")
        context_digest = hashlib.sha256(context_raw.encode("utf-8")).hexdigest()
        if context_digest != str(run[1]):
            raise RuntimeError("run-context SHA-256 binding failed")
        budget = context.get("charged_evaluation_budget")
        if type(budget) is not int or budget <= 0:
            raise RuntimeError("run context omits an exact positive evaluation budget")
        if expected_evaluations is not None and budget != expected_evaluations:
            raise RuntimeError("trace budget disagrees with expected_evaluations")
        rows = list(connection.execute(
            "SELECT evaluation_index,objectives_json FROM evaluations ORDER BY evaluation_index"
        ))
        decision_count = int(
            connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        )
        attempt_count = int(
            connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
        )
        terminal_row = connection.execute(
            "SELECT receipt_json FROM terminal_receipts WHERE run_id=1"
        ).fetchone()
        if terminal_row is None:
            raise RuntimeError("trace omits terminal receipt")
        terminal_raw = str(terminal_row[0])
        terminal = json.loads(terminal_raw)
        if not isinstance(terminal, dict) or _canonical_json(terminal) != terminal_raw:
            raise RuntimeError("terminal receipt is not canonical JSON")
    finally:
        connection.close()
    if (
        len(rows) != budget
        or decision_count != budget
        or terminal.get("status") != "SUCCESS"
        or terminal.get("charged_evaluation_count") != budget
        or terminal.get("decision_count") != budget
        or terminal.get("attempt_count") != attempt_count
    ):
        raise RuntimeError("trace terminal/accounting completeness gate failed")
    points: list[tuple[float, float]] = []
    hv_before: list[float] = []
    hv_after: list[float] = []
    current = 0.0
    for expected, (index, raw) in enumerate(rows, start=1):
        if int(index) != expected:
            raise RuntimeError("noncontiguous evaluation index")
        hv_before.append(current)
        points.append(_normalize(_decode_objectives(raw), lower, upper))
        current = hypervolume_2d(points)
        hv_after.append(current)
    return {
        "schema": "v21e3r1_independent_metric_reimplementation_v2",
        "status": "PASS_INDEPENDENT_METRIC_IMPLEMENTATION",
        "trace": str(path),
        "evaluation_count": len(rows),
        "attempt_count": attempt_count,
        "decision_count": decision_count,
        "exact_left_continuous_hv_auc": sum(hv_before) / len(hv_before),
        "terminal_hv": hv_after[-1],
        "trace_sha256": _sha256(path),
        "run_context_digest_sha256": context_digest,
        "reimplementation_source_sha256": _sha256(Path(__file__).resolve()),
        "terminal_accounting_gate": "PASS",
        "implementation_independence_from_project_metrics": True,
        "algorithm_execution_independence": False,
        "scientific_independence": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--lower", required=True, help="comma-separated two-vector")
    parser.add_argument("--upper", required=True, help="comma-separated two-vector")
    parser.add_argument("--output")
    parser.add_argument("--expected-evaluations", type=int)
    args = parser.parse_args()
    lower = tuple(float(x) for x in args.lower.split(","))
    upper = tuple(float(x) for x in args.upper.split(","))
    result = recompute(
        args.trace,
        lower,
        upper,
        expected_evaluations=args.expected_evaluations,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

