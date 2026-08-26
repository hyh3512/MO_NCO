from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .metrics import empirical_reference_front, ideal_nadir
from .types import ObjectiveVector


THEORY_ABLATION_ARMS = {
    "ips-theory-heavy-no-prior",
    "ips-theory-endpoint-only",
    "ips-theory-move-only",
    "ips-neural-mv-jitgreedy-targetflow-theory-optimized",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_archive(path: Path) -> Tuple[ObjectiveVector, ...]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    points: List[ObjectiveVector] = []
    for row in rows:
        objective_fields = sorted(
            (field for field in row if field.startswith("objective_")),
            key=lambda field: int(field.rsplit("_", 1)[1]),
        )
        if len(objective_fields) != 2:
            raise ValueError(f"Expected two objective columns in archive: {path}")
        points.append(tuple(float(row[field]) for field in objective_fields))
    if not points:
        raise ValueError(f"Calibration archive is empty: {path}")
    return tuple(points)


def freeze_metric_references(
    calibration_outputs: Sequence[Path],
    *,
    reference_margin: float = 0.10,
    forbid_algorithms: Iterable[str] = THEORY_ABLATION_ARMS,
) -> Dict[str, object]:
    if not calibration_outputs:
        raise ValueError("At least one calibration output is required.")
    forbidden = set(forbid_algorithms)
    fronts_by_case: Dict[str, List[Tuple[ObjectiveVector, ...]]] = {}
    algorithms: set[str] = set()
    source_files: List[Dict[str, str]] = []
    for output in calibration_outputs:
        aggregate = output / "aggregate_runs.csv"
        if not aggregate.is_file():
            raise FileNotFoundError(f"Missing calibration aggregate: {aggregate}")
        source_files.append(
            {
                "path": str(aggregate.resolve()),
                "sha256": _sha256_file(aggregate),
            }
        )
        with aggregate.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            case = str(row["case"])
            algorithm = str(row["algorithm"])
            algorithms.add(algorithm)
            archive = Path(str(row["archive_csv"]))
            if not archive.is_absolute():
                archive = output / archive
            if not archive.is_file():
                raise FileNotFoundError(f"Missing calibration archive: {archive}")
            front = _read_archive(archive)
            fronts_by_case.setdefault(case, []).append(front)
            source_files.append(
                {
                    "path": str(archive.resolve()),
                    "sha256": _sha256_file(archive),
                }
            )
    overlap = sorted(algorithms.intersection(forbidden))
    if overlap:
        raise ValueError(
            "Calibration output contains evaluated theory-ablation arms; "
            f"external-reference freeze refused: {overlap}"
        )
    if reference_margin <= 0.0:
        raise ValueError("reference_margin must be positive.")
    cases: Dict[str, Dict[str, object]] = {}
    for case, fronts in sorted(fronts_by_case.items()):
        reference_front = tuple(empirical_reference_front(fronts))
        all_points = [point for front in fronts for point in front]
        ideal, nadir = ideal_nadir(all_points)
        hv_reference = tuple(
            float(nadir[axis])
            + reference_margin * max(1e-9, float(nadir[axis]) - float(ideal[axis]))
            for axis in range(2)
        )
        case_payload = {
            "contract": "frozen_external_v1",
            "hypervolume_reference": list(hv_reference),
            "ideal": list(ideal),
            "nadir": list(nadir),
            "reference_front": [list(point) for point in reference_front],
        }
        canonical = json.dumps(
            case_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        case_payload["reference_sha256"] = hashlib.sha256(
            canonical
        ).hexdigest()
        cases[case] = case_payload
    return {
        "schema_version": 1,
        "contract": "frozen_external_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_only": True,
        "evaluated_theory_arms_forbidden": sorted(forbidden),
        "calibration_algorithms": sorted(algorithms),
        "reference_margin": reference_margin,
        "source_files": source_files,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze external per-case HV/IGD references from calibration-only suite outputs."
    )
    parser.add_argument(
        "--calibration-output",
        type=Path,
        action="append",
        required=True,
        help="Calibration suite output containing aggregate_runs.csv; repeat as needed.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-margin", type=float, default=0.10)
    args = parser.parse_args()
    payload = freeze_metric_references(
        args.calibration_output,
        reference_margin=args.reference_margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote frozen metric reference manifest to {args.output}")


if __name__ == "__main__":
    main()
