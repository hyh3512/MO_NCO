from __future__ import annotations

"""Build disjoint, byte-bound IJOC calibration and formal case suites.

The calibration MOTSP cases are the existing synthetic V11 instances.  The
formal MOTSP cases are a frozen 15-case subset of the public 35-case suite.
MOKP cases are deterministic integer instances emitted as complete JSON
artifacts; calibration and formal seeds do not overlap.
"""

import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Any

from mo_nco.instance import MultiObjectiveTSPInstance, instance_sha256
from mo_nco.pareto_ijoc_problem import (
    MultiObjectiveKnapsackInstance,
    problem_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_ROOT = Path(__file__).resolve().parents[1]

FORMAL_MOTSP_CASES = (
    "public_bayg29_bays29",
    "public_dantzig42_swiss42",
    "public_att48_gr48",
    "public_att48_hk48",
    "public_eil76_pr76",
    "public_kroA100_kroB100",
    "public_kroA100_rd100",
    "public_ch150_kroA150",
    "public_ch150_kroB150",
    "public_kroA200_kroB200",
    "public_ts225_tsp225",
    "public_lin318_linhp318",
    "paquete_euclidAB300",
    "paquete_euclidEF300",
    "paquete_euclidAB500",
)


def canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload))
    return sha256(path)


def copy_bound_file(source: Path, target: Path) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return {
        "path": target.as_posix(),
        "sha256": sha256(target),
        "bytes": target.stat().st_size,
    }


def relative_artifact(artifact: dict[str, object], manifest_dir: Path) -> dict[str, object]:
    path = Path(str(artifact["path"])).resolve()
    return {
        **artifact,
        "path": path.relative_to(manifest_dir.resolve()).as_posix(),
    }


def load_suite(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError(f"Invalid suite: {path}")
    return payload


def build_mokp_payload(num_items: int, seed: int, case_id: str) -> dict[str, object]:
    rng = random.Random(seed)
    weights = [rng.randint(1, 30) for _ in range(num_items)]
    profits = [
        [rng.randint(1, 50) for _ in range(num_items)]
        for _ in range(2)
    ]
    return {
        "schema": "ijoc_mokp_integer_instance_v1",
        "case_id": case_id,
        "family": "MOKP",
        "num_items": num_items,
        "num_objectives": 2,
        "item_weights": weights,
        "profits_by_objective": profits,
        "capacity": max(1, int(0.35 * sum(weights))),
        "generator": {
            "name": "python_random_mt19937_integer_mokp_v1",
            "seed": seed,
            "weight_support": [1, 30],
            "profit_support": [1, 50],
            "capacity_fraction_floor": 0.35,
        },
    }


def copy_motsp_case(
    *,
    source_case: dict[str, Any],
    source_manifest_dir: Path,
    output_dir: Path,
    manifest_dir: Path,
    split: str,
    source_suite: str,
) -> dict[str, object]:
    case_id = str(source_case["name"])
    source_paths = [
        (source_manifest_dir / str(item)).resolve()
        for item in source_case["tsplib_files"]
    ]
    artifacts = []
    target_paths = []
    for index, source in enumerate(source_paths, start=1):
        suffix = source.suffix.lower() or ".tsp"
        target = output_dir / f"{case_id}-objective-{index}{suffix}"
        artifacts.append(copy_bound_file(source, target))
        target_paths.append(target)
    instance = MultiObjectiveTSPInstance.from_tsplib_files(target_paths)
    return {
        "case_id": case_id,
        "family": "MOTSP",
        "split": split,
        "size": instance.num_cities,
        "num_objectives": instance.num_objectives,
        "problem_sha256": instance_sha256(instance),
        "artifacts": [
            relative_artifact(item, manifest_dir) for item in artifacts
        ],
        "source_provenance": {
            "suite": source_suite,
            "source_case": case_id,
            "source_artifact_sha256": [sha256(path) for path in source_paths],
        },
    }


def emit_mokp_cases(
    *,
    output_dir: Path,
    manifest_dir: Path,
    split: str,
    sizes: tuple[int, ...],
    cases_per_size: int,
    seed_base: int,
) -> list[dict[str, object]]:
    cases = []
    for size_index, size in enumerate(sizes):
        for case_index in range(cases_per_size):
            seed = seed_base + size_index * 100 + case_index
            case_id = f"mokp-{split}-n{size}-s{case_index:02d}"
            payload = build_mokp_payload(size, seed, case_id)
            target = output_dir / f"{case_id}.json"
            digest = write_json(target, payload)
            problem = MultiObjectiveKnapsackInstance(
                item_weights=tuple(payload["item_weights"]),
                profits_by_objective=tuple(
                    tuple(row) for row in payload["profits_by_objective"]
                ),
                capacity=int(payload["capacity"]),
                name=case_id,
            )
            cases.append(
                {
                    "case_id": case_id,
                    "family": "MOKP",
                    "split": split,
                    "size": size,
                    "num_objectives": 2,
                    "problem_sha256": problem_sha256(problem),
                    "artifacts": [
                        {
                            "path": target.resolve()
                            .relative_to(manifest_dir.resolve())
                            .as_posix(),
                            "sha256": digest,
                            "bytes": target.stat().st_size,
                        }
                    ],
                    "source_provenance": {
                        "suite": "ijoc_integer_mokp_generated_v1",
                        "generator_seed": seed,
                    },
                }
            )
    return cases


def main() -> None:
    calibration_root = SUBMISSION_ROOT / "calibration"
    formal_root = SUBMISSION_ROOT / "formal_study"
    calibration_manifest_path = calibration_root / "case_manifest.json"
    formal_manifest_path = formal_root / "case_manifest.json"

    synthetic_suite_path = REPO_ROOT / "benchmarks" / "pareto_smc_v11_competitive_suite.json"
    synthetic_suite = load_suite(synthetic_suite_path)
    calibration_cases = []
    for source_case in synthetic_suite["cases"]:
        source_name = str(source_case["name"])
        case_suffix = int(source_name.rsplit("s", 1)[1])
        split = "selection" if case_suffix < 5 else "confirmation"
        calibration_cases.append(
            copy_motsp_case(
                source_case=source_case,
                source_manifest_dir=synthetic_suite_path.parent,
                output_dir=calibration_root / "instances" / "motsp",
                manifest_dir=calibration_manifest_path.parent,
                split=split,
                source_suite="pareto_smc_v11_predeclared_integer_biobjective",
            )
        )

    calibration_mokp = emit_mokp_cases(
        output_dir=calibration_root / "instances" / "mokp",
        manifest_dir=calibration_manifest_path.parent,
        split="selection",
        sizes=(100, 200, 500),
        cases_per_size=5,
        seed_base=2026073100,
    )
    calibration_mokp.extend(
        emit_mokp_cases(
            output_dir=calibration_root / "instances" / "mokp",
            manifest_dir=calibration_manifest_path.parent,
            split="confirmation",
            sizes=(100, 200, 500),
            cases_per_size=5,
            seed_base=2026074100,
        )
    )
    calibration_cases.extend(calibration_mokp)

    calibration_manifest = {
        "schema": "ijoc_case_suite_manifest_v1",
        "suite_id": "pareto_smc_v20_tail_calibration_disjoint_v1",
        "role": "algorithm_selection_only_not_competitive_evidence",
        "formal_overlap_count": 0,
        "split_contract": (
            "selection chooses one finite-menu candidate; confirmation applies "
            "the predeclared gate against uniform; neither split enters the "
            "formal matched matrix"
        ),
        "cases": sorted(
            calibration_cases,
            key=lambda item: (str(item["family"]), str(item["case_id"])),
        ),
    }
    calibration_sha = write_json(calibration_manifest_path, calibration_manifest)

    public_suite_path = REPO_ROOT / "benchmarks" / "suite_public_motsp_35.json"
    public_suite = load_suite(public_suite_path)
    public_by_name = {
        str(case["name"]): case for case in public_suite["cases"]
    }
    if set(FORMAL_MOTSP_CASES) - set(public_by_name):
        raise ValueError("The frozen formal MOTSP case list is not available.")
    formal_cases = [
        copy_motsp_case(
            source_case=public_by_name[name],
            source_manifest_dir=REPO_ROOT,
            output_dir=formal_root / "instances" / "motsp",
            manifest_dir=formal_manifest_path.parent,
            split="formal_test",
            source_suite="public_pair_motsp_suite_35_source_snapshot",
        )
        for name in FORMAL_MOTSP_CASES
    ]
    formal_cases.extend(
        emit_mokp_cases(
            output_dir=formal_root / "instances" / "mokp",
            manifest_dir=formal_manifest_path.parent,
            split="formal_test",
            sizes=(100, 200, 500),
            cases_per_size=5,
            seed_base=2026080100,
        )
    )
    formal_manifest = {
        "schema": "ijoc_case_suite_manifest_v1",
        "suite_id": "pareto_smc_v20_formal_motsp_mokp_30case_v1",
        "role": "formal_test_frozen_before_optimizer_runs",
        "calibration_overlap_count": 0,
        "scope_note": (
            "MOTSP uses frozen public TSPLIB/Paquete pairs. MOKP uses newly "
            "generated, fully released integer instances and is not claimed "
            "to be a legacy public benchmark."
        ),
        "cases": sorted(
            formal_cases,
            key=lambda item: (str(item["family"]), str(item["case_id"])),
        ),
    }
    formal_sha = write_json(formal_manifest_path, formal_manifest)

    summary = {
        "schema": "ijoc_case_suite_build_summary_v1",
        "calibration_manifest": {
            "path": calibration_manifest_path.relative_to(SUBMISSION_ROOT).as_posix(),
            "sha256": calibration_sha,
            "case_count": len(calibration_cases),
        },
        "formal_manifest": {
            "path": formal_manifest_path.relative_to(SUBMISSION_ROOT).as_posix(),
            "sha256": formal_sha,
            "case_count": len(formal_cases),
        },
        "calibration_formal_case_id_overlap": sorted(
            {str(item["case_id"]) for item in calibration_cases}
            & {str(item["case_id"]) for item in formal_cases}
        ),
    }
    write_json(SUBMISSION_ROOT / "artifacts" / "case_suite_build_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
