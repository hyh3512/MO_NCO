from __future__ import annotations

"""CLI for the independent IJOC metric-reference calibration."""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mo_nco.pareto_ijoc_reference import (
    build_reference_suite,
    verify_reference_suite,
)


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]


def _comma_separated_integers(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from error
    if not result:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-manifest",
        type=Path,
        default=SUBMISSION_ROOT / "formal_study" / "case_manifest.json",
    )
    parser.add_argument(
        "--tail-calibration-receipt",
        type=Path,
        default=(
            SUBMISSION_ROOT
            / "calibration"
            / "frozen"
            / "calibration_suite_receipt.json"
        ),
    )
    parser.add_argument(
        "--tail-policy",
        type=Path,
        default=(
            SUBMISSION_ROOT
            / "calibration"
            / "frozen"
            / "tail_policy_freeze.json"
        ),
    )
    parser.add_argument(
        "--instance-packet-manifest",
        type=Path,
        default=(
            SUBMISSION_ROOT
            / "formal_study"
            / "instance_packet_manifest.json"
        ),
    )
    parser.add_argument(
        "--replay-verifier",
        type=Path,
        default=SUBMISSION_ROOT / "scripts" / "ijoc_replay_verifier.py",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=SUBMISSION_ROOT / "formal_study" / "metric_references",
    )
    parser.add_argument(
        "--reference-seeds",
        type=_comma_separated_integers,
        default=(91000, 91001, 91002, 91003, 91004),
    )
    parser.add_argument(
        "--formal-seeds",
        type=_comma_separated_integers,
        default=(8100, 8101, 8102, 8103, 8104, 8105, 8106, 8107, 8108, 8109),
    )
    parser.add_argument(
        "--evaluation-budgets",
        type=_comma_separated_integers,
        default=(1000,),
    )
    parser.add_argument("--weight-grid-size", type=int, default=21)
    parser.add_argument("--restart-period", type=int, default=64)
    parser.add_argument("--case-limit-per-family", type=int)
    parser.add_argument("--time-limit-seconds", type=float)
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--rerun-cold-replay", action="store_true")
    parser.add_argument(
        "--artifact-source-root",
        type=Path,
        help=(
            "Directory against which completion run/case paths are written; "
            "defaults to the output directory's parent and must match the "
            "future freeze-request directory."
        ),
    )
    args = parser.parse_args()
    artifact_source_root = (
        args.artifact_source_root
        if args.artifact_source_root is not None
        else SUBMISSION_ROOT
    )

    if args.verify_only:
        evidence = verify_reference_suite(
            args.output_directory,
            instance_packet_manifest_path=args.instance_packet_manifest,
            replay_verifier_path=args.replay_verifier,
            rerun_cold_replay=args.rerun_cold_replay,
            artifact_source_root=artifact_source_root,
        )
    else:
        result = build_reference_suite(
            formal_case_manifest_path=args.case_manifest,
            instance_packet_manifest_path=args.instance_packet_manifest,
            tail_calibration_receipt_path=args.tail_calibration_receipt,
            tail_policy_path=args.tail_policy,
            replay_verifier_path=args.replay_verifier,
            output_directory=args.output_directory,
            reference_seeds=args.reference_seeds,
            formal_seeds=args.formal_seeds,
            evaluation_budgets=args.evaluation_budgets,
            weight_grid_size=args.weight_grid_size,
            restart_period=args.restart_period,
            case_limit_per_family=args.case_limit_per_family,
            time_limit_seconds=args.time_limit_seconds,
            replace_existing=args.replace_existing,
            artifact_source_root=artifact_source_root,
        )
        evidence = {
            "status": result.status,
            "output_directory": result.output_directory.as_posix(),
            "case_count": result.case_count,
            "family_counts": dict(result.family_counts),
            "elapsed_seconds": result.elapsed_seconds,
            "reused_existing": result.reused_existing,
            "verification": verify_reference_suite(
                result.output_directory,
                instance_packet_manifest_path=args.instance_packet_manifest,
                replay_verifier_path=args.replay_verifier,
                rerun_cold_replay=args.rerun_cold_replay,
                artifact_source_root=artifact_source_root,
            ),
        }
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
