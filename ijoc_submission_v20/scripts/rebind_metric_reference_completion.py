from __future__ import annotations

"""Rebase completed reference-calibration paths without rerunning the search.

Only the completion envelope, derived audit, and relative path strings change.
The precommit, per-run witnesses, replay receipts, and per-case references stay
byte-identical.  The previous output directory is preserved under a
content-addressed ``.superseded-*`` name.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mo_nco.pareto_ijoc_reference import (
    file_sha256,
    strict_json,
    verify_reference_suite,
    write_json,
)


def _rebase_binding(
    binding: dict[str, Any],
    *,
    old_source_root: Path,
    output: Path,
    staging: Path,
    new_source_root: Path,
) -> dict[str, object]:
    raw_path = Path(str(binding["path"]))
    if raw_path.is_absolute():
        raise ValueError("Completion artifact paths must be relative.")
    old_path = (old_source_root / raw_path).resolve()
    try:
        relative = old_path.relative_to(output.resolve())
    except ValueError as error:
        raise ValueError("Completion artifact is outside the reference output.") from error
    if (
        not old_path.is_file()
        or file_sha256(old_path) != binding.get("sha256")
        or old_path.stat().st_size != binding.get("bytes")
    ):
        raise ValueError("Completion artifact binding mismatch before rebase.")
    staged_path = staging / relative
    if (
        not staged_path.is_file()
        or file_sha256(staged_path) != binding.get("sha256")
    ):
        raise ValueError("Copied completion artifact changed during rebase.")
    final_path = (output / relative).resolve()
    try:
        new_relative = final_path.relative_to(new_source_root.resolve())
    except ValueError as error:
        raise ValueError("Reference output is outside the new source root.") from error
    return {
        **binding,
        "path": new_relative.as_posix(),
    }


def _archive_previous_output(output: Path) -> Path:
    evidence = output / "reference_calibration_completion_evidence.json"
    suffix = file_sha256(evidence)[:12]
    archived = output.with_name(f"{output.name}.superseded-{suffix}")
    if archived.exists():
        raise FileExistsError(f"Preserved previous output already exists: {archived}")
    os.replace(output, archived)
    return archived


def rebind_completion(
    *,
    output_directory: Path,
    old_source_root: Path,
    new_source_root: Path,
    tail_calibration_receipt_path: Path,
    tail_policy_path: Path,
    instance_packet_manifest_path: Path,
    replay_verifier_path: Path,
) -> dict[str, object]:
    output = output_directory.resolve()
    old_root = old_source_root.resolve()
    new_root = new_source_root.resolve()
    verify_reference_suite(
        output,
        instance_packet_manifest_path=instance_packet_manifest_path,
        replay_verifier_path=replay_verifier_path,
        artifact_source_root=old_root,
    )
    tail_receipt = strict_json(tail_calibration_receipt_path)
    tail_receipt_sha = file_sha256(tail_calibration_receipt_path)
    tail_policy = strict_json(tail_policy_path)
    if (
        tail_receipt.get("schema") != "ijoc_calibration_suite_receipt_v1"
        or tail_receipt.get("status") != "COMPLETE"
        or tail_policy.get("schema") != "ijoc_tail_policy_freeze_v1"
        or tail_policy.get("status") != "FROZEN"
        or tail_policy.get("calibration_suite_sha256") != tail_receipt_sha
    ):
        raise ValueError("Tail calibration/tail policy chain is not frozen.")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.rebase-", dir=output.parent)
    )
    committed = False
    archived: Path | None = None
    try:
        shutil.rmtree(staging)
        shutil.copytree(output, staging)
        evidence_path = staging / "reference_calibration_completion_evidence.json"
        receipt_path = staging / "reference_calibration_completion_receipt.json"
        audit_path = staging / "reference_calibration_audit.json"
        evidence = dict(strict_json(evidence_path))
        receipt = dict(strict_json(receipt_path))
        audit = dict(strict_json(audit_path))

        runs = []
        for raw_run in receipt["reference_runs"]:
            run = dict(raw_run)
            run["source_artifacts"] = [
                _rebase_binding(
                    dict(binding),
                    old_source_root=old_root,
                    output=output,
                    staging=staging,
                    new_source_root=new_root,
                )
                for binding in raw_run["source_artifacts"]
            ]
            runs.append(run)
        outputs = [
            _rebase_binding(
                dict(binding),
                old_source_root=old_root,
                output=output,
                staging=staging,
                new_source_root=new_root,
            )
            for binding in receipt["case_outputs"]
        ]

        bindings = dict(audit["bindings"])
        bindings["tail_policy_selection_receipt_sha256"] = tail_receipt_sha
        bindings["tail_policy_sha256"] = file_sha256(tail_policy_path)
        audit["bindings"] = bindings
        audit["completion_path_contract"] = (
            "all reference run and case-output paths are relative to the "
            "freeze-request parent (the IJOC submission root)"
        )
        write_json(audit_path, audit)

        evidence["reference_runs"] = runs
        evidence["case_outputs"] = outputs
        evidence["audit_artifact"] = {
            "path": "reference_calibration_audit.json",
            "sha256": file_sha256(audit_path),
            "bytes": audit_path.stat().st_size,
        }
        evidence_sha = write_json(evidence_path, evidence)
        receipt["reference_runs"] = runs
        receipt["case_outputs"] = outputs
        receipt["artifact_manifest"] = {
            "path": "reference_calibration_completion_evidence.json",
            "sha256": evidence_sha,
        }
        write_json(receipt_path, receipt)

        archived = _archive_previous_output(output)
        os.replace(staging, output)
        committed = True
        verification = verify_reference_suite(
            output,
            instance_packet_manifest_path=instance_packet_manifest_path,
            replay_verifier_path=replay_verifier_path,
            artifact_source_root=new_root,
            rerun_cold_replay=True,
        )
        return {
            "status": "PASS",
            "output_directory": output.as_posix(),
            "preserved_previous_output": archived.as_posix(),
            "tail_calibration_receipt_sha256": tail_receipt_sha,
            "tail_policy_sha256": file_sha256(tail_policy_path),
            "verification": verification,
        }
    except Exception:
        if archived is not None and archived.exists() and not output.exists():
            os.replace(archived, output)
        raise
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=SUBMISSION_ROOT / "formal_study" / "metric_references",
    )
    parser.add_argument(
        "--old-source-root",
        type=Path,
        default=SUBMISSION_ROOT / "formal_study",
    )
    parser.add_argument(
        "--new-source-root",
        type=Path,
        default=SUBMISSION_ROOT,
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
        default=SUBMISSION_ROOT / "formal_study" / "instance_packet_manifest.json",
    )
    parser.add_argument(
        "--replay-verifier",
        type=Path,
        default=SUBMISSION_ROOT / "scripts" / "ijoc_replay_verifier.py",
    )
    args = parser.parse_args()
    result = rebind_completion(
        output_directory=args.output_directory,
        old_source_root=args.old_source_root,
        new_source_root=args.new_source_root,
        tail_calibration_receipt_path=args.tail_calibration_receipt,
        tail_policy_path=args.tail_policy,
        instance_packet_manifest_path=args.instance_packet_manifest,
        replay_verifier_path=args.replay_verifier,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
