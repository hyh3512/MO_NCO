from __future__ import annotations

"""Build byte-bound case packets consumed by the IJOC cold-process adapter."""

import hashlib
import json
from pathlib import Path
from typing import Any


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = SUBMISSION_ROOT / "formal_study"
CASE_MANIFEST = FORMAL_ROOT / "case_manifest.json"
PACKET_ROOT = FORMAL_ROOT / "instances"
PACKET_MANIFEST = FORMAL_ROOT / "instance_packet_manifest.json"


def canonical_bytes(payload: object) -> bytes:
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


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload))
    return file_sha256(path)


def strict_json(path: Path) -> dict[str, Any]:
    def reject_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"Duplicate JSON key {key!r}: {path}")
            output[key] = value
        return output

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Non-finite constant {value!r}: {path}")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def main() -> None:
    manifest = strict_json(CASE_MANIFEST)
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Formal case manifest has no cases.")
    packet_bindings = []
    for case in cases:
        case_id = str(case["case_id"])
        family = str(case["family"])
        artifacts = case["artifacts"]
        expected_count = 2 if family == "MOTSP" else 1 if family == "MOKP" else 0
        if len(artifacts) != expected_count:
            raise ValueError(
                f"{case_id}: {family} requires {expected_count} source artifacts."
            )
        packet_path = PACKET_ROOT / f"{case_id}.packet.json"
        child_bindings = []
        for artifact in artifacts:
            source = (CASE_MANIFEST.parent / str(artifact["path"])).resolve()
            source.relative_to(PACKET_ROOT.resolve())
            actual_sha = file_sha256(source)
            if actual_sha != str(artifact["sha256"]):
                raise ValueError(f"{case_id}: source artifact SHA-256 mismatch.")
            child_bindings.append(
                {
                    "path": source.relative_to(PACKET_ROOT.resolve()).as_posix(),
                    "sha256": actual_sha,
                }
            )
        packet = {
            "schema": "ijoc_case_instance_packet_v1",
            "case_id": case_id,
            "family": family,
            "problem_sha256": str(case["problem_sha256"]),
            "artifacts": child_bindings,
        }
        packet_sha = write_json(packet_path, packet)
        packet_bindings.append(
            {
                "case_id": case_id,
                "family": family,
                "path": packet_path.relative_to(FORMAL_ROOT).as_posix(),
                "sha256": packet_sha,
                "child_artifact_sha256": [
                    item["sha256"] for item in child_bindings
                ],
            }
        )
    packet_manifest = {
        "schema": "ijoc_case_instance_packet_manifest_v1",
        "formal_case_manifest": {
            "path": CASE_MANIFEST.relative_to(FORMAL_ROOT).as_posix(),
            "sha256": file_sha256(CASE_MANIFEST),
        },
        "case_count": len(packet_bindings),
        "packets": sorted(packet_bindings, key=lambda item: item["case_id"]),
    }
    manifest_sha = write_json(PACKET_MANIFEST, packet_manifest)
    print(
        json.dumps(
            {
                "schema": "ijoc_case_instance_packet_build_result_v1",
                "case_count": len(packet_bindings),
                "packet_manifest": {
                    "path": PACKET_MANIFEST.relative_to(SUBMISSION_ROOT).as_posix(),
                    "sha256": manifest_sha,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
