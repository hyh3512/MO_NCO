from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mo_nco.pareto_v21e3_artifacts import ArtifactRoot
from mo_nco.pareto_v21e3_timing import load_timing_policy


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def test_timing_policy_is_hash_bound_and_diagnostic_only(tmp_path: Path) -> None:
    payload = {
        "schema": "pareto_v21e3_timing_policy_v1",
        "clock": "time.monotonic_ns_process_local_elapsed_v1",
        "elapsed_fields": ["elapsed_monotonic_ns"],
        "semantic_role": "DIAGNOSTIC_ONLY_NOT_IN_ALGORITHM_HASH_CHAIN",
        "formal_quality_decision_uses_timing": False,
        "resource_efficiency_claim_authorized": False,
        "future_efficiency_protocol_required": True,
    }
    raw = _canonical(payload)
    path = tmp_path / "protocol" / "timing.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)

    policy = load_timing_policy(
        ArtifactRoot(tmp_path),
        {
            "path": "protocol/timing.json",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        },
    )

    assert policy.sha256 == hashlib.sha256(raw).hexdigest()
    assert policy.resource_efficiency_claim_authorized is False
    assert policy.clock == "time.monotonic_ns_process_local_elapsed_v1"
    assert policy.elapsed_fields == ("elapsed_monotonic_ns",)

