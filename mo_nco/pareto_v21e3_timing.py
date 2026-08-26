from __future__ import annotations

"""Frozen timing semantics for the prospective V21e3 study."""

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

from .pareto_v21e3_artifacts import ArtifactRoot


_SCHEMA = "pareto_v21e3_timing_policy_v1"
_CLOCK = "time.monotonic_ns_process_local_elapsed_v1"
_ROLE = "DIAGNOSTIC_ONLY_NOT_IN_ALGORITHM_HASH_CHAIN"
_FIELDS = ("elapsed_monotonic_ns",)


@dataclass(frozen=True)
class V21E3TimingPolicy:
    sha256: str
    clock: str
    elapsed_fields: tuple[str, ...]
    semantic_role: str
    formal_quality_decision_uses_timing: bool
    resource_efficiency_claim_authorized: bool
    future_efficiency_protocol_required: bool


def load_timing_policy(
    artifact_root: ArtifactRoot,
    binding: Mapping[str, object],
) -> V21E3TimingPolicy:
    """Load the only admissible V21e3 timing policy by byte binding."""

    path = artifact_root.resolve_binding(binding)
    raw = path.read_bytes()
    payload = json.loads(raw)
    expected = {
        "schema": _SCHEMA,
        "clock": _CLOCK,
        "elapsed_fields": list(_FIELDS),
        "semantic_role": _ROLE,
        "formal_quality_decision_uses_timing": False,
        "resource_efficiency_claim_authorized": False,
        "future_efficiency_protocol_required": True,
    }
    if payload != expected:
        raise ValueError(
            "V21e3 supports only the frozen diagnostic-only timing policy; "
            "resource-efficiency claims require a new prospective protocol."
        )
    return V21E3TimingPolicy(
        sha256=hashlib.sha256(raw).hexdigest(),
        clock=_CLOCK,
        elapsed_fields=_FIELDS,
        semantic_role=_ROLE,
        formal_quality_decision_uses_timing=False,
        resource_efficiency_claim_authorized=False,
        future_efficiency_protocol_required=True,
    )


__all__ = ["V21E3TimingPolicy", "load_timing_policy"]
